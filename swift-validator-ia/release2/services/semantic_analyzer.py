"""
semantic_analyzer.py
====================
Analyse sémantique avancée — enrichissement et correction de l'extraction.

Architecture en 2 modes :

  Mode A — Analyse par blocs layout (si pdf_blocks fourni)
  ──────────────────────────────────────────────────────────
    Pour chaque bloc PyMuPDF (text + bbox + dimensions page) :
      1. Classifier le bloc : HEADER / PARTY_LEFT / PARTY_RIGHT /
                              TRANSPORT / CARGO / TOTALS / FOOTER / BODY
      2. Dans le bloc, extraire les couples label→valeur
      3. N'autoriser que les champs cohérents avec le rôle du bloc
      4. Scorer la confiance (exact label + type entité + rôle)
    Sélectionner le meilleur candidat par champ.
    À haute confiance (≥ 0.85), peut remplacer les tiers faibles
    (party_block, kv_fallback, semantic).

  Mode B — Analyse sur texte plat (fallback si pas de blocs)
  ─────────────────────────────────────────────────────────────
    Annote chaque ligne : rôle (section/label/aggregate/data) + type entité.
    Extrait les paires label→valeur par 3 stratégies (inline colon, pipe, bloc 2 lignes).

  Passe commune — Inférence domaine
    carrier ← préfixe BL / navire
    place_of_receipt ← port_of_loading
    place_of_delivery ← port_of_discharge
    currency ← code dans total_amount

Intégration :
  pipeline._process_one() appelle _run_semantic_analysis(text, doc_type, fields,
    pdf_blocks=blocks, existing_tiers=tiers)
  Les champs retournés sont ajoutés (ou remplacent les tiers faibles).
"""

from __future__ import annotations

import re
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Rôles de lignes textuelles (niveau micro)
# ─────────────────────────────────────────────────────────────────────────────

ROLE_SECTION   = "section"
ROLE_LABEL     = "label"
ROLE_AGGREGATE = "aggregate"
ROLE_DATA      = "data"

# ─────────────────────────────────────────────────────────────────────────────
# Rôles de blocs layout (niveau bloc PDF)
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_HEADER      = "HEADER"
BLOCK_PARTY_LEFT  = "PARTY_LEFT"
BLOCK_PARTY_RIGHT = "PARTY_RIGHT"
BLOCK_TRANSPORT   = "TRANSPORT"
BLOCK_CARGO       = "CARGO"
BLOCK_TOTALS      = "TOTALS"
BLOCK_FOOTER      = "FOOTER"
BLOCK_BODY        = "BODY"

# ─────────────────────────────────────────────────────────────────────────────
# Mots-clés de sections — jamais capturés comme valeurs
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_KEYWORDS: set[str] = {
    "PARTIES", "CARGO", "FREIGHT", "TOTALS", "HEADER",
    "CARGO DETAILS", "SHIPPING DETAILS", "SHIPMENT DETAILS",
    "DESCRIPTION OF GOODS", "TERMS AND CONDITIONS",
    "ROUTING DETAILS", "MARKS AND NUMBERS", "MARKS & NOS",
    "CONSIGNMENT DETAILS", "ORIGIN AND DESTINATION",
}

# ─────────────────────────────────────────────────────────────────────────────
# Valeurs connues tout en majuscules qui NE sont PAS des labels
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_VALUES: set[str] = {
    "AT SIGHT", "UPON SIGHT", "SIGHT DRAFT", "USANCE",
    "CASH ADVANCE", "OPEN ACCOUNT", "DEFERRED PAYMENT",
    "USANCE DRAFT", "LETTER OF CREDIT", "DOCUMENTS AGAINST PAYMENT",
    "DOCUMENTS AGAINST ACCEPTANCE", "30 DAYS", "60 DAYS", "90 DAYS",
    "FREIGHT PREPAID", "FREIGHT COLLECT", "PREPAID", "COLLECT",
    "AS ARRANGED", "TO COLLECT",
    "FOB", "CIF", "CFR", "EXW", "DDP", "FCA", "CPT", "CIP",
    "DAP", "DAT", "DPU", "FAS", "DDU",
    "CY/CY", "CY/CFS", "CFS/CY", "CFS/CFS",
    "FCL/FCL", "FCL/LCL", "LCL/FCL", "LCL/LCL",
    "FCL", "LCL", "RORO", "BREAKBULK",
    "ORIGINAL", "COPY", "NON-NEGOTIABLE", "NEGOTIABLE",
    "TELEX RELEASE", "EXPRESS BILL",
    "YES", "NO", "N/A", "NOT APPLICABLE", "AS PER CONTRACT",
    "ON BOARD", "SHIPPED ON BOARD",
    "THREE (3)", "TWO (2)", "ONE (1)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Préfixes d'agrégats — jamais des lignes de données
# ─────────────────────────────────────────────────────────────────────────────

_AGGREGATE_PREFIXES: tuple[str, ...] = (
    "TOTAL", "GRAND TOTAL", "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL",
    "NET TOTAL", "TOTAL NET", "TOTAL GROSS",
    "TOTAL PACKAGES", "TOTAL PKGS", "TOTAL WEIGHT",
    "TOTAL CBM", "TOTAL VOLUME", "TOTAL AMOUNT",
    "AMOUNT DUE", "BALANCE DUE",
)

# ─────────────────────────────────────────────────────────────────────────────
# Types d'entités sémantiques
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_COMPANY   = "company"
ENTITY_PORT_CITY = "port_city"
ENTITY_DATE      = "date"
ENTITY_AMOUNT    = "amount"
ENTITY_CONTAINER = "container"
ENTITY_REFERENCE = "reference"
ENTITY_CODE      = "code"
ENTITY_UNKNOWN   = "unknown"

_COMPANY_SUFFIXES: tuple[str, ...] = (
    r"\bLTD\b", r"\bLIMITED\b", r"\bGMBH\b", r"\bINC\b",
    r"\bLLC\b", r"\bSA\b", r"\bAG\b", r"\bSARL\b",
    r"\bCORP\b", r"\bCORPORATION\b",
    r"\bBV\b", r"\bNV\b", r"\bPLC\b", r"\bSRL\b",
    r"\bSPRL\b", r"\bPTY\b", r"\bKG\b", r"\bSAS\b",
    r"\bSPA\b", r"\bOY\b", r"\bAB\b",
)

_COMPANY_WORDS: set[str] = {
    "INTERNATIONAL", "GLOBAL", "TRADING", "COMMERCE", "SHIPPING",
    "LOGISTICS", "IMPORT", "EXPORT", "INDUSTRIES", "ENTERPRISES",
    "GROUP", "PARTNERS", "SERVICES", "SOLUTIONS", "SUPPLY", "CHAIN",
    "NETWORK", "FREIGHT", "MARITIME",
}

_CURRENCY_CODES: set[str] = {
    "USD", "EUR", "GBP", "CNY", "JPY", "CHF", "CAD", "AUD",
    "SGD", "HKD", "KWD", "AED", "SAR", "QAR", "TRY", "INR",
}

# ─────────────────────────────────────────────────────────────────────────────
# Contrainte type-entité par champ
# ─────────────────────────────────────────────────────────────────────────────

_FIELD_ENTITY_CONSTRAINT: dict[str, set[str]] = {
    "shipper":           {ENTITY_COMPANY, ENTITY_UNKNOWN},
    "consignee":         {ENTITY_COMPANY, ENTITY_UNKNOWN},
    "seller":            {ENTITY_COMPANY, ENTITY_UNKNOWN},
    "buyer":             {ENTITY_COMPANY, ENTITY_UNKNOWN},
    "notify_party":      {ENTITY_COMPANY, ENTITY_UNKNOWN},
    "carrier":           {ENTITY_COMPANY, ENTITY_CODE, ENTITY_UNKNOWN},
    "vessel":            {ENTITY_UNKNOWN, ENTITY_CODE},
    "port_of_loading":   {ENTITY_PORT_CITY, ENTITY_UNKNOWN},
    "port_of_discharge": {ENTITY_PORT_CITY, ENTITY_UNKNOWN},
    "place_of_receipt":  {ENTITY_PORT_CITY},        # strict : jamais UNKNOWN
    "place_of_delivery": {ENTITY_PORT_CITY},        # strict : jamais UNKNOWN
    "issue_date":        {ENTITY_DATE},
    "invoice_date":      {ENTITY_DATE},
    "pl_date":           {ENTITY_DATE},
    "etd":               {ENTITY_DATE},
    "eta":               {ENTITY_DATE},
    "total_amount":      {ENTITY_AMOUNT},
    "subtotal":          {ENTITY_AMOUNT},
    "container_number":  {ENTITY_CONTAINER},
    "bl_number":         {ENTITY_REFERENCE},
    "invoice_number":    {ENTITY_REFERENCE},
    "pl_number":         {ENTITY_REFERENCE},
    "bl_reference":      {ENTITY_REFERENCE},
}

# ─────────────────────────────────────────────────────────────────────────────
# Portée des champs par rôle de bloc layout
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK_FIELD_SCOPE: dict[str, dict[str, set[str]]] = {
    "bl": {
        BLOCK_HEADER: {"bl_number", "issue_date", "carrier"},
        BLOCK_PARTY_LEFT: {"shipper"},
        BLOCK_PARTY_RIGHT: {"consignee", "notify_party"},
        BLOCK_TRANSPORT: {
            "vessel", "voyage",
            "port_of_loading", "port_of_discharge",
            "place_of_receipt", "place_of_delivery",
            "etd", "eta",
        },
        BLOCK_CARGO: {
            "container_number", "container_type", "seal_number",
            "goods_description", "gross_weight_kg", "measurement",
            "number_of_packages",
        },
        BLOCK_TOTALS: {"freight_terms", "incoterm", "service_type"},
        BLOCK_FOOTER: {"issue_date", "issue_place", "number_of_originals"},
    },
    "invoice": {
        BLOCK_HEADER: {"invoice_number", "invoice_date", "currency", "bl_reference"},
        BLOCK_PARTY_LEFT: {"seller"},
        BLOCK_PARTY_RIGHT: {"buyer"},
        BLOCK_CARGO: {"goods_description", "hs_code"},
        BLOCK_TOTALS: {
            "total_amount", "subtotal", "incoterm",
            "payment_terms", "currency",
        },
        BLOCK_FOOTER: {"invoice_date", "payment_terms"},
    },
    "packing_list": {
        BLOCK_HEADER: {"pl_number", "pl_date"},
        BLOCK_PARTY_LEFT: {"shipper"},
        BLOCK_PARTY_RIGHT: {"consignee"},
        BLOCK_CARGO: {"goods_description", "container_number"},
        BLOCK_TOTALS: {
            "total_packages", "total_cbm",
            "gross_weight_kg", "net_weight_kg",
        },
    },
}

# Tiers qu'une extraction sémantique à haute confiance (≥ 0.85) peut remplacer
_OVERRIDABLE_TIERS: set[str] = {"party_block", "kv_fallback", "semantic"}

# ─────────────────────────────────────────────────────────────────────────────
# Table label → champ, par type de document
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_FIELD_MAP: dict[str, dict[str, str]] = {
    "bl": {
        "BILL OF LADING NO":       "bl_number",
        "BILL OF LADING NO.":      "bl_number",
        "BILL OF LADING NUMBER":   "bl_number",
        "B/L NO":                  "bl_number",
        "B/L NO.":                 "bl_number",
        "B/L NUMBER":              "bl_number",
        "BL NO":                   "bl_number",
        "BL NO.":                  "bl_number",
        "BL NUMBER":               "bl_number",
        "BL REF":                  "bl_number",
        "B/L REF":                 "bl_number",
        "LADING NUMBER":           "bl_number",
        "DATE OF ISSUE":           "issue_date",
        "ISSUE DATE":              "issue_date",
        "DATE OF LADING":          "issue_date",
        "DATE ISSUED":             "issue_date",
        "ISSUED DATE":             "issue_date",
        "B/L DATE":                "issue_date",
        "BILL OF LADING DATE":     "issue_date",
        "DATE OF B/L":             "issue_date",
        "SHIPPER":                 "shipper",
        "SHIPPER / EXPORTER":      "shipper",
        "SHIPPER/EXPORTER":        "shipper",
        "EXPORTER":                "shipper",
        "SENDER":                  "shipper",
        "CONSIGNEE":               "consignee",
        "TO ORDER":                "consignee",
        "TO THE ORDER OF":         "consignee",
        "ORDER OF":                "consignee",
        "NOTIFY PARTY":            "notify_party",
        "ALSO NOTIFY":             "notify_party",
        "NOTIFY":                  "notify_party",
        "NOTIFY ADDRESS":          "notify_party",
        "CARRIER":                 "carrier",
        "SHIPPING LINE":           "carrier",
        "OCEAN CARRIER":           "carrier",
        "VESSEL":                  "vessel",
        "VESSEL NAME":             "vessel",
        "OCEAN VESSEL":            "vessel",
        "SHIP NAME":               "vessel",
        "MOTHER VESSEL":           "vessel",
        "VOYAGE":                  "voyage",
        "VOYAGE NO":               "voyage",
        "VOYAGE NO.":              "voyage",
        "VOYAGE NUMBER":           "voyage",
        "VOY NO":                  "voyage",
        "VOY NO.":                 "voyage",
        "VOY.":                    "voyage",
        "VOY":                     "voyage",
        "PORT OF LOADING":         "port_of_loading",
        "LOADING PORT":            "port_of_loading",
        "PORT OF LOAD":            "port_of_loading",
        "LOAD PORT":               "port_of_loading",
        "POL":                     "port_of_loading",
        "PORT OF DEPARTURE":       "port_of_loading",
        "PORT OF DISCHARGE":       "port_of_discharge",
        "DISCHARGE PORT":          "port_of_discharge",
        "PORT OF DEST":            "port_of_discharge",
        "DESTINATION PORT":        "port_of_discharge",
        "POD":                     "port_of_discharge",
        "PLACE OF RECEIPT":        "place_of_receipt",
        "PLACE OF ORIGIN":         "place_of_receipt",
        "PRE-CARRIAGE FROM":       "place_of_receipt",
        "ORIGIN":                  "place_of_receipt",
        "PLACE OF DELIVERY":       "place_of_delivery",
        "FINAL DESTINATION":       "place_of_delivery",
        "DELIVERY PLACE":          "place_of_delivery",
        "ETD":                     "etd",
        "ETA":                     "eta",
        "DATE OF DEPARTURE":       "etd",
        "DEPARTURE DATE":          "etd",
        "DATE OF ARRIVAL":         "eta",
        "ARRIVAL DATE":            "eta",
        "CONTAINER NO":            "container_number",
        "CONTAINER NO.":           "container_number",
        "CONTAINER NUMBER":        "container_number",
        "CNTR NO":                 "container_number",
        "CNTR NO.":                "container_number",
        "CONT. NO":                "container_number",
        "SEAL NO":                 "seal_number",
        "SEAL NO.":                "seal_number",
        "SEAL NUMBER":             "seal_number",
        "SEAL":                    "seal_number",
        "CONTAINER TYPE":          "container_type",
        "CNTR TYPE":               "container_type",
        "SIZE/TYPE":               "container_type",
        "GROSS WEIGHT":            "gross_weight_kg",
        "TOTAL GROSS WEIGHT":      "gross_weight_kg",
        "GROSS WT":                "gross_weight_kg",
        "GROSS WT.":               "gross_weight_kg",
        "G/W":                     "gross_weight_kg",
        "NET WEIGHT":              "net_weight_kg",
        "TOTAL NET WEIGHT":        "net_weight_kg",
        "NET WT":                  "net_weight_kg",
        "NET WT.":                 "net_weight_kg",
        "N/W":                     "net_weight_kg",
        "MEASUREMENT":             "measurement",
        "CBM":                     "measurement",
        "TOTAL CBM":               "measurement",
        "CUBIC METRES":            "measurement",
        "VOLUME":                  "measurement",
        "DESCRIPTION OF GOODS":    "goods_description",
        "CARGO DESCRIPTION":       "goods_description",
        "NATURE OF GOODS":         "goods_description",
        "FREIGHT TERMS":           "freight_terms",
        "FREIGHT":                 "freight_terms",
        "FREIGHT PAYABLE":         "freight_terms",
        "PAYMENT OF FREIGHT":      "freight_terms",
        "INCOTERM":                "incoterm",
        "INCOTERMS":               "incoterm",
        "DELIVERY TERMS":          "incoterm",
        "NUMBER OF ORIGINALS":     "number_of_originals",
        "NO. OF ORIGINALS":        "number_of_originals",
        "ORIGINAL B/L":            "number_of_originals",
        "ORIGINALS":               "number_of_originals",
    },

    "invoice": {
        "INVOICE NO":              "invoice_number",
        "INVOICE NO.":             "invoice_number",
        "INVOICE NUMBER":          "invoice_number",
        "INV NO":                  "invoice_number",
        "INV NO.":                 "invoice_number",
        "INV NUMBER":              "invoice_number",
        "INV #":                   "invoice_number",
        "TAX INVOICE NO":          "invoice_number",
        "COMMERCIAL INVOICE NO":   "invoice_number",
        "PROFORMA INVOICE NO":     "invoice_number",
        "INVOICE REF":             "invoice_number",
        "INVOICE REFERENCE":       "invoice_number",
        "INVOICE DATE":            "invoice_date",
        "DATE OF INVOICE":         "invoice_date",
        "DATE":                    "invoice_date",
        "INV DATE":                "invoice_date",
        "ISSUE DATE":              "invoice_date",
        "ISSUED DATE":             "invoice_date",
        "SELLER":                  "seller",
        "SELLER / EXPORTER":       "seller",
        "SELLER/EXPORTER":         "seller",
        "EXPORTER":                "seller",
        "VENDOR":                  "seller",
        "SUPPLIER":                "seller",
        "SOLD BY":                 "seller",
        "FROM":                    "seller",
        "BUYER":                   "buyer",
        "BUYER / IMPORTER":        "buyer",
        "BUYER/IMPORTER":          "buyer",
        "IMPORTER":                "buyer",
        "CONSIGNEE":               "buyer",
        "BILL TO":                 "buyer",
        "SHIP TO":                 "buyer",
        "SOLD TO":                 "buyer",
        "CUSTOMER":                "buyer",
        "TOTAL AMOUNT":            "total_amount",
        "GRAND TOTAL":             "total_amount",
        "TOTAL":                   "total_amount",
        "INVOICE TOTAL":           "total_amount",
        "TOTAL INVOICE AMOUNT":    "total_amount",
        "TOTAL DUE":               "total_amount",
        "AMOUNT DUE":              "total_amount",
        "NET TOTAL":               "total_amount",
        "SUBTOTAL":                "subtotal",
        "SUB TOTAL":               "subtotal",
        "SUB-TOTAL":               "subtotal",
        "NET AMOUNT":              "subtotal",
        "CURRENCY":                "currency",
        "CCY":                     "currency",
        "PAYMENT TERMS":           "payment_terms",
        "TERMS OF PAYMENT":        "payment_terms",
        "PAYMENT":                 "payment_terms",
        "PAYMENT CONDITIONS":      "payment_terms",
        "INCOTERM":                "incoterm",
        "INCOTERMS":               "incoterm",
        "DELIVERY TERMS":          "incoterm",
        "TERMS":                   "incoterm",
        "B/L REFERENCE":           "bl_reference",
        "B/L REF":                 "bl_reference",
        "BL REF":                  "bl_reference",
        "BL REFERENCE":            "bl_reference",
        "LC REF":                  "bl_reference",
        "LC REFERENCE":            "bl_reference",
        "L/C REF":                 "bl_reference",
        "L/C NO":                  "bl_reference",
        "DESCRIPTION OF GOODS":    "goods_description",
        "DESCRIPTION":             "goods_description",
        "COMMODITY":               "goods_description",
        "HS CODE":                 "hs_code",
        "HS NO":                   "hs_code",
        "TARIFF CODE":             "hs_code",
        "HTS CODE":                "hs_code",
        "HS":                      "hs_code",
    },

    "packing_list": {
        "PACKING LIST NO":         "pl_number",
        "PACKING LIST NO.":        "pl_number",
        "PACKING LIST NUMBER":     "pl_number",
        "PL NO":                   "pl_number",
        "PL NO.":                  "pl_number",
        "PL NUMBER":               "pl_number",
        "PACKING NO":              "pl_number",
        "PACKING LIST DATE":       "pl_date",
        "DATE":                    "pl_date",
        "PL DATE":                 "pl_date",
        "PACKING DATE":            "pl_date",
        "SHIPPER":                 "shipper",
        "SHIPPER / EXPORTER":      "shipper",
        "SHIPPER/EXPORTER":        "shipper",
        "EXPORTER":                "shipper",
        "CONSIGNEE":               "consignee",
        "BUYER":                   "consignee",
        "CONTAINER NO":            "container_number",
        "CONTAINER NO.":           "container_number",
        "CONTAINER NUMBER":        "container_number",
        "CNTR NO":                 "container_number",
        "CNTR NO.":                "container_number",
        "SEAL NO":                 "seal_number",
        "SEAL NO.":                "seal_number",
        "SEAL NUMBER":             "seal_number",
        "CONTAINER TYPE":          "container_type",
        "CNTR TYPE":               "container_type",
        "SIZE/TYPE":               "container_type",
        "GROSS WEIGHT":            "gross_weight_kg",
        "TOTAL GROSS WEIGHT":      "gross_weight_kg",
        "TOTAL GROSS WT":          "gross_weight_kg",
        "GROSS WT":                "gross_weight_kg",
        "GROSS WT.":               "gross_weight_kg",
        "G/W":                     "gross_weight_kg",
        "NET WEIGHT":              "net_weight_kg",
        "TOTAL NET WEIGHT":        "net_weight_kg",
        "TOTAL NET WT":            "net_weight_kg",
        "NET WT":                  "net_weight_kg",
        "NET WT.":                 "net_weight_kg",
        "N/W":                     "net_weight_kg",
        "TOTAL CBM":               "total_cbm",
        "CBM":                     "total_cbm",
        "TOTAL VOLUME":            "total_cbm",
        "VOLUME":                  "total_cbm",
        "TOTAL PACKAGES":          "total_packages",
        "TOTAL PKGS":              "total_packages",
        "TOTAL CARTONS":           "total_packages",
        "TOTAL PIECES":            "total_packages",
        "TOTAL QTY":               "total_packages",
        "TOTAL BOXES":             "total_packages",
        "B/L REFERENCE":           "bl_reference",
        "B/L REF":                 "bl_reference",
        "BL REF":                  "bl_reference",
        "BL REFERENCE":            "bl_reference",
        "INVOICE REF":             "bl_reference",
        "INVOICE REFERENCE":       "bl_reference",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Préfixes BL / navire → transporteur
# ─────────────────────────────────────────────────────────────────────────────

_BL_CARRIER_PREFIXES: list[tuple[str, str]] = [
    ("MSCUN", "MSC"), ("MSCU", "MSC"),
    ("MAEUZ", "Maersk Line"), ("MAEU", "Maersk Line"),
    ("CMAUJ", "CMA CGM"), ("CMAU", "CMA CGM"),
    ("HLCUK", "Hapag-Lloyd"), ("HLCU", "Hapag-Lloyd"),
    ("ONEY", "ONE"),
    ("EVGU", "Evergreen"), ("EGLV", "Evergreen"),
    ("COSU", "COSCO"),
    ("ZIMU", "ZIM"),
    ("APLU", "APL"),
    ("YMLU", "Yang Ming"),
    ("OOLU", "OOCL"),
]

_VESSEL_CARRIER_PREFIXES: list[tuple[str, str]] = [
    ("MSC ", "MSC"),
    ("MAERSK ", "Maersk Line"),
    ("CMA CGM ", "CMA CGM"),
    ("HAPAG ", "Hapag-Lloyd"),
    ("EVER ", "Evergreen"),
    ("COSCO ", "COSCO"),
    ("ONE ", "ONE"),
    ("ZIM ", "ZIM"),
    ("APL ", "APL"),
    ("OOCL ", "OOCL"),
]

# Valeurs rejetées pour les champs de lieu (expressions non-géographiques)
_PORT_FIELD_BLACKLIST: set[str] = {
    "ON BOARD DATE", "ON BOARD", "BOARD DATE",
    "FREIGHT PREPAID", "FREIGHT COLLECT", "PREPAID", "COLLECT",
    "AS ARRANGED", "TO BE ADVISED", "TBA", "TBD",
    "SAME AS POD", "SAME AS POL",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    text:        str
    role:        str = ROLE_DATA
    entity_type: str = ENTITY_UNKNOWN
    line_index:  int = 0


@dataclass
class LayoutBlock:
    """Bloc PyMuPDF avec position spatiale et rôle sémantique."""
    text:         str
    bbox:         tuple[float, float, float, float]  # x0, y0, x1, y1
    page_width:   float = 595.0
    page_height:  float = 842.0
    block_role:   str   = BLOCK_BODY


# ─────────────────────────────────────────────────────────────────────────────
# Analyseur principal
# ─────────────────────────────────────────────────────────────────────────────

class SemanticAnalyzer:
    """
    Enrichissement sémantique de l'extraction documentaire.

    Usage :
        analyzer = SemanticAnalyzer()
        # Avec blocs layout (précis) :
        new_fields = analyzer.analyze(text, "bl", existing_fields,
                                      pdf_blocks=blocks, existing_tiers=tiers)
        # Fallback texte plat :
        new_fields = analyzer.analyze(text, "bl", existing_fields)
    """

    def analyze(
        self,
        text: str,
        doc_type: str,
        existing_fields: dict[str, str],
        pdf_blocks: list[dict] | None = None,
        existing_tiers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Retourne les champs à ajouter OU à remplacer.

        - Champs manquants  : retournés si confiance ≥ 0.60
        - Champs existants avec tier remplaçable : retournés si confiance ≥ 0.85
        """
        if (not text and not pdf_blocks) or not doc_type:
            return {}

        tiers = existing_tiers or {}

        if pdf_blocks:
            new_fields = self._analyze_blocks(pdf_blocks, doc_type, existing_fields, tiers)
        else:
            new_fields = self._analyze_flat(text, doc_type, existing_fields)

        # Inférence domaine
        combined = {**existing_fields, **new_fields}
        for field, value in self._infer_fields(combined, doc_type).items():
            if field not in existing_fields and field not in new_fields:
                new_fields[field] = value
                log.debug("[semantic] inferred %s = %r", field, value)

        return new_fields

    # ──────────────────────────────────────────────────────────────────────────
    # Mode A — Analyse par blocs layout
    # ──────────────────────────────────────────────────────────────────────────

    def _analyze_blocks(
        self,
        pdf_blocks: list[dict],
        doc_type: str,
        existing_fields: dict[str, str],
        existing_tiers: dict[str, str],
    ) -> dict[str, str]:
        """Analyse par blocs PDF avec contexte spatial."""
        label_map = _LABEL_FIELD_MAP.get(doc_type, {})
        # field → list of (value, confidence)
        candidates: dict[str, list[tuple[str, float]]] = {}

        for raw_block in pdf_blocks:
            text = raw_block.get("text", "").strip()
            if not text or len(text) < 2:
                continue

            bbox        = raw_block.get("bbox", (0, 0, 100, 100))
            page_width  = raw_block.get("page_width", 595.0)
            page_height = raw_block.get("page_height", 842.0)

            blk = LayoutBlock(
                text=text, bbox=bbox,
                page_width=page_width, page_height=page_height,
            )
            blk.block_role = self._classify_layout_block(blk)

            for field, value, conf in self._extract_from_block(blk, doc_type, label_map):
                candidates.setdefault(field, []).append((value, conf))

        new_fields: dict[str, str] = {}
        for field, cands in candidates.items():
            best_val, best_conf = max(cands, key=lambda x: x[1])
            existing_tier = existing_tiers.get(field)

            if field not in existing_fields:
                if best_conf >= 0.60:
                    new_fields[field] = best_val
                    log.debug("[semantic/block] NEW %s = %r (conf=%.2f)",
                              field, best_val, best_conf)
            elif existing_tier in _OVERRIDABLE_TIERS and best_conf >= 0.85:
                new_fields[field] = best_val
                log.debug(
                    "[semantic/block] OVERRIDE %s = %r (conf=%.2f, was=%r tier=%s)",
                    field, best_val, best_conf, existing_fields[field], existing_tier,
                )

        return new_fields

    def _classify_layout_block(self, blk: LayoutBlock) -> str:
        """Attribue un rôle de bloc : HEADER / PARTY_* / TRANSPORT / CARGO / TOTALS / FOOTER / BODY."""
        x0, y0, x1, y1 = blk.bbox
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        rx = cx / max(blk.page_width,  1)   # 0 = gauche, 1 = droite
        ry = cy / max(blk.page_height, 1)   # 0 = haut,   1 = bas

        upper = blk.text.strip().upper()

        # Zone d'en-tête (≤ 18 % de la hauteur)
        if ry < 0.18:
            return BLOCK_HEADER

        # Zone de pied de page (≥ 82 %, et le bord supérieur au-delà de 72 %)
        if ry > 0.82 and y0 / max(blk.page_height, 1) > 0.72:
            return BLOCK_FOOTER

        # Bloc transport / routage
        transport_kws = {
            "VESSEL", "VOYAGE", "PORT OF LOADING", "PORT OF DISCHARGE",
            "PORT OF LOAD", "LOAD PORT", "DISCHARGE PORT",
            "PLACE OF RECEIPT", "PLACE OF DELIVERY",
            "ETD", "ETA", "DATE OF DEPARTURE", "DATE OF ARRIVAL",
        }
        if any(kw in upper for kw in transport_kws):
            return BLOCK_TRANSPORT

        # Bloc totaux / termes financiers
        totals_kws = {
            "TOTAL AMOUNT", "GRAND TOTAL", "SUBTOTAL", "AMOUNT DUE",
            "FREIGHT TERMS", "PAYMENT TERMS", "INCOTERM",
            "COLLECT", "PREPAID",
        }
        if any(kw in upper for kw in totals_kws):
            # Pas un bloc d'entreprise
            if not any(re.search(sfx, upper) for sfx in _COMPANY_SUFFIXES):
                return BLOCK_TOTALS

        # Bloc cargo / conteneur
        cargo_kws = {
            "CONTAINER NO", "SEAL NO", "GROSS WEIGHT", "NET WEIGHT",
            "TOTAL PACKAGES", "DESCRIPTION OF GOODS", "CBM",
        }
        if any(kw in upper for kw in cargo_kws):
            return BLOCK_CARGO
        if re.search(r'\b[A-Z]{3,4}[UJ]\d{6,7}\b', upper):
            return BLOCK_CARGO

        # Bloc parties : contient un suffixe juridique ou un mot commercial
        has_company  = any(re.search(sfx, upper) for sfx in _COMPANY_SUFFIXES)
        has_co_word  = bool(set(upper.split()) & _COMPANY_WORDS)
        if has_company or has_co_word:
            return BLOCK_PARTY_LEFT if rx < 0.52 else BLOCK_PARTY_RIGHT

        return BLOCK_BODY

    def _extract_from_block(
        self,
        blk: LayoutBlock,
        doc_type: str,
        label_map: dict[str, str],
    ) -> list[tuple[str, str, float]]:
        """Extrait (field, value, confidence) depuis un bloc classifié."""
        block_scope = _BLOCK_FIELD_SCOPE.get(doc_type, {})
        allowed = block_scope.get(blk.block_role, set())
        if not allowed:
            # BLOCK_BODY ou rôle non contraint → tous les champs
            allowed = set(label_map.values())

        results: list[tuple[str, str, float]] = []
        line_blocks = self._annotate_blocks(blk.text)
        kv_pairs    = self._extract_kv_role_aware(line_blocks)

        for raw_label, (value, entity_type) in kv_pairs.items():
            label_up = raw_label.upper().strip()

            field = label_map.get(label_up)
            exact_match = bool(field)
            if not field:
                field = self._fuzzy_match_label(label_up, label_map)
            if not field:
                continue

            # Portée du bloc
            if field not in allowed:
                continue

            # Type entité
            allowed_types = _FIELD_ENTITY_CONSTRAINT.get(field)
            if allowed_types and entity_type not in allowed_types:
                continue

            clean = self._clean_value(field, value)
            if not clean:
                continue

            # Confiance de base
            conf = 0.65 if exact_match else 0.58

            # Bonus : entité correspondant parfaitement
            if (allowed_types
                    and entity_type in allowed_types
                    and entity_type != ENTITY_UNKNOWN):
                conf += 0.12

            # Bonus : champ dans la portée spécifique du rôle (pas BODY fallback)
            specific_roles = {
                r for r, fields in block_scope.items()
                if field in fields
            }
            if blk.block_role in specific_roles:
                conf += 0.10

            results.append((field, clean, min(conf, 0.95)))

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Mode B — Analyse sur texte plat (fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _analyze_flat(
        self,
        text: str,
        doc_type: str,
        existing_fields: dict[str, str],
    ) -> dict[str, str]:
        """Analyse KV sur texte plat (méthode originale)."""
        blocks   = self._annotate_blocks(text)
        kv_pairs = self._extract_kv_role_aware(blocks)
        label_map = _LABEL_FIELD_MAP.get(doc_type, {})
        new_fields: dict[str, str] = {}

        for raw_label, (value, entity_type) in kv_pairs.items():
            label_up = raw_label.upper().strip()
            field = label_map.get(label_up)
            if not field:
                field = self._fuzzy_match_label(label_up, label_map)
            if not field or field in existing_fields or field in new_fields:
                continue

            allowed = _FIELD_ENTITY_CONSTRAINT.get(field)
            if allowed and entity_type not in allowed:
                log.debug("[semantic/flat] rejet %s=%r type=%s attendu=%s",
                          field, value, entity_type, allowed)
                continue

            clean = self._clean_value(field, value)
            if clean:
                new_fields[field] = clean
                log.debug("[semantic/flat] %s = %r (label=%r type=%s)",
                          field, clean, raw_label, entity_type)

        return new_fields

    # ──────────────────────────────────────────────────────────────────────────
    # Annotation de lignes (rôle + type entité)
    # ──────────────────────────────────────────────────────────────────────────

    def _annotate_blocks(self, text: str) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for i, line in enumerate(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            role = self._classify_role(line)
            entity_type = (
                self._type_entity(line)
                if role not in (ROLE_SECTION, ROLE_LABEL, ROLE_AGGREGATE)
                else ENTITY_UNKNOWN
            )
            blocks.append(TextBlock(text=line, role=role,
                                    entity_type=entity_type, line_index=i))
        return blocks

    @classmethod
    def _classify_role(cls, text: str) -> str:
        upper = text.strip().upper()

        if upper in _SECTION_KEYWORDS:
            return ROLE_SECTION

        for prefix in _AGGREGATE_PREFIXES:
            if upper.startswith(prefix) and len(upper) <= len(prefix) + 30:
                return ROLE_AGGREGATE

        if upper in _KNOWN_VALUES:
            return ROLE_DATA

        for doc_map in _LABEL_FIELD_MAP.values():
            if upper in doc_map:
                return ROLE_LABEL

        etype = cls._type_entity(text)
        if etype in (ENTITY_COMPANY, ENTITY_PORT_CITY, ENTITY_DATE,
                     ENTITY_AMOUNT, ENTITY_CONTAINER, ENTITY_REFERENCE):
            return ROLE_DATA

        letters = [c for c in upper if c.isalpha()]
        if letters:
            upper_ratio  = sum(1 for c in letters if c.isupper()) / len(letters)
            has_few_digits = sum(1 for c in upper if c.isdigit()) <= 2
            short_enough   = len(upper) <= 45
            if (upper_ratio >= 0.80
                    and has_few_digits
                    and short_enough
                    and not upper[0].isdigit()):
                return ROLE_LABEL

        return ROLE_DATA

    @staticmethod
    def _type_entity(text: str) -> str:
        upper = text.strip().upper()

        # Numéro de conteneur
        if re.search(r"\b[A-Z]{3,4}[UJ]\d{6,7}\b", upper):
            return ENTITY_CONTAINER

        # Montant monétaire
        if any(cc in upper for cc in _CURRENCY_CODES) and re.search(r"\d[\d,\.]+", upper):
            return ENTITY_AMOUNT

        # Date
        if re.search(
            r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}"
            r"|\d{4}-\d{2}-\d{2}"
            r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
            r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}",
            upper,
        ):
            return ENTITY_DATE

        # Numéro de référence (lettres+chiffres, ≥ 8 chars)
        clean = re.sub(r"[\s\-/]", "", upper)
        if (len(clean) >= 8
                and re.search(r"[A-Z]{2,}", clean)
                and re.search(r"\d{4,}", clean)):
            return ENTITY_REFERENCE

        # Nom de navire : commence par un préfixe de transporteur connu
        # → ne pas classer comme port/ville
        for prefix, _ in _VESSEL_CARRIER_PREFIXES:
            if upper.startswith(prefix.rstrip()):
                return ENTITY_UNKNOWN  # navire, pas un port

        # Nom d'entreprise
        for suffix_rx in _COMPANY_SUFFIXES:
            if re.search(suffix_rx, upper):
                return ENTITY_COMPANY
        if set(upper.split()) & _COMPANY_WORDS:
            return ENTITY_COMPANY

        # Port / ville : tout en maj, ≤ 4 mots, pas de chiffres
        if (re.match(r'^[A-Z][A-Z\s,\.\-]+$', upper)
                and len(upper) >= 4
                and not re.search(r"\d", upper)
                and len(upper.split()) <= 4):
            return ENTITY_PORT_CITY

        # Code court (incoterm, devise seule…)
        if re.match(r'^[A-Z]{2,6}$', upper):
            return ENTITY_CODE

        return ENTITY_UNKNOWN

    # ──────────────────────────────────────────────────────────────────────────
    # Extraction KV rôle-consciente (3 stratégies)
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_kv_role_aware(
        self,
        blocks: list[TextBlock],
    ) -> dict[str, tuple[str, str]]:
        """
        Construit label → (valeur, entity_type).

        Stratégies :
          A. Inline colon  : "LABEL: valeur"
          B. Pipe séparé   : "LABEL: val1 | LABEL2: val2"
          C. Bloc 2 lignes : LABEL suivi d'un DATA / VALUE
        """
        pairs: dict[str, tuple[str, str]] = {}
        n = len(blocks)

        for i, blk in enumerate(blocks):

            # B. Pipe séparé
            if "|" in blk.text:
                for segment in blk.text.split("|"):
                    seg = segment.strip()
                    m = re.match(
                        r'^([A-Z][A-Z0-9 /\.\-&\'#]{1,45}?)\s*[:\s]\s*(.+)$', seg,
                    )
                    if m:
                        lbl, val = m.group(1).strip(), m.group(2).strip()
                        if self._is_clean_label(lbl) and val:
                            etype = self._type_entity(val)
                            pairs.setdefault(lbl, (val, etype))

            # A. Inline colon
            m = re.match(
                r'^([A-Z][A-Z0-9 /\.\-&\'#]{1,45}?)\s*:\s*(.+)$',
                blk.text,
            )
            if m:
                lbl, val = m.group(1).strip(), m.group(2).strip()
                if self._is_clean_label(lbl):
                    if len(val) < 3 and i + 1 < n:
                        next_blk = blocks[i + 1]
                        if next_blk.role not in (ROLE_SECTION, ROLE_LABEL, ROLE_AGGREGATE):
                            val = next_blk.text
                    etype = self._type_entity(val)
                    pairs.setdefault(lbl, (val, etype))
                continue

            # C. Bloc LABEL → valeur sur la ligne suivante
            if blk.role == ROLE_LABEL and len(blk.text) <= 45 and i + 1 < n:
                next_blk = blocks[i + 1]
                if next_blk.role not in (ROLE_SECTION, ROLE_LABEL, ROLE_AGGREGATE):
                    pairs.setdefault(blk.text, (next_blk.text, next_blk.entity_type))

        return pairs

    @staticmethod
    def _is_clean_label(s: str) -> bool:
        s = s.strip()
        if not s or len(s) < 2 or len(s) > 50:
            return False
        upper = s.upper()
        if upper in _SECTION_KEYWORDS:
            return False
        for prefix in _AGGREGATE_PREFIXES:
            if upper.startswith(prefix):
                return False
        letters = [c for c in s if c.isalpha()]
        if not letters:
            return False
        return sum(1 for c in letters if c.isupper()) / len(letters) >= 0.70

    # ──────────────────────────────────────────────────────────────────────────
    # Matching flou label → champ
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fuzzy_match_label(
        label: str,
        label_map: dict[str, str],
        threshold: float = 0.82,
    ) -> str | None:
        best_ratio = 0.0
        best_field = None
        for known, field in label_map.items():
            r = SequenceMatcher(None, label, known).ratio()
            if r > best_ratio:
                best_ratio = r
                best_field = field
        return best_field if best_ratio >= threshold else None

    # ──────────────────────────────────────────────────────────────────────────
    # Nettoyage et validation des valeurs
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_value(field: str, value: str) -> str:
        value = re.sub(r'\s+', ' ', value.strip().lstrip(":").strip())
        if not value:
            return ""

        _NUMERIC_FIELDS = {
            "gross_weight_kg", "net_weight_kg", "total_amount", "subtotal",
            "total_packages", "total_cbm", "measurement",
        }
        if field in _NUMERIC_FIELDS and not re.search(r"\d", value):
            return ""

        _DATE_FIELDS = {"issue_date", "invoice_date", "pl_date", "etd", "eta"}
        if field in _DATE_FIELDS and not re.search(r"\d{1,4}", value):
            return ""

        _FREE_TEXT_FIELDS = {
            "shipper", "consignee", "seller", "buyer", "notify_party",
            "vessel", "goods_description", "carrier",
        }
        if field in _FREE_TEXT_FIELDS and len(value) < 3:
            return ""

        # goods_description : rejeter les numéros de ligne (001, 1, A) et les codes courts
        if field == "goods_description":
            stripped = value.strip()
            if len(stripped) < 5:
                return ""  # trop court : numéro de ligne, code, etc.
            if re.fullmatch(r'\d{1,4}', stripped):
                return ""  # numéro de ligne pur (001, 1, 99…)

        # Champs de lieu : rejeter les dates et les phrases non géographiques
        _PORT_FIELDS = {
            "place_of_receipt", "place_of_delivery",
            "port_of_loading", "port_of_discharge", "issue_place",
        }
        if field in _PORT_FIELDS:
            upper_val = value.upper().strip()
            if upper_val in _PORT_FIELD_BLACKLIST:
                return ""
            if re.search(
                r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}"
                r"|\d{4}-\d{2}-\d{2}"
                r"|\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
                upper_val,
            ):
                return ""  # date dans un champ de lieu → rejet

        # number_of_originals : petit entier ou phrase connue — rejeter les dates
        if field == "number_of_originals":
            known_orig = {
                "ONE (1)", "TWO (2)", "THREE (3)", "FOUR (4)",
                "1", "2", "3", "4",
                "ORIGINAL", "ORIGINALS",
                "THREE ORIGINALS", "TWO ORIGINALS", "ONE ORIGINAL",
            }
            upper_val = value.upper().strip()
            if upper_val in known_orig:
                pass  # ok
            elif re.search(
                r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b",
                upper_val,
            ):
                return ""  # date
            elif re.search(r'\b\d{4}\b', value):
                return ""  # 4-digit year → date
            elif not re.search(r'^\s*[1-9]\d?\s*$', value.strip()):
                # Not a bare small number and not a known phrase
                if not re.search(r'\b[1-9]\b', value):
                    return ""

        # Champs de parties : rejeter les libellés évidents
        _PARTY_FIELDS = {"shipper", "consignee", "seller", "buyer", "notify_party"}
        if field in _PARTY_FIELDS:
            upper_val = value.upper()
            party_label_patterns = [
                r'\bNO\b\.?$', r'\bNUMBER\b', r'\bREF\b',
                r'\bDATE\b', r'\bCODE\b', r'\bCONTRACT\b',
            ]
            for pat in party_label_patterns:
                if re.search(pat, upper_val):
                    return ""

        if field == "container_number":
            if not re.search(r"[A-Z]{3,4}\d{6,7}", value.replace(" ", "")):
                return ""

        if field == "bl_number":
            clean_v = value.replace(" ", "")
            if len(clean_v) < 8 or not re.search(r"[A-Z]{2,4}\d{4,}", clean_v):
                return ""

        if field in ("invoice_number", "pl_number"):
            if len(value) < 5:
                return ""

        if field == "container_type":
            if not re.search(r"\b(20|40|45)\b", value):
                return ""

        return value

    # ──────────────────────────────────────────────────────────────────────────
    # Inférence domaine
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_fields(fields: dict[str, str], doc_type: str) -> dict[str, str]:
        inferred: dict[str, str] = {}

        if doc_type == "bl":
            # carrier ← préfixe bl_number
            if "carrier" not in fields:
                bl_num = fields.get("bl_number", "").upper()
                for prefix, carrier in sorted(_BL_CARRIER_PREFIXES, key=lambda x: -len(x[0])):
                    if bl_num.startswith(prefix):
                        inferred["carrier"] = carrier
                        break

            # carrier ← préfixe vessel (si toujours manquant)
            if "carrier" not in fields and "carrier" not in inferred:
                vessel = fields.get("vessel", "").upper()
                for prefix, carrier in _VESSEL_CARRIER_PREFIXES:
                    if vessel.startswith(prefix):
                        inferred["carrier"] = carrier
                        break

            # place_of_receipt / delivery ← ports
            if "place_of_receipt" not in fields and fields.get("port_of_loading"):
                inferred["place_of_receipt"] = fields["port_of_loading"]
            if "place_of_delivery" not in fields and fields.get("port_of_discharge"):
                inferred["place_of_delivery"] = fields["port_of_discharge"]

        if doc_type == "invoice" and "currency" not in fields:
            total = fields.get("total_amount", "")
            m = re.match(r'^([A-Z]{3})\s+[\d,\.]+', total.strip())
            if m:
                inferred["currency"] = m.group(1)

        return inferred
