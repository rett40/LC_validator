"""
native_extractor.py
===================
Extraction de champs par regex depuis PDFs natifs (texte vectoriel PyMuPDF).

Zéro ML — exploite la structure textuelle connue des documents générés.
Précision élevée sur documents synthétiques et réels bien formatés.

Usage :
  from release2.services.native_extractor import NativeExtractor
  ext = NativeExtractor()
  result = ext.extract("doc.pdf", doc_type="bl")
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import fitz

try:
    from .post_processor import PostProcessor
except ImportError:
    from post_processor import PostProcessor  # standalone fallback

log = logging.getLogger(__name__)

_pp = PostProcessor()

# ─────────────────────────────────────────────────────────────────────────────
# Patterns regex par type de document
# Les patterns essaient de capturer la valeur après le label connu.
# Ordre important : les plus spécifiques d'abord.
# ─────────────────────────────────────────────────────────────────────────────

_DATE_RX = (
    r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+\d{4}"
    r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+\d{1,2},?\s+\d{4})"
)

# Validator pattern used in _kv_fallback (more permissive than _DATE_RX full match)
_DATE_VALIDATOR_RX = (
    r"\d{1,4}[-/\.]\d{1,2}"
    r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"|\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}"
)

# Valeurs à rejeter — mots génériques capturés par erreur
_LABEL_WORDS = {
    "NAME", "DETAILS", "INFORMATION", "VALUE", "DATE", "NUMBER", "NO",
    "TYPE", "INCOTERM", "INCOTERMS",
    "PORT OF LOADING", "PORT OF DISCHARGE", "OF DISCHARGE", "OF LOADING",
    "FINAL DESTINATION",
    "PLACE OF DELIVERY", "PLACE OF RECEIPT", "PLACE OF ORIGIN",
    "CURRENCY", "ITEM", "ITEMS", "HS CODE", "DESCRIPTION", "GOODS", "TOTAL",
    "AMOUNT", "QUANTITY", "UNIT", "PRICE", "REFERENCE", "REF",
    "ARTICLE", "ARTICLES", "PRODUCT", "PRODUCTS", "COMMODITIES", "MERCHANDISE",
    "PARTICULARS", "NATURE OF GOODS", "CARGO DESCRIPTION",
    "RATE", "UNIT COST", "LINE TOTAL", "LINE AMOUNT", "NET AMOUNT",
    "TARIFF CODE", "TARIFF NO", "HTS", "COMMODITY CODE",
    "COUNTRY OF ORIGIN", "ORIGIN",
    "NOTIFY", "PARTY", "SHIPPER", "CONSIGNEE", "VESSEL", "VOYAGE",
    # Étiquettes composées fréquentes dans les factures / BL
    "BUYER / IMPORTER", "SELLER / EXPORTER",
    "BUYER/IMPORTER", "SELLER/EXPORTER",
    "SHIPPER/EXPORTER", "SHIPPER / EXPORTER",
    "PO NO", "PO NUMBER", "PO REF", "PO.",
    "BILL TO", "SHIP TO",
    # En-têtes de colonnes souvent capturés à tort par le KV-fallback
    "BOOKING REF", "BOOKING REFERENCE", "BOOKING NO",
    "COUNTRY OF ORIGIN", "ORIGIN", "COUNTRY",
    "SIZE/TYPE", "SIZE TYPE", "CONTAINER SIZE",
    "COMMODITY", "COMMODITY CODE",
    "MARKS AND NUMBERS", "MARKS & NOS",
    "NET/GROSS", "GROSS/NET",
    "B/L DATE", "ISSUE DATE", "DATE OF ISSUE",
    # En-têtes de tableau cargo (colonnes BL)
    "PACKAGES", "DESCRIPTION", "CBM",
    "CONTAINER NO", "CONTAINER NO.", "SEAL NO", "SEAL NO.",
    "CONTAINER TYPE", "CNTR TYPE",
    "GROSS WEIGHT", "NET WEIGHT",
}

_BL_PATTERNS: dict[str, list[str]] = {
    "shipper": [
        r"SHIPPER\s*(?:/EXPORTER)?\s*[:.]?\s*\n\s*([A-Z][A-Z\s,\-\.\d]{5,100}?)(?=\s*\n\s*(?:CONSIGNEE|NOTIFY|VESSEL|VOYAGE|PORT))",
        r"SHIPPER\s*(?:/EXPORTER)?\s*[:.]?\s*([A-Z][A-Z\s,\-\.\d]{5,100}?)(?=\s*\n\s*(?:CONSIGNEE|NOTIFY|VESSEL|VOYAGE|PORT))",
        # Grid format: "SHIPPER   TUNISIAN EXPORTER CO.   CONSIGNEE..."
        r"SHIPPER\s{2,}([A-Z][A-Z\s,\-\.\d]{5,80}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "consignee": [
        r"CONSIGNEE\s*(?:NAME)?\s*[:.]?\s*\n\s*([A-Z][A-Z\s,\-\.\d]{5,100}?)(?=\s*\n\s*(?:NOTIFY|VESSEL|VOYAGE|PORT))",
        r"CONSIGNEE\s*(?:NAME)?\s*[:.]?\s*([A-Z][A-Z\s,\-\.\d]{5,100}?)(?=\s*\n\s*(?:NOTIFY|VESSEL|VOYAGE|PORT))",
        # Grid: "CONSIGNEE   GLOBALTRADE GMBH   NOTIFY..."
        r"CONSIGNEE\s{2,}([A-Z][A-Z\s,\-\.\d]{5,80}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "notify_party": [
        r"NOTIFY\s+(?:PARTY)?\s*[:.]?\s*\n\s*([A-Z][A-Z\s,\-\.\d]{5,100}?)(?=\s*\n\s*(?:VESSEL|VOYAGE|PORT))",
        r"NOTIFY\s+(?:PARTY)?\s*[:.]?\s*([A-Z][A-Z\s,\-\.\d]{5,100}?)(?=\s*\n\s*(?:VESSEL|VOYAGE|PORT))",
        # Grid: "NOTIFY PARTY   SPEDITION MULLER   VESSEL..."
        r"NOTIFY\s+(?:PARTY)?\s{2,}([A-Z][A-Z\s,\-\.\d]{5,80}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "vessel": [
        r"VESSEL\s*[:.]?\s*([A-Z][A-Z\s\-]{3,40}?)(?=\s*\n|\s+VOYAGE|\s+IMO)",
        r"M\.?V\.?\s+([A-Z][A-Z\s\-]{2,35}?)(?=\s*\n|\s+VOYAGE)",
        r"VESSEL\s+NAME\s*[:.]?\s*([A-Z][A-Z\s\-]{3,40}?)(?=\s*\n|\s+VOYAGE|\s+IMO)",
        # Grid/tab-separated: "VESSEL   MSC BEATRICE   VOYAGE..."
        r"VESSEL\s{2,}([A-Z][A-Z\s\-]{3,35}?)(?=\s{2,}[A-Z]|\n)",
        # Inline block: "VESSEL NAME\nMSC BEATRICE"
        r"VESSEL(?:\s+NAME)?\s*\n\s*([A-Z][A-Z\s\-]{3,40}?)(?=\s*\n)",
    ],
    "voyage": [
        r"VOYAGE\s*(?:NO\.?|NUMBER|N[°o])\s*[:.]?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{2,16})",
        r"VOY\.?\s*(?:NO\.?)?\s*[:.]?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{2,16})",
        # Grid: "VOYAGE NO.   050A   PORT..."
        r"VOYAGE\s+(?:NO\.?\s+)?((?=[A-Z0-9]*\d)[A-Z0-9]{2,16})(?=\s{2,}|\n)",
        # Next-line: "VOYAGE NO.\n050A"
        r"VOYAGE\s*(?:NO\.?|NUMBER)?\s*\n\s*((?=[A-Z0-9]*\d)[A-Z0-9]{2,16})(?=\s*\n)",
    ],
    "port_of_loading": [
        r"PORT\s+OF\s+LOADING\s*\n([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        r"PORT\s+OF\s+LOADING\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,35}?)(?=\s*\n)",
        r"LOADING\s+PORT\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,35}?)(?=\s*\n)",
        r"POL\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,35}?)(?=\s*\n)",
        # Grid/tab: "PORT OF LOADING   RADES-TUNIS   PORT OF DISCHARGE..."
        r"PORT\s+OF\s+LOADING\s{2,}([A-Z][A-Z\s,\-]{2,35}?)(?=\s{2,}[A-Z]|\n)",
        # Freight box: "ORIGIN\nRADES PORT TUNISIA"
        r"(?:PORT\s+OF\s+)?ORIGIN\s*[:.]?\s*\n\s*([A-Z][A-Z\s,\-]{4,40}?)(?=\s*\n)",
    ],
    "place_of_receipt": [
        r"PLACE\s+OF\s+RECEIPT\s*\n([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        r"PLACE\s+OF\s+RECEIPT\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        # Grid
        r"PLACE\s+OF\s+RECEIPT\s{2,}([A-Z][A-Z\s,\-]{2,40}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "place_of_delivery": [
        r"PLACE\s+OF\s+DELIVERY\s*\n([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        r"PLACE\s+OF\s+DELIVERY\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        # Grid
        r"PLACE\s+OF\s+DELIVERY\s{2,}([A-Z][A-Z\s,\-]{2,40}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "place_of_receipt": [
        r"PLACE\s+OF\s+RECEIPT\s*\n([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        r"PLACE\s+OF\s+RECEIPT\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        # Grid
        r"PLACE\s+OF\s+RECEIPT\s{2,}([A-Z][A-Z\s,\-]{2,40}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "place_of_delivery": [
        r"PLACE\s+OF\s+DELIVERY\s*\n([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        r"PLACE\s+OF\s+DELIVERY\s*[:.]?\s*([A-Z][A-Z\s,\-]{2,40}?)(?=\s*\n)",
        # Grid
        r"PLACE\s+OF\s+DELIVERY\s{2,}([A-Z][A-Z\s,\-]{2,40}?)(?=\s{2,}[A-Z]|\n)",
    ],
    "etd": [
        rf"ETD\s*[:.]?\s*{_DATE_RX}",
        rf"ESTIMATED\s+(?:TIME|DATE)\s+OF\s+DEPARTURE\s*[:.]?\s*{_DATE_RX}",
    ],
    "eta": [
        rf"ETA\s*[:.]?\s*{_DATE_RX}",
        rf"ESTIMATED\s+(?:TIME|DATE)\s+OF\s+ARRIVAL\s*[:.]?\s*{_DATE_RX}",
    ],
    "issue_date": [
        rf"DATE\s+OF\s+ISSUE\s*[:.]?\s*{_DATE_RX}",
        rf"ISSUE\s+DATE\s*[:.]?\s*{_DATE_RX}",
        rf"DATE\s+ISSUED\s*[:.]?\s*{_DATE_RX}",
    ],
    "container_number": [
        r"([A-Z]{4}\d{6}\d)(?!\d)",   # ISO 6346 — pas de \b final (collision avec "IF2")
    ],
    "seal_number": [
        r"SEAL\s*(?:NO\.?|NUMBER)\s*[:.]?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{4,15})",
        r"SEAL\s*[:.]?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{4,15})(?!\w)",
        # Cargo table : conteneur puis joint sur la ligne suivante
        r"[A-Z]{4}\d{7}\n((?=[A-Z0-9]*\d)[A-Z0-9]{4,15})(?!\w)",
    ],
    "container_type": [
        # [^\S\n]* = pas de saut de ligne entre le chiffre et l'abréviation
        # PL retiré (trop ambigu : postal code + "PLACE OF RECEIPT" → faux positif)
        r"\b((?:20|40|45)['\"]?[^\S\n]*(?:GP|HC|OT|RF|TK|FR|FT|NOR|HQ|DRY|DC|HD|REEFER|TANK|OPEN[^\S\n]*TOP|FLAT[^\S\n]*RACK))\b",
        r"\b((?:20|40|45)['\"]?[^\S\n]*(?:FOOT|FT)?[^\S\n]*(?:DRY|STANDARD|GENERAL|TANK|HIGH[^\S\n]*CUBE?)(?:[^\S\n]*CONTAINER)?)\b",
    ],
    "gross_weight_kg": [
        r"GROSS\s+(?:WEIGHT|WT\.?)\s*(?:\(KGS?\))?\s*[:.]?\s*([\d\s.,]+?)\s*KGS?\b",
        # Format « XX,XXX.XX KG » (pas de S) — courant dans les BL MSC/Maersk
        r"([\d,]+\.?\d+)\s*KG\b\n[\d,]+\.?\d+\s*CBM\b",
        r"GROSS\s+(?:WEIGHT|WT\.?)\s*[:.]?\s*([\d\s.,]+)\b",
        r"G\.WEIGHT\s*(?:\(KGS?\))?\s*[:.]?\s*([\d.,]+)",
    ],
    "net_weight_kg": [
        r"NET\s+(?:WEIGHT|WT\.?)\s*(?:\(KGS?\))?\s*[:.]?\s*([\d\s.,]+?)\s*KGS?\b",
        r"NET\s+(?:WEIGHT|WT\.?)\s*[:.]?\s*([\d\s.,]+)\b",
    ],
    "measurement": [
        r"(?:MEASUREMENT|TOTAL\s+CBM)\s*[:.]?\s*([\d.,]+)\s*(?:CBM|M3)?",
        r"MEAS\s*(?:\(CBM\))?\s*[:.]?\s*([\d.,]+)\s*(?:CBM)?",
        r"([\d.,]+)\s*CBM\b",
    ],
    "freight_terms": [
        r"FREIGHT\s*(?:TERMS|PAYABLE\s+AT|&\s+CHARGES)\s*[:.]?\s*(PREPAID|COLLECT)",
        r"\b(FREIGHT\s+PREPAID|FREIGHT\s+COLLECT)\b",
        # Format BL bas de page : "PREPAID  3 ORIGINALS" ou "COLLECT  3 ORIGINALS"
        r"(?:^|\n)\s*(PREPAID|COLLECT)\s+\d+\s+ORIGINAL",
        # Standalone sur sa propre ligne : \nPREPAID\n
        r"(?:^|\n)\s*(PREPAID|COLLECT)\s*(?:\n|$)",
    ],
    "incoterm": [
        r"INCOTERMS?\s*[:.]?\s*(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP|DAP|DAT|DPU)\b",
        r"\b(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP|DAP|DAT|DPU)\b",
    ],
    "currency": [
        r"CURRENCY\s*[:.]?\s*([A-Z]{3})\b",
        r"FREIGHT\s+(?:CURRENCY|CCY)\s*[:.]?\s*([A-Z]{3})\b",
        r"\b(USD|EUR|GBP|CNY|JPY|CHF|AED|SAR|TND|MAD|DZD|EGP|INR|SGD|HKD|CAD|AUD|KWD|QAR)\b",
    ],
    "total_packages": [
        r"TOTAL\s+(?:PACKAGES|PKGS?)\s*[:.]?\s*(\d[\d,]*)\s*(?:CARTONS?|BOXES|PALLETS?|BAGS?|DRUMS?|PIECES?|PKGS?)?",
        r"(?:^|\n)(?:TOTAL\s+)?NO\.?\s+OF\s+PACKAGES?\s*[:.]?\s*(\d[\d,]*)",
        r"PACKAGES?\s*(?:TOTAL|:)\s*(\d[\d,]*)",
    ],
    "on_board_date": [
        rf"(?:SHIPPED\s+ON\s+BOARD|ON\s+BOARD\s+DATE|LADEN\s+ON\s+BOARD)\s*[:.]?\s*{_DATE_RX}",
        rf"DATE\s+OF\s+(?:SHIPMENT|LOADING)\s*[:.]?\s*{_DATE_RX}",
        # Short date without year: "ON BOARD 22 MARCH" — captured then normalized
        r"(?:SHIPPED\s+ON\s+BOARD|ON\s+BOARD)\s*[:.]?\s*(\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?(?:\s+\d{4})?)",
    ],
    "number_of_originals": [
        r"(?:NUMBER|NO\.?)\s+OF\s+ORIGINAL[S]?\s+B/?L\s*[:.]?\s*(\d+(?:\s+\([A-Z]+\))?)",
    ],
    "shipper": [
        r"SHIPPER\s*/\s*EXPORTER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"SHIPPER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"SHIPPER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
    "consignee": [
        # Parenthétique optionnel après CONSIGNEE : "(TO ORDER OF)", "(TO ORDER)", etc.
        r"CONSIGNEE\s*(?:\([^)]{0,40}\))?[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"CONSIGNEE[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
    "notify_party": [
        r"NOTIFY\s+PARTY[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"ALSO\s+NOTIFY[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"NOTIFY\s+PARTY[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
    "carrier": [
        r"(?:OCEAN\s+)?CARRIER[ \t]*[:.][ \t]*([A-Z][A-Z\s]{2,30}?)(?=[ \t]*(?:\n|\())",
        r"SHIPPING\s+(?:LINE|COMPANY)[ \t]*[:.][ \t]*([A-Z][A-Z\s]{2,30}?)(?=[ \t]*\n)",
    ],
}

_PARTY_FIELD_LABELS: dict[str, dict[str, list[str]]] = {
    "bl": {
        "shipper": [
            "SHIPPER / EXPORTER", "SHIPPER/EXPORTER", "EXPORTER / SHIPPER",
            "EXPORTER/SHIPPER", "SHIPPER", "EXPORTER", "CONSIGNOR",
            "SHIPPED BY", "EXPORTER OF RECORD",
        ],
        "consignee": [
            "CONSIGNEE (TO THE ORDER OF)", "CONSIGNEE (TO ORDER OF)",
            "CONSIGNEE", "TO THE ORDER OF", "TO ORDER OF",
            "CONSIGNEE ADDRESS", "CONSIGNEE / NOTIFY",
        ],
        "notify_party": [
            "NOTIFY PARTY", "ALSO NOTIFY", "NOTIFY", "NOTIFY ADDRESS",
            "NOTIFY TO", "THIRD NOTIFY PARTY",
        ],
    },
    "invoice": {
        "seller": [
            "SELLER / EXPORTER", "SELLER/EXPORTER", "EXPORTER / SELLER",
            "EXPORTER/SELLER", "SELLER", "EXPORTER", "SUPPLIER", "VENDOR",
            "SOLD BY",
        ],
        "buyer": [
            "BUYER / IMPORTER", "BUYER/IMPORTER", "IMPORTER / BUYER",
            "IMPORTER/BUYER", "BUYER", "IMPORTER", "BILL TO", "SHIP TO",
            "SOLD TO", "CUSTOMER", "CONSIGNEE",
        ],
    },
    "packing_list": {
        "shipper": [
            "EXPORTER / SHIPPER", "EXPORTER/SHIPPER", "SHIPPER / EXPORTER",
            "SHIPPER/EXPORTER", "SHIPPER", "EXPORTER", "CONSIGNOR",
        ],
        "consignee": [
            "CONSIGNEE", "BUYER / IMPORTER", "BUYER/IMPORTER", "BUYER",
            "IMPORTER", "CONSIGNEE ADDRESS",
        ],
    },
}

_PARTY_STOP_LABELS: set[str] = {
    "VESSEL", "VESSEL NAME", "VOYAGE", "VOYAGE NO", "VOYAGE NO.",
    "PORT OF LOADING", "PORT OF DISCHARGE", "PLACE OF RECEIPT",
    "PLACE OF DELIVERY", "CONTAINER NO", "CONTAINER NO.", "CONTAINER TYPE",
    "SEAL NO", "SEAL NO.", "B/L REFERENCE", "BL REFERENCE", "INVOICE REF",
    "INVOICE REFERENCE", "DESCRIPTION", "DESCRIPTION OF GOODS", "HS CODE",
    "QTY", "QUANTITY", "PKGS", "UNIT", "UNIT PRICE", "AMOUNT",
    "DIMENSIONS", "NET WT", "NET WT (KG)", "GROSS WT", "GROSS WT (KG)",
    "CBM", "MARKS", "MARKS AND NUMBERS", "PAYMENT TERMS", "INCOTERM",
    "INCOTERMS",
    "TOTAL PACKAGES", "TOTAL NET WEIGHT", "TOTAL GROSS WEIGHT",
    "TOTAL CBM", "TOTAL AMOUNT", "AUTHORIZED SIGNATORY",
    "AUTHORIZED SIGNATORY & STAMP", "COMMERCIAL INVOICE", "PACKING LIST",
    "BILL OF LADING",
}
for _labels_by_field in _PARTY_FIELD_LABELS.values():
    for _labels in _labels_by_field.values():
        _PARTY_STOP_LABELS.update(_labels)

_PARTY_COMPANY_SUFFIX_RE = re.compile(
    r"\b(?:LTD|LIMITED|GMBH|INC|LLC|S\.?A\.?|AG|SARL|CORP|CORPORATION|"
    r"CO\.?|BV|NV|PLC|SRL|SPRL|PTY|KG|OY|AB|SAS|SPA)\b",
    re.IGNORECASE,
)
_PARTY_COMPANY_WORD_RE = re.compile(
    r"\b(?:INTERNATIONAL|GLOBAL|TRADING|COMMERCE|SHIPPING|LOGISTICS|IMPORT|"
    r"EXPORT|INDUSTRIES|ENTERPRISES|GROUP|PARTNERS|SERVICES|SOLUTIONS|"
    r"SUPPLY|CHAIN|FREIGHT|MARITIME|TEXTILE|PHARMA|CHEMICAL|ELECTRONICS|"
    r"MANUFACTURING|TECHNOLOGIES|FOODS?|MACHINERY|MATERIALS?)\b",
    re.IGNORECASE,
)
_PARTY_ADDRESS_RE = re.compile(
    r"\b(?:ROAD|RD\.?|STREET|ST\.?|AVENUE|AVE\.?|PARK|ZONE|BUILDING|BLDG|"
    r"FLOOR|ROOM|SUITE|DISTRICT|INDUSTRIAL|HARBOR|HARBOUR|TRADE|P\.?O\.?|"
    r"PO BOX|ZIP|POSTAL)\b",
    re.IGNORECASE,
)
_PARTY_REFERENCE_RE = re.compile(
    r"\b(?:[A-Z]{4}\d{7}"
    r"|(?:INV|PL|LC|BL|B/L)[-/A-Z0-9]{4,}"
    r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)"
    r"|CONTRACT\s+(?:NO\.?|NUMBER|REF\.?|#)"
    r"|(?:P\.?O\.?|PURCHASE\s+ORDER)\s+(?:NO\.?|NUMBER|#)"
    r"|(?:REF|REFERENCE)\s*(?:NO\.?|:)\s*\w"
    r")\b",
    re.IGNORECASE,
)

_INVOICE_PATTERNS: dict[str, list[str]] = {
    "invoice_number": [
        r"INVOICE\s*(?:NUMBER|NO\.?|N[°o])\s*[:.]?\s*([A-Z0-9\-\/\.]{4,30})",
    ],
    "invoice_date": [
        rf"(?:INVOICE\s+DATE|DATE)\s*[:.]?\s*{_DATE_RX}",
    ],
    "total_amount": [
        r"TOTAL\s+(?:AMOUNT|INVOICE\s+AMOUNT|VALUE)\s*[:.]?\s*([A-Z]{3}\s*[\d\s.,]+)",
        r"INVOICE\s+AMOUNT\s*[:.]?\s*([A-Z]{3}\s*[\d\s.,]+)",
        r"AMOUNT\s+DUE\s*[:.]?\s*([A-Z]{3}\s*[\d\s.,]+)",
    ],
    "currency": [
        r"\b(USD|EUR|GBP|CNY|JPY|CHF|AED|SAR)\b",
    ],
    "payment_terms": [
        r"PAYMENT\s+TERMS\s*[:.]?\s*((?:L/?C|T/?T|CASH|SIGHT|ADVANCE|DEFERRED|AT\s+SIGHT|USANCE)[A-Z\s\/\d]{0,40}?)(?=\n|INCOTERM|$)",
        r"TERMS\s+OF\s+PAYMENT\s*[:.]?\s*((?:L/?C|T/?T|CASH|SIGHT|ADVANCE|DEFERRED)[A-Z\s\/\d]{0,40}?)(?=\n)",
        r"PAYMENT\s+TERMS\s*[:.]?\s*([A-Z][A-Z\s\/\d]{3,40}?)(?=\n)",
    ],
    "incoterm": [
        r"INCOTERMS?\s*[:.]?\s*(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP|DAP|DAT|DPU)\b",
        r"\b(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP)\b",
    ],
    "hs_code": [
        r"HS\s*(?:CODE|NO\.?)\s*[:.]?\s*(\d{6,10})",
    ],
    "bl_reference": [
        r"B/?L\s*(?:REFERENCE|REF\.?)\s*[:.]?\s*([A-Z]{3,5}[A-Z0-9]{8,18})",
        r"B/?L\s*(?:NO\.?|NUMBER)\s*[:.]?\s*([A-Z]{3,5}[A-Z0-9]{8,18})",
    ],
    "goods_description": [
        # Après les en-têtes de table : AMOUNT est le dernier header avant les données
        r"UNIT\s+PRICE\s*\nAMOUNT\s*\n([A-Z][A-Z\s,\/\-\d]{4,80}?)(?=\n)",
        r"AMOUNT\s*\n([A-Z][A-Z\s,\/\-\d]{4,80}?)(?=\n[\d])",
        r"DESCRIPTION\s+OF\s+GOODS\s*[:.]?\s*\n([A-Z][A-Z\s,\/\-\d]{4,100}?)(?=\n)",
        r"DESCRIPTION\s+OF\s+GOODS\s*[:.]?\s*([A-Z][A-Z\s,\/\-]{4,80}?)(?=\n|HS\s*CODE|QUANTITY|UNIT)",
    ],

    "seller": [
        r"SELLER\s*/\s*EXPORTER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"SELLER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"SELLER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"EXPORTER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
    "buyer": [
        r"BUYER\s*/\s*IMPORTER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"BUYER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"BUYER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"IMPORTER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
    "subtotal": [
        r"SUB[\s\-]?TOTAL\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
        r"GOODS\s+AMOUNT\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
        r"MERCHANDISE\s+VALUE\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
    ],
    "freight_charges": [
        r"FREIGHT\s*(?:CHARGES?|COST|AMOUNT|FEE)?\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
        r"OCEAN\s+FREIGHT\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
        r"SEA\s+FREIGHT\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
        r"AIR\s+FREIGHT\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
    ],
    "insurance": [
        r"INSURANCE\s*(?:PREMIUM|AMOUNT|CHARGES?)?\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
        r"INS\.?\s*[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
    ],
    "tax": [
        r"(?:VAT|TAX|GST)\s*(?:\d+%\s*)?[:.]?\s*(?:[A-Z]{3}\s*)?([\d\s.,]+)",
    ],
}

_PL_PATTERNS: dict[str, list[str]] = {
    "pl_number": [
        r"PACKING\s+LIST\s*(?:NUMBER|NO\.?|N[°o])\s*[:.]?\s*([A-Z0-9\-\/\.]{4,25})",
        r"PL\s*(?:NUMBER|NO\.?)\s*[:.]?\s*([A-Z0-9\-\/\.]{4,25})",
        # Format « No. PL-2026-00002 | … » — label "No." avant le numéro
        r"(?:^|\n|\|)\s*No\.?\s+((?:PL|INV|SHP|REF|DOC|PKG)[\-\/][A-Z0-9\-\/\.]{3,20})",
        r"PACKING\s+LIST\s*[:.]?\s*((?=[A-Z0-9\-\/\.]*\d)[A-Z0-9\-\/\.]{4,25})",
    ],
    "total_packages": [
        r"TOTAL\s+(?:PACKAGES|PKGS?)\s*[:.]?\s*([\d,]+\s*(?:CARTONS?|BOXES|PALLETS?|BAGS?|DRUMS?|PIECES?)?)",
        # Format « PKGS: 66 CARTONS » (résumé en bas de PL)
        r"(?:^|\n)PKGS?[:]\s*(\d[\d,]*)\s*(?:CARTONS?|BOXES|PALLETS?|BAGS?|PIECES?)?",
    ],
    "total_cbm": [
        r"TOTAL\s+(?:CBM|VOLUME)\s*[:.]?\s*([\d.,]+)\s*(?:CBM|M3|m3)?",
    ],
    "gross_weight_kg": [
        r"(?:TOTAL\s+)?GROSS\s+(?:WEIGHT|WT\.?)\s*(?:\(KG\))?\s*[:.]?\s*([\d\s.,]+)\s*KG\b",
    ],
    "net_weight_kg": [
        r"(?:TOTAL\s+)?NET\s+(?:WEIGHT|WT\.?)\s*(?:\(KG\))?\s*[:.]?\s*([\d\s.,]+)\s*KG\b",
    ],
    "container_number": [
        r"([A-Z]{4}\d{6}\d)(?!\d)",
    ],
    "bl_reference": [
        r"B/?L\s*(?:REFERENCE|REF\.?|NO\.?|NUMBER)\s*[:.]?\s*([A-Z]{3,5}[A-Z0-9]{8,18})",
    ],
    "incoterm": [
        r"INCOTERMS?\s*[:.]?\s*(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP|DAP|DAT|DPU)\b",
        r"TERMS?\s+OF\s+(?:TRADE|DELIVERY|SALE)\s*[:.]?\s*(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP|DAP|DAT|DPU)\b",
        r"\b(FOB|CIF|CFR|EXW|DDP|FCA|CPT|CIP|DAP|DAT|DPU)\b",
    ],
    "currency": [
        r"CURRENCY\s*[:.]?\s*([A-Z]{3})\b",
        r"\b(USD|EUR|GBP|CNY|JPY|CHF|AED|SAR|TND|MAD|DZD|EGP|INR|SGD|HKD|CAD|AUD|KWD|QAR)\b",
    ],
    "invoice_reference": [
        r"INVOICE\s*(?:REFERENCE|REF\.?|NO\.?|NUMBER)\s*[:.]?\s*([A-Z0-9\-\/\.]{4,25})",
    ],
    "goods_description": [
        r"(?:DESCRIPTION\s+OF\s+GOODS|COMMODITY|NATURE\s+OF\s+GOODS)\s*[:.]?\s*([A-Z][A-Z\s,\/\-\.]{4,100}?)(?=\n)",
        r"CARGO\s+DESCRIPTION\s*[:.]?\s*([A-Z][A-Z\s,\/\-\.]{4,100}?)(?=\n)",
    ],
    "pl_date": [
        rf"(?:PACKING\s+LIST\s+DATE|PL\s+DATE)\s*[:.]?\s*{_DATE_RX}",
        rf"DATE\s*[:.]?\s*{_DATE_RX}",
    ],
    "shipper": [
        r"SHIPPER[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"SHIPPER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"EXPORTER[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
    "consignee": [
        r"CONSIGNEE[ \t]*\n[ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
        r"CONSIGNEE[ \t]*[:.][ \t]*([A-Z][A-Z0-9\s,\.&\-\(\)]{3,60}?)(?=[ \t]*\n)",
    ],
}

_PATTERNS: dict[str, dict[str, list[str]]] = {
    "bl":           _BL_PATTERNS,
    "invoice":      _INVOICE_PATTERNS,
    "packing_list": _PL_PATTERNS,
}

# ─────────────────────────────────────────────────────────────────────────────
# Niveau 2 — Fallback générique key-value
# Variantes de labels connues pour chaque champ.
# Priorité : du plus spécifique au plus générique.
# ─────────────────────────────────────────────────────────────────────────────

_KV_LABELS: dict[str, list[str]] = {
    # Bill of Lading
    "bl_number": [
        "B/L NO", "B/L NUMBER", "BILL OF LADING NO", "BILL OF LADING NUMBER",
        "BL NO", "BL NUMBER", "B.L. NO", "B.L. NUMBER",
        "B/L N°", "BILL OF LADING N°", "BL N°", "B/L REF",
        "BL REFERENCE", "B/L REFERENCE", "BILL OF LADING REF",
        "B/L.", "CONNAISSEMENT NO", "CONNAISSEMENT N°",
    ],
    "vessel": [
        "VESSEL NAME", "VESSEL", "SHIP NAME", "SHIP", "M/V", "MV",
        "NAME OF VESSEL", "OCEAN VESSEL",
    ],
    "voyage": [
        "VOYAGE NO", "VOYAGE NUMBER", "VOY NO", "VOY. NO",
        "VOYAGE NO.", "VOY", "VOYAGE",
    ],
    "port_of_loading": [
        "PORT OF LOADING", "LOADING PORT", "PORT OF SHIPMENT",
        "POL", "LOAD PORT", "PORT OF ORIGIN",
    ],
    "port_of_discharge": [
        "PORT OF DISCHARGE", "DISCHARGE PORT", "DISCHARGING PORT",
        "POD", "UNLOADING PORT",
    ],
    "place_of_receipt": [
        "PLACE OF RECEIPT", "PORT OF RECEIPT", "PLACE OF ORIGIN",
        "RECEIPT PLACE", "PRE-CARRIAGE FROM",
    ],
    "place_of_delivery": [
        "PLACE OF DELIVERY", "DELIVERY PORT", "PLACE OF FINAL DELIVERY",
        "DELIVERY PLACE",
    ],
    "etd": [
        "ETD", "DATE OF DEPARTURE", "DEPARTURE DATE",
        "ESTIMATED DEPARTURE", "ESTIMATED TIME OF DEPARTURE",
    ],
    "eta": [
        "ETA", "DATE OF ARRIVAL", "ARRIVAL DATE",
        "ESTIMATED ARRIVAL", "ESTIMATED TIME OF ARRIVAL",
    ],
    "issue_date": [
        "DATE OF ISSUE", "ISSUE DATE", "DATE ISSUED",
        "ISSUED DATE", "DATE OF B/L",
    ],
    "seal_number": [
        "SEAL NO", "SEAL NUMBER", "CUSTOMS SEAL", "SEAL NO.",
        "SEAL #", "SEAL",
    ],
    "container_type": [
        "CONTAINER TYPE", "CNTR TYPE", "SIZE/TYPE", "CONT TYPE",
        "CONTAINER SIZE", "TYPE/SIZE",
    ],
    "gross_weight_kg": [
        "GROSS WEIGHT (KG)", "GROSS WEIGHT (KGS)", "GROSS WEIGHT",
        "G.WEIGHT (KG)", "G.WEIGHT (KGS)", "G.WEIGHT",
        "G.W.", "GWT", "GROSS WT",
    ],
    "net_weight_kg": [
        "NET WEIGHT (KG)", "NET WEIGHT (KGS)", "NET WEIGHT",
        "N.WEIGHT (KG)", "N.WEIGHT",
        "N.W.", "NWT", "NET WT",
    ],
    "measurement": [
        "MEASUREMENT (CBM)", "MEASUREMENT (M3)", "MEASUREMENT",
        "TOTAL CBM", "TOTAL VOLUME (CBM)",
        "MEAS (CBM)", "MEAS (M3)", "MEAS",
    ],
    "freight_terms": [
        "FREIGHT TERMS", "FREIGHT PAYABLE", "FREIGHT & CHARGES",
        "TERMS OF FREIGHT",
    ],
    "on_board_date": [
        "ON BOARD DATE", "SHIPPED ON BOARD", "DATE OF SHIPMENT",
        "DATE OF LOADING", "LADEN ON BOARD", "ON BOARD",
    ],
    "number_of_originals": [
        "NUMBER OF ORIGINAL B/L", "NO OF ORIGINAL B/L", "NO. OF ORIGINAL B/L",
        "ORIGINALS", "NUMBER OF ORIGINALS", "NO. OF ORIGINALS",
    ],
    # Commercial Invoice
    "invoice_number": [
        "INVOICE NO", "INVOICE NUMBER", "INV NO", "INV. NO",
        "INVOICE NO.", "COMMERCIAL INVOICE NO",
    ],
    "invoice_date": [
        "INVOICE DATE", "DATE OF INVOICE", "INV DATE",
    ],
    "total_amount": [
        "TOTAL AMOUNT", "INVOICE AMOUNT", "TOTAL INVOICE AMOUNT",
        "AMOUNT DUE", "GRAND TOTAL",
    ],
    "payment_terms": [
        "PAYMENT TERMS", "TERMS OF PAYMENT", "PAYMENT METHOD",
        "MODE OF PAYMENT",
    ],
    "bl_reference": [
        "B/L REFERENCE", "B/L NO", "B/L REF", "BILL OF LADING REF",
    ],
    "goods_description": [
        "DESCRIPTION OF GOODS", "GOODS DESCRIPTION",
        "DESCRIPTION OF COMMODITIES", "DESCRIPTION OF ITEMS",
        "COMMODITY", "COMMODITIES", "NATURE OF GOODS",
        "ITEMS", "ARTICLE", "ARTICLES", "PRODUCT", "PRODUCTS",
        "MERCHANDISE", "PARTICULARS", "CARGO DESCRIPTION",
    ],
    # BL party fields
    "shipper": [
        "SHIPPER", "SHIPPER/EXPORTER", "EXPORTER", "SHIPPED BY", "CONSIGNOR",
    ],
    "consignee": [
        "CONSIGNEE", "CONSIGNEE/NOTIFY", "CONSIGNEE ADDRESS",
    ],
    "notify_party": [
        "NOTIFY PARTY", "ALSO NOTIFY", "NOTIFY ADDRESS", "NOTIFY TO",
    ],
    "carrier": [
        "CARRIER", "OCEAN CARRIER", "SHIPPING LINE", "SHIPPING COMPANY", "OPERATED BY",
    ],
    # Invoice party fields
    "seller": [
        "SELLER", "EXPORTER", "SUPPLIER", "FROM", "SHIPPED BY",
    ],
    "buyer": [
        "BUYER", "IMPORTER", "BILL TO", "SHIP TO", "CUSTOMER",
    ],
    "subtotal": [
        "SUBTOTAL", "SUB-TOTAL", "SUB TOTAL", "AMOUNT BEFORE TAX",
        "GOODS AMOUNT", "MERCHANDISE VALUE", "GOODS VALUE",
    ],
    "freight_charges": [
        "FREIGHT CHARGES", "FREIGHT COST", "FREIGHT AMOUNT",
        "OCEAN FREIGHT", "SEA FREIGHT", "AIR FREIGHT", "FREIGHT",
    ],
    "insurance": [
        "INSURANCE", "INSURANCE PREMIUM", "INSURANCE CHARGES",
        "INS.", "INS",
    ],
    "tax": [
        "VAT", "TAX", "GST", "VAT AMOUNT", "TAX AMOUNT",
    ],
    # Packing List
    "pl_date": [
        "PACKING LIST DATE", "PL DATE", "DATE",
    ],
    "pl_number": [
        "PACKING LIST NO", "PACKING LIST NUMBER", "PL NO",
        "PL NUMBER", "PACKING LIST NO.",
    ],
    "total_packages": [
        "TOTAL PACKAGES", "TOTAL PKGS", "TOTAL CARTONS",
        "TOTAL BOXES", "TOTAL NO. OF PACKAGES", "TOTAL NO OF PACKAGES",
    ],
    "total_cbm": [
        "TOTAL CBM", "TOTAL VOLUME (CBM)", "TOTAL VOLUME",
        "TOTAL MEASUREMENT",
    ],
    "incoterm": [
        "INCOTERM", "INCOTERMS", "TERMS OF TRADE", "TERMS OF DELIVERY",
        "TERMS OF SALE", "TRADE TERMS",
    ],
    "currency": [
        "CURRENCY", "CCY", "INVOICE CURRENCY", "PAYMENT CURRENCY",
    ],
    "invoice_reference": [
        "INVOICE REFERENCE", "INVOICE REF", "INV REF", "INV NO", "INVOICE NO",
    ],
    "goods_description": [
        "DESCRIPTION OF GOODS", "GOODS DESCRIPTION", "COMMODITY",
        "NATURE OF GOODS", "CARGO DESCRIPTION", "DESCRIPTION OF COMMODITIES",
    ],
    # BL additional
    "total_packages": [
        "TOTAL PACKAGES", "TOTAL PKGS", "NO. OF PACKAGES",
        "NUMBER OF PACKAGES", "TOTAL NO OF PACKAGES",
    ],
    "bl_currency": [
        "CURRENCY", "FREIGHT CURRENCY",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

_COL_WIDTH = 100  # quantification de la colonne en points PDF


def _extract_pages_text(pdf_path: Path) -> list[str]:
    """
    Extrait le texte page par page en groupant les blocs par sous-colonne logique.

    Algorithme :
      • Chaque bloc est assigné à un « seau » de colonne = int(center_x // COL_WIDTH).
      • Au sein d'un seau, les blocs sont triés par y croissant.
      • Les seaux sont concaténés de gauche à droite.
    Résultat : label et valeur d'une même cellule de formulaire se retrouvent
    adjacents dans le texte, quelle que soit la disposition multi-colonnes.
    """
    doc   = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        blocks  = page.get_text("blocks")
        col_map: dict[int, list[tuple[float, str]]] = {}
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            cx      = (b[0] + b[2]) / 2
            col_key = int(cx // _COL_WIDTH)
            col_map.setdefault(col_key, []).append((b[1], text))
        lines: list[str] = []
        for col_key in sorted(col_map):
            for _, text in sorted(col_map[col_key], key=lambda t: t[0]):
                lines.append(text)
        pages.append("\n".join(lines))
    doc.close()
    return pages


def _canonical_label(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text.upper())
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_PARTY_STOP_KEYS = {_canonical_label(label) for label in _PARTY_STOP_LABELS}


def _is_party_stop_line(line: str) -> bool:
    key = _canonical_label(line.rstrip(":"))
    if not key:
        return True
    if key in _PARTY_STOP_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in (
        "TOTAL ", "SUBTOTAL", "GRAND TOTAL", "AUTHORIZED SIGNATORY",
    ))


def _inline_party_value(line: str, labels: list[str]) -> str | None:
    for label in sorted(labels, key=len, reverse=True):
        rx = rf"^\s*{re.escape(label)}\s*[:.\-]\s*(.+?)\s*$"
        m = re.match(rx, line, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _line_is_party_label(line: str, labels: list[str]) -> bool:
    key = _canonical_label(line.rstrip(":"))
    return key in {_canonical_label(label) for label in labels}


def _party_score(value: str) -> float:
    val = re.sub(r"\s+", " ", value.strip().strip("|:;"))
    if not val or _is_party_stop_line(val):
        return -10.0

    upper = val.upper()
    score = 0.0
    words = re.findall(r"[A-Z][A-Z.&'-]*", upper)

    if _PARTY_COMPANY_SUFFIX_RE.search(val):
        score += 5.0
    if _PARTY_COMPANY_WORD_RE.search(val):
        score += 3.0
    if 1 <= len(words) <= 7:
        score += min(len(words), 4) * 0.5
    if re.match(r"^\d", val):
        score -= 5.0
    if _PARTY_ADDRESS_RE.search(val):
        score -= 4.0
    if _PARTY_REFERENCE_RE.search(val):
        score -= 5.0
    if "," in val and not _PARTY_COMPANY_SUFFIX_RE.search(val):
        score -= 2.0
    if sum(ch.isdigit() for ch in val) > max(2, len(val) // 4):
        score -= 3.0

    return score


def _select_party_name(lines: list[str]) -> str:
    candidates: list[tuple[float, int, str]] = []
    for idx, line in enumerate(lines):
        val = re.sub(r"\s+", " ", line.strip().strip("|:;"))
        if not val:
            continue
        score = _party_score(val)
        if score > -5:
            candidates.append((score, -idx, val))

    if not candidates:
        return ""

    best_score, _, best_value = max(candidates, key=lambda item: (item[0], item[1]))
    return best_value if best_score > 0 else ""


def _extract_party_fields(text: str, doc_type: str) -> dict[str, str]:
    labels_by_field = _PARTY_FIELD_LABELS.get(doc_type, {})
    if not labels_by_field:
        return {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    found: dict[str, str] = {}

    for field, labels in labels_by_field.items():
        best_value = ""
        best_score = -10.0

        for i, line in enumerate(lines):
            inline = _inline_party_value(line, labels)
            if inline:
                value = _select_party_name([inline])
                score = _party_score(value) if value else -10.0
                if value and score > best_score:
                    best_value, best_score = value, score
                continue

            if not _line_is_party_label(line, labels):
                continue

            block: list[str] = []
            for next_line in lines[i + 1:i + 7]:
                if _is_party_stop_line(next_line):
                    break
                block.append(next_line)

            value = _select_party_name(block)
            score = _party_score(value) if value else -10.0
            if value and score > best_score:
                best_value, best_score = value, score

        if best_value:
            found[field] = best_value

    return found


def _party_value_is_suspicious(value: str) -> bool:
    if not value:
        return True
    return _party_score(value) <= 0


def _is_valid_value(val: str) -> bool:
    """Rejette les captures qui sont des labels/entêtes, pas des valeurs."""
    if not val or len(val) < 2:
        return False
    stripped = val.strip()
    # Rejeter les lignes de séparation / signature (____... ou ----... ou ===...)
    if re.fullmatch(r"[_\-=\.]{3,}", stripped):
        return False
    # Rejeter les valeurs commençant par des underscores (ligne de signature)
    if stripped.startswith("___"):
        return False
    upper = stripped.upper()
    # Rejeter les étiquettes de colonnes au format "MOT/MOT" (ex: "SIZE/TYPE", "NET/GROSS")
    if re.fullmatch(r"[A-Z]{2,12}/[A-Z]{2,12}", upper):
        return False
    # Rejeter les valeurs 100% majuscules sans chiffre de 2-4 mots (hors noms/incoterms)
    words_up = upper.split()
    if (2 <= len(words_up) <= 4
            and all(re.fullmatch(r"[A-Z]{2,15}", w) for w in words_up)
            and not any(c.isdigit() for c in upper)
            and upper not in {
                "FREIGHT PREPAID", "FREIGHT COLLECT",
                "FOB PORT", "CIF PORT", "FREE ON BOARD",
                # Termes de paiement courants
                "AT SIGHT", "UPON SIGHT", "SIGHT DRAFT",
                "CASH ADVANCE", "OPEN ACCOUNT", "DEFERRED PAYMENT",
                "USANCE DRAFT", "LETTER OF CREDIT", "DOCUMENTS AGAINST",
            }):
        _KNOWN_INCOTERMS = {"FOB","CIF","CFR","EXW","DDP","FCA","CPT","CIP","DAP","DPU"}
        # Suffixes légaux/commerciaux courants dans les noms de sociétés
        _COMPANY_SUFFIXES = {
            "LTD", "LIMITED", "GMBH", "INC", "LLC", "SA", "AG", "SARL",
            "CORP", "CORPORATION", "CO", "BV", "NV", "PLC", "SRL", "SPRL",
            "PTY", "KG", "OY", "AB", "SAS", "SPA",
        }
        # Mots caractéristiques d'un nom de société ou d'un navire
        _COMPANY_WORDS = {
            "INTERNATIONAL", "GLOBAL", "TRADING", "COMMERCE", "SHIPPING",
            "LOGISTICS", "IMPORT", "EXPORT", "INDUSTRIES", "ENTERPRISES",
            "GROUP", "PARTNERS", "SERVICES", "SOLUTIONS",
        }
        # Préfixes de compagnies maritimes (noms de navires : MSC OSCAR, OOCL HONG KONG…)
        _CARRIER_CODES = {
            "MSC", "MAERSK", "CMA", "HAPAG", "EVERGREEN", "ONE",
            "COSCO", "YANG", "ZIM", "HAMBURG", "OOCL", "APL", "PIL",
            "MOL", "NYK", "KLINE", "UASC", "WHL", "KMTC", "HMM",
            "SITC", "CSCL", "PACIFIC", "ORIENT", "CHINA", "NORDIC",
        }
        last_word  = words_up[-1]
        first_word = words_up[0]
        if (last_word  not in _COMPANY_SUFFIXES
                and not any(w in _COMPANY_WORDS for w in words_up)
                and first_word not in _CARRIER_CODES
                and first_word not in _KNOWN_INCOTERMS):
            return False
    # Rejet si valeur exactement dans la blacklist
    if upper in _LABEL_WORDS:
        return False
    # Rejet si valeur commence par un label connu (ex: "Port of Discharge ...")
    # Vérification mot-entier : "UNIT" ne doit pas rejeter "UNITED INDUSTRIES LTD"
    for label in _LABEL_WORDS:
        if upper.startswith(label) and (
            len(upper) == len(label)
            or upper[len(label)] in " /,;:.(["
        ):
            return False
    # Rejet des indicateurs d'unité purs : "(KGS)", "KG", "(CBM)", "MT", etc.
    if re.fullmatch(r"\(?\s*(?:KGS?|LBS?|MT|CBM|M3|CU\.?\s*M|PCS?|PKGS?|CTNS?)\s*\)?", upper):
        return False
    # Rejet si valeur ne contient que des majuscules génériques d'un seul mot
    if re.fullmatch(r"[A-Z]{3,15}", upper) and not any(c.isdigit() for c in upper):
        if upper in {"NAME", "DETAILS", "INFORMATION", "NUMBER", "VALUE",
                     "DATE", "TOTAL", "AMOUNT", "DESCRIPTION", "REFERENCE",
                     "CURRENCY", "QUANTITY", "VESSEL", "VOYAGE", "SHIPPER",
                     "MEASUREMENT", "WEIGHT", "VOLUME"}:
            return False
    return True


_PORT_LIKE_FIELDS: frozenset[str] = frozenset({
    "port_of_loading", "port_of_discharge", "place_of_receipt", "place_of_delivery",
})

_CARRIER_PREFIX_RE = re.compile(
    r"^(?:MSC|MAERSK|CMA(?:\s+CGM)?|HAPAG(?:\s*-?\s*LLOYD)?|EVERGREEN|ONE\b|"
    r"COSCO|YANG(?:\s+MING)?|ZIM|HAMBURG|OOCL|APL|PIL|MOL|NYK|HMM|SITC|CSCL|WHL)\b",
    re.IGNORECASE,
)


def detect_present_fields(text: str, doc_type: str) -> set[str]:
    """
    Pré-scan du texte pour détecter quels champs sont mentionnés.
    Utilise règles + modèle BERT si disponible pour prédiction.
    """
    present = set()
    text_up = text.upper()

    # Règles basiques
    if doc_type == "bl":
        if "SHIPPER" in text_up: present.add("shipper")
        if "CONSIGNEE" in text_up: present.add("consignee")
        if "NOTIFY" in text_up: present.add("notify_party")
        if "VESSEL" in text_up: present.add("vessel")
        if "VOYAGE" in text_up: present.add("voyage")
        if "PORT OF LOADING" in text_up: present.add("port_of_loading")
        if "PORT OF DISCHARGE" in text_up: present.add("port_of_discharge")
        if "COMMODITY" in text_up or "DESCRIPTION OF GOODS" in text_up: present.add("commodity")

    # Placeholder pour modèle BERT : entraîner sur dataset pour prédire champs présents
    # Ex. : from transformers import pipeline
    # classifier = pipeline("text-classification", model="bert-field-classifier")
    # predictions = classifier(text, candidate_labels=["shipper", "consignee", ...])
    # for pred in predictions: if pred["score"] > 0.5: present.add(pred["label"])

    return present


def _apply(text: str, patterns: dict[str, list[str]], doc_type: str) -> dict[str, str]:
    """
    Applique les patterns regex et retourne les champs extraits.
    Priorise les champs détectés présents dans le document.
    """
    fields: dict[str, str] = {}
    vessel_from_port: str = ""

    # Détecter champs présents pour priorisation
    present_fields = detect_present_fields(text, doc_type)
    prioritized_fields = list(present_fields) + [f for f in patterns.keys() if f not in present_fields]

    # Détecter layout : grid si beaucoup d'espaces multiples
    is_grid = len(re.findall(r'\s{3,}', text)) > 5  # Heuristique simple

    for field in prioritized_fields:
        if field not in patterns:
            continue
        rxs = patterns[field]
        # Adapter : si grid, privilégier patterns avec \s{2,}
        if is_grid:
            rxs = [rx for rx in rxs if r'\s{2,}' in rx] + rxs
        for rx in rxs:
            m = re.search(rx, text, re.IGNORECASE | re.MULTILINE)
            if m:
                val = m.group(1).strip().rstrip(".,;:")
                if not _is_valid_value(val):
                    continue
                if field in _PORT_LIKE_FIELDS and _CARRIER_PREFIX_RE.match(val):
                    if not vessel_from_port:
                        vessel_from_port = val
                    continue
                fields[field] = val
                break

    if vessel_from_port and "vessel" not in fields:
        fields["vessel"] = vessel_from_port

    return fields


# Champs nécessitant au moins un chiffre dans la valeur
_KV_NUMERIC_FIELDS: frozenset[str] = frozenset({
    "gross_weight_kg", "net_weight_kg", "measurement", "total_cbm",
    "total_amount", "total_packages", "number_of_originals", "subtotal", "tax",
})

# Champs nécessitant un format date (au moins dd/mm ou yyyy-mm)
_KV_DATE_FIELDS: frozenset[str] = frozenset({
    "etd", "eta", "issue_date", "invoice_date", "pl_date", "on_board_date",
})


def _kv_fallback(text: str, fields_needed: set[str]) -> dict[str, str]:
    """
    Niveau 2 — fallback générique key-value.
    Pour chaque champ manquant, cherche les variantes de label connues avec
    trois formats :
      • 'LABEL : valeur'   (même ligne, séparateur ponctuation)
      • 'LABEL   valeur'   (même ligne, séparateur ≥2 espaces)
      • 'LABEL\\nvaleur'   (valeur sur la ligne suivante)
    Applique des validateurs de type (numérique, date) pour éviter les faux positifs.
    """
    found: dict[str, str] = {}
    for field in fields_needed:
        for label in _KV_LABELS.get(field, []):
            escaped = re.escape(label)
            m = (
                # Format "LABEL : valeur" (séparateur ponctuation)
                re.search(
                    rf"(?:^|\n){escaped}\s*[:.\-]\s*(.+?)(?:\n|$)",
                    text, re.IGNORECASE | re.MULTILINE,
                )
                or
                # Format "LABEL   valeur" (séparateur ≥2 espaces, style tableau)
                re.search(
                    rf"(?:^|\n){escaped}\s{{2,}}(.+?)(?:\n|$)",
                    text, re.IGNORECASE | re.MULTILINE,
                )
                or
                # Format "LABEL\nvaleur" (deux lignes)
                re.search(
                    rf"(?:^|\n){escaped}\s*\n\s*(.+?)(?:\n|$)",
                    text, re.IGNORECASE | re.MULTILINE,
                )
            )
            if not m:
                continue
            val = m.group(1).strip().rstrip(".,;:")
            if not _is_valid_value(val):
                continue
            # Validation de type : les champs numériques doivent contenir un chiffre
            if field in _KV_NUMERIC_FIELDS and not re.search(r"\d", val):
                continue
            # Validation de type : les champs date doivent ressembler à une date
            if field in _KV_DATE_FIELDS and not re.search(
                _DATE_VALIDATOR_RX, val, re.IGNORECASE
            ):
                continue
            found[field] = val
            break
    return found


def _spacy_enrich(text: str, fields_needed: set[str]) -> dict[str, str]:
    """
    Niveau 3 — enrichissement par spaCy NER (optionnel).
    Active uniquement si 'spacy' et 'en_core_web_sm' sont installés.
    Couvre : dates (etd, eta, issue_date, invoice_date) et lieux (ports).
    """
    try:
        import spacy  # noqa: PLC0415
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return {}

    doc = nlp(text[:8000])  # limite pour performance

    date_ents  = [e.text.strip() for e in doc.ents if e.label_ == "DATE"]
    place_ents = [e.text.strip() for e in doc.ents if e.label_ in ("GPE", "LOC")]

    found: dict[str, str] = {}

    # Dates : assignées aux champs date par ordre d'apparition dans le document
    date_fields_ordered = ["issue_date", "invoice_date", "etd", "eta"]
    date_queue = list(date_ents)
    for df in date_fields_ordered:
        if df in fields_needed and date_queue:
            # Cherche la date qui apparaît le plus proche du label dans le texte
            label_hint = _KV_LABELS.get(df, [df])[0]
            best = None
            best_dist = 10_000
            for d in date_queue:
                m = re.search(rf"{re.escape(label_hint)}[^\n]{{0,80}}{re.escape(d)}", text, re.IGNORECASE)
                if m and m.start() < best_dist:
                    best_dist = m.start()
                    best = d
            if best:
                found[df] = best
                date_queue.remove(best)
            elif date_queue:
                found[df] = date_queue.pop(0)

    # Lieux : associe l'entité la plus proche du label de port
    port_label_map = {
        "port_of_loading":   "PORT OF LOADING",
        "port_of_discharge": "PORT OF DISCHARGE",
        "place_of_receipt":  "PLACE OF RECEIPT",
        "place_of_delivery": "PLACE OF DELIVERY",
    }
    for field, label_kw in port_label_map.items():
        if field not in fields_needed or not place_ents:
            continue
        for place in place_ents:
            m = re.search(
                rf"{re.escape(label_kw)}[^\n]{{0,60}}{re.escape(place)}",
                text, re.IGNORECASE,
            )
            if m:
                found[field] = place
                place_ents.remove(place)
                break

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Service principal
# ─────────────────────────────────────────────────────────────────────────────

class NativeExtractor:
    """
    Extraction de champs depuis PDFs natifs — pipeline 3 niveaux :
      1. Regex spécifiques par champ (précision maximale)
      2. Fallback générique key-value (couverture des formats inconnus)
      3. spaCy NER optionnel (dates et lieux résiduels)
    """

    def __init__(self, use_spacy: bool = False) -> None:
        self._use_spacy = use_spacy

    def debug_text(self, pdf_path: Path | str) -> str:
        """Retourne le texte brut extrait — utile pour diagnostiquer les patterns."""
        pages_text = _extract_pages_text(Path(pdf_path))
        return "\n===PAGE BREAK===\n".join(pages_text)

    def extract_from_text(self, text: str, doc_type: str) -> dict:
        """
        Extrait les champs depuis un texte déjà disponible (OCR ou PyMuPDF).
        Même pipeline 3 niveaux que extract(), sans lecture PDF.
        Utilisé comme fallback regex sur le texte OCR des documents scannés.
        """
        patterns = _PATTERNS.get(doc_type, {})

        fields = _apply(text, patterns, doc_type)
        tiers  = {f: "regex" for f in fields}

        party_fields = _extract_party_fields(text, doc_type)
        for f, v in party_fields.items():
            current = fields.get(f, "")
            if (
                not current
                or _party_value_is_suspicious(current)
                or _party_score(v) > _party_score(current)
                or (_party_score(v) == _party_score(current) and len(v) > len(current))
            ):
                fields[f] = v
                tiers[f]  = "party_block"

        missing_after1 = set(patterns.keys()) - fields.keys()
        if missing_after1:
            kv = _kv_fallback(text, missing_after1)
            for f, v in kv.items():
                fields[f] = v
                tiers[f]  = "kv_fallback"

        fields = _pp.clean(fields, doc_type)
        return {"doc_type": doc_type, "fields": fields, "extraction_tiers": tiers}

    def extract(self, pdf_path: Path | str, doc_type: str) -> dict:
        """
        Extrait les champs d'un PDF natif.

        Retour
        ------
        {
          "doc_type"       : str,
          "n_pages"        : int,
          "fields"         : {champ: valeur, ...},
          "extraction_tiers": {champ: "regex"|"kv_fallback"|"spacy", ...},
          "pages"          : [{"page": int, "fields": {...}}, ...],
        }
        """
        pdf_path = Path(pdf_path)
        patterns = _PATTERNS.get(doc_type, {})

        pages_text = _extract_pages_text(pdf_path)
        full_text  = "\n".join(pages_text)

        # — Niveau 1 : regex spécifiques ———————————————————————————————————————
        fields = _apply(full_text, patterns, doc_type)
        tiers  = {f: "regex" for f in fields}

        party_fields = _extract_party_fields(full_text, doc_type)
        for f, v in party_fields.items():
            current = fields.get(f, "")
            if (
                not current
                or _party_value_is_suspicious(current)
                or _party_score(v) > _party_score(current)
                or (_party_score(v) == _party_score(current) and len(v) > len(current))
            ):
                fields[f] = v
                tiers[f] = "party_block"

        # — Niveau 2 : fallback KV générique pour les champs manquants ————————
        all_fields     = set(patterns.keys())
        missing_after1 = all_fields - fields.keys()
        if missing_after1:
            kv = _kv_fallback(full_text, missing_after1)
            for f, v in kv.items():
                fields[f] = v
                tiers[f]  = "kv_fallback"

        # — Niveau 3 : enrichissement spaCy (optionnel) ———————————————————————
        missing_after2 = all_fields - fields.keys()
        if self._use_spacy and missing_after2:
            sp = _spacy_enrich(full_text, missing_after2)
            for f, v in sp.items():
                fields[f] = v
                tiers[f]  = "spacy"

        fields = _pp.clean(fields, doc_type)

        pages_result: list[dict] = []
        for i, page_text in enumerate(pages_text, start=1):
            page_fields = _apply(page_text, patterns, doc_type)
            pages_result.append({"page": i, "fields": page_fields})

        log.debug(
            "[%s] %d champs — regex:%d kv:%d spacy:%d",
            pdf_path.name, len(fields),
            sum(1 for t in tiers.values() if t == "regex"),
            sum(1 for t in tiers.values() if t == "kv_fallback"),
            sum(1 for t in tiers.values() if t == "spacy"),
        )

        return {
            "doc_type"        : doc_type,
            "n_pages"         : len(pages_result),
            "fields"          : fields,
            "extraction_tiers": tiers,
            "pages"           : pages_result,
        }
