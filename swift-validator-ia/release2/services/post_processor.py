"""
post_processor.py
=================
Post-extraction filter for LayoutLMv3 field outputs.

Fixes two systematic errors caused by synthetic-data training:
  1. Label prefix bleed — "VESSEL Vessel Name MSC OSCAR" → "MSC OSCAR"
  2. Whitespace noise — collapses multiple spaces, strips edges
  3. Field validators — rejects malformed dates, currencies, HS codes, etc.
"""

from __future__ import annotations
import re
import logging
from typing import Callable

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de validation
# ─────────────────────────────────────────────────────────────────────────────

VALID_INCOTERMS: frozenset[str] = frozenset({
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP",
    "DAP", "DPU", "DDP", "DAT",
})

# Principaux ports connus — vérification minimale (min 2 chars alphabétiques)
_PORT_MIN_ALPHA = 2

_DATE_PATTERNS = [
    re.compile(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}'),
    re.compile(r'\d{4}-\d{2}-\d{2}'),
    re.compile(
        r'\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+\d{1,2}',
        re.IGNORECASE,
    ),
]


_VALID_CURRENCIES: frozenset[str] = frozenset({
    "USD", "EUR", "GBP", "CNY", "JPY", "CHF", "AED", "SAR", "TND",
    "MAD", "DZD", "EGP", "INR", "SGD", "HKD", "CAD", "AUD", "KWD",
    "QAR", "BHD", "OMR", "JOD", "LBP", "PKR", "BDT", "MYR", "THB",
    "IDR", "VND", "PHP", "KRW", "SEK", "DKK", "NOK", "PLN", "CZK",
    "HUF", "RON", "RUB", "TRY", "BRL", "MXN", "ZAR", "NGN", "GHS",
})

# Montants raisonnables pour documents commerciaux LC (USD équivalent)
_AMOUNT_MIN = 0.01
_AMOUNT_MAX = 100_000_000.0


def _validate_date(v: str) -> bool:
    if not any(p.search(v) for p in _DATE_PATTERNS):
        return False
    # Rejet des valeurs manifestement absurdes (ex: années hors plage)
    year_m = re.search(r'\b(\d{4})\b', v)
    if year_m:
        year = int(year_m.group(1))
        if not (1990 <= year <= 2060):
            return False
    return True


def parse_amount(raw: str) -> float | None:
    """
    Normalise un montant commercial en float, quel que soit le format local.

    Détecte automatiquement :
      US/UK : 1,234,567.89  →  séparateur milliers = virgule, décimal = point
      EU/FR : 1.234.567,89  →  séparateur milliers = point,   décimal = virgule
      FR espace: 1 234 567,89  →  séparateur milliers = espace, décimal = virgule
      Sans séparateur: 1234567.89 ou 1234567,89 (1 seule virgule/point → décimale)
    """
    s = re.sub(r'[^\d.,\s]', '', raw).strip()
    if not s:
        return None

    n_dots   = s.count('.')
    n_commas = s.count(',')

    # Format US : "1,234.56" ou "1,234,567.89"
    if n_dots == 1 and n_commas >= 1:
        last_dot = s.rfind('.')
        last_com = s.rfind(',')
        if last_dot > last_com:
            s = s.replace(',', '')  # supprimer séparateurs milliers
    # Format EU : "1.234,56" ou "1.234.567,89"
    elif n_commas == 1 and n_dots >= 1:
        last_dot = s.rfind('.')
        last_com = s.rfind(',')
        if last_com > last_dot:
            s = s.replace('.', '').replace(',', '.')
    # Format FR espace : "1 234 567,89"
    elif n_commas == 1 and n_dots == 0:
        s = s.replace(' ', '').replace(',', '.')
    # Format espace + point : "1 234 567.89"
    elif n_dots == 1 and n_commas == 0:
        s = s.replace(' ', '')
    # Pas de décimale : "1234567"
    else:
        s = re.sub(r'[.,\s]', '', s)

    try:
        val = float(s)
        return val if val >= 0 else None
    except ValueError:
        return None


def _validate_positive_number(v: str) -> bool:
    """Accepte tout montant parseable > 0 dans un format commercial courant."""
    val = parse_amount(v)
    return val is not None and val > 0


def _validate_amount_range(v: str) -> bool:
    """Rejette les montants hors-plage pour un document commercial LC."""
    val = parse_amount(v)
    if val is None:
        return False
    return _AMOUNT_MIN <= val <= _AMOUNT_MAX


class PostProcessor:
    # Maps field name → list of known label strings that may bleed into values.
    # Listed longest-first so more specific prefixes are tried first.
    LABEL_PREFIXES: dict[str, list[str]] = {
        # ── Bill of Lading ───────────────────────────────────────────────────
        "bl_number": [
            "B/L NUMBER", "B/L NO.", "BL NO.", "BL N°", "BL NUMBER",
            "B/L NO", "B/L:",
        ],
        "vessel": [
            "VESSEL NAME", "VESSEL", "SHIP NAME", "M/V",
        ],
        "voyage": [
            "VOYAGE NO.", "VOYAGE NUMBER", "VOYAGE NO", "VOY NO.",
        ],
        "port_of_loading": [
            "PORT OF LOADING", "PORT OF LOAD", "LOADING PORT", "POL",
        ],
        "port_of_discharge": [
            "PORT OF DISCHARGE", "PORT OF DISCHARGING", "DISCHARGE PORT", "POD",
        ],
        "place_of_receipt": ["PLACE OF RECEIPT", "RECEIPT PLACE"],
        "place_of_delivery": ["PLACE OF DELIVERY", "DELIVERY PLACE"],
        "shipper_name": [
            "SHIPPER / EXPORTER", "SHIPPER/EXPORTER", "SHIPPER", "EXPORTER",
        ],
        "shipper": [
            "EXPORTER / SHIPPER", "EXPORTER/SHIPPER",
            "SHIPPER / EXPORTER", "SHIPPER/EXPORTER", "SHIPPER", "EXPORTER",
        ],
        "consignee_name": [
            "CONSIGNEE (TO ORDER OF)", "CONSIGNEE", "CONSIGNEE:",
        ],
        "consignee": [
            "CONSIGNEE (TO ORDER OF)", "CONSIGNEE TO ORDER OF",
            "CONSIGNEE", "BUYER / IMPORTER", "BUYER/IMPORTER", "BUYER",
        ],
        "notify_party": ["NOTIFY PARTY", "NOTIFY", "ALSO NOTIFY"],
        "container_number": [
            "CONTAINER NO.", "CONTAINER NUMBER", "CNTR NO.", "CONTAINER NO",
            "CNTR NO. / SEAL",
        ],
        "seal_number": ["SEAL NO.", "SEAL NUMBER", "SEAL NO"],
        "container_type": ["CONTAINER TYPE", "CNTR TYPE"],
        "gross_weight_kg": [
            "GROSS WEIGHT (KG)", "GROSS WEIGHT", "GROSS WT (KG)",
            "GROSS WT.", "GROSS WT",
        ],
        "net_weight_kg": [
            "NET WEIGHT (KG)", "NET WEIGHT", "NET WT (KG)",
            "NET WT.", "NET WT",
        ],
        "measurement": ["MEASUREMENT", "VOLUME", "TOTAL CBM", "CBM"],
        "etd": ["ESTIMATED TIME OF DEPARTURE", "ETD"],
        "eta": ["ESTIMATED TIME OF ARRIVAL", "ETA"],
        "freight_terms": [
            "FREIGHT & CHARGES", "FREIGHT PAYABLE AT", "FREIGHT TERMS",
        ],
        "number_of_originals": [
            "NUMBER OF ORIGINAL B/L", "NO. OF ORIGINAL B/L",
            "NUMBER OF ORIGINALS",
        ],
        "service_type": ["SERVICE TYPE", "SERVICE"],
        "issue_date": ["DATE OF ISSUE", "ISSUE DATE", "DATE ISSUED"],
        "issue_place": ["PLACE OF ISSUE", "ISSUE PLACE"],
        "incoterm": ["INCOTERMS", "INCOTERM"],

        # ── Commercial Invoice ────────────────────────────────────────────────
        "invoice_number": [
            "INVOICE NUMBER", "INVOICE NO.", "INVOICE NO", "INV NO.", "INV NO",
        ],
        "seller_name": [
            "SELLER / EXPORTER", "SELLER/EXPORTER", "SELLER", "EXPORTER",
        ],
        "seller": [
            "SELLER / EXPORTER", "SELLER/EXPORTER", "SELLER", "EXPORTER",
            "SUPPLIER", "VENDOR",
        ],
        "buyer_name": [
            "BUYER / IMPORTER", "BUYER/IMPORTER", "BUYER", "IMPORTER",
        ],
        "buyer": [
            "BUYER / IMPORTER", "BUYER/IMPORTER", "BUYER", "IMPORTER",
            "BILL TO", "SHIP TO", "SOLD TO", "CUSTOMER",
        ],
        "goods_description": [
            "DESCRIPTION OF GOODS", "GOODS DESCRIPTION", "DESCRIPTION",
        ],
        "hs_code": ["HS CODE", "HS NO.", "HS NO", "HARMONIZED CODE"],
        "unit_price": ["UNIT PRICE", "PRICE PER UNIT"],
        "total_amount": [
            "TOTAL AMOUNT", "INVOICE AMOUNT", "AMOUNT", "TOTAL",
        ],
        "currency": ["CURRENCY"],
        "lc_reference": [
            "LC REFERENCE", "LC REF.", "LC REF", "LC NO.", "L/C NO.",
        ],
        "bl_reference": [
            "B/L REFERENCE", "B/L REF.", "B/L REF", "BL REFERENCE", "BL REF",
        ],
        "payment_terms": ["PAYMENT TERMS", "TERMS OF PAYMENT"],

        # ── Packing List ─────────────────────────────────────────────────────
        "pl_number": [
            "PACKING LIST NO.", "PL NUMBER", "PL NO.", "PL NO", "PL N°",
        ],
        "exporter_name": [
            "EXPORTER / SHIPPER", "EXPORTER/SHIPPER", "EXPORTER", "SHIPPER",
        ],
        "exporter": [
            "EXPORTER / SHIPPER", "EXPORTER/SHIPPER", "EXPORTER", "SHIPPER",
        ],
        "total_packages": [
            "TOTAL PACKAGES", "TOTAL PKGS", "TOTAL PKG",
        ],
        "total_cbm": ["TOTAL CBM", "TOTAL VOLUME"],
    }

    _MULTI_SPACE = re.compile(r"\s+")

    # Valeurs qui sont systématiquement des en-têtes de colonne ou des libellés,
    # jamais des valeurs de champ réelles. Vérification insensible à la casse.
    _GLOBAL_LABEL_VALUES: frozenset[str] = frozenset({
        "BOOKING REF", "BOOKING REF.", "BOOKING REFERENCE",
        "BOOKING NO", "BOOKING NO.", "BOOKING NUMBER",
        "VESSEL / VOYAGE", "VESSEL/VOYAGE",
        "VESSEL NAME / VOYAGE NO", "VESSEL NAME/VOYAGE NO",
        "VESSEL / VOYAGE NO", "VESSEL / VOY",
        "PORT OF LOADING / PORT OF DISCHARGE",
        "POL / POD", "POL/POD",
        "CNTR NO / SEAL NO", "CNTR NO. / SEAL NO.",
        "CONTAINER NO / SEAL NO", "CONTAINER NO. / SEAL NO.",
        "PLACE OF RECEIPT / PLACE OF DELIVERY",
        "GROSS / NET WEIGHT",
    })

    # Champs scalaires qui doivent contenir au moins un chiffre
    _MUST_CONTAIN_DIGIT: frozenset[str] = frozenset({
        "bl_number", "invoice_number", "pl_number",
        "container_number", "seal_number",
    })

    # Validateurs par champ : retournent True si la valeur est acceptable
    FIELD_VALIDATORS: dict[str, Callable[[str], bool]] = {
        # Numéros de documents
        "container_number": lambda v: bool(
            re.fullmatch(r'[A-Z]{4}\d{7}', re.sub(r'[\s\-]', '', v.upper()))
        ),
        "bl_number": lambda v: len(v.strip()) >= 4,
        "invoice_number": lambda v: len(v.strip()) >= 3,
        "pl_number": lambda v: len(v.strip()) >= 3,
        "seal_number": lambda v: len(v.strip()) >= 2,
        # Dates
        "issue_date":    _validate_date,
        "invoice_date":  _validate_date,
        "etd":           _validate_date,
        "eta":           _validate_date,
        "pl_date":       _validate_date,
        # Codes commerce international
        "hs_code": lambda v: bool(
            re.fullmatch(r'\d{6,10}', re.sub(r'[.\s\-]', '', v))
        ),
        "currency": lambda v: bool(re.fullmatch(r'[A-Z]{3}', v.strip())),
        "incoterm": lambda v: v.strip().upper().rstrip('.') in VALID_INCOTERMS,
        # Champs textuels
        "vessel": lambda v: len(v.strip()) >= 2 and not v.strip().isdigit(),
        "port_of_loading":    lambda v: len(re.sub(r'\d', '', v).strip()) >= _PORT_MIN_ALPHA,
        "port_of_discharge":  lambda v: len(re.sub(r'\d', '', v).strip()) >= _PORT_MIN_ALPHA,
        "place_of_receipt":   lambda v: len(v.strip()) >= 2,
        "place_of_delivery":  lambda v: len(v.strip()) >= 2,
        # Champs numériques — validés avec détection format EU/US
        "gross_weight_kg":   _validate_positive_number,
        "net_weight_kg":     _validate_positive_number,
        "measurement":       _validate_positive_number,
        "total_cbm":         _validate_positive_number,
        "unit_price":        _validate_positive_number,
        "quantity":          _validate_positive_number,
        # Montants — plage raisonnable pour un document LC
        "total_amount":    _validate_amount_range,
        "subtotal":        _validate_amount_range,
        "freight_charges": _validate_amount_range,
        "insurance":       _validate_amount_range,
        "tax":             _validate_amount_range,
        # Devise — 3 lettres ISO uniquement
        "currency": lambda v: v.strip().upper() in _VALID_CURRENCIES,
        # Dates supplémentaires
        "on_board_date":  _validate_date,
        "bl_date":        _validate_date,
        # Incoterm — liste fermée
        "incoterm": lambda v: v.strip().upper().rstrip('.') in VALID_INCOTERMS,
    }

    def clean(self, fields: dict[str, str], doc_type: str = "unknown") -> dict[str, str]:
        """Apply all normalization passes to an extracted fields dict."""
        result: dict[str, str] = {}
        for field, value in fields.items():
            cleaned = self._normalize_field(field, value)
            if self._is_global_label(cleaned):
                continue
            if field in self._MUST_CONTAIN_DIGIT and not re.search(r'\d', cleaned):
                continue
            validator = self.FIELD_VALIDATORS.get(field)
            if validator is not None:
                try:
                    if not validator(cleaned):
                        log.debug("Champ '%s' rejeté (validation échouée): '%s'", field, cleaned[:60])
                        continue
                except Exception:
                    pass  # erreur dans le validateur → conserver la valeur
            result[field] = cleaned
        return result

    def _is_global_label(self, value: str) -> bool:
        """True si la valeur est un en-tête de colonne ou un libellé connu."""
        return value.strip().upper().rstrip(".") in {
            v.upper().rstrip(".") for v in self._GLOBAL_LABEL_VALUES
        }

    def _normalize_field(self, field_name: str, value: str) -> str:
        value = self._normalize_whitespace(value)
        value = self._strip_label_prefix(field_name, value)
        value = self._normalize_whitespace(value)
        return value

    def _normalize_whitespace(self, value: str) -> str:
        return self._MULTI_SPACE.sub(" ", str(value)).strip()

    def _strip_label_prefix(self, field_name: str, value: str) -> str:
        prefixes = self.LABEL_PREFIXES.get(field_name, [])
        # Try longest prefix first for specificity
        for prefix in sorted(prefixes, key=len, reverse=True):
            if value.upper().startswith(prefix.upper()):
                stripped = value[len(prefix):].lstrip(" :–—-").strip()
                if stripped:
                    return stripped
        return value
