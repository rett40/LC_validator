"""
rules_engine.py
===============
Couche 3 du pipeline — moteur de règles métier pour documents LC.

Valide, corrige et enrichit les champs extraits selon les contraintes métier.

Règles :
  Universelles : normalisation poids/CBM/montants, incoterm, currency,
                 freight_terms
  BL           : checksum ISO 6346 conteneur, ordre dates, format BL number
  Invoice      : normalisation montant total, dates
  PL           : packages entier positif, CBM positif
  Inter-docs   : container_number identique BL↔Invoice↔PL, poids ±5%

Usage :
  from release2.services.rules_engine import RulesEngine
  engine = RulesEngine()
  corrected_fields, results = engine.validate(fields, doc_type="bl")
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Structures
# ─────────────────────────────────────────────────────────────────────────────

RuleStatus = Literal["pass", "fail", "corrected", "warning", "skipped"]


@dataclass
class RuleResult:
    rule:      str
    field:     str | None
    status:    RuleStatus
    original:  str | None = None
    corrected: str | None = None
    message:   str = ""

    def to_dict(self) -> dict:
        return {
            "rule":      self.rule,
            "field":     self.field,
            "status":    self.status,
            "original":  self.original,
            "corrected": self.corrected,
            "message":   self.message,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Constantes métier
# ─────────────────────────────────────────────────────────────────────────────

VALID_INCOTERMS: frozenset[str] = frozenset({
    "FOB", "CIF", "CFR", "EXW", "DDP", "FCA", "CPT", "CIP", "DAP", "DAT", "DPU",
})

VALID_CURRENCIES: frozenset[str] = frozenset({
    "USD", "EUR", "GBP", "CNY", "JPY", "CHF", "AED", "SAR", "CAD", "AUD",
    "HKD", "SGD", "KWD", "QAR", "TND", "MAD", "EGP",
})

BL_CARRIER_PREFIXES: dict[str, str] = {
    "MSCU": "MSC",        "MSCL": "MSC",
    "MAEU": "Maersk",     "MRKU": "Maersk",
    "CMAU": "CMA CGM",    "CGMU": "CMA CGM",
    "HLCU": "Hapag-Lloyd",
    "ONEY": "ONE",
    "EITU": "Evergreen",  "EISU": "Evergreen",
    "OOLU": "OOCL",
    "COSU": "COSCO",      "CSNU": "COSCO",
    "YMLU": "Yang Ming",
    "ZIMU": "ZIM",
}

# Valeurs des lettres pour l'algorithme ISO 6346
_ISO_LETTER_VALUES: dict[str, int] = dict(zip(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24,
     25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38],
))

_DATE_FORMATS: list[str] = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y",
    "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%B %d, %Y", "%b %d, %Y",
]

_DATE_FIELDS_BY_TYPE: dict[str, list[str]] = {
    "bl":           ["issue_date", "on_board_date", "etd", "eta"],
    "invoice":      ["invoice_date"],
    "packing_list": ["pl_date"],
}

_WEIGHT_FIELDS:  frozenset[str] = frozenset({"gross_weight_kg", "net_weight_kg"})
_FLOAT_FIELDS:   frozenset[str] = frozenset({"measurement", "total_cbm"})

# Valeurs de mesure qui contiennent des mots non-numériques (ex: "2 × 20' GP")
_FLOAT_NON_NUMERIC_RE = re.compile(
    r"\b(?:GP|HC|OT|RF|TK|FR|PALLET|PALLETS|CONTAINER|CONT\.?|"
    r"PACKAGE|UNIT|UNITS|BAGS?|CARTON|BOX|BOXES|DRUM|DRUMS|PIECE|PIECES|"
    r"SKID|ROLL|BUNDLE|SET)\b",
    re.IGNORECASE,
)

# Valeurs de parties qui ressemblent à des labels/références (à rejeter)
_PARTY_LABEL_RE = re.compile(
    r"\b(?:INCOTERMS?|CONTRACT|INVOICE|PACKING\s+LIST|B/?L|BILL\s+OF\s+LADING|"
    r"REF(?:ERENCE)?|NO\.?|NUMBER|CODE|DATE|TERMS?|PAYMENT|DEPOSIT|"
    r"PURCHASE\s+ORDER|P\.?O\.?)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime | None:
    clean = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue
    return None


def _normalize_number(val: str) -> str | None:
    """
    Extrait et normalise la valeur numérique d'une chaîne.
    Gère : séparateurs milliers (virgule/point), MT→KG conversion.
    """
    is_mt = bool(re.search(r"\bMT\b|\bTONS?\b|\bTONNES?\b", val, re.IGNORECASE))
    m = re.search(r"[\d,.\s]+", val)
    if not m:
        return None
    num_str = m.group(0).strip()
    # Détecter le format : "1,234.56" (anglais) vs "1.234,56" (européen)
    # IMPORTANT : "67.291" est un décimal anglais, PAS un millier européen
    # → le format européen requiert obligatoirement une virgule décimale APRÈS le point-milliers
    if re.search(r"\d,\d{3}(?:\.|$)", num_str):
        # Anglais avec milliers-virgule : "1,234.56" ou "1,234"
        num_str = num_str.replace(",", "")
    elif re.search(r"\d\.\d{3},\d", num_str):
        # Européen avec milliers-point ET virgule-décimal : "1.234,56" uniquement
        num_str = num_str.replace(".", "").replace(",", ".")
    else:
        # Décimal simple ou virgule seule : "67.291", "1.5", "67,291" → normaliser
        num_str = num_str.replace(",", ".").strip()
    try:
        value = float(num_str)
    except ValueError:
        return None
    if is_mt:
        value *= 1000.0
    result = str(round(value, 3))
    # Supprimer les zéros décimaux inutiles
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def _iso6346_check(cntr: str) -> bool:
    """Valide le checksum ISO 6346 d'un numéro de conteneur (XXXX######C)."""
    s = re.sub(r"[\s\-]", "", cntr.upper())
    if not re.match(r"^[A-Z]{4}\d{7}$", s):
        return False
    total = 0
    for i, ch in enumerate(s[:10]):
        val = int(ch) if ch.isdigit() else _ISO_LETTER_VALUES.get(ch, 0)
        total += val * (2 ** i)
    check = (total % 11) % 10
    return check == int(s[10])


# ─────────────────────────────────────────────────────────────────────────────
# Moteur de règles
# ─────────────────────────────────────────────────────────────────────────────

class RulesEngine:
    """
    Moteur de règles métier LC.

    corrected_fields, results = RulesEngine().validate(fields, "bl")
    cross_results = RulesEngine().validate_cross([("bl", bl_fields), ("invoice", inv_fields)])
    """

    def __init__(self, strict: bool = False) -> None:
        # strict=True : les champs invalides sont supprimés au lieu d'être conservés
        self._strict = strict

    # ── API publique ──────────────────────────────────────────────────────────

    def validate(
        self,
        fields:   dict[str, str],
        doc_type: str,
    ) -> tuple[dict[str, str], list[RuleResult]]:
        """Applique toutes les règles au dict de champs. Retourne (corrigé, résultats)."""
        corrected = dict(fields)
        results:  list[RuleResult] = []

        # Règles universelles
        results += self._rule_normalize_weights(corrected)
        results += self._rule_normalize_floats(corrected)
        results += self._rule_validate_incoterm(corrected)
        results += self._rule_validate_currency(corrected)
        results += self._rule_normalize_freight_terms(corrected)
        results += self._rule_clean_party_fields(corrected)
        results += self._rule_deduplicate_parties(corrected)

        # Règles par type de document
        if doc_type == "bl":
            results += self._rule_rescue_vessel_from_port(corrected)
            results += self._rule_normalize_container_type(corrected)
            results += self._rule_validate_container_iso6346(corrected)
            results += self._rule_validate_bl_number(corrected)
            results += self._rule_date_ordering_bl(corrected)
            results += self._rule_normalize_dates(corrected, "bl")

        elif doc_type == "invoice":
            results += self._rule_normalize_dates(corrected, "invoice")
            results += self._rule_normalize_total_amount(corrected)

        elif doc_type == "packing_list":
            results += self._rule_validate_total_packages(corrected)
            results += self._rule_validate_total_cbm(corrected)

        summary = self.summarize(results)
        log.debug("[rules/%s] pass=%d corrected=%d warning=%d fail=%d",
                  doc_type, summary["pass"], summary["corrected"],
                  summary["warning"], summary["fail"])

        return corrected, results

    def validate_cross(
        self,
        docs:   list[tuple[str, dict[str, str]]],
        tables: dict[str, dict] | None = None,
    ) -> list[RuleResult]:
        """Cohérence inter-documents. docs = [(doc_type, fields), ...]"""
        by_type = {dt: f for dt, f in docs}
        results: list[RuleResult] = []
        results += self._rule_cross_container_number(by_type, tables or {})
        results += self._rule_cross_weight_tolerance(by_type)
        results += self._rule_cross_incoterm_consistency(by_type)
        return results

    def validate_tables(
        self,
        tables:   dict,
        doc_type: str,
    ) -> list[RuleResult]:
        """Valide les données tabulaires (conteneurs, lignes facture…)."""
        results: list[RuleResult] = []

        if doc_type == "bl":
            for i, row in enumerate(tables.get("containers", []), start=1):
                cntr = re.sub(r"[\s\-]", "", (row.get("container_number") or "").upper())
                if not cntr:
                    continue
                if _iso6346_check(cntr):
                    results.append(RuleResult(
                        rule="validate_container_iso6346",
                        field="container_number", status="pass",
                        original=cntr,
                        message=f"Conteneur {i} checksum ISO 6346 valide",
                    ))
                else:
                    results.append(RuleResult(
                        rule="validate_container_iso6346",
                        field="container_number", status="warning",
                        original=cntr,
                        message=f"Conteneur {i} '{cntr}' : checksum ISO 6346 invalide",
                    ))

        elif doc_type == "invoice":
            for i, row in enumerate(tables.get("invoice_items", []), start=1):
                try:
                    qty  = float(row.get("quantity",   "0").replace(",", ".").replace(" ", "") or "0")
                    prc  = float(row.get("unit_price", "0").replace(",", ".").replace(" ", "") or "0")
                    amt  = float(row.get("line_amount","0").replace(",", ".").replace(" ", "") or "0")
                    if qty > 0 and prc > 0 and amt > 0:
                        expected = round(qty * prc, 2)
                        diff_pct = abs(expected - amt) / max(amt, 0.01) * 100
                        status = "pass" if diff_pct <= 2.0 else "warning"
                        results.append(RuleResult(
                            rule="validate_invoice_line_amount",
                            field="line_amount", status=status,
                            message=(
                                f"Ligne {i}: montant cohérent ({amt})"
                                if status == "pass"
                                else f"Ligne {i}: montant {amt} ≠ qté×prix {expected} (écart {diff_pct:.1f}%)"
                            ),
                        ))
                except (ValueError, ZeroDivisionError):
                    pass

        return results

    # ── Règles universelles ───────────────────────────────────────────────────

    def _rule_normalize_weights(self, fields: dict) -> list[RuleResult]:
        results = []
        # sorted() : l'ordre d'itération d'un frozenset dépend du hash aléatoire
        # du processus → l'ordre des rule_results variait entre deux serveurs.
        for f in sorted(_WEIGHT_FIELDS):
            v = fields.get(f)
            if not v:
                continue
            norm = _normalize_number(v)
            if norm is None:
                results.append(RuleResult(
                    rule="normalize_weight", field=f, status="warning",
                    original=v, message=f"Impossible de parser le poids : '{v}'",
                ))
            elif norm != v:
                fields[f] = norm
                results.append(RuleResult(
                    rule="normalize_weight", field=f, status="corrected",
                    original=v, corrected=norm,
                    message=f"Poids normalisé : '{v}' → '{norm}'",
                ))
            else:
                results.append(RuleResult(
                    rule="normalize_weight", field=f, status="pass", original=v,
                ))
        return results

    def _rule_normalize_floats(self, fields: dict) -> list[RuleResult]:
        results = []
        for f in sorted(_FLOAT_FIELDS):  # ordre déterministe (voir _WEIGHT_FIELDS)
            v = fields.get(f)
            if not v:
                continue
            # Rejeter si la valeur contient des mots de description (ex: "2 × 20' GP")
            if _FLOAT_NON_NUMERIC_RE.search(v):
                del fields[f]
                results.append(RuleResult(
                    rule="normalize_float", field=f, status="corrected",
                    original=v, corrected=None,
                    message=f"Valeur non-numérique ignorée : '{v}'",
                ))
                continue
            norm = _normalize_number(v)
            if norm and norm != v:
                fields[f] = norm
                results.append(RuleResult(
                    rule="normalize_float", field=f, status="corrected",
                    original=v, corrected=norm,
                ))
            elif norm:
                results.append(RuleResult(
                    rule="normalize_float", field=f, status="pass", original=v,
                ))
        return results

    def _rule_validate_incoterm(self, fields: dict) -> list[RuleResult]:
        v = fields.get("incoterm")
        if not v:
            return []
        upper = v.strip().upper()
        if upper in VALID_INCOTERMS:
            if upper != v:
                fields["incoterm"] = upper
                return [RuleResult(rule="validate_incoterm", field="incoterm",
                                   status="corrected", original=v, corrected=upper)]
            return [RuleResult(rule="validate_incoterm", field="incoterm",
                               status="pass", original=v)]
        # Incoterm invalide : toujours supprimer (valeur inutilisable)
        del fields["incoterm"]
        return [RuleResult(
            rule="validate_incoterm", field="incoterm", status="fail",
            original=v,
            message=f"Incoterm invalide supprimé : '{v}'. Acceptés : {sorted(VALID_INCOTERMS)}",
        )]

    def _rule_validate_currency(self, fields: dict) -> list[RuleResult]:
        v = fields.get("currency")
        if not v:
            return []
        upper = v.strip().upper()
        if upper in VALID_CURRENCIES:
            if upper != v:
                fields["currency"] = upper
                return [RuleResult(rule="validate_currency", field="currency",
                                   status="corrected", original=v, corrected=upper)]
            return [RuleResult(rule="validate_currency", field="currency",
                               status="pass", original=v)]
        return [RuleResult(
            rule="validate_currency", field="currency", status="warning",
            original=v, message=f"Devise inconnue : '{v}' — conservée",
        )]

    def _rule_clean_party_fields(self, fields: dict) -> list[RuleResult]:
        """
        Rejette les valeurs de parties qui ressemblent à des labels ou des références.
        Ex : "Contract No.", "Invoice Ref.", "No." → supprimés.
        """
        results: list[RuleResult] = []
        for pfield in ("shipper", "consignee", "notify_party", "seller", "buyer"):
            val = fields.get(pfield, "")
            if not val:
                continue
            stripped = val.strip()
            # Trop court pour être un nom de société (< 4 caractères)
            if len(stripped) < 4:
                del fields[pfield]
                results.append(RuleResult(
                    rule="clean_party_fields", field=pfield,
                    status="corrected", original=val, corrected=None,
                    message=f"{pfield} trop court — supprimé",
                ))
                continue
            # Ressemble à un label/référence (ex: "Contract No.", "Invoice Ref.")
            # Seulement si la valeur est courte (< 30 chars) — évite les faux positifs
            if len(stripped) < 30 and _PARTY_LABEL_RE.search(stripped):
                del fields[pfield]
                results.append(RuleResult(
                    rule="clean_party_fields", field=pfield,
                    status="corrected", original=val, corrected=None,
                    message=f"{pfield} ressemble à un label/référence — supprimé",
                ))
        return results

    def _rule_deduplicate_parties(self, fields: dict) -> list[RuleResult]:
        """
        Corrige le cas où consignee == shipper (double-capture d'un seul bloc partie).

        Logique :
          - Si notify_party est différent → consignee ← notify_party
          - Sinon pour invoice : buyer == seller → même correction
          - Si aucune alternative : supprime le doublon (vaut mieux "manquant" que "faux")
        """
        results: list[RuleResult] = []

        for seller_f, buyer_f, alt_f in [
            ("shipper",  "consignee", "notify_party"),
            ("seller",   "buyer",     "notify_party"),
        ]:
            seller_v = fields.get(seller_f, "")
            buyer_v  = fields.get(buyer_f,  "")
            if not seller_v or not buyer_v or seller_v != buyer_v:
                continue

            alt_v = fields.get(alt_f, "")
            if alt_v and alt_v != seller_v:
                fields[buyer_f] = alt_v
                results.append(RuleResult(
                    rule="deduplicate_parties", field=buyer_f,
                    status="corrected", original=buyer_v, corrected=alt_v,
                    message=f"{buyer_f} == {seller_f} — remplacé par {alt_f}",
                ))
            else:
                del fields[buyer_f]
                results.append(RuleResult(
                    rule="deduplicate_parties", field=buyer_f,
                    status="warning", original=buyer_v,
                    message=f"{buyer_f} identique au {seller_f} — valeur supprimée",
                ))

        return results

    _CARRIER_PREFIX_RE = re.compile(
        # "HAMBURG" seul est aussi un nom de ville → n'accepter que "HAMBURG SUD"
        r"^(?:MSC|MAERSK|CMA(?:\s+CGM)?|HAPAG|EVERGREEN|ONE\b|"
        r"COSCO|YANG\s+MING|ZIM|HAMBURG\s+SUD|OOCL|APL|PIL|MOL|NYK|HMM|SITC|CSCL|WHL)\b",
        re.IGNORECASE,
    )

    def _rule_rescue_vessel_from_port(self, fields: dict) -> list[RuleResult]:
        """
        Quand un champ port/lieu contient un nom de navire (préfixe armateur connu),
        récupère la valeur dans 'vessel' si ce champ est vide, puis vide le champ port.
        Cas typique : PyMuPDF lit 'PORT OF DISCHARGE\\nMSC GULSUN' alors que 'MSC GULSUN'
        est en réalité sous l'étiquette VESSEL dans une grille multi-colonnes.
        """
        results: list[RuleResult] = []
        for port_field in ("port_of_discharge", "place_of_delivery", "place_of_receipt", "port_of_loading"):
            val = fields.get(port_field, "")
            if not val or not self._CARRIER_PREFIX_RE.match(val):
                continue
            if not fields.get("vessel"):
                fields["vessel"] = val
                results.append(RuleResult(
                    rule="rescue_vessel_from_port", field="vessel",
                    status="corrected", original=None, corrected=val,
                    message=f"vessel récupéré depuis {port_field} ('{val}')",
                ))
            del fields[port_field]
            results.append(RuleResult(
                rule="rescue_vessel_from_port", field=port_field,
                status="corrected", original=val, corrected=None,
                message=f"{port_field} contenait un nom de navire — champ vidé",
            ))
        return results

    def _rule_normalize_freight_terms(self, fields: dict) -> list[RuleResult]:
        v = fields.get("freight_terms")
        if not v:
            return []
        upper = v.strip().upper()
        if "PREPAID" in upper:
            norm = "Freight Prepaid"
        elif "COLLECT" in upper:
            norm = "Freight Collect"
        else:
            return [RuleResult(rule="normalize_freight_terms",
                               field="freight_terms", status="pass", original=v)]
        if norm != v:
            fields["freight_terms"] = norm
            return [RuleResult(rule="normalize_freight_terms",
                               field="freight_terms", status="corrected",
                               original=v, corrected=norm)]
        return [RuleResult(rule="normalize_freight_terms",
                           field="freight_terms", status="pass", original=v)]

    # ── Règles BL ─────────────────────────────────────────────────────────────

    def _rule_validate_container_iso6346(self, fields: dict) -> list[RuleResult]:
        v = fields.get("container_number")
        if not v:
            return []
        clean = re.sub(r"[\s\-]", "", v.upper())
        if _iso6346_check(clean):
            if clean != v:
                fields["container_number"] = clean
            return [RuleResult(rule="validate_container_iso6346",
                               field="container_number", status="pass",
                               original=v, message="Checksum ISO 6346 valide")]
        return [RuleResult(
            rule="validate_container_iso6346", field="container_number",
            status="warning", original=v,
            message=f"Checksum ISO 6346 invalide pour '{v}' — valeur conservée",
        )]

    def _rule_validate_bl_number(self, fields: dict) -> list[RuleResult]:
        v = fields.get("bl_number")
        if not v:
            return []
        m = re.match(r"^([A-Z]{4})", v.upper().strip())
        prefix = m.group(1) if m else None
        if prefix and prefix in BL_CARRIER_PREFIXES:
            carrier_name = BL_CARRIER_PREFIXES[prefix]
            # Override carrier if absent or if extracted value is suspiciously long / noisy
            existing = fields.get("carrier", "")
            if not existing or len(existing) > 40 or "|" in existing or "Booking" in existing:
                fields["carrier"] = carrier_name
            return [RuleResult(rule="validate_bl_number", field="bl_number",
                               status="pass", original=v,
                               message=f"Transporteur : {carrier_name}")]
        return [RuleResult(
            rule="validate_bl_number", field="bl_number", status="warning",
            original=v, message=f"Préfixe transporteur inconnu dans '{v}'",
        )]

    def _rule_date_ordering_bl(self, fields: dict) -> list[RuleResult]:
        dates: dict[str, datetime] = {}
        for f in ["issue_date", "etd", "eta"]:
            v = fields.get(f)
            if v:
                d = _parse_date(v)
                if d:
                    dates[f] = d
        if len(dates) < 2:
            return []
        results = []
        for before, after, desc in [
            ("issue_date", "etd", "issue_date ≤ ETD"),
            ("etd",        "eta", "ETD ≤ ETA"),
            ("issue_date", "eta", "issue_date ≤ ETA"),
        ]:
            if before in dates and after in dates:
                status = "pass" if dates[before] <= dates[after] else "warning"
                msg = (
                    f"Ordre correct : {desc}" if status == "pass"
                    else f"Ordre incohérent : {desc} ({fields.get(before)} > {fields.get(after)})"
                )
                results.append(RuleResult(
                    rule="date_ordering", field=None, status=status, message=msg,
                ))
        return results

    # Tokens that get physically truncated in narrow PDF columns → full form
    _CONTAINER_TYPE_EXPAND: dict[str, str] = {
        "HIGH CUB":  "HIGH CUBE",
        "HIGH CU":   "HIGH CUBE",
        "FLAT RAC":  "FLAT RACK",
        "FLAT RA":   "FLAT RACK",
        "OPEN TO":   "OPEN TOP",
        "OPEN T":    "OPEN TOP",
        "REEFER":    "REEFER",      # already correct — no-op entry for safety
        "REFRIG":    "REEFER",
    }

    def _rule_normalize_container_type(self, fields: dict) -> list[RuleResult]:
        v = fields.get("container_type")
        if not v:
            return []
        normalized = v
        for truncated, full in self._CONTAINER_TYPE_EXPAND.items():
            if truncated in normalized.upper() and full not in normalized.upper():
                normalized = re.sub(re.escape(truncated), full, normalized, flags=re.IGNORECASE)
                break
        if normalized == v:
            return [RuleResult(rule="normalize_container_type", field="container_type",
                               status="pass", original=v)]
        fields["container_type"] = normalized
        return [RuleResult(rule="normalize_container_type", field="container_type",
                           status="corrected", original=v, corrected=normalized)]

    def _rule_normalize_dates(self, fields: dict, doc_type: str) -> list[RuleResult]:
        results = []
        for f in _DATE_FIELDS_BY_TYPE.get(doc_type, []):
            v = fields.get(f)
            if not v:
                continue
            d = _parse_date(v)
            if d:
                norm = d.strftime("%Y-%m-%d")
                if norm != v:
                    fields[f] = norm
                    results.append(RuleResult(rule="normalize_date", field=f,
                                              status="corrected", original=v, corrected=norm))
                else:
                    results.append(RuleResult(rule="normalize_date", field=f,
                                              status="pass", original=v))
            else:
                results.append(RuleResult(
                    rule="normalize_date", field=f, status="warning", original=v,
                    message=f"Format de date non reconnu : '{v}'",
                ))
        return results

    # ── Règles Invoice ────────────────────────────────────────────────────────

    def _rule_normalize_total_amount(self, fields: dict) -> list[RuleResult]:
        v = fields.get("total_amount")
        if not v:
            return []
        currency_m = re.search(r"\b([A-Z]{3})\b", v)
        detected   = currency_m.group(1) if currency_m else None

        # Extraire et injecter currency si absente
        if not fields.get("currency") and detected and detected in VALID_CURRENCIES:
            fields["currency"] = detected

        # Format déjà propre : "USD 21,904.00" ou "EUR 11,698.50"
        # → ne pas passer par _normalize_number qui écrase les décimales
        if re.match(r'^[A-Z]{3}\s+[\d,]+\.\d{2}$', v.strip()):
            return [RuleResult(rule="normalize_total_amount", field="total_amount",
                               status="pass", original=v)]

        norm_num = _normalize_number(v)
        if not norm_num:
            return [RuleResult(rule="normalize_total_amount", field="total_amount",
                               status="warning", original=v,
                               message="Impossible de normaliser le montant")]
        norm = f"{detected} {norm_num}" if detected and detected in VALID_CURRENCIES else norm_num
        if norm != v:
            fields["total_amount"] = norm
            return [RuleResult(rule="normalize_total_amount", field="total_amount",
                               status="corrected", original=v, corrected=norm)]
        return [RuleResult(rule="normalize_total_amount", field="total_amount",
                           status="pass", original=v)]

    # ── Règles PL ─────────────────────────────────────────────────────────────

    def _rule_validate_total_packages(self, fields: dict) -> list[RuleResult]:
        v = fields.get("total_packages")
        if not v:
            return []
        m = re.search(r"\d+", v)
        if m and int(m.group(0)) > 0:
            return [RuleResult(rule="validate_total_packages",
                               field="total_packages", status="pass", original=v,
                               message=f"Colis : {m.group(0)}")]
        return [RuleResult(rule="validate_total_packages", field="total_packages",
                           status="warning", original=v,
                           message=f"Valeur suspecte : '{v}'")]

    def _rule_validate_total_cbm(self, fields: dict) -> list[RuleResult]:
        v = fields.get("total_cbm")
        if not v:
            return []
        norm = _normalize_number(v)
        if norm:
            try:
                val = float(norm)
                if 0 < val < 5000:
                    if norm != v:
                        fields["total_cbm"] = norm
                        return [RuleResult(rule="validate_total_cbm",
                                           field="total_cbm", status="corrected",
                                           original=v, corrected=norm)]
                    return [RuleResult(rule="validate_total_cbm",
                                       field="total_cbm", status="pass", original=v)]
                return [RuleResult(rule="validate_total_cbm", field="total_cbm",
                                   status="warning", original=v,
                                   message=f"CBM hors plage [0.1, 5000] : {val}")]
            except ValueError:
                pass
        return [RuleResult(rule="validate_total_cbm", field="total_cbm",
                           status="warning", original=v,
                           message=f"Impossible de parser le CBM : '{v}'")]

    # ── Règles inter-documents ────────────────────────────────────────────────

    def _rule_cross_container_number(
        self, by_type: dict, tables: dict | None = None
    ) -> list[RuleResult]:
        # Collect container sets per doc_type from scalar fields AND tables
        cntrs_by_type: dict[str, set[str]] = {}
        for dt, f in by_type.items():
            v = f.get("container_number")
            if v:
                cntrs_by_type.setdefault(dt, set()).add(re.sub(r"[\s\-]", "", v.upper()))

        if tables:
            for dt, tbl in tables.items():
                if dt == "bl":
                    for row in tbl.get("containers", []):
                        c = re.sub(r"[\s\-]", "", (row.get("container_number") or "").upper())
                        if c:
                            cntrs_by_type.setdefault(dt, set()).add(c)
                elif dt == "packing_list":
                    for row in tbl.get("packages", []):
                        c = re.sub(r"[\s\-]", "", (row.get("container_number") or "").upper())
                        if c:
                            cntrs_by_type.setdefault(dt, set()).add(c)

        if len(cntrs_by_type) < 2:
            return []

        # All docs must share at least one container number with BL
        bl_cntrs = cntrs_by_type.get("bl", set())
        if not bl_cntrs:
            # No BL — check all docs share same containers
            all_sets = list(cntrs_by_type.values())
            common = all_sets[0].intersection(*all_sets[1:])
            if common:
                return [RuleResult(rule="cross_container_number",
                                   field="container_number", status="pass",
                                   message=f"Conteneurs cohérents : {sorted(common)[:3]}")]
            return [RuleResult(
                rule="cross_container_number", field="container_number", status="fail",
                message="Conteneurs incohérents : " +
                        ", ".join(f"{dt}={sorted(v)[:2]}" for dt, v in cntrs_by_type.items()),
            )]

        mismatches = []
        for dt, cntrs in cntrs_by_type.items():
            if dt == "bl":
                continue
            if not (cntrs & bl_cntrs):
                mismatches.append(f"{dt}={sorted(cntrs)[:2]} vs BL={sorted(bl_cntrs)[:2]}")

        if not mismatches:
            return [RuleResult(rule="cross_container_number",
                               field="container_number", status="pass",
                               message=f"Conteneurs cohérents ({sorted(bl_cntrs)[:2]})")]
        return [RuleResult(
            rule="cross_container_number", field="container_number", status="fail",
            message="Conteneur incohérent entre documents : " + "; ".join(mismatches),
        )]

    def _rule_cross_weight_tolerance(self, by_type: dict) -> list[RuleResult]:
        weights: dict[str, float] = {}
        for dt in ["bl", "packing_list"]:
            v = by_type.get(dt, {}).get("gross_weight_kg")
            if v:
                n = _normalize_number(v)
                if n:
                    try:
                        weights[dt] = float(n)
                    except ValueError:
                        pass
        if len(weights) < 2:
            return []
        w_bl, w_pl = weights.get("bl", 0), weights.get("packing_list", 0)
        if not w_bl or not w_pl:
            return []
        diff_pct = abs(w_bl - w_pl) / max(w_bl, w_pl) * 100
        if diff_pct <= 5.0:
            return [RuleResult(rule="cross_weight_tolerance",
                               field="gross_weight_kg", status="pass",
                               message=f"Poids cohérent BL↔PL (écart {diff_pct:.1f}%)")]
        return [RuleResult(
            rule="cross_weight_tolerance", field="gross_weight_kg", status="warning",
            message=f"Poids BL ({w_bl} kg) et PL ({w_pl} kg) divergent de {diff_pct:.1f}% (seuil 5%)",
        )]

    def _rule_cross_incoterm_consistency(self, by_type: dict) -> list[RuleResult]:
        incoterms = {dt: f.get("incoterm")
                     for dt, f in by_type.items() if f.get("incoterm")}
        if len(incoterms) < 2:
            return []
        if len(set(incoterms.values())) == 1:
            return [RuleResult(rule="cross_incoterm_consistency",
                               field="incoterm", status="pass",
                               message=f"Incoterm cohérent ({next(iter(incoterms.values()))})")]
        return [RuleResult(
            rule="cross_incoterm_consistency", field="incoterm", status="warning",
            message="Incoterm différent entre documents : " +
                    ", ".join(f"{dt}={v}" for dt, v in incoterms.items()),
        )]

    # ── Utilitaire ────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(results: list[RuleResult]) -> dict:
        counts: dict[str, int] = {"pass": 0, "corrected": 0, "fail": 0,
                                   "warning": 0, "skipped": 0}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        total = sum(counts.values())
        return {
            **counts,
            "total":     total,
            "pass_rate": counts["pass"] / total if total else 0.0,
            "fail_rate": (counts["fail"] + counts["warning"]) / total if total else 0.0,
        }
