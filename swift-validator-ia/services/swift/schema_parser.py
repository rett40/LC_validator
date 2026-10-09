"""
schema_parser.py  —  v5
Parseur générique basé sur les schémas JSON.
Améliorations v5 :
- Mapping complet de tous les champs vers SwiftEntities
- Tolérance OCR accrue sur les séparateurs de tags
- Support des tags multi-variantes (41A/41D, 52A/52D…)
- Parsing typé intégré (dates, montants, séquences)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Tuple

from .entities import SwiftEntities


# ---------------------------------------------------------------------------
# Utilitaires de normalisation
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Nettoyage OCR général avant parsing."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Séparateurs mal lus
    t = t.replace("; ", ": ").replace(";:", ":").replace(" ;", " :")

    # Confusions OCR sur le type MT
    for wrong, right in [
        ("MT7OO", "MT700"), ("MT72O", "MT720"), ("MT76O", "MT760"),
        ("mt7oo", "MT700"), ("mt72o", "MT720"),
    ]:
        t = t.replace(wrong, right)

    # Doubles ':' résiduels (: :, ::)
    t = re.sub(r':\s*:', ':', t)

    # ':' orphelins après un tag (":40E: : UCP" → ":40E: UCP")
    t = re.sub(r'(:\d{2}[A-Z]?:)\s*:\s*', r'\1 ', t)

    return t


def yymmdd_to_iso(value: str) -> str | None:
    """Convertit YYMMDD en ISO YYYY-MM-DD."""
    v = value.strip()
    if len(v) < 6 or not v[:6].isdigit():
        return None
    yy, mm, dd = int(v[:2]), v[2:4], v[4:6]
    year = 2000 + yy if yy < 70 else 1900 + yy
    try:
        return datetime(year, int(mm), int(dd)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_currency_amount(value: str) -> Tuple[str | None, str | None]:
    """Extrait devise ISO (3 lettres) + montant numérique."""
    m = re.match(r"\s*([A-Z]{3})\s*([0-9][0-9\.,_ \']*)", value.strip())
    if not m:
        return None, None
    ccy = m.group(1)
    amt = m.group(2).replace("'", "").replace("_", "").replace(" ", "")
    if amt.count(",") == 1 and "." not in amt:
        amt = amt.replace(",", ".")
    else:
        amt = amt.replace(",", "")
    return ccy, amt


def parse_value_date_currency_amount(value: str) -> Tuple[str | None, str | None, str | None]:
    """Pour :32A: : YYMMDD + devise + montant."""
    m = re.match(r"\s*(\d{6})([A-Z]{3})([0-9][0-9\.,_ \']*)", value.strip())
    if not m:
        return None, None, None
    return yymmdd_to_iso(m.group(1)), *parse_currency_amount(m.group(2) + m.group(3))


def tag_matches(field_code: str, actual_tag: str) -> bool:
    """
    Gère les notations famille : '41a' → 41A/41D/41C…
    Sinon égalité stricte.
    """
    fc = field_code.upper()
    at = actual_tag.upper()
    if field_code[-1].islower() and fc[-1] in "ABCD":
        return at.startswith(fc[:-1]) and len(at) == len(fc)
    return fc == at


# ---------------------------------------------------------------------------
# Parseur principal
# ---------------------------------------------------------------------------

class SwiftSchemaParser:
    """
    Parseur générique basé sur les schémas JSON.
    Extrait les champs et les mappe vers SwiftEntities.
    """

    def __init__(self, schemas: Dict[str, Dict]) -> None:
        self.schemas = schemas

    def parse(self, text: str, swift_type: str) -> SwiftEntities:
        schema   = self.schemas.get(swift_type, {})
        entities = SwiftEntities(swift_type=swift_type)
        cleaned  = normalize_text(text)

        # ── Extraction des champs bruts ──────────────────────────────
        raw_fields = self._extract_raw_fields(cleaned, schema)
        entities.raw_fields = raw_fields

        # ── Mapping vers les attributs SwiftEntities ─────────────────
        self._map_fields(entities, raw_fields)

        # ── Validation des champs obligatoires ───────────────────────
        self._check_mandatory(entities, schema, raw_fields)

        return entities

    # ------------------------------------------------------------------
    # Extraction des champs bruts
    # ------------------------------------------------------------------

    def _extract_raw_fields(self, text: str, schema: dict) -> dict[str, str]:
        """
        Parcourt le texte ligne par ligne et collecte les champs tag→valeur.
        Tolère : ':27:' / '27:' / ':27 :' / '; 27 :' etc.
        """
        expected = list(schema.get("fields", {}).keys())

        # Regex principale : accepte : ; . optionnels autour du tag
        tag_re = re.compile(
            r"^\s*[:;.]?\s*(?P<tag>[0-9]{2}[A-Z]?)\s*[:;.]\s*(?P<rest>.*)",
            re.IGNORECASE,
        )
        # Fallback : tag sans séparateur (ex "20 LC2026")
        fallback_re = re.compile(
            r"^\s*(?P<tag>[0-9]{2}[A-Z]?)\s+(?P<rest>\S.+)",
            re.IGNORECASE,
        )

        raw: dict[str, str] = {}
        current_tag: str | None = None
        buffer: list[str] = []

        def flush():
            if current_tag is None:
                return
            val = "\n".join(buffer).strip()
            if val:
                raw[current_tag] = val

        for line in text.splitlines():
            m = tag_re.match(line)

            if not m:
                fb = fallback_re.match(line)
                if fb:
                    candidate = fb.group("tag").upper()
                    # Vérifier si ce tag fait partie du schéma attendu
                    if any(tag_matches(fc, candidate) for fc in expected):
                        m = fb

            if m:
                flush()
                tag = self._normalize_tag(m.group("tag").upper())
                current_tag = tag
                buffer = [m.group("rest").strip()]
            else:
                if current_tag is not None:
                    buffer.append(line)

        flush()
        return raw

    # ------------------------------------------------------------------
    # Mapping vers SwiftEntities
    # ------------------------------------------------------------------

    def _map_fields(self, e: SwiftEntities, raw: dict[str, str]) -> None:
        """Mappe tous les champs raw → attributs SwiftEntities."""

        for tag, value in raw.items():
            v = value.strip()

            # ── Références ──────────────────────────────────────────
            if tag == "20":
                e.lc_number = v
            elif tag == "21":
                e.related_lc = v
            elif tag == "27":
                e.sequence = v

            # ── Dates ───────────────────────────────────────────────
            elif tag == "31C":
                e.issue_date = yymmdd_to_iso(v[:6]) or v
            elif tag == "30":
                e.issue_date = yymmdd_to_iso(v[:6]) or v
            elif tag == "31D":
                e.expiry_date  = yymmdd_to_iso(v[:6]) or v[:6]
                e.expiry_place = v[6:].strip() if len(v) > 6 else None
            elif tag == "31E":
                e.expiry_date = yymmdd_to_iso(v[:6]) or v
            elif tag == "44C":
                e.shipment_date = yymmdd_to_iso(v[:6]) or v

            # ── Montants ────────────────────────────────────────────
            elif tag in ("32B", "33B"):
                ccy, amt = parse_currency_amount(v)
                e.currency = e.currency or ccy
                e.amount   = e.amount   or amt
            elif tag == "32A":
                iso, ccy, amt = parse_value_date_currency_amount(v)
                e.value_date = e.value_date or iso
                e.currency   = e.currency   or ccy
                e.amount     = e.amount     or amt
            elif tag == "39A":
                m = re.match(r"(\d+)/(\d+)", v)
                if m:
                    e.tolerance_plus  = m.group(1)
                    e.tolerance_minus = m.group(2)

            # ── Parties ─────────────────────────────────────────────
            elif tag.startswith("50"):
                e.applicant = e.applicant or v
            elif tag.startswith("59"):
                e.beneficiary = e.beneficiary or v
            elif tag.startswith("52"):
                e.issuing_bank = e.issuing_bank or v
            elif tag.startswith("41"):
                e.available_with = e.available_with or v

            # ── Conditions LC ────────────────────────────────────────
            elif tag in ("40A", "40B"):
                e.lc_form = v
            elif tag in ("40E", "40C"):
                e.applicable_rules = v
            elif tag == "42C":
                e.draft_at = v
            elif tag == "43P":
                e.partial_shipments = v
            elif tag == "43T":
                e.transhipment = v
            elif tag in ("44A", "44E"):
                e.port_loading = e.port_loading or v
            elif tag in ("44B", "44F"):
                e.port_discharge = e.port_discharge or v
            elif tag == "45A":
                e.goods_description = v
            elif tag == "46A":
                e.documents_required = v
            elif tag == "47A":
                e.additional_conditions = v
            elif tag in ("71B", "71D"):
                e.charges = v
            elif tag == "48":
                e.presentation_period = v
            elif tag == "49":
                e.confirmation = v

            # ── MT760 spécifique ────────────────────────────────────
            elif tag == "22A":
                e.purpose = v
            elif tag == "22D":
                e.undertaking_type = v
            elif tag == "23B":
                e.expiry_type = v
            elif tag == "77U":
                e.guarantee_text = v

            # ── MT790 spécifique ────────────────────────────────────
            elif tag == "25":
                e.account = v

            # ── MT799 spécifique ────────────────────────────────────
            elif tag == "79":
                e.free_text  = v
                e.message    = v

            # ── Commun ──────────────────────────────────────────────
            elif tag == "72":
                e.sender_info = v
            elif tag in ("77A",):
                e.message = (e.message + "\n" + v) if e.message else v

    # ------------------------------------------------------------------
    # Validation des champs obligatoires
    # ------------------------------------------------------------------

    def _check_mandatory(self, e: SwiftEntities, schema: dict, raw: dict) -> None:
        missing = []
        for fc, meta in schema.get("fields", {}).items():
            if not meta.get("mandatory"):
                continue
            if not any(tag_matches(fc, t) for t in raw):
                missing.append(fc)
        if missing:
            e.warnings.append(f"Champs obligatoires manquants: {', '.join(sorted(missing))}")

    # ------------------------------------------------------------------
    # Normalisation du tag
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_tag(tag: str) -> str:
        """Corrige les confusions OCR dans le tag lui-même (O→0)."""
        chars = list(tag)
        # Les deux premiers caractères d'un tag sont toujours des chiffres
        if len(chars) >= 1 and chars[0] == "O":
            chars[0] = "0"
        if len(chars) >= 2 and chars[1] == "O":
            chars[1] = "0"
        return "".join(chars)