"""
cross_validator/rules_engine.py
================================
Moteur de règles inter-documents (BL ↔ Invoice ↔ Packing List).

Chaque règle compare des champs ou des tables entre deux ou trois documents
et retourne un RuleResult avec statut pass / fail / warning / skipped.

Règles implémentées :
  cross_container_number     — numéro conteneur cohérent BL ↔ PL
  cross_seal_number          — numéro de scellé BL ↔ PL
  cross_weight_tolerance     — poids brut BL ↔ PL (tolérance ±10 %)
  cross_net_weight           — poids net BL ↔ PL (tolérance ±10 %)
  cross_incoterm_consistency — incoterm cohérent Invoice ↔ BL
  cross_currency             — devise cohérente tous documents
  cross_shipper_seller       — shipper (BL) ≈ seller (Invoice)
  cross_consignee_buyer      — consignee (BL) ≈ buyer (Invoice/PL)
  cross_goods_description    — descriptions marchandises alignées
  cross_hs_code_consistency  — codes SH cohérents Invoice ↔ PL
  cross_invoice_pl_qty       — quantités invoice ↔ packing list
  cross_total_weight_logic   — poids net ≤ poids brut (PL & BL)
  cross_voyage_lc_ref        — référence LC invoice ↔ voyage BL
  cross_date_ordering        — issue_date (Invoice) ≤ ETA (BL)
"""

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)

RuleStatus = Literal["pass", "fail", "corrected", "warning", "skipped"]


@dataclass
class RuleResult:
    rule: str
    field: str | None
    status: RuleStatus
    original: str | None = None
    corrected: str | None = None
    message: str = ""
    severity: str = "info"        # info | low | medium | high | critical
    doc_pair: str | None = None   # ex: "bl↔packing_list"

    def to_dict(self) -> dict:
        return {
            "rule":      self.rule,
            "field":     self.field,
            "status":    self.status,
            "original":  self.original,
            "corrected": self.corrected,
            "message":   self.message,
            "severity":  self.severity,
            "doc_pair":  self.doc_pair,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires internes
# ─────────────────────────────────────────────────────────────────────────────

def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(" ", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        cleaned = re.sub(r"[^0-9.\-]", "", text)
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None


def _normalize_str(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]+", " ", str(text or "")).strip().lower()


def _string_similarity(a: str, b: str) -> float:
    """Similarité de Sørensen–Dice sur tokens."""
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz.fuzz import token_set_ratio
        return token_set_ratio(a, b) / 100.0
    except ImportError:
        pass
    # Fallback SequenceMatcher
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _extract_container_numbers(value: str) -> list[str]:
    """Extrait tous les numéros ISO 6346 depuis une chaîne."""
    return re.findall(r"[A-Z]{3,4}[UJZ]\d{6,7}", str(value or "").upper())


def _normalize_container_list(docs_by_type: dict, doc_key: str, field: str) -> list[str]:
    fields = docs_by_type.get(doc_key, {})
    raw = str(fields.get(field, "") or "")
    # Essaie extraction ISO
    found = _extract_container_numbers(raw)
    if found:
        return found
    # Fallback : split sur virgule/espace/point-virgule
    return [t.strip().upper() for t in re.split(r"[,;\s/]+", raw) if t.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Moteur de règles inter-documents
# ─────────────────────────────────────────────────────────────────────────────

class RulesEngine:

    WEIGHT_TOLERANCE = 0.10    # 10 % écart autorisé
    PARTY_SIM_THRESHOLD = 0.60  # seuil similarité parties
    GOODS_SIM_THRESHOLD = 0.55  # seuil similarité description marchandises

    _PAIR_RULES: dict[str, list[str]] = {
        "bl↔invoice": [
            "incoterm_consistency",
            "currency",
            "shipper_seller",
            "consignee_buyer",
            "voyage_lc_ref",
            "date_ordering",
            "port_discharge_match",
        ],
        "invoice↔packing_list": [
            "goods_description",
            "hs_code_consistency",
            "invoice_pl_qty",
            "line_amount_plausibility",
            "weight_inv_pl_net",
        ],
        "bl↔packing_list": [
            "container_number",
            "seal_number",
            "weight_tolerance",
            "net_weight",
            "total_weight_logic",
        ],
    }

    def validate_pair(
        self,
        pair: str,
        by_type: dict[str, dict[str, Any]],
        tables: dict[str, dict[str, Any]],
    ) -> list[RuleResult]:
        """Runs only rules applicable to the given document pair."""
        _dispatch: dict[str, Any] = {
            "container_number":         lambda: self._rule_container_number(by_type, tables),
            "seal_number":              lambda: self._rule_seal_number(by_type, tables),
            "weight_tolerance":         lambda: self._rule_weight_tolerance(by_type, tables),
            "net_weight":               lambda: self._rule_net_weight(by_type, tables),
            "incoterm_consistency":     lambda: self._rule_incoterm_consistency(by_type),
            "currency":                 lambda: self._rule_currency(by_type),
            "shipper_seller":           lambda: self._rule_shipper_seller(by_type),
            "consignee_buyer":          lambda: self._rule_consignee_buyer(by_type),
            "goods_description":        lambda: self._rule_goods_description(by_type, tables),
            "hs_code_consistency":      lambda: self._rule_hs_code_consistency(tables),
            "invoice_pl_qty":           lambda: self._rule_invoice_pl_qty(tables),
            "total_weight_logic":       lambda: self._rule_total_weight_logic(by_type, tables),
            "voyage_lc_ref":            lambda: self._rule_voyage_lc_ref(by_type),
            "date_ordering":            lambda: self._rule_date_ordering(by_type),
            "port_discharge_match":     lambda: self._rule_port_discharge_match(by_type),
            "line_amount_plausibility": lambda: self._rule_line_amount_plausibility(tables),
            "weight_inv_pl_net":        lambda: self._rule_weight_inv_pl_net(by_type, tables),
        }
        results: list[RuleResult] = []
        for name in self._PAIR_RULES.get(pair, []):
            fn = _dispatch.get(name)
            if fn:
                results += fn()
        return results

    def validate_cross(
        self,
        docs: list[tuple[str, dict[str, str]]],
        tables: dict[str, dict[str, Any]] | None = None,
    ) -> list[RuleResult]:
        tables = tables or {}
        by_type: dict[str, dict[str, str]] = {dt: f for dt, f in docs}

        results: list[RuleResult] = []
        results += self._rule_container_number(by_type, tables)
        results += self._rule_seal_number(by_type, tables)
        results += self._rule_weight_tolerance(by_type, tables)
        results += self._rule_net_weight(by_type, tables)
        results += self._rule_incoterm_consistency(by_type)
        results += self._rule_currency(by_type)
        results += self._rule_shipper_seller(by_type)
        results += self._rule_consignee_buyer(by_type)
        results += self._rule_goods_description(by_type, tables)
        results += self._rule_hs_code_consistency(tables)
        results += self._rule_invoice_pl_qty(tables)
        results += self._rule_total_weight_logic(by_type, tables)
        results += self._rule_voyage_lc_ref(by_type)
        results += self._rule_date_ordering(by_type)
        return results

    # ── Règle 1 : numéro conteneur ───────────────────────────────────────────

    def _rule_container_number(
        self, by_type: dict, tables: dict
    ) -> list[RuleResult]:
        results: list[RuleResult] = []

        # Collecte numéros depuis champs scalar + tables conteneurs
        sets_by_doc: dict[str, set[str]] = {}
        for doc_key in ("bl", "packing_list", "invoice"):
            fields = by_type.get(doc_key)
            if fields is None:
                continue
            cnums: set[str] = set()

            # Depuis champ scalar
            raw = fields.get("container_number", "") or ""
            cnums.update(_extract_container_numbers(raw))
            if not cnums:
                parts = [t.strip().upper() for t in re.split(r"[,;\s/]+", raw) if t.strip()]
                cnums.update(parts)

            # Depuis table conteneurs (BL)
            if doc_key == "bl":
                for row in tables.get("bl", {}).get("containers", []):
                    raw_c = str(row.get("container_number", "") or "")
                    found = _extract_container_numbers(raw_c)
                    cnums.update(found or [raw_c.upper().strip()] if raw_c.strip() else [])

            # Depuis table packages (PL)
            if doc_key == "packing_list":
                for row in tables.get("packing_list", {}).get("packages", []):
                    raw_c = str(row.get("container_number", "") or "")
                    if raw_c.strip():
                        found = _extract_container_numbers(raw_c)
                        cnums.update(found or [raw_c.upper().strip()])

            if cnums:
                sets_by_doc[doc_key] = cnums

        if len(sets_by_doc) < 2:
            return [RuleResult(
                rule="cross_container_number", field="container_number",
                status="skipped", message="Conteneur non renseigné dans assez de documents.",
                severity="info",
            )]

        # Vérification inter-documents
        all_sets = list(sets_by_doc.items())
        for i, (da, sa) in enumerate(all_sets):
            for db, sb in all_sets[i + 1:]:
                common = sa & sb
                pair = f"{da}↔{db}"
                if common:
                    results.append(RuleResult(
                        rule="cross_container_number", field="container_number",
                        status="pass",
                        message=f"Conteneurs cohérents {pair} : {sorted(common)}",
                        severity="info", doc_pair=pair,
                    ))
                else:
                    results.append(RuleResult(
                        rule="cross_container_number", field="container_number",
                        status="fail",
                        original=f"{da}={sorted(sa)}, {db}={sorted(sb)}",
                        message=f"Numéros conteneurs différents {pair} : {sorted(sa)} vs {sorted(sb)}",
                        severity="critical", doc_pair=pair,
                    ))
        return results

    # ── Règle 2 : scellé ─────────────────────────────────────────────────────

    def _rule_seal_number(self, by_type: dict, tables: dict) -> list[RuleResult]:
        bl_fields = by_type.get("bl", {})
        pl_fields = by_type.get("packing_list", {})
        if not bl_fields or not pl_fields:
            return []

        bl_seal = _normalize_str(bl_fields.get("seal_number", ""))
        pl_seal = _normalize_str(pl_fields.get("seal_number", ""))

        # Cherche aussi dans les tables BL
        if not bl_seal:
            for row in tables.get("bl", {}).get("containers", []):
                bl_seal = _normalize_str(row.get("seal_number", "")) or bl_seal

        if not bl_seal or not pl_seal:
            return [RuleResult(
                rule="cross_seal_number", field="seal_number", status="skipped",
                message="Numéro scellé absent dans au moins un document.", severity="info",
            )]

        if bl_seal == pl_seal or _string_similarity(bl_seal, pl_seal) >= 0.90:
            return [RuleResult(
                rule="cross_seal_number", field="seal_number", status="pass",
                message=f"Scellé cohérent BL↔PL : {bl_fields.get('seal_number', '')}",
                severity="info", doc_pair="bl↔packing_list",
            )]
        return [RuleResult(
            rule="cross_seal_number", field="seal_number", status="warning",
            original=f"BL={bl_seal} / PL={pl_seal}",
            message=f"Scellés différents BL↔PL : '{bl_seal}' vs '{pl_seal}'",
            severity="medium", doc_pair="bl↔packing_list",
        )]

    # ── Règle 3 : poids brut ─────────────────────────────────────────────────

    def _rule_weight_tolerance(self, by_type: dict, tables: dict) -> list[RuleResult]:
        bl_fields = by_type.get("bl", {})
        pl_fields = by_type.get("packing_list", {})
        if not bl_fields or not pl_fields:
            return []

        bl_gw = _parse_float(bl_fields.get("gross_weight_kg"))
        pl_gw = _parse_float(pl_fields.get("gross_weight_kg"))

        # Fallback depuis tables
        if pl_gw is None:
            total = sum(
                _parse_float(r.get("gross_weight") or r.get("gross_weight_kg")) or 0
                for r in tables.get("packing_list", {}).get("packages", [])
            )
            pl_gw = total if total > 0 else None

        if bl_gw is None or pl_gw is None:
            return [RuleResult(
                rule="cross_weight_tolerance", field="gross_weight_kg", status="skipped",
                message="Poids brut absent dans au moins un document.", severity="info",
            )]

        ecart = abs(bl_gw - pl_gw) / max(bl_gw, pl_gw, 1.0)
        pct = round(ecart * 100, 1)
        if ecart <= self.WEIGHT_TOLERANCE:
            return [RuleResult(
                rule="cross_weight_tolerance", field="gross_weight_kg", status="pass",
                message=f"Poids brut cohérent BL↔PL (écart {pct}%) — BL:{bl_gw} kg / PL:{pl_gw} kg",
                severity="info", doc_pair="bl↔packing_list",
            )]
        sev = "medium" if ecart <= 0.20 else "high"
        return [RuleResult(
            rule="cross_weight_tolerance", field="gross_weight_kg", status="fail",
            original=f"BL={bl_gw} kg / PL={pl_gw} kg",
            message=f"Écart poids brut BL↔PL trop élevé ({pct}%) — BL:{bl_gw} kg / PL:{pl_gw} kg",
            severity=sev, doc_pair="bl↔packing_list",
        )]

    # ── Règle 4 : poids net ──────────────────────────────────────────────────

    def _rule_net_weight(self, by_type: dict, tables: dict) -> list[RuleResult]:
        bl_fields = by_type.get("bl", {})
        pl_fields = by_type.get("packing_list", {})
        if not bl_fields or not pl_fields:
            return []

        bl_nw = _parse_float(bl_fields.get("net_weight_kg"))
        pl_nw = _parse_float(pl_fields.get("net_weight_kg"))

        if pl_nw is None:
            total = sum(
                _parse_float(r.get("net_weight") or r.get("net_weight_kg")) or 0
                for r in tables.get("packing_list", {}).get("packages", [])
            )
            pl_nw = total if total > 0 else None

        if bl_nw is None or pl_nw is None:
            return []

        ecart = abs(bl_nw - pl_nw) / max(bl_nw, pl_nw, 1.0)
        pct = round(ecart * 100, 1)
        if ecart <= self.WEIGHT_TOLERANCE:
            return [RuleResult(
                rule="cross_net_weight", field="net_weight_kg", status="pass",
                message=f"Poids net cohérent BL↔PL (écart {pct}%) — BL:{bl_nw} kg / PL:{pl_nw} kg",
                severity="info", doc_pair="bl↔packing_list",
            )]
        return [RuleResult(
            rule="cross_net_weight", field="net_weight_kg", status="warning",
            original=f"BL={bl_nw} kg / PL={pl_nw} kg",
            message=f"Écart poids net BL↔PL ({pct}%) — BL:{bl_nw} kg / PL:{pl_nw} kg",
            severity="medium", doc_pair="bl↔packing_list",
        )]

    # ── Règle 5 : incoterm ───────────────────────────────────────────────────

    def _rule_incoterm_consistency(self, by_type: dict) -> list[RuleResult]:
        values: dict[str, str] = {}
        for doc in ("bl", "invoice", "packing_list"):
            val = str(by_type.get(doc, {}).get("incoterm", "") or "").strip().upper()
            if val:
                values[doc] = val

        if len(values) < 2:
            return [RuleResult(
                rule="cross_incoterm_consistency", field="incoterm", status="skipped",
                message="Incoterm absent dans assez de documents.", severity="info",
            )]

        unique_vals = set(values.values())
        if len(unique_vals) == 1:
            return [RuleResult(
                rule="cross_incoterm_consistency", field="incoterm", status="pass",
                message=f"Incoterm cohérent ({next(iter(unique_vals))}) dans tous les documents.",
                severity="info",
            )]
        return [RuleResult(
            rule="cross_incoterm_consistency", field="incoterm", status="fail",
            original=str(values),
            message=f"Incoterms incohérents entre documents : {values}",
            severity="critical",
        )]

    # ── Règle 6 : devise ─────────────────────────────────────────────────────

    def _rule_currency(self, by_type: dict) -> list[RuleResult]:
        values: dict[str, str] = {}
        for doc in ("invoice", "bl"):
            val = str(by_type.get(doc, {}).get("currency", "") or "").strip().upper()
            if val:
                values[doc] = val

        if len(values) < 2:
            return []

        unique = set(values.values())
        if len(unique) == 1:
            return [RuleResult(
                rule="cross_currency", field="currency", status="pass",
                message=f"Devise cohérente ({next(iter(unique))}).",
                severity="info",
            )]
        return [RuleResult(
            rule="cross_currency", field="currency", status="warning",
            original=str(values),
            message=f"Devises différentes entre documents : {values}",
            severity="medium",
        )]

    # ── Règle 7 : shipper ≈ seller ────────────────────────────────────────────

    def _rule_shipper_seller(self, by_type: dict) -> list[RuleResult]:
        bl_fields  = by_type.get("bl", {})
        inv_fields = by_type.get("invoice", {})
        pl_fields  = by_type.get("packing_list", {})
        if not bl_fields or (not inv_fields and not pl_fields):
            return []

        shipper = _normalize_str(bl_fields.get("shipper", ""))
        seller  = _normalize_str(
            inv_fields.get("seller", "") or pl_fields.get("shipper", "")
        )

        if not shipper or not seller:
            return [RuleResult(
                rule="cross_shipper_seller", field="shipper",
                status="skipped", message="Shipper/seller absent.", severity="info",
            )]

        sim = _string_similarity(shipper, seller)
        if sim >= self.PARTY_SIM_THRESHOLD:
            return [RuleResult(
                rule="cross_shipper_seller", field="shipper", status="pass",
                message=f"Shipper (BL) ≈ Seller (Invoice) — similarité {sim:.0%}.",
                severity="info", doc_pair="bl↔invoice",
            )]
        return [RuleResult(
            rule="cross_shipper_seller", field="shipper", status="warning",
            original=f"BL shipper='{bl_fields.get('shipper','')}' / Invoice seller='{inv_fields.get('seller','')}'",
            message=f"Shipper BL et Seller Invoice diffèrent significativement (similarité {sim:.0%}).",
            severity="medium", doc_pair="bl↔invoice",
        )]

    # ── Règle 8 : consignee ≈ buyer ──────────────────────────────────────────

    def _rule_consignee_buyer(self, by_type: dict) -> list[RuleResult]:
        bl_fields  = by_type.get("bl", {})
        inv_fields = by_type.get("invoice", {})
        if not bl_fields or not inv_fields:
            return []

        consignee = _normalize_str(bl_fields.get("consignee", ""))
        buyer     = _normalize_str(inv_fields.get("buyer", ""))

        if not consignee or not buyer:
            return []

        sim = _string_similarity(consignee, buyer)
        if sim >= self.PARTY_SIM_THRESHOLD:
            return [RuleResult(
                rule="cross_consignee_buyer", field="consignee", status="pass",
                message=f"Consignee (BL) ≈ Buyer (Invoice) — similarité {sim:.0%}.",
                severity="info", doc_pair="bl↔invoice",
            )]
        return [RuleResult(
            rule="cross_consignee_buyer", field="consignee", status="warning",
            original=f"BL='{bl_fields.get('consignee','')}' / Invoice='{inv_fields.get('buyer','')}'",
            message=f"Consignee BL et Buyer Invoice diffèrent (similarité {sim:.0%}).",
            severity="medium", doc_pair="bl↔invoice",
        )]

    # ── Règle 9 : description marchandises ───────────────────────────────────

    def _rule_goods_description(self, by_type: dict, tables: dict) -> list[RuleResult]:
        # Collecte les noms de marchandises depuis les tables
        inv_goods = self._collect_goods_names(
            tables.get("invoice", {}).get("invoice_items", []),
            ("description", "goods_description"),
        )
        pl_goods = self._collect_goods_names(
            tables.get("packing_list", {}).get("packages", []),
            ("package_type", "goods_description"),
        )

        # Fallback sur champ scalaire goods_description
        if not inv_goods:
            raw = by_type.get("invoice", {}).get("goods_description", "")
            if raw:
                inv_goods = {_normalize_str(raw)}
        if not pl_goods:
            raw = by_type.get("packing_list", {}).get("goods_description", "")
            if raw:
                pl_goods = {_normalize_str(raw)}

        if not inv_goods or not pl_goods:
            return [RuleResult(
                rule="cross_goods_description", field="goods_description",
                status="skipped", message="Description marchandises absente.", severity="info",
            )]

        # Cherche overlap : chaque item invoice trouve-t-il un match dans la PL ?
        matched = 0
        total   = len(inv_goods)
        for inv_item in inv_goods:
            best = max(
                (_string_similarity(inv_item, pl_item) for pl_item in pl_goods),
                default=0.0,
            )
            if best >= self.GOODS_SIM_THRESHOLD:
                matched += 1

        ratio = matched / max(total, 1)
        if ratio >= 0.80:
            return [RuleResult(
                rule="cross_goods_description", field="goods_description", status="pass",
                message=f"Marchandises alignées Invoice↔PL ({matched}/{total} items correspondants).",
                severity="info", doc_pair="invoice↔packing_list",
            )]
        return [RuleResult(
            rule="cross_goods_description", field="goods_description",
            status="warning" if ratio >= 0.50 else "fail",
            original=f"Invoice={sorted(inv_goods)} / PL={sorted(pl_goods)}",
            message=f"Alignement marchandises partiel Invoice↔PL ({matched}/{total} items, {ratio:.0%}).",
            severity="medium" if ratio >= 0.50 else "high",
            doc_pair="invoice↔packing_list",
        )]

    def _collect_goods_names(
        self, rows: list[dict], keys: tuple[str, ...]
    ) -> set[str]:
        names: set[str] = set()
        for row in rows:
            for key in keys:
                val = row.get(key, "")
                if val and isinstance(val, str) and len(val.strip()) > 3:
                    norm = _normalize_str(val)
                    if norm and not re.fullmatch(r"[\d\s\.,:;/]+", norm):
                        names.add(norm)
                        break
        return names

    # ── Règle 10 : codes SH ──────────────────────────────────────────────────

    def _rule_hs_code_consistency(self, tables: dict) -> list[RuleResult]:
        inv_hs = {
            str(r.get("hs_code", "") or "").strip()
            for r in tables.get("invoice", {}).get("invoice_items", [])
            if r.get("hs_code")
        }
        pl_hs = {
            str(r.get("hs_code", "") or "").strip()
            for r in tables.get("packing_list", {}).get("packages", [])
            if r.get("hs_code")
        }

        if not inv_hs or not pl_hs:
            return []

        common = inv_hs & pl_hs
        if common:
            return [RuleResult(
                rule="cross_hs_code_consistency", field="hs_code", status="pass",
                message=f"Codes SH cohérents Invoice↔PL : {sorted(common)}",
                severity="info", doc_pair="invoice↔packing_list",
            )]
        # Vérification sur préfixe 4 chiffres
        inv_pfx = {h[:4] for h in inv_hs if len(h) >= 4}
        pl_pfx  = {h[:4] for h in pl_hs  if len(h) >= 4}
        if inv_pfx & pl_pfx:
            return [RuleResult(
                rule="cross_hs_code_consistency", field="hs_code", status="warning",
                message=f"Codes SH partiellement alignés (préfixe 4 chiffres) Invoice↔PL.",
                severity="low", doc_pair="invoice↔packing_list",
            )]
        return [RuleResult(
            rule="cross_hs_code_consistency", field="hs_code", status="fail",
            original=f"Invoice={sorted(inv_hs)} / PL={sorted(pl_hs)}",
            message=f"Codes SH incohérents Invoice↔PL : {sorted(inv_hs)} vs {sorted(pl_hs)}",
            severity="high", doc_pair="invoice↔packing_list",
        )]

    # ── Règle 11 : quantités invoice ↔ PL ───────────────────────────────────

    def _rule_invoice_pl_qty(self, tables: dict) -> list[RuleResult]:
        inv_items = tables.get("invoice", {}).get("invoice_items", [])
        pl_items  = tables.get("packing_list", {}).get("packages", [])
        if not inv_items or not pl_items:
            return []

        # Total quantités invoice
        inv_total = sum(
            _parse_float(r.get("quantity", 0)) or 0
            for r in inv_items
        )
        pl_total = sum(
            _parse_float(r.get("quantity", 0) or r.get("number_of_packages", 0)) or 0
            for r in pl_items
        )

        if inv_total == 0 or pl_total == 0:
            return []

        ecart = abs(inv_total - pl_total) / max(inv_total, pl_total)
        pct   = round(ecart * 100, 1)
        if ecart <= 0.05:
            return [RuleResult(
                rule="cross_invoice_pl_qty", field="quantity", status="pass",
                message=f"Quantités totales cohérentes Invoice↔PL (écart {pct}%) — {inv_total} vs {pl_total}.",
                severity="info", doc_pair="invoice↔packing_list",
            )]
        return [RuleResult(
            rule="cross_invoice_pl_qty", field="quantity",
            status="warning" if ecart <= 0.15 else "fail",
            original=f"Invoice={inv_total} / PL={pl_total}",
            message=f"Écart quantités Invoice↔PL ({pct}%) — {inv_total} vs {pl_total}.",
            severity="medium" if ecart <= 0.15 else "high",
            doc_pair="invoice↔packing_list",
        )]

    # ── Règle 12 : poids net ≤ poids brut ───────────────────────────────────

    def _rule_total_weight_logic(self, by_type: dict, tables: dict) -> list[RuleResult]:
        results: list[RuleResult] = []
        for doc_key in ("bl", "packing_list"):
            fields = by_type.get(doc_key, {})
            gw = _parse_float(fields.get("gross_weight_kg"))
            nw = _parse_float(fields.get("net_weight_kg"))
            if gw is not None and nw is not None and gw > 0:
                if nw <= gw:
                    results.append(RuleResult(
                        rule="cross_total_weight_logic", field="gross_weight_kg",
                        status="pass",
                        message=f"[{doc_key}] Poids net ({nw} kg) ≤ poids brut ({gw} kg). ✓",
                        severity="info",
                    ))
                else:
                    results.append(RuleResult(
                        rule="cross_total_weight_logic", field="gross_weight_kg",
                        status="fail",
                        original=f"net={nw} > gross={gw}",
                        message=f"[{doc_key}] Poids net ({nw} kg) supérieur au poids brut ({gw} kg) — impossible.",
                        severity="critical",
                    ))
        return results

    # ── Règle 13 : référence LC ↔ voyage ─────────────────────────────────────

    def _rule_voyage_lc_ref(self, by_type: dict) -> list[RuleResult]:
        inv_fields = by_type.get("invoice", {})
        bl_fields  = by_type.get("bl", {})
        if not inv_fields or not bl_fields:
            return []

        lc_ref = str(inv_fields.get("lc_reference", "") or "").strip()
        voyage = str(bl_fields.get("voyage", "") or "").strip()

        if not lc_ref or not voyage:
            return []

        # Cherche si le voyage est mentionné dans la réf LC (ex: "V26-088" contient "V26")
        lc_norm  = _normalize_str(lc_ref)
        voy_norm = _normalize_str(voyage)

        if voy_norm and voy_norm in lc_norm:
            return [RuleResult(
                rule="cross_voyage_lc_ref", field="lc_reference", status="pass",
                message=f"Référence LC '{lc_ref}' contient la référence voyage '{voyage}'.",
                severity="info", doc_pair="invoice↔bl",
            )]
        return [RuleResult(
            rule="cross_voyage_lc_ref", field="lc_reference", status="warning",
            original=f"LC ref='{lc_ref}' / BL voyage='{voyage}'",
            message=f"Réf LC '{lc_ref}' ne contient pas la référence voyage BL '{voyage}'.",
            severity="low", doc_pair="invoice↔bl",
        )]

    # ── Règle 15 : port déchargement BL ↔ destination Invoice ───────────────

    def _rule_port_discharge_match(self, by_type: dict) -> list[RuleResult]:
        bl_fields  = by_type.get("bl", {})
        inv_fields = by_type.get("invoice", {})
        if not bl_fields or not inv_fields:
            return []

        bl_port  = _normalize_str(bl_fields.get("port_of_discharge", ""))
        inv_port = _normalize_str(
            inv_fields.get("port_of_destination", "")
            or inv_fields.get("port_of_discharge", "")
        )

        if not bl_port or not inv_port:
            return [RuleResult(
                rule="cross_port_discharge_match", field="port_of_discharge",
                status="skipped",
                message="Port de destination absent dans l'un des documents.", severity="info",
            )]

        sim = _string_similarity(bl_port, inv_port)
        if sim >= 0.75:
            return [RuleResult(
                rule="cross_port_discharge_match", field="port_of_discharge", status="pass",
                message=(
                    f"Port destination cohérent BL↔Invoice : "
                    f"'{bl_fields.get('port_of_discharge','')}' ≈ "
                    f"'{inv_fields.get('port_of_destination', inv_fields.get('port_of_discharge',''))}'"
                ),
                severity="info", doc_pair="bl↔invoice",
            )]
        return [RuleResult(
            rule="cross_port_discharge_match", field="port_of_discharge", status="warning",
            original=f"BL={bl_port} / Invoice={inv_port}",
            message=f"Port destination potentiellement incohérent BL↔Invoice (similarité {sim:.0%}).",
            severity="medium", doc_pair="bl↔invoice",
        )]

    # ── Règle 16 : plausibilité montants lignes facture ───────────────────────

    def _rule_line_amount_plausibility(self, tables: dict) -> list[RuleResult]:
        inv_items = tables.get("invoice", {}).get("invoice_items", [])
        if not inv_items:
            return []

        inconsistent: list[str] = []
        for row in inv_items:
            qty    = _parse_float(row.get("quantity"))
            uprice = _parse_float(row.get("unit_price"))
            ltotal = _parse_float(row.get("line_total") or row.get("line_amount"))
            if qty is None or uprice is None or ltotal is None or ltotal == 0:
                continue
            computed = qty * uprice
            ecart = abs(computed - ltotal) / max(abs(ltotal), 1)
            if ecart > 0.05:
                desc = str(row.get("description") or row.get("goods_description") or "?")[:40]
                inconsistent.append(f"'{desc}': {qty}×{uprice:.2f}≈{computed:.2f}≠{ltotal:.2f}")

        if not inconsistent:
            return [RuleResult(
                rule="cross_line_amount_plausibility", field="line_total", status="pass",
                message="Montants lignes facture cohérents (quantité × prix unitaire ≈ montant ligne).",
                severity="info", doc_pair="invoice↔packing_list",
            )]
        return [RuleResult(
            rule="cross_line_amount_plausibility", field="line_total", status="warning",
            original="; ".join(inconsistent[:3]),
            message=f"Incohérences montants lignes facture : {len(inconsistent)} ligne(s) avec écart >5%.",
            severity="medium", doc_pair="invoice↔packing_list",
        )]

    # ── Règle 17 : poids net Invoice ↔ PL ────────────────────────────────────

    def _rule_weight_inv_pl_net(self, by_type: dict, tables: dict) -> list[RuleResult]:
        inv_fields = by_type.get("invoice", {})
        pl_fields  = by_type.get("packing_list", {})
        if not inv_fields or not pl_fields:
            return []

        inv_nw = _parse_float(
            inv_fields.get("net_weight_kg") or inv_fields.get("total_net_kg")
        )
        pl_nw = _parse_float(
            pl_fields.get("net_weight_kg") or pl_fields.get("total_net_kg")
        )

        if pl_nw is None:
            total = sum(
                _parse_float(r.get("net_weight") or r.get("net_weight_kg")) or 0
                for r in tables.get("packing_list", {}).get("packages", [])
            )
            pl_nw = total if total > 0 else None

        if inv_nw is None or pl_nw is None:
            return []

        ecart = abs(inv_nw - pl_nw) / max(inv_nw, pl_nw, 1.0)
        pct   = round(ecart * 100, 1)
        if ecart <= self.WEIGHT_TOLERANCE:
            return [RuleResult(
                rule="cross_weight_inv_pl_net", field="net_weight_kg", status="pass",
                message=f"Poids net cohérent Invoice↔PL (écart {pct}%) — Invoice:{inv_nw} kg / PL:{pl_nw} kg",
                severity="info", doc_pair="invoice↔packing_list",
            )]
        sev = "medium" if ecart <= 0.20 else "high"
        return [RuleResult(
            rule="cross_weight_inv_pl_net", field="net_weight_kg",
            status="warning" if ecart <= 0.20 else "fail",
            original=f"Invoice={inv_nw} kg / PL={pl_nw} kg",
            message=f"Écart poids net Invoice↔PL ({pct}%) — Invoice:{inv_nw} kg / PL:{pl_nw} kg",
            severity=sev, doc_pair="invoice↔packing_list",
        )]

    # ── Règle 14 : ordre des dates ────────────────────────────────────────────

    def _rule_date_ordering(self, by_type: dict) -> list[RuleResult]:
        from datetime import datetime

        bl_fields  = by_type.get("bl", {})
        inv_fields = by_type.get("invoice", {})
        if not bl_fields or not inv_fields:
            return []

        eta_raw      = bl_fields.get("eta", "")
        inv_date_raw = inv_fields.get("invoice_date") or inv_fields.get("issue_date", "")

        def _parse_date(raw: str):
            if not raw:
                return None
            for fmt in ("%Y-%m-%d", "%d %b %Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
                try:
                    return datetime.strptime(raw.strip(), fmt)
                except ValueError:
                    continue
            return None

        eta      = _parse_date(str(eta_raw or ""))
        inv_date = _parse_date(str(inv_date_raw or ""))

        if eta is None or inv_date is None:
            return []

        if inv_date <= eta:
            return [RuleResult(
                rule="cross_date_ordering", field="eta", status="pass",
                message=f"Date facture ({inv_date_raw}) ≤ ETA ({eta_raw}). ✓",
                severity="info", doc_pair="invoice↔bl",
            )]
        return [RuleResult(
            rule="cross_date_ordering", field="eta", status="warning",
            original=f"invoice_date={inv_date_raw} > eta={eta_raw}",
            message=f"Date facture ({inv_date_raw}) postérieure à l'ETA ({eta_raw}).",
            severity="medium", doc_pair="invoice↔bl",
        )]
