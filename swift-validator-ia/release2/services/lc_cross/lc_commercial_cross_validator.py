"""
lc_commercial_cross_validator.py  —  v3.0
==========================================
Category-Centric Validation Engine

Architecture bancaire hiérarchique par catégorie :
  Layer 1 — Document Understanding     (schémas extraits BL / INV / PL)
  Layer 2 — Cross-document reconciliation (INV↔BL, INV↔PL, BL↔PL)
  Layer 3 — Category Validation vs MT700 (source de vérité contractuelle)
             · Financial   (35%) — montant, monnaie, tolérance, arithmétique
             · Goods       (25%) — marchandises 45A, tables, sémantique
             · Shipment    (10%) — dates, ports, transbordement, poids/CBM
             · Parties     (10%) — bénéficiaire, donneur d'ordre, consignataire
             · Documentary (20%) — complétude, incoterm, conditions 47A
  Layer 4 — Global Decision (ACCEPT / DISCREPANT / REJECT)

vs v2.2 :
  ✔ Pas de double pénalisation — chaque anomalie appartient à une seule catégorie
  ✔ Root-cause analysis — primary_issue + derived_issues par catégorie
  ✔ Explicabilité bancaire — catégorie → statut → cause principale
  ✔ Vue globale INV/BL/PL avant validation MT700 (cross_document)
  ✔ Pondération réaliste trade finance
  ✔ Décision bancaire ACCEPT / DISCREPANT / REJECT
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

from .lc_xai_engine import LcXaiEngine

log = logging.getLogger(__name__)

# Module-level XAI engine singleton (stateless — safe to share)
_xai_engine = LcXaiEngine()

# ── Category weights ────────────────────────────────────────────────────
_CATEGORY_WEIGHTS: dict[str, float] = {
    "financial":   0.35,
    "goods":       0.25,
    "documentary": 0.20,
    "shipment":    0.10,
    "parties":     0.10,
}

# Severity weights for check aggregation within a category
_CHECK_SEV_WEIGHT: dict[str, float] = {
    "critical": 3.0,
    "high":     2.0,
    "medium":   1.5,
    "low":      1.0,
    "info":     0.5,
}

_STATUS_SCORE: dict[str, float] = {
    "pass":      1.00,
    "corrected": 0.90,
    "warning":   0.65,
    "fail":      0.00,
}

# Skipped rules: graduated penalty based on severity (missing data ≠ free pass)
_SKIP_SCORE: dict[str, float] = {
    "critical": 0.70,
    "high":     0.78,
    "medium":   0.85,
    "low":      0.92,
    "info":     0.95,
}

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]


# ── Module-level helpers ────────────────────────────────────────────────

def _str(val: Any) -> str:
    return "" if val is None else str(val).strip()


def _extract_company_name(text: str) -> str:
    """
    Extract the company name from an LC party field that may contain
    a full address block (e.g. "(Seller) ACME LTD 12 Rue du Commerce, Tunis").
    """
    if not text:
        return text
    t = re.sub(r'^\([^)]{1,20}\)\s*', '', text.strip())
    for line in t.splitlines():
        line = line.strip()
        if line:
            t = line
            break
    t = re.sub(r'\s+\d+[,\s/].*$', '', t).strip()
    _addr_kw = r'(?:rue|street|ave|avenue|road|blvd|boulevard|way|lane|drive|str\.?|allée|zone)'
    t = re.sub(r'\s+' + _addr_kw + r'.*$', '', t, flags=re.IGNORECASE).strip()
    t = re.sub(r'\s+[+\(]?\d[\d\s\-]{6,}.*$', '', t).strip()
    return t or text


def _normalize_party(name: str) -> str:
    s = name.upper()
    for suffix in ["SARL", "SAS", "SA", "LTD", "LLC", "INC", "CORP", "CO.", "CO",
                   "GMBH", "AG", "BV", "NV", "PLC", "SPA", "OOO", "JSC", "PJSC"]:
        s = re.sub(r'\b' + re.escape(suffix) + r'\b\.?', '', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    if s1 == s2:
        return 1.0
    n1, n2 = len(s1), len(s2)
    if n1 == 0 or n2 == 0:
        return 0.0
    match_dist = max(n1, n2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * n1
    s2_matches = [False] * n2
    matches = 0
    transpositions = 0
    for i in range(n1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, n2)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(n1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / n1 + matches / n2 + (matches - transpositions / 2) / matches) / 3
    prefix = 0
    for i in range(min(4, n1, n2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * p * (1 - jaro)


def _fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_name = _extract_company_name(a)
    b_name = _extract_company_name(b)
    an = _normalize_party(a_name)
    bn = _normalize_party(b_name)
    if not an or not bn:
        return 0.0
    jw      = _jaro_winkler(an, bn)
    seq     = SequenceMatcher(None, an, bn).ratio()
    a_tok   = set(an.split())
    b_tok   = set(bn.split())
    overlap = len(a_tok & b_tok) / max(len(a_tok), len(b_tok)) if a_tok and b_tok else 0.0
    return max(jw, seq, overlap)


def _parse_date(s: Any) -> date | None:
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d",
                "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
                "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _find_date_in_dict(d: dict) -> date | None:
    _date_re = re.compile(
        r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{2}[/\-]\d{2}'
        r'|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        re.IGNORECASE,
    )
    for v in d.values():
        if not isinstance(v, str) or not v.strip():
            continue
        m = _date_re.search(v)
        if m:
            dt = _parse_date(m.group(0))
            if dt:
                return dt
    return None


def _bl_date(bl: dict) -> date | None:
    for key in ("on_board_date", "shipped_on_board_date", "onboard_date",
                "bl_date", "date_of_issue", "issue_date", "date", "board_date"):
        d = _parse_date(bl.get(key))
        if d:
            return d
    return _find_date_in_dict(bl)


def _invoice_date(invoice: dict) -> date | None:
    for key in ("invoice_date", "date", "issue_date", "document_date",
                "billing_date", "date_of_issue"):
        d = _parse_date(invoice.get(key))
        if d:
            return d
    return _find_date_in_dict(invoice)


def _parse_amount(val: Any) -> float | None:
    """Normalise un montant commercial en float — gère formats US, EU et FR."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Déléguer au post-processor qui gère EU/US/FR
    try:
        from release2.services.post_processor import parse_amount as _pa
        result = _pa(s)
        if result is not None:
            return result
    except Exception:
        pass
    # Fallback local minimal
    clean = re.sub(r"[^\d.,]", "", s)
    if not clean:
        return None
    if clean.count(",") == 1 and clean.count(".") == 0:
        clean = clean.replace(",", ".")
    else:
        clean = clean.replace(",", "")
    try:
        return float(clean)
    except ValueError:
        return None


def _normalize_port(port: str) -> str:
    p = re.sub(r"\s*,\s*\w[\w\s]*$", "", port.upper().strip())
    return p.strip()


def _port_match(a: str, b: str) -> bool:
    na, nb = _normalize_port(a), _normalize_port(b)
    return na in nb or nb in na or _fuzzy_score(na, nb) >= 0.70


# ── Rule result dataclass ───────────────────────────────────────────────

@dataclass
class LcRuleResult:
    rule:      str
    status:    str   # pass | fail | warning | skipped
    severity:  str   # critical | high | medium | low | info
    message:   str
    detail:    str = ""
    lc_value:  str = ""
    doc_value: str = ""
    doc_type:  str = ""


# ── Main validator ──────────────────────────────────────────────────────

class LcCommercialCrossValidator:
    """
    Category-Centric LC validation engine.
    Validates MT700 fields against BL, Invoice, and Packing List.
    Produces a hierarchical report by business category.
    """

    # ── Static helpers ─────────────────────────────────────────────────

    @staticmethod
    def _get_rows(tables: Any, key: str) -> list[dict]:
        """Extract row list from either {key: [rows]} dict or a bare list."""
        if isinstance(tables, dict):
            return tables.get(key) or []
        if isinstance(tables, list):
            return tables
        return []

    @staticmethod
    def _sum_col(rows: list[dict], *keys: str) -> float | None:
        total, found = 0.0, False
        for row in rows:
            for key in keys:
                val = _parse_amount(row.get(key))
                if val is not None:
                    total += val
                    found = True
                    break
        return total if found else None

    @staticmethod
    def _enrich_goods_from_rows(flat: str, rows: list[dict], *keys: str) -> str:
        """
        Concatène le champ plat + les descriptions uniques des lignes de tableau.
        Produit un texte enrichi pour la comparaison sémantique avec LC 45A.
        """
        parts: list[str] = [flat] if flat and flat.strip() else []
        seen:  set[str]  = {flat.upper()} if flat else set()
        for row in rows:
            for key in keys:
                val = _str(row.get(key))
                if val and val.upper() not in seen:
                    parts.append(val)
                    seen.add(val.upper())
                    break
        return " | ".join(parts)

    # ── Main entry point ───────────────────────────────────────────────

    def validate(self, lc: dict, commercial: dict) -> dict:
        """
        lc         : dict with MT700 parsed fields
        commercial : dict with keys 'bl', 'invoice', 'packing_list',
                     optionally 'invoice_tables', 'bl_tables', 'pl_tables'
        """
        bl         = commercial.get("bl") or {}
        invoice    = commercial.get("invoice") or {}
        pl         = commercial.get("packing_list") or {}
        inv_tables = commercial.get("invoice_tables") or {}
        bl_tables  = commercial.get("bl_tables")  or {}
        pl_tables  = commercial.get("pl_tables")  or {}

        inv_rows = self._get_rows(inv_tables, "invoice_items")
        bl_rows  = self._get_rows(bl_tables,  "containers")
        pl_rows  = self._get_rows(pl_tables,  "packages")

        # ── Layer 2: Cross-document reconciliation (sans MT700) ────────
        cross_doc = self._cross_document_reconciliation(
            bl, invoice, pl, inv_rows, bl_rows, pl_rows
        )

        # ── Layer 3: Category validation contre MT700 ──────────────────
        financial   = self._validate_financial(lc, invoice, inv_rows)
        goods       = self._validate_goods(lc, invoice, bl, inv_rows, bl_rows)
        shipment    = self._validate_shipment(lc, bl, invoice, pl, bl_rows, pl_rows)
        parties     = self._validate_parties(lc, bl, invoice, pl)
        documentary = self._validate_documentary(lc, bl, invoice, pl)

        categories = {
            "financial":   financial,
            "goods":       goods,
            "shipment":    shipment,
            "parties":     parties,
            "documentary": documentary,
        }

        # ── Layer 4: Score global et décision ─────────────────────────
        final_score = round(min(max(
            sum(cat["score"] * cat["weight"] for cat in categories.values()),
            0.0), 1.0), 4)

        risk_level = self._classify_risk(categories, final_score)
        decision   = self._build_decision(categories, final_score)

        # ── XAI: Shapley attribution + intelligent NL explanations ─────
        xai = _xai_engine.explain(
            categories=categories,
            cross_doc=cross_doc,
            final_score=final_score,
            decision=decision,
            risk_level=risk_level,
        )
        explanations = xai["explanations"]

        # ── Backward-compat flattened fields ──────────────────────────
        rule_results = (
            financial["checks"] +
            shipment["checks"] +
            [r for r in documentary["checks"]
             if not r.get("rule", "").startswith(("lc_cond", "lc_47a"))]
        )
        fuzzy_results    = parties.get("_fuzzy_compat") or []
        semantic_results = goods.get("semantic_detail") or {}
        condition_results = [
            r for r in documentary["checks"]
            if r.get("rule", "").startswith(("lc_cond", "lc_47a"))
        ]

        score_breakdown = {
            cat: round(data["score"], 4) for cat, data in categories.items()
        }

        return {
            "final_score":  final_score,
            "risk_level":   risk_level,
            "decision":     decision,

            # Primary: category-centric
            "categories": {
                k: {ck: cv for ck, cv in cat.items() if not ck.startswith("_")}
                for k, cat in categories.items()
            },
            "cross_document":   cross_doc,
            "category_weights": _CATEGORY_WEIGHTS,

            # XAI: Level 3 Shapley + Level 4 probabilistic/causal/counterfactual
            "xai": {
                "shapley_values":      xai["shapley_values"],
                "waterfall":           xai["waterfall"],
                "executive_summary":   xai["executive_summary"],
                "decision_rationale":  xai["decision_rationale"],
                "compliance_evidence": xai["compliance_evidence"],
                "remediation_plan":    xai["remediation_plan"],
                # Level 4
                "probabilistic":      xai["probabilistic"],
                "causal":             xai["causal"],
                "counterfactuals":    xai["counterfactuals"],
                "causal_narrative":   xai["causal_narrative"],
                "simulation":         xai["simulation"],
                "decision_confidence": xai["decision_confidence"],
            },

            # Backward-compat (consumed by Streamlit tabs)
            "rule_results":      rule_results,
            "fuzzy_results":     fuzzy_results,
            "semantic_results":  semantic_results,
            "condition_results": condition_results,
            "score_breakdown":   score_breakdown,
            "weights":           _CATEGORY_WEIGHTS,
            "explanations":      explanations,
        }

    # ── Layer 2: Cross-document reconciliation ─────────────────────────

    def _cross_document_reconciliation(
            self, bl: dict, invoice: dict, pl: dict,
            inv_rows: list, bl_rows: list, pl_rows: list) -> dict:
        """
        Cohérence inter-documents INV↔BL, INV↔PL, BL↔PL sans MT700.
        Produit la vue globale avant validation contre les contraintes LC.
        """

        def _fuzz(a: str, b: str, threshold: float = 0.70) -> dict:
            if not a or not b:
                return {"status": "skipped"}
            score = _fuzzy_score(a, b)
            return {
                "status":   "pass" if score >= threshold else "warning" if score >= 0.45 else "fail",
                "score":    round(score, 3),
                "value_a":  a[:120],
                "value_b":  b[:120],
            }

        def _port(a: str, b: str) -> dict:
            if not a or not b:
                return {"status": "skipped"}
            return {
                "status":  "pass" if _port_match(a, b) else "warning",
                "value_a": a,
                "value_b": b,
            }

        # INV ↔ BL ─────────────────────────────────────────────────────
        inv_goods = _str(invoice.get("goods_description"))
        bl_goods  = _str(bl.get("goods_description"))
        goods_sim: dict = {"status": "skipped"}
        if inv_goods and bl_goods:
            try:
                from sentence_transformers import util as st_util
                model = self._get_st_model()
                s = float(st_util.cos_sim(
                    model.encode(inv_goods, convert_to_tensor=True),
                    model.encode(bl_goods,  convert_to_tensor=True),
                ))
                goods_sim = {
                    "status": "pass" if s >= 0.60 else "warning" if s >= 0.35 else "fail",
                    "score":  round(s, 3),
                }
            except Exception:
                pass

        inv_bl = {
            "seller_vs_shipper": _fuzz(_str(invoice.get("seller")), _str(bl.get("shipper"))),
            "port_of_loading":   _port(_str(invoice.get("port_of_loading")), _str(bl.get("port_of_loading"))),
            "port_of_discharge": _port(
                _str(invoice.get("port_of_discharge") or invoice.get("port_of_destination")),
                _str(bl.get("port_of_discharge")),
            ),
            "vessel_voyage":     _fuzz(_str(invoice.get("vessel")), _str(bl.get("vessel")), threshold=0.60),
            "goods_similarity":  goods_sim,
        }

        # INV ↔ PL ─────────────────────────────────────────────────────
        inv_cur = _str(invoice.get("currency")).upper()
        pl_cur  = _str(pl.get("currency", "")).upper()
        cur_match: dict = {"status": "skipped"}
        if inv_cur and pl_cur:
            cur_match = {"status": "pass" if inv_cur == pl_cur else "warning",
                         "inv": inv_cur, "pl": pl_cur}

        inv_inc = _str(invoice.get("incoterm")).upper()
        pl_inc  = _str(pl.get("incoterm", "")).upper()
        inc_match: dict = {"status": "skipped"}
        if inv_inc and pl_inc:
            inc_match = {
                "status": "pass" if (inv_inc in pl_inc or pl_inc in inv_inc) else "warning",
                "inv": inv_inc, "pl": pl_inc,
            }

        inv_pl = {
            "seller_vs_exporter": _fuzz(_str(invoice.get("seller")), _str(pl.get("exporter"))),
            "currency_match":     cur_match,
            "incoterm_match":     inc_match,
        }

        # BL ↔ PL ─────────────────────────────────────────────────────
        wt_res  = self._rule_weight_consistency(bl_rows, pl_rows, bl, pl)
        cbm_res = self._rule_cbm_consistency(bl_rows, pl_rows, bl, pl)

        bl_pl = {
            "weight_consistency": self._rr_to_dict(wt_res),
            "cbm_consistency":    self._rr_to_dict(cbm_res),
            "vessel_voyage":      _fuzz(_str(bl.get("vessel")), _str(pl.get("vessel")), threshold=0.60),
            "port_of_loading":    _port(_str(bl.get("port_of_loading")), _str(pl.get("port_of_loading"))),
        }

        # Overall cross-doc score ────────────────────────────────────
        all_statuses = []
        for group in (inv_bl, inv_pl, bl_pl):
            for v in group.values():
                if isinstance(v, dict) and v.get("status") not in (None, "skipped"):
                    all_statuses.append(
                        _STATUS_SCORE.get(v["status"], 0.0)
                    )
        overall = round(sum(all_statuses) / len(all_statuses), 4) if all_statuses else 0.80

        return {
            "inv_bl":        inv_bl,
            "inv_pl":        inv_pl,
            "bl_pl":         bl_pl,
            "overall_score": overall,
        }

    # ── Layer 3: Category validators ───────────────────────────────────

    def _validate_financial(self, lc: dict, invoice: dict,
                             inv_rows: list[dict]) -> dict:
        """
        Financial (35%) — monnaie, montant LC, arithmétique interne facture.
        Double pénalisation évitée : une seule catégorie porte tous les risques financiers.
        """
        checks_raw = [
            self._rule_currency(lc, invoice),
            self._rule_amount_tolerance(lc, invoice),
            self._rule_invoice_items_total(inv_rows, invoice),
        ]
        score  = self._aggregate_checks(checks_raw)
        status = self._detect_category_status(checks_raw, score)
        primary, derived = self._detect_issues(checks_raw)
        return {
            "score":         round(score, 4),
            "status":        status,
            "weight":        _CATEGORY_WEIGHTS["financial"],
            "primary_issue": primary,
            "derived_issues": derived,
            "checks":        [self._rr_to_dict(r) for r in checks_raw],
        }

    def _validate_goods(self, lc: dict, invoice: dict, bl: dict,
                        inv_rows: list[dict], bl_rows: list[dict]) -> dict:
        """
        Goods (25%) — sémantique LC 45A vs documents (enrichis depuis tables).
        Toutes les vérifications de marchandises regroupées ici.
        """
        sem = self._run_semantic(lc, bl, invoice, inv_rows=inv_rows, bl_rows=bl_rows)

        checks_raw: list[LcRuleResult] = []
        lc_vs_inv = sem.get("lc_vs_invoice")
        if lc_vs_inv is not None:
            lbl = sem.get("lc_vs_invoice_label", "")
            s   = "pass" if lc_vs_inv >= 0.70 else "warning" if lc_vs_inv >= 0.45 else "fail"
            checks_raw.append(LcRuleResult(
                "goods_lc_vs_invoice", s, "critical",
                f"LC 45A ↔ Invoice goods : {lc_vs_inv:.0%} ({lbl})",
                doc_value=str(round(lc_vs_inv, 3)), doc_type="invoice",
            ))
        lc_vs_bl = sem.get("lc_vs_bl")
        if lc_vs_bl is not None:
            lbl = sem.get("lc_vs_bl_label", "")
            s   = "pass" if lc_vs_bl >= 0.70 else "warning" if lc_vs_bl >= 0.45 else "fail"
            checks_raw.append(LcRuleResult(
                "goods_lc_vs_bl", s, "high",
                f"LC 45A ↔ BL goods : {lc_vs_bl:.0%} ({lbl})",
                doc_value=str(round(lc_vs_bl, 3)), doc_type="bl",
            ))

        score  = self._aggregate_checks(checks_raw) if checks_raw else sem.get("overall_score", 0.70)
        status = self._detect_category_status(checks_raw, score)
        primary, derived = self._detect_issues(checks_raw)
        return {
            "score":          round(score, 4),
            "status":         status,
            "weight":         _CATEGORY_WEIGHTS["goods"],
            "primary_issue":  primary,
            "derived_issues": derived,
            "checks":         [self._rr_to_dict(r) for r in checks_raw],
            "semantic_detail": sem,
            "_sem":            sem,  # kept for backward compat extraction
        }

    def _validate_shipment(self, lc: dict, bl: dict, invoice: dict, pl: dict,
                            bl_rows: list[dict], pl_rows: list[dict]) -> dict:
        """
        Shipment (10%) — dates, ports, transbordement, poids/CBM.
        Regroupe tous les checks logistiques pour éviter la double pénalisation.
        """
        checks_raw = [
            self._rule_expiry_date(lc, invoice, bl),
            self._rule_shipment_date(lc, bl),
            self._rule_presentation_period(lc, invoice, bl),
            self._rule_port_of_discharge(lc, bl, invoice),
            self._rule_transhipment(lc, bl),
            self._rule_partial_shipments(lc, bl),
            self._rule_weight_consistency(bl_rows, pl_rows, bl, pl),
            self._rule_cbm_consistency(bl_rows, pl_rows, bl, pl),
        ]
        score  = self._aggregate_checks(checks_raw)
        status = self._detect_category_status(checks_raw, score)
        primary, derived = self._detect_issues(checks_raw)
        return {
            "score":          round(score, 4),
            "status":         status,
            "weight":         _CATEGORY_WEIGHTS["shipment"],
            "primary_issue":  primary,
            "derived_issues": derived,
            "checks":         [self._rr_to_dict(r) for r in checks_raw],
        }

    def _validate_parties(self, lc: dict, bl: dict, invoice: dict, pl: dict) -> dict:
        """
        Parties (10%) — bénéficiaire, donneur d'ordre, consignataire.
        Matching hybride Jaro-Winkler + token overlap sur noms d'entreprises.
        """
        fuzzy_results = self._run_fuzzy(lc, bl, invoice, pl)

        checks_raw: list[LcRuleResult] = []
        _sev_map = {
            "beneficiary_vs_seller":   "high",
            "beneficiary_vs_shipper":  "high",
            "applicant_vs_buyer":      "high",
            "applicant_vs_consignee":  "medium",
        }
        for fr in fuzzy_results:
            sev = _sev_map.get(fr["check"], "medium")
            checks_raw.append(LcRuleResult(
                fr["check"],
                fr["status"],
                sev,
                fr["message"],
                lc_value=fr.get("lc_value", ""),
                doc_value=fr.get("doc_value", ""),
                doc_type=fr.get("doc_type", ""),
            ))

        score  = self._aggregate_checks(checks_raw) if checks_raw else 0.80
        status = self._detect_category_status(checks_raw, score)
        primary, derived = self._detect_issues(checks_raw)
        return {
            "score":          round(score, 4),
            "status":         status,
            "weight":         _CATEGORY_WEIGHTS["parties"],
            "primary_issue":  primary,
            "derived_issues": derived,
            "checks":         [self._rr_to_dict(r) for r in checks_raw],
            "_fuzzy_compat":  fuzzy_results,  # backward compat for Streamlit tab
        }

    def _validate_documentary(self, lc: dict, bl: dict, invoice: dict, pl: dict) -> dict:
        """
        Documentary (20%) — complétude documentaire, incoterm, conditions 47A.
        """
        checks_raw: list[LcRuleResult] = [
            self._rule_incoterm(lc, invoice),
            self._rule_doc_completeness(lc, bl, invoice, pl),
        ]
        checks_raw += list(self._run_conditions(lc, bl, invoice, pl))

        score  = self._aggregate_checks(checks_raw)
        status = self._detect_category_status(checks_raw, score)
        primary, derived = self._detect_issues(checks_raw)
        return {
            "score":          round(score, 4),
            "status":         status,
            "weight":         _CATEGORY_WEIGHTS["documentary"],
            "primary_issue":  primary,
            "derived_issues": derived,
            "checks":         [self._rr_to_dict(r) for r in checks_raw],
        }

    # ── Aggregation helpers ────────────────────────────────────────────

    def _aggregate_checks(self, checks: list[LcRuleResult]) -> float:
        """
        Severity-weighted scoring.
        Critical checks count 3× more than low/info checks.
        Skipped checks apply a graduated penalty (not a free pass).
        """
        if not checks:
            return 0.80
        total_w = 0.0
        total_s = 0.0
        for r in checks:
            w = _CHECK_SEV_WEIGHT.get(r.severity, 1.0)
            s = _SKIP_SCORE.get(r.severity, 0.85) if r.status == "skipped" \
                else _STATUS_SCORE.get(r.status, 0.0)
            total_s += s * w
            total_w += w
        return round(total_s / total_w, 4) if total_w > 0 else 0.80

    @staticmethod
    def _detect_category_status(checks: list[LcRuleResult], score: float) -> str:
        """pass | partial | fail — driven by critical fails and score thresholds."""
        if any(r.status == "fail" and r.severity == "critical" for r in checks):
            return "fail"
        if score >= 0.80:
            return "pass"
        if score >= 0.55:
            return "partial"
        return "fail"

    @staticmethod
    def _detect_issues(checks: list[LcRuleResult]) -> tuple[str | None, list[str]]:
        """Return (primary_issue, derived_issues) for root-cause analysis."""
        primary = None
        for sev in _SEVERITY_ORDER:
            for r in checks:
                if r.status == "fail" and r.severity == sev:
                    primary = r.rule
                    break
            if primary:
                break
        # If no fail, check warnings
        if primary is None:
            for sev in _SEVERITY_ORDER:
                for r in checks:
                    if r.status == "warning" and r.severity == sev:
                        primary = r.rule
                        break
                if primary:
                    break
        derived = [
            r.rule for r in checks
            if r.status in ("fail", "warning") and r.rule != primary
        ]
        return primary, derived

    def _classify_risk(self, categories: dict, final_score: float) -> str:
        fin    = categories["financial"]
        n_fail = sum(1 for c in categories.values() if c["status"] == "fail")
        fin_crit_fail = any(
            r.get("severity") == "critical" and r.get("status") == "fail"
            for r in fin["checks"]
        )
        if fin_crit_fail or final_score < 0.50:
            return "critical"
        if n_fail >= 2 or final_score < 0.65:
            return "high"
        if n_fail >= 1 or final_score < 0.80:
            return "medium"
        return "low"

    @staticmethod
    def _build_decision(categories: dict, final_score: float) -> str:
        """ACCEPT / DISCREPANT / REJECT based on category statuses."""
        fin = categories["financial"]
        # Reject: financial critical fail (currency or amount out of tolerance)
        fin_crit_fail = any(
            r.get("severity") == "critical" and r.get("status") == "fail"
            for r in fin["checks"]
        )
        if fin_crit_fail:
            return "REJECT"
        n_fail = sum(1 for c in categories.values() if c["status"] == "fail")
        n_part = sum(1 for c in categories.values() if c["status"] == "partial")
        if n_fail >= 1:
            return "DISCREPANT"
        if n_part >= 2 or final_score < 0.75:
            return "DISCREPANT"
        return "ACCEPT"

    # ── Existing rule methods (building blocks) ────────────────────────

    def _rule_currency(self, lc, invoice) -> LcRuleResult:
        lc_cur  = _str(lc.get("currency")).upper()
        inv_cur = _str(invoice.get("currency")).upper()
        if not lc_cur or not inv_cur:
            return LcRuleResult("lc_currency", "skipped", "low",
                                "Currency not found in LC or Invoice",
                                lc_value=lc_cur, doc_value=inv_cur, doc_type="invoice")
        if lc_cur == inv_cur:
            return LcRuleResult("lc_currency", "pass", "critical",
                                f"Currency matches: {lc_cur}",
                                lc_value=lc_cur, doc_value=inv_cur, doc_type="invoice")
        return LcRuleResult("lc_currency", "fail", "critical",
                            f"Currency mismatch: LC={lc_cur}, Invoice={inv_cur}",
                            lc_value=lc_cur, doc_value=inv_cur, doc_type="invoice")

    def _rule_amount_tolerance(self, lc, invoice) -> LcRuleResult:
        lc_amount  = _parse_amount(lc.get("amount"))
        inv_amount = _parse_amount(invoice.get("total_amount"))
        tol_plus   = int(lc.get("tolerance_plus")  or 0)
        tol_minus  = int(lc.get("tolerance_minus") or 0)
        if lc_amount is None or inv_amount is None:
            return LcRuleResult("lc_amount_tolerance", "skipped", "critical",
                                "Amount not found in LC or Invoice",
                                lc_value=str(lc_amount), doc_value=str(inv_amount), doc_type="invoice")
        upper = lc_amount * (1 + tol_plus  / 100)
        lower = lc_amount * (1 - tol_minus / 100)
        if lower <= inv_amount <= upper:
            dev = abs(inv_amount - lc_amount) / lc_amount * 100
            return LcRuleResult("lc_amount_tolerance", "pass", "critical",
                                f"Invoice amount {inv_amount:,.2f} within tolerance [{lower:,.2f}–{upper:,.2f}]",
                                detail=f"Deviation: {dev:.1f}%",
                                lc_value=f"{lc_amount:,.2f} ±{tol_plus}/{tol_minus}%",
                                doc_value=f"{inv_amount:,.2f}", doc_type="invoice")
        direction = "exceeds" if inv_amount > upper else "below"
        dev = abs(inv_amount - lc_amount) / lc_amount * 100
        return LcRuleResult("lc_amount_tolerance", "fail", "critical",
                            f"Invoice amount {direction} LC tolerance: {inv_amount:,.2f} not in [{lower:,.2f}–{upper:,.2f}]",
                            detail=f"Deviation: {dev:.1f}%",
                            lc_value=f"{lc_amount:,.2f} ±{tol_plus}/{tol_minus}%",
                            doc_value=f"{inv_amount:,.2f}", doc_type="invoice")

    def _rule_expiry_date(self, lc, invoice, bl) -> LcRuleResult:
        lc_exp = _parse_date(lc.get("expiry_date"))
        inv_dt = _invoice_date(invoice)
        bl_dt  = _bl_date(bl)
        if lc_exp is None:
            return LcRuleResult("lc_expiry_date", "skipped", "critical",
                                "LC expiry date not found")
        check_dt, doc_type = (inv_dt, "invoice") if inv_dt else (bl_dt, "bl")
        if check_dt is None:
            return LcRuleResult("lc_expiry_date", "skipped", "medium",
                                "Neither invoice date nor BL date found for expiry check",
                                lc_value=str(lc_exp))
        if check_dt <= lc_exp:
            return LcRuleResult("lc_expiry_date", "pass", "critical",
                                f"Document date {check_dt} within LC expiry {lc_exp}",
                                lc_value=str(lc_exp), doc_value=str(check_dt), doc_type=doc_type)
        return LcRuleResult("lc_expiry_date", "fail", "critical",
                            f"Document date {check_dt} AFTER LC expiry {lc_exp}",
                            lc_value=str(lc_exp), doc_value=str(check_dt), doc_type=doc_type)

    def _rule_shipment_date(self, lc, bl) -> LcRuleResult:
        lc_ship    = _parse_date(lc.get("shipment_date"))
        bl_onboard = _bl_date(bl)
        if lc_ship is None:
            return LcRuleResult("lc_shipment_date", "skipped", "high",
                                "LC shipment date not found")
        if bl_onboard is None:
            return LcRuleResult("lc_shipment_date", "skipped", "high",
                                "BL on-board date not found",
                                lc_value=str(lc_ship), doc_type="bl")
        if bl_onboard <= lc_ship:
            return LcRuleResult("lc_shipment_date", "pass", "critical",
                                f"BL on-board date {bl_onboard} ≤ LC latest shipment {lc_ship}",
                                lc_value=str(lc_ship), doc_value=str(bl_onboard), doc_type="bl")
        return LcRuleResult("lc_shipment_date", "fail", "critical",
                            f"BL on-board date {bl_onboard} AFTER LC latest shipment {lc_ship}",
                            lc_value=str(lc_ship), doc_value=str(bl_onboard), doc_type="bl")

    def _rule_presentation_period(self, lc, invoice, bl) -> LcRuleResult:
        days   = int(lc.get("presentation_period_days") or 21)
        bl_dt  = _bl_date(bl)
        inv_dt = _invoice_date(invoice)
        if bl_dt is None or inv_dt is None:
            return LcRuleResult("lc_presentation_period", "skipped", "high",
                                "Cannot check presentation period — BL or Invoice date missing")
        delta = (inv_dt - bl_dt).days
        if delta <= days:
            return LcRuleResult("lc_presentation_period", "pass", "high",
                                f"Presentation period OK: {delta} days ≤ {days} days allowed",
                                lc_value=f"{days} days", doc_value=f"{delta} days", doc_type="invoice")
        return LcRuleResult("lc_presentation_period", "fail", "high",
                            f"Presentation period exceeded: {delta} days > {days} days",
                            lc_value=f"{days} days", doc_value=f"{delta} days", doc_type="invoice")

    def _rule_incoterm(self, lc, invoice) -> LcRuleResult:
        lc_inc  = _str(lc.get("incoterm")).upper()
        inv_inc = _str(invoice.get("incoterm")).upper()
        if not lc_inc:
            return LcRuleResult("lc_incoterm", "skipped", "medium",
                                "LC incoterm not specified")
        if not inv_inc:
            return LcRuleResult("lc_incoterm", "warning", "medium",
                                f"Invoice incoterm not found (LC requires {lc_inc})",
                                lc_value=lc_inc, doc_type="invoice")
        if lc_inc == inv_inc or lc_inc in inv_inc or inv_inc in lc_inc:
            return LcRuleResult("lc_incoterm", "pass", "high",
                                f"Incoterm matches: {lc_inc}",
                                lc_value=lc_inc, doc_value=inv_inc, doc_type="invoice")
        return LcRuleResult("lc_incoterm", "fail", "high",
                            f"Incoterm mismatch: LC={lc_inc}, Invoice={inv_inc}",
                            lc_value=lc_inc, doc_value=inv_inc, doc_type="invoice")

    def _rule_partial_shipments(self, lc, bl) -> LcRuleResult:
        allowed = lc.get("partial_shipments_allowed")
        if allowed is None:
            return LcRuleResult("lc_partial_shipments", "skipped", "medium",
                                "LC partial shipments flag not set")
        bl_partial = _str(bl.get("partial_shipment") or bl.get("partial_shipments", "")).upper()
        if allowed:
            return LcRuleResult("lc_partial_shipments", "pass", "medium",
                                "Partial shipments are allowed by the LC",
                                lc_value="ALLOWED", doc_type="bl")
        if "ALLOW" in bl_partial and "NOT" not in bl_partial:
            return LcRuleResult("lc_partial_shipments", "warning", "high",
                                "BL indicates partial shipments but LC prohibits them",
                                lc_value="NOT ALLOWED", doc_value=bl_partial, doc_type="bl")
        return LcRuleResult("lc_partial_shipments", "pass", "medium",
                            "No partial shipments detected — consistent with LC restriction",
                            lc_value="NOT ALLOWED", doc_value=bl_partial or "not stated", doc_type="bl")

    def _rule_transhipment(self, lc, bl) -> LcRuleResult:
        allowed = lc.get("transhipment_allowed")
        if allowed is None:
            return LcRuleResult("lc_transhipment", "skipped", "medium",
                                "LC transhipment flag not set")
        bl_trans = _str(bl.get("transhipment") or "").upper()
        if allowed:
            return LcRuleResult("lc_transhipment", "pass", "medium",
                                "Transhipment is allowed by the LC",
                                lc_value="ALLOWED", doc_type="bl")
        if "ALLOW" in bl_trans and "NOT" not in bl_trans:
            return LcRuleResult("lc_transhipment", "fail", "high",
                                "BL indicates transhipment but LC prohibits it",
                                lc_value="NOT ALLOWED", doc_value=bl_trans, doc_type="bl")
        return LcRuleResult("lc_transhipment", "pass", "medium",
                            "No transhipment detected — consistent with LC restriction",
                            lc_value="NOT ALLOWED", doc_value=bl_trans or "not stated", doc_type="bl")

    def _rule_port_of_discharge(self, lc, bl, invoice) -> LcRuleResult:
        bl_pod  = _str(bl.get("port_of_discharge"))
        inv_pod = _str(invoice.get("port_of_destination"))
        if not bl_pod and not inv_pod:
            return LcRuleResult("lc_port_discharge", "skipped", "medium",
                                "Port of discharge not found in BL or Invoice")
        if bl_pod and inv_pod:
            if _port_match(bl_pod, inv_pod):
                return LcRuleResult("lc_port_discharge", "pass", "high",
                                    f"Port of discharge consistent: BL={bl_pod}, Invoice={inv_pod}",
                                    lc_value=bl_pod, doc_value=inv_pod, doc_type="invoice")
            return LcRuleResult("lc_port_discharge", "warning", "medium",
                                f"Port discrepancy: BL={bl_pod}, Invoice={inv_pod}",
                                lc_value=bl_pod, doc_value=inv_pod, doc_type="invoice")
        return LcRuleResult("lc_port_discharge", "skipped", "low",
                            "Only one port found — consistency check skipped",
                            doc_value=bl_pod or inv_pod)

    def _rule_doc_completeness(self, lc, bl, invoice, pl) -> LcRuleResult:
        required_raw = _str(lc.get("documents_required"))
        present: set[str] = set()
        if bl:      present.add("bill of lading")
        if invoice: present.add("invoice")
        if pl:      present.add("packing list")
        if not required_raw:
            missing = {"bill of lading", "invoice", "packing list"} - present
            if not missing:
                return LcRuleResult("lc_doc_completeness", "pass", "high",
                                    "All 3 standard commercial documents present", doc_type="all")
            return LcRuleResult("lc_doc_completeness", "warning", "high",
                                f"Missing documents: {', '.join(missing)}", doc_type="all")
        problems = []
        text = required_raw.lower()
        if "bill of lading" in text and not bl:    problems.append("Bill of Lading")
        if "invoice" in text and not invoice:       problems.append("Invoice")
        if "packing list" in text and not pl:       problems.append("Packing List")
        if problems:
            return LcRuleResult("lc_doc_completeness", "fail", "high",
                                f"Required documents missing: {', '.join(problems)}",
                                lc_value=required_raw[:120], doc_type="all")
        return LcRuleResult("lc_doc_completeness", "pass", "high",
                            "Required document types are present",
                            lc_value=required_raw[:120], doc_type="all")

    @staticmethod
    def _extract_charges(invoice: dict) -> float | None:
        """
        Tente d'extraire le total des frais (freight, insurance, handling…)
        depuis les champs plats de la facture.
        Retourne None si aucun champ de frais n'est trouvé.
        """
        _charge_keys = (
            "freight", "freight_amount", "freight_charges",
            "insurance", "insurance_amount",
            "handling", "handling_charges",
            "other_charges", "surcharge", "additional_charges",
            "shipping_charges", "transport", "transport_cost",
        )
        total: float | None = None
        for key in _charge_keys:
            val = _parse_amount(invoice.get(key))
            if val is not None:
                total = (total or 0.0) + val
        return total

    def _rule_invoice_items_total(self, inv_rows: list[dict], invoice: dict) -> LcRuleResult:
        """
        Réconciliation arithmétique interne : Σ lignes invoice ≈ total header.

        Gère correctement les factures CIF/CFR/CIP où :
          header_total = goods_subtotal + freight + insurance + autres frais
        → le Σ lignes-produits est légitimement < header_total.

        Cas couverts :
          1. Σlignes ≈ header (±1%)               → PASS (FOB/EXW)
          2. Σlignes + frais extraits ≈ header     → PASS (charges trouvées)
          3. Incoterm CIF/CFR/DDP + delta ≤ 20%   → PASS (charges attendues)
          4. Delta 0-25% sans contexte incoterm    → WARNING (frais possibles)
          5. Σlignes > header                      → FAIL (erreur arithmétique réelle)
          6. Delta > 25% inexpliqué               → WARNING (vérification manuelle)
        """
        header_total = _parse_amount(invoice.get("total_amount"))
        if not inv_rows:
            return LcRuleResult("inv_items_total", "skipped", "medium",
                                "Pas de lignes invoice_items disponibles pour réconciliation")
        items_total = self._sum_col(inv_rows, "line_amount", "amount", "total", "subtotal")
        if items_total is None:
            return LcRuleResult("inv_items_total", "skipped", "medium",
                                "Colonnes de montant non trouvées dans les lignes invoice_items")
        if header_total is None:
            return LcRuleResult("inv_items_total", "skipped", "medium",
                                "Montant total header facture non trouvé",
                                doc_value=f"Σlignes={items_total:,.2f}", doc_type="invoice")

        delta     = header_total - items_total          # positif = frais inclus dans header
        delta_pct = delta / max(abs(header_total), 1)  # ratio du header
        dev       = abs(delta_pct)

        # Incoterm context — CIF/CFR/CIP/CPT/DAP/DDP impliquent des frais dans le total
        incoterm = _str(invoice.get("incoterm", "")).upper()
        _cif_terms = ("CIF", "CFR", "CIP", "CPT", "DAP", "DAT", "DDP", "DPU")
        is_cif_style = any(t in incoterm for t in _cif_terms)

        # ── Cas 1 : correspondance directe (±1%) ──────────────────────
        if dev <= 0.01:
            return LcRuleResult("inv_items_total", "pass", "high",
                                f"Arithmétique cohérente : Σlignes {items_total:,.2f} ≈ total {header_total:,.2f}",
                                detail=f"Écart={dev:.2%}", doc_value=f"{items_total:,.2f}", doc_type="invoice")

        # ── Cas 2 : frais extraits expliquent le delta ────────────────
        charges = self._extract_charges(invoice)
        if charges is not None and charges > 0:
            recon    = items_total + charges
            dev_recon = abs(recon - header_total) / max(abs(header_total), 1)
            if dev_recon <= 0.02:
                return LcRuleResult("inv_items_total", "pass", "high",
                                    f"Cohérent avec frais : Σmarchandises {items_total:,.2f} + frais {charges:,.2f} ≈ total {header_total:,.2f}",
                                    detail=f"Frais extraits={charges:,.2f} · écart résiduel={dev_recon:.2%}",
                                    doc_value=f"{items_total:,.2f}", doc_type="invoice")
            if dev_recon <= 0.05:
                return LcRuleResult("inv_items_total", "warning", "low",
                                    f"Légère divergence même avec frais : {items_total:,.2f}+{charges:,.2f}={recon:,.2f} vs {header_total:,.2f}",
                                    detail=f"Écart résiduel={dev_recon:.2%}",
                                    doc_value=f"{items_total:,.2f}", doc_type="invoice")

        # ── Cas 3 : Σlignes < header + incoterm CIF/CFR/DDP ─────────
        if delta > 0 and delta_pct <= 0.20 and is_cif_style:
            return LcRuleResult("inv_items_total", "pass", "medium",
                                f"Δ={delta_pct:.1%} cohérent incoterm {incoterm} : frais transport/assurance inclus dans total",
                                detail=f"Σmarchandises={items_total:,.2f} · total {incoterm}={header_total:,.2f} · charges implicites≈{delta:,.2f}",
                                doc_value=f"{items_total:,.2f}", doc_type="invoice")

        # ── Cas 4 : Σlignes < header sans contexte incoterm (frais possibles)
        if 0 < delta_pct <= 0.25:
            return LcRuleResult("inv_items_total", "warning", "medium",
                                f"Σlignes {items_total:,.2f} < total {header_total:,.2f} (Δ={delta_pct:.1%}) — frais probables non détaillés",
                                detail=f"Delta={delta:,.2f}. Vérifier freight/insurance non extraits des lignes produits.",
                                doc_value=f"{items_total:,.2f}", doc_type="invoice")

        # ── Cas 5 : Σlignes > header → erreur arithmétique réelle ────
        if items_total > header_total:
            return LcRuleResult("inv_items_total", "fail", "high",
                                f"Erreur arithmétique : Σlignes {items_total:,.2f} SUPÉRIEUR au total {header_total:,.2f}",
                                detail=f"Excédent={abs(delta):,.2f} ({dev:.1%}) — impossible avec des frais additionnels",
                                doc_value=f"{items_total:,.2f}", doc_type="invoice")

        # ── Cas 6 : écart > 25% inexpliqué → avertissement manuel ───
        return LcRuleResult("inv_items_total", "warning", "medium",
                            f"Écart important non expliqué : Σlignes={items_total:,.2f} vs total={header_total:,.2f} ({dev:.1%})",
                            detail="Delta > 25% sans frais extraits ni incoterm CIF/CFR. Vérification manuelle recommandée.",
                            doc_value=f"{items_total:,.2f}", doc_type="invoice")

    def _rule_weight_consistency(self, bl_rows: list[dict], pl_rows: list[dict],
                                  bl: dict, pl: dict) -> LcRuleResult:
        bl_gw = (self._sum_col(bl_rows, "gross_weight_kg", "gross_weight")
                 or _parse_amount(bl.get("gross_weight_kg")))
        pl_gw = (self._sum_col(pl_rows, "gross_weight", "gross_weight_kg")
                 or _parse_amount(pl.get("gross_weight_kg")))
        src_bl = "tableau BL" if bl_rows else "champ BL"
        src_pl = "tableau PL" if pl_rows else "champ PL"
        if bl_gw is None or pl_gw is None:
            missing = "BL" if bl_gw is None else "PL"
            return LcRuleResult("weight_consistency", "skipped", "medium",
                                f"Poids brut non disponible dans {missing}",
                                doc_value=f"BL={bl_gw} PL={pl_gw}", doc_type="bl")
        dev = abs(bl_gw - pl_gw) / max(bl_gw, pl_gw)
        detail = f"{src_bl}={bl_gw:,.1f}kg · {src_pl}={pl_gw:,.1f}kg"
        if dev <= 0.02:
            return LcRuleResult("weight_consistency", "pass", "high",
                                f"Poids brut cohérent : BL={bl_gw:,.1f}kg ≈ PL={pl_gw:,.1f}kg",
                                detail=detail, doc_value=f"Δ={dev:.1%}", doc_type="bl")
        if dev <= 0.05:
            return LcRuleResult("weight_consistency", "warning", "medium",
                                f"Légère différence de poids : BL={bl_gw:,.1f}kg vs PL={pl_gw:,.1f}kg ({dev:.1%})",
                                detail=detail, doc_value=f"Δ={dev:.1%}", doc_type="bl")
        return LcRuleResult("weight_consistency", "fail", "high",
                            f"Incohérence de poids : BL={bl_gw:,.1f}kg vs PL={pl_gw:,.1f}kg ({dev:.1%})",
                            detail=detail, doc_value=f"Δ={dev:.1%}", doc_type="bl")

    def _rule_cbm_consistency(self, bl_rows: list[dict], pl_rows: list[dict],
                               bl: dict, pl: dict) -> LcRuleResult:
        bl_cbm = (self._sum_col(bl_rows, "cbm", "measurement")
                  or _parse_amount(bl.get("measurement") or bl.get("total_cbm")))
        pl_cbm = (self._sum_col(pl_rows, "cbm", "total_cbm")
                  or _parse_amount(pl.get("total_cbm") or pl.get("measurement")))
        if bl_cbm is None or pl_cbm is None:
            missing = "BL" if bl_cbm is None else "PL"
            return LcRuleResult("cbm_consistency", "skipped", "low",
                                f"CBM non disponible dans {missing}",
                                doc_value=f"BL={bl_cbm} PL={pl_cbm}", doc_type="bl")
        dev = abs(bl_cbm - pl_cbm) / max(bl_cbm, pl_cbm)
        detail = f"BL={bl_cbm:.2f}m³ · PL={pl_cbm:.2f}m³"
        if dev <= 0.02:
            return LcRuleResult("cbm_consistency", "pass", "medium",
                                f"CBM cohérent : BL={bl_cbm:.2f}m³ ≈ PL={pl_cbm:.2f}m³",
                                detail=detail, doc_value=f"Δ={dev:.1%}", doc_type="bl")
        if dev <= 0.05:
            return LcRuleResult("cbm_consistency", "warning", "low",
                                f"Légère différence CBM : BL={bl_cbm:.2f}m³ vs PL={pl_cbm:.2f}m³ ({dev:.1%})",
                                detail=detail, doc_value=f"Δ={dev:.1%}", doc_type="bl")
        return LcRuleResult("cbm_consistency", "fail", "medium",
                            f"Incohérence CBM : BL={bl_cbm:.2f}m³ vs PL={pl_cbm:.2f}m³ ({dev:.1%})",
                            detail=detail, doc_value=f"Δ={dev:.1%}", doc_type="bl")

    # ── Layer 2 : Hybrid Entity Matching (kept for _validate_parties) ──

    def _run_fuzzy(self, lc, bl, invoice, pl) -> list[dict]:
        results: list[dict] = []
        ben       = _str(lc.get("beneficiary"))
        applicant = _str(lc.get("applicant"))
        seller    = _str(invoice.get("seller"))
        buyer     = _str(invoice.get("buyer"))
        shipper   = _str(bl.get("shipper"))
        consignee = _str(bl.get("consignee"))

        def _add(check, lc_val, doc_val, doc_type):
            if not lc_val or not doc_val:
                return
            score    = _fuzzy_score(lc_val, doc_val)
            lc_name  = _extract_company_name(lc_val)
            doc_name = _extract_company_name(doc_val)
            results.append({
                "check":     check,
                "lc_value":  lc_val,
                "doc_value": doc_val,
                "lc_name":   lc_name,
                "doc_name":  doc_name,
                "doc_type":  doc_type,
                "score":     round(score, 3),
                "status":    "pass"    if score >= 0.70 else
                             "warning" if score >= 0.50 else "fail",
                "message":   f"{check.replace('_', ' ').title()}: {round(score * 100)}% match",
            })

        _add("beneficiary_vs_seller",    ben,       seller,    "invoice")
        _add("beneficiary_vs_shipper",   ben,       shipper,   "bl")
        _add("applicant_vs_buyer",       applicant, buyer,     "invoice")
        _add("applicant_vs_consignee",   applicant, consignee, "bl")
        return results

    # ── Layer 3 : Semantic NLP (kept for _validate_goods) ─────────────

    def _item_coverage_score(self, lc_goods: str, inv_rows: list[dict]) -> float | None:
        """
        Checks what fraction of invoice line item descriptions are present in the
        LC 45A text using keyword lookup.

        LC 45A often contains the SAME goods as the invoice but with added quantities,
        prices, HS codes and specifications. A sentence-transformer embedding of the
        full LC 45A text is "diluted" by that extra content, producing artificially
        low cosine similarity even when every product name is present.

        This method side-steps the embedding entirely: for each invoice item we check
        whether at least 50% of its significant words appear verbatim in the LC text.
        """
        if not lc_goods or not inv_rows:
            return None
        stop = {"OF", "THE", "AND", "IN", "AT", "TO", "FOR", "A", "AN",
                "WITH", "ON", "IS", "ARE", "BE", "BY"}
        lc_upper = lc_goods.upper()
        found = 0
        total = 0
        for row in inv_rows:
            desc = str(row.get("description") or row.get("goods_description") or "").strip()
            if len(desc) < 3:
                continue
            total += 1
            tokens = set(re.findall(r'\b\w{3,}\b', desc.upper())) - stop
            if not tokens:
                found += 1
                continue
            hit_rate = sum(1 for tok in tokens if tok in lc_upper) / len(tokens)
            if hit_rate >= 0.50:
                found += 1
        return (found / total) if total else None

    def _run_semantic(self, lc: dict, bl: dict, invoice: dict,
                      inv_rows: list[dict] | None = None,
                      bl_rows:  list[dict] | None = None) -> dict:
        lc_goods  = _str(lc.get("goods_description"))
        inv_goods = _str(invoice.get("goods_description"))
        bl_goods  = _str(bl.get("goods_description"))

        inv_goods_rich = self._enrich_goods_from_rows(
            inv_goods, inv_rows or [], "description", "goods_description", "package_type")
        bl_goods_rich  = self._enrich_goods_from_rows(
            bl_goods, bl_rows or [], "description", "commodity", "goods_description")

        inv_main = inv_goods_rich if inv_goods_rich else inv_goods
        bl_main  = bl_goods_rich  if bl_goods_rich  else bl_goods

        if not lc_goods:
            return {
                "overall_score":       0.80,
                "lc_vs_invoice":       None,
                "lc_vs_bl":            None,
                "lc_vs_invoice_items": None,
                "lc_vs_bl_containers": None,
                "note":                "LC goods description (45A) not provided",
            }

        try:
            scores = self._semantic_transformers(lc_goods, inv_main, bl_main)
        except Exception:
            scores = self._semantic_token_overlap(lc_goods, inv_main, bl_main)

        # Blend sentence-transformer score with item-level keyword coverage.
        # Item coverage is more robust when LC 45A contains the same products but
        # surrounded by quantities, prices and specs that dilute the embedding.
        item_cov = self._item_coverage_score(lc_goods, inv_rows or [])
        if item_cov is not None and "lc_vs_invoice" in scores:
            sem_score = scores["lc_vs_invoice"]
            blended   = round(0.40 * sem_score + 0.60 * item_cov, 4)
            scores["lc_vs_invoice"] = blended
            scores["lc_vs_invoice_label"] = (
                "MATCH"    if blended >= 0.70 else
                "PARTIAL"  if blended >= 0.45 else
                "MISMATCH"
            )
            scores["item_coverage_score"] = round(item_cov, 4)

        flat_inv_score: float | None = None
        flat_bl_score:  float | None = None
        if inv_main != inv_goods and inv_goods:
            try:
                _flat = self._semantic_transformers(lc_goods, inv_goods, "")
            except Exception:
                _flat = self._semantic_token_overlap(lc_goods, inv_goods, "")
            flat_inv_score = _flat.get("lc_vs_invoice")
        if bl_main != bl_goods and bl_goods:
            try:
                _flat = self._semantic_transformers(lc_goods, "", bl_goods)
            except Exception:
                _flat = self._semantic_token_overlap(lc_goods, "", bl_goods)
            flat_bl_score = _flat.get("lc_vs_bl")

        all_scores = [v for v in [
            scores.get("lc_vs_invoice"),
            scores.get("lc_vs_bl"),
        ] if v is not None]
        overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.70

        return {
            "overall_score":             overall,
            **scores,
            "lc_vs_invoice_items":       scores.get("lc_vs_invoice") if inv_main != inv_goods else None,
            "lc_vs_invoice_items_label": scores.get("lc_vs_invoice_label") if inv_main != inv_goods else None,
            "lc_vs_bl_containers":       scores.get("lc_vs_bl") if bl_main != bl_goods else None,
            "lc_vs_bl_containers_label": scores.get("lc_vs_bl_label")      if bl_main != bl_goods else None,
            "lc_vs_invoice_flat":        flat_inv_score,
            "lc_vs_bl_flat":             flat_bl_score,
            "inv_goods_used":            inv_main[:300]       if inv_main       else None,
            "bl_goods_used":             bl_main[:300]        if bl_main        else None,
            "inv_goods_enriched":        inv_goods_rich[:300] if inv_goods_rich else None,
            "bl_goods_enriched":         bl_goods_rich[:300]  if bl_goods_rich  else None,
        }

    _st_model_cache: Any = None

    def _get_st_model(self):
        if LcCommercialCrossValidator._st_model_cache is None:
            try:
                # Registre central : révision épinglée + CPU forcé (parité prod/local)
                from config.model_registry import load_sentence_transformer
                LcCommercialCrossValidator._st_model_cache = load_sentence_transformer("all-MiniLM-L6-v2")
            except ImportError:
                from sentence_transformers import SentenceTransformer
                LcCommercialCrossValidator._st_model_cache = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return LcCommercialCrossValidator._st_model_cache

    def _semantic_transformers(self, lc_goods: str, inv_goods: str, bl_goods: str) -> dict:
        from sentence_transformers import util
        model  = self._get_st_model()
        lc_emb = model.encode(lc_goods, convert_to_tensor=True)
        result: dict = {"method": "sentence_transformers"}
        if inv_goods:
            score = float(util.cos_sim(lc_emb, model.encode(inv_goods, convert_to_tensor=True)))
            result["lc_vs_invoice"] = round(score, 4)
            result["lc_vs_invoice_label"] = "MATCH" if score >= 0.60 else "PARTIAL" if score >= 0.35 else "MISMATCH"
        if bl_goods:
            score = float(util.cos_sim(lc_emb, model.encode(bl_goods, convert_to_tensor=True)))
            result["lc_vs_bl"] = round(score, 4)
            result["lc_vs_bl_label"] = "MATCH" if score >= 0.60 else "PARTIAL" if score >= 0.35 else "MISMATCH"
        return result

    def _semantic_token_overlap(self, lc_goods: str, inv_goods: str, bl_goods: str) -> dict:
        def sim(a: str, b: str):
            if not a or not b:
                return None
            stop  = {"OF", "THE", "AND", "IN", "AT", "TO", "FOR", "A", "AN", "WITH", "ON"}
            a_tok = set(re.findall(r'\b\w{3,}\b', a.upper())) - stop
            b_tok = set(re.findall(r'\b\w{3,}\b', b.upper())) - stop
            if not a_tok or not b_tok:
                return 0.0
            return len(a_tok & b_tok) / max(len(a_tok), len(b_tok))
        result: dict = {"method": "token_overlap"}
        v = sim(lc_goods, inv_goods)
        if v is not None:
            result["lc_vs_invoice"] = round(v, 4)
            result["lc_vs_invoice_label"] = "MATCH" if v >= 0.40 else "PARTIAL" if v >= 0.20 else "MISMATCH"
        v = sim(lc_goods, bl_goods)
        if v is not None:
            result["lc_vs_bl"] = round(v, 4)
            result["lc_vs_bl_label"] = "MATCH" if v >= 0.40 else "PARTIAL" if v >= 0.20 else "MISMATCH"
        return result

    # ── Layer 4 : Condition Parser (47A) ──────────────────────────────

    _CONDITION_PATTERNS: list[tuple] = [
        (r"full\s+set\s+of\s+(\d+)/(\d+)\s+original\s+bills?\s+of\s+lading",  "_cond_full_set_bl",       "critical"),
        (r"clean\s+on\s*[-\s]?board\s+bill\s+of\s+lading",                    "_cond_clean_on_board",    "critical"),
        (r"freight\s+(prepaid|collect)",                                        "_cond_freight_terms",     "high"),
        (r"notify\s+party[:\s]+(.{5,80}?)(?:\n|\.)",                           "_cond_notify_party",      "medium"),
        (r"port\s+of\s+loading[:\s]+([a-z][a-z\s,]{2,40}?)(?:\n|,|\.|and)",   "_cond_port_loading",      "high"),
        (r"port\s+of\s+discharge[:\s]+([a-z][a-z\s,]{2,40}?)(?:\n|,|\.|and)", "_cond_port_discharge_47", "high"),
        (r"latest\s+(?:date\s+of\s+)?shipment[:\s]+(\d[\d\s\w./]{4,15})",     "_cond_latest_ship",       "critical"),
        (r"documents?\s+(?:must\s+)?be\s+presented?\s+within\s+(\d+)\s+days?","_cond_presentation_days", "high"),
        (r"insurance\s+(?:certificate|policy|cover)\s+required",               "_cond_insurance",         "medium"),
        (r"certificate\s+of\s+origin",                                          "_cond_cert_origin",       "medium"),
        (r"packing\s+list\s+(?:is\s+)?required",                               "_cond_packing_list",      "high"),
        (r"all\s+documents?\s+(?:must\s+be\s+)?(?:issued?\s+)?in\s+(english|french|arabic)", "_cond_language", "medium"),
    ]

    def _run_conditions(self, lc, bl, invoice, pl) -> list[LcRuleResult]:
        text = _str(lc.get("additional_conditions"))
        if not text:
            return [LcRuleResult("lc_47a_conditions", "skipped", "info",
                                 "No additional conditions (47A) in LC")]
        results: list[LcRuleResult] = []
        tl = text.lower()
        for pattern, handler_name, severity in self._CONDITION_PATTERNS:
            m = re.search(pattern, tl)
            if m:
                fn = getattr(self, handler_name, None)
                if fn:
                    r = fn(m, lc, bl, invoice, pl, severity)
                    if r:
                        results.append(r)
        if not results:
            results.append(LcRuleResult("lc_47a_conditions", "skipped", "info",
                                        f"47A conditions present ({len(text)} chars) — no verifiable patterns detected"))
        return results

    def _cond_full_set_bl(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        copies    = int(m.group(1))
        total     = int(m.group(2))
        bl_copies = _str(bl.get("copies") or bl.get("number_of_originals") or "")
        msg = f"LC requires full set {copies}/{total} original BL"
        if not bl_copies:
            return LcRuleResult("lc_cond_bl_full_set", "warning", severity,
                                msg + " — BL copies count not verifiable from extracted fields",
                                lc_value=f"{copies}/{total}", doc_type="bl")
        return LcRuleResult("lc_cond_bl_full_set", "pass", severity,
                            msg + f" — BL states: {bl_copies}",
                            lc_value=f"{copies}/{total}", doc_value=bl_copies, doc_type="bl")

    def _cond_clean_on_board(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        bl_status = _str(bl.get("bl_type") or bl.get("on_board") or "").upper()
        if "FOUL" in bl_status or "UNCLEAN" in bl_status:
            return LcRuleResult("lc_cond_clean_on_board", "fail", severity,
                                f"BL is not clean: {bl_status}",
                                lc_value="CLEAN ON BOARD", doc_value=bl_status, doc_type="bl")
        return LcRuleResult("lc_cond_clean_on_board", "pass", severity,
                            "Clean on-board BL condition — no adverse notation detected",
                            lc_value="CLEAN ON BOARD", doc_value=bl_status or "standard", doc_type="bl")

    def _cond_freight_terms(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        required   = m.group(1).upper()
        bl_freight = _str(bl.get("freight") or bl.get("freight_terms") or "").upper()
        if not bl_freight:
            return LcRuleResult("lc_cond_freight", "warning", severity,
                                f"LC requires freight {required} — not verifiable from BL",
                                lc_value=required, doc_type="bl")
        if required in bl_freight:
            return LcRuleResult("lc_cond_freight", "pass", severity,
                                f"Freight {required} confirmed on BL",
                                lc_value=required, doc_value=bl_freight, doc_type="bl")
        return LcRuleResult("lc_cond_freight", "fail", severity,
                            f"Freight terms mismatch: LC={required}, BL={bl_freight}",
                            lc_value=required, doc_value=bl_freight, doc_type="bl")

    def _cond_notify_party(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        required  = m.group(1).strip()
        bl_notify = _str(bl.get("notify_party") or "")
        if not bl_notify:
            return LcRuleResult("lc_cond_notify", "warning", severity,
                                f"LC requires notify party '{required}' — not found in BL",
                                lc_value=required, doc_type="bl")
        score = _fuzzy_score(required, bl_notify)
        if score >= 0.60:
            return LcRuleResult("lc_cond_notify", "pass", severity,
                                f"Notify party match ({round(score*100)}%): {required}",
                                lc_value=required, doc_value=bl_notify, doc_type="bl")
        return LcRuleResult("lc_cond_notify", "warning", severity,
                            f"Notify party mismatch ({round(score*100)}%): LC={required}, BL={bl_notify}",
                            lc_value=required, doc_value=bl_notify, doc_type="bl")

    def _cond_port_loading(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        required = m.group(1).strip()
        bl_pol   = _str(bl.get("port_of_loading") or "")
        if not bl_pol:
            return LcRuleResult("lc_cond_port_loading", "warning", severity,
                                f"LC 47A requires port of loading '{required}' — not found in BL",
                                lc_value=required, doc_type="bl")
        if _port_match(required, bl_pol):
            return LcRuleResult("lc_cond_port_loading", "pass", severity,
                                f"Port of loading matches 47A: {required}",
                                lc_value=required, doc_value=bl_pol, doc_type="bl")
        return LcRuleResult("lc_cond_port_loading", "warning", severity,
                            f"Port of loading 47A mismatch: LC={required}, BL={bl_pol}",
                            lc_value=required, doc_value=bl_pol, doc_type="bl")

    def _cond_port_discharge_47(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        required = m.group(1).strip()
        bl_pod   = _str(bl.get("port_of_discharge") or "")
        if not bl_pod:
            return LcRuleResult("lc_cond_port_discharge", "warning", severity,
                                f"LC 47A requires port of discharge '{required}' — not found in BL",
                                lc_value=required, doc_type="bl")
        if _port_match(required, bl_pod):
            return LcRuleResult("lc_cond_port_discharge", "pass", severity,
                                f"Port of discharge matches 47A: {required}",
                                lc_value=required, doc_value=bl_pod, doc_type="bl")
        return LcRuleResult("lc_cond_port_discharge", "warning", severity,
                            f"Port of discharge 47A mismatch: LC={required}, BL={bl_pod}",
                            lc_value=required, doc_value=bl_pod, doc_type="bl")

    def _cond_latest_ship(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        raw      = m.group(1).strip()
        req_date = _parse_date(raw)
        bl_dt    = _bl_date(bl)
        if req_date is None:
            return LcRuleResult("lc_cond_latest_ship", "skipped", severity,
                                f"Cannot parse 47A latest shipment date: {raw}")
        if bl_dt is None:
            return LcRuleResult("lc_cond_latest_ship", "warning", severity,
                                f"47A latest shipment {req_date} — BL date not found",
                                lc_value=str(req_date), doc_type="bl")
        if bl_dt <= req_date:
            return LcRuleResult("lc_cond_latest_ship", "pass", severity,
                                f"BL date {bl_dt} ≤ 47A latest shipment {req_date}",
                                lc_value=str(req_date), doc_value=str(bl_dt), doc_type="bl")
        return LcRuleResult("lc_cond_latest_ship", "fail", severity,
                            f"BL date {bl_dt} AFTER 47A latest shipment {req_date}",
                            lc_value=str(req_date), doc_value=str(bl_dt), doc_type="bl")

    def _cond_presentation_days(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        days   = int(m.group(1))
        bl_dt  = _bl_date(bl)
        inv_dt = _invoice_date(invoice)
        if bl_dt is None or inv_dt is None:
            return LcRuleResult("lc_cond_pres_days", "skipped", severity,
                                f"47A requires presentation within {days} days — dates not found")
        delta = (inv_dt - bl_dt).days
        if delta <= days:
            return LcRuleResult("lc_cond_pres_days", "pass", severity,
                                f"Presentation within 47A window: {delta} days ≤ {days}",
                                lc_value=f"{days} days", doc_value=f"{delta} days", doc_type="invoice")
        return LcRuleResult("lc_cond_pres_days", "fail", severity,
                            f"Presentation exceeds 47A window: {delta} days > {days}",
                            lc_value=f"{days} days", doc_value=f"{delta} days", doc_type="invoice")

    def _cond_insurance(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        return LcRuleResult("lc_cond_insurance", "warning", severity,
                            "LC requires insurance certificate — not verifiable from extracted fields",
                            lc_value="REQUIRED", doc_type="all")

    def _cond_cert_origin(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        return LcRuleResult("lc_cond_cert_origin", "warning", severity,
                            "LC requires certificate of origin — not verifiable from extracted fields",
                            lc_value="REQUIRED", doc_type="all")

    def _cond_packing_list(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        if pl:
            return LcRuleResult("lc_cond_packing_list", "pass", severity,
                                "Packing list required by LC and is present",
                                lc_value="REQUIRED", doc_value="PRESENT", doc_type="packing_list")
        return LcRuleResult("lc_cond_packing_list", "fail", severity,
                            "Packing list required by LC but NOT provided",
                            lc_value="REQUIRED", doc_value="MISSING", doc_type="packing_list")

    def _cond_language(self, m, lc, bl, invoice, pl, severity) -> LcRuleResult:
        lang = m.group(1).upper()
        return LcRuleResult("lc_cond_language", "warning", severity,
                            f"LC requires documents in {lang} — cannot verify from extracted fields",
                            lc_value=lang, doc_type="all")

    # ── Utils ──────────────────────────────────────────────────────────

    @staticmethod
    def _rr_to_dict(r: LcRuleResult) -> dict:
        return {
            "rule":      r.rule,
            "status":    r.status,
            "severity":  r.severity,
            "message":   r.message,
            "detail":    r.detail,
            "lc_value":  r.lc_value,
            "doc_value": r.doc_value,
            "doc_type":  r.doc_type,
        }
