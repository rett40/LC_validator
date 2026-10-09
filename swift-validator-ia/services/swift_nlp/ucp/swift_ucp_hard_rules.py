"""
Règles UCP 600 déterministes (liste rouge) sur le texte agrégé du crédit.
Complète le moteur sémantique : ne dépend pas des embeddings.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from .swift_ucp_clause_aligner import normalize_clause

logger = logging.getLogger(__name__)

_BLOCKLIST_PATH = Path(__file__).resolve().parent / "data" / "ucp_hard_blocklist.json"

_critical_excl_cache: Optional[frozenset[str]] = None
_hard_codes_cache: Optional[frozenset[str]] = None
_compiled_patterns_cache: Optional[list[tuple[re.Pattern, dict[str, Any]]]] = None


def _load_blocklist_raw() -> dict[str, Any]:
    if not _BLOCKLIST_PATH.exists():
        return {}
    try:
        with open(_BLOCKLIST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("ucp_hard_blocklist.json: %s", exc)
        return {}


def load_critical_exclusion_articles() -> frozenset[str]:
    global _critical_excl_cache
    if _critical_excl_cache is not None:
        return _critical_excl_cache
    data = _load_blocklist_raw()
    arts = data.get("critical_exclusion_articles") or ["Art.18", "Art.38i"]
    _critical_excl_cache = frozenset(str(a) for a in arts)
    return _critical_excl_cache


def load_hard_block_codes() -> frozenset[str]:
    global _hard_codes_cache
    if _hard_codes_cache is not None:
        return _hard_codes_cache
    data = _load_blocklist_raw()
    codes = data.get("hard_block_codes") or []
    extra = {p.get("code") for p in (data.get("patterns") or []) if p.get("code")}
    _hard_codes_cache = frozenset(str(c) for c in codes) | frozenset(c for c in extra if c)
    return _hard_codes_cache


def _compiled_patterns() -> list[tuple[re.Pattern, dict[str, Any]]]:
    global _compiled_patterns_cache
    if _compiled_patterns_cache is not None:
        return _compiled_patterns_cache
    data = _load_blocklist_raw()
    rows = data.get("patterns") or []
    out: list[tuple[re.Pattern, dict[str, Any]]] = []
    for p in rows:
        try:
            out.append((re.compile(p["regex"]), p))
        except re.error as exc:
            logger.warning("Pattern UCP hard ignoré: %s — %s", p.get("id"), exc)
    _compiled_patterns_cache = out
    return out


def build_aggregate_lc_text(ocr_text: str, clauses: list[dict], max_chars: int = 500_000) -> str:
    parts: list[str] = []
    if ocr_text and str(ocr_text).strip():
        parts.append(str(ocr_text).strip())
    for c in clauses or []:
        t = (c.get("text") or "").strip()
        if t:
            parts.append(t)
    full = "\n".join(parts)
    return full[:max_chars] if len(full) > max_chars else full


def evaluate_hard_rules(ocr_text: str, clauses: list[dict]) -> list[dict[str, Any]]:
    """
    Retourne des dicts compatibles _ucp_finding_from_dict.
    Les exclusions contractuelles ne sont pas des violations UCP automatiques:
    elles sont signalées en contractual_modification (warning), sauf cas structurels.
    Déduplication par code.
    """
    full = build_aggregate_lc_text(ocr_text, clauses)
    if len(normalize_clause(full)) < 8:
        return []

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rx, meta in _compiled_patterns():
        if not rx.search(full):
            continue
        code = str(meta.get("code") or "UCP_HARD_RULE")
        out_code = code
        sev = "error"
        nature = "true_violation"
        if code.startswith("UCP_HARD_EXCL_"):
            out_code = code.replace("UCP_HARD_EXCL_", "UCP_HARD_CONTRACT_")
            sev = "warning"
            nature = "contractual_modification"
        if code in seen:
            continue
        seen.add(code)
        art = str(meta.get("article") or "UCP")
        msg = str(meta.get("message") or "Règle UCP déterministe déclenchée")
        cat = str(meta.get("category") or "compliance")
        m = rx.search(full)
        span = full[m.start() : m.end()][:200] if m else ""
        out.append(
            {
                "article": art,
                "article_title": art,
                "code": out_code,
                "severity": sev,
                "category": cat,
                "message": msg,
                "detail": f"Motif déterministe ({meta.get('id', code)}).",
                "clause_text": span,
                "recommendation": "Revoir la formulation : exclusion d'article UCP standard est hautement risquée.",
                "confidence": 0.99,
                "exclusion_type": "hard_rule",
                "finding_nature": nature,
            }
        )
    return out


def findings_from_critical_exclusion_map(
    exclusion_map: dict[str, str],
    article_titles: dict[str, str],
) -> list[dict[str, Any]]:
    """Une finding par article critique présent dans exclusion_map."""
    critical = load_critical_exclusion_articles()
    out: list[dict[str, Any]] = []
    for art_key, snip in (exclusion_map or {}).items():
        if art_key not in critical:
            continue
        slug = art_key.replace("Art.", "ART").upper().replace(".", "")
        code = f"SEM_CONTRACT_STRUCT_{slug}"
        title = article_titles.get(art_key, art_key)
        out.append(
            {
                "article": art_key,
                "article_title": title,
                "code": code,
                "severity": "warning",
                "category": "compliance" if art_key.startswith("Art.38") else "document",
                "message": f"Exclusion explicite {art_key} UCP 600 détectée — risque contractuel majeur.",
                "detail": "Article déclaré non applicable / exclu selon art.1 UCP 600 (modification contractuelle).",
                "clause_text": (snip or "")[:200],
                "recommendation": "Confirmer l'intention avec la banque émettrice et la conformité interne.",
                "confidence": 0.97,
                "exclusion_type": "critical_exclusion",
                "finding_nature": "contractual_modification",
            }
        )
    return out


def merge_hard_findings(
    base: list[dict[str, Any]],
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Évite les doublons par code ; préserve l'ordre base puis extra."""
    seen = {d.get("code") for d in base if d.get("code")}
    merged = list(base)
    for d in extra:
        c = d.get("code")
        if c and c not in seen:
            seen.add(c)
            merged.append(d)
    return merged


def dedupe_ucp_finding_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Évite la double remontée Art.18 / Art.38i (ex. SEM_CONTRACT_* + UCP_HARD_CONTRACT_*).
    Si une règle dure existe pour l'article, on retire le finding sémantique équivalent.
    """
    if not rows:
        return rows
    has_hard_18 = any(d.get("code") in {"UCP_HARD_EXCL_ART18", "UCP_HARD_CONTRACT_ART18"} for d in rows)
    has_hard_38i = any(d.get("code") in {"UCP_HARD_EXCL_ART38I", "UCP_HARD_CONTRACT_ART38I"} for d in rows)
    filtered: list[dict[str, Any]] = []
    for d in rows:
        c = d.get("code", "")
        if has_hard_18 and c in {"SEM_VIOL_ART18", "SEM_CONTRACT_STRUCT_ART18"}:
            continue
        if has_hard_38i and c in {"SEM_VIOL_ART38I", "SEM_CONTRACT_STRUCT_ART38I"}:
            continue
        filtered.append(d)
    seen_code: set[str] = set()
    out: list[dict[str, Any]] = []
    for d in filtered:
        co = d.get("code")
        if co and co in seen_code:
            continue
        if co:
            seen_code.add(co)
        out.append(d)
    return out


def evaluate_ucp_decision(findings: List[Any], payment_blocked: bool) -> str:
    """
    Décision bancaire UCP 600 à 5 niveaux :
      REJECT            — violation UCP réelle bloquant le paiement
      CONDITIONAL_HOLD  — modification contractuelle Art.1 avec risque structurel élevé
      REVIEW            — anomalie ou erreur sans blocage paiement direct
      ACCEPT_WITH_FLAGS — alertes mineures / modifications contractuelles faibles
      ACCEPT            — aucune anomalie détectée

    Logique :
      true_violation + blocks_payment → REJECT
      contractual_modification + blocks_payment → CONDITIONAL_HOLD  (Art.1 UCP autorise)
      true_violation severity=error → REVIEW
      operational_discrepancy severity=error → REVIEW
      contractual_modification severity=warning → ACCEPT_WITH_FLAGS
      severity=info uniquement → ACCEPT
    """
    hard = load_hard_block_codes()

    # REJECT : violation UCP réelle ou discordance documentaire + blocage paiement
    for f in findings:
        nature = getattr(f, "finding_nature", "true_violation")
        code   = getattr(f, "code", "") or ""
        blocks = getattr(f, "blocks_payment", False)
        sev    = getattr(f, "severity", None)
        if nature in {"true_violation", "operational_discrepancy"}:
            if (blocks or code in hard) and sev == "error":
                return "REJECT"

    if payment_blocked:
        return "REJECT"

    # CONDITIONAL_HOLD : modification contractuelle Art.1 à risque élevé (≠ violation UCP)
    for f in findings:
        nature = getattr(f, "finding_nature", "true_violation")
        blocks = getattr(f, "blocks_payment", False)
        if nature == "contractual_modification" and blocks:
            return "CONDITIONAL_HOLD"

    # REVIEW : erreur vraie sans blocage paiement
    for f in findings:
        sev    = getattr(f, "severity", None)
        nature = getattr(f, "finding_nature", "true_violation")
        if sev == "error" and nature in {"true_violation", "operational_discrepancy"}:
            return "REVIEW"

    # ACCEPT_WITH_FLAGS : alertes / modifications contractuelles non bloquantes
    for f in findings:
        sev = getattr(f, "severity", None)
        if sev in {"warning", "error"}:
            return "ACCEPT_WITH_FLAGS"

    return "ACCEPT"


def evaluate_cross_field_rules(fields: dict[str, str]) -> list[dict[str, Any]]:
    """
    Valide la cohérence inter-champs SWIFT MT700.
    fields = {"expiry_date": "2025-12-01", "latest_shipment_date": "2025-11-15", ...}
    Charge les règles depuis cross_field_consistency_rules du JSON.
    """
    data = _load_blocklist_raw()
    rules = data.get("cross_field_consistency_rules") or []
    out: list[dict[str, Any]] = []
    for rule in rules:
        fa_key = rule.get("field_a", "")
        fb_key = rule.get("field_b", "")
        fa = (fields.get(fa_key) or "").strip()
        fb = (fields.get(fb_key) or "").strip()
        if not fa or not fb:
            continue
        op = rule.get("operator", "less_than")
        violated = False
        try:
            if op == "less_than":       # fa < fb (ex: shipment_date < expiry_date)
                violated = fa >= fb
            elif op == "greater_than":  # fa > fb
                violated = fa <= fb
            elif op == "not_equal":
                violated = fa == fb
            elif op == "not_empty":
                violated = not fa
        except Exception:
            continue
        if not violated:
            continue
        out.append({
            "article":       rule.get("article", "UCP"),
            "article_title": rule.get("article", "UCP"),
            "code":          rule.get("code", f"CROSS_{fa_key.upper()}_{fb_key.upper()}"),
            "severity":      rule.get("severity", "error"),
            "category":      "compliance",
            "message":       rule.get("message", f"Incohérence inter-champs : {fa_key} vs {fb_key}"),
            "detail":        f"{fa_key}={fa} / {fb_key}={fb}",
            "clause_text":   "",
            "recommendation": rule.get("remediation", "Corriger les champs SWIFT en contradiction."),
            "confidence":    0.99,
            "exclusion_type": "cross_field",
            "finding_nature": "operational_discrepancy",
        })
    return out


def ucp_blocking_violations_list(findings: List[Any]) -> list[dict[str, Any]]:
    """Synthèse des violations bloquantes pour l'UI / API (distinct du risque ML)."""
    hard = load_hard_block_codes()
    out: list[dict[str, Any]] = []
    for f in findings:
        if getattr(f, "severity", None) != "error":
            continue
        code = getattr(f, "code", "") or ""
        if not (getattr(f, "blocks_payment", False) or code in hard):
            continue
        ct = getattr(f, "clause_text", None) or ""
        out.append(
            {
                "article": getattr(f, "article", ""),
                "article_title": getattr(f, "article_title", ""),
                "code": code,
                "severity": "BLOCK",
                "reason": getattr(f, "message", ""),
                "clause_text": ct[:200] if isinstance(ct, str) else "",
                "recommendation": getattr(f, "recommendation", "") or "",
            }
        )
    return out
