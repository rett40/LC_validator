"""
swift_ucp_semantic_engine.py
============================
Moteur sémantique UCP : alignement clause ↔ article + classification
COMPLIANT / RESTRICTION / VIOLATION / CONTRACT_MODIFICATION / STRUCTURAL_MODIFICATION.
Sortie en dicts sérialisables pour UCP600Validator.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .swift_ucp_clause_aligner import (
    ClauseAlignment,
    UCPAlignerConfig,
    UCPClauseAligner,
    normalize_clause,
)
from .swift_ucp_nli import UCPNLIRefiner
from .swift_ucp_hard_rules import (
    dedupe_ucp_finding_dicts,
    findings_from_critical_exclusion_map,
    load_critical_exclusion_articles,
    merge_hard_findings,
)

logger = logging.getLogger(__name__)

# ── Exclusions explicites (pas de fenêtre « collated » arbitraire) ─────
_EXCLUSION_STRICT_RE = re.compile(
    r"[Aa]rt(?:icle)?\.?\s*(\d{1,2}[a-z]?)"
    r"(?:\s+of\s+UCP\s*(?:600|\(2007\s+REVISION\)|LATEST\s+VERSION)?)?"
    r"\s+(?:is\s+)?(?:not\s+applicable|excluded|is\s+excluded)",
    re.IGNORECASE,
)
_EXCLUSION_STRICT_RE_FR = re.compile(
    r"(?:Article|Art)\.?\s*(\d{1,2}[a-z]?)\b"
    r"[\s\S]{0,140}?"
    r"(?:non\s+applicable|n'est\s+pas\s+applicable|est\s+exclu(?:e)?|ne\s+s'applique\s+pas)",
    re.IGNORECASE,
)
_EXCLUSION_STRICT_RE_EN_GAP = re.compile(
    r"[Aa]rt(?:icle)?\.?\s*(\d{1,2}[a-z]?)"
    r"(?:\s+of\s+UCP\s*(?:600|\(2007\s+REVISION\)|LATEST\s+VERSION)?)?"
    r"[\s\S]{0,120}?"
    r"(?:does\s+not\s+apply|shall\s+not\s+apply)",
    re.IGNORECASE,
)

_EXCLUSION_STRICT_ALL: tuple[re.Pattern, ...] = (
    _EXCLUSION_STRICT_RE,
    _EXCLUSION_STRICT_RE_FR,
    _EXCLUSION_STRICT_RE_EN_GAP,
)


def parse_strict_exclusions(full_text: str, clauses: list[dict]) -> dict[str, str]:
    """Détecte les formulations explicites d'exclusion / non-applicabilité d'article UCP."""
    exclusions: dict[str, str] = {}
    sources = [(c.get("text") or "") for c in clauses]
    if full_text:
        sources.append(full_text)
    for source in sources:
        for rx in _EXCLUSION_STRICT_ALL:
            for m in rx.finditer(source):
                art_num = m.group(1).lower()
                art_key = f"Art.{art_num}"
                if art_key not in exclusions:
                    ctx_start = max(0, m.start() - 120)
                    ctx_end = min(len(source), m.end() + 120)
                    exclusions[art_key] = source[ctx_start:ctx_end].strip()
    return exclusions


def _exclusion_match_in_clause(text: str) -> Optional[re.Match]:
    for rx in _EXCLUSION_STRICT_ALL:
        m = rx.search(text)
        if m:
            return m
    return None


def _all_excluded_articles_in_text(text: str) -> list[str]:
    """Tous les articles mentionnés comme exclus/non applicables dans ce texte."""
    out: list[str] = []
    seen_set: set[str] = set()
    for rx in _EXCLUSION_STRICT_ALL:
        for m in rx.finditer(text):
            art = f"Art.{m.group(1).lower()}"
            if art not in seen_set:
                seen_set.add(art)
                out.append(art)
    return out


def _article_code_slug(article: str) -> str:
    return article.replace("Art.", "ART").upper().replace(".", "")


# Patterns : clause renforce une exigence type banque prudente (souvent alignée UCP)
_COMPLIANT_HINTS: dict[str, list[re.Pattern]] = {
    "Art.28": [
        re.compile(r"no\s+franchise|without\s+franchise|zero\s+franchise", re.I),
        re.compile(r"no\s+deductible|without\s+deductible|zero\s+deductible", re.I),
        re.compile(r"110\s*%\s*(?:of\s+)?(?:invoice|credit)|minimum\s+cover", re.I),
    ],
    "Art.17": [
        re.compile(r"original\s+document|must\s+be\s+original", re.I),
        re.compile(r"scanned\s+(?:copy|document).{0,40}(?:not\s+acceptable|prohibited|forbidden)", re.I),
        re.compile(r"(?:not\s+acceptable|prohibited).{0,30}scanned", re.I),
    ],
    "Art.18": [
        re.compile(r"invoice\s+must\s+be\s+issued\s+by\s+beneficiary", re.I),
        re.compile(r"third\s+party.{0,20}(?:not\s+acceptable|prohibited)", re.I),
    ],
    "Art.20": [
        re.compile(r"charter\s*[-\s]?party.{0,30}(?:not\s+acceptable|prohibited)", re.I),
        re.compile(r"blank\s+back.{0,40}(?:not\s+acceptable|prohibited)", re.I),
    ],
    "Art.20c": [
        re.compile(r"tranship(?:ment)?\s+(?:is\s+)?prohibit|no\s+tranship", re.I),
    ],
}

# Patterns : demande typiquement incompatible ou plus permissive que pratique stricte
_VIOLATION_HINTS: dict[str, list[re.Pattern]] = {
    "Art.28": [
        re.compile(r"franchise\s+of\s*\d|deductible\s+of\s*\d|\d+\s*%\s*franchise", re.I),
        re.compile(r"excess\s+deductible\s+of|minimum\s+franchise\s+of", re.I),
    ],
    "Art.17": [
        re.compile(r"scanned\s+(?:copy|document).{0,40}(?:acceptable|allowed|permitted)", re.I),
        re.compile(r"cop(?:y|ies)\s+instead\s+of\s+original", re.I),
    ],
    "Art.18": [
        re.compile(r"third\s+party\s+invoice.{0,30}(?:acceptable|allowed|permitted)", re.I),
        re.compile(r"formerly\s+known\s+as", re.I),
    ],
    "Art.14k": [
        re.compile(r"shipper\s+other\s+than\s+beneficiary.{0,30}(?:acceptable|allowed)", re.I),
    ],
}

_RESTRICTIVE_GENERIC = re.compile(
    r"must\s+not|shall\s+not|strictly\s+required|only\s+.*\s+acceptable|"
    r"not\s+acceptable\s+unless|interdit|sauf\s+accord",
    re.I,
)


def _patterns_match(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _hint_patterns(article: str, d: dict[str, list[re.Pattern]]) -> list[re.Pattern]:
    if article in d:
        return d[article]
    m = re.match(r"(Art\.\d+)", article)
    if m and m.group(1) in d:
        return d[m.group(1)]
    return []


def classify_relation(
    alignment: ClauseAlignment,
    exclusion_map: dict[str, str],
    cfg: UCPAlignerConfig,
    nli: Optional[UCPNLIRefiner] = None,
) -> tuple[str, float, Optional[str]]:
    """
    Retourne (
      COMPLIANT | RESTRICTION | VIOLATION | CONTRACT_MODIFICATION | STRUCTURAL_MODIFICATION,
      confidence,
      article_finding
    ).
    article_finding : article UCP à attribuer au finding si différent du best_article (exclusions).
    Les exclusions sont évaluées avant le seuil de similarité embedding.
    """
    text = alignment.normalized
    art = alignment.best_article
    sim = alignment.similarity
    critical = load_critical_exclusion_articles()

    for ea in _all_excluded_articles_in_text(text):
        if ea in critical:
            return "STRUCTURAL_MODIFICATION", 0.97, ea

    ex_m = _exclusion_match_in_clause(text)
    if ex_m:
        matched = f"Art.{ex_m.group(1).lower()}"
        if matched == art or art.lower().startswith(matched.lower()):
            return "CONTRACT_MODIFICATION", 0.94, matched
        num_m = re.match(r"Art\.(\d+)", matched)
        num_a = re.match(r"Art\.(\d+)", art)
        if num_m and num_a and num_m.group(1) == num_a.group(1):
            return "CONTRACT_MODIFICATION", 0.88, matched
        return "CONTRACT_MODIFICATION", 0.86, matched

    for ex_art, snip in (exclusion_map or {}).items():
        if ex_art not in critical or not snip or len(snip) < 12:
            continue
        core = normalize_clause(snip[: min(200, len(snip))])
        if len(core) > 12 and core in text:
            return "STRUCTURAL_MODIFICATION", 0.95, ex_art

    if sim < cfg.sim_min_align:
        return "COMPLIANT", 0.88, None

    excl_snip = exclusion_map.get(art, "")
    if excl_snip and len(excl_snip) > 12:
        core = normalize_clause(excl_snip[: min(100, len(excl_snip))])
        if len(core) > 12 and core in text:
            return "CONTRACT_MODIFICATION", 0.9, art

    v_hints = _hint_patterns(art, _VIOLATION_HINTS)
    c_hints = _hint_patterns(art, _COMPLIANT_HINTS)

    if v_hints and _patterns_match(v_hints, text) and sim >= cfg.sim_restriction:
        return "VIOLATION", min(0.95, 0.65 + sim * 0.3), art

    if c_hints and _patterns_match(c_hints, text) and sim >= cfg.sim_min_align:
        return "COMPLIANT", min(0.95, 0.6 + sim * 0.35), None

    if nli and nli.enabled and sim >= cfg.sim_restriction:
        premise = (alignment.raw_text or text).strip()
        refined = nli.refine(premise, art, alignment.best_title)
        if refined and refined[0] != "COMPLIANT":
            return refined[0], refined[1], art

    if sim >= cfg.sim_strong and _RESTRICTIVE_GENERIC.search(text):
        return "RESTRICTION", min(0.9, 0.55 + sim * 0.3), None

    if sim >= cfg.sim_restriction and _RESTRICTIVE_GENERIC.search(text):
        return "RESTRICTION", min(0.78, 0.48 + sim * 0.28), None

    return "COMPLIANT", min(0.85, 0.55 + (1.0 - sim) * 0.4), None


@dataclass
class SemanticUCPEngineResult:
    """Résultat brut du moteur sémantique."""

    per_clause: list[dict[str, Any]] = field(default_factory=list)
    finding_dicts: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    available: bool = True


def _finding_message(
    article: str, relation: str, title: str
) -> tuple[str, str, str, str, str]:
    """message, detail, recommendation, category, finding_nature"""
    cat = "document"
    if article in ("Art.20", "Art.20c"):
        cat = "shipment"
    if article.startswith("Art.38"):
        cat = "compliance"
    if relation == "STRUCTURAL_MODIFICATION":
        return (
            f"Modification contractuelle structurelle ({article})",
            (
                "Article UCP déclaré non applicable selon art.1 UCP 600. "
                "Ce n'est pas une violation juridique, mais un changement structurel du risque contractuel."
            ),
            "Vérifier l'acceptation explicite des parties et l'impact opérationnel sur le mécanisme documentaire.",
            cat,
            "contractual_modification",
        )
    if relation == "CONTRACT_MODIFICATION":
        return (
            f"Modification contractuelle ({article})",
            (
                "Clause d'exclusion/modification détectée (art.1 UCP 600). "
                "Interpréter comme restriction contractuelle, pas comme violation UCP."
            ),
            "Confirmer que cette modification est voulue et comprise par les parties.",
            cat,
            "contractual_modification",
        )
    if relation == "VIOLATION":
        return (
            f"Analyse sémantique — risque de non-conformité UCP ({article})",
            f"Clause alignée sur « {title} » avec signaux défavorables ou exclusion explicite.",
            "Vérifier la clause au regard des pratiques UCP 600 et des documents :46A:.",
            cat,
            "true_violation",
        )
    if relation == "RESTRICTION":
        return (
            f"Analyse sémantique — restriction additionnelle ({article})",
            f"Clause proche de « {title} » avec langage restrictif.",
            "Confirmer que la banque et le bénéficiaire acceptent cette stricte formulation.",
            cat,
            "contractual_modification",
        )
    return ("", "", "", cat, "contractual_modification")


def aggregate_findings(
    per_clause_rows: list[dict[str, Any]],
    article_titles: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Une finding par article au pire cas
    (VIOLATION > STRUCTURAL_MODIFICATION > CONTRACT_MODIFICATION > RESTRICTION).
    """
    by_art: dict[str, dict[str, Any]] = {}
    order = {
        "VIOLATION": 5,
        "STRUCTURAL_MODIFICATION": 4,
        "CONTRACT_MODIFICATION": 3,
        "RESTRICTION": 2,
        "COMPLIANT": 1,
    }

    for row in per_clause_rows:
        art = row["article"]
        rel = row["relation"]
        if rel == "COMPLIANT":
            continue
        prev = by_art.get(art)
        if not prev or order[rel] > order[prev["relation"]]:
            by_art[art] = row

    out: list[dict[str, Any]] = []
    for art, row in by_art.items():
        rel = row["relation"]
        slug = _article_code_slug(art)
        if rel == "VIOLATION":
            code = f"SEM_VIOL_{slug}"
            sev = "error"
        elif rel == "STRUCTURAL_MODIFICATION":
            code = f"SEM_CONTRACT_STRUCT_{slug}"
            sev = "warning"
        elif rel == "CONTRACT_MODIFICATION":
            code = f"SEM_CONTRACT_MOD_{slug}"
            sev = "warning"
        else:
            code = f"SEM_REST_{slug}"
            sev = "warning"
        msg, detail, rec, cat, nature = _finding_message(art, rel, article_titles.get(art, art))
        out.append(
            {
                "article":        art,
                "article_title":  article_titles.get(art, art),
                "code":           code,
                "severity":       sev,
                "category":       cat,
                "message":        msg,
                "detail":         detail,
                "clause_text":    (row.get("clause_text") or "")[:200],
                "recommendation": rec,
                "confidence":     float(row.get("confidence", 0.75)),
                "exclusion_type": "semantic",
                "finding_nature": nature,
                "clause_type":    row.get("clause_type", "document_format"),
                "impact_level":   row.get("impact_level", "MEDIUM"),
            }
        )
    return out


class SemanticUCPEngine:
    """Orchestration alignement + classification + agrégation."""

    def __init__(
        self,
        config: Optional[UCPAlignerConfig] = None,
        encoder: Any = None,
        model_name: Optional[str] = None,
        nli_refiner: Optional[UCPNLIRefiner] = None,
    ) -> None:
        self.config = config or UCPAlignerConfig()
        self._aligner = UCPClauseAligner(
            config=self.config,
            encoder=encoder,
            model_name=model_name,
        )
        self._nli = nli_refiner if nli_refiner is not None else UCPNLIRefiner()

    def run(
        self,
        clauses: list[dict],
        ocr_text: str = "",
        article_titles: Optional[dict[str, str]] = None,
    ) -> SemanticUCPEngineResult:
        titles = article_titles or {}
        exclusion_map = parse_strict_exclusions(ocr_text, clauses)

        if not self._aligner._ensure_profile_embeddings():
            return SemanticUCPEngineResult(
                available=False,
                meta={
                    "reason": "embeddings_unavailable",
                    "n_clauses": len(clauses),
                    "n_violation": 0,
                    "n_restriction": 0,
                    "n_compliant": 0,
                    "exclusions": exclusion_map,
                    "nli_enabled": self._nli.enabled,
                    "nli_model": self._nli.model_name if self._nli.enabled else None,
                    "nli_ready": bool(self._nli.enabled and self._nli.available),
                },
            )

        per_clause: list[dict[str, Any]] = []
        counts = {
            "COMPLIANT": 0,
            "RESTRICTION": 0,
            "VIOLATION": 0,
            "CONTRACT_MODIFICATION": 0,
            "STRUCTURAL_MODIFICATION": 0,
        }

        for i, c in enumerate(clauses):
            raw = (c.get("text") or "").strip()
            if len(normalize_clause(raw)) < 8:
                continue
            al = self._aligner.align_clause(raw, clause_index=i)
            if not al:
                continue
            rel, conf, art_f = classify_relation(
                al, exclusion_map, self.config, nli=self._nli
            )
            counts[rel] = counts.get(rel, 0) + 1
            eff_article = art_f if (art_f and rel in {"VIOLATION", "CONTRACT_MODIFICATION", "STRUCTURAL_MODIFICATION"}) else al.best_article
            eff_title = (
                titles.get(eff_article, eff_article)
                if eff_article != al.best_article
                else al.best_title
            )
            per_clause.append(
                {
                    "clause_index":  i,
                    "article":       eff_article,
                    "article_title": eff_title,
                    "similarity":    round(al.similarity, 4),
                    "relation":      rel,
                    "confidence":    round(conf, 3),
                    "clause_text":   raw[:300],
                    "clause_type":   al.clause_type,
                    "impact_level":  al.impact_level,
                    "top_k":         [{"article": a, "sim": round(s, 4)} for a, s in al.top_k],
                }
            )

        finding_dicts = aggregate_findings(per_clause, titles)
        finding_dicts = merge_hard_findings(
            finding_dicts,
            findings_from_critical_exclusion_map(exclusion_map, titles),
        )
        finding_dicts = dedupe_ucp_finding_dicts(finding_dicts)
        nli_ready = bool(self._nli.enabled and self._nli.available)
        meta = {
            "n_clauses_scored": len(per_clause),
            "n_violation": counts["VIOLATION"],
            "n_contract_modification": counts["CONTRACT_MODIFICATION"],
            "n_structural_modification": counts["STRUCTURAL_MODIFICATION"],
            "n_restriction": counts["RESTRICTION"],
            "n_compliant": counts["COMPLIANT"],
            "violation_ratio": round(
                counts["VIOLATION"] / max(1, len(per_clause)), 4
            ),
            "restriction_ratio": round(
                counts["RESTRICTION"] / max(1, len(per_clause)), 4
            ),
            "exclusions": exclusion_map,
            "model_name": self.config.model_name,
            "nli_enabled": self._nli.enabled,
            "nli_model": self._nli.model_name if self._nli.enabled else None,
            "nli_ready": nli_ready,
        }
        return SemanticUCPEngineResult(
            per_clause=per_clause,
            finding_dicts=finding_dicts,
            meta=meta,
            available=True,
        )
