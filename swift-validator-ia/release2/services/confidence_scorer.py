"""
confidence_scorer.py
====================
Couche 4 du pipeline — scoring de confiance par champ et global.

Score global [0, 1] pondéré sur 4 composantes :
  0.30 × coverage   — champs trouvés / champs attendus
  0.25 × quality    — qualité extracteur (regex > layoutlm > kv > spaCy)
  0.25 × rules      — taux de règles métier passées
  0.20 × classifier — confiance du classifieur doc_type

Score par champ :
  base : regex=0.95, layoutlm=0.90, kv_fallback=0.75, spacy=0.60
  ajusté selon les résultats des règles métier

Usage :
  from release2.services.confidence_scorer import ConfidenceScorer
  scorer = ConfidenceScorer()
  result = scorer.score(fields, rule_results, doc_type, classifier_conf, tiers)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_TIER_BASE_SCORE: dict[str, float] = {
    "regex":       0.95,
    "layoutlm":    0.90,
    "kv_fallback": 0.75,
    "spacy":       0.60,
    "unknown":     0.70,
}

_RULE_STATUS_WEIGHT: dict[str, float] = {
    "pass":      1.00,
    "corrected": 0.85,
    "skipped":   0.80,
    "warning":   0.50,
    "fail":      0.00,
}

_RULE_FIELD_DELTA: dict[str, float] = {
    "pass":      +0.05,
    "corrected": +0.02,
    "warning":   -0.15,
    "fail":      -0.35,
    "skipped":    0.00,
}

_EXPECTED_FIELDS: dict[str, list[str]] = {
    "bl": [
        "bl_number", "vessel", "voyage",
        "port_of_loading", "port_of_discharge",
        "issue_date", "etd", "eta",
        "container_number", "seal_number",
        "gross_weight_kg", "net_weight_kg",
        "measurement", "freight_terms", "incoterm",
    ],
    "invoice": [
        "invoice_number", "invoice_date",
        "total_amount", "currency",
        "payment_terms", "incoterm",
        "goods_description", "bl_reference",
    ],
    "packing_list": [
        "pl_number", "bl_reference",
        "container_number", "total_packages",
        "total_cbm", "gross_weight_kg", "net_weight_kg",
    ],
}

_WEIGHTS = {
    "coverage":   0.30,
    "quality":    0.25,
    "rules":      0.25,
    "classifier": 0.20,
}


# ─────────────────────────────────────────────────────────────────────────────
# Structure de résultat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoringResult:
    global_score:     float
    coverage_score:   float
    quality_score:    float
    rules_score:      float
    classifier_score: float
    field_scores:     dict[str, float] = field(default_factory=dict)
    fields_found:     int = 0
    fields_expected:  int = 0

    def to_dict(self) -> dict:
        return {
            "global_score":     round(self.global_score, 4),
            "coverage_score":   round(self.coverage_score, 4),
            "quality_score":    round(self.quality_score, 4),
            "rules_score":      round(self.rules_score, 4),
            "classifier_score": round(self.classifier_score, 4),
            "field_scores":     {k: round(v, 4) for k, v in self.field_scores.items()},
            "fields_found":     self.fields_found,
            "fields_expected":  self.fields_expected,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceScorer:
    """
    Calcule le score de confiance d'extraction — aucune dépendance ML.

    Formule déterministe basée sur couverture, qualité extracteur,
    résultats des règles métier et confiance du classifieur.
    """

    def score(
        self,
        fields:          dict[str, str],
        rule_results:    list,
        doc_type:        str,
        classifier_conf: float,
        tiers:           dict[str, str] | None = None,
    ) -> ScoringResult:
        """
        Paramètres
        ----------
        fields          : champs après correction des règles
        rule_results    : liste de RuleResult
        doc_type        : "bl", "invoice", "packing_list"
        classifier_conf : confiance brute du classifieur [0, 1]
        tiers           : champ → tier d'extraction

        Retourne
        --------
        ScoringResult avec scores global et par champ
        """
        tiers    = tiers or {}
        expected = _EXPECTED_FIELDS.get(doc_type, [])

        # 1. Coverage
        n_expected = max(len(expected), 1)
        n_found    = sum(1 for f in expected if f in fields)
        coverage   = n_found / n_expected

        # 2. Quality (moyenne des scores de tier)
        tier_scores = [_TIER_BASE_SCORE.get(tiers.get(f, "unknown"), 0.70)
                       for f in fields]
        quality = sum(tier_scores) / len(tier_scores) if tier_scores else 0.0

        # 3. Rules
        rules_score = self._rules_score(rule_results)

        # 4. Classifier
        classifier_score = float(max(0.0, min(1.0, classifier_conf)))

        # 5. Score global pondéré
        global_score = (
            _WEIGHTS["coverage"]   * coverage        +
            _WEIGHTS["quality"]    * quality          +
            _WEIGHTS["rules"]      * rules_score      +
            _WEIGHTS["classifier"] * classifier_score
        )
        global_score = round(max(0.0, min(1.0, global_score)), 4)

        # 6. Scores par champ
        field_scores = self._field_scores(fields, tiers, rule_results)

        log.debug(
            "[scorer/%s] global=%.3f  cov=%.2f  qual=%.2f  rules=%.2f  cls=%.2f  "
            "(%d/%d champs)",
            doc_type, global_score, coverage, quality,
            rules_score, classifier_score, n_found, n_expected,
        )

        return ScoringResult(
            global_score     = global_score,
            coverage_score   = round(coverage, 4),
            quality_score    = round(quality, 4),
            rules_score      = round(rules_score, 4),
            classifier_score = round(classifier_score, 4),
            field_scores     = field_scores,
            fields_found     = n_found,
            fields_expected  = n_expected,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rules_score(self, rule_results: list) -> float:
        if not rule_results:
            return 1.0
        total = sum(_RULE_STATUS_WEIGHT.get(r.status, 0.5) for r in rule_results)
        return total / len(rule_results)

    def _field_scores(
        self,
        fields:       dict[str, str],
        tiers:        dict[str, str],
        rule_results: list,
    ) -> dict[str, float]:
        # Index des résultats par champ
        by_field: dict[str, list] = {}
        for r in rule_results:
            if r.field:
                by_field.setdefault(r.field, []).append(r)

        scores: dict[str, float] = {}
        for f, v in fields.items():
            if not v:
                continue
            score = _TIER_BASE_SCORE.get(tiers.get(f, "unknown"), 0.70)
            for r in by_field.get(f, []):
                score += _RULE_FIELD_DELTA.get(r.status, 0.0)
            scores[f] = round(max(0.0, min(1.0, score)), 4)

        return scores
