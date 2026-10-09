"""
services/risk_service/__init__.py
==================================
Package Risk Service ML — MT700 Risk Scoring v4.0.

Exports principaux :
  RiskMLEngine         — moteur principal
  RiskMLResult         — dataclass résultat (inclut score_breakdown, explanation, missing_fields_detail)
  RiskFeatureExtractor — extraction 26 features depuis payload pipeline
  get_risk_engine      — factory singleton
  compute_score_breakdown — calcul décomposition 7 dimensions
  compute_explanation  — SHAP waterfall + dimension radar
"""

from services.risk_service.risk_engine import (
    RiskMLEngine,
    RiskMLResult,
    RiskFeatureExtractor,
    UCPRuleEngine,
    UCPViolation,
    NarrativeExplainer,
    compute_score_breakdown,
    compute_explanation,
    FEATURE_COLS,
    FEATURE_DEFAULTS,
    get_risk_engine,
)

__all__ = [
    "RiskMLEngine",
    "RiskMLResult",
    "RiskFeatureExtractor",
    "UCPRuleEngine",
    "UCPViolation",
    "NarrativeExplainer",
    "compute_score_breakdown",
    "compute_explanation",
    "FEATURE_COLS",
    "FEATURE_DEFAULTS",
    "get_risk_engine",
]
