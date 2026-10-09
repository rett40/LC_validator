"""Analyse sémantique SWIFT v4 — 7 axes de validation sémantique."""

from .swift_semantic_analyzer import (
    SwiftSemanticAnalyzer,
    SemanticAnalysisResult,
    train_risk_model,
    # Axe 1 — Cohérence :46A: documents vs contexte commercial
    DocumentContextValidator,
    # Axe 2 — Cohérence montant / tolérance / quantité
    AmountToleranceValidator,
    # Axe 3 — Cohérence temporelle étendue
    ExtendedTemporalValidator,
    # Axe 4 — Cohérence géographique avancée
    GeoCoherenceValidator,
)
from .swift_nli_contradiction import OfflineNLIPredictor, NLIScores
from .swift_cross_field_coherence import CrossFieldCoherenceAnalyzer, CrossFieldCoherenceResult
from .swift_entity_alignment import EntityAlignmentChecker

__all__ = [
    "SwiftSemanticAnalyzer",
    "SemanticAnalysisResult",
    "train_risk_model",
    "DocumentContextValidator",
    "AmountToleranceValidator",
    "ExtendedTemporalValidator",
    "GeoCoherenceValidator",
    "OfflineNLIPredictor",
    "NLIScores",
    "CrossFieldCoherenceAnalyzer",
    "CrossFieldCoherenceResult",
    "EntityAlignmentChecker",
]