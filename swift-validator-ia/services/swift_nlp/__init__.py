"""
swift_nlp  -  Package NLP v2 pour messages SWIFT
==================================================
Arborescence :
  common/     — datasets, cache SentenceTransformer
  ner/        — extraction d'entités
  clauses/    — classification de clauses UCP
  semantic/   — analyse sémantique, risques, NLI offline
  ucp/        — validateur UCP 600, règles dures, alignement
  ds/         — features ML et scoring risque LC
  spell_correction — module ``services.ocr.spell_correction`` (post-OCR)

Usage rapide :
    from services.swift_nlp import SwiftNLPService
    service = SwiftNLPService()
    result  = service.process(raw_text)

Mode avance (tous les modules ML) :
    from services.swift_nlp import SwiftNLPService, NLPServiceConfig
    config  = NLPServiceConfig(
        enable_embeddings=True,
        enable_semantic_coherence=True,
    )
    service = SwiftNLPService(config)
    result  = service.process(raw_text, swift_type="MT700")

Entrainement des modeles :
    from services.swift_nlp import train_clause_classifier, train_risk_model
    train_clause_classifier("./data/data_stati", "./models/clause-clf")
    train_risk_model("./data/data_stati", "./models/risk-clf")
"""

from services.ocr.spell_correction import SwiftSpellCorrector, SpellCorrectionReport

from .ner.swift_ner_extractor import (
    SwiftNERExtractor,
    NERResult,
    NamedEntity,
    EntityRelation,
)
from .clauses.swift_clause_classifier import (
    SwiftClauseClassifier,
    ClauseClassification,
    train_clause_classifier,
)
from .semantic.swift_semantic_analyzer import (
    SwiftSemanticAnalyzer,
    SemanticAnalysisResult,
    train_risk_model,
)
from .swift_nlp_service import (
    SwiftNLPService,
    NLPServiceConfig,
    NLPProcessingResult,
    format_nlp_report,
    format_nlp_report_html,
)

__version__ = "2.0.0"

__all__ = [
    "SwiftNLPService",
    "NLPServiceConfig",
    "NLPProcessingResult",
    "format_nlp_report",
    "format_nlp_report_html",
    "SwiftSpellCorrector",
    "SpellCorrectionReport",
    "SwiftNERExtractor",
    "NERResult",
    "NamedEntity",
    "EntityRelation",
    "SwiftClauseClassifier",
    "ClauseClassification",
    "SwiftSemanticAnalyzer",
    "SemanticAnalysisResult",
    "train_clause_classifier",
    "train_risk_model",
]
