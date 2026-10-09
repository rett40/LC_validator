"""
swift_semantic_analyzer.py  —  v3.0
====================================
Analyse sémantique avancée des messages SWIFT selon UCP 600 / URDG 758.

NOUVELLES TECHNIQUES v3 :
  ✦ Fuzzy clause matching (RapidFuzz) — résiste aux OCR bruités
  ✦ Scoring Bayésien par champ — intervalles de confiance, pas juste 0/1
  ✦ Moteur de contradictions tag/clause généralisé (UCP Art.5)
  ✦ Ontologie de risque avec propagation par graphe d'articles
  ✦ Matrice de risque Incoterm (FCA/FOB/CIF/etc.)
  ✦ Typage sémantique des exclusions (permissive vs restrictive)
  ✦ Normalisation phonétique des entités (noms de banques/entreprises)
  ✦ Détection de patterns temporels suspects (dates incohérentes)

Architecture :
  ExclusionSemanticTyper  — classifie chaque exclusion UCP
  ContradictionEngine     — détecte conflits tag SWIFT vs clause textuelle
  IncotermsRiskMatrix     — profil de risque par Incoterm
  RiskOntologyGraph       — propagation de risque entre articles liés
  BayesianFieldScorer     — scoring avec intervalles de confiance
  SwiftSemanticAnalyzer   — orchestrateur principal v3
"""

from __future__ import annotations

import logging
import re
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ── Offline NLI (optionnel) ───────────────────────────────────────────
# ── Path (pour NLI model dir) ────────────────────────────────────────
from pathlib import Path
try:
    from .swift_nli_contradiction import OfflineNLIPredictor
except Exception:
    OfflineNLIPredictor = None  # type: ignore

try:
    from .swift_cross_field_coherence import CrossFieldCoherenceAnalyzer, CrossFieldCoherenceResult
except Exception:
    CrossFieldCoherenceAnalyzer = None  # type: ignore
    CrossFieldCoherenceResult   = None  # type: ignore

# ── Fuzzy matching (optionnel, graceful fallback) ──────────────────
try:
    from rapidfuzz import fuzz as _rfuzz
    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False
    logger.info("rapidfuzz non disponible — fuzzy matching désactivé")


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SemanticWarning:
    code:       str
    severity:   str       # "error" | "warning" | "info"
    message:    str
    field:      str
    ucp_ref:    str  = ""
    suggestion: str  = ""
    confidence: float = 1.0   # NEW v3 : confiance [0-1]
    source:     str  = "rule" # NEW v3 : "rule"|"fuzzy"|"bayesian"|"ontology"

    def to_dict(self) -> dict:
        return {
            "code": self.code, "severity": self.severity,
            "message": self.message, "field": self.field,
            "ucp_ref": self.ucp_ref, "suggestion": self.suggestion,
            "confidence": round(self.confidence, 3), "source": self.source,
        }


@dataclass
class RiskFlag:
    code:           str
    description:    str
    severity:       str       # "high" | "medium" | "low"
    ucp_ref:        str
    ml_score:       float = 0.0
    category:       str   = ""
    recommendation: str   = ""
    propagated_from: str  = ""  # NEW v3 : article source de propagation
    confidence:     float = 1.0  # NEW v3

    def to_dict(self) -> dict:
        return {
            "code": self.code, "description": self.description,
            "severity": self.severity, "ucp_ref": self.ucp_ref,
            "ml_score": round(self.ml_score, 3), "category": self.category,
            "recommendation": self.recommendation,
            "propagated_from": self.propagated_from,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ContradictionFinding:
    """NEW v3 — Contradiction entre un tag SWIFT et une clause textuelle."""
    tag_field:    str    # ex: "transhipment" (:43T:)
    tag_value:    str    # ex: "ALLOWED"
    clause_text:  str    # extrait de la clause contradictoire
    clause_intent: str   # "prohibit" | "require" | "modify"
    ucp_basis:    str    # "Art.5" — la clause prime
    severity:     str    # toujours "error" en v3
    recommendation: str  = ""
    confidence:   float = 1.0

    def to_dict(self) -> dict:
        return {
            "tag_field": self.tag_field, "tag_value": self.tag_value,
            "clause_text": self.clause_text[:100],
            "clause_intent": self.clause_intent, "ucp_basis": self.ucp_basis,
            "severity": self.severity, "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class SemanticAnalysisResult:
    swift_type:           str
    warnings:             list[SemanticWarning]     = field(default_factory=list)
    risk_flags:           list[RiskFlag]            = field(default_factory=list)
    contradictions:       list[ContradictionFinding] = field(default_factory=list)
    completeness_score:   int   = 0
    semantic_score:       int   = 0
    bayesian_confidence:  float = 1.0   # NEW v3
    summary:              str   = ""
    enriched_fields:      dict  = field(default_factory=dict)
    coherence_scores:     dict  = field(default_factory=dict)
    fraud_indicators:     list  = field(default_factory=list)
    geo_coherence:        dict  = field(default_factory=dict)
    doc_quality_score:    int   = 0
    recommendations:      list  = field(default_factory=list)
    incoterm_risk:        dict  = field(default_factory=dict)   # NEW v3
    exclusion_types:      dict  = field(default_factory=dict)   # NEW v3
    ontology_propagations: list = field(default_factory=list)   # NEW v3

    # ── Axe 5 : Score composite + versionnage moteur ──────────────────
    final_score:          float = 0.0    # 0.4*completeness + 0.3*semantic + 0.3*(1-max_risk)
    engine_version:       str   = "4.0"  # piste d'audit réglementaire
    analysis_timestamp:   str   = ""     # ISO 8601
    routing_decision:     str   = ""     # "ACCEPT" | "REVIEW" | "REJECT"

    # ── AXE 7 : Cohérence inter-champs MT700 ──────────────────────────
    cross_field_coherence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "swift_type":           self.swift_type,
            "completeness_score":   self.completeness_score,
            "semantic_score":       self.semantic_score,
            "bayesian_confidence":  round(self.bayesian_confidence, 3),
            "final_score":          round(self.final_score, 3),
            "routing_decision":     self.routing_decision,
            "engine_version":       self.engine_version,
            "analysis_timestamp":   self.analysis_timestamp,
            "doc_quality_score":    self.doc_quality_score,
            "warnings":             [w.to_dict() for w in self.warnings],
            "risk_flags":           [r.to_dict() for r in self.risk_flags],
            "contradictions":       [c.to_dict() for c in self.contradictions],
            "coherence_scores":     self.coherence_scores,
            "fraud_indicators":     self.fraud_indicators,
            "geo_coherence":        self.geo_coherence,
            "recommendations":      self.recommendations,
            "incoterm_risk":        self.incoterm_risk,
            "exclusion_types":      self.exclusion_types,
            "ontology_propagations": self.ontology_propagations,
            "enriched_fields":      self.enriched_fields,
            "summary":              self.summary,
            "cross_field_coherence": self.cross_field_coherence,
        }

    @property
    def has_errors(self) -> bool:
        return (any(w.severity == "error" for w in self.warnings)
                or bool(self.contradictions))

    @property
    def has_high_risk(self) -> bool:
        return any(r.severity == "high" for r in self.risk_flags)


# ══════════════════════════════════════════════════════════════════════
# CHAMPS REQUIS PAR TYPE MT
# ══════════════════════════════════════════════════════════════════════

_REQUIRED_FIELDS: dict[str, list[str]] = {
    "MT700": [
        "lc_number", "lc_form", "issue_date", "expiry_date", "expiry_place",
        "applicant", "beneficiary", "currency", "amount", "available_by",
        "draft_at", "partial_shipments", "transhipment",
        "port_loading", "port_discharge", "shipment_date",
        "goods_description", "documents_required",
    ],
    "MT720": [
        "lc_number", "related_lc", "lc_form", "issue_date", "expiry_date",
        "applicant", "beneficiary", "currency", "amount", "issuing_bank",
    ],
    "MT760": [
        "lc_number", "issue_date", "expiry_date", "expiry_place",
        "applicant", "beneficiary", "currency", "amount",
        "undertaking_type", "guarantee_text",
    ],
    "MT790": ["lc_number", "related_lc", "currency", "amount", "charge_details"],
    "MT799": ["lc_number", "free_text"],
}

_OPTIONAL_IMPORTANT: dict[str, list[str]] = {
    "MT700": [
        "applicable_rules", "confirmation", "presentation_period",
        "tolerance_plus", "tolerance_minus", "charges",
    ],
    "MT760": ["applicable_rules", "purpose", "expiry_type"],
    "MT720": ["applicable_rules", "goods_description"],
    "MT790": ["account", "sender_info"],
    "MT799": ["sender_info"],
}

_DOC_QUALITY_WEIGHTS: dict[str, int] = {
    "BILL OF LADING":         20,
    "COMMERCIAL INVOICE":     20,
    "PACKING LIST":           10,
    "CERTIFICATE OF ORIGIN":  10,
    "INSURANCE":              15,
    "INSPECTION CERTIFICATE": 10,
    "AIRWAY BILL":             8,
    "DRAFT":                   7,
}

# Aliases reconnus pour chaque type de doc (fuzzy)
_DOC_ALIASES: dict[str, list[str]] = {
    "BILL OF LADING":         ["B/L", "BL", "LADING", "CONNAISSEMENT"],
    "COMMERCIAL INVOICE":     ["INVOICE", "FACTURE", "INV"],
    "PACKING LIST":           ["PACKING", "P/L", "COLISAGE"],
    "CERTIFICATE OF ORIGIN":  ["CERT ORIGIN", "C/O", "ORIGINE"],
    "INSURANCE":              ["INSURANCE POLICY", "POLICE", "ASSURANCE", "INSUR"],
    "INSPECTION CERTIFICATE": ["INSPECTION CERT", "SGS", "BUREAU VERITAS"],
    "AIRWAY BILL":            ["AWB", "AIR WAYBILL", "AIRWAY"],
    "DRAFT":                  ["BILL OF EXCHANGE", "TRAITE", "DRAFT"],
}


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — TYPAGE SÉMANTIQUE DES EXCLUSIONS
# ══════════════════════════════════════════════════════════════════════

class ExclusionSemanticTyper:
    """
    Classifie chaque exclusion d'article UCP en :
      - "restrictive"  : l'exclusion RENFORCE une contrainte (ex: Art.18b)
      - "permissive"   : l'exclusion ASSOUPLIT une règle (ex: Art.14c)
      - "prohibitive"  : l'exclusion INTERDIT quelque chose (ex: Art.20c, Art.14k)
      - "payment"      : l'exclusion touche le mécanisme de paiement (ex: Art.12b)
      - "charges"      : l'exclusion touche la répartition des frais (ex: Art.37c)

    BUG #2 FIX définitif : Art.18b "not applicable" = RESTRICTIVE (pas permissif).
    """

    _ARTICLE_TYPES: dict[str, str] = {
        "Art.12b":  "payment",      # pré-paiement interdit → payment
        "Art.14c":  "permissive",   # docs non prescrits examinés → assouplit
        "Art.14k":  "prohibitive",  # expéditeur ≠ bénéficiaire → interdit
        "Art.18b":  "restrictive",  # facture ≤ montant LC → renforce
        "Art.20":   "prohibitive",  # transbordement → interdit
        "Art.20c":  "prohibitive",  # transbordement containers → interdit
        "Art.37":   "charges",      # frais → charge bénéficiaire
        "Art.37c":  "charges",      # commissions → charge bénéficiaire
        "Art.38":   "restrictive",  # transfert LC → restrictif
        "Art.38i":  "restrictive",  # factures premier bénéficiaire → restrictif
    }

    # Patterns textuels pour affiner la classification dynamique
    _RESTRICTIVE_PATTERNS = re.compile(
        r"not\s+acceptable|must\s+not|shall\s+not|is\s+prohibited"
        r"|is\s+not\s+allowed|interdit|ne\s+peut\s+pas",
        re.IGNORECASE
    )
    _PERMISSIVE_PATTERNS = re.compile(
        r"may\s+be|is\s+permitted|is\s+allowed|peut\s+être|autorisé",
        re.IGNORECASE
    )
    _PROHIBITIVE_PATTERNS = re.compile(
        r"prohibited|forbidden|not\s+allowed|interdit|défendu",
        re.IGNORECASE
    )

    @classmethod
    def classify(cls, article_ref: str, clause_text: str) -> str:
        """Retourne le type sémantique de l'exclusion."""
        # 1. Lookup fixe (articles connus)
        known = cls._ARTICLE_TYPES.get(article_ref)
        if known:
            return known

        # 2. Analyse textuelle dynamique
        text = clause_text or ""
        if cls._PROHIBITIVE_PATTERNS.search(text):
            return "prohibitive"
        if cls._RESTRICTIVE_PATTERNS.search(text):
            return "restrictive"
        if cls._PERMISSIVE_PATTERNS.search(text):
            return "permissive"
        return "neutral"

    @classmethod
    def get_severity(cls, exclusion_type: str) -> str:
        """Sévérité recommandée selon le type d'exclusion."""
        return {
            "restrictive":  "warning",   # contrainte renforcée → info documentaire
            "permissive":   "info",
            "prohibitive":  "error",     # interdit → bloquant
            "payment":      "warning",
            "charges":      "warning",
            "neutral":      "info",
        }.get(exclusion_type, "info")


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — MOTEUR DE CONTRADICTIONS TAG / CLAUSE (UCP Art.5)
# ══════════════════════════════════════════════════════════════════════

class ContradictionEngine:
    """
    Détecte les contradictions entre les champs SWIFT (tags) et les clauses
    textuelles (:47A:). Selon UCP 600 Art.5, le crédit documentaire prime
    sur les instructions sous-jacentes — la clause textuelle prime sur le tag.

    Cas couverts :
      - :43T: ALLOWED  vs clause "transhipment is prohibited"
      - :43P: ALLOWED  vs clause "partial shipments not permitted"
      - :41A: BY PAYMENT vs clause "deferred payment"
      - :39A: tolérance vs clause "exact quantity required"
      - :49: CONFIRM  vs clause "confirmation not required"
    """

    _RULES: list[dict] = [
        {
            "tag_field":    "transhipment",
            "tag_prohibit_values": ["ALLOWED"],
            "clause_patterns": [
                # EN
                re.compile(r"trans(?:s)?hip(?:ment)?[\s\-]*(?:is[\s\-]*)?prohibit(?:ed)?", re.I),
                re.compile(r"(?:no|without)\s+trans(?:s)?hip(?:ment)?", re.I),
                re.compile(r"trans(?:s)?hip(?:ment)?\s+(?:not|is\s+not)\s+(?:allowed|permitted)", re.I),
                # FR (transbordement)
                re.compile(r"transbord(?:ement)?\s+(?:est\s+)?(?:interdit|prohib(?:e|é)|défendu)", re.I),
                re.compile(r"transbord(?:ement)?\s+(?:non\s+autorisé|pas\s+autorisé)", re.I),
                re.compile(r"(?:aucun|pas\s+de)\s+transbord(?:ement)?", re.I),
            ],
            "clause_intent": "prohibit",
            "ucp_basis":    "Art.5 / Art.20c",
            "recommendation": "Corriger :43T: → 'NOT ALLOWED' pour aligner avec la clause",
            "nli_topic_keywords": ["TRANSSHIPMENT", "TRANSHIPMENT", "TRANSSHIP", "TRANSBORDEMENT"],
            "nli_hypotheses": {
                "allow":   "Transshipment is allowed.",
                "prohibit":"Transshipment is prohibited.",
            },
        },
        {
            "tag_field":    "partial_shipments",
            "tag_prohibit_values": ["ALLOWED"],
            "clause_patterns": [
                re.compile(
                    r"partial\s+ship(?:ment)?s?[\s\-]*(?:is[\s\-]*)?(?:not|is\s+not)\s+(?:allowed|permitted|acceptable)",
                    re.I,
                ),
                re.compile(r"no\s+partial\s+ship(?:ment)?s?", re.I),
                # FR
                re.compile(r"exp[ée]dition(?:s)?\s+partielle(?:s)?\s+(?:est\s+)?(?:interdite|prohib(?:e|é)|défendue)", re.I),
                re.compile(r"exp[ée]dition(?:s)?\s+partielle(?:s)?\s+(?:non\s+autoris[ée]e?s?)", re.I),
                re.compile(r"(?:aucune|pas\s+d')exp[ée]dition(?:s)?\s+partielle(?:s)?", re.I),
            ],
            "clause_intent": "prohibit",
            "ucp_basis":    "Art.5 / Art.31",
            "recommendation": "Corriger :43P: → 'NOT ALLOWED' pour aligner avec la clause",
            "nli_topic_keywords": ["PARTIAL", "SHIPMENT", "EXPEDITION PARTIELLE", "EXPÉDITION PARTIELLE"],
            "nli_hypotheses": {
                "allow":   "Partial shipments are allowed.",
                "prohibit":"Partial shipments are prohibited.",
            },
        },
        {
            "tag_field":    "confirmation",
            "tag_prohibit_values": ["WITHOUT"],
            "clause_patterns": [
                re.compile(r"confirm(?:ation)?\s+(?:is\s+)?(?:required|mandatory|demandée)", re.I),
                # FR
                re.compile(r"confirmation\s+(?:requise|obligatoire|exig(?:ée|ee))", re.I),
                re.compile(r"cr[ée]dit\s+confirm[ée]", re.I),
            ],
            "clause_intent": "require",
            "ucp_basis":    "Art.5 / Art.8",
            "recommendation": "Corriger :49: → 'CONFIRM' pour aligner avec la clause",
            "nli_topic_keywords": ["CONFIRM", "CONFIRMATION", "CONFIRMÉ", "CONFIRME"],
            "nli_hypotheses": {
                "require": "Confirmation is required.",
                "prohibit":"Confirmation is not required.",
            },
        },
        {
            # Cas fréquent : clause conditionnelle "IN CASE CONFIRMATION IS NOT ADDED..."
            # alors que le tag :49: indique déjà CONFIRM. On remonte en warning.
            "tag_field":    "confirmation",
            "tag_prohibit_values": ["CONFIRM"],
            "clause_patterns": [
                re.compile(r"\bIN\s+CASE\s+CONFIRMATION\s+IS\s+NOT\s+ADDED\b", re.I),
                re.compile(r"\bIF\s+CONFIRMATION\s+IS\s+NOT\s+ADDED\b", re.I),
                re.compile(r"\bSI\s+LA\s+CONFIRMATION\s+N['’]EST\s+PAS\s+AJOUT[ÉE]E\b", re.I),
            ],
            "clause_intent": "modify",
            "ucp_basis":    "Art.5 / Art.8",
            "recommendation": "Clause conditionnelle sur la confirmation — vérifier cohérence entre :49: et :47A:",
            "severity": "warning",
        },
        {
            "tag_field":    "available_by",
            # Si le tag indique NEGOTIATION/ACCEPTANCE mais la clause dit "payment/sight",
            # on considère une divergence forte.
            "tag_prohibit_values": ["NEGOTIATION", "ACCEPTANCE"],
            "clause_patterns": [
                re.compile(r"by\s+payment\s+at\s+sight", re.I),
                re.compile(r"available\s+by\s+payment", re.I),
                re.compile(r"\bat\s+sight\b", re.I),
                re.compile(r"\b(?:paiement\s+à\s+vue|payable\s+à\s+vue)\b", re.I),
            ],
            "clause_intent": "modify",
            "ucp_basis":    "Art.5 / Art.6",
            "recommendation": "Vérifier la modalité de réalisation — clause et :41A: divergent",
            "nli_topic_keywords": ["AVAILABLE", "PAYMENT", "SIGHT", "DEFERRED", "USANCE", "PAIEMENT", "VUE"],
            "nli_hypotheses": {
                "payment":  "The credit is available by payment at sight.",
                "deferred": "The credit is available by deferred payment.",
                "acceptance":"The credit is available by acceptance.",
                "negotiation":"The credit is available by negotiation.",
            },
        },
        # ── Règle 6 : :43T: NOT ALLOWED vs clause permissive transhipment ─
        {
            "tag_field": "transhipment",
            "tag_prohibit_values": ["NOT ALLOWED", "NOTALLOWED"],
            "clause_patterns": [
                re.compile(r"trans(?:s)?hip(?:ment)?\s+(?:is\s+)?(?:permitted|allowed|authorized|accepted)", re.I),
                re.compile(r"trans(?:s)?hip(?:ment)?\s+(?:acceptable)", re.I),
                re.compile(r"transbord(?:ement)?\s+(?:autoris[eé]|permis|accept[eé])", re.I),
            ],
            "clause_intent": "allow",
            "ucp_basis":    "Art.5 / Art.20c",
            "recommendation": "Corriger :43T: → 'ALLOWED' ou supprimer la clause autorisant le transbordement",
            "nli_topic_keywords": ["TRANSSHIPMENT", "TRANSHIPMENT", "TRANSSHIP", "TRANSBORDEMENT"],
            "nli_hypotheses": {
                "allow":    "Transshipment is allowed.",
                "prohibit": "Transshipment is prohibited.",
            },
        },
        # ── Règle 7 : :43P: NOT ALLOWED vs clause permissive partial shipment ─
        {
            "tag_field": "partial_shipments",
            "tag_prohibit_values": ["NOT ALLOWED", "NOTALLOWED"],
            "clause_patterns": [
                re.compile(r"partial\s+ship(?:ment)?s?\s+(?:are\s+)?(?:permitted|allowed|authorized|acceptable)", re.I),
                re.compile(r"partial\s+ship(?:ment)?s?\s+(?:acceptable)", re.I),
                re.compile(r"exp[eé]dition(?:s)?\s+partielle(?:s)?\s+(?:autoris[eé]e?s?|permise?s?|accept[eé]e?s?)", re.I),
            ],
            "clause_intent": "allow",
            "ucp_basis":    "Art.5 / Art.31",
            "recommendation": "Corriger :43P: → 'ALLOWED' ou supprimer la clause autorisant les expéditions partielles",
            "nli_topic_keywords": ["PARTIAL", "SHIPMENT", "EXPEDITION PARTIELLE"],
            "nli_hypotheses": {
                "allow":    "Partial shipments are allowed.",
                "prohibit": "Partial shipments are prohibited.",
            },
        },
        # ── Règle 8 : :42C: AT SIGHT vs clause paiement différé ──────────
        {
            "tag_field": "draft_at",
            "tag_prohibit_values": ["SIGHT", "AT SIGHT", "VUE", "À VUE"],
            "clause_patterns": [
                re.compile(r"\bdeferred\s+payment\b", re.I),
                re.compile(r"\busance\b", re.I),
                re.compile(r"\b(\d+)\s+days?\s+after\s+(?:sight|shipment|date\s+of\s+bill|date\s+of\s+invoice)", re.I),
                re.compile(r"\bpaiement\s+diff[eé]r[eé]\b", re.I),
                re.compile(r"\b(\d+)\s+jours?\s+(?:apr[eè]s|de\s+vue)\b", re.I),
            ],
            "clause_intent": "deferred",
            "ucp_basis":    "Art.6 / Art.42 UCP 600",
            "recommendation": "Contradiction paiement à vue (:42C:) vs paiement différé (:47A:). Aligner les deux.",
            "severity":      "error",
            "nli_topic_keywords": ["DEFERRED", "USANCE", "SIGHT", "PAIEMENT", "VUE", "DAYS AFTER"],
            "nli_hypotheses": {
                "sight":    "Payment is at sight.",
                "deferred": "Payment is deferred.",
            },
        },
        # ── Règle 9 : :49: MAY ADD vs clause exigeant la confirmation ────
        {
            "tag_field": "confirmation",
            "tag_prohibit_values": ["MAY ADD"],
            "clause_patterns": [
                re.compile(r"confirmation\s+(?:is\s+)?(?:mandatory|obligatory|required|demand[eé]e|obligatoire)", re.I),
                re.compile(r"must\s+be\s+confirmed", re.I),
                re.compile(r"shall\s+be\s+confirmed", re.I),
            ],
            "clause_intent": "require",
            "ucp_basis":    "Art.5 / Art.8 UCP 600",
            "recommendation": "Tag :49: 'MAY ADD' mais clause exige la confirmation — clarifier avec la banque émettrice",
            "severity":      "warning",
            "nli_topic_keywords": ["CONFIRM", "CONFIRMATION", "CONFIRMED"],
            "nli_hypotheses": {
                "require":  "Confirmation of the credit is required.",
                "optional": "Confirmation of the credit is optional.",
            },
        },
        # ── Règle 10 : Tolerance :39A: vs clause "EXACT QUANTITY" ────────
        {
            "tag_field": "tolerance_plus",  # mapped from :39A: field
            "tag_prohibit_values": ["5", "10", "3", "2"],  # non-zero tolerance values as strings
            "clause_patterns": [
                re.compile(r"\bexact\s+(?:quantity|amount|value)\b", re.I),
                re.compile(r"\bno\s+tolerance\b", re.I),
                re.compile(r"\bstrict\s+(?:quantity|amount)\b", re.I),
                re.compile(r"\bexact(?:ement)?\s+(?:la\s+)?(?:quantit[eé]|valeur|montant)\b", re.I),
                re.compile(r"\bsans\s+tol[eé]rance\b", re.I),
            ],
            "clause_intent": "exact",
            "ucp_basis":    "Art.30 UCP 600",
            "recommendation": "Tag :39A: indique une tolérance mais une clause exige une quantité exacte (Art.30)",
            "severity":      "warning",
            "nli_topic_keywords": ["EXACT", "QUANTITY", "TOLERANCE", "STRICT", "PRÉCIS"],
            "nli_hypotheses": {
                "exact":     "The quantity must be exact with no tolerance.",
                "tolerance": "A tolerance in quantity is allowed.",
            },
        },
    ]

    @classmethod
    def detect(
        cls,
        entities: dict,
        clauses_text: str,
        nli: Any = None,
    ) -> list[ContradictionFinding]:
        """Retourne la liste des contradictions tag vs clause détectées."""
        findings = []

        def _norm(s: str) -> str:
            if not s:
                return ""
            s = unicodedata.normalize("NFKD", s)
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            return s.upper()

        def _normalize_ocr_collisions(text: str) -> str:
            """
            Fix common OCR word-collisions that break regex/NLI triggers, e.g.:
            - 'Transshipmentis' -> 'Transshipment is'
            - 'Transbordementest' -> 'Transbordement est'
            """
            if not text:
                return ""
            t = text
            t = re.sub(r"(trans(?:s)?hipment)\s*is\b", r"\1 is", t, flags=re.I)
            t = re.sub(r"(trans(?:s)?hipment)\s*is(?=[A-Z])", r"\1 is ", t, flags=re.I)
            t = re.sub(r"(transbord(?:ement)?)\s*est\b", r"\1 est", t, flags=re.I)
            t = re.sub(r"(transbord(?:ement)?)\s*est(?=[A-Z])", r"\1 est ", t, flags=re.I)
            t = re.sub(r"(partial\s+shipments?)\s*is\b", r"\1 is", t, flags=re.I)
            t = re.sub(r"(partial\s+shipments?)\s*is(?=[A-Z])", r"\1 is ", t, flags=re.I)
            # OCR : « shipmentsis not » (sans espace avant is)
            t = re.sub(r"(partial\s+shipments)is(\s+not\b)", r"\1 is\2", t, flags=re.I)
            t = re.sub(r"(exp[ée]ditions?\s+partielles?)\s*est\b", r"\1 est", t, flags=re.I)
            t = re.sub(r"(exp[ée]ditions?\s+partielles?)\s*est(?=[A-Z])", r"\1 est ", t, flags=re.I)
            return t

        text_for_matching = _normalize_ocr_collisions(clauses_text or "")

        def _sentences_for_rule(text: str, rule: dict) -> list[str]:
            kws = [_norm(str(k)) for k in (rule.get("nli_topic_keywords") or [])]
            if not kws:
                return []
            # Découpe simple en phrases/lignes pour rester rapide.
            chunks = re.split(r"[.\n;]+", text or "")
            out = []
            for c in chunks:
                s = c.strip()
                if len(s) < 8:
                    continue
                su = _norm(s)
                # Correspondance sur frontière de mot (et non sous-chaîne) :
                # sinon "SHIPMENT" (mot-clé de partial_shipments) matche à
                # l'intérieur de "TRANSSHIPMENT", et la règle partial capture
                # par erreur la clause de transbordement (faux positif 43P/43T).
                if any(re.search(rf"\b{re.escape(k)}\b", su) for k in kws):
                    out.append(s[:500])
            return out[:8]

        def _infer_intent_with_nli(rule: dict, sentence: str) -> Optional[tuple[str, float]]:
            """
            Retourne (intent, score_entailment) si un intent est fortement entailé.
            """
            hyps = rule.get("nli_hypotheses") or {}
            if not isinstance(hyps, dict) or not hyps:
                return None
            best_intent = None
            best_ent = 0.0
            for intent, hyp in hyps.items():
                sc = nli.predict(sentence, str(hyp)) if nli else None
                if sc is None:
                    continue
                if sc.entailment > best_ent:
                    best_ent = sc.entailment
                    best_intent = str(intent)
            if best_intent and best_ent >= 0.78:
                return best_intent, best_ent
            return None

        for rule in cls._RULES:
            _raw = entities.get(rule["tag_field"])
            tag_val = str(_raw).upper().strip() if _raw is not None else ""
            if not tag_val or tag_val not in rule["tag_prohibit_values"]:
                continue

            contradiction_found = False
            found_clause_text   = ""
            found_clause_intent = rule["clause_intent"]
            confidence          = float(rule.get("confidence", 1.0))

            # ── Step 1 : NLI — détecteur primaire sémantique ─────────────────
            # Capture les formulations indirectes, paraphrases et langage ambigu
            # que les patterns regex ne couvrent pas.
            if nli is not None and getattr(nli, "available", False):
                for sent in _sentences_for_rule(text_for_matching, rule):
                    inferred = _infer_intent_with_nli(rule, sent)
                    if not inferred:
                        continue
                    intent, nli_score = inferred
                    # La contradiction dépend de la DIRECTION du tag, portée par
                    # rule["clause_intent"] (l'intention de clause qui contredit ce tag) :
                    #   tag ALLOWED      → contradiction si clause "prohibit"  (règles 1,2)
                    #   tag NOT ALLOWED  → contradiction si clause "allow"     (règles 6,7)
                    #   tag WITHOUT      → contradiction si clause "require"   (règle 3)
                    # Avant, "prohibit" était codé en dur pour transhipment/partial,
                    # donc une clause "NOT ALLOWED" cohérente avec un tag NOT ALLOWED
                    # était signalée à tort comme contradictoire.
                    if rule["tag_field"] == "available_by":
                        is_contradictory = intent in ("payment", "deferred")
                    else:
                        is_contradictory = (intent == rule.get("clause_intent"))
                    if is_contradictory:
                        found_clause_text   = sent
                        found_clause_intent = intent
                        contradiction_found = True
                        confidence          = float(nli_score)
                        break

            # ── Step 2 : Regex — renforcement ────────────────────────────────
            # Si NLI a trouvé  → regex confirme et booste la confiance.
            # Si NLI n'a rien  → regex sert de fallback (patterns explicites).
            for pattern in rule["clause_patterns"]:
                m = pattern.search(text_for_matching)
                if m:
                    ctx_start = max(0, m.start() - 20)
                    ctx_end   = min(len(text_for_matching), m.end() + 40)
                    if contradiction_found:
                        # Confirmation regex → confiance renforcée
                        confidence = min(1.0, confidence + 0.10)
                    else:
                        # Fallback regex
                        found_clause_text   = text_for_matching[ctx_start:ctx_end]
                        found_clause_intent = rule["clause_intent"]
                        contradiction_found = True
                        confidence          = float(rule.get("confidence", 1.0))
                    break

            # ── Step 3 : UN seul finding par règle ───────────────────────────
            if contradiction_found:
                findings.append(ContradictionFinding(
                    tag_field=rule["tag_field"],
                    tag_value=tag_val,
                    clause_text=found_clause_text,
                    clause_intent=found_clause_intent,
                    ucp_basis=rule["ucp_basis"],
                    severity=rule.get("severity", "error"),
                    recommendation=rule["recommendation"],
                    confidence=confidence,
                ))
        return findings


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — MATRICE DE RISQUE INCOTERMS
# ══════════════════════════════════════════════════════════════════════

class IncotermsRiskMatrix:
    """
    Profil de risque documentaire par Incoterm.
    Chaque Incoterm a des implications sur :
      - Qui supporte l'assurance
      - Quel document de transport est exigible
      - Risque de contradiction avec les clauses LC
    """

    _MATRIX: dict[str, dict] = {
        "EXW": {
            "risk_score":    5,
            "insurance_resp": "BUYER",
            "transport_doc":  "any",
            "risk_notes":    ["Risque maximal acheteur", "Assurance non couverte par vendeur"],
            "lc_compatibility": "low",
        },
        "FCA": {
            "risk_score":    3,
            "insurance_resp": "BUYER_FROM_NAMED_PLACE",
            "transport_doc":  "multimodal_or_air",
            "risk_notes":    ["Compatible LC", "Vérifier lieu de remise précis"],
            "lc_compatibility": "high",
        },
        "FOB": {
            "risk_score":    2,
            "insurance_resp": "BUYER_FROM_SHIP",
            "transport_doc":  "bill_of_lading",
            "risk_notes":    ["Classique LC maritime", "BL exigible"],
            "lc_compatibility": "high",
        },
        "CFR": {
            "risk_score":    2,
            "insurance_resp": "BUYER",
            "transport_doc":  "bill_of_lading",
            "risk_notes":    ["Fret payé vendeur", "Assurance acheteur — vérifier police LC"],
            "lc_compatibility": "high",
        },
        "CIF": {
            "risk_score":    1,
            "insurance_resp": "SELLER",
            "transport_doc":  "bill_of_lading",
            "risk_notes":    ["Assurance minimale ICC(C) incluse", "Standard LC maritime"],
            "lc_compatibility": "very_high",
        },
        "CPT": {
            "risk_score":    3,
            "insurance_resp": "BUYER",
            "transport_doc":  "multimodal",
            "risk_notes":    ["Transport payé vendeur", "Assurance non incluse — vérifier :46A:"],
            "lc_compatibility": "medium",
        },
        "CIP": {
            "risk_score":    1,
            "insurance_resp": "SELLER",
            "transport_doc":  "multimodal",
            "risk_notes":    ["Assurance ICC(A) minimum", "Recommandé LC multimodal"],
            "lc_compatibility": "very_high",
        },
        "DAP": {
            "risk_score":    4,
            "insurance_resp": "SELLER",
            "transport_doc":  "any",
            "risk_notes":    ["Dédouanement import acheteur", "Complexe pour LC"],
            "lc_compatibility": "low",
        },
        "DPU": {
            "risk_score":    4,
            "insurance_resp": "SELLER",
            "transport_doc":  "any",
            "risk_notes":    ["Ex-DAT", "Déchargement inclus — rarement dans LC"],
            "lc_compatibility": "low",
        },
        "DDP": {
            "risk_score":    5,
            "insurance_resp": "SELLER",
            "transport_doc":  "any",
            "risk_notes":    ["Risque maximal vendeur", "Déconseillé pour LC"],
            "lc_compatibility": "very_low",
        },
        "FAS": {
            "risk_score":    2,
            "insurance_resp": "BUYER",
            "transport_doc":  "bill_of_lading",
            "risk_notes":    ["Marchandise le long du navire", "Maritime uniquement"],
            "lc_compatibility": "medium",
        },
    }

    @classmethod
    def analyze(cls, incoterm: Optional[str], lc_type: str = "MT700") -> dict:
        """Retourne le profil de risque Incoterm."""
        if not incoterm:
            return {"found": False, "incoterm": None}
        upper = incoterm.upper().strip()[:3]
        profile = cls._MATRIX.get(upper)
        if not profile:
            return {"found": False, "incoterm": upper}

        result = {
            "found":          True,
            "incoterm":       upper,
            "risk_score":     profile["risk_score"],
            "insurance_resp": profile["insurance_resp"],
            "transport_doc":  profile["transport_doc"],
            "risk_notes":     profile["risk_notes"],
            "lc_compatibility": profile["lc_compatibility"],
        }

        # Alertes spécifiques
        alerts = []
        if profile["lc_compatibility"] in ("very_low", "low"):
            alerts.append(f"Incoterm {upper} peu compatible avec LC documentaire")
        if profile["insurance_resp"] == "BUYER" and lc_type in ("MT700", "MT720"):
            alerts.append(f"Assurance à charge de l'acheteur — vérifier clause :47A: et :46A: (Art.28)")
        result["alerts"] = alerts
        return result


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — ONTOLOGIE DE RISQUE (PROPAGATION PAR GRAPHE)
# ══════════════════════════════════════════════════════════════════════

class RiskOntologyGraph:
    """
    Graphe de propagation de risque entre articles UCP.
    Quand une violation est détectée sur Art.X, certains articles Y
    liés sont automatiquement marqués comme "à vérifier".

    Principe : les articles UCP forment un réseau de dépendances.
    Ex: violation Art.20 (BL) → vérifier Art.14 (examen docs) → vérifier Art.16 (non-conformités)
    """

    # Arêtes: {article_source: [(article_cible, impact_factor, raison)]}
    _EDGES: dict[str, list[tuple]] = {
        "Art.20":  [("Art.14", 0.6, "Examen BL selon Art.14"),
                    ("Art.16", 0.5, "Non-conformité BL → Art.16 applicable"),
                    ("Art.19", 0.4, "Vérifier si multimodal s'applique")],
        "Art.18b": [("Art.30", 0.7, "Limite montant → cohérence tolérance Art.30"),
                    ("Art.14", 0.4, "Vérification facture selon Art.14")],
        "Art.14k": [("Art.19", 0.5, "Expéditeur doc transport — Art.19 si multimodal"),
                    ("Art.20", 0.5, "Expéditeur BL — Art.20 si maritime")],
        "Art.38i": [("Art.18", 0.8, "Factures premier bénéficiaire → Art.18 applicables"),
                    ("Art.14", 0.5, "Examen des factures")],
        "Art.28":  [("Art.14", 0.4, "Examen police assurance"),
                    ("Art.29", 0.3, "Prorogation LC si sinistre")],
        "Art.37c": [("Art.16", 0.5, "Frais levée réserve → Art.16 en cas de non-conformité")],
        "Art.17":  [("Art.14", 0.6, "Originalité docs → Art.14 examen"),
                    ("Art.18", 0.4, "Signature facture originale")],
    }

    @classmethod
    def propagate(
        cls,
        violated_articles: list[str],
        existing_findings_codes: set[str],
    ) -> list[dict]:
        """
        Pour chaque article violé, calcule les propagations vers les articles liés.
        Retourne des alertes de propagation avec impact_factor.
        """
        propagations = []
        seen: set[str] = set()

        for art in violated_articles:
            edges = cls._EDGES.get(art, [])
            for target_art, impact, reason in edges:
                key = f"{art}->{target_art}"
                if key in seen:
                    continue
                seen.add(key)

                # Ne propager que si le target n'est pas déjà violé
                target_already_flagged = any(
                    target_art.lower().replace(".", "") in code.lower()
                    for code in existing_findings_codes
                )
                if not target_already_flagged:
                    propagations.append({
                        "from_article":  art,
                        "to_article":    target_art,
                        "impact_factor": round(impact, 2),
                        "reason":        reason,
                        "severity":      "warning" if impact >= 0.5 else "info",
                    })

        return propagations


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — SCORING BAYÉSIEN PAR CHAMP
# ══════════════════════════════════════════════════════════════════════

class BayesianFieldScorer:
    """
    Calcule la confiance globale d'un document LC via un modèle Bayésien naïf.

    Chaque champ est évalué sur :
      - Présence (0/1)
      - Cohérence syntaxique (format attendu)
      - Cohérence sémantique avec les autres champs

    La confiance globale est le produit des probabilités individuelles.
    Cela pénalise fortement les documents avec plusieurs champs dégradés.
    """

    # P(champ valide | présent et bien formé)
    _FIELD_PRIORS: dict[str, float] = {
        "lc_number":          0.98,
        "expiry_date":        0.95,
        "issue_date":         0.95,
        "applicant":          0.90,
        "beneficiary":        0.90,
        "amount":             0.97,
        "currency":           0.99,
        "goods_description":  0.80,
        "documents_required": 0.75,
        "port_loading":       0.70,
        "port_discharge":     0.70,
        "shipment_date":      0.85,
        "incoterm":           0.82,
    }

    # Patterns de validation syntaxique
    _VALIDATORS: dict[str, Any] = {
        "lc_number":    re.compile(r'^[A-Z0-9]{4,20}$'),
        "expiry_date":  re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        "issue_date":   re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        "currency":     re.compile(r'^[A-Z]{3}$'),
        "amount":       re.compile(r'^\d+(?:[.,]\d+)?$'),
        "incoterm":     re.compile(r'^(?:EXW|FCA|CPT|CIP|DAP|DPU|DDP|FAS|FOB|CFR|CIF)$'),
    }

    @classmethod
    def compute(cls, entities: dict) -> tuple[float, dict]:
        """
        Retourne (confidence globale, détail par champ).
        confidence ∈ [0, 1] — product of per-field likelihoods.
        """
        field_scores = {}
        log_confidence = 0.0  # on travaille en log pour éviter underflow

        for field, prior in cls._FIELD_PRIORS.items():
            val = entities.get(field)
            if not val:
                # Champ absent : P = 1 - prior (manque attendu pour les optionnels)
                p = max(0.3, 1.0 - prior)
                field_scores[field] = {"value": None, "p": round(p, 3), "status": "missing"}
            else:
                # Champ présent : vérification syntaxique
                validator = cls._VALIDATORS.get(field)
                if validator:
                    clean_val = str(val).strip().upper().replace(",", ".")
                    syntax_ok = bool(validator.match(clean_val))
                    p = prior if syntax_ok else prior * 0.5
                    status = "valid" if syntax_ok else "syntax_error"
                else:
                    # Pas de validator strict → confiance = prior si non vide
                    p = prior if len(str(val).strip()) > 2 else prior * 0.6
                    status = "present"
                field_scores[field] = {
                    "value":  str(val)[:30],
                    "p":      round(p, 3),
                    "status": status,
                }
            log_confidence += math.log(max(1e-6, field_scores[field]["p"]))

        global_confidence = math.exp(log_confidence / max(1, len(cls._FIELD_PRIORS)))
        return round(global_confidence, 4), field_scores


# ══════════════════════════════════════════════════════════════════════
# RÈGLES UCP 600 (v3 — avec fuzzy + patterns étendus)
# ══════════════════════════════════════════════════════════════════════

class _UCPRules:

    @staticmethod
    def check_presentation_period(period_days: Optional[int]) -> Optional[SemanticWarning]:
        if period_days is None:
            return SemanticWarning(
                code="UCP_14C_MISSING", severity="warning",
                message="Période de présentation non spécifiée (défaut UCP: 21 jours)",
                field="presentation_period", ucp_ref="Art.14c",
                suggestion="Ajouter ':48: 21 DAYS AFTER SHIPMENT DATE'",
            )
        # Les 21 jours d'Art.14(c) sont la valeur PAR DÉFAUT appliquée quand le
        # crédit ne stipule aucune période. Si le LC stipule explicitement une
        # période plus longue (champ :48:), cette période prime et reste valide :
        # une fenêtre plus large est plus favorable au bénéficiaire (risque de
        # présentation tardive RÉDUIT), ce n'est donc pas une erreur.
        # On ne signale qu'un cas réellement anormal (fenêtre très longue).
        if period_days > 60:
            return SemanticWarning(
                code="UCP_14C_UNUSUAL", severity="warning",
                message=f"Période de présentation ({period_days}j) inhabituellement longue (défaut UCP: 21j)",
                field="presentation_period", ucp_ref="Art.14c",
                suggestion="Vérifier que la période stipulée n'excède pas la validité du crédit",
            )
        if period_days > 21:
            return SemanticWarning(
                code="UCP_14C_EXTENDED", severity="info",
                message=f"Période de présentation stipulée ({period_days}j) supérieure au défaut UCP de 21j — valide, fenêtre élargie",
                field="presentation_period", ucp_ref="Art.14c",
                suggestion="Aucune action — la période stipulée par le crédit prime sur le défaut de 21j",
            )
        return None

    @staticmethod
    def check_expiry_vs_shipment(expiry, shipment) -> Optional[SemanticWarning]:
        if not expiry or not shipment:
            return None
        exp = _parse_date(expiry)
        shp = _parse_date(shipment)
        if exp and shp and exp <= shp:
            return SemanticWarning(
                code="DATE_EXPIRY_BEFORE_SHIPMENT", severity="error",
                message=f"Date expiration ({expiry}) ≤ date expédition ({shipment})",
                field="expiry_date", ucp_ref="Art.6d",
                suggestion="L'expiration doit être postérieure à l'expédition",
            )
        return None

    @staticmethod
    def check_issue_vs_expiry(issue, expiry) -> Optional[SemanticWarning]:
        if not issue or not expiry:
            return None
        iss = _parse_date(issue)
        exp = _parse_date(expiry)
        if iss and exp and iss >= exp:
            return SemanticWarning(
                code="DATE_ISSUE_AFTER_EXPIRY", severity="error",
                message=f"Date émission ({issue}) ≥ date expiration ({expiry})",
                field="issue_date", ucp_ref="Art.6",
                suggestion="Émission doit précéder l'expiration",
            )
        return None

    @staticmethod
    def check_tolerance(plus, minus) -> Optional[SemanticWarning]:
        for val in [plus, minus]:
            if val:
                m = re.match(r'(\d+(?:\.\d+)?)', str(val))
                if m and float(m.group(1)) > 10:
                    return SemanticWarning(
                        code="UCP_30B_TOLERANCE", severity="warning",
                        message=f"Tolérance {val}% dépasse 10% (UCP Art.30b)",
                        field="tolerance", ucp_ref="Art.30b",
                        suggestion="Tolérance standard max : 10%",
                    )
        return None

    @staticmethod
    def check_applicable_rules(rules) -> Optional[SemanticWarning]:
        """
        BUG #5 FIX : "UCP LATEST VERSION" et "UCP (2007 REVISION)" sont valides.
        """
        if not rules:
            return SemanticWarning(
                code="RULES_MISSING", severity="info",
                message="Règles applicables non spécifiées (:40E: absent)",
                field="applicable_rules", ucp_ref="",
                suggestion="Ajouter ':40E: UCP LATEST VERSION'",
            )
        known = re.search(
            r'UCP\s*(?:6?0?0|LATEST\s+(?:VERSION|REVISION)|\(\d{4}\s+REVISION\))'
            r'|URDG\s*7?5?8|ISP\s*98|EUCP',
            rules, re.IGNORECASE
        )
        if not known:
            return SemanticWarning(
                code="RULES_UNKNOWN", severity="warning",
                message=f"Règles non reconnues: '{rules}'",
                field="applicable_rules",
                suggestion="Utiliser: UCP LATEST VERSION, UCP 600, URDG 758, ISP 98",
            )
        return None

    @staticmethod
    def check_date_temporal_consistency(
        issue: Optional[str],
        shipment: Optional[str],
        expiry: Optional[str],
    ) -> list[SemanticWarning]:
        """
        NEW v3 — Vérifie la cohérence temporelle globale :
          issue ≤ shipment ≤ expiry (avec marges raisonnables)
        """
        warnings = []
        dates = {k: _parse_date(v) for k, v in [
            ("issue", issue), ("shipment", shipment), ("expiry", expiry)
        ]}
        if all(dates.values()):
            # Délai issue→shipment trop court (< 1 jour)
            gap_is = (dates["shipment"] - dates["issue"]).days
            if gap_is < 0:
                warnings.append(SemanticWarning(
                    code="DATE_SHIPMENT_BEFORE_ISSUE", severity="error",
                    message=f"Date expédition ({shipment}) antérieure à émission ({issue})",
                    field="shipment_date", ucp_ref="Art.6",
                    suggestion="Vérifier les dates — expédition doit être après l'émission",
                ))
            # Délai shipment→expiry trop serré (< présentation standard 21j)
            gap_se = (dates["expiry"] - dates["shipment"]).days
            if 0 < gap_se < 22:
                warnings.append(SemanticWarning(
                    code="DATE_EXPIRY_TOO_TIGHT", severity="warning",
                    message=f"Seulement {gap_se}j entre expédition et expiration — présentation difficile",
                    field="expiry_date", ucp_ref="Art.14c",
                    suggestion="Envisager un délai ≥ 21 jours entre date expédition et expiration",
                    confidence=0.85,
                ))
        return warnings


# ══════════════════════════════════════════════════════════════════════
# DÉTECTEUR DE RISQUES v3
# ══════════════════════════════════════════════════════════════════════

class _RiskDetector:

    _SANCTIONED_KEYWORDS = [
        "IRAN", "NORTH KOREA", "DPRK", "CUBA", "SYRIA", "MYANMAR",
        "VENEZUELA", "SUDAN", "ZIMBABWE", "BELARUS", "CRIMEA",
        "DONBAS", "LUHANSK", "DONETSK", "RUSSIA", "RUSSIE",
    ]

    _SENSITIVE_GOODS = [
        "WEAPON", "ARMS", "AMMUNITION", "MILITARY", "NUCLEAR",
        "CHEMICAL", "BIOLOGICAL", "DUAL.USE", "EXPLOSIVE",
        "NARCOTICS", "DRUG", "TOBACCO", "RADIOACTIVE", "PRECURSOR",
    ]

    _AML_PATTERNS = [
        (re.compile(r'\bSHELL\s+COMPANY\b', re.I),           "Shell company mentionnée",      "high"),
        (re.compile(r'\bPO\s+BOX\b',         re.I),           "Adresse PO Box uniquement",     "medium"),
        (re.compile(r'\bNOMINEE\b',           re.I),           "Nominee mentionné",             "medium"),
        (re.compile(r'\bTHIRD\s+PARTY\s+PAY', re.I),          "Paiement à un tiers",           "high"),
        (re.compile(r'\bCASH\s+AGAINST\s+DOCUMENTS\b', re.I), "Paiement cash vs docs",         "medium"),
        (re.compile(r'\bBEARER\b',            re.I),           "Instrument au porteur",         "high"),
        (re.compile(r'\bFICTITIOUS\b',        re.I),           "Mention 'fictitious'",          "high"),
        (re.compile(r'\bFREE\s+OF\s+PAYMENT\b', re.I),        "Livraison sans paiement",       "high"),
        (re.compile(r'\bUNDISCLOSED\s+PRINCIPAL\b', re.I),    "Principal non divulgué",        "high"),
    ]

    _FRAUD_PATTERNS = [
        (re.compile(r'INSPECTION.*WAIVED|WAIVE.*INSPECTION', re.I),
         "Inspection renoncée",           "high",   "FRAUD_INSPECTION_WAIVED"),
        (re.compile(r'WITHOUT.*ORIGINAL|ORIGINAL.*NOT.*REQUIRED', re.I),
         "Originaux non requis",          "high",   "FRAUD_NO_ORIGINALS"),
        (re.compile(r'ANY\s+DOCUMENT|ALL\s+DOCUMENTS\s+ACCEPTABLE', re.I),
         "Acceptation sans vérification", "medium", "FRAUD_ANY_DOCS"),
        (re.compile(r'BENEFICIARY.*CERTIF.*COMPLY', re.I),
         "Auto-certification bénéficiaire","medium", "FRAUD_SELF_CERT"),
        (re.compile(r'CLEAN\s+ON\s+BOARD.*CLAUSE\s+ADDED', re.I),
         "BL 'clean on board' + réserve", "high",   "FRAUD_BL_CLAUSE"),
        # NEW v3
        (re.compile(r'TELEX\s+RELEASE|SEAWAY\s+BILL.*ORIGINAL\s+NOT\s+REQUIRED', re.I),
         "BL Telex release suspect",      "medium", "FRAUD_TELEX_RELEASE"),
        (re.compile(r'BACK\s+TO\s+BACK|BACK-TO-BACK', re.I),
         "LC back-to-back détecté",       "medium", "FRAUD_BACK_TO_BACK"),
    ]

    @classmethod
    def check_sanctions(cls, text: str) -> list[RiskFlag]:
        flags = []
        upper = text.upper()
        seen = set()
        for kw in cls._SANCTIONED_KEYWORDS:
            if kw in upper and kw not in seen:
                seen.add(kw)
                flags.append(RiskFlag(
                    code="SANCTIONS_MATCH",
                    description=f"Pays/territoire sous sanctions: {kw}",
                    severity="high", ucp_ref="",
                    category="sanctions",
                    recommendation=f"Vérifier si '{kw}' est sous sanctions OFAC/EU/UN",
                    confidence=0.95,
                ))
        return flags

    @classmethod
    def check_sensitive_goods(cls, goods) -> list[RiskFlag]:
        if not goods:
            return []
        flags = []
        upper = goods.upper()
        for kw in cls._SENSITIVE_GOODS:
            pattern = kw.replace(".", r"\W?")
            if re.search(rf'\b{pattern}\b', upper):
                severity = "high" if kw in (
                    "WEAPON", "ARMS", "NARCOTICS", "DRUG", "NUCLEAR", "PRECURSOR"
                ) else "medium"
                flags.append(RiskFlag(
                    code="SENSITIVE_GOODS",
                    description=f"Marchandises sensibles: {kw}",
                    severity=severity, ucp_ref="",
                    category="goods",
                    recommendation="Vérifier licences d'exportation et embargo applicable",
                ))
        return flags

    @classmethod
    def check_aml_patterns(cls, text: str) -> list[RiskFlag]:
        flags = []
        for pattern, desc, severity in cls._AML_PATTERNS:
            if pattern.search(text):
                flags.append(RiskFlag(
                    code="AML_PATTERN",
                    description=desc, severity=severity, ucp_ref="",
                    category="aml",
                    recommendation="Déclencher procédure KYC renforcée",
                ))
        return flags

    @classmethod
    def check_fraud_patterns(cls, text: str, entities: Any) -> list[dict]:
        indicators = []
        docs_text = ""
        if entities:
            docs_text = str(_get_field(entities, "documents_required") or "").upper()
            docs_text += " " + str(_get_field(entities, "goods_description") or "").upper()

        for pattern, desc, severity, code in cls._FRAUD_PATTERNS:
            if pattern.search(text):
                if code == "FRAUD_NO_ORIGINALS" and "ORIGINAL" in (text.upper() + docs_text):
                    continue
                indicators.append({"code": code, "description": desc, "severity": severity})
        return indicators

    @staticmethod
    def check_round_amount(amount) -> Optional[RiskFlag]:
        if not amount:
            return None
        m = re.search(r'[\d,]+', str(amount))
        if m:
            clean = m.group().replace(",", "")
            try:
                val = int(clean)
                if val >= 1_000_000 and val % 1_000_000 == 0:
                    return RiskFlag(
                        code="ROUND_AMOUNT",
                        description=f"Montant très rond ({amount}) — structuring possible",
                        severity="low", ucp_ref="", category="aml",
                        recommendation="Vérifier la justification économique",
                    )
            except ValueError:
                pass
        return None

    @staticmethod
    def check_very_short_validity(issue, expiry) -> Optional[RiskFlag]:
        if not issue or not expiry:
            return None
        iss = _parse_date(issue)
        exp = _parse_date(expiry)
        if iss and exp:
            delta = (exp - iss).days
            if 0 < delta < 30:
                return RiskFlag(
                    code="SHORT_VALIDITY",
                    description=f"Validité très courte ({delta} jours)",
                    severity="medium",
                    ucp_ref="UCP 600 Art.6",
                    category="compliance",
                    recommendation="Vérifier si le délai est réaliste pour présentation des documents",
                )
        return None


# ══════════════════════════════════════════════════════════════════════
# FUZZY MATCHING UTILITAIRE
# ══════════════════════════════════════════════════════════════════════

def _fuzzy_contains(text: str, query: str, threshold: int = 82) -> bool:
    """
    Retourne True si query est présent dans text avec une similarité ≥ threshold.
    Fallback sur exact match si RapidFuzz non disponible.
    """
    if not text or not query:
        return False
    text_upper  = text.upper()
    query_upper = query.upper()
    if query_upper in text_upper:
        return True
    if not _FUZZY_AVAILABLE or not hasattr(_rfuzz, 'partial_ratio'):
        return False
    # Sliding window sur le texte pour chercher la meilleure correspondance
    q_len = len(query_upper)
    best = 0
    try:
        for i in range(0, max(1, len(text_upper) - q_len + 1), max(1, q_len // 3)):
            window = text_upper[i:i + q_len + 10]
            score = _rfuzz.partial_ratio(query_upper, window)
            if score > best:
                best = score
    except Exception:
        return False
    return best >= threshold


# ══════════════════════════════════════════════════════════════════════
# GOODS-DOCS MATCHER (rule-based, replaces broken embedding approach)
# ══════════════════════════════════════════════════════════════════════

class _GoodsDocsMatcher:
    """Rule-based goods-category → expected docs checker (ICC/ISBP 745).

    Replaces embedding cosine-similarity which always returns < 0.15
    because goods text and document-list text live in disjoint semantic spaces.
    """

    _GOODS_EXPECTED: list = [
        (
            re.compile(
                r'\b(food|fish|seafood|seafoods|meat|poultry|dairy|milk|cheese|'
                r'egg|grain|cereal|wheat|rice|flour|sugar|oil|spice|beverage|'
                r'alimentaire|poisson|viande|lait|fromage|cereale|sucre|huile)\b',
                re.I,
            ),
            ["SANITARY CERTIFICATE", "HEALTH CERTIFICATE", "PHYTOSANITARY", "VETERINARY", "FUMIGATION"],
            "alimentaire",
        ),
        (
            re.compile(
                r'\b(chemical|acid|alkali|solvent|toxic|hazardous|dangerous goods|'
                r'chimique|acide|alcali|solvant|toxique|dangereux|produit chimique)\b',
                re.I,
            ),
            ["MSDS", "SDS", "IMDG", "DANGEROUS GOODS DECLARATION", "MATERIAL SAFETY DATA"],
            "chimique",
        ),
        (
            re.compile(
                r'\b(pharmaceutical|medicine|drug|medical device|vaccine|tablet|capsule|'
                r'pharmaceutique|medicament|vaccin|dispositif medical)\b',
                re.I,
            ),
            ["GMP CERTIFICATE", "FDA", "CE MARKING", "ANALYSIS CERTIFICATE", "HEALTH AUTHORITY"],
            "pharmaceutique",
        ),
        (
            re.compile(
                r'\b(textile|fabric|garment|clothing|apparel|yarn|fiber|cotton|wool|silk|'
                r'tissu|vetement|habillement|fil|fibre|coton|laine|soie)\b',
                re.I,
            ),
            ["INSPECTION CERTIFICATE", "GSP CERTIFICATE", "COUNTRY OF ORIGIN", "FORM A"],
            "textile",
        ),
        (
            re.compile(
                r'\b(machinery|equipment|engine|turbine|generator|pump|compressor|'
                r'machine|equipement|moteur|turbine|generateur|pompe|compresseur)\b',
                re.I,
            ),
            ["MANUFACTURER CERTIFICATE", "TEST REPORT", "INSPECTION CERTIFICATE", "CE CERTIFICATE"],
            "machine",
        ),
        (
            re.compile(
                r'\b(agricultural|crop|fertilizer|pesticide|seed|plant|lumber|timber|wood|'
                r'agricole|culture|engrais|pesticide|graine|plante|bois)\b',
                re.I,
            ),
            ["PHYTOSANITARY CERTIFICATE", "FUMIGATION CERTIFICATE", "QUARANTINE"],
            "agricole",
        ),
        (
            re.compile(
                r'\b(petroleum|crude oil|fuel|diesel|gasoline|kerosene|lng|lpg|'
                r'petrole|brut|carburant|diesel|essence|kerosene)\b',
                re.I,
            ),
            ["SGS REPORT", "QUALITY CERTIFICATE", "QUANTITY CERTIFICATE", "INSPECTION REPORT"],
            "petroleum",
        ),
        (
            re.compile(
                r'\b(steel|iron|metal|aluminium|copper|zinc|alloy|pipe|tube|bar|coil|'
                r'acier|fer|metal|aluminium|cuivre|zinc|alliage|tuyau|tube|barre)\b',
                re.I,
            ),
            ["MILL CERTIFICATE", "TEST CERTIFICATE", "MATERIAL CERTIFICATE", "QUALITY CERTIFICATE"],
            "metal",
        ),
        (
            re.compile(
                r'\b(timber|hardwood|softwood|plywood|mdf|chipboard|veneer|'
                r'bois dur|bois tendre|contreplaqu|fibre de bois)\b',
                re.I,
            ),
            ["PHYTOSANITARY CERTIFICATE", "CITES PERMIT", "FOREST CERTIFICATE", "FSC"],
            "bois",
        ),
    ]

    @classmethod
    def check(cls, goods_text: str, docs_text: str):
        """Return (score, warning_or_None).

        score=1.0  → no mismatch detected
        score=0.35 → category-specific docs completely missing
        score=0.65-1.0 → partial match
        """
        if not goods_text or not docs_text:
            return 1.0, None

        docs_up = docs_text.upper()

        for pattern, expected_docs, category in cls._GOODS_EXPECTED:
            if not pattern.search(goods_text):
                continue
            found = [d for d in expected_docs if d in docs_up]
            if not found:
                return 0.35, SemanticWarning(
                    code="GOODS_DOCS_MISSING_EXPECTED",
                    severity="warning",
                    message=(
                        f"Marchandises catégorie '{category}' détectées dans :45A: "
                        f"mais aucun document spécifique requis dans :46A:"
                    ),
                    field="documents_required",
                    ucp_ref="Art.14",
                    suggestion=f"Ajouter dans :46A: : {' / '.join(expected_docs[:2])}",
                    confidence=0.85,
                    source="rule",
                )
            partial_score = min(1.0, 0.65 + len(found) / len(expected_docs) * 0.35)
            return partial_score, None

        return 1.0, None


# ══════════════════════════════════════════════════════════════════════
# COHERENCE SEMANTIQUE (embeddings optionnel)
# ══════════════════════════════════════════════════════════════════════

class _SemanticCoherenceChecker:

    def __init__(self) -> None:
        self._encoder = None
        self._available = False
        try:
            # Cache partagé process-wide (même instance que le classifieur de
            # clauses) + révision épinglée + CPU forcé via le registre central.
            from ..common.swift_st_cache import get_sentence_transformer
            self._encoder = get_sentence_transformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._available = self._encoder is not None
        except (ImportError, Exception):
            pass

    @property
    def available(self) -> bool:
        return self._available

    def similarity(self, text1: str, text2: str) -> float:
        if not self._available or not self._encoder:
            return 0.5
        try:
            import numpy as np
            embs = self._encoder.encode([text1, text2])
            norm = np.linalg.norm(embs[0]) * np.linalg.norm(embs[1])
            return float((embs[0] @ embs[1]) / max(1e-8, norm))
        except Exception:
            return 0.5

    def check_parties_similarity(self, applicant: str, beneficiary: str):
        sim = self.similarity(applicant, beneficiary)
        if sim > 0.85:
            return sim, SemanticWarning(
                code="SUSPICIOUS_PARTIES_SIMILARITY", severity="warning",
                message=f"Applicant et Bénéficiaire très similaires (sim={sim:.2f}) — LC fictive possible",
                field="applicant/beneficiary",
                suggestion="Vérifier que les deux entités sont distinctes",
                confidence=sim,
                source="embedding",
            )
        return sim, None

    def check_goods_vs_docs(self, goods: str, docs: str):
        # Delegate to rule-based matcher — embedding similarity between goods text
        # and doc-list text is always < 0.15 (semantically disjoint by design).
        return _GoodsDocsMatcher.check(goods, docs)


# ══════════════════════════════════════════════════════════════════════
# SCORING ML (optionnel)
# ══════════════════════════════════════════════════════════════════════

class _MLRiskScorer:

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model = None
        self._available = False
        if not model_path:
            return
        try:
            import joblib
            self._model = joblib.load(f"{model_path}/risk_clf.pkl")
            if self._model is None:
                return
            self._available = True
            logger.info("MLRiskScorer: modèle chargé depuis '%s'", model_path)
        except Exception as exc:
            logger.warning("MLRiskScorer: non disponible: %s", exc)

    def predict(self, text: str, entities: dict) -> Optional[dict[str, float]]:
        if not self._available or not self._model:
            return None
        try:
            features = self._build_features(text, entities)
            from sklearn.pipeline import Pipeline as SKPipeline
            if isinstance(self._model, SKPipeline):
                proba   = self._model.predict_proba([features])[0]
                classes = self._model.classes_
            else:
                return None
            return {str(cls).upper(): float(p) for cls, p in zip(classes, proba)}
        except Exception as exc:
            logger.debug("MLRiskScorer error: %s", exc)
            return None

    @staticmethod
    def _build_features(text: str, entities: dict) -> str:
        parts = [text[:500]]
        for k, v in entities.items():
            if v:
                parts.append(f"{k}:{str(v)[:50]}")
        return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ══════════════════════════════════════════════════════════════════════

def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%y%m%d", "%Y%m%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _get_field(entities: Any, field_name: str) -> Optional[str]:
    if hasattr(entities, field_name):
        return getattr(entities, field_name)
    if isinstance(entities, dict):
        return entities.get(field_name)
    return None


def _entities_to_dict(entities: Any) -> dict:
    if isinstance(entities, dict):
        return entities
    result = {}
    for attr in ["swift_type", "applicant", "beneficiary", "issuing_bank",
                 "amount", "currency", "expiry_date", "issue_date",
                 "shipment_date", "goods_description", "incoterm",
                 "port_loading", "port_discharge", "applicable_rules",
                 # Champs utiles pour contradictions tag ↔ clause
                 "transhipment", "partial_shipments", "confirmation",
                 "available_by", "tolerance_plus", "tolerance_minus"]:
        val = getattr(entities, attr, None)
        if val:
            result[attr] = val
    return result


# ══════════════════════════════════════════════════════════════════════
# AXE 1 — COHÉRENCE :46A: DOCUMENTS vs CONTEXTE COMMERCIAL (v4)
# ══════════════════════════════════════════════════════════════════════

class DocumentContextValidator:
    """
    Valide la cohérence entre les documents requis (:46A:) et le contexte commercial :
      - Incoterm vs type de transport doc (ex: EXW + BL = contradiction)
      - CIF/CIP sans police d'assurance dans :46A: (Art.28)
      - Marchandises réfrigérées sans certificat sanitaire/phytosanitaire
      - Produits chimiques/dangereux sans MSDS ou certificat IMDG
    Technique : pattern matching enrichi + scoring de couverture documentaire.
    """

    # Incoterms maritime uniquement — un BL ne peut exister nativement avec EXW/DAP/DDP
    _INCOTERMS_NO_MARITIME_BL: set[str] = {"EXW", "DAP", "DPU", "DDP", "FCA", "CPT", "CIP"}
    _INCOTERMS_MARITIME: set[str]        = {"FOB", "CFR", "CIF", "FAS"}
    _INCOTERMS_SELLER_INSURES: set[str]  = {"CIF", "CIP"}

    # Marchandises nécessitant un certificat sanitaire/phytosanitaire
    _COLD_CHAIN_GOODS = re.compile(
        r'\b(fish|seafood|meat|poultry|dairy|milk|cheese|frozen|refrigerat|perishable'
        r'|poisson|viande|volaille|lait|fromage|surgel|réfrigér|périssable'
        r'|fruits?\s+de\s+mer|produits?\s+frais)\b',
        re.IGNORECASE,
    )
    # Marchandises dangereuses/chimiques
    _HAZMAT_GOODS = re.compile(
        r'\b(chemical|acid|flammable|explosive|toxic|corrosive|hazardous|dangerous\s+goods'
        r'|imdg|msds|sds|un\s*\d{4}|chimique|acide|inflammable|explosif|toxique|corrosif'
        r'|matières?\s+dangereuse|produits?\s+chimiques?)\b',
        re.IGNORECASE,
    )
    # Détection de présence de BL dans :46A:
    _BL_PATTERNS = re.compile(
        r'\b(bill\s+of\s+lading|b/?l\b|connaissement|lading)\b',
        re.IGNORECASE,
    )
    # Détection police assurance dans :46A:
    _INSURANCE_PATTERNS = re.compile(
        r'\b(insurance\s+policy|insurance\s+certificate|police\s+d.assurance'
        r'|certificate\s+of\s+insurance|assurance|insur)\b',
        re.IGNORECASE,
    )
    # Certificat sanitaire / phytosanitaire
    _SANITARY_PATTERNS = re.compile(
        r'\b(sanitary|phytosanitary|health\s+certificate|certificat\s+sanitaire'
        r'|certificat\s+phytosanitaire|veterinary|vétérinaire|hygiene|hygiène)\b',
        re.IGNORECASE,
    )
    # MSDS / certificat IMDG
    _MSDS_PATTERNS = re.compile(
        r'\b(msds|sds|material\s+safety|fiche\s+de\s+données|imdg|dangerous\s+goods\s+cert'
        r'|certificat\s+imdg|dgd\b)\b',
        re.IGNORECASE,
    )

    @classmethod
    def validate(
        cls,
        entities: Any,
        incoterm: Optional[str],
        documents_text: str,
        goods_text: str,
    ) -> list[SemanticWarning]:
        warnings: list[SemanticWarning] = []
        inc = (incoterm or "").upper().strip()[:3]
        docs_up = (documents_text or "").upper()
        goods_up = (goods_text or "").upper()

        # ── Règle 1 : EXW/DAP/DDP + Bill of Lading exigé ──────────────
        if inc in cls._INCOTERMS_NO_MARITIME_BL and cls._BL_PATTERNS.search(documents_text or ""):
            warnings.append(SemanticWarning(
                code="DOC_BL_INCOTERM_CONTRADICTION",
                severity="error",
                message=(
                    f"Incoterm {inc} incompatible avec un Bill of Lading maritime dans :46A:. "
                    f"Sous {inc}, le vendeur ne gère pas le transport maritime — un BL n'est pas naturellement produit."
                ),
                field="documents_required",
                ucp_ref="Art.20 / Art.5",
                suggestion=f"Remplacer le BL par un document de transport multimodal (Art.19) ou supprimer l'exigence BL pour {inc}.",
                confidence=0.95,
                source="rule",
            ))

        # ── Règle 2 : CIF/CIP sans police d'assurance dans :46A: ───────
        if inc in cls._INCOTERMS_SELLER_INSURES and not cls._INSURANCE_PATTERNS.search(documents_text or ""):
            warnings.append(SemanticWarning(
                code="DOC_CIF_NO_INSURANCE",
                severity="error",
                message=(
                    f"Incoterm {inc} oblige le vendeur à fournir une assurance (UCP Art.28), "
                    f"mais aucune police d'assurance n'est exigée dans :46A:."
                ),
                field="documents_required",
                ucp_ref="Art.28",
                suggestion="Ajouter dans :46A: 'INSURANCE POLICY/CERTIFICATE FOR 110% OF INVOICE VALUE'.",
                confidence=0.93,
                source="rule",
            ))

        # ── Règle 3 : Marchandises réfrigérées sans certificat sanitaire ─
        if cls._COLD_CHAIN_GOODS.search(goods_text or "") and not cls._SANITARY_PATTERNS.search(documents_text or ""):
            warnings.append(SemanticWarning(
                code="DOC_COLD_CHAIN_NO_SANITARY",
                severity="warning",
                message=(
                    "Marchandises périssables/réfrigérées détectées dans :45A:, "
                    "mais aucun certificat sanitaire ou phytosanitaire exigé dans :46A:."
                ),
                field="documents_required",
                ucp_ref="Art.14",
                suggestion="Ajouter 'SANITARY/PHYTOSANITARY CERTIFICATE' ou 'HEALTH CERTIFICATE' dans :46A:.",
                confidence=0.88,
                source="rule",
            ))

        # ── Règle 4 : Produits chimiques/dangereux sans MSDS/IMDG ───────
        if cls._HAZMAT_GOODS.search(goods_text or "") and not cls._MSDS_PATTERNS.search(documents_text or ""):
            warnings.append(SemanticWarning(
                code="DOC_HAZMAT_NO_MSDS",
                severity="warning",
                message=(
                    "Marchandises dangereuses/chimiques détectées dans :45A:, "
                    "mais aucun MSDS/SDS ou certificat IMDG exigé dans :46A:."
                ),
                field="documents_required",
                ucp_ref="Art.14",
                suggestion="Ajouter 'MATERIAL SAFETY DATA SHEET (MSDS)' et 'IMDG DANGEROUS GOODS CERTIFICATE' dans :46A:.",
                confidence=0.87,
                source="rule",
            ))

        return warnings


# ══════════════════════════════════════════════════════════════════════
# AXE 2 — COHÉRENCE MONTANT / TOLÉRANCE / QUANTITÉ (v4)
# ══════════════════════════════════════════════════════════════════════

class AmountToleranceValidator:
    """
    Valide la cohérence entre montant, tolérance et quantité :
      - ABOUT/CIRCA dans :45A: → tolérance implicite ±10% (UCP Art.30b)
      - :39B: NOT EXCEEDING + :43P: ALLOWED = contradiction Art.30
      - Écart devise :32B: vs pays bénéficiaire = alerte change
    Technique : regex + table de correspondance devise/pays ISO 4217.
    """

    _ABOUT_CIRCA = re.compile(r'\b(about|circa|approximately|environ|approximativement)\b', re.IGNORECASE)

    # Correspondances simplifiées devise → pays ISO (non exhaustif, couvre les LC courants)
    _CURRENCY_COUNTRY_MAP: dict[str, set[str]] = {
        "USD": {"US", "USA", "UNITED STATES", "DOLLAR"},
        "EUR": {"FR", "DE", "IT", "ES", "NL", "BE", "PT", "AT", "FI", "GR", "IE",
                "FRANCE", "GERMANY", "ITALY", "SPAIN", "EUROPE", "EUROZONE"},
        "GBP": {"GB", "UK", "UNITED KINGDOM", "ENGLAND", "BRITAIN"},
        "JPY": {"JP", "JAPAN"},
        "CNY": {"CN", "CHINA", "CHINESE"},
        "CHF": {"CH", "SWITZERLAND", "SUISSE"},
        "AED": {"AE", "UAE", "EMIRATES", "DUBAI", "ABU DHABI"},
        "SAR": {"SA", "SAUDI", "ARABIE"},
        "TND": {"TN", "TUNISIA", "TUNISIE"},
        "MAD": {"MA", "MOROCCO", "MAROC"},
        "DZD": {"DZ", "ALGERIA", "ALGÉRIE"},
        "EGP": {"EG", "EGYPT", "ÉGYPTE"},
        "INR": {"IN", "INDIA", "INDE"},
        "SGD": {"SG", "SINGAPORE", "SINGAPOUR"},
        "HKD": {"HK", "HONG KONG"},
        "KRW": {"KR", "KOREA", "CORÉE"},
        "TRY": {"TR", "TURKEY", "TURQUIE"},
        "BRL": {"BR", "BRAZIL", "BRÉSIL"},
    }
    # Devises "neutres" ou internationalement acceptées — pas d'alerte géo
    _NEUTRAL_CURRENCIES: set[str] = {"USD", "EUR", "GBP", "CHF"}

    @classmethod
    def validate(
        cls,
        entities: Any,
        goods_text: str,
        raw_text: str,
    ) -> list[SemanticWarning]:
        warnings: list[SemanticWarning] = []

        # ── Règle 1 : ABOUT/CIRCA dans :45A: → tolérance implicite ±10% ─
        if cls._ABOUT_CIRCA.search(goods_text or ""):
            tol_plus  = _get_field(entities, "tolerance_plus")
            tol_minus = _get_field(entities, "tolerance_minus")
            if not tol_plus and not tol_minus:
                warnings.append(SemanticWarning(
                    code="TOLERANCE_IMPLICIT_ABOUT",
                    severity="warning",
                    message=(
                        "Mention 'ABOUT'/'CIRCA' détectée dans :45A:. Selon UCP Art.30b, "
                        "une tolérance implicite de ±10% s'applique même si :39A: est absent. "
                        "Le montant LC doit couvrir 110% de la valeur nominale."
                    ),
                    field="goods_description",
                    ucp_ref="Art.30b",
                    suggestion="Ajouter ':39A: 10/10' ou s'assurer que le montant :32B: intègre déjà la marge de 10%.",
                    confidence=0.92,
                    source="rule",
                ))

        # ── Règle 2 : :39B: NOT EXCEEDING + :43P: ALLOWED ───────────────
        max_credit_field = _get_field(entities, "max_credit_amount") or ""
        partial_ship     = (_get_field(entities, "partial_shipments") or "").upper()
        not_exceeding    = "NOT EXCEEDING" in (max_credit_field or "").upper() or \
                           bool(re.search(r':39B:\s*NOT\s+EXCEEDING', raw_text or "", re.I))
        if not_exceeding and partial_ship == "ALLOWED":
            warnings.append(SemanticWarning(
                code="AMOUNT_NOT_EXCEEDING_VS_PARTIAL",
                severity="error",
                message=(
                    ":39B: NOT EXCEEDING combiné à :43P: ALLOWED est une contradiction UCP Art.30c. "
                    "Les envois partiels autorisés rendent impossible le contrôle du plafond montant par présentation."
                ),
                field="partial_shipments",
                ucp_ref="Art.30c",
                suggestion="Soit supprimer :39B:, soit passer :43P: à NOT ALLOWED.",
                confidence=0.96,
                source="rule",
            ))

        # ── Règle 3 : Écart devise vs pays bénéficiaire ──────────────────
        currency    = (_get_field(entities, "currency") or "").upper().strip()
        beneficiary = (_get_field(entities, "beneficiary") or "").upper()

        if currency and currency not in cls._NEUTRAL_CURRENCIES and beneficiary:
            expected_countries = cls._CURRENCY_COUNTRY_MAP.get(currency, set())
            if expected_countries:
                match_found = any(country in beneficiary for country in expected_countries)
                if not match_found:
                    warnings.append(SemanticWarning(
                        code="CURRENCY_COUNTRY_MISMATCH",
                        severity="warning",
                        message=(
                            f"Devise :32B: ({currency}) peu cohérente avec l'adresse du bénéficiaire. "
                            f"Risque de change non couvert ou signal de sous/sur-facturation."
                        ),
                        field="currency",
                        ucp_ref="",
                        suggestion=(
                            f"Vérifier la justification économique de l'utilisation de {currency} "
                            f"pour ce bénéficiaire. Envisager une couverture de change."
                        ),
                        confidence=0.75,
                        source="rule",
                    ))

        return warnings


# ══════════════════════════════════════════════════════════════════════
# AXE 3 — COHÉRENCE TEMPORELLE ÉTENDUE (v4)
# ══════════════════════════════════════════════════════════════════════

class ExtendedTemporalValidator:
    """
    Validations temporelles avancées :
      - Transit time estimé par route maritime (port loading → port discharge)
      - Fenêtre saisonnière pour marchandises agricoles
      - Expiry place chez banque émettrice vs correspondante distante (+5j business)
    Technique : base de données de routes maritimes + calendrier saisonnier.
    """

    # Transit times estimés (jours) par paires de régions maritimes
    # Format: (région_chargement, région_déchargement) → (min_days, max_days)
    _TRANSIT_MATRIX: dict[tuple[str, str], tuple[int, int]] = {
        ("ASIA",    "EUROPE"):    (25, 35),
        ("CHINA",   "EUROPE"):    (28, 38),
        ("CHINA",   "ROTTERDAM"): (28, 35),
        ("SHANGHAI","ROTTERDAM"): (28, 35),
        ("ASIA",    "US"):        (14, 21),
        ("CHINA",   "US"):        (14, 21),
        ("CHINA",   "USA"):       (14, 21),
        ("EUROPE",  "US"):        (7,  12),
        ("EUROPE",  "USA"):       (7,  12),
        ("MIDDLE",  "EUROPE"):    (18, 25),
        ("GULF",    "EUROPE"):    (18, 25),
        ("INDIA",   "EUROPE"):    (20, 28),
        ("AFRICA",  "EUROPE"):    (10, 20),
        ("BRAZIL",  "EUROPE"):    (14, 20),
        ("AMERICA", "EUROPE"):    (10, 18),
    }

    # Mots-clés pour identifier les régions portuaires
    _REGION_KEYWORDS: dict[str, list[str]] = {
        "CHINA":    ["SHANGHAI", "SHENZHEN", "GUANGZHOU", "NINGBO", "TIANJIN", "QINGDAO", "XIAMEN", "CHINA"],
        "ASIA":     ["SINGAPORE", "BUSAN", "TOKYO", "YOKOHAMA", "HONG KONG", "TAIPEI", "JAKARTA", "BANGKOK", "VIETNAM", "KOREA"],
        "EUROPE":   ["ROTTERDAM", "HAMBURG", "ANTWERP", "MARSEILLE", "GENOA", "BARCELONA", "LE HAVRE", "FELIXSTOWE", "EUROPE"],
        "ROTTERDAM":["ROTTERDAM"],
        "SHANGHAI": ["SHANGHAI"],
        "US":       ["NEW YORK", "LOS ANGELES", "LONG BEACH", "SAVANNAH", "HOUSTON", "SEATTLE", "MIAMI"],
        "USA":      ["USA", "UNITED STATES"],
        "MIDDLE":   ["DUBAI", "JEBEL ALI", "ABU DHABI", "MUSCAT", "OMAN", "BAHRAIN", "KUWAIT"],
        "GULF":     ["GULF", "ARABIAN"],
        "INDIA":    ["MUMBAI", "NHAVA SHEVA", "CHENNAI", "KOLKATA", "INDIA"],
        "AFRICA":   ["DAKAR", "ABIDJAN", "LAGOS", "DURBAN", "MOMBASA", "CASABLANCA", "AFRICA"],
        "BRAZIL":   ["SANTOS", "RIO", "BRAZIL", "BRASIL"],
        "AMERICA":  ["MEXICO", "COLOMBIA", "CHILE", "ARGENTINA", "PERU"],
    }

    # Marchandises saisonnières → mois d'embarquement cohérents (1=Jan, 12=Dec)
    _SEASONAL_GOODS: list[tuple[re.Pattern, list[int], str]] = [
        (re.compile(r'\b(agrumes|citrus|orange|lemon|mandarin|clementine)\b', re.I),
         [10, 11, 12, 1, 2, 3], "Agrumes — saison Oct-Mar"),
        (re.compile(r'\b(wheat|ble|blé|grain|céréale|cereale)\b', re.I),
         [6, 7, 8, 9], "Blé/céréales — récolte Jun-Sep"),
        (re.compile(r'\b(corn|maize|mais|maïs)\b', re.I),
         [9, 10, 11], "Maïs — récolte Sep-Nov"),
        (re.compile(r'\b(cotton|coton)\b', re.I),
         [10, 11, 12, 1], "Coton — récolte Oct-Jan"),
        (re.compile(r'\b(olive|huile\s+d.olive|olive\s+oil)\b', re.I),
         [11, 12, 1, 2, 3], "Olive/huile — récolte Nov-Mar"),
        (re.compile(r'\b(strawberry|fraise|berry|baie)\b', re.I),
         [4, 5, 6, 7], "Fraises/baies — saison Avr-Jul"),
    ]

    @classmethod
    def _detect_region(cls, port: str) -> Optional[str]:
        if not port:
            return None
        port_up = port.upper()
        for region, keywords in cls._REGION_KEYWORDS.items():
            if any(kw in port_up for kw in keywords):
                return region
        return None

    @classmethod
    def _estimate_transit(cls, port_loading: str, port_discharge: str) -> Optional[tuple[int, int]]:
        reg_l = cls._detect_region(port_loading)
        reg_d = cls._detect_region(port_discharge)
        if not reg_l or not reg_d:
            return None
        key = (reg_l, reg_d)
        if key in cls._TRANSIT_MATRIX:
            return cls._TRANSIT_MATRIX[key]
        # Essai symétrique
        key_rev = (reg_d, reg_l)
        if key_rev in cls._TRANSIT_MATRIX:
            t = cls._TRANSIT_MATRIX[key_rev]
            return t
        return None

    @classmethod
    def validate(
        cls,
        entities: Any,
        goods_text: str,
        raw_text: str,
    ) -> list[SemanticWarning]:
        warnings: list[SemanticWarning] = []

        shipment_date_str = _get_field(entities, "shipment_date")
        expiry_date_str   = _get_field(entities, "expiry_date")
        port_loading      = _get_field(entities, "port_loading") or ""
        port_discharge    = _get_field(entities, "port_discharge") or ""

        shipment_date = _parse_date(shipment_date_str)
        expiry_date   = _parse_date(expiry_date_str)

        # ── Règle 1 : Transit time estimé vs fenêtre expédition→expiration ─
        if shipment_date and expiry_date and port_loading and port_discharge:
            window_days = (expiry_date - shipment_date).days
            transit = cls._estimate_transit(port_loading, port_discharge)
            if transit:
                min_transit, max_transit = transit
                # Fenêtre = transit + présentation minimale (21j UCP)
                min_needed = min_transit + 21
                if window_days < min_transit:
                    warnings.append(SemanticWarning(
                        code="TRANSIT_TIME_IMPOSSIBLE",
                        severity="error",
                        message=(
                            f"Fenêtre expédition→expiration ({window_days}j) inférieure au transit maritime estimé "
                            f"{port_loading}→{port_discharge} ({min_transit}–{max_transit}j). "
                            f"La présentation de documents sera structurellement impossible."
                        ),
                        field="expiry_date",
                        ucp_ref="Art.14c / Art.6",
                        suggestion=(
                            f"Prolonger la validité d'au moins {min_needed - window_days} jours "
                            f"(transit {min_transit}j + présentation 21j = {min_needed}j minimum)."
                        ),
                        confidence=0.91,
                        source="rule",
                    ))
                elif window_days < min_needed:
                    warnings.append(SemanticWarning(
                        code="TRANSIT_TIME_VERY_TIGHT",
                        severity="warning",
                        message=(
                            f"Fenêtre expédition→expiration ({window_days}j) très serrée pour "
                            f"{port_loading}→{port_discharge} (transit estimé {min_transit}–{max_transit}j + 21j présentation = {min_needed}j)."
                        ),
                        field="expiry_date",
                        ucp_ref="Art.14c",
                        suggestion=f"Recommander {min_needed}j minimum entre date expédition et expiration.",
                        confidence=0.85,
                        source="rule",
                    ))

        # ── Règle 2 : Cohérence saisonnière des marchandises agricoles ──
        if shipment_date and goods_text:
            shipment_month = shipment_date.month
            for pattern, valid_months, label in cls._SEASONAL_GOODS:
                if pattern.search(goods_text):
                    if shipment_month not in valid_months:
                        months_str = ", ".join(str(m) for m in valid_months)
                        warnings.append(SemanticWarning(
                            code="SEASONAL_GOODS_MISMATCH",
                            severity="warning",
                            message=(
                                f"Incohérence saisonnière : {label} mais date d'embarquement "
                                f"en mois {shipment_month} (mois cohérents : {months_str})."
                            ),
                            field="shipment_date",
                            ucp_ref="Art.14",
                            suggestion="Vérifier la date de dernier embarquement :44C: ou la description des marchandises.",
                            confidence=0.78,
                            source="rule",
                        ))
                    break  # une seule alerte saisonnière

        # ── Règle 3 : Expiry place banque émettrice vs correspondante ────
        expiry_place    = (_get_field(entities, "expiry_place") or "").upper()
        issuing_bank    = (_get_field(entities, "issuing_bank") or "").upper()
        available_with  = (_get_field(entities, "available_with") or "").upper()

        # Si expiry place et banque désignée/émettrice semblent dans des pays différents
        if expiry_place and issuing_bank:
            # Heuristique : si le pays de la banque émettrice n'est pas mentionné dans expiry_place
            # et qu'une banque correspondante distante est désignée, recommander +5j
            if available_with and "ANY BANK" not in available_with and issuing_bank not in expiry_place:
                warnings.append(SemanticWarning(
                    code="EXPIRY_PLACE_DISTANT_BANK",
                    severity="info",
                    message=(
                        f"Expiry place '{expiry_place}' semble distant de la banque désignée. "
                        f"Recommander +5 jours ouvrés supplémentaires pour couvrir le délai de transmission."
                    ),
                    field="expiry_place",
                    ucp_ref="Art.6",
                    suggestion="Envisager d'ajouter 5 jours ouvrables à la date d'expiration pour les banques correspondantes distantes.",
                    confidence=0.72,
                    source="rule",
                ))

        return warnings


# ══════════════════════════════════════════════════════════════════════
# AXE 4 — COHÉRENCE GÉOGRAPHIQUE AVANCÉE (v4)
# ══════════════════════════════════════════════════════════════════════

class GeoCoherenceValidator:
    """
    Validations géographiques complémentaires au screening sanctions :
      - Port de chargement incohérent avec pays du vendeur (applicant)
      - Incoterm maritime (FOB/CFR/CIF/FAS) avec port intérieur non maritime
      - Port de chargement = port de déchargement (déjà couvert mais renforcé)
      - Expiry place incohérent avec banque désignée :41A:
    Technique : liste de ports maritimes + détection pays applicant/bénéficiaire.
    """

    # Villes intérieures (non-maritimes) fréquemment saisies par erreur
    _INLAND_PORTS: set[str] = {
        "PARIS", "BRUXELLES", "BRUSSELS", "MADRID", "BERLIN", "VIENNA", "VIENNE",
        "PRAGUE", "WARSAW", "VARSOVIE", "BUDAPEST", "BUCHAREST", "BUCURESTI",
        "BERNE", "BERN", "LUXEMBOURG", "BRATISLAVA", "ZAGREB", "BELGRADE", "BEOGRAD",
        "MOSCOW", "MOSCOU", "KYIV", "KIEV", "MINSK", "MILAN", "MILANO", "LYON",
        "ZURICH", "ZÜRICH", "FRANKFURT", "STUTTGART", "MUNICH", "MÜNCHEN",
        "ROME", "ROMA", "FLORENCE", "FIRENZE", "TORINO", "BIRMINGHAM",
        "RIYADH", "RIYAD", "NAIROBI", "KIGALI", "ADDIS ABABA", "ADDIS ABEBA",
        "CAIRO", "LE CAIRE", "AMMAN", "BAGDAD", "BAGHDAD", "TEHRAN",
        "BEIJING", "PEKIN", "CHENGDU", "WUHAN", "XIAN", "CHONGQING",
        "DELHI", "NEW DELHI", "HYDERABAD", "BANGALORE", "BENGALURU",
        "TUNIS", "ALGER", "RABAT", "ACCRA",
    }

    # Incoterms strictement maritimes (ne peuvent pas utiliser un port intérieur)
    _MARITIME_ONLY_INCOTERMS: set[str] = {"FOB", "CFR", "CIF", "FAS"}

    # Correspondances pays applicant → pays/région attendue pour port loading
    # (heuristique : applicant en Chine → port loading en Chine)
    _COUNTRY_PORT_HINTS: dict[str, list[str]] = {
        "CHINA":    ["SHANGHAI", "SHENZHEN", "GUANGZHOU", "NINGBO", "TIANJIN", "QINGDAO", "XIAMEN", "CHINA"],
        "FRANCE":   ["LE HAVRE", "MARSEILLE", "BORDEAUX", "DUNKERQUE", "FRANCE"],
        "GERMANY":  ["HAMBURG", "BREMEN", "GERMANY", "DEUTSCHLAND"],
        "ITALY":    ["GENOA", "GENOVA", "NAPLES", "NAPOLI", "LA SPEZIA", "TARANTO", "ITALY", "ITALIA"],
        "SPAIN":    ["BARCELONA", "VALENCIA", "ALGECIRAS", "BILBAO", "SPAIN", "ESPAGNE"],
        "NETHERLANDS": ["ROTTERDAM", "AMSTERDAM", "NETHERLANDS", "HOLLAND"],
        "BELGIUM":  ["ANTWERP", "ANVERS", "GHENT", "GENT", "BELGIUM", "BELGIQUE"],
        "UK":       ["FELIXSTOWE", "SOUTHAMPTON", "LONDON", "LIVERPOOL", "GLASGOW", "UK", "BRITAIN"],
        "TURKEY":   ["ISTANBUL", "IZMIR", "MERSIN", "TURKEY", "TURQUIE"],
        "INDIA":    ["MUMBAI", "NHAVA SHEVA", "CHENNAI", "MADRAS", "KOLKATA", "CALCUTTA", "INDIA", "INDE"],
        "JAPAN":    ["TOKYO", "YOKOHAMA", "OSAKA", "KOBE", "JAPAN", "JAPON"],
        "KOREA":    ["BUSAN", "PUSAN", "INCHEON", "KOREA", "CORÉE"],
        "UAE":      ["DUBAI", "JEBEL ALI", "ABU DHABI", "SHARJAH", "UAE", "EMIRATES"],
        "MALAYSIA": ["PORT KLANG", "PENANG", "TANJUNG", "MALAYSIA"],
        "THAILAND": ["LAEM CHABANG", "BANGKOK", "THAILAND", "THAÏLANDE"],
        "VIETNAM":  ["HO CHI MINH", "HAIPHONG", "HANOI", "VIETNAM"],
        "USA":      ["NEW YORK", "LOS ANGELES", "LONG BEACH", "SAVANNAH", "HOUSTON", "SEATTLE", "USA"],
        "BRAZIL":   ["SANTOS", "RIO", "BRAZIL", "BRASIL"],
        "MOROCCO":  ["CASABLANCA", "TANGER", "TANGIER", "MAROC", "MOROCCO"],
        "TUNISIA":  ["TUNIS", "SFAX", "SOUSSE", "TUNISIE", "TUNISIA", "BIZERTE", "RADES"],
        "EGYPT":    ["PORT SAID", "ALEXANDRIA", "ALEXANDRIE", "EGYPT", "ÉGYPTE"],
    }

    @classmethod
    def _detect_country_in_text(cls, text: str) -> Optional[str]:
        """Heuristique : détecte le pays dans une adresse ou un nom."""
        if not text:
            return None
        text_up = text.upper()
        for country in cls._COUNTRY_PORT_HINTS:
            if country in text_up:
                return country
        return None

    @classmethod
    def validate(
        cls,
        entities: Any,
        incoterm: Optional[str],
        raw_text: str,
    ) -> list[SemanticWarning]:
        warnings: list[SemanticWarning] = []
        inc = (incoterm or "").upper().strip()[:3]

        port_loading   = (_get_field(entities, "port_loading")   or "").strip()
        port_discharge = (_get_field(entities, "port_discharge")  or "").strip()
        applicant      = (_get_field(entities, "applicant")      or "").upper()
        expiry_place   = (_get_field(entities, "expiry_place")   or "").upper()
        available_with = (_get_field(entities, "available_with") or "").upper()

        port_loading_up   = port_loading.upper()
        port_discharge_up = port_discharge.upper()

        # ── Règle 1 : Incoterm maritime + port intérieur ─────────────────
        if inc in cls._MARITIME_ONLY_INCOTERMS:
            for port_field, port_val, field_name in [
                (port_loading_up, port_loading, "port_loading"),
                (port_discharge_up, port_discharge, "port_discharge"),
            ]:
                if any(inland in port_field for inland in cls._INLAND_PORTS):
                    warnings.append(SemanticWarning(
                        code="GEO_MARITIME_INCOTERM_INLAND_PORT",
                        severity="error",
                        message=(
                            f"Incoterm {inc} est strictement maritime mais le port "
                            f"'{port_val}' semble être une ville intérieure (non portuaire). "
                            f"Contradiction logistique : Incoterms {cls._MARITIME_ONLY_INCOTERMS} "
                            f"requièrent un port maritime réel."
                        ),
                        field=field_name,
                        ucp_ref="Art.20",
                        suggestion=(
                            f"Remplacer '{port_val}' par le port maritime le plus proche, "
                            f"ou utiliser FCA/CPT/CIP pour transport multimodal."
                        ),
                        confidence=0.94,
                        source="rule",
                    ))

        # ── Règle 2 : Port loading ≠ pays applicant ──────────────────────
        if applicant and port_loading:
            applicant_country = cls._detect_country_in_text(applicant)
            if applicant_country:
                expected_ports = cls._COUNTRY_PORT_HINTS.get(applicant_country, [])
                if expected_ports:
                    match = any(kw in port_loading_up for kw in expected_ports)
                    if not match:
                        warnings.append(SemanticWarning(
                            code="GEO_LOADING_PORT_VS_APPLICANT",
                            severity="warning",
                            message=(
                                f"Port de chargement '{port_loading}' incohérent avec le pays "
                                f"de l'applicant ({applicant_country}). "
                                f"Ports attendus : {', '.join(expected_ports[:3])}."
                            ),
                            field="port_loading",
                            ucp_ref="Art.14",
                            suggestion=(
                                "Vérifier si l'applicant exporte depuis un pays tiers. "
                                "Si intentionnel, documenter la justification."
                            ),
                            confidence=0.76,
                            source="rule",
                        ))

        # ── Règle 3 : Port loading = Port discharge ───────────────────────
        if port_loading and port_discharge and port_loading_up == port_discharge_up:
            warnings.append(SemanticWarning(
                code="GEO_SAME_LOADING_DISCHARGE",
                severity="error",
                message=(
                    f"Port de chargement et de déchargement identiques : '{port_loading}'. "
                    f"Une LC documentaire ne peut financer un transport sans déplacement réel."
                ),
                field="port_loading",
                ucp_ref="Art.20",
                suggestion="Corriger :44E: (port of loading) ou :44F: (port of discharge).",
                confidence=0.98,
                source="rule",
            ))

        # ── Règle 4 : Expiry place vs banque désignée :41A: ──────────────
        if expiry_place and available_with and "ANY BANK" not in available_with:
            # Détecter le pays dans expiry_place et dans available_with
            country_expiry = cls._detect_country_in_text(expiry_place)
            country_bank   = cls._detect_country_in_text(available_with)
            if country_expiry and country_bank and country_expiry != country_bank:
                warnings.append(SemanticWarning(
                    code="GEO_EXPIRY_PLACE_VS_DESIGNATED_BANK",
                    severity="warning",
                    message=(
                        f"Expiry place '{expiry_place}' ({country_expiry}) incohérent avec "
                        f"la banque désignée ':41A:' ({country_bank}). "
                        f"L'expiration devrait être dans le pays de la banque désignée (Art.6)."
                    ),
                    field="expiry_place",
                    ucp_ref="Art.6",
                    suggestion=(
                        f"Aligner l'expiry place sur le pays de la banque désignée ({country_bank}), "
                        f"ou modifier :41A: pour désigner une banque en {country_expiry}."
                    ),
                    confidence=0.82,
                    source="rule",
                ))

        return warnings


# ══════════════════════════════════════════════════════════════════════
# DOCUMENT DISCREPANCY ANTICIPATOR (ICC 2024 / ISBP 745)
# ══════════════════════════════════════════════════════════════════════

class DocumentDiscrepancyAnticipator:
    """Detects risky clause patterns in LC documents that systematically
    generate discrepancies under UCP 600 / ISBP 745.

    AXE 6 — proactive discrepancy risk detection.
    """

    _RISKY_PATTERNS: list = [
        (
            re.compile(
                r'\bclean\s+on\s+board\b(?!\s+(bill|b/?l))',
                re.I,
            ),
            "CLEAN_ON_BOARD_AMBIGUOUS",
            "warning",
            "Clause 'Clean on Board' sans référence explicite au B/L (Art.20b UCP 600)",
            "Préciser: 'Clean on Board Bill of Lading'",
        ),
        (
            re.compile(
                r'\bbeneficiary.{0,30}(certif|declar|attest)',
                re.I,
            ),
            "BENEFICIARY_SELF_CERT",
            "warning",
            "Auto-certification bénéficiaire — risque de refus bancaire (ISBP 745 §A24)",
            "Remplacer par une certification tierce indépendante",
        ),
        (
            re.compile(
                r'\b(signed?|sign(ature)?)\s+(by|from)\b(?!\s+(issuing|notary|surveyor|inspector))',
                re.I,
            ),
            "SIGNATURE_UNSPECIFIED",
            "info",
            "Exigence de signature sans préciser la qualité du signataire (ISBP 745 §A14)",
            "Spécifier: 'signed by issuing carrier / master / agent'",
        ),
        (
            re.compile(
                r'\bany\s+(other\s+)?documents?\b',
                re.I,
            ),
            "ANY_DOCS_CLAUSE",
            "warning",
            "Clause 'Any documents' vague — génère systématiquement des discordances (Art.14 UCP)",
            "Lister exhaustivement les documents requis dans :46A:",
        ),
        (
            re.compile(
                r'\b(original(s)?\s+not\s+required|copy\s+acceptable|copies?\s+accepted?)\b',
                re.I,
            ),
            "ORIGINAL_WAIVED",
            "info",
            "Originaux dispensés — vérifier cohérence avec Art.17 UCP 600",
            "Confirmer si les copies sont acceptables selon Art.17c",
        ),
        (
            re.compile(
                r'\b(telex\s+release|surrender(ed)?|express\s+release)\b',
                re.I,
            ),
            "TELEX_RELEASE_RISK",
            "warning",
            "Telex/Express release détecté — B/L original non requis, risque de fraude documentaire",
            "Exiger un B/L original ou utiliser une Sea Waybill (Art.21 UCP)",
        ),
        (
            re.compile(
                r'\bstale\s+b/?l\b|\bstale\s+bill\b|stale\s+document',
                re.I,
            ),
            "STALE_BL_ACCEPTED",
            "warning",
            "B/L périmé accepté — contourne la règle des 21 jours (Art.14c UCP 600)",
            "Supprimer la clause stale ou préciser la fenêtre maximale",
        ),
        (
            re.compile(
                r'\b(sight\s+draft|at\s+sight)\b.{0,80}\b(without|w/?o)\s+confirm',
                re.I,
            ),
            "SIGHT_DRAFT_CONFIRM",
            "info",
            "Traite à vue sans confirmation — risque de non-paiement bénéficiaire (Art.12 UCP)",
            "Envisager confirmation par banque notificatrice (Art.8 UCP)",
        ),
        (
            re.compile(
                r'\b(discrepanc(y|ies)\s+waived?|waiver\s+of\s+discrepanc)',
                re.I,
            ),
            "DISCREPANCY_ESCAPE_CLAUSE",
            "error",
            "Clause de renonciation aux discordances — invalide sous UCP 600 (Art.16)",
            "Supprimer cette clause non-conforme UCP 600",
        ),
    ]

    def validate(self, docs_text: str, conditions_text: str) -> list:
        """Return list of SemanticWarning for risky patterns found."""
        combined = f"{docs_text or ''}\n{conditions_text or ''}"
        warnings = []
        for pattern, code, severity, message, suggestion in self._RISKY_PATTERNS:
            if pattern.search(combined):
                warnings.append(SemanticWarning(
                    code=code,
                    severity=severity,
                    message=message,
                    field="documents_required/additional_conditions",
                    ucp_ref="Art.14-17 UCP 600 / ISBP 745",
                    suggestion=suggestion,
                    confidence=0.80,
                    source="rule",
                ))
        return warnings


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════

class SwiftSemanticAnalyzer:
    """
    Analyse sémantique avancée v3.
    Entrée  : entités (dataclass ou dict) + texte brut
    Sortie  : SemanticAnalysisResult enrichi v3
    """

    def __init__(
        self,
        risk_model_path:           Optional[str] = None,
        enable_semantic_coherence: bool          = True,
        fuzzy_threshold:           int           = 82,
        nli_model_dir:             Optional[str] = None,
    ) -> None:
        self._completeness_refs:     dict = {}
        self._dataset_risk_patterns: list = []
        try:
            try:
                from services.swift_nlp.common.swift_dataset_loader import SwiftDatasetLoader
            except ImportError:
                from ..common.swift_dataset_loader import SwiftDatasetLoader
            ds = SwiftDatasetLoader.instance().get()
            if ds.loaded:
                self._completeness_refs     = ds.completeness_refs
                self._dataset_risk_patterns = ds.risk_patterns
        except Exception:
            pass

        self._coherence      = _SemanticCoherenceChecker() if enable_semantic_coherence else None
        self._ml_risk        = _MLRiskScorer(risk_model_path)
        self._fuzzy_threshold = fuzzy_threshold

        # Offline NLI (optionnel) : activé seulement si un modèle local est présent.
        self._nli = None
        try:
            if OfflineNLIPredictor is not None:
                root = Path(__file__).resolve().parents[3]
                candidate = Path(nli_model_dir) if nli_model_dir else (root / "models" / "nli")
                self._nli = OfflineNLIPredictor(candidate)
                if getattr(self._nli, "available", False):
                    logger.info("SemanticAnalyzer: NLI offline activé (%s)", str(candidate))
                else:
                    logger.info("SemanticAnalyzer: NLI offline indisponible (%s)", str(candidate))
        except Exception:
            self._nli = None

        # NEW v3 : instances des modules avancés
        self._excl_typer      = ExclusionSemanticTyper()
        self._contradiction   = ContradictionEngine()
        self._incoterm_matrix = IncotermsRiskMatrix()
        self._bayesian        = BayesianFieldScorer()
        self._ontology        = RiskOntologyGraph()

        # NEW v4 : validateurs des 5 axes sémantiques
        self._doc_context_validator  = DocumentContextValidator()
        self._amount_tol_validator   = AmountToleranceValidator()
        self._temporal_validator     = ExtendedTemporalValidator()
        self._geo_coherence_validator = GeoCoherenceValidator()

        # AXE 6 : anticipateur de discordances documentaires (ICC 2024 / ISBP 745)
        self._discrepancy_anticipator = DocumentDiscrepancyAnticipator()

        # AXE 7 : cohérence inter-champs MT700 (NLI + règles)
        self._cross_field_analyzer = (
            CrossFieldCoherenceAnalyzer(nli=self._nli)
            if CrossFieldCoherenceAnalyzer is not None else None
        )

    def analyze(self, entities: Any, raw_text: str = "") -> SemanticAnalysisResult:
        """Pipeline d'analyse complet v4 — 5 axes sémantiques."""
        from datetime import timezone
        swift_type = _get_field(entities, "swift_type") or "UNKNOWN"
        result     = SemanticAnalysisResult(swift_type=swift_type)
        result.engine_version     = "4.0"
        result.analysis_timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Score de complétude
        result.completeness_score = self._compute_completeness(entities, swift_type)

        # 2. Scoring Bayésien (confidence globale)
        ent_dict = _entities_to_dict(entities) if not isinstance(entities, dict) else entities
        confidence, _field_scores = self._bayesian.compute(ent_dict)
        result.bayesian_confidence = confidence

        # 3. Règles UCP 600 + cohérence temporelle
        self._apply_ucp_rules(entities, result)

        # 4. Cohérence inter-champs (règles)
        self._check_cross_field_consistency(entities, result)

        # 5. Détection de contradictions tag/clause (UCP Art.5) — NEW v3
        clauses_text = raw_text
        result.contradictions = self._contradiction.detect(ent_dict, clauses_text, nli=self._nli)

        # 6. Cohérence sémantique (embeddings, si disponible)
        if self._coherence and self._coherence.available:
            self._check_semantic_coherence(entities, result)

        # 7. Risques (règles + keywords)
        self._detect_risks(entities, raw_text, result)

        # 8. Fraude documentaire
        result.fraud_indicators = _RiskDetector.check_fraud_patterns(raw_text, entities)

        # 9. Analyse Incoterm — NEW v3
        incoterm = _get_field(entities, "incoterm") or self._infer_incoterm(entities)
        result.incoterm_risk = self._incoterm_matrix.analyze(incoterm, swift_type)

        # Alertes Incoterm → warnings
        for alert in result.incoterm_risk.get("alerts", []):
            result.warnings.append(SemanticWarning(
                code="INCOTERM_RISK", severity="warning",
                message=alert, field="incoterm", ucp_ref="Art.28",
                source="incoterm_matrix", confidence=0.90,
            ))

        # ── AXE 1 : Cohérence :46A: documents vs contexte commercial ──────
        if swift_type in ("MT700", "MT720"):
            docs_text  = _get_field(entities, "documents_required") or ""
            goods_text = _get_field(entities, "goods_description")  or ""
            if not docs_text:
                # Extraire depuis raw_text si pas dans entities
                m = re.search(r':46A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
                if m:
                    docs_text = m.group(1)
            ax1_warnings = DocumentContextValidator.validate(entities, incoterm, docs_text, goods_text)
            result.warnings.extend(ax1_warnings)

        # ── AXE 2 : Cohérence montant / tolérance / quantité ─────────────
        if swift_type in ("MT700", "MT720"):
            ax2_warnings = AmountToleranceValidator.validate(
                entities,
                _get_field(entities, "goods_description") or "",
                raw_text,
            )
            result.warnings.extend(ax2_warnings)

        # ── AXE 3 : Cohérence temporelle étendue ──────────────────────────
        if swift_type in ("MT700", "MT720"):
            ax3_warnings = ExtendedTemporalValidator.validate(
                entities,
                _get_field(entities, "goods_description") or "",
                raw_text,
            )
            result.warnings.extend(ax3_warnings)

        # ── AXE 4 : Cohérence géographique avancée ────────────────────────
        ax4_warnings = GeoCoherenceValidator.validate(entities, incoterm, raw_text)
        result.warnings.extend(ax4_warnings)

        # 10. Scoring ML (si modèle disponible)
        if self._ml_risk._available:
            ml_scores = self._ml_risk.predict(raw_text, ent_dict)
            if ml_scores:
                self._apply_ml_risk_scores(ml_scores, result)

        # ── AXE 6 : Anticipateur de discordances documentaires (ICC 2024) ───
        docs_text = _get_field(entities, "documents_required") or ""
        cond_text = _get_field(entities, "additional_conditions") or ""
        ax6_warnings = self._discrepancy_anticipator.validate(docs_text, cond_text)
        result.warnings.extend(ax6_warnings)

        # 11. Score de qualité documentaire contextuel (Incoterms 2020 + goods)
        result.doc_quality_score = self._compute_doc_quality_contextual(
            entities, incoterm, raw_text
        )

        # 12. Score sémantique global (4-component positive scoring)
        result.semantic_score = self._compute_semantic_score(
            result, doc_quality=result.doc_quality_score
        )

        # 13. Cohérence géographique (legacy enrichment dict)
        result.geo_coherence = self._check_geo_coherence(entities, raw_text)

        # 14. Propagation ontologique — NEW v3
        violated = self._collect_violated_articles(result)
        existing_codes = {f.code for f in result.risk_flags}
        result.ontology_propagations = self._ontology.propagate(violated, existing_codes)

        # ── AXE 7 : Cohérence inter-champs MT700 (NLI + règles) ──────────
        if self._cross_field_analyzer is not None and swift_type in ("MT700", "MT720"):
            try:
                cf_result = self._cross_field_analyzer.analyze(ent_dict, raw_text)
                result.cross_field_coherence = {
                    "coherence_score": cf_result.coherence_score,
                    "verdict":         cf_result.verdict,
                    "axes_checked":    cf_result.axes_checked,
                    "processing_ms":   cf_result.processing_ms,
                    "issues": [
                        {
                            "fields":     issue.fields,
                            "code":       issue.code,
                            "technique":  issue.technique,
                            "severity":   issue.severity,
                            "message":    issue.message,
                            "suggestion": issue.suggestion,
                            "lc_value":   issue.lc_value,
                            "doc_value":  issue.doc_value,
                            "confidence": round(issue.confidence, 3),
                            "ucp_ref":    issue.ucp_ref,
                        }
                        for issue in cf_result.issues
                    ],
                }
                logger.info(
                    "AXE7 cross-field: %d issues — score=%.1f verdict=%s",
                    len(cf_result.issues), cf_result.coherence_score, cf_result.verdict,
                )
            except Exception as _cf_exc:
                logger.warning("AXE7 cross-field coherence failed (skipped): %s", _cf_exc, exc_info=True)
                result.cross_field_coherence = {
                    "coherence_score": 100.0, "verdict": "SKIPPED",
                    "axes_checked": [], "processing_ms": 0, "issues": [],
                    "_error": str(_cf_exc),
                }

        # ── AXE 5 : Score composite + versionnage moteur ──────────────────
        max_risk = max(
            (1.0 if r.severity == "high" else 0.5 if r.severity == "medium" else 0.2)
            for r in result.risk_flags
        ) if result.risk_flags else 0.0

        # cross-field score (from AXE 7)
        cf_score = (
            result.cross_field_coherence.get("coherence_score", 100.0)
            if result.cross_field_coherence else 100.0
        )
        cf_has_blocking_error = any(
            i.get("severity") == "error" and i.get("confidence", 0) >= 0.82
            for i in result.cross_field_coherence.get("issues", [])
        ) if result.cross_field_coherence else False

        result.final_score = round(
            0.28 * (result.completeness_score / 100.0)
            + 0.22 * (result.semantic_score    / 100.0)
            + 0.20 * (cf_score / 100.0)
            + 0.20 * (1.0 - max_risk)
            + 0.10 * result.bayesian_confidence,
            4,
        )
        if result.final_score >= 0.82:
            result.routing_decision = "ACCEPT"
        elif result.final_score >= 0.58:
            result.routing_decision = "REVIEW"
        else:
            result.routing_decision = "REJECT"

        # Override: any high-confidence cross-field error forces at least REVIEW
        if cf_has_blocking_error and result.routing_decision == "ACCEPT":
            result.routing_decision = "REVIEW"

        # 15. Recommandations enrichies
        result.recommendations = self._generate_recommendations(result, entities, swift_type)

        # 16. Résumé narratif
        result.summary = self._build_summary(entities, result, swift_type)

        # 17. Champs enrichis par inférence
        result.enriched_fields = self._infer_missing_fields(entities, swift_type)

        return result

    # ── Complétude ────────────────────────────────────────────────────

    def _compute_completeness(self, entities: Any, swift_type: str) -> int:
        required = _REQUIRED_FIELDS.get(swift_type, [])
        opt      = _OPTIONAL_IMPORTANT.get(swift_type, [])

        if not required:
            return int(self._completeness_refs.get(swift_type, 75))

        req_score = sum(1 for f in required if _get_field(entities, f)) / len(required)
        opt_score = (sum(1 for f in opt if _get_field(entities, f)) / len(opt)) if opt else 1.0
        raw = req_score * 0.80 + opt_score * 0.20
        score = round(raw * 100)
        ref = self._completeness_refs.get(swift_type)
        if ref and score > 0:
            score = round(score * 0.80 + ref * 0.20)
        return max(30, min(100, score))

    # ── Règles UCP 600 ────────────────────────────────────────────────

    def _apply_ucp_rules(self, entities: Any, result: SemanticAnalysisResult) -> None:
        swift_type = result.swift_type

        for check, args in [
            (_UCPRules.check_issue_vs_expiry,
             (_get_field(entities, "issue_date"), _get_field(entities, "expiry_date"))),
            (_UCPRules.check_expiry_vs_shipment,
             (_get_field(entities, "expiry_date"), _get_field(entities, "shipment_date"))),
            (_UCPRules.check_applicable_rules,
             (_get_field(entities, "applicable_rules"),)),
        ]:
            w = check(*args)
            if w:
                result.warnings.append(w)

        # Cohérence temporelle globale — NEW v3
        result.warnings.extend(_UCPRules.check_date_temporal_consistency(
            _get_field(entities, "issue_date"),
            _get_field(entities, "shipment_date"),
            _get_field(entities, "expiry_date"),
        ))

        if swift_type in ("MT700", "MT720"):
            pp = _get_field(entities, "presentation_period")
            days = None
            if pp:
                m = re.search(r'(\d+)', str(pp))
                if m:
                    days = int(m.group(1))
            w = _UCPRules.check_presentation_period(days)
            if w:
                result.warnings.append(w)
            w = _UCPRules.check_tolerance(
                _get_field(entities, "tolerance_plus"),
                _get_field(entities, "tolerance_minus"),
            )
            if w:
                result.warnings.append(w)

    # ── Cohérence inter-champs ─────────────────────────────────────

    def _check_cross_field_consistency(self, entities: Any, result: SemanticAnalysisResult) -> None:
        applicant   = _get_field(entities, "applicant")
        beneficiary = _get_field(entities, "beneficiary")

        if applicant and beneficiary:
            if applicant.upper().strip()[:20] == beneficiary.upper().strip()[:20]:
                result.warnings.append(SemanticWarning(
                    code="SAME_APPLICANT_BENEFICIARY", severity="error",
                    message="Applicant et Bénéficiaire identiques",
                    field="applicant/beneficiary",
                    suggestion="Vérifier l'identité des deux parties",
                ))

        amount = _get_field(entities, "amount")
        if amount:
            try:
                val = float(re.sub(r'[^\d.]', '', str(amount)))
                if val <= 0:
                    result.warnings.append(SemanticWarning(
                        code="INVALID_AMOUNT", severity="error",
                        message=f"Montant invalide: {amount}", field="amount",
                    ))
                elif val > 500_000_000:
                    result.warnings.append(SemanticWarning(
                        code="VERY_LARGE_AMOUNT", severity="warning",
                        message=f"Montant exceptionnellement élevé: {amount}",
                        field="amount", suggestion="Déclencher due diligence renforcée",
                    ))
            except (ValueError, TypeError):
                pass

        loading   = _get_field(entities, "port_loading")
        discharge = _get_field(entities, "port_discharge")
        if loading and discharge and loading.upper() == discharge.upper():
            result.warnings.append(SemanticWarning(
                code="SAME_PORT", severity="warning",
                message=f"Port chargement = déchargement: {loading}",
                field="port_loading/port_discharge",
            ))

        if result.swift_type == "MT720" and not _get_field(entities, "related_lc"):
            result.warnings.append(SemanticWarning(
                code="MT720_MISSING_RELATED", severity="error",
                message="MT720 : champ :21: manquant",
                field="related_lc", ucp_ref="UCP 600 Art.38",
            ))

    # ── Cohérence sémantique (embeddings) ─────────────────────────

    def _check_semantic_coherence(self, entities: Any, result: SemanticAnalysisResult) -> None:
        coherence_scores = {}
        applicant   = _get_field(entities, "applicant")
        beneficiary = _get_field(entities, "beneficiary")
        if applicant and beneficiary and self._coherence:
            sim, w = self._coherence.check_parties_similarity(applicant, beneficiary)
            coherence_scores["applicant_vs_beneficiary"] = round(sim, 3)
            if w:
                result.warnings.append(w)

        goods = _get_field(entities, "goods_description")
        docs  = _get_field(entities, "documents_required")
        if goods and docs and self._coherence:
            sim, w = self._coherence.check_goods_vs_docs(goods, docs)
            coherence_scores["goods_vs_docs"] = round(sim, 3)
            if w:
                result.warnings.append(w)
        result.coherence_scores = coherence_scores

    # ── Détection de risques ──────────────────────────────────────

    def _detect_risks(self, entities: Any, raw_text: str, result: SemanticAnalysisResult) -> None:
        search_text = raw_text or ""
        for fname in ["applicant", "beneficiary", "port_loading", "port_discharge",
                      "expiry_place", "goods_description"]:
            val = _get_field(entities, fname)
            if val:
                search_text += " " + str(val)

        result.risk_flags.extend(_RiskDetector.check_sanctions(search_text))
        result.risk_flags.extend(_RiskDetector.check_sensitive_goods(
            _get_field(entities, "goods_description")
        ))
        result.risk_flags.extend(_RiskDetector.check_aml_patterns(search_text))

        flag = _RiskDetector.check_round_amount(_get_field(entities, "amount"))
        if flag:
            result.risk_flags.append(flag)

        flag = _RiskDetector.check_very_short_validity(
            _get_field(entities, "issue_date"),
            _get_field(entities, "expiry_date"),
        )
        if flag:
            result.risk_flags.append(flag)

        # ── Montant suspicieusement bas ──────────────────────────
        try:
            _amt_raw = _get_field(entities, "amount")
            _cur     = (_get_field(entities, "currency") or "").upper().strip()
            if _amt_raw and _cur in ("EUR", "USD", "GBP", "CHF"):
                _amt_val = float(str(_amt_raw).replace(",", "."))
                if 0 < _amt_val < 500:
                    result.risk_flags.append(RiskFlag(
                        code="SUSPICIOUSLY_LOW_AMOUNT",
                        description=f"Montant {_cur} {_amt_val:.2f} anormalement bas pour une LC commerciale",
                        severity="medium",
                        ucp_ref="UCP 600 Art.6",
                        category="compliance",
                        recommendation="Vérifier si LC de test ou montant partiel — EUR/USD < 500 inhabituel",
                    ))
        except Exception:
            pass

        # ── Artefact OCR dans :45A: ──────────────────────────────
        _goods = _get_field(entities, "goods_description") or ""
        if re.search(r'\b(e{3,}|x{3,}|a{4,}|errr+|xxxx|aaaa|test123)\b', _goods, re.I):
            result.warnings.append(SemanticWarning(
                code="GOODS_OCR_ARTIFACT",
                severity="error",
                message=f"Artefact OCR détecté dans :45A: description marchandises : {_goods[:80]}",
                field="goods_description",
                ucp_ref="ISBP 745 §A1",
                suggestion="Corriger la description des marchandises — texte corrompu par OCR",
                source="rule",
            ))

        # ── :46A: fragment incomplet (pas de liste documentaire réelle) ─
        _docs = _get_field(entities, "documents_required") or ""
        _docs_clean = _docs.strip().lstrip("+").strip()
        _has_real_doc = re.search(
            r'\b(invoice|bill of lading|certificate|packing|insurance|B/L|B\.?L|'
            r'facture|connaissement|assurance|attestation|inspection|origine)\b',
            _docs_clean, re.I,
        )
        if _docs_clean and not _has_real_doc and len(_docs_clean) < 60:
            result.warnings.append(SemanticWarning(
                code="INCOMPLETE_DOCS_REQUIRED",
                severity="error",
                message=f":46A: ne contient pas de liste documentaire complète — fragment détecté : '{_docs_clean[:60]}'",
                field="documents_required",
                ucp_ref="UCP 600 Art.14 / Art.18",
                suggestion="Compléter :46A: avec la liste des documents requis (facture, B/L, certificats...)",
                source="rule",
            ))

        # ── Incoterms obsolètes (2000) ───────────────────────────
        _OBSOLETE_INCOTERMS = {"DES", "DEQ", "DAF", "DDU"}
        if _goods:
            _found_obs = re.findall(r'\b(DES|DEQ|DAF|DDU)\b', _goods, re.I)
            for _obs in set(t.upper() for t in _found_obs):
                if _obs in _OBSOLETE_INCOTERMS:
                    result.warnings.append(SemanticWarning(
                        code="OBSOLETE_INCOTERM",
                        severity="warning",
                        message=f"Incoterm '{_obs}' obsolète (supprimé dans Incoterms 2010/2020) dans :45A:",
                        field="goods_description",
                        ucp_ref="ICC Incoterms 2020",
                        suggestion=f"Remplacer '{_obs}' par DAP, DPU ou DDP (Incoterms 2020)",
                        source="rule",
                    ))

    # ── Scoring ML ────────────────────────────────────────────────

    def _apply_ml_risk_scores(self, ml_scores: dict[str, float], result: SemanticAnalysisResult) -> None:
        for flag in result.risk_flags:
            cat_upper = flag.category.upper()
            if cat_upper in ml_scores:
                flag.ml_score = ml_scores[cat_upper]

        _SKIP = {"NORMAL", "COMPLIANCE", "DOCUMENT_QUALITY"}
        for risk_type, prob in ml_scores.items():
            if prob >= 0.70 and risk_type not in _SKIP:
                if not any(r.category.upper() == risk_type for r in result.risk_flags):
                    result.risk_flags.append(RiskFlag(
                        code=f"ML_{risk_type}",
                        description=f"Risque {risk_type} ML (p={prob:.2f})",
                        severity="high" if prob >= 0.85 else "medium",
                        ucp_ref="", ml_score=prob,
                        category=risk_type.lower(),
                        recommendation="Vérification manuelle recommandée",
                    ))

    # ── Qualité documentaire v3 (fuzzy) ───────────────────────────

    def _compute_doc_quality_fuzzy(self, entities: Any, raw_text: str) -> int:
        """
        BUG #3 FIX + fuzzy matching — score non-nul pour "N originals",
        et matching souple des types de documents (alias + fuzzy).
        """
        docs = _get_field(entities, "documents_required") or ""
        if not docs and raw_text:
            tag_m = re.search(r':46A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
            if tag_m:
                docs = tag_m.group(1)
            else:
                tag_m2 = re.search(r':47A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
                if tag_m2:
                    docs = tag_m2.group(1)
        if not docs:
            return 0

        score = 0
        total_weight = sum(_DOC_QUALITY_WEIGHTS.values())

        for doc_type, weight in _DOC_QUALITY_WEIGHTS.items():
            # Exact match
            if doc_type.upper() in docs.upper():
                score += weight
                continue
            # Alias match
            aliases = _DOC_ALIASES.get(doc_type, [])
            matched = False
            for alias in aliases:
                if alias.upper() in docs.upper():
                    score += weight
                    matched = True
                    break
            if matched:
                continue
            # Fuzzy match (partial credit)
            if _FUZZY_AVAILABLE:
                for alias in [doc_type] + aliases:
                    if _fuzzy_contains(docs, alias, threshold=self._fuzzy_threshold):
                        score += weight * 0.7   # 70% du poids si match flou
                        break

        # BUG #3 FIX : "N originals" présent mais types non listés
        if score == 0:
            orig_m = re.search(r'\b(\d+)\s+originals?\b', docs, re.I)
            if orig_m:
                n = int(orig_m.group(1))
                score = min(30, n * 10)
            elif re.search(r'\b(?:original|copy|copies|document[s]?|certif)\b', docs, re.I):
                score = 15

        return min(100, round(score / total_weight * 100))

    # ── Qualité documentaire contextuelle (Incoterms + goods) ────

    _MARITIME_INCOTERMS = {"FOB", "CFR", "CIF", "FAS", "DAT"}
    _INSURANCE_INCOTERMS = {"CIF", "CIP"}
    _ANY_TRANSPORT_INCOTERMS = {"CPT", "CIP", "FCA", "DAP", "DPU", "DDP", "EXW"}

    def _compute_doc_quality_contextual(
        self, entities: Any, incoterm: str, raw_text: str
    ) -> int:
        """Context-aware doc quality scoring (Incoterms 2020 + goods category).

        Replaces fixed-weight _DOC_QUALITY_WEIGHTS which penalise valid LCs
        (e.g. air-freight LC: AWB + Invoice = 43/100 despite full coverage).
        """
        docs = _get_field(entities, "documents_required") or ""
        if not docs and raw_text:
            m = re.search(r':46A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
            if m:
                docs = m.group(1)
        if not docs:
            return 0

        docs_up = docs.upper()
        inco_up = (incoterm or "").strip().upper()[:3]

        goods = _get_field(entities, "goods_description") or ""

        # ── Required docs (mandatory for this context) ────────────
        required: list[tuple[str, list[str]]] = [
            ("COMMERCIAL INVOICE", ["INVOICE", "FACTURE"]),
        ]

        # Transport doc depends on incoterm / transport mode
        is_maritime = inco_up in self._MARITIME_INCOTERMS
        is_air = any(k in docs_up for k in ["AIRWAY BILL", "AIR WAYBILL", "AWB", "LTA"])
        is_multimodal = any(k in docs_up for k in ["MULTIMODAL", "COMBINED TRANSPORT"])

        if is_maritime and not is_air:
            required.append(("BILL OF LADING", ["B/L", "BL ", "B.L.", "CONNAISSEMENT"]))
        elif is_air:
            required.append(("AIRWAY BILL", ["AWB", "AIR WAYBILL", "LTA"]))
        elif is_multimodal:
            required.append(("TRANSPORT DOCUMENT", ["MULTIMODAL", "COMBINED TRANSPORT"]))
        else:
            # Any transport doc counts
            any_transport = any(k in docs_up for k in [
                "BILL OF LADING", "B/L", "AIRWAY BILL", "AWB",
                "MULTIMODAL", "CMR", "CIM", "ROAD", "RAIL",
            ])
            if any_transport:
                required.append(("TRANSPORT DOCUMENT", []))

        # Insurance: only mandatory for CIF / CIP (Art.28 UCP 600)
        if inco_up in self._INSURANCE_INCOTERMS:
            required.append(("INSURANCE", ["ASSURANCE", "POLICE D'ASSURANCE"]))

        # ── Recommended docs (goods-specific) ────────────────────
        recommended: list[tuple[str, list[str]]] = [
            ("PACKING LIST", ["LISTE DE COLISAGE", "PACKING"]),
            ("CERTIFICATE OF ORIGIN", ["CERTIFICAT D'ORIGINE", "CO "]),
        ]

        # Goods-category specific
        goods_up = goods.upper()
        if any(k in goods_up for k in ["FOOD", "FISH", "MEAT", "DAIRY", "ALIMENTAIRE", "POISSON", "VIANDE"]):
            recommended.append(("SANITARY CERTIFICATE", ["HEALTH CERT", "PHYTOSANITARY", "VETERINARY"]))
        if any(k in goods_up for k in ["CHEMICAL", "CHIMIQUE", "ACID", "HAZARDOUS", "DANGEREUX"]):
            recommended.append(("MSDS", ["SDS", "DANGEROUS GOODS", "MATERIAL SAFETY"]))
        if any(k in goods_up for k in ["PHARMACEUTICAL", "PHARMACEUTIQUE", "MEDICINE", "MEDICAMENT"]):
            recommended.append(("GMP CERTIFICATE", ["FDA", "ANALYSIS CERTIFICATE"]))
        if any(k in goods_up for k in ["STEEL", "ACIER", "IRON", "METAL", "ALUMINIUM", "PIPE", "TUYAU"]):
            recommended.append(("MILL CERTIFICATE", ["TEST CERTIFICATE", "MATERIAL CERT"]))

        def _doc_present(main: str, aliases: list[str]) -> bool:
            if main in docs_up:
                return True
            return any(a in docs_up for a in aliases)

        req_ok  = sum(1 for main, aliases in required    if _doc_present(main, aliases))
        rec_ok  = sum(1 for main, aliases in recommended if _doc_present(main, aliases))

        req_total = len(required)
        rec_total = len(recommended)

        req_score = (req_ok / req_total * 80) if req_total else 80
        rec_score = (rec_ok / rec_total * 20) if rec_total else 20

        return min(100, round(req_score + rec_score))

    # ── Cohérence géographique ────────────────────────────────────

    # ISO-2 code lookup: country name → code
    _COUNTRY_NAME_TO_ISO2: dict[str, str] = {
        "FRANCE": "FR", "FRENCH": "FR",
        "TUNISIA": "TN", "TUNISIE": "TN", "TUNISIAN": "TN",
        "MOROCCO": "MA", "MAROC": "MA", "MOROCCAN": "MA",
        "ALGERIA": "DZ", "ALGERIE": "DZ", "ALGERIAN": "DZ",
        "EGYPT": "EG", "EGYPTE": "EG", "EGYPTIAN": "EG",
        "GERMANY": "DE", "ALLEMAGNE": "DE", "GERMAN": "DE",
        "UNITED KINGDOM": "GB", "UK": "GB", "GREAT BRITAIN": "GB",
        "UNITED STATES": "US", "USA": "US", "AMERICA": "US",
        "CHINA": "CN", "CHINE": "CN", "CHINESE": "CN",
        "TURKEY": "TR", "TURQUIE": "TR", "TURKISH": "TR",
        "ITALY": "IT", "ITALIE": "IT", "ITALIAN": "IT",
        "SPAIN": "ES", "ESPAGNE": "ES", "SPANISH": "ES",
        "NETHERLANDS": "NL", "PAYS-BAS": "NL", "HOLLAND": "NL",
        "BELGIUM": "BE", "BELGIQUE": "BE",
        "SWITZERLAND": "CH", "SUISSE": "CH",
        "UNITED ARAB EMIRATES": "AE", "UAE": "AE",
        "SAUDI ARABIA": "SA", "ARABIE SAOUDITE": "SA",
        "INDIA": "IN", "INDE": "IN", "INDIAN": "IN",
        "JAPAN": "JP", "JAPON": "JP", "JAPANESE": "JP",
        "SOUTH KOREA": "KR", "KOREA": "KR",
        "SINGAPORE": "SG", "SINGAPOUR": "SG",
        "HONG KONG": "HK",
        "BRAZIL": "BR", "BRESIL": "BR",
        "SENEGAL": "SN", "SENEGAL": "SN",
        "IVORY COAST": "CI", "COTE D'IVOIRE": "CI",
        "NIGERIA": "NG", "GHANA": "GH", "KENYA": "KE",
        "PAKISTAN": "PK", "BANGLADESH": "BD", "SRI LANKA": "LK",
        "IRAN": "IR", "IRAQ": "IQ", "SYRIA": "SY",
        "LIBYA": "LY", "LIBYE": "LY",
    }

    # SWIFT /xyz suffix → ISO-2
    _SWIFT_SUFFIX_TO_ISO2: dict[str, str] = {
        "/fra": "FR", "/tun": "TN", "/mar": "MA", "/dza": "DZ",
        "/deu": "DE", "/gbr": "GB", "/usa": "US", "/chn": "CN",
        "/tur": "TR", "/ita": "IT", "/esp": "ES", "/nld": "NL",
        "/bel": "BE", "/che": "CH", "/are": "AE", "/sau": "SA",
        "/ind": "IN", "/jpn": "JP", "/kor": "KR", "/sgp": "SG",
        "/hkg": "HK", "/bra": "BR", "/sen": "SN", "/nga": "NG",
        "/pak": "PK", "/bgd": "BD",
    }

    def _resolve_iso2(self, text: str) -> Optional[str]:
        """Resolve a free-text place/country to an ISO-2 country code."""
        if not text:
            return None
        t = text.strip().upper()
        # Direct 2-letter ISO code
        if len(t) == 2 and t.isalpha():
            return t
        # 3-letter SWIFT country codes (e.g. TUN, FRA, DEU)
        _SWIFT3 = {"TUN": "TN", "FRA": "FR", "MAR": "MA", "DZA": "DZ",
                   "DEU": "DE", "GBR": "GB", "USA": "US", "CHN": "CN",
                   "TUR": "TR", "ITA": "IT", "ESP": "ES", "NLD": "NL",
                   "BEL": "BE", "CHE": "CH", "ARE": "AE", "SAU": "SA",
                   "IND": "IN", "JPN": "JP", "KOR": "KR", "SGP": "SG"}
        if len(t) == 3 and t.isalpha() and t in _SWIFT3:
            return _SWIFT3[t]
        # Known country names
        return self._COUNTRY_NAME_TO_ISO2.get(t)

    def _beneficiary_country_iso2(self, beneficiary: str, lc_country: str) -> Optional[str]:
        """Extract ISO-2 country code from beneficiary field or LC country."""
        if not beneficiary:
            return self._resolve_iso2(lc_country)
        bene_lower = beneficiary.lower().strip()
        # SWIFT /xyz suffix on beneficiary line
        for suffix, code in self._SWIFT_SUFFIX_TO_ISO2.items():
            if bene_lower.endswith(suffix) or f" {suffix[1:]}" in bene_lower:
                return code
        # LC country field as fallback
        return self._resolve_iso2(lc_country)

    def _check_geo_coherence(self, entities: Any, raw_text: str) -> dict:
        geo: dict = {}
        port_loading   = _get_field(entities, "port_loading")
        port_discharge = _get_field(entities, "port_discharge")
        expiry_place   = _get_field(entities, "expiry_place")
        beneficiary    = _get_field(entities, "beneficiary")
        lc_country     = _get_field(entities, "country")

        for k, v in [("port_loading", port_loading), ("port_discharge", port_discharge),
                     ("expiry_place", expiry_place)]:
            if v:
                geo[k] = v

        # ── Résoudre les codes ISO des parties ───────────────────
        expiry_iso    = self._resolve_iso2(expiry_place) if expiry_place else None
        bene_iso      = self._beneficiary_country_iso2(beneficiary, lc_country)
        discharge_iso = self._resolve_iso2(port_discharge) if port_discharge else None
        loading_iso   = self._resolve_iso2(port_loading)   if port_loading   else None

        if expiry_iso:
            geo["expiry_country_code"] = expiry_iso
        if bene_iso:
            geo["beneficiary_country_code"] = bene_iso

        # ── A : expiry place vs pays bénéficiaire ────────────────
        if expiry_iso and bene_iso:
            geo["expiry_beneficiary_match"] = (expiry_iso == bene_iso)
        elif expiry_place and beneficiary:
            # Fallback: name-in-string pour anciens formats
            geo["expiry_beneficiary_match"] = any(
                word.lower() in expiry_place.lower()
                for word in (beneficiary or "").split()
                if len(word) > 4
            )

        # ── B : port de déchargement vs pays bénéficiaire ────────
        if discharge_iso and bene_iso:
            geo["port_discharge_match"] = (discharge_iso == bene_iso)
            if discharge_iso != bene_iso:
                geo["port_discharge_country_code"] = discharge_iso

        # ── C : port de chargement (informatif) ──────────────────
        if loading_iso:
            geo["port_loading_country_code"] = loading_iso

        return geo

    # ── Score sémantique global (4-component positive) ───────────

    @staticmethod
    def _compute_semantic_score(
        result: SemanticAnalysisResult,
        doc_quality: int = 0,
    ) -> int:
        """4-component additive scoring (max 100 pts).

        comp1 (40 pts) — UCP compliance: errors, contradictions, fraud
        comp2 (30 pts) — field coherence: warnings
        comp3 (20 pts) — documentary coverage (contextual quality)
        comp4 (10 pts) — temporal coherence (risk_flags tagged temporal)

        bayesian_confidence is no longer used as a multiplier to avoid
        crushing valid LCs that have atypical field formats.
        """
        # comp1: UCP compliance (40 pts)
        comp1 = 40
        for w in result.warnings:
            if w.severity == "error":
                comp1 -= 12
        for r in result.risk_flags:
            if r.severity == "high":
                comp1 -= 10
            elif r.severity == "medium":
                comp1 -= 5
        for fi in result.fraud_indicators:
            comp1 -= 10 if fi.get("severity") == "high" else 4
        comp1 -= len(result.contradictions) * 8
        comp1 = max(0, comp1)

        # comp2: field coherence warnings (30 pts)
        comp2 = 30
        for w in result.warnings:
            if w.severity == "warning":
                comp2 -= 5
            elif w.severity == "info":
                comp2 -= 1
        comp2 = max(0, comp2)

        # comp3: documentary coverage (20 pts)
        comp3 = round(doc_quality / 100.0 * 20)

        # comp4: temporal coherence (10 pts)
        comp4 = 10
        temporal_flags = [
            r for r in result.risk_flags
            if "temporal" in (getattr(r, "category", "") or "").lower()
            or "date" in (getattr(r, "field", "") or "").lower()
            or "expir" in (getattr(r, "description", "") or "").lower()
        ]
        for r in temporal_flags:
            if r.severity == "high":
                comp4 -= 5
            elif r.severity == "medium":
                comp4 -= 3
            else:
                comp4 -= 1
        comp4 = max(0, comp4)

        return min(100, comp1 + comp2 + comp3 + comp4)

    # ── Propagation ontologique ───────────────────────────────────

    @staticmethod
    def _collect_violated_articles(result: SemanticAnalysisResult) -> list[str]:
        """Collecte les articles violés depuis les warnings et risk_flags."""
        arts = []
        for w in result.warnings:
            if w.ucp_ref:
                arts.extend(w.ucp_ref.split("/"))
        for r in result.risk_flags:
            if r.ucp_ref:
                arts.extend(r.ucp_ref.split("/"))
        # Normaliser
        return [a.strip() for a in arts if re.match(r'Art\.\d', a.strip())]

    # ── Inférence Incoterm ────────────────────────────────────────

    def _infer_incoterm(self, entities: Any) -> Optional[str]:
        goods = _get_field(entities, "goods_description") or ""
        for inc in ["EXW","FCA","CPT","CIP","DAP","DPU","DDP","FAS","FOB","CFR","CIF","DAT"]:
            if re.search(rf'\b{inc}\b', goods.upper()):
                return inc
        return None

    # ── Recommandations enrichies ─────────────────────────────────

    def _generate_recommendations(
        self,
        result: SemanticAnalysisResult,
        entities: Any,
        swift_type: str,
    ) -> list[str]:
        recs = []

        # Les contradictions ont leur propre section — ne pas les dupliquer ici

        errors = [w for w in result.warnings if w.severity == "error"]
        if errors:
            recs.extend(f"[ERREUR] {w.message}" for w in errors[:4])

        high_risks = [r for r in result.risk_flags if r.severity == "high"]
        if high_risks:
            recs.append(f"[RISQUE] {len(high_risks)} flag(s) critique(s) — compliance requise")

        if result.completeness_score < 70:
            missing = [
                f for f in _REQUIRED_FIELDS.get(swift_type, [])
                if not _get_field(entities, f)
            ]
            if missing:
                recs.append(f"[COMPLÉTUDE] Champs manquants: {', '.join(missing[:4])}")

        if result.doc_quality_score < 50:
            recs.append("[DOCUMENTS] Vérifier liste :46A: — score qualité faible")

        if result.bayesian_confidence < 0.5:
            recs.append(f"[CONFIANCE] Document dégradé — confiance Bayésienne: {result.bayesian_confidence:.2f}")

        # Propagations ontologiques
        high_prop = [p for p in result.ontology_propagations if p.get("severity") == "warning"]
        if high_prop:
            arts = ", ".join(p["to_article"] for p in high_prop[:3])
            recs.append(f"[ONTOLOGIE] Vérifier aussi: {arts} (propagation de risque)")

        # ── AXE 5 : Décision de routage basée sur final_score ─────────────
        recs.append(
            f"[ROUTAGE v4] Score composite: {result.final_score:.3f} → Décision: {result.routing_decision} "
            f"(moteur {result.engine_version} | {result.analysis_timestamp[:19]})"
        )

        if not recs and result.routing_decision == "ACCEPT":
            recs.append("✓ Document conforme — aucune action requise")

        return recs

    # ── Résumé narratif ───────────────────────────────────────────

    def _build_summary(self, entities: Any, result: SemanticAnalysisResult, swift_type: str) -> str:
        parts = []
        ref      = _get_field(entities, "lc_number") or "N/A"
        currency = _get_field(entities, "currency") or ""
        amount   = _get_field(entities, "amount") or ""

        def _clean(v: str) -> str:
            if not v: return v
            v = re.sub(r'\s*/adress\w*', '', v, flags=re.I).strip()
            v = re.sub(r'\s+Address\s+not\s+provided\s*$', '', v, flags=re.I).strip()
            return v

        applicant = _clean(_get_field(entities, "applicant") or "?")
        benef     = _clean(_get_field(entities, "beneficiary") or "?")
        expiry    = _get_field(entities, "expiry_date") or "?"

        type_labels = {
            "MT700": "Lettre de Crédit", "MT720": "LC Transférable",
            "MT760": "Garantie Bancaire", "MT790": "Message Frais",
            "MT799": "Message Libre",
        }
        lbl = type_labels.get(swift_type, swift_type)
        parts.append(f"{lbl} {ref}" + (f" — {currency} {amount}" if currency and amount else ""))

        if swift_type in ("MT700", "MT720"):
            parts += [f"Donneur d'ordre: {applicant}", f"Bénéficiaire: {benef}",
                      f"Expiration: {expiry}"]

        errors   = [w for w in result.warnings if w.severity == "error"]
        highrisk = [r for r in result.risk_flags if r.severity == "high"]
        contrad  = result.contradictions

        if errors:    parts.append(f"{len(errors)} erreur(s)")
        if highrisk:  parts.append(f"{len(highrisk)} risque(s) élevé(s)")
        if contrad:   parts.append(f"{len(contrad)} contradiction(s) tag/clause")

        parts.append(
            f"Score: {result.completeness_score}/100 | "
            f"Cohérence: {result.semantic_score}/100 | "
            f"Docs: {result.doc_quality_score}/100 | "
            f"Confiance: {result.bayesian_confidence:.2f} | "
            f"Final: {result.final_score:.3f} → {result.routing_decision}"
        )
        return " | ".join(parts)

    # ── Inférence champs manquants ────────────────────────────────

    def _infer_missing_fields(self, entities: Any, swift_type: str) -> dict:
        inferred = {}
        goods = _get_field(entities, "goods_description")
        if goods:
            for inc in ["EXW","FCA","CPT","CIP","DAP","DPU","DDP","FAS","FOB","CFR","CIF","DAT"]:
                if re.search(rf'\b{inc}\b', goods.upper()):
                    inferred["incoterm_inferred"] = inc
                    break

        draft = _get_field(entities, "draft_at")
        if draft and not _get_field(entities, "available_by"):
            if re.search(r'SIGHT', str(draft), re.I):
                inferred["available_by_inferred"] = "BY PAYMENT"
            elif re.search(r'ACCEPT', str(draft), re.I):
                inferred["available_by_inferred"] = "BY ACCEPTANCE"
            elif re.search(r'\d+\s*DAYS?', str(draft), re.I):
                inferred["available_by_inferred"] = "BY DEFERRED PAYMENT"

        if swift_type in ("MT700", "MT720") and not _get_field(entities, "applicable_rules"):
            inferred["applicable_rules_default"] = "UCP 600"
        if swift_type == "MT760" and not _get_field(entities, "applicable_rules"):
            inferred["applicable_rules_default"] = "URDG 758"

        return inferred


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRE : Entraîner le modèle ML
# ══════════════════════════════════════════════════════════════════════

def train_risk_model(dataset_dir: str, output_path: str) -> None:
    """Entraîne un classifieur de risque sur training_samples_fixed.jsonl."""
    import os, json
    from pathlib import Path
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline as SKPipeline
        import joblib
    except ImportError as e:
        logger.error("Dépendances manquantes: %s", e)
        return

    dataset_path = Path(dataset_dir)
    candidates   = [
        dataset_path / "training_samples_fixed.jsonl",
        dataset_path / "training_samples.jsonl",
        dataset_path / "ner_dataset_fixed.jsonl",
    ]
    path = next((c for c in candidates if c.exists()), None)

    if path is None:
        os.makedirs(output_path, exist_ok=True)
        joblib.dump(None, os.path.join(output_path, "risk_clf.pkl"))
        logger.warning("Aucune donnée d'entraînement — modèle vide sauvegardé")
        return

    def _label(rec: dict) -> str:
        text  = rec.get("raw_text", rec.get("text", "")).upper()
        flags = [str(f).upper() for f in rec.get("risk_flags", [])]
        if any(k in text for k in ["IRAN","SYRIA","NORTH KOREA","DPRK","CUBA",
                                     "MYANMAR","VENEZUELA","SUDAN","BELARUS"]):
            return "SANCTIONS"
        if any(k in text for k in ["WEAPON","ARMS","NUCLEAR","NARCOTICS","PRECURSOR"]):
            return "SENSITIVE_GOODS"
        if any(k in text for k in ["SHELL COMPANY","BEARER","NOMINEE","FICTITIOUS"]):
            return "AML_PATTERN"
        if any("EXPIR" in f or "TOLERANCE" in f for f in flags):
            return "COMPLIANCE"
        if any("DOCUMENT" in f or "INVOICE" in f for f in flags):
            return "DOCUMENT_QUALITY"
        return "NORMAL"

    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec  = json.loads(line)
                text = rec.get("raw_text", rec.get("text", ""))
                if text.strip():
                    texts.append(text[:500])
                    labels.append(_label(rec))
            except Exception:
                continue

    if len(texts) < 5:
        logger.error("Pas assez de données (%d samples)", len(texts))
        return

    n_classes = len(set(labels))
    cv_folds  = max(2, min(3, len(labels) // max(1, n_classes)))

    pipeline = SKPipeline([
        ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 3))),
        ("clf",   GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)),
    ])
    scores = cross_val_score(pipeline, texts, labels, cv=cv_folds)
    logger.info("Risk model CV: %.3f ± %.3f", scores.mean(), scores.std())
    pipeline.fit(texts, labels)

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(pipeline, os.path.join(output_path, "risk_clf.pkl"))
    joblib.dump(None,     os.path.join(output_path, "risk_vectorizer.pkl"))
    logger.info("Modèle sauvegardé dans '%s'", output_path)