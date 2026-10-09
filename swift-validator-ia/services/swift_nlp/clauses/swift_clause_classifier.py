"""
swift_clause_classifier.py  —  v2.0
======================================
Classification avancée des clauses SWIFT selon UCP 600.

AMÉLIORATIONS v2 :
  ✦ Embeddings sémantiques (SentenceTransformer) remplacent Jaccard
      → "The goods shall be shipped by CMA CGM vessels only"
         → correctement classé "shipment_condition" même sans mots-clés
  ✦ Classifieur LogisticRegression entraînable sur clause_classification_fixed.jsonl
      → Fine-tunable sur les 207 exemples du dataset
  ✦ Mode RAG : retrieval des exemples les plus similaires pour enrichir le contexte
  ✦ Détection de clauses contradictoires au sein d'un même message
  ✦ Score d'importance enrichi (3 niveaux : high / medium / normal)
  ✦ Fallback gracieux : si sentence-transformers absent → scoring pondéré v1

Usage :
    # Mode standard (scoring pondéré + kNN Jaccard) :
    clf = SwiftClauseClassifier()

    # Mode sémantique (sentence-transformers requis) :
    clf = SwiftClauseClassifier(enable_embeddings=True)

    # Mode supervisé (modèle pré-entraîné sur dataset) :
    clf = SwiftClauseClassifier(model_path="./models/clause-clf")

    result = clf.classify(text)
    results = clf.extract_and_classify_from_swift(raw_text)
    contradictions = clf.detect_contradictions(results)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Frontières de champ MT700 (lookahead) — :46A: doit s'arrêter avant :47A:
_CLAUSE_FIELD_STOP: dict[str, str] = {
    "46A": r"47A|48|49[GH]?|71[BD]?|72[ZD]?|77[BEU]|78",
    "47A": r"48|49[GH]?|71[BD]?|72[ZD]?|77[BEU]|78",
    "49G": r"48|49H|71[BD]?|72[ZD]?|77[BEU]|78",
    "49H": r"48|49G|71[BD]?|72[ZD]?|77[BEU]|78",
    "71B": r"48|49[GH]?|72[ZD]?|77[BEU]|78",
    "71D": r"48|49[GH]?|72[ZD]?|77[BEU]|78",
    "78": r"48|49[GH]?|71[BD]?|72[ZD]?|77[BEU]",
    "72": r"48|49[GH]?|71[BD]?|77[BEU]|78",
    "77U": r"48|49[GH]?|71[BD]?|72[ZD]?|77[BEU]|78",
}

# CF1 : champs MT700 porteurs de clauses (ordre = priorité d'extraction).
# Configurable — ajouter un tag ici suffit à l'inclure dans la segmentation.
_CLAUSE_BEARING_TAGS: list[str] = ["47A", "46A", "49G", "49H", "71B", "71D", "78", "72", "77U"]

# Libellés d'en-tête de champ à retirer quand ils fuient dans la 1re clause.
_CLAUSE_FIELD_LABEL = re.compile(
    r"^\s*(?:ADDITIONAL\s+CONDITIONS?|DOCUMENTS?\s+REQUIRED|"
    r"SPECIAL\s+PAYMENT\s+CONDITIONS?(?:\s+FOR\s+(?:BENEFICIARY|BANK))?|"
    r"CHARGES?|INSTRUCTIONS?\s+TO\s+THE\s+\w+(?:\s+BANK)?|"
    r"SENDER\s+TO\s+RECEIVER\s+INFORMATION)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def _swift_ocr_token_fixes(text: str) -> str:
    """Corrections OCR sûres avant extraction des clauses."""
    if not text:
        return text
    t = re.sub(r"\bONL\s+Y\b", "ONLY", text, flags=re.IGNORECASE)
    t = re.sub(r"\bU5D(\s*\d)", r"USD\1", t, flags=re.IGNORECASE)
    # Espaces/tabs multiples (hors retours ligne) → simple espace
    t = re.sub(r"[ \t]{2,}", " ", t)
    # Puces OCR fréquentes normalisées
    t = re.sub(r"[·▪●◦‣]", "•", t)
    return t


def _clause_stop_regex_for_tag(tag: str) -> str:
    return _CLAUSE_FIELD_STOP.get(tag.upper(), r"48|49|71[BD]?|72[ZD]?|77[BEU]|78")


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ClauseClassification:
    text:        str
    type:        str            # type de clause
    importance:  str            # "high" | "medium" | "normal"
    confidence:  float          # [0.0 – 1.0]
    method:      str            # "rule" | "scoring" | "embedding" | "supervised" | "fallback"
    scores:      dict           # scores bruts par catégorie
    # v2 : nouveaux champs
    similar_examples: list[dict] = None  # exemples similaires du dataset (RAG)
    ucp_refs:    list[str] = None        # articles UCP 600 applicables
    risk_level:  str = "normal"          # "high" | "medium" | "normal"

    def __post_init__(self):
        if self.similar_examples is None:
            self.similar_examples = []
        if self.ucp_refs is None:
            self.ucp_refs = []

    def to_dict(self) -> dict:
        return {
            "text":            self.text,
            "type":            self.type,
            "importance":      self.importance,
            "confidence":      round(self.confidence, 3),
            "method":          self.method,
            "risk_level":      self.risk_level,
            "ucp_refs":        self.ucp_refs,
            "similar_examples": self.similar_examples[:2],  # top 2 pour le rapport
        }


# ══════════════════════════════════════════════════════════════════════
# RÈGLES DÉTERMINISTES (inchangées, toujours en première ligne)
# ══════════════════════════════════════════════════════════════════════

_DETERMINISTIC_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(BILL OF LADING|INVOICE|CERTIFICATE|PACKING LIST|AIRWAY BILL|'
                r'INSPECTION CERT|WEIGHT CERT|HEALTH CERT|FUMIGATION|PHYTOSANITARY|'
                r'CARGO MANIFEST|VESSEL QUESTIONNAIRE|DRAFT|COPY OF)\b', re.I), "document_condition"),
    (re.compile(r'\b(PARTIAL SHIPMENT|TRANS(?:S)?HIPMENT|ON DECK|CHARTER PARTY|'
                r'STALE BILL|PORT OF LOADING|PORT OF DISCHARGE|LATEST SHIPMENT|'
                r'MULTIMODAL|COMBINED TRANSPORT|CONTAINER SHIP|BULK CARGO)\b', re.I), "shipment_condition"),
    (re.compile(r'\b(BANK CHARGES?|ALL CHARGES?|TELEGRAPHIC TRANSFER|REIMBURSEMENT|'
                r'DEFERRED PAYMENT|AT SIGHT|USANCE|ACCEPTANCE CREDIT|DISCOUNTING|'
                r'CONFIRMING BANK|INTEREST RATE|PREPAYMENT|BACK.TO.BACK)\b', re.I), "payment_condition"),
    (re.compile(r'\b(UCP\s*6?0?0|URDG\s*7?5?8|ISP\s*98|EUCP|PRESENTATION PERIOD|'
                r'BANKING DAYS?|FORCE MAJEURE|APPLICABLE LAW|JURISDICTION|'
                r'AMENDMENTS? MUST|EXAMINING BANK|COMPLIANT DOCUMENTS?|'
                r'DOCUMENTS? MUST BE PRESENTED WITHIN)\b', re.I), "compliance_condition"),
    (re.compile(r'\b(TOLERANCE|PLUS MINUS|PERCENT(?:AGE)?|QUANTITY TOLERANCE|'
                r'AMOUNT TOLERANCE|OVERCOLLECTION|UNDERDRAWING|MINIMUM DRAWING|'
                r'MAXIMUM DRAWING|AUTO.REINSTATEMENT|PRICE ESCALATION|'
                r'AMOUNT MAY VARY|CREDIT AMOUNT SHALL|INVOICE TO SHOW EXACT)\b', re.I), "tolerance_condition"),
]

_SCORING_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "document_condition": [
        ("DOCUMENT", 2), ("ORIGINAL", 2), ("COPY", 1), ("COPIES", 1),
        ("SIGNED", 2), ("ISSUED BY", 2), ("ENDORSED", 2), ("COUNTERSIGNED", 2),
        ("LEGALIZED", 1), ("NOTARIZED", 1), ("MUST INDICATE", 1), ("REQUIRED", 1),
        ("ACCEPTABLE", 1), ("IN LIEU", 2), ("LC NUMBER", 2),
    ],
    "shipment_condition": [
        ("SHIPMENT", 2), ("TRANSSHIP", 3), ("PARTIAL", 2), ("ON DECK", 3),
        ("PORT", 1), ("LOADING", 2), ("DISCHARGE", 2), ("VESSEL", 2),
        ("CONTAINER", 2), ("CARRIER", 1), ("ROUTING", 2), ("ALLOWED", 1),
        ("NOT ALLOWED", 2), ("NOT ACCEPTABLE", 2), ("LINER TERMS", 3),
    ],
    "payment_condition": [
        ("PAYMENT", 2), ("CHARGES", 2), ("ACCOUNT", 1), ("SIGHT", 2),
        ("DEFERRED", 2), ("USANCE", 3), ("ACCEPTANCE", 2), ("NEGOTIATE", 1),
        ("REIMBURSE", 3), ("TELEGRAPHIC", 2), ("DISCOUNT", 2), ("PREPAY", 2),
        ("INTEREST", 2), ("RATE", 1), ("TRANSFER", 1),
    ],
    "compliance_condition": [
        ("UCP", 3), ("URDG", 3), ("ISP98", 3), ("PRESENT", 1),
        ("DAYS", 1), ("BANKING DAY", 2), ("DEADLINE", 2), ("EXPIRY", 1),
        ("COMPLIANT", 2), ("DISCREPAN", 2), ("AMENDMENT", 2), ("APPLICABLE", 1),
        ("RULES", 1), ("LAW", 1), ("JURISDICTION", 2), ("FORCE MAJEURE", 3),
    ],
    "tolerance_condition": [
        ("TOLERANCE", 3), ("PERCENT", 2), ("%", 2), ("PLUS", 1), ("MINUS", 1),
        ("QUANTITY", 2), ("AMOUNT", 1), ("VARY", 2), ("ADJUST", 2),
        ("MAXIMUM", 1), ("MINIMUM", 1), ("DRAWING", 2), ("REINSTAT", 3),
    ],
}

# UCP 600 articles associés à chaque type
_UCP_REFS: dict[str, list[str]] = {
    "document_condition":  ["Art.14", "Art.17", "Art.18", "Art.19", "Art.20", "Art.28"],
    "shipment_condition":  ["Art.19", "Art.20", "Art.22", "Art.23", "Art.26"],
    "payment_condition":   ["Art.6", "Art.7", "Art.8", "Art.12", "Art.13"],
    "compliance_condition":["Art.14c", "Art.16", "Art.29", "Art.38", "Art.39"],
    "tolerance_condition": ["Art.30", "Art.31"],
}

# Importance (3 niveaux : high / medium / normal)
# Structure : {"high": Pattern, "normal": Pattern}  — defaut = "medium"
_IMPORTANCE_RULES: dict[str, dict[str, re.Pattern]] = {
    "document_condition": {
        # high = original requis, couverture specifique, obligation stricte
        "high": re.compile(
            r'FULL\s+SET|ORIGINAL.*SIGNED|SIGNED.*ORIGINAL|AUTHENTICATED|COUNTERSIGNED|'
            r'NOTARIZED|LEGALIZED|IN\s+ORIGINAL\b|ALL\s+ORIGINALS|ENDORSED.*ORIGINAL|'
            r'ORIGINAL\s+(?:SIGNED|INSURANCE|BILL|QUALITY|HEALTH|PHYTO|FUMIGATION|'
            r'AIRWAY|WEIGHT|VESSEL|BENEFICIARY|CARGO)|'
            r'MUST\s+COVER\b|MUST\s+BE\s+ISSUED\s+IN\b|MUST\s+BE\s+PRESENTED\s+WITHIN|'
            r'MUST\s+INDICATE\s+(?:CREDIT|LC|CONTRACT)|'
            r'CERTIFICATE\s+(?:MUST|REQUIRED.*ORIGINAL)|'
            r'THREE\s+ORIGINALS|TWO\s+ORIGINALS', re.I),
        # normal = copies, electronique, preference, boilerplate
        "normal": re.compile(
            r'ELECTRONIC\b|FAXED\b|EMAIL\b|IN\s+TRANSIT|PENDING\s+ORIGINAL|'
            r'NOT\s+REQUIRED\b|SUFFICIENT\b|MAY\s+SUBMIT|STANDARD\s+SHIPPING|'
            r'DRAFT\s+NOT\s+REQUIRED|VIA\s+(?:COURIER|EMAIL)|PRIOR\s+TO\s+ORIGINAL|'
            r'IN\s+THREE\s+COPIES\b|THREE\s+COPIES\b', re.I),
    },
    "shipment_condition": {
        # high = interdictions strictes, deadlines, obligations specifiques
        "high": re.compile(
            r'NOT\s+ALLOWED|NOT\s+ACCEPTABLE|NOT\s+PERMITTED|STRICTLY\s+PROHIBITED|'
            r'PROHIBITED\b|STRICTLY\s+FORBIDDEN|CHARTER\s+PARTY\s+(?:BILL|NOT)|'
            r'STALE\s+(?:BILL|DOC)|ON\s+DECK\s+NOT|FREIGHT\s+MUST\s+BE\s+PREPAID|'
            r'DIRECT\s+VESSEL\s+ONLY|NO\s+TRANSHIPMENT|AIR\s+SHIPMENT\s+NOT|'
            r'ROAD\s+TRANSPORT\s+ONLY|OCEAN\s+FREIGHT\s+ONLY|'
            r'LATEST\s+SHIPMENT|MUST\s+ARRIVE\s+DESTINATION|'
            r'GOODS\s+MUST\s+BE\s+SHIPPED\s+IN|MUST\s+BE\s+MADE\s+ON\s+LINER|'
            r'GOODS\s+MUST\s+ARRIVE|ON\s+BOARD\s+NOTATION\s+MANDATORY', re.I),
        # normal = preferences, options, flexibilite
        "normal": re.compile(
            r'PREFERRED\b(?!\s+BUT\s+NOT)|AT\s+CARRIER\s+OPTION|VESSEL\s+AGE\b|'
            r'MARKS\s+AS\s+PER|VIA\s+ANY\s+PORT|FEEDER\s+VESSEL|ROUTE\s+VIA|'
            r'CAPE\s+ROUTE|FLAG\s+VESSEL\b|BREAKBULK|AS\s+CONVENIENT\b', re.I),
    },
    "payment_condition": {
        # high = termes de paiement, credit disponible, prepaiement
        "high": re.compile(
            r'AT\s+SIGHT|DEFERRED\s+PAYMENT|USANCE\b|ACCEPTANCE\s+CREDIT|BACK.TO.BACK|'
            r'MIXED\s+PAYMENT|PREPAYMENT\b|NEGOTIATION\s+AUTHORIZED|TRANSFERABLE\b|'
            r'RED\s+CLAUSE|PAYMENT\s+(?:AT|ONLY)\b|CREDIT\s+AVAILABLE\s+BY|'
            r'USANCE\s+DRAFT|PAYMENT\s+EFFECTED|USANCE\s+CREDIT|'
            r'PAYMENT\s+AT\s+SIGHT|DEFERRED\s+\d+\s+DAYS|'
            r'ACCEPTANCE\s+OF\s+DRAFT|REIMBURSEMENT\s+BY\s+SWIFT', re.I),
        # normal = frais standards, boilerplate
        "normal": re.compile(
            r'STANDARD\s+SWIFT\b|STANDARD\s+(?:CHARGE|FEE|TARIFF)|MINOR\s+BANKING|'
            r'POSTAGE\b|STAMP\s+DUTI|COMMUNICATION\s+CHARGE|COURIER\s+CHARGE|'
            r'HANDLING\s+FEE\b|AS\s+PER\s+STANDARD\s+TARIFF|AS\s+PER\s+SCHEDULE|'
            r'AS\s+PER\s+BANK\s+TARIFF|AS\s+PER\s+LOCAL\s+PRACTICE', re.I),
    },
    "compliance_condition": {
        # high = regles applicables, expiry, force majeure, juridiction, presentations
        "high": re.compile(
            r'UCP\s*6?0?0|URDG\s*7?5?8|ISP\s*98|EUCP\b|PRESENTATION\s+PERIOD|'
            r'FORCE\s+MAJEURE|JURISDICTION\b|EXPIRY\s+DATE|EXPIRY\s+(?:AT|ON|IS)|'
            r'DOCUMENTS?\s+MUST\s+BE\s+PRESENTED\s+WITHIN|AFTER\s+EXPIRY|'
            r'EXPIRY\s+PLACE|COURTS?\s+OF\b|'
            r'AMENDMENTS?\s+(?:MUST|REQUIRE|REQUIRE\s+WRITTEN)|'
            r'AMENDMENTS?\s+MUST\s+BE\s+INCORPORATED|'
            r'COMPLIANT\s+PRESENTATION|PRESENTATION\s+STRICTLY|'
            r'DOCUMENTS?\s+MUST\s+COMPLY\s+WITH|PLACE\s+OF\s+EXPIRY', re.I),
        # normal = pratique standard, loi applicable info, non-applicabilite
        "normal": re.compile(
            r'STANDARD\s+(?:BANKING|PRACTICE|UCP)|NOT\s+APPLICABLE\b|'
            r'NOT\s+TRANSFERABLE\b|AUTO(?:MATIC)?\s+RENEWAL\s+NOT|'
            r'HOLIDAYS\b|INCOTERMS\b|SANCTIONS\s+COMPLIANCE|ANTI.MONEY\s+LAUNDERING|'
            r'LOCAL\s+BANKING\s+PRACTICE|GOVERNING\s+LAW\s+(?:FRENCH|LOCAL|UAE(?!\s+LAW\s+GOVERN))', re.I),
    },
    "tolerance_condition": {
        # high = zero tolerance, exacte, interdit
        "high": re.compile(
            r'NOT\s+PERMITTED\b|ZERO\s+(?:TOLERANCE|PERCENT)|'
            r'EXACT\s+(?:QUANTITY|AMOUNT|WEIGHT|INVOICE|UNIT)|'
            r'NO\s+VARIANCE\b|MUST\s+NOT\s+EXCEED\b|STRICTLY\s+ENFORCED|'
            r'MATCH\s+EXACTLY|FIXED\s+(?:AMOUNT|QUANTITY|NO)|'
            r'ZERO\s+TOLERANCE|TOLERANCE\s+NOT\s+PERMITTED|'
            r'CREDIT\s+AMOUNT\s+FIXED|QUANTITY\s+MUST\s+BE\s+EXACT|'
            r'TOTAL\s+DRAWINGS?\s+SHALL\s+NOT', re.I),
        # normal = reinstatement automatique, escalation, ajustements mineurs
        "normal": re.compile(
            r'AUTO.REINSTATEMENT|PRICE\s+ESCALATION\s+CLAUSE\b|REVOLVING\s+CREDIT|'
            r'REINSTATES?\s+AUTOMATICALLY|UNDERDRAWING\s+PERMITTED|'
            r'OVERSHIPMENT\s+OF\s+\d|REINSTATE[DS]?\s+AFTER|'
            r'DRAWING\s+AMOUNT\s+ADJUSTED|ORIGINAL\s+AMOUNT\b', re.I),
    },
}

# Patterns de contradiction entre clauses
_CONTRADICTION_PATTERNS: list[tuple[re.Pattern, re.Pattern, str]] = [
    (
        re.compile(r'PARTIAL SHIPMENT.*ALLOWED', re.I),
        re.compile(r'PARTIAL SHIPMENT.*NOT ALLOWED', re.I),
        "Contradiction: expédition partielle autorisée ET interdite",
    ),
    (
        re.compile(r'TRANS(?:S)?HIPMENT.*(?:ALLOWED|PERMITTED)', re.I),
        re.compile(r'TRANS(?:S)?HIPMENT.*(?:NOT ALLOWED|NOT PERMITTED|PROHIBITED)', re.I),
        "Contradiction: transbordement autorisé ET interdit",
    ),
    (
        re.compile(r'AT SIGHT', re.I),
        re.compile(r'DEFERRED|USANCE|\d+\s*DAYS', re.I),
        "Contradiction: paiement à vue ET à terme dans la même clause",
    ),
    (
        re.compile(r'(?:MAY\s+ADD|WITH)\s+CONFIRMATION|CONFIRMATION\s+(?:MAY\s+ADD|REQUESTED|ADDED)', re.I),
        re.compile(r'WITHOUT\s+CONFIRMATION|CONFIRMATION\s+WITHOUT', re.I),
        "Contradiction: confirmation demandée ET sans confirmation",
    ),
    (
        re.compile(r'(?:ALL\s+)?CHARGES?\s+(?:ARE\s+)?FOR\s+(?:THE\s+)?(?:ACCOUNT\s+OF\s+)?BENEFICIARY', re.I),
        re.compile(r'(?:ALL\s+)?CHARGES?\s+(?:ARE\s+)?FOR\s+(?:THE\s+)?(?:ACCOUNT\s+OF\s+)?APPLICANT', re.I),
        "Contradiction: frais à la charge du bénéficiaire ET du donneur d'ordre",
    ),
]

# CF6 : marqueurs de polarité négative (interdiction / négation) pour détecter
# deux clauses de même sujet mais de sens opposé.
_NEG_POLARITY = re.compile(
    r'\b(?:NOT|NO|NON|WITHOUT|PROHIBIT(?:ED)?|FORBIDD?EN|DISALLOW(?:ED)?|'
    r'NOT\s+ALLOWED|NOT\s+PERMITTED|NOT\s+ACCEPTABLE|EXCLUD(?:ED|ING)?|NEVER)\b',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════
# MODULE EMBEDDINGS (optionnel)
# ══════════════════════════════════════════════════════════════════════

class _EmbeddingClassifier:
    """
    Classifieur basé sur SentenceTransformer + LogisticRegression.

    Modes :
      1. kNN cosine sur les exemples du dataset (sans entraînement)
      2. LR supervisé si un modèle sauvegardé est disponible

    Modèle recommandé : paraphrase-multilingual-MiniLM-L12-v2
    (multi-langue, léger 120MB, bon pour FR/EN mixte)
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        clf_path:   Optional[str] = None,
    ) -> None:
        self._encoder   = None
        self._clf       = None
        self._available = False
        self._example_embeddings: Optional[list] = None
        self._example_labels:     Optional[list] = None

        try:
            from ..common.swift_st_cache import get_sentence_transformer

            self._encoder = get_sentence_transformer(model_name)
            if self._encoder is None:
                logger.info("sentence-transformers non installé ou échec — embeddings désactivés")
                return
            self._available = True
            logger.info("EmbeddingClassifier: modèle '%s' (cache partagé)", model_name)
        except Exception as exc:
            logger.warning("EmbeddingClassifier: échec chargement: %s", exc)
            return

        # Charger le classifieur LR supervisé si disponible
        if clf_path:
            try:
                import joblib
                self._clf = joblib.load(clf_path)
                logger.info("EmbeddingClassifier: classifieur LR chargé depuis '%s'", clf_path)
            except Exception as exc:
                logger.warning("Classifieur LR non chargé: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def index_examples(self, examples: list[dict]) -> None:
        """Pré-calcule les embeddings des exemples du dataset (pour kNN)."""
        if not self._available or not examples:
            return
        texts = [ex["text"] for ex in examples]
        try:
            self._example_embeddings = self._encoder.encode(
                texts, batch_size=32, show_progress_bar=False
            )
            self._example_labels = [ex["type"] for ex in examples]
            logger.info(
                "EmbeddingClassifier: %d exemples indexés", len(examples)
            )
        except Exception as exc:
            logger.warning("Erreur indexation embeddings: %s", exc)

    def classify_knn(
        self, text: str, k: int = 5
    ) -> tuple[Optional[str], float, list[dict]]:
        """
        kNN cosine sur les embeddings des exemples.
        Retourne (label, confidence, similar_examples).
        """
        if not self._available or self._example_embeddings is None:
            return None, 0.0, []

        import numpy as np

        try:
            query_emb = self._encoder.encode([text])[0]

            # Cosine similarity avec tous les exemples
            norms = np.linalg.norm(self._example_embeddings, axis=1, keepdims=True)
            norms_q = np.linalg.norm(query_emb)
            if norms_q < 1e-8:
                return None, 0.0, []

            cos_sims = (self._example_embeddings @ query_emb) / (norms.flatten() * norms_q + 1e-8)

            # Top-k voisins
            top_k_idx = np.argsort(cos_sims)[-k:][::-1]
            top_k_sims = cos_sims[top_k_idx]

            if top_k_sims[0] < 0.50:  # seuil de similarité minimum
                return None, 0.0, []

            # Vote pondéré par similarité
            vote: dict[str, float] = {}
            similar_examples = []
            for idx, sim in zip(top_k_idx, top_k_sims):
                label = self._example_labels[idx]
                vote[label] = vote.get(label, 0.0) + float(sim)
                similar_examples.append({
                    "type":       label,
                    "similarity": round(float(sim), 3),
                })

            winner = max(vote, key=vote.__getitem__)
            total  = sum(vote.values())
            conf   = vote[winner] / total if total > 0 else 0.0

            return winner, float(conf), similar_examples

        except Exception as exc:
            logger.debug("EmbeddingClassifier kNN error: %s", exc)
            return None, 0.0, []

    def classify_supervised(self, text: str) -> tuple[Optional[str], float]:
        """Classification via le modèle LR supervisé."""
        if not self._available or not self._clf or not self._encoder:
            return None, 0.0
        try:
            emb   = self._encoder.encode([text])
            proba = self._clf.predict_proba(emb)[0]
            import numpy as np
            best_idx = int(np.argmax(proba))
            return self._clf.classes_[best_idx], float(proba[best_idx])
        except Exception as exc:
            logger.debug("EmbeddingClassifier supervised error: %s", exc)
            return None, 0.0

    def check_coherence(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Calcule la similarité cosine entre deux clauses.
        Utile pour détecter les incohérences sémantiques.
        """
        if not self._available or not self._encoder:
            return 0.5
        try:
            import numpy as np
            embs = self._encoder.encode(
                [text1, text2], show_progress_bar=False
            )
            sim  = float(
                (embs[0] @ embs[1])
                / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
            )
            return sim
        except Exception:
            return 0.5


# ══════════════════════════════════════════════════════════════════════
# CLASSIFIEUR PRINCIPAL v2
# ══════════════════════════════════════════════════════════════════════

class SwiftClauseClassifier:
    """
    Classifieur de clauses SWIFT v2.
    Stratégie en cascade :
      1. Règles déterministes (conf 0.97)
      2. Classifieur LR supervisé (si modèle dispo)
      3. kNN sémantique avec embeddings (si sentence-transformers dispo)
      4. Scoring pondéré + kNN Jaccard (fallback universel)
    """

    _CONFIDENCE_THRESHOLD = 0.35
    _RULE_CONFIDENCE      = 0.97

    def __init__(
        self,
        enable_embeddings: bool = True,
        model_path:        Optional[str] = None,
        embedding_model:   str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        # Charger dataset
        self._dataset_features: dict[str, list[str]] = {}
        self._dataset_examples:  list[dict]           = []
        self._dataset_type_dist: dict[str, int]       = {}

        try:
            try:
                from services.swift_nlp.common.swift_dataset_loader import SwiftDatasetLoader
            except ImportError:
                from ..common.swift_dataset_loader import SwiftDatasetLoader
            ds = SwiftDatasetLoader.instance().get()
            if ds.loaded:
                self._dataset_features = ds.clause_features
                self._dataset_examples = ds.clause_examples
                self._dataset_type_dist= ds.clause_type_dist
                logger.debug(
                    "ClauseClassifier v2: %d exemples, %d feature sets",
                    len(self._dataset_examples), len(self._dataset_features)
                )
        except Exception as exc:
            logger.debug("Dataset non disponible: %s", exc)

        # Classifieur d'importance (TF-IDF + LR, trained offline)
        self._importance_clf = None
        try:
            import joblib
            from pathlib import Path
            _imp_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "importance_clf.pkl"
            if _imp_path.exists():
                self._importance_clf = joblib.load(str(_imp_path))
                logger.debug("Importance classifier loaded: %s", _imp_path)
        except Exception as _exc:
            logger.debug("Importance classifier unavailable: %s", _exc)

        # Module embeddings
        self._embedder: Optional[_EmbeddingClassifier] = None
        if enable_embeddings:
            # Auto-discovery: models/clause-clf/clf.pkl (trained by train_clause_models.py)
            # Explicit model_path takes priority
            if model_path:
                clf_path = model_path + "/clf.pkl"
            else:
                _auto = Path(__file__).resolve().parents[3] / "models" / "clause-clf" / "clf.pkl"
                clf_path = str(_auto) if _auto.exists() else None

            self._embedder = _EmbeddingClassifier(
                model_name=embedding_model,
                clf_path=clf_path,
            )
            if self._embedder.available and self._dataset_examples:
                self._embedder.index_examples(self._dataset_examples)

    # ── API publique ──────────────────────────────────────────────────

    def classify(self, text: str) -> ClauseClassification:
        """Classifie une clause avec la meilleure méthode disponible."""
        if not text or not text.strip():
            return self._make_result(text, "unknown", "normal", 0.0, "fallback", {})

        # 1. Règles déterministes (toujours en premier)
        result = self._apply_rules(text)
        if result and result.confidence >= 0.90:
            return self._enrich(result)

        # 2. Classifieur supervisé (si disponible)
        if self._embedder and self._embedder.available:
            label_s, conf_s = self._embedder.classify_supervised(text)
            if label_s and conf_s >= 0.85:
                importance = self._compute_importance(text, label_s)
                result_s = self._make_result(
                    text, label_s, importance, conf_s, "supervised", {}
                )
                return self._enrich(result_s)

        # 3. kNN sémantique
        if self._embedder and self._embedder.available:
            label_e, conf_e, similar = self._embedder.classify_knn(text, k=5)
            if label_e and conf_e >= 0.65:
                importance = self._compute_importance(text, label_e)
                result_e = self._make_result(
                    text, label_e, importance, conf_e, "embedding", {}
                )
                result_e.similar_examples = similar
                return self._enrich(result_e)

        # 4. Scoring pondéré + kNN Jaccard (fallback)
        if result:
            return self._enrich(result)
        return self._enrich(self._apply_scoring(text))

    def classify_batch(self, texts: list[str]) -> list[ClauseClassification]:
        """Classification en batch (optimisée pour les embeddings)."""
        if self._embedder and self._embedder.available and len(texts) >= 1:
            return self._classify_batch_embedding(texts)
        return [self.classify(t) for t in texts]

    def _classify_batch_embedding(
        self, texts: list[str]
    ) -> list[ClauseClassification]:
        """Vrai batch : encode TOUS les textes en une seule passe SentenceTransformer.
        Réduit le temps de N×encode(1) → 1×encode(N) : ~10× plus rapide sur CPU.
        """
        if not texts:
            return []

        embedder = self._embedder
        if not embedder or not embedder.available or not embedder._encoder:
            return [self.classify(t) for t in texts]

        import numpy as np
        try:
            results: list = [None] * len(texts)
            need_embed_idx: list[int] = []

            # 1. Pré-classifier par règles (instantané, pas d'embedding)
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    results[i] = self._make_result(text, "unknown", "normal", 0.0, "fallback", {})
                    continue
                rule_res = self._apply_rules(text)
                if rule_res and rule_res.confidence >= 0.90:
                    results[i] = self._enrich(rule_res)
                else:
                    need_embed_idx.append(i)

            if not need_embed_idx:
                return results

            # 2. Encode en UNE seule passe tous les textes restants
            batch_texts = [texts[i] for i in need_embed_idx]
            batch_embs  = embedder._encoder.encode(
                batch_texts, batch_size=32, show_progress_bar=False
            )

            # 3. Classifier chaque embedding pré-calculé
            for local_i, global_i in enumerate(need_embed_idx):
                text = texts[global_i]
                emb  = batch_embs[local_i]

                sup_label, sup_conf = None, 0.0
                if embedder._clf:
                    try:
                        emb_2d    = emb.reshape(1, -1)
                        sup_label = embedder._clf.predict(emb_2d)[0]
                        proba     = embedder._clf.predict_proba(emb_2d)[0]
                        sup_conf  = float(max(proba))
                    except Exception:
                        pass

                knn_label, knn_conf = None, 0.0
                if embedder._example_embeddings is not None:
                    try:
                        norms  = np.linalg.norm(embedder._example_embeddings, axis=1)
                        norm_q = np.linalg.norm(emb)
                        if norm_q > 1e-8:
                            sims    = (embedder._example_embeddings @ emb) / (norms * norm_q + 1e-8)
                            top_idx = np.argsort(sims)[-5:][::-1]
                            if sims[top_idx[0]] >= 0.50:
                                vote: dict = {}
                                for ii, sim in zip(top_idx, sims[top_idx]):
                                    lbl = embedder._example_labels[ii]
                                    vote[lbl] = vote.get(lbl, 0.0) + float(sim)
                                winner    = max(vote, key=vote.__getitem__)
                                total     = sum(vote.values())
                                knn_label = winner
                                knn_conf  = vote[winner] / total if total > 0 else 0.0
                    except Exception:
                        pass

                rule_res = self._apply_rules(text)
                if sup_label and sup_conf >= 0.55:
                    imp_s = self._compute_importance(text, sup_label)
                    base = self._make_result(text, sup_label, imp_s, sup_conf, "supervised", {})
                elif knn_label and knn_conf >= 0.55:
                    imp_k = self._compute_importance(text, knn_label)
                    base = self._make_result(text, knn_label, imp_k, knn_conf, "embedding", {})
                elif rule_res:
                    base = rule_res
                else:
                    scoring = self._apply_scoring(text)
                    base = scoring if scoring else self._make_result(
                        text, "compliance_condition", "normal", 0.3, "fallback", {}
                    )
                results[global_i] = self._enrich(base)

            return results

        except Exception as exc:
            logger.warning("_classify_batch_embedding error: %s", exc)
            return [self.classify(t) for t in texts]

    def extract_and_classify_from_swift(
        self, raw_text: str
    ) -> list[ClauseClassification]:
        """Extrait et classifie les clauses depuis un message SWIFT brut.
        
        Corrections v7.1 :
          - Séparation stricte :46A: / :47A: (boundary inline ET multiline)
          - Filtre tags de fin (:48:/:49:/:71B/D:/:72Z:/:78:) dès l'extraction
          - Utilisation de classify_batch() pour encodage SentenceTransformer
            en une seule passe (10× plus rapide sur CPU)
          - Nettoyage tags SWIFT résiduels dans les blocs
        """
        raw_text = _swift_ocr_token_fixes(raw_text)

        stops_union = "|".join(
            sorted({p for v in _CLAUSE_FIELD_STOP.values() for p in v.split("|")})
        )
        _END_TAGS = re.compile(rf":(?:{stops_union}):", re.IGNORECASE)

        raw_items: list[str] = []

        for tag in _CLAUSE_BEARING_TAGS:
            stop_alt = _clause_stop_regex_for_tag(tag)
            # Tolère OCR : espace optionnel avant « : », tag parfois sans premier « : »
            pat = re.compile(
                rf":{tag}:\s*(.*?)(?=\s*:?\s*(?:{stop_alt})\s*:|\Z)",
                re.DOTALL | re.IGNORECASE,
            )
            m = pat.search(raw_text)
            if not m:
                # OCR : « ONLY 47A: » sans « : » devant le tag
                loose = re.compile(
                    rf"(?<![0-9A-Za-z]){tag}\s*:\s*(.*?)(?=\s*:?\s*(?:{stop_alt})\s*:|\Z)",
                    re.DOTALL | re.IGNORECASE,
                )
                m = loose.search(raw_text)
            if not m:
                continue

            block = m.group(1).strip()

            # Retirer le libellé d'en-tête de champ qui fuit dans la 1re clause
            # (ex: « ADDITIONAL CONDITIONS », « DOCUMENTS REQUIRED », « CHARGES »)
            block = _CLAUSE_FIELD_LABEL.sub("", block, count=1).strip()

            # Tronquer sur tags de fin inline (sécurité supplémentaire)
            trunc = _END_TAGS.search(block)
            if trunc:
                block = block[:trunc.start()].strip()

            # Supprimer les tags SWIFT structurels résiduels qui commencent un champ
            # (pas les références UCP comme ":20: C ii" ou ":14: k")
            # Approche : remplacer les tags non suivis d'un texte UCP connu
            def _strip_residual_tags(text: str) -> str:
                # Protéger "Article :20: C ii" en remplaçant temporairement
                text = re.sub(r'([Aa]rt(?:icle)?\.?\s{1,3}):(\d{2}[A-Z]?):', r'\1#\2#', text)
                # Supprimer les tags structurels résiduels restants
                text = re.sub(r':\d{2}[A-Z]?:\s*', ' ', text)
                # Restaurer les références UCP protégées
                text = re.sub(r'#(\d{2}[A-Z]?)#', r':\1:', text)
                return text.strip()
            block = _strip_residual_tags(block)

            for item in self._split_clause_block(block):
                item = item.strip()
                if len(item) >= self._MIN_KEEP_LEN:
                    raw_items.append(item)

        # Classifier en batch (une seule passe SentenceTransformer)
        if not raw_items:
            return []

        clauses = self.classify_batch(raw_items)

        # CLF-02 : déduplication par similarité Jaccard
        return self._deduplicate_clauses(clauses, threshold=0.85)

    def _deduplicate_clauses(
        self, clauses: list, threshold: float = 0.85
    ) -> list:
        """CLF-02 : supprime les clauses quasi-identiques (sur-segmentation).
        Passe 1 : Jaccard mot-à-mot. Passe 2 (CF5) : sémantique via embeddings
        (même type uniquement) pour attraper les reformulations.
        """
        # ── Passe 1 : Jaccard ─────────────────────────────────────────
        unique = []
        for c in clauses:
            words_c = set(w for w in c.text.upper().split() if len(w) > 3)
            if not words_c:
                unique.append(c)
                continue
            is_dup = False
            for u in unique:
                words_u = set(w for w in u.text.upper().split() if len(w) > 3)
                if not words_u:
                    continue
                inter = len(words_c & words_u)
                union = len(words_c | words_u)
                if union > 0 and inter / union >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(c)

        # ── Passe 2 : sémantique (CF5) — reformulations de MÊME type ───
        if (self._embedder and self._embedder.available
                and self._embedder._encoder and len(unique) > 2):
            try:
                import numpy as np
                embs  = self._embedder._encoder.encode(
                    [c.text for c in unique], batch_size=32, show_progress_bar=False
                )
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs  = embs / (norms + 1e-8)
                kept, removed = [], set()
                for i in range(len(unique)):
                    if i in removed:
                        continue
                    kept.append(unique[i])
                    for j in range(i + 1, len(unique)):
                        if j in removed or unique[j].type != unique[i].type:
                            continue
                        if float(embs[i] @ embs[j]) >= 0.93:
                            removed.add(j)  # reformulation quasi-identique
                unique = kept
            except Exception:
                pass  # fallback silencieux : on garde la dédup Jaccard

        return unique

    def detect_contradictions(
        self, clauses: list[ClauseClassification]
    ) -> list[dict]:
        """
        Détecte les contradictions entre les clauses d'un même message.
        Utilise les patterns _CONTRADICTION_PATTERNS.
        """
        contradictions = []

        # Vérifications par patterns
        all_texts = " ".join(c.text for c in clauses)
        for pat_a, pat_b, msg in _CONTRADICTION_PATTERNS:
            if pat_a.search(all_texts) and pat_b.search(all_texts):
                clauses_a = [c for c in clauses if pat_a.search(c.text)]
                clauses_b = [c for c in clauses if pat_b.search(c.text)]
                # CLF-01 : vérifier que clause_a et clause_b sont bien distinctes
                if not clauses_a or not clauses_b:
                    continue
                text_a = clauses_a[0].text.strip()
                text_b = clauses_b[0].text.strip()
                if text_a[:60] == text_b[:60]:
                    continue  # même clause matchée par les deux patterns → faux positif
                contradictions.append({
                    "message":   msg,
                    "severity":  "error",
                    "clause_a":  text_a[:80],
                    "clause_b":  text_b[:80],
                })

        # Vérification sémantique (si embeddings disponibles)
        # Pré-encoder TOUTES les clauses en un seul batch → 1 seul appel encode()
        if self._embedder and self._embedder.available and self._embedder._encoder:
            import numpy as np
            all_texts   = [c.text for c in clauses]
            try:
                all_embs = self._embedder._encoder.encode(
                    all_texts, batch_size=32, show_progress_bar=False
                )
                # Normaliser une fois
                norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
                all_embs_norm = all_embs / (norms + 1e-8)

                same_type_groups: dict[str, list[tuple]] = {}
                for idx_c, c in enumerate(clauses):
                    same_type_groups.setdefault(c.type, []).append((idx_c, c))

                for clause_type, group in same_type_groups.items():
                    if len(group) < 2:
                        continue
                    for gi in range(len(group)):
                        for gj in range(gi + 1, len(group)):
                            idx_i, c_i = group[gi]
                            idx_j, c_j = group[gj]
                            # Cosine depuis les embeddings pré-calculés
                            sim = float(all_embs_norm[idx_i] @ all_embs_norm[idx_j])
                            # CF6 : contradiction = même sujet (similarité élevée)
                            # MAIS polarité opposée (l'une nie/interdit, pas l'autre).
                            # Le seuil sim<-0.10 d'origine ne se déclenchait jamais.
                            neg_i = bool(_NEG_POLARITY.search(c_i.text))
                            neg_j = bool(_NEG_POLARITY.search(c_j.text))
                            if sim >= 0.55 and neg_i != neg_j:
                                contradictions.append({
                                    "message":  f"Clauses {clause_type} potentiellement contradictoires "
                                                f"(même sujet sim={sim:.2f}, polarité opposée)",
                                    "severity": "warning",
                                    "clause_a": c_i.text[:80],
                                    "clause_b": c_j.text[:80],
                                })
            except Exception:
                pass  # Fallback silencieux — pas de détection sémantique

        return contradictions

    # ── Enrichissement ────────────────────────────────────────────────

    def _enrich(self, result: ClauseClassification) -> ClauseClassification:
        """Ajoute les références UCP et le niveau de risque."""
        result.ucp_refs = _UCP_REFS.get(result.type, [])

        imp = result.importance
        typ = result.type

        # Risque élevé : high + types critiques (paiement/conformité/document)
        if imp == "high" and typ in (
            "compliance_condition", "payment_condition", "document_condition"
        ):
            result.risk_level = "high"
        # Risque moyen : high + autres types OU medium + types critiques
        elif imp == "high" or (
            imp == "medium" and typ in ("compliance_condition", "payment_condition")
        ):
            result.risk_level = "medium"
        else:
            result.risk_level = "normal"

        return result

    # ── Règles déterministes ──────────────────────────────────────────

    def _apply_rules(self, text: str) -> Optional[ClauseClassification]:
        votes: dict[str, int] = {}
        for pattern, label in _DETERMINISTIC_RULES:
            if pattern.search(text):
                votes[label] = votes.get(label, 0) + 1
        if not votes:
            return None
        winner = max(votes, key=lambda k: votes[k])
        total  = sum(votes.values())
        confidence = min(0.97, 0.75 + votes[winner] / total * 0.22)
        importance = self._compute_importance(text, winner)
        return self._make_result(text, winner, importance, confidence, "rule", votes)

    # ── Scoring multi-critères (fallback) ─────────────────────────────

    def _apply_scoring(self, text: str) -> ClauseClassification:
        upper  = text.upper()
        scores: dict[str, float] = {}

        for category, keywords in _SCORING_KEYWORDS.items():
            score = sum(weight for kw, weight in keywords if kw in upper)
            scores[category] = float(score)

        if self._dataset_features:
            words = set(upper.split())
            for category, feat_words in self._dataset_features.items():
                bonus = sum(1.5 for fw in feat_words if fw in words)
                scores[category] = scores.get(category, 0.0) + bonus

        # kNN Jaccard sur dataset
        knn_winner, knn_conf = self._knn_jaccard(upper, k=5)
        if knn_winner and knn_conf >= 0.60:
            scores[knn_winner] = scores.get(knn_winner, 0.0) + knn_conf * 10

        if not any(scores.values()):
            return self._make_result(text, "compliance_condition", "normal", 0.30, "fallback", scores)

        total  = sum(scores.values())
        winner = max(scores, key=scores.__getitem__)
        conf   = min(0.93, scores[winner] / total) if total > 0 else 0.30

        if knn_winner == winner and knn_conf >= 0.60:
            conf   = min(0.95, conf + 0.05)
            method = "scoring+knn"
        else:
            method = "scoring"

        if conf < self._CONFIDENCE_THRESHOLD:
            winner, conf = "compliance_condition", 0.30

        importance = self._compute_importance(text, winner)
        return self._make_result(text, winner, importance, conf, method, scores)

    def _knn_jaccard(self, upper: str, k: int = 5) -> tuple[str | None, float]:
        if not self._dataset_examples:
            return None, 0.0
        words_q = set(w for w in upper.split() if len(w) > 3)
        if not words_q:
            return None, 0.0
        sims: list[tuple[float, str]] = []
        for ex in self._dataset_examples:
            words_ex = set(w for w in ex["text"].upper().split() if len(w) > 3)
            inter = len(words_q & words_ex)
            union = len(words_q | words_ex)
            if union > 0:
                sims.append((inter / union, ex["type"]))
        if not sims:
            return None, 0.0
        top_k = sorted(sims, key=lambda x: -x[0])[:k]
        if top_k[0][0] < 0.15:
            return None, 0.0
        vote: Counter = Counter()
        for sim, t in top_k:
            vote[t] += sim
        winner = vote.most_common(1)[0][0]
        total  = sum(vote.values())
        return winner, vote[winner] / total if total > 0 else 0.0

    # ── Importance ────────────────────────────────────────────────────

    def _compute_importance(self, text: str, clause_type: str) -> str:
        rules = _IMPORTANCE_RULES.get(clause_type, {})
        if rules.get("high") and rules["high"].search(text):
            return "high"
        if rules.get("normal") and rules["normal"].search(text):
            return "normal"
        # Fallback: trained TF-IDF classifier when rules are inconclusive
        if self._importance_clf is not None:
            try:
                feature = f"{clause_type} {text}"
                pred = self._importance_clf.predict([feature])[0]
                return pred
            except Exception:
                pass
        return "medium"

    # ── Utilitaires ───────────────────────────────────────────────────

    # CF3 : seuils de segmentation unifiés (remplacent les 8/10/5 épars)
    _MIN_CLAUSE_LEN  = 12   # en dessous → fragment recollé au précédent
    _MIN_KEEP_LEN    = 5    # longueur minimale d'une clause conservée
    _SENT            = "\x00"

    @classmethod
    def _split_clause_block(cls, block: str) -> list[str]:
        """CF2 : segmenteur robuste multi-stratégie.

        1. Tronque sur les tags de fin de message.
        2. Retire les tags SWIFT structurels résiduels (garde les réf. UCP « Art.20 C »).
        3. Découpe sur marqueurs d'énumération (puces, 1./1)/a)/(i), ' + ', ' ; ').
        4. À défaut : paragraphes (lignes vides) puis phrases (bloc OCR monolithique).
        5. Recolle les fragments trop courts au précédent.
        """
        # 1. Tronquer dès un tag de fin de message
        block = re.split(
            r':(?:47A|48|49[GH]?|71[BD]?|72[ZD]?|78|77[BEU]):', block, maxsplit=1
        )[0]
        # 2. Supprimer les tags structurels résiduels inline (sauf réf. UCP « Art.20 C »)
        block = re.sub(
            r'(?<![A-Za-z]):(?:20|27|31[CD]|32[AB]|39[ABC]|40[AE]|41[AaB]|42[ACDMP]'
            r'|43[PT]|44[ABCDEF]|45A|49[GH]?|50|51|52|53|57|59):(?!\s*C\s)',
            ' ', block,
        ).strip()
        if not block:
            return []

        parts = cls._segment_clauses(block)
        return [p for p in parts if len(p) >= cls._MIN_KEEP_LEN]

    @classmethod
    def _segment_clauses(cls, block: str) -> list[str]:
        """Découpage multi-stratégie d'un bloc texte en clauses."""
        SENT = cls._SENT
        s = block

        # a. Marqueurs d'énumération en début de ligne : 1. 1) (1) a) (a) (i) - • * +
        s = re.sub(
            r'(?:\n|^)[ \t]*(?:\(?\d{1,2}[.)]|\(?[A-Za-z][.)]|\([ivxlcIVXLC]+\)|[-•*+])[ \t]+',
            SENT, s,
        )
        # b. Marqueurs inline : ' + ', ' ; ', ' 1) ', ' (ii) '
        s = re.sub(r'\s+\+\s+', SENT, s)
        s = re.sub(r'\s*;[ \t]+', SENT, s)
        s = re.sub(r'(?<=\s)\(?\d{1,2}[.)][ \t]+', SENT, s)
        s = re.sub(r'(?<=\s)\([ivxlcIVXLC]+\)[ \t]+', SENT, s)

        if SENT in s:
            parts = s.split(SENT)
        else:
            # Pas de marqueur : paragraphes (lignes vides), sinon découpage par phrases
            paras = re.split(r'\n[ \t]*\n', s)
            if len(paras) > 1:
                parts = paras
            else:
                sentences = cls._sentence_split(s)
                parts = sentences if len(sentences) > 1 else [s]

        # Normaliser : retours-ligne internes → espace, espaces multiples compactés
        parts = [re.sub(r'\s+', ' ', p).strip() for p in parts]
        parts = [p for p in parts if p]
        return cls._merge_short(parts)

    @staticmethod
    def _sentence_split(text: str) -> list[str]:
        """Découpage par phrases avec garde-fous d'abréviations courantes."""
        protected = text
        for ab in ("NO.", "NOS.", "PCT.", "QTY.", "REF.", "ART.", "L/C.",
                   "CERT.", "ORIG.", "DOC.", "DOCS.", "MIN.", "MAX.", "APPROX.",
                   "I.E.", "E.G.", "ETC.", "INC.", "LTD.", "CO.", "MR.",
                   "MRS.", "ST.", "VS.", "U.S.", "NO.S"):
            protected = protected.replace(ab, ab.replace(".", "\x01"))
        # Fin de phrase = . ! ? suivi d'espace puis majuscule/chiffre
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', protected)
        return [p.replace("\x01", ".").strip() for p in parts if p.strip()]

    @classmethod
    def _merge_short(cls, parts: list[str]) -> list[str]:
        """Recolle au précédent tout fragment plus court que _MIN_CLAUSE_LEN."""
        out: list[str] = []
        for p in parts:
            if out and len(p) < cls._MIN_CLAUSE_LEN:
                out[-1] = f"{out[-1]} {p}".strip()
            else:
                out.append(p)
        return out

    @staticmethod
    def _make_result(
        text: str,
        clause_type: str,
        importance: str,
        confidence: float,
        method: str,
        scores: dict,
    ) -> ClauseClassification:
        return ClauseClassification(
            text=text,
            type=clause_type,
            importance=importance,
            confidence=confidence,
            method=method,
            scores=scores,
        )


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRE : Entraîner le classifieur LR sur le dataset
# ══════════════════════════════════════════════════════════════════════

def train_clause_classifier(
    dataset_dir: str,
    output_path: str,
    model_name:  str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> None:
    """
    Entraîne et sauvegarde un classifieur LR sur le dataset.

    Usage :
        train_clause_classifier(
            dataset_dir="./datasets",
            output_path="./models/clause-clf"
        )
    """
    import os
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np
        import joblib
    except ImportError as e:
        logger.error("Dépendances manquantes pour l'entraînement: %s", e)
        return

    try:
        from services.swift_nlp.common.swift_dataset_loader import SwiftDatasetLoader
    except ImportError:
        from ..common.swift_dataset_loader import SwiftDatasetLoader

    ds = SwiftDatasetLoader(dataset_dir).get()
    if not ds.loaded or not ds.clause_examples:
        logger.error("Dataset non disponible")
        return

    texts  = [ex["text"] for ex in ds.clause_examples]
    labels = [ex["type"] for ex in ds.clause_examples]

    logger.info("Encodage de %d clauses...", len(texts))
    encoder = SentenceTransformer(model_name)
    X = encoder.encode(texts, batch_size=32, show_progress_bar=True)

    # multi_class supprime dans scikit-learn >= 1.5 (multinomial est le defaut)
    import sklearn
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_ver >= (1, 5):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    else:
        clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial", solver="lbfgs")

    # cv adaptatif : evite l erreur si peu d exemples par classe
    n_classes = len(set(labels))
    cv_folds  = max(2, min(5, len(labels) // max(1, n_classes)))

    scores = cross_val_score(clf, X, labels, cv=cv_folds)
    logger.info("Cross-validation (cv=%d): %.3f +/- %.3f",
                cv_folds, scores.mean(), scores.std())

    clf.fit(X, labels)

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(clf, os.path.join(output_path, "clf.pkl"))
    encoder.save(output_path)
    logger.info("Modèle sauvegardé dans '%s'", output_path)