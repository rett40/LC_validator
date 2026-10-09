"""
spell_correction.py  —  v2.0 (couche OCR / pré-NLP)
======================================================
Correction orthographique avancée des messages SWIFT.

AMÉLIORATIONS v2 :
  ✦ Passe 4 : Validation contextuelle MLM (BERT fill-mask)
      → Valide que la correction est cohérente dans son contexte avant application
      → Évite les faux positifs (ex: "BANCK" dans un acronyme)
  ✦ Passe 5 : N-grammes de bigrammes SWIFT pour corrections multi-mots
      → "LETTRE DE CREDDIT" → "LETTRE DE CREDIT"
  ✦ Métriques de confiance par correction
  ✦ Cache LRU pour les suggestions Levenshtein (perf x5 sur batch)
  ✦ Détection de langue (FR/EN) pour adapter le dictionnaire

Usage :
    corrector = SwiftSpellCorrector()
    text_fixed, report = corrector.correct(raw_text)

    # Mode avancé avec validation contextuelle
    corrector = SwiftSpellCorrector(enable_contextual=True)
    text_fixed, report = corrector.correct(raw_text)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# DICTIONNAIRE STATIQUE — erreurs SWIFT connues
# ══════════════════════════════════════════════════════════════════════

_STATIC_DICT: dict[str, str] = {
    "CERTIFICAT":       "CERTIFICATE",
    "LOADNG":           "LOADING",
    "BENIFICIARY":      "BENEFICIARY",
    "BENIFECIARY":      "BENEFICIARY",
    "BENEFICARY":       "BENEFICIARY",
    "DISCHRGE":         "DISCHARGE",
    "ORGINAL":          "ORIGINAL",
    "SHPMENT":          "SHIPMENT",
    "SHIPING":          "SHIPPING",
    "REQURIED":         "REQUIRED",
    "IRREVOCBLE":       "IRREVOCABLE",
    "IRREVOKABLE":      "IRREVOCABLE",
    "INVOCE":           "INVOICE",
    "INSURENCE":        "INSURANCE",
    "INSURANSE":        "INSURANCE",
    "COMERCIAL":        "COMMERCIAL",
    "AMENDEMENT":       "AMENDMENT",
    "AMMENDMENT":       "AMENDMENT",
    "NEGOCIATION":      "NEGOTIATION",
    "NEGOTATION":       "NEGOTIATION",
    "NEGOCIATE":        "NEGOTIATE",
    "PRESENTATON":      "PRESENTATION",
    "PRESENTACION":     "PRESENTATION",
    "APLICANT":         "APPLICANT",
    "APLICATION":       "APPLICATION",
    "BANCK":            "BANK",
    "CONFORMATION":     "CONFIRMATION",
    "CONFIRMAION":      "CONFIRMATION",
    "CORRESPONDANT":    "CORRESPONDENT",
    "DOCUMANTS":        "DOCUMENTS",
    "DOCUMENTTS":       "DOCUMENTS",
    "DOCUMANTARY":      "DOCUMENTARY",
    "DRAFFT":           "DRAFT",
    "EXCHNAGE":         "EXCHANGE",
    "EXPIRRY":          "EXPIRY",
    "GAURANTEE":        "GUARANTEE",
    "GUARENTEE":        "GUARANTEE",
    "INDEMNIFCATION":   "INDEMNIFICATION",
    "INSTRCTION":       "INSTRUCTION",
    "LETER":            "LETTER",
    "NAGIVATION":       "NAVIGATION",
    "NOTIFICTION":      "NOTIFICATION",
    "PAYEMENT":         "PAYMENT",
    "PAYABEL":          "PAYABLE",
    "RECEPIENT":        "RECIPIENT",
    "REIMBOURSEMENT":   "REIMBURSEMENT",
    "REIMBURSMENT":     "REIMBURSEMENT",
    "SIGNATURY":        "SIGNATORY",
    "SUPLIER":          "SUPPLIER",
    "TRANSHIPMENT":     "TRANSSHIPMENT",
    "TRANSFERRABLE":    "TRANSFERABLE",
    "VALIDATY":         "VALIDITY",
    "RECIEVABLE":       "RECEIVABLE",
    "RECIEVE":          "RECEIVE",
    "FINACING":         "FINANCING",
    "FREEGHT":          "FREIGHT",
    "CUSTOMES":         "CUSTOMS",
    "DECLARTION":       "DECLARATION",
    "AUTHETICATION":    "AUTHENTICATION",
    "DISCREPENCY":      "DISCREPANCY",
    "COMODITY":         "COMMODITY",
    "MERCHANDIZE":      "MERCHANDISE",
    "COMPLYANCE":       "COMPLIANCE",
    "ARIVAL":           "ARRIVAL",
    "ENDORSMENT":       "ENDORSEMENT",
    "PREPAYEMENT":      "PREPAYMENT",
    "DEFFERED":         "DEFERRED",
    "CREDDIT":          "CREDIT",
    "STANDBYE":         "STANDBY",
    "CI F":             "CIF",
    "FO B":             "FOB",
    "CF R":             "CFR",
    "DA P":             "DAP",
    "IRRVOCABLE":       "IRREVOCABLE",
    "PARTAIL":          "PARTIAL",
    "TRANSSHIPMNT":     "TRANSSHIPMENT",
    "NEGOTAIBLE":       "NEGOTIABLE",
    "DOCUMENTRAY":      "DOCUMENTARY",
    "COMPLIANT":        "COMPLIANT",
    "ACCEPTABEL":       "ACCEPTABLE",
    "ENDORCEMENT":      "ENDORSEMENT",
    # v2 additions (SPL-05 : doublons "PRESENTATON","CONFIRMAION","DOCUMENTRAY" supprimés)
    "EXPRIY":           "EXPIRY",
    "EXPIRATON":        "EXPIRATION",
    "GARANTEE":         "GUARANTEE",
    "NEGOCIABLE":       "NEGOTIABLE",
    "PRSENTATION":      "PRESENTATION",
    "SHIPMNET":         "SHIPMENT",
    "SHIPPMENT":        "SHIPMENT",
    "ISSUENCE":         "ISSUANCE",
    "APLLICANT":        "APPLICANT",
    "BENFICIARY":       "BENEFICIARY",
    "BENEFICAIRY":      "BENEFICIARY",
    "AMENDENT":         "AMENDMENT",
    "INSUANCE":         "INSURANCE",
    "DOCUMNENT":        "DOCUMENT",
    "CORRESPODENT":     "CORRESPONDENT",
    "NEGOTAITION":      "NEGOTIATION",
    "REIMBURSMNET":     "REIMBURSEMENT",
}

# Bigrammes SWIFT fréquents (corrections multi-mots)
_BIGRAM_DICT: dict[str, str] = {
    "LETTER OF CREIDT":      "LETTER OF CREDIT",
    "LETTER OF CREDID":      "LETTER OF CREDIT",
    "BILL OF LAIDNG":        "BILL OF LADING",
    "BILL OF LAIDING":       "BILL OF LADING",
    "BILL OF LADNG":         "BILL OF LADING",
    "CERTIFICATE OF ORGINE": "CERTIFICATE OF ORIGIN",
    "CERTIFICATE OF ORGIN":  "CERTIFICATE OF ORIGIN",
    "PACKING LITS":          "PACKING LIST",
    "COMERCIAL INVOICE":     "COMMERCIAL INVOICE",
    "PARTIAL SHIPMNET":      "PARTIAL SHIPMENT",
    "TELEGRAPHIC TRANFER":   "TELEGRAPHIC TRANSFER",
    "DEFERRED PAYMNET":      "DEFERRED PAYMENT",
    "IRREVOCBLE CREDIT":     "IRREVOCABLE CREDIT",
    "STANDBY LETER":         "STANDBY LETTER",
    "DEMAND GUARENTEE":      "DEMAND GUARANTEE",
    "BANK GUARENTEE":        "BANK GUARANTEE",
}

# SPL-03/04 : pluriels et formes bancaires valides — jamais à corriger
_BANKING_VALID_FORMS = frozenset({
    "ORIGINALS", "COPIES", "DOCUMENTS", "INVOICES", "DRAFTS",
    "CERTIFICATES", "AMENDMENTS", "CORRECTIONS", "CHARGES",
    "ORIGINALS", "PAYMENTS", "SHIPMENTS", "ENDORSEMENTS",
    "DISCREPANCIES", "TRANSACTIONS", "INSTRUCTIONS",
    # Termes FR valides dans LC bilingues
    "MARCHANDISE", "MARCHANDISES", "QUALITE", "FACTURE",
    "CERTIFICAT",  # FR = valide (≠ EN CERTIFICATE)
    # Termes bancaires courts à ne pas toucher
    "UNDERNOTED", "TELEGRAPHIC", "SHIPPER", "INCOTERM",
    "PACKING", "IMPORTER", "DEFERMENT",
})

_SWIFT_VOCAB: list[str] = sorted({
    "IRREVOCABLE", "TRANSFERABLE", "CERTIFICATE", "DOCUMENT", "DOCUMENTS",
    "REQUIRED", "BENEFICIARY", "APPLICANT", "SHIPMENT", "LOADING",
    "DISCHARGE", "INSURANCE", "INVOICE", "ORIGINAL", "AMENDMENT",
    "NEGOTIATION", "CONFIRMATION", "PRESENTATION", "PAYMENT", "GUARANTEE",
    "COMPLIANCE", "DOCUMENTARY", "ACCEPTABLE", "NEGOTIABLE", "REIMBURSEMENT",
    "CORRESPONDENT", "ENDORSEMENT", "DISCREPANCY", "TRANSSHIPMENT",
    "MERCHANDISE", "FINANCING", "VALIDITY", "COMMERCIAL", "ADDITIONAL",
    "CONDITIONS", "AVAILABLE", "PARTIAL", "ALLOWED", "EXCHANGE",
    "CURRENCY", "TOLERANCE", "QUANTITY", "INTEREST", "STANDBY",
    "FREIGHT", "CUSTOMS", "DECLARATION", "AUTHENTICATION", "ISSUING",
    "CONFIRMING", "NOMINATED", "DEFERRED", "PREPAYMENT", "APPLICABLE",
    "ACCORDANCE", "REFERENCE", "SEQUENCE", "EXPIRY", "MATURITY",
    "PRINCIPAL", "REPRESENTATIVE", "SIGNATORY", "SUPPLIER", "RECIPIENT",
    "INSTRUCTION", "NOTIFICATION", "ISSUANCE", "UNDERTAKING", "OBLIGATION",
    "PRESENTATION", "EXAMINATION", "COMPLIANT", "DISCREPANT", "ACCEPTANCE",
})


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CorrectionEntry:
    original:    str
    corrected:   str
    method:      str   # "static" | "levenshtein" | "ocr_noise" | "bigram" | "contextual"
    position:    int
    confidence:  float = 1.0  # NEW: score de confiance [0.0-1.0]
    validated:   bool  = True  # NEW: validé par le modèle contextuel


@dataclass
class SpellCorrectionReport:
    corrections:      list[CorrectionEntry] = field(default_factory=list)
    total_changes:    int   = 0
    has_ocr_noise:    bool  = False
    rejected_count:   int   = 0   # NEW: corrections rejetées par validation contextuelle
    avg_confidence:   float = 1.0 # NEW: confiance moyenne
    language_detected: str  = "EN" # NEW: "EN" | "FR"

    def to_dict(self) -> dict:
        return {
            "total_changes":    self.total_changes,
            "has_ocr_noise":    self.has_ocr_noise,
            "rejected_count":   self.rejected_count,
            "avg_confidence":   round(self.avg_confidence, 3),
            "language_detected": self.language_detected,
            "corrections": [
                {
                    "original":   c.original,
                    "corrected":  c.corrected,
                    "method":     c.method,
                    "position":   c.position,
                    "confidence": round(c.confidence, 3),
                    "validated":  c.validated,
                }
                for c in self.corrections
            ],
        }


# ══════════════════════════════════════════════════════════════════════
# VALIDATEUR CONTEXTUEL (MLM BERT — optionnel)
# ══════════════════════════════════════════════════════════════════════

class _ContextualValidator:
    """
    Valide les corrections via un modèle BERT fill-mask.
    Si transformers n'est pas installé, passe en mode dégradé (toujours valide).

    Modèle utilisé : bert-base-uncased (ou multilingual pour FR/EN mixte)
    Usage mémoire  : ~400MB GPU / ~1.2GB CPU
    """

    _model_name = "bert-base-uncased"
    _mlm = None
    _available = False

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        if not enabled:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._mlm = hf_pipeline(
                "fill-mask",
                model=self._model_name,
                top_k=20,
                device=-1,   # CPU par défaut ; mettre 0 pour GPU
            )
            self._available = True
            logger.info("ContextualValidator: BERT MLM chargé (%s)", self._model_name)
        except ImportError:
            logger.info("transformers non installé — validation contextuelle désactivée")
        except Exception as exc:
            logger.warning("ContextualValidator: échec chargement BERT: %s", exc)

    def validate(self, context: str, wrong: str, candidate: str) -> tuple[bool, float]:
        """
        Retourne (is_valid, confidence).
        Remplace `wrong` par [MASK] dans le contexte et vérifie si
        `candidate` est parmi les top prédictions.
        """
        if not self._available or not self._mlm:
            return True, 0.85   # mode dégradé : toujours accepter

        # Créer le texte masqué (max 512 tokens pour BERT)
        window = self._extract_window(context, wrong, window=50)
        masked = re.sub(
            r'\b' + re.escape(wrong) + r'\b',
            "[MASK]",
            window,
            count=1,
            flags=re.IGNORECASE
        )
        if "[MASK]" not in masked:
            return True, 0.75

        try:
            predictions = self._mlm(masked.lower())
            pred_tokens = {p["token_str"].strip().upper(): p["score"] for p in predictions}
            cand_upper = candidate.upper()

            # Cherche correspondance exacte ou partielle
            if cand_upper in pred_tokens:
                return True, min(0.99, pred_tokens[cand_upper] + 0.3)

            # Correspondance avec les premiers chars (pour mots longs)
            for token, score in pred_tokens.items():
                if cand_upper.startswith(token[:4]) and len(token) >= 4:
                    return True, min(0.95, score + 0.2)

            # Pas trouvé dans le top-20 → correction suspecte
            return False, 0.20

        except Exception as exc:
            logger.debug("ContextualValidator error: %s", exc)
            return True, 0.70   # en cas d'erreur, accepter

    @staticmethod
    def _extract_window(text: str, word: str, window: int = 50) -> str:
        """Extrait une fenêtre de ±window mots autour du mot à corriger."""
        idx = text.upper().find(word.upper())
        if idx == -1:
            return text[:200]
        start = max(0, idx - window * 5)
        end   = min(len(text), idx + len(word) + window * 5)
        return text[start:end]


# ══════════════════════════════════════════════════════════════════════
# DÉTECTEUR DE LANGUE
# ══════════════════════════════════════════════════════════════════════

class _LanguageDetector:
    """Détection légère FR/EN basée sur mots fréquents (sans dépendance)."""

    _FR_MARKERS = {"DE", "DU", "LA", "LE", "LES", "ET", "EN", "POUR", "SUR", "AVEC",
                   "LETTRE", "CREDIT", "GARANTIE", "MONTANT", "DATE"}
    _EN_MARKERS = {"THE", "OF", "AND", "FOR", "BY", "IN", "AT", "TO", "WITH",
                   "LETTER", "CREDIT", "GUARANTEE", "AMOUNT", "DATE"}

    @classmethod
    def detect(cls, text: str) -> str:
        words = set(text.upper().split())
        fr_score = len(words & cls._FR_MARKERS)
        en_score = len(words & cls._EN_MARKERS)
        return "FR" if fr_score > en_score else "EN"


# ══════════════════════════════════════════════════════════════════════
# CORRECTEUR PRINCIPAL v2
# ══════════════════════════════════════════════════════════════════════

class SwiftSpellCorrector:
    """
    Correcteur orthographique spécialisé SWIFT v2.
    Pipeline en 5 passes :
      1. Bigrammes (corrections multi-mots)
      2. Dictionnaire statique étendu
      3. Distance de Levenshtein sur vocabulaire SWIFT (avec cache LRU)
      4. [OPTIONNEL] Validation contextuelle BERT
      5. Normalisation OCR (O→0, l→1)
    """

    _MAX_EDIT_DIST  = 1
    _MIN_WORD_LEN   = 7
    _TAG_PATTERN    = re.compile(r'(?<!\w):?\d{2}[A-Z]?:')  # accepte '50:' et ':50:' (OCR)
    # SPL-01/02 : tags dont le contenu contient des adresses/noms propres FR/EN
    # → ne jamais appliquer de correction Levenshtein sur ces segments
    _ADDRESS_TAGS   = re.compile(r':?(50|59|52[AD]?|57[AD]?|72[ZD]?|71[BDE]?|56[AC]?):')  # tolérant OCR

    def __init__(
        self,
        custom_dict:        dict[str, str] | None = None,
        enable_contextual:  bool = False,   # activer la validation BERT
        contextual_threshold: float = 0.30, # rejeter si confidence < seuil
    ) -> None:
        self._enable_contextual   = enable_contextual
        self._contextual_threshold = contextual_threshold

        # Initialiser le validateur contextuel
        self._validator = _ContextualValidator(enabled=enable_contextual)

        # Construire le dictionnaire (hardcodé + dataset)
        self._dict = dict(_STATIC_DICT)
        self._bigrams = dict(_BIGRAM_DICT)

        # Enrichir depuis le dataset
        try:
            from services.swift_nlp.common.swift_dataset_loader import SwiftDatasetLoader

            ds = SwiftDatasetLoader.instance().get()
            if ds.loaded and ds.spell_dict:
                merged = dict(self._dict)
                merged.update(ds.spell_dict)
                self._dict = merged
                logger.debug(
                    "SpellCorrector v2: %d paires chargées (dataset + builtin)",
                    len(self._dict)
                )
        except Exception as exc:
            logger.debug("Dataset non disponible pour SpellCorrector: %s", exc)

        # Vocabulaire Levenshtein
        self._vocab = set(_SWIFT_VOCAB)
        try:
            from services.swift_nlp.common.swift_dataset_loader import SwiftDatasetLoader

            ds = SwiftDatasetLoader.instance().get()
            if ds.loaded and ds.swift_vocab:
                self._vocab = self._vocab | ds.swift_vocab
        except Exception:
            pass

        # Vocab trié (pour cache LRU stable)
        self._vocab_tuple = tuple(sorted(self._vocab))

        # Custom dict (priorité max)
        if custom_dict:
            self._dict.update({k.upper(): v.upper() for k, v in custom_dict.items()})

    # ── API publique ──────────────────────────────────────────────────

    def correct(self, text: str) -> tuple[str, SpellCorrectionReport]:
        """
        Corrige le texte en 5 passes.
        Retourne (texte_corrigé, rapport).
        Préserve les tags SWIFT (:20:, :32B:, etc.) intacts.
        """
        report   = SpellCorrectionReport()
        rejected = 0

        # Détection de langue
        report.language_detected = _LanguageDetector.detect(text)

        # OCR sûr (liste blanche) — avant segment / bigrammes
        text = re.sub(r"\bONL\s+Y\b", "ONLY", text, flags=re.IGNORECASE)
        text = re.sub(r"\bU5D(\s*\d)", r"USD\1", text, flags=re.IGNORECASE)

        # Passe 0 : bigrammes (avant tokenisation)
        text, bigram_corrections = self._correct_bigrams(text)
        report.corrections.extend(bigram_corrections)

        # Segmenter en préservant les tags
        segments = self._split_segments(text)
        result   = []

        for seg_type, seg_text, seg_pos in segments:
            if seg_type == "tag":
                result.append(seg_text)
                continue
            # SPL-01/02 : segments adresse/nom propre → pas de correction
            if seg_type == "addr":
                result.append(seg_text)
                continue

            corrected_seg, seg_corrections = self._correct_segment(
                seg_text, base_pos=seg_pos, full_text=text
            )

            # Validation contextuelle (passe 4)
            if self._enable_contextual:
                validated_corrections = []
                for corr in seg_corrections:
                    is_valid, conf = self._validator.validate(
                        text, corr.original, corr.corrected
                    )
                    if is_valid or conf >= self._contextual_threshold:
                        corr.confidence = conf
                        corr.validated  = True
                        validated_corrections.append(corr)
                    else:
                        # Annuler la correction dans le texte corrigé
                        corrected_seg = corrected_seg.replace(corr.corrected, corr.original, 1)
                        corr.validated = False
                        rejected += 1
                        logger.debug(
                            "Correction rejetée par BERT: %s→%s (conf=%.2f)",
                            corr.original, corr.corrected, conf
                        )
                seg_corrections = validated_corrections

            report.corrections.extend(seg_corrections)
            result.append(corrected_seg)

        corrected_text       = "".join(result)
        report.total_changes = len(report.corrections)
        report.rejected_count= rejected
        report.has_ocr_noise = any(c.method == "ocr_noise" for c in report.corrections)

        if report.corrections:
            report.avg_confidence = sum(
                c.confidence for c in report.corrections
            ) / len(report.corrections)

        return corrected_text, report

    def correct_word(self, word: str) -> tuple[str, str | None, float]:
        """
        Corrige un mot seul.
        Retourne (mot_corrigé, méthode, confidence) ou (mot, None, 1.0).
        """
        upper = word.upper()
        if upper in self._dict:
            return self._dict[upper], "static", 0.98
        suggestion = self._levenshtein_suggest_cached(upper)
        if suggestion:
            return suggestion, "levenshtein", 0.82
        return word, None, 1.0

    # ── Passe 0 : Bigrammes ───────────────────────────────────────────

    def _correct_bigrams(
        self, text: str
    ) -> tuple[str, list[CorrectionEntry]]:
        """Corrige les erreurs multi-mots (bigrammes/trigrammes)."""
        corrections = []
        upper = text.upper()

        for wrong_bigram, correct_bigram in self._bigrams.items():
            if wrong_bigram in upper:
                idx = upper.find(wrong_bigram)
                original_fragment = text[idx: idx + len(wrong_bigram)]
                corrected_fragment = self._preserve_case(original_fragment, correct_bigram)
                if original_fragment != corrected_fragment:
                    corrections.append(CorrectionEntry(
                        original=original_fragment,
                        corrected=corrected_fragment,
                        method="bigram",
                        position=idx,
                        confidence=0.95,
                    ))
                    text  = text[:idx] + corrected_fragment + text[idx + len(wrong_bigram):]
                    upper = text.upper()

        return text, corrections

    # ── Segmentation ─────────────────────────────────────────────────

    def _split_segments(self, text: str) -> list[tuple[str, str, int]]:
        """SPL-01/02 : les segments après un tag adresse sont marqués 'addr'
        et ne subissent pas la correction Levenshtein.
        """
        segments = []
        last = 0
        current_is_addr = False
        for m in self._TAG_PATTERN.finditer(text):
            if m.start() > last:
                seg_type = "addr" if current_is_addr else "content"
                segments.append((seg_type, text[last:m.start()], last))
            tag_str = m.group()
            segments.append(("tag", tag_str, m.start()))
            # Marquer si le prochain segment est un champ adresse
            current_is_addr = bool(self._ADDRESS_TAGS.match(tag_str))
            last = m.end()
        if last < len(text):
            seg_type = "addr" if current_is_addr else "content"
            segments.append((seg_type, text[last:], last))
        return segments

    # ── Correction de segment ─────────────────────────────────────────

    def _correct_segment(
        self, text: str, base_pos: int, full_text: str = ""
    ) -> tuple[str, list[CorrectionEntry]]:
        corrections = []
        word_positions = []

        for m in re.finditer(r'[A-Za-z]{%d,}' % self._MIN_WORD_LEN, text):
            word_positions.append((m.start(), m.end(), m.group()))

        corrected = list(text)

        for start, end, word in word_positions:
            upper    = word.upper()
            new_word = None
            method   = None
            conf     = 1.0

            # SPL-03 : jamais corriger les formes valides (pluriels bancaires, termes FR)
            if upper in _BANKING_VALID_FORMS:
                continue

            if upper in self._dict:
                new_word = self._preserve_case(word, self._dict[upper])
                method   = "static"
                conf     = 0.98
            else:
                suggest = self._levenshtein_suggest_cached(upper)
                if suggest:
                    new_word = self._preserve_case(word, suggest)
                    method   = "levenshtein"
                    conf     = 0.82

            if new_word and new_word != word:
                corrections.append(CorrectionEntry(
                    original=word,
                    corrected=new_word,
                    method=method,
                    position=base_pos + start,
                    confidence=conf,
                ))
                corrected[start:end] = list(new_word)

        # Passe OCR
        segment_str     = "".join(corrected)
        ocr_fixed, ocr_corrections = self._fix_ocr_noise(segment_str, base_pos)
        corrections.extend(ocr_corrections)

        return ocr_fixed, corrections

    # ── OCR Noise ─────────────────────────────────────────────────────

    def _fix_ocr_noise(
        self, text: str, base_pos: int
    ) -> tuple[str, list[CorrectionEntry]]:
        corrections = []

        def replace_in_amount(m: re.Match) -> str:
            orig  = m.group()
            fixed = re.sub(r'(?<=[0-9])[Oo](?=[0-9,])', '0', orig)
            fixed = re.sub(r'(?<=[0-9])[lI](?=[0-9,])', '1', fixed)
            if fixed != orig:
                corrections.append(CorrectionEntry(
                    original=orig,
                    corrected=fixed,
                    method="ocr_noise",
                    position=base_pos + m.start(),
                    confidence=0.92,
                ))
            return fixed

        result = re.sub(
            r'[A-Z]{3}[0-9OolI][0-9OolI,\.\']+',
            replace_in_amount,
            text,
        )
        return result, corrections

    # ── Levenshtein avec cache LRU ────────────────────────────────────

    def _levenshtein_suggest_cached(self, word: str) -> str | None:
        """Suggestion Levenshtein avec cache (x5 perf sur batch)."""
        return _levenshtein_suggest_static(word, self._vocab_tuple, self._MAX_EDIT_DIST)

    @staticmethod
    def _preserve_case(original: str, corrected: str) -> str:
        if original.isupper():   return corrected.upper()
        if original.islower():   return corrected.lower()
        if original[0].isupper(): return corrected.capitalize()
        return corrected


# Cache LRU global pour Levenshtein (partagé entre instances)
@lru_cache(maxsize=2048)
def _levenshtein_suggest_static(
    word: str,
    vocab_tuple: tuple,
    max_dist: int,
) -> str | None:
    """Fonction pure cachée pour suggestions Levenshtein."""
    if len(word) < 7:
        return None
    best_word = None
    best_dist = max_dist + 1

    for vocab_word in vocab_tuple:
        # Optimisation : ignorer si la différence de longueur est trop grande
        if abs(len(word) - len(vocab_word)) > max_dist + 1:
            continue
        d = _levenshtein(word, vocab_word)
        if d < best_dist:
            best_dist = d
            best_word = vocab_word

    return best_word if best_dist <= max_dist else None


def _levenshtein(s1: str, s2: str) -> int:
    """Distance de Levenshtein optimisée O(min(m,n))."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    if abs(m - n) > 2:
        return abs(m - n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev  = temp
    return dp[n]