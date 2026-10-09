"""
swift_detector.py  —  v5.2
Détection du type MT*** par :
  1. Header bloc {2:Innn...} → résolution directe
  2. Mention explicite "MT700" dans le texte (si >= 1 tag réel présent)
  3. Score pondéré tags (SWIFT_TAG_WEIGHTS) + mots-clés (SWIFT_KEYWORD_WEIGHTS)
  4. Garde-fou free-text : si < MIN_SWIFT_TAGS tags réels ET pas de mention MT → MT799
  5. Fallback "UNKNOWN"

Corrections v5.2 :
  F1 — Garde-fou revu : si mention MT### explicite ET >= 1 tag réel → ne pas forcer MT799,
       laisser le scoring décider (évite la régression MT760 OCR dégradé)
  F2 — _has_exclusive_mt720_tags : accepte '21 ' et '40B' sans ':' initial (format verbeux)
       + détection par mots-clés MT720 exclusifs ('FIRST BENEFICIARY', 'TRANSFERRING BANK')
"""

from __future__ import annotations

import re
from typing import Dict, Any

from .patterns import (
    SWIFT_TAG_WEIGHTS, SWIFT_KEYWORD_WEIGHTS, MT_CODES,
    MIN_SWIFT_TAGS_FOR_STRUCTURED,
)
from .swift_tag_normalizer import SwiftTagNormalizer

# Instance partagée du normaliseur (construction du regex faite une seule fois)
_normalizer = SwiftTagNormalizer()


class SwiftDetector:
    """
    Détecte le type de message SWIFT avec scoring pondéré.
    """

    def __init__(self, schemas: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.schemas = schemas or {}

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def detect(self, text: str) -> str:
        """
        Retourne le type détecté ('MT700', 'MT720', …) ou 'UNKNOWN'.
        """
        upper = text.upper()

        # ── Étape 1 : Header bloc 2 (priorité absolue) ───────────────
        result = self._from_header(upper)
        if result:
            return result

        real_tag_count  = self._count_real_tags(upper)
        has_explicit_mt = self._has_any_explicit_mention(upper)

        # ── Garde-fou free-text ───────────────────────────────────────
        # Texte libre sans structure SWIFT → MT799
        # SAUF si une mention MT### explicite est présente ET qu'on a
        # au moins 1 tag réel (document SWIFT OCR dégradé)
        if real_tag_count < MIN_SWIFT_TAGS_FOR_STRUCTURED:
            if not (has_explicit_mt and real_tag_count >= 1):
                return "MT799"

        # ── Étape 2 : Mention explicite MT### ────────────────────────
        result = self._from_explicit_mention(upper)
        if result:
            return result

        # ── Étape 3 : Scoring pondéré ────────────────────────────────
        return self._from_scoring(upper)

    def detect_with_scores(self, text: str) -> Dict[str, int]:
        """Retourne les scores bruts pour debug/diagnostic."""
        upper = text.upper()
        return {mt: self._score(upper, mt) for mt in SWIFT_TAG_WEIGHTS}

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _count_real_tags(self, upper: str) -> int:
        """
        Compte le nombre de tags SWIFT réels dans le texte.
        v5.3 : utilise SwiftTagNormalizer pour détecter AUSSI les tags sans ':' initial
        (ex: '27:', '40c:', '310:') que le pattern :NN: ne capture pas.
        """
        # Méthode 1 : tags formels :NN:
        formal = len(re.findall(r':\d{2}[A-Z]?:', upper))
        if formal >= MIN_SWIFT_TAGS_FOR_STRUCTURED:
            return formal
        # Méthode 2 : normalizer (détecte aussi les tags OCR dégradés)
        return _normalizer.count_tags(upper)

    def _from_header(self, upper: str) -> str | None:
        """
        Cherche {2:I700...}, {2:I720...} etc. dans le header SWIFT.
        Tolère les erreurs OCR (O à la place de 0).
        """
        m = re.search(r"\{2\s*:\s*[A-Z]?([0-9O]{3})", upper)
        if m:
            code = m.group(1).replace("O", "0")
            candidate = MT_CODES.get(code)
            if candidate and (not self.schemas or candidate in self.schemas):
                return candidate

        m2 = re.search(r"\bI([0-9O]{3})[A-Z]+\b", upper)
        if m2:
            code = m2.group(1).replace("O", "0")
            candidate = MT_CODES.get(code)
            if candidate and (not self.schemas or candidate in self.schemas):
                return candidate

        return None

    def _from_explicit_mention(self, upper: str) -> str | None:
        """
        Cherche 'MT700', 'MT 700', 'MT7OO' etc. en texte libre.
        Ignoré si le texte est du free-text (déjà filtré par _count_real_tags).
        """
        m = re.search(r"\bMT\s*([0-9O]{3})\b", upper)
        if m:
            code = m.group(1).replace("O", "0")
            candidate = MT_CODES.get(code)
            if candidate and (not self.schemas or candidate in self.schemas):
                return candidate
        return None

    def _from_scoring(self, upper: str) -> str:
        """
        Scoring pondéré : tags (poids fort) + mots-clés (poids faible).

        Règle spéciale MT720 vs MT700 :
        Si les tags exclusifs :21: ou :40B: sont présents → MT720 prioritaire
        même si le score global MT700 est plus élevé (tags communs).
        """
        scores: dict[str, int] = {}
        supported = list(self.schemas.keys()) if self.schemas else list(SWIFT_TAG_WEIGHTS.keys())

        for mt in supported:
            scores[mt] = self._score(upper, mt)

        # Règle spéciale : tags exclusifs MT720
        if self._has_exclusive_mt720_tags(upper):
            # Forcer MT720 si son score est raisonnable (>= 5)
            if scores.get("MT720", 0) >= 5:
                return "MT720"

        best_mt    = max(scores, key=lambda k: scores[k]) if scores else "UNKNOWN"
        best_score = scores.get(best_mt, 0)

        if best_score < 3:
            return "UNKNOWN"

        # Ambiguïté : deux types à moins de 2 points → UNKNOWN
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < 2:
            exclusive = {
                mt for mt, tag_w in SWIFT_TAG_WEIGHTS.items()
                for tag, w in tag_w.items()
                if w >= 5 and tag in upper and mt == best_mt
            }
            if not exclusive:
                return "UNKNOWN"

        return best_mt

    def _has_any_explicit_mention(self, upper: str) -> bool:
        """
        Vérifie si le texte contient une mention MT### explicite (MT760, MT700…).
        Utilisé par le garde-fou pour ne pas bloquer les OCR dégradés.
        """
        return bool(re.search(r'\bMT\s*[0-9O]{3}\b', upper))

    def _has_exclusive_mt720_tags(self, upper: str) -> bool:
        """
        Vérifie la présence des tags/mots-clés exclusifs MT720.
        v5.2 : accepte aussi les formats verbeux sans ':' initial,
        et les mots-clés textuels ('FIRST BENEFICIARY', 'TRANSFERRING BANK').
        """
        # Tags formels avec ':'
        has_21  = bool(re.search(r':21\s*:', upper))
        has_40b = bool(re.search(r':40B\s*:', upper))
        # Tags OCR sans ':' initial (format verbeux : "21 Documentary Credit Number")
        has_21  = has_21  or bool(re.search(r'(?:^|\n|\s)21\s+(?:DOCUMENTARY|CREDIT)', upper))
        has_40b = has_40b or bool(re.search(r'(?:^|\n|\s)40B\s+(?:FORM|IRREVOCABLE)', upper))
        # Mots-clés textuels exclusifs MT720
        has_kw = (
            'FIRST BENEFICIARY' in upper
            or 'SECOND BENEFICIARY' in upper
            or 'TRANSFERRING BANK' in upper
            or 'TRANSFER OF A DOCUMENTARY CREDIT' in upper
            or 'TRANSFER OF DOCUMENTARY CREDIT' in upper
        )
        return has_21 or has_40b or has_kw

    def _score(self, upper: str, mt: str) -> int:
        score = 0
        for tag, weight in SWIFT_TAG_WEIGHTS.get(mt, {}).items():
            pattern = re.escape(tag).replace(r"\:", r"\s*:\s*")
            if re.search(pattern, upper):
                score += weight
        for kw, weight in SWIFT_KEYWORD_WEIGHTS.get(mt, {}).items():
            if kw in upper:
                score += weight
        return score