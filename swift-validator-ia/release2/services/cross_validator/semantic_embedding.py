from __future__ import annotations

import logging
import math
import re
from typing import Any, Iterable, List, Optional

log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    _cosine_similarity = None


class SemanticEmbedding:
    def __init__(self, model_name: str | None = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self._use_sentence_transformers = False

        if not model_name:
            log.info("SemanticEmbedding: disabled")
            return

        if SentenceTransformer is not None:
            try:
                try:
                    # Registre central : révision épinglée + CPU forcé (parité prod/local)
                    from config.model_registry import load_sentence_transformer
                    self.model = load_sentence_transformer(model_name)
                except ImportError:
                    self.model = SentenceTransformer(model_name, device="cpu")
                self._use_sentence_transformers = True
                log.info("SemanticEmbedding: loaded SentenceTransformer %s", model_name)
            except Exception as exc:
                log.warning("SemanticEmbedding: impossible de charger %s : %s", model_name, exc)
                self.model = None

        if self.model is None and TfidfVectorizer is not None:
            self.vectorizer = TfidfVectorizer(strip_accents="unicode", stop_words="english")
            log.info("SemanticEmbedding: fallback TF-IDF vectorizer prêt")

    def available(self) -> bool:
        return bool(self.model or self.vectorizer)

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        texts_list = [str(t) for t in texts]
        if self.model is not None:
            return self.model.encode(texts_list, convert_to_numpy=True).tolist()
        if self.vectorizer is not None:
            return self.vectorizer.fit_transform(texts_list).toarray().tolist()
        log.warning("SemanticEmbedding: aucun encodeur disponible, renvoie des vecteurs nuls")
        return [[0.0] * 32 for _ in texts_list]

    def similarity(self, left: str, right: str) -> float:
        if left is None or right is None:
            return 0.0
        if self.model is not None:
            embeddings = self.encode([left, right])
            return self._cosine(embeddings[0], embeddings[1])
        if self.vectorizer is not None:
            vectors = self.vectorizer.fit_transform([left, right]).toarray()
            return self._cosine(vectors[0], vectors[1])
        return self._sequence_ratio(left, right)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def _sequence_ratio(self, left: str, right: str) -> float:
        try:
            from difflib import SequenceMatcher
        except ImportError:
            return 0.0
        return SequenceMatcher(None, self._normalize(left), self._normalize(right)).ratio()

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", " ", text or "").strip().lower()
