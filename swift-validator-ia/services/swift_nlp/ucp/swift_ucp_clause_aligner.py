"""
swift_ucp_clause_aligner.py
===========================
Alignement clause ↔ profils d'articles UCP (embeddings + similarité cosinus).
Réutilisable par le validateur UCP sémantique ; pas de dépendance à swift_ucp_validator.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_PROFILES_PATH = Path(__file__).resolve().parent / "data" / "ucp_article_profiles.json"


def normalize_clause(text: str) -> str:
    """Lowercase, espaces, bruit OCR fréquent."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    t = re.sub(r"[`|]{2,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class UCPAlignerConfig:
    """Seuils calibrables (cosinus sur embeddings normalisés, typ. [-1,1] mais ST souvent >0)."""

    sim_min_align: float = 0.28
    sim_restriction: float = 0.38
    sim_strong: float = 0.48
    model_name: str = DEFAULT_MODEL_NAME


@dataclass
class ArticleProfile:
    article: str
    title: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    category: str = "document"
    embedding_text: str = ""
    clause_type: str = "document_format"   # financial | transport | authentication | identity | compliance | document_format
    impact_level: str = "MEDIUM"           # HIGH | MEDIUM | LOW

    def text_for_embedding(self) -> str:
        base = self.embedding_text.strip() or self.summary
        if self.keywords:
            base = f"{base}. Keywords: {', '.join(self.keywords)}"
        return base


@dataclass
class ClauseAlignment:
    """Résultat d'alignement pour une clause."""

    clause_index: int
    raw_text: str
    normalized: str
    best_article: str
    best_title: str
    similarity: float
    category: str
    clause_type: str = "document_format"
    impact_level: str = "MEDIUM"
    top_k: list[tuple[str, float]] = field(default_factory=list)


def load_article_profiles(path: Optional[Path] = None) -> list[ArticleProfile]:
    p = path or _PROFILES_PATH
    if not p.exists():
        logger.warning("UCP profiles missing: %s", p)
        return []
    with open(p, encoding="utf-8") as f:
        rows = json.load(f)
    out: list[ArticleProfile] = []
    for row in rows:
        out.append(
            ArticleProfile(
                article=row["article"],
                title=row.get("title", row["article"]),
                summary=row.get("summary", ""),
                keywords=list(row.get("keywords", [])),
                category=row.get("category", "document"),
                embedding_text=row.get("embedding_text") or "",
                clause_type=row.get("clause_type", "document_format"),
                impact_level=row.get("impact_level", "MEDIUM"),
            )
        )
    return out


class UCPClauseAligner:
    """
    Encode les profils UCP une fois ; aligne chaque clause par cosinus.
    """

    def __init__(
        self,
        config: Optional[UCPAlignerConfig] = None,
        profiles: Optional[list[ArticleProfile]] = None,
        encoder: Any = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.config = config or UCPAlignerConfig()
        if model_name:
            self.config.model_name = model_name
        self._profiles = profiles or load_article_profiles()
        self._encoder = encoder
        self._profile_matrix: Optional[np.ndarray] = None
        self._available = False

    def _ensure_encoder(self) -> bool:
        if self._encoder is not None:
            return True
        try:
            from ..common.swift_st_cache import get_sentence_transformer

            self._encoder = get_sentence_transformer(self.config.model_name)
            return self._encoder is not None
        except Exception as exc:
            logger.warning("SentenceTransformer indisponible — alignement UCP désactivé: %s", exc)
            return False

    def _ensure_profile_embeddings(self) -> bool:
        if self._profile_matrix is not None:
            return True
        if not self._profiles:
            return False
        if not self._ensure_encoder():
            return False
        texts = [pr.text_for_embedding() for pr in self._profiles]
        try:
            emb = self._encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            emb = np.asarray(emb, dtype=np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            self._profile_matrix = emb / norms
            self._available = True
            return True
        except Exception as exc:
            logger.warning("Erreur encodage profils UCP: %s", exc)
            return False

    @property
    def available(self) -> bool:
        return self._available and self._profile_matrix is not None

    def align_clause(
        self,
        text: str,
        clause_index: int = 0,
        top_k: int = 3,
    ) -> Optional[ClauseAlignment]:
        if not self._ensure_profile_embeddings() or self._profile_matrix is None:
            return None
        norm = normalize_clause(text)
        if len(norm) < 8:
            return None
        try:
            q = self._encoder.encode(
                [norm],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            q = np.asarray(q, dtype=np.float32)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            sims = (self._profile_matrix @ q.T).flatten()
        except Exception as exc:
            logger.warning("Erreur alignement clause: %s", exc)
            return None

        order = np.argsort(-sims)
        ranked: list[tuple[str, float]] = []
        for idx in order[:top_k]:
            pr = self._profiles[int(idx)]
            ranked.append((pr.article, float(sims[int(idx)])))

        best_i = int(order[0])
        best_pr = self._profiles[best_i]
        best_sim = float(sims[best_i])
        return ClauseAlignment(
            clause_index=clause_index,
            raw_text=text,
            normalized=norm,
            best_article=best_pr.article,
            best_title=best_pr.title,
            similarity=best_sim,
            category=best_pr.category,
            clause_type=best_pr.clause_type,
            impact_level=best_pr.impact_level,
            top_k=ranked,
        )

    def align_all(self, clauses: list[dict], top_k: int = 3) -> list[ClauseAlignment]:
        out: list[ClauseAlignment] = []
        for i, c in enumerate(clauses):
            t = (c.get("text") or "").strip()
            if not t:
                continue
            al = self.align_clause(t, clause_index=i, top_k=top_k)
            if al:
                out.append(al)
        return out
