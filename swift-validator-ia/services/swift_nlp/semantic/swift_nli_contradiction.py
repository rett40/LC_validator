"""
swift_nli_contradiction.py
==========================
Offline NLI helper to classify (premise, hypothesis) into
ENTAILMENT / CONTRADICTION / NEUTRAL.

This module is intentionally optional:
- It only activates when a local model directory is present.
- It never downloads weights at runtime (offline-friendly).

Expected model format: HuggingFace sequence classification model directory
containing config.json + tokenizer files + weights (pytorch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float

    def best_label(self) -> str:
        m = max(self.entailment, self.contradiction, self.neutral)
        if m == self.entailment:
            return "entailment"
        if m == self.contradiction:
            return "contradiction"
        return "neutral"


class OfflineNLIPredictor:
    """
    Minimal offline NLI predictor using transformers.

    It requires a local model directory and will use local_files_only=True.
    """

    def __init__(self, model_dir: str | Path) -> None:
        self._model_dir = Path(model_dir)
        self._available = False
        self._load_failed = False
        self._tokenizer = None
        self._model = None
        self._cross_encoder = None
        self._label_map: dict[str, int] = {}

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @staticmethod
    def _resolve_nli_label_map(id2label: dict, num_labels: int) -> dict[str, int]:
        # Normalize label names; support common NLI conventions.
        label_to_id: dict[str, int] = {}
        for i, raw in (id2label or {}).items():
            try:
                idx = int(i)
            except Exception:
                continue
            name = str(raw).strip().lower()
            label_to_id[name] = idx

        def _pick(*names: str) -> Optional[int]:
            for n in names:
                if n in label_to_id:
                    return label_to_id[n]
            return None

        ent = _pick("entailment", "entailed")
        con = _pick("contradiction", "contradict")
        neu = _pick("neutral")

        # Fallback: many MNLI models use {0: contradiction, 1: neutral, 2: entailment}
        if ent is None and con is None and neu is None and num_labels == 3:
            con, neu, ent = 0, 1, 2

        if ent is None or con is None or neu is None:
            raise ValueError(f"Unsupported NLI label mapping: {id2label}")

        return {"entailment": ent, "contradiction": con, "neutral": neu}

    def _ensure_loaded(self) -> bool:
        if self._available:
            return True
        if self._load_failed:
            return False
        if not self._model_dir.exists():
            self._load_failed = True
            return False

        try:
            # Force offline behavior even if user has HF cache configured.
            import os
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

            # 1) Try standard transformers sequence-classification model
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self._model_dir),
                    local_files_only=True,
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    str(self._model_dir),
                    local_files_only=True,
                )
                self._model.eval()

                cfg = getattr(self._model, "config", None)
                id2label = getattr(cfg, "id2label", None) or {}
                num_labels = int(getattr(cfg, "num_labels", 0) or 0)
                self._label_map = self._resolve_nli_label_map(id2label, num_labels)

                self._available = True
                logger.info("OfflineNLIPredictor loaded (transformers) from %s", self._model_dir)
                return True
            except Exception as exc:
                logger.info("OfflineNLIPredictor transformers load failed: %s", exc)

            # 2) Try sentence-transformers CrossEncoder (common for 'cross-encoder/nli-*')
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(str(self._model_dir), device="cpu")
                self._available = True
                logger.info("OfflineNLIPredictor loaded (cross-encoder) from %s", self._model_dir)
                return True
            except Exception as exc:
                logger.info("OfflineNLIPredictor cross-encoder load failed: %s", exc)

            raise RuntimeError("No compatible offline NLI backend found")

        except Exception as exc:
            self._load_failed = True
            logger.warning("OfflineNLIPredictor unavailable: %s", exc)
            return False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    def predict(self, premise: str, hypothesis: str) -> Optional[NLIScores]:
        if not premise or not hypothesis:
            return None
        if not self._ensure_loaded():
            return None

        try:
            # transformers backend
            if self._model is not None and self._tokenizer is not None and self._label_map:
                import torch

                inputs = self._tokenizer(
                    premise[:700],
                    hypothesis[:220],
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out = self._model(**inputs)
                    logits = out.logits[0]
                    probs = torch.softmax(logits, dim=-1)

                ent = float(probs[self._label_map["entailment"]].item())
                con = float(probs[self._label_map["contradiction"]].item())
                neu = float(probs[self._label_map["neutral"]].item())
                return NLIScores(entailment=ent, contradiction=con, neutral=neu)

            # cross-encoder backend (returns logits for [contradiction, entailment, neutral] commonly)
            if self._cross_encoder is not None:
                import numpy as np

                logits = self._cross_encoder.predict([(premise[:700], hypothesis[:220])], show_progress_bar=False)
                arr = np.asarray(logits, dtype=np.float64).reshape(-1)
                if arr.size < 3:
                    return None
                x = arr - np.max(arr)
                e = np.exp(x)
                sm = e / (np.sum(e) + 1e-12)
                # Best-effort mapping for standard NLI: [contradiction, entailment, neutral]
                con = float(sm[0])
                ent = float(sm[1])
                neu = float(sm[2])
                return NLIScores(entailment=ent, contradiction=con, neutral=neu)

            return None
        except Exception as exc:
            logger.debug("OfflineNLIPredictor.predict failed: %s", exc)
            return None

