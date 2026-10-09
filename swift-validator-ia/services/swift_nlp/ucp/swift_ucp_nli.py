"""
NLI optionnelle (CrossEncoder) pour lever l'ambiguïté clause ↔ article UCP
après alignement par embeddings. Activé avec UCP_USE_NLI=1.

Modèle par défaut : cross-encoder/nli-deberta-v3-xsmall (léger, anglais ;
textes LC souvent en anglais). Surcharge : UCP_NLI_MODEL.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class UCPNLIRefiner:
    """
    Trois hypothèses NLI (conflit / alignement / restriction) ; on retient
    la plus forte probabilité d'entailment (index 1 des logits NLI standard).
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        # Toujours ON (activation en dur) — le modèle reste optionnel:
        # si indisponible, available=False et le moteur sémantique continue en fallback.
        self._enabled = True
        self._model_name = model_name or os.environ.get(
            "UCP_NLI_MODEL", _DEFAULT_NLI_MODEL
        )
        self._model: Any = None
        self._load_failed = False

        logger.info("UCPNLIRefiner: activé (model=%s)", self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _get_model(self) -> Optional[Any]:
        if not self._enabled or self._load_failed:
            return None
        if self._model is not None:
            return self._model
        try:
            try:
                # Registre central : révision épinglée + device CPU (parité prod/local)
                from config.model_registry import load_cross_encoder
                self._model = load_cross_encoder(self._model_name)
            except ImportError:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name, device="cpu")
            logger.info("UCPNLIRefiner: CrossEncoder '%s' chargé", self._model_name)
            return self._model
        except Exception as exc:
            self._load_failed = True
            logger.warning("UCPNLIRefiner: chargement impossible — %s", exc)
            return None

    @property
    def available(self) -> bool:
        return self._get_model() is not None

    def refine(
        self,
        premise: str,
        article: str,
        title: str,
    ) -> Optional[tuple[str, float]]:
        """
        Retourne (VIOLATION|RESTRICTION|COMPLIANT, confiance) ou None si pas de signal.
        """
        model = self._get_model()
        if model is None:
            return None

        import numpy as np

        p = (premise or "").strip()
        if len(p) < 12:
            return None
        p = p[:480]
        ttl = (title or article or "UCP topic").strip()[:120]

        h_viol = (
            f"The documentary credit clause conflicts with or opts out of "
            f"standard UCP 600 requirements regarding {ttl} ({article})."
        )
        h_ok = (
            f"The documentary credit clause is compatible with usual UCP 600 "
            f"practice regarding {ttl} ({article})."
        )
        h_rest = (
            f"The documentary credit clause imposes stricter obligations than "
            f"the typical UCP 600 minimum regarding {ttl} ({article})."
        )
        pairs = [(p, h_viol), (p, h_ok), (p, h_rest)]
        try:
            logits = model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            logger.warning("UCPNLIRefiner.predict: %s", exc)
            return None

        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] < 3:
            return None

        def _entailment_prob(row: np.ndarray) -> float:
            x = row - np.max(row)
            e = np.exp(x)
            sm = e / (np.sum(e) + 1e-12)
            return float(sm[1])

        ents = [_entailment_prob(logits[i]) for i in range(3)]
        best_i = int(np.argmax(ents))
        best = ents[best_i]
        second = float(np.partition(np.asarray(ents), -2)[-2])
        margin = 0.07
        if best < 0.33:
            return None
        if best < second + margin:
            return None

        if best_i == 0:
            return "VIOLATION", min(0.94, 0.52 + best * 0.42)
        if best_i == 2:
            return "RESTRICTION", min(0.88, 0.48 + best * 0.38)
        return "COMPLIANT", min(0.9, 0.5 + best * 0.38)
