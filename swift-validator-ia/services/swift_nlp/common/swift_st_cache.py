"""
Cache process-wide des modèles SentenceTransformer (évite double chargement
UCP aligner + classifieur de clauses lorsque le même `model_name` est utilisé).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ST_MODELS: dict[str, Any] = {}


def get_sentence_transformer(model_name: str) -> Optional[Any]:
    """
    Retourne une instance partagée de SentenceTransformer pour `model_name`,
    ou None si import / chargement échoue.
    """
    if not model_name:
        return None
    if model_name in _ST_MODELS:
        return _ST_MODELS[model_name]
    try:
        try:
            # Registre central : révision épinglée + device CPU forcé (parité prod/local)
            from config.model_registry import load_sentence_transformer
            enc = load_sentence_transformer(model_name)
        except ImportError:
            from sentence_transformers import SentenceTransformer
            enc = SentenceTransformer(model_name, device="cpu")
        _ST_MODELS[model_name] = enc
        logger.info("SentenceTransformer cache: chargé '%s'", model_name)
        return enc
    except ImportError:
        logger.info("sentence-transformers non installé")
        return None
    except Exception as exc:
        logger.warning("SentenceTransformer cache: échec '%s': %s", model_name, exc)
        return None


def clear_sentence_transformer_cache() -> None:
    """Utile pour les tests."""
    _ST_MODELS.clear()
