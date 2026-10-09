"""
Registre central des modèles — source unique de vérité (MLOps).

Garantit la parité local ↔ production :
  1. RÉVISION ÉPINGLÉE : chaque modèle HuggingFace Hub est figé sur un commit
     exact. Local (online) et prod (HF_HUB_OFFLINE=1, cache pré-rempli)
     chargent bit-à-bit les mêmes poids, même si le repo Hub évolue.
  2. DEVICE FORCÉ : l'inférence tourne sur CPU des deux côtés. Sans cela,
     sentence-transformers choisit CUDA quand un GPU existe (cas du poste
     local) → kernels flottants différents → scores sémantiques différents.
  3. AUDIT : versions() expose libs + modèles + device pour le /health,
     ce qui permet de vérifier la parité d'un simple curl.

Toute nouvelle dépendance à un modèle Hub DOIT passer par ce module
(load_sentence_transformer / load_cross_encoder / pinned_revision).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Device d'inférence unique (parité local/prod) ──────────────────────
# "cpu" par défaut : la prod n'a pas de GPU ; le GPU local (MX130) est de
# toute façon trop faible pour être un gain, mais changerait les numériques.
INFERENCE_DEVICE: str = os.environ.get("SWIFT_INFERENCE_DEVICE", "cpu")

# ── Révisions épinglées (commit hash HuggingFace Hub) ───────────────────
# Snapshots présents dans .hf_cache (prod, offline) ET dans le cache local.
PINNED_HUB_REVISIONS: dict[str, str] = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
    "sentence-transformers/all-MiniLM-L6-v2":
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "cross-encoder/nli-deberta-v3-small":
        "fa2804872c3b4bd748f38c0185cc85775361e735",
    "cross-encoder/nli-deberta-v3-xsmall":
        "a150876415327c80daeff35ca6f68f5ed8cf5c24",
    "dslim/bert-base-NER":
        "d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc",
}


def _canonical_hub_name(name: str) -> str:
    """'paraphrase-multilingual-MiniLM-L12-v2' → nom complet 'org/nom'."""
    if "/" in name:
        return name
    for full in PINNED_HUB_REVISIONS:
        if full.split("/", 1)[1] == name:
            return full
    return name


def _is_local_path(name: str) -> bool:
    # Un nom Hub est "org/nom" : jamais de backslash, n'existe pas sur disque.
    return "\\" in name or Path(name).exists()


def pinned_revision(model_name: str) -> Optional[str]:
    """Révision épinglée pour un nom Hub (None si chemin local / non épinglé)."""
    if _is_local_path(model_name):
        return None
    rev = PINNED_HUB_REVISIONS.get(_canonical_hub_name(model_name))
    if rev is None:
        logger.warning(
            "model_registry: '%s' n'a PAS de révision épinglée — "
            "ajoutez-la à PINNED_HUB_REVISIONS pour garantir la parité.",
            model_name,
        )
    return rev


def load_sentence_transformer(model_name: str) -> Any:
    """SentenceTransformer avec révision épinglée + device forcé."""
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"device": INFERENCE_DEVICE}
    rev = pinned_revision(model_name)
    if rev:
        kwargs["revision"] = rev
    model = SentenceTransformer(model_name, **kwargs)
    logger.info(
        "model_registry: SentenceTransformer '%s' (rev=%s, device=%s)",
        model_name, rev or "n/a", INFERENCE_DEVICE,
    )
    return model


def load_cross_encoder(model_name: str, **extra: Any) -> Any:
    """CrossEncoder avec révision épinglée + device forcé."""
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {"device": INFERENCE_DEVICE, **extra}
    rev = pinned_revision(model_name)
    if rev:
        kwargs["revision"] = rev
    model = CrossEncoder(model_name, **kwargs)
    logger.info(
        "model_registry: CrossEncoder '%s' (rev=%s, device=%s)",
        model_name, rev or "n/a", INFERENCE_DEVICE,
    )
    return model


def hf_pipeline_kwargs(model_name: str) -> dict[str, Any]:
    """kwargs à passer à transformers.pipeline() : device CPU + révision."""
    kwargs: dict[str, Any] = {"device": -1 if INFERENCE_DEVICE == "cpu" else 0}
    rev = pinned_revision(model_name)
    if rev:
        kwargs["revision"] = rev
    return kwargs


def versions() -> dict[str, Any]:
    """Snapshot libs + modèles + device — exposé par /health pour audit parité."""
    libs: dict[str, str] = {}
    for lib in ("torch", "transformers", "sentence_transformers", "numpy", "sklearn"):
        try:
            mod = __import__(lib)
            libs[lib] = getattr(mod, "__version__", "?")
        except Exception:
            libs[lib] = "absent"
    cuda_visible = None
    try:
        import torch
        cuda_visible = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "inference_device": INFERENCE_DEVICE,
        "cuda_available_but_unused": cuda_visible if INFERENCE_DEVICE == "cpu" else False,
        "libs": libs,
        "hub_models_pinned": dict(PINNED_HUB_REVISIONS),
    }
