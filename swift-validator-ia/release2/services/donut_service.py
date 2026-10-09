"""
donut_service.py
================
Service d'inférence Donut (Document Understanding Transformer).

Gère la classification ET l'extraction de champs en un seul appel image → JSON.
Utilisé pour PDFs scannés et images (pas de texte vectoriel disponible).

Modèle : naver-clova-ix/donut-base fine-tuné sur documents LC
         → models/donut-finetuned/best/

Usage :
  from release2.services.donut_service import DonutService
  svc = DonutService()
  result = svc.predict_image(pil_image)
  result = svc.predict_file(path, input_type="pdf_scanned")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


def _parse_token_sequence(seq: str) -> dict:
    """Parse une séquence de tokens Donut → dict. Remplace token2json instable."""
    result = {}
    matches = re.findall(r"<([^/][^>]*)>(.*?)</\1>", seq, re.DOTALL)
    for key, value in matches:
        if key != "s_lc_parse":
            result[key] = value.strip()
    return result

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL  = RELEASE2_ROOT / "models" / "donut-finetuned" / "best"

DOC_TYPES    = ["bl", "invoice", "packing_list"]
TASK_PROMPT  = "<s_lc_parse>"
RENDER_DPI   = 150


class DonutService:
    """
    Classification + extraction de champs en un seul appel via Donut.

    Donut est un modèle end-to-end (Swin encoder + BART decoder) qui prend
    une image en entrée et génère directement du JSON — sans OCR.

    Paramètres
    ----------
    model_dir : répertoire du checkpoint fine-tuné
                (défaut : models/donut-finetuned/best/)
    """

    def __init__(self, model_dir: Path | str | None = None):
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        import torch

        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Modèle Donut introuvable : {model_dir}\n"
                "Exécuter d'abord :\n"
                "  python scripts/06_export_donut_dataset.py\n"
                "  python scripts/07_finetune_donut.py"
            )

        log.info("Chargement DonutService depuis %s ...", model_dir)
        # CPU forcé : parité numérique local (GPU dispo) ↔ prod (CPU only)
        self.device    = "cpu"
        self.processor = DonutProcessor.from_pretrained(str(model_dir))
        self.model     = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        log.info("DonutService prêt | device : %s", self.device)

    # ──────────────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────────────

    def predict_image(self, image: Image.Image) -> dict:
        """
        Classification + extraction depuis une image PIL.

        Retour
        ------
        {
          "doc_type"    : "bl" | "invoice" | "packing_list",
          "confidence"  : float (0-1),
          "class_probas": {"bl": float, "invoice": float, "packing_list": float},
          "fields"      : {champ: valeur, ...},
          "n_pages"     : 1,
          "pages"       : [{"page": 1, "fields": {...}}],
        }
        """
        import torch

        pixel_values = self.processor(
            image.convert("RGB"),
            return_tensors="pt",
        ).pixel_values.to(self.device)

        task_ids = self.processor.tokenizer(
            TASK_PROMPT,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=task_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                no_repeat_ngram_size=3,
                early_stopping=True,
                return_dict_in_generate=True,
            )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "")

        parsed = _parse_token_sequence(sequence)

        return self._build_result(parsed)

    def predict_file(self, path: Path, input_type: str) -> dict:
        """
        Rasterise le fichier (PDF scanné ou image) et appelle predict_image.

        Paramètres
        ----------
        path       : chemin vers le fichier
        input_type : "pdf_scanned" | "image"
        """
        image = self._load_image(path, input_type)
        result = self.predict_image(image)
        # Pour les PDFs multi-pages, on traite seulement la première page
        # (Donut est conçu pour des documents single-page ; étendre si besoin)
        result["n_pages"] = 1
        result["pages"]   = [{"page": 1, "fields": result["fields"]}]
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────────────────

    def _load_image(self, path: Path, input_type: str) -> Image.Image:
        """Charge le fichier en image PIL RGB."""
        if input_type == "image":
            return Image.open(path).convert("RGB")

        # pdf_scanned : rasteriser la première page
        import fitz
        import numpy as np

        doc = fitz.open(str(path))
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = doc[0].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        doc.close()
        return Image.fromarray(arr)

    def _build_result(self, parsed: dict) -> dict:
        """Convertit le JSON parsé Donut en structure de résultat pipeline."""
        doc_type = str(parsed.pop("doc_type", "unknown")).lower().strip()

        # Donut est génératif — on attribue une confiance élevée si le
        # doc_type est reconnu, faible sinon
        if doc_type in DOC_TYPES:
            confidence = 0.95
            probas = {
                t: (0.95 if t == doc_type else round(0.05 / (len(DOC_TYPES) - 1), 3))
                for t in DOC_TYPES
            }
        else:
            confidence = 0.30
            doc_type   = "unknown"
            probas     = {t: round(1 / len(DOC_TYPES), 3) for t in DOC_TYPES}

        fields = {
            k: str(v).strip()
            for k, v in parsed.items()
            if v and str(v).strip()
        }

        return {
            "doc_type"    : doc_type,
            "confidence"  : confidence,
            "class_probas": probas,
            "fields"      : fields,
            "n_pages"     : 1,
            "pages"       : [{"page": 1, "fields": fields}],
        }
