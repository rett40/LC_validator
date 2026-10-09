"""
extraction_service.py
=====================
Service d'inférence LayoutLMv3 — extraction de champs documentaires.

Entrée  : PDF ou image (BL, Invoice, Packing List)
Sortie  : dict {champ: valeur} extrait par le modèle fine-tuné

Usage (standalone) :
  python services/extraction_service.py --pdf path/to/doc.pdf --doc-type bl

Usage (import) :
  from release2.services.extraction_service import ExtractionService
  svc = ExtractionService()
  result = svc.extract_from_pdf("doc.pdf", doc_type="bl")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import fitz                  # PyMuPDF
import cv2
from PIL import Image
from sklearn.cluster import DBSCAN

try:
    from .feature_builder  import FeatureBuilder
    from .post_processor   import PostProcessor
    from .layout_analyzer  import LayoutAnalyzer
    from .native_extractor import _PATTERNS, _apply, _kv_fallback, _spacy_enrich
except ImportError:
    from feature_builder  import FeatureBuilder
    from post_processor   import PostProcessor  # standalone script fallback
    from layout_analyzer  import LayoutAnalyzer
    from native_extractor import _PATTERNS, _apply, _kv_fallback, _spacy_enrich

# Singleton partagé — instancié une seule fois au chargement du module
_layout_analyzer = LayoutAnalyzer()

log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = RELEASE2_ROOT / "data"
MODELS_DIR    = RELEASE2_ROOT / "models"
_FINETUNED_DIR = MODELS_DIR / "layoutlmv3-finetuned"
# Support both flat layout (best/) and nested layout (layoutlmv3-finetuned/best/)
DEFAULT_MODEL = (
    _FINETUNED_DIR / "layoutlmv3-finetuned" / "best"
    if (_FINETUNED_DIR / "layoutlmv3-finetuned" / "best").exists()
    else _FINETUNED_DIR / "best"
)

RENDER_DPI  = 150
BBOX_SCALE  = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Extraction OCR (réutilise la logique du script 01)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_page_tokens(page: fitz.Page, img_w: int, img_h: int) -> list[dict]:
    rect   = page.rect
    pw, ph = rect.width, rect.height
    tokens = []
    for word_tuple in page.get_text("words"):
        x0, y0, x1, y1, text = word_tuple[:5]
        text = text.strip()
        if not text:
            continue
        sx, sy = img_w / pw, img_h / ph
        def nx(v): return max(0, min(BBOX_SCALE, int(v * sx / img_w * BBOX_SCALE)))
        def ny(v): return max(0, min(BBOX_SCALE, int(v * sy / img_h * BBOX_SCALE)))
        tokens.append({"text": text, "bbox": [nx(x0), ny(y0), nx(x1), ny(y1)]})
    return tokens


def _render_page(page: fitz.Page) -> tuple[Image.Image, int, int]:
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bgr = cv2.cvtColor(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3),
        cv2.COLOR_RGB2BGR
    )
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), w, h


# ─────────────────────────────────────────────────────────────────────────────
# Post-traitement : tokens + labels → champs extraits
# ─────────────────────────────────────────────────────────────────────────────

def _tokens_to_fields(words: list[str], labels: list[str]) -> dict[str, str]:
    """
    Convertit une séquence de tokens IOB2 en dict {champ: valeur}.
    Concatène les tokens consécutifs du même champ.
    """
    fields: dict[str, list[str]] = defaultdict(list)
    current_field = None

    for word, label in zip(words, labels):
        if label == "O":
            current_field = None
            continue

        prefix, field = label.split("-", 1)

        if prefix == "B":
            current_field = field
            fields[field].append(word)
        elif prefix == "I" and current_field == field:
            fields[field].append(word)
        else:
            current_field = field
            fields[field].append(word)

    return {k: " ".join(v) for k, v in fields.items()}


def _tokens_to_fields_and_tables(
    words: list[str],
    labels: list[str],
    bboxes: list[list[int]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    Convertit tokens et labels en champs plats et structure de tables.
    """
    fields: dict[str, list[str]] = defaultdict(list)
    table_tokens: list[tuple[int, str, str, list[int]]] = []
    current_field: Optional[str] = None

    for word, label, bbox in zip(words, labels, bboxes):
        if label == "O":
            current_field = None
            continue

        prefix, field = label.split("-", 1)

        if field.startswith("T_"):
            table_tokens.append((len(table_tokens), word, field, bbox))
            current_field = None
            continue

        if prefix == "B":
            current_field = field
            fields[field].append(word)
        elif prefix == "I" and current_field == field:
            fields[field].append(word)
        else:
            current_field = field
            fields[field].append(word)

    tables = _reconstruct_table_rows(table_tokens)
    return {k: " ".join(v) for k, v in fields.items()}, tables


def _build_line_text(words: list[str], bboxes: list[list[int]]) -> str:
    if not words:
        return ""

    y_centers = np.array([((bbox[1] + bbox[3]) / 2) for bbox in bboxes])
    order = sorted(range(len(words)), key=lambda i: (y_centers[i], bboxes[i][0]))

    lines: list[str] = []
    current_line: list[str] = []
    current_y = y_centers[order[0]]
    threshold = 20.0

    for i in order:
        if abs(y_centers[i] - current_y) > threshold:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = []
            current_y = y_centers[i]
        current_line.append(words[i])

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def _fill_missing_fields_from_text(
    fields: dict[str, str],
    words: list[str],
    bboxes: list[list[int]],
    doc_type: str,
) -> dict[str, str]:
    if not doc_type or doc_type not in _PATTERNS:
        return fields

    text = _build_line_text(words, bboxes)
    if not text:
        return fields

    missing = set(_PATTERNS[doc_type].keys()) - set(fields.keys())
    if not missing:
        return fields

    text_fields = _apply(text, {k: _PATTERNS[doc_type][k] for k in missing}, doc_type)
    for field, value in text_fields.items():
        if field not in fields and value:
            fields[field] = value

    missing = missing - set(fields.keys())
    if missing:
        kv_fields = _kv_fallback(text, missing)
        for field, value in kv_fields.items():
            if field not in fields and value:
                fields[field] = value

    missing = missing - set(fields.keys())
    if missing:
        spacy_fields = _spacy_enrich(text, missing)
        for field, value in spacy_fields.items():
            if field not in fields and value:
                fields[field] = value

    return fields


def _cluster_1d(values: np.ndarray, eps: float) -> list[int]:
    if values.size == 0:
        return []

    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(values.reshape(-1, 1))
    if labels.min() < 0:
        labels = labels.copy()
        next_label = int(labels.max()) + 1
        for i, label in enumerate(labels):
            if label == -1:
                labels[i] = next_label
                next_label += 1
    return labels.tolist()


# Taille de fenêtre en mots (conservateur : ~330 subword-tokens ≪ 512 limite modèle)
_SLIDING_WINDOW_WORDS = 256
_SLIDING_OVERLAP_WORDS = 64


def _reconstruct_table_rows(
    table_tokens: list[tuple[int, str, str, list[int]]]
) -> list[dict[str, Any]]:
    """Regroupe les tokens de table en lignes selon DBSCAN sur Y et X."""
    if not table_tokens:
        return []

    x_centers = np.array([((bbox[0] + bbox[2]) / 2) for _, _, _, bbox in table_tokens])
    y_centers = np.array([((bbox[1] + bbox[3]) / 2) for _, _, _, bbox in table_tokens])
    widths = np.array([bbox[2] - bbox[0] for _, _, _, bbox in table_tokens])
    heights = np.array([bbox[3] - bbox[1] for _, _, _, bbox in table_tokens])

    row_eps = max(20.0, float(np.median(heights)) * 1.25)
    col_eps = max(30.0, float(np.median(widths)) * 1.25)

    row_labels = _cluster_1d(y_centers, row_eps)
    col_labels = _cluster_1d(x_centers, col_eps)

    rows_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    row_centers: dict[int, list[float]] = defaultdict(list)
    col_centers: dict[int, list[float]] = defaultdict(list)

    for (idx, word, field, bbox), row_label, col_label, x_center, y_center in zip(
        table_tokens, row_labels, col_labels, x_centers, y_centers
    ):
        rows_map[row_label].append({
            "word": word,
            "field": field,
            "bbox": bbox,
            "col_label": col_label,
            "x_center": x_center,
        })
        row_centers[row_label].append(y_center)
        col_centers[col_label].append(x_center)

    sorted_rows = sorted(rows_map.items(), key=lambda item: np.mean(row_centers[item[0]]))
    sorted_cols = sorted(col_centers.items(), key=lambda item: np.mean(item[1]))
    col_rank = {col_label: rank for rank, (col_label, _) in enumerate(sorted_cols)}

    rows: list[dict[str, Any]] = []
    for row_idx, (row_label, cell_tokens) in enumerate(sorted_rows, start=1):
        cells: dict[str, list[str]] = defaultdict(list)
        for token in sorted(cell_tokens, key=lambda t: (col_rank[t["col_label"]], t["x_center"])):
            column_label = token["field"][2:] if token["field"].startswith("T_") else token["field"]
            cells[column_label].append(token["word"])

        rows.append({
            "row_index": row_idx,
            "cells": {col: " ".join(words) for col, words in cells.items()},
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Service principal
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionService:
    """
    Service d'extraction de champs documentaires basé sur LayoutLMv3.

    Charge le modèle fine-tuné une seule fois et expose `extract_from_pdf`.
    """

    def __init__(self, model_dir: Path | str | None = None):
        model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_dir}\n"
                "Exécuter 04_finetune_layoutlm.py d'abord."
            )

        log.info("Chargement modèle LayoutLMv3 depuis %s", model_dir)

        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        import torch

        self.device = torch.device("cpu")
        self.processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)
        self.model     = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

        # Labels
        id2label_raw = json.loads((model_dir / "id2label.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in id2label_raw.items()}

        log.info("Modèle prêt — %d labels | device: %s", len(self.id2label), self.device)
        self._pp = PostProcessor()

    # ── Inférence LayoutLMv3 ──────────────────────────────────────────────────

    def _run_layoutlm_on_chunk(
        self,
        image:       Image.Image,
        chunk_words: list[str],
        chunk_bboxes: list[list[int]],
        global_offset: int,
        all_labels:  list[str],
        all_logits:  list[float],
        n_global:    int,
    ) -> None:
        """
        Exécute LayoutLMv3 sur un chunk de tokens et met à jour all_labels/all_logits.
        Les prédictions ne remplacent une valeur existante que si le logit est supérieur.
        """
        import torch

        if not chunk_words:
            return

        try:
            encoding = self.processor(
                images         = image,
                text           = chunk_words,
                boxes          = chunk_bboxes,
                truncation     = True,
                max_length     = 512,
                padding        = "max_length",
                return_tensors = "pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)

            logits_np   = outputs.logits[0].cpu().numpy()
            predictions = logits_np.argmax(axis=-1)
            max_logit_v = logits_np.max(axis=-1)

            enc_bboxes = encoding["bbox"][0].cpu().numpy().tolist()
            bbox_to_local: dict[tuple, int] = {}
            for local_idx, b in enumerate(chunk_bboxes):
                key = tuple(b)
                if key not in bbox_to_local:
                    bbox_to_local[key] = local_idx

            for enc_bbox, pred_id, max_l in zip(enc_bboxes, predictions, max_logit_v):
                local_idx  = bbox_to_local.get(tuple(enc_bbox))
                if local_idx is None:
                    continue
                global_idx = global_offset + local_idx
                if global_idx >= n_global:
                    continue
                if max_l > all_logits[global_idx]:
                    all_logits[global_idx] = float(max_l)
                    all_labels[global_idx] = self.id2label.get(int(pred_id), "O")

        except Exception as exc:
            log.warning("LayoutLMv3 chunk erreur: %s", exc)

    def _run_layoutlm_sliding(
        self,
        image:  Image.Image,
        words:  list[str],
        bboxes: list[list[int]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Sliding window sur TOUS les tokens — fallback si block_segmenter indisponible.
        """
        n = len(words)
        if n == 0:
            return {}, []

        stride     = _SLIDING_WINDOW_WORDS - _SLIDING_OVERLAP_WORDS
        all_labels = ["O"] * n
        all_logits = [-float("inf")] * n
        start = 0
        while start < n:
            end = min(start + _SLIDING_WINDOW_WORDS, n)
            self._run_layoutlm_on_chunk(
                image, words[start:end], bboxes[start:end],
                start, all_labels, all_logits, n,
            )
            if end >= n:
                break
            start += stride

        return _tokens_to_fields_and_tables(words, all_labels, bboxes)

    def _run_layoutlm_blocks(
        self,
        image:    Image.Image,
        words:    list[str],
        bboxes:   list[list[int]],
        doc_type: str = "unknown",
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """
        Segmente la page en blocs sémantiques (header / parties / transport /
        cargo / totals), puis exécute LayoutLMv3 sur chaque bloc avec :
          - L'image recadrée sur la zone du bloc  → meilleur contexte visuel
          - Les bboxes re-normalisées dans le repère du bloc  → meilleure attention
          - Un budget 512 tokens complet par bloc  → pas de troncation inutile

        Pour chaque token, conserve la prédiction avec le logit maximal
        (cohérence inter-blocs).  Retourne (fields, table_rows).
        """
        try:
            from .block_segmenter import BlockSegmenter
        except ImportError:
            from block_segmenter import BlockSegmenter

        n = len(words)
        if n == 0:
            return {}, []

        blocks = BlockSegmenter().segment(words, bboxes, doc_type)
        if not blocks:
            log.debug("BlockSegmenter: aucun bloc → fallback sliding window")
            return self._run_layoutlm_sliding(image, words, bboxes)

        log.debug(
            "BlockSegmenter: %d blocs détectés: %s",
            len(blocks),
            [f"{b.block_type}({len(b.tokens)}tok)" for b in blocks],
        )

        all_labels = ["O"] * n
        all_logits = [-float("inf")] * n

        # Index (text, bbox_tuple) → indice global pour retrouver chaque token
        word_key_to_idx: dict[tuple, int] = {}
        for i, (w, b) in enumerate(zip(words, bboxes)):
            key = (w, tuple(b))
            if key not in word_key_to_idx:
                word_key_to_idx[key] = i

        img_w, img_h = image.width, image.height

        for block in blocks:
            if not block.tokens:
                continue

            blk_words  = [t["text"] for t in block.tokens]
            blk_bboxes = [t["bbox"]  for t in block.tokens]

            # ── Recadrage de l'image sur le bloc (avec marge de 15 unités) ───
            pad    = 15
            y0_px  = max(0,     int((block.y_start - pad) / 1000 * img_h))
            y1_px  = min(img_h, int((block.y_end   + pad) / 1000 * img_h))
            if y1_px - y0_px < 4:
                continue
            crop_img = image.crop((0, y0_px, img_w, y1_px))
            blk_h_px = y1_px - y0_px

            # ── Re-normalisation des bboxes Y dans le repère du bloc ─────────
            renorm: list[list[int]] = []
            for bbox in blk_bboxes:
                orig_y0_px = bbox[1] / 1000 * img_h
                orig_y1_px = bbox[3] / 1000 * img_h
                new_y0 = max(0, min(1000, int((orig_y0_px - y0_px) / blk_h_px * 1000)))
                new_y1 = max(0, min(1000, int((orig_y1_px - y0_px) / blk_h_px * 1000)))
                renorm.append([bbox[0], new_y0, bbox[2], new_y1])

            # ── Inférence LayoutLMv3 sur ce bloc ─────────────────────────────
            try:
                import torch
                enc = self.processor(
                    images         = crop_img,
                    text           = blk_words,
                    boxes          = renorm,
                    truncation     = True,
                    max_length     = 512,
                    padding        = "max_length",
                    return_tensors = "pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self.model(**enc)

                logits_np   = out.logits[0].cpu().numpy()
                predictions = logits_np.argmax(axis=-1)
                max_logit_v = logits_np.max(axis=-1)

                enc_bboxes = enc["bbox"][0].cpu().numpy().tolist()
                renorm_to_local: dict[tuple, int] = {}
                for local_idx, rb in enumerate(renorm):
                    key = tuple(rb)
                    if key not in renorm_to_local:
                        renorm_to_local[key] = local_idx

                for enc_bbox, pred_id, max_l in zip(enc_bboxes, predictions, max_logit_v):
                    local_idx = renorm_to_local.get(tuple(enc_bbox))
                    if local_idx is None:
                        continue
                    orig_word    = blk_words[local_idx]
                    orig_bbox    = blk_bboxes[local_idx]
                    global_idx   = word_key_to_idx.get((orig_word, tuple(orig_bbox)))
                    if global_idx is None or global_idx >= n:
                        continue
                    label = self.id2label.get(int(pred_id), "O")
                    if max_l > all_logits[global_idx]:
                        all_logits[global_idx] = float(max_l)
                        all_labels[global_idx] = label

            except Exception as exc:
                log.warning(
                    "LayoutLMv3 bloc '%s' (%d tok) erreur: %s",
                    block.block_type, len(block.tokens), exc,
                )

        return _tokens_to_fields_and_tables(words, all_labels, bboxes)

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        doc_type: str = "unknown",
    ) -> dict:
        """
        Extrait les champs d'un PDF et retourne un dict structuré.

        Returns:
          {
            "doc_type": "bl",
            "n_pages": 1,
            "fields": {"BL_NUMBER": "CMAUJKWB660781151", "SHIPPER": "...", ...},
            "pages": [{"page": 1, "fields": {...}}],
          }
        """
        pdf_path = Path(pdf_path)
        doc      = fitz.open(str(pdf_path))
        all_fields: dict[str, list[str]] = defaultdict(list)
        all_tables: list[dict[str, Any]] = []
        all_words: list[str] = []
        all_bboxes: list[list[int]] = []
        pages_result = []
        last_image: Optional[Image.Image] = None

        for page_num_0 in range(len(doc)):
            page      = doc[page_num_0]
            image, w, h = _render_page(page)
            tokens    = _extract_page_tokens(page, w, h)

            if not tokens:
                continue

            words  = [t["text"] for t in tokens]
            bboxes = [t["bbox"]  for t in tokens]
            all_words.extend(words)
            all_bboxes.extend(bboxes)

            # ── Structure-first : segmentation zones + tableaux ────────────────
            layout_result = _layout_analyzer.analyze(words, bboxes, doc_type)

            # ── LayoutLMv3 par blocs sémantiques ──────────────────────────────
            page_fields, page_tables = self._run_layoutlm_blocks(image, words, bboxes, doc_type)

            # Regex / KV fallback sur le texte complet pour les champs manquants
            page_fields = _fill_missing_fields_from_text(page_fields, words, bboxes, doc_type)

            # ── Fusion structure-first : comble les champs absents du ML ───────
            for fname, val in layout_result["table_fields"].items():
                if fname not in page_fields:
                    page_fields[fname] = val

            last_image = image

            pages_result.append({
                "page":            page_num_0 + 1,
                "fields":          page_fields,
                "tables":          page_tables,
                "structured_rows": layout_result["table_rows"],
            })
            all_tables.extend(page_tables)
            all_tables.extend(layout_result["legacy_tables"])

            for field, value in page_fields.items():
                all_fields[field].append(value)

        doc.close()

        merged_fields = {
            field: values[0]
            for field, values in all_fields.items()
            if values
        }
        merged_fields = self._pp.clean(merged_fields, doc_type)

        features = FeatureBuilder.build_multimodal_features(
            image=last_image if last_image is not None else Image.new("RGB", (1000, 1000), color=255),
            words=all_words,
            bboxes=all_bboxes,
            doc_type=doc_type,
        ) if all_words else {}

        return {
            "doc_type": doc_type,
            "source":   str(pdf_path.name),
            "n_pages":  len(pages_result),
            "fields":   merged_fields,
            "tables":   all_tables,
            "features": features,
            "pages":    pages_result,
        }

    def extract_from_pil_image(
        self,
        image: Image.Image,
        doc_type: str = "unknown",
        words: list[str] | None = None,
        bboxes: list[list[int]] | None = None,
    ) -> dict[str, str]:
        """
        Extrait les champs depuis une image PIL avec tokens pré-extraits (OCR).

        Args:
            image   : image PIL RGB
            doc_type: type de document ("bl", "invoice", "packing_list", "unknown")
            words   : liste de mots extraits par OCR
            bboxes  : liste de bboxes normalisées [x0,y0,x1,y1] 0-1000

        Returns:
            dict {champ: valeur} extrait par LayoutLMv3
        """
        if not words:
            return {}

        # ── Structure-first ────────────────────────────────────────────────────
        layout_result = _layout_analyzer.analyze(words, bboxes, doc_type)

        # ── LayoutLMv3 par blocs sémantiques ─────────────────────────────────
        fields, tables = self._run_layoutlm_blocks(image, words, bboxes, doc_type)

        # Regex/kv fallback sur le texte complet
        fields = _fill_missing_fields_from_text(fields, words, bboxes, doc_type)

        # ── Fusion structure-first ─────────────────────────────────────────────
        for fname, val in layout_result["table_fields"].items():
            if fname not in fields:
                fields[fname] = val
        tables.extend(layout_result["legacy_tables"])

        cleaned_fields = self._pp.clean(fields, doc_type)
        features = FeatureBuilder.build_multimodal_features(
            image=image, words=words, bboxes=bboxes, doc_type=doc_type,
        )

        return {
            "fields":          cleaned_fields,
            "tables":          tables,
            "structured_rows": layout_result["table_rows"],
            "table_fields":    layout_result["table_fields"],
            "features":        features,
        }

    def extract_from_ocr_json(self, ocr_json_path: str | Path) -> dict:
        """
        Extrait les champs depuis un JSON OCR déjà produit par 01_ocr_extract.py.
        Plus rapide que extract_from_pdf car l'OCR est déjà fait.
        """
        ocr_data = json.loads(Path(ocr_json_path).read_text(encoding="utf-8"))
        all_fields: dict[str, list[str]] = defaultdict(list)
        all_tables: list[dict[str, Any]] = []
        pages_result = []

        ocr_doc_type = ocr_data.get("doc_type", "unknown")

        for page_data in ocr_data["pages"]:
            tokens = page_data["tokens"]
            if not tokens:
                continue

            words  = [t["text"] for t in tokens]
            bboxes = [t["bbox"]  for t in tokens]

            img_path = DATA_ROOT / page_data["image_path"]
            if img_path.exists():
                image = Image.open(img_path).convert("RGB")
            else:
                image = Image.new("RGB", (224, 224), color=255)

            # ── Structure-first ────────────────────────────────────────────────
            layout_result = _layout_analyzer.analyze(words, bboxes, ocr_doc_type)

            # ── LayoutLMv3 par blocs sémantiques ─────────────────────────────
            page_fields, page_tables = self._run_layoutlm_blocks(image, words, bboxes, ocr_doc_type)
            page_fields = _fill_missing_fields_from_text(page_fields, words, bboxes, ocr_doc_type)

            # ── Fusion structure-first ─────────────────────────────────────────
            for fname, val in layout_result["table_fields"].items():
                if fname not in page_fields:
                    page_fields[fname] = val
            page_tables.extend(layout_result["legacy_tables"])

            pages_result.append({
                "page":            page_data["page"],
                "fields":          page_fields,
                "tables":          page_tables,
                "structured_rows": layout_result["table_rows"],
            })
            all_tables.extend(page_tables)
            for field, value in page_fields.items():
                all_fields[field].append(value)

        merged_fields = {k: v[0] for k, v in all_fields.items() if v}
        merged_fields = self._pp.clean(merged_fields, ocr_data.get("doc_type", "unknown"))
        features = FeatureBuilder.build_multimodal_features(
            image=Image.open(DATA_ROOT / ocr_data["pages"][-1]["image_path"]).convert("RGB") if ocr_data["pages"] else Image.new("RGB", (1000, 1000), color=255),
            words=[word for page in ocr_data["pages"] for word in [t["text"] for t in page["tokens"]]],
            bboxes=[bbox for page in ocr_data["pages"] for bbox in [t["bbox"] for t in page["tokens"]]],
            doc_type=ocr_data.get("doc_type", "unknown"),
        ) if ocr_data.get("pages") else {}

        return {
            "doc_id":   ocr_data["doc_id"],
            "doc_type": ocr_data["doc_type"],
            "n_pages":  len(pages_result),
            "fields":   merged_fields,
            "tables":   all_tables,
            "features": features,
            "pages":    pages_result,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="extraction_service — extraction de champs via LayoutLMv3"
    )
    parser.add_argument("--pdf",       required=True, help="Chemin vers le PDF")
    parser.add_argument("--doc-type",  default="unknown",
                        choices=["bl", "invoice", "packing_list", "unknown"])
    parser.add_argument("--model-dir", default=None,
                        help="Répertoire du modèle fine-tuné (default: models/layoutlmv3-finetuned/best)")
    parser.add_argument("--output",    default=None, help="Sauvegarder le résultat JSON ici")
    args = parser.parse_args()

    svc    = ExtractionService(model_dir=args.model_dir)
    result = svc.extract_from_pdf(args.pdf, doc_type=args.doc_type)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Résultat sauvegardé → %s", args.output)


if __name__ == "__main__":
    main()
