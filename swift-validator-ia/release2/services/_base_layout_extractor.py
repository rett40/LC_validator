"""
_base_layout_extractor.py
=========================
Socle commun aux extracteurs spécialisés BL / Invoice / PL.

Pipeline complet en 4 couches :

  [1] LABEL DETECTION
      Scan du doc pour localiser chaque label (champ) et inférer la zone valeur attendue.
      → Empêche de confondre le texte du label avec la valeur extraite.

  [2] TRIPLE SEGMENTATION + fusion confiance max
      Exécute vertical, horizontal ET grid ; pour chaque champ garde
      la valeur avec le score de confiance le plus élevé.

  [3] TABLES + validation sémantique
      Champs multi-valeurs (conteneurs, lignes marchandises) → lignes de tableau
      regroupées par Y-clustering. Validation de format par ligne.

  [4] CHAMPS CALCULÉS + validation arithmétique
      Calcule les totaux depuis les lignes (TOTAL_AMOUNT, TOTAL_PACKAGES, TOTAL_CBM…).
      Compare avec les valeurs extraites et signale les écarts > tolérance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

BBOX_SCALE = 1000

# ─────────────────────────────────────────────────────────────────────────────
# Extraction mots / boîtes
# ─────────────────────────────────────────────────────────────────────────────

def extract_words_pdfplumber(pdf_path: str | Path, page_num: int = 0) -> tuple[list[str], list[list[int]]]:
    """Extraction mots + boîtes normalisées [0-1000] via pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber requis : pip install pdfplumber")

    words_out: list[str] = []
    bboxes_out: list[list[int]] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_num >= len(pdf.pages):
            return words_out, bboxes_out
        page = pdf.pages[page_num]
        pw, ph = page.width, page.height
        if pw <= 0 or ph <= 0:
            return words_out, bboxes_out

        for w in page.extract_words():
            text = str(w.get("text", "")).strip()
            if not text:
                continue
            x0 = max(0, min(BBOX_SCALE, int(w["x0"]    / pw * BBOX_SCALE)))
            y0 = max(0, min(BBOX_SCALE, int(w["top"]    / ph * BBOX_SCALE)))
            x1 = max(0, min(BBOX_SCALE, int(w["x1"]     / pw * BBOX_SCALE)))
            y1 = max(0, min(BBOX_SCALE, int(w["bottom"] / ph * BBOX_SCALE)))
            if x1 <= x0: x1 = min(BBOX_SCALE, x0 + 1)
            if y1 <= y0: y1 = min(BBOX_SCALE, y0 + 1)
            words_out.append(text)
            bboxes_out.append([x0, y0, x1, y1])

    return words_out, bboxes_out


def extract_words_fitz(page: Any, img_w: int, img_h: int) -> tuple[list[str], list[list[int]]]:
    """Extraction mots + boîtes via PyMuPDF (fallback)."""
    rect  = page.rect
    pw, ph = rect.width, rect.height
    words_out: list[str]        = []
    bboxes_out: list[list[int]] = []

    for wt in page.get_text("words"):
        x0, y0, x1, y1, text = wt[:5]
        text = text.strip()
        if not text:
            continue
        sx, sy = img_w / pw, img_h / ph
        words_out.append(text)
        bboxes_out.append([
            max(0, min(BBOX_SCALE, int(x0 * sx / img_w * BBOX_SCALE))),
            max(0, min(BBOX_SCALE, int(y0 * sy / img_h * BBOX_SCALE))),
            max(0, min(BBOX_SCALE, int(x1 * sx / img_w * BBOX_SCALE))),
            max(0, min(BBOX_SCALE, int(y1 * sy / img_h * BBOX_SCALE))),
        ])

    return words_out, bboxes_out


# ─────────────────────────────────────────────────────────────────────────────
# Couche 1 — Détection des labels dans le document
# ─────────────────────────────────────────────────────────────────────────────

def _merge_bboxes(bboxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    ]


def detect_label_zones(
    words:          list[str],
    bboxes:         list[list[int]],
    field_synonyms: dict[str, list[str]],
) -> dict[str, dict]:
    """
    Parcourt le document et localise chaque label connu (champ).
    Infère la zone valeur attendue (à droite du label ou en dessous).

    Retourne :
      {field_name: {
          "label_bbox":  [x0,y0,x1,y1],  ← zone du label lui-même (à rejeter)
          "value_zone":  [x0,y0,x1,y1],  ← zone attendue pour la valeur
          "pattern":     "inline"|"block" ← disposition label→valeur
      }}
    """
    result: dict[str, dict] = {}
    n       = len(words)
    # Strip trailing punctuation so "NAME:" matches "NAME", "LOADING," matches "LOADING"
    words_up = [re.sub(r'[\.:;,!?()\[\]]+$', '', w.upper().strip()) for w in words]

    for field_name, synonyms in field_synonyms.items():
        if field_name in result:
            continue

        for syn in sorted(synonyms, key=len, reverse=True):  # plus long = plus spécifique
            syn_tokens = syn.upper().split()
            k = len(syn_tokens)
            if k == 0:
                continue

            for i in range(n - k + 1):
                if words_up[i : i + k] == syn_tokens:
                    label_bbox = _merge_bboxes(bboxes[i : i + k])
                    lx0, ly0, lx1, ly1 = label_bbox
                    ly_c = (ly0 + ly1) / 2

                    # Chercher le mot suivant pour déterminer la disposition
                    next_idx = i + k
                    if next_idx < n:
                        nx_c = (bboxes[next_idx][1] + bboxes[next_idx][3]) / 2
                        is_inline = abs(nx_c - ly_c) < 25  # même ligne (±25 unités)
                    else:
                        is_inline = True

                    if is_inline:
                        # Valeur à droite du label sur la même ligne
                        value_zone = [lx1, max(0, ly0 - 15), BBOX_SCALE, min(BBOX_SCALE, ly1 + 15)]
                        pattern    = "inline"
                    else:
                        # Valeur en dessous du label (bloc)
                        value_zone = [
                            max(0, lx0 - 20), ly1,
                            min(BBOX_SCALE, lx1 + 200), min(BBOX_SCALE, ly1 + 120),
                        ]
                        pattern = "block"

                    result[field_name] = {
                        "label_bbox": label_bbox,
                        "value_zone": value_zone,
                        "pattern":    pattern,
                    }
                    break

            if field_name in result:
                break

    return result


def is_in_label_zone(
    entity_bbox:  list[int],
    label_zones:  dict[str, dict],
    overlap_thr:  float = 0.5,
) -> bool:
    """
    Retourne True si la bbox d'une entité chevauche significativement
    une zone label connue (le token serait donc le label, pas la valeur).
    """
    ex0, ey0, ex1, ey1 = entity_bbox
    e_area = max(1, (ex1 - ex0) * (ey1 - ey0))

    for zones in label_zones.values():
        lx0, ly0, lx1, ly1 = zones["label_bbox"]
        ov_x = max(0, min(ex1, lx1) - max(ex0, lx0))
        ov_y = max(0, min(ey1, ly1) - max(ey0, ly0))
        if (ov_x * ov_y) / e_area >= overlap_thr:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Couche 2 — Segmentation du layout
# ─────────────────────────────────────────────────────────────────────────────

_ALL_METHODS = ("horizontal",)


def segment_layout(
    words:  list[str],
    bboxes: list[list[int]],
    method: str = "grid",
) -> dict[str, list[dict]]:
    """
    Découpe la page en régions logiques.
    Retourne dict {segment_name: [{"text": str, "bbox": [...]}]}
    """
    segments: dict[str, list[dict]] = defaultdict(list)

    for word, bbox in zip(words, bboxes):
        x0, y0, x1, y1 = bbox
        xc = (x0 + x1) / 2
        yc = (y0 + y1) / 2

        if method == "vertical":
            if   yc < 200:  seg = "header"
            elif yc < 500:  seg = "parties"
            elif yc < 850:  seg = "table"
            else:           seg = "footer"

        elif method == "horizontal":
            seg = "left_column" if xc < 500 else "right_column"

        else:  # grid
            if yc >= 850:
                seg = "footer"
            elif yc < 250:
                seg = "header_left" if xc < 500 else "header_right"
            else:
                seg = "body_left" if xc < 500 else "body_right"

        segments[seg].append({"text": word, "bbox": bbox})

    return dict(segments)


# ─────────────────────────────────────────────────────────────────────────────
# Agrégation IOB2 → entités avec confiance
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_entities(
    words:       list[str],
    labels:      list[str],
    bboxes:      list[list[int]],
    logits_list: list[np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """
    Convertit une séquence IOB2 en entités agrégées.
    Retourne liste de dicts {field, value, bbox, confidence}.
    """
    entities: list[dict] = []
    cur_field:  str | None      = None
    cur_tokens: list[str]       = []
    cur_bboxes: list[list[int]] = []
    cur_scores: list[float]     = []

    def _flush():
        if cur_field and cur_tokens:
            merged = _merge_bboxes(cur_bboxes)
            entities.append({
                "field":      cur_field,
                "value":      " ".join(cur_tokens),
                "bbox":       merged,
                "confidence": float(np.mean(cur_scores)) if cur_scores else 0.0,
            })

    for i, (word, label) in enumerate(zip(words, labels)):
        if label == "O" or "-" not in label:
            _flush()
            cur_field = None; cur_tokens = []; cur_bboxes = []; cur_scores = []
            continue

        prefix, fld = label.split("-", 1)
        score = 1.0
        if logits_list and i < len(logits_list):
            lg = logits_list[i]
            exp_lg = np.exp(lg - lg.max())
            score  = float(np.max(exp_lg / exp_lg.sum()))

        bb = bboxes[i] if i < len(bboxes) else [0, 0, 0, 0]

        if prefix == "B":
            _flush()
            cur_field = fld; cur_tokens = [word]; cur_bboxes = [bb]; cur_scores = [score]
        elif prefix == "I" and cur_field == fld:
            cur_tokens.append(word); cur_bboxes.append(bb); cur_scores.append(score)
        else:
            _flush()
            cur_field = fld; cur_tokens = [word]; cur_bboxes = [bb]; cur_scores = [score]

    _flush()
    return entities


# ─────────────────────────────────────────────────────────────────────────────
# Couche 2 — Filtrage anti-hallucination
# ─────────────────────────────────────────────────────────────────────────────

def _strip_label_prefix(value: str, forbidden: list[str]) -> str:
    val_up = value.strip().upper()
    for kw in sorted(forbidden, key=len, reverse=True):
        if val_up.startswith(kw.upper()):
            remainder = value.strip()[len(kw):].strip()
            remainder = re.sub(r'^[\s:/\-–|]+', '', remainder).strip()
            if remainder:
                return remainder
    m = re.match(r'^[A-Z][A-Z0-9\s/\.\-]{2,45}[:]\s*(.+)', value.strip())
    if m:
        return m.group(1).strip()
    return ""


def filter_hallucination(value: str, forbidden: list[str]) -> tuple[bool, str]:
    """
    Vérifie si la valeur est un nom de champ / synonyme.
    Nettoie le préfixe label si la valeur est de la forme "LABEL: valeur".
    Retourne (is_hallucination, cleaned_value).
    """
    if not value or not value.strip():
        return True, ""
    val_lower = value.lower().strip()
    cleaned   = _strip_label_prefix(value, forbidden)
    if cleaned:
        cl_low = cleaned.lower().strip()
        for kw in forbidden:
            if cl_low == kw.lower() or cl_low.startswith(kw.lower() + " "):
                return True, ""
        return False, cleaned
    for kw in forbidden:
        if val_lower == kw.lower():
            return True, ""
        if val_lower.startswith(kw.lower()):
            suffix = val_lower[len(kw):].strip().lstrip(":").strip()
            if not suffix:
                return True, ""
    return False, value.strip()


def validate_numeric_field(value: str) -> bool:
    return bool(re.search(r'\d', value))


def validate_date_field(value: str) -> bool:
    has_digit = bool(re.search(r'\d', value))
    has_sep   = bool(re.search(
        r'[-/.]|\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b',
        value.upper()
    ))
    return has_digit and has_sep


# ─────────────────────────────────────────────────────────────────────────────
# Couche 3 — Construction des lignes de tableau
# ─────────────────────────────────────────────────────────────────────────────

def build_table_rows(
    candidates:       list[dict],
    row_anchor_field: str,
    row_fields:       set[str],
    y_threshold:      int = 35,
) -> list[dict[str, str]]:
    """
    Groupe les candidats multi-valeurs en lignes de tableau.

    Stratégie :
      1. Trouver toutes les occurrences de row_anchor_field (ex: CONTAINER_NO)
      2. Pour chaque ancre, collecter les row_fields dont le Y-center est
         dans y_threshold de l'ancre
      3. Retourner la liste triée par Y-position

    Retourne :
      [{"CONTAINER_NO": "MSCU1234567", "SEAL_NO": "123", ...}, ...]
    """
    # Filtrer uniquement les candidats de row_fields
    row_candidates = [c for c in candidates if c["field"] in row_fields]
    if not row_candidates:
        return []

    # Trouver les ancres (row_anchor_field)
    anchors = [c for c in row_candidates if c["field"] == row_anchor_field]
    if not anchors:
        # Pas d'ancre → grouper par Y-clustering sans ancre
        return _cluster_rows_by_y(row_candidates, y_threshold)

    # Pour chaque ancre, construire une ligne
    anchors_sorted = sorted(anchors, key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    rows: list[dict[str, str]] = []

    for anchor in anchors_sorted:
        ay_c = (anchor["bbox"][1] + anchor["bbox"][3]) / 2
        row: dict[str, dict] = {row_anchor_field: anchor}

        for cand in row_candidates:
            if cand["field"] == row_anchor_field:
                continue
            cy_c = (cand["bbox"][1] + cand["bbox"][3]) / 2
            if abs(cy_c - ay_c) <= y_threshold:
                f = cand["field"]
                # Garde la confiance la plus élevée pour ce champ sur cette ligne
                if f not in row or cand["confidence"] > row[f]["confidence"]:
                    row[f] = cand

        rows.append({f: v["value"] for f, v in row.items()})

    return rows


def _cluster_rows_by_y(
    candidates:  list[dict],
    y_threshold: int,
) -> list[dict[str, str]]:
    """Regroupe des candidats par Y-cluster quand pas d'ancre disponible."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: (c["bbox"][1] + c["bbox"][3]) / 2)
    clusters: list[list[dict]] = []
    current: list[dict] = [sorted_cands[0]]
    cur_y = (sorted_cands[0]["bbox"][1] + sorted_cands[0]["bbox"][3]) / 2

    for c in sorted_cands[1:]:
        cy = (c["bbox"][1] + c["bbox"][3]) / 2
        if abs(cy - cur_y) <= y_threshold:
            current.append(c)
        else:
            clusters.append(current)
            current = [c]
            cur_y   = cy
    clusters.append(current)

    rows: list[dict[str, str]] = []
    for cluster in clusters:
        row: dict[str, dict] = {}
        for cand in cluster:
            f = cand["field"]
            if f not in row or cand["confidence"] > row[f]["confidence"]:
                row[f] = cand
        rows.append({f: v["value"] for f, v in row.items()})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Couche 4 — Champs calculés + validation arithmétique
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputedFieldSpec:
    """Spécification d'un champ calculé depuis les lignes du tableau."""
    target:    str           # champ à calculer / valider
    op:        str           # "sum" | "sum_product" | "count"
    fields:    list[str]     # colonnes source dans les lignes
    tolerance: float = 0.05  # écart relatif toléré (5% par défaut)


def _to_number(s: str) -> float | None:
    """Parse un nombre depuis une chaîne (gère virgules, espaces, unités)."""
    if not s:
        return None
    # Retirer unités courantes
    s = re.sub(r'(?i)\b(kg|kgs|mt|cbm|m3|usd|eur|cny|gbp)\b', '', s)
    s = s.strip()
    # Normaliser séparateurs
    val = s.replace(" ", "")
    if re.search(r'\d,\d{3}', val):     # format 1,234.56
        val = val.replace(",", "")
    else:
        val = val.replace(",", ".")     # format 1.234,56 → 1234.56
    try:
        return float(val)
    except ValueError:
        return None


def validate_computed_fields(
    fields:         dict[str, str],
    table_rows:     list[dict[str, str]],
    computed_specs: list[ComputedFieldSpec],
) -> tuple[dict[str, str], list[dict]]:
    """
    Pour chaque ComputedFieldSpec :
      1. Calcule la valeur depuis les lignes du tableau
      2. Compare avec la valeur extraite (si présente)
      3. Si écart > tolérance → alerte
      4. Si valeur absente → remplace par la valeur calculée

    Retourne (updated_fields, validation_results).
    """
    updated   = dict(fields)
    results:  list[dict] = []

    for spec in computed_specs:
        computed_val: float | None = None

        if spec.op == "sum" and len(spec.fields) == 1:
            total = 0.0
            found = False
            for row in table_rows:
                n = _to_number(row.get(spec.fields[0], ""))
                if n is not None:
                    total += n
                    found  = True
            computed_val = total if found else None

        elif spec.op == "sum_product" and len(spec.fields) == 2:
            total = 0.0
            found = False
            for row in table_rows:
                a = _to_number(row.get(spec.fields[0], ""))
                b = _to_number(row.get(spec.fields[1], ""))
                if a is not None and b is not None:
                    total += a * b
                    found  = True
            computed_val = total if found else None

        elif spec.op == "count":
            computed_val = float(len(table_rows)) if table_rows else None

        if computed_val is None:
            continue

        computed_str = str(round(computed_val, 3)).rstrip("0").rstrip(".")

        extracted_raw = updated.get(spec.target)
        extracted_num = _to_number(extracted_raw) if extracted_raw else None

        if extracted_num is not None and extracted_num != 0:
            diff = abs(extracted_num - computed_val) / abs(extracted_num)
            if diff > spec.tolerance:
                results.append({
                    "field":     spec.target,
                    "status":    "mismatch",
                    "extracted": extracted_raw,
                    "computed":  computed_str,
                    "diff_pct":  round(diff * 100, 1),
                })
            else:
                results.append({
                    "field":     spec.target,
                    "status":    "ok",
                    "extracted": extracted_raw,
                    "computed":  computed_str,
                })
        else:
            # Valeur absente → remplir avec la valeur calculée
            updated[spec.target] = computed_str
            results.append({
                "field":    spec.target,
                "status":   "filled",
                "computed": computed_str,
            })

    return updated, results


# ─────────────────────────────────────────────────────────────────────────────
# Validation sémantique des lignes de tableau
# ─────────────────────────────────────────────────────────────────────────────

def validate_table_rows_semantic(
    rows:              list[dict[str, str]],
    row_validators:    dict[str, Any],   # {field: callable(value) -> bool}
) -> list[dict[str, str]]:
    """
    Filtre les lignes de tableau en appliquant des validateurs sémantiques par champ.
    Les lignes avec un champ ancre invalide sont rejetées.
    Retourne les lignes valides.
    """
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        row_ok = True
        for field_name, validator in row_validators.items():
            val = row.get(field_name)
            if val and not validator(val):
                row_ok = False
                break
        if row_ok:
            valid_rows.append(row)
    return valid_rows


# ─────────────────────────────────────────────────────────────────────────────
# Classe de base
# ─────────────────────────────────────────────────────────────────────────────

# Segmentation par blocs : horizontal uniquement (left_column / right_column).
# Vertical (4 segments) + grid (5 segments) supprimés pour performance :
# horizontal = 2 inférences au lieu de 11, pour ~82 % de gain sur cette phase.
_BLOCK_METHODS = ("horizontal",)

# Pondérations de confiance par méthode (full_page > adaptive > blocs)
_METHOD_WEIGHT: dict[str, float] = {
    "full_page":  1.00,   # pass complet — entités jamais coupées
    "adaptive":   0.95,   # zone ciblée par label détecté — très précis
    "horizontal": 0.85,   # blocs X — seule méthode de segmentation active
}


class BaseLayoutExtractor:
    """
    Extracteur LayoutLMv3 spécialisé (BL / Invoice / Packing List).

    Pipeline d'extraction en 3 passes + 4 couches :

      Pass 0 : full_page sliding window  (conf ×1.00)
               → parcourt la PAGE ENTIÈRE en fenêtres chevauchantes (350 mots, stride 175)
               → garantit qu'aucune entité multi-tokens n'est coupée par une frontière

      Pass 1 : segmentation horizontale   (conf ×0.85)
               → left_column / right_column (axe X, seuil x=500)
               → 2 inférences au lieu de 11 (vertical+horizontal+grid supprimés)

      Pass 2 : zones adaptatives          (conf ×0.95)
               → une zone de lecture par label détecté dans le document
               → très efficace pour SHIPPER, CONSIGNEE, NOTIFY_PARTY,
                 PORT_OF_LOADING, PORT_OF_DISCHARGE, etc.

      Couche 1 : détection des labels avec normalisation des synonymes
                 (VESSEL NAME → zone VESSEL, SHIPPER/EXPORTER → zone SHIPPER)
      Couche 2 : nettoyage valeurs par synonymes + anti-hallucination
                 "VESSEL NAME MSC OSCAR" → "MSC OSCAR"
                 "PORT OF LOADING" (seul) → rejeté
      Couche 3 : construction des lignes de tableau (multi-valeurs par Y-clustering)
      Couche 4 : champs calculés + validation arithmétique

    Toutes les passes sont fusionnées par confiance maximale pondérée.
    La réponse est COMPLÈTE : tous les EXPECTED_FIELDS apparaissent (trouvés ou manquants).

    Attributs de classe à définir dans les sous-classes :
      MODEL_DIR, EXPECTED_FIELDS, TABLE_NAME,
      REJECT_IN_SEGMENT, FIELD_NAME_MAP, FORBIDDEN_KEYWORDS,
      NUMERIC_FIELDS, DATE_FIELDS, FIELD_SYNONYMS,
      TABLE_FIELDS, ROW_ANCHOR_FIELD, ROW_FIELDS,
      COMPUTED_SPECS, ROW_VALIDATORS
    """

    MODEL_DIR:           Path | None              = None
    EXPECTED_FIELDS:     set[str]                 = set()
    TABLE_NAME:          str                      = "items"
    REJECT_IN_SEGMENT:   dict[str, set[str]]      = {}
    FIELD_NAME_MAP:      dict[str, str]            = {}
    FORBIDDEN_KEYWORDS:  list[str]                 = []
    NUMERIC_FIELDS:      set[str]                  = set()
    DATE_FIELDS:         set[str]                  = set()
    FIELD_SYNONYMS:      dict[str, list[str]]      = {}
    TABLE_FIELDS:        set[str]                  = set()
    ROW_ANCHOR_FIELD:    str                        = ""
    ROW_FIELDS:          set[str]                  = set()
    COMPUTED_SPECS:      list                       = []   # list[ComputedFieldSpec]
    ROW_VALIDATORS:      dict[str, Any]            = {}

    # Paramètres de la fenêtre glissante full-page
    _WINDOW_SIZE: int = 350    # mots par fenêtre (bien sous la limite 512 tokens)
    _WINDOW_STRIDE: int = 175  # chevauchement 50 % pour ne pas couper d'entités

    def __init__(self, model_dir: Path | str | None = None):
        import torch
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

        model_path = Path(model_dir) if model_dir else self.MODEL_DIR
        if not model_path or not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                "Vérifier models/models_layout/"
            )

        self.device = "cpu"
        self.processor = LayoutLMv3Processor.from_pretrained(str(model_path), apply_ocr=False)
        self.model     = LayoutLMv3ForTokenClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

        id2label_raw  = json.loads((model_path / "id2label.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in id2label_raw.items()}

        # Charger SpaCy si disponible
        self._spacy_nlp = None
        try:
            import spacy
            self._spacy_nlp = spacy.load("en_core_web_sm")
        except ImportError:
            pass
        self._all_synonyms_sorted: list[str] = self._build_synonym_list()
        self._all_synonym_norms:   frozenset[str] = frozenset(
            s.rstrip(".:;/ ") for s in self._all_synonyms_sorted
        )

        # Cache mono-slot du prétraitement image (resize 224 + normalisation).
        # La même page PIL traverse les 13-25 forwards d'un document : le
        # pixel_values est identique à chaque fois — le recalculer par segment
        # est du travail perdu. Clé = id(image) : hit garanti au sein d'un
        # document, invalidation automatique au document suivant.
        self._pv_cache_key: int | None = None
        self._pv_cache: Any = None

        log.info("%-22s prêt — %d labels | %d synonymes | device: %s",
                 self.__class__.__name__, len(self.id2label),
                 len(self._all_synonyms_sorted), self.device)

    # ── Inférence LayoutLMv3 ──────────────────────────────────────────────────

    def _decode_sample(
        self,
        encoded_bboxes: list[list[int]],
        logits_2d:      np.ndarray,
        bboxes:         list[list[int]],
        n:              int,
    ) -> tuple[list[str], list[np.ndarray]]:
        """
        Reconstruit (labels IOB2, logits) par mot à partir des logits d'un échantillon.

        Logique de remapping identique pour les chemins single et batch :
        on associe chaque token à son mot via sa bbox encodée (les tokens de
        padding ont une bbox [0,0,0,0] qui ne correspond à aucun mot → ignorés).
        """
        predictions = np.argmax(logits_2d, axis=-1)
        bbox_to_word = {tuple(b): i for i, b in enumerate(bboxes)}

        word_labels: dict[int, str]        = {}
        word_logits: dict[int, np.ndarray] = {}
        for j, (enc_bb, pred_id) in enumerate(zip(encoded_bboxes, predictions)):
            widx = bbox_to_word.get(tuple(enc_bb))
            if widx is not None and widx not in word_labels:
                word_labels[widx] = self.id2label.get(int(pred_id), "O")
                word_logits[widx] = logits_2d[j]

        zero = np.zeros(len(self.id2label))
        return (
            [word_labels.get(i, "O") for i in range(n)],
            [word_logits.get(i, zero) for i in range(n)],
        )

    def _predict_segment(
        self,
        image:  Any,
        words:  list[str],
        bboxes: list[list[int]],
    ) -> tuple[list[str], list[np.ndarray]]:
        """
        Inférence LayoutLMv3 sur UN bloc de mots.

        NB perf (mesuré sur CPU/MX130) : on fait UN appel par segment avec
        padding="longest". Batcher plusieurs segments en un seul forward s'est
        révélé ~15 % PLUS LENT sur CPU : le padding au plus long segment du batch
        gaspille plus de calcul que l'overhead par appel qu'il économise. Sans GPU,
        l'appel unitaire (paddé à SA seule longueur) est optimal.
        """
        import torch

        if not words:
            return [], []

        # padding="longest" : chaque segment n'est paddé qu'à SA longueur réelle
        # (au lieu de 512 systématiques). Le masque d'attention neutralise déjà le
        # padding → sorties identiques à padding="max_length" (bruit flottant ~1e-6).
        # Le prétraitement image (partie image_processor du processor) est caché
        # par page : tokenizer seul ici + pixel_values réutilisés.
        # R2_DISABLE_PV_CACHE=1 réactive l'ancien chemin (A/B du harnais perf).
        if os.environ.get("R2_DISABLE_PV_CACHE") == "1":
            encoding = self.processor(
                images=image, text=words, boxes=bboxes,
                truncation=True, max_length=512, padding="longest", return_tensors="pt",
            )
        else:
            encoding = self.processor.tokenizer(
                text=words, boxes=bboxes,
                truncation=True, max_length=512, padding="longest", return_tensors="pt",
            )
            if self._pv_cache_key != id(image):
                self._pv_cache = self.processor.image_processor(
                    images=image, return_tensors="pt"
                ).pixel_values
                self._pv_cache_key = id(image)
            encoding["pixel_values"] = self._pv_cache
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.inference_mode():
            outputs = self.model(**encoding)

        logits         = outputs.logits[0].cpu().numpy()        # (seq_len, n_labels)
        encoded_bboxes = encoding["bbox"][0].cpu().numpy().tolist()
        return self._decode_sample(encoded_bboxes, logits, bboxes, len(words))

    # ── Pass 0 : full-page sliding window ─────────────────────────────────────

    def _predict_full_page(
        self,
        image:  Any,
        words:  list[str],
        bboxes: list[list[int]],
    ) -> list[dict]:
        """
        Parcourt la PAGE ENTIÈRE avec une fenêtre glissante.

        Avantage vs segmentation par blocs : les entités multi-tokens qui
        traversent une frontière de segment sont capturées complètement ici.

        Tri préalable des mots par Y puis X (ordre de lecture naturel) avant
        de découper en fenêtres.  Chevauchement 50 % entre fenêtres contigües.

        Pas de restriction REJECT_IN_SEGMENT → les labels doivent passer
        uniquement le filtre anti-hallucination + contraintes sémantiques.
        """
        n = len(words)
        if n == 0:
            return []

        # Trier par ordre de lecture (Y croissant, puis X croissant)
        order = sorted(
            range(n),
            key=lambda i: ((bboxes[i][1] + bboxes[i][3]) / 2,
                           (bboxes[i][0] + bboxes[i][2]) / 2),
        )
        sw = [words[i]  for i in order]
        sb = [bboxes[i] for i in order]

        ws  = self._WINDOW_SIZE
        st  = self._WINDOW_STRIDE
        w   = _METHOD_WEIGHT["full_page"]
        candidates: list[dict] = []

        for start in range(0, n, st):
            end = min(start + ws, n)
            win_words  = sw[start:end]
            win_bboxes = sb[start:end]

            labels, logits_list = self._predict_segment(image, win_words, win_bboxes)
            entities = aggregate_entities(win_words, labels, win_bboxes, logits_list)
            for ent in entities:
                ok, cleaned = self._validate_entity(
                    ent["field"], ent["value"], "full_page",
                    entity_bbox=ent["bbox"], label_zones=None,
                )
                if ok:
                    candidates.append({
                        "field":      ent["field"],
                        "value":      cleaned,
                        "confidence": ent["confidence"] * w,
                        "segment":    "full_page",
                        "method":     "full_page",
                        "bbox":       ent["bbox"],
                    })

            if end == n:
                break

        return candidates

    # ── Pass 1-3 : segmentation par blocs ─────────────────────────────────────

    def _predict_blocks(
        self,
        image:       Any,
        words:       list[str],
        bboxes:      list[list[int]],
        label_zones: dict[str, dict],
    ) -> list[dict]:
        """
        Exécute la segmentation horizontale (left_column / right_column) uniquement.
        Vertical et grid supprimés : 2 inférences au lieu de 11 pour cette phase.
        """
        candidates: list[dict] = []

        for method in _BLOCK_METHODS:
            w        = _METHOD_WEIGHT[method]
            segments = segment_layout(words, bboxes, method=method)
            for seg_name, seg_tokens in segments.items():
                if not seg_tokens:
                    continue
                sw_b = [t["text"] for t in seg_tokens]
                sb_b = [t["bbox"] for t in seg_tokens]

                labels, logits_list = self._predict_segment(image, sw_b, sb_b)
                entities = aggregate_entities(sw_b, labels, sb_b, logits_list)
                for ent in entities:
                    ok, cleaned = self._validate_entity(
                        ent["field"], ent["value"], seg_name,
                        entity_bbox=ent["bbox"], label_zones=label_zones,
                    )
                    if ok:
                        candidates.append({
                            "field":      ent["field"],
                            "value":      cleaned,
                            "confidence": ent["confidence"] * w,
                            "segment":    seg_name,
                            "method":     method,
                            "bbox":       ent["bbox"],
                        })

        return candidates

    # ── Normalisation des synonymes (calculé une fois au chargement) ─────────────

    def _build_synonym_list(self) -> list[str]:
        """
        Construit la liste triée (plus long d'abord) de tous les synonymes de champs.
        Utilisée pour détecter et nettoyer les préfixes de labels dans les valeurs.
        Exemple : "VESSEL NAME MSC OSCAR" → strip "VESSEL NAME" → "MSC OSCAR"
        """
        syns: set[str] = set()
        for syn_list in self.FIELD_SYNONYMS.values():
            syns.update(s.upper().strip() for s in syn_list)
        for kw in self.FORBIDDEN_KEYWORDS:
            syns.add(kw.upper().strip())
        return sorted(syns, key=len, reverse=True)

    def _clean_with_all_synonyms(self, value: str) -> str:
        """
        Nettoie une valeur extraite :
          1. Si la valeur est EXACTEMENT un synonyme de champ → rejeté (retourne "")
          2. Si la valeur COMMENCE PAR un synonyme → strip du préfixe + séparateurs
          3. Si après strip la valeur est elle-même un synonyme → rejeté
          4. Sinon → retourne la valeur nettoyée

        Exemples :
          "VESSEL NAME MSC OSCAR"      → "MSC OSCAR"
          "PORT OF LOADING"            → ""   (label pur)
          "SHIPPER/EXPORTER: TUNISIE"  → "TUNISIE"
          "CONSIGNEE EUROALIMENTA GmbH"→ "EUROALIMENTA GmbH"
          "PORT OF LOADING PORT OF DISCHARGE" → "" (deux labels)
        """
        val = value.strip()
        if not val:
            return ""

        val_up = val.upper()

        # 1. Rejet exact (avec ou sans ponctuation finale)
        val_norm = val_up.rstrip(".:;/ \t\n")
        if val_norm in self._all_synonym_norms:
            return ""

        # 2. Strip du préfixe de synonyme (du plus long au plus court)
        for syn_up in self._all_synonyms_sorted:
            if val_up.startswith(syn_up):
                remainder = val[len(syn_up):].lstrip(" :/–—-|\n\t").strip()
                if not remainder:
                    return ""  # valeur = label pur
                # 3. Vérifier si le reste est lui-même un label
                rem_norm = remainder.upper().rstrip(".:;/ \t\n")
                if rem_norm in self._all_synonym_norms:
                    return ""  # "LABEL_A LABEL_B" → rejeté
                if len(remainder) >= 2:
                    return remainder
                return ""

        return val

    # ── Pass 4 : zones adaptatives (ciblées par label détecté) ─────────────────

    def _predict_adaptive_zones(
        self,
        image:       Any,
        words:       list[str],
        bboxes:      list[list[int]],
        label_zones: dict[str, dict],
    ) -> list[dict]:
        """
        Pour chaque zone de label détectée, extrait les mots dans la zone valeur
        attendue et fait tourner le modèle UNIQUEMENT sur ces mots.

        Avantage vs segmentation générique : la zone est centrée exactement sur le
        champ concerné, ce qui réduit le bruit d'autres champs voisins.
        Particulièrement efficace pour les champs importants (SHIPPER, CONSIGNEE,
        NOTIFY_PARTY, PORT_OF_LOADING, PORT_OF_DISCHARGE).
        """
        if not label_zones:
            return []

        candidates: list[dict] = []
        w_conf = _METHOD_WEIGHT["adaptive"]
        # Marge de contexte autour de la zone valeur (en unités 0-1000)
        PAD_X, PAD_Y = 60, 50

        for field_name, zone_info in label_zones.items():
            vx0, vy0, vx1, vy1 = zone_info["value_zone"]

            # Étendre légèrement pour capturer les mots partiellement dans la zone
            ex0 = max(0, vx0 - PAD_X)
            ey0 = max(0, vy0 - PAD_Y)
            ex1 = min(BBOX_SCALE, vx1 + PAD_X)
            ey1 = min(BBOX_SCALE, vy1 + PAD_Y)

            zone_words:  list[str]       = []
            zone_bboxes: list[list[int]] = []
            for word, bbox in zip(words, bboxes):
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                    zone_words.append(word)
                    zone_bboxes.append(bbox)

            if len(zone_words) < 2:
                continue

            labels, logits_list = self._predict_segment(image, zone_words, zone_bboxes)
            entities = aggregate_entities(zone_words, labels, zone_bboxes, logits_list)
            for ent in entities:
                ok, cleaned = self._validate_entity(
                    ent["field"], ent["value"], "adaptive",
                    entity_bbox=ent["bbox"], label_zones=label_zones,
                )
                if ok:
                    candidates.append({
                        "field":      ent["field"],
                        "value":      cleaned,
                        "confidence": ent["confidence"] * w_conf,
                        "segment":    f"adaptive_{field_name}",
                        "method":     "adaptive",
                        "bbox":       ent["bbox"],
                    })

        return candidates

    # ── Validation d'une entité ────────────────────────────────────────────────

    def _validate_entity(
        self,
        field_name:  str,
        value:       str,
        segment:     str,
        entity_bbox: list[int]       | None = None,
        label_zones: dict[str, dict] | None = None,
    ) -> tuple[bool, str]:
        """
        Valide une entité extraite (5 couches) :
          1. Rejet par segment  — ignoré pour full_page et adaptive
          2. Zone label         — bbox entité ∩ zone label > seuil → rejeté
          3. Normalisation synonymes — strip des préfixes label (VESSEL NAME → strip)
             + rejet si valeur = label pur (PORT OF LOADING pur)
             + rejet si valeur = deux labels concaténés
          4. Contraintes sémantiques — numérique / date
          5. Longueur minimale ≥ 2
        """
        # [1] Rejet par segment (pas applicable aux passes non-segmentées)
        if segment not in ("full_page", "adaptive"):
            if field_name in self.REJECT_IN_SEGMENT.get(segment, set()):
                return False, ""

        # [2] Zone label
        if entity_bbox and label_zones and is_in_label_zone(entity_bbox, label_zones):
            return False, ""

        # [3-a] Nettoyage par synonymes de champs (normalisation du nom de champ)
        #       "VESSEL NAME MSC OSCAR" → "MSC OSCAR"
        #       "PORT OF LOADING" (seul) → ""  rejeté
        if self._all_synonyms_sorted:
            cleaned = self._clean_with_all_synonyms(value)
            if not cleaned:
                return False, ""
        else:
            cleaned = value.strip()

        # [3-b] Filtrage anti-hallucination complémentaire (FORBIDDEN_KEYWORDS)
        is_halluc, cleaned = filter_hallucination(cleaned, self.FORBIDDEN_KEYWORDS)
        if is_halluc or not cleaned.strip():
            return False, ""

        # [4] Contraintes sémantiques
        if field_name in self.NUMERIC_FIELDS and not validate_numeric_field(cleaned):
            return False, ""
        if field_name in self.DATE_FIELDS and not validate_date_field(cleaned):
            return False, ""

        # [5] Longueur minimale (flexible)
        if len(cleaned.strip()) < 1:
            return False, ""

        # [6] Post-validation SpaCy (si activé et modèle chargé)
        if hasattr(self, '_spacy_nlp') and self._spacy_nlp:
            if field_name == "VESSEL" and not self._is_likely_vessel(cleaned):
                return False, ""
            if field_name in ["PORT_OF_LOADING", "PORT_OF_DISCHARGE"] and not self._is_likely_place(cleaned):
                return False, ""

        return True, cleaned.strip()

    def _is_likely_vessel(self, text: str) -> bool:
        """Vérifie si le texte ressemble à un nom de navire (ex. : contient préfixes courants)."""
        vessel_prefixes = ["MSC", "MAERSK", "CMA", "HAPAG", "EVER", "COSCO", "ONE", "YANG", "ZIM"]
        text_up = text.upper()
        return any(prefix in text_up for prefix in vessel_prefixes) or len(text.split()) > 1

    def _is_likely_place(self, text: str) -> bool:
        """Vérifie si le texte ressemble à un lieu (ex. : contient virgules ou mots géographiques)."""
        if "," in text or any(word in text.upper() for word in ["PORT", "CITY", "HARBOR"]):
            return True
        # SpaCy NER si disponible
        if self._spacy_nlp:
            doc = self._spacy_nlp(text)
            return any(ent.label_ in ["GPE", "LOC"] for ent in doc.ents)
        return True  # Flexible, accepte par défaut

    # ── Fusion : max confiance pondérée ───────────────────────────────────────

    @staticmethod
    def _fuse_candidates(candidates: list[dict]) -> dict[str, dict]:
        """
        Pour chaque champ, utilise vote majoritaire pondéré par confiance.
        Si égalité, prend la valeur avec confiance max.
        """
        from collections import defaultdict
        votes: dict[str, dict[str, float]] = defaultdict(dict)
        for c in candidates:
            f = c["field"]
            val = c["value"]
            conf = c["confidence"]
            votes[f][val] = votes[f].get(val, 0) + conf

        best: dict[str, dict] = {}
        for f, val_confs in votes.items():
            # Vote majoritaire : valeur avec somme conf max
            best_val = max(val_confs, key=val_confs.get)
            # Récupérer le candidat réel portant cette valeur (conf individuelle max).
            # NE PAS comparer à best_conf=somme des votes : si la valeur a été votée
            # par ≥2 passes, la somme n'égale aucune confiance individuelle → le champ
            # serait perdu silencieusement (bug historique : couverture effondrée).
            cands_fv = [
                c for c in candidates
                if c["field"] == f and c["value"] == best_val
            ]
            if cands_fv:
                best[f] = max(cands_fv, key=lambda c: c["confidence"])
        return best

    def _gate_by_label_presence(
        self,
        best:        dict[str, dict],
        label_zones: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Ne conserve un champ scalaire que si son label a été repéré dans le document
        (detect_label_zones). Coupe les faux positifs sur champs ABSENTS (ex. ETD,
        ISSUE_PLACE, LC_REFERENCE fabriqués à partir d'une valeur voisine).

        Sur les vrais documents (labels standards présents dans FIELD_SYNONYMS) le
        signal est fiable. Si aucune zone n'est détectée (label_zones vide), on ne
        filtre pas (pas de signal). N'affecte PAS evaluate_release2_models.py qui
        évalue le modèle brut sans passer par cet extracteur.
        """
        if not label_zones:
            return best
        return {f: v for f, v in best.items() if f in label_zones}

    # ── Pipeline complet ───────────────────────────────────────────────────────

    def _run_full_pipeline(
        self,
        image:  Any,
        words:  list[str],
        bboxes: list[list[int]],
    ) -> dict:
        """
        4 passes + 4 couches sur une page.
        Retourne une réponse COMPLÈTE couvrant tous les EXPECTED_FIELDS.
        """
        # ── Couche 1 : détection des labels ─────────────────────────────────
        label_zones = (
            detect_label_zones(words, bboxes, self.FIELD_SYNONYMS)
            if self.FIELD_SYNONYMS else {}
        )

        # ── Pass 0 : full page (fenêtre glissante) ───────────────────────────
        fp_cands = self._predict_full_page(image, words, bboxes)

        # ── Pass 1 : blocs horizontaux (left_column / right_column) ─────────
        block_cands = self._predict_blocks(image, words, bboxes, label_zones)

        # ── Pass 4 : zones adaptatives (une zone par label détecté) ─────────
        adaptive_cands = self._predict_adaptive_zones(image, words, bboxes, label_zones)

        # ── Fusion de toutes les passes ──────────────────────────────────────
        all_cands = fp_cands + block_cands + adaptive_cands

        scalar_cands = [c for c in all_cands if c["field"] not in self.TABLE_FIELDS]
        table_cands  = [c for c in all_cands if c["field"] in  self.TABLE_FIELDS]

        best = self._fuse_candidates(scalar_cands)
        best = self._gate_by_label_presence(best, label_zones)
        fields: dict[str, str]  = {f: v["value"]      for f, v in best.items()}
        confidences: dict[str, float] = {f: v["confidence"] for f, v in best.items()}
        sources: dict[str, str] = {
            f: f"{v['method']}/{v['segment']}" for f, v in best.items()
        }

        # ── Couche 3 : tableau (multi-valeurs) ───────────────────────────────
        table_rows: list[dict[str, str]] = []
        if self.ROW_FIELDS and table_cands:
            raw_rows = build_table_rows(
                table_cands,
                row_anchor_field=self.ROW_ANCHOR_FIELD,
                row_fields=self.ROW_FIELDS,
            )
            table_rows = (
                validate_table_rows_semantic(raw_rows, self.ROW_VALIDATORS)
                if self.ROW_VALIDATORS else raw_rows
            )

        # ── Couche 4 : champs calculés + validation arithmétique ─────────────
        computed_validations: list[dict] = []
        if self.COMPUTED_SPECS and table_rows:
            fields, computed_validations = validate_computed_fields(
                fields, table_rows, self.COMPUTED_SPECS
            )

        # ── Réponse complète : tous les EXPECTED_FIELDS ──────────────────────
        field_status: dict[str, str] = {
            f: ("found" if f in fields else "not_found")
            for f in self.EXPECTED_FIELDS
        }
        n_found = sum(1 for s in field_status.values() if s == "found")

        return {
            # Champs scalaires extraits
            "fields":               fields,
            "field_status":         field_status,
            "field_confidences":    confidences,
            "field_source":         sources,
            # Tableau structuré
            "tables":               {self.TABLE_NAME: table_rows} if table_rows else {},
            "table_rows":           table_rows,
            # Validations arithmétiques
            "computed_validations": computed_validations,
            # Méta
            "label_zones_found":    list(label_zones.keys()),
            "coverage":             f"{n_found}/{len(self.EXPECTED_FIELDS)}",
        }

    # ── API publique : image PIL ───────────────────────────────────────────────

    def extract_from_pil_image(
        self,
        image:  Any,
        words:  list[str],
        bboxes: list[list[int]],
    ) -> dict:
        """Extraction depuis une image PIL avec tokens OCR pré-extraits (PDF scanné / image)."""
        if not words:
            return {
                "fields": {}, "field_status": {f: "not_found" for f in self.EXPECTED_FIELDS},
                "tables": {}, "table_rows": [], "computed_validations": [],
                "coverage": f"0/{len(self.EXPECTED_FIELDS)}",
            }
        return self._run_full_pipeline(image, words, bboxes)

    # ── API publique : PDF ─────────────────────────────────────────────────────

    def extract_from_pdf(
        self,
        pdf_path:   str | Path,
        input_type: str = "pdf_native",
    ) -> dict:
        """
        Extraction complète depuis PDF (natif ou scanné).

        Pour PDF natif  → extraction mots via pdfplumber (coordonnées précises)
        Pour PDF scanné → extraction mots via PyMuPDF (texte OCR de fitz)
        """
        import fitz
        from PIL import Image as PILImage

        pdf_path = Path(pdf_path)
        doc      = fitz.open(str(pdf_path))
        n_pages  = len(doc)

        all_fp_cands:    list[dict]       = []   # full_page candidates
        all_block_cands: list[dict]       = []   # bloc candidates
        all_table_cands: list[dict]       = []   # table candidates
        all_label_zones: dict[str, dict]  = {}

        for page_num in range(n_pages):
            page = doc[page_num]
            mat  = fitz.Matrix(150 / 72, 150 / 72)
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, 3
            )
            image = PILImage.fromarray(arr)

            # Extraction mots / boîtes
            if input_type == "pdf_native":
                try:
                    words, bboxes = extract_words_pdfplumber(pdf_path, page_num)
                except Exception as exc:
                    log.debug("pdfplumber p%d (%s) → fallback fitz", page_num, exc)
                    words, bboxes = extract_words_fitz(page, pix.width, pix.height)
            else:
                words, bboxes = extract_words_fitz(page, pix.width, pix.height)

            if not words:
                continue

            log.debug("page %d — %d mots", page_num + 1, len(words))

            # Détection des labels
            page_lz = (
                detect_label_zones(words, bboxes, self.FIELD_SYNONYMS)
                if self.FIELD_SYNONYMS else {}
            )
            all_label_zones.update(page_lz)

            # Pass 0 : full page
            fp = self._predict_full_page(image, words, bboxes)

            # Pass 1 : blocs horizontaux (left_column / right_column)
            bl = self._predict_blocks(image, words, bboxes, page_lz)

            # Pass 2 : zones adaptatives (ciblées par label détecté)
            ad = self._predict_adaptive_zones(image, words, bboxes, page_lz)

            # Répartir entre scalaires et table
            for c in fp + bl + ad:
                if c["field"] in self.TABLE_FIELDS:
                    all_table_cands.append(c)
                elif c["method"] == "full_page":
                    all_fp_cands.append(c)
                else:
                    all_block_cands.append(c)

        doc.close()

        # Fusion scalaires
        best = self._fuse_candidates(all_fp_cands + all_block_cands)
        best = self._gate_by_label_presence(best, all_label_zones)
        fields      = {f: v["value"]      for f, v in best.items()}
        confidences = {f: v["confidence"] for f, v in best.items()}
        sources     = {f: f"{v['method']}/{v['segment']}" for f, v in best.items()}

        # Tableau
        table_rows: list[dict[str, str]] = []
        if self.ROW_FIELDS and all_table_cands:
            raw_rows = build_table_rows(
                all_table_cands,
                row_anchor_field=self.ROW_ANCHOR_FIELD,
                row_fields=self.ROW_FIELDS,
            )
            table_rows = (
                validate_table_rows_semantic(raw_rows, self.ROW_VALIDATORS)
                if self.ROW_VALIDATORS else raw_rows
            )

        # Champs calculés
        computed_validations: list[dict] = []
        if self.COMPUTED_SPECS and table_rows:
            fields, computed_validations = validate_computed_fields(
                fields, table_rows, self.COMPUTED_SPECS
            )

        # Réponse complète
        field_status = {
            f: ("found" if f in fields else "not_found")
            for f in self.EXPECTED_FIELDS
        }
        n_found = sum(1 for s in field_status.values() if s == "found")

        return {
            "fields":               fields,
            "field_status":         field_status,
            "field_confidences":    confidences,
            "field_source":         sources,
            "tables":               {self.TABLE_NAME: table_rows} if table_rows else {},
            "table_rows":           table_rows,
            "computed_validations": computed_validations,
            "label_zones_found":    list(all_label_zones.keys()),
            "coverage":             f"{n_found}/{len(self.EXPECTED_FIELDS)}",
            "n_pages":              n_pages,
        }

    # ── Affichage console ──────────────────────────────────────────────────────

    def print_results(self, result: dict | None = None, fields: dict | None = None) -> None:
        """Affiche la réponse complète : champs, tableau, calculs, couverture."""
        if result is None and fields is not None:
            result = {"fields": fields, "field_status": {}, "tables": {},
                      "table_rows": [], "computed_validations": [], "coverage": "?"}
        if result is None:
            return

        flds      = result.get("fields", {})
        fstatus   = result.get("field_status", {})
        fsrc      = result.get("field_source", {})
        fconf     = result.get("field_confidences", {})
        rows      = result.get("table_rows", [])
        computed  = result.get("computed_validations", [])
        tables    = result.get("tables", {})
        coverage  = result.get("coverage", "?")
        lz        = result.get("label_zones_found", [])

        print(f"\n{'═' * 70}")
        print(f"  {self.__class__.__name__}  —  couverture {coverage}")
        print(f"{'─' * 70}")

        # Champs scalaires
        for fname in sorted(fstatus.keys()):
            label  = self.FIELD_NAME_MAP.get(fname, fname)
            status = fstatus.get(fname, "?")
            if status == "found":
                val  = flds.get(fname, "")
                conf = fconf.get(fname, 0.0)
                src  = fsrc.get(fname, "")
                print(f"  ✓ {label:<32} {val:<35} [{conf:.2f}] via {src}")
            else:
                print(f"  ✗ {label:<32} — non trouvé")

        # Tableau structuré
        for tname, trows in tables.items():
            if trows:
                print(f"\n  {tname} ({len(trows)} ligne(s)) :")
                for i, row in enumerate(trows, 1):
                    cells = " | ".join(f"{k}: {v}" for k, v in row.items())
                    print(f"    [{i}] {cells}")

        # Validations arithmétiques
        if computed:
            print(f"\n  Validations arithmétiques :")
            icons = {"ok": "✓", "mismatch": "⚠", "filled": "→"}
            for v in computed:
                sym = icons.get(v["status"], "?")
                if v["status"] == "mismatch":
                    print(f"    {sym} {v['field']:<22} extrait={v['extracted']} "
                          f"calculé={v['computed']} écart={v['diff_pct']}%")
                elif v["status"] == "filled":
                    print(f"    {sym} {v['field']:<22} = {v['computed']} (calculé depuis tableau)")
                else:
                    print(f"    {sym} {v['field']:<22} = {v['computed']} (cohérent)")

        if lz:
            print(f"\n  Labels détectés : {', '.join(lz)}")

        print(f"{'═' * 70}\n")
