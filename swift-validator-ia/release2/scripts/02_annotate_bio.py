"""
02_annotate_bio.py
==================
Étape 2 — Auto-annotation IOB2 : tokens OCR → labels B-FIELD / I-FIELD / O

Stratégie de matching (par ordre de priorité) :
  [1] Bbox overlap + filtrage valeur : centre du token dans la bbox annotation,
                                       puis on ne garde que les tokens qui
                                       correspondent au TEXTE de la valeur
                                       (évite le label-leakage).
  [2] Fuzzy text (fallback)          : RapidFuzz sur la séquence de tokens,
                                       uniquement si le texte est distinctif
                                       (longueur >= 3 mots OU score >= 90).

Améliorations v2 :
  - filter_to_value_tokens() : filtre les tokens bbox pour ne garder que ceux
    qui correspondent à la valeur annotée (élimine les préfixes "INVOICE NO:",
    "SHIPPER:", etc. inclus par erreur dans le bbox).
  - is_field_label_token() : détecte les tokens qui sont des étiquettes de
    champ typiques et les exclut même s'ils sont dans la bbox.
  - Seuil fuzzy relevé à 88 + vérification longueur minimale.
  - Filtre de couverture minimale : les docs < MIN_DOC_COVERAGE sont rejetés.
"""

from __future__ import annotations

import sys
import re
import json
import logging
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

from rapidfuzz import fuzz
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = RELEASE2_ROOT / "data"
RAW_ROOT      = DATA_ROOT / "raw"
PROC_ROOT     = DATA_ROOT / "processed"


def _configure_paths(data_root_override: str | None) -> None:
    global DATA_ROOT, RAW_ROOT, PROC_ROOT, DOC_ANNOTATION_DIRS
    if data_root_override:
        DATA_ROOT = Path(data_root_override).resolve()
        RAW_ROOT  = DATA_ROOT / "raw"
        PROC_ROOT = DATA_ROOT / "processed"
        DOC_ANNOTATION_DIRS = {
            "bl":           RAW_ROOT / "bl"           / "annotations",
            "invoice":      RAW_ROOT / "invoice"      / "annotations",
            "packing_list": RAW_ROOT / "packing_list" / "annotations",
        }

LOGS_DIR = RELEASE2_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "02_annotate_bio.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Seuils ────────────────────────────────────────────────────────────────────

BBOX_CENTER_MARGIN  = 10   # tolérance (unités 0-1000) — légèrement augmentée
FUZZY_MATCH_THRESH  = 88   # seuil RapidFuzz relevé (v1 = 82)
MIN_FIELD_CHARS     = 2    # ignorer les valeurs de champ trop courtes
MIN_VALUE_WORDS     = 1    # min mots pour déclencher le fuzzy fallback
FUZZY_MIN_WORDS     = 2    # en-dessous, exiger score >= FUZZY_STRICT_THRESH
FUZZY_STRICT_THRESH = 95   # seuil pour valeurs courtes (1 mot) → évite faux positifs
MIN_DOC_COVERAGE    = 0.05 # rejeter les docs avec < 5 % de tokens annotés
VALUE_MATCH_MIN_SCORE = 55 # seuil pour filter_to_value_tokens (ratio simple)


# ─────────────────────────────────────────────────────────────────────────────
# Mots-clés d'étiquettes de champs à exclure (pour chaque entité)
# Ces tokens sont des libellés imprimés SUR le document, pas des valeurs.
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_KEYWORDS: dict[str, set[str]] = {
    "BL_NUMBER":         {"B/L", "BL", "NO", "NUMBER", "REF", "REFERENCE",
                          "BILL", "LADING", "No.", "B.L.", "BLNO", "MBL"},
    "SHIPPER":           {"SHIPPER", "EXPORTER", "SHIPPER/EXPORTER",
                          "SHIPPED", "BY", "FROM"},
    "CONSIGNEE":         {"CONSIGNEE", "IMPORTER", "TO", "CONSIGNE"},
    "NOTIFY_PARTY":      {"NOTIFY", "PARTY", "ALSO", "N/P", "NOTIFICATION",
                          "NOTIFIER", "ADVISE"},
    "VESSEL":            {"VESSEL", "SHIP", "M/V", "SS", "MV", "VESSEL/VOYAGE"},
    "VOYAGE":            {"VOYAGE", "VOY", "VOY.", "VYG", "VOYAGE/FLIGHT"},
    "PORT_OF_LOADING":   {"PORT", "LOADING", "POL", "ORIGIN", "FROM",
                          "DEPARTURE", "OF"},
    "PORT_OF_DISCHARGE": {"PORT", "DISCHARGE", "POD", "DESTINATION",
                          "TO", "ARRIVAL", "OF"},
    "PLACE_OF_RECEIPT":  {"PLACE", "RECEIPT", "POR", "OF", "ORIGIN"},
    "PLACE_OF_DELIVERY": {"PLACE", "DELIVERY", "POD", "OF", "DESTINATION"},
    "ETD":               {"ETD", "DEPARTURE", "DATE", "ESTIMATED"},
    "ETA":               {"ETA", "ARRIVAL", "DATE", "ESTIMATED"},
    "ISSUE_DATE":        {"DATE", "ISSUED", "ISSUE", "ISSUANCE", "OF"},
    "ISSUE_PLACE":       {"PLACE", "ISSUE", "ISSUED", "AT", "OF"},
    "CONTAINER_NO":      {"CONTAINER", "CONT", "CNT", "NO", "NUMBER", "CTR"},
    "CONTAINER_TYPE":    {"CONTAINER", "TYPE", "SIZE", "EQUIPMENT"},
    "SEAL_NO":           {"SEAL", "SEAL/S", "NO", "NUMBER"},
    "COMMODITY":         {"COMMODITY", "DESCRIPTION", "GOODS", "CARGO",
                          "PARTICULARS", "OF"},
    "GROSS_WEIGHT":      {"GROSS", "WEIGHT", "KGS", "KG", "LBS", "G/W", "GW"},
    "NET_WEIGHT":        {"NET", "WEIGHT", "KGS", "KG", "N/W", "NW"},
    "MEASUREMENT":       {"MEASUREMENT", "CBM", "M3", "VOLUME", "MEAS"},
    "PACKAGES":          {"PACKAGES", "PKGS", "NO", "NUMBER", "OF",
                          "PIECES", "PCS", "UNITS"},
    "FREIGHT_TERMS":     {"FREIGHT", "TERMS", "CHARGES", "COLLECT", "PREPAID"},
    "SERVICE_TYPE":      {"SERVICE", "TYPE", "MODE", "TRANSPORT"},
    "NUMBER_OF_ORIGINALS":{"ORIGINALS", "NUMBER", "OF", "B/L", "BILLS"},
    # Invoice
    "INV_NUMBER":        {"INVOICE", "INV", "NO", "NUMBER", "REF"},
    "SELLER":            {"SELLER", "EXPORTER", "SHIPPER", "SOLD", "BY",
                          "SUPPLIER", "SHIPPER/EXPORTER"},
    "BUYER":             {"BUYER", "IMPORTER", "CONSIGNEE", "SOLD", "TO",
                          "SHIP", "SHIPPED"},
    "GOODS_DESC":        {"DESCRIPTION", "GOODS", "COMMODITY", "ITEM",
                          "PARTICULARS", "OF"},
    "HS_CODE":           {"HS", "CODE", "HTS", "TARIFF", "HS/HTS"},
    "QUANTITY":          {"QUANTITY", "QTY", "UNITS", "PCS", "PIECES", "NO"},
    "PACKAGE_UNIT":      {"PACKAGE", "UNIT", "PKG", "PKGS", "TYPE"},
    "UNIT_PRICE":        {"UNIT", "PRICE", "UP", "RATE", "PER"},
    "CURRENCY":          {"CURRENCY", "CCY", "CUR"},
    "TOTAL_AMOUNT":      {"TOTAL", "AMOUNT", "VALUE", "TOT", "GRAND", "INVOICE"},
    "INCOTERM":          {"INCOTERM", "DELIVERY", "TERMS", "TRADE"},
    "PAYMENT_TERMS":     {"PAYMENT", "TERMS", "PAY"},
    "LC_REF":            {"L/C", "LC", "CREDIT", "LETTER", "REF"},
    "BL_REF":            {"B/L", "BL", "REF", "REFERENCE"},
    # Packing list
    "PL_NUMBER":         {"PACKING", "LIST", "PL", "NO", "NUMBER", "P/L"},
    "EXPORTER":          {"EXPORTER", "SELLER", "SHIPPER", "SHIPPED", "BY"},
    "TOTAL_PACKAGES":    {"TOTAL", "PACKAGES", "PKGS", "NO", "NUMBER", "OF"},
    "TOTAL_CBM":         {"TOTAL", "CBM", "VOLUME", "M3", "MEASUREMENT"},
    "MARKS_NUMBERS":     {"MARKS", "NUMBERS", "MARKS/NOS", "M/N"},
    "DIMENSIONS":        {"DIMENSIONS", "SIZE", "DIM"},
    "INV_REF":           {"INVOICE", "INV", "REF", "REFERENCE"},
}

# Normalisation des keywords (majuscules, sans accents)
_LABEL_KEYWORDS_NORM: dict[str, set[str]] = {
    entity: {k.upper().strip(".:/ ") for k in kws}
    for entity, kws in _LABEL_KEYWORDS.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# Schéma de labels par type de document (inchangé)
# ─────────────────────────────────────────────────────────────────────────────

BL_FIELD_LABELS: dict[str, str] = {
    "bl_number":          "BL_NUMBER",
    "bl_n":               "BL_NUMBER",
    "shipper":            "SHIPPER",
    "shipper_name":       "SHIPPER",
    "consignee":          "CONSIGNEE",
    "consignee_(to_order_of)": "CONSIGNEE",
    "consignee_name":     "CONSIGNEE",
    "notify_party":       "NOTIFY_PARTY",
    "vessel":             "VESSEL",
    "voyage":             "VOYAGE",
    "port_of_loading":    "PORT_OF_LOADING",
    "port_of_discharge":  "PORT_OF_DISCHARGE",
    "place_of_receipt":   "PLACE_OF_RECEIPT",
    "place_of_delivery":  "PLACE_OF_DELIVERY",
    "etd":                "ETD",
    "eta":                "ETA",
    "issue_date":         "ISSUE_DATE",
    "issue_place":        "ISSUE_PLACE",
    "container_number":   "CONTAINER_NO",
    "container_type":     "CONTAINER_TYPE",
    "seal_number":        "SEAL_NO",
    "commodity":          "COMMODITY",
    "gross_weight_kg":    "GROSS_WEIGHT",
    "measurement":        "MEASUREMENT",
    "packages":           "PACKAGES",
    "freight_terms":      "FREIGHT_TERMS",
    "service_type":       "SERVICE_TYPE",
    "number_of_originals":"NUMBER_OF_ORIGINALS",
}

INVOICE_FIELD_LABELS: dict[str, str] = {
    "invoice_number":    "INV_NUMBER",
    "invoice_date":      "ISSUE_DATE",
    "lc_reference":      "LC_REF",
    "bl_reference":      "BL_REF",
    "seller_name":       "SELLER",
    "seller":            "SELLER",
    "buyer_name":        "BUYER",
    "buyer":             "BUYER",
    "vessel":            "VESSEL",
    "voyage":            "VOYAGE",
    "port_of_loading":   "PORT_OF_LOADING",
    "port_of_discharge": "PORT_OF_DISCHARGE",
    "goods_description": "GOODS_DESC",
    "hs_code":           "HS_CODE",
    "quantity":          "QUANTITY",
    "package_unit":      "PACKAGE_UNIT",
    "gross_weight_kg":   "GROSS_WEIGHT",
    "unit_price":        "UNIT_PRICE",
    "currency":          "CURRENCY",
    "total_amount":      "TOTAL_AMOUNT",
    "incoterm":          "INCOTERM",
    "payment_terms":     "PAYMENT_TERMS",
}

PL_FIELD_LABELS: dict[str, str] = {
    "pl_number":         "PL_NUMBER",
    "pl_date":           "ISSUE_DATE",
    "inv_reference":     "INV_REF",
    "bl_reference":      "BL_REF",
    "exporter_name":     "EXPORTER",
    "exporter":          "EXPORTER",
    "consignee_name":    "CONSIGNEE",
    "consignee":         "CONSIGNEE",
    "vessel":            "VESSEL",
    "voyage":            "VOYAGE",
    "port_of_loading":   "PORT_OF_LOADING",
    "port_of_discharge": "PORT_OF_DISCHARGE",
    "container_number":  "CONTAINER_NO",
    "container_type":    "CONTAINER_TYPE",
    "seal_number":       "SEAL_NO",
    "goods_description": "GOODS_DESC",
    "package_unit":      "PACKAGE_UNIT",
    "total_packages":    "TOTAL_PACKAGES",
    "net_weight_kg":     "NET_WEIGHT",
    "gross_weight_kg":   "GROSS_WEIGHT",
    "total_cbm":         "TOTAL_CBM",
    "marks_numbers":     "MARKS_NUMBERS",
    "dimensions":        "DIMENSIONS",
}

DOC_FIELD_LABELS = {
    "bl":           BL_FIELD_LABELS,
    "invoice":      INVOICE_FIELD_LABELS,
    "packing_list": PL_FIELD_LABELS,
}

DOC_ANNOTATION_DIRS = {
    "bl":           RAW_ROOT / "bl"           / "annotations",
    "invoice":      RAW_ROOT / "invoice"      / "annotations",
    "packing_list": RAW_ROOT / "packing_list" / "annotations",
}

DOC_ANNOTATION_PREFIX = {
    "bl":           "bl_",
    "invoice":      "inv_",
    "packing_list": "pl_",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires texte
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.upper()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def value_to_words(value: str) -> list[str]:
    norm = normalize_text(value)
    return [w for w in norm.split() if w]


def is_field_label_token(text: str, label: str) -> bool:
    """
    Retourne True si le token est un libellé de champ typique pour ce label.
    Ex : "INVOICE" est un libellé pour INV_NUMBER, pas une valeur.
    """
    norm = normalize_text(text).strip(".:/ ")
    kws  = _LABEL_KEYWORDS_NORM.get(label, set())
    return norm in kws


# ─────────────────────────────────────────────────────────────────────────────
# Matching 1 — Bbox overlap
# ─────────────────────────────────────────────────────────────────────────────

def annot_bbox_to_norm(
    ann_bbox: list,
    page_w_pts: float,
    page_h_pts: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = ann_bbox
    x0 = max(0, min(1000, int(x           / page_w_pts * 1000)))
    y0 = max(0, min(1000, int(y           / page_h_pts * 1000)))
    x1 = max(0, min(1000, int((x + w)     / page_w_pts * 1000)))
    y1 = max(0, min(1000, int((y + h)     / page_h_pts * 1000)))
    return x0, y0, x1, y1


def token_center_in_bbox(
    tok_bbox: list,
    ann_norm: tuple,
    margin: int = BBOX_CENTER_MARGIN,
) -> bool:
    cx = (tok_bbox[0] + tok_bbox[2]) // 2
    cy = (tok_bbox[1] + tok_bbox[3]) // 2
    ax0, ay0, ax1, ay1 = ann_norm
    return (ax0 - margin <= cx <= ax1 + margin) and (ay0 - margin <= cy <= ay1 + margin)


def match_by_bbox(
    page_tokens: list[dict],
    ann_bbox_norm: tuple,
) -> list[int]:
    indices = [
        i for i, t in enumerate(page_tokens)
        if token_center_in_bbox(t["bbox"], ann_bbox_norm)
    ]
    indices.sort(key=lambda i: (page_tokens[i]["bbox"][1], page_tokens[i]["bbox"][0]))
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# NOUVEAU — Filtrage valeur-seulement après bbox match
# ─────────────────────────────────────────────────────────────────────────────

def filter_to_value_tokens(
    candidate_indices: list[int],
    page_tokens: list[dict],
    ann_value: str,
    label: str,
) -> list[int]:
    """
    Parmi les tokens trouvés par bbox match, ne garde que ceux qui correspondent
    au TEXTE de la valeur annotée (ann_value).

    Algorithme :
      1. Normaliser la valeur et les candidats.
      2. Sliding window sur les candidats pour trouver la fenêtre qui correspond
         le mieux à la valeur (fuzz.ratio).
      3. Exclure en supplément les tokens qui sont des étiquettes connues du label.

    Cela élimine les "INVOICE NO:" / "SHIPPER:" / "CARRIER:" qui se retrouvaient
    dans les bboxes et contaminaient les labels IOB2.
    """
    if not candidate_indices or not ann_value:
        return candidate_indices

    value_words = value_to_words(ann_value)
    if not value_words:
        return candidate_indices

    n_val = len(value_words)
    cand_texts = [normalize_text(page_tokens[i]["text"]) for i in candidate_indices]
    n_cand = len(cand_texts)

    if n_val >= n_cand:
        # Valeur aussi longue que les candidats → pas de fenêtrage, garder tout
        # mais quand même filtrer les labels-keyword en début
        filtered = _strip_leading_label_tokens(candidate_indices, page_tokens, label)
        return filtered if filtered else candidate_indices

    # Sliding window : trouver le meilleur alignement de n_val tokens
    best_score = 0
    best_start = 0
    target = " ".join(value_words)

    for start in range(n_cand - n_val + 1):
        window = " ".join(cand_texts[start: start + n_val])
        score  = fuzz.ratio(window, target)
        if score > best_score:
            best_score = score
            best_start = start

    if best_score >= VALUE_MATCH_MIN_SCORE:
        selected = candidate_indices[best_start: best_start + n_val]
    else:
        # Mauvais alignement : garder les candidats mais enlever les labels-keyword
        selected = candidate_indices

    # Deuxième passe : retirer les tokens qui sont des étiquettes de champ typiques
    selected = _strip_leading_label_tokens(selected, page_tokens, label)
    return selected if selected else candidate_indices


def _strip_leading_label_tokens(
    indices: list[int],
    page_tokens: list[dict],
    label: str,
) -> list[int]:
    """
    Supprime les tokens en TÊTE de la liste qui sont des étiquettes de champ
    connues pour ce label. Ne supprime PAS depuis la fin (valeur possible).
    """
    result = list(indices)
    while result and is_field_label_token(page_tokens[result[0]]["text"], label):
        result = result[1:]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Matching 2 — Fuzzy text (fallback amélioré)
# ─────────────────────────────────────────────────────────────────────────────

def match_by_fuzzy(
    page_tokens: list[dict],
    value_words: list[str],
    threshold: int = FUZZY_MATCH_THRESH,
    label: str = "",
) -> list[int] | None:
    """
    Cherche la séquence de tokens la plus proche de value_words.
    Améliorations v2 :
      - Les valeurs trop courtes (1 mot) utilisent un seuil plus strict.
      - Les tokens étiquettes connus sont pénalisés.
    """
    n = len(value_words)
    if n < MIN_VALUE_WORDS:
        return None

    # Pour les valeurs courtes, seuil relevé pour éviter les faux positifs
    effective_thresh = FUZZY_STRICT_THRESH if n < FUZZY_MIN_WORDS else threshold

    tok_texts = [normalize_text(t["text"]) for t in page_tokens]
    target    = " ".join(value_words)
    best_score = 0
    best_start = -1

    for start in range(len(tok_texts) - n + 1):
        # Ignorer les fenêtres qui commencent par une étiquette connue
        if label and is_field_label_token(page_tokens[start]["text"], label):
            continue
        window = " ".join(tok_texts[start: start + n])
        score  = fuzz.token_set_ratio(window, target)
        if score > best_score:
            best_score = score
            best_start = start

    if best_score >= effective_thresh and best_start >= 0:
        return list(range(best_start, best_start + n))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Annotation d'un document
# ─────────────────────────────────────────────────────────────────────────────

def annotate_document(
    ocr_data:   dict,
    annot_data: dict,
    field_labels: dict[str, str],
) -> dict:
    """
    Aligne les champs sur les tokens OCR avec labels IOB2.
    Version 2 : filtre les tokens pour n'annoter que la VALEUR, pas le libellé.
    """
    pages_bio = []
    all_label_counts: dict[str, int] = defaultdict(int)

    for page_ocr in ocr_data["pages"]:
        page_num   = page_ocr["page"]
        tokens     = page_ocr["tokens"]
        page_w_pts = page_ocr["width_pts"]
        page_h_pts = page_ocr["height_pts"]
        labels     = ["O"] * len(tokens)
        assigned: set[int] = set()

        # ── Priorité 1 : bbox_annotations ─────────────────────────────────────
        for ann in annot_data.get("bbox_annotations", []):
            if ann.get("page", 1) != page_num:
                continue

            field_key = ann["field"].lower().strip()
            label     = field_labels.get(field_key)
            if label is None:
                short = re.split(r"[_(]", field_key)[0].strip("_")
                label = field_labels.get(short)
            if label is None:
                continue

            ann_value = str(ann.get("value", "")).strip()
            if len(ann_value) < MIN_FIELD_CHARS:
                continue

            ann_norm    = annot_bbox_to_norm(ann["bbox"], page_w_pts, page_h_pts)
            bbox_matched = match_by_bbox(tokens, ann_norm)

            if bbox_matched:
                # ← NOUVEAU : filtrer pour ne garder que les tokens valeur
                matched = filter_to_value_tokens(bbox_matched, tokens, ann_value, label)
            else:
                # Fallback fuzzy restreint à la zone bbox (±20% de la page)
                value_words = value_to_words(ann_value)
                matched = []
                if value_words:
                    # Chercher uniquement dans la région proche de la bbox
                    ax0, ay0, ax1, ay1 = ann_norm
                    region_margin = 50  # ±50 unités autour de la bbox
                    region_indices = [
                        i for i, t in enumerate(tokens)
                        if (ay0 - region_margin <= (t["bbox"][1] + t["bbox"][3]) // 2
                            <= ay1 + region_margin)
                    ]
                    if region_indices:
                        region_tokens = [tokens[i] for i in region_indices]
                        local_match = match_by_fuzzy(region_tokens, value_words,
                                                     label=label)
                        if local_match:
                            matched = [region_indices[j] for j in local_match]
                    # Fallback global uniquement pour valeurs longues et distinctives
                    if not matched and len(value_words) >= 3:
                        global_match = match_by_fuzzy(tokens, value_words, label=label)
                        if global_match:
                            matched = global_match

            for rank, idx in enumerate(matched):
                if idx in assigned:
                    continue
                labels[idx] = f"B-{label}" if rank == 0 else f"I-{label}"
                assigned.add(idx)
                all_label_counts[label] += 1

        # ── Priorité 2 : champs textuels sans bbox ────────────────────────────
        for field_key, label in field_labels.items():
            value = str(annot_data.get("fields", {}).get(field_key, "")).strip()
            if len(value) < MIN_FIELD_CHARS:
                continue

            value_words = value_to_words(value)
            if not value_words:
                continue

            already_annotated = any(
                f"B-{label}" in labels[i] or f"I-{label}" in labels[i]
                for i in range(len(labels))
            )
            if already_annotated:
                continue

            # Exiger au moins 3 mots pour le fuzzy global (évite les faux positifs)
            if len(value_words) < 3:
                continue

            matched = match_by_fuzzy(tokens, value_words, label=label)
            if matched:
                for rank, idx in enumerate(matched):
                    if idx in assigned:
                        continue
                    labels[idx] = f"B-{label}" if rank == 0 else f"I-{label}"
                    assigned.add(idx)
                    all_label_counts[label] += 1

        pages_bio.append({
            "page":       page_num,
            "image_path": page_ocr["image_path"],
            "tokens":     [t["text"]  for t in tokens],
            "bboxes":     [t["bbox"]  for t in tokens],
            "labels":     labels,
        })

    total_tokens = sum(len(p["tokens"]) for p in pages_bio)
    annotated    = sum(1 for p in pages_bio for lbl in p["labels"] if lbl != "O")
    coverage     = round(annotated / total_tokens, 3) if total_tokens > 0 else 0.0

    label_set = sorted({
        lbl.split("-", 1)[1]
        for p in pages_bio for lbl in p["labels"]
        if lbl != "O"
    })

    return {
        "doc_id":              ocr_data["doc_id"],
        "doc_type":            ocr_data["doc_type"],
        "n_tokens":            total_tokens,
        "annotation_coverage": coverage,
        "label_set":           label_set,
        "label_counts":        dict(all_label_counts),
        "pages":               pages_bio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Résolution du fichier annotation
# ─────────────────────────────────────────────────────────────────────────────

def find_annotation_file(doc_id: str, doc_type: str) -> Path | None:
    ann_dir    = DOC_ANNOTATION_DIRS[doc_type]
    prefix_map = DOC_ANNOTATION_PREFIX[doc_type]

    candidate = ann_dir / f"{doc_id}.json"
    if candidate.exists():
        return candidate

    for other_prefix in ["bl_", "inv_", "pl_"]:
        if doc_id.startswith(other_prefix):
            stem       = doc_id[len(other_prefix):]
            new_doc_id = prefix_map + stem
            alt        = ann_dir / f"{new_doc_id}.json"
            if alt.exists():
                return alt

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def run_annotation(doc_types: list[str], overwrite: bool = False,
                   min_coverage: float = MIN_DOC_COVERAGE) -> dict:
    stats = {
        "total": 0, "processed": 0, "skipped": 0,
        "no_annotation": 0, "errors": 0, "low_coverage": 0,
        "coverage_sum": 0.0,
    }

    for doc_type in doc_types:
        proc_dir   = PROC_ROOT / doc_type
        field_lbls = DOC_FIELD_LABELS[doc_type]
        ocr_files  = sorted(proc_dir.glob("*_ocr.json"))
        stats["total"] += len(ocr_files)

        log.info("=" * 60)
        log.info("[%s] %d fichiers OCR", doc_type.upper(), len(ocr_files))

        for ocr_path in tqdm(ocr_files, desc=f"  BIO {doc_type}", unit="doc"):
            doc_id   = ocr_path.stem.replace("_ocr", "")
            bio_path = proc_dir / f"{doc_id}_bio.json"

            if bio_path.exists() and not overwrite:
                stats["skipped"] += 1
                continue

            try:
                ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
                ann_path = find_annotation_file(doc_id, doc_type)

                if ann_path is None:
                    log.warning("[%s] Annotation introuvable pour %s", doc_type, doc_id)
                    stats["no_annotation"] += 1
                    continue

                annot_data = json.loads(ann_path.read_text(encoding="utf-8"))
                bio_data   = annotate_document(ocr_data, annot_data, field_lbls)

                # ← NOUVEAU : rejeter les docs avec couverture trop faible
                if bio_data["annotation_coverage"] < min_coverage:
                    log.debug("[%s] Couverture trop faible (%.1f%%) : %s",
                              doc_type, bio_data["annotation_coverage"] * 100, doc_id)
                    stats["low_coverage"] += 1
                    # Sauvegarder quand même pour inspection, mais marquer
                    bio_data["rejected"] = True

                bio_path.write_text(
                    json.dumps(bio_data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                stats["processed"]    += 1
                stats["coverage_sum"] += bio_data["annotation_coverage"]

            except Exception as exc:
                import traceback
                log.error("Erreur sur %s : %s\n%s", doc_id, exc, traceback.format_exc())
                stats["errors"] += 1

        done = len(list(proc_dir.glob("*_bio.json")))
        log.info("[%s] %d BIO générés", doc_type.upper(), done)

    return stats


def print_summary(stats: dict) -> None:
    processed = stats["processed"]
    avg_cov   = stats["coverage_sum"] / processed if processed else 0.0

    log.info("")
    log.info("╔═══════════════════════════════════════════╗")
    log.info("║   RÉSUMÉ — 02_annotate_bio  v2            ║")
    log.info("╠═══════════════════════════════════════════╣")
    log.info("║  Total docs         : %4d               ║", stats["total"])
    log.info("║  Traités            : %4d               ║", processed)
    log.info("║  Déjà traités       : %4d  (skip)        ║", stats["skipped"])
    log.info("║  Sans annotation    : %4d               ║", stats["no_annotation"])
    log.info("║  Couverture faible  : %4d  (rejetés)     ║", stats["low_coverage"])
    log.info("║  Erreurs            : %4d               ║", stats["errors"])
    log.info("╠═══════════════════════════════════════════╣")
    log.info("║  Couverture moy.    : %.1f%%              ║", avg_cov * 100)
    log.info("╚═══════════════════════════════════════════╝")

    if avg_cov < 0.10:
        log.warning("⚠  Couverture très faible — vérifier le matching annotation↔tokens")
    elif avg_cov < 0.25:
        log.warning("⚠  Couverture faible — certains champs ne matchent pas")
    else:
        log.info("✓  Couverture annotation correcte")

    bio_files = list(PROC_ROOT.rglob("*_bio.json"))
    if bio_files:
        label_total: dict[str, int] = defaultdict(int)
        for p in bio_files[:100]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("rejected"):
                    continue
                for k, v in d.get("label_counts", {}).items():
                    label_total[k] += v
            except Exception:
                pass
        if label_total:
            log.info("\nDistribution labels (échantillon 100 docs valides) :")
            for lbl, cnt in sorted(label_total.items(), key=lambda x: -x[1])[:20]:
                log.info("  %-25s : %d", lbl, cnt)

    log.info("\nProchaine étape : python scripts/03_build_per_doctype_dataset.py")


def main():
    parser = argparse.ArgumentParser(
        description="02_annotate_bio v2 - tokens OCR + annotations → labels IOB2 (value-only)"
    )
    parser.add_argument(
        "--doc-types", nargs="+",
        choices=["bl", "invoice", "packing_list"],
        default=["bl", "invoice", "packing_list"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument(
        "--min-coverage", type=float, default=MIN_DOC_COVERAGE,
        help=f"Couverture minimale pour conserver un doc (défaut {MIN_DOC_COVERAGE:.0%})"
    )
    args = parser.parse_args()

    _configure_paths(args.data_root)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Script 02 v2 — Auto-annotation IOB2 (value-only)   ║")
    log.info("║  Bbox overlap + filtrage valeur + Fuzzy amélioré    ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    for dt in args.doc_types:
        n = len(list((PROC_ROOT / dt).glob("*_ocr.json")))
        log.info("  %-15s : %d OCR JSON", dt, n)

    stats = run_annotation(args.doc_types, overwrite=args.overwrite,
                           min_coverage=args.min_coverage)
    print_summary(stats)


if __name__ == "__main__":
    main()
