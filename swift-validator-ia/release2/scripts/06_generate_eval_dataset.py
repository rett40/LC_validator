"""
06_generate_eval_dataset.py
===========================
Génère un dataset d'évaluation indépendant avec un seed différent de l'entraînement.

Objectif :
  Tester la généralisation du modèle sur de NOUVEAUX contenus (nouveaux BL numbers,
  adresses, dates, etc.) dans les mêmes layouts que le test set courant.
  → Si F1 reste ~1.0, le modèle apprend des positions fixes (surajustement layout).
  → Si F1 baisse, le modèle dépend du contenu spécifique vu à l'entraînement.

Étapes intégrées (pipeline complet auto-contenu) :
  1. Détection automatique des layouts du test set courant (ou --layouts forcé)
  2. Génération des PDFs + annotations JSON via les générateurs existants
  3. Extraction OCR native (PyMuPDF) → tokens + bboxes normalisées 0-1000
  4. Annotation BIO (bbox matching token → champ)
  5. Construction JSONL prêts pour l'évaluation

Usage :
  cd swift-validator-ia
  python release2/scripts/06_generate_eval_dataset.py
  python release2/scripts/06_generate_eval_dataset.py --count-per-layout 40 --seed 9999
  python release2/scripts/06_generate_eval_dataset.py --doc-types bl --layouts bl-11 bl-15 bl-20
  python release2/scripts/06_generate_eval_dataset.py --all-layouts   # tous les layouts

Sortie :
  release2/data/complex/eval_independent/dataset_bl/test/data.jsonl
  release2/data/complex/eval_independent/dataset_invoice/test/data.jsonl
  release2/data/complex/eval_independent/dataset_packing_list/test/data.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict

import fitz          # PyMuPDF
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT   = Path(__file__).resolve().parent.parent
GEN_DIR         = RELEASE2_ROOT / "document_generator"
DATA_COMPLEX    = RELEASE2_ROOT / "data" / "complex"
DEFAULT_OUT_DIR = DATA_COMPLEX / "eval_independent"

sys.path.insert(0, str(GEN_DIR))   # pour importer les générateurs

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constantes pipeline ───────────────────────────────────────────────────────

RENDER_DPI         = 150
BBOX_SCALE         = 1000
BBOX_CENTER_MARGIN = 10
FUZZY_MATCH_THRESH = 88
FUZZY_STRICT_THRESH = 95
VALUE_MATCH_MIN_SCORE = 55
MIN_DOC_COVERAGE   = 0.05

# ── Maps champs JSON annotation → label NER ───────────────────────────────────

BL_FIELD_LABELS: dict[str, str] = {
    "bl_number":          "BL_NUMBER",
    "shipper_name":       "SHIPPER",
    "shipper":            "SHIPPER",
    "consignee_name":     "CONSIGNEE",
    "consignee":          "CONSIGNEE",
    "notify_party_name":  "NOTIFY_PARTY",
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
    "number_of_originals": "NUMBER_OF_ORIGINALS",
}

INVOICE_FIELD_LABELS: dict[str, str] = {
    "inv_number":        "INV_NUMBER",
    "invoice_number":    "INV_NUMBER",
    "issue_date":        "ISSUE_DATE",
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
    "pl_number":        "PL_NUMBER",
    "pl_date":          "ISSUE_DATE",
    "issue_date":       "ISSUE_DATE",
    "inv_reference":    "INV_REF",
    "bl_reference":     "BL_REF",
    "exporter_name":    "EXPORTER",
    "exporter":         "EXPORTER",
    "consignee_name":   "CONSIGNEE",
    "consignee":        "CONSIGNEE",
    "vessel":           "VESSEL",
    "voyage":           "VOYAGE",
    "port_of_loading":  "PORT_OF_LOADING",
    "port_of_discharge":"PORT_OF_DISCHARGE",
    "container_number": "CONTAINER_NO",
    "container_type":   "CONTAINER_TYPE",
    "seal_number":      "SEAL_NO",
    "goods_description":"GOODS_DESC",
    "package_unit":     "PACKAGE_UNIT",
    "total_packages":   "TOTAL_PACKAGES",
    "net_weight_kg":    "NET_WEIGHT",
    "gross_weight_kg":  "GROSS_WEIGHT",
    "total_cbm":        "TOTAL_CBM",
    "marks_numbers":    "MARKS_NUMBERS",
    "dimensions":       "DIMENSIONS",
}

DOC_FIELD_LABELS = {
    "bl":           BL_FIELD_LABELS,
    "invoice":      INVOICE_FIELD_LABELS,
    "packing_list": PL_FIELD_LABELS,
}

# ── Mots-clés d'étiquettes à exclure du BIO ──────────────────────────────────

_LABEL_KEYWORDS: dict[str, set[str]] = {
    "BL_NUMBER":         {"B/L", "BL", "NO", "NUMBER", "REF", "BILL", "LADING", "No."},
    "SHIPPER":           {"SHIPPER", "EXPORTER", "SHIPPED", "BY", "FROM"},
    "CONSIGNEE":         {"CONSIGNEE", "IMPORTER", "TO"},
    "NOTIFY_PARTY":      {"NOTIFY", "PARTY", "N/P", "NOTIFICATION"},
    "VESSEL":            {"VESSEL", "SHIP", "M/V", "SS", "MV"},
    "VOYAGE":            {"VOYAGE", "VOY", "VYG"},
    "PORT_OF_LOADING":   {"PORT", "LOADING", "POL", "OF"},
    "PORT_OF_DISCHARGE": {"PORT", "DISCHARGE", "POD", "OF"},
    "ETD":               {"ETD", "DEPARTURE", "DATE", "ESTIMATED"},
    "ETA":               {"ETA", "ARRIVAL", "DATE", "ESTIMATED"},
    "ISSUE_DATE":        {"DATE", "ISSUED", "ISSUE", "OF"},
    "ISSUE_PLACE":       {"PLACE", "ISSUE", "ISSUED", "AT"},
    "CONTAINER_NO":      {"CONTAINER", "CONT", "CNT", "NO", "NUMBER"},
    "CONTAINER_TYPE":    {"CONTAINER", "TYPE", "SIZE"},
    "SEAL_NO":           {"SEAL", "NO", "NUMBER"},
    "GROSS_WEIGHT":      {"GROSS", "WEIGHT", "KGS", "KG", "G/W"},
    "FREIGHT_TERMS":     {"FREIGHT", "TERMS", "CHARGES"},
    "INV_NUMBER":        {"INVOICE", "INV", "NO", "NUMBER"},
    "SELLER":            {"SELLER", "EXPORTER", "FROM"},
    "BUYER":             {"BUYER", "IMPORTER", "TO"},
    "GOODS_DESC":        {"DESCRIPTION", "GOODS", "CARGO"},
    "QUANTITY":          {"QUANTITY", "QTY", "PCS"},
    "TOTAL_AMOUNT":      {"TOTAL", "AMOUNT", "GRAND"},
    "PL_NUMBER":         {"PACKING", "LIST", "NO", "NUMBER"},
    "EXPORTER":          {"EXPORTER", "SHIPPER", "FROM"},
}
_LABEL_KEYWORDS_NORM = {
    lbl: {unicodedata.normalize("NFD", k).upper().strip(".:/ ") for k in kws}
    for lbl, kws in _LABEL_KEYWORDS.items()
}


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 0 : Détection des layouts du test set courant
# ─────────────────────────────────────────────────────────────────────────────

def detect_test_layouts(doc_type: str) -> list[str]:
    """Lit le JSONL de test courant et retourne les layout IDs présents."""
    ds_name = "dataset_packing_list" if doc_type == "packing_list" else f"dataset_{doc_type}"
    jsonl = DATA_COMPLEX / ds_name / "test" / "data.jsonl"
    if not jsonl.exists():
        return []
    layouts = set()
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            parts = rec["doc_id"].split("_")
            if len(parts) >= 2:
                layouts.add(parts[1].lower())   # e.g. "bl-11"
    return sorted(layouts)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : Génération PDFs + annotations
# ─────────────────────────────────────────────────────────────────────────────

def generate_docs(
    doc_type: str,
    layout_ids: list[str],
    count_per_layout: int,
    out_raw_dir: Path,
    seed: int,
) -> list[dict]:
    """
    Génère count_per_layout documents pour chaque layout_id spécifié.
    Retourne la liste des records (metadata + path).
    """
    if seed:
        random.seed(seed)
        try:
            from faker import Faker
            Faker.seed(seed)
        except ImportError:
            pass

    pdf_dir = out_raw_dir / "pdfs"
    ann_dir = out_raw_dir / "annotations"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    if doc_type == "bl":
        from layout_engine import BL_LAYOUTS
        from layout_engine.renderer_bl import BLParametricRenderer
        from generate_bl import build_bl_data

        all_cfg = {cfg.layout_id.lower(): cfg for cfg in BL_LAYOUTS}
        records = []
        doc_idx = 0

        for lid in layout_ids:
            cfg = all_cfg.get(lid)
            if cfg is None:
                log.warning("Layout BL inconnu : %s — ignore", lid)
                continue

            log.info("  Génération BL layout=%s × %d docs...", lid, count_per_layout)
            for _ in range(count_per_layout):
                doc_idx += 1
                data     = build_bl_data("msc")
                bl_num   = data["bl_number"]
                filename = f"bl_{lid}_{doc_idx:05d}_{bl_num}.pdf"
                pdf_path = str(pdf_dir / filename)

                renderer    = BLParametricRenderer(pdf_path, data, cfg)
                renderer.render()
                annotations = renderer.save()

                ann_path = ann_dir / filename.replace(".pdf", ".json")
                ann_path.write_text(
                    json.dumps({
                        "id": doc_idx, "filename": filename,
                        "layout_id": cfg.layout_id, "bl_number": bl_num,
                        "n_containers": data["n_containers"],
                        "fields": data,
                        "bbox_annotations": annotations,
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                records.append({
                    "doc_id":   filename.replace(".pdf", ""),
                    "pdf_path": pdf_path,
                    "ann_path": str(ann_path),
                    "layout_id": cfg.layout_id,
                })
        return records

    elif doc_type == "invoice":
        from layout_engine import INV_LAYOUTS
        from layout_engine.renderer_inv import InvoiceParametricRenderer
        from generate_invoice import build_invoice_data
        from generate_bl import build_bl_data as _build_bl

        all_cfg = {cfg.layout_id.lower(): cfg for cfg in INV_LAYOUTS}
        rng     = random.Random(seed)
        records = []
        doc_idx = 0

        for lid in layout_ids:
            cfg = all_cfg.get(lid)
            if cfg is None:
                log.warning("Layout Invoice inconnu : %s — ignore", lid)
                continue

            log.info("  Generation Invoice layout=%s x %d docs...", lid, count_per_layout)
            for _ in range(count_per_layout):
                doc_idx += 1
                bl_raw  = _build_bl("msc")
                # Construire le bl_record au format attendu par build_invoice_data
                bl_record = {
                    "id": doc_idx,
                    "fields": {
                        "bl_number":         bl_raw["bl_number"],
                        "shipper_name":      bl_raw["shipper"]["name"],
                        "consignee_name":    bl_raw["consignee"]["name"],
                        "vessel":            bl_raw["vessel"],
                        "voyage":            bl_raw["voyage"],
                        "port_of_loading":   bl_raw["port_of_loading"],
                        "port_of_discharge": bl_raw["port_of_discharge"],
                        "etd":               bl_raw["etd"],
                        "eta":               bl_raw["eta"],
                        "container_number":  bl_raw["container_number"],
                        "container_type":    bl_raw["container_type"],
                        "seal_number":       bl_raw["seal_number"],
                        "commodity":         bl_raw["commodity"],
                        "packages":          bl_raw["packages"]["description"],
                        "gross_weight_kg":   bl_raw["gross_weight_kg"],
                        "measurement":       bl_raw["measurement"],
                        "issue_date":        bl_raw.get("issue_date", ""),
                        "issue_place":       bl_raw.get("issue_place", ""),
                    },
                }
                data     = build_invoice_data(bl_record, rng, 0.0)
                inv_num  = data.get("inv_number", f"INV{doc_idx:05d}")
                filename = f"inv_{lid}_{doc_idx:05d}_{re.sub(r'[^A-Za-z0-9]', '', str(inv_num))}.pdf"
                pdf_path = str(pdf_dir / filename)

                renderer    = InvoiceParametricRenderer(pdf_path, data, cfg)
                renderer.render()
                annotations = renderer.save()

                ann_path = ann_dir / filename.replace(".pdf", ".json")
                ann_path.write_text(
                    json.dumps({
                        "id": doc_idx, "filename": filename,
                        "layout_id": cfg.layout_id,
                        "fields": data, "bbox_annotations": annotations,
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                records.append({
                    "doc_id":    filename.replace(".pdf", ""),
                    "pdf_path":  pdf_path,
                    "ann_path":  str(ann_path),
                    "layout_id": cfg.layout_id,
                })
        return records

    elif doc_type == "packing_list":
        from layout_engine import PL_LAYOUTS
        from layout_engine.renderer_pl import PLParametricRenderer
        from generate_packing_list import build_pl_data
        from generate_bl import build_bl_data as _build_bl

        all_cfg = {cfg.layout_id.lower(): cfg for cfg in PL_LAYOUTS}
        rng     = random.Random(seed)
        records = []
        doc_idx = 0

        for lid in layout_ids:
            cfg = all_cfg.get(lid)
            if cfg is None:
                log.warning("Layout PL inconnu : %s — ignore", lid)
                continue

            log.info("  Generation PL layout=%s x %d docs...", lid, count_per_layout)
            for _ in range(count_per_layout):
                doc_idx += 1
                bl_raw  = _build_bl("msc")
                bl_record = {
                    "id": doc_idx,
                    "fields": {
                        "bl_number":         bl_raw["bl_number"],
                        "shipper_name":      bl_raw["shipper"]["name"],
                        "consignee_name":    bl_raw["consignee"]["name"],
                        "vessel":            bl_raw["vessel"],
                        "voyage":            bl_raw["voyage"],
                        "port_of_loading":   bl_raw["port_of_loading"],
                        "port_of_discharge": bl_raw["port_of_discharge"],
                        "etd":               bl_raw["etd"],
                        "eta":               bl_raw["eta"],
                        "container_number":  bl_raw["container_number"],
                        "container_type":    bl_raw["container_type"],
                        "seal_number":       bl_raw["seal_number"],
                        "commodity":         bl_raw["commodity"],
                        "packages":          bl_raw["packages"]["description"],
                        "gross_weight_kg":   bl_raw["gross_weight_kg"],
                        "measurement":       bl_raw["measurement"],
                    },
                }
                data     = build_pl_data(bl_record, rng, 0.0)
                pl_num   = data.get("pl_number", f"PL{doc_idx:05d}")
                filename = f"pl_{lid}_{doc_idx:05d}_{re.sub(r'[^A-Za-z0-9]', '', str(pl_num))}.pdf"
                pdf_path = str(pdf_dir / filename)

                renderer    = PLParametricRenderer(pdf_path, data, cfg)
                renderer.render()
                annotations = renderer.save()

                ann_path = ann_dir / filename.replace(".pdf", ".json")
                ann_path.write_text(
                    json.dumps({
                        "id": doc_idx, "filename": filename,
                        "layout_id": cfg.layout_id,
                        "fields": data, "bbox_annotations": annotations,
                    }, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                records.append({
                    "doc_id":    filename.replace(".pdf", ""),
                    "pdf_path":  pdf_path,
                    "ann_path":  str(ann_path),
                    "layout_id": cfg.layout_id,
                })
        return records

    else:
        raise ValueError(f"doc_type inconnu : {doc_type}")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : OCR natif (PyMuPDF)
# ─────────────────────────────────────────────────────────────────────────────

def extract_ocr(pdf_path: str, img_dir: Path) -> dict | None:
    """
    Extrait les tokens depuis un PDF vectoriel via PyMuPDF.
    Rend une image PNG à 150 DPI pour LayoutLMv3.
    Retourne {page_w_pts, page_h_pts, image_path, tokens:[{text, bbox}]}.
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)

        # Rendu PNG
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_w, img_h = pix.width, pix.height

        stem = Path(pdf_path).stem
        img_path = img_dir / f"{stem}_page1.png"
        pix.save(str(img_path))

        pw, ph = page.rect.width, page.rect.height

        tokens = []
        for word_tuple in page.get_text("words"):
            x0, y0, x1, y1, text = word_tuple[:5]
            text = text.strip()
            if not text:
                continue
            sx, sy = img_w / pw, img_h / ph

            def nx(v, s, sz): return max(0, min(BBOX_SCALE, int(v * s / sz * BBOX_SCALE)))

            tokens.append({
                "text": text,
                "bbox": [
                    nx(x0, sx, img_w), nx(y0, sy, img_h),
                    nx(x1, sx, img_w), nx(y1, sy, img_h),
                ],
            })

        doc.close()
        return {
            "page_w_pts": pw,
            "page_h_pts": ph,
            "image_path": str(img_path),
            "tokens":     tokens,
        }
    except Exception as exc:
        log.warning("OCR échoué pour %s : %s", pdf_path, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : Annotation BIO
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.upper()).strip()


def _is_label_token(text: str, label: str) -> bool:
    norm = _normalize(text).strip(".:/ ")
    return norm in _LABEL_KEYWORDS_NORM.get(label, set())


def _annot_bbox_to_norm(ann_bbox: list, pw: float, ph: float) -> tuple:
    x, y, w, h = ann_bbox
    return (
        max(0, min(1000, int(x / pw * 1000))),
        max(0, min(1000, int(y / ph * 1000))),
        max(0, min(1000, int((x + w) / pw * 1000))),
        max(0, min(1000, int((y + h) / ph * 1000))),
    )


def _center_in_bbox(tok_bbox, ann_norm, margin=BBOX_CENTER_MARGIN) -> bool:
    cx = (tok_bbox[0] + tok_bbox[2]) // 2
    cy = (tok_bbox[1] + tok_bbox[3]) // 2
    ax0, ay0, ax1, ay1 = ann_norm
    return (ax0 - margin <= cx <= ax1 + margin) and (ay0 - margin <= cy <= ay1 + margin)


def _filter_value_tokens(indices, tokens, value, label):
    """Garde uniquement les tokens qui correspondent au texte de la valeur."""
    if not indices or not value:
        return indices
    value_words = _normalize(value).split()
    if not value_words:
        return indices
    n_val  = len(value_words)
    n_cand = len(indices)
    if n_val >= n_cand:
        return [i for i in indices if not _is_label_token(tokens[i]["text"], label)]

    cand_texts = [_normalize(tokens[i]["text"]) for i in indices]
    target = " ".join(value_words)
    best_score, best_start = 0, 0
    for start in range(n_cand - n_val + 1):
        window = " ".join(cand_texts[start:start + n_val])
        try:
            from rapidfuzz import fuzz as rfuzz
            score = rfuzz.ratio(window, target)
        except ImportError:
            # Fallback: simple char overlap
            score = 100 * sum(c in target for c in window) / max(len(window), 1)
        if score > best_score:
            best_score, best_start = score, start

    if best_score >= VALUE_MATCH_MIN_SCORE:
        return indices[best_start: best_start + n_val]
    return [i for i in indices if not _is_label_token(tokens[i]["text"], label)]


def annotate_bio(
    tokens: list[dict],
    ann_records: list[dict],
    page_w_pts: float,
    page_h_pts: float,
    field_labels: dict[str, str],
) -> list[str]:
    """
    Assigne un tag BIO à chaque token.
    ann_records : liste de {field, bbox:[x,y,w,h], value} en points PDF.
    """
    n = len(tokens)
    labels = ["O"] * n

    for ann in ann_records:
        field = ann.get("field", "")
        label = field_labels.get(field)
        if label is None:
            # Essai avec les sous-champs (containers[].container_number, etc.)
            for fk, lbl in field_labels.items():
                if fk in field:
                    label = lbl
                    break
        if label is None:
            continue

        bbox_raw = ann.get("bbox")
        if not bbox_raw or len(bbox_raw) != 4:
            continue

        value = str(ann.get("value", ""))
        ann_norm = _annot_bbox_to_norm(bbox_raw, page_w_pts, page_h_pts)
        indices  = [i for i in range(n) if _center_in_bbox(tokens[i]["bbox"], ann_norm)]
        indices  = sorted(indices, key=lambda i: (tokens[i]["bbox"][1], tokens[i]["bbox"][0]))
        indices  = _filter_value_tokens(indices, tokens, value, label)

        for rank, idx in enumerate(indices):
            if labels[idx] == "O":   # ne pas écraser un label déjà assigné
                labels[idx] = f"B-{label}" if rank == 0 else f"I-{label}"

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 : Construction JSONL
# ─────────────────────────────────────────────────────────────────────────────

def build_jsonl(
    records: list[dict],
    doc_type: str,
    out_dataset_dir: Path,
    existing_label2id: dict[str, int],
) -> tuple[int, int]:
    """
    Pour chaque record : OCR → BIO → écrit dans test/data.jsonl.
    Retourne (n_ok, n_skip).
    """
    test_dir = out_dataset_dir / "test"
    img_dir  = out_dataset_dir / "images"
    test_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    field_labels = DOC_FIELD_LABELS[doc_type]
    label2id     = existing_label2id

    jsonl_path = test_dir / "data.jsonl"
    n_ok = n_skip = 0

    with open(jsonl_path, "w", encoding="utf-8") as out:
        for rec in tqdm(records, desc=f"  JSONL [{doc_type}]", unit="doc"):
            # OCR
            ocr = extract_ocr(rec["pdf_path"], img_dir)
            if ocr is None or len(ocr["tokens"]) < 10:
                n_skip += 1
                continue

            # Annotations
            ann_data = json.loads(Path(rec["ann_path"]).read_text(encoding="utf-8"))
            ann_records = ann_data.get("bbox_annotations", [])

            # Containers : aplatir les annotations multi-container
            extra_anns = []
            for container in ann_data.get("fields", {}).get("containers", []):
                for fk in ["container_number", "container_type", "seal_number",
                           "commodity", "packages", "gross_weight_kg", "measurement"]:
                    v = container.get(fk)
                    if v is None:
                        continue
                    if isinstance(v, dict):
                        v = v.get("description") or v.get("count", "")
                    matched = next(
                        (a for a in ann_records
                         if a.get("field") == fk and str(a.get("value", "")) == str(v)),
                        None,
                    )
                    if matched:
                        extra_anns.append(matched)
            all_anns = ann_records + extra_anns

            # BIO
            bio_labels = annotate_bio(
                ocr["tokens"], all_anns,
                ocr["page_w_pts"], ocr["page_h_pts"],
                field_labels,
            )

            # Couverture minimale
            n_annotated = sum(1 for l in bio_labels if l != "O")
            coverage    = n_annotated / max(len(bio_labels), 1)
            if coverage < MIN_DOC_COVERAGE:
                n_skip += 1
                log.debug("Skip %s : couverture trop faible (%.2f%%)",
                          rec["doc_id"], coverage * 100)
                continue

            # ner_tags : conversion label → ID (utilise le label2id existant du dataset)
            ner_tags = [label2id.get(lbl, 0) for lbl in bio_labels]

            n_entity_types = len(set(l.split("-", 1)[-1] for l in bio_labels if l != "O"))

            out.write(json.dumps({
                "doc_id":         rec["doc_id"],
                "doc_type":       doc_type,
                "page":           1,
                "image_path":     ocr["image_path"],
                "words":          [t["text"]  for t in ocr["tokens"]],
                "bboxes":         [t["bbox"]  for t in ocr["tokens"]],
                "ner_tags":       ner_tags,
                "n_annotated":    n_annotated,
                "n_entity_types": n_entity_types,
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    log.info("  [%s] Écrits : %d | Ignorés : %d → %s", doc_type, n_ok, n_skip, jsonl_path)
    return n_ok, n_skip


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_label2id(doc_type: str) -> dict[str, int]:
    """Charge le label2id du dataset d'entraînement courant."""
    ds_name = "dataset_packing_list" if doc_type == "packing_list" else f"dataset_{doc_type}"
    p = DATA_COMPLEX / ds_name / "label2id.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # Fallback : BIO dynamique (O=0, B-X=1, I-X=2, ...)
    log.warning("label2id.json introuvable pour %s — label IDs dynamiques", doc_type)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Génère un dataset d'évaluation indépendant")
    p.add_argument("--doc-types", nargs="+",
                   default=["bl", "invoice", "packing_list"],
                   choices=["bl", "invoice", "packing_list"],
                   help="Types de documents à générer")
    p.add_argument("--layouts", nargs="+", default=None,
                   help="Layouts à utiliser (ex: bl-11 bl-15 bl-20). Défaut: auto-détection depuis test set")
    p.add_argument("--all-layouts", action="store_true",
                   help="Utiliser TOUS les layouts (pas seulement ceux du test set)")
    p.add_argument("--count-per-layout", type=int, default=40,
                   help="Nombre de documents par layout (défaut: 40)")
    p.add_argument("--seed", type=int, default=9999,
                   help="Seed aléatoire (défaut: 9999 ≠ entraînement)")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR),
                   help=f"Répertoire de sortie (défaut: {DEFAULT_OUT_DIR})")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)

    log.info("=" * 65)
    log.info("  Génération dataset d'évaluation indépendant")
    log.info("=" * 65)
    log.info("Seed     : %d (entraînement utilisait 0 ou 42)", args.seed)
    log.info("Sortie   : %s", out_root)

    for doc_type in args.doc_types:
        log.info("\n── Type : %s ──────────────────────────────────────────", doc_type)

        # Déterminer les layouts
        if args.layouts:
            # Filtrer par préfixe du type
            prefix = "pl" if doc_type == "packing_list" else doc_type
            layout_ids = [l for l in args.layouts if l.startswith(prefix)]
            if not layout_ids:
                layout_ids = args.layouts   # si pas de préfixe, utiliser tous
        elif args.all_layouts:
            layout_ids = None   # signal pour "tous les layouts"
        else:
            layout_ids = detect_test_layouts(doc_type)

        if not layout_ids:
            # Utiliser tous les layouts disponibles
            try:
                if doc_type == "bl":
                    from layout_engine import BL_LAYOUTS
                    layout_ids = [cfg.layout_id.lower() for cfg in BL_LAYOUTS]
                elif doc_type == "invoice":
                    from layout_engine import INV_LAYOUTS
                    layout_ids = [cfg.layout_id.lower() for cfg in INV_LAYOUTS]
                else:
                    from layout_engine import PL_LAYOUTS
                    layout_ids = [cfg.layout_id.lower() for cfg in PL_LAYOUTS]
            except ImportError:
                log.error("Impossible d'importer les layouts — exécuter depuis swift-validator-ia/")
                continue

        log.info("Layouts  : %s", layout_ids)
        log.info("Docs/layout: %d | Total: ~%d", args.count_per_layout,
                 len(layout_ids) * args.count_per_layout)

        # Chemins de sortie
        ds_name      = "dataset_packing_list" if doc_type == "packing_list" else f"dataset_{doc_type}"
        raw_dir      = out_root / "raw" / ("packing_list" if doc_type == "packing_list" else doc_type)
        out_ds_dir   = out_root / ds_name

        label2id = load_existing_label2id(doc_type)

        # Étape 1 : générer
        log.info("Étape 1/3 : Génération PDFs...")
        records = generate_docs(doc_type, layout_ids, args.count_per_layout,
                                raw_dir, seed=args.seed)
        log.info("  → %d PDFs générés", len(records))

        if not records:
            log.warning("Aucun document généré pour %s — vérifier les layouts.", doc_type)
            continue

        # Étapes 2+3 : OCR + BIO + JSONL
        log.info("Étape 2/3 : OCR extraction...")
        log.info("Étape 3/3 : Annotation BIO + construction JSONL...")
        n_ok, n_skip = build_jsonl(records, doc_type, out_ds_dir, label2id)

        # Copier le label2id/id2label dans le nouveau dataset dir
        src_ds = DATA_COMPLEX / ds_name
        for fname in ("label2id.json", "id2label.json"):
            src = src_ds / fname
            dst = out_ds_dir / fname
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())

        log.info("  Résultat [%s] : %d documents écrits / %d ignorés", doc_type, n_ok, n_skip)

    log.info("\n" + "=" * 65)
    log.info("Dataset d'évaluation disponible dans : %s", out_root)
    log.info("Lancer l'évaluation :")
    log.info("  python evaluation/release2/evaluate_release2_models.py \\")
    log.info("    --data-root %s", out_root)
    log.info("=" * 65)


if __name__ == "__main__":
    main()
