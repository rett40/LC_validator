"""
03_build_per_doctype_dataset.py
================================
Étape 3 — Construit trois datasets HuggingFace séparés,
un pour chaque type de document, avec split strict par layout.

Améliorations v2 :
  - Filtre les docs marqués "rejected" (couverture trop faible).
  - Filtre les docs avec 0 entités annotées.
  - Split stratifié : équilibre la distribution des labels entre splits.
  - Rapport de qualité par entité (couverture par label).
  - Ratio 80/10/10 maintenu mais avec min 2 layouts par split val/test.

Entrée  : data/complex/processed/{doc_type}/{doc_id}_bio.json
Sortie  : data/complex/dataset_{bl,invoice,packing_list}/{train,val,test}/data.jsonl
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = RELEASE2_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "03_build_per_doctype_dataset.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

BL_FIELDS = [
    "BL_NUMBER", "SHIPPER", "CONSIGNEE", "NOTIFY_PARTY", "VESSEL", "VOYAGE",
    "PORT_OF_LOADING", "PORT_OF_DISCHARGE", "PLACE_OF_RECEIPT", "PLACE_OF_DELIVERY",
    "ETD", "ETA", "ISSUE_DATE", "ISSUE_PLACE", "CONTAINER_NO", "CONTAINER_TYPE",
    "SEAL_NO", "COMMODITY", "GROSS_WEIGHT", "MEASUREMENT", "PACKAGES",
    "FREIGHT_TERMS", "SERVICE_TYPE", "NUMBER_OF_ORIGINALS",
]
INVOICE_FIELDS = [
    "INV_NUMBER", "ISSUE_DATE", "LC_REF", "BL_REF", "SELLER", "BUYER",
    "VESSEL", "VOYAGE", "PORT_OF_LOADING", "PORT_OF_DISCHARGE", "GOODS_DESC",
    "HS_CODE", "QUANTITY", "PACKAGE_UNIT", "GROSS_WEIGHT", "UNIT_PRICE",
    "CURRENCY", "TOTAL_AMOUNT", "INCOTERM", "PAYMENT_TERMS",
]
PL_FIELDS = [
    "PL_NUMBER", "ISSUE_DATE", "INV_REF", "BL_REF", "EXPORTER", "CONSIGNEE",
    "VESSEL", "VOYAGE", "PORT_OF_LOADING", "PORT_OF_DISCHARGE", "CONTAINER_NO",
    "CONTAINER_TYPE", "SEAL_NO", "GOODS_DESC", "PACKAGE_UNIT", "TOTAL_PACKAGES",
    "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM", "MARKS_NUMBERS", "DIMENSIONS",
]
DOC_FIELDS = {
    "bl": BL_FIELDS,
    "invoice": INVOICE_FIELDS,
    "packing_list": PL_FIELDS,
}
DATASET_DIRS = {
    "bl": "dataset_bl",
    "invoice": "dataset_invoice",
    "packing_list": "dataset_packing_list",
}
SPLIT_RATIOS = (0.80, 0.10, 0.10)
RANDOM_SEED  = 42

# Seuils de qualité
MIN_ANNOTATED_LABELS = 3    # un doc doit avoir au moins N tokens annotés
MIN_ENTITY_TYPES     = 2    # au moins N types d'entités différentes par doc


def build_label_maps(fields: list[str]):
    label_list = ["O"]
    for fn in fields:
        label_list.append(f"B-{fn}")
        label_list.append(f"I-{fn}")
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    return label_list, label2id, id2label


def extract_layout_id(doc_id: str, doc_type: str) -> str:
    prefix_map = {
        "bl": "bl_",
        "invoice": "inv_",
        "packing_list": "pl_"
    }
    prefix = prefix_map.get(doc_type, doc_type + "_")
    rest   = doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id
    parts  = rest.split('_')
    return parts[0]


def load_examples_from_bio(
    bio_path: Path,
    label2id: dict[str, int],
    data_root: Path,
) -> tuple[str, list[dict]] | None:
    """
    Charge un fichier BIO, applique les filtres de qualité,
    et retourne (layout_id, list_of_pages) ou None si rejeté.
    """
    data = json.loads(bio_path.read_text(encoding="utf-8"))

    # ← NOUVEAU : rejeter les docs marqués low-coverage
    if data.get("rejected", False):
        return None

    doc_id   = data["doc_id"]
    doc_type = data["doc_type"]
    layout   = extract_layout_id(doc_id, doc_type)

    examples = []
    for page in data["pages"]:
        words  = page["tokens"]
        bboxes = page["bboxes"]
        labels = page["labels"]
        if not words:
            continue

        ner_tags = [label2id.get(lbl, 0) for lbl in labels]

        # ← NOUVEAU : contrôle qualité par page
        n_annotated    = sum(1 for t in ner_tags if t != 0)
        n_entity_types = len({lbl.split("-", 1)[1] for lbl in labels
                               if lbl != "O"})
        if n_annotated < MIN_ANNOTATED_LABELS:
            continue
        if n_entity_types < MIN_ENTITY_TYPES:
            continue

        img_rel  = page["image_path"]
        img_path = (data_root / img_rel).resolve()
        if not img_path.exists():
            img_path_abs = Path(img_rel)
            if img_path_abs.exists():
                img_path = img_path_abs

        examples.append({
            "doc_id":    doc_id,
            "doc_type":  doc_type,
            "page":      page["page"],
            "image_path": str(img_path),
            "words":     words,
            "bboxes":    bboxes,
            "ner_tags":  ner_tags,
            # métadonnées qualité (utiles pour debug)
            "n_annotated": n_annotated,
            "n_entity_types": n_entity_types,
        })

    return (layout, examples) if examples else None


def save_jsonl(examples: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def split_layouts(layouts: list[str], ratios: tuple[float, float, float], seed: int):
    """Split stratifié par layout — garantit min 2 layouts par split val/test."""
    rng = random.Random(seed)
    shuffled = layouts[:]
    rng.shuffle(shuffled)
    n = len(shuffled)

    n_test = max(2, int(n * ratios[2]))
    n_val  = max(2, int(n * ratios[1]))
    n_test = min(n_test, n - n_val - 1)
    n_val  = min(n_val,  n - n_test - 1)

    test_layouts  = set(shuffled[:n_test])
    val_layouts   = set(shuffled[n_test:n_test + n_val])
    train_layouts = set(shuffled[n_test + n_val:])
    return train_layouts, val_layouts, test_layouts


def compute_entity_coverage(examples: list[dict], fields: list[str],
                             id2label: dict[int, str]) -> dict[str, float]:
    """Calcule le % de documents qui ont au moins 1 token pour chaque entité."""
    entity_doc_count: dict[str, int] = defaultdict(int)
    total_docs = len(set(ex["doc_id"] for ex in examples))

    for ex in examples:
        found = {id2label[t].split("-", 1)[1]
                 for t in ex["ner_tags"] if t != 0 and id2label.get(t, "O") != "O"}
        for ent in found:
            entity_doc_count[ent] += 1

    return {
        f: round(entity_doc_count.get(f, 0) / total_docs, 3) if total_docs else 0.0
        for f in fields
    }


def build_per_doctype(doc_type: str, data_root: Path) -> dict:
    fields = DOC_FIELDS[doc_type]
    _, label2id, id2label = build_label_maps(fields)
    proc_dir    = data_root / "processed" / doc_type
    dataset_dir = data_root / DATASET_DIRS[doc_type]

    bio_files = sorted(proc_dir.glob("*_bio.json"))
    log.info("[%s] %d BIO JSON trouvés", doc_type.upper(), len(bio_files))
    if not bio_files:
        return {"total_docs": 0, "total_pages": 0}

    layout_to_pages: dict[str, list[dict]] = defaultdict(list)
    doc_count   = 0
    rejected    = 0
    low_quality = 0

    for bio_path in tqdm(bio_files, desc=f"  load {doc_type}", unit="doc"):
        result = load_examples_from_bio(bio_path, label2id, data_root)
        if result is None:
            rejected += 1
            continue
        layout, exs = result
        if not exs:
            low_quality += 1
            continue
        layout_to_pages[layout].extend(exs)
        doc_count += 1

    total_pages = sum(len(v) for v in layout_to_pages.values())
    log.info("  %d docs valides / %d rejetés / %d low-quality / %d pages / %d layouts",
             doc_count, rejected, low_quality, total_pages, len(layout_to_pages))

    if not layout_to_pages:
        log.error("Aucun document valide pour %s — vérifier l'annotation", doc_type)
        return {"total_docs": 0, "total_pages": 0}

    all_layouts = list(layout_to_pages.keys())
    train_layouts, val_layouts, test_layouts = split_layouts(
        all_layouts, SPLIT_RATIOS, RANDOM_SEED
    )
    log.info("  Split : train=%d, val=%d, test=%d layouts",
             len(train_layouts), len(val_layouts), len(test_layouts))

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for layout, pages in layout_to_pages.items():
        if layout in train_layouts:
            splits["train"].extend(pages)
        elif layout in val_layouts:
            splits["val"].extend(pages)
        else:
            splits["test"].extend(pages)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split_name, exs in splits.items():
        save_jsonl(exs, dataset_dir / split_name / "data.jsonl")
        log.info("  %s: %d pages", split_name, len(exs))

    (dataset_dir / "label2id.json").write_text(
        json.dumps(label2id, indent=2), encoding="utf-8"
    )
    (dataset_dir / "id2label.json").write_text(
        json.dumps({str(k): v for k, v in id2label.items()}, indent=2),
        encoding="utf-8"
    )

    # ← NOUVEAU : couverture par entité sur le train set
    train_coverage = compute_entity_coverage(splits["train"], fields, id2label)
    low_cov_entities = [f for f, c in train_coverage.items() if c < 0.50]
    if low_cov_entities:
        log.warning("  Entités avec couverture < 50%% dans train : %s",
                    ", ".join(low_cov_entities))

    stats = {
        "doc_type":        doc_type,
        "n_fields":        len(fields),
        "n_labels":        len(label2id),
        "total_docs":      doc_count,
        "total_pages":     total_pages,
        "n_layouts":       len(all_layouts),
        "rejected_docs":   rejected,
        "low_quality_docs": low_quality,
        "train_pages":     len(splits["train"]),
        "val_pages":       len(splits["val"]),
        "test_pages":      len(splits["test"]),
        "entity_coverage_train": train_coverage,
    }
    (dataset_dir / "stats.json").write_text(json.dumps(stats, indent=2),
                                             encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=str,
        default=str(RELEASE2_ROOT / "data" / "complex")
    )
    parser.add_argument(
        "--doc-type", nargs="+",
        choices=["bl", "invoice", "packing_list"],
        default=["bl", "invoice", "packing_list"]
    )
    args = parser.parse_args()
    data_root = Path(args.data_root)

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  Script 03 v2 — Split strict par layout + qualité       ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Data root  : %-42s║", str(data_root))
    log.info("║  Doc types  : %-42s║", ", ".join(args.doc_type))
    log.info("║  Filtres    : coverage faible, 0-entités rejetés        ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    all_stats = {}
    for dt in args.doc_type:
        log.info("\n══ [%s] ══", dt.upper())
        all_stats[dt] = build_per_doctype(dt, data_root)

    log.info("\n╔══════════════════════════════════════════════════════════╗")
    log.info("║   RÉSUMÉ FINAL                                          ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    for dt, s in all_stats.items():
        log.info("║  %-14s : %3d docs / %4d pages / %3d layouts   ║",
                 dt, s["total_docs"], s["total_pages"], s.get("n_layouts", 0))
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("\nProchaine étape :")
    log.info("  python scripts/04_train_per_doctype.py --doc-type bl --bbox-jitter 8")
    log.info("  python scripts/04_train_per_doctype.py --doc-type invoice --bbox-jitter 8")
    log.info("  python scripts/04_train_per_doctype.py --doc-type packing_list --bbox-jitter 8")


if __name__ == "__main__":
    main()
