"""
dataset_stats.py
================
Analyse statistique du dataset généré.
Affiche la distribution, vérifie la cohérence des annotations,
et génère un rapport de validation.

Usage:
    python dataset_stats.py --dataset ./bl_dataset/dataset.json
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def load_dataset(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze(dataset: dict):
    docs = dataset["documents"]
    meta = dataset["metadata"]

    print("\n" + "="*60)
    print("  DATASET STATISTICS REPORT")
    print("="*60)

    # Distribution par carrier
    carriers = Counter(d["carrier"] for d in docs)
    print(f"\n📦 Total documents: {len(docs)}")
    print("\n  Distribution by carrier:")
    for carrier, count in carriers.most_common():
        bar = "█" * (count // 5)
        print(f"    {carrier:15s}  {count:4d}  {bar}")

    # Champs annotés
    all_fields = set()
    field_counts = Counter()
    for doc in docs:
        for ann in doc.get("bbox_annotations", []):
            field_counts[ann["field"]] += 1
            all_fields.add(ann["field"])

    print(f"\n📝 Annotated fields ({len(all_fields)} unique):")
    for field, count in field_counts.most_common():
        coverage = count / len(docs) * 100
        bar = "█" * int(coverage / 5)
        print(f"    {field:25s}  {count:4d} ({coverage:5.1f}%)  {bar}")

    # Validation intégrité
    print("\n✅ Integrity checks:")
    missing_bl = [d for d in docs if not d.get("bl_number")]
    print(f"    Documents with BL number : {len(docs) - len(missing_bl)}/{len(docs)}")

    no_annotations = [d for d in docs if not d.get("bbox_annotations")]
    print(f"    Documents with bboxes    : {len(docs) - len(no_annotations)}/{len(docs)}")

    # Splits
    print("\n📊 Dataset splits:")
    dataset_dir = Path(args.dataset).parent
    for split in ["train", "val", "test"]:
        split_file = dataset_dir / f"{split}.json"
        if split_file.exists():
            with open(split_file) as f:
                split_data = json.load(f)
            print(f"    {split:8s}: {split_data['count']:4d} documents")

    # Exemples de valeurs
    print("\n🔍 Sample field values (first 5 docs):")
    for doc in docs[:5]:
        print(f"    [{doc['carrier']:12s}] BL: {doc['bl_number']}")
        fields = doc.get("fields", {})
        print(f"               → POL: {fields.get('port_of_loading', 'N/A')}")
        print(f"               → POD: {fields.get('port_of_discharge', 'N/A')}")
        print(f"               → CTR: {fields.get('container_number', 'N/A')}")
        print()

    print("="*60)
    print("  Report complete ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./bl_dataset/dataset.json")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset)
    analyze(dataset)
