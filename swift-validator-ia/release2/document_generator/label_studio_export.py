"""
label_studio_export.py
======================
Convertit le dataset.json généré en format d'import Label Studio.
Supporte deux modes :
  - OCR / Document Understanding (zones + valeurs)
  - Classification simple (type de B/L)

Usage:
    python label_studio_export.py --dataset ./bl_dataset/dataset.json \
                                   --output ./label_studio_tasks.json \
                                   --mode ocr
"""

import json
import argparse
import os
from pathlib import Path


# Mapping champs → labels Label Studio
FIELD_LABELS = {
    "bl_number":        {"label": "BL_NUMBER",        "color": "#FF0000"},
    "shipper_name":     {"label": "SHIPPER",           "color": "#0000FF"},
    "consignee_name":   {"label": "CONSIGNEE",         "color": "#00AA00"},
    "notify_party":     {"label": "NOTIFY_PARTY",      "color": "#FF8800"},
    "vessel":           {"label": "VESSEL",            "color": "#AA00FF"},
    "voyage":           {"label": "VOYAGE",            "color": "#FF00AA"},
    "port_of_loading":  {"label": "PORT_OF_LOADING",   "color": "#00FFAA"},
    "port_of_discharge":{"label": "PORT_OF_DISCHARGE", "color": "#00AAFF"},
    "container_number": {"label": "CONTAINER_NO",      "color": "#FF5500"},
    "container_type":   {"label": "CONTAINER_TYPE",    "color": "#55FF00"},
    "seal_number":      {"label": "SEAL_NO",           "color": "#0055FF"},
    "gross_weight_kg":  {"label": "GROSS_WEIGHT",      "color": "#FF0055"},
    "measurement":      {"label": "MEASUREMENT",       "color": "#55FFAA"},
    "etd":              {"label": "ETD",               "color": "#AAFF55"},
    "eta":              {"label": "ETA",               "color": "#FF55AA"},
    "issue_date":       {"label": "ISSUE_DATE",        "color": "#AAFFFF"},
    "freight_terms":    {"label": "FREIGHT_TERMS",     "color": "#FFAAFF"},
    "commodity":        {"label": "COMMODITY",         "color": "#AAAAFF"},
}


def generate_label_studio_config() -> str:
    """Génère le XML de configuration Label Studio pour l'annotation OCR."""
    labels_xml = "\n".join([
        f'      <Label value="{v["label"]}" background="{v["color"]}"/>'
        for v in FIELD_LABELS.values()
    ])
    return f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image" showInline="true">
{labels_xml}
  </RectangleLabels>
  <TextArea name="transcription" toName="image"
            editable="true" perRegion="true" required="false"
            maxSubmissions="1" rows="2" placeholder="Transcription du champ"/>
</View>"""


def convert_to_label_studio_ocr(dataset: dict, pdf_base_url: str = "http://localhost:8000/pdfs") -> list:
    """
    Convertit le dataset en tâches Label Studio avec pré-annotations.
    Les B/L sont des PDF → on fournit les bboxes pour pré-annotation.
    """
    tasks = []
    page_w, page_h = 595.28, 841.89  # A4 en points

    for doc in dataset["documents"]:
        annotations = []
        results = []

        for ann in doc.get("bbox_annotations", []):
            field = ann["field"]
            bbox = ann["bbox"]  # [x, y, w, h] en points PDF
            value = ann["value"]

            if field not in FIELD_LABELS:
                continue

            # Convertit en pourcentages (Label Studio attend 0-100%)
            x_pct = (bbox[0] / page_w) * 100
            y_pct = (bbox[1] / page_h) * 100
            w_pct = (bbox[2] / page_w) * 100
            h_pct = (bbox[3] / page_h) * 100

            region_id = f"{doc['id']}_{field}"
            label = FIELD_LABELS[field]["label"]

            results.append({
                "id": region_id,
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": int(page_w),
                "original_height": int(page_h),
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": round(x_pct, 2),
                    "y": round(y_pct, 2),
                    "width": round(w_pct, 2),
                    "height": round(h_pct, 2),
                    "rectanglelabels": [label],
                }
            })
            results.append({
                "id": f"{region_id}_text",
                "type": "textarea",
                "from_name": "transcription",
                "to_name": "image",
                "value": {
                    "text": [value]
                },
                "parent_id": region_id,
            })

        task = {
            "id": doc["id"],
            "data": {
                "image": f"{pdf_base_url}/{doc['filename']}",
                "bl_number": doc["bl_number"],
                "carrier": doc["carrier"],
                "meta": doc["fields"],
            },
            "annotations": [{
                "result": results,
                "ground_truth": True,
            }] if results else [],
            "predictions": [],
        }
        tasks.append(task)

    return tasks


def convert_to_label_studio_classification(dataset: dict, pdf_base_url: str) -> list:
    """
    Tâches de classification simple : quel type de B/L ?
    """
    choices_map = {
        "msc": "MSC",
        "maersk": "Maersk",
        "cmacgm": "CMA CGM",
        "hapaglloyd": "Hapag-Lloyd",
    }

    tasks = []
    for doc in dataset["documents"]:
        task = {
            "id": doc["id"],
            "data": {
                "image": f"{pdf_base_url}/{doc['filename']}",
                "bl_number": doc["bl_number"],
            },
            "annotations": [{
                "result": [{
                    "type": "choices",
                    "from_name": "carrier_type",
                    "to_name": "image",
                    "value": {"choices": [choices_map.get(doc["carrier"], doc["carrier"])]},
                }],
                "ground_truth": True,
            }],
        }
        tasks.append(task)

    return tasks


def generate_classification_config() -> str:
    return """<View>
  <Image name="image" value="$image" zoom="true"/>
  <Choices name="carrier_type" toName="image" choice="single">
    <Choice value="MSC"/>
    <Choice value="Maersk"/>
    <Choice value="CMA CGM"/>
    <Choice value="Hapag-Lloyd"/>
  </Choices>
</View>"""


def main():
    parser = argparse.ArgumentParser(description="Export dataset to Label Studio format")
    parser.add_argument("--dataset", default="./bl_dataset/dataset.json")
    parser.add_argument("--output", default="./label_studio_tasks.json")
    parser.add_argument("--config", default="./label_studio_config.xml")
    parser.add_argument("--mode", choices=["ocr", "classification"], default="ocr")
    parser.add_argument("--pdf-url", default="http://localhost:8000/pdfs",
                        help="Base URL where PDFs are served")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Converting {len(dataset['documents'])} documents to Label Studio format ({args.mode})...")

    if args.mode == "ocr":
        tasks = convert_to_label_studio_ocr(dataset, args.pdf_url)
        config_xml = generate_label_studio_config()
    else:
        tasks = convert_to_label_studio_classification(dataset, args.pdf_url)
        config_xml = generate_classification_config()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    with open(args.config, "w", encoding="utf-8") as f:
        f.write(config_xml)

    print(f"✓ {len(tasks)} tasks → {args.output}")
    print(f"✓ Label Studio config → {args.config}")
    print()
    print("Next steps:")
    print("  1. Install Label Studio:  pip install label-studio")
    print("  2. Start server:          label-studio start")
    print("  3. Create project → Import tasks from:", args.output)
    print("  4. Import labeling config from:", args.config)
    print(f"  5. Serve PDFs:            cd bl_dataset && python -m http.server 8000")


if __name__ == "__main__":
    main()
