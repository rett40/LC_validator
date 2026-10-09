"""
pl_extraction_service.py
========================
Extracteur spécialisé Packing List.

Modèle      : models/models_layout/best_pl/best_pl/  (LayoutLMv3, 42 labels)
Segmentation: grid  (header_left / header_right / body_left / body_right / footer)

Champs extraits (21 types) :
  PL_NUMBER, ISSUE_DATE, INV_REF, BL_REF,
  EXPORTER, CONSIGNEE,
  VESSEL, VOYAGE, PORT_OF_LOADING, PORT_OF_DISCHARGE,
  CONTAINER_NO, CONTAINER_TYPE, SEAL_NO,
  GOODS_DESC, PACKAGE_UNIT, TOTAL_PACKAGES,
  NET_WEIGHT, GROSS_WEIGHT, TOTAL_CBM,
  MARKS_NUMBERS, DIMENSIONS

Usage standalone :
  python -m release2.services.pl_extractor.pl_extraction_service --pdf packing_list.pdf
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

try:
    from .._base_layout_extractor import BaseLayoutExtractor, ComputedFieldSpec
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from services._base_layout_extractor import BaseLayoutExtractor, ComputedFieldSpec

log = logging.getLogger(__name__)

_RELEASE2_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_LAYOUT = _RELEASE2_ROOT / "models" / "models_layout"


class PLExtractor(BaseLayoutExtractor):
    """
    Extracteur LayoutLMv3 spécialisé pour les Packing Lists.

    Segmentation grid :
      header_left / header_right → PL_NUMBER, dates, références, Exporter, Consignee
      body_left / body_right     → détails colis, conteneurs, poids
      footer                     → zones à rejeter pour les champs d'en-tête
    """

    MODEL_DIR      = _MODELS_LAYOUT / "best_pl" / "best_pl"
    TABLE_NAME     = "packages"

    EXPECTED_FIELDS = {
        "PL_NUMBER", "ISSUE_DATE", "INV_REF", "BL_REF",
        "EXPORTER", "CONSIGNEE",
        "VESSEL", "VOYAGE", "PORT_OF_LOADING", "PORT_OF_DISCHARGE",
        "CONTAINER_NO", "CONTAINER_TYPE", "SEAL_NO",
        "GOODS_DESC", "PACKAGE_UNIT", "TOTAL_PACKAGES",
        "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM",
        "MARKS_NUMBERS", "DIMENSIONS",
    }

    REJECT_IN_SEGMENT: dict[str, set[str]] = {
        "footer": {
            "PL_NUMBER", "ISSUE_DATE", "INV_REF", "BL_REF",
            "EXPORTER", "CONSIGNEE",
            "GOODS_DESC",
        },
        "header_left": {
            "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM", "TOTAL_PACKAGES",
        },
        "header_right": {
            "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM", "TOTAL_PACKAGES",
        },
        "body_left": {
            "CONSIGNEE",
        },
        "body_right": {
            "EXPORTER",
        },
    }

    NUMERIC_FIELDS = {
        "TOTAL_PACKAGES", "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM",
    }
    DATE_FIELDS = {"ISSUE_DATE"}

    FIELD_NAME_MAP = {
        "PL_NUMBER":       "Packing List No.",
        "ISSUE_DATE":      "Issue Date",
        "INV_REF":         "Invoice Reference",
        "BL_REF":          "B/L Reference",
        "EXPORTER":        "Exporter",
        "CONSIGNEE":       "Consignee",
        "VESSEL":          "Vessel",
        "VOYAGE":          "Voyage",
        "PORT_OF_LOADING": "Port of Loading",
        "PORT_OF_DISCHARGE": "Port of Discharge",
        "CONTAINER_NO":    "Container No.",
        "CONTAINER_TYPE":  "Container Type",
        "SEAL_NO":         "Seal No.",
        "GOODS_DESC":      "Goods Description",
        "PACKAGE_UNIT":    "Package Unit",
        "TOTAL_PACKAGES":  "Total Packages",
        "NET_WEIGHT":      "Net Weight",
        "GROSS_WEIGHT":    "Gross Weight",
        "TOTAL_CBM":       "Total CBM",
        "MARKS_NUMBERS":   "Marks & Numbers",
        "DIMENSIONS":      "Dimensions",
    }

    # ── Labels connus dans les documents Packing List ─────────────────────────
    FIELD_SYNONYMS: dict[str, list[str]] = {
        "PL_NUMBER":       ["PACKING LIST NO.", "PACKING LIST NO", "PACKING LIST NUMBER",
                            "PL NO.", "PL NO", "PL NUMBER"],
        "ISSUE_DATE":      ["ISSUE DATE", "DATE OF ISSUE", "PL DATE", "DATE ISSUED"],
        "INV_REF":         ["INVOICE REF", "INVOICE REFERENCE", "INV REF", "INV. NO.", "INV NO",
                            "REF INVOICE NO", "REF. INVOICE NO.", "REF INVOICE", "REF. INVOICE"],
        "BL_REF":          ["B/L REFERENCE", "B/L NO.", "B/L NO", "BL REF", "BL NO.",
                            "REF B/L NO", "REF. B/L NO.", "REF B/L", "REF. B/L", "REF BL NO"],
        "EXPORTER":        ["EXPORTER", "SHIPPER", "CONSIGNOR", "SELLER"],
        "CONSIGNEE":       ["CONSIGNEE", "CONSIGNEE NAME", "TO"],
        "VESSEL":          ["VESSEL", "VESSEL NAME", "M/V", "SHIP NAME"],
        "VOYAGE":          ["VOYAGE NO.", "VOYAGE NO", "VOYAGE NUMBER", "VOY"],
        "PORT_OF_LOADING":   ["PORT OF LOADING", "LOADING PORT", "POL"],
        "PORT_OF_DISCHARGE": ["PORT OF DISCHARGE", "DISCHARGE PORT", "POD"],
        "CONTAINER_NO":    ["CONTAINER NO.", "CONTAINER NO", "CONTAINER NUMBER", "CNTR NO."],
        "CONTAINER_TYPE":  ["CONTAINER TYPE", "CNTR TYPE", "SIZE/TYPE"],
        "SEAL_NO":         ["SEAL NO.", "SEAL NO", "SEAL NUMBER", "SEAL"],
        "GOODS_DESC":      ["GOODS DESCRIPTION", "DESCRIPTION OF GOODS", "COMMODITY",
                            "DESCRIPTION", "GOODS", "MARKS AND NUMBERS"],
        "PACKAGE_UNIT":    ["PACKAGE UNIT", "UNIT", "TYPE OF PACKING", "PKG UNIT"],
        "TOTAL_PACKAGES":  ["TOTAL PACKAGES", "NO. OF PACKAGES", "NO. OF PKGS",
                            "TOTAL PIECES", "TOTAL PKGS"],
        "NET_WEIGHT":      ["NET WEIGHT", "NET WT", "N.W.", "NET WEIGHT KG"],
        "GROSS_WEIGHT":    ["GROSS WEIGHT", "GROSS WT", "G.W.", "GROSS WEIGHT KG"],
        "TOTAL_CBM":       ["TOTAL CBM", "CBM", "TOTAL VOLUME", "MEASUREMENT", "M3"],
        "MARKS_NUMBERS":   ["MARKS AND NUMBERS", "MARKS & NOS", "MARKS & NUMBERS",
                            "SHIPPING MARKS"],
        "DIMENSIONS":      ["DIMENSIONS", "SIZE", "L X W X H", "MEASUREMENTS"],
    }

    # ── Champs formant les lignes de la packing list ──────────────────────────
    TABLE_FIELDS     = {"GOODS_DESC", "PACKAGE_UNIT", "TOTAL_PACKAGES",
                        "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM"}
    ROW_ANCHOR_FIELD = "GOODS_DESC"
    ROW_FIELDS       = {"GOODS_DESC", "PACKAGE_UNIT", "TOTAL_PACKAGES",
                        "NET_WEIGHT", "GROSS_WEIGHT", "TOTAL_CBM",
                        "MARKS_NUMBERS", "DIMENSIONS"}

    # ── Champs calculés depuis les lignes de tableau ───────────────────────────
    COMPUTED_SPECS = [
        ComputedFieldSpec(target="TOTAL_PACKAGES", op="sum", fields=["TOTAL_PACKAGES"], tolerance=0.05),
        ComputedFieldSpec(target="NET_WEIGHT",     op="sum", fields=["NET_WEIGHT"],     tolerance=0.05),
        ComputedFieldSpec(target="GROSS_WEIGHT",   op="sum", fields=["GROSS_WEIGHT"],   tolerance=0.05),
        ComputedFieldSpec(target="TOTAL_CBM",      op="sum", fields=["TOTAL_CBM"],      tolerance=0.05),
    ]

    # ── Validation sémantique des lignes ──────────────────────────────────────
    ROW_VALIDATORS: dict = {
        "TOTAL_PACKAGES": lambda v: bool(re.search(r'\d', v)),
        "NET_WEIGHT":     lambda v: bool(re.search(r'\d', v)),
        "GROSS_WEIGHT":   lambda v: bool(re.search(r'\d', v)),
        "TOTAL_CBM":      lambda v: bool(re.search(r'\d', v)),
    }

    FORBIDDEN_KEYWORDS: list[str] = [
        # ── Anglais ───────────────────────────────────────────────────────────
        "packing list", "packing list no", "packing list no.", "packing list number",
        "pl number", "pl no", "pl no.", "pl ref",
        "issue date", "date of issue", "pl date", "date issued",
        "invoice ref", "invoice reference", "inv ref", "inv reference",
        "bl ref", "bl reference", "b/l reference", "b/l no",
        "exporter", "shipper", "consignor",
        "consignee", "consignee name",
        "vessel", "vessel name", "m/v", "ship name",
        "voyage", "voyage no", "voyage no.", "voyage number",
        "port of loading", "loading port", "pol",
        "port of discharge", "discharge port", "pod",
        "container no", "container no.", "container number", "cntr no",
        "container type", "cntr type", "container size",
        "seal no", "seal no.", "seal number", "seal",
        "goods description", "description of goods", "commodity", "description",
        "package unit", "unit", "packing unit",
        "total packages", "no of packages", "number of packages", "total pieces",
        "net weight", "net wt", "n.w.", "net weight kg",
        "gross weight", "gross wt", "g.w.", "gross weight kg",
        "total cbm", "cbm", "m3", "cubic meters", "measurement",
        "marks and numbers", "marks & nos", "marks", "shipping marks",
        "dimensions", "size", "l x w x h",
        # ── Français ──────────────────────────────────────────────────────────
        "liste de colisage", "colisage", "numéro pl", "n° pl",
        "date d'émission",
        "référence facture", "réf. facture",
        "référence bl", "réf. bl",
        "exportateur", "expéditeur",
        "destinataire",
        "navire",
        "port de chargement", "port de déchargement",
        "numéro de conteneur",
        "type de conteneur",
        "numéro de scellé",
        "désignation des marchandises", "description des marchandises",
        "type d'emballage", "unité d'emballage",
        "nombre total de colis", "total colis",
        "poids net",
        "poids brut",
        "volume total", "cubage total",
        "marques et numéros", "marques",
        "dimensions",
        # ── Espagnol ──────────────────────────────────────────────────────────
        "lista de empaque", "lista de embalaje", "número pl",
        "exportador",
        "consignatario",
        "buque",
        "puerto de carga", "puerto de descarga",
        "número de contenedor",
        "tipo de contenedor",
        "número de sello",
        "descripción de mercancías",
        "unidad de embalaje",
        "total de bultos",
        "peso neto", "peso bruto",
        "volumen total",
        "marcas y números",
        # ── Arabe translittéré ────────────────────────────────────────────────
        "قائمة التعبئة", "رقم قائمة التعبئة",
        "المصدر", "المرسل إليه",
        "وصف البضاعة",
        "الوزن الصافي", "الوزن الإجمالي",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="PLExtractor — extraction champs Packing List")
    parser.add_argument("--pdf",        required=True)
    parser.add_argument("--input-type", default="pdf_native",
                        choices=["pdf_native", "pdf_scanned", "image"])
    parser.add_argument("--model-dir",  default=None)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    ext    = PLExtractor(model_dir=args.model_dir)
    result = ext.extract_from_pdf(args.pdf, input_type=args.input_type)
    ext.print_results(result["fields"])

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Résultat sauvegardé → %s", args.output)


if __name__ == "__main__":
    main()
