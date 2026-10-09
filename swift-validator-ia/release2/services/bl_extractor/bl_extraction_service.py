"""
bl_extraction_service.py
========================
Extracteur spécialisé Bill of Lading.

Modèle      : models/models_layout/best_bl/best_bl/  (LayoutLMv3, 48 labels)
Segmentation: grid  (header_left / header_right / body_left / body_right / footer)

Champs extraits (24 types) :
  BL_NUMBER, SHIPPER, CONSIGNEE, NOTIFY_PARTY, VESSEL, VOYAGE,
  PORT_OF_LOADING, PORT_OF_DISCHARGE, PLACE_OF_RECEIPT, PLACE_OF_DELIVERY,
  ETD, ETA, ISSUE_DATE, ISSUE_PLACE,
  CONTAINER_NO, CONTAINER_TYPE, SEAL_NO,
  COMMODITY, GROSS_WEIGHT, MEASUREMENT, PACKAGES,
  FREIGHT_TERMS, SERVICE_TYPE, NUMBER_OF_ORIGINALS

Usage standalone :
  python -m release2.services.bl_extractor.bl_extraction_service --pdf doc.pdf
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


class BLExtractor(BaseLayoutExtractor):
    """
    Extracteur LayoutLMv3 spécialisé pour les Bills of Lading.

    Segmentation grid :
      header_left / header_right → informations navire, ports, dates, BL number
      body_left                  → Shipper, place of receipt
      body_right                 → Consignee, Notify Party, place of delivery
      footer                     → zones à rejeter pour les champs principaux
    """

    MODEL_DIR      = _MODELS_LAYOUT / "best_bl" / "best_bl"
    TABLE_NAME     = "containers"

    EXPECTED_FIELDS = {
        "BL_NUMBER", "SHIPPER", "CONSIGNEE", "NOTIFY_PARTY",
        "VESSEL", "VOYAGE",
        "PORT_OF_LOADING", "PORT_OF_DISCHARGE",
        "PLACE_OF_RECEIPT", "PLACE_OF_DELIVERY",
        "ETD", "ETA", "ISSUE_DATE", "ISSUE_PLACE",
        "CONTAINER_NO", "CONTAINER_TYPE", "SEAL_NO",
        "COMMODITY", "GROSS_WEIGHT", "MEASUREMENT", "PACKAGES",
        "FREIGHT_TERMS", "SERVICE_TYPE", "NUMBER_OF_ORIGINALS",
    }

    # Champs à rejeter selon le segment où ils sont détectés
    REJECT_IN_SEGMENT: dict[str, set[str]] = {
        "footer": {
            "BL_NUMBER", "SHIPPER", "CONSIGNEE", "NOTIFY_PARTY",
            "VESSEL", "VOYAGE",
            "PORT_OF_LOADING", "PORT_OF_DISCHARGE",
            "PLACE_OF_RECEIPT", "PLACE_OF_DELIVERY",
            "ISSUE_DATE", "ISSUE_PLACE", "ETD", "ETA",
        },
        "header_left": {
            "CONSIGNEE", "NOTIFY_PARTY",
        },
        "header_right": {
            "SHIPPER",
        },
        "body_left": {
            "CONSIGNEE", "NOTIFY_PARTY",
        },
        "body_right": {
            "SHIPPER",
        },
    }

    NUMERIC_FIELDS = {
        "GROSS_WEIGHT", "MEASUREMENT", "PACKAGES", "NUMBER_OF_ORIGINALS",
    }
    DATE_FIELDS = {"ETD", "ETA", "ISSUE_DATE"}

    FIELD_NAME_MAP = {
        "BL_NUMBER":           "B/L Number",
        "SHIPPER":             "Shipper",
        "CONSIGNEE":           "Consignee",
        "NOTIFY_PARTY":        "Notify Party",
        "VESSEL":              "Vessel",
        "VOYAGE":              "Voyage",
        "PORT_OF_LOADING":     "Port of Loading",
        "PORT_OF_DISCHARGE":   "Port of Discharge",
        "PLACE_OF_RECEIPT":    "Place of Receipt",
        "PLACE_OF_DELIVERY":   "Place of Delivery",
        "ETD":                 "ETD",
        "ETA":                 "ETA",
        "ISSUE_DATE":          "Issue Date",
        "ISSUE_PLACE":         "Issue Place",
        "CONTAINER_NO":        "Container No.",
        "CONTAINER_TYPE":      "Container Type",
        "SEAL_NO":             "Seal No.",
        "COMMODITY":           "Commodity",
        "GROSS_WEIGHT":        "Gross Weight",
        "MEASUREMENT":         "Measurement (CBM)",
        "PACKAGES":            "Packages",
        "FREIGHT_TERMS":       "Freight Terms",
        "SERVICE_TYPE":        "Service Type",
        "NUMBER_OF_ORIGINALS": "Number of Originals",
    }

    # Mots interdits dans les valeurs extraites (noms de champs + synonymes, multi-langues)
    # ── Labels connus dans les documents BL (pour détection zones label/valeur) ──
    FIELD_SYNONYMS: dict[str, list[str]] = {
        "BL_NUMBER":           ["B/L NUMBER", "BL NO.", "B/L NO.", "B/L NO", "BILL OF LADING NO",
                                 "B/L N°", "BL NUMBER", "BL NO", "WAYBILL NO"],
        "SHIPPER":             ["SHIPPER", "SHIPPER/EXPORTER", "SHIPPER / EXPORTER", "EXPORTER", "SHIPPER NAME", "SHIPPER/EXPORTER NAME", "EXPORTER NAME"],
        "CONSIGNEE":           ["CONSIGNEE", "CONSIGNEE NAME", "CONSIGNEE ADDRESS", "BUYER", "BUYER NAME"],
        "NOTIFY_PARTY":        ["NOTIFY PARTY", "NOTIFY", "ALSO NOTIFY", "NOTIFY ADDRESS", "NOTIFY NAME"],
        "VESSEL":              ["VESSEL", "VESSEL NAME", "OCEAN VESSEL", "M/V", "SHIP NAME", "VESSEL/VOYAGE"],
        "VOYAGE":              ["VOYAGE NO.", "VOYAGE NO", "VOYAGE NUMBER", "VOY NO.", "VOY", "VOYAGE"],
        "PORT_OF_LOADING":     ["PORT OF LOADING", "PORT OF SHIPMENT", "LOADING PORT", "POL", "LOADING"],
        "PORT_OF_DISCHARGE":   ["PORT OF DISCHARGE", "DISCHARGING PORT", "POD", "DISCHARGE", "PORT OF DESTINATION"],
        "PLACE_OF_RECEIPT":    ["PLACE OF RECEIPT", "PRE-CARRIAGE PLACE", "RECEIPT PLACE"],
        "PLACE_OF_DELIVERY":   ["PLACE OF DELIVERY", "FINAL DESTINATION", "DELIVERY PLACE"],
        "ETD":                 ["ETD", "ESTIMATED DEPARTURE", "DATE OF DEPARTURE", "DEPARTURE DATE"],
        "ETA":                 ["ETA", "ESTIMATED ARRIVAL", "DATE OF ARRIVAL", "ARRIVAL DATE"],
        "ISSUE_DATE":          ["ISSUE DATE", "DATE OF ISSUE", "B/L DATE", "DATE ISSUED", "DATE OF BL"],
        "ISSUE_PLACE":         ["ISSUE PLACE", "PLACE OF ISSUE", "ISSUED AT", "PLACE OF BL"],
        "CONTAINER_NO":        ["CONTAINER NO.", "CONTAINER NO", "CONTAINER NUMBER", "CNTR NO.", "CNTR NO"],
        "CONTAINER_TYPE":      ["CONTAINER TYPE", "CNTR TYPE", "SIZE/TYPE", "CONTAINER SIZE"],
        "SEAL_NO":             ["SEAL NO.", "SEAL NO", "SEAL NUMBER", "SEAL"],
        "COMMODITY":           ["COMMODITY", "DESCRIPTION OF GOODS", "NATURE OF GOODS", "GOODS DESCRIPTION"],
        "GROSS_WEIGHT":        ["GROSS WEIGHT", "GROSS WT", "GROSS WEIGHT (KG)", "G.W.", "WEIGHT"],
        "MEASUREMENT":         ["MEASUREMENT", "CBM", "VOLUME", "M3", "MEAS"],
        "PACKAGES":            ["NO. OF PACKAGES", "NUMBER OF PACKAGES", "PACKAGES", "PKGS", "QUANTITY"],
        "FREIGHT_TERMS":       ["FREIGHT TERMS", "FREIGHT", "PAYMENT", "FREIGHT PAYABLE"],
        "SERVICE_TYPE":        ["SERVICE TYPE", "SERVICE", "CY/CY", "CFS/CFS", "DOOR/DOOR"],
        "NUMBER_OF_ORIGINALS": ["NUMBER OF ORIGINALS", "NO. OF ORIGINALS", "ORIGINALS", "ORIGINAL B/L"],
    }

    # ── Champs formant les lignes du tableau des conteneurs ───────────────────
    TABLE_FIELDS     = {"CONTAINER_NO", "CONTAINER_TYPE", "SEAL_NO",
                        "PACKAGES", "GROSS_WEIGHT", "MEASUREMENT"}
    ROW_ANCHOR_FIELD = "CONTAINER_NO"
    ROW_FIELDS       = {"CONTAINER_NO", "CONTAINER_TYPE", "SEAL_NO",
                        "PACKAGES", "GROSS_WEIGHT", "MEASUREMENT"}

    # ── Champs calculés depuis les lignes de tableau ───────────────────────────
    COMPUTED_SPECS = [
        # Total poids brut = somme des poids par conteneur
        ComputedFieldSpec(target="GROSS_WEIGHT",  op="sum", fields=["GROSS_WEIGHT"],  tolerance=0.05),
        # Total colis = somme des colis par conteneur
        ComputedFieldSpec(target="PACKAGES",      op="sum", fields=["PACKAGES"],      tolerance=0.05),
        # Total CBM = somme des mesures par conteneur
        ComputedFieldSpec(target="MEASUREMENT",   op="sum", fields=["MEASUREMENT"],   tolerance=0.05),
    ]

    # ── Validation sémantique des lignes de tableau ───────────────────────────
    # CONTAINER_NO doit contenir un format ISO 6346 approximatif : ABCU1234567
    ROW_VALIDATORS: dict = {
        "CONTAINER_NO": lambda v: bool(
            re.search(r'[A-Z]{3,4}[UJZujz]?\d{6,7}', re.sub(r'\s+', '', v.upper()))
        ),
        "GROSS_WEIGHT": lambda v: bool(re.search(r'\d', v)),
        "MEASUREMENT":  lambda v: bool(re.search(r'\d', v)),
    }

    FORBIDDEN_KEYWORDS: list[str] = [
        # ── Anglais ───────────────────────────────────────────────────────────
        "bill of lading", "bill of loading", "b/l number", "b/l no", "b/l no.",
        "b.l.", "bl number", "bl no", "bl no.", "bl ref", "bl reference",
        "waybill number", "seaway bill",
        "shipper", "exporter", "shipper/exporter", "shipper / exporter",
        "consignee", "consignee name", "consignee address",
        "notify party", "notify", "also notify", "to order",
        "vessel", "vessel name", "ship name", "m/v", "m.v.", "ocean vessel",
        "voyage", "voyage no", "voyage no.", "voyage number", "voyage num", "voy",
        "port of loading", "loading port", "pol", "port of shipment",
        "port of discharge", "discharge port", "pod", "port of destination",
        "place of receipt", "receipt", "place of pre-receipt",
        "place of delivery", "final destination",
        "etd", "estimated departure", "departure date", "date of departure",
        "eta", "estimated arrival", "arrival date", "date of arrival",
        "issue date", "date of issue", "b/l date", "date issued", "date of bl",
        "issue place", "place of issue", "issued at",
        "container no", "container no.", "container number", "cntr no", "cntr no.",
        "container type", "cntr type", "container size", "size/type",
        "seal no", "seal no.", "seal number", "seal",
        "commodity", "description of goods", "nature of goods", "goods description",
        "gross weight", "gross wt", "gross weight kg", "gross wt (kg)", "g.w.",
        "measurement", "cbm", "m3", "cubic meters", "cubic metre",
        "packages", "no of packages", "number of packages", "no. of pkgs",
        "no. of packages", "total packages",
        "freight terms", "freight", "freight prepaid", "freight collect",
        "service type", "service", "cargo type",
        "number of originals", "no of originals", "no. of originals", "originals",
        # ── Français ──────────────────────────────────────────────────────────
        "connaissement", "numéro bl", "n° bl", "lettre de transport maritime",
        "expéditeur", "chargeur",
        "destinataire", "consignataire",
        "partie à notifier", "à notifier",
        "navire", "nom du navire", "bateau",
        "port de chargement", "port d'embarquement", "port d'expédition",
        "port de déchargement", "port de destination", "port de livraison",
        "lieu de prise en charge", "lieu de réception",
        "lieu de livraison",
        "date de départ estimée", "date d'émission", "date b/l",
        "lieu d'émission", "lieu de délivrance",
        "numéro de conteneur", "no de conteneur",
        "type de conteneur",
        "numéro de scellé", "plomb",
        "marchandise", "description des marchandises", "nature des marchandises",
        "poids brut",
        "volume", "cubage",
        "colis", "nombre de colis",
        "conditions de fret",
        "nombre d'originaux",
        # ── Espagnol ──────────────────────────────────────────────────────────
        "conocimiento de embarque", "número b/l", "numero b/l",
        "embarcador", "exportador",
        "consignatario",
        "buque", "nombre del buque",
        "puerto de carga", "puerto de embarque",
        "puerto de descarga", "destino",
        "fecha de embarque", "fecha de emisión",
        "contenedor", "número de contenedor",
        "sello", "número de sello",
        "descripción de mercancías",
        "peso bruto",
        "volumen",
        "bultos", "número de bultos",
        # ── Arabe translittéré ────────────────────────────────────────────────
        "سند الشحن", "رقم سند الشحن",
        "الشاحن", "المرسل إليه",
        "السفينة", "الرحلة",
        "ميناء الشحن", "ميناء التفريغ",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="BLExtractor — extraction champs Bill of Lading")
    parser.add_argument("--pdf",        required=True)
    parser.add_argument("--input-type", default="pdf_native",
                        choices=["pdf_native", "pdf_scanned", "image"])
    parser.add_argument("--model-dir",  default=None)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    ext    = BLExtractor(model_dir=args.model_dir)
    result = ext.extract_from_pdf(args.pdf, input_type=args.input_type)
    ext.print_results(result["fields"])

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Résultat sauvegardé → %s", args.output)


if __name__ == "__main__":
    main()
