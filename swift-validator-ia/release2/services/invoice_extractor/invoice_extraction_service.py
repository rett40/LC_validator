"""
invoice_extraction_service.py
==============================
Extracteur spécialisé Commercial Invoice.

Modèle      : models/models_layout/best_invoice/best_invoice/  (LayoutLMv3, 40 labels)
Segmentation: grid  (header_left / header_right / body_left / body_right / footer)

Champs extraits (20 types) :
  INV_NUMBER, ISSUE_DATE, LC_REF, BL_REF,
  SELLER, BUYER,
  VESSEL, VOYAGE, PORT_OF_LOADING, PORT_OF_DISCHARGE,
  GOODS_DESC, HS_CODE, QUANTITY, PACKAGE_UNIT,
  GROSS_WEIGHT, UNIT_PRICE, CURRENCY, TOTAL_AMOUNT,
  INCOTERM, PAYMENT_TERMS

Usage standalone :
  python -m release2.services.invoice_extractor.invoice_extraction_service --pdf invoice.pdf
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


class InvoiceExtractor(BaseLayoutExtractor):
    """
    Extracteur LayoutLMv3 spécialisé pour les Factures commerciales.

    Segmentation grid :
      header_left / header_right → INV_NUMBER, dates, références, Seller, Buyer
      body_left / body_right     → détails marchandises, montants
      footer                     → zones à rejeter pour les champs d'en-tête
    """

    MODEL_DIR      = _MODELS_LAYOUT / "best_invoice" / "best_invoice"
    TABLE_NAME     = "invoice_items"

    EXPECTED_FIELDS = {
        "INV_NUMBER", "ISSUE_DATE", "LC_REF", "BL_REF",
        "SELLER", "BUYER",
        "VESSEL", "VOYAGE", "PORT_OF_LOADING", "PORT_OF_DISCHARGE",
        "GOODS_DESC", "HS_CODE", "QUANTITY", "PACKAGE_UNIT",
        "GROSS_WEIGHT", "UNIT_PRICE", "CURRENCY", "TOTAL_AMOUNT",
        "INCOTERM", "PAYMENT_TERMS",
    }

    REJECT_IN_SEGMENT: dict[str, set[str]] = {
        "footer": {
            "INV_NUMBER", "ISSUE_DATE", "LC_REF", "BL_REF",
            "SELLER", "BUYER",
            "GOODS_DESC",
        },
        "header_left": {
            "UNIT_PRICE", "TOTAL_AMOUNT",
        },
        "header_right": {
            "UNIT_PRICE", "TOTAL_AMOUNT",
        },
        "body_left": {
            "BUYER",
        },
        "body_right": {
            "SELLER",
        },
    }

    NUMERIC_FIELDS = {
        "QUANTITY", "GROSS_WEIGHT", "UNIT_PRICE", "TOTAL_AMOUNT",
    }
    DATE_FIELDS = {"ISSUE_DATE"}

    FIELD_NAME_MAP = {
        "INV_NUMBER":     "Invoice Number",
        "ISSUE_DATE":     "Issue Date",
        "LC_REF":         "L/C Reference",
        "BL_REF":         "B/L Reference",
        "SELLER":         "Seller",
        "BUYER":          "Buyer",
        "VESSEL":         "Vessel",
        "VOYAGE":         "Voyage",
        "PORT_OF_LOADING":   "Port of Loading",
        "PORT_OF_DISCHARGE": "Port of Discharge",
        "GOODS_DESC":     "Goods Description",
        "HS_CODE":        "HS Code",
        "QUANTITY":       "Quantity",
        "PACKAGE_UNIT":   "Package Unit",
        "GROSS_WEIGHT":   "Gross Weight",
        "UNIT_PRICE":     "Unit Price",
        "CURRENCY":       "Currency",
        "TOTAL_AMOUNT":   "Total Amount",
        "INCOTERM":       "Incoterm",
        "PAYMENT_TERMS":  "Payment Terms",
    }

    # ── Labels connus dans les documents Invoice ──────────────────────────────
    FIELD_SYNONYMS: dict[str, list[str]] = {
        "INV_NUMBER":     ["INVOICE NUMBER", "INVOICE NO.", "INVOICE NO", "INV NO.",
                           "INV NO", "INV. NO", "INVOICE N°"],
        "ISSUE_DATE":     ["ISSUE DATE", "DATE OF ISSUE", "INVOICE DATE", "DATE ISSUED"],
        "LC_REF":         ["L/C REFERENCE", "L/C NO.", "L/C NO", "LC REF", "LC NUMBER",
                           "LETTER OF CREDIT NO", "CREDIT NUMBER"],
        "BL_REF":         ["B/L REFERENCE", "B/L NO.", "B/L NO", "BL REF", "BL NO."],
        "SELLER":         ["SELLER", "EXPORTER", "SELLER/EXPORTER", "VENDOR", "SUPPLIER"],
        "BUYER":          ["BUYER", "IMPORTER", "BUYER/IMPORTER", "CONSIGNEE", "BILL TO"],
        "VESSEL":         ["VESSEL", "VESSEL NAME", "M/V", "SHIP NAME"],
        "VOYAGE":         ["VOYAGE NO.", "VOYAGE NO", "VOYAGE NUMBER", "VOY"],
        "PORT_OF_LOADING":   ["PORT OF LOADING", "LOADING PORT", "POL"],
        "PORT_OF_DISCHARGE": ["PORT OF DISCHARGE", "DISCHARGE PORT", "POD"],
        "GOODS_DESC":     ["GOODS DESCRIPTION", "DESCRIPTION OF GOODS", "COMMODITY",
                           "DESCRIPTION", "GOODS"],
        "HS_CODE":        ["HS CODE", "HS NO.", "HS NO", "TARIFF CODE", "H.S. CODE"],
        "QUANTITY":       ["QUANTITY", "QTY", "NO. OF UNITS", "UNITS"],
        "PACKAGE_UNIT":   ["PACKAGE UNIT", "UNIT", "PACKING UNIT", "TYPE OF PACKING"],
        "GROSS_WEIGHT":   ["GROSS WEIGHT", "GROSS WT", "G.W."],
        "UNIT_PRICE":     ["UNIT PRICE", "PRICE PER UNIT", "UNIT RATE"],
        "CURRENCY":       ["CURRENCY", "CURR."],
        "TOTAL_AMOUNT":   ["TOTAL AMOUNT", "GRAND TOTAL", "INVOICE TOTAL", "TOTAL VALUE",
                           "AMOUNT", "TOTAL"],
        "INCOTERM":       ["INCOTERM", "INCOTERMS", "TRADE TERM"],
        "PAYMENT_TERMS":  ["PAYMENT TERMS", "TERMS OF PAYMENT", "PAYMENT CONDITIONS"],
    }

    # ── Champs formant les lignes de la facture (items) ───────────────────────
    TABLE_FIELDS     = {"GOODS_DESC", "HS_CODE", "QUANTITY", "UNIT_PRICE", "PACKAGE_UNIT"}
    ROW_ANCHOR_FIELD = "GOODS_DESC"
    ROW_FIELDS       = {"GOODS_DESC", "HS_CODE", "QUANTITY", "UNIT_PRICE",
                        "PACKAGE_UNIT", "GROSS_WEIGHT"}

    # ── Champs calculés : TOTAL_AMOUNT = Σ(QUANTITY × UNIT_PRICE) par ligne ───
    COMPUTED_SPECS = [
        ComputedFieldSpec(
            target="TOTAL_AMOUNT",
            op="sum_product",
            fields=["QUANTITY", "UNIT_PRICE"],
            tolerance=0.05,
        ),
    ]

    # ── Validation sémantique des lignes de facture ────────────────────────────
    ROW_VALIDATORS: dict = {
        "QUANTITY":   lambda v: bool(re.search(r'\d', v)),
        "UNIT_PRICE": lambda v: bool(re.search(r'\d', v)),
        "HS_CODE":    lambda v: bool(re.match(r'\d{4}', v.strip())),
    }

    FORBIDDEN_KEYWORDS: list[str] = [
        # ── Anglais ───────────────────────────────────────────────────────────
        "commercial invoice", "invoice", "invoice number", "invoice no",
        "invoice no.", "inv number", "inv no", "inv no.", "inv.",
        "issue date", "date of issue", "invoice date", "date issued",
        "lc ref", "lc reference", "l/c reference", "l/c no", "l/c number",
        "letter of credit", "credit number",
        "bl ref", "bl reference", "b/l reference", "b/l no", "b/l number",
        "seller", "exporter", "seller/exporter", "vendor", "supplier",
        "buyer", "importer", "buyer/importer", "consignee",
        "vessel", "vessel name", "ship name", "m/v",
        "voyage", "voyage no", "voyage no.", "voyage number",
        "port of loading", "loading port", "pol",
        "port of discharge", "discharge port", "pod",
        "goods description", "description of goods", "description",
        "goods", "commodity",
        "hs code", "hs no", "tariff code", "harmonized code",
        "quantity", "qty", "number of units",
        "package unit", "unit", "package type", "packing",
        "gross weight", "gross wt", "g.w.", "gross weight kg",
        "unit price", "price per unit", "unit rate",
        "currency", "curr.",
        "total amount", "total", "grand total", "amount",
        "incoterm", "incoterms", "trade term",
        "payment terms", "payment condition", "terms of payment",
        # ── Français ──────────────────────────────────────────────────────────
        "facture commerciale", "facture", "numéro de facture", "n° facture",
        "date d'émission", "date de facturation",
        "référence crédit documentaire", "réf. lc",
        "référence bl", "réf. bl",
        "vendeur", "exportateur", "fournisseur",
        "acheteur", "importateur", "destinataire",
        "navire", "nom du navire",
        "port de chargement", "port de déchargement",
        "description des marchandises", "désignation des marchandises",
        "code sh", "code douanier",
        "quantité",
        "type d'emballage", "unité d'emballage",
        "poids brut",
        "prix unitaire",
        "devise",
        "montant total", "valeur totale",
        "incoterm",
        "modalités de paiement", "conditions de paiement",
        # ── Espagnol ──────────────────────────────────────────────────────────
        "factura comercial", "factura", "número de factura",
        "fecha de emisión",
        "vendedor", "exportador",
        "comprador", "importador",
        "buque",
        "puerto de carga", "puerto de descarga",
        "descripción de mercancías",
        "código hs",
        "cantidad",
        "precio unitario",
        "moneda",
        "monto total", "importe total",
        "condiciones de pago",
        # ── Arabe translittéré ────────────────────────────────────────────────
        "فاتورة تجارية", "رقم الفاتورة",
        "البائع", "المشتري",
        "وصف البضاعة",
        "المبلغ الإجمالي",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="InvoiceExtractor — extraction champs Invoice")
    parser.add_argument("--pdf",        required=True)
    parser.add_argument("--input-type", default="pdf_native",
                        choices=["pdf_native", "pdf_scanned", "image"])
    parser.add_argument("--model-dir",  default=None)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    ext    = InvoiceExtractor(model_dir=args.model_dir)
    result = ext.extract_from_pdf(args.pdf, input_type=args.input_type)
    ext.print_results(result["fields"])

    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("Résultat sauvegardé → %s", args.output)


if __name__ == "__main__":
    main()
