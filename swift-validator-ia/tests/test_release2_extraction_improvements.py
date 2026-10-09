"""Regression tests for release2 commercial document extraction."""

import unittest

from release2.services.layout_analyzer import LayoutAnalyzer
from release2.services.native_extractor import _extract_party_fields
from release2.services.pipeline import DocumentPipeline


class TestPackingListTableExtraction(unittest.TestCase):
    def test_single_line_packing_table_is_reconstructed_from_headers(self):
        tokens = [
            ("DESCRIPTION", [50, 316, 125, 327]),
            ("PKGS", [279, 316, 310, 327]),
            ("UNIT", [374, 316, 400, 327]),
            ("DIMENSIONS", [470, 316, 539, 327]),
            ("(CM)", [542, 316, 566, 327]),
            ("NET", [603, 316, 625, 327]),
            ("WT", [628, 316, 645, 327]),
            ("(KG)", [648, 316, 671, 327]),
            ("GROSS", [717, 316, 757, 327]),
            ("WT", [760, 316, 777, 327]),
            ("(KG)", [780, 316, 803, 327]),
            ("CBM", [831, 316, 856, 327]),
            ("TOYS,", [52, 352, 90, 364]),
            ("GAMES", [93, 352, 139, 364]),
            ("AND", [142, 352, 169, 364]),
            ("SPORTS", [173, 352, 224, 364]),
            ("EQUIP", [228, 352, 267, 364]),
            ("293", [281, 352, 302, 364]),
            ("CARTONS", [376, 352, 438, 364]),
            ("55x45x38", [471, 352, 526, 364]),
            ("5,461.03", [605, 352, 654, 364]),
            ("6,472.60", [719, 352, 768, 364]),
            ("33.370", [833, 352, 872, 364]),
        ]

        result = LayoutAnalyzer().analyze(
            [word for word, _ in tokens],
            [bbox for _, bbox in tokens],
            "packing_list",
        )

        self.assertEqual(len(result["table_rows"]), 1)
        row = result["table_rows"][0]
        self.assertEqual(row["goods_description"], "TOYS, GAMES AND SPORTS EQUIP")
        self.assertEqual(row["number_of_packages"], "293")
        self.assertEqual(row["package_unit"], "CARTONS")
        self.assertEqual(row["dimensions"], "55x45x38")
        self.assertEqual(row["net_weight_kg"], "5,461.03")
        self.assertEqual(row["gross_weight_kg"], "6,472.60")
        self.assertEqual(row["measurement"], "33.370")


class TestPartyBlockExtraction(unittest.TestCase):
    def test_invoice_buyer_seller_prefer_company_line(self):
        text = "\n".join([
            "SELLER / EXPORTER",
            "ACME PHARMA LTD",
            "12 EXPORT ROAD",
            "SHANGHAI, CHINA",
            "BUYER / IMPORTER",
            "NOVA RETAIL LLC",
            "500 INDUSTRIAL AVENUE",
            "TUNIS, TUNISIA",
            "DESCRIPTION OF GOODS",
        ])

        fields = _extract_party_fields(text, "invoice")

        self.assertEqual(fields["seller"], "ACME PHARMA LTD")
        self.assertEqual(fields["buyer"], "NOVA RETAIL LLC")

    def test_packing_list_shipper_consignee_prefer_company_line(self):
        text = "\n".join([
            "EXPORTER / SHIPPER",
            "CENTRAL SUPPLY CHAIN CORP",
            "404 TRADE PARK",
            "BUSAN, SOUTH KOREA",
            "CONSIGNEE",
            "DELTA COMMERCE S.A.",
            "60 INDUSTRIAL AVENUE",
            "FELIXSTOWE, UK",
            "CONTAINER NO.",
        ])

        fields = _extract_party_fields(text, "packing_list")

        self.assertEqual(fields["shipper"], "CENTRAL SUPPLY CHAIN CORP")
        self.assertEqual(fields["consignee"], "DELTA COMMERCE S.A.")


class TestDocumentTableNormalization(unittest.TestCase):
    def test_invoice_amounts_with_currency_are_kept_as_item_rows(self):
        tables = DocumentPipeline._build_document_tables("invoice", [
            {
                "goods_description": "TOYS",
                "quantity": "293",
                "unit": "CARTONS",
                "unit_price": "GBP 14.11",
                "line_amount": "GBP 4,134.75",
            },
            {"unit_price": "GROSS WEIGHT:", "line_amount": "6,472.60 KG"},
        ])

        self.assertEqual(len(tables["invoice_items"]), 1)
        self.assertEqual(tables["invoice_items"][0]["unit_price"], "GBP 14.11")
        self.assertEqual(tables["invoice_items"][0]["line_amount"], "GBP 4,134.75")


if __name__ == "__main__":
    unittest.main()
