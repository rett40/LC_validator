"""
Commercial Invoice Synthetic Generator
=======================================
Reads bl_dataset/dataset.json and generates one Commercial Invoice PDF
per B/L, deriving all key fields (seller, buyer, goods, amount) directly
from the existing B/L data so cross-document consistency is guaranteed.

Discrepancies are injected in ~27% of dossiers (configurable).

Outputs:
  bl_dataset/invoices/*.pdf          — PDF invoices
  bl_dataset/invoices/annotations/   — per-invoice JSON with bbox annotations
  bl_dataset/lc_dataset.json         — master index: BL + Invoice linkage

Usage:
    python generate_invoice.py
    python generate_invoice.py --input ./bl_dataset --discrepancy-rate 0.27
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────
#  Commodity metadata tables
# ─────────────────────────────────────────────

COMMODITY_CATEGORIES = {
    "ELECTRONIC COMPONENTS AND PARTS":   "electronics",
    "TEXTILE FABRICS AND GARMENTS":      "textile",
    "AUTOMOTIVE PARTS AND ACCESSORIES":  "automotive",
    "INDUSTRIAL MACHINERY AND EQUIPMENT":"machinery",
    "PLASTIC GOODS AND ARTICLES":        "plastics",
    "FURNITURE AND PARTS THEREOF":       "furniture",
    "FOOTWEAR OF ALL TYPES":             "footwear",
    "TOYS, GAMES AND SPORTS EQUIPMENT":  "toys",
    "HOUSEHOLD APPLIANCES":              "appliances",
    "STEEL PRODUCTS AND STRUCTURES":     "steel",
    "CHEMICAL PRODUCTS NEC":             "chemicals",
    "AGRICULTURAL PRODUCTS AND FOODSTUFFS":"food",
    "PAPER AND PAPERBOARD ARTICLES":     "paper",
    "RUBBER PRODUCTS AND TYRES":         "rubber",
    "OPTICAL AND PHOTOGRAPHIC INSTRUMENTS":"optical",
    "MEDICAL DEVICES AND EQUIPMENT":     "medical",
    "BUILDING MATERIALS":                "construction",
    "PERSONAL CARE PRODUCTS":            "cosmetics",
    "CERAMIC TILES AND SANITARY WARE":   "ceramics",
    "COPPER AND COPPER PRODUCTS":        "metals",
    "WOOD PRODUCTS AND TIMBER":          "wood",
    "SEAFOOD AND FROZEN FISH PRODUCTS":  "food",
    "BEVERAGES AND SPIRITS":             "food",
    "SOLAR PANELS AND PHOTOVOLTAIC CELLS":"energy",
    "LITHIUM-ION BATTERIES":             "energy",
}

UNIT_PRICES = {
    "electronics":  (40.0,  250.0),
    "textile":      (5.0,   35.0),
    "automotive":   (25.0,  150.0),
    "machinery":    (80.0,  600.0),
    "plastics":     (3.0,   20.0),
    "furniture":    (15.0,  120.0),
    "footwear":     (8.0,   45.0),
    "toys":         (5.0,   30.0),
    "appliances":   (30.0,  200.0),
    "steel":        (8.0,   60.0),
    "chemicals":    (10.0,  80.0),
    "food":         (4.0,   25.0),
    "paper":        (3.0,   15.0),
    "rubber":       (6.0,   40.0),
    "optical":      (50.0,  400.0),
    "medical":      (30.0,  300.0),
    "construction": (5.0,   40.0),
    "cosmetics":    (8.0,   60.0),
    "ceramics":     (4.0,   30.0),
    "metals":       (10.0,  80.0),
    "wood":         (5.0,   35.0),
    "energy":       (20.0,  200.0),
}

HS_CODES = {
    "electronics": "8542.31", "textile": "5208.21", "automotive": "8708.99",
    "machinery": "8428.90", "plastics": "3926.90", "furniture": "9403.20",
    "footwear": "6404.11", "toys": "9503.00", "appliances": "8516.60",
    "steel": "7308.90", "chemicals": "3824.99", "food": "0709.99",
    "paper": "4819.10", "rubber": "4016.99", "optical": "9013.80",
    "medical": "9018.90", "construction": "6810.19", "cosmetics": "3304.99",
    "ceramics": "6908.90", "metals": "7408.19", "wood": "4421.99",
    "energy": "8541.40",
}

# Alternate descriptions used when injecting description_mismatch
ALTERNATIVE_DESCRIPTIONS = {
    "ELECTRONIC COMPONENTS AND PARTS":    "ELECTRONIC GOODS AND ACCESSORIES",
    "TEXTILE FABRICS AND GARMENTS":       "WOVEN FABRICS AND CLOTHING",
    "AUTOMOTIVE PARTS AND ACCESSORIES":   "VEHICLE COMPONENTS AND SPARES",
    "INDUSTRIAL MACHINERY AND EQUIPMENT": "MECHANICAL APPARATUS AND MACHINERY",
    "PLASTIC GOODS AND ARTICLES":         "POLYMER GOODS AND ARTICLES",
    "FURNITURE AND PARTS THEREOF":        "HOME FURNITURE AND FITTINGS",
    "FOOTWEAR OF ALL TYPES":              "SHOES AND LEATHER GOODS",
    "TOYS, GAMES AND SPORTS EQUIPMENT":   "LEISURE GOODS AND GAMES",
    "HOUSEHOLD APPLIANCES":               "DOMESTIC ELECTRICAL APPLIANCES",
    "STEEL PRODUCTS AND STRUCTURES":      "IRON AND STEEL ARTICLES",
    "CHEMICAL PRODUCTS NEC":              "MISCELLANEOUS CHEMICAL PREPARATIONS",
    "AGRICULTURAL PRODUCTS AND FOODSTUFFS":"FOOD PRODUCTS AND PROVISIONS",
    "PAPER AND PAPERBOARD ARTICLES":      "PAPER PRODUCTS AND STATIONERY",
    "RUBBER PRODUCTS AND TYRES":          "VULCANIZED RUBBER ARTICLES",
    "OPTICAL AND PHOTOGRAPHIC INSTRUMENTS":"PRECISION OPTICAL INSTRUMENTS",
    "MEDICAL DEVICES AND EQUIPMENT":      "HEALTHCARE EQUIPMENT AND SUPPLIES",
    "BUILDING MATERIALS":                 "CONSTRUCTION MATERIALS NEC",
    "PERSONAL CARE PRODUCTS":             "COSMETICS AND TOILETRIES",
    "CERAMIC TILES AND SANITARY WARE":    "CERAMIC PRODUCTS AND TILES",
    "COPPER AND COPPER PRODUCTS":         "COPPER ALLOYS AND ARTICLES",
    "WOOD PRODUCTS AND TIMBER":           "TIMBER AND WOOD ARTICLES",
    "SEAFOOD AND FROZEN FISH PRODUCTS":   "FROZEN SEAFOOD AND FISH",
    "BEVERAGES AND SPIRITS":              "ALCOHOLIC AND NON-ALCOHOLIC BEVERAGES",
    "SOLAR PANELS AND PHOTOVOLTAIC CELLS":"PHOTOVOLTAIC MODULES AND CELLS",
    "LITHIUM-ION BATTERIES":              "RECHARGEABLE BATTERY CELLS",
}

PAYMENT_TERMS = [
    "AT SIGHT",
    "30 DAYS AFTER DATE OF BILL OF LADING",
    "60 DAYS AFTER DATE OF SHIPMENT",
    "DEFERRED PAYMENT 90 DAYS",
    "USANCE 60 DAYS",
]

INCOTERMS = ["FOB", "CIF", "CFR", "EXW", "DAP", "DDP", "FCA", "CIP"]

ALT_PORTS = [
    "HAMBURG, GERMANY", "SINGAPORE", "BUSAN, SOUTH KOREA",
    "NEW YORK, USA", "ANTWERP, BELGIUM", "LE HAVRE, FRANCE",
    "TOKYO, JAPAN", "MUMBAI, INDIA", "VALENCIA, SPAIN",
]


# ─────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────

def _parse_packages(s: str) -> tuple:
    """'150 CARTONS' → (150, 'CARTONS')"""
    parts = s.strip().split(None, 1)
    try:
        count = int(parts[0].replace(",", ""))
        unit = parts[1] if len(parts) > 1 else "PACKAGES"
    except (ValueError, IndexError):
        count = 100
        unit = "PACKAGES"
    return count, unit


def _parse_weight(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 10000.0


def _city_country(port_str: str) -> tuple:
    """'SHANGHAI, CHINA' → ('SHANGHAI', 'CHINA')"""
    if "," in port_str:
        city, country = port_str.split(",", 1)
        return city.strip(), country.strip()
    return port_str.strip(), "N/A"


def _build_address(company_name: str, port_str: str, rng: random.Random) -> str:
    city, country = _city_country(port_str)
    num = rng.randint(1, 999)
    street = rng.choice(["INDUSTRIAL AVENUE", "TRADE PARK", "COMMERCE STREET",
                          "HARBOR ROAD", "MAIN BOULEVARD", "EXPORT ZONE"])
    return f"{company_name}\n{num} {street}\n{city}, {country}"


# ─────────────────────────────────────────────
#  Invoice data builder (source of truth)
# ─────────────────────────────────────────────

def build_invoice_data(bl_record: dict, rng: random.Random, disc_rate: float) -> dict:
    """Derive all invoice fields from the B/L record, inject discrepancies if chosen."""
    f = bl_record["fields"]
    bl_id = bl_record["id"]

    qty, pkg_unit = _parse_packages(f["packages"])
    commodity = f["commodity"]
    category = COMMODITY_CATEGORIES.get(commodity, "machinery")

    lo, hi = UNIT_PRICES.get(category, (20.0, 150.0))
    unit_price = round(rng.uniform(lo, hi), 2)
    total_amount = round(unit_price * qty, 2)
    hs_code = HS_CODES.get(category, "9999.99")

    try:
        bl_date = datetime.strptime(f["issue_date"], "%d %b %Y")
        inv_date = bl_date - timedelta(days=rng.randint(0, 3))
    except (ValueError, KeyError):
        inv_date = datetime.now()
    invoice_date = inv_date.strftime("%d %b %Y").upper()

    currency = "USD"
    incoterm = rng.choice(INCOTERMS)
    payment_terms = rng.choice(PAYMENT_TERMS)

    # Derive addresses deterministically from names + ports
    seller_address = _build_address(f["shipper_name"], f["port_of_loading"], rng)
    buyer_address = _build_address(f["consignee_name"], f["port_of_discharge"], rng)

    # LC reference
    bank = rng.choice(["HSBC", "CITI", "BNP", "SCB", "DBS", "BARCLAYS", "DB"])
    lc_ref = f"LC/{bank}/{bl_id:04d}/2026"

    is_consistent = True
    discrepancies = []
    port_of_loading = f["port_of_loading"]  # may be overwritten by discrepancy

    if rng.random() < disc_rate:
        is_consistent = False
        possible = ["amount_mismatch", "description_mismatch", "port_mismatch", "currency_mismatch"]
        chosen = rng.sample(possible, rng.randint(1, 2))

        for disc_type in chosen:
            if disc_type == "amount_mismatch":
                factor = rng.uniform(1.06, 1.18) if rng.random() > 0.4 else rng.uniform(0.83, 0.94)
                original = total_amount
                total_amount = round(total_amount * factor, 2)
                unit_price = round(total_amount / max(qty, 1), 2)
                discrepancies.append({
                    "type": "amount_mismatch",
                    "field": "total_amount",
                    "expected": f"{original:.2f} {currency}",
                    "actual": f"{total_amount:.2f} {currency}",
                })

            elif disc_type == "description_mismatch":
                alt = ALTERNATIVE_DESCRIPTIONS.get(commodity, commodity + " NEC")
                discrepancies.append({
                    "type": "description_mismatch",
                    "field": "goods_description",
                    "expected": commodity,
                    "actual": alt,
                })
                commodity = alt

            elif disc_type == "port_mismatch":
                alt_pol = rng.choice([p for p in ALT_PORTS if p != f["port_of_loading"]])
                discrepancies.append({
                    "type": "port_mismatch",
                    "field": "port_of_loading",
                    "expected": f["port_of_loading"],
                    "actual": alt_pol,
                })
                port_of_loading = alt_pol

            elif disc_type == "currency_mismatch":
                alt_ccy = rng.choice(["EUR", "GBP"])
                discrepancies.append({
                    "type": "currency_mismatch",
                    "field": "currency",
                    "expected": "USD",
                    "actual": alt_ccy,
                })
                currency = alt_ccy

    return {
        "invoice_number": f"INV-2026-{bl_id:05d}",
        "invoice_date": invoice_date,
        "lc_reference": lc_ref,
        "bl_reference": f["bl_number"],
        "seller_name": f["shipper_name"],
        "seller_address": seller_address,
        "buyer_name": f["consignee_name"],
        "buyer_address": buyer_address,
        "vessel": f["vessel"],
        "voyage": f["voyage"],
        "port_of_loading": port_of_loading,
        "port_of_discharge": f["port_of_discharge"],
        "goods_description": commodity,
        "hs_code": hs_code,
        "quantity": qty,
        "package_unit": pkg_unit,
        "gross_weight_kg": f["gross_weight_kg"],
        "unit_price": unit_price,
        "currency": currency,
        "total_amount": total_amount,
        "incoterm": incoterm,
        "payment_terms": payment_terms,
        "is_consistent": is_consistent,
        "discrepancies": discrepancies,
    }


# ─────────────────────────────────────────────
#  Base renderer
# ─────────────────────────────────────────────

class InvoiceRenderer:
    PAGE_W, PAGE_H = A4
    MARGIN = 1.0 * cm

    def __init__(self, output_path: str, data: dict):
        self.path = output_path
        self.data = data
        self.c = canvas.Canvas(output_path, pagesize=A4)
        self.annotations = []

    def _annotate(self, field, x, y, w, h, value):
        self.annotations.append({
            "field": field,
            "bbox": [round(x, 2), round(self.PAGE_H - y - h, 2), round(w, 2), round(h, 2)],
            "value": str(value),
            "page": 1,
        })

    def _text(self, x, y, text, font="Helvetica", size=8, color=colors.black):
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        self.c.drawString(x, y, str(text))

    def _rtext(self, x, y, text, font="Helvetica", size=8, color=colors.black):
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        self.c.drawRightString(x, y, str(text))

    def _ctext(self, x, y, text, font="Helvetica", size=8, color=colors.white):
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        self.c.drawCentredString(x, y, str(text))

    def _multiline(self, x, y, text, font="Helvetica", size=7.5, line_h=None, color=colors.black):
        if line_h is None:
            line_h = size * 1.35
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        for line in str(text).split("\n"):
            self.c.drawString(x, y, line)
            y -= line_h
        return y

    def _labeled_box(self, x, y, w, h, label, value, label_h=0.38*cm,
                     label_bg=None, box_bg=None, border=None,
                     font_size=7.5, multiline=False,
                     field_name=None, label_color=colors.white):
        c = self.c
        label_bg = label_bg or colors.HexColor("#1A3C5E")
        box_bg = box_bg or colors.HexColor("#F7F9FC")
        border = border or colors.HexColor("#CCCCCC")

        # Label band
        c.setFillColor(label_bg)
        c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
        c.setFillColor(label_color)
        c.setFont("Helvetica-Bold", 6)
        c.drawString(x + 2, y - label_h + 3, label)

        # Content box
        content_h = h - label_h
        c.setFillColor(box_bg)
        c.setStrokeColor(border)
        c.rect(x, y - h, w, content_h, fill=1, stroke=1)

        # Value
        c.setFillColor(colors.black)
        if multiline:
            lines = str(value).split("\n")
            ly = y - label_h - font_size * 1.2
            c.setFont("Helvetica", font_size)
            for line in lines:
                if ly < y - h + 2:
                    break
                c.drawString(x + 3, ly, line)
                ly -= font_size * 1.35
        else:
            c.setFont("Helvetica", font_size)
            text = str(value)
            max_w = w - 6
            while c.stringWidth(text, "Helvetica", font_size) > max_w and len(text) > 4:
                text = text[:-1]
            ty = y - label_h - (content_h / 2) - font_size / 3
            c.drawString(x + 3, ty, text)

        if field_name:
            self._annotate(field_name, x, y - h, w, h, value)

    def render(self):
        raise NotImplementedError

    def save(self):
        self.c.save()
        return self.annotations


# ─────────────────────────────────────────────
#  Style A — Standard Blue  (~133 invoices)
# ─────────────────────────────────────────────

class StandardInvoiceRenderer(InvoiceRenderer):
    PRIMARY  = colors.HexColor("#1A3C5E")
    ACCENT   = colors.HexColor("#2E86C1")
    BG_LIGHT = colors.HexColor("#F0F4F8")
    BORDER   = colors.HexColor("#B0BEC5")
    TABLE_HD = colors.HexColor("#1A3C5E")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # ── Header ──
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.6*cm, W, 2.6*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.6*cm, 0.6*cm, 2.6*cm, fill=1, stroke=0)
        self._ctext(W/2, H - 1.3*cm, "COMMERCIAL INVOICE",
                    "Helvetica-Bold", 18)
        c.setFillColor(colors.HexColor("#A9CCE3"))
        c.setFont("Helvetica", 7.5)
        c.drawRightString(W - M, H - 2.15*cm, "ORIGINAL")

        # ── Invoice meta band ──
        c.setFillColor(self.BG_LIGHT)
        c.rect(0, H - 3.2*cm, W, 0.6*cm, fill=1, stroke=0)
        c.setStrokeColor(self.BORDER)
        c.line(0, H - 3.2*cm, W, H - 3.2*cm)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 2.95*cm, f"INVOICE NO: {d['invoice_number']}")
        c.drawString(W/3, H - 2.95*cm, f"DATE: {d['invoice_date']}")
        c.drawRightString(W - M, H - 2.95*cm, f"LC REF: {d['lc_reference']}")
        self._annotate("invoice_number", M, H - 3.2*cm, 6*cm, 0.6*cm, d['invoice_number'])
        self._annotate("lc_reference", W/2, H - 3.2*cm, W/2 - M, 0.6*cm, d['lc_reference'])

        # ── Seller / Buyer ──
        top = H - 3.5*cm
        half = (W - 2*M) / 2
        row_h = 2.5*cm
        self._labeled_box(M, top, half, row_h, "SELLER / EXPORTER",
                          d['seller_address'], label_bg=self.PRIMARY,
                          box_bg=self.BG_LIGHT, border=self.BORDER,
                          font_size=7.5, multiline=True,
                          field_name="seller_name")
        self._labeled_box(M + half, top, half, row_h, "BUYER / IMPORTER",
                          d['buyer_address'], label_bg=self.PRIMARY,
                          box_bg=colors.white, border=self.BORDER,
                          font_size=7.5, multiline=True,
                          field_name="buyer_name")

        # ── Shipment info row ──
        y = top - row_h - 0.15*cm
        q4 = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",             d['vessel'],           "vessel"),
            ("VOYAGE NO.",         d['voyage'],           "voyage"),
            ("PORT OF LOADING",    d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE",  d['port_of_discharge'],"port_of_discharge"),
        ]):
            self._labeled_box(M + i*q4, y, q4, 1.1*cm, lbl, val,
                              label_bg=self.ACCENT, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=6.5, field_name=fn)

        # ── B/L reference row ──
        y -= 1.1*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE",  d['bl_reference'],   "bl_reference"),
            ("INCOTERM",       d['incoterm'],        None),
            ("PAYMENT TERMS",  d['payment_terms'],   None),
        ]):
            self._labeled_box(M + i*thirds, y, thirds, 1.0*cm, lbl, val,
                              label_bg=self.PRIMARY, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=7.0, field_name=fn)

        # ── Goods table ──
        y -= 1.0*cm + 0.3*cm
        col_w = [5.5*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION OF GOODS", "HS CODE", "QTY", "UNIT", "UNIT PRICE", "AMOUNT"]
        total_w = W - 2*M
        hdr_h = 0.5*cm
        row_h2 = 1.6*cm

        # Table header
        c.setFillColor(self.TABLE_HD)
        c.rect(M, y - hdr_h, total_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 6.5)
        cx = M
        for hdr, cw in zip(hdrs, col_w):
            c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw

        # Table row
        c.setFillColor(colors.HexColor("#EBF5FB"))
        c.setStrokeColor(self.BORDER)
        c.rect(M, y - hdr_h - row_h2, total_w, row_h2, fill=1, stroke=1)

        # Vertical separators
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.setStrokeColor(self.BORDER)
            c.line(cx, y - hdr_h, cx, y - hdr_h - row_h2)

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        vals = [
            d['goods_description'],
            d['hs_code'],
            str(d['quantity']),
            d['package_unit'],
            f"{d['currency']} {d['unit_price']:,.2f}",
            f"{d['currency']} {d['total_amount']:,.2f}",
        ]
        field_names = ["goods_description", None, "quantity", None, "unit_price", "total_amount"]
        cx = M
        for val, cw, fn in zip(vals, col_w, field_names):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, y - hdr_h - row_h2/2 - 3, text)
            if fn:
                self._annotate(fn, cx, y - hdr_h - row_h2, cw, row_h2, val)
            cx += cw

        # ── Totals block ──
        y -= hdr_h + row_h2 + 0.2*cm
        tot_x = M + total_w * 0.6
        tot_w = total_w * 0.4

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tot_x, y - 1.8*cm, tot_w, 1.8*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        c.drawString(tot_x + 3, y - 0.5*cm, "GROSS WEIGHT:")
        c.drawRightString(tot_x + tot_w - 3, y - 0.5*cm, f"{d['gross_weight_kg']} KG")
        c.line(tot_x, y - 0.8*cm, tot_x + tot_w, y - 0.8*cm)
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(self.PRIMARY)
        c.drawString(tot_x + 3, y - 1.3*cm, "TOTAL AMOUNT:")
        c.drawRightString(tot_x + tot_w - 3, y - 1.3*cm,
                          f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("total_amount", tot_x, y - 1.8*cm, tot_w, 1.8*cm,
                       f"{d['currency']} {d['total_amount']:,.2f}")

        # ── Currency field annotation ──
        self._annotate("currency", tot_x, y - 1.8*cm, tot_w/2, 0.6*cm, d['currency'])

        # ── Certification & Signature ──
        y -= 1.8*cm + 0.3*cm
        c.setFont("Helvetica", 6.5)
        c.setFillColor(colors.HexColor("#555555"))
        cert = ("We hereby certify that the goods described herein are of the stated origin and "
                "that the prices shown are the actual prices charged to the buyer.")
        c.drawString(M, y, cert)
        y -= 1.2*cm
        c.setStrokeColor(self.PRIMARY)
        c.line(M, y, M + 5*cm, y)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATURE & STAMP")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['seller_name'])
        c.drawString(M + 6*cm, y - 0.35*cm, d['invoice_date'])

        # ── Footer ──
        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"COMMERCIAL INVOICE  |  {d['invoice_number']}  |  B/L REF: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style B — Dark Navy  (~133 invoices)
# ─────────────────────────────────────────────

class NavyInvoiceRenderer(InvoiceRenderer):
    PRIMARY  = colors.HexColor("#0D1B2A")
    ACCENT   = colors.HexColor("#E63946")
    BG_LIGHT = colors.HexColor("#F8F9FA")
    BORDER   = colors.HexColor("#ADB5BD")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Header: full-width dark band + red stripe
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 3.0*cm, W, 3.0*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 3.0*cm, W, 0.25*cm, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(M, H - 1.7*cm, "COMMERCIAL INVOICE")
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#CED4DA"))
        c.drawString(M, H - 2.2*cm, f"No. {d['invoice_number']}   |   Date: {d['invoice_date']}")
        c.drawString(M, H - 2.6*cm, f"LC Reference: {d['lc_reference']}   |   B/L: {d['bl_reference']}")
        self._annotate("invoice_number", M, H - 3.0*cm, 6*cm, 3.0*cm, d['invoice_number'])
        self._annotate("bl_reference", M + 8*cm, H - 3.0*cm, 6*cm, 3.0*cm, d['bl_reference'])

        top = H - 3.3*cm
        half = (W - 2*M) / 2
        row_h = 2.6*cm
        lbl_h = 0.38*cm

        def box(x, y, w, h, label, val, ml=False, fn=None):
            self._labeled_box(x, y, w, h, label, val,
                              label_h=lbl_h, label_bg=self.PRIMARY,
                              box_bg=self.BG_LIGHT, border=self.BORDER,
                              font_size=7.5, multiline=ml, field_name=fn)

        box(M, top, half, row_h, "SELLER", d['seller_address'], ml=True, fn="seller_name")
        box(M + half, top, half, row_h, "BUYER", d['buyer_address'], ml=True, fn="buyer_name")

        y = top - row_h - 0.15*cm
        q = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE",            d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            box(M + i*q, y, q, 1.1*cm, lbl, val, fn=fn)

        y -= 1.1*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE", d['bl_reference'], "bl_reference"),
            ("INCOTERM",      d['incoterm'],      None),
            ("PAYMENT TERMS", d['payment_terms'], None),
        ]):
            box(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        # ── Goods table ──
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [5.5*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION OF GOODS", "HS CODE", "QTY", "UNIT", "UNIT PRICE", "AMOUNT"]
        hdr_h, row_h2 = 0.5*cm, 1.6*cm

        c.setFillColor(self.PRIMARY)
        c.rect(M, y - hdr_h, total_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 6.5)
        cx = M
        for hdr, cw in zip(hdrs, col_w):
            c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, y - hdr_h - row_h2, total_w, row_h2, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, y - hdr_h, cx, y - hdr_h - row_h2)

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        vals = [
            d['goods_description'], d['hs_code'], str(d['quantity']),
            d['package_unit'],
            f"{d['currency']} {d['unit_price']:,.2f}",
            f"{d['currency']} {d['total_amount']:,.2f}",
        ]
        fns = ["goods_description", None, "quantity", None, "unit_price", "total_amount"]
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, y - hdr_h - row_h2/2 - 3, text)
            if fn:
                self._annotate(fn, cx, y - hdr_h - row_h2, cw, row_h2, val)
            cx += cw

        # Totals
        y -= hdr_h + row_h2 + 0.2*cm
        tx = M + total_w * 0.6
        tw = total_w * 0.4
        c.setFillColor(self.PRIMARY)
        c.rect(tx, y - 0.8*cm, tw, 0.8*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(tx + 3, y - 0.55*cm, "TOTAL AMOUNT:")
        c.drawRightString(tx + tw - 3, y - 0.55*cm,
                          f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("total_amount", tx, y - 0.8*cm, tw, 0.8*cm,
                       f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("currency", tx, y - 0.8*cm, tw/3, 0.8*cm, d['currency'])

        c.setStrokeColor(self.BORDER)
        c.setFillColor(self.BG_LIGHT)
        c.rect(tx, y - 1.5*cm, tw, 0.7*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(tx + 3, y - 1.2*cm, "GROSS WEIGHT:")
        c.drawRightString(tx + tw - 3, y - 1.2*cm, f"{d['gross_weight_kg']} KG")

        # Signature
        y -= 1.5*cm + 0.5*cm
        c.setStrokeColor(self.ACCENT)
        c.setLineWidth(1.5)
        c.line(M, y, M + 5*cm, y)
        c.setLineWidth(1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['seller_name'])

        # Footer
        c.setFillColor(self.ACCENT)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"{d['invoice_number']}  |  B/L: {d['bl_reference']}  |  ORIGINAL COPY")


# ─────────────────────────────────────────────
#  Style C — Green Trading  (~134 invoices)
# ─────────────────────────────────────────────

class TradingInvoiceRenderer(InvoiceRenderer):
    PRIMARY  = colors.HexColor("#1B5E20")
    ACCENT   = colors.HexColor("#4CAF50")
    BG_LIGHT = colors.HexColor("#F1F8E9")
    BORDER   = colors.HexColor("#A5D6A7")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Green top bar
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 0.4*cm, W, 0.4*cm, fill=1, stroke=0)

        # Title area
        c.setFillColor(self.BG_LIGHT)
        c.rect(0, H - 2.8*cm, W, 2.4*cm, fill=1, stroke=0)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 19)
        c.drawString(M, H - 1.6*cm, "COMMERCIAL INVOICE")
        c.setFont("Helvetica", 8)
        c.drawString(M, H - 2.1*cm, f"Invoice No: {d['invoice_number']}   |   Date: {d['invoice_date']}")
        c.drawString(M, H - 2.5*cm, f"LC Ref: {d['lc_reference']}   |   B/L Ref: {d['bl_reference']}")
        self._annotate("invoice_number", M, H - 2.8*cm, 6*cm, 2.4*cm, d['invoice_number'])
        self._annotate("bl_reference", M, H - 2.8*cm, W - 2*M, 0.5*cm, d['bl_reference'])

        # Green divider
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.8*cm, W, 0.1*cm, fill=1, stroke=0)

        top = H - 3.1*cm
        half = (W - 2*M) / 2
        lbl_h = 0.36*cm

        def gbox(x, y, w, h, label, val, ml=False, fn=None):
            self._labeled_box(x, y, w, h, label, val,
                              label_h=lbl_h, label_bg=self.PRIMARY,
                              box_bg=self.BG_LIGHT, border=self.BORDER,
                              font_size=7.5, multiline=ml, field_name=fn)

        gbox(M, top, half, 2.6*cm, "SELLER / EXPORTER",
             d['seller_address'], ml=True, fn="seller_name")
        gbox(M + half, top, half, 2.6*cm, "BUYER / IMPORTER",
             d['buyer_address'], ml=True, fn="buyer_name")

        y = top - 2.6*cm - 0.15*cm
        q = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE NO.",        d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            gbox(M + i*q, y, q, 1.1*cm, lbl, val, fn=fn)

        y -= 1.1*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE", d['bl_reference'], "bl_reference"),
            ("INCOTERM",      d['incoterm'],      None),
            ("PAYMENT TERMS", d['payment_terms'], None),
        ]):
            gbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        # ── Goods table ──
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [5.5*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION OF GOODS", "HS CODE", "QTY", "UNIT", "UNIT PRICE", "AMOUNT"]
        hdr_h, row_h2 = 0.5*cm, 1.6*cm

        c.setFillColor(self.PRIMARY)
        c.rect(M, y - hdr_h, total_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 6.5)
        cx = M
        for hdr, cw in zip(hdrs, col_w):
            c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, y - hdr_h - row_h2, total_w, row_h2, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, y - hdr_h, cx, y - hdr_h - row_h2)

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        vals = [
            d['goods_description'], d['hs_code'], str(d['quantity']),
            d['package_unit'],
            f"{d['currency']} {d['unit_price']:,.2f}",
            f"{d['currency']} {d['total_amount']:,.2f}",
        ]
        fns = ["goods_description", None, "quantity", None, "unit_price", "total_amount"]
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, y - hdr_h - row_h2/2 - 3, text)
            if fn:
                self._annotate(fn, cx, y - hdr_h - row_h2, cw, row_h2, val)
            cx += cw

        # Totals
        y -= hdr_h + row_h2 + 0.2*cm
        tx = M + total_w * 0.6
        tw = total_w * 0.4

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tx, y - 1.8*cm, tw, 1.8*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        c.drawString(tx + 3, y - 0.5*cm, "GROSS WEIGHT:")
        c.drawRightString(tx + tw - 3, y - 0.5*cm, f"{d['gross_weight_kg']} KG")
        c.setStrokeColor(self.ACCENT)
        c.line(tx + 3, y - 0.8*cm, tx + tw - 3, y - 0.8*cm)
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(self.PRIMARY)
        c.drawString(tx + 3, y - 1.3*cm, "TOTAL AMOUNT:")
        c.drawRightString(tx + tw - 3, y - 1.3*cm,
                          f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("total_amount", tx, y - 1.8*cm, tw, 1.8*cm,
                       f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("currency", tx, y - 1.8*cm, tw/3, 0.6*cm, d['currency'])

        # Signature
        y -= 1.8*cm + 0.5*cm
        c.setStrokeColor(self.PRIMARY)
        c.setLineWidth(1.2)
        c.line(M, y, M + 5*cm, y)
        c.setLineWidth(1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY & STAMP")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['seller_name'])

        # Footer
        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"COMMERCIAL INVOICE  |  {d['invoice_number']}  |  B/L: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style D — Minimal Clean  (slate + light gray)
# ─────────────────────────────────────────────

class MinimalInvoiceRenderer(InvoiceRenderer):
    PRIMARY  = colors.HexColor("#2C3E50")
    ACCENT   = colors.HexColor("#7F8C8D")
    BG_LIGHT = colors.HexColor("#FAFAFA")
    BORDER   = colors.HexColor("#DEE2E6")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Thin top rule + title (no filled header block)
        c.setStrokeColor(self.PRIMARY)
        c.setLineWidth(2)
        c.line(M, H - 0.8*cm, W - M, H - 0.8*cm)
        c.setLineWidth(1)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(M, H - 1.6*cm, "COMMERCIAL INVOICE")
        c.setFont("Helvetica", 8)
        c.setFillColor(self.ACCENT)
        c.drawRightString(W - M, H - 1.25*cm, "ORIGINAL")

        # Meta line
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, H - 2.5*cm, W - 2*M, 0.8*cm, fill=1, stroke=1)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(M + 3, H - 2.2*cm, f"INVOICE NO: {d['invoice_number']}")
        c.drawString(W/3, H - 2.2*cm, f"DATE: {d['invoice_date']}")
        c.drawRightString(W - M - 3, H - 2.2*cm, f"LC REF: {d['lc_reference']}")
        self._annotate("invoice_number", M, H - 2.5*cm, 6*cm, 0.8*cm, d['invoice_number'])
        self._annotate("lc_reference", W/2, H - 2.5*cm, W/2 - M, 0.8*cm, d['lc_reference'])

        top = H - 2.8*cm
        half = (W - 2*M) / 2
        lbl_h = 0.36*cm

        def mbox(x, y, w, h, label, val, ml=False, fn=None, fs=7.5):
            c.setFillColor(self.ACCENT)
            c.rect(x, y - lbl_h, w, lbl_h, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - lbl_h + 3, label)
            c.setFillColor(self.BG_LIGHT)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, h - lbl_h, fill=1, stroke=1)
            c.setFillColor(colors.black)
            if ml:
                lines = str(val).split("\n")
                ly = y - lbl_h - fs * 1.2
                c.setFont("Helvetica", fs)
                for line in lines:
                    if ly < y - h + 2:
                        break
                    c.drawString(x + 3, ly, line)
                    ly -= fs * 1.35
            else:
                c.setFont("Helvetica", fs)
                text = str(val)
                while c.stringWidth(text, "Helvetica", fs) > w - 6 and len(text) > 3:
                    text = text[:-1]
                c.drawString(x + 3, y - lbl_h - (h - lbl_h)/2 - fs/3, text)
            if fn:
                self._annotate(fn, x, y - h, w, h, val)

        mbox(M, top, half, 2.5*cm, "SELLER / EXPORTER",
             d['seller_address'], ml=True, fn="seller_name")
        mbox(M + half, top, half, 2.5*cm, "BUYER / IMPORTER",
             d['buyer_address'], ml=True, fn="buyer_name")

        y = top - 2.5*cm - 0.15*cm
        q = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE NO.",        d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            mbox(M + i*q, y, q, 1.1*cm, lbl, val, fn=fn, fs=6.5)

        y -= 1.1*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE",  d['bl_reference'],   "bl_reference"),
            ("INCOTERM",       d['incoterm'],        None),
            ("PAYMENT TERMS",  d['payment_terms'],   None),
        ]):
            mbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn, fs=7.0)

        # Goods table
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [5.5*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION OF GOODS", "HS CODE", "QTY", "UNIT", "UNIT PRICE", "AMOUNT"]
        hdr_h, row_h2 = 0.45*cm, 1.6*cm

        c.setFillColor(self.PRIMARY)
        c.rect(M, y - hdr_h, total_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 6.5)
        cx = M
        for hdr, cw in zip(hdrs, col_w):
            c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, y - hdr_h - row_h2, total_w, row_h2, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, y - hdr_h, cx, y - hdr_h - row_h2)

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        vals = [
            d['goods_description'], d['hs_code'], str(d['quantity']),
            d['package_unit'],
            f"{d['currency']} {d['unit_price']:,.2f}",
            f"{d['currency']} {d['total_amount']:,.2f}",
        ]
        fns = ["goods_description", None, "quantity", None, "unit_price", "total_amount"]
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, y - hdr_h - row_h2/2 - 3, text)
            if fn:
                self._annotate(fn, cx, y - hdr_h - row_h2, cw, row_h2, val)
            cx += cw

        # Totals
        y -= hdr_h + row_h2 + 0.2*cm
        tx = M + total_w * 0.6
        tw = total_w * 0.4
        c.setStrokeColor(self.BORDER)
        c.setFillColor(self.BG_LIGHT)
        c.rect(tx, y - 1.8*cm, tw, 1.8*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        c.drawString(tx + 3, y - 0.5*cm, "GROSS WEIGHT:")
        c.drawRightString(tx + tw - 3, y - 0.5*cm, f"{d['gross_weight_kg']} KG")
        c.line(tx + 3, y - 0.8*cm, tx + tw - 3, y - 0.8*cm)
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(self.PRIMARY)
        c.drawString(tx + 3, y - 1.3*cm, "TOTAL AMOUNT:")
        c.drawRightString(tx + tw - 3, y - 1.3*cm,
                          f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("total_amount", tx, y - 1.8*cm, tw, 1.8*cm,
                       f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("currency", tx, y - 1.8*cm, tw/2, 0.6*cm, d['currency'])

        # Signature + footer rule
        y -= 1.8*cm + 0.5*cm
        c.setStrokeColor(self.ACCENT)
        c.line(M, y, M + 5*cm, y)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['seller_name'])

        c.setStrokeColor(self.PRIMARY)
        c.setLineWidth(1.5)
        c.line(M, 0.9*cm, W - M, 0.9*cm)
        c.setLineWidth(1)
        c.setFont("Helvetica", 6)
        c.setFillColor(self.ACCENT)
        c.drawCentredString(W/2, 0.55*cm,
            f"COMMERCIAL INVOICE  |  {d['invoice_number']}  |  B/L: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style E — Bold Amber  (black + amber accent)
# ─────────────────────────────────────────────

class BoldInvoiceRenderer(InvoiceRenderer):
    PRIMARY  = colors.HexColor("#212121")
    ACCENT   = colors.HexColor("#FFA000")
    BG_LIGHT = colors.HexColor("#FFFDE7")
    BORDER   = colors.HexColor("#FFE082")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Full-width black header
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.8*cm, W, 2.8*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.8*cm, W, 0.3*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(M, H - 1.5*cm, "COMMERCIAL INVOICE")
        c.setFont("Helvetica", 8)
        c.setFillColor(self.ACCENT)
        c.drawString(M, H - 2.0*cm, f"No. {d['invoice_number']}   |   Date: {d['invoice_date']}")
        c.drawString(M, H - 2.45*cm, f"LC: {d['lc_reference']}   |   B/L: {d['bl_reference']}")
        self._annotate("invoice_number", M, H - 2.8*cm, 6*cm, 2.8*cm, d['invoice_number'])
        self._annotate("bl_reference", M + 8*cm, H - 2.8*cm, 6*cm, 0.5*cm, d['bl_reference'])

        top = H - 3.1*cm
        half = (W - 2*M) / 2
        lbl_h = 0.38*cm

        def bbox(x, y, w, h, label, val, ml=False, fn=None, fs=7.5):
            c.setFillColor(self.ACCENT)
            c.rect(x, y - lbl_h, w, lbl_h, fill=1, stroke=0)
            c.setFillColor(self.PRIMARY)
            c.setFont("Helvetica-Bold", 6)
            c.drawString(x + 2, y - lbl_h + 3, label)
            c.setFillColor(self.BG_LIGHT)
            c.setStrokeColor(self.BORDER)
            c.rect(x, y - h, w, h - lbl_h, fill=1, stroke=1)
            c.setFillColor(colors.black)
            if ml:
                lines = str(val).split("\n")
                ly = y - lbl_h - fs * 1.2
                c.setFont("Helvetica", fs)
                for line in lines:
                    if ly < y - h + 2:
                        break
                    c.drawString(x + 3, ly, line)
                    ly -= fs * 1.35
            else:
                c.setFont("Helvetica", fs)
                text = str(val)
                while c.stringWidth(text, "Helvetica", fs) > w - 6 and len(text) > 3:
                    text = text[:-1]
                c.drawString(x + 3, y - lbl_h - (h - lbl_h)/2 - fs/3, text)
            if fn:
                self._annotate(fn, x, y - h, w, h, val)

        bbox(M, top, half, 2.6*cm, "SELLER / EXPORTER",
             d['seller_address'], ml=True, fn="seller_name")
        bbox(M + half, top, half, 2.6*cm, "BUYER / IMPORTER",
             d['buyer_address'], ml=True, fn="buyer_name")

        y = top - 2.6*cm - 0.15*cm
        q = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE NO.",        d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            bbox(M + i*q, y, q, 1.1*cm, lbl, val, fn=fn, fs=6.5)

        y -= 1.1*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE", d['bl_reference'], "bl_reference"),
            ("INCOTERM",      d['incoterm'],      None),
            ("PAYMENT TERMS", d['payment_terms'], None),
        ]):
            bbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn, fs=7.0)

        # Goods table
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [5.5*cm, 2.0*cm, 2.2*cm, 2.2*cm, 2.5*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION OF GOODS", "HS CODE", "QTY", "UNIT", "UNIT PRICE", "AMOUNT"]
        hdr_h, row_h2 = 0.5*cm, 1.6*cm

        c.setFillColor(self.PRIMARY)
        c.rect(M, y - hdr_h, total_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.setFont("Helvetica-Bold", 6.5)
        cx = M
        for hdr, cw in zip(hdrs, col_w):
            c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, y - hdr_h - row_h2, total_w, row_h2, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, y - hdr_h, cx, y - hdr_h - row_h2)

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        vals = [
            d['goods_description'], d['hs_code'], str(d['quantity']),
            d['package_unit'],
            f"{d['currency']} {d['unit_price']:,.2f}",
            f"{d['currency']} {d['total_amount']:,.2f}",
        ]
        fns = ["goods_description", None, "quantity", None, "unit_price", "total_amount"]
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, y - hdr_h - row_h2/2 - 3, text)
            if fn:
                self._annotate(fn, cx, y - hdr_h - row_h2, cw, row_h2, val)
            cx += cw

        # Totals: amber highlight
        y -= hdr_h + row_h2 + 0.2*cm
        tx = M + total_w * 0.6
        tw = total_w * 0.4
        c.setFillColor(self.PRIMARY)
        c.rect(tx, y - 0.8*cm, tw, 0.8*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(tx + 3, y - 0.55*cm, "TOTAL AMOUNT:")
        c.drawRightString(tx + tw - 3, y - 0.55*cm,
                          f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("total_amount", tx, y - 0.8*cm, tw, 0.8*cm,
                       f"{d['currency']} {d['total_amount']:,.2f}")
        self._annotate("currency", tx, y - 0.8*cm, tw/3, 0.8*cm, d['currency'])
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tx, y - 1.5*cm, tw, 0.7*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(tx + 3, y - 1.2*cm, "GROSS WEIGHT:")
        c.drawRightString(tx + tw - 3, y - 1.2*cm, f"{d['gross_weight_kg']} KG")

        # Signature
        y -= 1.5*cm + 0.5*cm
        c.setStrokeColor(self.ACCENT)
        c.setLineWidth(2)
        c.line(M, y, M + 5*cm, y)
        c.setLineWidth(1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['seller_name'])

        # Footer
        c.setFillColor(self.ACCENT)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"COMMERCIAL INVOICE  |  {d['invoice_number']}  |  B/L: {d['bl_reference']}")


# Style assignment: rotate across 5 styles for visual diversity
RENDERERS = [
    StandardInvoiceRenderer,
    NavyInvoiceRenderer,
    TradingInvoiceRenderer,
    MinimalInvoiceRenderer,
    BoldInvoiceRenderer,
]


# ─────────────────────────────────────────────
#  Main generation function
# ─────────────────────────────────────────────

def generate_invoices(input_dir: str, output_dir: str, disc_rate: float, seed: int):
    rng = random.Random(seed)

    dataset_path = Path(input_dir) / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    bl_records = dataset["documents"]

    inv_dir = Path(output_dir) / "invoices"
    ann_dir = inv_dir / "annotations"
    inv_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    lc_records = []
    n_inconsistent = 0

    print(f"\n{'='*58}")
    print(f"  Commercial Invoice Generator")
    print(f"  Input  : {dataset_path}")
    print(f"  Output : {inv_dir}")
    print(f"  Count  : {len(bl_records)}")
    print(f"  Discrepancy rate : {disc_rate:.0%}")
    print(f"{'='*58}\n")

    for i, bl_rec in enumerate(bl_records):
        inv_data = build_invoice_data(bl_rec, rng, disc_rate)

        # File name mirrors BL naming convention
        bl_filename = bl_rec["filename"]           # e.g. bl_msc_0001_MSCUNWWJL813676065.pdf
        inv_filename = bl_filename.replace("bl_", "inv_", 1)
        pdf_path = str(inv_dir / inv_filename)

        renderer_cls = RENDERERS[i % len(RENDERERS)]
        renderer = renderer_cls(pdf_path, inv_data)
        renderer.render()
        annotations = renderer.save()

        ann_record = {
            "invoice_number": inv_data["invoice_number"],
            "filename": inv_filename,
            "style": renderer_cls.__name__,
            "bl_id": bl_rec["id"],
            "fields": {
                "invoice_number": inv_data["invoice_number"],
                "invoice_date": inv_data["invoice_date"],
                "lc_reference": inv_data["lc_reference"],
                "bl_reference": inv_data["bl_reference"],
                "seller_name": inv_data["seller_name"],
                "buyer_name": inv_data["buyer_name"],
                "vessel": inv_data["vessel"],
                "voyage": inv_data["voyage"],
                "port_of_loading": inv_data["port_of_loading"],
                "port_of_discharge": inv_data["port_of_discharge"],
                "goods_description": inv_data["goods_description"],
                "hs_code": inv_data["hs_code"],
                "quantity": inv_data["quantity"],
                "package_unit": inv_data["package_unit"],
                "gross_weight_kg": inv_data["gross_weight_kg"],
                "unit_price": inv_data["unit_price"],
                "currency": inv_data["currency"],
                "total_amount": inv_data["total_amount"],
                "incoterm": inv_data["incoterm"],
                "payment_terms": inv_data["payment_terms"],
            },
            "bbox_annotations": annotations,
            "is_consistent": inv_data["is_consistent"],
            "discrepancies": inv_data["discrepancies"],
        }

        ann_path = ann_dir / inv_filename.replace(".pdf", ".json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann_record, f, ensure_ascii=False, indent=2)

        if not inv_data["is_consistent"]:
            n_inconsistent += 1

        lc_records.append({
            "lc_id": bl_rec["id"],
            "bl_filename": bl_filename,
            "invoice_filename": inv_filename,
            "lc_reference": inv_data["lc_reference"],
            "is_consistent": inv_data["is_consistent"],
            "discrepancies": inv_data["discrepancies"],
        })

        if (i + 1) % 50 == 0 or i == len(bl_records) - 1:
            print(f"  [{i+1:4d}/{len(bl_records)}] {inv_filename}")

    # Write master LC dataset
    lc_dataset = {
        "metadata": {
            "total_dossiers": len(lc_records),
            "consistent": len(lc_records) - n_inconsistent,
            "inconsistent": n_inconsistent,
            "actual_discrepancy_rate": round(n_inconsistent / len(lc_records), 4),
            "discrepancy_types": ["amount_mismatch", "description_mismatch",
                                   "port_mismatch", "currency_mismatch"],
            "generation_date": datetime.now().isoformat(),
            "bbox_format": "[x, y, width, height]",
            "coordinate_system": "PDF points, origin bottom-left",
        },
        "dossiers": lc_records,
    }

    lc_path = Path(output_dir) / "lc_dataset.json"
    with open(lc_path, "w", encoding="utf-8") as f:
        json.dump(lc_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*58}")
    print(f"  [OK] {len(bl_records)} invoices generated")
    print(f"  [OK] Consistent     : {len(lc_records) - n_inconsistent}")
    print(f"  [OK] Inconsistent   : {n_inconsistent} "
          f"({n_inconsistent/len(lc_records):.1%})")
    print(f"  [OK] LC dataset     : {lc_path}")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────
#  Parametric generation (21 layouts × N docs)
# ─────────────────────────────────────────────

def generate_dataset_parametric(count_per_layout: int, input_dir: str, output_dir: str,
                                 disc_rate: float = 0.27, seed: int = 0):
    """
    Generates count_per_layout invoices for each of the 21 INV layouts.
    Total: count_per_layout × 21 documents.
    Derives invoice data from the BL records in input_dir/dataset.json.
    """
    from layout_engine import INV_LAYOUTS
    from layout_engine.renderer_inv import InvoiceParametricRenderer

    rng = random.Random(seed if seed else None)

    dataset_path = Path(input_dir) / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    bl_records = dataset["documents"]

    inv_dir = Path(output_dir) / "invoices"
    ann_dir = inv_dir / "annotations"
    inv_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    total = count_per_layout * len(INV_LAYOUTS)
    print(f"\n{'='*60}")
    print(f"  Invoice Parametric — {len(INV_LAYOUTS)} layouts × {count_per_layout} = {total}")
    print(f"{'='*60}\n")

    doc_idx = 0
    bl_iter = 0

    for cfg in INV_LAYOUTS:
        layout_count = 0
        while layout_count < count_per_layout:
            doc_idx += 1
            layout_count += 1

            bl_rec = bl_records[bl_iter % len(bl_records)]
            bl_iter += 1

            inv_data = build_invoice_data(bl_rec, rng, disc_rate)
            filename = f"inv_{cfg.layout_id.lower()}_{doc_idx:05d}.pdf"
            pdf_path = str(inv_dir / filename)

            renderer = InvoiceParametricRenderer(pdf_path, inv_data, cfg)
            renderer.render()
            annotations = renderer.save()

            ann_record = {
                "invoice_number":   inv_data["invoice_number"],
                "filename":         filename,
                "layout_id":        cfg.layout_id,
                "bl_id":            bl_rec["id"],
                "fields": {
                    "invoice_number":    inv_data["invoice_number"],
                    "invoice_date":      inv_data["invoice_date"],
                    "lc_reference":      inv_data["lc_reference"],
                    "bl_reference":      inv_data["bl_reference"],
                    "seller_name":       inv_data["seller_name"],
                    "buyer_name":        inv_data["buyer_name"],
                    "vessel":            inv_data["vessel"],
                    "voyage":            inv_data["voyage"],
                    "port_of_loading":   inv_data["port_of_loading"],
                    "port_of_discharge": inv_data["port_of_discharge"],
                    "goods_description": inv_data["goods_description"],
                    "hs_code":           inv_data["hs_code"],
                    "quantity":          inv_data["quantity"],
                    "package_unit":      inv_data["package_unit"],
                    "gross_weight_kg":   inv_data["gross_weight_kg"],
                    "unit_price":        inv_data["unit_price"],
                    "currency":          inv_data["currency"],
                    "total_amount":      inv_data["total_amount"],
                    "incoterm":          inv_data["incoterm"],
                    "payment_terms":     inv_data["payment_terms"],
                },
                "bbox_annotations":  annotations,
                "is_consistent":     inv_data["is_consistent"],
                "discrepancies":     inv_data["discrepancies"],
            }

            ann_path = ann_dir / filename.replace(".pdf", ".json")
            with open(ann_path, "w", encoding="utf-8") as fp:
                json.dump(ann_record, fp, ensure_ascii=False, indent=2)

        print(f"  [{cfg.layout_id}] {layout_count} docs générés")

    print(f"\n{'='*60}")
    print(f"  [OK] {doc_idx} invoices → {inv_dir}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Commercial Invoice dataset")
    parser.add_argument("--input",  type=str, default="./bl_dataset",
                        help="Directory containing dataset.json")
    parser.add_argument("--output", type=str, default="./bl_dataset",
                        help="Output directory (invoices/ subfolder created here)")
    parser.add_argument("--discrepancy-rate", type=float, default=0.27,
                        help="Fraction of invoices with injected discrepancies (default 0.27)")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--parametric",       action="store_true",
                        help="Utiliser les 21 layouts paramétriques (recommandé)")
    parser.add_argument("--count-per-layout", type=int,   default=50,
                        help="Docs par layout (mode --parametric)")
    args = parser.parse_args()

    if args.parametric:
        generate_dataset_parametric(args.count_per_layout, args.input, args.output,
                                    args.discrepancy_rate, args.seed)
    else:
        generate_invoices(args.input, args.output, args.discrepancy_rate, args.seed)
