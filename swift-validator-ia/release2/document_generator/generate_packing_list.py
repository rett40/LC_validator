"""
Packing List Synthetic Generator
=================================
Reads bl_dataset/dataset.json and generates one Packing List PDF per B/L,
deriving all fields from existing B/L data (same container, same goods,
same parties) so cross-document consistency with the B/L and Invoice is
guaranteed.

Discrepancies are injected in ~27% of dossiers. After generation, the
existing bl_dataset/lc_dataset.json is updated to add the packing list
reference to each dossier (3-document dossier: BL + Invoice + PL).

Outputs:
  bl_dataset/packing_lists/*.pdf
  bl_dataset/packing_lists/annotations/  — per-PL JSON with bbox
  bl_dataset/lc_dataset.json             — updated with packing_list_filename

Usage:
    python generate_packing_list.py
    python generate_packing_list.py --input ./bl_dataset --discrepancy-rate 0.27
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
#  Reference tables
# ─────────────────────────────────────────────

COMMODITY_CATEGORIES = {
    "ELECTRONIC COMPONENTS AND PARTS":    "electronics",
    "TEXTILE FABRICS AND GARMENTS":       "textile",
    "AUTOMOTIVE PARTS AND ACCESSORIES":   "automotive",
    "INDUSTRIAL MACHINERY AND EQUIPMENT": "machinery",
    "PLASTIC GOODS AND ARTICLES":         "plastics",
    "FURNITURE AND PARTS THEREOF":        "furniture",
    "FOOTWEAR OF ALL TYPES":              "footwear",
    "TOYS, GAMES AND SPORTS EQUIPMENT":   "toys",
    "HOUSEHOLD APPLIANCES":               "appliances",
    "STEEL PRODUCTS AND STRUCTURES":      "steel",
    "CHEMICAL PRODUCTS NEC":              "chemicals",
    "AGRICULTURAL PRODUCTS AND FOODSTUFFS":"food",
    "PAPER AND PAPERBOARD ARTICLES":      "paper",
    "RUBBER PRODUCTS AND TYRES":          "rubber",
    "OPTICAL AND PHOTOGRAPHIC INSTRUMENTS":"optical",
    "MEDICAL DEVICES AND EQUIPMENT":      "medical",
    "BUILDING MATERIALS":                 "construction",
    "PERSONAL CARE PRODUCTS":             "cosmetics",
    "CERAMIC TILES AND SANITARY WARE":    "ceramics",
    "COPPER AND COPPER PRODUCTS":         "metals",
    "WOOD PRODUCTS AND TIMBER":           "wood",
    "SEAFOOD AND FROZEN FISH PRODUCTS":   "food",
    "BEVERAGES AND SPIRITS":              "food",
    "SOLAR PANELS AND PHOTOVOLTAIC CELLS":"energy",
    "LITHIUM-ION BATTERIES":              "energy",
}

# Typical box/package dimensions (cm) per category (L x W x H)
PACKAGE_DIMS = {
    "electronics":  [(60, 40, 30), (50, 35, 25), (80, 50, 40)],
    "textile":      [(70, 50, 40), (60, 45, 35), (80, 60, 50)],
    "automotive":   [(80, 60, 50), (100, 70, 60), (90, 65, 55)],
    "machinery":    [(100, 80, 70), (120, 90, 80), (110, 85, 75)],
    "plastics":     [(60, 40, 30), (50, 35, 25), (70, 50, 40)],
    "furniture":    [(120, 80, 20), (100, 60, 15), (110, 70, 18)],
    "footwear":     [(60, 40, 30), (50, 35, 25), (65, 42, 32)],
    "toys":         [(50, 40, 35), (60, 50, 40), (55, 45, 38)],
    "appliances":   [(70, 55, 50), (80, 60, 55), (75, 58, 52)],
    "steel":        [(100, 80, 10), (120, 90, 12), (110, 85, 11)],
    "chemicals":    [(50, 40, 40), (55, 45, 45), (60, 50, 50)],
    "food":         [(40, 30, 25), (50, 35, 30), (45, 32, 28)],
    "paper":        [(80, 60, 40), (70, 55, 35), (75, 58, 38)],
    "rubber":       [(60, 50, 40), (65, 55, 45), (70, 60, 50)],
    "optical":      [(40, 30, 20), (35, 25, 18), (45, 32, 22)],
    "medical":      [(50, 40, 30), (55, 42, 32), (60, 45, 35)],
    "construction": [(80, 40, 30), (90, 45, 35), (85, 42, 32)],
    "cosmetics":    [(40, 30, 25), (45, 35, 28), (50, 38, 30)],
    "ceramics":     [(60, 60, 30), (70, 70, 35), (65, 65, 32)],
    "metals":       [(80, 30, 20), (90, 35, 25), (85, 32, 22)],
    "wood":         [(100, 50, 10), (120, 60, 12), (110, 55, 11)],
    "energy":       [(80, 60, 10), (100, 70, 12), (90, 65, 11)],
}

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


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _parse_packages(s: str) -> tuple:
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


def _parse_volume(s: str) -> float:
    """'27.843 CBM' → 27.843"""
    try:
        return float(s.split()[0])
    except (ValueError, IndexError):
        return 25.0


def _city_country(port_str: str) -> tuple:
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
#  Packing List data builder
# ─────────────────────────────────────────────

def build_pl_data(bl_record: dict, rng: random.Random, disc_rate: float) -> dict:
    f = bl_record["fields"]
    bl_id = bl_record["id"]

    qty_total, pkg_unit = _parse_packages(f["packages"])
    gross_kg = _parse_weight(f["gross_weight_kg"])
    total_cbm = _parse_volume(f.get("measurement", "25.0 CBM"))
    commodity = f["commodity"]
    category = COMMODITY_CATEGORIES.get(commodity, "machinery")

    # Net weight ~ gross - tare (~15-20%)
    net_kg = round(gross_kg * rng.uniform(0.80, 0.87), 2)

    # Package dimensions
    dims_options = PACKAGE_DIMS.get(category, [(60, 40, 30)])
    L, W, H = rng.choice(dims_options)
    cbm_per_pkg = round((L * W * H) / 1_000_000, 4)

    # Issue date
    try:
        bl_date = datetime.strptime(f["issue_date"], "%d %b %Y")
        pl_date = bl_date - timedelta(days=rng.randint(0, 2))
    except (ValueError, KeyError):
        pl_date = datetime.now()
    pl_date_str = pl_date.strftime("%d %b %Y").upper()

    exporter_address = _build_address(f["shipper_name"], f["port_of_loading"], rng)
    consignee_address = _build_address(f["consignee_name"], f["port_of_discharge"], rng)

    # PL number
    pl_number = f"PL-2026-{bl_id:05d}"
    inv_reference = f"INV-2026-{bl_id:05d}"

    is_consistent = True
    discrepancies = []
    # Working copies that may be altered
    qty_doc = qty_total
    gross_doc = gross_kg
    net_doc = net_kg
    commodity_doc = commodity

    if rng.random() < disc_rate:
        is_consistent = False
        possible = ["weight_mismatch", "quantity_mismatch", "description_mismatch",
                    "container_mismatch"]
        chosen = rng.sample(possible, rng.randint(1, 2))

        for disc_type in chosen:
            if disc_type == "weight_mismatch":
                factor = rng.uniform(1.05, 1.15) if rng.random() > 0.4 else rng.uniform(0.87, 0.95)
                original_gross = gross_doc
                gross_doc = round(gross_doc * factor, 2)
                net_doc = round(gross_doc * rng.uniform(0.80, 0.87), 2)
                discrepancies.append({
                    "type": "weight_mismatch",
                    "field": "gross_weight_kg",
                    "expected": f"{original_gross:,.2f}",
                    "actual": f"{gross_doc:,.2f}",
                })

            elif disc_type == "quantity_mismatch":
                delta = rng.randint(5, 30)
                sign = 1 if rng.random() > 0.5 else -1
                original_qty = qty_doc
                qty_doc = max(1, qty_doc + sign * delta)
                discrepancies.append({
                    "type": "quantity_mismatch",
                    "field": "total_packages",
                    "expected": str(original_qty),
                    "actual": str(qty_doc),
                })

            elif disc_type == "description_mismatch":
                alt = ALTERNATIVE_DESCRIPTIONS.get(commodity, commodity + " NEC")
                discrepancies.append({
                    "type": "description_mismatch",
                    "field": "goods_description",
                    "expected": commodity,
                    "actual": alt,
                })
                commodity_doc = alt

            elif disc_type == "container_mismatch":
                # Alter one digit in the container number
                cn = f["container_number"]
                pos = rng.randint(4, len(cn) - 1)
                wrong_digit = str((int(cn[pos]) + rng.randint(1, 8)) % 10) if cn[pos].isdigit() else cn[pos]
                alt_cn = cn[:pos] + wrong_digit + cn[pos+1:]
                discrepancies.append({
                    "type": "container_mismatch",
                    "field": "container_number",
                    "expected": cn,
                    "actual": alt_cn,
                })

    # Container number (possibly altered)
    container_number = f["container_number"]
    for d in discrepancies:
        if d["type"] == "container_mismatch":
            container_number = d["actual"]

    return {
        "pl_number": pl_number,
        "pl_date": pl_date_str,
        "inv_reference": inv_reference,
        "bl_reference": f["bl_number"],
        "exporter_name": f["shipper_name"],
        "exporter_address": exporter_address,
        "consignee_name": f["consignee_name"],
        "consignee_address": consignee_address,
        "vessel": f["vessel"],
        "voyage": f["voyage"],
        "port_of_loading": f["port_of_loading"],
        "port_of_discharge": f["port_of_discharge"],
        "container_number": container_number,
        "container_type": f["container_type"],
        "seal_number": f["seal_number"],
        "goods_description": commodity_doc,
        "package_unit": pkg_unit,
        "total_packages": qty_doc,
        "pkg_length_cm": L,
        "pkg_width_cm": W,
        "pkg_height_cm": H,
        "cbm_per_pkg": cbm_per_pkg,
        "total_cbm": round(total_cbm, 3),
        "net_weight_kg": f"{net_doc:,.2f}",
        "gross_weight_kg": f"{gross_doc:,.2f}",
        "marks_numbers": f"{container_number}\n{f['seal_number']}\n{_city_country(f['port_of_discharge'])[0]}",
        "is_consistent": is_consistent,
        "discrepancies": discrepancies,
    }


# ─────────────────────────────────────────────
#  Base renderer
# ─────────────────────────────────────────────

class PLRenderer:
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

    def _labeled_box(self, x, y, w, h, label, value, label_h=0.36*cm,
                     label_bg=None, box_bg=None, border=None,
                     font_size=7.5, multiline=False, field_name=None,
                     label_color=colors.white):
        c = self.c
        label_bg = label_bg or colors.HexColor("#1A3C5E")
        box_bg   = box_bg   or colors.HexColor("#F7F9FC")
        border   = border   or colors.HexColor("#CCCCCC")

        c.setFillColor(label_bg)
        c.rect(x, y - label_h, w, label_h, fill=1, stroke=0)
        c.setFillColor(label_color)
        c.setFont("Helvetica-Bold", 6)
        c.drawString(x + 2, y - label_h + 3, label)

        content_h = h - label_h
        c.setFillColor(box_bg)
        c.setStrokeColor(border)
        c.rect(x, y - h, w, content_h, fill=1, stroke=1)

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

    def _table_header(self, x, y, total_w, col_w, hdrs, bg, hdr_h=0.5*cm):
        c = self.c
        c.setFillColor(bg)
        c.rect(x, y - hdr_h, total_w, hdr_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 6.5)
        cx = x
        for hdr, cw in zip(hdrs, col_w):
            c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw
        return y - hdr_h

    def render(self):
        raise NotImplementedError

    def save(self):
        self.c.save()
        return self.annotations


# ─────────────────────────────────────────────
#  Style A — Standard Blue
# ─────────────────────────────────────────────

class StandardPLRenderer(PLRenderer):
    PRIMARY  = colors.HexColor("#1A3C5E")
    ACCENT   = colors.HexColor("#2E86C1")
    BG_LIGHT = colors.HexColor("#EBF5FB")
    BORDER   = colors.HexColor("#B0BEC5")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Header
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.6*cm, W, 2.6*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.6*cm, 0.6*cm, 2.6*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(W/2, H - 1.35*cm, "PACKING LIST")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#A9CCE3"))
        c.drawRightString(W - M, H - 2.15*cm, "ORIGINAL")

        # Meta band
        c.setFillColor(self.BG_LIGHT)
        c.rect(0, H - 3.2*cm, W, 0.6*cm, fill=1, stroke=0)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(M, H - 2.95*cm, f"PL NO: {d['pl_number']}")
        c.drawString(W/3, H - 2.95*cm, f"DATE: {d['pl_date']}")
        c.drawRightString(W - M, H - 2.95*cm, f"INV REF: {d['inv_reference']}")
        self._annotate("pl_number", M, H - 3.2*cm, 6*cm, 0.6*cm, d['pl_number'])
        self._annotate("inv_reference", W/2, H - 3.2*cm, W/2 - M, 0.6*cm, d['inv_reference'])

        # Exporter / Consignee
        top = H - 3.5*cm
        half = (W - 2*M) / 2
        self._labeled_box(M, top, half, 2.5*cm, "EXPORTER / SHIPPER",
                          d['exporter_address'], label_bg=self.PRIMARY,
                          box_bg=self.BG_LIGHT, border=self.BORDER,
                          font_size=7.5, multiline=True, field_name="exporter_name")
        self._labeled_box(M + half, top, half, 2.5*cm, "CONSIGNEE",
                          d['consignee_address'], label_bg=self.PRIMARY,
                          box_bg=colors.white, border=self.BORDER,
                          font_size=7.5, multiline=True, field_name="consignee_name")

        # Routing row
        y = top - 2.5*cm - 0.15*cm
        q4 = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE NO.",        d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            self._labeled_box(M + i*q4, y, q4, 1.0*cm, lbl, val,
                              label_bg=self.ACCENT, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=6.5, field_name=fn)

        # Container row
        y -= 1.0*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("CONTAINER NO.",  d['container_number'], "container_number"),
            ("CONTAINER TYPE", d['container_type'],   None),
            ("SEAL NO.",       d['seal_number'],       "seal_number"),
        ]):
            self._labeled_box(M + i*thirds, y, thirds, 1.0*cm, lbl, val,
                              label_bg=self.PRIMARY, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=7.0, field_name=fn)

        # Ref row
        y -= 1.0*cm + 0.15*cm
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE",   d['bl_reference'],   "bl_reference"),
            ("INVOICE REF",     d['inv_reference'],  None),
            ("MARKS & NUMBERS", d['marks_numbers'].replace('\n', ' / '), None),
        ]):
            self._labeled_box(M + i*thirds, y, thirds, 1.0*cm, lbl, val,
                              label_bg=self.PRIMARY, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=6.5, field_name=fn)

        # ── Packing details table ──
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [4.8*cm, 2.0*cm, 2.0*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION", "PKGS", "UNIT", "DIMENSIONS (CM)", "NET WT (KG)",
                 "GROSS WT (KG)", "CBM"]
        hdr_h, row_h = 0.5*cm, 1.7*cm

        bottom_y = self._table_header(M, y, total_w, col_w, hdrs,
                                      self.PRIMARY, hdr_h)

        # Data row
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, bottom_y - row_h, total_w, row_h, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, bottom_y, cx, bottom_y - row_h)

        dims_str = f"{d['pkg_length_cm']}x{d['pkg_width_cm']}x{d['pkg_height_cm']}"
        vals = [
            d['goods_description'],
            str(d['total_packages']),
            d['package_unit'],
            dims_str,
            d['net_weight_kg'],
            d['gross_weight_kg'],
            f"{d['total_cbm']:.3f}",
        ]
        fns = ["goods_description", "total_packages", None, None,
               "net_weight_kg", "gross_weight_kg", "total_cbm"]

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, bottom_y - row_h/2 - 3, text)
            if fn:
                self._annotate(fn, cx, bottom_y - row_h, cw, row_h, val)
            cx += cw

        # Totals block
        y = bottom_y - row_h - 0.2*cm
        tx = M + total_w * 0.55
        tw = total_w * 0.45
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tx, y - 2.2*cm, tw, 2.2*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        row_spacing = 0.55*cm

        def tot_row(label, val, yy, bold=False):
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 7.5 if bold else 7)
            c.setFillColor(self.PRIMARY if bold else colors.black)
            c.drawString(tx + 3, yy, label)
            c.drawRightString(tx + tw - 3, yy, val)

        tot_row("TOTAL PACKAGES:",   f"{d['total_packages']} {d['package_unit']}",
                y - 0.45*cm)
        tot_row("TOTAL NET WEIGHT:", f"{d['net_weight_kg']} KG",
                y - 0.45*cm - row_spacing)
        tot_row("TOTAL GROSS WEIGHT:", f"{d['gross_weight_kg']} KG",
                y - 0.45*cm - 2*row_spacing, bold=True)
        tot_row("TOTAL CBM:", f"{d['total_cbm']:.3f} M³",
                y - 0.45*cm - 3*row_spacing)
        self._annotate("gross_weight_kg", tx, y - 2.2*cm, tw, 2.2*cm, d['gross_weight_kg'])
        self._annotate("total_packages", tx, y - 2.2*cm, tw/2, 0.6*cm, str(d['total_packages']))

        # Signature
        y -= 2.2*cm + 0.5*cm
        c.setStrokeColor(self.PRIMARY)
        c.line(M, y, M + 5*cm, y)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATURE & STAMP")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['exporter_name'])

        # Footer
        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"PACKING LIST  |  {d['pl_number']}  |  B/L: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style B — Dark Slate
# ─────────────────────────────────────────────

class SlatePLRenderer(PLRenderer):
    PRIMARY  = colors.HexColor("#263238")
    ACCENT   = colors.HexColor("#FF6F00")
    BG_LIGHT = colors.HexColor("#FAFAFA")
    BORDER   = colors.HexColor("#B0BEC5")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Header: full dark with orange stripe
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.8*cm, W, 2.8*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.8*cm, W, 0.2*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 19)
        c.drawString(M, H - 1.6*cm, "PACKING LIST")
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#CFD8DC"))
        c.drawString(M, H - 2.1*cm, f"No. {d['pl_number']}   |   Date: {d['pl_date']}")
        c.drawString(M, H - 2.5*cm,
                     f"Inv: {d['inv_reference']}   |   B/L: {d['bl_reference']}")
        self._annotate("pl_number", M, H - 2.8*cm, 6*cm, 2.8*cm, d['pl_number'])
        self._annotate("bl_reference", M, H - 2.8*cm, W - 2*M, 0.5*cm, d['bl_reference'])

        top = H - 3.1*cm
        half = (W - 2*M) / 2
        lbl_h = 0.36*cm

        def box(x, y, w, h, label, val, ml=False, fn=None):
            self._labeled_box(x, y, w, h, label, val, label_h=lbl_h,
                              label_bg=self.PRIMARY, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=7.5,
                              multiline=ml, field_name=fn)

        box(M, top, half, 2.5*cm, "EXPORTER", d['exporter_address'], ml=True, fn="exporter_name")
        box(M+half, top, half, 2.5*cm, "CONSIGNEE", d['consignee_address'], ml=True, fn="consignee_name")

        y = top - 2.5*cm - 0.15*cm
        q = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE",            d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            box(M + i*q, y, q, 1.0*cm, lbl, val, fn=fn)

        y -= 1.0*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("CONTAINER NO.",  d['container_number'], "container_number"),
            ("CONTAINER TYPE", d['container_type'],   None),
            ("SEAL NO.",       d['seal_number'],       "seal_number"),
        ]):
            box(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        y -= 1.0*cm + 0.15*cm
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE", d['bl_reference'],  "bl_reference"),
            ("INVOICE REF",   d['inv_reference'], None),
            ("MARKS",         d['marks_numbers'].replace('\n', ' / '), None),
        ]):
            box(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        # Table
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [4.8*cm, 2.0*cm, 2.0*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION", "PKGS", "UNIT", "DIMENSIONS (CM)", "NET WT (KG)",
                 "GROSS WT (KG)", "CBM"]
        hdr_h, row_h = 0.5*cm, 1.7*cm

        bottom_y = self._table_header(M, y, total_w, col_w, hdrs, self.PRIMARY, hdr_h)

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, bottom_y - row_h, total_w, row_h, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, bottom_y, cx, bottom_y - row_h)

        dims_str = f"{d['pkg_length_cm']}x{d['pkg_width_cm']}x{d['pkg_height_cm']}"
        vals = [d['goods_description'], str(d['total_packages']), d['package_unit'],
                dims_str, d['net_weight_kg'], d['gross_weight_kg'], f"{d['total_cbm']:.3f}"]
        fns  = ["goods_description", "total_packages", None, None,
                "net_weight_kg", "gross_weight_kg", "total_cbm"]

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, bottom_y - row_h/2 - 3, text)
            if fn:
                self._annotate(fn, cx, bottom_y - row_h, cw, row_h, val)
            cx += cw

        # Totals
        y = bottom_y - row_h - 0.2*cm
        tx = M + total_w * 0.55
        tw = total_w * 0.45
        c.setFillColor(self.PRIMARY)
        c.rect(tx, y - 0.7*cm, tw, 0.7*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(tx + 3, y - 0.48*cm, "TOTAL GROSS WEIGHT:")
        c.drawRightString(tx + tw - 3, y - 0.48*cm, f"{d['gross_weight_kg']} KG")
        self._annotate("gross_weight_kg", tx, y - 0.7*cm, tw, 0.7*cm, d['gross_weight_kg'])

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tx, y - 1.7*cm, tw, 1.0*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(tx + 3, y - 1.0*cm, f"PKGS: {d['total_packages']} {d['package_unit']}")
        c.drawString(tx + 3, y - 1.5*cm, f"NET: {d['net_weight_kg']} KG  |  CBM: {d['total_cbm']:.3f}")
        self._annotate("total_packages", tx, y - 1.7*cm, tw, 0.5*cm, str(d['total_packages']))

        # Signature
        y -= 1.7*cm + 0.5*cm
        c.setStrokeColor(self.ACCENT)
        c.setLineWidth(1.5)
        c.line(M, y, M + 5*cm, y)
        c.setLineWidth(1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['exporter_name'])

        c.setFillColor(self.ACCENT)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"PACKING LIST  |  {d['pl_number']}  |  B/L: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style C — Teal / Corporate
# ─────────────────────────────────────────────

class TealPLRenderer(PLRenderer):
    PRIMARY  = colors.HexColor("#00695C")
    ACCENT   = colors.HexColor("#4DB6AC")
    BG_LIGHT = colors.HexColor("#E0F2F1")
    BORDER   = colors.HexColor("#80CBC4")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Teal top stripe
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 0.45*cm, W, 0.45*cm, fill=1, stroke=0)

        # Light background title area
        c.setFillColor(self.BG_LIGHT)
        c.rect(0, H - 2.8*cm, W, 2.35*cm, fill=1, stroke=0)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 19)
        c.drawString(M, H - 1.6*cm, "PACKING LIST")
        c.setFont("Helvetica", 8)
        c.drawString(M, H - 2.1*cm, f"PL No: {d['pl_number']}   |   Date: {d['pl_date']}")
        c.drawString(M, H - 2.5*cm, f"Invoice Ref: {d['inv_reference']}   |   B/L: {d['bl_reference']}")
        self._annotate("pl_number", M, H - 2.8*cm, 6*cm, 2.4*cm, d['pl_number'])
        self._annotate("bl_reference", M, H - 2.8*cm, W - 2*M, 0.5*cm, d['bl_reference'])

        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.8*cm, W, 0.12*cm, fill=1, stroke=0)

        top = H - 3.05*cm
        half = (W - 2*M) / 2
        lbl_h = 0.36*cm

        def tbox(x, y, w, h, label, val, ml=False, fn=None):
            self._labeled_box(x, y, w, h, label, val, label_h=lbl_h,
                              label_bg=self.PRIMARY, box_bg=self.BG_LIGHT,
                              border=self.BORDER, font_size=7.5,
                              multiline=ml, field_name=fn)

        tbox(M, top, half, 2.5*cm, "EXPORTER / SHIPPER",
             d['exporter_address'], ml=True, fn="exporter_name")
        tbox(M+half, top, half, 2.5*cm, "CONSIGNEE",
             d['consignee_address'], ml=True, fn="consignee_name")

        y = top - 2.5*cm - 0.15*cm
        q = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE NO.",        d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            tbox(M + i*q, y, q, 1.0*cm, lbl, val, fn=fn)

        y -= 1.0*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("CONTAINER NO.",  d['container_number'], "container_number"),
            ("CONTAINER TYPE", d['container_type'],   None),
            ("SEAL NO.",       d['seal_number'],       "seal_number"),
        ]):
            tbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        y -= 1.0*cm + 0.15*cm
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE", d['bl_reference'],  "bl_reference"),
            ("INVOICE REF",   d['inv_reference'], None),
            ("MARKS",         d['marks_numbers'].replace('\n', ' / '), None),
        ]):
            tbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        # Table
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [4.8*cm, 2.0*cm, 2.0*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION", "PKGS", "UNIT", "DIMENSIONS (CM)", "NET WT (KG)",
                 "GROSS WT (KG)", "CBM"]
        hdr_h, row_h = 0.5*cm, 1.7*cm

        bottom_y = self._table_header(M, y, total_w, col_w, hdrs, self.PRIMARY, hdr_h)

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, bottom_y - row_h, total_w, row_h, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, bottom_y, cx, bottom_y - row_h)

        dims_str = f"{d['pkg_length_cm']}x{d['pkg_width_cm']}x{d['pkg_height_cm']}"
        vals = [d['goods_description'], str(d['total_packages']), d['package_unit'],
                dims_str, d['net_weight_kg'], d['gross_weight_kg'], f"{d['total_cbm']:.3f}"]
        fns  = ["goods_description", "total_packages", None, None,
                "net_weight_kg", "gross_weight_kg", "total_cbm"]

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, bottom_y - row_h/2 - 3, text)
            if fn:
                self._annotate(fn, cx, bottom_y - row_h, cw, row_h, val)
            cx += cw

        # Totals
        y = bottom_y - row_h - 0.2*cm
        tx = M + total_w * 0.55
        tw = total_w * 0.45

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tx, y - 2.2*cm, tw, 2.2*cm, fill=1, stroke=1)
        row_sp = 0.55*cm

        def tot_row(label, val, yy, bold=False):
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 7.5 if bold else 7)
            c.setFillColor(self.PRIMARY if bold else colors.black)
            c.drawString(tx + 3, yy, label)
            c.drawRightString(tx + tw - 3, yy, val)

        tot_row("TOTAL PACKAGES:",     f"{d['total_packages']} {d['package_unit']}",
                y - 0.45*cm)
        tot_row("TOTAL NET WEIGHT:",   f"{d['net_weight_kg']} KG",
                y - 0.45*cm - row_sp)
        tot_row("TOTAL GROSS WEIGHT:", f"{d['gross_weight_kg']} KG",
                y - 0.45*cm - 2*row_sp, bold=True)
        tot_row("TOTAL CBM:",          f"{d['total_cbm']:.3f} M³",
                y - 0.45*cm - 3*row_sp)
        self._annotate("gross_weight_kg", tx, y - 2.2*cm, tw, 2.2*cm, d['gross_weight_kg'])
        self._annotate("total_packages",  tx, y - 2.2*cm, tw/2, 0.6*cm, str(d['total_packages']))

        # Signature
        y -= 2.2*cm + 0.5*cm
        c.setStrokeColor(self.PRIMARY)
        c.line(M, y, M + 5*cm, y)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY & STAMP")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['exporter_name'])

        c.setFillColor(self.PRIMARY)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"PACKING LIST  |  {d['pl_number']}  |  B/L: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style D — Minimal (dark slate, thin borders)
# ─────────────────────────────────────────────

class MinimalPLRenderer(PLRenderer):
    PRIMARY  = colors.HexColor("#37474F")
    ACCENT   = colors.HexColor("#78909C")
    BG_LIGHT = colors.HexColor("#FAFAFA")
    BORDER   = colors.HexColor("#CFD8DC")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Minimal: just a rule + title
        c.setStrokeColor(self.PRIMARY)
        c.setLineWidth(2)
        c.line(M, H - 0.8*cm, W - M, H - 0.8*cm)
        c.setLineWidth(1)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(M, H - 1.6*cm, "PACKING LIST")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(self.ACCENT)
        c.drawRightString(W - M, H - 1.2*cm, "ORIGINAL")

        # Meta band
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, H - 2.5*cm, W - 2*M, 0.8*cm, fill=1, stroke=1)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(M + 3, H - 2.2*cm, f"PL NO: {d['pl_number']}")
        c.drawString(W/3, H - 2.2*cm, f"DATE: {d['pl_date']}")
        c.drawRightString(W - M - 3, H - 2.2*cm, f"INV REF: {d['inv_reference']}")
        self._annotate("pl_number", M, H - 2.5*cm, 6*cm, 0.8*cm, d['pl_number'])
        self._annotate("inv_reference", W/2, H - 2.5*cm, W/2 - M, 0.8*cm, d['inv_reference'])

        top = H - 2.85*cm
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
                while c.stringWidth(text, "Helvetica", fs) > w - 6 and len(text) > 4:
                    text = text[:-1]
                c.drawString(x + 3, y - lbl_h - (h - lbl_h)/2 - fs/3, text)
            if fn:
                self._annotate(fn, x, y - h, w, h, val)

        mbox(M, top, half, 2.5*cm, "EXPORTER / SHIPPER",
             d['exporter_address'], ml=True, fn="exporter_name")
        mbox(M + half, top, half, 2.5*cm, "CONSIGNEE",
             d['consignee_address'], ml=True, fn="consignee_name")

        y = top - 2.5*cm - 0.15*cm
        q4 = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE NO.",        d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            mbox(M + i*q4, y, q4, 1.0*cm, lbl, val, fn=fn, fs=6.5)

        y -= 1.0*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("CONTAINER NO.",  d['container_number'], "container_number"),
            ("CONTAINER TYPE", d['container_type'],   None),
            ("SEAL NO.",       d['seal_number'],       "seal_number"),
        ]):
            mbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        y -= 1.0*cm + 0.15*cm
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE",   d['bl_reference'],  "bl_reference"),
            ("INVOICE REF",     d['inv_reference'], None),
            ("MARKS & NUMBERS", d['marks_numbers'].replace('\n', ' / '), None),
        ]):
            mbox(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn, fs=6.5)

        # Table
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [4.8*cm, 2.0*cm, 2.0*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION", "PKGS", "UNIT", "DIMENSIONS (CM)",
                 "NET WT (KG)", "GROSS WT (KG)", "CBM"]
        hdr_h, row_h = 0.45*cm, 1.7*cm

        bottom_y = self._table_header(M, y, total_w, col_w, hdrs, self.PRIMARY, hdr_h)
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, bottom_y - row_h, total_w, row_h, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, bottom_y, cx, bottom_y - row_h)

        dims_str = f"{d['pkg_length_cm']}x{d['pkg_width_cm']}x{d['pkg_height_cm']}"
        vals = [d['goods_description'], str(d['total_packages']), d['package_unit'],
                dims_str, d['net_weight_kg'], d['gross_weight_kg'], f"{d['total_cbm']:.3f}"]
        fns  = ["goods_description", "total_packages", None, None,
                "net_weight_kg", "gross_weight_kg", "total_cbm"]

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, bottom_y - row_h/2 - 3, text)
            if fn:
                self._annotate(fn, cx, bottom_y - row_h, cw, row_h, val)
            cx += cw

        # Totals
        y = bottom_y - row_h - 0.2*cm
        tx = M + total_w * 0.55
        tw = total_w * 0.45
        c.setStrokeColor(self.BORDER)
        c.setFillColor(self.BG_LIGHT)
        c.rect(tx, y - 2.2*cm, tw, 2.2*cm, fill=1, stroke=1)
        row_sp = 0.55*cm

        def tot_row(label, val, yy, bold=False):
            c.setFont("Helvetica-Bold" if bold else "Helvetica", 7.5 if bold else 7)
            c.setFillColor(self.PRIMARY if bold else colors.black)
            c.drawString(tx + 3, yy, label)
            c.drawRightString(tx + tw - 3, yy, val)

        tot_row("TOTAL PACKAGES:",     f"{d['total_packages']} {d['package_unit']}", y - 0.45*cm)
        tot_row("TOTAL NET WEIGHT:",   f"{d['net_weight_kg']} KG", y - 0.45*cm - row_sp)
        tot_row("TOTAL GROSS WEIGHT:", f"{d['gross_weight_kg']} KG", y - 0.45*cm - 2*row_sp, bold=True)
        tot_row("TOTAL CBM:",          f"{d['total_cbm']:.3f} M³",  y - 0.45*cm - 3*row_sp)
        self._annotate("gross_weight_kg", tx, y - 2.2*cm, tw, 2.2*cm, d['gross_weight_kg'])
        self._annotate("total_packages",  tx, y - 2.2*cm, tw/2, 0.6*cm, str(d['total_packages']))

        # Signature
        y -= 2.2*cm + 0.5*cm
        c.setStrokeColor(self.ACCENT)
        c.line(M, y, M + 5*cm, y)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['exporter_name'])

        c.setStrokeColor(self.PRIMARY)
        c.setLineWidth(1.5)
        c.line(M, 0.9*cm, W - M, 0.9*cm)
        c.setLineWidth(1)
        c.setFont("Helvetica", 6)
        c.setFillColor(self.ACCENT)
        c.drawCentredString(W/2, 0.55*cm,
            f"PACKING LIST  |  {d['pl_number']}  |  B/L: {d['bl_reference']}")


# ─────────────────────────────────────────────
#  Style E — Bold Red (high-contrast red/amber)
# ─────────────────────────────────────────────

class BoldPLRenderer(PLRenderer):
    PRIMARY  = colors.HexColor("#B71C1C")
    ACCENT   = colors.HexColor("#FF6F00")
    BG_LIGHT = colors.HexColor("#FFF8E1")
    BORDER   = colors.HexColor("#FFCC80")

    def render(self):
        c = self.c
        d = self.data
        W, H = self.PAGE_W, self.PAGE_H
        M = self.MARGIN

        # Bold dark-red header
        c.setFillColor(self.PRIMARY)
        c.rect(0, H - 2.8*cm, W, 2.8*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.rect(0, H - 2.8*cm, W, 0.28*cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(M, H - 1.5*cm, "PACKING LIST")
        c.setFont("Helvetica", 8)
        c.setFillColor(self.ACCENT)
        c.drawString(M, H - 2.0*cm, f"No. {d['pl_number']}   |   Date: {d['pl_date']}")
        c.drawString(M, H - 2.45*cm,
                     f"Inv: {d['inv_reference']}   |   B/L: {d['bl_reference']}")
        self._annotate("pl_number", M, H - 2.8*cm, 6*cm, 2.8*cm, d['pl_number'])
        self._annotate("bl_reference", M + 8*cm, H - 2.8*cm, 6*cm, 0.5*cm, d['bl_reference'])

        top = H - 3.1*cm
        half = (W - 2*M) / 2
        lbl_h = 0.38*cm

        def bx(x, y, w, h, label, val, ml=False, fn=None, fs=7.5):
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
                while c.stringWidth(text, "Helvetica", fs) > w - 6 and len(text) > 4:
                    text = text[:-1]
                c.drawString(x + 3, y - lbl_h - (h - lbl_h)/2 - fs/3, text)
            if fn:
                self._annotate(fn, x, y - h, w, h, val)

        bx(M, top, half, 2.5*cm, "EXPORTER", d['exporter_address'], ml=True, fn="exporter_name")
        bx(M + half, top, half, 2.5*cm, "CONSIGNEE", d['consignee_address'], ml=True, fn="consignee_name")

        y = top - 2.5*cm - 0.15*cm
        q4 = (W - 2*M) / 4
        for i, (lbl, val, fn) in enumerate([
            ("VESSEL",            d['vessel'],           "vessel"),
            ("VOYAGE",            d['voyage'],           "voyage"),
            ("PORT OF LOADING",   d['port_of_loading'],  "port_of_loading"),
            ("PORT OF DISCHARGE", d['port_of_discharge'],"port_of_discharge"),
        ]):
            bx(M + i*q4, y, q4, 1.0*cm, lbl, val, fn=fn, fs=6.5)

        y -= 1.0*cm + 0.15*cm
        thirds = (W - 2*M) / 3
        for i, (lbl, val, fn) in enumerate([
            ("CONTAINER NO.",  d['container_number'], "container_number"),
            ("CONTAINER TYPE", d['container_type'],   None),
            ("SEAL NO.",       d['seal_number'],       "seal_number"),
        ]):
            bx(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn)

        y -= 1.0*cm + 0.15*cm
        for i, (lbl, val, fn) in enumerate([
            ("B/L REFERENCE", d['bl_reference'],  "bl_reference"),
            ("INVOICE REF",   d['inv_reference'], None),
            ("MARKS",         d['marks_numbers'].replace('\n', ' / '), None),
        ]):
            bx(M + i*thirds, y, thirds, 1.0*cm, lbl, val, fn=fn, fs=6.5)

        # Table
        y -= 1.0*cm + 0.3*cm
        total_w = W - 2*M
        col_w = [4.8*cm, 2.0*cm, 2.0*cm, 2.8*cm, 2.4*cm, 2.4*cm, 2.6*cm]
        hdrs  = ["DESCRIPTION", "PKGS", "UNIT", "DIMENSIONS (CM)",
                 "NET WT (KG)", "GROSS WT (KG)", "CBM"]
        hdr_h, row_h = 0.5*cm, 1.7*cm

        bottom_y = self._table_header(M, y, total_w, col_w, hdrs, self.PRIMARY, hdr_h)
        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(M, bottom_y - row_h, total_w, row_h, fill=1, stroke=1)
        cx = M
        for cw in col_w[:-1]:
            cx += cw
            c.line(cx, bottom_y, cx, bottom_y - row_h)

        dims_str = f"{d['pkg_length_cm']}x{d['pkg_width_cm']}x{d['pkg_height_cm']}"
        vals = [d['goods_description'], str(d['total_packages']), d['package_unit'],
                dims_str, d['net_weight_kg'], d['gross_weight_kg'], f"{d['total_cbm']:.3f}"]
        fns  = ["goods_description", "total_packages", None, None,
                "net_weight_kg", "gross_weight_kg", "total_cbm"]

        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.black)
        cx = M
        for val, cw, fn in zip(vals, col_w, fns):
            text = str(val)
            while c.stringWidth(text, "Helvetica", 7.5) > cw - 4 and len(text) > 3:
                text = text[:-1]
            c.drawString(cx + 3, bottom_y - row_h/2 - 3, text)
            if fn:
                self._annotate(fn, cx, bottom_y - row_h, cw, row_h, val)
            cx += cw

        # Totals: amber band
        y = bottom_y - row_h - 0.2*cm
        tx = M + total_w * 0.55
        tw = total_w * 0.45
        c.setFillColor(self.PRIMARY)
        c.rect(tx, y - 0.7*cm, tw, 0.7*cm, fill=1, stroke=0)
        c.setFillColor(self.ACCENT)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(tx + 3, y - 0.48*cm, "TOTAL GROSS WEIGHT:")
        c.drawRightString(tx + tw - 3, y - 0.48*cm, f"{d['gross_weight_kg']} KG")
        self._annotate("gross_weight_kg", tx, y - 0.7*cm, tw, 0.7*cm, d['gross_weight_kg'])

        c.setFillColor(self.BG_LIGHT)
        c.setStrokeColor(self.BORDER)
        c.rect(tx, y - 1.7*cm, tw, 1.0*cm, fill=1, stroke=1)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(tx + 3, y - 1.0*cm, f"PKGS: {d['total_packages']} {d['package_unit']}")
        c.drawString(tx + 3, y - 1.5*cm, f"NET: {d['net_weight_kg']} KG  |  CBM: {d['total_cbm']:.3f}")
        self._annotate("total_packages", tx, y - 1.7*cm, tw, 0.5*cm, str(d['total_packages']))

        # Signature
        y -= 1.7*cm + 0.5*cm
        c.setStrokeColor(self.ACCENT)
        c.setLineWidth(2)
        c.line(M, y, M + 5*cm, y)
        c.setLineWidth(1)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(self.PRIMARY)
        c.drawString(M, y - 0.35*cm, "AUTHORIZED SIGNATORY")
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.black)
        c.drawString(M + 6*cm, y, d['exporter_name'])

        # Footer
        c.setFillColor(self.ACCENT)
        c.rect(0, 0, W, 0.75*cm, fill=1, stroke=0)
        c.setFillColor(self.PRIMARY)
        c.setFont("Helvetica-Bold", 6)
        c.drawCentredString(W/2, 0.28*cm,
            f"PACKING LIST  |  {d['pl_number']}  |  B/L: {d['bl_reference']}")


RENDERERS = [
    StandardPLRenderer,
    SlatePLRenderer,
    TealPLRenderer,
    MinimalPLRenderer,
    BoldPLRenderer,
]


# ─────────────────────────────────────────────
#  Main generation + lc_dataset update
# ─────────────────────────────────────────────

def generate_packing_lists(input_dir: str, output_dir: str, disc_rate: float, seed: int):
    rng = random.Random(seed + 1)  # different seed from invoice generator

    dataset_path = Path(input_dir) / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    bl_records = dataset["documents"]

    pl_dir  = Path(output_dir) / "packing_lists"
    ann_dir = pl_dir / "annotations"
    pl_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    lc_path = Path(output_dir) / "lc_dataset.json"
    lc_index = {}
    if lc_path.exists():
        with open(lc_path, encoding="utf-8") as f:
            lc_data = json.load(f)
        for dossier in lc_data["dossiers"]:
            lc_index[dossier["lc_id"]] = dossier
    else:
        lc_data = {"metadata": {}, "dossiers": []}

    print(f"\n{'='*58}")
    print(f"  Packing List Generator")
    print(f"  Input  : {dataset_path}")
    print(f"  Output : {pl_dir}")
    print(f"  Count  : {len(bl_records)}")
    print(f"  Discrepancy rate : {disc_rate:.0%}")
    print(f"{'='*58}\n")

    n_inconsistent = 0
    pl_records_ann = []

    for i, bl_rec in enumerate(bl_records):
        pl_data = build_pl_data(bl_rec, rng, disc_rate)

        bl_filename = bl_rec["filename"]
        pl_filename = bl_filename.replace("bl_", "pl_", 1)
        pdf_path = str(pl_dir / pl_filename)

        renderer_cls = RENDERERS[i % len(RENDERERS)]
        renderer = renderer_cls(pdf_path, pl_data)
        renderer.render()
        annotations = renderer.save()

        ann_record = {
            "pl_number": pl_data["pl_number"],
            "filename": pl_filename,
            "style": renderer_cls.__name__,
            "bl_id": bl_rec["id"],
            "fields": {
                "pl_number":        pl_data["pl_number"],
                "pl_date":          pl_data["pl_date"],
                "inv_reference":    pl_data["inv_reference"],
                "bl_reference":     pl_data["bl_reference"],
                "exporter_name":    pl_data["exporter_name"],
                "consignee_name":   pl_data["consignee_name"],
                "vessel":           pl_data["vessel"],
                "voyage":           pl_data["voyage"],
                "port_of_loading":  pl_data["port_of_loading"],
                "port_of_discharge":pl_data["port_of_discharge"],
                "container_number": pl_data["container_number"],
                "container_type":   pl_data["container_type"],
                "seal_number":      pl_data["seal_number"],
                "goods_description":pl_data["goods_description"],
                "package_unit":     pl_data["package_unit"],
                "total_packages":   pl_data["total_packages"],
                "net_weight_kg":    pl_data["net_weight_kg"],
                "gross_weight_kg":  pl_data["gross_weight_kg"],
                "total_cbm":        pl_data["total_cbm"],
            },
            "bbox_annotations": annotations,
            "is_consistent": pl_data["is_consistent"],
            "discrepancies": pl_data["discrepancies"],
        }

        ann_path = ann_dir / pl_filename.replace(".pdf", ".json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann_record, f, ensure_ascii=False, indent=2)

        if not pl_data["is_consistent"]:
            n_inconsistent += 1

        pl_records_ann.append(ann_record)

        # Update lc_index
        bl_id = bl_rec["id"]
        if bl_id in lc_index:
            lc_index[bl_id]["packing_list_filename"] = pl_filename
            # Merge PL discrepancies into the dossier
            for disc in pl_data["discrepancies"]:
                # Tag source so ConsistencyEngine knows which document has the issue
                disc_tagged = dict(disc, source="packing_list")
                existing_types = {d["type"] for d in lc_index[bl_id]["discrepancies"]}
                if disc["type"] not in existing_types:
                    lc_index[bl_id]["discrepancies"].append(disc_tagged)
            if not pl_data["is_consistent"]:
                lc_index[bl_id]["is_consistent"] = False

        if (i + 1) % 50 == 0 or i == len(bl_records) - 1:
            print(f"  [{i+1:4d}/{len(bl_records)}] {pl_filename}")

    # Re-write lc_dataset.json with PL data merged
    dossiers = list(lc_index.values())
    total_inconsistent = sum(1 for d in dossiers if not d["is_consistent"])
    lc_data["metadata"].update({
        "total_dossiers": len(dossiers),
        "consistent": len(dossiers) - total_inconsistent,
        "inconsistent": total_inconsistent,
        "actual_discrepancy_rate": round(total_inconsistent / len(dossiers), 4),
        "document_types": ["bl", "invoice", "packing_list"],
        "discrepancy_types_invoice": ["amount_mismatch", "description_mismatch",
                                       "port_mismatch", "currency_mismatch"],
        "discrepancy_types_pl": ["weight_mismatch", "quantity_mismatch",
                                  "description_mismatch", "container_mismatch"],
        "generation_date": datetime.now().isoformat(),
    })
    lc_data["dossiers"] = dossiers

    with open(lc_path, "w", encoding="utf-8") as f:
        json.dump(lc_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*58}")
    print(f"  [OK] {len(bl_records)} packing lists generated")
    print(f"  [OK] PL inconsistent : {n_inconsistent} ({n_inconsistent/len(bl_records):.1%})")
    print(f"  [OK] Dossier total inconsistent : {total_inconsistent} "
          f"({total_inconsistent/len(dossiers):.1%})")
    print(f"  [OK] lc_dataset.json updated : {lc_path}")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────
#  Parametric generation (21 layouts × N docs)
# ─────────────────────────────────────────────

def generate_dataset_parametric(count_per_layout: int, input_dir: str, output_dir: str,
                                 disc_rate: float = 0.27, seed: int = 0):
    """
    Generates count_per_layout packing lists for each of the 21 PL layouts.
    Total: count_per_layout × 21 documents.
    Derives PL data from the BL records in input_dir/dataset.json.
    """
    from layout_engine import PL_LAYOUTS
    from layout_engine.renderer_pl import PLParametricRenderer

    rng = random.Random(seed + 1 if seed else None)

    dataset_path = Path(input_dir) / "dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)
    bl_records = dataset["documents"]

    pl_dir  = Path(output_dir) / "packing_lists"
    ann_dir = pl_dir / "annotations"
    pl_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    total = count_per_layout * len(PL_LAYOUTS)
    print(f"\n{'='*60}")
    print(f"  Packing List Parametric — {len(PL_LAYOUTS)} layouts × {count_per_layout} = {total}")
    print(f"{'='*60}\n")

    doc_idx = 0
    bl_iter = 0

    for cfg in PL_LAYOUTS:
        layout_count = 0
        while layout_count < count_per_layout:
            doc_idx += 1
            layout_count += 1

            bl_rec = bl_records[bl_iter % len(bl_records)]
            bl_iter += 1

            pl_data = build_pl_data(bl_rec, rng, disc_rate)
            filename = f"pl_{cfg.layout_id.lower()}_{doc_idx:05d}.pdf"
            pdf_path = str(pl_dir / filename)

            renderer = PLParametricRenderer(pdf_path, pl_data, cfg)
            renderer.render()
            annotations = renderer.save()

            ann_record = {
                "pl_number":        pl_data["pl_number"],
                "filename":         filename,
                "layout_id":        cfg.layout_id,
                "bl_id":            bl_rec["id"],
                "fields": {
                    "pl_number":         pl_data["pl_number"],
                    "pl_date":           pl_data["pl_date"],
                    "inv_reference":     pl_data["inv_reference"],
                    "bl_reference":      pl_data["bl_reference"],
                    "exporter_name":     pl_data["exporter_name"],
                    "consignee_name":    pl_data["consignee_name"],
                    "vessel":            pl_data["vessel"],
                    "voyage":            pl_data["voyage"],
                    "port_of_loading":   pl_data["port_of_loading"],
                    "port_of_discharge": pl_data["port_of_discharge"],
                    "container_number":  pl_data["container_number"],
                    "container_type":    pl_data["container_type"],
                    "seal_number":       pl_data["seal_number"],
                    "goods_description": pl_data["goods_description"],
                    "package_unit":      pl_data["package_unit"],
                    "total_packages":    pl_data["total_packages"],
                    "net_weight_kg":     pl_data["net_weight_kg"],
                    "gross_weight_kg":   pl_data["gross_weight_kg"],
                    "total_cbm":         pl_data["total_cbm"],
                },
                "bbox_annotations":  annotations,
                "is_consistent":     pl_data["is_consistent"],
                "discrepancies":     pl_data["discrepancies"],
            }

            ann_path = ann_dir / filename.replace(".pdf", ".json")
            with open(ann_path, "w", encoding="utf-8") as fp:
                json.dump(ann_record, fp, ensure_ascii=False, indent=2)

        print(f"  [{cfg.layout_id}] {layout_count} docs générés")

    print(f"\n{'='*60}")
    print(f"  [OK] {doc_idx} packing lists → {pl_dir}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Packing List dataset")
    parser.add_argument("--input",  type=str, default="./bl_dataset")
    parser.add_argument("--output", type=str, default="./bl_dataset")
    parser.add_argument("--discrepancy-rate", type=float, default=0.27)
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
        generate_packing_lists(args.input, args.output, args.discrepancy_rate, args.seed)
