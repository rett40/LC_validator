"""
renderer_bl.py — Renderer BL générique paramétrique.
Prend un LayoutConfig + un dict de données BL et produit un PDF + annotations.
"""

from __future__ import annotations

import random
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

from .config import LayoutConfig
from .color_schemes import COLOR_SCHEMES


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires couleur
# ─────────────────────────────────────────────────────────────────────────────

def _c(hex_str: str):
    return colors.HexColor(hex_str)


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────

class BLParametricRenderer:
    PAGE_W, PAGE_H = A4

    def __init__(self, output_path: str, data: dict, cfg: LayoutConfig):
        self.path = output_path
        self.data = data
        self.cfg  = cfg
        self.c    = canvas.Canvas(output_path, pagesize=A4)
        self.annotations: list[dict] = []

        scheme = COLOR_SCHEMES[cfg.color_scheme]
        self.COL_PRIMARY   = _c(scheme["primary"])
        self.COL_SECONDARY = _c(scheme["secondary"])
        self.COL_ACCENT    = _c(scheme["accent"])
        self.COL_LIGHT_BG  = _c(scheme["light_bg"])
        self.COL_HDR_TXT   = _c(scheme["header_txt"])

        self.FONT      = cfg.font_body
        self.FONT_BOLD = cfg.font_body.split("-")[0] + "-Bold"
        self.FS        = cfg.font_size_body
        self.FS_LBL    = cfg.font_size_label

        self.ML = cfg.margin_left  * cm
        self.MR = cfg.margin_right * cm
        self.MT = cfg.margin_top   * cm
        self.CW = self.PAGE_W - self.ML - self.MR   # content width

    # ── Annotations ──────────────────────────────────────────────────────────

    def _ann(self, field: str, x, y, w, h, value):
        self.annotations.append({
            "field": field,
            "bbox":  [round(x,2), round(self.PAGE_H - y - h, 2), round(w,2), round(h,2)],
            "value": value,
            "page":  1,
        })

    # ── Primitives texte ─────────────────────────────────────────────────────

    def _text(self, x, y, txt, font=None, size=None, color=None):
        self.c.setFont(font or self.FONT, size or self.FS)
        self.c.setFillColor(color or colors.black)
        self.c.drawString(x, y, str(txt))

    def _text_field(self, field, x, y, w, h, value, size=None, align="left", color=None):
        """Dessine un champ texte + annotation."""
        font = self.FONT
        sz   = size or self.FS
        self.c.setFont(font, sz)
        self.c.setFillColor(color or colors.black)
        txt = str(value).replace("\n", " ")
        max_w = max(w - 4, 1)
        while self.c.stringWidth(txt, font, sz) > max_w and len(txt) > 4:
            txt = txt[:-1]
        ty = y - h / 2 - sz / 3
        if align == "center":
            self.c.drawCentredString(x + w / 2, ty, txt)
        elif align == "right":
            self.c.drawRightString(x + w - 2, ty, txt)
        else:
            self.c.drawString(x + 2, ty, txt)
        self._ann(field, x, y - h, w, h, value)

    def _multiline_field(self, field, x, y, w, h, value, size=None):
        sz = size or self.FS
        self.c.setFont(self.FONT, sz)
        self.c.setFillColor(colors.black)
        lines = str(value).split("\n")
        lh = sz * 1.4
        cy = y - lh
        for line in lines:
            if cy < y - h + sz:
                break
            while self.c.stringWidth(line, self.FONT, sz) > w - 4 and len(line) > 4:
                line = line[:-1]
            self.c.drawString(x + 2, cy, line)
            cy -= lh
        self._ann(field, x, y - h, w, h, value)

    def _label(self, x, y, txt):
        self.c.setFont(self.FONT_BOLD, self.FS_LBL)
        self.c.setFillColor(_c("#555555"))
        self.c.drawString(x, y, txt)

    def _hrule(self, y, color=None, width=0.5):
        self.c.setStrokeColor(color or _c("#CCCCCC"))
        self.c.setLineWidth(width)
        self.c.line(self.ML, y, self.ML + self.CW, y)

    def _box(self, x, y, w, h, fill=None, stroke=None, lw=0.5):
        self.c.setLineWidth(lw)
        if fill:
            self.c.setFillColor(fill)
        if stroke:
            self.c.setStrokeColor(stroke)
        self.c.rect(x, y, w, h,
                    fill=1 if fill else 0,
                    stroke=1 if stroke else 0)

    # ── Header styles ─────────────────────────────────────────────────────────

    def _draw_header(self) -> float:
        """Dessine le header selon cfg.header_style. Retourne y après le header."""
        style = self.cfg.header_style
        if style == "banner":
            return self._header_banner()
        elif style == "split":
            return self._header_split()
        elif style == "centered":
            return self._header_centered()
        elif style == "sidebar":
            return self._header_sidebar()
        else:  # minimal
            return self._header_minimal()

    def _header_banner(self) -> float:
        H = 2.8 * cm
        y_top = self.PAGE_H - self.MT
        self._box(self.ML, y_top - H, self.CW, H, fill=self.COL_PRIMARY)
        # BL title
        self.c.setFont(self.FONT_BOLD, 16)
        self.c.setFillColor(self.COL_HDR_TXT)
        self.c.drawCentredString(self.PAGE_W / 2, y_top - H / 2 - 4, "BILL OF LADING")
        # BL number top-right
        bl_num = self.data.get("bl_number", "")
        self.c.setFont(self.FONT, 8)
        self.c.drawRightString(self.ML + self.CW - 4,
                               y_top - H + 6, f"B/L No: {bl_num}")
        self._ann("bl_number", self.ML + self.CW * 0.6, y_top - H, self.CW * 0.4, H * 0.4, bl_num)
        return y_top - H - 0.3 * cm

    def _header_split(self) -> float:
        H = 2.5 * cm
        y_top = self.PAGE_H - self.MT
        # Left: color block with title
        w_left = self.CW * 0.55
        self._box(self.ML, y_top - H, w_left, H, fill=self.COL_PRIMARY)
        self.c.setFont(self.FONT_BOLD, 15)
        self.c.setFillColor(self.COL_HDR_TXT)
        self.c.drawString(self.ML + 6, y_top - H / 2 - 4, "BILL OF LADING")
        # Right: BL number block
        w_right = self.CW - w_left
        self._box(self.ML + w_left, y_top - H, w_right, H,
                  fill=self.COL_LIGHT_BG, stroke=self.COL_PRIMARY, lw=0.8)
        bl_num = self.data.get("bl_number", "")
        self.c.setFont(self.FONT_BOLD, 8)
        self.c.setFillColor(self.COL_PRIMARY)
        self.c.drawString(self.ML + w_left + 6, y_top - H * 0.35, "B/L NUMBER")
        self.c.setFont(self.FONT, 11)
        self.c.setFillColor(colors.black)
        self.c.drawString(self.ML + w_left + 6, y_top - H * 0.72, bl_num)
        self._ann("bl_number", self.ML + w_left, y_top - H, w_right, H, bl_num)
        return y_top - H - 0.3 * cm

    def _header_centered(self) -> float:
        H = 1.5 * cm
        y_top = self.PAGE_H - self.MT
        self.c.setFont(self.FONT_BOLD, 18)
        self.c.setFillColor(self.COL_PRIMARY)
        self.c.drawCentredString(self.PAGE_W / 2, y_top - H * 0.5, "BILL OF LADING")
        self._hrule(y_top - H - 0.1 * cm, color=self.COL_PRIMARY, width=1.2)
        # BL number inline
        bl_num = self.data.get("bl_number", "")
        self.c.setFont(self.FONT, 8)
        self.c.setFillColor(colors.black)
        self.c.drawRightString(self.ML + self.CW,
                               y_top - H - 0.35 * cm, f"B/L No.: {bl_num}")
        self._ann("bl_number", self.ML + self.CW * 0.5, y_top - H - 0.5 * cm,
                  self.CW * 0.5, 0.4 * cm, bl_num)
        return y_top - H - 0.7 * cm

    def _header_sidebar(self) -> float:
        H = 3.0 * cm
        y_top = self.PAGE_H - self.MT
        # Left sidebar
        sw = 1.5 * cm
        self._box(self.ML, y_top - H, sw, H, fill=self.COL_PRIMARY)
        # Title in main area
        self.c.setFont(self.FONT_BOLD, 15)
        self.c.setFillColor(self.COL_PRIMARY)
        self.c.drawString(self.ML + sw + 8, y_top - H * 0.45, "BILL OF LADING")
        # Accent line under title
        self._box(self.ML + sw + 8, y_top - H * 0.6,
                  self.CW - sw - 8, 0.15 * cm, fill=self.COL_ACCENT)
        # BL number
        bl_num = self.data.get("bl_number", "")
        self.c.setFont(self.FONT_BOLD, 9)
        self.c.setFillColor(_c("#333333"))
        self.c.drawString(self.ML + sw + 8, y_top - H * 0.8, f"B/L No.: {bl_num}")
        self._ann("bl_number", self.ML + sw + 8, y_top - H * 0.9,
                  self.CW * 0.5, H * 0.3, bl_num)
        return y_top - H - 0.3 * cm

    def _header_minimal(self) -> float:
        H = 1.2 * cm
        y_top = self.PAGE_H - self.MT
        self.c.setFont(self.FONT_BOLD, 14)
        self.c.setFillColor(colors.black)
        self.c.drawString(self.ML, y_top - H * 0.6, "BILL OF LADING")
        bl_num = self.data.get("bl_number", "")
        self.c.setFont(self.FONT, 9)
        self.c.drawRightString(self.ML + self.CW, y_top - H * 0.6, f"B/L No.: {bl_num}")
        self._ann("bl_number", self.ML + self.CW * 0.5, y_top - H,
                  self.CW * 0.5, H, bl_num)
        self._hrule(y_top - H - 0.1 * cm)
        return y_top - H - 0.3 * cm

    # ── Section : Parties (Shipper / Consignee / Notify) ─────────────────────

    def _draw_section_parties(self, y: float) -> float:
        d    = self.data
        cols = self.cfg.info_cols
        col_w = self.CW / cols

        def _party(key):
            """Accepte dict {name, address} ou clés plates key_name / key_address."""
            obj = d.get(key, {})
            if isinstance(obj, dict) and obj:  # non-empty dict avec clés name/address
                return obj.get("name", ""), obj.get("address", "")
            return d.get(f"{key}_name", ""), d.get(f"{key}_address", "")

        sn, sa   = _party("shipper")
        cn, ca   = _party("consignee")
        npn, npa = _party("notify_party")
        fields = [
            ("shipper",      "SHIPPER",      sn,  sa ),
            ("consignee",    "CONSIGNEE",    cn,  ca ),
            ("notify_party", "NOTIFY PARTY", npn, npa),
        ]

        cell_h = 3.2 * cm
        border = _c("#CCCCCC")

        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, width=0.8)
            self.c.setFont(self.FONT_BOLD, 7)
            self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y + 2, "PARTIES")
            y -= 0.25 * cm

        for i, (fname, label, name, addr) in enumerate(fields):
            col_idx = i % cols
            if col_idx == 0 and i > 0:
                y -= cell_h

            x = self.ML + col_idx * col_w

            # Fond alterné selon border_style
            if self.cfg.border_style == "zebra" and i % 2 == 0:
                self._box(x, y - cell_h, col_w, cell_h,
                          fill=self.COL_LIGHT_BG, stroke=border, lw=0.4)
            elif self.cfg.border_style in ("full", "header_only"):
                self._box(x, y - cell_h, col_w, cell_h,
                          stroke=border, lw=0.4)

            lp = self.cfg.label_position
            if lp == "above":
                self._label(x + 2, y - 0.35 * cm, label)
                self._multiline_field(fname, x, y - 0.4 * cm,
                                      col_w, cell_h - 0.4 * cm,
                                      f"{name}\n{addr}")
            elif lp == "inline":
                self.c.setFont(self.FONT_BOLD, self.FS_LBL)
                self.c.setFillColor(self.COL_SECONDARY)
                self.c.drawString(x + 2, y - 0.4 * cm, f"{label}:")
                self._multiline_field(fname, x + 2, y - 0.6 * cm,
                                      col_w - 4, cell_h - 0.6 * cm,
                                      f"{name}\n{addr}")
            else:  # prefix
                txt = f"{label}: {name}"
                self._text(x + 2, y - 0.45 * cm, txt,
                           font=self.FONT_BOLD, size=self.FS_LBL + 0.5)
                self._multiline_field(fname, x, y - 0.7 * cm,
                                      col_w, cell_h - 0.7 * cm, addr)
                self._ann(fname, x, y - cell_h, col_w, cell_h, f"{name}\n{addr}")

        # Avancer y après la dernière ligne de colonnes
        n_rows = -(-len(fields) // cols)   # ceil division
        return y - n_rows * cell_h - 0.3 * cm

    # ── Section : Routing ────────────────────────────────────────────────────

    def _draw_section_routing(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, width=0.8)
            self.c.setFont(self.FONT_BOLD, 7)
            self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y + 2, "ROUTING")
            y -= 0.25 * cm

        cols  = min(self.cfg.info_cols, 4)
        col_w = self.CW / cols
        h_row = 1.2 * cm
        border = _c("#CCCCCC")

        routing_fields = [
            ("vessel",             "VESSEL",              d.get("vessel", "")),
            ("voyage",             "VOYAGE NO.",           d.get("voyage", "")),
            ("port_of_loading",    "PORT OF LOADING",     d.get("port_of_loading", "")),
            ("port_of_discharge",  "PORT OF DISCHARGE",   d.get("port_of_discharge", "")),
            ("place_of_receipt",   "PLACE OF RECEIPT",    d.get("place_of_receipt", "")),
            ("place_of_delivery",  "PLACE OF DELIVERY",   d.get("place_of_delivery", "")),
            ("etd",                "ETD",                  d.get("etd", "")),
            ("eta",                "ETA",                  d.get("eta", "")),
        ]

        for i, (fname, label, value) in enumerate(routing_fields):
            col_idx = i % cols
            row_idx = i // cols
            x = self.ML + col_idx * col_w
            ry = y - row_idx * h_row

            if self.cfg.border_style in ("full", "zebra"):
                bg = self.COL_LIGHT_BG if (row_idx % 2 == 0 and self.cfg.border_style == "zebra") else None
                self._box(x, ry - h_row, col_w, h_row,
                          fill=bg, stroke=border, lw=0.4)

            self._draw_labeled_cell(fname, label, value, x, ry, col_w, h_row)

        n_rows = -(-len(routing_fields) // cols)
        return y - n_rows * h_row - 0.3 * cm

    def _draw_labeled_cell(self, fname, label, value, x, y, w, h):
        lp = self.cfg.label_position
        if lp == "above":
            self._label(x + 2, y - 0.3 * cm, label)
            self._text_field(fname, x, y - 0.35 * cm, w, h - 0.35 * cm, value)
        elif lp == "inline":
            self.c.setFont(self.FONT_BOLD, self.FS_LBL)
            self.c.setFillColor(self.COL_SECONDARY)
            self.c.drawString(x + 2, y - h * 0.35, f"{label}:")
            self._text_field(fname, x + 2, y - h * 0.6, w - 4, h * 0.45, value)
        else:  # prefix
            full = f"{label}: {value}"
            self.c.setFont(self.FONT, self.FS)
            self.c.setFillColor(colors.black)
            txt = full
            while self.c.stringWidth(txt, self.FONT, self.FS) > w - 4 and len(txt) > 6:
                txt = txt[:-1]
            self.c.drawString(x + 2, y - h * 0.5, txt)
            self._ann(fname, x, y - h, w, h, value)

    # ── Section : Cargo (tableau containers) ─────────────────────────────────

    def _draw_section_cargo(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, width=0.8)
            self.c.setFont(self.FONT_BOLD, 7)
            self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y + 2, "CARGO DETAILS")
            y -= 0.25 * cm

        containers = d.get("containers", [])
        if not containers:
            containers = [{
                "container_number": d.get("container_number", ""),
                "seal_number":      d.get("seal_number", ""),
                "container_type":   d.get("container_type", ""),
                "commodity":        d.get("commodity", ""),
                "packages":         d.get("packages", ""),
                "gross_weight_kg":  d.get("gross_weight_kg", ""),
                "measurement":      d.get("measurement", ""),
            }]

        headers  = ["CONTAINER NO.", "SEAL NO.", "TYPE", "COMMODITY", "PKGS", "G.WEIGHT (KG)", "MEAS (CBM)"]
        col_pcts = [0.17, 0.10, 0.08, 0.25, 0.10, 0.15, 0.15]
        col_ws   = [self.CW * p for p in col_pcts]
        hdr_h    = 0.55 * cm
        row_h    = 1.3 * cm
        border   = _c("#CCCCCC")

        # Header row
        self._box(self.ML, y - hdr_h, self.CW, hdr_h,
                  fill=self.COL_PRIMARY, stroke=None)
        self.c.setFillColor(self.COL_HDR_TXT)
        self.c.setFont(self.FONT_BOLD, 6.5)
        cx = self.ML
        for hdr, cw in zip(headers, col_ws):
            self.c.drawString(cx + 2, y - hdr_h + 4, hdr)
            cx += cw

        row_top_ys = []
        current_y = y - hdr_h
        for idx, cont in enumerate(containers):
            bg = self.COL_LIGHT_BG if (idx % 2 == 0 and self.cfg.border_style == "zebra") else colors.white
            self._box(self.ML, current_y - row_h, self.CW, row_h,
                      fill=bg, stroke=border, lw=0.4)
            # Separateurs colonnes
            cx = self.ML
            for cw in col_ws[:-1]:
                cx += cw
                self.c.setStrokeColor(border)
                self.c.setLineWidth(0.3)
                self.c.line(cx, current_y, cx, current_y - row_h)

            row_top_ys.append(current_y)
            values = [
                cont.get("container_number", ""),
                cont.get("seal_number", ""),
                cont.get("container_type", ""),
                cont.get("commodity", ""),
                str(cont.get("packages", "")),
                str(cont.get("gross_weight_kg", "")),
                str(cont.get("measurement", "")),
            ]
            field_names = ["container_number","seal_number","container_type",
                           "commodity","packages","gross_weight_kg","measurement"]

            self.c.setFont(self.FONT, 7)
            self.c.setFillColor(colors.black)
            cx = self.ML
            for fname, val, cw in zip(field_names, values, col_ws):
                txt = str(val)
                while self.c.stringWidth(txt, self.FONT, 7) > cw - 4 and len(txt) > 3:
                    txt = txt[:-1]
                self.c.drawString(cx + 2, current_y - row_h / 2 - 3, txt)
                self._ann(fname, cx, current_y - row_h, cw, row_h, val)
                cx += cw

            current_y -= row_h

        return current_y - 0.3 * cm

    # ── Section : Terms ──────────────────────────────────────────────────────

    def _draw_section_terms(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, width=0.8)
            self.c.setFont(self.FONT_BOLD, 7)
            self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y + 2, "TERMS & CONDITIONS")
            y -= 0.25 * cm

        cols  = min(self.cfg.info_cols, 4)
        col_w = self.CW / cols
        h_row = 1.0 * cm
        border = _c("#CCCCCC")

        term_fields = [
            ("freight_terms",       "FREIGHT TERMS",       d.get("freight_terms", "")),
            ("service_type",        "SERVICE TYPE",        d.get("service_type", "")),
            ("issue_date",          "DATE OF ISSUE",       d.get("issue_date", "")),
            ("issue_place",         "PLACE OF ISSUE",      d.get("issue_place", "")),
            ("number_of_originals", "NO. OF ORIGINALS",   str(d.get("number_of_originals", ""))),
        ]

        for i, (fname, label, value) in enumerate(term_fields):
            col_idx = i % cols
            row_idx = i // cols
            x  = self.ML + col_idx * col_w
            ry = y - row_idx * h_row

            if self.cfg.border_style in ("full", "header_only"):
                self._box(x, ry - h_row, col_w, h_row, stroke=border, lw=0.4)

            self._draw_labeled_cell(fname, label, value, x, ry, col_w, h_row)

        n_rows = -(-len(term_fields) // cols)
        return y - n_rows * h_row - 0.2 * cm

    # ── Render principal ─────────────────────────────────────────────────────

    def render(self):
        y = self._draw_header()

        section_map = {
            "parties": self._draw_section_parties,
            "routing": self._draw_section_routing,
            "cargo":   self._draw_section_cargo,
            "terms":   self._draw_section_terms,
        }

        for section in self.cfg.section_order:
            if y < 3.0 * cm:
                break
            fn = section_map.get(section)
            if fn:
                y = fn(y)

    def save(self) -> list[dict]:
        self.c.save()
        return self.annotations
