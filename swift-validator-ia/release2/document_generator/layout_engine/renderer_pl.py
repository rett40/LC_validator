"""
renderer_pl.py — Renderer Packing List générique paramétrique.
"""
from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from .config import LayoutConfig
from .color_schemes import COLOR_SCHEMES


def _c(h): return colors.HexColor(h)


class PLParametricRenderer:
    PAGE_W, PAGE_H = A4

    def __init__(self, output_path: str, data: dict, cfg: LayoutConfig):
        self.path = output_path
        self.data = data
        self.cfg  = cfg
        self.c    = canvas.Canvas(output_path, pagesize=A4)
        self.annotations: list[dict] = []

        s = COLOR_SCHEMES[cfg.color_scheme]
        self.COL_PRIMARY   = _c(s["primary"])
        self.COL_SECONDARY = _c(s["secondary"])
        self.COL_ACCENT    = _c(s["accent"])
        self.COL_LIGHT_BG  = _c(s["light_bg"])
        self.COL_HDR_TXT   = _c(s["header_txt"])

        self.FONT      = cfg.font_body
        self.FONT_BOLD = cfg.font_body.split("-")[0] + "-Bold"
        self.FS        = cfg.font_size_body
        self.FS_LBL    = cfg.font_size_label
        self.ML = cfg.margin_left  * cm
        self.MR = cfg.margin_right * cm
        self.MT = cfg.margin_top   * cm
        self.CW = self.PAGE_W - self.ML - self.MR

    # ── Primitives ────────────────────────────────────────────────────────────

    def _ann(self, field, x, y, w, h, value):
        self.annotations.append({
            "field": field,
            "bbox":  [round(x,2), round(self.PAGE_H-y-h,2), round(w,2), round(h,2)],
            "value": value, "page": 1,
        })

    def _text(self, x, y, txt, font=None, size=None, color=None):
        self.c.setFont(font or self.FONT, size or self.FS)
        self.c.setFillColor(color or colors.black)
        self.c.drawString(x, y, str(txt))

    def _field(self, fname, x, y, w, h, value, size=None, align="left"):
        font = self.FONT; sz = size or self.FS
        self.c.setFont(font, sz); self.c.setFillColor(colors.black)
        txt = str(value).replace("\n"," ")
        while self.c.stringWidth(txt, font, sz) > max(w-4,1) and len(txt) > 4:
            txt = txt[:-1]
        ty = y-h/2-sz/3
        if align == "center":  self.c.drawCentredString(x+w/2, ty, txt)
        elif align == "right": self.c.drawRightString(x+w-2, ty, txt)
        else:                  self.c.drawString(x+2, ty, txt)
        self._ann(fname, x, y-h, w, h, value)

    def _label(self, x, y, txt):
        self.c.setFont(self.FONT_BOLD, self.FS_LBL)
        self.c.setFillColor(_c("#555555"))
        self.c.drawString(x, y, txt)

    def _hrule(self, y, color=None, lw=0.5):
        self.c.setStrokeColor(color or _c("#CCCCCC"))
        self.c.setLineWidth(lw)
        self.c.line(self.ML, y, self.ML+self.CW, y)

    def _box(self, x, y, w, h, fill=None, stroke=None, lw=0.5):
        self.c.setLineWidth(lw)
        if fill:   self.c.setFillColor(fill)
        if stroke: self.c.setStrokeColor(stroke)
        self.c.rect(x, y, w, h, fill=1 if fill else 0, stroke=1 if stroke else 0)

    def _labeled_cell(self, fname, label, value, x, y, w, h):
        lp = self.cfg.label_position; bdr = _c("#CCCCCC")
        if self.cfg.border_style in ("full","header_only"):
            self._box(x, y-h, w, h, stroke=bdr, lw=0.4)
        if lp == "above":
            self._label(x+2, y-0.3*cm, label)
            self._field(fname, x, y-0.35*cm, w, h-0.35*cm, value)
        elif lp == "inline":
            self.c.setFont(self.FONT_BOLD, self.FS_LBL); self.c.setFillColor(self.COL_SECONDARY)
            self.c.drawString(x+2, y-h*0.35, f"{label}:")
            self._field(fname, x+2, y-h*0.6, w-4, h*0.45, value)
        else:
            full = f"{label}: {value}"
            self.c.setFont(self.FONT, self.FS); self.c.setFillColor(colors.black)
            while self.c.stringWidth(full, self.FONT, self.FS) > w-4 and len(full) > 6:
                full = full[:-1]
            self.c.drawString(x+2, y-h*0.5, full)
            self._ann(fname, x, y-h, w, h, value)

    # ── Headers ───────────────────────────────────────────────────────────────

    def _draw_header(self) -> float:
        s = self.cfg.header_style
        if s == "banner":   return self._hdr_banner()
        if s == "split":    return self._hdr_split()
        if s == "centered": return self._hdr_centered()
        if s == "sidebar":  return self._hdr_sidebar()
        return self._hdr_minimal()

    def _hdr_banner(self) -> float:
        H = 2.8*cm; yt = self.PAGE_H - self.MT
        self._box(self.ML, yt-H, self.CW, H, fill=self.COL_PRIMARY)
        self.c.setFont(self.FONT_BOLD, 16); self.c.setFillColor(self.COL_HDR_TXT)
        self.c.drawCentredString(self.PAGE_W/2, yt-H/2-4, "PACKING LIST")
        d = self.data
        self.c.setFont(self.FONT, 8)
        self.c.drawString(self.ML+6, yt-H+6, f"PL No: {d.get('pl_number','')}")
        self.c.drawRightString(self.ML+self.CW-4, yt-H+6, f"Date: {d.get('pl_date','')}")
        self._ann("pl_number", self.ML, yt-H, self.CW*0.4, H*0.35, d.get("pl_number",""))
        return yt - H - 0.3*cm

    def _hdr_split(self) -> float:
        H = 2.5*cm; yt = self.PAGE_H - self.MT
        wl = self.CW*0.55
        self._box(self.ML, yt-H, wl, H, fill=self.COL_PRIMARY)
        self.c.setFont(self.FONT_BOLD, 15); self.c.setFillColor(self.COL_HDR_TXT)
        self.c.drawString(self.ML+6, yt-H/2-3, "PACKING LIST")
        wr = self.CW-wl
        self._box(self.ML+wl, yt-H, wr, H, fill=self.COL_LIGHT_BG, stroke=self.COL_PRIMARY, lw=0.8)
        pl_no = self.data.get("pl_number","")
        self.c.setFont(self.FONT_BOLD, 7); self.c.setFillColor(self.COL_PRIMARY)
        self.c.drawString(self.ML+wl+6, yt-H*0.35, "PACKING LIST NO.")
        self.c.setFont(self.FONT, 10); self.c.setFillColor(colors.black)
        self.c.drawString(self.ML+wl+6, yt-H*0.72, pl_no)
        self._ann("pl_number", self.ML+wl, yt-H, wr, H, pl_no)
        return yt - H - 0.3*cm

    def _hdr_centered(self) -> float:
        H = 1.5*cm; yt = self.PAGE_H - self.MT
        self.c.setFont(self.FONT_BOLD, 18); self.c.setFillColor(self.COL_PRIMARY)
        self.c.drawCentredString(self.PAGE_W/2, yt-H*0.5, "PACKING LIST")
        self._hrule(yt-H-0.1*cm, color=self.COL_PRIMARY, lw=1.2)
        d = self.data
        self.c.setFont(self.FONT, 8); self.c.setFillColor(colors.black)
        self.c.drawString(self.ML, yt-H-0.4*cm, f"PL No.: {d.get('pl_number','')}")
        self.c.drawRightString(self.ML+self.CW, yt-H-0.4*cm, f"Date: {d.get('pl_date','')}")
        self._ann("pl_number", self.ML, yt-H-0.5*cm, self.CW*0.4, 0.4*cm, d.get("pl_number",""))
        return yt - H - 0.8*cm

    def _hdr_sidebar(self) -> float:
        H = 3.0*cm; yt = self.PAGE_H - self.MT; sw = 1.5*cm
        self._box(self.ML, yt-H, sw, H, fill=self.COL_PRIMARY)
        self.c.setFont(self.FONT_BOLD, 15); self.c.setFillColor(self.COL_PRIMARY)
        self.c.drawString(self.ML+sw+8, yt-H*0.45, "PACKING LIST")
        self._box(self.ML+sw+8, yt-H*0.6, self.CW-sw-8, 0.15*cm, fill=self.COL_ACCENT)
        pl_no = self.data.get("pl_number","")
        self.c.setFont(self.FONT_BOLD, 9); self.c.setFillColor(_c("#333333"))
        self.c.drawString(self.ML+sw+8, yt-H*0.8, f"PL No.: {pl_no}")
        self._ann("pl_number", self.ML+sw+8, yt-H*0.9, self.CW*0.5, H*0.3, pl_no)
        return yt - H - 0.3*cm

    def _hdr_minimal(self) -> float:
        H = 1.2*cm; yt = self.PAGE_H - self.MT
        self.c.setFont(self.FONT_BOLD, 14); self.c.setFillColor(colors.black)
        self.c.drawString(self.ML, yt-H*0.6, "PACKING LIST")
        pl_no = self.data.get("pl_number","")
        self.c.setFont(self.FONT, 9)
        self.c.drawRightString(self.ML+self.CW, yt-H*0.6, f"No.: {pl_no}")
        self._ann("pl_number", self.ML+self.CW*0.5, yt-H, self.CW*0.5, H, pl_no)
        self._hrule(yt-H-0.1*cm)
        return yt - H - 0.3*cm

    # ── Sections ──────────────────────────────────────────────────────────────

    def _draw_header_info(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, lw=0.8)
            self.c.setFont(self.FONT_BOLD, 7); self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y+2, "DOCUMENT REFERENCES")
            y -= 0.25*cm

        cols  = min(self.cfg.info_cols, 4); cw = self.CW/cols; h_row = 1.0*cm
        refs  = [
            ("pl_date",      "PL DATE",          d.get("pl_date","")),
            ("inv_reference","INVOICE REFERENCE", d.get("inv_reference","")),
            ("bl_reference", "B/L REFERENCE",     d.get("bl_reference","")),
            ("vessel",       "VESSEL",             d.get("vessel","")),
            ("voyage",       "VOYAGE",             d.get("voyage","")),
            ("port_of_loading","PORT OF LOADING",  d.get("port_of_loading","")),
        ]
        for i,(fn,lbl,val) in enumerate(refs):
            xi = self.ML+(i%cols)*cw; yi = y-(i//cols)*h_row
            self._labeled_cell(fn, lbl, val, xi, yi, cw, h_row)
        n_rows = -(-len(refs)//cols)
        return y - n_rows*h_row - 0.3*cm

    def _draw_parties(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, lw=0.8)
            self.c.setFont(self.FONT_BOLD, 7); self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y+2, "PARTIES")
            y -= 0.25*cm

        def _party(key):
            obj = d.get(key, {})
            if isinstance(obj, dict) and obj:  # non-empty dict avec clés name/address
                return obj.get("name",""), obj.get("address","")
            return d.get(f"{key}_name",""), d.get(f"{key}_address","")

        en, ea = _party("exporter"); cn, ca = _party("consignee")
        parties = [("exporter","EXPORTER / SHIPPER", en, ea),
                   ("consignee","CONSIGNEE",           cn, ca)]
        cols = min(self.cfg.info_cols, 2); cw = self.CW/cols
        cell_h = 2.8*cm; bdr = _c("#CCCCCC")

        for i,(fn,lbl,name,addr) in enumerate(parties):
            x = self.ML+(i%cols)*cw
            if self.cfg.border_style in ("full","zebra"):
                bg = self.COL_LIGHT_BG if (i%2==0 and self.cfg.border_style=="zebra") else None
                self._box(x, y-cell_h, cw, cell_h, fill=bg, stroke=bdr, lw=0.4)
            self._label(x+2, y-0.35*cm, lbl)
            self.c.setFont(self.FONT_BOLD, self.FS); self.c.setFillColor(colors.black)
            self.c.drawString(x+2, y-0.6*cm, str(name)[:40])
            for li, line in enumerate(str(addr).split("\n")[:3]):
                self.c.setFont(self.FONT, self.FS-0.5)
                self.c.drawString(x+2, y-0.9*cm-li*0.38*cm, line[:45])
            self._ann(fn, x, y-cell_h, cw, cell_h, f"{name}\n{addr}")

        n_rows = -(-len(parties)//cols)
        return y - n_rows*cell_h - 0.3*cm

    def _draw_items(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, lw=0.8)
            self.c.setFont(self.FONT_BOLD, 7); self.c.setFillColor(self.COL_PRIMARY)
            self.c.drawString(self.ML, y+2, "PACKING DETAILS")
            y -= 0.25*cm

        headers  = ["MARKS & NOS.", "DESCRIPTION", "PKG UNIT", "QTY", "NET WT (KG)", "GROSS WT (KG)", "DIMS (CM)", "CBM"]
        col_pcts = [0.12, 0.24, 0.08, 0.07, 0.12, 0.12, 0.13, 0.12]
        col_ws   = [self.CW*p for p in col_pcts]
        hdr_h    = 0.55*cm; row_h = 2.0*cm
        bdr = _c("#CCCCCC")

        self._box(self.ML, y-hdr_h, self.CW, hdr_h, fill=self.COL_PRIMARY)
        self.c.setFillColor(self.COL_HDR_TXT); self.c.setFont(self.FONT_BOLD, 6.0)
        cx = self.ML
        for hdr,cw in zip(headers, col_ws):
            self.c.drawString(cx+2, y-hdr_h+4, hdr); cx += cw

        self._box(self.ML, y-hdr_h-row_h, self.CW, row_h, fill=colors.white, stroke=bdr, lw=0.4)
        cx = self.ML
        for cw in col_ws[:-1]:
            cx += cw
            self.c.setStrokeColor(bdr); self.c.setLineWidth(0.3)
            self.c.line(cx, y-hdr_h, cx, y-hdr_h-row_h)

        dims = (f"{d.get('pkg_length_cm','')}×{d.get('pkg_width_cm','')}×"
                f"{d.get('pkg_height_cm','')} CM")
        values = [
            d.get("marks_numbers",""), d.get("goods_description",""),
            d.get("package_unit",""),  str(d.get("total_packages","")),
            str(d.get("net_weight_kg","")), str(d.get("gross_weight_kg","")),
            dims, str(d.get("total_cbm","")),
        ]
        fnames = ["marks_numbers","goods_description","package_unit","total_packages",
                  "net_weight_kg","gross_weight_kg","dimensions","total_cbm"]
        self.c.setFont(self.FONT, 7); self.c.setFillColor(colors.black)
        cx = self.ML
        for fn,val,cw in zip(fnames, values, col_ws):
            txt = str(val)
            while self.c.stringWidth(txt, self.FONT, 7) > cw-4 and len(txt) > 3:
                txt = txt[:-1]
            self.c.drawString(cx+2, y-hdr_h-row_h/2-3, txt)
            self._ann(fn, cx, y-hdr_h-row_h, cw, row_h, val)
            cx += cw

        return y - hdr_h - row_h - 0.3*cm

    def _draw_totals(self, y: float) -> float:
        d = self.data
        if self.cfg.use_dividers:
            self._hrule(y, color=self.COL_PRIMARY, lw=0.8)
            y -= 0.2*cm

        tw = self.CW*0.5; tx = self.ML+self.CW-tw; h = 0.7*cm
        bdr = _c("#CCCCCC")
        rows = [
            ("total_packages","TOTAL PACKAGES", str(d.get("total_packages",""))),
            ("net_weight_kg", "TOTAL NET WT",   f"{d.get('net_weight_kg','')} KG"),
            ("gross_weight_kg","TOTAL GROSS WT",f"{d.get('gross_weight_kg','')} KG"),
            ("total_cbm",     "TOTAL CBM",       str(d.get("total_cbm",""))),
        ]
        for i,(fn,lbl,val) in enumerate(rows):
            bg = self.COL_LIGHT_BG if i%2==0 else colors.white
            self._box(tx, y-h, tw, h, fill=bg, stroke=bdr, lw=0.4)
            self.c.setFont(self.FONT_BOLD, self.FS_LBL); self.c.setFillColor(self.COL_SECONDARY)
            self.c.drawString(tx+4, y-h*0.5, lbl)
            self.c.setFont(self.FONT, self.FS); self.c.setFillColor(colors.black)
            self.c.drawRightString(tx+tw-4, y-h*0.5, val)
            self._ann(fn, tx+tw*0.5, y-h, tw*0.5, h, val)
            y -= h
        return y - 0.3*cm

    def _draw_container_info(self, y: float) -> float:
        d = self.data
        cols = min(self.cfg.info_cols, 3); cw = self.CW/cols; h_row = 1.0*cm
        cfs  = [
            ("container_number","CONTAINER NO.",  d.get("container_number","")),
            ("container_type",  "CONTAINER TYPE", d.get("container_type","")),
            ("seal_number",     "SEAL NO.",        d.get("seal_number","")),
            ("port_of_discharge","PORT OF DISCHARGE", d.get("port_of_discharge","")),
        ]
        for i,(fn,lbl,val) in enumerate(cfs):
            xi = self.ML+(i%cols)*cw; yi = y-(i//cols)*h_row
            self._labeled_cell(fn, lbl, val, xi, yi, cw, h_row)
        n_rows = -(-len(cfs)//cols)
        return y - n_rows*h_row - 0.3*cm

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        y = self._draw_header()
        sec_map = {
            "header_info":    self._draw_header_info,
            "parties":        self._draw_parties,
            "items":          self._draw_items,
            "totals":         self._draw_totals,
            "container_info": self._draw_container_info,
        }
        for sec in self.cfg.section_order:
            if y < 3.0*cm: break
            fn = sec_map.get(sec)
            if fn: y = fn(y)

    def save(self) -> list[dict]:
        self.c.save(); return self.annotations
