"""
LayoutConfig — configuration paramétrique d'un layout de document.
64 configs prédéfinies : 22 BL + 21 Invoice + 21 PL.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LayoutConfig:
    layout_id:      str
    doc_type:       str           # "bl" | "invoice" | "packing_list"

    # Palette (clé dans color_schemes.COLOR_SCHEMES)
    color_scheme:   str

    # Typo
    font_body:      str           # "Helvetica" | "Times-Roman" | "Courier"
    font_size_body: float         # 7.5 – 9.0
    font_size_label:float         # 5.5 – 7.5

    # Header
    header_style:   str           # "banner" | "split" | "centered" | "sidebar" | "minimal"

    # Structure page
    info_cols:      int           # 2 | 3 | 4  colonnes pour les champs parties/route
    section_order:  list[str]     # ordre des blocs sur la page

    # Style tableau
    border_style:   str           # "full" | "header_only" | "zebra" | "none"
    label_position: str           # "above" | "inline" | "prefix"
    use_dividers:   bool          # lignes horizontales entre sections

    # Marges (cm)
    margin_left:    float = 1.0
    margin_right:   float = 1.0
    margin_top:     float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 22 Layouts — Bill of Lading
# ─────────────────────────────────────────────────────────────────────────────

BL_LAYOUTS: list[LayoutConfig] = [
    LayoutConfig("BL-01", "bl", "navy",     "Helvetica",   8.0, 6.5, "banner",   3, ["parties","routing","cargo","terms"], "full",        "above",  True ),
    LayoutConfig("BL-02", "bl", "crimson",  "Helvetica",   8.0, 6.5, "split",    2, ["parties","routing","cargo","terms"], "header_only", "above",  False),
    LayoutConfig("BL-03", "bl", "forest",   "Times-Roman", 8.5, 7.0, "banner",   4, ["routing","parties","cargo","terms"], "zebra",       "inline", True ),
    LayoutConfig("BL-04", "bl", "orange",   "Helvetica",   8.0, 6.5, "split",    3, ["routing","parties","cargo","terms"], "full",        "prefix", False),
    LayoutConfig("BL-05", "bl", "magenta",  "Helvetica",   7.5, 6.0, "centered", 2, ["parties","cargo","routing","terms"], "header_only", "above",  True ),
    LayoutConfig("BL-06", "bl", "slate",    "Courier",     8.0, 6.5, "banner",   3, ["parties","routing","cargo","terms"], "none",        "prefix", False),
    LayoutConfig("BL-07", "bl", "amber",    "Helvetica",   8.5, 7.0, "sidebar",  4, ["routing","cargo","parties","terms"], "full",        "above",  True ),
    LayoutConfig("BL-08", "bl", "teal",     "Times-Roman", 8.0, 6.5, "minimal",  2, ["parties","routing","cargo","terms"], "zebra",       "inline", False),
    LayoutConfig("BL-09", "bl", "indigo",   "Helvetica",   7.5, 6.0, "split",    3, ["parties","routing","cargo","terms"], "full",        "above",  True ),
    LayoutConfig("BL-10", "bl", "graphite", "Helvetica",   8.0, 6.5, "banner",   2, ["cargo","parties","routing","terms"], "header_only", "prefix", False),
    LayoutConfig("BL-11", "bl", "burgundy", "Times-Roman", 8.5, 7.0, "centered", 3, ["routing","parties","cargo","terms"], "full",        "above",  True ),
    LayoutConfig("BL-12", "bl", "steel",    "Helvetica",   8.0, 6.5, "sidebar",  2, ["parties","cargo","routing","terms"], "zebra",       "inline", False),
    LayoutConfig("BL-13", "bl", "navy",     "Courier",     7.5, 6.0, "minimal",  4, ["parties","routing","cargo","terms"], "none",        "prefix", True ),
    LayoutConfig("BL-14", "bl", "crimson",  "Helvetica",   8.0, 6.5, "banner",   3, ["cargo","routing","parties","terms"], "full",        "above",  False),
    LayoutConfig("BL-15", "bl", "forest",   "Helvetica",   8.5, 7.0, "split",    2, ["parties","routing","cargo","terms"], "header_only", "above",  True ),
    LayoutConfig("BL-16", "bl", "orange",   "Times-Roman", 8.0, 6.5, "centered", 3, ["routing","cargo","parties","terms"], "zebra",       "inline", False),
    LayoutConfig("BL-17", "bl", "magenta",  "Helvetica",   7.5, 6.0, "banner",   4, ["parties","routing","cargo","terms"], "full",        "above",  True ),
    LayoutConfig("BL-18", "bl", "slate",    "Helvetica",   8.0, 6.5, "split",    2, ["cargo","parties","routing","terms"], "header_only", "prefix", False),
    LayoutConfig("BL-19", "bl", "amber",    "Times-Roman", 8.5, 7.0, "sidebar",  3, ["parties","routing","cargo","terms"], "full",        "above",  True ),
    LayoutConfig("BL-20", "bl", "teal",     "Courier",     8.0, 6.5, "minimal",  2, ["routing","parties","cargo","terms"], "none",        "inline", False),
    LayoutConfig("BL-21", "bl", "indigo",   "Helvetica",   7.5, 6.0, "banner",   3, ["parties","cargo","routing","terms"], "zebra",       "above",  True ),
    LayoutConfig("BL-22", "bl", "graphite", "Times-Roman", 8.0, 6.5, "split",    4, ["routing","parties","cargo","terms"], "full",        "prefix", False),
]

# ─────────────────────────────────────────────────────────────────────────────
# 21 Layouts — Commercial Invoice
# ─────────────────────────────────────────────────────────────────────────────

INV_LAYOUTS: list[LayoutConfig] = [
    LayoutConfig("INV-01", "invoice", "navy",     "Helvetica",   8.0, 6.5, "banner",   3, ["header_info","parties","goods","totals","terms"], "full",        "above",  True ),
    LayoutConfig("INV-02", "invoice", "crimson",  "Helvetica",   8.0, 6.5, "split",    2, ["header_info","parties","goods","totals","terms"], "header_only", "above",  False),
    LayoutConfig("INV-03", "invoice", "forest",   "Times-Roman", 8.5, 7.0, "banner",   4, ["parties","header_info","goods","totals","terms"], "zebra",       "inline", True ),
    LayoutConfig("INV-04", "invoice", "orange",   "Helvetica",   8.0, 6.5, "split",    3, ["header_info","goods","parties","totals","terms"], "full",        "prefix", False),
    LayoutConfig("INV-05", "invoice", "magenta",  "Helvetica",   7.5, 6.0, "centered", 2, ["header_info","parties","goods","totals","terms"], "header_only", "above",  True ),
    LayoutConfig("INV-06", "invoice", "slate",    "Courier",     8.0, 6.5, "banner",   3, ["header_info","parties","goods","totals","terms"], "none",        "prefix", False),
    LayoutConfig("INV-07", "invoice", "amber",    "Helvetica",   8.5, 7.0, "sidebar",  4, ["parties","header_info","goods","totals","terms"], "full",        "above",  True ),
    LayoutConfig("INV-08", "invoice", "teal",     "Times-Roman", 8.0, 6.5, "minimal",  2, ["header_info","parties","goods","totals","terms"], "zebra",       "inline", False),
    LayoutConfig("INV-09", "invoice", "indigo",   "Helvetica",   7.5, 6.0, "split",    3, ["header_info","parties","goods","totals","terms"], "full",        "above",  True ),
    LayoutConfig("INV-10", "invoice", "graphite", "Helvetica",   8.0, 6.5, "banner",   2, ["goods","header_info","parties","totals","terms"], "header_only", "prefix", False),
    LayoutConfig("INV-11", "invoice", "burgundy", "Times-Roman", 8.5, 7.0, "centered", 3, ["parties","header_info","goods","totals","terms"], "full",        "above",  True ),
    LayoutConfig("INV-12", "invoice", "steel",    "Helvetica",   8.0, 6.5, "sidebar",  2, ["header_info","goods","parties","totals","terms"], "zebra",       "inline", False),
    LayoutConfig("INV-13", "invoice", "navy",     "Courier",     7.5, 6.0, "minimal",  4, ["header_info","parties","goods","totals","terms"], "none",        "prefix", True ),
    LayoutConfig("INV-14", "invoice", "crimson",  "Helvetica",   8.0, 6.5, "banner",   3, ["goods","parties","header_info","totals","terms"], "full",        "above",  False),
    LayoutConfig("INV-15", "invoice", "forest",   "Helvetica",   8.5, 7.0, "split",    2, ["header_info","parties","goods","totals","terms"], "header_only", "above",  True ),
    LayoutConfig("INV-16", "invoice", "orange",   "Times-Roman", 8.0, 6.5, "centered", 3, ["parties","goods","header_info","totals","terms"], "zebra",       "inline", False),
    LayoutConfig("INV-17", "invoice", "magenta",  "Helvetica",   7.5, 6.0, "banner",   4, ["header_info","parties","goods","totals","terms"], "full",        "above",  True ),
    LayoutConfig("INV-18", "invoice", "slate",    "Helvetica",   8.0, 6.5, "split",    2, ["header_info","goods","parties","totals","terms"], "header_only", "prefix", False),
    LayoutConfig("INV-19", "invoice", "amber",    "Times-Roman", 8.5, 7.0, "sidebar",  3, ["header_info","parties","goods","totals","terms"], "full",        "above",  True ),
    LayoutConfig("INV-20", "invoice", "teal",     "Courier",     8.0, 6.5, "minimal",  2, ["parties","header_info","goods","totals","terms"], "none",        "inline", False),
    LayoutConfig("INV-21", "invoice", "indigo",   "Helvetica",   7.5, 6.0, "banner",   3, ["header_info","goods","parties","totals","terms"], "zebra",       "above",  True ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 21 Layouts — Packing List
# ─────────────────────────────────────────────────────────────────────────────

PL_LAYOUTS: list[LayoutConfig] = [
    LayoutConfig("PL-01", "packing_list", "navy",     "Helvetica",   8.0, 6.5, "banner",   3, ["header_info","parties","items","totals"], "full",        "above",  True ),
    LayoutConfig("PL-02", "packing_list", "crimson",  "Helvetica",   8.0, 6.5, "split",    2, ["header_info","parties","items","totals"], "header_only", "above",  False),
    LayoutConfig("PL-03", "packing_list", "forest",   "Times-Roman", 8.5, 7.0, "banner",   4, ["parties","header_info","items","totals"], "zebra",       "inline", True ),
    LayoutConfig("PL-04", "packing_list", "orange",   "Helvetica",   8.0, 6.5, "split",    3, ["header_info","items","parties","totals"], "full",        "prefix", False),
    LayoutConfig("PL-05", "packing_list", "magenta",  "Helvetica",   7.5, 6.0, "centered", 2, ["header_info","parties","items","totals"], "header_only", "above",  True ),
    LayoutConfig("PL-06", "packing_list", "slate",    "Courier",     8.0, 6.5, "banner",   3, ["header_info","parties","items","totals"], "none",        "prefix", False),
    LayoutConfig("PL-07", "packing_list", "amber",    "Helvetica",   8.5, 7.0, "sidebar",  4, ["parties","header_info","items","totals"], "full",        "above",  True ),
    LayoutConfig("PL-08", "packing_list", "teal",     "Times-Roman", 8.0, 6.5, "minimal",  2, ["header_info","parties","items","totals"], "zebra",       "inline", False),
    LayoutConfig("PL-09", "packing_list", "indigo",   "Helvetica",   7.5, 6.0, "split",    3, ["header_info","parties","items","totals"], "full",        "above",  True ),
    LayoutConfig("PL-10", "packing_list", "graphite", "Helvetica",   8.0, 6.5, "banner",   2, ["items","header_info","parties","totals"], "header_only", "prefix", False),
    LayoutConfig("PL-11", "packing_list", "burgundy", "Times-Roman", 8.5, 7.0, "centered", 3, ["parties","header_info","items","totals"], "full",        "above",  True ),
    LayoutConfig("PL-12", "packing_list", "steel",    "Helvetica",   8.0, 6.5, "sidebar",  2, ["header_info","items","parties","totals"], "zebra",       "inline", False),
    LayoutConfig("PL-13", "packing_list", "navy",     "Courier",     7.5, 6.0, "minimal",  4, ["header_info","parties","items","totals"], "none",        "prefix", True ),
    LayoutConfig("PL-14", "packing_list", "crimson",  "Helvetica",   8.0, 6.5, "banner",   3, ["items","parties","header_info","totals"], "full",        "above",  False),
    LayoutConfig("PL-15", "packing_list", "forest",   "Helvetica",   8.5, 7.0, "split",    2, ["header_info","parties","items","totals"], "header_only", "above",  True ),
    LayoutConfig("PL-16", "packing_list", "orange",   "Times-Roman", 8.0, 6.5, "centered", 3, ["parties","items","header_info","totals"], "zebra",       "inline", False),
    LayoutConfig("PL-17", "packing_list", "magenta",  "Helvetica",   7.5, 6.0, "banner",   4, ["header_info","parties","items","totals"], "full",        "above",  True ),
    LayoutConfig("PL-18", "packing_list", "slate",    "Helvetica",   8.0, 6.5, "split",    2, ["header_info","items","parties","totals"], "header_only", "prefix", False),
    LayoutConfig("PL-19", "packing_list", "amber",    "Times-Roman", 8.5, 7.0, "sidebar",  3, ["header_info","parties","items","totals"], "full",        "above",  True ),
    LayoutConfig("PL-20", "packing_list", "teal",     "Courier",     8.0, 6.5, "minimal",  2, ["parties","header_info","items","totals"], "none",        "inline", False),
    LayoutConfig("PL-21", "packing_list", "indigo",   "Helvetica",   7.5, 6.0, "banner",   3, ["header_info","items","parties","totals"], "zebra",       "above",  True ),
]

ALL_LAYOUTS: dict[str, list[LayoutConfig]] = {
    "bl":           BL_LAYOUTS,
    "invoice":      INV_LAYOUTS,
    "packing_list": PL_LAYOUTS,
}
