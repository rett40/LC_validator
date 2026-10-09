"""
layout_analyzer.py
==================
Analyse structure-first (layout-driven) pour documents commerciaux.

Pipeline :
  1. Segmentation de zones  : header (top) / body / table (spatial clustering)
  2. Détection de tableaux  : alignement colonnes + densité numérique (DBSCAN)
  3. Reconstruction grille  : (row_id, col_id) → cellule
  4. Mapping sémantique     : header de colonne → nom de champ métier

Usage :
  from release2.services.layout_analyzer import LayoutAnalyzer
  analyzer = LayoutAnalyzer()
  result = analyzer.analyze(words, bboxes, doc_type="bl")
  # result["header_tokens"]  → pour LayoutLMv3
  # result["table_fields"]   → champs extraits sans ML
  # result["table_rows"]     → toutes les lignes de données
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mapping sémantique : en-tête de colonne → nom de champ métier
# Clés en minuscules ; la comparaison est "pattern in header_lower"
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_SEMANTIC_MAP: dict[str, dict[str, str]] = {
    "bl": {
        # ── Conteneur ─────────────────────────────────────────────────────────
        "container no":           "container_number",
        "container no.":          "container_number",
        "container number":       "container_number",
        "container":              "container_number",
        "cntr no":                "container_number",
        "cntr no.":               "container_number",
        "cntr":                   "container_number",
        "cont. no.":              "container_number",
        "cont no":                "container_number",
        "container id":           "container_number",
        "cntr id":                "container_number",
        # ── Plomb ─────────────────────────────────────────────────────────────
        "seal no":                "seal_number",
        "seal no.":               "seal_number",
        "seal number":            "seal_number",
        "seal":                   "seal_number",
        "customs seal":           "seal_number",
        # ── Type conteneur ────────────────────────────────────────────────────
        "container type":         "container_type",
        "cntr type":              "container_type",
        "type":                   "container_type",
        "size":                   "container_type",
        "size/type":              "container_type",
        "container size":         "container_type",
        # ── Colis ─────────────────────────────────────────────────────────────
        "no of pkgs":             "number_of_packages",
        "no. of pkgs":            "number_of_packages",
        "no. of packages":        "number_of_packages",
        "no. pkgs":               "number_of_packages",
        "packages":               "number_of_packages",
        "pkgs":                   "number_of_packages",
        "pcs":                    "number_of_packages",
        "cartons":                "number_of_packages",
        "ctns":                   "number_of_packages",
        # ── Poids ─────────────────────────────────────────────────────────────
        "gross weight":           "gross_weight_kg",
        "gross wt":               "gross_weight_kg",
        "gross weight (kg)":      "gross_weight_kg",
        "gross weight (kgs)":     "gross_weight_kg",
        "gross wt (kg)":          "gross_weight_kg",
        "gross wt (kgs)":         "gross_weight_kg",
        "g.w.":                   "gross_weight_kg",
        "g.w":                    "gross_weight_kg",
        "gw":                     "gross_weight_kg",
        "net weight":             "net_weight_kg",
        "net wt":                 "net_weight_kg",
        "net weight (kg)":        "net_weight_kg",
        "net weight (kgs)":       "net_weight_kg",
        "net wt (kg)":            "net_weight_kg",
        "net wt (kgs)":           "net_weight_kg",
        "n.w.":                   "net_weight_kg",
        "n.w":                    "net_weight_kg",
        "nw":                     "net_weight_kg",
        # ── Mesure ────────────────────────────────────────────────────────────
        "measurement":            "measurement",
        "volume":                 "measurement",
        "cbm":                    "measurement",
        "m3":                     "measurement",
        "vol":                    "measurement",
        "volume (cbm)":           "measurement",
        "volume (m3)":            "measurement",
        # ── Description marchandise ───────────────────────────────────────────
        "description":            "goods_description",
        "description of goods":   "goods_description",
        "goods description":      "goods_description",
        "cargo description":      "goods_description",
        "nature of goods":        "goods_description",
        "nature & quantity":      "goods_description",
        "particulars":            "goods_description",
        "commodity":              "goods_description",
        "commodities":            "goods_description",
        "goods":                  "goods_description",
        "items":                  "goods_description",
        "item":                   "goods_description",
        "article":                "goods_description",
        "articles":               "goods_description",
        # ── Marques ───────────────────────────────────────────────────────────
        "marks and numbers":      "shipping_marks",
        "marks & nos":            "shipping_marks",
        "marks":                  "shipping_marks",
        "shipping marks":         "shipping_marks",
    },
    "invoice": {
        # ── Description article ───────────────────────────────────────────────
        "description of goods":   "goods_description",
        "description of items":   "goods_description",
        "description":            "goods_description",
        "goods description":      "goods_description",
        "goods":                  "goods_description",
        "item":                   "goods_description",
        "items":                  "goods_description",
        "article":                "goods_description",
        "articles":               "goods_description",
        "product":                "goods_description",
        "products":               "goods_description",
        "commodity":              "goods_description",
        "commodities":            "goods_description",
        "merchandise":            "goods_description",
        "nature of goods":        "goods_description",
        "particulars":            "goods_description",
        # ── Quantité ──────────────────────────────────────────────────────────
        "quantity":               "quantity",
        "qty":                    "quantity",
        "qty.":                   "quantity",
        "qté":                    "quantity",
        "pcs":                    "quantity",
        "units":                  "quantity",
        "no.":                    "quantity",   # numéro de ligne → souvent quantité
        # ── Prix unitaire ─────────────────────────────────────────────────────
        "unit price":             "unit_price",
        "unit price (usd)":       "unit_price",
        "price per unit":         "unit_price",
        "price":                  "unit_price",
        "rate":                   "unit_price",
        "unit cost":              "unit_price",
        "p/u":                    "unit_price",
        "p.u.":                   "unit_price",
        "per unit":               "unit_price",
        # ── Unité ─────────────────────────────────────────────────────────────
        "unit":                   "unit",
        "uom":                    "unit",
        "u/m":                    "unit",
        # ── Montant ligne ─────────────────────────────────────────────────────
        "total amount":           "total_amount",
        "amount":                 "line_amount",
        "line amount":            "line_amount",
        "net amount":             "line_amount",
        "ext. price":             "line_amount",
        "ext price":              "line_amount",
        "total":                  "line_total",
        "line total":             "line_total",
        "subtotal":               "line_amount",
        "value":                  "line_amount",
        "total value":            "line_total",
        # ── Code douanier ─────────────────────────────────────────────────────
        "hs code":                "hs_code",
        "hs no":                  "hs_code",
        "hs no.":                 "hs_code",
        "hs":                     "hs_code",
        "harmonized":             "hs_code",
        "tariff code":            "hs_code",
        "tariff no":              "hs_code",
        "tariff no.":             "hs_code",
        "htc":                    "hs_code",
        "commodity code":         "hs_code",
        "customs code":           "hs_code",
        "hts":                    "hs_code",
        # ── Origine ───────────────────────────────────────────────────────────
        "country of origin":      "country_of_origin",
        "origin":                 "country_of_origin",
        "c/o":                    "country_of_origin",
    },
    "packing_list": {
        # ── Conteneur ─────────────────────────────────────────────────────────
        "container no":           "container_number",
        "container no.":          "container_number",
        "container number":       "container_number",
        "container":              "container_number",
        "cntr no":                "container_number",
        "cntr no.":               "container_number",
        # ── Description ───────────────────────────────────────────────────────
        "description":            "goods_description",
        "description of goods":   "goods_description",
        "goods description":      "goods_description",
        "goods":                  "goods_description",
        "item":                   "goods_description",
        "items":                  "goods_description",
        "article":                "goods_description",
        "articles":               "goods_description",
        "product":                "goods_description",
        "products":               "goods_description",
        "commodity":              "goods_description",
        "commodities":            "goods_description",
        "nature of goods":        "goods_description",
        "particulars":            "goods_description",
        # ── Colis ─────────────────────────────────────────────────────────────
        "cartons":                "number_of_packages",
        "carton":                 "number_of_packages",
        "packages":               "number_of_packages",
        "pkgs":                   "number_of_packages",
        "no. of ctns":            "number_of_packages",
        "ctns":                   "number_of_packages",
        "ctn":                    "number_of_packages",
        "boxes":                  "number_of_packages",
        "pcs":                    "number_of_packages",
        "pieces":                 "number_of_packages",
        "qty":                    "number_of_packages",
        "quantity":               "number_of_packages",
        # ── Unité ─────────────────────────────────────────────────────────────
        "unit":                   "package_unit",
        "uom":                    "package_unit",
        "u/m":                    "package_unit",
        # ── Poids brut ────────────────────────────────────────────────────────
        "gross weight":           "gross_weight_kg",
        "gross wt":               "gross_weight_kg",
        "gross weight (kg)":      "gross_weight_kg",
        "gross weight (kgs)":     "gross_weight_kg",
        "gross wt (kg)":          "gross_weight_kg",
        "gross wt (kgs)":         "gross_weight_kg",
        "gross":                  "gross_weight_kg",
        "g.w.":                   "gross_weight_kg",
        "g.w":                    "gross_weight_kg",
        "gw":                     "gross_weight_kg",
        "g/w":                    "gross_weight_kg",
        # ── Poids net ─────────────────────────────────────────────────────────
        "net weight":             "net_weight_kg",
        "net wt":                 "net_weight_kg",
        "net weight (kg)":        "net_weight_kg",
        "net weight (kgs)":       "net_weight_kg",
        "net wt (kg)":            "net_weight_kg",
        "net wt (kgs)":           "net_weight_kg",
        "net":                    "net_weight_kg",
        "n.w.":                   "net_weight_kg",
        "n.w":                    "net_weight_kg",
        "nw":                     "net_weight_kg",
        "n/w":                    "net_weight_kg",
        # ── Mesure ────────────────────────────────────────────────────────────
        "cbm":                    "measurement",
        "measurement":            "measurement",
        "m3":                     "measurement",
        "vol":                    "measurement",
        "volume":                 "measurement",
        "volume (cbm)":           "measurement",
        "volume (m3)":            "measurement",
        "dimensions":             "dimensions",
        # ── Marques ───────────────────────────────────────────────────────────
        "marks and numbers":      "shipping_marks",
        "marks & nos":            "shipping_marks",
        "marks":                  "shipping_marks",
        "shipping marks":         "shipping_marks",
    },
}

# Fallback : union de tous les doc_types (utilisé si doc_type="unknown")
_FALLBACK_MAP: dict[str, str] = {}
for _dt_map in COLUMN_SEMANTIC_MAP.values():
    for _k, _v in _dt_map.items():
        if _k not in _FALLBACK_MAP:
            _FALLBACK_MAP[_k] = _v

# Regex pour détecter les textes d'en-tête de colonne (à exclure des data_rows)
_HEADER_CELL_RE = re.compile(
    r"^("
    # Conteneur
    r"container\s*(no\.?|number|n[o°]|id)?\.?|cntr(\s*no\.?)?|cont\.?\s*no\.?|container\s*id|"
    # Plomb
    r"seal\s*(no\.?|number)?|customs\s*seal|"
    # Type/taille conteneur
    r"container\s*type|cntr\s*type|size[/\s]?type|container\s*size|type|size|"
    # Colis / quantité
    r"no\.?\s*of\s*pkg|no\.?\s*of\s*ctns?|packages?|pkgs?|cartons?|ctns?|pcs|pieces?|"
    r"qty\.?|quantity|units?|"
    # Poids brut
    r"gross\s*(weight|wt)\.?(\s*\(kgs?\))?|g\.w\.?|gw|g/w|"
    # Poids net
    r"net\s*(weight|wt)\.?(\s*\(kgs?\))?|n\.w\.?|nw|n/w|"
    # Mesure
    r"measurement|volume\s*(\(cbm\)|\(m3\))?|cbm|m3|vol|dimensions?|"
    # Description marchandise
    r"description(\s*of\s*(goods|items?))?|goods\s*description|cargo\s*description|"
    r"nature\s*of\s*goods|nature\s*&\s*quantity|particulars|"
    r"goods|items?|articles?|products?|commodit(y|ies)|merchandise|"
    # Prix / montant
    r"unit\s*price(\s*\([a-z]{3}\))?|price\s*per\s*unit|rate|unit\s*cost|p\.?u\.?|per\s*unit|price|"
    r"line\s*(total|amount)|net\s*amount|ext\.?\s*price|subtotal|"
    r"amount|total(\s*amount)?|total\s*value|value|"
    # Code douanier
    r"hs\s*(code|no\.?)|harmonized|tariff\s*(code|no\.?)?|htc|hts|commodity\s*code|customs\s*code|"
    # Marques
    r"marks(\s*(and|\&)\s*numbers?)?|marks\s*&\s*nos|shipping\s*marks|"
    # Origine
    r"country\s*of\s*origin|origin|c/o|"
    # Unité
    r"uom|u/m"
    r")$",
    re.IGNORECASE,
)

_AGGREGATE_CELL_RE = re.compile(
    r"^(total|grand\s+total|subtotal|sub-total|sub\s+total|authorized|"
    r"signatory|signature|stamp)\b",
    re.IGNORECASE,
)


def _is_aggregate_row_values(values) -> bool:
    text = " ".join(str(v).strip() for v in values if str(v).strip())
    if not text:
        return True
    return bool(_AGGREGATE_CELL_RE.match(text))


# ─────────────────────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentZone:
    """Région fonctionnelle d'une page."""
    zone_type: str          # "header" | "body" | "table"
    tokens:    list[dict]   # [{"text": str, "bbox": [x0,y0,x1,y1]}, ...]
    y_start:   float = 0.0
    y_end:     float = 0.0


@dataclass
class ReconstructedTable:
    """Tableau reconstruit sous forme de grille logique."""
    header_row:    dict[int, str]        # col_id → texte d'en-tête
    data_rows:     list[dict[int, str]]  # [{col_id → texte cellule}, ...]
    col_semantics: dict[int, str] = field(default_factory=dict)  # col_id → field_name

    # ── Conversions ───────────────────────────────────────────────────────────

    def to_field_rows(self) -> list[dict[str, str]]:
        """Chaque data_row → {field_name: valeur}. Lignes sans mapping ou avec
        valeurs qui ressemblent à des en-têtes de colonnes sont exclues."""
        rows = []
        for row in self.data_rows:
            if _is_aggregate_row_values(row.values()):
                continue
            mapped = {}
            for col_id, cell in row.items():
                fname = self.col_semantics.get(col_id)
                val   = cell.strip()
                if not fname or not val:
                    continue
                # Rejeter les valeurs qui sont en fait des labels de colonnes
                if fname not in {"package_unit", "unit"} and _HEADER_CELL_RE.match(val):
                    continue
                mapped[fname] = val
            if mapped:
                rows.append(mapped)
        return rows

    def first_values(self) -> dict[str, str]:
        """Première valeur non-vide et non-label par champ sémantique."""
        result: dict[str, str] = {}
        for row in self.data_rows:
            if _is_aggregate_row_values(row.values()):
                continue
            for col_id, cell in row.items():
                fname = self.col_semantics.get(col_id)
                val   = cell.strip()
                if not fname or not val or fname in result:
                    continue
                if fname not in {"package_unit", "unit"} and _HEADER_CELL_RE.match(val):
                    continue
                result[fname] = val
        return result

    def to_legacy_rows(self) -> list[dict[str, Any]]:
        """Format compatible avec le pipeline existant (tables par row_index/cells)."""
        rows = []
        for i, row in enumerate(self.data_rows, start=1):
            if _is_aggregate_row_values(row.values()):
                continue
            cells = {
                self.col_semantics.get(col_id, f"col_{col_id}"): cell
                for col_id, cell in row.items()
            }
            rows.append({"row_index": i, "cells": cells})
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# Analyseur principal
# ─────────────────────────────────────────────────────────────────────────────

class LayoutAnalyzer:
    """
    Analyse structure-first d'un document commercial.

    Paramètres
    ----------
    header_ratio   : fraction de la hauteur de page réservée à la zone header (défaut 0.38)
    min_table_rows : nombre minimum de lignes de données pour valider un tableau (défaut 2)
    min_table_cols : nombre minimum de colonnes pour valider un tableau (défaut 2)
    """

    def __init__(
        self,
        header_ratio:   float = 0.38,
        min_table_rows: int   = 1,
        min_table_cols: int   = 2,
    ) -> None:
        self.header_ratio   = header_ratio
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        self._num_re        = re.compile(r"[\d.,]{2,}")

    # ── API publique ──────────────────────────────────────────────────────────

    def analyze(
        self,
        words:    list[str],
        bboxes:   list[list[int]],
        doc_type: str = "unknown",
    ) -> dict[str, Any]:
        """
        Analyse le layout d'une page et retourne :

          header_tokens  : liste de tokens {text, bbox} pour LayoutLMv3
                           (vide → utiliser tous les tokens)
          tables         : list[ReconstructedTable]
          table_fields   : dict {field_name: première_valeur}  (sans ML)
          table_rows     : list[{field_name: valeur}]  (toutes les lignes)
          legacy_tables  : format pipeline existant [{row_index, cells}, ...]
        """
        empty = {"header_tokens": [], "tables": [], "table_fields": {}, "table_rows": [], "legacy_tables": []}
        if not words:
            return empty

        tokens = [{"text": w, "bbox": b} for w, b in zip(words, bboxes)]
        zones  = self._segment_zones(tokens, doc_type)

        header_tokens = [t for z in zones if z.zone_type == "header" for t in z.tokens]
        table_zones   = [z for z in zones if z.zone_type == "table"]

        tables: list[ReconstructedTable] = []
        for zone in table_zones:
            table = self._reconstruct_table(zone.tokens, doc_type)
            if table is not None:
                table.col_semantics = self._map_column_semantics(table.header_row, doc_type)
                tables.append(table)
                log.debug(
                    "Table reconstruite : %d cols × %d lignes | sémantique: %s",
                    len(table.header_row), len(table.data_rows),
                    list(table.col_semantics.values()),
                )

        table_fields: dict[str, str] = {}
        table_rows:   list[dict[str, str]] = []
        legacy_tables: list[dict[str, Any]] = []
        for table in tables:
            table_rows.extend(table.to_field_rows())
            legacy_tables.extend(table.to_legacy_rows())
            for fname, val in table.first_values().items():
                if fname not in table_fields:
                    table_fields[fname] = val

        # Fallback : si aucun header token détecté, garder liste vide
        # (extraction_service utilise tous les tokens dans ce cas)
        return {
            "header_tokens": header_tokens,
            "tables":        tables,
            "table_fields":  table_fields,
            "table_rows":    table_rows,
            "legacy_tables": legacy_tables,
        }

    # ── Segmentation de zones ─────────────────────────────────────────────────

    def _segment_zones(self, tokens: list[dict], doc_type: str = "unknown") -> list[DocumentZone]:
        if not tokens:
            return []

        page_bottom = max(t["bbox"][3] for t in tokens)
        header_cut  = page_bottom * self.header_ratio
        first_table_y = self._first_semantic_table_header_y(tokens, doc_type)
        if first_table_y is not None:
            header_cut = min(header_cut, max(0.0, first_table_y - 8.0))

        header_tokens = [t for t in tokens if t["bbox"][3] <= header_cut]
        body_tokens   = [t for t in tokens if t["bbox"][3] >  header_cut]

        zones: list[DocumentZone] = []
        if header_tokens:
            zones.append(DocumentZone("header", header_tokens, 0.0, header_cut))

        if body_tokens:
            zones.extend(self._split_body(body_tokens, doc_type))

        log.debug(
            "Zones : header=%d tokens | %d zone(s) corps",
            len(header_tokens), len(zones) - (1 if header_tokens else 0),
        )
        return zones

    def _first_semantic_table_header_y(
        self,
        tokens: list[dict],
        doc_type: str,
    ) -> float | None:
        if not tokens:
            return None

        heights = np.array([max(1, t["bbox"][3] - t["bbox"][1]) for t in tokens])
        y_centers = np.array([(t["bbox"][1] + t["bbox"][3]) / 2.0 for t in tokens])
        row_eps = max(12.0, float(np.median(heights)) * 1.4)
        labels = self._fix_noise(
            DBSCAN(eps=row_eps, min_samples=1).fit_predict(y_centers.reshape(-1, 1))
        )

        rows_map: dict[int, list[dict]] = defaultdict(list)
        row_y: dict[int, list[float]] = defaultdict(list)
        for token, label, yc in zip(tokens, labels, y_centers):
            rows_map[int(label)].append(token)
            row_y[int(label)].append(float(yc))

        for row_id in sorted(rows_map, key=lambda k: float(np.mean(row_y[k]))):
            row_tokens = sorted(rows_map[row_id], key=lambda t: t["bbox"][0])
            row_text = " ".join(t["text"] for t in row_tokens)
            if _AGGREGATE_CELL_RE.match(row_text.strip()):
                continue
            if self._header_keyword_score(row_text, doc_type) >= 2:
                return float(min(t["bbox"][1] for t in row_tokens))
        return None

    def _split_body(self, tokens: list[dict], doc_type: str = "unknown") -> list[DocumentZone]:
        """Découpe le corps en blocs (table vs texte libre) par clustering de lignes."""
        if not tokens:
            return []

        heights   = np.array([max(1, t["bbox"][3] - t["bbox"][1]) for t in tokens])
        y_centers = np.array([(t["bbox"][1] + t["bbox"][3]) / 2.0  for t in tokens])
        row_eps   = max(12.0, float(np.median(heights)) * 1.4)

        raw_labels = DBSCAN(eps=row_eps, min_samples=1).fit_predict(y_centers.reshape(-1, 1))
        row_labels = self._fix_noise(raw_labels)

        rows_map: dict[int, list[dict]]   = defaultdict(list)
        row_y:    dict[int, list[float]]  = defaultdict(list)
        for token, lbl, yc in zip(tokens, row_labels, y_centers):
            rows_map[int(lbl)].append(token)
            row_y[int(lbl)].append(float(yc))

        sorted_rows = sorted(rows_map.items(), key=lambda kv: float(np.mean(row_y[kv[0]])))

        # Seuil de grand écart vertical : 2.5× la hauteur médiane des lignes
        median_row_h = float(np.median(heights))
        gap_threshold = median_row_h * 2.5

        # Classifier chaque ligne puis grouper les consécutives de même nature
        # en coupant aussi sur les grands espaces verticaux (séparateurs de sections)
        classified: list[tuple[str, list[dict], float]] = []  # (type, tokens, y_center)
        for _, r_tokens in sorted_rows:
            yc   = float(np.mean([(t["bbox"][1] + t["bbox"][3]) / 2.0 for t in r_tokens]))
            kind = "table" if self._is_table_row(r_tokens, doc_type) else "body"
            classified.append((kind, r_tokens, yc))

        zones: list[DocumentZone] = []
        if not classified:
            return zones

        cur_type   = classified[0][0]
        cur_tokens = list(classified[0][1])
        prev_yc    = classified[0][2]

        for zone_type, row_tokens, yc in classified[1:]:
            # Couper si grand écart vertical (nouvelle section) OU changement de type
            table_gap_threshold = max(gap_threshold, median_row_h * 4.5)
            big_gap = (
                (yc - prev_yc) > table_gap_threshold
                if zone_type == cur_type == "table"
                else (yc - prev_yc) > gap_threshold
            )
            if zone_type != cur_type or big_gap:
                zones.append(self._make_zone(cur_type, cur_tokens))
                cur_type   = zone_type
                cur_tokens = list(row_tokens)
            else:
                cur_tokens.extend(row_tokens)
            prev_yc = yc
        zones.append(self._make_zone(cur_type, cur_tokens))

        # Rétrograder les zones table avec trop peu de lignes en "body"
        result: list[DocumentZone] = []
        for zone in zones:
            if zone.zone_type == "table" and self._count_rows(zone.tokens) < 2:
                zone = DocumentZone("body", zone.tokens, zone.y_start, zone.y_end)
            result.append(zone)

        return result

    # ── Utilitaires de zone ───────────────────────────────────────────────────

    @staticmethod
    def _make_zone(zone_type: str, tokens: list[dict]) -> DocumentZone:
        y_start = float(min(t["bbox"][1] for t in tokens))
        y_end   = float(max(t["bbox"][3] for t in tokens))
        return DocumentZone(zone_type, tokens, y_start, y_end)

    def _count_rows(self, tokens: list[dict]) -> int:
        if not tokens:
            return 0
        return len(self._cluster_rows_by_gap(tokens))

    def _is_table_row(self, row_tokens: list[dict], doc_type: str = "unknown") -> bool:
        """
        Heuristique : une ligne est tabulaire si :
          - densité numérique ≥ 40 %, OU
          - ≥ 3 tokens avec grand écart inter-colonne (> 80 unités normalisées)
        """
        if len(row_tokens) < 2:
            return False
        row_text = " ".join(t["text"] for t in sorted(row_tokens, key=lambda t: t["bbox"][0]))
        if self._header_keyword_score(row_text, doc_type) >= 2:
            return True
        numeric_ratio = sum(
            1 for t in row_tokens if self._num_re.search(t["text"])
        ) / len(row_tokens)
        if numeric_ratio >= 0.4:
            return True
        if len(row_tokens) >= 3:
            xs    = sorted((t["bbox"][0] + t["bbox"][2]) / 2.0 for t in row_tokens)
            gaps  = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
            if gaps and max(gaps) > 80:
                return True
        return False

    @staticmethod
    def _header_keyword_score(text: str, doc_type: str = "unknown") -> int:
        lower = text.lower()
        sem_map = COLUMN_SEMANTIC_MAP.get(doc_type, _FALLBACK_MAP)
        return sum(1 for pattern in sem_map if pattern in lower)

    # ── Reconstruction tabulaire ──────────────────────────────────────────────

    def _cluster_rows_by_gap(self, tokens: list[dict]) -> list[list[dict]]:
        """
        Regroupe les tokens en lignes par analyse de gaps verticaux.
        Plus robuste que DBSCAN pour les tableaux avec cellules vides ou
        espacements irréguliers (fusion de lignes, multi-conteneurs).
        """
        if not tokens:
            return []

        sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))
        heights = [max(1, t["bbox"][3] - t["bbox"][1]) for t in sorted_tokens]
        median_h = float(np.median(heights)) if heights else 10.0
        # Seuil de gap : moitié de la hauteur médiane pour tolérer le chevauchement vertical
        gap_threshold = max(4.0, median_h * 0.5)

        rows: list[list[dict]] = []
        current_row: list[dict] = [sorted_tokens[0]]
        current_y_max = float(sorted_tokens[0]["bbox"][3])

        for token in sorted_tokens[1:]:
            token_y_top = float(token["bbox"][1])
            if token_y_top > current_y_max + gap_threshold:
                rows.append(current_row)
                current_row = [token]
                current_y_max = float(token["bbox"][3])
            else:
                current_row.append(token)
                current_y_max = max(current_y_max, float(token["bbox"][3]))

        if current_row:
            rows.append(current_row)

        # Trier les tokens à l'intérieur de chaque ligne par X croissant
        return [
            sorted(row, key=lambda t: (t["bbox"][0], t["bbox"][1]))
            for row in rows
        ]

    def _reconstruct_table(self, tokens: list[dict], doc_type: str = "unknown") -> ReconstructedTable | None:
        if len(tokens) < 4:
            return None

        # Détection des lignes par gap Y (remplace DBSCAN pour plus de robustesse
        # sur les tableaux avec cellules vides ou multi-conteneurs)
        rows = self._cluster_rows_by_gap(tokens)
        n_rows = len(rows)
        if n_rows < 2:
            return None

        header_idx, header_score = self._find_best_header_row_from_rows(rows, doc_type)
        if header_score <= 0:
            return None

        header_cells = self._build_header_cells(rows[header_idx])
        if len(header_cells) < self.min_table_cols:
            return None

        header_row = {idx: cell["text"] for idx, cell in enumerate(header_cells)}
        boundaries = self._column_boundaries(header_cells)
        data_rows: list[dict[int, str]] = []
        for row_tokens in rows[header_idx + 1:]:
            row = self._assign_row_to_columns(row_tokens, boundaries, len(header_cells))
            if _is_aggregate_row_values(row.values()):
                continue
            if any(v.strip() and not _HEADER_CELL_RE.match(v.strip()) for v in row.values()):
                data_rows.append(row)

        if len(data_rows) < self.min_table_rows:
            return None

        return ReconstructedTable(header_row=header_row, data_rows=data_rows)

    def _find_best_header_row_from_rows(
        self,
        rows: list[list[dict]],
        doc_type: str,
    ) -> tuple[int, int]:
        best_idx = 0
        best_score = -1
        for row_idx, row_tokens in enumerate(rows[: min(5, len(rows))]):
            row_text = " ".join(t["text"] for t in row_tokens)
            if _AGGREGATE_CELL_RE.match(row_text.strip()):
                score = -1
            else:
                score = self._header_keyword_score(row_text, doc_type)
            if score > best_score:
                best_idx = row_idx
                best_score = score
        return best_idx, best_score

    @staticmethod
    def _build_header_cells(header_tokens: list[dict]) -> list[dict[str, Any]]:
        if not header_tokens:
            return []

        tokens = sorted(header_tokens, key=lambda t: (t["bbox"][0], t["bbox"][1]))
        widths = [max(1, t["bbox"][2] - t["bbox"][0]) for t in tokens]
        gap_threshold = max(18.0, float(np.median(widths)) * 0.55)

        cells: list[dict[str, Any]] = []
        current: list[dict] = [tokens[0]]

        for token in tokens[1:]:
            gap = token["bbox"][0] - current[-1]["bbox"][2]
            if gap > gap_threshold:
                cells.append(LayoutAnalyzer._tokens_to_cell(current))
                current = [token]
            else:
                current.append(token)

        cells.append(LayoutAnalyzer._tokens_to_cell(current))
        return [cell for cell in cells if cell["text"]]

    @staticmethod
    def _tokens_to_cell(tokens: list[dict]) -> dict[str, Any]:
        ordered = sorted(tokens, key=lambda t: (t["bbox"][0], t["bbox"][1]))
        return {
            "text": " ".join(t["text"] for t in ordered).strip(),
            "x0": min(t["bbox"][0] for t in ordered),
            "x1": max(t["bbox"][2] for t in ordered),
        }

    @staticmethod
    def _column_boundaries(header_cells: list[dict[str, Any]]) -> list[float]:
        boundaries: list[float] = []
        for left, right in zip(header_cells, header_cells[1:]):
            gap = float(right["x0"]) - float(left["x1"])
            midpoint = (float(left["x1"]) + float(right["x0"])) / 2.0
            right_guard = float(right["x0"]) - min(18.0, max(6.0, gap * 0.30))
            boundaries.append(max(midpoint, right_guard))
        return boundaries

    @staticmethod
    def _assign_row_to_columns(
        row_tokens: list[dict],
        boundaries: list[float],
        n_cols: int,
    ) -> dict[int, str]:
        cells: dict[int, list[dict]] = defaultdict(list)
        for token in row_tokens:
            x_center = (token["bbox"][0] + token["bbox"][2]) / 2.0
            col = 0
            while col < len(boundaries) and x_center > boundaries[col]:
                col += 1
            col = max(0, min(n_cols - 1, col))
            cells[col].append(token)

        return {
            col: " ".join(
                t["text"] for t in sorted(cells.get(col, []), key=lambda t: t["bbox"][0])
            ).strip()
            for col in range(n_cols)
        }

    # ── Sélection de la ligne d'en-tête ──────────────────────────────────────

    @staticmethod
    def _find_best_header_row(
        grid:   dict[tuple[int, int], list[str]],
        n_cols: int,
        n_rows: int,
    ) -> int:
        """
        Parmi les 3 premières lignes, retourne l'index de celle dont les cellules
        contiennent le plus de mots-clés correspondant à COLUMN_SEMANTIC_MAP.
        Fallback : index 0 (comportement par défaut).
        """
        all_patterns = list(_FALLBACK_MAP.keys())
        best_idx   = 0
        best_score = -1

        for row_idx in range(min(3, n_rows)):
            row_text = " ".join(
                " ".join(grid.get((row_idx, col), []))
                for col in range(n_cols)
            ).lower()
            score = sum(1 for p in all_patterns if p in row_text)
            if score > best_score:
                best_score = score
                best_idx   = row_idx

        return best_idx

    # ── Mapping sémantique ────────────────────────────────────────────────────

    def _map_column_semantics(
        self, header_row: dict[int, str], doc_type: str
    ) -> dict[int, str]:
        sem_map = COLUMN_SEMANTIC_MAP.get(doc_type, _FALLBACK_MAP)
        result: dict[int, str] = {}
        for col_id, header_text in header_row.items():
            key = header_text.lower().strip()
            # Longest-match first pour éviter "type" de matcher avant "container type"
            for pattern in sorted(sem_map, key=len, reverse=True):
                if pattern in key:
                    result[col_id] = sem_map[pattern]
                    break
        return result

    # ── Utilitaires DBSCAN ────────────────────────────────────────────────────

    @staticmethod
    def _fix_noise(labels: np.ndarray) -> np.ndarray:
        """Réassigne les points bruit (-1) à des clusters uniques."""
        if not (labels == -1).any():
            return labels
        labels = labels.copy()
        next_lbl = int(labels.max()) + 1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = next_lbl
                next_lbl += 1
        return labels
