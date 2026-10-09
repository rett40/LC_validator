"""
zone_extractor.py
=================
Extraction par zones sémantiques pour PDFs natifs (texte vectoriel).

Algorithme en 3 étapes :
  1. Segmentation — localisation des labels-ancres dans les blocs PyMuPDF
                    → délimitation des zones par bandes Y
  2. Reconstruction — texte de chaque zone en ordre de lecture colonne-par-colonne
  3. Extraction — patterns regex ciblés par zone → moins de faux positifs
                  que l'extraction globale

Zones :
  header   — numéro de document, date d'émission
  parties  — expéditeur, destinataire, partie notifiée
  cargo    — conteneur, navire, ports, poids, marchandises
  totals   — montants, packages totaux, conditions de paiement/fret
  footer   — clauses, signatures (ignoré)

Intégration :
  Appelé depuis pipeline._process_one pour les PDFs natifs.
  Comble uniquement les champs MANQUANTS après l'extraction regex standard.
  Tier reporté = "zone_{nom_zone}" (ex: "zone_parties").
"""

from __future__ import annotations

import re
import logging
from pathlib import Path

import fitz

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Labels-ancres qui délimitent le début de chaque zone
# ─────────────────────────────────────────────────────────────────────────────

_ZONE_ANCHORS: dict[str, dict[str, list[str]]] = {
    "header": {
        "bl": [
            "BILL OF LADING", "NON-NEGOTIABLE", "B/L NUMBER", "B/L NO",
        ],
        "invoice": [
            "COMMERCIAL INVOICE", "PROFORMA INVOICE", "TAX INVOICE",
            "INVOICE NO", "INVOICE NUMBER",
        ],
        "packing_list": [
            "PACKING LIST NO", "PACKING LIST NUMBER", "PL NO", "PL NUMBER",
        ],
    },
    "parties": {
        "bl": [
            "SHIPPER / EXPORTER", "SHIPPER/EXPORTER", "SHIPPER",
            "CONSIGNEE", "NOTIFY PARTY",
        ],
        "invoice": [
            "SELLER / EXPORTER", "SELLER/EXPORTER", "SELLER",
            "BUYER / IMPORTER", "BUYER/IMPORTER", "BUYER",
            "EXPORTER", "IMPORTER",
        ],
        "packing_list": [
            "SHIPPER / EXPORTER", "SHIPPER/EXPORTER", "SHIPPER",
            "CONSIGNEE", "EXPORTER",
        ],
    },
    "cargo": {
        "bl": [
            "PORT OF LOADING", "VESSEL", "CONTAINER NO",
            "VOYAGE", "PORT OF DISCHARGE",
        ],
        "invoice": [
            "DESCRIPTION OF GOODS", "DESCRIPTION", "HS CODE",
            "UNIT PRICE", "QUANTITY",
        ],
        "packing_list": [
            "CONTAINER NO", "MARKS AND NUMBERS", "MARKS & NOS",
            "NET WEIGHT", "GROSS WEIGHT", "PACKAGES",
        ],
    },
    "totals": {
        "bl": [
            "FREIGHT & CHARGES", "FREIGHT TERMS", "FREIGHT PAYABLE",
            "NUMBER OF ORIGINAL", "NO. OF ORIGINAL",
        ],
        "invoice": [
            "B/L REFERENCE", "B/L REF",
            "PAYMENT TERMS", "TERMS OF PAYMENT",
            "INCOTERM",
        ],
        "packing_list": [
            "TOTAL PACKAGES", "TOTAL PKGS", "TOTAL GROSS",
            "TOTAL NET", "TOTAL CBM", "TOTAL WEIGHT",
            "PKGS:",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Champs autorisés par zone ET par type de document
# Empêche les captures inter-zones et inter-types (ex: pl_date dans un BL)
# ─────────────────────────────────────────────────────────────────────────────

_ZONE_FIELD_SCOPE: dict[str, dict[str, set[str]]] = {
    "bl": {
        "header": {
            "bl_number",
            "issue_date",
        },
        "parties": {
            "shipper", "consignee", "notify_party", "carrier",
        },
        "cargo": {
            "vessel", "voyage",
            "port_of_loading", "port_of_discharge",
            "place_of_receipt", "place_of_delivery",
            "etd", "eta",
            "container_number", "container_type", "seal_number",
            "gross_weight_kg", "net_weight_kg", "measurement",
        },
        "totals": {
            "freight_terms", "number_of_originals", "incoterm",
        },
    },
    "invoice": {
        "header": {
            "invoice_number", "invoice_date",
        },
        "parties": {
            "seller", "buyer",
            "payment_terms", "incoterm",
        },
        "cargo": {
            "goods_description", "hs_code",
            "currency", "incoterm",
            # total_amount physiquement présent en bas de la zone cargo dans les PDFs
            "total_amount",
        },
        "totals": {
            "total_amount", "subtotal",
            "payment_terms", "incoterm", "currency",
            "bl_reference",
            "invoice_number", "invoice_date",
        },
    },
    "packing_list": {
        "header": {
            "pl_number", "pl_date",
        },
        "parties": {
            "shipper", "consignee",
        },
        "cargo": {
            "container_number", "container_type", "seal_number",
            "gross_weight_kg", "net_weight_kg",
            "total_cbm", "bl_reference",
        },
        "totals": {
            "total_packages", "total_cbm",
            "gross_weight_kg", "net_weight_kg",
        },
    },
}

# Largeur d'une colonne logique en points PDF
_COL_WIDTH = 100


# ─────────────────────────────────────────────────────────────────────────────
# Extracteur principal
# ─────────────────────────────────────────────────────────────────────────────

class ZoneExtractor:
    """
    Extracteur par zones sémantiques pour PDFs natifs.

    Usage :
        result = ZoneExtractor().extract("doc.pdf", "bl")
        # {"fields": {...}, "zone_map": {field: zone_name}}
    """

    def extract(self, pdf_path: Path | str, doc_type: str) -> dict:
        pdf_path = Path(pdf_path)
        all_fields: dict[str, str] = {}
        zone_map:   dict[str, str] = {}

        try:
            doc = fitz.open(str(pdf_path))
            for page in doc:
                blocks = self._get_blocks(page)
                if not blocks:
                    continue

                zones = self._detect_zones(blocks, page.rect.height, doc_type)

                for zone_name, zone_text in zones.items():
                    scope = _ZONE_FIELD_SCOPE.get(doc_type, {}).get(zone_name, set())
                    if not scope:
                        continue
                    fields = self._extract_from_zone(zone_name, zone_text, doc_type)
                    for fname, val in fields.items():
                        if fname not in all_fields:
                            all_fields[fname] = val
                            zone_map[fname] = zone_name

            doc.close()

        except Exception as exc:
            log.warning("[ZoneExtractor] échec sur %s : %s", pdf_path.name, exc)

        return {"fields": all_fields, "zone_map": zone_map}

    # ──────────────────────────────────────────────────────────────────────────
    # Étape 1 — Extraction des blocs PyMuPDF
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_blocks(page) -> list[dict]:
        rect = page.rect
        ph, pw = rect.height, rect.width
        if ph <= 0 or pw <= 0:
            return []

        blocks = []
        for b in page.get_text("blocks"):
            x0, y0, x1, y1, text, _bno, btype = b
            text = text.strip()
            if not text or btype != 0:
                continue
            blocks.append({
                "text": text,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "cx": (x0 + x1) / 2,
            })

        return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))

    # ──────────────────────────────────────────────────────────────────────────
    # Étape 2 — Détection des zones par bandes Y
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_zones(
        blocks: list[dict],
        page_height: float,
        doc_type: str,
    ) -> dict[str, str]:
        """
        Trouve la coordonnée Y de la première ancre de chaque zone,
        construit des intervalles [y_start, y_end[ et affecte chaque bloc.
        """
        ZONE_ORDER = ["header", "parties", "cargo", "totals"]

        zone_entry_y: dict[str, float] = {}
        for zone_name in ZONE_ORDER:
            anchors = _ZONE_ANCHORS.get(zone_name, {}).get(doc_type, [])
            for block in blocks:
                text_up = block["text"].upper()
                if any(a in text_up for a in anchors):
                    zone_entry_y[zone_name] = block["y0"]
                    break

        # Header implicite à y=0 si non détecté
        if "header" not in zone_entry_y:
            zone_entry_y["header"] = 0.0

        # Intervalles triés par Y
        sorted_entries = sorted(zone_entry_y.items(), key=lambda x: x[1])
        intervals: list[tuple[str, float, float]] = []
        for i, (zname, y_start) in enumerate(sorted_entries):
            y_end = sorted_entries[i + 1][1] if i + 1 < len(sorted_entries) else float("inf")
            intervals.append((zname, y_start, y_end))

        # Affectation des blocs
        zone_blocks: dict[str, list[dict]] = {z: [] for z in ZONE_ORDER + ["unassigned"]}
        for block in blocks:
            assigned = "unassigned"
            for zname, y_start, y_end in intervals:
                if y_start <= block["y0"] < y_end:
                    assigned = zname
                    break
            zone_blocks[assigned].append(block)

        # Reconstruction du texte par zone
        return {
            zname: _reconstruct_zone_text(zblocks)
            for zname, zblocks in zone_blocks.items()
            if zblocks
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Étape 3 — Extraction des champs dans une zone
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_from_zone(
        zone_name: str,
        zone_text: str,
        doc_type: str,
    ) -> dict[str, str]:
        """
        Applique les patterns du doc_type au texte de la zone,
        restreints aux champs du scope (doc_type, zone_name).
        """
        try:
            from release2.services.native_extractor import _PATTERNS, _apply, _kv_fallback
        except ImportError:
            from services.native_extractor import _PATTERNS, _apply, _kv_fallback  # type: ignore[no-redef]

        scope = _ZONE_FIELD_SCOPE.get(doc_type, {}).get(zone_name, set())
        if not scope:
            return {}

        all_patterns = _PATTERNS.get(doc_type, {})
        zone_patterns = {f: rxs for f, rxs in all_patterns.items() if f in scope}

        fields = _apply(zone_text, zone_patterns, doc_type)

        # KV-fallback pour les champs encore manquants
        missing = scope - set(fields)
        if missing:
            for fname, val in _kv_fallback(zone_text, missing).items():
                if fname not in fields:
                    # Nettoyer les valeurs avec deux-points parasites en début
                    val = val.lstrip(": \t")
                    if val:
                        fields[fname] = val

        return fields


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction du texte d'une zone (colonne-consciente)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_zone_text(blocks: list[dict]) -> str:
    """
    Reconstruit le texte en préservant l'ordre de lecture colonne-par-colonne.
    Les newlines internes aux blocs PyMuPDF sont conservées (une ligne = un token).
    """
    col_map: dict[int, list[tuple[float, str]]] = {}
    for b in blocks:
        col_key = int(b["cx"] // _COL_WIDTH)
        col_map.setdefault(col_key, []).append((b["y0"], b["text"]))

    lines: list[str] = []
    for col_key in sorted(col_map):
        for _, text in sorted(col_map[col_key], key=lambda t: t[0]):
            lines.append(text)

    return "\n".join(lines)
