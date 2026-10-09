"""
block_segmenter.py
==================
Segmente une page de document LC en blocs sémantiques distincts.

Chaque bloc regroupe des tokens ayant le même rôle fonctionnel.
Le recadrage image correspondant est utilisé pour donner à LayoutLMv3
un meilleur contexte visuel localisé.

Blocs reconnus :
  header_meta — en-tête : numéro de document, date, société émettrice
  parties     — parties commerciales : shipper, consignee, notify
  transport   — infos transport : vessel, ports, ETD/ETA, freight
  cargo       — tableau marchandises : conteneurs, produits, poids, prix
  totals      — totaux, signatures, informations bancaires

Usage :
  from release2.services.block_segmenter import BlockSegmenter
  seg = BlockSegmenter()
  blocks = seg.segment(words, bboxes, doc_type="bl")
  for blk in blocks:
      print(blk.block_type, blk.y_start, blk.y_end, len(blk.tokens))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Mots-clés par type de bloc — pondération par longueur du pattern
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_KEYWORDS: dict[str, list[str]] = {
    "header_meta": [
        "bill of lading", "invoice no", "packing list no", "b/l no", "inv no",
        "pl no", "bl number", "bl n°", "date of issue", "issue date",
        "place of issue", "original", "copy", "number of originals",
    ],
    "parties": [
        "shipper", "exporter", "consignee", "notify party", "also notify",
        "buyer", "seller", "importer", "bill to", "ship to", "sold to",
    ],
    "transport": [
        "vessel", "voyage", "port of loading", "port of discharge",
        "place of receipt", "place of delivery", "pre-carriage",
        "final destination", "etd", "eta", "freight payable",
        "freight terms", "service type", "incoterm",
    ],
    "cargo": [
        "container no", "container number", "seal no", "seal number",
        "description of goods", "goods description", "description",
        "quantity", "unit price", "hs code", "gross weight", "net weight",
        "cbm", "measurement", "packages", "marks and numbers",
        "no of pkgs", "container type",
    ],
    "totals": [
        "total amount", "grand total", "subtotal", "sub-total",
        "authorized", "signatory", "signature", "stamp", "bank",
        "swift", "iban", "bic", "domiciliation", "payment terms",
    ],
}

# Ordre de lecture naturel dans un document commercial
_BLOCK_ORDER = ["header_meta", "parties", "transport", "cargo", "totals"]
_BLOCK_RANK  = {bt: i for i, bt in enumerate(_BLOCK_ORDER)}

# Regex pour détecter les lignes à contenu purement numérique (typiques du cargo)
_NUMERIC_RE = re.compile(r'\d')


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass bloc
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentBlock:
    """Un bloc sémantique d'une page de document."""
    block_type: str
    tokens:     list[dict]    # [{"text": str, "bbox": [x0,y0,x1,y1]}, ...]
    y_start:    float = 0.0   # coordonnée 0-1000
    y_end:      float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Segmenteur principal
# ─────────────────────────────────────────────────────────────────────────────

class BlockSegmenter:
    """
    Segmente les tokens d'une page en blocs sémantiques
    en combinant position verticale, densité de mots-clés et continuité.
    """

    def segment(
        self,
        words:    list[str],
        bboxes:   list[list[int]],
        doc_type: str = "unknown",
    ) -> list[DocumentBlock]:
        """
        Retourne une liste ordonnée de DocumentBlock couvrant toute la page.
        Les blocs sont dans l'ordre de leur position verticale.
        """
        if not words:
            return []

        tokens = [{"text": w, "bbox": b} for w, b in zip(words, bboxes)]
        rows   = self._group_into_rows(tokens)
        if not rows:
            return []

        page_bottom = max(t["bbox"][3] for t in tokens)
        row_types   = [self._classify_row(row, doc_type, page_bottom) for row in rows]

        # Éviter les sauts en arrière dans l'ordre des blocs
        row_types = self._enforce_forward_order(row_types)

        blocks = self._merge_consecutive_rows(rows, row_types)
        blocks = self._coalesce_blocks(blocks)
        return blocks

    # ── Groupement en lignes ──────────────────────────────────────────────────

    def _group_into_rows(self, tokens: list[dict]) -> list[list[dict]]:
        sorted_tokens = sorted(tokens, key=lambda t: (t["bbox"][1], t["bbox"][0]))
        if not sorted_tokens:
            return []

        heights   = [max(1, t["bbox"][3] - t["bbox"][1]) for t in sorted_tokens]
        median_h  = float(np.median(heights)) if heights else 10.0
        gap_thr   = max(3.0, median_h * 0.5)

        rows: list[list[dict]] = []
        cur_row  = [sorted_tokens[0]]
        cur_ymax = float(sorted_tokens[0]["bbox"][3])

        for tok in sorted_tokens[1:]:
            if float(tok["bbox"][1]) > cur_ymax + gap_thr:
                rows.append(sorted(cur_row, key=lambda t: t["bbox"][0]))
                cur_row  = [tok]
                cur_ymax = float(tok["bbox"][3])
            else:
                cur_row.append(tok)
                cur_ymax = max(cur_ymax, float(tok["bbox"][3]))

        if cur_row:
            rows.append(sorted(cur_row, key=lambda t: t["bbox"][0]))

        return rows

    # ── Classification d'une ligne ────────────────────────────────────────────

    def _classify_row(
        self,
        row_tokens: list[dict],
        doc_type:   str,
        page_bottom: float,
    ) -> str:
        row_text = " ".join(t["text"] for t in row_tokens).lower()
        scores: dict[str, float] = {bt: 0.0 for bt in _BLOCK_ORDER}

        # Score basé sur les mots-clés
        for block_type, keywords in BLOCK_KEYWORDS.items():
            for kw in keywords:
                if kw in row_text:
                    # Pondérer par la longueur du pattern (les patterns longs sont plus précis)
                    scores[block_type] += len(kw.split())

        # Boost doc_type
        if doc_type == "bl":
            scores["transport"] *= 1.4
            scores["cargo"]     *= 1.2
        elif doc_type == "invoice":
            scores["cargo"]     *= 1.3
            scores["totals"]    *= 1.2
        elif doc_type == "packing_list":
            scores["cargo"]     *= 1.4

        # Heuristique numérique : ligne dense en chiffres → probablement cargo
        num_ratio = sum(
            1 for t in row_tokens if _NUMERIC_RE.search(t["text"])
        ) / max(len(row_tokens), 1)
        if num_ratio > 0.5:
            scores["cargo"] += 3.0

        # Heuristique position : si pas de mot-clé, estimer par position Y
        row_y    = float(np.mean([(t["bbox"][1] + t["bbox"][3]) / 2 for t in row_tokens]))
        y_ratio  = row_y / max(page_bottom, 1.0)

        if max(scores.values()) == 0:
            if y_ratio < 0.20:
                return "header_meta"
            elif y_ratio < 0.45:
                return "parties"
            elif y_ratio < 0.65:
                return "transport"
            elif y_ratio < 0.88:
                return "cargo"
            else:
                return "totals"

        return max(scores, key=lambda k: (scores[k], -_BLOCK_RANK[k]))

    # ── Ordre cohérent (pas de retour en arrière) ─────────────────────────────

    @staticmethod
    def _enforce_forward_order(row_types: list[str]) -> list[str]:
        """
        Empêche les transitions en arrière (ex: cargo → parties).
        Remplace le type incohérent par le type précédent ou le type courant.
        """
        if not row_types:
            return row_types
        result = [row_types[0]]
        max_rank = _BLOCK_RANK[row_types[0]]
        for rt in row_types[1:]:
            rank = _BLOCK_RANK.get(rt, 0)
            if rank >= max_rank:
                result.append(rt)
                max_rank = rank
            else:
                result.append(result[-1])  # conserver le type précédent
        return result

    # ── Fusion et coalescing ──────────────────────────────────────────────────

    @staticmethod
    def _merge_consecutive_rows(
        rows:      list[list[dict]],
        row_types: list[str],
    ) -> list[DocumentBlock]:
        if not rows:
            return []

        cur_type   = row_types[0]
        cur_tokens = list(rows[0])
        blocks: list[DocumentBlock] = []

        for row_toks, rt in zip(rows[1:], row_types[1:]):
            if rt == cur_type:
                cur_tokens.extend(row_toks)
            else:
                blocks.append(BlockSegmenter._make_block(cur_type, cur_tokens))
                cur_type   = rt
                cur_tokens = list(row_toks)

        blocks.append(BlockSegmenter._make_block(cur_type, cur_tokens))
        return blocks

    @staticmethod
    def _make_block(block_type: str, tokens: list[dict]) -> DocumentBlock:
        y_start = float(min(t["bbox"][1] for t in tokens))
        y_end   = float(max(t["bbox"][3] for t in tokens))
        return DocumentBlock(block_type=block_type, tokens=tokens, y_start=y_start, y_end=y_end)

    @staticmethod
    def _coalesce_blocks(
        blocks: list[DocumentBlock], gap_threshold: float = 25.0
    ) -> list[DocumentBlock]:
        """Fusionne les blocs du même type séparés par moins de `gap_threshold` unités."""
        if not blocks:
            return []
        result = [blocks[0]]
        for blk in blocks[1:]:
            prev = result[-1]
            if blk.block_type == prev.block_type and (blk.y_start - prev.y_end) < gap_threshold:
                result[-1] = DocumentBlock(
                    block_type = prev.block_type,
                    tokens     = prev.tokens + blk.tokens,
                    y_start    = prev.y_start,
                    y_end      = blk.y_end,
                )
            else:
                result.append(blk)
        return result
