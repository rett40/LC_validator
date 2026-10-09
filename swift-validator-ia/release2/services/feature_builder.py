from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np
from PIL import Image


class FeatureBuilder:
    """Generateur de features multimodales pour extraction et classification."""

    KEYWORD_GROUPS: dict[str, list[str]] = {
        "bl": ["bill of lading", "bl", "shipper", "consignee", "vessel", "cargo", "port"],
        "invoice": ["invoice", "inv", "amount", "total", "tax", "payment", "due date", "bill"],
        "packing_list": ["packing list", "pl", "item", "quantity", "description", "weight", "volume"],
        "table": ["qty", "quantity", "unit price", "total", "amount", "description", "item", "pcs", "weight"],
    }

    @classmethod
    def build_multimodal_features(
        cls,
        image: Image.Image,
        words: list[str],
        bboxes: list[list[int]],
        doc_type: str = "unknown",
    ) -> dict[str, Any]:
        text = " ".join(words).lower()
        n_tokens = len(words)
        token_lines, avg_tokens_per_line = cls._estimate_text_lines(bboxes)
        width, height = image.size
        avg_width, avg_height = cls._bbox_size_stats(bboxes)
        table_keyword_count = cls._count_keywords(text, cls.KEYWORD_GROUPS["table"])

        return {
            "doc_type_hint": 0 if doc_type == "unknown" else 1,
            "n_tokens": n_tokens,
            "n_ocr_lines": token_lines,
            "avg_tokens_per_line": round(avg_tokens_per_line, 2),
            "avg_bbox_width": round(avg_width, 2),
            "avg_bbox_height": round(avg_height, 2),
            "token_density": round(n_tokens / max(1, width * height) * 1_000_000, 4),
            "table_keyword_count": table_keyword_count,
            "table_keyword_ratio": round(table_keyword_count / max(1, n_tokens), 4),
            "bl_keyword_ratio": round(cls._count_keywords(text, cls.KEYWORD_GROUPS["bl"]) / max(1, n_tokens), 4),
            "invoice_keyword_ratio": round(cls._count_keywords(text, cls.KEYWORD_GROUPS["invoice"]) / max(1, n_tokens), 4),
            "packing_keyword_ratio": round(cls._count_keywords(text, cls.KEYWORD_GROUPS["packing_list"]) / max(1, n_tokens), 4),
            "layout_balance": round(cls._estimate_layout_balance(bboxes), 4),
        }

    @classmethod
    def _count_keywords(cls, text: str, keywords: Iterable[str]) -> int:
        return sum(text.count(keyword) for keyword in keywords)

    @classmethod
    def _estimate_text_lines(cls, bboxes: list[list[int]]) -> tuple[int, float]:
        if not bboxes:
            return 0, 0.0

        y_centers = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]
        heights = [bbox[3] - bbox[1] for bbox in bboxes if bbox[3] > bbox[1]]
        line_threshold = max(20.0, np.median(heights) * 1.25 if heights else 30.0)

        line_groups: list[list[float]] = []
        for y in sorted(y_centers):
            if not line_groups or abs(line_groups[-1][-1] - y) > line_threshold:
                line_groups.append([y])
            else:
                line_groups[-1].append(y)

        n_lines = len(line_groups)
        avg_tokens_per_line = len(y_centers) / max(1, n_lines)
        return n_lines, avg_tokens_per_line

    @classmethod
    def _bbox_size_stats(cls, bboxes: list[list[int]]) -> tuple[float, float]:
        if not bboxes:
            return 0.0, 0.0

        widths = [bbox[2] - bbox[0] for bbox in bboxes]
        heights = [bbox[3] - bbox[1] for bbox in bboxes]
        return float(np.mean(widths)), float(np.mean(heights))

    @classmethod
    def _estimate_layout_balance(cls, bboxes: list[list[int]]) -> float:
        if not bboxes:
            return 0.0

        x_centers = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
        left_ratio = sum(1 for x in x_centers if x < 500) / max(1, len(x_centers))
        return round(abs(left_ratio - 0.5), 4)
