"""
field_router.py
================
Per-pair field routing for cross-validation.

In international trade, BL / Invoice / Packing List are COMPLEMENTARY documents
with distinct scopes. Comparing them on irrelevant fields generates false
CONTRADICTION signals in NLI. PAIR_FIELDS defines which fields are structurally
relevant for each document pair so only those fields are injected into scoring.
"""
from __future__ import annotations
from typing import Any

PAIR_FIELDS: dict[str, dict[str, list[str]]] = {
    "bl↔invoice": {
        "bl": [
            "shipper", "consignee", "incoterm", "currency",
            "port_of_loading", "port_of_discharge", "vessel_name",
            "carrier", "bl_number",
        ],
        "invoice": [
            "seller", "buyer", "incoterm", "currency", "payment_terms",
            "total_amount", "invoice_date", "port_of_destination",
        ],
    },
    "invoice↔packing_list": {
        "invoice": [
            "seller", "total_amount", "currency",
            "goods_description", "net_weight_kg",
        ],
        "packing_list": [
            "shipper", "total_gross_kg", "total_net_kg", "total_cbm",
            "goods_description", "net_weight_kg",
        ],
    },
    "bl↔packing_list": {
        "bl": [
            "shipper", "container_number", "seal_number",
            "gross_weight_kg", "net_weight_kg", "vessel_name", "port_of_discharge",
        ],
        "packing_list": [
            "shipper", "container_number", "seal_number",
            "gross_weight_kg", "net_weight_kg",
        ],
    },
}

# NLI hypothesis categories relevant per pair
PAIR_NLI_HINTS: dict[str, list[str]] = {
    "bl↔invoice":           ["parties", "incoterm", "currency", "reference"],
    "invoice↔packing_list": ["weight"],
    "bl↔packing_list":      ["container", "weight", "parties"],
}


def build_pair_context(
    pair: str,
    docs_by_type: dict[str, dict[str, Any]],
    tables: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Returns a routing context for the given pair with only structurally relevant fields.
    Used by NLI and semantic layers to avoid false CONTRADICTION from irrelevant sections.
    """
    routing = PAIR_FIELDS.get(pair, {})
    doc_keys = list(routing.keys())

    if len(doc_keys) != 2:
        return {}

    left_type, right_type = doc_keys

    def _filter(doc_type: str, allowed: list[str]) -> dict[str, Any]:
        raw = docs_by_type.get(doc_type, {})
        return {k: v for k, v in raw.items() if k in allowed and v not in (None, "", [])}

    return {
        "pair":         pair,
        "left_type":    left_type,
        "right_type":   right_type,
        "left_fields":  _filter(left_type, routing[left_type]),
        "right_fields": _filter(right_type, routing[right_type]),
        "left_tables":  tables.get(left_type, {}),
        "right_tables": tables.get(right_type, {}),
        "nli_hints":    PAIR_NLI_HINTS.get(pair, []),
    }
