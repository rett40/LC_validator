"""
semantic_business_narrator.py
==============================
Convertit les champs JSON et les tableaux extraits en représentation textuelle
métier canonique, puis en texte narratif structuré pour l'analyse sémantique.

Chaque type de document (BL, Invoice, PL) dispose d'un schéma canonique
(SemanticDocument) et d'un générateur de texte dédié qui inclut :
  • les champs scalaires (parties, route, cargo, finance)
  • les lignes de tableau (items facture, colis, conteneurs)
  • les totaux calculés

Ce texte enrichi est ensuite utilisé par :
  • SemanticEmbedding  — similarité cosine via Sentence Transformers
  • NLIAnalyzer        — entailment / contradiction via DeBERTa
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_CANONICAL_DOC_TYPE: dict[str, str] = {
    "bl":           "bill_of_lading",
    "invoice":      "commercial_invoice",
    "packing_list": "packing_list",
}


@dataclass
class SemanticDocument:
    document_type: str
    shipment: dict[str, Any] = field(default_factory=dict)
    transport: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)
    cargo: dict[str, Any] = field(default_factory=dict)
    finance: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    raw_fields: dict[str, Any] = field(default_factory=dict)


class SemanticBusinessNarrator:
    """Convertit les extractions JSON en schéma métier canonique + texte narratif."""

    def build_semantic_document(
        self,
        doc_type: str,
        fields: dict[str, str],
        tables: dict[str, list[dict[str, str]]] | None = None,
    ) -> SemanticDocument:
        semantic_type = _CANONICAL_DOC_TYPE.get(doc_type, doc_type)
        doc = SemanticDocument(document_type=semantic_type)
        doc.raw_fields = {k: v for k, v in fields.items() if v is not None}
        doc.tables = tables or {}

        if doc_type == "bl":
            self._build_bl(doc)
        elif doc_type == "invoice":
            self._build_invoice(doc)
        elif doc_type == "packing_list":
            self._build_packing_list(doc)
        else:
            self._build_generic(doc)
        return doc

    def build_business_text(self, doc: SemanticDocument) -> str:
        if doc.document_type == "bill_of_lading":
            return self._describe_bl(doc)
        if doc.document_type == "commercial_invoice":
            return self._describe_invoice(doc)
        if doc.document_type == "packing_list":
            return self._describe_packing_list(doc)
        return self._describe_generic(doc)

    # ─────────────────────────────────────────────────────────────────────────
    # Canonisation par type
    # ─────────────────────────────────────────────────────────────────────────

    def _build_bl(self, doc: SemanticDocument) -> None:
        f = doc.raw_fields
        doc.shipment["shipper"]    = f.get("shipper") or f.get("seller")
        doc.shipment["consignee"]  = f.get("consignee")
        doc.shipment["notify"]     = f.get("notify_party")
        doc.shipment["carrier"]    = f.get("carrier") or f.get("vessel")
        doc.transport["vessel"]    = f.get("vessel")
        doc.transport["voyage"]    = f.get("voyage")
        doc.transport["container_numbers"] = self._listify(f.get("container_number"))
        doc.transport["seal_numbers"]      = self._listify(f.get("seal_number"))
        doc.transport["container_type"]    = f.get("container_type")
        doc.route["port_of_loading"]  = f.get("port_of_loading")
        doc.route["place_of_receipt"] = f.get("place_of_receipt")
        doc.route["place_of_delivery"] = f.get("place_of_delivery")
        doc.route["eta"]          = self._normalize_date(f.get("eta"))
        doc.route["issue_date"]   = self._normalize_date(f.get("issue_date"))
        doc.route["incoterm"]     = f.get("incoterm")
        doc.cargo["gross_weight_kg"] = self._parse_number(f.get("gross_weight_kg"))
        doc.cargo["net_weight_kg"]   = self._parse_number(f.get("net_weight_kg"))
        doc.cargo["goods_description"] = f.get("goods_description")
        doc.tables["containers"] = self._normalize_table_rows(
            doc.tables.get("containers", []),
            ["container_number", "seal_number", "container_type", "gross_weight_kg"],
        )

    def _build_invoice(self, doc: SemanticDocument) -> None:
        f = doc.raw_fields
        doc.shipment["seller"]         = f.get("seller")
        doc.shipment["buyer"]          = f.get("buyer")
        doc.shipment["invoice_number"] = f.get("invoice_number")
        doc.shipment["lc_reference"]   = f.get("lc_reference")
        doc.finance["currency"]       = f.get("currency")
        doc.finance["incoterm"]       = self._normalize_incoterm(f.get("incoterm"))
        doc.finance["total_amount"]   = self._parse_number(f.get("total_amount"))
        doc.finance["subtotal"]       = self._parse_number(f.get("subtotal"))
        doc.finance["payment_terms"]  = f.get("payment_terms")
        doc.tables["invoice_items"] = self._normalize_table_rows(
            doc.tables.get("invoice_items", []),
            ["quantity", "description", "hs_code", "unit", "unit_price", "line_total", "line_amount"],
        )

    def _build_packing_list(self, doc: SemanticDocument) -> None:
        f = doc.raw_fields
        doc.shipment["shipper"]   = f.get("shipper")
        doc.shipment["consignee"] = f.get("consignee")
        doc.shipment["pl_number"] = f.get("pl_number")
        doc.transport["container_numbers"] = self._listify(f.get("container_number"))
        doc.route["port_of_loading"]  = f.get("port_of_loading")
        doc.route["place_of_delivery"] = f.get("place_of_delivery")
        doc.cargo["net_weight_kg"]  = self._parse_number(f.get("net_weight_kg"))
        doc.cargo["gross_weight_kg"] = self._parse_number(f.get("gross_weight_kg"))
        doc.cargo["total_cbm"]      = self._parse_number(f.get("total_cbm"))
        doc.cargo["total_packages"] = self._parse_number(f.get("total_packages"))
        doc.tables["packages"] = self._normalize_table_rows(
            doc.tables.get("packages", []),
            ["package_type", "quantity", "unit", "gross_weight_kg", "gross_weight",
             "net_weight_kg", "net_weight", "measurement", "cbm"],
        )

    def _build_generic(self, doc: SemanticDocument) -> None:
        doc.shipment.update({
            k: v for k, v in doc.raw_fields.items()
            if k in {"shipper", "consignee", "seller", "buyer", "carrier"}
        })
        doc.route.update({
            k: v for k, v in doc.raw_fields.items()
            if k.endswith("place") or k in {"eta", "etd", "issue_date"}
        })
        doc.cargo.update({
            k: self._parse_number(v) if k.endswith("_kg") or k.endswith("_cbm") else v
            for k, v in doc.raw_fields.items()
            if k in {"gross_weight_kg", "net_weight_kg", "total_cbm"}
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Génération de texte narratif riche
    # ─────────────────────────────────────────────────────────────────────────

    def _describe_bl(self, doc: SemanticDocument) -> str:
        lines: list[str] = []
        carrier   = doc.shipment.get("carrier") or doc.transport.get("vessel")
        shipper   = doc.shipment.get("shipper")
        consignee = doc.shipment.get("consignee")
        notify    = doc.shipment.get("notify")
        voyage    = doc.transport.get("voyage")
        vessel    = doc.transport.get("vessel")
        load_port = doc.route.get("port_of_loading")
        receipt   = doc.route.get("place_of_receipt")
        delivery  = doc.route.get("place_of_delivery")
        eta       = doc.route.get("eta")
        issue_date = doc.route.get("issue_date")
        incoterm  = doc.route.get("incoterm")
        containers = doc.transport.get("container_numbers") or []
        seals      = doc.transport.get("seal_numbers") or []
        ctype      = doc.transport.get("container_type")
        gross = self._format_number(doc.cargo.get("gross_weight_kg"))
        net   = self._format_number(doc.cargo.get("net_weight_kg"))

        # Entête
        if carrier and shipper and consignee:
            lines.append(
                f"A Bill of Lading was issued by carrier {carrier} for a shipment "
                f"managed by shipper {shipper} and consigned to {consignee}."
            )
        elif shipper and consignee:
            lines.append(
                f"A Bill of Lading describes a shipment from shipper {shipper} "
                f"to consignee {consignee}."
            )
        elif shipper:
            lines.append(f"A Bill of Lading was issued for shipper {shipper}.")

        if notify:
            lines.append(f"Notify party is {notify}.")

        # Route
        route_parts: list[str] = []
        if receipt:
            route_parts.append(f"place of receipt {receipt}")
        if load_port:
            route_parts.append(f"port of loading {load_port}")
        if vessel:
            route_parts.append(f"vessel {vessel}")
        if voyage:
            route_parts.append(f"voyage {voyage}")
        if delivery:
            route_parts.append(f"place of delivery {delivery}")
        if eta:
            route_parts.append(f"ETA {eta}")
        if route_parts:
            lines.append("The shipment route is: " + "; ".join(route_parts) + ".")

        if issue_date:
            lines.append(f"Document issued on {issue_date}.")
        if incoterm:
            lines.append(f"Incoterm is {incoterm}.")

        # Conteneurs
        if containers:
            lines.append(f"The shipment includes container(s): {', '.join(containers)}.")
        if seals:
            lines.append(f"Seal number(s): {', '.join(seals)}.")
        if ctype:
            lines.append(f"Container type: {ctype}.")

        # Goods description
        goods_desc = doc.cargo.get("goods_description")
        if goods_desc:
            lines.append(f"Goods: {goods_desc}.")

        # Poids
        if gross or net:
            parts = []
            if gross:
                parts.append(f"gross weight {gross} kg")
            if net:
                parts.append(f"net weight {net} kg")
            lines.append("Cargo: " + ", ".join(parts) + ".")

        # Table conteneurs détail
        container_rows = doc.tables.get("containers", [])
        if container_rows:
            for row in container_rows[:5]:
                row_parts = []
                if row.get("container_number"):
                    row_parts.append(f"container {row['container_number']}")
                if row.get("seal_number"):
                    row_parts.append(f"seal {row['seal_number']}")
                if row.get("container_type"):
                    row_parts.append(f"type {row['container_type']}")
                gw = self._format_number(row.get("gross_weight_kg"))
                if gw:
                    row_parts.append(f"gross {gw} kg")
                if row_parts:
                    lines.append("Container detail: " + ", ".join(row_parts) + ".")

        return " ".join(filter(None, lines)).strip()

    def _describe_invoice(self, doc: SemanticDocument) -> str:
        lines: list[str] = []
        seller   = doc.shipment.get("seller")
        buyer    = doc.shipment.get("buyer")
        inv_num  = doc.shipment.get("invoice_number")
        lc_ref   = doc.shipment.get("lc_reference")
        incoterm = doc.finance.get("incoterm")
        currency = doc.finance.get("currency")
        total    = self._format_number(doc.finance.get("total_amount"))
        subtotal = self._format_number(doc.finance.get("subtotal"))
        terms    = doc.finance.get("payment_terms")

        # Entête
        if inv_num and seller:
            lines.append(f"Commercial invoice {inv_num} was issued by seller {seller}.")
        elif seller:
            lines.append(f"Commercial invoice issued by seller {seller}.")

        if buyer:
            lines.append(f"The buyer is {buyer}.")
        if lc_ref:
            lines.append(f"Letter of credit reference: {lc_ref}.")
        if incoterm and currency:
            lines.append(f"The invoice uses {incoterm} terms in {currency}.")
        elif incoterm:
            lines.append(f"Incoterm is {incoterm}.")
        if terms:
            lines.append(f"Payment terms: {terms}.")
        if total:
            cur = currency or ""
            lines.append(f"Total invoice amount is {total} {cur}.")
        elif subtotal:
            cur = currency or ""
            lines.append(f"Invoice subtotal is {subtotal} {cur}.")

        # Lignes de facture
        items = doc.tables.get("invoice_items", [])
        if items:
            item_parts: list[str] = []
            for row in items[:10]:
                desc = row.get("description") or row.get("goods_description", "")
                qty  = self._format_number(row.get("quantity"))
                unit = row.get("unit", "")
                price = self._format_number(row.get("unit_price"))
                amount = self._format_number(row.get("line_total") or row.get("line_amount"))
                hs   = row.get("hs_code", "")
                parts = []
                if desc:
                    parts.append(desc)
                if qty:
                    parts.append(f"qty {qty}")
                if unit:
                    parts.append(unit)
                if price:
                    parts.append(f"at {price} {currency or ''}")
                if amount:
                    parts.append(f"total {amount}")
                if hs:
                    parts.append(f"HS {hs}")
                if parts:
                    item_parts.append("; ".join(parts))
            if item_parts:
                lines.append(
                    f"The invoice contains {len(items)} line item(s): "
                    + " | ".join(item_parts) + "."
                )
        return " ".join(filter(None, lines)).strip()

    def _describe_packing_list(self, doc: SemanticDocument) -> str:
        lines: list[str] = []
        pl_number  = doc.shipment.get("pl_number")
        shipper    = doc.shipment.get("shipper")
        consignee  = doc.shipment.get("consignee")
        containers = doc.transport.get("container_numbers") or []
        load_port  = doc.route.get("port_of_loading")
        delivery   = doc.route.get("place_of_delivery")
        total_pkg  = self._format_number(doc.cargo.get("total_packages"))
        net        = self._format_number(doc.cargo.get("net_weight_kg"))
        gross      = self._format_number(doc.cargo.get("gross_weight_kg"))
        cbm        = self._format_number(doc.cargo.get("total_cbm"))

        # Entête
        if pl_number:
            lines.append(f"Packing list {pl_number} details cargo packed for shipment.")
        if shipper:
            lines.append(f"Shipper (exporter) is {shipper}.")
        if consignee:
            lines.append(f"Consignee is {consignee}.")

        # Conteneurs
        if containers:
            lines.append(f"Container(s): {', '.join(containers)}.")

        # Route
        if load_port or delivery:
            rp = []
            if load_port:
                rp.append(f"port of loading {load_port}")
            if delivery:
                rp.append(f"place of delivery {delivery}")
            lines.append("Route: " + ", ".join(rp) + ".")

        # Totaux
        summary: list[str] = []
        if total_pkg:
            summary.append(f"{total_pkg} packages")
        if net:
            summary.append(f"total net weight {net} kg")
        if gross:
            summary.append(f"total gross weight {gross} kg")
        if cbm:
            summary.append(f"total volume {cbm} CBM")
        if summary:
            lines.append("Summary: " + ", ".join(summary) + ".")

        # Lignes de colis
        packages = doc.tables.get("packages", [])
        if packages:
            pkg_parts: list[str] = []
            for row in packages[:10]:
                desc = row.get("package_type") or row.get("goods_description", "")
                qty  = self._format_number(row.get("quantity") or row.get("number_of_packages"))
                unit = row.get("unit", "")
                gw   = self._format_number(row.get("gross_weight_kg") or row.get("gross_weight"))
                nw   = self._format_number(row.get("net_weight_kg") or row.get("net_weight"))
                cb   = self._format_number(row.get("cbm") or row.get("measurement"))
                parts = []
                if desc:
                    parts.append(desc)
                if qty:
                    parts.append(f"qty {qty}")
                if unit:
                    parts.append(unit)
                if gw:
                    parts.append(f"gross {gw} kg")
                if nw:
                    parts.append(f"net {nw} kg")
                if cb:
                    parts.append(f"CBM {cb}")
                if parts:
                    pkg_parts.append("; ".join(parts))
            if pkg_parts:
                lines.append(
                    f"The packing list contains {len(packages)} line(s): "
                    + " | ".join(pkg_parts) + "."
                )
        return " ".join(filter(None, lines)).strip()

    def _describe_generic(self, doc: SemanticDocument) -> str:
        parts: list[str] = []
        if doc.document_type:
            parts.append(f"Document type is {doc.document_type}.")
        if doc.shipment.get("shipper"):
            parts.append(f"Shipper is {doc.shipment['shipper']}.")
        if doc.shipment.get("consignee"):
            parts.append(f"Consignee is {doc.shipment['consignee']}.")
        for k, v in doc.cargo.items():
            if v is not None:
                parts.append(f"{k} is {v}.")
        return " ".join(parts).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_incoterm(self, value: Optional[str]) -> Optional[str]:
        if not value or not isinstance(value, str):
            return None
        value = value.strip().upper()
        return value if re.fullmatch(r"[A-Z]{2,3}", value) else value

    def _normalize_date(self, value: Optional[str]) -> Optional[str]:
        if not value or not isinstance(value, str):
            return None
        return re.sub(r"[^0-9A-Z\-:/ ]+", " ", value.upper()).strip()

    def _parse_number(self, value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().replace(" ", "").replace(",", ".")
        try:
            return float(text)
        except ValueError:
            try:
                return float(re.sub(r"[^0-9.\-]", "", text))
            except ValueError:
                return None

    def _format_number(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            f = float(value)
            if f.is_integer():
                return str(int(f))
            return f"{f:.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(value)

    def _listify(self, value: Optional[str]) -> list[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if v]
        return [t.strip() for t in re.split(r"[,;/\\]|and", str(value)) if t.strip()]

    def _normalize_table_rows(
        self,
        rows: list[dict[str, Any]],
        allowed_keys: list[str],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cleaned: dict[str, Any] = {}
            for key in allowed_keys:
                value = row.get(key)
                if value is None:
                    continue
                is_numeric_key = (
                    key.endswith("_kg") or key in {
                        "quantity", "line_total", "line_amount",
                        "unit_price", "cbm", "measurement",
                        "gross_weight", "net_weight",
                    }
                )
                if is_numeric_key:
                    parsed = self._parse_number(value)
                    cleaned[key] = parsed if parsed is not None else str(value).strip()
                else:
                    cleaned[key] = str(value).strip()
            if cleaned:
                normalized.append(cleaned)
        return normalized
