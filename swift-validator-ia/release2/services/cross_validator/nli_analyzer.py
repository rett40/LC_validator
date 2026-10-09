"""
nli_analyzer.py
================
Analyse NLI (Natural Language Inference) inter-documents.

Utilise DeBERTa-v3-base-mnli (ou fallback heuristique) pour classer
chaque paire (premise, hypothesis) en :
  ENTAILMENT   — le document A implique le document B
  NEUTRAL      — pas de lien détectable
  CONTRADICTION — les documents se contredisent

Les paires NLI sont construites au niveau grain fin :
  • comparaison globale des narratifs métier
  • comparaison shipper ↔ seller
  • comparaison biens (Invoice ↔ PL)
  • comparaison route / conteneur
  • comparaison poids
  • comparaison référence (voyage ↔ LC ref)
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    hf_pipeline = None
    _HF_AVAILABLE = False


class NLIAnalyzer:
    """NLI multi-modèles avec fallback heuristique robuste."""

    def __init__(self, model_name: str | None = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model_name = model_name
        self.pipeline   = None

        if not model_name:
            log.info("NLIAnalyzer: désactivé")
            return

        if _HF_AVAILABLE:
            for candidate in (model_name, "cross-encoder/nli-deberta-v3-small",
                               "facebook/bart-large-mnli"):
                try:
                    # Registre central : révision épinglée + device CPU (parité prod/local)
                    try:
                        from config.model_registry import hf_pipeline_kwargs
                        _kwargs = hf_pipeline_kwargs(candidate)
                    except ImportError:
                        _kwargs = {"device": -1}
                    self.pipeline = hf_pipeline(
                        "text-classification",
                        model=candidate,
                        return_all_scores=True,
                        **_kwargs,
                    )
                    log.info("NLIAnalyzer: modèle chargé — %s", candidate)
                    self.model_name = candidate
                    break
                except Exception as exc:
                    log.warning("NLIAnalyzer: impossible de charger %s : %s", candidate, exc)

        if self.pipeline is None:
            log.info("NLIAnalyzer: fallback heuristique activé (aucun modèle disponible)")

    def available(self) -> bool:
        return True  # toujours disponible grâce au fallback heuristique

    def predict(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """Prédit le label NLI + score de confiance."""
        if self.pipeline is not None:
            try:
                # Cross-encoder models need text_pair format
                # MNLI-style models (bart-large-mnli, roberta) accept concatenated strings
                # Use text_pair format universally — works for both architectures
                inp = {"text": str(premise)[:500], "text_pair": str(hypothesis)[:500]}

                # top_k=None replaced deprecated return_all_scores=True (transformers 4.45+)
                try:
                    raw = self.pipeline(inp, top_k=None, truncation=True)
                except TypeError:
                    raw = self.pipeline(inp, return_all_scores=True, truncation=True)

                # Normalize nested list [[...]] or flat list [...]
                if raw and isinstance(raw, list):
                    items = raw[0] if isinstance(raw[0], list) else raw
                    scores: dict[str, float] = {}
                    for item in items:
                        lbl = self._normalize_label(item["label"])
                        scores[lbl] = max(scores.get(lbl, 0.0), float(item["score"]))
                    if scores:
                        label = max(scores, key=scores.get)
                        return {
                            "label": label,
                            "score": round(scores[label], 4),
                            "all_scores": {k: round(v, 4) for k, v in scores.items()},
                        }
            except Exception as exc:
                log.warning("NLIAnalyzer: erreur prediction : %s", exc)

        # Fallback heuristique
        return self._heuristic_predict(premise, hypothesis)

    def build_nli_pairs(
        self,
        entries: list[dict[str, Any]],
    ) -> list[tuple[str, str, str, str]]:
        """
        Construit les paires NLI (premise, hypothesis, description, category).

        Principe : BL, Invoice et PL sont des documents COMPLÉMENTAIRES — ils décrivent
        des aspects différents d'une même opération commerciale. Les comparer en texte
        complet génère des faux positifs CONTRADICTION (un BL ne parle pas de prix, une
        facture ne parle pas de route, etc.). On construit donc des paires au grain fin,
        ciblées sur les champs qui DOIVENT être cohérents entre deux documents.
        """
        pairs: list[tuple[str, str, str, str]] = []
        by_type = {e["doc_type"]: e for e in entries}

        inv = by_type.get("invoice")
        bl  = by_type.get("bl")
        pl  = by_type.get("packing_list")

        # ── Invoice ↔ BL — champs complémentaires ──────────────────────────
        if inv and bl:
            # Seller (Invoice) ↔ Shipper (BL)
            seller  = self._extract_field(inv, "seller", "shipment")
            shipper = self._extract_field(bl, "shipper", "shipment")
            if seller and shipper:
                pairs.append((
                    f"The seller of the goods is {seller}.",
                    f"The shipper on the bill of lading is {shipper}.",
                    "Seller (Invoice) matches Shipper (BL)",
                    "parties",
                ))

            # Buyer (Invoice) ↔ Consignee (BL)
            buyer     = self._extract_field(inv, "buyer", "shipment")
            consignee = self._extract_field(bl, "consignee", "shipment")
            if buyer and consignee:
                pairs.append((
                    f"The buyer is {buyer}.",
                    f"The consignee on the bill of lading is {consignee}.",
                    "Buyer (Invoice) matches Consignee (BL)",
                    "parties",
                ))

            # Incoterm
            inv_inc = self._extract_field(inv, "incoterm", "finance")
            bl_inc  = self._extract_field(bl, "incoterm", "route")
            if inv_inc and bl_inc:
                pairs.append((
                    f"The commercial invoice uses {inv_inc} incoterm.",
                    f"The bill of lading specifies {bl_inc} incoterm.",
                    "Incoterm consistency Invoice vs BL",
                    "incoterm",
                ))

            # LC reference ↔ voyage BL
            lc_ref = self._extract_field(inv, "lc_reference", "shipment")
            voyage = self._extract_field(bl, "voyage", "transport")
            if lc_ref and voyage:
                pairs.append((
                    f"The letter of credit reference is {lc_ref}.",
                    f"The bill of lading voyage number is {voyage}.",
                    "LC reference contains BL voyage reference",
                    "reference",
                ))

            # Goods: 3 premiers items facture ↔ description cargo BL
            inv_items = (inv.get("tables") or {}).get("invoice_items", [])
            bl_goods  = self._extract_field(bl, "goods_description", "cargo")
            if inv_items and bl_goods:
                items_summary = "; ".join(
                    str(r.get("description") or r.get("goods_description") or "")
                    for r in inv_items[:3]
                    if r.get("description") or r.get("goods_description")
                )
                if items_summary:
                    pairs.append((
                        f"The invoice covers the following goods: {items_summary}.",
                        f"The BL cargo description is: {bl_goods}.",
                        "Invoice goods consistent with BL cargo description",
                        "goods",
                    ))

        # ── Invoice ↔ PL — biens et poids ──────────────────────────────────
        if inv and pl:
            # Marchandises (textes métier restent pertinents ici car même sujet)
            pairs.append((
                inv["business_text"],
                pl["business_text"],
                "Invoice goods match packing list contents",
                "goods",
            ))

            # Poids brut
            inv_gw = self._extract_cargo(inv, "gross_weight_kg")
            pl_gw  = self._extract_cargo(pl, "gross_weight_kg")
            if inv_gw and pl_gw:
                pairs.append((
                    f"The invoice declares a total gross weight of {inv_gw} kg.",
                    f"The packing list total gross weight is {pl_gw} kg.",
                    "Gross weight consistency Invoice vs PL",
                    "weight",
                ))

        # ── BL ↔ PL — conteneurs, poids et parties ────────────────────────
        if bl and pl:
            # Conteneur
            bl_cnts = " and ".join(self._extract_field(bl, "container_numbers", "transport") or [])
            pl_cnts = " and ".join(self._extract_field(pl, "container_numbers", "transport") or [])
            if bl_cnts and pl_cnts:
                pairs.append((
                    f"The bill of lading covers container(s) {bl_cnts}.",
                    f"The packing list is for container(s) {pl_cnts}.",
                    "Container number consistency BL vs PL",
                    "container",
                ))

            # Poids brut
            bl_gw = self._extract_cargo(bl, "gross_weight_kg")
            pl_gw = self._extract_cargo(pl, "gross_weight_kg")
            if bl_gw and pl_gw:
                pairs.append((
                    f"The bill of lading declares gross weight {bl_gw} kg.",
                    f"The packing list total gross weight is {pl_gw} kg.",
                    "Gross weight consistency BL vs PL",
                    "weight",
                ))

            # Poids net
            bl_nw = self._extract_cargo(bl, "net_weight_kg")
            pl_nw = self._extract_cargo(pl, "net_weight_kg")
            if bl_nw and pl_nw:
                pairs.append((
                    f"The bill of lading net weight is {bl_nw} kg.",
                    f"The packing list net weight is {pl_nw} kg.",
                    "Net weight consistency BL vs PL",
                    "weight",
                ))

            # Shipper (BL) ↔ Exporter (PL)
            bl_shipper = self._extract_field(bl, "shipper", "shipment")
            pl_shipper = self._extract_field(pl, "shipper", "shipment")
            if bl_shipper and pl_shipper:
                bl_clean = bl_shipper.split("|")[0].strip()
                pl_clean = pl_shipper.split("|")[0].strip()
                pairs.append((
                    f"The BL shipper is {bl_clean}.",
                    f"The packing list exporter is {pl_clean}.",
                    "Shipper (BL) matches Exporter (PL)",
                    "parties",
                ))

        return pairs

    # ─────────────────────────────────────────────────────────────────────────
    # Per-pair NLI evaluation (field-router aware)
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_pair(
        self,
        pair: str,
        ctx: dict[str, Any],
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Per-pair NLI evaluation using only fields from the routing context.
        Replaces build_nli_pairs() for the main validation flow to avoid
        false CONTRADICTION signals from irrelevant document sections.
        """
        by_type = {e["doc_type"]: e for e in entries}
        nli_pairs = self._build_pair_hypotheses(pair, ctx, by_type)
        results: list[dict[str, Any]] = []
        for premise, hypothesis, description, category in nli_pairs:
            prediction = self.predict(premise, hypothesis)
            results.append({
                "description": description,
                "category":    category,
                "premise":     premise[:300],
                "hypothesis":  hypothesis[:300],
                "nli_label":   prediction.get("label"),
                "score":       prediction.get("score"),
                "all_scores":  prediction.get("all_scores", {}),
            })
        return results

    def _build_pair_hypotheses(
        self,
        pair: str,
        ctx: dict[str, Any],
        by_type: dict[str, Any],
    ) -> list[tuple[str, str, str, str]]:
        """Calibrated NLI hypotheses for a specific document pair."""
        hints = ctx.get("nli_hints", [])
        pairs: list[tuple[str, str, str, str]] = []

        if pair == "bl↔invoice":
            bl  = by_type.get("bl")
            inv = by_type.get("invoice")
            if not bl or not inv:
                return []

            if "parties" in hints:
                seller  = self._extract_field(inv, "seller", "shipment")
                shipper = self._extract_field(bl, "shipper", "shipment")
                if seller and shipper:
                    pairs.append((
                        f"The seller of the goods is {seller}.",
                        f"The shipper on the bill of lading is {shipper}.",
                        "Seller (Invoice) matches Shipper (BL)",
                        "parties",
                    ))
                buyer     = self._extract_field(inv, "buyer", "shipment")
                consignee = self._extract_field(bl, "consignee", "shipment")
                if buyer and consignee:
                    pairs.append((
                        f"The buyer is {buyer}.",
                        f"The consignee on the bill of lading is {consignee}.",
                        "Buyer (Invoice) matches Consignee (BL)",
                        "parties",
                    ))

            if "incoterm" in hints:
                inv_inc = self._extract_field(inv, "incoterm", "finance")
                bl_inc  = self._extract_field(bl, "incoterm", "route")
                if inv_inc and bl_inc:
                    pairs.append((
                        f"The commercial invoice uses {inv_inc} incoterm.",
                        f"The bill of lading specifies {bl_inc} incoterm.",
                        "Incoterm consistency Invoice vs BL",
                        "incoterm",
                    ))

            if "currency" in hints:
                inv_cur = self._extract_field(inv, "currency", "finance")
                bl_cur  = self._extract_field(bl, "currency", "finance")
                if inv_cur and bl_cur:
                    pairs.append((
                        f"The invoice currency is {inv_cur}.",
                        f"The bill of lading currency is {bl_cur}.",
                        "Currency consistency Invoice vs BL",
                        "currency",
                    ))

            if "reference" in hints:
                lc_ref = self._extract_field(inv, "lc_reference", "shipment")
                voyage = self._extract_field(bl, "voyage", "transport")
                if lc_ref and voyage:
                    pairs.append((
                        f"The letter of credit reference is {lc_ref}.",
                        f"The bill of lading voyage number is {voyage}.",
                        "LC reference contains BL voyage reference",
                        "reference",
                    ))

            # Goods: targeted items summary only, not full business_text
            inv_items = (inv.get("tables") or {}).get("invoice_items", [])
            bl_goods  = self._extract_field(bl, "goods_description", "cargo")
            if inv_items and bl_goods:
                items_summary = "; ".join(
                    str(r.get("description") or r.get("goods_description") or "")
                    for r in inv_items[:3]
                    if r.get("description") or r.get("goods_description")
                )
                if items_summary:
                    pairs.append((
                        f"The invoice covers: {items_summary}.",
                        f"The BL cargo description is: {bl_goods}.",
                        "Invoice goods consistent with BL cargo",
                        "goods",
                    ))

        elif pair == "invoice↔packing_list":
            inv = by_type.get("invoice")
            pl  = by_type.get("packing_list")
            if not inv or not pl:
                return []

            # Targeted item summary — NOT full business_text (avoids false CONTRADICTION)
            if "goods" in hints:
                inv_items = (inv.get("tables") or {}).get("invoice_items", [])
                pl_items  = (pl.get("tables") or {}).get("packages", [])
                if inv_items and pl_items:
                    inv_summary = "; ".join(
                        str(r.get("description") or r.get("goods_description") or "")
                        for r in inv_items[:3]
                        if r.get("description") or r.get("goods_description")
                    )
                    pl_summary = "; ".join(
                        str(r.get("package_type") or r.get("goods_description") or "")
                        for r in pl_items[:3]
                        if r.get("package_type") or r.get("goods_description")
                    )
                    if inv_summary and pl_summary:
                        pairs.append((
                            f"The invoice covers: {inv_summary}.",
                            f"The packing list contains: {pl_summary}.",
                            "Invoice goods match packing list contents",
                            "goods",
                        ))

            if "weight" in hints:
                inv_gw = self._extract_cargo(inv, "gross_weight_kg")
                pl_gw  = self._extract_cargo(pl, "gross_weight_kg")
                if inv_gw and pl_gw:
                    pairs.append((
                        f"The invoice declares a total gross weight of {inv_gw} kg.",
                        f"The packing list total gross weight is {pl_gw} kg.",
                        "Gross weight consistency Invoice vs PL",
                        "weight",
                    ))

                inv_nw = self._extract_cargo(inv, "net_weight_kg")
                pl_nw  = self._extract_cargo(pl, "net_weight_kg")
                if inv_nw and pl_nw:
                    pairs.append((
                        f"The invoice net weight is {inv_nw} kg.",
                        f"The packing list net weight is {pl_nw} kg.",
                        "Net weight consistency Invoice vs PL",
                        "weight",
                    ))

        elif pair == "bl↔packing_list":
            bl = by_type.get("bl")
            pl = by_type.get("packing_list")
            if not bl or not pl:
                return []

            if "container" in hints:
                bl_cnts = " and ".join(
                    self._extract_field(bl, "container_numbers", "transport") or []
                )
                pl_cnts = " and ".join(
                    self._extract_field(pl, "container_numbers", "transport") or []
                )
                if bl_cnts and pl_cnts:
                    pairs.append((
                        f"The bill of lading covers container(s) {bl_cnts}.",
                        f"The packing list is for container(s) {pl_cnts}.",
                        "Container number consistency BL vs PL",
                        "container",
                    ))

            if "weight" in hints:
                bl_gw = self._extract_cargo(bl, "gross_weight_kg")
                pl_gw = self._extract_cargo(pl, "gross_weight_kg")
                if bl_gw and pl_gw:
                    pairs.append((
                        f"The bill of lading declares gross weight {bl_gw} kg.",
                        f"The packing list total gross weight is {pl_gw} kg.",
                        "Gross weight consistency BL vs PL",
                        "weight",
                    ))

                bl_nw = self._extract_cargo(bl, "net_weight_kg")
                pl_nw = self._extract_cargo(pl, "net_weight_kg")
                if bl_nw and pl_nw:
                    pairs.append((
                        f"The bill of lading net weight is {bl_nw} kg.",
                        f"The packing list net weight is {pl_nw} kg.",
                        "Net weight consistency BL vs PL",
                        "weight",
                    ))

            if "parties" in hints:
                bl_shipper = self._extract_field(bl, "shipper", "shipment")
                pl_shipper = self._extract_field(pl, "shipper", "shipment")
                if bl_shipper and pl_shipper:
                    bl_clean = bl_shipper.split("|")[0].strip()
                    pl_clean = pl_shipper.split("|")[0].strip()
                    pairs.append((
                        f"The BL shipper is {bl_clean}.",
                        f"The packing list exporter is {pl_clean}.",
                        "Shipper (BL) matches Exporter (PL)",
                        "parties",
                    ))

        return pairs

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_label(self, label: str) -> str:
        label = label.upper().replace("-", "_")
        if "ENTAIL" in label:
            return "ENTAILMENT"
        if "CONTRA" in label:
            return "CONTRADICTION"
        return "NEUTRAL"

    def _extract_field(
        self,
        entry: dict[str, Any],
        field_name: str,
        category: str,
    ) -> Any:
        sem_doc = entry.get("semantic_doc", {})
        if isinstance(sem_doc, dict):
            return sem_doc.get(category, {}).get(field_name)
        try:
            category_obj = getattr(sem_doc, category, None)
            # SemanticDocument stores its sections as plain dicts, not objects
            if isinstance(category_obj, dict):
                return category_obj.get(field_name)
            return getattr(category_obj, field_name, None)
        except Exception:
            return None

    def _extract_cargo(self, entry: dict[str, Any], field_name: str) -> str | None:
        val = self._extract_field(entry, field_name, "cargo")
        if val is None:
            return None
        try:
            f = float(val)
            return f"{f:.0f}" if f.is_integer() else f"{f:.2f}"
        except (TypeError, ValueError):
            return str(val).strip() or None

    def _heuristic_predict(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """Heuristique lexicale robuste en absence de modèle NLI."""
        p_tokens = set(self._tokenize(premise))
        h_tokens = set(self._tokenize(hypothesis))

        if not p_tokens or not h_tokens:
            return {"label": "NEUTRAL", "score": 0.5, "all_scores": {"NEUTRAL": 0.5}}

        overlap = p_tokens & h_tokens
        overlap_ratio = len(overlap) / max(len(h_tokens), 1)

        # Détection de contradiction sur valeurs numériques
        p_nums = self._extract_numbers(premise)
        h_nums = self._extract_numbers(hypothesis)
        if p_nums and h_nums:
            matched = any(
                abs(pn - hn) / max(abs(pn), abs(hn), 1) < 0.05
                for pn in p_nums
                for hn in h_nums
            )
            contradicted = all(
                abs(pn - hn) / max(abs(pn), abs(hn), 1) > 0.20
                for pn in p_nums
                for hn in h_nums
            ) and len(p_nums) > 0 and len(h_nums) > 0

            if contradicted:
                return {
                    "label": "CONTRADICTION",
                    "score": 0.75,
                    "all_scores": {"ENTAILMENT": 0.10, "NEUTRAL": 0.15, "CONTRADICTION": 0.75},
                }
            if matched:
                overlap_ratio = max(overlap_ratio, 0.70)

        if overlap_ratio >= 0.60:
            conf = round(min(0.95, 0.60 + overlap_ratio * 0.35), 3)
            return {
                "label": "ENTAILMENT",
                "score": conf,
                "all_scores": {"ENTAILMENT": conf, "NEUTRAL": round(1 - conf, 3), "CONTRADICTION": 0.01},
            }

        if overlap_ratio >= 0.30:
            return {
                "label": "NEUTRAL",
                "score": 0.60,
                "all_scores": {"ENTAILMENT": 0.30, "NEUTRAL": 0.60, "CONTRADICTION": 0.10},
            }

        return {
            "label": "NEUTRAL",
            "score": 0.55,
            "all_scores": {"ENTAILMENT": 0.20, "NEUTRAL": 0.55, "CONTRADICTION": 0.25},
        }

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^A-Za-z0-9]+", " ", (text or "")).lower().split()

    def _extract_numbers(self, text: str) -> list[float]:
        nums: list[float] = []
        for m in re.finditer(r"\d[\d,\.]*", text or ""):
            clean = m.group().replace(",", "")
            try:
                nums.append(float(clean))
            except ValueError:
                pass
        return nums
