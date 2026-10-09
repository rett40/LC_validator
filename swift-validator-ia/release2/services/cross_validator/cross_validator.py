"""
cross_validator.py
==================
Orchestrateur de validation croisée inter-documents (BL + Invoice + Packing List).

Architecture :
  1. Normalisation sémantique   — SemanticBusinessNarrator
  2. Embedding sémantique       — SemanticEmbedding (MiniLM / TF-IDF)
  3. NLI par paires             — NLIAnalyzer (DeBERTa / heuristique)
  4. Règles métier inter-docs   — RulesEngine
  5. Alignement tables          — matching lignes Invoice ↔ PL
  6. Scoring pondéré            — score final + décomposition
  7. XAI explicatif             — explications par type + sévérité

Score final :
  semantic * 0.15 + nli * 0.20 + rules * 0.50 + party * 0.15 + table * 0.00

Note sémantique :
  Le score sémantique agrégé ne s'appuie PAS sur la similarité des `business_text`
  entiers (documents complémentaires → similarité globale faible et non pertinente),
  mais sur les comparaisons ALIGNÉES PAR SECTION (parties + cargo), qui mesurent une
  vraie cohérence inter-documents. Les similarités globales restent calculées et
  exposées à titre informatif (monitoring), sans pénaliser le score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .field_router import build_pair_context
from .nli_analyzer import NLIAnalyzer
from .semantic_business_narrator import SemanticBusinessNarrator, SemanticDocument
from .semantic_embedding import SemanticEmbedding
from .rules_engine import RuleResult, RulesEngine

log = logging.getLogger(__name__)

_SCORE_WEIGHTS = {
    "semantic": 0.15,   # cohérence sémantique alignée par section (parties + cargo)
    "nli":      0.20,
    "rules":    0.50,
    "table":    0.00,   # déjà reflété dans les règles cross_* — évite le double comptage
    "party":    0.15,
}

_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}

# Arbitration: maps NLI category → rule names that cover the same business fact
_NLI_TO_RULES: dict[str, list[str]] = {
    "incoterm":   ["cross_incoterm_consistency"],
    "weight":     ["cross_weight_tolerance", "cross_net_weight", "cross_weight_inv_pl_net"],
    "container":  ["cross_container_number"],
    "parties":    ["cross_shipper_seller", "cross_consignee_buyer"],
    "currency":   ["cross_currency"],
    "goods":      ["cross_goods_description"],
    "reference":  ["cross_voyage_lc_ref"],
}


@dataclass
class CrossValidationReport:
    rule_results: list[dict[str, Any]] = field(default_factory=list)
    semantic_scores: dict[str, float] = field(default_factory=dict)
    nli_results: list[dict[str, Any]] = field(default_factory=list)
    table_matches: list[dict[str, Any]] = field(default_factory=list)
    table_analysis: dict[str, Any] = field(default_factory=dict)
    explanations: list[dict[str, Any]] = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    risk_level: str = "unknown"
    documents: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_results":    self.rule_results,
            "semantic_scores": self.semantic_scores,
            "nli_results":     self.nli_results,
            "table_matches":   self.table_matches,
            "table_analysis":  self.table_analysis,
            "explanations":    self.explanations,
            "score_breakdown": self.score_breakdown,
            "final_score":     self.final_score,
            "risk_level":      self.risk_level,
            "documents":       self.documents,
        }


class CrossValidator:
    def __init__(
        self,
        enable_semantic: bool = True,
        enable_nli: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
    ) -> None:
        self.rules_engine = RulesEngine()
        self.narrator     = SemanticBusinessNarrator()
        self.embedder     = SemanticEmbedding(embedding_model if enable_semantic else None)
        self.nli_analyzer = NLIAnalyzer(nli_model if enable_nli else None)

    # ─────────────────────────────────────────────────────────────────────────
    # API publique
    # ─────────────────────────────────────────────────────────────────────────

    def validate_documents(
        self,
        docs: list[tuple[str, dict[str, str]]],
        tables: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tables = tables or {}
        docs_by_type = {dt: f for dt, f in docs}
        entries = self._build_document_entries(docs, tables)
        report  = CrossValidationReport()

        # 1 — Règles métier (per-pair routing)
        all_rule_results: list[RuleResult] = []
        for pair in ("bl↔invoice", "invoice↔packing_list", "bl↔packing_list"):
            all_rule_results += self.rules_engine.validate_pair(pair, docs_by_type, tables)
        report.rule_results = [r.to_dict() for r in all_rule_results]

        # 2 — Sémantique (monitoring — poids=0.00)
        report.semantic_scores = self._evaluate_semantic(entries)

        # 3 — NLI par paires (per-pair routing, champs ciblés uniquement)
        report.nli_results = []
        for pair in ("bl↔invoice", "invoice↔packing_list", "bl↔packing_list"):
            ctx = build_pair_context(pair, docs_by_type, tables)
            report.nli_results += self.nli_analyzer.evaluate_pair(pair, ctx, entries)

        # 3b — Arbitration : règles déterministes corrigent les artefacts NLI
        report.nli_results = self._arbitrate_nli(report.nli_results, report.rule_results)

        # 4 — Alignement tables (matching strict pour compatibilité)
        report.table_matches = self._evaluate_tables(entries)

        # 4b — Analyse croisée des tableaux (conteneurs, marchandises, parties)
        report.table_analysis = self._build_table_analysis(entries)

        # 5 — Documents (textes métier)
        report.documents = [
            {
                "doc_type":      entry["doc_type"],
                "business_text": entry["business_text"],
                "semantic_doc":  entry["semantic_doc"].__dict__,
            }
            for entry in entries
        ]

        # 6 — Score final + décomposition
        report.score_breakdown, report.final_score = self._compute_final_score(
            report.semantic_scores, report.nli_results,
            report.rule_results, report.table_matches,
        )
        report.risk_level = self._classify_risk(report.final_score, report.rule_results)

        # 7 — XAI
        report.explanations = self._build_explanations(
            report.rule_results, report.semantic_scores,
            report.nli_results, report.table_matches,
        )

        return report.to_dict()

    # ─────────────────────────────────────────────────────────────────────────
    # Construction des entrées sémantiques
    # ─────────────────────────────────────────────────────────────────────────

    def _build_document_entries(
        self,
        docs: list[tuple[str, dict[str, str]]],
        tables: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for doc_type, fields in docs:
            doc_tables  = tables.get(doc_type, {}) or {}
            semantic_doc = self.narrator.build_semantic_document(doc_type, fields, doc_tables)
            business_text = self.narrator.build_business_text(semantic_doc)
            entries.append({
                "doc_type":      doc_type,
                "fields":        fields,
                "tables":        doc_tables,
                "semantic_doc":  semantic_doc,
                "business_text": business_text,
            })
        self._enrich_bl_goods(entries)
        return entries

    def _enrich_bl_goods(self, entries: list[dict[str, Any]]) -> None:
        """
        When the BL has no goods_description, inject item descriptions from the
        invoice (or PL) so that cargo semantic comparison has something to compare.
        The BL and Invoice describe the same physical goods — just from different
        perspectives (transport vs commercial) — so this enrichment is semantically valid.
        """
        by_type = {e["doc_type"]: e for e in entries}
        bl = by_type.get("bl")
        if bl is None or bl["semantic_doc"].cargo.get("goods_description"):
            return

        inv = by_type.get("invoice")
        pl  = by_type.get("packing_list")
        desc_parts: list[str] = []

        if inv:
            for row in inv["tables"].get("invoice_items", [])[:5]:
                desc = str(row.get("description") or row.get("goods_description") or "").strip()
                if desc and desc not in desc_parts:
                    desc_parts.append(desc)

        if not desc_parts and pl:
            for row in pl["tables"].get("packages", [])[:5]:
                desc = str(row.get("package_type") or row.get("goods_description") or "").strip()
                if desc and desc not in desc_parts:
                    desc_parts.append(desc)

        if desc_parts:
            bl["semantic_doc"].cargo["goods_description"] = "; ".join(desc_parts)
            bl["business_text"] = self.narrator.build_business_text(bl["semantic_doc"])
            log.debug("[cross] BL goods_description enriched from invoice/PL (%d items)", len(desc_parts))

    # ─────────────────────────────────────────────────────────────────────────
    # Sémantique
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_semantic(self, entries: list[dict[str, Any]]) -> dict[str, float]:
        if len(entries) < 2 or not self.embedder.available():
            return {}

        scores: dict[str, float] = {}
        for i, left in enumerate(entries):
            for right in entries[i + 1:]:
                key   = f"{left['doc_type']}-{right['doc_type']}"
                score = self.embedder.similarity(
                    left["business_text"], right["business_text"]
                )
                scores[key] = round(score, 4)

        # Similarités par sous-section (parties, cargo, route)
        for i, left in enumerate(entries):
            for right in entries[i + 1:]:
                base_key = f"{left['doc_type']}-{right['doc_type']}"
                left_party  = self._build_party_text(left["semantic_doc"])
                right_party = self._build_party_text(right["semantic_doc"])
                if left_party and right_party:
                    s = self.embedder.similarity(left_party, right_party)
                    scores[f"{base_key}:parties"] = round(s, 4)

                left_cargo, right_cargo = self._build_aligned_cargo_text(
                    left["semantic_doc"], right["semantic_doc"]
                )
                if left_cargo and right_cargo:
                    s = self.embedder.similarity(left_cargo, right_cargo)
                    scores[f"{base_key}:cargo"] = round(s, 4)

        return scores

    # Rôles canoniques : on compare des parties comparables d'un type de doc à
    # l'autre (shipper/seller/exporter = fournisseur ; consignee/buyer = acheteur),
    # plutôt que des paires clé/valeur aux clés divergentes. Les champs non-partie
    # (notify, carrier, invoice_number, pl_number…) sont ignorés car ils n'ont pas
    # d'équivalent dans les autres documents et diluent artificiellement la similarité.
    _PARTY_ROLE_MAP = {
        "shipper":  "supplier",
        "seller":   "supplier",
        "exporter": "supplier",
        "consignee": "buyer",
        "buyer":     "buyer",
    }

    def _build_party_text(self, sem_doc: SemanticDocument) -> str:
        parts: list[str] = []
        for k, v in sem_doc.shipment.items():
            canon = self._PARTY_ROLE_MAP.get(k)
            if not canon or not v:
                continue
            # Retire le bloc adresse/contact (souvent collé au nom dans le PL :
            # "GLOBAL TRADE SARL | 12, Rue du Commerce | Tunis ...") pour ne garder
            # que la dénomination de la partie.
            name = re.split(r"[|\n]", str(v))[0].strip()
            if name:
                parts.append(f"{canon}: {name}")
        return ". ".join(sorted(parts))

    def _build_cargo_text(self, sem_doc: SemanticDocument) -> str:
        parts = []
        for k, v in sem_doc.cargo.items():
            if v is not None:
                parts.append(f"{k}: {v}")
        tables = sem_doc.tables
        for tbl_name, rows in tables.items():
            for row in rows[:5]:
                desc = row.get("description") or row.get("package_type") or row.get("goods_description")
                if desc:
                    parts.append(str(desc))
        return ". ".join(parts)

    def _build_aligned_cargo_text(
        self,
        left: SemanticDocument,
        right: SemanticDocument,
    ) -> tuple[str, str]:
        """
        Build cargo texts using only fields that are semantically COMMON to both
        document types in a pair.  Prevents false-low similarity from comparing
        BL transport metadata (weights, vessel) against Invoice product details.

        BL ↔ Invoice  → goods descriptions (the same physical goods, different angle)
        BL ↔ PL       → shared weight/volume + goods descriptions
        Invoice ↔ PL  → item descriptions present in both tables
        """
        lt = left.document_type
        rt = right.document_type
        pair = frozenset([lt, rt])

        if pair == frozenset(["bill_of_lading", "commercial_invoice"]):
            bl_doc  = left  if lt == "bill_of_lading"    else right
            inv_doc = right if rt == "commercial_invoice" else left

            bl_goods = bl_doc.cargo.get("goods_description") or ""
            inv_parts: list[str] = []
            for row in inv_doc.tables.get("invoice_items", [])[:5]:
                desc = str(row.get("description") or row.get("goods_description") or "").strip()
                if desc:
                    inv_parts.append(desc)
            return bl_goods, "; ".join(inv_parts)

        if pair == frozenset(["bill_of_lading", "packing_list"]):
            bl_doc = left if lt == "bill_of_lading" else right
            pl_doc = right if lt == "bill_of_lading" else left

            # Signal primaire : descriptions marchandises (le BL est enrichi depuis
            # invoice/PL en amont). Les embeddings comparent mal des nombres bruts
            # (1438 vs 1437 paraît « différent ») — l'écart de poids est déjà couvert
            # par les règles de tolérance, on ne le re-pénalise donc pas ici.
            bl_goods = (bl_doc.cargo.get("goods_description") or "").strip()
            pl_goods = "; ".join(
                str(r.get("package_type") or r.get("goods_description") or "").strip()
                for r in pl_doc.tables.get("packages", [])[:5]
                if (r.get("package_type") or r.get("goods_description"))
            )
            if bl_goods and pl_goods:
                return bl_goods, pl_goods

            # Repli : champs cargo partagés (si aucune description disponible).
            shared = {"gross_weight_kg", "net_weight_kg", "goods_description"}
            lp = [f"{k}: {v}" for k, v in left.cargo.items()  if k in shared and v is not None]
            rp = [f"{k}: {v}" for k, v in right.cargo.items() if k in shared and v is not None]
            return ". ".join(lp), ". ".join(rp)

        if pair == frozenset(["commercial_invoice", "packing_list"]):
            inv_doc = left  if lt == "commercial_invoice" else right
            pl_doc  = right if rt == "packing_list"       else left

            inv_parts2: list[str] = []
            for row in inv_doc.tables.get("invoice_items", [])[:5]:
                desc = str(row.get("description") or row.get("goods_description") or "").strip()
                if desc:
                    inv_parts2.append(desc)

            pl_parts: list[str] = []
            for row in pl_doc.tables.get("packages", [])[:5]:
                desc = str(row.get("package_type") or row.get("goods_description") or "").strip()
                if desc:
                    pl_parts.append(desc)

            return "; ".join(inv_parts2), "; ".join(pl_parts)

        # Fallback for unknown pairs
        return self._build_cargo_text(left), self._build_cargo_text(right)

    # ─────────────────────────────────────────────────────────────────────────
    # NLI
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_nli(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(entries) < 2:
            return []

        pairs = self.nli_analyzer.build_nli_pairs(entries)
        results: list[dict[str, Any]] = []
        for premise, hypothesis, description, category in pairs:
            prediction = self.nli_analyzer.predict(premise, hypothesis)
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

    # ─────────────────────────────────────────────────────────────────────────
    # Alignement tables
    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_tables(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        by_type = {e["doc_type"]: e for e in entries}

        inv_entry = by_type.get("invoice")
        pl_entry  = by_type.get("packing_list")

        if inv_entry and pl_entry:
            inv_items = inv_entry["tables"].get("invoice_items", [])
            pl_items  = pl_entry["tables"].get("packages", [])
            if inv_items:
                matched, unmatched_list = self._match_invoice_to_pl_rows(
                    inv_items, pl_items
                )
                matches.append({
                    "doc_pair":      "invoice↔packing_list",
                    "invoice_rows":  len(inv_items),
                    "pl_rows":       len(pl_items),
                    "matched_rows":  matched,
                    "match_rate":    round(matched / max(len(inv_items), 1), 4),
                    "unmatched_rows": unmatched_list,
                })

        # BL conteneurs ↔ PL
        # Les numéros de conteneur dans le PL sont au niveau des champs (transport),
        # pas dans les lignes de colis — on les lit depuis semantic_doc.transport.
        bl_entry = by_type.get("bl")
        if bl_entry and pl_entry:
            bl_container_rows = bl_entry["tables"].get("containers", [])
            bl_nums: set[str] = set()
            for r in bl_container_rows:
                cnt = str(r.get("container_number") or "").strip().upper()
                if cnt:
                    bl_nums.add(cnt)
            # Fallback BL field-level si la table est vide
            if not bl_nums:
                for cnt in (bl_entry["semantic_doc"].transport.get("container_numbers") or []):
                    c = str(cnt).strip().upper()
                    if c:
                        bl_nums.add(c)

            pl_nums: set[str] = set()
            for cnt in (pl_entry["semantic_doc"].transport.get("container_numbers") or []):
                c = str(cnt).strip().upper()
                if c:
                    pl_nums.add(c)

            if bl_nums or pl_nums:
                common = bl_nums & pl_nums
                matches.append({
                    "doc_pair":           "bl↔packing_list",
                    "bl_containers":      sorted(bl_nums),
                    "pl_containers":      sorted(pl_nums),
                    "common_containers":  sorted(common),
                    "alignment_rate":     round(len(common) / max(len(bl_nums | pl_nums), 1), 4),
                })

        return matches

    def _build_table_analysis(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Analyse croisée des tableaux inter-documents.

        Produit trois sections :
          • parties    — toutes les parties identifiées (expéditeur, destinataire, vendeur…)
          • containers — détails conteneur croisés BL / PL (numéro, scellé, type, poids)
          • goods      — marchandises croisées Invoice / PL (description, HS, prix, poids, CBM)
        """
        by_type = {e["doc_type"]: e for e in entries}
        bl  = by_type.get("bl")
        inv = by_type.get("invoice")
        pl  = by_type.get("packing_list")

        # ── Parties ──────────────────────────────────────────────────────────
        parties: list[dict[str, Any]] = []
        if bl:
            sem = bl["semantic_doc"]
            if sem.shipment.get("shipper"):
                parties.append({"role": "shipper", "document": "bl",
                                 "name": sem.shipment["shipper"]})
            if sem.shipment.get("consignee"):
                parties.append({"role": "consignee", "document": "bl",
                                 "name": sem.shipment["consignee"]})
            if sem.shipment.get("notify"):
                parties.append({"role": "notify_party", "document": "bl",
                                 "name": sem.shipment["notify"]})
            if sem.shipment.get("carrier"):
                parties.append({"role": "carrier", "document": "bl",
                                 "name": sem.shipment["carrier"]})
        if inv:
            sem = inv["semantic_doc"]
            if sem.shipment.get("seller"):
                parties.append({"role": "seller", "document": "invoice",
                                 "name": sem.shipment["seller"]})
            if sem.shipment.get("buyer"):
                parties.append({"role": "buyer", "document": "invoice",
                                 "name": sem.shipment["buyer"]})
        if pl:
            sem = pl["semantic_doc"]
            if sem.shipment.get("shipper"):
                parties.append({"role": "exporter", "document": "packing_list",
                                 "name": sem.shipment["shipper"]})
            if sem.shipment.get("consignee"):
                parties.append({"role": "consignee", "document": "packing_list",
                                 "name": sem.shipment["consignee"]})

        # ── Containers ───────────────────────────────────────────────────────
        containers_map: dict[str, dict[str, Any]] = {}

        if bl:
            for row in bl["tables"].get("containers", []):
                cnt = str(row.get("container_number") or "").strip().upper()
                if not cnt:
                    continue
                containers_map[cnt] = {
                    "container_number": cnt,
                    "bl": {
                        "seal_number":    row.get("seal_number"),
                        "container_type": row.get("container_type"),
                        "gross_weight_kg": row.get("gross_weight_kg"),
                    },
                }
            # Conteneurs BL field-level (si table vide)
            if not containers_map:
                seals = bl["semantic_doc"].transport.get("seal_numbers") or []
                ctype = bl["semantic_doc"].transport.get("container_type")
                for i, cnt in enumerate(
                    bl["semantic_doc"].transport.get("container_numbers") or []
                ):
                    c = str(cnt).strip().upper()
                    if c:
                        containers_map[c] = {
                            "container_number": c,
                            "bl": {
                                "seal_number":    seals[i] if i < len(seals) else None,
                                "container_type": ctype,
                                "gross_weight_kg": None,
                            },
                        }

        if pl:
            for cnt in (pl["semantic_doc"].transport.get("container_numbers") or []):
                c = str(cnt).strip().upper()
                if not c:
                    continue
                if c in containers_map:
                    containers_map[c]["pl"] = {"confirmed": True}
                else:
                    containers_map[c] = {"container_number": c, "pl": {"confirmed": True}}

        # ── Goods ─────────────────────────────────────────────────────────────
        goods_list: list[dict[str, Any]] = []
        currency = inv["semantic_doc"].finance.get("currency") if inv else None

        if inv:
            for row in inv["tables"].get("invoice_items", []):
                desc = str(row.get("description") or row.get("goods_description") or "").strip()
                if not desc:
                    continue
                goods_list.append({
                    "description": desc,
                    "hs_code":     str(row.get("hs_code") or "").strip() or None,
                    "invoice": {
                        "quantity":   row.get("quantity"),
                        "unit":       row.get("unit"),
                        "unit_price": row.get("unit_price"),
                        "line_total": row.get("line_total") or row.get("line_amount"),
                        "currency":   currency,
                    },
                    "packing_list": None,
                })

        if pl:
            for row in pl["tables"].get("packages", []):
                desc = str(row.get("package_type") or row.get("goods_description") or "").strip()
                if not desc:
                    continue
                pl_data = {
                    "quantity":      row.get("quantity") or row.get("number_of_packages"),
                    "unit":          row.get("unit"),
                    "gross_weight_kg": row.get("gross_weight") or row.get("gross_weight_kg"),
                    "net_weight_kg":   row.get("net_weight") or row.get("net_weight_kg"),
                    "cbm":           row.get("cbm") or row.get("measurement"),
                }
                # Try to match with an existing invoice goods entry
                best_idx, best_score = -1, 0.0
                desc_norm = self._normalize_text(desc)
                for i, g in enumerate(goods_list):
                    s = self._string_ratio(desc_norm, self._normalize_text(g["description"]))
                    if s > best_score:
                        best_score, best_idx = s, i
                if best_score >= 0.65 and best_idx >= 0:
                    goods_list[best_idx]["packing_list"] = pl_data
                else:
                    # PL item without matching invoice row
                    goods_list.append({
                        "description":  desc,
                        "hs_code":      None,
                        "invoice":      None,
                        "packing_list": pl_data,
                    })

        return {
            "parties":    parties,
            "containers": list(containers_map.values()),
            "goods":      goods_list,
        }

    def _match_invoice_to_pl_rows(
        self,
        invoice_rows: list[dict[str, Any]],
        pl_rows: list[dict[str, Any]],
    ) -> tuple[int, list[dict[str, Any]]]:
        matched   = 0
        unmatched: list[dict[str, Any]] = []
        for inv_row in invoice_rows:
            best_score = 0.0
            best_pl_row = None
            for pl_row in pl_rows:
                score = self._row_similarity(inv_row, pl_row)
                if score > best_score:
                    best_score = score
                    best_pl_row = pl_row
            if best_score >= 0.55:
                matched += 1
            else:
                unmatched.append({
                    "invoice_row": {
                        k: v for k, v in inv_row.items()
                        if k in ("description", "goods_description", "quantity", "hs_code")
                    },
                    "best_score": round(best_score, 3),
                    "best_pl_match": {
                        k: v for k, v in (best_pl_row or {}).items()
                        if k in ("package_type", "goods_description", "quantity", "hs_code")
                    } if best_pl_row else {},
                })
        return matched, unmatched

    def _row_similarity(self, inv_row: dict, pl_row: dict) -> float:
        # Similarité description
        inv_desc = self._normalize_text(
            str(inv_row.get("description") or inv_row.get("goods_description") or "")
        )
        pl_desc = self._normalize_text(
            str(pl_row.get("package_type") or pl_row.get("goods_description") or "")
        )
        name_score = self._string_ratio(inv_desc, pl_desc)

        # Similarité HS code
        inv_hs = str(inv_row.get("hs_code", "") or "").strip()
        pl_hs  = str(pl_row.get("hs_code", "") or "").strip()
        hs_score = 1.0 if (inv_hs and pl_hs and inv_hs[:4] == pl_hs[:4]) else (
            0.5 if (inv_hs and pl_hs) else 0.0
        )

        # Similarité quantité
        try:
            inv_qty = float(str(inv_row.get("quantity", 0) or 0).replace(",", "."))
            pl_qty  = float(str(
                pl_row.get("quantity", 0) or pl_row.get("number_of_packages", 0) or 0
            ).replace(",", "."))
            qty_score = (
                1.0 - min(1.0, abs(inv_qty - pl_qty) / max(inv_qty, pl_qty, 1.0))
                if inv_qty > 0 and pl_qty > 0 else 0.4
            )
        except (ValueError, TypeError):
            qty_score = 0.4

        weights = [
            (name_score, 0.55),
            (hs_score,   0.30),
            (qty_score,  0.15),
        ]
        return sum(s * w for s, w in weights)

    # ─────────────────────────────────────────────────────────────────────────
    # Arbitration — Rules override NLI model artifacts
    # ─────────────────────────────────────────────────────────────────────────

    def _arbitrate_nli(
        self,
        nli_results: list[dict[str, Any]],
        rule_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Reconcile NLI model outputs against deterministic rule results.

        The cross-encoder NLI model has two known failure modes on trade docs:
          • Same value in two document types → CONTRADICTION (domain gap)
          • Near-match numbers (e.g. 1438 vs 1437) → CONTRADICTION (strict numeric)

        When a deterministic rule confirms PASS for the same business fact, the NLI
        CONTRADICTION or NEUTRAL is overridden to ENTAILMENT (rule-confirmed confidence
        score 0.80), signalling that the pair is coherent without over-claiming.
        """
        passed = {
            r["rule"] for r in rule_results
            if r.get("status") in ("pass", "corrected")
        }

        adjusted: list[dict[str, Any]] = []
        for res in nli_results:
            r = dict(res)
            cat = r.get("category", "")
            related = _NLI_TO_RULES.get(cat, [])
            rule_confirms = any(rule in passed for rule in related)

            if rule_confirms and r.get("nli_label") in ("CONTRADICTION", "NEUTRAL"):
                # Deterministic rule confirms the fact — trust rules over model
                r["nli_label"] = "ENTAILMENT"
                r["score"] = 0.80
                r["arbitrated"] = True

            adjusted.append(r)
        return adjusted

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_final_score(
        self,
        semantic_scores: dict[str, float],
        nli_results: list[dict[str, Any]],
        rule_results: list[dict[str, Any]],
        table_matches: list[dict[str, Any]],
    ) -> tuple[dict[str, float], float]:
        semantic = self._aggregate_semantic(semantic_scores)
        nli      = self._aggregate_nli(nli_results)
        rules    = self._aggregate_rules(rule_results)
        table    = self._aggregate_table(table_matches)
        party    = self._aggregate_party(rule_results)

        breakdown = {
            "semantic": round(semantic, 4),
            "nli":      round(nli, 4),
            "rules":    round(rules, 4),
            "table":    round(table, 4),
            "party":    round(party, 4),
        }

        final = sum(
            breakdown[k] * _SCORE_WEIGHTS[k]
            for k in _SCORE_WEIGHTS
        )
        return breakdown, round(final, 4)

    def _aggregate_semantic(self, scores: dict[str, float]) -> float:
        """
        Agrège le signal sémantique sur les comparaisons ALIGNÉES PAR SECTION
        (cargo + parties), qui sont sémantiquement comparables entre documents
        complémentaires. Les similarités de `business_text` globaux sont ignorées
        ici (elles restent exposées pour monitoring) car elles mesurent un simple
        recouvrement textuel, faible par nature et non pertinent pour la cohérence.
        """
        cargo  = [v for k, v in scores.items() if k.endswith(":cargo")]
        party  = [v for k, v in scores.items() if k.endswith(":parties")]

        if cargo or party:
            cargo_avg = sum(cargo) / len(cargo) if cargo else None
            party_avg = sum(party) / len(party) if party else None
            if cargo_avg is not None and party_avg is not None:
                # Le cargo (marchandises) porte le signal métier le plus fort.
                return cargo_avg * 0.6 + party_avg * 0.4
            return cargo_avg if cargo_avg is not None else party_avg

        # Repli : aucune section disponible → moyenne des blobs globaux.
        global_scores = [v for k, v in scores.items() if ":" not in k]
        return sum(global_scores) / max(len(global_scores), 1) if global_scores else 0.0

    def _aggregate_nli(self, nli_results: list[dict[str, Any]]) -> float:
        if not nli_results:
            return 0.0
        total = 0.0
        weights = {"global": 1.5, "goods": 1.3, "weight": 1.2, "container": 1.2,
                   "parties": 1.1, "incoterm": 1.0, "reference": 0.8}
        w_sum = 0.0
        for result in nli_results:
            label    = result.get("nli_label", "NEUTRAL")
            score    = float(result.get("score", 0.5))
            category = result.get("category", "global")
            w        = weights.get(category, 1.0)
            if label == "ENTAILMENT":
                contribution = score
            elif label == "CONTRADICTION":
                contribution = max(0.0, 0.2 - score)
            else:
                contribution = score * 0.5
            total += contribution * w
            w_sum += w
        return total / max(w_sum, 1.0)

    def _aggregate_rules(self, rule_results: list[dict[str, Any]]) -> float:
        if not rule_results:
            return 0.0
        status_map = {
            "pass": 1.0, "corrected": 0.95, "warning": 0.65,
            "fail": 0.0, "skipped": 0.85,
        }
        sev_weight = {"critical": 2.0, "high": 1.5, "medium": 1.2, "low": 1.0, "info": 0.8}
        total = 0.0
        w_sum = 0.0
        for r in rule_results:
            status = r.get("status", "skipped")
            sev    = r.get("severity", "info")
            s      = status_map.get(status, 0.5)
            w      = sev_weight.get(sev, 1.0)
            total += s * w
            w_sum += w
        return total / max(w_sum, 1.0)

    def _aggregate_table(self, table_matches: list[dict[str, Any]]) -> float:
        if not table_matches:
            return 0.0
        total = 0.0
        for match in table_matches:
            rate = match.get("match_rate") or match.get("alignment_rate")
            if rate is not None:
                total += float(rate)
            else:
                matched = match.get("matched_rows", 0)
                rows    = max(match.get("invoice_rows", 1), 1)
                total  += matched / rows
        return min(1.0, total / len(table_matches))

    def _aggregate_party(self, rule_results: list[dict[str, Any]]) -> float:
        _party_prefixes = (
            "cross_shipper", "cross_consignee",
            "cross_container_number", "cross_incoterm_consistency", "cross_currency",
        )
        party_rules = [
            r for r in rule_results
            if any(r.get("rule", "").startswith(pfx) for pfx in _party_prefixes)
        ]
        if not party_rules:
            return 0.65
        good = sum(1 for r in party_rules if r.get("status") in ("pass", "corrected"))
        return good / len(party_rules)

    # ─────────────────────────────────────────────────────────────────────────
    # Classification du risque
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_risk(
        self,
        score: float,
        rule_results: list[dict[str, Any]],
    ) -> str:
        # Escalade automatique si règle critique en échec
        has_critical = any(
            r.get("severity") == "critical" and r.get("status") == "fail"
            for r in rule_results
        )
        has_high_fail = any(
            r.get("severity") == "high" and r.get("status") == "fail"
            for r in rule_results
        )
        if has_critical or score < 0.40:
            return "critical"
        if has_high_fail or score < 0.60:
            return "high"
        if score < 0.75:
            return "medium"
        return "low"

    # ─────────────────────────────────────────────────────────────────────────
    # XAI — Explainability
    # ─────────────────────────────────────────────────────────────────────────

    def _build_explanations(
        self,
        rule_results: list[dict[str, Any]],
        semantic_scores: dict[str, float],
        nli_results: list[dict[str, Any]],
        table_matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        explanations: list[dict[str, Any]] = []

        # ── Règles ───────────────────────────────────────────────────────────
        for r in rule_results:
            status = r.get("status", "skipped")
            sev    = r.get("severity", "info")
            msg    = r.get("message", "")
            impact = "positive" if status in ("pass", "corrected") else (
                "warning" if status == "warning" else (
                    "negative" if status == "fail" else "neutral"
                )
            )
            explanations.append({
                "type":     "rule",
                "category": "business_rule",
                "impact":   impact,
                "severity": sev,
                "message":  msg,
                "rule":     r.get("rule"),
                "field":    r.get("field"),
                "doc_pair": r.get("doc_pair"),
                "status":   status,
            })

        # ── Sémantique ────────────────────────────────────────────────────────
        for key, value in semantic_scores.items():
            if ":" in key:
                # Sections alignées (parties / cargo) → vrai signal de cohérence.
                pair, sub = key.split(":", 1)
                label = f"Semantic similarity [{sub}] for {pair}: {value:.0%}"
                impact = "positive" if value >= 0.80 else (
                    "warning" if value >= 0.60 else "negative"
                )
            else:
                # Blob global : documents complémentaires → recouvrement partiel
                # attendu, purement informatif (n'indique pas une incohérence).
                label = (
                    f"Structural text overlap for {key}: {value:.0%} "
                    f"(complementary documents — informational)"
                )
                impact = "positive" if value >= 0.65 else "neutral"
            explanations.append({
                "type":     "semantic",
                "category": "embedding",
                "impact":   impact,
                "severity": "info",
                "message":  label,
                "score":    value,
                "doc_pair": key.split(":")[0],
            })

        # ── NLI ──────────────────────────────────────────────────────────────
        for result in nli_results:
            label    = result.get("nli_label", "NEUTRAL")
            score    = result.get("score", 0.0)
            category = result.get("category", "global")
            desc     = result.get("description", "")
            impact = "positive" if label == "ENTAILMENT" else (
                "negative" if label == "CONTRADICTION" else "neutral"
            )
            sev = "high" if label == "CONTRADICTION" else "info"
            explanations.append({
                "type":     "nli",
                "category": category,
                "impact":   impact,
                "severity": sev,
                "message":  f"NLI [{label}] {desc} (confidence {score:.0%})",
                "nli_label": label,
                "score":    score,
            })

        # ── Tables ────────────────────────────────────────────────────────────
        for match in table_matches:
            pair = match.get("doc_pair", "")
            if "invoice" in pair and "packing" in pair:
                rate = match.get("match_rate", 0)
                inv_rows = match.get("invoice_rows", 0)
                matched  = match.get("matched_rows", 0)
                impact   = "positive" if rate >= 0.80 else ("warning" if rate >= 0.50 else "negative")
                explanations.append({
                    "type":     "table",
                    "category": "row_alignment",
                    "impact":   impact,
                    "severity": "low" if rate >= 0.50 else "medium",
                    "message":  (
                        f"Invoice line items matched {matched}/{inv_rows} "
                        f"rows in packing list ({rate:.0%} match rate)."
                    ),
                    "doc_pair": pair,
                    "match_rate": rate,
                })
            elif "bl" in pair and "packing" in pair:
                rate = match.get("alignment_rate", 0)
                common = match.get("common_containers", [])
                impact = "positive" if rate >= 0.80 else ("warning" if rate > 0 else "negative")
                explanations.append({
                    "type":     "table",
                    "category": "container_alignment",
                    "impact":   impact,
                    "severity": "info" if rate >= 0.80 else "medium",
                    "message":  (
                        f"Container alignment BL↔PL: {rate:.0%}. "
                        f"Common containers: {sorted(common) if common else 'none'}."
                    ),
                    "doc_pair": pair,
                    "match_rate": rate,
                })

        # Tri par sévérité décroissante
        explanations.sort(
            key=lambda e: -_SEVERITY_ORDER.get(e.get("severity", "info"), 0)
        )
        return explanations

    # ─────────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", " ", text or "").strip().lower()

    def _string_ratio(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        try:
            from rapidfuzz.fuzz import token_set_ratio
            return token_set_ratio(a, b) / 100.0
        except ImportError:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio()
