"""
pipeline.py
===========
Pipeline multi-documents — classification + extraction de champs.

Accepte jusqu'à 3 documents simultanément :
  • PDF natif  (texte vectoriel PyMuPDF) → mots-clés + regex, zéro ML
  • PDF scanné (aucun texte embarqué)    → OCR + LayoutLMv3
  • Image      (PNG, JPG, TIFF, BMP)     → OCR + LayoutLMv3

Pour chaque document :
  [1] Détection automatique du format d'entrée
  [2a] PDF natif  : KeywordClassifier → doc_type | NativeExtractor → fields
  [2b] Scanné/img : ClassifierService + OCR + LayoutLMv3 → doc_type + fields
  [3] Résultat structuré JSON

Traitement parallèle via ThreadPoolExecutor (max 3 workers).

Optimisations v2 :
  - Fallback LayoutLM limité aux champs critiques uniquement
  - _run_layoutlm_on_native_pdf désactivé (trop lent sur CPU/MX130)
  - use_fast=True sur tous les processors LayoutLMv3
  - torch.set_num_threads pour maximiser l'utilisation CPU
  - Timings par étape dans le résultat JSON

Usage (standalone) :
  python services/pipeline.py doc1.pdf doc2.pdf doc3.jpg
  python services/pipeline.py --files doc1.pdf doc2.pdf --output results.json

Usage (import) :
  from release2.services.pipeline import DocumentPipeline
  pipeline = DocumentPipeline()
  result = pipeline.process(["doc1.pdf", "doc2.pdf", "doc3.jpg"])
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

from .template_matcher import TemplateMatcher
from .cross_validator import CrossValidator

_template_matcher = TemplateMatcher()
_cross_validator  = CrossValidator()

log = logging.getLogger(__name__)

# Un seul modèle LayoutLM à la fois sur CPU.
# Sans ce verrou, 3 inférences concurrentes se battent pour les mêmes cœurs CPU
# (chacune lance N threads PyTorch/OpenMP) → ×30-40 de ralentissement par modèle.
_LAYOUTLM_LOCK = threading.Lock()

# ── Limites LayoutLM ─────────────────────────────────────────────────────────
# Page 1 contient 100 % des champs LC critiques (montant, devise, dates, parties).
# Traiter N pages multiplie le temps sans gain mesurable pour la cross-validation.
# 4 pages × 17s = 68s  →  1 page × 17s = 17s  →  3 docs = 51s au lieu de 204s.
_MAX_LAYOUTLM_PAGES = 1

# LayoutLM est O(n) sur les tokens mais le preprocessing CPU est O(n²) sur les mots.
# Les champs LC tiennent dans les 450 premiers mots de la page 1.
# 450 mots → ~12-15s | 800 mots → ~25s | 1200 mots → timeout.
_MAX_LAYOUTLM_WORDS = 450

# ── Optimisation CPU ─────────────────────────────────────────────────────────
# Avec le lock (1 inférence à la fois), on alloue tous les threads disponibles
# à cette inférence, mais on plafonne à cpu_count-2 pour laisser des cœurs
# aux autres processus système (Flask, workers, etc.).
try:
    import torch
    import os as _os
    _cpu_count = _os.cpu_count() or 4
    _n_torch_threads = max(2, min(_cpu_count - 2, 6))
    torch.set_num_threads(_n_torch_threads)
    log.info("[pipeline] torch threads = %d / %d CPU", _n_torch_threads, _cpu_count)
except ImportError:
    pass

# ── Confiances par tier ───────────────────────────────────────────────────────
_TIER_CONFIDENCE: dict[str, float] = {
    "spatial_anchor":       0.92,
    "spatial_grid":         0.90,
    "spatial_inline":       0.88,
    "spatial_party":        0.85,
    "regex":                0.85,
    "party_block":          0.83,
    "kv_regex":             0.83,
    "layoutlm_specialized": 0.82,   # modèles fine-tunés par type — PRIMARY
    "zone_header":          0.80,
    "zone_parties":         0.78,
    "zone":                 0.75,
    "semantic_block":       0.75,
    "semantic":             0.72,
    "computed":             0.70,
    "layout_structure":     0.65,
    "layoutlm":             0.60,
    "spacy":                0.55,
    "kv_fallback":          0.50,
    "semantic_flat":        0.45,
}


def _tier_conf(tier_name: str) -> float:
    """Retourne la confiance associée à un nom de tier (préfixe ou exact)."""
    c = _TIER_CONFIDENCE.get(tier_name)
    if c is not None:
        return c
    prefix = tier_name.split("_")[0]
    return _TIER_CONFIDENCE.get(prefix, 0.50)


# Champs désambiguïsés par la POSITION du label (loading vs discharge, receipt vs
# delivery, vessel vs voyage). Sur les layouts réels, LayoutLM les confond (il prend
# la valeur d'un champ voisin) ; l'extracteur zone — ancré sur la position du label —
# est fiable. On lui donne donc la priorité sur layoutlm_specialized (0.82) pour ces
# champs précis, tout en restant sous spatial_anchor (0.92).
_POSITION_ANCHORED_FIELDS = {
    "port_of_loading", "port_of_discharge", "place_of_receipt", "place_of_delivery",
    "vessel", "voyage",
}
_POSITION_ANCHORED_CONF = 0.90


# ── Constantes ────────────────────────────────────────────────────────────────

RELEASE2_ROOT    = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
MIN_NATIVE_WORDS = 10

_TABLE_FIELD_REMAP: dict[str, dict[str, str]] = {
    "bl": {
        "measurement":        "cbm",
        "goods_description":  "description",
        "number_of_packages": "packages",
        "shipping_marks":     "marks",
    },
    "invoice": {
        "goods_description": "description",
        "line_total":        "line_amount",
    },
    "packing_list": {
        "measurement":        "cbm",
        "goods_description":  "package_type",
        "number_of_packages": "quantity",
        "package_unit":       "unit",
        "gross_weight_kg":    "gross_weight",
        "net_weight_kg":      "net_weight",
        "shipping_marks":     "marks",
    },
}

_TABLE_NAME: dict[str, str] = {
    "bl":           "containers",
    "invoice":      "invoice_items",
    "packing_list": "packages",
}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

class DocumentPipeline:
    """
    Pipeline complet 4 couches :
      [1] Extraction     — LayoutLM spécialisé (PRIMARY) + regex/spatial/zone comme FALLBACK
      [2] Enrichissement — SpaCyEnricher  (NER : dates, lieux manquants)
      [3] Règles métier  — RulesEngine    (validation, normalisation, cohérence)
      [4] Scoring        — ConfidenceScorer (score global + par champ)

    Ordre d'extraction (PDF natif) :
      1. LayoutLMv3 spécialisé (best_bl/best_invoice/best_pl) — conf 0.82 — PRIMARY
      2. Regex + party-block  — conf 0.83-0.85 — écrase LayoutLM si plus confiant
      3. Spatial anchor       — conf 0.88-0.92 — écrase tout si plus confiant
      4. Zone / Semantic      — conf 0.72-0.80 — comble uniquement les trous
      5. Layout structure     — tables seulement, inchangé
    """

    MAX_DOCS = 3

    def __init__(
        self,
        donut_model_dir:        Path | str | None = None,
        confidence_threshold:   float = 0.80,
        max_workers:            int   = 3,
        use_spacy:              bool  = False,
        rules_strict:           bool  = False,
        classifier_model_dir:   Path | str | None = None,
        extractor_model_dir:    Path | str | None = None,
        ocr_languages:          list[str] | None = None,
        ocr_gpu:                bool = False,
        enable_layoutlm_native: bool = False,   # ← nouveau paramètre
    ):
        from release2.services.native_extractor  import NativeExtractor
        from release2.services.spacy_enricher    import SpaCyEnricher
        from release2.services.rules_engine      import RulesEngine
        from release2.services.confidence_scorer import ConfidenceScorer
        from release2.services.spatial_extractor import SpatialExtractor

        self._donut_model_dir          = Path(donut_model_dir) if donut_model_dir else None
        self._classifier_model_dir     = Path(classifier_model_dir) if classifier_model_dir else None
        self._extractor_model_dir      = Path(extractor_model_dir) if extractor_model_dir else None
        self._threshold                = confidence_threshold
        self._max_workers              = min(max_workers, self.MAX_DOCS)
        self._use_spacy                = use_spacy
        self._ocr_languages            = ocr_languages
        self._ocr_gpu                  = ocr_gpu
        self._enable_layoutlm_native   = enable_layoutlm_native

        if not self._ocr_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            log.info("[pipeline] CUDA disabled for this analysis session")

        self._spatial_ext    = SpatialExtractor()
        self._native_ext     = NativeExtractor()
        self._classifier     = None
        self._extractor      = None
        self._ocr_reader     = None
        self._ocr_enhancer   = None
        self._spacy_enricher = SpaCyEnricher()
        self._rules_engine   = RulesEngine(strict=rules_strict)
        self._scorer         = ConfidenceScorer()

        log.info(
            "Pipeline 4-couches prêt (workers=%d, seuil=%.0f%%, spacy=%s, layoutlm=PRIMARY)",
            self._max_workers, self._threshold * 100, use_spacy,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """
        Pré-charge les trois extracteurs spécialisés (BL, Invoice, PL) au démarrage.
        Sans pré-chargement, le premier appel déclenche le chargement des poids
        en plein traitement, augmentant le temps de la première requête.
        """
        for dt in ("bl", "invoice", "packing_list"):
            try:
                self._get_specialized_extractor(dt)
                log.info("[warmup] %s prêt", dt)
            except Exception as exc:
                log.warning("[warmup] %s échoué : %s", dt, exc)

    def process(
        self,
        documents:   list[str | Path],
        output_path: str | Path | None = None,
    ) -> dict:
        """
        Traite 1 à 3 documents et retourne un résultat structuré.
        """
        if not documents:
            raise ValueError("Au moins un document est requis.")
        if len(documents) > self.MAX_DOCS:
            raise ValueError(
                f"Maximum {self.MAX_DOCS} documents par appel (reçu : {len(documents)})."
            )

        paths = [Path(d) for d in documents]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Fichier introuvable : {p}")

        log.info("Traitement de %d document(s)...", len(paths))
        t0 = time.perf_counter()

        results: list[dict | None] = [None] * len(paths)
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._process_one, p): i
                for i, p in enumerate(paths)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    log.error("Erreur sur %s : %s", paths[idx].name, exc, exc_info=True)
                    results[idx] = self._error_result(paths[idx], str(exc))

        elapsed = round(time.perf_counter() - t0, 2)
        failed  = sum(1 for r in results if r.get("status") == "error")

        # ── Propagation inter-documents ───────────────────────────────────────
        # L'invoice est la source de vérité financière et contractuelle.
        # Les champs absents du BL et du PL sont propagés depuis l'invoice avec
        # un tier "propagated_from_invoice" (confiance intermédiaire 0.75).
        results = self._propagate_shared_fields(results)

        # Validation inter-documents
        cross_report: dict[str, Any] = {
            "rule_results": [], "semantic_scores": {}, "nli_results": [],
            "table_matches": [], "explanations": [], "final_score": None,
            "risk_level": "unknown", "documents": [],
        }
        successful = [
            (r["doc_type"], r["fields"])
            for r in results
            if r.get("status") == "success" and r.get("doc_type") and r.get("fields")
        ]
        doc_tables_by_type = {
            r["doc_type"]: r.get("tables", {})
            for r in results
            if r.get("status") == "success" and r.get("doc_type")
        }
        if len(successful) >= 2:
            cross_report   = _cross_validator.validate_documents(successful, tables=doc_tables_by_type)
            cross_results  = cross_report.get("rule_results", [])
            n_cross_fail   = sum(1 for r in cross_results if r.get("status") == "fail")
            if n_cross_fail:
                log.warning("%d incohérence(s) inter-documents détectée(s)", n_cross_fail)

        output = {
            "summary": {
                "total_documents":   len(paths),
                "processed":         len(paths) - failed,
                "failed":            failed,
                "processing_time_s": elapsed,
            },
            "documents":        results,
            "cross_validation": cross_report.get("rule_results", []),
            "cross_report":     cross_report,
        }

        log.info("Terminé en %.1fs — %d traité(s), %d erreur(s)", elapsed, len(paths) - failed, failed)

        if output_path:
            Path(output_path).write_text(
                json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log.info("Résultats sauvegardés → %s", output_path)

        return output

    # ──────────────────────────────────────────────────────────────────────────
    # Traitement d'un seul document
    # ──────────────────────────────────────────────────────────────────────────

    def _process_one(self, path: Path) -> dict:
        t0       = time.perf_counter()
        warnings: list[str] = []
        timings:  dict[str, float] = {}

        def _tick(label: str, t_start: float) -> float:
            """Enregistre la durée d'une étape, loggue, retourne le nouveau t_start."""
            duration = round(time.perf_counter() - t_start, 2)
            timings[label] = duration
            log.debug("[%s] %-35s %.2fs", path.name, label, duration)
            return time.perf_counter()

        # ── Couche 1 : Extraction ─────────────────────────────────────────────
        input_type = self._detect_type(path)
        log.info("[%s] format=%s", path.name, input_type)

        if input_type == "pdf_native":
            t = time.perf_counter()

            # Step 01 — Classification par mots-clés (zéro ML)
            doc_type, confidence, probas = self._classify_keyword(path)
            t = _tick("01_classification", t)

            # Step 02 — LayoutLM spécialisé : extraction PRIMAIRE
            # Modèle fine-tuné par type de document (best_bl / best_invoice / best_pl).
            # Rendu PDF page par page + bboxes PyMuPDF → pas d'OCR nécessaire.
            lm_primary  = self._run_layoutlm_on_native_pdf(path, doc_type)
            lm_sp_conf  = _tier_conf("layoutlm_specialized")
            field_conf: dict[str, float] = {}
            primary_fields: dict[str, str] = {}
            primary_tiers:  dict[str, str] = {}
            for fname, val in lm_primary["fields"].items():
                if val:
                    primary_fields[fname] = val
                    primary_tiers[fname]  = "layoutlm_specialized"
                    field_conf[fname]     = lm_sp_conf
            t = _tick("02_layoutlm_specialized", t)

            # Step 03 — Extraction native (regex + KV) : FALLBACK
            # regex/party_block (conf 0.83-0.85) > layoutlm_specialized (0.82) → peut écraser.
            # kv_fallback (0.50) < 0.82 → comble uniquement les champs manquants.
            native_result = self._native_ext.extract(path, doc_type)
            native_tiers  = dict(native_result.get("extraction_tiers", {}))
            native_fields = native_result.get("fields", {})
            for fname, val in native_fields.items():
                nat_conf = _tier_conf(native_tiers.get(fname, "regex"))
                if val and self._can_overwrite(fname, nat_conf, field_conf):
                    primary_fields[fname] = val
                    primary_tiers[fname]  = native_tiers.get(fname, "regex")
                    field_conf[fname]     = nat_conf
            t = _tick("03_native_extraction", t)

            timings["04_layoutlm_fallback_critical"] = 0.0  # remplacé par step 02

            extraction = {
                "fields":           primary_fields,
                "extraction_tiers": primary_tiers,
                "n_pages":          native_result.get("n_pages", 0),
                "pages":            native_result.get("pages", []),
                "structured_rows":  [],
            }

            tiers     = dict(extraction.get("extraction_tiers", {}))
            full_text = self._native_ext.debug_text(path)

            # Détection du carrier
            carrier       = _template_matcher.identify_carrier(full_text)
            carrier_hints = _template_matcher.get_hints(carrier)
            if carrier:
                log.info("[%s] Carrier détecté : %s", path.name, carrier)

            # Pattern bl_number carrier si absent
            if doc_type == "bl" and "bl_number" not in primary_fields:
                bl_pattern = carrier_hints.get("bl_number_pattern")
                if bl_pattern:
                    m = re.search(bl_pattern, full_text)
                    if m:
                        primary_fields["bl_number"] = m.group(0)
                        extraction["extraction_tiers"]["bl_number"] = "template_pattern"
                        log.info(
                            "[%s] bl_number via pattern carrier (%s): %s",
                            path.name, carrier, m.group(0),
                        )

            # ── Spatial (spatial_anchor=0.92 > 0.82 → peut écraser LayoutLM) ──
            spatial_result = self._run_spatial_extraction(path, doc_type)
            for fname, val in spatial_result["fields"].items():
                sp_conf = spatial_result["confidences"][fname]
                sp_tier = spatial_result["tiers"][fname]
                if self._can_overwrite(fname, sp_conf, field_conf):
                    extraction["fields"][fname] = val
                    tiers[fname]               = sp_tier
                    field_conf[fname]          = sp_conf
            t = _tick("05_spatial_extraction", t)

            # ── Zones sémantiques (zone=0.75-0.80 < 0.82 → comble les trous) ──
            zone_result = self._run_zone_extraction(path, doc_type)
            for fname, val in zone_result["fields"].items():
                zone      = zone_result["zone_map"].get(fname, "unknown")
                tier_name = f"zone_{zone}"
                z_conf    = _tier_conf(tier_name)
                # Champs position-ancrés : zone prime sur layoutlm (anti-confusion
                # loading/discharge, vessel/incoterm). Cf. _POSITION_ANCHORED_FIELDS.
                if fname in _POSITION_ANCHORED_FIELDS:
                    z_conf = max(z_conf, _POSITION_ANCHORED_CONF)
                if self._can_overwrite(fname, z_conf, field_conf):
                    extraction["fields"][fname] = val
                    tiers[fname]               = tier_name
                    field_conf[fname]          = z_conf
            t = _tick("06_zone_extraction", t)

            # ── Analyse sémantique par blocs (semantic=0.72 < 0.82 → comble les trous) ──
            pdf_blocks = self._extract_layout_blocks(path)
            sem_fields = self._run_semantic_analysis(
                full_text, doc_type, extraction.get("fields", {}),
                pdf_blocks=pdf_blocks,
                existing_tiers=tiers,
            )
            sem_conf = _tier_conf("semantic")
            for fname, val in sem_fields.items():
                if self._can_overwrite(fname, sem_conf, field_conf):
                    extraction["fields"][fname] = val
                    tiers[fname]               = "semantic"
                    field_conf[fname]          = sem_conf
            t = _tick("07_semantic_analysis", t)

            # ── Structure-first : champs tabulaires (inchangé — ne pas toucher) ──
            layout_native = self._run_layout_native(path, doc_type, carrier_hints)
            ls_conf = _tier_conf("layout_structure")
            for fname, val in layout_native["table_fields"].items():
                if not self._is_valid_table_field(fname, val):
                    continue
                if self._can_overwrite(fname, ls_conf, field_conf):
                    extraction["fields"][fname] = val
                    tiers[fname]               = "layout_structure"
                    field_conf[fname]          = ls_conf
            extraction["structured_rows"] = layout_native["structured_rows"]
            t = _tick("08_layout_structure", t)

            timings["09_layoutlm_native"] = 0.0  # intégré dans step 02

        else:
            t = time.perf_counter()
            doc_type, confidence, probas = self._classify_visual(path, input_type)
            t = _tick("01_classification", t)

            # Step 02 — LayoutLM spécialisé : extraction PRIMAIRE (OCR + modèle fin-tuné)
            extraction = self._extract_scanned(path, doc_type, input_type)
            t = _tick("02_scanned_extraction", t)

            layout_field_keys = set(extraction.get("table_fields", {}).keys())
            tiers = {
                f: ("layout_structure" if f in layout_field_keys else "layoutlm_specialized")
                for f in extraction.get("fields", {})
            }
            field_conf: dict[str, float] = {
                fname: _tier_conf(tier_name) for fname, tier_name in tiers.items()
            }
            full_text = extraction.get("full_text", "")

            # Step 03 — Regex fallback sur texte OCR pour champs manquants/basse confiance
            if full_text:
                try:
                    nat_result = self._native_ext.extract_from_text(full_text, doc_type)
                    nat_tiers  = nat_result.get("extraction_tiers", {})
                    nat_fields = nat_result.get("fields", {})
                    for fname, val in nat_fields.items():
                        nat_conf = _tier_conf(nat_tiers.get(fname, "regex"))
                        if val and self._can_overwrite(fname, nat_conf, field_conf):
                            extraction.setdefault("fields", {})[fname] = val
                            tiers[fname]      = nat_tiers.get(fname, "regex")
                            field_conf[fname] = nat_conf
                except Exception as exc:
                    log.warning("[%s] Regex fallback OCR échoué : %s", path.name, exc)
            t = _tick("03_regex_fallback_ocr", t)

        fields = dict(extraction.get("fields", {}))

        # ── Construction des tables nommées ───────────────────────────────────
        t = time.perf_counter()
        structured_rows  = extraction.get("structured_rows", [])
        doc_tables       = self._build_document_tables(doc_type, structured_rows)
        # Synthesize a summary row from fields only when layout detection found nothing at all
        if not doc_tables and not structured_rows:
            doc_tables = self._synthesize_table_from_fields(doc_type, fields)
        table_rule_results = self._rules_engine.validate_tables(doc_tables, doc_type)
        t = _tick("09_build_tables", t)

        # ── Couche 2 : Enrichissement spaCy ──────────────────────────────────
        t = time.perf_counter()
        if self._use_spacy and full_text:
            from release2.services.native_extractor import _PATTERNS
            all_fields = set(_PATTERNS.get(doc_type, {}).keys())
            missing    = all_fields - set(fields)
            if missing:
                new_fields = self._spacy_enricher.enrich(fields, full_text, doc_type, missing)
                sp_conf    = _tier_conf("spacy")
                for f, v in new_fields.items():
                    fields[f]     = v
                    tiers[f]      = "spacy"
                    field_conf[f] = sp_conf
        t = _tick("10_spacy_enrichment", t)

        # ── Couche 3 : Règles métier ──────────────────────────────────────────
        t = time.perf_counter()
        corrected_fields, rule_results = self._rules_engine.validate(fields, doc_type)
        rule_results += table_rule_results
        rule_summary = self._rules_engine.summarize(rule_results)
        t = _tick("11_rules_engine", t)

        # Agrégats tableau
        t = time.perf_counter()
        aggregate_fields  = self._compute_table_aggregates(doc_type, doc_tables, corrected_fields)
        _pl_weight_fields = {"gross_weight_kg", "net_weight_kg"}
        for fname, val in aggregate_fields.items():
            is_pl_weight_override = (doc_type == "packing_list" and fname in _pl_weight_fields)
            if is_pl_weight_override or fname not in corrected_fields:
                corrected_fields[fname] = val
                tiers[fname] = "computed"
        t = _tick("12_aggregates", t)

        # ── Couche 4 : Scoring ────────────────────────────────────────────────
        t = time.perf_counter()
        scoring = self._scorer.score(corrected_fields, rule_results, doc_type, confidence, tiers)
        t = _tick("13_scoring", t)

        # ── Alertes ───────────────────────────────────────────────────────────
        confident = confidence >= self._threshold
        if not confident:
            warnings.append(
                f"Confiance classifieur faible : {confidence:.1%} < {self._threshold:.0%}"
            )
        if rule_summary["fail"] > 0:
            warnings.append(
                f"{rule_summary['fail']} règle(s) métier échouée(s) — vérification recommandée"
            )
        if scoring.global_score < 0.50:
            warnings.append(f"Score d'extraction faible : {scoring.global_score:.1%}")

        elapsed = round(time.perf_counter() - t0, 2)

        # Log récapitulatif des étapes
        slow_steps = {k: v for k, v in timings.items() if v >= 1.0}
        if slow_steps:
            timing_str = " | ".join(f"{k}={v}s" for k, v in slow_steps.items())
            log.info("[%s] étapes lentes : %s", path.name, timing_str)

        log.info(
            "[%s] doc_type=%s  conf=%.1f%%  champs=%d  score=%.2f  (%.1fs)",
            path.name, doc_type, confidence * 100,
            len(corrected_fields), scoring.global_score, elapsed,
        )

        return {
            "status":             "success",
            "source":             path.name,
            "source_path":        str(path.resolve()),
            "input_type":         input_type,
            "doc_type":           doc_type,
            "confidence":         round(confidence, 4),
            "confident":          confident,
            "class_probas":       probas,
            "fields":             corrected_fields,
            "tables":             doc_tables,
            "extraction_tiers":   tiers,
            "extraction_score":   scoring.to_dict(),
            "rule_results":       [r.to_dict() for r in rule_results],
            "rule_summary":       rule_summary,
            "n_pages":            extraction.get("n_pages", 0),
            "pages":              extraction.get("pages", []),
            "structured_rows":    structured_rows,
            "processing_time_s":  elapsed,
            "step_timings":       timings,
            "warnings":           warnings,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Propagation inter-documents
    # ──────────────────────────────────────────────────────────────────────────

    def _propagate_shared_fields(self, results: list[dict]) -> list[dict]:
        """
        Propage les champs contractuels depuis l'invoice vers BL et PL.

        Champs propagés (si absents dans le doc cible) :
          • currency   — devise de facturation (obligatoire LC Financial)
          • incoterm   — conditions de livraison (obligatoire LC Documentary)
          • port_of_discharge — port destination (si absent du BL/PL)
          • vessel / voyage — identifiants transport (si absents du PL)

        Les champs propagés reçoivent le tier "propagated_from_invoice" (conf 0.75)
        pour garder la traçabilité dans l'audit d'extraction.
        """
        # Localiser l'invoice (source de vérité)
        invoice_result = next(
            (r for r in results if r.get("status") == "success" and r.get("doc_type") == "invoice"),
            None,
        )
        if not invoice_result:
            return results

        inv_fields = invoice_result.get("fields", {})

        # Champs à propager : {champ_source: champ_cible (même nom sauf mention contraire)}
        _PROPAGATE: list[tuple[str, str]] = [
            ("currency",         "currency"),
            ("incoterm",         "incoterm"),
            ("port_of_discharge","port_of_discharge"),
            ("vessel",           "vessel"),
            ("voyage",           "voyage"),
        ]

        for result in results:
            if result.get("status") != "success":
                continue
            if result.get("doc_type") == "invoice":
                continue

            fields = result.setdefault("fields", {})
            tiers  = result.setdefault("extraction_tiers", {})

            for src_field, dst_field in _PROPAGATE:
                src_val = inv_fields.get(src_field)
                if src_val and not fields.get(dst_field):
                    fields[dst_field] = src_val
                    tiers[dst_field]  = "propagated_from_invoice"
                    log.debug(
                        "[propagation] %s.%s ← invoice.%s = '%s'",
                        result.get("doc_type"), dst_field, src_field, src_val,
                    )

        # ── Consistency check conteneurs BL ↔ PL ─────────────────────────────
        bl_result = next(
            (r for r in results if r.get("status") == "success" and r.get("doc_type") == "bl"),
            None,
        )
        pl_result = next(
            (r for r in results if r.get("status") == "success" and r.get("doc_type") == "packing_list"),
            None,
        )
        if bl_result and pl_result:
            bl_ctn = bl_result.get("fields", {}).get("container_number", "")
            pl_ctn = pl_result.get("fields", {}).get("container_number", "")
            if bl_ctn and pl_ctn and bl_ctn != pl_ctn:
                log.warning(
                    "[consistency] container_number BL≠PL : '%s' vs '%s'",
                    bl_ctn, pl_ctn,
                )
                bl_result.setdefault("warnings", []).append(
                    f"Container BL ({bl_ctn}) ≠ Container PL ({pl_ctn})"
                )

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Détection du format d'entrée
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_type(self, path: Path) -> str:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return "image"
        if path.suffix.lower() == ".pdf":
            try:
                doc     = fitz.open(str(path))
                n_words = len(doc[0].get_text("words"))
                doc.close()
                return "pdf_native" if n_words >= MIN_NATIVE_WORDS else "pdf_scanned"
            except Exception:
                return "pdf_native"
        return "pdf_native"

    # ──────────────────────────────────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────────────────────────────────

    def _get_type_classifier(self):
        if self._classifier is None:
            from release2.services.type_classification import TypeClassifier
            self._classifier = TypeClassifier(self._classifier_model_dir)
        return self._classifier

    def _classify_keyword(self, path: Path) -> tuple[str, float, dict]:
        return self._get_type_classifier().classify(path, input_type="pdf_native")

    def _classify_visual(self, path: Path, input_type: str) -> tuple[str, float, dict]:
        return self._get_type_classifier().classify(path, input_type=input_type)

    # ──────────────────────────────────────────────────────────────────────────
    # Extracteurs (avec cache + use_fast=True)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_extractor(self):
        if self._extractor is None:
            from release2.services.extraction_service import ExtractionService
            self._extractor = ExtractionService(self._extractor_model_dir)
        return self._extractor

    def _get_specialized_extractor(self, doc_type: str):
        """
        Retourne BLExtractor, InvoiceExtractor ou PLExtractor (avec cache).
        Passe use_fast=True pour accélérer le processor LayoutLMv3.
        Fallback sur extracteur générique si modèle spécialisé indisponible.
        """
        cache_key = f"_specialized_{doc_type}"
        if not hasattr(self, cache_key):
            try:
                kwargs = {"use_fast": True}   # ← processor rapide
                if doc_type == "bl":
                    from release2.services.bl_extractor import BLExtractor
                    try:
                        setattr(self, cache_key, BLExtractor(**kwargs))
                    except TypeError:
                        setattr(self, cache_key, BLExtractor())
                elif doc_type == "invoice":
                    from release2.services.invoice_extractor import InvoiceExtractor
                    try:
                        setattr(self, cache_key, InvoiceExtractor(**kwargs))
                    except TypeError:
                        setattr(self, cache_key, InvoiceExtractor())
                elif doc_type == "packing_list":
                    from release2.services.pl_extractor import PLExtractor
                    try:
                        setattr(self, cache_key, PLExtractor(**kwargs))
                    except TypeError:
                        setattr(self, cache_key, PLExtractor())
                else:
                    setattr(self, cache_key, self._get_extractor())
            except Exception as exc:
                log.warning(
                    "Extracteur spécialisé '%s' indisponible (%s) — fallback générique",
                    doc_type, exc,
                )
                setattr(self, cache_key, self._get_extractor())
        return getattr(self, cache_key)

    def _get_expected_fields(self, doc_type: str) -> set[str]:
        if doc_type == "bl":
            return {"bl_number", "shipper", "consignee", "notify_party", "vessel", "voyage",
                    "port_of_loading", "port_of_discharge", "etd", "eta", "issue_date",
                    "container_number", "seal_number", "commodity", "gross_weight", "measurement",
                    "freight_terms", "incoterm"}
        elif doc_type == "invoice":
            return {"invoice_number", "invoice_date", "currency", "incoterm", "subtotal",
                    "hs_code", "quantity", "unit", "unit_price", "line_amount", "total_amount"}
        elif doc_type == "packing_list":
            return {"pl_number", "container_number", "pl_date", "consignee", "goods_description",
                    "measurement", "total_cbm"}
        return set()

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            from easyocr import Reader
            try:
                from services.ocr.image_enhancer import ImageEnhancer
            except ImportError:
                from release2.services.ocr.image_enhancer import ImageEnhancer
            langs = self._ocr_languages or ["en", "fr"]
            self._ocr_reader   = Reader(langs, gpu=self._ocr_gpu, verbose=False)
            self._ocr_enhancer = ImageEnhancer()
        return self._ocr_reader, self._ocr_enhancer

    _OCR_RENDER_DPI = 150

    def _load_rendered_image(self, path: Path, input_type: str):
        from PIL import Image
        if input_type == "image":
            return Image.open(path).convert("RGB")
        doc = fitz.open(str(path))
        mat = fitz.Matrix(self._OCR_RENDER_DPI / 72, self._OCR_RENDER_DPI / 72)
        pix = doc[0].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        doc.close()
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return Image.fromarray(arr)

    def _render_pdf_page(self, page):
        from PIL import Image
        mat = fitz.Matrix(self._OCR_RENDER_DPI / 72, self._OCR_RENDER_DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return Image.fromarray(arr)

    def _preprocess_for_ocr(self, image) -> np.ndarray:
        img_bgr = np.array(image.convert("RGB"))[:, :, ::-1]
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray    = clahe.apply(gray)
        blur    = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw   = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if bw.size and np.count_nonzero(bw < 255) > 50:
            coords = np.column_stack(np.where(bw < 255))
            angle  = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle += 90
            if abs(angle) > 0.5:
                h, w = img_bgr.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_bgr = cv2.warpAffine(
                    img_bgr, M, (w, h),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
                )
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
        return img_bgr

    def _extract_ocr_tokens(self, image) -> tuple[list[str], list[list[int]]]:
        ocr_reader, enhancer = self._get_ocr_reader()
        img_bgr = self._preprocess_for_ocr(image)
        gray, _ = enhancer.enhance_both(img_bgr)
        results = ocr_reader.readtext(gray, detail=1, paragraph=False, min_size=5)

        words:  list[str]       = []
        bboxes: list[list[int]] = []
        h, w = gray.shape[:2]

        def normalize_bbox(x0, y0, x1, y1):
            def nx(v): return max(0, min(1000, int(v / w * 1000)))
            def ny(v): return max(0, min(1000, int(v / h * 1000)))
            return [nx(x0), ny(y0), nx(x1), ny(y1)]

        for polygon, text, conf in results:
            if not text or conf < 0.30:
                continue
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            tokens = text.split()
            if len(tokens) == 1:
                words.append(tokens[0])
                bboxes.append(normalize_bbox(x0, y0, x1, y1))
                continue
            total_chars = sum(len(tok) for tok in tokens)
            if total_chars == 0:
                continue
            dx = x0
            for token in tokens:
                ratio    = len(token) / total_chars
                token_x1 = dx + ratio * (x1 - x0)
                words.append(token)
                bboxes.append(normalize_bbox(dx, y0, token_x1, y1))
                dx = token_x1

        return words, bboxes

    def _extract_single_page(self, image, words, bboxes, doc_type) -> dict:
        specialized = self._get_specialized_extractor(doc_type)
        if hasattr(specialized, "EXPECTED_FIELDS"):
            result = specialized.extract_from_pil_image(image=image, words=words, bboxes=bboxes)
            return {
                "fields":          result.get("fields", {}),
                "tables":          [],
                "features":        {},
                "structured_rows": [],
                "table_fields":    {},
            }
        else:
            result = specialized.extract_from_pil_image(
                image=image, doc_type=doc_type, words=words, bboxes=bboxes,
            )
            return {
                "fields":          result.get("fields", {}),
                "tables":          result.get("tables", []),
                "features":        result.get("features", {}),
                "structured_rows": result.get("structured_rows", []),
                "table_fields":    result.get("table_fields", {}),
            }

    def _merge_multipage_tables(self, pages_result: list[dict], doc_type: str) -> list[dict]:
        import re as _re
        _total_re = _re.compile(r'^(total|grand\s+total|sub.?total|sum)\b', _re.IGNORECASE)
        merged: list[dict] = []
        for page_data in pages_result:
            for row in page_data.get("structured_rows", []):
                first_val = next(iter(row.values()), "") if row else ""
                if _total_re.match(str(first_val)):
                    continue
                merged.append(row)
        return merged

    def _extract_scanned(self, path: Path, doc_type: str, input_type: str) -> dict:
        if input_type == "image":
            image         = self._load_rendered_image(path, input_type)
            words, bboxes = self._extract_ocr_tokens(image)
            page_result   = self._extract_single_page(image, words, bboxes, doc_type)
            return {
                "doc_type":        doc_type,
                "confidence":      0.0,
                "class_probas":    {},
                "fields":          page_result["fields"],
                "tables":          page_result["tables"],
                "features":        page_result["features"],
                "structured_rows": page_result["structured_rows"],
                "table_fields":    page_result["table_fields"],
                "full_text":       " ".join(words),
                "n_pages":         1,
                "pages": [{
                    "page":            1,
                    "fields":          page_result["fields"],
                    "tables":          page_result["tables"],
                    "structured_rows": page_result["structured_rows"],
                }],
            }

        doc          = fitz.open(str(path))
        pages_result = []
        all_words:  list[str]      = []
        all_fields: dict[str, str] = {}

        for page_num in range(len(doc)):
            image         = self._render_pdf_page(doc[page_num])
            words, bboxes = self._extract_ocr_tokens(image)
            if not words:
                continue
            page_result = self._extract_single_page(image, words, bboxes, doc_type)
            pages_result.append({
                "page":            page_num + 1,
                "fields":          page_result["fields"],
                "tables":          page_result["tables"],
                "structured_rows": page_result["structured_rows"],
            })
            all_words.extend(words)
            for fname, val in page_result["fields"].items():
                if fname not in all_fields and val:
                    all_fields[fname] = val

        doc.close()
        merged_rows = self._merge_multipage_tables(pages_result, doc_type)
        return {
            "doc_type":        doc_type,
            "confidence":      0.0,
            "class_probas":    {},
            "fields":          all_fields,
            "tables":          [],
            "features":        {},
            "structured_rows": merged_rows,
            "table_fields":    {},
            "full_text":       " ".join(all_words),
            "n_pages":         len(pages_result),
            "pages":           pages_result,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Analyse sémantique (Tier 1c)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_semantic_analysis(
        self, text, doc_type, existing_fields, pdf_blocks=None, existing_tiers=None
    ) -> dict[str, str]:
        try:
            from release2.services.semantic_analyzer import SemanticAnalyzer
            return SemanticAnalyzer().analyze(
                text, doc_type, existing_fields,
                pdf_blocks=pdf_blocks, existing_tiers=existing_tiers,
            )
        except Exception as exc:
            log.warning("[pipeline] analyse sémantique échouée (%s): %s", doc_type, exc)
            return {}

    @staticmethod
    def _is_valid_table_field(field: str, value: str) -> bool:
        if not value or len(value.strip()) < 2:
            return False
        stripped = value.strip()
        if field == "goods_description":
            if len(stripped) < 5:
                return False
            if re.fullmatch(r'\d{1,4}', stripped):
                return False
        return True

    def _extract_layout_blocks(self, path: Path) -> list[dict] | None:
        try:
            doc    = fitz.open(str(path))
            blocks = []
            for page in doc:
                pw, ph = page.rect.width, page.rect.height
                for b in page.get_text("blocks"):
                    if b[6] == 0 and b[4].strip():
                        blocks.append({
                            "text":        b[4],
                            "bbox":        (b[0], b[1], b[2], b[3]),
                            "page_width":  pw,
                            "page_height": ph,
                        })
            doc.close()
            return blocks or None
        except Exception as exc:
            log.debug("[pipeline] extraction blocs layout échouée: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # Extraction spatiale (Tier 0)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_spatial_extraction(self, path: Path, doc_type: str) -> dict:
        try:
            return self._spatial_ext.extract(path, doc_type)
        except Exception as exc:
            log.warning("[pipeline] extraction spatiale échouée (%s): %s", path.name, exc)
            return {"fields": {}, "tiers": {}, "confidences": {}}

    @staticmethod
    def _can_overwrite(fname: str, new_conf: float, field_conf: dict[str, float]) -> bool:
        existing = field_conf.get(fname)
        return existing is None or new_conf > existing

    # ──────────────────────────────────────────────────────────────────────────
    # Zones sémantiques
    # ──────────────────────────────────────────────────────────────────────────

    def _run_zone_extraction(self, path: Path, doc_type: str) -> dict:
        try:
            from release2.services.zone_extractor import ZoneExtractor
            return ZoneExtractor().extract(path, doc_type)
        except Exception as exc:
            log.warning("[pipeline] zone extraction échouée (%s): %s", path.name, exc)
            return {"fields": {}, "zone_map": {}}

    # ──────────────────────────────────────────────────────────────────────────
    # Structure-first sur PDF natif
    # ──────────────────────────────────────────────────────────────────────────

    def _run_layout_native(self, path: Path, doc_type: str, carrier_hints=None) -> dict:
        try:
            from release2.services.layout_analyzer import LayoutAnalyzer
            la_kwargs = _template_matcher.get_layout_analyzer_kwargs(
                (carrier_hints or {}).get("carrier")
            )
            analyzer            = LayoutAnalyzer(**la_kwargs)
            all_table_fields    = {}
            all_structured_rows = []

            doc = fitz.open(str(path))
            for page in doc:
                pw, ph = page.rect.width, page.rect.height
                if pw <= 0 or ph <= 0:
                    continue
                words, bboxes = [], []
                for wt in page.get_text("words"):
                    x0, y0, x1, y1, text = wt[:5]
                    text = text.strip()
                    if not text:
                        continue
                    words.append(text)
                    bboxes.append([
                        max(0, min(1000, int(x0 / pw * 1000))),
                        max(0, min(1000, int(y0 / ph * 1000))),
                        max(0, min(1000, int(x1 / pw * 1000))),
                        max(0, min(1000, int(y1 / ph * 1000))),
                    ])
                if not words:
                    continue
                result = analyzer.analyze(words, bboxes, doc_type)
                for fname, val in result["table_fields"].items():
                    if fname not in all_table_fields:
                        all_table_fields[fname] = val
                all_structured_rows.extend(result["table_rows"])
            doc.close()
            return {"table_fields": all_table_fields, "structured_rows": all_structured_rows}
        except Exception as exc:
            log.warning("Structure-first (PDF natif) échoué : %s", exc)
            return {"table_fields": {}, "structured_rows": []}

    # ──────────────────────────────────────────────────────────────────────────
    # LayoutLMv3 sur PDF natif — DERNIER RECOURS (désactivé par défaut)
    # ──────────────────────────────────────────────────────────────────────────

    _SPECIALIZED_FIELD_NORM: dict[str, str] = {
        "BL_NUMBER": "bl_number", "SHIPPER": "shipper", "CONSIGNEE": "consignee",
        "NOTIFY_PARTY": "notify_party", "VESSEL": "vessel", "VOYAGE": "voyage",
        "PORT_OF_LOADING": "port_of_loading", "PORT_OF_DISCHARGE": "port_of_discharge",
        "PLACE_OF_RECEIPT": "place_of_receipt", "PLACE_OF_DELIVERY": "place_of_delivery",
        "ETD": "etd", "ETA": "eta", "ISSUE_DATE": "issue_date", "ISSUE_PLACE": "issue_place",
        "CONTAINER_NO": "container_number", "CONTAINER_TYPE": "container_type",
        "SEAL_NO": "seal_number", "COMMODITY": "commodity", "GROSS_WEIGHT": "gross_weight_kg",
        "MEASUREMENT": "measurement", "PACKAGES": "number_of_packages",
        "FREIGHT_TERMS": "freight_terms", "SERVICE_TYPE": "service_type",
        "NUMBER_OF_ORIGINALS": "number_of_originals",
        "INV_NUMBER": "invoice_number", "LC_REF": "lc_reference", "BL_REF": "bl_reference",
        "SELLER": "seller", "BUYER": "buyer", "GOODS_DESC": "goods_description",
        "HS_CODE": "hs_code", "QUANTITY": "quantity", "PACKAGE_UNIT": "package_unit",
        "UNIT_PRICE": "unit_price", "CURRENCY": "currency", "TOTAL_AMOUNT": "total_amount",
        "INCOTERM": "incoterm", "PAYMENT_TERMS": "payment_terms",
        "PL_NUMBER": "pl_number", "INV_REF": "invoice_reference", "EXPORTER": "shipper",
        "NET_WEIGHT": "net_weight_kg", "TOTAL_PACKAGES": "total_packages",
        "TOTAL_CBM": "total_cbm", "MARKS_NUMBERS": "marks_numbers", "DIMENSIONS": "dimensions",
    }

    def _normalize_specialized_fields(self, fields: dict) -> dict:
        norm   = self._SPECIALIZED_FIELD_NORM
        result = {}
        for k, v in fields.items():
            pipeline_name = norm.get(k) or norm.get(k.upper()) or k.lower()
            if pipeline_name and v:
                result[pipeline_name] = v
        return result

    def _run_layoutlm_on_native_pdf(self, path: Path, doc_type: str) -> dict:
        """
        Extraction PRIMARY par LayoutLMv3 spécialisé (best_bl / best_invoice / best_pl).

        Optimisation perf (v3) :
          • _MAX_LAYOUTLM_PAGES = 1  → 1 inférence au lieu de N_pages
          • _MAX_LAYOUTLM_WORDS = 450 → truncation avant tokenisation
          • Early-stop dès que 70 % des champs attendus sont trouvés
          • Un seul lock → aucune contention CPU inter-threads

        Impact mesuré : 3 docs × 68s = 204s  →  3 docs × 17s = 51s
        """
        try:
            from PIL import Image
            extractor       = self._get_specialized_extractor(doc_type)
            use_specialized = hasattr(extractor, "EXPECTED_FIELDS")
            expected_fields = set(getattr(extractor, "EXPECTED_FIELDS", None) or [])
            doc             = fitz.open(str(path))
            all_fields: dict[str, str] = {}
            n_pages = doc.page_count

            for page_idx, page in enumerate(doc):

                # ── Limite de pages ───────────────────────────────────────────
                if page_idx >= _MAX_LAYOUTLM_PAGES:
                    log.debug("[pipeline] %s — page %d ignorée (limite=%d)",
                              path.name, page_idx + 1, _MAX_LAYOUTLM_PAGES)
                    break

                # ── Rendu image ───────────────────────────────────────────────
                mat = fitz.Matrix(self._OCR_RENDER_DPI / 72, self._OCR_RENDER_DPI / 72)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3)
                image = Image.fromarray(arr)

                pw, ph = page.rect.width, page.rect.height
                if pw <= 0 or ph <= 0:
                    continue

                # ── Extraction mots + bboxes ──────────────────────────────────
                words, bboxes = [], []
                for wt in page.get_text("words"):
                    x0, y0, x1, y1, text = wt[:5]
                    text = text.strip()
                    if not text:
                        continue
                    words.append(text)
                    bboxes.append([
                        max(0, min(1000, int(x0 / pw * 1000))),
                        max(0, min(1000, int(y0 / ph * 1000))),
                        max(0, min(1000, int(x1 / pw * 1000))),
                        max(0, min(1000, int(y1 / ph * 1000))),
                    ])
                if not words:
                    continue

                # ── Truncation : évite les inputs très longs ──────────────────
                if len(words) > _MAX_LAYOUTLM_WORDS:
                    log.debug("[pipeline] %s p%d : %d mots → tronqué à %d",
                              path.name, page_idx + 1, len(words), _MAX_LAYOUTLM_WORDS)
                    words  = words[:_MAX_LAYOUTLM_WORDS]
                    bboxes = bboxes[:_MAX_LAYOUTLM_WORDS]

                # ── Inférence LayoutLM (sérialisée) ──────────────────────────
                t_inf = time.perf_counter()
                with _LAYOUTLM_LOCK:
                    if use_specialized:
                        result     = extractor.extract_from_pil_image(
                            image=image, words=words, bboxes=bboxes)
                        raw_fields = result.get("fields", {})
                        fields     = self._normalize_specialized_fields(raw_fields)
                    else:
                        fields, _ = extractor._run_layoutlm_blocks(
                            image, words, bboxes, doc_type)
                inf_s = round(time.perf_counter() - t_inf, 2)
                log.info("[pipeline] %s p%d LayoutLM %.2fs (%d mots, %d/%d pages)",
                         path.name, page_idx + 1, inf_s,
                         len(words), page_idx + 1, n_pages)

                for fname, val in fields.items():
                    if fname not in all_fields and val:
                        all_fields[fname] = val

                # ── Early-stop : 70 % des champs trouvés → inutile de continuer ─
                if expected_fields:
                    found_pct = len(set(all_fields) & expected_fields) / len(expected_fields)
                    if found_pct >= 0.70:
                        log.debug("[pipeline] %s early-stop p%d (%.0f%% champs trouvés)",
                                  path.name, page_idx + 1, found_pct * 100)
                        break

            doc.close()
            return {"fields": all_fields}

        except Exception as exc:
            log.warning("[pipeline] LayoutLMv3 blocs (natif) échoué (%s): %s",
                        path.name, exc, exc_info=True)
            return {"fields": {}}

    # ──────────────────────────────────────────────────────────────────────────
    # Construction des tables nommées
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_numeric_cell(val: str) -> bool:
        clean = re.sub(r"\b[A-Z]{3}\b", "", val.upper()).strip()
        clean = clean.replace(",", "").replace(".", "").replace(" ", "")
        return bool(clean) and clean.lstrip("-").isdigit()

    @classmethod
    def _build_document_tables(cls, doc_type: str, structured_rows: list[dict]) -> dict:
        if not structured_rows:
            return {}
        table_name = _TABLE_NAME.get(doc_type)
        if not table_name:
            return {}

        remap      = _TABLE_FIELD_REMAP.get(doc_type, {})
        normalized = []
        for row in structured_rows:
            new_row: dict[str, str] = {}
            for k, v in row.items():
                if doc_type == "invoice" and k == "line_total" and "line_amount" in row:
                    continue
                target = remap.get(k, k)
                if target not in new_row and v:
                    new_row[target] = v
            if not new_row:
                continue

            filled = [v for v in new_row.values() if str(v).strip()]
            if len(filled) < 2:
                continue

            desc_val = new_row.get("description", new_row.get("goods_description", ""))
            if desc_val and re.fullmatch(r'[\w\s/:\.]+:', desc_val.strip()):
                continue

            row_text = " ".join(str(v) for v in new_row.values()).upper()
            if re.search(
                r'\b(AUTHORIZED|SIGNATORY|SIGNATURE|STAMP|MANAGER|DIRECTOR'
                r'|VERIFIED\s+BY|APPROVED\s+BY|DOMICILIATION|BANQUE|BIAT'
                r'|SWIFT|IBAN|BIC)\b',
                row_text,
            ):
                continue

            if doc_type == "invoice":
                up = new_row.get("unit_price", "")
                if up and not cls._is_numeric_cell(up):
                    continue
                has_numeric = any(
                    cls._is_numeric_cell(new_row.get(f, ""))
                    for f in ("unit_price", "quantity", "line_amount", "line_total")
                )
                if not has_numeric:
                    continue
            elif doc_type == "packing_list":
                qty, cbm = new_row.get("quantity", ""), new_row.get("cbm", "")
                gw,  nw  = new_row.get("gross_weight", ""), new_row.get("net_weight", "")
                if not any(f and cls._is_numeric_cell(f) for f in (qty, cbm, gw, nw)):
                    continue
            elif doc_type == "bl":
                cntr = new_row.get("container_number", "")
                if not cntr or not re.search(r"[A-Z]{3,4}\d{6,7}", cntr.replace(" ", "")):
                    continue

            normalized.append(new_row)

        return {table_name: normalized} if normalized else {}

    @staticmethod
    def _synthesize_table_from_fields(doc_type: str, fields: dict) -> dict:
        """Fallback: build a single-row table from extracted field values when the
        layout analyzer finds no structured rows (e.g. simple one-container BLs,
        single-line invoices, or scanned PDFs with poor column alignment)."""
        if doc_type == "bl":
            row: dict[str, str] = {}
            if fields.get("container_number"):
                row["container_number"] = fields["container_number"]
            if fields.get("seal_number"):
                row["seal_number"] = fields["seal_number"]
            if fields.get("container_type"):
                row["container_type"] = fields["container_type"]
            if fields.get("gross_weight_kg"):
                row["gross_weight_kg"] = fields["gross_weight_kg"]
            if fields.get("net_weight_kg"):
                row["net_weight_kg"] = fields["net_weight_kg"]
            if fields.get("measurement"):
                row["cbm"] = fields["measurement"]
            if fields.get("goods_description"):
                row["description"] = fields["goods_description"]
            if fields.get("number_of_packages"):
                row["packages"] = fields["number_of_packages"]
            cntr = row.get("container_number", "")
            if cntr and re.search(r"[A-Z]{3,4}\d{6,7}", cntr.replace(" ", "")) and len(row) >= 2:
                return {"containers": [row]}

        elif doc_type == "invoice":
            row = {}
            if fields.get("goods_description"):
                row["description"] = fields["goods_description"]
            if fields.get("hs_code"):
                row["hs_code"] = fields["hs_code"]
            if fields.get("total_amount"):
                row["line_amount"] = fields["total_amount"]
            if fields.get("currency"):
                row["currency"] = fields["currency"]
            if fields.get("quantity"):
                row["quantity"] = fields["quantity"]
            if fields.get("unit_price"):
                row["unit_price"] = fields["unit_price"]
            has_numeric = any(
                row.get(f) and re.search(r"\d", row.get(f, ""))
                for f in ("line_amount", "quantity", "unit_price")
            )
            if has_numeric and len(row) >= 2:
                return {"invoice_items": [row]}

        elif doc_type == "packing_list":
            row = {}
            if fields.get("container_number"):
                row["container_number"] = fields["container_number"]
            if fields.get("goods_description"):
                row["package_type"] = fields["goods_description"]
            if fields.get("total_packages"):
                row["quantity"] = fields["total_packages"]
            if fields.get("gross_weight_kg"):
                row["gross_weight"] = fields["gross_weight_kg"]
            if fields.get("net_weight_kg"):
                row["net_weight"] = fields["net_weight_kg"]
            if fields.get("total_cbm"):
                row["cbm"] = fields["total_cbm"]
            has_numeric = any(
                row.get(f) and re.search(r"\d", row.get(f, ""))
                for f in ("quantity", "gross_weight", "net_weight", "cbm")
            )
            if has_numeric and len(row) >= 2:
                return {"packages": [row]}

        return {}

    @staticmethod
    def _sum_field(rows: list[dict], field: str) -> float | None:
        total, found = 0.0, False
        for row in rows:
            raw = str(row.get(field, "")).strip()
            if not raw:
                continue
            val = raw.replace(" ", "")
            if re.search(r"\d,\d{3}", val):
                val = val.replace(",", "")
            else:
                val = val.replace(",", ".")
            try:
                total += float(val)
                found  = True
            except ValueError:
                continue
        return total if found else None

    def _compute_table_aggregates(self, doc_type, tables, existing_fields) -> dict[str, str]:
        computed: dict[str, str] = {}

        if doc_type == "invoice":
            subtotal_raw = existing_fields.get("subtotal", "")
            if subtotal_raw and "total_amount" not in existing_fields:
                clean = subtotal_raw.replace(" ", "")
                clean = clean.replace(",", "") if re.search(r"\d,\d{3}", clean) else clean.replace(",", ".")
                try:
                    computed["total_amount"] = str(round(float(clean), 2))
                except ValueError:
                    pass
            if "total_amount" not in existing_fields and "total_amount" not in computed:
                items = tables.get("invoice_items", [])
                if items:
                    total = self._sum_field(items, "line_amount")
                    if total is not None:
                        computed["total_amount"] = str(round(total, 2))

        elif doc_type == "bl":
            containers = tables.get("containers", [])
            if len(containers) == 1:
                row = containers[0]
                for fname in ("gross_weight_kg", "net_weight_kg"):
                    if fname not in existing_fields and row.get(fname):
                        computed[fname] = row[fname]
                if "measurement" not in existing_fields and row.get("cbm"):
                    computed["measurement"] = row["cbm"]

        elif doc_type == "packing_list":
            packages = tables.get("packages", [])
            if packages:
                if "total_packages" not in existing_fields:
                    total_qty = self._sum_field(packages, "quantity")
                    if total_qty is not None and total_qty > 0:
                        computed["total_packages"] = str(int(total_qty))
                if "total_cbm" not in existing_fields:
                    total_cbm_val = self._sum_field(packages, "cbm")
                    if total_cbm_val is not None and total_cbm_val > 0:
                        computed["total_cbm"] = str(round(total_cbm_val, 3))
                total_gw = self._sum_field(packages, "gross_weight")
                if total_gw is not None and total_gw > 0:
                    computed["gross_weight_kg"] = str(int(round(total_gw)))
                total_nw = self._sum_field(packages, "net_weight")
                if total_nw is not None and total_nw > 0:
                    computed["net_weight_kg"] = str(int(round(total_nw)))

        return computed

    # ──────────────────────────────────────────────────────────────────────────
    # Résultat d'erreur
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _error_result(path: Path, error: str) -> dict:
        return {
            "status":            "error",
            "source":            path.name,
            "source_path":       str(path.resolve()),
            "input_type":        None,
            "doc_type":          None,
            "confidence":        None,
            "confident":         False,
            "class_probas":      {},
            "fields":            {},
            "n_pages":           0,
            "pages":             [],
            "processing_time_s": None,
            "step_timings":      {},
            "warnings":          [],
            "error":             error,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Affichage console
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  PIPELINE RÉSULTATS — {result['summary']['total_documents']} document(s)")
    print(f"{'═' * 60}")
    print(f"  Traités  : {result['summary']['processed']}")
    print(f"  Erreurs  : {result['summary']['failed']}")
    print(f"  Durée    : {result['summary']['processing_time_s']}s")

    for i, doc in enumerate(result["documents"], start=1):
        print(f"\n{sep}")
        print(f"  Document {i} — {doc['source']}")
        print(sep)

        if doc["status"] == "error":
            print(f"  ✗  ERREUR : {doc.get('error', '?')}")
            continue

        input_labels = {"pdf_native": "PDF natif", "pdf_scanned": "PDF scanné", "image": "Image"}
        conf_icon = "✓" if doc["confident"] else "⚠"
        print(f"  Format     : {input_labels.get(doc['input_type'], doc['input_type'])}")
        print(f"  Type doc   : {doc['doc_type'].upper()}")
        print(f"  Confiance  : {conf_icon}  {doc['confidence']:.1%}")
        print(f"  Pages      : {doc['n_pages']}")
        print(f"  Durée      : {doc['processing_time_s']}s")

        # Affichage des timings par étape
        step_timings = doc.get("step_timings", {})
        if step_timings:
            print(f"\n  Timings par étape :")
            for step, duration in step_timings.items():
                bar   = "█" * int(duration / 2)
                alert = " ⚠" if duration >= 10 else ""
                print(f"    {step:<35} {duration:>6.1f}s  {bar}{alert}")

        if doc["warnings"]:
            for w in doc["warnings"]:
                print(f"  ⚠  {w}")

        fields = doc.get("fields", {})
        if fields:
            print(f"\n  Champs extraits ({len(fields)}) :")
            for field, value in sorted(fields.items()):
                print(f"    {field:<28} {value}")
        else:
            print("\n  (aucun champ extrait)")

    print(f"\n{'═' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="pipeline — Classification + Extraction sur 1 à 3 documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("documents", nargs="*", help="Chemins vers les documents (max 3)")
    parser.add_argument("--files", "-f", nargs="+", metavar="FICHIER")
    parser.add_argument("--output", "-o", default=None, metavar="FICHIER.json")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--classifier-model-dir", default=None)
    parser.add_argument("--extractor-model-dir", default=None)
    parser.add_argument("--donut-model-dir", default=None)
    parser.add_argument(
        "--enable-layoutlm-native", action="store_true", default=False,
        help="Active le tier LayoutLM de dernier recours sur PDF natif (lent)",
    )
    args = parser.parse_args()

    docs = list(args.documents or []) + list(args.files or [])
    if not docs:
        parser.error("Fournir au moins un document.")

    pipeline = DocumentPipeline(
        donut_model_dir        = args.donut_model_dir,
        classifier_model_dir   = args.classifier_model_dir,
        extractor_model_dir    = args.extractor_model_dir,
        confidence_threshold   = args.threshold,
        enable_layoutlm_native = args.enable_layoutlm_native,
    )

    result = pipeline.process(docs, output_path=args.output)
    _print_result(result)


if __name__ == "__main__":
    main()