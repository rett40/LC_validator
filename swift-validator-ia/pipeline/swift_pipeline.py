"""
swift_pipeline.py  —  v6.6
============================
Pipeline hybride complet : Règles + NLP + Analyse sémantique + Validation externe + Risk ML

MODIFICATIONS v6.6 (par rapport à v6.5) :
  ✦ RISK-01 : self._risk_engine singleton lazy dans __init__
  ✦ RISK-02 : propriété risk_engine — charge RiskMLEngine
  ✦ RISK-03 : _enrich_with_risk_ml() — appel après _enrich_with_external()
  ✦ RISK-04 : appel de _enrich_with_risk_ml() dans process() et analyze_text()
  ✦ RISK-05 : /health expose risk_ml_engine status (géré par l'application)

MODIFICATIONS v6.5 :
  ✦ CACHE-01 : Démarrage du scheduler APScheduler dans __init__
               → refresh automatique des listes de sanctions toutes les 30 jours
               → scraping en arrière-plan au premier lancement si cache absent
               → dégradation gracieuse si APScheduler non installé
  ✦ CACHE-02 : Les checkers (country, bank, company) utilisent automatiquement
               le cache JSON local via cache_manager.py — fallback API transparent

MODIFICATIONS v6.4 :
  ✦ MRG-01 : NER_HIGH_THRESHOLD = 0.80 — NER écrase le parser si conf ≥ seuil
  ✦ MRG-02 : NER_LOW_THRESHOLD  = 0.70 — NER ignoré si conf < seuil
  ✦ MRG-03 : _merge_entities refactorisé — logique NER-first sur TOUS les champs
               • conf ≥ 0.80 → NER prime (même si parser avait une valeur)
               • conf ∈ [0.70, 0.80[ → parser prioritaire ; NER = fallback si vide
               • conf < 0.70 → NER ignoré (tracé dans _<field>_ner_ignored)
  ✦ MRG-04 : PORT / AMOUNT appliquent les mêmes seuils
  ✦ MRG-05 : méta-clés de traçabilité (_override_reason, _ner_candidate, _ner_ignored)

MODIFICATIONS v6.3 :
  ✦ EXT-01 : self._ext_engine singleton lazy dans __init__
  ✦ EXT-02 : propriété ext_engine — charge ExternalValidationEngine (approche B)
  ✦ EXT-03 : _enrich_with_external() — appel après _enrich_with_ucp_and_ds()
  ✦ EXT-04 : appel de _enrich_with_external() dans process()

MODIFICATIONS v6.2 (patches bugfix) :
  ✦ PL-01 : geo_coherence default corrigé → {} (était hardcodé True)
  ✦ PL-02 : double port NLP — PORT[0]→port_loading, PORT[1]→port_discharge
  ✦ PL-03 : double passe spell éliminée — texte pré-corrigé passé au NLP
  ✦ PL-04 : _EntityDictAdapter.__getattr__ robustifié (hasattr safe)

MODIFICATIONS v6.1 :
  ✦ NLPServiceConfig enrichi avec les modèles entraînés (clause-clf, risk-clf)
  ✦ enable_embeddings, enable_semantic_coherence, detect_clause_contradictions activés
  ✦ Chemins des modèles résolus dynamiquement depuis la racine du projet
  ✦ Log de démarrage indiquant si les modèles ML sont chargés ou non
  ✦ _serialize_nlp enrichi : contradictions, module_timings, doc_quality_score

Pipeline en 9 étapes :
  [OCR]  Extraction texte depuis PDF/image
  [SPL]  Correction orthographique (OCRService + spell_correction)
  [DET]  Détection type MT*** (SwiftDetector)
  [EXT]  Extraction règles métier (parseurs dédiés MT700…MT799)
  [LC]   Critères d'acceptation LC (LCAcceptanceCriteriaExtractor)
  [NLP]  NER + Classification clauses + Analyse sémantique (SwiftNLPService)
  [MRG]  Fusion règles + NLP (NER-first avec seuils de confiance)
  [VAL]  Validation SWIFT (SwiftValidator)
  [EXT]  Validation externe (OFAC, GLEIF, EU, UN, OpenCorporates, Nominatim)
         → v6.5 : lecture cache local JSON en priorité, fallback API si absent
  [RISK] Risk ML (XGBoost + IsolationForest + KMeans + SHAP) — v6.6

Variables d'environnement (v6.6) :
  SANCTIONS_DATA_DIR     — répertoire des fichiers JSON (défaut: data/scraping)
  CACHE_MAX_AGE_DAYS     — durée de validité du cache en jours (défaut: 30)
  INCLUDE_GLEIF_SNAPSHOT — activer snapshot GLEIF, lent (défaut: 0)
  OPENCORPORATES_API_KEY — clé API OpenCorporates (optionnel)
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

# ── Seuils NER-first (MRG-01 / MRG-02) ───────────────────────────────
# conf ≥ NER_HIGH_THRESHOLD  → NER écrase le parser
# conf ∈ [NER_LOW_THRESHOLD, NER_HIGH_THRESHOLD[ → parser prioritaire ; NER = fallback
# conf < NER_LOW_THRESHOLD   → NER ignoré
NER_HIGH_THRESHOLD = 0.80
NER_LOW_THRESHOLD  = 0.70

# Lazy singletons DS/UCP (chargés au premier appel, pas au démarrage Flask)
_ucp_validator_singleton = None
_ds_extractor_singleton  = None
_ds_scorer_singleton     = None

logger = logging.getLogger(__name__)


class SwiftPipeline:

    def __init__(self):
        # ── OCR ───────────────────────────────────────────────────────
        from services.ocr.ocr_service import OCRService
        self.ocr = OCRService()

        # ── Pipeline SWIFT (règles) ────────────────────────────────────
        from services.swift.swift_service import SwiftService
        self.swift_service = SwiftService()

        # ── Service NLP ───────────────────────────────────────────────
        self._nlp_service = None

        # ── EXT-01 : Validation externe (lazy) ────────────────────────
        self._ext_engine = None

        # ── RISK-01 : Risk ML Engine (lazy) ───────────────────────────
        self._risk_engine = None

        # ── CACHE-01 : Démarrage du scheduler de mise à jour des listes ──
        # Le scheduler vérifie si le cache JSON est absent ou expiré (>30j).
        # Si oui, lance run_scraper() en arrière-plan immédiatement,
        # puis programme un refresh automatique toutes les CACHE_MAX_AGE_DAYS jours.
        # Le pipeline continue même si le cache est en cours de construction
        # (fallback API transparent dans les checkers).
        try:
            # FIX CACHE-01 : chemin corrigé → services.external_validation_b.scheduler
            from services.external_validation.scheduler import start_scheduler
            _started = start_scheduler()
            if _started:
                logger.info(
                    "Cache sanctions démarré — refresh automatique toutes les %s jours "
                    "(CACHE_MAX_AGE_DAYS). Données : OFAC, EU, UN, FATF, RestCountries.",
                    # FIX CACHE-01b : défaut aligné sur scheduler.py (30 jours, pas 15)
                    os.environ.get("CACHE_MAX_AGE_DAYS", "30"),
                )
            else:
                logger.info(
                    "Cache sanctions : APScheduler non disponible — "
                    "installer avec : pip install apscheduler. "
                    "La validation externe utilisera les APIs web directement."
                )
        except ImportError as exc:
            # CACHE-02 : dégradation gracieuse si le module n'est pas encore installé
            logger.warning(
                "scheduler non disponible (%s) — "
                "la validation externe utilisera les APIs web directement.",
                exc,
            )
        except Exception as exc:
            # Ne jamais bloquer le démarrage de la pipeline
            logger.warning("Erreur démarrage scheduler cache : %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Propriété lazy NLP — v6.1 : modèles ML injectés
    # ──────────────────────────────────────────────────────────────────

    @property
    def nlp(self):
        if self._nlp_service is None:
            from services.swift_nlp import SwiftNLPService, NLPServiceConfig

            _root        = Path(__file__).resolve().parent.parent
            _models_dir  = _root / "models"
            _clause_path = _models_dir / "clause-clf"
            _risk_path   = _models_dir / "risk-clf"

            _ner_path  = _models_dir / "swift-ner"
            _ner_model = str(_ner_path) if _ner_path.exists() else None

            cfg = NLPServiceConfig(
                enable_ner                   = True,
                enable_clause_classifier     = True,
                enable_semantic_analysis     = True,
                ner_confidence_threshold     = 0.65,
                clause_confidence_threshold  = 0.35,
                enable_embeddings            = True,
                clause_model_path            = str(_clause_path) if _clause_path.exists() else None,
                risk_model_path              = str(_risk_path)   if _risk_path.exists()   else None,
                transformer_ner_model        = _ner_model,
                enable_semantic_coherence    = True,
                enable_cache                 = True,
                cache_size                   = 128,
                detect_clause_contradictions = True,
            )

            self._nlp_service = SwiftNLPService(config=cfg)

            logger.info(
                "SwiftNLPService v2 chargé — "
                "clause_model=%s | risk_model=%s | ner_model=%s | embeddings=on | contradictions=on",
                "OK " + str(_clause_path) if _clause_path.exists() else "ABSENT",
                "OK " + str(_risk_path)   if _risk_path.exists()   else "ABSENT",
                "OK swift-ner" if _ner_path.exists() else "bert-base-NER (généraliste)",
            )

        return self._nlp_service

    # ──────────────────────────────────────────────────────────────────
    # EXT-02 : Propriété lazy ExternalValidationEngine (approche B)
    # ──────────────────────────────────────────────────────────────────

    @property
    def ext_engine(self):
        """
        Charge ExternalValidationEngine (approche B) au premier appel.

        v6.5 : les checkers sous-jacents (country, bank, company) lisent
        en priorité le cache JSON local (data/scraping/latest/*.json) produit par
        scraper.py et maintenu par scheduler.py.
        Fallback automatique vers les APIs web si le cache est absent.

        APIs utilisées (directement ou via cache) :
          OFAC SDN, GLEIF, EU Sanctions XML, UN XML,
          OpenCorporates, Nominatim/OSM, REST Countries, FATF.
        """
        if self._ext_engine is None:
            try:
                # FIX EXT-02 : chemin corrigé → services.external_validation_b
                # L'ancien import (services.external_validation) ne correspondait
                # pas au package réel external_validation_b, provoquant une
                # ImportError silencieuse et désactivant toute la validation externe.
                from services.external_validation import ExternalValidationEngine
                self._ext_engine = ExternalValidationEngine()
                logger.info("ExternalValidationEngine (approche B + cache local) initialisé")
            except ImportError as exc:
                logger.warning(
                    "external_validation_b non disponible : %s — "
                    "vérifier que le dossier external_validation_b/ est dans PYTHONPATH",
                    exc,
                )
                self._ext_engine = False   # sentinel : ne pas re-tenter
        return self._ext_engine or None

    # ──────────────────────────────────────────────────────────────────
    # RISK-02 : Propriété lazy RiskMLEngine
    # ──────────────────────────────────────────────────────────────────

    @property
    def risk_engine(self):
        """
        Charge RiskMLEngine au premier appel.
        Retourne None si les modèles ne sont pas encore entraînés.
        """
        if self._risk_engine is None:
            try:
                from services.risk_service.risk_engine import get_risk_engine
                engine = get_risk_engine()
                if engine.is_available():
                    self._risk_engine = engine
                    logger.info("RiskMLEngine chargé — modèles disponibles dans %s", engine._models_dir)
                else:
                    logger.warning(
                        "RiskMLEngine : modèles introuvables dans %s — "
                        "lancer scripts/train_models.py pour entraîner.",
                        engine._models_dir,
                    )
                    self._risk_engine = False   # sentinel
            except ImportError as exc:
                logger.warning("risk_engine non disponible : %s", exc)
                self._risk_engine = False
            except Exception as exc:
                logger.warning("Erreur chargement RiskMLEngine : %s", exc)
                self._risk_engine = False
        return self._risk_engine if self._risk_engine else None

    # ──────────────────────────────────────────────────────────────────
    # Modules intelligents — lazy load
    # ──────────────────────────────────────────────────────────────────

    def _get_ucp_validator(self):
        global _ucp_validator_singleton
        if _ucp_validator_singleton is None:
            try:
                from services.swift_nlp.ucp.swift_ucp_validator import UCP600Validator
                _ucp_validator_singleton = UCP600Validator()
                logger.info("UCP600Validator initialisé")
            except Exception as exc:
                logger.warning("UCP600Validator non disponible : %s", exc)
                _ucp_validator_singleton = False
        return _ucp_validator_singleton or None

    def _get_ds_modules(self):
        global _ds_extractor_singleton, _ds_scorer_singleton
        if _ds_extractor_singleton is None:
            try:
                from services.swift_nlp.ds.swift_ds_pipeline import LCFeatureExtractor, LCRiskScorer
                _ds_extractor_singleton = LCFeatureExtractor()
                _ds_scorer_singleton    = LCRiskScorer.load(
                    str(Path(__file__).resolve().parent.parent / "models" / "lc-risk-ds")
                )
                logger.info("DS modules initialisés")
            except Exception as exc:
                logger.warning("DS modules non disponibles : %s", exc)
                _ds_extractor_singleton = False
        return (_ds_extractor_singleton or None, _ds_scorer_singleton or None)

    # ──────────────────────────────────────────────────────────────────
    # API publique
    # ──────────────────────────────────────────────────────────────────

    def process(self, file_path: str) -> dict:
        """Pipeline complet depuis un fichier (PDF ou image)."""
        try:
            ocr_result = self.ocr.process(file_path)
        except Exception as exc:
            logger.error("OCR failed: %s", exc, exc_info=True)
            return {
                "success": False,
                "error":   f"Erreur OCR : {exc}",
                "ocr":     {"success": False, "text": "", "confidence": 0.0},
                "swift":   None,
            }

        ocr_text       = ocr_result.get("text", "")
        ocr_confidence = ocr_result.get("confidence", 0.0)

        ocr_payload = {
            "success":    bool(ocr_text and ocr_text.strip()),
            "text":       ocr_text,
            "confidence": ocr_confidence,
        }

        if not ocr_text.strip():
            return {
                "success": False,
                "error":   "OCR n'a produit aucun texte",
                "ocr":     ocr_payload,
                "swift":   None,
            }

        swift_result = self._run_hybrid(
            ocr_text,
            ocr_confidence=ocr_confidence,
            spell_report=ocr_result.get("spell_report"),
        )

        pipeline_payload = {
            "success": swift_result.get("success", False),
            "ocr":     ocr_payload,
            "swift":   swift_result,
        }

        if swift_result.get("success"):
            self._enrich_with_ucp_and_ds(pipeline_payload)
            # EXT-04 : validation externe après UCP + DS
            self._enrich_with_external(pipeline_payload)
            # RISK-04 : Risk ML après external validation
            self._enrich_with_risk_ml(pipeline_payload)
            # Réconciliation : verdict final unique (hard override UCP prioritaire)
            self._resolve_final_decision(pipeline_payload)

        return pipeline_payload

    def analyze_text(self, text: str) -> dict:
        """Analyse SWIFT hybride sur texte brut (sans OCR)."""
        if not text or not text.strip():
            return {"success": False, "error": "Texte vide"}
        result = self._run_hybrid(text, ocr_confidence=1.0)

        pipeline_payload = {
            "success": result.get("success", False),
            "ocr":     {"success": True, "text": text, "confidence": 1.0},
            "swift":   result,
        }

        if result.get("success"):
            self._enrich_with_ucp_and_ds(pipeline_payload)
            # EXT-04 : validation externe après UCP + DS
            self._enrich_with_external(pipeline_payload)
            # RISK-04 : Risk ML après external validation
            self._enrich_with_risk_ml(pipeline_payload)
            # Réconciliation : verdict final unique (hard override UCP prioritaire)
            self._resolve_final_decision(pipeline_payload)

        return pipeline_payload

    # ──────────────────────────────────────────────────────────────────
    # EXT-03 : Enrichissement validation externe
    # ──────────────────────────────────────────────────────────────────

    def _enrich_with_external(self, pipeline_payload: dict) -> None:
        """
        Enrichit le résultat du pipeline avec la validation externe (approche B).

        v6.5 : les checkers lisent le cache JSON local en priorité.
        Les sources consultées reflètent l'origine réelle (LOCAL ou API).

        Ajoute pipeline_payload["external_validation"] avec :
          - risk_level      : CLEAN | LOW | MEDIUM | HIGH | CRITICAL
          - risk_score      : score pondéré 0-∞ (agrégé pays + devise)
          - risk_factors    : liste des facteurs détectés
          - recommendations : actions recommandées
          - details         : résultats détaillés par checker
                              (countries, banks, companies, geo)
          - sources_consulted : sources effectivement interrogées
                                (ex: "OFAC-LOCAL", "EU-LOCAL", "GLEIF-API"…)
          - processing_ms   : temps total mesuré en interne par ExternalValidationResult

        Met également à jour pipeline_payload["ds_analysis"]["risk"]["risk_level"]
        si le niveau externe est plus élevé que le niveau DS.

        Toujours silencieux en cas d'erreur (ne bloque pas le pipeline).
        """
        engine = self.ext_engine
        if engine is None:
            logger.debug("External validation ignorée — moteur non disponible (stub émis)")
            self._inject_external_stub(pipeline_payload, reason="engine_unavailable")
            return

        try:
            result = engine.validate(pipeline_payload)
            engine.inject_into_report(pipeline_payload, result)

            # FIX EXT-03 : on conserve le processing_ms calculé en interne par
            # ExternalValidationResult (t_end - t_start dans engine.validate).
            # L'ancienne version l'écrasait avec le timing externe total (qui
            # englobait aussi inject_into_report), rendant la valeur incohérente
            # avec ce qu'indique le champ "approach" dans to_dict().
            # Le champ est déjà correctement positionné par inject_into_report
            # via result.to_dict() → aucun écrasement nécessaire ici.

            logger.info(
                "External validation — risk=%s  score=%.2f  "
                "sources=%s  (%dms)",
                result.risk_level.value,
                result.raw_score,
                ",".join(result.sources_consulted[:4]),
                result.processing_ms,
            )

        except Exception as exc:
            logger.warning("External validation error (stub émis): %s", exc, exc_info=True)
            self._inject_external_stub(pipeline_payload, reason=f"error:{type(exc).__name__}")

    def _inject_external_stub(self, pipeline_payload: dict, reason: str) -> None:
        """
        Garantit la présence de pipeline_payload['external_validation'] même quand
        le moteur de validation externe est indisponible ou échoue.

        Le backend Spring Boot exige cette clé dans le contrat du pipeline ;
        son absence ferait échouer toute l'analyse. Le stub est neutre (CLEAN)
        et ne dégrade pas le score.
        """
        if pipeline_payload.get("external_validation"):
            return  # déjà rempli par inject_into_report
        pipeline_payload["external_validation"] = {
            "lc_number":         pipeline_payload.get("swift", {}).get("entities", {}).get("lc_number", ""),
            "processed_at":      "",
            "risk_level":        "CLEAN",
            "risk_score":        0.0,
            "risk_factors":      [],
            "recommendations":   [],
            "sources_consulted": [],
            "processing_ms":     0,
            "approach":          f"unavailable ({reason})",
            "details":           {"countries": [], "currency": None},
            "_stub":             True,
        }

    # ──────────────────────────────────────────────────────────────────
    # RISK-03 : Enrichissement Risk ML
    # ──────────────────────────────────────────────────────────────────

    def _enrich_with_risk_ml(self, pipeline_payload: dict) -> None:
        """
        Enrichit le payload avec le scoring ML (XGBoost + IsolationForest + KMeans + SHAP).
        Ajoute pipeline_payload["risk_ml"] avec RiskMLResult.to_dict().
        Silencieux en cas d'erreur (ne bloque jamais le pipeline).
        """
        engine = self.risk_engine
        if engine is None:
            logger.debug("Risk ML ignoré — moteur non disponible (modèles non entraînés)")
            return
        try:
            result = engine.score(pipeline_payload)
            engine.inject_into_report(pipeline_payload, result)
            logger.info(
                "Risk ML — label=%s  score=%d  anomaly=%s  segment=%s  (%dms)",
                result.risk_label, result.risk_score,
                result.is_anomaly, result.segment, result.processing_ms,
            )
        except Exception as exc:
            logger.warning("Risk ML error (skipped): %s", exc, exc_info=True)

    # ──────────────────────────────────────────────────────────────────
    # Réconciliation des décisions — verdict final unique
    # ──────────────────────────────────────────────────────────────────

    # Rangs de sévérité partagés (plus haut = plus sévère)
    _DECISION_RANK = {
        "REJECT": 3,
        "CONDITIONAL_HOLD": 2,
        "REVIEW": 2,
        "ACCEPT_WITH_FLAGS": 1,
        "ACCEPT": 0,
    }
    _RANK_LABEL = {3: "REJECT", 2: "REVIEW", 1: "ACCEPT_WITH_FLAGS", 0: "ACCEPT"}

    @staticmethod
    def _normalize_ucp_decision(raw) -> str:
        """ucp_decision peut être une string ou un dict {decision}."""
        if isinstance(raw, dict):
            raw = raw.get("decision")
        return str(raw or "").upper()

    def _resolve_final_decision(self, pipeline_payload: dict) -> None:
        """
        Réconcilie en UN verdict unique les décisions des modules.

        Principe métier (UCP 600) : un blocage documentaire dur (hard override)
        prime sur le score composite pondéré — on ne peut pas « moyenner » un REJECT
        bancaire avec un bon score de complétude. Sinon, on prend la décision la plus
        sévère entre l'UCP et le routeur composite (évite l'incohérence inverse où
        un module signale une erreur que l'autre masque).

        Écrit pipeline_payload["final_decision"] = {decision, source, reason, inputs}.
        Silencieux et non bloquant.
        """
        try:
            ucp = pipeline_payload.get("ucp_compliance") or {}
            sem = (
                (pipeline_payload.get("swift") or {})
                .get("nlp", {})
                .get("semantic", {})
            ) or {}

            ucp_decision    = self._normalize_ucp_decision(ucp.get("ucp_decision"))
            payment_blocked = bool(ucp.get("payment_blocked"))
            composite       = str(sem.get("routing_decision") or "").upper()

            # 1. Hard override : un blocage paiement UCP force REJECT, sans pondération
            if payment_blocked or ucp_decision == "REJECT":
                decision = "REJECT"
                source   = "ucp_hard_block"
                reason   = ("Violation UCP réelle bloquant le paiement — "
                            "prime sur le score composite (hard override).")
            else:
                # 2. Sinon : décision la plus sévère entre UCP et routeur composite
                ucp_rank  = self._DECISION_RANK.get(ucp_decision, -1)
                comp_rank = self._DECISION_RANK.get(composite, -1)
                if ucp_rank < 0 and comp_rank < 0:
                    decision, source = "REVIEW", "fallback"
                    reason = "Aucune décision exploitable — revue manuelle par défaut."
                elif ucp_rank >= comp_rank:
                    decision = self._RANK_LABEL[max(ucp_rank, 0)]
                    source   = "ucp_compliance"
                    reason   = "Aucun blocage paiement — verdict aligné sur la conformité UCP."
                else:
                    decision = self._RANK_LABEL[max(comp_rank, 0)]
                    source   = "composite_router"
                    reason   = "Conformité UCP OK — verdict issu du score composite sémantique."

            pipeline_payload["final_decision"] = {
                "decision": decision,
                "source":   source,
                "reason":   reason,
                "inputs": {
                    "ucp_decision":         ucp_decision or None,
                    "ucp_payment_blocked":  payment_blocked,
                    "ucp_score":            ucp.get("global_score"),
                    "composite_decision":   composite or None,
                    "composite_score":      sem.get("final_score"),
                },
            }
            logger.info(
                "Decision finale — %s (source=%s) | ucp=%s blocked=%s | composite=%s",
                decision, source, ucp_decision or "—", payment_blocked, composite or "—",
            )
        except Exception as exc:
            logger.warning("resolve_final_decision error (skipped): %s", exc, exc_info=True)

    # ──────────────────────────────────────────────────────────────────
    # Enrichissement UCP + Data Science
    # ──────────────────────────────────────────────────────────────────

    def _enrich_with_ucp_and_ds(self, pipeline_payload: dict) -> None:
        """
        Enrichit le résultat du pipeline avec :
          1. Validation UCP 600 intelligente (UCP600Validator)
          2. Risk Scoring Data Science     (LCRiskScorer)

        Modifie pipeline_payload en place.
        Toujours silencieux en cas d'erreur (ne bloque pas le pipeline).
        """
        # ── 1. Validation UCP 600 ────────────────────────────────────
        ucp_report_obj = None
        try:
            ucp = self._get_ucp_validator()
            if ucp:
                t0 = time.monotonic()
                ucp_report_obj = ucp.validate(pipeline_payload)
                ucp_ms = int((time.monotonic() - t0) * 1000)
                ucp_dict = ucp_report_obj.to_dict()
                ucp_dict["processing_ms"] = ucp_ms
                pipeline_payload["ucp_compliance"] = ucp_dict
                logger.info(
                    "UCP validation — score=%d/100  errors=%d  warnings=%d  (%dms)",
                    ucp_report_obj.global_score,
                    len(ucp_report_obj.errors),
                    len(ucp_report_obj.warnings),
                    ucp_ms,
                )
        except Exception as exc:
            logger.warning("UCP validation error (skipped): %s", exc)

        # ── 2. Risk Scoring DS ───────────────────────────────────────
        try:
            extractor, scorer = self._get_ds_modules()
            if extractor and scorer:
                t0 = time.monotonic()

                ucp_proxy = None
                if ucp_report_obj is not None:
                    _ucp_dict = pipeline_payload.get("ucp_compliance", {})
                    class _UcpProxy:
                        def to_dict(self): return _ucp_dict
                    ucp_proxy = _UcpProxy()

                features = extractor.extract(pipeline_payload, ucp_proxy)
                lc_number = (
                    pipeline_payload.get("swift", {})
                    .get("entities", {})
                    .get("lc_number", "?")
                )
                risk_result = scorer.score(features, lc_number)
                ds_ms = int((time.monotonic() - t0) * 1000)

                def _keep_ds_feature(name: str, val) -> bool:
                    if val not in (0, 0.0, False):
                        return True
                    return name in (
                        "validity_days",
                        "country_risk",
                        "days_to_shipment",
                        "presentation_days",
                    )

                pipeline_payload["ds_analysis"] = {
                    "risk":          risk_result.to_dict(),
                    "features":      {
                        k: v for k, v in features.to_dict().items()
                        if _keep_ds_feature(k, v)
                    },
                    "processing_ms": ds_ms,
                }
                logger.info(
                    "DS analysis — risk_level=%s  risk_score=%.3f  (%dms)",
                    risk_result.risk_level, risk_result.risk_score, ds_ms,
                )
        except Exception as exc:
            logger.warning("DS analysis error (skipped): %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # Pipeline hybride interne
    # ──────────────────────────────────────────────────────────────────

    def _run_hybrid(
        self,
        raw_text: str,
        ocr_confidence: float = 1.0,
        spell_report: object | None = None,
    ) -> dict:
        """Orchestration : [SPL] → [DET] → [EXT] → [LC] → [NLP] → [MRG] → [VAL]"""

        # [SPL] Déjà appliquée dans OCRService ; repli si texte brut (ex. analyze_text)
        corrected_text = raw_text
        if spell_report is None:
            try:
                from services.ocr.spell_correction import SwiftSpellCorrector

                if not hasattr(self, "_spell_corrector") or self._spell_corrector is None:
                    self._spell_corrector = SwiftSpellCorrector()
                corrected_text, spell_report = self._spell_corrector.correct(raw_text)
            except Exception as exc:
                logger.warning("SpellCorrector failed: %s", exc)
                corrected_text = raw_text
                spell_report = None

        # [DET + EXT + LC + VAL] Pipeline règles métier
        try:
            rules_result = self.swift_service.process(corrected_text)
        except Exception as exc:
            logger.error("SwiftService failed: %s", exc, exc_info=True)
            return {"success": False, "error": f"Erreur analyse SWIFT : {exc}"}

        if not rules_result.get("success"):
            swift_type = rules_result.get("swift_type", "UNKNOWN")
            nlp_result = self._run_nlp(
                corrected_text, entities=None,
                swift_type=swift_type,
                spell_report=spell_report,
            )
            rules_result["nlp"] = nlp_result
            return rules_result

        swift_type     = rules_result["swift_type"]
        rules_entities = rules_result.get("entities", {})

        # Corriger la confusion available_with ↔ available_by
        lc_criteria = rules_result.get("lc_criteria") or {}
        if lc_criteria:
            bic      = rules_entities.get("available_with")
            modality = rules_entities.get("available_by")

            if not modality:
                raw_text_41 = rules_result.get("ocr_text", "") or ""
                m41 = re.search(
                    r":41[Aa]:.*?(BY\s+(?:PAYMENT|ACCEPTANCE|NEGOTIATION"
                    r"|DEF(?:ERRED)?\s+PAYMENT|MIXED\s+PYMT))",
                    raw_text_41, re.IGNORECASE | re.DOTALL
                )
                if m41:
                    modality = m41.group(1).strip().upper()
                    rules_entities["available_by"] = modality

            lc_avail = lc_criteria.get("available_by", "")
            if lc_avail and re.match(r"^[A-Z0-9]{8,11}$", str(lc_avail).strip()):
                lc_criteria["available_by"] = modality
                rules_result["lc_criteria"] = lc_criteria
            if modality and not rules_entities.get("available_by"):
                rules_entities["available_by"] = modality

        rules_entities["_ocr_text"] = raw_text
        entity_adapter = _EntityDictAdapter(rules_entities)

        nlp_result = self._run_nlp(
            corrected_text,
            entities=entity_adapter,
            swift_type=swift_type,
            spell_report=spell_report,
        )

        if nlp_result is None:
            nlp_result = {"ner": None, "semantic": None, "clauses": [], "contradictions": []}

        merged_entities = self._merge_entities(
            rules_entities,
            (nlp_result.get("ner") or {}).get("entities", []),
        )

        if (nlp_result.get("semantic") or {}).get("enriched_fields"):
            merged_entities = self._apply_inferences(
                merged_entities,
                nlp_result["semantic"]["enriched_fields"],
            )

        return {
            "success":          True,
            "swift_type":       swift_type,
            "detection_scores": rules_result.get("detection_scores", {}),
            "entities":         merged_entities,
            "lc_criteria":      rules_result.get("lc_criteria"),
            "nlp":              nlp_result,
            "validation":       rules_result.get("validation", {}),
        }

    # ──────────────────────────────────────────────────────────────────
    # NLP runner
    # ──────────────────────────────────────────────────────────────────

    def _run_nlp(
        self, text: str, entities, swift_type: str, spell_report,
        skip_internal_spell: bool = False,
    ) -> dict:
        try:
            nlp_processing = self.nlp.process(
                raw_text=text,
                entities=entities,
                swift_type=swift_type,
            )
            if spell_report is not None:
                nlp_processing.spell_report = spell_report

            return self._serialize_nlp(nlp_processing)

        except Exception as exc:
            logger.error("NLP service failed: %s", exc, exc_info=True)
            return {
                "score":   0,
                "spell":   None,
                "ner":     None,
                "clauses": [],
                "contradictions": [],
                "module_timings": {},
                "semantic": {
                    "completeness_score":  0,
                    "semantic_score":      0,
                    "doc_quality_score":   0,
                    "warnings":            [],
                    "risk_flags":          [],
                    "fraud_indicators":    [],
                    "recommendations":     [],
                    "summary":             "",
                    "enriched_fields":     {},
                    "bayesian_confidence": 0.0,
                    "final_score":         0.0,
                    "routing_decision":    "REJECT",
                    "cross_field_coherence": {},
                },
                "error": str(exc),
            }

    @staticmethod
    def _serialize_nlp(result) -> dict:
        """Sérialise NLPProcessingResult → dict JSON-serializable."""

        spell_dict = None
        if result.spell_report:
            spell_dict = {
                "total_changes":     result.spell_report.total_changes,
                "has_ocr_noise":     result.spell_report.has_ocr_noise,
                "language_detected": getattr(result.spell_report, "language_detected", ""),
                "avg_confidence":    round(getattr(result.spell_report, "avg_confidence", 0), 3),
                "rejected_count":    getattr(result.spell_report, "rejected_count", 0),
                "corrections": [
                    {
                        "original":   c.original,
                        "corrected":  c.corrected,
                        "method":     c.method,
                        "position":   c.position,
                        "confidence": round(getattr(c, "confidence", 1.0), 3),
                        "validated":  getattr(c, "validated", True),
                    }
                    for c in result.spell_report.corrections
                ],
            }

        ner_dict = None
        if result.ner:
            ner_dict = {
                "entities": [
                    {
                        "label":      e.label,
                        "value":      e.value,
                        "start":      e.start,
                        "end":        e.end,
                        "confidence": round(e.confidence, 3),
                        "source":     e.source,
                        "ner_label":  e.ner_label,
                        "country":    getattr(e, "country", None),
                        "normalized": getattr(e, "normalized", None),
                    }
                    for e in result.ner.entities
                ],
                "spacy_spans": result.ner.spacy_spans,
                "source_mode": getattr(result.ner, "source_mode", "rules"),
                "relations": [
                    {
                        "subject":      r.subject,
                        "predicate":    r.predicate,
                        "object_label": r.object_label,
                        "object_value": r.object_value,
                    }
                    for r in getattr(result.ner, "relations", [])
                ],
            }

        clauses_list = [
            {
                "text":       c.text,
                "type":       c.type,
                "importance": c.importance,
                "confidence": round(c.confidence, 3),
                "method":     c.method,
                "ucp_refs":   getattr(c, "ucp_refs", []),
                "risk_level": getattr(c, "risk_level", "normal"),
            }
            for c in result.clauses
        ]

        contradictions = [
            {
                "source":   "NLP",
                "severity": cont.get("severity", "warning"),
                "message":  cont.get("message", ""),
                "clause_a": cont.get("clause_a", ""),
                "clause_b": cont.get("clause_b", ""),
            }
            for cont in getattr(result, "contradictions", [])
        ]

        semantic_dict = {
            "completeness_score": 0,
            "semantic_score":     0,
            "doc_quality_score":  0,
            "warnings":           [],
            "risk_flags":         [],
            "contradictions":     [],
            "fraud_indicators":   [],
            "recommendations":    [],
            "summary":            "",
            "enriched_fields":    {},
            "coherence_scores":   {},
            "geo_coherence":      {},
        }
        if result.semantic:
            sem = result.semantic
            semantic_dict = {
                "completeness_score": sem.completeness_score,
                "semantic_score":     sem.semantic_score,
                "doc_quality_score":  getattr(sem, "doc_quality_score", 0),
                "geo_coherence":      getattr(sem, "geo_coherence", {}),
                "coherence_scores":   getattr(sem, "coherence_scores", {}),
                "warnings": [
                    {
                        "code":       w.code,
                        "severity":   w.severity,
                        "message":    w.message,
                        "field":      w.field,
                        "ucp_ref":    getattr(w, "ucp_ref", None),
                        "suggestion": getattr(w, "suggestion", None),
                    }
                    for w in sem.warnings
                ],
                "risk_flags": [
                    {
                        "code":        r.code,
                        "description": r.description,
                        "severity":    r.severity,
                        "category":    getattr(r, "category", ""),
                        "ucp_ref":     r.ucp_ref,
                        "ml_score":    round(getattr(r, "ml_score", 0.0), 3),
                    }
                    for r in sem.risk_flags
                ],
                "contradictions": [
                    c.to_dict() if hasattr(c, "to_dict") else dict(c)
                    for c in getattr(sem, "contradictions", []) or []
                ],
                "fraud_indicators":     getattr(sem, "fraud_indicators", []),
                "recommendations":      getattr(sem, "recommendations", []),
                "summary":              sem.summary,
                "enriched_fields":      sem.enriched_fields,
                "bayesian_confidence":  round(getattr(sem, "bayesian_confidence", 0.0), 3),
                "final_score":          round(getattr(sem, "final_score", 0.0), 4),
                "routing_decision":     getattr(sem, "routing_decision", ""),
                "cross_field_coherence": getattr(sem, "cross_field_coherence", {}),
            }

        # ── Liste canonique : fusionner les contradictions tag↔clause du moteur
        # sémantique (ContradictionEngine) dans nlp.contradictions, taguées source.
        # Sinon nlp.contradictions reste vide (les contradictions ne vivent que dans
        # semantic.contradictions) et tout consommateur lisant le top-level voit 0.
        for c in (semantic_dict.get("contradictions") or []):
            tag_field = c.get("tag_field", "")
            contradictions.append({
                "source":     "SEMANTIC",
                "severity":   c.get("severity", "error"),
                "message":    c.get("recommendation") or c.get("clause_intent") or "",
                "clause_a":   f":{tag_field}: = {c.get('tag_value', '')}" if tag_field else "",
                "clause_b":   c.get("clause_text", ""),
                # champs d'origine conservés pour l'affichage sémantique (tag↔clause)
                "tag_field":      tag_field,
                "tag_value":      c.get("tag_value", ""),
                "clause_text":    c.get("clause_text", ""),
                "clause_intent":  c.get("clause_intent", ""),
                "ucp_basis":      c.get("ucp_basis", ""),
                "ucp_ref":        c.get("ucp_basis", ""),
                "recommendation": c.get("recommendation", ""),
                "confidence":     c.get("confidence", 1.0),
            })

        return {
            "score":            result.nlp_score,
            "processing_ms":    result.processing_time_ms,
            "modules_applied":  result.modules_applied,
            "module_timings":   getattr(result, "module_timings", {}),
            "llm_triggered":    getattr(result, "llm_triggered", False),
            "from_cache":       getattr(result, "from_cache", False),
            "spell":            spell_dict,
            "ner":              ner_dict,
            "clauses":          clauses_list,
            "contradictions":   contradictions,
            "semantic":         semantic_dict,
        }

    # ──────────────────────────────────────────────────────────────────
    # Fusion règles + NLP  —  v6.4 : logique NER-first
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _disambiguate_korea_country(merged: dict) -> None:
        """Corée du Nord / Sud : corrige country ambigu (ex. KOREA) via le contexte LC."""
        blob = " ".join(
            str(merged.get(k) or "")
            for k in (
                "beneficiary",
                "port_loading",
                "port_discharge",
                "goods_description",
                "country",
                "_ocr_text",
            )
        ).upper()
        if not blob.strip():
            return
        nk = bool(
            re.search(
                r"NORTH\s+KOREA|DPRK|DEMOCRATIC\s+PEOPLE'?S\s+REPUBLIC|PYONGYANG|\bKP\b",
                blob,
            )
        )
        sk = bool(
            re.search(
                r"SOUTH\s+KOREA|REPUBLIC\s+OF\s+KOREA|\bSEOUL\b|\bBUSAN\b",
                blob,
            )
        )
        cur = (merged.get("country") or "").strip().upper()
        if nk and not sk:
            if cur in ("", "KOREA", "KR", "KP", "KO"):
                merged["country"] = "KP"
                merged["_country_override_reason"] = "nk_context_heuristic"
        elif sk and not nk and cur in ("KOREA", "KO", ""):
            merged["country"] = "KR"
            merged["_country_override_reason"] = "sk_context_heuristic"

    @staticmethod
    def _clean_party_value(val: str) -> str:
        """Nettoie les artefacts OCR/parseur dans les noms de parties."""
        if not val:
            return val
        import re as _re
        val = _re.sub(r'\s*/adress\w*', '', val, flags=_re.IGNORECASE).strip()
        val = _re.sub(r'\s+Address\s+not\s+provided\s*$', '', val, flags=_re.IGNORECASE).strip()
        return val

    @staticmethod
    def _merge_entities(rules_entities: dict, ner_entities: list) -> dict:
        """
        Fusionne les entités issues du parser (règles métier) et du NER.

        Priorité appliquée à TOUS les champs (MRG-03) :
          • conf NER ≥ NER_HIGH_THRESHOLD (0.80)
              → NER écrase le parser, même si le parser avait déjà une valeur.
                Trace : _<field>_override_reason = "ner_high_conf"

          • conf NER ∈ [NER_LOW_THRESHOLD, NER_HIGH_THRESHOLD[ (0.70 … 0.80)
              → Parser prioritaire.
                NER = fallback uniquement si le parser est vide.
                Si le parser a déjà une valeur, la valeur NER est tracée dans
                _<field>_ner_candidate / _<field>_ner_candidate_conf.

          • conf NER < NER_LOW_THRESHOLD (0.70)
              → NER ignoré.
                Tracé dans _<field>_ner_ignored / _<field>_ner_ignored_conf.

        PORT et AMOUNT suivent les mêmes seuils (MRG-04).
        """
        LABEL_TO_FIELD = {
            "APPLICANT":     "applicant",
            "BENEFICIARY":   "beneficiary",
            "ISSUING_BANK":  "issuing_bank",
            "GOODS":         "goods_description",
            "INCOTERM":      "incoterm",
            "EXPIRY_DATE":   "expiry_date",
            "ISSUE_DATE":    "issue_date",
            "SHIPMENT_DATE": "shipment_date",
            "LC_NUMBER":     "lc_number",
            "COUNTRY":       "country",
        }

        merged = dict(rules_entities)

        # ── Étape 1 : entités scalaires (hors PORT / AMOUNT) ──────────
        for ent in ner_entities:
            # Les entités source="parser" viennent déjà de rules_entities
            if ent.get("source") == "parser":
                continue

            label = ent.get("label", "")
            field = LABEL_TO_FIELD.get(label)
            if not field:
                continue

            value = ent.get("value")
            if not value:
                continue

            conf           = ent.get("confidence", 0.0)
            parser_value   = merged.get(field)
            ner_source_tag = f"nlp:{ent.get('source', 'ner')}"

            if conf >= NER_HIGH_THRESHOLD:
                # NER prime — écrase le parser
                merged[field]                       = value
                merged[f"_{field}_source"]          = ner_source_tag
                merged[f"_{field}_conf"]            = conf
                merged[f"_{field}_override_reason"] = "ner_high_conf"

            elif conf >= NER_LOW_THRESHOLD:
                # Parser prioritaire ; NER = fallback si parser vide
                if not parser_value:
                    merged[field]              = value
                    merged[f"_{field}_source"] = ner_source_tag
                    merged[f"_{field}_conf"]   = conf
                else:
                    # Parser déjà présent : on trace le candidat NER sans l'appliquer
                    merged[f"_{field}_ner_candidate"]      = value
                    merged[f"_{field}_ner_candidate_conf"] = conf

            else:
                # conf < NER_LOW_THRESHOLD → NER ignoré (trace de débogage)
                merged[f"_{field}_ner_ignored"]      = value
                merged[f"_{field}_ner_ignored_conf"] = conf

        # ── Étape 2 : entités PORT (double port, seuils appliqués) ────
        _COND_RE = re.compile(
            r"SHOULD\s+READ|FIELD\s+:\d+[A-Z]?:|IN\s+CASE\s+CONFIRMATION"
            r"|fees\s+outside|charges\s+outside",
            re.IGNORECASE,
        )
        ocr_full = merged.get("_ocr_text", "")

        def _port_valid(ent: dict) -> bool:
            if ent.get("source") == "parser":
                return False
            val = ent.get("value", "")
            if not val:
                return False
            start = ent.get("start")
            if start is not None and ocr_full:
                ctx = ocr_full[max(0, start - 200): start + len(val) + 200]
                if _COND_RE.search(ctx):
                    return False
            return True

        port_ents = [
            e for e in ner_entities
            if e.get("label") == "PORT" and _port_valid(e)
        ]

        # PORT[0] → port_loading
        if port_ents:
            p0   = port_ents[0]
            conf = p0.get("confidence", 0.0)
            tag  = f"nlp:{p0.get('source', 'ner')}"
            if conf >= NER_HIGH_THRESHOLD:
                merged["port_loading"]                      = p0["value"]
                merged["_port_loading_source"]              = tag
                merged["_port_loading_conf"]                = conf
                merged["_port_loading_override_reason"]     = "ner_high_conf"
            elif conf >= NER_LOW_THRESHOLD and not merged.get("port_loading"):
                merged["port_loading"]         = p0["value"]
                merged["_port_loading_source"] = tag
                merged["_port_loading_conf"]   = conf

        # PORT[1] → port_discharge
        if len(port_ents) >= 2:
            p1   = port_ents[1]
            conf = p1.get("confidence", 0.0)
            tag  = f"nlp:{p1.get('source', 'ner')}"
            if conf >= NER_HIGH_THRESHOLD:
                merged["port_discharge"]                    = p1["value"]
                merged["_port_discharge_source"]            = tag
                merged["_port_discharge_conf"]              = conf
                merged["_port_discharge_override_reason"]   = "ner_high_conf"
            elif conf >= NER_LOW_THRESHOLD and not merged.get("port_discharge"):
                merged["port_discharge"]         = p1["value"]
                merged["_port_discharge_source"] = tag
                merged["_port_discharge_conf"]   = conf

        # ── Étape 3 : entité AMOUNT (seuils appliqués) ────────────────
        if not merged.get("amount"):
            for ent in ner_entities:
                if ent.get("label") != "AMOUNT" or not ent.get("value"):
                    continue
                conf = ent.get("confidence", 0.0)
                if conf < NER_LOW_THRESHOLD:
                    continue
                val = str(ent["value"])
                m = re.match(r'([A-Z]{3})\s*([\d,\.]+)', val)
                if m:
                    if not merged.get("currency"):
                        merged["currency"] = m.group(1)
                    merged["amount"]         = m.group(2).replace(",", "")
                    merged["_amount_source"] = "nlp:ner"
                    merged["_amount_conf"]   = conf
                break

        # ── Étape 4 : nettoyage des parties ───────────────────────────
        import re as _re
        for party_field in ("applicant", "beneficiary", "issuing_bank"):
            val = merged.get(party_field, "")
            if not val:
                continue
            val = _re.sub(r'\s*/adress\w*', '', val, flags=_re.IGNORECASE).strip()
            val = _re.sub(r'\s+Address\s+not\s+provided\s*$', '', val,
                          flags=_re.IGNORECASE).strip()
            m_acc = _re.search(r'for\s+the\s+account\s+of\s+(.+)', val, _re.IGNORECASE)
            if m_acc and party_field == "applicant":
                after = _re.sub(
                    r'\s+Address.*$', '', m_acc.group(1).strip(),
                    flags=_re.IGNORECASE,
                ).strip()
                if after and len(after) > 3:
                    merged[party_field] = after
                    merged[f"_{party_field}_extracted"] = True
                    continue
            merged[party_field] = val

        for field in ("applicant", "beneficiary", "issuing_bank"):
            if merged.get(field):
                merged[field] = SwiftPipeline._clean_party_value(merged[field])

        SwiftPipeline._disambiguate_korea_country(merged)

        # ── Étape 5 : champ 47A (conditions supplémentaires) ──────────
        ocr_txt = merged.get("_ocr_text", "")
        if ocr_txt:
            m47 = re.search(
                r":47A:\s*(.*?)(?=\s*:?\s*(?:48|49|71[BD]?|72[ZD]?|77[BEU]|78)\s*:|\Z)",
                ocr_txt,
                re.DOTALL | re.IGNORECASE,
            )
            if m47:
                full_47a = m47.group(1).strip()
                current  = merged.get("additional_conditions") or ""
                if len(full_47a) > len(current):
                    merged["additional_conditions"] = full_47a

        # ── Étape 6 : fallback BIC émetteur depuis l'en-tête SWIFT ───
        if not merged.get("issuing_bank") and ocr_txt:
            m_hdr = re.search(r"\{1:F01([A-Z0-9]{8,11})\d", ocr_txt)
            if m_hdr:
                merged["issuing_bank"]         = m_hdr.group(1).strip()
                merged["_issuing_bank_source"] = "header_bic"

        return merged

    @staticmethod
    def _apply_inferences(entities: dict, inferences: dict) -> dict:
        enriched = dict(entities)
        INFERENCE_MAP = {
            "incoterm_inferred":        "incoterm",
            "available_by_inferred":    "available_by",
            "applicable_rules_default": "applicable_rules",
        }
        for infer_key, entity_field in INFERENCE_MAP.items():
            if infer_key in inferences and not enriched.get(entity_field):
                enriched[entity_field]                = inferences[infer_key]
                enriched[f"_{entity_field}_inferred"] = True
        return enriched


# ──────────────────────────────────────────────────────────────────────
# Adaptateur dict → objet attributs (pour SwiftNLPService)
# ──────────────────────────────────────────────────────────────────────

class _EntityDictAdapter:
    """PL-04 : adaptateur robuste."""
    def __init__(self, d: dict):
        self._d = d or {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            return None
        return self._d.get(name)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __contains__(self, key):
        return key in self._d
