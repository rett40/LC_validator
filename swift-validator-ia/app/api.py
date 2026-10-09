"""
api.py  —  v6.4  (+ Commercial Document Validation)
=====================================================
MODIFICATIONS v6.4 (par rapport à v6.3) :
  ✦ COMM-API-01 : route POST /commercial/validate — Pipeline multi-documents
                  BL + Invoice + Packing List avec validation croisée
  ✦ COMM-API-02 : route GET  /commercial/health   — statut du pipeline commercial

Routes complètes (v6.4) :
  GET  /                      — info + version
  GET  /health                — santé + statut modules + modèles ML + APIs externes + Risk ML
  POST /ocr                   — OCR seul
  POST /pipeline              — Pipeline complet hybride (OCR + règles + NLP + ext + risk ML)
  POST /swift/analyze         — Analyse SWIFT sur texte brut
  POST /swift/nlp             — NLP seul
  POST /swift/spell           — Correction orthographique
  POST /swift/classify        — Classification clauses
  GET  /risk/health           — Statut Risk ML Service (modèles, version)
  POST /risk/score            — Scoring Risk ML depuis payload pipeline complet
  POST /risk/score/raw        — Scoring Risk ML depuis features brutes (dict)
  POST /commercial/validate   — Validation multi-documents commerciaux (BL+Invoice+PL)
  GET  /commercial/health     — Statut du pipeline commercial (release2)
"""

import sys
import os
import tempfile
import logging
from pathlib import Path

from flask import Flask, request, jsonify

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ── Dev local : charge le .env racine (mêmes valeurs que docker-compose) ───
# No-op si python-dotenv est absent. override=False : ne JAMAIS écraser une
# variable déjà fournie par l'environnement → en Docker, compose reste maître.
try:
    from dotenv import load_dotenv
    for _env in (ROOT_DIR / ".env", ROOT_DIR.parent / ".env"):
        if _env.is_file():
            load_dotenv(_env, override=False)
            break
except ModuleNotFoundError:
    pass

from services.ocr.ocr_service import OCRService
from pipeline.swift_pipeline  import SwiftPipeline
from app.pdf                  import pdf_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB
CONTRACT_VERSION = "1.0"

# ── Blueprints ───────────────────────────────────────────────────────────
app.register_blueprint(pdf_bp)

ocr              = OCRService()
pipeline_service = SwiftPipeline()

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

_MODELS_DIR  = ROOT_DIR / "models"
_CLAUSE_PATH = _MODELS_DIR / "clause-clf"
_RISK_PATH   = _MODELS_DIR / "risk-clf"

# ── Risk ML Engine (lazy) ────────────────────────────────────────────
_risk_engine = None

def _get_risk_engine():
    global _risk_engine
    if _risk_engine is None:
        try:
            from services.risk_service.risk_engine import get_risk_engine
            _risk_engine = get_risk_engine()
        except Exception as exc:
            logger.warning("Risk ML Engine non disponible : %s", exc)
            _risk_engine = False
    return _risk_engine if _risk_engine else None

EXTERNAL_APIS = {
    "ofac_sdn":       {"url": "https://sanctionssearch.ofac.treas.gov/api/search/v2/search",
                       "description": "OFAC SDN", "required": True, "free": True, "auth": False},
    "gleif":          {"url": "https://api.gleif.org/api/v1/lei-records",
                       "description": "GLEIF LEI/BIC", "required": True, "free": True, "auth": False},
    "eu_sanctions":   {"url": "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content",
                       "description": "EU Sanctions XML", "required": True, "free": True, "auth": False},
    "un_sanctions":   {"url": "https://scsanctions.un.org/resources/xml/en/consolidated.xml",
                       "description": "UN XML", "required": True, "free": True, "auth": False},
    "opencorporates": {"url": "https://api.opencorporates.com/v0.4/companies/search",
                       "description": "Vérification sociétés", "required": False, "free": True, "auth": False},
    "nominatim":      {"url": "https://nominatim.openstreetmap.org/search",
                       "description": "Résolution port → pays", "required": False, "free": True, "auth": False},
    "rest_countries": {"url": "https://restcountries.com/v3.1/alpha/",
                       "description": "Informations pays", "required": False, "free": True, "auth": False},
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def save_temp(file) -> str:
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        return tmp.name

def cleanup(path) -> None:
    if path and Path(path).exists():
        Path(path).unlink()

def validate_upload(req) -> tuple:
    if "file" not in req.files:
        return None, {"success": False, "error": "Champ 'file' manquant"}
    f = req.files["file"]
    if not f.filename:
        return None, {"success": False, "error": "Nom de fichier vide"}
    if not allowed_file(f.filename):
        return None, {"success": False, "error": f"Type non supporté. Autorisés : {sorted(ALLOWED_EXTENSIONS)}"}
    return f, None

def require_json_text(req) -> tuple:
    body = req.get_json(silent=True)
    if not body or "text" not in body:
        return None, {"success": False, "error": "Body JSON attendu avec le champ 'text'"}
    text = body.get("text", "")
    if not text or not text.strip():
        return None, {"success": False, "error": "Texte vide"}
    return text, None


# ──────────────────────────────────────────────────────────────────────
# Routes info
# ──────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "message": "SWIFT OCR Pipeline API",
        "version": "6.3-risk-ml",
        "mode":    "hybrid (rules + NLP v2 + external validation + Risk ML)",
        "routes": {
            "GET  /health":         "Santé + statut modules + ML + APIs + Risk ML",
            "POST /ocr":            "OCR seul (PDF/image)",
            "POST /pipeline":       "Pipeline hybride complet + ext + Risk ML",
            "POST /swift/analyze":  "Analyse SWIFT texte brut",
            "POST /swift/nlp":      "NLP seul",
            "POST /swift/spell":    "Correction orthographique",
            "POST /swift/classify": "Classification clauses",
            "GET  /risk/health":    "Statut Risk ML Service",
            "POST /risk/score":     "Scoring Risk ML (payload pipeline)",
            "POST /risk/score/raw": "Scoring Risk ML (features brutes)",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    modules = {"ocr": "ok", "rules": "ok", "nlp": "unknown"}
    try:
        from services.swift_nlp import SwiftNLPService
        modules["nlp"] = "ok"
    except ImportError:
        modules["nlp"] = "unavailable"

    ext_engine_status = "unknown"
    try:
        from services.external_validation import ExternalValidationEngine
        ext_engine_status = "ok"
        modules["external_validation"] = "ok"
    except ImportError as e:
        ext_engine_status = f"unavailable ({e})"
        modules["external_validation"] = ext_engine_status

    # ── RISK-API-04 : Risk ML status ─────────────────────────────────
    risk_ml_status = "unknown"
    try:
        from services.risk_service.risk_engine import get_risk_engine
        eng = get_risk_engine()
        if eng.is_available():
            risk_ml_status = "ok"
            modules["risk_ml"] = "ok"
        else:
            risk_ml_status = "absent (run train_models.py)"
            modules["risk_ml"] = risk_ml_status
    except ImportError as e:
        risk_ml_status = f"unavailable ({e})"
        modules["risk_ml"] = risk_ml_status

    ml_models = {
        "clause_classifier": {
            "path":   str(_CLAUSE_PATH),
            "loaded": _CLAUSE_PATH.exists(),
            "status": "ok" if _CLAUSE_PATH.exists() else "absent",
        },
        "risk_model": {
            "path":   str(_RISK_PATH),
            "loaded": _RISK_PATH.exists(),
            "status": "ok" if _RISK_PATH.exists() else "absent",
        },
    }

    external_apis_status = {
        name: {**cfg, "configured": True}
        for name, cfg in EXTERNAL_APIS.items()
    }

    # ── Parité MLOps : versions libs + révisions modèles + device ────
    # Comparer ce bloc entre local et prod suffit à auditer la parité.
    try:
        from config.model_registry import versions as _registry_versions
        parity = _registry_versions()
    except Exception as e:
        parity = {"error": str(e)}

    manifest = {}
    _manifest_path = ROOT_DIR / "models.yaml"
    if _manifest_path.exists():
        try:
            import yaml
            manifest = yaml.safe_load(_manifest_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            manifest = {"error": str(e)}

    return jsonify({
        "status":   "ok",
        "version":  "6.5-mlops",
        "modules":  modules,
        "ml_models": ml_models,
        "parity":   parity,
        "model_manifest": manifest,
        "warmup":   dict(_WARMUP_STATUS),
        "risk_ml_engine": {
            "status":  risk_ml_status,
            "models_dir": str(ROOT_DIR / "services" / "risk_service" / "models"),
        },
        "external_validation": {
            "engine_status": ext_engine_status,
            "approach":      "B — APIs web (OFAC, GLEIF, EU, UN, OpenCorp, Nominatim)",
            "apis":          external_apis_status,
        },
    })


# ──────────────────────────────────────────────────────────────────────
# OCR seul
# ──────────────────────────────────────────────────────────────────────

@app.route("/ocr", methods=["POST"])
def run_ocr():
    file, err = validate_upload(request)
    if err:
        return jsonify(err), 400
    temp_path = None
    try:
        temp_path = save_temp(file)
        result    = ocr.process(temp_path)
        return jsonify({"success": True, "confidence": result.get("confidence", 0),
                        "text": result.get("text", "")})
    except Exception as exc:
        logger.error("OCR failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Erreur interne OCR"}), 500
    finally:
        cleanup(temp_path)


# ──────────────────────────────────────────────────────────────────────
# Pipeline complet (v6.3 : inclut Risk ML via SwiftPipeline v6.6)
# ──────────────────────────────────────────────────────────────────────

@app.route("/pipeline", methods=["POST"])
def run_pipeline():
    file, err = validate_upload(request)
    if err:
        return jsonify(err), 400
    temp_path = None
    try:
        temp_path = save_temp(file)
        result    = pipeline_service.process(temp_path)
        result["contract_version"] = CONTRACT_VERSION
        return jsonify(result)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Erreur interne pipeline"}), 500
    finally:
        cleanup(temp_path)


@app.route("/swift/analyze", methods=["POST"])
def analyze_swift_text():
    text, err = require_json_text(request)
    if err:
        return jsonify(err), 400
    try:
        result = pipeline_service.analyze_text(text)
        return jsonify(result)
    except Exception as exc:
        logger.error("SWIFT analyze failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Erreur interne analyse SWIFT"}), 500


@app.route("/swift/nlp", methods=["POST"])
def run_nlp_only():
    body = request.get_json(silent=True)
    if not body or "text" not in body:
        return jsonify({"success": False, "error": "Body JSON attendu avec 'text'"}), 400
    text       = body.get("text", "")
    swift_type = body.get("swift_type", "")
    if not text.strip():
        return jsonify({"success": False, "error": "Texte vide"}), 400
    try:
        service    = pipeline_service.nlp
        nlp_result = service.process(raw_text=text, swift_type=swift_type)
        serialized = SwiftPipeline._serialize_nlp(nlp_result)
        return jsonify({"success": True, "swift_type": nlp_result.swift_type, "nlp": serialized})
    except Exception as exc:
        logger.error("NLP failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Erreur interne NLP"}), 500


@app.route("/swift/spell", methods=["POST"])
def run_spell_check():
    text, err = require_json_text(request)
    if err:
        return jsonify(err), 400
    try:
        corrected, report = pipeline_service.nlp._spell.correct(text)
        return jsonify({
            "success":          True, "corrected": corrected,
            "changes":          report.total_changes,
            "has_ocr_noise":    report.has_ocr_noise,
            "language_detected":getattr(report, "language_detected", ""),
            "corrections": [
                {"original": c.original, "corrected": c.corrected, "method": c.method,
                 "position": c.position, "confidence": round(getattr(c, "confidence", 1.0), 3)}
                for c in report.corrections
            ],
        })
    except Exception as exc:
        logger.error("Spell check failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Erreur correction orthographique"}), 500


@app.route("/swift/classify", methods=["POST"])
def classify_clauses():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Body JSON requis"}), 400
    try:
        clf = pipeline_service.nlp._clf
        if "clauses" in body:
            clauses_in = body["clauses"]
            if not isinstance(clauses_in, list):
                return jsonify({"success": False, "error": "'clauses' doit être une liste"}), 400
            results = clf.classify_batch(clauses_in)
        elif "text" in body:
            text = body["text"]
            if not text.strip():
                return jsonify({"success": False, "error": "Texte vide"}), 400
            results = clf.extract_and_classify_from_swift(text)
        else:
            return jsonify({"success": False, "error": "Body doit contenir 'clauses' ou 'text'"}), 400

        return jsonify({
            "success": True, "count": len(results),
            "results": [
                {"text": r.text, "type": r.type, "importance": r.importance,
                 "confidence": round(r.confidence, 3), "method": r.method,
                 "ucp_refs": getattr(r, "ucp_refs", [])}
                for r in results
            ],
        })
    except Exception as exc:
        logger.error("Clause classification failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": "Erreur classification"}), 500


# ──────────────────────────────────────────────────────────────────────
# RISK-API-01/02/03 : Routes Risk ML dédiées
# ──────────────────────────────────────────────────────────────────────

@app.route("/risk/health", methods=["GET"])
def risk_health():
    """Statut détaillé du Risk ML Service."""
    engine = _get_risk_engine()
    if engine is None:
        return jsonify({
            "status":       "unavailable",
            "reason":       "Modèles non entraînés ou service non chargé",
            "action":       "cd services/risk_service && python scripts/train_models.py",
        }), 503

    from services.risk_service.risk_engine import FEATURE_COLS
    md = engine._models_dir
    files = {}
    for fname in ["scaler.pkl", "xgboost_risk.pkl", "isolation_forest.pkl",
                  "kmeans.pkl", "cluster_mapping.json", "metadata.json"]:
        p = md / fname
        files[fname] = {"exists": p.exists(),
                        "size_kb": round(p.stat().st_size / 1024, 1) if p.exists() else 0}

    return jsonify({
        "status":        "ok" if engine.is_available() else "degraded",
        "models_dir":    str(md),
        "models_loaded": engine._loaded,
        "model_version": engine._metadata.get("version", "?") if engine._metadata else "?",
        "features":      FEATURE_COLS,
        "files":         files,
        "shap_ready":    engine._shap_explainer is not None if engine._loaded else False,
    })


@app.route("/risk/score", methods=["POST"])
def risk_score():
    """
    Scoring Risk ML depuis un payload pipeline SWIFT complet.
    Body : JSON du résultat de /pipeline ou /swift/analyze.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Body JSON requis"}), 400

    engine = _get_risk_engine()
    if engine is None:
        return jsonify({"success": False,
                        "error": "Risk ML non disponible — lancer train_models.py"}), 503

    try:
        result = engine.score(body)
        return jsonify({"success": True, **result.to_dict()})
    except Exception as exc:
        logger.error("Risk score failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/risk/score/raw", methods=["POST"])
def risk_score_raw():
    """
    Scoring Risk ML depuis des features brutes.
    Body : dict avec les features (partielles acceptées, défauts appliqués).

    Exemple :
        {
          "lc_amount": 2500000,
          "country_risk": 2,
          "nlp_score": 45,
          "n_ucp_warnings": 6,
          "days_to_shipment": 5
        }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Body JSON requis"}), 400

    engine = _get_risk_engine()
    if engine is None:
        return jsonify({"success": False, "error": "Risk ML non disponible"}), 503

    try:
        from services.risk_service.risk_engine import FEATURE_DEFAULTS
        feat_dict = FEATURE_DEFAULTS.copy()
        feat_dict.update({k: v for k, v in body.items() if k in FEATURE_DEFAULTS})

        # Payload synthétique minimal
        payload = {
            "swift": {
                "entities": {
                    "lc_amount":    feat_dict["lc_amount"],
                    "transshipment":"YES" if feat_dict["transshipment_allowed"] else "NO",
                    "charges":      "BENEFICIARY" if feat_dict["charges_beneficiary"] else "ALL",
                },
                "nlp": {
                    "score":   feat_dict["nlp_score"],
                    "clauses": [None] * int(feat_dict["n_clauses"]),
                    "semantic": {
                        "coherence_score":    feat_dict["semantic_score"],
                        "completeness_score": feat_dict["completeness_score"],
                    },
                },
                "lc_criteria": {
                    "days_to_shipment":    feat_dict["days_to_shipment"],
                    "presentation_period": feat_dict["presentation_days"],
                    "validity_days":       feat_dict["validity_days"],
                    "documents_required":  feat_dict["n_documents_required"],
                },
            },
            "ucp_compliance": {
                "global_score": int(feat_dict["ucp_compliance_score"] * 100),
                "warnings":     [None] * int(feat_dict["n_ucp_warnings"]),
                "errors":       [],
            },
        }

        result = engine.score(payload)
        result.features_used = feat_dict
        return jsonify({"success": True, **result.to_dict()})
    except Exception as exc:
        logger.error("Risk score raw failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# COMM-API-01/02 : Routes validation documents commerciaux (release2)
# ──────────────────────────────────────────────────────────────────────

_commercial_pipeline = None

def _get_commercial_pipeline():
    global _commercial_pipeline
    if _commercial_pipeline is None:
        try:
            release2_path = ROOT_DIR / "release2"
            if str(release2_path.parent) not in sys.path:
                sys.path.insert(0, str(release2_path.parent))
            from release2.services.pipeline import DocumentPipeline
            _commercial_pipeline = DocumentPipeline(
                confidence_threshold=0.0,
                max_workers=3,
            )
            # Pré-charger les modèles LayoutLM (BL, Invoice, PL) maintenant,
            # pas au moment de la première requête utilisateur.
            _commercial_pipeline.warmup()
            logger.info("Commercial DocumentPipeline (release2) chargé et prêt.")
        except Exception as exc:
            logger.warning("Commercial pipeline non disponible : %s", exc)
            _commercial_pipeline = False
    return _commercial_pipeline if _commercial_pipeline else None


@app.route("/commercial/health", methods=["GET"])
def commercial_health():
    """Statut du pipeline commercial release2."""
    pipeline = _get_commercial_pipeline()
    if pipeline is None:
        return jsonify({
            "status":  "unavailable",
            "reason":  "release2/services/pipeline.py non chargé",
            "action":  "Vérifiez les dépendances Python du pipeline release2",
        }), 503
    return jsonify({
        "status":  "ok",
        "version": "release2",
        "capabilities": ["bl", "invoice", "packing_list"],
        "max_documents": 3,
        "cross_validation": True,
    })


@app.route("/commercial/validate", methods=["POST"])
def validate_commercial_docs():
    """
    Validation multi-documents commerciaux.

    Accepte 1 à 3 fichiers via multipart/form-data (champ 'files').
    Retourne le résultat complet du pipeline release2 :
      - Classification + extraction par document
      - Validation croisée inter-documents (règles, sémantique, NLI, tables)
    """
    files = request.files.getlist("files")
    if not files:
        # Compatibilité : essayer le champ 'file' unique
        single = request.files.get("file")
        if single:
            files = [single]
        else:
            return jsonify({"success": False, "error": "Aucun fichier reçu (champ 'files' attendu)"}), 400

    if len(files) > 3:
        return jsonify({"success": False, "error": "Maximum 3 documents par appel"}), 400

    pipeline = _get_commercial_pipeline()
    if pipeline is None:
        return jsonify({"success": False, "error": "Pipeline commercial indisponible"}), 503

    temp_paths = []
    try:
        for f in files:
            if not f.filename:
                continue
            suffix = Path(f.filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS and suffix not in {".tif", ".tiff", ".bmp"}:
                return jsonify({
                    "success": False,
                    "error": f"Type non supporté pour '{f.filename}'. Autorisés : PDF, PNG, JPG, TIFF, BMP",
                }), 400
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                f.save(tmp.name)
                temp_paths.append(tmp.name)

        if not temp_paths:
            return jsonify({"success": False, "error": "Aucun fichier valide reçu"}), 400

        logger.info("Pipeline commercial : traitement de %d document(s)", len(temp_paths))
        result = pipeline.process([Path(p) for p in temp_paths])
        result["success"] = True
        return jsonify(result)

    except Exception as exc:
        logger.error("Commercial pipeline failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": f"Erreur pipeline commercial : {exc}"}), 500
    finally:
        for p in temp_paths:
            cleanup(p)


# ──────────────────────────────────────────────────────────────────────
# LC-CROSS-01/02 : Routes validation croisée LC ↔ documents commerciaux
# ──────────────────────────────────────────────────────────────────────

_lc_commercial_validator = None


def _get_lc_commercial_validator():
    global _lc_commercial_validator
    if _lc_commercial_validator is None:
        try:
            release2_path = ROOT_DIR / "release2"
            if str(release2_path.parent) not in sys.path:
                sys.path.insert(0, str(release2_path.parent))
            from release2.services.lc_cross.lc_commercial_cross_validator import LcCommercialCrossValidator
            _lc_commercial_validator = LcCommercialCrossValidator()
            logger.info("LcCommercialCrossValidator (release2) chargé.")
        except Exception as exc:
            logger.warning("LcCommercialCrossValidator non disponible : %s", exc)
            _lc_commercial_validator = False
    return _lc_commercial_validator if _lc_commercial_validator else None


@app.route("/lc-commercial/health", methods=["GET"])
def lc_commercial_health():
    """Statut du validateur LC vs documents commerciaux."""
    validator = _get_lc_commercial_validator()
    if validator is None:
        return jsonify({
            "status": "unavailable",
            "reason": "release2/services/lc_cross non chargé",
        }), 503
    return jsonify({
        "status":  "ok",
        "version": "release2",
        "layers":  ["rules", "fuzzy", "semantic", "conditions"],
        "weights": {"rules": 0.40, "fuzzy": 0.20, "semantic": 0.25, "conditions": 0.15},
    })


@app.route("/lc-commercial/validate", methods=["POST"])
def validate_lc_commercial():
    """
    Validation croisée LC (MT700) vs Documents Commerciaux.

    Body JSON :
      {
        "lc":         { ... champs MT700 extraits de SwiftAnalysis ... },
        "commercial": {
          "bl":           { ... champs BL ... },
          "invoice":      { ... champs Invoice ... },
          "packing_list": { ... champs PL ... },
          "invoice_tables": [...],
          "bl_tables":      [...],
          "pl_tables":      [...]
        }
      }

    Retourne :
      final_score, risk_level, rule_results, fuzzy_results,
      semantic_results, condition_results, score_breakdown, explanations
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"success": False, "error": "Body JSON attendu"}), 400

    lc_data         = body.get("lc")
    commercial_data = body.get("commercial")

    if not lc_data:
        return jsonify({"success": False, "error": "Champ 'lc' manquant"}), 400
    if not commercial_data:
        return jsonify({"success": False, "error": "Champ 'commercial' manquant"}), 400

    validator = _get_lc_commercial_validator()
    if validator is None:
        return jsonify({"success": False, "error": "Validateur LC-Commercial indisponible"}), 503

    try:
        result = validator.validate(lc_data, commercial_data)
        result["success"] = True
        return jsonify(result)
    except Exception as exc:
        logger.error("LC-Commercial validation failed: %s", exc, exc_info=True)
        return jsonify({"success": False, "error": f"Erreur validation LC-Commercial : {exc}"}), 500


# ──────────────────────────────────────────────────────────────────────
# Warmup — précharge TOUS les modèles au démarrage (pas pendant les requêtes)
# ──────────────────────────────────────────────────────────────────────
_warmed_up = False
# Statut par composant, exposé dans /health → readiness vérifiable de l'extérieur
_WARMUP_STATUS: dict = {}

def warmup_all() -> None:
    """
    Charge en mémoire, une seule fois, l'ensemble des modèles lourds :
      - NLP (SentenceTransformer, NER, clause-clf, risk-clf, NLI)
      - RiskMLEngine (XGBoost + SHAP)
      - ExternalValidationEngine
      - Pipeline commercial release2 (LayoutLM BL/Invoice/PL, Donut)
    Ainsi chaque requête réutilise les modèles déjà chargés → pas de rechargement.
    Désactivable via WARMUP_ON_START=false.
    """
    global _warmed_up
    if _warmed_up:
        return
    _warmed_up = True

    if os.getenv("WARMUP_ON_START", "true").lower() != "true":
        logger.info("Warmup désactivé (WARMUP_ON_START=false).")
        return

    import time as _t
    t0 = _t.time()
    logger.info("⏳ Warmup : préchargement des modèles…")

    # 1. NLP (embeddings + NER + classifieurs + NLI)
    try:
        _ = pipeline_service.nlp
        _WARMUP_STATUS["nlp"] = "ok"
        logger.info("  ✓ NLP préchargé")
    except Exception as e:
        _WARMUP_STATUS["nlp"] = f"failed: {e}"
        logger.warning("  ✗ NLP warmup: %s", e)

    # 2. Risk ML Engine (XGBoost + SHAP)
    try:
        _ = pipeline_service.risk_engine
        _WARMUP_STATUS["risk_ml"] = "ok"
        logger.info("  ✓ RiskMLEngine préchargé")
    except Exception as e:
        _WARMUP_STATUS["risk_ml"] = f"failed: {e}"
        logger.warning("  ✗ RiskMLEngine warmup: %s", e)

    # 3. Validation externe (cache sanctions)
    try:
        _ = pipeline_service.ext_engine
        _WARMUP_STATUS["external_validation"] = "ok"
        logger.info("  ✓ ExternalValidation préchargé")
    except Exception as e:
        _WARMUP_STATUS["external_validation"] = f"failed: {e}"
        logger.warning("  ✗ ExternalValidation warmup: %s", e)

    # 4. Pipeline commercial release2 (LayoutLM/Donut — le plus lourd)
    try:
        ok = _get_commercial_pipeline() is not None
        _WARMUP_STATUS["commercial_pipeline"] = "ok" if ok else "unavailable"
        logger.info("  ✓ Pipeline commercial préchargé")
    except Exception as e:
        _WARMUP_STATUS["commercial_pipeline"] = f"failed: {e}"
        logger.warning("  ✗ Commercial warmup: %s", e)

    # 5. Validateur LC ↔ commercial + son encodeur ST (sinon chargé à la
    #    1ère requête utilisateur → latence de plusieurs secondes)
    try:
        validator = _get_lc_commercial_validator()
        if validator is not None and hasattr(validator, "_get_st_model"):
            validator._get_st_model().encode("warmup")
        _WARMUP_STATUS["lc_commercial_validator"] = "ok" if validator else "unavailable"
        logger.info("  ✓ LC-Commercial validator préchargé")
    except Exception as e:
        _WARMUP_STATUS["lc_commercial_validator"] = f"failed: {e}"
        logger.warning("  ✗ LC-Commercial warmup: %s", e)

    logger.info("✅ Warmup terminé en %.1fs — modèles en mémoire.", _t.time() - t0)


# Exécuté à l'import du module → couvre `flask run` (Docker) ET `python app/api.py`.
warmup_all()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode)
