"""
routes.py — Blueprint Flask pour la génération de rapports PDF.

Endpoints :
  POST /pdf/mt700/<analysis_id>  — Génère et upload le rapport MT700
  POST /pdf/full/<analysis_id>   — Génère et upload le rapport complet
  GET  /pdf/url/<object_name>    — Régénère une URL présignée (valide 7j)
  GET  /pdf/health               — Statut du service PDF
"""

import logging
from flask import Blueprint, request, jsonify

from .generator     import generate_mt700_report, generate_full_report
from .minio_storage import get_storage, MinioReportStorage

logger = logging.getLogger(__name__)

pdf_bp = Blueprint("pdf", __name__, url_prefix="/pdf")


def _generate_and_upload(
    analysis_id: str,
    data: dict,
    report_type: str,   # "MT700_REPORT" | "FULL_REPORT"
    generator_fn,
) -> dict:
    storage = get_storage()
    pdf_bytes   = generator_fn(data)
    object_name = MinioReportStorage.build_object_name(analysis_id, report_type)
    storage.upload(
        pdf_bytes,
        object_name,
        metadata={"analysis-id": analysis_id, "report-type": report_type},
    )
    url = storage.presigned_url(object_name, days=7)
    return {
        "object_name": object_name,
        "download_url": url,
        "file_size": len(pdf_bytes),
        "report_type": report_type,
    }


@pdf_bp.post("/mt700/<analysis_id>")
def generate_mt700(analysis_id: str):
    """Génère le rapport MT700 et l'uploade dans MinIO."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = _generate_and_upload(
            analysis_id, data, "MT700_REPORT", generate_mt700_report
        )
        return jsonify(result), 201
    except Exception as exc:
        logger.exception("Erreur génération PDF MT700 pour %s", analysis_id)
        return jsonify({"error": str(exc)}), 500


@pdf_bp.post("/full/<analysis_id>")
def generate_full(analysis_id: str):
    """Génère le rapport complet et l'uploade dans MinIO."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = _generate_and_upload(
            analysis_id, data, "FULL_REPORT", generate_full_report
        )
        return jsonify(result), 201
    except Exception as exc:
        logger.exception("Erreur génération PDF FULL pour %s", analysis_id)
        return jsonify({"error": str(exc)}), 500


@pdf_bp.get("/url/<path:object_name>")
def refresh_url(object_name: str):
    """Régénère une URL présignée valide 7 jours pour un objet existant."""
    try:
        url = get_storage().presigned_url(object_name, days=7)
        return jsonify({"download_url": url, "object_name": object_name})
    except Exception as exc:
        logger.exception("Erreur refresh URL pour %s", object_name)
        return jsonify({"error": str(exc)}), 500


@pdf_bp.get("/health")
def health():
    try:
        storage = get_storage()
        exists  = storage.client.bucket_exists(storage.bucket)
        return jsonify({"status": "ok", "minio_bucket": storage.bucket, "accessible": exists})
    except Exception as exc:
        return jsonify({"status": "error", "detail": str(exc)}), 500
