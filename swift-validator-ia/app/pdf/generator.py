"""
generator.py — Génération PDF avec WeasyPrint + Jinja2.
Deux types de rapports :
  - MT700_REPORT  : rapport de validation MT700 seul
  - FULL_REPORT   : rapport complet (MT700 + docs commerciaux + cohérence)
"""

import logging
from pathlib import Path
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
CSS_FILE      = TEMPLATES_DIR / "styles" / "report.css"


def _build_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )


def _register_gtk_dll_dir() -> None:
    """
    Sous Windows, enregistre le dossier bin du runtime GTK3 pour que WeasyPrint
    charge les bonnes DLLs (libgobject/pango/cairo) plutôt que celles, incomplètes,
    de Tesseract présentes sur le PATH. No-op sous Linux/Docker.

    Priorité : variable d'env GTK_BIN_DIR, sinon emplacements d'install classiques.
    """
    import os, sys
    if sys.platform != "win32":
        return
    candidates = []
    env_dir = os.getenv("GTK_BIN_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates += [
        r"C:\Program Files\GTK3-Runtime Win64\bin",
        r"C:\Program Files\GTK3-Runtime\bin",
        r"C:\msys64\mingw64\bin",
    ]
    for d in candidates:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)            # priorité de recherche DLL
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                logger.info("GTK DLL directory enregistré : %s", d)
                return
            except Exception as e:
                logger.warning("Échec add_dll_directory(%s): %s", d, e)


def _render_pdf(html_str: str) -> bytes:
    # Windows : forcer la découverte des DLLs GTK3 correctes AVANT l'import weasyprint,
    # sinon weasyprint charge la libgobject incomplète de Tesseract (sur le PATH) → erreur.
    _register_gtk_dll_dir()

    # Import paresseux : WeasyPrint exige des libs natives GTK/Pango/Cairo.
    # Sur Windows sans GTK, l'import échoue — on ne veut PAS planter toute l'app,
    # seulement les endpoints PDF. En Docker, les libs sont installées (Dockerfile).
    try:
        from weasyprint import HTML, CSS
    except OSError as exc:
        raise RuntimeError(
            "WeasyPrint indisponible : librairies natives GTK/Pango/Cairo manquantes. "
            "Sous Windows : installez le runtime GTK3 puis définissez GTK_BIN_DIR vers son dossier 'bin'. "
            "Voir https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows. "
            "En Docker, c'est déjà géré."
        ) from exc

    css = CSS(filename=str(CSS_FILE))
    return HTML(string=html_str, base_url=str(TEMPLATES_DIR)).write_pdf(
        stylesheets=[css]
    )


def _base_context(data: dict) -> dict:
    """Contexte commun aux deux rapports."""
    return {
        "generated_at": datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC"),
        "lc_number":    data.get("lc_number", "—"),
        "applicant":    data.get("applicant", "—"),
        "beneficiary":  data.get("beneficiary", "—"),
        "issuing_bank": data.get("issuing_bank", "—"),
        "currency":     data.get("currency", "—"),
        "amount":       data.get("amount"),
        "issue_date":   data.get("issue_date", "—"),
        "expiry_date":  data.get("expiry_date", "—"),
        "risk_level":   data.get("risk_level", "—"),
        "risk_score":   data.get("risk_score"),
        "ml_risk_score": data.get("ml_risk_score"),
        "ucp_score":    data.get("ucp_compliance_score"),
        "nlp_score":    data.get("nlp_score"),
        "completeness_score": data.get("completeness_score"),
        "payment_blocked": data.get("payment_blocked", False),
        "n_ucp_errors":   data.get("n_ucp_errors", 0),
        "n_ucp_warnings": data.get("n_ucp_warnings", 0),
        "ucp_findings":   data.get("ucp_findings", []),
        "nlp_clauses":    data.get("nlp_clauses", []),
        "risk_ml":        data.get("risk_ml", {}),
        "external_validation": data.get("external_validation", {}),
    }


def generate_mt700_report(data: dict) -> bytes:
    """
    Génère le rapport PDF de validation MT700.
    data = payload complet de l'analyse (full_payload ou construit par Spring Boot).
    """
    env  = _build_env()
    tmpl = env.get_template("mt700_report.html")
    ctx  = _base_context(data)
    ctx.update({
        "report_type":  "Rapport de Validation MT700",
        "report_badge": "MT700",
        "incoterm":     data.get("incoterm", "—"),
        "country":      data.get("country", "—"),
        "validity_days": data.get("validity_days"),
        "presentation_period_days": data.get("presentation_period_days"),
        "goods_description": data.get("goods_description", ""),
        "documents_required": data.get("documents_required", ""),
    })
    html = tmpl.render(**ctx)
    logger.info("Génération PDF MT700 pour LC %s", data.get("lc_number"))
    return _render_pdf(html)


def generate_full_report(data: dict) -> bytes:
    """
    Génère le rapport PDF de validation complète.
    data = payload complet incluant commercial_validation.
    """
    env  = _build_env()
    tmpl = env.get_template("full_report.html")
    ctx  = _base_context(data)
    ctx.update({
        "report_type":  "Rapport de Validation Complète",
        "report_badge": "COMPLET",
        "incoterm":     data.get("incoterm", "—"),
        "country":      data.get("country", "—"),
        "validity_days": data.get("validity_days"),
        "presentation_period_days": data.get("presentation_period_days"),
        "goods_description":  data.get("goods_description", ""),
        "documents_required": data.get("documents_required", ""),
        # Docs commerciaux
        "commercial_validation": data.get("commercial_validation", {}),
        "commercial_documents":  data.get("commercial_documents", []),
        "cross_validation_report": data.get("cross_validation_report", {}),
        # Cohérence LC/MT700 ↔ documents commerciaux (validation croisée LC)
        "lc_coherence":          data.get("lc_coherence", {}),
    })
    html = tmpl.render(**ctx)
    logger.info("Génération PDF FULL pour LC %s", data.get("lc_number"))
    return _render_pdf(html)
