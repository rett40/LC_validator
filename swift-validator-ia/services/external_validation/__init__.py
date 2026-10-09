"""
external_validation_b
=====================
Architecture v4 — cache local JSON + fallback APIs web
Répertoire de données : data/scraping/

Structure :
  data/scraping/
    latest/   ← fichiers courants (toujours à jour)
    archive/  ← archives mensuelles (country_risk_YYYY-MM.json)

Modules :
  scraper.py        — Construit le dataset pays/risque/devises (mise à jour mensuelle)
  cache_manager.py  — Accès au cache JSON + validation pays & devise
  scheduler.py      — Planificateur mensuel (APScheduler)
  country_checker   — Validation pays (cache → API)
  engine.py         — Orchestrateur principal

Sources gérées :
  - FATF Black & Grey Lists     → fatf_lists.json
  - OFAC SDN                    → ofac_sdn.json
  - EU Sanctions                → eu_sanctions.json
  - UN Security Council         → un_sanctions.json
  - Basel AML Index             → intégré dans country_risk_latest.json
  - TI Corruption Perception    → intégré dans country_risk_latest.json
  - ISO 4217 Currency reference → intégré dans country_risk_latest.json

Usage rapide :
    from external_validation_b import start_scheduler, ExternalValidationEngine
    start_scheduler()                         # Refresh mensuel automatique

    engine = ExternalValidationEngine()
    result = engine.validate(pipeline_output)

    # Validation standalone d'un pays + devise
    from external_validation_b import validate_country, validate_currency
    print(validate_country("TN"))
    print(validate_currency("TND", country_alpha2="TN"))

    # Statut du cache
    from external_validation_b import cache_status
    print(cache_status())

    # Refresh manuel
    from external_validation_b import run_refresh
    run_refresh(force=True)

Variables d'environnement :
    SANCTIONS_DATA_DIR     — répertoire racine (défaut: data/scraping)
    CACHE_MAX_AGE_DAYS     — durée de validité en jours (défaut: 30)
    INCLUDE_GLEIF_SNAPSHOT — activer snapshot GLEIF (défaut: 0 → 1 pour activer)
"""

from .engine import ExternalValidationEngine, ExternalValidationResult
from .country_checker import check_country, check_all_countries, RiskLevel
from .cache_manager import (
    cache_status,
    invalidate_all,
    validate_country,
    validate_currency,
    validate_pair,
)
from .scheduler import start_scheduler, stop_scheduler, run_refresh, get_next_run
from .scraper import run_scraper

__all__ = [
    # Moteur principal
    "ExternalValidationEngine",
    "ExternalValidationResult",
    # Checkers
    "RiskLevel",
    "check_country",
    "check_all_countries",
    # Validation pays & devise
    "validate_country",
    "validate_currency",
    "validate_pair",
    # Cache & scheduler
    "cache_status",
    "invalidate_all",
    "start_scheduler",
    "stop_scheduler",
    "run_refresh",
    "get_next_run",
    # Scraper direct
    "run_scraper",
]