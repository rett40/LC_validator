"""
external_validation_b/cache_manager.py
Gestionnaire de cache  –  répertoire : data/scraping/

MODIFICATIONS :
  • DATA_DIR par défaut → data/scraping/latest/
  • CACHE_MAX_AGE → 30 jours (mensuel, aligné avec le scheduler)
  • Ajout de validate_country() et validate_currency() exposés depuis scraper
  • cache_status() inclut des infos sur l'archive mensuelle
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("cache_manager")

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Les fichiers "latest" sont dans data/scraping/latest/
# Les archives mensuelles sont dans data/scraping/archive/
# Chemin ancré au module (parents[2] = racine swift-validator-ia), JAMAIS au CWD.
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_SCRAPING_ROOT = Path(os.environ.get("SANCTIONS_DATA_DIR", str(_PROJECT_ROOT / "data" / "scraping")))
DATA_DIR       = _SCRAPING_ROOT / "latest"
ARCHIVE_DIR    = _SCRAPING_ROOT / "archive"
CACHE_MAX_AGE  = timedelta(days=int(os.environ.get("CACHE_MAX_AGE_DAYS", "30")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers attendus dans latest/
FILES = {
    "risk":      "country_risk_latest.json",      # ← produit par scraper.py
    "ofac":      "ofac_sdn.json",                  # produit par scraper OFAC
    "eu":        "eu_sanctions.json",              # produit par scraper EU
    "un":        "un_sanctions.json",              # produit par scraper UN
    "fatf":      "fatf_lists.json",                # produit par scraper FATF
    "countries": "country_reference.json",         # référentiel pays
    "gleif":     "gleif_snapshot.json",            # optionnel
}

STOP_WORDS = {"THE", "AND", "OF", "FOR", "CO", "LTD", "LLC", "INC", "SA", "SARL", "SPA", "CORP"}


# ─────────────────────────────────────────────────────────────────────────────
#  CacheFile  (lecture paresseuse d'un fichier JSON)
# ─────────────────────────────────────────────────────────────────────────────

class CacheFile:
    """Représente un fichier de cache JSON chargé en mémoire (lazy)."""

    def __init__(self, key: str):
        self.key     = key
        self.path    = DATA_DIR / FILES[key]
        self._data   = None
        self._meta   = {}
        self._loaded = False

    def _load(self) -> bool:
        if self._loaded:
            return True
        if not self.path.exists():
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Support deux formats :
            #   Format scraper risk  : liste directe
            #   Format scraper OFAC/EU/UN : {"updated_at": ..., "count": ..., "data": [...]}
            if isinstance(payload, list):
                self._data = payload
                self._meta = {"updated_at": None, "count": len(payload)}
            else:
                self._meta = {
                    "updated_at": payload.get("updated_at"),
                    "count":      payload.get("count") or len(payload.get("data", [])),
                }
                self._data = payload.get("data", payload)
            self._loaded = True
            return True
        except Exception as e:
            log.warning("Impossible de charger %s : %s", self.path, e)
            return False

    @property
    def available(self) -> bool:
        return self._load()

    @property
    def data(self) -> Any:
        self._load()
        return self._data

    @property
    def updated_at(self) -> datetime | None:
        self._load()
        ts = self._meta.get("updated_at")
        if not ts:
            # Fallback : date de modification du fichier
            try:
                mtime = self.path.stat().st_mtime
                return datetime.fromtimestamp(mtime, tz=timezone.utc)
            except Exception:
                return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return None

    @property
    def age_days(self) -> float | None:
        ua = self.updated_at
        if ua is None:
            return None
        now = datetime.now(timezone.utc)
        if ua.tzinfo is None:
            ua = ua.replace(tzinfo=timezone.utc)
        return (now - ua).total_seconds() / 86400

    @property
    def needs_refresh(self) -> bool:
        age = self.age_days
        return age is None or age > CACHE_MAX_AGE.days

    def invalidate(self):
        self._loaded = False
        self._data   = None
        self._meta   = {}


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton
# ─────────────────────────────────────────────────────────────────────────────

_cache: dict[str, CacheFile] = {k: CacheFile(k) for k in FILES}


def get_cache(key: str) -> CacheFile:
    return _cache[key]


def invalidate_all():
    for cf in _cache.values():
        cf.invalidate()


def cache_status() -> dict:
    status = {}
    for key, cf in _cache.items():
        if cf.available:
            status[key] = {
                "available":     True,
                "path":          str(cf.path),
                "updated_at":    cf._meta.get("updated_at") or (
                    cf.updated_at.isoformat() if cf.updated_at else None
                ),
                "count":         cf._meta.get("count"),
                "age_days":      round(cf.age_days, 1) if cf.age_days is not None else None,
                "needs_refresh": cf.needs_refresh,
                "refresh_every": f"{CACHE_MAX_AGE.days} jours",
            }
        else:
            status[key] = {
                "available":     False,
                "path":          str(cf.path),
                "needs_refresh": True,
                "refresh_every": f"{CACHE_MAX_AGE.days} jours",
            }

    # Infos sur les archives mensuelles disponibles
    archives = sorted(ARCHIVE_DIR.glob("country_risk_*.json"))
    status["_archives"] = {
        "count":  len(archives),
        "months": [p.stem.replace("country_risk_", "") for p in archives],
        "dir":    str(ARCHIVE_DIR),
    }
    return status


# ─────────────────────────────────────────────────────────────────────────────
#  Validation pays & devise  (délègue à scraper.py)
# ─────────────────────────────────────────────────────────────────────────────

def validate_country(alpha2: str) -> dict:
    """Valide un code pays ISO-3166-1 alpha-2. Délègue à scraper.validate_country_code."""
    try:
        from .scraper import validate_country_code
    except ImportError:
        from scraper import validate_country_code
    return validate_country_code(alpha2)


def validate_currency(iso_code: str, country_alpha2: str = "") -> dict:
    """Valide un code devise ISO-4217. Délègue à scraper.validate_currency_code."""
    try:
        from .scraper import validate_currency_code
    except ImportError:
        from scraper import validate_currency_code
    return validate_currency_code(iso_code, country_alpha2)


def validate_pair(alpha2: str, currency_code: str) -> dict:
    """Valide le couple pays/devise. Délègue à scraper.validate_country_currency_pair."""
    try:
        from .scraper import validate_country_currency_pair
    except ImportError:
        from scraper import validate_country_currency_pair
    return validate_country_currency_pair(alpha2, currency_code)


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitaires de recherche textuelle
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    n = name.upper().strip()
    n = unicodedata.normalize("NFKD", n)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r'[^\w\s]', ' ', n)
    return re.sub(r'\s+', ' ', n).strip()


def _tokens(name: str, min_len: int = 4) -> set[str]:
    return {t for t in _normalize(name).split()
            if t not in STOP_WORDS and len(t) >= min_len}


def _match_score(query_tokens: set[str], entry_tokens: set[str]) -> float:
    if not query_tokens or not entry_tokens:
        return 0.0
    common      = query_tokens & entry_tokens
    long_common = {t for t in common if len(t) >= 7}
    base_score  = len(common) / max(len(query_tokens), len(entry_tokens))
    return min(1.0, base_score + 0.2 * len(long_common))


# ─────────────────────────────────────────────────────────────────────────────
#  Recherches dans les caches
# ─────────────────────────────────────────────────────────────────────────────

def search_risk_local(alpha2: str) -> dict | None:
    """
    Recherche l'entrée de risque d'un pays par son code ISO-2
    dans le fichier country_risk_latest.json.
    """
    cf = get_cache("risk")
    if not cf.available:
        return None
    alpha2u = alpha2.upper()
    if isinstance(cf.data, list):
        for entry in cf.data:
            if entry.get("country_code") == alpha2u:
                return {**entry, "source": "RISK-LOCAL",
                        "cache_age_days": round(cf.age_days, 1) if cf.age_days else None}
    return None


def search_ofac_local(
    query: str,
    entity_type: str | None = None,
    min_score: float = 0.45,
    max_results: int = 5,
) -> dict:
    cf = get_cache("ofac")
    if not cf.available:
        return {"source": "OFAC-LOCAL", "matched": False,
                "error": "Cache OFAC absent — lancer le scraper"}

    qt = _tokens(query)
    if not qt:
        return {"source": "OFAC-LOCAL", "matched": False, "entries": []}

    results = []
    for entry in (cf.data or []):
        if entity_type and entry.get("type", "").lower() != entity_type.lower():
            continue
        entry_tokens = set(entry.get("search_tokens", [])) | set(entry.get("alias_tokens", []))
        score = _match_score(qt, entry_tokens)
        if score >= min_score:
            results.append({
                "name":      entry["name"],
                "type":      entry.get("type", ""),
                "uid":       entry.get("uid", ""),
                "programs":  entry.get("programs", []),
                "score":     round(score * 100),
                "countries": entry.get("countries", []),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "source":         "OFAC-LOCAL",
        "matched":        len(results) > 0,
        "count":          len(results),
        "entries":        results[:max_results],
        "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
    }


def search_ofac_bic(bic: str) -> dict:
    return search_ofac_local(bic.strip().upper()[:8], min_score=0.8)


def search_eu_local(
    query: str,
    entity_only: bool = False,
    min_score: float = 0.40,
    max_results: int = 5,
) -> dict:
    cf = get_cache("eu")
    if not cf.available:
        return {"source": "EU-LOCAL", "matched": False,
                "error": "Cache EU absent — lancer le scraper"}

    qt = _tokens(query)
    if not qt:
        return {"source": "EU-LOCAL", "matched": False, "entries": []}

    results = []
    for entry in (cf.data or []):
        if entity_only and entry.get("subject_type") != "E":
            continue
        entry_tokens = set(entry.get("search_tokens", []))
        score = _match_score(qt, entry_tokens)
        if score >= min_score:
            results.append({
                "name":           entry["primary_name"],
                "logical_id":     entry.get("logical_id", ""),
                "regulation_url": entry.get("regulation_url", ""),
                "score":          round(score * 100),
                "countries":      entry.get("countries", []),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "source":         "EU-LOCAL",
        "matched":        len(results) > 0,
        "count":          len(results),
        "entries":        results[:max_results],
        "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
    }


def search_un_local(
    query: str,
    entity_only: bool = False,
    min_score: float = 0.40,
    max_results: int = 5,
) -> dict:
    cf = get_cache("un")
    if not cf.available:
        return {"source": "UN-LOCAL", "matched": False,
                "error": "Cache UN absent — lancer le scraper"}

    qt = _tokens(query)
    if not qt:
        return {"source": "UN-LOCAL", "matched": False, "entries": []}

    results = []
    for entry in (cf.data or []):
        if entity_only and entry.get("type") != "ENTITY":
            continue
        entry_tokens = set(entry.get("search_tokens", []))
        score = _match_score(qt, entry_tokens)
        if score >= min_score:
            results.append({
                "name":  entry["name"],
                "un_id": entry.get("data_id", ""),
                "type":  entry.get("type", ""),
                "score": round(score * 100),
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "source":         "UN-LOCAL",
        "matched":        len(results) > 0,
        "count":          len(results),
        "entries":        results[:max_results],
        "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
    }


def check_fatf_local(iso2: str) -> dict:
    """Vérifie le statut FATF — cache local en priorité, fallback statique."""
    cf = get_cache("fatf")

    # ── Fallback statique si cache absent ─────────────────────────────────
    if not cf.available:
        BLACK = {"IR", "KP", "MM"}
        GREY  = {
            "BG", "BF", "CM", "HR", "CD", "HT", "JM", "KE", "ML",
            "MZ", "NA", "NG", "PH", "SN", "ZA", "SS", "SY", "TZ", "VN", "YE",
        }
        iso2u = iso2.upper()
        if iso2u in BLACK:
            return {"source": "FATF-STATIC", "list": "BLACK",
                    "description": "Juridiction à haut risque"}
        if iso2u in GREY:
            return {"source": "FATF-STATIC", "list": "GREY",
                    "description": "Sous surveillance renforcée"}
        return {"source": "FATF-STATIC", "list": "NONE",
                "description": "Aucune restriction FATF"}

    # ── Cache disponible ───────────────────────────────────────────────────
    data  = cf.data
    iso2u = iso2.upper()

    if isinstance(data, dict):
        for entry in data.get("black_list", {}).get("countries", []):
            if entry.get("iso2") == iso2u:
                return {
                    "source": "FATF-LOCAL", "list": "BLACK",
                    "description": data["black_list"].get("description", ""),
                    "since":       entry.get("since", ""),
                    "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
                }
        for entry in data.get("grey_list", {}).get("countries", []):
            if entry.get("iso2") == iso2u:
                return {
                    "source": "FATF-LOCAL", "list": "GREY",
                    "description": data["grey_list"].get("description", ""),
                    "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
                }

    return {"source": "FATF-LOCAL", "list": "NONE",
            "description": "Aucune restriction FATF",
            "cache_age_days": round(cf.age_days, 1) if cf.age_days else None}


def lookup_gleif_local(bic: str) -> dict | None:
    cf = get_cache("gleif")
    if not cf.available:
        return None

    bic8      = bic.strip().upper()[:8]
    data      = cf.data
    bic_index = data.get("bic_index", {}) if isinstance(data, dict) else {}

    if bic8 in bic_index:
        entry = bic_index[bic8]
        return {
            "source":     "GLEIF-LOCAL",
            "found":      True,
            "lei":        entry.get("lei", ""),
            "legal_name": entry.get("legal_name", ""),
            "status":     entry.get("status", "UNKNOWN"),
            "country":    entry.get("country", ""),
            "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
        }
    return {"source": "GLEIF-LOCAL", "found": False,
            "status": "BIC introuvable dans le snapshot GLEIF local",
            "cache_age_days": round(cf.age_days, 1) if cf.age_days else None}


def lookup_country_local(name_or_iso2: str) -> dict | None:
    cf = get_cache("countries")
    if not cf.available:
        return None

    data = cf.data
    if not isinstance(data, dict):
        return None

    iso2_index = data.get("iso2_index", {})
    name_index = data.get("name_index", {})
    query      = name_or_iso2.strip().upper()

    if query in iso2_index:
        return iso2_index[query]
    if query in name_index:
        return iso2_index.get(name_index[query])
    for name, iso2 in name_index.items():
        if query in name or name in query:
            return iso2_index.get(iso2)
    return None