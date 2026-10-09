"""
external_validation_b/country_checker.py
Validation pays  –  cache local data/scraping/ + fallback APIs web

MODIFICATIONS :
  • Utilise search_risk_local() pour enrichir le résultat depuis country_risk_latest.json
  • Appel validate_pair() depuis cache_manager pour valider pays + devise
  • Fallback OFAC/EU/FATF inchangés
"""

from __future__ import annotations

import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    CLEAN    = "CLEAN"


# ─────────────────────────────────────────────────────────────────────────────
#  Mapping statique ISO-2 (fallback si cache absent)
# ─────────────────────────────────────────────────────────────────────────────
COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    "TUNISIA": "TN", "TUNISIE": "TN",
    "ITALY": "IT", "ITALIE": "IT",
    "FRANCE": "FR",
    "GERMANY": "DE", "ALLEMAGNE": "DE",
    "IRAN": "IR", "ISLAMIC REPUBLIC OF IRAN": "IR",
    "SYRIA": "SY", "SYRIAN ARAB REPUBLIC": "SY",
    "KOREA": "KP", "NORTH KOREA": "KP", "DPRK": "KP", "SOUTH KOREA": "KR",
    "RUSSIA": "RU", "RUSSIAN FEDERATION": "RU",
    "BELARUS": "BY",
    "MYANMAR": "MM", "BURMA": "MM",
    "CUBA": "CU", "VENEZUELA": "VE",
    "SUDAN": "SD", "SOMALIA": "SO",
    "UKRAINE": "UA",
    "UNITED STATES": "US", "USA": "US",
    "UNITED KINGDOM": "GB", "UK": "GB",
    "CHINA": "CN",
    "UAE": "AE", "UNITED ARAB EMIRATES": "AE",
}

FATF_BLACK_ISO2 = {"IR", "KP", "MM"}
FATF_GREY_ISO2  = {
    "BG", "BF", "CM", "HR", "CD", "HT", "JM", "KE", "ML",
    "MZ", "NA", "NG", "PH", "SN", "ZA", "SS", "SY", "TZ", "VN", "YE",
}

OFAC_SDN_URL = "https://sanctionssearch.ofac.treas.gov/api/search/v2/sdn"


@dataclass
class CountryResult:
    country_raw:         str
    country_iso2:        str
    risk_level:          RiskLevel
    sources:             list[str]       = field(default_factory=list)
    details:             str             = ""
    score_contribution:  int             = 0
    api_responses:       dict            = field(default_factory=dict)
    # Champs de validation
    country_valid:       bool            = True
    currency_code:       str             = ""
    currency_name:       str             = ""
    currency_valid:      bool            = True
    validation_warnings: list[str]       = field(default_factory=list)
    # Enrichissement depuis le cache de risque local
    cached_risk_level:   str             = ""
    cached_risk_score:   int             = 0


# ─────────────────────────────────────────────────────────────────────────────
#  Résolution ISO-2
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_to_iso2(country: str) -> str:
    c = country.strip().upper()
    if len(c) == 2:
        return c

    # 1. Cache local RestCountries
    try:
        from .cache_manager import lookup_country_local
        result = lookup_country_local(c)
        if result:
            return result["iso2"]
    except (ImportError, Exception):
        pass

    # 2. Mapping statique
    if c in COUNTRY_NAME_TO_ISO2:
        return COUNTRY_NAME_TO_ISO2[c]

    # 3. Recherche partielle
    for name, iso2 in COUNTRY_NAME_TO_ISO2.items():
        if name in c:
            return iso2

    return c  # Retour brut plutôt qu'un faux ISO-2


# ─────────────────────────────────────────────────────────────────────────────
#  Validation pays & devise
# ─────────────────────────────────────────────────────────────────────────────

def _validate_country_and_currency(iso2: str) -> dict:
    """
    Valide le code pays et récupère sa devise depuis le cache de risque local.
    Retourne un dict avec : country_valid, currency_code, currency_name,
    currency_valid, warnings.
    """
    result = {
        "country_valid":   True,
        "currency_code":   "",
        "currency_name":   "",
        "currency_valid":  True,
        "warnings":        [],
    }

    # Validation via cache_manager (qui délègue à scraper.py)
    try:
        from .cache_manager import validate_pair, search_risk_local
    except ImportError:
        from cache_manager import validate_pair, search_risk_local

    # Récupérer la devise depuis le cache de risque local
    risk_entry = search_risk_local(iso2)
    if risk_entry:
        result["currency_code"] = risk_entry.get("currency_code", "")
        result["currency_name"] = risk_entry.get("currency_name", "")

    # Valider le couple pays/devise
    pair = validate_pair(iso2, result["currency_code"])
    result["country_valid"]  = pair["country_valid"]
    result["currency_valid"] = pair["currency_valid"]
    result["warnings"]       = pair.get("warnings", [])

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  OFAC : cache local → fallback API
# ─────────────────────────────────────────────────────────────────────────────

def _check_ofac_country(iso2: str) -> dict:
    try:
        from .cache_manager import search_ofac_local
        result = search_ofac_local(iso2, min_score=0.85)
        country_hits = [
            e for e in result.get("entries", [])
            if iso2 in e.get("countries", [])
        ]
        return {
            "source":         result.get("source", "OFAC"),
            "sanctioned":     len(country_hits) > 5,
            "entity_count":   len(country_hits),
            "cache_age_days": result.get("cache_age_days"),
        }
    except (ImportError, Exception):
        pass

    # Fallback API
    payload = {"searchType": "consolidated", "query": iso2, "score": 0, "records": 50}
    try:
        resp = requests.post(OFAC_SDN_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data    = resp.json()
        entries = data.get("sdnList", {}).get("sdnEntry", [])
        country_hits = [
            e for e in entries
            if iso2 in str(e.get("nationalityList", "")).upper()
            or iso2 in str(e.get("citizenshipList", "")).upper()
        ]
        return {"source": "OFAC-API", "sanctioned": len(country_hits) > 5,
                "entity_count": len(country_hits)}
    except requests.HTTPError as e:
        return {"source": "OFAC-API",
                "error": f"HTTP {e.response.status_code}", "sanctioned": False}
    except requests.RequestException as e:
        return {"source": "OFAC-API", "error": str(e), "sanctioned": False}


# ─────────────────────────────────────────────────────────────────────────────
#  EU Sanctions : cache local → fallback API XML
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _fetch_eu_sanctions_xml() -> ET.Element | None:
    url = (
        "https://webgate.ec.europa.eu/fsd/fsf/public/files/"
        "xmlFullSanctionsList_1_1/content"
    )
    try:
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        return ET.fromstring(resp.content)
    except Exception as e:
        print(f"[WARN] EU Sanctions list fetch failed: {e}")
        return None


def _check_eu_country(iso2: str) -> dict:
    try:
        from .cache_manager import get_cache
        cf = get_cache("eu")
        if cf.available:
            count = sum(
                1 for entry in (cf.data or [])
                if iso2 in entry.get("countries", [])
            )
            return {
                "source":         "EU-LOCAL",
                "sanctioned":     count > 10,
                "entity_count":   count,
                "cache_age_days": round(cf.age_days, 1) if cf.age_days else None,
            }
    except (ImportError, Exception):
        pass

    # Fallback API XML
    root = _fetch_eu_sanctions_xml()
    if root is None:
        return {"source": "EU-API", "error": "EU list unavailable", "sanctioned": False}

    ns = {"eu": "http://eu.europa.ec/fsd/export/xmlFullSanctionsList_1_1"}
    count = 0
    for entity in root.findall(".//eu:sanctionEntity", ns):
        for citizen in entity.findall(".//eu:citizenship", ns):
            el = citizen.find("eu:countryIso2Code", ns)
            if el is not None and el.text == iso2:
                count += 1
    return {"source": "EU-API", "sanctioned": count > 10, "entity_count": count}


# ─────────────────────────────────────────────────────────────────────────────
#  FATF : cache local → fallback statique
# ─────────────────────────────────────────────────────────────────────────────

def _check_fatf_country(iso2: str) -> dict:
    try:
        from .cache_manager import check_fatf_local
        return check_fatf_local(iso2)
    except (ImportError, Exception):
        pass

    if iso2 in FATF_BLACK_ISO2:
        return {"source": "FATF-STATIC", "list": "BLACK",
                "description": "Juridiction à haut risque (appel à l'action)"}
    if iso2 in FATF_GREY_ISO2:
        return {"source": "FATF-STATIC", "list": "GREY",
                "description": "Juridiction sous surveillance renforcée"}
    return {"source": "FATF-STATIC", "list": "NONE", "description": "Aucune restriction FATF"}


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

def check_country(country_raw: str) -> CountryResult:
    iso2 = _normalize_to_iso2(country_raw)

    sources: list[str] = []
    details_parts: list[str] = []
    score = 0
    api_responses: dict = {}

    # ── Enrichissement depuis le cache de risque local ─────────────────────
    cached_risk_level = ""
    cached_risk_score = 0
    try:
        from .cache_manager import search_risk_local
        risk_entry = search_risk_local(iso2)
        if risk_entry:
            cached_risk_level = risk_entry.get("risk_level", "")
            cached_risk_score = risk_entry.get("risk_score", 0)
            api_responses["risk_cache"] = {
                "source":      "RISK-LOCAL",
                "risk_level":  cached_risk_level,
                "risk_score":  cached_risk_score,
                "sources":     risk_entry.get("sources", ""),
            }
    except (ImportError, Exception):
        pass

    # ── Validation pays + devise ───────────────────────────────────────────
    val = _validate_country_and_currency(iso2)
    api_responses["validation"] = val

    # ── OFAC ──────────────────────────────────────────────────────────────
    ofac = _check_ofac_country(iso2)
    api_responses["ofac"] = ofac
    if ofac.get("sanctioned"):
        score += 10
        sources.append(ofac.get("source", "OFAC"))
        details_parts.append(f"Programmes OFAC : {iso2}")

    # ── EU Sanctions ───────────────────────────────────────────────────────
    eu = _check_eu_country(iso2)
    api_responses["eu"] = eu
    if eu.get("sanctioned"):
        score += 8
        sources.append(eu.get("source", "EU-SANCTIONS"))
        details_parts.append(f"Entités UE sanctionnées : {eu.get('entity_count', 0)}")

    # ── FATF ───────────────────────────────────────────────────────────────
    fatf = _check_fatf_country(iso2)
    api_responses["fatf"] = fatf
    if fatf.get("list") == "BLACK":
        score += 10
        sources.append("FATF-BLACK")
        details_parts.append("Liste noire FATF")
    elif fatf.get("list") == "GREY":
        score += 5
        sources.append("FATF-GREY")
        details_parts.append("Liste grise FATF")

    # ── Escalade depuis le cache de risque ─────────────────────────────────
    # Si le cache indique CRITICAL et le score courant ne le serait pas → escalade
    if cached_risk_level == "CRITICAL" and score < 10:
        score = max(score, 10)

    # ── Niveau de risque final ─────────────────────────────────────────────
    if score >= 10:
        level = RiskLevel.CRITICAL
    elif score >= 7:
        level = RiskLevel.HIGH
    elif score >= 4:
        level = RiskLevel.MEDIUM
    elif score > 0:
        level = RiskLevel.LOW
    else:
        level = RiskLevel.CLEAN

    return CountryResult(
        country_raw          = country_raw,
        country_iso2         = iso2,
        risk_level           = level,
        sources              = sources,
        details              = " | ".join(details_parts) if details_parts else "Aucune sanction détectée",
        score_contribution   = score,
        api_responses        = api_responses,
        country_valid        = val["country_valid"],
        currency_code        = val["currency_code"],
        currency_name        = val["currency_name"],
        currency_valid       = val["currency_valid"],
        validation_warnings  = val["warnings"],
        cached_risk_level    = cached_risk_level,
        cached_risk_score    = cached_risk_score,
    )


def check_all_countries(countries: list[str]) -> list[CountryResult]:
    return [check_country(c) for c in countries if c]