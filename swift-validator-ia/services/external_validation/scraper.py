"""
=============================================================
  scraper.py  –  Multi-Source Country Risk & Sanctions Builder
  Sauvegarde : data/scraping/
  Mise à jour : mensuelle (30 jours)
  Nouveautés  :
    • Répertoire unifié  data/scraping/
    • Validateur pays    (ISO-2, ISO-3, nom officiel via pycountry)
    • Validateur devise  (ISO-4217 via pycountry.currencies)
    • Fichier latest/    → toujours pointer vers la dernière version
    • Résumé JSON enrichi avec métadonnées de validation
=============================================================
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pycountry

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Répertoires de sortie
# ─────────────────────────────────────────────────────────────────────────────
# Chemin ancré au module (parents[2] = racine swift-validator-ia), JAMAIS au CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR      = Path(os.environ.get("SANCTIONS_DATA_DIR", str(_PROJECT_ROOT / "data" / "scraping")))
ARCHIVE_DIR   = BASE_DIR / "archive"
LATEST_DIR    = BASE_DIR / "latest"

for _d in (BASE_DIR, ARCHIVE_DIR, LATEST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m")   # Granularité mensuelle

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 1 – FATF   (ref: fatf-gafi.org – mis à jour juin 2025)
# ─────────────────────────────────────────────────────────────────────────────
FATF_HIGH_RISK_CALL_ACTION: set[str] = {
    "Iran", "North Korea", "Myanmar",
}
FATF_GREY_LIST: set[str] = {
    "Algeria", "Angola", "Bolivia", "Bulgaria", "Burkina Faso",
    "Cameroon", "Côte d'Ivoire", "Democratic Republic of the Congo",
    "Haiti", "Kenya", "Lao PDR", "Lebanon", "Mali", "Monaco",
    "Mozambique", "Namibia", "Nepal", "Nigeria", "South Africa",
    "South Sudan", "Syria", "Tanzania", "Trinidad and Tobago",
    "Venezuela", "Vietnam", "British Virgin Islands", "Yemen",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 2 – OFAC   (ref: ofac.treasury.gov – 2025)
# ─────────────────────────────────────────────────────────────────────────────
OFAC_COMPREHENSIVE_SANCTIONS: set[str] = {
    "Cuba", "Iran", "North Korea", "Syria",
    "Russia", "Belarus", "Venezuela",
}
OFAC_PARTIAL_SANCTIONS: set[str] = {
    "China", "Somalia", "Libya", "Sudan", "Zimbabwe",
    "Myanmar", "Eritrea", "Central African Republic",
    "Democratic Republic of the Congo", "Ethiopia",
    "Iraq", "Mali", "Haiti", "Nicaragua",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 3 – EU     (ref: EU Delegated Regulation 2016/1675 – 2025)
# ─────────────────────────────────────────────────────────────────────────────
EU_HIGH_RISK_AML: set[str] = {
    "Afghanistan", "Algeria", "Angola", "Burkina Faso", "Cameroon",
    "Côte d'Ivoire", "Democratic Republic of the Congo", "Haiti",
    "Kenya", "Laos", "Lebanon", "Mali", "Monaco", "Mozambique",
    "Myanmar", "Namibia", "Nepal", "Nigeria", "South Africa",
    "South Sudan", "Syria", "Tanzania", "Trinidad and Tobago",
    "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Russia",
    "Iran", "North Korea",
}
EU_RESTRICTIVE_MEASURES: set[str] = {
    "Russia", "Belarus", "Iran", "North Korea", "Syria",
    "Venezuela", "Myanmar", "Guinea", "Guinea-Bissau",
    "Libya", "Mali", "Nicaragua", "Somalia", "Sudan",
    "Zimbabwe", "Eritrea", "Central African Republic",
    "Democratic Republic of the Congo", "Haiti",
    "Moldova", "Georgia", "Bosnia and Herzegovina",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 4 – UN Security Council Sanctions (2025)
# ─────────────────────────────────────────────────────────────────────────────
UN_SANCTIONS: set[str] = {
    "North Korea", "Iran", "Libya", "Sudan", "Somalia",
    "Democratic Republic of the Congo", "Central African Republic",
    "Mali", "South Sudan", "Yemen", "Guinea-Bissau", "Ethiopia", "Haiti",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 5 – Basel AML Index 2024
# ─────────────────────────────────────────────────────────────────────────────
BASEL_CRITICAL: set[str] = {
    "Afghanistan", "Haiti", "Myanmar", "North Korea",
    "Syria", "South Sudan", "Yemen", "Venezuela",
}
BASEL_HIGH: set[str] = {
    "Angola", "Bolivia", "Burkina Faso", "Cameroon",
    "Central African Republic", "Côte d'Ivoire",
    "Democratic Republic of the Congo", "Eritrea",
    "Ethiopia", "Guinea", "Guinea-Bissau", "Iran",
    "Iraq", "Kenya", "Laos", "Lebanon", "Liberia",
    "Libya", "Madagascar", "Mali", "Moldova",
    "Mozambique", "Namibia", "Nepal", "Nicaragua",
    "Niger", "Nigeria", "Pakistan", "Papua New Guinea",
    "Russia", "Senegal", "Sierra Leone", "Somalia",
    "Sudan", "Tanzania", "Tajikistan", "Turkmenistan",
    "Uganda", "Uzbekistan", "Vietnam", "Zimbabwe",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SOURCE 6 – Transparency International CPI 2024 (score < 30)
# ─────────────────────────────────────────────────────────────────────────────
TI_HIGH_CORRUPTION: set[str] = {
    "Afghanistan", "Burundi", "Cameroon", "Central African Republic",
    "Chad", "Democratic Republic of the Congo", "Equatorial Guinea",
    "Eritrea", "Guinea-Bissau", "Haiti", "Iraq", "Libya",
    "Myanmar", "Nicaragua", "North Korea", "Pakistan",
    "Russia", "Somalia", "South Sudan", "Sudan", "Syria",
    "Tajikistan", "Turkmenistan", "Uzbekistan", "Venezuela",
    "Yemen", "Zimbabwe",
}

# ─────────────────────────────────────────────────────────────────────────────
#  ISO 4217 Currency Map  (alpha-2 pays → devise)
# ─────────────────────────────────────────────────────────────────────────────
CURRENCY_MAP: dict[str, tuple[str, str]] = {
    "AD": ("Euro", "EUR"), "AE": ("UAE Dirham", "AED"),
    "AF": ("Afghan Afghani", "AFN"), "AG": ("East Caribbean Dollar", "XCD"),
    "AI": ("East Caribbean Dollar", "XCD"), "AL": ("Albanian Lek", "ALL"),
    "AM": ("Armenian Dram", "AMD"), "AO": ("Angolan Kwanza", "AOA"),
    "AQ": ("No universal currency", ""), "AR": ("Argentine Peso", "ARS"),
    "AS": ("US Dollar", "USD"), "AT": ("Euro", "EUR"),
    "AU": ("Australian Dollar", "AUD"), "AW": ("Aruban Florin", "AWG"),
    "AX": ("Euro", "EUR"), "AZ": ("Azerbaijani Manat", "AZN"),
    "BA": ("Convertible Mark", "BAM"), "BB": ("Barbadian Dollar", "BBD"),
    "BD": ("Bangladeshi Taka", "BDT"), "BE": ("Euro", "EUR"),
    "BF": ("West African CFA Franc", "XOF"), "BG": ("Bulgarian Lev", "BGN"),
    "BH": ("Bahraini Dinar", "BHD"), "BI": ("Burundian Franc", "BIF"),
    "BJ": ("West African CFA Franc", "XOF"), "BL": ("Euro", "EUR"),
    "BM": ("Bermudian Dollar", "BMD"), "BN": ("Brunei Dollar", "BND"),
    "BO": ("Bolivian Boliviano", "BOB"), "BQ": ("US Dollar", "USD"),
    "BR": ("Brazilian Real", "BRL"), "BS": ("Bahamian Dollar", "BSD"),
    "BT": ("Bhutanese Ngultrum", "BTN"), "BV": ("Norwegian Krone", "NOK"),
    "BW": ("Botswana Pula", "BWP"), "BY": ("Belarusian Ruble", "BYN"),
    "BZ": ("Belize Dollar", "BZD"), "CA": ("Canadian Dollar", "CAD"),
    "CC": ("Australian Dollar", "AUD"), "CD": ("Congolese Franc", "CDF"),
    "CF": ("Central African CFA Franc", "XAF"), "CG": ("Central African CFA Franc", "XAF"),
    "CH": ("Swiss Franc", "CHF"), "CI": ("West African CFA Franc", "XOF"),
    "CK": ("New Zealand Dollar", "NZD"), "CL": ("Chilean Peso", "CLP"),
    "CM": ("Central African CFA Franc", "XAF"), "CN": ("Chinese Yuan", "CNY"),
    "CO": ("Colombian Peso", "COP"), "CR": ("Costa Rican Colón", "CRC"),
    "CU": ("Cuban Peso", "CUP"), "CV": ("Cape Verdean Escudo", "CVE"),
    "CW": ("Netherlands Antillean Guilder", "ANG"), "CX": ("Australian Dollar", "AUD"),
    "CY": ("Euro", "EUR"), "CZ": ("Czech Koruna", "CZK"),
    "DE": ("Euro", "EUR"), "DJ": ("Djiboutian Franc", "DJF"),
    "DK": ("Danish Krone", "DKK"), "DM": ("East Caribbean Dollar", "XCD"),
    "DO": ("Dominican Peso", "DOP"), "DZ": ("Algerian Dinar", "DZD"),
    "EC": ("US Dollar", "USD"), "EE": ("Euro", "EUR"),
    "EG": ("Egyptian Pound", "EGP"), "EH": ("Moroccan Dirham", "MAD"),
    "ER": ("Eritrean Nakfa", "ERN"), "ES": ("Euro", "EUR"),
    "ET": ("Ethiopian Birr", "ETB"), "FI": ("Euro", "EUR"),
    "FJ": ("Fijian Dollar", "FJD"), "FK": ("Falkland Islands Pound", "FKP"),
    "FM": ("US Dollar", "USD"), "FO": ("Danish Krone", "DKK"),
    "FR": ("Euro", "EUR"), "GA": ("Central African CFA Franc", "XAF"),
    "GB": ("British Pound Sterling", "GBP"), "GD": ("East Caribbean Dollar", "XCD"),
    "GE": ("Georgian Lari", "GEL"), "GF": ("Euro", "EUR"),
    "GG": ("British Pound Sterling", "GBP"), "GH": ("Ghanaian Cedi", "GHS"),
    "GI": ("Gibraltar Pound", "GIP"), "GL": ("Danish Krone", "DKK"),
    "GM": ("Gambian Dalasi", "GMD"), "GN": ("Guinean Franc", "GNF"),
    "GP": ("Euro", "EUR"), "GQ": ("Central African CFA Franc", "XAF"),
    "GR": ("Euro", "EUR"), "GS": ("British Pound Sterling", "GBP"),
    "GT": ("Guatemalan Quetzal", "GTQ"), "GU": ("US Dollar", "USD"),
    "GW": ("West African CFA Franc", "XOF"), "GY": ("Guyanese Dollar", "GYD"),
    "HK": ("Hong Kong Dollar", "HKD"), "HM": ("Australian Dollar", "AUD"),
    "HN": ("Honduran Lempira", "HNL"), "HR": ("Euro", "EUR"),
    "HT": ("Haitian Gourde", "HTG"), "HU": ("Hungarian Forint", "HUF"),
    "ID": ("Indonesian Rupiah", "IDR"), "IE": ("Euro", "EUR"),
    "IL": ("Israeli New Shekel", "ILS"), "IM": ("British Pound Sterling", "GBP"),
    "IN": ("Indian Rupee", "INR"), "IO": ("US Dollar", "USD"),
    "IQ": ("Iraqi Dinar", "IQD"), "IR": ("Iranian Rial", "IRR"),
    "IS": ("Icelandic Króna", "ISK"), "IT": ("Euro", "EUR"),
    "JE": ("British Pound Sterling", "GBP"), "JM": ("Jamaican Dollar", "JMD"),
    "JO": ("Jordanian Dinar", "JOD"), "JP": ("Japanese Yen", "JPY"),
    "KE": ("Kenyan Shilling", "KES"), "KG": ("Kyrgyzstani Som", "KGS"),
    "KH": ("Cambodian Riel", "KHR"), "KI": ("Australian Dollar", "AUD"),
    "KM": ("Comorian Franc", "KMF"), "KN": ("East Caribbean Dollar", "XCD"),
    "KP": ("North Korean Won", "KPW"), "KR": ("South Korean Won", "KRW"),
    "KW": ("Kuwaiti Dinar", "KWD"), "KY": ("Cayman Islands Dollar", "KYD"),
    "KZ": ("Kazakhstani Tenge", "KZT"), "LA": ("Lao Kip", "LAK"),
    "LB": ("Lebanese Pound", "LBP"), "LC": ("East Caribbean Dollar", "XCD"),
    "LI": ("Swiss Franc", "CHF"), "LK": ("Sri Lankan Rupee", "LKR"),
    "LR": ("Liberian Dollar", "LRD"), "LS": ("Lesotho Loti", "LSL"),
    "LT": ("Euro", "EUR"), "LU": ("Euro", "EUR"),
    "LV": ("Euro", "EUR"), "LY": ("Libyan Dinar", "LYD"),
    "MA": ("Moroccan Dirham", "MAD"), "MC": ("Euro", "EUR"),
    "MD": ("Moldovan Leu", "MDL"), "ME": ("Euro", "EUR"),
    "MF": ("Euro", "EUR"), "MG": ("Malagasy Ariary", "MGA"),
    "MH": ("US Dollar", "USD"), "MK": ("Macedonian Denar", "MKD"),
    "ML": ("West African CFA Franc", "XOF"), "MM": ("Myanmar Kyat", "MMK"),
    "MN": ("Mongolian Tögrög", "MNT"), "MO": ("Macanese Pataca", "MOP"),
    "MP": ("US Dollar", "USD"), "MQ": ("Euro", "EUR"),
    "MR": ("Mauritanian Ouguiya", "MRU"), "MS": ("East Caribbean Dollar", "XCD"),
    "MT": ("Euro", "EUR"), "MU": ("Mauritian Rupee", "MUR"),
    "MV": ("Maldivian Rufiyaa", "MVR"), "MW": ("Malawian Kwacha", "MWK"),
    "MX": ("Mexican Peso", "MXN"), "MY": ("Malaysian Ringgit", "MYR"),
    "MZ": ("Mozambican Metical", "MZN"), "NA": ("Namibian Dollar", "NAD"),
    "NC": ("CFP Franc", "XPF"), "NE": ("West African CFA Franc", "XOF"),
    "NF": ("Australian Dollar", "AUD"), "NG": ("Nigerian Naira", "NGN"),
    "NI": ("Nicaraguan Córdoba", "NIO"), "NL": ("Euro", "EUR"),
    "NO": ("Norwegian Krone", "NOK"), "NP": ("Nepalese Rupee", "NPR"),
    "NR": ("Australian Dollar", "AUD"), "NU": ("New Zealand Dollar", "NZD"),
    "NZ": ("New Zealand Dollar", "NZD"), "OM": ("Omani Rial", "OMR"),
    "PA": ("Panamanian Balboa", "PAB"), "PE": ("Peruvian Sol", "PEN"),
    "PF": ("CFP Franc", "XPF"), "PG": ("Papua New Guinean Kina", "PGK"),
    "PH": ("Philippine Peso", "PHP"), "PK": ("Pakistani Rupee", "PKR"),
    "PL": ("Polish Złoty", "PLN"), "PM": ("Euro", "EUR"),
    "PN": ("New Zealand Dollar", "NZD"), "PR": ("US Dollar", "USD"),
    "PS": ("Israeli New Shekel", "ILS"), "PT": ("Euro", "EUR"),
    "PW": ("US Dollar", "USD"), "PY": ("Paraguayan Guaraní", "PYG"),
    "QA": ("Qatari Riyal", "QAR"), "RE": ("Euro", "EUR"),
    "RO": ("Romanian Leu", "RON"), "RS": ("Serbian Dinar", "RSD"),
    "RU": ("Russian Ruble", "RUB"), "RW": ("Rwandan Franc", "RWF"),
    "SA": ("Saudi Riyal", "SAR"), "SB": ("Solomon Islands Dollar", "SBD"),
    "SC": ("Seychellois Rupee", "SCR"), "SD": ("Sudanese Pound", "SDG"),
    "SE": ("Swedish Krona", "SEK"), "SG": ("Singapore Dollar", "SGD"),
    "SH": ("Saint Helena Pound", "SHP"), "SI": ("Euro", "EUR"),
    "SJ": ("Norwegian Krone", "NOK"), "SK": ("Euro", "EUR"),
    "SL": ("Sierra Leonean Leone", "SLE"), "SM": ("Euro", "EUR"),
    "SN": ("West African CFA Franc", "XOF"), "SO": ("Somali Shilling", "SOS"),
    "SR": ("Surinamese Dollar", "SRD"), "SS": ("South Sudanese Pound", "SSP"),
    "ST": ("São Tomé Dobra", "STN"), "SV": ("US Dollar", "USD"),
    "SX": ("Netherlands Antillean Guilder", "ANG"), "SY": ("Syrian Pound", "SYP"),
    "SZ": ("Swazi Lilangeni", "SZL"), "TC": ("US Dollar", "USD"),
    "TD": ("Central African CFA Franc", "XAF"), "TF": ("Euro", "EUR"),
    "TG": ("West African CFA Franc", "XOF"), "TH": ("Thai Baht", "THB"),
    "TJ": ("Tajikistani Somoni", "TJS"), "TK": ("New Zealand Dollar", "NZD"),
    "TL": ("US Dollar", "USD"), "TM": ("Turkmenistani Manat", "TMT"),
    "TN": ("Tunisian Dinar", "TND"), "TO": ("Tongan Paʻanga", "TOP"),
    "TR": ("Turkish Lira", "TRY"), "TT": ("Trinidad and Tobago Dollar", "TTD"),
    "TV": ("Australian Dollar", "AUD"), "TW": ("New Taiwan Dollar", "TWD"),
    "TZ": ("Tanzanian Shilling", "TZS"), "UA": ("Ukrainian Hryvnia", "UAH"),
    "UG": ("Ugandan Shilling", "UGX"), "UM": ("US Dollar", "USD"),
    "US": ("US Dollar", "USD"), "UY": ("Uruguayan Peso", "UYU"),
    "UZ": ("Uzbekistani Som", "UZS"), "VA": ("Euro", "EUR"),
    "VC": ("East Caribbean Dollar", "XCD"), "VE": ("Venezuelan Bolívar", "VES"),
    "VG": ("US Dollar", "USD"), "VI": ("US Dollar", "USD"),
    "VN": ("Vietnamese Đồng", "VND"), "VU": ("Vanuatu Vatu", "VUV"),
    "WF": ("CFP Franc", "XPF"), "WS": ("Samoan Tālā", "WST"),
    "XK": ("Euro", "EUR"), "YE": ("Yemeni Rial", "YER"),
    "YT": ("Euro", "EUR"), "ZA": ("South African Rand", "ZAR"),
    "ZM": ("Zambian Kwacha", "ZMW"), "ZW": ("US Dollar / RTGS", "ZWL"),
}

# ─────────────────────────────────────────────────────────────────────────────
#  Aliases pycountry → noms communs utilisés dans les listes sources
# ─────────────────────────────────────────────────────────────────────────────
PYCOUNTRY_ALIAS: dict[str, str] = {
    "Bolivia, Plurinational State of":        "Bolivia",
    "Congo, The Democratic Republic of the":  "Democratic Republic of the Congo",
    "Iran, Islamic Republic of":              "Iran",
    "Korea, Democratic People's Republic of": "North Korea",
    "Korea, Republic of":                     "South Korea",
    "Lao People's Democratic Republic":       "Lao PDR",
    "Moldova, Republic of":                   "Moldova",
    "Palestine, State of":                    "Palestine",
    "Russian Federation":                     "Russia",
    "Syrian Arab Republic":                   "Syria",
    "Tanzania, United Republic of":           "Tanzania",
    "Venezuela, Bolivarian Republic of":      "Venezuela",
    "Viet Nam":                               "Vietnam",
    "Virgin Islands, British":                "British Virgin Islands",
    "Virgin Islands, U.S.":                   "US Virgin Islands",
    "Timor-Leste":                            "Timor-Leste",
    "Cabo Verde":                             "Cape Verde",
    "Eswatini":                               "Eswatini",
    "North Macedonia":                        "North Macedonia",
    "Türkiye":                                "Turkey",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Validateurs  pays & devise
# ─────────────────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Erreur de validation détectée sur un pays ou une devise."""


def validate_country_code(alpha2: str) -> dict:
    """
    Valide un code pays ISO-3166-1 alpha-2.

    Returns:
        {
          "valid": bool,
          "alpha_2": str, "alpha_3": str, "numeric": str,
          "name": str, "official_name": str,
          "errors": list[str]
        }
    """
    errors: list[str] = []
    result = {
        "valid": False, "alpha_2": alpha2,
        "alpha_3": "", "numeric": "",
        "name": "", "official_name": "",
        "errors": errors,
    }

    if not alpha2 or len(alpha2) != 2 or not alpha2.isalpha():
        errors.append(f"Code pays invalide (doit être 2 lettres) : '{alpha2}'")
        return result

    pc = pycountry.countries.get(alpha_2=alpha2.upper())
    if pc is None:
        errors.append(f"Code pays ISO-3166-1 introuvable : '{alpha2}'")
        return result

    result.update({
        "valid":         True,
        "alpha_2":       pc.alpha_2,
        "alpha_3":       getattr(pc, "alpha_3", ""),
        "numeric":       getattr(pc, "numeric", ""),
        "name":          pc.name,
        "official_name": getattr(pc, "official_name", pc.name),
    })
    return result


def validate_currency_code(iso_code: str, country_alpha2: str = "") -> dict:
    """
    Valide un code devise ISO-4217 et vérifie la cohérence avec le pays.

    Returns:
        {
          "valid": bool,
          "code": str, "name": str,
          "matches_country": bool | None,
          "errors": list[str], "warnings": list[str]
        }
    """
    errors: list[str] = []
    warnings: list[str] = []
    result = {
        "valid": False, "code": iso_code,
        "name": "", "matches_country": None,
        "errors": errors, "warnings": warnings,
    }

    if not iso_code:
        # Devise vide = acceptable pour territoires sans monnaie propre
        result["valid"] = True
        result["name"] = "No currency"
        return result

    if len(iso_code) != 3 or not iso_code.isalpha():
        errors.append(f"Code devise invalide (doit être 3 lettres) : '{iso_code}'")
        return result

    cur = pycountry.currencies.get(alpha_3=iso_code.upper())
    if cur is None:
        errors.append(f"Code devise ISO-4217 introuvable : '{iso_code}'")
        return result

    result.update({"valid": True, "code": cur.alpha_3, "name": cur.name})

    # Vérification de cohérence pays ↔ devise
    if country_alpha2:
        expected = CURRENCY_MAP.get(country_alpha2.upper(), ("", ""))[1]
        if expected and expected != iso_code.upper():
            # Certains pays utilisent plusieurs devises (ex: PS → ILS/JOD)
            warnings.append(
                f"Devise '{iso_code}' différente de la devise attendue "
                f"'{expected}' pour le pays '{country_alpha2}'"
            )
            result["matches_country"] = False
        else:
            result["matches_country"] = True

    return result


def validate_country_currency_pair(alpha2: str, currency_code: str) -> dict:
    """
    Valide ensemble un code pays ET sa devise.
    Retourne un rapport consolidé.
    """
    country_v  = validate_country_code(alpha2)
    currency_v = validate_currency_code(currency_code, alpha2)

    all_errors   = country_v["errors"] + currency_v["errors"]
    all_warnings = currency_v.get("warnings", [])

    return {
        "alpha_2":         alpha2,
        "currency_code":   currency_code,
        "country_valid":   country_v["valid"],
        "currency_valid":  currency_v["valid"],
        "pair_consistent": country_v["valid"] and currency_v["valid"]
                           and currency_v.get("matches_country") is not False,
        "country_info":    country_v,
        "currency_info":   currency_v,
        "errors":          all_errors,
        "warnings":        all_warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass & classification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountryRecord:
    country_code:         str
    country_name:         str
    official_name:        str
    alpha_3:              str
    numeric_code:         str
    risk_level:           str          # CRITICAL / HIGH / MEDIUM / LOW
    risk_score:           int          # 0-100
    currency_name:        str
    currency_code:        str
    # Validation flags
    country_valid:        bool = True
    currency_valid:       bool = True
    currency_consistent:  bool = True
    validation_warnings:  str = ""
    # Source flags
    fatf_blacklist:       bool = False
    fatf_greylist:        bool = False
    ofac_comprehensive:   bool = False
    ofac_partial:         bool = False
    eu_high_risk_aml:     bool = False
    eu_restrictive:       bool = False
    un_sanctions:         bool = False
    basel_critical:       bool = False
    basel_high:           bool = False
    ti_high_corruption:   bool = False
    # Summary
    sources:              str = ""
    last_updated:         str = ""
    data_month:           str = ""


def normalize_name(name: str) -> str:
    return PYCOUNTRY_ALIAS.get(name, name)


def classify(common_name: str) -> tuple[dict, int, str, str]:
    flags = {
        "fatf_blacklist":     common_name in FATF_HIGH_RISK_CALL_ACTION,
        "fatf_greylist":      common_name in FATF_GREY_LIST,
        "ofac_comprehensive": common_name in OFAC_COMPREHENSIVE_SANCTIONS,
        "ofac_partial":       common_name in OFAC_PARTIAL_SANCTIONS,
        "eu_high_risk_aml":   common_name in EU_HIGH_RISK_AML,
        "eu_restrictive":     common_name in EU_RESTRICTIVE_MEASURES,
        "un_sanctions":       common_name in UN_SANCTIONS,
        "basel_critical":     common_name in BASEL_CRITICAL,
        "basel_high":         common_name in BASEL_HIGH,
        "ti_high_corruption": common_name in TI_HIGH_CORRUPTION,
    }
    score = min(100, (
        flags["fatf_blacklist"]     * 35 +
        flags["ofac_comprehensive"] * 30 +
        flags["un_sanctions"]       * 15 +
        flags["eu_restrictive"]     * 10 +
        flags["fatf_greylist"]      * 12 +
        flags["ofac_partial"]       * 8  +
        flags["eu_high_risk_aml"]   * 8  +
        flags["basel_critical"]     * 10 +
        flags["basel_high"]         * 5  +
        flags["ti_high_corruption"] * 5
    ))
    level = (
        "CRITICAL" if score >= 55 else
        "HIGH"     if score >= 25 else
        "MEDIUM"   if score >= 10 else
        "LOW"
    )
    label_map = {
        "fatf_blacklist":     "FATF Black List",
        "fatf_greylist":      "FATF Grey List",
        "ofac_comprehensive": "OFAC Comprehensive Sanctions",
        "ofac_partial":       "OFAC Partial Sanctions",
        "eu_high_risk_aml":   "EU High-Risk AML",
        "eu_restrictive":     "EU Restrictive Measures",
        "un_sanctions":       "UN Security Council Sanctions",
        "basel_critical":     "Basel AML Index – Critical",
        "basel_high":         "Basel AML Index – High",
        "ti_high_corruption": "TI Corruption Perception Index – High",
    }
    sources = ", ".join(v for k, v in label_map.items() if flags[k]) or "No specific sanction"
    return flags, score, level, sources


# ─────────────────────────────────────────────────────────────────────────────
#  Construction du dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset() -> tuple[list[CountryRecord], dict]:
    """
    Construit la liste complète des pays avec classification + validation.

    Returns:
        (records, validation_report)
    """
    records: list[CountryRecord] = []
    now   = datetime.now().isoformat(timespec="seconds")
    month = datetime.now().strftime("%Y-%m")

    validation_report = {
        "total": 0,
        "country_errors": [],
        "currency_errors": [],
        "currency_warnings": [],
        "missing_currency": [],
    }

    for pc in pycountry.countries:
        common   = normalize_name(pc.name)
        alpha2   = pc.alpha_2
        alpha3   = getattr(pc, "alpha_3", "")
        numeric  = getattr(pc, "numeric", "")
        official = getattr(pc, "official_name", pc.name)

        flags, score, level, sources = classify(common)

        # Devise
        cur_name, cur_code = CURRENCY_MAP.get(alpha2, ("", ""))

        # ── Validation pays ────────────────────────────────────────────────
        country_v  = validate_country_code(alpha2)
        currency_v = validate_currency_code(cur_code, alpha2)

        country_valid     = country_v["valid"]
        currency_valid    = currency_v["valid"]
        currency_consist  = currency_v.get("matches_country") is not False
        val_warnings      = "; ".join(currency_v.get("warnings", []))

        # Rapport
        validation_report["total"] += 1
        if not country_valid:
            validation_report["country_errors"].append(
                {"alpha2": alpha2, "errors": country_v["errors"]}
            )
        if not currency_valid:
            validation_report["currency_errors"].append(
                {"alpha2": alpha2, "currency": cur_code, "errors": currency_v["errors"]}
            )
        if currency_v.get("warnings"):
            validation_report["currency_warnings"].append(
                {"alpha2": alpha2, "warnings": currency_v["warnings"]}
            )
        if not cur_code:
            validation_report["missing_currency"].append(alpha2)

        rec = CountryRecord(
            country_code        = alpha2,
            country_name        = common,
            official_name       = official,
            alpha_3             = alpha3,
            numeric_code        = numeric,
            risk_level          = level,
            risk_score          = score,
            currency_name       = cur_name,
            currency_code       = cur_code,
            country_valid       = country_valid,
            currency_valid      = currency_valid,
            currency_consistent = currency_consist,
            validation_warnings = val_warnings,
            sources             = sources,
            last_updated        = now,
            data_month          = month,
            **flags,
        )
        records.append(rec)

    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    records.sort(key=lambda r: (order[r.risk_level], -r.risk_score, r.country_name))
    return records, validation_report


# ─────────────────────────────────────────────────────────────────────────────
#  Sauvegarde dans data/scraping/
# ─────────────────────────────────────────────────────────────────────────────

def _paths_for(month: str) -> dict[str, Path]:
    """Retourne les chemins archive et latest pour un mois donné."""
    stem = f"country_risk_{month}"
    return {
        "csv_archive":     ARCHIVE_DIR / f"{stem}.csv",
        "json_archive":    ARCHIVE_DIR / f"{stem}.json",
        "report_archive":  ARCHIVE_DIR / f"validation_report_{month}.json",
        "csv_latest":      LATEST_DIR  / "country_risk_latest.csv",
        "json_latest":     LATEST_DIR  / "country_risk_latest.json",
        "report_latest":   LATEST_DIR  / "validation_report_latest.json",
        "summary_latest":  LATEST_DIR  / "summary_latest.txt",
    }


FIELDNAMES = [
    "country_code", "country_name", "official_name", "alpha_3", "numeric_code",
    "risk_level", "risk_score",
    "currency_name", "currency_code",
    "country_valid", "currency_valid", "currency_consistent", "validation_warnings",
    "fatf_blacklist", "fatf_greylist",
    "ofac_comprehensive", "ofac_partial",
    "eu_high_risk_aml", "eu_restrictive",
    "un_sanctions",
    "basel_critical", "basel_high",
    "ti_high_corruption",
    "sources", "last_updated", "data_month",
]


def save_csv(records: list[CountryRecord], path: Path) -> None:
    bool_flags = [
        "fatf_blacklist", "fatf_greylist", "ofac_comprehensive", "ofac_partial",
        "eu_high_risk_aml", "eu_restrictive", "un_sanctions",
        "basel_critical", "basel_high", "ti_high_corruption",
        "country_valid", "currency_valid", "currency_consistent",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in records:
            row = asdict(r)
            for k in bool_flags:
                row[k] = "Yes" if row[k] else "No"
            writer.writerow(row)
    log.info("CSV  → %s", path)


def save_json(records: list[CountryRecord], path: Path) -> None:
    data = [asdict(r) for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("JSON → %s", path)


def save_validation_report(report: dict, path: Path) -> None:
    report_out = {
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "data_month":     datetime.now().strftime("%Y-%m"),
        **report,
        "summary": {
            "country_errors_count":   len(report["country_errors"]),
            "currency_errors_count":  len(report["currency_errors"]),
            "currency_warnings_count": len(report["currency_warnings"]),
            "missing_currency_count":  len(report["missing_currency"]),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_out, f, ensure_ascii=False, indent=2)
    log.info("Validation report → %s", path)


def save_summary(records: list[CountryRecord], report: dict, path: Path) -> None:
    counts = {lvl: [r for r in records if r.risk_level == lvl]
              for lvl in ("CRITICAL", "HIGH", "MEDIUM", "LOW")}

    lines = [
        "=" * 66,
        f"  COUNTRY RISK CLASSIFICATION  –  {date.today()}  (mensuel)",
        "=" * 66,
        f"  Répertoire  : {BASE_DIR}",
        f"  Total pays  : {len(records)}",
        "",
        f"  🔴 CRITICAL : {len(counts['CRITICAL'])} pays",
        f"  🟠 HIGH     : {len(counts['HIGH'])} pays",
        f"  🟡 MEDIUM   : {len(counts['MEDIUM'])} pays",
        f"  🟢 LOW      : {len(counts['LOW'])} pays",
        "",
        f"  ✅ Validations OK        : {report['total'] - len(report['country_errors'])} pays",
        f"  ❌ Erreurs pays          : {len(report['country_errors'])}",
        f"  ❌ Erreurs devise        : {len(report['currency_errors'])}",
        f"  ⚠️  Avertissements devise : {len(report['currency_warnings'])}",
        f"  ➖ Sans devise           : {len(report['missing_currency'])}",
        "",
        "  Sources :",
        "    1. FATF Black & Grey Lists    (2025)",
        "    2. OFAC Comprehensive/Partial (2025)",
        "    3. EU AML High-Risk + Restrictive Measures (2025)",
        "    4. UN Security Council Sanctions (2025)",
        "    5. Basel AML Index (2024)",
        "    6. Transparency International CPI (2024)",
        "",
        "─" * 66,
        "  Pays CRITICAL :",
    ]
    for r in counts["CRITICAL"]:
        cur  = f"{r.currency_code}" if r.currency_code else "N/A"
        warn = " ⚠" if r.validation_warnings else ""
        lines.append(
            f"    [{r.country_code}] {r.country_name:35s} score={r.risk_score:3d}  {cur}{warn}"
        )
    lines += ["", "─" * 66, "  Pays HIGH :"]
    for r in counts["HIGH"]:
        cur  = f"{r.currency_code}" if r.currency_code else "N/A"
        warn = " ⚠" if r.validation_warnings else ""
        lines.append(
            f"    [{r.country_code}] {r.country_name:35s} score={r.risk_score:3d}  {cur}{warn}"
        )
    lines.append("=" * 66)

    txt = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


def save_all(records: list[CountryRecord], report: dict, month: str) -> dict[str, Path]:
    """Sauvegarde en archive + latest. Retourne le dictionnaire des chemins."""
    paths = _paths_for(month)

    # Archive (versionnée par mois)
    save_csv(records, paths["csv_archive"])
    save_json(records, paths["json_archive"])
    save_validation_report(report, paths["report_archive"])

    # Latest (toujours la version la plus récente)
    save_csv(records, paths["csv_latest"])
    save_json(records, paths["json_latest"])
    save_validation_report(report, paths["report_latest"])
    save_summary(records, report, paths["summary_latest"])

    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  Point d'entrée principal  (appelé aussi par le scheduler)
# ─────────────────────────────────────────────────────────────────────────────

def run_scraper(force: bool = False) -> dict:
    """
    Lance la construction du dataset.

    Args:
        force: si True, ignore la vérification de mise à jour mensuelle.

    Returns:
        dict avec les statistiques de l'exécution.
    """
    log.info("=== Démarrage du scraper de risque pays (mensuel) ===")

    # Vérification : a-t-on déjà généré ce mois-ci ?
    month = datetime.now().strftime("%Y-%m")
    if not force:
        existing = ARCHIVE_DIR / f"country_risk_{month}.json"
        if existing.exists():
            log.info("✅ Données du mois %s déjà présentes → pas de recalcul (--force pour forcer)", month)
            return {"skipped": True, "month": month, "path": str(existing)}

    records, report = build_dataset()
    paths = save_all(records, report, month)

    stats = {
        "skipped":         False,
        "month":           month,
        "total_countries": len(records),
        "critical":        len([r for r in records if r.risk_level == "CRITICAL"]),
        "high":            len([r for r in records if r.risk_level == "HIGH"]),
        "medium":          len([r for r in records if r.risk_level == "MEDIUM"]),
        "low":             len([r for r in records if r.risk_level == "LOW"]),
        "validation": {
            "country_errors":   len(report["country_errors"]),
            "currency_errors":  len(report["currency_errors"]),
            "warnings":         len(report["currency_warnings"]),
            "missing_currency": len(report["missing_currency"]),
        },
        "output_dir":  str(BASE_DIR),
        "archive_dir": str(ARCHIVE_DIR),
        "latest_dir":  str(LATEST_DIR),
        "files": {k: str(v) for k, v in paths.items()},
    }

    log.info(
        "✅ Scraper terminé — %d pays | CRITICAL:%d HIGH:%d MEDIUM:%d LOW:%d",
        stats["total_countries"], stats["critical"],
        stats["high"], stats["medium"], stats["low"],
    )
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scraper de risque pays – mise à jour mensuelle")
    parser.add_argument("--force", action="store_true", help="Forcer le recalcul même si les données du mois existent")
    parser.add_argument("--validate", metavar="ALPHA2", help="Valider un code pays (ex: TN, FR)")
    parser.add_argument("--validate-pair", nargs=2, metavar=("ALPHA2", "CURRENCY"),
                        help="Valider un couple pays/devise (ex: TN TND)")
    args = parser.parse_args()

    if args.validate:
        result = validate_country_code(args.validate.upper())
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.validate_pair:
        result = validate_country_currency_pair(args.validate_pair[0].upper(),
                                                args.validate_pair[1].upper())
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        run_scraper(force=args.force)