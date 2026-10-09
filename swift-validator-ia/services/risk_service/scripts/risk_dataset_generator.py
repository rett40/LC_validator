"""
MT700 Risk Dataset Generator v4.1
====================================
Génère des Lettres de Crédit synthétiques pour l'entraînement du modèle risk ML.

Changements v4.1 (vs v4.0) :
  CRITICAL FIX-1 : 26 features canoniques (vs 15) — alignées sur risk_feature_schema.py v4.0
  CRITICAL FIX-2 : expert_label() avec ZONE SÛRE — UCP>=90%+0_erreur → toujours LOW
  CRITICAL FIX-3 : temporal features ne distinguent plus LOW vs MEDIUM (source principale
                   des discordances ML/Métier — validity_days=43j ≠ MEDIUM)
  CRITICAL FIX-4 : _profile_medium() UCP cappé à 0.80 (ne peut plus atteindre 1.0)
  NEW-1 : Profils dédiés conformes : _profile_perfect_conforme() + _profile_clean_lc()
  NEW-2 : 11 nouvelles features : ucp_errors_count, ucp_warnings_count,
           missing_critical_score, coherence_dates, contradiction_flag,
           country_risk_applicant, country_risk_beneficiary, amount_tier,
           red_clause_flag, bank_tier, tenor_risk
  NEW-3 : expert_label() utilise les nouvelles features v4.0
  NEW-4 : Distribution rééquilibrée — 35% LOW (augmenté), cas conformes représentés

Règle d'or de labeling (non-circulaire) :
  LOW   = UCP >=85%, pays tier 0-1, documents complets, aucun flag rouge
  MEDIUM = UCP 40-84% OU pays tier 1-2 OU warnings modérés (pas à cause du temporel seul)
  HIGH   = UCP <40% OU pays sanctionné/tier 3 OU erreurs multiples OU flags réglementaires
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Import schéma partagé ──────────────────────────────────────────────────
_RISK_SVC_ROOT = Path(__file__).resolve().parent.parent
if str(_RISK_SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RISK_SVC_ROOT))

from risk_feature_schema import (
    FEATURE_COLS_CANONICAL,
    INCOTERM_RISK_MAP,
    INCOTERM_RISK_DEFAULT,
    SANCTIONED_ISO2,
    compute_amount_tier,
)

FEATURE_COLS = list(FEATURE_COLS_CANONICAL)  # 26 features

# ─────────────────────────────────────────────────────────────────────────────
# DONNÉES DE RÉFÉRENCE
# ─────────────────────────────────────────────────────────────────────────────

COUNTRIES_BY_RISK = {
    0: ["DE", "FR", "CH", "NL", "US", "JP", "CA", "GB", "AU", "SG", "SE", "NO", "DK", "FI", "AT"],
    1: ["CN", "IN", "BR", "MX", "ZA", "PL", "TR", "AE", "SA", "EG", "MA", "TN", "KW", "QA", "OM"],
    2: ["NG", "PK", "BD", "VN", "ID", "UA", "KE", "GH", "ET", "DZ", "IQ", "LB", "JO", "AM", "GE"],
    3: ["AF", "SY", "YE", "LY", "MM", "SD", "ZW", "VE", "KP", "IR", "SO", "CU", "BY", "ER", "CF"],
}

_COUNTRY_SCORE_DIST: dict[int, tuple] = {
    0: (7.0,  6.0),   # CLEAN : 0-20
    1: (28.0, 10.0),  # LOW/MEDIUM : 10-45
    2: (52.0, 12.0),  # HIGH : 30-70
    3: (74.0,  9.0),  # VERY HIGH : 55-90
    4: (94.0,  4.0),  # CRITICAL / sanctionné : 88-100
}

CURRENCY_RISK_LEVEL: dict[str, int] = {
    "USD": 0, "EUR": 0, "GBP": 0, "JPY": 0, "CHF": 0, "CAD": 0,
    "AUD": 0, "NZD": 0, "SEK": 0, "NOK": 0, "DKK": 0, "SGD": 0,
    "HKD": 0, "CZK": 0, "ILS": 0,
    "CNY": 1, "AED": 1, "SAR": 1, "QAR": 1, "KWD": 1, "BHD": 1,
    "OMR": 1, "INR": 1, "BRL": 1, "MXN": 1, "ZAR": 1, "TRY": 1,
    "PLN": 1, "HUF": 1, "RON": 1, "CLP": 1, "COP": 1, "TND": 1,
    "JOD": 1, "MAD": 1,
    "NGN": 2, "PKR": 2, "BDT": 2, "VND": 2, "IDR": 2, "UAH": 2,
    "EGP": 2, "DZD": 2, "GHS": 2, "KES": 2, "ETB": 2, "GEL": 2,
    "AMD": 2, "IQD": 2, "LBP": 2, "SOS": 2, "AFN": 2, "YER": 2,
    "IRR": 3, "KPW": 3, "SYP": 3, "SDG": 3, "MMK": 3, "VES": 3,
    "CUP": 3, "ERN": 3, "ZWL": 3, "LYD": 3,
}

COUNTRY_DEFAULT_CURRENCY: dict[str, str] = {
    "DE": "EUR", "FR": "EUR", "CH": "CHF", "NL": "EUR", "US": "USD",
    "JP": "JPY", "CA": "CAD", "GB": "GBP", "AU": "AUD", "SG": "SGD",
    "SE": "SEK", "NO": "NOK", "DK": "DKK", "FI": "EUR", "AT": "EUR",
    "CN": "CNY", "IN": "INR", "BR": "BRL", "MX": "MXN", "ZA": "ZAR",
    "PL": "PLN", "TR": "TRY", "AE": "AED", "SA": "SAR", "EG": "EGP",
    "MA": "MAD", "TN": "TND", "KW": "KWD", "QA": "QAR", "OM": "OMR",
    "NG": "NGN", "PK": "PKR", "BD": "BDT", "VN": "VND", "ID": "IDR",
    "UA": "UAH", "KE": "KES", "GH": "GHS", "ET": "ETB", "DZ": "DZD",
    "IQ": "IQD", "LB": "LBP", "JO": "JOD", "AM": "AMD", "GE": "GEL",
    "AF": "AFN", "SY": "SYP", "YE": "YER", "LY": "LYD", "MM": "MMK",
    "SD": "SDG", "ZW": "ZWL", "VE": "VES", "KP": "KPW", "IR": "IRR",
    "SO": "SOS", "CU": "CUP", "BY": "USD", "ER": "ERN", "CF": "USD",
}

INCOTERMS_ALL = list(INCOTERM_RISK_MAP.keys())

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def _tier_from_iso(iso: str) -> int:
    if not iso or not str(iso).strip():
        return 1
    iso2 = str(iso).strip().upper()[:2]
    for tier, countries in COUNTRIES_BY_RISK.items():
        if iso2 in countries:
            return tier
    return 1


def _country_risk_score(iso2: str, rng: np.random.Generator) -> float:
    tier = _tier_from_iso(iso2)
    if iso2 in SANCTIONED_ISO2:
        tier = 4
    mu, sigma = _COUNTRY_SCORE_DIST[tier]
    return float(np.clip(rng.normal(mu, sigma), 0.0, 100.0))


def _pick_currency(country_iso2: str, rng: np.random.Generator) -> str:
    local = COUNTRY_DEFAULT_CURRENCY.get(country_iso2, "USD")
    if country_iso2 in SANCTIONED_ISO2:
        return local
    p = rng.random()
    if p < 0.60:
        return "USD"
    if p < 0.80 and country_iso2 in COUNTRIES_BY_RISK[0]:
        return local
    return local


def _currency_risk(currency: str, country_iso2: str) -> int:
    if country_iso2 in SANCTIONED_ISO2:
        return 3
    return CURRENCY_RISK_LEVEL.get(currency.upper(), 1)


def _split_ucp_warnings(n_total: int, is_high_risk: bool, rng: np.random.Generator) -> tuple[int, int]:
    """
    Répartit n_total violations en (erreurs, avertissements).
    Pour les LC HIGH : plus d'erreurs (poids 1.0).
    Pour les LC LOW/MEDIUM : plus d'avertissements (poids 0.4).
    """
    if n_total == 0:
        return 0, 0
    if is_high_risk:
        # HIGH : 50-80% d'erreurs
        error_frac = float(rng.uniform(0.5, 0.8))
    else:
        # LOW/MEDIUM : 0-30% d'erreurs
        error_frac = float(rng.uniform(0.0, 0.3))
    errors   = int(round(n_total * error_frac))
    warnings = n_total - errors
    return max(0, errors), max(0, warnings)


def _missing_critical_from_completeness(completeness: float, rng: np.random.Generator) -> float:
    """
    Dérive missing_critical_score depuis completeness_score.
    missing_critical_score >= completeness (les critiques sont un sous-ensemble des champs).
    """
    # Plus la complétude est haute, plus les critiques sont présents
    base = float(np.clip(completeness * 1.1 + rng.normal(0, 0.05), 0, 1))
    return round(float(np.clip(base, 0.0, 1.0)), 4)


def _bank_tier_for_country(country_tier: int, rng: np.random.Generator) -> int:
    """
    Les pays tier 0 ont souvent des banques Tier 1-2.
    Les pays tier 3 ont souvent des banques Tier 4-5.
    """
    if country_tier == 0:
        return int(rng.choice([1, 1, 2, 2, 3], p=[0.30, 0.30, 0.25, 0.10, 0.05]))
    elif country_tier == 1:
        return int(rng.choice([1, 2, 2, 3, 4], p=[0.10, 0.25, 0.30, 0.25, 0.10]))
    elif country_tier == 2:
        return int(rng.choice([2, 3, 3, 4, 5], p=[0.10, 0.25, 0.30, 0.25, 0.10]))
    else:
        return int(rng.choice([3, 4, 4, 5, 5], p=[0.10, 0.20, 0.30, 0.25, 0.15]))


# ─────────────────────────────────────────────────────────────────────────────
# LABELING EXPERT v4.1 — AVEC ZONE SÛRE (non-circulaire)
# ─────────────────────────────────────────────────────────────────────────────

def expert_label(row: dict, rng: np.random.Generator) -> str:
    """
    Label de risque basé sur l'expertise documentaire crédit v4.1.

    ZONE SÛRE (CRITICAL FIX) :
      Si UCP >= 90%, 0 erreur UCP, <= 2 avert, pays non risqué, docs couverts
      → TOUJOURS LOW quelle que soit la valeur des features temporelles.
      Cela corrige la discordance ML/Métier sur les LCs "conformes mais imparfaits".

    IMPORTANT — Temporel marginalisé :
      validity_days et days_to_shipment n'interviennent QUE pour des valeurs
      EXTRÊMES (<5 jours ou <15 jours). Des valeurs normales (43 jours, 180 jours)
      ne contribuent PAS au score de risque. C'est la réalité documentaire UCP 600.
    """
    cr_score    = float(row.get("country_risk_score", 25.0))
    ucp_s       = float(row.get("ucp_compliance_score", 0.5))
    ucp_errors  = float(row.get("ucp_errors_count", 0.0))
    ucp_warns_c = float(row.get("ucp_warnings_count", 0.0))
    n_warn      = float(row.get("n_ucp_warnings", 0.0))
    comp        = float(row.get("completeness_score", 0.5))
    miss_crit   = float(row.get("missing_critical_score", 0.75))
    red_clause  = float(row.get("red_clause_flag", 0.0))
    contradict  = float(row.get("contradiction_flag", 0.0))
    bank_t      = float(row.get("bank_tier", 3.0))
    tenor_r     = float(row.get("tenor_risk", 0.5))

    # ── ZONE SÛRE — retour direct LOW ─────────────────────────────────────────
    # Conditions strictes : document entièrement conforme, sans flag réglementaire
    is_safe = (
        ucp_s       >= 0.90
        and ucp_errors  == 0
        and ucp_warns_c <= 2
        and n_warn      <= 2
        and cr_score    < 50
        and miss_crit   >= 0.75
        and comp        >= 0.70
        and red_clause  == 0
        and contradict  == 0
    )
    if is_safe:
        return "LOW"

    # ── Règle hard : pays sanctionné / CRITICAL ───────────────────────────────
    if cr_score >= 88 or red_clause >= 1:
        return "HIGH"

    score = 0.0

    # 1. Risque pays consolidé — PRINCIPAL DRIVER
    if cr_score >= 70:
        score += 0.45
    elif cr_score >= 45:
        score += 0.28
    elif cr_score >= 22:
        score += 0.12
    elif cr_score >= 10:
        score += 0.04

    # 2. Risque devise
    cur_risk = int(row.get("external_currency_risk", 0))
    score += cur_risk * 0.04  # max 0.16 pour level=4

    # 3. Erreurs UCP (driver fort) — weight 1.0 par erreur
    if ucp_errors >= 5:
        score += 0.40
    elif ucp_errors >= 3:
        score += 0.28
    elif ucp_errors >= 1:
        score += 0.15

    # 4. Avertissements UCP (driver modéré) — weight 0.4 par avertissement
    if ucp_warns_c >= 8:
        score += 0.20
    elif ucp_warns_c >= 5:
        score += 0.12
    elif ucp_warns_c >= 3:
        score += 0.05

    # 5. Qualité documentaire composite (NLP + completeness + semantic + UCP)
    nlp_n     = float(row.get("nlp_score", 50.0)) / 100.0
    sem       = float(row.get("semantic_score", 0.5))
    doc_qual  = nlp_n * 0.25 + comp * 0.30 + sem * 0.20 + ucp_s * 0.25
    if doc_qual < 0.25:
        score += 0.35
    elif doc_qual < 0.42:
        score += 0.18
    elif doc_qual < 0.60:
        score += 0.06

    # 6. Champs critiques manquants
    if miss_crit < 0.50:
        score += 0.20
    elif miss_crit < 0.75:
        score += 0.08

    # 7. Cohérence et contradictions
    if contradict >= 1:
        score += 0.18
    if float(row.get("coherence_dates", 1.0)) == 0.0:
        score += 0.12

    # 8. Incoterm risqué
    inc_r = int(row.get("incoterm_risk", INCOTERM_RISK_DEFAULT))
    if inc_r >= 5:
        score += 0.07
    elif inc_r >= 4:
        score += 0.04

    # 9. Indicateurs structurels
    if row.get("transshipment_allowed", 0) == 1:
        score += 0.04
    if row.get("charges_beneficiary", 0) == 1:
        score += 0.03

    # 10. Tier banque (banque inconnue = risque contrepartie)
    if bank_t >= 5:
        score += 0.08
    elif bank_t >= 4:
        score += 0.04

    # 11. Tenor (paiement différé long = risque de crédit)
    if tenor_r >= 0.85:
        score += 0.10
    elif tenor_r >= 0.65:
        score += 0.05

    # 12. Montant — patterns TBML (Trade-Based Money Laundering)
    amount = float(row.get("lc_amount", 500_000))
    if amount < 3_000 or amount > 20_000_000:
        score += 0.10
    elif amount < 10_000 or amount > 10_000_000:
        score += 0.05

    # 13. Pression temporelle EXTRÊME UNIQUEMENT
    # CORRECTION CRITIQUE : seules les valeurs EXTRÊMES sont des signaux de risque.
    # validity=43j, days_to_shipment=30j = NORMAUX en commerce international → score 0
    days     = max(1, int(row.get("days_to_shipment", 30)))
    validity = max(1, int(row.get("validity_days", 90)))
    pres     = max(1, int(row.get("presentation_days", 21)))

    if days < 5:
        score += 0.30  # critique : impossible à traiter
    elif days < 10:
        score += 0.15  # très serré
    # days >= 10 → contribution ZÉRO (plage normale 10-730 jours)

    if pres < 3:
        score += 0.25  # Art.14(c) non respecté
    elif pres < 5:
        score += 0.12
    # pres >= 5 → contribution ZÉRO

    # Incohérence temporelle structurelle (validité < shipment + présentation)
    if validity < days + pres:
        score += 0.20
    elif validity < days + pres + 7:
        score += 0.08

    if score >= 0.55:
        return "HIGH"
    if score >= 0.22:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEUR PRINCIPAL v4.1
# ─────────────────────────────────────────────────────────────────────────────

class MT700DataGenerator:
    """
    Génère des LCs synthétiques avec 26 features canoniques v4.0.

    Changements v4.1 :
      - Toutes les méthodes de profil génèrent les 26 features
      - MEDIUM ne peut pas avoir UCP >= 0.85 (cap strict → évite les discordances)
      - Temporal features ont des ranges LARGES dans LOW (21-365j) → la temporel
        n'est plus un signal discriminant entre LOW et MEDIUM
      - Nouveaux profils : _profile_perfect_conforme(), _profile_clean_lc()
      - expert_label() utilise les 26 features avec zone sûre
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def _new_features_low(self, n_total_warnings: int, completeness: float, cr_score: float) -> dict:
        """Génère les 11 nouvelles features v4.0 pour un profil LOW."""
        errors, warns = _split_ucp_warnings(n_total_warnings, is_high_risk=False, rng=self._rng)
        country_tier = 0 if cr_score < 15 else (1 if cr_score < 40 else 2)
        return {
            "ucp_errors_count":        float(errors),
            "ucp_warnings_count":      float(warns),
            "missing_critical_score":  _missing_critical_from_completeness(completeness, self._rng),
            "coherence_dates":         float(self._rng.choice([1.0, 1.0, 1.0, 0.5], p=[0.80, 0.10, 0.05, 0.05])),
            "contradiction_flag":      0.0,
            "country_risk_applicant":  float(np.clip(self._rng.normal(cr_score * 0.8, 5), 0, 100)),
            "country_risk_beneficiary":float(np.clip(self._rng.normal(cr_score, 5), 0, 100)),
            "amount_tier":             0.0,  # sera calculé après lc_amount
            "red_clause_flag":         0.0,
            "bank_tier":               float(_bank_tier_for_country(country_tier, self._rng)),
            "tenor_risk":              float(self._rng.choice([0.10, 0.10, 0.35, 0.35, 0.50],
                                                               p=[0.40, 0.25, 0.15, 0.15, 0.05])),
        }

    def _new_features_medium(self, n_total_warnings: int, completeness: float, cr_score: float) -> dict:
        """Génère les 11 nouvelles features v4.0 pour un profil MEDIUM."""
        errors, warns = _split_ucp_warnings(n_total_warnings, is_high_risk=False, rng=self._rng)
        country_tier = 1 if cr_score < 45 else 2
        return {
            "ucp_errors_count":        float(errors),
            "ucp_warnings_count":      float(warns),
            "missing_critical_score":  _missing_critical_from_completeness(completeness, self._rng),
            "coherence_dates":         float(self._rng.choice([1.0, 0.5, 0.5, 0.0], p=[0.55, 0.25, 0.15, 0.05])),
            "contradiction_flag":      float(self._rng.choice([0, 0, 0, 1], p=[0.70, 0.15, 0.05, 0.10])),
            "country_risk_applicant":  float(np.clip(self._rng.normal(cr_score * 0.85, 8), 0, 100)),
            "country_risk_beneficiary":float(np.clip(self._rng.normal(cr_score, 8), 0, 100)),
            "amount_tier":             0.0,
            "red_clause_flag":         0.0,
            "bank_tier":               float(_bank_tier_for_country(country_tier, self._rng)),
            "tenor_risk":              float(self._rng.choice([0.10, 0.35, 0.50, 0.65, 0.75],
                                                               p=[0.15, 0.25, 0.30, 0.20, 0.10])),
        }

    def _new_features_high(self, n_total_warnings: int, completeness: float, cr_score: float) -> dict:
        """Génère les 11 nouvelles features v4.0 pour un profil HIGH."""
        errors, warns = _split_ucp_warnings(n_total_warnings, is_high_risk=True, rng=self._rng)
        country_tier = 2 if cr_score < 70 else 3
        return {
            "ucp_errors_count":        float(errors),
            "ucp_warnings_count":      float(warns),
            "missing_critical_score":  _missing_critical_from_completeness(completeness, self._rng),
            "coherence_dates":         float(self._rng.choice([1.0, 0.5, 0.0, 0.0], p=[0.25, 0.30, 0.25, 0.20])),
            "contradiction_flag":      float(self._rng.choice([0, 0, 1, 1], p=[0.30, 0.20, 0.25, 0.25])),
            "country_risk_applicant":  float(np.clip(self._rng.normal(cr_score * 0.9, 10), 0, 100)),
            "country_risk_beneficiary":float(np.clip(self._rng.normal(cr_score, 10), 0, 100)),
            "amount_tier":             0.0,
            "red_clause_flag":         float(self._rng.choice([0, 0, 1], p=[0.60, 0.25, 0.15])),
            "bank_tier":               float(_bank_tier_for_country(country_tier, self._rng)),
            "tenor_risk":              float(self._rng.choice([0.35, 0.50, 0.65, 0.80, 0.90],
                                                               p=[0.10, 0.20, 0.25, 0.25, 0.20])),
        }

    def _finalize_row(self, row: dict) -> dict:
        """Post-traitement : calcule amount_tier depuis lc_amount."""
        amount = float(row.get("lc_amount", 500_000))
        row["amount_tier"] = float(compute_amount_tier(amount))
        return row

    # ── Profils de base ───────────────────────────────────────────────────────

    def _profile_low(self) -> dict:
        """LC standard faible risque : pays sûr, documents complets, UCP conforme."""
        ben      = self._rng.choice(COUNTRIES_BY_RISK[0] + COUNTRIES_BY_RISK[1])
        currency = _pick_currency(ben, self._rng)
        cr_score = _country_risk_score(ben, self._rng)
        cr_score = float(np.clip(cr_score, 0, 40))  # plafonné LOW
        cur_risk = _currency_risk(currency, ben)
        n_clauses  = int(self._rng.integers(4, 14))
        n_warnings = int(self._rng.choice([0, 0, 0, 1, 1, 2]))
        inc        = str(self._rng.choice(["FOB", "CIF", "CPT", "DDP", "FCA"]))
        amount     = float(self._rng.lognormal(12.5, 0.8))
        comp       = float(self._rng.beta(8, 2))

        row = {
            "lc_amount":             amount,
            "country_risk_score":    cr_score,
            "external_currency_risk": float(min(cur_risk, 1)),
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get(inc, 3)),
            "ucp_compliance_score":  float(self._rng.beta(8, 2)),   # 0.7-1.0
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(80, 8), 50, 100)),
            "semantic_score":        float(self._rng.beta(7, 2)),
            "completeness_score":    comp,
            # CORRECTION : temporel LARGE range — 21 à 365 jours est NORMAL
            "days_to_shipment":      int(np.clip(self._rng.lognormal(3.8, 0.6), 10, 200)),
            "presentation_days":     int(self._rng.integers(10, 45)),
            "validity_days":         int(self._rng.integers(21, 365)),
            "transshipment_allowed": int(self._rng.choice([0, 1], p=[0.82, 0.18])),
            "charges_beneficiary":   int(self._rng.choice([0, 1], p=[0.87, 0.13])),
        }
        row.update(self._new_features_low(n_warnings, comp, cr_score))
        return self._finalize_row(row)

    def _profile_perfect_conforme(self) -> dict:
        """
        LC parfaitement conforme : UCP=1.0, 0 erreur, pays clean, docs complets.
        NOUVEAU v4.1 — représente les cas "documents exemplaires" → TOUJOURS LOW.
        Ces profils enseignent au modèle qu'un LC excellent = LOW quel que soit le temporel.
        """
        ben      = self._rng.choice(COUNTRIES_BY_RISK[0])
        currency = _pick_currency(ben, self._rng)
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 0, 15))
        cur_risk = _currency_risk(currency, ben)
        amount   = float(self._rng.lognormal(12.0, 1.2))
        comp     = float(self._rng.beta(9, 1.5))  # 0.80-1.0

        row = {
            "lc_amount":             amount,
            "country_risk_score":    cr_score,
            "external_currency_risk": 0.0,
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get(
                str(self._rng.choice(["DDP", "CIF", "FOB", "FCA"])), 2)),
            "ucp_compliance_score":  float(np.clip(self._rng.beta(9, 1), 0.88, 1.0)),
            "n_ucp_warnings":        float(self._rng.choice([0, 0, 0, 0, 1])),
            "ucp_warning_rate":      0.0,
            "nlp_score":             float(np.clip(self._rng.normal(87, 5), 75, 100)),
            "semantic_score":        float(np.clip(self._rng.beta(8, 1.5), 0.75, 1.0)),
            "completeness_score":    comp,
            # Temporal VARIÉ — le modèle doit apprendre que 43j est aussi correct que 180j
            "days_to_shipment":      int(np.clip(self._rng.lognormal(3.7, 0.7), 10, 250)),
            "presentation_days":     int(self._rng.integers(10, 45)),
            "validity_days":         int(self._rng.integers(21, 400)),
            "transshipment_allowed": 0,
            "charges_beneficiary":   0,
            # Nouvelles features : parfaites
            "ucp_errors_count":        0.0,
            "ucp_warnings_count":      float(self._rng.choice([0, 0, 0, 1])),
            "missing_critical_score":  float(np.clip(comp * 1.05 + self._rng.normal(0, 0.03), 0.85, 1.0)),
            "coherence_dates":         1.0,
            "contradiction_flag":      0.0,
            "country_risk_applicant":  float(np.clip(self._rng.normal(cr_score * 0.9, 3), 0, 20)),
            "country_risk_beneficiary":cr_score,
            "amount_tier":             0.0,
            "red_clause_flag":         0.0,
            "bank_tier":               float(self._rng.choice([1, 1, 2], p=[0.50, 0.35, 0.15])),
            "tenor_risk":              float(self._rng.choice([0.10, 0.10, 0.35], p=[0.60, 0.30, 0.10])),
        }
        row["_true_override"] = "LOW"
        return self._finalize_row(row)

    def _profile_clean_lc(self) -> dict:
        """
        LC propre avec pays tier 1 et UCP >= 85% — TOUJOURS LOW.
        Variante avec montants et temporel variés (incl. petits montants tier 1).
        """
        tier     = int(self._rng.choice([0, 1], p=[0.6, 0.4]))
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        currency = _pick_currency(ben, self._rng)
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 0, 35))
        cur_risk = _currency_risk(currency, ben)
        n_clauses  = int(self._rng.integers(5, 15))
        n_warnings = int(self._rng.choice([0, 0, 1, 1, 2]))
        inc        = str(self._rng.choice(["FOB", "CIF", "CPT", "DDP"]))
        amount     = float(self._rng.lognormal(11.5, 1.5))  # large variance de montant
        comp       = float(np.clip(self._rng.beta(7, 2), 0.65, 1.0))

        row = {
            "lc_amount":             amount,
            "country_risk_score":    cr_score,
            "external_currency_risk": float(min(cur_risk, 1)),
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get(inc, 3)),
            "ucp_compliance_score":  float(np.clip(self._rng.beta(7, 2), 0.70, 1.0)),
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(75, 8), 60, 100)),
            "semantic_score":        float(self._rng.beta(6, 2)),
            "completeness_score":    comp,
            "days_to_shipment":      int(np.clip(self._rng.lognormal(3.6, 0.7), 10, 300)),
            "presentation_days":     int(self._rng.integers(10, 40)),
            "validity_days":         int(self._rng.integers(21, 400)),
            "transshipment_allowed": int(self._rng.choice([0, 1], p=[0.80, 0.20])),
            "charges_beneficiary":   int(self._rng.choice([0, 1], p=[0.85, 0.15])),
        }
        row.update(self._new_features_low(n_warnings, comp, cr_score))
        row["_true_override"] = "LOW"
        return self._finalize_row(row)

    def _profile_medium(self) -> dict:
        """
        LC risque intermédiaire : UCP 40-80%, pays émergent, quelques warnings.
        CORRECTION v4.1 : UCP cappé à 0.80 (jamais > 0.80 dans MEDIUM).
        La distinction LOW/MEDIUM est portée par UCP + country_risk,
        JAMAIS par les seules features temporelles.
        """
        tier     = int(self._rng.choice([1, 1, 2]))
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        currency = _pick_currency(ben, self._rng)
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 15.0, 70.0))
        cur_risk = _currency_risk(currency, ben)
        n_clauses  = int(self._rng.integers(10, 22))
        n_warnings = int(self._rng.choice(range(2, 8), p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]))
        inc        = str(self._rng.choice(["CIF", "EXW", "FOB", "DAP", "FCA"]))
        comp       = float(self._rng.beta(4, 3))

        row = {
            "lc_amount":             float(self._rng.lognormal(13.3, 1.0)),
            "country_risk_score":    cr_score,
            "external_currency_risk": float(min(cur_risk, 2)),
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get(inc, 3)),
            # CORRECTION : UCP cappé à 0.80 — un MEDIUM ne peut PAS avoir UCP parfait
            "ucp_compliance_score":  float(np.clip(self._rng.beta(4, 4), 0.0, 0.80)),
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(57, 10), 30, 85)),
            "semantic_score":        float(self._rng.beta(3, 3)),
            "completeness_score":    comp,
            # CORRECTION : même range temporal que LOW → pas de signal parasite
            "days_to_shipment":      int(np.clip(self._rng.lognormal(3.5, 0.7), 10, 250)),
            "presentation_days":     int(self._rng.integers(8, 40)),
            "validity_days":         int(self._rng.integers(21, 365)),
            "transshipment_allowed": int(self._rng.choice([0, 1], p=[0.55, 0.45])),
            "charges_beneficiary":   int(self._rng.choice([0, 1], p=[0.55, 0.45])),
        }
        row.update(self._new_features_medium(n_warnings, comp, cr_score))
        return self._finalize_row(row)

    def _profile_high(self) -> dict:
        """LC haut risque : pays sensible, documents insuffisants, violations UCP nombreuses."""
        tier     = int(self._rng.choice([2, 3, 3]))
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        currency = _pick_currency(ben, self._rng)
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 50.0, 100.0))
        cur_risk = _currency_risk(currency, ben)
        n_clauses  = int(self._rng.integers(18, 38))
        n_warnings = int(self._rng.integers(5, 16))
        inc        = str(self._rng.choice(["EXW", "FCA", "FOB"]))
        if self._rng.random() < 0.35:
            amount = float(self._rng.uniform(500, 8_000))
        else:
            amount = float(self._rng.lognormal(15.5, 0.7))
        comp = float(self._rng.beta(2, 5))

        row = {
            "lc_amount":             amount,
            "country_risk_score":    cr_score,
            "external_currency_risk": float(min(cur_risk, 3)),
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get(inc, 3)),
            "ucp_compliance_score":  float(self._rng.beta(2, 6)),
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(32, 12), 0, 65)),
            "semantic_score":        float(self._rng.beta(2, 6)),
            "completeness_score":    comp,
            "days_to_shipment":      int(max(1, self._rng.exponential(5))),   # très court
            "presentation_days":     int(self._rng.integers(1, 7)),
            "validity_days":         int(self._rng.integers(5, 28)),   # max 28 — gap net avec normal (30-365)
            "transshipment_allowed": 1,
            "charges_beneficiary":   1,
        }
        row.update(self._new_features_high(n_warnings, comp, cr_score))
        return self._finalize_row(row)

    # ── Cas spéciaux ──────────────────────────────────────────────────────────

    def _case_sanctioned_clean_docs(self) -> dict:
        """Pays sanctionné avec documents parfaits — contournement OFAC."""
        ben      = self._rng.choice(list(SANCTIONED_ISO2))
        currency = COUNTRY_DEFAULT_CURRENCY.get(ben, "USD")
        cur_risk = _currency_risk(currency, ben)
        n_clauses  = int(self._rng.integers(8, 18))
        n_warnings = int(self._rng.integers(1, 4))
        row = self._profile_low()
        row.update({
            "country_risk_score":      float(np.clip(self._rng.normal(92, 4), 85, 100)),
            "external_currency_risk":  float(cur_risk),
            "nlp_score":               float(np.clip(self._rng.normal(83, 4), 0, 100)),
            "ucp_compliance_score":    float(self._rng.beta(9, 2)),
            "n_ucp_warnings":          float(n_warnings),
            "ucp_warning_rate":        n_warnings / max(n_clauses, 1),
            "country_risk_applicant":  float(np.clip(self._rng.normal(92, 4), 85, 100)),
            "country_risk_beneficiary":float(np.clip(self._rng.normal(92, 4), 85, 100)),
        })
        row["_true_override"] = "HIGH"
        return row

    def _case_hidden_anomaly_amount(self) -> dict:
        """LC apparemment LOW mais montant très suspect + délai ultra-court."""
        row = self._profile_low()
        amount = float(self._rng.lognormal(16.0, 0.3))
        row["lc_amount"]         = amount
        row["amount_tier"]       = float(compute_amount_tier(amount))
        row["days_to_shipment"]  = int(self._rng.integers(2, 8))
        row["presentation_days"] = int(self._rng.integers(3, 7))
        n_clauses  = int(self._rng.integers(22, 32))
        n_warnings = int(self._rng.integers(3, 7))
        row["n_ucp_warnings"]    = float(n_warnings)
        row["ucp_warning_rate"]  = n_warnings / max(n_clauses, 1)
        errors, warns = _split_ucp_warnings(n_warnings, False, self._rng)
        row["ucp_errors_count"]   = float(errors)
        row["ucp_warnings_count"] = float(warns)
        row["_true_override"]    = "MEDIUM"
        return row

    def _case_contradictory(self) -> dict:
        """Bon NLP mais warnings UCP élevés + documents incomplets."""
        row = self._profile_medium()
        row["nlp_score"]          = float(np.clip(self._rng.normal(82, 5), 0, 100))
        row["semantic_score"]     = float(self._rng.beta(7, 2))
        row["contradiction_flag"] = 1.0
        n_clauses  = int(self._rng.integers(12, 22))
        n_warnings = int(self._rng.integers(6, 12))
        row["n_ucp_warnings"]   = float(n_warnings)
        row["ucp_warning_rate"] = n_warnings / max(n_clauses, 1)
        row["completeness_score"] = float(self._rng.beta(2, 5))
        errors, warns = _split_ucp_warnings(n_warnings, False, self._rng)
        row["ucp_errors_count"]   = float(errors)
        row["ucp_warnings_count"] = float(warns)
        return row

    def _case_human_error(self) -> dict:
        """Incohérence temporelle : validity < days_to_shipment + presentation_days."""
        row = self._profile_medium()
        days = int(self._rng.integers(60, 100))
        row["days_to_shipment"]  = days
        row["validity_days"]     = int(self._rng.integers(20, days - 5))
        row["presentation_days"] = int(self._rng.integers(1, 5))
        row["coherence_dates"]   = 0.0
        row["contradiction_flag"] = 1.0
        return row

    def _case_fraud_pattern(self) -> dict:
        """Montant rond + pays risque 2-3 + délai extrême + transbordement."""
        tier     = int(self._rng.choice([2, 3]))
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        currency = _pick_currency(ben, self._rng)
        cur_risk = _currency_risk(currency, ben)
        base_amounts = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
        amount   = float(self._rng.choice(base_amounts)) * float(self._rng.uniform(0.97, 1.03))
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 55.0, 100.0))
        n_clauses  = int(self._rng.integers(25, 40))
        n_warnings = int(self._rng.integers(7, 14))
        comp = float(self._rng.beta(3, 5))
        errors, warns = _split_ucp_warnings(n_warnings, True, self._rng)

        row = {
            "lc_amount":             amount,
            "country_risk_score":    cr_score,
            "external_currency_risk": float(cur_risk),
            "incoterm_risk":         float(INCOTERM_RISK_MAP["EXW"]),
            "ucp_compliance_score":  float(self._rng.beta(3, 5)),
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(42, 10), 0, 100)),
            "semantic_score":        float(self._rng.beta(2, 5)),
            "completeness_score":    comp,
            "days_to_shipment":      int(self._rng.integers(1, 6)),
            "presentation_days":     int(self._rng.integers(2, 6)),
            "validity_days":         int(self._rng.integers(10, 30)),
            "transshipment_allowed": 1,
            "charges_beneficiary":   1,
            "ucp_errors_count":       float(errors),
            "ucp_warnings_count":     float(warns),
            "missing_critical_score": _missing_critical_from_completeness(comp, self._rng),
            "coherence_dates":        float(self._rng.choice([1.0, 0.5, 0.0], p=[0.30, 0.35, 0.35])),
            "contradiction_flag":     float(self._rng.choice([0, 1], p=[0.40, 0.60])),
            "country_risk_applicant": float(np.clip(self._rng.normal(cr_score * 0.9, 10), 0, 100)),
            "country_risk_beneficiary":cr_score,
            "amount_tier":            0.0,
            "red_clause_flag":        float(self._rng.choice([0, 1], p=[0.55, 0.45])),
            "bank_tier":              float(_bank_tier_for_country(2 if tier == 2 else 3, self._rng)),
            "tenor_risk":             float(self._rng.choice([0.50, 0.65, 0.80, 0.90])),
            "_true_override":         "HIGH",
        }
        return self._finalize_row(row)

    def _case_usance_lc(self) -> dict:
        """LC à terme (usance) : présentation étendue — risque MEDIUM."""
        row = self._profile_medium()
        row["presentation_days"] = int(self._rng.integers(60, 180))
        row["validity_days"]     = int(self._rng.integers(120, 370))
        row["days_to_shipment"]  = int(self._rng.integers(5, 30))
        row["tenor_risk"]        = float(self._rng.choice([0.65, 0.75, 0.88]))
        return row

    def _case_red_clause(self) -> dict:
        """Avance pré-expédition : montant élevé + UCP warnings notables."""
        row = self._profile_medium()
        amount = float(self._rng.lognormal(14.5, 0.5))
        row["lc_amount"]             = amount
        row["amount_tier"]           = float(compute_amount_tier(amount))
        n_clauses  = int(self._rng.integers(20, 35))
        n_warnings = int(self._rng.integers(4, 9))
        row["n_ucp_warnings"]        = float(n_warnings)
        row["ucp_warning_rate"]      = n_warnings / max(n_clauses, 1)
        row["ucp_compliance_score"]  = float(np.clip(self._rng.beta(3, 4), 0.0, 0.75))
        row["transshipment_allowed"] = 1
        row["charges_beneficiary"]   = 1
        row["red_clause_flag"]       = 1.0
        row["_true_override"]        = "HIGH"  # clause rouge → toujours HIGH
        return row

    def _case_transferable(self) -> dict:
        """LC transférable : bénéficiaire intermédiaire."""
        tier     = int(self._rng.choice([1, 2]))
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 20.0, 65.0))
        currency = _pick_currency(ben, self._rng)
        cur_risk = _currency_risk(currency, ben)
        row = self._profile_medium()
        row.update({
            "country_risk_score":      cr_score,
            "external_currency_risk":  float(cur_risk),
            "ucp_compliance_score":    float(np.clip(self._rng.beta(3, 4), 0.0, 0.78)),
            "country_risk_applicant":  float(np.clip(self._rng.normal(cr_score * 0.9, 8), 0, 80)),
            "country_risk_beneficiary":cr_score,
        })
        row["_true_override"] = "MEDIUM"
        return row

    def _case_commodity_risk(self) -> dict:
        """Marchandises dual-use : nombreuses clauses UCP + pays tier 2."""
        tier     = 2
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 35.0, 75.0))
        currency = _pick_currency(ben, self._rng)
        cur_risk = _currency_risk(currency, ben)
        n_clauses  = int(self._rng.integers(22, 38))
        n_warnings = int(self._rng.integers(3, 9))
        inc        = str(self._rng.choice(["CIF", "DAP", "EXW"]))
        comp       = float(self._rng.beta(3, 4))
        errors, warns = _split_ucp_warnings(n_warnings, False, self._rng)

        row = {
            "lc_amount":             float(self._rng.lognormal(14.0, 0.7)),
            "country_risk_score":    cr_score,
            "external_currency_risk": float(cur_risk),
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get(inc, 3)),
            "ucp_compliance_score":  float(np.clip(self._rng.beta(3, 4), 0.0, 0.78)),
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(52, 10), 0, 100)),
            "semantic_score":        float(self._rng.beta(3, 4)),
            "completeness_score":    comp,
            "days_to_shipment":      int(np.clip(self._rng.lognormal(3.2, 0.6), 10, 250)),
            "presentation_days":     int(self._rng.integers(8, 40)),
            "validity_days":         int(self._rng.integers(21, 365)),
            "transshipment_allowed": int(self._rng.choice([0, 1], p=[0.40, 0.60])),
            "charges_beneficiary":   int(self._rng.choice([0, 1], p=[0.50, 0.50])),
            "ucp_errors_count":       float(errors),
            "ucp_warnings_count":     float(warns),
            "missing_critical_score": _missing_critical_from_completeness(comp, self._rng),
            "coherence_dates":        float(self._rng.choice([1.0, 0.5, 0.5], p=[0.60, 0.25, 0.15])),
            "contradiction_flag":     float(self._rng.choice([0, 0, 1], p=[0.65, 0.20, 0.15])),
            "country_risk_applicant": float(np.clip(self._rng.normal(cr_score * 0.85, 8), 0, 100)),
            "country_risk_beneficiary":cr_score,
            "amount_tier":            0.0,
            "red_clause_flag":        0.0,
            "bank_tier":              float(_bank_tier_for_country(2, self._rng)),
            "tenor_risk":             float(self._rng.choice([0.35, 0.50, 0.65])),
            "_true_override":         "MEDIUM",
        }
        return self._finalize_row(row)

    def _case_micro_lc_suspicious(self) -> dict:
        """Micro-LC (< 5 000 USD) : potentiellement structuration / test."""
        tier     = int(self._rng.choice([0, 1, 2]))
        ben      = self._rng.choice(COUNTRIES_BY_RISK[tier])
        cr_score = float(_country_risk_score(ben, self._rng))
        currency = _pick_currency(ben, self._rng)
        cur_risk = _currency_risk(currency, ben)
        row      = self._profile_low()
        amount   = float(self._rng.uniform(300, 4_500))
        row.update({
            "lc_amount":             amount,
            "amount_tier":           float(compute_amount_tier(amount)),
            "country_risk_score":    cr_score,
            "external_currency_risk": float(cur_risk),
            "country_risk_applicant": float(np.clip(self._rng.normal(cr_score * 0.9, 5), 0, 100)),
            "country_risk_beneficiary": cr_score,
        })
        return row

    def _case_high_amount_clean_country(self) -> dict:
        """Montant très élevé, pays clean mais structure documentaire fragile."""
        ben      = self._rng.choice(COUNTRIES_BY_RISK[0])
        currency = _pick_currency(ben, self._rng)
        cur_risk = _currency_risk(currency, ben)
        cr_score = float(np.clip(_country_risk_score(ben, self._rng), 0, 20))
        n_clauses  = int(self._rng.integers(15, 28))
        n_warnings = int(self._rng.integers(3, 8))
        comp       = float(self._rng.beta(5, 3))
        amount     = float(self._rng.lognormal(16.5, 0.3))  # > 10M
        errors, warns = _split_ucp_warnings(n_warnings, False, self._rng)

        row = {
            "lc_amount":             amount,
            "country_risk_score":    cr_score,
            "external_currency_risk": float(cur_risk),
            "incoterm_risk":         float(INCOTERM_RISK_MAP.get("CIF", 2)),
            "ucp_compliance_score":  float(np.clip(self._rng.beta(4, 4), 0.0, 0.78)),
            "n_ucp_warnings":        float(n_warnings),
            "ucp_warning_rate":      n_warnings / max(n_clauses, 1),
            "nlp_score":             float(np.clip(self._rng.normal(62, 10), 0, 100)),
            "semantic_score":        float(self._rng.beta(5, 3)),
            "completeness_score":    comp,
            "days_to_shipment":      int(np.clip(self._rng.lognormal(3.4, 0.6), 10, 200)),
            "presentation_days":     int(self._rng.integers(14, 35)),
            "validity_days":         int(self._rng.integers(60, 300)),
            "transshipment_allowed": int(self._rng.choice([0, 1], p=[0.60, 0.40])),
            "charges_beneficiary":   int(self._rng.choice([0, 1], p=[0.60, 0.40])),
            "ucp_errors_count":       float(errors),
            "ucp_warnings_count":     float(warns),
            "missing_critical_score": _missing_critical_from_completeness(comp, self._rng),
            "coherence_dates":        float(self._rng.choice([1.0, 0.5, 0.5], p=[0.65, 0.25, 0.10])),
            "contradiction_flag":     float(self._rng.choice([0, 0, 1], p=[0.60, 0.25, 0.15])),
            "country_risk_applicant": float(np.clip(self._rng.normal(cr_score * 0.9, 5), 0, 30)),
            "country_risk_beneficiary": cr_score,
            "amount_tier":            0.0,
            "red_clause_flag":        0.0,
            "bank_tier":              float(_bank_tier_for_country(0, self._rng)),
            "tenor_risk":             float(self._rng.choice([0.35, 0.50, 0.65])),
            "_true_override":         "MEDIUM",
        }
        return self._finalize_row(row)

    # ── Pipeline principale ────────────────────────────────────────────────────

    def generate(self, n_samples: int = 30_000) -> pd.DataFrame:
        print(f"Generation de {n_samples} LCs synthetiques v4.1...")
        print(f"Features: {len(FEATURE_COLS)} colonnes ({FEATURE_COLS[:5]}...)")

        # Distribution v4.1 — 35% LOW (augmenté), profils "parfait" représentés
        distribution = {
            "low":                   int(n_samples * 0.22),
            "perfect_conforme":      int(n_samples * 0.08),  # NOUVEAU — expl. zone sure
            "clean_lc":              int(n_samples * 0.06),  # NOUVEAU — variante LOW
            "medium":                int(n_samples * 0.22),
            "high":                  int(n_samples * 0.16),
            "sanctioned_clean":      int(n_samples * 0.05),
            "hidden_anomaly_amount": int(n_samples * 0.04),
            "contradictory":         int(n_samples * 0.04),
            "human_error":           int(n_samples * 0.03),
            "fraud_pattern":         int(n_samples * 0.04),
            "usance_lc":             int(n_samples * 0.02),
            "red_clause":            int(n_samples * 0.02),
            "transferable":          int(n_samples * 0.01),
            "commodity_risk":        int(n_samples * 0.01),
            "micro_lc":              int(n_samples * 0.00),
            "high_amount_clean":     0,  # complète jusqu'à n_samples
        }
        total = sum(distribution.values())
        distribution["high_amount_clean"] = n_samples - total

        generators = {
            "low":                   self._profile_low,
            "perfect_conforme":      self._profile_perfect_conforme,
            "clean_lc":              self._profile_clean_lc,
            "medium":                self._profile_medium,
            "high":                  self._profile_high,
            "sanctioned_clean":      self._case_sanctioned_clean_docs,
            "hidden_anomaly_amount": self._case_hidden_anomaly_amount,
            "contradictory":         self._case_contradictory,
            "human_error":           self._case_human_error,
            "fraud_pattern":         self._case_fraud_pattern,
            "usance_lc":             self._case_usance_lc,
            "red_clause":            self._case_red_clause,
            "transferable":          self._case_transferable,
            "commodity_risk":        self._case_commodity_risk,
            "micro_lc":              self._case_micro_lc_suspicious,
            "high_amount_clean":     self._case_high_amount_clean_country,
        }

        records = []
        for profile_name, n in distribution.items():
            gen_fn = generators[profile_name]
            for _ in range(max(0, n)):
                row = gen_fn()
                true_override = row.pop("_true_override", None)
                row["_base_profile"] = profile_name
                records.append((row, true_override))

        df_raw    = pd.DataFrame([r for r, _ in records])
        overrides = [o for _, o in records]

        # Nettoyage et validation
        df = self._clip_features(df_raw)

        # Labels expert v4.1
        true_labels = []
        for i, row_dict in enumerate(df.to_dict("records")):
            override = overrides[i]
            if override:
                true_labels.append(override)
            else:
                true_labels.append(expert_label(row_dict, self._rng))
        df["true_label"] = true_labels

        # Weak supervision
        df["ws_label"]       = df.apply(lambda r: _ws_label(r.to_dict()), axis=1)
        df["label_mismatch"] = (df["true_label"] != df["ws_label"]).astype(int)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.drop(columns=["_base_profile"], errors="ignore")
        return df

    def _clip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip les 26 features aux plages valides."""
        clips = {
            "lc_amount":              (100.0,   50_000_000.0),
            "country_risk_score":     (0.0,     100.0),
            "external_currency_risk": (0.0,     4.0),
            "incoterm_risk":          (1.0,     5.0),
            "ucp_compliance_score":   (0.0,     1.0),
            "n_ucp_warnings":         (0.0,     50.0),
            "ucp_warning_rate":       (0.0,     10.0),
            "nlp_score":              (0.0,     100.0),
            "semantic_score":         (0.0,     1.0),
            "completeness_score":     (0.0,     1.0),
            "days_to_shipment":       (1,       730),
            "presentation_days":      (1,       180),
            "validity_days":          (5,       730),
            "transshipment_allowed":  (0,       1),
            "charges_beneficiary":    (0,       1),
            # Nouvelles v4.0
            "ucp_errors_count":       (0.0,     50.0),
            "ucp_warnings_count":     (0.0,     50.0),
            "missing_critical_score": (0.0,     1.0),
            "coherence_dates":        (0.0,     1.0),
            "contradiction_flag":     (0.0,     1.0),
            "country_risk_applicant": (0.0,     100.0),
            "country_risk_beneficiary":(0.0,    100.0),
            "amount_tier":            (1.0,     5.0),
            "red_clause_flag":        (0.0,     1.0),
            "bank_tier":              (1.0,     5.0),
            "tenor_risk":             (0.0,     1.0),
        }
        for col, (lo, hi) in clips.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        int_cols = ("days_to_shipment", "presentation_days", "validity_days",
                    "transshipment_allowed", "charges_beneficiary",
                    "n_ucp_warnings", "ucp_errors_count", "ucp_warnings_count")
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].round().astype(float)

        # Légère perturbation réaliste (8% des données)
        n   = len(df)
        idx = self._rng.choice(df.index, int(n * 0.08), replace=False)
        df.loc[idx, "nlp_score"] = np.clip(
            df.loc[idx, "nlp_score"] + self._rng.normal(0, 6, len(idx)), 0, 100)
        df.loc[idx, "ucp_compliance_score"] = np.clip(
            df.loc[idx, "ucp_compliance_score"] + self._rng.normal(0, 0.04, len(idx)), 0, 1)
        df.loc[idx, "completeness_score"] = np.clip(
            df.loc[idx, "completeness_score"] + self._rng.normal(0, 0.04, len(idx)), 0, 1)

        # S'assurer que les profils MEDIUM gardent UCP < 0.82 après perturbation
        medium_like = (df["ucp_compliance_score"] > 0.82) & (df["n_ucp_warnings"] >= 2)
        if medium_like.any():
            df.loc[medium_like, "ucp_compliance_score"] = np.clip(
                df.loc[medium_like, "ucp_compliance_score"] - 0.05, 0, 0.82)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# WEAK SUPERVISION — rule engine simplifié (mesure désaccord vs expert)
# ─────────────────────────────────────────────────────────────────────────────

def _ws_score(row: dict) -> float:
    s = 0.0
    cr = float(row.get("country_risk_score", 0))
    if cr >= 88:                                              s += 0.40
    elif cr >= 70:                                            s += 0.30
    elif cr >= 45:                                            s += 0.15
    elif cr >= 22:                                            s += 0.06
    if row.get("ucp_errors_count", 0) >= 3:                  s += 0.28
    elif row.get("ucp_errors_count", 0) >= 1:                s += 0.12
    if row.get("ucp_warnings_count", 0) >= 6:                s += 0.15
    elif row.get("ucp_warnings_count", 0) >= 3:              s += 0.07
    if row.get("n_ucp_warnings", 0) >= 8:                    s += 0.10
    if row.get("days_to_shipment", 30) < 5:                  s += 0.25
    elif row.get("days_to_shipment", 30) < 10:               s += 0.12
    if row.get("nlp_score", 50) < 30:                        s += 0.15
    if row.get("ucp_compliance_score", 0.5) < 0.3:           s += 0.15
    if row.get("completeness_score", 0.5) < 0.5:             s += 0.08
    if row.get("missing_critical_score", 0.75) < 0.5:        s += 0.12
    if row.get("contradiction_flag", 0) >= 1:                s += 0.15
    if row.get("red_clause_flag", 0) >= 1:                   s += 0.20
    if row.get("lc_amount", 0) > 8_000_000:                  s += 0.06
    if row.get("lc_amount", 0) < 5_000:                      s += 0.06
    if row.get("transshipment_allowed", 0) == 1:             s += 0.04
    if row.get("charges_beneficiary", 0) == 1:               s += 0.03
    if row.get("bank_tier", 3) >= 4:                         s += 0.05
    if row.get("tenor_risk", 0.5) >= 0.80:                   s += 0.06
    return min(s, 1.0)


def _ws_label(row: dict) -> str:
    s = _ws_score(row)
    if s >= 0.45: return "HIGH"
    if s >= 0.20: return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_and_save(
    output_path: str = "services/risk_service/data/risk_dataset.csv",
    n_samples: int = 30_000,
) -> pd.DataFrame:
    gen = MT700DataGenerator(seed=42)
    df  = gen.generate(n_samples=n_samples)

    print(f"\nDataset v4.1 : {len(df)} lignes x {len(df.columns)} colonnes")
    print(f"\nFeatures ML ({len(FEATURE_COLS)}) :")
    print("  " + ", ".join(FEATURE_COLS))

    print(f"\nDistribution true_label :")
    print(df["true_label"].value_counts().to_frame("count")
          .assign(pct=lambda x: (x["count"] / len(df) * 100).round(1)))

    print(f"\nDistribution ws_label :")
    print(df["ws_label"].value_counts().to_frame("count")
          .assign(pct=lambda x: (x["count"] / len(df) * 100).round(1)))

    mismatch = df["label_mismatch"].mean() * 100
    print(f"\nTaux desaccord expert vs rule engine : {mismatch:.1f}%")
    print(f"  (objectif 15-30% — {'OK' if 10 <= mismatch <= 35 else 'AJUSTER'})")

    # Vérification zone sûre : les LCs avec UCP=1.0 et 0 erreur doivent être LOW
    safe_zone_mask = (
        (df["ucp_compliance_score"] >= 0.90)
        & (df["ucp_errors_count"] == 0)
        & (df["ucp_warnings_count"] <= 2)
        & (df["country_risk_score"] < 50)
    )
    if safe_zone_mask.any():
        dist_safe = df.loc[safe_zone_mask, "true_label"].value_counts()
        pct_low   = dist_safe.get("LOW", 0) / safe_zone_mask.sum() * 100
        print(f"\n[VALIDATION] Zone sure (UCP>=90%, 0 erreur, pays sain) :")
        print(f"  {safe_zone_mask.sum()} cas | {pct_low:.1f}% labelises LOW")
        if pct_low < 90:
            print("  ATTENTION : < 90% LOW dans la zone sure — revoir expert_label()")

    df_ml = df[FEATURE_COLS + ["true_label", "ws_label", "label_mismatch"]]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_ml.to_csv(output_path, index=False)
    print(f"\nSauvegarde -> {output_path}")
    print(f"  ({len(df_ml)} lignes, {len(df_ml.columns)} colonnes)")
    return df_ml


if __name__ == "__main__":
    df = generate_and_save("services/risk_service/data/risk_dataset.csv", n_samples=30_000)
