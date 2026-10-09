"""
external_validation_b/engine.py
Orchestrateur — Approche B (APIs web uniquement).

MODIFICATIONS :
  ✦ MODE PAYS + DEVISE SEULEMENT
    - Vérifications bancaires, sociétés et géographie désactivées.
    - Ajout du contrôle de devise (code ISO 4217 + liste devises à risque).
    - Poids : pays 0.7 / devise 0.3
"""

from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .country_checker import RiskLevel, check_all_countries, CountryResult
from .country_checker import COUNTRY_NAME_TO_ISO2


WEIGHT_COUNTRY  = 0.7
WEIGHT_CURRENCY = 0.3

# Somme des score_contribution pays, plafonnée avant pondération (AML multi-juridictions)
COUNTRY_SCORE_CAP = 12

SCORE_THRESHOLDS = {
    RiskLevel.CRITICAL: 12,
    RiskLevel.HIGH:      8,
    RiskLevel.MEDIUM:    4,
    RiskLevel.LOW:       1,
}

_RISK_LEVEL_ORDER = (
    RiskLevel.CLEAN,
    RiskLevel.LOW,
    RiskLevel.MEDIUM,
    RiskLevel.HIGH,
    RiskLevel.CRITICAL,
)


def _dedupe_country_results(results: list[CountryResult]) -> list[CountryResult]:
    """Une entrée par ISO2 — conserve le niveau de risque le plus sévère."""
    by_iso: dict[str, CountryResult] = {}

    def _rank(level: RiskLevel) -> int:
        try:
            return _RISK_LEVEL_ORDER.index(level)
        except ValueError:
            return 0

    for r in results:
        iso = (r.country_iso2 or "").strip().upper()[:2]
        if len(iso) < 2:
            continue
        prev = by_iso.get(iso)
        if prev is None or _rank(r.risk_level) > _rank(prev.risk_level):
            by_iso[iso] = r
        elif prev is not None and _rank(r.risk_level) == _rank(prev.risk_level):
            if int(r.score_contribution) > int(prev.score_contribution):
                by_iso[iso] = r
    return list(by_iso.values())

# Liste simplifiée des devises à risque (exemples)
HIGH_RISK_CURRENCIES = {"IRR", "KPW", "SYP", "CUP", "CNY", "RUB"}   # à ajuster selon besoins
MEDIUM_RISK_CURRENCIES = {"TRY", "ZAR", "VEF"}

@dataclass
class CurrencyResult:
    currency_code: str
    is_valid: bool
    risk_level: RiskLevel
    score_contribution: float
    details: str
    sources: list[str] = field(default_factory=list)
    api_responses: dict = field(default_factory=dict)


@dataclass
class ExternalValidationResult:
    lc_number: str
    processed_at: str
    country_results: list[CountryResult] = field(default_factory=list)
    currency_result: CurrencyResult | None = None
    raw_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.CLEAN
    risk_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    processing_ms: int = 0
    sources_consulted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "lc_number": self.lc_number,
            "processed_at": self.processed_at,
            "risk_level": self.risk_level.value,
            "risk_score": round(self.raw_score, 2),
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "sources_consulted": self.sources_consulted,
            "processing_ms": self.processing_ms,
            "approach": "B — Pays + Devise uniquement (OFAC, EU, UN, FATF + ISO 4217)",
            "details": {
                "countries": [
                    {
                        "raw": r.country_raw,
                        "iso2": r.country_iso2,
                        "risk": r.risk_level.value,
                        "sources": r.sources,
                        "details": r.details,
                        "api_responses": r.api_responses,
                    }
                    for r in self.country_results
                ],
                "currency": {
                    "code": self.currency_result.currency_code if self.currency_result else None,
                    "is_valid": self.currency_result.is_valid if self.currency_result else False,
                    "risk": self.currency_result.risk_level.value if self.currency_result else "UNKNOWN",
                    "details": self.currency_result.details if self.currency_result else "",
                } if self.currency_result else None,
            },
        }


class ExternalValidationEngine:

    def _extract_entities(self, pipeline_output: dict) -> dict:
        """
        Extrait les entités nécessaires pour les vérifications pays + devise.
        """
        swift    = pipeline_output.get("swift", {})
        entities = swift.get("entities", {})

        # Pays du bénéficiaire (extrait depuis l'adresse)
        beneficiary_addr = (entities.get("beneficiary") or "").strip()
        beneficiary_country = None
        if beneficiary_addr:
            two_word_match = re.search(
                r'(NORTH KOREA|SOUTH KOREA|DPRK|NORTH COREA)',
                beneficiary_addr.upper()
            )
            if two_word_match:
                beneficiary_country = two_word_match.group(1)
            else:
                parts = beneficiary_addr.split()
                candidate = parts[-1] if parts else None
                # Heuristique: éviter de considérer un mot métier (ex: CHEMICAL) comme un pays.
                # On ne garde le dernier token que s'il ressemble à un ISO2 ou à un nom de pays connu.
                if candidate:
                    cand_up = candidate.strip().upper()
                    is_iso2 = len(cand_up) == 2 and cand_up.isalpha()
                    is_known_country = cand_up in COUNTRY_NAME_TO_ISO2
                    if is_iso2 or is_known_country:
                        beneficiary_country = candidate
                    else:
                        beneficiary_country = None

        return {
            "lc_number":          entities.get("lc_number", "UNKNOWN"),
            "applicant":          entities.get("applicant") or "",
            "beneficiary":        entities.get("beneficiary") or "",
            "country":            entities.get("country", ""),
            "beneficiary_country": beneficiary_country,
            "currency":           entities.get("currency", ""),          # nouveau
            "amount":             entities.get("amount", ""),
        }

    def _check_currency(self, currency_code: str, country_iso2: str = "") -> CurrencyResult:
        """
        Vérifie la devise via la validation ISO 4217 complète (cache_manager → pycountry)
        puis applique les listes de risque.
        """
        if not currency_code or not isinstance(currency_code, str):
            return CurrencyResult(
                currency_code=currency_code or "",
                is_valid=False,
                risk_level=RiskLevel.MEDIUM,
                score_contribution=3.0,
                details="Devise manquante ou invalide",
                sources=["internal"],
            )

        code = currency_code.strip().upper()

        # Validation ISO 4217 via cache_manager (délègue à pycountry)
        try:
            from .cache_manager import validate_currency as _validate_currency
            iso_result = _validate_currency(code, country_iso2)
        except Exception:
            iso_result = {"valid": len(code) == 3 and code.isalpha(), "errors": [], "warnings": []}

        if not iso_result.get("valid", False):
            errors = iso_result.get("errors", [])
            detail = errors[0] if errors else f"Code devise ISO-4217 invalide : {code}"
            return CurrencyResult(
                currency_code=code,
                is_valid=False,
                risk_level=RiskLevel.MEDIUM,
                score_contribution=3.0,
                details=detail,
                sources=["ISO-4217"],
            )

        # Niveau de risque selon les listes prédéfinies
        if code in HIGH_RISK_CURRENCIES:
            level = RiskLevel.HIGH
            score = 7.0
            details = f"Devise à haut risque : {code} ({iso_result.get('name', '')})"
        elif code in MEDIUM_RISK_CURRENCIES:
            level = RiskLevel.MEDIUM
            score = 4.0
            details = f"Devise à risque modéré : {code} ({iso_result.get('name', '')})"
        else:
            level = RiskLevel.CLEAN
            score = 0.0
            details = f"Devise acceptable : {code} ({iso_result.get('name', '')})"

        extra_warnings = iso_result.get("warnings", [])
        if extra_warnings and level == RiskLevel.CLEAN:
            level = RiskLevel.LOW
            score = 1.0
            details += f" — {extra_warnings[0]}"

        return CurrencyResult(
            currency_code=code,
            is_valid=True,
            risk_level=level,
            score_contribution=score,
            details=details,
            sources=["ISO-4217", "internal-currency-list"],
        )

    def validate(self, pipeline_output: dict) -> ExternalValidationResult:
        """
        Exécute en parallèle les vérifications pays et devise.
        """
        t_start = datetime.now()
        entities = self._extract_entities(pipeline_output)

        if not entities["beneficiary"]:
            t_end = datetime.now()
            return ExternalValidationResult(
                lc_number=entities["lc_number"],
                processed_at=t_end.isoformat(),
                risk_level=RiskLevel.CRITICAL,
                raw_score=12.0,
                risk_factors=["Bénéficiaire absent — validation externe automatiquement critique"],
                recommendations=["⛔ BLOQUER — Bénéficiaire manquant dans le message SWIFT"],
                processing_ms=int((t_end - t_start).total_seconds() * 1000),
                sources_consulted=["internal"],
            )

        sources_consulted: list[str] = []

        raw_countries = [c for c in [entities["country"], entities["beneficiary_country"]] if c]
        seen_c: set[str] = set()
        countries: list[str] = []
        for c in raw_countries:
            k = c.strip().upper()
            if k and k not in seen_c:
                seen_c.add(k)
                countries.append(c.strip())
        currency = entities.get("currency", "")

        country_results: list[CountryResult] = []
        currency_result: CurrencyResult | None = None

        # ISO2 du premier pays (pour cohérence pays↔devise)
        first_iso2 = entities.get("country", "") or ""

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if countries:
                futures[executor.submit(check_all_countries, countries)] = "countries"
            if currency:
                futures[executor.submit(self._check_currency, currency, first_iso2)] = "currency"

            for future in as_completed(futures, timeout=15):
                key = futures[future]
                try:
                    result = future.result()
                    if key == "countries":
                        country_results = result
                        sources_consulted += ["OFAC-COUNTRIES", "EU-SANCTIONS", "FATF"]
                    elif key == "currency":
                        currency_result = result
                        sources_consulted += ["ISO-4217", "internal-currency-list"]
                except Exception as exc:
                    print(f"[WARN] Module '{key}' a échoué : {exc}")

        country_results = _dedupe_country_results(country_results)

        # Calcul des scores (somme des pays, plafonnée — plusieurs juridictions critiques comptent)
        country_score = sum(r.score_contribution for r in country_results)
        country_score = min(float(country_score), float(COUNTRY_SCORE_CAP))
        currency_score = currency_result.score_contribution if currency_result else 0

        raw_score = country_score * WEIGHT_COUNTRY + currency_score * WEIGHT_CURRENCY

        # Détermination du niveau de risque global
        level = RiskLevel.CLEAN
        for lvl, threshold in SCORE_THRESHOLDS.items():
            if raw_score >= threshold:
                level = lvl
                break

        # Escalade si un sous-module est CRITICAL
        all_levels = [r.risk_level for r in country_results]
        if currency_result:
            all_levels.append(currency_result.risk_level)
        if RiskLevel.CRITICAL in all_levels:
            level = RiskLevel.CRITICAL

        # Construction des facteurs de risque et recommandations
        risk_factors: list[str] = []
        recommendations: list[str] = []
        seen_rf: set[str] = set()
        seen_rec: set[str] = set()

        for r in country_results:
            if r.risk_level not in (RiskLevel.CLEAN, RiskLevel.LOW):
                line = f"Pays {r.country_iso2} — {r.risk_level.value} ({r.details})"
                if line not in seen_rf:
                    seen_rf.add(line)
                    risk_factors.append(line)
                if r.risk_level == RiskLevel.CRITICAL:
                    rec = f"⛔ BLOQUER — Pays sous sanctions : {r.country_iso2}"
                else:
                    rec = f"⚠ KYC renforcé : {r.country_iso2}"
                if rec not in seen_rec:
                    seen_rec.add(rec)
                    recommendations.append(rec)

        if currency_result and currency_result.risk_level not in (RiskLevel.CLEAN, RiskLevel.LOW):
            cf = f"Devise {currency_result.currency_code} — {currency_result.risk_level.value} ({currency_result.details})"
            if cf not in seen_rf:
                seen_rf.add(cf)
                risk_factors.append(cf)
            if currency_result.risk_level == RiskLevel.HIGH:
                rec = f"⚠ Restreindre ou surveiller la devise : {currency_result.currency_code}"
            elif currency_result.risk_level == RiskLevel.MEDIUM:
                rec = f"⚠ Vérification supplémentaire sur la devise : {currency_result.currency_code}"
            else:
                rec = ""
            if rec and rec not in seen_rec:
                seen_rec.add(rec)
                recommendations.append(rec)

        if not risk_factors:
            risk_factors.append("Aucun risque détecté via les vérifications pays et devise")

        t_end = datetime.now()

        return ExternalValidationResult(
            lc_number=entities["lc_number"],
            processed_at=t_end.isoformat(),
            country_results=country_results,
            currency_result=currency_result,
            raw_score=raw_score,
            risk_level=level,
            risk_factors=risk_factors,
            recommendations=recommendations,
            processing_ms=int((t_end - t_start).total_seconds() * 1000),
            sources_consulted=list(set(sources_consulted)),
        )

    def inject_into_report(self, pipeline_output: dict, result: ExternalValidationResult) -> dict:
        pipeline_output["external_validation"] = result.to_dict()
        ds = pipeline_output.get("ds_analysis", {})
        risk = ds.get("risk", {})
        ORDER = ["CLEAN", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        current = risk.get("risk_level", "CLEAN")
        ext = result.risk_level.value
        if ORDER.index(ext) > ORDER.index(current):
            risk["risk_level"] = ext
            risk["_external_escalated"] = True
            prev_factors = list(risk.get("risk_factors") or [])
            seen = set(prev_factors)
            for f in result.risk_factors:
                line = f"[EXTERNAL-B] {f}"
                if line not in seen:
                    prev_factors.append(line)
                    seen.add(line)
            risk["risk_factors"] = prev_factors
        return pipeline_output