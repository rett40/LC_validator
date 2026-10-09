"""
risk_engine.py  —  v4.0  (compatible train_models.py v5.0 / risk_feature_schema.py v4.0)
==========================================================================================
Moteur de scoring de risque ML — intégration dans le pipeline SWIFT hybride.

Changements v4.0 (vs v3.0) :
  NEW-1 : 11 nouvelles features extraites (26 au total)
          — ucp_errors_count / ucp_warnings_count séparés
          — missing_critical_score : taux présence champs MT700 critiques
          — country_risk_applicant / country_risk_beneficiary : risque pays dual
          — coherence_dates : cohérence séquentielle dates LC
          — contradiction_flag : clauses contradictoires
          — amount_tier : classification montant LC
          — red_clause_flag : détection clause rouge 47A
          — bank_tier : tier banque advising/confirming
          — tenor_risk : risque lié au tenor
  NEW-2 : 14 règles UCP hard (vs 6) — nouveaux contrôles manquants, contradictions,
          montant anomalie, validité critique, clause rouge, double pays
  NEW-3 : score_breakdown 7 dimensions avec poids et contributions
          (completeness, ucp_compliance, coherence, country_risk, temporal, amount, ml_model)
  NEW-4 : missing_fields_detail — liste champs MT700 manquants avec criticité + impact
  NEW-5 : explanation schema — SHAP waterfall + dimension radar pour XAI lisible
  NEW-6 : NarrativeExplainer enrichie — labels 26 features + templates détaillés
  COMPAT : ML backward compat — le scaler v3.0 (15 features) continue de fonctionner ;
           les 11 nouvelles features alimentent uniquement les règles et le breakdown.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .risk_feature_schema import (
    FEATURE_COLS_CANONICAL,
    FEATURE_COLS_V3,
    FEATURE_SCHEMA_VERSION,
    FEATURE_DEFAULTS,
    INCOTERM_RISK_MAP,
    INCOTERM_RISK_DEFAULT,
    SANCTIONED_ISO2,
    MT700_CRITICAL_FIELDS,
    MT700_HIGH_FIELDS,
    MT700_OPTIONAL_FIELDS,
    AMOUNT_TIER_THRESHOLDS,
    BANK_TIER_KEYWORDS,
    TENOR_RISK_MAP,
    TENOR_RISK_DEFAULT,
    compute_amount_tier,
    compute_bank_tier,
    compute_commodity_risk,
    compute_tenor_risk,
    resolve_matrix_feature_columns,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = list(FEATURE_COLS_CANONICAL)

LABEL_MAP     = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
LABEL_MAP_INV = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

_LEVEL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": -1}

_EXT_RL_TO_SCORE: dict[str, float] = {
    "CLEAN":    0.0,
    "LOW":      25.0,
    "MEDIUM":   50.0,
    "HIGH":     75.0,
    "CRITICAL": 100.0,
}

_COUNTRY_RISK_TIER: dict[str, int] = {
    **{c: 0 for c in ["DE","FR","CH","NL","US","JP","CA","GB","AU","SG","SE","NO","DK","FI","AT","NZ","BE","LU","IE","HK"]},
    **{c: 1 for c in ["CN","IN","BR","MX","ZA","PL","TR","AE","SA","EG","MA","TN","KW","QA","OM","UA","RO","CZ","HU","SK","HR"]},
    **{c: 2 for c in ["NG","PK","BD","VN","ID","KE","GH","ET","DZ","IQ","LB","JO","AM","GE","AZ","UZ","KZ","TH","PH","MY","CO","PE","AR"]},
    **{c: 3 for c in ["AF","SY","YE","LY","MM","SD","ZW","VE","KP","IR","SO","CU","BY","ER","CF","ML","NE","GN","TD","MZ"]},
}
_TIER_TO_SCORE: dict[int, float] = {0: 5.0, 1: 25.0, 2: 50.0, 3: 75.0}

# Noms de pays → ISO2. Le parseur renvoie souvent le nom complet ("GERMANY")
# et non l'ISO ; tronquer à 2 lettres donnait "GE" (≠ "DE") → défaut 25 erroné.
_COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    "GERMANY": "DE", "FRANCE": "FR", "NETHERLANDS": "NL", "HOLLAND": "NL",
    "SWITZERLAND": "CH", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
    "USA": "US", "UNITED KINGDOM": "GB", "GREAT BRITAIN": "GB", "ENGLAND": "GB",
    "JAPAN": "JP", "CANADA": "CA", "AUSTRALIA": "AU", "SINGAPORE": "SG",
    "SWEDEN": "SE", "NORWAY": "NO", "DENMARK": "DK", "FINLAND": "FI",
    "AUSTRIA": "AT", "NEW ZEALAND": "NZ", "BELGIUM": "BE", "LUXEMBOURG": "LU",
    "IRELAND": "IE", "HONG KONG": "HK", "SPAIN": "ES", "ITALY": "IT", "PORTUGAL": "PT",
    "CHINA": "CN", "INDIA": "IN", "BRAZIL": "BR", "MEXICO": "MX", "SOUTH AFRICA": "ZA",
    "POLAND": "PL", "TURKEY": "TR", "TURKIYE": "TR", "UNITED ARAB EMIRATES": "AE",
    "UAE": "AE", "SAUDI ARABIA": "SA", "EGYPT": "EG", "MOROCCO": "MA", "TUNISIA": "TN",
    "KUWAIT": "KW", "QATAR": "QA", "OMAN": "OM", "UKRAINE": "UA", "ROMANIA": "RO",
    "CZECH REPUBLIC": "CZ", "HUNGARY": "HU", "SLOVAKIA": "SK", "CROATIA": "HR",
    "NIGERIA": "NG", "PAKISTAN": "PK", "BANGLADESH": "BD", "VIETNAM": "VN", "VIET NAM": "VN",
    "INDONESIA": "ID", "KENYA": "KE", "GHANA": "GH", "ETHIOPIA": "ET", "ALGERIA": "DZ",
    "IRAQ": "IQ", "LEBANON": "LB", "JORDAN": "JO", "THAILAND": "TH", "PHILIPPINES": "PH",
    "MALAYSIA": "MY", "COLOMBIA": "CO", "PERU": "PE", "ARGENTINA": "AR", "KAZAKHSTAN": "KZ",
    # Pays sanctionnés / haut risque
    "NORTH KOREA": "KP", "IRAN": "IR", "MYANMAR": "MM", "BURMA": "MM", "SYRIA": "SY",
    "CUBA": "CU", "SUDAN": "SD", "BELARUS": "BY", "AFGHANISTAN": "AF", "YEMEN": "YE",
    "LIBYA": "LY", "ZIMBABWE": "ZW", "VENEZUELA": "VE", "SOMALIA": "SO",
    "SOUTH KOREA": "KR", "REPUBLIC OF KOREA": "KR",
}

# Patterns clause rouge (field 47A)
_RED_CLAUSE_PATTERNS = [
    "red clause", "clause rouge", "paiement avance", "advance payment",
    "advance clause", "pre-shipment finance", "preamble clause",
    "payment before shipment", "pré-expédition",
]

# Score de contribution par sévérité de règle (pour la narration et le score impact)
_RULE_SEVERITY_CONTRIBUTION = {"BLOCK": "+100", "HIGH": "+30", "WARN": "+10"}

# Seuil de décision HIGH : le modèle est entraîné sur un dataset biaisé (42.6% HIGH).
# P(HIGH) doit atteindre ce seuil pour déclencher le label HIGH, ce qui compense
# la sur-représentation des cas HIGH en entraînement.
_HIGH_DECISION_THRESHOLD = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# UCP 600 RULE ENGINE — 14 règles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UCPViolation:
    article:        str
    field:          str
    description:    str
    severity:       str
    override_score: Optional[str] = None


class UCPRuleEngine:
    """
    14 règles UCP 600 hard-codées — prioritaires sur le modèle ML.
    Opère sur features_dict (inclut toutes les features v4.0 + clés internes).
    """

    RULES = [
        # ── Règles contractuelles BLOCK ───────────────────────────────────────
        {
            "id": "UCP_COMPLIANCE_REJECT",
            "check": lambda f: f.get("ucp_decision_reject", 0) >= 1,
            "violation": UCPViolation(
                article="UCP 600",
                field="47A / Multiple",
                description="ucp_compliance.ucp_decision=REJECT — red flags contractuels UCP détectés",
                severity="BLOCK",
                override_score="HIGH",
            ),
        },
        {
            "id": "OFAC_COUNTRY",
            "check": lambda f: f.get("country_risk_score", 0) >= 90,
            "violation": UCPViolation(
                article="OFAC / EU Embargo",
                field="59 / 50",
                description="Pays sous sanctions — country_risk_score ≥ 90 (OFAC/EU/UN/FATF)",
                severity="BLOCK",
                override_score="HIGH",
            ),
        },
        {
            "id": "DUAL_COUNTRY_SANCTION",
            "check": lambda f: (
                f.get("country_risk_applicant", 0) >= 90
                or f.get("country_risk_beneficiary", 0) >= 90
            ),
            "violation": UCPViolation(
                article="OFAC / EU Embargo Art.1",
                field="50 / 59",
                description="Applicant ou bénéficiaire dans un pays sanctionné (risk ≥ 90)",
                severity="BLOCK",
                override_score="HIGH",
            ),
        },
        {
            "id": "UCP_PRESENTATION_CRITICAL",
            "check": lambda f: f.get("presentation_days", 21) < 3,
            "violation": UCPViolation(
                article="UCP 600 Art.14(c)",
                field="48",
                description="Délai de présentation < 3 jours — Art.14(c) non respecté",
                severity="BLOCK",
                override_score="HIGH",
            ),
        },
        # ── Règles HIGH ───────────────────────────────────────────────────────
        {
            "id": "UCP_14_COMPLIANCE",
            "check": lambda f: f.get("ucp_compliance_score", 1.0) < 0.30,
            "violation": UCPViolation(
                article="UCP 600 Art.14",
                field="47A / Multiple",
                description="Score compliance critique (< 0.30) — discordance documentaire probable",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "UCP_16_ERRORS",
            "check": lambda f: f.get("ucp_errors_count", 0) >= 3,
            "violation": UCPViolation(
                article="UCP 600 Art.16",
                field="47A",
                description="Erreurs UCP ≥ 3 — discrepancies critiques, paiement susceptible d'être refusé",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "UCP_16_WARNINGS",
            "check": lambda f: f.get("n_ucp_warnings", 0) >= 5,
            "violation": UCPViolation(
                article="UCP 600 Art.16",
                field="47A",
                description="Avertissements UCP totaux ≥ 5 — conflits de clauses multiples",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "UCP_MISSING_CRITICAL",
            "check": lambda f: f.get("missing_critical_count", 0) >= 3,
            "violation": UCPViolation(
                article="UCP 600 Art.5 / Art.14",
                field="Multiple",
                description="≥ 3 champs critiques MT700 manquants — LC incomplet juridiquement",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "UCP_29_PRESENTATION",
            "check": lambda f: 3 <= f.get("presentation_days", 21) < 5,
            "violation": UCPViolation(
                article="UCP 600 Art.29",
                field="48",
                description="Délai de présentation critique (< 5 jours) — risque non-conformité Art.14(c)",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "UCP_CONTRADICTION",
            "check": lambda f: f.get("contradiction_flag", 0) >= 1,
            "violation": UCPViolation(
                article="UCP 600 Art.14(d)",
                field="47A / Multiple",
                description="Clauses contradictoires détectées — incohérence documentaire interne",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "RED_CLAUSE_DETECTED",
            "check": lambda f: f.get("red_clause_flag", 0) >= 1,
            "violation": UCPViolation(
                article="UCP 600 / ISBP 821",
                field="47A",
                description="Clause rouge (red clause) détectée — paiement avant expédition, risque élevé",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        {
            "id": "VALIDITY_CRITICAL",
            "check": lambda f: (
                f.get("validity_days", 90) < 15
                or f.get("validity_days", 90) > 548
            ),
            "violation": UCPViolation(
                article="UCP 600 Art.6 / Art.29",
                field="31D / 44C",
                description="Durée validité anormale (< 15 jours ou > 18 mois) — risque opérationnel",
                severity="HIGH",
                override_score="HIGH",
            ),
        },
        # ── Règles WARN ───────────────────────────────────────────────────────
        {
            "id": "COMPLETENESS_LOW",
            "check": lambda f: f.get("completeness_score", 1.0) < 0.50,
            "violation": UCPViolation(
                article="UCP 600 Art.5",
                field="Multiple",
                description="Complétude MT700 < 50% — champs importants manquants",
                severity="WARN",
            ),
        },
        {
            "id": "UCP_HIGH_TENOR",
            "check": lambda f: f.get("tenor_risk", 0.5) >= 0.80,
            "violation": UCPViolation(
                article="UCP 600 Art.42",
                field="42C",
                description="Tenor ≥ 180 jours — risque de crédit à terme élevé",
                severity="WARN",
            ),
        },
    ]

    def evaluate(
        self, features: dict
    ) -> tuple[list[UCPViolation], Optional[str]]:
        """
        Retourne (violations, forced_label).

        forced_label n'est positionné QUE par les violations de sévérité BLOCK
        (sanctions OFAC, REJECT UCP contractuel, présentation < 3j) : ce sont des
        conditions contractuelles authoritatives qui priment toujours sur le modèle.

        Les violations HIGH (signaux heuristiques dérivés du NLP : contradictions,
        warnings, complétude…) NE forcent PAS le label ici. Leur escalade est
        décidée en aval dans score(), de façon bornée et tenant compte de la
        confiance du modèle — pour éviter qu'une seule règle heuristique classe
        HIGH un LC que le modèle juge LOW à 94 % et que le validateur UCP accepte.
        """
        violations:  list[UCPViolation] = []
        block_label: Optional[str]      = None

        for rule in self.RULES:
            if rule["check"](features):
                v = rule["violation"]
                violations.append(v)
                if v.severity == "BLOCK" and v.override_score:
                    block_label = v.override_score

        return violations, block_label


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE EXPLAINER — enrichi pour 26 features
# ─────────────────────────────────────────────────────────────────────────────

class NarrativeExplainer:
    """Explications en langage naturel enrichies pour prédictions MT700 v4.0."""

    _DEFAULT_FEATURE_LABELS = {
        "lc_amount":               "LC face value (USD)",
        "country_risk_score":      "country risk score (consolidated 0-100)",
        "external_currency_risk":  "currency risk level (0-4)",
        "incoterm_risk":           "incoterm risk (DDP=1 … EXW=5)",
        "ucp_compliance_score":    "UCP 600 compliance score (0-1)",
        "n_ucp_warnings":          "total UCP violations (errors + warnings)",
        "ucp_warning_rate":        "UCP warning density (per clause)",
        "ucp_errors_count":        "UCP errors count (critical violations)",
        "ucp_warnings_count":      "UCP warnings count (advisory violations)",
        "nlp_score":               "clause quality score (NLP 0-100)",
        "semantic_score":          "semantic coherence (0-1)",
        "completeness_score":      "MT700 overall completeness (0-1)",
        "missing_critical_score":  "critical fields present rate (0-1)",
        "coherence_dates":         "date sequence coherence (0-1)",
        "contradiction_flag":      "contradictory clauses detected (0/1)",
        "country_risk_applicant":  "applicant country risk (0-100)",
        "country_risk_beneficiary":"beneficiary country risk (0-100)",
        "days_to_shipment":        "days until latest shipment",
        "presentation_days":       "document presentation period (days)",
        "validity_days":           "LC validity duration (days)",
        "amount_tier":             "LC amount tier (1=<50k … 5=>5M USD)",
        "transshipment_allowed":   "transshipment authorized (0/1)",
        "charges_beneficiary":     "charges on beneficiary (0/1)",
        "red_clause_flag":         "red clause detected in 47A (0/1)",
        "bank_tier":               "advising/confirming bank tier (1=best … 5=unknown)",
        "tenor_risk":              "tenor risk (SIGHT=0.1 … 360d=0.95)",
    }

    _DEFAULT_TEMPLATES = {
        "HIGH": (
            "This Letter of Credit is classified as HIGH RISK. "
            "Primary risk drivers: {drivers}. "
            "{rule_note}"
            "{missing_note}"
            "Immediate review by a senior documentary credit officer is required."
        ),
        "MEDIUM": (
            "This Letter of Credit is classified as MEDIUM RISK. "
            "Contributing factors: {drivers}. "
            "{rule_note}"
            "{missing_note}"
            "Documentary review applies with attention to identified risk factors."
        ),
        "LOW": (
            "This Letter of Credit is classified as LOW RISK. "
            "Key positive indicators: {drivers}. "
            "{rule_note}"
            "Standard processing applies."
        ),
    }

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._feature_labels = cfg.get("feature_labels", self._DEFAULT_FEATURE_LABELS)
        self._templates      = cfg.get("templates",       self._DEFAULT_TEMPLATES)

    def explain(
        self,
        predicted_label:     str,
        confidence:          float,
        shap_row:            dict,
        violations:          list[UCPViolation],
        features:            dict,
        missing_fields:      Optional[list] = None,
    ) -> str:
        top = sorted(shap_row.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        driver_parts = []
        for feat, val in top:
            direction = "elevated" if val > 0 else "reduced"
            label     = self._feature_labels.get(feat, feat)
            actual    = features.get(feat, 0.0)
            try:
                driver_parts.append(
                    f"{label} ({float(actual):.2f}) — {direction} risk contribution"
                )
            except (ValueError, TypeError):
                driver_parts.append(f"{label} — {direction} risk contribution")

        drivers = "; ".join(driver_parts) if driver_parts else "N/A"

        rule_note = ""
        if violations:
            block_rules = [v for v in violations if v.severity == "BLOCK"]
            high_rules  = [v for v in violations if v.severity == "HIGH"]
            if block_rules:
                articles = ", ".join(sorted({v.article for v in block_rules}))
                rule_note = f"BLOCKING UCP violations ({articles}). "
            elif high_rules:
                articles = ", ".join(sorted({v.article for v in high_rules}))
                rule_note = f"UCP HIGH violations ({articles}). "

        missing_note = ""
        if missing_fields:
            critical_missing = [m for m in missing_fields if m.get("criticality") == "CRITICAL"]
            if critical_missing:
                fields_str = ", ".join(m["field"] for m in critical_missing[:4])
                missing_note = f"Missing critical MT700 fields: {fields_str}. "

        template  = self._templates.get(predicted_label, self._templates["MEDIUM"])
        narrative = template.format(
            drivers=drivers,
            rule_note=rule_note,
            missing_note=missing_note,
        )
        return f"[Confidence: {confidence * 100:.1f}%] {narrative}"


# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskMLResult:
    risk_label:          str
    risk_proba:          dict
    risk_score:          int
    is_anomaly:          bool
    anomaly_score:       float
    segment:             str
    segment_id:          int
    shap_top_features:   list
    ucp_violations:      list
    narrative:           str
    features_used:       dict
    model_version:       str
    # v4.0 additions
    score_breakdown:     dict                = field(default_factory=dict)
    missing_fields_detail: list             = field(default_factory=list)
    explanation:         dict               = field(default_factory=dict)
    ucp_detail:          dict               = field(default_factory=dict)
    # existants
    challenger_label:    Optional[str]      = None
    processing_ms:       int                = 0
    error:               Optional[str]      = None

    def to_dict(self) -> dict:
        return {
            "risk_label":            self.risk_label,
            "risk_proba":            self.risk_proba,
            "risk_score":            self.risk_score,
            "is_anomaly":            self.is_anomaly,
            "anomaly_score":         round(self.anomaly_score, 4),
            "segment":               self.segment,
            "segment_id":            self.segment_id,
            "shap_top_features":     self.shap_top_features,
            "ucp_violations":        self.ucp_violations,
            "ucp_detail":            self.ucp_detail,
            "narrative":             self.narrative,
            "features_used":         self.features_used,
            "model_version":         self.model_version,
            "score_breakdown":       self.score_breakdown,
            "missing_fields_detail": self.missing_fields_detail,
            "explanation":           self.explanation,
            "challenger_label":      self.challenger_label,
            "processing_ms":         self.processing_ms,
            "error":                 self.error,
        }

    @classmethod
    def error_result(cls, msg: str) -> "RiskMLResult":
        return cls(
            risk_label="UNKNOWN", risk_proba={}, risk_score=-1,
            is_anomaly=False, anomaly_score=0.0,
            segment="UNKNOWN", segment_id=-1,
            shap_top_features=[], ucp_violations=[],
            narrative="", features_used={},
            model_version="error", error=msg,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR — v4.0 (26 features)
# ─────────────────────────────────────────────────────────────────────────────

class RiskFeatureExtractor:
    """
    Extrait les 26 features ML depuis le payload du pipeline SWIFT.

    Sources prioritaires :
      1. external_validation  (OFAC/EU — source de vérité pour country_risk)
      2. ds_analysis.features (LCFeatureExtractor)
      3. ucp_compliance        (rapport UCP 600)
      4. swift.nlp / swift.entities
      5. swift.fields          (champs bruts MT700)
      6. FEATURE_DEFAULTS      (valeurs neutres/prudentes)

    Clés internes (non-features ML) :
      _missing_fields      : liste des champs MT700 manquants
      n_clauses            : dénominateur ucp_warning_rate
      ucp_decision_reject  : flag pour UCPRuleEngine
      missing_critical_count : count pour règle UCP_MISSING_CRITICAL
    """

    _FEATURE_RANGES: dict[str, tuple] = {
        # Existantes v3.0
        "lc_amount":               (100.0,        50_000_000.0),
        "country_risk_score":      (0.0,          100.0),
        "external_currency_risk":  (0.0,          4.0),
        "incoterm_risk":           (1.0,          5.0),
        "ucp_compliance_score":    (0.0,          1.0),
        "n_ucp_warnings":          (0.0,          50.0),
        "ucp_warning_rate":        (0.0,          10.0),
        "nlp_score":               (0.0,          100.0),
        "semantic_score":          (0.0,          1.0),
        "completeness_score":      (0.0,          1.0),
        "days_to_shipment":        (0.0,          730.0),
        "presentation_days":       (1.0,          180.0),
        "validity_days":           (0.0,          730.0),
        "transshipment_allowed":   (0.0,          1.0),
        "charges_beneficiary":     (0.0,          1.0),
        # Nouvelles v4.0
        "ucp_errors_count":        (0.0,          50.0),
        "ucp_warnings_count":      (0.0,          50.0),
        "missing_critical_score":  (0.0,          1.0),
        "coherence_dates":         (0.0,          1.0),
        "contradiction_flag":      (0.0,          1.0),
        "country_risk_applicant":  (0.0,          100.0),
        "country_risk_beneficiary":(0.0,          100.0),
        "amount_tier":             (1.0,          5.0),
        "red_clause_flag":         (0.0,          1.0),
        "bank_tier":               (1.0,          5.0),
        "tenor_risk":              (0.0,          1.0),
    }

    _EXT_RISK_LEVEL_MAP: dict[str, int] = {
        "CLEAN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4,
    }

    @staticmethod
    def _normalize_country(raw) -> str:
        """Renvoie un ISO2 depuis un code ISO2 ou un nom de pays complet."""
        if not raw or not isinstance(raw, str):
            return ""
        s = raw.strip().upper()
        if not s:
            return ""
        # Nom complet connu → ISO2
        if s in _COUNTRY_NAME_TO_ISO2:
            return _COUNTRY_NAME_TO_ISO2[s]
        # Déjà un code ISO2
        if len(s) == 2 and s.isalpha():
            return s
        # Nom présent en sous-chaîne (ex. "HAMBURG GERMANY")
        for name, iso in _COUNTRY_NAME_TO_ISO2.items():
            if re.search(rf"\b{re.escape(name)}\b", s):
                return iso
        # Dernier recours : préfixe 2 lettres (ancien comportement)
        if len(s) >= 2 and s[:2].isalpha():
            return s[:2]
        return ""

    def _iso_to_risk(self, raw: str) -> float:
        iso2 = self._normalize_country(raw)
        if not iso2 or len(iso2) < 2 or not iso2.isalpha():
            return FEATURE_DEFAULTS["country_risk_score"]
        if iso2 in SANCTIONED_ISO2:
            return 100.0
        tier = _COUNTRY_RISK_TIER.get(iso2, 1)
        return _TIER_TO_SCORE.get(tier, 25.0)

    def _parse_date(self, d) -> Optional[datetime]:
        if not d:
            return None
        s = str(d).strip()
        for fmt in ["%Y%m%d", "%d%m%y", "%y%m%d", "%Y-%m-%d", "%d/%m/%Y",
                    "%d-%m-%Y", "%m/%d/%Y", "%d.%m.%Y"]:
            try:
                return datetime.strptime(s[:8] if "%" not in fmt else s, fmt)
            except ValueError:
                pass
        return None

    def _check_field_present(
        self,
        keys: list[str],
        entities: dict,
        lc: dict,
        raw_fields: dict,
        field_id: str,
    ) -> bool:
        for k in keys:
            v = entities.get(k) or lc.get(k)
            if v not in (None, "", 0, "0", "N/A", "UNKNOWN"):
                return True
        if raw_fields.get(field_id) not in (None, "", 0):
            return True
        return False

    def _extract_missing_fields(
        self,
        entities: dict,
        lc: dict,
        raw_fields: dict,
    ) -> tuple[float, float, int, list]:
        """
        Returns:
            missing_critical_score (0-1, fraction CRITICAL présents — 1=tous présents)
            missing_critical_count (int)
            missing_fields_detail  (list of dicts)
        """
        detail: list[dict] = []
        critical_present = 0

        for fid, meta in MT700_CRITICAL_FIELDS.items():
            present = self._check_field_present(meta["keys"], entities, lc, raw_fields, fid)
            if present:
                critical_present += 1
            else:
                detail.append({
                    "field":       fid,
                    "name":        meta["name"],
                    "criticality": "CRITICAL",
                    "impact_score": meta["impact"],
                })

        for fid, meta in MT700_HIGH_FIELDS.items():
            present = self._check_field_present(meta["keys"], entities, lc, raw_fields, fid)
            if not present:
                detail.append({
                    "field":       fid,
                    "name":        meta["name"],
                    "criticality": "HIGH",
                    "impact_score": meta["impact"],
                })

        for fid, meta in MT700_OPTIONAL_FIELDS.items():
            present = self._check_field_present(meta["keys"], entities, lc, raw_fields, fid)
            if not present:
                detail.append({
                    "field":       fid,
                    "name":        meta["name"],
                    "criticality": "OPTIONAL",
                    "impact_score": meta["impact"],
                })

        n_critical = len(MT700_CRITICAL_FIELDS)
        missing_count = n_critical - critical_present
        missing_score = float(critical_present) / float(n_critical)

        return missing_score, float(missing_count), missing_count, detail

    def extract(self, pipeline_payload: dict) -> dict:
        features: dict[str, float] = dict(FEATURE_DEFAULTS)

        swift      = pipeline_payload.get("swift", {}) or {}
        entities   = swift.get("entities", {}) or {}
        nlp        = swift.get("nlp", {}) or {}
        sem        = nlp.get("semantic", {}) or {}
        lc         = swift.get("lc_criteria", {}) or {}
        raw_fields = swift.get("fields", {}) or {}
        ucp        = pipeline_payload.get("ucp_compliance", {}) or {}
        ds_feats   = (pipeline_payload.get("ds_analysis", {}) or {}).get("features", {}) or {}
        ext        = pipeline_payload.get("external_validation", {}) or {}

        # ── lc_amount ─────────────────────────────────────────────────────────
        for src in [ds_feats, entities, lc]:
            for key in ["lc_amount", "amount"]:
                v = src.get(key)
                if v is not None:
                    try:
                        features["lc_amount"] = float(str(v).replace(",", "").replace(" ", ""))
                        break
                    except (ValueError, TypeError):
                        pass
            else:
                continue
            break

        # ── country_risk_score consolidé ──────────────────────────────────────
        # Source 1 : lookup interne (accepte noms complets ou ISO2)
        internal_score = 0.0
        seen_any = False
        for fld in ["country_beneficiary", "country_applicant", "country", "beneficiary"]:
            iso_raw = entities.get(fld, "")
            iso2 = self._normalize_country(iso_raw)
            if not iso2:
                continue
            seen_any = True
            sc = self._iso_to_risk(iso2)
            if sc > internal_score:
                internal_score = sc

        if not seen_any:
            internal_score = FEATURE_DEFAULTS["country_risk_score"]

        # Source 2 : DS analysis
        ds_score: Optional[float] = None
        cr_legacy = ds_feats.get("country_risk")
        if cr_legacy is not None:
            try:
                ds_score = min(float(cr_legacy) * 33.3, 100.0)
            except (TypeError, ValueError):
                pass

        # Source 3 : external_validation
        # IMPORTANT : ne PAS forcer le score à 0 sur un CLEAN issu d'un STUB
        # (moteur externe indisponible — ex. pycountry manquant). Sinon tout LC
        # se retrouve à country_risk_score=0 alors que le DS voit un risque pays.
        # CLEAN réel (ext_score=0) → max(internal, ds, 0) = on garde le risque
        #   pays de base ; un vrai hit sanctions (HIGH/CRITICAL) escalade via max.
        ext_rl    = str(ext.get("risk_level") or "").upper()
        ext_score = _EXT_RL_TO_SCORE.get(ext_rl)
        ext_is_stub = bool(ext.get("_stub")) or "unavailable" in str(ext.get("approach", "")).lower()

        if ext_score is not None and not ext_is_stub:
            features["country_risk_score"] = max(internal_score, ds_score or 0.0, ext_score)
        else:
            features["country_risk_score"] = max(internal_score, ds_score or 0.0)

        # ── country_risk_applicant / country_risk_beneficiary ─────────────────
        # Repli sur le pays consolidé du document si le pays par partie est absent,
        # pour éviter le défaut figé à 25 quand le pays réel est connu.
        country_fallback = entities.get("country") or ""
        iso_app = (entities.get("country_applicant") or entities.get("applicant_country")
                   or country_fallback)
        iso_ben = (entities.get("country_beneficiary") or entities.get("beneficiary_country")
                   or country_fallback)

        features["country_risk_applicant"]   = self._iso_to_risk(iso_app)
        features["country_risk_beneficiary"] = self._iso_to_risk(iso_ben)

        # ── external_currency_risk ────────────────────────────────────────────
        ext_details  = ext.get("details") or {}
        ext_currency = ext_details.get("currency") or {}
        if isinstance(ext_currency, dict) and ext_currency:
            cr  = str(ext_currency.get("risk") or "CLEAN").upper()
            features["external_currency_risk"] = float(self._EXT_RISK_LEVEL_MAP.get(cr, 0))

        # ── incoterm_risk ─────────────────────────────────────────────────────
        inc_ds = ds_feats.get("incoterm_risk")
        if inc_ds is not None:
            try:
                features["incoterm_risk"] = float(inc_ds)
            except (TypeError, ValueError):
                pass
        else:
            incoterm = str(lc.get("incoterm") or entities.get("incoterm") or "")
            if incoterm:
                inc_key = re.sub(r"[^A-Z]", "", incoterm.upper())[:3]
                features["incoterm_risk"] = float(INCOTERM_RISK_MAP.get(inc_key, INCOTERM_RISK_DEFAULT))

        # ── ucp_compliance_score ──────────────────────────────────────────────
        ucp_score = ucp.get("global_score")
        if ucp_score is not None:
            try:
                features["ucp_compliance_score"] = float(ucp_score) / 100.0
            except (ValueError, TypeError):
                pass

        # ── UCP errors / warnings séparés ─────────────────────────────────────
        summary = ucp.get("summary") if isinstance(ucp.get("summary"), dict) else {}
        errors_count   = int(summary.get("errors", 0))
        warnings_count = int(summary.get("warnings", 0))

        # Fallback : compter depuis les listes
        if errors_count == 0 and warnings_count == 0:
            errors_count   = len(ucp.get("errors",   []))
            warnings_count = len(ucp.get("warnings", []))

        features["ucp_errors_count"]   = float(errors_count)
        features["ucp_warnings_count"] = float(warnings_count)

        # n_ucp_warnings = total (backward compat ML v3.0)
        n_warn = errors_count + warnings_count
        if n_warn > 0:
            features["n_ucp_warnings"] = float(n_warn)

        # ── ucp_decision_reject — rule engine uniquement ───────────────────────
        if str(ucp.get("ucp_decision", "")).upper() == "REJECT":
            features["ucp_decision_reject"] = 1.0

        # ── n_clauses interne (dénominateur ucp_warning_rate) ─────────────────
        n_clauses_ucp = ucp.get("total_clauses_checked") or ucp.get("n_clauses")
        if n_clauses_ucp is not None:
            try:
                n_clauses = max(1, int(n_clauses_ucp))
            except (ValueError, TypeError):
                n_clauses = max(1, len(nlp.get("clauses", []))) or 10
        else:
            n_clauses = max(1, len(nlp.get("clauses", []))) or 10

        features["ucp_warning_rate"] = float(features["n_ucp_warnings"]) / float(n_clauses)

        # ── nlp_score ─────────────────────────────────────────────────────────
        nlp_score = nlp.get("score")
        if nlp_score is not None:
            try:
                features["nlp_score"] = float(nlp_score)
            except (ValueError, TypeError):
                pass

        # ── semantic_score ────────────────────────────────────────────────────
        sem_score = sem.get("semantic_score") or sem.get("coherence_score") or sem.get("doc_quality_score")
        if sem_score is not None:
            try:
                sv = float(sem_score)
                features["semantic_score"] = sv / 100.0 if sv > 1.0 else sv
            except (ValueError, TypeError):
                pass

        # ── completeness_score ────────────────────────────────────────────────
        comp = (
            sem.get("completeness_score")
            or (swift.get("validation") or {}).get("completeness_score")
            or ds_feats.get("completeness_score")
        )
        if comp is not None:
            try:
                cv = float(comp)
                features["completeness_score"] = cv / 100.0 if cv > 1.0 else cv
            except (ValueError, TypeError):
                pass

        # ── missing_critical_score — champs MT700 manquants ───────────────────
        missing_score, missing_count_f, missing_count_i, missing_detail = (
            self._extract_missing_fields(entities, lc, raw_fields)
        )
        features["missing_critical_score"]  = missing_score
        features["missing_critical_count"]  = missing_count_f  # interne pour règles
        features["_missing_fields"]         = missing_detail    # interne pour reporting

        # ── coherence_dates ───────────────────────────────────────────────────
        issue_date   = entities.get("issue_date")   or lc.get("issue_date")
        shipment_dt  = entities.get("latest_shipment_date") or lc.get("latest_shipment_date") or lc.get("shipment_date")
        expiry_date  = entities.get("expiry_date")  or lc.get("expiry_date") or lc.get("validity_date")

        d_issue  = self._parse_date(issue_date)
        d_expiry = self._parse_date(expiry_date)
        d_ship   = self._parse_date(shipment_dt)

        if d_issue and d_expiry:
            seq_ok = d_expiry > d_issue
            if d_ship:
                seq_ok = seq_ok and d_ship <= d_expiry
            features["coherence_dates"] = 1.0 if seq_ok else 0.0
        else:
            features["coherence_dates"] = 0.5

        # ── contradiction_flag ────────────────────────────────────────────────
        contradictions = (
            sem.get("contradictions", [])
            or nlp.get("contradictions", [])
            or sem.get("conflicts", [])
            or nlp.get("conflicts", [])
        )
        features["contradiction_flag"] = 1.0 if (len(contradictions) > 0 or features["coherence_dates"] == 0.0) else 0.0

        # ── days_to_shipment ──────────────────────────────────────────────────
        for src, key in [(ds_feats, "days_to_shipment"), (lc, "days_to_shipment")]:
            v = src.get(key)
            if v is not None:
                try:
                    features["days_to_shipment"] = max(0.0, float(v))
                    break
                except (ValueError, TypeError):
                    pass

        # ── presentation_days ─────────────────────────────────────────────────
        for src, key in [
            (lc,       "presentation_period"),
            (lc,       "presentation_period_days"),
            (entities, "presentation_period"),
            (ds_feats, "presentation_days"),
        ]:
            v = src.get(key)
            if v is not None:
                try:
                    features["presentation_days"] = max(1.0, float(v))
                    break
                except (ValueError, TypeError):
                    pass

        # ── validity_days ─────────────────────────────────────────────────────
        found_validity = False
        for src, key in [(ds_feats, "validity_days"), (lc, "validity_days")]:
            v = src.get(key)
            if v is not None:
                try:
                    features["validity_days"] = max(0.0, float(v))
                    found_validity = True
                    break
                except (ValueError, TypeError):
                    pass
        if not found_validity and d_issue and d_expiry:
            try:
                features["validity_days"] = max(0.0, float((d_expiry - d_issue).days))
            except Exception:
                pass

        # ── amount_tier ───────────────────────────────────────────────────────
        features["amount_tier"] = float(compute_amount_tier(features["lc_amount"]))

        # ── transshipment_allowed ─────────────────────────────────────────────
        ts = lc.get("transshipment") or entities.get("transshipment")
        if ts is not None:
            features["transshipment_allowed"] = (
                1.0 if str(ts).upper() in ("YES", "ALLOWED", "TRUE", "1") else 0.0
            )
        elif ds_feats.get("transshipment_allowed") is not None:
            try:
                features["transshipment_allowed"] = float(ds_feats["transshipment_allowed"])
            except (ValueError, TypeError):
                pass

        # ── charges_beneficiary ───────────────────────────────────────────────
        chg = lc.get("charges") or entities.get("charges")
        if chg is not None:
            features["charges_beneficiary"] = (
                1.0 if str(chg).upper() in ("BENEFICIARY", "OUR", "TRUE", "1") else 0.0
            )
        elif ds_feats.get("charges_beneficiary") is not None:
            try:
                features["charges_beneficiary"] = float(ds_feats["charges_beneficiary"])
            except (ValueError, TypeError):
                pass

        # ── red_clause_flag ───────────────────────────────────────────────────
        field_47a = str(
            lc.get("additional_conditions")
            or raw_fields.get("47A")
            or nlp.get("field_47a")
            or entities.get("additional_conditions")
            or ""
        ).lower()
        features["red_clause_flag"] = 1.0 if any(p in field_47a for p in _RED_CLAUSE_PATTERNS) else 0.0

        # ── bank_tier ─────────────────────────────────────────────────────────
        # On évalue toutes les contreparties bancaires disponibles et on retient
        # le MEILLEUR tier (chiffre le plus bas). Inclut issuing_bank : souvent
        # le seul champ portant un nom complet (ex. "DEUTSCHE BANK AG"), alors que
        # :57A:/:41A: ne portent qu'un BIC. compute_bank_tier gère désormais le BIC.
        bank_candidates = [
            entities.get("advising_bank"),
            entities.get("confirming_bank"),
            entities.get("issuing_bank"),
            entities.get("reimbursement_bank"),
            raw_fields.get("57A"),
            raw_fields.get("58A"),
            raw_fields.get("52A"),
            lc.get("advising_bank"),
        ]
        tiers = [
            compute_bank_tier(str(b))
            for b in bank_candidates
            if b and str(b).strip()
        ]
        # Tier 4 = "non identifié" : ne le retenir que si aucune banque n'est classée.
        classified = [t for t in tiers if t < 4]
        features["bank_tier"] = float(min(classified) if classified else (min(tiers) if tiers else 4))

        # ── tenor_risk ────────────────────────────────────────────────────────
        tenor_text = str(
            entities.get("tenor")
            or entities.get("drafts_at")
            or lc.get("drafts_at")
            or lc.get("tenor")
            or raw_fields.get("42C")
            or ""
        )
        features["tenor_risk"] = compute_tenor_risk(tenor_text)

        # ── Validation et clip ────────────────────────────────────────────────
        features = self._validate_features(features)
        return features

    def _validate_features(self, features: dict) -> dict:
        issues = []
        for col, (lo, hi) in self._FEATURE_RANGES.items():
            val = features.get(col)
            if val is None:
                continue
            try:
                fval    = float(val)
                clipped = max(float(lo), min(float(hi), fval))
                if abs(clipped - fval) > 1e-9:
                    issues.append(f"{col}={fval:.4g}→[{lo},{hi}]")
                    features[col] = clipped
            except (TypeError, ValueError):
                pass
        if issues:
            logger.warning("Feature validation clips: %s", "; ".join(issues))
            features["_validation_issues"] = issues
        return features


# ─────────────────────────────────────────────────────────────────────────────
# SCORE BREAKDOWN COMPUTER — 7 dimensions
# ─────────────────────────────────────────────────────────────────────────────

# Poids v4.1 — recalibrés selon importance réelle en trade finance UCP 600
# UCP = critère n°1, complétude et cohérence = n°2, pays = n°3
# Temporel et montant ont un rôle opérationnel, pas un rôle de risque dominant
_BREAKDOWN_WEIGHTS = {
    "completeness":    0.18,   # champs MT700 manquants → invalidité juridique
    "ucp_compliance":  0.30,   # premier critère UCP 600 Art.14 (was 0.25)
    "coherence":       0.18,   # cohérence inter-clauses → discrepancy risk
    "country_risk":    0.18,   # sanctions / OFAC (légèrement réduit, not dominant seul)
    "temporal":        0.06,   # paramètre opérationnel — 43 jours ≠ risque grave (was 0.10)
    "amount":          0.06,   # montant seul n'est pas un indicateur de risque (was 0.10)
    "ml_model":        0.04,   # challenger uniquement (was 0.05)
}


def compute_score_breakdown(
    features:    dict,
    risk_score:  int,
    risk_proba:  dict,
    risk_label:  str,
    violations:  list,
) -> dict:
    """
    Calcule un score_breakdown en 7 dimensions depuis les 26 features v4.0.
    Chaque dimension a un score 0-100 (100 = risque maximal).
    """

    def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return max(lo, min(hi, x))

    # ── 1. Completeness dimension (weight 0.15) ───────────────────────────────
    # missing_critical_score = 1 → tous présents → risque 0
    # missing_critical_score = 0 → tous absents → risque 100
    miss_crit   = features.get("missing_critical_score", 0.75)
    comp_global = features.get("completeness_score", 0.5)
    miss_count  = int(features.get("missing_critical_count", 0))
    completeness_risk = _clamp((1.0 - miss_crit) * 70.0 + (1.0 - comp_global) * 30.0)

    # ── 2. UCP compliance dimension (weight 0.25) ─────────────────────────────
    ucp_score    = features.get("ucp_compliance_score", 0.5)
    ucp_errors   = features.get("ucp_errors_count", 1.0)
    ucp_warns    = features.get("ucp_warnings_count", 1.0)
    # Weighted violation index : errors poids 1.0, warnings poids 0.4
    ucp_weighted_violations = ucp_errors * 1.0 + ucp_warns * 0.4
    # Score de risque : base compliance + bonus violations
    ucp_risk = _clamp(
        (1.0 - ucp_score) * 60.0
        + min(ucp_errors * 8.0, 25.0)
        + min(ucp_warns * 2.0, 15.0)
    )

    # ── 3. Coherence dimension (weight 0.15) ──────────────────────────────────
    semantic_ok    = features.get("semantic_score", 0.5)
    dates_ok       = features.get("coherence_dates", 0.5)
    contradiction  = features.get("contradiction_flag", 0.0)
    coherence_risk = _clamp(
        (1.0 - semantic_ok) * 40.0
        + (1.0 - dates_ok) * 35.0
        + contradiction * 25.0
    )

    # ── 4. Country risk dimension (weight 0.20) ───────────────────────────────
    app_risk = features.get("country_risk_applicant",   25.0)
    ben_risk = features.get("country_risk_beneficiary", 25.0)
    # Risque dual : max des deux (principe du maillon faible)
    country_risk = _clamp(max(app_risk, ben_risk))

    # ── 5. Temporal dimension (weight 0.10) ───────────────────────────────────
    validity   = features.get("validity_days", 90.0)
    days_ship  = features.get("days_to_shipment", 30.0)
    pres_days  = features.get("presentation_days", 21.0)

    # validity_days — plage normale (30–365j) = risque très faible (3–8)
    # Un LC de 43 jours est parfaitement standard en commerce international
    if validity < 15:
        validity_risk = 100.0  # critique : impossible à traiter
    elif validity < 30:
        validity_risk = 65.0   # très serré
    elif validity <= 90:
        validity_risk = 5.0    # plage normale courte — risque minimal
    elif validity <= 180:
        validity_risk = 3.0    # standard — risque quasi nul
    elif validity <= 365:
        validity_risk = 6.0    # légèrement plus long, toujours normal
    elif validity <= 548:
        validity_risk = 25.0   # LC long — exposition accrue mais acceptable
    else:
        validity_risk = min(70.0, 25.0 + (validity - 548) / 10.0)  # anormal

    # days_to_shipment — délai ≥ 21 jours = plage normale
    if days_ship < 3:
        shipment_risk = 100.0
    elif days_ship < 10:
        shipment_risk = 65.0
    elif days_ship < 21:
        shipment_risk = 30.0
    elif days_ship <= 60:
        shipment_risk = 8.0    # délai opérationnel normal
    else:
        shipment_risk = 4.0    # délai confortable

    # presentation_days — standard UCP 600 Art.14(c) = 21 jours
    if pres_days < 3:
        pres_risk = 100.0
    elif pres_days < 5:
        pres_risk = 75.0
    elif pres_days < 10:
        pres_risk = 45.0
    elif pres_days <= 30:
        pres_risk = 8.0    # plage normale (10–30j)
    elif pres_days <= 60:
        pres_risk = 5.0    # légèrement long mais acceptable
    else:
        pres_risk = 20.0   # très long — risque de présentation tardive

    temporal_risk = _clamp(validity_risk * 0.40 + shipment_risk * 0.35 + pres_risk * 0.25)

    # ── 6. Amount dimension (weight 0.10) ─────────────────────────────────────
    amount_tier = features.get("amount_tier", 3.0)
    red_clause  = features.get("red_clause_flag", 0.0)
    tenor_r     = features.get("tenor_risk", 0.5)
    bank_t      = features.get("bank_tier", 3.0)
    amount_risk = _clamp(
        (amount_tier - 1.0) / 4.0 * 50.0  # 0-50 selon tier
        + red_clause * 25.0                # +25 si clause rouge
        + tenor_r * 15.0                   # +0-15 selon tenor
        + (bank_t - 1.0) / 4.0 * 10.0     # +0-10 selon tier banque
    )

    # ── 7. ML model dimension (weight 0.05) ───────────────────────────────────
    ml_risk = float(_clamp(risk_score))

    # ── Assemblage ────────────────────────────────────────────────────────────
    scores = {
        "completeness":   completeness_risk,
        "ucp_compliance": ucp_risk,
        "coherence":      coherence_risk,
        "country_risk":   country_risk,
        "temporal":       temporal_risk,
        "amount":         amount_risk,
        "ml_model":       ml_risk,
    }

    weighted_total = sum(scores[k] * _BREAKDOWN_WEIGHTS[k] for k in scores)
    weighted_total = round(min(100.0, weighted_total), 1)

    # Niveau analytique basé sur le score pondéré
    if weighted_total >= 70:
        analytical_label = "HIGH"
    elif weighted_total >= 40:
        analytical_label = "MEDIUM"
    else:
        analytical_label = "LOW"

    return {
        "analytical_score":  weighted_total,
        "analytical_label":  analytical_label,
        "weights":           _BREAKDOWN_WEIGHTS,
        "dimensions": {
            "completeness": {
                "score":                   round(completeness_risk, 1),
                "weight":                  _BREAKDOWN_WEIGHTS["completeness"],
                "contribution":            round(completeness_risk * _BREAKDOWN_WEIGHTS["completeness"], 2),
                "missing_critical_fields": miss_count,
                "critical_fields_present_rate": round(miss_crit * 100, 1),
                "completeness_rate":       round(comp_global * 100, 1),
            },
            "ucp_compliance": {
                "score":                round(ucp_risk, 1),
                "weight":               _BREAKDOWN_WEIGHTS["ucp_compliance"],
                "contribution":         round(ucp_risk * _BREAKDOWN_WEIGHTS["ucp_compliance"], 2),
                "errors_count":         int(ucp_errors),
                "warnings_count":       int(ucp_warns),
                "weighted_violations":  round(ucp_weighted_violations, 2),
                "compliance_score_pct": round(ucp_score * 100, 1),
                "error_weight":         1.0,
                "warning_weight":       0.4,
            },
            "coherence": {
                "score":                round(coherence_risk, 1),
                "weight":               _BREAKDOWN_WEIGHTS["coherence"],
                "contribution":         round(coherence_risk * _BREAKDOWN_WEIGHTS["coherence"], 2),
                "date_coherence_pct":   round(dates_ok * 100, 1),
                "semantic_coherence_pct": round(semantic_ok * 100, 1),
                "contradiction_detected": bool(contradiction),
            },
            "country_risk": {
                "score":             round(country_risk, 1),
                "weight":            _BREAKDOWN_WEIGHTS["country_risk"],
                "contribution":      round(country_risk * _BREAKDOWN_WEIGHTS["country_risk"], 2),
                "applicant_risk":    round(app_risk, 1),
                "beneficiary_risk":  round(ben_risk, 1),
                "is_sanctioned":     country_risk >= 90,
                "dual_sanction":     bool(app_risk >= 90 or ben_risk >= 90),
            },
            "temporal": {
                "score":             round(temporal_risk, 1),
                "weight":            _BREAKDOWN_WEIGHTS["temporal"],
                "contribution":      round(temporal_risk * _BREAKDOWN_WEIGHTS["temporal"], 2),
                "validity_days":     int(validity),
                "days_to_shipment":  int(days_ship),
                "presentation_days": int(pres_days),
                "validity_risk":     round(validity_risk, 1),
                "shipment_risk":     round(shipment_risk, 1),
                "presentation_risk": round(pres_risk, 1),
            },
            "amount": {
                "score":         round(amount_risk, 1),
                "weight":        _BREAKDOWN_WEIGHTS["amount"],
                "contribution":  round(amount_risk * _BREAKDOWN_WEIGHTS["amount"], 2),
                "tier":          int(amount_tier),
                "lc_amount_usd": round(features.get("lc_amount", 500_000.0), 2),
                "red_clause":    bool(red_clause),
                "tenor_risk":    round(tenor_r, 3),
                "bank_tier":     int(bank_t),
            },
            "ml_model": {
                "score":        round(ml_risk, 1),
                "weight":       _BREAKDOWN_WEIGHTS["ml_model"],
                "contribution": round(ml_risk * _BREAKDOWN_WEIGHTS["ml_model"], 2),
                "label":        risk_label,
                "probabilities": risk_proba,
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION COMPUTER — SHAP waterfall + dimension radar
# ─────────────────────────────────────────────────────────────────────────────

def _business_direction(feature: str, value: Optional[float]) -> str:
    """
    Retourne la direction métier attendue pour une feature donnée.
    'reduces_risk'  : valeur actuelle est dans la plage basse-risque (bonne)
    'increases_risk': valeur actuelle est dans la plage haute-risque (problème réel)
    'neutral'       : paramètre opérationnel sans implication directe de risque

    Sert à détecter les discordances ML vs Métier :
    si SHAP dit "↑ risque" mais business_direction = "reduces_risk" → DISCORDANT.
    """
    if value is None:
        return "neutral"
    v = float(value)

    # Features où une VALEUR ÉLEVÉE = RISQUE ÉLEVÉ (monotone croissant)
    high_is_bad = {
        "country_risk_score", "country_risk_applicant", "country_risk_beneficiary",
        "external_currency_risk", "incoterm_risk",
        "ucp_errors_count", "ucp_warnings_count", "n_ucp_warnings", "ucp_warning_rate",
        "contradiction_flag", "red_clause_flag",
        "bank_tier", "amount_tier", "tenor_risk",
        "missing_critical_count",
    }
    # Features où une VALEUR ÉLEVÉE = RISQUE FAIBLE (monotone décroissant)
    high_is_good = {
        "ucp_compliance_score", "nlp_score", "semantic_score",
        "completeness_score", "missing_critical_score", "coherence_dates",
    }
    # Features binaires (0 = bien, 1 = mauvais)
    binary_bad_if_one = {"transshipment_allowed", "charges_beneficiary"}
    binary_bad_if_zero = set()  # aucune pour l'instant

    if feature in high_is_bad:
        if v == 0.0:
            return "reduces_risk"
        elif feature in ("country_risk_score", "country_risk_applicant", "country_risk_beneficiary"):
            return "increases_risk" if v >= 50 else ("neutral" if v >= 25 else "reduces_risk")
        elif feature == "external_currency_risk":
            return "reduces_risk" if v == 0 else ("neutral" if v <= 1 else "increases_risk")
        elif feature in ("ucp_errors_count", "ucp_warnings_count", "n_ucp_warnings"):
            return "reduces_risk" if v == 0 else ("neutral" if v <= 2 else "increases_risk")
        else:
            return "increases_risk" if v > 0 else "reduces_risk"

    elif feature in high_is_good:
        if feature == "ucp_compliance_score":
            return "reduces_risk" if v >= 0.85 else ("neutral" if v >= 0.60 else "increases_risk")
        elif feature in ("nlp_score",):
            return "reduces_risk" if v >= 75 else ("neutral" if v >= 50 else "increases_risk")
        else:
            return "reduces_risk" if v >= 0.75 else ("neutral" if v >= 0.50 else "increases_risk")

    elif feature in binary_bad_if_one:
        return "neutral" if v == 0 else "increases_risk"

    elif feature == "validity_days":
        if v < 15: return "increases_risk"
        elif v <= 365: return "reduces_risk"   # plage normale
        elif v <= 548: return "neutral"
        else: return "increases_risk"

    elif feature == "days_to_shipment":
        if v < 10: return "increases_risk"
        elif v >= 21: return "reduces_risk"
        else: return "neutral"

    elif feature == "presentation_days":
        if v < 5: return "increases_risk"
        elif v <= 45: return "reduces_risk"
        else: return "neutral"

    elif feature == "lc_amount":
        # Montant seul = neutre (n'est pas un indicateur de risque direct)
        return "neutral"

    return "neutral"


def compute_explanation(
    shap_top:       list,
    shap_row:       dict,
    features:       dict,
    score_breakdown: dict,
    violations:     list,
    risk_score:     int,
    final_label:    str,
) -> dict:
    """
    Construit l'explication SHAP waterfall + dimension radar.
    Enrichi avec business_direction et is_discordant pour chaque feature.

    RÈGLE D'INTERPRÉTATION DU SIGNE SHAP (CRITIQUE) :
      Les valeurs SHAP sont calculées pour la classe PRÉDITE.
      - Classe LOW  (0) : shap > 0 = pousse vers LOW = RÉDUIT le risque
      - Classe HIGH (2) : shap > 0 = pousse vers HIGH = AUGMENTE le risque
      - Classe MEDIUM(1): shap > 0 = pousse vers MEDIUM = ambigu

    is_discordant = True seulement si :
      - La classe prédite est HIGH/MEDIUM ET le signe SHAP dit "augmente risque" (pour HIGH)
      - MAIS la valeur réelle de la feature est dans la plage basse-risque métier
      Pour une prédiction LOW, un SHAP positif signifie "renforce LOW" = conforme métier.
    """
    # Sens du signe SHAP selon la classe prédite
    pred_label_upper = (final_label or "LOW").upper()
    def _ml_direction(sv: float) -> str:
        if pred_label_upper == "HIGH":
            return "increases_risk" if sv > 0 else "reduces_risk"
        elif pred_label_upper == "LOW":
            # positif = renforce LOW = réduit le risque
            return "reduces_risk" if sv > 0 else "increases_risk"
        else:  # MEDIUM — ambigu
            return "neutral"

    # 1. SHAP waterfall enrichi
    waterfall = []
    discordant_features: list[dict] = []

    for feat, shap_val in sorted(shap_row.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(shap_val) < 0.003:
            continue
        raw_val = features.get(feat)
        try:
            display_val = round(float(raw_val), 4) if raw_val is not None else None
        except (TypeError, ValueError):
            display_val = None

        ml_direction  = _ml_direction(shap_val)
        biz_direction = _business_direction(feat, display_val)
        # Discordant : ML dit risque augmente, valeur est métier-safe
        # ET la prédiction finale est HIGH ou MEDIUM (pas LOW).
        # Justification : si le modèle prédit LOW à 94%, les valeurs SHAP individuelles
        # contre-intuitives sont des artefacts mathématiques sans impact business.
        # On signale uniquement quand la PRÉDICTION est incorrecte du point de vue métier.
        is_discordant = (
            ml_direction == "increases_risk"
            and biz_direction == "reduces_risk"
            and pred_label_upper != "LOW"
        )

        entry = {
            "feature":            feat,
            "value":              display_val,
            "shap_value":         round(float(shap_val), 5),
            "contribution_pct":   round(float(shap_val) * 100, 2),
            "direction":          "up" if shap_val > 0 else "down",
            "business_direction": biz_direction,
            "is_discordant":      is_discordant,
        }
        waterfall.append(entry)
        if is_discordant:
            discordant_features.append({
                "feature":     feat,
                "value":       display_val,
                "shap_value":  round(float(shap_val), 5),
                "explanation": (
                    f"Le modele assigne un impact positif (risque) a '{feat}' = {display_val} "
                    f"alors que cette valeur est dans la plage basse-risque attendue (metier). "
                    f"Correlation non-causale probable depuis donnees synthetiques."
                ),
            })

    # 2. Dimension radar (0-100 par dimension depuis score_breakdown)
    dims = score_breakdown.get("dimensions", {})
    radar = {
        k: round(float(v.get("score", 50.0)), 1)
        for k, v in dims.items()
        if k != "ml_model"
    }

    # 3. Rule contributions
    rule_contributions = []
    for v in violations:
        rule_contributions.append({
            "rule":         getattr(v, "article", ""),
            "field":        getattr(v, "field", ""),
            "severity":     getattr(v, "severity", ""),
            "description":  getattr(v, "description", ""),
            "score_impact": _RULE_SEVERITY_CONTRIBUTION.get(getattr(v, "severity", ""), "+10"),
        })

    # 4. Score composition narrative
    weighted_total = score_breakdown.get("analytical_score", float(risk_score))
    top_dimensions = sorted(
        [(k, v.get("contribution", 0)) for k, v in dims.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # ── Waterfall analytique (style LIME) ────────────────────────────────────
    # Explication additive du score final : chaque dimension contribue de façon
    # transparente et causale. Contrairement au SHAP ML (corrélations opaques),
    # chaque barre de ce waterfall représente une règle métier explicite.
    #
    # Propriété LIME respectée : somme des contributions = analytical_score
    # Propriété locale : calculé pour ce LC spécifique, pas une moyenne globale
    dim_order     = ["completeness","ucp_compliance","coherence","country_risk","temporal","amount","ml_model"]
    dim_labels    = {
        "completeness":   "Champs MT700",
        "ucp_compliance": "Compliance UCP 600",
        "coherence":      "Cohérence documentaire",
        "country_risk":   "Risque pays",
        "temporal":       "Temporel",
        "amount":         "Montant & Tenor",
        "ml_model":       "Modèle ML (challenger)",
    }
    dim_colors = {
        "completeness":   "#3b82f6",
        "ucp_compliance": "#8b5cf6",
        "coherence":      "#06b6d4",
        "country_risk":   "#ef4444",
        "temporal":       "#f59e0b",
        "amount":         "#10b981",
        "ml_model":       "#6b7280",
    }

    waterfall_items = []
    cumulative = 0.0
    for dim_key in dim_order:
        dim_data = dims.get(dim_key, {})
        contrib  = round(float(dim_data.get("contribution", 0) or 0), 3)
        score_v  = round(float(dim_data.get("score", 0) or 0), 1)
        weight_v = round(float(dim_data.get("weight", 0) or 0), 3)
        start    = round(cumulative, 3)
        cumulative = round(cumulative + contrib, 3)

        waterfall_items.append({
            "dimension":    dim_key,
            "label":        dim_labels.get(dim_key, dim_key),
            "color":        dim_colors.get(dim_key, "#94a3b8"),
            "score":        score_v,
            "weight_pct":   round(weight_v * 100, 0),
            "contribution": contrib,
            "start":        start,
            "end":          cumulative,
            "risk_tone":    "high" if score_v >= 70 else ("medium" if score_v >= 40 else "low"),
        })

    # ── SHAP waterfall — réactivé, direction corrigée selon classe prédite ──
    # Pour une prédiction LOW : shap > 0 = renforce LOW = réduit le risque (vert)
    # Pour une prédiction HIGH: shap > 0 = renforce HIGH = augmente le risque (rouge)
    # Les discordances ne sont signalées que pour les prédictions HIGH/MEDIUM
    pred_label_upper = (final_label or "LOW").upper()

    def _ml_dir(sv: float) -> str:
        if pred_label_upper == "HIGH":
            return "up" if sv > 0 else "down"
        elif pred_label_upper == "LOW":
            return "down" if sv > 0 else "up"
        return "neutral"

    shap_enriched = []
    for item in waterfall:
        sv      = item["shap_value"]
        raw_val = item.get("value")
        ml_d    = _ml_dir(sv)
        biz_d   = _business_direction(item["feature"], raw_val)
        disc    = (ml_d == "up" and biz_d == "reduces_risk" and pred_label_upper != "LOW")
        shap_enriched.append({
            **item,
            "direction":          ml_d,
            "business_direction": biz_d,
            "is_discordant":      disc,
        })

    discordant_out = [
        {"feature": i["feature"], "value": i.get("value"), "shap_value": i["shap_value"]}
        for i in shap_enriched if i["is_discordant"]
    ]

    return {
        "method":           "analytical_waterfall_lime",
        "final_label":      final_label,
        "risk_score_ml":    risk_score,
        "analytical_score": weighted_total,
        # Waterfall analytique (LIME-style) — explication additive causale
        "analytical_waterfall": waterfall_items,
        # SHAP ML — validation par le modèle, direction corrigée
        "shap_waterfall":           shap_enriched,
        "top_shap_features":        shap_top,
        "n_discordant_features":    len(discordant_out),
        "discordant_features":      discordant_out,
        # Radar + règles
        "dimension_radar":   radar,
        "rule_contributions": rule_contributions,
        "top_contributing_dimensions": [
            {"dimension": k, "contribution": round(float(c), 2)}
            for k, c in top_dimensions[:3]
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# RISK ENGINE — v4.0
# ─────────────────────────────────────────────────────────────────────────────

class RiskMLEngine:
    """
    Charge les modèles ML (train_models.py v5.0) et expose .score() → RiskMLResult.
    Backward compat : scaler v3.0 (15 features) continue de fonctionner ;
    les 11 nouvelles features v4.0 alimentent les règles + score_breakdown + explanation.
    """

    _REQUIRED_FILES = [
        "scaler.pkl",
        "xgboost_calibrated.pkl",
        "isolation_forest.pkl",
        "kmeans.pkl",
        "shap_explainer.pkl",
        "cluster_names.json",
        "narrative_config.json",
        "metadata.json",
    ]

    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"
        self._models_dir = Path(models_dir)
        self._loaded     = False

        self._scaler          = None
        self._xgb_calibrated  = None
        self._lgb             = None
        self._iso             = None
        self._kmeans          = None
        self._shap_explainer  = None

        self._cluster_names:    dict       = {}
        self._metadata:         dict       = {}
        self._narrative_config: dict       = {}
        self._matrix_cols:      list[str]  = []

        self._extractor   = RiskFeatureExtractor()
        self._rule_engine = UCPRuleEngine()
        self._narrator:   Optional[NarrativeExplainer] = None

    def _load(self) -> None:
        if self._loaded:
            return
        import joblib

        md = self._models_dir
        try:
            self._scaler         = joblib.load(md / "scaler.pkl")
            self._xgb_calibrated = joblib.load(md / "xgboost_calibrated.pkl")
            self._iso            = joblib.load(md / "isolation_forest.pkl")
            self._kmeans         = joblib.load(md / "kmeans.pkl")
            self._shap_explainer = joblib.load(md / "shap_explainer.pkl")

            # L'explainer est en espace log-odds (model_output="raw", défaut).
            # Les contributions SHAP sont normalisées dans _build_lime_score_waterfall
            # pour s'additionner au risk_score.

            lgb_path = md / "lightgbm.pkl"
            if lgb_path.exists():
                self._lgb = joblib.load(lgb_path)

            with open(md / "cluster_names.json", encoding="utf-8") as f:
                self._cluster_names = json.load(f)

            with open(md / "metadata.json", encoding="utf-8") as f:
                self._metadata = json.load(f)

            n_sf = getattr(self._scaler, "n_features_in_", None)
            self._matrix_cols = resolve_matrix_feature_columns(self._metadata, n_sf)

            if n_sf is not None and len(self._matrix_cols) != n_sf:
                raise RuntimeError(
                    f"Schéma incohérent : {len(self._matrix_cols)} colonnes résolues "
                    f"!= scaler.n_features_in_={n_sf}. "
                    f"Réentraîner avec FEATURE_SCHEMA_VERSION={FEATURE_SCHEMA_VERSION}."
                )

            with open(md / "narrative_config.json", encoding="utf-8") as f:
                self._narrative_config = json.load(f)
            self._narrator = NarrativeExplainer(self._narrative_config)

            self._loaded = True
            logger.info(
                "RiskMLEngine v4.0 chargé — model_version=%s  schema=%s  n_ml_features=%d  n_total_features=26",
                self._metadata.get("version", "?"),
                self._metadata.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
                len(self._matrix_cols),
            )

        except RuntimeError:
            raise
        except Exception as exc:
            logger.error("Impossible de charger les modèles ML : %s", exc, exc_info=True)
            raise RuntimeError(f"Modèles risk ML introuvables dans {md} : {exc}") from exc

    def is_available(self) -> bool:
        return all((self._models_dir / f).exists() for f in self._REQUIRED_FILES)

    def score(self, pipeline_payload: dict) -> RiskMLResult:
        t0 = time.monotonic()

        try:
            self._load()
        except RuntimeError as exc:
            return RiskMLResult.error_result(str(exc))

        try:
            # ── 1. Extraction 26 features ─────────────────────────────────────
            features_dict  = self._extractor.extract(pipeline_payload)
            missing_detail = features_dict.pop("_missing_fields", [])

            # ── 2. Matrice ML (15 ou 26 selon modèle chargé) ─────────────────
            cols  = self._matrix_cols or list(FEATURE_COLS_V3)
            row   = [float(features_dict.get(c, FEATURE_DEFAULTS.get(c, 0.0))) for c in cols]
            X_raw    = np.array([row], dtype=float)
            X_scaled = self._scaler.transform(X_raw)

            # ── 3. UCP Rule Engine (14 règles) ────────────────────────────────
            violations, forced_label = self._rule_engine.evaluate(features_dict)

            # ── 4. Anomaly detection ──────────────────────────────────────────
            iso_score  = float(self._iso.decision_function(X_scaled)[0])
            is_anomaly = self._iso.predict(X_scaled)[0] == -1

            # ── 5. Segmentation ───────────────────────────────────────────────
            segment_id   = int(self._kmeans.predict(X_scaled)[0])
            segment_name = self._cluster_names.get(str(segment_id), f"SEGMENT_{segment_id}")

            # ── 6. Classification XGBoost calibré ─────────────────────────────
            proba_arr  = self._xgb_calibrated.predict_proba(X_scaled)[0]
            pred_class = int(np.argmax(proba_arr))
            ml_label_raw = LABEL_MAP[pred_class]

            # ── 6b. Calibration métier (post-ML) ─────────────────────────────
            # Réduit la sévérité sur les cas "conformes mais imparfaits"
            # Ne touche JAMAIS aux violations BLOCK (hard rules prioritaires)
            proba_arr, ml_label, calibration_reason = self._apply_business_calibration(
                features_dict, proba_arr, forced_label
            )
            if calibration_reason:
                logger.debug(
                    "Calibration métier appliquée : %s → %s (%s)",
                    ml_label_raw, ml_label, calibration_reason,
                )

            risk_proba = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(proba_arr)}

            # ── 7. Override rule engine ───────────────────────────────────────
            # forced_label = uniquement les BLOCK (contractuel, authoritative).
            final_label = forced_label if forced_label else ml_label

            # ── 7b. Escalade bornée par les violations HIGH (heuristiques) ────
            # Les règles HIGH (contradictions NLP, warnings, complétude…) ne
            # doivent pas, à elles seules, classer HIGH un LC que le modèle juge
            # fortement LOW. On escalade proportionnellement à l'évidence :
            #   • ≥ 2 signaux HIGH concordants, OU modèle pas confiant-LOW (P(LOW)<0.85)
            #       → HIGH
            #   • un seul signal HIGH isolé sur un cas modèle-LOW
            #       → MEDIUM (escalade d'un cran, pas HIGH)
            if not forced_label:
                high_violations = [v for v in violations if v.severity == "HIGH"]
                if high_violations:
                    p_low = float(proba_arr[0])
                    if len(high_violations) >= 2 or p_low < 0.85:
                        final_label = "HIGH"
                    elif final_label == "LOW":
                        final_label = "MEDIUM"

            # ── 8. Score ML 0-80 — poids réduits pour éviter la sévérité excessive ──
            # Coefficients 0/40/80 au lieu de 0/50/100 :
            # P(HIGH)=1.0 → 80  |  P(MED)=1.0 → 40  |  P(HIGH)=0.60 → 48
            risk_score = int(proba_arr[0] * 0 + proba_arr[1] * 40 + proba_arr[2] * 80)
            if is_anomaly and final_label != "HIGH":
                risk_score = min(risk_score + 10, 80)

            # Override contractuel UCP : label seulement, score inchangé
            ucp_payload = pipeline_payload.get("ucp_compliance") or {}
            if (
                str(ucp_payload.get("ucp_decision", "")).upper() == "REJECT"
                or bool(ucp_payload.get("payment_blocked"))
            ):
                final_label = "HIGH"

            # ── 9. SHAP + LIME waterfall ─────────────────────────────────────
            shap_top, shap_row, shap_base_high = self._compute_shap(X_scaled, pred_class, cols)

            lime_waterfall = self._build_lime_score_waterfall(
                shap_row        = shap_row,
                base_value      = shap_base_high,
                feature_values  = features_dict,
                risk_score      = risk_score,
            )

            # ── 10. LightGBM challenger ───────────────────────────────────────
            challenger_label: Optional[str] = None
            if self._lgb is not None:
                lgb_pred         = int(self._lgb.predict(X_scaled)[0])
                challenger_label = LABEL_MAP.get(lgb_pred, "UNKNOWN")
                if challenger_label != ml_label:
                    logger.info("Challenger divergence : XGB=%s LGB=%s", ml_label, challenger_label)

            # ── 11. Score breakdown 7 dimensions (référence seulement) ───────
            breakdown = compute_score_breakdown(
                features   = features_dict,
                risk_score = risk_score,
                risk_proba = risk_proba,
                risk_label = final_label,
                violations = violations,
            )

            # ── 12. Explanation : LIME waterfall + SHAP ───────────────────────
            explanation = compute_explanation(
                shap_top        = shap_top,
                shap_row        = shap_row,
                features        = features_dict,
                score_breakdown = breakdown,
                violations      = violations,
                risk_score      = risk_score,
                final_label     = final_label,
            )
            # Injecter le LIME score waterfall dans l'explication
            explanation["lime_score_waterfall"]       = lime_waterfall
            explanation["lime_score_base"]            = round(shap_base_high * 80, 2)
            explanation["lime_score_total"]           = risk_score

            # ── 13. UCP detail enrichi ────────────────────────────────────────
            ucp_detail = self._build_ucp_detail(features_dict, violations, ucp_payload)

            # ── 14. Narrative ─────────────────────────────────────────────────
            narrator  = self._narrator or NarrativeExplainer()
            narrative = narrator.explain(
                predicted_label = final_label,
                confidence      = float(proba_arr[pred_class]),
                shap_row        = shap_row,
                violations      = violations,
                features        = features_dict,
                missing_fields  = missing_detail,
            )

            ms = int((time.monotonic() - t0) * 1000)

            # Injecter la calibration dans l'explanation si elle a été appliquée
            if calibration_reason:
                explanation["_calibration_applied"] = True
                explanation["_calibration_reason"]  = calibration_reason
                explanation["_ml_label_pre_calib"]  = ml_label_raw

            # Export propre des features (sans clés internes)
            _internal_keys = {"ucp_decision_reject", "_validation_issues", "missing_critical_count"}
            features_export = {
                k: round(float(v), 4)
                for k, v in features_dict.items()
                if k not in _internal_keys
                and not k.startswith("_")
                and isinstance(v, (int, float))
            }

            return RiskMLResult(
                risk_label           = final_label,
                risk_proba           = risk_proba,
                risk_score           = risk_score,
                is_anomaly           = bool(is_anomaly),
                anomaly_score        = iso_score,
                segment              = segment_name,
                segment_id           = segment_id,
                shap_top_features    = shap_top,
                ucp_violations       = [asdict(v) for v in violations],
                ucp_detail           = ucp_detail,
                narrative            = narrative,
                features_used        = features_export,
                model_version        = self._metadata.get("version", "?"),
                score_breakdown      = breakdown,
                missing_fields_detail = missing_detail,
                explanation          = explanation,
                challenger_label     = challenger_label,
                processing_ms        = ms,
            )

        except Exception as exc:
            logger.error("RiskMLEngine.score() error: %s", exc, exc_info=True)
            return RiskMLResult.error_result(str(exc))

    def _build_ucp_detail(
        self,
        features: dict,
        violations: list[UCPViolation],
        ucp_payload: dict,
    ) -> dict:
        """Construit le bloc ucp_detail avec erreurs/warnings séparés + weighted score."""
        errors_count  = int(features.get("ucp_errors_count", 0))
        warns_count   = int(features.get("ucp_warnings_count", 0))
        weighted      = errors_count * 1.0 + warns_count * 0.4

        # Extraire les listes depuis le payload UCP
        ucp_errors_list   = ucp_payload.get("errors",   []) or []
        ucp_warnings_list = ucp_payload.get("warnings", []) or []

        return {
            "errors_count":              errors_count,
            "warnings_count":            warns_count,
            "weighted_violation_index":  round(weighted, 2),
            "error_weight":              1.0,
            "warning_weight":            0.4,
            "compliance_score_pct":      round(features.get("ucp_compliance_score", 0.5) * 100, 1),
            "ucp_decision":              str(ucp_payload.get("ucp_decision", "PENDING")).upper(),
            "payment_blocked":           bool(ucp_payload.get("payment_blocked", False)),
            "rule_violations_count":     len(violations),
            "block_violations":          [asdict(v) for v in violations if v.severity == "BLOCK"],
            "high_violations":           [asdict(v) for v in violations if v.severity == "HIGH"],
            "warn_violations":           [asdict(v) for v in violations if v.severity == "WARN"],
            "errors_detail":             ucp_errors_list[:10],    # top 10
            "warnings_detail":           ucp_warnings_list[:10],  # top 10
        }

    def _apply_business_calibration(
        self,
        features: dict,
        proba_arr: np.ndarray,
        forced_label: Optional[str],
    ) -> tuple[np.ndarray, str, Optional[str]]:
        """
        Calibration métier post-ML (UCP 600 / ISBP 821 / trade finance practice).

        Corrige la sévérité excessive du modèle ML sur les LC "conformes mais imparfaits"
        (cas où UCP est élevé, aucune erreur réglementaire, mais des paramètres
        opérationnels normaux comme validity_days ou days_to_shipment font monter
        artificiellement le score vers MEDIUM/HIGH).

        Règles de calibration (du plus au moins fort) :
          1. Zone sûre totale     → HIGH ≤ 5%,  MEDIUM ≤ 20%
          2. UCP excellence       → HIGH ≤ 12%
          3. Zone modérée propre  → HIGH ≤ 30%
          4. Bruit temporel isolé → HIGH ≤ 35%

        Le dataset d'entraînement contient 42.6% de cas HIGH (biais de sur-représentation).
        Le seuil de décision HIGH est relevé à _HIGH_DECISION_THRESHOLD (0.60) pour compenser.

        Garanties :
          - Ne touche jamais aux violations BLOCK (hard rules prioritaires)
          - Ne modifie pas les cas avec erreurs UCP critiques, contradictions, pays sanctionnés
          - La renormalisation préserve la somme des probabilités = 1
        """
        # Les hard rules BLOCK ont toujours la priorité absolue
        if forced_label == "HIGH":
            raw_class = int(np.argmax(proba_arr))
            return proba_arr, LABEL_MAP[raw_class], None

        ucp_score     = float(features.get("ucp_compliance_score", 0.5))
        ucp_errors    = float(features.get("ucp_errors_count", 0))
        ucp_warns     = float(features.get("ucp_warnings_count", 0))
        contradiction = float(features.get("contradiction_flag", 0))
        country_risk  = float(features.get("country_risk_score", 25))
        missing_crit  = float(features.get("missing_critical_score", 0.75))
        red_clause    = float(features.get("red_clause_flag", 0))
        pres_days     = float(features.get("presentation_days", 21))
        validity_d    = float(features.get("validity_days", 90))

        proba = proba_arr.copy().astype(float)
        reason: Optional[str] = None

        # ── 1. Zone sûre totale ─────────────────────────────────────────────
        # LC conforme : UCP ≥ 85%, 0 erreur, ≤ 2 warnings,
        # pas de contradiction, pays non risqué, champs couverts, pas de clause rouge
        is_safe_zone = (
            ucp_score    >= 0.85
            and ucp_errors   == 0
            and ucp_warns    <= 2
            and contradiction == 0
            and country_risk  < 55
            and missing_crit  >= 0.70
            and red_clause    == 0
            and pres_days     >= 5
            and validity_d    >= 15
        )
        if is_safe_zone:
            excess_high   = max(0.0, proba[2] - 0.05)
            excess_medium = max(0.0, proba[1] - 0.20)
            proba[2] = min(proba[2], 0.05)
            proba[1] = min(proba[1], 0.20)
            proba[0] = min(1.0, proba[0] + excess_high + excess_medium)
            proba /= proba.sum()
            reason = "safe_zone:UCP_gte85pct,0_erreur,pas_contradiction,pays_sain"

        # ── 2. UCP excellence (≥ 92%, 0 erreur, pays non sanctionné) ────────
        elif ucp_score >= 0.92 and ucp_errors == 0 and country_risk < 75 and red_clause == 0:
            excess = max(0.0, proba[2] - 0.12)
            proba[2] = min(proba[2], 0.12)
            proba[0] = min(1.0, proba[0] + excess)
            proba /= proba.sum()
            reason = "ucp_excellence:UCP_gte92pct,0_erreur"

        # ── 3. Zone modérée propre ───────────────────────────────────────────
        # Document avec quelques imperfections mais rien de critique :
        # UCP ≥ 75%, ≤ 1 erreur, pays non risqué, pas de contradiction
        elif (
            ucp_score    >= 0.75
            and ucp_errors   <= 1
            and contradiction == 0
            and country_risk  < 65
            and red_clause    == 0
            and missing_crit  >= 0.60
        ):
            excess = max(0.0, proba[2] - 0.30)
            proba[2] = min(proba[2], 0.30)
            proba[0] = min(1.0, proba[0] + excess * 0.5)
            proba[1] = min(1.0, proba[1] + excess * 0.5)
            proba /= proba.sum()
            reason = "moderate_zone:UCP_gte75pct,le1_erreur,pas_contradiction"

        # ── 4. Bruit temporel isolé (opérationnel propre) ───────────────────
        # Seul driver suspecté = temporel (validity/days_to_shipment dans plage normale)
        elif (
            ucp_score    >= 0.78
            and ucp_errors   == 0
            and contradiction == 0
            and country_risk  < 45
            and red_clause    == 0
            and pres_days     >= 10
            and 21 <= validity_d <= 548
        ):
            excess = max(0.0, proba[2] - 0.35)
            proba[2] = min(proba[2], 0.35)
            proba[0] = min(1.0, proba[0] + excess)
            proba /= proba.sum()
            reason = "temporal_noise:indicateurs_propres,validity_dans_plage_normale"

        # ── 5. Profil analytiquement LOW — bride le ML si les features sont saines ──
        # Quand l'ensemble des indicateurs métier pointe vers LOW, le ML ne peut pas
        # imposer HIGH seul (probable artefact de corrélation du dataset synthétique).
        analytically_low = (
            ucp_score    >= 0.80
            and ucp_errors   == 0
            and contradiction == 0
            and country_risk  < 50
            and missing_crit  >= 0.65
            and red_clause    == 0
        )
        if analytically_low and proba[2] > 0.45:
            excess = max(0.0, proba[2] - 0.45)
            proba[2] = 0.45
            proba[1] = min(1.0, proba[1] + excess * 0.6)
            proba[0] = min(1.0, proba[0] + excess * 0.4)
            proba /= proba.sum()
            reason = (reason or "") + "|analytically_low:ML_bridé_à_45pct"

        # ── Seuil de décision HIGH ────────────────────────────────────────────
        # Dataset biaisé (42.6% HIGH en train) : P(HIGH) doit être ≥ 0.60
        # pour prononcer HIGH, sinon on reste sur MEDIUM.
        if proba[2] >= _HIGH_DECISION_THRESHOLD:
            calibrated_class = 2
        elif proba[1] >= 0.50:
            calibrated_class = 1
        else:
            calibrated_class = 0

        return proba, LABEL_MAP[calibrated_class], reason

    # Mapping feature → label français pour le LIME waterfall
    _FEAT_LABELS: dict[str, str] = {
        "lc_amount":                "Montant LC",
        "country_risk_score":       "Risque pays",
        "country_risk_applicant":   "Risque applicant",
        "country_risk_beneficiary": "Risque bénéficiaire",
        "external_currency_risk":   "Risque devise",
        "incoterm_risk":            "Incoterm",
        "ucp_compliance_score":     "Score UCP",
        "n_ucp_warnings":           "Violations UCP",
        "ucp_errors_count":         "Erreurs UCP",
        "ucp_warnings_count":       "Avertissements UCP",
        "ucp_warning_rate":         "Densité UCP",
        "nlp_score":                "Qualité NLP",
        "semantic_score":           "Cohérence sémantique",
        "completeness_score":       "Complétude",
        "missing_critical_score":   "Champs critiques",
        "coherence_dates":          "Cohérence dates",
        "contradiction_flag":       "Contradiction",
        "days_to_shipment":         "Délai expédition",
        "presentation_days":        "Délai présentation",
        "validity_days":            "Durée validité",
        "amount_tier":              "Tier montant",
        "transshipment_allowed":    "Transbordement",
        "charges_beneficiary":      "Frais bénéf.",
        "red_clause_flag":          "Clause rouge",
        "bank_tier":                "Tier banque",
        "tenor_risk":               "Risque tenor",
    }

    _TONE_COLORS = {
        "up":   ("#ef4444", "high"),
        "down": ("#22c55e", "low"),
    }

    def _compute_shap(
        self,
        X_scaled:      np.ndarray,
        pred_class:    int,
        feature_names: list[str],
    ) -> tuple[list, dict, float]:
        """Retourne (shap_top_5, shap_row, base_value_for_HIGH)."""
        names    = feature_names or list(FEATURE_COLS_V3)
        zero_row = {f: 0.0 for f in names}
        if self._shap_explainer is None:
            return [], zero_row, 0.0

        try:
            shap_values = self._shap_explainer.shap_values(X_scaled)

            # Valeurs SHAP en log-odds pour la classe HIGH (index 2)
            if isinstance(shap_values, list):
                sv_high = np.array(shap_values[2][0]).reshape(-1)
                sv_pred = np.array(shap_values[pred_class][0]).reshape(-1)
            elif shap_values.ndim == 3:
                sv_high = shap_values[0, :, 2]
                sv_pred = shap_values[0, :, pred_class]
            else:
                sv_high = shap_values[0]
                sv_pred = sv_high

            # base_value log-odds pour HIGH (utilisé uniquement comme ancrage du waterfall)
            ev = getattr(self._shap_explainer, "expected_value", None)
            if ev is not None and hasattr(ev, "__len__") and len(ev) >= 3:
                base_value_high = float(ev[2])
            elif ev is not None and not hasattr(ev, "__len__"):
                base_value_high = float(ev)
            else:
                base_value_high = 0.0

            if len(sv_high) != len(names):
                n       = min(len(sv_high), len(names))
                sv_high = sv_high[:n]
                sv_pred = sv_pred[:n]
                names   = names[:n]

            # shap_row contient les valeurs HIGH (log-odds) pour le waterfall
            shap_row = dict(zip(names, sv_high.tolist()))
            ranked   = sorted(zip(names, sv_pred), key=lambda x: abs(x[1]), reverse=True)[:5]

            shap_top = [
                {
                    "feature":    feat,
                    "shap_value": round(float(val), 5),
                    "direction":  "↑ risque" if val > 0 else "↓ risque",
                    "value":      None,
                }
                for feat, val in ranked
            ]
            return shap_top, shap_row, base_value_high

        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return [], zero_row, 0.0

    def _build_lime_score_waterfall(
        self,
        shap_row:       dict,   # {feature: shap_value_HIGH (log-odds)}
        base_value:     float,  # E[log_odds_HIGH] — base log-odds
        feature_values: dict,   # valeurs réelles des features
        risk_score:     int,    # risk_score (0–80) — cible du waterfall
        top_n:          int = 12,
    ) -> list[dict]:
        """
        LIME score waterfall normalisé : base_score + Σ contributions = risk_score.

        Les valeurs SHAP sont en log-odds (raw). On les normalise
        proportionnellement pour que leur somme corresponde à (risk_score − base_score),
        ce qui garantit la propriété additive dans l'espace du score.
        """
        import math

        # Convertir base log-odds → probabilité → score
        try:
            base_prob  = 1.0 / (1.0 + math.exp(-base_value))
        except OverflowError:
            base_prob  = 0.0 if base_value < 0 else 1.0
        base_score = round(base_prob * 80, 2)

        # Somme totale des SHAP log-odds (dénominateur pour la normalisation)
        total_shap = sum(shap_row.values())
        delta      = risk_score - base_score  # ce que les features doivent expliquer en pts

        pairs = []
        for feat, sv in shap_row.items():
            if abs(sv) < 1e-6:
                continue
            # Contribution proportionnelle : sv / total_shap × delta
            if abs(total_shap) > 1e-9:
                score_contrib = (sv / total_shap) * delta
            else:
                score_contrib = 0.0
            pairs.append((feat, float(sv), score_contrib, feature_values.get(feat)))

        # Trier par contribution absolue au score
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top = pairs[:top_n]

        items: list[dict] = []
        cumulative = base_score

        for feat, shap_val, score_contrib, feat_val in top:
            direction = "up" if score_contrib > 0 else "down"
            color, tone = self._TONE_COLORS[direction]
            start = cumulative
            end   = round(cumulative + score_contrib, 2)

            items.append({
                "feature":            feat,
                "label":              self._FEAT_LABELS.get(feat, feat.replace("_", " ")),
                "value":              round(float(feat_val), 4) if feat_val is not None else None,
                "shap_value":         round(shap_val, 5),
                "score_contribution": round(score_contrib, 3),
                "start":              round(start, 2),
                "end":                end,
                "direction":          direction,
                "color":              color,
                "risk_tone":          tone,
            })
            cumulative = end

        return items

    def inject_into_report(self, pipeline_payload: dict, result: RiskMLResult) -> None:
        """
        Injecte risk_ml dans pipeline_payload["risk_ml"].
        Decision Engine hiérarchique : OFAC/CRITICAL > UCP contractuel > ML > DS.
        """
        pipeline_payload["risk_ml"] = result.to_dict()
        if result.error:
            return

        ext       = pipeline_payload.get("external_validation", {}) or {}
        ext_level = ext.get("risk_level", "")

        # Niveau 1 : Hard override external CRITICAL
        if ext_level == "CRITICAL":
            original_label = result.risk_label
            original_score = result.risk_score
            pipeline_payload["risk_ml"]["risk_label"]     = "HIGH"
            pipeline_payload["risk_ml"]["risk_score"]     = 100
            pipeline_payload["risk_ml"]["_hard_override"] = "external_validation=CRITICAL"
            pipeline_payload["risk_ml"]["_ml_insight"]    = (
                f"ML prédit {original_label} (score={original_score}) "
                f"— overridé CRITICAL (OFAC/EU/UN/FATF)"
            )
            effective_ml_label = "HIGH"
        else:
            effective_ml_label = result.risk_label

        # Niveau 2 : UCP contractuel — label seulement, score reflète les probabilités ML
        ucp = pipeline_payload.get("ucp_compliance") or {}
        if (
            str(ucp.get("ucp_decision", "")).upper() == "REJECT"
            or bool(ucp.get("payment_blocked"))
        ) and ext_level != "CRITICAL":
            pipeline_payload["risk_ml"]["risk_label"]                = "HIGH"
            pipeline_payload["risk_ml"]["_ucp_contractual_override"] = (
                "ucp_compliance REJECT ou payment_blocked — label forcé HIGH, score ML inchangé"
            )
            effective_ml_label = "HIGH"

        # Niveaux 3 & 4 : escalade croisée ML ↔ ds_analysis
        ds      = pipeline_payload.get("ds_analysis", {}) or {}
        ds_risk = ds.get("risk") or {}
        ds_level = ds_risk.get("risk_level", "")

        ml_rank = _LEVEL_RANK.get(effective_ml_label, -1)
        ds_rank = _LEVEL_RANK.get(ds_level, -1)

        if ml_rank > ds_rank and ds_level:
            if "risk" in ds:
                ds["risk"]["risk_level"]    = effective_ml_label
                ds["risk"]["_ml_escalated"] = True
        elif ds_rank > ml_rank and ds_level:
            if ds_risk.get("_external_escalated"):
                pipeline_payload["risk_ml"]["_ds_external_note"] = (
                    f"ds_analysis.risk_level={ds_level} inclut escalade externe ; ml={effective_ml_label}"
                )
            else:
                pipeline_payload["risk_ml"]["_ds_override_warning"] = (
                    f"ds_analysis={ds_level} > ml={effective_ml_label} — divergence : revue manuelle"
                )

        # ── Score final = score ML (source unique, expliqué par LIME) ──────────
        # Le score analytique n'est plus utilisé pour le calcul final.
        # Le LIME waterfall dans explanation.lime_score_waterfall explique le score.
        rm   = pipeline_payload["risk_ml"]
        ml_r = int(rm.get("risk_score", 0) or 0)
        try:
            ds_raw = float(ds_risk.get("risk_score", 0) or 0)
        except (TypeError, ValueError):
            ds_raw = 0.0
        ds_100 = int(round(max(0.0, min(1.0, ds_raw)) * 100))

        final = max(ml_r, ds_100)
        if ext_level == "CRITICAL":
            final = 100
        rm["final_risk_score"] = min(100, final)
        rm["final_risk_score_components"] = {
            "ml_0_80":   ml_r,
            "ds_0_100":  ds_100,
        }
        rm["risk_feature_schema_version"] = self._metadata.get(
            "feature_schema_version", FEATURE_SCHEMA_VERSION
        )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON thread-safe
# ─────────────────────────────────────────────────────────────────────────────

_risk_engine_singleton: Optional[RiskMLEngine] = None
_singleton_lock = threading.Lock()


def get_risk_engine() -> RiskMLEngine:
    global _risk_engine_singleton
    if _risk_engine_singleton is None:
        with _singleton_lock:
            if _risk_engine_singleton is None:
                _risk_engine_singleton = RiskMLEngine()
    return _risk_engine_singleton
