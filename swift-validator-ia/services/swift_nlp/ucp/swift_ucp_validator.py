"""
swift_ucp_validator.py  —  v3.0
================================
Validation UCP 600 avancée des documents LC.

Le rapport `ucp_compliance` (findings, ucp_decision, payment_blocked) est la
source de vérité contractuelle UCP ; le scoring ML en aval ne remplace pas
cette décision.

NOUVELLES TECHNIQUES v3 :
  ✦ RuleWeightRegistry — poids configurables par article (YAML ou code)
  ✦ Pénalités cascade — une violation Art.20 aggrave le score Art.14/16
  ✦ Typage sémantique des exclusions (ExclusionSemanticTyper intégré)
  ✦ Résolveur Art.5 — la clause prime TOUJOURS sur le tag SWIFT
  ✦ Couverture étendue : Art.19/22/23/26 (multimodal, aérien, postal)
  ✦ UCPFinding v3 enrichi : weight + confidence + cascade_impact
  ✦ Score global basé sur poids pondérés, pas sur simple comptage
  ✦ Seuils de compliance configurables par type de LC

Architecture :
  RuleWeightRegistry      — registre des poids par article
  UCPFinding v3           — finding enrichi
  CascadePenaltyEngine    — propagation de pénalités entre articles
  UCP600RuleEngine v3     — base de règles étendue
  ClauseExclusionParser v3 — parser amélioré + contexte
  UCP600Validator v3      — orchestrateur principal
"""

from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, date

from .swift_ucp_hard_rules import (
    dedupe_ucp_finding_dicts,
    evaluate_hard_rules,
    evaluate_ucp_decision,
    findings_from_critical_exclusion_map,
    merge_hard_findings,
    ucp_blocking_violations_list,
)

logger = logging.getLogger(__name__)


def _merge_semantic_rule_weights(weights: dict) -> None:
    """Poids pour codes sémantiques (violation, restriction, contractual_mod)."""
    slugs = [
        "ART14", "ART14C", "ART14K", "ART17", "ART18", "ART18B",
        "ART20", "ART20C", "ART28", "ART38", "ART38I",
    ]
    viol_blocks = {"ART14K", "ART17", "ART18", "ART20", "ART20C", "ART28", "ART38I"}
    for slug in slugs:
        weights[f"SEM_VIOL_{slug}"] = {
            "weight": 0.88,
            "tier": 1 if slug in viol_blocks else 2,
            "blocks_payment": slug in viol_blocks,
        }
        weights[f"SEM_REST_{slug}"] = {
            "weight": 0.42,
            "tier": 3,
            "blocks_payment": False,
        }
        weights[f"SEM_CONTRACT_MOD_{slug}"] = {
            "weight": 0.30,
            "tier": 3,
            "blocks_payment": False,
        }
        weights[f"SEM_CONTRACT_STRUCT_{slug}"] = {
            "weight": 0.72 if slug == "ART38I" else 0.58,
            "tier": 2 if slug != "ART38I" else 1,
            "blocks_payment": slug == "ART38I",
        }
    for hb in ("UCP_HARD_EXCL_ART18", "UCP_HARD_EXCL_ART38I", "UCP_HARD_CONTRACT_ART18", "UCP_HARD_CONTRACT_ART38I"):
        weights[hb] = {"weight": 0.72 if hb.endswith("ART38I") else 0.58, "tier": 2, "blocks_payment": hb.endswith("ART38I")}


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — REGISTRE DES POIDS PAR ARTICLE
# ══════════════════════════════════════════════════════════════════════

class RuleWeightRegistry:
    """
    Poids de criticité par article UCP 600.
    Tier 1 (weight 1.0) : violations bloquantes pour le paiement
    Tier 2 (weight 0.7) : violations importantes
    Tier 3 (weight 0.4) : alertes
    Tier 4 (weight 0.1) : informatif

    Configurable via YAML (charge depuis rule_weights.yaml si présent).
    """

    _DEFAULTS: dict[str, dict] = {
        # Tier 1 — bloquant pour paiement
        "ART14K_EXCLUDED":               {"weight": 0.65, "tier": 2, "blocks_payment": False},
        "ART17_NO_SCANNED":              {"weight": 1.0, "tier": 1, "blocks_payment": True},
        "ART18B_STRICT_LIMIT":           {"weight": 0.55, "tier": 2, "blocks_payment": False},
        "ART18_NO_THIRD_PARTY_INVOICE":  {"weight": 1.0, "tier": 1, "blocks_payment": True},
        "ART20_NO_BLANK_BACK":           {"weight": 0.9, "tier": 1, "blocks_payment": True},
        "ART28_NO_FRANCHISE":            {"weight": 0.9, "tier": 1, "blocks_payment": True},
        "ART28_NO_COVERAGE_RESTRICTION": {"weight": 0.9, "tier": 1, "blocks_payment": True},
        "ART38I_EXCLUDED":               {"weight": 0.78, "tier": 1, "blocks_payment": True},
        "ART20C_CLAUSE_OVERRIDES_TAG":   {"weight": 1.0, "tier": 1, "blocks_payment": True},
        "ART20C_EXCLUDED":               {"weight": 0.62, "tier": 2, "blocks_payment": False},
        # Tier 2 — important
        "ART6_NO_EXPIRY":                {"weight": 0.8, "tier": 2, "blocks_payment": False},
        "ART6_SHORT_VALIDITY":           {"weight": 0.7, "tier": 2, "blocks_payment": False},
        "ART14C_EXCLUDED":               {"weight": 0.6, "tier": 2, "blocks_payment": False},
        "ART14_NO_DOCS":                 {"weight": 0.6, "tier": 2, "blocks_payment": False},
        "ART19_MULTIMODAL_RESTRICTION":  {"weight": 0.7, "tier": 2, "blocks_payment": False},
        "ART23_AWB_RESTRICTION":         {"weight": 0.7, "tier": 2, "blocks_payment": False},
        # Tier 3 — alerte
        "ART6_EXPIRED":                  {"weight": 0.3, "tier": 3, "blocks_payment": False},
        "ART6_NO_EXPIRY_PLACE":          {"weight": 0.3, "tier": 3, "blocks_payment": False},
        "ART6_NO_AVAILABLE_BY":          {"weight": 0.4, "tier": 3, "blocks_payment": False},
        "ART16_RESERVATION_FEE":         {"weight": 0.4, "tier": 3, "blocks_payment": False},
        "ART20_LCL_RESTRICTION":         {"weight": 0.4, "tier": 3, "blocks_payment": False},
        "ART28_NO_WARRANTED":            {"weight": 0.4, "tier": 3, "blocks_payment": False},
        "ART37C_EXCLUDED_CHARGES_BENEFICIARY": {"weight": 0.3, "tier": 3, "blocks_payment": False},
        "ART30_TOLERANCE_CONFLICT":      {"weight": 0.7, "tier": 2, "blocks_payment": False},
        "ART12B_EXCLUDED":               {"weight": 0.3, "tier": 3, "blocks_payment": False},
        "UCP_HARD_CONTRACT_ART18":       {"weight": 0.58, "tier": 2, "blocks_payment": False},
        "UCP_HARD_CONTRACT_ART38I":      {"weight": 0.72, "tier": 1, "blocks_payment": True},
        # Tier 4 — informatif
        "ART8_CONFIRMATION_REQUESTED":   {"weight": 0.1, "tier": 4, "blocks_payment": False},
        "ART17_BLUE_INK_SIGNATURE":      {"weight": 0.1, "tier": 4, "blocks_payment": False},
        "ART30_ZERO_TOLERANCE":          {"weight": 0.0, "tier": 4, "blocks_payment": False},
        # Art.10 — Amendements
        "ART10_PARTIAL_AMENDMENT":       {"weight": 0.85, "tier": 1, "blocks_payment": True},
        "ART10_AUTO_ACCEPTANCE_CLAUSE":  {"weight": 0.40, "tier": 3, "blocks_payment": False},
        "ART10_UNILATERAL_AMENDMENT":    {"weight": 0.80, "tier": 1, "blocks_payment": True},
        # Art.31 — Expéditions partielles
        "ART31_PARTIAL_PROHIBITED":      {"weight": 0.55, "tier": 2, "blocks_payment": False},
        # Art.32 — Tranches
        "ART32_INSTALMENT_SCHEDULE":     {"weight": 0.60, "tier": 2, "blocks_payment": False},
        # Art.39 — Cession
        "ART39_ASSIGNMENT_PROHIBITED":   {"weight": 0.30, "tier": 3, "blocks_payment": False},
        "ART39_ASSIGNMENT_NOTED":        {"weight": 0.10, "tier": 4, "blocks_payment": False},
        # Art.22 — Charter-partie
        "ART22_CHARTER_PROHIBITED":      {"weight": 0.70, "tier": 2, "blocks_payment": False},
        # Art.19/23/26 — Transport
        "ART19_MULTIMODAL_RESTRICTION":  {"weight": 0.55, "tier": 2, "blocks_payment": False},
        "ART19_INCOTERM_MULTIMODAL_MISMATCH": {"weight": 0.20, "tier": 4, "blocks_payment": False},
        "ART23_AWB_RESTRICTION":         {"weight": 0.50, "tier": 2, "blocks_payment": False},
        "ART23_AWB_POSSIBLE":            {"weight": 0.10, "tier": 4, "blocks_payment": False},
        "ART26_COURIER_RESTRICTED":      {"weight": 0.45, "tier": 3, "blocks_payment": False},
        # Cross-field
        "CROSS_FIELD_VIOLATION":         {"weight": 0.90, "tier": 1, "blocks_payment": True},
    }

    def __init__(self, custom_weights: Optional[dict] = None) -> None:
        self._weights = dict(self._DEFAULTS)
        _merge_semantic_rule_weights(self._weights)
        if custom_weights:
            self._weights.update(custom_weights)
        # Tentative de chargement YAML
        self._load_yaml()

    def _load_yaml(self) -> None:
        try:
            import yaml
            from pathlib import Path
            p = Path("rule_weights.yaml")
            if p.exists():
                with open(p) as f:
                    data = yaml.safe_load(f)
                if data:
                    self._weights.update(data)
                    logger.info("RuleWeightRegistry: poids chargés depuis rule_weights.yaml")
        except Exception:
            pass

    def get_weight(self, code: str) -> float:
        return self._weights.get(code, {}).get("weight", 0.5)

    def get_tier(self, code: str) -> int:
        return self._weights.get(code, {}).get("tier", 3)

    def blocks_payment(self, code: str) -> bool:
        return self._weights.get(code, {}).get("blocks_payment", False)

    def get_penalty(self, code: str, severity: str) -> float:
        """Pénalité pondérée pour le scoring."""
        base = {"error": 20, "warning": 8, "info": 0}.get(severity, 5)
        weight = self.get_weight(code)
        return base * weight


# ══════════════════════════════════════════════════════════════════════
# NEW v3 — MOTEUR DE PÉNALITÉS EN CASCADE
# ══════════════════════════════════════════════════════════════════════

class CascadePenaltyEngine:
    """
    Quand une violation est détectée sur Art.X, certains articles liés
    reçoivent une pénalité supplémentaire ("cascade impact").

    Ex : violation BL (Art.20) → pénalité supplémentaire sur doc score
         violation assurance (Art.28) → cascade sur compliance score
    """

    _CASCADES: dict[str, list[tuple[str, float, str]]] = {
        # (code_source): [(catégorie_cible, malus_additionnel, raison)]
        "ART20_NO_BLANK_BACK":           [("document",   5.0, "BL non conforme → examen Art.14 requis")],
        "ART20C_CLAUSE_OVERRIDES_TAG":   [("document",   8.0, "Transbordement → impact BL et Art.14"),
                                           ("shipment",   6.0, "Contrainte expédition")],
        "ART28_NO_FRANCHISE":            [("document",   5.0, "Police non conforme"),
                                           ("compliance",  3.0, "Impact présentation Art.7")],
        "ART28_NO_COVERAGE_RESTRICTION": [("document",   6.0, "Couverture insuffisante"),
                                           ("compliance",  4.0, "Risque non-présentation")],
        "ART38I_EXCLUDED":               [("document",   5.0, "Factures premier bénéficiaire"),
                                           ("compliance",  5.0, "LC transférable — vérification Art.38")],
        "ART14K_EXCLUDED":               [("document",   7.0, "Expéditeur doc transport — vérif toutes docs"),
                                           ("shipment",   5.0, "Impact BL et connaissement")],
        "ART17_NO_SCANNED":              [("document",   8.0, "Tous docs concernés — originalité"),
                                           ("compliance",  4.0, "Impact présentation Art.16")],
        "SEM_VIOL_ART28":                [("document",   5.0, "Police non conforme"),
                                           ("compliance",  3.0, "Impact présentation Art.7")],
        "SEM_VIOL_ART17":                [("document",   8.0, "Tous docs concernés — originalité"),
                                           ("compliance",  4.0, "Impact présentation Art.16")],
        "SEM_VIOL_ART14K":               [("document",   7.0, "Expéditeur doc transport"),
                                           ("shipment",   5.0, "Impact BL")],
        "SEM_VIOL_ART38I":               [("document",   5.0, "Factures premier bénéficiaire"),
                                           ("compliance",  5.0, "LC transférable")],
        "SEM_VIOL_ART20":                [("document",   5.0, "BL non conforme"),
                                           ("shipment",   4.0, "Contrainte expédition")],
        "SEM_CONTRACT_STRUCT_ART38I":    [("document",   4.0, "Exclusion structurelle sur substitution facture"),
                                           ("compliance", 4.0, "Impact crédit transférable")],
    }

    _MAX_CASCADE_PER_SOURCE: float = 15.0   # plafond par finding source

    @classmethod
    def compute(
        cls,
        finding_code: str,
        categories: dict,
    ) -> list[dict]:
        """Applique les pénalités cascade avec plafond par source (évite score négatif)."""
        applied = []
        cascades = cls._CASCADES.get(finding_code, [])
        cumul = 0.0
        for cat, malus, reason in cascades:
            remaining = min(malus, cls._MAX_CASCADE_PER_SOURCE - cumul)
            if remaining <= 0:
                break
            if cat in categories:
                categories[cat].score = max(0.0, categories[cat].score - remaining)
                cumul += remaining
                applied.append({
                    "source_code": finding_code,
                    "target_cat":  cat,
                    "malus":       round(remaining, 2),
                    "reason":      reason,
                })
        return applied


# ══════════════════════════════════════════════════════════════════════
# MODÈLE DE DONNÉES v3
# ══════════════════════════════════════════════════════════════════════

@dataclass
class UCPFinding:
    """Non-conformité UCP 600 — enrichie v4 (impact_level + clause_type)."""
    article:       str
    article_title: str
    code:          str
    severity:      str    # "error" | "warning" | "info"
    category:      str
    message:       str
    detail:        str  = ""
    clause_text:   str  = ""
    recommendation: str = ""
    # v3
    weight:          float = 0.5
    tier:            int   = 3
    cascade_impact:  float = 0.0
    blocks_payment:  bool  = False
    confidence:      float = 1.0
    exclusion_type:  str   = ""
    finding_nature:  str   = "true_violation"  # true_violation | contractual_modification | operational_discrepancy
    blocks_payment_reason: str = ""
    # v4 — taxonomie métier
    impact_level: str = "MEDIUM"       # HIGH | MEDIUM | LOW
    clause_type:  str = "document_format"  # financial | transport | authentication | identity | compliance | document_format

    def to_dict(self) -> dict:
        return {
            "article":        self.article,
            "article_title":  self.article_title,
            "code":           self.code,
            "severity":       self.severity,
            "category":       self.category,
            "message":        self.message,
            "detail":         self.detail,
            "clause_text":    self.clause_text[:120] if self.clause_text else "",
            "recommendation": self.recommendation,
            "weight":         round(self.weight, 3),
            "tier":           self.tier,
            "cascade_impact": round(self.cascade_impact, 2),
            "blocks_payment": self.blocks_payment,
            "confidence":     round(self.confidence, 3),
            "exclusion_type": self.exclusion_type,
            "finding_nature": self.finding_nature,
            "blocks_payment_reason": self.blocks_payment_reason,
            "impact_level":   self.impact_level,
            "clause_type":    self.clause_type,
        }


@dataclass
class UCPCategoryScore:
    category:     str
    score:        float = 100.0
    max_score:    float = 100.0
    findings:     list[UCPFinding] = field(default_factory=list)
    articles_ok:  list[str]        = field(default_factory=list)
    articles_nok: list[str]        = field(default_factory=list)


@dataclass
class UCPValidationReport:
    lc_number:        str
    swift_type:       str
    ucp_version:      str
    findings:         list[UCPFinding]             = field(default_factory=list)
    category_scores:  dict[str, UCPCategoryScore]  = field(default_factory=dict)
    exclusions:       dict[str, str]               = field(default_factory=dict)
    exclusion_types:  dict[str, str]               = field(default_factory=dict)
    global_score:     int                          = 0
    compliant:        bool                         = False
    payment_blocked:  bool                         = False   # violation réelle bloquante
    contractual_hold: bool                         = False   # modification contractuelle Art.1 à risque élevé
    cascade_log:      list[dict]                   = field(default_factory=list)
    processed_at:     str                          = ""
    semantic:         dict                         = field(default_factory=dict)

    @property
    def legal_compliance_status(self) -> str:
        if any(f.finding_nature == "true_violation" and f.severity == "error" for f in self.findings):
            return "NON_COMPLIANT"
        return "COMPLIANT_WITH_CONTRACTUAL_MODIFICATIONS"

    @property
    def contractual_risk_level(self) -> str:
        has_structural = any(f.finding_nature == "contractual_modification" and "STRUCT" in f.code for f in self.findings)
        has_contract = any(f.finding_nature == "contractual_modification" for f in self.findings)
        if has_structural:
            return "HIGH"
        if has_contract:
            return "MEDIUM"
        return "LOW"

    @property
    def documentary_discrepancy_risk(self) -> str:
        has_doc_block = any(
            f.finding_nature == "operational_discrepancy" and (f.blocks_payment or f.severity == "error")
            for f in self.findings
        )
        has_doc_warn = any(f.finding_nature == "operational_discrepancy" for f in self.findings)
        if has_doc_block:
            return "HIGH"
        if has_doc_warn:
            return "MEDIUM"
        return "LOW"

    @property
    def errors(self)    -> list[UCPFinding]: return [f for f in self.findings if f.severity == "error"]
    @property
    def warnings(self)  -> list[UCPFinding]: return [f for f in self.findings if f.severity == "warning"]
    @property
    def infos(self)     -> list[UCPFinding]: return [f for f in self.findings if f.severity == "info"]
    @property
    def blocking(self)  -> list[UCPFinding]: return [f for f in self.findings if f.blocks_payment]

    def summary(self) -> str:
        decision = evaluate_ucp_decision(self.findings, self.payment_blocked)
        _decision_icon = {
            "REJECT":            "🔴 REJECT",
            "CONDITIONAL_HOLD":  "🟠 CONDITIONAL HOLD",
            "REVIEW":            "🟡 REVIEW REQUIS",
            "ACCEPT_WITH_FLAGS": "🟢 ACCEPTÉ (avec alertes)",
            "ACCEPT":            "✅ ACCEPTÉ",
        }.get(decision, decision)

        violations   = [f for f in self.findings if f.finding_nature == "true_violation"]
        contrat_mods = [f for f in self.findings if f.finding_nature == "contractual_modification"]
        op_disc      = [f for f in self.findings if f.finding_nature == "operational_discrepancy"]

        _pay_status = (
            "🔴 BLOQUÉ — violation réelle" if self.payment_blocked
            else ("🟠 SOUS RÉSERVE — modification contractuelle (Art.1 UCP)" if self.contractual_hold
            else "✅ NON BLOQUÉ")
        )

        lines = [
            "╔══ RAPPORT CONFORMITÉ UCP 600 v4 ════════════════════════════╗",
            f"  LC             : {self.lc_number}",
            f"  Type SWIFT     : {self.swift_type}",
            f"  Règles         : {self.ucp_version}",
            f"  Score global   : {self.global_score}/100",
            f"  Décision UCP   : {_decision_icon}",
            f"  Paiement       : {_pay_status}",
            f"  Statut légal   : {self.legal_compliance_status}",
            f"  Risque contrac.: {self.contractual_risk_level}",
            f"  Risque doc.    : {self.documentary_discrepancy_risk}",
            "  ─────────────────────────────────────────────────────────────",
            f"  Violations UCP réelles       : {len(violations)}",
            f"  Modifications contractuelles : {len(contrat_mods)}  (Art.1 UCP autorisé)",
            f"  Discordances documentaires   : {len(op_disc)}",
            f"  Bloquants paiement           : {len(self.blocking)}",
            f"  Impact HIGH / MEDIUM / LOW   : "
            f"{len([f for f in self.findings if f.impact_level=='HIGH'])} / "
            f"{len([f for f in self.findings if f.impact_level=='MEDIUM'])} / "
            f"{len([f for f in self.findings if f.impact_level=='LOW'])}",
        ]
        if self.exclusions:
            lines.append(f"  Articles exclus (Art.1) : {', '.join(self.exclusions.keys())}")
        lines.append("╚══════════════════════════════════════════════════════════════╝")

        # Violations UCP réelles
        if violations:
            lines.append(f"\n{'='*60}")
            lines.append("🔴 VIOLATIONS UCP RÉELLES (nécessitent correction)")
            lines.append('='*60)
            for f in sorted(violations, key=lambda x: x.weight, reverse=True):
                icon = "🔒" if f.blocks_payment else "❌"
                lines.append(f"  {icon} [{f.impact_level}] [{f.clause_type}] [{f.article}]")
                lines.append(f"     {f.message}")
                if f.recommendation:
                    lines.append(f"     → {f.recommendation}")

        # Discordances documentaires
        if op_disc:
            lines.append(f"\n{'='*60}")
            lines.append("🟠 DISCORDANCES DOCUMENTAIRES")
            lines.append('='*60)
            for f in sorted(op_disc, key=lambda x: x.weight, reverse=True):
                lines.append(f"  [{f.impact_level}] [{f.article}] {f.message}")
                if f.recommendation:
                    lines.append(f"     → {f.recommendation}")

        # Modifications contractuelles Art.1
        if contrat_mods:
            lines.append(f"\n{'='*60}")
            lines.append("🟡 MODIFICATIONS CONTRACTUELLES — Art.1 UCP 600")
            lines.append("   (dérogations acceptées entre parties, pas des violations UCP)")
            lines.append('='*60)
            for f in sorted(contrat_mods, key=lambda x: x.weight, reverse=True):
                icon = "🟠" if f.blocks_payment else "🟡"
                lines.append(f"  {icon} [{f.impact_level}] [{f.clause_type}] [{f.article}]")
                lines.append(f"     {f.message}")
                if f.recommendation:
                    lines.append(f"     → {f.recommendation}")

        # Alertes informationnelles
        infos = [f for f in self.findings if f.severity == "info"]
        if infos:
            lines.append(f"\n{'='*60}")
            lines.append("ℹ INFO")
            lines.append('='*60)
            for f in infos:
                lines.append(f"  [{f.article}] {f.message}")

        return "\n".join(lines)

    def ucp_clause_analysis_payload(self) -> dict:
        """Vue structurée clause ↔ article (sémantique) + findings contractuels."""
        sem = self.semantic or {}
        aligned = []
        for row in sem.get("per_clause") or []:
            if not isinstance(row, dict):
                continue
            aligned.append(
                {
                    "clause_index": row.get("clause_index"),
                    "article": row.get("article"),
                    "article_title": row.get("article_title"),
                    "relation": row.get("relation"),
                    "similarity": row.get("similarity"),
                    "confidence": row.get("confidence"),
                }
            )
        contractual = [
            {
                "article": f.article,
                "code": f.code,
                "severity": f.severity,
                "blocks_payment": f.blocks_payment,
                "blocks_payment_reason": f.blocks_payment_reason,
                "finding_nature": f.finding_nature,
                "message": f.message,
            }
            for f in self.findings
        ]
        return {
            "aligned_clauses": aligned,
            "contractual_findings": contractual,
        }

    def to_dict(self) -> dict:
        ucp_decision = evaluate_ucp_decision(self.findings, self.payment_blocked)
        violations   = [f for f in self.findings if f.finding_nature == "true_violation"]
        contrat_mods = [f for f in self.findings if f.finding_nature == "contractual_modification"]
        op_disc      = [f for f in self.findings if f.finding_nature == "operational_discrepancy"]
        return {
            "lc_number":        self.lc_number,
            "swift_type":       self.swift_type,
            "ucp_version":      self.ucp_version,
            "global_score":     self.global_score,
            "compliant":        self.compliant,
            "payment_blocked":  self.payment_blocked,
            "contractual_hold": self.contractual_hold,
            "ucp_decision":     ucp_decision,
            "processed_at":     self.processed_at,
            "exclusions":       self.exclusions,
            "exclusion_types":  self.exclusion_types,
            "cascade_log":      self.cascade_log,
            "semantic":         self.semantic,
            "ucp_clause_analysis": self.ucp_clause_analysis_payload(),
            "legal_compliance_status": self.legal_compliance_status,
            "contractual_risk_level":  self.contractual_risk_level,
            "documentary_discrepancy_risk": self.documentary_discrepancy_risk,
            "summary": {
                "errors":                len(self.errors),
                "warnings":              len(self.warnings),
                "infos":                 len(self.infos),
                "blocking":              len(self.blocking),
                "true_violations":       len(violations),
                "contractual_modifications": len(contrat_mods),
                "operational_discrepancies": len(op_disc),
                "high_impact":  len([f for f in self.findings if f.impact_level == "HIGH"]),
                "medium_impact": len([f for f in self.findings if f.impact_level == "MEDIUM"]),
                "low_impact":   len([f for f in self.findings if f.impact_level == "LOW"]),
            },
            "findings":   [f.to_dict() for f in self.findings],
            "ucp_blocking_violations": ucp_blocking_violations_list(self.findings),
            "categories": {
                k: {
                    "score":        round(v.score),
                    "max_score":    round(v.max_score),
                    "articles_ok":  v.articles_ok,
                    "articles_nok": v.articles_nok,
                }
                for k, v in self.category_scores.items()
            },
        }


# ══════════════════════════════════════════════════════════════════════
# PARSER D'EXCLUSIONS v3
# ══════════════════════════════════════════════════════════════════════

class ClauseExclusionParser:
    """
    Détecte les articles UCP exclus/modifiés.
    v3 : contexte étendu (±80 chars), typage sémantique intégré.
    """

    _EXCLUSION_RE = re.compile(
        r"[Aa]rt(?:icle)?\.?\s*(\d{1,2}[a-z]?)"
        r"(?:\s+of\s+UCP\s*(?:600|\(2007\s+REVISION\)|LATEST\s+VERSION)?)?"
        r"\s+(?:is\s+)?(?:not\s+applicable|excluded|is\s+excluded)",
        re.IGNORECASE
    )

    _COLLATED_RE = re.compile(
        r"\barticle\s*(\d{1,2}[a-z])\b",
        re.IGNORECASE
    )

    @classmethod
    def parse(cls, full_text: str, clauses: list[dict]) -> dict[str, str]:
        exclusions: dict[str, str] = {}
        sources = [c.get("text", "") for c in clauses]
        sources.append(full_text)

        for source in sources:
            for m in cls._EXCLUSION_RE.finditer(source):
                art_num = m.group(1).lower()
                art_key = f"Art.{art_num}"
                if art_key not in exclusions:
                    # v3 : contexte étendu ±80 chars
                    ctx_start = max(0, m.start() - 40)
                    ctx_end   = min(len(source), m.end() + 40)
                    exclusions[art_key] = source[ctx_start:ctx_end].strip()

            for m in cls._COLLATED_RE.finditer(source):
                art_num = m.group(1).lower()
                art_key = f"Art.{art_num}"
                ctx     = source[max(0, m.start()-10):m.end()+40]
                if re.search(r'not\s+applicable|excluded', ctx, re.IGNORECASE):
                    if art_key not in exclusions:
                        exclusions[art_key] = ctx.strip()

        return exclusions


# ══════════════════════════════════════════════════════════════════════
# TYPAGE SÉMANTIQUE DES EXCLUSIONS (partagé avec semantic analyzer)
# ══════════════════════════════════════════════════════════════════════

class ExclusionSemanticTyper:
    """
    BUG #2 FIX définitif : classifie chaque exclusion.
    Art.18b "not applicable" = RESTRICTIVE (renforce la contrainte).
    """
    _ARTICLE_TYPES: dict[str, str] = {
        "Art.12b":  "payment",
        "Art.14c":  "permissive",
        "Art.14k":  "prohibitive",
        "Art.18b":  "restrictive",
        "Art.20":   "prohibitive",
        "Art.20c":  "prohibitive",
        "Art.37":   "charges",
        "Art.37c":  "charges",
        "Art.38":   "restrictive",
        "Art.38i":  "restrictive",
    }
    _RESTRICTIVE = re.compile(r"not\s+acceptable|must\s+not|shall\s+not|is\s+prohibited|is\s+not\s+allowed", re.I)
    _PROHIBITIVE = re.compile(r"prohibited|forbidden|not\s+allowed|interdit", re.I)
    _PERMISSIVE  = re.compile(r"may\s+be|is\s+permitted|is\s+allowed", re.I)

    @classmethod
    def classify(cls, article_ref: str, clause_text: str) -> str:
        known = cls._ARTICLE_TYPES.get(article_ref)
        if known:
            return known
        text = clause_text or ""
        if cls._PROHIBITIVE.search(text):  return "prohibitive"
        if cls._RESTRICTIVE.search(text):  return "restrictive"
        if cls._PERMISSIVE.search(text):   return "permissive"
        return "neutral"

    @classmethod
    def get_severity(cls, exclusion_type: str) -> str:
        return {
            "restrictive":  "warning",
            "permissive":   "info",
            "prohibitive":  "error",
            "payment":      "warning",
            "charges":      "warning",
            "neutral":      "info",
        }.get(exclusion_type, "info")


# ══════════════════════════════════════════════════════════════════════
# MOTEUR DE RÈGLES UCP 600 v3
# ══════════════════════════════════════════════════════════════════════

class UCP600RuleEngine:

    ARTICLE_TITLES = {
        "Art.6":   "Disponibilité, Date d'expiration et Lieu",
        "Art.7":   "Obligations banque émettrice",
        "Art.8":   "Obligations banque confirmatrice",
        "Art.12b": "Désignation — pré-paiement différé",
        "Art.14":  "Norme d'examen des documents",
        "Art.14c": "Documents non prescrits par le crédit",
        "Art.14k": "Expéditeur/chargeur sur documents",
        "Art.16":  "Documents non conformes",
        "Art.17":  "Documents originaux",
        "Art.18":  "Facture commerciale",
        "Art.18b": "Facture commerciale — montant",
        "Art.19":  "Transport multimodal",
        "Art.20":  "Connaissement",
        "Art.20c": "Connaissement — transbordement",
        "Art.22":  "Connaissement charter-partie",
        "Art.23":  "Document de transport aérien",
        "Art.26":  "Messageries et courrier postal",
        "Art.28":  "Police ou certificat d'assurance",
        "Art.29":  "Prorogation de la date d'expiration",
        "Art.30":  "Tolérance montant/quantité",
        "Art.31":  "Tirages ou envois partiels",
        "Art.37":  "Décharge — partie mandatée",
        "Art.37c": "Commissions et frais",
        "Art.38":  "Crédits transférables",
        "Art.38i": "Transfert — factures du premier bénéficiaire",
        "Art.39":  "Cession du produit",
    }

    def _f(
        self, art: str, code: str, severity: str, cat: str, msg: str,
        detail: str = "", clause: str = "", rec: str = "",
        confidence: float = 1.0, excl_type: str = "",
        nature: str = "true_violation", block_reason: str = "",
    ) -> UCPFinding:
        """Factory UCPFinding v3."""
        return UCPFinding(
            article=art, article_title=self.ARTICLE_TITLES.get(art, art),
            code=code, severity=severity, category=cat,
            message=msg, detail=detail, clause_text=clause,
            recommendation=rec, confidence=confidence,
            exclusion_type=excl_type,
            finding_nature=nature,
            blocks_payment_reason=block_reason,
        )

    # ── Art.6 ─────────────────────────────────────────────────────────

    def check_article_6(self, entities: dict, lc_criteria: dict, exclusions: dict) -> list[UCPFinding]:
        findings = []
        art = "Art.6"

        expiry_str = entities.get("expiry_date")
        if not expiry_str:
            findings.append(self._f(art, "ART6_NO_EXPIRY", "error", "compliance",
                "Date d'expiration absente — Art.6 obligatoire",
                rec="Vérifier :31D:"))
        else:
            try:
                expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                today  = date.today()
                if expiry < today:
                    findings.append(self._f(art, "ART6_EXPIRED", "warning", "compliance",
                        f"LC expirée le {expiry_str} — document historique ou test",
                        detail="Date expiration antérieure à aujourd'hui"))
                issue_str = entities.get("issue_date")
                if issue_str:
                    issue = datetime.strptime(issue_str, "%Y-%m-%d").date()
                    validity_days = (expiry - issue).days
                    if 0 < validity_days < 7:
                        findings.append(self._f(art, "ART6_SHORT_VALIDITY", "warning", "compliance",
                            f"Validité très courte: {validity_days} jours (recommandé ≥ 21j)",
                            rec="Vérifier si intentionnel"))
            except ValueError:
                pass

        if not entities.get("expiry_place"):
            findings.append(self._f(art, "ART6_NO_EXPIRY_PLACE", "warning", "compliance",
                "Lieu d'expiration non précisé (Art.6d)", rec="Ajouter pays/ville dans :31D:"))

        available_by = lc_criteria.get("available_by") or entities.get("available_by")
        if not available_by:
            findings.append(self._f(art, "ART6_NO_AVAILABLE_BY", "warning", "compliance",
                "Modalité de réalisation (BY PAYMENT/ACCEPTANCE) non extraite",
                detail="Champ :41A: présent mais modalité non parsée",
                rec="Vérifier :41A:"))

        pres = lc_criteria.get("presentation_period_days") or int(entities.get("presentation_period") or 0)
        if pres and pres > 21:
            findings.append(self._f(art, "ART6_LONG_PRESENTATION", "info", "document",
                f"Délai présentation étendu: {pres}j (standard = 21j)"))

        return findings

    # ── Art.14 ────────────────────────────────────────────────────────

    def check_article_14(self, entities: dict, lc_criteria: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []

        if "Art.14c" in exclusions:
            findings.append(self._f("Art.14c", "ART14C_EXCLUDED", "warning", "document",
                "Art.14c exclu — documents non prescrits fournis seront examinés",
                detail=exclusions["Art.14c"], clause=exclusions["Art.14c"],
                rec="S'assurer que la banque n'examine pas les docs non demandés",
                excl_type="permissive", nature="contractual_modification"))

        if "Art.14k" in exclusions:
            findings.append(self._f("Art.14k", "ART14K_EXCLUDED", "warning", "document",
                "Art.14k exclu — l'expéditeur/chargeur DOIT être le bénéficiaire",
                detail=exclusions.get("Art.14k", ""),
                rec="Vérifier que tous les docs de transport indiquent le bénéficiaire comme expéditeur",
                excl_type="prohibitive", nature="contractual_modification"))

        docs = lc_criteria.get("documents", [])
        if not docs:
            findings.append(self._f("Art.14", "ART14_NO_DOCS", "warning", "document",
                "Liste documents requis (:46A:) vide ou non parsée",
                rec="Vérifier :46A:"))

        return findings

    # ── Art.16 ────────────────────────────────────────────────────────

    def check_article_16(self, entities: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []
        _FEE_RE = re.compile(
            r"(?:reservation\s+lifting|lifting\s+fee|levée\s+de\s+réserve)"
            r".*?(?:EUR|USD|GBP|TND)\s*(\d[\d\s,.]*)",
            re.IGNORECASE | re.DOTALL
        )
        for clause in clauses:
            text = clause.get("text", "")
            m = _FEE_RE.search(text)
            if m:
                fee_raw = re.sub(r'\s', '', m.group(1))
                findings.append(self._f("Art.16", "ART16_RESERVATION_FEE", "warning", "compliance",
                    f"Frais levée réserve: {fee_raw} par jeu de documents",
                    detail="Frais déduit même si LC prévoit charges pour payeur",
                    clause=text[:150],
                    rec="Informer le bénéficiaire — déduit du montant payable"))
                break
        return findings

    # ── Art.17 ────────────────────────────────────────────────────────

    def check_scanned_docs(self, clauses: list[dict]) -> list[UCPFinding]:
        findings = []
        _SCAN_RE = re.compile(r"scanned\s+document|scanned\s+signature|bearing\s+scanned", re.I)
        for clause in clauses:
            if _SCAN_RE.search(clause.get("text", "")):
                findings.append(self._f("Art.17", "ART17_NO_SCANNED", "error", "document",
                    "Documents scannés INTERDITS — même portant mention ORIGINAL",
                    clause=clause.get("text", "")[:150],
                    rec="Tous docs: originaux physiques, signature manuscrite encre bleue + tampon"))
                break
        return findings

    def check_signature_requirements(self, clauses: list[dict]) -> list[UCPFinding]:
        findings = []
        _SIG_RE = re.compile(r"signed\s+in\s+handwriting.*?blue\s+ink|blue\s+ink\s+ball\s+pen", re.I | re.DOTALL)
        for clause in clauses:
            if _SIG_RE.search(clause.get("text", "")):
                findings.append(self._f("Art.17", "ART17_BLUE_INK_SIGNATURE", "info", "document",
                    "Signature manuscrite encre bleue + tampon obligatoire sur TOUS les documents",
                    clause=clause.get("text", "")[:150],
                    rec="Informer bénéficiaire : signer à l'encre bleue + tampon + nom + fonction"))
                break
        return findings

    # ── Art.18 ────────────────────────────────────────────────────────

    def check_article_18(self, entities: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []

        if "Art.18b" in exclusions:
            # BUG #2 FIX : Art.18b exclu = restriction RENFORCÉE, pas permissive
            amount   = entities.get("amount", "?")
            currency = entities.get("currency", "")
            findings.append(self._f("Art.18b", "ART18B_STRICT_LIMIT", "warning", "document",
                f"Art.18b exclu — LIMITE STRICTE: facture ≤ {currency} {amount} "
                f"(pas de tolérance 5% UCP Art.30)",
                detail="Contrairement à UCP standard, aucune tolérance sur le montant",
                clause=exclusions.get("Art.18b", ""),
                rec=f"Informer bénéficiaire: facture EXACTEMENT ≤ {currency} {amount}",
                excl_type="restrictive", nature="contractual_modification"))

        goods = entities.get("goods_description", "")
        if goods and len(goods) < 10:
            findings.append(self._f("Art.18", "ART18_VAGUE_GOODS", "warning", "document",
                "Description marchandise trop vague dans :45A:",
                rec="La facture doit reprendre exactement la description :45A:"))

        return findings

    def check_third_party_invoice(self, clauses: list[dict]) -> list[UCPFinding]:
        findings = []
        _THIRD_RE = re.compile(r"name\s+of\s+a\s+third\s+party|formerly\s+known\s+as", re.I)
        for clause in clauses:
            if _THIRD_RE.search(clause.get("text", "")):
                findings.append(self._f("Art.18", "ART18_NO_THIRD_PARTY_INVOICE", "error", "document",
                    "Facture au nom d'un tiers ou 'formerly known as' INTERDITE",
                    clause=clause.get("text", "")[:150],
                    rec="Facture au nom exact du bénéficiaire du crédit uniquement",
                    nature="operational_discrepancy",
                    block_reason="documentary_discrepancy"))
                break
        return findings

    # ── Art.19 — Transport multimodal (NEW v3) ────────────────────────

    def check_article_19(self, entities: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        """Art.19 — Transport multimodal — NOUVEAU v3."""
        findings = []
        _MULTI_RE = re.compile(r"multimodal|combined\s+transport|through\s+bill\s+of\s+lading", re.I)
        _MULTI_RESTRICT = re.compile(
            r"multimodal.*?(?:not\s+acceptable|prohibited|not\s+allowed)"
            r"|(?:not\s+acceptable|prohibited).*?multimodal",
            re.I | re.DOTALL
        )
        for clause in clauses:
            text = clause.get("text", "")
            if _MULTI_RESTRICT.search(text):
                findings.append(self._f("Art.19", "ART19_MULTIMODAL_RESTRICTION", "warning", "document",
                    "Transport multimodal restreint par clause — BL standard seul accepté",
                    clause=text[:150],
                    rec="Vérifier que le doc de transport n'est pas un BL multimodal"))
                break

        # Cohérence Incoterm vs transport
        incoterm = entities.get("incoterm", "")
        if incoterm in ("FCA", "CPT", "CIP") and not any(
            _MULTI_RE.search(c.get("text", "")) for c in clauses
        ):
            findings.append(self._f("Art.19", "ART19_INCOTERM_MULTIMODAL_MISMATCH", "info", "document",
                f"Incoterm {incoterm} typiquement multimodal — vérifier :46A: (Art.19 applicable)",
                confidence=0.75,
                rec="S'assurer que le doc de transport accepté couvre le transport multimodal"))

        return findings

    # ── Art.20 — Connaissement v3 ─────────────────────────────────────

    def check_article_20(self, entities: dict, lc_criteria: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []

        # BUG #1 FIX : contradiction tag :43T: vs clause textuelle (UCP Art.5)
        _PROHIBIT_RE = re.compile(r"tranship(?:ment)?\s+(?:is\s+)?prohibit(?:ed)?|no\s+tranship", re.I)
        tag_allows = (entities.get("transhipment", "") or "").upper() == "ALLOWED"
        clause_prohibits = any(_PROHIBIT_RE.search(c.get("text", "")) for c in clauses)

        if clause_prohibits and tag_allows:
            findings.append(self._f("Art.20c", "ART20C_CLAUSE_OVERRIDES_TAG", "error", "shipment",
                "CONTRADICTION Art.5 : :43T:=ALLOWED mais clause interdit le transbordement — la clause prime",
                detail="UCP 600 Art.5 : le crédit documentaire prime. La clause textuelle annule le tag :43T:",
                rec="Corriger :43T: → 'NOT ALLOWED' OU supprimer la clause de prohibition"))
        elif "Art.20c" in exclusions or "Art.20" in exclusions:
            if "NOT ALLOWED" in (entities.get("transhipment", "") or "").upper() or clause_prohibits:
                findings.append(self._f("Art.20c", "ART20C_EXCLUDED", "warning", "shipment",
                    "Art.20 C ii exclu — transbordement INTERDIT même en containers",
                    detail="Clause standard UCP permet transbordement en containers — exclusion le supprime",
                    clause=exclusions.get("Art.20c", exclusions.get("Art.20", "")),
                    rec="BL direct (port à port) requis — vérifier le BL final",
                    nature="contractual_modification"))

        # BL blank back / short form
        _BL_RE = re.compile(r"blank\s+back.*?bill\s+of\s+lading|short\s+form\s+bill\s+of\s+lading", re.I)
        for clause in clauses:
            if _BL_RE.search(clause.get("text", "")):
                findings.append(self._f("Art.20", "ART20_NO_BLANK_BACK", "error", "document",
                    "BL blank back et short form bill of lading INTERDITS",
                    clause=clause.get("text", "")[:150],
                    rec="Exiger BL recto-verso complet avec conditions au verso"))
                break

        # LCL restrictions
        _LCL_RE = re.compile(r"LCL.*?(?:CY/CFS|LCL/FCL|CY/CFT)", re.I | re.DOTALL)
        for clause in clauses:
            if _LCL_RE.search(clause.get("text", "")):
                findings.append(self._f("Art.20", "ART20_LCL_RESTRICTION", "warning", "document",
                    "BL LCL : mentions CY/CFS, LCL/FCL et CY/CFT INTERDITES",
                    clause=clause.get("text", "")[:150],
                    rec="BL LCL ne doit mentionner ni CY/CFS ni LCL/FCL ni CY/CFT"))
                break

        # Notify party supplémentaire
        _NOTIFY_RE = re.compile(r"additional\s+notify\s+party.*?prohibited|notify.*?other\s+than.*?required.*?prohibited", re.I | re.DOTALL)
        for clause in clauses:
            if _NOTIFY_RE.search(clause.get("text", "")):
                findings.append(self._f("Art.20", "ART20_NO_EXTRA_NOTIFY", "warning", "document",
                    "Notify party supplémentaire sur BL INTERDIT",
                    clause=clause.get("text", "")[:150],
                    rec="Le BL ne peut mentionner qu'un seul notify party conforme au crédit"))
                break

        return findings

    # ── Art.22 — Connaissement charter-partie (NEW v3) ────────────────

    def check_article_22(self, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        """Art.22 — Charter-partie — NOUVEAU v3."""
        findings = []
        _CHARTER_RE = re.compile(r"charter\s*[-\s]?party|charterparty|affrètement", re.I)
        _CHARTER_PROHIBIT = re.compile(r"charter.*?(?:not\s+acceptable|prohibited)|no\s+charter", re.I)
        for clause in clauses:
            text = clause.get("text", "")
            if _CHARTER_PROHIBIT.search(text):
                findings.append(self._f("Art.22", "ART22_CHARTER_PROHIBITED", "error", "document",
                    "Connaissement charter-partie REFUSÉ par clause",
                    clause=text[:150],
                    rec="Exiger un connaissement ordinaire — pas de charter-partie"))
                break
        return findings

    # ── Art.23 — Document transport aérien (NEW v3) ───────────────────

    def check_article_23(self, entities: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        """Art.23 — AWB — NOUVEAU v3."""
        findings = []
        _AWB_RE  = re.compile(r"airway\s+bill|AWB|air\s+waybill|lettre\s+de\s+transport\s+aérien", re.I)
        _AWB_RESTRICT = re.compile(r"(?:airway\s+bill|AWB).*?(?:not\s+acceptable|prohibited)|no\s+(?:airway\s+bill|AWB)", re.I | re.DOTALL)

        incoterm = entities.get("incoterm", "")
        for clause in clauses:
            text = clause.get("text", "")
            if _AWB_RESTRICT.search(text):
                findings.append(self._f("Art.23", "ART23_AWB_RESTRICTION", "warning", "document",
                    "Airway Bill restreint par clause — vérifier mode de transport",
                    clause=text[:150],
                    rec="Confirmer mode transport maritime/terrestre uniquement"))
                break

        # Si Incoterm suggère aérien mais pas de mention AWB dans docs
        if incoterm in ("FCA", "CPT", "CIP"):
            has_awb_mention = any(
                _AWB_RE.search(c.get("text", "")) for c in clauses
            )
            # Pas d'alerte ici — juste une info si aérien possible
            if has_awb_mention:
                findings.append(self._f("Art.23", "ART23_AWB_POSSIBLE", "info", "document",
                    f"Incoterm {incoterm} + mention AWB — Art.23 applicable si transport aérien",
                    confidence=0.70,
                    rec="Vérifier que le AWB est acceptable selon :46A:"))
        return findings

    # ── Art.26 — Messagerie / courrier postal (NEW v3) ────────────────

    def check_article_26(self, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        """Art.26 — Messagerie — NOUVEAU v3."""
        findings = []
        _COURIER_RE = re.compile(r"courier|post\s+parcel|messagerie|colissimo|DHL|FedEx|UPS", re.I)
        _COURIER_RESTRICT = re.compile(r"courier.*?(?:not\s+acceptable|prohibited)|no\s+courier", re.I | re.DOTALL)
        for clause in clauses:
            text = clause.get("text", "")
            if _COURIER_RESTRICT.search(text):
                findings.append(self._f("Art.26", "ART26_COURIER_RESTRICTED", "warning", "document",
                    "Documents de messagerie/courrier non acceptés — Art.26",
                    clause=text[:150],
                    rec="Exiger document de transport standard (BL ou AWB)"))
                break
        return findings

    # ── Art.28 — Assurance ────────────────────────────────────────────

    def check_article_28(self, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []
        _FRANCHISE_RE = re.compile(r"franchise|excess\s+deductible|deductible", re.I)
        _WARRANTED_RE = re.compile(r"warranted\s+(?:professionally\s+packed|safe\s+at\s+time)", re.I)
        _RESTRICT_RE  = re.compile(r"restriction\s+clause|institute\s+(?:cargo|war\s+risks|strikes)", re.I)
        # NEW v3 : valeur minimum assurance
        _MIN_COVER_RE = re.compile(r"110\s*%|minimum\s+(?:cover|insurance|value)", re.I)

        for clause in clauses:
            text = clause.get("text", "")
            if _FRANCHISE_RE.search(text):
                findings.append(self._f("Art.28", "ART28_NO_FRANCHISE", "error", "document",
                    "Assurance : franchise et excédent déductible INTERDITS",
                    clause=text[:150],
                    rec="Exiger police sans franchise ni excédent déductible"))
            if _WARRANTED_RE.search(text):
                findings.append(self._f("Art.28", "ART28_NO_WARRANTED", "warning", "document",
                    "Assurance : clauses 'warranted professionally packed' et 'warranted safe' INTERDITES",
                    clause=text[:150],
                    rec="Vérifier que la police ne contient aucune de ces clauses"))
            if _RESTRICT_RE.search(text):
                findings.append(self._f("Art.28", "ART28_NO_COVERAGE_RESTRICTION", "error", "document",
                    "Assurance : restriction couverture Institute Cargo Clauses INTERDITE",
                    clause=text[:150],
                    rec="Police doit couvrir sans restriction ICC, Institute War Risks, Institute Strikes"))

        return findings

    # ── Art.30 — Tolérance ────────────────────────────────────────────

    def check_article_30(self, entities: dict, lc_criteria: dict, clauses: list[dict]) -> list[UCPFinding]:
        findings = []
        tol_plus  = entities.get("tolerance_plus",  "0")
        tol_minus = entities.get("tolerance_minus", "0")
        try:
            t_plus  = float(tol_plus  or 0)
            t_minus = float(tol_minus or 0)
        except (ValueError, TypeError):
            t_plus = t_minus = 0.0

        _QTY_RE = re.compile(
            r"quantity\s+greater\s+than.*?not\s+acceptable"
            r"|greater\s+than.*?LC\s+conditions.*?not\s+acceptable",
            re.I | re.DOTALL
        )
        qty_prohibited = any(_QTY_RE.search(c.get("text", "")) for c in clauses)

        if qty_prohibited and (t_plus > 0 or t_minus > 0):
            findings.append(self._f("Art.30", "ART30_TOLERANCE_CONFLICT", "error", "compliance",
                f"Contradiction: tolérance {t_plus:+.0f}%/{t_minus:+.0f}% dans :39A: "
                f"mais clause interdit quantité supérieure",
                rec="Aligner :39A: (0/0) avec la clause de quantité stricte"))
        elif qty_prohibited and t_plus == 0 and t_minus == 0:
            findings.append(self._f("Art.30", "ART30_ZERO_TOLERANCE", "info", "compliance",
                "Tolérance quantité 0/0 — quantité exacte exigée (confirmée par clause)"))

        return findings

    # ── Art.37 — Frais ────────────────────────────────────────────────

    def check_article_37(self, entities: dict, lc_criteria: dict, exclusions: dict) -> list[UCPFinding]:
        findings = []
        if "Art.37c" in exclusions or "Art.37" in exclusions:
            charges = entities.get("charges") or lc_criteria.get("charges_raw", "")
            if charges and "BENEFICIARY" in charges.upper():
                findings.append(self._f("Art.37c", "ART37C_EXCLUDED_CHARGES_BENEFICIARY", "warning", "compliance",
                    "Art.37C exclu — TOUS les frais hors Tunisie à charge bénéficiaire",
                    detail=exclusions.get("Art.37c", exclusions.get("Art.37", "")),
                    rec="Informer bénéficiaire: frais banc. SWIFT, commission, levée réserve → débités",
                    excl_type="charges", nature="contractual_modification"))
        return findings

    # ── Art.38 — Crédits transférables ────────────────────────────────

    def check_article_38(self, entities: dict, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []
        if "Art.38i" in exclusions or "Art.38" in exclusions:
            _INV_RE = re.compile(
                r"invoices.*?first\s+beneficiary|established\s+by\s+the\s+first\s+beneficiary",
                re.I | re.DOTALL
            )
            for clause in clauses:
                if _INV_RE.search(clause.get("text", "")):
                    benef = entities.get("beneficiary") or "le premier bénéficiaire"
                    findings.append(self._f("Art.38i", "ART38I_EXCLUDED", "warning", "compliance",
                        "Art.38i exclu — factures DOIVENT être établies par le premier bénéficiaire",
                        detail="Transfert LC n'autorise pas second bénéficiaire à émettre factures",
                        clause=clause.get("text", "")[:150],
                        rec=f"Vérifier: toutes factures signées et émises par {benef}",
                        excl_type="restrictive",
                        nature="contractual_modification",
                        block_reason="structural_transfer_constraint"))
                    break
        return findings

    # ── Art.12b — Pré-paiement ────────────────────────────────────────

    # ── Art.10 — Amendements ─────────────────────────────────────────────

    def check_article_10(self, clauses: list[dict], entities: dict) -> list[UCPFinding]:
        """Art.10 — Amendments: partial acceptance and auto-acceptance clauses."""
        findings = []
        _PARTIAL_AMEND = re.compile(
            r"partial\s+acceptance\s+of\s+amendment"
            r"|accept\s+part\s+of\s+(?:the\s+)?amendment"
            r"|partially\s+accept\s+amendment",
            re.I
        )
        _AUTO_AMEND = re.compile(
            r"automatically\s+(?:applies|incorporated|accepted|deemed\s+accepted)"
            r"|tacit(?:ly)?\s+(?:accept|agree).*?amendment"
            r"|silence\s+constitutes\s+acceptance",
            re.I
        )
        _UNILATERAL_AMEND = re.compile(
            r"may\s+(?:unilaterally\s+)?amend\s+(?:the\s+)?credit\s+without"
            r"|amendment\s+without\s+(?:consent|agreement|acceptance)",
            re.I
        )
        for clause in clauses:
            text = clause.get("text", "")
            if _PARTIAL_AMEND.search(text):
                findings.append(self._f("Art.10", "ART10_PARTIAL_AMENDMENT", "error", "compliance",
                    "Acceptation partielle d'amendement INTERDITE — Art.10 UCP 600",
                    detail="Un amendement doit être accepté intégralement ou rejeté — toute acceptation partielle est nulle",
                    clause=text[:150],
                    rec="L'amendement doit être accepté en bloc ou formellement rejeté par écrit",
                    nature="true_violation"))
                break
            if _AUTO_AMEND.search(text):
                findings.append(self._f("Art.10", "ART10_AUTO_ACCEPTANCE_CLAUSE", "warning", "compliance",
                    "Clause d'acceptation automatique / tacite d'amendement — risque contractuel Art.10",
                    detail="UCP 600 Art.10 ne prévoit pas l'acceptation tacite. Une telle clause est une modification contractuelle.",
                    clause=text[:150],
                    rec="Vérifier que tout amendement a été expressément accepté par toutes les parties",
                    excl_type="restrictive", nature="contractual_modification"))
                break
            if _UNILATERAL_AMEND.search(text):
                findings.append(self._f("Art.10", "ART10_UNILATERAL_AMENDMENT", "error", "compliance",
                    "Clause d'amendement unilatéral sans consentement INTERDITE — Art.10 UCP 600",
                    detail="Tout amendement requiert l'accord de la banque émettrice, de la banque confirmatrice et du bénéficiaire",
                    clause=text[:150],
                    rec="Supprimer toute clause permettant l'amendement unilatéral",
                    nature="true_violation"))
                break
        return findings

    # ── Art.31 — Expéditions partielles ──────────────────────────────────

    def check_article_31(self, entities: dict, lc_criteria: dict) -> list[UCPFinding]:
        """Art.31 — Partial drawings/shipments."""
        findings = []
        partial = (
            (entities.get("partial_shipments") or lc_criteria.get("partial_shipments") or "")
        ).upper().strip()
        if partial == "PROHIBITED":
            findings.append(self._f("Art.31", "ART31_PARTIAL_PROHIBITED", "warning", "shipment",
                "Expéditions partielles INTERDITES — :43P: = PROHIBITED",
                detail="La totalité de la marchandise doit être expédiée en une seule livraison et présentée en un seul jeu de documents",
                rec="Vérifier que le BL et la facture couvrent la totalité de la commande — aucun tirage partiel autorisé",
                nature="true_violation"))
        return findings

    # ── Art.32 — Tirages par tranches ────────────────────────────────────

    def check_article_32(self, entities: dict, lc_criteria: dict) -> list[UCPFinding]:
        """Art.32 — Instalment drawings/shipments."""
        findings = []
        schedule = lc_criteria.get("instalment_schedule") or entities.get("instalment_schedule")
        if schedule:
            findings.append(self._f("Art.32", "ART32_INSTALMENT_SCHEDULE", "warning", "shipment",
                "Crédit à tirages par tranches — Art.32 : tout manquement clôture définitivement les tranches suivantes",
                detail="Si une tranche n'est pas utilisée dans son délai, le crédit cesse d'être disponible pour cette tranche et toutes les suivantes",
                rec="Surveiller strictement le calendrier des tirages — aucune prorogation automatique n'est prévue par UCP 600",
                nature="true_violation"))
        return findings

    # ── Art.39 — Cession du produit ──────────────────────────────────────

    def check_article_39(self, clauses: list[dict]) -> list[UCPFinding]:
        """Art.39 — Assignment of proceeds."""
        findings = []
        _PROHIBIT_RE = re.compile(
            r"assignment\s+of\s+proceeds.*?(?:not\s+allowed|prohibited|forbidden|not\s+permitted)"
            r"|no\s+assignment\s+of\s+proceeds"
            r"|cession\s+du\s+produit.*?(?:interdite|non\s+autorisée)",
            re.I | re.DOTALL
        )
        _ASSIGN_RE = re.compile(
            r"assignment\s+of\s+(?:the\s+)?proceeds|cession\s+du\s+produit|assignee\s+of\s+proceeds",
            re.I
        )
        for clause in clauses:
            text = clause.get("text", "")
            if _PROHIBIT_RE.search(text):
                findings.append(self._f("Art.39", "ART39_ASSIGNMENT_PROHIBITED", "warning", "compliance",
                    "Cession du produit INTERDITE par clause — Art.39 UCP 600",
                    detail="Art.39 UCP 600 autorise le bénéficiaire à céder son droit au produit, sauf clause contraire explicite",
                    clause=text[:150],
                    rec="Informer le bénéficiaire qu'il ne peut pas céder le produit du crédit à un tiers",
                    excl_type="prohibitive", nature="contractual_modification"))
                break
            if _ASSIGN_RE.search(text):
                findings.append(self._f("Art.39", "ART39_ASSIGNMENT_NOTED", "info", "compliance",
                    "Cession du produit mentionnée — vérifier notification formelle Art.39",
                    detail="Toute cession doit être formellement notifiée à la banque concernée",
                    clause=text[:150],
                    rec="S'assurer que la cession est notifiée par écrit à la banque désignée",
                    nature="true_violation"))
                break
        return findings

    def check_article_12b(self, clauses: list[dict], exclusions: dict) -> list[UCPFinding]:
        findings = []
        if "Art.12b" in exclusions:
            findings.append(self._f("Art.12b", "ART12B_EXCLUDED", "warning", "payment",
                "Art.12b exclu — pré-paiement engagement différé INTERDIT sans accord préalable",
                detail=exclusions.get("Art.12b", ""),
                rec="Demande pré-paiement/escompte doit être soumise banque émettrice",
                excl_type="payment", nature="contractual_modification"))
        return findings

    # ── Art.8 — Confirmation ──────────────────────────────────────────

    def check_confirmation(self, entities: dict, lc_criteria: dict) -> list[UCPFinding]:
        findings = []
        conf = entities.get("confirmation") or lc_criteria.get("confirmation", "")
        if conf and "CONFIRM" in conf.upper():
            findings.append(self._f("Art.8", "ART8_CONFIRMATION_REQUESTED", "info", "compliance",
                "Confirmation demandée (Art.8) — banque présentatrice doit confirmer le LC",
                rec="Banque présentatrice: endosser montant tirage au verso LC original + certifier frais payés"))
        return findings

    def check_article_14_documents_list(
        self, lc_criteria: dict,
    ) -> list[UCPFinding]:
        """Liste :46A: vide — seul contrôle Art.14 conservé en mode sémantique."""
        findings: list[UCPFinding] = []
        docs = lc_criteria.get("documents", [])
        if not docs:
            findings.append(self._f("Art.14", "ART14_NO_DOCS", "warning", "document",
                "Liste documents requis (:46A:) vide ou non parsée",
                rec="Vérifier :46A:"))
        return findings

    def check_structural_transhipment(
        self, entities: dict, clauses: list[dict],
    ) -> list[UCPFinding]:
        """Art.5 : contradiction :43T: vs clause (hors exclusions regex legacy)."""
        findings: list[UCPFinding] = []
        _PROHIBIT_RE = re.compile(
            r"tranship(?:ment)?\s+(?:is\s+)?prohibit(?:ed)?|no\s+tranship", re.I
        )
        tag_allows = (entities.get("transhipment", "") or "").upper() == "ALLOWED"
        clause_prohibits = any(_PROHIBIT_RE.search(c.get("text", "")) for c in clauses)
        if clause_prohibits and tag_allows:
            findings.append(self._f("Art.20c", "ART20C_CLAUSE_OVERRIDES_TAG", "error", "shipment",
                "CONTRADICTION Art.5 : :43T:=ALLOWED mais clause interdit le transbordement — la clause prime",
                detail="UCP 600 Art.5 : le crédit documentaire prime. La clause textuelle annule le tag :43T:",
                rec="Corriger :43T: → 'NOT ALLOWED' OU supprimer la clause de prohibition"))
        return findings


# ── Taxonomie métier : impact_level et clause_type ───────────────────

_HIGH_IMPACT_ARTICLES = frozenset({
    "Art.5", "Art.6", "Art.7", "Art.10", "Art.14", "Art.14k",
    "Art.17", "Art.18", "Art.20", "Art.20c", "Art.28", "Art.32",
    "Art.38", "Art.38i",
})
_MEDIUM_IMPACT_ARTICLES = frozenset({
    "Art.8", "Art.9", "Art.12b", "Art.14a", "Art.14c", "Art.16",
    "Art.18b", "Art.19", "Art.22", "Art.23", "Art.26", "Art.29",
    "Art.30", "Art.31", "Art.37", "Art.37c",
})
_LOW_IMPACT_ARTICLES = frozenset({"Art.1", "Art.33", "Art.35", "Art.39"})

_FINANCIAL_ARTS      = frozenset({
    "Art.7", "Art.8", "Art.12b", "Art.18", "Art.18b",
    "Art.37", "Art.37c", "Art.38", "Art.38i", "Art.39",
})
_TRANSPORT_ARTS      = frozenset({
    "Art.19", "Art.20", "Art.20c", "Art.22", "Art.23", "Art.26", "Art.31", "Art.32",
})
_AUTH_ARTS           = frozenset({"Art.17"})
_IDENTITY_ARTS       = frozenset({"Art.14k"})
_COMPLIANCE_ARTS     = frozenset({
    "Art.1", "Art.5", "Art.6", "Art.9", "Art.10", "Art.16",
    "Art.29", "Art.33", "Art.35",
})


def _compute_impact_level(f: "UCPFinding") -> str:
    """Calcule l'impact opérationnel réel selon nature + article + sévérité."""
    if f.finding_nature in {"true_violation", "operational_discrepancy"} and f.blocks_payment:
        return "HIGH"
    if f.finding_nature == "true_violation" and f.severity == "error":
        return "HIGH" if f.article in _HIGH_IMPACT_ARTICLES else "MEDIUM"
    if f.finding_nature == "contractual_modification" and "STRUCT" in f.code:
        return "HIGH"
    if f.finding_nature == "contractual_modification":
        return "HIGH" if f.article in _HIGH_IMPACT_ARTICLES else "MEDIUM"
    if f.finding_nature == "operational_discrepancy" and f.severity == "error":
        return "HIGH"
    if f.severity == "warning":
        return "LOW" if f.article in _LOW_IMPACT_ARTICLES else "MEDIUM"
    if f.severity == "info" or f.tier >= 4:
        return "LOW"
    return "MEDIUM"


def _compute_clause_type(f: "UCPFinding") -> str:
    """Détermine le type de clause selon l'article UCP."""
    if f.article in _FINANCIAL_ARTS:   return "financial"
    if f.article in _TRANSPORT_ARTS:   return "transport"
    if f.article in _AUTH_ARTS:        return "authentication"
    if f.article in _IDENTITY_ARTS:    return "identity"
    if f.article in _COMPLIANCE_ARTS:  return "compliance"
    return "document_format"


def _ucp_finding_from_dict(data: dict) -> UCPFinding:
    """Construit UCPFinding depuis la sortie du moteur sémantique ou hard_rules."""
    return UCPFinding(
        article=data["article"],
        article_title=data.get("article_title", data["article"]),
        code=data["code"],
        severity=data["severity"],
        category=data["category"],
        message=data["message"],
        detail=data.get("detail", ""),
        clause_text=data.get("clause_text", ""),
        recommendation=data.get("recommendation", ""),
        confidence=float(data.get("confidence", 0.8)),
        exclusion_type=data.get("exclusion_type", "semantic"),
        finding_nature=data.get("finding_nature", "true_violation"),
        blocks_payment_reason=data.get("blocks_payment_reason", ""),
        clause_type=data.get("clause_type", "document_format"),
        impact_level=data.get("impact_level", "MEDIUM"),
    )


# ══════════════════════════════════════════════════════════════════════
# VALIDATEUR PRINCIPAL v3
# ══════════════════════════════════════════════════════════════════════

class UCP600Validator:
    """
    Orchestre la validation UCP 600 v3.
    Input  : résultat de pipeline (dict JSON)
    Output : UCPValidationReport v3
    """

    # Seuils de compliance par type de LC
    _COMPLIANCE_THRESHOLDS = {
        "MT700": {"min_score": 70, "max_errors": 0},
        "MT720": {"min_score": 70, "max_errors": 0},
        "MT760": {"min_score": 65, "max_errors": 0},
        "MT790": {"min_score": 80, "max_errors": 0},
        "MT799": {"min_score": 90, "max_errors": 0},
    }

    def __init__(self, custom_weights: Optional[dict] = None) -> None:
        self._engine        = UCP600RuleEngine()
        self._excl_parser   = ClauseExclusionParser()
        self._excl_typer    = ExclusionSemanticTyper()
        self._weight_reg    = RuleWeightRegistry(custom_weights)
        self._cascade       = CascadePenaltyEngine()
        self._semantic_ucp: Any = None

    def _get_semantic_engine(self) -> Any:
        if self._semantic_ucp is None:
            from .swift_ucp_semantic_engine import SemanticUCPEngine

            self._semantic_ucp = SemanticUCPEngine()
        return self._semantic_ucp

    def _structural_ucp_findings(
        self,
        entities: dict,
        lc: dict,
        clauses: list,
        exclusions: dict,
    ) -> list[UCPFinding]:
        """Checks SWIFT structurés + cross-field + articles systématiques."""
        from .swift_ucp_hard_rules import evaluate_cross_field_rules
        out: list[UCPFinding] = []
        out += self._engine.check_article_6(entities, lc, exclusions)
        out += self._engine.check_article_14_documents_list(lc)
        out += self._engine.check_structural_transhipment(entities, clauses)
        out += self._engine.check_article_30(entities, lc, clauses)
        out += self._engine.check_article_31(entities, lc)
        out += self._engine.check_article_32(entities, lc)
        out += self._engine.check_confirmation(entities, lc)
        # Cross-field SWIFT MT700
        cross_fields = {
            "expiry_date":           entities.get("expiry_date", ""),
            "latest_shipment_date":  entities.get("latest_shipment_date", "") or entities.get("shipment_date", ""),
            "issue_date":            entities.get("issue_date", ""),
            "presentation_period":   str(entities.get("presentation_period") or ""),
        }
        for d in evaluate_cross_field_rules(cross_fields):
            out.append(_ucp_finding_from_dict(d))
        return out

    def _append_hard_rule_findings(
        self,
        all_findings: list[UCPFinding],
        ocr_text: str,
        clauses: list,
    ) -> None:
        """Ajoute les findings de swift_ucp_hard_rules sans doublon de code."""
        codes = {f.code for f in all_findings}
        for d in evaluate_hard_rules(ocr_text, clauses):
            c = d.get("code")
            if c and c not in codes:
                codes.add(c)
                all_findings.append(_ucp_finding_from_dict(d))

    def validate(self, pipeline_result: dict) -> UCPValidationReport:
        use_legacy = os.environ.get("UCP_USE_LEGACY_RULES", "").lower() in (
            "1", "true", "yes",
        )
        if use_legacy:
            return self._validate_rule_based(pipeline_result)
        return self._validate_semantic(pipeline_result)

    def _validate_rule_based(self, pipeline_result: dict) -> UCPValidationReport:
        """Comportement historique : règles regex + exclusions collated."""
        swift    = pipeline_result.get("swift", {})
        entities = swift.get("entities", {})
        lc       = swift.get("lc_criteria", {})
        nlp      = swift.get("nlp", {}) or pipeline_result.get("nlp", {})
        clauses  = nlp.get("clauses", [])
        ocr_text = entities.get("_ocr_text", "") or \
                   pipeline_result.get("ocr", {}).get("text", "")

        lc_number  = entities.get("lc_number", "?")
        swift_type = entities.get("swift_type", "MT700")
        ucp_ver    = entities.get("applicable_rules", "UCP LATEST VERSION")

        exclusions = self._excl_parser.parse(ocr_text, clauses)
        exclusion_types: dict[str, str] = {}
        for art_ref, clause_text in exclusions.items():
            exclusion_types[art_ref] = self._excl_typer.classify(art_ref, clause_text)

        all_findings: list[UCPFinding] = []
        all_findings += self._engine.check_article_6(entities, lc, exclusions)
        all_findings += self._engine.check_article_14(entities, lc, clauses, exclusions)
        all_findings += self._engine.check_article_16(entities, clauses, exclusions)
        all_findings += self._engine.check_article_18(entities, clauses, exclusions)
        all_findings += self._engine.check_article_19(entities, clauses, exclusions)
        all_findings += self._engine.check_article_20(entities, lc, clauses, exclusions)
        all_findings += self._engine.check_article_22(clauses, exclusions)
        all_findings += self._engine.check_article_23(entities, clauses, exclusions)
        all_findings += self._engine.check_article_26(clauses, exclusions)
        all_findings += self._engine.check_article_28(clauses, exclusions)
        all_findings += self._engine.check_article_30(entities, lc, clauses)
        all_findings += self._engine.check_article_37(entities, lc, exclusions)
        all_findings += self._engine.check_article_10(clauses, entities)
        all_findings += self._engine.check_article_38(entities, clauses, exclusions)
        all_findings += self._engine.check_article_39(clauses)
        all_findings += self._engine.check_article_12b(clauses, exclusions)
        all_findings += self._engine.check_scanned_docs(clauses)
        all_findings += self._engine.check_signature_requirements(clauses)
        all_findings += self._engine.check_third_party_invoice(clauses)
        all_findings += self._engine.check_confirmation(entities, lc)
        all_findings += self._engine.check_article_31(entities, lc)
        all_findings += self._engine.check_article_32(entities, lc)

        self._append_hard_rule_findings(all_findings, ocr_text, clauses)

        return self._finalize_report(
            lc_number=lc_number,
            swift_type=swift_type,
            ucp_ver=ucp_ver,
            all_findings=all_findings,
            exclusions=exclusions,
            exclusion_types=exclusion_types,
            semantic_meta={"mode": "legacy"},
        )

    def _validate_semantic(self, pipeline_result: dict) -> UCPValidationReport:
        swift    = pipeline_result.get("swift", {})
        entities = swift.get("entities", {})
        lc       = swift.get("lc_criteria", {})
        nlp      = swift.get("nlp", {}) or pipeline_result.get("nlp", {})
        clauses  = nlp.get("clauses", [])
        ocr_text = entities.get("_ocr_text", "") or \
                   pipeline_result.get("ocr", {}).get("text", "")

        lc_number  = entities.get("lc_number", "?")
        swift_type = entities.get("swift_type", "MT700")
        ucp_ver    = entities.get("applicable_rules", "UCP LATEST VERSION")

        sem_eng = self._get_semantic_engine()
        sem_res = sem_eng.run(clauses, ocr_text, self._engine.ARTICLE_TITLES)

        semantic_meta: dict = {
            "mode": "semantic",
            "available": sem_res.available,
            **sem_res.meta,
        }
        semantic_meta["per_clause"] = sem_res.per_clause

        if not sem_res.available:
            strict = os.environ.get("UCP_SEMANTIC_STRICT", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if strict:
                logger.error(
                    "UCP_SEMANTIC_STRICT: embeddings indisponibles (%s) — pas de repli legacy",
                    sem_res.meta.get("reason"),
                )
                return self._validate_semantic_strict_fallback(
                    pipeline_result, sem_res.meta
                )
            logger.warning("UCP sémantique indisponible — repli sur moteur legacy")
            return self._validate_rule_based(pipeline_result)

        exclusions = dict(sem_res.meta.get("exclusions") or {})
        exclusion_types = {}
        for art_ref, clause_text in exclusions.items():
            exclusion_types[art_ref] = self._excl_typer.classify(art_ref, clause_text)

        merged_dicts = merge_hard_findings(
            list(sem_res.finding_dicts or []),
            evaluate_hard_rules(ocr_text, clauses),
        )
        merged_dicts = dedupe_ucp_finding_dicts(merged_dicts)
        all_findings: list[UCPFinding] = [
            _ucp_finding_from_dict(d) for d in merged_dicts
        ]
        all_findings += self._structural_ucp_findings(
            entities, lc, clauses, exclusions
        )

        return self._finalize_report(
            lc_number=lc_number,
            swift_type=swift_type,
            ucp_ver=ucp_ver,
            all_findings=all_findings,
            exclusions=exclusions,
            exclusion_types=exclusion_types,
            semantic_meta=semantic_meta,
        )

    def _validate_semantic_strict_fallback(
        self,
        pipeline_result: dict,
        failure_meta: dict,
    ) -> UCPValidationReport:
        """
        UCP_SEMANTIC_STRICT=1 : pas de repli sur ClauseExclusionParser + regex lourdes ;
        uniquement checks structurés + finding d'erreur explicite.
        """
        swift = pipeline_result.get("swift", {})
        entities = swift.get("entities", {})
        lc = swift.get("lc_criteria", {})
        nlp = swift.get("nlp", {}) or pipeline_result.get("nlp", {})
        clauses = nlp.get("clauses", [])
        ocr_text = entities.get("_ocr_text", "") or pipeline_result.get(
            "ocr", {}
        ).get("text", "")

        lc_number = entities.get("lc_number", "?")
        swift_type = entities.get("swift_type", "MT700")
        ucp_ver = entities.get("applicable_rules", "UCP LATEST VERSION")

        from .swift_ucp_semantic_engine import parse_strict_exclusions

        exclusions = parse_strict_exclusions(ocr_text, clauses)
        exclusion_types: dict[str, str] = {}
        for art_ref, clause_text in exclusions.items():
            exclusion_types[art_ref] = self._excl_typer.classify(art_ref, clause_text)

        all_findings = self._structural_ucp_findings(
            entities, lc, clauses, exclusions
        )
        all_findings.insert(
            0,
            UCPFinding(
                article="UCP",
                article_title="Analyse sémantique",
                code="UCP_SEMANTIC_ENGINE_UNAVAILABLE",
                severity="error",
                category="compliance",
                message="Moteur UCP sémantique (SentenceTransformer) indisponible.",
                detail=str(failure_meta.get("reason", "")),
                recommendation=(
                    "Installer sentence-transformers et relancer, ou définir "
                    "UCP_USE_LEGACY_RULES=true. Désactiver UCP_SEMANTIC_STRICT "
                    "si un repli legacy est acceptable."
                ),
                confidence=1.0,
            ),
        )

        codes_seen = {f.code for f in all_findings}
        for d in findings_from_critical_exclusion_map(
            exclusions, self._engine.ARTICLE_TITLES
        ):
            c = d.get("code")
            if c and c not in codes_seen:
                codes_seen.add(c)
                all_findings.append(_ucp_finding_from_dict(d))
        self._append_hard_rule_findings(all_findings, ocr_text, clauses)

        semantic_meta: dict = {
            "mode": "semantic_strict_failed",
            "available": False,
            **failure_meta,
        }

        return self._finalize_report(
            lc_number=lc_number,
            swift_type=swift_type,
            ucp_ver=ucp_ver,
            all_findings=all_findings,
            exclusions=exclusions,
            exclusion_types=exclusion_types,
            semantic_meta=semantic_meta,
        )

    def _finalize_report(
        self,
        lc_number: str,
        swift_type: str,
        ucp_ver: str,
        all_findings: list[UCPFinding],
        exclusions: dict,
        exclusion_types: dict,
        semantic_meta: dict,
    ) -> UCPValidationReport:
        seen_codes: set[str] = set()
        unique_findings: list[UCPFinding] = []
        for f in all_findings:
            if f.code not in seen_codes:
                seen_codes.add(f.code)
                unique_findings.append(f)

        for f in unique_findings:
            f.weight         = self._weight_reg.get_weight(f.code)
            f.tier           = self._weight_reg.get_tier(f.code)
            if not f.blocks_payment:
                f.blocks_payment = self._weight_reg.blocks_payment(f.code)
            if f.article in exclusion_types:
                f.exclusion_type = exclusion_types[f.article]
            if not f.finding_nature:
                f.finding_nature = "true_violation"
            if f.finding_nature == "contractual_modification" and f.blocks_payment and not f.blocks_payment_reason:
                f.blocks_payment_reason = "structural_contractual_risk"
            if f.finding_nature == "operational_discrepancy" and f.blocks_payment and not f.blocks_payment_reason:
                f.blocks_payment_reason = "documentary_discrepancy"
            # v4 — taxonomie métier
            f.impact_level = _compute_impact_level(f)
            f.clause_type  = _compute_clause_type(f)

        categories = {
            "document":   UCPCategoryScore("document"),
            "compliance": UCPCategoryScore("compliance"),
            "payment":    UCPCategoryScore("payment"),
            "shipment":   UCPCategoryScore("shipment"),
        }
        cascade_log: list[dict] = []

        for f in unique_findings:
            cat = f.category if f.category in categories else "compliance"
            categories[cat].findings.append(f)

            penalty = self._weight_reg.get_penalty(f.code, f.severity)
            categories[cat].score = max(0.0, categories[cat].score - penalty)

            if f.severity == "error" and f.finding_nature == "true_violation":
                categories[cat].articles_nok.append(f.article)
            else:
                categories[cat].articles_ok.append(f.article)

            cascade_applied = self._cascade.compute(f.code, categories)
            for c in cascade_applied:
                c["cascade_impact_on_finding"] = c["malus"]
                f.cascade_impact += c["malus"]
            cascade_log.extend(cascade_applied)

        cat_weights = {"document": 0.35, "compliance": 0.40, "payment": 0.15, "shipment": 0.10}
        global_score = int(sum(
            categories[cat].score * cat_weights[cat]
            for cat in categories
        ))

        threshold = self._COMPLIANCE_THRESHOLDS.get(swift_type, {"min_score": 70, "max_errors": 0})
        n_errors  = len([f for f in unique_findings if f.severity == "error" and f.finding_nature == "true_violation"])
        compliant = (global_score >= threshold["min_score"] and n_errors <= threshold["max_errors"])

        # payment_blocked : uniquement violations UCP réelles ou discordances documentaires
        payment_blocked = any(
            f.blocks_payment and f.finding_nature in {"true_violation", "operational_discrepancy"}
            for f in unique_findings
        )
        # contractual_hold : modification contractuelle Art.1 à risque structurel (≠ violation)
        contractual_hold = (not payment_blocked) and any(
            f.blocks_payment and f.finding_nature == "contractual_modification"
            for f in unique_findings
        )

        return UCPValidationReport(
            lc_number=lc_number,
            swift_type=swift_type,
            ucp_version=ucp_ver,
            findings=unique_findings,
            category_scores=categories,
            exclusions=exclusions,
            exclusion_types=exclusion_types,
            global_score=global_score,
            compliant=compliant,
            payment_blocked=payment_blocked,
            contractual_hold=contractual_hold,
            cascade_log=cascade_log,
            processed_at=datetime.now().isoformat(),
            semantic=semantic_meta,
        )