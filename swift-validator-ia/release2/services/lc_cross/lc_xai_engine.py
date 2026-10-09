"""
lc_xai_engine.py — LC Cross-Validation XAI Engine  v2.0
=========================================================
Level 3 : Shapley-exact attribution + banker-grade NL explanations.
Level 4 : Probabilistic scoring + Causal graph + Counterfactual engine.

─── Level 3 — Shapley exact (linear model) ───────────────────────────────────
  φ_i = (s_i − 1.0) × (sev_w_i / Σ_j sev_w_j) × cat_weight_k
  Efficiency axiom : Σ_i φ_i = final_score − 1.0

─── Level 4A — Probabilistic (Monte Carlo / Beta) ────────────────────────────
  Each check status → Beta(α, β) distribution over its effective score.
  1 500 MC samples → empirical distribution of final_score.
  Outputs : mean, std, IC 95%, P(ACCEPT), P(DISCREPANT), P(REJECT).

─── Level 4B — Causal graph (DAG) ───────────────────────────────────────────
  Static DAG of causal dependencies between rules (domain knowledge).
  Root cause = failed node with no failed parent.
  Causal impact = φ_root + Σ φ_downstream_children.

─── Level 4C — Counterfactual engine ────────────────────────────────────────
  For each negative factor : simulate do(status_i = "pass").
  direct_gain  = −φ_i
  causal_gain  = −Σ φ_j  for j in causal_chain(i)
  counterfactual_score   = mean_score + causal_gain
  counterfactual_decision = classify(counterfactual_score)

Output schema
─────────────
  shapley_values      : {factor_id: φ_i}
  waterfall           : [{step, factor, delta, cumulative, label, impact_class, category}]
  explanations        : [{rank, shapley_value, shapley_pct, impact, severity, type,
                           category, rule, status, message, detail, remediation,
                           lc_value, doc_value, doc_type, is_primary}]
  executive_summary   : str
  decision_rationale  : str
  compliance_evidence : [str]
  remediation_plan    : [str]
  probabilistic       : {mean_score, std_dev, ci_95_lower, ci_95_upper,
                          decision_proba:{ACCEPT,DISCREPANT,REJECT}, n_samples}
  causal              : {root_causes, causal_chains, derived_failures,
                          graph_edges, n_root_causes, n_derived}
  counterfactuals     : [{rank, factor_id, rule, category, category_label, rule_label,
                           direct_gain, causal_gain, counterfactual_score,
                           counterfactual_decision, is_root_cause, n_downstream, message}]
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

# ── Constants (must match lc_commercial_cross_validator) ────────────────────

_CATEGORY_WEIGHTS: dict[str, float] = {
    "financial":   0.35,
    "goods":       0.25,
    "documentary": 0.20,
    "shipment":    0.10,
    "parties":     0.10,
}

_CHECK_SEV_WEIGHT: dict[str, float] = {
    "critical": 3.0,
    "high":     2.0,
    "medium":   1.5,
    "low":      1.0,
    "info":     0.5,
}

_STATUS_SCORE: dict[str, float] = {
    "pass":      1.00,
    "corrected": 0.90,
    "warning":   0.65,
    "fail":      0.00,
    "skipped":   None,   # special: graduated penalty
}

_SKIP_SCORE: dict[str, float] = {
    "critical": 0.70,
    "high":     0.78,
    "medium":   0.85,
    "low":      0.92,
    "info":     0.95,
}

_CAT_LABELS: dict[str, str] = {
    "financial":   "Financier",
    "goods":       "Marchandises (45A)",
    "shipment":    "Transport",
    "parties":     "Parties",
    "documentary": "Documentaire",
}

_CAT_ICONS: dict[str, str] = {
    "financial":   "💰",
    "goods":       "📦",
    "shipment":    "🚢",
    "parties":     "👥",
    "documentary": "📋",
}

_SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]

# ── Level 4A — Beta distribution parameters per status ──────────────────────
# Beta(α, β) over [0,1] models our uncertainty about the true score given status.
# pass    → concentrated near 1.0, high confidence
# warning → centred at ~0.50, broad uncertainty
# fail    → concentrated near 0.0, high confidence
# skipped → below 0.5, broad uncertainty (we don't know the true state)
_BETA_PARAMS: dict[str, tuple[float, float]] = {
    "pass":      (18.0,  2.0),
    "corrected": (14.0,  2.0),
    "warning":   ( 4.0,  6.0),
    "skipped":   ( 3.0,  5.0),
    "fail":      ( 2.0, 18.0),
}
_N_MC_SAMPLES = 1500

# ── Level 4B — Causal DAG (domain knowledge) ────────────────────────────────
# Each edge (parent, child) means: parent failing CAUSES or AMPLIFIES child failure.
# factor_id format: "{category}.{rule}"
_CAUSAL_EDGES: list[tuple[str, str]] = [
    # Wrong currency → amount comparison is meaningless
    ("financial.lc_currency",          "financial.lc_amount_tolerance"),
    # Late shipment date → presentation period inevitably exceeded
    ("shipment.lc_shipment_date",      "shipment.lc_presentation_period"),
    # Wrong goods description → semantic match always fails
    ("goods.lc_goods_description",     "goods.lc_goods_semantic"),
    # Wrong goods description → invoice items don't match
    ("goods.lc_goods_description",     "goods.lc_invoice_items_total"),
    # Wrong incoterm → port obligations misunderstood
    ("shipment.lc_incoterm",           "shipment.lc_port_of_discharge"),
    # Incoterm CIF/CFR → freight in total causes invoice arithmetic gap
    ("shipment.lc_incoterm",           "goods.lc_invoice_items_total"),
    # Partial shipments → weight and volume figures split across BLs
    ("shipment.lc_partial_shipments",  "goods.lc_weight_consistency"),
    ("shipment.lc_partial_shipments",  "goods.lc_cbm_consistency"),
    # Transhipment → final port of discharge may differ
    ("shipment.lc_transhipment",       "shipment.lc_port_of_discharge"),
]


# ── Rule-level NL templates ─────────────────────────────────────────────────
# Each entry: (message_template, detail_template, remediation_template)
# Placeholders: {lc}, {doc}, {dev}, {doc_type}

_RULE_TEMPLATES: dict[str, dict] = {
    "lc_currency": {
        "fail":    ("Devise LC ({lc}) différente de la facture ({doc})",
                    "UCP 600 art. 18(a)(ii) — la devise du crédit documentaire doit être "
                    "identique à celle de tous les documents présentés.",
                    "Émettre une nouvelle facture en {lc} ou demander un avenant LC."),
        "warning": ("Devise non vérifiable : LC={lc}, Doc={doc}",
                    "Données manquantes — vérification manuelle requise.",
                    "Compléter les champs devise dans les documents."),
    },
    "lc_amount_tolerance": {
        "fail":    ("Montant facture ({doc}) hors tolérance LC ({lc})",
                    "Écart de {dev} — hors plage UCP 600 art. 18(b). "
                    "Tolérance contractuelle : ±{tol}%.",
                    "Demander un avenant ou émettre une facture corrigée au montant LC "
                    "avec justification de l'écart."),
        "warning": ("Montant facture ({doc}) proche de la limite de tolérance",
                    "Déviation : {dev}. Vérifié mais à surveiller.",
                    "Aucune action requise si dans les limites contractuelles."),
    },
    "lc_expiry_date": {
        "fail":    ("Date document ({doc}) postérieure à l'expiration LC ({lc})",
                    "UCP 600 art. 6(d)(i) — les documents doivent être présentés "
                    "avant ou à la date d'expiration du crédit.",
                    "Vérifier la date d'émission des documents. "
                    "Si tardive, demander une prorogation LC."),
    },
    "lc_shipment_date": {
        "fail":    ("Date embarquement BL ({doc}) postérieure à la limite LC ({lc})",
                    "UCP 600 art. 20(a)(ii) — le connaissement doit indiquer "
                    "une date d'embarquement ≤ dernière date d'expédition LC.",
                    "Demander un avenant sur la date limite d'expédition LC."),
    },
    "lc_presentation_period": {
        "fail":    ("Délai de présentation dépassé : {doc} jours après embarquement (max {lc})",
                    "UCP 600 art. 14(c) — délai de présentation contractuel dépassé.",
                    "Négocier une acceptation tardive avec la banque émettrice ou "
                    "soumettre une demande de dérogation."),
    },
    "lc_incoterm": {
        "fail":    ("Incoterm LC ({lc}) incohérent avec les documents ({doc})",
                    "L'incoterm conditionne la répartition des risques et frais "
                    "— une discordance peut invalider le claim.",
                    "Harmoniser l'incoterm sur tous les documents commerciaux."),
    },
    "lc_partial_shipments": {
        "fail":    ("Expédition partielle détectée alors qu'elle est interdite par le LC",
                    "UCP 600 art. 31 — expéditions partielles interdites sauf stipulation contraire.",
                    "Regrouper les expéditions en un seul envoi ou demander un avenant."),
    },
    "lc_transhipment": {
        "fail":    ("Transbordement détecté alors qu'il est interdit par le LC",
                    "UCP 600 art. 20(b) — transbordement interdit sauf stipulation LC.",
                    "Utiliser un transport direct ou demander un avenant autorisant le transbordement."),
    },
    "lc_port_of_discharge": {
        "fail":    ("Port de déchargement BL ({doc}) ≠ port LC ({lc})",
                    "Le port de destination doit correspondre exactement au champ 44E du MT700.",
                    "Corriger le connaissement ou demander un avenant LC."),
    },
    "lc_goods_semantic": {
        "fail":    ("Description marchandises insuffisamment concordante avec LC (similarité {dev})",
                    "UCP 600 art. 18(c) — la description des marchandises sur la facture doit "
                    "correspondre à celle du crédit. Score sémantique : {dev} (seuil ≥ 0.60).",
                    "Reformuler la description de la facture pour correspondre exactement "
                    "au champ 45A du LC."),
        "warning": ("Concordance partielle description marchandises (similarité {dev})",
                    "Correspondance acceptable mais non optimale. Vérification manuelle conseillée.",
                    "Vérifier la cohérence terminologique entre LC et facture."),
    },
    "lc_doc_completeness": {
        "fail":    ("Documents requis manquants : {doc}",
                    "UCP 600 art. 14(a) — tous les documents requis par le LC doivent être présentés.",
                    "Compléter le dossier avec les documents manquants avant re-présentation."),
    },
    "lc_invoice_items_total": {
        "fail":    ("Incohérence arithmétique facture : total items ({doc}) ≠ total facture ({lc})",
                    "L'addition des lignes items ne correspond pas au total facture. "
                    "Écart : {dev}.",
                    "Corriger la facture ou justifier l'écart (ex. frais CIF non détaillés)."),
        "warning": ("Écart facture probable frais transport/assurance ({dev})",
                    "Incoterm CIF/CFR détecté — l'écart pourrait correspondre à des frais "
                    "inclus dans le total mais absents des lignes items.",
                    "Ajouter une ligne dédiée aux frais fret/assurance sur la facture."),
    },
    "lc_weight_consistency": {
        "fail":    ("Poids BL ({doc}) incohérent avec packing list ({lc})",
                    "Écart de poids : {dev}. Les documents de transport doivent être "
                    "cohérents entre eux.",
                    "Corriger la packing list ou le connaissement selon la pesée réelle."),
    },
    "lc_cbm_consistency": {
        "fail":    ("Volume CBM incohérent entre documents ({dev} d'écart)",
                    "Les volumes déclarés sur les différents documents doivent concorder.",
                    "Vérifier et harmoniser les volumes sur BL et packing list."),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# XAI Engine
# ─────────────────────────────────────────────────────────────────────────────

class LcXaiEngine:
    """
    Shapley-exact attribution and intelligent NL explanation engine
    for LC cross-validation results.

    Usage::
        engine = LcXaiEngine()
        xai = engine.explain(
            categories=categories,
            cross_doc=cross_doc,
            final_score=final_score,
            decision=decision,
            risk_level=risk_level,
        )
        # xai["shapley_values"]    → {factor_id: φ_i}
        # xai["waterfall"]         → waterfall chart data
        # xai["explanations"]      → ranked NL explanations
        # xai["executive_summary"] → 2-sentence banker summary
    """

    def explain(
        self,
        categories:  dict[str, dict],
        cross_doc:   dict,
        final_score: float,
        decision:    str,
        risk_level:  str,
    ) -> dict:
        """Main entry point — returns full XAI output (Level 3 + Level 4)."""
        shapley      = self._compute_shapley_values(categories)
        waterfall    = self._build_waterfall(shapley, final_score)
        explanations = self._build_explanations(shapley, categories, cross_doc, final_score)
        executive    = self._build_executive_summary(
            categories, final_score, decision, risk_level, explanations
        )
        rationale = self._build_decision_rationale(
            categories, final_score, decision, explanations
        )
        evidence     = self._collect_compliance_evidence(categories)
        remediation  = self._build_remediation_plan(explanations)

        # ── Level 4 ──────────────────────────────────────────────────────────
        prob            = self._probabilistic_scoring(categories)
        causal          = self._causal_analysis(shapley)
        counterfactuals = self._counterfactual_fixes(
            shapley, causal, prob["mean_score"], decision
        )
        causal_narrative   = self._build_causal_narrative(causal, final_score, decision, risk_level)
        simulation         = self._sequential_simulation(causal, prob["mean_score"], decision)
        decision_confidence = self._decision_confidence(prob, decision)

        return {
            # Level 3
            "shapley_values":      shapley,
            "waterfall":           waterfall,
            "explanations":        explanations,
            "executive_summary":   executive,
            "decision_rationale":  rationale,
            "compliance_evidence": evidence,
            "remediation_plan":    remediation,
            # Level 4
            "probabilistic":        prob,
            "causal":               causal,
            "counterfactuals":      counterfactuals,
            "causal_narrative":     causal_narrative,
            "simulation":           simulation,
            "decision_confidence":  decision_confidence,
        }

    # ── 1. Shapley value computation ────────────────────────────────────────

    def _compute_shapley_values(
        self, categories: dict[str, dict]
    ) -> dict[str, float]:
        """
        Compute exact Shapley values for a linear weighted model.

        φ_i = (s_i − 1.0) × (sev_w_i / Σ_j sev_w_j) × cat_weight_k

        where s_i = effective status score (0 for fail, 0.65 warning, 1.0 pass,
                    graduated for skipped), baseline = 1.0.

        Satisfies:  Σ_i φ_i = final_score − 1.0
        """
        shapley: dict[str, float] = {}

        for cat_name, cat_data in categories.items():
            cat_w = _CATEGORY_WEIGHTS.get(cat_name, 0.0)
            checks = cat_data.get("checks", [])
            if not checks:
                continue

            # Total severity weight in this category (denominator)
            total_sev_w = sum(
                _CHECK_SEV_WEIGHT.get(chk.get("severity", "medium"), 1.5)
                for chk in checks
            )
            if total_sev_w <= 0:
                continue

            for chk in checks:
                rule    = chk.get("rule", "unknown")
                status  = chk.get("status", "skipped")
                sev     = chk.get("severity", "medium")
                sev_w   = _CHECK_SEV_WEIGHT.get(sev, 1.5)

                # Effective status score
                if status == "skipped":
                    s_i = _SKIP_SCORE.get(sev, 0.85)
                else:
                    s_i = _STATUS_SCORE.get(status, 0.65)

                norm_w = sev_w / total_sev_w
                factor_id = f"{cat_name}.{rule}"
                shapley[factor_id] = round((s_i - 1.0) * norm_w * cat_w, 6)

        return shapley

    # ── 2. Waterfall decomposition ──────────────────────────────────────────

    def _build_waterfall(
        self,
        shapley:     dict[str, float],
        final_score: float,
    ) -> list[dict]:
        """
        Build SHAP-style waterfall decomposition.

        Steps: baseline (1.0) → sorted factors by |φ| → final_score.
        Negatives first (failures driving score down), then positives.
        """
        # Separate negative from positive (skipped/warnings vs passes)
        negatives = sorted(
            [(fid, v) for fid, v in shapley.items() if v < -0.0001],
            key=lambda x: x[1],          # most negative first
        )
        neutrals = [(fid, v) for fid, v in shapley.items() if abs(v) <= 0.0001]
        positives = sorted(
            [(fid, v) for fid, v in shapley.items() if v > 0.0001],
            key=lambda x: -x[1],         # largest positive first
        )

        steps: list[dict] = [{
            "step":         0,
            "factor":       "baseline",
            "delta":        0.0,
            "cumulative":   1.0,
            "label":        "Conformité parfaite (référence)",
            "impact_class": "baseline",
            "category":     "",
        }]

        cumulative = 1.0
        idx = 1

        for fid, phi in negatives + positives + neutrals:
            cat_name, rule = fid.split(".", 1) if "." in fid else ("", fid)
            cumulative = round(cumulative + phi, 6)
            steps.append({
                "step":         idx,
                "factor":       fid,
                "delta":        round(phi, 6),
                "cumulative":   max(0.0, min(1.0, cumulative)),
                "label":        self._factor_label(cat_name, rule, phi),
                "impact_class": self._impact_class(phi),
                "category":     cat_name,
            })
            idx += 1

        # Final marker (reconcile floating-point drift)
        steps.append({
            "step":         idx,
            "factor":       "final_score",
            "delta":        0.0,
            "cumulative":   round(final_score, 4),
            "label":        f"Score final · {final_score:.1%}",
            "impact_class": "final",
            "category":     "",
        })
        return steps

    @staticmethod
    def _factor_label(cat_name: str, rule: str, phi: float) -> str:
        icon = _CAT_ICONS.get(cat_name, "•")
        lbl  = _CAT_LABELS.get(cat_name, cat_name)
        rule_readable = rule.replace("lc_", "").replace("_", " ").title()
        sign = f"−{abs(phi):.3f}" if phi < 0 else f"+{phi:.3f}"
        return f"{icon} [{lbl}] {rule_readable}  ({sign})"

    @staticmethod
    def _impact_class(phi: float) -> str:
        if phi <= -0.15:  return "critical_negative"
        if phi <= -0.05:  return "high_negative"
        if phi <   0.0:   return "low_negative"
        if phi >= 0.05:   return "positive"
        return "neutral"

    # ── 3. Intelligent NL explanations ─────────────────────────────────────

    def _build_explanations(
        self,
        shapley:     dict[str, float],
        categories:  dict[str, dict],
        cross_doc:   dict,
        final_score: float,
    ) -> list[dict]:
        """
        Produce ranked, contextual NL explanations sorted by |φ_i|.
        Each entry includes: message, detail, remediation, Shapley value,
        lc_value / doc_value, doc_type, is_primary flag.
        """
        raw: list[dict] = []

        # ── Per-check explanations ────────────────────────────────────────
        for cat_name, cat_data in categories.items():
            primary_issue = cat_data.get("primary_issue")
            checks        = cat_data.get("checks", [])

            for chk in checks:
                rule    = chk.get("rule",     "unknown")
                status  = chk.get("status",   "skipped")
                sev     = chk.get("severity", "medium")
                msg     = chk.get("message",  "")
                detail  = chk.get("detail",   "")
                lc_val  = chk.get("lc_value",  "")
                doc_val = chk.get("doc_value", "")
                doc_tp  = chk.get("doc_type",  "")

                factor_id = f"{cat_name}.{rule}"
                phi       = shapley.get(factor_id, 0.0)
                phi_pct   = round(phi * 100, 2)

                # Status-aware NL text
                nl_msg, nl_detail, nl_remed = self._render_rule_nl(
                    rule, status, sev, msg, detail, lc_val, doc_val
                )

                impact = (
                    "negative" if status == "fail"
                    else "warning" if status in ("warning", "skipped")
                    else "positive"
                )

                _MISSING = {"", "N/D", "N/A", "null", "none", "n/d", "n/a", "-", "—"}
                ocr_suspect = (
                    status == "fail" and
                    any(
                        (not v) or str(v).strip() in _MISSING or len(str(v).strip()) <= 1
                        for v in (lc_val, doc_val)
                    )
                )

                raw.append({
                    "factor_id":     factor_id,
                    "shapley_value": phi,
                    "shapley_pct":   phi_pct,
                    "impact":        impact,
                    "severity":      sev,
                    "type":          self._check_type(cat_name, rule),
                    "category":      cat_name,
                    "rule":          rule,
                    "status":        status,
                    "message":       nl_msg or msg,
                    "detail":        nl_detail or detail,
                    "remediation":   nl_remed,
                    "lc_value":      lc_val,
                    "doc_value":     doc_val,
                    "doc_type":      doc_tp,
                    "is_primary":    rule == primary_issue,
                    "ocr_suspect":   ocr_suspect,
                })

        # ── Cross-document explanations ───────────────────────────────────
        cross_items = self._explain_cross_doc(cross_doc)
        raw.extend(cross_items)

        # ── Rank by |φ_i|, then severity ─────────────────────────────────
        sev_rank = {s: i for i, s in enumerate(_SEVERITY_ORDER)}
        raw.sort(key=lambda e: (
            round(-abs(e["shapley_value"]), 4),
            sev_rank.get(e["severity"], 99),
            0 if e["impact"] == "negative" else 1,
        ))

        # Assign rank
        for i, entry in enumerate(raw, 1):
            entry["rank"] = i

        return raw

    def _render_rule_nl(
        self,
        rule:    str,
        status:  str,
        sev:     str,
        msg:     str,
        detail:  str,
        lc_val:  str,
        doc_val: str,
    ) -> tuple[str, str, str]:
        """
        Render a context-aware NL triple (message, detail, remediation)
        for a given rule + status, enriched with actual LC/doc values.
        """
        tpl = _RULE_TEMPLATES.get(rule, {}).get(status)
        if not tpl:
            # Fallback: enrich the raw message with values
            enriched_msg = msg
            if lc_val and doc_val:
                enriched_msg = f"{msg} [LC={lc_val} · Doc={doc_val}]"
            elif lc_val:
                enriched_msg = f"{msg} [LC={lc_val}]"
            return enriched_msg, detail, self._default_remediation(status, sev)

        msg_tpl, det_tpl, rem_tpl = tpl

        # Compute deviation string from raw msg if present
        dev_str = self._extract_deviation(msg, detail)

        # Extract tolerance string
        tol_str = ""
        if "tolerance_plus" in lc_val.lower() if lc_val else False:
            tol_str = lc_val
        else:
            import re as _re
            m = _re.search(r'±?(\d+)', lc_val or "")
            if m:
                tol_str = m.group(1)

        def _sub(s: str) -> str:
            return (s
                    .replace("{lc}",  lc_val  or "N/D")
                    .replace("{doc}", doc_val or "N/D")
                    .replace("{dev}", dev_str or "N/D")
                    .replace("{tol}", tol_str or "N/D"))

        return _sub(msg_tpl), _sub(det_tpl), _sub(rem_tpl)

    @staticmethod
    def _extract_deviation(msg: str, detail: str) -> str:
        """Extract numeric deviation from message or detail strings."""
        import re as _re
        for text in (detail, msg):
            if not text:
                continue
            m = _re.search(r'(\d+\.?\d*\s*%)', text)
            if m:
                return m.group(1)
            m = _re.search(r'([+-]?\d+\.?\d*)', text)
            if m:
                return m.group(1)
        return ""

    @staticmethod
    def _default_remediation(status: str, sev: str) -> str:
        if status == "fail" and sev in ("critical", "high"):
            return ("Discordance critique — re-présentation requise après correction. "
                    "Contacter la banque émettrice pour statut.")
        if status == "fail":
            return "Corriger le document concerné avant re-présentation."
        if status == "warning":
            return "Vérification manuelle recommandée avant présentation."
        if status == "skipped":
            return "Données manquantes — compléter et re-valider."
        return ""

    @staticmethod
    def _check_type(cat_name: str, rule: str) -> str:
        if "semantic" in rule:
            return "semantic"
        if cat_name == "parties":
            return "fuzzy"
        if rule.startswith(("lc_cond", "lc_47a")):
            return "condition"
        return "rule"

    def _explain_cross_doc(self, cross_doc: dict) -> list[dict]:
        """Generate explanations for inter-document discrepancies."""
        items: list[dict] = []
        pair_map = {
            "inv_bl": "Invoice ↔ Bill of Lading",
            "inv_pl": "Invoice ↔ Packing List",
            "bl_pl":  "B/L ↔ Packing List",
        }
        for pair_key, pair_label in pair_map.items():
            pair = cross_doc.get(pair_key)
            if not isinstance(pair, dict):
                continue
            for chk in pair.get("checks", []):
                if chk.get("status") not in ("fail", "warning"):
                    continue
                chk_name = chk.get("check", "")
                sev      = chk.get("severity", "medium")
                status   = chk.get("status", "warning")
                items.append({
                    "factor_id":     f"cross_doc.{pair_key}.{chk_name}",
                    "shapley_value": 0.0,
                    "shapley_pct":   0.0,
                    "impact":        "negative" if status == "fail" else "warning",
                    "severity":      sev,
                    "type":          "cross_document",
                    "category":      "inter_docs",
                    "rule":          f"{pair_key}.{chk_name}",
                    "status":        status,
                    "message":       (f"[{pair_label}] {chk.get('message', chk_name)}"),
                    "detail":        chk.get("detail", ""),
                    "remediation":   "Harmoniser les données entre les deux documents.",
                    "lc_value":      chk.get("lc_value", ""),
                    "doc_value":     chk.get("doc_value", ""),
                    "doc_type":      pair_key,
                    "is_primary":    False,
                    "ocr_suspect":   False,
                })
        return items

    # ── 4. Executive summary ────────────────────────────────────────────────

    def _build_executive_summary(
        self,
        categories:   dict,
        final_score:  float,
        decision:     str,
        risk_level:   str,
        explanations: list[dict],
    ) -> str:
        """
        2–3 sentence banker-grade executive summary.
        """
        fail_cats  = [k for k, v in categories.items() if v["status"] == "fail"]
        part_cats  = [k for k, v in categories.items() if v["status"] == "partial"]
        pass_cats  = [k for k, v in categories.items() if v["status"] == "pass"]

        # Primary issue sentence
        top_issues = [
            e for e in explanations
            if e["impact"] == "negative" and e["shapley_value"] < -0.01
        ][:3]

        decision_phrase = {
            "ACCEPT":     "Le dossier est conforme et peut être accepté.",
            "DISCREPANT": "Le dossier présente des discordances et nécessite correction.",
            "REJECT":     "Le dossier présente une défaillance critique et doit être rejeté.",
        }.get(decision, "Décision non déterminée.")

        if fail_cats:
            fail_labels = " et ".join(_CAT_LABELS.get(c, c) for c in fail_cats)
            sentence1 = (
                f"Le crédit documentaire obtient un score global de {final_score:.1%} "
                f"(risque {risk_level.upper()}) avec {len(fail_cats)} catégorie(s) en échec : "
                f"{fail_labels}."
            )
        elif part_cats:
            part_labels = " et ".join(_CAT_LABELS.get(c, c) for c in part_cats)
            sentence1 = (
                f"Le crédit documentaire obtient un score de {final_score:.1%} "
                f"(risque {risk_level.upper()}) avec des non-conformités partielles en "
                f"{part_labels}."
            )
        else:
            sentence1 = (
                f"Le crédit documentaire est conforme sur les {len(pass_cats)} catégories "
                f"contrôlées avec un score de {final_score:.1%} (risque {risk_level.upper()})."
            )

        if top_issues:
            issue_msgs = "; ".join(e["message"] for e in top_issues[:2])
            sentence2 = f"Problèmes principaux : {issue_msgs}."
        else:
            sentence2 = "Aucune discordance critique détectée."

        return f"{sentence1} {sentence2} {decision_phrase}"

    # ── 5. Decision rationale ───────────────────────────────────────────────

    def _build_decision_rationale(
        self,
        categories:   dict,
        final_score:  float,
        decision:     str,
        explanations: list[dict],
    ) -> str:
        """
        Structured rationale for the ACCEPT/DISCREPANT/REJECT decision,
        referencing UCP 600 articles where applicable.
        """
        fail_cats    = [k for k, v in categories.items() if v["status"] == "fail"]
        part_cats    = [k for k, v in categories.items() if v["status"] == "partial"]
        n_crit_fails = sum(
            1 for e in explanations
            if e["severity"] == "critical" and e["impact"] == "negative"
        )
        top_phi = min(
            (e["shapley_value"] for e in explanations if e["impact"] == "negative"),
            default=0.0,
        )

        if decision == "REJECT":
            fin = categories.get("financial", {})
            is_crit = any(
                c.get("severity") == "critical" and c.get("status") == "fail"
                for c in fin.get("checks", [])
            )
            reason = (
                "Défaillance financière critique (montant ou devise hors conformité UCP 600 art. 18)"
                if is_crit else
                f"Score insuffisant ({final_score:.1%}) avec {n_crit_fails} "
                "non-conformité(s) critique(s)"
            )
            return (
                f"DÉCISION REJET : {reason}. "
                "La banque présentatrice ne peut pas honorer le crédit en l'état. "
                "Une levée de réserves formelle doit être émise (UCP 600 art. 16)."
            )

        if decision == "DISCREPANT":
            cats_str = ""
            if fail_cats:
                cats_str = f"Catégorie(s) en échec : {', '.join(_CAT_LABELS.get(c,c) for c in fail_cats)}. "
            elif part_cats:
                cats_str = f"Catégorie(s) partielles : {', '.join(_CAT_LABELS.get(c,c) for c in part_cats)}. "
            return (
                f"DÉCISION DISCREPANT : score global {final_score:.1%} — "
                f"impact Shapley maximal : {top_phi:.3f}. "
                f"{cats_str}"
                "Les discordances doivent être corrigées ou soumises à l'approbation "
                "du donneur d'ordre (UCP 600 art. 16)."
            )

        # ACCEPT
        return (
            f"DÉCISION ACCEPT : score global {final_score:.1%}, "
            "toutes les catégories conformes ou partiellement conformes. "
            "Les documents peuvent être présentés à la banque émettrice "
            "pour paiement/acceptation (UCP 600 art. 15)."
        )

    # ── 6. Compliance evidence ──────────────────────────────────────────────

    def _collect_compliance_evidence(self, categories: dict) -> list[str]:
        """Collect positive factors (passing checks worth mentioning)."""
        evidence: list[str] = []
        for cat_name, cat_data in categories.items():
            if cat_data["status"] == "pass":
                lbl = _CAT_LABELS.get(cat_name, cat_name)
                icon = _CAT_ICONS.get(cat_name, "✓")
                evidence.append(
                    f"{icon} {lbl} : conforme à 100% — "
                    f"{sum(1 for c in cat_data['checks'] if c.get('status')=='pass')} "
                    f"vérification(s) réussie(s)"
                )
            else:
                # List individual passes within failing categories
                passes = [
                    c for c in cat_data.get("checks", [])
                    if c.get("status") == "pass" and
                    c.get("severity") in ("critical", "high")
                ]
                for p in passes:
                    lbl = _CAT_LABELS.get(cat_name, cat_name)
                    rule_readable = p["rule"].replace("lc_", "").replace("_", " ").title()
                    evidence.append(f"✓ [{lbl}] {rule_readable} : conforme")
        return evidence[:10]  # cap at 10 items

    # ── 7. Remediation plan ─────────────────────────────────────────────────

    def _build_remediation_plan(self, explanations: list[dict]) -> list[str]:
        """
        Ordered list of actionable remediation steps, de-duplicated,
        critical failures first.
        """
        seen:    set[str]  = set()
        actions: list[str] = []

        # Critical and high failures first
        for priority in ("critical", "high", "medium", "low"):
            for e in explanations:
                if e.get("severity") != priority:
                    continue
                if e.get("impact") not in ("negative", "warning"):
                    continue
                rem = e.get("remediation", "").strip()
                if not rem or rem in seen:
                    continue
                seen.add(rem)
                prefix = "🔴" if priority == "critical" else \
                         "🟠" if priority == "high"     else \
                         "🟡" if priority == "medium"   else "🔵"
                cat    = _CAT_LABELS.get(e.get("category", ""), e.get("category", ""))
                actions.append(f"{prefix} [{cat}] {rem}")

        return actions[:12]  # cap at 12 items

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 4A — Probabilistic scoring (Monte Carlo / Beta distributions)
    # ════════════════════════════════════════════════════════════════════════

    def _probabilistic_scoring(self, categories: dict) -> dict:
        """
        Monte Carlo simulation over Beta(α,β) distributions.

        For each check, its effective score is no longer a point estimate
        but a Beta random variable whose parameters encode our confidence
        in the stated status.  We sample _N_MC_SAMPLES times to obtain an
        empirical distribution of the final_score, then derive:
          • mean, std, IC 95%
          • P(ACCEPT), P(DISCREPANT), P(REJECT)
        """
        # Collect (alpha, beta, norm_w, cat_w) for every check
        check_dists: list[tuple[float, float, float, float]] = []
        for cat_name, cat_data in categories.items():
            cat_w  = _CATEGORY_WEIGHTS.get(cat_name, 0.0)
            checks = cat_data.get("checks", [])
            total_sev_w = sum(
                _CHECK_SEV_WEIGHT.get(c.get("severity", "medium"), 1.5)
                for c in checks
            )
            if total_sev_w <= 0:
                continue
            for chk in checks:
                status = chk.get("status", "skipped")
                sev_w  = _CHECK_SEV_WEIGHT.get(chk.get("severity", "medium"), 1.5)
                alpha, beta = _BETA_PARAMS.get(status, (3.0, 5.0))
                check_dists.append((alpha, beta, sev_w / total_sev_w, cat_w))

        # Monte Carlo
        samples: list[float] = []
        for _ in range(_N_MC_SAMPLES):
            score = 1.0
            for alpha, beta_v, norm_w, cat_w in check_dists:
                s_i = random.betavariate(alpha, beta_v)
                score += (s_i - 1.0) * norm_w * cat_w
            samples.append(max(0.0, min(1.0, score)))

        samples.sort()
        n    = len(samples)
        mean = sum(samples) / n
        var  = sum((s - mean) ** 2 for s in samples) / n
        std  = math.sqrt(var)

        # Decision probability thresholds (mirrors _build_decision logic)
        p_accept     = sum(1 for s in samples if s >= 0.80) / n
        p_reject     = sum(1 for s in samples if s <  0.50) / n
        p_discrepant = max(0.0, 1.0 - p_accept - p_reject)

        return {
            "mean_score":  round(mean, 4),
            "std_dev":     round(std,  4),
            "ci_95_lower": round(samples[int(0.025 * n)], 4),
            "ci_95_upper": round(samples[int(0.975 * n)], 4),
            "decision_proba": {
                "ACCEPT":     round(p_accept,     3),
                "DISCREPANT": round(p_discrepant, 3),
                "REJECT":     round(p_reject,     3),
            },
            "n_samples": _N_MC_SAMPLES,
        }

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 4B — Causal graph analysis (DAG)
    # ════════════════════════════════════════════════════════════════════════

    def _causal_analysis(self, shapley: dict) -> dict:
        """
        Build a causal sub-graph restricted to failing/warning checks.

        Root cause  = failed node with no failed parent in the DAG.
        Derived     = failed node whose failure is amplified by a failed parent.
        Causal impact = own φ + Σ φ of all downstream nodes (BFS).

        A single root-cause fix propagates through the causal chain,
        making the intervention_gain > direct Shapley gain.
        """
        bad: set[str] = {fid for fid, phi in shapley.items() if phi < -0.0005}

        # Build adjacency restricted to bad nodes
        children: dict[str, list[str]] = defaultdict(list)
        parents:  dict[str, list[str]] = defaultdict(list)
        for parent, child in _CAUSAL_EDGES:
            if parent in bad and child in bad:
                children[parent].append(child)
                parents[child].append(parent)

        roots   = [fid for fid in bad if not parents[fid]]
        derived = [fid for fid in bad if parents[fid]]

        def bfs_chain(root: str) -> list[str]:
            visited, queue = [], [root]
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.append(node)
                    queue.extend(children[node])
            return visited

        root_cause_list: list[dict] = []
        causal_chains:   list[dict] = []

        for root in sorted(roots, key=lambda f: shapley.get(f, 0.0)):
            chain          = bfs_chain(root)
            total_impact   = sum(shapley.get(f, 0.0) for f in chain)
            cat_name, rule = root.split(".", 1) if "." in root else ("", root)

            root_cause_list.append({
                "factor_id":           root,
                "rule":                rule,
                "category":            cat_name,
                "shapley_value":       round(shapley.get(root, 0.0), 6),
                "causal_children":     [c for c in chain if c != root],
                "total_causal_impact": round(total_impact, 6),
                "intervention_gain":   round(-total_impact, 6),
                "is_root":             True,
            })
            if len(chain) > 1:
                causal_chains.append({
                    "root":              root,
                    "chain":             chain,
                    "total_impact":      round(total_impact, 6),
                    "intervention_gain": round(-total_impact, 6),
                    "n_downstream":      len(chain) - 1,
                })

        return {
            "root_causes":     root_cause_list,
            "causal_chains":   sorted(causal_chains, key=lambda x: x["total_impact"]),
            "derived_failures": derived,
            "graph_edges":     [[p, c] for p, c in _CAUSAL_EDGES if p in bad and c in bad],
            "n_root_causes":   len(roots),
            "n_derived":       len(derived),
        }

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 4C — Counterfactual engine  do(fix_i) → Δscore
    # ════════════════════════════════════════════════════════════════════════

    def _counterfactual_fixes(
        self,
        shapley:     dict[str, float],
        causal:      dict,
        mean_score:  float,
        decision:    str,
    ) -> list[dict]:
        """
        For every negative factor, simulate fixing it (status → pass).

        direct_gain  = −φ_i                           (own Shapley impact)
        causal_gain  = −Σ φ_j for j in chain(i)       (propagated impact)

        Sorted by causal_gain descending → highest-leverage actions first.
        Includes a "decision flip" flag when the counterfactual changes REJECT→ACCEPT.
        """
        # Build causal gain map from root cause list
        causal_gain_map: dict[str, float] = {
            rc["factor_id"]: rc["intervention_gain"]
            for rc in causal["root_causes"]
        }
        derived_set = set(causal["derived_failures"])

        results: list[dict] = []
        seen_rules: set[str] = set()

        negatives = sorted(
            [(fid, phi) for fid, phi in shapley.items() if phi < -0.0005],
            key=lambda x: x[1],
        )

        for factor_id, phi in negatives:
            cat_name, rule = factor_id.split(".", 1) if "." in factor_id else ("", factor_id)
            if rule in seen_rules:
                continue
            seen_rules.add(rule)

            direct_gain = round(-phi, 6)
            c_gain      = causal_gain_map.get(factor_id, direct_gain)
            cf_score    = min(1.0, round(mean_score + c_gain, 4))
            cf_decision = self._score_to_decision(cf_score)

            is_root = factor_id not in derived_set
            n_down  = next(
                (len(rc["causal_children"]) for rc in causal["root_causes"]
                 if rc["factor_id"] == factor_id),
                0,
            )

            cat_label    = _CAT_LABELS.get(cat_name, cat_name)
            rule_label   = rule.replace("lc_", "").replace("_", " ").title()
            flips        = cf_decision != decision

            results.append({
                "rank":                   0,          # assigned after sort
                "factor_id":              factor_id,
                "rule":                   rule,
                "category":              cat_name,
                "category_label":        cat_label,
                "rule_label":            rule_label,
                "direct_gain":           direct_gain,
                "causal_gain":           round(c_gain, 6),
                "counterfactual_score":  cf_score,
                "counterfactual_decision": cf_decision,
                "is_root_cause":         is_root,
                "n_downstream":          n_down,
                "flips_decision":        flips,
                "message": (
                    f"Corriger [{cat_label}] {rule_label} : "
                    f"{mean_score:.1%} → {cf_score:.1%} (+{c_gain:.1%})"
                    + (f" — décision : {decision} → {cf_decision} 🔄" if flips else "")
                ),
            })

        results.sort(key=lambda x: -x["causal_gain"])
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return results[:10]

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 4D — Causal narrative (NL paragraph)
    # ════════════════════════════════════════════════════════════════════════

    def _build_causal_narrative(
        self,
        causal:      dict,
        final_score: float,
        decision:    str,
        risk_level:  str,
    ) -> str:
        """
        Single AI-written paragraph that tells the causal story:
        root causes → cascade → projected best fix → decision implication.
        """
        roots   = causal["root_causes"]
        n_roots = causal["n_root_causes"]
        n_deriv = causal["n_derived"]

        if not roots:
            total = n_roots + n_deriv
            return (
                f"Aucune relation causale directe identifiée entre les {total} anomalies. "
                f"Chaque défaillance est indépendante. Score global : {final_score:.1%}."
            )

        top        = roots[0]
        cat_label  = _CAT_LABELS.get(top["category"], top["category"])
        rule_label = top["rule"].replace("lc_", "").replace("_", " ").title()

        parts: list[str] = [
            f"L'analyse causale identifie {n_roots} cause(s) racine indépendante(s). "
        ]

        parts.append(
            f"La défaillance principale est [{cat_label}] {rule_label} "
            f"(φ = {top['shapley_value']:.3f})"
        )
        if top["causal_children"]:
            children_labels = ", ".join(
                f.split(".", 1)[1].replace("lc_", "").replace("_", " ").title()
                for f in top["causal_children"]
            )
            parts.append(f", qui provoque en cascade : {children_labels}")
        parts.append(". ")

        if n_deriv > 0:
            parts.append(
                f"Sur {n_roots + n_deriv} anomalies totales, {n_deriv} sont des effets dérivés "
                f"qui disparaîtraient si leurs causes racines étaient corrigées. "
            )

        best_gain = max(rc["intervention_gain"] for rc in roots)
        best_rc   = next(rc for rc in roots if rc["intervention_gain"] == best_gain)
        best_cat  = _CAT_LABELS.get(best_rc["category"], best_rc["category"])
        best_rule = best_rc["rule"].replace("lc_", "").replace("_", " ").title()
        projected = min(1.0, final_score + best_gain)
        proj_dec  = self._score_to_decision(projected)
        flip_str  = f" → décision {proj_dec}" if proj_dec != decision else ""

        parts.append(
            f"L'intervention prioritaire est [{best_cat}] {best_rule} : "
            f"gain estimé +{best_gain:.1%}, score {final_score:.1%} → {projected:.1%}{flip_str}."
        )

        return "".join(parts)

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 4E — Sequential simulation  do(fix_1) → do(fix_2) → …
    # ════════════════════════════════════════════════════════════════════════

    def _sequential_simulation(
        self,
        causal:     dict,
        mean_score: float,
        decision:   str,
    ) -> list[dict]:
        """
        Simulate fixing root causes one by one (ordered by intervention_gain).
        Returns the cumulative score trajectory — each step is a "what if" state.
        """
        roots = sorted(
            causal["root_causes"],
            key=lambda rc: -rc["intervention_gain"],
        )
        steps: list[dict] = [{
            "step":           0,
            "action":         "Situation actuelle",
            "score":          round(mean_score, 4),
            "decision":       decision,
            "gain":           0.0,
            "fix":            None,
            "category_label": "",
            "is_flip":        False,
        }]
        current_score = mean_score
        prev_decision = decision

        for i, rc in enumerate(roots[:5], 1):
            gain = rc["intervention_gain"]
            if gain <= 0.0005:
                continue
            current_score = min(1.0, round(current_score + gain, 4))
            new_decision  = self._score_to_decision(current_score)
            cat_label     = _CAT_LABELS.get(rc["category"], rc["category"])
            rule_label    = rc["rule"].replace("lc_", "").replace("_", " ").title()
            steps.append({
                "step":           i,
                "action":         f"Corriger {rule_label}",
                "score":          current_score,
                "decision":       new_decision,
                "gain":           round(gain, 4),
                "fix":            rc["factor_id"],
                "category_label": cat_label,
                "is_flip":        new_decision != prev_decision,
            })
            prev_decision = new_decision

        return steps

    # ════════════════════════════════════════════════════════════════════════
    # LEVEL 4F — Decision confidence (stability analysis)
    # ════════════════════════════════════════════════════════════════════════

    def _decision_confidence(
        self,
        probabilistic: dict,
        decision:      str,
    ) -> dict:
        """
        How stable is the decision given the Monte Carlo distribution?
        A high-confidence REJECT (P=0.95) is very different from P=0.51.
        Includes margin vs closest competitor and borderline detection.
        """
        proba     = probabilistic["decision_proba"]
        p_current = proba.get(decision, 0.0)

        competitors = {k: v for k, v in proba.items() if k != decision}
        closest     = max(competitors, key=lambda k: competitors[k]) if competitors else None
        p_closest   = competitors.get(closest, 0.0) if closest else 0.0
        margin      = round(p_current - p_closest, 3)

        if p_current >= 0.85:   level, color = "Très élevée",        "#10B981"
        elif p_current >= 0.65: level, color = "Élevée",             "#6366F1"
        elif p_current >= 0.45: level, color = "Modérée",            "#F59E0B"
        else:                   level, color = "Faible — borderline", "#EF4444"

        return {
            "probability":            round(p_current, 3),
            "level":                  level,
            "color":                  color,
            "margin":                 margin,
            "closest_competitor":     closest,
            "competitor_probability": round(p_closest, 3),
            "is_borderline":          p_current < 0.55,
            "pct_width":              round(p_current * 100),
        }

    @staticmethod
    def _score_to_decision(score: float) -> str:
        if score >= 0.80:
            return "ACCEPT"
        if score < 0.50:
            return "REJECT"
        return "DISCREPANT"
