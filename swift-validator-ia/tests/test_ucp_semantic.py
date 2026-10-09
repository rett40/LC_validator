"""Tests unitaires — moteur UCP sémantique et validateur."""

import os
import unittest

from services.swift_nlp.ucp.swift_ucp_clause_aligner import normalize_clause, load_article_profiles
from services.swift_nlp.ucp.swift_ucp_semantic_engine import (
    parse_strict_exclusions,
    aggregate_findings,
)
from services.swift_nlp.ucp.swift_ucp_validator import (
    RuleWeightRegistry,
    UCP600Validator,
    UCPFinding,
    UCPValidationReport,
    _merge_semantic_rule_weights,
)


class TestNormalizeAndProfiles(unittest.TestCase):
    def test_normalize_clause(self):
        self.assertEqual(normalize_clause("  HELLO   WORLD  "), "hello world")
        self.assertEqual(normalize_clause(""), "")

    def test_profiles_load(self):
        profiles = load_article_profiles()
        self.assertGreaterEqual(len(profiles), 8)
        arts = {p.article for p in profiles}
        self.assertIn("Art.28", arts)
        self.assertIn("Art.17", arts)


class TestStrictExclusions(unittest.TestCase):
    def test_explicit_exclusion_detected(self):
        clauses = [{"text": "Article 14k of UCP 600 is excluded"}]
        ex = parse_strict_exclusions("", clauses)
        self.assertIn("Art.14k", ex)

    def test_french_exclusion_detected(self):
        clauses = [{"text": "L'article 17 n'est pas applicable à ce crédit"}]
        ex = parse_strict_exclusions("", clauses)
        self.assertIn("Art.17", ex)

    def test_english_gap_does_not_apply(self):
        clauses = [{"text": "Article 28 of UCP 600 does not apply"}]
        ex = parse_strict_exclusions("", clauses)
        self.assertIn("Art.28", ex)

    def test_collated_false_positive_removed(self):
        """Pas de détection sur seule proximité « article 20 » + « not applicable » ailleurs."""
        clauses = [
            {"text": "article 20 bill of lading. Other terms not applicable to this shipment."},
        ]
        ex = parse_strict_exclusions("", clauses)
        self.assertEqual(ex, {})


class TestSemanticWeights(unittest.TestCase):
    def test_sem_codes_registered(self):
        w = {}
        _merge_semantic_rule_weights(w)
        self.assertIn("SEM_VIOL_ART28", w)
        self.assertIn("SEM_REST_ART14", w)
        self.assertIn("SEM_CONTRACT_MOD_ART18B", w)
        self.assertIn("SEM_CONTRACT_STRUCT_ART38I", w)
        self.assertTrue(w["SEM_VIOL_ART28"]["blocks_payment"])
        self.assertIn("UCP_HARD_CONTRACT_ART18", w)
        self.assertFalse(w["UCP_HARD_CONTRACT_ART18"]["blocks_payment"])

    def test_registry_includes_semantic(self):
        reg = RuleWeightRegistry()
        self.assertGreater(reg.get_weight("SEM_VIOL_ART17"), 0)


class TestAggregateFindings(unittest.TestCase):
    def test_worst_relation_wins(self):
        titles = {"Art.28": "Insurance"}
        rows = [
            {
                "article": "Art.28",
                "relation": "RESTRICTION",
                "confidence": 0.7,
                "clause_text": "a",
            },
            {
                "article": "Art.28",
                "relation": "VIOLATION",
                "confidence": 0.9,
                "clause_text": "b",
            },
        ]
        fds = aggregate_findings(rows, titles)
        self.assertEqual(len(fds), 1)
        self.assertEqual(fds[0]["code"], "SEM_VIOL_ART28")
        self.assertEqual(fds[0]["severity"], "error")

    def test_contract_modification_code(self):
        titles = {"Art.18b": "Invoice amount"}
        rows = [
            {
                "article": "Art.18b",
                "relation": "CONTRACT_MODIFICATION",
                "confidence": 0.9,
                "clause_text": "article 18b is not applicable",
            },
        ]
        fds = aggregate_findings(rows, titles)
        self.assertEqual(len(fds), 1)
        self.assertEqual(fds[0]["code"], "SEM_CONTRACT_MOD_ART18B")
        self.assertEqual(fds[0]["finding_nature"], "contractual_modification")


@unittest.skipUnless(
    os.environ.get("UCP_RUN_SLOW_TESTS", "").lower() in ("1", "true", "yes"),
    "set UCP_RUN_SLOW_TESTS=1 to run (charge SentenceTransformer)",
)
class TestSemanticEngineSlow(unittest.TestCase):
    def test_aligner_runs(self):
        from services.swift_nlp.ucp.swift_ucp_clause_aligner import UCPClauseAligner

        aligner = UCPClauseAligner()
        if not aligner._ensure_profile_embeddings():
            self.skipTest("embeddings unavailable")
        al = aligner.align_clause(
            "Insurance policy must not contain any franchise or deductible clause."
        )
        self.assertIsNotNone(al)
        self.assertGreater(al.similarity, 0.2)


class TestValidatorModes(unittest.TestCase):
    def test_semantic_strict_fallback_no_legacy(self):
        from unittest.mock import MagicMock, patch

        os.environ["UCP_SEMANTIC_STRICT"] = "true"
        try:
            v = UCP600Validator()
            mock_eng = MagicMock()
            mock_eng.run.return_value = MagicMock(
                available=False,
                finding_dicts=[],
                per_clause=[],
                meta={"reason": "embeddings_unavailable"},
            )
            with patch.object(v, "_get_semantic_engine", return_value=mock_eng):
                rep = v.validate(
                    {
                        "swift": {
                            "entities": {
                                "lc_number": "X",
                                "swift_type": "MT700",
                                "applicable_rules": "UCP",
                                "expiry_date": "2030-01-01",
                                "expiry_place": "Tunis",
                            },
                            "lc_criteria": {"documents": [{"name": "INV"}]},
                            "nlp": {"clauses": []},
                        }
                    }
                )
            self.assertEqual(rep.semantic.get("mode"), "semantic_strict_failed")
            codes = [f.code for f in rep.findings]
            self.assertIn("UCP_SEMANTIC_ENGINE_UNAVAILABLE", codes)
        finally:
            del os.environ["UCP_SEMANTIC_STRICT"]

    def test_legacy_env_uses_rule_engine(self):
        os.environ["UCP_USE_LEGACY_RULES"] = "true"
        try:
            v = UCP600Validator()
            payload = {
                "swift": {
                    "entities": {
                        "lc_number": "X",
                        "swift_type": "MT700",
                        "applicable_rules": "UCP",
                        "expiry_date": "2030-01-01",
                        "expiry_place": "Tunis",
                    },
                    "lc_criteria": {"documents": [{"name": "INV"}]},
                    "nlp": {"clauses": []},
                }
            }
            rep = v.validate(payload)
            self.assertEqual(rep.semantic.get("mode"), "legacy")
        finally:
            del os.environ["UCP_USE_LEGACY_RULES"]


class TestRiskEngineUcpSummary(unittest.TestCase):
    def test_n_ucp_warnings_from_summary_dict(self):
        from services.risk_service.risk_engine import RiskFeatureExtractor

        rfx = RiskFeatureExtractor()
        feats = rfx.extract(
            {
                "swift": {"entities": {}, "nlp": {}},
                "ucp_compliance": {
                    "summary": {"warnings": 2, "errors": 1},
                    "global_score": 72,
                },
            }
        )
        self.assertEqual(feats.get("n_ucp_warnings"), 3.0)

    def test_ucp_decision_reject_feature_and_rule(self):
        from services.risk_service.risk_engine import RiskFeatureExtractor, UCPRuleEngine

        rfx = RiskFeatureExtractor()
        feats = rfx.extract(
            {
                "swift": {"entities": {}, "nlp": {}},
                "ucp_compliance": {
                    "ucp_decision": "REJECT",
                    "summary": {"errors": 1, "warnings": 0},
                },
            }
        )
        self.assertEqual(feats.get("ucp_decision_reject"), 1.0)
        viol, _ = UCPRuleEngine().evaluate(feats)
        self.assertTrue(any(getattr(v, "severity", "") == "BLOCK" for v in viol))


class TestHardRulesAndDecision(unittest.TestCase):
    def test_evaluate_hard_rules_mt700_phrases(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_hard_rules

        txt = (
            "ARTICLE 18 OF UCP 600 IS NOT APPLICABLE. "
            "ARTICLE 38i OF UCP 600 IS NOT APPLICABLE."
        )
        fds = evaluate_hard_rules(txt, [])
        codes = {d["code"] for d in fds}
        self.assertIn("UCP_HARD_CONTRACT_ART18", codes)
        self.assertIn("UCP_HARD_CONTRACT_ART38I", codes)

    def test_critical_exclusion_map_findings(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import findings_from_critical_exclusion_map

        ex = {
            "Art.18": "article 18 of ucp 600 is not applicable",
        }
        fds = findings_from_critical_exclusion_map(ex, {"Art.18": "Commercial invoice"})
        codes = {d["code"] for d in fds}
        self.assertIn("SEM_CONTRACT_STRUCT_ART18", codes)

    def test_classify_relation_exclusion_before_low_sim(self):
        from services.swift_nlp.ucp.swift_ucp_clause_aligner import (
            ClauseAlignment,
            UCPAlignerConfig,
        )
        from services.swift_nlp.ucp.swift_ucp_semantic_engine import classify_relation

        al = ClauseAlignment(
            clause_index=0,
            raw_text="ARTICLE 18 OF UCP 600 IS NOT APPLICABLE",
            normalized="article 18 of ucp 600 is not applicable",
            best_article="Art.28",
            best_title="Insurance",
            similarity=0.05,
            category="document",
            top_k=[],
        )
        rel, _, art_f = classify_relation(al, {}, UCPAlignerConfig(), nli=None)
        self.assertEqual(rel, "STRUCTURAL_MODIFICATION")
        self.assertEqual(art_f, "Art.18")

    def test_ucp_validation_report_to_dict_decision(self):
        from services.swift_nlp.ucp.swift_ucp_validator import UCPFinding, UCPValidationReport

        rep = UCPValidationReport(
            lc_number="L1",
            swift_type="MT700",
            ucp_version="UCP 600",
            findings=[
                UCPFinding(
                    article="Art.18",
                    article_title="Invoice",
                    code="SEM_CONTRACT_STRUCT_ART18",
                    severity="warning",
                    category="document",
                    message="Exclusion Art.18",
                    finding_nature="contractual_modification",
                ),
            ],
            global_score=40,
            compliant=False,
            payment_blocked=False,
        )
        d = rep.to_dict()
        # v4 : 5 niveaux de décision possibles
        self.assertIn(d.get("ucp_decision"), {"ACCEPT", "REVIEW", "ACCEPT_WITH_FLAGS", "CONDITIONAL_HOLD", "REJECT"})
        self.assertEqual(d.get("contractual_risk_level"), "HIGH")


class TestUcpContractualRecalibration(unittest.TestCase):
    def test_contractual_exclusions_not_true_violation(self):
        rep = UCPValidationReport(
            lc_number="X1",
            swift_type="MT700",
            ucp_version="UCP 600",
            global_score=72,
            compliant=True,
            payment_blocked=False,
        )
        rep.findings = [
            UCPFinding(article="Art.37c", article_title="Fees", code="SEM_CONTRACT_MOD_ART37C", severity="warning", category="compliance", message="x", finding_nature="contractual_modification"),
            UCPFinding(article="Art.18b", article_title="Invoice", code="SEM_CONTRACT_MOD_ART18B", severity="warning", category="document", message="x", finding_nature="contractual_modification"),
            UCPFinding(article="Art.12b", article_title="Prepayment", code="SEM_CONTRACT_MOD_ART12B", severity="warning", category="payment", message="x", finding_nature="contractual_modification"),
            UCPFinding(article="Art.14k", article_title="Shipper", code="SEM_CONTRACT_MOD_ART14K", severity="warning", category="document", message="x", finding_nature="contractual_modification"),
        ]
        d = rep.to_dict()
        self.assertEqual(d.get("legal_compliance_status"), "COMPLIANT_WITH_CONTRACTUAL_MODIFICATIONS")
        self.assertEqual(d.get("contractual_risk_level"), "MEDIUM")

    def test_operational_discrepancy_can_block(self):
        rep = UCPValidationReport(
            lc_number="L2",
            swift_type="MT700",
            ucp_version="UCP 600",
            findings=[
                UCPFinding(
                    article="Art.18",
                    article_title="Invoice",
                    code="ART18_NO_THIRD_PARTY_INVOICE",
                    severity="error",
                    category="document",
                    message="Formerly known as prohibited",
                    finding_nature="operational_discrepancy",
                    blocks_payment=True,
                    blocks_payment_reason="documentary_discrepancy",
                ),
            ],
            global_score=55,
            compliant=False,
            payment_blocked=True,
        )
        d = rep.to_dict()
        self.assertEqual(d.get("documentary_discrepancy_risk"), "HIGH")
        self.assertEqual(d.get("ucp_decision"), "REJECT")

    def test_structural_art38i_high_risk(self):
        rep = UCPValidationReport(
            lc_number="L3",
            swift_type="MT700",
            ucp_version="UCP 600",
            findings=[
                UCPFinding(
                    article="Art.38i",
                    article_title="Transfer",
                    code="SEM_CONTRACT_STRUCT_ART38I",
                    severity="warning",
                    category="compliance",
                    message="Art.38i excluded",
                    finding_nature="contractual_modification",
                    blocks_payment=True,
                    blocks_payment_reason="structural_transfer_constraint",
                ),
            ],
            global_score=60,
            compliant=False,
            payment_blocked=True,
        )
        d = rep.to_dict()
        self.assertEqual(d.get("contractual_risk_level"), "HIGH")
        self.assertEqual(d.get("ucp_decision"), "REJECT")


class TestSemanticContradictions(unittest.TestCase):
    def test_transhipment_allowed_but_clause_prohibits_french(self):
        from services.swift_nlp.semantic.swift_semantic_analyzer import SwiftSemanticAnalyzer

        analyzer = SwiftSemanticAnalyzer(enable_semantic_coherence=False)
        entities = {"swift_type": "MT700", "transhipment": "ALLOWED"}
        text = (
            ":43T: ALLOWED\n"
            ":47A: LE TRANSBORDEMENT EST INTERDIT.\n"
        )
        sem = analyzer.analyze(entities, text)
        self.assertGreaterEqual(len(sem.contradictions), 1)
        d0 = sem.contradictions[0].to_dict()
        self.assertEqual(d0.get("tag_field"), "transhipment")
        self.assertEqual(d0.get("tag_value"), "ALLOWED")

    def test_transhipment_collated_word_detected(self):
        from services.swift_nlp.semantic.swift_semantic_analyzer import SwiftSemanticAnalyzer

        analyzer = SwiftSemanticAnalyzer(enable_semantic_coherence=False)
        # Use a lightweight object with attributes to mimic pipeline adapter behavior.
        class _Ent:
            swift_type = "MT700"
            transhipment = "ALLOWED"

        text = ":47A: TTransshipmentis prohibited even if shipment is effected by container.\n"
        sem = analyzer.analyze(_Ent(), text)
        self.assertGreaterEqual(len(sem.contradictions), 1)
        self.assertEqual(sem.contradictions[0].tag_field, "transhipment")

    def test_partial_shipments_collated_word_detected(self):
        from services.swift_nlp.semantic.swift_semantic_analyzer import SwiftSemanticAnalyzer

        analyzer = SwiftSemanticAnalyzer(enable_semantic_coherence=False)
        class _Ent:
            swift_type = "MT700"
            partial_shipments = "ALLOWED"

        text = ":47A: Partial shipmentsis not allowed.\n"
        sem = analyzer.analyze(_Ent(), text)
        self.assertGreaterEqual(len(sem.contradictions), 1)
        self.assertEqual(sem.contradictions[0].tag_field, "partial_shipments")

    def test_confirmation_conditional_warning(self):
        from services.swift_nlp.semantic.swift_semantic_analyzer import SwiftSemanticAnalyzer

        analyzer = SwiftSemanticAnalyzer(enable_semantic_coherence=False)
        class _Ent:
            swift_type = "MT700"
            confirmation = "CONFIRM"

        text = ":47A: IN CASE CONFIRMATION IS NOT ADDED TO THIS LC, FIELD :31D: SHOULD READ 'TUNISIA'.\n"
        sem = analyzer.analyze(_Ent(), text)
        self.assertTrue(any(c.tag_field == "confirmation" and c.severity == "warning" for c in sem.contradictions))

    def test_pipeline_serializes_semantic_contradictions(self):
        from services.swift_nlp.swift_nlp_service import NLPProcessingResult
        from pipeline.swift_pipeline import SwiftPipeline

        class _DummyContradiction:
            def to_dict(self):
                return {
                    "tag_field": "transhipment",
                    "tag_value": "ALLOWED",
                    "clause_text": "TRANSBORDEMENT INTERDIT",
                    "clause_intent": "prohibit",
                    "ucp_basis": "Art.5 / Art.20c",
                    "severity": "error",
                    "recommendation": "Fix :43T:",
                }

        class _DummySemantic:
            completeness_score = 80
            semantic_score = 70
            doc_quality_score = 50
            warnings = []
            risk_flags = []
            contradictions = [_DummyContradiction()]
            fraud_indicators = []
            recommendations = []
            summary = ""
            enriched_fields = {}
            coherence_scores = {}
            geo_coherence = {}

        r = NLPProcessingResult(raw_text="x", swift_type="MT700")
        r.semantic = _DummySemantic()

        out = SwiftPipeline._serialize_nlp(r)
        sem = out.get("semantic") or {}
        self.assertIn("contradictions", sem)
        self.assertEqual(len(sem.get("contradictions") or []), 1)
        self.assertEqual((sem["contradictions"][0] or {}).get("tag_field"), "transhipment")


class TestSemanticContradictionsNLI(unittest.TestCase):
    def test_contradiction_detected_via_nli_mock(self):
        from services.swift_nlp.semantic.swift_semantic_analyzer import ContradictionEngine

        class _MockScores:
            def __init__(self, ent: float):
                self.entailment = ent
                self.contradiction = 1.0 - ent
                self.neutral = 0.0

        class _MockNLI:
            available = True

            def predict(self, premise: str, hypothesis: str):
                # Pretend the clause entails "prohibited" strongly.
                if "prohibited" in hypothesis.lower():
                    return _MockScores(0.90)
                return _MockScores(0.05)

        entities = {"transhipment": "ALLOWED"}
        txt = "Transshipment is prohibited unless the goods are containerized."
        out = ContradictionEngine.detect(entities, txt, nli=_MockNLI())
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(out[0].tag_field, "transhipment")


# ══════════════════════════════════════════════════════════════════════
# v4 — Tests décision 5 niveaux, impact_level, clause_type, Art.10/31/32/39
# ══════════════════════════════════════════════════════════════════════

class TestV4DecisionLevels(unittest.TestCase):
    """Tests de la décision 5 niveaux : REJECT / CONDITIONAL_HOLD / REVIEW / ACCEPT_WITH_FLAGS / ACCEPT."""

    def _f(self, sev, nature, blocks, code="X"):
        from types import SimpleNamespace
        return SimpleNamespace(severity=sev, finding_nature=nature, blocks_payment=blocks, code=code)

    def test_reject_true_violation_blocks_payment(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        findings = [self._f("error", "true_violation", True, "ART18_NO_THIRD_PARTY_INVOICE")]
        self.assertEqual(evaluate_ucp_decision(findings, False), "REJECT")

    def test_conditional_hold_contractual_modification_blocks(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        findings = [self._f("warning", "contractual_modification", True, "ART38I_EXCLUDED")]
        self.assertEqual(evaluate_ucp_decision(findings, False), "CONDITIONAL_HOLD")

    def test_review_true_violation_no_payment_block(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        findings = [self._f("error", "true_violation", False, "ART6_NO_EXPIRY")]
        self.assertEqual(evaluate_ucp_decision(findings, False), "REVIEW")

    def test_accept_with_flags_contractual_warning(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        findings = [self._f("warning", "contractual_modification", False, "ART12B_EXCLUDED")]
        self.assertEqual(evaluate_ucp_decision(findings, False), "ACCEPT_WITH_FLAGS")

    def test_accept_no_findings(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        self.assertEqual(evaluate_ucp_decision([], False), "ACCEPT")

    def test_payment_blocked_flag_still_triggers_reject(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        findings = [self._f("info", "contractual_modification", False)]
        self.assertEqual(evaluate_ucp_decision(findings, True), "REJECT")

    def test_operational_discrepancy_blocks_is_reject(self):
        from services.swift_nlp.ucp.swift_ucp_hard_rules import evaluate_ucp_decision
        findings = [self._f("error", "operational_discrepancy", True, "ART18_NO_THIRD_PARTY_INVOICE")]
        self.assertEqual(evaluate_ucp_decision(findings, False), "REJECT")


class TestV4ImpactAndClauseType(unittest.TestCase):
    """Tests de _compute_impact_level et _compute_clause_type."""

    def _finding(self, article, nature="true_violation", blocks=False, sev="error", tier=2):
        from services.swift_nlp.ucp.swift_ucp_validator import UCPFinding
        return UCPFinding(
            article=article, article_title=article, code=f"CODE_{article.replace('.','').upper()}",
            severity=sev, category="compliance", message="x",
            finding_nature=nature, blocks_payment=blocks, tier=tier,
        )

    def test_high_impact_true_violation_blocks(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_impact_level
        f = self._finding("Art.18", nature="true_violation", blocks=True)
        self.assertEqual(_compute_impact_level(f), "HIGH")

    def test_high_impact_art20_violation(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_impact_level
        f = self._finding("Art.20", nature="true_violation", blocks=False, sev="error")
        self.assertEqual(_compute_impact_level(f), "HIGH")

    def test_medium_impact_contractual_mod_non_critical(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_impact_level
        f = self._finding("Art.12b", nature="contractual_modification", blocks=False, sev="warning")
        self.assertEqual(_compute_impact_level(f), "MEDIUM")

    def test_high_impact_contractual_struct_art38i(self):
        from services.swift_nlp.ucp.swift_ucp_validator import UCPFinding, _compute_impact_level
        f = UCPFinding(
            article="Art.38i", article_title="Transfer", code="SEM_CONTRACT_STRUCT_ART38I",
            severity="warning", category="compliance", message="x",
            finding_nature="contractual_modification", blocks_payment=True,
        )
        self.assertEqual(_compute_impact_level(f), "HIGH")

    def test_low_impact_info_tier4(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_impact_level
        f = self._finding("Art.17", nature="true_violation", sev="info", tier=4)
        self.assertEqual(_compute_impact_level(f), "LOW")

    def test_clause_type_financial(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_clause_type
        for art in ["Art.18", "Art.38i", "Art.12b", "Art.37c"]:
            self.assertEqual(_compute_clause_type(self._finding(art)), "financial", f"Failed for {art}")

    def test_clause_type_transport(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_clause_type
        for art in ["Art.20", "Art.20c", "Art.31", "Art.32"]:
            self.assertEqual(_compute_clause_type(self._finding(art)), "transport", f"Failed for {art}")

    def test_clause_type_authentication(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_clause_type
        self.assertEqual(_compute_clause_type(self._finding("Art.17")), "authentication")

    def test_clause_type_identity(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_clause_type
        self.assertEqual(_compute_clause_type(self._finding("Art.14k")), "identity")

    def test_clause_type_compliance(self):
        from services.swift_nlp.ucp.swift_ucp_validator import _compute_clause_type
        for art in ["Art.1", "Art.5", "Art.10", "Art.6"]:
            self.assertEqual(_compute_clause_type(self._finding(art)), "compliance", f"Failed for {art}")


class TestV4ContractualHold(unittest.TestCase):
    """Tests du nouveau flag contractual_hold (distinct de payment_blocked)."""

    def test_contractual_hold_set_when_art38i_excluded(self):
        from services.swift_nlp.ucp.swift_ucp_validator import UCPFinding, UCPValidationReport
        rep = UCPValidationReport(
            lc_number="L5", swift_type="MT700", ucp_version="UCP 600",
            findings=[
                UCPFinding(
                    article="Art.38i", article_title="Transfer", code="ART38I_EXCLUDED",
                    severity="warning", category="compliance", message="Art.38i exclu",
                    finding_nature="contractual_modification", blocks_payment=True,
                    blocks_payment_reason="structural_transfer_constraint",
                ),
            ],
            global_score=70, compliant=False, payment_blocked=False, contractual_hold=True,
        )
        d = rep.to_dict()
        self.assertTrue(d.get("contractual_hold"))
        self.assertFalse(d.get("payment_blocked"))
        self.assertEqual(d.get("ucp_decision"), "CONDITIONAL_HOLD")

    def test_payment_blocked_not_contaminated_by_contractual_mod(self):
        from services.swift_nlp.ucp.swift_ucp_validator import UCPFinding, UCPValidationReport
        rep = UCPValidationReport(
            lc_number="L6", swift_type="MT700", ucp_version="UCP 600",
            findings=[
                UCPFinding(
                    article="Art.12b", article_title="Prepayment", code="ART12B_EXCLUDED",
                    severity="warning", category="payment", message="Art.12b exclu",
                    finding_nature="contractual_modification", blocks_payment=False,
                ),
            ],
            global_score=85, compliant=True, payment_blocked=False,
        )
        d = rep.to_dict()
        self.assertFalse(d.get("payment_blocked"))
        self.assertFalse(d.get("contractual_hold"))
        self.assertEqual(d.get("ucp_decision"), "ACCEPT_WITH_FLAGS")


class TestV4NewArticleChecks(unittest.TestCase):
    """Tests des nouveaux checks Art.10, Art.31, Art.32, Art.39."""

    def _validator(self):
        import os
        os.environ["UCP_USE_LEGACY_RULES"] = "true"
        from services.swift_nlp.ucp.swift_ucp_validator import UCP600Validator
        return UCP600Validator()

    def _payload(self, clauses, entities=None):
        base_entities = {
            "lc_number": "TEST", "swift_type": "MT700",
            "applicable_rules": "UCP 600",
            "expiry_date": "2026-12-31", "expiry_place": "Tunis",
        }
        if entities:
            base_entities.update(entities)
        return {
            "swift": {
                "entities": base_entities,
                "lc_criteria": {"documents": [{"name": "Invoice"}]},
                "nlp": {"clauses": clauses},
            }
        }

    def test_art10_partial_amendment_detected(self):
        import os
        os.environ["UCP_USE_LEGACY_RULES"] = "true"
        try:
            v = self._validator()
            payload = self._payload([{"text": "Partial acceptance of amendment is allowed under this credit."}])
            rep = v.validate(payload)
            codes = [f.code for f in rep.findings]
            self.assertIn("ART10_PARTIAL_AMENDMENT", codes)
            f = next(f for f in rep.findings if f.code == "ART10_PARTIAL_AMENDMENT")
            self.assertEqual(f.severity, "error")
            self.assertEqual(f.finding_nature, "true_violation")
        finally:
            del os.environ["UCP_USE_LEGACY_RULES"]

    def test_art39_assignment_prohibited_detected(self):
        import os
        os.environ["UCP_USE_LEGACY_RULES"] = "true"
        try:
            v = self._validator()
            payload = self._payload([{"text": "Assignment of proceeds is not allowed under this credit."}])
            rep = v.validate(payload)
            codes = [f.code for f in rep.findings]
            self.assertIn("ART39_ASSIGNMENT_PROHIBITED", codes)
            f = next(f for f in rep.findings if f.code == "ART39_ASSIGNMENT_PROHIBITED")
            self.assertEqual(f.finding_nature, "contractual_modification")
        finally:
            del os.environ["UCP_USE_LEGACY_RULES"]

    def test_art31_partial_prohibited_detected(self):
        import os
        os.environ["UCP_USE_LEGACY_RULES"] = "true"
        try:
            v = self._validator()
            payload = self._payload(
                clauses=[],
                entities={"partial_shipments": "PROHIBITED"},
            )
            rep = v.validate(payload)
            codes = [f.code for f in rep.findings]
            self.assertIn("ART31_PARTIAL_PROHIBITED", codes)
        finally:
            del os.environ["UCP_USE_LEGACY_RULES"]


class TestV4ArticleProfiles(unittest.TestCase):
    """Tests que les nouveaux profils (Art.1/5/10/12b/37c/39) sont bien chargés."""

    def test_new_articles_loaded(self):
        from services.swift_nlp.ucp.swift_ucp_clause_aligner import load_article_profiles
        profiles = load_article_profiles()
        arts = {p.article for p in profiles}
        for expected in ["Art.1", "Art.5", "Art.10", "Art.12b", "Art.37c", "Art.39"]:
            self.assertIn(expected, arts, f"{expected} manquant dans les profils")

    def test_all_profiles_have_clause_type_and_impact(self):
        from services.swift_nlp.ucp.swift_ucp_clause_aligner import load_article_profiles
        profiles = load_article_profiles()
        for p in profiles:
            self.assertIn(p.clause_type, {"financial", "transport", "authentication", "identity", "compliance", "document_format"},
                         f"{p.article}: clause_type invalide '{p.clause_type}'")
            self.assertIn(p.impact_level, {"HIGH", "MEDIUM", "LOW"},
                         f"{p.article}: impact_level invalide '{p.impact_level}'")

    def test_art5_is_high_compliance(self):
        from services.swift_nlp.ucp.swift_ucp_clause_aligner import load_article_profiles
        profiles = load_article_profiles()
        art5 = next(p for p in profiles if p.article == "Art.5")
        self.assertEqual(art5.clause_type, "compliance")
        self.assertEqual(art5.impact_level, "HIGH")

    def test_art17_is_authentication_high(self):
        from services.swift_nlp.ucp.swift_ucp_clause_aligner import load_article_profiles
        profiles = load_article_profiles()
        art17 = next(p for p in profiles if p.article == "Art.17")
        self.assertEqual(art17.clause_type, "authentication")
        self.assertEqual(art17.impact_level, "HIGH")


class TestV4SummaryFields(unittest.TestCase):
    """Tests que to_dict() expose les nouveaux champs v4."""

    def test_to_dict_contains_v4_fields(self):
        from services.swift_nlp.ucp.swift_ucp_validator import UCPFinding, UCPValidationReport
        finding = UCPFinding(
            article="Art.20", article_title="BL", code="ART20_NO_BLANK_BACK",
            severity="error", category="document", message="Blank back interdit",
            finding_nature="true_violation", blocks_payment=True,
            impact_level="HIGH", clause_type="transport",
        )
        rep = UCPValidationReport(
            lc_number="T1", swift_type="MT700", ucp_version="UCP 600",
            findings=[finding], global_score=60, compliant=False, payment_blocked=True,
        )
        d = rep.to_dict()
        # Décision
        self.assertEqual(d.get("ucp_decision"), "REJECT")
        # Nouveaux champs rapport
        self.assertIn("contractual_hold", d)
        # Champs finding
        fd = d["findings"][0]
        self.assertEqual(fd["impact_level"], "HIGH")
        self.assertEqual(fd["clause_type"], "transport")
        self.assertIn("finding_nature", fd)
        # Summary enrichi
        sm = d.get("summary", {})
        self.assertIn("true_violations", sm)
        self.assertIn("contractual_modifications", sm)
        self.assertIn("high_impact", sm)
        self.assertIn("medium_impact", sm)
        self.assertIn("low_impact", sm)


if __name__ == "__main__":
    unittest.main()
