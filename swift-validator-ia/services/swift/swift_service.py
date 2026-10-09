"""
swift_service.py  —  v5 (avec parseurs dédiés)
Pipeline complet :
  1. normalize_text()              — nettoyage OCR
  2. SwiftDetector.detect()        — détection type MT*** pondérée
  3. SwiftParserFactory.parse()    — parsing dédié par type → SwiftEntities
  4. LCAcceptanceCriteriaExtractor — critères d'acceptation LC (MT700/720)
  5. SwiftValidator.validate()     — validation → ValidationReport
"""

from __future__ import annotations
from pathlib import Path

from .schema_loader           import SwiftSchemaLoader
from .schema_parser           import normalize_text
from .swift_detector          import SwiftDetector
from .parsers.parser_factory  import SwiftParserFactory
from .lc_acceptance_criteria  import LCAcceptanceCriteriaExtractor
from .swift_validator         import SwiftValidator


class SwiftService:

    def __init__(self, schema_dir: Path | None = None):
        base_dir = Path(__file__).resolve().parents[2]
        self.schema_loader = SwiftSchemaLoader(base_dir=base_dir, schema_dir=schema_dir)
        self.detector      = SwiftDetector(self.schema_loader.schemas)
        self.parser_factory= SwiftParserFactory()
        self.lc_extractor  = LCAcceptanceCriteriaExtractor()
        self.validator     = SwiftValidator()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def process(self, text: str) -> dict:
        if not text or not text.strip():
            return {"success": False, "error": "Texte vide"}

        cleaned = normalize_text(text)

        # ── 1. Détection ──────────────────────────────────────────────
        swift_type       = self.detector.detect(cleaned)
        detection_scores = self.detector.detect_with_scores(cleaned)

        if swift_type == "UNKNOWN":
            return {
                "success":          False,
                "error":            "Type SWIFT non détecté",
                "detection_scores": detection_scores,
            }

        # ── 2. Parsing (parseur dédié) ────────────────────────────────
        try:
            entities = self.parser_factory.parse(cleaned, swift_type)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        # ── 3. Critères d'acceptation LC ─────────────────────────────
        lc_criteria = None
        if swift_type in ("MT700", "MT720"):
            lc_criteria = self.lc_extractor.extract(entities).to_dict()

        # ── 4. Validation ─────────────────────────────────────────────
        report = self.validator.validate(entities)

        return {
            "success":          True,
            "swift_type":       swift_type,
            "detection_scores": detection_scores,
            "entities":         self._entities_dict(entities),
            "lc_criteria":      lc_criteria,
            "validation":       report.to_dict(),
        }

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _entities_dict(e) -> dict:
        return {
            "swift_type":             e.swift_type,
            "lc_number":              e.lc_number,
            "related_lc":             e.related_lc,
            "sequence":               e.sequence,
            "issue_date":             e.issue_date,
            "expiry_date":            e.expiry_date,
            "expiry_place":           e.expiry_place,
            "value_date":             e.value_date,
            "shipment_date":          e.shipment_date,
            "currency":               e.currency,
            "amount":                 e.amount,
            "tolerance_plus":         e.tolerance_plus,
            "tolerance_minus":        e.tolerance_minus,
            "applicant":              e.applicant,
            "beneficiary":            e.beneficiary,
            "issuing_bank":           e.issuing_bank,
            "advising_bank":          e.advising_bank,
            "reimbursement_bank":     e.reimbursement_bank,
            "applicant_bank":         e.applicant_bank,
            "drawee":                 e.drawee,
            "available_with":         e.available_with,
            "available_by":           e.available_by,
            "lc_form":                e.lc_form,
            "applicable_rules":       e.applicable_rules,
            "draft_at":               e.draft_at,
            "mixed_payment_details":      e.mixed_payment_details,
            "deferred_payment_details":   e.deferred_payment_details,
            "max_amount":             e.max_amount,
            "additional_amounts":     e.additional_amounts,
            "partial_shipments":      e.partial_shipments,
            "transhipment":           e.transhipment,
            "port_loading":           e.port_loading,
            "port_discharge":         e.port_discharge,
            "shipment_period":        e.shipment_period,
            "goods_description":      e.goods_description,
            "documents_required":     e.documents_required,
            "additional_conditions":  e.additional_conditions,
            "charges":                e.charges,
            "presentation_period":    e.presentation_period,
            "confirmation":           e.confirmation,
            "instructions":           e.instructions,
            "pre_advice":             e.pre_advice,
            "purpose":                e.purpose,
            "undertaking_type":       e.undertaking_type,
            "expiry_type":            e.expiry_type,
            "guarantee_text":         e.guarantee_text,
            "account":                e.account,
            "charge_details":         e.charge_details,
            "free_text":              e.free_text,
            "sender_info":            e.sender_info,
            "warnings":               e.warnings,
            "raw_fields":             e.raw_fields,
        }