"""
entities.py
Dataclasses enrichies pour tous les types SWIFT supportés.
MT700, MT720, MT760, MT790, MT799.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class SwiftEntities:
    """Entités extraites d'un message SWIFT (générique tous types)."""

    swift_type: str

    # ── Références ────────────────────────────────────────────────────
    lc_number:   Optional[str] = None   # :20:
    related_lc:  Optional[str] = None   # :21:
    sequence:    Optional[str] = None   # :27:

    # ── Dates ─────────────────────────────────────────────────────────
    issue_date:  Optional[str] = None   # :31C: / :30:
    expiry_date: Optional[str] = None   # :31D:
    expiry_place:Optional[str] = None   # :31D: partie lieu
    value_date:  Optional[str] = None   # :32A: (MT790)
    shipment_date: Optional[str] = None # :44C:

    # ── Montant ───────────────────────────────────────────────────────
    currency:    Optional[str] = None
    amount:      Optional[str] = None
    tolerance_plus:  Optional[str] = None   # :39A: plus
    tolerance_minus: Optional[str] = None   # :39A: minus

    # ── Parties ───────────────────────────────────────────────────────
    applicant:        Optional[str] = None   # :50:
    beneficiary:      Optional[str] = None   # :59:
    issuing_bank:     Optional[str] = None   # :52a:
    advising_bank:      Optional[str] = None   # :57a:
    reimbursement_bank: Optional[str] = None   # :53a:
    applicant_bank:     Optional[str] = None   # :51a:
    drawee:             Optional[str] = None   # :42a: (drawee / tiré)
    available_with:   Optional[str] = None   # :41a: (banque / place)
    available_by:     Optional[str] = None   # :41a: (mode de règlement)

    # ── Conditions LC (MT700/MT720) ───────────────────────────────────
    lc_form:              Optional[str] = None   # :40A: / :40B:
    applicable_rules:     Optional[str] = None   # :40E: / :40C:
    draft_at:             Optional[str] = None   # :42C:
    mixed_payment_details:    Optional[str] = None   # :42M:
    deferred_payment_details: Optional[str] = None   # :42P:
    max_amount:           Optional[str] = None   # :39B:
    additional_amounts:   Optional[str] = None   # :39C:
    partial_shipments:    Optional[str] = None   # :43P:
    transhipment:         Optional[str] = None   # :43T:
    port_loading:         Optional[str] = None   # :44A: / :44E:
    port_discharge:       Optional[str] = None   # :44B: / :44F:
    shipment_period:      Optional[str] = None   # :44D:
    goods_description:    Optional[str] = None   # :45A:
    documents_required:   Optional[str] = None   # :46A:
    additional_conditions:Optional[str] = None   # :47A:
    charges:              Optional[str] = None   # :71B: / :71D:
    presentation_period:  Optional[str] = None   # :48:
    confirmation:         Optional[str] = None   # :49:
    instructions:         Optional[str] = None   # :78:
    pre_advice:           Optional[str] = None   # :23:

    # ── Garantie (MT760) ──────────────────────────────────────────────
    undertaking_type:   Optional[str] = None   # :22D:
    purpose:            Optional[str] = None   # :22A:
    expiry_type:        Optional[str] = None   # :23B:
    guarantee_text:     Optional[str] = None   # :77U:

    # ── Frais (MT790) ─────────────────────────────────────────────────
    account:            Optional[str] = None   # :25:
    charge_details:     Optional[str] = None   # :71B:

    # ── Message libre (MT799) ─────────────────────────────────────────
    free_text:          Optional[str] = None   # :79:

    # ── Méta ──────────────────────────────────────────────────────────
    sender_info:        Optional[str] = None   # :72:
    message:            Optional[str] = None   # :77A: / :79:

    raw_fields: Dict[str, str]  = field(default_factory=dict)
    warnings:   List[str]       = field(default_factory=list)