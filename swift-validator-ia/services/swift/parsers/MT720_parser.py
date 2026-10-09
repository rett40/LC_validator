"""
mt720_parser.py  —  v5.1
Parseur complet pour MT720 — Transfer of a Documentary Credit.

v5.1 :
  - strip_label appliqué sur :27: :20: :21: :50: :59: :45A: :46A: :47A:
  - _extract_clean_reference partagé avec MT700
  - :71B: validé (confusion :48: évitée)
  - _clean_party_field pour premier/second bénéficiaire
  - extract_date supporte labels verbeux
"""

import re
from typing import Optional
from ..swift_base_parser import SwiftBaseParser
from ..entities import SwiftEntities


class MT720Parser(SwiftBaseParser):

    _RULES         = ["UCP LATEST VERSION", "UCPURR LATEST VERSION",
                      "EUCP LATEST VERSION", "EUCPURR LATEST VERSION", "OTHR"]
    _AVAIL_BY      = ["BY PAYMENT", "BY ACCEPTANCE", "BY NEGOTIATION",
                      "BY DEF PAYMENT", "BY MIXED PYMT"]
    _SHIPMENT_OPTS = ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]
    _CONFIRM_OPTS  = ["CONFIRM", "MAY ADD", "WITHOUT"]

    def parse(self, text: str) -> SwiftEntities:
        e = SwiftEntities(swift_type="MT720")

        # ── Références ────────────────────────────────────────────────
        raw_20 = self.extract_tag(text, ":20:")
        e.lc_number = self._extract_clean_reference(raw_20)

        raw_21 = self.extract_tag(text, ":21:")
        e.related_lc = self._extract_clean_reference(raw_21)

        raw_27 = self.extract_tag(text, ":27:")
        e.sequence = self.strip_label(raw_27, r'\d+/\d+')

        # ── Forme du crédit :40B: ─────────────────────────────────────
        # :40B: : type LC + niveau confirmation (2 lignes)
        raw_40b = self.extract_tag_lines(text, ":40B:")
        if raw_40b:
            # Nettoyer le label "FORM OF DOCUMENTARY CREDIT" de la première ligne
            first = re.sub(
                r'^(?:FORM\s+OF\s+(?:DOCUMENTARY\s+)?CREDIT)\s*',
                '', raw_40b[0], flags=re.IGNORECASE
            ).strip()
            e.lc_form = first if first else raw_40b[0].strip()
            if len(raw_40b) > 1:
                e.raw_fields["40B_confirmation"] = raw_40b[1].strip()

        # ── Règles applicables ────────────────────────────────────────
        e.applicable_rules = self.normalize_enum(
            self.extract_tag(text, ":40E:"), self._RULES
        )

        # ── Dates ─────────────────────────────────────────────────────
        e.issue_date = self.extract_date(text, ":31C:")
        e.expiry_date, e.expiry_place = self.extract_date_and_place(text, ":31D:")
        e.shipment_date = self.extract_date(text, ":44C:")

        # ── Banque émettrice du crédit original :52a: ─────────────────
        e.issuing_bank = (
            self.extract_tag_multiline(text, ":52A:")
            or self.extract_tag_multiline(text, ":52D:")
            or self.extract_tag_multiline(text, ":52a:")
        )

        # ── Parties ───────────────────────────────────────────────────
        # :50: = Premier bénéficiaire (First Beneficiary)
        # :59: = Second bénéficiaire (nouveau bénéficiaire)
        raw_50 = self.extract_tag_multiline(text, ":50:")
        raw_59 = self.extract_tag_multiline(text, ":59:")

        e.applicant   = self._clean_party_field(raw_50, first=True)
        e.beneficiary = self._clean_party_field(raw_59, first=False)

        # ── Montant et tolérance ──────────────────────────────────────
        e.currency, e.amount = self.extract_amount(text, ":32B:")
        e.tolerance_plus, e.tolerance_minus = self.extract_tolerance(text, ":39A:")

        # ── Disponibilité ─────────────────────────────────────────────
        avail_lines = (
            self.extract_tag_lines(text, ":41A:")
            or self.extract_tag_lines(text, ":41D:")
            or self.extract_tag_lines(text, ":41a:")
        )
        if avail_lines:
            mode_found = None
            bank_found = None
            for line in avail_lines:
                norm = self.normalize_enum(line, self._AVAIL_BY)
                if norm in self._AVAIL_BY:
                    mode_found = norm
                elif not bank_found:
                    bank_found = line
            e.available_with = mode_found
            e.available_by   = mode_found   # mode de règlement (champ déclaré)
            if bank_found and not e.issuing_bank:
                e.raw_fields["41_bank"] = bank_found
        e.draft_at = self.extract_tag(text, ":42C:")

        # ── Expédition ────────────────────────────────────────────────
        e.partial_shipments = self.normalize_enum(
            self.extract_tag(text, ":43P:"), self._SHIPMENT_OPTS
        )
        e.transhipment = self.normalize_enum(
            self.extract_tag(text, ":43T:"), self._SHIPMENT_OPTS
        )

        # ── Ports ─────────────────────────────────────────────────────
        e.port_loading = (
            self.extract_tag(text, ":44A:")
            or self.extract_tag(text, ":44E:")
        )
        e.port_discharge = (
            self.extract_tag(text, ":44B:")
            or self.extract_tag(text, ":44F:")
        )

        # ── Marchandises et documents ─────────────────────────────────
        e.goods_description     = self.extract_tag_multiline(text, ":45A:")
        e.documents_required    = self.extract_tag_multiline(text, ":46A:")
        e.additional_conditions = self.extract_tag_multiline(text, ":47A:")

        # ── Présentation ──────────────────────────────────────────────
        days, qualifier       = self.extract_presentation_period(text, ":48:")
        e.presentation_period = days
        if qualifier:
            e.raw_fields["48_qualifier"] = qualifier

        e.confirmation = self.normalize_enum(
            self.extract_tag(text, ":49:"), self._CONFIRM_OPTS
        )

        # ── Frais ─────────────────────────────────────────────────────
        raw_charges = self.extract_tag_multiline(text, ":71B:")
        if self.is_charges_value(raw_charges):
            e.charges = raw_charges
        else:
            e.raw_fields["71B_raw"] = raw_charges

        # ── Informations ──────────────────────────────────────────────
        e.sender_info = self.extract_tag(text, ":72:")

        return e

    # ------------------------------------------------------------------
    # Helpers MT720
    # ------------------------------------------------------------------

    def _extract_clean_reference(self, raw: Optional[str]) -> Optional[str]:
        """
        Extrait la référence réelle depuis un champ verbeux.
        Ex: "Transferring Bank's Reference XYZ123LCEXP 21 Documentary Credit Number ABC999"
            → 'XYZ123LCEXP' (premier token de 6+ car. après le label)
        """
        if not raw:
            return None

        # Supprimer les labels connus un à un
        cleaned = re.sub(
            r'\s*(?:TRANSFERRING\s+BANK[\'S\s]*REFERENCE|'
            r'DOCUMENTARY\s+CREDIT\s+NUMBER|'
            r'DATE\s+OF\s+ISSUE|REFERENCE\s+NUMBER|'
            r'TRANSACTION\s+REFERENCE)\s*',
            ' ', raw, flags=re.IGNORECASE
        ).strip()

        # Prendre le premier token alphanum de 6+ caractères
        tokens = re.findall(r'[A-Z0-9]{6,}', cleaned.upper())
        if tokens:
            return tokens[0]

        return cleaned if cleaned else raw

    def _clean_party_field(self, value: Optional[str], first: bool = True) -> Optional[str]:
        """
        Nettoie les champs :50: (premier bénéficiaire) et :59: (second bénéficiaire).
        Supprime les labels 'FIRST BENEFICIARY', 'SECOND BENEFICIARY', 'BENEFICIARY', etc.
        """
        if not value:
            return None
        cleaned = re.sub(
            r'^(?:FIRST\s+BENEFICIARY|SECOND\s+BENEFICIARY|BENEFICIARY|'
            r'APPLICANT|ORDERING\s+PARTY|INSTRUCTING\s+PARTY)\s*[:\-]?\s*',
            '', value, flags=re.IGNORECASE
        ).strip()

        # Supprimer aussi ce qui suit si ça ressemble à un label de champ suivant
        next_label = re.search(
            r'\b(?:SECOND\s+BENEFICIARY|CURRENCY\s+CODE|AMOUNT|'
            r'AVAILABLE\s+WITH|PARTIAL\s+SHIPMENTS)\b',
            cleaned, re.IGNORECASE
        )
        if next_label:
            cleaned = cleaned[:next_label.start()].strip()

        return cleaned if cleaned else value