"""
mt760_parser.py  —  v5.1
Parseur complet pour MT760 — Demand Guarantee / Standby Letter of Credit.

v5.1 :
  - guarantee_text : lookahead NEXT inclut ':71D:' et ':71B:' pour éviter le
    débordement du texte de garantie dans les frais
  - _extract_parties : nettoyage labels verbeux via _clean_labeled_value de la base
  - _extract_amount_from_labeled : gère séparateur décimal européen
  - Fallback expiry_date/place depuis :31D: si :31E: absent
"""

import re
from typing import Optional, Tuple
from ..swift_base_parser import SwiftBaseParser
from ..entities import SwiftEntities


class MT760Parser(SwiftBaseParser):

    _PURPOSE_VALS = ["ISSU", "ISCO", "ICCO", "ADVI", "ACNF"]
    _UNDERTAKING  = ["DGAR", "STBY"]
    _EXPIRY_TYPES = ["FIXD", "COND", "OPEN"]
    _RULES        = ["URDG", "UCP", "ISPR", "NONE", "OTHR"]

    def parse(self, text: str) -> SwiftEntities:
        e = SwiftEntities(swift_type="MT760")

        # :27:
        seq_raw = self.extract_tag(text, ":27:")
        e.sequence = self.strip_label(seq_raw, r'\d+/\d+')

        # :20:
        ref_raw = self.extract_tag(text, ":20:")
        e.lc_number = self._extract_reference(ref_raw)

        # Objet et forme
        e.purpose          = self.normalize_enum(self.extract_tag(text, ":22A:"), self._PURPOSE_VALS)
        e.undertaking_type = self.normalize_enum(self.extract_tag(text, ":22D:"), self._UNDERTAKING)
        e.expiry_type      = self.normalize_enum(self.extract_tag(text, ":23B:"), self._EXPIRY_TYPES)

        # :40C: règles
        e.applicable_rules = self._extract_rules(self.extract_tag(text, ":40C:"))

        # :30: date d'émission
        e.issue_date = self._extract_date_from_labeled(self.extract_tag(text, ":30:"))

        # :31E: puis :31D: en fallback
        e.expiry_date, e.expiry_place = self._extract_expiry_date_place(
            self.extract_tag(text, ":31E:")
        )
        if not e.expiry_date:
            e.expiry_date, e.expiry_place = self.extract_date_and_place(text, ":31D:")

        # :32B: montant
        e.currency, e.amount = self._extract_amount_from_labeled(
            self.extract_tag(text, ":32B:")
        )

        # Parties
        self._extract_parties(text, e)

        # :77U: texte garantie — stopper AVANT :71D:/:71B: pour éviter débordement
        guarantee_raw = self._extract_guarantee_text(text)
        e.guarantee_text = self._strip_field_label(
            guarantee_raw,
            r'(?:TERMS\s+AND\s+CONDITIONS|PAYABLE\s+AGAINST)\s*[:\-]?\s*'
        )
        if not e.guarantee_text:
            e.guarantee_text = self.extract_tag_multiline(text, ":77A:")

        # :71D: / :71B: frais
        charges_raw = (
            self.extract_tag_multiline(text, ":71D:")
            or self.extract_tag_multiline(text, ":71B:")
        )
        if not charges_raw:
            m_charges = re.search(
                r'(?:Z1D|71D|71B)\s*:?\s*(CHARGES?\b.{5,200}?)(?=\s*(?:\d{2}[A-Z]?:|$))',
                text, re.IGNORECASE
            )
            if not m_charges:
                m_charges = re.search(
                    r'\bCHARGES\b(.{5,200}?)(?=\s*(?:\d{2}[A-Z]?:|$))',
                    text, re.IGNORECASE
                )
            charges_raw = m_charges.group(0) if m_charges else None

        e.charges = self._strip_field_label(
            charges_raw,
            r'(?:Z1D|71[DB])\s*:?\s*CHARGES?\s*[:\-]?\s*'
        )
        if not e.charges and charges_raw:
            e.charges = charges_raw.strip()

        # :72: sender info
        sender_raw = self.extract_tag_multiline(text, ":72:")
        e.sender_info = self._strip_field_label(
            sender_raw,
            r'SENDER\s+TO\s+RECEIVER\s+INFORMATION\s*[:\-]?\s*'
        )

        return e

    # ------------------------------------------------------------------
    # Helpers spécifiques MT760
    # ------------------------------------------------------------------

    def _extract_guarantee_text(self, text: str) -> Optional[str]:
        """
        Extrait :77U: en s'assurant que la valeur ne déborde pas sur :71D:/:71B:.
        Utilise un lookahead plus strict que _extract_raw standard.
        """
        tag_code = "77U"
        tc = re.escape(tag_code)
        normalized = self._normalize_ocr_text(text)

        # Lookahead étendu : s'arrêter aussi sur :71D:, :71B:, Z1D:
        NEXT_77U = (
            r'(?=(?:^|\n|\s+):?(?:\d{2}[A-Z]?|Z1D|7[lL1][DB]):|'
            r'\Z|'
            r'(?:CHARGES?\s*(?:ALL\s+BANK|FOR\s+(?:ACCOUNT|BENEFICIARY)))|'
            r'(?:\n\s*(?:Z1D|71D|71B)\s*:))'
        )

        for src, pattern in [
            (normalized, rf'(?:^|\n)\s*:{tc}:\s*(.*?){NEXT_77U}'),
            (normalized, rf'(?:^|\n)\s*{tc}:\s*(.*?){NEXT_77U}'),
            (text,       rf':{tc}:\s*(.*?){NEXT_77U}'),
        ]:
            m = re.search(pattern, src, re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                value = " ".join(lines).strip()
                value = re.sub(r'^:\s*', '', value)
                # Supprimer les résidus de tags en fin de valeur
                value = re.sub(r'\s*:?\d{2}[A-Z]?\s*:\s*$', '', value).strip()
                if value:
                    return value

        return None

    def _extract_reference(self, raw: Optional[str]) -> Optional[str]:
        """
        Extrait la référence réelle depuis un champ avec label.
        Ex: 'TRANSATTION REFERENCE 61234567890' → '61234567890'
        """
        if not raw:
            return None
        tokens = re.findall(r'[A-Z0-9]{6,}', raw.upper())
        return tokens[-1] if tokens else raw.strip()

    def _extract_date_from_labeled(self, raw: Optional[str]) -> Optional[str]:
        """Extrait une date depuis un champ avec label textuel."""
        if not raw:
            return None
        return self._find_date_in_string(raw)

    def _extract_expiry_date_place(self, raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrait (date_iso, lieu) depuis :31E: avec label.
        Ex: 'DATE AND PLACE OF EXPIRY 12-OCT-2021 AT TUNIS, TUNISIA'
            → ('2021-10-12', 'TUNIS, TUNISIA')
        """
        if not raw:
            return None, None
        iso = self._find_date_in_string(raw)
        place = None
        m = re.search(r'\bAT\s+([A-Z][A-Z\s,\.]+?)(?:\s+AT\s+|\s*$)', raw, re.IGNORECASE)
        if m:
            place = m.group(1).strip().rstrip(',').strip()
        return iso, place

    def _extract_amount_from_labeled(self, raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrait (devise, montant) depuis un champ avec label.
        Ex: 'GUARANTEE AMOUNT USD500,000.00' → ('USD', '500000.00')
        Gère aussi le séparateur décimal européen.
        """
        if not raw:
            return None, None
        m = re.search(r'([A-Z]{3})\s*([0-9][0-9,\.]+)', raw)
        if m:
            return self._parse_currency_amount(m.group(1) + m.group(2))
        return None, None

    def _extract_parties(self, text: str, e: SwiftEntities) -> None:
        """
        Résout applicant / bénéficiaire par le label textuel dans la valeur.
        """
        for tag in (":50:", ":50B:", ":50a:", ":59:", ":59A:"):
            val = self.extract_tag_multiline(text, tag)
            if not val:
                continue
            upper = val.upper()
            clean = self._strip_field_label(
                val,
                r'(?:APPLICANT|BENEFICIARY|ORDERING\s+PARTY|INSTRUCTING\s+PARTY)\s*[:\-]?\s*'
            )
            if 'BENEFICIARY' in upper:
                if not e.beneficiary:
                    e.beneficiary = clean
            elif 'APPLICANT' in upper or 'ORDERING' in upper or 'INSTRUCTING' in upper:
                if not e.applicant:
                    e.applicant = clean
            else:
                if tag.startswith(':50') and not e.applicant:
                    e.applicant = clean
                elif tag.startswith(':59') and not e.beneficiary:
                    e.beneficiary = clean

    def _extract_rules(self, raw: Optional[str]) -> Optional[str]:
        """
        Valide que :40C: est bien un code de règles et non un applicant mal segmenté.
        """
        if not raw:
            return None
        upper = raw.upper().strip()
        for rule in self._RULES:
            if rule in upper:
                return rule
        biz_keywords = ['LTD', 'CO.', 'CORP', 'SARL', 'IMPORTS', 'TRADING', 'COMPANY', 'INC']
        if any(kw in upper for kw in biz_keywords):
            return None
        return upper if len(upper) <= 10 else None

    @staticmethod
    def _strip_field_label(value: Optional[str], label_pattern: str) -> Optional[str]:
        """Retire un label textuel en début de valeur."""
        if not value:
            return None
        cleaned = re.sub(r'^\s*' + label_pattern, '', value, flags=re.IGNORECASE).strip()
        return cleaned if cleaned else value.strip()