"""
mt700_parser.py  —  v5.1
Parseur complet pour MT700 — Issue of a Documentary Credit.

v5.1 :
  - strip_label appliqué sur :27: :20: :50: :59: :45A: :46A: :47A:
  - :71B: validé (pas de confusion avec :48:)
  - :32B: cherche devise+montant même si label devant
  - ports : nettoyage label verbeux
"""
from typing import Optional
import re
from ..swift_base_parser import SwiftBaseParser
from ..entities import SwiftEntities


class MT700Parser(SwiftBaseParser):

    _LC_FORMS      = ["IRREVOCABLE", "IRREVOCABLE TRANSFERABLE"]
    _RULES         = ["UCP LATEST VERSION", "UCPURR LATEST VERSION",
                      "EUCP LATEST VERSION", "EUCPURR LATEST VERSION", "OTHR"]
    _AVAIL_BY      = ["BY PAYMENT", "BY ACCEPTANCE", "BY NEGOTIATION",
                      "BY DEF PAYMENT", "BY MIXED PYMT"]
    _SHIPMENT_OPTS = ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]
    _CONFIRM_OPTS  = ["CONFIRM", "MAY ADD", "WITHOUT"]

    def parse(self, text: str) -> SwiftEntities:
        e = SwiftEntities(swift_type="MT700")

        # ── Références ────────────────────────────────────────────────
        # strip_label intégré dans _clean_labeled_value via extract_tag
        raw_20 = self.extract_tag(text, ":20:")
        e.lc_number = self._extract_clean_reference(raw_20)

        raw_27 = self.extract_tag(text, ":27:")
        e.sequence = self.strip_label(raw_27, r'\d+/\d+')

        # ── Forme et règles ───────────────────────────────────────────
        e.lc_form = self.normalize_enum(
            self.extract_tag(text, ":40A:"), self._LC_FORMS
        )
        e.applicable_rules = self.normalize_enum(
            self.extract_tag(text, ":40E:"), self._RULES
        )

        # ── Dates ─────────────────────────────────────────────────────
        e.issue_date = self.extract_date(text, ":31C:")
        e.expiry_date, e.expiry_place = self.extract_date_and_place(text, ":31D:")
        e.shipment_date = self.extract_date(text, ":44C:")

        # ── Montant et tolérance ──────────────────────────────────────
        e.currency, e.amount = self.extract_amount(text, ":32B:")
        e.tolerance_plus, e.tolerance_minus = self.extract_tolerance(text, ":39A:")

        # ── Montant maximum / montants additionnels ───────────────────
        e.max_amount         = self.extract_tag(text, ":39B:")
        e.additional_amounts = self.extract_tag_multiline(text, ":39C:")

        # ── Parties ───────────────────────────────────────────────────
        # _clean_labeled_value supprime "APPLICANT", "FIRST BENEFICIARY" etc.
        e.applicant   = self.extract_tag_multiline(text, ":50:")
        e.beneficiary = self.extract_tag_multiline(text, ":59:")

        # Nettoyage supplémentaire si label résiduel détecté
        e.applicant   = self._clean_party_field(e.applicant)
        e.beneficiary = self._clean_party_field(e.beneficiary)

        # ── Banques ───────────────────────────────────────────────────
        e.issuing_bank      = self._first_tag(text, ":52A:", ":52D:")   # banque émettrice
        if not e.issuing_bank:
            # Fallback : BIC émetteur depuis l'en-tête SWIFT {1:F01<BIC>...}
            e.issuing_bank = self._issuing_bank_from_header(text)
        e.advising_bank     = self._first_tag(text, ":57A:", ":57D:", ":57B:")  # advise through
        e.reimbursement_bank= self._first_tag(text, ":53A:", ":53D:")   # banque de remboursement
        e.applicant_bank    = self._first_tag(text, ":51A:", ":51D:")   # banque du donneur d'ordre

        # ── Disponibilité ─────────────────────────────────────────────
        avail_lines = (
            self.extract_tag_lines(text, ":41A:")
            or self.extract_tag_lines(text, ":41D:")
        )
        if avail_lines:
            e.available_with, e.available_by = self._parse_field_41(avail_lines)
        e.draft_at = self.extract_tag(text, ":42C:")
        e.drawee   = self._first_tag(text, ":42A:", ":42D:")            # tiré
        e.mixed_payment_details    = self.extract_tag_multiline(text, ":42M:")
        e.deferred_payment_details = self.extract_tag_multiline(text, ":42P:")

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
        e.shipment_period = self.extract_tag_multiline(text, ":44D:")

        # ── Description marchandises et documents ─────────────────────
        e.goods_description     = self.extract_tag_multiline(text, ":45A:")
        e.documents_required    = self.extract_tag_multiline(text, ":46A:")
        e.additional_conditions = self.extract_tag_multiline(text, ":47A:")

        # ── Présentation et confirmation ──────────────────────────────
        days, qualifier       = self.extract_presentation_period(text, ":48:")
        e.presentation_period = days
        if qualifier:
            e.raw_fields["48_qualifier"] = qualifier

        # :49: Confirmation — valeur CONTRAINTE : CONFIRM / MAY ADD / WITHOUT.
        # Toute autre réponse (ex: n° de tél. capté par OCR) est ignorée ; si
        # aucune valeur valide n'est trouvée, le champ reste None (manquant).
        e.confirmation = self.extract_enum_strict(text, ":49:", self._CONFIRM_OPTS)

        # ── Frais ─────────────────────────────────────────────────────
        # Validation : :71B: (standard) et :71D: (alias Banque Zitouna non standard)
        raw_charges = (
            self.extract_tag_multiline(text, ":71B:") or
            self.extract_tag_multiline(text, ":71D:")
        )
        if self.is_charges_value(raw_charges):
            e.charges = raw_charges
        else:
            # Ce n'est pas des frais (ex: "21 DAYS", "SEE F47A") → ignorer
            e.raw_fields["71B_raw"] = raw_charges

        # ── Pré-avis / instructions / informations émetteur ───────────
        e.pre_advice   = self.extract_tag(text, ":23:")
        e.instructions = self.extract_tag_multiline(text, ":78:")
        e.sender_info  = self.extract_tag(text, ":72:")

        # ── Post-traitement : placeholders + complétude ───────────────
        self._drop_placeholders(e)
        self._add_completeness_warnings(e)

        return e

    # ------------------------------------------------------------------
    # Helpers MT700
    # ------------------------------------------------------------------

    def _first_tag(self, text: str, *tags: str) -> Optional[str]:
        """Retourne la valeur du premier tag présent parmi `tags`.
        Sert aux champs à options de lettre (ex: :52A:/:52D:)."""
        for tag in tags:
            value = self.extract_tag_multiline(text, tag)
            if value:
                return value
        return None

    # Champs texte susceptibles de contenir un placeholder de template
    # (XXXXXXX, ADDRESS, AMOUNT…) — on les annule via is_placeholder().
    _PLACEHOLDER_FIELDS = (
        "lc_number", "applicant", "beneficiary", "issuing_bank", "advising_bank",
        "reimbursement_bank", "applicant_bank", "drawee",
        "port_loading", "port_discharge", "expiry_place", "goods_description",
    )

    # Champs obligatoires MT700 (cf. data/swift_schema/mt700_schema.json)
    _MANDATORY_FIELDS = (
        ("lc_number",        ":20:"),
        ("lc_form",          ":40A:"),
        ("applicable_rules", ":40E:"),
        ("issue_date",       ":31C:"),
        ("expiry_date",      ":31D:"),
        ("applicant",        ":50:"),
        ("beneficiary",      ":59:"),
        ("goods_description",":45A:"),
    )

    def _drop_placeholders(self, e: SwiftEntities) -> None:
        """Annule les valeurs qui ne sont que des placeholders de template."""
        for fname in self._PLACEHOLDER_FIELDS:
            value = getattr(e, fname, None)
            if value and self.is_placeholder(value):
                setattr(e, fname, None)

    def _add_completeness_warnings(self, e: SwiftEntities) -> None:
        """Ajoute un warning pour chaque champ obligatoire absent."""
        for fname, tag in self._MANDATORY_FIELDS:
            if not getattr(e, fname, None):
                e.warnings.append(f"Champ obligatoire manquant : {tag}")
        if not (e.currency and e.amount):
            e.warnings.append("Champ obligatoire manquant : :32B: (devise/montant)")

    def _parse_field_41(self, lines: list) -> tuple:
        """Parse le tag :41A:/:41D: → (available_with, available_by).

        Gère les deux formats SWIFT :
          Multiline standard : ['ATBKTNTTXXX', 'BY PAYMENT']
          Inline OCR         : ['ATBKTNTTXXX BY PAYMENT']

        Retourne (bic_ou_nom_banque, modalite_reglement).
        Ex : ('ATBKTNTTXXX', 'BY PAYMENT')
             (None, 'BY PAYMENT')   si pas de BIC
        """
        _INLINE_RE = re.compile(
            r'^(?P<bic>[A-Z0-9]{8,11}(?:XXX)?)\s+'
            r'(?P<mode>BY\s+(?:PAYMENT|ACCEPTANCE|NEGOTIATION|'
            r'DEF(?:ERRED)?\s+PAYMENT|MIXED\s+PYMT))\s*$',
            re.IGNORECASE
        )
        _BIC_RE = re.compile(r'^[A-Z0-9]{8,11}(?:XXX)?$', re.IGNORECASE)
        # Mode de règlement 'BY …' repérable n'importe où dans la ligne
        _MODE_RE = re.compile(
            r'\bBY\s+(?:PAYMENT|ACCEPTANCE|NEGOTIATION|'
            r'DEF(?:ERRED)?\s+PAYMENT|MIXED\s+PYMT)\b',
            re.IGNORECASE,
        )

        bank_found = None
        mode_found = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Cas inline strict : "ATBKTNTTXXX BY PAYMENT"
            m = _INLINE_RE.match(line)
            if m:
                if not bank_found:
                    bank_found = m.group("bic").strip()
                if not mode_found:
                    mode_found = m.group("mode").strip().upper()
                continue

            # La ligne contient un mode 'BY …' (souvent collé au nom de banque
            # après aplatissement multiligne : "DEUTSCH BANK AG … BY PAYMENT").
            mm = _MODE_RE.search(line)
            if mm:
                if not mode_found:
                    mode_found = self.normalize_enum(mm.group(0), self._AVAIL_BY)
                prefix = line[:mm.start()].strip(" ,.-")
                if prefix and not bank_found:
                    bank_found = prefix
                continue

            # Cas BIC seul (8-11 chars alphanumériques)
            if _BIC_RE.match(line) and not bank_found:
                bank_found = line.strip()
                continue

            # Nom libre de banque (ex: "ANY BANK", "DEUTSCH BANK AG"),
            # éventuellement réparti sur plusieurs lignes (nom + localité).
            if not mode_found:
                bank_found = f"{bank_found} {line}".strip() if bank_found else line.strip()

        return bank_found, mode_found

    def _extract_clean_reference(self, raw: Optional[str]) -> Optional[str]:
        """
        Extrait la référence réelle depuis un champ :20: potentiellement verbeux.
        Ex: 'XXXXXXX DATE OF ISSUE' → 'XXXXXXX'
        """
        if not raw:
            return None
        cleaned = re.sub(
            r'\s*(?:DATE\s+OF\s+ISSUE|DOCUMENTARY\s+CREDIT\s+NUMBER|'
            r'TRANSFERRING\s+BANK[\'S\s]*REFERENCE)\s*',
            ' ', raw, flags=re.IGNORECASE
        ).strip()
        # Réf. LC : peut contenir tirets/slashs (ex: 'LC20260518-0451') — ne pas
        # tronquer au 1er '-'. On capture un token alphanumérique avec - et /.
        tokens = re.findall(r'[A-Z0-9][A-Z0-9\-/]{4,}', cleaned.upper())
        if tokens:
            return tokens[0].rstrip('-/')
        return cleaned if cleaned else raw

    @staticmethod
    def _issuing_bank_from_header(text: str) -> Optional[str]:
        """Extrait le BIC émetteur depuis l'en-tête SWIFT bloc 1 {1:F01<BIC>…}.
        Tolère les variantes OCR '{I:FOI'. Retourne un BIC8 (X de bourrage retirés).
        """
        m = re.search(
            r'\{[1I]:F[O0][1I]([A-Z]{4}[A-Z]{2}[A-Z0-9]{2})',
            text, re.IGNORECASE,
        )
        if m:
            bic = m.group(1).upper().rstrip("X")
            return bic or None
        return None

    def _clean_party_field(self, value: Optional[str]) -> Optional[str]:
        """
        Nettoie les champs :50: et :59: si le label n'a pas été supprimé.
        Retire aussi les lignes de contact (TEL/FAX/EMAIL) — bruit qui peut
        polluer l'extraction aval (ex: un n° de tél. pris pour un tag :49:).
        """
        if not value:
            return None
        cleaned = re.sub(
            r'^(?:FIRST\s+BENEFICIARY|SECOND\s+BENEFICIARY|BENEFICIARY|'
            r'APPLICANT|ORDERING\s+PARTY|INSTRUCTING\s+PARTY)\s*[:\-]?\s*',
            '', value, flags=re.IGNORECASE
        ).strip()
        # Couper à la 1re mention de contact (TEL/TÉL/FAX/EMAIL/E-MAIL/PHONE)
        cleaned = re.split(
            r'\s*(?:TEL|T[ÉE]L|FAX|E-?MAIL|PHONE|MOB(?:ILE)?)\b\s*[:.\-]?',
            cleaned, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        return cleaned if cleaned else value