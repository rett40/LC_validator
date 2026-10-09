"""
mt790_parser.py  —  v5
Parseur complet pour MT790 — Advice of Charges, Interest and Other Adjustments.

Champs extraits :
  :20:  lc_number (référence transaction)
  :21:  related_lc (référence liée)
  :25:  account (compte débité/crédité)
  :32A: value_date / currency / amount
  :52a: issuing_bank (institution donneuse d'ordre)
  :71B: charge_details (détail des frais)
  :72:  sender_info
  :77A: message (informations supplémentaires)
"""

from ..swift_base_parser import SwiftBaseParser
from ..entities import SwiftEntities


class MT790Parser(SwiftBaseParser):

    def parse(self, text: str) -> SwiftEntities:
        e = SwiftEntities(swift_type="MT790")

        # ── Références ────────────────────────────────────────────────
        e.lc_number  = self.extract_tag(text, ":20:")
        e.related_lc = self.extract_tag(text, ":21:")

        # ── Compte ────────────────────────────────────────────────────
        e.account = self.extract_tag(text, ":25:")

        # ── :32A: Date valeur + devise + montant ─────────────────────
        vdate, ccy, amt = self.extract_value_date_amount(text, ":32A:")
        e.value_date = vdate
        e.currency   = ccy
        e.amount     = amt

        # ── Banque donneuse d'ordre ───────────────────────────────────
        e.issuing_bank = (
            self.extract_tag_multiline(text, ":52A:")
            or self.extract_tag_multiline(text, ":52D:")
        )

        # ── Détail des frais :71B: ────────────────────────────────────
        e.charge_details = self.extract_tag_multiline(text, ":71B:")
        e.charges        = e.charge_details   # alias

        # ── Informations émetteur et message ─────────────────────────
        e.sender_info = self.extract_tag(text, ":72:")
        e.message     = self.extract_tag_multiline(text, ":77A:")

        return e