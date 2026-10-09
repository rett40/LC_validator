"""
mt799_parser.py  —  v5.2
Parseur pour MT799 — Free Format Message.

v5.2 :
  F6 — si :79: absent (texte libre sans tags SWIFT), stocker le texte brut complet
       dans entities.free_text pour ne pas perdre le contenu du document.
"""

import re
from ..swift_base_parser import SwiftBaseParser
from ..entities import SwiftEntities

# Pattern pour nettoyer les entêtes parasites du texte brut OCR
_HEADER_NOISE = re.compile(
    r'^\s*(?:SWIFT\s+MT\d{3}[^\n]*\n|'   # header SWIFT MT799
    r'\{[^}]+\}\s*\n?|'                    # blocs {…}
    r'IIB[^\n]*\n?)',                       # entêtes banque
    re.IGNORECASE | re.MULTILINE
)


class MT799Parser(SwiftBaseParser):

    def parse(self, text: str) -> SwiftEntities:
        e = SwiftEntities(swift_type="MT799")

        # ── Références ────────────────────────────────────────────────
        e.lc_number  = self.extract_tag(text, ":20:")
        e.related_lc = self.extract_tag(text, ":21:")

        # ── Texte libre :79: ──────────────────────────────────────────
        e.free_text = self.extract_tag_multiline(text, ":79:")

        # F6 : si :79: absent, stocker le texte brut complet
        # (cas MT799 free-text sans aucun tag SWIFT formel)
        if not e.free_text:
            cleaned = _HEADER_NOISE.sub('', text).strip()
            if cleaned:
                e.free_text = cleaned
                e.raw_fields["free_text_source"] = "raw_ocr"

        e.message = e.free_text

        # Statistiques sur le texte libre
        if e.free_text:
            lines = e.free_text.splitlines()
            e.raw_fields["79_line_count"] = str(len(lines))
            long_lines = [i + 1 for i, l in enumerate(lines) if len(l) > 50]
            if long_lines:
                e.raw_fields["79_long_lines"] = str(long_lines)
            if len(lines) > 35:
                e.warnings.append(
                    f":79: dépasse 35 lignes ({len(lines)} lignes) — limite SWIFT"
                )

        # ── Informations émetteur ─────────────────────────────────────
        e.sender_info = self.extract_tag(text, ":72:")

        return e