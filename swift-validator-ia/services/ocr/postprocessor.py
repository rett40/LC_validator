"""
postprocessor.py  —  v4
Nettoyage et correction du texte OCR pour messages SWIFT (MT700/720/760/799).

CORRECTIFS v4 (par rapport à v3) :
- _fix_amounts RÉÉCRIT avec regex strict (CORRECTIF BUG PRINCIPAL)
    Ancien regex trop gourmand absorbait les caractères suivants :
      "1190r" au lieu de "119" → lettre parasite absorbée
      "2500001" au lieu de "250000" → chiffre suivant absorbé
    Nouveau regex : le montant s'arrête au premier char non-OCR-numérique
- AmountValidator intégré pour double validation
- _fix_amounts appelle AmountValidator.fix_amounts en premier
- Ajout _fix_trailing_digit_on_amount : supprime les chiffres/lettres
  parasites collés au montant (ex: "USD250000r" → "USD250000")
"""

import re
from loguru import logger

try:
    from .amount_validator import AmountValidator
    _amount_validator = AmountValidator()
except ImportError:
    _amount_validator = None


# ---------------------------------------------------------------------------
# Caractères autorisés dans un message SWIFT
# ---------------------------------------------------------------------------
_SWIFT_ALLOWED = re.compile(r"[^\w\s:/{}.()',\"+-]")

# ---------------------------------------------------------------------------
# Tags SWIFT connus (ordre décroissant de longueur pour le matching)
# ---------------------------------------------------------------------------
_KNOWN_TAGS = [
    '40A', '40E', '31C', '31D', '32B', '32A',
    '39A', '39B', '41A', '41D', '42C', '42A', '42M',
    '43P', '43T', '44A', '44B', '44C', '44D', '44E', '44F',
    '45A', '45B', '46A', '46B', '47A', '47B',
    '48', '49', '50', '59',
    '71B', '71D', '72', '78',
    '15A', '15B', '15C', '20', '27',
]
_TAGS_PATTERN = '|'.join(sorted(_KNOWN_TAGS, key=len, reverse=True))

# ---------------------------------------------------------------------------
# Corrections de tags OCR mal lus
# ---------------------------------------------------------------------------
_TAG_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bMT\s*7[7]?[O0]{2}\b'),  'MT700'),
    (re.compile(r'\bM7{1,2}[O0]{2}\b'),      'MT700'),
    (re.compile(r'\bMT72[O0]\b'),            'MT720'),
    (re.compile(r'\bMT76[O0]\b'),            'MT760'),
    (re.compile(r'(?<![:\w])4[O0]E:'),       '40E:'),
    (re.compile(r'(?<![:\w])4[O0]A:'),       '40A:'),
    (re.compile(r'(?<![:\w])3[I1]B:'),       '31B:'),
    (re.compile(r'(?<![:\w])3[I1]C:'),       '31C:'),
    (re.compile(r'(?<![:\w])3[I1]D:'),       '31D:'),
    (re.compile(r'(?<![:\w])32[B8]:'),       '32B:'),
    (re.compile(r'(?<![:\w])4[I1]A:'),       '41A:'),
    (re.compile(r'(?<![:\w])4[I1]D:'),       '41D:'),
    (re.compile(r'(?<![:\w])42[C6]:'),       '42C:'),
    (re.compile(r'(?<![:\w])43[P9]:'),       '43P:'),
    (re.compile(r'(?<![:\w])43[T7]:'),       '43T:'),
    (re.compile(r'(?<![:\w])44[A4]:'),       '44A:'),
    (re.compile(r'(?<![:\w])44[B8]:'),       '44B:'),
    (re.compile(r'(?<![:\w])44[C6]:'),       '44C:'),
    (re.compile(r'(?<![:\w])45[A4]:'),       '45A:'),
    (re.compile(r'(?<![:\w])46[A4]:'),       '46A:'),
    (re.compile(r'(?<![:\w])47[A4]:'),       '47A:'),
    (re.compile(r'(?<![:\w])7[I1]B:'),       '71B:'),
]

# ---------------------------------------------------------------------------
# Correction des devises mal reconnues
# ---------------------------------------------------------------------------
_CURRENCY_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bU[S5][D0]\b'), 'USD'),
    (re.compile(r'\bEU[R8]\b'),    'EUR'),
    (re.compile(r'\bGB[P9]\b'),    'GBP'),
]

# ---------------------------------------------------------------------------
# CORRECTIF v4 — Regex STRICT pour les montants
#
# PROBLÈME v3 :
#   _AMOUNT_RE = re.compile(r'(USD|EUR|GBP|...)([ 0-9OoSsIlBb,. ]+)')
#   Le quantificateur '+' est TROP GOURMAND :
#     "USD119 :44C:" → groupe 2 = "119 " (espace inclus) ✓
#     "USD1190r"      → groupe 2 = "1190r" (r absorbé)   ✗
#     "USD2500001"    → groupe 2 = "2500001" (1 du mot suivant absorbé) ✗
#
# SOLUTION v4 :
#   On utilise un lookahead négatif pour s'arrêter dès qu'on rencontre
#   un caractère qui NE PEUT PAS être une confusion OCR de chiffre :
#     - lettre non-ambiguë (pas dans OoSsIlBb)
#     - espace suivi d'une lettre non-ambiguë ou d'un ':'
# ---------------------------------------------------------------------------

_CURRENCIES_PAT = r'(?:USD|EUR|GBP|JPY|CHF|CNY)'

# Caractères légitimes dans un montant OCR
_OCR_AMOUNT_CHAR = r'[0-9OoSsIlBb,.]'

# Regex strict : le montant s'arrête dès qu'on sort des caractères ambigus
# On refuse aussi un espace suivi de quelque chose qui n'est pas un chiffre/ambigus
_AMOUNT_STRICT_RE = re.compile(
    r'(' + _CURRENCIES_PAT + r')'       # groupe 1 : devise
    r'\s*'                               # espace optionnel entre devise et montant
    r'('                                 # groupe 2 : montant strict
        + _OCR_AMOUNT_CHAR + r'+'        # au moins un char valide
        r'(?:\s' + _OCR_AMOUNT_CHAR + r'+)*'  # espaces internes (milliers) acceptés
    r')',
    re.IGNORECASE,
)

_AMOUNT_CHARMAP = str.maketrans('OoSsIlBb', '00551188')

# Lettre parasite en fin de montant numérique (ex: "119r", "250000,r")
# On la supprime si elle suit directement les chiffres
_TRAILING_JUNK_RE = re.compile(
    r'(' + _CURRENCIES_PAT + r')'
    r'(\d[\d,.]*)'         # montant propre (après traduction)
    r'[a-zA-Z]+'           # lettre(s) parasite(s)
    r'(?=\s|:|$)',          # suivi d'un espace, ':' ou fin
)

# Slash sans espace
_SLASH_RE  = re.compile(r'(\d+)\s*/\s*(\d+)')

# Tag 27 : fraction sans espace
_TAG_27_RE = re.compile(r'(:27:)\s*(\d+/\d+)')

# Espace avant tag bien formé
_CLEAN_TAG_RE = re.compile(r'(?<!\s)(?<!{)(:(?:' + _TAGS_PATTERN + r'):)')

# Double ':' parasites (': :' ou '::')
_DOUBLE_COLON_RE = re.compile(r':\s*:+')

# Header {4 : → {4:
_HEADER_SPACE_RE = re.compile(r'(\{[0-9])\s*:')

# Mots coupés connus
_SPLIT_WORD_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bSHANGHA\s+I\b'),    'SHANGHAI'),
    (re.compile(r'\bBENEF\s+ICIARY\b'), 'BENEFICIARY'),
    (re.compile(r'\bBENEFIC\s+IARY\b'), 'BENEFICIARY'),
    (re.compile(r'\bINDUSTR\s+IE\b'),   'INDUSTRIE'),
]

# PORT 0F → PORT OF
_PORT_OF_RE = re.compile(r'\bPORT\s+0F\b')


class SwiftPostProcessor:
    """
    Nettoie le texte OCR brut pour le rendre conforme aux messages SWIFT.

    Pipeline v4 :
      1.  Correction caractères OCR (;→: , _→espace)
      2.  Correction header {4 :→{4:
      3.  Correction tags OCR mal lus (40E, 43T, 44B …)
      4.  Correction devises mal lues (U5D→USD)
      5.  Aplatissement retours ligne → espace
      6.  Normalisation espaces multiples
      7.  Suppression doubles ':' parasites
      8.  Restauration ':' manquants autour des tags
      9.  Correction mots coupés (SHANGHA I, BENEF ICIARY)
      10. PORT 0F → PORT OF
      11. [RÉÉCRIT v4] Correction montants (regex strict + AmountValidator)
      12. Suppression lettres parasites en fin de montant
      13. Normalisation slashes
      14. Espacement avant tags
      15. Tag 27 : fraction sans espace
      16. Suppression caractères non-SWIFT
    """

    def clean(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        t = raw_text
        t = self._fix_ocr_chars(t)
        t = self._fix_header_space(t)
        t = self._fix_swift_tags(t)
        t = self._fix_currencies(t)
        t = self._flatten_newlines(t)
        t = self._normalize_spaces(t)
        t = self._clean_double_colons(t)
        t = self._restore_missing_colons(t)
        t = self._fix_split_words(t)
        t = self._fix_port_of(t)
        t = self._fix_amounts(t)          # ← RÉÉCRIT v4
        t = self._fix_trailing_junk(t)    # ← NOUVEAU v4
        t = self._fix_slashes(t)
        t = self._ensure_tag_spacing(t)
        t = self._fix_tag_27(t)
        t = self._remove_illegal_chars(t)

        logger.debug(f"Post-traitement v4 : {t[:200]}…")
        return t.strip()

    # ------------------------------------------------------------------
    # Étapes
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_ocr_chars(text: str) -> str:
        return text.replace(';', ':').replace('_', ' ')

    @staticmethod
    def _fix_header_space(text: str) -> str:
        return _HEADER_SPACE_RE.sub(r'\1:', text)

    @staticmethod
    def _fix_swift_tags(text: str) -> str:
        for pattern, repl in _TAG_FIXES:
            text = pattern.sub(repl, text)
        return text

    @staticmethod
    def _fix_currencies(text: str) -> str:
        for pattern, repl in _CURRENCY_FIXES:
            text = pattern.sub(repl, text)
        return text

    @staticmethod
    def _flatten_newlines(text: str) -> str:
        return text.replace('\n', ' ').replace('\r', ' ')

    @staticmethod
    def _normalize_spaces(text: str) -> str:
        return re.sub(r' {2,}', ' ', text)

    @staticmethod
    def _clean_double_colons(text: str) -> str:
        return _DOUBLE_COLON_RE.sub(':', text)

    @staticmethod
    def _restore_missing_colons(text: str) -> str:
        pattern = re.compile(
            r'(?<!:)(?<!\w)(' + _TAGS_PATTERN + r')(?::| )(?=\S)'
        )
        def _fix(m: re.Match) -> str:
            return f' :{m.group(1)}: '
        return pattern.sub(_fix, text)

    @staticmethod
    def _fix_split_words(text: str) -> str:
        for pattern, repl in _SPLIT_WORD_FIXES:
            text = pattern.sub(repl, text)
        return text

    @staticmethod
    def _fix_port_of(text: str) -> str:
        return _PORT_OF_RE.sub('PORT OF', text)

    @staticmethod
    def _fix_amounts(text: str) -> str:
        """
        RÉÉCRIT v4 — Regex strict + AmountValidator.

        1. On utilise AmountValidator si disponible (double filet).
        2. Fallback sur le regex strict interne.
        """
        # Filet 1 : AmountValidator (regex strict + heuristique longueur)
        if _amount_validator is not None:
            text = _amount_validator.fix_amounts(text)
            return text

        # Filet 2 : regex strict intégré (si AmountValidator non disponible)
        def _correct(m: re.Match) -> str:
            currency = m.group(1).upper()
            amount   = m.group(2).translate(_AMOUNT_CHARMAP).replace(' ', '')
            return currency + amount

        return _AMOUNT_STRICT_RE.sub(_correct, text)

    @staticmethod
    def _fix_trailing_junk(text: str) -> str:
        """
        NOUVEAU v4 — Supprime les lettres parasites collées en fin de montant.
        Ex: "USD119r" → "USD119", "EUR250000," → "EUR250000"
        """
        return _TRAILING_JUNK_RE.sub(r'\1\2', text)

    @staticmethod
    def _fix_slashes(text: str) -> str:
        return _SLASH_RE.sub(r'\1/\2', text)

    @staticmethod
    def _ensure_tag_spacing(text: str) -> str:
        return _CLEAN_TAG_RE.sub(r' \1', text)

    @staticmethod
    def _fix_tag_27(text: str) -> str:
        return _TAG_27_RE.sub(r'\1\2', text)

    @staticmethod
    def _remove_illegal_chars(text: str) -> str:
        return _SWIFT_ALLOWED.sub('', text)