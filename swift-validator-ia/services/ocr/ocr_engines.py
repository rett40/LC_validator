"""
ocr_engines.py  —  v3
Moteurs OCR (Tesseract + EasyOCR) avec méthode hybride principale.

CORRECTIFS v3 :
- correction numérique uniquement sur tokens non-tags (évite 43T→437, 44B→448)
- heuristique étendue pour tokens 100% ambigus sans vrai chiffre (OOOOOOOOOO→0000000000)
- _fix_split_words renforcé (SHANGHA I, BENEF ICIARY)
- correction devise : USD/EUR traités séparément du reste du token
- PORT 0F → PORT OF : correction O→lettre en contexte alphabétique
"""

import re
import easyocr
import numpy as np
import pytesseract
from pytesseract import Output


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Format exact d'un tag SWIFT : 2 chiffres + lettre optionnelle
_SWIFT_TAG_RE = re.compile(r'^\d{2}[A-Z]?$')

# Token à dominante numérique (contient chiffres + lettres ambiguës)
_NUMERIC_TOKEN_RE = re.compile(r'^[A-Z0-9OoIlSsBbZzGgTt]{2,}$')

# Devises reconnues
_CURRENCIES = frozenset(['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CNY'])

# Mots alphabétiques connus — ne jamais appliquer correction numérique dessus
_ALPHA_KEYWORDS = frozenset([
    'IRREVOCABLE', 'SIGHT', 'ALLOWED', 'CHINA', 'TUNISIA',
    'LATEST', 'VERSION', 'PARTIAL', 'SHIPMENTS', 'TRANSSHIPMENT',
    'LOADING', 'DISCHARGE', 'INDUSTRIAL', 'MACHINERY', 'MODEL',
    'SIGNED', 'COMMERCIAL', 'INVOICE', 'COPIES', 'PACKING',
    'LIST', 'CERTIFICATE', 'ORIGIN', 'DOCUMENTS', 'INDICATE',
    'ISSUED', 'ENGLISH', 'BANKING', 'CHARGES', 'OUTSIDE',
    'BENEFICIARY', 'WITHOUT', 'BRAND', 'NEW', 'IMPORT', 'EXPORT',
    'ROAD', 'PORT', 'BOARD', 'FULL', 'SET', 'CLEAN', 'BILL',
    'LADING', 'MUST', 'FOR', 'ALL', 'NOT', 'SWIFT', 'CREDIT',
    'DOCUMENTARY', 'ISSUE', 'MESSAGE', 'TEST', 'OF', 'AT',
    'SARL', 'LTD', 'CO', 'CIF', 'FOB', 'CFR', 'AND', 'IN',
    'RADES', 'TUNIS', 'SHANGHAI', 'DAYS', 'NUMBER', 'BANKING', 'CHARGES',"OR", "AND", "OF", "FOR", "IN", "AT"
])

# Mots W↔N connus
_WN_FIXES = {
    'ALLONED':   'ALLOWED',
    'ALLOMED':   'ALLOWED',
    'NITHOUT':   'WITHOUT',
    'MITOUT':    'WITHOUT',
    'MITHOUT':   'WITHOUT',
    'NILL':      'WILL',
    'FOLLOVING': 'FOLLOWING',
    'DRAN':      'DRAW',
}

# Reconstitution de mots coupés connus (SHANGHA I, BENEF ICIARY, etc.)
# Format : (regex, remplacement)
_SPLIT_WORD_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\bSHANGHA\s+I\b'),    'SHANGHAI'),
    (re.compile(r'\bBENEF\s+ICIARY\b'), 'BENEFICIARY'),
    (re.compile(r'\bBENEFIC\s+IARY\b'), 'BENEFICIARY'),
    (re.compile(r'\bINDUSTR\s+IE\b'),   'INDUSTRIE'),
    (re.compile(r'\bMACHINER\s+Y\b'),   'MACHINERY'),
    (re.compile(r'\bCERTIF\s+ICATE\b'), 'CERTIFICATE'),
    (re.compile(r'\bDOCUMEN\s+TS\b'),   'DOCUMENTS'),
    (re.compile(r'\bSHIPMEN\s+TS\b'),   'SHIPMENTS'),
]


class OCREngines:
    """
    Encapsule Tesseract et EasyOCR.

    Méthode principale : hybrid_ocr(gray_img, binary_img)
    """

    TESSERACT_BIAS  = 0.05
    LINE_Y_TOLERANCE = 10

    def __init__(self, languages: list[str] | None = None, gpu: bool = False):
        langs = languages or ['en']
        self.easyocr_reader = easyocr.Reader(langs, gpu=gpu)

        self._punctuation_map = str.maketrans({
            ';':  ':',
            '；': ':',
            '|':  ':',
            '¦':  ':',
        })

        # Lettre → chiffre (contexte numérique)
        self._num_map = str.maketrans('OoIlSsBbZzGg', '001155882266')

        # Chiffre → lettre (contexte alpha, ex: PORT 0F)
        self._alpha_map = str.maketrans('01568', 'OISGB')

        # ATTENTION : pas de quote (' ou ") dans la whitelist — pytesseract
        # parse ce config via shlex.split(posix=True) sous Linux : une quote
        # non fermée fait crasher l'OCR en prod ("No closing quotation"),
        # alors que Windows (posix=False) laisse passer. Parité oblige.
        self._tess_config = (
            "--oem 3 --psm 6 "
            "-c preserve_interword_spaces=1 "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789:.,/()+-{}"
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def hybrid_ocr(self,
                   gray_img: np.ndarray,
                   binary_img: np.ndarray) -> tuple[str, float]:
        boxes = self.easyocr_reader.readtext(gray_img, detail=1, paragraph=False)
        if not boxes:
            return "", 0.0

        processed = [
            self._process_box(box, txt_e, conf_e, binary_img)
            for box, txt_e, conf_e in boxes
        ]

        lines = self._group_into_lines(processed)

        final_text = "\n".join(
            " ".join(item[1] for item in line)
            for line in lines
        )
        all_confs = [item[2] for line in lines for item in line]
        overall_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        return final_text, round(overall_conf, 2)

    def tesseract_with_confidence(self, image: np.ndarray) -> tuple[str, float]:
        data = pytesseract.image_to_data(
            image, output_type=Output.DICT, config=self._tess_config
        )
        texts = [data['text'][i]
                 for i in range(len(data['text']))
                 if data['text'][i].strip()]
        confs = [float(data['conf'][i])
                 for i in range(len(data['conf']))
                 if data['conf'][i] != '-1']
        text = self._normalize(" ".join(texts))
        conf = (sum(confs) / len(confs) / 100) if confs else 0.0
        return text, conf

    def easyocr_with_confidence(self, image: np.ndarray) -> tuple[str, float]:
        results = self.easyocr_reader.readtext(image)
        texts = [r[1] for r in results]
        confs = [r[2] for r in results]
        text = self._normalize("".join(texts))
        conf = (sum(confs) / len(confs)) if confs else 0.0
        return text, conf

    def tesseract(self, image: np.ndarray) -> str:
        return pytesseract.image_to_string(image, config=self._tess_config)

    def easyocr(self, image: np.ndarray) -> str:
        results = self.easyocr_reader.readtext(image)
        return self._normalize("".join(r[1] for r in results))

    # ------------------------------------------------------------------
    # Traitement d'une boîte
    # ------------------------------------------------------------------

    
    def _process_box(
        self,
        box,
        txt_e: str,
        conf_e: float,
        binary_img: np.ndarray
    ) -> tuple:
        # Convertir les coordonnées EasyOCR en entiers
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]

        height, width = binary_img.shape[:2]

        x1 = max(0, min(width, int(np.floor(min(xs)))))
        y1 = max(0, min(height, int(np.floor(min(ys)))))
        x2 = max(0, min(width, int(np.ceil(max(xs)))))
        y2 = max(0, min(height, int(np.ceil(max(ys)))))

        # Boîte invalide : conserver le résultat EasyOCR
        if x2 <= x1 or y2 <= y1:
            return box, self._correct_text(
                self._normalize(txt_e)
            ), conf_e

        crop = binary_img[y1:y2, x1:x2]

        if crop.size == 0:
            return box, self._correct_text(
                self._normalize(txt_e)
            ), conf_e

        data = pytesseract.image_to_data(
            crop,
            output_type=Output.DICT,
            config=self._tess_config
        )

        words = [
            data['text'][i]
            for i in range(len(data['text']))
            if data['text'][i].strip()
        ]

        raw_confs = [
            float(c)
            for c in data['conf']
            if c != '-1'
        ]

        txt_t = self._normalize(" ".join(words))
        conf_t = (
            sum(raw_confs) / len(raw_confs) / 100
            if raw_confs else 0.0
        )

        txt_t_c = self._correct_text(txt_t)
        txt_e_c = self._correct_text(self._normalize(txt_e))

        if conf_t + self.TESSERACT_BIAS >= conf_e:
            return box, txt_t_c, conf_t

        return box, txt_e_c, conf_e
        # ------------------------------------------------------------------
    # Regroupement en lignes logiques
    # ------------------------------------------------------------------

    def _group_into_lines(self, items: list[tuple]) -> list[list[tuple]]:
        if not items:
            return []

        def cy(item):
            return sum(p[1] for p in item[0]) / len(item[0])

        def cx(item):
            return sum(p[0] for p in item[0]) / len(item[0])

        sorted_items = sorted(items, key=cy)
        lines: list[list[tuple]] = []
        current_line = [sorted_items[0]]
        current_y    = cy(sorted_items[0])

        for item in sorted_items[1:]:
            if abs(cy(item) - current_y) <= self.LINE_Y_TOLERANCE:
                current_line.append(item)
            else:
                current_line.sort(key=cx)
                lines.append(current_line)
                current_line = [item]
                current_y = cy(item)

        current_line.sort(key=cx)
        lines.append(current_line)
        return lines

    # ------------------------------------------------------------------
    # Corrections texte
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        return text.translate(self._punctuation_map)

    def _correct_text(self, text: str) -> str:
        """Applique toutes les corrections token par token."""
        # 1. Mots coupés (multi-tokens) — avant le split
        text = self._fix_split_words(text)
        tokens = text.split()
        corrected = []
        for token in tokens:
            token = self._fix_wn(token)
            token = self._fix_token(token)
            corrected.append(token)
        return " ".join(corrected)

    @staticmethod
    def _fix_split_words(text: str) -> str:
        """Reconstitue les mots connus coupés par un espace parasite."""
        for pattern, replacement in _SPLIT_WORD_FIXES:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _fix_wn(token: str) -> str:
        upper = token.upper()
        if upper in _WN_FIXES:
            fixed = _WN_FIXES[upper]
            return fixed if token.isupper() else fixed.capitalize()
        return token

    def _fix_token(self, token: str) -> str:
        """
        Correction contextuelle d'un token :

        RÈGLE 1 — Tag SWIFT (ex: '43T', '44B') → ne jamais corriger
        RÈGLE 2 — Devise (USD/EUR/…) → ne jamais corriger la partie alphabétique
        RÈGLE 3 — Mot alphabétique connu → ne pas corriger
        RÈGLE 4 — Token numérique (chiffres + lettres ambiguës) → lettre→chiffre
        RÈGLE 5 — Token 100% ambigus sans chiffre (OOOOOOOOOO) → lettre→chiffre
                  si ratio O/o > 60% et longueur ≥ 6
        RÈGLE 6 — Mot alphabétique avec chiffre parasite (PORT 0F) → chiffre→lettre
                  si le token est majoritairement alphabétique
        """
        upper = token.upper()

        # Règle 1 : tag SWIFT → intouchable
        if _SWIFT_TAG_RE.match(upper):
            return token

        # Règle 2 : devise → intouchable
        if upper in _CURRENCIES:
            return token

        # Règle 3 : mot connu → intouchable
        if upper in _ALPHA_KEYWORDS:
            return token

        has_digit  = any(c.isdigit() for c in token)
        has_alpha  = any(c.isalpha() for c in token)
        has_ambig  = any(c in 'OoIlSsBbZzGg' for c in token)

        # Règle 4 : contexte numérique (au moins un vrai chiffre + lettres ambiguës)
        if has_digit and has_alpha and has_ambig and _NUMERIC_TOKEN_RE.match(upper):
            return token.translate(self._num_map)

        # Règle 5 : token 100% lettres ambiguës ressemblant à un nombre
        if not has_digit and _NUMERIC_TOKEN_RE.match(upper):
            o_ratio = sum(1 for c in token if c in 'Oo') / len(token)
            if o_ratio > 0.6 and len(token) >= 6:
                return token.translate(self._num_map)

        # Règle 6 : mot alphabétique avec chiffre parasite (ex: 0F → OF)
        if has_digit and has_alpha:
            alpha_ratio = sum(1 for c in token if c.isalpha()) / len(token)
            if alpha_ratio > 0.6:
                return token.translate(self._alpha_map)

        return token