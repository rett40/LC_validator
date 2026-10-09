"""
swift_base_parser.py  —  v6.0
Classe de base pour tous les parseurs MT***.

v5.1 :
  - _parse_currency_amount : séparateur décimal européen (USD90.000, → 90000)
  - strip_label : généralisé avec patterns verbeux prédéfinis
  - _clean_labeled_value : supprime les labels verbeux des champs communs
  - _extract_raw : stoppers contextuels (noms de labels) pour éviter débordement
  - extract_date_and_place : supporte aussi DD.MM.YYYY (format template)
  - _normalize_ocr_text : améliorations mineures
v5.3 :
  - normalize_enum : 3-pass strategy (exact, long-first, short-first)
  - _VERBOSE_STOPPERS : [ \\t]* au lieu de \\s* pour BENEFICIARY/APPLICANT
  - \\bCHARGES? word boundary — évite DISCHARGE
v5.4 :
  - _TAG_OWN_STOPPER : chaque tag exclut son propre stopper
  - _VERBOSE_LABELS['27'] : strip 'SEQUENCE OF TOTAL'
  - port extraction : B1/B2 déblocage 43P/43T/44A/44B/44E/44F
v6.0 :
  - B9  : _VERBOSE_LABELS['30'] — MT760 :30: DATE OF ISSUE label strippé
  - B10 : AIRPORT OF DEPARTURE/DESTINATION dans _VERBOSE_STOPPERS
  - is_placeholder() : détection templates XXXXXXX / 0X.XX.2023 / AMOUNT
  - _ISO4217_COMMON + _is_valid_currency() : rejet devises OCR (DAM, AMO…)
  - _parse_currency_amount : USDAMOUNT → (None, None)
  - _clean_labeled_value B11/B12 : strip BENEFICIARY résiduel :50:, ADDRESS :59:
  - _find_date_in_string B13 : rejette dates template (XX.XX.2023)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Labels verbeux connus — préfixes à supprimer des valeurs de champs
# Format : 'SEQUENCE OF TOTAL 1/1' → le label avant la vraie valeur
# ---------------------------------------------------------------------------
_VERBOSE_LABELS: dict[str, str] = {
    # v5.4 : :27: — 'SEQUENCE OF TOTAL 1/1' → '1/1'
    "27":  r'(?:SEQUENCE\s+OF\s+TOTAL|SEQ(?:UENCE)?\s+OF\s+TOTAL)\s*',
    # tag_code → pattern regex du label à supprimer
    "27":  r'(?:SEQUENCE\s+OF\s+TOTAL|SEQ(?:UENCE)?)\s*',
    "20":  r'(?:DOCUMENTARY\s+CREDIT\s+NUMBER|REFERENCE\s+NUMBER|TRANSATTION\s+REFERENCE|'
           r'TRANSACTION\s+REFERENCE|TRANSFERRING\s+BANK[\'S\s]+REFERENCE|'
           r'CREDIT\s+NUMBER|LC\s+NUMBER|REFERENCE)\s*',
    "21":  r'(?:DOCUMENTARY\s+CREDIT\s+NUMBER|CREDIT\s+NUMBER|RELATED\s+REFERENCE)\s*',
    "40A": r'(?:FORM\s+OF\s+DOCUMENTARY\s+CREDIT|FORM\s+OF\s+CREDIT)\s*',
    "40B": r'(?:FORM\s+OF\s+DOCUMENTARY\s+CREDIT|FORM\s+OF\s+CREDIT)\s*',
    "40E": r'(?:APPLICABLE\s+RULES?)\s*',
    # B9: MT760 :30: — 'DATE OF ISSUE 12-APR-2021' → '12-APR-2021'
    "30":  r'(?:DATE\s+OF\s+ISSUE|ISSUE\s+DATE)\s*',
    "31C": r'(?:DATE\s+OF\s+ISSUE|ISSUE\s+DATE)\s*',
    "31D": r'(?:DATE\s+(?:AND\s+)?PLACE\s+OF\s+EXPIRY|EXPIRY\s+DATE(?:\s+AND\s+PLACE)?)\s*',
    "31E": r'(?:DATE\s+(?:AND\s+)?PLACE\s+OF\s+EXPIRY|EXPIRY\s+DATE(?:\s+AND\s+PLACE)?)\s*',
    "50":  r'(?:FIRST\s+BENEFICIARY|APPLICANT|ORDERING\s+PARTY|INSTRUCTING\s+PARTY)\s*',
    "59":  r'(?:SECOND\s+BENEFICIARY|BENEFICIARY|ADDRESS)\s*',
    "32B": r'(?:CURRENCY\s+CODE[,\s]+AMOUNT|GUARANTEE\s+AMOUNT|AMOUNT)\s*',
    "45A": r'(?:DESCRIPTION\s+OF\s+GOODS\s+(?:AND(?:/OR)?\s+SERVICES?)?|'
           r'GOODS\s+DESCRIPTION)\s*',
    "46A": r'(?:DOCUMENTS?\s+REQUIRED)\s*',
    "47A": r'(?:ADDITIONAL\s+CONDITIONS?)\s*',
    "48":  r'(?:PERIOD\s+FOR\s+PRESENTATION)\s*',
    "49":  r'(?:CONFIRMATION\s+INSTRUCTIONS?)\s*',
    "44A": r'(?:PLACE\s+OF\s+TAKING\s+IN\s+CHARGE|PORT\s+OF\s+LOADING(?:/AIRPORT\s+OF\s+DEPARTURE)?)\s*:?\s*',
    "44B": r'(?:PLACE\s+OF\s+FINAL\s+DESTINATION|FOR\s+TRANSPORTATION\s+TO|PLACE\s+OF\s+DELIVERY|'
           r'PORT\s+OF\s+(?:FINAL\s+)?DISCHARGE(?:/AIRPORT\s+OF\s+DESTINATION)?)\s*:?\s*',
    "44E": r'(?:PORT\s+OF\s+LOADING(?:/AIRPORT\s+OF\s+DEPARTURE)?)\s*:?\s*',
    "44F": r'(?:PORT\s+OF\s+DISCHARGE(?:/AIRPORT\s+OF\s+DESTINATION)?)\s*:?\s*',
    "44C": r'(?:LATEST\s+DATE\s+OF\s+SHIPMENT)\s*',
    "43P": r'(?:PARTIAL\s+SHIPMENTS?)\s*',
    "43T": r'(?:TRANS(?:SHIP)?MENT)\s*',
    "41D": r'(?:AVAILABLE\s+WITH\s+[\.]{0,3}\s*BY\s*[\.]{0,3})\s*',
    "41A": r'(?:AVAILABLE\s+WITH\s+[\.]{0,3}\s*BY\s*[\.]{0,3})\s*',
    "71B": r'(?:CHARGES?)\s*',
    "71D": r'(?:CHARGES?)\s*',
    "72":  r'(?:SENDER\s+TO\s+RECEIVER\s+INFORMATION)\s*',
    "78":  r'(?:INSTRUCTIONS?\s+TO\s+THE\s+(?:PAYING|ACCEPTING|NEGOTIATING)\s*(?:BANK)?)\s*',
}

# Stoppers contextuels : mots qui indiquent le début d'un autre champ
# même sans tag SWIFT formel (format verbeux)
_VERBOSE_STOPPERS = [
    r'SEQUENCE\s+OF\s+TOTAL\b',
    r'FORM\s+OF\s+DOCUMENTARY\s+CREDIT\b',
    r'DOCUMENTARY\s+CREDIT\s+NUMBER\b',
    r'DATE\s+OF\s+ISSUE\b',
    r'APPLICABLE\s+RULES\b',
    r'DATE\s+(?:AND\s+)?PLACE\s+OF\s+EXPIRY\b',
    r'CURRENCY\s+CODE[,\s]+AMOUNT\b',
    r'AVAILABLE\s+WITH\b',
    r'PARTIAL\s+SHIPMENTS\b',
    r'TRANS(?:SHIP)?MENT\b',
    r'PORT\s+OF\s+LOADING\b',                               # B10: séparé de DISCHARGE
    r'PORT\s+OF\s+DISCHARGE\b',                              # B10: séparé de LOADING
    r'AIRPORT\s+OF\s+DEPARTURE\b',                           # B10: stopper port verbeux MT760
    r'AIRPORT\s+OF\s+DESTINATION\b',                         # B10: stopper port verbeux MT760
    r'PLACE\s+OF\s+(?:FINAL\s+DESTINATION|TAKING)\b',
    r'LATEST\s+DATE\s+OF\s+SHIPMENT\b',
    r'DESCRIPTION\s+OF\s+GOODS\b',
    r'DOCUMENTS?\s+REQUIRED\b',
    r'ADDITIONAL\s+CONDITIONS\b',
    r'PERIOD\s+FOR\s+PRESENTATION\b',
    r'CONFIRMATION\s+INSTRUCTIONS\b',
    r'INSTRUCTIONS?\s+TO\s+THE\b',
    r'SENDER\s+TO\s+RECEIVER\b',
    r'(?:FIRST|SECOND)\s+BENEFICIARY\b',
    # v5.3 : [ \t]* au lieu de \s* — évite de matcher 'BENEFICIARY\n\n:48:'
    # v5.4 : \b devant CHARGES — évite de matcher 'CHARGE:' dans 'DISCHARGE:'
    r'\bCHARGES?[ \t]*:',
    r'BENEFICIARY[ \t]*:',
    r'APPLICANT[ \t]*:',
]


# ---------------------------------------------------------------------------
# Mapping tag → stopper à EXCLURE du NEXT lookahead.
# Problème : un stopper comme 'PARTIAL SHIPMENTS' est utile pour détecter
# les limites de champs verbeux, mais bloque l'extraction de :43P: lui-même
# car 'PARTIAL SHIPMENTS' est aussi le début de la VALEUR.
# Solution : exclure le stopper correspondant au tag en cours d'extraction.
# ---------------------------------------------------------------------------
_TAG_OWN_STOPPER: dict[str, str] = {
    '27':  r'SEQUENCE\s+OF\s+TOTAL\b',
    '30':  r'DATE\s+OF\s+ISSUE\b',             # B9: MT760 :30: — stopper 'DATE OF ISSUE' bloquait la valeur
    '40A': r'FORM\s+OF\s+DOCUMENTARY\s+CREDIT\b',
    '40B': r'FORM\s+OF\s+DOCUMENTARY\s+CREDIT\b',
    '40E': r'APPLICABLE\s+RULES\b',
    '31C': r'DATE\s+OF\s+ISSUE\b',
    '31D': r'DATE\s+(?:AND\s+)?PLACE\s+OF\s+EXPIRY\b',
    '31E': r'DATE\s+(?:AND\s+)?PLACE\s+OF\s+EXPIRY\b',
    '32B': r'CURRENCY\s+CODE[,\s]+AMOUNT\b',
    '41A': r'AVAILABLE\s+WITH\b',
    '41D': r'AVAILABLE\s+WITH\b',
    '43P': r'PARTIAL\s+SHIPMENTS\b',
    '43T': r'TRANS(?:SHIP)?MENT\b',
    '44A': r'PORT\s+OF\s+LOADING\b',            # B10: exclut LOADING, DISCHARGE reste actif
    '44B': r'PORT\s+OF\s+DISCHARGE\b',           # B10: exclut DISCHARGE, LOADING reste actif
    '44C': r'LATEST\s+DATE\s+OF\s+SHIPMENT\b',
    '44E': r'PORT\s+OF\s+LOADING\b',             # B10: exclut LOADING → AIRPORT OF DEPARTURE stoppe
    '44F': r'AIRPORT\s+OF\s+DESTINATION\b',       # B10: exclut AIRPORT OF DESTINATION (propre valeur)
    '45A': r'DESCRIPTION\s+OF\s+GOODS\b',
    '46A': r'DOCUMENTS?\s+REQUIRED\b',
    '47A': r'ADDITIONAL\s+CONDITIONS\b',
    '48':  r'PERIOD\s+FOR\s+PRESENTATION\b',
    '49':  r'CONFIRMATION\s+INSTRUCTIONS\b',
    '72':  r'SENDER\s+TO\s+RECEIVER\b',
}


class SwiftBaseParser:

    def extract_tag(self, text: str, tag: str) -> Optional[str]:
        """
        Extrait la valeur d'un tag SWIFT (première ligne significative).
        Applique automatiquement le nettoyage de label verbeux si applicable.
        """
        value = self._extract_raw(text, tag)
        if value is None:
            return None
        lines = [l.strip() for l in value.splitlines() if l.strip()]
        raw = lines[0] if lines else None
        return self._clean_labeled_value(raw, self._tag_code(tag))

    def extract_tag_multiline(self, text: str, tag: str) -> Optional[str]:
        """
        Extrait la valeur complète d'un tag (multilignes).
        Nettoie le label verbeux en début de valeur si applicable.
        """
        raw = self._extract_raw(text, tag)
        if not raw:
            return None
        return self._clean_labeled_value(raw, self._tag_code(tag))

    def extract_tag_lines(self, text: str, tag: str) -> list[str]:
        """Retourne la valeur d'un tag sous forme de liste de lignes non vides."""
        raw = self._extract_raw(text, tag)
        if not raw:
            return []
        cleaned = self._clean_labeled_value(raw, self._tag_code(tag))
        if not cleaned:
            return []
        return [l.strip() for l in cleaned.splitlines() if l.strip()]

    # ------------------------------------------------------------------
    # Parsers typés
    # ------------------------------------------------------------------

    def extract_amount(self, text: str, tag: str = ":32B:") -> Tuple[Optional[str], Optional[str]]:
        """Extrait (devise ISO, montant string) depuis un champ montant."""
        raw = self.extract_tag(text, tag)
        if not raw:
            return None, None
        # Chercher devise+montant même si label devant
        m = re.search(r'([A-Z0-9]{3})\s*([0-9OoIl][0-9OoIl\.,\s\']*)', raw, re.IGNORECASE)
        if m:
            return self._parse_currency_amount(m.group(1) + m.group(2))
        return None, None

    def extract_date(self, text: str, tag: str) -> Optional[str]:
        """Extrait une date ISO depuis un champ YYMMDD (ou autres formats)."""
        raw = self.extract_tag(text, tag)
        if not raw:
            return None
        # Chercher une date dans la valeur (peut contenir un label)
        return self._find_date_in_string(raw)

    def extract_date_and_place(self, text: str, tag: str = ":31D:") -> Tuple[Optional[str], Optional[str]]:
        """
        Pour :31D: / :31E: : retourne (date_iso, lieu).
        Supporte : YYMMDD + place, YYMMDD seul, date verbeux avec label.
        """
        raw = self.extract_tag(text, tag)
        if not raw:
            return None, None

        # Format standard SWIFT : YYMMDDPLACE
        iso = self._yymmdd_to_iso(raw[:6])
        if iso:
            place = raw[6:].strip() if len(raw) > 6 else None
            # Nettoyer le lieu : supprimer les labels résiduels
            if place:
                place = self._clean_expiry_place(place)
            return iso, place or None

        # Format verbeux : la date est quelque part dans la chaîne
        iso = self._find_date_in_string(raw)
        place = self._extract_place_from_expiry(raw)
        return iso, place

    def extract_sequence(self, text: str, tag: str = ":27:") -> Optional[str]:
        """Extrait la séquence (ex: '1/1')."""
        return self.extract_tag(text, tag)

    def extract_tolerance(self, text: str, tag: str = ":39A:") -> Tuple[Optional[str], Optional[str]]:
        """Extrait (plus%, minus%) depuis :39A:."""
        raw = self.extract_tag(text, tag)
        if not raw:
            return None, None
        m = re.search(r"(\d+)/(\d+)", raw.strip())
        return (m.group(1), m.group(2)) if m else (None, None)

    def extract_presentation_period(self, text: str, tag: str = ":48:") -> Tuple[Optional[str], Optional[str]]:
        """Extrait (jours, qualificatif) depuis :48:."""
        raw = self.extract_tag(text, tag)
        if not raw:
            return None, None
        # Ne pas confondre "WITHIN L/C VALIDITY" avec un nombre de jours
        m = re.match(r"(\d+)\s*(.*)", raw.strip())
        if m:
            return m.group(1), m.group(2).strip() or None
        # Format verbeux : "DOCUMENTS MUST BE PRESENTED WITHIN 15 DAYS …"
        # → extraire le nombre de jours même s'il n'est pas en tête.
        m2 = re.search(r'(\d{1,3})\s*DAYS?\b', raw, re.IGNORECASE)
        if m2:
            return m2.group(1), raw.strip()
        # Valeur textuelle (WITHIN L/C VALIDITY etc.)
        return None, raw.strip()

    def extract_value_date_amount(self, text: str, tag: str = ":32A:") -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Pour :32A: (MT790) : retourne (date_iso, devise, montant)."""
        raw = self.extract_tag(text, tag)
        if not raw:
            return None, None, None
        m = re.match(r"(\d{6})([A-Z]{3})(.+)", raw.strip())
        if not m:
            return None, None, None
        iso      = self._yymmdd_to_iso(m.group(1))
        ccy, amt = self._parse_currency_amount(m.group(2) + m.group(3))
        return iso, ccy, amt

    def normalize_enum(self, value: Optional[str], allowed: list[str]) -> Optional[str]:
        """
        Normalise une valeur vers la valeur canonique autorisée.

        Stratégie en 3 passes pour éviter les faux positifs :
          1. Exact match (priorité absolue)
          2. La valeur canonique EST CONTENUE dans le texte — long first
             ex: 'NOT ALLOWED' avant 'ALLOWED' dans 'PARTIAL SHIPMENTS NOT ALLOWED'
          3. Le texte EST CONTENU dans une valeur canonique — short first
             ex: 'UCP' → 'UCP 600' (abréviation acceptable)
             (ne pas tester long first ici pour éviter 'UCP' → 'EUCP LATEST VERSION')
        """
        if not value:
            return None
        v = value.upper().strip()

        # 1. Exact match
        for a in allowed:
            if v == a.upper():
                return a

        # 2. a.upper() IN v (canonique dans le texte) — valeurs LONGUES d'abord
        for a in sorted(allowed, key=len, reverse=True):
            if a.upper() in v:
                return a

        # 3. v IN a.upper() (texte dans canonique — abbréviations) — valeurs COURTES d'abord
        for a in sorted(allowed, key=len):
            if v in a.upper():
                return a

        return v

    def extract_enum_strict(
        self, text: str, tag: str, options: list[str]
    ) -> Optional[str]:
        """Extrait un champ à valeurs CONTRAINTES (ex: :49: → CONFIRM/MAY ADD/WITHOUT).

        Parcourt TOUTES les occurrences du tag et retourne la première valeur qui
        correspond à une option canonique (frontière de mot). Si une occurrence
        contient une autre réponse (bruit OCR, n° de téléphone…), on la SKIPPE et
        on continue. Si aucune valeur valide n'est trouvée → None (champ manquant).
        """
        tag_code = self._tag_code(tag)
        tc       = re.escape(tag_code)
        # Options longues d'abord (ex: 'MAY ADD' avant des options plus courtes)
        opts_sorted = sorted(options, key=len, reverse=True)
        norm = self._normalize_ocr_text(text)
        # Valeur de chaque occurrence = jusqu'au prochain tag SWIFT
        pat = rf'(?:^|\n)\s*:?{tc}:\s*(.*?)(?=\n\s*:?\d{{2}}[A-Z]?:|\Z)'
        for m in re.finditer(pat, norm, re.DOTALL | re.IGNORECASE):
            val = self._clean_labeled_value(m.group(1).strip(), tag_code) or ""
            up  = val.upper()
            for opt in opts_sorted:
                if re.search(r'\b' + re.escape(opt.upper()) + r'\b', up):
                    return opt
        return None

    def strip_label(self, value: Optional[str], pattern: str) -> Optional[str]:
        """
        Supprime un label textuel en debut de valeur.
        Ex: 'SEQUENCE OF TOTAL 1/1' + r'\\d+/\\d+' -> '1/1'
        """
        if not value:
            return None
        m = re.search(pattern, value)
        return m.group(0).strip() if m else value.strip()

    def extract_date_any_format(self, value: Optional[str]) -> Optional[str]:
        """Convertit une date dans n'importe quel format vers ISO YYYY-MM-DD."""
        if not value:
            return None
        return self._find_date_in_string(value)

    @staticmethod
    def is_placeholder(value: Optional[str]) -> bool:
        """
        Détecte si une valeur est un placeholder de template SWIFT.
        v6.0 B-PLACEHOLDER : traiter comme null les valeurs non renseignées.
        Ex: 'XXXXXXX', 'XXXX', '0X.XX.2023', 'ADDRESS', 'AMOUNT', 'CODE'
        """
        if not value:
            return True
        v = value.upper().strip()
        # Séquences de X (XXXXXXX, XXXX, XXX...)
        if re.fullmatch(r'X+', v):
            return True
        # Séquences de 0 (000000)
        if re.fullmatch(r'0+', v):
            return True
        # Dates template (0X.XX.2023, XX.XX.2023, XX/XX/XXXX)
        if re.search(r'\b0X\b|XX[./]XX|\bXX\b.*\d{4}', v):
            return True
        # Labels résiduels connus
        if v in {'ADDRESS', 'AMOUNT', 'PLACE', 'NAME', 'CODE', 'XXXXXX', 'XXXXX',
                 'YYYY', 'NNNN', 'BANK', 'COUNTRY', 'CURRENCY', 'DATE'}:
            return True
        return False

    def is_charges_value(self, value: Optional[str]) -> bool:
        """
        Vérifie qu'une valeur de :71B: est bien des frais (pas une durée de présentation).
        Retourne False si la valeur ressemble à ':48: DAYS' ou 'SEE F47A'.
        """
        if not value:
            return False
        v = value.upper().strip()
        # Faux positifs connus
        if re.match(r'^\d+\s*DAYS?', v):
            return False
        if v.startswith('SEE ') or v.startswith('WITHIN '):
            return False
        if re.match(r'^WITHIN\s+', v):
            return False
        return True

    # ------------------------------------------------------------------
    # Implémentation interne
    # ------------------------------------------------------------------

    def _clean_labeled_value(self, value: Optional[str], tag_code: str) -> Optional[str]:
        """
        Supprime le label verbeux en début de valeur pour un tag donné.
        Ex: tag='45A', value='Description of Goods 10,000 TOWELS...'
            → '10,000 TOWELS...'
        B11 : supprime aussi les labels résiduels EN FIN de valeur pour :50:
            Ex: 'XXXX XXXX BENEFICIARY' → 'XXXX XXXX'
        """
        if not value:
            return None

        label_pattern = _VERBOSE_LABELS.get(tag_code.upper())
        if not label_pattern:
            return value

        # Supprimer le label en début de valeur (insensible à la casse)
        cleaned = re.sub(r'^\s*' + label_pattern, '', value, flags=re.IGNORECASE).strip()

        # Supprimer aussi les '+' ou '.' en début de liste (ex: '+ COMMERCIAL INVOICE')
        cleaned = re.sub(r'^[\+\.\-]+\s*', '', cleaned).strip()

        # B11 : pour :50:, supprimer aussi 'BENEFICIARY' résiduel en fin de valeur
        # Ex: ':50: XXXX XXXX BENEFICIARY :59:' → _extract_raw retourne 'XXXX XXXX BENEFICIARY'
        if tag_code.upper() == '50':
            cleaned = re.sub(r'\s+BENEFICIARY\s*$', '', cleaned, flags=re.IGNORECASE).strip()

        # B12 : si la valeur est vide après strip (ex: :59: 'ADDRESS' → ''), retourner None
        if not cleaned:
            return None

        return cleaned if cleaned else value

    def _find_date_in_string(self, s: str) -> Optional[str]:
        """
        Cherche et convertit une date dans une chaîne, tous formats.
        Supporte : YYMMDD, DD-MMM-YYYY, DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD.
        B13 : ignore explicitement les dates template avec 'X' ou '0X' (0X.XX.2023, XX.XX.2023).
        """
        if not s:
            return None

        # B13 : Rejeter immédiatement les dates template (contiennent X ou 0X)
        if re.search(r'\b0X\b|XX[./]XX|[Xx]{2}[./][Xx0-9]{2}[./]\d{4}', s, re.IGNORECASE):
            return None

        # YYYY-MM-DD (déjà ISO)
        m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', s)
        if m:
            return m.group(0)

        # DD-MMM-YYYY
        m = re.search(
            r'\b(\d{1,2})[-/\s](JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[-/\s](\d{4})\b',
            s, re.IGNORECASE
        )
        if m:
            month_map = {
                'JAN':'01','FEB':'02','MAR':'03','APR':'04',
                'MAY':'05','JUN':'06','JUL':'07','AUG':'08',
                'SEP':'09','OCT':'10','NOV':'11','DEC':'12'
            }
            dd = m.group(1).zfill(2)
            mm = month_map[m.group(2).upper()]
            yyyy = m.group(3)
            return f"{yyyy}-{mm}-{dd}"

        # YYMMDD (6 chiffres consécutifs, pas de X)
        m = re.search(r'\b(\d{6})\b', s)
        if m:
            iso = self._yymmdd_to_iso(m.group(1))
            if iso:
                return iso

        # DD.MM.YYYY ou DD/MM/YYYY (sans X)
        m = re.search(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b', s)
        if m and 'X' not in s[m.start():m.end()].upper():
            dd, mm, yyyy = m.group(1).zfill(2), m.group(2).zfill(2), m.group(3)
            try:
                datetime(int(yyyy), int(mm), int(dd))
                return f"{yyyy}-{mm}-{dd}"
            except ValueError:
                pass

        return None

    def _extract_place_from_expiry(self, raw: str) -> Optional[str]:
        """Extrait le lieu d'expiration depuis un champ :31D:/:31E: verbeux."""
        if not raw:
            return None
        # Après la date, chercher le lieu
        # Supprimer le label et la date
        cleaned = re.sub(
            r'(?:DATE\s+(?:AND\s+)?PLACE\s+OF\s+EXPIRY|EXPIRY\s+DATE(?:\s+AND\s+PLACE)?)\s*',
            '', raw, flags=re.IGNORECASE
        ).strip()
        # Supprimer la date elle-même
        cleaned = re.sub(
            r'\b\d{6}\b|\b\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\b|'
            r'\b\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}\b',
            '', cleaned, flags=re.IGNORECASE
        ).strip()
        # Supprimer "PLACE" résiduel en début
        cleaned = re.sub(r'^PLACE\s*', '', cleaned, flags=re.IGNORECASE).strip()
        # Ne pas retourner des résidus de tags ou labels
        if len(cleaned) < 3 or re.match(r'^\d', cleaned):
            return None
        # Stopper si on tombe sur un label de tag suivant
        for stopper in _VERBOSE_STOPPERS:
            m = re.search(stopper, cleaned, re.IGNORECASE)
            if m:
                cleaned = cleaned[:m.start()].strip()
                break
        return cleaned or None

    def _clean_expiry_place(self, place: str) -> str:
        """Nettoie le lieu d'expiration (supprime les labels et tags résiduels)."""
        # Supprimer tout ce qui ressemble à un tag ou label après le lieu
        cleaned = place.strip()
        for stopper in _VERBOSE_STOPPERS:
            m = re.search(stopper, cleaned, re.IGNORECASE)
            if m:
                cleaned = cleaned[:m.start()].strip()
        # Supprimer les tags SWIFT résiduels (ex: ":52D:")
        cleaned = re.sub(r'\s*:\d{2}[A-Z]?\s*:.*', '', cleaned).strip()
        # Supprimer les codes de tags OCR (ex: "52D")
        cleaned = re.sub(r'\s+\d{2}[A-Z]\b.*', '', cleaned).strip()
        return cleaned

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        """
        Pré-normalisation du texte OCR.
        v5.1 : ajout stoppers verbeux pour insérer \n avant les labels de tags.
        """
        t = text
        # 1. Fins de ligne
        t = t.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Séparer montant collé au tag suivant AVANT corrections de tags
        collated_tags = [
            '77U', '77C', '77A', '77u', '77c',
            '71D', '71B', 'Z1D', '7lD', '7lB',
            '72Z', '72',
            '40C', '40E', '40A',
        ]
        for ct in collated_tags:
            t = re.sub(
                r'([\d\.])(' + re.escape(ct) + r'\s*:)',
                r'\1 \2',
                t, flags=re.IGNORECASE
            )

        # 3. Corrections OCR tags erronés
        ocr_tag_fixes = [
            (r'\b40[cC]\s*:', '50:'),
            (r'\b310\s*:',   '31E:'),
            (r'\b77[cCuU]\s*:', '77U:'),
            (r'\bZ1D\s*:',   '71D:'),
            (r'\b7[lL1]D\s*:', '71D:'),
            (r'\b7[lL1]B\s*:', '71B:'),
            (r'\b4[oO]E\s*:', '40E:'),
            (r'\b4[oO]A\s*:', '40A:'),
        ]
        for pattern, replacement in ocr_tag_fixes:
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)

        # 4a. Insérer \n devant ':NN:' / ':NNA:'
        t = re.sub(r'(?<=[^\n])(?=\s*:\d{2}[A-Z]?:)', '\n', t)

        # 4b. Insérer \n devant 'NNA:' sans ':' initial (précédé d'espace)
        t = re.sub(r'(?<=\s)(?=(\d{2}[A-Z]?):(?!\d))', '\n', t)

        # 4c. Tag collé à ':' de ponctuation
        t = re.sub(r'(:)(:?\d{2}[A-Z]?:)', r'\1 \n\2', t)

        # 5. ':40E: : UCP' → ':40E: UCP'
        t = re.sub(r'(:\d{2}[A-Z]?:)\s*:\s*', r'\1 ', t)

        # 6. '; 48 ;' → ':48:'
        t = re.sub(r';\s*(\d{2}[A-Z]?)\s*[;:]', r':\1:', t)

        # 7. ': 40E :' → ':40E:'
        t = re.sub(r':\s*(\d{2}[A-Z]?)\s*:', r':\1:', t)

        return t

    def _extract_raw(self, text: str, tag: str) -> Optional[str]:
        """
        Extrait la valeur brute d'un tag depuis le texte OCR.
        v5.1 : NEXT lookahead inclut les stoppers verbeux.
        v5.4 : exclut le stopper propre au tag (_TAG_OWN_STOPPER) pour éviter
               qu'un tag bloque l'extraction de sa propre valeur
               (ex: :43P: 'PARTIAL SHIPMENTS NOT ALLOWED' stoppait sur 'PARTIAL SHIPMENTS').
        """
        tag_code   = self._tag_code(tag)
        tc         = re.escape(tag_code)

        # Exclure le stopper correspondant au tag lui-même
        own_stopper = _TAG_OWN_STOPPER.get(tag_code.upper())
        active_stoppers = [s for s in _VERBOSE_STOPPERS if s != own_stopper]
        verbose_stopper = '|'.join(active_stoppers)
        NEXT = r'(?=(?:^|\n|\s+):?\d{2}[A-Z]?:|\Z|(?:' + verbose_stopper + r'))'

        normalized = self._normalize_ocr_text(text)

        patterns = [
            (normalized, rf'(?:^|\n)\s*:{tc}:\s*(.*?){NEXT}'),
            (normalized, rf'(?:^|\n)\s*{tc}:\s*(.*?){NEXT}'),
            (text,       rf':{tc}:\s*(.*?){NEXT}'),
            (text,       rf'(?:^|\s){tc}:\s*(.*?){NEXT}'),
        ]

        for src, pattern in patterns:
            m = re.search(pattern, src, re.DOTALL | re.IGNORECASE)
            if m:
                raw = m.group(1)
                lines = []
                for line in raw.splitlines():
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
                value = " ".join(lines).strip()
                value = re.sub(r'^:\s*', '', value)
                if value:
                    return value

        return None

    @staticmethod
    def _tag_code(tag: str) -> str:
        return re.sub(r'[:\s]', '', tag).upper()

    @staticmethod
    def _yymmdd_to_iso(value: str) -> Optional[str]:
        v = value.strip()
        if len(v) < 6 or not v[:6].isdigit():
            return None
        yy = int(v[:2])
        year = 2000 + yy if yy < 70 else 1900 + yy
        try:
            return datetime(year, int(v[2:4]), int(v[4:6])).strftime("%Y-%m-%d")
        except ValueError:
            return None

    _CCY_OCR_FIXES: dict = {
        "U5D": "USD", "U$D": "USD", "U5O": "USD", "USO": "USD",
        "EU R": "EUR", "EUP": "EUR", "EVR": "EUR",
        "GBP": "GBP", "GB P": "GBP",
        "CH F": "CHF", "CNY": "CNY",
    }

    # Subset ISO 4217 — devises les plus courantes dans les messages SWIFT
    # (liste suffisante pour rejeter les fragments OCR comme 'DAM', 'AMO', etc.)
    _ISO4217_COMMON: frozenset = frozenset({
        "USD","EUR","GBP","JPY","CHF","CNY","CNH","HKD","SGD","AUD","CAD","NZD",
        "SEK","NOK","DKK","PLN","CZK","HUF","RON","BGN","HRK","TRY","RUB","UAH",
        "BRL","MXN","ARS","CLP","COP","PEN","VES","DZD","MAD","TND","EGP","LYD",
        "SAR","AED","QAR","KWD","BHD","OMR","JOD","ILS","INR","PKR","BDT","LKR",
        "THB","IDR","MYR","PHP","VND","KRW","TWD","XOF","XAF","XPF","ZAR","KES",
        "NGN","GHS","ETB","TZS","UGX","RWF","MZN","AOA","XAU","XAG","SDR","XDR",
    })

    @classmethod
    def _is_valid_currency(cls, ccy: str) -> bool:
        """Vérifie qu'une devise est un code ISO 4217 reconnu."""
        return ccy.upper() in cls._ISO4217_COMMON

    @classmethod
    def _fix_currency(cls, raw_ccy: str) -> str:
        cleaned = raw_ccy.replace(" ", "").upper()
        return cls._CCY_OCR_FIXES.get(cleaned, cleaned)

    @classmethod
    def _parse_currency_amount(cls, raw: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extrait (devise, montant) depuis une chaîne brute.
        v5.1 : gère le séparateur décimal européen (USD90.000, → 90000)
               et les montants avec lettre nulle (USDAMOUNT → None).
        v5.2 : strip virgule finale avant parsing (USD90.000, → USD90.000)
        v6.0 : B-PLACEHOLDER : 'USDAMOUNT' → (None, None) — détecte le mot AMOUNT
               B-CURRENCY    : devise rejetée si suivie immédiatement de lettres (non-ISO)
        """
        s = raw.strip()

        # B-PLACEHOLDER : si le champ contient 'AMOUNT' comme token entier → template
        if re.search(r'\bAMOUNT\b', s, re.IGNORECASE):
            return None, None

        m = re.match(r"([A-Z0-9]{3})\s*([0-9OoIl][0-9OoIl\.,\s\']*)", s, re.IGNORECASE)
        if not m:
            return None, None

        ccy_raw = m.group(1).upper()
        ccy = cls._fix_currency(ccy_raw)

        # B-CURRENCY : valider que la devise est un code ISO plausible (3 lettres, connue)
        # Si le code extrait est suivi immédiatement de lettres (ex: 'DAM' de 'USDAMOUNT'),
        # et ne figure pas dans la liste ISO connue, rejeter.
        if not cls._is_valid_currency(ccy):
            return None, None

        amt_raw = m.group(2)
        # Vérifier que ce n'est pas un template (AMOUNT, XXXXX)
        if re.search(r'[A-Z]{3,}', amt_raw.upper()):
            return ccy, None

        # Corriger les chiffres mal lus
        amt_raw = amt_raw.replace("O", "0").replace("o", "0")
        amt_raw = amt_raw.replace("I", "1").replace("l", "1")
        amt_raw = amt_raw.replace("'", "").replace(" ", "")

        # F7 : supprimer la virgule/point final parasite (ex: '90.000,' → '90.000')
        amt_raw = amt_raw.rstrip(",.")

        # Détecter le format européen : NNN.NNN,NN (point = milliers, virgule = décimal)
        if re.match(r'^\d{1,3}(\.\d{3})+,\d{2}$', amt_raw):
            amt_raw = amt_raw.replace(".", "").replace(",", ".")
        # Format européen sans décimales : NNN.NNN (point = milliers seulement)
        elif re.match(r'^\d{1,3}(\.\d{3})+$', amt_raw):
            amt_raw = amt_raw.replace(".", "")
        elif amt_raw.count(",") == 1 and "." not in amt_raw:
            # Virgule seule = décimal
            amt_raw = amt_raw.replace(",", ".")
        else:
            # Sinon supprimer les virgules (séparateurs de milliers)
            amt_raw = amt_raw.replace(",", "")

        try:
            float(amt_raw)  # Valider
        except ValueError:
            return ccy, None

        return ccy, amt_raw