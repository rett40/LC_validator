"""
swift_tag_normalizer.py  —  v1.0
Pré-processeur universel de tags SWIFT OCR.

Problème résolu :
  Le texte OCR peut contenir des tags SWIFT dans plusieurs formes :
    - Standard   : :27:  :40A:  :32B:
    - Sans ':'   : 27:   40B    32B:   (OCR manque le ':' initial)
    - OCR erroné : 40c:  310:   77c:   Z1D:
    - Collés     : USD500,000.0077c:   (montant collé au tag suivant)
    - Label avant: 'Sequence of Total 1/1' (label textuel avant la valeur)

Ce module :
  1. Détecte TOUS les tags (formels, sans ':', OCR erronés)
  2. Les normalise en :TAG: standard avec ':' de part et d'autre
  3. Retourne un dict {tag_canonique: valeur_brute_nettoyée}
  4. Nettoie les valeurs (caractères spéciaux, espaces multiples)

Usage :
    from swift_tag_normalizer import SwiftTagNormalizer

    normalizer = SwiftTagNormalizer()
    tags = normalizer.extract(ocr_text)
    # → {'27': '1/1', '20': 'LC2026TEST001', '32B': 'USD250000', ...}

    normalized_text = normalizer.normalize_text(ocr_text)
    # → ':27: 1/1\n:20: LC2026TEST001\n:32B: USD250000\n...'
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Catalogue complet des tags SWIFT valides (tous types MT7xx/MT799)
# ---------------------------------------------------------------------------

_ALL_KNOWN_TAGS: frozenset[str] = frozenset({
    # Références et séquences
    '20', '21', '23', '23B', '25', '27',
    # Dates
    '30', '31C', '31D', '31E', '32A', '32B',
    # Forme, règles, objet
    '22A', '22D', '39A',
    '40A', '40B', '40C', '40E',
    # Disponibilité, traite
    '41A', '41D', '42C',
    # Expédition
    '43P', '43T',
    '44A', '44B', '44C', '44D', '44E', '44F',
    # Marchandises, documents, conditions
    '45A', '46A', '47A', '48', '49',
    # Parties
    '50', '50B', '51A',
    '52A', '52D',
    '53A', '53B', '54A', '55A', '56A', '57A',
    '58A', '59', '59A',
    # Frais, instructions
    '70', '71A', '71B', '71D',
    # Informations banque / garantie
    '72', '72Z', '77A', '77U', '78', '79',
})

# Variantes OCR → tag canonique
_OCR_TAG_FIXES: dict[str, str] = {
    # Applicant mal lu
    '40C': '50',    # '40c:' → ':50:' (Applicant)
    '40c': '50',
    # Date d'expiration
    '310': '31E',   # '310:' → ':31E:'
    # Texte de garantie
    '77C': '77U',   # '77c:' → ':77U:'
    '77c': '77U',
    '77u': '77U',
    # Frais
    'Z1D': '71D',   # 'Z1D:' → ':71D:'
    'Z1d': '71D',
    '7lD': '71D',   # '7lD:' → ':71D:' (l=1 OCR)
    '7lB': '71B',
    '7lA': '71A',
    # Règles
    '4OE': '40E',   # '4OE:' → ':40E:' (O=0 OCR)
    '4OA': '40A',
    '4OB': '40B',
    '4OC': '40C',
}

# Caractères spéciaux à supprimer des valeurs
# (conserver: lettres, chiffres, espace, . , / - + ( ) ' & % @ # = _ ?)
_SPECIAL_CHARS_RE = re.compile(r'[\{\}\[\]\\|^~`<>]')

# Bruit de début de document à ignorer (headers OCR, blocs SWIFT)
_HEADER_NOISE_RE = re.compile(
    r'(?:'
    r'\{[0-9]:[^}]*\}'          # blocs SWIFT {1:...} {2:...} {4:
    r'|'
    r'\{[A-Z0-9:]+\}'           # autres blocs {5:} etc.
    r'|'
    r'SWIFT\s+(?:MT\s*)?\d{3}\b'  # "SWIFT MT700"
    r'|'
    r'SWIFT\s*\}'                # "SWIFT}"
    r')',
    re.IGNORECASE
)


class SwiftTagNormalizer:
    """
    Pré-processeur universel : détecte, normalise et extrait les tags SWIFT
    depuis un texte OCR dégradé.
    """

    def __init__(self, extra_tags: set[str] | None = None,
                 extra_fixes: dict[str, str] | None = None) -> None:
        """
        extra_tags  : tags supplémentaires à reconnaître
        extra_fixes : corrections OCR supplémentaires {raw: canonical}
        """
        self._known_tags = _ALL_KNOWN_TAGS | (extra_tags or set())
        self._fixes = {**_OCR_TAG_FIXES, **(extra_fixes or {})}

        # Construire le regex principal une seule fois
        self._tag_re = self._build_tag_regex()

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict[str, str]:
        """
        Extrait tous les tags et leurs valeurs depuis le texte OCR.

        Retourne un dict ordonné {tag_canonique: valeur_nettoyée}.
        Les tags dupliqués gardent la dernière valeur (ou on peut choisir la plus longue).

        Exemple :
            {'27': '1/1', '20': 'LC2026TEST', '32B': 'USD250000', ...}
        """
        prepared = self._prepare(text)
        positions = list(self._tag_re.finditer(prepared))

        result: dict[str, str] = {}
        for i, m in enumerate(positions):
            raw_tag   = m.group(1)
            canonical = self._normalize_tag(raw_tag)

            start = m.end()
            end   = positions[i + 1].start() if i + 1 < len(positions) else len(prepared)
            raw_value = prepared[start:end]

            value = self._clean_value(raw_value)
            if value:
                # Si le tag apparaît plusieurs fois, on concatène (ex: :72: multilignes)
                if canonical in result:
                    result[canonical] = result[canonical] + ' ' + value
                else:
                    result[canonical] = value

        return result

    def normalize_text(self, text: str) -> str:
        """
        Retourne le texte avec tous les tags normalisés en :TAG: standard,
        chaque tag sur sa propre ligne, valeur nettoyée.

        Utilisable directement par les parseurs existants.

        Exemple :
            ':27: 1/1\n:20: LC2026TEST\n:32B: USD250000\n...'
        """
        tags = self.extract(text)
        lines = []
        for tag, value in tags.items():
            lines.append(f':{tag}: {value}')
        return '\n'.join(lines)

    def count_tags(self, text: str) -> int:
        """Compte le nombre de tags détectés (utile pour le garde-fou détecteur)."""
        prepared = self._prepare(text)
        return len(list(self._tag_re.finditer(prepared)))

    # ------------------------------------------------------------------
    # Construction du regex
    # ------------------------------------------------------------------

    def _build_tag_regex(self) -> re.Pattern:
        """
        Construit le pattern qui matche tous les tags connus (+ variantes OCR).

        Pattern : optionnel ':' + TAG + ':' (pas suivi d'un chiffre)
        Priorité : les tags les plus longs d'abord (31E avant 31).
        """
        # Inclure aussi les variantes OCR dans la liste
        all_forms = set(self._known_tags) | set(self._fixes.keys())
        sorted_forms = sorted(all_forms, key=len, reverse=True)
        tag_alts = '|'.join(re.escape(t) for t in sorted_forms)

        # Matcher :
        #   - ':' optionnel avant le tag (peut manquer en OCR)
        #   - Le tag lui-même
        #   - ':' obligatoire après le tag
        #   - NON suivi d'un chiffre (évite de matcher "12-APR-2021")
        #   - Le tout précédé d'un début de ligne, espace ou ':' (contexte de tag)
        pattern = (
            r'(?:'
            r'(?:^|(?<=\n)|(?<=\s)|(?<=:))'  # contexte gauche
            r':?'                              # ':' optionnel avant
            r'(' + tag_alts + r')'            # TAG capturé
            r'\s*:(?!\d)'                      # ':' après, pas suivi chiffre
            r')'
        )
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    # ------------------------------------------------------------------
    # Traitement interne
    # ------------------------------------------------------------------

    def _prepare(self, text: str) -> str:
        """
        Prépare le texte avant extraction :
        1. Normalise les fins de ligne
        2. Supprime le bruit de header SWIFT
        3. Sépare les montants collés aux tags ('USD500.0077c:' → 'USD500.00 77c:')
        4. Corrige '; 48 ;' → ':48:' et ': 40E :' → ':40E:'
        """
        t = text

        # 1. Fins de ligne
        t = t.replace('\r\n', '\n').replace('\r', '\n')

        # 2. Nettoyer le header SWIFT (blocs {…}, "SWIFT MT700")
        t = _HEADER_NOISE_RE.sub(' ', t)

        # 3. Séparer montant collé au tag suivant
        #    Ex: 'USD500,000.0077c:' → 'USD500,000.00 77c:'
        #    Pattern précis : chiffre final d'un montant + tag connu + ':'
        #    Les tags les plus longs d'abord pour éviter '72' avant '72Z'
        glued_tag_alts = '|'.join(
            sorted(
                {k.upper() for k in self._known_tags} | {k.upper() for k in self._fixes},
                key=len, reverse=True
            )
        )
        glued_re = re.compile(r'(\d)(' + glued_tag_alts + r'):', re.IGNORECASE)
        t = glued_re.sub(lambda m: m.group(1) + ' ' + m.group(2) + ':', t)

        # 4. '; 48 ;' ou '; 48:' → ':48:'
        t = re.sub(r';\s*(\d{2}[A-Z]?)\s*[;:]', r':\1:', t)

        # 5. ': 40E :' → ':40E:'
        t = re.sub(r':\s*(\d{2}[A-Z]?)\s*:', r':\1:', t)

        return t

    def _normalize_tag(self, raw: str) -> str:
        """Retourne le tag canonique (corrige les erreurs OCR)."""
        upper = raw.upper()
        return self._fixes.get(raw, self._fixes.get(upper, upper))

    @staticmethod
    def _clean_value(raw: str) -> str:
        """
        Nettoie la valeur brute d'un tag :
        1. Supprime les caractères spéciaux parasites { } [ ] \\ | ^ ~ `
        2. Supprime les espaces multiples
        3. Supprime les ':' résiduels en début de valeur
        4. Supprime les espaces en début/fin
        """
        v = raw
        # 1. Caractères spéciaux → espace
        v = _SPECIAL_CHARS_RE.sub(' ', v)
        # 2. Espaces multiples
        v = re.sub(r'[ \t]{2,}', ' ', v)
        # 3. Sauts de ligne multiples
        v = re.sub(r'\n{2,}', '\n', v)
        # 4. ':' résiduel en début (ex: ':40E: : UCP' → 'UCP')
        v = v.strip()
        v = re.sub(r'^:\s*', '', v).strip()
        # 5. Trim final
        return v.strip()


# ---------------------------------------------------------------------------
# Fonction utilitaire standalone
# ---------------------------------------------------------------------------

def extract_swift_tags(text: str,
                       extra_tags: set[str] | None = None,
                       extra_fixes: dict[str, str] | None = None) -> dict[str, str]:
    """
    Raccourci : extrait les tags SWIFT depuis un texte OCR.

    Retourne {tag: value} pour tous les tags détectés.

    Usage :
        tags = extract_swift_tags(ocr_text)
        print(tags.get('32B'))   # 'USD250000'
        print(tags.get('50'))    # 'ABC IMPORT SARL...'
    """
    return SwiftTagNormalizer(extra_tags, extra_fixes).extract(text)


def normalize_swift_text(text: str) -> str:
    """
    Raccourci : retourne le texte avec tags normalisés, prêt pour les parseurs.

    Usage :
        normalized = normalize_swift_text(ocr_text)
        # Passer normalized au parseur existant
    """
    return SwiftTagNormalizer().normalize_text(text)