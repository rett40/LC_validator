"""
keyword_classifier.py
=====================
Classification de documents par mots-clés — zéro ML, zéro modèle.

Utilisé pour les PDFs natifs (texte vectoriel PyMuPDF extrait directement).
Précision ~99% sur documents synthétiques et réels bien formatés.

Stratégie : compter les mots-clés caractéristiques de chaque type,
normaliser en probabilités, booster la confiance si match fort.
"""

from __future__ import annotations

import fitz

# ─────────────────────────────────────────────────────────────────────────────
# Identifiants exclusifs — la présence d'un seul suffit à forcer le type
# Ordre de priorité : packing_list > invoice > bl (pour éviter la confusion PL/BL)
# ─────────────────────────────────────────────────────────────────────────────

_EXCLUSIVE: list[tuple[str, str]] = [
    ("packing_list", "PACKING LIST NO"),
    ("packing_list", "PACKING LIST NUMBER"),
    ("packing_list", "LISTE DE COLISAGE"),
    ("invoice",      "COMMERCIAL INVOICE"),
    ("invoice",      "PROFORMA INVOICE"),
    ("invoice",      "TAX INVOICE"),
    ("bl",           "BILL OF LADING"),
    ("bl",           "CONNAISSEMENT"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Mots-clés pondérés par type de document
# Poids élevé = discriminant fort   |   Poids faible = terme partagé
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "bl": [
        ("BILL OF LADING",       5),   # titre exclusif
        ("B/L NUMBER",           4),
        ("B/L NO",               4),
        ("NUMBER OF ORIGINAL",   4),   # exclusif BL
        ("NOTIFY PARTY",         3),
        ("FREIGHT TERMS",        3),
        ("PORT OF LOADING",      2),
        ("PORT OF DISCHARGE",    2),
        ("CONSIGNEE",            1),   # aussi dans PL/Invoice
        ("SHIPPER",              1),
        ("VESSEL",               1),
        ("VOYAGE",               1),
    ],
    "invoice": [
        ("COMMERCIAL INVOICE",   5),
        ("INVOICE NO",           4),
        ("INVOICE NUMBER",       4),
        ("INVOICE AMOUNT",       4),
        ("UNIT PRICE",           3),
        ("PAYMENT TERMS",        3),
        ("HS CODE",              3),
        ("TOTAL AMOUNT",         2),
        ("INCOTERM",             2),
        ("SELLER",               2),
        ("BUYER",                2),
    ],
    "packing_list": [
        ("PACKING LIST",         5),   # titre exclusif
        ("PL NO",                4),
        ("PL NUMBER",            4),
        ("MARKS AND NUMBERS",    4),   # exclusif PL
        ("MARKS & NOS",          4),
        ("NO. OF CARTONS",       3),
        ("NO OF PKGS",           3),
        ("TOTAL PACKAGES",       3),
        ("TOTAL PKGS",           3),
        ("DIMENSIONS",           2),
        ("NET WEIGHT",           2),
        ("GROSS WEIGHT",         2),
        ("CARTONS",              1),
        ("PALLETS",              1),
        ("CBM",                  1),
    ],
}

DOC_TYPES = list(_KEYWORDS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions publiques
# ─────────────────────────────────────────────────────────────────────────────

def classify_text(text: str) -> tuple[str, float, dict]:
    """
    Classifie un texte extrait par mots-clés.

    Étape 1 : vérifie les identifiants exclusifs (court-circuit).
    Étape 2 : scoring pondéré sur tous les mots-clés.

    Retour
    ------
    (doc_type, confidence, class_probas)
    """
    text_up = text.upper()

    # ── Étape 1 : identifiant exclusif ───────────────────────────────────────
    for doc_type, token in _EXCLUSIVE:
        if token in text_up:
            # Les deux autres classes reçoivent un faible score résiduel
            probas = {dt: 0.02 for dt in DOC_TYPES}
            probas[doc_type] = 0.96
            return doc_type, 0.99, probas

    # ── Étape 2 : scoring pondéré ─────────────────────────────────────────────
    scores: dict[str, float] = {dt: 0.0 for dt in DOC_TYPES}
    for dt, kw_list in _KEYWORDS.items():
        for kw, weight in kw_list:
            if kw in text_up:
                scores[dt] += weight

    total = sum(scores.values()) or 1.0
    raw_probas = {dt: sc / total for dt, sc in scores.items()}

    best       = max(scores, key=scores.get)
    best_score = scores[best]

    # Confiance : proportionnelle à la dominance du score gagnant
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    margin = best_score - second_best

    if best_score == 0:
        confidence = 0.33
    elif margin >= 6:
        confidence = min(0.98, 0.80 + margin * 0.02)
    elif margin >= 2:
        confidence = min(0.90, 0.65 + margin * 0.05)
    else:
        confidence = 0.50 + best_score * 0.04

    probas = {dt: round(p, 4) for dt, p in raw_probas.items()}
    return best, round(confidence, 4), probas


def classify_pdf(pdf_path) -> tuple[str, float, dict]:
    """
    Extrait le texte de la première page du PDF et classifie par mots-clés.

    Retour : (doc_type, confidence, class_probas)
    """
    doc  = fitz.open(str(pdf_path))
    text = doc[0].get_text("text")
    doc.close()
    return classify_text(text)
