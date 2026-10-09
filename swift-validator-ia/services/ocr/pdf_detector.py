"""
pdf_detector.py
Détecte si un PDF est scanné (images) ou textuel (couche texte native).
"""

import fitz  # PyMuPDF


class PDFDetector:
    """
    Heuristique : si aucune page ne contient plus de MIN_CHARS caractères,
    le PDF est considéré comme scanné.
    """

    MIN_CHARS = 50

    @classmethod
    def is_scanned(cls, pdf_path: str) -> bool:
        with fitz.open(pdf_path) as doc:
            return all(
                len(page.get_text().strip()) <= cls.MIN_CHARS
                for page in doc
            )