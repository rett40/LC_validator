"""
pdf_text_extractor.py
Extraction du texte natif d'un PDF textuel (non scanné).
"""

import fitz  # PyMuPDF


class PDFTextExtractor:
    """Extrait le texte de chaque page d'un PDF textuel via PyMuPDF."""

    @staticmethod
    def extract(pdf_path: str) -> str:
        with fitz.open(pdf_path) as doc:
            return "".join(page.get_text("text") for page in doc)