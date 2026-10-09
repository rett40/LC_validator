"""
ocr_service.py
Point d'entrée principal : traite un fichier (PDF ou image) et retourne
le texte extrait avec un score de confiance global.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from loguru import logger
from pdf2image import convert_from_path

from .image_enhancer import ImageEnhancer
from .ocr_engines import OCREngines
from .pdf_detector import PDFDetector
from .pdf_text_extractor import PDFTextExtractor
from .postprocessor import SwiftPostProcessor


class OCRService:
    """
    Orchestre le pipeline OCR complet.

    Flux :
      PDF textuel  → extraction directe (PyMuPDF)
      PDF scanné   → conversion en images page par page → OCR hybride
      Image        → OCR hybride
    """

    def __init__(self,
                 dpi: int = 350,
                 pdf_thread_count: int = 4):
        self.dpi = dpi
        self.pdf_thread_count = pdf_thread_count

        self.enhancer = ImageEnhancer()
        self.engines = OCREngines()
        self.pdf_detector = PDFDetector()
        self.pdf_extractor = PDFTextExtractor()
        self.postprocessor = SwiftPostProcessor()
        self._spell_corrector: Any = None

    @property
    def spell_corrector(self) -> Any:
        """SwiftSpellCorrector (lazy) — même logique que l’ancienne étape [SPL] du pipeline."""
        if self._spell_corrector is None:
            from .spell_correction import SwiftSpellCorrector

            self._spell_corrector = SwiftSpellCorrector()
        return self._spell_corrector

    def _apply_spell(self, cleaned: str) -> tuple[str, Any]:
        try:
            return self.spell_corrector.correct(cleaned)
        except Exception as exc:
            logger.warning("Correction orthographique (OCR) ignorée : {}", exc)
            return cleaned, None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def process(self, file_path: str) -> dict:
        """
        Traite un fichier et retourne :
          {"text": str, "confidence": float ∈ [0.0, 1.0], "spell_report": ...|None}
        """
        logger.info(f"Début du traitement : {file_path}")

        if file_path.lower().endswith('.pdf'):
            return self._process_pdf(file_path)

        return self._process_image_file(file_path)

    # ------------------------------------------------------------------
    # Traitement PDF
    # ------------------------------------------------------------------

    def _process_pdf(self, file_path: str) -> dict:
        if not self.pdf_detector.is_scanned(file_path):
            logger.info("PDF textuel détecté — extraction directe")
            text = self.pdf_extractor.extract(file_path)
            cleaned = self.postprocessor.clean(text)
            corrected, spell_report = self._apply_spell(cleaned)
            return {
                "text": corrected,
                "confidence": 1.0,
                "spell_report": spell_report,
            }

        logger.info("PDF scanné détecté — conversion page par page")
        return self._process_scanned_pdf(file_path)

    def _process_scanned_pdf(self, file_path: str) -> dict:
        """
        Génère les pages une par une pour limiter la consommation mémoire.
        Chaque page PIL est libérée dès son traitement.
        """
        try:
            pages = convert_from_path(
                file_path,
                dpi=self.dpi,
                fmt='png',
                thread_count=self.pdf_thread_count,
            )
        except Exception as exc:
            logger.error(f"Échec de la conversion PDF : {exc}")
            return {"text": "", "confidence": 0.0, "spell_report": None}

        parts: list[str] = []
        confs: list[float] = []

        for i, pil_page in enumerate(pages):
            bgr = cv2.cvtColor(np.array(pil_page), cv2.COLOR_RGB2BGR)
            text, conf = self._ocr_image(bgr)
            parts.append(text)
            confs.append(conf)
            logger.debug(f"Page {i + 1}/{len(pages)} — confiance = {conf:.2f}")
            del pil_page  # libération mémoire immédiate

        return self._assemble(parts, confs)

    # ------------------------------------------------------------------
    # Traitement image
    # ------------------------------------------------------------------

    def _process_image_file(self, file_path: str) -> dict:
        img = cv2.imread(file_path)
        if img is None:
            logger.error(f"Impossible de lire l'image : {file_path}")
            return {"text": "", "confidence": 0.0, "spell_report": None}

        text, conf = self._ocr_image(img)
        return self._assemble([text], [conf])

    # ------------------------------------------------------------------
    # OCR d'une image numpy
    # ------------------------------------------------------------------

    def _ocr_image(self, bgr: np.ndarray) -> tuple[str, float]:
        """Améliore l'image et lance l'OCR hybride."""
        gray, binary = self.enhancer.enhance_both(bgr)
        return self.engines.hybrid_ocr(gray, binary)

    # ------------------------------------------------------------------
    # Assemblage final
    # ------------------------------------------------------------------

    def _assemble(self,
                  parts: list[str],
                  confs: list[float]) -> dict:
        raw_text = " ".join(parts)
        cleaned = self.postprocessor.clean(raw_text)
        corrected, spell_report = self._apply_spell(cleaned)
        overall_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
        logger.success(f"Traitement terminé — confiance globale = {overall_conf}")
        return {
            "text": corrected,
            "confidence": overall_conf,
            "spell_report": spell_report,
        }