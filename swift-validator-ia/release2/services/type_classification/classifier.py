"""
classifier.py
=============
TypeClassifier — classification unifiée PDF natif / scanné / image.

Routing :
  PDF natif  → KeywordClassifier  (règles + mots-clés, zéro ML)
  PDF scanné → ClassifierService  (ResNet-18 fine-tuné)
  Image      → ClassifierService  (ResNet-18 fine-tuné)

Usage :
  clf = TypeClassifier()
  doc_type, confidence, probas = clf.classify("doc.pdf")
  # doc_type ∈ {"bl", "invoice", "packing_list"}
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
_MIN_NATIVE_WORDS = 10


class TypeClassifier:
    """
    Classifieur de type de document LC (BL / Invoice / Packing List).

    Lazy-loading : les modèles ML ne sont chargés que si un PDF scanné
    ou une image est présenté.
    """

    def __init__(
        self,
        classifier_model_dir: Path | str | None = None,
    ):
        self._classifier_model_dir = (
            Path(classifier_model_dir) if classifier_model_dir else None
        )
        self._visual_clf = None  # ClassifierService chargé à la demande

    # ── API publique ──────────────────────────────────────────────────────────

    def classify(
        self,
        path:       str | Path,
        input_type: str | None = None,
    ) -> tuple[str, float, dict]:
        """
        Classifie un document.

        Paramètres
        ----------
        path       : chemin vers le fichier (PDF ou image)
        input_type : "pdf_native" | "pdf_scanned" | "image" (auto-détecté si None)

        Retourne
        --------
        (doc_type: str, confidence: float, probas: dict)
          doc_type ∈ {"bl", "invoice", "packing_list"}
        """
        path = Path(path)
        if input_type is None:
            input_type = self._detect_type(path)

        log.info("[TypeClassifier] %s  format=%s", path.name, input_type)

        if input_type == "pdf_native":
            return self._classify_keyword(path)
        else:
            return self._classify_visual(path, input_type)

    def detect_type(self, path: str | Path) -> str:
        """
        Détecte le format d'entrée du fichier.

        Retourne "image" | "pdf_native" | "pdf_scanned"
        """
        return self._detect_type(Path(path))

    # ── Détection du format ───────────────────────────────────────────────────

    def _detect_type(self, path: Path) -> str:
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            return "image"

        if path.suffix.lower() == ".pdf":
            try:
                import fitz
                doc    = fitz.open(str(path))
                page   = doc[0]
                n_words = len(page.get_text("words"))
                doc.close()
                return "pdf_native" if n_words >= _MIN_NATIVE_WORDS else "pdf_scanned"
            except Exception:
                return "pdf_native"

        return "pdf_native"

    # ── Classifieur par mots-clés (PDF natif) ────────────────────────────────

    @staticmethod
    def _classify_keyword(path: Path) -> tuple[str, float, dict]:
        """Utilise le KeywordClassifier (règles, zéro ML)."""
        try:
            from release2.services.keyword_classifier import classify_pdf
        except ImportError:
            from services.keyword_classifier import classify_pdf

        return classify_pdf(path)

    # ── Classifieur visuel (PDF scanné / image) ───────────────────────────────

    def _classify_visual(self, path: Path, input_type: str) -> tuple[str, float, dict]:
        """Utilise le ClassifierService (ResNet-18)."""
        from PIL import Image

        clf = self._get_visual_classifier()

        if input_type == "image":
            image = Image.open(path).convert("RGB")
        else:
            import fitz
            doc   = fitz.open(str(path))
            pix   = doc.load_page(0).get_pixmap(dpi=150)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            doc.close()

        return clf.predict_hierarchical(image)

    def _get_visual_classifier(self):
        if self._visual_clf is None:
            try:
                from release2.services.classifier_service import ClassifierService
            except ImportError:
                from services.classifier_service import ClassifierService

            self._visual_clf = ClassifierService(self._classifier_model_dir)
        return self._visual_clf
