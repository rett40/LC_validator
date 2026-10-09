"""
classifier_service.py
=====================
Service de classification visuelle de documents — sans OCR.

Charge le modèle ResNet-18 fine-tuné et prédit le type de document
(bl / invoice / packing_list) depuis un PDF ou une image.

Usage (standalone) :
  python services/classifier_service.py --pdf path/to/doc.pdf

Usage (import) :
  from release2.services.classifier_service import ClassifierService
  clf = ClassifierService()
  doc_type, confidence = clf.predict_pdf("doc.pdf")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import fitz
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
try:
    import pytesseract as _pytesseract
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False

log = logging.getLogger(__name__)

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR    = RELEASE2_ROOT / "models" / "doc_classifier"

RENDER_DPI = 150
IMG_SIZE   = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires image
# ─────────────────────────────────────────────────────────────────────────────

def _pdf_to_pil(pdf_path: str | Path, page_num: int = 0) -> Image.Image:
    """Rendu de la première page d'un PDF en image PIL RGB."""
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = doc[page_num].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img_bgr = cv2.cvtColor(
        np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3),
        cv2.COLOR_RGB2BGR
    )
    doc.close()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def _preprocess(image: Image.Image):
    """Applique les transforms ImageNet standard et retourne un tensor batch."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)   # [1, 3, 224, 224]


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierService:
    """
    Classifieur visuel de documents basé sur ResNet-18.

    Charge le modèle une seule fois et expose :
      - predict_pdf(pdf_path)    → (doc_type, confidence, probas)
      - predict_image(pil_image) → (doc_type, confidence, probas)
    """

    def __init__(self, model_dir: Path | str | None = None):
        import torch
        import torch.nn as nn
        from torchvision import models

        model_dir = Path(model_dir) if model_dir else MODELS_DIR

        model_path = model_dir / "best_model.pt"
        names_path = model_dir / "class_names.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}\n"
                "Exécuter 05_train_classifier.py d'abord."
            )

        self.class_names = json.loads(names_path.read_text(encoding="utf-8"))
        self.device = torch.device("cpu")

        # Reconstruction de l'architecture
        base = models.resnet18(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.class_names)),
        )
        base.load_state_dict(torch.load(model_path, map_location=self.device))
        base.to(self.device)
        base.eval()
        self.model = base

        # Charger les modèles d'ensemble
        ensemble_dir = RELEASE2_ROOT.parent / "models" / "ensemble"
        if ensemble_dir.exists():
            # RandomForest sur features métier
            rf_path = ensemble_dir / "random_forest.joblib"
            if rf_path.exists():
                self.rf_model = joblib.load(rf_path)
                log.info("RandomForest chargé depuis %s", rf_path)
            else:
                self.rf_model = None
                log.warning("RandomForest non trouvé à %s", rf_path)

            # TF-IDF + LogisticRegression
            tfidf_path = ensemble_dir / "tfidf_vectorizer.joblib"
            lr_path = ensemble_dir / "logistic_regression.joblib"
            if tfidf_path.exists() and lr_path.exists():
                self.tfidf = joblib.load(tfidf_path)
                self.lr_model = joblib.load(lr_path)
                log.info("TF-IDF + LogisticRegression chargés")
            else:
                self.tfidf = None
                self.lr_model = None
                log.warning("TF-IDF ou LogisticRegression non trouvés")

        else:
            self.rf_model = None
            self.tfidf = None
            self.lr_model = None
            log.warning("Dossier ensemble non trouvé à %s", ensemble_dir)

        # Cache OCR par image (hash → texte) — évite les appels redondants
        self._ocr_cache: dict[int, str] = {}

    def predict_image(self, image: Image.Image) -> tuple[str, float, dict]:
        """
        Prédit le type de document depuis une image PIL.

        Returns:
            doc_type   (str)   : "bl" | "invoice" | "packing_list"
            confidence (float) : probabilité de la classe prédite (0-1)
            probas     (dict)  : {classe: probabilité} pour toutes les classes
        """
        import torch

        tensor  = _preprocess(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0]
            probas = torch.softmax(logits, dim=0).cpu().numpy()

        pred_idx   = int(np.argmax(probas))
        doc_type   = self.class_names[pred_idx]
        confidence = float(probas[pred_idx])
        probas_dict = {cls: round(float(p), 4)
                       for cls, p in zip(self.class_names, probas)}

        return doc_type, confidence, probas_dict

    def _ocr_text(self, image: Image.Image) -> str:
        """
        OCR rapide via Tesseract (1-3s vs 100s EasyOCR sur CPU).
        Résultat mis en cache par hash image — appelé N fois sur la même image = 1 seul OCR.
        """
        key = hash(image.tobytes())
        if key in self._ocr_cache:
            return self._ocr_cache[key]
        if _TESSERACT_OK:
            try:
                # PSM 6 = bloc de texte uniforme, OEM 1 = LSTM uniquement (plus rapide)
                text = _pytesseract.image_to_string(
                    image, config="--psm 6 --oem 1 -l eng"
                ).lower()
            except Exception as exc:
                log.warning("[classifier] Tesseract échoué : %s", exc)
                text = ""
        else:
            text = ""
        self._ocr_cache[key] = text
        return text

    def _extract_metier_features(self, image: Image.Image) -> np.ndarray:
        """
        Extrait des features métier via OCR rapide (Tesseract).
        - Présence de mots-clés pour BL, Invoice, Packing List
        - Comptage de mots liés à la logistique/finance
        """
        text = self._ocr_text(image)
        lines = [l for l in text.splitlines() if l.strip()]

        # Mots-clés pour chaque classe
        bl_keywords = ["bill of lading", "bl", "shipper", "consignee", "vessel", "port", "cargo"]
        invoice_keywords = ["invoice", "inv", "amount", "total", "tax", "payment", "due date"]
        packing_keywords = ["packing list", "pl", "item", "quantity", "description", "weight"]

        # Features : présence (1/0) et comptage
        features = []
        for keywords in [bl_keywords, invoice_keywords, packing_keywords]:
            presence = any(kw in text for kw in keywords)
            count = sum(text.count(kw) for kw in keywords)
            features.extend([int(presence), count])

        # Features supplémentaires : longueur du texte, nombre de lignes, ratio mots-clés/total
        features.append(len(text))
        features.append(len(lines))
        features.append(sum(features[:6]) / max(1, len(text.split())))  # ratio mots-clés

        return np.array(features).reshape(1, -1)

    def predict_ensemble(self, image: Image.Image) -> tuple[str, float, dict]:
        """
        Classification avec ensemble : ResNet + RandomForest + TF-IDF/LR
        Utilise stacking fusion avec pondération.
        """
        # Prédiction ResNet
        doc_type_vision, conf_vision, probas_vision = self.predict_image(image)

        # Features métier (OCR Tesseract via cache — calculé une seule fois)
        metier_features = self._extract_metier_features(image)
        text = self._ocr_text(image)

        # Prédiction RandomForest si disponible
        if self.rf_model:
            rf_probas = self.rf_model.predict_proba(metier_features)[0]
            rf_pred = self.class_names[np.argmax(rf_probas)]
            rf_conf = float(np.max(rf_probas))
        else:
            rf_pred, rf_conf, rf_probas = doc_type_vision, 0.0, [0.0] * len(self.class_names)

        # Prédiction TF-IDF + LR si disponible
        if self.tfidf and self.lr_model:
            text_features = self.tfidf.transform([text])  # Vectoriser le texte OCR
            lr_probas = self.lr_model.predict_proba(text_features)[0]
            lr_pred = self.class_names[np.argmax(lr_probas)]
            lr_conf = float(np.max(lr_probas))
        else:
            lr_pred, lr_conf, lr_probas = doc_type_vision, 0.0, [0.0] * len(self.class_names)

        # Stacking fusion : moyenne pondérée (vision 0.5, RF 0.3, LR 0.2)
        weights = {'vision': 0.5, 'rf': 0.3, 'lr': 0.2}
        ensemble_probas = {}
        for i, cls in enumerate(self.class_names):
            ensemble_probas[cls] = (
                probas_vision[cls] * weights['vision'] +
                rf_probas[i] * weights['rf'] +
                lr_probas[i] * weights['lr']
            )

        # Normaliser
        total = sum(ensemble_probas.values())
        if total > 0:
            ensemble_probas = {cls: p / total for cls, p in ensemble_probas.items()}

        pred_idx = np.argmax(list(ensemble_probas.values()))
        doc_type = self.class_names[pred_idx]
        confidence = float(ensemble_probas[doc_type])

        return doc_type, confidence, ensemble_probas

    def predict_hierarchical(self, image: Image.Image) -> tuple[str, float, dict]:
        """
        Classification hiérarchique :
        1. Financial vs Logistics
        2. Si Logistics : BL vs Packing List
        """
        # Étape 1 : Financial vs Logistics
        # Mapper : invoice = financial, bl + packing_list = logistics
        doc_type, conf, probas = self.predict_ensemble(image)

        if doc_type == "invoice":
            # Financial
            final_type = "invoice"
            final_conf = conf
            final_probas = probas
        else:
            # Logistics : distinguer BL vs Packing List via mots-clés (texte OCR depuis cache)
            text = self._ocr_text(image)
            bl_score = text.count("bill of lading") + text.count("bl") + text.count("vessel")
            pl_score = text.count("packing list") + text.count("pl") + text.count("item")

            if bl_score > pl_score:
                final_type = "bl"
                final_conf = conf * 0.9  # Ajuster confiance
            else:
                final_type = "packing_list"
                final_conf = conf * 0.9

            # Probabilités ajustées
            final_probas = {cls: 0.0 for cls in self.class_names}
            final_probas[final_type] = final_conf
            # Distribuer le reste
            remaining = 1.0 - final_conf
            for cls in self.class_names:
                if cls != final_type:
                    final_probas[cls] = remaining / (len(self.class_names) - 1)

        return final_type, final_conf, final_probas

    def predict_pdf(
        self,
        pdf_path: str | Path,
        page_num: int = 0,
    ) -> tuple[str, float, dict]:
        """
        Prédit le type de document depuis la première page d'un PDF.

        Returns:
            doc_type   (str)
            confidence (float)
            probas     (dict)
        """
        image = _pdf_to_pil(pdf_path, page_num=page_num)
        return self.predict_image(image)

    def predict_image_path(self, image_path: str | Path) -> tuple[str, float, dict]:
        """Prédit depuis un chemin vers une image PNG/JPG."""
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)

    def is_confident(self, confidence: float, threshold: float = 0.80) -> bool:
        """Retourne True si la confiance dépasse le seuil."""
        return confidence >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# CLI standalone
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="classifier_service — classification visuelle de document (sans OCR)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",   help="Chemin vers le PDF")
    group.add_argument("--image", help="Chemin vers l'image PNG/JPG")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Seuil de confiance (défaut: 0.80)")
    args = parser.parse_args()

    clf = ClassifierService(model_dir=args.model_dir)

    if args.pdf:
        doc_type, confidence, probas = clf.predict_pdf(args.pdf)
        source = args.pdf
    else:
        doc_type, confidence, probas = clf.predict_image_path(args.image)
        source = args.image

    print(f"\nDocument : {source}")
    print(f"Type prédit  : {doc_type}")
    print(f"Confiance    : {confidence:.1%}")
    print(f"Probabilités : {probas}")

    if not clf.is_confident(confidence, args.threshold):
        print(f"\n⚠  Confiance faible ({confidence:.1%} < {args.threshold:.0%}) — vérification manuelle recommandée")
    else:
        print(f"\n✓  Classification fiable")


if __name__ == "__main__":
    main()
