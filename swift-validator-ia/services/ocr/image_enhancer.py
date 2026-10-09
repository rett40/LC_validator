"""
image_enhancer.py
Prétraitement d'image pour optimiser la reconnaissance OCR.
"""

import cv2
import numpy as np


class ImageEnhancer:
    """
    Fournit deux versions d'une image pour l'OCR :
    - gray_enhanced  : niveaux de gris + CLAHE + débruitage + redressement  → EasyOCR
    - binary_enhanced: binarisation Otsu + fermeture morphologique          → Tesseract
    """

    # ---------- API publique ----------

    def enhance_both(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne (gray_enhanced, binary_enhanced).
        Accepte une image BGR, BGRA ou déjà en niveaux de gris.
        """
        img = self._upscale(image)
        gray = self._to_gray(img)
        denoised = cv2.medianBlur(gray, 3)
        contrasted = self._apply_clahe(denoised)
        deskewed = self._deskew(contrasted)
        binary = self._binarize(deskewed)
        return deskewed, binary

    # Compatibilité avec l'ancien code
    def enhance(self, image: np.ndarray, return_binary: bool = True) -> np.ndarray:
        gray, binary = self.enhance_both(image)
        return binary if return_binary else gray

    # ---------- Étapes internes ----------

    @staticmethod
    def _upscale(image: np.ndarray, scale: float = 1.5) -> np.ndarray:
        return cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _apply_clahe(gray: np.ndarray,
                     clip_limit: float = 2.0,
                     tile_grid: tuple = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(gray)

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """
        Redresse l'image en ciblant les pixels d'encre (sombres < 128).
        CORRECTIF : l'ancienne version ciblait les pixels clairs (> 0),
        ce qui donnait un angle incorrect sur fond blanc.
        """
        coords = np.column_stack(np.where(gray < 128))
        if coords.size == 0:
            return gray

        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        if abs(angle) < 0.5:
            return gray

        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _binarize(gray: np.ndarray) -> np.ndarray:
        """Otsu + fermeture verticale pour reconnecter les deux-points."""
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)