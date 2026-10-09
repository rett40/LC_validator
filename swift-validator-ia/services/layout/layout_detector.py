import re
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

try:  # lazy import to keep optional dependencies light
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except Exception:  # pragma: no cover - doctr may be absent in dev env
    DocumentFile = None
    ocr_predictor = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class LayoutDetector:
    """
    Detects text layout and SWIFT field tags on a page.
    - docTR handles fine-grained text boxes and provides confidences.
    - (Optional) YOLOv8 can be used to detect coarse blocks/tables if weights are supplied.
    """

    SWIFT_TAG_RE = re.compile(
        r"\b(20|21|23|27|31[ACD]?|32[AB]?|33[AB]?|40E|50[AB]?"
        r"|52[AB]?|53[AB]?|57[ABD]?|59[AB]?|70|71[ABD]?)[: ]",
        flags=re.IGNORECASE,
    )

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        assume_straight_pages: bool = True,
        yolo_weights: Optional[str] = None,
    ) -> None:
        self._init_error = None
        self.ocr_model = None
        if ocr_predictor is not None and DocumentFile is not None:
            try:
                self.ocr_model = ocr_predictor(
                    det_arch=det_arch,
                    reco_arch=reco_arch,
                    pretrained=True,
                    assume_straight_pages=assume_straight_pages,
                )
            except Exception as exc:  # pragma: no cover - network/download issues
                self._init_error = str(exc)
        else:
            self._init_error = "docTR is not installed"

        self.yolo = None
        if yolo_weights and YOLO is not None:
            try:
                self.yolo = YOLO(yolo_weights)
            except Exception as exc:  # pragma: no cover
                self._init_error = (self._init_error or "") + f"; YOLO load failed: {exc}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Returns a dict with detected lines, swift tags, and optional block/table boxes.
        Each line contains bbox (normalized x0,y0,x1,y1), text, and confidence.
        """
        if image is None:
            raise ValueError("detect: image is None")
        if self.ocr_model is None:
            raise RuntimeError(f"docTR model unavailable: {self._init_error}")

        # docTR expects RGB PIL images; DocumentFile.from_images handles np arrays.
        doc = DocumentFile.from_images([image[:, :, ::-1] if image.ndim == 3 else image])
        result = self.ocr_model(doc)
        export = result.export()
        page = export["pages"][0]

        lines: List[Dict[str, Any]] = []
        swift_tags: List[Dict[str, Any]] = []

        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(word["value"] for word in line.get("words", []))
                if not text:
                    continue
                confs = [word.get("confidence", 0.0) for word in line.get("words", [])]
                line_conf = float(np.mean(confs)) if confs else 0.0
                entry = {
                    "bbox": line.get("geometry"),
                    "text": text,
                    "confidence": round(line_conf, 3),
                }
                lines.append(entry)
                if self.SWIFT_TAG_RE.search(text):
                    swift_tags.append(entry)

        yolo_blocks: List[Dict[str, Any]] = []
        if self.yolo:
            yolo_blocks = self._detect_blocks_with_yolo(image)

        return {
            "dimensions": page.get("dimensions"),
            "lines": lines,
            "swift_tags": swift_tags,
            "blocks": yolo_blocks,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _detect_blocks_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Optional coarse block detection with YOLOv8 if weights were provided."""
        results = self.yolo.predict(source=image, verbose=False)
        blocks: List[Dict[str, Any]] = []
        for r in results:
            if not hasattr(r, "boxes"):
                continue
            for b in r.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0]) if b.cls is not None else None
                blocks.append(
                    {
                        "bbox": (x0, y0, x1, y1),
                        "confidence": round(conf, 3),
                        "class_id": cls,
                    }
                )
        return blocks
