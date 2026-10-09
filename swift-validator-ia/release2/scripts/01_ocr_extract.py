"""
01_ocr_extract.py
=================
Étape 1 — Extraction de tokens + bounding boxes depuis les PDFs.

Stratégie :
  [NATIF]   PDF vectoriel (ReportLab) → PyMuPDF get_text('words')
              → texte exact + bboxes précises, pas d'OCR nécessaire
  [FALLBACK] PDF scanné / image → EasyOCR readtext(detail=1)
              → utilisé si PyMuPDF ne trouve pas de texte

Sortie par document :
  data/processed/{doc_type}/{doc_id}_ocr.json
  data/processed/{doc_type}/{doc_id}_page{n}.png  ← rendu 150 DPI pour LayoutLMv3

Format JSON :
  {
    "doc_id": "bl_cmacgm_0003_CMAUJKWB660781151",
    "doc_type": "bl",
    "source_pdf": "raw/bl/pdfs/...",
    "extraction_mode": "native" | "ocr",
    "n_pages": 1,
    "overall_confidence": 1.0,
    "pages": [{
      "page": 1,
      "image_path": "processed/bl/bl_cmacgm_0003_page1.png",
      "width_pts": 595.0,
      "height_pts": 842.0,
      "width_px": 1240,
      "height_px": 1754,
      "n_tokens": 145,
      "tokens": [{
        "text": "BILL",
        "bbox": [470, 46, 497, 66],   ← [x0,y0,x1,y1] normalisé 0-1000
        "confidence": 1.0
      }, ...]
    }]
  }
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import fitz                    # PyMuPDF
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT  = RELEASE2_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = RELEASE2_ROOT / "data"
RAW_ROOT  = DATA_ROOT / "raw"
PROC_ROOT = DATA_ROOT / "processed"

DOC_TYPES = {
    "bl":           RAW_ROOT / "bl",
    "invoice":      RAW_ROOT / "invoice",
    "packing_list": RAW_ROOT / "packing_list",
}

# ── Config ────────────────────────────────────────────────────────────────────

RENDER_DPI     = 150    # rendu image pour LayoutLMv3 (150 DPI = bon équilibre taille/qualité)
BBOX_SCALE     = 1000   # convention LayoutLMv3 : coordonnées normalisées 0-1000
NATIVE_MIN_WORDS = 10   # seuil pour considérer l'extraction native comme réussie
SCAN_SIM_PROB  = 0.60   # fraction des docs dégradés avec simulation scan

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            RELEASE2_ROOT / "logs" / "01_ocr_extract.log", encoding="utf-8"
        ),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction native (PDF vectoriel)
# ─────────────────────────────────────────────────────────────────────────────

def extract_native_page(page: fitz.Page, img_w: int, img_h: int) -> list[dict]:
    """
    Extrait les mots depuis une page PDF vectorielle.
    Retourne une liste de tokens avec bbox normalisée en [0-1000].

    PyMuPDF get_text('words') retourne :
      (x0, y0, x1, y1, text, block_no, line_no, word_no)
    Les coordonnées sont en PDF points (origine en haut à gauche).
    """
    rect   = page.rect
    pw, ph = rect.width, rect.height       # dimensions en points PDF

    tokens = []
    for word_tuple in page.get_text("words"):
        x0, y0, x1, y1, text = word_tuple[:5]
        text = text.strip()
        if not text:
            continue

        # Normalisation : points PDF → pixels de l'image rendue → 0-1000
        # scale = img_w / pw  (identique pour x) et img_h / ph (pour y)
        sx, sy = img_w / pw, img_h / ph

        def nx(v): return max(0, min(BBOX_SCALE, int(v * sx / img_w * BBOX_SCALE)))
        def ny(v): return max(0, min(BBOX_SCALE, int(v * sy / img_h * BBOX_SCALE)))

        tokens.append({
            "text":       text,
            "bbox":       [nx(x0), ny(y0), nx(x1), ny(y1)],
            "confidence": 1.0,
        })

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Fallback OCR (PDF scanné / image)
# ─────────────────────────────────────────────────────────────────────────────

def _get_easyocr_reader(gpu: bool = False):
    """Instanciation lazy de l'EasyOCR reader."""
    import easyocr
    from services.ocr.image_enhancer import ImageEnhancer
    return easyocr.Reader(["en", "fr"], gpu=gpu, verbose=False), ImageEnhancer()


def extract_ocr_page(
    page: fitz.Page,
    img_bgr: np.ndarray,
    reader,
    enhancer,
) -> list[dict]:
    """
    Fallback OCR via EasyOCR pour les pages sans texte natif.
    """
    h, w = img_bgr.shape[:2]
    gray, _ = enhancer.enhance_both(img_bgr)
    detections = reader.readtext(gray, detail=1, paragraph=False)

    tokens = []
    for polygon, text, conf in detections:
        text = text.strip()
        if not text or conf < 0.20:
            continue

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

        def nx(v): return max(0, min(BBOX_SCALE, int(v / w * BBOX_SCALE)))
        def ny(v): return max(0, min(BBOX_SCALE, int(v / h * BBOX_SCALE)))

        words = text.split()
        if len(words) == 1:
            tokens.append({"text": words[0], "bbox": [nx(x0), ny(y0), nx(x1), ny(y1)],
                           "confidence": round(float(conf), 3)})
        else:
            total_chars = sum(len(ww) for ww in words)
            bw          = x1 - x0
            cx          = x0
            for ww in words:
                ratio = len(ww) / total_chars if total_chars else 1 / len(words)
                wx1   = cx + bw * ratio
                tokens.append({"text": ww,
                                "bbox": [nx(cx), ny(y0), nx(wx1), ny(y1)],
                                "confidence": round(float(conf), 3)})
                cx = wx1

    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Simulation de scan (dégradation d'image pour entraînement robuste)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_scan(img_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simule l'apparence d'un document scanné sans modifier les positions de tokens.
    Applique uniquement des dégradations non-spatiales (sauf légère rotation ±1.5°
    dont le décalage bbox reste dans la tolérance BBOX_CENTER_MARGIN=8).
    """
    h, w = img_bgr.shape[:2]

    # 1. Légère rotation de scan (±1.5°) — décalage bbox max ~6 unités sur 1000
    angle = rng.uniform(-1.5, 1.5)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # 2. Légère teinte papier vieilli (jaune/gris)
    tint_type = rng.choice(["none", "yellow", "gray"], p=[0.4, 0.35, 0.25])
    if tint_type == "yellow":
        overlay = np.full_like(img_bgr, (210, 220, 235))  # BGR légèrement jaunâtre
        alpha = rng.uniform(0.04, 0.12)
        img_bgr = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
    elif tint_type == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 3. Bruit gaussien (simule grain de scanner)
    sigma = rng.uniform(1.5, 6.0)
    noise = rng.normal(0, sigma, img_bgr.shape).astype(np.int16)
    img_bgr = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 4. Variation de contraste/luminosité
    alpha_c = rng.uniform(0.88, 1.12)   # contraste
    beta_c  = rng.uniform(-12.0, 12.0)  # luminosité
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha_c + beta_c, 0, 255).astype(np.uint8)

    # 5. Flou léger (50% du temps)
    if rng.random() < 0.5:
        ksize = int(rng.choice([3, 3, 3, 5]))
        img_bgr = cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    # 6. Artefacts de compression JPEG
    quality = int(rng.uniform(72, 94))
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'un PDF complet
# ─────────────────────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: Path,
    doc_type: str,
    out_dir: Path,
    ocr_reader=None,
    ocr_enhancer=None,
    scan_sim: bool = False,
    rng: np.random.Generator | None = None,
) -> dict | None:
    """
    Traite un PDF et sauvegarde le JSON + images.
    Retourne None si déjà traité (mode resume).
    """
    doc_id   = pdf_path.stem
    out_json = out_dir / f"{doc_id}_ocr.json"

    if out_json.exists():
        return None

    try:
        # Résolution des chemins absolus (évite les erreurs relative_to)
        pdf_path = pdf_path.resolve()
        out_dir  = out_dir.resolve()

        doc        = fitz.open(str(pdf_path))
        mat        = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pages_data = []
        modes_used = []

        for page_num_0 in range(len(doc)):
            page     = doc[page_num_0]
            page_num = page_num_0 + 1

            # Rendu image (toujours nécessaire pour LayoutLMv3)
            pix     = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bgr = cv2.cvtColor(
                np.frombuffer(pix.samples, dtype=np.uint8)
                  .reshape(pix.height, pix.width, 3),
                cv2.COLOR_RGB2BGR
            )
            img_h, img_w = img_bgr.shape[:2]

            # Sauvegarde image (avec simulation scan si activée)
            img_filename = f"{doc_id}_page{page_num}.png"
            img_path     = out_dir / img_filename
            if scan_sim and rng is not None and rng.random() < SCAN_SIM_PROB:
                img_save = _simulate_scan(img_bgr, rng)
            else:
                img_save = img_bgr
            cv2.imwrite(str(img_path), img_save)

            # Extraction native
            tokens = extract_native_page(page, img_w, img_h)
            mode   = "native"

            # Fallback OCR si pas assez de mots natifs
            if len(tokens) < NATIVE_MIN_WORDS:
                if ocr_reader is None:
                    log.warning("[%s] p.%d : texte natif insuffisant (%d mots) "
                                "et EasyOCR non initialisé — données partielles",
                                doc_id, page_num, len(tokens))
                else:
                    tokens = extract_ocr_page(page, img_bgr, ocr_reader, ocr_enhancer)
                    mode   = "ocr"

            modes_used.append(mode)
            pages_data.append({
                "page":        page_num,
                "image_path":  str(img_path.relative_to(DATA_ROOT)),
                "width_pts":   round(page.rect.width, 2),
                "height_pts":  round(page.rect.height, 2),
                "width_px":    img_w,
                "height_px":   img_h,
                "n_tokens":    len(tokens),
                "tokens":      tokens,
            })

        doc.close()

        # Mode global : "native" si toutes les pages sont natives
        extraction_mode   = "native" if all(m == "native" for m in modes_used) else "ocr"
        all_confs         = [t["confidence"] for pg in pages_data for t in pg["tokens"]]
        overall_confidence = round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.0

        result = {
            "doc_id":             doc_id,
            "doc_type":           doc_type,
            "source_pdf":         str(pdf_path.relative_to(DATA_ROOT)),
            "extraction_mode":    extraction_mode,
            "n_pages":            len(pages_data),
            "overall_confidence": overall_confidence,
            "pages":              pages_data,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    except Exception as exc:
        log.error("Erreur sur %s : %s\n%s", pdf_path.name, exc, traceback.format_exc())
        return {"doc_id": pdf_path.stem, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction(doc_types: list[str], gpu: bool, workers: int,
                   scan_sim: bool = False) -> dict:
    stats = {"total": 0, "processed": 0, "skipped": 0, "errors": 0,
             "native": 0, "ocr_fallback": 0}

    ocr_reader = ocr_enhancer = None
    rng = np.random.default_rng(seed=None)  # non-déterministe intentionnel

    if scan_sim:
        log.info("Simulation scan ACTIVÉE (%.0f%% des images dégradées)", SCAN_SIM_PROB * 100)

    for doc_type in doc_types:
        out_dir   = PROC_ROOT / doc_type
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = sorted((DOC_TYPES[doc_type] / "pdfs").glob("*.pdf"))
        stats["total"] += len(pdf_files)

        log.info("=" * 60)
        log.info("[%s] %d PDFs → %s", doc_type.upper(), len(pdf_files), out_dir)

        for pdf_path in tqdm(pdf_files, desc=f"  {doc_type}", unit="pdf"):
            result = process_pdf(pdf_path, doc_type, out_dir,
                                 ocr_reader, ocr_enhancer,
                                 scan_sim=scan_sim, rng=rng)
            if result is None:
                stats["skipped"] += 1
            elif "error" in result:
                stats["errors"] += 1
            else:
                stats["processed"] += 1
                if result.get("extraction_mode") == "native":
                    stats["native"] += 1
                else:
                    stats["ocr_fallback"] += 1

        done = len(list(out_dir.glob("*_ocr.json")))
        log.info("[%s] %d JSON générés", doc_type.upper(), done)

    return stats


def print_summary(stats: dict) -> None:
    log.info("")
    log.info("╔═══════════════════════════════════════╗")
    log.info("║   RÉSUMÉ — 01_ocr_extract             ║")
    log.info("╠═══════════════════════════════════════╣")
    log.info("║  Total PDFs      : %4d               ║", stats["total"])
    log.info("║  Traités         : %4d               ║", stats["processed"])
    log.info("║    dont natifs   : %4d               ║", stats["native"])
    log.info("║    dont OCR      : %4d               ║", stats["ocr_fallback"])
    log.info("║  Déjà traités    : %4d  (skip)        ║", stats["skipped"])
    log.info("║  Erreurs         : %4d               ║", stats["errors"])
    log.info("╚═══════════════════════════════════════╝")

    # Qualité : échantillon 50 docs
    all_jsons = list(PROC_ROOT.rglob("*_ocr.json"))
    if all_jsons:
        sample   = all_jsons[:50]
        n_tokens = []
        for p in sample:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                for pg in d.get("pages", []):
                    n_tokens.append(pg.get("n_tokens", 0))
            except Exception:
                pass
        if n_tokens:
            avg = sum(n_tokens) / len(n_tokens)
            log.info("Tokens moyens / page (échantillon %d) : %.0f", len(sample), avg)
            if avg < 20:
                log.warning("⚠  Tokens/page très bas — vérifier l'extraction sur quelques docs")
            else:
                log.info("✓  Extraction semble correcte")

    log.info("")
    log.info("Prochaine étape : python scripts/02_annotate_bio.py")


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="01_ocr_extract — PDF → tokens + bboxes (natif + fallback OCR)"
    )
    parser.add_argument(
        "--doc-types", nargs="+",
        choices=["bl", "invoice", "packing_list"],
        default=["bl", "invoice", "packing_list"],
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Workers parallèles (1 recommandé — PyMuPDF est déjà rapide)"
    )
    parser.add_argument(
        "--scan-sim", action="store_true",
        help=(
            f"Simule l'apparence de documents scannés sur {SCAN_SIM_PROB:.0%} des images "
            "(bruit, JPEG, légère rotation, teinte papier). "
            "Recommandé pour améliorer la robustesse sur vrais documents."
        )
    )
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Script 01 — PDF → tokens + bboxes                 ║")
    log.info("║  Mode : natif PyMuPDF  (fallback EasyOCR)          ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("  Scan-sim : %s", "ON" if args.scan_sim else "OFF")

    for dt in args.doc_types:
        n = len(list((DOC_TYPES[dt] / "pdfs").glob("*.pdf")))
        log.info("  %-15s : %d PDFs", dt, n)

    stats = run_extraction(args.doc_types, args.gpu, args.workers,
                           scan_sim=args.scan_sim)
    print_summary(stats)


if __name__ == "__main__":
    main()
