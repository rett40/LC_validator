"""
00_generate_dataset.py
======================
Étape 0 — Génération du dataset synthétique complet.

Enchaîne les trois générateurs dans le bon ordre et recopie les PDFs dans
les chemins attendus par 01_ocr_extract.py :

  data/raw/bl/pdfs/*.pdf           ← Bill of Lading (6 carriers × 100 = 600)
  data/raw/invoice/pdfs/*.pdf      ← Commercial Invoice (5 styles × 120 = 600)
  data/raw/packing_list/pdfs/*.pdf ← Packing List (5 styles × 120 = 600)

Usage :
  python scripts/00_generate_dataset.py
  python scripts/00_generate_dataset.py --count 300 --seed 7
  python scripts/00_generate_dataset.py --clean      # supprime l'ancien dataset d'abord
  python scripts/00_generate_dataset.py --bl-only    # régénère uniquement les BL
"""

from __future__ import annotations

import sys
import logging
import argparse
import shutil
import subprocess
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = RELEASE2_ROOT / "data"
RAW_ROOT      = DATA_ROOT / "raw"
GEN_DIR       = RELEASE2_ROOT / "document_generator"
LOGS_DIR      = RELEASE2_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Répertoire de travail des générateurs (dataset.json, pdfs/, invoices/, …)
WORKSPACE     = RAW_ROOT / "bl_workspace"

# Destinations finales pour script 01 (PDFs) et script 02 (annotations)
BL_PDF_DIR    = RAW_ROOT / "bl"           / "pdfs"
INV_PDF_DIR   = RAW_ROOT / "invoice"      / "pdfs"
PL_PDF_DIR    = RAW_ROOT / "packing_list" / "pdfs"

BL_ANN_DIR    = RAW_ROOT / "bl"           / "annotations"
INV_ANN_DIR   = RAW_ROOT / "invoice"      / "annotations"
PL_ANN_DIR    = RAW_ROOT / "packing_list" / "annotations"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "00_generate_dataset.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: Path) -> None:
    log.info("$ %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        log.error("Commande échouée (code %d) : %s", result.returncode, cmd)
        sys.exit(result.returncode)


def _copy_files(src_dir: Path, dst_dir: Path, pattern: str, label: str) -> int:
    """Copie tous les fichiers matchant pattern de src_dir → dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob(pattern)) if src_dir.exists() else []
    for f in files:
        shutil.copy2(f, dst_dir / f.name)
    log.info("[%s] %d fichiers (%s) copiés → %s", label, len(files), pattern, dst_dir)
    return len(files)


def _copy_pdfs(src_dir: Path, dst_dir: Path, label: str) -> int:
    return _copy_files(src_dir, dst_dir, "*.pdf", label)


def _copy_annotations(src_dir: Path, dst_dir: Path, label: str) -> int:
    return _copy_files(src_dir, dst_dir, "*.json", label)


def _clean_dir(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
        log.info("Supprimé : %s", directory)


# ─────────────────────────────────────────────────────────────────────────────
# Étapes de génération
# ─────────────────────────────────────────────────────────────────────────────

def step_bl(count_per_layout: int, seed: int) -> None:
    total = count_per_layout * 22
    log.info("━━━ ÉTAPE 1/3 — Bill of Lading (%d layouts × %d = %d docs) ━━━",
             22, count_per_layout, total)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable, str(GEN_DIR / "generate_bl.py"),
            "--count-per-layout", str(count_per_layout),
            "--output",           str(WORKSPACE),
            "--parametric",
            "--seed",             str(seed),
        ],
        cwd=GEN_DIR,
    )
    _copy_pdfs(WORKSPACE / "pdfs", BL_PDF_DIR, "BL")
    _copy_annotations(WORKSPACE / "annotations", BL_ANN_DIR, "BL-ANN")


def step_invoice(count_per_layout: int, seed: int) -> None:
    log.info("━━━ ÉTAPE 2/3 — Commercial Invoice (21 layouts × %d = %d docs) ━━━",
             count_per_layout, 21 * count_per_layout)
    _run(
        [
            sys.executable, str(GEN_DIR / "generate_invoice.py"),
            "--input",             str(WORKSPACE),
            "--output",            str(WORKSPACE),
            "--discrepancy-rate",  "0.27",
            "--count-per-layout",  str(count_per_layout),
            "--parametric",
            "--seed",              str(seed),
        ],
        cwd=GEN_DIR,
    )
    _copy_pdfs(WORKSPACE / "invoices", INV_PDF_DIR, "INVOICE")
    _copy_annotations(WORKSPACE / "invoices" / "annotations", INV_ANN_DIR, "INV-ANN")


def step_packing_list(count_per_layout: int, seed: int) -> None:
    log.info("━━━ ÉTAPE 3/3 — Packing List (21 layouts × %d = %d docs) ━━━",
             count_per_layout, 21 * count_per_layout)
    _run(
        [
            sys.executable, str(GEN_DIR / "generate_packing_list.py"),
            "--input",             str(WORKSPACE),
            "--output",            str(WORKSPACE),
            "--discrepancy-rate",  "0.27",
            "--count-per-layout",  str(count_per_layout),
            "--parametric",
            "--seed",              str(seed),
        ],
        cwd=GEN_DIR,
    )
    _copy_pdfs(WORKSPACE / "packing_lists", PL_PDF_DIR, "PL")
    _copy_annotations(WORKSPACE / "packing_lists" / "annotations", PL_ANN_DIR, "PL-ANN")


# ─────────────────────────────────────────────────────────────────────────────
# Vérification finale
# ─────────────────────────────────────────────────────────────────────────────

def verify() -> bool:
    ok = True
    for label, pdf_dir in [("BL", BL_PDF_DIR), ("INVOICE", INV_PDF_DIR), ("PL", PL_PDF_DIR)]:
        n = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
        status = "OK" if n > 0 else "VIDE"
        log.info("  [%s] %-12s : %d PDFs", status, label, n)
        if n == 0:
            ok = False
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="00_generate_dataset — génération du dataset synthétique complet (mode paramétrique)"
    )
    parser.add_argument(
        "--count-per-layout", type=int, default=50,
        help="Docs par layout : BL=22 layouts, Invoice=21, PL=21 (total ≈ 50×64=3200)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Graine aléatoire (0 = sans graine fixe pour maximiser la diversité)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Supprime les anciens PDFs bruts et le workspace avant de régénérer",
    )
    parser.add_argument(
        "--bl-only", action="store_true",
        help="Régénère uniquement les B/L (sans Invoice ni Packing List)",
    )
    parser.add_argument(
        "--invoice-only", action="store_true",
        help="Régénère uniquement les Invoices (workspace BL doit exister)",
    )
    parser.add_argument(
        "--pl-only", action="store_true",
        help="Régénère uniquement les Packing Lists (workspace BL doit exister)",
    )
    args = parser.parse_args()

    cpl = args.count_per_layout
    total_expected = cpl * 22 + cpl * 21 + cpl * 21

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Script 00 — Génération dataset synthétique         ║")
    log.info("╠══════════════════════════════════════════════════════╣")
    log.info("║  Count/layout : %-36d║", cpl)
    log.info("║  Total docs   : %-36d║", total_expected)
    log.info("║  Seed         : %-36d║", args.seed)
    log.info("║  Clean        : %-36s║", str(args.clean))
    log.info("╚══════════════════════════════════════════════════════╝")

    if args.clean:
        log.info("Nettoyage des anciens fichiers...")
        _clean_dir(WORKSPACE)
        _clean_dir(BL_PDF_DIR)
        _clean_dir(INV_PDF_DIR)
        _clean_dir(PL_PDF_DIR)
        _clean_dir(BL_ANN_DIR)
        _clean_dir(INV_ANN_DIR)
        _clean_dir(PL_ANN_DIR)

    only_flag = args.bl_only or args.invoice_only or args.pl_only

    if not only_flag or args.bl_only:
        step_bl(cpl, args.seed)

    if not only_flag or args.invoice_only:
        if not (WORKSPACE / "dataset.json").exists():
            log.error("dataset.json introuvable dans %s — exécuter sans --invoice-only d'abord", WORKSPACE)
            sys.exit(1)
        step_invoice(cpl, args.seed)

    if not only_flag or args.pl_only:
        if not (WORKSPACE / "dataset.json").exists():
            log.error("dataset.json introuvable dans %s — exécuter sans --pl-only d'abord", WORKSPACE)
            sys.exit(1)
        step_packing_list(cpl, args.seed)

    log.info("")
    log.info("━━━ VÉRIFICATION FINALE ━━━")
    ok = verify()

    if ok:
        log.info("")
        log.info("Dataset prêt. Prochaine étape :")
        log.info("  python scripts/01_ocr_extract.py")
    else:
        log.error("Certains répertoires sont vides — vérifier les erreurs ci-dessus.")
        sys.exit(1)


if __name__ == "__main__":
    main()
