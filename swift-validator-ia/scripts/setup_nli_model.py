"""
setup_nli_model.py
==================
Utility script to prepare an offline NLI model directory under `models/nli/`.

This script is intended to be run in an environment with internet access.
At runtime, the application uses `local_files_only=True` and will not download.

Example:
  python scripts/setup_nli_model.py --model cross-encoder/nli-deberta-v3-xsmall
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="cross-encoder/nli-deberta-v3-xsmall",
        help="HuggingFace model id to download",
    )
    ap.add_argument(
        "--backend",
        choices=("auto", "transformers", "cross-encoder"),
        default="auto",
        help="How to package the model for offline usage.",
    )
    ap.add_argument(
        "--out",
        default="models/nli",
        help="Output directory for offline usage",
    )
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer CrossEncoder packaging for 'cross-encoder/*' models.
    want_ce = args.backend == "cross-encoder" or (
        args.backend == "auto" and str(args.model).lower().startswith("cross-encoder/")
    )

    if want_ce:
        from sentence_transformers import CrossEncoder

        ce = CrossEncoder(args.model)
        ce.save(str(out_dir))
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForSequenceClassification.from_pretrained(args.model)

        tok.save_pretrained(str(out_dir))
        mdl.save_pretrained(str(out_dir))

    print(f"Saved offline NLI model to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

