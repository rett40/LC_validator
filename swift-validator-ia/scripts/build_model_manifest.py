"""
Génère models.yaml — manifest de versions des modèles servis (MLOps).

Pour chaque modèle déployé : chemin, fichiers de poids, SHA-256, taille, date.
Le /health de l'API expose ce manifest ; comparer les sha256 entre le dépôt
local et la prod garantit que les DEUX servent exactement les mêmes poids.

Usage :  python scripts/build_model_manifest.py
Ré-exécuter après tout ré-entraînement, puis committer models.yaml.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Modèles effectivement SERVIS (pas les checkpoints intermédiaires).
# nom → (dossier, [fichiers de poids à hasher])
SERVED_MODELS: dict[str, tuple[str, list[str]]] = {
    "swift-ner":            ("models/swift-ner",  ["model.safetensors"]),
    "clause-clf":           ("models/clause-clf", ["model.safetensors", "clf.pkl"]),
    "nli-offline":          ("models/nli",        ["model.safetensors"]),
    "risk-clf":             ("models/risk-clf",   ["risk_clf.pkl", "risk_vectorizer.pkl"]),
    "risk-ml-engine":       ("services/risk_service/models",
                             ["xgboost.pkl", "xgboost_calibrated.pkl", "lightgbm.pkl",
                              "isolation_forest.pkl", "kmeans.pkl", "scaler.pkl",
                              "shap_explainer.pkl"]),
    "r2-doc-classifier":    ("release2/models/doc_classifier", ["best_model.pt"]),
    "r2-layoutlm-universal": ("release2/models/layoutlmv3-finetuned/best/best",
                              ["model.safetensors"]),
    "r2-layoutlm-bl":       ("release2/models/models_layout/best_bl/best_bl",
                             ["model.safetensors"]),
    "r2-layoutlm-invoice":  ("release2/models/models_layout/best_invoice/best_invoice",
                             ["model.safetensors"]),
    "r2-layoutlm-pl":       ("release2/models/models_layout/best_pl/best_pl",
                             ["model.safetensors"]),
    "r2-ensemble":          ("release2/models/ensemble",
                             ["logistic_regression.joblib", "random_forest.joblib",
                              "tfidf_vectorizer.joblib"]),
}


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> int:
    entries: dict[str, dict] = {}
    for name, (rel_dir, weight_files) in SERVED_MODELS.items():
        model_dir = ROOT / rel_dir
        if not model_dir.exists():
            print(f"  ! {name}: dossier absent ({rel_dir}) — ignoré")
            continue
        files = {}
        for wf in weight_files:
            fp = model_dir / wf
            if not fp.exists():
                print(f"  ! {name}: {wf} absent — ignoré")
                continue
            stat = fp.stat()
            print(f"  hashing {name}/{wf} ({stat.st_size / 1e6:.0f} MB)…")
            files[wf] = {
                "sha256": sha256_of(fp),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).strftime("%Y-%m-%d"),
            }
        if files:
            entries[name] = {"path": rel_dir, "files": files}

    # Révisions Hub épinglées (source : config/model_registry.py)
    sys.path.insert(0, str(ROOT))
    from config.model_registry import PINNED_HUB_REVISIONS, INFERENCE_DEVICE

    lines = [
        "# Manifest des modèles servis — généré par scripts/build_model_manifest.py",
        "# NE PAS éditer à la main : ré-exécuter le script après tout ré-entraînement.",
        f"generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"inference_device: {INFERENCE_DEVICE}",
        "hub_models:",
    ]
    for full, rev in sorted(PINNED_HUB_REVISIONS.items()):
        lines.append(f"  {full}: {rev}")
    lines.append("local_models:")
    for name, entry in sorted(entries.items()):
        lines.append(f"  {name}:")
        lines.append(f"    path: {entry['path']}")
        lines.append("    files:")
        for fn, info in entry["files"].items():
            lines.append(f"      {fn}:")
            lines.append(f"        sha256: {info['sha256']}")
            lines.append(f"        size_bytes: {info['size_bytes']}")
            lines.append(f"        modified: '{info['modified']}'")

    out = ROOT / "models.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # ASCII uniquement : la console Windows (cp1252) rejette les glyphes Unicode
    print(f"\n[OK] {out} ecrit ({len(entries)} modeles locaux, "
          f"{len(PINNED_HUB_REVISIONS)} modeles Hub epingles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
