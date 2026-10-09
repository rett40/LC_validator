"""
Rapport de derive de donnees (data drift) — Evidently.

Compare la distribution des features du modele de risque entre :
  - REFERENCE : la periode d'entrainement (80 % du dataset, 24 000 LC)
  - COURANT   : le dernier lot recu (20 %, 6 000 LC — simule un flux frais)

Produit un rapport HTML autonome, demontrable en soutenance. En production
reelle, 'COURANT' serait remplace par la fenetre glissante des dossiers
recemment analyses : toute derive significative (donnees reelles vs corpus
synthetique d'entrainement) declenchera l'alerte — c'est exactement le
risque n.1 identifie dans les perspectives du memoire.

Usage :
  .mlops_venv/Scripts/python.exe swift-validator-ia/scripts/drift_report_evidently.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "services" / "risk_service" / "data" / "risk_dataset.csv"
OUT_DIR = ROOT / "evaluation" / "drift"
OUT_HTML = OUT_DIR / "drift_report.html"

REFERENCE_FRACTION = 0.8  # 80 % = periode d'entrainement


def main() -> None:
    df = pd.read_csv(DATASET)
    # Cible du modele = true_label ; on ecarte les colonnes de diagnostic
    df = df.drop(columns=[c for c in ("ws_label", "label_mismatch") if c in df.columns])
    df = df.rename(columns={"true_label": "target"})

    split = int(len(df) * REFERENCE_FRACTION)
    reference, current = df.iloc[:split], df.iloc[split:]
    print(f"Reference (entrainement) : {len(reference)} LC | Courant (flux frais) : {len(current)} LC")

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(OUT_HTML))

    # Resume console : nombre de colonnes en derive
    summary = report.as_dict()
    for metric in summary.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            r = metric["result"]
            print(f"Colonnes en derive : {r['number_of_drifted_columns']}/{r['number_of_columns']} "
                  f"(seuil dataset : {r['drift_share']:.0%}) -> drift dataset : {r['dataset_drift']}")
    print(f"\n[OK] Rapport : {OUT_HTML}")


if __name__ == "__main__":
    main()
