"""
Enregistrement retroactif des modeles servis dans le registre MLflow.

Execute DANS le conteneur mlflow (aucune dependance sur les venvs locaux) :
  docker cp swift-validator-ia/scripts/register_models_mlflow.py swift-mlflow:/tmp/
  docker cp swift-validator-ia/models.yaml                       swift-mlflow:/tmp/
  docker exec swift-mlflow python /tmp/register_models_mlflow.py

Pour chaque modele du manifest models.yaml : un run MLflow avec ses metriques
d'evaluation (celles du memoire), ses SHA-256 en tags, le manifest en artefact,
puis creation d'une version dans le Model Registry avec l'alias 'production'.
Idempotent : re-executer ne cree pas de doublons de registered models.
"""

from __future__ import annotations

import yaml
import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://localhost:5000"
EXPERIMENT = "swift-validator-models"
MANIFEST = "/tmp/models.yaml"

# Metriques d'evaluation par modele (source : memoire / harnais d'eval).
# Cle = nom dans models.yaml.
MODEL_METRICS: dict[str, dict[str, float]] = {
    "swift-ner":            {"f1_entity": 0.9595, "f1_token": 0.9933},
    "clause-clf":           {"f1_type": 0.929},
    "risk-clf":             {"f1_importance": 0.897},
    "risk-ml-engine":       {"accuracy_calibrated": 0.925, "brier": 0.055,
                             "auc": 0.98, "accuracy_xgboost_raw": 0.944,
                             "accuracy_lightgbm_challenger": 0.941},
    "r2-doc-classifier":    {"accuracy": 0.9972},
    "r2-layoutlm-invoice":  {"f1": 0.993},
    "r2-layoutlm-bl":       {"f1": 0.952},
    "r2-layoutlm-pl":       {"f1": 0.947},
    "r2-layoutlm-universal": {"f1_token": 0.9995},
    # nli-offline / r2-ensemble : pas de metrique dediee -> params seulement
}

MODEL_DESC: dict[str, str] = {
    "swift-ner":            "BERT-NER fine-tune - extraction des entites MT700",
    "clause-clf":           "SentenceTransformer + regression logistique - type de clause (5 classes)",
    "nli-offline":          "Cross-encoder NLI offline - contradictions MT700",
    "risk-clf":             "TF-IDF + regression logistique - importance de clause (3 niveaux)",
    "risk-ml-engine":       "XGBoost calibre (production) + brut/LightGBM (challengers) + IsolationForest + KMeans + SHAP",
    "r2-doc-classifier":    "ResNet-18 transfer learning - type de document commercial",
    "r2-layoutlm-invoice":  "LayoutLMv3 specialise facture (20 champs)",
    "r2-layoutlm-bl":       "LayoutLMv3 specialise connaissement (24 champs)",
    "r2-layoutlm-pl":       "LayoutLMv3 specialise liste de colisage (21 champs)",
    "r2-layoutlm-universal": "LayoutLMv3 universel 87 labels IOB2 (fallback)",
    "r2-ensemble":          "Ensemble TF-IDF + LogReg/RandomForest (classification texte)",
}


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient(TRACKING_URI)

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    local_models: dict = manifest.get("local_models", {})
    hub_models: dict = manifest.get("hub_models", {})
    print(f"Manifest : {len(local_models)} modeles locaux, {len(hub_models)} modeles Hub epingles")

    for name, entry in sorted(local_models.items()):
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("path", entry.get("path", ""))
            mlflow.log_param("inference_device", manifest.get("inference_device", "cpu"))
            mlflow.set_tag("description", MODEL_DESC.get(name, ""))
            mlflow.set_tag("platform_version", "2.1.0")

            total_bytes = 0
            for fname, finfo in (entry.get("files") or {}).items():
                mlflow.set_tag(f"sha256.{fname}", finfo.get("sha256", ""))
                total_bytes += int(finfo.get("size_bytes", 0))
            mlflow.log_param("weights_size_mb", round(total_bytes / 1e6, 1))

            for metric, value in MODEL_METRICS.get(name, {}).items():
                mlflow.log_metric(metric, value)

            mlflow.log_artifact(MANIFEST, artifact_path="manifest")

            # Model Registry : creer (si absent) + nouvelle version + alias production
            try:
                client.create_registered_model(name)
            except Exception:
                pass  # existe deja
            version = client.create_model_version(
                name=name,
                source=f"{run.info.artifact_uri}/manifest",
                run_id=run.info.run_id,
            )
            client.set_registered_model_alias(name, "production", version.version)
            print(f"  [OK] {name} -> version {version.version} (alias production)")

    # Modeles Hub epingles : un run recapitulatif (revisions exactes = preuve de parite)
    with mlflow.start_run(run_name="hub-models-pinned"):
        for full, rev in hub_models.items():
            mlflow.set_tag(f"revision.{full}", rev)
        mlflow.log_artifact(MANIFEST, artifact_path="manifest")
    print("  [OK] revisions Hub epinglees journalisees")

    print("\nTermine. UI : http://localhost:5500")


if __name__ == "__main__":
    main()
