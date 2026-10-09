"""
evaluate_models.py — Evaluation des modèles NLP SWIFT
=====================================================

Calcule les métriques :
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion matrix

Usage :
    python scripts/evaluate_models.py \
        --dataset data/data_stati/training_samples.jsonl \
        --model models/clause-clf/model.joblib \
        --type clause
"""

import argparse
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from sentence_transformers import SentenceTransformer


def load_dataset(path):

    import pandas as pd

    df = pd.read_json(path, lines=True)

    if "text" not in df.columns:
        raise ValueError("Colonne 'text' manquante")

    # détection automatique du label
    if "label" in df.columns:
        labels = df["label"]
    elif "type" in df.columns:
        labels = df["type"]
    elif "clause_type" in df.columns:
        labels = df["clause_type"]
    else:
        raise ValueError("Aucune colonne label trouvée")

    return df["text"].tolist(), labels.tolist()
    """Charge dataset JSONL"""
    df = pd.read_json(path, lines=True)

    if "text" not in df.columns:
        raise ValueError("Colonne 'text' manquante")

    if "label" not in df.columns:
        raise ValueError("Colonne 'label' manquante")

    return df["text"].tolist(), df["label"].tolist()


def evaluate_clause_model(model_path, texts, labels, embedding_model):
    """Evaluation du classifieur de clauses"""

    print("\nChargement modèle clause...")
    model = joblib.load(model_path)

    print("Chargement SentenceTransformer...")
    embedder = SentenceTransformer(embedding_model)

    print("Création embeddings...")
    embeddings = embedder.encode(texts)

    print("Prédictions...")
    y_pred = model.predict(embeddings)

    return compute_metrics(labels, y_pred)


def evaluate_risk_model(model_path, texts, labels):
    """Evaluation du modèle de risque"""

    print("\nChargement modèle risk...")
    model = joblib.load(model_path)

    print("Prédictions...")
    y_pred = model.predict(texts)

    return compute_metrics(labels, y_pred)


def compute_metrics(y_true, y_pred):
    """Calcule toutes les métriques"""

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print("\n==============================")
    print("RESULTATS EVALUATION")
    print("==============================")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report :\n")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix :\n")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    parser = argparse.ArgumentParser(description="Evaluation modèle SWIFT NLP")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset JSONL"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Chemin du modèle"
    )

    parser.add_argument(
        "--type",
        choices=["clause", "risk"],
        required=True,
        help="Type de modèle"
    )

    parser.add_argument(
        "--embedding-model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model"
    )

    args = parser.parse_args()

    print("Chargement dataset...")
    texts, labels = load_dataset(args.dataset)

    print("Nombre d'échantillons :", len(texts))

    if args.type == "clause":
        evaluate_clause_model(
            args.model,
            texts,
            labels,
            args.embedding_model
        )

    elif args.type == "risk":
        evaluate_risk_model(
            args.model,
            texts,
            labels
        )


if __name__ == "__main__":
    main()