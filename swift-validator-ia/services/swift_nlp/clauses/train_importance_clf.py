#!/usr/bin/env python3
"""
train_importance_clf.py
=======================
Entraine un classifieur TF-IDF + LogisticRegression pour predire
l'importance des clauses SWIFT (high / medium / normal).

Le modele est sauvegarde dans models/importance_clf.pkl et charge
automatiquement par SwiftClauseClassifier.

Usage :
    cd swift-validator-ia
    python services/swift_nlp/clauses/train_importance_clf.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

DATASET  = ROOT / "data/data_stati/clause_classification.jsonl"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "importance_clf.pkl"

IMPORTANCE_LABELS = ["high", "medium", "normal"]


def load_data():
    records = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                imp  = r.get("importance", "").strip()
                text = r.get("text", "").strip()
                ctype = r.get("type", "").strip()
                if imp in IMPORTANCE_LABELS and text and ctype:
                    records.append({"text": text, "type": ctype, "importance": imp})
            except json.JSONDecodeError:
                pass
    return records


def build_features(records):
    """Concatene type + texte pour donner plus de contexte au TF-IDF."""
    return [f"{r['type']} {r['text']}" for r in records]


def main():
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
    except ImportError as e:
        print(f"Dependances manquantes : {e}")
        sys.exit(1)

    print("=" * 60)
    print("  Entrainement classifieur d'importance des clauses")
    print("=" * 60)

    records = load_data()
    print(f"\n  Dataset : {len(records)} exemples")
    print(f"  Distribution : {dict(Counter(r['importance'] for r in records))}")

    features = build_features(records)
    labels   = [r["importance"] for r in records]

    # Split 80/20 stratifie
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=42
    )

    # Pipeline TF-IDF (unigrams + bigrams) + LR calibree
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=1,
            sublinear_tf=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w[\w\-]+\b",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipeline, features, labels, cv=5, scoring="f1_weighted")
    print(f"\n  Cross-validation (cv=5) F1-weighted : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Entrainement final
    pipeline.fit(X_train, y_train)

    # Evaluation sur test set
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Test accuracy : {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, labels=IMPORTANCE_LABELS, zero_division=0)}")

    # Sauvegarde
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  Modele sauvegarde -> {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
