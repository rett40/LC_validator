#!/usr/bin/env python3
"""
train_clause_models.py
======================
Entraîne le classifieur de TYPE de clause sur embeddings SentenceTransformer.

  TYPE classifier (5 classes) → models/clause-clf/clf.pkl
    Chargé automatiquement par SwiftClauseClassifier (auto-discovery).

Note : le classifieur d'IMPORTANCE est entraîné séparément via
  train_importance_clf.py → models/importance_clf.pkl  (TF-IDF + LR)

Usage :
    cd swift-validator-ia
    python services/swift_nlp/clauses/train_clause_models.py
"""
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

DATASET   = ROOT / "data/data_stati/clause_classification.jsonl"
MODEL_DIR = ROOT / "models"
# Chemin attendu par SwiftClauseClassifier (auto-discovery + model_path)
TYPE_CLF_DIR  = MODEL_DIR / "clause-clf"
TYPE_CLF_PATH = TYPE_CLF_DIR / "clf.pkl"

CLAUSE_TYPES  = [
    "document_condition", "shipment_condition", "payment_condition",
    "compliance_condition", "tolerance_condition",
]
EMBED_MODEL   = "paraphrase-multilingual-MiniLM-L12-v2"


def load_data():
    records = []
    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                t   = r.get("type", "").strip()
                txt = r.get("text", "").strip()
                if t in CLAUSE_TYPES and txt:
                    records.append({"text": txt, "type": t})
            except json.JSONDecodeError:
                pass
    return records


def encode_all(records, model_name):
    from sentence_transformers import SentenceTransformer
    print(f"\n  Chargement modele : {model_name}")
    model = SentenceTransformer(model_name)
    texts = [r["text"] for r in records]
    print(f"  Encodage de {len(texts)} clauses...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True)
    return embeddings


def train_type_classifier(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    print(f"\n  === Classifieur TYPE ({len(set(y))} classes) ===")
    print(f"  Distribution : {dict(Counter(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    clf = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced", solver="lbfgs")

    cv = cross_val_score(clf, X, y, cv=5, scoring="f1_weighted")
    print(f"  CV F1-weighted : {cv.mean():.4f} +/- {cv.std():.4f}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"  Test accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, labels=CLAUSE_TYPES, zero_division=0))
    return clf


def main():
    import joblib

    print("=" * 60)
    print("  Entrainement classifieur TYPE clause (embeddings)")
    print("=" * 60)

    records = load_data()
    print(f"\n  Dataset : {len(records)} exemples")

    X = encode_all(records, EMBED_MODEL)
    y = [r["type"] for r in records]

    clf = train_type_classifier(X, y)

    TYPE_CLF_DIR.mkdir(parents=True, exist_ok=True)
    # Sauvegarder le clf directement (pas dans un dict)
    # _EmbeddingClassifier.classify_supervised() appelle clf.predict_proba() directement
    joblib.dump(clf, TYPE_CLF_PATH)

    print(f"\n  Modele sauvegarde -> {TYPE_CLF_PATH}")
    print("  Charge automatiquement par SwiftClauseClassifier (auto-discovery)")
    print("=" * 60)


if __name__ == "__main__":
    main()
