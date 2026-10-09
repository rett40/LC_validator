#!/usr/bin/env python3
"""
Script pour entraîner les modèles d'ensemble pour ClassifierService.
- RandomForest sur features métier
- TF-IDF + LogisticRegression sur texte OCR
"""

import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import easyocr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"  # swift-validator-ia/models
ENSEMBLE_DIR = MODEL_DIR / "ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

# Classes
CLASS_NAMES = ["bl", "invoice", "packing_list"]

# Données d'entraînement fictives (à remplacer par vraies données)
def generate_sample_data(num_samples=100):
    """Génère des données d'exemple pour l'entraînement."""
    np.random.seed(42)

    features = []
    labels = []
    texts = []

    for _ in range(num_samples):
        # Features métier : présence et comptage pour chaque classe
        feat = np.random.randint(0, 5, size=9)  # 3 classes * (présence + comptage) + 3 extra
        features.append(feat)

        # Label aléatoire
        label = np.random.choice(CLASS_NAMES)
        labels.append(CLASS_NAMES.index(label))

        # Texte OCR simulé
        if label == "bl":
            text = "bill of lading shipper consignee vessel port cargo"
        elif label == "invoice":
            text = "invoice amount total tax payment due date"
        else:
            text = "packing list item quantity description weight"
        texts.append(text + " " + " ".join(np.random.choice(["word1", "word2", "word3"], 10)))

    return np.array(features), np.array(labels), texts

def train_random_forest(X, y):
    """Entraîne RandomForest sur features métier."""
    log.info("Entraînement RandomForest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    log.info("Rapport RandomForest:\n%s", classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return rf

def train_tfidf_lr(texts, y):
    """Entraîne TF-IDF + LogisticRegression sur texte OCR."""
    log.info("Entraînement TF-IDF + LogisticRegression...")
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_tfidf, y_train)

    y_pred = lr.predict(X_test_tfidf)
    log.info("Rapport TF-IDF + LR:\n%s", classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return tfidf, lr

def main():
    log.info("Génération des données d'entraînement...")
    X, y, texts = generate_sample_data(500)  # 500 échantillons

    # Entraîner RandomForest
    rf_model = train_random_forest(X, y)
    joblib.dump(rf_model, ENSEMBLE_DIR / "random_forest.joblib")
    log.info("RandomForest sauvegardé à %s", ENSEMBLE_DIR / "random_forest.joblib")

    # Entraîner TF-IDF + LR
    tfidf, lr_model = train_tfidf_lr(texts, y)
    joblib.dump(tfidf, ENSEMBLE_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(lr_model, ENSEMBLE_DIR / "logistic_regression.joblib")
    log.info("TF-IDF et LogisticRegression sauvegardés")

    log.info("Entraînement terminé. Modèles disponibles dans %s", ENSEMBLE_DIR)

if __name__ == "__main__":
    main()