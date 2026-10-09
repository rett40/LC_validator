# Évaluation des modèles — swift-validator-ia

## Structure

```
evaluation/
├── risk_service/
│   ├── evaluate_risk_models.py   ← évaluation XGBoost / LightGBM / IsolationForest
│   └── reports/                  ← générés automatiquement
│       ├── report.json
│       ├── confusion_xgb_calibrated.png
│       ├── confusion_lightgbm.png
│       ├── roc_xgb_calibrated.png
│       ├── roc_lightgbm.png
│       ├── calibration_curve.png
│       ├── shap_importance.png
│       ├── anomaly_score_distribution.png
│       └── class_distribution.png
└── clauses/
    ├── evaluate_clause_classifier.py  ← évaluation SwiftClauseClassifier v2.0
    └── reports/
        ├── report.json
        ├── confusion_matrix.png
        ├── f1_per_class.png
        ├── method_distribution.png
        ├── confidence_distribution.png
        └── dataset_distribution.png
```

## Utilisation

```bash
cd swift-validator-ia

# Évaluer les modèles risk_service
python evaluation/risk_service/evaluate_risk_models.py

# Évaluer le classifieur de clauses
python evaluation/clauses/evaluate_clause_classifier.py
```

## Métriques produites

### risk_service
| Métrique | Description |
|----------|-------------|
| Accuracy | % de prédictions correctes |
| F1 (weighted/macro) | Moyenne F1 pondérée et macro |
| Precision / Recall | Par classe (LOW / MEDIUM / HIGH) |
| Matrice de confusion | Erreurs de classification |
| Courbes ROC + AUC | One-vs-rest pour chaque classe |
| Reliability diagram | Calibration des probabilités |
| SHAP importance | Top 10 features par classe |
| Anomaly score | Distribution IsolationForest par classe |
| Brier score | Qualité des probabilités calibrées |

### clauses
| Métrique | Description |
|----------|-------------|
| Accuracy | % de types correctement identifiés |
| F1 (weighted/macro) | Moyenne F1 pondérée et macro |
| Matrice de confusion | Confusions entre types de clauses |
| Méthodes utilisées | rule / scoring / embedding / supervised |
| Distribution confiance | Correct vs incorrect par niveau de confiance |
| Analyse erreurs | Top confusions avec exemples |
