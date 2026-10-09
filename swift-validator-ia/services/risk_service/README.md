# risk_service — MT700 Risk ML Pipeline

ML pipeline for automated risk scoring of SWIFT MT700 Letters of Credit.
Produces a **LOW / MEDIUM / HIGH** risk label with calibrated probabilities,
SHAP-based explanations, anomaly detection, and UCP 600 hard-rule overrides.

---

## Architecture

```
risk_service/
├── risk_feature_schema.py       ← Single source of truth: 15 features, constants
├── risk_engine.py               ← Inference engine (RiskMLService, UCPRuleEngine, SHAP)
├── scripts/
│   ├── risk_dataset_generator.py  ← Synthetic training data (30 000 LCs)
│   └── train_models.py            ← Full training pipeline (9 models)
├── models/                      ← Trained artefacts (git-ignored)
│   ├── scaler.pkl
│   ├── xgboost.pkl
│   ├── xgboost_calibrated.pkl
│   ├── lightgbm.pkl
│   ├── isolation_forest.pkl
│   ├── kmeans.pkl
│   ├── shap_explainer.pkl
│   ├── cluster_names.json
│   ├── narrative_config.json
│   └── metadata.json
└── data/
    └── risk_dataset.csv         ← Generated training data (git-ignored)
```

---

## Feature Schema v3.0 — 15 canonical features

Defined in `risk_feature_schema.py` as `FEATURE_COLS_CANONICAL`.
This file is the **single source of truth** shared by the generator, training, and inference.

| Feature | Scale | Source |
|---|---|---|
| `lc_amount` | USD | MT700 field 32B |
| `country_risk_score` | 0–100 | max(internal tier, DS analysis × 33.3, OFAC/EU level × 25) |
| `external_currency_risk` | 0–4 | `external_validation.details.currency.risk` |
| `incoterm_risk` | 0–5 | `ds_analysis.features.incoterm_risk` (EXW=5, DDP=1) |
| `ucp_compliance_score` | 0–1 | `ucp_compliance.global_score / 100` |
| `n_ucp_warnings` | int | `ucp_compliance.summary.errors + warnings` |
| `ucp_warning_rate` | 0–1 | warnings / clauses analysées |
| `nlp_score` | 0–100 | `ds_analysis.features.nlp_score` |
| `semantic_score` | 0–1 | `ds_analysis.features.semantic_score` |
| `completeness_score` | 0–1 | `ds_analysis.features.completeness_score` |
| `days_to_shipment` | days | derived from MT700 field 44C |
| `presentation_days` | days | MT700 field 48 |
| `validity_days` | days | derived from fields 31D / 44C |
| `transshipment_allowed` | 0/1 | MT700 field 43T |
| `charges_beneficiary` | 0/1 | MT700 field 71B |

### Removed from v2.2 (22 features → 15)

`country_risk`, `country_risk_level`, `external_country_score`, `external_country_level`
→ merged into single `country_risk_score`

`n_clauses`, `n_documents_required`, `ucp_decision_reject`
→ removed (low discriminance or circular with rule engine)

### Shared constants

```python
from risk_feature_schema import SANCTIONED_ISO2, INCOTERM_RISK_MAP
```

- `SANCTIONED_ISO2` — frozenset of 7 ISO2 codes (OFAC/EU/UN/FATF)
- `INCOTERM_RISK_MAP` — `{"EXW": 5, "DDP": 1, …}` aligned with `swift_ds_pipeline.py`

---

## Models

| Artefact | Role | Algorithm |
|---|---|---|
| `scaler.pkl` | Feature normalisation | MinMaxScaler (fit on train only) |
| `xgboost.pkl` | Risk classification (main) | XGBClassifier (800 trees) |
| `xgboost_calibrated.pkl` | Reliable probabilities | CalibratedClassifierCV / Platt |
| `lightgbm.pkl` | Challenger comparison | LGBMClassifier (800 trees) |
| `isolation_forest.pkl` | Anomaly detection | IsolationForest (contamination=8%) |
| `kmeans.pkl` | LC segmentation | KMeans (4 clusters) |
| `shap_explainer.pkl` | Per-prediction XAI | TreeExplainer on raw XGBoost |
| `metadata.json` | Pipeline contract | version, feature_cols, dataset_sha256, eval |

### Cluster names

| ID | Name | Typical profile |
|---|---|---|
| 0 | `LOW_RISK_STANDARD` | Tier-0/1 countries, simple terms |
| 1 | `MEDIUM_RISK_COMPLEX` | Mixed flags, multi-modal |
| 2 | `HIGH_RISK_SENSITIVE` | Sanctioned-adjacent, high warnings |
| 3 | `STRUCTURED_TRADE_FINANCE` | Large amounts, deferred payment |

---

## Training Pipeline

### 1. Generate data

```bash
cd swift-validator-ia
python services/risk_service/scripts/risk_dataset_generator.py
```

Produces `data/risk_dataset.csv` — 30 000 synthetic MT700 LCs with:
- 15 canonical features
- `true_label` (expert non-circular labeling)
- `ws_label` (rule engine baseline for comparison)
- Distribution: HIGH ≈ 43%, MEDIUM ≈ 35%, LOW ≈ 23%

### 2. Train

```bash
python services/risk_service/scripts/train_models.py
```

Split: **60% train / 20% calibration / 20% test** (SMOTE on train only, no leakage).

Typical results (test set, XGBoost calibrated):

```
              precision  recall  f1-score  support
LOW             0.969   0.969    0.969     1370
MEDIUM          0.912   0.928    0.920     2076
HIGH            0.958   0.944    0.951     2554
accuracy                         0.944     6000
```

> **Note**: CV F1 (~0.94) is computed on SMOTE-resampled data and overestimates
> real-world performance. Use test-set F1 for production estimates.

> **SHAP vs calibrated**: SHAP explainer is fitted on the raw XGBoost model.
> Feature attributions are accurate; probabilities at inference use the calibrated wrapper.

### 3. Evaluate

```bash
python evaluation/risk_service/evaluate_risk_models.py
```

Produces in `evaluation/risk_service/reports/`:
- `report.json` — full metrics JSON
- `confusion_xgb_calibrated.png`
- `roc_xgb_calibrated.png`, `roc_lightgbm.png`
- `calibration_curve.png` — reliability diagram (raw vs calibrated)
- `shap_importance.png` — top-10 features per class
- `anomaly_score_distribution.png`

---

## Inference

The service is consumed by `swift_pipeline.py` as the final stage of the pipeline:

```
UCP + DS analysis → External validation → RiskML
```

### Pipeline payload expected

```python
pipeline_payload = {
    "swift":              { "lc_criteria": {...}, "entities": {...} },
    "ucp_compliance":     { "global_score": float, "ucp_decision": str,
                            "summary": {"errors": int, "warnings": int} },
    "ds_analysis":        { "features": { "country_risk": int, "incoterm_risk": int,
                                          "nlp_score": float, ... } },
    "external_validation":{ "risk_level": str,  # CLEAN/LOW/MEDIUM/HIGH/CRITICAL
                            "details": { "currency": { "risk": str } } },
}
```

### Python API

```python
from services.risk_service.risk_engine import RiskMLService

svc = RiskMLService()  # loads models once on first call
result = svc.predict(pipeline_payload)

result["label"]          # "LOW" | "MEDIUM" | "HIGH"
result["confidence"]     # float 0–1
result["probabilities"]  # {"LOW": 0.05, "MEDIUM": 0.12, "HIGH": 0.83}
result["segment"]        # "HIGH_RISK_SENSITIVE"
result["is_anomaly"]     # bool
result["shap_values"]    # {feature: shap_value} for predicted class
result["narrative"]      # human-readable explanation string
result["ucp_violations"] # list of rule violations (may be empty)
result["ml_label"]       # ML prediction before rule override
```

---

## UCP 600 Rule Engine

Hard rules evaluated **before** the ML model. A BLOCK/HIGH override forces
`label = "HIGH"` regardless of ML output.

| Rule ID | Condition | Severity |
|---|---|---|
| `UCP_COMPLIANCE_REJECT` | `ucp_decision == "REJECT"` | BLOCK |
| `OFAC_COUNTRY` | `country_risk_score >= 90` | BLOCK |
| `UCP_14_COMPLIANCE` | `ucp_compliance_score < 0.30` | HIGH |
| `UCP_16_WARNINGS` | `n_ucp_warnings >= 5` | HIGH |
| `UCP_29_PRESENTATION` | `presentation_days < 5` | HIGH |
| `COMPLETENESS_LOW` | `completeness_score < 0.50` | WARN |

---

## country_risk_score consolidation

Three sources merged into a single 0–100 score:

1. **Internal** — ISO2 country tier lookup (tier 0→5, tier 4/sanctioned→95)
2. **DS analysis** — `ds_analysis.features.country_risk` (0–3) × 33.3
3. **External OFAC/EU** — `external_validation.risk_level` mapped to 0/25/50/75/100

Resolution rule:
- External `CLEAN` → force 0.0 (validated clean)
- External non-null → `max(base, ext_score)`
- External absent → `max(internal, ds_score)`

---

## Feature Validation

`RiskFeatureExtractor._validate_features()` clips every ML feature to its known range
and logs a warning for each anomaly. Clipped values are still passed to the model;
the list of issues is available at `result["_validation_issues"]` (internal key,
not exported in the API response).

---

## Adding new features

1. Add the feature to `FEATURE_COLS_CANONICAL` in `risk_feature_schema.py`
   and bump `FEATURE_SCHEMA_VERSION`
2. Implement extraction in `RiskFeatureExtractor.extract()` in `risk_engine.py`
3. Add generation logic in `_generate_lc_profile()` in `risk_dataset_generator.py`
4. Add range to `_FEATURE_RANGES` in `risk_engine.py`
5. Add label to `NarrativeExplainer.FEATURE_LABELS` in `train_models.py`
6. Retrain: `risk_dataset_generator.py` → `train_models.py`

Candidate features tracked in `metadata.json["future_features_roadmap"]`:
`confirming_bank_risk`, `doc_discrepancy_rate`, `red_clause_flag`,
`commodity_risk`, `bank_tier`, `applicant_history_score`, `tenor_type`, `embargo_flag`.

---

## Versioning

`metadata.json` embeds `feature_schema_version` and `dataset_sha256`.
At model load, `scaler.n_features_in_` is asserted against the resolved schema length.
A mismatch raises `RuntimeError` — the model cannot be used with the wrong schema.
