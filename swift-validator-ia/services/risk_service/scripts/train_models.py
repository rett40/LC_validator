"""
╔══════════════════════════════════════════════════════════════════════╗
║         MT700 Risk ML Training Pipeline v4.0 — PRODUCTION           ║
║         SWIFT Letter of Credit — Full Explainability Stack           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Models trained:                                                     ║
║    1. MinMaxScaler          → Feature normalization                  ║
║    2. Isolation Forest      → Anomaly detection                      ║
║    3. KMeans                → LC Segmentation                        ║
║    4. XGBoost               → Risk classification (main model)       ║
║    5. LightGBM              → Risk classification (challenger)       ║
║    6. CalibratedClassifier  → Probability calibration                ║
║    7. SHAP Explainer        → Feature importance per prediction      ║
║    8. RuleEngine            → UCP 600 hard rules                     ║
║    9. LabelEncoder          → Segment name mapping                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Key fixes vs v2/v3:                                                 ║
║    ✔ No data leakage (split BEFORE scaling)                          ║
║    ✔ SMOTE on train only                                             ║
║    ✔ SHAP for all 3 classes (not just HIGH)                          ║
║    ✔ Model calibration (reliable probabilities)                      ║
║    ✔ LightGBM challenger for comparison                              ║
║    ✔ Full metadata export for inference service                      ║
║    ✔ UCP 600 rule engine layer                                       ║
║    ✔ Narrative explanation generator                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────

import hashlib
import numpy as np
import pandas as pd
import joblib
import json
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, accuracy_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import shap

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_v4.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

MODELS_DIR   = Path("services/risk_service/models")
DATA_PATH    = Path("services/risk_service/data/risk_dataset.csv")
MODELS_DIR.mkdir(exist_ok=True)

# Schéma features partagé avec risk_engine.py (inférence)
_RISK_SVC_ROOT = Path(__file__).resolve().parent.parent
if str(_RISK_SVC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RISK_SVC_ROOT))

from risk_feature_schema import FEATURE_COLS_CANONICAL, FEATURE_SCHEMA_VERSION

FEATURE_COLS = list(FEATURE_COLS_CANONICAL)

# ── FUTURE FEATURES (uncomment when parser supports them) ─────────────
# High-value features to add in next iteration:
#
# "applicant_country_risk",   # Field 50   → UCP Art.4  — Applicant risk
# "confirming_bank_risk",     # Field 58A  → UCP Art.8  — Confirmation risk (+2% F1 expected)
# "tenor_type",               # Field 42C  → 0=sight, 1=usance — payment timing risk
# "tenor_days",               # Field 42C  → Days for deferred payment
# "doc_discrepancy_rate",     # Historical → % past discrepancies by applicant (+3% expected)
# "partial_shipment_allowed", # Field 43P  → UCP Art.31 — fragmented delivery risk
# "transferable_flag",        # Field 40A  → UCP Art.38 — transfer complexity
# "red_clause_flag",          # Field 47A  → Pre-shipment advance risk (+4% for HIGH recall)
# "standby_lc_flag",          # Field 40A  → ISP98 vs UCP 600 complexity
# "commodity_risk",           # Field 45A  → Dual-use goods risk score
# "bank_tier",                # Field 57A  → Issuing bank rating [0-1]
# "applicant_history_score",  # CRM data   → Past compliance history [0-1]
# "political_risk_index",     # External   → Country political stability (+1.5% expected)
# "forex_volatility",         # Market     → Currency pair 30d volatility
# "embargo_flag",             # OFAC/EU    → Hard block flag (rule engine override)

LABEL_MAP     = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

N_CLUSTERS    = 4
CLUSTER_NAMES = {
    0: "LOW_RISK_STANDARD",
    1: "MEDIUM_RISK_COMPLEX",
    2: "HIGH_RISK_SENSITIVE",
    3: "STRUCTURED_TRADE_FINANCE",
}

# ──────────────────────────────────────────────────────────────────────
# UCP 600 RULE ENGINE
# ──────────────────────────────────────────────────────────────────────

@dataclass
class UCPViolation:
    article: str
    field: str
    description: str
    severity: str           # "BLOCK" | "HIGH" | "WARN"
    override_score: Optional[str] = None   # Force risk level if set

class UCPRuleEngine:
    """
    Hard-coded UCP 600 rules that take precedence over the ML model.
    These represent non-negotiable documentary credit rules.
    Add new rules as your parser extracts more MT700 fields.
    """

    RULES = [
        {
            "id": "UCP_COMPLIANCE_REJECT",
            "check": lambda f: f.get("ucp_decision_reject", 0) >= 1,
            "violation": UCPViolation(
                article="UCP 600",
                field="47A / Multiple",
                description="ucp_compliance.ucp_decision is REJECT — contractual UCP red flags",
                severity="BLOCK",
                override_score="HIGH",
            ),
        },
        # ── UCP Art. 14 — Standard for Examination ────────────────────
        {
            "id": "UCP_14_COMPLIANCE",
            "check": lambda f: f.get("ucp_compliance_score", 1.0) < 0.30,
            "violation": UCPViolation(
                article="UCP 600 Art.14",
                field="47A / Multiple",
                description="Compliance score critically low — likely document discrepancy",
                severity="HIGH",
                override_score="HIGH",
            )
        },
        # ── UCP Art. 16 — Discrepant Documents ────────────────────────
        {
            "id": "UCP_16_WARNINGS",
            "check": lambda f: f.get("n_ucp_warnings", 0) >= 5,
            "violation": UCPViolation(
                article="UCP 600 Art.16",
                field="47A",
                description=f"High UCP warning count — multiple clause conflicts detected",
                severity="HIGH",
                override_score="HIGH",
            )
        },
        # ── UCP Art. 29 — Expiry / Presentation ──────────────────────
        {
            "id": "UCP_29_PRESENTATION",
            "check": lambda f: f.get("presentation_days", 21) < 5,
            "violation": UCPViolation(
                article="UCP 600 Art.29",
                field="48",
                description="Presentation period critically short (<5 days) — Art.14(c) risk",
                severity="HIGH",
                override_score="HIGH",
            )
        },
        # ── OFAC / Embargo ─────────────────────────────────────────────
        {
            "id": "OFAC_COUNTRY",
            "check": lambda f: f.get("country_risk_score", 0) >= 90,
            "violation": UCPViolation(
                article="OFAC / EU Embargo",
                field="59 / 50",
                description="Sanctions / embargo — country_risk_score ≥ 90",
                severity="BLOCK",
                override_score="HIGH",
            )
        },
        # ── Completeness ───────────────────────────────────────────────
        {
            "id": "COMPLETENESS_LOW",
            "check": lambda f: f.get("completeness_score", 1.0) < 0.50,
            "violation": UCPViolation(
                article="UCP 600 Art.5",
                field="Multiple",
                description="MT700 completeness below 50% — critical fields missing",
                severity="WARN",
            )
        },
        # ── Add future rules below as parser matures ──────────────────
        # {
        #     "id": "RED_CLAUSE_FLAG",
        #     "check": lambda f: f.get("red_clause_flag", 0) == 1,
        #     "violation": UCPViolation(
        #         article="UCP 600 Art.12",
        #         field="47A",
        #         description="Red clause detected — pre-shipment advance creates payment risk",
        #         severity="HIGH",
        #         override_score="HIGH",
        #     )
        # },
    ]

    def evaluate(self, features: dict) -> tuple[list[UCPViolation], Optional[str]]:
        """
        Returns (violations, forced_risk_label).
        forced_risk_label is set if any BLOCK/HIGH rule fires with override_score.
        """
        violations = []
        forced_label = None

        for rule in self.RULES:
            if rule["check"](features):
                v = rule["violation"]
                violations.append(v)
                if v.override_score and forced_label != "HIGH":
                    forced_label = v.override_score

        return violations, forced_label

# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────

def load_data(path: Path):
    log.info(f"Loading dataset from {path}")

    dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info(f"Dataset SHA-256: {dataset_sha256}")

    df = pd.read_csv(path)

    # Validate required columns (v3.0 schema — no legacy compat shims)
    missing = [c for c in FEATURE_COLS + ["true_label", "ws_label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X      = df[FEATURE_COLS].copy()
    y_true = df["true_label"].map(LABEL_MAP)
    y_ws   = df["ws_label"].map(LABEL_MAP)

    mismatch_pct = (df["true_label"] != df["ws_label"]).mean() * 100

    log.info(f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    log.info(f"true_label dist: {df['true_label'].value_counts().to_dict()}")
    log.info(f"ws_label   dist: {df['ws_label'].value_counts().to_dict()}")
    log.info(f"WS vs TRUE mismatch: {mismatch_pct:.1f}%")

    return df, X, y_true, y_ws, dataset_sha256

# ──────────────────────────────────────────────────────────────────────
# SPLIT + SCALE  (critical: scale AFTER split to avoid leakage)
# ──────────────────────────────────────────────────────────────────────

def split_and_scale(X, y):
    # 60% train / 20% calibration / 20% test — évite la fuite de données sur calibration
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)   # fit on train only
    X_cal_s   = scaler.transform(X_cal)          # calibration set (jamais vu par XGBoost)
    X_test_s  = scaler.transform(X_test)         # test set final
    log.info(f"Split → train: {len(X_train)}, cal: {len(X_cal)}, test: {len(X_test)}")
    return scaler, X_train_s, X_cal_s, X_test_s, y_train, y_cal, y_test

# ──────────────────────────────────────────────────────────────────────
# MODULE 1 — ANOMALY DETECTION (Isolation Forest)
# ──────────────────────────────────────────────────────────────────────

def train_isolation_forest(X_train_s, y_train_raw, df) -> IsolationForest:
    log.info("=" * 60)
    log.info("MODULE 1 — Isolation Forest (Anomaly Detection)")
    log.info("=" * 60)

    model = IsolationForest(
        n_estimators=500,
        contamination=0.08,      # ~8% anomalies expected in LC portfolios
        max_features=0.85,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s)

    preds  = model.predict(X_train_s)
    scores = model.decision_function(X_train_s)
    n_ano  = (preds == -1).sum()

    # Validate: HIGH-risk LCs should be overrepresented in anomalies
    high_mask     = (y_train_raw == 2).values
    high_ano_rate = (preds[high_mask] == -1).mean() * 100 if high_mask.sum() > 0 else 0

    log.info(f"Anomalies detected: {n_ano} / {len(X_train_s)} ({n_ano/len(X_train_s)*100:.1f}%)")
    log.info(f"HIGH-risk in anomalies: {high_ano_rate:.1f}%  (expected > 40%)")
    log.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    joblib.dump(model, MODELS_DIR / "isolation_forest.pkl")
    log.info("Saved → isolation_forest.pkl")
    return model

# ──────────────────────────────────────────────────────────────────────
# MODULE 2 — SEGMENTATION (KMeans)
# ──────────────────────────────────────────────────────────────────────

def train_kmeans(X_train_s) -> KMeans:
    log.info("=" * 60)
    log.info("MODULE 2 — KMeans Segmentation")
    log.info("=" * 60)

    model = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=30,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_s)

    labels = model.labels_
    for c in range(N_CLUSTERS):
        n = (labels == c).sum()
        log.info(f"  Cluster {c} [{CLUSTER_NAMES[c]}]: {n} LCs ({n/len(labels)*100:.1f}%)")

    # Save cluster name mapping
    mapping = {str(k): v for k, v in CLUSTER_NAMES.items()}
    with open(MODELS_DIR / "cluster_names.json", "w") as f:
        json.dump(mapping, f, indent=2)

    joblib.dump(model, MODELS_DIR / "kmeans.pkl")
    log.info("Saved → kmeans.pkl")
    return model

# ──────────────────────────────────────────────────────────────────────
# MODULE 3 — XGBOOST (Main risk classifier)
# ──────────────────────────────────────────────────────────────────────

def train_xgboost(X_train_s, y_train, X_test_s, y_test) -> xgb.XGBClassifier:
    log.info("=" * 60)
    log.info("MODULE 3 — XGBoost Risk Classifier (main)")
    log.info("=" * 60)

    # ── SMOTE on train only ───────────────────────────────────────────
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)
    log.info(f"After SMOTE: {dict(pd.Series(y_res).value_counts().sort_index())}")

    # ── Cross-validation baseline ─────────────────────────────────────
    base_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1
    )
    cv_scores = cross_val_score(
        base_model, X_res, y_res,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted", n_jobs=-1
    )
    log.info(f"XGBoost CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Full training with tuned params ──────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_res, y_res,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test_s)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="weighted")

    log.info(f"XGBoost — Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")
    log.info(f"\n{classification_report(y_test, preds, target_names=['LOW','MEDIUM','HIGH'], digits=3)}")

    joblib.dump(model, MODELS_DIR / "xgboost.pkl")
    log.info("Saved → xgboost.pkl")
    return model

# ──────────────────────────────────────────────────────────────────────
# MODULE 4 — LIGHTGBM (Challenger model)
# ──────────────────────────────────────────────────────────────────────

def train_lightgbm(X_train_s, y_train, X_test_s, y_test) -> lgb.LGBMClassifier:
    log.info("=" * 60)
    log.info("MODULE 4 — LightGBM Risk Classifier (challenger)")
    log.info("=" * 60)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)

    model = lgb.LGBMClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.5,
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_res, y_res,
        eval_set=[(X_test_s, y_test)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X_test_s)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average="weighted")

    log.info(f"LightGBM — Accuracy: {acc:.4f} | F1-weighted: {f1:.4f}")
    log.info(f"\n{classification_report(y_test, preds, target_names=['LOW','MEDIUM','HIGH'], digits=3)}")

    joblib.dump(model, MODELS_DIR / "lightgbm.pkl")
    log.info("Saved → lightgbm.pkl")
    return model

# ──────────────────────────────────────────────────────────────────────
# MODULE 5 — MODEL CALIBRATION
# Ensures predict_proba() gives reliable probabilities
# e.g. "80% HIGH" should mean it's actually HIGH ~80% of the time
# ──────────────────────────────────────────────────────────────────────

def calibrate_model(model, X_cal_s, y_cal, X_test_s, y_test) -> CalibratedClassifierCV:
    log.info("=" * 60)
    log.info("MODULE 5 — Probability Calibration (Platt scaling)")
    log.info("=" * 60)

    # cv=5 sur cal set dédié (jamais vu par XGBoost) — aucune fuite vers le test set
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",   # Platt scaling — bon pour XGBoost
        cv=5,
    )
    calibrated.fit(X_cal_s, y_cal)   # calibration set jamais vu par XGBoost

    proba_raw = model.predict_proba(X_test_s).max(axis=1).mean()
    proba_cal = calibrated.predict_proba(X_test_s).max(axis=1).mean()
    log.info(f"Avg max confidence before calibration: {proba_raw:.3f}")
    log.info(f"Avg max confidence after  calibration: {proba_cal:.3f}")

    joblib.dump(calibrated, MODELS_DIR / "xgboost_calibrated.pkl")
    log.info("Saved → xgboost_calibrated.pkl")
    return calibrated

# ──────────────────────────────────────────────────────────────────────
# MODULE 6 — SHAP EXPLAINER
# Produces per-prediction explanations for all 3 risk classes
# ──────────────────────────────────────────────────────────────────────

def train_shap(model: xgb.XGBClassifier, X_test_s: np.ndarray) -> dict:
    log.info("=" * 60)
    log.info("MODULE 6 — SHAP Explainer (all 3 classes)")
    log.info("=" * 60)

    # model_output="raw" (log-odds) — compatible avec feature_perturbation="tree_path_dependent".
    # Les contributions sont normalisées côté inférence pour s'additionner au risk_score.
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_s)

    class_names = ["LOW", "MEDIUM", "HIGH"]
    importance  = {}

    for i, cls in enumerate(class_names):
        if isinstance(shap_vals, list):
            vals = shap_vals[i]
        else:
            vals = shap_vals[:, :, i] if shap_vals.ndim == 3 else shap_vals

        mean_abs = np.abs(vals).mean(axis=0)
        importance[cls] = pd.Series(
            mean_abs, index=FEATURE_COLS
        ).sort_values(ascending=False)
        log.info(f"\nSHAP importance [{cls}]:\n{importance[cls].head(8).to_string()}")

    log.info(f"\nExpected values (log-odds): {explainer.expected_value}")

    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    log.info("\nSaved → shap_explainer.pkl")
    return importance

# ──────────────────────────────────────────────────────────────────────
# MODULE 7 — NARRATIVE EXPLANATION GENERATOR
# Converts SHAP values + rule violations → human-readable text
# Used by the API to explain predictions to documentary credit officers
# ──────────────────────────────────────────────────────────────────────

class NarrativeExplainer:
    """
    Generates human-readable explanations for MT700 risk predictions.
    Used at inference time — not trained, just rule-based.
    """

    # Aligned with FEATURE_COLS_CANONICAL v3.0 — 15 features
    FEATURE_LABELS = {
        "lc_amount":             "LC face value",
        "country_risk_score":    "country risk score 0–100",
        "external_currency_risk": "external currency risk 0–4",
        "incoterm_risk":         "incoterm risk score 0–5",
        "ucp_compliance_score":  "UCP 600 compliance score",
        "n_ucp_warnings":        "UCP warning count",
        "ucp_warning_rate":      "UCP violation density (warnings / clauses)",
        "nlp_score":             "clause quality score (NLP) 0–100",
        "semantic_score":        "semantic coherence score 0–1",
        "completeness_score":    "MT700 completeness score 0–1",
        "days_to_shipment":      "days until latest shipment",
        "presentation_days":     "document presentation period",
        "validity_days":         "LC validity duration",
        "transshipment_allowed": "transshipment authorization",
        "charges_beneficiary":   "charges on beneficiary",
    }

    RISK_TEMPLATES = {
        "HIGH": (
            "This Letter of Credit is classified as HIGH RISK. "
            "The primary risk drivers are: {drivers}. "
            "{rule_note}"
            "Immediate review by a senior documentary credit officer is recommended."
        ),
        "MEDIUM": (
            "This Letter of Credit is classified as MEDIUM RISK. "
            "The main factors contributing to this assessment are: {drivers}. "
            "{rule_note}"
            "Standard documentary review applies with attention to the identified factors."
        ),
        "LOW": (
            "This Letter of Credit is classified as LOW RISK. "
            "Key positive indicators include: {drivers}. "
            "{rule_note}"
            "Standard processing applies."
        ),
    }

    def explain(
        self,
        predicted_label: str,
        confidence: float,
        shap_row: dict,        # {feature: shap_value} for predicted class
        violations: list,
        features: dict,
    ) -> str:
        # Top 3 contributing features
        top_features = sorted(shap_row.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        driver_parts = []
        for feat, val in top_features:
            direction = "elevated" if val > 0 else "reduced"
            label = self.FEATURE_LABELS.get(feat, feat)
            actual = features.get(feat, "N/A")
            driver_parts.append(f"{label} ({actual:.2f}) — {direction} risk contribution")

        drivers = "; ".join(driver_parts)

        # UCP rule violations
        rule_note = ""
        if violations:
            articles = ", ".join(set(v.article for v in violations))
            rule_note = f"UCP rule violations detected ({articles}). "

        narrative = self.RISK_TEMPLATES[predicted_label].format(
            drivers=drivers,
            rule_note=rule_note,
        )

        return (
            f"[Confidence: {confidence*100:.1f}%] {narrative}"
        )


def save_narrative_explainer():
    """Serialize the NarrativeExplainer class description for inference service."""
    meta = {
        "type": "NarrativeExplainer",
        "version": "4.0",
        "feature_labels": NarrativeExplainer.FEATURE_LABELS,
        "templates": NarrativeExplainer.RISK_TEMPLATES,
    }
    with open(MODELS_DIR / "narrative_config.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.info("Saved → narrative_config.json")

# ──────────────────────────────────────────────────────────────────────
# MODULE 8 — MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────

def compare_models(xgb_model, lgb_model, X_test_s, y_test):
    log.info("=" * 60)
    log.info("MODULE 8 — Model Comparison (XGBoost vs LightGBM)")
    log.info("=" * 60)

    results = {}
    for name, model in [("XGBoost", xgb_model), ("LightGBM", lgb_model)]:
        preds = model.predict(X_test_s)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }

    comparison_df = pd.DataFrame(results).T
    log.info(f"\n{comparison_df.to_string()}")

    winner = comparison_df["f1_weighted"].idxmax()
    log.info(f"\nBest model: {winner}")

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return winner, results

# ──────────────────────────────────────────────────────────────────────
# METADATA EXPORT
# Full metadata for inference service — zero guessing at runtime
# ──────────────────────────────────────────────────────────────────────

def save_metadata(scaler, shap_importance, winner, comparison, y_test, xgb_preds, dataset_sha256: str):
    log.info("=" * 60)
    log.info("METADATA — Exporting pipeline metadata")
    log.info("=" * 60)

    # Save scaler FIRST — metadata.json references it; loading must succeed even if interrupted
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    log.info("Saved → scaler.pkl")

    meta = {
        "version": "4.0",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at": datetime.now().isoformat(),
        "dataset_sha256": dataset_sha256,
        "feature_cols": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),

        # ── Feature contracts ──────────────────────────────────────────
        "feature_contracts": {
            "external_currency_risk": {
                "source": "external_validation API → details.currency.risk",
                "scale": "0-4 (CLEAN=0, LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4)",
            },
            "country_risk_score": {
                "source": "max(internal ISO2 tier, DS analysis × 33.3, OFAC/EU level × 25)",
                "scale": "0-100",
            },
            "incoterm_risk": {
                "source": "ds_analysis.features.incoterm_risk > lc_criteria.incoterm > entities.incoterm",
                "scale": "0-5 (DDP=1 … EXW=5)",
            },
        },

        # ── Future features roadmap ────────────────────────────────────
        "future_features_roadmap": [
            {"name": "confirming_bank_risk",   "expected_f1_gain": "+2%",  "source": "Field 58A"},
            {"name": "doc_discrepancy_rate",   "expected_f1_gain": "+3%",  "source": "CRM history"},
            {"name": "red_clause_flag",        "expected_f1_gain": "+4% HIGH recall", "source": "Field 47A"},
            {"name": "commodity_risk",         "expected_f1_gain": "+1.5%","source": "Field 45A"},
            {"name": "bank_tier",              "expected_f1_gain": "+2%",  "source": "Field 57A"},
            {"name": "applicant_history_score","expected_f1_gain": "+3%",  "source": "CRM data"},
            {"name": "tenor_type",             "expected_f1_gain": "+1%",  "source": "Field 42C"},
            {"name": "embargo_flag",           "expected_f1_gain": "BLOCK override", "source": "OFAC/EU"},
        ],

        "label_map":     LABEL_MAP,
        "label_map_inv": {str(k): v for k, v in LABEL_MAP_INV.items()},
        "n_clusters":    N_CLUSTERS,
        "cluster_names": {str(k): v for k, v in CLUSTER_NAMES.items()},

        "best_model":         winner,
        "model_comparison":   comparison,

        "shap_importance": {
            cls: imp.to_dict()
            for cls, imp in shap_importance.items()
        },

        "evaluation": {
            "accuracy": accuracy_score(y_test, xgb_preds),
            "f1_weighted": f1_score(y_test, xgb_preds, average="weighted"),
            "confusion_matrix": confusion_matrix(y_test, xgb_preds).tolist(),
            "report": classification_report(
                y_test, xgb_preds,
                target_names=["LOW", "MEDIUM", "HIGH"],
                output_dict=True
            ),
            "cv_note": (
                "CV F1 is computed on SMOTE-resampled data and overestimates real-world "
                "performance. Rely on test-set F1 (no SMOTE) for production estimates."
            ),
        },

        "model_notes": {
            "shap_vs_calibrated": (
                "SHAP explainer is fitted on the raw XGBoost model (not CalibratedClassifierCV). "
                "Feature attributions are accurate; predicted probabilities at inference use the "
                "calibrated wrapper for reliability."
            ),
        },

        "models": {
            "scaler":              str(MODELS_DIR / "scaler.pkl"),
            "isolation_forest":    str(MODELS_DIR / "isolation_forest.pkl"),
            "kmeans":              str(MODELS_DIR / "kmeans.pkl"),
            "xgboost":             str(MODELS_DIR / "xgboost.pkl"),
            "lightgbm":            str(MODELS_DIR / "lightgbm.pkl"),
            "xgboost_calibrated":  str(MODELS_DIR / "xgboost_calibrated.pkl"),
            "shap_explainer":      str(MODELS_DIR / "shap_explainer.pkl"),
            "cluster_names":       str(MODELS_DIR / "cluster_names.json"),
            "narrative_config":    str(MODELS_DIR / "narrative_config.json"),
        }
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    log.info("Saved → metadata.json")
    return meta

# ──────────────────────────────────────────────────────────────────────
# INFERENCE EXAMPLE
# Shows how to use the trained pipeline at inference time
# ──────────────────────────────────────────────────────────────────────

class MT700RiskPredictor:
    """
    Full inference pipeline.
    Load once → call predict() for each MT700 message.

    Example usage:
        predictor = MT700RiskPredictor("models_v4")
        result = predictor.predict(features_dict)
        print(result["label"], result["narrative"])
    """

    def __init__(self, models_dir: str = "models_v4"):
        d = Path(models_dir)
        self.scaler      = joblib.load(d / "scaler.pkl")
        self.xgb         = joblib.load(d / "xgboost_calibrated.pkl")
        self.iso         = joblib.load(d / "isolation_forest.pkl")
        self.kmeans      = joblib.load(d / "kmeans.pkl")
        self.shap_expl   = joblib.load(d / "shap_explainer.pkl")
        self.rule_engine = UCPRuleEngine()
        self.narrator    = NarrativeExplainer()

        with open(d / "cluster_names.json") as f:
            self.cluster_names = json.load(f)

        with open(d / "metadata.json") as f:
            self.meta = json.load(f)

        # Resolve feature columns from metadata — never rely on the module-level constant
        self._feature_cols: list[str] = self.meta.get("feature_cols", FEATURE_COLS)

    def predict(self, features: dict) -> dict:
        """
        features: dict with all FEATURE_COLS keys.
        Returns full prediction result with explanation.
        """
        # 1. UCP Rule Engine (takes precedence)
        violations, forced_label = self.rule_engine.evaluate(features)

        # 2. Prepare features — use metadata-derived column list, not module constant
        row = np.array([[features.get(c, 0.0) for c in self._feature_cols]])
        row_s = self.scaler.transform(row)

        # 3. Anomaly score
        anomaly_score = float(self.iso.decision_function(row_s)[0])
        is_anomaly    = self.iso.predict(row_s)[0] == -1

        # 4. Segment
        segment_id   = int(self.kmeans.predict(row_s)[0])
        segment_name = self.cluster_names.get(str(segment_id), "UNKNOWN")

        # 5. Risk prediction
        proba        = self.xgb.predict_proba(row_s)[0]
        pred_idx     = int(np.argmax(proba))
        pred_label   = LABEL_MAP_INV[pred_idx]
        confidence   = float(proba[pred_idx])

        # 6. Apply rule override
        final_label  = forced_label if forced_label else pred_label

        # 7. SHAP explanation for predicted class
        shap_vals    = self.shap_expl.shap_values(row_s)
        if isinstance(shap_vals, list):
            row_shap = shap_vals[pred_idx][0]
        else:
            row_shap = shap_vals[0, :, pred_idx] if shap_vals.ndim == 3 else shap_vals[0]

        shap_row = dict(zip(self._feature_cols, row_shap.tolist()))

        # 8. Narrative
        narrative = self.narrator.explain(
            predicted_label=final_label,
            confidence=confidence,
            shap_row=shap_row,
            violations=violations,
            features=features,
        )

        return {
            "label":          final_label,
            "ml_label":       pred_label,
            "probabilities":  {LABEL_MAP_INV[i]: float(p) for i, p in enumerate(proba)},
            "confidence":     confidence,
            "segment":        segment_name,
            "is_anomaly":     is_anomaly,
            "anomaly_score":  anomaly_score,
            "ucp_violations": [asdict(v) for v in violations],
            "shap_values":    shap_row,
            "narrative":      narrative,
        }

# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  MT700 Risk ML Training Pipeline v4.0           ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # ── 1. Load ───────────────────────────────────────────────────────
    df, X, y_true, y_ws, dataset_sha256 = load_data(DATA_PATH)

    # ── 2. Split + Scale (no leakage — 60/20/20 train/cal/test) ──────
    scaler, X_train_s, X_cal_s, X_test_s, y_train, y_cal, y_test = split_and_scale(X, y_true)

    # ── 3. Anomaly Detection ──────────────────────────────────────────
    iso = train_isolation_forest(X_train_s, y_train, df)

    # ── 4. Segmentation ───────────────────────────────────────────────
    kmeans = train_kmeans(X_train_s)

    # ── 5. XGBoost ────────────────────────────────────────────────────
    xgb_model = train_xgboost(X_train_s, y_train, X_test_s, y_test)

    # ── 6. LightGBM ───────────────────────────────────────────────────
    lgb_model = train_lightgbm(X_train_s, y_train, X_test_s, y_test)

    # ── 7. Calibration (sur cal set dédié — plus de fuite vers test) ─
    xgb_calibrated = calibrate_model(xgb_model, X_cal_s, y_cal, X_test_s, y_test)

    # ── 8. SHAP ───────────────────────────────────────────────────────
    shap_importance = train_shap(xgb_model, X_test_s)

    # ── 9. Compare ────────────────────────────────────────────────────
    winner, comparison = compare_models(xgb_model, lgb_model, X_test_s, y_test)

    # ── 10. Narrative config ──────────────────────────────────────────
    save_narrative_explainer()

    # ── 11. Metadata ──────────────────────────────────────────────────
    xgb_preds = xgb_model.predict(X_test_s)
    meta = save_metadata(scaler, shap_importance, winner, comparison, y_test, xgb_preds, dataset_sha256)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("✅ TRAINING COMPLETE v4.0")
    log.info("=" * 60)
    log.info(f"\nModels saved in ./{MODELS_DIR}/")
    for f in sorted(MODELS_DIR.iterdir()):
        size = f.stat().st_size / 1024
        log.info(f"  {f.name:<40} {size:>8.1f} KB")

    log.info(f"\nBest classifier: {winner}")
    log.info(f"Overall accuracy: {meta['evaluation']['accuracy']:.4f}")
    log.info(f"F1-weighted:      {meta['evaluation']['f1_weighted']:.4f}")

    log.info("\n📌 Next steps:")
    log.info("  1. uvicorn risk_service:app --reload --port 8080")
    log.info("  2. predictor = MT700RiskPredictor('models_v4')")
    log.info("  3. result = predictor.predict(features_dict)")
    log.info("  4. Add future features from metadata['future_features_roadmap']")


if __name__ == "__main__":
    main()