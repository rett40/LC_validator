"""
Évaluation complète des modèles risk_service
=============================================
Métriques produites :
  - Accuracy, Precision, Recall, F1 (macro + weighted) par classe
  - Matrice de confusion (texte + graphique)
  - Courbes ROC (one-vs-rest pour chaque classe)
  - Courbe de calibration (reliability diagram)
  - Importance SHAP par classe
  - Rapport JSON structuré

Usage :
    cd swift-validator-ia
    python evaluation/risk_service/evaluate_risk_models.py
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # sans display (serveur / CI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, auc,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent.parent   # swift-validator-ia/
DATA_PATH   = ROOT / "services/risk_service/data/risk_dataset.csv"
MODELS_DIR  = ROOT / "services/risk_service/models"
OUT_DIR     = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

LABEL_MAP     = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LABEL_MAP_INV = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
CLASSES       = ["LOW", "MEDIUM", "HIGH"]
COLORS        = {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNÉES — même split que train_models.py
# ─────────────────────────────────────────────────────────────────────────────

def load_test_set():
    log.info("Chargement dataset : %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # Colonnes features (identiques à train_models.py)
    sys.path.insert(0, str(ROOT))
    from services.risk_service.risk_feature_schema import FEATURE_COLS_CANONICAL
    feature_cols = list(FEATURE_COLS_CANONICAL)

    missing = [c for c in feature_cols + ["true_label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    X = df[feature_cols]
    y = df["true_label"].map(LABEL_MAP)

    # Split 60/20/20 — MÊME seed que train_models.py pour obtenir le même test set
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    log.info("Test set : %d lignes", len(X_test))
    return X_test, y_test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def load_models():
    log.info("Chargement des modèles depuis %s", MODELS_DIR)
    scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    xgb_cal   = joblib.load(MODELS_DIR / "xgboost_calibrated.pkl")
    xgb_raw   = joblib.load(MODELS_DIR / "xgboost.pkl")
    lgb       = joblib.load(MODELS_DIR / "lightgbm.pkl")
    iso       = joblib.load(MODELS_DIR / "isolation_forest.pkl")
    shap_expl = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    with open(MODELS_DIR / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    log.info("Modèles chargés (version %s)", meta.get("version", "?"))
    return scaler, xgb_cal, xgb_raw, lgb, iso, shap_expl, meta


# ─────────────────────────────────────────────────────────────────────────────
# 3. MÉTRIQUES DE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred, y_proba, model_name: str) -> dict:
    acc        = accuracy_score(y_true, y_pred)
    f1_w       = f1_score(y_true, y_pred, average="weighted")
    f1_m       = f1_score(y_true, y_pred, average="macro")
    prec_w     = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w      = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    report     = classification_report(
        y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm         = confusion_matrix(y_true, y_pred)

    log.info("─── %s ───", model_name)
    log.info("Accuracy  : %.4f", acc)
    log.info("F1 weighted: %.4f | F1 macro: %.4f", f1_w, f1_m)
    log.info("Precision : %.4f | Recall: %.4f", prec_w, rec_w)
    log.info("\n%s", classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    # Brier score par classe (one-vs-rest)
    brier = {}
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    for i, cls in enumerate(CLASSES):
        brier[cls] = round(float(brier_score_loss(y_bin[:, i], y_proba[:, i])), 4)
    log.info("Brier scores : %s", brier)

    return {
        "model":       model_name,
        "accuracy":    round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro":    round(f1_m, 4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted":    round(rec_w, 4),
        "per_class":   {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
                "support":   int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
        "confusion_matrix": cm.tolist(),
        "brier_scores": brier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(CLASSES)),
        yticks=range(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ylabel="Réel",
        xlabel="Prédit",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Matrice de confusion → %s", out_path)


def plot_roc_curves(y_true, y_proba, title: str, out_path: Path):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[cls], lw=2,
                label=f"{cls} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.02],
           xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs",
           title=title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Courbes ROC → %s", out_path)


def plot_calibration(y_true, y_proba_raw, y_proba_cal, out_path: Path):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, cls in enumerate(CLASSES):
        ax = axes[i]
        for proba, label, ls in [
            (y_proba_raw[:, i], "XGBoost brut",      "--"),
            (y_proba_cal[:, i], "XGBoost calibré",   "-"),
        ]:
            frac_pos, mean_pred = calibration_curve(y_bin[:, i], proba, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", ls=ls, label=label)
        ax.plot([0, 1], [0, 1], "k:", label="Parfait")
        ax.set(title=f"Calibration — {cls}", xlabel="Confiance prédite",
               ylabel="Fraction positive réelle")
        ax.legend(fontsize=8)
    plt.suptitle("Reliability Diagram (calibration des probabilités)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Courbes de calibration → %s", out_path)


def plot_shap_importance(shap_expl, X_test_s, feature_cols: list, out_path: Path):
    try:
        import shap
        shap_values = shap_expl.shap_values(X_test_s[:500])  # 500 samples pour vitesse

        fig = plt.figure(figsize=(18, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig)

        for i, cls in enumerate(CLASSES):
            ax = fig.add_subplot(gs[0, i])
            if isinstance(shap_values, list):
                sv = np.abs(shap_values[i]).mean(axis=0)
            elif shap_values.ndim == 3:
                sv = np.abs(shap_values[:, :, i]).mean(axis=0)
            else:
                sv = np.abs(shap_values).mean(axis=0)

            idx_sorted = np.argsort(sv)[::-1][:10]
            feat_names = [feature_cols[j] for j in idx_sorted]
            values     = sv[idx_sorted]

            ax.barh(feat_names[::-1], values[::-1], color=COLORS[cls], alpha=0.8)
            ax.set(title=f"SHAP — {cls}", xlabel="Importance moyenne |SHAP|")
            ax.tick_params(labelsize=8)

        plt.suptitle("Importance SHAP par classe (top 10 features)", fontsize=12)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Importance SHAP → %s", out_path)
    except Exception as exc:
        log.warning("SHAP plot échoué : %s", exc)


def plot_class_distribution(y_true, out_path: Path):
    counts = {LABEL_MAP_INV[k]: int(v) for k, v in zip(*np.unique(y_true, return_counts=True))}
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.keys(), counts.values(),
                  color=[COLORS[c] for c in counts.keys()], edgecolor="white", linewidth=1.2)
    total = sum(counts.values())
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{val}\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set(title="Distribution des classes — test set", ylabel="Nombre de LCs")
    ax.set_ylim(0, max(counts.values()) * 1.15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Distribution classes → %s", out_path)


def plot_anomaly_score(y_true, iso_scores: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in CLASSES:
        mask = y_true == LABEL_MAP[cls]
        ax.hist(iso_scores[mask], bins=30, alpha=0.6, label=cls, color=COLORS[cls], density=True)
    ax.axvline(0, color="black", ls="--", lw=1, label="Seuil anomalie")
    ax.set(xlabel="Anomaly score (decision_function)", ylabel="Densité",
           title="Distribution des scores d'anomalie par classe")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Anomaly scores → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RAPPORT FINAL
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results: dict, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Rapport JSON → %s", out_path)


def print_summary(results: dict):
    print("\n" + "=" * 65)
    print("  RÉSUMÉ ÉVALUATION RISK SERVICE")
    print("=" * 65)
    for model_name, r in results["models"].items():
        print(f"\n  {model_name}")
        print(f"    Accuracy  : {r['accuracy']:.4f}")
        print(f"    F1 (w)    : {r['f1_weighted']:.4f}  |  F1 (macro): {r['f1_macro']:.4f}")
        print(f"    Precision : {r['precision_weighted']:.4f}  |  Recall: {r['recall_weighted']:.4f}")
        print(f"    Brier     : { {k: v for k, v in r['brier_scores'].items()} }")
        print()
        cm = np.array(r["confusion_matrix"])
        print("    Confusion matrix :")
        header = f"{'':>10}" + "".join(f"{c:>10}" for c in CLASSES)
        print(f"    {header}")
        for i, cls in enumerate(CLASSES):
            row = f"    {cls:>10}" + "".join(f"{cm[i, j]:>10}" for j in range(len(CLASSES)))
            print(row)
    print("\n" + "=" * 65)
    print(f"  Rapports sauvegardés dans : {OUT_DIR}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║  Évaluation Risk Service — rapport complet      ║")
    log.info("╚══════════════════════════════════════════════════╝")

    # 1. Données
    X_test, y_test, feature_cols = load_test_set()

    # 2. Modèles
    scaler, xgb_cal, xgb_raw, lgb, iso, shap_expl, meta = load_models()

    X_test_s = scaler.transform(X_test)
    y_test_np = y_test.to_numpy()

    # 3. Prédictions
    log.info("Prédictions en cours...")
    y_pred_xgb_cal  = xgb_cal.predict(X_test_s)
    y_proba_xgb_cal = xgb_cal.predict_proba(X_test_s)
    y_pred_xgb_raw  = xgb_raw.predict(X_test_s)
    y_proba_xgb_raw = xgb_raw.predict_proba(X_test_s)
    y_pred_lgb      = lgb.predict(X_test_s)
    y_proba_lgb     = lgb.predict_proba(X_test_s)
    iso_scores      = iso.decision_function(X_test_s)

    # 4. Métriques
    log.info("Calcul des métriques...")
    metrics_xgb_cal = compute_classification_metrics(y_test_np, y_pred_xgb_cal, y_proba_xgb_cal, "XGBoost Calibré")
    metrics_xgb_raw = compute_classification_metrics(y_test_np, y_pred_xgb_raw, y_proba_xgb_raw, "XGBoost Brut")
    metrics_lgb     = compute_classification_metrics(y_test_np, y_pred_lgb,     y_proba_lgb,     "LightGBM")

    # 5. Graphiques
    log.info("Génération des graphiques...")

    plot_class_distribution(y_test_np, OUT_DIR / "class_distribution.png")

    plot_confusion_matrix(
        np.array(metrics_xgb_cal["confusion_matrix"]),
        "Matrice de confusion — XGBoost Calibré", OUT_DIR / "confusion_xgb_calibrated.png"
    )
    plot_confusion_matrix(
        np.array(metrics_lgb["confusion_matrix"]),
        "Matrice de confusion — LightGBM", OUT_DIR / "confusion_lightgbm.png"
    )

    plot_roc_curves(y_test_np, y_proba_xgb_cal, "Courbes ROC — XGBoost Calibré",
                    OUT_DIR / "roc_xgb_calibrated.png")
    plot_roc_curves(y_test_np, y_proba_lgb, "Courbes ROC — LightGBM",
                    OUT_DIR / "roc_lightgbm.png")

    plot_calibration(y_test_np, y_proba_xgb_raw, y_proba_xgb_cal,
                     OUT_DIR / "calibration_curve.png")

    plot_shap_importance(shap_expl, X_test_s, feature_cols, OUT_DIR / "shap_importance.png")

    plot_anomaly_score(y_test_np, iso_scores, OUT_DIR / "anomaly_score_distribution.png")

    # 6. Rapport JSON
    report = {
        "evaluation_date": str(pd.Timestamp.now()),
        "model_version":   meta.get("version", "?"),
        "dataset":         str(DATA_PATH),
        "test_size":       int(len(y_test_np)),
        "models": {
            "xgboost_calibrated": metrics_xgb_cal,
            "xgboost_raw":        metrics_xgb_raw,
            "lightgbm":           metrics_lgb,
        },
        "anomaly_detection": {
            "contamination":       0.08,
            "n_anomalies_detected": int((iso.predict(X_test_s) == -1).sum()),
            "high_risk_in_anomalies_pct": round(
                float((iso.predict(X_test_s[y_test_np == 2]) == -1).mean()) * 100, 1
            ),
        },
        "outputs": [str(p) for p in sorted(OUT_DIR.glob("*.png"))] + [str(OUT_DIR / "report.json")],
    }
    save_report(report, OUT_DIR / "report.json")
    print_summary(report)


if __name__ == "__main__":
    main()
