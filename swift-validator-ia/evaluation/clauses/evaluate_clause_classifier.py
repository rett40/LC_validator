"""
Évaluation du classifieur de clauses SWIFT (SwiftClauseClassifier v2.0)
=========================================================================
Métriques produites :
  [A] Classification du TYPE de clause (5 types)
      - Accuracy, F1, Precision, Recall par type
      - Matrice de confusion 5×5

  [B] Classification de l'IMPORTANCE de clause (critical/high/medium/low/info)
      - Accuracy, F1, Precision, Recall par niveau
      - Matrice de confusion 5×5

  [C] Classification du RISK LEVEL (high/medium/normal)
      - Accuracy, F1 par niveau
      - Distribution du risk_level par type de clause
      - Matrice de confusion 3×3

  [D] Diagnostics transversaux
      - Distribution des méthodes utilisées (rule/embedding/scoring)
      - Distribution des niveaux de confiance (correct vs incorrect)
      - Analyse des confusions les plus fréquentes
      - Rapport JSON structuré

Usage :
    cd swift-validator-ia
    python evaluation/clauses/evaluate_clause_classifier.py
"""

import json
import logging
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT         = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT / "data/data_stati/clause_classification.jsonl"
OUT_DIR      = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# RÉFÉRENTIELS
# ─────────────────────────────────────────────────────────────────────────────

CLAUSE_TYPES = [
    "document_condition", "shipment_condition", "payment_condition",
    "compliance_condition", "tolerance_condition",
]
IMPORTANCE_LEVELS = ["high", "medium", "normal"]
RISK_LEVELS       = ["high", "medium", "normal"]

SHORT_TYPE = {
    "document_condition":   "document",
    "shipment_condition":   "shipment",
    "payment_condition":    "payment",
    "compliance_condition": "compliance",
    "tolerance_condition":  "tolerance",
}
TYPE_COLORS = {
    "document_condition":   "#3498db",
    "shipment_condition":   "#2ecc71",
    "payment_condition":    "#e74c3c",
    "compliance_condition": "#9b59b6",
    "tolerance_condition":  "#f39c12",
}
IMPORTANCE_COLORS = {
    "high":   "#e74c3c",
    "medium": "#f39c12",
    "normal": "#2ecc71",
}
RISK_COLORS = {
    "high":   "#e74c3c",
    "medium": "#f39c12",
    "normal": "#2ecc71",
}


def _derive_risk_level(importance: str, clause_type: str) -> str:
    """Meme logique que SwiftClauseClassifier._enrich() — 3 niveaux."""
    if importance == "high" and clause_type in (
        "compliance_condition", "payment_condition", "document_condition"
    ):
        return "high"
    if importance == "high" or (
        importance == "medium" and clause_type in ("compliance_condition", "payment_condition")
    ):
        return "medium"
    return "normal"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DU DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    log.info("Chargement dataset : %s", DATASET_PATH)
    records = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t   = obj.get("type", "").strip()
                x   = obj.get("text", "").strip()
                imp = obj.get("importance", "medium").strip()
                if t and x and t in CLAUSE_TYPES:
                    imp = imp if imp in IMPORTANCE_LEVELS else "medium"
                    records.append({
                        "text":       x,
                        "type":       t,
                        "importance": imp,
                        "risk_level": _derive_risk_level(imp, t),
                    })
            except json.JSONDecodeError:
                pass

    log.info("Dataset : %d exemples", len(records))
    log.info("  Types      : %s", dict(Counter(r["type"] for r in records)))
    log.info("  Importance : %s", dict(Counter(r["importance"] for r in records)))
    log.info("  Risk level : %s", dict(Counter(r["risk_level"] for r in records)))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2. ÉVALUATION DU CLASSIFIEUR
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(records_test: list) -> dict:
    from services.swift_nlp.clauses.swift_clause_classifier import SwiftClauseClassifier

    log.info("Initialisation SwiftClauseClassifier (enable_embeddings=True)...")
    clf = SwiftClauseClassifier(enable_embeddings=True)

    texts = [r["text"] for r in records_test]
    log.info("Classification de %d clauses...", len(texts))
    results = clf.classify_batch(texts)

    return {
        "true_type":       [r["type"]       for r in records_test],
        "true_importance": [r["importance"] for r in records_test],
        "true_risk":       [r["risk_level"] for r in records_test],
        "pred_type":       [res.type        for res in results],
        "pred_importance": [res.importance  for res in results],
        "pred_risk":       [res.risk_level  for res in results],
        "confidence":      [res.confidence  for res in results],
        "method":          [res.method      for res in results],
        "texts":           texts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CALCUL DES MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, all_labels: list, section: str) -> dict:
    present = [l for l in all_labels if l in set(y_true) | set(y_pred)]
    acc    = accuracy_score(y_true, y_pred)
    f1_w   = f1_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)
    f1_m   = f1_score(y_true, y_pred, average="macro",    labels=present, zero_division=0)
    prec_w = precision_score(y_true, y_pred, average="weighted", labels=present, zero_division=0)
    rec_w  = recall_score(y_true, y_pred,   average="weighted", labels=present, zero_division=0)
    report = classification_report(y_true, y_pred, labels=present, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=present)

    log.info("── [%s] ──", section)
    log.info("Accuracy   : %.4f", acc)
    log.info("F1 (w/m)   : %.4f / %.4f", f1_w, f1_m)
    log.info("\n%s", classification_report(y_true, y_pred, labels=present, zero_division=0))

    return {
        "accuracy":           round(acc,    4),
        "f1_weighted":        round(f1_w,   4),
        "f1_macro":           round(f1_m,   4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted":    round(rec_w,  4),
        "per_class": {
            lbl: {
                "precision": round(report.get(lbl, {}).get("precision", 0), 4),
                "recall":    round(report.get(lbl, {}).get("recall",    0), 4),
                "f1":        round(report.get(lbl, {}).get("f1-score",  0), 4),
                "support":   int(report.get(lbl, {}).get("support",     0)),
            }
            for lbl in present
        },
        "confusion_matrix": cm.tolist(),
        "confusion_labels": present,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion(cm: np.ndarray, labels: list, title: str, short_fn, colors, out_path: Path):
    short = [short_fn.get(l, l) for l in labels]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), max(4, len(labels) * 1.2)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=short, yticklabels=short,
        ylabel="Réel", xlabel="Prédit", title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=10,
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("%s → %s", title, out_path)


def plot_type_confusion(metrics: dict, out_path: Path):
    _plot_confusion(
        np.array(metrics["confusion_matrix"]), metrics["confusion_labels"],
        "Matrice de confusion — Type de clause",
        SHORT_TYPE, TYPE_COLORS, out_path,
    )


def plot_importance_confusion(metrics: dict, out_path: Path):
    _plot_confusion(
        np.array(metrics["confusion_matrix"]), metrics["confusion_labels"],
        "Matrice de confusion — Importance",
        {}, IMPORTANCE_COLORS, out_path,
    )


def plot_risk_confusion(metrics: dict, out_path: Path):
    _plot_confusion(
        np.array(metrics["confusion_matrix"]), metrics["confusion_labels"],
        "Matrice de confusion — Risk Level",
        {}, RISK_COLORS, out_path,
    )


def plot_f1_bars(metrics: dict, labels: list, colors: dict, title: str, out_path: Path):
    present = [l for l in labels if l in metrics["per_class"]]
    x    = np.arange(len(present))
    w    = 0.25
    prec = [metrics["per_class"][l]["precision"] for l in present]
    rec  = [metrics["per_class"][l]["recall"]     for l in present]
    f1s  = [metrics["per_class"][l]["f1"]         for l in present]
    sup  = [metrics["per_class"][l]["support"]     for l in present]

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 1.5), 5))
    ax.bar(x - w, prec, w, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x,     rec,  w, label="Recall",    color="#2ecc71", alpha=0.85)
    ax.bar(x + w, f1s,  w, label="F1",        color="#e74c3c", alpha=0.85)
    for xi, v, s in zip(x + w, f1s, sup):
        ax.text(xi, v + 0.01, f"{v:.2f}\n(n={s})", ha="center", va="bottom", fontsize=8)
    ax.set(xticks=x, xticklabels=present, ylim=[0, 1.12], title=title, ylabel="Score")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("%s → %s", title, out_path)


def plot_risk_by_type(pred_types, pred_risks, out_path: Path):
    """Heatmap du risk_level prédit par type de clause."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = {}
    for t in CLAUSE_TYPES:
        risk_counts = Counter(r for ty, r in zip(pred_types, pred_risks) if ty == t)
        total = sum(risk_counts.values()) or 1
        data[t] = {lv: risk_counts.get(lv, 0) / total for lv in RISK_LEVELS}

    matrix = np.array([[data[t][lv] for lv in RISK_LEVELS] for t in CLAUSE_TYPES])
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Proportion")
    ax.set(
        xticks=range(len(RISK_LEVELS)), yticks=range(len(CLAUSE_TYPES)),
        xticklabels=RISK_LEVELS,
        yticklabels=[SHORT_TYPE[t] for t in CLAUSE_TYPES],
        title="Risk Level prédit par type de clause (proportions)",
        xlabel="Risk Level", ylabel="Type de clause",
    )
    for i in range(len(CLAUSE_TYPES)):
        for j in range(len(RISK_LEVELS)):
            ax.text(j, i, f"{matrix[i,j]:.0%}", ha="center", va="center",
                    fontsize=10, color="black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Risk par type → %s", out_path)


def plot_importance_by_type(true_types, true_importances, pred_importances, out_path: Path):
    """Comparaison importance réelle vs prédite par type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, importances, title in [
        (axes[0], true_importances, "Importance réelle (dataset)"),
        (axes[1], pred_importances, "Importance prédite (classifieur)"),
    ]:
        data = {}
        for t in CLAUSE_TYPES:
            imp_counts = Counter(i for ty, i in zip(true_types, importances) if ty == t)
            total = sum(imp_counts.values()) or 1
            data[t] = {lv: imp_counts.get(lv, 0) / total for lv in IMPORTANCE_LEVELS}

        x  = np.arange(len(CLAUSE_TYPES))
        w  = 0.15
        for k, lv in enumerate(IMPORTANCE_LEVELS):
            vals = [data[t][lv] for t in CLAUSE_TYPES]
            ax.bar(x + k * w - 0.3, vals, w, label=lv,
                   color=IMPORTANCE_COLORS[lv], alpha=0.85)
        ax.set(xticks=x, xticklabels=[SHORT_TYPE[t] for t in CLAUSE_TYPES],
               ylim=[0, 1.0], title=title, ylabel="Proportion")
        ax.legend(fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.suptitle("Distribution d'importance par type — réel vs prédit", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Importance par type → %s", out_path)


def plot_method_distribution(methods: list, out_path: Path):
    counts  = Counter(methods)
    labels  = list(counts.keys())
    values  = [counts[m] for m in labels]
    colors  = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    total = sum(values)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val}\n({val/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set(title="Méthode de classification utilisée", ylabel="Nombre de clauses")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Méthodes → %s", out_path)


def plot_confidence_vs_accuracy(confidences, y_true, y_pred, out_path: Path):
    correct = [c for c, t, p in zip(confidences, y_true, y_pred) if t == p]
    wrong   = [c for c, t, p in zip(confidences, y_true, y_pred) if t != p]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(correct, bins=20, alpha=0.7, label=f"Correct  ({len(correct)})",  color="#2ecc71", density=True)
    ax.hist(wrong,   bins=20, alpha=0.7, label=f"Incorrect ({len(wrong)})", color="#e74c3c", density=True)
    ax.axvline(0.70, color="black", ls="--", lw=1, label="Seuil 0.70")
    ax.set(xlabel="Confiance", ylabel="Densité",
           title="Confiance — type correct vs incorrect")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Confiance type → %s", out_path)


def plot_dataset_distribution(records: list, out_path: Path):
    types  = [r["type"] for r in records]
    counts = Counter(types)
    names  = [SHORT_TYPE[t] for t in CLAUSE_TYPES if t in counts]
    values = [counts[t] for t in CLAUSE_TYPES if t in counts]
    colors = [TYPE_COLORS[t] for t in CLAUSE_TYPES if t in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bars = ax1.bar(names, values, color=colors, alpha=0.85, edgecolor="white")
    total = sum(values)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val} ({val/total*100:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax1.set(title="Distribution par type", ylabel="Nombre d'exemples")
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")
    ax2.pie(values, labels=names, colors=colors, autopct="%1.0f%%", startangle=140)
    ax2.set_title("Répartition proportionnelle")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_risk_distribution_global(true_risks, pred_risks, out_path: Path):
    """Comparaison globale risk_level réel vs prédit."""
    x   = np.arange(len(RISK_LEVELS))
    w   = 0.35
    t_c = [Counter(true_risks).get(lv, 0) for lv in RISK_LEVELS]
    p_c = [Counter(pred_risks).get(lv, 0) for lv in RISK_LEVELS]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, t_c, w, label="Réel (dataset)",  color=[RISK_COLORS[l] for l in RISK_LEVELS], alpha=0.9)
    ax.bar(x + w / 2, p_c, w, label="Prédit (classif.)",
           color=[RISK_COLORS[l] for l in RISK_LEVELS], alpha=0.5, hatch="//", edgecolor="grey")
    for xi, tv, pv in zip(x, t_c, p_c):
        ax.text(xi - w/2, tv + 0.3, str(tv), ha="center", va="bottom", fontsize=9)
        ax.text(xi + w/2, pv + 0.3, str(pv), ha="center", va="bottom", fontsize=9)
    ax.set(xticks=x, xticklabels=RISK_LEVELS,
           title="Distribution du Risk Level — réel vs prédit",
           ylabel="Nombre de clauses")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Risk distribution globale → %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANALYSE DES ERREURS
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(preds: dict) -> list[dict]:
    errors = []
    for text, tt, pt, ti, pi, tr, pr, conf, meth in zip(
        preds["texts"], preds["true_type"], preds["pred_type"],
        preds["true_importance"], preds["pred_importance"],
        preds["true_risk"], preds["pred_risk"],
        preds["confidence"], preds["method"],
    ):
        is_type_err  = tt != pt
        is_imp_err   = ti != pi
        is_risk_err  = tr != pr
        if is_type_err or is_imp_err or is_risk_err:
            errors.append({
                "text":           text[:100],
                "type_error":     is_type_err,
                "imp_error":      is_imp_err,
                "risk_error":     is_risk_err,
                "true_type":      tt,  "pred_type":       pt,
                "true_importance":ti,  "pred_importance": pi,
                "true_risk":      tr,  "pred_risk":       pr,
                "confidence":     round(conf, 3),
                "method":         meth,
            })
    return errors


def print_error_summary(errors: list, preds: dict):
    n_total     = len(preds["texts"])
    n_type_err  = sum(1 for e in errors if e["type_error"])
    n_imp_err   = sum(1 for e in errors if e["imp_error"])
    n_risk_err  = sum(1 for e in errors if e["risk_error"])

    print(f"\n  -- Analyse des erreurs --")
    print(f"  Total clauses   : {n_total}")
    print(f"  Erreurs type    : {n_type_err} ({n_type_err/n_total*100:.1f}%)")
    print(f"  Erreurs importance : {n_imp_err} ({n_imp_err/n_total*100:.1f}%)")
    print(f"  Erreurs risk_level : {n_risk_err} ({n_risk_err/n_total*100:.1f}%)")

    # Top confusions type
    type_conf = Counter(
        (e["true_type"], e["pred_type"]) for e in errors if e["type_error"]
    )
    if type_conf:
        print(f"\n  Top confusions TYPE :")
        for (tt, pt), cnt in type_conf.most_common(5):
            print(f"    {SHORT_TYPE[tt]:>11} -> {SHORT_TYPE[pt]:>11}  ({cnt}x)")

    # Top confusions importance
    imp_conf = Counter(
        (e["true_importance"], e["pred_importance"]) for e in errors if e["imp_error"]
    )
    if imp_conf:
        print(f"\n  Top confusions IMPORTANCE :")
        for (ti, pi), cnt in imp_conf.most_common(5):
            print(f"    {ti:>10} -> {pi:>10}  ({cnt}x)")

    # Top confusions risk_level
    risk_conf = Counter(
        (e["true_risk"], e["pred_risk"]) for e in errors if e["risk_error"]
    )
    if risk_conf:
        print(f"\n  Top confusions RISK LEVEL :")
        for (tr, pr), cnt in risk_conf.most_common(5):
            print(f"    {tr:>8} -> {pr:>8}  ({cnt}x)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 56)
    log.info("  Evaluation Clause Classifier - rapport complet")
    log.info("=" * 56)

    # 1. Dataset
    records = load_dataset()
    plot_dataset_distribution(records, OUT_DIR / "dataset_distribution.png")

    # 2. Split 80/20 stratifié sur le type
    from sklearn.model_selection import train_test_split
    types_all = [r["type"] for r in records]
    idx_train, idx_test = train_test_split(
        range(len(records)), test_size=0.20, stratify=types_all, random_state=42
    )
    records_test = [records[i] for i in idx_test]
    log.info("Split → train: %d, test: %d", len(idx_train), len(records_test))

    # 3. Évaluation du classifieur
    preds = evaluate_classifier(records_test)

    # ── A. TYPE ──────────────────────────────────────────────────────────────
    log.info("=== A. Classification du TYPE ===")
    metrics_type = compute_metrics(
        preds["true_type"], preds["pred_type"], CLAUSE_TYPES, "TYPE"
    )
    plot_type_confusion(metrics_type, OUT_DIR / "confusion_type.png")
    plot_f1_bars(metrics_type, CLAUSE_TYPES, TYPE_COLORS,
                 "F1 / Precision / Recall — Type de clause",
                 OUT_DIR / "f1_type.png")

    # ── B. IMPORTANCE ─────────────────────────────────────────────────────────
    log.info("=== B. Classification de l'IMPORTANCE ===")
    metrics_imp = compute_metrics(
        preds["true_importance"], preds["pred_importance"],
        IMPORTANCE_LEVELS, "IMPORTANCE"
    )
    plot_importance_confusion(metrics_imp, OUT_DIR / "confusion_importance.png")
    plot_f1_bars(metrics_imp, IMPORTANCE_LEVELS, IMPORTANCE_COLORS,
                 "F1 / Precision / Recall — Importance",
                 OUT_DIR / "f1_importance.png")
    plot_importance_by_type(
        preds["true_type"], preds["true_importance"], preds["pred_importance"],
        OUT_DIR / "importance_by_type.png",
    )

    # ── C. RISK LEVEL ─────────────────────────────────────────────────────────
    log.info("=== C. Classification du RISK LEVEL ===")
    metrics_risk = compute_metrics(
        preds["true_risk"], preds["pred_risk"], RISK_LEVELS, "RISK LEVEL"
    )
    plot_risk_confusion(metrics_risk, OUT_DIR / "confusion_risk_level.png")
    plot_f1_bars(metrics_risk, RISK_LEVELS, RISK_COLORS,
                 "F1 / Precision / Recall — Risk Level",
                 OUT_DIR / "f1_risk_level.png")
    plot_risk_by_type(preds["pred_type"], preds["pred_risk"],
                      OUT_DIR / "risk_by_type_heatmap.png")
    plot_risk_distribution_global(preds["true_risk"], preds["pred_risk"],
                                  OUT_DIR / "risk_distribution_comparison.png")

    # ── D. DIAGNOSTICS ────────────────────────────────────────────────────────
    plot_method_distribution(preds["method"], OUT_DIR / "method_distribution.png")
    plot_confidence_vs_accuracy(preds["confidence"], preds["true_type"],
                                preds["pred_type"], OUT_DIR / "confidence_type.png")

    # ── Analyse erreurs ───────────────────────────────────────────────────────
    errors = error_analysis(preds)
    print_error_summary(errors, preds)

    # ── Rapport JSON ──────────────────────────────────────────────────────────
    import pandas as pd
    report = {
        "evaluation_date": str(pd.Timestamp.now()),
        "classifier":      "SwiftClauseClassifier v2.0",
        "dataset":         str(DATASET_PATH),
        "n_total":         len(records),
        "n_train":         len(idx_train),
        "n_test":          len(records_test),
        "A_type_classification":       metrics_type,
        "B_importance_classification": metrics_imp,
        "C_risk_level_classification": metrics_risk,
        "diagnostics": {
            "method_distribution":      dict(Counter(preds["method"])),
            "avg_confidence":           round(float(np.mean(preds["confidence"])), 4),
            "low_confidence_pct":       round(float(np.mean(np.array(preds["confidence"]) < 0.60)) * 100, 1),
            "n_errors_type":            sum(1 for e in errors if e["type_error"]),
            "n_errors_importance":      sum(1 for e in errors if e["imp_error"]),
            "n_errors_risk_level":      sum(1 for e in errors if e["risk_error"]),
        },
        "error_details": errors[:20],
        "outputs": [str(p) for p in sorted(OUT_DIR.glob("*.png"))],
    }
    out_json = OUT_DIR / "report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("Rapport JSON → %s", out_json)

    # ── Résumé console ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RÉSUMÉ ÉVALUATION CLAUSE CLASSIFIER")
    print("=" * 65)
    for section, m, label in [
        ("A — TYPE",       metrics_type, CLAUSE_TYPES),
        ("B — IMPORTANCE", metrics_imp,  IMPORTANCE_LEVELS),
        ("C — RISK LEVEL", metrics_risk, RISK_LEVELS),
    ]:
        print(f"\n  [{section}]")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    F1 (w/m)  : {m['f1_weighted']:.4f} / {m['f1_macro']:.4f}")
        present = [l for l in label if l in m["per_class"]]
        print(f"    {'Classe':<24} {'Precision':>10} {'Recall':>8} {'F1':>6} {'n':>5}")
        print("    " + "-" * 55)
        for lbl in present:
            pc = m["per_class"][lbl]
            print(f"    {lbl:<24} {pc['precision']:>10.3f} {pc['recall']:>8.3f} {pc['f1']:>6.3f} {pc['support']:>5}")
    print()
    print(f"  Méthodes : { dict(Counter(preds['method'])) }")
    print(f"  Rapports -> {OUT_DIR}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
