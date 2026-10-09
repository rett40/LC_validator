"""
Evaluation du modele swift-ner (BERT Token Classification - SWIFT MT700)
========================================================================

Modele evalue :
  models/swift-ner/  -- BertForTokenClassification, 21 labels BIO
  Entites : APPLICANT, BENEFICIARY, ISSUING_BANK, AMOUNT, GOODS,
            INCOTERM, EXPIRY_DATE, ISSUE_DATE, SHIPMENT_DATE, PORT

Donnees :
  data/data_stati/ner_dataset.jsonl  (10 000 messages SWIFT)
  Le split test est reproduit a l'identique de fine_tune_ner.py :
    random.seed(42); shuffle; test_split=0.15  -> 1500 exemples de test

Metriques produites :
  - seqeval entity-level : precision / recall / f1 (micro, macro, weighted)
  - precision / recall / f1 PAR entite + support
  - matrice de confusion token-level (21 labels BIO)
  - matrice de confusion entity-level (10 entites + O)
  - accuracy token-level
  - rapport JSON structure

Figures :
  reports/confusion_matrix_bio.png
  reports/confusion_matrix_entity.png
  reports/per_entity_f1.png
  reports/precision_recall_scatter.png
  reports/entity_support_distribution.png

Usage :
    python evaluation/swift_ner/evaluate_swift_ner.py
    python evaluation/swift_ner/evaluate_swift_ner.py --max-samples 300 --batch-size 16
"""

import argparse
import json
import logging
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval-swift-ner")

# ── Paths ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent      # swift-validator-ia/
MODEL_DIR  = ROOT / "models" / "swift-ner"
DATA_PATH  = ROOT / "data" / "data_stati" / "ner_dataset.jsonl"
OUT_DIR    = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENTITY_TYPES = [
    "APPLICANT", "BENEFICIARY", "ISSUING_BANK", "AMOUNT", "GOODS",
    "INCOTERM", "EXPIRY_DATE", "ISSUE_DATE", "SHIPMENT_DATE", "PORT",
]
VALID_LABELS = set(ENTITY_TYPES)


# ── Reproduire le split test de fine_tune_ner.py ─────────────────────────
def load_test_split(data_path: Path, test_split: float = 0.15,
                    max_samples=None, full: bool = False):
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    log.info("Dataset complet : %d messages SWIFT (%s)", len(records), data_path.name)

    if full:
        test_records = records
        log.info("Mode --full : evaluation sur les %d exemples", len(test_records))
    else:
        random.seed(42)
        random.shuffle(records)
        split = int(len(records) * (1 - test_split))
        test_records = records[split:]
        log.info("Split test (seed=42, test=%.0f%%) : %d exemples",
                 test_split * 100, len(test_records))
    if max_samples:
        test_records = test_records[:max_samples]
    return test_records


# ── Construire les labels BIO gold alignes sur les tokens ────────────────
def char_labels_for(text: str, entities: list) -> list:
    char_labels = ["O"] * len(text)
    for ent in entities:
        label = ent.get("label", "")
        start = ent.get("start")
        end = ent.get("end")
        if start is None or end is None or label not in VALID_LABELS:
            continue
        start = max(0, start)
        end = min(end, len(text))
        if start >= end:
            continue
        char_labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            char_labels[i] = f"I-{label}"
    return char_labels


def main():
    ap = argparse.ArgumentParser(description="Evaluation swift-ner")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--dataset", default=str(DATA_PATH),
                    help="Fichier .jsonl a evaluer (defaut: dataset d'entrainement)")
    ap.add_argument("--full", action="store_true",
                    help="Evaluer tout le fichier (pas de split) - pour donnees fraiches")
    ap.add_argument("--tag", default="",
                    help="Suffixe pour les fichiers de sortie (ex: _fresh)")
    args = ap.parse_args()
    global OUT_DIR
    tag = args.tag

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from seqeval.metrics import (
            classification_report as seq_report,
            precision_score, recall_score, f1_score,
        )
    except ImportError as e:
        log.error("Dependances manquantes : pip install transformers torch seqeval")
        log.error(str(e))
        sys.exit(1)

    if not MODEL_DIR.exists():
        log.error("Modele introuvable : %s", MODEL_DIR)
        sys.exit(1)
    if not DATA_PATH.exists():
        log.error("Dataset introuvable : %s", DATA_PATH)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device : %s", device)

    log.info("Chargement du modele : %s", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR)).to(device)
    model.eval()
    id2label = model.config.id2label
    label_list = [id2label[i] for i in range(len(id2label))]
    log.info("Labels (%d) : %s", len(label_list), ", ".join(label_list))

    test_records = load_test_split(Path(args.dataset), args.test_split,
                                   args.max_samples, full=args.full)

    all_gold_seqs, all_pred_seqs = [], []      # pour seqeval (entity-level)
    flat_gold_tok, flat_pred_tok = [], []      # pour confusion matrix (token-level)

    batch_texts, batch_meta = [], []

    def flush():
        if not batch_texts:
            return
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1).cpu().numpy()

        for bi, (text, ents) in enumerate(batch_meta):
            char_lbls = char_labels_for(text, ents)
            gold_seq, pred_seq = [], []
            for ti, (s, e) in enumerate(offsets[bi].tolist()):
                if s == e:            # tokens speciaux / padding
                    continue
                gold = char_lbls[s] if s < len(char_lbls) else "O"
                pred = id2label[int(preds[bi][ti])]
                gold_seq.append(gold)
                pred_seq.append(pred)
                flat_gold_tok.append(gold)
                flat_pred_tok.append(pred)
            all_gold_seqs.append(gold_seq)
            all_pred_seqs.append(pred_seq)

        batch_texts.clear()
        batch_meta.clear()

    log.info("Inference en cours...")
    for rec in test_records:
        text = rec.get("text", "")
        if not text:
            continue
        batch_texts.append(text)
        batch_meta.append((text, rec.get("entities", [])))
        if len(batch_texts) >= args.batch_size:
            flush()
    flush()
    log.info("Inference terminee : %d sequences", len(all_gold_seqs))

    # ── Metriques entity-level (seqeval) ─────────────────────────────────
    micro_p = precision_score(all_gold_seqs, all_pred_seqs)
    micro_r = recall_score(all_gold_seqs, all_pred_seqs)
    micro_f = f1_score(all_gold_seqs, all_pred_seqs)
    rep_dict = seq_report(all_gold_seqs, all_pred_seqs, output_dict=True, zero_division=0)
    rep_txt = seq_report(all_gold_seqs, all_pred_seqs, zero_division=0)

    print("\n" + "=" * 70)
    print("  RAPPORT seqeval (entity-level)  — modele swift-ner")
    print("=" * 70)
    print(rep_txt)
    print(f"  Micro  -> P={micro_p:.4f}  R={micro_r:.4f}  F1={micro_f:.4f}")

    # ── Accuracy token-level ─────────────────────────────────────────────
    flat_gold_arr = np.array(flat_gold_tok)
    flat_pred_arr = np.array(flat_pred_tok)
    tok_acc = float((flat_gold_arr == flat_pred_arr).mean())
    log.info("Accuracy token-level : %.4f", tok_acc)

    # ── Matrice de confusion token-level (BIO) ───────────────────────────
    cm_bio = confusion_matrix(flat_gold_tok, flat_pred_tok, labels=label_list)
    plot_confusion(cm_bio, label_list, OUT_DIR / f"confusion_matrix_bio{tag}.png",
                   "Matrice de confusion token-level (BIO) - swift-ner")

    # ── Matrice de confusion entity-level (collapse B-/I-) ───────────────
    def collapse(lbl):
        return lbl.split("-", 1)[1] if lbl != "O" else "O"
    ent_order = ["O"] + ENTITY_TYPES
    g_ent = [collapse(x) for x in flat_gold_tok]
    p_ent = [collapse(x) for x in flat_pred_tok]
    cm_ent = confusion_matrix(g_ent, p_ent, labels=ent_order)
    plot_confusion(cm_ent, ent_order, OUT_DIR / f"confusion_matrix_entity{tag}.png",
                   "Matrice de confusion entity-level - swift-ner")

    # ── Figures F1 / PR / support ────────────────────────────────────────
    per_entity = {k: v for k, v in rep_dict.items()
                  if k in VALID_LABELS}
    plot_per_entity_f1(per_entity, OUT_DIR / f"per_entity_f1{tag}.png")
    plot_pr_scatter(per_entity, OUT_DIR / f"precision_recall_scatter{tag}.png")
    plot_support(per_entity, OUT_DIR / f"entity_support_distribution{tag}.png")

    # ── Rapport JSON ─────────────────────────────────────────────────────
    report = {
        "model": "swift-ner",
        "model_path": str(MODEL_DIR),
        "dataset": str(DATA_PATH),
        "n_test_examples": len(all_gold_seqs),
        "n_labels_bio": len(label_list),
        "labels_bio": label_list,
        "token_level_accuracy": tok_acc,
        "entity_level": {
            "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
            "macro": {
                "precision": rep_dict.get("macro avg", {}).get("precision"),
                "recall":    rep_dict.get("macro avg", {}).get("recall"),
                "f1":        rep_dict.get("macro avg", {}).get("f1-score"),
            },
            "weighted": {
                "precision": rep_dict.get("weighted avg", {}).get("precision"),
                "recall":    rep_dict.get("weighted avg", {}).get("recall"),
                "f1":        rep_dict.get("weighted avg", {}).get("f1-score"),
            },
            "per_entity": {
                k: {
                    "precision": v["precision"],
                    "recall":    v["recall"],
                    "f1":        v["f1-score"],
                    "support":   int(v["support"]),
                } for k, v in per_entity.items()
            },
        },
    }
    report["dataset"] = str(args.dataset)
    with open(OUT_DIR / f"report{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("Rapport JSON     -> %s", OUT_DIR / f"report{tag}.json")
    log.info("Figures (.png)   -> %s", OUT_DIR)
    log.info("Termine.")


# ── Helpers de plot ──────────────────────────────────────────────────────
def plot_confusion(cm, labels, path, title):
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.5)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
    ax.set_title(title, fontsize=11, fontweight="bold")
    thr = 0.5
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proportion (par ligne)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_per_entity_f1(per_entity, path):
    items = sorted(per_entity.items(), key=lambda kv: kv[1]["f1-score"])
    names = [k for k, _ in items]
    f1s = [v["f1-score"] for _, v in items]
    colors = ["#c0392b" if f < 0.7 else "#e67e22" if f < 0.9 else "#27ae60"
              for f in f1s]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, f1s, color=colors)
    for i, f in enumerate(f1s):
        ax.text(min(f + 0.01, 0.98), i, f"{f:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05); ax.set_xlabel("F1-score")
    ax.set_title("F1 par entite - swift-ner", fontsize=12, fontweight="bold")
    ax.axvline(0.9, color="grey", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_pr_scatter(per_entity, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for k, v in per_entity.items():
        ax.scatter(v["recall"], v["precision"], s=80, alpha=0.8)
        ax.annotate(k, (v["recall"], v["precision"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=0.8)
    ax.set_title("Precision vs Recall par entite - swift-ner",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_support(per_entity, path):
    items = sorted(per_entity.items(), key=lambda kv: kv[1]["support"], reverse=True)
    names = [k for k, _ in items]
    sup = [v["support"] for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, sup, color="#2980b9")
    for i, s in enumerate(sup):
        ax.text(i, s, str(s), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Support (nb d'entites gold)")
    ax.set_title("Distribution du support par entite (test) - swift-ner",
                 fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
