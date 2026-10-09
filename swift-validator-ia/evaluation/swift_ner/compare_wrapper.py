"""
compare_wrapper.py — Mesure l'effet des correctifs sans re-entrainement.

Compare, en metriques entity-level (seqeval), trois configurations sur un
jeu de test :
  - BRUT      : modele seul (argmax BIO)
  - WRAPPER   : normalisation AMOUNT + fusion GOODS

Usage :
  python evaluation/swift_ner/compare_wrapper.py \
      --dataset evaluation/swift_ner/fresh_mt700_perturb.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from swift_ner_wrapper import SwiftNERWrapper

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "models" / "swift-ner"


def spans_to_set(entities):
    """Set de tuples (label, start, end) pour comptage exact (entity-level)."""
    return {(e["label"], e["start"], e["end"]) for e in entities
            if e["label"] != "INCOTERM" or True}


def prf_by_label(gold_recs, pred_recs):
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for g, p in zip(gold_recs, pred_recs):
        gs = spans_to_set(g); ps = spans_to_set(p)
        for lab, s, e in ps:
            if (lab, s, e) in gs: tp[lab] += 1
            else: fp[lab] += 1
        for lab, s, e in gs:
            if (lab, s, e) not in ps: fn[lab] += 1
    labels = sorted(set(list(tp) + list(fp) + list(fn)))
    rows = {}
    TP = FP = FN = 0
    for lab in labels:
        t, f, n = tp[lab], fp[lab], fn[lab]
        TP += t; FP += f; FN += n
        prec = t / (t + f) if t + f else 0.0
        rec = t / (t + n) if t + n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows[lab] = (prec, rec, f1, t + n)
    micro_p = TP / (TP + FP) if TP + FP else 0.0
    micro_r = TP / (TP + FN) if TP + FN else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    return rows, (micro_p, micro_r, micro_f)


def gold_spans(rec):
    valid = {"APPLICANT","BENEFICIARY","ISSUING_BANK","AMOUNT","GOODS",
             "INCOTERM","EXPIRY_DATE","ISSUE_DATE","SHIPMENT_DATE","PORT"}
    return [e for e in rec.get("entities", []) if e["label"] in valid]


def print_table(title, rows, micro):
    print(f"\n=== {title} ===")
    print(f"{'ENTITE':<14}{'P':>7}{'R':>7}{'F1':>7}{'sup':>7}")
    for lab in sorted(rows):
        p, r, f, s = rows[lab]
        print(f"{lab:<14}{p:>7.2f}{r:>7.2f}{f:>7.2f}{s:>7d}")
    print(f"{'MICRO':<14}{micro[0]:>7.2f}{micro[1]:>7.2f}{micro[2]:>7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    recs = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            try: recs.append(json.loads(line))
            except Exception: pass
    if args.max:
        recs = recs[:args.max]
    print(f"Dataset : {args.dataset}  ({len(recs)} exemples)")

    w = SwiftNERWrapper(MODEL_DIR)

    golds = [gold_spans(r) for r in recs]
    raw_preds, wrap_preds = [], []
    for r in recs:
        t = r["text"]
        raw_preds.append(w.predict(t, fix_amount=False, fix_goods=False))
        wrap_preds.append(w.predict(t, fix_amount=True, fix_goods=True))

    r1, m1 = prf_by_label(golds, raw_preds)
    r2, m2 = prf_by_label(golds, wrap_preds)
    print_table("BRUT (modele seul)", r1, m1)
    print_table("WRAPPER (normalisation AMOUNT + fusion GOODS)", r2, m2)
    print(f"\nGain micro F1 : {m1[2]:.3f} -> {m2[2]:.3f}  "
          f"(+{(m2[2]-m1[2])*100:.1f} pts)")


if __name__ == "__main__":
    main()
