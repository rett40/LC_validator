"""
perf_diff_harness.py
====================
Garde-fou iso-précision pour l'optimisation LayoutLMv3 (release 2).

Deux mesures, mêmes échantillons que le test set :

  1. SNAPSHOT EXTRACTEUR  — sortie complète de extract_from_pil_image()
     (fields, confidences, sources, table_rows, coverage) par échantillon.
     => avant/après une optim, ces snapshots doivent être IDENTIQUES.
        Sorties identiques  ⇒  F1 trivialement identique (garantie forte).

  2. F1 MODÈLE BRUT (seqeval) — via le CHEMIN DE PRODUCTION _predict_segment()
     sur le document entier, comparé aux ner_tags gold.
     => mesure de précision « modèle » que le projet suit déjà.

Usage :
  python evaluation/release2/perf_diff_harness.py snapshot --tag baseline [--n 30]
  # ... appliquer une optim ...
  python evaluation/release2/perf_diff_harness.py snapshot --tag phase1 [--n 30]
  python evaluation/release2/perf_diff_harness.py compare --a baseline --b phase1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent          # swift-validator-ia/
import sys
sys.path.insert(0, str(ROOT))

DATA_COMPLEX = ROOT / "release2" / "data" / "complex"
OUT_DIR = Path(__file__).resolve().parent / "perf_snapshots"
OUT_DIR.mkdir(exist_ok=True)

DATASET_DIRS = {
    "bl":           DATA_COMPLEX / "dataset_bl",
    "invoice":      DATA_COMPLEX / "dataset_invoice",
    "packing_list": DATA_COMPLEX / "dataset_packing_list",
}


def _remap_image_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    parts = p.parts
    try:
        idx = next(i for i in range(len(parts) - 1, -1, -1)
                   if parts[i].lower() == "complex")
        return DATA_COMPLEX / Path(*parts[idx + 1:])
    except StopIteration:
        return p


def _get_extractor(doc_type: str):
    if doc_type == "bl":
        from release2.services.bl_extractor import BLExtractor
        return BLExtractor()
    if doc_type == "invoice":
        from release2.services.invoice_extractor import InvoiceExtractor
        return InvoiceExtractor()
    from release2.services.pl_extractor import PLExtractor
    return PLExtractor()


def _load_id2label(doc_type: str) -> dict[int, str]:
    f = DATASET_DIRS[doc_type] / "id2label.json"
    raw = json.loads(f.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _load_samples(doc_type: str, n: int) -> list[dict]:
    f = DATASET_DIRS[doc_type] / "test" / "data.jsonl"
    out = []
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def cmd_snapshot(tag: str, n: int) -> None:
    from PIL import Image
    from seqeval.metrics import f1_score as seq_f1, precision_score as seq_p, recall_score as seq_r

    snapshot: dict = {}
    timings: dict = {}
    f1_report: dict = {}

    for doc_type in ("bl", "invoice", "packing_list"):
        print(f"\n=== {doc_type} ===")
        id2label = _load_id2label(doc_type)
        samples = _load_samples(doc_type, n)
        extractor = _get_extractor(doc_type)

        doc_snap: dict = {}
        all_true: list[list[str]] = []
        all_pred: list[list[str]] = []
        t0 = time.perf_counter()

        for s in samples:
            sid = s["doc_id"]
            words = s["words"]
            bboxes = s["bboxes"]
            img_path = _remap_image_path(s["image_path"])
            img = (Image.open(img_path).convert("RGB")
                   if img_path.exists() else Image.new("RGB", (1240, 1754), (255, 255, 255)))

            # (1) snapshot extracteur (sortie complète)
            res = extractor.extract_from_pil_image(image=img, words=words, bboxes=bboxes)
            doc_snap[sid] = {
                "fields":      res.get("fields", {}),
                "confidences": {k: round(v, 6) for k, v in res.get("field_confidences", {}).items()},
                "sources":     res.get("field_source", {}),
                "table_rows":  res.get("table_rows", []),
                "coverage":    res.get("coverage", ""),
            }

            # (2) F1 brut via chemin production _predict_segment sur doc entier
            labels, _ = extractor._predict_segment(img, words, bboxes)
            true = [id2label.get(t, "O") for t in s["ner_tags"]]
            m = min(len(labels), len(true))
            if m:
                all_pred.append(labels[:m])
                all_true.append(true[:m])

        dt = round(time.perf_counter() - t0, 2)
        timings[doc_type] = {"total_s": dt, "per_doc_s": round(dt / max(1, len(samples)), 3),
                             "n": len(samples)}
        if all_true:
            f1_report[doc_type] = {
                "f1":        round(seq_f1(all_true, all_pred), 6),
                "precision": round(seq_p(all_true, all_pred), 6),
                "recall":    round(seq_r(all_true, all_pred), 6),
            }
            print(f"  F1={f1_report[doc_type]['f1']}  "
                  f"P={f1_report[doc_type]['precision']}  "
                  f"R={f1_report[doc_type]['recall']}  ({dt}s, {len(samples)} docs)")
        snapshot[doc_type] = doc_snap

    out = {"tag": tag, "n": n, "f1": f1_report, "timings": timings, "snapshot": snapshot}
    path = OUT_DIR / f"snap_{tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> snapshot ecrit : {path}")
    print("F1:", json.dumps(f1_report, indent=2))
    print("timings:", json.dumps(timings, indent=2))


def cmd_compare(a: str, b: str) -> None:
    pa = json.loads((OUT_DIR / f"snap_{a}.json").read_text(encoding="utf-8"))
    pb = json.loads((OUT_DIR / f"snap_{b}.json").read_text(encoding="utf-8"))

    print(f"\n{'='*60}\nCOMPARE  {a}  vs  {b}\n{'='*60}")

    # F1 + timing summary
    print("\n-- F1 (modele brut) --")
    for dt in pa.get("f1", {}):
        fa = pa["f1"].get(dt, {})
        fb = pb["f1"].get(dt, {})
        df1 = round(fb.get("f1", 0) - fa.get("f1", 0), 6)
        flag = "OK" if df1 == 0 else ("UP" if df1 > 0 else "DOWN-REGRESSION")
        print(f"  {dt:14s} {a}={fa.get('f1')}  {b}={fb.get('f1')}  delta={df1}  {flag}")

    print("\n-- temps / doc (s) --")
    for dt in pa.get("timings", {}):
        ta = pa["timings"][dt]["per_doc_s"]
        tb = pb["timings"].get(dt, {}).get("per_doc_s", 0)
        speedup = round(ta / tb, 2) if tb else 0
        print(f"  {dt:14s} {a}={ta}s  {b}={tb}s  speedup×{speedup}")

    # Snapshot diff (la vraie garantie iso-précision)
    # Les valeurs (fields/sources/coverage/table_rows) doivent être STRICTEMENT
    # identiques. Les confidences sont des floats issus de matmul CPU : une tolérance
    # CONF_TOL absorbe le bruit d'arrondi (~1e-6) qui ne change aucune décision.
    CONF_TOL = 1e-4
    print(f"\n-- diff sorties extracteur (valeurs exactes ; conf tol={CONF_TOL}) --")
    total_diff = 0
    max_conf_noise = 0.0
    for dt, docs_a in pa["snapshot"].items():
        docs_b = pb["snapshot"].get(dt, {})
        for sid, va in docs_a.items():
            vb = docs_b.get(sid)
            if vb is None:
                total_diff += 1
                print(f"  DIFF {dt}/{sid} : ABSENT dans {b}")
                continue
            diffs = []
            for key in ("fields", "sources", "table_rows", "coverage"):
                if va.get(key) != vb.get(key):
                    diffs.append(key)
            # confidences : comparaison avec tolérance
            ca, cb = va.get("confidences", {}), vb.get("confidences", {})
            if set(ca) != set(cb):
                diffs.append("confidences:keys")
            else:
                for k in ca:
                    d = abs(ca[k] - cb[k])
                    max_conf_noise = max(max_conf_noise, d)
                    if d > CONF_TOL:
                        diffs.append(f"confidences:{k}(d={d:.2e})")
            if diffs:
                total_diff += 1
                if total_diff <= 10:
                    print(f"  DIFF {dt}/{sid} : {diffs}")
    print(f"  (bruit flottant max sur confidences : {max_conf_noise:.2e})")
    if total_diff == 0:
        print("  [OK] AUCUNE difference - iso-precision PROUVEE sur tous les echantillons")
    else:
        print(f"  [X] {total_diff} echantillon(s) divergent(s) - NE PAS merger l'optim")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("snapshot")
    sp.add_argument("--tag", required=True)
    sp.add_argument("--n", type=int, default=30)
    cp = sub.add_parser("compare")
    cp.add_argument("--a", required=True)
    cp.add_argument("--b", required=True)
    args = ap.parse_args()

    if args.cmd == "snapshot":
        cmd_snapshot(args.tag, args.n)
    else:
        cmd_compare(args.a, args.b)


if __name__ == "__main__":
    main()
