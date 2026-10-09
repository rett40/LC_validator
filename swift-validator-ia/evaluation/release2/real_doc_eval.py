"""
real_doc_eval.py
================
Évaluation NIVEAU CHAMP sur de VRAIS documents (dossier gold/), par opposition
à evaluate_release2_models.py qui mesure le NER token-level sur du synthétique.

- Gold auto-parsé depuis la structure "LABEL:\\nvaleur" des PDF réels.
- Lance le pipeline COMPLET (DocumentPipeline) sur chaque PDF.
- Compare valeur extraite vs gold (normalisation texte/nombre/date).
- Sort : corrects / faux (mauvaise valeur) / manquants / FAUX POSITIFS (champ
  extrait alors qu'absent du document) + précision/rappel par type de doc.

Usage :
  python evaluation/release2/real_doc_eval.py --gold C:/Users/Aziz/Desktop/projet_lc/gold
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.ERROR)

import fitz

# label (tel qu'imprimé) -> nom de champ pipeline
GOLD_LABELS = {
    "bl": {
        "B/L No.": "bl_number", "SHIPPER / EXPORTER": "shipper", "CONSIGNEE": "consignee",
        "NOTIFY PARTY": "notify_party", "VESSEL NAME": "vessel", "VOYAGE NO.": "voyage",
        "PLACE OF RECEIPT": "place_of_receipt", "PORT OF LOADING": "port_of_loading",
        "PORT OF DISCHARGE": "port_of_discharge", "PLACE OF DELIVERY": "place_of_delivery",
        "ETA": "eta", "FREIGHT": "freight_terms", "INCOTERM": "incoterm",
        "DATE OF ISSUE": "issue_date", "NO. OF ORIGINALS": "number_of_originals",
        "CONTAINER NO.": "container_number", "SEAL NO.": "seal_number",
        "CONTAINER TYPE": "container_type", "TOTAL GROSS WEIGHT": "gross_weight_kg",
        "TOTAL NET WEIGHT": "net_weight_kg",
    },
    "invoice": {
        "Invoice No.": "invoice_number", "SELLER / EXPORTER": "seller",
        "BUYER / IMPORTER": "buyer", "DATE OF ISSUE": "invoice_date",
        "CURRENCY": "currency", "INCOTERM": "incoterm", "PAYMENT TERMS": "payment_terms",
        "REF. B/L NO.": "bl_reference", "PORT OF LOADING": "port_of_loading",
        "PORT OF DISCHARGE": "port_of_discharge", "TOTAL": "total_amount",
    },
    "packing_list": {
        "P/L No.": "pl_number", "SHIPPER / EXPORTER": "shipper",
        "CONSIGNEE / BUYER": "consignee", "REF. INVOICE NO.": "invoice_reference",
        "REF. B/L NO.": "bl_reference", "DATE OF ISSUE": "pl_date", "INCOTERM": "incoterm",
        "VESSEL NAME": "vessel", "VOYAGE NO.": "voyage", "PORT OF LOADING": "port_of_loading",
        "PORT OF DISCHARGE": "port_of_discharge", "TOTAL GROSS WEIGHT": "gross_weight_kg",
        "TOTAL NET WEIGHT": "net_weight_kg", "CONTAINER NO.": "container_number",
        "SEAL NO.": "seal_number",
    },
}

# Champs susceptibles d'être présents dans ces templates (pour ne pas compter en FP
# des champs présents mais non scorés : carrier, commodity, etc.)
PRESENT_EXTRA = {
    "bl": {"carrier", "commodity", "measurement", "service_type", "incoterm_place", "flag"},
    "invoice": {"subtotal", "vessel", "voyage", "hs_code", "quantity", "unit",
                "unit_price", "line_amount", "line_total", "currency", "goods_description",
                "freight_charges", "insurance", "seller"},
    "packing_list": {"total_cbm", "total_packages", "measurement", "container_type",
                     "incoterm_place", "marks_numbers", "dimensions", "goods_description",
                     "number_of_packages", "net_weight_kg", "quantity", "unit"},
}

# le pipeline peut nommer un champ date différemment du gold : aliases acceptés
FIELD_ALIASES = {
    "invoice_date": ["issue_date"],
    "pl_date":      ["issue_date", "pl_date"],
}
DATE_FIELDS = {"eta", "etd", "issue_date", "invoice_date", "pl_date", "date"}

_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
           "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _doc_type(name: str) -> str:
    n = name.upper()
    if n.startswith("BL"): return "bl"
    if n.startswith("INV"): return "invoice"
    if n.startswith("PL"): return "packing_list"
    return "bl"


def parse_gold(text: str, doc_type: str) -> dict:
    lines = [l.strip() for l in text.splitlines()]
    labels = GOLD_LABELS[doc_type]
    gold = {}
    for i, line in enumerate(lines):
        key = line.rstrip(":").strip()
        if key in labels:
            field = labels[key]
            # valeur = prochaine ligne non vide
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j]:
                    gold.setdefault(field, lines[j])
                    break
    return gold


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _norm_date(s: str) -> str:
    s = str(s).lower()
    nums = re.findall(r"\d+", s)
    mon = next((str(v) for k, v in _MONTHS.items() if k in s), "")
    parts = set(nums)
    if mon:
        parts.add(mon)
        parts.add(mon.zfill(2))
    return "|".join(sorted(parts))


def match(field: str, gold_val: str, ext_val: str) -> bool:
    if ext_val is None:
        return False
    g, e = str(gold_val), str(ext_val)
    if field in DATE_FIELDS or "date" in field:
        # comparer les composantes numériques + mois
        gn = set(re.findall(r"\d+", g)) | ({str(_MONTHS[k]) for k in _MONTHS if k in g.lower()})
        en = set(re.findall(r"\d+", e)) | ({str(_MONTHS[k]) for k in _MONTHS if k in e.lower()})
        # accepter si l'année et le jour coïncident
        return len(gn & en) >= 2
    gN, eN = _norm(g), _norm(e)
    if not gN or not eN:
        return False
    # nombre pur (poids/montant) : comparer les chiffres significatifs
    if re.fullmatch(r"[\d.,\s]+[a-z]*", g.strip().lower()) and re.search(r"\d", e):
        gd = re.sub(r"\D", "", g); ed = re.sub(r"\D", "", e)
        return gd in ed or ed in gd
    # texte : inclusion dans un sens ou l'autre
    return gN in eN or eN in gN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    from release2.services.pipeline import DocumentPipeline
    pipe = DocumentPipeline()

    pdfs = sorted(glob.glob(os.path.join(args.gold, "*.pdf")))
    agg = {}  # doc_type -> [tp, fp, fn, wrong]
    report = {}

    for pdf in pdfs:
        name = os.path.basename(pdf)
        dt = _doc_type(name)
        gold = parse_gold(fitz.open(pdf)[0].get_text(), dt)
        out = pipe.process([pdf])
        doc = out["documents"][0]
        ext = doc.get("fields", {})

        present = set(gold) | PRESENT_EXTRA.get(dt, set())
        tp = fn = wrong = 0
        wrongs, missings = [], []
        consumed = set()  # clés ext rattachées à un champ gold (nom direct ou alias)
        for f, gv in gold.items():
            keys = [f] + FIELD_ALIASES.get(f, [])
            ev_key = next((k for k in keys if k in ext), None)
            if ev_key is None:
                fn += 1; missings.append(f)
                continue
            consumed.add(ev_key)
            if match(f, gv, ext[ev_key]):
                tp += 1
            else:
                wrong += 1; wrongs.append(f"{f}: gold='{gv}' != ext='{ext[ev_key]}'")
        # faux positifs : champ extrait, ni rattaché à un gold, ni présent attendu
        fps = [f for f in ext if f not in present and f not in consumed]

        a = agg.setdefault(dt, [0, 0, 0, 0])
        a[0] += tp; a[1] += len(fps); a[2] += fn; a[3] += wrong
        report[name] = {"tp": tp, "fp": fps, "wrong": wrongs, "missing": missings,
                        "n_gold": len(gold)}
        print(f"\n### {name}  ({dt})  gold={len(gold)} extr={len(ext)}")
        print(f"   OK={tp}  FAUX={wrong}  MANQUANT={fn}  FAUX_POSITIFS={len(fps)}")
        for w in wrongs:   print(f"     [FAUX] {w}")
        for f in fps:      print(f"     [FP  ] {f} = {ext[f]}")
        for m in missings: print(f"     [MISS] {m}")

    print("\n" + "=" * 64)
    print(f"AGRÉGAT par type  (tag={args.tag})")
    print("=" * 64)
    for dt, (tp, fp, fn, wrong) in agg.items():
        prec = tp / (tp + fp + wrong) if (tp + fp + wrong) else 0
        rec = tp / (tp + fn + wrong) if (tp + fn + wrong) else 0
        print(f"  {dt:14s} OK={tp} FAUX={wrong} MANQ={fn} FP={fp}  "
              f"| precision={prec:.2f}  rappel={rec:.2f}")

    outp = Path(__file__).parent / f"real_eval_{args.tag}.json"
    json.dump(report, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {outp}")


if __name__ == "__main__":
    main()
