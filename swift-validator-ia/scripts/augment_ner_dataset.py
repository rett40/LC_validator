"""
augment_ner_dataset.py — Corrige et enrichit le dataset NER SWIFT
==================================================================

Corrige les 3 lacunes identifiees a l'evaluation du modele swift-ner :

  1. PORT  : le label existe mais 0 exemple dans le dataset -> le modele ne
             l'apprend jamais. On annote les ports des champs :44A:/:44B:.
  2. AMOUNT: les montants sont toujours colles sans separateur (USD2353000),
             donc le modele casse sur "USD 1,234,000". On diversifie les
             formats (espace, separateurs de milliers, decimales, europeen)
             avec remappage des offsets.
  3. GOODS : l'INCOTERM est annote A L'INTERIEUR du span GOODS (7184/10000),
             ce qui fragmente le span. On "de-chevauche" : GOODS s'arrete juste
             avant l'INCOTERM (les deux entites deviennent disjointes).

Entree  : data/data_stati/ner_dataset.jsonl
Sortie  : data/data_stati/ner_dataset_v2.jsonl

Usage :
  python scripts/augment_ner_dataset.py --seed 42 \
         --in data/data_stati/ner_dataset.jsonl \
         --out data/data_stati/ner_dataset_v2.jsonl
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Ville du port : apres "PORT OF LOADING/DISCHARGE: " jusqu'a la virgule/fin de ligne
_PORT_RE = re.compile(r":44[AB]: PORT OF (?:LOADING|DISCHARGE): ([A-Z][^\n,]*)")
# Montant compact SWIFT : 3 lettres devise + chiffres
_AMT_RE = re.compile(r"^([A-Z]{3})\s*([0-9]+)(?:[.,]([0-9]{2}))?$")


# ── 1. Reformatage d'un montant compact en formats varies ────────────────
def reformat_amount(raw: str, rng: random.Random) -> str:
    m = _AMT_RE.match(raw.strip())
    if not m:
        return raw
    ccy, digits, dec = m.group(1), m.group(2), m.group(3)
    grouped = "{:,}".format(int(digits))          # 2,353,000
    style = rng.choice([
        "compact",      # USD2353000   (format historique, on en garde)
        "space",        # USD 2353000
        "space_thou",   # USD 2,353,000
        "thou",         # USD2,353,000
        "space_dec",    # USD 2,353,000.00
        "euro",         # USD 2.353.000,00
    ])
    if style == "compact":
        return f"{ccy}{digits}"
    if style == "space":
        return f"{ccy} {digits}"
    if style == "space_thou":
        return f"{ccy} {grouped}"
    if style == "thou":
        return f"{ccy}{grouped}"
    if style == "space_dec":
        return f"{ccy} {grouped}.00"
    if style == "euro":
        return f"{ccy} {grouped.replace(',', '.')},00"
    return raw


# ── Remplacement de span avec remappage des offsets ──────────────────────
def replace_span(text, entities, start, end, new_sub):
    delta = len(new_sub) - (end - start)
    new_text = text[:start] + new_sub + text[end:]
    new_ents = []
    for e in entities:
        s, en = e["start"], e["end"]
        if s >= end:                       # entierement apres -> decaler
            s += delta; en += delta
        elif en <= start:                  # entierement avant -> inchange
            pass
        else:                              # c'est l'entite remplacee (AMOUNT)
            s, en = start, start + len(new_sub)
        ne = dict(e, start=s, end=en)
        new_ents.append(ne)
    return new_text, new_ents


def augment_record(rec, rng, stats):
    text = rec["text"]
    ents = [dict(e) for e in rec.get("entities", [])]

    # 2. PORT : annoter les ports des champs :44A:/:44B: ------------------
    existing_port_spans = {(e["start"], e["end"]) for e in ents if e["label"] == "PORT"}
    for m in _PORT_RE.finditer(text):
        city = m.group(1).rstrip()
        if not city:
            continue
        s = m.start(1); en = s + len(city)
        if (s, en) in existing_port_spans:
            continue
        ents.append({"label": "PORT", "value": city, "start": s, "end": en,
                     "matched_text": city, "in_text": True, "ner_label": "LOC"})
        stats["port_added"] += 1

    # 3. GOODS : de-chevaucher l'INCOTERM ---------------------------------
    goods = [e for e in ents if e["label"] == "GOODS"]
    incos = [e for e in ents if e["label"] == "INCOTERM"]
    for g in goods:
        nested = [i for i in incos if i["start"] < g["end"] and i["end"] > g["start"]]
        if not nested:
            continue
        cut = min(i["start"] for i in nested)
        cut_trim = cut
        while cut_trim > g["start"] and text[cut_trim - 1] == " ":
            cut_trim -= 1
        if cut_trim > g["start"]:
            # cas usuel : description AVANT l'incoterm -> couper la fin
            g["end"] = cut_trim
        else:
            # incoterm en tete du span -> demarrer GOODS apres le dernier incoterm
            new_start = max(i["end"] for i in nested)
            while new_start < g["end"] and text[new_start] == " ":
                new_start += 1
            if new_start < g["end"]:
                g["start"] = new_start
            else:
                continue
        g["value"] = text[g["start"]:g["end"]]
        g["matched_text"] = g["value"]
        stats["goods_trimmed"] += 1

    # 1. AMOUNT : diversifier le format (avec remappage) ------------------
    amt = next((e for e in ents if e["label"] == "AMOUNT"), None)
    if amt:
        raw = text[amt["start"]:amt["end"]]
        new_raw = reformat_amount(raw, rng)
        if new_raw != raw:
            text, ents = replace_span(text, ents, amt["start"], amt["end"], new_raw)
            # mettre a jour value/matched_text de l'AMOUNT
            for e in ents:
                if e["label"] == "AMOUNT":
                    e["value"] = new_raw
                    e["matched_text"] = new_raw
            stats["amount_reformatted"] += 1

    # recalculer les spacy_spans
    rec["text"] = text
    rec["entities"] = ents
    rec["spacy_spans"] = [[e["start"], e["end"], e["label"]] for e in ents]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/data_stati/ner_dataset.jsonl")
    ap.add_argument("--out", default="data/data_stati/ner_dataset_v2.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stats = Counter()
    n = 0
    with open(args.inp, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rec = augment_record(rec, rng, stats)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"OK -> {args.out}  ({n} records)")
    print(f"  PORT entites ajoutees      : {stats['port_added']}")
    print(f"  GOODS de-chevauches        : {stats['goods_trimmed']}")
    print(f"  AMOUNT reformates          : {stats['amount_reformatted']}")


if __name__ == "__main__":
    main()
