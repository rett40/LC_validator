"""
evaluate_realistic.py — Evaluation REALISTE du systeme swift-ner (modele + regles)
==================================================================================

Pourquoi ce script ?
--------------------
Le split test in-distribution (`ner_dataset.jsonl`) est TEMPLATE : le modele a vu
les memes gabarits a l'entrainement, donc il obtient ~1.0 partout (memorisation,
pas generalisation). Des scores parfaits ne sont pas credibles dans un memoire.

Ce script construit a la volee un jeu de test HOLD-OUT realiste :
  - vocabulaire ENTIEREMENT NOUVEAU (societes, banques, marchandises, ports... jamais vus),
  - schema d'annotation v2 (GOODS propre, INCOTERM separe, PORT sur :44A/:44B:,
    AMOUNT sur :32B:),
  - BRUIT DOCUMENTAIRE realiste calibre (variations d'espacement, de format de
    montant, de formulation des tags, fautes type OCR) -> simule de vrais documents
    scannes / saisis a la main.

Le systeme evalue est le systeme HYBRIDE (modele BERT + regles deterministes du
`SwiftNERWrapper` : regle AMOUNT :32B:, regle PORT :44A/B:, fusion/trim GOODS).

Metriques : seqeval entity-level (P/R/F1 par entite + micro/macro/weighted),
accuracy token-level. Sortie : reports/report_realistic.json + figures.

Usage :
  python evaluation/swift_ner/evaluate_realistic.py --n 600 --seed 2026 --noise 1.0
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "swift-ner"
OUT_DIR = HERE / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

ENTITY_TYPES = [
    "APPLICANT", "BENEFICIARY", "ISSUING_BANK", "AMOUNT", "GOODS",
    "INCOTERM", "EXPIRY_DATE", "ISSUE_DATE", "SHIPMENT_DATE", "PORT",
]
VALID = set(ENTITY_TYPES)

# ── Vocabulaire NOUVEAU (disjoint de l'entrainement) ─────────────────────
APPLICANTS = [
    "NORDWIND TEXTILES GMBH", "KESTREL TRADING PTY LTD", "ASTRA POLYMERS CO LTD",
    "MERIDIAN STEELWORKS BV", "SAKURA IMPORT KK", "VOLGA GRAIN OOO",
    "ANDES MINING SAC", "SAVANNAH AGRO HOLDINGS", "BLUEFIN SEAFOODS AB",
    "TERRA NOVA CHEMICALS SPA", "HELIOS SOLAR PARTS SL", "ORION MACHINERY SDN BHD",
    "CEDAR VALLEY FOODS INC", "ZEPHYR LOGISTICS DMCC", "GRANITE BAY EXPORTS LLC",
]
BENEFICIARIES = [
    "PACIFIC RIM SUPPLIERS LTD", "EMERALD COAST TRADERS", "IRONCLAD COMPONENTS GMBH",
    "LOTUS AGRICOMM PVT LTD", "SUMMIT TEXTILE MILLS", "AURORA FROZEN GOODS BV",
    "CRIMSON SPICE HOUSE", "NORTHERN STAR METALS CO", "DELTA FRESH PRODUCE SA",
    "TITAN MACHINE TOOLS AB", "OASIS PETRO SUPPLIES FZE", "SILVERLINE CERAMICS LTD",
    "HIGHLAND DAIRY EXPORTS", "MONSOON RUBBER INDUSTRIES", "FALCON ELECTRONICS PTE",
]
BANKS = [
    "RABOBANK", "DANSKE BANK", "OVERSEA CHINESE BANKING", "BANCO BRADESCO",
    "STANDARD CHARTERED", "NORDEA BANK", "BANK OF AYUDHYA", "EMIRATES NBD",
    "CAIXABANK", "KOOKMIN BANK", "INTESA SANPAOLO", "ABSA BANK",
]
GOODS_LIST = [
    "STAINLESS STEEL COILS GRADE 304 850 MT",
    "ROASTED ARABICA COFFEE BEANS 1200 BAGS",
    "LITHIUM ION BATTERY CELLS 50000 UNITS",
    "ORGANIC BASMATI RICE 600 MT",
    "CERAMIC FLOOR TILES 60X60CM 15000 BOXES",
    "NATURAL RUBBER SHEETS RSS3 400 MT",
    "FROZEN PACIFIC SHRIMP 18 TONS",
    "ALUMINIUM EXTRUSION PROFILES 320 MT",
    "COTTON YARN 30S COUNT 220 BALES",
    "SOLAR PHOTOVOLTAIC MODULES 8000 PANELS",
    "REFINED SUNFLOWER OIL 500 MT",
    "INDUSTRIAL CNC LATHE MACHINES 12 SETS",
]
PORTS = [
    "ROTTERDAM", "BUSAN", "SANTOS", "FELIXSTOWE", "TANJUNG PELEPAS",
    "JEBEL ALI", "VALENCIA", "DURBAN", "MANZANILLO", "HAIPHONG",
    "GOTHENBURG", "KAOHSIUNG", "GDANSK", "VERACRUZ", "MOMBASA",
]
INCOTERMS = ["CIF", "FOB", "CFR", "FCA", "CPT", "CIP", "DPU", "DDP", "FAS", "EXW"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "SEK", "AUD", "SGD", "ZAR", "AED", "BRL"]
APP_CITIES = [("LYON", "FRANCE"), ("OSAKA", "JAPAN"), ("LIMA", "PERU"),
              ("STOCKHOLM", "SWEDEN"), ("MILAN", "ITALY"), ("DUBAI", "UAE")]
REGIONS = ["EUROPE", "ASIA", "AFRICA", "AMERICAS", "MIDDLE EAST"]

# OCR-like confusions (n'affectent pas les bornes -> bruit cosmetique realiste)
_OCR = {"O": "0", "0": "O", "I": "1", "1": "I", "S": "5", "B": "8", "Z": "2"}


class SpanBuilder:
    def __init__(self):
        self.parts, self.pos, self.entities = [], 0, []

    def add(self, s, label=None):
        start = self.pos
        self.parts.append(s)
        self.pos += len(s)
        if label:
            self.entities.append({"label": label, "start": start,
                                  "end": self.pos, "value": s})
        return start, self.pos

    def text(self):
        return "".join(self.parts)


def rand_date(rng, noise=0.0):
    yy, mm, dd = rng.randint(22, 25), rng.randint(1, 12), rng.randint(1, 28)
    # format standard SWIFT YYMMDD ; parfois un format alternatif (separateurs)
    if noise > 0 and rng.random() < 0.08 * noise:
        sep = rng.choice(["/", ".", "-"])
        if rng.random() < 0.5:
            return f"{yy:02d}{sep}{mm:02d}{sep}{dd:02d}"
        return f"{dd:02d}{sep}{mm:02d}{sep}{yy:02d}"
    return f"{yy:02d}{mm:02d}{dd:02d}"


def ocr_noise(rng, s, p):
    """Substitution OCR d'AU PLUS un caractere (bruit cosmetique, bornes inchangees)."""
    if p <= 0 or rng.random() > p:
        return s
    idx = [i for i, c in enumerate(s) if c in _OCR]
    if not idx:
        return s
    i = rng.choice(idx)
    return s[:i] + _OCR[s[i]] + s[i + 1:]


def sp(rng, noise):
    """Espacement variable apres un tag : simule saisie/scan irreguliers."""
    if noise > 0 and rng.random() < 0.25 * noise:
        return " " * rng.choice([2, 3])
    return " "


def gen_one(rng, noise=1.0):
    b = SpanBuilder()
    app, ben = rng.choice(APPLICANTS), rng.choice(BENEFICIARIES)
    bank = rng.choice(BANKS)
    goods = rng.choice(GOODS_LIST)
    inco = rng.choice(INCOTERMS)
    cur = rng.choice(CURRENCIES)
    pol = rng.choice(PORTS)
    pod = rng.choice([p for p in PORTS if p != pol])
    inco_place = rng.choice(PORTS)
    issue = rand_date(rng, noise)
    expiry = rand_date(rng, noise)
    shipment = rand_date(rng, noise)
    app_city = rng.choice(APP_CITIES)
    nl = "\n"
    # suffixe juridique parfois ajoute au nom de banque (variation reelle)
    if noise > 0 and rng.random() < 0.14 * noise:
        bank = bank + rng.choice([" PLC", " LIMITED", " NV", " AG", " SA"])

    b.add(":27: 1/1" + nl)
    b.add(":40A: IRREVOCABLE" + nl)
    b.add(f":20: LC{rng.randint(1000000000, 9999999999)}" + nl)
    b.add(":31C:" + sp(rng, noise)); b.add(issue, "ISSUE_DATE"); b.add(nl)
    b.add(":40E: UCP 600" + nl)
    b.add(":31D:" + sp(rng, noise)); b.add(expiry, "EXPIRY_DATE"); b.add(" PLACE EXPIRY" + nl)
    b.add(":52A:" + sp(rng, noise)); b.add(ocr_noise(rng, bank, 0.3 * noise), "ISSUING_BANK"); b.add(nl)
    b.add(":50:" + sp(rng, noise)); b.add(ocr_noise(rng, app, 0.3 * noise), "APPLICANT")
    b.add(f"{nl}  {rng.randint(1,99)} {app_city[0]} STREET{nl}  {app_city[0]} {app_city[1]}" + nl)
    b.add(":59:" + sp(rng, noise)); b.add(ocr_noise(rng, ben, 0.3 * noise), "BENEFICIARY")
    b.add(f"{nl}  PO BOX {rng.randint(100,9999)}" + nl)

    # ── AMOUNT (:32B:) — formats varies ; parfois tag malforme (regex rate) ──
    amount_val = rng.randint(80, 4500) * 1000
    fmt = rng.random()
    if fmt < 0.45:
        amt = f"{cur}{amount_val}"
    elif fmt < 0.65:
        amt = f"{cur} {amount_val}"
    elif fmt < 0.80:
        amt = f"{cur} {amount_val:,}"
    elif fmt < 0.90:
        amt = f"{cur} {amount_val:,}.00"
    else:
        amt = f"{cur} {format(amount_val, ',').replace(',', '.')},00"
    # tag :32B: parfois malforme (espace parasite) -> la regle peut rater
    if noise > 0 and rng.random() < 0.06 * noise:
        b.add(":32B :" + sp(rng, noise))      # deux-points decale
    else:
        b.add(":32B:" + sp(rng, noise))
    b.add(amt, "AMOUNT"); b.add(nl)

    b.add(":39A: 10/10" + nl)
    b.add(":42C: AT SIGHT" + nl)
    b.add(":43P: PARTIAL SHIPMENTS ALLOWED" + nl)
    b.add(":43T: TRANSSHIPMENT NOT ALLOWED" + nl)

    # ── PORT (:44A/:44B:) — formulation parfois variee (regex rate) ──
    def port_line(tag, default_phrase, port, alt_phrase):
        if noise > 0 and rng.random() < 0.06 * noise:
            b.add(f":{tag}: {alt_phrase} ")        # formulation alternative
        else:
            b.add(f":{tag}: {default_phrase}: ")
        b.add(port, "PORT")
        b.add(f", {rng.choice(REGIONS)}" + nl)
    port_line("44A", "PORT OF LOADING", pol, "LOADING PORT")
    port_line("44B", "PORT OF DISCHARGE", pod, "DISCHARGE PORT")

    b.add(":44C:" + sp(rng, noise)); b.add(shipment, "SHIPMENT_DATE"); b.add(nl)

    # ── GOODS (:45A:) schema v2 : description propre + INCOTERM separe ──
    b.add(":45A:" + sp(rng, noise))
    g0 = b.pos
    b.add(ocr_noise(rng, goods, 0.3 * noise))
    g1 = b.pos
    b.entities.append({"label": "GOODS", "start": g0, "end": g1,
                       "value": b.text()[g0:g1]})
    b.add(" ")
    b.add(inco, "INCOTERM")
    b.add(" " + inco_place + nl)

    b.add(":46A: DOCUMENTS REQUIRED:" + nl)
    b.add("  1. SIGNED COMMERCIAL INVOICE 3 COPIES" + nl)
    b.add("  2. FULL SET BILL OF LADING" + nl)
    b.add(":47A: ADDITIONAL CONDITIONS:" + nl)
    b.add("  1. PRESENTATION PERIOD 21 DAYS FROM SHIPMENT DATE" + nl)
    b.add(":71B: ALL CHARGES OUTSIDE ISSUING BANK FOR BENEFICIARY" + nl)
    b.add(":48: 30 DAYS" + nl)
    b.add(":49: WITHOUT")
    return {"text": b.text(), "swift_type": "MT700", "entities": b.entities}


# ── BIO char-labels (gold) ───────────────────────────────────────────────
def char_labels_for(text, entities):
    lab = ["O"] * len(text)
    for e in entities:
        l = e["label"]
        if l not in VALID:
            continue
        s, en = max(0, e["start"]), min(e["end"], len(text))
        if s >= en:
            continue
        lab[s] = f"B-{l}"
        for i in range(s + 1, en):
            lab[i] = f"I-{l}"
    return lab


def spans_to_char_labels(text, spans):
    lab = ["O"] * len(text)
    for sp_ in spans:
        l = sp_["label"]
        if l not in VALID:
            continue
        s, en = max(0, sp_["start"]), min(sp_["end"], len(text))
        if s >= en:
            continue
        lab[s] = f"B-{l}"
        for i in range(s + 1, en):
            lab[i] = f"I-{l}"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--noise", type=float, default=1.0,
                    help="Intensite du bruit documentaire (0=propre, 1=realiste)")
    ap.add_argument("--tag", default="_realistic")
    ap.add_argument("--save-dataset", action="store_true")
    args = ap.parse_args()

    from swift_ner_wrapper import SwiftNERWrapper
    from seqeval.metrics import classification_report as seq_report
    from seqeval.metrics import precision_score, recall_score, f1_score
    import numpy as np
    from transformers import AutoTokenizer

    rng = random.Random(args.seed)
    recs = [gen_one(rng, noise=args.noise) for _ in range(args.n)]
    print(f"Jeu frais genere : {len(recs)} MT700 (seed={args.seed}, noise={args.noise})")
    if args.save_dataset:
        p = HERE / f"eval{args.tag}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  dataset -> {p}")

    w = SwiftNERWrapper(MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    gold_seqs, pred_seqs = [], []
    flat_g, flat_p = [], []
    for r in recs:
        text = r["text"]
        spans = w.predict(text, fix_goods=True, add_ports=True, rule_amount=True)
        g_char = char_labels_for(text, r["entities"])
        p_char = spans_to_char_labels(text, spans)
        enc = tok(text, truncation=True, max_length=512, return_offsets_mapping=True)
        gs, ps = [], []
        for (s, e) in enc["offset_mapping"]:
            if s == e:
                continue
            g = g_char[s] if s < len(g_char) else "O"
            pr = p_char[s] if s < len(p_char) else "O"
            gs.append(g); ps.append(pr)
            flat_g.append(g); flat_p.append(pr)
        gold_seqs.append(gs); pred_seqs.append(ps)

    rep = seq_report(gold_seqs, pred_seqs, output_dict=True, zero_division=0)
    rep_txt = seq_report(gold_seqs, pred_seqs, zero_division=0)
    micro = (precision_score(gold_seqs, pred_seqs),
             recall_score(gold_seqs, pred_seqs),
             f1_score(gold_seqs, pred_seqs))
    tok_acc = float(np.mean(np.array(flat_g) == np.array(flat_p)))

    print("\n" + "=" * 70)
    print("  EVALUATION REALISTE (systeme hybride, jeu frais bruite) — swift-ner")
    print("=" * 70)
    print(rep_txt)
    print(f"  Micro -> P={micro[0]:.4f} R={micro[1]:.4f} F1={micro[2]:.4f}")
    print(f"  Accuracy token-level : {tok_acc:.4f}")

    per_entity = {k: v for k, v in rep.items() if k in VALID}
    report = {
        "model": "swift-ner (systeme hybride : modele + regles)",
        "evaluation": "jeu frais hold-out, vocabulaire neuf, bruit documentaire realiste",
        "n_examples": len(recs), "seed": args.seed, "noise": args.noise,
        "token_level_accuracy": tok_acc,
        "entity_level": {
            "micro": {"precision": micro[0], "recall": micro[1], "f1": micro[2]},
            "macro": {"precision": rep.get("macro avg", {}).get("precision"),
                      "recall": rep.get("macro avg", {}).get("recall"),
                      "f1": rep.get("macro avg", {}).get("f1-score")},
            "weighted": {"precision": rep.get("weighted avg", {}).get("precision"),
                         "recall": rep.get("weighted avg", {}).get("recall"),
                         "f1": rep.get("weighted avg", {}).get("f1-score")},
            "per_entity": {k: {"precision": v["precision"], "recall": v["recall"],
                               "f1": v["f1-score"], "support": int(v["support"])}
                           for k, v in per_entity.items()},
        },
    }
    out = OUT_DIR / f"report{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Rapport JSON -> {out}")

    # figures (reutilise les helpers de evaluate_swift_ner)
    try:
        import evaluate_swift_ner as E
        from sklearn.metrics import confusion_matrix

        # diagrammes F1 / precision-recall / support
        E.plot_per_entity_f1(per_entity, OUT_DIR / f"per_entity_f1{args.tag}.png")
        E.plot_pr_scatter(per_entity, OUT_DIR / f"precision_recall_scatter{args.tag}.png")
        E.plot_support(per_entity, OUT_DIR / f"entity_support_distribution{args.tag}.png")

        # ── Matrice de confusion token-level (BIO, 21 labels) ──
        label_list = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
        cm_bio = confusion_matrix(flat_g, flat_p, labels=label_list)
        E.plot_confusion(cm_bio, label_list, OUT_DIR / f"confusion_matrix_bio{args.tag}.png",
                         "Matrice de confusion token-level (BIO) - eval realiste")

        # ── Matrice de confusion entity-level (B-/I- fusionnes) ──
        def collapse(lbl):
            return lbl.split("-", 1)[1] if lbl != "O" else "O"
        ent_order = ["O"] + ENTITY_TYPES
        g_ent = [collapse(x) for x in flat_g]
        p_ent = [collapse(x) for x in flat_p]
        cm_ent = confusion_matrix(g_ent, p_ent, labels=ent_order)
        E.plot_confusion(cm_ent, ent_order, OUT_DIR / f"confusion_matrix_entity{args.tag}.png",
                         "Matrice de confusion entity-level - eval realiste")
        print(f"  Figures (.png) -> {OUT_DIR}")
        print("    - per_entity_f1, precision_recall_scatter, entity_support_distribution")
        print("    - confusion_matrix_bio, confusion_matrix_entity")
    except Exception as ex:
        print(f"  (figures ignorees : {ex})")


if __name__ == "__main__":
    main()
