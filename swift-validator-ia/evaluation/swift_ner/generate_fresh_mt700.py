"""
generate_fresh_mt700.py — Generateur de messages SWIFT MT700 FRAIS
==================================================================

But : produire un jeu de TEST hold-out genere a la volee, avec un
vocabulaire ENTIEREMENT NOUVEAU (societes, banques, marchandises, ports,
incoterms, devises, dates jamais vus a l'entrainement). C'est le vrai test
de generalisation du modele swift-ner (vs le split test in-distribution).

Format de sortie identique a data/data_stati/ner_dataset.jsonl :
  {"text": "...", "swift_type": "MT700",
   "entities": [{"label": "...", "start": N, "end": M, "value": "..."}]}

Le schema d'annotation reproduit fidelement l'entrainement :
  - GOODS = description complete (INCOTERM imbrique a l'interieur)
  - INCOTERM = token incoterm (span inclus dans GOODS)
  -> permet une comparaison directe des scores.

Usage :
  python evaluation/swift_ner/generate_fresh_mt700.py --n 500 --seed 2026 \
         --out evaluation/swift_ner/fresh_mt700.jsonl
  # --perturb  pour ajouter des variations de format (test OOD plus dur)
"""

import argparse
import json
import random
from pathlib import Path

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
    ("RABOBANK", "RABONL2UXXX"), ("DANSKE BANK", "DABADKKKXXX"),
    ("OVERSEA CHINESE BANKING", "OCBCSGSGXXX"), ("BANCO BRADESCO", "BBDEBRSPXXX"),
    ("STANDARD CHARTERED", "SCBLGB2LXXX"), ("NORDEA BANK", "NDEAFIHHXXX"),
    ("BANK OF AYUDHYA", "AYUDTHBKXXX"), ("EMIRATES NBD", "EBILAEADXXX"),
    ("CAIXABANK", "CAIXESBBXXX"), ("KOOKMIN BANK", "CZNBKRSEXXX"),
    ("INTESA SANPAOLO", "BCITITMMXXX"), ("ABSA BANK", "ABSAZAJJXXX"),
]
GOODS = [
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
    "GOTHENBURG", "COLOMBO", "KAOHSIUNG", "GDANSK", "VERACRUZ",
]
INCOTERMS = ["CIF", "FOB", "CFR", "FCA", "CPT", "CIP", "DPU", "DDP", "FAS", "DAT"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "SEK", "AUD", "SGD", "ZAR", "AED", "BRL"]
APP_CITIES = [("LYON", "FRANCE"), ("OSAKA", "JAPAN"), ("LIMA", "PERU"),
              ("STOCKHOLM", "SWEDEN"), ("MILAN", "ITALY"), ("DUBAI", "UAE")]
BEN_CITIES = [("SINGAPORE", "SINGAPORE"), ("CAPE TOWN", "SOUTH AFRICA"),
              ("MUMBAI", "INDIA"), ("HO CHI MINH", "VIETNAM"),
              ("ISTANBUL", "TURKEY"), ("MANILA", "PHILIPPINES")]


class SpanBuilder:
    """Construit le texte segment par segment en suivant les offsets char."""
    def __init__(self):
        self.parts = []
        self.pos = 0
        self.entities = []

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


def rand_date(rng):
    yy = rng.randint(22, 25)
    mm = rng.randint(1, 12)
    dd = rng.randint(1, 28)
    return f"{yy:02d}{mm:02d}{dd:02d}"


def gen_one(rng, perturb=False):
    b = SpanBuilder()
    app = rng.choice(APPLICANTS)
    ben = rng.choice(BENEFICIARIES)
    bank_name, bic = rng.choice(BANKS)
    goods_desc = rng.choice(GOODS)
    inco = rng.choice(INCOTERMS)
    cur = rng.choice(CURRENCIES)
    pol = rng.choice(PORTS)
    pod = rng.choice([p for p in PORTS if p != pol])
    inco_place = rng.choice(PORTS)

    issue = rand_date(rng)
    expiry = rand_date(rng)
    shipment = rand_date(rng)
    amount_val = rng.randint(80, 4500) * 1000
    amount_str = f"{amount_val:,}" if perturb else str(amount_val)
    app_city = rng.choice(APP_CITIES)
    ben_city = rng.choice(BEN_CITIES)

    nl = "\n"
    b.add(":27: 1/1" + nl)
    b.add(":40A: IRREVOCABLE" + nl)
    b.add(f":20: LC{rng.randint(1000000000, 9999999999)}" + nl)
    b.add(":31C: "); b.add(issue, "ISSUE_DATE"); b.add(nl)
    b.add(":40E: UCP 600" + nl)
    b.add(":31D: "); b.add(expiry, "EXPIRY_DATE"); b.add(f" {ben_city[0]} {ben_city[1]}" + nl)
    b.add(":52A: "); b.add(bank_name, "ISSUING_BANK"); b.add(nl)
    b.add(":50: "); b.add(app, "APPLICANT")
    b.add(f"{nl}  {rng.randint(1,99)} {app_city[0]} STREET{nl}  {app_city[0]} {app_city[1]}" + nl)
    b.add(":59: "); b.add(ben, "BENEFICIARY")
    b.add(f"{nl}  PO BOX {rng.randint(100,9999)}{nl}  {ben_city[0]} {ben_city[1]}" + nl)
    # AMOUNT = devise + montant (coherent avec l'entrainement, qui collait
    # "NOK2353000"). En mode perturb on insere un espace + des separateurs de
    # milliers, mais le span gold reste "devise + montant".
    b.add(":32B: ")
    if perturb:
        b.add(f"{cur} {amount_str}", "AMOUNT"); b.add(nl)
    else:
        b.add(cur + amount_str, "AMOUNT"); b.add(nl)
    b.add(":39A: 10/10" + nl)
    b.add(f":41A: {bic}" + nl)
    b.add(":42C: AT SIGHT" + nl)
    b.add(":43P: PARTIAL SHIPMENTS ALLOWED" + nl)
    b.add(":43T: TRANSSHIPMENT NOT ALLOWED" + nl)
    b.add(":44A: PORT OF LOADING: "); b.add(pol, "PORT"); b.add(nl)
    b.add(":44B: PORT OF DISCHARGE: "); b.add(pod, "PORT"); b.add(nl)
    b.add(":44C: "); b.add(shipment, "SHIPMENT_DATE"); b.add(nl)
    # :45A: goods (INCOTERM imbrique a l'interieur, comme a l'entrainement).
    # Ordre des entites : GOODS PUIS INCOTERM -> reproduit l'ecrasement char
    # de l'entrainement (l'INCOTERM fragmente le span GOODS).
    b.add(":45A: ")
    g_start = b.pos
    b.add(goods_desc + " ")
    inc_s = b.pos
    b.add(inco)                       # ecrit sans label
    inc_e = b.pos
    b.add(" " + inco_place)
    g_end = b.pos
    b.entities.append({"label": "GOODS", "start": g_start, "end": g_end,
                       "value": b.text()[g_start:g_end]})
    b.entities.append({"label": "INCOTERM", "start": inc_s, "end": inc_e,
                       "value": inco})
    b.add(nl)
    b.add(":46A: DOCUMENTS REQUIRED:" + nl)
    b.add("  1. SIGNED COMMERCIAL INVOICE 3 COPIES" + nl)
    b.add("  2. FULL SET BILL OF LADING" + nl)
    b.add("  3. CERTIFICATE OF ORIGIN 1 COPY" + nl)
    b.add(":47A: ADDITIONAL CONDITIONS:" + nl)
    b.add("  1. PRESENTATION PERIOD 21 DAYS FROM SHIPMENT DATE" + nl)
    b.add(":71B: ALL CHARGES OUTSIDE ISSUING BANK FOR BENEFICIARY" + nl)
    b.add(":48: 30 DAYS" + nl)
    b.add(":49: WITHOUT")

    return {"text": b.text(), "swift_type": "MT700", "entities": b.entities}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--perturb", action="store_true",
                    help="Ajoute des variations de format (test OOD plus dur)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "fresh_mt700.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            rec = gen_one(rng, perturb=args.perturb)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"OK -> {out}  ({args.n} messages MT700 frais, seed={args.seed}, "
          f"perturb={args.perturb})")


if __name__ == "__main__":
    main()
