"""
fix_issue_date_eval_independent.py
==================================

Corrige l'omission d'annotation ISSUE_DATE dans le jeu d'evaluation independant
des INVOICES (release2/data/complex/eval_independent/dataset_invoice/test).

Cause racine
------------
Le generateur `document_generator/generate_invoice.py` DESSINE la date de la
facture (drawString "DATE: {invoice_date}") mais n'emet JAMAIS d'annotation
`issue_date` (pas d'appel a `self._annotate("issue_date", ...)`). Le script
`scripts/06_generate_eval_dataset.py` construit le BIO a partir des annotations ;
sans annotation, la date reste taggee "O". Resultat : 0 token ISSUE_DATE dans le
gold -> support=0 -> F1=0 dans le rapport (ner_invoice_per_entity_f1.png).

Le texte de la date EST present dans les mots OCR ("DATE[:] 20 MAY 2026"). Ce
script re-annote le gold a l'identique du schema d'entrainement (dataset_invoice) :
on repere l'occurrence "corps" (la derniere "DATE[:] <jour> <mois> [<annee>]",
situee juste au-dessus de LC_REF) et on tagge sa VALEUR :
    <jour> -> B-ISSUE_DATE   <mois>,<annee> -> I-ISSUE_DATE
(le mot-cle "DATE"/"DATE:" reste O, comme a l'entrainement). L'en-tete "Date:"
n'est pas labellise (idem entrainement).

Le script sauvegarde l'original (.bak) avant d'ecrire.

Usage :
  python evaluation/release2/fix_issue_date_eval_independent.py
  python evaluation/release2/fix_issue_date_eval_independent.py --dry-run
"""

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_JSONL = (ROOT / "release2" / "data" / "complex" / "eval_independent"
              / "dataset_invoice" / "test" / "data.jsonl")
ID2LABEL = TEST_JSONL.parent.parent / "id2label.json"

MONTHS = {"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"}


def issue_date_value_span(words: list[str]) -> list[int] | None:
    """Indices des tokens de la VALEUR de la date d'emission (corps).

    Repere toutes les occurrences "DATE[:] <jour> <mois> [<annee>]" dans la zone
    d'en-tete (k < 55) et retourne la DERNIERE (= corps, au-dessus de LC_REF),
    a l'identique de l'annotation d'entrainement.
    """
    occ = []
    for k in range(len(words) - 2):
        if (words[k].upper().rstrip(":") == "DATE"
                and re.match(r"^\d{1,2}$", words[k + 1])
                and words[k + 2].upper()[:3] in MONTHS
                and k < 55):
            span = [k + 1, k + 2]
            if k + 3 < len(words) and re.match(r"^\d{4}$", words[k + 3]):
                span.append(k + 3)
            occ.append(span)
    return occ[-1] if occ else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="N'ecrit rien, affiche seulement le bilan")
    args = ap.parse_args()

    id2label = {int(k): v for k, v in
                json.loads(ID2LABEL.read_text(encoding="utf-8")).items()}
    label2id = {v: k for k, v in id2label.items()}
    B = label2id["B-ISSUE_DATE"]
    I = label2id["I-ISSUE_DATE"]

    records = []
    for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    patched = skipped_existing = no_match = overwrite_guard = 0
    for rec in records:
        tags = rec["ner_tags"]
        if any(id2label.get(t, "O") == "B-ISSUE_DATE" for t in tags):
            skipped_existing += 1
            continue
        span = issue_date_value_span(rec["words"])
        if span is None:
            no_match += 1
            continue
        if not all(id2label.get(tags[i], "O") == "O" for i in span):
            overwrite_guard += 1
            continue
        tags[span[0]] = B
        for i in span[1:]:
            tags[i] = I
        patched += 1

    print(f"Fichier        : {TEST_JSONL}")
    print(f"Docs total     : {len(records)}")
    print(f"Deja annotes   : {skipped_existing}")
    print(f"Sans motif date: {no_match}")
    print(f"Conflit (skip) : {overwrite_guard}")
    print(f"PATCHES        : {patched}")

    if args.dry_run:
        print("\n[dry-run] aucune ecriture.")
        return

    bak = TEST_JSONL.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copy2(TEST_JSONL, bak)
        print(f"Sauvegarde     : {bak}")
    with open(TEST_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Ecriture OK.")


if __name__ == "__main__":
    main()
