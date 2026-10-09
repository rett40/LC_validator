# -*- coding: utf-8 -*-
"""
Cross-validation multi-modèles du PROTOCOLE NLI UCP (Phase 2)
============================================================

But
----
Mesurer la ROBUSTESSE du protocole NLI clause ↔ article UCP en le ré-exécutant
avec des modèles INDÉPENDANTS, puis en quantifiant leur accord. Si la décision
(VIOLATION / RESTRICTION / COMPLIANT) ne dépend pas du modèle sous-jacent, le
protocole est robuste ; si les modèles divergent fortement, la décision est
fragile et dépend d'un artefact de modèle.

Protocole testé (identique à la production — swift_ucp_nli.py)
-------------------------------------------------------------
Pour une clause (premise), on évalue 3 hypothèses (conflit / compatible /
restriction) et on retient celle de plus forte probabilité d'entailment, avec
une marge d'abstention. On garde EXACTEMENT ces hypothèses et cette marge pour
chaque modèle — seul le modèle change.

Modèles comparés (tous locaux, en cache HF — pas de clé API requise)
--------------------------------------------------------------------
  A  cross-encoder/nli-deberta-v3-xsmall   ← modèle de PRODUCTION (UCPNLIRefiner)
  B  cross-encoder/nli-deberta-v3-small    ← même famille, plus gros (stabilité intra-famille)
  C  facebook/bart-large-mnli              ← architecture/entraînement DIFFÉRENTS (indépendance)

Le couple A↔C est le chiffre phare : accord entre deux familles de modèles
indépendantes. Si une clé LLM (ANTHROPIC_API_KEY / OPENAI_API_KEY) est présente,
on peut brancher un labeler LLM supplémentaire (non requis ici).

Métriques
---------
  • Accord (%) entre paires de modèles
  • Kappa de Cohen (accord corrigé du hasard)
  • Exactitude vs gold (par modèle)
  • Taux d'abstention (par modèle)
  • Liste des désaccords (pour inspection qualitative)

Usage
-----
    cd swift-validator-ia
    python evaluation/nli_cross_validation/cross_validate_nli.py
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

LABELS = ("VIOLATION", "RESTRICTION", "COMPLIANT")
ABSTAIN = "ABSTAIN"

# Hypothèses + marge IDENTIQUES à la production (swift_ucp_nli.py)
_MARGIN = 0.07
_MIN_ENT = 0.33


def _hypotheses(article: str, title: str) -> dict[str, str]:
    ttl = (title or article or "UCP topic").strip()[:120]
    return {
        "VIOLATION": (
            f"The documentary credit clause conflicts with or opts out of "
            f"standard UCP 600 requirements regarding {ttl} ({article})."
        ),
        "COMPLIANT": (
            f"The documentary credit clause is compatible with usual UCP 600 "
            f"practice regarding {ttl} ({article})."
        ),
        "RESTRICTION": (
            f"The documentary credit clause imposes stricter obligations than "
            f"the typical UCP 600 minimum regarding {ttl} ({article})."
        ),
    }


def _decide(ent_by_label: dict[str, float]) -> tuple[str, bool]:
    """
    Retourne (label_forcé, abstained).
      label_forcé : argmax des probabilités d'entailment (relation choisie).
      abstained   : True si la marge d'abstention production aurait coupé.
    On garde les deux pour distinguer la ROBUSTESSE de la relation (accord sur
    l'argmax) de la PRUDENCE du protocole (taux d'abstention en production).
    """
    ordered = sorted(ent_by_label.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    abstained = best < _MIN_ENT or best < second + _MARGIN
    return best_label, abstained


# ════════════════════════════════════════════════════════════════════════
# GOLD — clauses UCP représentatives (label par construction métier)
# ════════════════════════════════════════════════════════════════════════

GOLD: list[dict] = [
    # ── VIOLATION : la clause s'oppose / déroge au standard UCP ──────────
    {"id": "V1", "article": "Art.20c", "title": "Connaissement — transbordement",
     "clause": "Transshipment is prohibited even if effected in containers; article 20(c)(ii) of UCP 600 is not applicable.",
     "gold": "VIOLATION"},
    {"id": "V2", "article": "Art.17", "title": "Documents originaux",
     "clause": "Documents bearing scanned signatures are not acceptable even if marked as original.",
     "gold": "VIOLATION"},
    {"id": "V3", "article": "Art.18", "title": "Facture commerciale",
     "clause": "Invoices issued in the name of a third party or 'formerly known as' are not acceptable and will be rejected.",
     "gold": "VIOLATION"},
    {"id": "V4", "article": "Art.16", "title": "Documents non conformes",
     "clause": "Any discrepancy fee or reservation lifting fee will be deducted from the proceeds regardless of who bears the charges.",
     "gold": "VIOLATION"},
    {"id": "V5", "article": "Art.20", "title": "Connaissement",
     "clause": "Blank back and short form bills of lading are not acceptable under this credit.",
     "gold": "VIOLATION"},
    {"id": "V6", "article": "Art.28", "title": "Assurance",
     "clause": "Insurance documents containing the Institute Cargo Clauses restriction or any coverage restriction are not acceptable.",
     "gold": "VIOLATION"},

    # ── RESTRICTION : exigence plus stricte que le minimum UCP ───────────
    {"id": "R1", "article": "Art.14c", "title": "Délai de présentation",
     "clause": "Documents must be presented within 7 days after the date of shipment but within the validity of the credit.",
     "gold": "RESTRICTION"},
    {"id": "R2", "article": "Art.28", "title": "Assurance",
     "clause": "Insurance must cover 130 percent of the invoice value.",
     "gold": "RESTRICTION"},
    {"id": "R3", "article": "Art.31", "title": "Expéditions partielles",
     "clause": "Partial shipments are not allowed under this documentary credit.",
     "gold": "RESTRICTION"},
    {"id": "R4", "article": "Art.17", "title": "Documents originaux",
     "clause": "All documents must be signed in handwriting with a blue ink ball pen and bear an original company stamp.",
     "gold": "RESTRICTION"},
    {"id": "R5", "article": "Art.18b", "title": "Facture — montant",
     "clause": "The invoice amount must not exceed the credit amount; no tolerance whatsoever is allowed.",
     "gold": "RESTRICTION"},

    # ── COMPLIANT : aligné sur la pratique UCP usuelle ───────────────────
    {"id": "C1", "article": "Art.20c", "title": "Connaissement — transbordement",
     "clause": "Transshipment is allowed.",
     "gold": "COMPLIANT"},
    {"id": "C2", "article": "Art.31", "title": "Expéditions partielles",
     "clause": "Partial shipments are permitted.",
     "gold": "COMPLIANT"},
    {"id": "C3", "article": "Art.6", "title": "Disponibilité",
     "clause": "This credit is available by payment at sight with the issuing bank.",
     "gold": "COMPLIANT"},
    {"id": "C4", "article": "Art.14c", "title": "Délai de présentation",
     "clause": "Documents must be presented within 21 days after the date of shipment but within the validity of the credit.",
     "gold": "COMPLIANT"},
    {"id": "C5", "article": "Art.28", "title": "Assurance",
     "clause": "Insurance to be effected for 110 percent of the CIF value of the goods.",
     "gold": "COMPLIANT"},
    {"id": "C6", "article": "Art.6", "title": "Règles applicables",
     "clause": "This documentary credit is issued subject to UCP 600.",
     "gold": "COMPLIANT"},
]


# ════════════════════════════════════════════════════════════════════════
# Labelers (chacun applique LE MÊME protocole, avec un modèle différent)
# ════════════════════════════════════════════════════════════════════════

class CrossEncoderLabeler:
    """Réplique exacte du protocole production (UCPNLIRefiner) pour un CrossEncoder NLI."""

    def __init__(self, name: str, model_name: str) -> None:
        self.name = name
        self.model_name = model_name
        self._model = None
        self.available = False
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            self.available = True
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] {name} ({model_name}) indisponible : {exc}")

    def label(self, clause: str, article: str, title: str) -> tuple[str, bool]:
        import numpy as np
        p = (clause or "").strip()[:480]
        hyps = _hypotheses(article, title)
        pairs = [(p, hyps["VIOLATION"]), (p, hyps["COMPLIANT"]), (p, hyps["RESTRICTION"])]
        logits = np.asarray(self._model.predict(pairs, show_progress_bar=False), dtype=np.float64)
        if logits.ndim != 2 or logits.shape[1] < 3:
            return ABSTAIN, True

        def ent(row):  # index 1 = entailment (nli-deberta), softmax 3 classes
            x = row - np.max(row)
            e = np.exp(x)
            return float((e / (np.sum(e) + 1e-12))[1])

        return _decide({
            "VIOLATION":   ent(logits[0]),
            "COMPLIANT":   ent(logits[1]),
            "RESTRICTION": ent(logits[2]),
        })


class HFSeqClsLabeler:
    """Protocole NLI via AutoModelForSequenceClassification (ex. BART-MNLI, archi indépendante)."""

    def __init__(self, name: str, model_name: str) -> None:
        self.name = name
        self.model_name = model_name
        self._ok = False
        try:
            import torch  # noqa
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tok = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.eval()
            # Index d'entailment robuste via id2label
            id2label = self._model.config.id2label
            self._ent_idx = next(
                i for i, lbl in id2label.items() if "entail" in str(lbl).lower()
            )
            self._ok = True
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] {name} ({model_name}) indisponible : {exc}")

    @property
    def available(self) -> bool:
        return self._ok

    def _ent_prob(self, premise: str, hypothesis: str) -> float:
        import torch
        with torch.no_grad():
            inp = self._tok(premise, hypothesis, return_tensors="pt",
                            truncation=True, max_length=512)
            logits = self._model(**inp).logits[0]
            probs = torch.softmax(logits, dim=-1)
            return float(probs[self._ent_idx])

    def label(self, clause: str, article: str, title: str) -> tuple[str, bool]:
        p = (clause or "").strip()[:480]
        hyps = _hypotheses(article, title)
        return _decide({k: self._ent_prob(p, h) for k, h in hyps.items()})


# ════════════════════════════════════════════════════════════════════════
# Métriques
# ════════════════════════════════════════════════════════════════════════

def cohen_kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    """Kappa de Cohen (sklearn si dispo, sinon implémentation locale)."""
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(a, b, labels=labels))
    except Exception:
        pass
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lbl in labels:
        pa = sum(1 for x in a if x == lbl) / n
        pb = sum(1 for x in b if x == lbl) / n
        pe += pa * pb
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else float("nan")


def _binary(label: str) -> str:
    """Vue opérationnelle : la clause dévie-t-elle de l'UCP, oui ou non ?"""
    return "COMPLIANT" if label == "COMPLIANT" else "DEVIATION"


def pair_metrics(forced: dict[str, list[str]], names: tuple[str, str]) -> dict:
    """Accord/kappa sur labels FORCÉS — 3 classes ET vue binaire (déviation/non)."""
    n1, n2 = names
    a, b = forced[n1], forced[n2]
    agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    kappa = cohen_kappa(a, b, list(LABELS))
    ab, bb = [_binary(x) for x in a], [_binary(x) for x in b]
    agree_bin = sum(1 for x, y in zip(ab, bb) if x == y) / len(ab)
    kappa_bin = cohen_kappa(ab, bb, ["COMPLIANT", "DEVIATION"])
    return {
        "pair": f"{n1} ↔ {n2}",
        "agreement": round(agree, 3),
        "cohen_kappa": round(kappa, 3),
        "agreement_binary": round(agree_bin, 3),
        "cohen_kappa_binary": round(kappa_bin, 3),
        "n": len(a),
    }


def accuracy_vs_gold(forced: list[str], abstain: list[bool], gold: list[str]) -> dict:
    """Exactitude sur labels forcés + taux d'abstention production (séparés)."""
    acc = sum(1 for p, g in zip(forced, gold) if p == g) / len(gold)
    abst = sum(1 for a in abstain if a) / len(abstain)
    return {"accuracy_forced": round(acc, 3),
            "abstention_rate": round(abst, 3), "n": len(gold)}


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("CROSS-VALIDATION MULTI-MODÈLES — PROTOCOLE NLI UCP")
    print("=" * 70)
    print(f"Jeu de test : {len(GOLD)} clauses "
          f"({sum(1 for g in GOLD if g['gold']=='VIOLATION')} VIOLATION / "
          f"{sum(1 for g in GOLD if g['gold']=='RESTRICTION')} RESTRICTION / "
          f"{sum(1 for g in GOLD if g['gold']=='COMPLIANT')} COMPLIANT)\n")

    labelers = [
        CrossEncoderLabeler("A_deberta_xsmall (PROD)", "cross-encoder/nli-deberta-v3-xsmall"),
        CrossEncoderLabeler("B_deberta_small",          "cross-encoder/nli-deberta-v3-small"),
        HFSeqClsLabeler("C_bart_large_mnli",            "facebook/bart-large-mnli"),
    ]
    labelers = [l for l in labelers if getattr(l, "available", False)]
    if len(labelers) < 2:
        print("[ERREUR] Moins de 2 modèles disponibles — cross-validation impossible.")
        sys.exit(1)

    gold = [g["gold"] for g in GOLD]
    forced: dict[str, list[str]] = {}
    abstain: dict[str, list[bool]] = {}
    for lab in labelers:
        print(f"→ Étiquetage avec {lab.name} ({lab.model_name}) ...")
        results = [lab.label(g["clause"], g["article"], g["title"]) for g in GOLD]
        forced[lab.name]  = [r[0] for r in results]
        abstain[lab.name] = [r[1] for r in results]

    # Métriques par paire (sur labels forcés)
    names = [l.name for l in labelers]
    pairs = [pair_metrics(forced, (x, y)) for x, y in combinations(names, 2)]

    # Exactitude / abstention par modèle
    per_model = {n: accuracy_vs_gold(forced[n], abstain[n], gold) for n in names}

    # Désaccords (cas où les labels forcés ne sont pas tous d'accord)
    disagreements = []
    for i, g in enumerate(GOLD):
        row = {n: forced[n][i] for n in names}
        if len(set(row.values())) > 1:
            disagreements.append({"id": g["id"], "article": g["article"],
                                  "gold": g["gold"], "clause": g["clause"][:90],
                                  "preds": row})

    # ── Affichage ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("ACCORD INTER-MODÈLES (labels forcés = robustesse du protocole)")
    print("-" * 70)
    print(f"{'Paire':<44}{'3-cl.':>7}{'κ':>6} | {'binaire':>8}{'κ':>6}")
    for p in pairs:
        print(f"{p['pair']:<44}{p['agreement']:>6.0%}{p['cohen_kappa']:>6.2f} | "
              f"{p['agreement_binary']:>7.0%}{p['cohen_kappa_binary']:>6.2f}")
    print("  3-cl. = VIOLATION/RESTRICTION/COMPLIANT | binaire = DÉVIATION vs COMPLIANT")

    print("\n" + "-" * 70)
    print("EXACTITUDE vs GOLD (labels forcés) & ABSTENTION production")
    print("-" * 70)
    print(f"{'Modèle':<48}{'Exact.':>8}{'Abst.':>7}{'n':>5}")
    for n in names:
        m = per_model[n]
        print(f"{n:<48}{m['accuracy_forced']:>7.0%}{m['abstention_rate']:>7.0%}{m['n']:>5}")

    if disagreements:
        print("\n" + "-" * 70)
        print(f"DÉSACCORDS ({len(disagreements)}/{len(GOLD)} cas)")
        print("-" * 70)
        for d in disagreements:
            preds_str = ", ".join(f"{k.split('_')[0]}={v}" for k, v in d["preds"].items())
            print(f"  [{d['id']}] {d['article']} (gold={d['gold']}) → {preds_str}")
            print(f"        « {d['clause']} »")

    # Conclusion automatique (couple indépendant A↔C si présent)
    headline = next((p for p in pairs if "PROD" in p["pair"] and "bart" in p["pair"].lower()), pairs[0])
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    def interp(k: float) -> str:
        return ("quasi-parfait" if k >= 0.81 else "fort" if k >= 0.61 else
                "modéré" if k >= 0.41 else "faible" if k >= 0.21 else "négligeable")

    kp3 = headline["cohen_kappa"]
    kpb = headline["cohen_kappa_binary"]
    mean_abst = sum(per_model[n]["abstention_rate"] for n in names) / len(names)
    print(f"Couple indépendant le plus parlant : {headline['pair']}")
    print(f"  3 classes : accord {headline['agreement']:.0%}, κ={kp3:.2f} (accord {interp(kp3)})")
    print(f"  Binaire   : accord {headline['agreement_binary']:.0%}, κ={kpb:.2f} (accord {interp(kpb)})")
    print()
    print("Lecture honnête des résultats :")
    if kp3 < 0.41:
        print(f"  • La sous-distinction VIOLATION/RESTRICTION n'est PAS robuste entre")
        print(f"    familles de modèles (κ={kp3:.2f}) : elle dépend du modèle et de la")
        print(f"    frontière intrinsèquement floue entre 'déroge' et 'durcit'.")
    if kpb >= kp3 + 0.1:
        print(f"  • En revanche la décision opérationnelle 'la clause dévie-t-elle de")
        print(f"    l'UCP ?' est plus stable (κ={kpb:.2f}) — c'est elle qui pilote l'alerte.")
    print(f"  • Abstention production moyenne = {mean_abst:.0%} : le NLI ne tranche que")
    print(f"    les cas très nets ; sa fragilité de sous-classe se propage donc peu.")
    print("  → Recommandation : ne pas exposer VIOLATION vs RESTRICTION comme un verdict")
    print("    ferme ; garder le NLI en raffineur prudent, fusionner les deux en")
    print("    'déviation contractuelle' pour l'affichage. (Axe d'amélioration documenté.)")

    # ── Rapport JSON ──────────────────────────────────────────────────────
    report = {
        "dataset_size": len(GOLD),
        "label_distribution": {
            l: sum(1 for g in GOLD if g["gold"] == l) for l in LABELS
        },
        "models": [{"name": l.name, "model": l.model_name} for l in labelers],
        "pairwise_forced": pairs,
        "per_model_vs_gold": per_model,
        "forced_predictions": {n: forced[n] for n in names},
        "abstention_flags": {n: abstain[n] for n in names},
        "gold": gold,
        "disagreements": disagreements,
        "headline_pair": headline,
    }
    out = OUT_DIR / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport écrit : {out}")


if __name__ == "__main__":
    main()
