# -*- coding: utf-8 -*-
"""
Évaluation du TAUX DE DÉTECTION — Incohérences + Validation UCP 600
===================================================================

But
----
Mesurer, sur un *gold synthétique par injection*, dans quelle proportion le
système détecte :

  [A] les VIOLATIONS / MODIFICATIONS UCP 600 (findings par article)
        → moteur UCP600Validator (mode legacy, déterministe)
  [B] les INCOHÉRENCES tag SWIFT ↔ clause (contradictions Art.5)
        → SwiftSemanticAnalyzer.ContradictionEngine
  [C] les incohérences inter-champs (dates) — cross-field

Principe du gold
----------------
On part d'un MT700 PROPRE (aucune anomalie). Pour chaque cas de test, on
injecte EXACTEMENT une anomalie connue (clause, tag ou date). Le label
(l'article UCP attendu, ou le champ en contradiction) est donc connu *par
construction*. On compare ensuite ce que le système remonte.

Métriques
---------
  • Recall (taux de détection) = cas où l'anomalie injectée est bien détectée
  • Precision                  = findings remontés correspondant à une injection
                                 réelle (hors bruit de fond du LC propre)
  • Détail par article UCP, par type d'anomalie, et liste des manqués (FN).

PÉRIMÈTRE / HYPOTHÈSE
---------------------
On évalue la COUCHE de détection UCP + contradictions en lui fournissant les
clauses directement (comme le pipeline le fait après segmentation). La
segmentation de clauses et le NER ont leurs propres évaluations
(evaluation/clauses, evaluation/swift_ner) — ils ne sont pas réévalués ici.

Usage
-----
    cd swift-validator-ia
    python evaluation/ucp_coherence/evaluate_ucp_coherence.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Console Windows (cp1252) : forcer UTF-8 pour les pictogrammes du résumé.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Mode legacy = déterministe (regex + exclusions), pas de dépendance embeddings.
# Garantit une évaluation reproductible de la couche de règles UCP.
os.environ.setdefault("UCP_USE_LEGACY_RULES", "true")

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval_ucp_coherence")
log.setLevel(logging.INFO)

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════════════════════════════════
# 1. MT700 PROPRE DE RÉFÉRENCE (aucune anomalie attendue)
# ════════════════════════════════════════════════════════════════════════

BASE_TEXT = """:27: 1/1
:40A: IRREVOCABLE
:20: EVALLC0001
:23: CREDIT NUMBER
:31C: 260115
:40E: UCP 600
:31D: 261231 HAMBURG GERMANY
:50: KESTREL TRADING GMBH
:59: AURORA FROZEN GOODS BV
:32B: EUR250000
:39A: 00/00
:41A: DEUTDEFFXXX BY PAYMENT
:42C: AT SIGHT
:43P: ALLOWED
:43T: NOT ALLOWED
:44A: PORT OF LOADING: HAMBURG
:44B: PORT OF DISCHARGE: ROTTERDAM
:44C: 261120
:45A: COTTON YARN 30S COUNT 220 BALES CIF ROTTERDAM
:46A: 1. SIGNED COMMERCIAL INVOICE 2. FULL SET BILL OF LADING 3. PACKING LIST
:47A: ADDITIONAL CONDITIONS: NONE
:71B: CHARGES FOR APPLICANT
:49: WITHOUT
"""

def base_entities() -> dict:
    return {
        "lc_number":         "EVALLC0001",
        "swift_type":        "MT700",
        "applicable_rules":  "UCP 600",
        "expiry_date":       "2026-12-31",
        "expiry_place":      "HAMBURG GERMANY",
        "available_by":      "BY PAYMENT",
        "amount":            "250000",
        "currency":          "EUR",
        "incoterm":          "CIF",
        "transhipment":      "NOT ALLOWED",
        "partial_shipments": "ALLOWED",
        "tolerance_plus":    "0",
        "tolerance_minus":   "0",
        "_ocr_text":         BASE_TEXT,
    }

def base_lc_criteria() -> dict:
    return {
        "available_by": "BY PAYMENT",
        "documents": [
            {"name": "COMMERCIAL INVOICE"},
            {"name": "BILL OF LADING"},
            {"name": "PACKING LIST"},
        ],
    }


# ════════════════════════════════════════════════════════════════════════
# 2. CATALOGUE D'INJECTIONS (gold par construction)
# ════════════════════════════════════════════════════════════════════════
#
# Chaque entrée :
#   id          identifiant lisible
#   detector    "ucp" | "contradiction"
#   category    famille d'anomalie (pour l'agrégation)
#   clause      clause ajoutée à nlp.clauses (et au texte agrégé)
#   text_extra  fragment ajouté à _ocr_text (pour les règles "hard" plein-texte)
#   entities    surcharges d'entités (tags SWIFT)
#   expect_article  (detector=ucp)          article UCP attendu, ex "Art.28"
#   expect_field    (detector=contradiction) tag_field attendu, ex "transhipment"

INJECTIONS: list[dict] = [
    # ── A. Violations UCP basées clause ──────────────────────────────────
    {"id": "art28_franchise", "detector": "ucp", "category": "violation_insurance",
     "clause": "The insurance policy must not contain any franchise or excess deductible clause.",
     "expect_article": "Art.28"},
    {"id": "art28_coverage_restriction", "detector": "ucp", "category": "violation_insurance",
     "clause": "Coverage is restricted to Institute Cargo Clauses C only; Institute War Risks are excluded.",
     "expect_article": "Art.28"},
    {"id": "art17_scanned", "detector": "ucp", "category": "violation_authentication",
     "clause": "Scanned documents bearing a scanned signature are acceptable for presentation.",
     "expect_article": "Art.17"},
    {"id": "art18_third_party", "detector": "ucp", "category": "discrepancy_invoice",
     "clause": "The commercial invoice may be issued in the name of a third party.",
     "expect_article": "Art.18"},
    {"id": "art20_blank_back", "detector": "ucp", "category": "violation_transport",
     "clause": "A blank back bill of lading is acceptable under this credit.",
     "expect_article": "Art.20"},
    {"id": "art22_charter", "detector": "ucp", "category": "violation_transport",
     "clause": "Charter party bill of lading is not acceptable.",
     "expect_article": "Art.22"},
    {"id": "art10_partial_amendment", "detector": "ucp", "category": "violation_amendment",
     "clause": "Partial acceptance of amendment is allowed under this credit.",
     "expect_article": "Art.10"},
    {"id": "art16_reservation_fee", "detector": "ucp", "category": "charges",
     "clause": "A reservation lifting fee of EUR 150 per set of documents will be deducted.",
     "expect_article": "Art.16"},

    # ── B. Modifications contractuelles (exclusions d'articles) ──────────
    {"id": "art18_excluded", "detector": "ucp", "category": "exclusion_critical",
     "clause": "ARTICLE 18 OF UCP 600 IS NOT APPLICABLE.",
     "text_extra": "ARTICLE 18 OF UCP 600 IS NOT APPLICABLE.",
     "expect_article": "Art.18"},
    {"id": "art38i_excluded", "detector": "ucp", "category": "exclusion_critical",
     "clause": "ARTICLE 38(I) OF UCP 600 IS NOT APPLICABLE.",
     "text_extra": "ARTICLE 38(I) OF UCP 600 IS NOT APPLICABLE.",
     "expect_article": "Art.38i"},
    {"id": "art14k_excluded", "detector": "ucp", "category": "exclusion",
     "clause": "Article 14k of UCP 600 is excluded.",
     "text_extra": "Article 14k of UCP 600 is excluded.",
     "expect_article": "Art.14k"},
    {"id": "art39_assignment_prohibited", "detector": "ucp", "category": "exclusion",
     "clause": "Assignment of proceeds is not allowed under this credit.",
     "expect_article": "Art.39"},

    # ── C. Incohérences tag SWIFT ↔ clause (Art.5) — double canal ────────
    # Vu côté UCP (ART20C_CLAUSE_OVERRIDES_TAG, Art.20c)
    {"id": "transhipment_contradiction_ucp", "detector": "ucp", "category": "contradiction_tag_clause",
     "clause": "Transshipment is prohibited even if effected by container.",
     "entities": {"transhipment": "ALLOWED"},
     "expect_article": "Art.20c"},
    # Vu côté analyseur sémantique (contradiction tag_field=transhipment)
    {"id": "transhipment_contradiction_sem", "detector": "contradiction", "category": "contradiction_tag_clause",
     "text_extra": ":47A: TRANSSHIPMENT IS PROHIBITED.",
     "entities": {"transhipment": "ALLOWED"},
     "expect_field": "transhipment"},
    {"id": "partial_contradiction_sem", "detector": "contradiction", "category": "contradiction_tag_clause",
     "text_extra": ":47A: PARTIAL SHIPMENTS IS NOT ALLOWED.",
     "entities": {"partial_shipments": "ALLOWED"},
     "expect_field": "partial_shipments"},

    # ── D. Anomalies basées tags/entités ────────────────────────────────
    {"id": "art6_no_expiry", "detector": "ucp", "category": "missing_field",
     "entities": {"expiry_date": ""},
     "expect_article": "Art.6"},
    {"id": "art31_partial_prohibited", "detector": "ucp", "category": "shipment_constraint",
     "entities": {"partial_shipments": "PROHIBITED"},
     "expect_article": "Art.31"},
    {"id": "art30_tolerance_conflict", "detector": "ucp", "category": "tolerance_conflict",
     "clause": "Quantity greater than LC conditions is not acceptable.",
     "entities": {"tolerance_plus": "5", "tolerance_minus": "5"},
     "expect_article": "Art.30"},

    # ── E. Cohérence inter-champs (dates) ───────────────────────────────
    # Texte explicite (voie hard pattern UCP_WARN_SHIPMENT_VS_EXPIRY)
    {"id": "xfield_shipment_after_expiry_text", "detector": "ucp", "category": "cross_field_dates",
     "text_extra": "LATEST SHIPMENT DATE IS AFTER THE EXPIRY DATE :31D:.",
     "expect_article": "Art.6"},
    # Voie structurée (dates entités) — teste evaluate_cross_field_rules
    {"id": "xfield_shipment_after_expiry_dates", "detector": "ucp", "category": "cross_field_dates",
     "entities": {"expiry_date": "2026-11-01",
                  "latest_shipment_date": "2026-12-01",
                  "shipment_date": "2026-12-01"},
     "expect_article": "Art.6"},
]


# ════════════════════════════════════════════════════════════════════════
# 3. EXÉCUTION DES DÉTECTEURS
# ════════════════════════════════════════════════════════════════════════

def _build_payload(injection: dict) -> dict:
    ent = base_entities()
    ent.update(injection.get("entities", {}))
    text = BASE_TEXT
    if injection.get("text_extra"):
        text = BASE_TEXT + "\n" + injection["text_extra"]
    ent["_ocr_text"] = text
    clauses = []
    if injection.get("clause"):
        clauses.append({"text": injection["clause"]})
    return {
        "swift": {
            "entities":    ent,
            "lc_criteria": base_lc_criteria(),
            "nlp":         {"clauses": clauses},
        },
        "ocr": {"text": text},
    }


def run_ucp(validator, injection: dict) -> tuple[set[str], list[dict]]:
    """Retourne (articles détectés, findings bruts)."""
    payload = _build_payload(injection)
    report  = validator.validate(payload)
    d       = report.to_dict()
    findings = d.get("findings", []) or []
    articles = {f.get("article", "") for f in findings}
    return articles, findings


def run_contradiction(analyzer, injection: dict) -> tuple[set[str], list[dict]]:
    """Retourne (tag_fields en contradiction, contradictions brutes)."""
    ent = base_entities()
    ent.update(injection.get("entities", {}))
    text = BASE_TEXT
    if injection.get("text_extra"):
        text = BASE_TEXT + "\n" + injection["text_extra"]
    if injection.get("clause"):
        text = text + "\n" + injection["clause"]
    res = analyzer.analyze(ent, text)
    conts = [c.to_dict() if hasattr(c, "to_dict") else dict(c)
             for c in (getattr(res, "contradictions", []) or [])]
    fields = {c.get("tag_field", "") for c in conts}
    return fields, conts


# ════════════════════════════════════════════════════════════════════════
# 4. ÉVALUATION
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    from services.swift_nlp.ucp.swift_ucp_validator import UCP600Validator
    from services.swift_nlp.semantic.swift_semantic_analyzer import SwiftSemanticAnalyzer

    log.info("Chargement UCP600Validator (mode legacy) + SwiftSemanticAnalyzer…")
    validator = UCP600Validator()
    analyzer  = SwiftSemanticAnalyzer(enable_semantic_coherence=False)

    # ── Bruit de fond : le LC propre, sans injection ────────────────────
    clean = {"id": "__clean__", "detector": "ucp", "category": "baseline"}
    baseline_articles, baseline_findings = run_ucp(validator, clean)
    baseline_fields, _ = run_contradiction(analyzer, clean)
    log.info("Bruit de fond UCP (LC propre) : %d findings → articles=%s",
             len(baseline_findings), sorted(baseline_articles) or "∅")
    log.info("Bruit de fond contradictions  : %s", sorted(baseline_fields) or "∅")

    results: list[dict] = []
    tp = fp = fn = 0
    per_article = defaultdict(lambda: {"expected": 0, "detected": 0})
    per_category = defaultdict(lambda: {"expected": 0, "detected": 0})

    for inj in INJECTIONS:
        if inj["detector"] == "ucp":
            detected_set, raw = run_ucp(validator, inj)
            target = inj["expect_article"]
            hit = target in detected_set
            extra = detected_set - baseline_articles - {target}
            key = target
        else:
            detected_set, raw = run_contradiction(analyzer, inj)
            target = inj["expect_field"]
            hit = target in detected_set
            extra = detected_set - baseline_fields - {target}
            key = f"contradiction:{target}"

        # extra = findings/contradictions déclenchés en plus de l'attendu et
        # du bruit de fond → faux positifs potentiels.
        extra = {e for e in extra if e}
        tp += int(hit)
        fn += int(not hit)
        fp += len(extra)
        per_article[key]["expected"] += 1
        per_article[key]["detected"] += int(hit)
        per_category[inj["category"]]["expected"] += 1
        per_category[inj["category"]]["detected"] += int(hit)

        results.append({
            "id":        inj["id"],
            "detector":  inj["detector"],
            "category":  inj["category"],
            "target":    key,
            "detected":  hit,
            "extra":     sorted(extra),
            "detected_set": sorted(x for x in detected_set if x),
        })
        status = "OK " if hit else "MISS"
        log.info("[%s] %-34s target=%-22s extra=%s",
                 status, inj["id"], key, sorted(extra) or "∅")

    n = len(INJECTIONS)
    recall    = tp / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    # ── Recall par type de détecteur ────────────────────────────────────
    by_detector = defaultdict(lambda: [0, 0])  # [detected, total]
    for r in results:
        by_detector[r["detector"]][1] += 1
        by_detector[r["detector"]][0] += int(r["detected"])

    report = {
        "scope": "Détection incohérences (contradictions/cross-field) + findings UCP par article",
        "gold":  "synthétique par injection (label connu par construction)",
        "ucp_mode": os.environ.get("UCP_USE_LEGACY_RULES") and "legacy" or "semantic",
        "n_cases": n,
        "overall": {
            "recall":        round(recall, 4),
            "precision":     round(precision, 4),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
        },
        "by_detector": {
            k: {"recall": round(v[0] / v[1], 4) if v[1] else 0.0,
                "detected": v[0], "total": v[1]}
            for k, v in by_detector.items()
        },
        "by_category": {
            k: {"recall": round(v["detected"] / v["expected"], 4) if v["expected"] else 0.0,
                **v}
            for k, v in sorted(per_category.items())
        },
        "by_target": {
            k: {"recall": round(v["detected"] / v["expected"], 4) if v["expected"] else 0.0,
                **v}
            for k, v in sorted(per_article.items())
        },
        "baseline_noise": {
            "ucp_articles":   sorted(x for x in baseline_articles if x),
            "contradictions": sorted(x for x in baseline_fields if x),
        },
        "misses": [r["id"] for r in results if not r["detected"]],
        "cases": results,
    }

    out_json = OUT_DIR / "report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _plot(report)

    # ── Console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(" ÉVALUATION — TAUX DE DÉTECTION incohérences + UCP")
    print("=" * 68)
    print(f"  Cas de test           : {n}")
    print(f"  Recall (détection)    : {recall:6.1%}  ({tp}/{n})")
    print(f"  Precision             : {precision:6.1%}  (FP={fp})")
    print(f"  Manqués (FN)          : {fn}")
    print("  ─────────────────────────────────────────────────────────────")
    for k, v in report["by_detector"].items():
        print(f"  recall[{k:13s}] : {v['recall']:6.1%}  ({v['detected']}/{v['total']})")
    print("  ─────────────────────────────────────────────────────────────")
    for k, v in report["by_category"].items():
        print(f"  {k:26s} : {v['recall']:6.1%}  ({v['detected']}/{v['expected']})")
    if report["misses"]:
        print("  ─────────────────────────────────────────────────────────────")
        print("  MANQUÉS :", ", ".join(report["misses"]))
    print("=" * 68)
    print(f"  Rapport JSON : {out_json}")
    print(f"  Figures      : {OUT_DIR}")


def _plot(report: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib indisponible — figures ignorées (%s)", exc)
        return

    cats = report["by_category"]
    names  = list(cats.keys())
    recall = [cats[k]["recall"] for k in names]
    colors = ["#2ecc71" if r >= 0.999 else "#e67e22" if r > 0 else "#e74c3c" for r in recall]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(names))))
    ax.barh(names, recall, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall (taux de détection)")
    ax.set_title("Taux de détection par type d'anomalie — incohérences + UCP")
    for i, r in enumerate(recall):
        ax.text(min(r + 0.02, 0.92), i, f"{r:.0%}", va="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recall_by_category.png", dpi=120)
    plt.close(fig)

    ov = report["overall"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["TP", "FN", "FP"],
           [ov["true_positive"], ov["false_negative"], ov["false_positive"]],
           color=["#2ecc71", "#e74c3c", "#e67e22"])
    ax.set_title(f"Recall={ov['recall']:.0%}  Precision={ov['precision']:.0%}")
    ax.set_ylabel("Nombre de cas")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_summary.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
