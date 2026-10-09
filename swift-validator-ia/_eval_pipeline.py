# -*- coding: utf-8 -*-
"""Ad-hoc driver: test the post-OCR MT700 pipeline (extraction -> UCP -> coherence -> DS -> risk ML)."""
import json, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pipeline.swift_pipeline import SwiftPipeline

# Realistic MT700 sample (clean, low-risk LC). 48 presentation period bumped to test concern #3.
MT700 = """:27: 1/1
:40A: IRREVOCABLE
:20: LC4507963309
:23: CREDIT NUMBER
:31C: 251018
:40E: UCP 600
:31D: 251215 HAMBURG GERMANY
:52A: DEUTSCHE BANK AG
:50: KESTREL TRADING GMBH
  58 LYON STREET
  HAMBURG GERMANY
:59: AURORA FROZEN GOODS BV
  PO BOX 288
  ROTTERDAM NETHERLANDS
:32B: EUR250000
:39A: 10/10
:41A: DEUTDEFFXXX BY PAYMENT
:42C: AT SIGHT
:43P: PARTIAL SHIPMENTS ALLOWED
:43T: TRANSSHIPMENT NOT ALLOWED
:44A: PORT OF LOADING: HAMBURG
:44B: PORT OF DISCHARGE: ROTTERDAM
:44C: 251120
:45A: COTTON YARN 30S COUNT 220 BALES CIF ROTTERDAM
:46A: DOCUMENTS REQUIRED:
  1. SIGNED COMMERCIAL INVOICE 3 COPIES
  2. FULL SET BILL OF LADING
  3. CERTIFICATE OF ORIGIN 1 COPY
  4. PACKING LIST
:47A: ADDITIONAL CONDITIONS:
  1. PRESENTATION PERIOD 46 DAYS FROM SHIPMENT DATE
:57A: DEUTDEFFXXX
:71B: ALL CHARGES OUTSIDE ISSUING BANK FOR BENEFICIARY
:48: 46 DAYS
:49: WITHOUT
"""

def dig(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur

pipe = SwiftPipeline()
res = pipe.analyze_text(MT700)

swift = res.get("swift", {}) or {}
print("=" * 70)
print("SUCCESS:", res.get("success"), "| swift_type:", swift.get("swift_type"))

# --- MT700 validation (rules) ---
val = swift.get("validation", {}) or {}
print("\n--- [VAL] MT700 SwiftValidator ---")
print(json.dumps({k: val.get(k) for k in ("is_valid","score","completeness_score","errors","warnings","missing_fields")}, ensure_ascii=False, indent=2)[:1500])

# --- NLP / semantic (concern #1) ---
sem = dig(swift, "nlp", "semantic", default={}) or {}
print("\n--- [NLP] semantic (concern #1: low score) ---")
print(json.dumps({
    "nlp_score": dig(swift, "nlp", "score"),
    "semantic_score": sem.get("semantic_score"),
    "completeness_score": sem.get("completeness_score"),
    "doc_quality_score": sem.get("doc_quality_score"),
    "bayesian_confidence": sem.get("bayesian_confidence"),
    "final_score": sem.get("final_score"),
    "routing_decision": sem.get("routing_decision"),
    "n_warnings": len(sem.get("warnings", [])),
    "n_risk_flags": len(sem.get("risk_flags", [])),
    "n_contradictions": len(sem.get("contradictions", [])),
}, ensure_ascii=False, indent=2))

# --- UCP compliance ---
ucp = res.get("ucp_compliance", {}) or {}
print("\n--- [UCP] ucp_compliance ---")
print(json.dumps({
    "global_score": ucp.get("global_score"),
    "ucp_decision": ucp.get("ucp_decision"),
    "summary": ucp.get("summary"),
    "n_errors": len(ucp.get("errors", [])),
    "n_warnings": len(ucp.get("warnings", [])),
}, ensure_ascii=False, indent=2))

# --- DS analysis (concern #4 country_risk=25) ---
ds = res.get("ds_analysis", {}) or {}
print("\n--- [DS] ds_analysis ---")
print(json.dumps({
    "risk": ds.get("risk"),
    "features": ds.get("features"),
}, ensure_ascii=False, indent=2)[:1800])

# --- external validation ---
ext = res.get("external_validation", {}) or {}
print("\n--- [EXT] external_validation ---")
print(json.dumps({k: ext.get(k) for k in ("risk_level","risk_score","approach","_stub")}, ensure_ascii=False, indent=2))

# --- Risk ML (concerns #2,#3,#4,#5) ---
rml = res.get("risk_ml", {}) or {}
print("\n--- [RISK ML] risk_ml ---")
print(json.dumps({
    "risk_label": rml.get("risk_label"),
    "risk_score": rml.get("risk_score"),
    "risk_proba": rml.get("risk_proba"),
    "is_anomaly": rml.get("is_anomaly"),
    "segment": rml.get("segment"),
    "error": rml.get("error"),
}, ensure_ascii=False, indent=2))
feats = rml.get("features_used", {}) or {}
print("\nrisk_ml.features_used (key ones):")
for k in ("bank_tier","country_risk_score","country_risk_applicant","country_risk_beneficiary",
          "presentation_days","validity_days","tenor_risk","semantic_score","nlp_score",
          "ucp_compliance_score","completeness_score","missing_critical_score","amount_tier"):
    print(f"   {k:28s} = {feats.get(k)}")
bd = rml.get("score_breakdown", {}) or {}
print("\nrisk_ml.score_breakdown.analytical_score =", bd.get("analytical_score"), "label=", bd.get("analytical_label"))
dims = bd.get("dimensions", {}) or {}
for k, v in dims.items():
    print(f"   dim {k:16s} score={v.get('score')} contrib={v.get('contribution')}")
temporal = dims.get("temporal", {})
print("\ntemporal detail:", json.dumps(temporal, ensure_ascii=False))
print("\nnarrative:", rml.get("narrative"))

# --- detail: what fired the HIGH override + low semantic score ---
print("\n" + "=" * 70)
print("DETAIL — triggers")
print("--- semantic.warnings ---")
for w in sem.get("warnings", []):
    print(f"   [{w.get('severity')}] {w.get('code')} ({w.get('field')}/{w.get('ucp_ref')}): {w.get('message')}")
print("--- semantic.risk_flags ---")
for r in sem.get("risk_flags", []):
    print(f"   [{r.get('severity')}] {r.get('code')} ({r.get('ucp_ref')}): {r.get('description')}")
print("--- semantic.contradictions ---")
for c in sem.get("contradictions", []):
    print(f"   {json.dumps(c, ensure_ascii=False)[:300]}")
print("--- semantic.cross_field_coherence.issues ---")
cf = sem.get("cross_field_coherence", {}) or {}
print("   verdict:", cf.get("verdict"), "score:", cf.get("coherence_score"))
for i in cf.get("issues", []):
    print(f"   [{i.get('severity')}] {i.get('code')} {i.get('fields')} conf={i.get('confidence')}: {i.get('message')}")
print("--- ucp_compliance.errors + warnings ---")
for e in (ucp.get("errors", []) or []):
    print(f"   ERROR: {json.dumps(e, ensure_ascii=False)[:250]}")
for w in (ucp.get("warnings", []) or []):
    print(f"   WARN: {json.dumps(w, ensure_ascii=False)[:250]}")
print("--- risk_ml.ucp_violations (rule engine — what forced the label) ---")
for v in rml.get("ucp_violations", []):
    print(f"   [{v.get('severity')}] {v.get('article')} ({v.get('field')}) override={v.get('override_score')}: {v.get('description')}")

with open("_eval_result.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print("\nFull result dumped to _eval_result.json")
