"""
streamlit_complete.py  —  v9.0
===============================
Plateforme complète LC / Trade Finance — 3 modules intégrés :

  ① Validation MT700   — pipeline complet (OCR · NLP · UCP 600 · Risk ML · Externe)
  ② Validation Docs    — BL + Invoice + Packing List (règles + NLI + sémantique + tables)
  ③ Cohérence LC↔Docs  — 4 couches : règles UCP · matching flou · sémantique 45A · conditions 47A

Lancer : streamlit run app/streamlit_complete.py
"""

import json
import math
import requests
import streamlit as st

API_BASE              = "http://localhost:5000"
PIPELINE_URL          = f"{API_BASE}/pipeline"
COMMERCIAL_URL        = f"{API_BASE}/commercial/validate"
LC_COMMERCIAL_URL     = f"{API_BASE}/lc-commercial/validate"
HEALTH_URL            = f"{API_BASE}/health"
COMM_HEALTH_URL       = f"{API_BASE}/commercial/health"
LC_COMM_HEALTH_URL    = f"{API_BASE}/lc-commercial/health"

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LC Trade Validator v9",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base ── */
section.main > div { padding-top: 1rem; }

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #0f172a 100%);
  border-radius: 14px; padding: 24px 32px; margin-bottom: 20px;
  border: 1px solid #334155;
}
.hero h1 { font-size: 22px; font-weight: 700; color: #f8fafc; margin: 0 0 4px; }
.hero p  { font-size: 12px; color: #94a3b8; margin: 0; }

/* ── Score card ── */
.score-banner {
  border-radius: 12px; padding: 18px 24px; margin-bottom: 16px;
  border: 1px solid; display: flex; align-items: center; justify-content: space-between;
}
.score-val { font-size: 2.5rem; font-weight: 800; }

/* ── KPI grid ── */
.kpi-grid { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
.kpi-card {
  flex:1; min-width:120px;
  background:#1e293b; border:1px solid #334155;
  border-radius:10px; padding:12px 14px; text-align:center;
}
.kpi-label { font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:.5px; margin-bottom:4px; }
.kpi-val   { font-size:20px; font-weight:700; color:#f1f5f9; }
.kpi-sub   { font-size:11px; color:#94a3b8; }

/* ── Rule card ── */
.rule-card {
  border-left:3px solid; border-radius:8px; padding:10px 14px;
  margin-bottom:6px;
}
.rule-card .rule-header {
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
}
.rule-name { font-size:13px; font-weight:600; color:#e2e8f0; flex:1; }
.rule-msg  { font-size:12px; color:#cbd5e1; margin:4px 0 0; }
.rule-vals { font-size:11px; color:#94a3b8; margin-top:4px; }

/* ── Badge ── */
.badge {
  display:inline-block; padding:2px 8px; border-radius:12px;
  font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.5px;
}
.badge-critical { background:#450a0a; color:#ff4444; }
.badge-high     { background:#7f1d1d; color:#fca5a5; }
.badge-medium   { background:#78350f; color:#fde68a; }
.badge-low      { background:#1a2e1a; color:#86efac; }
.badge-pass     { background:#064e3b; color:#6ee7b7; }
.badge-fail     { background:#7f1d1d; color:#fca5a5; }
.badge-warning  { background:#78350f; color:#fde68a; }
.badge-skipped  { background:#1e293b; color:#94a3b8; }
.badge-doc      { background:#1e3a5f; color:#7dd3fc; }

/* ── Breakdown bar ── */
.bdown-wrap { margin-bottom:10px; }
.bdown-header { display:flex; justify-content:space-between; font-size:12px; color:#94a3b8; margin-bottom:4px; }
.bdown-bar-bg { height:8px; background:#334155; border-radius:4px; overflow:hidden; }
.bdown-bar-fill { height:100%; border-radius:4px; }

/* ── Fuzzy card ── */
.fuzzy-card { background:#1e293b; border-radius:8px; padding:12px 14px; margin-bottom:6px; }
.fuzzy-match-bar { height:6px; background:#334155; border-radius:3px; overflow:hidden; margin-top:6px; }
.fuzzy-match-fill { height:100%; border-radius:3px; }

/* ── Semantic card ── */
.sem-card { background:#1e293b; border-radius:8px; padding:14px 16px; margin-bottom:8px; }
.sem-title { font-size:13px; font-weight:600; color:#e2e8f0; margin-bottom:8px; }
.sem-bar-bg { height:8px; background:#334155; border-radius:4px; overflow:hidden; margin:6px 0 4px; }
.sem-bar-fill { height:100%; border-radius:4px; }

/* ── Exp card ── */
.exp-card {
  border-left:3px solid; border-radius:8px; padding:8px 12px; margin-bottom:4px;
  display:flex; align-items:flex-start; gap:10px;
}
.exp-msg { font-size:12px; color:#cbd5e1; }
.exp-badges { display:flex; gap:4px; margin-top:4px; flex-wrap:wrap; }

/* ── Doc card ── */
.doc-card {
  background:#0f172a; border:1px solid #1e293b;
  border-radius:10px; padding:14px 18px; margin-bottom:10px;
}
.doc-type-badge {
  display:inline-block; padding:3px 10px; border-radius:16px;
  font-size:11px; font-weight:700; text-transform:uppercase;
}

/* ── NLI card ── */
.nli-card { border-radius:8px; padding:10px 14px; margin-bottom:6px; }
.nli-label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; }
.nli-desc  { font-size:12px; color:#94a3b8; margin-top:2px; }

/* ── Status colors (light context) ── */
.c-pass     { color:#10b981; }
.c-fail     { color:#ef4444; }
.c-warning  { color:#f59e0b; }
.c-skipped  { color:#64748b; }
.c-entail   { color:#10b981; }
.c-contra   { color:#ef4444; }
.c-neutral  { color:#94a3b8; }

/* ── UCP finding ── */
.ucp-finding {
  border-left:3px solid; padding:8px 12px; margin-bottom:4px;
  border-radius:0 6px 6px 0; background:#1e293b33;
}

/* ── Risk gauge ── */
.gauge-score {
  font-size:54px; font-weight:800; text-align:center;
  background:linear-gradient(135deg,#10b981,#f59e0b,#ef4444);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _pct(v, digits=0):
    if v is None: return "—"
    return f"{round(v * 100, digits)}%"

def _score_color(v: float) -> str:
    if v >= 0.80: return "#10b981"
    if v >= 0.65: return "#f59e0b"
    return "#ef4444"

def _score_color_int(s: int) -> str:
    if s >= 80: return "#10b981"
    if s >= 50: return "#f59e0b"
    return "#ef4444"

def _risk_colors(level: str) -> dict:
    k = (level or "").lower()
    m = {
        "low":      {"bg": "#064e3b", "fg": "#6ee7b7"},
        "medium":   {"bg": "#78350f", "fg": "#fde68a"},
        "high":     {"bg": "#7f1d1d", "fg": "#fca5a5"},
        "critical": {"bg": "#450a0a", "fg": "#ff4444"},
    }
    return m.get(k, {"bg": "#1e293b", "fg": "#94a3b8"})

def _risk_icon(level: str) -> str:
    k = (level or "").lower()
    return {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨"}.get(k, "⚪")

def _status_icon(s: str) -> str:
    return {"pass": "✅", "fail": "❌", "warning": "⚠️", "skipped": "⏭️",
            "corrected": "🔧"}.get(s, "⏭️")

def _status_color(s: str) -> str:
    return {"pass": "#10b981", "fail": "#ef4444", "warning": "#f59e0b",
            "corrected": "#6366f1"}.get(s, "#475569")

def _status_bg(s: str) -> str:
    return {"pass": "#064e3b22", "fail": "#7f1d1d22",
            "warning": "#78350f22"}.get(s, "#1e293b22")

def _sev_class(sev: str) -> str:
    return f"badge-{sev}" if sev in ("critical","high","medium","low") else "badge-low"

def _badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{text}</span>'

def _doc_label(dt: str) -> str:
    return {"bl": "Bill of Lading", "invoice": "Invoice",
            "packing_list": "Packing List"}.get(dt, dt.upper())

def _doc_color(dt: str) -> str:
    return {"bl": "#1e3a5f", "invoice": "#1a2e1a",
            "packing_list": "#3b1f00"}.get(dt, "#1e293b")

def _doc_fg(dt: str) -> str:
    return {"bl": "#7dd3fc", "invoice": "#86efac",
            "packing_list": "#fde68a"}.get(dt, "#94a3b8")

def _layer_color(k: str) -> str:
    return {
        # v3.0 category keys
        "financial":   "#10b981",
        "goods":       "#6366f1",
        "documentary": "#ec4899",
        "shipment":    "#f59e0b",
        "parties":     "#3b82f6",
        # legacy layer keys (backward compat)
        "rules": "#f59e0b", "fuzzy": "#3b82f6",
        "semantic": "#6366f1", "conditions": "#ec4899",
    }.get(k, "#6b7280")

def _layer_label(k: str) -> str:
    return {
        # v3.0 category keys
        "financial":   "💰 Financier",
        "goods":       "📦 Marchandises",
        "documentary": "📋 Documentaire",
        "shipment":    "🚢 Transport",
        "parties":     "👥 Parties",
        # legacy layer keys (backward compat)
        "rules": "Règles UCP 600", "fuzzy": "Entités (Flou)",
        "semantic": "Sémantique 45A", "conditions": "Conditions 47A",
    }.get(k, k)

def _impact_icon(imp: str) -> str:
    return {"positive": "✅", "negative": "❌", "warning": "⚠️"}.get(imp, "ℹ️")

def _impact_color(imp: str) -> str:
    return {"positive": "#10b981", "negative": "#ef4444", "warning": "#f59e0b"}.get(imp, "#475569")


# ── JSON Explorer helpers ─────────────────────────────────────────────

def _jx_count_keys(obj, depth: int = 0) -> tuple[int, int]:
    if isinstance(obj, dict):
        total, max_d = len(obj), depth
        for v in obj.values():
            c, d = _jx_count_keys(v, depth + 1)
            total += c
            max_d = max(max_d, d)
        return total, max_d
    if isinstance(obj, list):
        total, max_d = 0, depth
        for item in obj[:20]:
            c, d = _jx_count_keys(item, depth)
            total += c
            max_d = max(max_d, d)
        return total, max_d
    return 0, depth


def _jx_matches(obj, q: str) -> bool:
    q = q.lower()
    if isinstance(obj, dict):
        return any(q in str(k).lower() or _jx_matches(v, q) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_jx_matches(i, q) for i in obj)
    return q in str(obj).lower()


# Generic icon/color mapping — covers all result types in this app
_JX_META: dict[str, tuple[str, str, str]] = {
    # MT700 / pipeline
    "swift":               ("🔑", "SWIFT / LC Data",        "#3b82f6"),
    "ocr":                 ("🔍", "OCR",                    "#8b5cf6"),
    "ucp_compliance":      ("⚖️",  "UCP 600 Compliance",    "#f59e0b"),
    "ds_analysis":         ("🤖", "Data Science",           "#10b981"),
    "external_validation": ("🌐", "Validation Externe",     "#ef4444"),
    "risk_ml":             ("⚡", "Risk ML / SHAP",          "#f97316"),
    # Commercial
    "documents":           ("📄", "Documents",              "#3b82f6"),
    "cross_validation":    ("🔗", "Cross-Validation",       "#f59e0b"),
    "overall_score":       ("📊", "Score global",           "#10b981"),
    "risk_level":          ("🎯", "Niveau de risque",       "#ef4444"),
    "nli_results":         ("🧠", "NLI",                    "#6366f1"),
    "semantic_scores":     ("🌐", "Sémantique",             "#6366f1"),
    "explanations":        ("💡", "Explications XAI",       "#94a3b8"),
    "rule_results":        ("⚖️",  "Règles UCP",            "#f59e0b"),
    # LC Cross
    "final_score":         ("📊", "Score Final",            "#10b981"),
    "fuzzy_results":       ("🔍", "Entités (Flou)",         "#3b82f6"),
    "semantic_results":    ("🧠", "Sémantique 45A",         "#6366f1"),
    "condition_results":   ("📋", "Conditions 47A",         "#ec4899"),
    "score_breakdown":     ("📈", "Décomposition scores",   "#10b981"),
    "weights":             ("⚖️",  "Poids des couches",     "#94a3b8"),
    # Common
    "success":             ("✅", "Statut",                  "#22c55e"),
    "processing_time_s":   ("⏱️",  "Temps de traitement",   "#94a3b8"),
}


def show_json_explorer(data: dict, file_name: str = "result.json",
                       default_open_key: str | None = None):
    """
    Interactive JSON explorer:
      - Stats bar (sections, clés, profondeur, taille)
      - Toolbar : téléchargement + recherche + bascule vue
      - Mode "Par section" : expanders colorés par clé top-level
      - Mode "JSON complet" : st.json avec filtrage si recherche active
    """
    total_keys, max_depth = _jx_count_keys(data)
    json_str  = json.dumps(data, ensure_ascii=False, indent=2)
    json_size = len(json_str.encode("utf-8"))

    # Stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sections",     len(data))
    c2.metric("Clés totales", total_keys)
    c3.metric("Profondeur",   max_depth)
    c4.metric("Taille",       f"{json_size / 1024:.1f} Ko")
    c5.metric("Lignes",       json_str.count("\n"))

    st.divider()

    # Toolbar
    uid = file_name.replace(".", "_")   # unique key suffix per call
    col_dl, col_search, col_mode = st.columns([2, 4, 2])
    with col_dl:
        st.download_button(
            "⬇️ Télécharger JSON",
            data=json_str,
            file_name=file_name,
            mime="application/json",
            use_container_width=True,
            key=f"jx_dl_{uid}",
        )
    with col_search:
        search = st.text_input(
            "Filtre",
            placeholder="Clé ou valeur — ex: risk_level, USD…",
            key=f"jx_search_{uid}",
            label_visibility="collapsed",
        )
    with col_mode:
        view_mode = st.radio(
            "Vue",
            ["Par section", "JSON complet"],
            horizontal=True,
            key=f"jx_mode_{uid}",
            label_visibility="collapsed",
        )

    st.divider()

    # ── JSON complet ───────────────────────────────────────────────────
    if view_mode == "JSON complet":
        if search:
            filtered = {k: v for k, v in data.items()
                        if search.lower() in k.lower() or _jx_matches(v, search)}
            if filtered:
                st.caption(f"{len(filtered)} section(s) filtrée(s) sur {len(data)}")
                st.json(filtered)
            else:
                st.info(f"Aucune section ne correspond à « {search} »")
        else:
            st.json(data)
        return

    # ── Par section ────────────────────────────────────────────────────
    shown = 0
    for key, section_data in data.items():
        if search and not (search.lower() in key.lower() or _jx_matches(section_data, search)):
            continue
        shown += 1

        icon, label, color = _JX_META.get(
            key, ("📦", key.replace("_", " ").title(), "#64748b")
        )

        if isinstance(section_data, dict):
            n_info = f"{len(section_data)} clés"
        elif isinstance(section_data, list):
            n_info = f"{len(section_data)} éléments"
        elif isinstance(section_data, bool):
            n_info = "✅ true" if section_data else "❌ false"
        elif isinstance(section_data, (int, float)):
            n_info = str(round(section_data, 4) if isinstance(section_data, float) else section_data)
        else:
            n_info = f"str · {len(str(section_data))} car."

        auto_expand = (default_open_key and key == default_open_key) or (
            not default_open_key and shown == 1 and not search
        )

        with st.expander(
            f"{icon} **{label}** — `{key}` · *{n_info}*",
            expanded=auto_expand,
        ):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;'
                f'padding:8px 12px;background:{color}22;border-radius:8px;'
                f'border-left:3px solid {color}">'
                f'<span style="font-size:20px">{icon}</span>'
                f'<span style="font-weight:700;color:{color};font-size:14px">{label}</span>'
                f'<code style="margin-left:auto;font-size:11px;color:#94a3b8;'
                f'background:#1e293b;padding:2px 8px;border-radius:4px">{key}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if section_data is None or isinstance(section_data, (bool, int, float, str)):
                st.code(json.dumps(section_data, ensure_ascii=False), language="json")
            elif search and isinstance(section_data, dict):
                filtered_sec = {
                    k: v for k, v in section_data.items()
                    if search.lower() in str(k).lower() or _jx_matches(v, search)
                }
                if filtered_sec:
                    st.caption(f"Résultats filtrés : {len(filtered_sec)}/{len(section_data)} clés")
                    st.json(filtered_sec)
                    with st.expander("Voir la section complète", expanded=False):
                        st.json(section_data)
                else:
                    st.json(section_data)
            else:
                st.json(section_data)

    if shown == 0 and search:
        st.info(f"Aucune section ne correspond à « {search} »")


# ── Payload extractors ────────────────────────────────────────────────

def _extract_lc_payload(pipeline_result: dict) -> dict:
    """Build LC dict from /pipeline result (swift.entities)."""
    swift = pipeline_result.get("swift") or {}
    ent   = swift.get("entities") or {}
    nlp   = swift.get("nlp") or {}
    sem   = nlp.get("semantic") or {}
    lc_cr = pipeline_result.get("ucp_compliance", {}) or {}

    return {
        "applicant":                ent.get("applicant") or ent.get("buyer"),
        "beneficiary":              ent.get("beneficiary") or ent.get("seller"),
        "currency":                 ent.get("currency"),
        "amount":                   ent.get("lc_amount") or ent.get("amount"),
        "tolerance_plus":           ent.get("tolerance_plus") or ent.get("tolerance"),
        "tolerance_minus":          ent.get("tolerance_minus") or ent.get("tolerance"),
        "expiry_date":              ent.get("expiry_date"),
        "shipment_date":            ent.get("shipment_date") or ent.get("latest_shipment_date"),
        "presentation_period_days": ent.get("presentation_period") or ent.get("presentation_period_days"),
        "incoterm":                 ent.get("incoterm"),
        "partial_shipments_allowed":ent.get("partial_shipments_allowed") or ent.get("partial_shipments"),
        "transhipment_allowed":     ent.get("transhipment_allowed") or ent.get("transhipment"),
        "goods_description":        ent.get("goods_description") or sem.get("goods_description"),
        "documents_required":       ent.get("documents_required"),
        "additional_conditions":    ent.get("additional_conditions"),
    }


def _extract_commercial_payload(commercial_result: dict) -> dict:
    """Build commercial dict (bl/invoice/packing_list) from /commercial/validate result."""
    docs   = commercial_result.get("documents") or []
    result = {}
    for doc in docs:
        dt = doc.get("doc_type") or ""
        fields = doc.get("fields") or {}
        tables = doc.get("tables") or {}
        if dt in ("bl", "invoice", "packing_list"):
            result[dt] = fields
            tbl_key = {"bl": "bl_tables", "invoice": "invoice_tables",
                       "packing_list": "pl_tables"}[dt]
            first_tbl = next(
                (v for v in tables.values() if isinstance(v, list) and v), []
            )
            result[tbl_key] = first_tbl
    return result


# ══════════════════════════════════════════════════════════════════════
# ① MT700 DISPLAY
# ══════════════════════════════════════════════════════════════════════

def show_mt700_result(data: dict):
    swift  = data.get("swift") or {}
    ent    = swift.get("entities") or {}
    nlp    = swift.get("nlp") or {}
    val    = swift.get("validation") or {}
    ucp    = data.get("ucp_compliance") or {}
    ext    = data.get("external_validation") or {}
    rml    = data.get("risk_ml") or {}
    ds     = data.get("ds_analysis") or {}

    sw_type = swift.get("swift_type") or ent.get("swift_type") or "MT700"
    nlp_sc  = nlp.get("score", 0)
    val_sc  = val.get("score", 0)
    ucp_sc  = ucp.get("global_score", 0)
    risk_lv = ds.get("risk_level") or rml.get("risk_label") or "UNKNOWN"
    ocr_c   = round((data.get("ocr") or {}).get("confidence", 1) * 100)

    c = _risk_colors(risk_lv)
    st.markdown(f"""
    <div class="score-banner" style="background:{c['bg']}22;border-color:{c['fg']}">
      <div>
        <div style="font-size:13px;color:{c['fg']};font-weight:700">{_risk_icon(risk_lv)} Risque MT700 : {risk_lv}</div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">Type {sw_type} — OCR {ocr_c}%</div>
      </div>
      <div class="score-val" style="color:{_score_color_int(nlp_sc)}">{nlp_sc}<span style="font-size:14px">/100</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Type SWIFT</div><div class="kpi-val">{sw_type}</div></div>
      <div class="kpi-card"><div class="kpi-label">Score NLP</div><div class="kpi-val" style="color:{_score_color_int(nlp_sc)}">{nlp_sc}</div><div class="kpi-sub">/100</div></div>
      <div class="kpi-card"><div class="kpi-label">Validation</div><div class="kpi-val" style="color:{_score_color_int(val_sc)}">{val_sc}</div><div class="kpi-sub">/100</div></div>
      <div class="kpi-card"><div class="kpi-label">UCP 600</div><div class="kpi-val" style="color:{_score_color_int(ucp_sc)}">{ucp_sc}</div><div class="kpi-sub">/100</div></div>
      <div class="kpi-card"><div class="kpi-label">OCR</div><div class="kpi-val">{ocr_c}%</div></div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📋 Entités", "⚖️ UCP 600", "✅ Validation", "🔬 Analyse DS", "🤖 Risk ML", "🌐 Externe"
    ])

    with t1:
        _show_mt700_entities(ent, nlp)
    with t2:
        _show_mt700_ucp(ucp)
    with t3:
        _show_mt700_validation(val)
    with t4:
        _show_mt700_ds(ds, nlp)
    with t5:
        _show_mt700_risk_ml(rml)
    with t6:
        _show_mt700_external(ext)


def _show_mt700_entities(ent: dict, nlp: dict):
    st.subheader("Entités extraites")
    KEY_FIELDS = [
        ("Applicant (acheteur)",   "applicant"),
        ("Beneficiary (vendeur)",  "beneficiary"),
        ("Banque émettrice",       "issuing_bank"),
        ("N° LC",                  "lc_number"),
        ("Devise",                 "currency"),
        ("Montant",                "lc_amount"),
        ("Tolérance",              "tolerance"),
        ("Date émission",          "issue_date"),
        ("Date expiry",            "expiry_date"),
        ("Date expédition",        "shipment_date"),
        ("Incoterm",               "incoterm"),
        ("Port chargement",        "port_of_loading"),
        ("Port déchargement",      "port_of_discharge"),
        ("Paiement partiel",       "partial_shipments_allowed"),
        ("Transbordement",         "transhipment_allowed"),
        ("Présentation (jours)",   "presentation_period"),
    ]
    rows = [(lbl, str(ent.get(k) or "—")) for lbl, k in KEY_FIELDS if ent.get(k)]
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["Champ", "Valeur"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    long_keys = ["goods_description", "documents_required", "additional_conditions"]
    long_labels = {"goods_description": "45A — Description marchandises",
                   "documents_required": "46A — Documents requis",
                   "additional_conditions": "47A — Conditions additionnelles"}
    for k in long_keys:
        v = ent.get(k) or ""
        if v:
            st.text_area(long_labels[k], v, height=100, disabled=True)

    # NER entities
    ner = (nlp.get("ner") or {}).get("entities") or []
    if ner:
        with st.expander(f"🔷 Entités NER ({len(ner)})", expanded=False):
            for e in ner:
                col1, col2, col3 = st.columns([2, 3, 1])
                col1.markdown(f"`{e.get('label','')}`")
                col2.markdown(f"**{e.get('text','')[:80]}**")
                col3.markdown(f"{round(e.get('confidence',0)*100)}%")


def _show_mt700_ucp(ucp: dict):
    score = ucp.get("global_score", 0)
    st.metric("Score UCP 600 global", f"{score}/100", delta=None)

    cats = ucp.get("category_scores") or {}
    if cats:
        import pandas as pd
        df = pd.DataFrame(
            [(k, v) for k, v in cats.items()],
            columns=["Catégorie", "Score"]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

    findings = ucp.get("findings") or []
    errors   = [f for f in findings if f.get("type") == "error"]
    warnings = [f for f in findings if f.get("type") == "warning"]
    infos    = [f for f in findings if f.get("type") == "info"]

    if errors:
        with st.expander(f"❌ Erreurs UCP ({len(errors)})", expanded=True):
            for f in errors:
                st.markdown(
                    f'<div class="ucp-finding" style="border-color:#ef4444">'
                    f'<strong>{f.get("article","")}</strong> — {f.get("message","")}'
                    f'</div>', unsafe_allow_html=True)
    if warnings:
        with st.expander(f"⚠️ Avertissements UCP ({len(warnings)})"):
            for f in warnings:
                st.markdown(
                    f'<div class="ucp-finding" style="border-color:#f59e0b">'
                    f'<strong>{f.get("article","")}</strong> — {f.get("message","")}'
                    f'</div>', unsafe_allow_html=True)
    if infos:
        with st.expander(f"ℹ️ Infos UCP ({len(infos)})"):
            for f in infos:
                st.markdown(
                    f'<div class="ucp-finding" style="border-color:#6366f1">'
                    f'{f.get("message","")}</div>', unsafe_allow_html=True)

    if not findings:
        st.success("✅ Aucune non-conformité UCP détectée")


def _show_mt700_validation(val: dict):
    errors   = val.get("field_errors") or val.get("errors") or []
    warnings = val.get("warnings") or []

    c1, c2 = st.columns(2)
    c1.metric("Erreurs champs", len(errors))
    c2.metric("Avertissements", len(warnings))

    if errors:
        with st.expander(f"❌ Erreurs de champs ({len(errors)})", expanded=True):
            for e in errors:
                msg = e.get("message") or e.get("error") or str(e)
                fld = e.get("field") or ""
                st.error(f"**{fld}** — {msg}" if fld else msg)

    if warnings:
        with st.expander(f"⚠️ Avertissements ({len(warnings)})"):
            for w in warnings:
                msg = w.get("message") or w.get("warning") or str(w)
                st.warning(msg)

    if not errors and not warnings:
        st.success("✅ Aucune erreur de validation SWIFT détectée")


def _show_mt700_ds(ds: dict, nlp: dict):
    risk_lv = ds.get("risk_level") or "UNKNOWN"
    risk_sc = ds.get("risk_score") or 0
    c = _risk_colors(risk_lv)

    st.markdown(
        f'<div class="score-banner" style="background:{c["bg"]}22;border-color:{c["fg"]}">'
        f'<div><div class="kpi-label">Niveau de risque DS</div>'
        f'<div style="font-size:20px;font-weight:700;color:{c["fg"]}">'
        f'{_risk_icon(risk_lv)} {risk_lv}</div></div>'
        f'<div class="score-val" style="color:{c["fg"]}">{risk_sc}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    factors = ds.get("risk_factors") or []
    if factors:
        with st.expander("📊 Facteurs de risque"):
            for f in factors:
                sev = f.get("severity") or "medium"
                st.markdown(
                    f'<div class="rule-card" style="border-color:{_status_color(sev)};'
                    f'background:{_status_bg(sev)}">'
                    f'<span class="badge {_sev_class(sev)}">{sev}</span> '
                    f'{f.get("description") or f.get("message") or str(f)}'
                    f'</div>', unsafe_allow_html=True)

    sem = nlp.get("semantic") or {}
    risk_flags = sem.get("risk_flags") or []
    if risk_flags:
        with st.expander(f"🚩 Risk flags NLP ({len(risk_flags)})"):
            for rf in risk_flags:
                sev = rf.get("severity") or "medium"
                st.markdown(
                    f'<div class="rule-card" style="border-color:{_status_color(sev)};'
                    f'background:{_status_bg(sev)}">'
                    f'<span class="badge {_sev_class(sev)}">{sev}</span> '
                    f'{rf.get("message") or rf.get("description") or str(rf)}'
                    f'</div>', unsafe_allow_html=True)


def _show_mt700_risk_ml(rml: dict):
    if not rml or rml.get("error"):
        st.info("ℹ️ Risk ML non disponible (modèles non entraînés)")
        return

    label  = rml.get("risk_label", "?")
    score  = rml.get("risk_score", 0)
    proba  = rml.get("probabilities") or {}
    anomaly= rml.get("anomaly") or False
    seg    = rml.get("cluster_label") or "?"
    shap   = rml.get("shap_top_features") or []

    c = _risk_colors(label)
    st.markdown(
        f'<div class="gauge-score">{score}</div>'
        f'<div style="text-align:center;margin-bottom:12px">'
        f'<span class="badge badge-{label.lower()}" style="background:{c["bg"]};color:{c["fg"]};font-size:14px;padding:6px 20px">'
        f'{_risk_icon(label)} {label}</span></div>',
        unsafe_allow_html=True
    )

    st.caption("Probabilités par classe")
    for cls, p in proba.items():
        pval = round(float(p) * 100)
        col = _score_color_int(100 - pval if cls == "HIGH" else pval)
        st.markdown(
            f'<div class="bdown-wrap">'
            f'<div class="bdown-header"><span>{cls}</span><span style="color:{col}">{pval}%</span></div>'
            f'<div class="bdown-bar-bg"><div class="bdown-bar-fill" style="width:{pval}%;background:{col}"></div></div>'
            f'</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Anomalie", "🚨 Oui" if anomaly else "✅ Non")
    c2.metric("Segment", seg)

    if shap:
        st.caption("SHAP — Top features")
        for ft in shap[:6]:
            name = ft.get("feature", "")
            val  = ft.get("shap_value", 0)
            direction = ft.get("direction") or ("positive" if val > 0 else "negative")
            bar_color = "#ef4444" if direction == "positive" else "#10b981"
            bar_width = min(abs(round(float(val) * 200)), 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                f'<span style="font-size:11px;color:#94a3b8;width:180px;flex-shrink:0">{name}</span>'
                f'<div style="flex:1;height:8px;background:#334155;border-radius:4px;overflow:hidden">'
                f'<div style="width:{bar_width}%;height:8px;background:{bar_color};border-radius:4px"></div></div>'
                f'<span style="font-size:11px;color:#64748b">{val:+.3f}</span>'
                f'</div>', unsafe_allow_html=True)


def _show_mt700_external(ext: dict):
    if not ext:
        st.info("ℹ️ Validation externe non disponible")
        return

    risk_lv = ext.get("risk_level", "CLEAN")
    c = _risk_colors(risk_lv)
    st.markdown(
        f'<div class="score-banner" style="background:{c["bg"]}22;border-color:{c["fg"]}">'
        f'<div style="color:{c["fg"]};font-weight:700">{_risk_icon(risk_lv)} Risque externe : {risk_lv}</div>'
        f'</div>', unsafe_allow_html=True)

    for section in ["ofac", "eu_sanctions", "un_sanctions", "gleif", "opencorporates"]:
        sec = ext.get(section) or {}
        if sec:
            status = sec.get("status") or sec.get("risk_level") or "OK"
            with st.expander(f"🌐 {section.upper()} — {status}"):
                st.json(sec)


# ══════════════════════════════════════════════════════════════════════
# ② COMMERCIAL DOCS DISPLAY
# ══════════════════════════════════════════════════════════════════════

def show_commercial_result(data: dict):
    summary     = data.get("summary") or {}
    cross_report= data.get("cross_report") or {}
    docs        = data.get("documents") or []

    n_doc  = summary.get("total_documents", len(docs))
    n_proc = summary.get("processed", 0)
    n_fail = summary.get("failed", 0)
    proc_t = summary.get("processing_time_s") or data.get("processing_time_s") or 0
    risk_lv= (cross_report.get("risk_level") or "UNKNOWN").upper()
    fin_sc = cross_report.get("final_score")

    c = _risk_colors(risk_lv)
    st.markdown(f"""
    <div class="score-banner" style="background:{c['bg']}22;border-color:{c['fg']}">
      <div>
        <div style="font-size:13px;color:{c['fg']};font-weight:700">
          {_risk_icon(risk_lv)} Risque inter-documents : {risk_lv}
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">
          {n_doc} document(s) · {n_proc} traité(s) · {round(float(proc_t), 1)}s
        </div>
      </div>
      <div class="score-val" style="color:{_score_color(fin_sc or 0)}">{_pct(fin_sc)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Score breakdown
    bd = cross_report.get("score_breakdown") or {}
    if bd:
        st.caption("Décomposition du score inter-documents")
        cols = st.columns(len(bd))
        COMM_LAYERS = {
            "rules": ("Règles", "#f59e0b"), "nli": ("NLI", "#ec4899"),
            "semantic": ("Sémantique", "#6366f1"), "table": ("Tables", "#10b981"),
            "party": ("Parties", "#3b82f6"),
        }
        for i, (k, v) in enumerate(bd.items()):
            lbl, col = COMM_LAYERS.get(k, (k, "#6b7280"))
            vf = float(v)
            with cols[i]:
                st.markdown(
                    f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
                    f'<div class="kpi-val" style="color:{col}">{_pct(vf)}</div></div>',
                    unsafe_allow_html=True)

    st.divider()
    t_docs, t_cross, t_nli, t_sem, t_xai = st.tabs([
        "📄 Documents", "📏 Règles croisées", "🧠 NLI", "📊 Sémantique", "💡 XAI"
    ])

    with t_docs:
        for i, doc in enumerate(docs):
            _show_comm_doc(doc, i)

    with t_cross:
        _show_comm_rules(cross_report.get("rule_results") or [])

    with t_nli:
        _show_comm_nli(cross_report.get("nli_results") or [])

    with t_sem:
        _show_comm_semantic(cross_report.get("semantic_scores") or {})

    with t_xai:
        _show_comm_xai(cross_report.get("explanations") or [])


def _show_comm_doc(doc: dict, idx: int):
    dt    = doc.get("doc_type") or "unknown"
    status= doc.get("status") or "success"
    _conf = doc.get("confidence") or 0
    conf  = float(_conf) if not isinstance(_conf, dict) else 0.0
    _extr = doc.get("extraction_score") or 0
    extr  = _extr.get("global_score", 0) if isinstance(_extr, dict) else float(_extr or 0)
    fields= doc.get("fields") or {}
    n_fld = doc.get("n_fields") or len(fields)
    src   = doc.get("source_filename") or doc.get("sourceFilename") or doc.get("source") or f"Doc {idx+1}"
    tables= doc.get("tables") or {}
    rules = doc.get("rule_results") or []
    rs    = doc.get("rule_summary") or {}

    bg_c = _doc_color(dt)
    fg_c = _doc_fg(dt)

    with st.expander(
        f"{'✅' if status == 'success' else '❌'} [{_doc_label(dt)}] {src}  "
        f"— Conf: {round(float(conf)*100)}%",
        expanded=(idx == 0)
    ):
        st.markdown(
            f'<span class="doc-type-badge" style="background:{bg_c};color:{fg_c}">'
            f'{_doc_label(dt)}</span>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Confiance",       f"{round(float(conf)*100)}%")
        c2.metric("Score extraction",f"{round(float(extr)*100)}%")
        c3.metric("Champs extraits", n_fld)
        c4.metric("Règles PASS",     rs.get("pass", 0))

        if fields:
            tab_f, tab_t, tab_r = st.tabs(["Champs", "Tables", f"Règles ({len(rules)})"])
            with tab_f:
                if fields:
                    import pandas as pd
                    df = pd.DataFrame(list(fields.items()), columns=["Champ","Valeur"])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            with tab_t:
                if tables:
                    for tname, rows in tables.items():
                        if isinstance(rows, list) and rows:
                            st.caption(tname)
                            import pandas as pd
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Aucune table extraite")
            with tab_r:
                _show_comm_rules(rules)


def _show_comm_rules(rules: list):
    if not rules:
        st.info("Aucune règle disponible")
        return

    pass_n  = sum(1 for r in rules if r.get("status") == "pass")
    fail_n  = sum(1 for r in rules if r.get("status") == "fail")
    warn_n  = sum(1 for r in rules if r.get("status") == "warning")
    skip_n  = sum(1 for r in rules if r.get("status") == "skipped")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ PASS",    pass_n)
    c2.metric("❌ FAIL",    fail_n)
    c3.metric("⚠️ WARN",   warn_n)
    c4.metric("⏭️ SKIP",   skip_n)

    filter_opt = st.selectbox("Filtre statut", ["Tous", "FAIL", "WARNING", "PASS", "SKIPPED"],
                               key=f"rule_filter_{id(rules)}")
    shown = [r for r in rules
             if filter_opt == "Tous" or r.get("status","").upper() == filter_opt]

    for r in shown:
        s   = r.get("status", "skipped")
        sev = r.get("severity", "info")
        msg = r.get("message", "")
        rule= r.get("rule", "")
        pair= r.get("doc_pair") or r.get("doc_type") or ""

        st.markdown(
            f'<div class="rule-card" style="border-color:{_status_color(s)};background:{_status_bg(s)}">'
            f'<div class="rule-header">'
            f'{_status_icon(s)} <span class="rule-name">{rule}</span>'
            f'<span class="badge {_sev_class(sev)}">{sev}</span>'
            + (f'<span class="badge badge-doc">{pair}</span>' if pair else '') +
            f'</div><div class="rule-msg">{msg}</div>'
            f'</div>', unsafe_allow_html=True)


def _show_comm_nli(nli_results: list):
    if not nli_results:
        st.info("Aucun résultat NLI disponible")
        return

    ent_n  = sum(1 for r in nli_results if r.get("nli_label") == "ENTAILMENT")
    cont_n = sum(1 for r in nli_results if r.get("nli_label") == "CONTRADICTION")
    neut_n = sum(1 for r in nli_results if r.get("nli_label") == "NEUTRAL")
    arb_n  = sum(1 for r in nli_results if r.get("arbitrated"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ ENTAILMENT",    ent_n)
    c2.metric("❌ CONTRADICTION", cont_n)
    c3.metric("➖ NEUTRAL",       neut_n)
    c4.metric("🔧 Arbitrés",     arb_n)

    for r in nli_results:
        lbl   = r.get("nli_label", "NEUTRAL")
        score = round(float(r.get("score", 0) or 0) * 100)
        cat   = r.get("category", "")
        desc  = r.get("description", "")
        arb   = r.get("arbitrated", False)

        nli_color = {"ENTAILMENT": "#064e3b", "CONTRADICTION": "#7f1d1d"}.get(lbl, "#1e293b")
        nli_fg    = {"ENTAILMENT": "#6ee7b7", "CONTRADICTION": "#fca5a5"}.get(lbl, "#94a3b8")
        nli_icon  = {"ENTAILMENT": "✅", "CONTRADICTION": "❌"}.get(lbl, "➖")

        st.markdown(
            f'<div class="nli-card" style="background:{nli_color}22;border:1px solid {nli_fg}33">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
            f'<span class="nli-label" style="color:{nli_fg}">{nli_icon} {lbl}</span>'
            f'<span class="badge badge-doc">{cat}</span>'
            + (f'<span class="badge" style="background:#1a1a4a;color:#a5b4fc">🔧 arbitré</span>' if arb else '') +
            f'<span style="font-size:12px;color:{nli_fg};font-weight:700">{score}%</span>'
            f'</div>'
            f'<div class="nli-desc">{desc}</div>'
            f'</div>', unsafe_allow_html=True)


def _show_comm_semantic(scores: dict):
    if not scores:
        st.info("Aucun score sémantique disponible")
        return

    for k, v in scores.items():
        if not isinstance(v, (int, float)):
            continue
        vf = float(v)
        col = _score_color(vf)
        st.markdown(
            f'<div class="bdown-wrap">'
            f'<div class="bdown-header"><span>{k}</span><span style="color:{col}">{_pct(vf)}</span></div>'
            f'<div class="bdown-bar-bg"><div class="bdown-bar-fill" style="width:{round(vf*100)}%;background:{col}"></div></div>'
            f'</div>', unsafe_allow_html=True)


def _show_comm_xai(explanations: list):
    if not explanations:
        st.info("Aucune explication disponible")
        return

    neg  = [e for e in explanations if e.get("impact") == "negative"]
    warn = [e for e in explanations if e.get("impact") == "warning"]
    pos  = [e for e in explanations if e.get("impact") == "positive"]

    for group, title, title_color in [
        (neg,  f"❌ Problèmes ({len(neg)})",       "#ef4444"),
        (warn, f"⚠️ Avertissements ({len(warn)})", "#f59e0b"),
        (pos,  f"✅ Points conformes ({len(pos)})", "#10b981"),
    ]:
        if not group:
            continue
        st.markdown(f'<div style="font-size:13px;font-weight:600;color:{title_color};margin:8px 0 4px">{title}</div>',
                    unsafe_allow_html=True)
        for e in group:
            imp = e.get("impact", "neutral")
            sev = e.get("severity", "info")
            msg = e.get("message", "")
            cat = e.get("category", "")
            st.markdown(
                f'<div class="exp-card" style="border-color:{_impact_color(imp)}">'
                f'<span>{_impact_icon(imp)}</span>'
                f'<div><div class="exp-msg">{msg}</div>'
                f'<div class="exp-badges">'
                f'<span class="badge {_sev_class(sev)}">{sev}</span>'
                f'<span class="badge badge-doc">{cat}</span>'
                f'</div></div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ③ LC ↔ COMMERCIAL CROSS-VALIDATION DISPLAY
# ══════════════════════════════════════════════════════════════════════

def show_lc_cross_result(data: dict):
    fin_sc = data.get("final_score") or 0
    risk   = (data.get("risk_level") or "unknown").upper()
    bd     = data.get("score_breakdown") or {}
    lc_sum = data.get("lc_summary") or {}
    cm_inf = data.get("commercial_info") or {}

    decision = (data.get("decision") or "").upper()
    decision_color = {"ACCEPT": "#10b981", "DISCREPANT": "#f59e0b", "REJECT": "#ef4444"}.get(decision, "#94a3b8")
    decision_icon  = {"ACCEPT": "✅", "DISCREPANT": "⚠️", "REJECT": "❌"}.get(decision, "•")

    c = _risk_colors(risk)
    st.markdown(f"""
    <div class="score-banner" style="background:{c['bg']}22;border-color:{c['fg']}">
      <div>
        <div style="font-size:13px;color:{c['fg']};font-weight:700">
          {_risk_icon(risk)} Cohérence LC ↔ Documents : {risk}
        </div>
        <div style="font-size:11px;color:#94a3b8;margin-top:2px">
          5 catégories · Financier · Marchandises · Transport · Parties · Documentaire
        </div>
        {f'<div style="font-size:12px;font-weight:700;color:{decision_color};margin-top:4px">{decision_icon} Décision : {decision}</div>' if decision else ''}
      </div>
      <div class="score-val" style="color:{_score_color(float(fin_sc))}">{_pct(float(fin_sc))}</div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs LC summary
    lc_infos = [
        ("Bénéficiaire",     lc_sum.get("beneficiary")),
        ("Donneur d'ordre",  lc_sum.get("applicant")),
        ("Devise",           lc_sum.get("currency")),
        ("Montant LC",       (f"{lc_sum.get('amount'):,.2f}" if lc_sum.get("amount") else None)),
        ("Tolérance",        (f"+{lc_sum.get('tolerancePlus')}% / -{lc_sum.get('toleranceMinus')}%"
                              if lc_sum.get("tolerancePlus") is not None else None)),
        ("Expiry",           lc_sum.get("expiryDate")),
        ("Latest shipment",  lc_sum.get("shipmentDate")),
        ("Incoterm",         lc_sum.get("incoterm")),
    ]
    items_html = "".join(
        f'<div class="kpi-card" style="min-width:140px"><div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-val" style="font-size:13px">{val}</div></div>'
        for lbl, val in lc_infos if val
    )
    st.markdown(f'<div class="kpi-grid">{items_html}</div>', unsafe_allow_html=True)

    # Score breakdown bars
    if bd:
        st.caption("Score par catégorie métier")
        _cat_weights = {
            "financial": 35, "goods": 25, "documentary": 20, "shipment": 10, "parties": 10,
            # legacy fallback
            "rules": 40, "fuzzy": 20, "semantic": 25, "conditions": 15,
        }
        for k, v in bd.items():
            vf  = float(v)
            col = _layer_color(k)
            wt  = _cat_weights.get(k, 0)
            cat_status = (data.get("categories") or {}).get(k, {}).get("status", "")
            status_icon = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(cat_status, "")
            st.markdown(
                f'<div class="bdown-wrap">'
                f'<div class="bdown-header">'
                f'<span>{status_icon} {_layer_label(k)}</span>'
                f'<span style="color:#475569">poids {wt}%</span>'
                f'<span style="color:{col};font-weight:700">{_pct(vf)}</span></div>'
                f'<div class="bdown-bar-bg">'
                f'<div class="bdown-bar-fill" style="width:{round(vf*100)}%;background:{col}"></div>'
                f'</div></div>', unsafe_allow_html=True)

    st.divider()
    tabs = st.tabs([
        "📋 Vue d'ensemble", "🏷️ Catégories", "🔗 Inter-Documents",
        "📏 Règles", "👥 Entités", "📝 Sémantique 45A",
        "⚙️ Conditions 47A", "💡 Explications",
    ])

    with tabs[0]:
        _lc_cross_overview(data)
    with tabs[1]:
        _lc_cross_categories(data.get("categories") or {})
    with tabs[2]:
        _lc_cross_cross_doc(data.get("cross_document") or {})
    with tabs[3]:
        _lc_cross_rules(data.get("rule_results") or [])
    with tabs[4]:
        _lc_cross_fuzzy(data.get("fuzzy_results") or [])
    with tabs[5]:
        _lc_cross_semantic(data.get("semantic_results") or {})
    with tabs[6]:
        _lc_cross_conditions(data.get("condition_results") or [])
    with tabs[7]:
        _lc_cross_explanations(data.get("explanations") or [])


def _lc_cross_categories(categories: dict):
    """Vue hiérarchique par catégorie métier (v3.0)."""
    if not categories:
        st.info("Aucune donnée de catégorie disponible"); return

    _cat_meta = {
        "financial":   ("💰", "Financier",    0.35, "Montant, monnaie, tolérance LC, arithmétique facture"),
        "goods":       ("📦", "Marchandises", 0.25, "Cohérence 45A vs Invoice / BL (enrichi depuis tables)"),
        "shipment":    ("🚢", "Transport",    0.10, "Dates, ports, transbordement, poids, CBM"),
        "parties":     ("👥", "Parties",      0.10, "Bénéficiaire, donneur d'ordre, consignataire"),
        "documentary": ("📋", "Documentaire", 0.20, "Complétude documentaire, incoterm, conditions 47A"),
    }

    for cat_key, cat_data in categories.items():
        icon, label, wt, desc = _cat_meta.get(cat_key, ("•", cat_key, 0, ""))
        score  = float(cat_data.get("score", 0))
        status = cat_data.get("status", "")
        primary = cat_data.get("primary_issue")
        derived = cat_data.get("derived_issues") or []
        checks  = cat_data.get("checks") or []

        s_color = {"pass": "#10b981", "partial": "#f59e0b", "fail": "#ef4444"}.get(status, "#94a3b8")
        s_icon  = {"pass": "✅", "partial": "⚠️", "fail": "❌"}.get(status, "•")

        primary_html = (
            '<div style="font-size:11px;color:#f87171;margin-top:2px">⚡ '
            + primary.replace("_", " ") + "</div>"
        ) if primary else ""
        header_html = (
            f'<div style="display:flex;align-items:center;gap:10px;padding:10px 14px;'
            f'background:{s_color}15;border-left:4px solid {s_color};border-radius:6px;margin-bottom:6px">'
            f'<div style="font-size:22px">{icon}</div>'
            f'<div style="flex:1">'
            f'<div style="font-size:13px;font-weight:700;color:#e2e8f0">{label}'
            f'  <span style="font-size:11px;color:#94a3b8;font-weight:400">poids {int(wt*100)}%</span></div>'
            f'<div style="font-size:11px;color:#94a3b8">{desc}</div>'
            f'{primary_html}'
            f'</div>'
            f'<div style="text-align:right">'
            f'<div style="font-size:18px;font-weight:700;color:{s_color}">{_pct(score)}</div>'
            f'<div style="font-size:11px;color:{s_color}">{s_icon} {status.upper()}</div>'
            f'</div></div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        if derived:
            st.caption(f"Problèmes secondaires : {', '.join(d.replace('_', ' ') for d in derived)}")

        if checks:
            with st.expander(f"Détails ({len(checks)} vérifications)", expanded=(status == "fail")):
                for chk in checks:
                    s   = chk.get("status", "skipped")
                    sev = chk.get("severity", "info")
                    msg = chk.get("message", "")
                    det = chk.get("detail", "")
                    lcv = chk.get("lc_value", "")
                    docv= chk.get("doc_value", "")
                    dt  = chk.get("doc_type", "")
                    html = (
                        f'<div class="rule-card" style="border-color:{_status_color(s)};background:{_status_bg(s)}">'
                        f'<div class="rule-header">'
                        f'{_status_icon(s)} <span class="rule-name">{chk.get("rule","")}</span>'
                        f'<span class="badge {_sev_class(sev)}">{sev}</span>'
                        + (f'<span class="badge badge-doc">{dt}</span>' if dt else '') +
                        f'</div><div class="rule-msg">{msg}</div>'
                    )
                    if det:
                        html += f'<div class="rule-vals">{det}</div>'
                    if lcv or docv:
                        html += '<div class="rule-vals">'
                        if lcv:
                            html += f'<span style="background:#1a2e1a;color:#86efac;padding:1px 6px;border-radius:4px;font-size:11px">LC : {lcv}</span> '
                        if docv:
                            html += f'<span style="background:#1e293b;color:#93c5fd;padding:1px 6px;border-radius:4px;font-size:11px">Doc : {docv}</span>'
                        html += '</div>'
                    html += '</div>'
                    st.markdown(html, unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def _lc_cross_cross_doc(cross_doc: dict):
    """Vue réconciliation inter-documents INV↔BL, INV↔PL, BL↔PL (sans MT700)."""
    if not cross_doc:
        st.info("Données de réconciliation inter-documents non disponibles"); return

    overall = float(cross_doc.get("overall_score", 0))
    st.metric("Score global réconciliation", _pct(overall))
    st.caption("Cohérence entre les 3 documents commerciaux avant validation MT700.")

    _pair_meta = {
        "inv_bl": ("📄🚢", "Invoice ↔ Bill of Lading"),
        "inv_pl": ("📄📦", "Invoice ↔ Packing List"),
        "bl_pl":  ("🚢📦", "Bill of Lading ↔ Packing List"),
    }
    _check_labels = {
        "seller_vs_shipper":    "Vendeur ↔ Chargeur",
        "seller_vs_exporter":   "Vendeur ↔ Exportateur",
        "port_of_loading":      "Port de chargement",
        "port_of_discharge":    "Port de déchargement",
        "vessel_voyage":        "Navire / Voyage",
        "goods_similarity":     "Similarité marchandises",
        "currency_match":       "Monnaie",
        "incoterm_match":       "Incoterm",
        "weight_consistency":   "Cohérence poids brut",
        "cbm_consistency":      "Cohérence CBM",
    }

    for pair_key, (pair_icon, pair_label) in _pair_meta.items():
        grp = cross_doc.get(pair_key) or {}
        if not grp:
            continue

        n_pass = sum(1 for v in grp.values() if isinstance(v, dict) and v.get("status") == "pass")
        n_fail = sum(1 for v in grp.values() if isinstance(v, dict) and v.get("status") in ("fail", "warning"))
        n_skip = sum(1 for v in grp.values() if isinstance(v, dict) and v.get("status") == "skipped")

        col_hdr = "#10b981" if n_fail == 0 else "#f59e0b" if n_fail <= 1 else "#ef4444"
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{col_hdr};margin:12px 0 6px">'
            f'{pair_icon} {pair_label} — ✅ {n_pass} / ⚠️ {n_fail} / ⏭️ {n_skip}</div>',
            unsafe_allow_html=True)

        for chk_key, chk_val in grp.items():
            if not isinstance(chk_val, dict):
                continue
            status = chk_val.get("status", "skipped")
            lbl    = _check_labels.get(chk_key, chk_key.replace("_", " ").title())
            score  = chk_val.get("score")
            va     = chk_val.get("value_a") or chk_val.get("inv") or chk_val.get("message", "")
            vb     = chk_val.get("value_b") or chk_val.get("pl") or ""

            s_col  = _status_color(status)
            s_ico  = _status_icon(status)
            score_badge = f'<span style="color:{_score_color(float(score))};font-weight:700">{round(float(score)*100)}%</span>' if score is not None else ""

            html = (
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;'
                f'border-left:3px solid {s_col};background:{s_col}11;border-radius:4px;margin-bottom:4px">'
                f'<span>{s_ico}</span>'
                f'<span style="font-size:12px;font-weight:600;color:#cbd5e1;flex:1">{lbl}</span>'
                f'{score_badge}'
            )
            if va:
                html += f'<span style="font-size:11px;color:#94a3b8">{str(va)[:60]}'
                if vb:
                    html += f' → {str(vb)[:60]}'
                html += '</span>'
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


def _lc_cross_overview(data: dict):
    exps = data.get("explanations") or []
    neg  = [e for e in exps if e.get("impact") == "negative"]
    warn = [e for e in exps if e.get("impact") == "warning"]
    pos  = [e for e in exps if e.get("impact") == "positive"]

    c1, c2, c3 = st.columns(3)
    c1.metric("❌ Problèmes",       len(neg))
    c2.metric("⚠️ Avertissements",  len(warn))
    c3.metric("✅ Points conformes", len(pos))

    for group, title, col in [
        (neg,  f"❌ Problèmes détectés ({len(neg)})",      "#ef4444"),
        (warn, f"⚠️ Avertissements ({len(warn)})",         "#f59e0b"),
        (pos,  f"✅ Points conformes ({len(pos)})",        "#10b981"),
    ]:
        if not group:
            continue
        st.markdown(
            f'<div style="font-size:13px;font-weight:600;color:{col};margin:10px 0 4px">{title}</div>',
            unsafe_allow_html=True)
        for e in group:
            imp = e.get("impact", "neutral")
            sev = e.get("severity", "info")
            tp  = e.get("type", "rule")
            msg = e.get("message", "")
            st.markdown(
                f'<div class="exp-card" style="border-color:{_impact_color(imp)}">'
                f'<span>{_impact_icon(imp)}</span>'
                f'<div><div class="exp-msg">{msg}</div>'
                f'<div class="exp-badges">'
                f'<span class="badge {_sev_class(sev)}">{sev}</span>'
                f'<span class="badge badge-doc">{tp}</span>'
                f'</div></div></div>', unsafe_allow_html=True)


def _lc_cross_rules(rules: list):
    if not rules:
        st.info("Aucune règle disponible"); return

    pass_n = sum(1 for r in rules if r.get("status") == "pass")
    fail_n = sum(1 for r in rules if r.get("status") == "fail")
    warn_n = sum(1 for r in rules if r.get("status") == "warning")
    skip_n = sum(1 for r in rules if r.get("status") == "skipped")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ PASS",  pass_n)
    c2.metric("❌ FAIL",  fail_n)
    c3.metric("⚠️ WARN", warn_n)
    c4.metric("⏭️ SKIP", skip_n)

    for r in rules:
        s    = r.get("status", "skipped")
        sev  = r.get("severity", "info")
        rule = r.get("rule", "")
        msg  = r.get("message", "")
        det  = r.get("detail", "")
        lcv  = r.get("lc_value", "")
        docv = r.get("doc_value", "")
        dt   = r.get("doc_type", "")

        html = (
            f'<div class="rule-card" style="border-color:{_status_color(s)};background:{_status_bg(s)}">'
            f'<div class="rule-header">'
            f'{_status_icon(s)} <span class="rule-name">{rule}</span>'
            f'<span class="badge {_sev_class(sev)}">{sev}</span>'
            + (f'<span class="badge badge-doc">{dt}</span>' if dt else '') +
            f'</div><div class="rule-msg">{msg}</div>'
        )
        if det:
            html += f'<div class="rule-vals">{det}</div>'
        if lcv or docv:
            html += f'<div class="rule-vals">'
            if lcv:
                html += f'<span style="background:#1a2e1a;color:#86efac;padding:1px 6px;border-radius:4px;font-size:11px">LC : {lcv}</span> '
            if docv:
                html += f'<span style="background:#1e293b;color:#93c5fd;padding:1px 6px;border-radius:4px;font-size:11px">Doc : {docv}</span>'
            html += '</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


def _lc_cross_fuzzy(fuzzy: list):
    if not fuzzy:
        st.info("Aucun résultat de matching disponible"); return

    st.caption("Correspondance flou entre parties LC et documents commerciaux (seuil : 70%)")

    for f in fuzzy:
        check = f.get("check", "").replace("_", " ").title()
        score = float(f.get("score", 0))
        s     = f.get("status", "warning")
        lcv   = f.get("lc_value", "")
        docv  = f.get("doc_value", "")
        dt    = f.get("doc_type", "")
        col   = _score_color(score)

        st.markdown(
            f'<div class="fuzzy-card" style="border-left:3px solid {_status_color(s)}">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px">'
            f'{_status_icon(s)} '
            f'<span style="font-size:13px;font-weight:600;color:#e2e8f0;flex:1">{check}</span>'
            f'<span style="font-size:14px;font-weight:700;color:{col}">{round(score*100)}%</span>'
            f'<span class="badge badge-doc">{dt}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#94a3b8">LC : <span style="color:#e2e8f0">{lcv}</span></div>'
            f'<div style="font-size:12px;color:#94a3b8;margin-top:2px">Doc : <span style="color:#e2e8f0">{docv}</span></div>'
            f'<div class="fuzzy-match-bar"><div class="fuzzy-match-fill" style="width:{round(score*100)}%;background:{col}"></div></div>'
            f'</div>', unsafe_allow_html=True)


def _lc_cross_semantic(sem: dict):
    if not sem:
        st.info("Résultats sémantiques non disponibles"); return

    overall = float(sem.get("overall_score", 0))
    note    = sem.get("note", "")
    method  = sem.get("method", "")

    st.metric("Score sémantique global", _pct(overall))

    lc_vs_inv = sem.get("lc_vs_invoice")
    lc_vs_bl  = sem.get("lc_vs_bl")

    for pair_name, val, label_key in [
        ("LC (45A) ↔ Invoice",       lc_vs_inv, "lc_vs_invoice_label"),
        ("LC (45A) ↔ Bill of Lading", lc_vs_bl,  "lc_vs_bl_label"),
    ]:
        if val is None:
            continue
        vf  = float(val)
        col = _score_color(vf)
        lbl = sem.get(label_key, "")
        lbl_icon = {"MATCH": "✅ Concordant", "PARTIAL": "⚠️ Partiel",
                    "MISMATCH": "❌ Divergent"}.get(lbl, lbl)

        st.markdown(
            f'<div class="sem-card">'
            f'<div class="sem-title">{pair_name}</div>'
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<span style="color:{col};font-size:18px;font-weight:700">{_pct(vf)}</span>'
            f'<span style="font-size:12px;color:#94a3b8">{lbl_icon}</span>'
            f'</div>'
            f'<div class="sem-bar-bg"><div class="sem-bar-fill" style="width:{round(vf*100)}%;background:{col}"></div></div>'
            f'</div>', unsafe_allow_html=True)

    if note:
        st.info(f"ℹ️ {note}")
    if method:
        st.caption(f"Méthode : {method}")


def _lc_cross_conditions(conditions: list):
    if not conditions:
        st.info("Aucune condition 47A disponible"); return

    st.caption("Vérification des conditions additionnelles LC (champ 47A)")

    for r in conditions:
        s    = r.get("status", "skipped")
        sev  = r.get("severity", "info")
        rule = r.get("rule", "")
        msg  = r.get("message", "")
        lcv  = r.get("lc_value", "")
        docv = r.get("doc_value", "")
        dt   = r.get("doc_type", "")

        html = (
            f'<div class="rule-card" style="border-color:{_status_color(s)};background:{_status_bg(s)}">'
            f'<div class="rule-header">'
            f'{_status_icon(s)} <span class="rule-name">{rule}</span>'
            f'<span class="badge {_sev_class(sev)}">{sev}</span>'
            + (f'<span class="badge badge-doc">{dt}</span>' if dt else '') +
            f'</div><div class="rule-msg">{msg}</div>'
        )
        if lcv or docv:
            html += (
                f'<div class="rule-vals">'
                + (f'<span style="background:#1a2e1a;color:#86efac;padding:1px 6px;border-radius:4px;font-size:11px">LC : {lcv}</span> ' if lcv else '')
                + (f'<span style="background:#1e293b;color:#93c5fd;padding:1px 6px;border-radius:4px;font-size:11px">Doc : {docv}</span>' if docv else '')
                + '</div>'
            )
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)


def _lc_cross_explanations(explanations: list):
    if not explanations:
        st.info("Aucune explication disponible"); return

    for e in explanations:
        imp = e.get("impact", "neutral")
        sev = e.get("severity", "info")
        tp  = e.get("type", "rule")
        cat = e.get("category", "")
        msg = e.get("message", "")
        st.markdown(
            f'<div class="exp-card" style="border-color:{_impact_color(imp)}">'
            f'<span>{_impact_icon(imp)}</span>'
            f'<div><div class="exp-msg">{msg}</div>'
            f'<div class="exp-badges">'
            f'<span class="badge {_sev_class(sev)}">{sev}</span>'
            f'<span class="badge badge-doc">{tp}</span>'
            + (f'<span class="badge" style="background:#1e293b;color:#64748b">{cat}</span>' if cat else '')
            + f'</div></div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏦 LC Trade Validator v9")
    st.caption("MT700 · Documents · Cohérence LC↔Docs")
    st.divider()

    if st.button("🔌 Vérifier l'API", use_container_width=True):
        for url, label in [
            (HEALTH_URL,       "Pipeline MT700"),
            (COMM_HEALTH_URL,  "Commercial"),
            (LC_COMM_HEALTH_URL,"LC↔Docs"),
        ]:
            try:
                r = requests.get(url, timeout=5)
                data = r.json()
                status = data.get("status", "?")
                ico = "🟢" if status == "ok" else "🟡"
                st.success(f"{ico} {label} : {status}")
            except Exception as ex:
                st.error(f"🔴 {label} — {ex}")

    st.divider()
    st.caption("État des résultats")

    has_mt700  = "mt700_result"  in st.session_state
    has_comm   = "commercial_result" in st.session_state
    has_cross  = "lc_cross_result"   in st.session_state

    st.markdown(
        f"{'✅' if has_mt700 else '⬜'} MT700 analysé\n\n"
        f"{'✅' if has_comm  else '⬜'} Documents commerciaux\n\n"
        f"{'✅' if has_cross else '⬜'} Cohérence LC↔Docs"
    )

    if has_mt700 or has_comm or has_cross:
        st.divider()
        if st.button("🗑️ Réinitialiser tout", use_container_width=True):
            for k in ["mt700_result", "commercial_result", "lc_cross_result"]:
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>🏦 LC Trade Validator — Plateforme complète</h1>
  <p>① Validation MT700 · UCP 600 · Risk ML · Validation externe &nbsp;|&nbsp;
     ② Validation BL + Invoice + Packing List &nbsp;|&nbsp;
     ③ Cohérence LC ↔ Documents Commerciaux</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════

TAB1, TAB2, TAB3 = st.tabs([
    "📑 ① Validation MT700",
    "📦 ② Documents Commerciaux",
    "🔗 ③ Cohérence LC ↔ Documents",
])


# ════════════════════════════════
# TAB 1 — MT700
# ════════════════════════════════
with TAB1:
    st.subheader("① Validation MT700 — Pipeline complet")
    st.caption("OCR · NLP · UCP 600 · Risk ML · Validation externe OFAC/GLEIF/EU/UN")

    uploaded_mt700 = st.file_uploader(
        "📎 Déposez votre fichier SWIFT MT700 (PDF ou image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        key="uploader_mt700",
    )

    col_btn, col_text = st.columns([1, 2])
    with col_btn:
        run_mt700 = st.button("🚀 Analyser le MT700", type="primary",
                              disabled=(uploaded_mt700 is None), key="btn_mt700")

    if run_mt700 and uploaded_mt700:
        with st.spinner("Pipeline MT700 — OCR · Règles · NLP · UCP · Risk ML · Externe…"):
            try:
                resp = requests.post(
                    PIPELINE_URL,
                    files={"file": (uploaded_mt700.name, uploaded_mt700.getvalue())},
                    timeout=180,
                )
                if resp.status_code == 200:
                    st.session_state["mt700_result"] = resp.json()
                    st.success("✅ Analyse MT700 terminée")
                else:
                    st.error(f"Erreur API : {resp.status_code} — {resp.text[:200]}")
            except Exception as ex:
                st.error(f"Connexion échouée : {ex}")

    if "mt700_result" in st.session_state:
        st.divider()
        show_mt700_result(st.session_state["mt700_result"])

        with st.expander("🗃️ JSON Explorer — MT700", expanded=False):
            show_json_explorer(
                st.session_state["mt700_result"],
                file_name="mt700_result.json",
                default_open_key="swift",
            )


# ════════════════════════════════
# TAB 2 — COMMERCIAL DOCS
# ════════════════════════════════
with TAB2:
    st.subheader("② Validation Documents Commerciaux")
    st.caption("Bill of Lading · Invoice · Packing List — Extraction + Règles + NLI + Sémantique + Tables")

    lc_ref = st.text_input("Référence LC (optionnel)", placeholder="ex: LC2024-001234",
                           key="lc_ref_input")

    uploaded_docs = st.file_uploader(
        "📎 Déposez 1 à 3 documents (PDF ou image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
        key="uploader_commercial",
    )

    if uploaded_docs:
        if len(uploaded_docs) > 3:
            st.warning("⚠️ Maximum 3 documents. Seuls les 3 premiers seront traités.")
            uploaded_docs = uploaded_docs[:3]

        cols = st.columns(min(len(uploaded_docs), 3))
        for i, f in enumerate(uploaded_docs):
            with cols[i]:
                ext = f.name.split(".")[-1].upper()
                sz  = f"{round(f.size / 1024)} Ko"
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-val" style="font-size:24px">{"📄" if ext=="PDF" else "🖼️"}</div>'
                    f'<div class="kpi-label">{f.name[:25]}</div>'
                    f'<div class="kpi-sub">{ext} · {sz}</div>'
                    f'</div>', unsafe_allow_html=True)

    run_comm = st.button(
        "🚀 Valider les documents commerciaux", type="primary",
        disabled=(not uploaded_docs), key="btn_commercial"
    )

    if run_comm and uploaded_docs:
        with st.spinner("Pipeline commercial — Classification · Extraction · Validation croisée…"):
            try:
                files = [("files", (f.name, f.getvalue())) for f in uploaded_docs[:3]]
                data  = {}
                if lc_ref.strip():
                    data["lcReference"] = lc_ref.strip()
                resp = requests.post(COMMERCIAL_URL, files=files, data=data, timeout=300)
                if resp.status_code == 200:
                    st.session_state["commercial_result"] = resp.json()
                    st.success("✅ Validation commerciale terminée")
                else:
                    st.error(f"Erreur API : {resp.status_code} — {resp.text[:300]}")
            except Exception as ex:
                st.error(f"Connexion échouée : {ex}")

    if "commercial_result" in st.session_state:
        st.divider()
        show_commercial_result(st.session_state["commercial_result"])

        with st.expander("🗃️ JSON Explorer — Documents Commerciaux", expanded=False):
            show_json_explorer(
                st.session_state["commercial_result"],
                file_name="commercial_result.json",
                default_open_key="documents",
            )


# ════════════════════════════════
# TAB 3 — LC ↔ COMMERCIAL
# ════════════════════════════════
with TAB3:
    st.subheader("③ Cohérence LC ↔ Documents Commerciaux")
    st.caption("4 couches : Règles UCP 600 · Matching flou entités · Sémantique 45A · Conditions 47A")

    has_mt700 = "mt700_result"       in st.session_state
    has_comm  = "commercial_result"  in st.session_state

    if not has_mt700 and not has_comm:
        st.info(
            "🔷 Complétez les deux étapes précédentes pour activer la validation de cohérence :\n\n"
            "- **Onglet ①** : Analysez un MT700\n"
            "- **Onglet ②** : Validez les documents commerciaux"
        )
    elif not has_mt700:
        st.warning("⚠️ Analysez d'abord un MT700 (Onglet ①)")
    elif not has_comm:
        st.warning("⚠️ Validez d'abord les documents commerciaux (Onglet ②)")
    else:
        mt700_res = st.session_state["mt700_result"]
        comm_res  = st.session_state["commercial_result"]

        # Preview des données qui seront utilisées
        lc_payload   = _extract_lc_payload(mt700_res)
        comm_payload = _extract_commercial_payload(comm_res)

        with st.expander("🔍 Aperçu des données LC extraites du MT700"):
            non_null = {k: v for k, v in lc_payload.items() if v is not None}
            import pandas as pd
            if non_null:
                df = pd.DataFrame(list(non_null.items()), columns=["Champ LC", "Valeur"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun champ LC extrait — vérifiez le résultat MT700")

        with st.expander("🔍 Documents commerciaux disponibles"):
            docs_types = [k for k in ["bl", "invoice", "packing_list"] if k in comm_payload]
            if docs_types:
                st.success(f"Documents disponibles : {', '.join(docs_types)}")
                for dt in docs_types:
                    fields = comm_payload.get(dt) or {}
                    st.caption(f"**{_doc_label(dt)}** — {len(fields)} champs")
                    if fields:
                        df2 = pd.DataFrame(list(fields.items())[:10], columns=["Champ", "Valeur"])
                        st.dataframe(df2, use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun document extrait — vérifiez le résultat commercial")

        st.divider()

        run_cross = st.button(
            "🔗 Lancer la validation de cohérence LC ↔ Documents",
            type="primary",
            key="btn_lc_cross",
            disabled=(not docs_types or not non_null),
        )

        if run_cross:
            with st.spinner("Validation de cohérence LC ↔ Documents — 4 couches en cours…"):
                try:
                    payload = {
                        "lc":         lc_payload,
                        "commercial": {k: v for k, v in comm_payload.items()
                                      if not k.endswith("_types")},
                    }
                    resp = requests.post(LC_COMMERCIAL_URL, json=payload, timeout=120)
                    if resp.status_code == 200:
                        st.session_state["lc_cross_result"] = resp.json()
                        st.success("✅ Validation de cohérence terminée")
                    else:
                        st.error(f"Erreur API : {resp.status_code} — {resp.text[:300]}")
                except Exception as ex:
                    st.error(f"Connexion échouée : {ex}")

        if "lc_cross_result" in st.session_state:
            st.divider()
            show_lc_cross_result(st.session_state["lc_cross_result"])

            with st.expander("🗃️ JSON Explorer — Cohérence LC↔Docs", expanded=False):
                show_json_explorer(
                    st.session_state["lc_cross_result"],
                    file_name="lc_cross_result.json",
                    default_open_key="score_breakdown",
                )
