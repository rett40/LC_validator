"""
streamlit_app.py  —  v8.0
=======================================
Interface SWIFT Pipeline Hybride — Dashboard Global Redesigné

NOUVEAUTÉS v8.0 :
  ✦ Dashboard de validation global avec KPIs visuels en haut
  ✦ Tableau complet des champs SWIFT (tag | nom | valeur)
  ✦ Tableau des clauses avec classification et risk level
  ✦ Navigation par onglets : Champs SWIFT | UCP | Cohérence | Validation | Externe | Risk ML
  ✦ Résumé exécutif condensé (XIA) avec verdict global
  ✦ Design épuré : dark header + cartes colorées
"""

import json
import requests
import streamlit as st
from collections import Counter

API_BASE        = "http://localhost:5000"
PIPELINE_URL    = f"{API_BASE}/pipeline"
ANALYZE_URL     = f"{API_BASE}/swift/analyze"
NLP_URL         = f"{API_BASE}/swift/nlp"
SPELL_URL       = f"{API_BASE}/swift/spell"
CLASSIFY_URL    = f"{API_BASE}/swift/classify"
HEALTH_URL      = f"{API_BASE}/health"
RISK_HEALTH_URL = f"{API_BASE}/risk/health"

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SWIFT Intelligence v8",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }

  /* ── HEADER HERO ── */
  .hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid #334155;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #3b82f620 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 26px; font-weight: 700; color: #f8fafc;
    letter-spacing: -0.5px; margin: 0 0 4px 0;
  }
  .hero-subtitle { font-size: 13px; color: #94a3b8; margin: 0; }
  .hero-type-badge {
    display: inline-block;
    background: #3b82f6; color: white;
    padding: 4px 14px; border-radius: 20px;
    font-size: 13px; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1px; margin-bottom: 12px;
  }

  /* ── VERDICT CARD ── */
  .verdict-card {
    border-radius: 14px; padding: 20px 24px;
    display: flex; flex-direction: column; gap: 6px;
    border: 1px solid transparent;
    height: 100%;
  }
  .verdict-VALID   { background: #f0fdf4; border-color: #86efac; }
  .verdict-INVALID { background: #fff1f2; border-color: #fda4af; }
  .verdict-WARN    { background: #fffbeb; border-color: #fcd34d; }
  .verdict-INFO    { background: #eff6ff; border-color: #93c5fd; }
  .verdict-title   { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #64748b; }
  .verdict-score   { font-size: 36px; font-weight: 800; line-height: 1; color: #1e293b; font-family: 'IBM Plex Mono', monospace; }
  .verdict-label   { font-size: 13px; font-weight: 500; }
  .verdict-sub     { font-size: 11px; color: #94a3b8; }

  /* ── SECTION TABS ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f8fafc;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    color: #64748b !important;
  }
  .stTabs [aria-selected="true"] {
    background: white !important;
    color: #1e293b !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
  }

  /* ── SWIFT FIELD TABLE ── */
  .swift-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .swift-table thead tr {
    background: #1e293b; color: #f8fafc;
  }
  .swift-table thead th {
    padding: 10px 14px; text-align: left;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
  }
  .swift-table tbody tr { border-bottom: 1px solid #f1f5f9; }
  .swift-table tbody tr:hover { background: #f8fafc; }
  .swift-table tbody td { padding: 9px 14px; vertical-align: top; }
  .tag-mono {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 600;
    background: #eff6ff; color: #1d4ed8;
    padding: 2px 8px; border-radius: 5px;
  }
  .field-name { color: #475569; font-size: 12px; }
  .field-value { color: #1e293b; font-weight: 500; }
  .field-value-long {
    color: #1e293b;
    font-size: 12px;
    background: #f8fafc;
    border-radius: 6px;
    padding: 6px 10px;
    border: 1px solid #e2e8f0;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 80px;
    overflow: hidden;
  }
  .src-badge {
    display: inline-block; font-size: 10px;
    background: #e0f2fe; color: #0369a1;
    padding: 1px 6px; border-radius: 4px; margin-left: 4px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* ── CLAUSE TABLE ── */
  .clause-row {
    border-radius: 8px; margin-bottom: 8px;
    border: 1px solid #e2e8f0; overflow: hidden;
  }
  .clause-head {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
  }
  .clause-body { padding: 10px 14px; font-size: 13px; color: #374151; line-height: 1.5; }
  .risk-pill {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .risk-high   { background: #fee2e2; color: #991b1b; }
  .risk-medium { background: #fef3c7; color: #92400e; }
  .risk-low    { background: #d1fae5; color: #065f46; }
  .risk-normal { background: #eff6ff; color: #1e40af; }
  .type-pill {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 500;
    background: #f1f5f9; color: #475569;
  }
  .method-pill {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 10px; font-weight: 500; font-family: 'IBM Plex Mono', monospace;
    background: #faf5ff; color: #7c3aed; margin-left: auto;
  }

  /* ── GLOBAL VERDICT HERO ── */
  .global-verdict {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border-radius: 16px; padding: 24px 32px;
    display: flex; align-items: center; gap: 24px;
    border: 1px solid #334155; margin-bottom: 20px;
  }
  .gv-score-big {
    font-size: 72px; font-weight: 800; line-height: 1;
    font-family: 'IBM Plex Mono', monospace;
  }
  .gv-score-valid   { color: #4ade80; }
  .gv-score-warning { color: #fbbf24; }
  .gv-score-invalid { color: #f87171; }
  .gv-label { font-size: 22px; font-weight: 700; color: #f8fafc; }
  .gv-sub   { font-size: 13px; color: #94a3b8; margin-top: 4px; }
  .gv-sep   { width: 1px; background: #334155; height: 80px; }

  /* ── KPI STRIP ── */
  .kpi-strip {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;
  }
  .kpi-card {
    flex: 1; min-width: 110px;
    background: white; border-radius: 12px;
    padding: 14px 16px; border: 1px solid #e2e8f0;
    text-align: center;
  }
  .kpi-icon   { font-size: 22px; margin-bottom: 4px; }
  .kpi-value  { font-size: 22px; font-weight: 700; color: #1e293b; font-family: 'IBM Plex Mono', monospace; }
  .kpi-label  { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }

  /* ── XIA BLOCK ── */
  .xia-block {
    background: #fafafa; border-radius: 12px;
    border: 1px solid #e2e8f0; padding: 20px 24px;
    margin-bottom: 20px;
  }
  .xia-title { font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; }
  .xia-row   { display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid #f1f5f9; }
  .xia-row:last-child { border-bottom: none; }
  .xia-name  { font-size: 13px; color: #475569; flex: 1; }
  .xia-val   { font-size: 13px; font-weight: 600; color: #1e293b; font-family: 'IBM Plex Mono', monospace; }

  /* ── EXT RISK ── */
  .ext-risk-bar {
    height: 8px; border-radius: 4px; margin-top: 4px;
    background: linear-gradient(to right, #4ade80, #fbbf24, #f87171);
  }

  /* ── SHAP ── */
  .shap-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .shap-feat { font-size: 12px; color: #374151; width: 170px; flex-shrink: 0; font-family: 'IBM Plex Mono', monospace; font-size: 11px; }
  .shap-bar-bg { flex: 1; background: #f1f5f9; border-radius: 4px; height: 10px; overflow: hidden; }
  .shap-bar-fill { height: 10px; border-radius: 4px; }
  .shap-dir { font-size: 11px; width: 55px; text-align: right; font-weight: 600; }
  .shap-num { font-size: 10px; color: #94a3b8; width: 55px; text-align: right; font-family: 'IBM Plex Mono', monospace; }

  /* ── SECTION HEADER ── */
  .sec-header {
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #64748b; border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px; margin: 16px 0 12px 0;
  }

  /* ── FINDING ROW ── */
  .finding-row {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 8px 12px; border-radius: 8px; margin-bottom: 4px;
    font-size: 13px;
  }
  .finding-error   { background: #fff1f2; border-left: 3px solid #f87171; }
  .finding-warning { background: #fffbeb; border-left: 3px solid #fbbf24; }
  .finding-info    { background: #eff6ff; border-left: 3px solid #93c5fd; }
  .finding-ok      { background: #f0fdf4; border-left: 3px solid #4ade80; }

  div[data-testid="stExpander"] > div:first-child { font-weight: 500; }

  /* score bar */
  .score-bar-wrap { margin: 6px 0; }
  .score-bar-bg { background: #e2e8f0; border-radius: 6px; height: 8px; overflow: hidden; }
  .score-bar-fill { height: 8px; border-radius: 6px; transition: width 0.5s; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def score_color(s):
    if s >= 80: return "#22c55e"
    if s >= 50: return "#f59e0b"
    return "#ef4444"

def score_emoji(s):
    if s >= 80: return "🟢"
    if s >= 50: return "🟡"
    return "🔴"

def _get_swift_payload(data: dict) -> dict:
    """Compat: supporte payload pipeline (avec `swift`) et payload legacy."""
    swift = (data or {}).get("swift")
    if isinstance(swift, dict):
        return swift
    return data or {}

def _coerce_list(value):
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]

def ner_icon(label):
    return {
        "APPLICANT": "🏢", "BENEFICIARY": "🏢", "ISSUING_BANK": "🏦",
        "ADVISING_BANK": "🏦", "AMOUNT": "💰", "GOODS": "📦",
        "INCOTERM": "🚢", "EXPIRY_DATE": "📅", "ISSUE_DATE": "📅",
        "SHIPMENT_DATE": "📅", "LC_NUMBER": "🔑", "PORT": "⚓", "COUNTRY": "🌍",
    }.get(label, "🔷")

EXT_ICONS  = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "CLEAN": "🟢"}
EXT_COLORS = {"CRITICAL": "#7b0000", "HIGH": "#dc3545", "MEDIUM": "#fd7e14", "LOW": "#0dcaf0", "CLEAN": "#198754"}
RISK_ML_ICONS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

SWIFT_FIELD_NAMES = {
    "27":  "Séquence totale",
    "40A": "Forme du crédit",
    "20":  "Référence LC",
    "31C": "Date d'émission",
    "40E": "Règles applicables",
    "31D": "Date / lieu d'expiration",
    "50":  "Donneur d'ordre",
    "59":  "Bénéficiaire",
    "32B": "Devise / Montant",
    "39A": "Tolérance (%)",
    "39B": "Montant maximum",
    "39C": "Montant additionnel",
    "41A": "Disponible auprès de / par (BIC)",
    "41D": "Disponible auprès de / par",
    "42C": "Tirage à",
    "42A": "Banque tirée (BIC)",
    "42D": "Banque tirée",
    "43P": "Expéditions partielles",
    "43T": "Transbordement",
    "44E": "Port de chargement",
    "44F": "Port de déchargement",
    "44C": "Dernière date d'expédition",
    "44D": "Période d'expédition",
    "45A": "Description des marchandises",
    "46A": "Documents requis",
    "47A": "Conditions supplémentaires",
    "48":  "Délai de présentation",
    "49":  "Confirmation",
    "53A": "Banque de remboursement (BIC)",
    "53D": "Banque de remboursement",
    "71B": "Frais et commissions",
    "72":  "Sender to Receiver",
    "77U": "Conditions de la garantie",
    "78":  "Instructions à la banque",
    "57A": "Banque avisante (BIC)",
    "57D": "Banque avisante",
}

CLAUSE_TYPE_LABELS = {
    "document_condition":   ("📄", "Document"),
    "shipment_condition":   ("🚢", "Expédition"),
    "payment_condition":    ("💰", "Paiement"),
    "compliance_condition": ("⚖️", "Conformité"),
    "tolerance_condition":  ("📏", "Tolérance"),
}


# ══════════════════════════════════════════════════════════════════════
# BLOC A — DASHBOARD GLOBAL (hero + KPIs)
# ══════════════════════════════════════════════════════════════════════

def show_global_dashboard(data: dict):
    swift   = _get_swift_payload(data)
    nlp     = swift.get("nlp", {}) or {}
    val     = swift.get("validation", {}) or {}
    ocr     = data.get("ocr", {}) or {}
    sem     = nlp.get("semantic", {}) or {}
    ucp     = data.get("ucp_compliance", {}) or {}
    ds      = data.get("ds_analysis", {}) or {}
    ext     = data.get("external_validation", {}) or {}
    risk_ml = data.get("risk_ml", {}) or {}

    nlp_score    = nlp.get("score", 0)
    val_score    = val.get("score", 0)
    val_valid    = val.get("overall_valid", False)
    completeness = sem.get("completeness_score", 0)
    ocr_conf     = int(ocr.get("confidence", 0) * 100)
    swift_type   = swift.get("swift_type", "—")
    n_entities   = len((nlp.get("ner") or {}).get("entities", []))
    n_clauses    = len(nlp.get("clauses", []))
    n_contradict = len(nlp.get("contradictions", []))
    ucp_score    = ucp.get("global_score", 0)
    ucp_ok       = ucp.get("compliant", False)
    ext_level    = ext.get("risk_level", "CLEAN") if ext else "—"
    ds_risk      = (ds.get("risk", {}) or {}).get("risk_level", "—") if ds else "—"
    ml_label     = risk_ml.get("risk_label", "—") if risk_ml and not risk_ml.get("error") else "—"
    ml_score     = risk_ml.get("risk_score", 0) if risk_ml else 0
    missing_m    = len(val.get("missing_mandatory", []))
    risk_flags   = len(sem.get("risk_flags", []))

    # Score composite global
    scores = [s for s in [nlp_score, val_score, ucp_score, completeness] if s > 0]
    global_score = int(sum(scores) / len(scores)) if scores else 0

    if global_score >= 80 and val_valid and not missing_m:
        verdict_class = "gv-score-valid"
        verdict_label = "✅ CONFORME"
        verdict_sub   = "Document LC validé — traitement recommandé"
    elif missing_m or not val_valid or global_score < 50:
        verdict_class = "gv-score-invalid"
        verdict_label = "❌ NON CONFORME"
        verdict_sub   = f"{missing_m} champ(s) obligatoire(s) manquant(s)" if missing_m else "Validation échouée"
    else:
        verdict_class = "gv-score-warning"
        verdict_label = "⚠️ À VÉRIFIER"
        verdict_sub   = f"{risk_flags} flag(s) de risque détecté(s)"

    # HERO banner
    st.markdown(f"""
    <div class="global-verdict">
      <div>
        <div class="gv-score-big {verdict_class}">{global_score}</div>
        <div style="font-size:11px;color:#64748b;font-family:'IBM Plex Mono',monospace;margin-top:2px">SCORE /100</div>
      </div>
      <div class="gv-sep"></div>
      <div>
        <div class="hero-type-badge">{swift_type}</div>
        <div class="gv-label">{verdict_label}</div>
        <div class="gv-sub">{verdict_sub}</div>
      </div>
      <div class="gv-sep"></div>
      <div style="display:flex;flex-direction:column;gap:8px;flex:1">
        <div style="display:flex;gap:16px;flex-wrap:wrap">
          <div style="color:#94a3b8;font-size:12px">📊 NLP <strong style="color:#f8fafc">{nlp_score}/100</strong></div>
          <div style="color:#94a3b8;font-size:12px">✅ Validation <strong style="color:#f8fafc">{val_score}/100</strong></div>
          <div style="color:#94a3b8;font-size:12px">⚖️ UCP <strong style="color:#f8fafc">{ucp_score}/100</strong></div>
          <div style="color:#94a3b8;font-size:12px">📋 Complétude <strong style="color:#f8fafc">{completeness}%</strong></div>
        </div>
        <div style="display:flex;gap:16px;flex-wrap:wrap">
          <div style="color:#94a3b8;font-size:12px">🌐 Ext <strong style="color:#f8fafc">{EXT_ICONS.get(ext_level,'?')} {ext_level}</strong></div>
          <div style="color:#94a3b8;font-size:12px">🤖 DS <strong style="color:#f8fafc">{ds_risk}</strong></div>
          <div style="color:#94a3b8;font-size:12px">⚡ ML <strong style="color:#f8fafc">{RISK_ML_ICONS.get(ml_label,'?')} {ml_label}</strong></div>
          <div style="color:#94a3b8;font-size:12px">🔍 OCR <strong style="color:#f8fafc">{ocr_conf}%</strong></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    kpis = [
        ("🔑", n_entities, "Entités"),
        ("📌", n_clauses, "Clauses"),
        ("⚠️", n_contradict, "Contradictions"),
        ("🔴", missing_m, "Champs manquants"),
        ("🚩", risk_flags, "Risk flags"),
        ("📄", len(val.get("fields", {})), "Champs validés"),
    ]
    cols = st.columns(len(kpis))
    for i, (icon, val_num, label) in enumerate(kpis):
        with cols[i]:
            color = "#ef4444" if (label in ["Contradictions", "Champs manquants", "Risk flags"] and val_num > 0) else "#1e293b"
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value" style="color:{color}">{val_num}</div>
              <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# BLOC B — TABLEAU CHAMPS SWIFT
# ══════════════════════════════════════════════════════════════════════

def show_swift_fields_tab(data: dict):
    swift    = _get_swift_payload(data)
    entities = swift.get("entities", {}) or {}
    val      = swift.get("validation", {}) or {}
    val_fields = val.get("fields", {}) or {}

    st.markdown('<div class="sec-header">📋 Champs SWIFT — Vue complète</div>', unsafe_allow_html=True)

    # Construire les lignes depuis raw_fields + entités
    raw_fields = entities.get("raw_fields", {}) or {}

    STRUCTURED = [
        ("27",  "sequence",            entities.get("sequence")),
        ("40A", "lc_form",             entities.get("lc_form")),
        ("20",  "lc_number",           entities.get("lc_number")),
        ("31C", "issue_date",          entities.get("issue_date")),
        ("40E", "applicable_rules",    entities.get("applicable_rules")),
        ("31D", "expiry_date",         entities.get("expiry_date")),
        ("50",  "applicant",           entities.get("applicant")),
        ("59",  "beneficiary",         entities.get("beneficiary")),
        ("32B", "amount",              f"{entities.get('currency','')} {entities.get('amount','')}".strip() or None),
        ("39A", "tolerance",           f"+{entities.get('tolerance_plus','')}% / -{entities.get('tolerance_minus','')}%"
                                        if entities.get("tolerance_plus") else None),
        ("41D", "available_with",      entities.get("available_with")),
        ("41D", "available_by",        entities.get("available_by")),
        ("42C", "draft_at",            entities.get("draft_at")),
        ("43P", "partial_shipments",   entities.get("partial_shipments")),
        ("43T", "transhipment",        entities.get("transhipment")),
        ("44E", "port_loading",        entities.get("port_loading")),
        ("44F", "port_discharge",      entities.get("port_discharge")),
        ("44C", "shipment_date",       entities.get("shipment_date")),
        ("48",  "presentation_period", entities.get("presentation_period")),
        ("49",  "confirmation",        entities.get("confirmation")),
        ("45A", "goods_description",   entities.get("goods_description")),
        ("46A", "documents_required",  entities.get("documents_required")),
        ("47A", "additional_conditions", entities.get("additional_conditions")),
        ("71B", "charges",             entities.get("charges")),
        ("72",  "sender_info",         entities.get("sender_info")),
        ("77U", "guarantee_text",      entities.get("guarantee_text")),
    ]

    # Ajouter les champs bruts non encore couverts
    all_tags_shown = set(tag for tag, _, _ in STRUCTURED)

    rows_html = ""
    row_count = 0
    for tag, key, value in STRUCTURED:
        if not value:
            # Essayer depuis raw_fields
            value = raw_fields.get(tag) or raw_fields.get(f":{tag}:")
        if not value:
            continue

        field_name = SWIFT_FIELD_NAMES.get(tag, key.replace("_", " ").title())
        src        = entities.get(f"_{key}_source", "")
        src_badge  = f'<span class="src-badge">🤖 NLP</span>' if src else ""
        vfield     = val_fields.get(tag, {})
        errs       = vfield.get("errors", [])
        warns      = vfield.get("warnings", [])
        status_dot = "🔴" if errs else ("🟡" if warns else "🟢")

        val_str = str(value).strip()
        is_long = len(val_str) > 100
        if is_long:
            val_html = f'<div class="field-value-long">{val_str[:400]}{"…" if len(val_str)>400 else ""}</div>'
        else:
            val_html = f'<span class="field-value">{val_str[:200]}</span>'

        bg = "#fff1f2" if errs else ("#fffbeb" if warns else "")
        bg_style = f'style="background:{bg}"' if bg else ""

        rows_html += f"""
        <tr {bg_style}>
          <td><span class="tag-mono">:{tag}:</span></td>
          <td><span class="field-name">{field_name}</span></td>
          <td>{val_html}{src_badge}</td>
          <td style="text-align:center">{status_dot}</td>
        </tr>"""
        row_count += 1

    # Champs bruts supplémentaires
    for tag, raw_val in raw_fields.items():
        clean_tag = tag.replace(":", "")
        if clean_tag in all_tags_shown or not raw_val:
            continue
        fname  = SWIFT_FIELD_NAMES.get(clean_tag, f"Champ {clean_tag}")
        val_str = str(raw_val).strip()
        is_long = len(val_str) > 100
        if is_long:
            val_html = f'<div class="field-value-long">{val_str[:400]}</div>'
        else:
            val_html = f'<span class="field-value">{val_str}</span>'

        rows_html += f"""
        <tr>
          <td><span class="tag-mono">:{clean_tag}:</span></td>
          <td><span class="field-name">{fname}</span></td>
          <td>{val_html}</td>
          <td style="text-align:center">⚪</td>
        </tr>"""
        row_count += 1

    if not rows_html:
        st.info("Aucun champ SWIFT extrait. Vérifiez le fichier source.")
        return

    st.markdown(f"""
    <table class="swift-table">
      <thead>
        <tr>
          <th style="width:80px">Tag</th>
          <th style="width:220px">Nom du champ</th>
          <th>Valeur</th>
          <th style="width:60px;text-align:center">État</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div style="margin-top:8px;font-size:11px;color:#94a3b8">{row_count} champ(s) extrait(s) · 🟢 Valide · 🟡 Avertissement · 🔴 Erreur · 🤖 Enrichi par NLP</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# BLOC C — CLAUSES
# ══════════════════════════════════════════════════════════════════════

def show_clauses_tab(data: dict):
    swift   = _get_swift_payload(data)
    nlp     = swift.get("nlp", {}) or {}
    clauses = nlp.get("clauses", [])
    contras = nlp.get("contradictions", [])

    st.markdown('<div class="sec-header">📌 Clauses extraites & Classification</div>', unsafe_allow_html=True)

    if not clauses:
        st.info("Aucune clause extraite.")
        return

    # Résumé
    type_counts = Counter(c.get("type", "unknown") for c in clauses)
    risk_counts = Counter(c.get("risk_level", "normal") for c in clauses)
    ml_count    = sum(1 for c in clauses if "embedding" in c.get("method", "") or "supervised" in c.get("method", ""))

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Total clauses",      len(clauses))
    with col2: st.metric("🔴 Haut risque",     risk_counts.get("high", 0))
    with col3: st.metric("🟡 Risque moyen",    risk_counts.get("medium", 0))
    with col4: st.metric("🤖 Classifiées ML",  ml_count)
    with col5: st.metric("⚠️ Contradictions",  len(contras))

    # Contradictions
    if contras:
        st.error(f"**⚠️ {len(contras)} contradiction(s) détectée(s)**")
        for cont in contras:
            with st.container():
                st.markdown(f"""
                <div style="background:#fff1f2;border:1px solid #fda4af;border-radius:8px;padding:12px 16px;margin-bottom:8px">
                  <div style="font-weight:600;color:#991b1b;margin-bottom:8px">🔴 {cont.get("message","")[:100]}</div>
                  <div style="display:flex;gap:12px">
                    <div style="flex:1;background:#fff8f8;border-radius:6px;padding:8px 10px;font-size:12px">
                      <div style="font-size:10px;color:#94a3b8;margin-bottom:4px">CLAUSE A</div>
                      {cont.get('clause_a','')[:150]}
                    </div>
                    <div style="flex:1;background:#fff8f8;border-radius:6px;padding:8px 10px;font-size:12px">
                      <div style="font-size:10px;color:#94a3b8;margin-bottom:4px">CLAUSE B</div>
                      {cont.get('clause_b','')[:150]}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Filtres
    cf1, cf2 = st.columns([2, 2])
    with cf1:
        filter_type = st.selectbox("Filtrer par type", ["Tous"] + list(type_counts.keys()), key="clause_type_filter")
    with cf2:
        filter_risk = st.selectbox("Filtrer par risque", ["Tous", "high", "medium", "low", "normal"], key="clause_risk_filter")

    filtered = [c for c in clauses
                if (filter_type == "Tous" or c.get("type") == filter_type)
                and (filter_risk == "Tous" or c.get("risk_level", "normal") == filter_risk)]

    st.caption(f"Affichage : {len(filtered)}/{len(clauses)} clause(s)")

    for c in filtered:
        risk    = c.get("risk_level", "normal")
        ctype   = c.get("type", "unknown")
        method  = c.get("method", "")
        conf    = c.get("confidence", 0)
        ucp_refs = c.get("ucp_refs", [])
        text    = c.get("text", "")
        imp     = c.get("importance", "low")

        t_icon, t_label = CLAUSE_TYPE_LABELS.get(ctype, ("📌", ctype))
        risk_class = f"risk-{risk}"

        border_colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e", "normal": "#3b82f6"}
        bc = border_colors.get(risk, "#94a3b8")

        ucp_str = " · ".join(ucp_refs[:3]) if ucp_refs else ""
        method_label = "🤖 ML" if ("embedding" in method or "supervised" in method) else "📏 Règle"

        st.markdown(f"""
        <div style="border-left:3px solid {bc};border:1px solid #e2e8f0;border-left:4px solid {bc};
                    border-radius:8px;margin-bottom:8px;overflow:hidden">
          <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;background:#f8fafc;flex-wrap:wrap">
            <span class="type-pill">{t_icon} {t_label}</span>
            <span class="risk-pill {risk_class}">{risk.upper()}</span>
            <span style="font-size:11px;color:#94a3b8">{method_label}</span>
            <span style="font-size:11px;color:#94a3b8">conf {conf:.2f}</span>
            {"<span style='font-size:11px;color:#7c3aed;font-family:monospace'>" + ucp_str + "</span>" if ucp_str else ""}
            <span style="font-size:11px;color:#94a3b8;margin-left:auto">imp: {imp}</span>
          </div>
          <div style="padding:10px 14px;font-size:13px;color:#374151;line-height:1.5">
            {text[:350]}{"…" if len(text) > 350 else ""}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# BLOC D — VALIDATION CHAMPS
# ══════════════════════════════════════════════════════════════════════

def show_validation_tab(data: dict):
    swift = _get_swift_payload(data)
    val   = swift.get("validation", {}) or {}

    if not val:
        st.info("Aucune donnée de validation disponible.")
        return

    overall  = val.get("overall_valid", False)
    score    = val.get("score", 0)
    missing  = val.get("missing_mandatory", [])
    cross_e  = val.get("cross_field_errors", [])
    cross_w  = val.get("cross_field_warnings", [])
    fields   = val.get("fields", {}) or {}

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Statut global",  "✅ VALIDE" if overall else "❌ INVALIDE")
    with c2: st.metric("Score",          f"{score}/100")
    with c3: st.metric("Champs manquants", len(missing))
    with c4: st.metric("Erreurs inter-champs", len(cross_e))

    sc = score_color(score)
    st.markdown(f"""
    <div class="score-bar-wrap">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:4px">
        <span>Score de validation</span><span>{score}/100</span>
      </div>
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score}%;background:{sc}"></div></div>
    </div>
    """, unsafe_allow_html=True)

    if missing:
        st.error(f"**Champs obligatoires manquants :** " + "  ·  ".join(f"`:{t}:`" for t in missing))

    for e in cross_e:
        st.markdown(f'<div class="finding-row finding-error">🔴 {e}</div>', unsafe_allow_html=True)
    for w in cross_w:
        st.markdown(f'<div class="finding-row finding-warning">🟡 {w}</div>', unsafe_allow_html=True)

    if fields:
        errs  = {t: f for t, f in fields.items() if f.get("errors")}
        warns = {t: f for t, f in fields.items() if not f.get("errors") and f.get("warnings")}
        ok    = {t: f for t, f in fields.items() if not f.get("errors") and not f.get("warnings")}

        def _render(fd, icon):
            if not fd: return
            rows = []
            for tag, fv in sorted(fd.items()):
                msgs = fv.get("errors", []) or fv.get("warnings", [])
                rows.append({"Tag": f":{tag}:", "Champ": fv.get("name", tag),
                             "Valeur": str(fv.get("value") or "—")[:70],
                             "Note": msgs[0][:80] if msgs else "—"})
            st.dataframe(rows, hide_index=True, width="stretch")

        if errs:
            st.markdown(f'<div class="sec-header">🔴 Erreurs ({len(errs)})</div>', unsafe_allow_html=True)
            _render(errs, "🔴")
        if warns:
            st.markdown(f'<div class="sec-header">🟡 Avertissements ({len(warns)})</div>', unsafe_allow_html=True)
            _render(warns, "🟡")
        if ok:
            with st.expander(f"🟢 Champs valides ({len(ok)})", expanded=False):
                _render(ok, "🟢")


# ══════════════════════════════════════════════════════════════════════
# BLOC E — UCP 600
# ══════════════════════════════════════════════════════════════════════

def show_ucp_tab(data: dict):
    ucp = data.get("ucp_compliance", {}) or {}
    if not ucp:
        st.info("Analyse UCP non disponible.")
        return

    global_score = ucp.get("global_score", 0)
    compliant    = ucp.get("compliant", False)
    findings     = ucp.get("findings", []) or []
    categories   = ucp.get("categories", {}) or {}
    exclusions   = ucp.get("exclusions", {}) or {}

    errors   = [f for f in findings if f.get("severity") == "error"]
    warnings = [f for f in findings if f.get("severity") == "warning"]
    infos    = [f for f in findings if f.get("severity") == "info"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Score UCP",   f"{global_score}/100")
    with c2: st.metric("Conformité",  "✅ Conforme" if compliant else "⚠️ Non conforme")
    with c3: st.metric("🔴 Erreurs",  len(errors))
    with c4: st.metric("🟡 Alertes",  len(warnings))
    with c5: st.metric("ℹ️ Infos",    len(infos))

    p1, p2, p3 = st.columns(3)
    with p1: st.metric("Blocage paiement", "Oui" if ucp.get("payment_blocked") else "Non")
    with p2: st.metric("Temps", f"{ucp.get('processing_ms', 0)}ms")
    with p3:
        decision_raw = ucp.get("ucp_decision")
        if isinstance(decision_raw, dict):
            decision = decision_raw.get("decision", "—")
        elif isinstance(decision_raw, str):
            decision = decision_raw
        else:
            decision = "—"
        st.metric("Décision", decision)

    sc = score_color(global_score)
    st.markdown(f"""
    <div class="score-bar-wrap">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:4px">
        <span>Score UCP 600</span><span>{global_score}/100</span>
      </div>
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:{global_score}%;background:{sc}"></div></div>
    </div>
    """, unsafe_allow_html=True)

    if categories:
        st.markdown('<div class="sec-header">Scores par catégorie</div>', unsafe_allow_html=True)
        cat_icons = {"document": "📄", "compliance": "⚖️", "payment": "💰", "shipment": "🚢"}
        cols = st.columns(len(categories))
        for i, (cat, cdata) in enumerate(categories.items()):
            score = cdata.get("score", 100)
            nok   = cdata.get("articles_nok", [])
            icon  = cat_icons.get(cat, "•")
            with cols[i]:
                st.metric(f"{icon} {cat.capitalize()}", f"{score_emoji(score)} {score}/100")
                if nok: st.caption(", ".join(nok[:3]))

    if exclusions:
        with st.expander(f"📋 Articles exclus ({len(exclusions)})", expanded=False):
            for art, ctx in exclusions.items():
                st.markdown(f"- **{art}** : {ctx[:80]}")

    sem_meta = ucp.get("semantic", {}) or {}
    if sem_meta:
        mode = sem_meta.get("mode", "unknown")
        if mode == "semantic" and sem_meta.get("available"):
            st.caption(
                "Moteur UCP sémantique actif · "
                f"violations: {sem_meta.get('n_violation', 0)} · "
                f"restrictions: {sem_meta.get('n_restriction', 0)} · "
                f"conformes: {sem_meta.get('n_compliant', 0)}"
            )
        elif mode == "semantic_strict_failed":
            st.warning(
                "Moteur UCP sémantique indisponible en mode strict. "
                "Le rapport est limité aux contrôles structurés."
            )

    def _finding(f, sev):
        art   = f.get("article", "")
        msg   = f.get("message", "")[:100]
        a_title = f.get("article_title", "")[:50]
        clause  = f.get("clause_text", "")[:150]
        rec     = f.get("recommendation", "")

        icon_map = {"error": "🔴", "warning": "🟡", "info": "ℹ️"}
        cls_map  = {"error": "finding-error", "warning": "finding-warning", "info": "finding-info"}

        st.markdown(f"""
        <div class="finding-row {cls_map.get(sev,'finding-info')}" style="flex-direction:column;gap:4px">
          <div><strong>[{art}]</strong> {msg}</div>
          {f'<div style="font-size:11px;color:#94a3b8">{a_title}</div>' if a_title else ""}
          {f'<div style="font-size:11px;background:#fff8;padding:3px 6px;border-radius:4px;margin-top:2px">📌 {clause}</div>' if clause else ""}
          {f'<div style="font-size:11px;color:#059669;margin-top:2px">💡 {rec}</div>' if rec else ""}
        </div>
        """, unsafe_allow_html=True)

    if errors:
        st.markdown(f'<div class="sec-header">🔴 Erreurs bloquantes ({len(errors)})</div>', unsafe_allow_html=True)
        for f in errors: _finding(f, "error")

    if warnings:
        st.markdown(f'<div class="sec-header">🟡 Alertes ({len(warnings)})</div>', unsafe_allow_html=True)
        for f in warnings: _finding(f, "warning")

    if infos:
        with st.expander(f"ℹ️ Informations ({len(infos)})", expanded=False):
            for f in infos: _finding(f, "info")


# ══════════════════════════════════════════════════════════════════════
# BLOC F — COHÉRENCE SÉMANTIQUE
# ══════════════════════════════════════════════════════════════════════

def show_coherence_tab(data: dict):
    swift = _get_swift_payload(data)
    nlp   = swift.get("nlp", {}) or {}
    sem   = nlp.get("semantic", {}) or {}

    if not sem:
        st.info("Analyse sémantique non disponible.")
        return

    comp   = sem.get("completeness_score", 0)
    coh    = sem.get("semantic_score", 0)
    doc_q  = sem.get("doc_quality_score", 0)
    summ   = sem.get("summary", "")
    risks  = _coerce_list(sem.get("risk_flags"))
    fraud  = _coerce_list(sem.get("fraud_indicators"))
    warns  = _coerce_list(sem.get("warnings"))
    recs   = _coerce_list(sem.get("recommendations"))
    geo    = sem.get("geo_coherence", {})
    coh_sc = sem.get("coherence_scores", {}) or {}

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Complétude",    f"{score_emoji(comp)} {comp}%")
    with c2: st.metric("Cohérence NLP", f"{score_emoji(coh)} {coh}%")
    with c3: st.metric("Qualité docs",  f"{score_emoji(doc_q)} {doc_q}%")

    if summ:
        st.info(f"📝 {summ}")

    for s_name, s_val in [("Complétude", comp), ("Cohérence", coh), ("Qualité", doc_q)]:
        sc = score_color(s_val)
        st.markdown(f"""
        <div class="score-bar-wrap">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:#94a3b8;margin-bottom:3px">
            <span>{s_name}</span><span>{s_val}%</span>
          </div>
          <div class="score-bar-bg"><div class="score-bar-fill" style="width:{s_val}%;background:{sc}"></div></div>
        </div>
        """, unsafe_allow_html=True)

    if coh_sc:
        st.markdown('<div class="sec-header">Scores de cohérence détaillés</div>', unsafe_allow_html=True)
        cols = st.columns(min(4, len(coh_sc)))
        for i, (k, v) in enumerate(coh_sc.items()):
            pct = int(v * 100) if v <= 1 else int(v)
            with cols[i % 4]:
                st.metric(k.replace("_", " ").title()[:22], f"{pct}%")

    if risks:
        st.markdown(f'<div class="sec-header">🚨 Risk Flags ({len(risks)})</div>', unsafe_allow_html=True)
        for r in risks:
            if isinstance(r, dict):
                sev = r.get("severity", "low")
                code = r.get("code", "")
                desc = r.get("description", "")
                ucp = r.get("ucp_ref", "")
            else:
                sev = "warning"
                code = ""
                desc = str(r)
                ucp = ""
            cls  = {"high": "finding-error", "medium": "finding-warning"}.get(sev, "finding-info")
            icon = {"high": "🔴", "medium": "🟡"}.get(sev, "🔵")
            st.markdown(f"""
            <div class="finding-row {cls}">
              <div>{icon} <strong>[{code}]</strong> {desc} {f"<em>({ucp})</em>" if ucp else ""}</div>
            </div>
            """, unsafe_allow_html=True)

    if fraud:
        st.markdown(f'<div class="sec-header">🔒 Indicateurs de fraude ({len(fraud)})</div>', unsafe_allow_html=True)
        for fi in fraud:
            if isinstance(fi, dict):
                sev = fi.get("severity", "warning")
                code = fi.get("code", "")
                desc = fi.get("description", "")
            else:
                sev = "warning"
                code = ""
                desc = str(fi)
            cls  = "finding-error" if sev == "high" else "finding-warning"
            st.markdown(f'<div class="finding-row {cls}">🔴 <strong>[{code}]</strong> {desc}</div>',
                        unsafe_allow_html=True)

    if warns:
        st.markdown(f'<div class="sec-header">⚠️ Avertissements UCP ({len(warns)})</div>', unsafe_allow_html=True)
        for w in warns:
            if isinstance(w, dict):
                sev = w.get("severity", "info")
                code = w.get("code", "")
                msg = w.get("message", "")
                sugg = w.get("suggestion", "")
                ucp = w.get("ucp_ref", "")
            else:
                sev = "warning"
                code = ""
                msg = str(w)
                sugg = ""
                ucp = ""
            cls  = {"error": "finding-error", "warning": "finding-warning"}.get(sev, "finding-info")
            st.markdown(f"""
            <div class="finding-row {cls}" style="flex-direction:column;gap:2px">
              <div><strong>[{code}]</strong> {msg} {f"<em>{ucp}</em>" if ucp else ""}</div>
              {f'<div style="font-size:11px;color:#059669">💡 {sugg}</div>' if sugg else ""}
            </div>
            """, unsafe_allow_html=True)

    if recs:
        st.markdown(f'<div class="sec-header">💡 Recommandations ({len(recs)})</div>', unsafe_allow_html=True)
        for i, rec in enumerate(recs, 1):
            st.markdown(f'<div class="finding-row finding-info"><strong>{i}.</strong> {str(rec)}</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# BLOC G — VALIDATION EXTERNE
# ══════════════════════════════════════════════════════════════════════

def show_external_tab(data: dict):
    ext = data.get("external_validation", {}) or {}
    ds  = data.get("ds_analysis", {}) or {}

    if not ext and not ds:
        st.info("Validation externe non disponible.")
        return

    # DS Risk
    if ds:
        risk     = ds.get("risk", {}) or {}
        risk_lvl = risk.get("risk_level", "?")
        risk_sc  = risk.get("risk_score", 0)
        factors  = risk.get("risk_factors", []) or []
        model    = risk.get("model_used", "rules")

        LEVEL_CFG = {
            "LOW":      ("#22c55e", "🟢"),
            "MEDIUM":   ("#f59e0b", "🟡"),
            "HIGH":     ("#ef4444", "🔴"),
            "CRITICAL": ("#7b0000", "🚨"),
        }
        color, icon = LEVEL_CFG.get(risk_lvl, ("#94a3b8", "?"))

        st.markdown(f'<div class="sec-header">🤖 Analyse Data Science (DS Risk)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Niveau DS",    f"{icon} {risk_lvl}")
        with c2: st.metric("Risk Score",   f"{int(risk_sc*100)}/100")
        with c3: st.metric("Modèle",       "🤖 ML" if model != "rules" else "📏 Règles")

        if factors:
            for f in factors:
                st.markdown(f'<div class="finding-row finding-warning">• {f}</div>', unsafe_allow_html=True)

    if not ext:
        return

    risk_level = ext.get("risk_level", "CLEAN")
    risk_score = ext.get("risk_score", 0.0)
    sources    = ext.get("sources_consulted", []) or []
    factors    = ext.get("risk_factors", []) or []
    recs       = ext.get("recommendations", []) or []
    details    = ext.get("details", {}) or {}
    ext_ms     = ext.get("processing_ms", 0)

    icon  = EXT_ICONS.get(risk_level, "?")
    color = EXT_COLORS.get(risk_level, "#94a3b8")

    st.markdown('<div class="sec-header">🌐 Validation Externe (OFAC · GLEIF · EU · UN · OpenCorp)</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Niveau ext.",     f"{icon} {risk_level}")
    with c2: st.metric("Score pondéré",   f"{risk_score:.2f}")
    with c3: st.metric("Sources",         len(sources))
    with c4: st.metric("Temps",           f"{ext_ms}ms")

    if sources:
        st.caption("Sources : " + "  ·  ".join(f"`{s}`" for s in sources))

    # Pays
    countries = details.get("countries", [])
    if countries:
        st.markdown('<div class="sec-header">🌍 Pays</div>', unsafe_allow_html=True)
        rows = []
        for c in countries:
            lvl  = c.get("risk", "CLEAN")
            rows.append({
                "ISO2":    c.get("iso2") or c.get("country_iso2", "?"),
                "Pays":    c.get("normalized", c.get("raw", "?")),
                "Risque":  f"{EXT_ICONS.get(lvl,'?')} {lvl}",
                "Détail":  c.get("details", "—")[:60],
                "Sources": " · ".join(c.get("sources", [])),
            })
        st.dataframe(rows, hide_index=True, width="stretch")

    # Banques
    banks = details.get("banks", [])
    if banks:
        st.markdown('<div class="sec-header">🏦 Banques (GLEIF)</div>', unsafe_allow_html=True)
        rows = []
        for b in banks:
            lvl = b.get("risk", "CLEAN")
            rows.append({
                "BIC":        b.get("bic", "?"),
                "Nom GLEIF":  b.get("gleif_name", "—")[:40],
                "Statut":     b.get("gleif_status", "?"),
                "Blacklisté": "🚨 OUI" if b.get("blacklisted") else "✅ NON",
                "Pays":       b.get("country", "?"),
                "Risque":     f"{EXT_ICONS.get(lvl,'?')} {lvl}",
            })
        st.dataframe(rows, hide_index=True, width="stretch")

    # Entités
    companies = details.get("companies", [])
    if companies:
        st.markdown('<div class="sec-header">🏢 Entités (OFAC · EU · UN · OpenCorp)</div>',
                    unsafe_allow_html=True)
        for co in companies:
            lvl     = co.get("risk", "CLEAN")
            co_icon = EXT_ICONS.get(lvl, "?")
            co_col  = EXT_COLORS.get(lvl, "#94a3b8")
            name    = co.get("name", "?")[:60]
            flags   = []
            if co.get("ofac_matched"): flags.append("OFAC")
            if co.get("eu_matched"):   flags.append("EU")
            if co.get("un_matched"):   flags.append("UN")
            flags_str = " ".join(f"`{f}`" for f in flags)
            cls = "finding-error" if lvl in ("HIGH", "CRITICAL") else ("finding-warning" if lvl == "MEDIUM" else "finding-ok")
            st.markdown(f"""
            <div class="finding-row {cls}">
              <div>
                <strong>{co_icon} {name}</strong>
                {f" — {flags_str}" if flags_str else ""}
                <span style="color:{co_col};margin-left:8px">{lvl}</span>
                {f'<br><small style="color:#94a3b8">{co.get("details","")[:100]}</small>' if co.get("details") else ""}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Facteurs & recs
    if factors and not all("Aucun risque" in f for f in factors):
        st.markdown(f'<div class="sec-header">⚠️ Facteurs de risque ({len(factors)})</div>',
                    unsafe_allow_html=True)
        for f in factors:
            cls = "finding-error" if "BLOQUER" in f else "finding-warning"
            st.markdown(f'<div class="finding-row {cls}">• {f}</div>', unsafe_allow_html=True)

    if recs:
        st.markdown(f'<div class="sec-header">📋 Recommandations ({len(recs)})</div>',
                    unsafe_allow_html=True)
        for rec in recs:
            cls = "finding-error" if rec.startswith("⛔") else ("finding-warning" if rec.startswith("⚠") else "finding-info")
            st.markdown(f'<div class="finding-row {cls}">{rec}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# BLOC H — RISK ML
# ══════════════════════════════════════════════════════════════════════

def show_risk_ml_tab(data: dict):
    risk_ml = data.get("risk_ml", {}) or {}

    if not risk_ml:
        st.info("Risk ML non disponible.")
        return

    if risk_ml.get("error"):
        st.warning(f"⚠️ Risk ML indisponible : {risk_ml['error']}")
        st.caption("Lancer `python scripts/train_models.py` pour entraîner les modèles.")
        return

    risk_label = risk_ml.get("risk_label", "UNKNOWN")
    risk_score = risk_ml.get("risk_score", 0)
    risk_proba = risk_ml.get("risk_proba", {}) or {}
    is_anomaly = risk_ml.get("is_anomaly", False)
    ano_score  = risk_ml.get("anomaly_score", 0.0)
    segment    = risk_ml.get("segment", "?")
    shap_top   = risk_ml.get("shap_top_features", []) or []
    features   = risk_ml.get("features_used", {}) or {}
    version    = risk_ml.get("model_version", "?")
    ms         = risk_ml.get("processing_ms", 0)

    gauge_color = "#22c55e" if risk_score < 35 else ("#f59e0b" if risk_score < 65 else "#ef4444")
    badge_colors = {"LOW": ("#f0fdf4", "#065f46"), "MEDIUM": ("#fffbeb", "#92400e"), "HIGH": ("#fff1f2", "#991b1b")}
    bg, fg = badge_colors.get(risk_label, ("#f1f5f9", "#1e293b"))

    col_g, col_p, col_d = st.columns([1.2, 2, 2])

    with col_g:
        st.markdown(f"""
        <div style="text-align:center;padding:16px 0">
          <div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#94a3b8;margin-bottom:8px">Risk Score ML</div>
          <div style="font-size:64px;font-weight:800;color:{gauge_color};line-height:1;font-family:'IBM Plex Mono',monospace">{risk_score}</div>
          <div style="font-size:11px;color:#94a3b8">/100</div>
          <div style="display:inline-block;margin-top:10px;padding:5px 18px;border-radius:20px;
                      background:{bg};color:{fg};font-weight:700;font-size:13px;letter-spacing:1px">
            {risk_label}
          </div>
          <div style="font-size:10px;color:#94a3b8;margin-top:8px">v{version} · {ms}ms</div>
        </div>
        """, unsafe_allow_html=True)

    with col_p:
        st.markdown('<div class="sec-header">Probabilités</div>', unsafe_allow_html=True)
        PROBA_COLORS = {"LOW": "#22c55e", "MEDIUM": "#f59e0b", "HIGH": "#ef4444"}
        for lbl in ["LOW", "MEDIUM", "HIGH"]:
            prob = risk_proba.get(lbl, 0.0)
            pct  = int(prob * 100)
            col  = PROBA_COLORS[lbl]
            st.markdown(f"""
            <div style="margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
                <span style="font-weight:600;color:{col}">{lbl}</span>
                <span style="color:#94a3b8;font-family:'IBM Plex Mono',monospace">{prob:.1%}</span>
              </div>
              <div style="background:#f1f5f9;border-radius:6px;height:12px;overflow:hidden">
                <div style="width:{pct}%;height:12px;background:{col};border-radius:6px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="sec-header">Détection & Segment</div>', unsafe_allow_html=True)
        if is_anomaly:
            st.markdown(f"""
            <div style="background:#fff1f2;border:1px solid #fda4af;border-radius:8px;padding:12px 16px;margin-bottom:10px">
              <div style="font-weight:700;color:#991b1b">⚠️ Anomalie détectée</div>
              <div style="font-size:12px;color:#94a3b8;margin-top:2px">Score : {ano_score:.4f}</div>
              <div style="font-size:11px;color:#6b7280;margin-top:4px">IsolationForest — Top 10% suspects</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:12px 16px;margin-bottom:10px">
              <div style="font-weight:700;color:#065f46">✅ Comportement normal</div>
              <div style="font-size:12px;color:#94a3b8;margin-top:2px">Score : {ano_score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        SEG_ICONS = {"LOW_RISK_TRADE": "🟢", "STANDARD_LC": "🔵", "HIGH_RISK_LC": "🔴", "COMPLEX_LC": "🟠"}
        seg_icon = SEG_ICONS.get(segment, "🔷")
        st.markdown(f"""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:12px 16px">
          <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px">Segment KMeans</div>
          <div style="font-size:16px;font-weight:700;color:#1e40af;margin-top:4px">{seg_icon} {segment}</div>
        </div>
        """, unsafe_allow_html=True)

    if shap_top:
        st.markdown('<div class="sec-header">🔍 SHAP — Top features explicatives</div>', unsafe_allow_html=True)
        max_shap = max(abs(f["shap_value"]) for f in shap_top) or 1
        for feat_info in shap_top:
            feat  = feat_info["feature"]
            val   = feat_info["shap_value"]
            direc = feat_info["direction"]
            pct   = int(abs(val) / max_shap * 100)
            is_pos = val > 0
            bar_color = "#ef4444" if is_pos else "#22c55e"
            dir_color = "#ef4444" if is_pos else "#22c55e"
            st.markdown(f"""
            <div class="shap-row">
              <div class="shap-feat">{feat}</div>
              <div class="shap-bar-bg">
                <div class="shap-bar-fill" style="width:{pct}%;background:{bar_color}"></div>
              </div>
              <div class="shap-dir" style="color:{dir_color}">{direc}</div>
              <div class="shap-num">{val:+.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    if features:
        with st.expander("📋 Features utilisées (26 variables)", expanded=False):
            FEATURE_COLS = [
                "validity_days", "presentation_days", "days_to_shipment",
                "lc_amount", "tolerance_plus", "tolerance_minus", "currency_risk",
                "n_documents_required", "n_clauses", "n_high_risk_clauses",
                "n_critical_clauses", "n_compliance_clauses", "n_ucp_exclusions",
                "n_ucp_errors", "n_ucp_warnings", "transshipment_allowed",
                "partial_ship_allowed", "confirmation_required", "charges_beneficiary",
                "incoterm_risk", "country_risk", "completeness_score", "semantic_score",
                "nlp_score", "ml_risk_score", "ucp_compliance_score",
            ]
            rows = []
            for feat in FEATURE_COLS:
                v = features.get(feat, "—")
                rows.append({
                    "Feature": feat,
                    "Valeur":  f"{v:.4f}" if isinstance(v, float) else str(v),
                })
            st.dataframe(rows, hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════
# BLOC I — JSON EXPLORER
# ══════════════════════════════════════════════════════════════════════

def _count_keys_recursive(obj, depth: int = 0) -> tuple[int, int]:
    """Return (total_leaf_or_key_count, max_depth)."""
    if isinstance(obj, dict):
        total, max_d = len(obj), depth
        for v in obj.values():
            c, d = _count_keys_recursive(v, depth + 1)
            total += c
            max_d = max(max_d, d)
        return total, max_d
    if isinstance(obj, list):
        total, max_d = 0, depth
        for item in obj[:20]:
            c, d = _count_keys_recursive(item, depth)
            total += c
            max_d = max(max_d, d)
        return total, max_d
    return 0, depth


def _json_matches(obj, q: str) -> bool:
    """Recursive search: True if q appears in any key or value."""
    q = q.lower()
    if isinstance(obj, dict):
        return any(q in str(k).lower() or _json_matches(v, q) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_json_matches(i, q) for i in obj)
    return q in str(obj).lower()


_JSON_SECTION_META: dict[str, tuple[str, str, str]] = {
    "swift":               ("🔑", "SWIFT / LC Data",        "#3b82f6"),
    "ocr":                 ("🔍", "OCR",                    "#8b5cf6"),
    "ucp_compliance":      ("⚖️",  "UCP 600 Compliance",    "#f59e0b"),
    "ds_analysis":         ("🤖", "Data Science",           "#10b981"),
    "external_validation": ("🌐", "Validation Externe",     "#ef4444"),
    "risk_ml":             ("⚡", "Risk ML / SHAP",          "#f97316"),
    "success":             ("✅", "Statut",                  "#22c55e"),
    "processing_time_s":   ("⏱️",  "Temps de traitement",   "#94a3b8"),
}


def show_json_explorer(data: dict):
    """Interactive JSON explorer: section navigator + search + stats."""

    # ── Stats ──────────────────────────────────────────────────────────
    total_keys, max_depth = _count_keys_recursive(data)
    json_str  = json.dumps(data, ensure_ascii=False, indent=2)
    json_size = len(json_str.encode("utf-8"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sections",     len(data))
    c2.metric("Clés totales", total_keys)
    c3.metric("Profondeur",   max_depth)
    c4.metric("Taille",       f"{json_size / 1024:.1f} Ko")
    c5.metric("Lignes",       json_str.count("\n"))

    st.divider()

    # ── Toolbar: download + search + view mode ─────────────────────────
    col_dl, col_search, col_mode = st.columns([2, 4, 2])
    with col_dl:
        st.download_button(
            "⬇️ Télécharger JSON",
            data=json_str,
            file_name="swift_result.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_search:
        search = st.text_input(
            "🔍 Filtrer",
            placeholder="Clé ou valeur — ex: risk_level, entities…",
            key="json_search_filter",
            label_visibility="collapsed",
        )
    with col_mode:
        view_mode = st.radio(
            "Vue", ["Par section", "JSON complet"],
            horizontal=True, key="json_view_mode",
            label_visibility="collapsed",
        )

    st.divider()

    # ── Full JSON view ─────────────────────────────────────────────────
    if view_mode == "JSON complet":
        if search:
            # Rebuild a filtered dict for display only
            filtered = {k: v for k, v in data.items()
                        if search.lower() in k.lower() or _json_matches(v, search)}
            if filtered:
                st.caption(f"{len(filtered)} section(s) correspondent à « {search} »")
                st.json(filtered)
            else:
                st.info(f"Aucune section ne correspond à « {search} »")
        else:
            st.json(data)
        return

    # ── Section navigator ──────────────────────────────────────────────
    shown = 0
    for key, section_data in data.items():
        if search and not (search.lower() in key.lower() or _json_matches(section_data, search)):
            continue

        shown += 1
        icon, label, color = _JSON_SECTION_META.get(
            key, ("📦", key.replace("_", " ").title(), "#64748b")
        )

        if isinstance(section_data, dict):
            n_info = f"{len(section_data)} clés"
        elif isinstance(section_data, list):
            n_info = f"{len(section_data)} éléments"
        elif isinstance(section_data, bool):
            n_info = "bool"
        elif isinstance(section_data, (int, float)):
            n_info = "nombre"
        else:
            n_info = f"string · {len(str(section_data))} car."

        expanded = (key == "swift") and not search

        with st.expander(
            f"{icon} **{label}** — `{key}` · *{n_info}*",
            expanded=expanded,
        ):
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;'
                f'padding:8px 12px;background:{color}18;border-radius:8px;'
                f'border-left:3px solid {color}">'
                f'<span style="font-size:20px">{icon}</span>'
                f'<span style="font-weight:700;color:{color};font-size:14px">{label}</span>'
                f'<code style="margin-left:auto;font-size:11px;color:#64748b;'
                f'background:#f1f5f9;padding:2px 8px;border-radius:4px">{key}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if section_data is None or isinstance(section_data, (bool, int, float, str)):
                st.code(json.dumps(section_data, ensure_ascii=False), language="json")
            else:
                # If search active, highlight matching sub-keys in a pre-filtered view
                if search and isinstance(section_data, dict):
                    filtered_section = {
                        k: v for k, v in section_data.items()
                        if search.lower() in str(k).lower() or _json_matches(v, search)
                    }
                    if filtered_section:
                        st.caption(f"Résultats filtrés ({len(filtered_section)}/{len(section_data)} clés)")
                        st.json(filtered_section)
                        with st.expander("Voir la section complète", expanded=False):
                            st.json(section_data)
                    else:
                        st.json(section_data)
                else:
                    st.json(section_data)

    if shown == 0:
        st.info(f"Aucune section ne correspond à « {search} »")


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR — show_result
# ══════════════════════════════════════════════════════════════════════

def show_result(data: dict):
    """Affiche le résultat complet du pipeline — v8."""
    swift = _get_swift_payload(data)
    nlp   = swift.get("nlp", {}) or {}

    # HERO DASHBOARD
    show_global_dashboard(data)

    if not swift.get("success"):
        st.error(f"❌ {swift.get('error', 'Erreur pipeline')}")
        return

    # ONGLETS DE NAVIGATION
    tabs = st.tabs([
        "📋 Champs SWIFT",
        "📌 Clauses",
        "✅ Validation champs",
        "⚖️ UCP 600",
        "🔬 Cohérence",
        "🌐 Externe & DS",
        "⚡ Risk ML",
        "🗃️ JSON Explorer",
    ])

    with tabs[0]:
        show_swift_fields_tab(data)

    with tabs[1]:
        show_clauses_tab(data)

    with tabs[2]:
        show_validation_tab(data)

    with tabs[3]:
        show_ucp_tab(data)

    with tabs[4]:
        show_coherence_tab(data)

    with tabs[5]:
        show_external_tab(data)

    with tabs[6]:
        show_risk_ml_tab(data)

    with tabs[7]:
        show_json_explorer(data)


# ══════════════════════════════════════════════════════════════════════
# NLP HELPERS (inchangés)
# ══════════════════════════════════════════════════════════════════════

def _show_spell(spell: dict):
    if not spell: return
    changes = spell.get("total_changes", 0)
    if changes == 0:
        st.success("✅ Aucune faute détectée.")
        return
    corrs = spell.get("corrections", [])
    if corrs:
        rows = []
        for c in corrs:
            rows.append({"Original": c.get("original",""), "→ Corrigé": c.get("corrected",""),
                         "Méthode": c.get("method",""), "Conf.": f"{c.get('confidence',1.0):.2f}"})
        st.dataframe(rows, hide_index=True, width="stretch")

def _show_clauses_simple(clauses, contras):
    if not clauses: return
    for c in clauses[:20]:
        risk = c.get("risk_level", "normal")
        bc   = {"high": "#ef4444", "medium": "#f59e0b"}.get(risk, "#3b82f6")
        st.markdown(
            f'<div style="border-left:3px solid {bc};padding:6px 12px;margin:4px 0;font-size:12px;background:#f8fafc;border-radius:0 6px 6px 0">{c.get("text","")[:200]}</div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚡ SWIFT Intelligence")

    try:
        h = requests.get(HEALTH_URL, timeout=2)
        if h.status_code == 200:
            hdata   = h.json()
            ml      = hdata.get("ml_models", {})
            ext_v   = hdata.get("external_validation", {})
            risk_ml_info = hdata.get("risk_ml_engine", {})
            clf_ok  = ml.get("clause_classifier", {}).get("loaded", False)
            risk_ok = ml.get("risk_model", {}).get("loaded", False)
            ext_ok  = ext_v.get("engine_status", "") == "ok"
            rml_ok  = risk_ml_info.get("status", "") == "ok"

            st.success("🟢 API connectée")
            st.caption(
                f"{'✅' if clf_ok else '❌'} Clauses  "
                f"{'✅' if risk_ok else '❌'} DS Risk  "
                f"{'✅' if ext_ok else '❌'} Ext.Val  "
                f"{'✅' if rml_ok else '❌'} Risk ML"
            )
        else:
            st.warning("⚠️ API — réponse inattendue")
    except Exception:
        st.error("🔴 API non joignable")

    st.divider()

    sidebar_mode = st.selectbox(
        "Mode d'analyse",
        ["Pipeline complet", "NLP seul (texte)", "Correction orthographique", "Classification clauses"],
    )

    raw_text = st.text_area("Texte SWIFT", height=160,
                             placeholder=":27: 1/1\n:40A: IRREVOCABLE\n:20: LC2026…")
    swift_type_sel = st.selectbox("Type MT", ["", "MT700", "MT720", "MT760", "MT790", "MT799"])

    if st.button("▶ Analyser", width="stretch", type="primary"):
        text = raw_text.strip()
        if not text:
            st.warning("Saisissez un texte SWIFT.")
        else:
            with st.spinner("Analyse…"):
                try:
                    if sidebar_mode == "Pipeline complet":
                        resp = requests.post(ANALYZE_URL, json={"text": text})
                    elif sidebar_mode == "NLP seul (texte)":
                        payload = {"text": text}
                        if swift_type_sel: payload["swift_type"] = swift_type_sel
                        resp = requests.post(NLP_URL, json=payload)
                    elif sidebar_mode == "Correction orthographique":
                        resp = requests.post(SPELL_URL, json={"text": text})
                    else:
                        resp = requests.post(CLASSIFY_URL, json={"text": text})

                    if resp.status_code == 200:
                        st.session_state["sidebar_result"]    = resp.json()
                        st.session_state["sidebar_mode_used"] = sidebar_mode
                    else:
                        st.error(f"Erreur {resp.status_code}")
                except Exception as ex:
                    st.error(f"Connexion échouée : {ex}")

    if "sidebar_result" in st.session_state:
        res  = st.session_state["sidebar_result"]
        mode = st.session_state.get("sidebar_mode_used", "")
        st.divider()
        st.caption("**Dernier résultat**")
        if mode == "Pipeline complet":
            swift_s = _get_swift_payload(res)
            nlp_s = (swift_s.get("nlp", {}) or {})
            val_s = (swift_s.get("validation", {}) or {})
            st.metric("Score NLP",  f"{nlp_s.get('score', 0)}/100")
            st.metric("Validation", f"{val_s.get('score', 0)}/100")
        elif mode == "NLP seul (texte)":
            nlp_s = res.get("nlp", {}) or {}
            st.metric("Entités", len((nlp_s.get("ner") or {}).get("entities", [])))
            st.metric("Clauses", len(nlp_s.get("clauses", [])))
        elif mode == "Correction orthographique":
            st.metric("Corrections", res.get("changes", 0))
        else:
            st.metric("Clauses", res.get("count", 0))


# ══════════════════════════════════════════════════════════════════════
# CORPS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">⚡ SWIFT Intelligence Platform</div>
  <div class="hero-subtitle">
    MT700 · MT720 · MT760 · MT790 · MT799 &nbsp;—&nbsp;
    UCP 600 · URDG 758 · ISP 98 &nbsp;—&nbsp;
    OCR · NLP · NER · Clauses · Validation · OFAC · GLEIF · EU · UN · XGBoost · SHAP
  </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📎 Déposez un fichier SWIFT (PDF ou image)",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
)

if uploaded_file:
    if st.button("🚀 Lancer le pipeline complet", width="stretch", type="primary"):
        with st.spinner("Pipeline — OCR · Règles · NLP · UCP · DS · Validation externe · Risk ML…"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(PIPELINE_URL, files=files, timeout=120)
                if resp.status_code == 200:
                    st.session_state["pipeline_result"] = resp.json()
                else:
                    st.error(f"Erreur API : {resp.status_code}")
            except Exception as ex:
                st.error(f"Connexion échouée : {ex}")

if "pipeline_result" in st.session_state:
    show_result(st.session_state["pipeline_result"])

elif "sidebar_result" in st.session_state:
    mode = st.session_state.get("sidebar_mode_used", "")
    res  = st.session_state["sidebar_result"]
    st.divider()
    st.subheader(f"Résultat — {mode}")

    if mode == "Pipeline complet":
        show_result(res)
    elif mode == "NLP seul (texte)":
        nlp = res.get("nlp")
        if nlp:
            _show_clauses_simple(nlp.get("clauses", []), nlp.get("contradictions", []))
    elif mode == "Correction orthographique":
        if res.get("success"):
            st.text_area("Texte corrigé", res.get("corrected", ""), height=200)
            _show_spell({
                "total_changes":     res.get("changes", 0),
                "language_detected": res.get("language_detected", ""),
                "avg_confidence":    res.get("avg_confidence", 0),
                "corrections":       res.get("corrections", []),
            })
    else:
        _show_clauses_simple(res.get("results", []), [])