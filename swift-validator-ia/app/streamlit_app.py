"""
streamlit_app.py  —  v7.3  (complet)
=======================================
Interface SWIFT Pipeline Hybride — avec Validation Externe + Risk ML

MODIFICATIONS v7.3 :
  ✦ RISK-UI-01 : show_risk_ml_section()  — section ⑧ Risk ML
    • Gauge de risque animée (CSS)
    • Probabilités LOW/MEDIUM/HIGH en barres de progression
    • Anomaly Detection badge
    • Segment KMeans
    • SHAP top features (barre horizontale colorée)
    • Features utilisées (tableau compact)
  ✦ RISK-UI-02 : show_dashboard() — metric "ML Risk" dans le bandeau
  ✦ RISK-UI-03 : show_result()    — appel de show_risk_ml_section()
  ✦ RISK-UI-04 : sidebar /health  — statut risk_ml_engine
  ✦ RISK-UI-05 : URL /risk/health exposée dans la sidebar

MODIFICATIONS v7.2 :
  ✦ Correction StreamlitAPIException : expanders imbriqués supprimés
    - show_entities_section() : LONG fields → st.text_area disabled
    - show_entities_section() : raw_fields → st.json direct
    - _show_clauses() : contradictions → st.container au lieu d'expander
    - _show_semantic() : warnings → st.container au lieu d'expander dans expander
    - show_validation_section() : fields tables → st.container
    - show_ucp_section() : findings → st.container
    - show_ds_section() : features → st.container
    - show_external_section() : factors/recs → st.container

MODIFICATIONS v7.1 :
  ✦ show_external_section() — section ⑦ affichant le rapport external_validation
  ✦ show_result() — appel de show_external_section() après show_ds_section()
  ✦ show_dashboard() — metric "Ext. Risk" dans le bandeau si external_validation présent
"""

import json
import requests
import streamlit as st
from collections import Counter

API_BASE     = "http://localhost:5000"
PIPELINE_URL = f"{API_BASE}/pipeline"
ANALYZE_URL  = f"{API_BASE}/swift/analyze"
NLP_URL      = f"{API_BASE}/swift/nlp"
SPELL_URL    = f"{API_BASE}/swift/spell"
CLASSIFY_URL = f"{API_BASE}/swift/classify"
HEALTH_URL   = f"{API_BASE}/health"
RISK_HEALTH_URL = f"{API_BASE}/risk/health"


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SWIFT Pipeline v7.3",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .metric-card {
    background: #f8f9fa; border-radius: 8px; padding: 12px 16px;
    border: 1px solid #e9ecef; text-align: center;
  }
  .metric-card .label {
    font-size: 11px; color: #6c757d; text-transform: uppercase;
    letter-spacing: .5px; margin-bottom: 4px;
  }
  .metric-card .value { font-size: 22px; font-weight: 600; color: #212529; }
  .clause-card {
    border-left: 3px solid #dee2e6; padding: 8px 12px; margin-bottom: 6px;
    background: #fdfdfd; border-radius: 0 6px 6px 0;
  }
  .clause-card.high   { border-left-color: #dc3545; }
  .clause-card.medium { border-left-color: #fd7e14; }
  .clause-card.normal { border-left-color: #0d6efd; }
  .entity-row { font-size: 13px; }
  .tag-pill {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 500; margin-right: 4px;
  }
  .section-header {
    font-size: 15px; font-weight: 600; color: #495057;
    border-bottom: 2px solid #e9ecef; padding-bottom: 4px; margin-bottom: 12px;
  }
  div[data-testid="stExpander"] > div:first-child { font-weight: 500; }
  .ext-row {
    padding: 5px 10px; margin: 3px 0; background: #f8f9fa;
    border-radius: 4px; font-size: 13px;
  }
  .inner-block {
    background: #f8f9fa; border-radius: 6px; padding: 10px 14px;
    margin: 6px 0; border: 1px solid #e9ecef;
  }
  .long-field-label {
    font-size: 11px; color: #6c757d; text-transform: uppercase;
    letter-spacing: .4px; margin-bottom: 2px; margin-top: 10px;
  }

  /* RISK-UI-01 : Risk Gauge */
  .risk-gauge-wrap { text-align: center; padding: 10px 0; }
  .risk-gauge-score {
    font-size: 52px; font-weight: 700; line-height: 1;
    background: linear-gradient(135deg, #198754, #fd7e14, #dc3545);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .risk-label-badge {
    display: inline-block; padding: 4px 20px; border-radius: 20px;
    font-size: 14px; font-weight: 600; letter-spacing: 1px;
    margin-top: 6px;
  }
  .risk-badge-LOW    { background: #d1fae5; color: #065f46; }
  .risk-badge-MEDIUM { background: #fef3c7; color: #92400e; }
  .risk-badge-HIGH   { background: #fee2e2; color: #991b1b; }
  .risk-badge-UNKNOWN{ background: #e5e7eb; color: #374151; }

  .shap-bar-wrap { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  .shap-bar-label { font-size: 12px; color: #374151; width: 180px; flex-shrink: 0; }
  .shap-bar-outer { flex: 1; background: #e5e7eb; border-radius: 4px; height: 10px; }
  .shap-bar-inner-pos { height: 10px; border-radius: 4px; background: #dc3545; }
  .shap-bar-inner-neg { height: 10px; border-radius: 4px; background: #198754; }
  .shap-bar-val { font-size: 11px; color: #6b7280; width: 60px; text-align: right; }

  .anomaly-badge-yes {
    background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5;
    border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 600;
  }
  .anomaly-badge-no {
    background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7;
    border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 600;
  }
  .segment-badge {
    background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe;
    border-radius: 6px; padding: 6px 14px; font-size: 13px; font-weight: 600;
  }
  .feat-table { font-size: 12px; width: 100%; border-collapse: collapse; }
  .feat-table th {
    background: #f3f4f6; text-align: left; padding: 4px 8px;
    font-size: 11px; color: #6b7280; font-weight: 600; text-transform: uppercase;
  }
  .feat-table td { padding: 4px 8px; border-bottom: 1px solid #f3f4f6; }
  .feat-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def score_color(s: int) -> str:
    if s >= 80: return "#198754"
    if s >= 50: return "#fd7e14"
    return "#dc3545"

def score_emoji(s: int) -> str:
    if s >= 80: return "🟢"
    if s >= 50: return "🟡"
    return "🔴"

def conf_badge(c: float) -> str:
    if c >= 0.85: return "🟢"
    if c >= 0.60: return "🟡"
    return "🔴"

def risk_color(sev: str) -> str:
    return {"high": "#dc3545", "medium": "#fd7e14", "low": "#0dcaf0"}.get(sev, "#6c757d")

def method_badge(method: str) -> str:
    if "supervised" in method: return "🤖 supervisé"
    if "embedding"  in method: return "🤖 embedding"
    if "rule"       in method: return "📏 règle"
    if "scoring"    in method: return "📊 scoring"
    if "fallback"   in method: return "⬇ fallback"
    return f"• {method}"

def importance_badge(imp: str) -> str:
    return {"critical": "🔴 critique", "high": "🟠 haute",
            "medium": "🟡 moyenne", "low": "🔵 faible"}.get(imp, imp)

def clause_type_label(t: str) -> str:
    return {
        "document_condition":   "📄 Document",
        "shipment_condition":   "🚢 Expédition",
        "payment_condition":    "💰 Paiement",
        "compliance_condition": "⚖️ Conformité",
        "tolerance_condition":  "📏 Tolérance",
    }.get(t, f"📌 {t}")

def ner_icon(label: str) -> str:
    return {
        "APPLICANT": "🏢", "BENEFICIARY": "🏢", "ISSUING_BANK": "🏦",
        "ADVISING_BANK": "🏦", "AMOUNT": "💰", "GOODS": "📦",
        "INCOTERM": "🚢", "EXPIRY_DATE": "📅", "ISSUE_DATE": "📅",
        "SHIPMENT_DATE": "📅", "LC_NUMBER": "🔑", "PORT": "⚓",
        "COUNTRY": "🌍",
    }.get(label, "🔷")

def _get_swift_payload(data: dict) -> dict:
    """Compat: supporte payload pipeline (avec `swift`) et payload direct."""
    swift = (data or {}).get("swift")
    if isinstance(swift, dict):
        return swift
    return data or {}

EXT_ICONS = {
    "CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵", "CLEAN": "🟢",
}
EXT_COLORS = {
    "CRITICAL": "#7b0000", "HIGH": "#dc3545",
    "MEDIUM": "#fd7e14", "LOW": "#0dcaf0", "CLEAN": "#198754",
}


# ══════════════════════════════════════════════════════════════════════
# BLOC 0 — DASHBOARD SYNTHÈSE  (v7.3 : + ML Risk metric)
# ══════════════════════════════════════════════════════════════════════

def show_dashboard(data: dict):
    swift   = data.get("swift", {}) or {}
    nlp     = swift.get("nlp", {}) or {}
    val     = swift.get("validation", {}) or {}
    ocr     = data.get("ocr", {}) or {}
    sem     = nlp.get("semantic", {}) or {}
    timings = nlp.get("module_timings", {}) or {}
    risk_ml = data.get("risk_ml", {}) or {}

    nlp_score    = nlp.get("score", 0)
    val_score    = val.get("overall_valid", False)
    val_num      = val.get("score", 0)
    ocr_conf     = ocr.get("confidence", 0)
    swift_type   = swift.get("swift_type", "—")
    total_ms     = nlp.get("processing_ms", 0)
    completeness = sem.get("completeness_score", 0)
    n_clauses    = len(nlp.get("clauses", []))
    n_entities   = len((nlp.get("ner") or {}).get("entities", []))
    from_cache   = nlp.get("from_cache", False)

    st.markdown("---")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)

    with c1: st.metric("Type SWIFT", swift_type)
    with c2: st.metric("OCR", f"{ocr_conf*100:.0f}%")
    with c3: st.metric("Score NLP", f"{score_emoji(nlp_score)} {nlp_score}/100")
    with c4: st.metric("Validation", "✅ Valide" if val_score else "❌ Invalide",
                        delta=f"{val_num}/100")
    with c5: st.metric("Complétude", f"{score_emoji(completeness)} {completeness}%")
    with c6: st.metric("Entités / Cl.", f"{n_entities} / {n_clauses}")

    with c7:
        if risk_ml and not risk_ml.get("error"):
            ml_label = risk_ml.get("risk_label", "?")
            ml_score = risk_ml.get("risk_score", 0)
            icons    = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            st.metric("🤖 ML Risk", f"{icons.get(ml_label,'?')} {ml_label}",
                      delta=f"score {ml_score}/100")
        else:
            st.metric("🤖 ML Risk", "—")

    with c8:
        cache_str = "✅ cache" if from_cache else f"⏱ {total_ms}ms"
        st.metric("Temps NLP", cache_str)

    if timings:
        parts = []
        for mod, ms_val in timings.items():
            icon = {"spell": "🔤", "ner": "🔷", "clauses": "📌", "semantic": "🔬"}.get(mod, "•")
            parts.append(f"{icon} `{mod}` **{ms_val}ms**")
        st.caption("Modules : " + "  ·  ".join(parts))

    ucp = data.get("ucp_compliance")
    ds  = data.get("ds_analysis")
    ext = data.get("external_validation")

    band_cols_data = []
    if ucp:
        band_cols_data.append(("⚖️ UCP Score",
                                f"{ucp.get('global_score', 0)}/100",
                                "✅" if ucp.get("compliant") else "⚠️"))
    if ds and ds.get("risk"):
        risk = ds["risk"]
        LEVEL_ICONS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "🚨"}
        level = risk.get("risk_level", "?")
        band_cols_data.append(("🤖 DS Risk",
                                f"{LEVEL_ICONS.get(level,'?')} {level}",
                                f"score {risk.get('risk_score',0):.2f}"))
    if ext:
        ext_level = ext.get("risk_level", "CLEAN")
        ext_score = ext.get("risk_score", 0)
        ext_icon  = EXT_ICONS.get(ext_level, "?")
        band_cols_data.append(("🌐 Ext. Risk", f"{ext_icon} {ext_level}", f"score {ext_score:.2f}"))

    if risk_ml and not risk_ml.get("error"):
        ml_seg = risk_ml.get("segment", "?")
        ml_ano = "⚠️ Anomalie" if risk_ml.get("is_anomaly") else "✅ Normal"
        band_cols_data.append(("🔬 ML Segment", ml_seg, ml_ano))

    if band_cols_data:
        cols_band = st.columns(len(band_cols_data) + 4)
        for i, (label, val_str, sub) in enumerate(band_cols_data):
            with cols_band[i]:
                st.metric(label, val_str, delta=sub)

    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — ENTITÉS EXTRAITES
# ══════════════════════════════════════════════════════════════════════

def show_entities_section(entities: dict, nlp_ner: dict = None):
    with st.expander("**① Entités extraites**", expanded=True):
        if not entities:
            st.info("Aucune entité disponible.")
            return

        parties = [
            ("applicant",    "Donneur d'ordre",  "🏢"),
            ("beneficiary",  "Bénéficiaire",     "🏢"),
            ("issuing_bank", "Banque émettrice",  "🏦"),
        ]
        has_party = any(entities.get(k) for k, _, _ in parties)
        if has_party:
            st.markdown('<div class="section-header">Parties</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (key, label, icon) in enumerate(parties):
                val = entities.get(key)
                src = entities.get(f"_{key}_source", "")
                with cols[i]:
                    nlp_tag = " `🤖`" if src else ""
                    st.markdown(f"**{icon} {label}**{nlp_tag}")
                    st.text(val[:120] if val else "—")

        st.divider()

        st.markdown('<div class="section-header">Champs LC</div>', unsafe_allow_html=True)

        FIELDS = [
            ("lc_number",           "Réf. LC",                "🔑"),
            ("issue_date",          "Date émission",          "📅"),
            ("expiry_date",         "Date expiration",        "📅"),
            ("expiry_place",        "Lieu expiration",        "📍"),
            ("shipment_date",       "Dernière expédition",    "📅"),
            ("currency",            "Devise",                  "💱"),
            ("amount",              "Montant",                 "💰"),
            ("tolerance_plus",      "Tolérance +%",           "📏"),
            ("tolerance_minus",     "Tolérance -%",           "📏"),
            ("lc_form",             "Forme du crédit",        "📋"),
            ("applicable_rules",    "Règles applicables",     "⚖️"),
            ("confirmation",        "Confirmation",           "✅"),
            ("presentation_period", "Délai présentation",     "⏳"),
            ("partial_shipments",   "Expéditions partielles", "🚢"),
            ("transhipment",        "Transbordement",         "🔄"),
            ("port_loading",        "Port chargement",        "⚓"),
            ("port_discharge",      "Port déchargement",      "⚓"),
            ("available_with",      "Disponible auprès de",  "🏦"),
            ("available_by",        "Disponible par",         "💳"),
            ("incoterm",            "Incoterm",               "📦"),
            ("draft_at",            "Tirage à",               "📝"),
            ("sequence",            "Séquence",               "🔢"),
        ]

        filled = [(k, l, i, entities[k]) for k, l, i in FIELDS if entities.get(k)]
        cols = st.columns(3)
        for idx, (key, label, icon, val) in enumerate(filled):
            src = entities.get(f"_{key}_source", "")
            inf = entities.get(f"_{key}_inferred", False)
            tag = " 🤖" if src else (" ∿" if inf else "")
            with cols[idx % 3]:
                conf = entities.get(f"_{key}_conf")
                conf_str = f" `{conf:.2f}`" if conf else ""
                st.markdown(
                    f"<small style='color:#6c757d'>{icon} {label}{tag}</small>  \n"
                    f"**{str(val)[:60]}**{conf_str}",
                    unsafe_allow_html=True,
                )

        LONG = [
            ("goods_description",     "📦 Marchandises (:45A:)"),
            ("documents_required",    "📄 Documents requis (:46A:)"),
            ("additional_conditions", "📌 Conditions supplémentaires (:47A:)"),
            ("charges",               "💰 Frais (:71B:)"),
            ("sender_info",           "ℹ️ Sender to Receiver (:72:)"),
            ("guarantee_text",        "🔒 Texte garantie (:77U:)"),
        ]
        for key, label in LONG:
            val = entities.get(key)
            if val:
                st.markdown(
                    f"<div class='long-field-label'>{label}  —  {len(val)} car.</div>",
                    unsafe_allow_html=True,
                )
                st.text_area(
                    label, val, height=100, disabled=True,
                    key=f"long_{key}", label_visibility="collapsed",
                )

        raw = entities.get("raw_fields", {})
        if raw:
            st.markdown(
                "<div class='long-field-label'>🗂️ Champs bruts</div>",
                unsafe_allow_html=True,
            )
            st.json(raw)

        warns = entities.get("warnings", [])
        if warns:
            for w in warns:
                st.warning(w)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — CRITÈRES LC
# ══════════════════════════════════════════════════════════════════════

def show_lc_section(lc: dict):
    if not lc:
        return
    with st.expander("**② Critères d'acceptation LC**", expanded=True):

        docs = lc.get("documents", [])
        if docs:
            st.markdown('<div class="section-header">Documents requis</div>',
                        unsafe_allow_html=True)
            for i, doc in enumerate(docs, 1):
                copies    = f"  {doc.get('copies')} ex." if doc.get("copies") else ""
                conds     = " · ".join(doc.get("conditions", []))
                conds_str = f"  —  `{conds}`" if conds else ""
                st.markdown(f"**{i}.** `{doc.get('document_type','?')}`{copies}{conds_str}")

        c1, c2, c3, c4 = st.columns(4)
        p = lc.get("partial_shipments_allowed")
        t = lc.get("transhipment_allowed")
        d = lc.get("presentation_period_days")
        a = lc.get("available_by")
        with c1:
            if p is not None:
                st.metric("Expéd. partielles", "✅ Autorisées" if p else "❌ Interdites")
        with c2:
            if t is not None:
                st.metric("Transbordement", "✅ Autorisé" if t else "❌ Interdit")
        with c3:
            if d: st.metric("Délai présentation", f"{d} jours")
        with c4:
            if a: st.metric("Disponible par", a)

        pl  = lc.get("port_of_loading")
        pd_ = lc.get("port_of_discharge")
        if pl or pd_:
            c_pl, c_pd = st.columns(2)
            with c_pl:
                if pl: st.metric("⚓ Port chargement", pl)
            with c_pd:
                if pd_: st.metric("⚓ Port déchargement", pd_)

        add = lc.get("additional_conditions", [])
        if add:
            st.markdown('<div class="section-header">📌 Conditions supplémentaires</div>',
                        unsafe_allow_html=True)
            for c in add:
                st.markdown(f"- {c}")

        warns = lc.get("warnings", [])
        if warns:
            for w in warns:
                st.warning(w)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — NLP
# ══════════════════════════════════════════════════════════════════════

def show_nlp_section(nlp: dict):
    if not nlp:
        return
    with st.expander("**③ Analyse NLP**", expanded=True):
        _show_spell(nlp.get("spell"))
        st.divider()
        _show_ner(nlp.get("ner"))
        st.divider()
        _show_clauses(nlp.get("clauses", []), nlp.get("contradictions", []))
        st.divider()
        _show_semantic(nlp.get("semantic"))


def _show_spell(spell: dict):
    st.markdown('<div class="section-header">🔤 Correction orthographique</div>',
                unsafe_allow_html=True)
    if not spell:
        st.caption("—")
        return
    changes = spell.get("total_changes", 0)
    if changes == 0:
        st.success("✅ Aucune faute détectée.")
        return

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Corrections", changes)
    with c2: st.metric("Langue",      spell.get("language_detected", "?").upper())
    with c3: st.metric("Conf. moy.",  f"{spell.get('avg_confidence', 0):.2f}")

    corrs = spell.get("corrections", [])
    if corrs:
        rows = []
        for c in corrs:
            icon = {"static": "📖", "levenshtein": "🔤", "ocr_noise": "🖨️"}.get(
                c.get("method", ""), "•")
            rows.append({
                "Statut":    "✅" if c.get("validated", True) else "❌",
                "Original":  c.get("original", ""),
                "→ Corrigé": c.get("corrected", ""),
                "Méthode":   f"{icon} {c.get('method','')}",
                "Conf.":     f"{c.get('confidence', 1.0):.2f}",
                "Position":  c.get("position", ""),
            })
        st.dataframe(rows, hide_index=True, width="stretch")


def _show_ner(ner: dict):
    st.markdown('<div class="section-header">🔷 Entités NER</div>',
                unsafe_allow_html=True)
    if not ner:
        st.caption("—")
        return

    entities = ner.get("entities", [])
    if not entities:
        st.info("Aucune entité détectée.")
        return

    mode       = ner.get("source_mode", "rules")
    mode_label = "🤖 Ensemble (règles + transformer)" if "ensemble" in mode else "📏 Règles seules"
    st.caption(f"Mode : **{mode_label}**  —  {len(entities)} entité(s)")

    rows = []
    for e in entities:
        label = e.get("label", "")
        span  = (f"{e['start']}–{e['end']}" if e.get("start") is not None else "—")
        conf  = e.get("confidence", 0)
        norm  = e.get("normalized", "")
        rows.append({
            "Label":   f"{ner_icon(label)} {label}",
            "Valeur":  (e.get("value") or "")[:55] + (f" → {norm}" if norm else ""),
            "Pays":    e.get("country", "") or "—",
            "Span":    span,
            "Conf.":   f"{conf_badge(conf)} {conf:.2f}",
            "Source":  e.get("source", ""),
        })
    st.dataframe(rows, hide_index=True, width="stretch")

    rels = ner.get("relations", [])
    if rels:
        st.markdown(f"**🔗 Relations ({len(rels)})**")
        for r in rels:
            st.markdown(
                f"**{r.get('subject','')}** —_{r.get('predicate','')}_→ "
                f"**{r.get('object_label','')}** : `{r.get('object_value','')}`"
            )


def _show_clauses(clauses: list, contradictions: list):
    st.markdown('<div class="section-header">📌 Clauses UCP 600</div>',
                unsafe_allow_html=True)
    if not clauses:
        st.info("Aucune clause extraite.")
        return

    type_counts = Counter(c.get("type", "?") for c in clauses)
    ml_count    = sum(1 for c in clauses if "embedding" in c.get("method", "")
                                         or "supervised" in c.get("method", ""))

    cols = st.columns(min(6, len(type_counts) + 1))
    with cols[0]: st.metric("Total", len(clauses))
    for i, (t, cnt) in enumerate(type_counts.items(), 1):
        if i < len(cols):
            with cols[i]: st.metric(clause_type_label(t).split(" ", 1)[1], cnt)

    if ml_count:
        st.caption(f"🤖 {ml_count}/{len(clauses)} classifiées par ML")

    if contradictions:
        st.error(f"⚠️ **{len(contradictions)} contradiction(s) détectée(s)**")
        for cont in contradictions:
            st.markdown(
                f'<div class="inner-block"><strong>🔴 {cont.get("message","")[:80]}</strong></div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Clause A")
                st.markdown(f"> {cont.get('clause_a','')[:120]}")
            with c2:
                st.caption("Clause B")
                st.markdown(f"> {cont.get('clause_b','')[:120]}")

    by_type: dict[str, list] = {}
    for c in clauses:
        by_type.setdefault(c.get("type", "unknown"), []).append(c)

    for clause_type, group in by_type.items():
        label     = clause_type_label(clause_type)
        high_risk = sum(1 for c in group if c.get("risk_level") == "high")
        risk_badge = f"  🔴 {high_risk} high" if high_risk else ""

        st.markdown(
            f"**{label}  —  {len(group)} clause(s){risk_badge}**"
        )
        for c in group:
            risk     = c.get("risk_level", "normal")
            imp      = c.get("importance", "low")
            conf     = c.get("confidence", 0)
            method   = c.get("method", "")
            ucp_refs = c.get("ucp_refs", [])
            text     = c.get("text", "")

            border_color = {"high": "#dc3545", "medium": "#fd7e14"}.get(risk, "#0d6efd")
            ucp_str      = ", ".join(ucp_refs[:3]) if ucp_refs else ""

            st.markdown(
                f"""<div style="border-left:3px solid {border_color};
                    padding:8px 12px;margin-bottom:6px;border-radius:0 6px 6px 0;
                    background:#fdfdfd">
                    <div style="font-size:12px;color:#6c757d;margin-bottom:3px">
                      {importance_badge(imp)} &nbsp;·&nbsp;
                      {method_badge(method)} &nbsp;·&nbsp;
                      conf <strong>{conf:.2f}</strong>
                      {"&nbsp;·&nbsp;<code>" + ucp_str + "</code>" if ucp_str else ""}
                    </div>
                    <div style="font-size:13px">{text[:280]}{"…" if len(text)>280 else ""}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("")


def _show_semantic(sem: dict):
    st.markdown('<div class="section-header">🔬 Analyse sémantique</div>',
                unsafe_allow_html=True)
    if not sem:
        st.info("Analyse sémantique non disponible.")
        return

    comp  = sem.get("completeness_score", 0)
    coh   = sem.get("semantic_score", 0)
    doc_q = sem.get("doc_quality_score", 0)
    geo   = sem.get("geo_coherence", {})
    risks = sem.get("risk_flags", [])
    fraud = sem.get("fraud_indicators", [])
    warns = sem.get("warnings", [])
    recs  = sem.get("recommendations", [])
    summ  = sem.get("summary", "")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Complétude",    f"{score_emoji(comp)} {comp}%")
    with c2: st.metric("Cohérence NLP", f"{score_emoji(coh)} {coh}%")
    with c3: st.metric("Qualité docs",  f"{score_emoji(doc_q)} {doc_q}%")
    with c4:
        if isinstance(geo, dict) and geo:
            match = geo.get("expiry_beneficiary_match", None)
            place = geo.get("expiry_place", "")
            st.metric("Cohérence géo", "✅ OK" if match else (f"⚠️ {place}" if place else "—"))
        elif geo is True:
            st.metric("Cohérence géo", "✅ OK")
        else:
            st.metric("Cohérence géo", "—")

    if summ:
        st.info(f"📝 {summ}")

    coh_scores = sem.get("coherence_scores", {})
    if coh_scores:
        st.caption("Scores de cohérence :")
        cols = st.columns(len(coh_scores))
        for i, (k, v) in enumerate(coh_scores.items()):
            pct = int(v * 100) if v <= 1 else int(v)
            with cols[i]: st.metric(k.replace("_", " "), f"{pct}%")

    if risks:
        st.markdown("**🚨 Risk flags**")
        for r in risks:
            sev      = r.get("severity", "low")
            code     = r.get("code", "")
            desc     = r.get("description", "")
            ml_score = r.get("ml_score", 0)
            ucp      = r.get("ucp_ref", "")
            cat      = r.get("category", "")
            ml_str   = f"  ML={ml_score:.2f}" if ml_score > 0 else ""
            cat_str  = f" `{cat}`" if cat else ""
            ucp_str  = f"  _{ucp}_" if ucp else ""
            line     = f"**[{code}]**{cat_str} — {desc}{ucp_str}{ml_str}"
            if sev == "high":     st.error(f"🔴 {line}")
            elif sev == "medium": st.warning(f"🟡 {line}")
            else:                 st.info(f"🔵 {line}")

    if fraud:
        st.markdown("**🔒 Indicateurs de fraude**")
        for fi in fraud:
            sev  = fi.get("severity", "warning")
            code = fi.get("code", "")
            desc = fi.get("description", "")
            if sev == "high": st.error(f"🔴 **{code}** — {desc}")
            else:             st.warning(f"🟡 **{code}** — {desc}")

    if warns:
        st.markdown(f"**⚠️ Warnings UCP ({len(warns)})**")
        with st.container():
            for w in warns:
                sev  = w.get("severity", "info")
                code = w.get("code", "")
                msg  = w.get("message", "")
                sugg = w.get("suggestion", "")
                ucp  = w.get("ucp_ref", "")
                ucp_str = f"  _{ucp}_" if ucp else ""
                line    = f"**[{code}]** {msg}{ucp_str}"
                if sev == "error":    st.error(line)
                elif sev == "warning": st.warning(line)
                else:                  st.info(line)
                if sugg: st.caption(f"💡 {sugg}")

    if recs:
        st.markdown(f"**💡 Recommandations ({len(recs)})**")
        with st.container():
            for i, rec in enumerate(recs, 1):
                st.markdown(f"**{i}.** {rec}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — VALIDATION
# ══════════════════════════════════════════════════════════════════════

def show_validation_section(val: dict):
    if not val:
        return
    with st.expander("**④ Validation SWIFT (champs obligatoires)**", expanded=True):
        overall = val.get("overall_valid", False)
        score   = val.get("score", 0)
        summ    = val.get("summary", "")
        stype   = val.get("swift_type", "")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Type",   stype)
        with c2: st.metric("Statut", "✅ VALIDE" if overall else "❌ INVALIDE")
        with c3: st.metric("Score",  f"{score}/100",
                            delta=None if score == 100 else f"{100-score} points manquants")

        if summ:
            (st.success if overall else st.error)(summ)

        missing = val.get("missing_mandatory", [])
        if missing:
            st.error(f"**Champs obligatoires manquants :** {', '.join(f':{t}:' for t in missing)}")

        cross_errs  = val.get("cross_field_errors", [])
        cross_warns = val.get("cross_field_warnings", [])

        if cross_errs:
            st.markdown(f"**🚫 Erreurs inter-champs ({len(cross_errs)})**")
            for e in cross_errs:
                st.error(e)

        if cross_warns:
            st.markdown(f"**⚠️ Avertissements inter-champs ({len(cross_warns)})**")
            for w in cross_warns:
                st.warning(w)

        fields = val.get("fields", {})
        if fields:
            errs_fields = {t: f for t, f in fields.items() if f.get("errors")}
            warn_fields = {t: f for t, f in fields.items()
                           if not f.get("errors") and f.get("warnings")}
            ok_fields   = {t: f for t, f in fields.items()
                           if not f.get("errors") and not f.get("warnings")}

            def _render_field_table(field_dict: dict, icon: str):
                rows = []
                for tag, fv in sorted(field_dict.items()):
                    msgs = fv.get("errors", []) or fv.get("warnings", [])
                    rows.append({
                        "Tag":    f":{tag}:",
                        "Champ":  fv.get("name", tag),
                        "Valeur": str(fv.get("value") or "—")[:60],
                        "Note":   msgs[0][:80] if msgs else "—",
                    })
                if rows:
                    st.dataframe(rows, hide_index=True, width="stretch")

            if errs_fields:
                st.markdown(f"**🔴 Champs en erreur ({len(errs_fields)})**")
                _render_field_table(errs_fields, "🔴")

            if warn_fields:
                st.markdown(f"**🟡 Champs avec avertissements ({len(warn_fields)})**")
                _render_field_table(warn_fields, "🟡")

            st.markdown(f"**🟢 Champs valides ({len(ok_fields)})**")
            _render_field_table(ok_fields, "🟢")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — CONFORMITÉ UCP 600
# ══════════════════════════════════════════════════════════════════════

def show_ucp_section(ucp: dict):
    if not ucp:
        return

    with st.expander(
        f"**⑤ Conformité UCP 600** — Score {ucp.get('global_score', 0)}/100  "
        f"{'✅ Conforme' if ucp.get('compliant') else '⚠️ Non conforme'}",
        expanded=True,
    ):
        global_score = ucp.get("global_score", 0)
        compliant    = ucp.get("compliant", False)
        exclusions   = ucp.get("exclusions", {})
        findings     = ucp.get("findings", [])
        categories   = ucp.get("categories", {})
        processed_ms = ucp.get("processing_ms", 0)

        errors   = [f for f in findings if f.get("severity") == "error"]
        errors_blocking = [f for f in errors if f.get("blocks_payment")]
        errors_non_blocking = [f for f in errors if not f.get("blocks_payment")]
        warnings = [f for f in findings if f.get("severity") == "warning"]
        infos    = [f for f in findings if f.get("severity") == "info"]
        pay_block = bool(ucp.get("payment_blocked"))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("Score UCP",  f"{global_score}/100")
        with c2: st.metric("Statut",     "✅ Conforme" if compliant else "⚠ Non conforme")
        with c3: st.metric("🔴 Erreurs", len(errors))
        with c4: st.metric("🟡 Alertes", len(warnings))
        with c5: st.metric("Blocage paiement", "Oui" if pay_block else "Non")
        with c6: st.metric("⏱ Temps",   f"{processed_ms}ms")

        if exclusions:
            st.markdown("**📋 Articles UCP exclus dans ce crédit :**")
            excl_cols = st.columns(min(4, len(exclusions)))
            for i, (art, ctx) in enumerate(exclusions.items()):
                with excl_cols[i % 4]:
                    st.markdown(
                        f'<div style="background:#fff3cd;border:1px solid #ffc107;'
                        f'border-radius:6px;padding:6px 10px;text-align:center;'
                        f'font-size:12px"><strong>{art}</strong><br>'
                        f'<span style="color:#6c757d">{ctx[:40]}…</span></div>',
                        unsafe_allow_html=True,
                    )
            st.caption("")

        if categories:
            st.markdown("**Scores par catégorie :**")
            cat_cols  = st.columns(len(categories))
            cat_icons = {"document": "📄", "compliance": "⚖️", "payment": "💰", "shipment": "🚢"}
            for i, (cat_name, cat_data) in enumerate(categories.items()):
                score = cat_data.get("score", 100)
                nok   = cat_data.get("articles_nok", [])
                icon  = cat_icons.get(cat_name, "•")
                emoji = "🟢" if score >= 80 else ("🟡" if score >= 50 else "🔴")
                with cat_cols[i]:
                    st.metric(f"{icon} {cat_name.capitalize()}", f"{emoji} {score}/100")
                    if nok: st.caption(", ".join(nok[:3]))

        st.divider()

        sem = ucp.get("semantic") or {}
        if sem.get("mode") == "semantic" and sem.get("available"):
            cap = (
                f"Moteur UCP sémantique — violations clause: {sem.get('n_violation', 0)} · "
                f"restrictions: {sem.get('n_restriction', 0)} · conformes: {sem.get('n_compliant', 0)}"
            )
            if sem.get("nli_enabled"):
                if sem.get("nli_ready"):
                    cap += " · NLI (CrossEncoder) actif"
                else:
                    cap += " · NLI demandé (UCP_USE_NLI) mais modèle non chargé"
            st.caption(cap)
        elif sem.get("mode") == "semantic_strict_failed":
            st.warning(
                "Moteur UCP sémantique indisponible (UCP_SEMANTIC_STRICT) — "
                "rapport limité aux contrôles structurés SWIFT + message d'erreur."
            )

        def _render_error_block(f, accent: str, border: str):
            st.markdown(
                f'<div class="inner-block">'
                f'<strong>{accent} [{f.get("article","")}]</strong> {f.get("message","")[:90]}<br>'
                f'<em style="font-size:12px;color:#6c757d">{f.get("article_title","")}</em>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if f.get("clause_text"):
                st.markdown(
                    f'<div style="background:#fff8f8;border-left:3px solid {border};'
                    f'padding:6px 10px;font-size:12px;margin:4px 0 8px 0">'
                    f'📌 Clause source :<br><em>{f["clause_text"][:200]}</em></div>',
                    unsafe_allow_html=True,
                )
            if f.get("recommendation"):
                st.success(f"💡 **Action :** {f['recommendation']}")

        if errors_blocking:
            st.markdown(
                f"**🔴 Erreurs bloquant le paiement ({len(errors_blocking)}) — Action requise**"
            )
            for f in errors_blocking:
                _render_error_block(f, "🔒", "#b02a37")

        if errors_non_blocking:
            st.markdown(
                f"**🔴 Autres erreurs ({len(errors_non_blocking)}) — À corriger**"
            )
            for f in errors_non_blocking:
                _render_error_block(f, "🔴", "#dc3545")

        if warnings:
            st.markdown(f"**🟡 Alertes ({len(warnings)}) — À vérifier**")
            for f in warnings:
                st.markdown(
                    f'<div class="inner-block">'
                    f'<strong>🟡 [{f.get("article","")}]</strong> {f.get("message","")[:90]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if f.get("clause_text"): st.caption(f"Clause : {f['clause_text'][:150]}")
                if f.get("recommendation"): st.info(f"💡 {f['recommendation']}")

        if infos:
            st.markdown(f"**ℹ️ Informations ({len(infos)})**")
            for f in infos:
                st.markdown(f"**[{f.get('article','')}]** {f.get('message','')}")
                if f.get("recommendation"): st.caption(f"→ {f['recommendation']}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — ANALYSE DATA SCIENCE
# ══════════════════════════════════════════════════════════════════════

def show_ds_section(ds: dict):
    if not ds:
        return

    risk     = ds.get("risk", {})
    features = ds.get("features", {})
    ds_ms    = ds.get("processing_ms", 0)

    risk_level = risk.get("risk_level", "?")
    risk_score = risk.get("risk_score", 0)
    factors    = risk.get("risk_factors", [])
    model_used = risk.get("model_used", "rules")
    top_feat   = risk.get("top_features", {})

    LEVEL_CONFIG = {
        "LOW":      ("🟢", "#198754", "Risque faible — LC standard"),
        "MEDIUM":   ("🟡", "#fd7e14", "Risque modéré — vérification recommandée"),
        "HIGH":     ("🔴", "#dc3545", "Risque élevé — examen approfondi requis"),
        "CRITICAL": ("🚨", "#7b0000", "Risque critique — escalade requise"),
    }
    icon, color, label = LEVEL_CONFIG.get(risk_level, ("?", "#6c757d", ""))

    with st.expander(
        f"**⑥ Analyse Data Science** — {icon} {risk_level}  (score {risk_score:.2f})",
        expanded=True,
    ):
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Niveau de risque", f"{icon} {risk_level}")
        with c2:
            pct = int(risk_score * 100)
            st.metric("Risk Score", f"{pct}/100",
                      delta=None if pct < 50 else f"+{pct-50} vs seuil",
                      delta_color="inverse")
        with c3: st.metric("Modèle", "🤖 ML" if model_used != "rules" else "📏 Règles")
        with c4: st.metric("⏱ Temps", f"{ds_ms}ms")

        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:8px 14px;border-radius:0 6px 6px 0;margin:8px 0">'
            f'<strong style="color:{color}">{icon} {label}</strong></div>',
            unsafe_allow_html=True,
        )

        if factors:
            st.markdown("**Facteurs ayant contribué au score :**")
            for factor in factors:
                ficon = ("⏱" if "validit" in factor.lower() or "jour" in factor.lower()
                         else "📋" if "clause" in factor.lower()
                         else "⚖️" if "ucp" in factor.lower() or "article" in factor.lower()
                         else "🌍" if "pays" in factor.lower()
                         else "💱" if "devise" in factor.lower()
                         else "🚢" if "incoterm" in factor.lower()
                         else "•")
                st.markdown(
                    f'<div style="padding:4px 10px;margin:3px 0;background:#f8f9fa;'
                    f'border-radius:4px;font-size:13px">{ficon} {factor}</div>',
                    unsafe_allow_html=True,
                )

        if top_feat:
            st.markdown("**Top features ML :**")
            feat_cols = st.columns(min(5, len(top_feat)))
            for i, (feat_name, importance) in enumerate(list(top_feat.items())[:5]):
                with feat_cols[i]:
                    st.metric(feat_name.replace("_", " "), f"{int(importance*100)}%")

        if features:
            st.markdown(f"**🔢 Features extraites ({len(features)} non-nulles)**")
            groups = {
                "⏱ Temporel":    ["validity_days", "presentation_days", "days_to_shipment"],
                "💰 Financier":   ["lc_amount", "tolerance_plus", "tolerance_minus", "currency_risk"],
                "📋 Structure":   ["n_documents_required", "n_clauses", "n_high_risk_clauses",
                                   "n_critical_clauses", "n_compliance_clauses",
                                   "n_ucp_exclusions", "n_ucp_errors", "n_ucp_warnings"],
                "🚢 Transport":   ["transshipment_allowed", "partial_ship_allowed",
                                   "confirmation_required", "charges_beneficiary",
                                   "incoterm_risk", "country_risk"],
                "🤖 Scores NLP": ["completeness_score", "semantic_score",
                                   "nlp_score", "ml_risk_score", "ucp_compliance_score"],
            }
            for group_name, keys in groups.items():
                group_vals = {k: features[k] for k in keys if k in features}
                if not group_vals:
                    continue
                st.caption(group_name)
                g_cols = st.columns(min(4, len(group_vals)))
                for i, (k, v) in enumerate(group_vals.items()):
                    with g_cols[i % 4]:
                        val_str = (f"{v:.0f}" if isinstance(v, float) and v == int(v)
                                   else f"{v:.2f}" if isinstance(v, float) else str(v))
                        st.metric(k.replace("_", " ").title()[:20], val_str)


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — VALIDATION EXTERNE
# ══════════════════════════════════════════════════════════════════════

def show_external_section(ext: dict):
    if not ext:
        return

    risk_level = ext.get("risk_level", "CLEAN")
    risk_score = ext.get("risk_score", 0.0)
    approach   = ext.get("approach", "")
    sources    = ext.get("sources_consulted", [])
    ext_ms     = ext.get("processing_ms", 0)
    factors    = ext.get("risk_factors", [])
    recs       = ext.get("recommendations", [])
    details    = ext.get("details", {}) or {}

    icon  = EXT_ICONS.get(risk_level, "?")
    color = EXT_COLORS.get(risk_level, "#6c757d")

    with st.expander(
        f"**⑦ Validation externe** — {icon} {risk_level}  "
        f"(score {risk_score:.2f}  ·  {len(sources)} sources  ·  {ext_ms}ms)",
        expanded=True,
    ):
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Niveau",        f"{icon} {risk_level}")
        with c2: st.metric("Score pondéré", f"{risk_score:.2f}")
        with c3: st.metric("Sources",       len(sources))
        with c4: st.metric("Temps API",     f"{ext_ms}ms")

        st.markdown(
            f'<div style="background:{color}22;border-left:4px solid {color};'
            f'padding:8px 14px;border-radius:0 6px 6px 0;margin:8px 0">'
            f'<strong style="color:{color}">{icon} {risk_level}</strong>'
            f'{"  —  " + approach if approach else ""}</div>',
            unsafe_allow_html=True,
        )

        if sources:
            st.caption("Sources consultées : " + "  ·  ".join(f"`{s}`" for s in sources))

        st.divider()

        countries = details.get("countries", [])
        if countries:
            st.markdown('<div class="section-header">🌍 Pays</div>',
                        unsafe_allow_html=True)
            for c in countries:
                lvl      = c.get("risk", "CLEAN")
                c_icon   = EXT_ICONS.get(lvl, "?")
                c_color  = EXT_COLORS.get(lvl, "#6c757d")
                iso2     = c.get("iso2") or c.get("country_iso2", "?")
                name     = c.get("normalized", c.get("raw", "?"))
                c_detail = c.get("details", "")
                srcs     = " · ".join(c.get("sources", []))
                detail_str = f"  ·  {c_detail}" if c_detail else ""
                srcs_str   = (f'  <small style="color:#6c757d">[{srcs}]</small>'
                               if srcs else "")
                st.markdown(
                    f'<div class="ext-row">'
                    f'<span style="color:{c_color};font-weight:600">{c_icon} {iso2}</span>'
                    f'  <strong>{name}</strong>  —  '
                    f'<span style="color:{c_color}">{lvl}</span>'
                    f'{detail_str}{srcs_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        banks = details.get("banks", [])
        if banks:
            st.markdown('<div class="section-header">🏦 Banques</div>',
                        unsafe_allow_html=True)
            rows = []
            for b in banks:
                lvl    = b.get("risk", "CLEAN")
                b_icon = EXT_ICONS.get(lvl, "?")
                rows.append({
                    "BIC":          b.get("bic", "?"),
                    "Nom GLEIF":    b.get("gleif_name", "—")[:40] or "—",
                    "Statut GLEIF": b.get("gleif_status", "?"),
                    "Blacklisté":   "🚨 OUI" if b.get("blacklisted") else "✅ NON",
                    "Pays":         b.get("country", "?"),
                    "Risque":       f"{b_icon} {lvl}",
                    "Score":        b.get("score", 0),
                })
            st.dataframe(rows, hide_index=True, width="stretch")

        companies = details.get("companies", [])
        if companies:
            st.markdown('<div class="section-header">🏢 Entités</div>',
                        unsafe_allow_html=True)
            for co in companies:
                lvl      = co.get("risk", "CLEAN")
                co_icon  = EXT_ICONS.get(lvl, "?")
                co_color = EXT_COLORS.get(lvl, "#6c757d")
                name     = co.get("name", "?")[:60]
                flags    = []
                if co.get("ofac_matched"): flags.append("OFAC")
                if co.get("eu_matched"):   flags.append("EU")
                if co.get("un_matched"):   flags.append("UN")
                flags_html = "".join(
                    f'<span style="background:#dc354522;color:#dc3545;'
                    f'padding:1px 6px;border-radius:4px;font-size:11px;margin-left:4px">'
                    f'{f}</span>'
                    for f in flags
                )
                oc_found  = co.get("opencorp_found")
                oc_status = co.get("opencorp_status", "")
                oc_str    = ""
                if oc_found is True:
                    oc_str = (f'  <small style="color:#198754">'
                              f'✅ OpenCorp: {oc_status}</small>')
                elif oc_found is False:
                    oc_str = '  <small style="color:#dc3545">❌ Introuvable OpenCorp</small>'

                co_detail   = co.get("details", "")
                detail_html = (
                    f'<br><small style="color:#6c757d;margin-left:20px">{co_detail}</small>'
                    if co_detail and lvl != "CLEAN" else ""
                )

                st.markdown(
                    f'<div class="ext-row">'
                    f'<span style="color:{co_color};font-weight:600">{co_icon}</span>'
                    f'  <strong>{name}</strong>{flags_html}{oc_str}'
                    f'  —  <span style="color:{co_color}">{lvl}</span>'
                    f'{detail_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        geo = details.get("geo")
        if geo:
            st.markdown('<div class="section-header">🗺️ Géographie</div>',
                        unsafe_allow_html=True)
            g_c1, g_c2, g_c3 = st.columns(3)
            with g_c1:
                load_c = geo.get("loading_country_resolved", "?")
                st.metric("Port chargement → pays", load_c if load_c else "Non résolu")
            with g_c2:
                disc_c = geo.get("discharge_country_resolved", "?")
                st.metric("Port déchargement → pays", disc_c if disc_c else "Non résolu")
            with g_c3:
                geo_lvl = geo.get("risk", "CLEAN")
                st.metric("Risque géo", f"{EXT_ICONS.get(geo_lvl,'?')} {geo_lvl}")

            for issue in geo.get("issues", []):
                sev = issue.get("severity", "info")
                msg = issue.get("message", "")
                cod = issue.get("code", "")
                line = f"**[{cod}]** {msg}" if cod else msg
                if sev == "error":     st.error(f"🚨 {line}")
                elif sev == "warning": st.warning(f"⚠️ {line}")
                else:                  st.info(f"ℹ️ {line}")

        if factors and not all("Aucun risque" in f for f in factors):
            st.markdown(f"**⚠️ Facteurs de risque ({len(factors)})**")
            with st.container():
                for f in factors:
                    f_icon = "🚨" if "BLOQUER" in f else "⚠️"
                    st.markdown(
                        f'<div style="padding:4px 10px;margin:3px 0;background:#f8f9fa;'
                        f'border-radius:4px;font-size:13px">{f_icon} {f}</div>',
                        unsafe_allow_html=True,
                    )

        if recs:
            st.markdown(f"**📋 Recommandations ({len(recs)})**")
            with st.container():
                for rec in recs:
                    if rec.startswith("⛔"):  st.error(rec)
                    elif rec.startswith("⚠"): st.warning(rec)
                    else:                      st.info(rec)


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — RISK ML  (v7.3)
# ══════════════════════════════════════════════════════════════════════

def show_risk_ml_section(risk_ml: dict):
    if not risk_ml:
        return

    with st.expander("**⑧ Risk ML — XGBoost + IsolationForest + KMeans + SHAP**", expanded=True):

        if risk_ml.get("error"):
            st.warning(f"⚠️ Risk ML indisponible : {risk_ml['error']}")
            st.caption("Lancer `python scripts/train_models.py` pour entraîner les modèles.")
            return

        risk_label = risk_ml.get("risk_label", "UNKNOWN")
        risk_score = risk_ml.get("risk_score", 0)
        risk_proba = risk_ml.get("risk_proba", {})
        is_anomaly = risk_ml.get("is_anomaly", False)
        ano_score  = risk_ml.get("anomaly_score", 0.0)
        segment    = risk_ml.get("segment", "?")
        shap_top   = risk_ml.get("shap_top_features", [])
        features   = risk_ml.get("features_used", {})
        version    = risk_ml.get("model_version", "?")
        ms         = risk_ml.get("processing_ms", 0)

        col_gauge, col_proba, col_detail = st.columns([1.5, 2, 2.5])

        with col_gauge:
            badge_class = f"risk-badge-{risk_label}"
            gauge_color = "#198754" if risk_score < 35 else ("#fd7e14" if risk_score < 65 else "#dc3545")
            st.markdown(f"""
            <div class="risk-gauge-wrap">
              <div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px">
                Risk Score
              </div>
              <div style="font-size:56px;font-weight:800;color:{gauge_color};line-height:1">
                {risk_score}
              </div>
              <div style="font-size:11px;color:#9ca3af;margin-bottom:8px">/&nbsp;100</div>
              <div class="risk-label-badge {badge_class}">{risk_label}</div>
              <div style="font-size:10px;color:#9ca3af;margin-top:8px">
                v{version} · {ms}ms
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_proba:
            st.markdown('<div class="section-header">Distribution de probabilité</div>',
                        unsafe_allow_html=True)
            PROBA_COLORS = {"LOW": "#198754", "MEDIUM": "#fd7e14", "HIGH": "#dc3545"}
            for lbl in ["LOW", "MEDIUM", "HIGH"]:
                prob = risk_proba.get(lbl, 0.0)
                pct  = int(prob * 100)
                col  = PROBA_COLORS[lbl]
                st.markdown(f"""
                <div style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;
                              font-size:12px;margin-bottom:3px">
                    <span style="font-weight:600;color:{col}">{lbl}</span>
                    <span style="color:#6b7280">{prob:.1%}</span>
                  </div>
                  <div style="background:#e5e7eb;border-radius:6px;height:12px;overflow:hidden">
                    <div style="width:{pct}%;height:12px;background:{col};
                                border-radius:6px;transition:width 0.4s"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_detail:
            st.markdown('<div class="section-header">Détection & Segmentation</div>',
                        unsafe_allow_html=True)
            if is_anomaly:
                st.markdown(
                    f'<div class="anomaly-badge-yes">⚠️ Anomalie détectée'
                    f'<br><span style="font-size:11px;font-weight:400">'
                    f'Score : {ano_score:.4f}</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption("IsolationForest identifie ce LC comme atypique (top 10% suspects).")
            else:
                st.markdown(
                    f'<div class="anomaly-badge-no">✅ Comportement normal'
                    f'<br><span style="font-size:11px;font-weight:400">'
                    f'Score : {ano_score:.4f}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            _SEG_ICONS = {
                "LOW_RISK_TRADE": "🟢",
                "STANDARD_LC":   "🔵",
                "HIGH_RISK_LC":  "🔴",
                "COMPLEX_LC":    "🟠",
            }
            seg_icon = _SEG_ICONS.get(segment, "🔷")
            st.markdown(
                f'<div class="segment-badge">{seg_icon} Segment KMeans<br>'
                f'<span style="font-weight:700;font-size:15px">{segment}</span></div>',
                unsafe_allow_html=True,
            )

        if shap_top:
            st.markdown("---")
            st.markdown('<div class="section-header">🔍 SHAP — Top 5 features explicatives (classe prédite)</div>',
                        unsafe_allow_html=True)
            max_shap = max(abs(f["shap_value"]) for f in shap_top) or 1
            for feat_info in shap_top:
                feat  = feat_info["feature"]
                val   = feat_info["shap_value"]
                direc = feat_info["direction"]
                pct   = int(abs(val) / max_shap * 100)
                is_pos = val > 0
                bar_color  = "#dc3545" if is_pos else "#198754"
                dir_color  = "#dc3545" if is_pos else "#198754"

                st.markdown(f"""
                <div class="shap-bar-wrap">
                  <div class="shap-bar-label">{feat}</div>
                  <div class="shap-bar-outer">
                    <div class="{'shap-bar-inner-pos' if is_pos else 'shap-bar-inner-neg'}"
                         style="width:{pct}%;background:{bar_color}"></div>
                  </div>
                  <div class="shap-bar-val" style="color:{dir_color}">{direc}</div>
                  <div style="font-size:11px;color:#9ca3af;width:55px;text-align:right">
                    {val:+.4f}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        if features:
            st.markdown("---")
            with st.expander("📋 Features utilisées pour le scoring", expanded=False):
                # Liste des features utilisées par le modèle (importée depuis risk_engine si besoin)
                FEATURE_COLS = [
                    "validity_days", "presentation_days", "days_to_shipment",
                    "lc_amount", "tolerance_plus", "tolerance_minus", "currency_risk",
                    "n_documents_required", "n_clauses", "n_high_risk_clauses",
                    "n_critical_clauses", "n_compliance_clauses", "n_ucp_exclusions",
                    "n_ucp_errors", "n_ucp_warnings", "transshipment_allowed",
                    "partial_ship_allowed", "confirmation_required", "charges_beneficiary",
                    "incoterm_risk", "country_risk", "completeness_score", "semantic_score",
                    "nlp_score", "ml_risk_score", "ucp_compliance_score"
                ]
                rows_html = ""
                for feat in FEATURE_COLS:
                    v = features.get(feat, "—")
                    if isinstance(v, float):
                        v_str = f"{v:.4f}"
                    else:
                        v_str = str(v)
                    rows_html += f"<tr><td><code>{feat}</code></td><td style='text-align:right'>{v_str}</td></tr>"

                st.markdown(f"""
                <table class="feat-table">
                  <thead>
                    <tr><th>Feature</th><th style="text-align:right">Valeur</th></tr>
                  </thead>
                  <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def show_result(data: dict):
    """Affiche le résultat complet du pipeline (sections ①–⑧)."""
    swift = _get_swift_payload(data)
    nlp   = swift.get("nlp", {}) or {}

    show_dashboard(data)

    if not swift.get("success"):
        st.error(f"❌ {swift.get('error', 'Erreur pipeline')}")
        if nlp:
            show_nlp_section(nlp)
        return

    show_entities_section(swift.get("entities", {}), nlp.get("ner"))
    show_lc_section(swift.get("lc_criteria"))
    show_nlp_section(nlp)
    show_validation_section(swift.get("validation"))
    show_ucp_section(data.get("ucp_compliance"))
    show_ds_section(data.get("ds_analysis"))
    show_external_section(data.get("external_validation"))
    show_risk_ml_section(data.get("risk_ml"))

    st.divider()
    c_dl, c_exp = st.columns([1, 4])
    with c_dl:
        st.download_button(
            "⬇️ Télécharger JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2),
            file_name="swift_result.json",
            mime="application/json",
            key="dl_json_bottom",
        )
    with c_exp:
        st.markdown("**🗃️ JSON brut**")
        st.json(data)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("🔧 Outils")

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
                f"Modèles ML :  "
                f"{'✅' if clf_ok else '❌'} Clauses  "
                f"{'✅' if risk_ok else '❌'} DS Risk  "
                f"{'✅' if ext_ok else '❌'} Ext.Val  "
                f"{'✅' if rml_ok else '❌'} Risk ML"
            )
            if rml_ok:
                st.caption(f"🤖 Risk ML v{hdata.get('risk_ml_version','?')} — prêt")
            elif not rml_ok:
                st.warning("⚠️ Risk ML : modèles non entraînés\n"
                           "`python scripts/train_models.py`")
        else:
            st.warning("⚠️ API — réponse inattendue")
    except Exception:
        st.error("🔴 API non joignable")

    st.divider()

    sidebar_mode = st.selectbox(
        "Mode d'analyse",
        [
            "Pipeline complet",
            "NLP seul (texte)",
            "Correction orthographique",
            "Classification clauses",
        ],
        key="sidebar_mode",
    )

    raw_text = st.text_area("Texte SWIFT", height=160, key="sidebar_text",
                             placeholder=":27: 1/1\n:40A: IRREVOCABLE\n:20: LC2026…")
    swift_type_sel = st.selectbox(
        "Type MT (optionnel)",
        ["", "MT700", "MT720", "MT760", "MT790", "MT799"],
        key="sidebar_mt",
    )

    if st.button("▶ Analyser", width="stretch", type="primary"):
        text = raw_text.strip()
        if not text:
            st.warning("Saisissez un texte SWIFT.")
        else:
            with st.spinner("Analyse en cours…"):
                try:
                    if sidebar_mode == "Pipeline complet":
                        resp = requests.post(ANALYZE_URL, json={"text": text})
                    elif sidebar_mode == "NLP seul (texte)":
                        payload = {"text": text}
                        if swift_type_sel:
                            payload["swift_type"] = swift_type_sel
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
            swift_res = _get_swift_payload(res)
            nlp = swift_res.get("nlp", {}) or {}
            val = swift_res.get("validation", {}) or {}
            ext = res.get("external_validation", {}) or {}
            rml = res.get("risk_ml", {}) or {}
            st.markdown(f"Type : **{swift_res.get('swift_type','?')}**")
            st.metric("Score NLP",  f"{nlp.get('score', 0)}/100")
            st.metric("Validation", f"{val.get('score', 0)}/100")
            if ext:
                ext_lvl = ext.get("risk_level", "CLEAN")
                st.metric("Ext. Risk", f"{EXT_ICONS.get(ext_lvl,'?')} {ext_lvl}")
            if rml and not rml.get("error"):
                ml_lbl = rml.get("risk_label", "?")
                ml_sc  = rml.get("risk_score", 0)
                icons  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                st.metric("🤖 ML Risk", f"{icons.get(ml_lbl,'?')} {ml_lbl} ({ml_sc})")
            n_cont = len(nlp.get("contradictions", []))
            if n_cont:
                st.warning(f"⚠️ {n_cont} contradiction(s)")
        elif mode == "NLP seul (texte)":
            nlp = res.get("nlp", {}) or {}
            c1, c2 = st.columns(2)
            with c1: st.metric("Entités", len((nlp.get("ner") or {}).get("entities", [])))
            with c2: st.metric("Clauses", len(nlp.get("clauses", [])))
        elif mode == "Correction orthographique":
            ch = res.get("changes", 0)
            st.metric("Corrections", ch)
            if ch == 0: st.success("✅ Texte correct")
        else:
            st.metric("Clauses", res.get("count", 0))


# ══════════════════════════════════════════════════════════════════════
# CORPS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

st.title("📑 SWIFT Pipeline v7.3 — Règles + NLP ML + Validation externe + Risk ML")
st.caption(
    "MT700 · MT720 · MT760 · MT790 · MT799  —  "
    "UCP 600 · URDG 758 · ISP 98  —  "
    "OCR · Correction · NER · Clauses · Sémantique · OFAC · GLEIF · EU · UN · "
    "OpenCorp · 🤖 XGBoost Risk + IsolationForest + KMeans + SHAP"
)

uploaded_file = st.file_uploader(
    "📎 Déposez un fichier SWIFT (PDF ou image)",
    type=["pdf", "png", "jpg", "jpeg", "tiff"],
    label_visibility="visible",
)

if uploaded_file:
    if st.button("🚀 Lancer le pipeline complet", width="stretch", type="primary"):
        with st.spinner(
            "Pipeline — OCR · Règles · NLP · UCP · DS · Validation externe · Risk ML…"
        ):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                resp = requests.post(PIPELINE_URL, files=files, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["pipeline_result"] = data
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
            show_nlp_section(nlp)
    elif mode == "Correction orthographique":
        if res.get("success"):
            st.text_area("Texte corrigé", res.get("corrected", ""), height=200)
            _show_spell({
                "total_changes":     res.get("changes", 0),
                "has_ocr_noise":     res.get("has_ocr_noise", False),
                "language_detected": res.get("language_detected", ""),
                "avg_confidence":    res.get("avg_confidence", 0),
                "corrections":       res.get("corrections", []),
            })
    else:
        _show_clauses(res.get("results", []), [])