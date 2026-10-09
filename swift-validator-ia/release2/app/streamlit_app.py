"""
streamlit_app.py
================
Interface Streamlit — Pipeline multi-documents Swift Validator IA

Lancement depuis la racine du projet :
  streamlit run release2/app/streamlit_app.py

Ou depuis n'importe où :
  streamlit run c:/chemin/vers/release2/app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import fitz
import numpy as np
import streamlit as st
from PIL import Image

# ── Chemins & imports projet ──────────────────────────────────────────────────

APP_DIR    = Path(__file__).resolve().parent          # release2/app/
RELEASE2   = APP_DIR.parent                           # release2/
SWIFT_ROOT = RELEASE2.parent                          # swift-validator-ia/
sys.path.insert(0, str(SWIFT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Swift Validator IA — Pipeline",
    page_icon  = "📄",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Badges ─────────────────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 3px 11px; border-radius: 20px;
    font-size: 11px; font-weight: 700; letter-spacing: .6px; text-transform: uppercase;
}
.badge-bl           { background:#1D4ED8; color:#fff; }
.badge-invoice      { background:#065F46; color:#fff; }
.badge-packing_list { background:#92400E; color:#fff; }
.badge-unknown      { background:#4B5563; color:#fff; }

.badge-pdf_native   { background:#1E3A5F; color:#BAE6FD; }
.badge-pdf_scanned  { background:#3B1F00; color:#FEF3C7; }
.badge-image        { background:#1A1A2E; color:#C4B5FD; }

.badge-ok  { background:#064E3B; color:#6EE7B7; }
.badge-warn{ background:#78350F; color:#FDE68A; }
.badge-err { background:#7F1D1D; color:#FCA5A5; }

/* ── Table des champs ────────────────────────────────────────────────────── */
.fields-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #334155;
}
.fields-table tr:nth-child(odd)  { background: #0F172A; }
.fields-table tr:nth-child(even) { background: #1E293B; }
.fields-table tr:hover           { background: #1E3A5F; }

.fields-table td {
    padding: 7px 14px;
    vertical-align: top;
    border-bottom: 1px solid #1E293B;
}
.fields-table td.fn {
    width: 220px;
    white-space: nowrap;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #38BDF8;          /* sky-400 — bright, readable */
    letter-spacing: .3px;
}
.fields-table td.fv {
    color: #F1F5F9;          /* slate-100 — near-white, high contrast */
    font-size: 13px;
    word-break: break-word;
    line-height: 1.5;
}

/* ── Tableau de données (containers / invoice_items…) ───────────────────── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    border: 1px solid #334155;
    border-radius: 8px;
    overflow: hidden;
}
.data-table thead tr { background: #1E3A5F; }
.data-table thead th {
    padding: 8px 12px;
    text-align: left;
    color: #7DD3FC;          /* sky-300 */
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
    text-transform: uppercase;
    white-space: nowrap;
    border-bottom: 2px solid #38BDF8;
}
.data-table tbody tr:nth-child(odd)  { background: #0F172A; }
.data-table tbody tr:nth-child(even) { background: #162032; }
.data-table tbody tr:hover           { background: #1E3A5F; }
.data-table tbody td {
    padding: 6px 12px;
    color: #E2E8F0;          /* slate-200 */
    border-bottom: 1px solid #1E293B;
    vertical-align: top;
}

/* ── Bandeaux alerte ─────────────────────────────────────────────────────── */
.warn-banner {
    background:#78350F33; border-left:3px solid #F59E0B;
    padding:8px 14px; border-radius:0 8px 8px 0;
    font-size:13px; color:#FDE68A; margin:6px 0;
}
.err-banner {
    background:#7F1D1D33; border-left:3px solid #EF4444;
    padding:8px 14px; border-radius:0 8px 8px 0;
    font-size:13px; color:#FCA5A5; margin:6px 0;
}

/* ── Barres de probabilité ───────────────────────────────────────────────── */
.proba-row { display:flex; align-items:center; gap:8px; font-size:12px; margin:3px 0; }
.proba-bar-bg { flex:1; background:#1E293B; border-radius:4px; height:7px; }
.proba-bar    { height:7px; border-radius:4px; }

/* ── En-tête de section ──────────────────────────────────────────────────── */
.fields-header {
    font-size: 11px; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 16px 0 8px 0; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ── Constantes UI ─────────────────────────────────────────────────────────────

TYPE_LABELS = {
    "bl":           "Bill of Lading",
    "invoice":      "Invoice",
    "packing_list": "Packing List",
}
FORMAT_ICONS = {
    "pdf_native":  "📄",
    "pdf_scanned": "🔍",
    "image":       "🖼️",
}
FORMAT_LABELS = {
    "pdf_native":  "PDF natif",
    "pdf_scanned": "PDF scanné",
    "image":       "Image",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def render_pdf_thumbnail(pdf_bytes: bytes, scale: float = 1.2) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(scale, scale)
    pix = doc[0].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    doc.close()
    return Image.fromarray(arr)


def conf_color(conf: float, thr: float) -> str:
    if conf >= thr:      return "#10B981"
    if conf >= thr * .7: return "#F59E0B"
    return "#EF4444"


def badge_html(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{text}</span>'


def proba_bars_html(probas: dict, doc_type: str) -> str:
    colors = {"bl": "#3B82F6", "invoice": "#10B981", "packing_list": "#F59E0B"}
    rows = ""
    for cls, p in sorted(probas.items(), key=lambda x: -x[1]):
        pct  = round(p * 100, 1)
        col  = colors.get(cls, "#6B7280")
        bold = "font-weight:700;" if cls == doc_type else ""
        rows += (
            f'<div class="proba-row">'
            f'  <span style="min-width:90px;{bold}">{TYPE_LABELS.get(cls,cls)}</span>'
            f'  <div class="proba-bar-bg">'
            f'    <div class="proba-bar" style="width:{pct}%;background:{col}"></div>'
            f'  </div>'
            f'  <span style="min-width:40px;text-align:right;{bold}">{pct}%</span>'
            f'</div>'
        )
    return rows


_TIER_BADGE: dict[str, str] = {
    "layout_structure": "background:#064E3B;color:#6EE7B7",
    "layoutlm":         "background:#1E3A5F;color:#BAE6FD",
    "regex":            "background:#3B1F00;color:#FEF3C7",
    "spacy":            "background:#2E1065;color:#DDD6FE",
    "computed":         "background:#1C1917;color:#FCD34D",
    "kv":               "background:#1C1917;color:#D6D3D1",
    # Analyse sémantique avancée
    "semantic":         "background:#1A1A3E;color:#A5B4FC",
    # Zones sémantiques (ZoneExtractor — couleur par zone)
    "zone_header":      "background:#1E3A5F;color:#7DD3FC",
    "zone_parties":     "background:#1A2E1A;color:#86EFAC",
    "zone_cargo":       "background:#2D1A00;color:#FCD34D",
    "zone_totals":      "background:#2D0A0A;color:#FCA5A5",
}

_TABLE_LABELS: dict[str, tuple[str, str]] = {
    "containers":    ("📦", "Conteneurs"),
    "invoice_items": ("📋", "Lignes de facture"),
    "packages":      ("📦", "Colis"),
}

def fields_table_html(fields: dict, tiers: dict | None = None) -> str:
    import html as _html
    if not fields:
        return '<p style="color:#64748B;font-size:13px;padding:8px 0">Aucun champ extrait.</p>'
    tiers = tiers or {}
    rows = ""
    for name, val in sorted(fields.items()):
        tier  = tiers.get(name, "")
        style = _TIER_BADGE.get(tier, "")
        tier_badge = (
            f'<span style="font-size:9px;padding:1px 7px;border-radius:10px;'
            f'margin-left:7px;vertical-align:middle;{style}">{tier}</span>'
        ) if style else ""
        rows += (
            f'<tr>'
            f'<td class="fn">{_html.escape(name)}{tier_badge}</td>'
            f'<td class="fv">{_html.escape(str(val))}</td>'
            f'</tr>'
        )
    return f'<table class="fields-table">{rows}</table>'


def structured_rows_html(rows: list[dict]) -> str:
    """Affiche les lignes de tableau structurées (conteneurs, lignes facture, colis…)."""
    import html as _html
    if not rows:
        return ""
    # Collecter toutes les colonnes dans l'ordre d'apparition
    all_cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_cols.append(k)
                seen.add(k)

    th = "".join(f'<th>{_html.escape(c)}</th>' for c in all_cols)
    tr_rows = ""
    for row in rows:
        tds = "".join(
            f'<td>{_html.escape(str(row.get(c, "")))}</td>'
            for c in all_cols
        )
        tr_rows += f'<tr>{tds}</tr>'
    return (
        f'<div style="overflow-x:auto;margin-top:6px">'
        f'<table class="data-table">'
        f'<thead><tr>{th}</tr></thead>'
        f'<tbody>{tr_rows}</tbody>'
        f'</table></div>'
    )


# ── Cross-validation dashboard ────────────────────────────────────────────────

_RISK_COLORS = {
    "low":      ("#064E3B", "#6EE7B7"),
    "medium":   ("#78350F", "#FDE68A"),
    "high":     ("#7F1D1D", "#FCA5A5"),
    "critical": ("#450A0A", "#FF4444"),
    "unknown":  ("#1E293B", "#94A3B8"),
}
_IMPACT_ICON = {"positive": "✅", "warning": "⚠️", "negative": "❌", "neutral": "ℹ️"}
_NLI_ICON    = {"ENTAILMENT": "✅", "CONTRADICTION": "❌", "NEUTRAL": "➖"}
_NLI_COLOR   = {"ENTAILMENT": "#064E3B", "CONTRADICTION": "#7F1D1D", "NEUTRAL": "#1E293B"}
_STATUS_ICON = {"pass": "✅", "corrected": "🔧", "warning": "⚠️", "fail": "❌", "skipped": "⏭️"}
_STATUS_CLR  = {"pass": "#064E3B", "corrected": "#1E3A5F", "warning": "#78350F",
                "fail": "#7F1D1D", "skipped": "#1C1917"}
_SEV_BADGE   = {
    "critical": "background:#450A0A;color:#FF4444",
    "high":     "background:#7F1D1D;color:#FCA5A5",
    "medium":   "background:#78350F;color:#FDE68A",
    "low":      "background:#1A2E1A;color:#86EFAC",
    "info":     "background:#1E293B;color:#94A3B8",
}


def _score_gauge_html(score: float, label: str = "Cohérence globale") -> str:
    """Jauge horizontale colorée pour le score de cohérence."""
    import html as _h
    pct = round(score * 100, 1)
    if score >= 0.75:
        color = "#10B981"
    elif score >= 0.55:
        color = "#F59E0B"
    else:
        color = "#EF4444"
    return f"""
<div style="margin:4px 0 10px 0">
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#94A3B8;margin-bottom:4px">
    <span>{_h.escape(label)}</span>
    <span style="font-weight:700;color:{color};font-size:16px">{pct}%</span>
  </div>
  <div style="background:#1E293B;border-radius:6px;height:12px;overflow:hidden">
    <div style="width:{pct}%;background:{color};height:100%;border-radius:6px;
         transition:width .4s ease;box-shadow:0 0 8px {color}55"></div>
  </div>
</div>"""


def _component_bar_html(breakdown: dict[str, float]) -> str:
    """Barres par composante avec poids (source de vérité : CrossValidator)."""
    try:
        from release2.services.cross_validator.cross_validator import _SCORE_WEIGHTS
        weights = dict(_SCORE_WEIGHTS)
    except Exception:
        weights = {"semantic": 0.15, "nli": 0.20, "rules": 0.50, "table": 0.00, "party": 0.15}
    labels  = {"semantic": "Semantic Embedding", "nli": "NLI Inference",
                "rules": "Business Rules", "table": "Table Alignment", "party": "Party Matching"}
    colors  = {"semantic": "#6366F1", "nli": "#EC4899", "rules": "#F59E0B",
                "table": "#10B981",   "party": "#3B82F6"}
    rows = ""
    for k, w in weights.items():
        v   = breakdown.get(k, 0.0)
        pct = round(v * 100, 1)
        col = colors.get(k, "#6B7280")
        lbl = labels.get(k, k)
        rows += f"""
<div style="margin:5px 0">
  <div style="display:flex;justify-content:space-between;font-size:11px;color:#CBD5E1;margin-bottom:3px">
    <span>{lbl} <span style="color:#64748B;font-size:10px">({int(w*100)}%)</span></span>
    <span style="font-weight:600;color:{col}">{pct}%</span>
  </div>
  <div style="background:#1E293B;border-radius:4px;height:7px">
    <div style="width:{pct}%;background:{col};height:100%;border-radius:4px"></div>
  </div>
</div>"""
    return f'<div style="padding:6px 0">{rows}</div>'


def _rule_card_html(r: dict) -> str:
    import html as _h
    status = r.get("status", "skipped")
    sev    = r.get("severity", "info")
    bg     = _STATUS_CLR.get(status, "#1E293B")
    icon   = _STATUS_ICON.get(status, "•")
    sev_style = _SEV_BADGE.get(sev, "")
    rule   = _h.escape(r.get("rule", ""))
    msg    = _h.escape(r.get("message", ""))
    field  = _h.escape(r.get("field", "") or "")
    pair   = _h.escape(r.get("doc_pair", "") or "")
    orig   = _h.escape(str(r.get("original", "") or ""))
    detail = f'<div style="font-size:10px;color:#94A3B8;margin-top:3px">{orig}</div>' if orig else ""
    pair_badge = (
        f'<span style="font-size:9px;padding:1px 6px;border-radius:8px;background:#1E293B;'
        f'color:#7DD3FC;margin-left:6px">{pair}</span>'
    ) if pair else ""
    return f"""
<div style="background:{bg}33;border-left:3px solid {'#10B981' if status=='pass' else '#EF4444' if status=='fail' else '#F59E0B'};
     padding:8px 12px;border-radius:0 8px 8px 0;margin:4px 0">
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
    <span>{icon}</span>
    <code style="font-size:11px;color:#7DD3FC">{rule}</code>
    {f'<code style="font-size:10px;color:#94A3B8">{field}</code>' if field else ""}
    <span style="font-size:9px;padding:1px 6px;border-radius:8px;{sev_style}">{sev}</span>
    {pair_badge}
  </div>
  <div style="font-size:12px;color:#E2E8F0;margin-top:4px">{msg}</div>
  {detail}
</div>"""


def _nli_card_html(r: dict) -> str:
    import html as _h
    label    = r.get("nli_label", "NEUTRAL")
    score    = r.get("score", 0.0)
    desc     = _h.escape(r.get("description", ""))
    category = _h.escape(r.get("category", ""))
    premise  = _h.escape(r.get("premise", ""))
    hypo     = _h.escape(r.get("hypothesis", ""))
    icon     = _NLI_ICON.get(label, "➖")
    bg       = _NLI_COLOR.get(label, "#1E293B")
    all_sc   = r.get("all_scores", {})
    score_bars = ""
    for lbl_k, lbl_sc in all_sc.items():
        col = {"ENTAILMENT": "#10B981", "CONTRADICTION": "#EF4444", "NEUTRAL": "#94A3B8"}.get(lbl_k, "#6B7280")
        score_bars += (
            f'<span style="margin-right:8px;font-size:10px;color:{col}">'
            f'{lbl_k[:3]} {lbl_sc:.0%}</span>'
        )
    return f"""
<div style="background:{bg}22;border:1px solid {bg};border-radius:8px;padding:10px 14px;margin:4px 0">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
    <span style="font-size:18px">{icon}</span>
    <span style="font-weight:700;color:#F1F5F9;font-size:13px">{desc}</span>
    <span style="font-size:10px;color:#64748B">[{category}]</span>
    <span style="margin-left:auto;font-weight:700;color:#F1F5F9">{score:.0%}</span>
  </div>
  <div style="font-size:10px;color:#94A3B8;margin-bottom:4px">{score_bars}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px">
    <div style="background:#0F172A;border-radius:6px;padding:8px">
      <div style="font-size:9px;color:#38BDF8;font-weight:700;margin-bottom:3px">PREMISE</div>
      <div style="font-size:11px;color:#CBD5E1;line-height:1.4">{premise}</div>
    </div>
    <div style="background:#0F172A;border-radius:6px;padding:8px">
      <div style="font-size:9px;color:#A78BFA;font-weight:700;margin-bottom:3px">HYPOTHESIS</div>
      <div style="font-size:11px;color:#CBD5E1;line-height:1.4">{hypo}</div>
    </div>
  </div>
</div>"""


def _semantic_score_html(scores: dict[str, float]) -> str:
    import html as _h
    rows = ""
    for key, val in sorted(scores.items(), key=lambda x: -x[1]):
        if ":" in key:
            pair, sub = key.split(":", 1)
            label = f"{pair} <span style='color:#64748B;font-size:10px'>[{sub}]</span>"
        else:
            label = _h.escape(key)
        pct = round(val * 100, 1)
        col = "#10B981" if val >= 0.75 else ("#F59E0B" if val >= 0.50 else "#EF4444")
        rows += f"""
<div style="display:flex;align-items:center;gap:8px;margin:4px 0">
  <div style="min-width:220px;font-size:12px;color:#CBD5E1">{label}</div>
  <div style="flex:1;background:#1E293B;border-radius:4px;height:8px">
    <div style="width:{pct}%;background:{col};height:100%;border-radius:4px"></div>
  </div>
  <span style="min-width:45px;text-align:right;font-weight:700;color:{col};font-size:12px">{pct}%</span>
</div>"""
    return f'<div style="padding:6px 0">{rows}</div>' if rows else ""


def _explanation_card_html(e: dict) -> str:
    import html as _h
    impact = e.get("impact", "neutral")
    icon   = _IMPACT_ICON.get(impact, "ℹ️")
    msg    = _h.escape(e.get("message", ""))
    sev    = e.get("severity", "info")
    etype  = _h.escape(e.get("type", ""))
    cat    = _h.escape(e.get("category", ""))
    sev_style = _SEV_BADGE.get(sev, "")
    bg = {"positive": "#064E3B", "warning": "#78350F",
          "negative": "#7F1D1D", "neutral": "#1E293B"}.get(impact, "#1E293B")
    return f"""
<div style="background:{bg}22;border-left:3px solid {'#10B981' if impact=='positive' else '#EF4444' if impact=='negative' else '#F59E0B' if impact=='warning' else '#475569'};
     padding:7px 11px;border-radius:0 6px 6px 0;margin:3px 0">
  <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
    <span>{icon}</span>
    <span style="font-size:9px;padding:1px 6px;border-radius:8px;background:#1E293B;color:#7DD3FC">{etype}</span>
    <span style="font-size:9px;padding:1px 6px;border-radius:8px;{sev_style}">{sev}</span>
    {f'<span style="font-size:9px;color:#64748B">{cat}</span>' if cat else ""}
  </div>
  <div style="font-size:12px;color:#E2E8F0;margin-top:4px">{msg}</div>
</div>"""


def _render_cross_report(cross_report: dict) -> None:
    """Affiche la section validation croisée complète."""
    st.markdown("### 🔗 Validation Croisée — AI Document Intelligence")

    score       = cross_report.get("final_score")
    risk        = cross_report.get("risk_level", "unknown")
    breakdown   = cross_report.get("score_breakdown", {})
    rule_res    = cross_report.get("rule_results", [])
    semantic_sc = cross_report.get("semantic_scores", {})
    nli_res     = cross_report.get("nli_results", [])
    tbl_matches = cross_report.get("table_matches", [])
    explanations = cross_report.get("explanations", [])

    # ── Bandeau risque ──────────────────────────────────────────────────────
    risk_bg, risk_fg = _RISK_COLORS.get(risk, _RISK_COLORS["unknown"])
    risk_icons = {"low": "🟢", "medium": "🟡", "high": "🔴", "critical": "🚨", "unknown": "⚪"}
    st.markdown(
        f'<div style="background:{risk_bg};color:{risk_fg};padding:12px 20px;'
        f'border-radius:10px;display:flex;align-items:center;gap:12px;margin-bottom:8px">'
        f'<span style="font-size:24px">{risk_icons.get(risk,"⚪")}</span>'
        f'<div>'
        f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:1px;opacity:.8">Niveau de risque</div>'
        f'<div style="font-size:20px;font-weight:800">{str(risk).upper()}</div>'
        f'</div>'
        f'<div style="margin-left:auto;text-align:right">'
        f'<div style="font-size:11px;opacity:.8">Score de cohérence</div>'
        f'<div style="font-size:28px;font-weight:900">{f"{score:.0%}" if score is not None else "—"}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Métriques rapides ───────────────────────────────────────────────────
    n_fail    = sum(1 for r in rule_res if r.get("status") == "fail")
    n_warn    = sum(1 for r in rule_res if r.get("status") == "warning")
    n_pass    = sum(1 for r in rule_res if r.get("status") == "pass")
    n_contra  = sum(1 for n in nli_res if n.get("nli_label") == "CONTRADICTION")
    n_entail  = sum(1 for n in nli_res if n.get("nli_label") == "ENTAILMENT")

    mc = st.columns(5)
    mc[0].metric("✅ Règles OK",           n_pass)
    mc[1].metric("⚠️ Avertissements",      n_warn,  delta=f"+{n_warn}"  if n_warn  else None, delta_color="inverse")
    mc[2].metric("❌ Échecs",              n_fail,  delta=f"+{n_fail}"  if n_fail  else None, delta_color="inverse")
    mc[3].metric("✅ NLI Entailments",     n_entail)
    mc[4].metric("❌ NLI Contradictions",  n_contra, delta=f"+{n_contra}" if n_contra else None, delta_color="inverse")

    st.divider()

    # ── Décomposition du score + jauge ─────────────────────────────────────
    col_gauge, col_break = st.columns([1, 1])
    with col_gauge:
        st.markdown('<div class="fields-header">Score global</div>', unsafe_allow_html=True)
        if score is not None:
            st.markdown(_score_gauge_html(score), unsafe_allow_html=True)
        else:
            st.info("Score non calculé.")

    with col_break:
        st.markdown('<div class="fields-header">Décomposition par composante</div>', unsafe_allow_html=True)
        if breakdown:
            st.markdown(_component_bar_html(breakdown), unsafe_allow_html=True)

    st.divider()

    # ── Onglets principaux ─────────────────────────────────────────────────
    tabs = st.tabs([
        "📋 Règles métier",
        "🔵 Sémantique",
        "🤖 NLI Inference",
        "📊 Tables",
        "💡 Explications XAI",
        "📄 Textes métier",
    ])

    # ── Tab 0 : Règles ──────────────────────────────────────────────────────
    with tabs[0]:
        if rule_res:
            # Filtres
            statuses = sorted({r.get("status", "") for r in rule_res})
            sel_status = st.multiselect(
                "Filtrer par statut", statuses,
                default=statuses, key="cross_rule_filter",
            )
            filtered = [r for r in rule_res if r.get("status") in sel_status]
            st.caption(f"{len(filtered)} / {len(rule_res)} règle(s) affichée(s)")
            for r in filtered:
                st.markdown(_rule_card_html(r), unsafe_allow_html=True)
        else:
            st.info("Aucune règle inter-documents calculée.")

    # ── Tab 1 : Sémantique ──────────────────────────────────────────────────
    with tabs[1]:
        if semantic_sc:
            st.caption(
                "Similarité cosine entre les représentations textuelles métier "
                "(Sentence Transformers MiniLM-L6-v2)."
            )
            st.markdown(_semantic_score_html(semantic_sc), unsafe_allow_html=True)

            # Tableau synthétique des scores globaux
            global_sc = {k: v for k, v in semantic_sc.items() if ":" not in k}
            if global_sc:
                st.divider()
                st.caption("Scores globaux (narratif complet)")
                st.table([
                    {"Paire de documents": k, "Similarité": f"{v:.1%}"}
                    for k, v in sorted(global_sc.items(), key=lambda x: -x[1])
                ])
        else:
            st.info("Analyse sémantique non disponible (sentence-transformers non installé).")

    # ── Tab 2 : NLI ─────────────────────────────────────────────────────────
    with tabs[2]:
        if nli_res:
            # Filtres NLI
            categories = sorted({n.get("category", "global") for n in nli_res})
            sel_cat = st.multiselect(
                "Filtrer par catégorie", categories,
                default=categories, key="cross_nli_filter",
            )
            filtered_nli = [n for n in nli_res if n.get("category") in sel_cat]
            st.caption(
                f"{len(filtered_nli)} paires NLI — modèle : DeBERTa v3 / fallback heuristique"
            )
            for r in filtered_nli:
                st.markdown(_nli_card_html(r), unsafe_allow_html=True)
        else:
            st.info("Aucun résultat NLI disponible.")

    # ── Tab 3 : Tables ──────────────────────────────────────────────────────
    with tabs[3]:
        if tbl_matches:
            for match in tbl_matches:
                pair = match.get("doc_pair", "")
                st.markdown(f"**{pair}**")

                if "invoice" in pair and "packing" in pair:
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Lignes Invoice",   match.get("invoice_rows", 0))
                    col_b.metric("Lignes PL",        match.get("pl_rows", 0))
                    col_c.metric("Lignes matchées",  match.get("matched_rows", 0))
                    rate = match.get("match_rate", 0)
                    st.markdown(_score_gauge_html(rate, "Taux d'alignement"), unsafe_allow_html=True)

                    unmatched = match.get("unmatched_rows", [])
                    if unmatched:
                        with st.expander(f"⚠️ {len(unmatched)} ligne(s) sans correspondance"):
                            st.markdown(structured_rows_html(
                                [{
                                    "Invoice": str(u.get("invoice_row", {})),
                                    "Meilleur match PL": str(u.get("best_pl_match", {})),
                                    "Score": str(u.get("best_score", "")),
                                } for u in unmatched]
                            ), unsafe_allow_html=True)

                elif "bl" in pair and "packing" in pair:
                    col_a, col_b = st.columns(2)
                    col_a.markdown(
                        f"**BL containers :** `{match.get('bl_containers', [])}`"
                    )
                    col_b.markdown(
                        f"**PL containers :** `{match.get('pl_containers', [])}`"
                    )
                    common = match.get("common_containers", [])
                    rate   = match.get("alignment_rate", 0)
                    if common:
                        st.success(f"✅ Conteneurs communs : {common}")
                    else:
                        st.error("❌ Aucun conteneur en commun BL ↔ PL")
                    st.markdown(_score_gauge_html(rate, "Taux d'alignement conteneurs"), unsafe_allow_html=True)

                st.divider()
        else:
            st.info("Aucun alignement de tables calculé.")

    # ── Tab 4 : XAI Explications ────────────────────────────────────────────
    with tabs[4]:
        if explanations:
            # Grouper par impact
            for impact_group, impact_label in [
                ("negative",  "❌  Problèmes détectés"),
                ("warning",   "⚠️  Avertissements"),
                ("positive",  "✅  Points de cohérence"),
                ("neutral",   "ℹ️  Informations"),
            ]:
                group = [e for e in explanations if e.get("impact") == impact_group]
                if not group:
                    continue
                st.markdown(f"**{impact_label}** ({len(group)})")
                for e in group:
                    st.markdown(_explanation_card_html(e), unsafe_allow_html=True)
                st.markdown("")
        else:
            st.info("Aucune explication XAI générée.")

    # ── Tab 5 : Textes métier ───────────────────────────────────────────────
    with tabs[5]:
        doc_entries = cross_report.get("documents", [])
        if doc_entries:
            for doc_entry in doc_entries:
                doc_type = doc_entry.get("doc_type", "")
                biz_text = doc_entry.get("business_text", "")
                type_label = {"bl": "Bill of Lading", "invoice": "Invoice",
                              "packing_list": "Packing List"}.get(doc_type, doc_type.upper())
                with st.expander(f"📄 {type_label} — Texte métier canonique", expanded=False):
                    st.markdown(
                        f'<div style="background:#0F172A;border-radius:8px;padding:14px;'
                        f'font-size:13px;color:#E2E8F0;line-height:1.7;white-space:pre-wrap">'
                        f'{biz_text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Textes métier non disponibles.")


# ── Chargement pipeline (mis en cache, modèles chargés une seule fois) ────────

@st.cache_resource(show_spinner="⏳  Chargement des modèles IA — ResNet-18 + LayoutLMv3…")
def load_pipeline(ocr_lang_key: str, ocr_gpu: bool):
    from release2.services.pipeline import DocumentPipeline
    return DocumentPipeline(
        confidence_threshold = 0.0,                    # géré dans l'UI
        ocr_languages        = ocr_lang_key.split(","),
        ocr_gpu              = ocr_gpu,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    st.markdown("**Classification**")
    threshold = st.slider(
        "Seuil de confiance",
        min_value=0.50, max_value=1.00, value=0.80, step=0.05,
        help="En dessous de ce seuil, une alerte est affichée sur le document",
    )
    st.caption(f"Alerte si confiance < **{threshold:.0%}**")

    st.divider()
    st.markdown("**OCR** *(PDFs scannés & images)*")
    ocr_languages = st.multiselect(
        "Langues détectées",
        options  = ["en", "fr", "ar", "zh", "de", "es", "it"],
        default  = ["en"],
        help     = "Langues passées à EasyOCR",
    )
    ocr_gpu = st.checkbox(
        "GPU pour EasyOCR",
        value = False,
        help  = "Désactivé par défaut. Activez uniquement si CUDA fonctionne bien sur votre machine.",
    )

    st.divider()
    st.caption("**Swift Validator IA**")
    st.caption("Release 2 — ResNet-18 · LayoutLMv3 · Zone · Sémantique")
    st.caption(f"Modèles chargés depuis :\n`{RELEASE2 / 'models'}`")

    st.divider()
    if st.button("🔄 Recharger le pipeline", help="Vide le cache et recharge les modèles"):
        st.cache_resource.clear()
        st.success("Cache vidé — relancez l'analyse.")
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# En-tête principal
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# 📄 Swift Validator IA — Pipeline Documentaire")
st.markdown(
    "Classifiez et extrayez les champs de vos documents commerciaux "
    "(**Bill of Lading**, **Invoice**, **Packing List**) "
    "— jusqu'à **3 documents simultanément**."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Zone d'upload
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📂 Documents à analyser")

uploaded_files = st.file_uploader(
    label    = "Déposez 1 à 3 documents (PDF natif, PDF scanné, PNG, JPG, TIFF…)",
    type     = ["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
    accept_multiple_files = True,
    help     = "Maximum 3 fichiers. Formats : PDF, PNG, JPG, TIFF, BMP",
)

if len(uploaded_files) > 3:
    st.warning("⚠️ Maximum 3 documents — seuls les 3 premiers seront traités.")
    uploaded_files = uploaded_files[:3]

# Aperçus côte à côte
if uploaded_files:
    prev_cols = st.columns(len(uploaded_files))
    for col, uf in zip(prev_cols, uploaded_files):
        with col:
            ext = Path(uf.name).suffix.lower()
            st.caption(f"**{uf.name}**  ({uf.size // 1024} Ko)")
            try:
                raw = uf.read()
                if ext == ".pdf":
                    st.image(render_pdf_thumbnail(raw), width="stretch")
                else:
                    st.image(Image.open(uf), width="stretch")
                uf.seek(0)
            except Exception:
                st.info("Aperçu non disponible")
    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Bouton d'analyse
# ─────────────────────────────────────────────────────────────────────────────

run = st.button(
    "🚀  Analyser les documents",
    type             = "primary",
    disabled         = not uploaded_files,
    use_container_width = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Traitement & affichage des résultats
# ─────────────────────────────────────────────────────────────────────────────

if run and uploaded_files:
    ocr_lang_key = ",".join(sorted(ocr_languages)) if ocr_languages else "en"

    # ── Sauvegarde temporaire ─────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = []
        for uf in uploaded_files:
            dest = Path(tmpdir) / uf.name
            dest.write_bytes(uf.read())
            tmp_paths.append(dest)

        pipeline  = load_pipeline(ocr_lang_key, ocr_gpu)
        progress  = st.progress(0, text="Analyse en cours…")

        with st.spinner("Classification + extraction — veuillez patienter…"):
            try:
                result = pipeline.process(tmp_paths)
                progress.progress(100, text="Analyse terminée ✓")
            except Exception as exc:
                st.error(f"**Erreur pipeline :** {exc}")
                st.stop()

    # ── Résumé global ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Résumé")

    s = result["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents",      s["total_documents"])
    c2.metric("Traités",        s["processed"])
    c3.metric("Erreurs",        s["failed"],
              delta=f"+{s['failed']}" if s["failed"] else None,
              delta_color="inverse")
    c4.metric("Durée totale",   f"{s['processing_time_s']} s")

    st.download_button(
        label     = "⬇️  Télécharger les résultats JSON",
        data      = json.dumps(result, ensure_ascii=False, indent=2),
        file_name = "pipeline_results.json",
        mime      = "application/json",
    )

    # ── Validation croisée / cross validator ───────────────────────────────────
    cross_report = result.get("cross_report") or {}
    if cross_report:
        st.divider()
        _render_cross_report(cross_report)

    st.divider()
    st.markdown("### 📋 Résultats par document")

    docs = result["documents"]
    tab_labels = [f"{'✓' if d['status']=='success' else '✗'}  {d['source']}"
                  for d in docs]
    tabs = st.tabs(tab_labels)

    for tab, doc in zip(tabs, docs):
        with tab:
            if doc["status"] == "error":
                st.markdown(
                    f'<div class="err-banner">✗ Erreur : {doc.get("error","?")}</div>',
                    unsafe_allow_html=True,
                )
                continue

            # ── En-tête document ──────────────────────────────────────────────
            conf       = doc["confidence"]
            doc_type   = doc.get("doc_type", "unknown")
            input_type = doc.get("input_type", "")
            confident  = conf >= threshold

            hcol1, hcol2, hcol3 = st.columns([3, 2, 2])

            with hcol1:
                type_label = TYPE_LABELS.get(doc_type, doc_type.upper())
                st.markdown(
                    badge_html(type_label, f"badge-{doc_type}") + "&nbsp;" +
                    badge_html(FORMAT_LABELS.get(input_type, input_type),
                               f"badge-{input_type}") + "&nbsp;" +
                    badge_html("Fiable" if confident else "Vérifier",
                               "badge-ok" if confident else "badge-warn"),
                    unsafe_allow_html=True,
                )

            with hcol2:
                color = conf_color(conf, threshold)
                st.markdown(
                    f"**Confiance**&nbsp;&nbsp;"
                    f"<span style='color:{color};font-size:1.3rem;font-weight:700'>"
                    f"{conf:.1%}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(conf)

            with hcol3:
                st.markdown(
                    f"**Pages :** {doc['n_pages']}  \n"
                    f"**Champs :** {len(doc.get('fields', {}))}  \n"
                    f"**Durée :** {doc['processing_time_s']} s"
                )

            # ── Avertissements ────────────────────────────────────────────────
            for w in doc.get("warnings", []):
                st.markdown(
                    f'<div class="warn-banner">⚠ {w}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── Deux colonnes : probas | champs ───────────────────────────────
            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.markdown(
                    '<div class="fields-header">Probabilités de classification</div>',
                    unsafe_allow_html=True,
                )
                probas = doc.get("class_probas", {})
                if probas:
                    st.markdown(
                        proba_bars_html(probas, doc_type),
                        unsafe_allow_html=True,
                    )

                st.divider()

                # Infos fichier
                st.markdown(
                    '<div class="fields-header">Fichier source</div>',
                    unsafe_allow_html=True,
                )
                icon = FORMAT_ICONS.get(input_type, "📄")
                st.markdown(
                    f"{icon} `{doc['source']}`  \n"
                    f"Format : **{FORMAT_LABELS.get(input_type, input_type)}**"
                )

            with right_col:
                fields = doc.get("fields", {})
                tiers  = doc.get("extraction_tiers", {})
                n_layout   = sum(1 for v in tiers.values() if v == "layout_structure")
                n_computed = sum(1 for v in tiers.values() if v == "computed")
                n_semantic = sum(1 for v in tiers.values() if v == "semantic")
                n_zone     = sum(1 for v in tiers.values() if str(v).startswith("zone_"))
                extras = []
                if n_layout:
                    extras.append(f'<span style="font-size:9px;padding:1px 6px;border-radius:10px;background:#064E3B;color:#6EE7B7">+{n_layout} structure</span>')
                if n_zone:
                    extras.append(f'<span style="font-size:9px;padding:1px 6px;border-radius:10px;background:#1E3A5F;color:#7DD3FC">+{n_zone} zone</span>')
                if n_semantic:
                    extras.append(f'<span style="font-size:9px;padding:1px 6px;border-radius:10px;background:#1A1A3E;color:#A5B4FC">+{n_semantic} sémantique</span>')
                if n_computed:
                    extras.append(f'<span style="font-size:9px;padding:1px 6px;border-radius:10px;background:#1C1917;color:#FCD34D">+{n_computed} calculé</span>')
                label_extra = " " + " ".join(extras) if extras else ""
                st.markdown(
                    f'<div class="fields-header">Champs extraits ({len(fields)}){label_extra}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    fields_table_html(fields, tiers),
                    unsafe_allow_html=True,
                )

            # ── Tables nommées (containers / invoice_items / packages) ────────
            doc_tables = doc.get("tables", {})
            if doc_tables:
                st.divider()
                for table_key, table_rows in doc_tables.items():
                    if not table_rows:
                        continue
                    icon, label = _TABLE_LABELS.get(table_key, ("📊", table_key))
                    with st.expander(
                        f"{icon} {label} ({len(table_rows)} ligne{'s' if len(table_rows) > 1 else ''})",
                        expanded=True,
                    ):
                        st.markdown(
                            structured_rows_html(table_rows),
                            unsafe_allow_html=True,
                        )

            # ── Lignes structurées brutes (debug) ─────────────────────────────
            structured_rows = doc.get("structured_rows", [])
            if structured_rows and not doc_tables:
                st.divider()
                with st.expander(f"📊 Lignes de tableau structurées ({len(structured_rows)} lignes)"):
                    st.markdown(
                        structured_rows_html(structured_rows),
                        unsafe_allow_html=True,
                    )

            # ── Détail par page ───────────────────────────────────────────────
            pages = doc.get("pages", [])
            if len(pages) > 1:
                st.divider()
                with st.expander(f"Détail par page ({len(pages)} pages)"):
                    for page in pages:
                        st.markdown(f"**Page {page['page']}**")
                        pf = page.get("fields", {})
                        if pf:
                            st.markdown(
                                fields_table_html(pf, tiers),
                                unsafe_allow_html=True,
                            )
                        else:
                            st.caption("Aucun champ sur cette page.")
                        sr = page.get("structured_rows", [])
                        if sr:
                            st.markdown(structured_rows_html(sr), unsafe_allow_html=True)

    # ── JSON brut ─────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("🔎 Voir le JSON complet"):
        st.json(result)
