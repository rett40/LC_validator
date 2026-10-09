"""
swift_cross_field_coherence.py  —  v1.0
=========================================
Analyse de cohérence inter-champs d'un message SWIFT MT700.

Principe : valider les relations sémantiques ENTRE les champs,
pas chaque champ isolément.

5 axes de cohérence :
  AXE A — Goods (:45A:) ↔ Documents (:46A:)
           NLI : "ces documents couvrent-ils ces marchandises ?"
  AXE B — Goods (:45A:) ↔ Conditions (:47A:)
           NLI : "les conditions sont-elles cohérentes avec les marchandises ?"
  AXE C — Amount (:32B:) ↔ Goods (:45A:)
           Règle : prix unitaire aberrant pour le type de bien
  AXE D — Tags (:43T:/:43P:/:49:) ↔ Toutes les clauses (:47A:) étendu
           NLI étendu : détection générale de contradictions tag/clause
  AXE E — Incoterm (:45A:) ↔ Ensemble des champs
           Règle : incoterm implique transport doc + assurance + port type

Technique :
  NLI (primaire) + Regex (renforcement/fallback)
  Compatible avec OfflineNLIPredictor (local_files_only, jamais de download runtime)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ── EntityAlignmentChecker (optionnel) ───────────────────────────────
try:
    from .swift_entity_alignment import EntityAlignmentChecker
except Exception:
    try:
        from swift_entity_alignment import EntityAlignmentChecker  # type: ignore
    except Exception:
        EntityAlignmentChecker = None  # type: ignore

# ── Helper local ──────────────────────────────────────────────────────

def _get(entities: Any, key: str, default: str = "") -> str:
    if entities is None:
        return default
    if isinstance(entities, dict):
        return str(entities.get(key) or default)
    return str(getattr(entities, key, None) or default)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CrossFieldIssue:
    """Une incohérence détectée entre deux champs MT700."""
    fields:     list[str]   # ex: [":45A:", ":46A:"]
    code:       str
    technique:  str         # "NLI" | "NLI+regex" | "regex" | "rule"
    severity:   str         # "error" | "warning" | "info"
    message:    str
    suggestion: str
    lc_value:   str = ""    # valeur trouvée dans le premier champ
    doc_value:  str = ""    # valeur trouvée dans le second champ
    confidence: float = 1.0
    ucp_ref:    str = ""

    def to_dict(self) -> dict:
        return {
            "fields":     self.fields,
            "code":       self.code,
            "technique":  self.technique,
            "severity":   self.severity,
            "message":    self.message,
            "suggestion": self.suggestion,
            "lc_value":   self.lc_value[:200] if self.lc_value else "",
            "doc_value":  self.doc_value[:200] if self.doc_value else "",
            "confidence": round(self.confidence, 3),
            "ucp_ref":    self.ucp_ref,
        }


@dataclass
class CrossFieldCoherenceResult:
    """Résultat complet de l'analyse inter-champs."""
    issues:           list[CrossFieldIssue] = field(default_factory=list)
    coherence_score:  float = 100.0   # 0-100
    verdict:          str   = "COHERENT"   # COHERENT | PARTIALLY_COHERENT | INCOHERENT
    processing_ms:    int   = 0
    axes_checked:     list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "issues":          [i.to_dict() for i in self.issues],
            "coherence_score": round(self.coherence_score, 1),
            "verdict":         self.verdict,
            "processing_ms":   self.processing_ms,
            "axes_checked":    self.axes_checked,
            "n_errors":        sum(1 for i in self.issues if i.severity == "error"),
            "n_warnings":      sum(1 for i in self.issues if i.severity == "warning"),
        }


# ══════════════════════════════════════════════════════════════════════
# AXE A — GOODS ↔ DOCUMENTS
# ══════════════════════════════════════════════════════════════════════

class _AxeA_GoodsDocs:
    """NLI-based : les documents requis couvrent-ils les marchandises ?"""

    # (goods_pattern, required_doc_keywords, doc_label, nli_hypothesis)
    _RULES: list[tuple] = [
        (
            re.compile(r'\b(food|fish|seafood|meat|poultry|dairy|milk|cheese|frozen|'
                       r'alimentaire|poisson|viande|lait|fromage|surgel)\b', re.I),
            ["SANITARY", "HEALTH CERT", "PHYTOSANITARY", "VETERINARY", "FUMIGATION"],
            "certificat sanitaire/phytosanitaire",
            "A sanitary or health certificate is required for these goods.",
        ),
        (
            re.compile(r'\b(chemical|acid|hazardous|dangerous goods|toxic|explosive|'
                       r'chimique|acide|dangereux|toxique|explosif)\b', re.I),
            ["MSDS", "SDS", "DANGEROUS GOODS", "IMDG", "MATERIAL SAFETY"],
            "MSDS / certificat IMDG",
            "A material safety data sheet or dangerous goods certificate is required.",
        ),
        (
            re.compile(r'\b(pharmaceutical|medicine|drug|vaccine|tablet|capsule|'
                       r'pharmaceutique|medicament|vaccin)\b', re.I),
            ["GMP", "FDA", "CE MARKING", "ANALYSIS CERT", "HEALTH AUTHORITY"],
            "certificat GMP / FDA",
            "A pharmaceutical quality certificate is required for these goods.",
        ),
        (
            re.compile(r'\b(steel|iron|aluminium|copper|alloy|pipe|tube|bar|coil|'
                       r'acier|fer|aluminium|cuivre|alliage|tuyau)\b', re.I),
            ["MILL CERT", "TEST CERT", "MATERIAL CERT", "QUALITY CERT"],
            "certificat de matière (Mill Certificate)",
            "A mill certificate or material test certificate is required.",
        ),
        (
            re.compile(r'\b(agricultural|crop|wheat|grain|cotton|fertilizer|seed|'
                       r'agricole|ble|cereale|coton|engrais|graine)\b', re.I),
            ["PHYTOSANITARY", "FUMIGATION", "QUARANTINE CERT"],
            "certificat phytosanitaire / fumigation",
            "A phytosanitary or fumigation certificate is required for agricultural goods.",
        ),
    ]

    @classmethod
    def check(
        cls,
        goods: str,
        docs: str,
        nli: Any = None,
    ) -> list[CrossFieldIssue]:
        issues = []
        docs_up = docs.upper()
        goods_up = goods.upper()

        for goods_pat, expected_kws, doc_label, nli_hypothesis in cls._RULES:
            if not goods_pat.search(goods):
                continue

            found_regex = any(kw in docs_up for kw in expected_kws)
            nli_confirms_missing = False
            confidence = 1.0

            # Step 1: NLI — confirme-t-il que le document est requis ?
            if not found_regex and nli is not None and getattr(nli, "available", False):
                try:
                    premise = f"Documents required: {docs[:400]}"
                    score = nli.predict(premise, nli_hypothesis)
                    if score is not None:
                        # Si entailment faible → document absent confirmé par NLI
                        nli_confirms_missing = score.entailment < 0.35
                        confidence = max(0.5, 1.0 - score.entailment)
                except Exception:
                    pass

            if not found_regex:
                technique = "NLI+regex" if nli_confirms_missing else "regex"
                issues.append(CrossFieldIssue(
                    fields=[":45A:", ":46A:"],
                    code=f"GOODS_DOCS_MISSING_{goods_pat.pattern[:20].upper().replace(' ', '_')}",
                    technique=technique,
                    severity="warning",
                    message=(
                        f"Marchandises nécessitant {doc_label} détectées dans :45A:, "
                        f"mais aucun {doc_label} trouvé dans :46A:."
                    ),
                    suggestion=f"Ajouter dans :46A: : {' / '.join(expected_kws[:2])}",
                    lc_value=goods[:150],
                    doc_value=docs[:150],
                    confidence=round(confidence, 3),
                    ucp_ref="Art.14 UCP 600 / ISBP 745",
                ))

        return issues


# ══════════════════════════════════════════════════════════════════════
# AXE B — GOODS ↔ CONDITIONS
# ══════════════════════════════════════════════════════════════════════

class _AxeB_GoodsConditions:
    """NLI-based : les conditions (:47A:) sont-elles cohérentes avec les marchandises (:45A:)?"""

    _RULES: list[tuple] = [
        (
            re.compile(r'\b(perishable|frozen|refrigerat|fresh|cold.chain|'
                       r'périssable|surgel|réfrigér|frais|chaîne.du.froid)\b', re.I),
            re.compile(r'\b(temperature|cold|refrigerat|frozen|reefer|'
                       r'température|froid|réfrigér|surgel|reefer)\b', re.I),
            ":45A: mentionne des marchandises périssables mais :47A: ne contient aucune condition de température",
            "Ajouter dans :47A: les conditions de transport frigorifique (ex: 'CARGO TO BE SHIPPED AT -18°C')",
            "COLD_CHAIN_NO_TEMP_CONDITION",
            "warning",
        ),
        (
            re.compile(r'\b(bulk|vrac|loose)\b', re.I),
            re.compile(r'\b(packing|emballage|container|contenant|bulk|vrac)\b', re.I),
            ":45A: mentionne des marchandises en vrac mais :47A: ne précise pas les conditions d'emballage/transport",
            "Préciser dans :47A: les conditions d'emballage pour marchandises en vrac",
            "BULK_NO_PACKING_CONDITION",
            "info",
        ),
        (
            re.compile(r'\b(pharmaceutical|medicament|vaccin|medicine)\b', re.I),
            re.compile(r'\b(gmp|fda|batch|lot.number|expiry|shelf.life|'
                       r'numéro.de.lot|date.de.péremption)\b', re.I),
            ":45A: mentionne des produits pharmaceutiques mais :47A: ne contient pas de conditions de traçabilité",
            "Ajouter dans :47A: : numéro de lot, date de péremption, conditions de stockage",
            "PHARMA_NO_TRACEABILITY_CONDITION",
            "warning",
        ),
    ]

    @classmethod
    def check(
        cls,
        goods: str,
        conditions: str,
        nli: Any = None,
    ) -> list[CrossFieldIssue]:
        issues = []
        if not conditions:
            return issues

        for goods_pat, cond_pat, message, suggestion, code, severity in cls._RULES:
            if not goods_pat.search(goods):
                continue

            found_regex = bool(cond_pat.search(conditions))
            nli_confirms = False
            confidence = 0.85

            # NLI: confirme-t-il la cohérence ?
            if not found_regex and nli is not None and getattr(nli, "available", False):
                try:
                    premise = f"Goods: {goods[:300]}. Conditions: {conditions[:300]}"
                    hyp = "The conditions address specific requirements for these goods."
                    score = nli.predict(premise, hyp)
                    if score is not None:
                        nli_confirms = score.entailment > 0.60
                        confidence = round(1.0 - score.entailment, 3)
                except Exception:
                    pass

            if not found_regex and not nli_confirms:
                issues.append(CrossFieldIssue(
                    fields=[":45A:", ":47A:"],
                    code=code,
                    technique="NLI+regex" if nli is not None and getattr(nli, "available", False) else "regex",
                    severity=severity,
                    message=message,
                    suggestion=suggestion,
                    lc_value=goods[:150],
                    doc_value=conditions[:150],
                    confidence=confidence,
                    ucp_ref="Art.14 UCP 600",
                ))

        return issues


# ══════════════════════════════════════════════════════════════════════
# AXE C — AMOUNT ↔ GOODS
# ══════════════════════════════════════════════════════════════════════

class _AxeC_AmountGoods:
    """Détecte les prix unitaires aberrants pour le type de marchandise."""

    # (goods_pattern, currency, min_unit_price, max_unit_price, unit_label)
    _PRICE_RANGES: list[tuple] = [
        (re.compile(r'\b(crude.oil|petroleum|fuel|diesel|kerosene)\b', re.I),
         None, 300, 1_500, "USD/MT"),
        (re.compile(r'\b(wheat|ble|grain|cereale)\b', re.I),
         None, 150, 600, "USD/MT"),
        (re.compile(r'\b(cotton|coton)\b', re.I),
         None, 1_000, 3_000, "USD/MT"),
        (re.compile(r'\b(steel|acier|iron)\b', re.I),
         None, 400, 3_000, "USD/MT"),
        (re.compile(r'\b(copper|cuivre)\b', re.I),
         None, 6_000, 15_000, "USD/MT"),
        (re.compile(r'\b(gold|or)\b', re.I),
         None, 50_000, 70_000, "USD/KG"),
        (re.compile(r'\b(diamond|diamant)\b', re.I),
         None, 5_000, 1_000_000, "USD/CARAT"),
    ]

    # Pattern pour extraire la quantité depuis :45A:
    _QTY_PATTERN = re.compile(
        r'(\d[\d,.]*)[\s]*(mt|ton|kg|lb|barrel|bbl|carat|kg|'
        r'mton|metric.ton|tonne)', re.I
    )

    @classmethod
    def check(cls, goods: str, amount_str: str, currency: str) -> list[CrossFieldIssue]:
        issues = []
        if not amount_str or not goods:
            return issues

        try:
            amount = float(re.sub(r'[^\d.]', '', str(amount_str)))
        except (ValueError, TypeError):
            return issues

        qty_match = cls._QTY_PATTERN.search(goods)
        if not qty_match:
            return issues

        try:
            qty = float(qty_match.group(1).replace(",", ""))
        except (ValueError, TypeError):
            return issues

        if qty <= 0:
            return issues

        unit_price = amount / qty

        for goods_pat, ccy_filter, min_price, max_price, unit in cls._PRICE_RANGES:
            if not goods_pat.search(goods):
                continue
            if ccy_filter and currency.upper() != ccy_filter:
                continue

            if unit_price < min_price * 0.3:
                issues.append(CrossFieldIssue(
                    fields=[":32B:", ":45A:"],
                    code="AMOUNT_GOODS_UNIT_PRICE_TOO_LOW",
                    technique="rule",
                    severity="warning",
                    message=(
                        f"Prix unitaire estimé ({unit_price:,.0f} {currency}/{unit.split('/')[1]}) "
                        f"très inférieur au marché pour {goods_pat.pattern.split('|')[0].strip('(').strip()} "
                        f"(fourchette habituelle : {min_price:,}–{max_price:,} {unit})."
                    ),
                    suggestion="Vérifier le montant :32B: ou la quantité dans :45A: — risque de sous-facturation.",
                    lc_value=f"{amount:,.0f} {currency}",
                    doc_value=f"Qté estimée : {qty} → prix unitaire : {unit_price:,.0f}",
                    confidence=0.72,
                    ucp_ref="",
                ))
            elif unit_price > max_price * 3:
                issues.append(CrossFieldIssue(
                    fields=[":32B:", ":45A:"],
                    code="AMOUNT_GOODS_UNIT_PRICE_TOO_HIGH",
                    technique="rule",
                    severity="warning",
                    message=(
                        f"Prix unitaire estimé ({unit_price:,.0f} {currency}/{unit.split('/')[1]}) "
                        f"très supérieur au marché pour {goods_pat.pattern.split('|')[0].strip('(').strip()} "
                        f"(fourchette habituelle : {min_price:,}–{max_price:,} {unit})."
                    ),
                    suggestion="Vérifier le montant :32B: ou la quantité dans :45A: — risque de sur-facturation.",
                    lc_value=f"{amount:,.0f} {currency}",
                    doc_value=f"Qté estimée : {qty} → prix unitaire : {unit_price:,.0f}",
                    confidence=0.72,
                    ucp_ref="",
                ))
            break  # une seule règle par LC

        return issues


# ══════════════════════════════════════════════════════════════════════
# AXE D — TAGS ↔ TOUTES LES CLAUSES (étendu)
# ══════════════════════════════════════════════════════════════════════

class _AxeD_TagsConditionsExtended:
    """NLI étendu : détecte les contradictions entre tags SWIFT et clauses :47A:
    au-delà des 5 règles fixes de ContradictionEngine."""

    # Règles supplémentaires (pas couvertes par ContradictionEngine)
    _EXTENDED_RULES: list[dict] = [
        {
            "tag_field": "presentation_period",
            "tag_extract": re.compile(r'(\d+)\s*days?', re.I),
            "clause_patterns": [
                re.compile(r'presentation\s+within\s+(\d+)\s*days?', re.I),
                re.compile(r'documents?\s+(?:must\s+be\s+)?presented?\s+within\s+(\d+)\s*days?', re.I),
                re.compile(r'présentation\s+dans\s+(\d+)\s*jours?', re.I),
            ],
            "code": "PRESENTATION_PERIOD_CONTRADICTION",
            "message_tpl": "Période de présentation :48: ({tag_val}j) contradite par clause :47A: ({clause_val}j)",
            "severity": "error",
            "ucp_ref": "Art.14c UCP 600",
            "nli_hypothesis": "The presentation period stated in the conditions differs from the standard period.",
        },
        {
            "tag_field": "charges",
            "tag_value_pattern": re.compile(r'OUR|BENEFICIARY', re.I),
            "clause_patterns": [
                re.compile(r'all\s+charges?\s+(?:for\s+)?(?:account\s+of\s+)?applicant', re.I),
                re.compile(r'frais?\s+(?:à\s+la\s+charge\s+de\s+)?l.?acheteur', re.I),
            ],
            "code": "CHARGES_TAG_CLAUSE_MISMATCH",
            "message_tpl": "Clause :71B: ({tag_val}) contradite par clause :47A: qui met les frais à charge de l'acheteur",
            "severity": "warning",
            "ucp_ref": "Art.37 UCP 600",
            "nli_hypothesis": "The bank charges are for the account of the applicant.",
        },
        {
            "tag_field": "draft_at",
            "tag_value_pattern": re.compile(r'SIGHT|VUE', re.I),
            "clause_patterns": [
                re.compile(r'deferred\s+payment|usance|différ[eé]', re.I),
                re.compile(r'\d+\s*days?\s+after\s+(?:sight|shipment|invoice)', re.I),
                re.compile(r'\d+\s*jours?\s+(?:après|de\s+vue)', re.I),
            ],
            "code": "DRAFT_AT_SIGHT_VS_DEFERRED_CLAUSE",
            "message_tpl": "Tag :42C: indique paiement à vue mais une clause :47A: suggère un paiement différé",
            "severity": "error",
            "ucp_ref": "Art.6 / Art.42 UCP 600",
            "nli_hypothesis": "The payment is deferred and not at sight.",
        },
    ]

    @classmethod
    def check(
        cls,
        entities: Any,
        conditions_text: str,
        nli: Any = None,
    ) -> list[CrossFieldIssue]:
        issues = []
        if not conditions_text:
            return issues

        for rule in cls._EXTENDED_RULES:
            tag_val = _get(entities, rule["tag_field"])
            if not tag_val:
                continue

            # Check tag value matches trigger pattern
            trigger_pat = rule.get("tag_value_pattern")
            tag_extract  = rule.get("tag_extract")
            if trigger_pat and not trigger_pat.search(tag_val):
                continue

            contradiction_found = False
            clause_val = ""
            confidence = float(rule.get("confidence", 0.85))

            # Step 1: NLI
            if nli is not None and getattr(nli, "available", False):
                try:
                    premise = f"Tag value: {tag_val}. Conditions: {conditions_text[:400]}"
                    score = nli.predict(premise, rule["nli_hypothesis"])
                    if score is not None and score.entailment >= 0.70:
                        contradiction_found = True
                        confidence = float(score.entailment)
                        clause_val = "[NLI detected]"
                except Exception:
                    pass

            # Step 2: Regex (renforcement ou fallback)
            for pattern in rule["clause_patterns"]:
                m = pattern.search(conditions_text)
                if m:
                    ctx = conditions_text[max(0, m.start()-20):m.end()+40]
                    # Extract numeric value if applicable
                    clause_val = ctx
                    if tag_extract:
                        cm = tag_extract.search(str(m.group()))
                        if cm:
                            clause_val = cm.group(1)
                    if contradiction_found:
                        confidence = min(1.0, confidence + 0.10)
                    else:
                        contradiction_found = True
                        confidence = 0.90
                    break

            if contradiction_found:
                message = rule["message_tpl"].format(
                    tag_val=str(tag_val)[:50],
                    clause_val=str(clause_val)[:50],
                )
                issues.append(CrossFieldIssue(
                    fields=[f":{rule['tag_field'].upper()}:", ":47A:"],
                    code=rule["code"],
                    technique="NLI+regex" if nli is not None and getattr(nli, "available", False) else "regex",
                    severity=rule["severity"],
                    message=message,
                    suggestion=f"Aligner la clause :47A: avec le tag ou corriger le tag. Référence : {rule['ucp_ref']}",
                    lc_value=str(tag_val)[:100],
                    doc_value=str(clause_val)[:100],
                    confidence=round(confidence, 3),
                    ucp_ref=rule["ucp_ref"],
                ))

        return issues


# ══════════════════════════════════════════════════════════════════════
# AXE E — INCOTERM ↔ ENSEMBLE DES CHAMPS
# ══════════════════════════════════════════════════════════════════════

class _AxeE_IncotermCrossCheck:
    """Vérifie la cohérence globale de l'Incoterm avec tous les champs MT700."""

    _MARITIME_ONLY = {"FOB", "CFR", "CIF", "FAS"}
    _SELLER_INSURES = {"CIF", "CIP"}
    _BUYER_INSURES  = {"FOB", "CFR", "FAS", "FCA", "EXW", "CPT", "DAP", "DPU", "DDP"}

    _BL_PATTERN  = re.compile(r'\b(bill\s+of\s+lading|b/?l\b|connaissement)\b', re.I)
    _INS_PATTERN = re.compile(r'\b(insurance|assurance|police)\b', re.I)
    _AWB_PATTERN = re.compile(r'\b(airway\s+bill|awb|air\s+waybill)\b', re.I)

    _INLAND_CITIES = {
        "PARIS", "BERLIN", "MADRID", "ROME", "WARSAW", "PRAGUE", "VIENNA", "BRUSSELS",
        "MOSCOW", "BEIJING", "DELHI", "NAIROBI", "RIYADH", "TEHRAN", "BAGHDAD",
    }

    @classmethod
    def check(
        cls,
        incoterm: str,
        docs: str,
        conditions: str,
        port_loading: str,
        port_discharge: str,
        goods: str,
    ) -> list[CrossFieldIssue]:
        issues = []
        inc = (incoterm or "").upper().strip()[:3]
        if not inc:
            return issues

        docs_up  = docs.upper()
        cond_up  = conditions.upper() if conditions else ""
        port_l_up = port_loading.upper() if port_loading else ""
        port_d_up = port_discharge.upper() if port_discharge else ""

        # ── Règle 1 : Incoterm maritime + port intérieur ─────────────
        if inc in cls._MARITIME_ONLY:
            for port_up, port_raw, tag in [(port_l_up, port_loading, ":44A:"),
                                            (port_d_up, port_discharge, ":44B:")]:
                if port_up and any(city in port_up for city in cls._INLAND_CITIES):
                    issues.append(CrossFieldIssue(
                        fields=[":45A:", tag],
                        code="INCOTERM_MARITIME_INLAND_PORT",
                        technique="rule",
                        severity="error",
                        message=f"Incoterm {inc} est maritime mais {tag} '{port_raw}' est une ville intérieure",
                        suggestion=f"Remplacer '{port_raw}' par un port maritime réel ou utiliser FCA/CPT",
                        lc_value=inc,
                        doc_value=port_raw,
                        confidence=0.94,
                        ucp_ref="Art.20 UCP 600",
                    ))

        # ── Règle 2 : CIF/CIP sans police d'assurance dans :46A: ─────
        if inc in cls._SELLER_INSURES and not cls._INS_PATTERN.search(docs):
            issues.append(CrossFieldIssue(
                fields=[":45A:", ":46A:"],
                code="INCOTERM_CIF_NO_INSURANCE",
                technique="rule",
                severity="error",
                message=f"Incoterm {inc} oblige le vendeur à assurer (Art.28) mais aucune police d'assurance dans :46A:",
                suggestion="Ajouter dans :46A: : 'INSURANCE POLICY/CERTIFICATE FOR 110% OF INVOICE VALUE'",
                lc_value=inc,
                doc_value=docs[:100],
                confidence=0.95,
                ucp_ref="Art.28 UCP 600",
            ))

        # ── Règle 3 : FOB/CFR/FAS + BL absent dans :46A: ─────────────
        if inc in cls._MARITIME_ONLY and not cls._BL_PATTERN.search(docs):
            issues.append(CrossFieldIssue(
                fields=[":45A:", ":46A:"],
                code="INCOTERM_MARITIME_NO_BL",
                technique="rule",
                severity="warning",
                message=f"Incoterm {inc} est maritime mais aucun Bill of Lading requis dans :46A:",
                suggestion="Ajouter dans :46A: : 'FULL SET (3/3) CLEAN ON BOARD BILL OF LADING'",
                lc_value=inc,
                doc_value=docs[:100],
                confidence=0.88,
                ucp_ref="Art.20 UCP 600",
            ))

        # ── Règle 4 : EXW/DAP/DDP + BL exigé (contradictoire) ────────
        if inc in {"EXW", "DAP", "DPU", "DDP"} and cls._BL_PATTERN.search(docs):
            issues.append(CrossFieldIssue(
                fields=[":45A:", ":46A:"],
                code="INCOTERM_NON_MARITIME_BL",
                technique="rule",
                severity="error",
                message=f"Incoterm {inc} ne génère pas de BL maritime mais :46A: l'exige",
                suggestion=f"Remplacer le BL par un doc de transport multimodal (Art.19), ou changer l'incoterm",
                lc_value=inc,
                doc_value=docs[:100],
                confidence=0.93,
                ucp_ref="Art.20 / Art.5 UCP 600",
            ))

        # ── Règle 5 : Incoterm dans :45A: ≠ Incoterm dans :47A: ──────
        inco_in_cond = re.search(
            r'\b(EXW|FCA|CPT|CIP|DAP|DPU|DDP|FAS|FOB|CFR|CIF)\b',
            cond_up,
        )
        if inco_in_cond:
            inco_cond = inco_in_cond.group(1)
            if inco_cond != inc:
                issues.append(CrossFieldIssue(
                    fields=[":45A:", ":47A:"],
                    code="INCOTERM_MISMATCH_45A_47A",
                    technique="regex",
                    severity="error",
                    message=f"Incoterm :45A: ({inc}) différent de l'Incoterm mentionné dans :47A: ({inco_cond})",
                    suggestion="Harmoniser l'Incoterm entre :45A: et :47A:. Selon UCP Art.5, la clause prime.",
                    lc_value=inc,
                    doc_value=inco_cond,
                    confidence=0.98,
                    ucp_ref="Art.5 UCP 600",
                ))

        return issues


# ══════════════════════════════════════════════════════════════════════
# AXE H — QUANTITY ↔ DOCUMENTS COHERENCE
# ══════════════════════════════════════════════════════════════════════

class _AxeH_QuantityDocs:
    """
    Checks quantity mentions in :45A: vs document multiplicity requirements in :46A:.

    Rules:
    1. Bulk cargo in :45A: + packing list in :46A: → inconsistency warning
       (bulk goods don't have individual packing units)
    2. Multi-commodity shipment in :45A: + only 1 invoice → info issue
       (multiple commodity lines may need multiple invoices)
    3. Only-copies clause in :46A: for all docs while goods suggest high-value → warning
    """

    _BULK_PATTERN   = re.compile(r'\b(bulk|vrac|loose\s+cargo)\b', re.I)
    _MULTI_COMMODITY = re.compile(
        r'\b(?:and|et|plus|&)\b.{1,30}\b(?:and|et|plus|&)\b', re.I
    )
    _PACKING_PATTERN = re.compile(r'\b(packing\s+list|liste\s+de\s+colisage)\b', re.I)
    _ONLY_COPIES    = re.compile(
        r'\b(?:1|one)\s+(?:copy|copie|original)(?:\s+only)?\b', re.I
    )
    _INVOICE_PATTERN = re.compile(
        r'\b(?:commercial\s+)?invoice|facture\s+commerciale\b', re.I
    )
    _MULTI_INVOICE_PATTERN = re.compile(
        r'\b(?:2|3|4|5|two|three|four|five)\s+(?:originals?\s+)?invoices?\b', re.I
    )
    _HIGH_VALUE_GOODS = re.compile(
        r'\b(?:gold|or\b|diamond|diamant|platinum|platine|jewel|bijou'
        r'|pharmaceutical|pharmaceutique|electronic|electronique)\b', re.I
    )

    @classmethod
    def check(cls, goods: str, docs: str) -> list:
        """
        Returns list of CrossFieldIssue.
        goods = :45A: content, docs = :46A: content.
        """
        issues: list = []
        if not goods or not docs:
            return issues

        docs_up  = docs.upper()
        goods_up = goods.upper()

        # ── Rule 1: BULK cargo + packing list required ─────────────────
        if cls._BULK_PATTERN.search(goods):
            if cls._PACKING_PATTERN.search(docs):
                issues.append(CrossFieldIssue(
                    fields=[":45A:", ":46A:"],
                    code="BULK_CARGO_PACKING_LIST_INCONSISTENCY",
                    technique="rule",
                    severity="warning",
                    message=(
                        "Marchandises en vrac (':45A:' contient 'BULK/VRAC') mais ':46A:' exige "
                        "une liste de colisage (packing list). Les cargaisons en vrac ne sont "
                        "généralement pas conditionnées en unités individuelles."
                    ),
                    suggestion=(
                        "Remplacer le 'PACKING LIST' par un 'WEIGHT/QUANTITY CERTIFICATE' "
                        "ou 'DRAFT SURVEY REPORT' adapté aux cargaisons en vrac."
                    ),
                    lc_value=goods[:150],
                    doc_value=docs[:150],
                    confidence=0.82,
                    ucp_ref="Art.14 UCP 600 / ISBP 745",
                ))

        # ── Rule 2: Multi-commodity shipment + single invoice ──────────
        # Detect multiple commodity lines (e.g., "ITEM A AND ITEM B AND ITEM C")
        commodity_lines = re.findall(
            r'\n|(?:^|\.\s+)(?:[A-Z][A-Z\s,]{5,})',
            goods,
            re.MULTILINE,
        )
        has_multi = cls._MULTI_COMMODITY.search(goods) or len(commodity_lines) >= 3

        if has_multi:
            # Check if only one invoice type is required (not multiple)
            invoice_found = cls._INVOICE_PATTERN.search(docs)
            multi_invoice_found = cls._MULTI_INVOICE_PATTERN.search(docs)
            if invoice_found and not multi_invoice_found:
                issues.append(CrossFieldIssue(
                    fields=[":45A:", ":46A:"],
                    code="MULTI_COMMODITY_SINGLE_INVOICE_INFO",
                    technique="rule",
                    severity="info",
                    message=(
                        "Expédition multi-marchandises possible détectée dans ':45A:' (plusieurs "
                        "lignes de commodités). ':46A:' ne spécifie qu'une seule facture. "
                        "Selon ISBP 745, des factures séparées peuvent être nécessaires pour "
                        "des marchandises de catégories différentes."
                    ),
                    suggestion=(
                        "Vérifier si des factures séparées sont requises pour chaque catégorie de "
                        "marchandises, ou préciser dans :46A: qu'une facture consolidée est acceptable."
                    ),
                    lc_value=goods[:150],
                    doc_value=docs[:150],
                    confidence=0.65,
                    ucp_ref="ISBP 745 §C1",
                ))

        # ── Rule 3: High-value goods + only copies (no originals) ──────
        if cls._HIGH_VALUE_GOODS.search(goods):
            all_entries = re.findall(r'[A-Z][A-Z\s/,]{3,}', docs_up)
            only_copies_count = sum(
                1 for entry in all_entries
                if cls._ONLY_COPIES.search(entry)
            )
            total_doc_entries = len(all_entries)
            if total_doc_entries > 0 and only_copies_count >= total_doc_entries:
                issues.append(CrossFieldIssue(
                    fields=[":45A:", ":46A:"],
                    code="HIGH_VALUE_GOODS_ONLY_COPIES",
                    technique="rule",
                    severity="warning",
                    message=(
                        "Marchandises à haute valeur détectées dans ':45A:' mais ':46A:' "
                        "ne semble exiger que des copies (non des originaux). "
                        "Pour les marchandises à haute valeur, des documents originaux sont "
                        "fortement recommandés (UCP Art.17)."
                    ),
                    suggestion=(
                        "Exiger au minimum 1 exemplaire original pour les documents clés "
                        "(facture, BL, certificat d'origine) lorsque les marchandises sont à haute valeur."
                    ),
                    lc_value=goods[:150],
                    doc_value=docs[:150],
                    confidence=0.70,
                    ucp_ref="Art.17 UCP 600",
                ))

        return issues


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class CrossFieldCoherenceAnalyzer:
    """
    Analyse la cohérence sémantique INTER-CHAMPS d'un MT700.

    Utilise NLI comme détecteur primaire + regex comme renforcement/fallback.
    Compatible avec OfflineNLIPredictor (offline-first, jamais de download).

    Usage :
        analyzer = CrossFieldCoherenceAnalyzer(nli=nli_predictor)
        result   = analyzer.analyze(entities, raw_text)
        payload  = result.to_dict()
    """

    def __init__(self, nli: Any = None) -> None:
        self._nli = nli

    def analyze(self, entities: Any, raw_text: str = "") -> CrossFieldCoherenceResult:
        import time
        t0 = time.monotonic()

        result = CrossFieldCoherenceResult()

        goods      = _get(entities, "goods_description")
        docs       = _get(entities, "documents_required")
        conditions = _get(entities, "additional_conditions")
        amount_str = _get(entities, "amount")
        currency   = _get(entities, "currency")
        incoterm   = _get(entities, "incoterm")
        port_l     = _get(entities, "port_loading")
        port_d     = _get(entities, "port_discharge")

        # Fallback: extraire depuis raw_text si absents
        if not docs and raw_text:
            m = re.search(r':46A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
            if m:
                docs = m.group(1).strip()
        if not conditions and raw_text:
            m = re.search(r':47A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
            if m:
                conditions = m.group(1).strip()
        if not goods and raw_text:
            m = re.search(r':45A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
            if m:
                goods = m.group(1).strip()
        if not incoterm and goods:
            m = re.search(r'\b(EXW|FCA|CPT|CIP|DAP|DPU|DDP|FAS|FOB|CFR|CIF)\b', goods, re.I)
            if m:
                incoterm = m.group(1).upper()

        # ── AXE A ────────────────────────────────────────────────────
        if goods and docs:
            axe_a = _AxeA_GoodsDocs.check(goods, docs, self._nli)
            result.issues.extend(axe_a)
            result.axes_checked.append("A:goods_docs")

        # ── AXE B ────────────────────────────────────────────────────
        if goods and conditions:
            axe_b = _AxeB_GoodsConditions.check(goods, conditions, self._nli)
            result.issues.extend(axe_b)
            result.axes_checked.append("B:goods_conditions")

        # ── AXE C ────────────────────────────────────────────────────
        if goods and amount_str:
            axe_c = _AxeC_AmountGoods.check(goods, amount_str, currency)
            result.issues.extend(axe_c)
            result.axes_checked.append("C:amount_goods")

        # ── AXE D ────────────────────────────────────────────────────
        if conditions:
            axe_d = _AxeD_TagsConditionsExtended.check(entities, conditions, self._nli)
            result.issues.extend(axe_d)
            result.axes_checked.append("D:tags_conditions_extended")

        # ── AXE E ────────────────────────────────────────────────────
        if incoterm:
            axe_e = _AxeE_IncotermCrossCheck.check(
                incoterm, docs, conditions, port_l, port_d, goods
            )
            result.issues.extend(axe_e)
            result.axes_checked.append("E:incoterm_crosscheck")

        # ── AXE F : Alignement d'entités inter-champs ───────────────
        if EntityAlignmentChecker is not None:
            try:
                ea = EntityAlignmentChecker(nli=self._nli)
                axe_f = ea.check(entities, raw_text)
                result.issues.extend(axe_f)
                result.axes_checked.append("F:entity_alignment")
            except Exception as e:
                logger.warning("AxeF entity alignment error: %s", e)

        # ── AXE H : Cohérence quantité ↔ documents ──────────────────
        if goods and docs:
            axe_h = _AxeH_QuantityDocs.check(goods, docs)
            result.issues.extend(axe_h)
            result.axes_checked.append("H:quantity_docs")

        # ── Score global ─────────────────────────────────────────────
        result.coherence_score = self._compute_score(result.issues)
        result.verdict = (
            "COHERENT"           if result.coherence_score >= 80 else
            "PARTIALLY_COHERENT" if result.coherence_score >= 50 else
            "INCOHERENT"
        )
        result.processing_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "CrossFieldCoherence: %d issues — score=%.1f verdict=%s (%dms)",
            len(result.issues), result.coherence_score, result.verdict, result.processing_ms,
        )
        return result

    @staticmethod
    def _compute_score(issues: list[CrossFieldIssue]) -> float:
        score = 100.0
        for i in issues:
            w = i.confidence
            if i.severity == "error":
                score -= 18.0 * w
            elif i.severity == "warning":
                score -= 8.0 * w
            else:
                score -= 2.0 * w
        return max(0.0, round(score, 1))
