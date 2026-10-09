"""
swift_entity_alignment.py — v1.0
==================================
Entity alignment across MT700 fields using fuzzy + dynamic NLI.

7 alignment axes:
  F1 — :59: beneficiary ↔ :46A: invoice issuer ("invoice issued by X")
  F2 — :50: applicant ↔ :47A: applicant mentions
  F3 — :52A: issuing bank ↔ :47A: bank references
  F4 — :32B:+:39A: amount/tolerance ↔ :47A: invoice value clauses
  F5 — :44A:/:44B: ports ↔ :45A: goods origin mention
  F6 — :31D: expiry ↔ :48: presentation period (logical impossibility)
  F7 — :42C: draft_at ↔ :42A: drawee (payment mode consistency)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, date
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── CrossFieldIssue import ─────────────────────────────────────────────
try:
    from .swift_cross_field_coherence import CrossFieldIssue
except Exception:
    from swift_cross_field_coherence import CrossFieldIssue  # type: ignore

# ── rapidfuzz graceful fallback ────────────────────────────────────────
try:
    from rapidfuzz import fuzz as _rfuzz
    _FUZZY_OK = True
except ImportError:
    _FUZZY_OK = False
    logger.debug("rapidfuzz not available — falling back to exact string matching")


# ── Helper: field accessor ─────────────────────────────────────────────

def _get(entities: Any, key: str, default: str = "") -> str:
    if entities is None:
        return default
    if isinstance(entities, dict):
        return str(entities.get(key) or default)
    return str(getattr(entities, key, None) or default)


# ── Fuzzy ratio helper ─────────────────────────────────────────────────

def _fuzzy_ratio(a: str, b: str) -> float:
    """Return token_sort_ratio(a, b) using rapidfuzz, or 100 if exact, else 0."""
    if not a or not b:
        return 0.0
    a_up = a.upper().strip()
    b_up = b.upper().strip()
    if a_up == b_up:
        return 100.0
    if a_up in b_up or b_up in a_up:
        return 90.0
    if not _FUZZY_OK:
        return 0.0
    try:
        return float(_rfuzz.token_sort_ratio(a_up, b_up))
    except Exception:
        return 0.0


def _fuzzy_partial(a: str, b: str) -> float:
    """Return partial_ratio(a, b) using rapidfuzz, fallback exact check."""
    if not a or not b:
        return 0.0
    a_up = a.upper().strip()
    b_up = b.upper().strip()
    if a_up in b_up or b_up in a_up:
        return 100.0
    if not _FUZZY_OK:
        return 0.0
    try:
        return float(_rfuzz.partial_ratio(a_up, b_up))
    except Exception:
        return 0.0


# ── Date parsing helper ────────────────────────────────────────────────

def _parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%y%m%d", "%Y%m%d", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


# ── Known landlocked countries (for F6 port/goods coherence) ──────────

_LANDLOCKED_COUNTRIES: set[str] = {
    "AFGHANISTAN", "ANDORRA", "ARMENIA", "AUSTRIA", "AZERBAIJAN",
    "BELARUS", "BHUTAN", "BOLIVIA", "BOTSWANA", "BURKINA FASO",
    "BURUNDI", "CENTRAL AFRICAN REPUBLIC", "CHAD", "CZECH REPUBLIC",
    "CZECHIA", "ETHIOPIA", "HUNGARY", "KAZAKHSTAN", "KOSOVO",
    "KYRGYZSTAN", "LAOS", "LESOTHO", "LIECHTENSTEIN", "LUXEMBOURG",
    "MALAWI", "MALI", "MOLDOVA", "MONGOLIA", "NEPAL", "NIGER",
    "NORTH MACEDONIA", "PARAGUAY", "RWANDA", "SAN MARINO",
    "SERBIA", "SLOVAKIA", "SOUTH SUDAN", "SWAZILAND", "ESWATINI",
    "SWITZERLAND", "TAJIKISTAN", "TURKMENISTAN", "UGANDA",
    "UZBEKISTAN", "VATICAN", "ZAMBIA", "ZIMBABWE",
}


# ══════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════

class EntityAlignmentChecker:
    """
    Detects entity inconsistencies across MT700 fields.

    Uses fuzzy matching (with graceful fallback if rapidfuzz unavailable)
    and dynamic NLI hypotheses generated from actual field content.

    7 alignment axes (F1–F7).
    """

    def __init__(self, nli: Any = None, fuzzy_threshold: int = 72) -> None:
        self._nli = nli
        self._fuzzy_threshold = fuzzy_threshold

    # ── Public API ────────────────────────────────────────────────────

    def check(self, entities: Any, raw_text: str = "") -> list:
        """Returns list of CrossFieldIssue from all F axes."""
        issues: list = []
        issues.extend(self._check_f1_beneficiary_invoice(entities, raw_text))
        issues.extend(self._check_f2_applicant_conditions(entities, raw_text))
        issues.extend(self._check_f3_amount_clauses(entities, raw_text))
        issues.extend(self._check_f4_temporal_window(entities))
        issues.extend(self._check_f5_draft_payment(entities, raw_text))
        issues.extend(self._check_f6_port_goods(entities, raw_text))
        issues.extend(self._check_f7_charges(entities, raw_text))
        return issues

    # ── NLI helper ────────────────────────────────────────────────────

    def _nli_predict(self, premise: str, hypothesis: str) -> Optional[float]:
        """Return entailment score [0-1] or None if NLI unavailable."""
        if self._nli is None or not getattr(self._nli, "available", False):
            return None
        try:
            score = self._nli.predict(premise, hypothesis)
            if score is None:
                return None
            return float(getattr(score, "entailment", 0.0))
        except Exception as exc:
            logger.debug("NLI predict error: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────
    # F1 — Beneficiary (:59:) ↔ Invoice issuer (:46A:)
    # ──────────────────────────────────────────────────────────────────

    # Patterns that introduce the invoice issuer name in :46A:
    _F1_ISSUER_PATTERNS: list = [
        re.compile(
            r'(?:INVOICE\s+(?:ISSUED|DRAWN|MADE)\s+BY|INVOICE\s+FROM|ISSUED\s+BY'
            r'|FACTURE\s+[ÉE]MISE\s+PAR|FACTURE\s+DE)\s*[:\-]?\s*(.{5,80}?)(?:\n|$|\.|,)',
            re.I,
        ),
    ]

    def _check_f1_beneficiary_invoice(
        self, entities: Any, raw_text: str
    ) -> list:
        issues: list = []
        beneficiary = _get(entities, "beneficiary")
        docs_required = _get(entities, "documents_required")

        # Also check raw_text for :46A: section
        if not docs_required and raw_text:
            m46 = re.search(
                r':46A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I
            )
            if m46:
                docs_required = m46.group(1).strip()

        if not beneficiary or not docs_required:
            return issues

        # Extract beneficiary name (first meaningful line)
        benef_name = beneficiary.strip().split("\n")[0].strip()[:80]

        # Look for issuer phrase in docs_required
        issuer_name: Optional[str] = None
        for pat in self._F1_ISSUER_PATTERNS:
            m = pat.search(docs_required)
            if m:
                issuer_name = m.group(1).strip()[:80]
                break

        if not issuer_name:
            return issues

        # Fuzzy match beneficiary vs issuer name
        ratio = _fuzzy_ratio(benef_name, issuer_name)
        partial = _fuzzy_partial(benef_name, issuer_name)
        best_fuzzy = max(ratio, partial)

        # NLI dynamic hypothesis from actual field content
        nli_used = False
        nli_confidence: Optional[float] = None
        hyp = f"The invoice must be issued by {benef_name[:60]}."
        premise = f"Documents required: {docs_required[:400]}"
        ent_score = self._nli_predict(premise, hyp)
        if ent_score is not None:
            nli_used = True
            nli_confidence = ent_score

        # Decide technique label
        if nli_used and _FUZZY_OK:
            technique = "NLI+fuzzy"
        elif nli_used:
            technique = "NLI"
        elif _FUZZY_OK:
            technique = "fuzzy"
        else:
            technique = "rule"

        # Mismatch: fuzzy score below threshold AND (NLI says not entailed OR NLI unavailable)
        fuzzy_mismatch = best_fuzzy < self._fuzzy_threshold
        nli_mismatch = (nli_confidence is not None and nli_confidence < 0.40)
        nli_match = (nli_confidence is not None and nli_confidence >= 0.60)

        if fuzzy_mismatch and not nli_match:
            confidence = round(
                1.0 - (best_fuzzy / 100.0) if best_fuzzy > 0 else 0.85,
                3,
            )
            if nli_mismatch:
                confidence = min(1.0, confidence + 0.10)
            issues.append(CrossFieldIssue(
                fields=[":59:", ":46A:"],
                code="BENEFICIARY_INVOICE_ISSUER_MISMATCH",
                technique=technique,
                severity="error",
                message=(
                    f"Bénéficiaire ':59:' ({benef_name[:40]}) ne correspond pas à l'émetteur "
                    f"de facture dans ':46A:' ({issuer_name[:40]}). "
                    f"Similarité: {best_fuzzy:.0f}%."
                ),
                suggestion=(
                    "Aligner le nom de l'émetteur de facture dans :46A: avec le bénéficiaire "
                    "mentionné dans :59:. Selon UCP Art.14, les documents doivent émaner du bénéficiaire."
                ),
                lc_value=benef_name,
                doc_value=issuer_name,
                confidence=confidence,
                ucp_ref="Art.14 UCP 600",
            ))

        return issues

    # ──────────────────────────────────────────────────────────────────
    # F2 — Applicant (:50:) ↔ :47A: applicant mentions
    # ──────────────────────────────────────────────────────────────────

    _F2_APPLICANT_PATTERNS: list = [
        re.compile(
            r"(?:APPLICANT|BUYER|DONNEUR\s+D['’]ORDRE|ACHETEUR|PURCHASER)"
            r"[\s:,\-]+([A-Z][A-Z\s&.,\-']{3,60}?)(?:\n|$|\.|,|\b(?:MUST|SHALL|WILL)\b)",
            re.I,
        ),
    ]

    def _check_f2_applicant_conditions(
        self, entities: Any, raw_text: str
    ) -> list:
        issues: list = []
        applicant = _get(entities, "applicant")
        conditions = _get(entities, "additional_conditions")

        if not conditions and raw_text:
            m47 = re.search(
                r':47A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I
            )
            if m47:
                conditions = m47.group(1).strip()

        if not applicant or not conditions:
            return issues

        appl_name = applicant.strip().split("\n")[0].strip()[:80]

        # Find applicant-like names in conditions
        found_names: list[str] = []
        for pat in self._F2_APPLICANT_PATTERNS:
            for m in pat.finditer(conditions):
                candidate = m.group(1).strip()[:80]
                if len(candidate) > 3:
                    found_names.append(candidate)

        if not found_names:
            return issues

        # Check if any found name fuzzy-matches the applicant
        best_ratio = 0.0
        best_name = found_names[0]
        for name in found_names:
            r = max(_fuzzy_ratio(appl_name, name), _fuzzy_partial(appl_name, name))
            if r > best_ratio:
                best_ratio = r
                best_name = name

        # NLI
        nli_used = False
        nli_confidence: Optional[float] = None
        hyp = f"The applicant is {appl_name[:60]}."
        premise = f"Conditions: {conditions[:400]}"
        ent_score = self._nli_predict(premise, hyp)
        if ent_score is not None:
            nli_used = True
            nli_confidence = ent_score

        technique = (
            "NLI+fuzzy" if nli_used and _FUZZY_OK
            else "NLI" if nli_used
            else "fuzzy" if _FUZZY_OK
            else "rule"
        )

        fuzzy_mismatch = best_ratio < self._fuzzy_threshold
        nli_match = (nli_confidence is not None and nli_confidence >= 0.55)

        if fuzzy_mismatch and not nli_match:
            confidence = round(max(0.5, 1.0 - best_ratio / 100.0), 3)
            issues.append(CrossFieldIssue(
                fields=[":50:", ":47A:"],
                code="APPLICANT_CONDITION_MISMATCH",
                technique=technique,
                severity="warning",
                message=(
                    f"Nom de l'applicant ':50:' ({appl_name[:40]}) ne correspond pas "
                    f"au nom mentionné dans ':47A:' ({best_name[:40]}). "
                    f"Similarité: {best_ratio:.0f}%."
                ),
                suggestion=(
                    "Vérifier que le nom de l'applicant dans les conditions :47A: correspond "
                    "exactement au champ :50:. Toute divergence peut engendrer des discordances documentaires."
                ),
                lc_value=appl_name,
                doc_value=best_name,
                confidence=confidence,
                ucp_ref="Art.14 UCP 600",
            ))

        return issues

    # ──────────────────────────────────────────────────────────────────
    # F3 — Amount (:32B:) + Tolerance (:39A:) ↔ Invoice value clauses (:47A:)
    # ──────────────────────────────────────────────────────────────────

    _F3_CLAUSE_PATTERNS: list = [
        re.compile(
            r'(?:INVOICE\s+VALUE\s+NOT\s+TO\s+EXCEED'
            r'|FACTURE\s+NE\s+DEVANT\s+PAS\s+D[ÉE]PASSER'
            r'|MAXIMUM\s+INVOICE\s+VALUE'
            r'|INVOICE\s+(?:AMOUNT\s+)?(?:NOT\s+TO\s+EXCEED|MAX(?:IMUM)?)'
            r'|VALEUR\s+(?:DE\s+FACTURE\s+)?(?:MAXIMALE?|PLAFOND))'
            r'\s*[:\-]?\s*([A-Z]{0,3}\s*[\d,.\s]+)',
            re.I,
        ),
    ]

    def _check_f3_amount_clauses(
        self, entities: Any, raw_text: str
    ) -> list:
        issues: list = []
        amount_str = _get(entities, "amount")
        tolerance_str = _get(entities, "tolerance_plus")  # from :39A: "X/Y" format
        conditions = _get(entities, "additional_conditions")

        if not conditions and raw_text:
            m47 = re.search(
                r':47A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I
            )
            if m47:
                conditions = m47.group(1).strip()

        if not amount_str or not conditions:
            return issues

        try:
            amount = float(re.sub(r'[^\d.]', '', str(amount_str)))
        except (ValueError, TypeError):
            return issues

        if amount <= 0:
            return issues

        # Parse tolerance from :39A: (format "X/Y" pct +/-)
        tol_plus_pct = 0.0
        tol_minus_pct = 0.0
        if tolerance_str:
            tol_m = re.match(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', str(tolerance_str))
            if tol_m:
                tol_plus_pct = float(tol_m.group(1))
                tol_minus_pct = float(tol_m.group(2))
            else:
                tol_m2 = re.match(r'(\d+(?:\.\d+)?)', str(tolerance_str))
                if tol_m2:
                    tol_plus_pct = float(tol_m2.group(1))

        max_allowed = amount * (1 + tol_plus_pct / 100.0)

        # Look for invoice value cap clause
        for pat in self._F3_CLAUSE_PATTERNS:
            m = pat.search(conditions)
            if not m:
                continue
            clause_raw = m.group(1).strip()
            # Extract numeric value from clause
            num_m = re.search(r'[\d,.\s]+', clause_raw)
            if not num_m:
                continue
            try:
                clause_amount = float(re.sub(r'[^\d.]', '', num_m.group().replace(' ', '')))
            except (ValueError, TypeError):
                continue

            if clause_amount <= 0:
                continue

            # Check: if clause cap < amount (minus tolerance) → mismatch
            if clause_amount < amount * (1 - tol_minus_pct / 100.0) * 0.95:
                # NLI
                nli_used = False
                ent_score = self._nli_predict(
                    f"Amount: {amount_str}. Conditions: {conditions[:300]}",
                    f"The invoice value must not exceed {clause_amount:,.2f}.",
                )
                if ent_score is not None:
                    nli_used = True

                technique = (
                    "NLI+fuzzy" if nli_used and _FUZZY_OK
                    else "NLI" if nli_used
                    else "rule"
                )

                issues.append(CrossFieldIssue(
                    fields=[":32B:", ":47A:"],
                    code="AMOUNT_INVOICE_CLAUSE_MISMATCH",
                    technique=technique,
                    severity="error",
                    message=(
                        f"Montant LC ':32B:' ({amount:,.2f}) supérieur au plafond de facture "
                        f"stipulé dans ':47A:' ({clause_amount:,.2f}). "
                        f"Tolérance appliquée: +{tol_plus_pct}% / -{tol_minus_pct}%."
                    ),
                    suggestion=(
                        "Aligner le montant :32B: avec le plafond de facture indiqué dans :47A:, "
                        "ou ajuster la clause de plafond. Référence : UCP Art.18b."
                    ),
                    lc_value=amount_str,
                    doc_value=str(clause_amount),
                    confidence=0.88,
                    ucp_ref="Art.18b UCP 600",
                ))
            break  # one check per LC

        return issues

    # ──────────────────────────────────────────────────────────────────
    # F4 — Expiry (:31D:) ↔ Presentation period (:48:) temporal check
    # ──────────────────────────────────────────────────────────────────

    def _check_f4_temporal_window(self, entities: Any) -> list:
        issues: list = []
        expiry_str = _get(entities, "expiry_date")
        period_str = _get(entities, "presentation_period")
        shipment_str = _get(entities, "shipment_date")

        if not expiry_str or not period_str or not shipment_str:
            return issues

        expiry_date = _parse_date(expiry_str)
        shipment_date = _parse_date(shipment_str)

        if not expiry_date or not shipment_date:
            return issues

        # Extract presentation period in days
        pp_m = re.search(r'(\d+)', str(period_str))
        if not pp_m:
            return issues
        presentation_days = int(pp_m.group(1))

        from datetime import timedelta
        deadline = shipment_date + timedelta(days=presentation_days)

        if deadline > expiry_date:
            overrun = (deadline - expiry_date).days
            issues.append(CrossFieldIssue(
                fields=[":31D:", ":48:"],
                code="EXPIRY_PRESENTATION_IMPOSSIBLE_WINDOW",
                technique="rule",
                severity="error",
                message=(
                    f"Fenêtre de présentation impossible : date expédition ({shipment_str}) "
                    f"+ période de présentation ({presentation_days}j) = {deadline.isoformat()} "
                    f"dépasse la date d'expiration ({expiry_str}) de {overrun} jour(s)."
                ),
                suggestion=(
                    f"Prolonger la date d'expiration ':31D:' d'au moins {overrun} jour(s), "
                    f"ou réduire la période de présentation ':48:' (UCP Art.14c : max 21j)."
                ),
                lc_value=expiry_str,
                doc_value=f"{presentation_days}j après {shipment_str}",
                confidence=0.97,
                ucp_ref="Art.14c UCP 600",
            ))

        return issues

    # ──────────────────────────────────────────────────────────────────
    # F5 — Draft mode (:42C:) ↔ Payment mode coherence (:47A:)
    # ──────────────────────────────────────────────────────────────────

    _F5_DEFERRED_PATTERNS: list = [
        re.compile(r'\bdeferred\s+payment\b', re.I),
        re.compile(r'\busance\b', re.I),
        re.compile(r'\b(\d+)\s+days?\s+after\s+(?:sight|shipment|date\s+of\s+bill|date\s+of\s+invoice)', re.I),
        re.compile(r'\bpaiement\s+diff[eé]r[eé]\b', re.I),
        re.compile(r'\b(\d+)\s+jours?\s+(?:apr[eè]s|de\s+vue)\b', re.I),
    ]

    def _check_f5_draft_payment(
        self, entities: Any, raw_text: str
    ) -> list:
        issues: list = []
        draft_at = _get(entities, "draft_at")
        conditions = _get(entities, "additional_conditions")

        if not conditions and raw_text:
            m47 = re.search(
                r':47A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I
            )
            if m47:
                conditions = m47.group(1).strip()

        if not draft_at or not conditions:
            return issues

        draft_upper = draft_at.upper().strip()
        is_at_sight = any(kw in draft_upper for kw in ["AT SIGHT", "SIGHT", "VUE", "À VUE"])

        if not is_at_sight:
            return issues

        # Check if conditions mention deferred/usance
        deferred_found = any(pat.search(conditions) for pat in self._F5_DEFERRED_PATTERNS)
        if not deferred_found:
            return issues

        # Extract context
        ctx = ""
        for pat in self._F5_DEFERRED_PATTERNS:
            m = pat.search(conditions)
            if m:
                start = max(0, m.start() - 20)
                end = min(len(conditions), m.end() + 40)
                ctx = conditions[start:end]
                break

        # NLI
        nli_used = False
        nli_confidence = 0.85
        hyp_sight = "Payment is at sight."
        hyp_deferred = "Payment is deferred."
        ent_sight = self._nli_predict(f"Conditions: {conditions[:400]}", hyp_sight)
        ent_deferred = self._nli_predict(f"Conditions: {conditions[:400]}", hyp_deferred)
        if ent_sight is not None and ent_deferred is not None:
            nli_used = True
            nli_confidence = min(1.0, ent_deferred + 0.10)

        technique = (
            "NLI+fuzzy" if nli_used and _FUZZY_OK
            else "NLI" if nli_used
            else "rule"
        )

        issues.append(CrossFieldIssue(
            fields=[":42C:", ":47A:"],
            code="DRAFT_PAYMENT_MODE_INCOHERENT",
            technique=technique,
            severity="error",
            message=(
                f"Tag ':42C:' ({draft_at.strip()}) indique un paiement à vue, "
                f"mais ':47A:' contient une clause de paiement différé/usance : '{ctx[:60]}'."
            ),
            suggestion=(
                "Corriger la contradiction entre :42C: (paiement à vue) et la clause de paiement "
                "différé dans :47A:. Selon UCP Art.6/Art.42, les modalités de paiement doivent être cohérentes."
            ),
            lc_value=draft_at.strip(),
            doc_value=ctx[:80],
            confidence=round(nli_confidence, 3),
            ucp_ref="Art.6 / Art.42 UCP 600",
        ))

        return issues

    # ──────────────────────────────────────────────────────────────────
    # F6 — Port (:44A:/:44E:) ↔ Goods origin (:45A:)
    # ──────────────────────────────────────────────────────────────────

    _F6_ORIGIN_PATTERNS: list = [
        re.compile(
            r'(?:MADE\s+IN|ORIGIN[:\s]+|SHIPPED\s+FROM|COUNTRY\s+OF\s+ORIGIN[:\s]+|'
            r'ORIGINE[:\s]+|FABRIQUÉ\s+EN|PRODUIT\s+(?:EN|DE))\s+([A-Z][A-Z\s]{2,30}?)(?:\n|$|\.|\,|\b)',
            re.I,
        ),
    ]

    def _check_f6_port_goods(
        self, entities: Any, raw_text: str
    ) -> list:
        issues: list = []
        port_loading = _get(entities, "port_loading")
        place_receipt = _get(entities, "place_of_receipt")  # :44A:
        goods = _get(entities, "goods_description")

        if not goods and raw_text:
            m45 = re.search(
                r':45A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I
            )
            if m45:
                goods = m45.group(1).strip()

        port = port_loading or place_receipt
        if not port or not goods:
            return issues

        # Look for origin mentions in goods
        origin_country: Optional[str] = None
        for pat in self._F6_ORIGIN_PATTERNS:
            m = pat.search(goods)
            if m:
                origin_country = m.group(1).strip().upper()
                break

        if not origin_country:
            return issues

        # Check if origin country is landlocked (and port implies maritime)
        is_landlocked = any(
            ll in origin_country for ll in _LANDLOCKED_COUNTRIES
        )

        if not is_landlocked:
            return issues

        # Check if port suggests maritime (contains common maritime indicators)
        port_upper = port.upper()
        maritime_indicators = [
            "PORT", "HARBOUR", "HARBOR", "QUAY", "WHARF",
            "PIER", "TERMINAL", "DOCK",
        ]
        # Known landlocked but has international airports — skip if air
        air_indicators = ["AIRPORT", "AIR", "AWB", "AIRWAY"]
        if any(air in port_upper for air in air_indicators):
            return issues

        has_maritime_indicator = any(ind in port_upper for ind in maritime_indicators)

        if has_maritime_indicator or port_upper:
            technique = "fuzzy" if _FUZZY_OK else "rule"
            issues.append(CrossFieldIssue(
                fields=[":44A:", ":45A:"],
                code="PORT_GOODS_ORIGIN_INCOMPATIBLE",
                technique=technique,
                severity="warning",
                message=(
                    f"Les marchandises (':45A:') mentionnent une origine dans '{origin_country}', "
                    f"pays enclavé (sans accès maritime), mais le port mentionné est '{port.strip()[:50]}'. "
                    f"Incohérence géographique possible."
                ),
                suggestion=(
                    f"Vérifier si les marchandises transitent par un pays tiers avec accès maritime, "
                    f"et préciser le port d'embarquement réel. "
                    f"Utiliser un incoterm multimodal si applicable (CPT/CIP/FCA)."
                ),
                lc_value=port.strip()[:50],
                doc_value=origin_country[:50],
                confidence=0.78,
                ucp_ref="Art.14 UCP 600",
            ))

        return issues

    # ──────────────────────────────────────────────────────────────────
    # F7 — Charges (:71B:) ↔ Conditions (:47A:)
    # ──────────────────────────────────────────────────────────────────

    _F7_APPLICANT_CHARGES_PATTERNS: list = [
        re.compile(r"ALL\s+(?:BANKING\s+)?CHARGES?\s+FOR\s+(?:ACCOUNT\s+OF\s+)?(?:APPLICANT|BUYER|ACHETEUR)", re.I),
        re.compile(r"FRAIS?\s+(?:BANCAIRES?\s+)?(?:[A-Z]\s+LA\s+CHARGE\s+DE\s+L[''])?(?:ACHETEUR|APPLICANT|DONNEUR\s+D['']ORDRE)", re.I),
        re.compile(r"CHARGES?\s+(?:TO\s+BE\s+BORNE\s+BY|AT\s+CHARGE\s+OF)\s+(?:APPLICANT|BUYER)", re.I),
    ]

    _F7_BENEFICIARY_CHARGES_PATTERNS: list = [
        re.compile(r"ALL\s+(?:BANKING\s+)?CHARGES?\s+FOR\s+(?:ACCOUNT\s+OF\s+)?(?:BENEFICIARY|SELLER|BEN\b)", re.I),
        re.compile(r"CHARGES?\s+(?:TO\s+BE\s+BORNE\s+BY|AT\s+CHARGE\s+OF)\s+(?:BENEFICIARY|SELLER)", re.I),
        re.compile(r"FRAIS?\s+(?:BANCAIRES?\s+)?(?:[A-Z]\s+LA\s+CHARGE\s+(?:DU\s+)?)?(?:B[EE]N[EE]FICIAIRE|VENDEUR)", re.I),
    ]

    def _check_f7_charges(
        self, entities: Any, raw_text: str
    ) -> list:
        issues: list = []
        charges_tag = _get(entities, "charges")  # :71B:
        conditions = _get(entities, "additional_conditions")

        if not conditions and raw_text:
            m47 = re.search(
                r':47A:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I
            )
            if m47:
                conditions = m47.group(1).strip()

        # Also try :71B: from raw_text
        if not charges_tag and raw_text:
            m71 = re.search(r':71B:\s*(.*?)(?=\n:\d{2}[A-Z]?:|\Z)', raw_text, re.DOTALL | re.I)
            if m71:
                charges_tag = m71.group(1).strip()

        if not charges_tag or not conditions:
            return issues

        charges_upper = charges_tag.upper().strip()

        # :71B: OUR/BEN → charges on beneficiary/seller side
        is_our = "OUR" in charges_upper
        is_ben = "BEN" in charges_upper
        is_sha = "SHA" in charges_upper

        # Check for contradicting clause in :47A:
        applicant_charge_found = any(
            pat.search(conditions) for pat in self._F7_APPLICANT_CHARGES_PATTERNS
        )
        beneficiary_charge_found = any(
            pat.search(conditions) for pat in self._F7_BENEFICIARY_CHARGES_PATTERNS
        )

        # Get clause context
        def _get_clause_ctx(patterns: list, text: str) -> str:
            for pat in patterns:
                m = pat.search(text)
                if m:
                    s = max(0, m.start() - 10)
                    e = min(len(text), m.end() + 40)
                    return text[s:e]
            return ""

        mismatch = False
        clause_ctx = ""
        code = "CHARGES_TAG_CLAUSE_MISMATCH"
        message = ""

        if (is_our or is_ben) and applicant_charge_found:
            # :71B: says charges on beneficiary side, but clause says applicant pays
            mismatch = True
            clause_ctx = _get_clause_ctx(self._F7_APPLICANT_CHARGES_PATTERNS, conditions)
            message = (
                f"Tag ':71B:' ({charges_tag.strip()}) indique que les frais sont à charge du "
                f"bénéficiaire/vendeur, mais ':47A:' contient une clause mettant les frais à charge "
                f"de l'applicant/acheteur : '{clause_ctx[:60]}'."
            )

        elif is_sha and beneficiary_charge_found:
            # :71B: SHA means shared, but clause puts all on beneficiary
            mismatch = True
            clause_ctx = _get_clause_ctx(self._F7_BENEFICIARY_CHARGES_PATTERNS, conditions)
            message = (
                f"Tag ':71B:' ({charges_tag.strip()}) indique des frais partagés (SHA), "
                f"mais ':47A:' contient une clause mettant tous les frais à charge du bénéficiaire : "
                f"'{clause_ctx[:60]}'."
            )

        if not mismatch:
            return issues

        # NLI
        nli_used = False
        nli_confidence = 0.82
        ent_score = self._nli_predict(
            f"Charges tag: {charges_tag}. Conditions: {conditions[:400]}",
            "The bank charges are for the account of the applicant.",
        )
        if ent_score is not None:
            nli_used = True
            nli_confidence = min(1.0, ent_score + 0.10)

        technique = (
            "NLI+fuzzy" if nli_used and _FUZZY_OK
            else "NLI" if nli_used
            else "rule"
        )

        issues.append(CrossFieldIssue(
            fields=[":71B:", ":47A:"],
            code=code,
            technique=technique,
            severity="warning",
            message=message,
            suggestion=(
                "Aligner le champ :71B: avec les clauses de frais dans :47A:. "
                "Selon UCP Art.37, la répartition des frais doit être cohérente dans tout le document."
            ),
            lc_value=charges_tag.strip(),
            doc_value=clause_ctx[:80],
            confidence=round(nli_confidence, 3),
            ucp_ref="Art.37 UCP 600",
        ))

        return issues
