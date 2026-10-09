"""
swift_validator.py  —  v5.1
Validation complète des messages SWIFT extraits.

v5.1 :
  - MT700/MT720 : :41a: accepte banque désignée (raw_fields["41_bank"]) même sans mode
  - MT700/MT720 : :71B: warning si valeur non reconnue comme frais (P5)
  - MT799 : score calculé différemment (texte libre — pas de champs obligatoires manquants)
  - Références : tolérance plus large (espaces, caractères spéciaux dans les refs OCR)
  - Date : warning plutôt qu'erreur pour les dates passées sur MT700 (LC historiques)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Optional

from .entities import SwiftEntities


# ---------------------------------------------------------------------------
# Structures de résultat
# ---------------------------------------------------------------------------

@dataclass
class FieldValidation:
    tag:      str
    name:     str
    value:    Any
    valid:    bool
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    swift_type:             str
    overall_valid:          bool                       = False
    fields:                 dict[str, FieldValidation] = field(default_factory=dict)
    cross_field_errors:     list[str]                  = field(default_factory=list)
    cross_field_warnings:   list[str]                  = field(default_factory=list)
    missing_mandatory:      list[str]                  = field(default_factory=list)
    summary:                str                        = ""
    score:                  int                        = 0

    def to_dict(self) -> dict:
        return {
            "swift_type":             self.swift_type,
            "overall_valid":          self.overall_valid,
            "score":                  self.score,
            "summary":                self.summary,
            "missing_mandatory":      self.missing_mandatory,
            "cross_field_errors":     self.cross_field_errors,
            "cross_field_warnings":   self.cross_field_warnings,
            "fields": {
                tag: {
                    "name":     fv.name,
                    "value":    fv.value,
                    "valid":    fv.valid,
                    "errors":   fv.errors,
                    "warnings": fv.warnings,
                }
                for tag, fv in self.fields.items()
            },
        }


# ---------------------------------------------------------------------------
# Bibliothèque de validateurs atomiques
# ---------------------------------------------------------------------------

_VALID_CURRENCIES = frozenset([
    "USD", "EUR", "GBP", "CHF", "JPY", "CNY", "CAD", "AUD",
    "HKD", "SGD", "NOK", "SEK", "DKK", "TND", "MAD", "DZD",
    "SAR", "AED", "QAR", "KWD", "EGP", "NGN", "ZAR", "INR", "BRL",
])


class V:
    """Validateurs atomiques statiques."""

    @staticmethod
    def date(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return False, "Date absente"
        try:
            datetime.strptime(str(value).strip()[:10], "%Y-%m-%d")
            return True, None
        except ValueError:
            return False, f"Format de date invalide : {value!r} (attendu YYYY-MM-DD)"

    @staticmethod
    def future_date(value: str | None) -> tuple[bool, str | None]:
        """Date non expirée. Génère un WARNING (pas une erreur bloquante)."""
        if not value:
            return True, None
        try:
            d = datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
            if d < date.today():
                return False, f"Date expirée : {value} (document historique ou test)"
            return True, None
        except ValueError:
            return True, None

    @staticmethod
    def currency(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return False, "Devise absente"
        if value.upper().strip() not in _VALID_CURRENCIES:
            return False, f"Devise non reconnue : {value!r}"
        return True, None

    @staticmethod
    def positive_amount(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return False, "Montant absent"
        try:
            if float(str(value).replace(",", "")) <= 0:
                return False, f"Montant doit être > 0 (reçu : {value})"
            return True, None
        except (ValueError, AttributeError):
            return False, f"Montant invalide : {value!r}"

    @staticmethod
    def reference(value: str | None, max_len: int = 35) -> tuple[bool, str | None]:
        """
        Référence SWIFT — v5.1 : max_len étendu à 35 et caractères tolérés élargis.
        Les références OCR contiennent parfois des espaces ou tirets.
        """
        if not value:
            return False, "Référence vide"
        v = value.strip()
        if len(v) > max_len:
            return False, f"Référence trop longue : {len(v)} > {max_len} car."
        # Accepter lettres, chiffres, /, -, ., +, espace (OCR)
        if not re.match(r'^[A-Za-z0-9/\-\.\+\s]+$', v):
            return False, f"Caractères non autorisés dans la référence : {v!r}"
        return True, None

    @staticmethod
    def sequence(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return False, "Séquence absente"
        m = re.match(r"(\d+)/(\d+)", str(value).strip())
        if not m:
            return False, f"Format séquence invalide : {value!r} (attendu n/N)"
        seq, tot = int(m.group(1)), int(m.group(2))
        if seq < 1 or tot < 1:
            return False, "Séquence/total doivent être >= 1"
        if seq > tot:
            return False, f"Séquence {seq} > total {tot}"
        return True, None

    @staticmethod
    def enum(value: str | None, allowed: list[str]) -> tuple[bool, str | None]:
        if not value:
            return False, f"Valeur absente. Attendu : {allowed}"
        v = str(value).upper().strip()
        for a in allowed:
            if v == a.upper() or a.upper() in v:
                return True, None
        return False, f"Valeur {value!r} non autorisée. Attendu : {allowed}"

    @staticmethod
    def non_empty(value: Any) -> tuple[bool, str | None]:
        if not value:
            return False, "Champ vide ou absent"
        if isinstance(value, str) and not value.strip():
            return False, "Champ vide"
        return True, None

    @staticmethod
    def tolerance(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return True, None
        m = re.match(r"(\d+)/(\d+)", str(value).strip())
        if not m:
            return False, f"Format tolérance invalide : {value!r} (attendu n/n)"
        for part in [m.group(1), m.group(2)]:
            if not (0 <= int(part) <= 100):
                return False, f"Tolérance hors limites : {part} (0-100)"
        return True, None

    @staticmethod
    def presentation_period(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return True, None
        m = re.match(r"(\d+)", str(value).strip())
        if not m:
            return False, f"Délai invalide : {value!r}"
        days = int(m.group(1))
        if not (1 <= days <= 365):
            return False, f"Délai {days} jours hors limites (1-365)"
        return True, None

    @staticmethod
    def bic(value: str | None) -> tuple[bool, str | None]:
        if not value:
            return False, "BIC absent"
        v = value.strip().split()[0].upper()
        if re.match(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$', v):
            return True, None
        return False, f"BIC invalide : {v!r} (attendu 8 ou 11 car.)"

    @staticmethod
    def date_order(early: str | None, late: str | None,
                   label_e: str, label_l: str) -> tuple[bool, str | None]:
        if not early or not late:
            return True, None
        try:
            d1 = datetime.strptime(early[:10], "%Y-%m-%d").date()
            d2 = datetime.strptime(late[:10],  "%Y-%m-%d").date()
            if d1 > d2:
                return False, f"{label_e} ({early}) est postérieure à {label_l} ({late})"
            return True, None
        except ValueError:
            return True, None

    @staticmethod
    def charges_value(value: str | None) -> tuple[bool, str | None]:
        """
        Vérifie que :71B: contient vraiment des frais bancaires (P5).
        Warning si la valeur ressemble à une durée de présentation (:48:).
        """
        if not value:
            return True, None
        v = value.upper().strip()
        if re.match(r'^\d+\s*DAYS?', v):
            return False, f":71B: ressemble à une durée de présentation, pas des frais : {value!r}"
        if v.startswith('SEE ') or v.startswith('WITHIN '):
            return False, f":71B: valeur non standard pour des frais : {value!r}"
        return True, None


# ---------------------------------------------------------------------------
# Validateur principal
# ---------------------------------------------------------------------------

class SwiftValidator:

    def validate(self, entities: SwiftEntities) -> ValidationReport:
        r = ValidationReport(swift_type=entities.swift_type)

        dispatch = {
            "MT700": self._validate_mt700,
            "MT720": self._validate_mt720,
            "MT760": self._validate_mt760,
            "MT790": self._validate_mt790,
            "MT799": self._validate_mt799,
        }

        fn = dispatch.get(entities.swift_type)
        if fn is None:
            r.cross_field_errors.append(
                f"Type {entities.swift_type!r} non supporté par le validateur"
            )
        else:
            fn(entities, r)

        r.missing_mandatory = [
            tag for tag, fv in r.fields.items()
            if not fv.valid and any("obligatoire" in e for e in fv.errors)
        ]

        total_checks  = len(r.fields) or 1
        passed_checks = sum(1 for fv in r.fields.values() if fv.valid)
        r.score = round(passed_checks / total_checks * 100)

        has_errors = (
            bool(r.cross_field_errors)
            or any(not fv.valid for fv in r.fields.values())
        )
        r.overall_valid = not has_errors
        r.summary = self._summary(r)
        return r

    # ==================================================================
    # Validateurs par type
    # ==================================================================

    def _validate_mt700(self, e: SwiftEntities, r: ValidationReport) -> None:

        self._field(r, "27",  "Sequence of Total",          e.sequence,
                    [(V.sequence, True)])

        self._field(r, "40A", "Form of Documentary Credit", e.lc_form,
                    [(lambda v: V.enum(v, ["IRREVOCABLE", "IRREVOCABLE TRANSFERABLE"]), True)])

        self._field(r, "20",  "Documentary Credit Number",  e.lc_number,
                    [(V.reference, True)])

        self._field(r, "31C", "Date of Issue",              e.issue_date,
                    [(V.date, True)])

        self._field(r, "40E", "Applicable Rules",           e.applicable_rules,
                    [(lambda v: V.enum(v, [
                        "UCP LATEST VERSION", "UCPURR LATEST VERSION",
                        "EUCP LATEST VERSION", "EUCPURR LATEST VERSION", "OTHR"
                    ]), True)])

        self._field(r, "31D", "Date and Place of Expiry",   e.expiry_date,
                    [(V.date, True),
                     (V.future_date, False)])   # warning seulement (pas erreur bloquante)

        self._field(r, "50",  "Applicant",                  e.applicant,
                    [(V.non_empty, True)])

        self._field(r, "59",  "Beneficiary",                e.beneficiary,
                    [(V.non_empty, True)])

        self._field(r, "32B", "Currency Code, Amount",      e.currency,
                    [(V.currency, True),
                     (lambda _: V.positive_amount(e.amount), True)])

        # :41a: — v5.1 : accepte banque désignée même sans mode de disponibilité
        avail_value = e.available_with or e.raw_fields.get("41_bank")
        self._field(r, "41a", "Available With / By",        avail_value,
                    [(lambda v: V.non_empty(v), False)], mandatory=False)

        self._field(r, "42C", "Drafts At",                  e.draft_at,
                    [(V.non_empty, False)], mandatory=False)

        self._field(r, "43P", "Partial Shipments",          e.partial_shipments,
                    [(lambda v: V.enum(v, ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]),
                      False)], mandatory=False)

        self._field(r, "43T", "Transhipment",               e.transhipment,
                    [(lambda v: V.enum(v, ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]),
                      False)], mandatory=False)

        self._field(r, "39A", "Percentage Credit Tolerance",
                    f"{e.tolerance_plus}/{e.tolerance_minus}" if e.tolerance_plus else None,
                    [(V.tolerance, False)], mandatory=False)

        self._field(r, "44C", "Latest Date of Shipment",    e.shipment_date,
                    [(V.date, False)], mandatory=False)

        self._field(r, "45A", "Description of Goods",       e.goods_description,
                    [(V.non_empty, False)], mandatory=False)

        self._field(r, "46A", "Documents Required",         e.documents_required,
                    [(V.non_empty, False)], mandatory=False)

        self._field(r, "47A", "Additional Conditions",      e.additional_conditions,
                    [(V.non_empty, False)], mandatory=False)

        self._field(r, "48",  "Period for Presentation",    e.presentation_period,
                    [(V.presentation_period, False)], mandatory=False)

        self._field(r, "49",  "Confirmation Instructions",  e.confirmation,
                    [(lambda v: V.enum(v, ["CONFIRM", "MAY ADD", "WITHOUT"]),
                      False)], mandatory=False)

        # :71B: v5.1 — warning si valeur ressemble à :48:
        raw_71b = e.charges or e.raw_fields.get("71B_raw")
        self._field(r, "71B", "Charges",                    raw_71b,
                    [(V.non_empty, False),
                     (V.charges_value, False)],             # warning si durée détectée
                    mandatory=False)

        # ── Cross-champs ─────────────────────────────────────────────
        raw = e.raw_fields
        has_42c = bool(e.draft_at)
        has_41a = bool(e.available_with or raw.get("41_bank"))
        if has_42c and not has_41a:
            r.cross_field_warnings.append("C1: :42C: présent mais :41a: absent")

        if "44C" in raw and "44D" in raw:
            r.cross_field_errors.append("C3: :44C: et :44D: ne peuvent pas coexister")

        ok, err = V.date_order(e.issue_date, e.expiry_date, "Date d'émission", "Date d'expiration")
        if not ok:
            r.cross_field_errors.append(err)

        ok, err = V.date_order(e.shipment_date, e.expiry_date, "Date d'expédition", "Date d'expiration")
        if not ok:
            r.cross_field_warnings.append(err)

        if e.presentation_period:
            m = re.match(r"(\d+)", e.presentation_period)
            if m and int(m.group(1)) > 21:
                r.cross_field_warnings.append(
                    f"Délai de présentation de {m.group(1)} jours dépasse les 21 jours recommandés par l'UCP"
                )

    def _validate_mt720(self, e: SwiftEntities, r: ValidationReport) -> None:

        self._field(r, "27",  "Sequence of Total",              e.sequence,
                    [(V.sequence, True)])

        self._field(r, "20",  "Transferring Bank's Reference",  e.lc_number,
                    [(V.reference, True)])

        self._field(r, "21",  "Documentary Credit Number",      e.related_lc,
                    [(V.reference, True)])

        self._field(r, "31C", "Date of Issue",                  e.issue_date,
                    [(V.date, True)])

        self._field(r, "31D", "Date and Place of Expiry",       e.expiry_date,
                    [(V.date, True), (V.future_date, False)])

        self._field(r, "50",  "First Beneficiary",              e.applicant,
                    [(V.non_empty, True)])

        self._field(r, "59",  "Second Beneficiary",             e.beneficiary,
                    [(V.non_empty, True)])

        self._field(r, "32B", "Currency Code, Amount",          e.currency,
                    [(V.currency, True),
                     (lambda _: V.positive_amount(e.amount), True)])

        self._field(r, "40E", "Applicable Rules",               e.applicable_rules,
                    [(lambda v: V.enum(v, [
                        "UCP LATEST VERSION", "UCPURR LATEST VERSION",
                        "EUCP LATEST VERSION", "EUCPURR LATEST VERSION", "OTHR"
                    ]), True)])

        self._field(r, "43P", "Partial Shipments",              e.partial_shipments,
                    [(lambda v: V.enum(v, ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]),
                      False)], mandatory=False)

        self._field(r, "43T", "Transhipment",                   e.transhipment,
                    [(lambda v: V.enum(v, ["ALLOWED", "CONDITIONAL", "NOT ALLOWED"]),
                      False)], mandatory=False)

        self._field(r, "48",  "Period for Presentation",        e.presentation_period,
                    [(V.presentation_period, False)], mandatory=False)

        # :71B: v5.1
        raw_71b = e.charges or e.raw_fields.get("71B_raw")
        self._field(r, "71B", "Charges",                        raw_71b,
                    [(V.non_empty, False), (V.charges_value, False)], mandatory=False)

        # Cross-champs
        raw = e.raw_fields
        has_52  = any(t.startswith("52") for t in raw)
        has_50b = "50B" in raw
        if has_52 and has_50b:
            r.cross_field_errors.append(
                "C4: :52a: et :50B: ne peuvent pas coexister"
            )

        if e.applicant and e.beneficiary and e.applicant.strip() == e.beneficiary.strip():
            r.cross_field_warnings.append(
                "Premier bénéficiaire (:50:) et second bénéficiaire (:59:) sont identiques"
            )

        ok, err = V.date_order(e.issue_date, e.expiry_date, "Date d'émission", "Date d'expiration")
        if not ok:
            r.cross_field_errors.append(err)

    def _validate_mt760(self, e: SwiftEntities, r: ValidationReport) -> None:

        self._field(r, "27",  "Sequence of Total",               e.sequence,
                    [(V.sequence, True)])

        self._field(r, "20",  "Undertaking Number",              e.lc_number,
                    [(V.reference, True)])

        self._field(r, "32B", "Undertaking Amount",              e.currency,
                    [(V.currency, True),
                     (lambda _: V.positive_amount(e.amount), True)])

        self._field(r, "50",  "Applicant / Instructing Party",   e.applicant,
                    [(V.non_empty, True)])

        self._field(r, "59",  "Beneficiary",                     e.beneficiary,
                    [(V.non_empty, True)])

        self._field(r, "77U", "Undertaking Terms and Conditions",e.guarantee_text,
                    [(V.non_empty, True)])

        self._field(r, "22A", "Purpose of Message",              e.purpose,
                    [(lambda v: V.enum(v, ["ISSU", "ISCO", "ICCO", "ADVI", "ACNF"]),
                      True)])

        self._field(r, "22D", "Form of Undertaking",             e.undertaking_type,
                    [(lambda v: V.enum(v, ["DGAR", "STBY"]), True)])

        self._field(r, "23B", "Expiry Type",                     e.expiry_type,
                    [(lambda v: V.enum(v, ["FIXD", "COND", "OPEN"]), True)])

        self._field(r, "40C", "Applicable Rules",                e.applicable_rules,
                    [(lambda v: V.enum(v, ["URDG", "UCP", "ISPR", "NONE", "OTHR"]),
                      True)])

        self._field(r, "30",  "Date of Issue",                   e.issue_date,
                    [(V.date, False)], mandatory=False)

        self._field(r, "31E", "Date of Expiry",                  e.expiry_date,
                    [(V.date, False), (V.future_date, False)], mandatory=False)

    def _validate_mt790(self, e: SwiftEntities, r: ValidationReport) -> None:

        self._field(r, "20",  "Transaction Reference Number",    e.lc_number,
                    [(V.reference, True)])

        self._field(r, "25",  "Account Identification",          e.account,
                    [(V.non_empty, True)])

        self._field(r, "32A", "Value Date / Currency / Amount",  e.value_date,
                    [(V.date, True)])
        self._field(r, "32A_ccy", "Currency (32A)",              e.currency,
                    [(V.currency, True)])
        self._field(r, "32A_amt", "Amount (32A)",                e.amount,
                    [(V.positive_amount, True)])

        self._field(r, "71B", "Details of Charges",              e.charges,
                    [(V.non_empty, True)])

        self._field(r, "21",  "Related Reference",               e.related_lc,
                    [(V.reference, False)], mandatory=False)

    def _validate_mt799(self, e: SwiftEntities, r: ValidationReport) -> None:

        self._field(r, "20",  "Transaction Reference Number",    e.lc_number,
                    [(V.reference, True)])

        self._field(r, "79",  "Narrative",                       e.free_text,
                    [(V.non_empty, True)])

        self._field(r, "21",  "Related Reference",               e.related_lc,
                    [(V.reference, False)], mandatory=False)

        if e.free_text:
            lines = e.free_text.splitlines()
            warns = []
            if len(lines) > 35:
                warns.append(f":79: dépasse 35 lignes ({len(lines)} lignes)")
            long_lines = [i + 1 for i, l in enumerate(lines) if len(l) > 50]
            if long_lines:
                warns.append(f":79: lignes trop longues (> 50 car.) : {long_lines}")
            if warns:
                fv = r.fields.get("79")
                if fv:
                    fv.warnings.extend(warns)

    # ==================================================================
    # Helper interne
    # ==================================================================

    def _field(self,
               report: ValidationReport,
               tag: str,
               name: str,
               value: Any,
               checks: list[tuple[Callable, bool]],
               mandatory: bool = True) -> None:
        errs:  list[str] = []
        warns: list[str] = []

        if value is None or (isinstance(value, str) and not value.strip()):
            if mandatory:
                errs.append("Champ obligatoire absent")
            else:
                warns.append("Champ optionnel absent")
        else:
            for check_fn, is_error in checks:
                ok, msg = check_fn(value)
                if not ok and msg:
                    (errs if is_error else warns).append(msg)

        report.fields[tag] = FieldValidation(
            tag=tag, name=name,
            value=value,
            valid=not errs,
            errors=errs, warnings=warns,
        )

    @staticmethod
    def _summary(r: ValidationReport) -> str:
        n_err  = sum(len(fv.errors) for fv in r.fields.values()) + len(r.cross_field_errors)
        n_warn = sum(len(fv.warnings) for fv in r.fields.values()) + len(r.cross_field_warnings)
        status = "✅ VALIDE" if r.overall_valid else "❌ INVALIDE"
        return (
            f"{status} | {r.swift_type} | Score {r.score}/100 | "
            f"{n_err} erreur(s) | {n_warn} avertissement(s)"
        )