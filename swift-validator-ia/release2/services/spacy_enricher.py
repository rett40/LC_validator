"""
spacy_enricher.py
=================
Couche 2 du pipeline — enrichissement par spaCy NER.

Remplit les champs manquants et valide sémantiquement les valeurs extraites
via la reconnaissance d'entités nommées (DATE, GPE, LOC).

Optionnel : si spaCy ou le modèle 'en_core_web_sm' n'est pas installé,
toutes les méthodes retournent des résultats vides sans lever d'exception.

Usage :
  from release2.services.spacy_enricher import SpaCyEnricher
  enricher = SpaCyEnricher()
  new_fields = enricher.enrich(fields, full_text, doc_type, missing_fields)
"""

from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)

_DATE_FIELDS: frozenset[str] = frozenset({
    "etd", "eta", "issue_date", "invoice_date",
})

_PLACE_FIELDS: frozenset[str] = frozenset({
    "port_of_loading", "port_of_discharge",
    "place_of_receipt", "place_of_delivery",
})

_PLACE_CONTEXT: dict[str, str] = {
    "port_of_loading":   "PORT OF LOADING",
    "port_of_discharge": "PORT OF DISCHARGE",
    "place_of_receipt":  "PLACE OF RECEIPT",
    "place_of_delivery": "PLACE OF DELIVERY",
}

_DATE_CONTEXT: dict[str, list[str]] = {
    "issue_date":   ["DATE OF ISSUE", "ISSUE DATE", "DATE ISSUED"],
    "invoice_date": ["INVOICE DATE", "DATE OF INVOICE"],
    "etd":          ["ETD", "DATE OF DEPARTURE", "DEPARTURE DATE"],
    "eta":          ["ETA", "DATE OF ARRIVAL", "ARRIVAL DATE"],
}


class SpaCyEnricher:
    """
    Enrichisseur NER basé sur spaCy (lazy-load).

    Rôles :
      • Remplir les champs de date manquants (entités DATE)
      • Remplir les champs de port manquants (entités GPE/LOC)
      • Valider sémantiquement une valeur extraite
    """

    def __init__(self, model: str = "en_core_web_sm") -> None:
        self._model_name = model
        self._nlp        = None
        self._available: bool | None = None

    # ── API publique ──────────────────────────────────────────────────────────

    def enrich(
        self,
        fields:        dict[str, str],
        full_text:     str,
        doc_type:      str,
        fields_needed: set[str] | None = None,
    ) -> dict[str, str]:
        """
        Retourne uniquement les NOUVEAUX champs trouvés (ne surcharge jamais fields).

        Paramètres
        ----------
        fields        : champs déjà extraits (ne sont jamais écrasés)
        full_text     : texte brut du document
        doc_type      : "bl", "invoice", "packing_list"
        fields_needed : restreint l'enrichissement à ces champs si fourni
        """
        nlp = self._get_nlp()
        if nlp is None or not fields_needed:
            return {}

        missing = set(fields_needed) - set(fields)
        if not missing:
            return {}

        try:
            doc = nlp(full_text[:6000])
        except Exception as e:
            log.warning("spaCy NER échoué : %s", e)
            return {}

        date_ents  = [e.text.strip() for e in doc.ents if e.label_ == "DATE"]
        place_ents = [e.text.strip() for e in doc.ents
                      if e.label_ in ("GPE", "LOC") and len(e.text.strip()) > 2]

        found: dict[str, str] = {}

        if missing & _DATE_FIELDS and date_ents:
            found.update(self._fill_dates(missing & _DATE_FIELDS, date_ents, full_text))

        if missing & _PLACE_FIELDS and place_ents:
            found.update(self._fill_places(missing & _PLACE_FIELDS, place_ents, full_text))

        # Sécurité : ne jamais écraser un champ existant
        found = {k: v for k, v in found.items() if k not in fields and v}

        if found:
            log.debug("spaCy a enrichi %d champ(s) : %s", len(found), list(found.keys()))
        return found

    def validate_field(self, field_name: str, value: str) -> tuple[bool, str]:
        """Valide sémantiquement une valeur. Retourne (is_valid, message)."""
        nlp = self._get_nlp()
        if nlp is None:
            return True, "spaCy non disponible"
        if field_name in _DATE_FIELDS:
            return self._validate_date(value, nlp)
        if field_name in _PLACE_FIELDS:
            return self._validate_place(value, nlp)
        return True, "ok"

    # ── Chargement modèle ─────────────────────────────────────────────────────

    def _get_nlp(self):
        if self._available is False:
            return None
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy  # noqa: PLC0415
            self._nlp      = spacy.load(self._model_name)
            self._available = True
            log.info("spaCy modèle '%s' chargé", self._model_name)
        except (ImportError, OSError) as e:
            log.info("spaCy non disponible (%s) — enrichissement désactivé", e)
            self._available = False
        return self._nlp

    # ── Remplissage ───────────────────────────────────────────────────────────

    def _fill_dates(
        self,
        missing:   set[str],
        date_ents: list[str],
        text:      str,
    ) -> dict[str, str]:
        found:     dict[str, str] = {}
        available: list[str]      = list(date_ents)

        for field in ["issue_date", "invoice_date", "etd", "eta"]:
            if field not in missing or not available:
                continue
            # Cherche la date la plus proche d'un label contextuel
            best, best_pos = None, 10_000
            for label in _DATE_CONTEXT.get(field, []):
                for date_str in available:
                    m = re.search(
                        rf"{re.escape(label)}[^\n]{{0,100}}{re.escape(date_str)}",
                        text, re.IGNORECASE,
                    )
                    if m and m.start() < best_pos:
                        best_pos = m.start()
                        best     = date_str
            if best:
                found[field] = best
                available.remove(best)
            elif available:
                found[field] = available.pop(0)

        return found

    def _fill_places(
        self,
        missing:    set[str],
        place_ents: list[str],
        text:       str,
    ) -> dict[str, str]:
        found:     dict[str, str] = {}
        available: list[str]      = list(place_ents)

        for field, label in _PLACE_CONTEXT.items():
            if field not in missing or not available:
                continue
            for place in available:
                m = re.search(
                    rf"{re.escape(label)}[^\n]{{0,80}}{re.escape(place)}",
                    text, re.IGNORECASE,
                )
                if m:
                    found[field] = place
                    available.remove(place)
                    break

        return found

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_date(self, value: str, nlp) -> tuple[bool, str]:
        has_date = any(e.label_ == "DATE" for e in nlp(value).ents)
        has_date = has_date or bool(re.search(r"\d{1,4}[-/\.]\d{1,2}", value))
        return (True, "DATE détectée") if has_date else (False, f"'{value}' ne ressemble pas à une date")

    def _validate_place(self, value: str, nlp) -> tuple[bool, str]:
        has_place = any(e.label_ in ("GPE", "LOC") for e in nlp(value).ents)
        if not has_place:
            has_place = bool(re.match(r"^[A-Z\s,\-]{4,}$", value.strip()))
        return (True, "Lieu détecté") if has_place else (False, f"'{value}' ne semble pas être un lieu")
