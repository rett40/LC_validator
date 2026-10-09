"""
lc_acceptance_criteria.py  —  v6.0
Extraction et vérification des critères d'acceptation des Lettres de Crédit.
Applicable aux MT700 et MT720.

v5.1 :
  - Fallback : si :46A: vide, chercher les documents dans :45A: (goods_description)
    (certains formats fusionnent goods + documents dans :45A:)
  - _parse_documents_from_goods : détecte les listes numérotées dans :45A:
  - charges_for_beneficiary : vérification plus robuste
v5.4 :
  - _split_document_items v3 : split inline + séparateur '+'
v6.0 :
  - B14 : fallback presentation_period depuis :47A: si :48: absent
  - B15 : charges_for_beneficiary depuis :47A: pattern OUTSIDE→BENEFICIARY
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .entities import SwiftEntities


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentRequirement:
    """Un document requis avec ses conditions."""
    document_type: str
    copies:        Optional[int]  = None
    conditions:    list[str]      = field(default_factory=list)
    raw:           str            = ""


@dataclass
class LCAcceptanceCriteria:
    """
    Critères d'acceptation complets d'une LC (MT700/MT720).
    Extraits depuis les champs :46A:, :47A:, :43P:, :43T:, :48:, etc.
    """

    documents:                 list[DocumentRequirement] = field(default_factory=list)
    additional_conditions:     list[str]                 = field(default_factory=list)
    partial_shipments_allowed: Optional[bool]            = None
    transhipment_allowed:      Optional[bool]            = None
    latest_shipment_date:      Optional[str]             = None
    port_of_loading:           Optional[str]             = None
    port_of_discharge:         Optional[str]             = None
    presentation_period_days:  Optional[int]             = None
    expiry_date:               Optional[str]             = None
    expiry_place:              Optional[str]             = None
    available_by:              Optional[str]             = None
    draft_at:                  Optional[str]             = None
    charges_for_beneficiary:   Optional[bool]            = None
    charges_raw:               Optional[str]             = None
    confirmation:              Optional[str]             = None
    goods_description:         Optional[str]             = None
    applicable_rules:          Optional[str]             = None
    warnings:                  list[str]                 = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "documents": [
                {
                    "document_type": d.document_type,
                    "copies":        d.copies,
                    "conditions":    d.conditions,
                    "raw":           d.raw,
                }
                for d in self.documents
            ],
            "additional_conditions":      self.additional_conditions,
            "partial_shipments_allowed":  self.partial_shipments_allowed,
            "transhipment_allowed":       self.transhipment_allowed,
            "latest_shipment_date":       self.latest_shipment_date,
            "port_of_loading":            self.port_of_loading,
            "port_of_discharge":          self.port_of_discharge,
            "presentation_period_days":   self.presentation_period_days,
            "expiry_date":                self.expiry_date,
            "expiry_place":               self.expiry_place,
            "available_by":               self.available_by,
            "draft_at":                   self.draft_at,
            "charges_for_beneficiary":    self.charges_for_beneficiary,
            "charges_raw":                self.charges_raw,
            "confirmation":               self.confirmation,
            "goods_description":          self.goods_description,
            "applicable_rules":           self.applicable_rules,
            "warnings":                   self.warnings,
        }


# ---------------------------------------------------------------------------
# Extracteur
# ---------------------------------------------------------------------------

class LCAcceptanceCriteriaExtractor:

    _DOCUMENT_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("COMMERCIAL INVOICE",
         re.compile(r"COMMERCIAL\s+INVOICE|INVOICE", re.I)),
        ("BILL OF LADING",
         re.compile(r"BILL\s+OF\s+LADING|B/?L\b|BILLS?\s+OF\s+LADING", re.I)),
        ("AIRWAY BILL",
         re.compile(r"AIR\s*WAY\s+BILL|AWB\b|AIR\s+WAYBILL", re.I)),
        ("PACKING LIST",
         re.compile(r"PACKING\s+LIST|PACKING\s+SLIP", re.I)),
        ("CERTIFICATE OF ORIGIN",
         re.compile(r"CERTIFICATE\s+OF\s+ORIGIN|C/?O\s+CERTIFICATE|ORIGIN\s+CERT", re.I)),
        ("INSURANCE CERTIFICATE",
         re.compile(r"INSURANCE\s+(?:CERTIFICATE|POLICY|COVER\s+NOTE)|INSURANCE\b", re.I)),
        ("INSPECTION CERTIFICATE",
         re.compile(r"INSPECTION\s+CERTIFICATE|INSPECTION\s+REPORT", re.I)),
        ("WEIGHT CERTIFICATE",
         re.compile(r"WEIGHT\s+(?:CERTIFICATE|LIST|NOTE)", re.I)),
        ("PHYTOSANITARY CERTIFICATE",
         re.compile(r"PHYTO\w*\s+CERT|SANITARY\s+CERT", re.I)),
        ("HEALTH CERTIFICATE",
         re.compile(r"HEALTH\s+CERTIFICATE", re.I)),
        ("DRAFT",
         re.compile(r"\bDRAFT\b|\bBILL\s+OF\s+EXCHANGE\b", re.I)),
        ("BENEFICIARY CERTIFICATE",
         re.compile(r"BENEFICIARY\s+CERT", re.I)),
    ]

    _CONDITION_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("SIGNED",          re.compile(r"\bSIGNED\b", re.I)),
        ("ORIGINAL",        re.compile(r"\bORIGINAL\b", re.I)),
        ("FULL SET",        re.compile(r"\bFULL\s+SET\b", re.I)),
        ("CLEAN",           re.compile(r"\bCLEAN\b", re.I)),
        ("ON BOARD",        re.compile(r"\bON\s+BOARD\b", re.I)),
        ("FREIGHT PREPAID", re.compile(r"\bFREIGHT\s+PREPAID\b", re.I)),
        ("NOTIFY",          re.compile(r"\bNOTIFY\b", re.I)),
        ("IN ENGLISH",      re.compile(r"\bIN\s+ENGLISH\b", re.I)),
        ("INDICATING LC NUMBER", re.compile(r"\bLC\s+NUMBER\b|\bL/?C\s+NO\b", re.I)),
    ]

    # Pattern pour reconnaître une liste de documents dans :45A:
    # (quand :46A: est absent ou vide)
    # v5.2 : pattern assoupli — déclenche sur liste numérotée même si le mot-clé
    # est à la fin de la ligne (ex: '1. SIGNED COMMERCIAL INVOICE')
    _DOC_IN_GOODS_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:\d+[\.\)]\s*|/\d+/\s*|[-\+\*]\s*)'
        r'(?:SIGNED|ORIGINAL|FULL\s+SET|'
        r'COMMERCIAL\s+INVOICE|COMMERCIAL\s+INV|'
        r'BILL\s+OF\s+LADING|BILLS?\s+OF\s+LAD|'
        r'PACKING\s+LIST|CERTIFICATE\s+OF\s+ORIGIN|'
        r'INSURANCE|INSPECTION|AIRWAY\s+BILL|DRAFT\b|'
        r'CLEAN\s+ON\s+BOARD)',
        re.IGNORECASE
    )

    def extract(self, entities: SwiftEntities) -> LCAcceptanceCriteria:
        if entities.swift_type not in ("MT700", "MT720"):
            criteria = LCAcceptanceCriteria()
            criteria.warnings.append(
                f"Critères d'acceptation LC non applicables pour {entities.swift_type}"
            )
            return criteria

        criteria = LCAcceptanceCriteria()

        # ── Documents requis :46A: ────────────────────────────────────
        if entities.documents_required:
            criteria.documents = self._parse_documents(entities.documents_required)

        # ── Fallback P6 : chercher documents dans :45A: si :46A: vide ─
        # Certains formats OCR fusionnent goods + documents dans :45A:
        if not criteria.documents and entities.goods_description:
            docs_from_goods = self._parse_documents_from_goods(entities.goods_description)
            if docs_from_goods:
                criteria.documents = docs_from_goods
                criteria.warnings.append(
                    "Documents extraits de :45A: (goods_description) car :46A: absent — "
                    "vérification manuelle recommandée"
                )

        # ── Conditions supplémentaires :47A: ─────────────────────────
        if entities.additional_conditions:
            criteria.additional_conditions = self._parse_conditions(entities.additional_conditions)

        # ── Expéditions partielles :43P: ─────────────────────────────
        if entities.partial_shipments:
            v = entities.partial_shipments.upper()
            criteria.partial_shipments_allowed = "NOT ALLOWED" not in v and "ALLOWED" in v

        # ── Transbordement :43T: ─────────────────────────────────────
        if entities.transhipment:
            v = entities.transhipment.upper()
            criteria.transhipment_allowed = "NOT ALLOWED" not in v and "ALLOWED" in v

        # ── Ports ────────────────────────────────────────────────────
        criteria.port_of_loading   = entities.port_loading
        criteria.port_of_discharge = entities.port_discharge

        # ── Date limite d'expédition :44C: ───────────────────────────
        criteria.latest_shipment_date = entities.shipment_date

        # ── Délai de présentation :48: ────────────────────────────────
        if entities.presentation_period:
            m = re.match(r"(\d+)", entities.presentation_period)
            if m:
                criteria.presentation_period_days = int(m.group(1))

        # B14 : Fallback — chercher 'N DAYS' dans :47A: si :48: est absent
        # Ex Doc10: ':48: absent' mais ':47A: ... 21 DAYS AFTER SHIPMENT DATE ...'
        if criteria.presentation_period_days is None and entities.additional_conditions:
            m = re.search(
                r'\b(\d+)\s+DAYS?\b.*?(?:AFTER\s+SHIPMENT|PRESENTATION|VALIDITY)',
                entities.additional_conditions,
                re.IGNORECASE
            )
            if m:
                criteria.presentation_period_days = int(m.group(1))
                criteria.warnings.append(
                    f"Délai de présentation extrait de :47A: (fallback) : {m.group(1)} jours"
                )

        # ── Date d'expiration :31D: ───────────────────────────────────
        criteria.expiry_date  = entities.expiry_date
        criteria.expiry_place = entities.expiry_place

        # ── Disponibilité :41a: ───────────────────────────────────────
        # available_by = mode de règlement (BY PAYMENT/NEGOTIATION…).
        # MT700 le stocke dans available_by ; MT720 (historique) dans available_with.
        criteria.available_by = (
            getattr(entities, "available_by", None) or entities.available_with
        )
        criteria.draft_at     = entities.draft_at

        # ── Frais :71B: ───────────────────────────────────────────────
        if entities.charges:
            v = entities.charges.upper()
            # Chercher explicitement "BENEFICIARY" dans les frais
            # et vérifier que la valeur ressemble vraiment à des frais
            is_charges = bool(re.search(
                r'(?:BANK\s+CHARGES?|ALL\s+CHARGES?|FOR\s+ACCOUNT|'
                r'FOR\s+(?:BENEFICIARY|APPLICANT)|OUTSIDE|INSIDE)',
                v
            ))
            if is_charges:
                criteria.charges_for_beneficiary = "BENEFICIARY" in v
                criteria.charges_raw = entities.charges
            else:
                # Valeur ambiguë — conserver raw mais ne pas déduire charges_for_beneficiary
                criteria.charges_raw = entities.charges
                criteria.warnings.append(
                    f":71B: valeur ambiguë (non reconnue comme frais) : {entities.charges!r}"
                )

        # B15 : Fallback — extraire charges_for_beneficiary depuis :47A: si :71B: null
        # Pattern : 'ALL BANK CHARGES OUTSIDE [COUNTRY] [ARE] FOR BENEFICIARY'
        #           'ALL BANK CHARGES OUTSIDE [COUNTRY] FOR BENEFICIARY'S ACCOUNT'
        if criteria.charges_for_beneficiary is None and entities.additional_conditions:
            ac = entities.additional_conditions.upper()
            # Chercher la règle OUTSIDE → BENEFICIARY (exportation standard)
            m_outside = re.search(
                r'(?:ALL\s+)?BANK\s+CHARGES?\s+OUTSIDE\s+\w+\s+'
                r'(?:ARE\s+)?FOR\s+(BENEFICIARY|APPLICANT)',
                ac
            )
            if m_outside:
                criteria.charges_for_beneficiary = m_outside.group(1) == "BENEFICIARY"
                # Extraire la phrase complète comme raw
                start = max(0, m_outside.start() - 10)
                end   = min(len(ac), m_outside.end() + 20)
                criteria.charges_raw = entities.additional_conditions[start:end].strip()
                criteria.warnings.append(
                    "Frais extraits de :47A: (fallback — :71B: absent)"
                )

        # ── Confirmation :49: ─────────────────────────────────────────
        criteria.confirmation = entities.confirmation

        # ── Marchandises :45A: ────────────────────────────────────────
        criteria.goods_description = entities.goods_description

        # ── Règles :40E: ──────────────────────────────────────────────
        criteria.applicable_rules = entities.applicable_rules

        # ── Avertissements ────────────────────────────────────────────
        self._add_warnings(criteria)

        return criteria

    # ------------------------------------------------------------------
    # Parsing des documents requis
    # ------------------------------------------------------------------

    def _parse_documents(self, text: str) -> list[DocumentRequirement]:
        docs: list[DocumentRequirement] = []
        items = self._split_document_items(text)
        for item in items:
            if not item.strip():
                continue
            doc_type   = self._identify_document_type(item)
            copies     = self._extract_copies(item)
            conditions = self._extract_document_conditions(item)
            docs.append(DocumentRequirement(
                document_type=doc_type,
                copies=copies,
                conditions=conditions,
                raw=item.strip(),
            ))
        return docs

    def _parse_documents_from_goods(self, goods_text: str) -> list[DocumentRequirement]:
        """
        Tente d'extraire des documents depuis :45A: quand :46A: est absent.
        v5.2 : détecte aussi les listes dans le texte inline (sans saut de ligne)
        en cherchant le premier item numéroté.
        """
        # Chercher une liste numérotée (avec ou sans newline avant)
        # Pattern plus large : '1.' peut apparaître inline après la description
        numbered_start = re.search(r'(?:^|\s)1[\.\)]\s*(?:SIGNED|ORIGINAL|FULL|COMMERCIAL|BILL|PACK|CERT|INS)',
                                   goods_text, re.IGNORECASE)
        if not numbered_start and not self._DOC_IN_GOODS_PATTERN.search(goods_text):
            return []

        # Extraire la partie "liste de documents" du champ goods
        doc_section = goods_text
        if numbered_start:
            doc_section = goods_text[numbered_start.start():]

        return self._parse_documents(doc_section)

    def _split_document_items(self, text: str) -> list[str]:
        """
        Sépare le texte en items de documents individuels.
        v5.4 : supporte aussi les listes numérotées INLINE (sans saut de ligne).
        Ex: '1. SIGNED INVOICE (3 COPIES) 2. FULL SET BILL OF LADING 3. PACKING LIST'
        """
        # 1. Split sur numérotation avec saut de ligne (format normal)
        numbered = re.split(r'(?:^|\n)\s*(?:\d+[\.\)]\s*|/\d+/\s*)', text)
        if len(numbered) > 1:
            items = [i.strip() for i in numbered if i.strip()]
            # Chaque item peut encore contenir un numéro inline → re-split si nécessaire
            final = []
            for item in items:
                sub = re.split(r'\s+(?=\d+[\.\)]\s+[A-Z])', item)
                sub = [re.sub(r'^\d+[\.\)]\s*', '', s).strip() for s in sub if s.strip()]
                final.extend(sub)
            return final

        # 2. Split sur numérotation INLINE (même ligne, espace avant chiffre+point+majuscule)
        #    Ex: 'SIGNED INVOICE 2. FULL SET BILL OF LADING 3. PACKING LIST'
        inline_split = re.split(r'\s+(?=\d+[\.\)]\s+[A-Z])', text)
        if len(inline_split) > 1:
            return [re.sub(r'^\d+[\.\)]\s*', '', i).strip() for i in inline_split if i.strip()]

        # 3. Split sur séparateurs '+' ou '-' en début d'item
        plus_split = re.split(r'\n\s*[\+\-]\s*', text)
        if len(plus_split) > 1:
            return [re.sub(r'^[\+\-]\s*', '', i).strip() for i in plus_split if i.strip()]

        # 4. Fallback : lignes
        return [l.strip() for l in text.splitlines() if l.strip()]

    def _identify_document_type(self, text: str) -> str:
        """Identifie le type de document par pattern matching."""
        for doc_type, pattern in self._DOCUMENT_PATTERNS:
            if pattern.search(text):
                return doc_type
        words = text.strip().split()
        return " ".join(words[:4]).upper() if words else "UNKNOWN DOCUMENT"

    @staticmethod
    def _extract_copies(text: str) -> Optional[int]:
        """Extrait le nombre de copies demandées."""
        patterns = [
            r"\((\d+)\s*COPIES?\)",
            r"(\d+)\s*COPIES?",
            r"(\d+)\s*ORIGINALS?",
            r"(\d+)\s+(?:FOLD|SETS?)",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                return int(m.group(1))
        if re.search(r"\bORIGINAL\b", text, re.I):
            return 1
        return None

    def _extract_document_conditions(self, text: str) -> list[str]:
        """Extrait les conditions associées au document."""
        conditions = []
        for cond_name, pattern in self._CONDITION_PATTERNS:
            if pattern.search(text):
                conditions.append(cond_name)
        return conditions

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_conditions(text: str) -> list[str]:
        """Extrait chaque condition individuelle du champ :47A:."""
        conditions = []
        for line in text.splitlines():
            line = line.strip()
            if line and len(line) > 3:
                conditions.append(line)
        return conditions

    @staticmethod
    def _add_warnings(criteria: LCAcceptanceCriteria) -> None:
        if not criteria.documents:
            criteria.warnings.append("Aucun document requis trouvé dans :46A: ni :45A:")
        if criteria.presentation_period_days is None:
            criteria.warnings.append("Délai de présentation :48: absent — défaut UCP : 21 jours")
        if criteria.expiry_date is None:
            criteria.warnings.append("Date d'expiration :31D: absente")
        if criteria.available_by is None:
            criteria.warnings.append("Disponibilité :41a: absente")