"""
Schéma de features risk ML v4.0 — source unique pour l'entraînement et l'inférence.

Changements v4.0 (vs v3.0) :
  - 26 features au lieu de 15 : +11 features couvrant les dimensions manquantes
  - ucp_errors_count / ucp_warnings_count séparés (poids différents en règles + breakdown)
  - missing_critical_score : taux de présence des champs MT700 critiques
  - country_risk_applicant / country_risk_beneficiary : risque pays séparé applicant/bénéficiaire
  - contradiction_flag : détection de clauses contradictoires
  - coherence_dates : cohérence séquentielle issue_date ≤ shipment ≤ expiry
  - amount_tier : classification montant LC (1=très petit … 5=très large)
  - red_clause_flag : détection clause rouge dans champ 47A
  - bank_tier : tier de la banque advising/confirming (1=Tier1 … 5=inconnu)
  - tenor_risk : risque lié au type de tenor (SIGHT=0.1 … 180 DAYS=0.9)
  - Backward compat : FEATURE_COLS_V3 = 15 features pour modèles déjà entraînés

Règle d'or :
  À l'inférence, la matrice X suit exactement metadata.json["feature_cols"]
  (alignée sur scaler.n_features_in_), jamais une liste codée en dur.
  Les 11 nouvelles features alimentent le rule engine et le score_breakdown
  même si le modèle ML courant ne les utilise pas encore.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# VERSION
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_SCHEMA_VERSION = "4.0"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES PARTAGÉES générateur ↔ moteur ↔ validation externe
# ─────────────────────────────────────────────────────────────────────────────

SANCTIONED_ISO2: frozenset = frozenset({"KP", "IR", "MM", "SY", "CU", "SD", "BY"})

INCOTERM_RISK_MAP: dict[str, int] = {
    "EXW": 5,
    "FCA": 4, "FAS": 4, "FOB": 4,
    "CFR": 3, "CPT": 3, "CIP": 3,
    "CIF": 2, "DAP": 2, "DPU": 2,
    "DDP": 1,
}
INCOTERM_RISK_DEFAULT = 3

# ─────────────────────────────────────────────────────────────────────────────
# CHAMPS MT700 — inventaire criticité pour missing fields analysis
# ─────────────────────────────────────────────────────────────────────────────

# CRITICAL : sans ces champs, le LC est juridiquement incomplet (pénalité -10 pts/champ)
MT700_CRITICAL_FIELDS: dict[str, dict] = {
    "20":  {"name": "Documentary Credit Number",    "keys": ["lc_number", "documentary_credit_number", "reference", "lc_ref", "doc_credit_number"], "impact": -10},
    "31D": {"name": "Date and Place of Expiry",     "keys": ["expiry_date", "validity_date", "date_expiry", "expiry_place"], "impact": -10},
    "32B": {"name": "Currency Code / Amount",       "keys": ["amount", "lc_amount", "currency", "currency_amount", "currency_code"], "impact": -10},
    "40A": {"name": "Form of Documentary Credit",   "keys": ["form", "lc_type", "credit_type", "form_of_credit", "documentary_credit_type"], "impact": -10},
    "50":  {"name": "Applicant",                    "keys": ["applicant", "applicant_name", "buyer", "importer"], "impact": -10},
    "59":  {"name": "Beneficiary",                  "keys": ["beneficiary", "beneficiary_name", "seller", "exporter"], "impact": -10},
    "41D": {"name": "Available With ... By",        "keys": ["available_with", "available_by", "availability", "available_with_by"], "impact": -10},
    "45A": {"name": "Description of Goods / Services", "keys": ["goods_description", "goods", "commodity", "description_of_goods", "goods_services"], "impact": -10},
}

# HIGH : champs importants dont l'absence génère des discrepancies fréquentes (pénalité -5 pts)
MT700_HIGH_FIELDS: dict[str, dict] = {
    "44C": {"name": "Latest Date of Shipment",      "keys": ["latest_shipment_date", "shipment_date", "latest_date_shipment", "latest_ship_date"], "impact": -5},
    "48":  {"name": "Period for Presentation",      "keys": ["presentation_period", "presentation_period_days", "period_presentation", "presentation_days"], "impact": -5},
    "42C": {"name": "Drafts at",                    "keys": ["drafts_at", "tenor", "draft_tenor", "drafts_tenor"], "impact": -5},
    "43P": {"name": "Partial Shipments",            "keys": ["partial_shipments", "partial_shipment", "partials"], "impact": -5},
    "43T": {"name": "Transshipment",                "keys": ["transshipment", "transshipment_allowed", "trans_shipment"], "impact": -5},
    "46A": {"name": "Documents Required",           "keys": ["documents_required", "required_documents", "docs_required"], "impact": -5},
    "47A": {"name": "Additional Conditions",        "keys": ["additional_conditions", "special_conditions", "conditions", "field_47a"], "impact": -5},
}

# OPTIONAL : présence souhaitable mais non bloquante (pénalité -1 pt)
MT700_OPTIONAL_FIELDS: dict[str, dict] = {
    "57A": {"name": "Advise Through Bank",          "keys": ["advising_bank", "advise_through_bank", "advising_bank_name"], "impact": -1},
    "58A": {"name": "Requested Confirmation Party", "keys": ["confirming_bank", "requested_confirmation", "confirmation_party"], "impact": -1},
    "71B": {"name": "Charges",                      "keys": ["charges", "bank_charges", "charges_beneficiary"], "impact": -1},
    "72":  {"name": "Sender to Receiver Info",      "keys": ["sender_info", "sender_to_receiver", "sender_receiver_info"], "impact": -1},
    "26E": {"name": "Number of Amendment",          "keys": ["amendment_number", "number_of_amendment", "amendment_no"], "impact": -1},
}

# ─────────────────────────────────────────────────────────────────────────────
# AMOUNT TIERS — classification du montant LC
# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 : < 50k USD (très petit — risque opérationnel faible mais surveillance fraude)
# Tier 2 : 50k–250k (petit standard)
# Tier 3 : 250k–1M (standard)
# Tier 4 : 1M–5M (large — surveillance renforcée)
# Tier 5 : > 5M (très large — revue senior obligatoire)
AMOUNT_TIER_THRESHOLDS: list[tuple[float, int]] = [
    (50_000.0,      1),
    (250_000.0,     2),
    (1_000_000.0,   3),
    (5_000_000.0,   4),
    (float("inf"),  5),
]

# ─────────────────────────────────────────────────────────────────────────────
# COMMODITY RISK — risque lié à la marchandise (champ 45A)
# ─────────────────────────────────────────────────────────────────────────────
COMMODITY_RISK_MAP: dict[str, float] = {
    # Très haut risque (4-5)
    "arms":          5.0, "weapons":      5.0, "munitions":    5.0, "explosives":   5.0,
    "oil":           4.5, "crude":        4.5, "petroleum":    4.5, "fuel":         4.0,
    "diamonds":      4.0, "precious stones": 4.0,
    "chemicals":     4.0, "chemical":     4.0, "hazardous":    4.0,
    # Haut risque (3-4)
    "pharmaceutical": 3.5, "pharma":      3.5, "medicine":     3.5, "drugs":        3.5,
    "fertilizer":    3.5, "fertilisers":  3.5,
    "gold":          3.5, "silver":       3.0, "precious metals": 3.5,
    "luxury":        3.0, "jewelry":      3.5, "watches":      3.0,
    "tobacco":       3.5, "cigarettes":   3.5,
    "electronics":   3.0, "semiconductors": 3.5, "microchips":  3.5,
    # Risque moyen (2-3)
    "tech":          2.5, "technology":   2.5,
    "vehicles":      2.5, "automotive":   2.5, "cars":         2.5,
    "steel":         2.0, "iron":         2.0, "metals":       2.5, "aluminum":     2.0,
    "machinery":     2.0, "equipment":    2.0, "industrial":   2.0,
    "timber":        2.5, "wood":         2.0, "lumber":       2.0,
    # Faible risque (1-2)
    "food":          1.5, "grain":        1.5, "wheat":        1.5, "rice":         1.5,
    "textile":       1.5, "clothing":     1.5, "garments":     1.5, "fabric":       1.5,
    "furniture":     1.5, "plastics":     1.5,
    "coffee":        1.5, "tea":          1.5, "cocoa":        1.5,
    "fish":          1.5, "seafood":      1.5,
    "cotton":        1.5, "leather":      2.0,
}
COMMODITY_RISK_DEFAULT: float = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# BANK TIER — risque lié au rang de la banque (champs 57A / 58A)
# Tier 1 = grandes banques mondiales (risque minimal)
# Tier 5 = banque inconnue / non identifiée (risque élevé)
# ─────────────────────────────────────────────────────────────────────────────
BANK_TIER_KEYWORDS: dict[int, list[str]] = {
    1: [
        "HSBC", "BNP PARIBAS", "BNP", "BNPP",
        "CITIBANK", "CITI",
        "STANDARD CHARTERED", "STANCHART", "SCBL",
        "DEUTSCHE BANK", "DEUTSCHE",
        "BARCLAYS",
        "JP MORGAN", "JPMORGAN", "JPMORGANCHASE",
        "WELLS FARGO",
        "SOCIETE GENERALE", "SG",
        "CREDIT AGRICOLE", "CA-CIB",
        "UNICREDIT",
        "ING BANK", "ING",
        "NATIXIS",
        "ABN AMRO",
        "RABOBANK",
        "BANK OF AMERICA", "BOFA",
        "GOLDMAN SACHS",
        "MORGAN STANLEY",
        "COMMERZBANK",
        "INTESA SANPAOLO",
        "SUMITOMO", "MUFG", "SMBC",
        "DBS", "OCBC", "UOB",
    ],
    2: [
        "SANTANDER", "BBVA",
        "ATTIJARIWAFA", "WAFA",
        "BMCE", "BANK OF AFRICA",
        "BMCI",
        "BANQUE POPULAIRE", "BP MAROC",
        "CIH BANK", "CIH",
        "AL BARID BANK",
        "CREDIT DU MAROC", "CDM",
        "CIB EGYPT",
        "EFG HERMES", "EFG",
        "MASHREQ",
        "EMIRATES NBD", "ENBD",
        "FIRST ABU DHABI", "FAB",
        "DUBAI ISLAMIC", "DIB",
        "COMMERCIAL BANK", "CBQ",
        "RIYAD BANK",
        "NCB", "NATIONAL COMMERCIAL",
        "ARAB NATIONAL BANK", "ANB",
        "BANQUE ZITOUNA",
        "STB", "BIAT", "UBCI",
        "BNA", "BH BANK",
        "GULF BANK",
        "KUWAIT FINANCE",
        "BOUBYAN",
    ],
    3: [
        "AFREXIMBANK", "AFRILAND", "UBA", "GTB", "ZENITH",
        "HABIB", "MCB", "ALLIED BANK", "FAYSAL",
        "BANK ALFALAH", "ASKARI",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# BANK TIER par préfixe BIC (4 premières lettres = institution code)
# Indispensable : les champs :52A:/:57A:/:41A: portent souvent un BIC
# (ex. DEUTDEFFXXX) et non un nom — les mots-clés ci-dessus ne matchent jamais.
# ─────────────────────────────────────────────────────────────────────────────
BANK_BIC_PREFIX_TIER: dict[str, int] = {
    # ── Tier 1 ────────────────────────────────────────────────────────────────
    "DEUT": 1,                          # Deutsche Bank
    "HSBC": 1, "HBUK": 1, "MIDL": 1,    # HSBC
    "BNPA": 1,                          # BNP Paribas
    "CITI": 1,                          # Citibank
    "SCBL": 1,                          # Standard Chartered
    "BARC": 1,                          # Barclays
    "CHAS": 1, "MORG": 1,               # JP Morgan Chase
    "WFBI": 1,                          # Wells Fargo
    "SOGE": 1,                          # Societe Generale
    "AGRI": 1, "BSUI": 1,               # Credit Agricole / CA-CIB
    "UNCR": 1,                          # UniCredit
    "INGB": 1,                          # ING
    "NATX": 1,                          # Natixis
    "ABNA": 1,                          # ABN AMRO
    "RABO": 1,                          # Rabobank
    "BOFA": 1,                          # Bank of America
    "GOLD": 1,                          # Goldman Sachs
    "COBA": 1,                          # Commerzbank
    "BCIT": 1,                          # Intesa Sanpaolo
    "SMBC": 1, "BOTK": 1, "MHCB": 1,    # SMBC / MUFG / Mizuho
    "DBSS": 1, "OCBC": 1, "UOVB": 1,    # DBS / OCBC / UOB
    # ── Tier 2 ────────────────────────────────────────────────────────────────
    "BSCH": 2,                          # Santander
    "BBVA": 2,                          # BBVA
    "BCMA": 2,                          # Attijariwafa Bank
    "BMCE": 2,                          # BMCE / Bank of Africa
    "BMCI": 2,                          # BMCI
    "BCPO": 2, "GPBM": 2,               # Banque Populaire (Maroc)
    "CIHM": 2,                          # CIH Bank
    "EBILAEAD": 2, "EBIL": 2,           # Emirates NBD
    "FABM": 2, "NBAD": 2,               # First Abu Dhabi
    "DUIB": 2,                          # Dubai Islamic
    "RIBL": 2,                          # Riyad Bank
    "NCBK": 2,                          # National Commercial Bank
    "ARNB": 2,                          # Arab National Bank
    "BZIT": 2,                          # Banque Zitouna
    "STBK": 2, "BIAT": 2, "UBCI": 2,    # STB / BIAT / UBCI (Tunisie)
    "BNTE": 2, "BANBTNTT": 2,           # BNA Tunisie
}

# ─────────────────────────────────────────────────────────────────────────────
# TENOR RISK — risque lié au tenor (champ 42C)
# ─────────────────────────────────────────────────────────────────────────────
TENOR_RISK_MAP: dict[str, float] = {
    "AT SIGHT":   0.10, "SIGHT":     0.10,
    "7 DAYS":     0.20, "7D":        0.20,
    "15 DAYS":    0.25, "15D":       0.25,
    "30 DAYS":    0.35, "30D":       0.35, "ONE MONTH":  0.35,
    "45 DAYS":    0.42, "45D":       0.42,
    "60 DAYS":    0.50, "60D":       0.50, "TWO MONTHS": 0.50,
    "90 DAYS":    0.65, "90D":       0.65, "THREE MONTHS": 0.65,
    "120 DAYS":   0.75, "120D":      0.75,
    "150 DAYS":   0.80, "150D":      0.80,
    "180 DAYS":   0.88, "180D":      0.88, "SIX MONTHS": 0.88,
    "270 DAYS":   0.92, "270D":      0.92,
    "360 DAYS":   0.95, "360D":      0.95, "ONE YEAR":   0.95,
    "USANCE":     0.60,
    "DEFERRED":   0.70,
    "ACCEPTANCE": 0.65,
    "MIXED PAYMENT": 0.55,
    "NEGOTIATION": 0.40,
}
TENOR_RISK_DEFAULT: float = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMA v4.0 — 26 features canoniques
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS_CANONICAL: list[str] = [
    # ── Financier / Géographique ──────────────────────────────────────────────
    "lc_amount",               # Montant LC (USD ou équivalent)
    "country_risk_score",      # Risque pays consolidé 0-100 (max applicant/bénéficiaire/OFAC)
    "external_currency_risk",  # Risque devise 0-4 (external_validation API)
    "incoterm_risk",           # Risque incoterm 0-5 (DDP=1 … EXW=5)

    # ── Compliance UCP 600 granulaire ─────────────────────────────────────────
    "ucp_compliance_score",    # Score compliance global 0-1 (global_score/100)
    "n_ucp_warnings",          # Total violations UCP (errors + warnings, backward compat)
    "ucp_warning_rate",        # Densité violations = total / clauses analysées
    "ucp_errors_count",        # Erreurs UCP seules (poids 1.0)
    "ucp_warnings_count",      # Avertissements UCP seuls (poids 0.4)

    # ── Qualité documentaire NLP ──────────────────────────────────────────────
    "nlp_score",               # Score qualité clauses 0-100
    "semantic_score",          # Cohérence sémantique 0-1
    "completeness_score",      # Complétude champs MT700 0-1 (score global)
    "missing_critical_score",  # Taux de présence des champs CRITICAL 0-1 (1=tous présents)

    # ── Cohérence croisée ─────────────────────────────────────────────────────
    "coherence_dates",         # Cohérence dates issue ≤ shipment ≤ expiry (0-1)
    "contradiction_flag",      # Clauses contradictoires détectées (0/1)

    # ── Risque pays dual ──────────────────────────────────────────────────────
    "country_risk_applicant",  # Risque pays applicant 0-100
    "country_risk_beneficiary",# Risque pays bénéficiaire 0-100

    # ── Temporel ──────────────────────────────────────────────────────────────
    "days_to_shipment",        # Jours avant expédition
    "presentation_days",       # Période de présentation en jours
    "validity_days",           # Durée validité LC en jours

    # ── Montant ───────────────────────────────────────────────────────────────
    "amount_tier",             # Tier montant 1-5 (1=<50k … 5=>5M USD)

    # ── Indicateurs structurels ───────────────────────────────────────────────
    "transshipment_allowed",   # Transbordement autorisé (0/1)
    "charges_beneficiary",     # Frais à charge du bénéficiaire (0/1)
    "red_clause_flag",         # Clause rouge détectée dans 47A (0/1)

    # ── Contreparties et tenor ────────────────────────────────────────────────
    "bank_tier",               # Tier banque advising/confirming (1=Tier1 … 5=inconnu)
    "tenor_risk",              # Risque tenor 0-1 (SIGHT=0.1 … 360 jours=0.95)
]

# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMA LEGACY v3.0 — 15 features — backward compat avec modèles déjà entraînés
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS_V3: list[str] = [
    "lc_amount",
    "country_risk_score",
    "external_currency_risk",
    "incoterm_risk",
    "ucp_compliance_score",
    "n_ucp_warnings",
    "ucp_warning_rate",
    "nlp_score",
    "semantic_score",
    "completeness_score",
    "days_to_shipment",
    "presentation_days",
    "validity_days",
    "transshipment_allowed",
    "charges_beneficiary",
]

# SCHÉMA LEGACY v2.2 — 22 features
FEATURE_COLS_LEGACY_22: list[str] = [
    "lc_amount", "country_risk", "country_risk_level", "country_risk_score",
    "external_country_score", "external_country_level", "external_currency_risk",
    "incoterm_risk", "n_clauses", "n_documents_required", "ucp_compliance_score",
    "n_ucp_warnings", "ucp_warning_rate", "ucp_decision_reject",
    "nlp_score", "semantic_score", "completeness_score",
    "days_to_shipment", "presentation_days", "validity_days",
    "transshipment_allowed", "charges_beneficiary",
]

# Aliases backward-compat
FEATURE_COLS_LEGACY_20 = FEATURE_COLS_LEGACY_22


# ─────────────────────────────────────────────────────────────────────────────
# VALEURS PAR DÉFAUT (26 features)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DEFAULTS_V4: dict[str, float] = {
    # Existantes v3.0
    "lc_amount":               500_000.0,
    "country_risk_score":          25.0,
    "external_currency_risk":       0.0,
    "incoterm_risk":                float(INCOTERM_RISK_DEFAULT),
    "ucp_compliance_score":         0.5,
    "n_ucp_warnings":               2.0,
    "ucp_warning_rate":             0.2,
    "nlp_score":                   50.0,
    "semantic_score":               0.5,
    "completeness_score":           0.5,
    "days_to_shipment":            30.0,
    "presentation_days":           21.0,
    "validity_days":               90.0,
    "transshipment_allowed":        0.0,
    "charges_beneficiary":          0.0,
    # Nouvelles v4.0
    "ucp_errors_count":             1.0,
    "ucp_warnings_count":           1.0,
    "missing_critical_score":       0.75,  # 6/8 champs présents par défaut (prudent)
    "coherence_dates":              0.5,
    "contradiction_flag":           0.0,
    "country_risk_applicant":      25.0,
    "country_risk_beneficiary":    25.0,
    "amount_tier":                  3.0,   # Standard par défaut
    "red_clause_flag":              0.0,
    "bank_tier":                    3.0,   # Banque inconnue = tier 3 (prudent)
    "tenor_risk":                   TENOR_RISK_DEFAULT,
}

# Alias pour backward compat (risk_engine.py référence FEATURE_DEFAULTS)
FEATURE_DEFAULTS = FEATURE_DEFAULTS_V4


# ─────────────────────────────────────────────────────────────────────────────
# RÉSOLVEUR DE SCHÉMA — aligne les colonnes sur scaler.n_features_in_
# ─────────────────────────────────────────────────────────────────────────────

def resolve_matrix_feature_columns(
    metadata: dict | None,
    scaler_n_features: int | None,
) -> list[str]:
    """
    Retourne la liste de colonnes pour X avant MinMaxScaler.transform.
    Alignée sur scaler.n_features_in_ pour éviter tout mismatch silencieux.

    Priorité :
      1. metadata["feature_cols"] si len == scaler_n_features
      2. Fallback par taille connue (26=v4.0, 15=v3.0, 22=v2.2 legacy)
      3. metadata["feature_cols"] sans vérification de taille
      4. FEATURE_COLS_CANONICAL par défaut
    """
    meta      = metadata or {}
    meta_cols = meta.get("feature_cols")

    if scaler_n_features is not None:
        if isinstance(meta_cols, list) and len(meta_cols) == scaler_n_features:
            return list(meta_cols)

        if isinstance(meta_cols, list) and meta_cols:
            logger.warning(
                "metadata.feature_cols (%d cols) != scaler.n_features_in_ (%d) "
                "— fallback résolution par taille",
                len(meta_cols),
                scaler_n_features,
            )

        known: dict[int, list[str]] = {
            len(FEATURE_COLS_CANONICAL):  FEATURE_COLS_CANONICAL,
            len(FEATURE_COLS_V3):         FEATURE_COLS_V3,
            len(FEATURE_COLS_LEGACY_22):  FEATURE_COLS_LEGACY_22,
        }
        if scaler_n_features in known:
            version_label = {
                len(FEATURE_COLS_CANONICAL): "v4.0",
                len(FEATURE_COLS_V3):        "v3.0",
                len(FEATURE_COLS_LEGACY_22): "v2.2 legacy",
            }[scaler_n_features]
            logger.info(
                "Schéma résolu par taille : %d cols → %s",
                scaler_n_features,
                version_label,
            )
            return list(known[scaler_n_features])

        logger.error(
            "Taille scaler inconnue (%d) — aucun schéma correspondant. "
            "Réentraîner avec FEATURE_SCHEMA_VERSION=%s.",
            scaler_n_features,
            FEATURE_SCHEMA_VERSION,
        )

    if isinstance(meta_cols, list) and meta_cols:
        return list(meta_cols)

    return list(FEATURE_COLS_CANONICAL)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE — compute amount tier
# ─────────────────────────────────────────────────────────────────────────────

def compute_amount_tier(amount_usd: float) -> int:
    for threshold, tier in AMOUNT_TIER_THRESHOLDS:
        if amount_usd < threshold:
            return tier
    return 5


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE — compute commodity risk depuis texte 45A
# ─────────────────────────────────────────────────────────────────────────────

def compute_commodity_risk(goods_text: str) -> float:
    if not goods_text:
        return COMMODITY_RISK_DEFAULT
    text_lower = goods_text.lower()
    max_risk = 0.0
    for keyword, risk_val in COMMODITY_RISK_MAP.items():
        if keyword in text_lower:
            max_risk = max(max_risk, risk_val)
    return max_risk if max_risk > 0.0 else COMMODITY_RISK_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE — compute bank tier depuis nom banque
# ─────────────────────────────────────────────────────────────────────────────

def compute_bank_tier(bank_name: str) -> int:
    if not bank_name:
        return 4  # inconnu → tier 4 (prudent)
    name_upper = bank_name.upper().strip()

    # 1. Correspondance par mot-clé de nom (ex. "DEUTSCHE BANK AG")
    for tier, keywords in BANK_TIER_KEYWORDS.items():
        if any(kw in name_upper for kw in keywords):
            return tier

    # 2. Correspondance par BIC (ex. "DEUTDEFFXXX") — préfixe 4 lettres.
    #    Un BIC valide fait 8 ou 11 caractères alphanumériques sans espace.
    bic = re.sub(r"[^A-Z0-9]", "", name_upper)
    if len(bic) in (8, 11) and bic[:6].isalpha():
        # Préfixe 4 lettres (institution), puis repli sur des préfixes plus longs
        for plen in (8, 4):
            tier = BANK_BIC_PREFIX_TIER.get(bic[:plen])
            if tier is not None:
                return tier

    return 4  # banque connue mais non classée → tier 4


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE — compute tenor risk depuis texte 42C
# ─────────────────────────────────────────────────────────────────────────────

def compute_tenor_risk(tenor_text: str) -> float:
    if not tenor_text:
        return TENOR_RISK_DEFAULT
    text_upper = tenor_text.upper().strip()
    # Tri par longueur décroissante pour matcher les patterns les plus spécifiques d'abord
    for key in sorted(TENOR_RISK_MAP.keys(), key=len, reverse=True):
        if key in text_upper:
            return TENOR_RISK_MAP[key]
    # Tentative extraction numérique (ex: "USANCE 90 DAYS AFTER B/L DATE")
    import re
    nums = re.findall(r"\b(\d{1,3})\b", text_upper)
    for n_str in nums:
        n = int(n_str)
        if n <= 7:
            return 0.20
        elif n <= 30:
            return 0.35
        elif n <= 60:
            return 0.50
        elif n <= 90:
            return 0.65
        elif n <= 120:
            return 0.75
        elif n <= 180:
            return 0.88
        elif n <= 360:
            return 0.95
    return TENOR_RISK_DEFAULT
