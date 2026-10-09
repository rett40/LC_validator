"""
patterns.py  —  v5.1
Tags SWIFT et mots-clés par type de message.
Corrections v5.1 :
  - MT720 : poids :21: 4→8, :40B: 5→8 pour éviter confusion avec MT700
  - MT799 : ajout mots-clés free-text pour détecter les textes sans tags
  - MT760 : réduction poids mots-clés génériques (GUARANTEE seul → 2)
"""

SWIFT_TAGS: dict[str, list[str]] = {

    "MT700": [
        ":20:", ":27:", ":40A:", ":31C:", ":40E:", ":31D:",
        ":50:", ":59:", ":32B:", ":41A:", ":41D:",
        ":42C:", ":43P:", ":43T:",
        ":44A:", ":44B:", ":44C:", ":44E:", ":44F:",
        ":45A:", ":46A:", ":47A:", ":48:", ":49:", ":71B:",
    ],

    "MT720": [
        ":20:", ":21:", ":27:", ":40B:", ":31C:", ":40E:", ":31D:",
        ":50:", ":59:", ":32B:", ":52A:", ":52D:",
        ":41A:", ":41D:", ":42C:", ":43P:", ":43T:",
        ":44A:", ":44B:", ":44C:", ":45A:", ":46A:", ":47A:",
        ":48:", ":49:", ":71B:",
    ],

    "MT760": [
        ":20:", ":27:", ":22A:", ":22D:", ":23B:",
        ":30:", ":40C:", ":32B:", ":50:", ":59:", ":77U:",
        ":72Z:", ":31E:",
    ],

    "MT790": [
        ":20:", ":21:", ":25:", ":32A:", ":52A:", ":52D:",
        ":71B:", ":72:",
    ],

    "MT799": [
        ":20:", ":21:", ":79:", ":72:",
    ],
}

# ---------------------------------------------------------------------------
# Poids des tags — v5.1
# MT720 : :21: et :40B: très discriminants (exclusifs MT720)
# ---------------------------------------------------------------------------

SWIFT_TAG_WEIGHTS: dict[str, dict[str, int]] = {

    "MT700": {
        ":40A:": 5, ":45A:": 4, ":46A:": 4, ":47A:": 4,
        ":43P:": 3, ":43T:": 3, ":42C:": 3, ":49:":  3,
        ":40E:": 2, ":48:":  2, ":71B:": 2,
        ":20:":  1, ":27:":  1, ":32B:": 1, ":50:":  1, ":59:": 1,
    },

    "MT720": {
        # Tags exclusifs MT720 → poids très élevés
        ":21:":  8,   # Documentary Credit Number (absent de MT700)
        ":40B:": 8,   # Form of Documentary Credit (absent de MT700)
        ":52A:": 4, ":52D:": 4,  # Issuing bank of original credit
        ":43P:": 2, ":43T:": 2, ":42C:": 2,
        ":40E:": 2, ":48:":  2,
        ":20:":  1, ":27:":  1, ":32B:": 1, ":50:":  1, ":59:": 1,
    },

    "MT760": {
        ":22A:": 5, ":22D:": 5, ":77U:": 5,
        ":23B:": 4, ":40C:": 4,
        ":30:":  3, ":72Z:": 3, ":31E:": 3,
        ":20:":  1, ":32B:": 1, ":50:":  1, ":59:": 1,
    },

    "MT790": {
        ":25:":  5, ":32A:": 5,
        ":52A:": 3, ":52D:": 3,
        ":71B:": 2, ":72:":  2,
        ":20:":  1, ":21:":  1,
    },

    "MT799": {
        ":79:":  6,
        ":72:":  2,
        ":20:":  1, ":21:":  1,
    },
}

# ---------------------------------------------------------------------------
# Mots-clés par type
# ---------------------------------------------------------------------------

SWIFT_KEYWORDS: dict[str, list[str]] = {

    "MT700": [
        "DOCUMENTARY CREDIT",
        "IRREVOCABLE",
        "IRREVOCABLE TRANSFERABLE",
        "ISSUE OF A DOCUMENTARY",
        "LETTER OF CREDIT",
        "UCP",
        "UCPURR",
        "EUCP",
        "PARTIAL SHIPMENTS",
        "TRANSHIPMENT",
        "PORT OF LOADING",
        "PORT OF DISCHARGE",
        "DESCRIPTION OF GOODS",
        "DOCUMENTS REQUIRED",
        "PERIOD FOR PRESENTATION",
        "CONFIRMATION INSTRUCTIONS",
    ],

    "MT720": [
        "TRANSFER OF A DOCUMENTARY CREDIT",
        "TRANSFER OF DOCUMENTARY CREDIT",
        "FIRST BENEFICIARY",
        "SECOND BENEFICIARY",
        "TRANSFERRING BANK",
        "ORIGINAL CREDIT",
        "NOMINATED",
    ],

    "MT760": [
        "STANDBY LETTER OF CREDIT",
        "SBLC",
        "DEMAND GUARANTEE",
        "UNDERTAKING",
        "URDG",
        "ISPR",
        "PERFORMANCE BOND",
        "BID BOND",
        "ADVANCE PAYMENT GUARANTEE",
    ],

    "MT790": [
        "ADVICE OF CHARGES",
        "CHARGES",
        "INTEREST",
        "REIMBURSEMENT",
        "ADJUSTMENT",
        "VALUE DATE",
    ],

    "MT799": [
        "FREE FORMAT",
        "FREE FORMAT MESSAGE",
        "BANK TO BANK",
        "BANK MESSAGE",
        "PRE-ADVICE",
        "PREADVICE",
        "HEREBY CONFIRM",
        "HEREBY IRREVOCABLY",
        "SAMPLE MT",
        "RESPONSE FROM",
    ],
}

# ---------------------------------------------------------------------------
# Poids des mots-clés — v5.1
# MT760 : "GUARANTEE" seul réduit à 2 (trop générique, présent dans MT799)
# MT799 : boost free-text indicators
# ---------------------------------------------------------------------------

SWIFT_KEYWORD_WEIGHTS: dict[str, dict[str, int]] = {

    "MT700": {
        "DOCUMENTARY CREDIT": 4,
        "IRREVOCABLE":        3,
        "UCP":                3,
        "PARTIAL SHIPMENTS":  3,
        "DOCUMENTS REQUIRED": 3,
        "PORT OF LOADING":    2,
        "LETTER OF CREDIT":   2,
        "PERIOD FOR PRESENTATION": 2,
    },

    "MT720": {
        "TRANSFER OF A DOCUMENTARY CREDIT": 8,
        "TRANSFER OF DOCUMENTARY CREDIT":   8,
        "FIRST BENEFICIARY":  5,
        "SECOND BENEFICIARY": 5,
        "TRANSFERRING BANK":  4,
        "NOMINATED":          3,
    },

    "MT760": {
        "DEMAND GUARANTEE":           5,
        "STANDBY LETTER OF CREDIT":   5,
        "SBLC":                       4,
        "URDG":                       4,
        "UNDERTAKING":                3,
        "PERFORMANCE BOND":           3,
        "ADVANCE PAYMENT GUARANTEE":  3,
        # Réduit : trop générique
        "GUARANTEE":                  2,
    },

    "MT790": {
        "ADVICE OF CHARGES":  5,
        "VALUE DATE":         3,
        "REIMBURSEMENT":      3,
        "CHARGES":            2,
    },

    "MT799": {
        "FREE FORMAT MESSAGE": 5,
        "FREE FORMAT":         4,
        "PRE-ADVICE":          4,
        "PREADVICE":           4,
        "BANK TO BANK":        3,
        "HEREBY CONFIRM":      3,
        "HEREBY IRREVOCABLY":  3,
        "SAMPLE MT":           3,
        "RESPONSE FROM":       2,
    },
}

# ---------------------------------------------------------------------------
# Codes MT dans le bloc 2 du header SWIFT {2:Innn...}
# ---------------------------------------------------------------------------

MT_CODES: dict[str, str] = {
    "700": "MT700",
    "720": "MT720",
    "760": "MT760",
    "790": "MT790",
    "799": "MT799",
}

# ---------------------------------------------------------------------------
# Seuil minimal de tags SWIFT réels pour valider une détection
# En dessous → forcer MT799 (texte libre sans structure SWIFT)
# ---------------------------------------------------------------------------
MIN_SWIFT_TAGS_FOR_STRUCTURED = 3