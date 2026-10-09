"""
template_matcher.py
===================
Identifie la compagnie maritime émettrice (carrier) d'un document LC
et retourne des hints d'extraction spécifiques à ce carrier.

Les hints permettent d'ajuster dynamiquement :
  • header_ratio_override  — fraction de page réservée au header metadata
  • cargo_zone             — (y_start_ratio, y_end_ratio) de la zone cargo
  • bl_number_pattern      — regex spécifique du numéro de connaissement

Usage :
  from release2.services.template_matcher import TemplateMatcher
  tm = TemplateMatcher()
  carrier = tm.identify_carrier(full_text)   # "MSC" | "MAERSK" | None
  hints   = tm.get_hints(carrier)
"""

from __future__ import annotations

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Signatures des carriers connus
# ─────────────────────────────────────────────────────────────────────────────

CARRIER_SIGNATURES: dict[str, dict] = {
    "MSC": {
        "keywords": ["MSC MEDITERRANEAN", "MEDITERRANEAN SHIPPING CO", "MSC.COM"],
        "bl_number_pattern": r"MSC[A-Z]{2}\d{7}",
        "header_ratio_override": 0.30,
        "cargo_zone": (0.40, 0.85),
    },
    "MAERSK": {
        "keywords": ["MAERSK LINE", "MAERSK.COM", "A.P. MOLLER", "AP MOLLER"],
        "bl_number_pattern": r"\d{9}",
        "header_ratio_override": 0.28,
        "cargo_zone": (0.38, 0.80),
    },
    "CMA_CGM": {
        "keywords": ["CMA CGM", "CMA-CGM", "CMACGM.COM"],
        "bl_number_pattern": r"[A-Z]{4}\d{9}",
        "header_ratio_override": 0.32,
        "cargo_zone": (0.42, 0.82),
    },
    "HAPAG_LLOYD": {
        "keywords": ["HAPAG-LLOYD", "HAPAG LLOYD", "HAPAGCARRIERS.COM", "HLAG.COM"],
        "bl_number_pattern": r"\d{12}",
        "header_ratio_override": 0.30,
        "cargo_zone": (0.40, 0.80),
    },
    "EVERGREEN": {
        "keywords": ["EVERGREEN LINE", "EVERGREEN MARINE", "EVERGREENMARINE.COM"],
        "bl_number_pattern": r"[A-Z]{4}\d{10}",
        "header_ratio_override": 0.35,
        "cargo_zone": (0.45, 0.85),
    },
    "COSCO": {
        "keywords": ["COSCO SHIPPING", "COSCO LINE", "COSCOSHIPPING.COM"],
        "bl_number_pattern": r"[A-Z]{4}\d{10}",
        "header_ratio_override": 0.32,
        "cargo_zone": (0.42, 0.82),
    },
    "ONE": {
        "keywords": ["OCEAN NETWORK EXPRESS", "ONE LINE", "ONE-LINE.COM"],
        "bl_number_pattern": r"[A-Z]{4}\d{8}",
        "header_ratio_override": 0.30,
        "cargo_zone": (0.40, 0.82),
    },
    "YANG_MING": {
        "keywords": ["YANG MING", "YANGMING", "YML"],
        "bl_number_pattern": r"[A-Z]{4}\d{8}",
        "header_ratio_override": 0.32,
        "cargo_zone": (0.42, 0.82),
    },
    "ZIM": {
        "keywords": ["ZIM LINE", "ZIM INTEGRATED", "ZIM.COM"],
        "bl_number_pattern": r"[A-Z]{4}\d{8}",
        "header_ratio_override": 0.30,
        "cargo_zone": (0.40, 0.80),
    },
    "PIL": {
        "keywords": ["PACIFIC INTERNATIONAL LINES", "PIL GROUP", "PIL.COM.SG"],
        "bl_number_pattern": r"[A-Z]{4}\d{9}",
        "header_ratio_override": 0.32,
        "cargo_zone": (0.42, 0.82),
    },
}


class TemplateMatcher:
    """
    Identifie le carrier à partir du texte du document et retourne
    les hints d'extraction associés pour optimiser les zones d'analyse.
    """

    def identify_carrier(self, text: str) -> Optional[str]:
        """
        Identifie le carrier en cherchant ses keywords dans le texte.
        Retourne le nom du carrier ou None si non reconnu.
        """
        upper = text.upper()
        for carrier, sig in CARRIER_SIGNATURES.items():
            if any(kw.upper() in upper for kw in sig["keywords"]):
                log.debug("Carrier détecté : %s", carrier)
                return carrier
        return None

    def get_hints(self, carrier: Optional[str]) -> dict:
        """
        Retourne les hints d'extraction pour un carrier donné.
        Retourne un dict vide si carrier non reconnu.

        Clés retournées :
          carrier               — nom du carrier
          header_ratio_override — float, remplace LayoutAnalyzer.header_ratio
          cargo_zone            — (y_start_ratio, y_end_ratio) 0.0-1.0
          bl_number_pattern     — str regex ou None
        """
        if not carrier or carrier not in CARRIER_SIGNATURES:
            return {}
        sig = CARRIER_SIGNATURES[carrier]
        return {
            "carrier":               carrier,
            "header_ratio_override": sig.get("header_ratio_override"),
            "cargo_zone":            sig.get("cargo_zone"),
            "bl_number_pattern":     sig.get("bl_number_pattern"),
        }

    def get_layout_analyzer_kwargs(self, carrier: Optional[str]) -> dict:
        """
        Retourne les kwargs à passer au constructeur LayoutAnalyzer selon le carrier.
        """
        hints = self.get_hints(carrier)
        kwargs: dict = {}
        if "header_ratio_override" in hints:
            kwargs["header_ratio"] = hints["header_ratio_override"]
        return kwargs

    def extract_bl_number(self, text: str) -> Optional[str]:
        """
        Tente d'extraire le numéro de B/L en utilisant le pattern spécifique
        du carrier détecté. Retourne la première correspondance ou None.
        """
        carrier = self.identify_carrier(text)
        if not carrier:
            return None
        pattern = CARRIER_SIGNATURES[carrier].get("bl_number_pattern")
        if not pattern:
            return None
        match = re.search(pattern, text)
        return match.group(0) if match else None
