"""
spatial_extractor.py
====================
Tier 0 — Extraction spatiale depuis PDFs natifs (PyMuPDF, sans ML).

Quatre stratégies complémentaires :
  A. Inline KV     : "LABEL: value" ou "LABEL\\nVALUE" dans le même bloc
  B. Anchor        : bloc-label seul → bloc-valeur le plus proche dans
                     la même bande X (|Δcx| < 80 pts)
  C. Grid          : rangée de labels à même Y → rangée de valeurs en-dessous
  D. Party multi-  : ligne 0 = label de partie, lignes 1..n = valeur (
     ligne          société + adresse)

Retourne {"fields": {}, "tiers": {}, "confidences": {}}.

Confidences de base par stratégie :
  spatial_inline   0.88   (LABEL: value même ligne)
  spatial_party    0.85   (bloc multi-lignes partie)
  spatial_anchor   0.92   (label block → valeur block même X)
  spatial_grid     0.90   (grille header/data)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz

log = logging.getLogger(__name__)

# ── Confiances de base par stratégie ─────────────────────────────────────────

_CONF: dict[str, float] = {
    "spatial_inline":  0.88,
    "spatial_anchor":  0.92,
    "spatial_party":   0.85,
    "spatial_grid":    0.90,
}

# ── Dataclass bloc spatial ────────────────────────────────────────────────────

@dataclass
class SpatialBlock:
    text:         str
    x0:           float
    y0:           float
    x1:           float
    y1:           float
    page_width:   float
    page_height:  float
    page:         int
    lines:        list[str] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def rel_x(self) -> float:
        return self.cx / max(self.page_width, 1)

    @property
    def rel_y(self) -> float:
        return self.cy / max(self.page_height, 1)


# ── Tables de correspondance label → champ ───────────────────────────────────
# Clé = texte normalisé (upper, strip, espace unique)
# Valeur = (field_name, frozenset(doc_types acceptés))

_ALL = frozenset({"bl", "invoice", "packing_list"})
_BL  = frozenset({"bl"})
_INV = frozenset({"invoice"})
_PL  = frozenset({"packing_list"})
_BPL = frozenset({"bl", "packing_list"})
_BIL = frozenset({"bl", "invoice"})
_IPL = frozenset({"invoice", "packing_list"})

_LABEL_FIELD_MAP: dict[str, tuple[str, frozenset[str]]] = {
    # ── BL / PL : parties ────────────────────────────────────────────────
    "SHIPPER":                      ("shipper",              _BPL),
    "SHIPPER / EXPORTER":           ("shipper",              _BPL),
    "SHIPPER/EXPORTER":             ("shipper",              _BPL),
    "EXPORTER / SHIPPER":           ("shipper",              _BPL),
    "EXPORTER/SHIPPER":             ("shipper",              _BPL),
    "EXPORTER":                     ("shipper",              _ALL),
    "CONSIGNOR":                    ("shipper",              _BL),
    "CONSIGNEE":                    ("consignee",            _BPL),
    "CONSIGNEE ADDRESS":            ("consignee",            _BL),
    "NOTIFY PARTY":                 ("notify_party",         _BL),
    "NOTIFY":                       ("notify_party",         _BL),
    "ALSO NOTIFY":                  ("notify_party",         _BL),
    "NOTIFY ADDRESS":               ("notify_party",         _BL),
    # ── BL : transport ───────────────────────────────────────────────────
    "VESSEL":                       ("vessel",               _BL),
    "VESSEL NAME":                  ("vessel",               _BL),
    "M.V.":                         ("vessel",               _BL),
    "MV":                           ("vessel",               _BL),
    "VOYAGE":                       ("voyage",               _BL),
    "VOYAGE NO":                    ("voyage",               _BL),
    "VOYAGE NO.":                   ("voyage",               _BL),
    "VOYAGE NUMBER":                ("voyage",               _BL),
    "VOY. NO":                      ("voyage",               _BL),
    "VOY NO":                       ("voyage",               _BL),
    "VOY.":                         ("voyage",               _BL),
    "PORT OF LOADING":              ("port_of_loading",      _BL),
    "LOADING PORT":                 ("port_of_loading",      _BL),
    "PORT OF SHIPMENT":             ("port_of_loading",      _BL),
    "PORT OF ORIGIN":               ("port_of_loading",      _BL),
    "LOAD PORT":                    ("port_of_loading",      _BL),
    "POL":                          ("port_of_loading",      _BL),
    "PORT OF DISCHARGE":            ("port_of_discharge",    _BL),
    "DISCHARGE PORT":               ("port_of_discharge",    _BL),
    "POD":                          ("port_of_discharge",    _BL),
    "PLACE OF RECEIPT":             ("place_of_receipt",     _BL),
    "PLACE OF ORIGIN":              ("place_of_receipt",     _BL),
    "PORT OF RECEIPT":              ("place_of_receipt",     _BL),
    "PRE-CARRIAGE FROM":            ("place_of_receipt",     _BL),
    "PLACE OF DELIVERY":            ("place_of_delivery",    _BL),
    "FINAL DESTINATION":            ("place_of_delivery",    _BL),
    "DELIVERY PLACE":               ("place_of_delivery",    _BL),
    "CARRIER":                      ("carrier",              _BL),
    "OCEAN CARRIER":                ("carrier",              _BL),
    "SHIPPING LINE":                ("carrier",              _BL),
    "OPERATED BY":                  ("carrier",              _BL),
    # ── BL : références ──────────────────────────────────────────────────
    "B/L NO":                       ("bl_number",            _BL),
    "B/L NO.":                      ("bl_number",            _BL),
    "B/L NUMBER":                   ("bl_number",            _BL),
    "BL NO":                        ("bl_number",            _BL),
    "BL NO.":                       ("bl_number",            _BL),
    "BL NUMBER":                    ("bl_number",            _BL),
    "BILL OF LADING NO":            ("bl_number",            _BL),
    "BILL OF LADING NUMBER":        ("bl_number",            _BL),
    "DATE OF ISSUE":                ("issue_date",           _BL),
    "ISSUE DATE":                   ("issue_date",           _BIL),
    "B/L DATE":                     ("issue_date",           _BL),
    "DATE ISSUED":                  ("issue_date",           _BL),
    "FREIGHT TERMS":                ("freight_terms",        _BL),
    "FREIGHT PAYABLE":              ("freight_terms",        _BL),
    "FREIGHT & CHARGES":            ("freight_terms",        _BL),
    "FREIGHT PAYMENT":              ("freight_terms",        _BL),
    "FREIGHT":                      ("freight_terms",        _BL),
    "ETD":                          ("etd",                  _BL),
    "ESTIMATED DEPARTURE":          ("etd",                  _BL),
    "ESTIMATED TIME OF DEPARTURE":  ("etd",                  _BL),
    "ETA":                          ("eta",                  _BL),
    "ESTIMATED ARRIVAL":            ("eta",                  _BL),
    "ESTIMATED TIME OF ARRIVAL":    ("eta",                  _BL),
    "ON BOARD DATE":                ("on_board_date",        _BL),
    "SHIPPED ON BOARD":             ("on_board_date",        _BL),
    "DATE OF SHIPMENT":             ("on_board_date",        _BL),
    "LADEN ON BOARD":               ("on_board_date",        _BL),
    "NUMBER OF ORIGINALS":          ("number_of_originals",  _BL),
    "NO. OF ORIGINAL B/L":          ("number_of_originals",  _BL),
    "NO OF ORIGINAL B/L":           ("number_of_originals",  _BL),
    "ORIGINALS":                    ("number_of_originals",  _BL),
    "INCOTERM":                     ("incoterm",             _BIL),
    "INCOTERMS":                    ("incoterm",             _BIL),
    # ── Invoice : parties ────────────────────────────────────────────────
    "SELLER":                       ("seller",               _INV),
    "SELLER / EXPORTER":            ("seller",               _INV),
    "SELLER/EXPORTER":              ("seller",               _INV),
    "SUPPLIER":                     ("seller",               _INV),
    "VENDOR":                       ("seller",               _INV),
    "SOLD BY":                      ("seller",               _INV),
    "BUYER":                        ("buyer",                _INV),
    "BUYER / IMPORTER":             ("buyer",                _INV),
    "BUYER/IMPORTER":               ("buyer",                _INV),
    "IMPORTER":                     ("buyer",                _INV),
    "BILL TO":                      ("buyer",                _INV),
    "SOLD TO":                      ("buyer",                _INV),
    "CUSTOMER":                     ("buyer",                _INV),
    # ── Invoice : références ─────────────────────────────────────────────
    "INVOICE NO":                   ("invoice_number",       _INV),
    "INVOICE NO.":                  ("invoice_number",       _INV),
    "INVOICE NUMBER":               ("invoice_number",       _INV),
    "INV NO":                       ("invoice_number",       _INV),
    "INV. NO":                      ("invoice_number",       _INV),
    "COMMERCIAL INVOICE NO":        ("invoice_number",       _INV),
    "INVOICE DATE":                 ("invoice_date",         _INV),
    "DATE OF INVOICE":              ("invoice_date",         _INV),
    "PAYMENT TERMS":                ("payment_terms",        _INV),
    "TERMS OF PAYMENT":             ("payment_terms",        _INV),
    "MODE OF PAYMENT":              ("payment_terms",        _INV),
    "B/L REF":                      ("bl_reference",         _INV),
    "B/L REFERENCE":                ("bl_reference",         _INV),
    "BILL OF LADING REF":           ("bl_reference",         _INV),
    "TOTAL AMOUNT":                 ("total_amount",         _INV),
    "INVOICE AMOUNT":               ("total_amount",         _INV),
    "TOTAL INVOICE AMOUNT":         ("total_amount",         _INV),
    "AMOUNT DUE":                   ("total_amount",         _INV),
    "GRAND TOTAL":                  ("total_amount",         _INV),
    "CURRENCY":                     ("currency",             _BIL),
    "HS CODE":                      ("hs_code",              _INV),
    "HS NO":                        ("hs_code",              _INV),
    # ── Packing List ─────────────────────────────────────────────────────
    "PACKING LIST NO":              ("pl_number",            _PL),
    "PACKING LIST NO.":             ("pl_number",            _PL),
    "PACKING LIST NUMBER":          ("pl_number",            _PL),
    "PL NO":                        ("pl_number",            _PL),
    "PL NO.":                       ("pl_number",            _PL),
    "PL NUMBER":                    ("pl_number",            _PL),
    "TOTAL PACKAGES":               ("total_packages",       _PL),
    "TOTAL PKGS":                   ("total_packages",       _PL),
    "TOTAL CBM":                    ("total_cbm",            _PL),
    "TOTAL GROSS WEIGHT":           ("gross_weight_kg",      _BPL),
    "TOTAL NET WEIGHT":             ("net_weight_kg",        _BPL),
}

# Labels qui stoppent la lecture multi-lignes d'une partie
_PARTY_STOP_LABELS: frozenset[str] = frozenset(
    _LABEL_FIELD_MAP.keys()
) | frozenset({
    "VESSEL", "VESSEL NAME", "VOYAGE", "VOYAGE NO", "VOYAGE NO.",
    "PORT OF LOADING", "PORT OF DISCHARGE", "PLACE OF RECEIPT", "PLACE OF DELIVERY",
    "CONTAINER NO", "CONTAINER NO.", "CONTAINER TYPE", "SEAL NO", "SEAL NO.",
    "DESCRIPTION", "DESCRIPTION OF GOODS", "HS CODE", "QTY", "QUANTITY",
    "PKGS", "UNIT", "UNIT PRICE", "AMOUNT", "NET WT", "GROSS WT", "CBM",
    "MARKS", "MARKS AND NUMBERS", "PAYMENT TERMS",
    "TOTAL PACKAGES", "TOTAL NET WEIGHT", "TOTAL GROSS WEIGHT", "TOTAL CBM",
    "TOTAL AMOUNT", "AUTHORIZED SIGNATORY", "AUTHORIZED SIGNATORY & STAMP",
    "COMMERCIAL INVOICE", "PACKING LIST", "BILL OF LADING",
    # Libellés composés fréquents dans les en-têtes de colonne
    "VESSEL / VOYAGE", "VESSEL/VOYAGE", "VESSEL NAME / VOYAGE NO",
    "BOOKING REF", "BOOKING REF.", "BOOKING REFERENCE",
    "BOOKING NO", "BOOKING NO.", "BOOKING NUMBER",
    "CNTR NO / SEAL NO", "CNTR NO. / SEAL NO.", "CONTAINER NO / SEAL NO",
    "PORT OF LOADING / PORT OF DISCHARGE", "POL / POD", "POL/POD",
})

# Champs désignant des parties (valeurs multi-lignes)
_PARTY_FIELDS: frozenset[str] = frozenset({
    "shipper", "consignee", "notify_party", "seller", "buyer",
})

# Champs scalaires (valeur courte, ligne unique)
_SCALAR_FIELDS: frozenset[str] = frozenset({
    "vessel", "voyage", "port_of_loading", "port_of_discharge",
    "place_of_receipt", "place_of_delivery", "bl_number", "issue_date",
    "invoice_number", "invoice_date", "pl_number", "freight_terms",
    "number_of_originals", "carrier", "incoterm", "payment_terms",
    "currency", "hs_code", "total_amount", "total_packages", "total_cbm",
    "gross_weight_kg", "net_weight_kg", "bl_reference",
    "etd", "eta", "on_board_date",
})

# Valeurs qui ressemblent trop à des labels (à rejeter)
_LABEL_LIKE_RE = re.compile(
    r"^\s*(?:DATE|NUMBER|NO\.?|TYPE|DETAILS|INFORMATION|VALUE|REFERENCE|REF\.?|CODE|"
    r"NAME|DESCRIPTION|QUANTITY|UNIT|AMOUNT|CURRENCY|GOODS|TOTAL|WEIGHT|"
    r"PORT\s+OF\s+[A-Z]+|PLACE\s+OF\s+[A-Z]+|"
    r"VESSEL(?:\s*/\s*VOYAGE(?:\s+NO\.?)?)?|VOYAGE(?:\s+NO\.?)?|"
    r"SHIPPER|CONSIGNEE|NOTIFY(?:\s+PARTY)?|SELLER|BUYER|EXPORTER|IMPORTER|"
    r"BOOKING\s+(?:REF|REFERENCE|NO|NUMBER)\.?"
    r")\s*$",
    re.IGNORECASE,
)

# Libellés composés (LABEL / LABEL) qui ne peuvent jamais être des valeurs.
# Déjà normalisés (majuscules, espaces uniques) — pas besoin d'appeler _norm.
_COMPOUND_LABEL_NORMS: frozenset[str] = frozenset({
    "VESSEL / VOYAGE",        "VESSEL/VOYAGE",
    "VESSEL NAME / VOYAGE NO","VESSEL NAME/VOYAGE NO",
    "VESSEL / VOYAGE NO",     "VESSEL / VOY",
    "BOOKING REF",            "BOOKING REF.",  "BOOKING REFERENCE",
    "BOOKING NO",             "BOOKING NO.",   "BOOKING NUMBER",
    "CNTR NO / SEAL NO",      "CNTR NO. / SEAL NO.",
    "CONTAINER NO / SEAL NO", "CONTAINER NO. / SEAL NO.",
    "PORT OF LOADING / PORT OF DISCHARGE",
    "POL / POD", "POL/POD",
    "PLACE OF RECEIPT / PLACE OF DELIVERY",
    "GROSS / NET WEIGHT",     "TOTAL GROSS / NET WEIGHT",
})

# Validation par champ : uniquement les champs avec un domaine fini
_INCOTERM_CODES: frozenset[str] = frozenset({
    "FOB", "CIF", "CFR", "EXW", "DDP", "FCA", "CPT", "CIP", "DAP", "DAT", "DPU",
})
_FREIGHT_TERMS_KEYWORDS: frozenset[str] = frozenset({
    "PREPAID", "COLLECT",
})

# Préfixes d'armateur/navire — rejetés comme valeur de port/lieu
_CARRIER_NAME_PREFIXES: frozenset[str] = frozenset({
    "MSC", "MAERSK", "CMA", "HAPAG", "EVERGREEN", "ONE",
    "COSCO", "YANG", "ZIM", "HAMBURG", "OOCL", "APL", "PIL",
    "MOL", "NYK", "KLINE", "WHL", "HMM", "SITC", "CSCL",
})


def _validate_field_value(fname: str, val: str) -> bool:
    """
    Retourne False si la valeur n'est pas cohérente avec le domaine du champ.
    Utilisé pour rejeter les extractions spatiales erronées avant de les accepter.
    """
    upper = val.upper().strip()
    words = set(re.split(r"[\s/\-,\.]+", upper))

    # Rejet global : libellés composés (LABEL / LABEL)
    if _norm(val) in _COMPOUND_LABEL_NORMS:
        return False

    if fname == "incoterm":
        return bool(words & _INCOTERM_CODES)
    if fname == "freight_terms":
        return bool(words & _FREIGHT_TERMS_KEYWORDS)
    if fname in ("port_of_discharge", "port_of_loading", "place_of_receipt", "place_of_delivery"):
        first_word = upper.split()[0] if upper.split() else ""
        if first_word in _CARRIER_NAME_PREFIXES:
            return False
    if fname == "bl_number":
        # Un numéro BL doit contenir au moins un chiffre ; les libellés purs sont rejetés
        if not re.search(r'\d', val):
            return False
    if fname == "carrier":
        # Rejeter les valeurs qui sont de purs libellés (ex: "VESSEL / VOYAGE")
        if _LABEL_LIKE_RE.match(val):
            return False
    return True

# Lignes à sauter dans les blocs parties (références, en-têtes)
_PARTY_LINE_SKIP_RE = re.compile(
    r"^\s*(?:B/?L|INVOICE|PACKING\s+LIST|PL|LC|T/?T|REF|NO\.?|"
    r"TO\s+THE\s+ORDER|TO\s+ORDER)\s*[:\-\.#]?\s*[A-Z0-9\-\/]{0,20}\s*$",
    re.IGNORECASE,
)


# ── Normalisation ─────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper().strip())


def _clean_val(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _strip_parenthetical(norm_text: str) -> str:
    """Enlève les parenthétiques : "CONSIGNEE (TO ORDER OF)" → "CONSIGNEE"."""
    return re.sub(r"\s*\([^)]{0,60}\)", "", norm_text).strip()


def _lookup_label(text: str) -> tuple[str, frozenset[str]] | None:
    """Cherche un label dans _LABEL_FIELD_MAP (avec et sans parenthétiques)."""
    norm = _norm(text)
    result = _LABEL_FIELD_MAP.get(norm)
    if result:
        return result
    clean = _strip_parenthetical(norm)
    return _LABEL_FIELD_MAP.get(clean)


# ── Classe principale ─────────────────────────────────────────────────────────

class SpatialExtractor:
    """
    Extraction Tier-0 par analyse spatiale des blocs PDF.
    Sans ML — exploite la géométrie des blocs PyMuPDF.
    """

    _X_COL_TOL   = 80    # tolérance X (pts) pour aligner label ↔ valeur
    _GRID_Y_TOL  = 14    # tolérance Y (pts) pour une même rangée de grille
    _BELOW_Y_MAX = 130   # distance max (pts) pour le bloc valeur en-dessous
    _ANCHOR_Y_MIN = -8   # léger chevauchement toléré

    def extract(
        self,
        pdf_path: Path | str,
        doc_type: str,
    ) -> dict:
        """
        Retourne {"fields": {}, "tiers": {}, "confidences": {}}.
        """
        blocks = self._load_blocks(Path(pdf_path))
        if not blocks:
            return {"fields": {}, "tiers": {}, "confidences": {}}

        fields:      dict[str, str]   = {}
        tiers:       dict[str, str]   = {}
        confidences: dict[str, float] = {}

        def _merge(fname: str, val: str, tier: str, conf: float) -> None:
            val = _clean_val(val)
            if not val or _LABEL_LIKE_RE.match(val):
                return
            if _norm(val) in _COMPOUND_LABEL_NORMS:
                return
            if not _validate_field_value(fname, val):
                return
            if fname in confidences and conf <= confidences[fname]:
                return
            fields[fname]      = val
            tiers[fname]       = tier
            confidences[fname] = conf

        # Stratégie A — inline KV (LABEL: value ou LABEL\nVALUE dans le bloc)
        for t in self._extract_inline(blocks, doc_type):
            _merge(*t)

        # Stratégie D — parties multi-lignes (label sur ligne 0, valeur lignes 1..n)
        for t in self._extract_party_multiline(blocks, doc_type):
            _merge(*t)

        # Stratégie B — anchor cross-block (label seul → valeur bloc en-dessous)
        for t in self._extract_anchor(blocks, doc_type):
            _merge(*t)

        # Stratégie C — grilles header/data
        for t in self._detect_grids(blocks, doc_type):
            _merge(*t)

        return {"fields": fields, "tiers": tiers, "confidences": confidences}

    # ── Chargement des blocs ──────────────────────────────────────────────────

    def _load_blocks(self, pdf_path: Path) -> list[SpatialBlock]:
        try:
            doc = fitz.open(str(pdf_path))
            result: list[SpatialBlock] = []
            for page_no, page in enumerate(doc):
                pw = page.rect.width
                ph = page.rect.height
                for b in page.get_text("blocks"):
                    if b[6] != 0:
                        continue
                    raw = b[4]
                    if not raw.strip():
                        continue
                    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                    result.append(SpatialBlock(
                        text=raw,
                        x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                        page_width=pw, page_height=ph,
                        page=page_no,
                        lines=lines,
                    ))
            doc.close()
            return result
        except Exception as exc:
            log.debug("[spatial] chargement blocs échoué: %s", exc)
            return []

    # ── Stratégie A : inline KV ───────────────────────────────────────────────

    def _extract_inline(
        self,
        blocks: list[SpatialBlock],
        doc_type: str,
    ) -> list[tuple[str, str, str, float]]:
        """
        Analyse chaque bloc ligne par ligne pour détecter :
        - "LABEL: value"  → même ligne
        - "LABEL\\nVALUE" → valeur sur la ligne suivante dans le même bloc
        """
        results: list[tuple[str, str, str, float]] = []

        for blk in blocks:
            n = len(blk.lines)
            for idx, line in enumerate(blk.lines):

                # ── Pattern LABEL: value (même ligne) ────────────────────
                m = re.match(
                    r'^([A-Z][A-Z0-9\s/\-\.&,\(\)]{1,55}?)\s*:\s*(.+)$',
                    line,
                )
                if m:
                    mapping = _lookup_label(m.group(1))
                    if mapping:
                        fname, valid_types = mapping
                        if doc_type in valid_types:
                            val = _clean_val(m.group(2))
                            if val and not _LABEL_LIKE_RE.match(val):
                                results.append(
                                    (fname, val, "spatial_inline", _CONF["spatial_inline"])
                                )
                                continue

                # ── Pattern LABEL (ligne seule) → valeur sur ligne suivante ──
                mapping = _lookup_label(line)
                if mapping is None:
                    continue
                fname, valid_types = mapping
                if doc_type not in valid_types:
                    continue
                if idx + 1 >= n:
                    continue

                next_line = blk.lines[idx + 1]
                # Ne pas capturer une autre étiquette comme valeur
                if _lookup_label(next_line) is not None:
                    continue
                if _norm(next_line) in _PARTY_STOP_LABELS:
                    continue

                if fname in _PARTY_FIELDS:
                    val = self._collect_lines(blk.lines, idx + 1)
                    tier = "spatial_party"
                    conf = _CONF["spatial_party"]
                else:
                    val = _clean_val(next_line)
                    tier = "spatial_inline"
                    conf = _CONF["spatial_inline"]

                if val and not _LABEL_LIKE_RE.match(val):
                    results.append((fname, val, tier, conf))

        return results

    # ── Stratégie D : parties multi-lignes ───────────────────────────────────

    def _extract_party_multiline(
        self,
        blocks: list[SpatialBlock],
        doc_type: str,
    ) -> list[tuple[str, str, str, float]]:
        """
        Blocs dont la 1ʳᵉ ligne est un label de partie et les suivantes = valeur.
        Un même bloc peut contenir plusieurs sections (SHIPPER, puis CONSIGNEE…).
        """
        results: list[tuple[str, str, str, float]] = []

        for blk in blocks:
            if len(blk.lines) < 2:
                continue

            i = 0
            while i < len(blk.lines):
                mapping = _lookup_label(blk.lines[i])
                if mapping is None:
                    i += 1
                    continue
                fname, valid_types = mapping
                if doc_type not in valid_types or fname not in _PARTY_FIELDS:
                    i += 1
                    continue

                val = self._collect_lines(blk.lines, i + 1)
                if val:
                    results.append(
                        (fname, val, "spatial_party", _CONF["spatial_party"])
                    )

                # Avancer jusqu'au prochain label (déjà consommé par _collect_lines)
                i += 1
                while i < len(blk.lines) and _lookup_label(blk.lines[i]) is None:
                    i += 1

        return results

    # ── Stratégie B : anchor cross-block ─────────────────────────────────────

    def _extract_anchor(
        self,
        blocks: list[SpatialBlock],
        doc_type: str,
    ) -> list[tuple[str, str, str, float]]:
        """
        Pour chaque bloc réduit à un label connu (une seule ligne) :
        cherche le bloc valeur le plus proche dans la même bande X, en-dessous.
        """
        results: list[tuple[str, str, str, float]] = []

        for lbl_blk in blocks:
            if not lbl_blk.lines:
                continue

            # Accepte un seul label, éventuellement parenthétique
            norm = _norm(lbl_blk.text)
            norm_clean = _strip_parenthetical(norm)
            mapping = _LABEL_FIELD_MAP.get(norm_clean) or _LABEL_FIELD_MAP.get(norm)
            if mapping is None:
                continue
            fname, valid_types = mapping
            if doc_type not in valid_types:
                continue

            val_blk = self._find_nearest_below(lbl_blk, blocks, fname)
            if val_blk is None:
                continue

            if fname in _PARTY_FIELDS:
                val  = self._collect_lines(val_blk.lines, 0)
                tier = "spatial_party"
                conf = _CONF["spatial_party"]
            else:
                val  = _clean_val(val_blk.lines[0]) if val_blk.lines else ""
                tier = "spatial_anchor"
                conf = _CONF["spatial_anchor"]

            if val and not _LABEL_LIKE_RE.match(val):
                results.append((fname, val, tier, conf))

        return results

    def _find_nearest_below(
        self,
        label_blk: SpatialBlock,
        all_blocks: list[SpatialBlock],
        fname: str,
    ) -> SpatialBlock | None:
        best: SpatialBlock | None = None
        best_dy = float("inf")

        for blk in all_blocks:
            if blk is label_blk:
                continue
            if blk.page != label_blk.page:
                continue
            dy = blk.y0 - label_blk.y1
            if dy < self._ANCHOR_Y_MIN or dy > self._BELOW_Y_MAX:
                continue
            if abs(blk.cx - label_blk.cx) > self._X_COL_TOL:
                continue
            # Rejeter les blocs qui sont eux-mêmes des labels
            first = blk.lines[0] if blk.lines else ""
            if _lookup_label(first) is not None:
                continue
            if _norm(first) in _PARTY_STOP_LABELS:
                continue
            if dy < best_dy:
                best_dy = dy
                best = blk

        return best

    # ── Stratégie C : grilles header/data ────────────────────────────────────

    def _detect_grids(
        self,
        blocks: list[SpatialBlock],
        doc_type: str,
    ) -> list[tuple[str, str, str, float]]:
        """
        Détecte les rangées d'en-têtes (ex : VESSEL | VOYAGE | POL | POD)
        et récupère les valeurs dans la rangée immédiatement en-dessous.
        """
        results: list[tuple[str, str, str, float]] = []

        # Grouper par page
        pages: dict[int, list[SpatialBlock]] = {}
        for blk in blocks:
            pages.setdefault(blk.page, []).append(blk)

        for page_no, page_blocks in pages.items():
            # Identifier les blocs qui contiennent uniquement un label scalaire
            scalar_label_blks: list[SpatialBlock] = []
            for blk in page_blocks:
                norm = _norm(blk.text)
                norm_clean = _strip_parenthetical(norm)
                mapping = (
                    _LABEL_FIELD_MAP.get(norm_clean)
                    or _LABEL_FIELD_MAP.get(norm)
                )
                if (
                    mapping
                    and doc_type in mapping[1]
                    and mapping[0] in _SCALAR_FIELDS
                ):
                    scalar_label_blks.append(blk)

            # Regrouper par rangée Y
            clusters = self._cluster_by_y(scalar_label_blks)

            for cluster in clusters:
                if len(cluster) < 2:
                    continue  # grille = ≥ 2 labels sur la même ligne

                row_y_bottom = max(b.y1 for b in cluster)

                for lbl_blk in cluster:
                    norm = _norm(lbl_blk.text)
                    norm_clean = _strip_parenthetical(norm)
                    mapping = (
                        _LABEL_FIELD_MAP.get(norm_clean)
                        or _LABEL_FIELD_MAP.get(norm)
                    )
                    if not mapping:
                        continue
                    fname = mapping[0]

                    # Cherche le bloc valeur dans la même bande X, sous la rangée
                    best_val: SpatialBlock | None = None
                    best_dy = float("inf")
                    for blk in page_blocks:
                        if blk in cluster:
                            continue
                        dy = blk.y0 - row_y_bottom
                        if dy < self._ANCHOR_Y_MIN or dy > 70:
                            continue
                        if abs(blk.cx - lbl_blk.cx) > self._X_COL_TOL:
                            continue
                        first = blk.lines[0] if blk.lines else ""
                        if _lookup_label(first) is not None:
                            continue
                        if _norm(first) in _PARTY_STOP_LABELS:
                            continue
                        if dy < best_dy:
                            best_dy = dy
                            best_val = blk

                    if best_val:
                        val = _clean_val(
                            best_val.lines[0] if best_val.lines else ""
                        )
                        if val and not _LABEL_LIKE_RE.match(val):
                            results.append(
                                (fname, val, "spatial_grid", _CONF["spatial_grid"])
                            )

        return results

    def _cluster_by_y(
        self, blocks: list[SpatialBlock]
    ) -> list[list[SpatialBlock]]:
        """Regroupe les blocs par rangée Y (tolérance ±_GRID_Y_TOL)."""
        if not blocks:
            return []
        sorted_blks = sorted(blocks, key=lambda b: (b.page, b.cy))
        clusters: list[list[SpatialBlock]] = []
        current: list[SpatialBlock] = [sorted_blks[0]]

        for blk in sorted_blks[1:]:
            if (
                blk.page == current[-1].page
                and abs(blk.cy - current[-1].cy) <= self._GRID_Y_TOL
            ):
                current.append(blk)
            else:
                clusters.append(current)
                current = [blk]
        clusters.append(current)
        return clusters

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collect_lines(self, lines: list[str], start: int) -> str:
        """
        Collecte les lignes à partir de `start` jusqu'au premier stop label.
        Retourne les lignes jointes par " | ".
        """
        collected: list[str] = []
        for line in lines[start:]:
            norm = _norm(line)
            if norm in _PARTY_STOP_LABELS:
                break
            if _lookup_label(line) is not None:
                break
            if _PARTY_LINE_SKIP_RE.match(line):
                continue
            stripped = line.strip()
            if len(stripped) < 2:
                continue
            collected.append(stripped)
        return " | ".join(collected) if collected else ""
