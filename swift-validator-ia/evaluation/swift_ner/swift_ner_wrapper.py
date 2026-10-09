"""
swift_ner_wrapper.py — Correctifs SANS re-entrainement du modele swift-ner
==========================================================================

Trois corrections appliquees AUTOUR du modele (poids inchanges) :

1. AMOUNT — normalisation d'entree : on supprime les separateurs de milliers
   (virgules) des nombres avant l'inference, puis on remappe les offsets vers
   le texte ORIGINAL. Le modele gere deja "USD 980000" ; il ne cassait que sur
   "USD 1,234,000". Suppression = caracteres supprimes uniquement -> remap trivial.

2. GOODS — post-traitement : on fusionne les fragments GOODS separes uniquement
   par un INCOTERM (ou un petit interstice) en un seul span continu.

3. PORT — regex deterministe sur les champs :44A:/:44B: (format SWIFT fixe).

API :
    w = SwiftNERWrapper(model_dir)
    entities = w.predict(text)   # -> [{"label","start","end","value"}, ...]
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Nombre avec separateurs de milliers : 1,234  /  1,234,000
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_PORT_RE = re.compile(r":44[AB]:\s*PORT OF (?:LOADING|DISCHARGE):\s*([A-Z][A-Z .'-]+)")
# Montant SWIFT : tout le contenu de la ligne du tag :32B:  (ex "CHF 2.477.000,00")
_AMOUNT_RE = re.compile(r":32B:[ \t]*(\S[^\r\n]*?)[ \t]*(?:\r?\n|$)")

# Incoterms (ICC 2020) — servent a couper la queue d'un span GOODS
_INCOTERMS = (
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF",
    "CPT", "CIP", "DAP", "DPU", "DDP", "DDU",
)
# Tag SWIFT en tete de span : "45A:" / ":45A:" / ":45A" + espaces
_GOODS_TAG = re.compile(r"^[:\s]*\d{1,2}[A-Z]:?\s+")
# INCOTERM en queue de span : " ... CIF ISTANBUL"
_GOODS_TAIL = re.compile(r"\s+(?:%s)\b" % "|".join(_INCOTERMS))


def normalize_amounts(text: str):
    """Supprime les virgules de milliers. Retourne (texte_normalise, idx_map)
    ou idx_map[i] = index dans le texte ORIGINAL du i-eme char normalise."""
    out_chars = []
    idx_map = []
    for i, ch in enumerate(text):
        # supprimer une virgule uniquement si elle separe des milliers
        if ch == "," and _THOUSANDS.match(text, i):
            continue
        out_chars.append(ch)
        idx_map.append(i)
    idx_map.append(len(text))  # sentinelle pour les fins de span
    return "".join(out_chars), idx_map


class SwiftNERWrapper:
    def __init__(self, model_dir, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.id2label = self.model.config.id2label

    # ── inference brute (BIO -> spans) sur un texte donne ────────────────
    def _raw_spans(self, text):
        enc = self.tok(text, truncation=True, max_length=512,
                       return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            preds = self.model(**enc).logits.argmax(-1)[0].tolist()

        spans, cur = [], None
        for (s, e), p in zip(offsets, preds):
            if s == e:
                continue
            lab = self.id2label[p]
            if lab == "O":
                if cur:
                    spans.append(cur); cur = None
                continue
            tag, ent = lab.split("-", 1)
            if tag == "B" or cur is None or cur["label"] != ent:
                if cur:
                    spans.append(cur)
                cur = {"label": ent, "start": s, "end": e}
            else:  # I- continuation
                cur["end"] = e
        if cur:
            spans.append(cur)
        return spans

    # ── post-traitement GOODS : fusion des fragments autour d'un INCOTERM ─
    @staticmethod
    def _merge_goods(spans, text):
        spans = sorted(spans, key=lambda s: s["start"])
        out = []
        for sp in spans:
            if (out and sp["label"] == "GOODS" and out[-1]["label"] == "GOODS"
                    and sp["start"] - out[-1]["end"] <= 12):
                # fusionner si l'interstice ne contient pas de retour ligne
                gap = text[out[-1]["end"]:sp["start"]]
                if "\n" not in gap:
                    out[-1]["end"] = sp["end"]
                    continue
            # cas GOODS ... INCOTERM ... GOODS : on absorbe l'incoterm mais on
            # garde l'entite INCOTERM separee (elle reste valable)
            out.append(dict(sp))
        # 2e passe : recoller GOODS-INCOTERM-GOODS
        merged = []
        i = 0
        while i < len(out):
            if (i + 2 < len(out) and out[i]["label"] == "GOODS"
                    and out[i + 1]["label"] == "INCOTERM"
                    and out[i + 2]["label"] == "GOODS"
                    and "\n" not in text[out[i]["end"]:out[i + 2]["start"]]):
                big = {"label": "GOODS", "start": out[i]["start"], "end": out[i + 2]["end"]}
                merged.append(big)
                merged.append(out[i + 1])  # garder l'INCOTERM
                i += 3
            else:
                merged.append(out[i]); i += 1
        return merged

    # ── trim GOODS : retire le tag SWIFT de tete et l'INCOTERM de queue ──
    @staticmethod
    def _clean_goods(spans, text):
        out = []
        for sp in spans:
            if sp["label"] != "GOODS":
                out.append(sp)
                continue
            s, e = sp["start"], sp["end"]
            frag = text[s:e]
            # 1) couper le tag de tete (":45A: ")
            m = _GOODS_TAG.match(frag)
            if m:
                s += m.end()
                frag = text[s:e]
            # 2) couper l'INCOTERM de queue (et tout ce qui suit)
            m = _GOODS_TAIL.search(frag)
            if m:
                e = s + m.start()
                frag = text[s:e]
            # 3) rogner les espaces residuels
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                sp = dict(sp); sp["start"] = s; sp["end"] = e
                out.append(sp)
        return out

    # ── regex AMOUNT deterministe (tag :32B:) ────────────────────────────
    @staticmethod
    def _regex_amounts(text):
        out = []
        for m in _AMOUNT_RE.finditer(text):
            out.append({"label": "AMOUNT", "start": m.start(1), "end": m.end(1),
                        "value": m.group(1)})
        return out

    # ── regex PORT deterministe ──────────────────────────────────────────
    @staticmethod
    def _regex_ports(text):
        ports = []
        for m in _PORT_RE.finditer(text):
            val = m.group(1).rstrip()
            ports.append({"label": "PORT", "start": m.start(1),
                          "end": m.start(1) + len(val), "value": val})
        return ports

    # ── pipeline complet ─────────────────────────────────────────────────
    def predict(self, text, fix_amount=True, fix_goods=True, add_ports=False,
                rule_amount=False):
        if fix_amount and not rule_amount:
            norm, idx_map = normalize_amounts(text)
            raw = self._raw_spans(norm)
            # remap vers offsets du texte original
            for sp in raw:
                sp["start"] = idx_map[sp["start"]]
                sp["end"] = idx_map[sp["end"]]
        else:
            raw = self._raw_spans(text)

        if fix_goods:
            raw = self._merge_goods(raw, text)
            raw = self._clean_goods(raw, text)
        if rule_amount:
            # le tag :32B: fait autorite -> on remplace l'AMOUNT du modele
            raw = [s for s in raw if s["label"] != "AMOUNT"] + self._regex_amounts(text)
        if add_ports:
            raw = [s for s in raw if s["label"] != "PORT"] + self._regex_ports(text)

        for sp in raw:
            sp["value"] = text[sp["start"]:sp["end"]]
        return raw
