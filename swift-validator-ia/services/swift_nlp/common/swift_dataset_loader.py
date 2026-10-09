"""
swift_dataset_loader.py  —  v1.0
===================================
Charge les datasets JSONL générés et expose des structures
prêtes à l'emploi pour les 3 services NLP.

Datasets consommés :
  spell_correction.jsonl       -> paires wrong→correct
  clause_classification.jsonl  -> 1500 clauses avec type + importance (300/type × 5 types)
  ner_dataset.jsonl            -> messages, entités avec spans
  training_samples.jsonl       -> samples, risk flags, completeness

Usage :
  from swift_dataset_loader import SwiftDatasetLoader
  ds = SwiftDatasetLoader().get()    # ou SwiftDatasetLoader.default().get()

  ds.spell_dict           # dict[str, str]       paires wrong→correct
  ds.clause_examples      # list[dict]           exemples de clauses
  ds.clause_features      # dict[type, list]     features TF-IDF par type
  ds.known_banks          # set[str]             18 banques émettrices connues
  ds.swift_vocab          # set[str]             300 mots SWIFT fréquents
  ds.risk_patterns        # list[str]            patterns empiriques
  ds.completeness_refs    # dict[type, float]    score moyen par MT type
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_SWIFT_NLP_ROOT = Path(__file__).resolve().parents[1]
_FALLBACK_DIRS = [
    Path("data/data_stati"),
    _SWIFT_NLP_ROOT / "datasets",
    _SWIFT_NLP_ROOT.parent.parent / "data" / "data_stati",
]

_SENTINEL_FILE = "spell_correction.jsonl"


@dataclass
class SwiftDatasets:
    spell_dict:          dict = field(default_factory=dict)
    spell_contexts:      dict = field(default_factory=dict)
    clause_examples:     list = field(default_factory=list)
    clause_features:     dict = field(default_factory=dict)
    clause_type_dist:    dict = field(default_factory=dict)
    known_banks:         set  = field(default_factory=set)
    known_applicants:    set  = field(default_factory=set)
    known_beneficiaries: set  = field(default_factory=set)
    swift_vocab:         set  = field(default_factory=set)
    ner_label_vocab:     dict = field(default_factory=dict)
    risk_patterns:       list = field(default_factory=list)
    completeness_refs:   dict = field(default_factory=dict)
    validation_patterns: list = field(default_factory=list)
    loaded:              bool = False
    dataset_dir:         str  = ""
    stats:               dict = field(default_factory=dict)


class SwiftDatasetLoader:
    _default_instance = None
    _cache: SwiftDatasets | None = None

    def __init__(self, dataset_dir: str | Path | None = None):
        self._dir = self._resolve_dir(dataset_dir)

    @classmethod
    def default(cls) -> "SwiftDatasetLoader":
        """Singleton global sans argument."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    # Alias pour compatibilité
    @classmethod
    def instance(cls, dataset_dir=None):
        return cls.default()

    def get(self) -> SwiftDatasets:
        if self._cache is None:
            self._cache = self._load_all()
        return self._cache

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_dir(dataset_dir) -> Path:
        if dataset_dir:
            p = Path(dataset_dir)
            if (p / _SENTINEL_FILE).exists():
                return p
        for candidate in _FALLBACK_DIRS:
            if (candidate / _SENTINEL_FILE).exists():
                logger.info("Dataset dir resolved: %s", candidate)
                return candidate
        logger.warning("No dataset dir found — NLP will use built-in defaults only")
        return Path()

    def _load_all(self) -> SwiftDatasets:
        spell_dict, spell_ctx           = self._load_spell()
        cl_ex, cl_feat, cl_dist         = self._load_clauses()
        banks, appl, bene, vocab, lv    = self._load_ner()
        risk, comp_refs, val_pats       = self._load_training()

        stats = {
            "spell_pairs":     len(spell_dict),
            "clause_examples": len(cl_ex),
            "known_banks":     len(banks),
            "vocab_size":      len(vocab),
            "risk_patterns":   len(risk),
        }
        logger.info("SwiftDatasets loaded: %s", stats)

        return SwiftDatasets(
            spell_dict=spell_dict, spell_contexts=spell_ctx,
            clause_examples=cl_ex, clause_features=cl_feat,
            clause_type_dist=cl_dist,
            known_banks=banks, known_applicants=appl,
            known_beneficiaries=bene, swift_vocab=vocab,
            ner_label_vocab=lv,
            risk_patterns=risk, completeness_refs=comp_refs,
            validation_patterns=val_pats,
            loaded=True, dataset_dir=str(self._dir), stats=stats,
        )

    # ─── Spell ───────────────────────────────────────────────────────
    def _load_spell(self):
        """DS-01/IC-02 : charge les paires wrong→correct.
        Garde : wrong != correct (déjà présent) + normalisation espaces/ponctuation.
        Les 120 paires bug du dataset (wrong==correct) sont automatiquement filtrées.
        """
        path = self._dir / "spell_correction.jsonl"
        sd, sc = {}, defaultdict(list)
        if not path.exists():
            return sd, {}
        for rec in self._read_jsonl(path):
            w   = re.sub(r'\s+', ' ', str(rec.get("wrong",""))).strip().upper()
            c   = re.sub(r'\s+', ' ', str(rec.get("correct",""))).strip().upper()
            ctx = str(rec.get("context","")).strip()
            # DS-01 : rejeter si wrong == correct après normalisation
            if not w or not c or w == c:
                continue
            # Rejeter les paires non-alphabétiques pures (artefacts de génération)
            if not re.search(r'[A-Z]', w):
                continue
            if w not in sd or not rec.get("augmented"):
                sd[w] = c
            if ctx and len(sc[w]) < 3:
                sc[w].append(ctx[:200])
        return sd, dict(sc)

    # ─── Clauses ─────────────────────────────────────────────────────
    def _load_clauses(self):
        path = self._dir / "clause_classification.jsonl"
        examples, type_tf, type_cnt = [], defaultdict(Counter), Counter()
        if not path.exists():
            return examples, {}, {}
        for rec in self._read_jsonl(path):
            text = str(rec.get("text","")).strip()
            t    = str(rec.get("type","")).strip()
            if not text or not t:
                continue
            examples.append({
                "text":       text,
                "type":       t,
                "importance": rec.get("importance","medium"),
                "swift_type": rec.get("swift_type",""),
                "augmented":  rec.get("augmented", False),
            })
            type_cnt[t] += 1
            for w in text.upper().split():
                wc = re.sub(r'[^A-Z]', '', w)
                if len(wc) > 3:
                    type_tf[t][wc] += 1
        return examples, self._tfidf(type_tf), dict(type_cnt)

    @staticmethod
    def _tfidf(type_tf, top_n=20):
        types = list(type_tf.keys())
        if not types:
            return {}
        all_words = set(w for tf in type_tf.values() for w in tf)
        word_df = Counter()
        for w in all_words:
            for t in types:
                if type_tf[t].get(w, 0) > 0:
                    word_df[w] += 1
        features = {}
        for t in types:
            scores = {w: tf * math.log(len(types) / word_df.get(w, 1))
                      for w, tf in type_tf[t].items()}
            features[t] = [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
        return features

    # ─── NER ─────────────────────────────────────────────────────────
    def _load_ner(self):
        path = self._dir / "ner_dataset.jsonl"
        banks, appl, bene, vocab, lv = set(), Counter(), Counter(), Counter(), defaultdict(Counter)
        if not path.exists():
            return banks, set(), set(), set(), {}
        for rec in self._read_jsonl(path):
            for w in rec.get("text","").upper().split():
                wc = re.sub(r'[^A-Z]', '', w)
                if len(wc) > 5:
                    vocab[wc] += 1
            for ent in rec.get("entities", []):
                lbl = ent.get("label","")
                val = str(ent.get("value","")).strip().upper()
                if not val or len(val) < 3:
                    continue
                # DS-04 : pour le NER label vocab, ne garder que les entités avec span valide
                # Les ISSUING_BANK sans span (in_text=False) polluent le vocabulaire
                has_span = ent.get("start") is not None and ent.get("in_text", True)
                if has_span:
                    lv[lbl][val] += 1
                else:
                    # Garder quand même dans known_banks (utile pour le correcteur)
                    lv[lbl][val]  # pas d'incrémentation du lv pour span-less
                if lbl == "ISSUING_BANK":   banks.add(val)
                elif lbl == "APPLICANT":    appl[val] += 1
                elif lbl == "BENEFICIARY":  bene[val] += 1
        return (
            banks,
            {v for v, c in appl.items() if c >= 2},
            {v for v, c in bene.items() if c >= 2},
            {w for w, _ in vocab.most_common(300)},
            dict(lv),
        )

    # ─── Training ────────────────────────────────────────────────────
    def _load_training(self):
        path = self._dir / "training_samples.jsonl"
        risk_ctr, val_ctr, type_sc = Counter(), Counter(), defaultdict(list)
        if not path.exists():
            return [], {}, []
        for rec in self._read_jsonl(path):
            st    = rec.get("swift_type","")
            score = rec.get("labels",{}).get("completeness_score", 0)
            if st and score:
                type_sc[st].append(score)
            for rf in rec.get("risk_flags", []):
                risk_ctr[str(rf)] += 1
            for e in rec.get("validation",{}).get("errors",[]):
                val_ctr[str(e)] += 1
        comp_refs = {t: round(sum(s)/len(s), 1) for t, s in type_sc.items() if s}
        # IC-01 : filtrer les pseudo-patterns "N spelling issue(s)" — pas des catégories ML
        _SKIP_RISK_PATTERNS = re.compile(
            r'\d+\s+spelling\s+issue|spelling\s+issue|validation error',
            re.IGNORECASE,
        )
        risk_patterns = [
            p for p, c in risk_ctr.most_common(20)
            if c >= 5 and not _SKIP_RISK_PATTERNS.search(p)
        ]
        return (
            risk_patterns,
            comp_refs,
            [p for p, c in val_ctr.most_common(10) if c >= 3],
        )

    # ─── Utilitaire ──────────────────────────────────────────────────
    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        records = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError as e:
            logger.error("Cannot read %s: %s", path, e)
        return records