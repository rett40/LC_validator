"""
swift_ds_pipeline.py — Pipeline Data Science pour analyse de risque LC
======================================================================

Transforme les sorties du pipeline NLP en features ML exploitables.
Implémente 3 niveaux d'intelligence :

  Niveau 1 : Feature Engineering + Risk Scoring (XGBoost/RandomForest)
  Niveau 2 : Anomaly Detection (IsolationForest)
  Niveau 3 : LC Similarity Search (embeddings cosine)

Usage :
    # Extraire les features d'un résultat de pipeline
    extractor = LCFeatureExtractor()
    features  = extractor.extract(pipeline_result, ucp_report)

    # Scorer le risque
    scorer = LCRiskScorer.load("models/lc-risk-ds")
    result = scorer.score(features)

    # Détecter les anomalies
    detector = LCAnomalyDetector.load("models/lc-anomaly")
    result   = detector.predict(features)

    # Chercher des LC similaires
    search = LCSimilaritySearch.load("models/lc-embeddings")
    similar = search.find_similar(features, top_k=5)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

# Risque Incoterm : plus la banque a de responsabilité, plus le risque est élevé
INCOTERM_RISK = {
    "EXW": 5, "FCA": 4, "CPT": 3, "CIP": 3,
    "DAP": 2, "DPU": 2, "DDP": 1,
    "FAS": 4, "FOB": 4, "CFR": 3, "CIF": 2,
}

# Risque pays — aligné sur risk_service / external_validation (ordinal 0–3)
# 3 = liste noire FATF (IR, KP, MM) ; 2 = haut risque ; 1 = modéré ; 0 = faible
FATF_BLACKLIST_ISO2 = {"IR", "KP", "MM"}
HIGH_RISK_COUNTRIES = {
    "SY", "CU", "SD", "BY", "RU",  # OFAC / sensibles (hors blacklist déjà traitée)
    "AF", "ML", "BF", "SS",
}
LOW_COUNTRY_ISO2 = {
    "FR", "DE", "US", "GB", "JP", "CH", "NL", "CA", "AU", "SG", "SE", "NO", "DK", "FI", "AT",
}

# Devises à risque de change élevé
HIGH_RISK_CURRENCIES = {"IRR", "SYP", "KPW", "VES", "ZWL"}


@dataclass
class LCFeatures:
    """
    Vecteur de features d'une Lettre de Crédit.
    20 features numériques/booléennes pour ML.
    """

    # ── Temporel ──────────────────────────────────────────────────────
    validity_days:        float = 0.0   # expiry - issue
    presentation_days:    float = 21.0  # présentation_period
    days_to_shipment:     float = 0.0   # shipment - issue

    # ── Financier ─────────────────────────────────────────────────────
    lc_amount:            float = 0.0
    tolerance_plus:       float = 0.0
    tolerance_minus:      float = 0.0
    currency_risk:        float = 0.0   # 0=EUR/USD low, 1=autres, 2=high risk

    # ── Structure documentaire ────────────────────────────────────────
    n_documents_required: float = 0.0
    n_clauses:            float = 0.0
    n_high_risk_clauses:  float = 0.0
    n_critical_clauses:   float = 0.0
    n_compliance_clauses: float = 0.0
    n_ucp_exclusions:     float = 0.0
    n_ucp_errors:         float = 0.0
    n_ucp_warnings:       float = 0.0

    # ── Règles transport/paiement ─────────────────────────────────────
    transshipment_allowed:  float = 0.0  # 1=allowed, 0=not
    partial_ship_allowed:   float = 0.0
    confirmation_required:  float = 0.0  # 1=CONFIRM
    charges_beneficiary:    float = 0.0  # 1=frais pour bénéf.

    # ── Incoterm / Pays ───────────────────────────────────────────────
    incoterm_risk:        float = 3.0   # 1-5
    country_risk:         float = 0.0   # 0–3 (3 = sanctions / FATF blacklist)

    # ── Scores NLP ────────────────────────────────────────────────────
    completeness_score:   float = 0.0   # 0-100
    semantic_score:       float = 0.0   # 0-100
    nlp_score:            float = 0.0   # 0-100
    ml_risk_score:        float = 0.0   # 0-1 (risk-clf existant)
    ucp_compliance_score: float = 0.0   # 0-100 (nouveau)

    # ── Présence d'entités clés ───────────────────────────────────────
    has_issuing_bank:     float = 0.0
    has_port_loading:     float = 0.0
    has_port_discharge:   float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convertit en vecteur numpy pour ML."""
        return np.array([
            self.validity_days, self.presentation_days, self.days_to_shipment,
            self.lc_amount, self.tolerance_plus, self.tolerance_minus,
            self.currency_risk,
            self.n_documents_required, self.n_clauses, self.n_high_risk_clauses,
            self.n_critical_clauses, self.n_compliance_clauses,
            self.n_ucp_exclusions, self.n_ucp_errors, self.n_ucp_warnings,
            self.transshipment_allowed, self.partial_ship_allowed,
            self.confirmation_required, self.charges_beneficiary,
            self.incoterm_risk, self.country_risk,
            self.completeness_score, self.semantic_score,
            self.nlp_score, self.ml_risk_score, self.ucp_compliance_score,
            self.has_issuing_bank, self.has_port_loading, self.has_port_discharge,
        ], dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.__dataclass_fields__.values()]

    def to_dict(self) -> dict:
        return asdict(self)


class LCFeatureExtractor:
    """
    Extrait les features ML depuis un résultat de pipeline + rapport UCP.
    """

    @staticmethod
    def _country_risk_from_entities(entities: dict) -> float:
        """Risque pays max : bénéficiaire, ports, goods, champ country (ISO2 ou contexte NK)."""
        blob = " ".join(
            str(entities.get(k) or "")
            for k in (
                "beneficiary",
                "port_loading",
                "port_discharge",
                "goods_description",
                "country",
            )
        )
        u = blob.upper()
        r = 0.0
        if re.search(
            r"NORTH\s+KOREA|DPRK|PYONGYANG|\bKP\b|DEMOCRATIC\s+PEOPLE'?S\s+REPUBLIC",
            u,
        ):
            r = max(r, 3.0)
        c2 = (entities.get("country") or "").strip().upper()[:2]
        if len(c2) == 2 and c2.isalpha():
            if c2 in FATF_BLACKLIST_ISO2:
                r = max(r, 3.0)
            elif c2 in HIGH_RISK_COUNTRIES:
                r = max(r, 2.0)
            elif c2 not in LOW_COUNTRY_ISO2:
                r = max(r, 1.0)
        return r

    def extract(
        self,
        pipeline_result: dict,
        ucp_report: Optional[object] = None,
    ) -> LCFeatures:
        """
        Extrait 29 features depuis pipeline_result + ucp_report.

        Args:
            pipeline_result: dict JSON du pipeline Swift
            ucp_report: UCPValidationReport (optionnel)
        """
        swift    = pipeline_result.get("swift", {})
        entities = swift.get("entities", {})
        lc       = swift.get("lc_criteria", {})
        nlp      = swift.get("nlp", {}) or pipeline_result.get("nlp", {})
        clauses  = nlp.get("clauses", [])
        semantic = nlp.get("semantic") or {}

        feat = LCFeatures()

        # ── Temporel ──────────────────────────────────────────────────
        try:
            issue  = datetime.strptime(entities.get("issue_date", ""), "%Y-%m-%d").date()
            expiry = datetime.strptime(entities.get("expiry_date", ""), "%Y-%m-%d").date()
            feat.validity_days = float((expiry - issue).days)
        except (ValueError, TypeError):
            feat.validity_days = 0.0

        try:
            ship = datetime.strptime(entities.get("shipment_date", ""), "%Y-%m-%d").date()
            feat.days_to_shipment = float((ship - issue).days) if "issue" in dir() else 0.0
        except (ValueError, TypeError):
            feat.days_to_shipment = 0.0

        feat.presentation_days = float(
            lc.get("presentation_period_days")
            or int(entities.get("presentation_period") or 21)
        )

        # ── Financier ─────────────────────────────────────────────────
        try:
            feat.lc_amount = float(str(entities.get("amount", "0")).replace(",", "."))
        except (ValueError, TypeError):
            feat.lc_amount = 0.0

        try:
            feat.tolerance_plus  = float(entities.get("tolerance_plus",  0) or 0)
            feat.tolerance_minus = float(entities.get("tolerance_minus", 0) or 0)
        except (ValueError, TypeError):
            pass

        currency = (entities.get("currency") or "").upper()
        if currency in HIGH_RISK_CURRENCIES:
            feat.currency_risk = 2.0
        elif currency not in ("EUR", "USD", "GBP", "CHF", "JPY"):
            feat.currency_risk = 1.0
        else:
            feat.currency_risk = 0.0

        # ── Structure documentaire ────────────────────────────────────
        feat.n_documents_required = float(len(lc.get("documents") or []))
        feat.n_clauses            = float(len(clauses))
        feat.n_high_risk_clauses  = float(
            sum(1 for c in clauses if c.get("risk_level") == "high")
        )
        feat.n_critical_clauses   = float(
            sum(1 for c in clauses if c.get("importance") == "critical")
        )
        feat.n_compliance_clauses = float(
            sum(1 for c in clauses if c.get("type") == "compliance_condition")
        )

        # ── UCP validator (si disponible) ─────────────────────────────
        if ucp_report is not None:
            ucp_dict = ucp_report.to_dict() if hasattr(ucp_report, "to_dict") else {}
            feat.n_ucp_exclusions    = float(len(ucp_dict.get("exclusions", {})))
            feat.n_ucp_errors        = float(ucp_dict.get("summary", {}).get("errors", 0))
            feat.n_ucp_warnings      = float(ucp_dict.get("summary", {}).get("warnings", 0))
            feat.ucp_compliance_score = float(ucp_dict.get("global_score", 0))
        else:
            # Compter les exclusions depuis les clauses (fallback)
            _EXCL_RE = re.compile(
                r"[Aa]rt(?:icle)?\.?\s*\d{1,2}[a-z]?.*?(?:not\s+applicable|excluded)",
                re.IGNORECASE
            )
            feat.n_ucp_exclusions = float(
                sum(1 for c in clauses if _EXCL_RE.search(c.get("text", "")))
            )

        # ── Règles transport/paiement ─────────────────────────────────
        feat.transshipment_allowed = float(
            "ALLOWED" in (entities.get("transhipment") or "").upper()
            and "NOT" not in (entities.get("transhipment") or "").upper()
        )
        feat.partial_ship_allowed = float(
            "ALLOWED" in (entities.get("partial_shipments") or "").upper()
            and "NOT" not in (entities.get("partial_shipments") or "").upper()
        )
        confirmation = entities.get("confirmation") or lc.get("confirmation") or ""
        feat.confirmation_required = float("CONFIRM" in confirmation.upper())
        feat.charges_beneficiary   = float(bool(lc.get("charges_for_beneficiary")))

        # ── Incoterm (priorité lc_criteria — aligné avec RiskFeatureExtractor)
        inc_raw = (lc.get("incoterm") or entities.get("incoterm") or "").strip()
        inc_key = re.sub(r"[^A-Z]", "", inc_raw.upper())[:3]
        feat.incoterm_risk = float(INCOTERM_RISK.get(inc_key, 3))

        # ── Pays : agrégat bénéficiaire / ports / NK context
        feat.country_risk = float(self._country_risk_from_entities(entities))

        # ── Scores NLP ────────────────────────────────────────────────
        feat.completeness_score = float(
            (semantic.get("completeness_score") or 0)
        )
        feat.semantic_score = float(semantic.get("semantic_score") or 0)
        feat.nlp_score      = float(nlp.get("score") or 0)

        # ML risk score existant
        risk_flags = semantic.get("risk_flags") or []
        if risk_flags:
            feat.ml_risk_score = float(max(
                f.get("ml_score", 0) for f in risk_flags
            ))

        # ── Présence entités ──────────────────────────────────────────
        feat.has_issuing_bank  = float(bool(entities.get("issuing_bank")))
        feat.has_port_loading  = float(bool(entities.get("port_loading")))
        feat.has_port_discharge = float(bool(entities.get("port_discharge")))

        return feat


# ══════════════════════════════════════════════════════════════════════
# DATASET GENERATOR — auto-génère des LC annotés depuis ner_dataset
# ══════════════════════════════════════════════════════════════════════

class LCDatasetGenerator:
    """
    Génère un dataset ML depuis ner_dataset.jsonl.
    Chaque texte SWIFT devient une ligne de features.
    """

    def __init__(self):
        self._extractor = LCFeatureExtractor()

    def generate_from_jsonl(
        self,
        jsonl_path: str,
        output_csv: str,
        max_samples: int = None,
    ) -> int:
        """
        Lit ner_dataset.jsonl, extrait les features de chaque texte,
        et sauvegarde en CSV.

        Returns: nombre de lignes générées
        """
        import csv

        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

        rows = []
        for rec in records:
            # Simuler un pipeline_result minimal depuis le texte brut
            entities = {}
            for ent in rec.get("entities", []):
                label = ent.get("label", "")
                val   = ent.get("value", "")
                if label == "APPLICANT":     entities["applicant"]     = val
                elif label == "BENEFICIARY": entities["beneficiary"]   = val
                elif label == "ISSUING_BANK":entities["issuing_bank"]  = val
                elif label == "AMOUNT":
                    # Extraire montant et devise
                    m = re.match(r"([A-Z]{3})\s*([\d,\.]+)", val)
                    if m:
                        entities["currency"] = m.group(1)
                        entities["amount"]   = m.group(2).replace(",", "")
                elif label == "INCOTERM":    entities["incoterm"]    = val
                elif label == "EXPIRY_DATE": entities["expiry_date"] = val
                elif label == "ISSUE_DATE":  entities["issue_date"]  = val
                elif label == "SHIPMENT_DATE": entities["shipment_date"] = val

            pseudo_pipeline = {
                "swift": {
                    "entities": entities,
                    "lc_criteria": {},
                    "nlp": {"clauses": [], "score": 0, "semantic": {}},
                }
            }

            try:
                feat = self._extractor.extract(pseudo_pipeline)
                row  = feat.to_dict()
                row["lc_id"]    = rec.get("id", f"lc_{len(rows)}")
                row["swift_type"] = rec.get("swift_type", "MT700")
                rows.append(row)
            except Exception as e:
                logger.debug("Erreur extraction features: %s", e)

        if not rows:
            return 0

        # Écrire CSV
        fieldnames = ["lc_id", "swift_type"] + list(LCFeatures().to_dict().keys())
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Dataset généré : %d lignes → %s", len(rows), output_csv)
        return len(rows)


# ══════════════════════════════════════════════════════════════════════
# RISK SCORER — Niveau 1
# ══════════════════════════════════════════════════════════════════════

@dataclass
class RiskScoringResult:
    lc_number:      str
    risk_score:     float         # 0.0 = faible, 1.0 = très risqué
    risk_level:     str           # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    risk_factors:   list[str]     # facteurs ayant contribué au risque
    feature_importances: dict     # feature → importance ML
    model_used:     str           # "rules" | "xgboost" | "random_forest"

    def to_dict(self) -> dict:
        return {
            "lc_number":     self.lc_number,
            "risk_score":    round(self.risk_score, 4),
            "risk_level":    self.risk_level,
            "risk_factors":  self.risk_factors,
            "top_features":  dict(list(self.feature_importances.items())[:5]),
            "model_used":    self.model_used,
        }


class LCRiskScorer:
    """
    Score de risque ML pour les Lettres de Crédit.

    Utilise XGBoost ou Random Forest si entraîné,
    sinon fallback sur un scoring par règles pondérées.
    """

    RISK_THRESHOLDS = {"LOW": 0.30, "MEDIUM": 0.55, "HIGH": 0.75, "CRITICAL": 1.0}

    def __init__(self, model=None, model_name: str = "rules"):
        self._model     = model
        self._model_name = model_name

    @classmethod
    def train(
        cls,
        features_csv: str,
        labels_csv:   str,
        model_type:   str = "xgboost",
        output_dir:   str = "models/lc-risk-ds",
    ) -> "LCRiskScorer":
        """
        Entraîne un modèle de scoring sur les features CSV.

        Args:
            features_csv: fichier CSV généré par LCDatasetGenerator
            labels_csv:   CSV avec colonnes lc_id, risk_label (0/1/2)
            model_type:   "xgboost" | "random_forest" | "logistic"
            output_dir:   dossier de sauvegarde
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report
        import pickle

        logger.info("Chargement données: %s + %s", features_csv, labels_csv)
        X_df  = pd.read_csv(features_csv)
        y_df  = pd.read_csv(labels_csv)

        merged = X_df.merge(y_df, on="lc_id")
        feature_cols = [c for c in X_df.columns if c not in ("lc_id", "swift_type")]

        X = merged[feature_cols].fillna(0).values
        y = merged["risk_label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    use_label_encoder=False, eval_metric="mlogloss",
                    random_state=42,
                )
            except ImportError:
                logger.warning("xgboost non installé, fallback Random Forest")
                model_type = "random_forest"

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
            model = LogisticRegression(max_iter=1000, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=["LOW","MEDIUM","HIGH"])
        logger.info("Classification report:\n%s", report)

        # Sauvegarde
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "lc_risk_model.pkl", "wb") as f:
            pickle.dump({
                "model":        model,
                "model_type":   model_type,
                "feature_cols": feature_cols,
                "report":       report,
            }, f)

        logger.info("Modèle sauvegardé dans %s", output_dir)
        return cls(model=model, model_name=model_type)

    @classmethod
    def load(cls, model_dir: str) -> "LCRiskScorer":
        """Charge un modèle pré-entraîné."""
        import pickle
        model_path = Path(model_dir) / "lc_risk_model.pkl"
        if not model_path.exists():
            logger.warning("Modèle non trouvé : %s — fallback règles", model_path)
            return cls()
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        return cls(model=data["model"], model_name=data["model_type"])

    def score(self, features: LCFeatures, lc_number: str = "?") -> RiskScoringResult:
        """
        Score le risque d'une LC.
        Si aucun modèle ML n'est chargé, utilise un scoring par règles.
        """
        if self._model is not None:
            return self._score_ml(features, lc_number)
        return self._score_rules(features, lc_number)

    def _score_ml(self, features: LCFeatures, lc_number: str) -> RiskScoringResult:
        """Scoring via modèle ML."""
        X = features.to_vector().reshape(1, -1)
        try:
            proba = self._model.predict_proba(X)[0]
            # Classe 0=LOW, 1=MEDIUM, 2=HIGH
            risk_score = float(proba[1] * 0.5 + proba[2] * 1.0)
        except AttributeError:
            pred = self._model.predict(X)[0]
            risk_score = float(pred) / 2.0

        risk_level, factors = self._classify_risk(risk_score, features)

        # Feature importances si disponible
        importances = {}
        if hasattr(self._model, "feature_importances_"):
            names = features.feature_names
            imps  = self._model.feature_importances_
            importances = dict(sorted(
                zip(names, imps), key=lambda x: -x[1]
            )[:10])

        return RiskScoringResult(
            lc_number=lc_number,
            risk_score=risk_score,
            risk_level=risk_level,
            risk_factors=factors,
            feature_importances=importances,
            model_used=self._model_name,
        )

    def _score_rules(self, features: LCFeatures, lc_number: str) -> RiskScoringResult:
        """
        Scoring par règles pondérées — fallback sans modèle ML.
        Chaque règle contribue à un score 0-1.
        """
        score   = 0.0
        factors = []

        # Validité très courte (ignore 0 = date d'expiration absente / inconnue)
        if 0 < features.validity_days < 7:
            score += 0.25
            factors.append(f"Validité très courte ({features.validity_days:.0f} jours)")
        elif 0 < features.validity_days < 21:
            score += 0.10
            factors.append(f"Validité courte ({features.validity_days:.0f} jours)")

        # Clauses critiques / risque élevé
        if features.n_critical_clauses >= 3:
            score += 0.15
            factors.append(f"{features.n_critical_clauses:.0f} clauses critiques")
        if features.n_high_risk_clauses >= 3:
            score += 0.10
            factors.append(f"{features.n_high_risk_clauses:.0f} clauses à risque élevé")

        # Violations UCP
        if features.n_ucp_errors >= 5:
            score += 0.20
            factors.append(f"{features.n_ucp_errors:.0f} violations UCP bloquantes")
        elif features.n_ucp_errors >= 2:
            score += 0.10
            factors.append(f"{features.n_ucp_errors:.0f} violations UCP")

        # Exclusions d'articles clés
        if features.n_ucp_exclusions >= 3:
            score += 0.15
            factors.append(f"{features.n_ucp_exclusions:.0f} articles UCP exclus")

        # Incoterm risqué
        if features.incoterm_risk >= 4:
            score += 0.08
            factors.append(f"Incoterm à risque élevé (score {features.incoterm_risk:.0f})")

        # Pays à risque
        if features.country_risk >= 3:
            score += 0.28
            factors.append("Juridiction liste noire FATF / sanctions sévères")
        elif features.country_risk >= 2:
            score += 0.20
            factors.append("Pays à risque élevé (liste OFAC/FATF)")
        elif features.country_risk >= 1:
            score += 0.08
            factors.append("Pays à risque modéré")

        # Devise à risque
        if features.currency_risk >= 2:
            score += 0.15
            factors.append("Devise à risque de change très élevé")

        # Complétude faible
        if features.completeness_score < 50:
            score += 0.10
            factors.append(f"Complétude faible ({features.completeness_score:.0f}%)")

        # Banque émettrice absente
        if features.has_issuing_bank == 0:
            score += 0.05
            factors.append("Banque émettrice non identifiée")

        # Score ML existant (risk-clf pipeline)
        score += features.ml_risk_score * 0.15

        # Plafonner à 1.0
        score = min(1.0, score)
        risk_level, _ = self._classify_risk(score, features)

        return RiskScoringResult(
            lc_number=lc_number,
            risk_score=score,
            risk_level=risk_level,
            risk_factors=factors,
            feature_importances={},
            model_used="rules",
        )

    def _classify_risk(
        self, score: float, features: LCFeatures
    ) -> tuple[str, list[str]]:
        factors = []
        if score < 0.30:
            level = "LOW"
        elif score < 0.55:
            level = "MEDIUM"
        elif score < 0.75:
            level = "HIGH"
        else:
            level = "CRITICAL"
        return level, factors


# ══════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION — Niveau 2
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyResult:
    lc_number:     str
    is_anomaly:    bool
    anomaly_score: float      # plus négatif = plus anormal
    percentile:    float      # percentile parmi les LC normaux
    anomalous_features: list[str]

    def to_dict(self) -> dict:
        return {
            "lc_number":          self.lc_number,
            "is_anomaly":         self.is_anomaly,
            "anomaly_score":      round(self.anomaly_score, 4),
            "percentile":         round(self.percentile, 1),
            "anomalous_features": self.anomalous_features,
        }


class LCAnomalyDetector:
    """
    Détection d'anomalies sur les Lettres de Crédit.
    Utilise IsolationForest (non supervisé — pas de labels requis).
    Idéal pour détecter des LC frauduleux ou inhabituels.
    """

    def __init__(self, model=None, scaler=None, train_scores: np.ndarray = None):
        self._model        = model
        self._scaler       = scaler
        self._train_scores = train_scores

    @classmethod
    def train(
        cls,
        features_csv: str,
        output_dir:   str = "models/lc-anomaly",
        contamination: float = 0.05,
    ) -> "LCAnomalyDetector":
        """
        Entraîne un IsolationForest sur un ensemble de LC "normaux".

        Args:
            features_csv:  CSV généré par LCDatasetGenerator
            contamination: fraction attendue d'anomalies (5% par défaut)
        """
        import pandas as pd
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        import pickle

        df = pd.read_csv(features_csv)
        feature_cols = [c for c in df.columns if c not in ("lc_id", "swift_type")]
        X = df[feature_cols].fillna(0).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled)

        # Calculer les scores sur les données d'entraînement
        train_scores = model.score_samples(X_scaled)

        # Sauvegarde
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "lc_anomaly_model.pkl", "wb") as f:
            pickle.dump({
                "model":        model,
                "scaler":       scaler,
                "train_scores": train_scores,
                "feature_cols": feature_cols,
            }, f)

        n_anomalies = (model.predict(X_scaled) == -1).sum()
        logger.info(
            "IsolationForest entraîné : %d LC normaux, %d anomalies détectées (%.1f%%)",
            len(X) - n_anomalies, n_anomalies, 100 * n_anomalies / len(X)
        )
        return cls(model=model, scaler=scaler, train_scores=train_scores)

    @classmethod
    def load(cls, model_dir: str) -> "LCAnomalyDetector":
        import pickle
        path = Path(model_dir) / "lc_anomaly_model.pkl"
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            model=data["model"],
            scaler=data["scaler"],
            train_scores=data.get("train_scores"),
        )

    def predict(self, features: LCFeatures, lc_number: str = "?") -> AnomalyResult:
        """Détecte si une LC est anormale."""
        if self._model is None:
            return AnomalyResult(
                lc_number=lc_number,
                is_anomaly=False,
                anomaly_score=0.0,
                percentile=50.0,
                anomalous_features=["Modèle non chargé — utiliser .train() d'abord"],
            )

        X = features.to_vector().reshape(1, -1)
        X_scaled = self._scaler.transform(X)

        prediction = self._model.predict(X_scaled)[0]   # -1 = anomalie
        score      = float(self._model.score_samples(X_scaled)[0])
        is_anomaly = (prediction == -1)

        # Percentile par rapport aux LC d'entraînement
        percentile = 50.0
        if self._train_scores is not None and len(self._train_scores) > 0:
            percentile = float(100 * (score > self._train_scores).mean())

        # Identifier les features anormales
        anomalous = self._identify_anomalous_features(features, X_scaled[0])

        return AnomalyResult(
            lc_number=lc_number,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            percentile=percentile,
            anomalous_features=anomalous,
        )

    def _identify_anomalous_features(
        self, features: LCFeatures, X_scaled: np.ndarray
    ) -> list[str]:
        """Identifie les features dont les valeurs sont extrêmes."""
        anomalous = []
        feat_dict = features.to_dict()
        names     = list(feat_dict.keys())

        # Features avec valeur standardisée > 2.5σ
        for i, (name, val) in enumerate(zip(names, X_scaled)):
            if abs(val) > 2.5:
                raw_val = feat_dict[name]
                anomalous.append(f"{name}={raw_val:.2f} ({val:+.1f}σ)")

        return anomalous[:5]  # top 5


# ══════════════════════════════════════════════════════════════════════
# SIMILARITY SEARCH — Niveau 3
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SimilarLC:
    lc_id:      str
    similarity: float
    features:   dict


class LCSimilaritySearch:
    """
    Recherche les LC les plus similaires à une LC donnée.
    Utilise la similarité cosine sur le vecteur de features.
    Utile pour : retrouver des précédents, benchmarker un LC.
    """

    def __init__(self, index: np.ndarray = None, lc_ids: list[str] = None):
        self._index  = index   # (N, D) matrice normalisée
        self._lc_ids = lc_ids or []

    @classmethod
    def build_index(
        cls,
        features_csv: str,
        output_dir: str = "models/lc-similarity",
    ) -> "LCSimilaritySearch":
        """Construit l'index de similarité depuis un CSV de features."""
        import pandas as pd
        import pickle
        from sklearn.preprocessing import normalize

        df = pd.read_csv(features_csv)
        lc_ids      = df["lc_id"].tolist()
        feature_cols = [c for c in df.columns if c not in ("lc_id", "swift_type")]
        X = df[feature_cols].fillna(0).values.astype(np.float32)

        # Normaliser pour cosine similarity
        X_norm = normalize(X, norm="l2")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "lc_similarity_index.pkl", "wb") as f:
            pickle.dump({"index": X_norm, "lc_ids": lc_ids, "feature_cols": feature_cols}, f)

        logger.info("Index similarité : %d LC indexés", len(lc_ids))
        return cls(index=X_norm, lc_ids=lc_ids)

    @classmethod
    def load(cls, model_dir: str) -> "LCSimilaritySearch":
        import pickle
        path = Path(model_dir) / "lc_similarity_index.pkl"
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(index=data["index"], lc_ids=data["lc_ids"])

    def find_similar(
        self, features: LCFeatures, top_k: int = 5
    ) -> list[SimilarLC]:
        """Retourne les top_k LC les plus similaires."""
        if self._index is None:
            return []

        from sklearn.preprocessing import normalize

        q     = features.to_vector().reshape(1, -1).astype(np.float32)
        q_norm = normalize(q, norm="l2")[0]

        # Cosine similarity avec tout l'index
        sims = self._index @ q_norm

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append(SimilarLC(
                lc_id      = self._lc_ids[idx],
                similarity = float(sims[idx]),
                features   = {},
            ))
        return results


# ══════════════════════════════════════════════════════════════════════
# PIPELINE DS COMPLET
# ══════════════════════════════════════════════════════════════════════

class SwiftDSPipeline:
    """
    Pipeline Data Science complet pour LC.

    Orchestration :
        pipeline_result
            ↓
        Feature Engineering (LCFeatureExtractor)
            ↓
        Risk Scoring     (LCRiskScorer)       → risk_score
        Anomaly Detection (LCAnomalyDetector) → is_anomaly
        UCP Compliance   (UCP600Validator)    → ucp_score
        Similarity       (LCSimilaritySearch) → similar_lcs
            ↓
        DS Report
    """

    def __init__(
        self,
        models_dir: str = "models",
        ucp_validator=None,
    ):
        self._extractor  = LCFeatureExtractor()
        self._scorer     = LCRiskScorer.load(f"{models_dir}/lc-risk-ds")
        self._detector   = LCAnomalyDetector.load(f"{models_dir}/lc-anomaly")
        self._search     = LCSimilaritySearch.load(f"{models_dir}/lc-similarity")
        self._ucp        = ucp_validator

    def analyze(self, pipeline_result: dict) -> dict:
        """
        Analyse complète Data Science d'un LC.
        Retourne un dict enrichi avec tous les scores ML.
        """
        swift    = pipeline_result.get("swift", {})
        entities = swift.get("entities", {})
        lc_number = entities.get("lc_number", "?")

        # 1. UCP validation
        ucp_report = None
        if self._ucp:
            ucp_report = self._ucp.validate(pipeline_result)

        # 2. Feature engineering
        features = self._extractor.extract(pipeline_result, ucp_report)

        # 3. Risk scoring
        risk_result = self._scorer.score(features, lc_number)

        # 4. Anomaly detection
        anomaly_result = self._detector.predict(features, lc_number)

        # 5. Similarity search
        similar = self._search.find_similar(features, top_k=3)

        return {
            "lc_number":   lc_number,
            "ds_analysis": {
                "features":   features.to_dict(),
                "risk":       risk_result.to_dict(),
                "anomaly":    anomaly_result.to_dict(),
                "similar_lcs": [
                    {"lc_id": s.lc_id, "similarity": round(s.similarity, 3)}
                    for s in similar
                ],
                "ucp_compliance": ucp_report.to_dict() if ucp_report else None,
            }
        }