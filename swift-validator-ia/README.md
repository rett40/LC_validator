# swift-validator-ia

Système de validation automatique de Lettres de Crédit (LC) et d'analyse de messages SWIFT, combinant OCR, extraction structurée, NLP avancé, vérification externe de conformité et scoring de risque par Machine Learning.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture générale](#2-architecture-générale)
3. [Pipeline — Flux de données](#3-pipeline--flux-de-données)
4. [Couche 1 — OCR & Prétraitement image](#4-couche-1--ocr--prétraitement-image)
5. [Couche 2 — Correction orthographique SWIFT](#5-couche-2--correction-orthographique-swift)
6. [Couche 3 — Détection & Parsing SWIFT](#6-couche-3--détection--parsing-swift)
7. [Couche 4 — NLP avancé](#7-couche-4--nlp-avancé)
8. [Couche 5 — Validation UCP 600](#8-couche-5--validation-ucp-600)
9. [Couche 6 — Pipeline Data Science (DS)](#9-couche-6--pipeline-data-science-ds)
10. [Couche 7 — Validation externe (Sanctions & AML)](#10-couche-7--validation-externe-sanctions--aml)
11. [Couche 8 — Risk ML Engine](#11-couche-8--risk-ml-engine)
12. [Modèles IA — Détail technique](#12-modèles-ia--détail-technique)
13. [Stack technologique](#13-stack-technologique)
14. [Structure du projet](#14-structure-du-projet)
15. [API REST](#15-api-rest)
16. [Installation & Démarrage](#16-installation--démarrage)
17. [Entraînement des modèles](#17-entraînement-des-modèles)
18. [Format de sortie](#18-format-de-sortie)
19. [Performance](#19-performance)

---

## 1. Vue d'ensemble

**swift-validator-ia** est un moteur d'analyse et de validation de Lettres de Crédit documentaires (LC). Il traite des messages SWIFT bruts (formats MT700, MT710, MT720, MT760, MT790, MT799) issus de PDFs, images scannées ou texte structuré, et produit un rapport JSON unifié couvrant :

- L'extraction et la normalisation de toutes les entités LC (parties, montants, dates, marchandises…)
- La conformité aux règles UCP 600 (Uniform Customs and Practice for Documentary Credits)
- La vérification des contreparties contre les listes de sanctions internationales (OFAC, UE, ONU, GAFI)
- Un score de risque ML (XGBoost + ensemble) avec explication SHAP et segment client

**Version pipeline :** 6.6  
**Version API :** 6.3  
**Langage :** Python 3.10+  
**Framework API :** Flask  
**Interface analyse :** Streamlit

### Cas d'usage principaux

| Cas d'usage | Description |
|---|---|
| Validation documentaire | Vérifier qu'un SWIFT MT700 est conforme UCP 600 avant prise en charge bancaire |
| Screening AML/Sanctions | Détecter toute contrepartie ou juridiction sous sanctions OFAC/UE/ONU |
| Scoring de risque | Produire un score de risque (LOW / MEDIUM / HIGH) avec explication en langage naturel |
| Audit trail | Générer un rapport d'analyse traçable et reproductible par dossier LC |

---

## 2. Architecture générale

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRÉE (PDF / Image / Texte)                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │           SwiftPipeline v6.6                │
          │  Orchestrateur central — 9 étapes           │
          └──────────────────────┬──────────────────────┘
                                 │
    ┌────────────────────────────▼──────────────────────────────────────┐
    │  [1] OCR          [2] SPELL       [3] DETECT      [4] PARSE       │
    │  EasyOCR +        SWIFT dict +    Tag-scoring     Parsers MT7xx   │
    │  Tesseract        Bigrams         per schema      + règles métier │
    └────────────────────────────┬──────────────────────────────────────┘
                                 │
    ┌────────────────────────────▼──────────────────────────────────────┐
    │  [5] NLP (parallèle)                                              │
    │    ├─ NER Ensemble    (règles + BERT fine-tuné)                   │
    │    ├─ Clause Clf      (SentenceTransformer + LogisticRegression)  │
    │    └─ Sémantique v3   (NLI, RapidFuzz, graphe de risque UCP)      │
    └────────────────────────────┬──────────────────────────────────────┘
                                 │
    ┌────────────────────────────▼──────────────────────────────────────┐
    │  [6] UCP 600 v3       [7] DS Pipeline      [8] Validation externe │
    │  Art.5,14,16,18,20    LCFeatureExtractor   OFAC + EU + ONU + GAFI │
    │  29 — cascade rules   LCRiskScorer (rules) Country + Currency     │
    └────────────────────────────┬──────────────────────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │         [9] Risk ML Engine v2.0             │
          │   XGBoost + IsolationForest + KMeans + SHAP │
          └──────────────────────┬──────────────────────┘
                                 │
          ┌──────────────────────▼──────────────────────┐
          │       SORTIE — Payload JSON unifié          │
          │  ocr · swift · ucp · ds_analysis ·          │
          │  external_validation · risk_ml              │
          └─────────────────────────────────────────────┘
```

---

## 3. Pipeline — Flux de données

```
INPUT
  │
  ├─ [OCR]          PDF/image → texte brut + score de confiance
  ├─ [SPELL]        Correction SWIFT (dictionnaire + bigrammes)
  ├─ [DETECT]       Identification du type MT (MT700, MT720…)
  ├─ [PARSE]        Extraction des entités via parsers métier
  ├─ [LC_CRITERIA]  Vérification acceptation LC (MT700/MT720 seulement)
  │
  ├─ [NLP]  ── Parallèle ──────────────────────────────────────────────┐
  │         ├─ NER (extraction des entités nommées)                    │
  │         ├─ Classification des clauses                              │
  │         └─ Analyse sémantique + contradictions NLI                 │
  │                                                                    │
  ├─ [MERGE]        Fusion entités règles ↔ NER (seuils 0.70 / 0.80)
  ├─ [VALIDATE]     Validation champs + cross-field
  │
  ├─ [UCP 600]      Scoring de conformité UCP 600
  ├─ [DS]           Extraction features + scoring risque règles
  │
  ├─ [EXT_VAL]      Vérification pays + devise (cache local → API)
  └─ [RISK ML]      Prédiction ML + anomalie + segment + SHAP

OUTPUT: payload JSON unifié
```

---

## 4. Couche 1 — OCR & Prétraitement image

**Service :** `services/ocr/`  
**Objectif :** Transformer un document physique (PDF scanné, photo, image) en texte exploitable, en maximisant la fidélité au message SWIFT d'origine.

### Composants

| Fichier | Rôle |
|---|---|
| `ocr_service.py` | Orchestrateur OCR — choisit le moteur selon le type de fichier |
| `pdf_detector.py` | Distingue PDF natif (texte extractible) et PDF scanné |
| `pdf_text_extractor.py` | Extraction directe via PyMuPDF pour PDF natifs |
| `ocr_engines.py` | Moteurs OCR hybrides (EasyOCR + Tesseract) |
| `image_enhancer.py` | Prétraitement image (OpenCV) |
| `postprocessor.py` | Nettoyage post-OCR |
| `spell_correction.py` | Correction SWIFT (dictionnaire + bigrammes) |

### Technique — Moteur OCR hybride

Le service combine deux moteurs complémentaires :

**EasyOCR** (moteur primaire)
- Basé sur PyTorch + deep learning (CRAFT pour la détection de zones, CRNN pour la reconnaissance de caractères)
- Supporte 80+ langues, robuste aux textes inclinés ou dégradés
- Retourne un score de confiance par zone détectée

**Tesseract** (moteur de secours / validation)
- OCR classique (LSTM dans les versions récentes)
- Utilisé pour les zones à faible confiance EasyOCR
- Profil de langue `eng+fra`

**Logique de sélection :**
```
PDF natif  → PyMuPDF extraction directe (le plus fiable)
PDF scanné → Conversion page par page via pdf2image → OCR hybride
Image      → Prétraitement OpenCV → OCR hybride
```

### Prétraitement image (OpenCV)

Avant OCR, chaque image passe par un pipeline de prétraitement :
1. **Conversion niveaux de gris** — réduction du bruit couleur
2. **Binarisation adaptative** (Otsu ou Bradley) — contraste texte/fond
3. **Deskewing** — correction de l'inclinaison
4. **Débruitage** — filtre médian ou gaussien
5. **Upscaling** — résolution normalisée à 300 DPI
6. **Suppression des zones vides** — concentration sur les zones de texte

### Sortie

```json
{
  "text": "...",
  "confidence": 0.94,
  "spell_report": {
    "corrections_applied": 12,
    "changes": [{"original": "BENFICIARY", "corrected": "BENEFICIARY"}]
  }
}
```

---

## 5. Couche 2 — Correction orthographique SWIFT

**Service :** `services/ocr/spell_correction.py`  
**Objectif :** Corriger les erreurs OCR spécifiques au vocabulaire SWIFT avant tout traitement NLP. Un tag mal reconnu (`27C` → `27G`) ou un mot-clé corrompu (`IRREVOCAVLE` → `IRREVOCABLE`) peut bloquer toute l'analyse en aval.

### Technique

**Dictionnaire SWIFT dédié**  
Un lexique de plusieurs centaines de termes spécifiques au Trade Finance : noms de tags SWIFT (`40A`, `41D`, `42C`…), mots-clés UCP (`IRREVOCABLE`, `CONFIRMED`, `DEFERRED`…), incoterms (`FOB`, `CIF`, `DAP`…), devises, noms de banques courants.

**Modèle de bigrammes**  
Corrige les séquences improbables de caractères dans le contexte SWIFT. Par exemple, la séquence `BENIFICIARY` est statistiquement improbable dans un message SWIFT ; le modèle la substitue par `BENEFICIARY`.

**Validation contextuelle**  
La correction vérifie que le terme substitué est cohérent avec son contexte immédiat (tag SWIFT dans lequel il apparaît).

**Sortie :** texte corrigé + rapport de modifications (utile pour l'audit).

---

## 6. Couche 3 — Détection & Parsing SWIFT

**Services :** `services/swift/`  
**Objectif :** Identifier le type de message SWIFT et extraire structurellement toutes les entités du formulaire.

### Détection du type MT (`swift_detector.py`)

**Algorithme :** Pondération des tags SWIFT présents dans le texte.

Chaque type MT possède un schéma de tags attendus défini dans `data/swift_schema/mt7xx_schema.json`. Le détecteur calcule un score par type en comptant les tags reconnus pondérés par leur importance dans ce type.

```
Score(MT700) = Σ (poids_tag_i × présent_i) pour tous les tags du schéma MT700
Type retenu  = argmax(Score)
```

### Parsing structuré

**Factory de parsers (`parsers/parser_factory.py`)** — instancie le parser spécifique au type MT détecté.

Chaque parser hérite de `SwiftBaseParser` et implémente l'extraction des champs propres à son type. L'extraction combine :
- **Regex patterns** (`patterns.py`) — expressions régulières pour chaque tag SWIFT
- **Règles métier** — validation et normalisation (dates ISO 8601, montants numériques, codes ISO 4217…)
- **Normalisation des tags** (`swift_tag_normalizer.py`) — harmonisation des variantes d'un même tag

### Entités extraites (`SwiftEntities`)

| Entité | Description |
|---|---|
| `lc_number` | Numéro de crédit documentaire |
| `applicant` | Donneur d'ordre (acheteur) |
| `beneficiary` | Bénéficiaire (vendeur) |
| `issuing_bank` | Banque émettrice |
| `advising_bank` | Banque notificatrice |
| `amount` | Montant LC (numérique) |
| `currency` | Devise (ISO 4217) |
| `expiry_date` | Date d'expiration |
| `issue_date` | Date d'émission |
| `shipment_date` | Date limite d'expédition |
| `goods_description` | Description des marchandises |
| `incoterm` | Incoterm (FOB, CIF, DAP…) |
| `country` | Pays principal |
| `port_of_loading` | Port de chargement |
| `port_of_discharge` | Port de déchargement |
| `payment_terms` | Conditions de paiement |
| `partial_shipment` | Expéditions partielles (autorisé/interdit) |
| `transshipment` | Transbordement (autorisé/interdit) |
| `presentation_period` | Délai de présentation documents |
| `clauses` | Clauses additionnelles (texte libre) |

### Validation SWIFT (`swift_validator.py`)

Après l'extraction, un validateur vérifie :
- **Champs obligatoires** selon le type MT
- **Format des champs** (longueur, caractères autorisés par tag)
- **Contraintes cross-field** (ex. : date expiration > date émission, montant cohérent avec devise)

---

## 7. Couche 4 — NLP avancé

**Service :** `services/swift_nlp/` — `SwiftNLPService` v2.0  
**Objectif :** Analyser le contenu sémantique du message SWIFT au-delà de l'extraction structurelle : identifier les entités nommées par des modèles de langage, classifier les clauses par nature, détecter les contradictions et incohérences.

Cette couche se décompose en trois modules exécutés en parallèle.

---

### 4.1 — NER : Extraction d'entités nommées

**Composant :** `services/swift_nlp/ner/swift_ner_extractor.py` v2.0  
**Objectif :** Extraire et valider les entités (parties, montants, dates, pays…) en combinant règles et modèles Transformer.

#### Architecture en ensemble

```
Texte SWIFT
    │
    ├─── [Mode 1] Règles + Regex
    │     Tags SWIFT → patterns dédiés par entité
    │     Confiance basée sur la spécificité du pattern
    │
    ├─── [Mode 2] Transformer (BERT fine-tuné)
    │     Token classification (IOB2)
    │     Modèle : models/swift-ner/checkpoint-2660/
    │     Confiance = max(softmax) par token
    │
    └─── [Ensemble] Moyenne pondérée
          Règles : 0.65 · BERT : 0.35
          Entité retenue si confiance ≥ 0.70
          Override règles si confiance BERT ≥ 0.80
```

#### Modèle NER SWIFT fine-tuné

- **Base :** `bert-base-multilingual-cased`
- **Tâche :** Token classification — étiquetage IOB2
- **Étiquettes :** 13 catégories d'entités LC
- **Checkpoint :** `models/swift-ner/checkpoint-2660/`
- **Fine-tuning :** `scripts/fine_tune_ner.py` sur corpus SWIFT annoté
- **Normalisation post-extraction :** dates → ISO 8601, montants → float, codes pays → ISO 3166-2

#### Labels NER

```
APPLICANT, BENEFICIARY, ISSUING_BANK, ADVISING_BANK,
AMOUNT, CURRENCY,
GOODS, INCOTERM,
EXPIRY_DATE, ISSUE_DATE, SHIPMENT_DATE,
COUNTRY, PORT, LC_NUMBER
```

#### Fusion avec les entités règles (`_merge_entities`)

```
Si confiance NER ≥ 0.80  → NER prend le dessus (override)
Si confiance NER ≥ 0.70  → NER prend le dessus si règles vides
Sinon                    → entités règles conservées
```

---

### 4.2 — Classification des clauses

**Composant :** `services/swift_nlp/clauses/swift_clause_classifier.py` v2.0  
**Objectif :** Classer chaque clause additionnelle par type fonctionnel pour permettre la détection d'incohérences UCP et le scoring de risque.

#### Pipeline de classification (4 niveaux en cascade)

```
Clause texte
    │
    ├─ [Niveau 1] Règles déterministes (regex)
    │   → pattern très spécifique → assignation directe, confiance 0.95
    │
    ├─ [Niveau 2] Scoring par mots-clés pondérés
    │   → score par type basé sur fréquence termes-clés
    │
    ├─ [Niveau 3] Similarité d'embedding (SentenceTransformer)
    │   → comparaison cosinus avec prototypes de chaque type
    │   → modèle : paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
    │
    └─ [Niveau 4] Classificateur supervisé (LogisticRegression)
        → features : embedding 384-dim
        → modèle : models/clause-clf/clf.pkl
```

#### Types de clauses

| Type | Description |
|---|---|
| `document_condition` | Exigences documentaires, certifications requises |
| `shipment_condition` | Expédition partielle, transbordement, ports |
| `payment_condition` | Conditions de paiement, frais, remboursement |
| `compliance_condition` | UCP, délai de présentation, règles de conformité |
| `tolerance_condition` | Tolérance sur montant ou quantité |
| `standard_clause` | Clause standard sans classification spécifique |

#### Attributs produits par clause

- `type` — type fonctionnel
- `confidence` — confiance (0-1)
- `importance` — criticité (critical / high / medium / low / info)
- `ucp_articles` — articles UCP pertinents (ex. Art.14, Art.20)
- `risk_indicators` — indicateurs de risque détectés

---

### 4.3 — Analyse sémantique v3.0

**Composants :** `services/swift_nlp/semantic/swift_semantic_analyzer.py` + `swift_nli_contradiction.py`  
**Objectif :** Détecter les contradictions entre les champs structurés SWIFT et les clauses en langage naturel, identifier les incohérences temporelles, les indicateurs de fraude, et produire une décision de routage.

#### Techniques

**Fuzzy matching (RapidFuzz)**  
Résistance au bruit OCR : rapprochement approximatif des clauses avant analyse, tolère les fautes d'orthographe résiduelles.

**Natural Language Inference (NLI)**  
- Modèle : `cross-encoder/nli-deberta-v3-xsmall` (offline, `models/nli/`)
- Tâche : détection de contradiction entre la valeur d'un tag SWIFT (prémisse) et le texte d'une clause (hypothèse)
- Résultat : ENTAILMENT / NEUTRAL / CONTRADICTION + score de confiance

**Scoring Bayésien par champ**  
Estimation de la fiabilité de chaque entité extraite (intervalle de confiance) basée sur la cohérence entre sources (tag, clause, NER).

**Graphe de risque UCP**  
Un graphe de propagation encode les dépendances entre articles UCP. La violation de l'Art.14 peut aggraver le score de l'Art.16. Ce graphe permet de calculer un impact en cascade lors du scoring de conformité.

**Matrice de risque Incoterm**  
Profil de risque ordinalisé (1-4) par Incoterm selon l'exposition acheteur/vendeur.

**Détection de patterns temporels**  
Vérification de la cohérence des dates : date expiration > date expédition > date émission, délai de présentation ≥ 5 jours (Art.14c UCP 600).

#### Indicateurs de fraude

Le moteur identifie une liste de red flags documentaires :
- Montant LC très supérieur à la valeur commerciale apparente
- Bénéficiaire dans une juridiction incohérente avec les marchandises
- Clause excluant tous les documents standards
- Délai de présentation anormalement court
- Contradiction entre pays de chargement et pays du bénéficiaire

#### Sortie

```json
{
  "warnings": [...],
  "risk_flags": [...],
  "contradictions": [
    {"tag": "42C", "clause_index": 3, "type": "CONTRADICTION", "confidence": 0.87}
  ],
  "completeness_score": 82,
  "semantic_score": 74,
  "fraud_indicators": [...],
  "recommendations": [...],
  "routing_decision": "REVIEW"
}
```

---

## 8. Couche 5 — Validation UCP 600

**Composant :** `services/swift_nlp/ucp/swift_ucp_validator.py` v3.0  
**Objectif :** Vérifier la conformité du message SWIFT aux règles de la publication UCP 600 de la CCI (Chambre de Commerce Internationale) — le standard international régissant les crédits documentaires.

### Articles couverts

| Article | Titre | Type de vérification |
|---|---|---|
| Art. 5 | Primauté des clauses sur les tags | Règle hard — la clause prend le dessus sur le tag |
| Art. 14 | Standard d'examen des documents | Délai de présentation (5 jours), caractère exhaustif |
| Art. 16 | Documents non conformes (discrepant) | Détection des irrégularités documentaires |
| Art. 17 | Cachets & apostilles | Conformité des documents originaux |
| Art. 18 | Documents exclus | Exclusions déclarées dans les clauses |
| Art. 20 | Date d'expiration | Validité temporelle + expiration vs. expédition |
| Art. 29 | Délai de présentation | Cohérence présentation ≤ expiration |
| OFAC/Embargo | Sanctions internationales | Pays avec risk level CRITICAL → blocage |

### Moteur de scoring

**UCPFinding** — chaque violation est caractérisée par :
- `weight` — poids dans le score global (0-1)
- `confidence` — confiance de la détection (0-1)
- `cascade_impact` — impact sur d'autres articles (graphe de propagation)
- `blocks_payment` — si `True`, bloque le paiement LC

**Règles hard (blocantes)** — violations qui forcent `payment_blocked = True` indépendamment du score :
- Pays bénéficiaire sous embargo OFAC/UE
- Absence totale de document exigé
- Date d'expiration passée

**Moteur de cascade** — une violation d'un article A peut aggraver le score d'un article B lié. Le graphe de dépendances propage les pénalités de façon multiplicative.

### Sortie

```json
{
  "global_score": 67,
  "ucp_decision": "REVIEW",
  "payment_blocked": false,
  "findings": [...],
  "errors": ["Art.14 — Délai de présentation < 5 jours"],
  "warnings": ["Art.20 — Marge expiration/expédition < 15 jours"],
  "processing_ms": 312
}
```

---

## 9. Couche 6 — Pipeline Data Science (DS)

**Composant :** `services/swift_nlp/ds/swift_ds_pipeline.py`  
**Objectif :** Construire le vecteur de features structuré qui alimentera le Risk ML Engine, et produire un premier scoring de risque basé sur des règles métier déterministes.

### LCFeatureExtractor — Construction du vecteur de features

Agrège les sorties de toutes les couches précédentes en 21 features canoniques :

#### Groupe 1 — Fondamentaux LC

| Feature | Source | Description |
|---|---|---|
| `lc_amount` | Entités SWIFT | Montant LC (numérique) |
| `currency` | Entités SWIFT | Code devise ISO 4217 |
| `days_to_shipment` | Dates extraites | Jours jusqu'à la date limite d'expédition |
| `presentation_days` | Entités SWIFT | Délai de présentation documentaire |
| `validity_days` | Dates extraites | Durée de validité totale du LC |

#### Groupe 2 — Conformité UCP

| Feature | Source | Description |
|---|---|---|
| `ucp_compliance_score` | UCP 600 Layer | Score de conformité (0-1) |
| `n_ucp_warnings` | UCP 600 Layer | Nombre de violations non-bloquantes |
| `ucp_decision_reject` | UCP 600 Layer | 1 si décision = REJECT |

#### Groupe 3 — Structurel

| Feature | Source | Description |
|---|---|---|
| `n_clauses` | NLP Clauses | Nombre de clauses additionnelles |
| `n_documents_required` | Clauses documentaires | Nombre de documents requis |
| `incoterm_risk` | Incoterm | Risque ordinalisé 1-4 (EXW=1, DDP=4) |
| `transshipment_allowed` | Entités SWIFT | Transbordement autorisé (0/1) |
| `charges_beneficiary` | Entités SWIFT | Frais à charge bénéficiaire (0/1) |

#### Groupe 4 — Risque pays & devise

| Feature | Source | Description |
|---|---|---|
| `country_risk` | Entités + NLP | Risque pays ordinalisé (0-3) |
| `country_risk_level` | Validation externe | Niveau risque (0=CLEAN, 4=CRITICAL) |
| `country_risk_score` | Validation externe | Score brut pays (0-100) |
| `external_country_score` | ExternalValidation | Score OFAC/UE/ONU/GAFI |
| `external_country_level` | ExternalValidation | Niveau risque pays externe |
| `external_currency_risk` | ExternalValidation | Risque devise externe |

#### Groupe 5 — Qualité NLP

| Feature | Source | Description |
|---|---|---|
| `nlp_score` | NLP Service | Score NLP global (0-100) |
| `semantic_score` | Analyse sémantique | Score cohérence sémantique (0-1) |
| `completeness_score` | Analyse sémantique | Score complétude (0-1) |

### LCRiskScorer — Scoring règles métier

Premier scoring de risque déterministe (avant ML) basé sur une pondération des features critiques.

**Sortie :** `risk_label` (LOW / MEDIUM / HIGH) + `risk_score` (0-100)

---

## 10. Couche 7 — Validation externe (Sanctions & AML)

**Composant :** `services/external_validation/engine.py` — `ExternalValidationEngine` (approche B)  
**Objectif :** Vérifier les pays et devises identifiés dans le LC contre les listes de sanctions et les registres de risque internationaux, pour détecter toute exposition AML ou violation d'embargo.

### Sources de données

| Source | Type | Format | Mise à jour |
|---|---|---|---|
| OFAC SDN List | Sanctions individuelles + entités | JSON (cache local) | Toutes les 30 jours |
| OFAC Consolidated | Liste consolidée US | JSON (cache local) | Toutes les 30 jours |
| EU Sanctions | Règlement UE | XML → JSON | Toutes les 30 jours |
| UN Sanctions | Résolutions ONU | XML → JSON | Toutes les 30 jours |
| FATF Grey/Black Lists | Juridictions à risque GAFI | JSON (cache local) | Toutes les 30 jours |
| RestCountries | Référentiel pays | JSON (cache local) | Statique |

### Règle bénéficiaire absent

Si aucun bénéficiaire n'est détecté dans le message SWIFT, la validation externe retourne **automatiquement CRITICAL**, sans exécution des vérifications pays/devise, avec la recommandation `⛔ BLOQUER — Bénéficiaire manquant`.

### Système de cache (`cache_manager.py` + `scheduler.py`)

- Listes téléchargées et stockées localement dans `data/scraping/` au format JSON
- `APScheduler` lancé au démarrage du pipeline — rafraîchissement automatique toutes les `CACHE_MAX_AGE_DAYS` (défaut : 30 jours)
- Fallback transparent vers l'API HTTP si le cache est absent ou expiré
- Lookup local O(1) via index dict en mémoire

### Scoring des pays

```
country_score = Σ score_contribution(pays_i)    pour tous les pays du LC
country_score = min(country_score, 12)          plafonnement AML multi-juridictions
```

Pondération finale :
```
raw_score = country_score × 0.7 + currency_score × 0.3
```

### Niveaux de risque

| Niveau | Score | Signification |
|---|---|---|
| CLEAN | 0 | Aucun risque détecté |
| LOW | 1-3 | Risque faible, surveillance recommandée |
| MEDIUM | 4-7 | Risque modéré, KYC renforcé |
| HIGH | 8-11 | Risque élevé, vérification approfondie |
| CRITICAL | ≥12 | Embargo / sanctions — blocage recommandé |

**Escalade CRITICAL :** Si un sous-module (pays ou devise) est CRITICAL, le niveau global est forcé à CRITICAL indépendamment du score agrégé.

### Sortie

```json
{
  "risk_level": "HIGH",
  "raw_score": 9.4,
  "risk_factors": ["Pays IR — HIGH (OFAC SDN)"],
  "recommendations": ["⛔ BLOQUER — Pays sous sanctions : IR"],
  "sources_consulted": ["OFAC-COUNTRIES", "EU-SANCTIONS", "FATF"],
  "details": {
    "countries": [...],
    "currency": {"code": "USD", "is_valid": true, "risk": "CLEAN"}
  },
  "processing_ms": 143
}
```

---

## 11. Couche 8 — Risk ML Engine

**Composant :** `services/risk_service/risk_engine.py` — `RiskMLEngine` v2.0  
**Objectif :** Produire un score de risque probabiliste en exploitant l'ensemble des features calculées par les couches précédentes, avec explications par valeurs SHAP et segmentation client.

### Architecture du modèle ensemble

```
Features (21 dims)
    │
    ├─ [MinMaxScaler]         → normalisation 0-1
    │
    ├─ [XGBoost Calibré]      → P(LOW), P(MEDIUM), P(HIGH)
    │   └─ CalibratedClassifierCV (Platt scaling)
    │
    ├─ [IsolationForest]      → score anomalie (-1 à +1)
    │
    ├─ [KMeans k=4]           → segment client (cluster 0-3)
    │
    ├─ [SHAP TreeExplainer]   → contribution de chaque feature
    │
    └─ [LightGBM Challenger]  → scoring indépendant pour comparaison
```

### Règles hard — Moteur UCP pré-ML

Avant la prédiction ML, des règles déterministes peuvent forcer le label :

| Condition | Action |
|---|---|
| `ucp_decision_reject == 1` | label = HIGH (bloqué) |
| `country_risk_level ≥ 4` (CRITICAL) | label = HIGH (embargo) |
| `ucp_compliance_score < 0.30` | label = HIGH |
| `presentation_days < 5` | label = HIGH (Art.14 UCP) |

### Modèle XGBoost calibré

- **Type :** `CalibratedClassifierCV` encapsulant `XGBClassifier`
- **Calibration :** Platt scaling — transforme les scores en probabilités bien calibrées
- **Classes :** 0=LOW, 1=MEDIUM, 2=HIGH
- **Fichier :** `services/risk_service/models/xgboost_calibrated.pkl`

### Détection d'anomalies (IsolationForest)

- **Algorithme :** Isolation Forest — isole les observations atypiques via des arbres de partition aléatoire
- **Interprétation :** `decision_function` < 0 → anomalie potentielle
- **Usage :** flaguer les LCs avec un profil de features inhabituel, indépendamment du label de risque
- **Fichier :** `services/risk_service/models/isolation_forest.pkl`

### Segmentation client (KMeans)

- **Algorithme :** KMeans k=4 — clustering non supervisé des profils LC
- **Segments :** 4 groupes nommés (ex. `HIGH_RISK_SENSITIVE`, `STANDARD_TRADE`, `COMPLEX_MULTIPARTY`)
- **Utilité :** contextualisation du risque par type de transaction
- **Fichiers :** `models/kmeans.pkl` + `cluster_names.json`

### Explications SHAP

- **Algorithme :** SHAP (SHapley Additive exPlanations) — TreeExplainer sur XGBoost
- **Sortie :** top 5 features avec valeur SHAP + direction (augmente/diminue le risque)
- **Explainer pré-calculé :** `models/shap_explainer.pkl` (évite le recalcul à chaque inférence)

### Narration automatique (`narrative_config.json`)

Templates de phrases en langage naturel instanciés avec les valeurs réelles :
- "Le score UCP de 34/100 indique des non-conformités significatives."
- "Le pays du bénéficiaire (IR) est classé CRITICAL selon OFAC."
- "Anomalie détectée : profil LC inhabituel pour ce segment."

### Challenger LightGBM

Un modèle LightGBM secondaire est scoré en parallèle. En cas de divergence (ex. XGB=LOW, LGB=MEDIUM), le pipeline applique une **escalade conservatrice** : le niveau le plus élevé est retenu.

### Sortie Risk ML

```json
{
  "risk_label": "HIGH",
  "risk_score": 87,
  "risk_proba": {"LOW": 0.05, "MEDIUM": 0.18, "HIGH": 0.77},
  "is_anomaly": false,
  "anomaly_score": 0.12,
  "segment": "HIGH_RISK_SENSITIVE",
  "segment_id": 2,
  "shap_top_features": [
    {"feature": "country_risk_level", "shap_value": 0.34, "direction": "up"},
    {"feature": "ucp_compliance_score", "shap_value": -0.21, "direction": "down"}
  ],
  "narrative": "Risque élevé — pays bénéficiaire sous sanctions OFAC, conformité UCP insuffisante.",
  "model_version": "2.0",
  "processing_ms": 89
}
```

---

## 12. Modèles IA — Détail technique

### Inventaire des modèles

| Modèle | Emplacement | Type | Tâche |
|---|---|---|---|
| SWIFT NER | `models/swift-ner/checkpoint-2660/` | BERT fine-tuné (HuggingFace) | Token classification IOB2 |
| Clause Classifier | `models/clause-clf/` | SentenceTransformer + LogisticRegression | Multi-class |
| NLI Contradictions | `models/nli/` | Cross-encoder (DeBERTa) | Entailment / Contradiction |
| XGBoost Risk | `services/risk_service/models/xgboost_calibrated.pkl` | XGBoost calibré | Multi-class (LOW/MEDIUM/HIGH) |
| IsolationForest | `services/risk_service/models/isolation_forest.pkl` | Ensemble non supervisé | Détection d'anomalies |
| KMeans | `services/risk_service/models/kmeans.pkl` | Clustering | Segmentation (4 classes) |
| SHAP Explainer | `services/risk_service/models/shap_explainer.pkl` | TreeExplainer | Explicabilité XGBoost |
| MinMaxScaler | `services/risk_service/models/scaler.pkl` | Preprocessing | Normalisation features |
| Risk Classifier (legacy) | `models/risk-clf/risk_clf.pkl` | GradientBoosting + TF-IDF | Scoring risque textuel |

### Modèle NER SWIFT (BERT fine-tuné)

```
Architecture : BERT Encoder → Linear → Softmax (par token)
Base          : bert-base-multilingual-cased
Tâche         : Sequence labeling — IOB2
Labels        : 13 types d'entités LC
Checkpoint    : 2660 steps d'entraînement
Tokenizer     : WordPiece (multilingue)
Inférence     : HuggingFace pipeline("token-classification")
```

### Modèle Clause Classifier

```
Architecture  : [Texte] → SentenceTransformer → [Embedding 384-dim] → LogisticRegression
Backbone      : paraphrase-multilingual-MiniLM-L12-v2
               (12 couches Transformer, optimisé similarité sémantique)
Entraînement  : scripts/train_models.py
Dataset       : Corpus de clauses LC annotées manuellement
Output        : 6 classes + probabilité
```

### Modèle NLI (Cross-Encoder)

```
Architecture  : Cross-encoder — [tag SWIFT | clause] → [ENTAILMENT, NEUTRAL, CONTRADICTION]
Base          : cross-encoder/nli-deberta-v3-xsmall (DeBERTa v3)
Input         : paires (prémisse=tag, hypothèse=clause)
Output        : 3 classes + scores softmax
Avantage      : inférence offline, sans appel API externe
```

### XGBoost Risk ML

```
Architecture  : XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
               → CalibratedClassifierCV(method='sigmoid', cv=5)
Features      : 21 features canoniques (voir section 9)
Preprocessing : MinMaxScaler (fitted sur données d'entraînement)
Classes       : 0=LOW (0-40), 1=MEDIUM (41-70), 2=HIGH (71-100)
Dataset       : Synthétique + weak supervision (risk_dataset_generator.py)
SMOTE         : Rééchantillonnage pour équilibrer les classes
```

---

## 13. Stack technologique

### OCR & Vision

| Bibliothèque | Rôle |
|---|---|
| EasyOCR | Moteur OCR principal (deep learning) |
| Tesseract / pytesseract | Moteur OCR secondaire (LSTM) |
| OpenCV (cv2) | Prétraitement image |
| Pillow | Manipulation d'images |
| pdf2image | Conversion PDF → images |
| PyMuPDF (fitz) | Extraction texte PDF natif |

### NLP & Transformers

| Bibliothèque | Rôle |
|---|---|
| transformers (HuggingFace) | Modèles BERT, DeBERTa (NER, NLI) |
| sentence-transformers | SentenceTransformer (embeddings clausaux) |
| scikit-learn | LogisticRegression, TF-IDF, GradientBoosting |
| RapidFuzz | Fuzzy string matching (résistance OCR) |
| spacy | NLP utilitaire (tokenization, POS) |

### Machine Learning & Risque

| Bibliothèque | Rôle |
|---|---|
| XGBoost | Classificateur de risque principal |
| LightGBM | Modèle challenger |
| SHAP | Explicabilité locale (TreeExplainer) |
| imbalanced-learn | SMOTE — équilibrage des classes |
| joblib | Sérialisation/désérialisation modèles (.pkl) |
| numpy / pandas | Calcul matriciel et manipulation des features |

### API & Infrastructure

| Bibliothèque | Rôle |
|---|---|
| Flask | API REST |
| Streamlit | Interface d'analyse interactive |
| APScheduler | Rafraîchissement automatique des caches |
| requests | Appels HTTP APIs sanctions |
| loguru | Logging structuré |

---

## 14. Structure du projet

```
swift-validator-ia/
│
├── app/
│   ├── api.py                         # API Flask REST (v6.3)
│   ├── streamlit_app.py               # Interface Streamlit
│   └── streamlit_dashboard.py
│
├── pipeline/
│   └── swift_pipeline.py              # Orchestrateur principal v6.6
│
├── services/
│   ├── ocr/                           # Couche OCR
│   │   ├── ocr_service.py
│   │   ├── pdf_detector.py
│   │   ├── pdf_text_extractor.py
│   │   ├── ocr_engines.py
│   │   ├── image_enhancer.py
│   │   ├── postprocessor.py
│   │   └── spell_correction.py
│   │
│   ├── swift/                         # Parsing SWIFT
│   │   ├── swift_service.py
│   │   ├── swift_detector.py
│   │   ├── swift_base_parser.py
│   │   ├── parsers/parser_factory.py
│   │   ├── swift_validator.py
│   │   ├── lc_acceptance_criteria.py
│   │   ├── entities.py
│   │   ├── schema_loader.py
│   │   ├── patterns.py
│   │   └── swift_tag_normalizer.py
│   │
│   ├── swift_nlp/                     # Couche NLP
│   │   ├── swift_nlp_service.py
│   │   ├── ner/swift_ner_extractor.py
│   │   ├── clauses/swift_clause_classifier.py
│   │   ├── semantic/
│   │   │   ├── swift_semantic_analyzer.py
│   │   │   └── swift_nli_contradiction.py
│   │   ├── ucp/
│   │   │   ├── swift_ucp_validator.py
│   │   │   ├── swift_ucp_hard_rules.py
│   │   │   ├── swift_ucp_semantic_engine.py
│   │   │   ├── swift_ucp_nli.py
│   │   │   └── swift_ucp_clause_aligner.py
│   │   ├── ds/swift_ds_pipeline.py
│   │   └── common/
│   │       ├── swift_dataset_loader.py
│   │       └── swift_st_cache.py
│   │
│   ├── external_validation/           # Validation sanctions & AML
│   │   ├── engine.py
│   │   ├── country_checker.py
│   │   ├── cache_manager.py
│   │   ├── scheduler.py
│   │   └── scraper.py
│   │
│   └── risk_service/                  # Risk ML Engine
│       ├── risk_engine.py
│       ├── risk_feature_schema.py
│       ├── models/
│       │   ├── xgboost_calibrated.pkl
│       │   ├── isolation_forest.pkl
│       │   ├── kmeans.pkl
│       │   ├── scaler.pkl
│       │   ├── shap_explainer.pkl
│       │   ├── metadata.json
│       │   ├── narrative_config.json
│       │   └── cluster_names.json
│       └── scripts/
│           ├── train_models.py
│           └── risk_dataset_generator.py
│
├── models/
│   ├── clause-clf/                    # SentenceTransformer + LR
│   ├── risk-clf/                      # GradientBoosting + TF-IDF (legacy)
│   ├── swift-ner/checkpoint-2660/     # BERT NER fine-tuné
│   └── nli/                           # DeBERTa NLI offline
│
├── data/
│   ├── swift_schema/                  # Schémas SWIFT par type MT
│   └── scraping/                      # Cache listes sanctions
│
├── scripts/
│   ├── train_models.py
│   ├── fine_tune_ner.py
│   ├── evaluate_models.py
│   └── setup_nli_model.py
│
├── tests/
└── requirement.txt
```

---

## 15. API REST

**Base URL :** `http://localhost:5000`

### Santé & Diagnostic

```
GET  /              → Info API, version, routes disponibles
GET  /health        → Santé globale (modules, modèles ML, APIs)
GET  /risk/health   → Statut Risk ML (modèles, features, SHAP)
```

### Pipeline complet

```
POST /pipeline
  Content-Type: multipart/form-data
  Body: file (PDF / image)
```

### Analyse texte SWIFT (sans OCR)

```
POST /swift/analyze
  Content-Type: application/json
  Body: { "text": ":40A:IRREVOCABLE\n:20:LC123456\n..." }
```

### NLP seul

```
POST /swift/nlp
  Body: { "text": "...", "swift_type": "MT700" }
```

### OCR seul

```
POST /ocr
  Body: file (multipart)
```

### Scoring de risque

```
POST /risk/score          → depuis payload pipeline complet
POST /risk/score/raw      → depuis features brutes (dict)
```

---

## 16. Installation & Démarrage

### Prérequis système

- Python 3.10+
- Tesseract OCR (`apt install tesseract-ocr tesseract-ocr-fra`)
- Poppler (`apt install poppler-utils`)
- GPU CUDA (optionnel — accélère les modèles Transformer)

### Installation

```bash
git clone <repo-url>
cd swift-validator-ia

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

pip install -r requirement.txt
```

### Démarrage

```bash
# API Flask
export FLASK_APP=app/api.py
flask run --port 5000

# Interface Streamlit
streamlit run app/streamlit_app.py
```

### Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `SANCTIONS_DATA_DIR` | `data/scraping` | Répertoire cache des listes de sanctions |
| `CACHE_MAX_AGE_DAYS` | `30` | Durée de validité du cache (jours) |
| `OPENCORPORATES_API_KEY` | (vide) | Clé API OpenCorporates (optionnel) |
| `FLASK_DEBUG` | `false` | Mode debug Flask |

---

## 17. Entraînement des modèles

### Classificateur de clauses

```bash
python scripts/train_models.py --dataset-dir data/data_stati --output-dir models
```

### Risk ML (XGBoost + ensemble)

```bash
python services/risk_service/scripts/risk_dataset_generator.py
python services/risk_service/scripts/train_models.py
```

### Fine-tuning NER SWIFT

```bash
python scripts/fine_tune_ner.py \
  --model bert-base-multilingual-cased \
  --dataset path/to/ner_data.json
```

### Setup modèle NLI

```bash
python scripts/setup_nli_model.py \
  --model cross-encoder/nli-deberta-v3-xsmall \
  --out models/nli
```

### Tests

```bash
pytest tests/ -v
```

---

## 18. Format de sortie

```json
{
  "success": true,
  "ocr": {
    "text": "...",
    "confidence": 0.94,
    "spell_report": { "corrections_applied": 7, "changes": [...] }
  },
  "swift": {
    "swift_type": "MT700",
    "entities": {
      "lc_number": "LC-2024-001",
      "applicant": "SOCIÉTÉ IMPORT SA",
      "beneficiary": "EXPORT CORP LTD",
      "amount": 500000.0,
      "currency": "USD",
      "expiry_date": "2024-12-31",
      "incoterm": "CIF"
    },
    "validation": { "errors": [], "warnings": [...] },
    "nlp": {
      "ner": { "entities": [...] },
      "clauses": [...],
      "semantic": { "routing_decision": "REVIEW" },
      "contradictions": [...]
    }
  },
  "ucp_compliance": {
    "global_score": 67,
    "ucp_decision": "REVIEW",
    "payment_blocked": false,
    "findings": [...]
  },
  "ds_analysis": {
    "risk": { "risk_label": "MEDIUM", "risk_score": 54 },
    "features": { "lc_amount": 500000, "ucp_compliance_score": 0.67 }
  },
  "external_validation": {
    "risk_level": "LOW",
    "raw_score": 1.4,
    "sources_consulted": ["OFAC-COUNTRIES", "EU-SANCTIONS"]
  },
  "risk_ml": {
    "risk_label": "MEDIUM",
    "risk_score": 58,
    "risk_proba": { "LOW": 0.22, "MEDIUM": 0.63, "HIGH": 0.15 },
    "is_anomaly": false,
    "segment": "STANDARD_TRADE",
    "shap_top_features": [...],
    "narrative": "Risque modéré — conformité UCP partielle, pays bénéficiaire à risque faible."
  },
  "contract_version": "1.0"
}
```

---

## 19. Performance

### Temps de traitement typiques

| Module | Temps (ms) |
|---|---|
| OCR (PDF 1 page) | 2 000 – 5 000 |
| Correction orthographique | 50 – 150 |
| Détection & Parsing SWIFT | 100 – 200 |
| NLP complet (NER + clauses + sémantique) | 800 – 1 500 |
| Validation UCP 600 | 200 – 400 |
| Validation externe (cache chaud) | 50 – 200 |
| Validation externe (API HTTP) | 500 – 2 000 |
| Risk ML (XGBoost + SHAP) | 50 – 150 |
| **Pipeline end-to-end** | **3 000 – 8 000** |

### Optimisations

- **Chargement lazy** — NLP, validation externe et Risk ML chargés à la première requête
- **Cache en mémoire** — listes de sanctions en dict (lookup O(1))
- **Parallélisme NLP** — NER, classification et analyse sémantique en `ThreadPoolExecutor`
- **SHAP Explainer pré-calculé** — évite de reconstruire le TreeExplainer à chaque inférence
- **GPU optionnel** — les modèles Transformer bénéficient de CUDA si disponible

---

## Licence

Propriétaire — Usage interne.
