# SwiftValidator AI — Documentation complète

Plateforme bancaire d'intelligence artificielle dédiée à la validation automatique des crédits documentaires SWIFT MT700, avec conformité UCP 600 / ISBP 745.

---

## Sommaire

1. [Architecture globale](#architecture-globale)
2. [Prérequis & démarrage](#prérequis--démarrage)
3. [Pages publiques](#pages-publiques)
4. [Authentification & sécurité](#authentification--sécurité)
5. [Espace Agent — Analyse MT700](#espace-agent--analyse-mt700)
6. [Espace Agent — Documents commerciaux](#espace-agent--documents-commerciaux)
7. [Espace Administrateur](#espace-administrateur)
8. [Pipeline IA (Python)](#pipeline-ia-python)
9. [API REST](#api-rest)

---


## Architecture globale

```
┌─────────────────────────────────────────────────────┐
│                Navigateur (port 4200)                │
│           Angular 21 · PrimeNG · Chart.js            │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP + Cookie JWT (proxy /api)
┌───────────────────────▼─────────────────────────────┐
│           Spring Boot 3.2.4  (port 8080)             │
│  Spring Security · JWT · JPA · PostgreSQL · MinIO    │
└───────────┬───────────────────────┬─────────────────┘
            │ JPA/SQL               │ REST HTTP
┌───────────▼──────────┐  ┌────────▼────────────────┐
│   PostgreSQL          │  │  Pipeline IA Python      │
│  Utilisateurs, LC,    │  │  NLP · UCP · Géo · SHAP  │
│  Analyses, Docs       │  │  Flask · port 5000       │
└──────────────────────┘  └─────────────────────────┘
```

| Couche          | Technologie                                                   |
|-----------------|---------------------------------------------------------------|
| Frontend        | Angular 21, PrimeNG 21, Chart.js 4, zone.js                   |
| Backend         | Spring Boot 3.2.4, Java 17, Spring Security 6, Lombok         |
| Auth            | JWT (jjwt 0.11.5), BCrypt, cookies HttpOnly / SameSite=Lax    |
| Base de données | PostgreSQL 14+, Spring Data JPA, Liquibase                    |
| Stockage        | MinIO (documents uploadés, optionnel)                         |
| IA Pipeline     | Python 3.10+, Flask, transformers, XGBoost, SHAP, EasyOCR    |

---

## Prérequis & démarrage

**Prérequis** : Java 17+, Node.js 20+, Python 3.10+, PostgreSQL 14+, Tesseract OCR

### 1 — Pipeline IA (Python)

```bash
cd swift-validator-ia
python -m venv .venv
.venv\Scripts\activate.bat      # Windows
pip install -r requirements.txt
python app/api.py
# Démarre sur http://localhost:5000
```

### 2 — Backend (Spring Boot)

```bash
cd swift-validator-backend
mvn spring-boot:run
# Démarre sur http://localhost:8080
```

Variables d'environnement clés :

```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/swiftvalidator
spring.datasource.username=...
spring.datasource.password=...

jwt.secret=<clé-256-bits-minimum>
jwt.expiration-ms=86400000          # 24 heures
jwt.cookie-name=swift_auth
jwt.cookie-secure=false             # true en production (HTTPS)
jwt.cookie-same-site=Lax

cors.allowed-origins=http://localhost:4200

PYTHON_PIPELINE_URL=http://localhost:5000
MINIO_ENABLED=false
```

Au démarrage, `DataInitializer` crée automatiquement un compte administrateur :

| Champ    | Valeur       |
|----------|--------------|
| Username | `admin`      |
| Password | `Admin@1234` |
| Rôle     | `ADMIN`      |

### 3 — Frontend (Angular 21)

```bash
cd swift_validator_frontend
npm install
ng serve
# Démarre sur http://localhost:4200
```

Le fichier `proxy.conf.json` route toutes les requêtes `/api/*` vers `localhost:8080`, évitant les problèmes de cookies cross-port en développement :

```json
{
  "/api": {
    "target": "http://localhost:8080",
    "secure": false,
    "changeOrigin": true
  }
}
```

---

## Pages publiques

Accessibles sans authentification via le `PublicShellComponent` (header + footer communs).

### Accueil (`/home`)

| Section               | Contenu                                                                            |
|-----------------------|------------------------------------------------------------------------------------|
| **Hero**              | Slogan, carte d'exemple d'analyse MT700 animée (score, décision, barres de score) |
| **Statistiques**      | Compteurs : analyses effectuées, taux de précision, conformité UCP                |
| **Fonctionnalités**   | Grille 6 cartes : OCR SWIFT, NLP sémantique, UCP 600, scoring, géo, SHAP          |
| **Comment ça marche** | 3 étapes : Chargement → Analyse multicritère → Décision documentée                |
| **CTA**               | Bouton « Commencer l'analyse » → `/login`                                          |

### Services (`/services`)

Présentation détaillée de chaque service de la plateforme :

- Validation UCP 600 & ISBP 745 (articles, sévérité, recommandations)
- Analyse NLP (détection de clauses, contradictions sémantiques)
- Validation géographique (pays, sanctions OFAC/EU, listes FATF)
- Scoring bayésien & explicabilité SHAP (top features)
- Validation croisée LC / documents commerciaux

Inclut une **FAQ accordéon** sur l'utilisation de la plateforme.

### À propos (`/about`)

| Section          | Contenu                                                           |
|------------------|-------------------------------------------------------------------|
| **Mission**      | Automatiser la conformité MT700 — libérer les agents des contrôles standardisés |
| **Chiffres**     | 10+ axes de contrôle · UCP 600 · Analyse < 2s · 3 décisions      |
| **Architecture** | Diagramme de flux : Document → Pipeline IA → API Spring Boot → Interface Angular |
| **Technologies** | Grille de toutes les technologies utilisées par couche            |
| **Organisation** | Cartes de présentation de l'équipe projet                        |

---

## Authentification & sécurité

### Flux de connexion

```
POST /api/v1/auth/login  →  { username, password }

1. Spring Security → DaoAuthenticationProvider
2. BCryptPasswordEncoder vérifie le hash en base
3. JwtTokenProvider génère le token signé (HS256)
4. Token placé dans un cookie HttpOnly (invisible à JavaScript)
5. Réponse JSON : { userId, username, role, expiresIn }

Set-Cookie: swift_auth=<token>; HttpOnly; SameSite=Lax; Path=/; Max-Age=86400
```

### Modèle utilisateur (entité JPA)

```java
@Entity @Table(name = "users")
public class User {
    Long id;
    String username;      // unique
    String email;         // unique
    String passwordHash;  // BCrypt $2a$10$...
    String role;          // "ADMIN" | "AGENT"  (défaut : AGENT)
    Boolean enabled;      // activer / désactiver le compte
    LocalDateTime createdAt;
    List<SwiftAnalysis> analyses; // OneToMany lazy
}
```

### RBAC — Contrôle d'accès par rôle

| Rôle    | Routes accessibles                | Comportement à la connexion         |
|---------|------------------------------------|-------------------------------------|
| `AGENT` | `/app`, `/commercial`              | Redirigé vers `/app`               |
| `ADMIN` | `/admin/**`                        | Redirigé vers `/admin/dashboard`   |

Un ADMIN qui accède à `/app` est automatiquement redirigé vers `/admin/dashboard`.  
Un AGENT qui accède à `/admin` est redirigé vers `/app`.

#### Guards Angular (`CanActivateFn`)

```typescript
// adminGuard — protège /admin/**
auth.me().pipe(
  map(user => user.role === 'ADMIN' ? true : router.createUrlTree(['/app'])),
  catchError(() => of(router.createUrlTree(['/login'])))
)

// agentGuard — protège /app et /commercial
auth.me().pipe(
  map(user => user.role === 'ADMIN'
    ? router.createUrlTree(['/admin/dashboard'])
    : true),
  catchError(() => of(router.createUrlTree(['/login'])))
)
```

#### Règles Spring Security (`SecurityConfig`)

```java
.requestMatchers("/api/v1/auth/**").permitAll()        // public
.requestMatchers("/api/v1/admin/**").hasRole("ADMIN")  // ADMIN seulement
.requestMatchers("/api/v1/**").authenticated()          // tout le reste
```

Les endpoints admin sont doublement protégés : filtre JWT **ET** `@PreAuthorize("hasRole('ADMIN')")` sur le contrôleur.

### JWT — Détails techniques

| Paramètre       | Valeur                          |
|-----------------|---------------------------------|
| Algorithme      | HS256                           |
| Bibliothèque    | jjwt 0.11.5                     |
| Durée de vie    | 24h (configurable)              |
| Transport       | Cookie HttpOnly (pas en header) |
| Nom du cookie   | `swift_auth`                    |
| SameSite        | `Lax`                           |
| Secure          | `false` dev / `true` prod       |

Le filtre `JwtAuthFilter` intercepte chaque requête, extrait et valide le cookie `swift_auth`, puis injecte l'utilisateur dans le `SecurityContext`.

### Hashage des mots de passe

```java
@Bean
public PasswordEncoder passwordEncoder() {
    return new BCryptPasswordEncoder(); // facteur de coût 10, salt automatique
}
```

BCrypt génère un salt aléatoire à chaque hash — deux mots de passe identiques produisent des hashs différents. Aucun mot de passe n'est jamais stocké en clair.

### Déconnexion

```
POST /api/v1/auth/logout
→ Serveur retourne Set-Cookie: swift_auth=; Max-Age=0  (cookie effacé)
→ Angular redirige vers /login
```

---

## Espace Agent — Analyse MT700

**Route** : `/app` · **Composant** : `AppComponent`  
Protégé par `agentGuard` (rôle `AGENT` requis).

### Vue d'ensemble

L'agent uploade un fichier MT700 (PDF natif, PDF scanné, image). Le système :
1. Envoie le fichier au backend Spring Boot
2. Le backend appelle le pipeline Python
3. Le pipeline effectue OCR → parsing SWIFT → NLP → UCP → risque ML
4. Les résultats sont stockés en PostgreSQL et retournés au frontend

### Champs SWIFT extraits (tags MT700)

| Tag   | Champ                  | Tag   | Champ                  |
|-------|------------------------|-------|------------------------|
| 20    | Numéro LC              | 44E   | Port of loading        |
| 27    | Sequence               | 44F   | Port of discharge      |
| 31C   | Date d'émission        | 44C   | Date d'expédition      |
| 31D   | Date d'expiration      | 48    | Presentation period    |
| 32B   | Montant & devise       | 49    | Confirmation           |
| 39A   | Tolérance              | 45A   | Description marchandises |
| 40A   | Forme du crédit        | 46A   | Documents requis       |
| 40E   | Règles applicables     | 71B   | Charges                |
| 41D   | Disponible avec/par    | 72    | Info émetteur          |
| 50    | Applicant              | 52A   | Banque émettrice       |
| 59    | Beneficiary            | 77U   | Texte garantie         |

### Panneaux d'analyse

Après l'upload et l'analyse, les résultats sont présentés dans une interface à onglets :

| Panneau              | Contenu                                                                         |
|----------------------|---------------------------------------------------------------------------------|
| **Overview**         | Décision globale, score composite, badges KPI, résumé des axes                 |
| **Champs SWIFT**     | Tous les tags extraits : valeur, statut (OK/WARN/ERROR), confiance NER, erreurs |
| **UCP 600**          | Violations et avertissements UCP : article, titre, sévérité, recommandation    |
| **Clauses NLP**      | Clauses détectées : type, niveau de risque, méthode, références UCP            |
| **Contradictions**   | Contradictions NLP et sémantiques entre clauses, sévérité, clauseA / clauseB  |
| **Géographie**       | Validation pays (Applicant, Beneficiary, ports) : sanctions OFAC, FATF, risque |
| **ML / Risk**        | Score de risque bayésien, niveau (LOW/MEDIUM/HIGH/CRITICAL), confiance ML      |
| **SHAP / XAI**       | Contribution de chaque feature au score — top 3 explications narratives        |
| **Performance**      | Temps de traitement par module (ms) sous forme de barres                       |

### Décisions possibles

| Décision | Signification                                               |
|----------|--------------------------------------------------------------|
| `ACCEPT` | Crédit documentaire conforme — paiement recommandé          |
| `REVIEW` | Anomalies détectées — révision humaine requise              |
| `REJECT` | Non-conformité majeure — paiement bloqué                    |

### Historique personnel

- Les 20 dernières analyses de l'agent connecté sont accessibles dans un panneau latéral
- Cliquer sur une entrée recharge l'analyse complète
- Si le payload brut n'est pas en cache local, une requête `GET /api/v1/analysis/{id}` est lancée
- Timer de sécurité de 15s pour éviter tout blocage du panneau historique

### Statistiques de l'agent

Compteurs visibles dans le header de l'espace agent :
- Nombre total d'analyses effectuées
- Répartition ACCEPT / REVIEW / REJECT
- Score moyen

---

## Espace Agent — Documents commerciaux

**Route** : `/commercial` · **Composant** : `CommercialDocsComponent`  
Protégé par `agentGuard`.

### Upload de documents

| Paramètre        | Valeur                                              |
|------------------|-----------------------------------------------------|
| Fichiers max     | 3 simultanés                                       |
| Formats acceptés | `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`   |
| Référence LC     | Numéro du crédit documentaire (obligatoire)         |
| Types documents  | Bill of Lading · Invoice · Packing List             |

L'interface supporte le **glisser-déposer** (drag & drop) et la sélection manuelle.

### Résultats par document

Pour chaque document uploadé, 3 onglets :

| Onglet      | Contenu                                              |
|-------------|------------------------------------------------------|
| **Champs**  | Champs extraits par OCR/parsing avec valeurs         |
| **Tables**  | Données tabulaires (quantités, montants, ports...)   |
| **Règles**  | Résultats de validation par règle métier             |

Niveaux de risque : `low` (vert) · `medium` (orange) · `high` (rouge) · `critical` (rouge foncé)

### Validation croisée LC (`LcCrossValidationComponent`)

Composant dédié à la confrontation des documents commerciaux avec le crédit documentaire MT700 source :

| Onglet         | Contenu                                                                      |
|----------------|------------------------------------------------------------------------------|
| **Règles**     | Contrôles de cohérence LC ↔ documents (conformité, délais, montants)        |
| **Sémantique** | Scores NLP de similarité entre clauses LC et texte des documents             |
| **NLI**        | Natural Language Inference — entailment / contradiction / neutral            |
| **Tables**     | Comparaison tabulaire des champs clés (montant, date, port...)               |
| **XAI**        | Explication des scores de validation croisée (contribution par feature)     |

---

## Espace Administrateur

**Route** : `/admin` · **Shell** : `AdminShellComponent`  
Protégé par `adminGuard` (rôle `ADMIN` uniquement).

### Layout de l'espace admin

Sidebar verticale collapsible avec navigation, avatar et déconnexion :

```
╔══════════════════════════════╗
║  ✦ SwiftValidatorAI    [◀]  ║
╠══════════════════════════════╣
║  📊 Tableau de bord          ║
║  👥 Gestion agents           ║
║  📋 Historique               ║
╠══════════════════════════════╣
║  [AD] admin                  ║
║       Administrateur         ║
║  [⬡ Déconnexion]             ║
╚══════════════════════════════╝
```

---

### Tableau de bord (`/admin/dashboard`)

Vue d'ensemble en temps réel de l'activité de la plateforme.  
Données chargées depuis `GET /api/v1/admin/stats`.

#### 4 cartes KPI

| Carte                | Indicateur                                          |
|----------------------|-----------------------------------------------------|
| **Agents actifs**    | Nombre total d'agents enregistrés                  |
| **Analyses totales** | Cumul de toutes les analyses + delta du jour        |
| **Score moyen**      | Score composite moyen sur 100                       |
| **Taux d'acceptation** | Pourcentage ACCEPT sur total                     |

#### Graphiques (Chart.js 4)

| Graphique                  | Type    | Données                                          |
|----------------------------|---------|--------------------------------------------------|
| **Tendance hebdomadaire**  | Barres  | Nombre d'analyses par jour sur les 7 derniers jours |
| **Répartition décisions**  | Donut   | Proportion ACCEPT / REVIEW / REJECT              |

#### Détail des décisions

3 cartes avec : valeur absolue · barre de progression · pourcentage
- ACCEPTÉ (vert)
- RÉVISION (orange)
- REJETÉ (rouge)

**États gérés** :
- `loading` : squelettes PrimeNG (4 cartes + 2 graphiques)
- `error` : message d'erreur + bouton « Réessayer »
- Données affichées après chargement

---

### Gestion des agents (`/admin/agents`)

Interface CRUD complète pour les comptes agents.

#### Tableau des agents

| Colonne           | Détail                                              |
|-------------------|-----------------------------------------------------|
| Nom d'utilisateur | Identifiant de connexion (unique)                  |
| Email             | Adresse email (unique)                             |
| Rôle              | Badge `ADMIN` (orange) ou `AGENT` (bleu)           |
| Statut            | Toggle actif / désactivé (bascule instantanée)     |
| Analyses          | Compteur d'analyses effectuées                     |
| Créé le           | Date de création du compte                         |
| Actions           | Boutons Modifier · Supprimer                       |

Filtre texte global sur l'ensemble du tableau (recherche en temps réel).

#### Création d'un agent (Dialog)

`POST /api/v1/admin/users`

| Champ             | Règle                                          |
|-------------------|------------------------------------------------|
| Nom d'utilisateur | Obligatoire, unique en base                   |
| Email             | Obligatoire, unique en base                   |
| Mot de passe      | Obligatoire, minimum 8 caractères             |
| Rôle              | Sélecteur : `Agent Bancaire` ou `Administrateur` |

Le mot de passe est hashé en BCrypt côté backend avant persistance — jamais stocké en clair.

#### Modification d'un agent (Dialog)

`PUT /api/v1/admin/users/{id}`

Mêmes champs sauf le mot de passe (non requis en modification).

#### Activation / désactivation

`PATCH /api/v1/admin/users/{id}/status`  · Body : `{ "enabled": true/false }`

Toggle visible directement dans le tableau — changement sans rechargement de page.

#### Suppression

`DELETE /api/v1/admin/users/{id}`

Confirmation obligatoire via dialog PrimeNG (`ConfirmDialog`) avant la suppression définitive.

---

### Historique des analyses (`/admin/history`)

Consultation de l'activité de chaque agent.

#### Panneau gauche — Sélection d'agent

Liste de tous les agents enregistrés.  
Un clic charge l'historique de l'agent via `GET /api/v1/admin/users/{id}/analyses`.

#### Panneau droit — Tableau des analyses

| Colonne           | Détail                                              |
|-------------------|-----------------------------------------------------|
| Référence LC      | Numéro du crédit documentaire analysé              |
| Décision          | Badge coloré : ACCEPTÉ / RÉVISION / REJETÉ          |
| Score             | Score composite sur 100                            |
| Date              | Horodatage de l'analyse                            |
| Avertissements    | Nombre d'avertissements détectés                  |
| Contradictions    | Nombre de contradictions détectées                |

**Filtre décision** : Tous · ACCEPTÉ · RÉVISION · REJETÉ

**Résumé en bas de panneau** :
- Compteur par décision (ACCEPT / REVIEW / REJECT)
- Score moyen de l'agent sélectionné

---

## Pipeline IA (Python)

Le pipeline s'exécute sur port 5000 (Flask) et est appelé par le backend Spring Boot pour chaque analyse.

### Étapes du pipeline (SwiftPipeline v6.6)

```
Upload PDF/Image
      │
      ▼
OCR ─────────── PyMuPDF (texte natif)
                EasyOCR (PDF scanné / images)
                Tesseract (fallback)
      │
      ▼
Correction orthographique ── Dictionnaire SWIFT + bigrammes métier
      │
      ▼
Détection MT ─── Schémas JSON (MT700 / MT720 / MT760 / MT79x)
      │
      ▼
Parsing SWIFT ── Extraction tags :20: :32B: :39A: :41A/D: :44A/B/C/E/F:
                 :45A: :46A: :47A: :71B: :48: :49:
      │
      ▼
NLP v2 ──────── NER (CamemBERT/DistilBERT finetuné + regex/dico fallback)
                Clause classifier (TF-IDF+GBT ou SentenceTransformer+LogReg)
                Analyse sémantique (embeddings, détection contradictions)
      │
      ▼
Fusion NER-first ── Confiance ≥ 0.80 → remplace  |  ≥ 0.70 → ajoute
      │
      ▼
Validation SWIFT ── Champs obligatoires, cohérence devise/montant
      │
      ▼
UCP 600 ─────── Similarité clauses / articles UCP (14, 18, 20, 28...)
                Seuil 0.75 → rattachement; incohérences signalées
      │
      ▼
Validation externe ─ Scraping OFAC SDN · EU consolidated · FATF · ISO 4217
                     Score pays : CLEAN → CRITICAL (peut forcer override ML)
      │
      ▼
Risk ML + SHAP ─── XGBoost calibré (principal)
                   LightGBM (challenger)
                   Isolation Forest (anomalies)
                   KMeans (clusters explicabilité)
                   Explications SHAP top-3 → narrative_config.json
      │
      ▼
JSON résultat → Spring Boot → PostgreSQL → Frontend
```

### Clause classifier (champs 46A / 47A)

Catégories supportées : `INVOICE` · `BILL_OF_LADING` · `INSURANCE` · `PACKING_LIST` · `CERTIFICATE_OF_ORIGIN` · `INSPECTION_CERTIFICATE` · `SHIPPING_ADVICE` · `PAYMENT_TERMS` · `PRESENTATION_PERIOD` · `CONFIRMATION` · `CHARGES` · `RISK_INDICATOR`

### Validation externe (ETL autonome)

Scrapers APScheduler (rafraîchissement tous les 30 jours) :
- **OFAC SDN** — liste des sanctions américaines
- **EU Consolidated** — liste des sanctions européennes
- **FATF** — pays à risque de blanchiment
- **Indices risque pays** + ISO 4217 (devises sensibles)
- Cache `data/scraping/` avec TTL et fallback offline

---

## API REST

### Authentification

| Méthode | Endpoint              | Accès  | Description                           |
|---------|-----------------------|--------|---------------------------------------|
| POST    | `/api/v1/auth/login`  | Public | Connexion → cookie JWT                |
| POST    | `/api/v1/auth/logout` | Public | Suppression du cookie JWT             |
| GET     | `/api/v1/auth/me`     | Auth   | Informations de l'utilisateur courant |

### Administration (ADMIN uniquement)

| Méthode | Endpoint                             | Description                      |
|---------|--------------------------------------|----------------------------------|
| GET     | `/api/v1/admin/stats`                | KPIs du dashboard                |
| GET     | `/api/v1/admin/users`                | Liste de tous les agents         |
| POST    | `/api/v1/admin/users`                | Créer un agent                   |
| PUT     | `/api/v1/admin/users/{id}`           | Modifier un agent                |
| PATCH   | `/api/v1/admin/users/{id}/status`    | Activer / désactiver             |
| DELETE  | `/api/v1/admin/users/{id}`           | Supprimer un agent               |
| GET     | `/api/v1/admin/users/{id}/analyses`  | Historique d'analyses d'un agent |

### Analyse MT700

| Méthode | Endpoint                   | Description                       |
|---------|----------------------------|-----------------------------------|
| POST    | `/api/v1/analysis/upload`  | Uploader et analyser un MT700     |
| GET     | `/api/v1/analysis`         | Liste paginée des analyses        |
| GET     | `/api/v1/analysis/{id}`    | Détail complet d'une analyse      |
| GET     | `/api/v1/dashboard/stats`  | Statistiques de l'agent connecté  |

### Documents commerciaux

| Méthode | Endpoint                      | Description                        |
|---------|-------------------------------|------------------------------------|
| POST    | `/api/v1/commercial/validate` | Valider des documents commerciaux  |
| POST    | `/api/v1/lc-cross/validate`   | Validation croisée LC ↔ documents  |

---

## Notes de sécurité

- Mots de passe hashés avec **BCrypt** (salt aléatoire automatique, facteur de coût 10)
- Sessions **stateless** — aucune session HTTP côté serveur (`SessionCreationPolicy.STATELESS`)
- Cookie JWT **HttpOnly** : inaccessible depuis JavaScript, protège contre le vol XSS
- **CSRF désactivé** — protection assurée par `SameSite=Lax` + absence de session serveur
- Endpoints `/api/v1/admin/**` doublement protégés : filtre JWT **ET** `@PreAuthorize("hasRole('ADMIN')")`
- Le proxy Angular (`proxy.conf.json`) évite les problèmes de cookies cross-origin en développement
- En production : activer `jwt.cookie-secure=true` (HTTPS obligatoire) et `cors.allowed-origins` restreint

---

## Gestion Git des modèles IA

Git LFS est utilisé pour versionner les poids de modèles (`.safetensors`, `.pt`, `.pth`, `.pkl`, `.onnx` et `.bin`). Installez Git LFS avant de cloner le dépôt :

```bash
git lfs install
git clone <URL-du-depot>
```

Les environnements virtuels, caches, journaux d'entraînement, checkpoints, états d'optimiseur et jeux de données générés de `release2` sont exclus du dépôt. Les poids finaux suivis par Git LFS restent disponibles pour l'inférence.

---

## Licence

Usage interne projet SwiftValidator AI.
