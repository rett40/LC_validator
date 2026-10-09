# SWIFT Validator — Intégration de la plateforme

Documentation de la **partie intégration** : comment les deux pipelines Python (IA)
sont branchés au backend Spring Boot, exposés via le frontend Angular, et
orchestrés ensemble grâce à Docker. Le détail métier des modèles IA reste dans le
[README principal](README.md) ; ce document couvre uniquement l'**assemblage des
briques** (backend / frontend / dockerisation, login, et le pont Python ↔ Spring ↔ Angular).

---

## 1. Vue d'ensemble

La plateforme assemble **4 services applicatifs** + 2 services d'infrastructure,
le tout derrière un reverse-proxy Nginx unique (port `80`) :

```
                         ┌──────────────────────────────────────────┐
   Navigateur  ──:80──►  │                 NGINX                      │
                         │  /            → frontend (Angular SPA)     │
                         │  /api/        → api (Spring Boot)          │
                         │  /api/reports/download/ → api (streaming)  │
                         │  /actuator/   → api (réseau interne)       │
                         └───────┬───────────────────┬───────────────┘
                                 │                   │
                    ┌────────────▼──────────┐   ┌────▼─────────────────────┐
                    │  Angular (Nginx :80)  │   │  Spring Boot API (:8080) │
                    │  SPA — pages + écrans │   │  Auth JWT, REST, MinIO   │
                    └───────────────────────┘   └────┬──────────────┬──────┘
                                                      │              │
                                       ┌──────────────▼───┐   ┌──────▼─────────┐
                                       │  Python AI (:5000)│   │ PostgreSQL :5432│
                                       │  Flask — 2 pipes  │   │  MinIO    :9000 │
                                       └───────────────────┘   └────────────────┘
```

| Service        | Techno                | Rôle                                                   | Port interne |
|----------------|-----------------------|--------------------------------------------------------|--------------|
| `nginx`        | Nginx 1.25            | Reverse-proxy / point d'entrée unique                  | 80           |
| `frontend`     | Angular + Nginx       | SPA (pages publiques + espace agent + espace admin)    | 80           |
| `api`          | Spring Boot 3 (Java)  | Auth JWT, orchestration, persistance, stockage         | 8080         |
| `ai-service`   | Python 3.11 / Flask   | Les **2 pipelines IA** + génération PDF                | 5000         |
| `postgres`     | PostgreSQL 16         | Base de données relationnelle                          | 5432         |
| `minio`        | MinIO                 | Stockage objet (documents + rapports PDF)              | 9000 / 9001  |

---

## 2. Les deux pipelines Python intégrés

Le service Flask (`swift-validator-ia/app/api.py`) expose deux chaînes de
traitement distinctes, toutes deux consommées par le backend Spring Boot via
`PipelineClientService.java`.

### Pipeline 1 — Analyse SWIFT / LC (MT700)
- **Endpoint Python** : `POST /pipeline` (upload de fichier)
- **Appelé par** : `SwiftAnalysisController` → `PipelineClientService.analyze(...)`
- **Sortie** : contrat JSON versionné (`contract_version` `1.x`) contenant
  `ocr`, `swift.entities`, `ucp_compliance`, `ds_analysis`, `external_validation`.
- **Validation de contrat** : le backend rejette toute réponse non conforme
  (champs obligatoires manquants ou version incompatible) — voir
  `PipelineClientService.validateContract(...)`.

### Pipeline 2 — Validation documents commerciaux + cross-validation LC
- **Endpoints Python** :
  - `POST /commercial/validate` (upload multi-fichiers) →
    `PipelineClientService.analyzeCommercialDocuments(...)`
  - `POST /lc-commercial/validate` (JSON) →
    `PipelineClientService.callLcCommercialValidate(...)`
- **Appelés par** : `CommercialValidationController`, `LcCrossValidationController`
- **Sortie** : JSON `{ success, ... }` ; le backend lève une erreur si `success=false`.

> Les modèles ML (LayoutLM, Donut, XGBoost/SHAP, NLP, NER…) sont **montés en
> volume** dans le conteneur (`docker-compose.yml`), jamais embarqués dans l'image —
> d'où un `start_period` de **420s** au healthcheck pour le warmup complet.

---

## 3. L'intégration Python ↔ Spring Boot ↔ Angular

Le flux nominal d'une analyse, de bout en bout :

```
1. Angular         L'agent uploade un document (commercial.service.ts / analysis.service.ts)
                   → POST /api/v1/... (cookie JWT joint automatiquement)
                          │
2. Nginx           /api/ → proxy_pass vers api:8080
                          │
3. Spring Boot     JwtAuthFilter valide le cookie → contrôleur → service
                   → DocumentStorageService stocke l'original dans MinIO
                   → PipelineClientService relaie le fichier au service Python
                          │  POST http://ai-service:5000/pipeline (multipart)
4. Python/Flask    Exécute le pipeline IA → renvoie le contrat JSON
                          │
5. Spring Boot     valideContract() → AnalysisMapper → persistance PostgreSQL
                   → (option) ReportService génère/stocke le PDF dans MinIO
                   → renvoie un DTO JSON au frontend
                          │
6. Angular         Affiche les résultats ; le PDF se télécharge via
                   /api/reports/download/... (streaming Nginx sans buffer)
```

### Câblage côté backend (configuration)
| Variable                | Valeur (Docker)              | Fichier                         |
|-------------------------|------------------------------|---------------------------------|
| `pipeline.base-url`     | `http://ai-service:5000`     | `application.yml` / `PYTHON_PIPELINE_URL` |
| `MINIO_ENDPOINT`        | `http://minio:9000`          | `docker-compose.yml`            |
| `MINIO_PUBLIC_ENDPOINT` | `http://localhost:9000`      | URLs présignées vues du navigateur |
| `DB_URL`                | `jdbc:postgresql://postgres:5432/...` | `docker-compose.yml`   |
| `cors.allowed-origins`  | origine du frontend          | `SecurityConfig.java`           |

Le client HTTP utilisé est un `RestTemplate` (multipart pour les uploads, JSON
pour la cross-validation) ; chaque appel logue la latence et l'état HTTP.

### Câblage côté frontend
- `environment.apiUrl` est **vide** (`''`) → les requêtes partent en chemin
  relatif (`/api/...`) et sont routées par Nginx. Pas d'URL en dur.
- Tous les appels passent `withCredentials: true` pour transporter le cookie JWT.
- Les services Angular reflètent les contrôleurs Spring :
  `analysis.service.ts`, `commercial.service.ts`, `lc-cross.service.ts`,
  `report.service.ts`, `admin.service.ts`, `auth.service.ts`.

---

## 4. Authentification (login)

Auth par **JWT stocké dans un cookie HttpOnly** (`swift_auth`), stateless côté serveur.

### Backend (`AuthController` + `SecurityConfig`)
| Endpoint                     | Méthode | Accès        | Rôle                                    |
|------------------------------|---------|--------------|-----------------------------------------|
| `/api/v1/auth/login`         | POST    | public       | authentifie, pose le cookie JWT HttpOnly |
| `/api/v1/auth/logout`        | POST    | public       | efface le cookie (maxAge=0)             |
| `/api/v1/auth/me`            | GET     | authentifié  | renvoie l'utilisateur courant + rôle    |

- Mots de passe hachés en **BCrypt**, sessions `STATELESS`.
- Filtre `JwtAuthFilter` placé avant `UsernamePasswordAuthenticationFilter`.
- Règles d'accès : `/api/v1/auth/**` public, `/api/v1/admin/**` → `ROLE_ADMIN`,
  tout le reste de `/api/v1/**` → authentifié.
- **CORS** : `allowCredentials=true` + origines whitelistées (indispensable pour
  le cookie cross-origin en dev sur le port 4200).
- **Compte admin par défaut** créé au démarrage (`DataInitializer`) :
  - utilisateur : `admin`
  - mot de passe : `Admin@1234`
  - ⚠️ à changer en production.

### Frontend (`auth.service.ts` + guards + interceptor)
- `auth.service.ts` : `login()`, `logout()`, `me()` — tous avec `withCredentials`.
- `jwt.interceptor.ts` : joint les credentials aux requêtes sortantes.
- **Guards de routes** :
  - `auth.guard.ts` — exige une session valide ;
  - `agent.guard.ts` — protège l'espace agent (`/app`, `/commercial`) ;
  - `admin.guard.ts` — protège l'espace admin (`/admin/**`).
- Au login réussi, l'ADMIN est redirigé vers `/admin/dashboard`, l'agent vers son workspace.

---

## 5. Pages frontend (Angular)

SPA en *standalone components* + *lazy-loading* (`app.routes.ts`).

### Pages publiques — layout `PublicShellComponent`
| Route        | Composant                | Contenu                          |
|--------------|--------------------------|----------------------------------|
| `/home`      | `HomeComponent`          | Page d'accueil                   |
| `/services`  | `ServicesPageComponent`  | Présentation des services        |
| `/about`     | `AboutComponent`         | À propos                         |
| `/login`     | `LoginComponent`         | Formulaire de connexion          |

(`''` redirige vers `/home`, `**` également.)

### Espace agent (guard `agentGuard`)
| Route          | Composant                  | Rôle                                       |
|----------------|----------------------------|--------------------------------------------|
| `/app`         | `AppComponent`             | Analyse SWIFT / LC (pipeline 1)            |
| `/commercial`  | `CommercialDocsComponent`  | Validation documents commerciaux + cross-LC (pipeline 2) |

### Espace admin (guard `adminGuard`) — layout `AdminShellComponent`
| Route               | Composant            | Rôle                          |
|---------------------|----------------------|-------------------------------|
| `/admin/dashboard`  | `DashboardComponent` | Statistiques / supervision    |
| `/admin/agents`     | `AgentListComponent` | Gestion des comptes agents     |
| `/admin/history`    | `HistoryComponent`   | Historique des analyses        |

---

## 6. Dockerisation

Tout est orchestré par `docker-compose.yml` à la racine, sur un réseau bridge
`swift-net`. Chaque service a un `Dockerfile` dédié.

### Images
- **`ai-service`** : `python:3.11-slim-bookworm` + dépendances système WeasyPrint
  (pango/cairo/gdk-pixbuf), OCR (tesseract fr/eng, poppler), torch **CPU**.
  Modèles ML **montés en volume** (≈ 27 GB au total), pas dans l'image. Utilisateur non-root.
- **`api`** : build Maven → JRE Alpine. Healthcheck via `wget` sur `/actuator/health`.
- **`frontend`** : build multi-stage `node:20-alpine` (`npm run build --configuration=production`)
  → artefacts servis par `nginx:1.25-alpine`.
- **`nginx`** (proxy) : image officielle + `nginx/conf.d/default.conf`.

### Dépendances de démarrage (`depends_on` + healthchecks)
```
postgres (healthy) ─┐
minio    (healthy) ─┼─► ai-service (healthy) ─► api (healthy) ─► nginx
                    └──────────────────────────►┘
minio-init : crée les buckets swift-documents / swift-reports puis s'arrête
```

### Réseau & ports exposés vers l'hôte
| Hôte                         | → Conteneur            | Usage                              |
|------------------------------|------------------------|------------------------------------|
| `${APP_PORT:-80}`            | nginx:80               | Application (point d'entrée unique)|
| `${MINIO_API_PORT:-9000}`    | minio:9000             | Téléchargement PDF via URL présignée (navigateur) |
| `${MINIO_CONSOLE_PORT:-9001}`| minio:9001             | Console d'admin MinIO              |

Les services `api`, `ai-service` et `postgres` **ne sont pas** exposés
directement : tout transite par Nginx.

### Détails Nginx importants
- `client_max_body_size 55M` + timeouts 600s (gros documents + warmup IA).
- `resolver 127.0.0.11` : re-résolution DNS Docker pour éviter les 502 après rebuild.
- `/api/reports/download/` : `proxy_buffering off` pour le **streaming** des PDF.
- `/actuator/` restreint au réseau interne.

---

## 7. Démarrage

### Prérequis
- Docker + Docker Compose
- Les modèles ML présents dans `swift-validator-ia/models` et
  `swift-validator-ia/release2/models` (montés en volume).

### Configuration
```bash
cp .env.example .env
# Éditer .env : POSTGRES_PASSWORD, MINIO_SECRET_KEY, JWT_SECRET (≥ 32 caractères)…
```

Variables clés de `.env` :
| Variable             | Rôle                                              |
|----------------------|---------------------------------------------------|
| `POSTGRES_*`         | Identifiants PostgreSQL                            |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | Identifiants MinIO              |
| `JWT_SECRET`         | Secret de signature JWT (≥ 32 caractères)          |
| `JWT_EXPIRATION_MS`  | Durée de validité du token (défaut 86400000 = 24h)|
| `APP_PORT`           | Port d'exposition de l'application (défaut 80)     |
| `FLASK_ENV`          | `production` recommandé                            |

### Lancer la plateforme complète
```bash
docker compose up -d --build
# Premier démarrage : warmup IA jusqu'à ~7 min (start_period 420s)
docker compose ps          # vérifier que api/ai-service sont "healthy"
docker compose logs -f api ai-service
```

### Accès
| Quoi                  | URL                              | Identifiants        |
|-----------------------|----------------------------------|---------------------|
| Application           | http://localhost                 | `admin` / `Admin@1234` |
| Console MinIO         | http://localhost:9001            | clés MinIO du `.env` |

---

## 8. Développement local (hors Docker)

Pour itérer sans rebuild des conteneurs :

| Service   | Commande                                                            | Port |
|-----------|---------------------------------------------------------------------|------|
| Python IA | `cd swift-validator-ia && python app/api.py`                        | 5000 |
| Backend   | `cd swift-validator-backend && mvn spring-boot:run`                 | 8080 |
| Frontend  | `cd swift_validator_frontend && npm install && npm start` (`ng serve`) | 4200 |

En local, le backend pointe `pipeline.base-url` sur `http://localhost:5000` et
les origines CORS autorisent `http://localhost:4200`. Garder PostgreSQL et MinIO
via `docker compose up -d postgres minio minio-init`.

> Pièges connus de l'env dev Windows (double dossier `project_lc/projet_lc`, venv,
> pins WeasyPrint/pydyf, GTK3) : voir les notes mémoire du projet.

---

## 9. Cartographie rapide du code

| Préoccupation                  | Fichier(s)                                                        |
|--------------------------------|-------------------------------------------------------------------|
| Orchestration des conteneurs   | `docker-compose.yml`                                              |
| Reverse-proxy                  | `nginx/conf.d/default.conf`                                       |
| Routes Python (2 pipelines)    | `swift-validator-ia/app/api.py`                                  |
| Pont Spring → Python           | `swift-validator-backend/.../service/PipelineClientService.java`  |
| Sécurité / JWT                 | `.../config/SecurityConfig.java`, `.../security/Jwt*.java`         |
| Login (REST)                   | `.../controller/AuthController.java`                              |
| Admin par défaut               | `.../config/DataInitializer.java`                                |
| Routes & guards frontend       | `swift_validator_frontend/src/app/app.routes.ts`, `core/guards/` |
| Services HTTP frontend         | `swift_validator_frontend/src/app/core/services/`                |
| Auth frontend                  | `core/services/auth.service.ts`, `core/interceptors/jwt.interceptor.ts` |
