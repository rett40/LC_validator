# Méthodologie MLOps — Déploiement de la plateforme

## Principes appliqués

| Principe | Implémentation |
|---|---|
| Versions de modèles épinglées | `swift-validator-ia/config/model_registry.py` (révisions HF exactes) + `models.yaml` (SHA-256 des poids locaux) |
| Parité numérique local ↔ prod | Device d'inférence forcé `cpu` des deux côtés (`SWIFT_INFERENCE_DEVICE`) ; mêmes versions torch/transformers |
| Images versionnées | `docker-compose.yml` : `swift-validator/*:${PLATFORM_VERSION}` ; MinIO épinglé par digest |
| Serveur de production | gunicorn `--workers 1 --threads 4 --preload` (les modèles ~10 GB ne sont chargés qu'une fois) |
| Warmup au démarrage | `warmup_all()` dans `app/api.py` — tous les modèles chargés avant de servir ; statut par composant dans `/health` |
| Gate de non-régression | `scripts/parity_check.py` + harnais d'éval existants |

## Pourquoi la sémantique divergeait (corrigé)

1. **Le local utilisait le GPU (MX130), la prod le CPU** : sentence-transformers
   choisit CUDA automatiquement → kernels flottants différents → scores différents.
   Corrigé : device `cpu` forcé partout via le registre central.
2. **Modèles Hub non épinglés** : le local résolvait `main` en ligne, la prod
   servait un snapshot figé. Corrigé : révisions épinglées par commit.
3. **Cache prod incomplet** : `all-MiniLM-L6-v2` et `nli-deberta-v3-small`
   absents de `.hf_cache` → la prod (offline) retombait en TF-IDF/heuristique
   alors que le local utilisait les vrais modèles. Corrigé : cache pré-rempli.

## Réponse à « faut-il initialiser le modèle une seule fois ? »

**Oui — une fois par processus, au démarrage.** C'est ce que fait `warmup_all()`
(exécuté à l'import) + les singletons (`swift_st_cache`, `_commercial_pipeline`).
Chaque requête réutilise les modèles déjà en RAM. gunicorn est configuré avec
**1 worker** exprès : N workers = N copies des modèles en RAM sans gain CPU.
La concurrence vient des threads (4).

## Procédure de déploiement

```bash
# 1. (après tout ré-entraînement) régénérer le manifest de versions
python swift-validator-ia/scripts/build_model_manifest.py

# 2. incrémenter la version dans .env
echo "PLATFORM_VERSION=2.1.0" >> .env

# 3. build + démarrage
docker compose build ai-service
docker compose up -d

# 4. GATE DE PARITÉ (bloquant) : comparer local et conteneur
#    - démarrer l'API locale (port 5000) ; le conteneur est publié sur 127.0.0.1:5001
python swift-validator-ia/scripts/parity_check.py \
    --a http://localhost:5000 --b http://localhost:5001

# 5. vérifier les versions servies à distance
curl -s http://localhost:5001/health | python -m json.tool
#    → blocs "parity" (libs, device, révisions Hub) et "model_manifest" (sha256)
```

**Règle : on ne déploie que si `parity_check.py` sort `PARITE OK`.**

## Performance LayoutLM (extraction d'entités)

Optimisations mesurées iso-précision (harnais `evaluation/release2/perf_diff_harness.py`) :
- `padding="longest"` (×3.3–3.8, déjà en place)
- Cache du prétraitement image par page : le `pixel_values` (resize 224 +
  normalisation) était recalculé à chacun des 13-25 forwards d'un document
  alors qu'il est identique — désormais calculé une fois par page.
- `torch.inference_mode()` au lieu de `no_grad()`.
- Escape hatch : `R2_DISABLE_PV_CACHE=1` restaure l'ancien chemin (pour A/B).

## Registre MLflow & monitoring de dérive Evidently

### MLflow (conteneur `swift-mlflow`, UI : http://localhost:5500)
- **Purement additif** : aucun service n'en dépend, aucun code applicatif ne
  l'appelle pendant une analyse. Ressources bornées (`mem_limit: 512m`,
  `cpus: 0.5`) → ne peut jamais empiéter sur le service IA.
- Backend SQLite sur volume dédié `mlflow_data` ; artefacts dans le bucket
  MinIO `mlflow` (séparé des buckets applicatifs).
- Les **11 modèles servis** y sont enregistrés (expérience
  `swift-validator-models`) avec leurs métriques d'évaluation, leurs SHA-256
  en tags et l'alias **production** ; un run `hub-models-pinned` journalise
  les 5 révisions HuggingFace épinglées.
- Ré-enregistrement (après ré-entraînement + regénération de models.yaml) :
  ```bash
  docker cp swift-validator-ia/scripts/register_models_mlflow.py swift-mlflow:/tmp/
  docker cp swift-validator-ia/models.yaml swift-mlflow:/tmp/
  docker exec swift-mlflow python /tmp/register_models_mlflow.py
  ```
- Risque zéro pendant une démo d'analyse : `docker compose stop mlflow`
  (redémarre en ~5 s avec `docker compose start mlflow`).
- Rollback complet : restaurer `docker-compose.yml.bak-avant-mlflow` puis
  `docker compose up -d --remove-orphans`.

### Evidently (hors ligne, environnement isolé `.mlops_venv`)
- Rapport de dérive des features du modèle de risque :
  ```bash
  .mlops_venv/Scripts/python.exe swift-validator-ia/scripts/drift_report_evidently.py
  # → swift-validator-ia/evaluation/drift/drift_report.html
  ```
- Référence = 24 000 LC (entraînement) vs courant = 6 000 LC (flux frais
  simulé). Résultat actuel : 0/27 colonnes en dérive (attendu : même corpus
  synthétique). En production réelle, remplacer « courant » par la fenêtre
  des dossiers récents — la dérive synthétique→réel y sera visible.
- N'est PAS installé dans l'image du service IA (aucun impact runtime).

### Performance en production (Docker/WSL2) — règles d'exploitation

Mesuré : conteneur **au repos** = 12,9 s/BL (identique au local). Les ralentissements
×4 observés venaient de la **contention RAM** (API locale ~10 GB + VM Docker sur
un hôte 16 GB), pas de Docker lui-même.

1. `C:\Users\Aziz\.wslconfig` fixe `memory=10GB` (défaut 8 GB trop juste pour
   garder les ~10 GB de modèles résidents). Prise d'effet : `wsl --shutdown`
   puis relancer Docker Desktop.
2. **Ne pas faire tourner l'API locale et la stack Docker en même temps** en
   usage normal — la machine n'a pas la RAM pour deux copies des modèles.
   Exception : le gate de parité (accepter qu'il soit lent, les résultats
   comparés ne dépendent pas du temps d'exécution).
3. Si un jour la perf prod devient critique : remplacer les bind mounts
   Windows des modèles par un volume Docker nommé (I/O natives ext4 au lieu
   du protocole de partage Windows→WSL2, plus rapide en cas de page-fault).

Vérification avant tout déploiement touchant l'extracteur :
```bash
python evaluation/release2/perf_diff_harness.py snapshot --tag avant --n 6
# ... modification ...
python evaluation/release2/perf_diff_harness.py snapshot --tag apres --n 6
python evaluation/release2/perf_diff_harness.py compare --a avant --b apres
```
