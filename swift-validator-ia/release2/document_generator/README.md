# Bill of Lading Synthetic Dataset Generator

Générateur de B/L synthétiques pour la Phase 2 du mémoire : **validation documentaire multi-source**.

## Structure du projet

```
bl_generator/
├── generate_bl.py          # Script principal de génération
├── data_pools.py           # Données réalistes (ports, marchandises, etc.)
├── label_studio_export.py  # Export vers Label Studio
├── dataset_stats.py        # Statistiques et validation du dataset
└── README.md
```

## Installation

```bash
pip install reportlab faker
```

## Génération du dataset

### Commande de base (400 B/L, tous formats)
```bash
python generate_bl.py --count 400 --output ./bl_dataset
```

### Options avancées
```bash
# 500 B/L uniquement MSC + Maersk
python generate_bl.py --count 500 --output ./bl_dataset --formats msc,maersk

# Seed fixe pour reproductibilité
python generate_bl.py --count 400 --output ./bl_dataset --seed 42
```

### Formats disponibles
| Clé          | Compagnie                  | Couleur principale |
|--------------|----------------------------|--------------------|
| `msc`        | Mediterranean Shipping Co. | Rouge #D42E12      |
| `maersk`     | Maersk Line A/S            | Bleu marine #00243D|
| `cmacgm`     | CMA CGM S.A.               | Rouge #E30613      |
| `hapaglloyd` | Hapag-Lloyd AG             | Orange #E05206     |

## Structure du dataset généré

```
bl_dataset/
├── pdfs/                   # PDF générés (un par B/L)
│   ├── bl_msc_0001_MSCUN...pdf
│   ├── bl_maersk_0002_MAEU...pdf
│   └── ...
├── annotations/            # JSON par document
│   ├── bl_msc_0001_....json
│   └── ...
├── dataset.json            # Dataset complet
├── train.json              # 70% des données
├── val.json                # 15% des données
└── test.json               # 15% des données
```

## Format des annotations

Chaque document JSON contient :

```json
{
  "id": 1,
  "filename": "bl_msc_0001_MSCUNABCD123456789.pdf",
  "carrier": "msc",
  "bl_number": "MSCUNABCD123456789",
  "fields": {
    "bl_number": "MSCUNABCD123456789",
    "shipper_name": "ACME CORPORATION",
    "port_of_loading": "SHANGHAI, CHINA",
    "port_of_discharge": "ROTTERDAM, NETHERLANDS",
    "container_number": "MSCU1234567",
    "gross_weight_kg": "18234.50",
    ...
  },
  "bbox_annotations": [
    {
      "field": "bl_number",
      "bbox": [42.52, 780.10, 120.0, 15.0],
      "value": "MSCUNABCD123456789",
      "page": 1
    },
    ...
  ]
}
```

**Format bbox** : `[x, y, width, height]` en points PDF (1 point = 0.353 mm).
Origine : **bas-gauche** de la page (convention PDF/ReportLab).

## Champs annotés automatiquement

| Champ                | Label             | Description                        |
|----------------------|-------------------|------------------------------------|
| `bl_number`          | BL_NUMBER         | Numéro du connaissement            |
| `shipper_name`       | SHIPPER           | Nom + adresse expéditeur           |
| `consignee_name`     | CONSIGNEE         | Nom + adresse destinataire         |
| `notify_party`       | NOTIFY_PARTY      | Partie à notifier                  |
| `vessel`             | VESSEL            | Nom du navire                      |
| `voyage`             | VOYAGE            | Numéro de voyage                   |
| `port_of_loading`    | PORT_OF_LOADING   | Port de chargement                 |
| `port_of_discharge`  | PORT_OF_DISCHARGE | Port de déchargement               |
| `container_number`   | CONTAINER_NO      | Numéro de conteneur                |
| `container_type`     | CONTAINER_TYPE    | Type de conteneur (20' DRY, etc.)  |
| `seal_number`        | SEAL_NO           | Numéro de plomb                    |
| `gross_weight_kg`    | GROSS_WEIGHT      | Poids brut en KG                   |
| `measurement`        | MEASUREMENT       | Volume en CBM                      |
| `etd`                | ETD               | Date de départ estimée             |
| `eta`                | ETA               | Date d'arrivée estimée             |
| `issue_date`         | ISSUE_DATE        | Date d'émission                    |
| `freight_terms`      | FREIGHT_TERMS     | Conditions de fret (PREPAID, etc.) |

## Vérification du dataset

```bash
python dataset_stats.py --dataset ./bl_dataset/dataset.json
```

## Export vers Label Studio

```bash
# Installer Label Studio
pip install label-studio

# Générer les tâches d'annotation
python label_studio_export.py \
    --dataset ./bl_dataset/dataset.json \
    --output ./label_studio_tasks.json \
    --config ./label_studio_config.xml \
    --mode ocr \
    --pdf-url http://localhost:8000/pdfs

# Servir les PDFs localement
cd bl_dataset && python -m http.server 8000

# Lancer Label Studio
label-studio start

# Dans l'interface Label Studio :
# 1. Créer un projet "Bill of Lading Annotation"
# 2. Labeling Setup → Importer label_studio_config.xml
# 3. Import → JSON → Importer label_studio_tasks.json
```

## Justification académique (pour le chapitre Méthodologie)

> Les données réelles de connaissements maritimes ne sont pas publiquement disponibles en raison de leur caractère confidentiel (informations commerciales, données clients). En l'absence de dataset annoté public pour ce type documentaire, nous avons généré un corpus synthétique de N documents via le framework Python ReportLab.
>
> Chaque document est visuellement fidèle aux formats des quatre principaux armateurs mondiaux (MSC, Maersk, CMA CGM, Hapag-Lloyd), avec des données générées aléatoirement mais réalistes : codes de ports LOCODE, numéros de conteneurs au format ISO 6346, nomenclature de marchandises cohérente avec le Système Harmonisé. Les annotations bounding-box sont générées automatiquement lors de la création du PDF, garantissant une précision pixel-perfect sans annotation manuelle.
>
> Cette approche, similaire à celle employée dans [SynthText, PubLayNet, DocBank], est académiquement défendable pour la phase de développement et les expériences ablation. Une validation sur un échantillon de documents réels anonymisés (fournis par un partenaire industriel) serait nécessaire pour valider la généralisabilité du modèle.

## Références
- SROIE Dataset (receipts) : https://arxiv.org/abs/2103.10213
- CORD Dataset (receipts) : https://github.com/clovaai/cord
- RVL-CDIP (document classification) : https://huggingface.co/datasets/aharley/rvl_cdip
- DocBank : https://arxiv.org/abs/2006.01038
- LayoutLM : https://arxiv.org/abs/1912.13318
