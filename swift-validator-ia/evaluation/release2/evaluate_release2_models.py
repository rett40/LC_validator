"""
Evaluation complete des modeles Release 2
==========================================
Modeles evalues :
  1. LayoutLMv3 BL      (best_bl)      — NER token classification, 49 labels
  2. LayoutLMv3 Invoice  (best_invoice) — NER token classification, 41 labels
  3. LayoutLMv3 PL       (best_pl)      — NER token classification, 43 labels
  4. ResNet-18 Classifier               — Classification BL / Invoice / PL

Metriques produites :
  - F1 macro + weighted, precision, recall (seqeval entity-level pour NER)
  - F1 par entite (top N entites)
  - Matrice de confusion
  - Distribution des labels (treemap)
  - Rapport JSON structure

Usage :
    cd swift-validator-ia
    python evaluation/release2/evaluate_release2_models.py
    # Options :
    python evaluation/release2/evaluate_release2_models.py --batch-size 4 --max-samples 200
"""

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent.parent   # swift-validator-ia/
RELEASE2_ROOT = ROOT / "release2"
MODELS_ROOT   = RELEASE2_ROOT / "models"
_DEFAULT_DATA = RELEASE2_ROOT / "data" / "complex"
OUT_DIR       = Path(__file__).resolve().parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

# Peut etre remplace par --data-root en CLI
DATA_COMPLEX: Path = _DEFAULT_DATA

def _build_dataset_dirs(data_root: Path) -> dict[str, Path]:
    return {
        "bl":           data_root / "dataset_bl",
        "invoice":      data_root / "dataset_invoice",
        "packing_list": data_root / "dataset_packing_list",
    }

DATASET_DIRS = _build_dataset_dirs(DATA_COMPLEX)

# Les modeles LayoutLMv3 speciialises (nested best_xx/best_xx/)
LAYOUTLM_MODELS = {
    "bl":           MODELS_ROOT / "models_layout" / "best_bl"      / "best_bl",
    "invoice":      MODELS_ROOT / "models_layout" / "best_invoice"  / "best_invoice",
    "packing_list": MODELS_ROOT / "models_layout" / "best_pl"       / "best_pl",
}
CLASSIFIER_DIR = MODELS_ROOT / "doc_classifier"

DOC_TYPE_DISPLAY = {
    "bl":           "Bill of Lading",
    "invoice":      "Invoice",
    "packing_list": "Packing List",
}
COLORS = {"bl": "#2980b9", "invoice": "#27ae60", "packing_list": "#e67e22"}

# Prefixe des chemins images dans les JSONL (generes sur une autre machine)
# -> on le remplace par DATA_COMPLEX au chargement
_IMG_PATH_ANCHOR = "data/complex"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNEES DE TEST
# ─────────────────────────────────────────────────────────────────────────────

def _remap_image_path(raw: str) -> Path:
    """
    Traduit les chemins absolus generes sur la machine d'entrainement vers
    le chemin local : on prend la partie apres 'data/complex/' et on prefixe
    avec DATA_COMPLEX.

    Si le chemin existe tel quel (ex: eval_independent generee en local),
    on le retourne directement sans remapping pour eviter le double segment.
    """
    p = Path(raw)
    if p.exists():
        return p
    parts = p.parts
    try:
        idx = next(i for i in range(len(parts) - 1, -1, -1)
                   if parts[i].lower() == "complex")
        return DATA_COMPLEX / Path(*parts[idx + 1:])
    except StopIteration:
        return p   # chemin inconnu — on le retourne tel quel


def load_test_data(
    max_samples: int | None = None,
    doc_types: list[str] | None = None,
) -> tuple[dict[str, list[dict]], dict[str, dict[int, str]]]:
    """
    Charge les JSONL de test depuis data/complex/dataset_<type>/test/data.jsonl.

    Retourne :
      samples_by_type  : {"bl": [...], "invoice": [...], "packing_list": [...]}
      id2label_by_type : {"bl": {0: "O", 1: "B-BL_NUMBER", ...}, ...}

    Chaque sample : {doc_id, doc_type, image_path (remapped), words, bboxes, ner_tags}
    Les bboxes sont deja en format [x0, y0, x1, y1] dans la plage 0-1000.
    """
    target_types = doc_types or list(DATASET_DIRS.keys())

    samples_by_type:  dict[str, list]        = {dt: [] for dt in target_types}
    id2label_by_type: dict[str, dict[int, str]] = {}

    for dt in target_types:
        ds_dir = DATASET_DIRS.get(dt)
        if ds_dir is None or not ds_dir.exists():
            log.warning("Dataset manquant pour [%s] : %s", dt, ds_dir)
            continue

        # Label map propre a ce dataset (= espace de labels du modele)
        id2label_path = ds_dir / "id2label.json"
        id2label_by_type[dt] = {
            int(k): v
            for k, v in json.loads(id2label_path.read_text(encoding="utf-8")).items()
        }

        jsonl_path = ds_dir / "test" / "data.jsonl"
        if not jsonl_path.exists():
            log.warning("JSONL test manquant : %s", jsonl_path)
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                img_path = _remap_image_path(rec["image_path"])
                samples_by_type[dt].append({
                    "doc_id":     rec["doc_id"],
                    "doc_type":   dt,
                    "image_path": img_path,
                    "words":      rec["words"],
                    "bboxes":     rec["bboxes"],   # deja en 0-1000, format [x0,y0,x1,y1]
                    "ner_tags":   rec["ner_tags"],  # IDs du dataset local
                })

        if max_samples:
            samples_by_type[dt] = samples_by_type[dt][:max_samples]

        log.info("Test set [%s] : %d samples | %d labels | %s",
                 dt, len(samples_by_type[dt]),
                 len(id2label_by_type[dt]),
                 jsonl_path)

    return samples_by_type, id2label_by_type


# ─────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION NER — LayoutLMv3
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_preprocessor_config(model_dir: Path) -> None:
    """
    Certaines versions de transformers cherchent preprocessor_config.json mais le
    modele sauvegarde processor_config.json avec la config imbriquee sous "image_processor".
    Si preprocessor_config.json est absent, on l'extrait automatiquement.
    """
    pre_cfg = model_dir / "preprocessor_config.json"
    if pre_cfg.exists():
        return
    proc_cfg = model_dir / "processor_config.json"
    if not proc_cfg.exists():
        return
    data = json.loads(proc_cfg.read_text(encoding="utf-8"))
    img_cfg = data.get("image_processor", data)
    pre_cfg.write_text(json.dumps(img_cfg, indent=2), encoding="utf-8")
    log.info("preprocessor_config.json cree depuis processor_config.json (%s)", model_dir.name)


def evaluate_layoutlm(
    doc_type: str,
    samples: list[dict],
    id2label: dict[int, str],
    batch_size: int = 4,
) -> dict:
    """
    Evalue le modele LayoutLMv3 specialise pour un doc_type.

    - id2label : table {int -> str} propre a ce dataset/modele (chargee depuis
                 data/complex/dataset_<type>/id2label.json).
    - Les bboxes dans samples sont deja en format [x0, y0, x1, y1] plage 0-1000.
    - Les ner_tags sont des IDs locaux alignes sur id2label.
    """
    try:
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
        import torch
        from seqeval.metrics import (
            classification_report as seq_report,
            f1_score as seq_f1,
            precision_score as seq_prec,
            recall_score as seq_rec,
        )
        from PIL import Image
    except ImportError as e:
        log.error("Dependances manquantes pour LayoutLMv3 evaluation : %s", e)
        return {}

    model_dir = LAYOUTLM_MODELS[doc_type]
    log.info("=" * 65)
    log.info("NER — %s (%s)", DOC_TYPE_DISPLAY[doc_type], model_dir.name)
    log.info("=" * 65)

    if not model_dir.exists():
        log.error("Modele introuvable : %s", model_dir)
        return {"error": f"Model not found: {model_dir}"}

    _ensure_preprocessor_config(model_dir)

    # Charger le processor : tokenizer depuis le modele de base cache pour eviter
    # les incompatibilites de version entre tokenizer.json et la librairie installee.
    log.info("Chargement modele LayoutLMv3 [%s]...", doc_type)
    try:
        from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast
        img_proc  = LayoutLMv3ImageProcessor.from_pretrained(str(model_dir), apply_ocr=False)
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
        processor = LayoutLMv3Processor(image_processor=img_proc, tokenizer=tokenizer)
    except Exception as e:
        log.warning("Processor composite echoue (%s), fallback...", e)
        processor = LayoutLMv3Processor.from_pretrained(str(model_dir), apply_ocr=False)

    model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    # L'espace de labels du modele est le meme que celui du dataset
    model_id2label = {int(k): v for k, v in model.config.id2label.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Device : %s | Labels : %d | Samples : %d", device, len(model_id2label), len(samples))

    all_true_seqs: list[list[str]] = []
    all_pred_seqs: list[list[str]] = []
    n_errors = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        b_images, b_words, b_bboxes, b_true_tags = [], [], [], []

        for sample in batch:
            try:
                img_path = Path(sample["image_path"])
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), color=(255, 255, 255))

                # Bboxes deja en 0-1000 [x0, y0, x1, y1] — pas de normalisation
                bboxes = sample["bboxes"]

                # ner_tags sont des IDs locaux du dataset -> label strings
                true_labels_str = [
                    id2label.get(tag, "O") for tag in sample["ner_tags"]
                ]

                b_images.append(img)
                b_words.append(sample["words"])
                b_bboxes.append(bboxes)
                b_true_tags.append(true_labels_str)
            except Exception as exc:
                log.debug("Sample %s error : %s", sample.get("doc_id"), exc)
                n_errors += 1
                continue

        if not b_images:
            continue

        try:
            encoding = processor(
                b_images,
                b_words,
                boxes=b_bboxes,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()
                        if k in ("input_ids", "attention_mask", "bbox", "pixel_values")}

            with torch.no_grad():
                outputs = model(**encoding)

            logits = outputs.logits          # (B, seq_len, num_labels)
            preds  = logits.argmax(-1).cpu().numpy()

            # Aligner sur les mots originaux (ignorer padding et tokens speciaux)
            for b_idx, words in enumerate(b_words):
                try:
                    word_ids = encoding.get("input_ids")  # fallback
                    # Recuperer word_ids depuis l'encoding (non-batched call)
                    enc_single = processor(
                        b_images[b_idx],
                        [b_words[b_idx]],
                        boxes=[b_bboxes[b_idx]],
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    wids = enc_single.word_ids(batch_index=0) if hasattr(enc_single, "word_ids") else None
                except Exception:
                    wids = None

                seq_preds = []
                seq_trues = b_true_tags[b_idx]

                if wids is not None:
                    seen = set()
                    for tok_idx, wid in enumerate(wids):
                        if wid is None or wid in seen:
                            continue
                        seen.add(wid)
                        if tok_idx < preds.shape[1]:
                            pred_id   = int(preds[b_idx, tok_idx])
                            seq_preds.append(model_id2label.get(pred_id, "O"))
                else:
                    # Fallback : prendre les premiers len(words) tokens non-speciaux
                    valid_preds = [int(p) for p in preds[b_idx]
                                   if p != -100][:len(words)]
                    seq_preds = [model_id2label.get(p, "O") for p in valid_preds]

                # Aligner les longueurs
                min_len = min(len(seq_preds), len(seq_trues))
                seq_preds = seq_preds[:min_len]
                seq_trues = seq_trues[:min_len]

                if seq_preds and seq_trues:
                    all_pred_seqs.append(seq_preds)
                    all_true_seqs.append(seq_trues)

        except Exception as exc:
            log.warning("Batch %d/%d error : %s", i // batch_size + 1,
                        (len(samples) + batch_size - 1) // batch_size, exc)
            n_errors += batch_size
            continue

        if (i // batch_size) % 5 == 0:
            log.info("  Batch %d/%d traite",
                     i // batch_size + 1, (len(samples) + batch_size - 1) // batch_size)

    if not all_true_seqs:
        log.error("[%s] Aucune prediction valide.", doc_type)
        return {"error": "No valid predictions"}

    # ── Metriques seqeval (entity-level) ─────────────────────────────────────
    f1_w   = seq_f1(all_true_seqs, all_pred_seqs, average="weighted", zero_division=0)
    f1_m   = seq_f1(all_true_seqs, all_pred_seqs, average="macro",    zero_division=0)
    prec_w = seq_prec(all_true_seqs, all_pred_seqs, average="weighted", zero_division=0)
    rec_w  = seq_rec(all_true_seqs, all_pred_seqs,  average="weighted", zero_division=0)
    report_str = seq_report(all_true_seqs, all_pred_seqs, zero_division=0, digits=3)
    report_dict = seq_report(all_true_seqs, all_pred_seqs, zero_division=0, output_dict=True)

    log.info("[%s] F1 weighted : %.4f | F1 macro : %.4f", doc_type, f1_w, f1_m)
    log.info("[%s] Precision   : %.4f | Recall   : %.4f", doc_type, prec_w, rec_w)
    log.info("\n%s", report_str)

    # ── F1 par entite ─────────────────────────────────────────────────────────
    per_entity: dict[str, dict] = {}
    for entity, vals in report_dict.items():
        if isinstance(vals, dict) and entity not in ("micro avg", "macro avg", "weighted avg"):
            per_entity[entity] = {
                "precision": round(vals.get("precision", 0), 4),
                "recall":    round(vals.get("recall",    0), 4),
                "f1":        round(vals.get("f1-score",  0), 4),
                "support":   int(vals.get("support",     0)),
            }

    return {
        "doc_type":     doc_type,
        "n_samples":    len(samples),
        "n_errors":     n_errors,
        "n_labels":     len(model_id2label),
        "f1_weighted":  round(f1_w,   4),
        "f1_macro":     round(f1_m,   4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted":    round(rec_w,  4),
        "per_entity":   per_entity,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATION CLASSIFICATEUR — ResNet-18
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(samples_by_type: dict[str, list[dict]]) -> dict:
    """Evalue le classificateur ResNet-18 sur l'ensemble des samples de test."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from PIL import Image
    except ImportError as e:
        log.error("Dependances manquantes pour classifier evaluation : %s", e)
        return {}

    log.info("=" * 65)
    log.info("CLASSIFIER — ResNet-18 (BL / Invoice / PL)")
    log.info("=" * 65)

    class_names_path = CLASSIFIER_DIR / "class_names.json"
    model_path       = CLASSIFIER_DIR / "best_model.pt"
    if not model_path.exists():
        log.error("Modele ResNet-18 introuvable : %s", model_path)
        return {"error": f"Model not found: {model_path}"}

    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    n_classes   = len(class_names)
    class2idx   = {c: i for i, c in enumerate(class_names)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruction architecturale identique au training
    base = models.resnet18(weights=None)
    in_features = base.fc.in_features   # 512
    base.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )
    base.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    base.to(device)
    base.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    y_true, y_pred = [], []
    n_errors = 0

    all_samples = [
        (s, dt) for dt, samples in samples_by_type.items() for s in samples
    ]
    log.info("Classification de %d images...", len(all_samples))

    for sample, true_dt in all_samples:
        try:
            img    = Image.open(sample["image_path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = base(tensor)
                proba  = torch.softmax(logits, dim=1)[0]
                pred_idx = int(proba.argmax())
            y_true.append(class2idx.get(true_dt, -1))
            y_pred.append(pred_idx)
        except Exception as exc:
            log.debug("Classifier sample error : %s", exc)
            n_errors += 1

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    valid = (y_true_np >= 0)
    y_true_np = y_true_np[valid]
    y_pred_np = y_pred_np[valid]

    acc  = accuracy_score(y_true_np, y_pred_np)
    f1_w = f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
    f1_m = f1_score(y_true_np, y_pred_np, average="macro",    zero_division=0)
    prec = precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
    rec  = recall_score(y_true_np,  y_pred_np, average="weighted", zero_division=0)
    # Labels present dans ce sous-ensemble (peut etre < n_classes si max-samples est petit)
    present_labels = sorted(set(y_true_np.tolist()) | set(y_pred_np.tolist()))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]

    cm   = confusion_matrix(y_true_np, y_pred_np, labels=list(range(n_classes)))
    report_dict = classification_report(
        y_true_np, y_pred_np,
        labels=present_labels, target_names=present_names,
        output_dict=True, zero_division=0,
    )

    log.info("Accuracy    : %.4f", acc)
    log.info("F1 weighted : %.4f | F1 macro : %.4f", f1_w, f1_m)
    log.info("\n%s", classification_report(
        y_true_np, y_pred_np,
        labels=present_labels, target_names=present_names,
        zero_division=0,
    ))

    per_class = {
        cls: {
            "precision": round(report_dict[cls]["precision"], 4),
            "recall":    round(report_dict[cls]["recall"],    4),
            "f1":        round(report_dict[cls]["f1-score"],  4),
            "support":   int(report_dict[cls]["support"]),
        }
        for cls in present_names
        if cls in report_dict
    }

    return {
        "n_samples":          len(y_true_np),
        "n_errors":           n_errors,
        "accuracy":           round(acc,  4),
        "f1_weighted":        round(f1_w, 4),
        "f1_macro":           round(f1_m, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted":    round(rec,  4),
        "per_class":          per_class,
        "confusion_matrix":   cm.tolist(),
        "class_names":        class_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_entity_f1(metrics: dict, doc_type: str, top_n: int = 20):
    """Graphique en barres horizontales : F1 par entite (top N)."""
    per_entity = metrics.get("per_entity", {})
    if not per_entity:
        return

    # Trier par F1 decroissant, garder top_n
    sorted_ents = sorted(per_entity.items(), key=lambda x: x[1]["f1"], reverse=True)[:top_n]
    labels = [e for e, _ in sorted_ents]
    f1s    = [v["f1"]     for _, v in sorted_ents]
    supps  = [v["support"] for _, v in sorted_ents]

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.4)))
    bars = ax.barh(labels[::-1], f1s[::-1], color=COLORS.get(doc_type, "#555"), alpha=0.85)
    for bar, supp in zip(bars, supps[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={supp}", va="center", fontsize=7.5, color="#333")
    ax.set_xlim(0, 1.12)
    ax.axvline(0.90, color="red",   ls="--", lw=0.8, label="F1=0.90")
    ax.axvline(0.75, color="orange", ls=":",  lw=0.8, label="F1=0.75")
    ax.set(
        title=f"F1 par entite — {DOC_TYPE_DISPLAY.get(doc_type, doc_type)} "
              f"(F1 macro={metrics['f1_macro']:.3f}, weighted={metrics['f1_weighted']:.3f})",
        xlabel="F1-score (entity-level, seqeval)",
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / f"ner_{doc_type}_per_entity_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("F1 par entite [%s] -> %s", doc_type, out)


def plot_ner_summary(results_ner: dict[str, dict]):
    """Graphique comparatif des 3 modeles NER : F1 macro + weighted."""
    fig, ax = plt.subplots(figsize=(7, 4))
    doc_types = list(results_ner.keys())
    x = np.arange(len(doc_types))
    w = 0.35
    f1_macro    = [results_ner[dt].get("f1_macro",    0) for dt in doc_types]
    f1_weighted = [results_ner[dt].get("f1_weighted", 0) for dt in doc_types]

    bars1 = ax.bar(x - w / 2, f1_macro,    w, label="F1 macro",    alpha=0.85,
                   color=[COLORS.get(dt, "#555") for dt in doc_types])
    bars2 = ax.bar(x + w / 2, f1_weighted, w, label="F1 weighted", alpha=0.55,
                   color=[COLORS.get(dt, "#555") for dt in doc_types])

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set(
        title="Comparaison modeles NER — LayoutLMv3 specialises",
        xticks=x,
        xticklabels=[DOC_TYPE_DISPLAY.get(dt, dt) for dt in doc_types],
        ylabel="F1-score (seqeval entity-level)",
        ylim=(0, 1.10),
    )
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "ner_models_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Comparaison NER -> %s", out)


def plot_ner_precision_recall(results_ner: dict[str, dict]):
    """Scatter Precision vs Recall par entite pour les 3 modeles."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (dt, metrics) in zip(axes, results_ner.items()):
        pe = metrics.get("per_entity", {})
        if not pe:
            ax.set_title(f"{DOC_TYPE_DISPLAY.get(dt, dt)} — no data")
            continue
        precs  = [v["precision"] for v in pe.values()]
        recalls = [v["recall"]   for v in pe.values()]
        f1s    = [v["f1"]        for v in pe.values()]
        sc = ax.scatter(recalls, precs, c=f1s, cmap="RdYlGn",
                        vmin=0, vmax=1, s=60, alpha=0.8)
        for entity, v in pe.items():
            ent_short = entity.replace("B-", "").replace("I-", "")
            ax.annotate(ent_short, (v["recall"], v["precision"]),
                        fontsize=6, ha="left", va="bottom")
        plt.colorbar(sc, ax=ax, label="F1")
        ax.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax.set(
            title=f"{DOC_TYPE_DISPLAY.get(dt, dt)}",
            xlabel="Recall", ylabel="Precision",
            xlim=(-0.05, 1.10), ylim=(-0.05, 1.10),
        )
    plt.suptitle("Precision vs Recall par entite — modeles NER LayoutLMv3", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "ner_precision_recall_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Precision-Recall scatter -> %s", out)


def plot_classifier_confusion(metrics: dict):
    """Matrice de confusion du classificateur ResNet-18."""
    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]
    labels = [DOC_TYPE_DISPLAY.get(c, c) for c in class_names]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)), yticks=range(len(labels)),
        xticklabels=labels, yticklabels=labels,
        xlabel="Predit", ylabel="Reel",
        title=(f"Matrice de confusion — ResNet-18\n"
               f"Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1_weighted']:.4f}"),
    )
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    out = OUT_DIR / "classifier_confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Matrice de confusion ResNet-18 -> %s", out)


def plot_classifier_per_class(metrics: dict):
    """Barres Precision / Recall / F1 par classe pour le classificateur."""
    pc = metrics["per_class"]
    # Use only classes actually present in per_class (may be fewer than class_names)
    class_names = [c for c in metrics["class_names"] if c in pc]
    labels = [DOC_TYPE_DISPLAY.get(c, c) for c in class_names]
    x = np.arange(len(labels))
    w = 0.28

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, [pc[c]["precision"] for c in class_names], w, label="Precision", alpha=0.85, color="#3498db")
    ax.bar(x,     [pc[c]["recall"]    for c in class_names], w, label="Recall",    alpha=0.85, color="#2ecc71")
    ax.bar(x + w, [pc[c]["f1"]        for c in class_names], w, label="F1",        alpha=0.85, color="#e74c3c")
    ax.set(
        title="ResNet-18 — Precision / Recall / F1 par classe",
        xticks=x, xticklabels=labels,
        ylabel="Score", ylim=(0, 1.12),
    )
    ax.legend()
    for i, c in enumerate(class_names):
        for j, (val, offset) in enumerate([
            (pc[c]["precision"], -w),
            (pc[c]["recall"],     0),
            (pc[c]["f1"],         w),
        ]):
            ax.text(i + offset, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "classifier_per_class_metrics.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Metriques par classe ResNet-18 -> %s", out)


def plot_support_distribution(results_ner: dict[str, dict]):
    """Histogramme du support (nombre d'entites) par entite et par modele."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (dt, metrics) in zip(axes, results_ner.items()):
        pe = metrics.get("per_entity", {})
        if not pe:
            continue
        entities = list(pe.keys())
        supports = [pe[e]["support"] for e in entities]
        sorted_idx = np.argsort(supports)[::-1]
        top = min(20, len(sorted_idx))
        top_ents = [entities[i].replace("B-", "") for i in sorted_idx[:top]]
        top_sups = [supports[i] for i in sorted_idx[:top]]
        ax.barh(top_ents[::-1], top_sups[::-1], color=COLORS.get(dt, "#555"), alpha=0.7)
        ax.set(title=f"{DOC_TYPE_DISPLAY.get(dt, dt)} — Top {top} entites",
               xlabel="Support (nb entites gold)")
    plt.suptitle("Distribution du support par entite — test set", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "ner_entity_support_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Distribution support -> %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# 5. RAPPORT JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results: dict):
    out = OUT_DIR / "report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Rapport JSON -> %s", out)


def print_summary(results: dict):
    print()
    print("=" * 70)
    print("  RESUME EVALUATION — Release 2 Models")
    print("=" * 70)

    print("\n  LayoutLMv3 NER (entity-level seqeval)\n")
    print(f"  {'Modele':<25} {'F1 macro':>10} {'F1 weighted':>12} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*65}")
    for dt, m in results.get("ner", {}).items():
        if "error" in m:
            print(f"  {DOC_TYPE_DISPLAY.get(dt, dt):<25}  ERREUR: {m['error']}")
            continue
        print(f"  {DOC_TYPE_DISPLAY.get(dt, dt):<25} {m['f1_macro']:>10.4f} "
              f"{m['f1_weighted']:>12.4f} {m['precision_weighted']:>10.4f} "
              f"{m['recall_weighted']:>8.4f}")

    clf = results.get("classifier", {})
    if clf and "error" not in clf:
        print(f"\n  ResNet-18 Classificateur\n")
        print(f"  Accuracy    : {clf['accuracy']:.4f}")
        print(f"  F1 weighted : {clf['f1_weighted']:.4f}  |  F1 macro : {clf['f1_macro']:.4f}")
        for c in clf.get("class_names", []):
            pc = clf["per_class"].get(c, {})
            print(f"    {DOC_TYPE_DISPLAY.get(c, c):<20} "
                  f"P={pc.get('precision', 0):.3f}  "
                  f"R={pc.get('recall', 0):.3f}  "
                  f"F1={pc.get('f1', 0):.3f}")

    print(f"\n  Rapports sauvegardes dans : {OUT_DIR}")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluation Release 2 models")
    p.add_argument("--batch-size",  type=int, default=4,
                   help="Batch size pour l'inference LayoutLMv3 (defaut: 4)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limiter a N samples par doc_type (debug)")
    p.add_argument("--skip-ner",    action="store_true",
                   help="Sauter l'evaluation NER (seulement le classificateur)")
    p.add_argument("--skip-clf",    action="store_true",
                   help="Sauter l'evaluation du classificateur")
    p.add_argument("--doc-types",   nargs="+",
                   default=["bl", "invoice", "packing_list"],
                   choices=["bl", "invoice", "packing_list"],
                   help="Doc types a evaluer pour le NER")
    p.add_argument("--data-root", type=str, default=None,
                   help="Racine des datasets JSONL (defaut: release2/data/complex). "
                        "Utile pour evaluer sur le dataset genere par 06_generate_eval_dataset.py")
    return p.parse_args()


def main():
    args = parse_args()

    # Appliquer --data-root si fourni
    global DATA_COMPLEX, DATASET_DIRS
    if args.data_root:
        DATA_COMPLEX = Path(args.data_root).resolve()
        DATASET_DIRS = _build_dataset_dirs(DATA_COMPLEX)

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Evaluation Release 2 — LayoutLMv3 + ResNet-18      ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("Datasets  : %s", DATA_COMPLEX)
    log.info("Output    : %s", OUT_DIR)

    # ── 1. Charger les donnees ────────────────────────────────────────────────
    samples_by_type, id2label_by_type = load_test_data(
        max_samples=args.max_samples,
        doc_types=args.doc_types,
    )

    results: dict = {
        "evaluation_date": str(__import__("datetime").datetime.now()),
        "data_root":       str(DATA_COMPLEX),
        "ner":             {},
        "classifier":      {},
        "outputs":         [],
    }

    # ── 2. NER — LayoutLMv3 ──────────────────────────────────────────────────
    if not args.skip_ner:
        for dt in args.doc_types:
            if dt not in samples_by_type or not samples_by_type[dt]:
                log.warning("Pas de samples [%s] dans le test set.", dt)
                continue
            metrics = evaluate_layoutlm(
                doc_type=dt,
                samples=samples_by_type[dt],
                id2label=id2label_by_type[dt],
                batch_size=args.batch_size,
            )
            results["ner"][dt] = metrics
            if "error" not in metrics:
                plot_per_entity_f1(metrics, dt)

        if len(results["ner"]) >= 2:
            plot_ner_summary(results["ner"])
            plot_ner_precision_recall(results["ner"])
            plot_support_distribution(results["ner"])

    # ── 3. Classificateur ResNet-18 ───────────────────────────────────────────
    if not args.skip_clf:
        clf_metrics = evaluate_classifier(samples_by_type)
        results["classifier"] = clf_metrics
        if clf_metrics and "error" not in clf_metrics:
            plot_classifier_confusion(clf_metrics)
            plot_classifier_per_class(clf_metrics)

    # ── 4. Rapport ────────────────────────────────────────────────────────────
    results["outputs"] = [str(p) for p in sorted(OUT_DIR.glob("*.png"))]
    results["outputs"].append(str(OUT_DIR / "report.json"))

    save_report(results)
    print_summary(results)


if __name__ == "__main__":
    main()
