"""
04_train_per_doctype.py
=======================
Fine-tuning LayoutLMv3 — un modèle séparé par type de document.

Améliorations v2 :
  - Label smoothing (0.1) → réduit la sur-confiance sur les tokens O
  - Perte pondérée par classe (class-weighted CE) → compense le déséquilibre O vs entités
  - Bbox jitter PER-TOKEN indépendant → simule mieux les imprécisions OCR réelles
  - Augmentation image plus agressive : RandomPerspective + RandomAffine + ColorJitter
  - Hyperparamètres affinés : LR 1.5e-5, epochs 20, patience 7
  - CustomTrainer pour injecter les poids de classe dans la perte
  - Suivi F1 par entité toutes les N epochs pendant l'entraînement

Usage :
  python scripts/04_train_per_doctype.py --doc-type bl
  python scripts/04_train_per_doctype.py --doc-type invoice --epochs 20 --bbox-jitter 8
  python scripts/04_train_per_doctype.py --doc-type packing_list --resume
"""

from __future__ import annotations

import sys
import json
import logging
import argparse
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = RELEASE2_ROOT / "data"
LOGS_DIR      = RELEASE2_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "04_train_per_doctype.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME          = "microsoft/layoutlmv3-base"
MAX_LENGTH          = 512
DOC_STRIDE          = 128
DEFAULT_EPOCHS      = 20          # v2 : 20 (était 12)
DEFAULT_BATCH       = 2
DEFAULT_LR          = 1.5e-5      # v2 : 1.5e-5 (était 2e-5)
DEFAULT_GRAD_ACCUM  = 4
DEFAULT_SEED        = 42
WARMUP_RATIO        = 0.10
WEIGHT_DECAY        = 0.01
MAX_GRAD_NORM       = 1.0
LOGGING_STEPS       = 20
LABEL_SMOOTHING     = 0.1         # v2 : lissage des labels

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP16   = torch.cuda.is_available()

DATASET_DIRS = {
    "bl":           "dataset_bl",
    "invoice":      "dataset_invoice",
    "packing_list": "dataset_packing_list",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_label_maps(dataset_dir: Path) -> tuple[dict, dict]:
    l2i = json.loads((dataset_dir / "label2id.json").read_text(encoding="utf-8"))
    i2l = {
        int(k): v
        for k, v in json.loads(
            (dataset_dir / "id2label.json").read_text(encoding="utf-8")
        ).items()
    }
    return l2i, i2l


# ─────────────────────────────────────────────────────────────────────────────
# Dataset PyTorch — bbox jitter PER-TOKEN + augmentation améliorée
# ─────────────────────────────────────────────────────────────────────────────

class LayoutLMv3Dataset:
    """
    Charge un fichier JSONL word-level et applique le processor LayoutLMv3.

    v2 — améliorations :
      - bbox_jitter appliqué PER TOKEN de façon indépendante (simulation
        d'imprécision OCR réelle, pas un décalage global uniforme).
      - Augmentation image : RandomPerspective + RandomAffine (train seulement).
      - Statistiques de labels collectées pour le calcul des poids de classe.
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor,
        label2id: dict[str, int],
        max_length: int = MAX_LENGTH,
        doc_stride: int = DOC_STRIDE,
        augment: bool = False,
        bbox_jitter: int = 0,
    ):
        self.processor   = processor
        self.label2id    = label2id
        self.max_length  = max_length
        self.doc_stride  = doc_stride
        self.bbox_jitter = bbox_jitter if augment else 0
        self.label_counts: Counter = Counter()

        if augment:
            from torchvision import transforms
            self._aug = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.25, contrast=0.25, saturation=0.10, hue=0.05
                ),
                transforms.RandomGrayscale(p=0.08),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # v2 : transformations géométriques légères (simule scan incliné)
                transforms.RandomAffine(
                    degrees=2,
                    translate=(0.01, 0.01),
                    scale=(0.98, 1.02),
                    fill=255,
                ),
                transforms.RandomPerspective(distortion_scale=0.03, p=0.3, fill=255),
            ])
        else:
            self._aug = None

        self.examples = self._load_and_encode(jsonl_path)

    def _load_and_encode(self, path: Path) -> list[dict]:
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        encoded = []
        for ex in tqdm(raw, desc=f"  encoding {path.parent.name}", unit="page"):
            chunks = self._encode_example(ex)
            encoded.extend(chunks)
            # Collecte des labels pour le calcul des poids
            for c in chunks:
                for lbl in c["labels"]:
                    if lbl != -100:
                        self.label_counts[lbl] += 1
        return encoded

    def _apply_bbox_jitter(self, bboxes: list[list[int]]) -> list[list[int]]:
        """
        v2 : jitter PER TOKEN indépendant — chaque token reçoit un décalage
        différent sur x et y, ce qui simule des imprécisions OCR locales.
        """
        if self.bbox_jitter == 0:
            return bboxes
        new_bboxes = []
        for box in bboxes:
            jx = random.randint(-self.bbox_jitter, self.bbox_jitter)
            jy = random.randint(-self.bbox_jitter, self.bbox_jitter)
            new_box = [
                max(0, min(1000, box[0] + jx)),
                max(0, min(1000, box[1] + jy)),
                max(0, min(1000, box[2] + jx)),
                max(0, min(1000, box[3] + jy)),
            ]
            # Garantir x0 <= x1, y0 <= y1
            new_box[0], new_box[2] = min(new_box[0], new_box[2]), max(new_box[0], new_box[2])
            new_box[1], new_box[3] = min(new_box[1], new_box[3]), max(new_box[1], new_box[3])
            new_bboxes.append(new_box)
        return new_bboxes

    def _encode_example(self, ex: dict) -> list[dict]:
        words    = ex["words"]
        bboxes   = ex["bboxes"]
        ner_tags = ex["ner_tags"]

        bboxes = self._apply_bbox_jitter(bboxes)

        img_path = Path(ex["image_path"])
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        if self._aug is not None:
            image = self._aug(image)

        encoding = self.processor(
            images                    = image,
            text                      = words,
            boxes                     = bboxes,
            word_labels               = ner_tags,
            truncation                = True,
            max_length                = self.max_length,
            stride                    = self.doc_stride,
            padding                   = "max_length",
            return_overflowing_tokens = True,
            return_offsets_mapping    = False,
        )

        chunks = []
        for i in range(len(encoding["input_ids"])):
            chunk = {
                "input_ids":      encoding["input_ids"][i],
                "attention_mask": encoding["attention_mask"][i],
                "bbox":           encoding["bbox"][i],
                "pixel_values":   encoding["pixel_values"][i],
                "labels":         encoding["labels"][i],
            }
            if "token_type_ids" in encoding:
                chunk["token_type_ids"] = encoding["token_type_ids"][i]
            chunks.append(chunk)
        return chunks

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


# ─────────────────────────────────────────────────────────────────────────────
# Calcul des poids de classe (inverse-frequency weighting)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    label_counts: Counter,
    num_labels: int,
    o_label_id: int = 0,
    o_weight: float = 0.1,   # downweight O — très sur-représenté
) -> torch.Tensor:
    """
    Calcule les poids par classe pour la perte pondérée.
    Les classes rares (entités) reçoivent un poids élevé, O est abaissé.
    """
    total = sum(label_counts.values()) or 1
    weights = torch.ones(num_labels)

    for label_id, count in label_counts.items():
        if label_id == o_label_id:
            weights[label_id] = o_weight
        elif count > 0:
            # Poids inversement proportionnel à la fréquence (plafonné à 10)
            freq = count / total
            weights[label_id] = min(10.0, 1.0 / (freq * num_labels))

    # Normaliser pour que la moyenne soit ≈ 1.0
    non_o = [i for i in range(num_labels) if i != o_label_id]
    if non_o:
        mean_w = weights[non_o].mean()
        if mean_w > 0:
            weights[non_o] = weights[non_o] / mean_w

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# CustomTrainer — perte pondérée + label smoothing
# ─────────────────────────────────────────────────────────────────────────────

class WeightedLossTrainer:
    """
    Wrapper minimal autour de HuggingFace Trainer qui surcharge compute_loss
    pour injecter les poids de classe et le label smoothing.
    """
    pass  # Défini dynamiquement après import de Trainer


def make_weighted_trainer_class(class_weights: torch.Tensor | None, label_smoothing: float):
    """Fabrique une sous-classe de Trainer avec perte pondérée."""
    from transformers import Trainer

    class _WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits  = outputs.logits  # (B, T, C)

            loss_fct = nn.CrossEntropyLoss(
                weight        = class_weights.to(logits.device)
                               if class_weights is not None else None,
                ignore_index  = -100,
                label_smoothing = label_smoothing,
            )
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    return _WeightedTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Métriques seqeval
# ─────────────────────────────────────────────────────────────────────────────

def make_compute_metrics(id2label: dict[int, str]):
    try:
        import evaluate
        seqeval = evaluate.load("seqeval")
    except Exception:
        seqeval = None

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_preds, true_labels = [], []
        for pred_seq, label_seq in zip(predictions, labels):
            pred_row, label_row = [], []
            for p_id, l_id in zip(pred_seq, label_seq):
                if l_id == -100:
                    continue
                pred_row.append(id2label[int(p_id)])
                label_row.append(id2label[int(l_id)])
            true_preds.append(pred_row)
            true_labels.append(label_row)

        if seqeval is not None:
            results = seqeval.compute(predictions=true_preds, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall":    results["overall_recall"],
                "f1":        results["overall_f1"],
                "accuracy":  results["overall_accuracy"],
            }

        correct = sum(
            p == l
            for preds, labs in zip(true_preds, true_labels)
            for p, l in zip(preds, labs)
        )
        total = sum(len(p) for p in true_preds)
        return {"accuracy": correct / total if total else 0.0}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation complète post-entraînement
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    model,
    processor,
    test_ds: LayoutLMv3Dataset,
    id2label: dict[int, str],
    output_dir: Path,
    doc_type: str,
) -> dict:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForTokenClassification

    log.info("Évaluation complète sur le jeu de test...")
    model.eval()
    model.to(DEVICE)

    collator = DataCollatorForTokenClassification(
        processor.tokenizer, pad_to_multiple_of=None
    )
    loader = DataLoader(test_ds, batch_size=4, collate_fn=collator)

    all_true_tokens  = []
    all_pred_tokens  = []
    all_true_seq     = []
    all_pred_seq     = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  inference test", unit="batch"):
            batch   = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits  = outputs.logits
            preds   = torch.argmax(logits, dim=-1).cpu()
            labels  = batch["labels"].cpu()

            for pred_seq, label_seq in zip(preds, labels):
                tok_preds, tok_true = [], []
                seq_preds, seq_true = [], []
                for p_id, l_id in zip(pred_seq.tolist(), label_seq.tolist()):
                    if l_id == -100:
                        continue
                    tok_true.append(l_id)
                    tok_preds.append(p_id)
                    seq_true.append(id2label[l_id])
                    seq_preds.append(id2label[p_id])

                all_true_tokens.extend(tok_true)
                all_pred_tokens.extend(tok_preds)
                all_true_seq.append(seq_true)
                all_pred_seq.append(seq_preds)

    per_entity_metrics: dict = {}
    overall_metrics: dict    = {}

    try:
        import evaluate
        seqeval = evaluate.load("seqeval")
        results = seqeval.compute(predictions=all_pred_seq, references=all_true_seq)

        overall_metrics = {
            "overall_f1":        results["overall_f1"],
            "overall_precision": results["overall_precision"],
            "overall_recall":    results["overall_recall"],
            "overall_accuracy":  results["overall_accuracy"],
        }

        for key, val in results.items():
            if isinstance(val, dict):
                per_entity_metrics[key] = {
                    "f1":        val.get("f1", 0.0),
                    "precision": val.get("precision", 0.0),
                    "recall":    val.get("recall", 0.0),
                    "number":    val.get("number", 0),
                }

        log.info("")
        log.info("╔══════════════════════════════════════════════════════════╗")
        log.info("║   MÉTRIQUES SEQEVAL — %s", doc_type.upper().ljust(34) + "║")
        log.info("╠══════════════════════════════════════════════════════════╣")
        log.info("║  Overall F1        : %.4f                             ║", overall_metrics["overall_f1"])
        log.info("║  Overall Precision : %.4f                             ║", overall_metrics["overall_precision"])
        log.info("║  Overall Recall    : %.4f                             ║", overall_metrics["overall_recall"])
        log.info("║  Overall Accuracy  : %.4f                             ║", overall_metrics["overall_accuracy"])
        log.info("╠══════════════════════════════════════════════════════════╣")
        log.info("║  Métriques par entité (F1 / Precision / Recall)         ║")
        log.info("╠══════════════════════════════════════════════════════════╣")
        for ent, m in sorted(per_entity_metrics.items(), key=lambda x: -x[1]["f1"]):
            log.info("║  %-20s F1=%.3f P=%.3f R=%.3f n=%d",
                     ent[:20], m["f1"], m["precision"], m["recall"], m["number"])
        log.info("╚══════════════════════════════════════════════════════════╝")

    except Exception as e:
        log.warning("seqeval non disponible : %s — fallback token-level", e)
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        acc = accuracy_score(all_true_tokens, all_pred_tokens)
        p, r, f1, _ = precision_recall_fscore_support(
            all_true_tokens, all_pred_tokens, average="weighted", zero_division=0
        )
        overall_metrics = {
            "overall_f1": float(f1),
            "overall_precision": float(p),
            "overall_recall": float(r),
            "overall_accuracy": float(acc),
        }

    try:
        from sklearn.metrics import classification_report

        present_labels = sorted(set(all_true_tokens) | set(all_pred_tokens))
        label_names    = [id2label.get(i, str(i)) for i in present_labels]

        report_str = classification_report(
            all_true_tokens, all_pred_tokens,
            labels=present_labels, target_names=label_names,
            zero_division=0, output_dict=False,
        )
        report_dict = classification_report(
            all_true_tokens, all_pred_tokens,
            labels=present_labels, target_names=label_names,
            zero_division=0, output_dict=True,
        )
        (output_dir / "classification_report.txt").write_text(
            f"Classification Report — {doc_type.upper()} (token-level)\n"
            + "=" * 70 + "\n" + report_str,
            encoding="utf-8",
        )
        log.info("  Classification report → classification_report.txt")

    except ImportError:
        log.warning("scikit-learn non disponible")
        report_dict = {}

    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        entity_labels = [i for i in sorted(set(all_true_tokens) | set(all_pred_tokens))
                         if id2label.get(i, "O") != "O"]

        if entity_labels:
            cm      = confusion_matrix(all_true_tokens, all_pred_tokens, labels=entity_labels)
            cm_norm = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm  = np.divide(cm_norm, row_sums, where=row_sums != 0)

            entity_names = [id2label.get(i, str(i)) for i in entity_labels]
            short_names  = [n.replace("B-", "").replace("I-", "")[:12] for n in entity_names]

            n        = len(entity_labels)
            fig_size = max(12, n * 0.55)
            fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set(
                xticks=np.arange(n), yticks=np.arange(n),
                xticklabels=short_names, yticklabels=short_names,
                xlabel="Prédit", ylabel="Réel",
                title=f"Matrice de confusion (normalisée) — {doc_type.upper()}",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            plt.setp(ax.get_yticklabels(), fontsize=7)

            for i in range(n):
                for j in range(n):
                    val = cm_norm[i, j]
                    if val > 0.05:
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color="white" if val > 0.5 else "black", fontsize=6)

            plt.tight_layout()
            fig.savefig(str(output_dir / "confusion_matrix.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("  Matrice de confusion → confusion_matrix.png")

    except ImportError:
        log.warning("matplotlib ou sklearn non disponible")

    if per_entity_metrics:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sorted_ents = sorted(per_entity_metrics.items(), key=lambda x: -x[1]["f1"])
            ent_names   = [e for e, _ in sorted_ents]
            f1_values   = [m["f1"]        for _, m in sorted_ents]
            prec_values = [m["precision"] for _, m in sorted_ents]
            rec_values  = [m["recall"]    for _, m in sorted_ents]

            x = np.arange(len(ent_names))
            w = 0.25
            fig, ax = plt.subplots(figsize=(max(14, len(ent_names) * 0.6), 6))
            ax.bar(x - w, f1_values,   w, label="F1",       color="#2196F3")
            ax.bar(x,     prec_values, w, label="Precision", color="#4CAF50")
            ax.bar(x + w, rec_values,  w, label="Recall",    color="#FF9800")

            ax.set(
                xticks=x, xticklabels=ent_names, ylim=(0, 1.05),
                ylabel="Score",
                title=f"F1 / Precision / Recall par entité — {doc_type.upper()}",
            )
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
            ax.axhline(y=0.9, color="red", linestyle="--", linewidth=0.8, label="Seuil 0.90")
            ax.legend(loc="lower right")
            ax.grid(axis="y", alpha=0.3)

            for xi, f1 in zip(x - w, f1_values):
                if f1 > 0.01:
                    ax.text(xi, f1 + 0.01, f"{f1:.2f}", ha="center", va="bottom", fontsize=6)

            plt.tight_layout()
            fig.savefig(str(output_dir / "f1_per_entity.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("  F1 par entité → f1_per_entity.png")

        except Exception as e:
            log.warning("Erreur graphique F1 : %s", e)

    full_metrics = {
        "doc_type":           doc_type,
        "overall":            overall_metrics,
        "per_entity":         per_entity_metrics,
        "token_level_report": report_dict,
    }
    (output_dir / "per_entity_metrics.json").write_text(
        json.dumps(full_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("  Métriques complètes → per_entity_metrics.json")

    return full_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning principal
# ─────────────────────────────────────────────────────────────────────────────

def run_finetuning(
    doc_type:   str,
    data_root:  Path,
    output_dir: Path,
    epochs:     int,
    batch:      int,
    lr:         float,
    grad_accum: int,
    seed:       int,
    resume:     bool,
    bf16:       bool,
    gradient_checkpointing: bool,
    fp16:       bool,
    bbox_jitter: int,
    use_class_weights: bool,
) -> None:
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForTokenClassification,
        TrainingArguments,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
    )

    set_seed(seed)

    dataset_dir = data_root / DATASET_DIRS[doc_type]
    if not (dataset_dir / "label2id.json").exists():
        log.error("label2id.json introuvable dans %s", dataset_dir)
        sys.exit(1)

    label2id, id2label = load_label_maps(dataset_dir)
    num_labels = len(label2id)
    log.info("Doc type : %s | Labels : %d", doc_type, num_labels)

    log.info("Chargement processor : %s", MODEL_NAME)
    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)

    log.info("Chargement datasets...")
    train_path = dataset_dir / "train" / "data.jsonl"
    val_path   = dataset_dir / "val"   / "data.jsonl"
    test_path  = dataset_dir / "test"  / "data.jsonl"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            log.error("Dataset manquant : %s", p)
            sys.exit(1)

    train_ds = LayoutLMv3Dataset(
        train_path, processor, label2id,
        augment=True, bbox_jitter=bbox_jitter
    )
    val_ds = LayoutLMv3Dataset(
        val_path, processor, label2id,
        augment=False, bbox_jitter=0
    )
    log.info("Train : %d chunks | Val : %d chunks", len(train_ds), len(val_ds))

    # ← NOUVEAU : calcul des poids de classe
    class_weights = None
    if use_class_weights and train_ds.label_counts:
        class_weights = compute_class_weights(
            train_ds.label_counts, num_labels, o_label_id=0
        )
        log.info("Poids de classe calculés (O=%.3f, entités range=[%.3f, %.3f])",
                 class_weights[0].item(),
                 class_weights[1:].min().item(),
                 class_weights[1:].max().item())

    log.info("Chargement modèle : %s", MODEL_NAME)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels             = num_labels,
        id2label               = {str(k): v for k, v in id2label.items()},
        label2id               = label2id,
        ignore_mismatched_sizes = True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = model.to(DEVICE)
    log.info("Device : %s", DEVICE)

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch,
        per_device_eval_batch_size  = batch,
        learning_rate               = lr,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = "cosine",
        max_grad_norm               = MAX_GRAD_NORM,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = LOGGING_STEPS,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_f1",
        greater_is_better           = True,
        save_total_limit            = 3,
        fp16                        = fp16 and not bf16,
        bf16                        = bf16,
        gradient_accumulation_steps = grad_accum,
        dataloader_num_workers      = 0,
        dataloader_pin_memory       = torch.cuda.is_available(),
        remove_unused_columns       = False,
        seed                        = seed,
        data_seed                   = seed,
        report_to                   = "none",
        logging_dir                 = str(LOGS_DIR / f"tb_{doc_type}"),
        # v2 : label smoothing intégré dans Trainer (complète la perte pondérée)
        label_smoothing_factor      = LABEL_SMOOTHING if class_weights is None else 0.0,
    )

    collator = DataCollatorForTokenClassification(
        processor.tokenizer,
        pad_to_multiple_of = 8 if fp16 else None,
    )

    # ← NOUVEAU : CustomTrainer avec perte pondérée si class_weights disponibles
    TrainerClass = make_weighted_trainer_class(class_weights, LABEL_SMOOTHING) \
        if use_class_weights else None

    if TrainerClass is None:
        from transformers import Trainer as TrainerClass

    trainer = TrainerClass(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collator,
        compute_metrics = make_compute_metrics(id2label),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=7)],  # v2 : 7 (était 5)
    )

    if resume and any(output_dir.glob("checkpoint-*")):
        log.info("Reprise depuis le dernier checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        log.info("Démarrage entraînement (%d epochs, lr=%.1e, batch=%d×%d=%d, jitter=%d)...",
                 epochs, lr, batch, grad_accum, batch * grad_accum, bbox_jitter)
        trainer.train()

    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    log.info("Meilleur modèle → %s", best_dir)

    for fname, content in [
        ("label2id.json", json.dumps(label2id, indent=2)),
        ("id2label.json", json.dumps({str(k): v for k, v in id2label.items()}, indent=2)),
        ("training_config.json", json.dumps({
            "doc_type":                  doc_type,
            "model_name":                MODEL_NAME,
            "epochs":                    epochs,
            "batch_size":                batch,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size":      batch * grad_accum,
            "learning_rate":             lr,
            "lr_scheduler_type":         "cosine",
            "warmup_ratio":              WARMUP_RATIO,
            "weight_decay":              WEIGHT_DECAY,
            "max_grad_norm":             MAX_GRAD_NORM,
            "label_smoothing":           LABEL_SMOOTHING,
            "use_class_weights":         use_class_weights,
            "bbox_jitter":               bbox_jitter,
            "bbox_jitter_mode":          "per_token_independent",
            "seed":                      seed,
            "fp16":                      fp16 and not bf16,
            "bf16":                      bf16,
            "gradient_checkpointing":    gradient_checkpointing,
            "num_labels":                num_labels,
            "early_stopping_patience":   7,
        }, indent=2)),
    ]:
        (best_dir / fname).write_text(content, encoding="utf-8")

    log.info("Évaluation sur test set...")
    test_ds = LayoutLMv3Dataset(test_path, processor, label2id,
                                augment=False, bbox_jitter=0)
    test_results  = trainer.predict(test_ds)
    test_metrics  = test_results.metrics

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   RÉSULTATS — jeu de test [%s]", doc_type.upper().ljust(29) + "║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    for k, v in sorted(test_metrics.items()):
        if isinstance(v, float):
            log.info("║  %-24s : %.4f                     ║", k.replace("test_", ""), v)
    log.info("╚══════════════════════════════════════════════════════════╝")

    (best_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )

    full_evaluation(
        model=model, processor=processor, test_ds=test_ds,
        id2label=id2label, output_dir=best_dir, doc_type=doc_type,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="04_train_per_doctype v2 — Fine-tuning LayoutLMv3 haute précision"
    )
    parser.add_argument("--doc-type", required=True,
                        choices=["bl", "invoice", "packing_list"])
    parser.add_argument("--data-root", type=str,
                        default=str(RELEASE2_ROOT / "data" / "complex"))
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--grad-accum",  type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--output-dir",  type=str,   default=None)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--bbox-jitter", type=int,   default=8,
                        help="Jitter par token (0-1000). Recommandé 8. v2 : per-token indépendant.")
    parser.add_argument("--no-class-weights", dest="class_weights",
                        action="store_false", default=True,
                        help="Désactiver la perte pondérée par classe")
    parser.add_argument("--fp16",     dest="fp16", action="store_true",  default=None)
    parser.add_argument("--no-fp16",  dest="fp16", action="store_false")
    parser.add_argument("--bf16",     action="store_true")
    parser.add_argument("--no-gradient-checkpointing",
                        dest="gradient_checkpointing", action="store_false", default=True)
    args = parser.parse_args()

    global FP16
    FP16 = torch.cuda.is_available() if args.fp16 is None else args.fp16

    if FP16 and not torch.cuda.is_available():
        log.error("--fp16 nécessite un GPU CUDA")
        sys.exit(1)
    if args.bf16 and not torch.cuda.is_available():
        log.error("--bf16 nécessite un GPU CUDA compatible")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else \
        RELEASE2_ROOT / "models" / "complex" / args.doc_type

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  Script 04 v2 — Fine-tuning LayoutLMv3 haute précision  ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("║  Doc type      : %-39s║", args.doc_type)
    log.info("║  Modèle        : %-39s║", MODEL_NAME)
    log.info("║  Epochs        : %-39d║", args.epochs)
    log.info("║  Batch×Accum   : %d×%d = %-33d║",
             args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    log.info("║  LR            : %-39s║", f"{args.lr:.1e}")
    log.info("║  Bbox jitter   : %-39d║", args.bbox_jitter)
    log.info("║  Class weights : %-39s║", str(args.class_weights))
    log.info("║  Label smooth  : %-39s║", str(LABEL_SMOOTHING))
    log.info("║  Device        : %-39s║", str(DEVICE))
    log.info("╚══════════════════════════════════════════════════════════╝")

    run_finetuning(
        doc_type              = args.doc_type,
        data_root             = Path(args.data_root),
        output_dir            = output_dir,
        epochs                = args.epochs,
        batch                 = args.batch_size,
        lr                    = args.lr,
        grad_accum            = args.grad_accum,
        seed                  = args.seed,
        resume                = args.resume,
        bf16                  = args.bf16,
        gradient_checkpointing = args.gradient_checkpointing,
        fp16                  = FP16,
        bbox_jitter           = args.bbox_jitter,
        use_class_weights     = args.class_weights,
    )

    log.info("✓ Entraînement [%s] terminé → %s/best/", args.doc_type.upper(), output_dir)


if __name__ == "__main__":
    main()
