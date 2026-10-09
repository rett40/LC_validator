"""
05_train_classifier.py
======================
Étape 5 — Classification visuelle de documents (sans OCR).

Modèle : ResNet-18 pré-entraîné (ImageNet) + fine-tuning tête de classification
Entrée : data/processed/{doc_type}/*_page1.png  (1200 images)
Sortie : models/doc_classifier/
           best_model.pt       ← meilleurs poids
           class_names.json    ← {0:"bl", 1:"invoice", 2:"packing_list"}
           metrics.json        ← accuracy / rapport de classification

Usage :
  python scripts/05_train_classifier.py
  python scripts/05_train_classifier.py --epochs 20 --batch-size 16
"""

from __future__ import annotations

import json
import logging
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────

RELEASE2_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT     = RELEASE2_ROOT / "data"
PROC_ROOT     = DATA_ROOT / "processed"
MODELS_DIR    = RELEASE2_ROOT / "models" / "doc_classifier"
LOGS_DIR      = RELEASE2_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "05_train_classifier.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

CLASS_NAMES   = ["bl", "invoice", "packing_list"]
IMG_SIZE      = 224          # ResNet-18 input standard
DEFAULT_EPOCHS = 15
DEFAULT_BATCH  = 16
DEFAULT_LR     = 1e-4
FREEZE_EPOCHS  = 3           # geler le backbone les N premières epochs
RANDOM_SEED    = 42
SPLIT_VAL      = 0.15
SPLIT_TEST     = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DocImageDataset:
    """
    Dataset simple : charge les images page1 et retourne (tensor, label).
    """

    def __init__(self, samples: list[tuple[Path, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def collect_samples(
    proc_root: Path,
    class_names: list[str],
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    """Collecte les images et split train/val/test par doc_id."""
    all_by_class: dict[int, list[Path]] = defaultdict(list)

    for cls_idx, cls_name in enumerate(class_names):
        imgs = sorted((proc_root / cls_name).glob("*_page1.png"))
        log.info("  %-15s : %d images", cls_name, len(imgs))
        all_by_class[cls_idx].extend(imgs)

    train_samples, val_samples, test_samples = [], [], []
    rng = random.Random(seed)

    for cls_idx, paths in all_by_class.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n        = len(shuffled)
        n_test   = max(1, int(n * SPLIT_TEST))
        n_val    = max(1, int(n * SPLIT_VAL))

        test_samples  += [(p, cls_idx) for p in shuffled[:n_test]]
        val_samples   += [(p, cls_idx) for p in shuffled[n_test: n_test + n_val]]
        train_samples += [(p, cls_idx) for p in shuffled[n_test + n_val:]]

    rng.shuffle(train_samples)
    return train_samples, val_samples, test_samples


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(train: bool):
    """Transforms standard ImageNet + augmentations légères pour le train."""
    from torchvision import transforms

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 48, IMG_SIZE + 48)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomRotation(degrees=3),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.4),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            transforms.RandomGrayscale(p=0.08),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Modèle ResNet-18
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, freeze_backbone: bool = True):
    """ResNet-18 pré-entraîné avec nouvelle tête de classification."""
    import torch
    import torch.nn as nn
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Remplacer la tête de classification
    in_features     = model.fc.in_features
    model.fc        = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )
    return model


def unfreeze_backbone(model) -> None:
    """Dégèle tous les paramètres pour le fine-tuning complet."""
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    import torch
    model.train()
    total_loss = 0.0
    correct    = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad() if False else lambda f: f   # annotation compatibilité
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    import torch
    model.eval()
    total_loss = 0.0
    correct    = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)
            total_loss  += loss.item() * imgs.size(0)
            correct     += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ─────────────────────────────────────────────────────────────────────────────

def run_training(epochs: int, batch: int, lr: float) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)

    # 1. Données
    log.info("Collecte des images...")
    train_s, val_s, test_s = collect_samples(PROC_ROOT, CLASS_NAMES)
    log.info("Train : %d | Val : %d | Test : %d", len(train_s), len(val_s), len(test_s))

    train_ds = DocImageDataset(train_s, get_transforms(train=True))
    val_ds   = DocImageDataset(val_s,   get_transforms(train=False))
    test_ds  = DocImageDataset(test_s,  get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, num_workers=0)

    # 2. Modèle
    model     = build_model(num_classes=len(CLASS_NAMES), freeze_backbone=True)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimiseur initial : seulement la tête
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history      = []

    log.info("Démarrage entraînement (%d epochs, batch=%d, lr=%.0e)...", epochs, batch, lr)

    for epoch in range(1, epochs + 1):
        # Dégeler le backbone après FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            log.info("Epoch %d — Dégel du backbone ResNet-18", epoch)
            unfreeze_backbone(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - FREEZE_EPOCHS
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
        })

        log.info(
            "Epoch %2d/%d | train_loss=%.4f | train_acc=%.3f | val_acc=%.3f%s",
            epoch, epochs, train_loss, train_acc, val_acc,
            "  ← best" if val_acc > best_val_acc else ""
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")

    # 3. Évaluation test
    log.info("Chargement du meilleur modèle pour évaluation test...")
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Rapport détaillé par classe
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    log.info("")
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   RÉSULTATS — jeu de test                       ║")
    log.info("╠══════════════════════════════════════════════════╣")
    log.info("║  Test accuracy : %.4f                          ║", test_acc)
    log.info("╠══════════════════════════════════════════════════╣")
    for cls in CLASS_NAMES:
        m = report[cls]
        log.info("║  %-15s  P=%.2f  R=%.2f  F1=%.2f          ║",
                 cls, m["precision"], m["recall"], m["f1-score"])
    log.info("╚══════════════════════════════════════════════════╝")

    log.info("\nMatrice de confusion (%s) :", ", ".join(CLASS_NAMES))
    for i, row in enumerate(cm):
        log.info("  %-15s : %s", CLASS_NAMES[i], row)

    # 4. Sauvegardes
    (MODELS_DIR / "class_names.json").write_text(
        json.dumps(CLASS_NAMES, indent=2), encoding="utf-8"
    )
    metrics = {
        "test_accuracy":    test_acc,
        "best_val_accuracy": best_val_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "history": history,
    }
    (MODELS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Modèle sauvegardé → %s", MODELS_DIR / "best_model.pt")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="05_train_classifier — classification visuelle de documents (ResNet-18)"
    )
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  Script 05 — Classifieur visuel de documents        ║")
    log.info("║  Modèle : ResNet-18  (transfer learning ImageNet)   ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info("  Classes  : %s", CLASS_NAMES)
    log.info("  Epochs   : %d", args.epochs)
    log.info("  Batch    : %d", args.batch_size)
    log.info("  LR       : %.0e", args.lr)
    log.info("  Output   : %s", MODELS_DIR)

    run_training(args.epochs, args.batch_size, args.lr)
    log.info("\nProchaine étape : utiliser services/classifier_service.py pour l'inférence")


if __name__ == "__main__":
    main()
