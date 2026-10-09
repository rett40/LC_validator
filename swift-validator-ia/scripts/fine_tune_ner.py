"""
fine_tune_ner.py — Fine-tuning bert-base-NER sur données SWIFT
==============================================================

Entraîne un modèle NER spécialisé pour les messages SWIFT MT700/MT760.

Labels cibles :
  APPLICANT, BENEFICIARY, ISSUING_BANK, AMOUNT, GOODS,
  INCOTERM, EXPIRY_DATE, ISSUE_DATE, SHIPMENT_DATE

Dataset requis :
  data/data_stati/ner_dataset.jsonl
  Format : {"text": "...", "entities": [{"label": "...", "start": N, "end": M}]}

Usage :
  python scripts/fine_tune_ner.py \
      --dataset-dir data/data_stati \
      --output-dir models/swift-ner \
      --epochs 5 \
      --batch-size 8

Sortie :
  models/swift-ner/   → modèle fine-tuné
  → Configurer dans NLPServiceConfig(transformer_ner_model="models/swift-ner")
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Labels SWIFT → BIO ──────────────────────────────────────────────────
SWIFT_LABELS = [
    "O",
    "B-APPLICANT",    "I-APPLICANT",
    "B-BENEFICIARY",  "I-BENEFICIARY",
    "B-ISSUING_BANK", "I-ISSUING_BANK",
    "B-AMOUNT",       "I-AMOUNT",
    "B-GOODS",        "I-GOODS",
    "B-INCOTERM",     "I-INCOTERM",
    "B-EXPIRY_DATE",  "I-EXPIRY_DATE",
    "B-ISSUE_DATE",   "I-ISSUE_DATE",
    "B-SHIPMENT_DATE","I-SHIPMENT_DATE",
    "B-PORT",         "I-PORT",
]
LABEL2ID = {lbl: i for i, lbl in enumerate(SWIFT_LABELS)}
ID2LABEL = {i: lbl for i, lbl in enumerate(SWIFT_LABELS)}


# ── Conversion spans → BIO tokens ───────────────────────────────────────
def align_labels_with_tokens(tokenizer, text: str, entities: list) -> dict:
    """
    Tokenize le texte et aligne les labels BIO avec les tokens.
    Gère le subword tokenization (wordpiece) de BERT.
    """
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding["offset_mapping"]
    
    # Construire une liste de labels par caractère
    char_labels = ["O"] * len(text)
    for ent in entities:
        label = ent.get("label", "")
        start = ent.get("start")
        end   = ent.get("end")
        if start is None or end is None or label not in [
            "APPLICANT","BENEFICIARY","ISSUING_BANK","AMOUNT",
            "GOODS","INCOTERM","EXPIRY_DATE","ISSUE_DATE","SHIPMENT_DATE","PORT"
        ]:
            continue
        end = min(end, len(text))
        char_labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            char_labels[i] = f"I-{label}"
    
    # Aligner avec les tokens
    token_labels = []
    for (tok_start, tok_end) in offset_mapping:
        if tok_start == tok_end:  # [CLS], [SEP], padding
            token_labels.append(-100)  # ignoré dans la loss
        else:
            char_lbl = char_labels[tok_start]
            token_labels.append(LABEL2ID.get(char_lbl, 0))
    
    encoding["labels"] = token_labels
    return encoding


# ── Dataset PyTorch ──────────────────────────────────────────────────────
class SwiftNERDataset:
    def __init__(self, records: list, tokenizer):
        self.data = []
        for rec in records:
            text     = rec.get("text", "")
            entities = rec.get("entities", [])
            if not text:
                continue
            enc = align_labels_with_tokens(tokenizer, text, entities)
            self.data.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels":         enc["labels"],
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        import torch
        item = self.data[idx]
        return {k: torch.tensor(v) for k, v in item.items()}


# ── Métriques seqeval ────────────────────────────────────────────────────
def compute_metrics(p, id2label):
    try:
        from seqeval.metrics import (
            classification_report, f1_score, precision_score, recall_score
        )
    except ImportError:
        logger.warning("seqeval non installé — pip install seqeval")
        import numpy as np
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Retourner des métriques vides mais valides
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    import numpy as np
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [id2label[pred] for pred, lbl in zip(row_pred, row_lbl) if lbl != -100]
        for row_pred, row_lbl in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[lbl] for lbl in row_lbl if lbl != -100]
        for row_lbl in labels
    ]
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall":    recall_score(true_labels, true_predictions),
        "f1":        f1_score(true_labels, true_predictions),
        "report":    classification_report(true_labels, true_predictions),
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune NER SWIFT")
    parser.add_argument("--dataset-dir",  required=True)
    parser.add_argument("--dataset-file", default="ner_dataset.jsonl",
                        help="Nom du fichier jsonl dans --dataset-dir")
    parser.add_argument("--output-dir",   default="./models/swift-ner")
    parser.add_argument("--base-model",   default="dslim/bert-base-NER",
                        help="Modèle de base HuggingFace")
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--max-samples",  type=int,   default=None,
                        help="Limiter le nombre d'exemples (debug)")
    parser.add_argument("--test-split",   type=float, default=0.15)
    args = parser.parse_args()
    
    # Vérifier les dépendances
    try:
        from transformers import (
            AutoTokenizer, AutoModelForTokenClassification,
            TrainingArguments, Trainer, DataCollatorForTokenClassification
        )
        import transformers
        import torch
        logger.info("transformers %s | torch %s | device %s",
                    transformers.__version__, torch.__version__,
                    "cuda" if torch.cuda.is_available() else "cpu")
    except ImportError as e:
        logger.error("Dépendances manquantes : pip install transformers torch seqeval")
        logger.error(str(e))
        sys.exit(1)
    
    # Charger le dataset
    ner_path = Path(args.dataset_dir) / args.dataset_file
    if not ner_path.exists():
        logger.error("Dataset NER introuvable : %s", ner_path)
        sys.exit(1)
    
    records = []
    with open(ner_path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    
    if args.max_samples:
        records = records[:args.max_samples]
    
    logger.info("Dataset : %d textes SWIFT chargés", len(records))
    
    # Mélanger et splitter
    import random
    random.seed(42)
    random.shuffle(records)
    split = int(len(records) * (1 - args.test_split))
    train_records = records[:split]
    test_records  = records[split:]
    logger.info("Train : %d  |  Test : %d", len(train_records), len(test_records))
    
    # Tokenizer
    logger.info("Chargement tokenizer : %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Datasets
    logger.info("Tokenization et alignement BIO...")
    train_dataset = SwiftNERDataset(train_records, tokenizer)
    test_dataset  = SwiftNERDataset(test_records,  tokenizer)
    logger.info("Train tokens : %d exemples", len(train_dataset))
    
    # Modèle
    logger.info("Chargement modèle base : %s", args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(SWIFT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # la tête NER change de taille
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device : %s", device)
    model.to(device)
    
    # Data collator (padding dynamique)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Arguments d'entraînement
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Détecter la version de transformers pour choisir le bon argument
    import transformers as _tf
    _tf_version = tuple(int(x) for x in _tf.__version__.split(".")[:2])
    _eval_strat_key = "eval_strategy" if _tf_version >= (4, 46) else "evaluation_strategy"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        **{_eval_strat_key: "epoch"},
        save_strategy="epoch",
        save_total_limit=2,           # Garder seulement les 2 meilleurs checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Trainer
    def _compute_metrics(p):
        return compute_metrics(p, ID2LABEL)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )
    
    # Entraînement
    logger.info("=" * 60)
    logger.info("Début du fine-tuning NER SWIFT")
    logger.info("  Labels   : %d", len(SWIFT_LABELS))
    logger.info("  Epochs   : %d", args.epochs)
    logger.info("  LR       : %s", args.lr)
    logger.info("  Batch    : %d", args.batch_size)
    logger.info("=" * 60)
    
    trainer.train()
    
    # Évaluation finale
    logger.info("Évaluation finale...")
    results = trainer.evaluate()
    logger.info("F1    : %.3f", results.get("eval_f1", 0))
    logger.info("Prec. : %.3f", results.get("eval_precision", 0))
    logger.info("Recall: %.3f", results.get("eval_recall", 0))
    if "eval_report" in results:
        print("\n" + results["eval_report"])
    
    # Sauvegarde
    logger.info("Sauvegarde dans %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Sauvegarder la config des labels
    with open(output_dir / "swift_ner_labels.json", "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": {str(k): v for k,v in ID2LABEL.items()}}, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Fine-tuning terminé !")
    logger.info("")
    logger.info("Pour utiliser dans le pipeline :")
    logger.info("  config = NLPServiceConfig(")
    logger.info("      transformer_ner_model=r'%s',", output_dir)
    logger.info("  )")
    logger.info("  service = SwiftNLPService(config)")


if __name__ == "__main__":
    main()