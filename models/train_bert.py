"""
models/train_bert.py — Phase 4: Fine-tune BERT for Social Engineering Detection
=================================================================================
This script:
  1. Loads preprocessed train/test datasets
  2. Tokenizes text using BERT tokenizer
  3. Creates PyTorch DataLoaders
  4. Fine-tunes bert-base-uncased for binary classification
  5. Evaluates: Accuracy, Precision, Recall, F1
  6. Saves the trained model + tokenizer

Usage:
    python models/train_bert.py

Output:
    models/saved/bert_model/         ← trained model weights
    models/saved/training_stats.json ← training history
"""

import os
import sys
import json
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ── Make config importable ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATASET_PROC,
    MODEL_SAVE_DIR,
    BERT_MODEL_NAME,
    MAX_TOKEN_LEN,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DEVICE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """
    Automatically picks GPU if available, otherwise CPU.
    GPU trains ~10x faster — but CPU works fine for sample data.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found — using CPU (slower but works fine)")
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class PhishingDataset(Dataset):
    """
    PyTorch Dataset for phishing/legitimate text classification.
    Tokenizes text using BERT tokenizer on-the-fly.

    Why custom Dataset?
    → PyTorch requires this structure to use DataLoader
      which handles batching, shuffling, and memory efficiency.
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text  = str(self.texts[idx])
        label = self.labels[idx]

        # BERT tokenization:
        # - adds [CLS] at start and [SEP] at end automatically
        # - pads/truncates to max_len
        # - returns input_ids, attention_mask, token_type_ids
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length         = self.max_len,
            padding            = "max_length",
            truncation         = True,
            return_tensors     = "pt",
        )

        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(label, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(tokenizer) -> tuple:
    """
    Loads train.csv and test.csv, creates PyTorch DataLoaders.
    Returns: (train_loader, test_loader, test_df)
    """
    train_path = os.path.join(DATASET_PROC, "train.csv")
    test_path  = os.path.join(DATASET_PROC, "test.csv")

    # Check files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("train.csv or test.csv not found!")
        logger.error("Run Phase 3 first: python preprocessing/clean.py")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    logger.info(f"Train set: {len(train_df)} rows")
    logger.info(f"Test set:  {len(test_df)} rows")

    # Create Dataset objects
    train_dataset = PhishingDataset(
        texts     = train_df["text"].tolist(),
        labels    = train_df["label"].tolist(),
        tokenizer = tokenizer,
        max_len   = MAX_TOKEN_LEN,
    )
    test_dataset = PhishingDataset(
        texts     = test_df["text"].tolist(),
        labels    = test_df["label"].tolist(),
        tokenizer = tokenizer,
        max_len   = MAX_TOKEN_LEN,
    )

    # DataLoaders handle batching automatically
    # num_workers=0 is safest for Windows compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = True,
        num_workers= 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle    = False,
        num_workers= 0,
    )

    return train_loader, test_loader, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    """
    Trains the model for one full epoch.
    Returns average training loss for this epoch.
    """
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(loader):
        # Move all tensors to device (GPU or CPU)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        # Zero gradients before each batch
        # (PyTorch accumulates gradients by default)
        optimizer.zero_grad()

        # Forward pass — BERT returns loss + logits
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass — compute gradients
        loss.backward()

        # Clip gradients to prevent exploding gradients
        # (common practice for transformer fine-tuning)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()
        scheduler.step()

        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            logger.info(
                f"  Batch {batch_idx+1}/{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device) -> dict:
    """
    Evaluates model on a DataLoader.
    Returns dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    # No gradient computation during evaluation (saves memory + speed)
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )

            # Get predicted class (0 or 1) from logits
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    metrics = {
        "accuracy"  : round(accuracy_score(all_labels, all_preds) * 100, 2),
        "precision" : round(precision_score(all_labels, all_preds,
                            zero_division=0) * 100, 2),
        "recall"    : round(recall_score(all_labels, all_preds,
                            zero_division=0) * 100, 2),
        "f1"        : round(f1_score(all_labels, all_preds,
                            zero_division=0) * 100, 2),
        "confusion_matrix": confusion_matrix(
                            all_labels, all_preds).tolist(),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SAVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(model, tokenizer, stats: dict) -> str:
    """
    Saves the fine-tuned BERT model + tokenizer to disk.
    HuggingFace's save_pretrained() saves everything needed for inference.
    Returns the save path.
    """
    save_path = os.path.join(MODEL_SAVE_DIR, "bert_model")
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save training stats alongside model
    stats_path = os.path.join(MODEL_SAVE_DIR, "training_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Stats saved to: {stats_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(stats: dict) -> None:
    """Prints a clean training summary."""
    print("\n" + "="*55)
    print("  SEAD-AI — Phase 4: BERT Training Complete ✅")
    print("="*55)
    print(f"  Epochs trained   : {stats['epochs_trained']}")
    print(f"  Total time       : {stats['total_time_mins']} mins")
    print()
    print("  ── Final Test Metrics ──────────────────────")
    m = stats["final_metrics"]
    print(f"  Accuracy         : {m['accuracy']}%")
    print(f"  Precision        : {m['precision']}%")
    print(f"  Recall           : {m['recall']}%")
    print(f"  F1 Score         : {m['f1']}%")
    print()
    print("  ── Confusion Matrix ────────────────────────")
    cm = m["confusion_matrix"]
    print(f"  True Negatives   : {cm[0][0]}  (Legit → Legit ✅)")
    print(f"  False Positives  : {cm[0][1]}  (Legit → Phishing ❌)")
    print(f"  False Negatives  : {cm[1][0]}  (Phishing → Legit ❌)")
    print(f"  True Positives   : {cm[1][1]}  (Phishing → Phishing ✅)")
    print()
    print("  Model saved at: models/saved/bert_model/")
    print("="*55)
    print("  Next → Run: python predictor/predict.py  (Phase 5)")
    print("="*55 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 55)
    logger.info("SEAD-AI — Phase 4: BERT Fine-Tuning")
    logger.info("=" * 55)

    start_time = time.time()
    device     = get_device()

    # ── Load tokenizer ───────────────────────────────────────────
    logger.info(f"Loading tokenizer: {BERT_MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    # ── Load data ────────────────────────────────────────────────
    train_loader, test_loader, _ = load_data(tokenizer)

    # ── Load BERT model ──────────────────────────────────────────
    # num_labels=2 → binary classification (phishing vs legitimate)
    logger.info(f"Loading BERT model: {BERT_MODEL_NAME}")
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels          = 2,
        output_attentions   = False,
        output_hidden_states= False,
    )
    model.to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────
    # AdamW is the standard optimizer for BERT fine-tuning
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

    # Linear warmup scheduler — gradually increases LR at start
    # then linearly decays. Prevents large early updates from
    # destroying pre-trained BERT weights.
    total_steps    = len(train_loader) * EPOCHS
    warmup_steps   = int(0.1 * total_steps)
    scheduler      = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── Training Loop ────────────────────────────────────────────
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"\n{'─'*45}")
        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"{'─'*45}")

        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )

        # Evaluate on test set
        metrics = evaluate(model, test_loader, device)

        epoch_mins = round((time.time() - epoch_start) / 60, 2)

        logger.info(f"Epoch {epoch} Results:")
        logger.info(f"  Train Loss : {train_loss:.4f}")
        logger.info(f"  Accuracy   : {metrics['accuracy']}%")
        logger.info(f"  F1 Score   : {metrics['f1']}%")
        logger.info(f"  Time       : {epoch_mins} mins")

        training_history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 4),
            "metrics"    : metrics,
            "time_mins"  : epoch_mins,
        })

    # ── Save model ───────────────────────────────────────────────
    total_mins = round((time.time() - start_time) / 60, 2)

    stats = {
        "epochs_trained"  : EPOCHS,
        "total_time_mins" : total_mins,
        "model_name"      : BERT_MODEL_NAME,
        "final_metrics"   : training_history[-1]["metrics"],
        "history"         : training_history,
    }

    save_model(model, tokenizer, stats)
    print_summary(stats)


if __name__ == "__main__":
    main()
