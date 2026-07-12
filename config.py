"""
config.py — Central configuration for SEAD-AI
All paths, hyperparameters, and settings in one place.
Import this file anywhere in the project.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_RAW     = os.path.join(BASE_DIR, "datasets", "raw")
DATASET_PROC    = os.path.join(BASE_DIR, "datasets", "processed")
MODEL_SAVE_DIR  = os.path.join(BASE_DIR, "models", "saved")
DB_PATH         = os.path.join(BASE_DIR, "database", "sead_ai.db")

# ─── BERT Model Settings ──────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"   # HuggingFace model ID
MAX_TOKEN_LEN   = 256                   # max tokens per input text
BATCH_SIZE      = 16                    # training batch size
LEARNING_RATE   = 2e-5                  # standard fine-tuning LR for BERT
EPOCHS          = 3                     # 3 epochs is enough for BERT fine-tune
TEST_SIZE       = 0.2                   # 80% train, 20% test split

# ─── Threat Score Weights ─────────────────────────────────────────────────────
# Final Threat Score = BERT_WEIGHT * bert_score + PSYCH_WEIGHT * psych_score
BERT_WEIGHT     = 0.7
PSYCH_WEIGHT    = 0.3

# ─── Risk Thresholds ─────────────────────────────────────────────────────────
SAFE_THRESHOLD        = 0.35   # below this → Safe
SUSPICIOUS_THRESHOLD  = 0.65   # between safe and this → Suspicious
                               # above this → High Risk

# ─── Flask Settings ───────────────────────────────────────────────────────────
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
FLASK_DEBUG     = True         # set False in production

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
