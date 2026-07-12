"""
predictor/predict.py — Phase 5: BERT Prediction Module for SEAD-AI
====================================================================
This script:
  1. Loads the saved fine-tuned BERT model
  2. Accepts any text input
  3. Returns:
     - phishing probability (0.0 to 1.0)
     - confidence score
     - prediction label (Phishing / Legitimate)

Usage (standalone test):
    python predictor/predict.py

Used by:
    predictor/threat_score.py  (Phase 7)
    app.py                     (Phase 10)
"""

import os
import sys

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# ── Make config importable ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_SAVE_DIR, MAX_TOKEN_LEN
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Path to saved BERT model ───────────────────────────────────────────────────
BERT_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "bert_model")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class BERTPredictor:
    """
    Singleton-style BERT predictor.
    Loads model once and reuses it for all predictions.

    Why singleton?
    → Loading BERT takes ~3 seconds and ~440MB RAM.
      We load it once at app startup and reuse the same
      instance for every API request.
    """

    _instance = None   # class-level cache

    def __new__(cls):
        """Ensures only one instance is ever created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return   # already loaded — skip

        logger.info("Loading BERT model for inference...")

        # Check model exists
        if not os.path.exists(BERT_MODEL_PATH):
            raise FileNotFoundError(
                f"BERT model not found at: {BERT_MODEL_PATH}\n"
                "Run Phase 4 first: python models/train_bert.py"
            )

        # Auto-select device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Load tokenizer + model from saved directory
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        self.model     = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_PATH
        )
        self.model.to(self.device)
        self.model.eval()   # set to evaluation mode (disables dropout)

        self._initialized = True
        logger.info("BERT model loaded successfully ✅")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict(text: str) -> dict:
    """
    Main prediction function.
    Takes raw text input and returns full prediction result.

    Args:
        text (str): Raw email or SMS text to analyze

    Returns:
        dict: {
            "label"          : "Phishing" or "Legitimate",
            "phishing_prob"  : float (0.0 to 1.0),
            "legitimate_prob": float (0.0 to 1.0),
            "confidence"     : float (0.0 to 1.0),
            "bert_score"     : float (0.0 to 1.0),  ← used by threat engine
        }
    """

    if not text or len(text.strip()) == 0:
        logger.warning("Empty text received — returning safe default")
        return _default_result()

    # Load model (cached after first call)
    predictor = BERTPredictor()

    # ── Tokenize input ───────────────────────────────────────────
    # Same tokenization as training — must match exactly
    encoding = predictor.tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length         = MAX_TOKEN_LEN,
        padding            = "max_length",
        truncation         = True,
        return_tensors     = "pt",
    )

    input_ids      = encoding["input_ids"].to(predictor.device)
    attention_mask = encoding["attention_mask"].to(predictor.device)

    # ── Inference ────────────────────────────────────────────────
    # torch.no_grad() prevents gradient computation during inference
    # (saves memory and speeds up by ~2x)
    with torch.no_grad():
        outputs = predictor.model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

    # ── Convert logits → probabilities ───────────────────────────
    # Softmax converts raw scores to probabilities that sum to 1.0
    # logits shape: [1, 2] → [legitimate_score, phishing_score]
    probs = F.softmax(outputs.logits, dim=1).squeeze(0)

    legitimate_prob = round(probs[0].item(), 4)
    phishing_prob   = round(probs[1].item(), 4)

    # Predicted class: 1 = phishing, 0 = legitimate
    predicted_class = torch.argmax(probs).item()
    label           = "Phishing" if predicted_class == 1 else "Legitimate"

    # Confidence = probability of the predicted class
    confidence = phishing_prob if predicted_class == 1 else legitimate_prob

    result = {
        "label"          : label,
        "phishing_prob"  : phishing_prob,
        "legitimate_prob": legitimate_prob,
        "confidence"     : round(confidence, 4),
        "bert_score"     : phishing_prob,   # alias used by threat engine
    }

    logger.info(
        f"Prediction: {label} | "
        f"Phishing: {phishing_prob:.2%} | "
        f"Confidence: {confidence:.2%}"
    )

    return result


def _default_result() -> dict:
    """Returns a safe default when input is empty or invalid."""
    return {
        "label"          : "Legitimate",
        "phishing_prob"  : 0.0,
        "legitimate_prob": 1.0,
        "confidence"     : 1.0,
        "bert_score"     : 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BATCH PREDICTION (for testing multiple messages at once)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_batch(texts: list) -> list:
    """
    Predicts for a list of texts.
    Returns a list of result dicts.
    Useful for bulk testing.
    """
    return [predict(text) for text in texts]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_test():
    """
    Tests the predictor on sample phishing and legitimate messages.
    Run with: python predictor/predict.py
    """
    test_messages = [
        {
            "text" : "URGENT: Your bank account has been suspended! "
                     "Click here immediately to verify your identity.",
            "expected": "Phishing",
        },
        {
            "text" : "Hi team, the meeting is rescheduled to 3pm today. "
                     "Please update your calendars.",
            "expected": "Legitimate",
        },
        {
            "text" : "Congratulations! You've won a FREE iPhone. "
                     "Claim your prize NOW before it expires!",
            "expected": "Phishing",
        },
        {
            "text" : "Your electricity bill of Rs. 1,240 is due on "
                     "November 30. Pay online to avoid late fees.",
            "expected": "Legitimate",
        },
        {
            "text" : "CEO Request: Purchase $500 iTunes gift cards urgently "
                     "for a client. Keep this confidential.",
            "expected": "Phishing",
        },
    ]

    print("\n" + "="*60)
    print("  SEAD-AI — Phase 5: BERT Prediction Module Test")
    print("="*60)

    correct = 0
    for i, item in enumerate(test_messages, 1):
        result = predict(item["text"])
        status = "✅" if result["label"] == item["expected"] else "❌"
        if result["label"] == item["expected"]:
            correct += 1

        print(f"\n  Test {i}: {status}")
        print(f"  Text      : {item['text'][:60]}...")
        print(f"  Expected  : {item['expected']}")
        print(f"  Predicted : {result['label']}")
        print(f"  Phishing% : {result['phishing_prob']:.2%}")
        print(f"  Confidence: {result['confidence']:.2%}")

    accuracy = (correct / len(test_messages)) * 100
    print(f"\n{'='*60}")
    print(f"  Test Accuracy: {correct}/{len(test_messages)} = {accuracy:.1f}%")
    print(f"  Model path   : {BERT_MODEL_PATH}")
    print(f"{'='*60}")
    print(f"  Next → Start Phase 6: Psychology Scorer")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_test()
