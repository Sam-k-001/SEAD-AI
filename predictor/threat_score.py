"""
predictor/threat_score.py — Phase 7: Threat Score Engine for SEAD-AI
======================================================================
Combines BERT probability + Psychology score into one final threat score.

Formula:
    Threat Score = 0.7 × BERT Score + 0.3 × Psychology Score

Risk Levels:
    Safe        → Threat Score < 0.35
    Suspicious  → Threat Score 0.35 – 0.65
    High Risk   → Threat Score > 0.65

Usage:
    from predictor.threat_score import calculate_threat_score
    result = calculate_threat_score("Your account is suspended! Act now!")

Returns:
    dict with threat_score, risk_level, bert_score, psychology_score,
    risk_color, risk_emoji, and full psychology details
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BERT_WEIGHT, PSYCH_WEIGHT, SAFE_THRESHOLD, SUSPICIOUS_THRESHOLD
from psychology.scorer import score_psychology
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RISK LEVEL MAPPER
# ═══════════════════════════════════════════════════════════════════════════════

def get_risk_level(threat_score: float) -> dict:
    """
    Maps a threat score (0.0–1.0) to a risk level with
    color and emoji for the frontend UI.

    Args:
        threat_score: float between 0.0 and 1.0

    Returns:
        dict: { level, color, emoji, description }
    """
    if threat_score >= SUSPICIOUS_THRESHOLD:
        return {
            "level"      : "High Risk",
            "color"      : "#FF3B30",   # red
            "emoji"      : "🚨",
            "description": "This message shows strong signs of a social "
                           "engineering attack. Do NOT click any links or "
                           "provide any information.",
        }
    elif threat_score >= SAFE_THRESHOLD:
        return {
            "level"      : "Suspicious",
            "color"      : "#FF9500",   # orange
            "emoji"      : "⚠️",
            "description": "This message has some suspicious characteristics. "
                           "Proceed with caution and verify the sender "
                           "through official channels.",
        }
    else:
        return {
            "level"      : "Safe",
            "color"      : "#34C759",   # green
            "emoji"      : "✅",
            "description": "This message appears to be legitimate. "
                           "No significant social engineering patterns detected.",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MAIN THREAT SCORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_threat_score(text: str, bert_result: dict = None) -> dict:
    """
    Calculates the final combined threat score.

    Why separate bert_result parameter?
    → In the Flask API (Phase 10), BERT prediction and psychology
      scoring run in parallel. We pass the BERT result in directly
      to avoid running it twice.

    Args:
        text        : Raw input text to analyze
        bert_result : Optional pre-computed BERT result dict.
                      If None, runs psychology only (useful for testing
                      Phase 7 before BERT model is available).

    Returns:
        dict: {
            "threat_score"     : float (0.0–1.0),
            "threat_percent"   : int (0–100),
            "risk_level"       : str,
            "risk_color"       : str,
            "risk_emoji"       : str,
            "risk_description" : str,
            "bert_score"       : float,
            "psychology_score" : float,
            "bert_weight"      : float,
            "psych_weight"     : float,
            "psychology_detail": dict,
            "formula"          : str,
        }
    """

    # ── Step 1: Get BERT score ────────────────────────────────────
    if bert_result is not None:
        bert_score = float(bert_result.get("bert_score", 0.0))
    else:
        # BERT not available yet — use 0.0 (psychology only mode)
        # This lets us test Phase 7 independently
        bert_score = 0.0
        logger.warning(
            "No BERT result provided — running psychology-only mode. "
            "Pass bert_result for full threat scoring."
        )

    # ── Step 2: Get Psychology score ──────────────────────────────
    psych_result = score_psychology(text)
    psych_score  = float(psych_result.get("psychology_score", 0.0))

    # ── Step 3: Apply threat formula ─────────────────────────────
    # Threat Score = 0.7 × BERT + 0.3 × Psychology
    # If BERT not available: use psychology score with higher weight
    if bert_result is not None:
        threat_score = (BERT_WEIGHT * bert_score) + (PSYCH_WEIGHT * psych_score)
    else:
        # Psychology-only fallback (for testing)
        threat_score = psych_score

    # Clamp to [0.0, 1.0]
    threat_score = max(0.0, min(1.0, threat_score))
    threat_score = round(threat_score, 4)

    # ── Step 4: Get risk level ────────────────────────────────────
    risk = get_risk_level(threat_score)

    # ── Step 5: Build formula string (for explainability) ────────
    if bert_result is not None:
        formula = (
            f"{BERT_WEIGHT} × {bert_score:.2f} (BERT) + "
            f"{PSYCH_WEIGHT} × {psych_score:.2f} (Psychology) = "
            f"{threat_score:.2f}"
        )
    else:
        formula = f"Psychology only: {psych_score:.2f} (BERT not available)"

    logger.info(
        f"Threat Score: {threat_score:.2%} | "
        f"Risk: {risk['level']} | "
        f"BERT: {bert_score:.2%} | "
        f"Psych: {psych_score:.2%}"
    )

    return {
        "threat_score"     : threat_score,
        "threat_percent"   : int(threat_score * 100),
        "risk_level"       : risk["level"],
        "risk_color"       : risk["color"],
        "risk_emoji"       : risk["emoji"],
        "risk_description" : risk["description"],
        "bert_score"       : round(bert_score, 4),
        "psychology_score" : round(psych_score, 4),
        "bert_weight"      : BERT_WEIGHT,
        "psych_weight"     : PSYCH_WEIGHT,
        "psychology_detail": psych_result,
        "formula"          : formula,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FULL PIPELINE (BERT + PSYCHOLOGY TOGETHER)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(text: str) -> dict:
    """
    Full analysis pipeline: BERT + Psychology → Threat Score.
    This is the main function called by Flask in Phase 10.

    Tries to load BERT model — if not available, falls back
    to psychology-only mode gracefully.

    Args:
        text: Raw input text

    Returns:
        Combined threat analysis result dict
    """
    bert_result = None

    # Try to run BERT prediction
    try:
        from predictor.predict import predict
        bert_result = predict(text)
        logger.info(f"BERT prediction: {bert_result['label']} "
                    f"({bert_result['phishing_prob']:.2%})")
    except FileNotFoundError:
        logger.warning("BERT model not found — using psychology-only mode")
    except Exception as e:
        logger.warning(f"BERT prediction failed: {e} — using psychology-only mode")

    # Calculate final threat score
    result = calculate_threat_score(text, bert_result)

    # Add BERT label if available
    if bert_result:
        result["bert_label"]      = bert_result.get("label", "Unknown")
        result["bert_confidence"] = bert_result.get("confidence", 0.0)
    else:
        result["bert_label"]      = "Unavailable"
        result["bert_confidence"] = 0.0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_test():
    """Tests the threat score engine with sample messages."""

    test_cases = [
        {
            "text"    : "URGENT: Your bank account has been suspended! "
                        "Click here immediately to verify your identity "
                        "or your account will be permanently deleted within 24 hours.",
            "expected": "High Risk",
            # Simulated BERT result (as if model predicted phishing)
            "bert"    : {"bert_score": 0.92, "label": "Phishing",
                         "phishing_prob": 0.92, "confidence": 0.92},
        },
        {
            "text"    : "Congratulations! You have won a FREE iPhone 15! "
                        "Limited time offer — claim your prize NOW "
                        "before it expires!",
            "expected": "High Risk",
            "bert"    : {"bert_score": 0.88, "label": "Phishing",
                         "phishing_prob": 0.88, "confidence": 0.88},
        },
        {
            "text"    : "Hi Sarah, can we reschedule the 2pm call to 3pm today? "
                        "Something came up on my end. Thanks!",
            "expected": "Safe",
            "bert"    : {"bert_score": 0.05, "label": "Legitimate",
                         "phishing_prob": 0.05, "confidence": 0.95},
        },
        {
            "text"    : "IRS Notice: You owe back taxes. Failure to respond "
                        "in 24 hours will result in immediate arrest. "
                        "This is your final notice.",
            "expected": "High Risk",
            "bert"    : {"bert_score": 0.95, "label": "Phishing",
                         "phishing_prob": 0.95, "confidence": 0.95},
        },
        {
            "text"    : "Your Amazon order #45892 has been shipped and will "
                        "arrive within 3-5 business days.",
            "expected": "Safe",
            "bert"    : {"bert_score": 0.08, "label": "Legitimate",
                         "phishing_prob": 0.08, "confidence": 0.92},
        },
    ]

    print("\n" + "="*65)
    print("  SEAD-AI — Phase 7: Threat Score Engine Test")
    print("="*65)
    print(f"  Formula: {BERT_WEIGHT} × BERT + {PSYCH_WEIGHT} × Psychology")
    print("="*65)

    correct = 0
    for i, case in enumerate(test_cases, 1):
        result = calculate_threat_score(case["text"], case["bert"])
        passed = result["risk_level"] == case["expected"]
        status = "✅" if passed else "❌"
        if passed:
            correct += 1

        print(f"\n  Test {i}: {status} {result['risk_emoji']}")
        print(f"  Text         : {case['text'][:60]}...")
        print(f"  Expected     : {case['expected']}")
        print(f"  Got          : {result['risk_level']}")
        print(f"  ─── Scores ───────────────────────────────────")
        print(f"  BERT Score   : {result['bert_score']:.2%}")
        print(f"  Psych Score  : {result['psychology_score']:.2%}")
        print(f"  Threat Score : {result['threat_percent']}%")
        print(f"  Formula      : {result['formula']}")
        print(f"  Description  : {result['risk_description'][:60]}...")

    accuracy = (correct / len(test_cases)) * 100
    print(f"\n{'='*65}")
    print(f"  Result: {correct}/{len(test_cases)} correct = {accuracy:.0f}%")
    print(f"{'='*65}")
    print(f"  Next → Start Phase 8: Explainable AI")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_test()
