"""
psychology/scorer.py — Phase 6: Psychology Scoring Engine for SEAD-AI
=======================================================================
Novel contribution: Encodes Cialdini's 6 Principles of Influence as
detectable ML features for social engineering detection.

Principles detected:
  1. Authority    — Fake authority impersonation
  2. Urgency      — Artificial time pressure
  3. Scarcity     — False shortage signals
  4. Social Proof — Herd behaviour manipulation
  5. Reciprocity  — Fake favors to create obligation
  6. Liking       — False rapport building
  7. Fear         — Threats and consequences
  8. Curiosity    — Clickbait and mystery
  9. Reward       — Fake prizes and money offers
  10. Trust       — False trust signals

Usage:
    from psychology.scorer import score_psychology
    result = score_psychology("Your account is suspended! Act now!")
"""

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KEYWORD DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════════

PSYCHOLOGY_KEYWORDS = {

    "authority": {
        "principle_weight": 0.15,
        "keywords": [
            "irs", "fbi", "microsoft", "apple", "google", "paypal",
            "amazon", "bank", "government", "official", "ceo", "director",
            "administrator", "support team", "security team", "law enforcement",
            "court order", "legal notice", "official notice", "headquarters",
            "department of", "authorized", "verified",
        ],
    },

    "urgency": {
        "principle_weight": 0.20,
        "keywords": [
            "urgent", "immediately", "act now", "right now", "expires",
            "last chance", "final notice", "deadline", "within 24 hours",
            "within 48 hours", "today only", "do not delay", "time sensitive",
            "respond immediately", "failure to respond", "will be terminated",
            "will be suspended", "will be cancelled", "will be deleted",
            "asap", "hurry", "now or never", "account will be",
        ],
    },

    "scarcity": {
        "principle_weight": 0.10,
        "keywords": [
            "limited time", "limited offer", "only today", "only available",
            "few spots left", "running out", "exclusive offer", "special access",
            "select few", "rare opportunity", "once in a lifetime", "never again",
            "while supplies last", "limited seats", "only for today",
        ],
    },

    "social_proof": {
        "principle_weight": 0.05,
        "keywords": [
            "thousands of users", "millions of people", "everyone is",
            "most people", "join others", "trusted by", "used by millions",
            "as seen on", "our community", "others have",
        ],
    },

    "reciprocity": {
        "principle_weight": 0.08,
        "keywords": [
            "free gift", "bonus", "we are giving", "complimentary",
            "at no charge", "as a thank you", "reward for", "special gift",
            "we owe you", "gift card", "voucher", "cashback", "free offer",
        ],
    },

    "liking": {
        "principle_weight": 0.05,
        "keywords": [
            "dear friend", "valued customer", "dear valued", "you are special",
            "chosen", "selected you", "loyal customer", "you are important",
            "personally selected", "especially for you", "just for you",
        ],
    },

    "fear": {
        "principle_weight": 0.20,
        "keywords": [
            "suspended", "terminated", "arrested", "legal action", "lawsuit",
            "penalty", "locked", "blocked", "disabled", "hacked",
            "compromised", "breach", "unauthorized access", "virus detected",
            "malware", "suspicious activity", "fraud detected", "identity theft",
            "immediate arrest", "your account has been", "account closed",
        ],
    },

    "curiosity": {
        "principle_weight": 0.05,
        "keywords": [
            "you won't believe", "click to find out", "shocking", "revealed",
            "secret", "find out now", "you have been selected", "surprise",
            "discover", "hidden", "leaked", "insider",
        ],
    },

    "reward": {
        "principle_weight": 0.07,
        "keywords": [
            "congratulations", "winner", "you have won", "prize", "lottery",
            "jackpot", "cash prize", "gift card", "iphone", "free ipad",
            "million dollars", "inheritance", "unclaimed funds", "reward",
            "earn money", "make money", "investment opportunity",
        ],
    },

    "trust": {
        "principle_weight": 0.05,
        "keywords": [
            "100% safe", "guaranteed", "no risk", "verified account",
            "official website", "click the secure link", "confirm your identity",
            "verify your account", "update your information",
            "confirm your details", "enter your password", "provide your",
            "submit your details",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def score_principle(text_lower: str, principle: str, data: dict) -> dict:
    """
    Scores one psychological principle.
    Uses keyword presence (not weights per keyword — simpler and more accurate).
    Score = matched_keywords / total_keywords in principle.
    """
    matched = []
    keywords = data["keywords"]

    for keyword in keywords:
        if re.search(re.escape(keyword.lower()), text_lower):
            matched.append(keyword)

    # Normalized score: how many of this principle's keywords matched
    normalized = len(matched) / len(keywords) if keywords else 0.0

    # Cap at 1.0
    normalized = min(normalized, 1.0)

    return {
        "normalized_score": round(normalized, 4),
        "matched_keywords": matched,
        "match_count"     : len(matched),
    }


def score_psychology(text: str) -> dict:
    """
    Main psychology scoring function.
    Returns combined psychology score across all 10 principles.

    Args:
        text (str): Raw input text

    Returns:
        dict with psychology_score, risk_level, principles, top_triggers
    """
    if not text or len(text.strip()) == 0:
        return _empty_result()

    text_lower   = text.lower()
    all_keywords = []
    principles   = {}
    weighted_sum = 0.0

    for principle, data in PSYCHOLOGY_KEYWORDS.items():
        result = score_principle(text_lower, principle, data)
        principles[principle] = result
        all_keywords.extend(result["matched_keywords"])

        # Weighted contribution
        weighted_sum += result["normalized_score"] * data["principle_weight"]

    # Normalize by total weight (all weights sum to 1.0 already)
    psych_score = weighted_sum

    # Sensitivity boost for security system
    # Small signals matter — even 1-2 keywords should register
    # Boost formula: score * 3.0 if low, capped at 1.0
    if psych_score > 0:
        psych_score = min(psych_score * 6.0, 1.0)

    psych_score = round(psych_score, 4)

    # Top triggered principles
    top_triggers = sorted(
        [p for p, s in principles.items() if s["match_count"] > 0],
        key=lambda p: principles[p]["normalized_score"],
        reverse=True,
    )[:3]

    # Risk level
    if psych_score >= 0.18:
        risk_level = "High Risk"
    elif psych_score >= 0.10:
        risk_level = "Suspicious"
    else:
        risk_level = "Safe"

    logger.info(
        f"Psychology Score: {psych_score:.2%} | "
        f"Risk: {risk_level} | "
        f"Triggers: {top_triggers}"
    )

    return {
        "psychology_score"    : psych_score,
        "risk_level"          : risk_level,
        "principles"          : principles,
        "top_triggers"        : top_triggers,
        "all_matched_keywords": list(set(all_keywords)),
        "total_matches"       : len(all_keywords),
    }


def _empty_result() -> dict:
    return {
        "psychology_score"    : 0.0,
        "risk_level"          : "Safe",
        "principles"          : {},
        "top_triggers"        : [],
        "all_matched_keywords": [],
        "total_matches"       : 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_test():
    test_cases = [
        {
            "text"    : "URGENT: Your bank account has been suspended! "
                        "Click here immediately to verify your identity "
                        "or your account will be permanently deleted within 24 hours.",
            "expected": "High Risk",
        },
        {
            "text"    : "Congratulations! You have been selected as today's "
                        "winner of a FREE iPhone 15! Limited time offer — "
                        "claim your prize NOW before it expires!",
            "expected": "High Risk",
        },
        {
            "text"    : "CEO Request: I need you to urgently purchase $500 "
                        "in gift cards for a client. Keep this confidential. "
                        "Legal action will follow if not done today.",
            "expected": "High Risk",
        },
        {
            "text"    : "IRS Notice: You owe back taxes. Failure to respond "
                        "in 24 hours will result in immediate arrest. "
                        "This is your final notice.",
            "expected": "High Risk",
        },
        {
            "text"    : "Hi team, the meeting notes from yesterday are "
                        "attached. Let me know if you have any questions.",
            "expected": "Safe",
        },
        {
            "text"    : "Your electricity bill is due on November 30. "
                        "Please pay online to avoid late fees.",
            "expected": "Safe",
        },
    ]

    print("\n" + "="*65)
    print("  SEAD-AI — Phase 6: Psychology Scorer Test")
    print("="*65)

    correct = 0
    for i, case in enumerate(test_cases, 1):
        result   = score_psychology(case["text"])
        passed   = result["risk_level"] == case["expected"]
        status   = "✅" if passed else "❌"
        if passed:
            correct += 1

        print(f"\n  Test {i}: {status}")
        print(f"  Text         : {case['text'][:60]}...")
        print(f"  Expected     : {case['expected']}")
        print(f"  Got          : {result['risk_level']}")
        print(f"  Psych Score  : {result['psychology_score']:.2%}")
        print(f"  Top Triggers : {result['top_triggers']}")
        print(f"  Keywords Hit : {result['all_matched_keywords']}")

        if result["principles"]:
            print(f"  Breakdown    :")
            for p, d in result["principles"].items():
                if d["match_count"] > 0:
                    print(f"    {p:15s}: {d['normalized_score']:.2%} "
                          f"({d['match_count']} keywords matched)")

    accuracy = (correct / len(test_cases)) * 100
    print(f"\n{'='*65}")
    print(f"  Result: {correct}/{len(test_cases)} correct = {accuracy:.0f}%")
    print(f"{'='*65}")
    print(f"  Next → Start Phase 7: Threat Score Engine")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_test()
