"""
predictor/explainer.py — Phase 8: Explainable AI Module for SEAD-AI
=====================================================================
Generates human-readable explanations for every threat detection.

Why XAI matters:
  → Users need to UNDERSTAND why a message is flagged
  → Builds trust in the system
  → Helps users learn to spot attacks themselves
  → Required for enterprise security compliance

Output Example:
  Threat Score: 91%  🚨 High Risk

  Reasons Detected:
  ✓ Urgency tactics detected — "act now", "within 24 hours"
  ✓ Authority spoofing — impersonating "IRS", "bank"
  ✓ Fear induction — "suspended", "arrested", "legal action"
  ✓ Credential harvesting — asking to "verify your account"
  ✓ BERT AI confidence: 95% phishing probability

Usage:
    from predictor.explainer import generate_explanation
    explanation = generate_explanation(text, threat_result)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRINCIPLE DESCRIPTIONS
# Human-readable descriptions for each psychological principle
# ═══════════════════════════════════════════════════════════════════════════════

PRINCIPLE_DESCRIPTIONS = {
    "authority"   : "Authority spoofing — impersonating trusted institutions or figures",
    "urgency"     : "Urgency tactics — creating artificial time pressure to force quick decisions",
    "scarcity"    : "Scarcity manipulation — false shortage signals to bypass rational thinking",
    "social_proof": "Social proof manipulation — 'everyone else is doing it' pressure",
    "reciprocity" : "Reciprocity trap — fake gifts or favors to create psychological obligation",
    "liking"      : "False rapport building — flattery and fake personal connection",
    "fear"        : "Fear induction — threats and consequences to trigger panic responses",
    "curiosity"   : "Curiosity exploitation — clickbait and mystery to compel action",
    "reward"      : "Fake reward lure — false prizes or money offers to lower guard",
    "trust"       : "Trust manipulation — fake security signals to request credentials",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SUSPICIOUS PATTERN DETECTOR
# Detects specific high-risk patterns beyond psychology keywords
# ═══════════════════════════════════════════════════════════════════════════════

SUSPICIOUS_PATTERNS = [
    {
        "name"   : "Credential Harvesting",
        "signals": ["verify your account", "confirm your details",
                    "enter your password", "provide your",
                    "update your information", "submit your details",
                    "login to confirm", "sign in to verify"],
        "reason" : "Requesting sensitive credentials or personal information",
    },
    {
        "name"   : "Impersonation",
        "signals": ["irs", "fbi", "microsoft support", "apple id",
                    "google account", "paypal account", "amazon security",
                    "bank security", "it department", "ceo", "director"],
        "reason" : "Impersonating a trusted organization or authority figure",
    },
    {
        "name"   : "Link Manipulation",
        "signals": ["click here", "click the link", "click below",
                    "tap here", "follow this link", "open the link",
                    "visit now", "go to"],
        "reason" : "Directing user to click suspicious links",
    },
    {
        "name"   : "Fake Prize/Reward",
        "signals": ["you have won", "winner", "lottery", "prize",
                    "gift card", "congratulations", "selected winner",
                    "unclaimed funds", "inheritance", "jackpot"],
        "reason" : "Using fake rewards to deceive and manipulate",
    },
    {
        "name"   : "Threat/Consequence",
        "signals": ["arrested", "legal action", "lawsuit", "penalty",
                    "suspended", "terminated", "account closed",
                    "immediate arrest", "law enforcement", "court order"],
        "reason" : "Using threats or legal consequences to coerce action",
    },
    {
        "name"   : "Secrecy Request",
        "signals": ["keep this confidential", "do not tell anyone",
                    "between us", "confidential", "do not share",
                    "private matter", "keep secret"],
        "reason" : "Requesting secrecy — common in CEO fraud and BEC attacks",
    },
    {
        "name"   : "Unusual Payment Request",
        "signals": ["gift card", "itunes", "google play", "bitcoin",
                    "wire transfer", "western union", "money order",
                    "cryptocurrency", "purchase cards"],
        "reason" : "Requesting unusual payment methods — classic scam signal",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PATTERN DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def detect_suspicious_patterns(text: str) -> list:
    """
    Scans text for high-risk patterns and returns
    a list of detected pattern dicts.
    """
    text_lower = text.lower()
    detected   = []

    for pattern in SUSPICIOUS_PATTERNS:
        matched_signals = [
            sig for sig in pattern["signals"]
            if sig.lower() in text_lower
        ]
        if matched_signals:
            detected.append({
                "name"           : pattern["name"],
                "reason"         : pattern["reason"],
                "matched_signals": matched_signals,
            })

    return detected


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPLANATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_explanation(text: str, threat_result: dict) -> dict:
    """
    Generates a full human-readable explanation for a threat result.

    Args:
        text         : Original input text
        threat_result: Result dict from threat_score.calculate_threat_score()

    Returns:
        dict: {
            "summary"           : str — one-line summary
            "threat_score"      : int — 0 to 100
            "risk_level"        : str
            "risk_emoji"        : str
            "bert_explanation"  : str — BERT's contribution explained
            "psych_explanation" : str — Psychology findings explained
            "reasons"           : list — bullet point reasons
            "suspicious_patterns": list — high-risk patterns found
            "triggered_principles": list — Cialdini principles triggered
            "safe_advice"       : str — what user should do
            "confidence_note"   : str — honest note about confidence
        }
    """

    threat_score   = threat_result.get("threat_score", 0.0)
    threat_percent = threat_result.get("threat_percent", 0)
    risk_level     = threat_result.get("risk_level", "Safe")
    risk_emoji     = threat_result.get("risk_emoji", "✅")
    bert_score     = threat_result.get("bert_score", 0.0)
    psych_score    = threat_result.get("psychology_score", 0.0)
    psych_detail   = threat_result.get("psychology_detail", {})
    top_triggers   = psych_detail.get("top_triggers", [])
    all_keywords   = psych_detail.get("all_matched_keywords", [])
    principles     = psych_detail.get("principles", {})

    # ── Build reasons list ────────────────────────────────────────
    reasons = []

    # 1. BERT contribution
    if bert_score >= 0.7:
        reasons.append(
            f"AI model detected {bert_score:.0%} phishing probability "
            f"based on language patterns"
        )
    elif bert_score >= 0.4:
        reasons.append(
            f"AI model flagged this message with {bert_score:.0%} "
            f"suspicion level"
        )

    # 2. Triggered psychological principles
    triggered_principles = []
    for principle in top_triggers:
        if principle in PRINCIPLE_DESCRIPTIONS:
            desc   = PRINCIPLE_DESCRIPTIONS[principle]
            p_data = principles.get(principle, {})
            kws    = p_data.get("matched_keywords", [])[:3]  # show up to 3
            kw_str = ", ".join([f'"{k}"' for k in kws]) if kws else ""
            reason = f"{desc}"
            if kw_str:
                reason += f" — keywords: {kw_str}"
            reasons.append(reason)
            triggered_principles.append({
                "principle": principle.replace("_", " ").title(),
                "description": desc,
                "keywords": kws,
            })

    # 3. All detected keywords (if any)
    if all_keywords and not triggered_principles:
        kw_str = ", ".join([f'"{k}"' for k in all_keywords[:5]])
        reasons.append(f"Suspicious keywords detected: {kw_str}")

    # ── Detect suspicious patterns ────────────────────────────────
    suspicious_patterns = detect_suspicious_patterns(text)
    for pattern in suspicious_patterns:
        sig_str = ", ".join([f'"{s}"' for s in pattern["matched_signals"][:2]])
        reasons.append(f"{pattern['name']} — {pattern['reason']} ({sig_str})")

    # ── BERT explanation ──────────────────────────────────────────
    if bert_score >= 0.7:
        bert_explanation = (
            f"Our BERT AI model analyzed the language patterns and "
            f"determined {bert_score:.0%} probability this is a phishing "
            f"or social engineering message."
        )
    elif bert_score >= 0.4:
        bert_explanation = (
            f"Our BERT AI model flagged some suspicious language patterns "
            f"with {bert_score:.0%} confidence."
        )
    elif bert_score > 0:
        bert_explanation = (
            f"Our BERT AI model found the language largely consistent "
            f"with legitimate messages ({(1-bert_score):.0%} legitimate confidence)."
        )
    else:
        bert_explanation = "BERT AI analysis not available for this scan."

    # ── Psychology explanation ────────────────────────────────────
    if psych_score >= 0.4:
        psych_explanation = (
            f"Strong psychological manipulation patterns detected. "
            f"This message uses {len(top_triggers)} influence technique(s): "
            f"{', '.join([t.replace('_',' ').title() for t in top_triggers])}."
        )
    elif psych_score >= 0.1:
        psych_explanation = (
            f"Some psychological influence patterns detected: "
            f"{', '.join([t.replace('_',' ').title() for t in top_triggers])}. "
            f"These are common social engineering tactics."
        )
    else:
        psych_explanation = (
            "No significant psychological manipulation patterns detected."
        )

    # ── Summary ───────────────────────────────────────────────────
    if risk_level == "High Risk":
        summary = (
            f"⚠️ This message is likely a social engineering attack. "
            f"Threat score: {threat_percent}%. Do NOT comply with any requests."
        )
    elif risk_level == "Suspicious":
        summary = (
            f"This message has suspicious characteristics. "
            f"Threat score: {threat_percent}%. Verify before taking action."
        )
    else:
        summary = (
            f"This message appears to be legitimate. "
            f"Threat score: {threat_percent}%. No major threats detected."
        )

    # ── Safe advice ───────────────────────────────────────────────
    if risk_level == "High Risk":
        safe_advice = (
            "DO NOT click any links, provide personal information, "
            "purchase gift cards, or transfer money. "
            "Report this message to your IT/security team immediately. "
            "Verify directly with the supposed sender through official channels."
        )
    elif risk_level == "Suspicious":
        safe_advice = (
            "Do not click links directly. Instead, go to the official "
            "website manually. Verify the sender's email address carefully. "
            "Contact the organization through their official phone number."
        )
    else:
        safe_advice = (
            "Message appears safe. Always stay vigilant — "
            "verify unexpected requests through official channels."
        )

    # ── Confidence note ───────────────────────────────────────────
    if threat_percent >= 80:
        confidence_note = "High confidence detection — multiple strong signals found."
    elif threat_percent >= 50:
        confidence_note = "Moderate confidence — some suspicious signals detected."
    elif threat_percent >= 30:
        confidence_note = "Low-moderate confidence — exercise caution."
    else:
        confidence_note = "Low threat signals — message likely legitimate."

    logger.info(
        f"Explanation generated: {risk_level} | "
        f"{len(reasons)} reasons | "
        f"{len(suspicious_patterns)} patterns"
    )

    return {
        "summary"              : summary,
        "threat_score"         : threat_percent,
        "risk_level"           : risk_level,
        "risk_emoji"           : risk_emoji,
        "bert_explanation"     : bert_explanation,
        "psych_explanation"    : psych_explanation,
        "reasons"              : reasons,
        "suspicious_patterns"  : suspicious_patterns,
        "triggered_principles" : triggered_principles,
        "safe_advice"          : safe_advice,
        "confidence_note"      : confidence_note,
        "keyword_count"        : len(all_keywords),
        "pattern_count"        : len(suspicious_patterns),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_test():
    """Tests the explainer with sample messages."""
    from predictor.threat_score import calculate_threat_score

    test_cases = [
        {
            "text": (
                "URGENT: Your bank account has been suspended! "
                "Click here immediately to verify your account details "
                "or your account will be permanently deleted within 24 hours. "
                "Failure to respond will result in legal action."
            ),
            "bert": {
                "bert_score": 0.93, "label": "Phishing",
                "phishing_prob": 0.93, "confidence": 0.93
            },
        },
        {
            "text": (
                "CEO Request: I need you to urgently purchase $500 in iTunes "
                "gift cards for a client. Keep this confidential. Do not tell "
                "anyone in the office. Send me the codes immediately."
            ),
            "bert": {
                "bert_score": 0.91, "label": "Phishing",
                "phishing_prob": 0.91, "confidence": 0.91
            },
        },
        {
            "text": (
                "Hi team, please find attached the Q3 report for review. "
                "Let me know if you have any comments by Friday."
            ),
            "bert": {
                "bert_score": 0.04, "label": "Legitimate",
                "phishing_prob": 0.04, "confidence": 0.96
            },
        },
    ]

    print("\n" + "="*65)
    print("  SEAD-AI — Phase 8: Explainable AI Module Test")
    print("="*65)

    for i, case in enumerate(test_cases, 1):
        threat = calculate_threat_score(case["text"], case["bert"])
        expl   = generate_explanation(case["text"], threat)

        print(f"\n{'─'*65}")
        print(f"  TEST {i} — {expl['risk_emoji']} {expl['risk_level']}")
        print(f"{'─'*65}")
        print(f"  Text    : {case['text'][:70]}...")
        print(f"\n  SUMMARY : {expl['summary']}")
        print(f"\n  BERT    : {expl['bert_explanation']}")
        print(f"\n  PSYCH   : {expl['psych_explanation']}")

        print(f"\n  REASONS DETECTED ({len(expl['reasons'])}):")
        for r in expl["reasons"]:
            print(f"    ✓ {r}")

        if expl["suspicious_patterns"]:
            print(f"\n  SUSPICIOUS PATTERNS ({len(expl['suspicious_patterns'])}):")
            for p in expl["suspicious_patterns"]:
                print(f"    🚩 {p['name']}: {p['reason']}")

        print(f"\n  ADVICE  : {expl['safe_advice']}")
        print(f"  NOTE    : {expl['confidence_note']}")

    print(f"\n{'='*65}")
    print(f"  Phase 8 Complete ✅")
    print(f"  Next → Start Phase 9: SQLite Database")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_test()
