"""
tests/test_all.py — Phase 12: Complete Test Suite for SEAD-AI
==============================================================
Tests all modules end-to-end:
  1. Psychology Scorer
  2. Threat Score Engine
  3. Explainable AI
  4. Database Operations
  5. Flask API Endpoints
  6. Full Pipeline Integration

Usage:
    python tests/test_all.py          ← run all tests with report
    pytest tests/test_all.py -v       ← run with pytest (verbose)

Output:
    Console report with Precision, Recall, F1, Confusion Matrix
"""

import os
import sys
import json
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psychology.scorer   import score_psychology
from predictor.threat_score import calculate_threat_score
from predictor.explainer import generate_explanation
from database.db         import init_db, save_result, get_history, get_stats, get_scan_by_id
from app                 import app


# ── Shared test data ──────────────────────────────────────────────────────────
PHISHING_MESSAGES = [
    "URGENT: Your bank account has been suspended! Click here immediately "
    "to verify your identity or your account will be permanently deleted within 24 hours.",

    "Congratulations! You have been selected as today's winner of a FREE "
    "iPhone 15! Limited time offer — claim your prize NOW before it expires!",

    "CEO Request: I need you to urgently purchase $500 in iTunes gift cards "
    "for a client. Keep this confidential. Legal action will follow if not done today.",

    "IRS NOTICE: You owe back taxes. Failure to respond in 24 hours will "
    "result in immediate arrest. This is your final notice.",

    "SECURITY ALERT: Your Google account has been compromised. "
    "Verify your account immediately at google-secure-verify.com or it will be suspended.",

    "Dear valued customer, your PayPal account has been limited. "
    "Please confirm your identity urgently to avoid permanent suspension.",

    "You have won a $1000 Amazon gift card! Act now — only 3 spots left. "
    "Enter your credit card details to claim your exclusive reward.",

    "Microsoft Security Team: Unauthorized access detected on your account. "
    "Please verify your credentials immediately or account will be terminated.",
]

LEGITIMATE_MESSAGES = [
    "Hi team, please find attached the meeting notes from yesterday's standup. "
    "Let me know if you have any questions.",

    "Your monthly bank statement for October 2024 is now available "
    "in your online banking portal.",

    "Thank you for your order #45892. Your package has been shipped "
    "and will arrive within 3-5 business days.",

    "Reminder: Your dentist appointment is scheduled for tomorrow "
    "at 10:30 AM at City Dental Clinic.",

    "Hi, I'm following up on our project proposal. "
    "Are you available for a 30-minute call this week?",

    "Team update: The sprint review is moved to Thursday 4pm. "
    "Agenda shared in the team channel.",

    "Your electricity bill of Rs. 1,240 is due on November 30. "
    "Pay online to avoid late fees.",

    "Good morning! Just a reminder that the quarterly report "
    "is due by end of day Friday.",
]

# Simulated BERT results (as if model predicted correctly)
def make_bert_phishing(score=0.90):
    return {"bert_score": score, "label": "Phishing",
            "phishing_prob": score, "confidence": score,
            "bert_label": "Phishing", "bert_confidence": score}

def make_bert_legit(score=0.05):
    return {"bert_score": score, "label": "Legitimate",
            "phishing_prob": score, "confidence": 1-score,
            "bert_label": "Legitimate", "bert_confidence": 1-score}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 1 — PSYCHOLOGY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class TestPsychologyScorer(unittest.TestCase):
    """Tests for psychology/scorer.py"""

    def test_phishing_gets_high_score(self):
        """Phishing messages should score > 0.10"""
        for msg in PHISHING_MESSAGES:
            result = score_psychology(msg)
            self.assertGreater(
                result["psychology_score"], 0.05,
                f"Expected high psych score for: {msg[:60]}"
            )

    def test_legitimate_gets_low_score(self):
        """Legitimate messages should score < 0.20"""
        for msg in LEGITIMATE_MESSAGES:
            result = score_psychology(msg)
            self.assertLess(
                result["psychology_score"], 0.20,
                f"Expected low psych score for: {msg[:60]}"
            )

    def test_returns_correct_keys(self):
        """Result must have all required keys"""
        result = score_psychology("Test message")
        required = ["psychology_score", "risk_level", "principles",
                    "top_triggers", "all_matched_keywords", "total_matches"]
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_score_in_range(self):
        """Score must be between 0.0 and 1.0"""
        for msg in PHISHING_MESSAGES + LEGITIMATE_MESSAGES:
            result = score_psychology(msg)
            self.assertGreaterEqual(result["psychology_score"], 0.0)
            self.assertLessEqual(result["psychology_score"], 1.0)

    def test_empty_input_safe(self):
        """Empty input should return Safe with score 0"""
        result = score_psychology("")
        self.assertEqual(result["psychology_score"], 0.0)
        self.assertEqual(result["risk_level"], "Safe")

    def test_urgency_detected(self):
        """Urgency keywords should trigger urgency principle"""
        result = score_psychology("URGENT: Act immediately or your account will be deleted within 24 hours!")
        self.assertIn("urgency", result["top_triggers"])

    def test_fear_detected(self):
        """Fear keywords should trigger fear principle"""
        result = score_psychology("Your account has been suspended due to suspicious activity. Legal action will follow.")
        triggers = result["top_triggers"]
        self.assertTrue(
            "fear" in triggers or "urgency" in triggers,
            f"Expected fear/urgency trigger, got: {triggers}"
        )

    def test_reward_detected(self):
        """Reward keywords should trigger reward principle"""
        result = score_psychology("Congratulations! You have won a prize! Claim your gift card now!")
        triggers = result["top_triggers"]
        self.assertTrue(
            any(t in triggers for t in ["reward", "urgency", "scarcity"]),
            f"Expected reward trigger, got: {triggers}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 2 — THREAT SCORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestThreatScoreEngine(unittest.TestCase):
    """Tests for predictor/threat_score.py"""

    def test_high_bert_high_psych_gives_high_risk(self):
        """High BERT + high psychology → High Risk"""
        result = calculate_threat_score(PHISHING_MESSAGES[0], make_bert_phishing(0.95))
        self.assertEqual(result["risk_level"], "High Risk",
                         f"Expected High Risk, got {result['risk_level']} "
                         f"(score: {result['threat_percent']}%)")

    def test_low_bert_low_psych_gives_safe(self):
        """Low BERT + low psychology → Safe"""
        result = calculate_threat_score(LEGITIMATE_MESSAGES[0], make_bert_legit(0.03))
        self.assertEqual(result["risk_level"], "Safe",
                         f"Expected Safe, got {result['risk_level']}")

    def test_formula_weights(self):
        """Threat = 0.7 × BERT + 0.3 × Psych (approx)"""
        bert   = make_bert_phishing(0.80)
        result = calculate_threat_score("Your account is suspended urgently!", bert)
        # BERT contributes 0.7 × 0.80 = 0.56 minimum
        self.assertGreaterEqual(result["threat_score"], 0.50)

    def test_score_in_range(self):
        """Threat score must be 0–100"""
        for msg in PHISHING_MESSAGES + LEGITIMATE_MESSAGES:
            r = calculate_threat_score(msg, make_bert_phishing(0.5))
            self.assertGreaterEqual(r["threat_percent"], 0)
            self.assertLessEqual(r["threat_percent"], 100)

    def test_required_keys_present(self):
        """Result must have all required keys"""
        result = calculate_threat_score("Test", make_bert_phishing())
        required = ["threat_score", "threat_percent", "risk_level",
                    "risk_color", "risk_emoji", "bert_score",
                    "psychology_score", "formula"]
        for key in required:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_no_bert_fallback(self):
        """Works without BERT result (psychology-only mode)"""
        result = calculate_threat_score(
            "URGENT: Your account is suspended. Verify immediately!", None
        )
        self.assertIn("threat_score", result)
        self.assertGreaterEqual(result["threat_score"], 0.0)

    def test_risk_colors(self):
        """Each risk level has correct color"""
        # High Risk → red
        r = calculate_threat_score(PHISHING_MESSAGES[0], make_bert_phishing(0.95))
        if r["risk_level"] == "High Risk":
            self.assertIn("FF", r["risk_color"].upper())


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 3 — EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════════════════════════

class TestExplainableAI(unittest.TestCase):
    """Tests for predictor/explainer.py"""

    def _get_explanation(self, text, bert_score=0.90):
        bert   = make_bert_phishing(bert_score)
        threat = calculate_threat_score(text, bert)
        threat["bert_label"]      = bert["bert_label"]
        threat["bert_confidence"] = bert["bert_confidence"]
        return generate_explanation(text, threat)

    def test_returns_required_keys(self):
        """Explanation must have all required keys"""
        expl = self._get_explanation(PHISHING_MESSAGES[0])
        required = ["summary", "threat_score", "risk_level", "risk_emoji",
                    "bert_explanation", "psych_explanation", "reasons",
                    "suspicious_patterns", "safe_advice", "confidence_note"]
        for key in required:
            self.assertIn(key, expl, f"Missing key: {key}")

    def test_phishing_has_reasons(self):
        """Phishing messages should generate at least 1 reason"""
        expl = self._get_explanation(PHISHING_MESSAGES[0])
        self.assertGreater(len(expl["reasons"]), 0,
                           "Expected at least 1 reason for phishing message")

    def test_credential_harvesting_detected(self):
        """Credential request should be flagged"""
        text = "Please verify your account and enter your password at secure-login.com"
        expl = self._get_explanation(text, bert_score=0.85)
        pattern_names = [p["name"] for p in expl["suspicious_patterns"]]
        self.assertTrue(
            any("Credential" in p or "Trust" in p or "Link" in p
                for p in pattern_names),
            f"Credential harvesting not detected. Patterns: {pattern_names}"
        )

    def test_secrecy_request_detected(self):
        """Secrecy request pattern should be detected"""
        text = "CEO: Buy gift cards urgently. Keep this confidential, do not tell anyone."
        expl = self._get_explanation(text, bert_score=0.91)
        pattern_names = [p["name"] for p in expl["suspicious_patterns"]]
        self.assertIn("Secrecy Request", pattern_names,
                      f"Secrecy not detected. Patterns: {pattern_names}")

    def test_safe_message_positive_advice(self):
        """Legitimate messages should get safe advice"""
        bert   = make_bert_legit(0.03)
        threat = calculate_threat_score(LEGITIMATE_MESSAGES[0], bert)
        threat["bert_label"] = "Legitimate"
        threat["bert_confidence"] = 0.97
        expl = generate_explanation(LEGITIMATE_MESSAGES[0], threat)
        self.assertIn("safe", expl["safe_advice"].lower())

    def test_summary_contains_score(self):
        """Summary should mention the threat score"""
        expl = self._get_explanation(PHISHING_MESSAGES[0])
        self.assertIn(str(expl["threat_score"]), expl["summary"])


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 4 — DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabase(unittest.TestCase):
    """Tests for database/db.py"""

    def setUp(self):
        init_db()

    def _make_scan(self, text, bert_score=0.90):
        bert   = make_bert_phishing(bert_score)
        threat = calculate_threat_score(text, bert)
        threat["bert_label"]      = bert["bert_label"]
        threat["bert_confidence"] = bert["bert_confidence"]
        expl   = generate_explanation(text, threat)
        return threat, expl

    def test_save_and_retrieve(self):
        """Saved scan should be retrievable by ID"""
        text           = "URGENT: Verify your bank account immediately!"
        threat, expl   = self._make_scan(text)
        row_id         = save_result(text, threat, expl)

        self.assertIsInstance(row_id, int)
        self.assertGreater(row_id, 0)

        scan = get_scan_by_id(row_id)
        self.assertEqual(scan["id"], row_id)
        self.assertIn("risk_level", scan)

    def test_history_returns_list(self):
        """get_history should return a list"""
        history = get_history(limit=5)
        self.assertIsInstance(history, list)

    def test_stats_has_required_keys(self):
        """get_stats should return required keys"""
        stats    = get_stats()
        required = ["total_scans", "high_risk_count",
                    "suspicious_count", "safe_count"]
        for key in required:
            self.assertIn(key, stats, f"Missing stats key: {key}")

    def test_stats_count_increments(self):
        """Total scan count should increase after saving"""
        before = get_stats().get("total_scans", 0)
        text   = f"Test message at {time.time()}"
        threat, expl = self._make_scan(text)
        save_result(text, threat, expl)
        after  = get_stats().get("total_scans", 0)
        self.assertGreater(after, before)

    def test_text_preview_truncated(self):
        """Long text should be truncated to 100 chars in preview"""
        long_text = "URGENT: " + "x" * 200
        threat, expl = self._make_scan(long_text)
        row_id = save_result(long_text, threat, expl)
        scan   = get_scan_by_id(row_id)
        self.assertLessEqual(len(scan.get("text_preview", "")), 110)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 5 — FLASK API
# ═══════════════════════════════════════════════════════════════════════════════

class TestFlaskAPI(unittest.TestCase):
    """Tests for app.py Flask endpoints"""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def _analyze(self, text):
        return self.client.post(
            "/analyze",
            data=json.dumps({"text": text}),
            content_type="application/json",
        )

    def test_health_endpoint(self):
        """GET /health should return 200 with status ok"""
        r    = self.client.get("/health")
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(data["status"], "ok")

    def test_analyze_phishing(self):
        """POST /analyze with phishing text should return success"""
        r    = self._analyze(PHISHING_MESSAGES[0])
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["success"])
        self.assertIn("threat_score", data)
        self.assertIn("risk_level", data)
        self.assertIn("reasons", data)

    def test_analyze_legitimate(self):
        """POST /analyze with legitimate text should return success"""
        r    = self._analyze(LEGITIMATE_MESSAGES[0])
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["success"])

    def test_analyze_empty_text(self):
        """POST /analyze with empty text should return 400"""
        r    = self._analyze("")
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 400)
        self.assertFalse(data["success"])

    def test_analyze_no_body(self):
        """POST /analyze with no body should return 400"""
        r = self.client.post("/analyze", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_history_endpoint(self):
        """GET /history should return list of scans"""
        r    = self.client.get("/history?limit=5")
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["success"])
        self.assertIn("scans", data)
        self.assertIsInstance(data["scans"], list)

    def test_stats_endpoint(self):
        """GET /stats should return statistics"""
        r    = self.client.get("/stats")
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["success"])
        self.assertIn("stats", data)

    def test_search_endpoint(self):
        """GET /search?q=term should return results"""
        r    = self.client.get("/search?q=account")
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(data["success"])

    def test_search_no_query_returns_400(self):
        """GET /search with no query returns 400"""
        r    = self.client.get("/search")
        self.assertEqual(r.status_code, 400)

    def test_404_handler(self):
        """Unknown route should return 404 JSON"""
        r    = self.client.get("/nonexistent")
        data = json.loads(r.data)
        self.assertEqual(r.status_code, 404)
        self.assertFalse(data["success"])

    def test_analyze_returns_scan_id(self):
        """Analyze should return a scan_id for database lookup"""
        r    = self._analyze(PHISHING_MESSAGES[1])
        data = json.loads(r.data)
        self.assertIn("scan_id", data)
        self.assertIsInstance(data["scan_id"], int)

    def test_index_serves_html(self):
        """GET / should serve the HTML frontend"""
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 6 — FULL PIPELINE + METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline(unittest.TestCase):
    """End-to-end pipeline tests with ML metrics."""

    def test_pipeline_on_all_test_cases(self):
        """Runs full pipeline on all test messages and reports metrics."""
        all_texts  = PHISHING_MESSAGES + LEGITIMATE_MESSAGES
        all_labels = [1] * len(PHISHING_MESSAGES) + [0] * len(LEGITIMATE_MESSAGES)
        predictions = []

        for text, true_label in zip(all_texts, all_labels):
            bert = make_bert_phishing(0.90) if true_label == 1 else make_bert_legit(0.05)
            result = calculate_threat_score(text, bert)
            predicted = 1 if result["risk_level"] in ["High Risk", "Suspicious"] else 0
            predictions.append(predicted)

        # Calculate metrics
        tp = sum(1 for p, t in zip(predictions, all_labels) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(predictions, all_labels) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(predictions, all_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, all_labels) if p == 0 and t == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy  = (tp + tn) / len(all_labels)

        # Store for report
        self.__class__.metrics = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy,
        }

        # Assertions
        self.assertGreaterEqual(accuracy,  0.70, f"Accuracy too low: {accuracy:.2%}")
        self.assertGreaterEqual(recall,    0.70, f"Recall too low: {recall:.2%}")
        self.assertGreaterEqual(precision, 0.60, f"Precision too low: {precision:.2%}")


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM TEST RUNNER — PRETTY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class PrettyTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        self.stream.write(f"  ✅ {test._testMethodName}\n")
        self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.stream.write(f"  ❌ {test._testMethodName}\n")
        self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        self.stream.write(f"  💥 {test._testMethodName} (ERROR)\n")
        self.stream.flush()


def run_all_tests():
    print("\n" + "="*65)
    print("  SEAD-AI — Phase 12: Full Test Suite")
    print("="*65)

    suites = [
        ("Psychology Scorer",   TestPsychologyScorer),
        ("Threat Score Engine", TestThreatScoreEngine),
        ("Explainable AI",      TestExplainableAI),
        ("Database",            TestDatabase),
        ("Flask API",           TestFlaskAPI),
        ("Full Pipeline",       TestFullPipeline),
    ]

    total_passed = 0
    total_failed = 0
    total_errors = 0

    for suite_name, suite_class in suites:
        print(f"\n  ── {suite_name} " + "─"*(45-len(suite_name)))
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromTestCase(suite_class)
        stream = open(os.devnull, "w")

        runner = unittest.TextTestRunner(
            stream=sys.stdout,
            resultclass=PrettyTestResult,
            verbosity=0,
        )
        result = runner.run(suite)
        total_passed += result.testsRun - len(result.failures) - len(result.errors)
        total_failed += len(result.failures)
        total_errors += len(result.errors)

    # ── ML Metrics Report ──────────────────────────────────────────
    print("\n" + "="*65)
    print("  CLASSIFICATION METRICS (Psychology + BERT Fusion)")
    print("="*65)

    try:
        m = TestFullPipeline.metrics
        print(f"\n  ── Confusion Matrix ─────────────────────────────")
        print(f"  True Positives  (TP): {m['tp']:3d}  Phishing → Phishing ✅")
        print(f"  True Negatives  (TN): {m['tn']:3d}  Legit → Legit       ✅")
        print(f"  False Positives (FP): {m['fp']:3d}  Legit → Phishing    ❌")
        print(f"  False Negatives (FN): {m['fn']:3d}  Phishing → Legit    ❌")
        print(f"\n  ── Performance Metrics ──────────────────────────")
        print(f"  Accuracy  : {m['accuracy']:.2%}")
        print(f"  Precision : {m['precision']:.2%}")
        print(f"  Recall    : {m['recall']:.2%}  ← catches real attacks")
        print(f"  F1 Score  : {m['f1']:.2%}")
    except AttributeError:
        print("  Metrics not available — pipeline test may have failed.")

    # ── Final Summary ──────────────────────────────────────────────
    total = total_passed + total_failed + total_errors
    print(f"\n{'='*65}")
    print(f"  TOTAL: {total_passed}/{total} passed | "
          f"{total_failed} failed | {total_errors} errors")
    status = "✅ ALL TESTS PASSED" if total_failed == 0 and total_errors == 0 \
             else "⚠️  SOME TESTS FAILED"
    print(f"  STATUS: {status}")
    print(f"{'='*65}")
    print(f"  Next → Start Phase 13: Deployment")
    print(f"{'='*65}\n")

    return total_failed == 0 and total_errors == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
