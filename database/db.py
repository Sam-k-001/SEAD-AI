"""
database/db.py — Phase 9: SQLite Database for SEAD-AI
=======================================================
Handles all database operations:
  - Create tables on startup
  - Save every scan result
  - Retrieve scan history
  - Search past scans
  - Generate statistics

Tables:
  messages — stores every analyzed message with full results

Usage:
    from database.db import save_result, get_history, get_stats
"""

import os
import sys
import sqlite3
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DB_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATABASE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    """
    Returns a SQLite connection with row_factory set to Row
    so results come back as dict-like objects instead of tuples.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # access columns by name: row["text"]
    return conn


def init_db() -> None:
    """
    Creates all tables if they don't exist.
    Safe to call multiple times — uses IF NOT EXISTS.
    Called automatically on app startup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── messages table ────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            text                TEXT    NOT NULL,
            text_preview        TEXT,
            bert_score          REAL    DEFAULT 0.0,
            psychology_score    REAL    DEFAULT 0.0,
            threat_score        REAL    DEFAULT 0.0,
            threat_percent      INTEGER DEFAULT 0,
            prediction          TEXT    DEFAULT 'Unknown',
            risk_level          TEXT    DEFAULT 'Safe',
            risk_emoji          TEXT    DEFAULT '✅',
            top_triggers        TEXT,
            matched_keywords    TEXT,
            suspicious_patterns TEXT,
            reasons             TEXT,
            explanation_summary TEXT,
            safe_advice         TEXT,
            formula             TEXT,
            timestamp           TEXT    NOT NULL
        )
    """)

    # ── stats table — tracks aggregate counts ─────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scan_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total_scans     INTEGER DEFAULT 0,
            high_risk_count INTEGER DEFAULT 0,
            suspicious_count INTEGER DEFAULT 0,
            safe_count      INTEGER DEFAULT 0,
            last_updated    TEXT
        )
    """)

    # Insert initial stats row if empty
    cursor.execute("SELECT COUNT(*) FROM scan_stats")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO scan_stats
            (total_scans, high_risk_count, suspicious_count, safe_count, last_updated)
            VALUES (0, 0, 0, 0, ?)
        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at: {DB_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SAVE RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def save_result(text: str, threat_result: dict, explanation: dict) -> int:
    """
    Saves a complete scan result to the database.

    Args:
        text         : Original input text
        threat_result: Result from threat_score.calculate_threat_score()
        explanation  : Result from explainer.generate_explanation()

    Returns:
        int: ID of the newly inserted row
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Extract values from result dicts
    psych_detail = threat_result.get("psychology_detail", {})

    # Convert lists to JSON strings for storage
    top_triggers = json.dumps(
        psych_detail.get("top_triggers", [])
    )
    matched_keywords = json.dumps(
        psych_detail.get("all_matched_keywords", [])
    )
    suspicious_patterns = json.dumps(
        [p["name"] for p in explanation.get("suspicious_patterns", [])]
    )
    reasons = json.dumps(explanation.get("reasons", []))

    # Short preview of text (first 100 chars)
    text_preview = text[:100] + "..." if len(text) > 100 else text

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO messages (
            text, text_preview,
            bert_score, psychology_score,
            threat_score, threat_percent,
            prediction, risk_level, risk_emoji,
            top_triggers, matched_keywords,
            suspicious_patterns, reasons,
            explanation_summary, safe_advice,
            formula, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        text,
        text_preview,
        round(threat_result.get("bert_score", 0.0), 4),
        round(threat_result.get("psychology_score", 0.0), 4),
        round(threat_result.get("threat_score", 0.0), 4),
        threat_result.get("threat_percent", 0),
        threat_result.get("bert_label", "Unknown"),
        threat_result.get("risk_level", "Safe"),
        threat_result.get("risk_emoji", "✅"),
        top_triggers,
        matched_keywords,
        suspicious_patterns,
        reasons,
        explanation.get("summary", ""),
        explanation.get("safe_advice", ""),
        threat_result.get("formula", ""),
        timestamp,
    ))

    row_id = cursor.lastrowid

    # Update aggregate stats
    risk_level = threat_result.get("risk_level", "Safe")
    _update_stats(cursor, risk_level)

    conn.commit()
    conn.close()

    logger.info(f"Saved scan result — ID: {row_id} | Risk: {risk_level}")
    return row_id


def _update_stats(cursor, risk_level: str) -> None:
    """Updates aggregate stats table after each scan."""
    if risk_level == "High Risk":
        col = "high_risk_count"
    elif risk_level == "Suspicious":
        col = "suspicious_count"
    else:
        col = "safe_count"

    cursor.execute(f"""
        UPDATE scan_stats SET
            total_scans   = total_scans + 1,
            {col}         = {col} + 1,
            last_updated  = ?
        WHERE id = 1
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RETRIEVE HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def get_history(limit: int = 20, offset: int = 0) -> list:
    """
    Returns scan history, newest first.

    Args:
        limit : Max rows to return (default 20)
        offset: For pagination (default 0)

    Returns:
        list of dicts with scan results
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id, text_preview, bert_score, psychology_score,
            threat_score, threat_percent, prediction,
            risk_level, risk_emoji, top_triggers,
            explanation_summary, timestamp
        FROM messages
        ORDER BY id DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        r = dict(row)
        # Parse JSON fields back to lists
        try:
            r["top_triggers"] = json.loads(r["top_triggers"] or "[]")
        except Exception:
            r["top_triggers"] = []
        results.append(r)

    return results


def get_scan_by_id(scan_id: int) -> dict:
    """Returns a single scan result by ID with full details."""
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM messages WHERE id = ?", (scan_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    result = dict(row)

    # Parse all JSON fields
    for field in ["top_triggers", "matched_keywords",
                  "suspicious_patterns", "reasons"]:
        try:
            result[field] = json.loads(result.get(field) or "[]")
        except Exception:
            result[field] = []

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def get_stats() -> dict:
    """
    Returns aggregate statistics for the dashboard.

    Returns:
        dict: {
            total_scans, high_risk_count, suspicious_count,
            safe_count, high_risk_percent, avg_threat_score,
            most_common_trigger, last_updated
        }
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Get aggregate counts
    cursor.execute("SELECT * FROM scan_stats WHERE id = 1")
    stats_row = cursor.fetchone()
    stats     = dict(stats_row) if stats_row else {}

    total = stats.get("total_scans", 0)

    # Calculate percentages
    if total > 0:
        stats["high_risk_percent"] = round(
            (stats.get("high_risk_count", 0) / total) * 100, 1
        )
        stats["safe_percent"] = round(
            (stats.get("safe_count", 0) / total) * 100, 1
        )
    else:
        stats["high_risk_percent"] = 0
        stats["safe_percent"]      = 0

    # Average threat score
    cursor.execute("SELECT AVG(threat_score) FROM messages")
    avg = cursor.fetchone()[0]
    stats["avg_threat_score"] = round((avg or 0) * 100, 1)

    # Most common psychological trigger
    cursor.execute("SELECT top_triggers FROM messages WHERE top_triggers != '[]'")
    trigger_rows = cursor.fetchall()
    trigger_counts = {}
    for row in trigger_rows:
        try:
            triggers = json.loads(row[0] or "[]")
            for t in triggers:
                trigger_counts[t] = trigger_counts.get(t, 0) + 1
        except Exception:
            pass

    stats["most_common_trigger"] = (
        max(trigger_counts, key=trigger_counts.get)
        if trigger_counts else "None"
    )

    conn.close()
    return stats


def get_total_scans() -> int:
    """Quick helper — returns total scan count."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM messages")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def search_history(query: str, limit: int = 10) -> list:
    """
    Searches scan history by text content or risk level.

    Args:
        query: Search string
        limit: Max results

    Returns:
        list of matching scan dicts
    """
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, text_preview, threat_percent, risk_level,
               risk_emoji, timestamp
        FROM messages
        WHERE text LIKE ? OR risk_level LIKE ?
        ORDER BY id DESC
        LIMIT ?
    """, (f"%{query}%", f"%{query}%", limit))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_test():
    """Tests all database operations."""
    from predictor.threat_score import calculate_threat_score
    from predictor.explainer    import generate_explanation

    print("\n" + "="*60)
    print("  SEAD-AI — Phase 9: SQLite Database Test")
    print("="*60)

    # 1. Initialize DB
    print("\n  [1] Initializing database...")
    init_db()
    print(f"  ✅ Database created at: {DB_PATH}")

    # 2. Save test scans
    print("\n  [2] Saving test scan results...")

    test_messages = [
        {
            "text": "URGENT: Your bank account has been suspended! "
                    "Verify immediately or account will be deleted.",
            "bert": {"bert_score": 0.93, "label": "Phishing",
                     "phishing_prob": 0.93, "confidence": 0.93,
                     "bert_label": "Phishing", "bert_confidence": 0.93},
        },
        {
            "text": "Congratulations! You have won a FREE iPhone! "
                    "Claim your prize now before it expires!",
            "bert": {"bert_score": 0.87, "label": "Phishing",
                     "phishing_prob": 0.87, "confidence": 0.87,
                     "bert_label": "Phishing", "bert_confidence": 0.87},
        },
        {
            "text": "Hi team, the sprint review is moved to Thursday 4pm. "
                    "Agenda shared in the team channel.",
            "bert": {"bert_score": 0.03, "label": "Legitimate",
                     "phishing_prob": 0.03, "confidence": 0.97,
                     "bert_label": "Legitimate", "bert_confidence": 0.97},
        },
    ]

    saved_ids = []
    for msg in test_messages:
        threat = calculate_threat_score(msg["text"], msg["bert"])
        threat["bert_label"]      = msg["bert"]["bert_label"]
        threat["bert_confidence"] = msg["bert"]["bert_confidence"]
        expl   = generate_explanation(msg["text"], threat)
        row_id = save_result(msg["text"], threat, expl)
        saved_ids.append(row_id)
        print(f"  ✅ Saved ID {row_id}: "
              f"{threat['risk_level']} ({threat['threat_percent']}%)")

    # 3. Retrieve history
    print("\n  [3] Retrieving scan history...")
    history = get_history(limit=5)
    print(f"  ✅ Found {len(history)} recent scans:")
    for h in history:
        print(f"     [{h['id']}] {h['risk_emoji']} {h['risk_level']:10s} "
              f"| {h['threat_percent']:3d}% | {h['timestamp']}")

    # 4. Get single scan by ID
    print(f"\n  [4] Retrieving scan ID {saved_ids[0]}...")
    scan = get_scan_by_id(saved_ids[0])
    print(f"  ✅ Text preview : {scan.get('text_preview', '')[:50]}...")
    print(f"  ✅ Risk level   : {scan.get('risk_level')}")
    print(f"  ✅ Reasons      : {len(scan.get('reasons', []))} reasons stored")

    # 5. Statistics
    print("\n  [5] Fetching statistics...")
    stats = get_stats()
    print(f"  ✅ Total scans     : {stats.get('total_scans', 0)}")
    print(f"  ✅ High Risk       : {stats.get('high_risk_count', 0)}")
    print(f"  ✅ Suspicious      : {stats.get('suspicious_count', 0)}")
    print(f"  ✅ Safe            : {stats.get('safe_count', 0)}")
    print(f"  ✅ Avg threat score: {stats.get('avg_threat_score', 0)}%")
    print(f"  ✅ Top trigger     : {stats.get('most_common_trigger')}")

    # 6. Search
    print("\n  [6] Testing search...")
    results = search_history("suspended")
    print(f"  ✅ Search 'suspended': {len(results)} result(s) found")

    print(f"\n{'='*60}")
    print(f"  Phase 9 Complete ✅ — All DB operations working!")
    print(f"  Database file: {DB_PATH}")
    print(f"{'='*60}")
    print(f"  Next → Start Phase 10: Flask Backend")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_test()
