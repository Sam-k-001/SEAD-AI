"""
app.py — Phase 10: Complete Flask Backend for SEAD-AI
======================================================
Connects all modules into a production-ready API:
  - POST /analyze      → Full threat analysis
  - GET  /history      → Scan history
  - GET  /history/<id> → Single scan detail
  - GET  /stats        → Dashboard statistics
  - GET  /search       → Search scan history
  - GET  /health       → Health check

Usage:
    python app.py
    Open: http://localhost:5000
"""

import os
import sys

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
from database.db import init_db, save_result, get_history, get_scan_by_id, get_stats, search_history
from predictor.threat_score import calculate_threat_score
from predictor.explainer import generate_explanation
from utils.logger import get_logger
from utils.helpers import ensure_dirs

logger = get_logger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow frontend JS to call API from any origin

# ── Startup ───────────────────────────────────────────────────────────────────
ensure_dirs()
init_db()
logger.info("SEAD-AI Flask backend ready")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 1 — MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serves the main frontend UI (Phase 11 template)."""
    try:
        return render_template("index.html")
    except Exception:
        # Fallback if template not built yet
        return """
        <h2>SEAD-AI Backend Running ✅</h2>
        <p>Frontend (Phase 11) not built yet.</p>
        <p>API is ready at <a href='/health'>/health</a></p>
        """, 200


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 2 — ANALYZE (main endpoint)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Full threat analysis endpoint.

    Request Body (JSON):
        { "text": "message to analyze" }

    Response (JSON):
        {
            "success"       : bool,
            "scan_id"       : int,
            "threat_score"  : int (0-100),
            "risk_level"    : str,
            "risk_emoji"    : str,
            "risk_color"    : str,
            "bert_score"    : float,
            "psychology_score": float,
            "formula"       : str,
            "explanation"   : dict,
            "summary"       : str,
        }
    """
    try:
        # ── Validate input ────────────────────────────────────────
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error"  : "No JSON body received. Send { 'text': '...' }"
            }), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "success": False,
                "error"  : "Text field is empty."
            }), 400

        if len(text) > 10000:
            return jsonify({
                "success": False,
                "error"  : "Text too long. Maximum 10,000 characters."
            }), 400

        logger.info(f"Analyzing text ({len(text)} chars)...")

        # ── Run BERT prediction ───────────────────────────────────
        bert_result = None
        try:
            from predictor.predict import predict
            bert_result = predict(text)
            logger.info(
                f"BERT: {bert_result['label']} "
                f"({bert_result['phishing_prob']:.2%})"
            )
        except FileNotFoundError:
            logger.warning("BERT model not found — psychology-only mode")
        except Exception as e:
            logger.warning(f"BERT error: {e} — psychology-only mode")

        # ── Calculate threat score ────────────────────────────────
        threat_result = calculate_threat_score(text, bert_result)

        # Add BERT label to result
        if bert_result:
            threat_result["bert_label"]      = bert_result.get("label", "Unknown")
            threat_result["bert_confidence"] = bert_result.get("confidence", 0.0)
        else:
            threat_result["bert_label"]      = "Unavailable"
            threat_result["bert_confidence"] = 0.0

        # ── Generate explanation ──────────────────────────────────
        explanation = generate_explanation(text, threat_result)

        # ── Save to database ──────────────────────────────────────
        scan_id = save_result(text, threat_result, explanation)

        # ── Build response ────────────────────────────────────────
        response = {
            "success"          : True,
            "scan_id"          : scan_id,
            "threat_score"     : threat_result["threat_percent"],
            "risk_level"       : threat_result["risk_level"],
            "risk_emoji"       : threat_result["risk_emoji"],
            "risk_color"       : threat_result["risk_color"],
            "risk_description" : threat_result["risk_description"],
            "bert_score"       : round(threat_result["bert_score"] * 100, 1),
            "bert_label"       : threat_result["bert_label"],
            "psychology_score" : round(threat_result["psychology_score"] * 100, 1),
            "formula"          : threat_result["formula"],
            "summary"          : explanation["summary"],
            "bert_explanation" : explanation["bert_explanation"],
            "psych_explanation": explanation["psych_explanation"],
            "reasons"          : explanation["reasons"],
            "safe_advice"      : explanation["safe_advice"],
            "confidence_note"  : explanation["confidence_note"],
            "triggered_principles": explanation["triggered_principles"],
            "suspicious_patterns" : [
                p["name"] for p in explanation["suspicious_patterns"]
            ],
            "keyword_count"    : explanation["keyword_count"],
        }

        logger.info(
            f"Scan complete — ID: {scan_id} | "
            f"Risk: {threat_result['risk_level']} | "
            f"Score: {threat_result['threat_percent']}%"
        )
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error"  : f"Analysis failed: {str(e)}"
        }), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 3 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/history", methods=["GET"])
def history():
    """
    Returns recent scan history.
    Query params: ?limit=20&offset=0
    """
    try:
        limit  = int(request.args.get("limit",  20))
        offset = int(request.args.get("offset",  0))

        # Cap limit for safety
        limit = min(limit, 100)

        scans = get_history(limit=limit, offset=offset)
        return jsonify({
            "success": True,
            "count"  : len(scans),
            "scans"  : scans,
        }), 200

    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 4 — SINGLE SCAN DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/history/<int:scan_id>", methods=["GET"])
def scan_detail(scan_id):
    """Returns full details of one scan by ID."""
    try:
        scan = get_scan_by_id(scan_id)
        if not scan:
            return jsonify({
                "success": False,
                "error"  : f"Scan ID {scan_id} not found"
            }), 404

        return jsonify({"success": True, "scan": scan}), 200

    except Exception as e:
        logger.error(f"Scan detail error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 5 — STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/stats", methods=["GET"])
def stats():
    """Returns aggregate dashboard statistics."""
    try:
        data = get_stats()
        return jsonify({"success": True, "stats": data}), 200
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 6 — SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/search", methods=["GET"])
def search():
    """
    Searches scan history.
    Query param: ?q=searchterm
    """
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({
                "success": False,
                "error"  : "Provide search query: ?q=searchterm"
            }), 400

        results = search_history(query, limit=10)
        return jsonify({
            "success": True,
            "query"  : query,
            "count"  : len(results),
            "results": results,
        }), 200

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE 7 — HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check — confirms all modules are loaded."""
    import torch

    # Check BERT model availability
    from config import MODEL_SAVE_DIR
    bert_path    = os.path.join(MODEL_SAVE_DIR, "bert_model")
    bert_ready   = os.path.exists(bert_path)

    # Check database
    db_ready = os.path.exists(os.path.dirname(
        os.path.join(os.path.dirname(__file__), "database", "sead_ai.db")
    ))

    return jsonify({
        "status"       : "ok",
        "system"       : "SEAD-AI",
        "version"      : "1.0.0",
        "bert_model"   : "ready" if bert_ready else "not trained yet",
        "database"     : "ready" if db_ready   else "not initialized",
        "gpu_available": torch.cuda.is_available(),
        "device"       : "cuda" if torch.cuda.is_available() else "cpu",
        "endpoints"    : [
            "POST /analyze",
            "GET  /history",
            "GET  /history/<id>",
            "GET  /stats",
            "GET  /search?q=term",
            "GET  /health",
        ],
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  SEAD-AI — Starting Flask Backend")
    logger.info("=" * 55)
    logger.info(f"  URL    : http://localhost:{FLASK_PORT}")
    logger.info(f"  Debug  : {FLASK_DEBUG}")
    logger.info("=" * 55)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
