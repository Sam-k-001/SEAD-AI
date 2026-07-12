/**
 * script.js — Phase 11: Frontend Logic for SEAD-AI
 * Handles all UI interactions, API calls, and result rendering.
 */

const API = "";   // empty = same origin (Flask serves both)

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — SECTION NAVIGATION
// ═══════════════════════════════════════════════════════════════════════════════

function showSection(name) {
  // Hide all sections
  document.querySelectorAll(".section").forEach(s => {
    s.classList.remove("active");
    s.classList.add("hidden");
  });
  // Show selected
  const target = document.getElementById(name + "Section");
  if (target) {
    target.classList.remove("hidden");
    target.classList.add("active");
  }
  // Update nav
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  event.target.classList.add("active");

  // Auto-load data for history/dashboard
  if (name === "history")   loadHistory();
  if (name === "dashboard") loadStats();
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — INPUT HANDLING
// ═══════════════════════════════════════════════════════════════════════════════

// Live character counter
document.getElementById("messageInput").addEventListener("input", function () {
  const len = this.value.length;
  document.getElementById("charCount").textContent = `${len.toLocaleString()} / 10,000`;
  document.getElementById("charCount").style.color =
    len > 9000 ? "var(--danger)" : len > 7000 ? "var(--warn)" : "";
});

function clearAll() {
  document.getElementById("messageInput").value = "";
  document.getElementById("charCount").textContent = "0 / 10,000";
  showEmptyState();
}

function loadExample() {
  const examples = [
    `URGENT: Your bank account has been suspended! Click here immediately to verify your identity or your account will be permanently deleted within 24 hours. Failure to respond will result in legal action. — Security Team`,

    `Congratulations! You have been selected as today's lucky winner of a FREE iPhone 15! This is a limited time offer — claim your prize NOW before it expires. Only 3 spots left!`,

    `CEO Request: I need you to urgently purchase $500 in iTunes gift cards for a client meeting today. Keep this confidential — do not tell anyone in the office. Send me the card codes immediately.`,

    `IRS NOTICE: You owe back taxes. Failure to respond within 24 hours will result in immediate arrest and legal prosecution. Call 1-800-000-0000 now to avoid penalties.`,
  ];
  const random = examples[Math.floor(Math.random() * examples.length)];
  document.getElementById("messageInput").value = random;
  document.getElementById("charCount").textContent =
    `${random.length.toLocaleString()} / 10,000`;
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — ANALYZE
// ═══════════════════════════════════════════════════════════════════════════════

async function analyze() {
  const text = document.getElementById("messageInput").value.trim();
  if (!text) {
    alert("Please paste a message to analyze.");
    return;
  }

  const btn = document.getElementById("analyzeBtn");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:16px;height:16px;border-width:2px"></span> Analyzing...`;

  showLoadingState();

  try {
    const res  = await fetch(`${API}/analyze`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!data.success) {
      alert("Error: " + (data.error || "Unknown error"));
      showEmptyState();
      return;
    }

    renderResult(data);

  } catch (err) {
    alert("Network error — is Flask running? (" + err.message + ")");
    showEmptyState();
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<span class="btn-icon">⬡</span> Analyze Message`;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — RENDER RESULT
// ═══════════════════════════════════════════════════════════════════════════════

function renderResult(data) {
  showResultContent();

  const score = data.threat_score;
  const level = data.risk_level;

  // ── Threat meter ──────────────────────────────────────────
  document.getElementById("resultEmoji").textContent     = data.risk_emoji;
  document.getElementById("resultRiskLevel").textContent = level;
  document.getElementById("resultScoreNum").textContent  = score + "%";

  // Color logic
  let color = "var(--safe)";
  if (level === "High Risk")   color = "var(--danger)";
  if (level === "Suspicious")  color = "var(--warn)";

  document.getElementById("resultRiskLevel").style.color = color;
  document.getElementById("resultScoreNum").style.color  = color;

  const fill = document.getElementById("meterFill");
  fill.style.width      = score + "%";
  fill.style.background = color;

  // ── Score breakdown ───────────────────────────────────────
  document.getElementById("bertScoreVal").textContent  = data.bert_score  + "%";
  document.getElementById("psychScoreVal").textContent = data.psychology_score + "%";
  document.getElementById("finalScoreVal").textContent = score + "%";
  document.getElementById("finalScoreVal").style.color = color;

  // ── Summary ───────────────────────────────────────────────
  document.getElementById("resultSummary").textContent = data.summary;

  // ── Triggered principles ──────────────────────────────────
  const principlesBlock = document.getElementById("principlesBlock");
  const principlesList  = document.getElementById("principlesList");
  principlesList.innerHTML = "";

  if (data.triggered_principles && data.triggered_principles.length > 0) {
    principlesBlock.classList.remove("hidden");
    data.triggered_principles.forEach(p => {
      const chip = document.createElement("span");
      chip.className   = "principle-chip";
      chip.textContent = p.principle;
      chip.title       = p.description;
      principlesList.appendChild(chip);
    });
  } else {
    principlesBlock.classList.add("hidden");
  }

  // ── Reasons ───────────────────────────────────────────────
  const reasonsBlock = document.getElementById("reasonsBlock");
  const reasonsList  = document.getElementById("reasonsList");
  reasonsList.innerHTML = "";

  if (data.reasons && data.reasons.length > 0) {
    reasonsBlock.classList.remove("hidden");
    data.reasons.forEach(r => {
      const li = document.createElement("li");
      li.textContent = r;
      reasonsList.appendChild(li);
    });
  } else {
    reasonsBlock.classList.add("hidden");
  }

  // ── Suspicious patterns ───────────────────────────────────
  const patternsBlock = document.getElementById("patternsBlock");
  const patternsList  = document.getElementById("patternsList");
  patternsList.innerHTML = "";

  if (data.suspicious_patterns && data.suspicious_patterns.length > 0) {
    patternsBlock.classList.remove("hidden");
    data.suspicious_patterns.forEach(p => {
      const chip = document.createElement("span");
      chip.className   = "pattern-chip";
      chip.textContent = "🚩 " + p;
      patternsList.appendChild(chip);
    });
  } else {
    patternsBlock.classList.add("hidden");
  }

  // ── Advice ────────────────────────────────────────────────
  document.getElementById("adviceText").textContent = data.safe_advice;
  const adviceBlock = document.getElementById("adviceBlock");
  adviceBlock.style.borderColor =
    level === "High Risk" ? "rgba(255,59,48,.3)"  :
    level === "Suspicious"? "rgba(255,149,0,.3)"  :
    "rgba(48,209,88,.2)";
  adviceBlock.style.background  =
    level === "High Risk" ? "rgba(255,59,48,.07)" :
    level === "Suspicious"? "rgba(255,149,0,.07)" :
    "rgba(48,209,88,.07)";
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 5 — STATE HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

function showEmptyState() {
  document.getElementById("emptyState").classList.remove("hidden");
  document.getElementById("loadingState").classList.add("hidden");
  document.getElementById("resultContent").classList.add("hidden");
}
function showLoadingState() {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("loadingState").classList.remove("hidden");
  document.getElementById("resultContent").classList.add("hidden");
}
function showResultContent() {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("loadingState").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 6 — HISTORY
// ═══════════════════════════════════════════════════════════════════════════════

async function loadHistory() {
  const container = document.getElementById("historyList");
  container.innerHTML = `<p class="loading-text" style="color:var(--text-2)">Loading...</p>`;

  try {
    const res  = await fetch(`${API}/history?limit=30`);
    const data = await res.json();

    if (!data.success || data.scans.length === 0) {
      container.innerHTML = `<p style="color:var(--text-2);font-size:14px">No scans yet. Analyze a message first.</p>`;
      return;
    }

    container.innerHTML = "";
    data.scans.forEach(scan => {
      const color =
        scan.risk_level === "High Risk"  ? "var(--danger)" :
        scan.risk_level === "Suspicious" ? "var(--warn)"   : "var(--safe)";

      const item = document.createElement("div");
      item.className = "history-item";
      item.innerHTML = `
        <span class="history-emoji">${scan.risk_emoji || "📧"}</span>
        <span class="history-text">${escHtml(scan.text_preview || "—")}</span>
        <div class="history-meta">
          <span class="history-score" style="color:${color}">${scan.threat_percent}%</span>
          <span class="history-risk"  style="color:${color}">${scan.risk_level}</span>
          <span class="history-time">${formatTime(scan.timestamp)}</span>
        </div>
      `;
      container.appendChild(item);
    });

  } catch (err) {
    container.innerHTML = `<p style="color:var(--danger)">Error loading history: ${err.message}</p>`;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 7 — DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════

async function loadStats() {
  try {
    const res  = await fetch(`${API}/stats`);
    const data = await res.json();
    if (!data.success) return;

    const s = data.stats;
    document.getElementById("statTotal").textContent      = s.total_scans        || 0;
    document.getElementById("statHighRisk").textContent   = s.high_risk_count     || 0;
    document.getElementById("statSuspicious").textContent = s.suspicious_count    || 0;
    document.getElementById("statSafe").textContent       = s.safe_count          || 0;
    document.getElementById("statAvgScore").textContent   = (s.avg_threat_score   || 0) + "%";
    document.getElementById("statTopTrigger").textContent =
      (s.most_common_trigger || "None").replace("_", " ");

  } catch (err) {
    console.error("Stats error:", err);
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 8 — UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

function escHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function formatTime(ts) {
  if (!ts) return "";
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleTimeString([], { hour:"2-digit", minute:"2-digit" });
}

// Enter key submits
document.getElementById("messageInput").addEventListener("keydown", function(e) {
  if (e.ctrlKey && e.key === "Enter") analyze();
});

// Load stats on page load for dashboard badge
window.addEventListener("load", () => {
  fetch(`${API}/health`)
    .then(r => r.json())
    .then(d => {
      const badge = document.getElementById("statusBadge");
      if (d.bert_model === "ready") {
        badge.innerHTML = `<span class="badge-dot"></span> BERT Ready`;
      } else {
        badge.innerHTML = `<span class="badge-dot" style="background:var(--warn);box-shadow:0 0 6px var(--warn)"></span> Psychology Mode`;
      }
    })
    .catch(() => {});
});
