# SEAD-AI — Dataset Schema Documentation (Phase 2)

## Final Combined Dataset Format

After preprocessing (Phase 3), all datasets are merged into one unified CSV:

**File:** `datasets/processed/final_dataset.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `text` | string | Cleaned email/SMS text | "Your account has been suspended..." |
| `label` | int | 1 = phishing, 0 = legitimate | `1` |
| `source` | string | Which dataset it came from | `"sms_spam"` |
| `original_length` | int | Character count before cleaning | `342` |

---

## Raw Dataset Schemas

### 1. SMS Spam Collection (UCI) — `sms_spam/SMSSpamCollection`
- **Format:** Tab-separated .txt file (no header)
- **Columns:** `label` (ham/spam), `text`
- **Size:** 5,574 rows
- **Label mapping:** `spam` → 1, `ham` → 0
- **Notes:** "spam" in SMS context = phishing/scam messages

### 2. Enron Email Dataset — `enron_emails.csv`
- **Format:** CSV
- **Columns:** `file` (path), `message` (raw email with headers)
- **Size:** ~500,000 emails (we sample 20,000)
- **Label:** All legitimate → 0
- **Notes:** Must strip email headers (From:, To:, Subject:) in preprocessing

### 3. PhishTank — `phishtank.csv`
- **Format:** CSV
- **Key Columns:** `url`, `phish_detail_url`, `verified`, `online`
- **Size:** ~60,000 entries (filter `verified=yes`)
- **Label:** All phishing → 1
- **Notes:** URL-based; we use the URL text as the message text

### 4. CEAS 2008 / Fraudulent Email Corpus — `ceas_phishing.txt`
- **Format:** Plain text (raw emails concatenated)
- **Label:** All phishing → 1
- **Notes:** Requires custom parser to split individual emails

### 5. Nazario Phishing Corpus — `nazario_phishing.csv`
- **Format:** CSV
- **Columns:** `Email Text`, `Email Type`
- **Size:** ~18,000 emails
- **Label mapping:** `Phishing Email` → 1, `Safe Email` → 0

### 6. Sample Dataset (built-in) — `sample_data.csv`
- **Format:** CSV with header
- **Columns:** `text`, `label`
- **Size:** 30 rows (15 phishing + 15 legitimate)
- **Purpose:** Immediate pipeline testing without waiting for downloads

---

## Preprocessing Strategy (Phase 3 Preview)

```
Raw Datasets
    │
    ▼
1. Load each dataset with its custom loader
    │
    ▼
2. Normalize columns → (text, label, source)
    │
    ▼
3. Clean text:
   - Strip HTML tags
   - Remove email headers (From/To/Subject)
   - Remove URLs (replace with [URL])
   - Remove special characters
   - Lowercase
   - Remove duplicates
    │
    ▼
4. Balance dataset:
   - Phishing : Legitimate = 1 : 1 (undersample majority)
    │
    ▼
5. Train/Test Split:
   - 80% train, 20% test
   - Stratified by label
    │
    ▼
6. Save:
   - datasets/processed/final_dataset.csv   (full)
   - datasets/processed/train.csv
   - datasets/processed/test.csv
```

---

## Target Dataset Size

| Split | Phishing | Legitimate | Total |
|-------|----------|------------|-------|
| Train | ~12,000 | ~12,000 | ~24,000 |
| Test  | ~3,000  | ~3,000  | ~6,000  |
| **Total** | **~15,000** | **~15,000** | **~30,000** |

Balanced 50/50 split prevents model from learning "just say legitimate" bias.
