"""
preprocessing/clean.py — Phase 3: Data Preprocessing Pipeline for SEAD-AI
=============================================================================
This script:
  1. Loads all available datasets (sample + any Kaggle datasets found)
  2. Cleans and normalizes text
  3. Removes HTML, headers, duplicates, special characters
  4. Balances phishing vs legitimate (50/50)
  5. Splits into train/test sets
  6. Saves processed datasets ready for BERT training

Usage:
    python preprocessing/clean.py

Output:
    datasets/processed/final_dataset.csv
    datasets/processed/train.csv
    datasets/processed/test.csv
    datasets/processed/stats.json
"""

import os
import sys
import re
import json

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# ── Make config importable from any working directory ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_RAW, DATASET_PROC, TEST_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def remove_html(text: str) -> str:
    """
    Strips all HTML tags from text using BeautifulSoup.
    Example: '<b>Click here</b>' → 'Click here'
    """
    try:
        return BeautifulSoup(text, "lxml").get_text(separator=" ")
    except Exception:
        return text


def remove_email_headers(text: str) -> str:
    """
    Removes email header lines (From:, To:, Subject:, Date:, etc.)
    These add noise and don't help the model learn attack patterns.
    """
    lines = text.split("\n")
    header_patterns = re.compile(
        r"^(From|To|Cc|Bcc|Subject|Date|Message-ID|"
        r"Content-Type|MIME-Version|Return-Path|Reply-To|"
        r"Received|X-|Delivered-To):",
        re.IGNORECASE,
    )
    cleaned = [line for line in lines if not header_patterns.match(line.strip())]
    return " ".join(cleaned)


def replace_urls(text: str) -> str:
    """
    Replaces all URLs with the token [URL].
    Reason: We don't want the model to memorize specific phishing URLs.
    It should learn language patterns, not URLs.
    Example: 'Click paypa1-secure.com now' → 'Click [URL] now'
    """
    url_pattern = re.compile(
        r"http[s]?://\S+|www\.\S+|\S+\.(com|net|org|io|xyz|ru|tk)\S*",
        re.IGNORECASE,
    )
    return url_pattern.sub("[URL]", text)


def replace_phone_numbers(text: str) -> str:
    """Replaces phone numbers with [PHONE] token."""
    phone_pattern = re.compile(
        r"(\+?\d[\d\s\-().]{7,}\d)"
    )
    return phone_pattern.sub("[PHONE]", text)


def replace_emails_in_text(text: str) -> str:
    """Replaces email addresses with [EMAIL] token."""
    email_pattern = re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,4}\b")
    return email_pattern.sub("[EMAIL]", text)


def clean_text(text: str) -> str:
    """
    Master cleaning function — runs all cleaning steps in order.
    This is the main function called on every row.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    text = remove_html(text)              # strip HTML tags
    text = remove_email_headers(text)     # strip email headers
    text = replace_urls(text)             # replace URLs with [URL]
    text = replace_phone_numbers(text)    # replace phone numbers
    text = replace_emails_in_text(text)   # replace email addresses

    # Remove non-ASCII characters (keeps [URL], [PHONE], [EMAIL] tokens)
    text = text.encode("ascii", errors="ignore").decode()

    # Collapse multiple spaces/newlines into single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET LOADERS
# Each function loads one raw dataset and returns a DataFrame
# with exactly two columns: (text, label)
# ═══════════════════════════════════════════════════════════════════════════════

def load_sample_data() -> pd.DataFrame:
    """Loads the built-in sample dataset (always available)."""
    path = os.path.join(DATASET_RAW, "sample_data.csv")
    if not os.path.exists(path):
        logger.warning("sample_data.csv not found. Run download_datasets.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[["text", "label"]].copy()
    df["source"] = "sample"
    logger.info(f"Loaded sample_data.csv: {len(df)} rows")
    return df


def load_sms_spam() -> pd.DataFrame:
    """
    Loads the SMS Spam Collection dataset.
    Format: tab-separated, columns: label(ham/spam), text
    """
    # Try multiple possible filenames/paths
    possible_paths = [
        os.path.join(DATASET_RAW, "sms_spam", "SMSSpamCollection"),
        os.path.join(DATASET_RAW, "SMSSpamCollection"),
        os.path.join(DATASET_RAW, "sms_spam.txt"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t", header=None,
                             names=["label", "text"], encoding="latin-1")
            # spam=1, ham=0
            df["label"] = df["label"].map({"spam": 1, "ham": 0})
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)
            df["source"] = "sms_spam"
            logger.info(f"Loaded SMS Spam Collection: {len(df)} rows from {path}")
            return df[["text", "label", "source"]]

    logger.info("SMS Spam Collection not found — skipping.")
    return pd.DataFrame()


def load_nazario() -> pd.DataFrame:
    """
    Loads the Nazario Phishing Email dataset.
    Expected columns: 'Email Text', 'Email Type'
    """
    path = os.path.join(DATASET_RAW, "nazario_phishing.csv")
    if not os.path.exists(path):
        logger.info("nazario_phishing.csv not found — skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path, encoding="latin-1")

    # Normalize column names (different versions have different names)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    text_col  = next((c for c in df.columns if "text"  in c or "body"    in c), None)
    label_col = next((c for c in df.columns if "type"  in c or "label"   in c), None)

    if not text_col or not label_col:
        logger.warning(f"Nazario CSV columns not recognized: {list(df.columns)}")
        return pd.DataFrame()

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # Map text labels to 0/1
    df["label"] = df["label"].str.lower().str.strip()
    df["label"] = df["label"].map(lambda x: 1 if "phish" in str(x) else 0)
    df["source"] = "nazario"

    logger.info(f"Loaded Nazario corpus: {len(df)} rows")
    return df[["text", "label", "source"]]


def load_ceas() -> pd.DataFrame:
    """
    Loads CEAS 2008 / Fraudulent email corpus.
    Format: raw text file with emails separated by blank lines.
    All entries are phishing (label=1).
    """
    path = os.path.join(DATASET_RAW, "ceas_phishing.txt")
    if not os.path.exists(path):
        logger.info("ceas_phishing.txt not found — skipping.")
        return pd.DataFrame()

    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        raw = f.read()

    # Split on double newlines to get individual emails
    emails = [e.strip() for e in raw.split("\n\n") if len(e.strip()) > 30]

    df = pd.DataFrame({"text": emails, "label": 1, "source": "ceas"})
    logger.info(f"Loaded CEAS corpus: {len(df)} rows")
    return df


def load_enron() -> pd.DataFrame:
    """
    Loads Enron Email dataset (legitimate emails only, label=0).
    Samples 15,000 rows max to avoid memory issues.
    """
    path = os.path.join(DATASET_RAW, "enron_emails.csv")
    if not os.path.exists(path):
        logger.info("enron_emails.csv not found — skipping.")
        return pd.DataFrame()

    # Enron CSV is large — load only what we need
    df = pd.read_csv(path, nrows=15000, encoding="latin-1")
    if "body" in df.columns:
        df = df[["body"]].copy()
    elif "message" in df.columns:
        df = df[["message"]].copy()
    else:
        df = df[[df.columns[0]]].copy()
    df.columns = ["text"]
    df["label"]  = 0        # all Enron emails are legitimate
    df["source"] = "enron"

    logger.info(f"Loaded Enron dataset: {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_datasets() -> pd.DataFrame:
    """Loads and concatenates all available datasets."""
    logger.info("Loading all available datasets...")

    frames = []
    loaders = [
        load_sample_data,
        load_sms_spam,
        load_nazario,
        load_ceas,
        load_enron,
    ]

    for loader in loaders:
        df = loader()
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        logger.error("No datasets found! Run datasets/download_datasets.py first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows loaded: {len(combined)}")
    return combined


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline on a combined DataFrame.
    Steps: clean → filter → deduplicate → balance → report
    """
    logger.info("Starting preprocessing pipeline...")
    initial_count = len(df)

    # ── Step 1: Clean all text ──────────────────────────────────
    logger.info("Step 1/5: Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # ── Step 2: Remove empty rows ────────────────────────────────
    logger.info("Step 2/5: Removing empty rows...")
    df = df[df["text"].str.len() > 20]   # minimum 20 characters

    # ── Step 3: Remove duplicates ────────────────────────────────
    logger.info("Step 3/5: Removing duplicates...")
    df = df.drop_duplicates(subset=["text"])

    # ── Step 4: Ensure labels are valid (0 or 1 only) ───────────
    logger.info("Step 4/5: Validating labels...")
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    # ── Step 5: Balance dataset (equal phishing & legitimate) ────
    logger.info("Step 5/5: Balancing dataset...")
    phishing   = df[df["label"] == 1]
    legitimate = df[df["label"] == 0]

    # Use smaller class size so both classes are equal
    min_size = min(len(phishing), len(legitimate))
    phishing   = phishing.sample(n=min_size, random_state=42)
    legitimate = legitimate.sample(n=min_size, random_state=42)

    df = pd.concat([phishing, legitimate], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    logger.info(
        f"Pipeline complete: {initial_count} → {len(df)} rows "
        f"({min_size} phishing + {min_size} legitimate)"
    )
    return df


def save_datasets(df: pd.DataFrame) -> dict:
    """
    Splits into train/test and saves all three CSV files.
    Returns stats dictionary for reporting.
    """
    os.makedirs(DATASET_PROC, exist_ok=True)

    # ── Save full dataset ────────────────────────────────────────
    full_path = os.path.join(DATASET_PROC, "final_dataset.csv")
    df.to_csv(full_path, index=False)
    logger.info(f"Saved full dataset: {full_path}")

    # ── Train / Test split (stratified) ─────────────────────────
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],   # keeps 50/50 ratio in both splits
        random_state=42,
    )

    train_path = os.path.join(DATASET_PROC, "train.csv")
    test_path  = os.path.join(DATASET_PROC, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train set: {train_path} ({len(train_df)} rows)")
    logger.info(f"Saved test set:  {test_path} ({len(test_df)} rows)")

    # ── Save stats ───────────────────────────────────────────────
    stats = {
        "total_rows"       : len(df),
        "train_rows"       : len(train_df),
        "test_rows"        : len(test_df),
        "phishing_count"   : int(df["label"].sum()),
        "legitimate_count" : int((df["label"] == 0).sum()),
        "sources"          : df["source"].value_counts().to_dict()
                             if "source" in df.columns else {},
        "avg_text_length"  : round(df["text"].str.len().mean(), 1),
    }

    stats_path = os.path.join(DATASET_PROC, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved stats: {stats_path}")

    return stats


def print_summary(stats: dict) -> None:
    """Prints a clean summary table to the console."""
    print("\n" + "="*55)
    print("  SEAD-AI — Phase 3: Preprocessing Complete ✅")
    print("="*55)
    print(f"  Total rows       : {stats['total_rows']}")
    print(f"  Train set        : {stats['train_rows']} rows")
    print(f"  Test set         : {stats['test_rows']} rows")
    print(f"  Phishing (1)     : {stats['phishing_count']}")
    print(f"  Legitimate (0)   : {stats['legitimate_count']}")
    print(f"  Avg text length  : {stats['avg_text_length']} chars")
    print(f"\n  Sources used:")
    for src, count in stats.get("sources", {}).items():
        print(f"    - {src:20s}: {count} rows")
    print("\n  Files saved:")
    print("    datasets/processed/final_dataset.csv")
    print("    datasets/processed/train.csv")
    print("    datasets/processed/test.csv")
    print("    datasets/processed/stats.json")
    print("="*55)
    print("  Next → Run: python models/train_bert.py  (Phase 4)")
    print("="*55 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("SEAD-AI — Phase 3: Data Preprocessing")
    logger.info("=" * 55)

    # 1. Load all datasets
    raw_df = load_all_datasets()

    # 2. Run full preprocessing pipeline
    clean_df = preprocess_pipeline(raw_df)

    # 3. Save train/test splits
    stats = save_datasets(clean_df)

    # 4. Print summary
    print_summary(stats)
