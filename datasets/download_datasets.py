"""
datasets/download_datasets.py — Phase 2: Dataset Collection for SEAD-AI
Downloads and organizes all required datasets automatically.

Datasets used:
1. SMS Spam Collection  — UCI ML Repository (SMS phishing)
2. Enron Email Dataset  — Kaggle mirror via direct URL
3. CEAS 2008 Phishing   — Public mirror
4. Nazario Phishing     — GitHub (Jose Nazario's phishing corpus)

Run this script once before Phase 3 preprocessing.
Usage: python datasets/download_datasets.py
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Add project root to path so config.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATASET_RAW
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Dataset Registry ─────────────────────────────────────────────────────────
# Each entry: (name, url, save_filename)
DATASETS = [
    {
        "name": "SMS Spam Collection (UCI)",
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "filename": "sms_spam.zip",
        "type": "zip",
        "description": "5,574 SMS messages — ham + spam/phishing labels",
    },
]


# ─── Download Helper ──────────────────────────────────────────────────────────
def download_file(url: str, save_path: str, name: str) -> bool:
    """
    Downloads a file from URL with a progress bar.
    Returns True on success, False on failure.
    """
    try:
        logger.info(f"Downloading: {name}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as f, tqdm(
            desc=name, total=total, unit="B",
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        logger.info(f"Saved to: {save_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {name}: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extracts a zip file to a directory."""
    logger.info(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    logger.info(f"Extracted to: {extract_to}")


# ─── Manual Dataset Instructions ──────────────────────────────────────────────
def print_manual_instructions() -> None:
    """Prints instructions for datasets that require manual download."""
    print("\n" + "="*65)
    print("  MANUAL DOWNLOAD REQUIRED FOR THESE DATASETS")
    print("="*65)

    print("""
─── 1. Enron Email Dataset ───────────────────────────────────
  Why needed : 30,000+ real legitimate company emails
  Download   : https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
  Steps      :
    1. Create a free Kaggle account
    2. Download 'emails.csv'
    3. Place file at: datasets/raw/enron_emails.csv

─── 2. PhishTank Phishing URLs + Emails ──────────────────────
  Why needed : Verified real phishing email database
  Download   : https://www.phishtank.com/developer_info.php
  Steps      :
    1. Register free at phishtank.com
    2. Download 'verified_online.csv'
    3. Place file at: datasets/raw/phishtank.csv

─── 3. CEAS 2008 Spam/Phishing Dataset ───────────────────────
  Why needed : Classic benchmark phishing email dataset
  Download   : https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
  Steps      :
    1. Download 'fradulent_emails.txt'
    2. Place file at: datasets/raw/ceas_phishing.txt

─── 4. Nazario Phishing Corpus ───────────────────────────────
  Why needed : 2,000+ raw phishing emails from honeypots
  Download   : https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
  Steps      :
    1. Download 'phishing_email.csv'
    2. Place file at: datasets/raw/nazario_phishing.csv

─── 5. SpamAssassin Public Mail Corpus ───────────────────────
  Why needed : Labeled spam + ham emails (Apache project)
  Download   : https://spamassassin.apache.org/old/publiccorpus/
  Files      : 20030228_easy_ham.tar.bz2 + 20030228_spam.tar.bz2
  Steps      :
    1. Download both .tar.bz2 files
    2. Extract and place .txt files in: datasets/raw/spamassassin/

""")
    print("="*65)
    print("  After placing files, run: python preprocessing/clean.py")
    print("="*65 + "\n")


# ─── Create Sample Dataset for Testing ───────────────────────────────────────
def create_sample_dataset() -> None:
    """
    Creates a small hand-crafted sample dataset so you can test
    the full pipeline immediately even before real datasets arrive.
    Saved to: datasets/raw/sample_data.csv
    """
    import csv

    sample_path = os.path.join(DATASET_RAW, "sample_data.csv")
    if os.path.exists(sample_path):
        logger.info("Sample dataset already exists. Skipping.")
        return

    samples = [
        # (text, label)   label: 1 = phishing, 0 = legitimate
        # ── Phishing examples ──
        ("URGENT: Your bank account has been suspended! Click here immediately to verify your identity or your account will be permanently closed.", 1),
        ("Congratulations! You have been selected to receive a FREE iPhone 15. Claim your prize NOW before it expires in 24 hours!", 1),
        ("Dear Customer, we noticed suspicious activity. Please verify your PayPal credentials immediately at paypa1-secure.com", 1),
        ("Your Apple ID has been locked due to too many failed attempts. Click to unlock: apple-id-verify.net/unlock", 1),
        ("WINNER! You are today's lucky visitor. Your IP has been selected to win a $1000 Amazon gift card. Click to claim!", 1),
        ("Security Alert: Someone tried to login to your Google account from Russia. Verify NOW or account will be disabled.", 1),
        ("Dear user, your Netflix account will be cancelled today unless you update your billing information immediately.", 1),
        ("Your package could not be delivered. Pay $2.99 customs fee at: dhl-parcel-track.com/pay to release your shipment.", 1),
        ("CEO Request: I need you to urgently purchase $500 in iTunes gift cards for a client. Keep this confidential.", 1),
        ("IRS NOTICE: You owe back taxes. Failure to respond in 24 hours will result in immediate arrest. Call 1-800-000-0000.", 1),
        ("Your computer has a virus! Call Microsoft Support immediately at 1-800-111-2222 to prevent data loss.", 1),
        ("Limited time offer: Get 90% off on all products! Only 3 spots left. Enter credit card to reserve your discount.", 1),
        ("BANK ALERT: Unusual transaction of $9,842 detected. If not you, click here to dispute: secure-bankverify.com", 1),
        ("Password expiry notice: Your work email password expires in 2 hours. Click to extend: mail-portal-login.com/extend", 1),
        ("You have inherited $4.5 million from a deceased relative. Reply with your bank details to claim immediately.", 1),

        # ── Legitimate examples ──
        ("Hi team, please find attached the meeting notes from yesterday's standup. Let me know if you have any questions.", 0),
        ("Your monthly bank statement for October 2024 is now available in your online banking portal.", 0),
        ("Thank you for your order #45892. Your package has been shipped and will arrive within 3-5 business days.", 0),
        ("Hi John, can we reschedule our 2pm call to 3pm today? Something came up on my end. Thanks!", 0),
        ("Newsletter: This month in tech — top 10 programming languages of 2024 and what to learn next.", 0),
        ("Your flight booking confirmation: Delhi to Mumbai, Dec 15. Please carry a valid ID for check-in.", 0),
        ("Reminder: Your dentist appointment is scheduled for tomorrow at 10:30 AM at City Dental Clinic.", 0),
        ("We are happy to inform you that your job application has moved to the next round. Interview on Friday.", 0),
        ("Your electricity bill of Rs. 1,240 is due on November 30. Pay online to avoid late fees.", 0),
        ("Hi, I'm following up on our project proposal. Are you available for a 30-minute call this week?", 0),
        ("Your Amazon order has been delivered. If you didn't receive it, please contact customer support.", 0),
        ("Team update: The sprint review is moved to Thursday 4pm. Agenda shared in the team channel.", 0),
        ("Thank you for attending our webinar. Here's the recording link and slides from today's session.", 0),
        ("Your subscription to Netflix has been renewed for Rs. 649. Your next billing date is Dec 15.", 0),
        ("Good morning! Just a reminder that the quarterly report is due by end of day Friday.", 0),
    ]

    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(samples)

    logger.info(f"Sample dataset created: {sample_path} ({len(samples)} rows)")
    print(f"\n  ✅ Sample dataset ready at: datasets/raw/sample_data.csv")
    print(f"     {len([s for s in samples if s[1]==1])} phishing + {len([s for s in samples if s[1]==0])} legitimate messages\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(DATASET_RAW, exist_ok=True)
    logger.info("Phase 2 — Dataset Collection Started")
    print("\n" + "="*65)
    print("  SEAD-AI — Phase 2: Dataset Collection")
    print("="*65)

    # 1. Auto-download what we can
    for ds in DATASETS:
        save_path = os.path.join(DATASET_RAW, ds["filename"])
        if os.path.exists(save_path):
            logger.info(f"Already exists, skipping: {ds['name']}")
            continue

        success = download_file(ds["url"], save_path, ds["name"])

        if success and ds["type"] == "zip":
            extract_dir = os.path.join(DATASET_RAW, ds["filename"].replace(".zip", ""))
            os.makedirs(extract_dir, exist_ok=True)
            extract_zip(save_path, extract_dir)

    # 2. Create sample dataset for immediate testing
    print("\n[*] Creating sample dataset for pipeline testing...")
    create_sample_dataset()

    # 3. Print manual download instructions
    print_manual_instructions()

    logger.info("Phase 2 Complete.")


if __name__ == "__main__":
    main()
