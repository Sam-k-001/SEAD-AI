"""
utils/helpers.py — Reusable helper functions shared across modules.
"""

import os
import json
from datetime import datetime
from config import BASE_DIR


def ensure_dirs() -> None:
    """
    Creates all required directories if they don't exist.
    Call once at app startup.
    """
    dirs = [
        os.path.join(BASE_DIR, "datasets", "raw"),
        os.path.join(BASE_DIR, "datasets", "processed"),
        os.path.join(BASE_DIR, "models", "saved"),
        os.path.join(BASE_DIR, "database"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_timestamp() -> str:
    """Returns current timestamp string for logging/filenames."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_json(data: dict, filepath: str) -> None:
    """Saves a dictionary to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """Loads a JSON file and returns a dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamps a float value between min and max."""
    return max(min_val, min(max_val, value))
