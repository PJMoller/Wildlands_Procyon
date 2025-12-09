# THIS FILE WILL CONTAIN ALL THE NECESSARY FILE PATHS TO RUN ALL THE SCRIPTS #
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2] # should point to repo root, still in need of testing on multiple pc's tho
DATA_DIR = PROJECT_ROOT /"data"
RAW_DIR = DATA_DIR /"raw"
PROCESSED_DIR = DATA_DIR /"processed"
PREDICTIONS_DIR = DATA_DIR /"predictions"
MODELS_DIR = DATA_DIR /"models"
SRC_DIR = PROJECT_ROOT /"src"
NEW_ATTEMPT_DIR = SRC_DIR /"new_attempt"

"""
--- Example of usage---

from paths.py import 'desired DIR path'
df = pd.read_csv(RAW_DIR / "visitordaily.csv")
"""