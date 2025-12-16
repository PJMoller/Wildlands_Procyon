from pathlib import Path

# Find repo root by looking UP from current file location
def find_project_root(start_path):
    """Walk up directories until finding repo root (contains 'data/')"""
    path = Path(start_path).resolve()
    while path != path.parent:
        if (path / "data").exists():
            return path
        path = path.parent
    raise FileNotFoundError("Could not find project root with 'data/' folder")

PROJECT_ROOT = find_project_root(__file__)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = DATA_DIR / "models"
SRC_DIR = PROJECT_ROOT / "src"

FAMILY_MODELS_PATH = PROCESSED_DIR / "family_models.pkl"



FAMILY_MODELS_PATH = "data/processed/family_models.pkl"
FEATURE_COLS_PATH = "data/processed/feature_cols.pkl"
PROCESSED_DATA_PATH = "data/processed/processed_merge.csv"
HOLIDAY_DATA_PATH = "data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx"
CAMPAIGN_DATA_PATH = "data/raw/campaings.xlsx"
RECURRING_EVENTS_PATH = "data/raw/recurring_events_drenthe.xlsx"
SEASONALITY_PROFILE_PATH = "data/processed/ticket_seasonality.csv"
TICKET_FAMILIES_PATH = "data/processed/ticket_families.csv"



NEW_ATTEMPT_DIR = SRC_DIR / "new_attempt"

print(f" PROJECT_ROOT: {PROJECT_ROOT}")  # Debug line
print(f" RAW_DIR: {RAW_DIR}")


"""
--- Example of usage---

from paths.py import 'desired DIR path'
df = pd.read_csv(RAW_DIR / "visitordaily.csv")
"""