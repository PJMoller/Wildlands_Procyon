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
IMG_DIR = DATA_DIR / "img"


FAMILY_MODELS_PATH = MODELS_DIR / "family_models.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"

PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_merge.csv"
HOLIDAY_DATA_PATH = RAW_DIR / "Holidays 2022-2026 Netherlands and Germany.xlsx"
CAMPAIGN_DATA_PATH = RAW_DIR / "campaigns 2022-2026.xlsx"
RECURRING_EVENTS_PATH = RAW_DIR / "recurring_events_drenthe.xlsx"
TICKET_FAMILIES_PATH = RAW_DIR / "ticketfamilies.xlsx"

print(f" PROJECT_ROOT: {PROJECT_ROOT}")  # Debug line
print(f" RAW_DIR: {RAW_DIR}")


"""
--- Example of usage---

from paths.py import 'desired DIR path'
df = pd.read_csv(RAW_DIR / "visitordaily.csv")
"""