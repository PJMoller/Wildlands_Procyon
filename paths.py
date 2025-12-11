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
NEW_ATTEMPT_DIR = SRC_DIR / "new_attempt"

print(f" PROJECT_ROOT: {PROJECT_ROOT}")  # Debug line
print(f" RAW_DIR: {RAW_DIR}")


"""
--- Example of usage---

from paths.py import 'desired DIR path'
df = pd.read_csv(RAW_DIR / "visitordaily.csv")
"""