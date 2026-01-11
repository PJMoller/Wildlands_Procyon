"""
NECESSARY FILES TO RUN THIS CODE AND MUST ALSO BE NAMED LIKE THIS:
must all be located correctly via paths.py

Required:
- processed_merge.csv          (PROCESSED_DATA_PATH)
- global_model.pkl             (in MODELS_DIR)
- family models pickle         (FAMILY_MODELS_PATH)
- feature columns pickle       (FEATURE_COLS_PATH)
- holiday data                 (HOLIDAY_DATA_PATH)
- campaign data                (CAMPAIGN_DATA_PATH)
- recurring events             (RECURRING_EVENTS_PATH)
- ticket families mapping      (TICKET_FAMILIES_PATH)
"""

import sys
from pathlib import Path

print("In run_make_predictions.py")

# Add project root to sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # WILDLANDS_PROCYON/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import prediction pipeline
from src.current_365days_predict import predict_next_365_days  # adjust path if needed


def main() -> None:
    """Run the 365-day prediction pipeline."""
    print("Starting 365-day zoo prediction pipeline...")
    predict_next_365_days()
    print("365-day prediction completed!")


if __name__ == "__main__":
    main()
