"""
NECESSARY FILES TO RUN THIS CODE AND MUST ALSO BE NAMED LIKE THIS:
must all be located in data/processed

- processed_merge.csv

"""

import sys
from pathlib import Path

print("In file")

# Add project root to sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # WILDLANDS_PROCYON/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import paths (from main folder) and data cleaning
from paths import RAW_DIR, PROCESSED_DIR 
from src.model_training.model_training import process_data 

def main() -> None:
    """Run the full model training pipeline."""
    print("Starting zoo model training pipeline...")
    process_data()
    print("Model training completed!")

if __name__ == "__main__":
    main()