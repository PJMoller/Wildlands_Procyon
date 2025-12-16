# python -m scripts.run_data_cleaning.py
"""
NECESSARY FILES TO RUN THIS CODE AND MUST ALSO BE NAMED LIKE THIS:
must all be located in data/raw

- campaings 2022-2026.xlsx
- recurring_events_drenthe.xlsx
- Holidays 2022-2026 Netherlands and Germany.xlsx
- visitors.csv
- weather.xlsx

"""


print("in file")

import sys
from pathlib import Path

# Add project root to sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # WILDLANDSPROCYON/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import paths (from main folder) and data cleaning
from paths import RAW_DIR, PROCESSED_DIR 
from src.data_cleaning import process_data 

# -> None is a type hint for tools like IDEs and static type checkers. 
# It helps clarify that our code does not return a value.
def main() -> None:
    """Run the full data cleaning pipeline."""
    print("Starting zoo data cleaning pipeline...")
    process_data()
    print("Data cleaning completed!")

if __name__ == "__main__":
    main()