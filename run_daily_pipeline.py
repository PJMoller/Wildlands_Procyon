import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable  # uses the same Python you started with

def check_dependencies():
    required = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "lightgbm"
    ]
    missing = []
    for item in required:
        try:
            __import__(item)
        except ImportError:
            missing.append(item)
    if missing:
        raise RuntimeError(
            f"Missing dependencies: {missing}. "
            f"Run: pip install -r requirements.txt"
        )

check_dependencies()

scripts = [
    "scripts/run_data_cleaning.py",
    "scripts/run_model_training.py",
    "scripts/run_make_predictions.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(
        [PYTHON, script],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        print(f"Failed at {script}")
        break
    else:
        print(f"Finished {script}")
