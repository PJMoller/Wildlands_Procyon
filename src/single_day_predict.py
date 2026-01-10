import pickle
import pandas as pd
import numpy as np
from paths import MODELS_DIR, PROCESSED_DATA_PATH

# ---------------- LOAD MODELS ----------------
with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f:
    global_model = pickle.load(f)

with open(MODELS_DIR / "family_models.pkl", "rb") as f:
    family_models = pickle.load(f)

with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
    model_features = pickle.load(f)

# ---------------- LOAD DATA ----------------
processed_df = pd.read_csv(PROCESSED_DATA_PATH)
processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce").dt.normalize()
processed_df = processed_df.dropna(subset=["date"]).copy()

ticket_list = (
    processed_df[["ticket_name", "ticket_family"]]
    .drop_duplicates()
    .sort_values(["ticket_family", "ticket_name"])
    .itertuples(index=False, name=None)
)

# ---------------- HELPERS ----------------
def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _choose_model(ticket_family):
    return family_models.get(ticket_family, global_model)

# ---------------- SINGLE DAY PREDICTION ----------------
def predict_single_day(date, temperature=15.0, rain=0.0):
    results = []

    for ticket_name, ticket_family in ticket_list:
        feat = {
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "day_of_year": date.dayofyear,
            "temperature": _safe_float(temperature),
            "total_rain": _safe_float(rain),
            f"ticket_{ticket_name}": 1,
            f"family_{ticket_family}": 1,
        }

        X = (
            pd.DataFrame([feat])
            .reindex(columns=model_features, fill_value=0)
        )

        model = _choose_model(ticket_family)
        pred_raw = model.predict(X)[0]

        if not np.isfinite(pred_raw):
            pred_raw = 0.0

        pred_raw = max(0.0, float(pred_raw))

        results.append({
            "ticket_name": ticket_name,
            "ticket_family": ticket_family,
            "predicted_sales": int(round(pred_raw))
        })

    return pd.DataFrame(results)
