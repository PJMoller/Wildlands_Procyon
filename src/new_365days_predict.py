import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- Paths (robust to naming differences) ----------
from paths import PREDICTIONS_DIR, PROCESSED_DIR, MODELS_DIR, HOLIDAY_DATA_PATH, CAMPAIGN_DATA_PATH, RECURRING_EVENTS_PATH

try:
    from paths import PROCESSED_DATA_PATH
except Exception:
    PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_merge.csv"

# ---------- Optional OpenMeteo deps ----------
OPENMETEO_AVAILABLE = True
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
except Exception:
    OPENMETEO_AVAILABLE = False

# ---------- Your OpenMeteo implementation (UNCHANGED) ----------
def get_openmeteo_for_future(days=16):
    """Fetches and processes HOURLY weather data for the future."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.7862,
        "longitude": 6.8917,
        "hourly": ["temperature_2m", "rain", "precipitation"],
        "forecast_days": days
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert('Europe/Amsterdam'),
        "temperature": hourly.Variables(0).ValuesAsNumpy(),
        "rain": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy()
    }
    hourly_df = pd.DataFrame(data=hourly_data)
    hourly_df['hour'] = hourly_df['date'].dt.hour
    hourly_df["grouping_date"] = hourly_df["date"].dt.date
    hourly_df['rain_morning'] = np.where(hourly_df['hour'] < 12, hourly_df['rain'], 0)
    hourly_df['rain_afternoon'] = np.where(hourly_df['hour'] >= 12, hourly_df['rain'], 0)
    hourly_df['precip_morning'] = np.where(hourly_df['hour'] < 12, hourly_df['precipitation'], 0)
    hourly_df['precip_afternoon'] = np.where(hourly_df['hour'] >= 12, hourly_df['precipitation'], 0)
    weather_daily = hourly_df.groupby("grouping_date").agg(
        temperature=('temperature', 'mean'),
        rain_morning=('rain_morning', 'sum'),
        rain_afternoon=('rain_afternoon', 'sum'),
        precip_morning=('precip_morning', 'sum'),
        precip_afternoon=('precip_afternoon', 'sum')
    ).reset_index()
    weather_daily.rename(columns={'grouping_date': 'date'}, inplace=True)
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])
    return weather_daily.round(1)

# ---------- Helpers ----------
def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default

def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def _compute_holiday_proximity(date_ts: pd.Timestamp, holiday_dates_sorted: np.ndarray) -> tuple[int, int]:
    # holiday_dates_sorted: datetime64[ns] sorted
    if holiday_dates_sorted.size == 0:
        return 90, 90
    d = np.datetime64(date_ts.normalize())
    pos = np.searchsorted(holiday_dates_sorted, d, side="left")
    # next holiday
    if pos >= holiday_dates_sorted.size:
        next_h = d + np.timedelta64(90, "D")
    else:
        next_h = holiday_dates_sorted[pos]
    # prev holiday
    if pos == 0:
        prev_h = d - np.timedelta64(90, "D")
    else:
        prev_h = holiday_dates_sorted[pos - 1]
    days_until = int((next_h - d) / np.timedelta64(1, "D"))
    days_since = int((d - prev_h) / np.timedelta64(1, "D"))
    return days_until, days_since

def _ticket_indicators(ticket_name: str) -> dict:
    t = (ticket_name or "").lower()
    group_keywords = ["group", "groep", "family", "familie", "package"]
    return {
        "is_actie_ticket": int(("actie" in t) or ("inkoop" in t)),
        "is_abonnement_ticket": int("abonnement" in t),
        "is_full_price": int(("vol betalend" in t) or ("volbetalend" in t)),
        "is_accommodation_ticket": int("accommodatiehouder" in t),
        "is_group_ticket": int(any(k in t for k in group_keywords)),
        "is_joint_promotion": int("joint promotion" in t),
    }

def _choose_model(ticket_family: str, global_model, family_models: dict, global_wape: float, family_wape_map: dict):
    """
    Use family model iff it exists AND its heldout WAPE is strictly better than global WAPE.
    Otherwise fallback to global.
    """
    fam_model = family_models.get(ticket_family)
    if fam_model is None:
        return global_model, "global"
    fam_wape = family_wape_map.get(ticket_family, np.inf)
    # If we don't know global_wape, be conservative and require family_wape < 50
    if np.isfinite(global_wape):
        if fam_wape < global_wape:
            return fam_model, "family"
        return global_model, "global"
    else:
        if fam_wape < 50.0:
            return fam_model, "family"
        return global_model, "global"

# ---------- Main ----------
def predict_next_365_days():
    # Load models + features
    model_path = MODELS_DIR / "lgbm_model.pkl"
    family_models_path = MODELS_DIR / "family_models.pkl"
    feature_cols_path = MODELS_DIR / "feature_cols.pkl"
    perf_path = MODELS_DIR / "model_performance.pkl"

    with open(model_path, "rb") as f:
        global_model = pickle.load(f)
    with open(family_models_path, "rb") as f:
        family_models = pickle.load(f)
    with open(feature_cols_path, "rb") as f:
        model_features = pickle.load(f)

    print(f"Predicting with {len(model_features)} features")

    # Performance (for model choice)
    global_wape = np.inf
    family_wape_map = {}
    try:
        with open(perf_path, "rb") as f:
            perf = pickle.load(f)
        global_wape = _safe_float(perf.get("global_model", {}).get("WAPE", np.inf), np.inf)
        family_wape_map = perf.get("family_models_heldout", {}) or {}
        # family_models_heldout is dict family -> metrics dict; normalize to family->WAPE float
        family_wape_map = {k: _safe_float(v.get("WAPE", np.inf), np.inf) for k, v in family_wape_map.items()}
    except Exception:
        pass

    # Load processed data
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    if "date" not in processed_df.columns:
        raise RuntimeError("processed_merge.csv must contain a 'date' column")
    processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce")
    processed_df = processed_df.dropna(subset=["date"]).copy()

    bt = processed_df.sort_values("date").copy()
    bt_last_date = bt["date"].max()
    bt_start = bt_last_date - pd.Timedelta(days=30)
    bt = bt[bt["date"] >= bt_start]

    X_bt = bt.reindex(columns=model_features, fill_value=0).copy()
    for c in X_bt.columns:
        if X_bt[c].dtype == "object":
            X_bt[c] = pd.to_numeric(X_bt[c], errors="coerce").fillna(0)

    y_bt = bt["ticket_num"].astype(float).values
    yhat_bt = global_model.predict(X_bt)

    print("\n=== BACKTEST (last 30 days, global model) ===")
    print("Actual total:", int(y_bt.sum()))
    print("Pred total:  ", int(np.maximum(0, yhat_bt).sum()))
    print("Ratio pred/actual:", float(np.maximum(0, yhat_bt).sum() / max(1.0, y_bt.sum())))


    # Ensure ticket columns exist
    if "ticket_name" not in processed_df.columns or "ticket_family" not in processed_df.columns:
        raise RuntimeError("processed_merge.csv must contain 'ticket_name' and 'ticket_family' columns")

    # Core sets
    processed_df["month"] = pd.to_datetime(processed_df["date"]).dt.month
    processed_df["day"] = pd.to_datetime(processed_df["date"]).dt.day

    last_known_date = processed_df["date"].max().normalize()
    tickets_df = processed_df[["ticket_name", "ticket_family"]].drop_duplicates().sort_values(["ticket_family", "ticket_name"])
    ticket_list = list(tickets_df.itertuples(index=False, name=None))  # (ticket_name, ticket_family)

    families = sorted(processed_df["ticket_family"].dropna().unique().tolist())

    # --- External data pre-processing (holidays / campaigns / events) ---
    # Holidays: build dummy columns exactly like your processing script (leading "_" is expected)
    holiday_og_df = pd.read_excel(HOLIDAY_DATA_PATH)
    # accept both 6 or 7 columns, but we only need regions + date
    # expected: ["NLNoord","NLMidden","NLZuid","Niedersachsen","Nordrhein-Westfalen","date", maybe "week"]
    # find date column
    if "Datum" not in [c.lower() for c in holiday_og_df.columns]:
        # if file has "Date" or something: try to find
        date_col = [c for c in holiday_og_df.columns if str(c).lower().strip() == "datum"]
        if not date_col:
            raise RuntimeError("Holiday file must contain a 'datum' column")
    holiday_og_df = holiday_og_df.copy()
    if "week" in [str(c).lower().strip() for c in holiday_og_df.columns]:
        # drop "week" if present
        for c in holiday_og_df.columns:
            if str(c).lower().strip() == "week":
                holiday_og_df = holiday_og_df.drop(columns=[c])
                break

    # Force known layout if file has unnamed columns
    # Use the last column as date if needed
    cols = list(holiday_og_df.columns)
    if len(cols) >= 6:
        # assume last col is date
        # try to align to known names
        region_cols = cols[:5]
        date_col = cols[5]
    else:
        raise RuntimeError("Holiday file does not have expected columns")

    long_df = holiday_og_df.melt(id_vars=[date_col], value_vars=region_cols, var_name="region", value_name="holiday")
    long_df.rename(columns={date_col: "date"}, inplace=True)
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df = long_df.dropna(subset=["date"]).copy()
    long_df["holiday"] = long_df["holiday"].fillna("None").astype(str).str.strip().replace("", "None")
    long_df["region_holiday"] = long_df["region"].astype(str) + "_" + long_df["holiday"].astype(str)

    # One-hot with leading "_" to match training
    holiday_dummies = pd.get_dummies(long_df.set_index("date")["region_holiday"], prefix="", prefix_sep="_")
    holiday_daily = holiday_dummies.groupby(level=0).sum().reset_index()
    # holiday intensity (count unique holidays per date) – keep the same column name
    holiday_intensity = long_df.groupby("date")["holiday"].nunique().reset_index(name="holiday_intensity")
    holiday_daily = holiday_daily.merge(holiday_intensity, on="date", how="left")
    holiday_daily["date"] = pd.to_datetime(holiday_daily["date"]).dt.normalize()

    holiday_lookup = holiday_daily.set_index("date").to_dict(orient="index")
    holiday_dates_sorted = np.array(sorted(holiday_daily["date"].unique()), dtype="datetime64[ns]")

    # Campaigns
    camp_og_df = pd.read_excel(CAMPAIGN_DATA_PATH)
    camp_og_df = camp_og_df.copy()
    # enforce expected structure: year, week + promo_...
    camp_og_df.rename(columns={c: str(c).strip() for c in camp_og_df.columns}, inplace=True)
    if "Week " in camp_og_df.columns and "week" not in camp_og_df.columns:
        camp_og_df.rename(columns={"Week ": "week"}, inplace=True)
    if "week" not in camp_og_df.columns:
        # sometimes "Week" appears
        if "Week" in camp_og_df.columns:
            camp_og_df.rename(columns={"Week": "week"}, inplace=True)
    promo_cols = [c for c in camp_og_df.columns if str(c).startswith("promo_")]
    camp_og_df["campaign_strength"] = camp_og_df[promo_cols].sum(axis=1) if promo_cols else 0
    camp_og_df["promotion_active"] = (camp_og_df["campaign_strength"] > 0).astype(int)
    camp_og_df["campaign_regions_active"] = (camp_og_df[promo_cols] > 0).sum(axis=1) if promo_cols else 0
    camp_lookup = {
        (int(r["year"]), int(r["week"])): {
            "campaign_strength": _safe_float(r["campaign_strength"], 0.0),
            "promotion_active": _safe_int(r["promotion_active"], 0),
            "campaign_regions_active": _safe_int(r["campaign_regions_active"], 0),
        }
        for _, r in camp_og_df.iterrows()
        if "year" in camp_og_df.columns and "week" in camp_og_df.columns and pd.notna(r.get("year")) and pd.notna(r.get("week"))
    }

    # Events
    recurring_og_df = pd.read_excel(RECURRING_EVENTS_PATH)
    recurring_og_df = recurring_og_df.copy()
    # expected columns: event_name, date
    if recurring_og_df.shape[1] >= 2:
        recurring_og_df.columns = ["event_name", "date"] + list(recurring_og_df.columns[2:])
        recurring_og_df = recurring_og_df[["event_name", "date"]]
    recurring_og_df["event_name"] = recurring_og_df["event_name"].fillna("").astype(str).str.split("/")
    recurring_df = recurring_og_df.explode("event_name")
    recurring_df["event_name"] = (
        recurring_df["event_name"].astype(str).str.strip().str.replace(" ", "_").str.lower()
        .str.replace(r"^fc_emmen_.*", "soccer", regex=True)
    )
    recurring_df.loc[recurring_df["event_name"] == "", "event_name"] = "no_event"
    recurring_df["date"] = pd.to_datetime(recurring_df["date"], errors="coerce").dt.normalize()
    recurring_df = recurring_df.dropna(subset=["date"]).drop_duplicates(subset=["date", "event_name"])

    events_pivot = recurring_df.pivot_table(index="date", columns="event_name", aggfunc="size", fill_value=0).astype(int)
    events_pivot.columns = [f"event_{c}" for c in events_pivot.columns]
    events_lookup = events_pivot.to_dict(orient="index")  # date -> {event_x:0/1}

    # Weather: 16-day OpenMeteo + historical month/day avg fallback
    weather_future = pd.DataFrame()
    if OPENMETEO_AVAILABLE:
        try:
            weather_future = get_openmeteo_for_future(days=16).copy()
            weather_future["date"] = pd.to_datetime(weather_future["date"], errors="coerce").dt.normalize()
        except Exception:
            weather_future = pd.DataFrame()

    hist_weather = (
        processed_df.groupby(["month", "day"], as_index=False)[
            ["temperature", "rain_morning", "rain_afternoon", "precip_morning", "precip_afternoon"]
        ].mean()
    )
    hist_weather_key = {(int(r["month"]), int(r["day"])): r for _, r in hist_weather.iterrows()}

    # --- Build initial per-ticket daily history series (INCLUDING zeros) ---
    processed_df = processed_df.sort_values(["ticket_name", "date"])
    ticket_history = {}
    for tname, _tfam in ticket_list:
        s = processed_df.loc[processed_df["ticket_name"] == tname, ["date", "ticket_num"]].copy()
        if s.empty:
            ticket_history[tname] = pd.Series(dtype=float)
            continue
        s = s.set_index("date")["ticket_num"].astype(float)
        # normalize index to midnight
        s.index = pd.to_datetime(s.index).normalize()
        # keep full history; it already contains zeros for expanded dates
        ticket_history[tname] = s

    # Seasonal baseline per ticket per month (simple but helps scale, without calibration)
    seasonal_ticket_month = processed_df.groupby(["ticket_name", "month"])["ticket_num"].mean().to_dict()

    # Availability state (if present)
    has_is_available = "is_available" in processed_df.columns
    last_avail = {}
    last_days_since_avail = {}
    if has_is_available:
        last_rows = processed_df.sort_values("date").groupby("ticket_name").tail(1)
        for _, r in last_rows.iterrows():
            tname = r["ticket_name"]
            last_avail[tname] = _safe_int(r.get("is_available", 1), 1)
            last_days_since_avail[tname] = _safe_int(r.get("days_since_available", 0), 0)
    else:
        for tname, _ in ticket_list:
            last_avail[tname] = 1
            last_days_since_avail[tname] = 0

    # Family sales features: names expected like "family_<fam>_sales"
    family_sales_feature_names = [c for c in model_features if c.startswith("family_") and c.endswith("_sales")]
    family_names_for_sales = [c[len("family_"):-len("_sales")] for c in family_sales_feature_names]

    # Initialize prev-day family totals from last known date (actual)
    fam_daily_actual = processed_df.groupby(["date", "ticket_family"])["ticket_num"].sum()
    prev_family_totals = {fam: 0.0 for fam in family_names_for_sales}
    for fam in family_names_for_sales:
        prev_family_totals[fam] = float(fam_daily_actual.get((last_known_date, fam), 0.0))

    # Sanity diagnostic (recent actual daily total)
    recent_actual_daily = processed_df.groupby("date")["ticket_num"].sum().tail(30).mean()

    # ---------- Forecast loop ----------
    all_rows = []
    for step in range(365):
        current_date = (last_known_date + timedelta(days=step + 1)).normalize()
        year = current_date.year
        week = int(current_date.isocalendar().week)
        weekday = int(current_date.weekday())
        month = int(current_date.month)
        day = int(current_date.day)
        doy = int(current_date.dayofyear)
        is_weekend = int(weekday >= 5)

        # Weather for the day: OpenMeteo if available, else historical avg
        w_row = None
        if not weather_future.empty:
            m = weather_future.loc[weather_future["date"] == current_date]
            if not m.empty:
                w_row = m.iloc[0].to_dict()

        if w_row is None:
            hw = hist_weather_key.get((month, day))
            if hw is not None:
                w_row = hw
            else:
                w_row = {"temperature": 10.0, "rain_morning": 0.0, "rain_afternoon": 0.0, "precip_morning": 0.0, "precip_afternoon": 0.0}

        temperature = _safe_float(w_row.get("temperature", 10.0), 10.0)
        rain_morning = _safe_float(w_row.get("rain_morning", 0.0), 0.0)
        rain_afternoon = _safe_float(w_row.get("rain_afternoon", 0.0), 0.0)
        precip_morning = _safe_float(w_row.get("precip_morning", 0.0), 0.0)
        precip_afternoon = _safe_float(w_row.get("precip_afternoon", 0.0), 0.0)

        # Holiday proximity
        days_until_holiday, days_since_holiday = _compute_holiday_proximity(current_date, holiday_dates_sorted)

        # Holiday one-hot row (already includes holiday_intensity)
        holiday_feats = holiday_lookup.get(current_date, {})
        holiday_intensity_val = _safe_float(holiday_feats.get("holiday_intensity", 0.0), 0.0)

        # Campaign feats
        camp_feats = camp_lookup.get((year, week), {"campaign_strength": 0.0, "promotion_active": 0, "campaign_regions_active": 0})

        # Events feats
        event_feats = events_lookup.get(current_date, {})

        # Precompute date+weather derived features shared across tickets
        is_month_start = int(day in (1, 2, 3))
        is_month_mid = int(day in (14, 15, 16))
        is_month_end = int(pd.Timestamp(current_date).is_month_end)

        day_of_year_sin = float(np.sin(2 * np.pi * doy / 365.25))
        day_of_year_cos = float(np.cos(2 * np.pi * doy / 365.25))
        month_sin = float(np.sin(2 * np.pi * month / 12))
        month_cos = float(np.cos(2 * np.pi * month / 12))

        christmas = pd.Timestamp(f"{year}-12-25")
        days_until_christmas = max(0, int((christmas - current_date).days))
        is_christmas_build_up = int(month == 12 and 1 <= day <= 24)
        is_peak_christmas_week = int(month == 12 and 18 <= day <= 24)
        is_twixtmas_period = int(month == 12 and 26 <= day <= 30)
        is_kings_day = int(month == 4 and day == 27)

        temp_x_weekend = temperature * is_weekend
        is_perfect_day = int((temperature > 20) and (rain_morning == 0) and (rain_afternoon == 0))
        temp_x_holiday_intensity = temperature * holiday_intensity_val

        # Predict all tickets for this date
        day_preds = []
        day_family_totals = {fam: 0.0 for fam in family_names_for_sales}

        for ticket_name, ticket_family in ticket_list:
            ind = _ticket_indicators(ticket_name)
            campaign_x_actie = _safe_float(camp_feats.get("campaign_strength", 0.0), 0.0) * ind["is_actie_ticket"]
            promotion_x_actie = _safe_int(camp_feats.get("promotion_active", 0), 0) * ind["is_actie_ticket"]
            weekend_x_group = is_weekend * ind["is_group_ticket"]

            # Availability state
            is_available = int(last_avail.get(ticket_name, 1))
            dsa = int(last_days_since_avail.get(ticket_name, 0))
            # simplistic forward state: if available stays available; increment days_since_available if available
            days_since_available = (dsa + 1) if is_available == 1 else 0

            # Lags/rolling from evolving history (we update history with predictions each day)
            hist = ticket_history.get(ticket_name)
            if hist is None or hist.empty:
                hist = pd.Series(dtype=float)

            def hist_get(ts):
                try:
                    return float(hist.get(ts, 0.0))
                except Exception:
                    return 0.0

            lag_vals = {}
            for lag in (1, 2, 3, 7, 14, 21, 30):
                lag_date = (current_date - timedelta(days=lag)).normalize()
                lag_vals[f"sales_lag_{lag}"] = hist_get(lag_date)

            roll_vals = {}
            prev_day = (current_date - timedelta(days=1)).normalize()
            hist_upto = hist.loc[hist.index <= prev_day] if not hist.empty else pd.Series(dtype=float)

            for w in (7, 14, 30):
                window = hist_upto.tail(w)
                if len(window) > 0:
                    roll_vals[f"sales_rolling_avg_{w}"] = float(window.mean())
                    roll_vals[f"sales_rolling_std_{w}"] = float(window.std(ddof=0))  # stable
                    roll_vals[f"sales_rolling_min_{w}"] = float(window.min())
                    roll_vals[f"sales_rolling_max_{w}"] = float(window.max())
                else:
                    roll_vals[f"sales_rolling_avg_{w}"] = 0.0
                    roll_vals[f"sales_rolling_std_{w}"] = 0.0
                    roll_vals[f"sales_rolling_min_{w}"] = 0.0
                    roll_vals[f"sales_rolling_max_{w}"] = 0.0

            sales_momentum_7d = lag_vals.get("sales_lag_1", 0.0) - lag_vals.get("sales_lag_7", 0.0)
            sales_trend_30d = lag_vals.get("sales_lag_1", 0.0) - lag_vals.get("sales_lag_30", 0.0)

            # Family sales features = prev-day totals (dynamic, based on predicted totals after last_known_date)
            fam_sales_feats = {}
            for fam in family_names_for_sales:
                fam_sales_feats[f"family_{fam}_sales"] = float(prev_family_totals.get(fam, 0.0))

            # One-hot ticket/family (CRITICAL)
            one_hot = {
                f"ticket_{ticket_name}": 1,
                f"family_{ticket_family}": 1,
            }

            # Assemble feature dict
            feat = {
                "year": year,
                "week": week,
                "weekday": weekday,
                "day": day,
                "month": month,
                "day_of_year": doy,
                "day_of_year_sin": day_of_year_sin,
                "day_of_year_cos": day_of_year_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "is_weekend": is_weekend,
                "is_month_start": is_month_start,
                "is_month_mid": is_month_mid,
                "is_month_end": is_month_end,
                "days_until_christmas": days_until_christmas,
                "is_christmas_build_up": is_christmas_build_up,
                "is_peak_christmas_week": is_peak_christmas_week,
                "is_twixtmas_period": is_twixtmas_period,
                "is_kings_day": is_kings_day,
                "temperature": temperature,
                "rain_morning": rain_morning,
                "rain_afternoon": rain_afternoon,
                "precip_morning": precip_morning,
                "precip_afternoon": precip_afternoon,
                "temp_x_weekend": temp_x_weekend,
                "is_perfect_day": is_perfect_day,
                "days_until_holiday": days_until_holiday,
                "days_since_holiday": days_since_holiday,
                "holiday_intensity": holiday_intensity_val,
                "temp_x_holiday_intensity": temp_x_holiday_intensity,
                "campaign_strength": _safe_float(camp_feats.get("campaign_strength", 0.0), 0.0),
                "promotion_active": _safe_int(camp_feats.get("promotion_active", 0), 0),
                "campaign_regions_active": _safe_int(camp_feats.get("campaign_regions_active", 0), 0),
                "campaign_x_actie": campaign_x_actie,
                "promotion_x_actie": promotion_x_actie,
                "weekend_x_group": weekend_x_group,
                "is_available": is_available,
                "days_since_available": days_since_available,
                "sales_momentum_7d": sales_momentum_7d,
                "sales_trend_30d": sales_trend_30d,
                **ind,
                **lag_vals,
                **roll_vals,
                **fam_sales_feats,
                **holiday_feats,   # includes the holiday dummy columns + holiday_intensity already
                **event_feats,
                **one_hot,
            }

            X = pd.DataFrame([feat]).reindex(columns=model_features, fill_value=0)

            # ensure numeric
            for c in X.columns:
                if X[c].dtype == "object":
                    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

            chosen_model, model_used = _choose_model(ticket_family, global_model, family_models, global_wape, family_wape_map)
            pred_raw = float(chosen_model.predict(X)[0])

            # Light seasonal blend (keeps scale without calibration)
            seasonal = float(seasonal_ticket_month.get((ticket_name, month), 0.0))
            pred = max(0.0, 0.95 * pred_raw + 0.05 * seasonal)

            # Store
            day_preds.append((ticket_name, ticket_family, pred, model_used))

        # Update family totals and ticket histories (so lags/rollings stay realistic across 365d)
        for ticket_name, ticket_family, pred, model_used in day_preds:
            # update history
            s = ticket_history.get(ticket_name)
            if s is None:
                s = pd.Series(dtype=float)
            # assign prediction to this date
            s.loc[current_date] = float(pred)
            ticket_history[ticket_name] = s

            # update family totals for next day features
            if ticket_family in day_family_totals:
                day_family_totals[ticket_family] += float(pred)

            # update availability counters
            if last_avail.get(ticket_name, 1) == 1:
                last_days_since_avail[ticket_name] = int(last_days_since_avail.get(ticket_name, 0) + 1)
            else:
                last_days_since_avail[ticket_name] = 0

            all_rows.append({
                "date": current_date,
                "ticket_name": ticket_name,
                "ticket_family": ticket_family,
                "predicted_sales": float(pred),
                "model_used": model_used
            })

        # roll prev_family_totals forward (for next day)
        for fam in family_names_for_sales:
            prev_family_totals[fam] = float(day_family_totals.get(fam, 0.0))

        # progress
        if step in (0, 15, 30, 60, 120, 240, 364):
            daily_total = sum(p for _, _, p, _ in day_preds)
            print(f"{current_date.date()} total={daily_total:.1f}")

    # ---------- Output ----------
    forecast_df = pd.DataFrame(all_rows)
    forecast_df.sort_values(["date", "ticket_family", "ticket_name"], inplace=True)

    # Round predicted_sales (tickets are counts)
    forecast_df["predicted_sales"] = forecast_df["predicted_sales"].round(0).astype(int)

    out_name = f"forecast_365days_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    out_path = PREDICTIONS_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(out_path, index=False)

    # Diagnostics + rounded summary prints
    forecast_daily_avg = forecast_df.groupby("date")["predicted_sales"].sum().mean()
    print("\n=== SUMMARY ===")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(forecast_df):,}")
    print(f"Total predicted sales: {int(forecast_df['predicted_sales'].sum()):,}")
    print(f"Avg daily total (rounded): {int(round(forecast_daily_avg, 0)):,}")
    print(f"Recent 30d actual avg daily (rounded): {int(round(recent_actual_daily, 0)):,}")

if __name__ == "__main__":
    predict_next_365_days()
