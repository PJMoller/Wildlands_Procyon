import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================
from paths import (
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    HOLIDAY_DATA_PATH,
    CAMPAIGN_DATA_PATH,
    RECURRING_EVENTS_PATH,
)

try:
    from paths import PROCESSED_DATA_PATH
except Exception:
    PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_merge.csv"

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================
OPENMETEO_AVAILABLE = True
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
except Exception:
    OPENMETEO_AVAILABLE = False


# =============================================================================
# HELPER FUNCTIONS: EXTREME SEASONALITY & DYNAMIC TRENDS
# =============================================================================

def _compute_extreme_multipliers(processed_df: pd.DataFrame) -> dict:
    """
    Calculates the intensity of peaks/troughs relative to a 'normal' baseline.
    Used to inject shape into the forecast.
    """
    # 1. Establish a 'Normal' Baseline (Spring/Autumn)
    baseline_months = [3, 4, 5, 9, 10]
    
    baseline_data = processed_df[processed_df['date'].dt.month.isin(baseline_months)]
    if baseline_data.empty:
        baseline_avg = processed_df.groupby('date')['ticket_num'].sum().mean()
    else:
        baseline_avg = baseline_data.groupby('date')['ticket_num'].sum().mean()
        
    if baseline_avg == 0: baseline_avg = 1.0

    # 2. Christmas Peak Intensity (Dec 18-24)
    xmas_data = processed_df[
        (processed_df['date'].dt.month == 12) & 
        (processed_df['date'].dt.day.between(18, 24))
    ]
    xmas_avg = xmas_data.groupby('date')['ticket_num'].sum().mean() if not xmas_data.empty else baseline_avg * 4.0
    
    # 3. Low Season Depth (Jan, Feb, Nov)
    low_data = processed_df[processed_df['date'].dt.month.isin([1, 2, 11])]
    low_avg = low_data.groupby('date')['ticket_num'].sum().mean() if not low_data.empty else baseline_avg * 0.3

    return {
        'baseline': float(baseline_avg),
        'xmas_multiplier': float(xmas_avg / baseline_avg),
        'low_season_multiplier': float(low_avg / baseline_avg)
    }


def _get_adaptive_scaling_bounds(current_date: datetime, fam_wape: float) -> tuple[float, float]:
    """
    Returns dynamic clipping bounds.
    Loosens restrictions during known extreme periods so the model can reach peaks/troughs.
    """
    month = current_date.month
    day = current_date.day
    
    # If model is trusted (low error), we allow it more freedom
    is_trusted = fam_wape < 15.0 

    # --- CHRISTMAS PEAK (Dec 15 - Dec 30) ---
    if month == 12 and 15 <= day <= 30:
        return (0.8, 5.0) # Allow up to 5x upward scaling
        
    # --- LOW SEASON (Jan, Feb, Nov) ---
    elif month in [1, 2, 11]:
        return (0.1, 1.5) # Allow dropping to 10%
        
    # --- NORMAL SEASON ---
    else:
        if is_trusted:
            return (0.85, 1.15)
        else:
            return (0.60, 1.70)


def _apply_extreme_shape_injection(pred: float, current_date: datetime, multipliers: dict) -> float:
    """
    Enforces specific curve shapes (exponential ramps) that tree models often smooth out.
    """
    month = current_date.month
    day = current_date.day
    
    # Christmas Ramp (Dec 18 - 24)
    if month == 12 and 18 <= day <= 24:
        # Progress 0.0 to 1.0
        progress = (day - 17) / 7.0 
        target_mult = multipliers['xmas_multiplier']
        baseline = multipliers['baseline']
        ideal_val = baseline * target_mult
        
        # If prediction is lagging behind the ideal curve, pull it up
        if pred < ideal_val:
            weight = progress * 0.6  # Influence grows as we get closer to Xmas
            return (pred * (1 - weight)) + (ideal_val * weight)

    return pred


# =============================================================================
# STANDARD UTILS
# =============================================================================

def get_openmeteo_for_future(days=16):
    """Fetches HOURLY weather data for the future."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.7862,
        "longitude": 6.8917,
        "hourly": ["temperature_2m", "rain", "precipitation"],
        "forecast_days": days,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.to_datetime(hourly.Time(), unit="s", utc=True).tz_convert("Europe/Amsterdam"),
        "temperature": hourly.Variables(0).ValuesAsNumpy(),
        "rain": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy(),
    }

    hourly_df = pd.DataFrame(data=hourly_data)
    hourly_df["hour"] = hourly_df["date"].dt.hour
    hourly_df["grouping_date"] = hourly_df["date"].dt.date
    hourly_df["rain_morning"] = np.where(hourly_df["hour"] < 12, hourly_df["rain"], 0)
    hourly_df["rain_afternoon"] = np.where(hourly_df["hour"] >= 12, hourly_df["rain"], 0)
    hourly_df["precip_morning"] = np.where(hourly_df["hour"] < 12, hourly_df["precipitation"], 0)
    hourly_df["precip_afternoon"] = np.where(hourly_df["hour"] >= 12, hourly_df["precipitation"], 0)

    weather_daily = (
        hourly_df.groupby("grouping_date")
        .agg(
            temperature=("temperature", "mean"),
            rain_morning=("rain_morning", "sum"),
            rain_afternoon=("rain_afternoon", "sum"),
            precip_morning=("precip_morning", "sum"),
            precip_afternoon=("precip_afternoon", "sum"),
        )
        .reset_index()
    )

    weather_daily.rename(columns={"grouping_date": "date"}, inplace=True)
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])
    return weather_daily.round(1)


def _safe_int(x, default=0):
    try: return int(x) if not pd.isna(x) else default
    except: return default

def _safe_float(x, default=0.0):
    try: return float(x) if not pd.isna(x) else default
    except: return default

def _compute_holiday_proximity(date_ts: pd.Timestamp, holiday_dates_sorted: np.ndarray) -> tuple[int, int]:
    if holiday_dates_sorted.size == 0: return 90, 90
    d = np.datetime64(date_ts.normalize())
    pos = np.searchsorted(holiday_dates_sorted, d, side="left")

    if pos >= holiday_dates_sorted.size:
        next_h = d + np.timedelta64(90, "D")
    else:
        next_h = holiday_dates_sorted[pos]

    if pos == 0:
        prev_h = d - np.timedelta64(90, "D")
    else:
        prev_h = holiday_dates_sorted[pos - 1]

    days_until = int((next_h - d) / np.timedelta64(1, "D"))
    days_since = int((d - prev_h) / np.timedelta64(1, "D"))
    return days_until, days_since

def _choose_model(ticket_family: str, global_model, family_models: dict, global_wape: float, family_wape_map: dict):
    fam_model = family_models.get(ticket_family)
    if fam_model is None: return global_model, "global"
    fam_wape = family_wape_map.get(ticket_family, np.inf)
    if np.isfinite(global_wape):
        if fam_wape < global_wape: return fam_model, "family"
        return global_model, "global"
    if fam_wape < 50.0: return fam_model, "family"
    return global_model, "global"

def _build_holiday_features(holiday_path: str | os.PathLike) -> tuple[dict, np.ndarray]:
    holiday_og_df = pd.read_excel(holiday_path).copy()
    for c in list(holiday_og_df.columns):
        if str(c).lower().strip() == "week":
            holiday_og_df = holiday_og_df.drop(columns=[c])
            break
            
    cols = list(holiday_og_df.columns)
    region_cols = cols[:5]
    date_col = cols[5]

    long_df = holiday_og_df.melt(id_vars=[date_col], value_vars=region_cols, var_name="region", value_name="holiday")
    long_df.rename(columns={date_col: "date"}, inplace=True)
    long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
    long_df = long_df.dropna(subset=["date"]).copy()
    long_df["holiday"] = long_df["holiday"].fillna("None").astype(str).str.strip().replace("", "None")
    long_df["region_holiday"] = long_df["region"].astype(str) + "_" + long_df["holiday"].astype(str)

    encoded = pd.get_dummies(long_df.set_index("date")["region_holiday"], prefix="", prefix_sep="_")
    holiday_daily = encoded.groupby(level=0).sum().reset_index()

    holiday_intensity = long_df.groupby("date")["holiday"].apply(lambda x: x.dropna().nunique()).reset_index(name="holiday_intensity")
    holiday_daily = holiday_daily.merge(holiday_intensity, on="date", how="left")
    holiday_daily["date"] = pd.to_datetime(holiday_daily["date"]).dt.normalize()

    holiday_lookup = holiday_daily.set_index("date").to_dict(orient="index")
    holiday_dates_sorted = np.array(sorted(holiday_daily["date"].unique()), dtype="datetime64[ns]")
    return holiday_lookup, holiday_dates_sorted

def _build_campaign_lookup(campaign_path: str | os.PathLike) -> dict:
    camp_og_df = pd.read_excel(campaign_path).copy()
    camp_og_df.rename(columns={c: str(c).strip() for c in camp_og_df.columns}, inplace=True)
    
    if "Week " in camp_og_df.columns: camp_og_df.rename(columns={"Week ": "week"}, inplace=True)
    if "Week" in camp_og_df.columns: camp_og_df.rename(columns={"Week": "week"}, inplace=True)

    promo_cols = [c for c in camp_og_df.columns if str(c).startswith("promo_")]
    camp_og_df["campaign_strength"] = camp_og_df[promo_cols].sum(axis=1) if promo_cols else 0
    camp_og_df["promotion_active"] = (camp_og_df["campaign_strength"] > 0).astype(int)
    camp_og_df["campaign_regions_active"] = (camp_og_df[promo_cols] > 0).sum(axis=1) if promo_cols else 0

    out = {}
    for _, r in camp_og_df.iterrows():
        if pd.isna(r.get("year")) or pd.isna(r.get("week")): continue
        out[(int(r["year"]), int(r["week"]))] = {
            "campaign_strength": _safe_float(r.get("campaign_strength"), 0.0),
            "promotion_active": _safe_int(r.get("promotion_active"), 0),
            "campaign_regions_active": _safe_int(r.get("campaign_regions_active"), 0),
        }
    return out

def _build_events_lookup(events_path: str | os.PathLike) -> dict:
    recurring_og_df = pd.read_excel(events_path).copy()
    if recurring_og_df.shape[1] >= 2:
        recurring_og_df.columns = ["event_name", "date"] + list(recurring_og_df.columns[2:])
        recurring_og_df = recurring_og_df[["event_name", "date"]]

    recurring_og_df["event_name"] = recurring_og_df["event_name"].fillna("").astype(str).str.split("/")
    recurring_df = recurring_og_df.explode("event_name")
    recurring_df["event_name"] = recurring_df["event_name"].astype(str).str.strip().str.replace(" ", "_").str.lower()
    recurring_df.loc[recurring_df["event_name"] == "", "event_name"] = "no_event"
    recurring_df["date"] = pd.to_datetime(recurring_df["date"], errors="coerce").dt.normalize()
    recurring_df = recurring_df.dropna(subset=["date"]).drop_duplicates(subset=["date", "event_name"])

    events_pivot = recurring_df.pivot_table(index="date", columns="event_name", aggfunc="size", fill_value=0).astype(int)
    events_pivot.columns = [f"event_{c}" for c in events_pivot.columns]
    return events_pivot.to_dict(orient="index")


# =============================================================================
# MAIN FORECASTING FUNCTION
# =============================================================================

def predict_next_365_days(forecast_days: int = 365, openmeteo_days: int = 14, manual_growth_override: float = None):
    """
    Forecasts sales with Dynamic Trend Projection (no hardcoded totals) and Extreme Seasonality correction.
    Incorporates SMOOTHED Rolling Targets to prevent "weekend shift" errors.
    """
    
    # ---------- Load Models & Data ----------
    print("Loading models...")
    with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f: global_model = pickle.load(f)
    with open(MODELS_DIR / "family_models.pkl", "rb") as f: family_models = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f: model_features = pickle.load(f)

    # Load Performance (WAPE)
    global_wape = np.inf
    family_wape_map = {}
    try:
        with open(MODELS_DIR / "model_performance.pkl", "rb") as f: perf = pickle.load(f)
        global_wape = _safe_float(perf.get("global_model", {}).get("WAPE", np.inf), np.inf)
        family_wape_map = {k: _safe_float(v.get("WAPE", np.inf), np.inf) for k, v in perf.get("family_models_heldout", {}).items()}
    except: pass

    # Load Processed Data
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce").dt.normalize()
    processed_df = processed_df.dropna(subset=["date"]).copy()
    processed_df["month"] = processed_df["date"].dt.month
    processed_df["day"] = processed_df["date"].dt.day
    processed_df["dayofyear"] = processed_df["date"].dt.dayofyear
    
    last_known_date = processed_df["date"].max().normalize()
    tickets_df = processed_df[["ticket_name", "ticket_family"]].drop_duplicates().sort_values(["ticket_family", "ticket_name"])
    ticket_list = list(tickets_df.itertuples(index=False, name=None))
    families = sorted(processed_df["ticket_family"].dropna().unique().tolist())

    print(f"Forecasting from {last_known_date.date()} for {len(ticket_list)} tickets")

    # =========================================================================
    # 1. COMPUTE EXTREME MULTIPLIERS (Shape Injection)
    # =========================================================================
    extreme_multipliers = _compute_extreme_multipliers(processed_df)

    # =========================================================================
    # 2. CALIBRATION (Last 30 Days)
    # =========================================================================
    bt = processed_df.sort_values("date").copy()
    bt = bt[bt["date"] >= (bt["date"].max() - pd.Timedelta(days=30))]
    
    family_calibration = {f: 1.0 for f in families}
    
    for fam in families:
        fam_bt = bt[bt["ticket_family"] == fam]
        if fam_bt.empty: continue
        
        # Build features for backtest
        X_fam = fam_bt.reindex(columns=model_features, fill_value=0).copy()
        for c in X_fam.columns:
            if X_fam[c].dtype == "object": X_fam[c] = pd.to_numeric(X_fam[c], errors="coerce").fillna(0)
            
        chosen_model, _ = _choose_model(fam, global_model, family_models, global_wape, family_wape_map)
        fam_pred = float(np.sum(np.maximum(0, chosen_model.predict(X_fam))))
        fam_actual = float(fam_bt["ticket_num"].sum())
        
        if fam_pred > 0:
            family_calibration[fam] = float(np.clip(fam_actual / fam_pred, 0.85, 1.15))

    # =========================================================================
    # 3. SMOOTHED SEASONALITY & DYNAMIC TREND PROJECTION
    # =========================================================================
    
    # --- A. Create SMOOTHED Rolling Targets (Fixes "Weekend Shift" Problem) ---
    daily_total = processed_df.groupby("date")["ticket_num"].sum().reset_index(name="total")
    daily_total = daily_total.sort_values("date")
    daily_total["doy"] = daily_total["date"].dt.dayofyear

    # 1. Calculate raw mean per day-of-year
    doy_means = daily_total.groupby("doy")["total"].mean().reset_index()
    # 2. Apply 7-day rolling smoothing (centers the volume, ignores specific weekday spikes)
    doy_means["smooth_total"] = doy_means["total"].rolling(window=7, center=True, min_periods=1).mean()
    # 3. Map for lookup
    raw_target_doy_map = doy_means.set_index("doy")["smooth_total"].to_dict()

    # Repeat for Families
    fam_daily = processed_df.groupby(["date", "ticket_family"])["ticket_num"].sum().reset_index(name="fam_total")
    fam_daily["doy"] = fam_daily["date"].dt.dayofyear
    
    # We need a map of (Family, DOY) -> Smoothed Volume
    raw_target_family_map = {}
    for fam in families:
        f_df = fam_daily[fam_daily["ticket_family"] == fam].copy()
        # Group by DOY to get mean across years
        f_means = f_df.groupby("doy")["fam_total"].mean().reset_index()
        # Rolling smooth
        f_means["smooth_total"] = f_means["fam_total"].rolling(window=7, center=True, min_periods=1).mean()
        # Store in dict
        for _, r in f_means.iterrows():
            raw_target_family_map[(fam, int(r["doy"]))] = float(r["smooth_total"])

    # --- B. Calculate Trend Ratio ---
    recent_actuals = bt["ticket_num"].sum()
    
    # Get historical volume for the same DOYs as the backtest
    recent_historical_baseline = sum(raw_target_doy_map.get(d.dayofyear, 0) for d in bt["date"].unique())
    if recent_historical_baseline == 0: recent_historical_baseline = 1.0
    
    trend_ratio = recent_actuals / recent_historical_baseline
    
    if manual_growth_override:
        trend_ratio = manual_growth_override
    else:
        # Safety clamp
        trend_ratio = np.clip(trend_ratio, 0.70, 1.30)
        
    hist_annual_sum = sum(raw_target_doy_map.values())
    projected_annual_sales = hist_annual_sum * trend_ratio

    print(f"\n=== DYNAMIC TREND PROJECTION ===")
    print(f"Historical Annual Baseline: {int(hist_annual_sum):,}")
    print(f"Calculated Trend Ratio:     {trend_ratio:.3f}")
    print(f"Projected Annual Target:    {int(projected_annual_sales):,}")

    # --- C. Scale the Maps ---
    target_doy_map = {k: v * trend_ratio for k, v in raw_target_doy_map.items()}
    target_family_map = {k: v * trend_ratio for k, v in raw_target_family_map.items()}

    # ---------- Prepare Feature Lookups ----------
    # History Features
    history_feature_cols = [c for c in model_features if "lag" in c or "rolling" in c or "sales" in c or "available" in c]
    hist_by_ticket = {}
    for tname, _ in ticket_list:
        tdf = processed_df.loc[processed_df["ticket_name"] == tname, ["date"] + history_feature_cols]
        hist_by_ticket[tname] = tdf.set_index("date").sort_index() if not tdf.empty else pd.DataFrame()
        
    # Ticket Meta
    meta_cols = ["ticket_name", "ticket_family", "groupID", "is_actie_ticket", "is_abonnement_ticket", "is_full_price", 
                 "is_accommodation_ticket", "is_group_ticket", "is_joint_promotion"]
    ticket_meta = processed_df[[c for c in meta_cols if c in processed_df.columns]].drop_duplicates("ticket_name").set_index("ticket_name")

    # External Data
    holiday_lookup, holiday_dates_sorted = _build_holiday_features(HOLIDAY_DATA_PATH)
    camp_lookup = _build_campaign_lookup(CAMPAIGN_DATA_PATH)
    events_lookup = _build_events_lookup(RECURRING_EVENTS_PATH)
    
    # Weather
    weather_future = pd.DataFrame()
    if OPENMETEO_AVAILABLE:
        try:
            weather_future = get_openmeteo_for_future(days=int(openmeteo_days))
            weather_future["date"] = pd.to_datetime(weather_future["date"]).dt.normalize()
        except: pass
    
    hist_weather = processed_df.groupby(["month", "day"])[["temperature", "rain_morning", "rain_afternoon", "precip_morning", "precip_afternoon"]].mean()
    hist_weather_key = hist_weather.to_dict(orient="index")

    # =========================================================================
    # FORECAST LOOP
    # =========================================================================
    print(f"\nSTARTING 365-DAY FORECAST Loop...")
    all_rows = []

    for step in range(forecast_days):
        today = pd.Timestamp.now().normalize()
        current_date = today + timedelta(days=step)

        year, month, day = current_date.year, current_date.month, current_date.day
        weekday, week = current_date.weekday(), current_date.isocalendar().week
        doy = current_date.dayofyear
        is_weekend = int(weekday >= 5)

        # Weather
        w_row = hist_weather_key.get((month, day), {"temperature": 10.0, "rain_morning": 0, "rain_afternoon": 0, "precip_morning": 0, "precip_afternoon": 0})
        if not weather_future.empty:
            match = weather_future.loc[weather_future["date"] == current_date]
            if not match.empty: w_row = match.iloc[0].to_dict()
            
        temperature = _safe_float(w_row.get("temperature", 10.0))
        rain = _safe_float(w_row.get("rain_morning", 0)) + _safe_float(w_row.get("rain_afternoon", 0))
        
        # Holiday/Events
        days_until_hol, days_since_hol = _compute_holiday_proximity(current_date, holiday_dates_sorted)
        hol_feats = holiday_lookup.get(current_date, {})
        camp_feats = camp_lookup.get((year, week), {})
        evt_feats = events_lookup.get(current_date, {})

        # Predict
        per_ticket = []
        per_family_tot = {fam: 0.0 for fam in families}
        
        for ticket_name, ticket_family in ticket_list:
            meta = ticket_meta.loc[ticket_name] if ticket_name in ticket_meta.index else None
            
            # Feature Construction
            yoy_date = (current_date - timedelta(days=365)).normalize()
            yoy_vals = hist_by_ticket.get(ticket_name, pd.DataFrame()).asof(yoy_date)
            if isinstance(yoy_vals, pd.Series): yoy_vals = yoy_vals.to_dict()
            else: yoy_vals = {}

            # Construct Feature Dict
            feat = {
                "year": year, "month": month, "day": day, "week": week, "weekday": weekday,
                "day_of_year": doy, "is_weekend": is_weekend,
                "temperature": temperature,
                "days_until_holiday": days_until_hol, "days_since_holiday": days_since_hol,
                "holiday_intensity": _safe_float(hol_feats.get("holiday_intensity", 0)),
                "campaign_strength": _safe_float(camp_feats.get("campaign_strength", 0)),
                **yoy_vals, # Lags
                **hol_feats, # Holiday dummies
                **evt_feats, # Event dummies
                f"ticket_{ticket_name}": 1, f"family_{ticket_family}": 1
            }
            if meta is not None:
                for c in meta.index: 
                    if c not in ["ticket_name", "ticket_family"]: feat[c] = meta[c]
            
            # Predict
            X = pd.DataFrame([feat]).reindex(columns=model_features, fill_value=0)
            chosen_model, model_used = _choose_model(ticket_family, global_model, family_models, global_wape, family_wape_map)
            
            pred_raw = float(chosen_model.predict(X)[0])
            pred_raw = max(0.0, pred_raw)
            
            # Apply Calibration
            pred = pred_raw * family_calibration.get(ticket_family, 1.0)
            
            per_ticket.append((ticket_name, ticket_family, pred, model_used))
            per_family_tot[ticket_family] += pred

        # =========================================================================
        # 4. ADAPTIVE SCALING (Using Smoothed Maps)
        # =========================================================================
        scaled = []
        
        # Family Scaling
        family_scaled_tot = {}
        for fam in families:
            fam_pred = per_family_tot[fam]
            
            # Use SMOOTHED target map
            fam_target = float(target_family_map.get((fam, doy), fam_pred))
            
            if fam_pred <= 0:
                family_scaled_tot[fam] = fam_pred
                continue
                
            fam_wape = float(family_wape_map.get(fam, 25.0))
            lo, hi = _get_adaptive_scaling_bounds(current_date, fam_wape)
            
            ratio = np.clip(fam_target / fam_pred, lo, hi)
            family_scaled_tot[fam] = fam_pred * ratio
            
        # Apply Scaling & Shape Injection
        for tname, tfam, pred, mused in per_ticket:
            fam_pred = per_family_tot[tfam]
            fam_sc = family_scaled_tot[tfam]
            
            # Family Ratio
            val = pred * (fam_sc / fam_pred) if fam_pred > 0 else pred
            
            # SHAPE INJECTION (Christmas/Low Season)
            val = _apply_extreme_shape_injection(val, current_date, extreme_multipliers)
            
            scaled.append((tname, tfam, val, mused))

        # Global Scaling (Using Smoothed Map)
        day_pred = sum(x[2] for x in scaled)
        # Use SMOOTHED target map
        day_target = float(target_doy_map.get(doy, day_pred))
        
        if day_pred > 0:
            lo, hi = _get_adaptive_scaling_bounds(current_date, 10.0)
            ratio = np.clip(day_target / day_pred, lo, hi)
            scaled = [(t, f, v * ratio, m) for t, f, v, m in scaled]

        # Save
        for tname, tfam, val, mused in scaled:
            all_rows.append({
                "date": current_date, "ticket_name": tname, "ticket_family": tfam,
                "predicted_sales": int(round(val)), "model_used": mused,
                "year": year, "month": month, "day": day
            })
            
        if step % 30 == 0:
            print(f"Day {step+1}: {int(sum(x[2] for x in scaled)):,} tickets")

    print(f"Predictions start: {today.date()} → {current_date.date()}")


    # ---------- Export ----------
    forecast_df = pd.DataFrame(all_rows)
    total_sales = forecast_df["predicted_sales"].sum()
    
    print("\n" + "="*50)
    print(f"FINAL FORECAST STATS")
    print(f"Total Predicted: {int(total_sales):,} (vs Proj. Target: {int(projected_annual_sales):,})(average per year 2022-2025 = 924180)")
    
    # Christmas Check
    xmas = forecast_df[(forecast_df.date.dt.month == 12) & (forecast_df.date.dt.day.between(18, 24))]
    print(f"Xmas Avg (Dec 18-24): {int(xmas.groupby('date')['predicted_sales'].sum().mean()):,}")

    out_path = PREDICTIONS_DIR / f"forecast_365days_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}")
    
    return forecast_df

if __name__ == "__main__":
    predict_next_365_days()