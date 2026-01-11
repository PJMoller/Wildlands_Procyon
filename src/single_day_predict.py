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
try:
    import requests_cache
    from retry_requests import retry
except Exception as e:
    print(f"Optional dependency missing: {e}")


# =============================================================================
# HELPER FUNCTIONS: EXTREME SEASONALITY & DYNAMIC TRENDS
# =============================================================================

def _compute_extreme_multipliers(processed_df: pd.DataFrame) -> dict:
    """
    Calculates the intensity of peaks/troughs relative to a 'normal' baseline.
    Used to inject shape into the forecast.
    """
    # 1. Establish a 'Normal' Baseline (Spring/Autumn)
    # Use MEDIAN to avoid outliers spiking the baseline
    baseline_months = [3, 4, 5, 9, 10]
    baseline_data = processed_df[processed_df['date'].dt.month.isin(baseline_months)]
    
    if baseline_data.empty:
        baseline_avg = processed_df.groupby('date')['ticket_num'].sum().median()
    else:
        baseline_avg = baseline_data.groupby('date')['ticket_num'].sum().median()
        
    if baseline_avg == 0: baseline_avg = 1.0

    # 2. Christmas Peak Intensity (Dec 27-30) - The REAL Peak
    recent_df = processed_df[processed_df['date'].dt.year >= 2022]
    
    xmas_data = recent_df[
        (recent_df['date'].dt.month == 12) & 
        (recent_df['date'].dt.day.between(27, 30))
    ]
    
    if not xmas_data.empty:
        xmas_avg = xmas_data[xmas_data['ticket_num'] > 0].groupby('date')['ticket_num'].sum().mean()
    else:
        xmas_avg = baseline_avg * 3.0 # Fallback if no recent data

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

    # --- CHRISTMAS PEAK (Dec 25 - Dec 31) ---
    # Allow massive spikes during the true holiday week
    if month == 12 and 25 <= day <= 31:
        return (0.8, 8.0) 
        
    # --- LOW SEASON (Jan, Feb, Nov) ---
    elif month in [1, 2, 11]:
        return (0.0, 1.5) # Allow dropping to 0%
        
    # --- NORMAL SEASON ---
    else:
        if is_trusted:
            return (0.70, 1.50) # Loosened slightly to catch summer peaks
        else:
            return (0.50, 1.80)


def _apply_extreme_shape_injection(pred: float, current_date: datetime, multipliers: dict) -> float:
    """
    Enforces specific curve shapes (exponential ramps) that tree models often smooth out.
    """
    month = current_date.month
    day = current_date.day

    if month == 12 and 21 <= day <= 24:
        # We want to hit ~3,000 to ~5,000.
        # Baseline is ~1,744.
        # So we need a multiplier of roughly 2.0x to 2.8x.
        
        baseline = multipliers['baseline']
        
        # Create a "mini-ramp" for these 4 days
        # Day 21 (0.0) -> Day 24 (1.0)
        ramp_progress = (day - 21) / 3.0
        
        # Target starts at 1.5x baseline (~2600) and ends at 3.0x baseline (~5200)
        ramp_mult = 1.5 + (ramp_progress * 1.5) 
        
        ideal_val = baseline * ramp_mult
        
        if pred < ideal_val:
            # Use moderate weight (0.7) to pull prediction up
            weight = 0.9
            injected = (pred * (1 - weight)) + (ideal_val * weight)
            return min(injected, 6000.0) # Safety Cap for Pre-Xmas
        
    # Ramp (Dec 27 - 30)
    elif month == 12 and 26 <= day <= 30:
        # Progress 0.0 to 1.0 (Day 27=0.0, Day 30=1.0)
        progress = (day - 26) / 3.0
        progress = min(max(progress, 0.0), 1.0) # Clamp safely
        
        target_mult = multipliers['xmas_multiplier']
        baseline = multipliers['baseline']
        ideal_val = (baseline * target_mult) * 1.3
        
        # Only inject if prediction is significantly lower
        if pred < ideal_val:
            weight = 0.5 + (progress * 0.4) # Max weight 0.9
            injected = (pred * (1 - weight)) + (ideal_val * weight)
            
            # SAFETY CAP: Never exceed 110% of the calculated ideal
            return min(injected, ideal_val * 1.1)

    return pred


# =============================================================================
# STANDARD UTILS
# =============================================================================


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

def predict_single_day(
    input_date: str,  # Format: "YYYY-MM-DD"
    rain_morning: float = 0.0,
    rain_afternoon: float = 0.0,
    precip_morning: float = 0.0,
    precip_afternoon: float = 0.0,
    temperature: float = 10.0,
    holiday_name: str = "",  # e.g., "christmas", "easter", "kingsday"
    event_name: str = "",
    manual_growth_override: float = None
):
    """
    Forecasts sales for a SINGLE specific day.
    
    Includes HEURISTIC LOGIC for manual overrides:
    - Disables all historical smoothing/calibration.
    - Applies manual penalties for Rain, Precipitation, and Temperature 
      to force the prediction to react realistically to extreme weather.
    """
    
    # ---------- Load Models & Data ----------
    with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f: 
        global_model = pickle.load(f)
    with open(MODELS_DIR / "family_models.pkl", "rb") as f: 
        family_models = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f: 
        model_features = pickle.load(f)

    try:
        with open(MODELS_DIR / "model_performance.pkl", "rb") as f: 
            perf = pickle.load(f)
        global_wape = _safe_float(perf.get("global_model", {}).get("WAPE", np.inf), np.inf)
        family_wape_map = {k: _safe_float(v.get("WAPE", np.inf), np.inf) 
                          for k, v in perf.get("family_models_heldout", {}).items()}
    except: 
        global_wape = np.inf
        family_wape_map = {}

    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce").dt.normalize()
    processed_df = processed_df.dropna(subset=["date"]).copy()
    processed_df["month"] = processed_df["date"].dt.month
    processed_df["day"] = processed_df["date"].dt.day
    processed_df["dayofyear"] = processed_df["date"].dt.dayofyear
    
    tickets_df = processed_df[["ticket_name", "ticket_family"]].drop_duplicates().sort_values(["ticket_family", "ticket_name"])
    ticket_list = list(tickets_df.itertuples(index=False, name=None))
    families = sorted(processed_df["ticket_family"].dropna().unique().tolist())

    # Parse input date
    current_date = pd.to_datetime(input_date).normalize()
    year, month, day = current_date.year, current_date.month, current_date.day
    weekday, week = current_date.weekday(), current_date.isocalendar().week
    doy = current_date.dayofyear
    is_weekend = int(weekday >= 5)

    # Calculate derived weather values
    rain = rain_morning + rain_afternoon
    precipitation = precip_morning + precip_afternoon

    # 1. CHECK FOR OVERRIDES
    # If any manual input is provided, we switch to "Simulation Mode"
    has_override = bool(manual_growth_override or holiday_name or event_name or rain > 0.0 or precipitation > 0.0)

    # Compute extreme multipliers (shape injection logic)
    extreme_multipliers = _compute_extreme_multipliers(processed_df)

    # ---------- Calibration Logic ----------
    bt = processed_df.sort_values("date").copy()
    bt = bt[bt["date"] >= (bt["date"].max() - pd.Timedelta(days=30))]
    
    family_calibration = {f: 1.0 for f in families}
    for fam in families:
        fam_bt = bt[bt["ticket_family"] == fam]
        if fam_bt.empty: 
            continue
        X_fam = fam_bt.reindex(columns=model_features, fill_value=0).copy()
        for c in X_fam.columns:
            if X_fam[c].dtype == "object": 
                X_fam[c] = pd.to_numeric(X_fam[c], errors="coerce").fillna(0)
        chosen_model, _ = _choose_model(fam, global_model, family_models, global_wape, family_wape_map)
        fam_pred = float(np.sum(np.maximum(0, chosen_model.predict(X_fam))))
        fam_actual = float(fam_bt["ticket_num"].sum())
        if fam_pred > 0:
            family_calibration[fam] = float(np.clip(fam_actual / fam_pred, 0.85, 1.15))

    # ---------- Build Rolling Targets ----------
    daily_total = processed_df.groupby("date")["ticket_num"].sum().reset_index(name="total")
    daily_total = daily_total.sort_values("date")
    daily_total["doy"] = daily_total["date"].dt.dayofyear
    doy_means = daily_total.groupby("doy")["total"].mean().reset_index()
    doy_means["smooth_total"] = doy_means["total"].rolling(window=7, center=True, min_periods=1).mean()
    raw_target_doy_map = doy_means.set_index("doy")["smooth_total"].to_dict()

    fam_daily = processed_df.groupby(["date", "ticket_family"])["ticket_num"].sum().reset_index(name="fam_total")
    fam_daily["doy"] = fam_daily["date"].dt.dayofyear
    raw_target_family_map = {}
    for fam in families:
        f_df = fam_daily[fam_daily["ticket_family"] == fam].copy()
        f_means = f_df.groupby("doy")["fam_total"].mean().reset_index()
        f_means["smooth_total"] = f_means["fam_total"].rolling(window=7, center=True, min_periods=1).mean()
        for _, r in f_means.iterrows():
            raw_target_family_map[(fam, int(r["doy"]))] = float(r["smooth_total"])

    # ---------- Trend Calculation ----------
    recent_actuals = bt["ticket_num"].sum()
    recent_historical_baseline = sum(raw_target_doy_map.get(d.dayofyear, 0) for d in bt["date"].unique())
    if recent_historical_baseline == 0: 
        recent_historical_baseline = 1.0
    
    trend_ratio = recent_actuals / recent_historical_baseline
    if manual_growth_override:
        trend_ratio = manual_growth_override
    else:
        trend_ratio = np.clip(trend_ratio, 0.70, 1.30)

    target_doy_map = {k: v * trend_ratio for k, v in raw_target_doy_map.items()}
    target_family_map = {k: v * trend_ratio for k, v in raw_target_family_map.items()}

    # ---------- Prepare History Features ----------
    history_feature_cols = [c for c in model_features if "lag" in c or "rolling" in c or "sales" in c or "available" in c]
    hist_by_ticket = {}
    for tname, _ in ticket_list:
        tdf = processed_df.loc[processed_df["ticket_name"] == tname, ["date"] + history_feature_cols]
        hist_by_ticket[tname] = tdf.set_index("date").sort_index() if not tdf.empty else pd.DataFrame()
        
    meta_cols = ["ticket_name", "ticket_family", "groupID", "is_actie_ticket", "is_abonnement_ticket", 
                 "is_full_price", "is_accommodation_ticket", "is_group_ticket", "is_joint_promotion"]
    ticket_meta = processed_df[[c for c in meta_cols if c in processed_df.columns]].drop_duplicates("ticket_name").set_index("ticket_name")

    # ---------- HOLIDAY LOGIC ----------
    try:
        real_holiday_lookup, holiday_dates_sorted = _build_holiday_features(HOLIDAY_DATA_PATH)
    except:
        real_holiday_lookup, holiday_dates_sorted = {}, []
    
    days_until_hol, days_since_hol = _compute_holiday_proximity(current_date, holiday_dates_sorted)
    
    hol_feats = {}
    if holiday_name:
        hol_normalized = holiday_name.lower().strip().replace(" ", "_")
        hol_feats[f"holiday_{hol_normalized}"] = 1
        hol_feats["is_public_holiday"] = 1
        hol_feats["holiday_intensity"] = 1.0
        # Common aliases
        for alias in ["christmas", "xmas", "kerst"]:
            if alias in hol_normalized: hol_feats["holiday_christmas"] = 1
        for alias in ["easter", "pasen"]:
            if alias in hol_normalized: hol_feats["holiday_easter"] = 1
        
        days_until_hol = 0
        days_since_hol = 0
    else:
        if current_date in real_holiday_lookup:
            hol_feats = real_holiday_lookup[current_date].copy()
        else:
            hol_feats["holiday_intensity"] = 0.0
            hol_feats["is_public_holiday"] = 0

    # ---------- CAMPAIGN & EVENTS ----------
    camp_lookup = _build_campaign_lookup(CAMPAIGN_DATA_PATH)
    camp_feats = camp_lookup.get((year, week), {})
    
    evt_feats = {}
    if event_name:
        events = [e.strip() for e in event_name.split(",")]
        for evt in events:
            if evt:
                evt_normalized = evt.lower().strip().replace(" ", "_")
                evt_feats[f"event_{evt_normalized}"] = 1

    # ---------- PREDICTION LOOP ----------
    per_ticket = []
    per_family_tot = {fam: 0.0 for fam in families}
    
    for ticket_name, ticket_family in ticket_list:
        meta = ticket_meta.loc[ticket_name] if ticket_name in ticket_meta.index else None
        
        yoy_date = (current_date - timedelta(days=364)).normalize()
        yoy_vals = hist_by_ticket.get(ticket_name, pd.DataFrame()).asof(yoy_date)
        if isinstance(yoy_vals, pd.Series): 
            yoy_vals = yoy_vals.to_dict()
        else: 
            yoy_vals = {}

        feat = {
            "year": year, "month": month, "day": day, "week": week, "weekday": weekday,
            "day_of_year": doy, "is_weekend": is_weekend,
            "temperature": temperature,
            "days_until_holiday": days_until_hol, 
            "days_since_holiday": days_since_hol,
            "holiday_intensity": _safe_float(hol_feats.get("holiday_intensity", 0)),
            "campaign_strength": _safe_float(camp_feats.get("campaign_strength", 0)),
            **yoy_vals, **hol_feats, **evt_feats, **camp_feats,
            f"ticket_{ticket_name}": 1, 
            f"family_{ticket_family}": 1
        }
        
        if meta is not None:
            for c in meta.index: 
                if c not in ["ticket_name", "ticket_family"]: 
                    feat[c] = meta[c]
        
        X = pd.DataFrame([feat]).reindex(columns=model_features, fill_value=0)
        chosen_model, model_used = _choose_model(ticket_family, global_model, family_models, 
                                                 global_wape, family_wape_map)
        
        pred_raw = float(chosen_model.predict(X)[0])
        pred_raw = max(0.0, pred_raw)
        
        # --- 1: DISABLE CALIBRATION ON OVERRIDE ---
        if has_override:
            pred = pred_raw
        else:
            pred = pred_raw * family_calibration.get(ticket_family, 1.0)
        
        per_ticket.append((ticket_name, ticket_family, pred, model_used))
        per_family_tot[ticket_family] += pred

    # ---------- SCALING ----------
    scaled = []
    family_scaled_tot = {}
    
    for fam in families:
        fam_pred = per_family_tot[fam]
        
        # --- 2: DISABLE FAMILY SCALING ON OVERRIDE ---
        if has_override:
             family_scaled_tot[fam] = fam_pred
        else:
            fam_target = float(target_family_map.get((fam, doy), fam_pred))
            if fam_pred <= 0:
                family_scaled_tot[fam] = fam_pred
                continue
            fam_wape = float(family_wape_map.get(fam, 25.0))
            lo, hi = _get_adaptive_scaling_bounds(current_date, fam_wape)
            ratio = np.clip(fam_target / fam_pred, lo, hi)
            family_scaled_tot[fam] = fam_pred * ratio
        
    for tname, tfam, pred, mused in per_ticket:
        fam_pred = per_family_tot[tfam]
        fam_sc = family_scaled_tot[tfam]
        val = pred * (fam_sc / fam_pred) if fam_pred > 0 else pred
        scaled.append((tname, tfam, val, mused))

    # --- 3: DISABLE GLOBAL SCALING ON OVERRIDE ---
    day_pred = sum(x[2] for x in scaled)
    day_target = float(target_doy_map.get(doy, day_pred))
    
    if day_pred > 0:
        if has_override:
             ratio = 1.0
        else:
             lo, hi = _get_adaptive_scaling_bounds(current_date, 10.0)
             ratio = np.clip(day_target / day_pred, lo, hi)
             is_holiday_window = (month == 12 and (21 <= day <= 24 or 27 <= day <= 30))
             if is_holiday_window and ratio < 1.0:
                 ratio = 1.0
        
        scaled = [(t, f, v * ratio, m) for t, f, v, m in scaled]
        day_pred = day_pred * ratio

    # --- 4: DISABLE SHAPE INJECTION ON OVERRIDE ---
    if not has_override:
        final_total = _apply_extreme_shape_injection(day_pred, current_date, extreme_multipliers)
        if day_pred > 0 and final_total != day_pred:
            global_ratio = final_total / day_pred
            scaled = [(t, f, v * global_ratio, m) for t, f, v, m in scaled]


    # ---------- HEURISTIC PENALTIES (Force Physics) ----------
    # !NOT CORRECT UNDER WORK!
    if has_override:
        # 1. Rain Penalty (Start penalty at 5mm, max out at 50mm)
        rain_total = rain_morning + rain_afternoon
        if rain_total > 5.0:
            drop = min(0.60, (rain_total - 5.0) * 0.015) 
            scaled = [(t, f, v * (1.0 - drop), m) for t, f, v, m in scaled]

        # 2. Precipitation Penalty (Similar to rain)
        precip_total = precip_morning + precip_afternoon
        if precip_total > 5.0:
            drop = min(0.50, (precip_total - 5.0) * 0.015)
            scaled = [(t, f, v * (1.0 - drop), m) for t, f, v, m in scaled]
            
        # Cold Penalty: Below 5°C, sales drop. Below -5°C, sales drop hard.
        if temperature < 5.0:
            # 5C -> 0% drop, -5C -> 20% drop, -10C -> 30% drop
            drop = min(0.40, (5.0 - temperature) * 0.02)
            scaled = [(t, f, v * (1.0 - drop), m) for t, f, v, m in scaled]
            
        elif temperature > 30.0:
            drop = min(0.30, (temperature - 30.0) * 0.03)
            scaled = [(t, f, v * (1.0 - drop), m) for t, f, v, m in scaled]


    # ---------- Force Closed Days ----------
    if (month == 1 and day == 1) or (month == 12 and day == 31):
        scaled = [(t, f, 0.0, m) for t, f, v, m in scaled]

    # ---------- Weekend/Weekday Modifier ----------
    if not has_override:  # --- 5: DISABLE MODIFIER ON OVERRIDE ---
        if is_weekend:
            final_modifier = 1.2
        else:
            final_modifier = 0.9
        scaled = [(t, f, v * final_modifier, m) for t, f, v, m in scaled]
    
    # ---------- Final Sum ----------
    total_sales = sum(int(round(val)) for _, _, val, _ in scaled)
    
    return total_sales

