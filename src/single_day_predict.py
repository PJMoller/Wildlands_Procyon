import os
import pickle
import warnings
from datetime import datetime, timedelta
import sys
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

# ... (Imports and Path Configuration same as before) ...
from src.paths import (
    PREDICTIONS_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    HOLIDAY_DATA_PATH,
    CAMPAIGN_DATA_PATH,
    RECURRING_EVENTS_PATH,
)

try:
    from src.paths import PROCESSED_DATA_PATH
except Exception:
    PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_merge.csv"

# ... (Optional Dependencies same as before) ...
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    OPENMETEO_AVAILABLE = True
except Exception as e:
    OPENMETEO_AVAILABLE = False

# ... (Helper Functions same as before) ...
def _safe_int(x, default=0):
    try: return int(x) if not pd.isna(x) else default
    except: return default

def _safe_float(x, default=0.0):
    try: return float(x) if not pd.isna(x) else default
    except: return default

def _compute_extreme_multipliers(processed_df):
    # (Same implementation as previous versions)
    baseline_months = [3, 4, 5, 9, 10]
    baseline_data = processed_df[processed_df['date'].dt.month.isin(baseline_months)]
    baseline_avg = processed_df.groupby('date')['ticket_num'].sum().median() if baseline_data.empty else baseline_data.groupby('date')['ticket_num'].sum().median()
    baseline_avg = baseline_avg or 1.0
    
    recent_df = processed_df[processed_df['date'].dt.year >= 2022]
    xmas_data = recent_df[(recent_df['date'].dt.month == 12) & (recent_df['date'].dt.day.between(27, 30))]
    xmas_avg = xmas_data.groupby('date')['ticket_num'].sum().mean() if not xmas_data.empty else baseline_avg * 3.0
    
    low_data = processed_df[processed_df['date'].dt.month.isin([1, 2, 11])]
    low_avg = low_data.groupby('date')['ticket_num'].sum().mean() if not low_data.empty else baseline_avg * 0.3
    
    return {'baseline': float(baseline_avg), 'xmas_multiplier': float(xmas_avg / baseline_avg), 'low_season_multiplier': float(low_avg / baseline_avg)}

def _get_adaptive_scaling_bounds(current_date, fam_wape):
    # (Same implementation as previous versions)
    month, day = current_date.month, current_date.day
    is_trusted = fam_wape < 15.0
    if month == 12 and 25 <= day <= 31: return (0.8, 8.0)
    if month in [1, 2, 11] or (month == 12 and day < 21): return (0.0, 2.2)
    return (0.70, 1.50) if is_trusted else (0.50, 1.30)

def _apply_extreme_shape_injection(pred, current_date, multipliers):
    # (Same implementation as previous versions)
    month, weekday, day = current_date.month, current_date.weekday(), current_date.day
    if month == 12 and 21 <= day <= 24:
        baseline = multipliers['baseline']
        ramp_progress = (day - 21) / 3.0
        ramp_mult = 1.8 + (ramp_progress * 1.8)
        ideal_val = baseline * ramp_mult
        if pred < ideal_val:
            weight = 0.9
            injected = (pred * (1 - weight)) + (ideal_val * weight)
            return min(injected, 7000.0)
    elif month == 12 and 26 <= day <= 30:
        progress = (day - 26) / 3.0
        progress = min(max(progress, 0.0), 1.0)
        target_mult = multipliers['xmas_multiplier']
        baseline = multipliers['baseline']
        ideal_val = baseline * target_mult
        if pred < ideal_val:
            weight = 0.1 + (progress * 0.1)
            injected = (pred * (1 - weight)) + (ideal_val * weight)
            return min(injected, ideal_val * 1.0)
    elif month in [7, 8] and weekday < 5:
        return pred * 2
    return pred

def _compute_holiday_proximity(date_ts, holiday_dates_sorted):
    # (Same implementation as previous versions)
    if holiday_dates_sorted.size == 0: return 90, 90
    d = np.datetime64(date_ts.normalize())
    pos = np.searchsorted(holiday_dates_sorted, d, side="left")
    next_h = holiday_dates_sorted[pos] if pos < holiday_dates_sorted.size else d + np.timedelta64(90, "D")
    prev_h = holiday_dates_sorted[pos - 1] if pos > 0 else d - np.timedelta64(90, "D")
    return int((next_h - d) / np.timedelta64(1, "D")), int((d - prev_h) / np.timedelta64(1, "D"))

def _choose_model(ticket_family, global_model, family_models, global_wape, family_wape_map):
    # (Same implementation as previous versions)
    fam_model = family_models.get(ticket_family)
    if fam_model is None: return global_model, "global"
    fam_wape = family_wape_map.get(ticket_family, np.inf)
    if np.isfinite(global_wape):
        if fam_wape < global_wape: return fam_model, "family"
        return global_model, "global"
    if fam_wape < 50.0: return fam_model, "family"
    return global_model, "global"

def _build_holiday_features(holiday_path):
    # (Same implementation)
    try:
        df = pd.read_excel(holiday_path).copy()
        for c in df.columns:
            if str(c).lower().strip() == "week": df.drop(columns=[c], inplace=True); break
        cols = list(df.columns)
        region_cols, date_col = cols[:5], cols[5]
        long_df = df.melt(id_vars=[date_col], value_vars=region_cols, var_name="region", value_name="holiday")
        long_df.rename(columns={date_col: "date"}, inplace=True)
        long_df["date"] = pd.to_datetime(long_df["date"], errors="coerce")
        long_df.dropna(subset=["date"], inplace=True)
        long_df["holiday"] = long_df["holiday"].fillna("None").astype(str).str.strip().replace("", "None")
        long_df["region_holiday"] = long_df["region"].astype(str) + "_" + long_df["holiday"].astype(str)
        encoded = pd.get_dummies(long_df.set_index("date")["region_holiday"], prefix="", prefix_sep="_")
        holiday_daily = encoded.groupby(level=0).sum().reset_index()
        intensity = long_df.groupby("date")["holiday"].apply(lambda x: x.dropna().nunique()).reset_index(name="holiday_intensity")
        holiday_daily = holiday_daily.merge(intensity, on="date", how="left")
        holiday_daily["date"] = pd.to_datetime(holiday_daily["date"]).dt.normalize()
        lookup = holiday_daily.set_index("date").to_dict(orient="index")
        holiday_dates_sorted = np.array(sorted(holiday_daily["date"].unique()), dtype="datetime64[ns]")
        return lookup, holiday_dates_sorted
    except: return {}, np.array([])

def _build_campaign_lookup(campaign_path):
    # (Same implementation)
    try:
        df = pd.read_excel(campaign_path).copy()
        df.rename(columns={c: str(c).strip() for c in df.columns}, inplace=True)
        for w in ["Week ", "Week"]:
            if w in df.columns: df.rename(columns={w: "week"}, inplace=True)
        promo_cols = [c for c in df.columns if str(c).startswith("promo_")]
        df["campaign_strength"] = df[promo_cols].sum(axis=1) if promo_cols else 0
        df["promotion_active"] = (df["campaign_strength"] > 0).astype(int)
        df["campaign_regions_active"] = (df[promo_cols] > 0).sum(axis=1) if promo_cols else 0
        out = {}
        for _, r in df.iterrows():
            if pd.isna(r.get("year")) or pd.isna(r.get("week")): continue
            out[(int(r["year"]), int(r["week"]))] = {
                "campaign_strength": _safe_float(r.get("campaign_strength"), 0.0),
                "promotion_active": _safe_int(r.get("promotion_active"), 0),
                "campaign_regions_active": _safe_int(r.get("campaign_regions_active"), 0)
            }
        return out
    except: return {}

def _build_events_lookup(events_path):
    # (Same implementation)
    try:
        df = pd.read_excel(events_path).copy()
        if df.shape[1] >= 2:
            df.columns = ["event_name", "date"] + list(df.columns[2:])
            df = df[["event_name", "date"]]
        df["event_name"] = df["event_name"].fillna("").astype(str).str.split("/")
        df = df.explode("event_name")
        df["event_name"] = df["event_name"].astype(str).str.strip().str.replace(" ", "_").str.lower()
        df.loc[df["event_name"] == "", "event_name"] = "no_event"
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df.dropna(subset=["date"], inplace=True)
        df.drop_duplicates(subset=["date", "event_name"], inplace=True)
        pivot = df.pivot_table(index="date", columns="event_name", aggfunc="size", fill_value=0).astype(int)
        pivot.columns = [f"event_{c}" for c in pivot.columns]
        return pivot.to_dict(orient="index")
    except: return {}


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_single_day(
    input_date: str,
    rain_morning: float = 0.0,
    rain_afternoon: float = 0.0,
    precip_morning: float = 0.0,
    precip_afternoon: float = 0.0,
    temperature: float = 10.0,
    holiday_name: str = "",
    event_name: str = "",
    manual_growth_override: float = 1.1
):
    np.random.seed(42)
    
    # Load models
    with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f: global_model = pickle.load(f)
    with open(MODELS_DIR / "family_models.pkl", "rb") as f: family_models = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f: model_features = pickle.load(f)
    
    try:
        with open(MODELS_DIR / "model_performance.pkl", "rb") as f: perf = pickle.load(f)
        global_wape = _safe_float(perf.get("global_model", {}).get("WAPE", np.inf), np.inf)
        family_wape_map = {k: _safe_float(v.get("WAPE", np.inf), np.inf) for k, v in perf.get("family_models_heldout", {}).items()}
    except:
        global_wape, family_wape_map = np.inf, {}

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).copy()
    
    tickets_df = df[["ticket_name", "ticket_family"]].drop_duplicates().sort_values(["ticket_family", "ticket_name"])
    ticket_list = list(tickets_df.itertuples(index=False, name=None))
    families = sorted(df["ticket_family"].dropna().unique().tolist())
    
    current_date = pd.to_datetime(input_date).normalize()
    year, month, day = current_date.year, current_date.month, current_date.day
    weekday, week = current_date.weekday(), current_date.isocalendar().week
    doy = current_date.dayofyear
    is_weekend = int(weekday >= 5)

    hard_override = bool(manual_growth_override is not None or holiday_name or event_name)
    extreme_multipliers = _compute_extreme_multipliers(df)

    # Calculate Targets (DOY & Family)
    daily_total = df.groupby("date")["ticket_num"].sum().reset_index(name="total").sort_values("date")
    daily_total["doy"] = daily_total["date"].dt.dayofyear
    doy_means = daily_total.groupby("doy")["total"].mean().reset_index()
    doy_means["smooth_total"] = doy_means["total"].rolling(window=1, center=True, min_periods=1).mean()
    raw_target_doy_map = doy_means.set_index("doy")["smooth_total"].to_dict()
    
    fam_daily = df.groupby(["date", "ticket_family"])["ticket_num"].sum().reset_index(name="fam_total")
    fam_daily["doy"] = fam_daily["date"].dt.dayofyear
    raw_target_family_map = {}
    for fam in families:
        f_df = fam_daily[fam_daily["ticket_family"] == fam].copy()
        f_means = f_df.groupby("doy")["fam_total"].mean().reset_index()
        f_means["smooth_total"] = f_means["fam_total"].rolling(window=1, center=True, min_periods=1).mean()
        for _, r in f_means.iterrows():
            raw_target_family_map[(fam, int(r["doy"]))] = float(r["smooth_total"])
    
    # ------------------------------------------------------------------
    # KEY FIX 1: ADJUST TARGETS DOWNWARDS IF WEATHER IS BAD
    # ------------------------------------------------------------------
    # Instead of fighting the model prediction, we lower the target itself.
    rain_total = rain_morning + rain_afternoon
    precip_total = precip_morning + precip_afternoon
    weather_severity = rain_total + precip_total
    
    weather_target_penalty = 1.0
    if weather_severity > 5.0:
        weather_target_penalty = 0.75  # Lower target by 25% for heavy rain
    elif weather_severity > 2.0:
        weather_target_penalty = 0.90  # Lower target by 10% for light rain
    
    # Apply manual growth AND weather penalty to targets
    trend_ratio = (manual_growth_override if manual_growth_override is not None else 1.0) * weather_target_penalty
    
    target_doy_map = {k: v * trend_ratio for k, v in raw_target_doy_map.items()}
    target_family_map = {k: v * trend_ratio for k, v in raw_target_family_map.items()}
    
    # Build Features
    try: real_holiday_lookup, holiday_dates_sorted = _build_holiday_features(HOLIDAY_DATA_PATH)
    except: real_holiday_lookup, holiday_dates_sorted = {}, np.array([])
    days_until_hol, days_since_hol = _compute_holiday_proximity(current_date, holiday_dates_sorted)
    
    hol_feats = {}
    if holiday_name:
        hol_normalized = holiday_name.lower().strip().replace(" ", "_")
        hol_feats[f"holiday_{hol_normalized}"] = 1
        hol_feats["is_public_holiday"] = 1
        hol_feats["holiday_intensity"] = 1.0
    elif current_date in real_holiday_lookup:
        hol_feats = real_holiday_lookup[current_date].copy()
    else:
        hol_feats["holiday_intensity"] = 0.0
        hol_feats["is_public_holiday"] = 0
        
    camp_lookup = _build_campaign_lookup(CAMPAIGN_DATA_PATH)
    camp_feats = camp_lookup.get((year, week), {})
    
    evt_feats = {}
    if event_name:
        for evt in event_name.split(","):
            if evt.strip(): evt_feats[f"event_{evt.lower().strip().replace(' ', '_')}"] = 1

    # Calibration
    bt = df.sort_values("date").copy()
    bt = bt[bt["date"] >= (bt["date"].max() - pd.Timedelta(days=30))]
    family_calibration = {f: 1.0 for f in families} # Simplified
    
    # Prediction Loop
    per_ticket = []
    per_family_tot = {fam: 0.0 for fam in families}
    
    # Pre-fetch history for speed
    meta_cols = ["ticket_name", "ticket_family", "groupID", "is_actie_ticket", "is_abonnement_ticket", "is_full_price",
                 "is_accommodation_ticket", "is_group_ticket", "is_joint_promotion"]
    ticket_meta = df[[c for c in meta_cols if c in df.columns]].drop_duplicates("ticket_name").set_index("ticket_name")
    hist_cols = [c for c in model_features if "lag" in c or "rolling" in c or "sales" in c or "available" in c]
    hist_by_ticket = {tname: df.loc[df["ticket_name"] == tname, ["date"] + hist_cols].set_index("date").sort_index()
                      for tname, _ in ticket_list}

    for tname, tfam in ticket_list:
        meta = ticket_meta.loc[tname] if tname in ticket_meta.index else None
        yoy_date = (current_date - timedelta(days=364)).normalize()
        yoy_vals = hist_by_ticket.get(tname, pd.DataFrame()).asof(yoy_date)
        if isinstance(yoy_vals, pd.Series): yoy_vals = yoy_vals.to_dict()
        else: yoy_vals = {}
        
        feat = {
            "year": year, "month": month, "day": day, "week": week, "weekday": weekday, "day_of_year": doy, "is_weekend": is_weekend,
            "temperature": temperature, 
            # PASS WEATHER FEATURES
            "rain_morning": rain_morning,
            "rain_afternoon": rain_afternoon,
            "precip_morning": precip_morning,
            "precip_afternoon": precip_afternoon,
            "days_until_holiday": days_until_hol, "days_since_holiday": days_since_hol,
            "holiday_intensity": _safe_float(hol_feats.get("holiday_intensity", 0)),
            "campaign_strength": _safe_float(camp_feats.get("campaign_strength", 0)),
            **yoy_vals, **hol_feats, **evt_feats, **camp_feats,
            f"ticket_{tname}": 1, f"family_{tfam}": 1
        }
        if meta is not None:
            for c in meta.index:
                if c not in ["ticket_name", "ticket_family"]: feat[c] = meta[c]
        
        X = pd.DataFrame([feat]).reindex(columns=model_features, fill_value=0)
        chosen_model, _ = _choose_model(tfam, global_model, family_models, global_wape, family_wape_map)
        
        pred_raw = float(chosen_model.predict(X)[0])
        pred_raw = max(0.0, pred_raw)
        pred = pred_raw * family_calibration.get(tfam, 1.0)
        per_ticket.append((tname, tfam, pred, ""))
        per_family_tot[tfam] += pred

    # ------------------------------------------------------------------
    # KEY FIX 2: DYNAMIC SCALING (Don't let scaling undo weather)
    # ------------------------------------------------------------------
    # If weather is bad, we reduce the 'strength' of scaling.
    scaling_strength = 1.0
    if weather_severity > 5.0:
        scaling_strength = 0.3  # Very weak scaling - trust model more
    elif weather_severity > 2.0:
        scaling_strength = 0.6  # Moderate scaling
        
    scaled = []
    family_scaled_tot = {}
    for fam in families:
        fam_pred = per_family_tot[fam]
        fam_target = float(target_family_map.get((fam, doy), fam_pred)) # Already lowered by weather_target_penalty
        
        if fam_pred > 0:
            fam_wape = float(family_wape_map.get(fam, 25.0))
            lo, hi = _get_adaptive_scaling_bounds(current_date, fam_wape)
            
            # Calculate standard ratio
            raw_ratio = np.clip(fam_target / fam_pred, lo, hi)
            
            # Apply scaling strength: move ratio closer to 1.0
            ratio = 1.0 + (raw_ratio - 1.0) * scaling_strength
            
            family_scaled_tot[fam] = fam_pred * ratio
        else:
            family_scaled_tot[fam] = fam_pred
            
    for tname, tfam, pred, _ in per_ticket:
        fam_pred = per_family_tot[tfam]
        fam_sc = family_scaled_tot[tfam]
        val = pred * (fam_sc / fam_pred) if fam_pred > 0 else pred
        scaled.append((tname, tfam, val, ""))

    # Global Scaling
    day_pred = sum(x[2] for x in scaled)
    day_target = float(target_doy_map.get(doy, day_pred)) # Already lowered by weather_target_penalty
    
    if day_pred > 0:
        lo, hi = _get_adaptive_scaling_bounds(current_date, 10.0)
        
        raw_ratio = np.clip(day_target / day_pred, lo, hi)
        
        # Apply scaling strength again
        ratio = 1.0 + (raw_ratio - 1.0) * scaling_strength
        
        # Holiday protection (only upscaling allowed for Xmas peak)
        is_holiday_window = (month == 12 and (21 <= day <= 24 or 27 <= day <= 30))
        if is_holiday_window and ratio < 1.0: ratio = 1.0
        
        scaled = [(t, f, v * ratio, m) for t, f, v, m in scaled]
        day_pred *= ratio

    # Extreme shape injection
    if not hard_override:
        final_total = _apply_extreme_shape_injection(day_pred, current_date, extreme_multipliers)
        if day_pred > 0 and final_total != day_pred:
            global_ratio = final_total / day_pred
            scaled = [(t, f, v * global_ratio, m) for t, f, v, m in scaled]

    # Weather Penalties (Final safeguard)
    # These apply ON TOP of everything else to guarantee a drop
    """
    if rain_total > 5.0: 
        drop = min(0.60, (rain_total - 5.0) * 0.02) # Slightly stronger slope
        scaled = [(t, f, v * (1 - drop), m) for t, f, v, m in scaled]
    
    if precip_total > 5.0: 
        drop = min(0.50, (precip_total - 5.0) * 0.02)
        scaled = [(t, f, v * (1 - drop), m) for t, f, v, m in scaled]
    
    if temperature < 5.0: 
        drop = min(0.40, (5.0 - temperature) * 0.02)
        scaled = [(t, f, v * (1 - drop), m) for t, f, v, m in scaled]
    elif temperature > 30.0: 
        drop = min(0.30, (temperature - 30.0) * 0.03)
        scaled = [(t, f, v * (1 - drop), m) for t, f, v, m in scaled]
"""
    # Closed Days & Weekend Modifiers
    if (month == 1 and day == 1) or (month == 12 and day == 31):
        scaled = [(t, f, 0.0, m) for t, f, v, m in scaled]

    if not hard_override:
        day_pred_current = sum(x[2] for x in scaled)
        if is_weekend and day_pred_current < 5000: modifier = 2.0
        elif is_weekend: modifier = 1.5
        else: modifier = 0.60
        scaled = [(t, f, v * modifier, m) for t, f, v, m in scaled]
        
    total_sales = sum(int(round(v)) for _, _, v, _ in scaled)
    return total_sales

if __name__ == "__main__":
    # Robust Test
    print("--- Test: Good Weather ---")
    res1 = predict_single_day("2026-07-15", temperature=22.0, rain_morning=0.0)
    print(f"Result: {res1:,}")
    
    print("\n--- Test: Bad Weather (10mm Rain) ---")
    res2 = predict_single_day("2026-07-15", temperature=15.0, rain_morning=5.0, rain_afternoon=5.0)
    print(f"Result: {res2:,}")
    
    if res1 > 0:
        drop_pct = (1 - res2/res1) * 100
        print(f"\nWeather Drop Impact: {drop_pct:.1f}%")
