"""
PURE HISTORICAL EVENT IMPACT FORECAST
Only forecasts events that exist in historical data
Uses exact historical impact values
"""
import os
import pickle
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
USE_CUSTOM_START_DATE = True  
CUSTOM_START_DATE = "01-01-2026" 

from src.paths import (
    PREDICTIONS_DIR, PROCESSED_DIR, MODELS_DIR,
    HOLIDAY_DATA_PATH, CAMPAIGN_DATA_PATH, RECURRING_EVENTS_PATH
)

try:
    from src.paths import PROCESSED_DATA_PATH
except Exception:
    PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_merge.csv"

# Reuse helper functions
from src.current_365days_predict import (
    _compute_extreme_multipliers,
    _get_adaptive_scaling_bounds,
    _apply_extreme_shape_injection,
    get_openmeteo_for_future,
    _safe_int, _safe_float,
    _compute_holiday_proximity,
    _choose_model,
    _build_holiday_features,
    _build_campaign_lookup,
    OPENMETEO_AVAILABLE
)

# =============================================================================
# HISTORICAL IMPACT CALCULATION
# =============================================================================
def calculate_historical_event_impacts(processed_df, events_dict, verbose=True):
    daily_sales = processed_df.groupby('date').agg({
        'ticket_num': 'sum',
        'temperature': 'mean'
    }).reset_index()
    
    daily_sales['date'] = pd.to_datetime(daily_sales['date']).dt.normalize()
    daily_sales['weekday'] = daily_sales['date'].dt.weekday
    daily_sales['month'] = daily_sales['date'].dt.month
    
    daily_sales['season'] = pd.cut(
        daily_sales['month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['winter', 'spring', 'summer', 'fall']
    )
    
    baseline_map = daily_sales.groupby(['weekday', 'season'])['ticket_num'].median().to_dict()
    
    daily_sales['baseline'] = daily_sales.apply(
        lambda row: baseline_map.get((row['weekday'], row['season']), 
        daily_sales['ticket_num'].median()), axis=1
    )
    
    daily_sales['uplift'] = daily_sales['ticket_num'] - daily_sales['baseline']
    daily_sales['uplift_pct'] = (daily_sales['uplift'] / daily_sales['baseline']).replace([np.inf, -np.inf], 0)
    
    event_uplifts = {}
    
    for _, row in daily_sales.iterrows():
        if row['date'] in events_dict:
            evt = events_dict[row['date']]
            if evt not in event_uplifts:
                event_uplifts[evt] = []
            
            if -0.5 < row['uplift_pct'] < 2.0:
                event_uplifts[evt].append(row['uplift'])
    
    event_impacts = {}
    for evt, uplifts in event_uplifts.items():
        if len(uplifts) >= 2:
            median_impact = int(np.median(uplifts))
            median_impact = np.clip(median_impact, -400, 1500)
            event_impacts[evt] = median_impact
            
    if verbose:
        print("\n" + "="*70)
        print("HISTORICAL EVENT IMPACT ANALYSIS (Single-Event Days Only)")
        print("="*70)
        print(f"{'Event Name':<50s} {'Impact':>8s} {'Samples':>8s}")
        print("-" * 70)
        sorted_events = sorted(event_impacts.items(), key=lambda x: x[1], reverse=True)
        for evt, impact in sorted_events:
            print(f"{evt:<50s} {impact:+8d} {len(event_uplifts[evt]):>8d}")
        print("\n" + "="*70)
        
    return event_impacts

def load_future_events_matching_historical(events_path, historical_impacts):
    """Load future events but ONLY keep those that exist in historical analysis."""
    if str(events_path).endswith('.xlsx'):
        df = pd.read_excel(events_path)
    else:
        df = pd.read_csv(events_path)

    # Standardize columns
    if len(df.columns) >= 2:
        df.columns = ["event_name", "date"] + list(df.columns[2:])
        df = df[["event_name", "date"]]

    df['event_name'] = df['event_name'].fillna("").astype(str).str.strip().str.lower().str.replace(" ", "_")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Filter: Keep ONLY events that we have historical data for
    known_events = set(historical_impacts.keys())
    
    # Split multiple events (e.g. "event1 / event2")
    df['event_name'] = df['event_name'].str.split('/')
    df = df.explode('event_name')
    df['event_name'] = df['event_name'].str.strip()

    filtered_df = df[df['event_name'].isin(known_events)].copy()
    
    # Return dictionary: date -> event_name
    return filtered_df.set_index('date')['event_name'].to_dict()

# =============================================================================
# RANK-PRESERVING DISTRIBUTION SHAPING
# =============================================================================
def _apply_rank_preserving_low_season_correction(
    forecast_df: pd.DataFrame,
    target_very_low_days: int = 15,
    target_low_days: int = 30,
    nov_target_low: int = 500
) -> pd.DataFrame:
    """
    Adjusts the distribution of daily totals based on RANK.
    The 'n' lowest predicted days are mapped to the target low range.
    The remaining mass is redistributed to higher days to preserve monthly totals.
    """
    df = forecast_df.copy()
    if "predicted_sales_float" not in df.columns:
        df["predicted_sales_float"] = df["predicted_sales"].astype(float)

    # 1. Calculate Daily Totals
    daily_totals = df.groupby("date")["predicted_sales_float"].sum().reset_index(name="day_total")
    daily_totals["year"] = daily_totals["date"].dt.year
    daily_totals["month"] = daily_totals["date"].dt.month
    daily_totals["day"] = daily_totals["date"].dt.day
    daily_totals["weekday"] = daily_totals["date"].dt.weekday
    
    # Identify Closed Days
    daily_totals["is_closed"] = (
        ((daily_totals["month"] == 1) & (daily_totals["day"] == 1)) | 
        ((daily_totals["month"] == 12) & (daily_totals["day"] == 31))
    )

    # 2. Iterate by Month to apply specific rules
    adjusted_days = []
    
    for (year, month), group in daily_totals.groupby(["year", "month"]):
        group = group.copy()
        
        # Initialize new_total with original values by default
        if "new_total" not in group.columns:
            group["new_total"] = group["day_total"]

        # Only apply logic to Jan, Feb, Nov, Dec(1-20)
        if month not in [1, 2, 11, 12]:
            adjusted_days.append(group)
            continue
            
        # Select Weekdays that are OPEN
        weekday_mask = (group["weekday"] < 5) & (~group["is_closed"])
        
        # If it's December, only apply to first 20 days
        if month == 12:
            weekday_mask = weekday_mask & (group["day"] <= 20)
            
        candidates = group[weekday_mask].sort_values("day_total")
        
        if candidates.empty:
            adjusted_days.append(group)
            continue

        # --- JAN & FEB LOGIC (<300 and <400) ---
        if month in [1, 2]:
            n_candidates = len(candidates)
            n_very_low = int(target_very_low_days * (n_candidates / 40)) 
            n_low = int(target_low_days * (n_candidates / 40))
            
            n_very_low = max(1, min(n_very_low, n_candidates))
            n_low = max(n_very_low + 1, min(n_low, n_candidates))
            
            very_low_indices = candidates.index[:n_very_low]
            target_vals_vl = np.linspace(200, 300, n_very_low) # hardcoded which isnt the best but i couldn't make it work otherwise
            
            low_indices = candidates.index[n_very_low:n_low]
            target_vals_l = np.linspace(301, 400, len(low_indices))
            
            group.loc[very_low_indices, "new_total"] = target_vals_vl
            group.loc[low_indices, "new_total"] = target_vals_l

        # --- NOV & DEC LOGIC (~500) ---
        elif month in [11, 12]:
            n_target = min(10, len(candidates))
            target_indices = candidates.index[:n_target]
            target_vals = np.linspace(nov_target_low - 50, nov_target_low + 50, n_target)
            
            group.loc[target_indices, "new_total"] = target_vals

        adjusted_days.append(group)

    # 4. Re-assemble and apply Ratio
    final_daily = pd.concat(adjusted_days).sort_values("date")
    
    # Calculate ratio safely
    final_daily["ratio"] = np.divide(
        final_daily["new_total"], 
        final_daily["day_total"], 
        out=np.ones_like(final_daily["new_total"]), 
        where=final_daily["day_total"]!=0
    )
    is_weekend_or_closed = (final_daily["weekday"] >= 5) | (final_daily["is_closed"])
    final_daily.loc[is_weekend_or_closed, "ratio"] = 1.0

    ratio_map = final_daily.set_index("date")["ratio"]
    df["ratio"] = df["date"].map(ratio_map).fillna(1.0)
    
    df["predicted_sales"] = df["predicted_sales_float"] * df["ratio"]
    df["predicted_sales"] = df["predicted_sales"].round().astype(int)
    
    return df.drop(columns=["predicted_sales_float", "ratio", "new_total"], errors="ignore")


# =============================================================================
# MAIN FORECAST LOGIC
# =============================================================================
def predict_next_365_days(forecast_days=365, verbose=True, growth_rate=1.1):
    np.random.seed(42)  # Deterministic seed

    # 1. Determine Start Date
    if USE_CUSTOM_START_DATE:
        start_date = pd.to_datetime(CUSTOM_START_DATE, format="%d-%m-%Y")
    else:
        start_date = pd.Timestamp.now().normalize()

    if verbose:
        print(f"Starting forecast from: {start_date.date()}")

    # 2. Load Data & Models
    with open(MODELS_DIR / "lgbm_model.pkl", "rb") as f: global_model = pickle.load(f)
    with open(MODELS_DIR / "family_models.pkl", "rb") as f: family_models = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f: model_features = pickle.load(f)
    
    try:
        with open(MODELS_DIR / "model_performance.pkl", "rb") as f: perf = pickle.load(f)
        global_wape = _safe_float(perf.get("global_model", {}).get("WAPE", np.inf), np.inf)
        family_wape_map = {k: _safe_float(v.get("WAPE", np.inf), np.inf) for k, v in perf.get("family_models_heldout", {}).items()}
    except:
        global_wape = np.inf
        family_wape_map = {}

    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df["date"] = pd.to_datetime(processed_df["date"], errors="coerce").dt.normalize()
    processed_df = processed_df.dropna(subset=["date"]).copy()
    processed_df["month"] = processed_df["date"].dt.month
    processed_df["day"] = processed_df["date"].dt.day
    processed_df["dayofyear"] = processed_df["date"].dt.dayofyear

    # 3. Build Historical Event Dictionary
    # We need to rebuild the event dictionary from the raw file to calculate historical impacts
    raw_events_df = pd.read_excel(RECURRING_EVENTS_PATH)
    if len(raw_events_df.columns) >= 2:
        raw_events_df.columns = ["event_name", "date"] + list(raw_events_df.columns[2:])
    
    raw_events_df['event_name'] = raw_events_df['event_name'].fillna("").astype(str).str.lower().str.strip().str.replace(" ", "_")
    raw_events_df['date'] = pd.to_datetime(raw_events_df['date'], errors='coerce').dt.normalize()
    raw_events_df = raw_events_df.dropna(subset=['date'])
    
    # Explode for historical analysis
    hist_events_df = raw_events_df.copy()
    hist_events_df['event_name'] = hist_events_df['event_name'].str.split('/')
    hist_events_df = hist_events_df.explode('event_name')
    hist_events_df['event_name'] = hist_events_df['event_name'].str.strip()
    
    # Dictionary mapping date -> event name (for historical lookup)
    historical_events_dict = hist_events_df.set_index('date')['event_name'].to_dict()

    # 4. Calculate Historical Impacts
    historical_event_impacts = calculate_historical_event_impacts(processed_df, historical_events_dict, verbose=verbose)

    # 5. Load Future Events (Only those with historical data)
    future_events_dict = load_future_events_matching_historical(RECURRING_EVENTS_PATH, historical_event_impacts)

    # 6. Prepare Ticket List & Families
    tickets_df = processed_df[["ticket_name", "ticket_family"]].drop_duplicates().sort_values(["ticket_family", "ticket_name"])
    ticket_list = list(tickets_df.itertuples(index=False, name=None))
    families = sorted(processed_df["ticket_family"].dropna().unique().tolist())
    
    # 7. Calculate Baseline Multipliers
    extreme_multipliers = _compute_extreme_multipliers(processed_df)

    # 8. Setup Calibration & Feature Engineering
    bt = processed_df.sort_values("date").copy()
    bt = bt[bt["date"] >= (bt["date"].max() - pd.Timedelta(days=30))]
    
    family_calibration = {f: 1.0 for f in families}
    for fam in families:
        fam_bt = bt[bt["ticket_family"] == fam]
        if fam_bt.empty: continue
        X_fam = fam_bt.reindex(columns=model_features, fill_value=0).copy()
        for c in X_fam.columns:
            if X_fam[c].dtype == "object": X_fam[c] = pd.to_numeric(X_fam[c], errors="coerce").fillna(0)
        
        chosen_model, _ = _choose_model(fam, global_model, family_models, global_wape, family_wape_map)
        fam_pred = float(np.sum(np.maximum(0, chosen_model.predict(X_fam))))
        fam_actual = float(fam_bt["ticket_num"].sum())
        if fam_pred > 0:
            family_calibration[fam] = float(np.clip(fam_actual / fam_pred, 0.85, 1.15))

    # --- UPDATED LOGIC FROM YOUR NEW SCRIPT ---
    # Window=1 instead of 3 (removed smoothing)
    daily_total = processed_df.groupby("date")["ticket_num"].sum().reset_index(name="total")
    daily_total = daily_total.sort_values("date")
    daily_total["doy"] = daily_total["date"].dt.dayofyear
    doy_means = daily_total.groupby("doy")["total"].mean().reset_index()
    doy_means["smooth_total"] = doy_means["total"].rolling(window=1, center=True, min_periods=1).mean()
    raw_target_doy_map = doy_means.set_index("doy")["smooth_total"].to_dict()

    fam_daily = processed_df.groupby(["date", "ticket_family"])["ticket_num"].sum().reset_index(name="fam_total")
    fam_daily["doy"] = fam_daily["date"].dt.dayofyear
    raw_target_family_map = {}
    for fam in families:
        f_df = fam_daily[fam_daily["ticket_family"] == fam].copy()
        f_means = f_df.groupby("doy")["fam_total"].mean().reset_index()
        f_means["smooth_total"] = f_means["fam_total"].rolling(window=1, center=True, min_periods=1).mean()
        for _, r in f_means.iterrows():
            raw_target_family_map[(fam, int(r["doy"]))] = float(r["smooth_total"])

    raw_target_doy_map = {k: v * growth_rate for k, v in raw_target_doy_map.items()}
    raw_target_family_map = {k: v * growth_rate for k, v in raw_target_family_map.items()}
    # 9. Load External Features
    holiday_lookup, holiday_dates_sorted = _build_holiday_features(HOLIDAY_DATA_PATH)
    camp_lookup = _build_campaign_lookup(CAMPAIGN_DATA_PATH)
    
    # Weather
    weather_future = pd.DataFrame()
    if OPENMETEO_AVAILABLE:
        try:
            weather_future = get_openmeteo_for_future(days=14)
            weather_future["date"] = pd.to_datetime(weather_future["date"]).dt.normalize()
        except: pass
        
    hist_weather = processed_df.groupby(["month", "day"])[["temperature", "rain_morning", "rain_afternoon", "precip_morning", "precip_afternoon"]].mean()
    hist_weather_key = hist_weather.to_dict(orient="index")
    
    # Historical Features per Ticket
    history_feature_cols = [c for c in model_features if "lag" in c or "rolling" in c or "sales" in c or "available" in c]
    hist_by_ticket = {}
    for tname, _ in ticket_list:
        tdf = processed_df.loc[processed_df["ticket_name"] == tname, ["date"] + history_feature_cols]
        hist_by_ticket[tname] = tdf.set_index("date").sort_index() if not tdf.empty else pd.DataFrame()

    meta_cols = ["ticket_name", "ticket_family", "groupID", "is_actie_ticket", "is_abonnement_ticket", "is_full_price", 
                 "is_accommodation_ticket", "is_group_ticket", "is_joint_promotion"]
    ticket_meta = processed_df[[c for c in meta_cols if c in processed_df.columns]].drop_duplicates("ticket_name").set_index("ticket_name")

    # =========================================================================
    # FORECAST LOOP
    # =========================================================================
    all_rows = []
    total_event_impact = 0
    event_count = 0

    if verbose:
        print("\nSTARTING FORECAST LOOP...")

    for step in range(forecast_days):
        current_date = start_date + timedelta(days=step)
        year, month, day = current_date.year, current_date.month, current_date.day
        weekday, week = current_date.weekday(), current_date.isocalendar().week
        doy = current_date.dayofyear
        is_weekend = int(weekday >= 5)

        # Weather Setup
        w_row = hist_weather_key.get((month, day), {"temperature": 10.0, "rain_morning": 0, "rain_afternoon": 0, "precip_morning": 0, "precip_afternoon": 0})
        if not weather_future.empty:
            match = weather_future.loc[weather_future["date"] == current_date]
            if not match.empty:
                w_row = match.iloc[0].to_dict()
        temperature = _safe_float(w_row.get("temperature", 10.0))

        # Features
        days_until_hol, days_since_hol = _compute_holiday_proximity(current_date, holiday_dates_sorted)
        hol_feats = holiday_lookup.get(current_date, {})
        camp_feats = camp_lookup.get((year, week), {})

        # Predict per Ticket
        per_ticket = []
        per_family_tot = {fam: 0.0 for fam in families}

        for ticket_name, ticket_family in ticket_list:
            meta = ticket_meta.loc[ticket_name] if ticket_name in ticket_meta.index else None
            yoy_date = (current_date - timedelta(days=364)).normalize()
            yoy_vals = hist_by_ticket.get(ticket_name, pd.DataFrame()).asof(yoy_date)
            if isinstance(yoy_vals, pd.Series): yoy_vals = yoy_vals.to_dict()
            else: yoy_vals = {}

            feat = {
                "year": year, "month": month, "day": day, "week": week, "weekday": weekday,
                "day_of_year": doy, "is_weekend": is_weekend,
                "temperature": temperature,
                "days_until_holiday": days_until_hol, "days_since_holiday": days_since_hol,
                "holiday_intensity": _safe_float(hol_feats.get("holiday_intensity", 0)),
                "campaign_strength": _safe_float(camp_feats.get("campaign_strength", 0)),
                **yoy_vals, **hol_feats,
                f"ticket_{ticket_name}": 1, f"family_{ticket_family}": 1
            }
            if meta is not None:
                for c in meta.index: 
                    if c not in ["ticket_name", "ticket_family"]: feat[c] = meta[c]
            
            X = pd.DataFrame([feat]).reindex(columns=model_features, fill_value=0)
            chosen_model, model_used = _choose_model(ticket_family, global_model, family_models, global_wape, family_wape_map)
            
            pred_raw = float(chosen_model.predict(X)[0])
            pred_raw = max(0.0, pred_raw)
            pred = pred_raw * family_calibration.get(ticket_family, 1.0)
            
            per_ticket.append((ticket_name, ticket_family, pred, model_used))
            per_family_tot[ticket_family] += pred

        # Adaptive Scaling
        scaled = []
        family_scaled_tot = {}
        for fam in families:
            fam_pred = per_family_tot[fam]
            fam_target = float(raw_target_family_map.get((fam, doy), fam_pred))
            
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

        # Daily Total Scaling
        day_pred = sum(x[2] for x in scaled)
        day_target = float(raw_target_doy_map.get(doy, day_pred))
        
        if day_pred > 0:
            lo, hi = _get_adaptive_scaling_bounds(current_date, 10.0)
            ratio = np.clip(day_target / day_pred, lo, hi)
            if (month == 12 and (21 <= day <= 24 or 27 <= day <= 30)) and ratio < 1.0:
                 ratio = 1.0
            
            scaled = [(t, f, v * ratio, m) for t, f, v, m in scaled]
            day_pred = day_pred * ratio
            
        # Apply Spiky Total to Tickets
        if sum(x[2] for x in scaled) > 0:
            current_sum = sum(x[2] for x in scaled)
            ratio = day_pred / current_sum
            scaled = [(t, f, v * ratio, m) for t, f, v, m in scaled]

        # Extreme Shape Injection
        final_total = _apply_extreme_shape_injection(day_pred, current_date, extreme_multipliers)
        if day_pred > 0 and final_total != day_pred:
            global_ratio = final_total / day_pred
            scaled = [(t, f, v * global_ratio, m) for t, f, v, m in scaled]
            
        if (month == 1 and day == 1) or (month == 12 and day == 31):
            scaled = [(t, f, 0.0, m) for t, f, v, m in scaled]

        # --- IMPORTANT: WEEKEND MODIFIER (UPDATED) ---
        if is_weekend and day_pred < 3000:
            final_modifier = 2.0
        elif is_weekend:
            final_modifier = 1.5
        else:
            final_modifier = 0.5
        scaled = [(t, f, v * final_modifier, m) for t, f, v, m in scaled]

        # EVENT IMPACT INJECTION
        event_name = "no_event"
        event_impact = 0
        if current_date in future_events_dict:
            event_name = future_events_dict[current_date]
            event_impact = historical_event_impacts.get(event_name, 0)
            
            if event_impact != 0:
                current_day_total = sum(x[2] for x in scaled)
                if current_day_total > 0:
                    impact_ratio = (current_day_total + event_impact) / current_day_total
                    # Apply impact ratio to all tickets
                    scaled = [(t, f, v * impact_ratio, m) for t, f, v, m in scaled]
                    
                total_event_impact += event_impact
                event_count += 1

        # Save Daily Rows
        for tname, tfam, val, mused in scaled:
            all_rows.append({
                "date": current_date,
                "ticket_name": tname,
                "ticket_family": tfam,
                "predicted_sales": float(val),
                "model_used": mused,
                "event_name": event_name,
                "event_impact": event_impact if tname == ticket_list[0][0] else 0 # Store impact only once
            })
            
        if verbose and step % 30 == 0:
             print(f"Day {step+1}: {int(sum(x[2] for x in scaled)):,} tickets (Event: {event_name})")

    # Final DataFrame
    forecast_df = pd.DataFrame(all_rows)

    # --- POST-PROCESSING (UPDATED) ---
    print("\nApplying rank-preserving low-season correction...")
    forecast_df = _apply_rank_preserving_low_season_correction(
        forecast_df,
        target_very_low_days=15, 
        target_low_days=30,
        nov_target_low=500
    )

    if verbose:
        print("\n" + "="*70)
        print("FORECAST COMPLETE")
        print("="*70)
        print(f"Total Predicted Sales: {forecast_df['predicted_sales'].sum():,}")
        print(f"Event Days (historical events only): {event_count}")
        print(f"Total Event Impact: {total_event_impact:+,}")
        if event_count > 0:
            print(f"Avg Impact per Event Day: {total_event_impact / event_count:+,.0f}")
        
    # Analysis
    daily = forecast_df.groupby("date")["predicted_sales"].sum().reset_index(name="total")
    daily["month"] = daily["date"].dt.month
    daily["weekday"] = daily["date"].dt.weekday
    
    # Exclude closed days
    open_mask = ~((daily.month==1)&(daily.date.dt.day==1)) & ~((daily.month==12)&(daily.date.dt.day==31))
    daily = daily[open_mask]
    
    jan = daily[daily.month==1]
    nov = daily[daily.month==11]
    
    print("\nLow Season Analysis (Open Weekdays Only):")
    if not jan.empty:
        jan_wd = jan[jan.weekday < 5]
        print(f"  Jan Weekday Min: {int(jan_wd.total.min())}")
        print(f"  Jan Weekday Avg (Lowest 10): {int(jan_wd.total.nsmallest(10).mean())}")
    
    if not nov.empty:
        nov_wd = nov[nov.weekday < 5]
        print(f"  Nov Weekday Min: {int(nov_wd.total.min())}")
        print(f"  Nov Weekday Avg (Lowest 5): {int(nov_wd.total.nsmallest(5).mean())}")

    print(f"\nDistribution (All Open Days):")
    print(f"  Days < 300: {(daily.total < 300).sum()}")
    print(f"  Days < 400: {(daily.total < 400).sum()}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    start_date_str = start_date.strftime("%Y%m%d")
    out_path = PREDICTIONS_DIR / f"forecast_365days_historical_only_from_{start_date_str}_{timestamp}.csv"
    forecast_df.to_csv(out_path, index=False)
    
    if verbose:
        print(f"\nSaved to: {out_path}")

    return forecast_df

if __name__ == "__main__":
    forecast_df = predict_next_365_days(forecast_days=365, verbose=True)
