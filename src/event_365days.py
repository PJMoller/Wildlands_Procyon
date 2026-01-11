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

warnings.filterwarnings("ignore")

from paths import (
    PREDICTIONS_DIR, PROCESSED_DIR, MODELS_DIR,
    HOLIDAY_DATA_PATH, CAMPAIGN_DATA_PATH, RECURRING_EVENTS_PATH
)

try:
    from paths import PROCESSED_DATA_PATH
except Exception:
    PROCESSED_DATA_PATH = PROCESSED_DIR / "processed_merge.csv"

from current_365days_predict import (
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


# ============================================================================
# Historical impact calculation - single events only
# ============================================================================

def calculate_historical_event_impacts(processed_df, events_dict, verbose=True):
    """Calculate historical uplift for single-event days."""
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
                                     daily_sales['ticket_num'].median()),
        axis=1
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
        print("-"*70)
        
        sorted_events = sorted(event_impacts.items(), key=lambda x: x[1], reverse=True)
        for evt, impact in sorted_events:
            sample_size = len(event_uplifts[evt])
            print(f"{evt:<50s} {impact:+8d} {sample_size:>8d}")
        
        print("\n" + "="*70)
        print(f"Total events with data: {len(event_impacts)}")
    
    return event_impacts


def load_future_events_matching_historical(events_path, historical_impacts):
    """
    Load future events but ONLY keep those that exist in historical analysis.
    """
    recurring_og_df = pd.read_excel(events_path).copy()
    
    if recurring_og_df.shape[1] >= 2:
        recurring_og_df.columns = ["event_name", "date"] + list(recurring_og_df.columns[2:])
    recurring_og_df = recurring_og_df[["event_name", "date"]]
    
    recurring_og_df["date"] = pd.to_datetime(recurring_og_df["date"], errors="coerce").dt.normalize()
    recurring_og_df = recurring_og_df.dropna(subset=["date", "event_name"])
    
    recurring_og_df["event_name"] = (
        recurring_og_df["event_name"]
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )
    
    recurring_og_df["event_name"] = recurring_og_df["event_name"].str.replace(
        pat=r'^fc_emmen_.*', repl='soccer', regex=True
    )
    
    recurring_og_df = recurring_og_df[recurring_og_df["event_name"] != ""]
    recurring_og_df = recurring_og_df[recurring_og_df["event_name"] != "no_event"]
    recurring_og_df = recurring_og_df[recurring_og_df["event_name"] != "nan"]
    
    # Count events per date
    events_per_date = recurring_og_df.groupby("date").size()
    single_event_dates = events_per_date[events_per_date == 1].index
    
    # Keep ONLY single-event dates
    recurring_df = recurring_og_df[recurring_og_df["date"].isin(single_event_dates)].copy()
    
    # CRITICAL: Only keep events that exist in historical_impacts
    recurring_df = recurring_df[recurring_df["event_name"].isin(historical_impacts.keys())]
    
    print(f"\n✓ Future events loaded: {len(recurring_df)}")
    print(f"✓ Unique dates: {recurring_df['date'].nunique()}")
    print(f"✓ Events matching historical data: {recurring_df['event_name'].nunique()}")
    
    # Create dict: {date: event_name}
    events_dict = recurring_df.set_index("date")["event_name"].to_dict()
    
    return events_dict


# ============================================================================
# Helper functions
# ============================================================================

def _predict_ticket_batch(
    ticket_list, features_dict, evt_feats, include_events,
    model_features, ticket_meta, hist_by_ticket, family_models,
    global_model, global_wape, family_wape_map, family_calibration, current_date
):
    """Runs predictions for all tickets."""
    per_ticket = []
    
    for ticket_name, ticket_family in ticket_list:
        meta = ticket_meta.loc[ticket_name] if ticket_name in ticket_meta.index else None
        feat = features_dict.copy()
        
        if include_events:
            feat.update(evt_feats)
        else:
            for evt_col in [c for c in model_features if c.startswith('event_')]:
                feat[evt_col] = 0
        
        feat[f'ticket_{ticket_name}'] = 1
        feat[f'family_{ticket_family}'] = 1
        
        if meta is not None:
            for c in meta.index:
                if c not in ['ticket_name', 'ticket_family']:
                    feat[c] = meta[c]
        
        yoy_date = (current_date - timedelta(days=364)).normalize()
        yoy_vals = hist_by_ticket.get(ticket_name, pd.DataFrame()).asof(yoy_date)
        if isinstance(yoy_vals, pd.Series):
            feat.update(yoy_vals.to_dict())
        
        X = pd.DataFrame([feat]).reindex(columns=model_features, fill_value=0)
        for c in X.columns:
            if X[c].dtype == 'object':
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        
        chosen_model, model_used = _choose_model(
            ticket_family, global_model, family_models, global_wape, family_wape_map
        )
        
        pred_raw = float(chosen_model.predict(X)[0])
        pred = max(0.0, pred_raw) * family_calibration.get(ticket_family, 1.0)
        
        per_ticket.append((ticket_name, ticket_family, pred, model_used))
    
    return per_ticket


def _apply_scaling_pipeline(
    per_ticket, current_date, target_family_map, target_doy_map,
    family_wape_map, extreme_multipliers, families,
    apply_shape_injection=True, apply_weekend_modifier=True
):
    """Applies scaling transformations."""
    doy = current_date.dayofyear
    month = current_date.month
    day = current_date.day
    is_weekend = current_date.weekday() >= 5
    
    per_family_tot = {fam: sum(p[2] for p in per_ticket if p[1] == fam) for fam in families}
    family_scaled_tot = {}
    
    for fam in families:
        fam_pred = per_family_tot[fam]
        fam_target = float(target_family_map.get((fam, doy), fam_pred))
        
        if fam_pred <= 0:
            family_scaled_tot[fam] = fam_pred
            continue
        
        fam_wape = float(family_wape_map.get(fam, 25.0))
        lo, hi = _get_adaptive_scaling_bounds(current_date, fam_wape)
        ratio = np.clip(fam_target / fam_pred, lo, hi)
        family_scaled_tot[fam] = fam_pred * ratio
    
    scaled = []
    for tname, tfam, pred, mused in per_ticket:
        fam_pred = per_family_tot[tfam]
        fam_sc = family_scaled_tot[tfam]
        val = pred * (fam_sc / fam_pred) if fam_pred > 0 else pred
        scaled.append((tname, tfam, val, mused))
    
    day_pred = sum(x[2] for x in scaled)
    day_target = float(target_doy_map.get(doy, day_pred))
    
    if day_pred > 0:
        lo, hi = _get_adaptive_scaling_bounds(current_date, 10.0)
        ratio = np.clip(day_target / day_pred, lo, hi)
        
        is_holiday_window = (month == 12 and (21 <= day <= 24 or 27 <= day <= 30))
        if is_holiday_window and ratio < 1.0:
            ratio = 1.0
        
        scaled = [(t, f, v * ratio, m) for t, f, v, m in scaled]
        day_pred = day_pred * ratio
    
    if apply_shape_injection:
        final_total = _apply_extreme_shape_injection(day_pred, current_date, extreme_multipliers)
        if day_pred > 0 and final_total != day_pred:
            global_ratio = final_total / day_pred
            scaled = [(t, f, v * global_ratio, m) for t, f, v, m in scaled]
    
    if (month == 1 and day == 1) or (month == 12 and day == 31):
        scaled = [(t, f, 0.0, m) for t, f, v, m in scaled]
    
    if apply_weekend_modifier:
        final_modifier = 1.2 if is_weekend else 0.9
        scaled = [(t, f, v * final_modifier, m) for t, f, v, m in scaled]
    
    return scaled


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def predict_next_365_days(
    forecast_days=365,
    openmeteo_days=14,
    manual_growth_override=None,
    verbose=True
):
    """
    Forecast with ONLY historical events.
    Event impacts are exact copies of historical analysis.
    """
    
    if verbose:
        print("="*70)
        print("FORECAST WITH HISTORICAL EVENTS ONLY")
        print("="*70)
    
    # Load models
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
        family_wape_map = {
            k: _safe_float(v.get("WAPE", np.inf), np.inf)
            for k, v in perf.get("family_models_heldout", {}).items()
        }
    except:
        global_wape = np.inf
        family_wape_map = {}
    
    # Load data
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df["date"] = pd.to_datetime(processed_df["date"]).dt.normalize()
    processed_df = processed_df.dropna(subset=["date"])
    
    tickets_df = processed_df[["ticket_name", "ticket_family"]].drop_duplicates()
    ticket_list = list(tickets_df.itertuples(index=False, name=None))
    families = sorted(processed_df["ticket_family"].dropna().unique())
    
    if verbose:
        print(f"Tickets: {len(ticket_list)} | Families: {len(families)}")
    
    # Setup (same as before - keeping full prediction logic)
    extreme_multipliers = _compute_extreme_multipliers(processed_df)
    
    bt = processed_df[processed_df["date"] >= (processed_df["date"].max() - pd.Timedelta(days=30))]
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
    
    daily_total = processed_df.groupby("date")["ticket_num"].sum().reset_index(name="total")
    daily_total["doy"] = daily_total["date"].dt.dayofyear
    doy_means = daily_total.groupby("doy")["total"].mean().reset_index()
    doy_means["smooth_total"] = doy_means["total"].rolling(7, center=True, min_periods=1).mean()
    raw_target_doy_map = doy_means.set_index("doy")["smooth_total"].to_dict()
    
    fam_daily = processed_df.groupby(["date", "ticket_family"])["ticket_num"].sum().reset_index(name="fam_total")
    fam_daily["doy"] = fam_daily["date"].dt.dayofyear
    raw_target_family_map = {}
    for fam in families:
        f_df = fam_daily[fam_daily["ticket_family"] == fam]
        f_means = f_df.groupby("doy")["fam_total"].mean().reset_index()
        f_means["smooth_total"] = f_means["fam_total"].rolling(7, center=True, min_periods=1).mean()
        for _, r in f_means.iterrows():
            raw_target_family_map[(fam, int(r["doy"]))] = float(r["smooth_total"])
    
    recent_actuals = bt["ticket_num"].sum()
    recent_baseline = sum(raw_target_doy_map.get(d.dayofyear, 0) for d in bt["date"].unique())
    if recent_baseline == 0:
        recent_baseline = 1.0
    trend_ratio = manual_growth_override if manual_growth_override else np.clip(recent_actuals / recent_baseline, 0.70, 1.30)
    
    target_doy_map = {k: v * trend_ratio for k, v in raw_target_doy_map.items()}
    target_family_map = {k: v * trend_ratio for k, v in raw_target_family_map.items()}
    
    history_cols = [c for c in model_features if "lag" in c or "rolling" in c or "sales" in c or "available" in c]
    hist_by_ticket = {}
    for tname, _ in ticket_list:
        tdf = processed_df.loc[processed_df["ticket_name"] == tname, ["date"] + history_cols]
        hist_by_ticket[tname] = tdf.set_index("date").sort_index() if not tdf.empty else pd.DataFrame()
    
    meta_cols = ["ticket_name", "ticket_family", "groupID", "is_actie_ticket", "is_abonnement_ticket",
                 "is_full_price", "is_accommodation_ticket", "is_group_ticket", "is_joint_promotion"]
    ticket_meta = processed_df[[c for c in meta_cols if c in processed_df.columns]]
    ticket_meta = ticket_meta.drop_duplicates("ticket_name").set_index("ticket_name")
    
    holiday_lookup, holiday_dates_sorted = _build_holiday_features(HOLIDAY_DATA_PATH)
    camp_lookup = _build_campaign_lookup(CAMPAIGN_DATA_PATH)
    
    # STEP 1: Calculate historical impacts from past data
    # First load historical single events
    recurring_hist_df = pd.read_excel(RECURRING_EVENTS_PATH).copy()
    if recurring_hist_df.shape[1] >= 2:
        recurring_hist_df.columns = ["event_name", "date"] + list(recurring_hist_df.columns[2:])
    recurring_hist_df = recurring_hist_df[["event_name", "date"]]
    recurring_hist_df["date"] = pd.to_datetime(recurring_hist_df["date"], errors="coerce").dt.normalize()
    recurring_hist_df = recurring_hist_df.dropna(subset=["date", "event_name"])
    recurring_hist_df["event_name"] = (
        recurring_hist_df["event_name"].astype(str).str.strip().str.replace(" ", "_").str.lower()
    )
    recurring_hist_df["event_name"] = recurring_hist_df["event_name"].str.replace(
        pat=r'^fc_emmen_.*', repl='soccer', regex=True
    )
    recurring_hist_df = recurring_hist_df[recurring_hist_df["event_name"] != ""]
    recurring_hist_df = recurring_hist_df[recurring_hist_df["event_name"] != "no_event"]
    
    events_per_date = recurring_hist_df.groupby("date").size()
    single_event_dates = events_per_date[events_per_date == 1].index
    recurring_hist_df = recurring_hist_df[recurring_hist_df["date"].isin(single_event_dates)]
    
    historical_events_dict = recurring_hist_df.set_index("date")["event_name"].to_dict()
    
    # Calculate historical impacts
    historical_event_impacts = calculate_historical_event_impacts(
        processed_df, historical_events_dict, verbose=verbose
    )
    
    # STEP 2: Load future events but only keep those in historical_event_impacts
    future_events_dict = load_future_events_matching_historical(
        RECURRING_EVENTS_PATH, historical_event_impacts
    )
    
    weather_future = pd.DataFrame()
    if OPENMETEO_AVAILABLE:
        try:
            weather_future = get_openmeteo_for_future(days=openmeteo_days)
            weather_future["date"] = pd.to_datetime(weather_future["date"]).dt.normalize()
        except:
            pass
    
    hist_weather = processed_df.groupby(["month", "day"])[
        ["temperature", "rain_morning", "rain_afternoon", "precip_morning", "precip_afternoon"]
    ].mean().to_dict(orient="index")
    
    # Forecast loop
    if verbose:
        print(f"\nForecasting {forecast_days} days...")
    
    all_rows = []
    event_count = 0
    total_event_impact = 0
    
    today = pd.Timestamp.now().normalize()
    
    for step in range(forecast_days):
        current_date = today + timedelta(days=step)
        year, month, day = current_date.year, current_date.month, current_date.day
        weekday, week = current_date.weekday(), current_date.isocalendar().week
        doy = current_date.dayofyear
        
        w_row = hist_weather.get((month, day), {
            "temperature": 10.0, "rain_morning": 0, "rain_afternoon": 0,
            "precip_morning": 0, "precip_afternoon": 0
        })
        if not weather_future.empty:
            match = weather_future[weather_future["date"] == current_date]
            if not match.empty:
                w_row = match.iloc[0].to_dict()
        
        temperature = _safe_float(w_row.get("temperature", 10.0))
        rain_morning = _safe_float(w_row.get("rain_morning", 0))
        rain_afternoon = _safe_float(w_row.get("rain_afternoon", 0))
        precip_morning = _safe_float(w_row.get("precip_morning", 0))
        precip_afternoon = _safe_float(w_row.get("precip_afternoon", 0))
        
        days_until_hol, days_since_hol = _compute_holiday_proximity(current_date, holiday_dates_sorted)
        hol_feats = holiday_lookup.get(current_date, {})
        camp_feats = camp_lookup.get((year, week), {})
        
        # Check if this date has an event (only those matching historical)
        event_name = future_events_dict.get(current_date, "no_event")
        has_event = event_name != "no_event"
        
        evt_feats = {f"event_{event_name}": 1} if has_event else {}
        
        features_dict = {
            "year": year, "month": month, "day": day, "week": week, "weekday": weekday,
            "day_of_year": doy, "is_weekend": int(weekday >= 5),
            "temperature": temperature, "rain_morning": rain_morning, "rain_afternoon": rain_afternoon,
            "precip_morning": precip_morning, "precip_afternoon": precip_afternoon,
            "days_until_holiday": days_until_hol, "days_since_holiday": days_since_hol,
            "holiday_intensity": _safe_float(hol_feats.get("holiday_intensity", 0)),
            "campaign_strength": _safe_float(camp_feats.get("campaign_strength", 0)),
            "promotion_active": _safe_int(camp_feats.get("promotion_active", 0)),
            "campaign_regions_active": _safe_int(camp_feats.get("campaign_regions_active", 0)),
        }
        features_dict.update(hol_feats)
        
        # Predict
        per_ticket = _predict_ticket_batch(
            ticket_list=ticket_list,
            features_dict=features_dict,
            evt_feats=evt_feats,
            include_events=True,
            model_features=model_features,
            ticket_meta=ticket_meta,
            hist_by_ticket=hist_by_ticket,
            family_models=family_models,
            global_model=global_model,
            global_wape=global_wape,
            family_wape_map=family_wape_map,
            family_calibration=family_calibration,
            current_date=current_date
        )
        
        scaled = _apply_scaling_pipeline(
            per_ticket=per_ticket,
            current_date=current_date,
            target_family_map=target_family_map,
            target_doy_map=target_doy_map,
            family_wape_map=family_wape_map,
            extreme_multipliers=extreme_multipliers,
            families=families,
            apply_shape_injection=True,
            apply_weekend_modifier=True
        )
        
        # Use exact historical impact
        daily_event_impact = 0
        
        if has_event:
            daily_event_impact = historical_event_impacts.get(event_name, 0)
            event_count += 1
            total_event_impact += daily_event_impact
        
        # Distribute across tickets
        day_total = sum(val for _, _, val, _ in scaled)
        
        for tname, tfam, val, mused in scaled:
            if day_total > 0 and has_event:
                ticket_share = val / day_total
                ticket_impact = int(round(daily_event_impact * ticket_share))
            else:
                ticket_impact = 0
            
            all_rows.append({
                "date": current_date,
                "ticket_name": tname,
                "ticket_family": tfam,
                "predicted_sales": int(round(val)),
                "event_impact": ticket_impact,
                "event_name": event_name,
                "model_used": mused,
                "year": year,
                "month": month,
                "day": day,
                "temperature": round(temperature, 1),
                "total_rain": round(rain_morning + rain_afternoon, 1),
                "total_precipitation": round(precip_morning + precip_afternoon, 1),
            })
        
        if verbose and (step + 1) % 50 == 0:
            print(f"Progress: {step+1}/{forecast_days} days")
    
    # Finalize
    forecast_df = pd.DataFrame(all_rows)
    
    if verbose:
        print("\n" + "="*70)
        print("FORECAST COMPLETE")
        print("="*70)
        print(f"Total Predicted Sales: {forecast_df['predicted_sales'].sum():,}")
        print(f"Event Days (historical events only): {event_count}")
        print(f"Total Event Impact: {total_event_impact:+,}")
        if event_count > 0:
            print(f"Avg Impact per Event Day: {total_event_impact / event_count:+,.0f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = PREDICTIONS_DIR / f"forecast_365days_historical_only_{timestamp}.csv"
    forecast_df.to_csv(out_path, index=False)
    
    if verbose:
        print(f"\nSaved to: {out_path}")
        
        print("\n" + "="*70)
        print("EVENT IMPACT VERIFICATION")
        print("="*70)
        print(f"{'Event':<45s} {'Historical':>12s} {'Forecast':>12s} {'Match':>8s}")
        print("-"*70)
        
        event_summary = forecast_df[forecast_df['event_name'] != 'no_event'].groupby('event_name').agg({
            'event_impact': 'sum',
            'date': 'nunique'
        }).rename(columns={'date': 'occurrences'})
        
        for evt_name in sorted(historical_event_impacts.keys(), key=lambda x: historical_event_impacts[x], reverse=True):
            hist_val = historical_event_impacts[evt_name]
            if evt_name in event_summary.index:
                forecast_total = event_summary.loc[evt_name, 'event_impact']
                occurrences = event_summary.loc[evt_name, 'occurrences']
                forecast_avg = int(forecast_total / occurrences)
                match = "✓" if abs(forecast_avg - hist_val) < 10 else "✗"
            else:
                forecast_avg = 0
                match = "-"
            print(f"{evt_name:<45s} {hist_val:+12d} {forecast_avg:+12d} {match:>8s}")
    
    return forecast_df


if __name__ == "__main__":
    forecast_df = predict_next_365_days(forecast_days=365, verbose=True)
