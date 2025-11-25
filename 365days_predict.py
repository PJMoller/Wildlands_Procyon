import pandas as pd
import lightgbm as lgb
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')



# --- Configuration ---
MODEL_PATH = "data/processed/lgbm_model.pkl"
FAMILY_MODELS_PATH = "data/processed/family_models.pkl"
FEATURE_COLS_PATH = "data/processed/feature_cols.pkl"
PROCESSED_DATA_PATH = "data/processed/processed_merge.csv"
HOLIDAY_DATA_PATH = "data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx"
CAMPAIGN_DATA_PATH = "data/raw/campaings.xlsx"
RECURRING_EVENTS_PATH = "data/raw/recurring_events_drenthe.xlsx"
SEASONALITY_PROFILE_PATH = "data/processed/ticket_seasonality.csv"
TICKET_FAMILIES_PATH = "data/processed/ticket_families.csv"
PREDICTIONS_DIR = "data/predictions/"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)



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



def calculate_ticket_activation_probability(ticket_name, current_date, current_month, seasonality_df, 
                                           ticket_history, ticket_families, processed_df):
    """
    Calculate activation probability for a ticket type based on multiple signals
    Returns probability score between 0 and 1
    """
    probability_score = 0.0
    
    # 1. Seasonality Score (40% weight)
    seasonality_df['active_months_list'] = seasonality_df['active_months'].astype(str).str.split(',')
    is_seasonal = seasonality_df[
        seasonality_df['active_months_list'].apply(lambda x: str(current_month) in x)
    ]['ticket_name'].tolist()
    
    if ticket_name in is_seasonal:
        seasonality_score = 0.8
    else:
        historical_month_sales = processed_df[
            (processed_df['ticket_name'] == ticket_name) & 
            (processed_df['date'].dt.month == current_month) &
            (processed_df['ticket_num'] > 0)
        ]
        seasonality_score = min(1.0, len(historical_month_sales) / 30.0) * 0.6
    
    probability_score += seasonality_score * 0.4
    
    # 2. Recency Score (30% weight)
    last_sale_date = ticket_history.index.max() if not ticket_history.empty else None
    if last_sale_date:
        days_since_last_sale = (current_date - last_sale_date).days
        if days_since_last_sale <= 7:
            recency_score = 1.0
        elif days_since_last_sale <= 30:
            recency_score = 0.7
        elif days_since_last_sale <= 60:
            recency_score = 0.4
        else:
            recency_score = 0.1
    else:
        recency_score = 0.0
    
    probability_score += recency_score * 0.3
    
    # 3. Weekday Pattern Score (20% weight)
    current_weekday = current_date.weekday()
    weekday_sales = processed_df[
        (processed_df['ticket_name'] == ticket_name) & 
        (processed_df['date'].dt.weekday == current_weekday) &
        (processed_df['ticket_num'] > 0)
    ]
    
    if len(weekday_sales) > 0:
        weekday_score = min(1.0, len(weekday_sales) / 10.0)
    else:
        weekday_score = 0.2
    
    probability_score += weekday_score * 0.2
    
    # 4. Ticket Family Baseline Score (10% weight)
    family = ticket_families.get(ticket_name, 'general')
    family_avg_sales = processed_df[
        (processed_df['ticket_family'] == family) & 
        (processed_df['ticket_num'] > 0)
    ]['ticket_num'].mean()
    
    if family_avg_sales > 0:
        family_score = min(1.0, family_avg_sales / 100.0)
    else:
        family_score = 0.1
    
    probability_score += family_score * 0.1
    
    return min(1.0, max(0.0, probability_score))



def create_family_level_predictions(model, family_models, input_df, ticket_name, ticket_family, family_performance):
    """
    Create predictions using both global and family-level models with adaptive weighting
    """
    # Global model prediction
    global_pred = model.predict(input_df)[0]
    
    # Family model prediction (if available)
    if ticket_family in family_models:
        family_pred = family_models[ticket_family].predict(input_df)[0]
        
        # Get family WAPE from training (lower WAPE = higher weight)
        family_wape = family_performance.get(ticket_family, {}).get('WAPE', 0.5)
        family_weight = max(0.2, 1.0 - (family_wape / 100.0))  # Convert WAPE to weight
        
        # Weighted average: more weight to better-performing models
        final_pred = (0.7 * global_pred) + (0.3 * family_weight * family_pred)
    else:
        final_pred = global_pred
    
    return max(0, final_pred)



def predict_next_365_days():
    # --- 1. Load All Data, Models, and Profiles ---
    print("Step 1: Loading all data, models, and profiles...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(FAMILY_MODELS_PATH, 'rb') as f:
            family_models = pickle.load(f)
        
        with open(FEATURE_COLS_PATH, 'rb') as f:
            model_features = pickle.load(f)
        
        ticket_families_df = pd.read_csv(TICKET_FAMILIES_PATH)
        ticket_families = dict(zip(ticket_families_df['ticket_name'], ticket_families_df['ticket_family']))
        
        seasonality_df = pd.read_csv(SEASONALITY_PROFILE_PATH)
        holiday_og_df = pd.read_excel(HOLIDAY_DATA_PATH)
        camp_og_df = pd.read_excel(CAMPAIGN_DATA_PATH)
        recurring_og_df = pd.read_excel(RECURRING_EVENTS_PATH)
        
    except Exception as e:
        print(f"CRITICAL ERROR loading initial files: {e}")
        return

    # --- 2. Prepare Lookups and Model Information ---
    print("Step 2: Preparing lookups and model information...")
    
    # Load and fix data types
    processed_df = pd.read_csv(PROCESSED_DATA_PATH)
    processed_df['date'] = pd.to_datetime(processed_df['date'])
    processed_df['has_sales'] = (processed_df['ticket_num'] > 0).astype(int)
    
    # CRITICAL FIX: Convert holiday columns to numeric
    holiday_cols = [c for c in processed_df.columns if c.startswith("holiday_")]
    for c in holiday_cols:
        processed_df[c] = (
            processed_df[c]
            .map({True: 1, False: 0, "True": 1, "False": 0, "1.0": 1, "0.0": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype('int8')
        )
    
    # Train binary classification model
    print("Training binary classification model...")
    
    # MAXIMUM FEATURE REMOVAL to prevent overfitting
    binary_features = [col for col in model_features 
                       if not any(x in col for x in ['sales_lag', 'sales_rolling', 'sales_momentum', 'sales_trend', 'family_', 'ticket_', 'campaign_strength']) 
                       and col not in ['ticket_num', 'date', 'ticket_name', 'ticket_family', 'has_sales']]
    
    X_binary = processed_df[binary_features]
    y_binary = processed_df['has_sales']
    
    # Add regularization to prevent overfitting
    binary_model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=5,      # Limit tree depth
        min_samples_leaf=20,  # Require more samples per leaf
        random_state=42, 
        n_jobs=-1
    )
    binary_model.fit(X_binary, y_binary)
    
    # WARN if accuracy is too high
    binary_accuracy = binary_model.score(X_binary, y_binary)
    print(f"Binary model trained: {binary_accuracy:.3f} accuracy")
    if binary_accuracy > 0.95:
        print("⚠️  WARNING: Binary model accuracy is suspiciously high (>0.95). May be overfitting!")
        print("⚠️  Consider removing more features or adding regularization.")
    
    # Save binary model
    with open("data/processed/binary_model.pkl", "wb") as f:
        pickle.dump(binary_model, f)
    
    # Calculate bias correction factor from validation data
    print("Calculating bias correction factors...")
    
    # Calculate GLOBAL calibration factor
    print("Calculating global calibration factor...")
    last_date = processed_df['date'].max()
    validation_start = last_date - pd.Timedelta(days=60)
    val_df = processed_df[processed_df['date'] >= validation_start]
    
    if len(val_df) > 0:
        val_features = val_df[model_features].copy()
        for c in holiday_cols:
            if c in val_features.columns:
                val_features[c] = pd.to_numeric(val_features[c], errors='coerce').fillna(0).astype('int8')
        
        val_pred_raw = model.predict(val_features)
        global_calibration = val_df['ticket_num'].sum() / val_pred_raw.sum()
        print(f"Global calibration: {global_calibration:.3f}")
    else:
        global_calibration = 1.0
        print("⚠️  No validation data, using calibration = 1.0")
    
    # Check if validation data exists
    bias_correction_factors = {}
    if len(val_df) == 0:
        print("ERROR: No validation data found. Using bias correction = 1.0 for all families")
        bias_correction_factors = {f: 1.0 for f in ['subscription', 'general', 'single_day', 'group_package', 'fixed_seasonal']}
    else:
        # FAMILY-SPECIFIC bias correction
        for family in ['subscription', 'general', 'single_day', 'group_package', 'fixed_seasonal']:
            family_val_df = val_df[val_df['ticket_family'] == family]
            if len(family_val_df) > 0:
                family_val_features = family_val_df[model_features].copy()
                
                # Fix dtypes in family_val_features
                for c in holiday_cols:
                    if c in family_val_features.columns:
                        family_val_features[c] = pd.to_numeric(family_val_features[c], errors='coerce').fillna(0).astype('int8')
                
                family_val_pred = model.predict(family_val_features)
                if family_val_pred.sum() > 0:
                    family_bias = family_val_df['ticket_num'].sum() / family_val_pred.sum()
                    bias_correction_factors[family] = max(0.5, family_bias)  # CRITICAL: Minimum 0.5
                else:
                    bias_correction_factors[family] = 1.0
                print(f"  {family}: {bias_correction_factors[family]:.3f}")
            else:
                bias_correction_factors[family] = 1.0
    
    # Load family performance metrics
    try:
        with open("data/processed/model_performance.pkl", 'rb') as f:
            performance_data = pickle.load(f)
            family_performance = performance_data.get('family_models', {})
    except:
        family_performance = {}
    
    # Holiday lookups
    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    all_holidays_lookup = holiday_og_df.melt(id_vars=['date'], value_vars=region_cols, var_name='region', value_name='holiday').dropna()
    all_holidays_lookup['date'] = pd.to_datetime(all_holidays_lookup['date'])
    holiday_dates = pd.to_datetime(pd.Series(all_holidays_lookup['date'].unique())).sort_values()
    summer_dates = all_holidays_lookup[all_holidays_lookup['holiday'].str.contains("Zomervakantie", na=False)]['date'].unique()
    
    # Campaign lookups
    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    promo_cols = [col for col in camp_og_df.columns if col.startswith('promo_')]
    camp_og_df['campaign_strength'] = camp_og_df[promo_cols].sum(axis=1)
    camp_og_df['campaign_regions_active'] = (camp_og_df[promo_cols] > 0).sum(axis=1)
    
    # Recurring events
    recurring_og_df.columns = ["event_name", "date"]
    recurring_df_lookup = pd.get_dummies(recurring_og_df.set_index('date')['event_name'].str.split('/', expand=True).stack(), prefix='event').groupby(level=0).max().reset_index()
    recurring_df_lookup['date'] = pd.to_datetime(recurring_df_lookup['date'])

    # --- 3. Prepare Historical Data ---
    print("Step 3: Preparing historical data...")
    
    # Get all ticket names and their families
    all_ticket_info = processed_df[['ticket_name', 'ticket_family']].drop_duplicates()
    ticket_families_full = dict(zip(all_ticket_info['ticket_name'], all_ticket_info['ticket_family']))
    
    # Get last known date
    last_known_date = processed_df['date'].max()
    print(f"Last known date: {last_known_date.date()}")
    
    # Load recent history (last 90 days) - CRITICAL: Only use ACTUAL sales
    recent_history = processed_df[
        processed_df['date'] >= last_known_date - timedelta(days=90)
    ].copy()
    
    # Create ticket history lookup
    ticket_history_lookup = {}
    for ticket_name in all_ticket_info['ticket_name']:
        ticket_data = recent_history[recent_history['ticket_name'] == ticket_name].set_index('date')['ticket_num']
        ticket_history_lookup[ticket_name] = ticket_data
    
    # Get ticket-specific historical max for capping predictions
    ticket_max_sales = processed_df.groupby('ticket_name')['ticket_num'].max().to_dict()

    # --- 4. Prepare Future Data & Fallbacks ---
    print("Step 4: Preparing future weather data and fallbacks...")
    weather_future_df = get_openmeteo_for_future(16)
    historical_weather_avg = processed_df.groupby(['month', 'day'])[
        ['temperature', 'rain_morning', 'rain_afternoon', 'precip_morning', 'precip_afternoon']
    ].mean().reset_index()

    # --- 5. Initialize Prediction Storage ---
    print("\nStep 5: Starting 365-day prediction loop...")
    all_predictions = []

    # --- 6. Main Prediction Loop ---
    for d in range(365):
        current_date = last_known_date + timedelta(days=d + 1)
        current_month = current_date.month
        current_weekday = current_date.weekday()
        
        print(f"\nPredicting for {current_date.date()}...")
        
        day_predictions = []
        
        for _, row in all_ticket_info.iterrows():
            ticket_name = row['ticket_name']
            ticket_family = row['ticket_family']
            
            # Get ticket history
            ticket_history = ticket_history_lookup.get(ticket_name, pd.Series())
            
            # Calculate activation probability
            activation_prob = calculate_ticket_activation_probability(
                ticket_name, current_date, current_month, seasonality_df,
                ticket_history, ticket_families_full, processed_df
            )
            
            # A. Date features
            date_features = {
                'year': current_date.year,
                'month': current_month,
                'day': current_date.day,
                'weekday': current_weekday,
                'week': current_date.isocalendar().week
            }
            date_features.update({
                'is_weekend': 1 if date_features['weekday'] >= 5 else 0,
                'is_month_start': 1 if date_features['day'] in [1, 2, 3] else 0,
                'is_month_mid': 1 if date_features['day'] in [14, 15, 16] else 0,
                'is_month_end': 1 if current_date.is_month_end else 0
            })
            
            # B. Weather features
            weather_match = weather_future_df[
                weather_future_df['date'].dt.date == current_date.date()
            ]
            if not weather_match.empty:
                weather_row = weather_match.iloc[0]
                weather_features = {
                    'temperature': float(weather_row['temperature']),
                    'rain_morning': float(weather_row['rain_morning']),
                    'rain_afternoon': float(weather_row['rain_afternoon']),
                    'precip_morning': float(weather_row['precip_morning']),
                    'precip_afternoon': float(weather_row['precip_afternoon']),
                }
            else:
                avg_match = historical_weather_avg[
                    (historical_weather_avg['month'] == current_month) &
                    (historical_weather_avg['day'] == current_date.day)
                ]
                if not avg_match.empty:
                    weather_row = avg_match.iloc[0]
                    weather_features = {
                        'temperature': float(weather_row['temperature']),
                        'rain_morning': float(weather_row['rain_morning']),
                        'rain_afternoon': float(weather_row['rain_afternoon']),
                        'precip_morning': float(weather_row['precip_morning']),
                        'precip_afternoon': float(weather_row['precip_afternoon']),
                    }
                else:
                    weather_features = {
                        'temperature': 10.0,
                        'rain_morning': 0.0,
                        'rain_afternoon': 0.0,
                        'precip_morning': 0.0,
                        'precip_afternoon': 0.0,
                    }
            
            # C. Holiday features
            next_h = holiday_dates[holiday_dates >= current_date].iloc[0] \
                if not holiday_dates[holiday_dates >= current_date].empty \
                else current_date + timedelta(days=90)
            prev_h = holiday_dates[holiday_dates <= current_date].iloc[-1] \
                if not holiday_dates[holiday_dates <= current_date].empty \
                else current_date - timedelta(days=90)
            
            date_features.update({
                'days_until_holiday': (next_h - current_date).days,
                'days_since_holiday': (current_date - prev_h).days,
                'is_summer_vacation': 1 if current_date in summer_dates else 0,
                'temp_x_weekend': weather_features['temperature'] * date_features['is_weekend']
            })
            
            # Holiday one-hot encoding
            holiday_one_hot = {
                f"{row['region']}_{row['holiday']}": 1
                for _, row in all_holidays_lookup[
                    all_holidays_lookup['date'] == current_date
                ].iterrows()
            }
            
            # D. Campaign features
            campaign_today = camp_og_df[
                (camp_og_df['year'] == date_features['year']) &
                (camp_og_df['week'] == date_features['week'])
            ]
            campaign_features = {
                col: campaign_today[col].iloc[0] if not campaign_today.empty else 0
                for col in campaign_today.columns if col.startswith('promo_')
            }
            
            if not campaign_today.empty:
                campaign_features['campaign_strength'] = campaign_today['campaign_strength'].iloc[0]
                campaign_features['campaign_regions_active'] = campaign_today['campaign_regions_active'].iloc[0]
            else:
                campaign_features['campaign_strength'] = 0
                campaign_features['campaign_regions_active'] = 0
            
            # E. Event features
            events_today = recurring_df_lookup[
                recurring_df_lookup['date'].dt.date == current_date.date()
            ]
            event_features = {
                col: 1 for col in events_today.columns if col.startswith('event_')
            } if not events_today.empty else {}
            
            # F. Lag/Rolling features (from ACTUAL history only, not predictions)
            lag_features = {}
            for lag in [1, 7, 14, 30]:
                lag_date = current_date - timedelta(days=lag)
                if lag_date in ticket_history.index:
                    lag_value = ticket_history.loc[lag_date]
                    # Handle potential Series
                    if isinstance(lag_value, pd.Series):
                        lag_value = lag_value.iloc[0] if len(lag_value) > 0 else 0
                    lag_features[f'sales_lag_{lag}'] = float(lag_value) if pd.notna(lag_value) else 0.0
                else:
                    lag_features[f'sales_lag_{lag}'] = 0.0
            
            # Rolling features
            rolling_end_date = current_date - timedelta(days=1)
            rolling_window = ticket_history.loc[
                :rolling_end_date
            ].tail(7)
            
            if not rolling_window.empty:
                rolling_mean = rolling_window.mean()
                rolling_std = rolling_window.std()
                lag_features['sales_rolling_avg_7'] = float(rolling_mean) if pd.notna(rolling_mean) else 0.0
                lag_features['sales_rolling_std_7'] = float(rolling_std) if pd.notna(rolling_std) else 0.0
            else:
                lag_features['sales_rolling_avg_7'] = 0.0
                lag_features['sales_rolling_std_7'] = 0.0
            
            # Add momentum features
            lag_features['sales_momentum_7d'] = float(lag_features['sales_lag_1'] - lag_features['sales_lag_7'])
            
            # G. Ticket-specific features
            ticket_features = {
                f'ticket_{ticket_name}': 1,
                f'family_{ticket_family}': 1
            }
            
            # Add family sales features (from previous day)
            for family_name in ticket_families_full.values():
                family_sales = processed_df[
                    (processed_df['date'] == current_date - timedelta(days=1)) &
                    (processed_df['ticket_family'] == family_name)
                ]['ticket_num'].sum()
                ticket_features[f'family_{family_name}_sales'] = float(family_sales)
            
            # H. Assemble final row
            input_row = {
                **date_features,
                **weather_features,
                **lag_features,
                **campaign_features,
                **holiday_one_hot,
                **event_features,
                **ticket_features
            }
            
            input_df = pd.DataFrame([input_row]).reindex(
                columns=model_features,
                fill_value=0
            )
            
            # CRITICAL: Ensure all values are numeric (especially holiday columns)
            for col in input_df.columns:
                if col.startswith('holiday_'):
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype('int8')
                elif input_df[col].dtype == 'object':
                    try:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                    except:
                        input_df[col] = 0
            

            # I. Binary classification prediction (use only for curiosity, not dampening)
            binary_input = input_df[binary_features].copy()
            for col in binary_features:
                if binary_input[col].dtype == 'object':
                    binary_input[col] = pd.to_numeric(binary_input[col], errors='coerce')
            
            will_sell_prob = binary_model.predict_proba(binary_input)[0][1]
            
            # J. Seasonal baseline
            seasonal_avg = processed_df[
                (processed_df['ticket_name'] == ticket_name) &
                (processed_df['date'].dt.month == current_month)
            ]['ticket_num'].mean()
            if pd.isna(seasonal_avg):
                seasonal_avg = 0
            
            # K. PREDICT - TRUST THE MODEL
            # Skip complex probability dampening - let model predictions through
            predicted_sales_raw = create_family_level_predictions(
                model, family_models, input_df, ticket_name, ticket_family, family_performance
            )
            
            # Only minimal safety net: ensure predictions are reasonable
            # If activation probability is very low AND no recent sales history, dampen slightly
            if activation_prob < 0.10 and ticket_history.sum() == 0:
                # This ticket has never sold and activation is low - heavy dampening
                predicted_sales = predicted_sales_raw * 0.1
            elif activation_prob < 0.15:
                # Light dampening for low confidence
                predicted_sales = predicted_sales_raw * 0.5
            else:
                # TRUST THE MODEL for everything else
                predicted_sales = predicted_sales_raw
            
            # L. Apply global calibration (CRITICAL - this fixes the scale)
            predicted_sales = predicted_sales * global_calibration
            
            # M. Blend with seasonal baseline (very light - just 5%)
            predicted_sales = (0.95 * predicted_sales) + (0.05 * seasonal_avg)
            
            # N. Cap at historical maximum
            historical_max = ticket_max_sales.get(ticket_name, 1000)
            final_prediction = max(0, min(round(predicted_sales), historical_max))
            
            # Always add to predictions (including zeros) for proper aggregation
            day_predictions.append({
                "date": current_date,
                "ticket_name": ticket_name,
                "ticket_family": ticket_family,
                "predicted_sales": final_prediction,
                "activation_probability": activation_prob,
                "binary_probability": will_sell_prob,
                "raw_prediction": predicted_sales_raw,
                "seasonal_baseline": seasonal_avg
            })

        
        # Add day predictions to all_predictions list
        if day_predictions:
            all_predictions.extend(day_predictions)

    # --- 7. Save Final Results ---
    print("\nStep 6: Saving final forecast...")
    if all_predictions:
        forecast_df = pd.DataFrame(all_predictions)
        # Sort and clean output
        forecast_df = forecast_df.sort_values(['date', 'ticket_family', 'ticket_name']).reset_index(drop=True)
        
        # Save
        filename = f"forecast_365_days_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        output_path = os.path.join(PREDICTIONS_DIR, filename)
        forecast_df.to_csv(output_path, index=False)
        
        print(f"✅ Forecast saved successfully to {output_path}")
        print(f"Total predictions: {len(forecast_df)}")
        print(f"Date range: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
        print(f"Ticket families: {forecast_df['ticket_family'].value_counts().to_dict()}")
        
        # Show summary statistics
        total_predicted = forecast_df['predicted_sales'].sum()
        avg_per_day = forecast_df.groupby('date')['predicted_sales'].sum().mean()
        print(f"\nSummary Statistics:")
        print(f"Total predicted sales: {total_predicted:,.0f}")
        print(f"Average daily total: {avg_per_day:,.0f}")
        
    else:
        print("WARNING: No predictions generated.")


if __name__ == "__main__":
    predict_next_365_days()
