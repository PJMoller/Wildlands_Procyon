import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import pickle
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import os

MODEL_PATH = "data/processed/lgbm_model.pkl"
PROCESSED_DATA_PATH = "data/processed/processed_merge.csv"
HOLIDAY_DATA_PATH = "data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx"
CAMPAIGN_DATA_PATH = "data/raw/campaings.xlsx"
PREDICTIONS_DIR = "data/predictions/"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def get_openmeteo_for_future(days=16):
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.7862, "longitude": 6.8917,
        "daily": ["temperature_2m_mean", "precipitation_sum"],
        "forecast_days": days
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_data = {
        "date": pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert(None).date,
        "temperature": daily.Variables(0).ValuesAsNumpy(),
        "precipitation": daily.Variables(1).ValuesAsNumpy(),
    }
    
    return pd.DataFrame(data=daily_data)


def predict_next_365_days():
    # --- Load Initial Data ---
    print("Loading all initial data...")
    try:
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
        holiday_og_df = pd.read_excel(HOLIDAY_DATA_PATH)
        camp_og_df = pd.read_excel(CAMPAIGN_DATA_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR loading initial files: {e}"); return

    # --- Define Column Sets and Get Last Date ---
    print("Defining column sets...")
    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    
    with open(PROCESSED_DATA_PATH, 'r') as f: header = f.readline().strip().split(',')
    ticket_type_cols = [col for col in header if col.startswith('ticket_')]
    all_ticket_names = [col.replace('ticket_', '') for col in ticket_type_cols]
    if not all_ticket_names: print("CRITICAL ERROR: 'all_ticket_names' list is empty."); return
    print(f"Found {len(all_ticket_names)} unique ticket names.")

    with open(PROCESSED_DATA_PATH, 'rb') as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n': f.seek(-2, os.SEEK_CUR)
        except OSError: f.seek(0)
        last_line = f.readline().decode()
    last_year, last_month, last_day = map(int, last_line.split(',')[:3])
    last_known_date = pd.to_datetime(f"{last_year}-{last_month}-{last_day}")

    # --- Prepare Historical Data ---
    print("Preparing historical datasets...")
    history_start_date = last_known_date - timedelta(days=30)
    required_history_cols = ['year', 'month', 'day', 'ticket_num'] + ticket_type_cols
    required_weather_cols = ['month', 'day', 'temperature', 'precipitation']
    all_required_cols = list(set(required_history_cols + required_weather_cols))

    recent_history_chunks = []
    weather_avg_chunks = []
    for chunk in pd.read_csv(PROCESSED_DATA_PATH, chunksize=100000, usecols=all_required_cols):
        chunk['date'] = pd.to_datetime(chunk[['year', 'month', 'day']])
        recent_chunk = chunk[chunk['date'] >= history_start_date]
        if not recent_chunk.empty: recent_history_chunks.append(recent_chunk)
    
    if not recent_history_chunks: print("Error: No recent data found."); return
    recent_history_df = pd.concat(recent_history_chunks)
    weather_avg_chunks.append(chunk)
    
    history_df = pd.melt(recent_history_df, id_vars=['date', 'ticket_num'], value_vars=ticket_type_cols, var_name='ticket_name_encoded', value_name='is_present')
    history_df = history_df[history_df['is_present'] == 1]
    history_df['ticket_name'] = history_df['ticket_name_encoded'].str.replace('ticket_', '', regex=False)
    history_df = history_df[["date", "ticket_name", "ticket_num"]]
    print("All data prepared.")
    
    historical_weather = pd.concat(weather_avg_chunks).groupby(['month', 'day']).mean().reset_index()
    weather_future_df = get_openmeteo_for_future()
    holiday_region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    all_holidays = holiday_og_df.melt(id_vars=['date'], value_vars=holiday_region_cols, var_name='region', value_name='holiday')
    holiday_dates = pd.to_datetime(pd.Series(all_holidays.dropna()['date'].unique())).sort_values()
    summer_dates = all_holidays[all_holidays['holiday'].str.contains("Zomervakantie", na=False)]['date'].unique()
    print("All data prepared.")

    unique_holidays = all_holidays['holiday'].dropna().unique()
    possible_holiday_features = {
        f"{region}_{holiday}".replace(' ', '_')
        for region in holiday_region_cols
        for holiday in unique_holidays
    }
    holiday_feature_names = [name for name in model.feature_name_ if name in possible_holiday_features]

    # --- 4. Main Prediction Loop ---
    print("Starting iterative prediction loop...")
    all_predictions = []
    loop_history_df = history_df.copy()

    for d in range(365):
        current_date = last_known_date + timedelta(days=d + 1)
        predictions_for_today = {}
        history_updates_for_today = []
        # --- Create Features for the Current Day ---
        # A. Basic Date Features
        date_features = {'year': current_date.year, 'month': current_date.month, 'day': current_date.day, 'weekday': current_date.weekday(), 'week': current_date.isocalendar().week}
        date_features['is_weekend'] = 1 if date_features['weekday'] >= 5 else 0

        # B. Holiday & Payday Features
        next_h = holiday_dates[holiday_dates >= current_date].iloc[0]
        prev_h = holiday_dates[holiday_dates <= current_date].iloc[-1]
        date_features['days_until_holiday'] = (next_h - current_date).days
        date_features['days_since_holiday'] = (current_date - prev_h).days
        date_features['is_month_start'] = 1 if date_features['day'] in [1, 2, 3] else 0
        date_features['is_month_mid'] = 1 if date_features['day'] in [14, 15, 16] else 0
        date_features['is_month_end'] = 1 if current_date.is_month_end else 0

        # C. Weather Features (use forecast or historical average)
        weather_match = weather_future_df[weather_future_df['date'] == current_date.date()]
        if not weather_match.empty:
            weather_features = weather_match.to_dict(orient='records')[0]
        else: # Fallback to historical average
            avg_weather = historical_weather[(historical_weather['month'] == current_date.month) & (historical_weather['day'] == current_date.day)]
            weather_features = avg_weather.to_dict(orient='records')[0] if not avg_weather.empty else {'temperature': 10.0, 'precipitation': 0.1}

        # D. Campaign & Event Features
        campaign_features = {}
        campaign_today = camp_og_df[(camp_og_df['year'] == date_features['year']) & (camp_og_df['week'] == date_features['week'])]
        if not campaign_today.empty:
            # Assuming your campaign columns are named 'promo_NLNoord', etc.
            promo_cols = [col for col in camp_og_df.columns if col.startswith('promo_')]
            for col in promo_cols:
                campaign_features[col] = campaign_today[col].iloc[0]
        
        # Get summer vacation flag
        date_features['is_summer_vacation'] = 1 if current_date in summer_dates else 0
        
        # Get interaction feature
        date_features['temp_x_weekend'] = weather_features['temperature'] * date_features['is_weekend']
        
        # Get one-hot-encoded holiday features
        holiday_one_hot_features = {}
        
        # Find which holiday (if any) is on the current_date
        current_holiday_info = all_holidays[all_holidays['date'] == current_date].dropna()
        for col_name in holiday_feature_names:
            holiday_one_hot_features[col_name] = 0 # Default to 0
        
        if not current_holiday_info.empty:
            for index, row in current_holiday_info.iterrows():
                feature_name = f"{row['region']}_{row['holiday']}".replace(' ', '_')
                if feature_name in holiday_one_hot_features:
                    holiday_one_hot_features[feature_name] = 1

        print(f"Starting prediction for {current_date.date()}...")
        for ticket_name in all_ticket_names:
            try:
                # Lag features must be created inside this ticket-specific loop
                ticket_history = loop_history_df[loop_history_df["ticket_name"] == ticket_name].set_index('date')['ticket_num']
                lag_features = {}
                lags = [1, 7, 14]
                for lag in lags: lag_features[f'sales_lag_{lag}'] = ticket_history.get(current_date - timedelta(days=lag), 0)
                rolling_window = ticket_history.loc[:current_date - timedelta(days=1)].tail(7)
                lag_features['sales_rolling_avg_7'] = round(rolling_window.mean(), 2) if not rolling_window.empty else 0.0
                lag_features['sales_rolling_std_7'] = round(rolling_window.std(), 2) if not rolling_window.empty else 0.0

                # Assemble the full input row for prediction
                input_row = {**date_features, **weather_features, **lag_features, **campaign_features, **holiday_one_hot_features}
                for tt_col in ticket_type_cols:
                    input_row[tt_col] = 1 if tt_col == f'ticket_{ticket_name}' else 0
                
                input_df = pd.DataFrame([input_row]).reindex(columns=model.feature_name_, fill_value=0)
                predicted_sales = model.predict(input_df)[0]
                
                final_prediction = max(0, round(predicted_sales))
                if final_prediction > 0:
                    all_predictions.append({"date": current_date, "ticket_name": ticket_name, "predicted_sales": final_prediction})
                
                history_value = max(0.01, predicted_sales)
                new_history_row = pd.DataFrame([{"date": current_date, "ticket_name": ticket_name, "ticket_num": history_value}])
                loop_history_df = pd.concat([loop_history_df, new_history_row], ignore_index=True)
            except Exception as e:
                print(f"Error on day {d+1} for ticket {ticket_name}: {e}"); continue
        
    # --- Save Final Results ---
    if not all_predictions:
        print("WARNING: No positive sales were predicted. The final CSV will be empty.")
    else:
        print("Forecast generation complete.")
        forecast_df = pd.DataFrame(all_predictions)
        filename = f"forecast_365_days_{datetime.now().strftime('%Y-%m-%d')}.csv"
        output_path = os.path.join(PREDICTIONS_DIR, filename)
        forecast_df.to_csv(output_path, index=False)
        print(f"✅ Forecast saved successfully to {output_path}")



if __name__ == "__main__":
    predict_next_365_days()