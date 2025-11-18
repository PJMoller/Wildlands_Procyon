import pandas as pd
import lightgbm as lgb
import pickle
from datetime import datetime, timedelta
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np
import os


# --- Configuration ---
MODEL_PATH = "data/processed/lgbm_model.pkl"
PROCESSED_DATA_PATH = "data/processed/processed_merge.csv"
HOLIDAY_DATA_PATH = "data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx"
CAMPAIGN_DATA_PATH = "data/raw/campaings.xlsx"
RECURRING_EVENTS_PATH = "data/raw/recurring_events_drenthe.xlsx"
SEASONALITY_PROFILE_PATH = "data/processed/ticket_seasonality.csv"
PREDICTIONS_DIR = "data/predictions/"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)



def get_openmeteo_for_future(days=16):
    """Fetches and processes HOURLY weather data for the future."""
    # This function is correct and remains unchanged
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



def predict_next_365_days():
    # --- 1. Load All Data, Model, and Profiles ---
    print("Step 1: Loading all data, model, and seasonality profile...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        holiday_og_df = pd.read_excel(HOLIDAY_DATA_PATH)
        camp_og_df = pd.read_excel(CAMPAIGN_DATA_PATH)
        recurring_og_df = pd.read_excel(RECURRING_EVENTS_PATH)
        seasonality_df = pd.read_csv(SEASONALITY_PROFILE_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR loading initial files: {e}")
        return


    # --- 2. Prepare Lookups and Model Info ---
    print("Step 2: Preparing lookups and model information...")
    # Holiday lookups
    holiday_og_df.columns = [
        "NLNoord",
        "NLMidden",
        "NLZuid",
        "Niedersachsen",
        "Nordrhein-Westfalen",
        "date"
    ]
    holiday_region_cols = ["NLNoord", "NLMidden", "NLZuid",
                           "Niedersachsen", "Nordrhein-Westfalen"]
    all_holidays_lookup = holiday_og_df.melt(
        id_vars=['date'],
        value_vars=holiday_region_cols,
        var_name='region',
        value_name='holiday'
    ).dropna()
    holiday_dates = pd.to_datetime(
        pd.Series(all_holidays_lookup['date'].unique())
    ).sort_values()
    summer_dates = all_holidays_lookup[
        all_holidays_lookup['holiday'].str.contains("Zomervakantie", na=False)
    ]['date'].unique()
    
    # Other lookups
    camp_og_df.columns = [
        "year",
        "week",
        "promo_NLNoord",
        "promo_NLMidden",
        "promo_NLZuid",
        "promo_Nordrhein-Westfalen",
        "promo_Niedersachsen"
    ]
    recurring_og_df.columns = ["event_name", "date"]
    recurring_df_lookup = pd.get_dummies(
        recurring_og_df.set_index('date')['event_name']
        .str.split('/', expand=True)
        .stack(),
        prefix='event'
    ).groupby(level=0).max().reset_index()
    
    # Model features
    model_features = model.feature_name_
    event_type_cols = [col for col in model_features if col.startswith('event_')]


    # --- FIX: Correct way to get the last known date ---
    last_date_series = pd.read_csv(
        PROCESSED_DATA_PATH,
        usecols=['year', 'month', 'day']
    ).tail(1).squeeze()
    last_known_date = pd.to_datetime(
        f"{int(last_date_series['year'])}-"
        f"{int(last_date_series['month'])}-"
        f"{int(last_date_series['day'])}"
    )
    print(f"Last known date from data: {last_known_date.date()}")
    # --- END OF FIX ---


    # --- 3. Prepare Historical Data for Iterative Loop ---
    print("Step 3: Preparing recent history for iterative prediction...")
    csv_cols = pd.read_csv(PROCESSED_DATA_PATH, nrows=0).columns.tolist()
    ticket_type_cols_from_csv = [
        col for col in csv_cols if col.startswith('ticket_') and col != 'ticket_num'
    ]
    
    history_df_raw = pd.read_csv(
        PROCESSED_DATA_PATH,
        usecols=['year', 'month', 'day', 'ticket_num'] + ticket_type_cols_from_csv
    )
    history_df_raw['date'] = pd.to_datetime(
        history_df_raw[['year', 'month', 'day']]
    )
    history_df_raw = history_df_raw[
        history_df_raw['date'] >= last_known_date - timedelta(days=30)
    ]
    
    melted_df = history_df_raw.melt(
        id_vars=['date', 'ticket_num'],
        value_vars=ticket_type_cols_from_csv,
        var_name='ticket_name_encoded',
        value_name='is_present'
    )
    active_tickets_df = melted_df[melted_df['is_present'] == 1]
    history_df = active_tickets_df[[
        'date',
        'ticket_name_encoded',
        'ticket_num'
    ]].rename(columns={'ticket_name_encoded': 'ticket_name'})
    history_df['ticket_name'] = history_df['ticket_name'].str.replace(
        'ticket_', '',
        regex=False
    )

    # --- NEW: Simple "live ticket" + weekday activity logic ---

    # Use recent non-zero history to decide if a ticket is live
    nonzero_history = history_df[history_df['ticket_num'] > 0].copy()
    ticket_last_sale = (
        nonzero_history.groupby('ticket_name')['date']
        .max()
        .to_dict()
    )

    # Grace window for "still live" status (matches the 30-day history window here)
    GRACE_DAYS = 30
    lookback_cutoff = last_known_date - timedelta(days=GRACE_DAYS)

    def is_ticket_live_on_date(ticket_name, current_date):
        last_sale = ticket_last_sale.get(ticket_name)
        if last_sale is None:
            return False
        return last_sale >= lookback_cutoff and current_date >= last_sale

    # Optional: whether the ticket has ever sold on this weekday in the recent window
    hist_nonzero = history_df[history_df['ticket_num'] > 0].copy()
    hist_nonzero['weekday'] = hist_nonzero['date'].dt.weekday
    ticket_weekday_active = (
        hist_nonzero.groupby(['ticket_name', 'weekday'])['date']
        .nunique()
        .gt(0)
        .to_dict()
    )

    def is_ticket_active_on_weekday(ticket_name, weekday):
        return ticket_weekday_active.get((ticket_name, weekday), False)


    # --- 4. Prepare Future Data & Fallbacks ---
    print("Step 4: Preparing future weather data and fallbacks...")
    weather_future_df = get_openmeteo_for_future(16)
    historical_weather_avg = pd.read_csv(
        PROCESSED_DATA_PATH,
        usecols=[
            'month',
            'day',
            'temperature',
            'rain_morning',
            'rain_afternoon',
            'precip_morning',
            'precip_afternoon'
        ]
    ).groupby(['month', 'day']).mean().reset_index()


    # --- 5. Main 365-Day Prediction Loop ---
    print("\nStep 5: Starting intelligent 365-day prediction loop...")
    all_predictions = []
    loop_history_df = history_df.copy()

    for d in range(365):
        current_date = last_known_date + timedelta(days=d + 1)
        current_month = current_date.month
        current_weekday = current_date.weekday()

        # Get the list of tickets to predict for THIS month (seasonality)
        seasonality_df['active_months_list'] = (
            seasonality_df['active_months'].astype(str).str.split(',')
        )
        tickets_for_this_month = seasonality_df[
            seasonality_df['active_months_list'].apply(
                lambda x: str(current_month) in x
            )
        ]['ticket_name'].tolist()

        # NEW: Filter to tickets that are live and active on this weekday
        tickets_for_this_month = [
            t for t in tickets_for_this_month
            if is_ticket_live_on_date(t, current_date)
               and is_ticket_active_on_weekday(t, current_weekday)
        ]
        
        if not tickets_for_this_month:
            continue
        
        print(f"Predicting for {current_date.date()} ({len(tickets_for_this_month)} active tickets)...")

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
            weather_features = weather_match.drop(columns=['date']).to_dict(
                orient='records'
            )[0]
        else:
            avg_match = historical_weather_avg[
                (historical_weather_avg['month'] == current_month)
                & (historical_weather_avg['day'] == current_date.day)
            ]
            weather_features = avg_match.drop(
                columns=['month', 'day']
            ).to_dict(orient='records')[0] if not avg_match.empty else {
                'temperature': 10.0,
                'rain_morning': 0,
                'rain_afternoon': 0,
                'precip_morning': 0,
                'precip_afternoon': 0
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
        holiday_one_hot = {
            f"{row['region']}_{row['holiday']}": 1
            for _, row in all_holidays_lookup[
                all_holidays_lookup['date'] == current_date
            ].iterrows()
        }


        # D. Campaign features
        campaign_today = camp_og_df[
            (camp_og_df['year'] == date_features['year'])
            & (camp_og_df['week'] == date_features['week'])
        ]
        campaign_features = {
            col: campaign_today[col].iloc[0]
            for col in campaign_today.columns if col.startswith('promo_')
        } if not campaign_today.empty else {}
        
        # E. Event features
        events_today = recurring_df_lookup[
            recurring_df_lookup['date'].dt.date == current_date.date()
        ]
        event_features = {
            col: 1 for col in events_today.columns if col.startswith('event_')
        } if not events_today.empty else {}
        
        for ticket_name in tickets_for_this_month:
            # F. Lag/Rolling features
            ticket_history = loop_history_df[
                loop_history_df["ticket_name"] == ticket_name
            ].set_index('date')['ticket_num']

            lag_features = {
                f'sales_lag_{lag}': ticket_history.get(
                    current_date - timedelta(days=lag), 0
                )
                for lag in [1, 7, 14]
            }

            rolling_window = ticket_history.loc[
                :current_date - timedelta(days=1)
            ].tail(7)
            lag_features.update({
                'sales_rolling_avg_7': round(rolling_window.mean(), 2)
                if not rolling_window.empty else 0.0,
                'sales_rolling_std_7': round(rolling_window.std(), 2)
                if not rolling_window.empty else 0.0,
            })

            # G. Assemble final row and predict
            input_row = {
                **date_features,
                **weather_features,
                **lag_features,
                **campaign_features,
                **holiday_one_hot,
                **event_features,
                f'ticket_{ticket_name}': 1
            }
            input_df = pd.DataFrame([input_row]).reindex(
                columns=model_features,
                fill_value=0
            )
            
            predicted_sales = model.predict(input_df)[0]
            final_prediction = max(0, round(predicted_sales))
            
            if final_prediction > 0:
                all_predictions.append({
                    "date": current_date,
                    "ticket_name": ticket_name,
                    "predicted_sales": final_prediction
                })
            
            loop_history_df = pd.concat([
                loop_history_df,
                pd.DataFrame([{
                    "date": current_date,
                    "ticket_name": ticket_name,
                    "ticket_num": max(0.01, predicted_sales)
                }])
            ], ignore_index=True)


    # --- 6. Save Final Results ---
    print("\nStep 6: Saving final forecast...")
    if all_predictions:
        forecast_df = pd.DataFrame(all_predictions)
        filename = f"forecast_365_days_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        output_path = os.path.join(PREDICTIONS_DIR, filename)
        forecast_df.to_csv(output_path, index=False)
        print(f"✅ Forecast saved successfully to {output_path}")
    else:
        print("WARNING: No positive sales were predicted.")


if __name__ == "__main__":
    predict_next_365_days()
