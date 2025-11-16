import pandas as pd
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
RECURRING_EVENTS_PATH = "data/raw/recurring_events_drenthe.xlsx"
PREDICTIONS_DIR = "data/predictions/"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def get_openmeteo_for_future(days=16):
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.7862, "longitude": 6.8917,
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
        "precipitation": hourly.Variables(2).ValuesAsNumpy(),
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
    try:
        with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
        holiday_og_df = pd.read_excel(HOLIDAY_DATA_PATH)
        camp_og_df = pd.read_excel(CAMPAIGN_DATA_PATH)
        recurring_og_df = pd.read_excel(RECURRING_EVENTS_PATH)
    except Exception as e:
        print(f"CRITICAL ERROR loading initial files: {e}"); return

    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    holiday_region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    all_holidays_lookup = holiday_og_df.melt(id_vars=['date'], value_vars=holiday_region_cols, var_name='region', value_name='holiday').dropna()
    all_holidays_lookup['holiday'] = all_holidays_lookup['holiday'].str.strip().replace('', 'None')
    holiday_dates = pd.to_datetime(pd.Series(all_holidays_lookup['date'].unique())).sort_values()
    summer_dates = all_holidays_lookup[all_holidays_lookup['holiday'].str.contains("Zomervakantie", na=False)]['date'].unique()
    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    recurring_og_df.columns = ["event_name", "date"]
    recurring_og_df['event_name'] = recurring_og_df['event_name'].fillna('')
    recurring_og_df['event_name'] = recurring_og_df['event_name'].str.split('/')
    recurring_df_lookup = recurring_og_df.explode('event_name')
    recurring_df_lookup['event_name'] = recurring_df_lookup['event_name'].str.strip().str.replace(' ', '_').str.lower()
    recurring_df_lookup['event_name'] = recurring_df_lookup['event_name'].str.replace(pat=r'^fc_emmen_.*', repl='soccer', regex=True)
    recurring_df_lookup.loc[recurring_df_lookup['event_name'] == '', 'event_name'] = 'no_event'
    recurring_df_lookup['date'] = pd.to_datetime(recurring_df_lookup['date'])
    
    model_features = model.feature_name_
    event_type_cols = [col for col in model_features if col.startswith('event_')]

    with open(PROCESSED_DATA_PATH, 'rb') as f:
        try: f.seek(-2, os.SEEK_END)
        except OSError: f.seek(0)
        while f.read(1) != b'\n': f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    last_year, last_month, last_day = map(int, last_line.split(',')[:3])
    last_known_date = pd.to_datetime(f"{last_year}-{last_month}-{last_day}")

    print("Preparing historical data for iterative prediction...")

    csv_cols = pd.read_csv(PROCESSED_DATA_PATH, nrows=0).columns.tolist()
    ticket_type_cols_from_csv = [col for col in csv_cols if col.startswith('ticket_')and col != 'ticket_num']
    all_ticket_names = [col.replace('ticket_', '') for col in ticket_type_cols_from_csv]
    print(f"Found {len(ticket_type_cols_from_csv)} ticket type columns.")

    cols_to_load = ['year', 'month', 'day', 'ticket_num'] + ticket_type_cols_from_csv
    raw_history_df = pd.read_csv(PROCESSED_DATA_PATH, usecols=cols_to_load)
    raw_history_df['date'] = pd.to_datetime(raw_history_df[['year', 'month', 'day']])
    raw_history_df = raw_history_df[raw_history_df['date'] >= last_known_date - timedelta(days=30)]
    melted_df = raw_history_df.melt(
        id_vars=['date', 'ticket_num'],
        value_vars=ticket_type_cols_from_csv,
        var_name='ticket_name_encoded',
        value_name='is_present'
    )
    active_tickets_df = melted_df[melted_df['is_present'] == 1].copy()

    active_tickets_df['ticket_name'] = active_tickets_df['ticket_name_encoded'].str.replace('ticket_', '', regex=False)

    history_df = active_tickets_df[['date', 'ticket_name', 'ticket_num']].copy()

    # historyical weather averages for fallback
    weather_cols_to_load = ['month', 'day', 'temperature', 'rain_morning', 'rain_afternoon', 'precip_morning', 'precip_afternoon']
    historical_weather_avg = pd.read_csv(PROCESSED_DATA_PATH, usecols=weather_cols_to_load).groupby(['month', 'day']).mean().reset_index()
    weather_future_df = get_openmeteo_for_future(16)
    
    # history
    cols_to_load = ['year', 'month', 'day', 'ticket_num'] + ticket_type_cols_from_csv
    history_df = pd.read_csv(PROCESSED_DATA_PATH, usecols=cols_to_load)
    history_df['date'] = pd.to_datetime(history_df[['year', 'month', 'day']])
    history_df = history_df[history_df['date'] >= last_known_date - timedelta(days=30)]
    history_df = pd.melt(history_df, id_vars=['date', 'ticket_num'], value_vars=ticket_type_cols_from_csv, var_name='ticket_name_encoded', value_name='is_present')
    history_df = history_df[history_df['is_present'] == 1]
    history_df['ticket_name'] = history_df['ticket_name_encoded'].str.replace('ticket_', '', regex=False)
    history_df = history_df[["date", "ticket_name", "ticket_num"]]
    
    # year long prediction loop
    all_predictions = []
    loop_history_df = history_df.copy()

    for d in range(365):
        current_date = last_known_date + timedelta(days=d + 1)
        print(f"Predicting for: {current_date.date()}...")

        # date features
        date_features = {'year': current_date.year, 'month': current_date.month, 'day': current_date.day, 'weekday': current_date.weekday(), 'week': current_date.isocalendar().week}
        date_features['is_weekend'] = 1 if date_features['weekday'] >= 5 else 0
        date_features['is_month_start'] = 1 if date_features['day'] in [1, 2, 3] else 0
        date_features['is_month_mid'] = 1 if date_features['day'] in [14, 15, 16] else 0
        date_features['is_month_end'] = 1 if current_date.is_month_end else 0

        # weather feature with fallback
        weather_features = {}
        weather_match = weather_future_df[weather_future_df['date'].dt.date == current_date.date()]
        if not weather_match.empty:
            # Use real forecast for the first 16 days
            weather_features = weather_match.drop(columns=['date']).to_dict(orient='records')[0]
        else:
            # Fallback to historical average for day 17 to 365
            avg_match = historical_weather_avg[(historical_weather_avg['month'] == current_date.month) & (historical_weather_avg['day'] == current_date.day)]
            if not avg_match.empty:
                weather_features = avg_match.drop(columns=['month', 'day']).to_dict(orient='records')[0]
            else: # Ultimate fallback if a day like Feb 29 has no historical average
                weather_features = {'temperature': 10.0, 'rain_morning': 0, 'rain_afternoon': 0, 'precip_morning': 0, 'precip_afternoon': 0}
        
        # holiday features
        next_h_series = holiday_dates[holiday_dates >= current_date]
        next_h = next_h_series.iloc[0] if not next_h_series.empty else current_date + timedelta(days=90)
        prev_h_series = holiday_dates[holiday_dates <= current_date]
        prev_h = prev_h_series.iloc[-1] if not prev_h_series.empty else current_date - timedelta(days=90)
        date_features['days_until_holiday'] = (next_h - current_date).days
        date_features['days_since_holiday'] = (current_date - prev_h).days
        date_features['is_summer_vacation'] = 1 if current_date in summer_dates else 0
        date_features['temp_x_weekend'] = weather_features['temperature'] * date_features['is_weekend']
        holiday_features_today = all_holidays_lookup[all_holidays_lookup['date'] == current_date]
        holiday_one_hot = {f"{row['region']}_{row['holiday']}": 1 for _, row in holiday_features_today.iterrows()}

        # campaign features
        campaign_features = {}
        campaign_today = camp_og_df[(camp_og_df['year'] == date_features['year']) & (camp_og_df['week'] == date_features['week'])]
        if not campaign_today.empty:
            promo_cols = [col for col in camp_og_df.columns if col.startswith('promo_')]
            for col in promo_cols: campaign_features[col] = campaign_today[col].iloc[0]
        
        # event features
        event_features = {event_col: 0 for event_col in event_type_cols}
        events_today = recurring_df_lookup[recurring_df_lookup['date'].dt.date == current_date.date()]
        if not events_today.empty:
            for event in events_today['event_name']:
                feature_name = f'event_{event}'
                if feature_name in event_features: event_features[feature_name] = 1
        elif 'event_no_event' in event_features: event_features['event_no_event'] = 1

        for ticket_name in all_ticket_names:
            # lag / rolling features
            ticket_history = loop_history_df[loop_history_df["ticket_name"] == ticket_name].set_index('date')['ticket_num']
            lag_features = {}
            lags = [1, 7, 14]
            for lag in lags: lag_features[f'sales_lag_{lag}'] = ticket_history.get(current_date - timedelta(days=lag), 0)
            rolling_window_7 = ticket_history.loc[:current_date - timedelta(days=1)].tail(7)
            lag_features['sales_rolling_avg_7'] = round(rolling_window_7.mean(), 2) if not rolling_window_7.empty else 0.0
            lag_features['sales_rolling_std_7'] = round(rolling_window_7.std(), 2) if not rolling_window_7.empty else 0.0
            lag_features['sales_same_day_last_week'] = ticket_history.get(current_date - timedelta(days=7), 0)

            # TODO add in seasonality features from seasonality profile
            # to make sure we predict only with the right ticket types or each month



            
            input_row = {**date_features, **weather_features, **lag_features, **campaign_features, **holiday_one_hot, **event_features}
            input_row[f'ticket_{ticket_name}'] = 1

            # make predictions and update the history for the rolling stuff
            input_df = pd.DataFrame([input_row]).reindex(columns=model_features, fill_value=0)
            predicted_sales = model.predict(input_df)[0]
            final_prediction = max(0, round(predicted_sales))
            if final_prediction > 0: all_predictions.append({"date": current_date, "ticket_name": ticket_name, "predicted_sales": final_prediction})
            
            new_history_row = pd.DataFrame([{"date": current_date, "ticket_name": ticket_name, "ticket_num": max(0.01, predicted_sales)}])
            loop_history_df = pd.concat([loop_history_df, new_history_row], ignore_index=True)
            
    # save predictions
    if all_predictions:
        print("Forecast generation complete.")
        forecast_df = pd.DataFrame(all_predictions)
        filename = f"forecast_365_days_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
        output_path = os.path.join(PREDICTIONS_DIR, filename)
        forecast_df.to_csv(output_path, index=False)
        print(f"✅ Forecast saved successfully to {output_path}")
    else:
        print("WARNING: No positive sales were predicted.")

if __name__ == "__main__":
    predict_next_365_days()

