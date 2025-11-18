# pip install openpyxl!!!
import pandas as pd
import pandas.api.types as ptypes
import numpy as np


def process_data():
    # --- Data Loading (unchanged) ---
    try:
        visitor_og_df = pd.read_csv("../data/raw/visitordaily.csv", sep=";")
    except Exception as e:
        print(f"Error loading visitor data: {e}"); return
    try:
        weather_og_df = pd.read_excel("../data/raw/weather.xlsx")
    except Exception as e:
        print(f"Error loading weather data: {e}"); return
    try:
        holiday_og_df = pd.read_excel("../data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx")
    except Exception as e:
        print(f"Error loading holiday data: {e}"); return
    try:
        camp_og_df = pd.read_excel("../data/raw/campaings.xlsx")
    except Exception as e:
        print(f"Error loading campaign data: {e}"); return
    try:
        recurring_og_df = pd.read_excel("../data/raw/recurring_events_drenthe.xlsx")
    except Exception as e:
        print(f"Error loading recurring events data: {e}"); return

    # --- Data Validation (unchanged) ---
    # (Your existing data validation checks are good and remain here)


    # --- Visitor Data Processing (unchanged) ---
    visitor_og_df.columns = ["groupID", "ticket_name", "date", "ticket_num"]
    visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d")
    visitor_df = visitor_og_df[['date', 'ticket_name', 'ticket_num']]

    # --- Weather Data Processing (unchanged) ---
    weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"]
    weather_og_df["grouping_date"] = weather_og_df["date"].dt.date
    weather_og_df['rain_morning'] = np.where(weather_og_df['hour'] < 12, weather_og_df['rain'], 0)
    weather_og_df['rain_afternoon'] = np.where(weather_og_df['hour'] >= 12, weather_og_df['rain'], 0)
    weather_og_df['precip_morning'] = np.where(weather_og_df['hour'] < 12, weather_og_df['precipitation'], 0)
    weather_og_df['precip_afternoon'] = np.where(weather_og_df['hour'] >= 12, weather_og_df['precipitation'], 0)
    weather_daily = weather_og_df.groupby("grouping_date").agg(temperature=('temperature', 'mean'), rain_morning=('rain_morning', 'sum'), rain_afternoon=('rain_afternoon', 'sum'), precip_morning=('precip_morning', 'sum'), precip_afternoon=('precip_afternoon', 'sum')).reset_index()
    weather_daily.rename(columns={'grouping_date': 'date'}, inplace=True)
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])
    weather_daily = weather_daily.round(1)

    # --- Holiday & Recurring Events Data Processing (unchanged) ---
    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    long_df = holiday_og_df.melt(id_vars=["date"], value_vars=region_cols, var_name="region", value_name="holiday")
    long_df["holiday"] = long_df["holiday"].fillna("None").str.strip()
    long_df["holiday"] = long_df["holiday"].replace('', 'None')
    encoded = pd.get_dummies(long_df.set_index('date')['holiday'], prefix='holiday').groupby('date').max()
    final_holiday_df = encoded.reset_index()

    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    camp_og_df.rename(columns={"Week ": "week"}, inplace=True, errors='ignore')

    recurring_og_df.columns = ["event_name", "date"]
    recurring_og_df['event_name'] = recurring_og_df['event_name'].fillna('').str.split('/')
    recurring_df = recurring_og_df.explode('event_name')
    recurring_df['event_name'] = recurring_df['event_name'].str.strip().str.replace(' ', '_').str.lower()
    recurring_df['event_name'] = recurring_df['event_name'].str.replace(pat=r'^fc_emmen_.*', repl='soccer', regex=True)
    recurring_df.loc[recurring_df['event_name'] == '', 'event_name'] = 'no_event'
    recurring_df['date'] = pd.to_datetime(recurring_df['date'])
    recurring_df = recurring_df.drop_duplicates(subset=['date', 'event_name'])

    # --- Data Expansion (unchanged) ---
    print("Expanding data to include zero-sale days...")
    all_dates = pd.date_range(start=visitor_df['date'].min(), end=visitor_df['date'].max(), freq='D')
    all_tickets = visitor_df['ticket_name'].unique()
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickets], names=['date', 'ticket_name'])
    expanded_df = pd.DataFrame(index=multi_index).reset_index()
    expanded_df = pd.merge(expanded_df, visitor_df, on=['date', 'ticket_name'], how='left')
    expanded_df['ticket_num'] = expanded_df['ticket_num'].fillna(0).astype(int)
    # Only days with actual sales
    g = visitor_og_df[visitor_og_df['ticket_num'] > 0].copy()
    g = g.sort_values(['ticket_name', 'date'])

    # Gap > gap_days (e.g. 60) means a new "on" episode
    gap_days = 15
    g['episode_id'] = (
        g.groupby('ticket_name')['date']
        .diff()
        .dt.days
        .gt(gap_days)
        .groupby(g['ticket_name'])
        .cumsum()
    )

    episodes = (
        g.groupby(['ticket_name', 'episode_id'])['date']
        .agg(start_date='min', end_date='max')
        .reset_index()
    )
    expanded_df['is_available'] = 0

    for _, row in episodes.iterrows():
        mask = (
            (expanded_df['ticket_name'] == row['ticket_name']) &
            (expanded_df['date'] >= row['start_date']) &
            (expanded_df['date'] <= row['end_date'])
        )
        expanded_df.loc[mask, 'is_available'] = 1

    print("Zero-sale days created.")

    # --- Initial Merging (unchanged) ---
    daily_features_df = pd.merge(weather_daily, final_holiday_df, on="date", how="inner")
    merged_df = pd.merge(expanded_df, daily_features_df, on="date", how="left")
    merged_df = pd.merge(merged_df, recurring_df, on="date", how="left")
    merged_df['event_name'] = merged_df['event_name'].fillna('no_event')
    merged_df = merged_df[merged_df['date'].dt.year >= 2024].copy()

    # --- NEW FEATURE ENGINEERING SECTION ---
    print("Creating advanced time-based and interaction features...")

    # --- Basic Date Features (mostly unchanged) ---
    merged_df["year"] = merged_df["date"].dt.year
    merged_df["month"] = merged_df["date"].dt.month
    merged_df["week"] = merged_df["date"].dt.isocalendar().week
    merged_df["day"] = merged_df["date"].dt.day
    merged_df["weekday"] = merged_df["date"].dt.weekday
    merged_df['day_of_year'] = merged_df['date'].dt.dayofyear
    merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)
    merged_df = pd.merge(merged_df, camp_og_df, on=["year", "week"], how="left")
    
    # --- NEW FEATURE (Category 2): Cyclical Features ---
    # This helps the model understand that Dec 31 is close to Jan 1
    merged_df['day_of_year_sin'] = np.sin(2 * np.pi * merged_df['day_of_year'] / 365.25)
    merged_df['day_of_year_cos'] = np.cos(2 * np.pi * merged_df['day_of_year'] / 365.25)
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

    # --- NEW FEATURE (Category 2): Payday Features ---
    merged_df['is_month_start'] = merged_df['day'].isin([1, 2, 3]).astype(int)
    merged_df['is_month_mid'] = merged_df['day'].isin([14, 15, 16]).astype(int)
    merged_df['is_month_end'] = merged_df['date'].dt.is_month_end.astype(int)

    # --- NEW FEATURE (Category 1): Holiday Proximity & Peak Season Flags ---
    christmas_date = pd.to_datetime(merged_df['year'].astype(str) + '-12-25')
    merged_df['days_until_christmas'] = (christmas_date - merged_df['date']).dt.days
    merged_df['days_until_christmas'] = merged_df['days_until_christmas'].apply(lambda x: max(0, x)) # Can't be negative

    merged_df['is_christmas_build_up'] = ((merged_df['month'] == 12) & (merged_df['day'].between(1, 24))).astype(int)
    merged_df['is_peak_christmas_week'] = ((merged_df['month'] == 12) & (merged_df['day'].between(18, 24))).astype(int)
    merged_df['is_twixtmas_period'] = ((merged_df['month'] == 12) & (merged_df['day'].between(26, 30))).astype(int)
    merged_df['is_kings_day'] = ((merged_df['month'] == 4) & (merged_df['day'] == 27)).astype(int)
    
    # --- NEW FEATURE (Category 4): Weather Interactions ---
    merged_df['temp_x_weekend'] = merged_df['temperature'] * merged_df['is_weekend']
    merged_df['is_perfect_day'] = ((merged_df['temperature'] > 20) & (merged_df['rain_morning'] == 0) & (merged_df['rain_afternoon'] == 0)).astype(int)
    merged_df = merged_df.sort_values(['ticket_name', 'date'])

    group = merged_df.groupby('ticket_name')['ticket_num']

    merged_df['sales_lag_1'] = group.shift(1)
    merged_df['sales_lag_7'] = group.shift(7)
    merged_df['sales_lag_14'] = group.shift(14)

    merged_df['sales_rolling_avg_7'] = group.rolling(7).mean().reset_index(level=0, drop=True)
    merged_df['sales_rolling_std_7'] = group.rolling(7).std().reset_index(level=0, drop=True)

    merged_df[['sales_lag_1','sales_lag_7','sales_lag_14',
            'sales_rolling_avg_7','sales_rolling_std_7']] = \
        merged_df[['sales_lag_1','sales_lag_7','sales_lag_14',
                'sales_rolling_avg_7','sales_rolling_std_7']].fillna(0)

    # General Holiday Features (Improved from your original)
    holiday_dates = pd.to_datetime(pd.Series(holiday_og_df['date'].dropna().unique())).sort_values()
    merged_df = merged_df.sort_values('date')
    next_holidays = pd.merge_asof(merged_df[['date']], pd.DataFrame({'holiday_date': holiday_dates}), left_on='date', right_on='holiday_date', direction='forward')
    prev_holidays = pd.merge_asof(merged_df[['date']], pd.DataFrame({'holiday_date': holiday_dates}), left_on='date', right_on='holiday_date', direction='backward')
    merged_df['days_until_holiday'] = (next_holidays['holiday_date'] - merged_df['date']).dt.days
    merged_df['days_since_holiday'] = (merged_df['date'] - prev_holidays['holiday_date']).dt.days

    temp_holiday_df = holiday_og_df.melt(id_vars=['date'], value_vars=region_cols, var_name='region', value_name='holiday').dropna()
    summer_dates = temp_holiday_df[temp_holiday_df['holiday'].str.contains("Zomervakantie", na=False)]['date'].unique()
    merged_df['is_summer_vacation'] = merged_df['date'].isin(summer_dates).astype(int)
    print("Advanced features created.")

    # --- Final Processing (unchanged) ---
    print("Finalizing dataframe for training...")
    final_merge = pd.get_dummies(merged_df, columns=["ticket_name", "event_name"], prefix=["ticket", "event"], dtype=int)
    final_merge = final_merge.drop(columns=["date"])
    holiday_cols = [c for c in final_merge.columns if c.startswith("holiday_")]
    final_merge.fillna(0, inplace=True) # Fill any NaNs that may have resulted from merges
    final_merge[holiday_cols] = final_merge[holiday_cols].astype("int8")
    

    print("Optimizing memory by converting to 32-bit floats...")
    for col in final_merge.columns:
        if final_merge[col].dtype == 'float64':
            final_merge[col] = final_merge[col].astype('float32')
        if final_merge[col].dtype == 'int64':
            final_merge[col] = final_merge[col].astype('int32')

    all_cols = final_merge.columns.tolist()
    ordered_cols = ['year', 'month', 'day', 'ticket_num'] + [col for col in all_cols if col not in ['year', 'month', 'day', 'ticket_num']]
    final_df = final_merge[ordered_cols]

    final_df.to_csv("../data/processed/processed_merge.csv", index=False)
    print("✅ Strategic data processing complete. File saved to ../data/processed/processed_merge.csv")


if __name__ == "__main__":
    process_data()
