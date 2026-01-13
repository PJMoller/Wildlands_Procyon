# pip install openpyxl!!!
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from datetime import timedelta
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paths import RAW_DIR, PROCESSED_DIR
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



def process_data():
    # --- Data Loading ---
    try:
        visitor_og_df = pd.read_csv(RAW_DIR / "visitors.csv", sep=";")
    except Exception as e:
        print(f"Error loading visitor data: {e}"); return
    
    try:
        weather_og_df = pd.read_excel(RAW_DIR / "weather.xlsx")
    except Exception as e:
        print(f"Error loading weather data: {e}"); return
    try:
        weather22_23_og_df = pd.read_csv(RAW_DIR / "WeatherData2022_2023.csv", skiprows=3, nrows=17520)
    except Exception as e:
        print(f"Error loading 2022-2023 weather data: {e}"); return
    
    try:
        holiday_og_df = pd.read_excel(RAW_DIR / "Holidays 2022-2026 Netherlands and Germany.xlsx")
    except Exception as e:
        print(f"Error loading holiday data: {e}"); return
    
    try:
        camp_og_df = pd.read_excel(RAW_DIR / "campaigns 2022-2026.xlsx")
    except Exception as e:
        print(f"Error loading campaign data: {e}"); return
    
    try:
        recurring_og_df = pd.read_excel(RAW_DIR / "recurring_events_drenthe.xlsx")
    except Exception as e:
        print(f"Error loading recurring events data: {e}"); return
    
    try:
        ticketfam_og_df = pd.read_excel(RAW_DIR / "ticketfamilies.xlsx")
    except Exception as e:
        print(f"Error loading ticket families data: {e}"); return



    # --- Visitor Data Processing ---
    visitor_og_df.columns = ["date", "groupID", "ticket_name", "ticket_num"]
    visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"])
    visitor_df = visitor_og_df.copy()

    
    print(f"Removed non-visitor tickets. Remaining: {len(visitor_df)} rows")
    
    # CRITICAL FIX 2: Add ticket type indicators from NAMES (since you don't have discount %)
    print("Creating ticket type indicators...")
    ticket_name_lower = visitor_df['ticket_name'].str.lower()
    
    # Add indicators using .loc to avoid SettingWithCopyWarning and ensure persistence
    visitor_df.loc[:, 'is_actie_ticket'] = ticket_name_lower.str.contains('actie|inkoop', na=False).astype(int)
    visitor_df.loc[:, 'is_abonnement_ticket'] = ticket_name_lower.str.contains('abonnement', na=False).astype(int)
    visitor_df.loc[:, 'is_full_price'] = ticket_name_lower.str.contains('vol betalend|volbetalend', na=False).astype(int)
    visitor_df.loc[:, 'is_accommodation_ticket'] = ticket_name_lower.str.contains('accommodatiehouder', na=False).astype(int)
    
    # Group tickets (from your answer: ticket names indicate group packages)
    group_keywords = ['group', 'groep', 'family', 'familie', 'package']
    visitor_df.loc[:, 'is_group_ticket'] = ticket_name_lower.str.contains('|'.join(group_keywords), na=False).astype(int)
    
    # Joint promotions (from your answer)
    visitor_df.loc[:, 'is_joint_promotion'] = ticket_name_lower.str.contains('joint promotion', na=False).astype(int)
    

    # Create ticket family mapping dynamically
    print("Creating ticket family classifications...")
    ###################################
    ticket_families = dict(zip(ticketfam_og_df['Subgroup'], ticketfam_og_df['ticket_family']))
    visitor_df.loc[:, 'ticket_family'] = visitor_df['ticket_name'].map(ticket_families)
    
    # --- Weather Data Processing ---
    weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"]

    weather22_23_og_df.columns = ["date","temperature", "rain", "hum", "precipitation", "snow"]
    weather22_23_og_df = weather22_23_og_df.drop(["hum", "snow"], axis=1)
    weather22_23_og_df["date"] = pd.to_datetime(weather22_23_og_df["date"])
    weather22_23_og_df["hour"] = weather22_23_og_df["date"].dt.hour
    weather22_23_og_df["date"] = weather22_23_og_df["date"].dt.date
    weather_og_df["date"] = pd.to_datetime(weather_og_df["date"]).dt.date
    weather_combined = pd.concat([weather_og_df, weather22_23_og_df],ignore_index=True)

    weather_combined["grouping_date"] = weather_combined["date"]
    weather_combined['rain_morning'] = np.where(weather_combined['hour'] < 12, weather_combined['rain'], 0)
    weather_combined['rain_afternoon'] = np.where(weather_combined['hour'] >= 12, weather_combined['rain'], 0)
    weather_combined['precip_morning'] = np.where(weather_combined['hour'] < 12, weather_combined['precipitation'], 0)
    weather_combined['precip_afternoon'] = np.where(weather_combined['hour'] >= 12, weather_combined['precipitation'], 0)
    weather_daily = weather_combined.groupby("grouping_date").agg(
        temperature=('temperature', 'mean'), 
        rain_morning=('rain_morning', 'sum'), 
        rain_afternoon=('rain_afternoon', 'sum'), 
        precip_morning=('precip_morning', 'sum'), 
        precip_afternoon=('precip_afternoon', 'sum')
    ).reset_index()
    weather_daily.rename(columns={'grouping_date': 'date'}, inplace=True)
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])
    weather_daily = weather_daily.round(1)





    # --- Holiday & Recurring Events Data Processing ---
    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date", "week"]
    holiday_og_df.drop("week", axis=1)
    region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    
    long_df = holiday_og_df.melt(id_vars=["date"], value_vars=region_cols, var_name="region", value_name="holiday")
    long_df["holiday"] = long_df["holiday"].fillna("None").str.strip()
    long_df["holiday"] = long_df["holiday"].replace('', 'None')
    
    # Create enhanced holiday features
    long_df['region_holiday'] = long_df['region'] + '_' + long_df['holiday']

    # One-hot encode ONLY the combined column
    encoded = pd.get_dummies(long_df.set_index('date')['region_holiday'], prefix='', prefix_sep='_')
    final_holiday_df = encoded.groupby('date').sum().reset_index()
    # Add holiday intensity (count of simultaneous holidays)
    holiday_intensity = long_df.groupby('date')['holiday'].apply(lambda x: x.dropna().nunique()).reset_index(name='holiday_intensity')
    final_holiday_df = pd.merge(final_holiday_df, holiday_intensity, on='date', how='left')
    
    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    camp_og_df.rename(columns={"Week ": "week"}, inplace=True, errors='ignore')
    
    # CRITICAL FIX 3: Separate campaign vs promotion features
    promo_cols = [col for col in camp_og_df.columns if col.startswith('promo_')]
    camp_og_df['campaign_strength'] = camp_og_df[promo_cols].sum(axis=1)
    camp_og_df['promotion_active'] = (camp_og_df['campaign_strength'] > 0).astype(int)  # Binary: discount available
    camp_og_df['campaign_regions_active'] = (camp_og_df[promo_cols] > 0).sum(axis=1)
    
    recurring_og_df.columns = ["event_name", "date"]
    recurring_og_df['event_name'] = recurring_og_df['event_name'].fillna('').str.split('/')
    recurring_df = recurring_og_df.explode('event_name')
    recurring_df['event_name'] = recurring_df['event_name'].str.strip().str.replace(' ', '_').str.lower()
    recurring_df['event_name'] = recurring_df['event_name'].str.replace(pat=r'^fc_emmen_.*', repl='soccer', regex=True)
    recurring_df.loc[recurring_df['event_name'] == '', 'event_name'] = 'no_event'
    recurring_df['date'] = pd.to_datetime(recurring_df['date'])
    recurring_df = recurring_df.drop_duplicates(subset=['date', 'event_name'])


    weather_final = weather_daily.copy()
    events_pivot = recurring_df.pivot_table(index='date', columns='event_name', aggfunc='size', fill_value=0).astype(int)
    events_pivot.columns = [f'event_{col}' for col in events_pivot.columns]
    events_pivot = events_pivot.reset_index()

    # Keep original event_name for reference
    events_combined = events_pivot.copy()
    camp_combined = camp_og_df[['year', 'week', 'campaign_strength', 'promotion_active', 'campaign_regions_active']].copy()


    # --- Data Expansion ---
    print("Expanding data to include zero-sale days...")
    # CRITICAL: Create full 2022-2025 date range using last sales date
    last_sales_date = visitor_df['date'].max()
    first_sales_date = visitor_df['date'].min()
    # earlier days are missing a lot of data since we have nothing for them or completely 0's
    all_dates = pd.date_range(start=first_sales_date, end=last_sales_date, freq='D')
    all_tickets = visitor_df['ticket_name'].unique()
    
    print(f"Creating {len(all_dates)} dates × {len(all_tickets)} tickets = {len(all_dates) * len(all_tickets):,} rows")
    
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickets], names=['date', 'ticket_name'])
    expanded_df = pd.DataFrame(index=multi_index).reset_index()
    
    # Do NOT include ticket_family - it's added via mapping
    merge_cols = ['date', 'ticket_name', 'ticket_num', 'groupID', 
                  'is_actie_ticket', 'is_abonnement_ticket', 'is_full_price', 
                  'is_accommodation_ticket', 'is_group_ticket', 'is_joint_promotion']
    
    # MERGE with indicator columns preserved
    expanded_df = pd.merge(expanded_df, visitor_df[merge_cols], on=['date', 'ticket_name'], how='left')
    expanded_df['ticket_num'] = expanded_df['ticket_num'].fillna(0).astype(int)
    
    # Add family information AFTER merge (not in merge_cols)
    expanded_df['ticket_family'] = expanded_df['ticket_name'].map(ticket_families)
    
    # Find available episodes (when tickets were actively sold)
    g = visitor_og_df[visitor_og_df['ticket_num'] > 0].copy()
    g = g.sort_values(['ticket_name', 'date'])
    
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
    
    print(f"Zero-sale days created: {len(expanded_df)} rows")






    # --- Merge with COMBINED weather, holidays, campaigns, events ---
    print("Merging with all external data...")
    merged_df = expanded_df.copy()
    merged_df = pd.merge(merged_df, weather_final, on="date", how="left")
    merged_df = pd.merge(merged_df, final_holiday_df, on="date", how="left")
    merged_df = pd.merge(merged_df, events_combined, on="date", how="left")
    
    # CRITICAL FIX: Add year/week columns BEFORE campaign merge
    merged_df["year"] = merged_df["date"].dt.year
    merged_df["week"] = merged_df["date"].dt.isocalendar().week
    
    # Merge campaigns on year/week
    merged_df = pd.merge(merged_df, camp_combined, on=["year", "week"], how="left")

    # CRITICAL FIX: Add weekday, day, and is_weekend immediately after all merges
    merged_df["weekday"] = merged_df["date"].dt.weekday
    merged_df["day"] = merged_df["date"].dt.day
    merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)

    # Get unique ticket indicators from visitor_df
    ticket_indicators = visitor_df[['ticket_name', 'is_actie_ticket', 'is_abonnement_ticket', 
                                   'is_full_price', 'is_accommodation_ticket', 'is_group_ticket', 
                                   'is_joint_promotion', 'groupID']].drop_duplicates()
    
    # Map indicators to ALL rows in merged_df based on ticket_name
    for col in ['groupID', 'is_actie_ticket', 'is_abonnement_ticket', 'is_full_price', 
                'is_accommodation_ticket', 'is_group_ticket', 'is_joint_promotion']:
        merged_df[col] = merged_df['ticket_name'].map(
            ticket_indicators.set_index('ticket_name')[col]
        )
        # Fill NaN values (for tickets that might not exist in visitor_df)
        if col == 'groupID':
            merged_df[col] = merged_df[col].fillna(-1).astype(int)
        else:
            merged_df[col] = merged_df[col].fillna(0).astype(int)

    # Fill weather NaNs
    weather_cols = ['temperature', 'rain_morning', 'rain_afternoon', 
                   'precip_morning', 'precip_afternoon']
    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    print(f"Merge complete: {len(merged_df)} rows")




    # --- FEATURE ENGINEERING ---
    print("Creating advanced time-based and interaction features...")
    
    # Basic Date Features (year, week, weekday, day already added above)
    merged_df["month"] = merged_df["date"].dt.month
    merged_df['day_of_year'] = merged_df['date'].dt.dayofyear
    
    # Cyclical Features
    merged_df['day_of_year_sin'] = np.sin(2 * np.pi * merged_df['day_of_year'] / 365.25)
    merged_df['day_of_year_cos'] = np.cos(2 * np.pi * merged_df['day_of_year'] / 365.25)
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)
    
    # Payday Features
    merged_df['is_month_start'] = merged_df['day'].isin([1, 2, 3]).astype(int)
    merged_df['is_month_mid'] = merged_df['day'].isin([14, 15, 16]).astype(int)
    merged_df['is_month_end'] = merged_df['date'].dt.is_month_end.astype(int)
    
    # Holiday Proximity & Peak Season Flags
    christmas_date = pd.to_datetime(merged_df['year'].astype(str) + '-12-25')
    merged_df['days_until_christmas'] = (christmas_date - merged_df['date']).dt.days
    merged_df['days_until_christmas'] = merged_df['days_until_christmas'].apply(lambda x: max(0, x))
    merged_df['is_christmas_build_up'] = ((merged_df['month'] == 12) & (merged_df['day'].between(1, 24))).astype(int)
    merged_df['is_peak_christmas_week'] = ((merged_df['month'] == 12) & (merged_df['day'].between(18, 24))).astype(int)
    merged_df['is_twixtmas_period'] = ((merged_df['month'] == 12) & (merged_df['day'].between(26, 30))).astype(int)
    merged_df['is_kings_day'] = ((merged_df['month'] == 4) & (merged_df['day'] == 27)).astype(int)
    
    # Weather Interactions
    merged_df['temp_x_weekend'] = merged_df['temperature'] * merged_df['is_weekend']
    merged_df['is_perfect_day'] = ((merged_df['temperature'] > 20) & (merged_df['rain_morning'] == 0) & (merged_df['rain_afternoon'] == 0)).astype(int)
    
    # CRITICAL FIX: Interaction features for ticket types
    merged_df['campaign_x_actie'] = merged_df['campaign_strength'] * merged_df['is_actie_ticket']
    merged_df['promotion_x_actie'] = merged_df['promotion_active'] * merged_df['is_actie_ticket']
    merged_df['weekend_x_group'] = merged_df['is_weekend'] * merged_df['is_group_ticket']
    
    # Enhanced Lag and Rolling Features (multi-scale)
    print("Creating multi-scale lag and rolling features...")
    merged_df = merged_df.sort_values(['ticket_name', 'date'])
    group = merged_df.groupby('ticket_name')['ticket_num']
    """
    # Uncomment if you want the higher trend features
    # This will have a higher R2 score (98.5%) but ignores many real-world factors like rain and holidays only predicts based on current trend
    # Multi-scale lags
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        merged_df[f'sales_lag_{lag}'] = group.shift(lag)
    
    # Multiple rolling windows
    for window in [7, 14, 30]:
        merged_df[f'sales_rolling_avg_{window}'] = group.rolling(window).mean().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_std_{window}'] = group.rolling(window).std().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_min_{window}'] = group.rolling(window).min().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_max_{window}'] = group.rolling(window).max().reset_index(level=0, drop=True)
    
    # Add Momentum Features
    merged_df['sales_momentum_7d'] = merged_df['sales_lag_1'] - merged_df['sales_lag_7']
    merged_df['sales_trend_30d'] = merged_df['sales_lag_1'] - merged_df['sales_lag_30']
    """
    # Fill NaN values
    lag_cols = [col for col in merged_df.columns if 'sales_lag_' in col or 'sales_rolling_' in col]
    merged_df[lag_cols] = merged_df[lag_cols].fillna(0)
    
    # Holiday Features (enhanced)
    holiday_dates = pd.to_datetime(pd.Series(holiday_og_df['date'].dropna().unique())).sort_values()
    merged_df = merged_df.sort_values('date')
    
    # Next and previous holiday
    next_holidays = pd.merge_asof(merged_df[['date']], pd.DataFrame({'holiday_date': holiday_dates}), left_on='date', right_on='holiday_date', direction='forward')
    prev_holidays = pd.merge_asof(merged_df[['date']], pd.DataFrame({'holiday_date': holiday_dates}), left_on='date', right_on='holiday_date', direction='backward')
    merged_df['days_until_holiday'] = (next_holidays['holiday_date'] - merged_df['date']).dt.days
    merged_df['days_since_holiday'] = (merged_df['date'] - prev_holidays['holiday_date']).dt.days
    
    
    
    # weather holiday interactions
    merged_df['temp_x_holiday_intensity'] = merged_df['temperature'] * merged_df['holiday_intensity']





    # Add Availability Features
    def calculate_days_since_available(group):
        if group.iloc[0] == 1:
            return group.cumsum()
        else:
            return pd.Series(np.zeros(len(group), dtype=int), index=group.index)





    merged_df['days_since_available'] = merged_df.groupby('ticket_name')['is_available'].apply(
        calculate_days_since_available
    ).reset_index(level=0, drop=True)
    """
    # Uncomment if you want the higher trend features
    # This will have a higher R2 score (98.5%) but ignores many real-world factors like rain and holidays only predicts based on current trend
    # Add Family-Level Features
    family_group = merged_df.groupby(['date', 'ticket_family'])['ticket_num'].sum().reset_index()
    family_pivot = family_group.pivot(index='date', columns='ticket_family', values='ticket_num').fillna(0)
    family_pivot.columns = [f'family_{col}_sales' for col in family_pivot.columns]
    merged_df = pd.merge(merged_df, family_pivot, on='date', how='left')
    """
    # Drop rows with critical missing values
    merged_df.dropna(subset=['ticket_num', 'date', 'ticket_name'], inplace=True)
    
    print("Removing Wildlands requested ticket types/families")
    merged_df = merged_df[~merged_df['ticket_name'].str.contains('Inkoop overig|Inkoop E-Tickets', na=False)]
    merged_df = merged_df[~merged_df['ticket_family'].isin(['group_package', 'b2b'])]
    merged_df.drop(columns=[col for col in merged_df.columns if 'group_package' in col or 'b2b' in col], inplace=True)

    # --- One-Hot Encoding for Categorical Variables ---
    print("One-hot encoding categorical variables...")
    
    # Ticket type one-hot encoding
    ticket_dummies = pd.get_dummies(merged_df['ticket_name'], prefix='ticket', dtype=int)
    merged_df = pd.concat([merged_df, ticket_dummies], axis=1)
    
    # Ticket family one-hot encoding
    family_dummies = pd.get_dummies(merged_df['ticket_family'], prefix='family', dtype=int)
    merged_df = pd.concat([merged_df, family_dummies], axis=1)
    



    # --- Final Cleanup ---
    print("Final cleanup...")

    print("Data processing completed successfully!")
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Ticket types: {merged_df['ticket_name'].nunique()}")
    print(f"Ticket families: {merged_df['ticket_family'].value_counts().to_dict()}")

    # droppping useless columns
    #merged_df = merged_df.drop(["ticket_name","ticket_family"], axis=1)

    merged_df.sort_values(['date'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    # Save processed data
    print("Saving processed data...")
    merged_df.to_csv(PROCESSED_DIR / "processed_merge.csv", index=False)
    




if __name__ == "__main__":
    process_data()