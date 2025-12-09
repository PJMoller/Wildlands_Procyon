# pip install openpyxl!!!
from paths import RAW_DIR, PROCESSED_DIR
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from datetime import timedelta

def classify_ticket_families(ticket_names):
    """
    Dynamically classify tickets into families based on naming patterns
    Returns: dictionary mapping ticket_name -> family_type
    """
    family_mapping = {}
    
    for ticket in ticket_names:
        name_lower = str(ticket).lower()
        
        if any(sub in name_lower for sub in ['subscriber', 'abonnement', 'lidmaatschap']):
            family_mapping[ticket] = 'subscription'
        elif any(sub in name_lower for sub in ['action', 'flex', 'dynamic']):
            family_mapping[ticket] = 'flexible_seasonal'
        elif any(sub in name_lower for sub in ['season', 'seizoen', 'winter', 'summer', 'lente', 'herfst']):
            family_mapping[ticket] = 'fixed_seasonal'
        elif any(sub in name_lower for sub in ['day', 'dag', 'single', 'enkel']):
            family_mapping[ticket] = 'single_day'
        elif any(sub in name_lower for sub in ['group', 'groep', 'family', 'familie']):
            family_mapping[ticket] = 'group_package'
        else:
            family_mapping[ticket] = 'general'
    
    return family_mapping



def create_ticket_lifecycle_features(df, ticket_col='ticket_name', date_col='date', sales_col='ticket_num'):
    """
    Create lifecycle features for each ticket type
    """
    df = df.sort_values([ticket_col, date_col])
    
    lifecycle = df.groupby(ticket_col).agg({
        date_col: ['min', 'max'],
        sales_col: ['mean', 'std', 'max']
    }).round(3)
    
    lifecycle.columns = ['first_sale_date', 'last_sale_date', 'avg_sales', 'sales_std', 'max_sales']
    lifecycle['ticket_lifespan_days'] = (lifecycle['last_sale_date'] - lifecycle['first_sale_date']).dt.days
    lifecycle['sales_cv'] = lifecycle['sales_std'] / lifecycle['avg_sales']
    
    return lifecycle


def process_data():
    # --- Data Loading ---
    try:
        visitor_og_df = pd.read_csv(RAW_DIR / "visitordaily.csv", sep=";")
    except Exception as e:
        print(f"Error loading visitor data: {e}"); return
    
    try:
        weather_og_df = pd.read_excel(RAW_DIR / "weather.xlsx")
    except Exception as e:
        print(f"Error loading weather data: {e}"); return
    
    try:
        holiday_og_df = pd.read_excel(RAW_DIR / "Holidays 2023-2026 Netherlands and Germany.xlsx")
    except Exception as e:
        print(f"Error loading holiday data: {e}"); return
    
    try:
        camp_og_df = pd.read_excel(RAW_DIR / "campaings.xlsx")
    except Exception as e:
        print(f"Error loading campaign data: {e}"); return
    
    try:
        recurring_og_df = pd.read_excel(RAW_DIR / "recurring_events_drenthe.xlsx")
    except Exception as e:
        print(f"Error loading recurring events data: {e}"); return



    # --- Visitor Data Processing ---
    visitor_og_df.columns = ["groupID", "ticket_name", "date", "ticket_num"]
    visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d")
    
    # CRITICAL FIX 1: Filter non-visitor tickets IMMEDIATELY
    print("Filtering out non-visitor tickets...")
    non_visitor_keywords = ['consumptiebon', 'niet meetellen', 'consumptiebonnen']
    visitor_df = visitor_og_df[
        ~visitor_og_df['ticket_name'].str.lower().str.contains('|'.join(non_visitor_keywords), na=False)
    ].copy()
    
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
    ticket_families = classify_ticket_families(visitor_df['ticket_name'].unique())
    visitor_df.loc[:, 'ticket_family'] = visitor_df['ticket_name'].map(ticket_families)
    
    # --- Weather Data Processing ---
    weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"]
    weather_og_df["grouping_date"] = weather_og_df["date"].dt.date
    weather_og_df['rain_morning'] = np.where(weather_og_df['hour'] < 12, weather_og_df['rain'], 0)
    weather_og_df['rain_afternoon'] = np.where(weather_og_df['hour'] >= 12, weather_og_df['rain'], 0)
    weather_og_df['precip_morning'] = np.where(weather_og_df['hour'] < 12, weather_og_df['precipitation'], 0)
    weather_og_df['precip_afternoon'] = np.where(weather_og_df['hour'] >= 12, weather_og_df['precipitation'], 0)
    weather_og_df['date'] = pd.to_datetime(weather_og_df['date']).dt.date
    weather_daily = weather_og_df.groupby("grouping_date").agg(
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
    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    
    long_df = holiday_og_df.melt(id_vars=["date"], value_vars=region_cols, var_name="region", value_name="holiday")
    long_df["holiday"] = long_df["holiday"].fillna("None").str.strip()
    long_df["holiday"] = long_df["holiday"].replace('', 'None')
    
    # Create enhanced holiday features
    encoded = pd.get_dummies(long_df.set_index('date')['holiday'], prefix='holiday').groupby('date').max()
    final_holiday_df = encoded.reset_index()
    
    # Add holiday intensity (count of simultaneous holidays)
    holiday_intensity = long_df.groupby('date').size().reset_index(name='holiday_intensity')
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






    # --- Data Expansion ---
    print("Expanding data to include zero-sale days...")
    # CRITICAL: Create full 2023-2025 date range using last sales date
    last_sales_date = visitor_df['date'].max()
    all_dates = pd.date_range(start='2023-01-01', end=last_sales_date, freq='D')
    all_tickets = visitor_df['ticket_name'].unique()
    
    print(f"Creating {len(all_dates)} dates × {len(all_tickets)} tickets = {len(all_dates) * len(all_tickets):,} rows")
    
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickets], names=['date', 'ticket_name'])
    expanded_df = pd.DataFrame(index=multi_index).reset_index()
    
    # CRITICAL FIX: Define EXACT columns to merge (including indicators)
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






    # --- Extract 2024 weather for synthetic 2023 ---
    print("Extracting 2024 weather data for synthetic 2023...")

    # Get 2024 weather data
    weather_2024 = weather_daily[
        weather_daily['date'].dt.year == 2024
    ].copy()

    # Shift it to 2023
    weather_2023 = weather_2024.copy()
    weather_2023['date'] = weather_2023['date'] - pd.DateOffset(years=1)

    # Add noise to weather (optional but recommended)
    weather_noise_cols = ['temperature', 'rain_morning', 'rain_afternoon', 
                         'precip_morning', 'precip_afternoon']
    for col in weather_noise_cols:
        noise = np.random.normal(0, 0.12, len(weather_2023))  # 12% noise
        weather_2023[col] = weather_2023[col] * (1 + noise)
        if 'rain' in col or 'precip' in col:
            weather_2023[col] = np.maximum(0, weather_2023[col])
        weather_2023[col] = weather_2023[col].round(1)

    # Create combined weather dataset (2023 synthetic + 2024-2025 real)
    weather_combined = pd.concat([
        weather_2023,
        weather_daily[weather_daily['date'].dt.year >= 2024]
    ], ignore_index=True).drop_duplicates(subset=['date'])

    print(f"Weather combined: {len(weather_combined)} days (2023 synthetic + 2024-2025 real)")






    # --- Extract 2024 events for synthetic 2023 ---
    print("Extracting 2024 events for synthetic 2023...")

    # Get 2024 events
    events_2024 = recurring_df[recurring_df['date'].dt.year == 2024].copy()
    events_2023 = events_2024.copy()
    events_2023['date'] = events_2023['date'] - pd.DateOffset(years=1)

    # Create combined events dataset (2023 synthetic + 2024-2025 real)
    events_combined = pd.concat([
        events_2023,
        recurring_df[recurring_df['date'].dt.year >= 2024]
    ], ignore_index=True).drop_duplicates(subset=['date', 'event_name'])

    print(f"Events combined: {len(events_combined)} days (2023 synthetic + 2024-2025 real)")






    # --- Extract 2024 campaigns for synthetic 2023 ---
    print("Extracting 2024 campaigns for synthetic 2023...")
    
    # Get 2024 campaigns
    camp_2024 = camp_og_df[camp_og_df['year'] == 2024].copy()
    camp_2023 = camp_2024.copy()
    camp_2023['year'] = 2023  # Shift year to 2023
    
    # Recalculate features for 2023
    promo_cols = [col for col in camp_og_df.columns if col.startswith('promo_')]
    camp_2023['campaign_strength'] = camp_2023[promo_cols].sum(axis=1)
    camp_2023['promotion_active'] = (camp_2023['campaign_strength'] > 0).astype(int)
    camp_2023['campaign_regions_active'] = (camp_2023[promo_cols] > 0).sum(axis=1)
    
    # Create combined campaign dataset (2023 synthetic + 2024-2025 real)
    camp_combined = pd.concat([
        camp_2023,
        camp_og_df[camp_og_df['year'] >= 2024]
    ], ignore_index=True).drop_duplicates(subset=['year', 'week'])

    print(f"Campaigns combined: {len(camp_combined)} rows (2023 synthetic + 2024-2025 real)")



    # --- Merge with COMBINED weather, holidays, campaigns, events ---
    print("Merging with all external data (including synthetic 2023)...")
    merged_df = pd.merge(merged_df, weather_combined, on="date", how="left")
    merged_df = pd.merge(merged_df, final_holiday_df, on="date", how="left")
    merged_df = pd.merge(merged_df, events_combined, on="date", how="left")
    merged_df['event_name'] = merged_df['event_name'].fillna('no_event')
    
    # CRITICAL FIX: Add year/week columns BEFORE campaign merge
    merged_df["year"] = merged_df["date"].dt.year
    merged_df["week"] = merged_df["date"].dt.isocalendar().week
    
    # Merge campaigns on year/week
    merged_df = pd.merge(merged_df, camp_combined, on=["year", "week"], how="left")

    # CRITICAL FIX: Add weekday, day, and is_weekend immediately after all merges
    merged_df["weekday"] = merged_df["date"].dt.weekday
    merged_df["day"] = merged_df["date"].dt.day
    merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)

    # CRITICAL FIX: Ensure ticket indicators are filled for ALL rows (including synthetic 2023)
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



    # --- Create ticket lifecycle features ---
    print("Creating ticket lifecycle features...")
    lifecycle_features = create_ticket_lifecycle_features(merged_df)
    merged_df = pd.merge(
        merged_df, 
        lifecycle_features[['avg_sales', 'sales_cv', 'ticket_lifespan_days']], 
        left_on='ticket_name', 
        right_index=True, 
        how='left'
    )



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
    
    # Multi-scale lags
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        merged_df[f'sales_lag_{lag}'] = group.shift(lag)
    
    # Multiple rolling windows
    for window in [7, 14, 30]:
        merged_df[f'sales_rolling_avg_{window}'] = group.rolling(window).mean().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_std_{window}'] = group.rolling(window).std().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_min_{window}'] = group.rolling(window).min().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_max_{window}'] = group.rolling(window).max().reset_index(level=0, drop=True)
    
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
    
    # Add Momentum Features
    merged_df['sales_momentum_7d'] = merged_df['sales_lag_1'] - merged_df['sales_lag_7']
    merged_df['sales_trend_30d'] = merged_df['sales_lag_1'] - merged_df['sales_lag_30']
    
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
    
    # Add Family-Level Features
    family_group = merged_df.groupby(['date', 'ticket_family'])['ticket_num'].sum().reset_index()
    family_pivot = family_group.pivot(index='date', columns='ticket_family', values='ticket_num').fillna(0)
    family_pivot.columns = [f'family_{col}_sales' for col in family_pivot.columns]
    merged_df = pd.merge(merged_df, family_pivot, on='date', how='left')
    
    # Drop rows with critical missing values
    merged_df.dropna(subset=['ticket_num', 'date', 'ticket_name'], inplace=True)
    
    # --- One-Hot Encoding for Categorical Variables ---
    print("One-hot encoding categorical variables...")
    event_dummies = pd.get_dummies(merged_df['event_name'], prefix='event')
    merged_df = pd.concat([merged_df, event_dummies], axis=1)
    
    # Ticket type one-hot encoding
    ticket_dummies = pd.get_dummies(merged_df['ticket_name'], prefix='ticket')
    merged_df = pd.concat([merged_df, ticket_dummies], axis=1)
    
    # Ticket family one-hot encoding
    family_dummies = pd.get_dummies(merged_df['ticket_family'], prefix='family')
    merged_df = pd.concat([merged_df, family_dummies], axis=1)
    
    # --- Final Cleanup ---
    print("Final cleanup...")

    print("Data processing completed successfully!")
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Ticket types: {merged_df['ticket_name'].nunique()}")
    print(f"Ticket families: {merged_df['ticket_family'].value_counts().to_dict()}")

    merged_df.sort_values(['ticket_name', 'date'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    print(merged_df[merged_df['ticket_family'] == 'fixed_seasonal']['ticket_num'].describe())
    # Save processed data
    print("Saving processed data...")
    merged_df.to_csv("../data/processed/processed_merge.csv", index=False)
    
    # Save ticket family mapping for later use
    pd.DataFrame(list(ticket_families.items()), columns=['ticket_name', 'ticket_family']).to_csv(
        "../data/processed/ticket_families.csv", index=False
    )
    
    # Save lifecycle features
    lifecycle_features.to_csv("../data/processed/ticket_lifecycle.csv")
    

    
    # DEBUG: Verify years in final data
    print(f"\nYears in final dataset: {sorted(merged_df['date'].dt.year.unique())}")
    year_counts = merged_df['date'].dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} rows")





if __name__ == "__main__":
    process_data()
