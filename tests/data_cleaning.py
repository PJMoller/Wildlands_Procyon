# pip install openpyxl!!!
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from datetime import timedelta
from sklearn.utils import resample


def classify_ticket_families(ticket_names):
    """
    Dynamically classify tickets into families based on naming patterns
    Returns: dictionary mapping ticket_name -> family_type
    """
    family_mapping = {}
    
    for ticket in ticket_names:
        name_lower = str(ticket).lower()
        
        # Define pattern-based rules (customize based on your actual ticket names)
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
    
    # Calculate days since first sale and days until last sale
    lifecycle = df.groupby(ticket_col).agg({
        date_col: ['min', 'max'],
        sales_col: ['mean', 'std', 'max']
    }).round(3)
    
    lifecycle.columns = ['first_sale_date', 'last_sale_date', 'avg_sales', 'sales_std', 'max_sales']
    lifecycle['ticket_lifespan_days'] = (lifecycle['last_sale_date'] - lifecycle['first_sale_date']).dt.days
    lifecycle['sales_cv'] = lifecycle['sales_std'] / lifecycle['avg_sales']  # Coefficient of variation
    
    return lifecycle


def generate_synthetic_data(base_df, target_year=2023, noise_level=0.15, n_variations=2):
    """
    SAFE synthetic data generation from 2024 data
    - Shifts dates to target_year (2023)
    - Adds Gaussian noise to numeric features
    - Creates multiple variations per row
    - Preserves temporal patterns but adds variation
    """
    print(f"Generating synthetic {target_year} data from 2024 patterns...")
    
    synthetic_rows = []
    
    # Group by ticket to preserve per-ticket patterns
    for ticket_name, group in base_df.groupby('ticket_name'):
        # Create n variations per row
        for _ in range(n_variations):
            synth_group = group.copy()
            
            # Shift date to target year
            date_shift = synth_group['date'] - pd.DateOffset(years=1)  # 2024 -> 2023
            synth_group['date'] = date_shift
            
            # Add noise to numeric features (preserve mean, add variance)
            numeric_cols = ['temperature', 'rain_morning', 'rain_afternoon', 'precip_morning', 'precip_afternoon']
            for col in numeric_cols:
                if col in synth_group.columns:
                    noise = np.random.normal(0, noise_level, len(synth_group))
                    synth_group[col] = synth_group[col] * (1 + noise)
                    # Keep physical bounds
                    if 'rain' in col or 'precip' in col:
                        synth_group[col] = np.maximum(0, synth_group[col])
            
            # Add noise to ticket_num (sales) - CRITICAL: preserve zeros
            if 'ticket_num' in synth_group.columns:
                # Only add noise to non-zero values
                non_zero_mask = synth_group['ticket_num'] > 0
                noise = np.random.normal(0, noise_level * 0.5, len(synth_group))  # Less noise for sales
                synth_group.loc[non_zero_mask, 'ticket_num'] = (
                    synth_group.loc[non_zero_mask, 'ticket_num'] * (1 + noise[non_zero_mask])
                ).round().astype(int)
                # Ensure non-negative
                synth_group['ticket_num'] = np.maximum(0, synth_group['ticket_num'])
            
            # Shuffle event assignments slightly (20% chance of change)
            if 'event_name' in synth_group.columns:
                unique_events = synth_group['event_name'].unique()
                if len(unique_events) > 1:
                    mask = np.random.random(len(synth_group)) < 0.2  # 20% shuffle
                    synth_group.loc[mask, 'event_name'] = np.random.choice(
                        unique_events, 
                        size=mask.sum()
                    )
            
            synthetic_rows.append(synth_group)
    
    synthetic_df = pd.concat(synthetic_rows, ignore_index=True)
    
    # Remove exact duplicates that might occur
    synthetic_df = synthetic_df.drop_duplicates(subset=['date', 'ticket_name', 'ticket_num'])
    
    print(f"Generated {len(synthetic_df)} synthetic rows for {target_year}")
    return synthetic_df


def process_data():
    # --- Data Loading ---
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


    # --- Visitor Data Processing ---
    visitor_og_df.columns = ["groupID", "ticket_name", "date", "ticket_num"]
    visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d")
    visitor_df = visitor_og_df[['date', 'ticket_name', 'ticket_num', "groupID"]]
    
    # Create ticket family mapping dynamically
    print("Creating ticket family classifications...")
    ticket_families = classify_ticket_families(visitor_df['ticket_name'].unique())
    visitor_df['ticket_family'] = visitor_df['ticket_name'].map(ticket_families)
    
    # --- Weather Data Processing ---
    weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"]
    weather_og_df["grouping_date"] = weather_og_df["date"].dt.date
    weather_og_df['rain_morning'] = np.where(weather_og_df['hour'] < 12, weather_og_df['rain'], 0)
    weather_og_df['rain_afternoon'] = np.where(weather_og_df['hour'] >= 12, weather_og_df['rain'], 0)
    weather_og_df['precip_morning'] = np.where(weather_og_df['hour'] < 12, weather_og_df['precipitation'], 0)
    weather_og_df['precip_afternoon'] = np.where(weather_og_df['hour'] >= 12, weather_og_df['precipitation'], 0)
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
    
    # Calculate campaign strength scores
    promo_cols = [col for col in camp_og_df.columns if col.startswith('promo_')]
    camp_og_df['campaign_strength'] = camp_og_df[promo_cols].sum(axis=1)
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
    all_dates = pd.date_range(start=visitor_df['date'].min(), end=visitor_df['date'].max(), freq='D')
    all_tickets = visitor_df['ticket_name'].unique()
    
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickets], names=['date', 'ticket_name'])
    expanded_df = pd.DataFrame(index=multi_index).reset_index()
    expanded_df = pd.merge(expanded_df, visitor_df, on=['date', 'ticket_name'], how='left')
    expanded_df['ticket_num'] = expanded_df['ticket_num'].fillna(0).astype(int)
    
    # Add family information to expanded df
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
    
    print("Zero-sale days created.")


    # --- SAFE SYNTHETIC DATA GENERATION ---
    print("Creating synthetic 2023 data from 2024 patterns...")
    
    # Filter 2024 data as base
    base_2024 = expanded_df[expanded_df['date'].dt.year == 2024].copy()
    
    if len(base_2024) > 0:
        # Generate synthetic 2023 data
        synthetic_2023 = generate_synthetic_data(
            base_df=base_2024,
            target_year=2023,
            noise_level=0.12,  # 12% noise - moderate variation
            n_variations=2     # 2x variations = 2x data
        )
        
        # Combine real 2024 + synthetic 2023
        expanded_df = pd.concat([
            expanded_df,
            synthetic_2023
        ], ignore_index=True)
        
        print(f"Combined dataset: {len(expanded_df)} rows (2024 real + 2023 synthetic)")
    else:
        print("⚠️  No 2024 data found for synthetic generation")


    # --- Create ticket lifecycle features ---
    print("Creating ticket lifecycle features...")
    lifecycle_features = create_ticket_lifecycle_features(expanded_df)
    expanded_df = pd.merge(
        expanded_df, 
        lifecycle_features[['avg_sales', 'sales_cv', 'ticket_lifespan_days']], 
        left_on='ticket_name', 
        right_index=True, 
        how='left'
    )


    # --- Initial Merging ---
    daily_features_df = pd.merge(weather_daily, final_holiday_df, on="date", how="inner")
    merged_df = pd.merge(expanded_df, daily_features_df, on="date", how="left")
    merged_df = pd.merge(merged_df, recurring_df, on="date", how="left")
    merged_df['event_name'] = merged_df['event_name'].fillna('no_event')
    merged_df = merged_df[merged_df['date'].dt.year >= 2023].copy()  # Keep both 2023 (synthetic) and 2024 (real)


    # --- FEATURE ENGINEERING ---
    print("Creating advanced time-based and interaction features...")
    
    # Basic Date Features
    merged_df["year"] = merged_df["date"].dt.year
    merged_df["month"] = merged_df["date"].dt.month
    merged_df["week"] = merged_df["date"].dt.isocalendar().week
    merged_df["day"] = merged_df["date"].dt.day
    merged_df["weekday"] = merged_df["date"].dt.weekday
    merged_df['day_of_year'] = merged_df['date'].dt.dayofyear
    merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)
    merged_df = pd.merge(merged_df, camp_og_df, on=["year", "week"], how="left")
    
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
    
    # Enhanced Lag and Rolling Features (multi-scale)
    print("Creating multi-scale lag and rolling features...")
    merged_df = merged_df.sort_values(['ticket_name', 'date'])
    group = merged_df.groupby('ticket_name')['ticket_num']
    
    # Multi-scale lags
    for lag in [1, 2, 3, 7, 14, 21, 28, 30, 60, 90]:
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
    
    # Add Availability Features
    def calculate_days_since_available(group):
        if group.iloc[0] == 1:
            # If ticket starts as available, calculate cumulative sum
            return group.cumsum()
        else:
            # If ticket starts as unavailable, return zeros
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
    merged_df.sort_values(['ticket_name', 'date'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # Save processed data
    print("Saving processed data...")
    merged_df.to_csv("../data/processed/processed_merge.csv", index=False)
    
    # Save ticket family mapping for later use
    pd.DataFrame(list(ticket_families.items()), columns=['ticket_name', 'ticket_family']).to_csv(
        "../data/processed/ticket_families.csv", index=False
    )
    
    # Save lifecycle features
    lifecycle_features.to_csv("../data/processed/ticket_lifecycle.csv")
    
    print("Data processing completed successfully!")
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Ticket types: {merged_df['ticket_name'].nunique()}")
    print(f"Ticket families: {merged_df['ticket_family'].value_counts().to_dict()}")


if __name__ == "__main__":
    process_data()
