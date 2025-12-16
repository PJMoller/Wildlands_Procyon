# pip install openpyxl!!!
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from datetime import timedelta
from pathlib import Path
from paths import RAW_DIR, PROCESSED_DIR

# Ensure output directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# HARD-CODED TICKET SUBGROUPS (3 elements: id, name, family)
# -------------------------------------------------------------------
TICKET_SUBGROUPS = [
    {"subgroupId": 30,   "subgroup": "Gratis",                               "ticket_family": "general"},
    {"subgroupId": 1001, "subgroup": "*Vol betalend",                        "ticket_family": "general"},
    {"subgroupId": 1002, "subgroup": "*Scholen WILDLANDS",                   "ticket_family": "group_package"},
    {"subgroupId": 1003, "subgroup": "*Groepen WILDLANDS",                   "ticket_family": "group_package"},
    {"subgroupId": 1007, "subgroup": "*Accomodatiehouders",                  "ticket_family": "b2b"},
    {"subgroupId": 1012, "subgroup": "*Acties Overig",                       "ticket_family": "actions"},
    {"subgroupId": 1013, "subgroup": "*Joint promotions open E-Ticket",     "ticket_family": "actions"},
    {"subgroupId": 1014, "subgroup": "*Joint promotions open Overig",       "ticket_family": "actions"},
    {"subgroupId": 1015, "subgroup": "*Joint promotions gesloten E-Ticket", "ticket_family": "actions"},
    {"subgroupId": 1016, "subgroup": "*Joint promotions gesloten Overig",   "ticket_family": "actions"},
    {"subgroupId": 1019, "subgroup": "*WILDLANDS abonnementen",             "ticket_family": "subscription"},
    {"subgroupId": 1209, "subgroup": "*Entree Evenementen E-tickets",       "ticket_family": "group_package"},
    {"subgroupId": 1272, "subgroup": "*Entree Congres & Events Scholen",    "ticket_family": "group_package"},
    {"subgroupId": 1273, "subgroup": "*Entree Congres & Events Groepen",    "ticket_family": "group_package"},
    {"subgroupId": 1274, "subgroup": "*Entree Congres & Events",            "ticket_family": "group_package"},
    {"subgroupId": 1004, "subgroup": "*Gratis",                             "ticket_family": "general"},
    {"subgroupId": 1005, "subgroup": "*Gratis E-Tickets",                   "ticket_family": "general"},
    {"subgroupId": 1011, "subgroup": "*Acties E-Ticket",                    "ticket_family": "actions"},
    {"subgroupId": 1208, "subgroup": "*Entree Evenementen",                 "ticket_family": "group_package"},
    {"subgroupId": 1226, "subgroup": "*Volbetalend e-tickets",             "ticket_family": "general"},
    {"subgroupId": 1230, "subgroup": "*Accommodatiehouders E-Ticket",      "ticket_family": "b2b"},
    {"subgroupId": 1242, "subgroup": "*Groepen E-Ticket",                  "ticket_family": "group_package"},
]

# Convenience dicts
SUBGROUP_FAMILY_MAP = {d["subgroupId"]: d["ticket_family"] for d in TICKET_SUBGROUPS}
SUBGROUP_NAME_MAP   = {d["subgroupId"]: d["subgroup"]       for d in TICKET_SUBGROUPS}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def classify_ticket_families(ticket_names):
    """
    Dynamic fallback classifier for *unmapped* tickets based on naming patterns.
    """
    family_mapping = {}
    for ticket in ticket_names:
        name_lower = str(ticket).lower()
        if any(sub in name_lower for sub in ['subscriber', 'abonnement', 'lidmaatschap']):
            family_mapping[ticket] = 'subscription'
        elif any(sub in name_lower for sub in ['action', 'actie', 'flex', 'dynamic']):
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
    df = df.sort_values([ticket_col, date_col])
    lifecycle = df.groupby(ticket_col).agg({
        date_col: ['min', 'max'],
        sales_col: ['mean', 'std', 'max']
    }).round(3)
    lifecycle.columns = ['first_sale_date', 'last_sale_date', 'avg_sales', 'sales_std', 'max_sales']
    lifecycle['ticket_lifespan_days'] = (lifecycle['last_sale_date'] - lifecycle['first_sale_date']).dt.days
    lifecycle['sales_cv'] = lifecycle['sales_std'] / lifecycle['avg_sales']
    return lifecycle


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
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
        holiday_og_df = pd.read_excel(RAW_DIR / "Holidays 2022-2026 Netherlands and Germany.xlsx")
    except Exception as e:
        print(f"Error loading holiday data: {e}"); return

    try:
        camp_og_df = pd.read_excel(RAW_DIR / "campaings 2022-2026.xlsx")
    except Exception as e:
        print(f"Error loading campaign data: {e}"); return

    try:
        recurring_og_df = pd.read_excel(RAW_DIR / "recurring_events_drenthe.xlsx")
    except Exception as e:
        print(f"Error loading recurring events data: {e}"); return

    # --- Visitor Data Processing ---
    visitor_og_df.columns = ["groupID", "ticket_name", "date", "ticket_num"]
    visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d")

    print("Filtering out non-visitor tickets...")
    non_visitor_keywords = ['consumptiebon', 'niet meetellen', 'consumptiebonnen']
    visitor_df = visitor_og_df[
        ~visitor_og_df['ticket_name'].str.lower().str.contains('|'.join(non_visitor_keywords), na=False)
    ].copy()
    print(f"Removed non-visitor tickets. Remaining: {len(visitor_df)} rows")

    print("Creating ticket type indicators...")
    ticket_name_lower = visitor_df['ticket_name'].str.lower()
    visitor_df.loc[:, 'is_actie_ticket'] = ticket_name_lower.str.contains('actie|inkoop', na=False).astype(int)
    visitor_df.loc[:, 'is_abonnement_ticket'] = ticket_name_lower.str.contains('abonnement', na=False).astype(int)
    visitor_df.loc[:, 'is_full_price'] = ticket_name_lower.str.contains('vol betalend|volbetalend', na=False).astype(int)
    visitor_df.loc[:, 'is_accommodation_ticket'] = ticket_name_lower.str.contains('accommodatiehouder', na=False).astype(int)
    group_keywords = ['group', 'groep', 'family', 'familie', 'package']
    visitor_df.loc[:, 'is_group_ticket'] = ticket_name_lower.str.contains('|'.join(group_keywords), na=False).astype(int)
    visitor_df.loc[:, 'is_joint_promotion'] = ticket_name_lower.str.contains('joint promotion', na=False).astype(int)

    # --- NEW: ticket_family from hardcoded TICKET_SUBGROUPS (groupID == subgroupId) ---
    print("Assigning ticket_family from hardcoded TICKET_SUBGROUPS...")
    visitor_df['ticket_family'] = visitor_df['groupID'].map(SUBGROUP_FAMILY_MAP)

    # Fallback for any groupID not in table: dynamic by name
    unmapped_mask = visitor_df['ticket_family'].isna()
    if unmapped_mask.any():
        unmapped_tickets = visitor_df.loc[unmapped_mask, 'ticket_name'].unique()
        print(f"Dynamic classifying {len(unmapped_tickets)} unmapped tickets (fallback)...")
        fallback = classify_ticket_families(unmapped_tickets)
        visitor_df.loc[unmapped_mask, 'ticket_family'] = \
            visitor_df.loc[unmapped_mask, 'ticket_name'].map(fallback)

    visitor_df['ticket_family'] = visitor_df['ticket_family'].fillna('general')
    print("Ticket families after hardcoding + fallback:")
    print(visitor_df['ticket_family'].value_counts())

    # Keep for expansion
    ticket_families = (
        visitor_df[['ticket_name', 'ticket_family']]
        .drop_duplicates()
        .set_index('ticket_name')['ticket_family']
        .to_dict()
    )

    # --- Weather ---
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

    # --- Holiday & campaigns & events ---
    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    long_df = holiday_og_df.melt(
        id_vars=["date"], value_vars=region_cols,
        var_name="region", value_name="holiday"
    )
    long_df["holiday"] = long_df["holiday"].fillna("None").str.strip()
    long_df["holiday"] = long_df["holiday"].replace('', 'None')

    encoded = pd.get_dummies(long_df.set_index('date')['holiday'],
                             prefix='holiday').groupby('date').max()
    final_holiday_df = encoded.reset_index()
    holiday_intensity = long_df.groupby('date').size().reset_index(name='holiday_intensity')
    final_holiday_df = pd.merge(final_holiday_df, holiday_intensity, on='date', how='left')

    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden",
                          "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    camp_og_df.rename(columns={"Week ": "week"}, inplace=True, errors='ignore')
    promo_cols = [c for c in camp_og_df.columns if c.startswith("promo_")]
    camp_og_df['campaign_strength'] = camp_og_df[promo_cols].sum(axis=1)
    camp_og_df['promotion_active'] = (camp_og_df['campaign_strength'] > 0).astype(int)
    camp_og_df['campaign_regions_active'] = (camp_og_df[promo_cols] > 0).sum(axis=1)

    recurring_og_df.columns = ["event_name", "date"]
    recurring_og_df['event_name'] = recurring_og_df['event_name'].fillna('').str.split('/')
    recurring_df = recurring_og_df.explode('event_name')
    recurring_df['event_name'] = recurring_df['event_name'].str.strip().str.replace(' ', '_').str.lower()
    recurring_df['event_name'] = recurring_df['event_name'].str.replace(
        pat=r'^fc_emmen_.*', repl='soccer', regex=True
    )
    recurring_df.loc[recurring_df['event_name'] == '', 'event_name'] = 'no_event'
    recurring_df['date'] = pd.to_datetime(recurring_df['date'])
    recurring_df = recurring_df.drop_duplicates(subset=['date', 'event_name'])

    # --- Data Expansion 2022–last date ---
    print("Expanding data to include zero-sale days from 2022...")
    last_sales_date = visitor_df['date'].max()
    all_dates = pd.date_range(start='2022-01-01', end=last_sales_date, freq='D')
    all_tickets = visitor_df['ticket_name'].unique()

    print(f"Creating {len(all_dates)} dates × {len(all_tickets)} tickets")
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickets],
                                             names=['date', 'ticket_name'])
    expanded_df = pd.DataFrame(index=multi_index).reset_index()

    merge_cols = [
        'date', 'ticket_name', 'ticket_num', 'groupID',
        'is_actie_ticket', 'is_abonnement_ticket', 'is_full_price',
        'is_accommodation_ticket', 'is_group_ticket', 'is_joint_promotion'
    ]
    expanded_df = pd.merge(expanded_df, visitor_df[merge_cols],
                           on=['date', 'ticket_name'], how='left')
    expanded_df['ticket_num'] = expanded_df['ticket_num'].fillna(0).astype(int)

    expanded_df['ticket_family'] = expanded_df['ticket_name'].map(ticket_families).fillna('general')

    # Availability episodes
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

    merged_df = expanded_df.copy()

    # --- Merge with external data ---
    print("Merging with external data...")
    merged_df = pd.merge(merged_df, weather_daily, on="date", how="left")
    merged_df = pd.merge(merged_df, final_holiday_df, on="date", how="left")
    merged_df = pd.merge(merged_df, recurring_df, on="date", how="left")
    merged_df['event_name'] = merged_df['event_name'].fillna('no_event')

    merged_df["year"] = merged_df["date"].dt.year
    merged_df["week"] = merged_df["date"].dt.isocalendar().week
    merged_df = pd.merge(merged_df, camp_og_df, on=["year", "week"], how="left")

    merged_df["weekday"] = merged_df["date"].dt.weekday
    merged_df["day"] = merged_df["date"].dt.day
    merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)

    ticket_indicators = visitor_df[
        ['ticket_name', 'groupID', 'is_actie_ticket', 'is_abonnement_ticket',
         'is_full_price', 'is_accommodation_ticket', 'is_group_ticket',
         'is_joint_promotion']
    ].drop_duplicates()

    for col in ['groupID', 'is_actie_ticket', 'is_abonnement_ticket', 'is_full_price',
                'is_accommodation_ticket', 'is_group_ticket', 'is_joint_promotion']:
        merged_df[col] = merged_df['ticket_name'].map(
            ticket_indicators.set_index('ticket_name')[col]
        )
        if col == 'groupID':
            merged_df[col] = merged_df[col].fillna(-1).astype(int)
        else:
            merged_df[col] = merged_df[col].fillna(0).astype(int)

    weather_cols = ['temperature', 'rain_morning', 'rain_afternoon',
                    'precip_morning', 'precip_afternoon']
    for col in weather_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    print(f"Merge complete: {len(merged_df)} rows")

    # --- Lifecycle & features (unchanged) ---
    print("Creating ticket lifecycle features...")
    lifecycle_features = create_ticket_lifecycle_features(merged_df)
    merged_df = pd.merge(
        merged_df,
        lifecycle_features[['avg_sales', 'sales_cv', 'ticket_lifespan_days']],
        left_on='ticket_name',
        right_index=True,
        how='left'
    )

    print("Creating advanced features...")
    merged_df["month"] = merged_df["date"].dt.month
    merged_df['day_of_year'] = merged_df['date'].dt.dayofyear

    merged_df['day_of_year_sin'] = np.sin(2 * np.pi * merged_df['day_of_year'] / 365.25)
    merged_df['day_of_year_cos'] = np.cos(2 * np.pi * merged_df['day_of_year'] / 365.25)
    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

    merged_df['is_month_start'] = merged_df['day'].isin([1, 2, 3]).astype(int)
    merged_df['is_month_mid'] = merged_df['day'].isin([14, 15, 16]).astype(int)
    merged_df['is_month_end'] = merged_df['date'].dt.is_month_end.astype(int)

    christmas_date = pd.to_datetime(merged_df['year'].astype(str) + '-12-25')
    merged_df['days_until_christmas'] = (christmas_date - merged_df['date']).dt.days
    merged_df['days_until_christmas'] = merged_df['days_until_christmas'].apply(lambda x: max(0, x))
    merged_df['is_christmas_build_up'] = ((merged_df['month'] == 12) & (merged_df['day'].between(1, 24))).astype(int)
    merged_df['is_peak_christmas_week'] = ((merged_df['month'] == 12) & (merged_df['day'].between(18, 24))).astype(int)
    merged_df['is_twixtmas_period'] = ((merged_df['month'] == 12) & (merged_df['day'].between(26, 30))).astype(int)
    merged_df['is_kings_day'] = ((merged_df['month'] == 4) & (merged_df['day'] == 27)).astype(int)

    merged_df['temp_x_weekend'] = merged_df['temperature'] * merged_df['is_weekend']
    merged_df['is_perfect_day'] = (
        (merged_df['temperature'] > 20) &
        (merged_df['rain_morning'] == 0) &
        (merged_df['rain_afternoon'] == 0)
    ).astype(int)

    merged_df['campaign_x_actie'] = merged_df['campaign_strength'] * merged_df['is_actie_ticket']
    merged_df['promotion_x_actie'] = merged_df['promotion_active'] * merged_df['is_actie_ticket']
    merged_df['weekend_x_group'] = merged_df['is_weekend'] * merged_df['is_group_ticket']

    print("Creating lag and rolling features...")
    merged_df = merged_df.sort_values(['ticket_name', 'date'])
    group = merged_df.groupby('ticket_name')['ticket_num']
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        merged_df[f'sales_lag_{lag}'] = group.shift(lag)
    for window in [7, 14, 30]:
        merged_df[f'sales_rolling_avg_{window}'] = group.rolling(window).mean().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_std_{window}'] = group.rolling(window).std().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_min_{window}'] = group.rolling(window).min().reset_index(level=0, drop=True)
        merged_df[f'sales_rolling_max_{window}'] = group.rolling(window).max().reset_index(level=0, drop=True)

    lag_cols = [c for c in merged_df.columns if 'sales_lag_' in c or 'sales_rolling_' in c]
    merged_df[lag_cols] = merged_df[lag_cols].fillna(0)

    holiday_dates = pd.to_datetime(pd.Series(holiday_og_df['date'].dropna().unique())).sort_values()
    merged_df = merged_df.sort_values('date')
    next_holidays = pd.merge_asof(
        merged_df[['date']],
        pd.DataFrame({'holiday_date': holiday_dates}),
        left_on='date', right_on='holiday_date', direction='forward'
    )
    prev_holidays = pd.merge_asof(
        merged_df[['date']],
        pd.DataFrame({'holiday_date': holiday_dates}),
        left_on='date', right_on='holiday_date', direction='backward'
    )
    merged_df['days_until_holiday'] = (next_holidays['holiday_date'] - merged_df['date']).dt.days
    merged_df['days_since_holiday'] = (merged_df['date'] - prev_holidays['holiday_date']).dt.days

    merged_df['sales_momentum_7d'] = merged_df['sales_lag_1'] - merged_df['sales_lag_7']
    merged_df['sales_trend_30d'] = merged_df['sales_lag_1'] - merged_df['sales_lag_30']
    merged_df['temp_x_holiday_intensity'] = merged_df['temperature'] * merged_df['holiday_intensity']

    def calculate_days_since_available(group_):
        if group_.iloc[0] == 1:
            return group_.cumsum()
        return pd.Series(np.zeros(len(group_), dtype=int), index=group_.index)

    merged_df['days_since_available'] = (
        merged_df.groupby('ticket_name')['is_available']
        .apply(calculate_days_since_available)
        .reset_index(level=0, drop=True)
    )

    family_group = merged_df.groupby(['date', 'ticket_family'])['ticket_num'].sum().reset_index()
    family_pivot = family_group.pivot(index='date', columns='ticket_family', values='ticket_num').fillna(0)
    family_pivot.columns = [f'family_{col}_sales' for col in family_pivot.columns]
    merged_df = pd.merge(merged_df, family_pivot, on='date', how='left')

    merged_df.dropna(subset=['ticket_num', 'date', 'ticket_name'], inplace=True)

    print("One-hot encoding categorical variables...")
    event_dummies = pd.get_dummies(merged_df['event_name'], prefix='event')
    merged_df = pd.concat([merged_df, event_dummies], axis=1)
    ticket_dummies = pd.get_dummies(merged_df['ticket_name'], prefix='ticket')
    merged_df = pd.concat([merged_df, ticket_dummies], axis=1)
    family_dummies = pd.get_dummies(merged_df['ticket_family'], prefix='family')
    merged_df = pd.concat([merged_df, family_dummies], axis=1)

    print("Final cleanup...")
    print("Data processing completed successfully!")
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Ticket types: {merged_df['ticket_name'].nunique()}")
    print(f"Ticket families: {merged_df['ticket_family'].value_counts().to_dict()}")

    merged_df.sort_values(['ticket_name', 'date'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    print("Saving processed data...")
    merged_df.to_csv(PROCESSED_DIR / "processed_merge.csv", index=False)

    # Save mapping (3-element style) for debugging / predictions
    final_ticket_families = visitor_df[['ticket_name', 'groupID', 'ticket_family']].drop_duplicates()
    final_ticket_families.rename(columns={'groupID': 'subgroupId'}, inplace=True)
    final_ticket_families.to_csv(PROCESSED_DIR / "ticket_families.csv", index=False)

    lifecycle_features.to_csv(PROCESSED_DIR / "ticket_lifecycle.csv")

    print(f"\nYears in final dataset: {sorted(merged_df['date'].dt.year.unique())}")
    year_counts = merged_df['date'].dt.year.value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} rows")


if __name__ == "__main__":
    process_data()
