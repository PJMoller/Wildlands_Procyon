# pip install openpyxl!!!
import pandas as pd
import pandas.api.types as ptypes


def process_data():
    try:
        visitor_og_df = pd.read_csv("../data/raw/visitordaily.csv", sep=";")
    except Exception as e:
        print(f"Error loading visitor data: {e}")
        visitor_og_df = pd.DataFrame()

    try:
        weather_og_df = pd.read_excel("../data/raw/weather.xlsx")
    except Exception as e:
        print(f"Error loading weather data: {e}")
        weather_og_df = pd.DataFrame()

    try:
        holiday_og_df = pd.read_excel("../data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx")
    except Exception as e:
        print(f"Error loading holiday data: {e}")
        holiday_og_df = pd.DataFrame()

    try:
        camp_og_df = pd.read_excel("../data/raw/campaings.xlsx")
    except Exception as e:
        print(f"Error loading campaign data: {e}")
        camp_og_df = pd.DataFrame()



    # basic checks for empty dataframes and correct columns
    # 0 = visitor, 1 = weather, 2 = holiday, 3 = campaings
    dfs = [visitor_og_df, weather_og_df, holiday_og_df, camp_og_df]
    for i in range(len(dfs)):
        if dfs[i].empty:
            print(f"DataFrame {i} is empty. Exiting process_data function.")
            return

    expected_columns = [["AccessGroupId", "Description", "Date", "NumberOfUsedEntrances"],
        ["Time", "Temperature", "Rain", "Precipitation", "Hour"],
        ["Noord", "Midden", "Zuid", "Niedersachsen", "Nordrhein-Westfalen","Datum"],
        ["year", "Week ", "Regio Noord", "Regio Midden", "Regio Zuid","Noordrijn-Westfalen", "Nedersaksen"]]
    
    for j in range(len(dfs)):
        if list(dfs[j].columns) != expected_columns[j]:
            print(f"DataFrame {j} has unexpected columns. Exiting process_data function.")
            return



    # checks for correct data types
    # visitor df expected dtypes
    visitor_dtypes = {
        "AccessGroupId": ptypes.is_integer_dtype,
        "Description": lambda dt: dt == 'object',
        "Date": lambda dt: dt == 'object',
        "NumberOfUsedEntrances": ptypes.is_integer_dtype,
    }
    for col, check_func in visitor_dtypes.items():
        if col not in visitor_og_df.columns or not check_func(visitor_og_df[col].dtype):
            print(f"Visitor DataFrame column '{col}' has incorrect dtype: {visitor_og_df[col].dtype if col in visitor_og_df else 'missing'}")
            return

    # weather df expected dtypes
    weather_dtypes = {
        "Time": ptypes.is_datetime64_any_dtype,
        "Temperature": ptypes.is_numeric_dtype,
        "Rain": ptypes.is_numeric_dtype,
        "Precipitation": ptypes.is_numeric_dtype,
        "Hour": ptypes.is_integer_dtype,
    }
    for col, check_func in weather_dtypes.items():
        if col not in weather_og_df.columns or not check_func(weather_og_df[col].dtype):
            print(f"Weather DataFrame column '{col}' has incorrect dtype: {weather_og_df[col].dtype if col in weather_og_df else 'missing'}")
            return

    # holiday df expected dtypes (mostly object or datetime)
    holiday_dtypes = {
        "Noord": lambda dt: dt == 'object',
        "Midden": lambda dt: dt == 'object',
        "Zuid": lambda dt: dt == 'object',
        "Niedersachsen": lambda dt: dt == 'object',
        "Nordrhein-Westfalen": lambda dt: dt == 'object',
        "Datum": ptypes.is_datetime64_any_dtype,
    }
    for col, check_func in holiday_dtypes.items():
        if col not in holiday_og_df.columns or not check_func(holiday_og_df[col].dtype):
            print(f"Holiday DataFrame column '{col}' has incorrect dtype: {holiday_og_df[col].dtype if col in holiday_og_df else 'missing'}")
            return

    # campaign df expected dtypes
    camp_dtypes = {
        "year": ptypes.is_integer_dtype,
        "Week ": ptypes.is_integer_dtype,
        "Regio Noord": ptypes.is_integer_dtype,
        "Regio Midden": ptypes.is_integer_dtype,
        "Regio Zuid": ptypes.is_integer_dtype,
        "Noordrijn-Westfalen": ptypes.is_integer_dtype,
        "Nedersaksen": ptypes.is_integer_dtype,
    }
    for col, check_func in camp_dtypes.items():
        if col not in camp_og_df.columns or not check_func(camp_og_df[col].dtype):
            print(f"Campaign DataFrame column '{col}' has incorrect dtype: {camp_og_df[col].dtype if col in camp_og_df else 'missing'}")
            return


    """
    #checks for basic analysis

    print(visitor_og_df.dtypes)
    print(weather_og_df.dtypes)
    print(holiday_og_df.dtypes)
    print(camp_og_df.dtypes)

    print(weather_og_df.head())
    print(visitor_og_df.head())
    print(holiday_og_df.head())
    print(camp_og_df.head())

    print(visitor_og_df.isnull().values.any()) # if no null values, returns false ; no null values
    print(weather_og_df.isnull().values.any()) # if no null values, returns false ; no null values
    print(holiday_og_df.isnull().values.any()) # if no null values, returns false ; has null values, but it's okay because im handling it later
    print(camp_og_df.isnull().values.any()) # if no null values, returns false ; no null values 

    print(visitor_og_df.duplicated().sum()) # no duplicates W
    print(weather_og_df.duplicated().sum()) # no duplicates here either, W
    print(holiday_og_df.duplicated().sum()) # no duplicates, W
    print(camp_og_df.duplicated().sum()) # no duplicates, W
    """
    visitor_og_df.columns = ["groupID", "ticket_name", "date", "ticket_num"]
    visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d")
    visitor_df = visitor_og_df[['date', 'ticket_name', 'ticket_num']]

    weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"]
    weather_og_df = weather_og_df.drop("hour", axis=1)
    weather_og_df["date"] = weather_og_df["date"].dt.date
    weather_daily = weather_og_df.groupby("date").agg({"temperature": "mean", "rain": "sum", "precipitation": "sum"}).reset_index()
    weather_daily["date"] = pd.to_datetime(weather_daily["date"], format="%Y-%m-%d")
    weather_daily = weather_daily.round({"temperature": 1, "rain": 1, "precipitation": 1})

    holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
    region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    long_df = holiday_og_df.melt(id_vars=["date"], value_vars=region_cols, var_name="region", value_name="holiday")
    long_df["holiday"] = long_df["holiday"].fillna("None")
    long_df["region_holiday"] = long_df["region"] + "_" + long_df["holiday"]
    encoded = pd.get_dummies(long_df["region_holiday"])
    result_df = pd.concat([long_df[["date"]], encoded], axis=1)
    final_holiday_df = result_df.groupby("date").max().reset_index()
    bool_cols = final_holiday_df.columns.drop("date")
    final_holiday_df[bool_cols] = final_holiday_df[bool_cols].astype(int)

    camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"]
    camp_og_df.rename(columns={"Week ": "week"}, inplace=True, errors='ignore')

    # the great expansion to include zero-sale days 
    print("Expanding data to include zero-sale days...")
    all_dates = pd.date_range(start=visitor_df['date'].min(), end=visitor_df['date'].max(), freq='D')
    all_tickets = visitor_df['ticket_name'].unique()
    multi_index = pd.MultiIndex.from_product([all_dates, all_tickets], names=['date', 'ticket_name'])
    expanded_df = pd.DataFrame(index=multi_index).reset_index()
    expanded_df = pd.merge(expanded_df, visitor_df, on=['date', 'ticket_name'], how='left')
    expanded_df['ticket_num'] = expanded_df['ticket_num'].fillna(0).astype(int)
    print("Zero-sale days created. Dataset is now complete.")

    # feature engineering 
    daily_features_df = pd.merge(weather_daily, final_holiday_df, on="date", how="inner")
    merged_df = pd.merge(expanded_df, daily_features_df, on="date", how="left")

    print(f"Original expanded size: {len(merged_df)} rows")
    merged_df = merged_df[merged_df['date'].dt.year >= 2024].copy()
    print(f"Filtered size (2024 onwards): {len(merged_df)} rows")

    merged_df["year"] = merged_df["date"].dt.year
    merged_df["month"] = merged_df["date"].dt.month
    merged_df["week"] = merged_df["date"].dt.isocalendar().week
    merged_df["day"] = merged_df["date"].dt.day
    merged_df["weekday"] = merged_df["date"].dt.weekday
    merged_df['is_weekend'] = (merged_df['weekday'] >= 5).astype(int)
    
    merged_df = pd.merge(merged_df, camp_og_df, on=["year", "week"], how="left")
    
    # more features
    print("Creating advanced time-based features...")

    holiday_region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]
    all_holidays = holiday_og_df.melt(id_vars=['date'], value_vars=holiday_region_cols)

    holiday_dates = pd.to_datetime(pd.Series(all_holidays.dropna()['date'].unique())).sort_values()
    merged_df = merged_df.sort_values('date')
    merged_df['next_holiday'] = pd.merge_asof(merged_df, pd.DataFrame({'holiday_date': holiday_dates}), left_on='date', right_on='holiday_date', direction='forward')['holiday_date']
    merged_df['prev_holiday'] = pd.merge_asof(merged_df, pd.DataFrame({'holiday_date': holiday_dates}), left_on='date', right_on='holiday_date', direction='backward')['holiday_date']
    merged_df['days_until_holiday'] = (merged_df['next_holiday'] - merged_df['date']).dt.days
    merged_df['days_since_holiday'] = (merged_df['date'] - merged_df['prev_holiday']).dt.days
    merged_df = merged_df.drop(columns=['next_holiday', 'prev_holiday'])
    
    merged_df['is_month_start'] = merged_df['day'].isin([1, 2, 3]).astype(int)
    merged_df['is_month_mid'] = merged_df['day'].isin([14, 15, 16]).astype(int)
    merged_df['is_month_end'] = merged_df['date'].dt.is_month_end.astype(int)
    merged_df['temp_x_weekend'] = merged_df['temperature'] * merged_df['is_weekend']
    
    temp_holiday_df = holiday_og_df.melt(id_vars=['date'], value_vars=region_cols, var_name='region', value_name='holiday').dropna()
    temp_holiday_df['region_holiday'] = temp_holiday_df['region'] + '_' + temp_holiday_df['holiday']
    summer_dates = temp_holiday_df[temp_holiday_df['region_holiday'].str.contains("Zomervakantie", na=False)]['date'].unique()
    merged_df['is_summer_vacation'] = merged_df['date'].isin(summer_dates).astype(int)
    print("Advanced features created.")

    # --- Lag and Rolling Features ---
    print("Creating lag and rolling features...")
    merged_df = merged_df.sort_values(by=['ticket_name', 'date'])
    
    merged_df['sales_same_day_last_week'] = merged_df.groupby(['ticket_name', 'weekday'])['ticket_num'].shift(1)
    merged_df['sales_rolling_std_7'] = merged_df.groupby('ticket_name')['ticket_num'].shift(1).rolling(window=7, min_periods=1).std()
    
    lags = [1, 7, 14]
    for lag in lags:
        merged_df[f'sales_lag_{lag}'] = merged_df.groupby('ticket_name')['ticket_num'].shift(lag)
        
    merged_df['sales_rolling_avg_7'] = merged_df.groupby('ticket_name')['ticket_num'].shift(1).rolling(window=7, min_periods=1).mean().round(2)
    merged_df.fillna(0, inplace=True)
    print("Lag and rolling features created.")


    final_merge = pd.get_dummies(merged_df, columns=["ticket_name"], prefix="ticket", dtype=int) # one hot encode ticket names

    final_merge = final_merge.drop(columns=["date"]) # drop date since we have year month day now

    print("Optimizing memory by converting to 32-bit floats...")
    for col in final_merge.columns:
        if final_merge[col].dtype == 'float64':
            final_merge[col] = final_merge[col].astype('float32')
        if final_merge[col].dtype == 'int64':
            final_merge[col] = final_merge[col].astype('int32')

    all_cols = final_merge.columns.tolist()
    
    # Define the required order
    ordered_cols = ['year', 'month', 'day'] + [col for col in all_cols if col not in ['year', 'month', 'day']]
    
    # Reorder the DataFrame
    final_df = final_merge[ordered_cols]

    final_df.to_csv("../data/processed/processed_merge.csv", index=False) # can be used for ML now

# training models in model_training.py

if __name__ == "__main__":
    process_data()
    #print("successful execution")