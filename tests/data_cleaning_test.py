# pip install openpyxl!!!
import pandas as pd

visitor_og_df = pd.read_csv("../data/raw/visitordaily.csv", sep=";")
weather_og_df = pd.read_excel("../data/raw/weather.xlsx")
holiday_og_df = pd.read_excel("../data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx")
camp_og_df = pd.read_excel("../data/raw/campaings.xlsx")


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


# changes to the visitors df
visitor_og_df.columns = ["groupID","ticket_name", "date", "ticket_num"] # lowercase columns
visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d") # convert to datetime
visitor_og_df["ticket_num"] = visitor_og_df["ticket_num"].astype(int) # convert to int

visitor_og_df = visitor_og_df.drop("groupID", axis=1) # drop access group id since its not needed

visitor_og_df = pd.get_dummies(visitor_og_df, columns=["ticket_name"], prefix="ticket") # one hot encode ticket names

bool_cols = visitor_og_df.columns.drop(["date","ticket_num"])
visitor_og_df[bool_cols] = visitor_og_df[bool_cols].astype(int) # since true = 1 and false = 0 its easy to convert like this



# changes to weather df

weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"] # rename columns
weather_og_df = weather_og_df.drop("hour", axis=1)
weather_og_df["date"] = weather_og_df["date"].dt.date # drop the hour from the date
weather_daily = weather_og_df.groupby("date").agg({"temperature": "mean", "rain": "sum", "precipitation": "sum"}).reset_index()
weather_daily["date"] = pd.to_datetime(weather_daily["date"], format="%Y-%m-%d") # convert to datetime again so we can merge later
weather_daily = weather_daily.round({"temperature": 1, "rain": 1, "precipitation": 1})



# changes to the holiday df

# Rename columns so all regions and date are clear
holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]

region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]

# Melt to long format: (date, region, holiday)
long_df = holiday_og_df.melt(id_vars=["date"], value_vars=region_cols,
                  var_name="region", value_name="holiday")
long_df["holiday"] = long_df["holiday"].fillna("None")

# Create combination label
long_df["region_holiday"] = long_df["region"] + "_" + long_df["holiday"]

# one hot encode the combination
encoded = pd.get_dummies(long_df["region_holiday"])
result_df = pd.concat([long_df[["date"]], encoded], axis=1)

# Aggregate, so each date's 1s only indicate a true region-holiday match
final_holiday_df = result_df.groupby("date").max().reset_index()

# Convert all bool columns to 1s and 0s
bool_cols = final_holiday_df.columns.drop("date")
final_holiday_df[bool_cols] = final_holiday_df[bool_cols].astype(int) # since true = 1 and false = 0 its easy to convert like this

# Now final_holiday_df can be used for merging, one hot encoded



# changes to campaign df
camp_og_df.columns = ["year", "week", "promo_NLNoord", "promo_NLMidden", "promo_NLZuid", "promo_Nordrhein-Westfalen", "promo_Niedersachsen"] # rename columns



# merge the 3 datasets, #1 weather + visitors, #2 add holidays

merged_wh_df = pd.merge(weather_daily, visitor_og_df, on="date", how="inner")

merged_wh_df["date"] = pd.to_datetime(merged_wh_df["date"].dt.date, format="%Y-%m-%d")

# now merge with holidays
merged_final_df = pd.merge(merged_wh_df, final_holiday_df, on="date", how="inner")

merged_final_df["year"] = merged_final_df["date"].dt.year
merged_final_df["month"] = merged_final_df["date"].dt.month
merged_final_df["week"] = merged_final_df["date"].dt.isocalendar().week
merged_final_df["day"] = merged_final_df["date"].dt.day
merged_final_df["weekday"] = merged_final_df["date"].dt.weekday

merged_final_df = pd.merge(merged_final_df, camp_og_df, on=["year", "week"], how="left")

merged_df = merged_final_df.drop(columns=["date"]) # drop date since we have year month day now

# print to csv for model training
merged_df.to_csv("../data/processed/processed_merge.csv", index=False) # can be used for ML now

# training models in model_training.py