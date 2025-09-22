# pip install openpyxl!!!
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

visitor_og_df = pd.read_csv("./data/visitor_sample.csv", sep=";", header=None, names=["ticket_id", "ticket_name", "date", "hour", "num_tickets_bought"])
#print(visitor_og_df.head())
#print(visitor_og_df.dtypes) # date is object weird
#print(visitor_og_df.describe())

weather_og_df = pd.read_excel("./data/weather.xlsx")
#print(weather_og_df.head())
#print(weather_og_df.dtypes) # date is datetime 
#print(weather_og_df.describe())

holiday_og_df = pd.read_excel("./data/Holidays 2024 Netherlands and Germany.xlsx")
#print(holiday_og_df.head())
#print(holiday_og_df.dtypes)
#print(holiday_og_df.describe())

# changes to the visitors df
visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d") # convert to datetime
visitor_og_df["num_tickets_bought"] = visitor_og_df["num_tickets_bought"].astype(int) # convert to int
visitor_og_df["ticket_id"] = visitor_og_df["ticket_id"].astype(int) # convert to int
visitor_og_df["ticket_name"] = visitor_og_df["ticket_name"].astype(str) # convert to str just to make sure
visitor_og_df["hour"] = pd.to_datetime(visitor_og_df["hour"], format="%H:%M:%S").dt.hour # convert to hour int


hourly_summary = visitor_og_df.groupby(["date", "hour"])["num_tickets_bought"].sum().reset_index()

#hourly_df = hourly_summary.pivot(index="date", columns="hour", values="num_tickets_bought").fillna(0) # just in case we need it like this

### now i need to concat the visitor + synthetic data that i dont have yet with 


#print(visitor_og_df.dtypes) # should be good now
#print(weather_og_df.dtypes) # both of them are good, nice

#print(weather_og_df.head())
#print(visitor_og_df.head())

#print(visitor_og_df.isnull().values.any()) # if no null values, returns false ; no null values W
#print(weather_og_df.isnull().values.any()) # if no null values, returns false ; no null values W

#print(visitor_og_df.duplicated().sum()) # no duplicates W
#print(weather_og_df.duplicated().sum()) # no duplicates here either, W



# changes to the holiday df

holiday_og_df = holiday_og_df.drop(columns=["Vakantie", "Regio's", "Feestdag"]) # drop the useless columns

holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "Datum"] # rename columns

region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]

# Melt to long format: (date, region, holiday)
long_df = holiday_og_df.melt(id_vars=["Datum"], value_vars=region_cols,
                  var_name="region", value_name="holiday")
long_df["holiday"] = long_df["holiday"].fillna("None")

# One-hot encode region and holiday columns
ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded = ohe.fit_transform(long_df[["region", "holiday"]])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(["region", "holiday"]))

# Concatenate encoded columns with dates
result_df = pd.concat([long_df[["Datum"]], encoded_df], axis=1)

# Aggregate: one row per date, with max() to combine region/holiday across regions
final_holiday_df = result_df.groupby("Datum").max().reset_index()

# Now final_holiday_df can be used for ML, one hot encoded
#print(final_holiday_df.head())

print(weather_og_df.head())
#print(hourly_df.head(20)) # test
print(hourly_summary.head(20))

final_holiday_df.to_csv("processed_holidays.csv", index=False) # for visual purposes for myself