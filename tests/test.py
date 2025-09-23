# pip install openpyxl!!!
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

visitor_og_df = pd.read_csv("../data/visitor_sample.csv", sep=";", header=None, names=["ticket_id", "ticket_name", "date", "hour", "num_tickets_bought"])
#print(visitor_og_df.head())
#print(visitor_og_df.dtypes) # date is object weird
#print(visitor_og_df.describe())

weather_og_df = pd.read_excel("../data/weather.xlsx")
#print(weather_og_df.head())
#print(weather_og_df.dtypes) # date is datetime 
#print(weather_og_df.describe())

holiday_og_df = pd.read_excel("../data/Holidays 2024 Netherlands and Germany.xlsx")
#print(holiday_og_df.head())
#print(holiday_og_df.dtypes)
#print(holiday_og_df.describe())

# checks for basic analysis

#print(visitor_og_df.dtypes) # should be good now
#print(weather_og_df.dtypes) # both of them are good, nice

#print(weather_og_df.head())
#print(visitor_og_df.head())

#print(visitor_og_df.isnull().values.any()) # if no null values, returns false ; no null values W
#print(weather_og_df.isnull().values.any()) # if no null values, returns false ; no null values W

#print(visitor_og_df.duplicated().sum()) # no duplicates W
#print(weather_og_df.duplicated().sum()) # no duplicates here either, W



# changes to the visitors df
visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d") # convert to datetime
visitor_og_df["num_tickets_bought"] = visitor_og_df["num_tickets_bought"].astype(int) # convert to int
visitor_og_df["ticket_id"] = visitor_og_df["ticket_id"].astype(int) # convert to int
visitor_og_df["ticket_name"] = visitor_og_df["ticket_name"].astype(str) # convert to str just to make sure
visitor_og_df["hour"] = pd.to_datetime(visitor_og_df["hour"], format="%H:%M:%S").dt.hour # convert to hour int

# merge hour with date to have a full datetime
visitor_og_df['date'] = pd.to_datetime(visitor_og_df['date'].astype(str) + ' ' + visitor_og_df['hour'].astype(str) + ':00:00') # merge hour with date

# get hourly summary
visitor_og_df = visitor_og_df.drop(columns=["hour", "ticket_name", "ticket_id"]) # drop hour and ticket name since we dont need it
hourly_summary = visitor_og_df.groupby(["date"])["num_tickets_bought"].sum().reset_index()

#hourly_df = hourly_summary.pivot(index="date", columns="hour", values="num_tickets_bought").fillna(0) # just in case we need it like this

#print(hourly_summary.head(20)) #test, looks good

# changes to weather df

weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"] # rename columns



# changes to the holiday df

holiday_og_df = holiday_og_df.drop(columns=["Vakantie", "Regio's", "Feestdag"]) # drop the useless columns

holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"] # rename columns

region_cols = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen"]

# Melt to long format: (date, region, holiday)
long_df = holiday_og_df.melt(id_vars=["date"], value_vars=region_cols,
                  var_name="region", value_name="holiday")
long_df["holiday"] = long_df["holiday"].fillna("None")

# One-hot encode region and holiday columns
ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded = ohe.fit_transform(long_df[["region", "holiday"]])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(["region", "holiday"]))

# Concatenate encoded columns with dates
result_df = pd.concat([long_df[["date"]], encoded_df], axis=1)

# Aggregate: one row per date, with max() to combine region/holiday across regions
final_holiday_df = result_df.groupby("date").max().reset_index()

# Now final_holiday_df can be used for ML, one hot encoded
#print(final_holiday_df.head())



### now i need to concat the hourly visitors +  weather data

# try to merge

merged_wh_df = pd.merge(weather_og_df, hourly_summary, on='date', how='inner')
print(merged_wh_df.head(20))

# save the small df for ML so we can get started with that 
merged_wh_df.to_csv("../data/ml_prot.csv", index=False) # use the csv it creates for ML, its going to be bad

print(final_holiday_df.head(20)) # #####################
merged_wh_df["hour"] = merged_wh_df["date"].dt.hour
merged_wh_df["date"] = pd.to_datetime(merged_wh_df["date"].dt.date, format="%Y-%m-%d")
#print(merged_wh_df.head(20))
#print(final_holiday_df.head(20))
merged_final_df = pd.merge(merged_wh_df, final_holiday_df, on='date', how='inner')


merged_final_df['year'] = merged_final_df['date'].dt.year
merged_final_df['month'] = merged_final_df['date'].dt.month
merged_final_df['day'] = merged_final_df['date'].dt.day
merged_final_df['weekday'] = merged_final_df['date'].dt.weekday

merged_df = merged_final_df.drop(columns=['date']) # drop date since we have year month day hour now
print(merged_df.head(20))

#print(weather_og_df.head())
#print(hourly_summary.head())

#merged_df.to_csv("../data/processed_merge.csv", index=False) # for visual purposes for myself
#final_holiday_df.to_csv("../data/processed_holidays.csv", index=False) # for visual purposes for myself