import pandas as pd
import numpy as np

visitor_og_df = pd.read_csv("../data/visitor_sample.csv", sep=";", header=None, names=["ticket_id", "ticket_name", "date", "hour", "num_tickets_bought"])
#print(visitor_og_df.head())
#print(visitor_og_df.dtypes) # date is object weird
#print(visitor_og_df.describe())

weather_og_df = pd.read_excel("../data/weather.xlsx")
print(weather_og_df.head())
print(weather_og_df.dtypes) # date is datetime 
print(weather_og_df.describe())

#print(visitor_og_df.isnull().values.any()) # if no null values, returns false ; no null values W
#print(weather_og_df.isnull().values.any())