from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import date, timedelta
import os

# Loading data and models
PROCESSED_CSV = "data/processed/processed_merge.csv"
MODEL_PATH = "data/processed/model.pkl"
HOLIDAYS_CSV = "data/processed/processed_holidays.csv"  # must include 'date' column
#PROMOTIONS_XLSX = "data/promotions.xlsx"  # future use
OUTPUT_DIR = "data/predictions/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE = date.today()

#DB_FILE = "data/database/database.db"


def get_openmeteo():
    # Open-Meteo API setup and retry if you get an error
    cache_session = requests_cache.CachedSession(".cache", expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # setting up location and parameters
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.7862,
        "longitude": 6.8917,
        "daily": ["temperature_2m_max", "temperature_2m_min" , "precipitation_sum", "rain_sum"],
        "forecast_days": 16 
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_data = {
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq = pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
        "temp_max": daily.Variables(0).ValuesAsNumpy(),
        "temp_min": daily.Variables(1).ValuesAsNumpy(),
        "precipitation": daily.Variables(2).ValuesAsNumpy(),
        "rain": daily.Variables(3).ValuesAsNumpy(),
    }

    df_weather = pd.DataFrame(daily_data)

    # Convert everything to float64 and round numerically
    df_weather["temperature"] = ((df_weather["temp_max"] + df_weather["temp_min"]) / 2).astype("float64").round(1)
    df_weather["precipitation"] = df_weather["precipitation"].astype("float64").round(1)
    df_weather["rain"] = df_weather["rain"].astype("float64").round(1)

    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.tz_convert(None)
    df_weather.drop(columns=["temp_max", "temp_min"], inplace=True)
    
    #print("\nDaily data\n", df_weather)

    # Round all relevant columns to one decimal
    pd.options.display.float_format = "{:.1f}".format

    return df_weather
    



def seperating(processed_df):

    """
    for loop for looking into 3 cattegorries
    1: api (temperature, rain, precipitation)
    2: start with ticket_
    3: holiday/special day
    """

    #headers = list(processed_df.columns)
    #print(headers)

    api = []
    ticket = []
    holiday = []


    for col in processed_df.columns:
        #print(col)

        if col == "temperature" or col == "rain" or col == "precipitation":
            #print("API")
            api.append(col)
        elif col.startswith("ticket_") and col != "ticket_num":
            #print("ticket_")
            ticket.append(col)
        else:
            holiday.append(col)
            # if col.startswith("NLNoord"):
            #     print("Noord")
            # elif col.startswith("NLMidden"):
            #     print("Midden")
            # elif col.startswith("NLZuid"):
            #     print("Zuid")
            # elif col.startswith("Niedersachsen"):
            #     print("Niedersachsen")
            # elif col.startswith("Nordrhein-Westfalen"):
            #     print("Nordrhein-Westfalen")
            # else:
            #     print("overig")
        
    # print(api)
    # print("")
    # print("")
    # print(ticket)
    # print("")
    # print("")
    # print(holiday)

    return api, ticket, holiday

def predict_next_365_days():
    print("loading model")
    model = joblib.load(MODEL_PATH)
    
    print("fetching open-meteo data")
    openmeteo_df = get_openmeteo()

    print("loading processed and holiday data")
    processed_df = pd.read_csv(PROCESSED_CSV)
    holidays_df = pd.read_csv(HOLIDAYS_CSV)
    holidays_df["date"] = pd.to_datetime(holidays_df["date"]).dt.date
    
    api_cols, ticket_cols, holiday_cols = seperating(processed_df)

    if "date" not in processed_df.columns:
        if {"year", "month", "day"}.issubset(processed_df.columns):
            processed_df["date"] = pd.to_datetime(processed_df[["year", "month", "day"]])
        elif {"year", "week"}.issubset(processed_df.columns):
            processed_df["date"] = processed_df.apply(
                lambda r: pd.to_datetime(f"{int(r['year'])}-W{int(r['week']):02d}-1", format="%G-W%V-%u"),
                axis=1
            )
        else:
            raise KeyError("No suitable columns found for 'date'.")

    processed_df["year"] = processed_df["date"].dt.year
    processed_df["month"] = processed_df["date"].dt.month
    processed_df["week"] = processed_df["date"].dt.isocalendar().week
    processed_df["day"] = processed_df["date"].dt.day
    processed_df["weekday"] = processed_df["date"].dt.weekday


    avg_weather = (
        processed_df.groupby(["month", "day"])[["temperature", "rain", "precipitation"]]
        .mean()
        .reset_index()
    )

    print("starting 365 pred loop")
    all_days_rows = []

    for d in range(365):
        current_date = DATE + timedelta(days=d)

        # if os.path.exists(csv_filename):
        #     print(f"{current_date}, already predicted - skipping ")
        #     continue

        #print("starting api part")
        # API PART #
        match = openmeteo_df[openmeteo_df["date"].dt.date == current_date]
        if not match.empty:
            temperature = match["temperature"].iloc[0]
            rain = match["rain"].iloc[0]
            precipitation = match["precipitation"].iloc[0]
        else:
            avg = avg_weather[(avg_weather["month"] == current_date.month)
                & (avg_weather["day"] == current_date.day)]
            temperature = avg["temperature"].iloc[0] if not avg.empty else np.nan
            rain = avg["rain"].iloc[0] if not avg.empty else np.nan
            precipitation = avg["precipitation"].iloc[0] if not avg.empty else np.nan
        #print("finished api part")

        # HOLIDAY PART # 
        #print("starting holiday part")
        match_holiday = holidays_df[holidays_df["date"] == current_date]
        if match_holiday.empty:
            #print(f"please update holidays file, cant find {current_date}")
            break

        holiday_part = match_holiday.iloc[0].to_dict()
        holiday_part.pop("date", None)
        holiday_part.update({
            "year": current_date.year,
            "month": current_date.month,
            "day": current_date.day,
            "weekday": current_date.weekday(),
        })

        #print("finished holiday part")
        # PROMOTION PART #
        ##########################
        # PROMOTION WILL BE HERE #
        ##########################


        # EVENT PART #
        ######################
        # EVENT WILL BE HERE #
        # EVENT INFLUENCE, RUN IT WIHTOUT EVENT AND THEN WITH AND CALCULATE DIFFERENCE
        ######################
        


        # TICKET PART # 
        #print("starting ticket part")
        predictions_per_ticket = {}
        for ticket_name in ticket_cols:
            ticket_part = {col: 0 for col in ticket_cols}
            ticket_part[ticket_name] = 1

            #print("finished ticket part")
            
            # combining features
            row = {
                "temperature": temperature,
                "rain": rain,
                "precipitation": precipitation,
                **holiday_part,
                **ticket_part
            }

            input_df = pd.DataFrame([row])

            #print("combined features")

            # if column does not exist, ignore it
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
                input_df = input_df.reindex(columns=expected_features, fill_value=0)
            #else:
                #print("model does not sure feature names, skipping them")

            predicted_total = model.predict(input_df)[0]
            predictions_per_ticket[ticket_name] = predicted_total

        total_visitors = sum(predictions_per_ticket.values())

        daily_summary = {
            "date": current_date,
            "year": current_date.year,
            "month": current_date.month,
            "week": current_date.isocalendar().week,
            "day": current_date.day,
            "weekday": current_date.weekday(),
            "temperature": round(temperature, 1),
            "rain": round(rain, 1),
            "precipitation": round(precipitation, 1),
            "total_visitors": round(total_visitors, 0),
            **predictions_per_ticket,
            **holiday_part  # Include holiday columns for display on dashboard
        }

        all_days_rows.append(daily_summary)
        
        #print("linked ticket type using bitshifting")


        print(
            f"{current_date} | Temp: {temperature:.2f}°C | Rain: {rain:.2f} | "
            f"Precipitation: {precipitation:.2f} | Total Visitors: {total_visitors:.0f}"
        )

        # Round all relevant columns to one decimal
        pd.options.display.float_format = "{:.1f}".format
        #print("made the predicted total")

        # Print daily summary
        ticket_distribution = {t:row.get(t+"_predicted", 0) for t in ticket_cols}
        


        #print("printed succesfully")

        # Save CSV per day (append to full file)
    csv_filename = os.path.join(OUTPUT_DIR, f"predictions_{current_date}.csv")
    pd.DataFrame(all_days_rows).to_csv(csv_filename, index=False, float_format="%.1f")
        #print(f"saved daily predictions for {current_date}")

    print("all 365 days predictions complete")


if __name__ == "__main__":
    predict_next_365_days()



# EVENT INFLUENCE, RUN IT WIHTOUT EVENT AND THEN WITH AND CALCULATE DIFFERENCE
