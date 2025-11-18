
from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
from datetime import datetime, date, timedelta
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os

# === File paths ===
PROCESSED_CSV = "data/processed/processed_merge.csv"
MODEL_PATH = "data/processed/model.pkl"
MODEL_PATH = "data/processed/RFRmodel.pkl"
HOLIDAYS_CSV = "data/processed/processed_holidays.csv"
OUTPUT_DIR = "data/predictions/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE = date.today()

# === Fetch Open-Meteo data ===
def get_openmeteo():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.7862,
        "longitude": 6.8917,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"], #"rain_sum"
        "forecast_days": 16
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
        "temp_max": daily.Variables(0).ValuesAsNumpy(),
        "temp_min": daily.Variables(1).ValuesAsNumpy(),
        "precipitation": daily.Variables(2).ValuesAsNumpy(),
        # "rain": daily.Variables(3).ValuesAsNumpy(),
    }

    df_weather = pd.DataFrame(daily_data)
    df_weather["temperature"] = ((df_weather["temp_max"] + df_weather["temp_min"]) / 2).astype("float64").round(1)
    df_weather["precipitation"] = df_weather["precipitation"].astype("float64").round(1)
    #df_weather["rain"] = df_weather["rain"].astype("float64").round(1)
    df_weather["date"] = pd.to_datetime(df_weather["date"]).dt.tz_convert(None)
    df_weather.drop(columns=["temp_max", "temp_min"], inplace=True)

    pd.options.display.float_format = "{:.1f}".format

    return df_weather


# === Separate columns into categories ===
def seperating(processed_df):
    api = []
    ticket = []
    holiday = []

    for col in processed_df.columns:
        if col in ["temperature", "precipitation"]:     # removed "rain"
            api.append(col)
        elif col.startswith("ticket_") and col != "ticket_num":
            ticket.append(col)
        else:
            holiday.append(col)

    print("--------------------------------------")
    print(ticket)
    print("--------------------------------------")

    return api, ticket, holiday


def predict_next_365_days():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)

    openmeteo_df = get_openmeteo()
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

    # avg_weather = (
    #     processed_df.groupby(["month", "day"])[["temperature", "rain", "precipitation"]]
    #     .mean()
    #     .reset_index()
    # )


    avg_rain_by_day = (
        processed_df.groupby(["month", "day"])[["rain"]]
        .mean()
        .reset_index()
        .rename(columns={"rain": "avg_rain"})
    )
    #print(avg_weather)

    print("Starting 365-day prediction loop...")
    all_days_rows = []

    for d in range(365):
        current_date = DATE + timedelta(days=d)

        # Weather data
        match = openmeteo_df[openmeteo_df["date"].dt.date == current_date]
        if not match.empty:
            temperature = match["temperature"].iloc[0]
            #rain = match["rain"].iloc[0]
            #precipitation = match["precipitation"].iloc[0]
            #print(f"if not match {temperature}, {rain}, {precipitation}")
        else:
            temp_avg = processed_df[
                (processed_df["month"] == current_date.month) &
                (processed_df["day"] == current_date.day)
            ]["temperature"].mean()
            temperature = temp_avg if not np.isnan(temp_avg) else np.nan
            
        rain_match = avg_rain_by_day[
            (avg_rain_by_day["month"] == current_date.month) &
            (avg_rain_by_day["day"] == current_date.day)
        ]

        #precipitation = float(rain_match["avg_rain"].iloc[0]) if not rain_match.empty else 0.0

        if not rain_match.empty:
            print("if not")
            precipitation = float(rain_match["avg_rain"].iloc[0])
        else:
            print("else")
            precipitation = float(0.0)

            # avg = avg_weather[(avg_weather["month"] == current_date.month) & (avg_weather["day"] == current_date.day)]
            # temperature = avg["temperature"].iloc[0] if not avg.empty else np.nan
            # rain = avg["rain"].iloc[0] if not avg.empty else np.nan
            # precipitation = avg["precipitation"].iloc[0] if not avg.empty else np.nan
            #print(f"else {temperature}, {rain}, {precipitation}")


        # Holiday data
        match_holiday = holidays_df[holidays_df["date"] == current_date]
        if match_holiday.empty:
            print(f"⚠ Missing holiday data for {current_date}, skipping day.")
            continue

        holiday_part = match_holiday.iloc[0].to_dict()
        holiday_part.pop("date", None)
        holiday_part.update({
            "year": current_date.year,
            "month": current_date.month,
            "week": current_date.isocalendar().week,
            "day": current_date.day,
            "weekday": current_date.weekday(),
        })


        # TICKET PART   
        # Prepare input rows for all ticket types, gather predictions per ticket
        predictions_per_ticket = {}
        for ticket_name in ticket_cols:
            ticket_part = {col: 0 for col in ticket_cols}
            ticket_part[ticket_name] = 1

            row = {
                "temperature": temperature,
                #"rain": rain,
                "precipitation": precipitation,
                **holiday_part,
                **ticket_part
            }

            input_df = pd.DataFrame([row])

            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
                input_df = input_df.reindex(columns=expected_features, fill_value=0)

            predicted_total = model.predict(input_df)[0]
            predictions_per_ticket[ticket_name] = predicted_total

        total_visitors = sum(predictions_per_ticket.values())

        if (current_date.day == 31 and current_date.month == 12) or (current_date.day == 1 and current_date.month == 1):
            print("oud en nieuwjaarsdag")
            daily_summary = {
                "date": current_date,
                "temperature": float(round(temperature, 1)),
                #"rain": float(round(rain, 1)),
                "precipitation": float(round(precipitation, 1)),
                "total_visitors": 0,
                **{ticket: 0 for ticket in ticket_cols},
                **holiday_part , # Include holiday columns for display on dashboard
                "year": current_date.year,
                "month": current_date.month,
                "week": current_date.isocalendar().week,
                "day": current_date.day,
                "weekday": current_date.weekday()
        }    
        else:
            print("normal cases")
            daily_summary = {
                "date": current_date,
                "temperature": round(temperature, 1),
                #"rain": round(rain, 1),
                "precipitation": round(precipitation, 1),
                "total_visitors": round(total_visitors, 0),
                **predictions_per_ticket,
                **holiday_part , # Include holiday columns for display on dashboard
                "year": current_date.year,
                "month": current_date.month,
                "week": current_date.isocalendar().week,
                "day": current_date.day,
                "weekday": current_date.weekday()
            }
        
        all_days_rows.append(daily_summary)
        # for day_dict in all_days_rows:
        #     print(f"Date: {day_dict["date"]}, {day_dict["total_visitors"]}")
        print(
            f"{daily_summary['date']} | Temp: {daily_summary['temperature']:.1f}°C | "
            #f"Rain: {daily_summary['rain']:.1f} | "
            f"Precipitation: {daily_summary['precipitation']:.1f} | "
            f"Total Visitors: {daily_summary['total_visitors']:.0f}"
        )

        # Suppose 'ticket_cols' is your list of ticket type column names

        # for day_dict in all_days_rows[:10]:  # First 10 days
        #     print(f"Date: {day_dict['date']}")
        #     # Collect ticket types and their predictions for the day
        #     ticket_predictions = [(ticket, day_dict[ticket]) for ticket in ticket_cols]
        #     # Sort by prediction value, descending
        #     ticket_predictions.sort(key=lambda x: x[1], reverse=True)
        #     # Print sorted results
        #     for ticket, value in ticket_predictions:
        #         print(f"  {ticket}: {value:.1f}")
        #     print("-" * 30)

        


            # Round all relevant columns to one decimal
        pd.options.display.float_format = "{:.1f}".format

            # Set float columns
        float_cols = ["temperature", "precipitation", "total_visitors"] + ticket_cols  
        df = pd.DataFrame(all_days_rows)
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df["date"] = df["date"].astype(str)

        #Remove 'rain' from column order for CSV
        column_order = [
            "date", "temperature", "precipitation", "total_visitors",
            *ticket_cols, *holiday_cols,
            "year", "month", "week", "day", "weekday"
        ]

# PROMOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO

#  if any of the event is not 0:
#        for loop through the events and set all to 0
#        predict again
#        compare the difference
#        add the difference to "event_impact" column
# else:
#        set "event_impact" column to 0

    # Save all days in one CSV file for dashboard ease
    df = df.reindex(columns=column_order)
    output_file = os.path.join(OUTPUT_DIR, "pre - rain temp app predictions_365days.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(all_days_rows).to_csv(output_file, index=False, float_format="%.1f")
    print(f"✅ Saved all 365-day predictions to {output_file}")
    print("✅ All 365-day predictions complete.")


if __name__ == "__main__":
    predict_next_365_days()



# TODO 
"""
- events
- promo/campaign
- wildnights/wildsummer fix
- precipitation / rain fix


"""