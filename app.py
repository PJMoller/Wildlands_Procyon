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
        "hourly": ["temperature_2m", "precipitation", "rain"],
        "forecast_days": 16 
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # processing the location
    response = responses[0]
    # print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation: {response.Elevation()} m asl")
    # print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True).date(),
            periods = daily.Variables(0).ValuesLength(),
            freq = "D"
    ),
    "temp_max": daily.Variables(0).ValuesAsNumpy(),
    "temp_min": daily.Variables(1).ValuesAsNumpy(),
    "precipitation": daily.Variables(2).ValuesAsNumpy()
    }

    df_weather = pd.DataFrame(daily_data)
    df_weather["temperature_2m"] = (df_weather["temp_max"] + df_weather["temp_min"]) / 2
    df_weather.drop(columns=["temp_max", "temp_min"], inplace=True)
    
    print("\nDaily data\n", df_weather)

    return df_weather
    



def seperating(processed_df):

    """
    for loop for looking into 3 cattegorries
    1: api (temperature, rain, percipitation)
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

        if col == "temperature" or col == "rain" or col == "percipitation":
            #print("API")
            api.append(col)
        elif col.startswith("ticket_"):
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
    
    api_cols, ticket_cols, holiday_cols = seperating(processed_df)

    avg_weather = (
        processed_df.groupby(["month", "day"])[["temperature", "rain", "precipitation"]]
        .mean()
        .reset_index
    )

    print("starting 365 pred loop")

    for d in range(365):
        current_date = DATE + timedelta(days=d)
        csv_filename = os.path.join(OUTPUT_DIR, f"predictions_{current_date}.csv")

        if os.path.exists(csv_filename):
            print(f"{current_date}, already predicted - skipping ")
            continue

    # API PART #
    match = openmeteo_df[openmeteo_df["date"].dt.date == current_date]
    if not match.empty:
        api_part = {
            "temperature": match["temperature_2m"].iloc[0],
            "rain": match["rain"].iloc[0],
            "precipitation": match["precipitation"].iloc[0]
        }
    else:
        avg = avg_weather[
            (avg_weather["month"] == current_date.month)
            & (avg_weather["day"] == current_date.day)
        ]
    api_part = {
        "temperatur": avg["temperature"].iloc[0] if not avg.empty else np.nan,
        "rain": avg["rain"].iloc[0] if not avg.empty else np.nan,
        "precipitation": avg["precipitation"].iloc[0] if not avg.empty else np.nan,
    }

    # HOLIDAY PART # 
    match_holiday = holidays_df[holidays_df["date"] == current_date]
    if match_holiday.empty:
        print(f"please update holidays file, cant find {current_date}")
        break

    holiday_part = match_holiday.iloc[0].to_dict()
    if "date" in holiday_part:
        del holiday_part["date"]

    holiday_part.update({
        "year": current_date.year,
        "month": current_date.month,
        "day": current_date.day,
        "weekday": current_date.weekday(),
    })

    # PROMOTION PART #
    ##########################
    # PROMOTION WILL BE HERE #
    ##########################


    # TICKET PART # 
    num_tickets = len(ticket_cols)
    bit_pattern = 1
    rows = []

    for _ in range(num_tickets):
        bits = list(map(int, f"{bit_pattern:0{num_tickets}b}"))
        ticket_part = dict(zip(ticket_cols, bits))

        full_row = {**api_part, **ticket_part, **holiday_part}
        input_df = pd.DataFrame([full_row])
        prediction = model.predict(input_df)[0]

        rows.append({"date": current_date, "prediction": prediction, **full_row})
        bit_patternpattern <<= 1

    daily_df = pd.DataFrame(rows)
    daily_df.to_csv(csv_filename, index=False)
    print(f"saved {len(rows)} predictions for {current_date}")

print("all 365 days predictions complete")


if __name__ == "__main__":
    get_openmeteo()
    seperating()
    #api_part()






















# # Helper: get DB connection
# def get_db_connection():
#     conn = sqlite3.connect(DB_FILE)
#     conn.row_factory = sqlite3.Row
#     return conn

# # Route: predict for chosen day
# @app.route("/api/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()
#         features = np.array(data["features"]).reshape(1, -1)
#         prediction = model.predict(features)

#         # Save to DB
#         conn = get_db_connection()
#         conn.execute(
#             "INSERT INTO predictions (features, result, date, timestamp) VALUES (?, ?, ?, ?)",
#             (str(data["features"]), str(prediction.tolist()), data.get("date", None), datetime.now().isoformat())
#         )
#         conn.commit()
#         conn.close()

#         return jsonify({"result": prediction.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# # Route: get latest prediction
# @app.route("/api/latest", methods=["GET"])
# def latest():
#     conn = get_db_connection()
#     row = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
#     conn.close()
#     if row:
#         return jsonify({
#             "features": row["features"],
#             "result": row["result"],
#             "date": row["date"],
#             "timestamp": row["timestamp"]
#         })
#     return jsonify({})

# # Route: get all predictions
# @app.route("/api/history", methods=["GET"])
# def history():
#     conn = get_db_connection()
#     rows = conn.execute("SELECT * FROM predictions ORDER BY id ASC").fetchall()
#     conn.close()
#     results = [{"features": r["features"], "result": r["result"], "date": r["date"], "timestamp": r["timestamp"]} for r in rows]
#     return jsonify(results)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)