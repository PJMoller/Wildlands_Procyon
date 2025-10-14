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

# Loading data and models
processed_df = pd.read_csv("data/processed/processed_merge.csv")
model = joblib.load("data/processed/model.pkl")
DB_FILE = "data/database/database.db"
DATE = date.today()
#print(DATE)

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
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

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
    
    #print("\nHourly data\n", hourly_dataframe)
    #current_date = hourly_dataframe["date"].iloc[0].date()

    return df_weather
    



def seperating():

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

def api_part():
    current_date = DATE
    for day in range(0,366):
        next_day = current_date + timedelta(days=day)
        #print(next_day)
        #try: 
            
    


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