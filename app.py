from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
from datetime import datetime
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Open-Meteo API setup and retry if you get an error
cache_session = requests_cache.CachedSession(".cache", expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# setting up location and parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 52.7862,
    "longitude": 6.8917,
    "hourly": ["temperature_2m", "precipitation", "rain"] 
}
responses = openmeteo.weather_api(url, params=params)

# processing the location
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)


app = Flask(__name__)

# Loading the ML model
model = joblib.load("data/processed/model.pkl")

# Database file
DB_FILE = "data/database/database.db"

"""
for loop for looking into 3 cattegorries
1: api (temperature, rain, percipitation)
2: start with ticket_
3: holiday/special day
"""
# # loading the processed data
# processed_df = pd.read_csv("data/processed/processed_merge.csv")

# headers = list(processed_df.columns)
# #print(headers)

# for col in processed_df.columns:
#     #print(col)

#     if col == "temperature" or col == "rain" or col == "percipitation":
#         print("API")
#     elif col.startswith("ticket_"):
#         print("ticket_")
#     else:
#         if col.startswith("NLNoord"):
#             print("Noord")
#         elif col.startswith("NLMidden"):
#             print("Midden")
#         elif col.startswith("NLZuid"):
#             print("Zuid")
#         elif col.startswith("Niedersachsen"):
#             print("Niedersachsen")
#         elif col.startswith("Nordrhein-Westfalen"):
#             print("Nordrhein-Westfalen")
#         else:
#             print("overig")
    





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