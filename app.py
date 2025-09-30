from flask import Flask, request, jsonify
import joblib
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load your ML model
model = joblib.load("model.pkl")

# Database file
DB_FILE = "data/database/database_sample.db"

# Helper: get DB connection
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

# Route: predict for chosen day
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)

        # Save to DB
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO predictions (features, result, date, timestamp) VALUES (?, ?, ?, ?)",
            (str(data["features"]), str(prediction.tolist()), data.get("date", None), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        return jsonify({"result": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Route: get latest prediction
@app.route("/api/latest", methods=["GET"])
def latest():
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    if row:
        return jsonify({
            "features": row["features"],
            "result": row["result"],
            "date": row["date"],
            "timestamp": row["timestamp"]
        })
    return jsonify({})

# Route: get all predictions
@app.route("/api/history", methods=["GET"])
def history():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id ASC").fetchall()
    conn.close()
    results = [{"features": r["features"], "result": r["result"], "date": r["date"], "timestamp": r["timestamp"]} for r in rows]
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
