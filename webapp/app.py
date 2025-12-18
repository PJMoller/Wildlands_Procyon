from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
import sqlite3
import hashlib

app = Flask(__name__)

app.secret_key = os.environ.get("SECRET_KEY", "fallback_dev_key")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "database.db")

def check_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed))
    user = cur.fetchone()
    conn.close()
    return user

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("Loginpage.html")

@app.route("/login", methods=["POST"])
def login_submit():
    username = request.form.get("username")
    password = request.form.get("password")
    if check_user(username, password):
        session["user"] = username
        return redirect(url_for("home"))
    return render_template("Loginpage.html", error="Incorrect username or password.")


@app.route("/save_variables", methods=["POST"])
def save_variables():
    global saved_slider_values
    try:
        saved_slider_values = request.json
        print("\n--- Slider Variables Received ---")
        for key, value in saved_slider_values.items():
            print(f"{key.capitalize():<10}: {value}")
        print("---------------------------------\n")
        return jsonify({"status": "success", "saved": saved_slider_values})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "data", "upload")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "predictions", "app_predictions_365days.csv")

df = pd.read_csv(CSV_PATH, low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
df["date"] = df["date"].dt.normalize()

saved_slider_values = {}

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("Dashboard.html")

@app.route("/slider")
def slider():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("Slider.html")

@app.route("/api/visitors")
def get_visitors():
    range_type = request.args.get("range", "week")
    date_str = request.args.get("date")
    try:
        selected_date = pd.to_datetime(date_str).normalize()
    except:
        selected_date = df["date"].max()

    data = df.copy()

    if range_type == "week":
        start = selected_date - pd.Timedelta(days=selected_date.weekday())
        end = start + pd.Timedelta(days=6)
    elif range_type == "month":
        start = selected_date.replace(day=1)
        end = (start + pd.offsets.MonthEnd(1)).normalize()
    elif range_type == "year":
        start = selected_date.replace(month=1, day=1)
        end = selected_date.replace(month=12, day=31)
    elif range_type == "day":
        start = end = selected_date
    else:
        start = df["date"].max() - pd.Timedelta(days=6)
        end = df["date"].max()

    data = data[(data["date"] >= start) & (data["date"] <= end)]

    return jsonify({
        "dates": data["date"].dt.strftime("%Y-%m-%d").tolist(),
        "visitors": data["total_visitors"].fillna(0).tolist()
    })

@app.route("/api/today")
def today_info():
    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    today_row = df[df["date"] == today]
    tomorrow_row = df[df["date"] == tomorrow]

    return jsonify({
        "today": {
            "date": today.strftime("%A, %b %d"),
            "visitors": int(today_row["total_visitors"].iloc[0]) if not today_row.empty else 0
        },
        "tomorrow": {
            "date": tomorrow.strftime("%A, %b %d"),
            "visitors": int(tomorrow_row["total_visitors"].iloc[0]) if not tomorrow_row.empty else 0
        }
    })

@app.route("/api/day-tickets")
def day_tickets():
    date_str = request.args.get("date")
    selected_date = pd.to_datetime(date_str).normalize()

    row = df[df["date"] == selected_date]
    if row.empty:
        return jsonify({"error": "No data"}), 404

    row = row.iloc[0]
    ticket_columns = [c for c in df.columns if c.startswith("ticket_")]

    tickets = {
        c.replace("ticket_", ""): int(row[c]) if not pd.isna(row[c]) else 0
        for c in ticket_columns
    }

    return jsonify({
        "date": selected_date.strftime("%Y-%m-%d"),
        "tickets": tickets,
        "total_visitors": int(row["total_visitors"])
    })

@app.route("/api/upload-status")
def upload_status():
    files = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
    ]
    return jsonify({"files": files})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return redirect(url_for("home"))

    return "Invalid file type", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
