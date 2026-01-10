from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import pandas as pd
import os
from werkzeug.utils import secure_filename
import sqlite3
import hashlib
from paths import PREDICTIONS_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "database.db")

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "data", "upload")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Latest prediction file
files = [f for f in PREDICTIONS_DIR.iterdir() if f.is_file()]
latest_file = max(files, key=lambda f: f.stat().st_mtime)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_dev_key")


# ------------------ DB ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


def check_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    hashed = hashlib.sha256(password.encode()).hexdigest()
    cur.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, hashed)
    )
    user = cur.fetchone()
    conn.close()
    return user


# ------------------ AUTH ------------------
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


# ------------------ DATA LOADING ------------------
df = None

def load_df():
    raw = pd.read_csv(latest_file, low_memory=False)

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])
    raw["date"] = raw["date"].dt.normalize()

    ticket_pivot = (
        raw
        .pivot_table(
            index="date",
            columns="ticket_name",
            values="predicted_sales",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    ticket_pivot.columns = [
        "date" if c == "date" else f"ticket_{c}"
        for c in ticket_pivot.columns
    ]

    ticket_cols = [c for c in ticket_pivot.columns if c.startswith("ticket_")]
    ticket_pivot["total_visitors"] = ticket_pivot[ticket_cols].sum(axis=1)

    return ticket_pivot.sort_values("date")


def get_df():
    global df
    if df is None:
        df = load_df()
    return df


# ------------------ ROUTES ------------------
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

    df_local = get_df()

    try:
        selected_date = pd.to_datetime(date_str).normalize()
    except:
        selected_date = df_local["date"].max()

    if range_type == "week":
        start = selected_date - pd.Timedelta(days=selected_date.weekday())
        end = start + pd.Timedelta(days=6)
    elif range_type == "month":
        start = selected_date.replace(day=1)
        end = (start + pd.offsets.MonthEnd(1)).normalize()
    elif range_type == "year":
        start = selected_date.replace(month=1, day=1)
        end = selected_date.replace(month=12, day=31)
    else:
        start = end = selected_date

    data = df_local[(df_local["date"] >= start) & (df_local["date"] <= end)]

    return jsonify({
        "dates": data["date"].dt.strftime("%Y-%m-%d").tolist(),
        "visitors": data["total_visitors"].tolist()
    })


@app.route("/api/today")
def today_info():
    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    df_local = get_df()

    def visitors_for(d):
        row = df_local[df_local["date"] == d]
        return int(row["total_visitors"].iloc[0]) if not row.empty else 0

    return jsonify({
        "today": {
            "date": today.strftime("%A, %b %d"),
            "visitors": visitors_for(today)
        },
        "tomorrow": {
            "date": tomorrow.strftime("%A, %b %d"),
            "visitors": visitors_for(tomorrow)
        }
    })


@app.route("/api/day-tickets")
def day_tickets():
    date_str = request.args.get("date")
    selected_date = pd.to_datetime(date_str).normalize()

    df_local = get_df()
    row = df_local[df_local["date"] == selected_date]

    if row.empty:
        return jsonify({"error": "No data"}), 404

    row = row.iloc[0]
    ticket_columns = [c for c in df_local.columns if c.startswith("ticket_")]

    tickets = {
        c.replace("ticket_", ""): int(row[c])
        for c in ticket_columns
        if row[c] > 0
    }

    return jsonify({
        "date": selected_date.strftime("%Y-%m-%d"),
        "tickets": tickets,
        "total_visitors": int(row["total_visitors"])
    })


# ------------------ SLIDER PREDICTION (NEW, SAFE) ------------------
@app.route("/api/slider-predict", methods=["POST"])
def slider_predict():
    data = request.json

    date = pd.to_datetime(data["date"]).normalize()
    rain = float(data["rain"])
    temperature = float(data["temperature"])

    df_local = get_df()
    row = df_local[df_local["date"] == date]

    if row.empty:
        return jsonify({"error": "No prediction for this date"}), 404

    baseline = int(row["total_visitors"].iloc[0])

    temp_effect = 1 + (temperature - 15) * 0.015
    rain_effect = 1 - min(rain * 0.02, 0.6)

    multiplier = max(0.4, temp_effect * rain_effect)
    adjusted = int(round(baseline * multiplier))

    return jsonify({
        "date": date.strftime("%Y-%m-%d"),
        "baseline": baseline,
        "adjusted": adjusted,
        "multiplier": round(multiplier, 3)
    })


# ------------------ UPLOAD ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
