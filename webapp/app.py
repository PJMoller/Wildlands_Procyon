from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_babel import Babel, gettext
import pandas as pd
import os
from werkzeug.utils import secure_filename
import sqlite3
import hashlib
from paths import PREDICTIONS_DIR

# ------------------ PATH FIX FOR ML ------------------
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.single_day_predict import predict_single_day 

# ------------------ PATHS ------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "data", "upload")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ FLASK ------------------
app = Flask(__name__)
app.config['LANGUAGES'] = {
    'en': 'English',
    'nl': 'Dutch',
    'de': 'German'
}

app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

app.secret_key = os.environ.get("SECRET_KEY", "fallback_dev_key")

babel = Babel(app)

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
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hashed)
    )
    user = cur.fetchone()
    conn.close()
    return user

@app.route('/set-language/<language>')
def set_language(language):
    print(f"Language selected: {language}")
    session['language'] = language
    print(f"Session language: {session.get('language')}")
    return redirect(request.referrer or '/')

def get_locale():
    locale = session.get('language')
    print(f"Current session language: {locale}")
    if locale:
        return locale
    return request.accept_languages.best_match(app.config['LANGUAGES'].keys()) or 'en'

babel.init_app(app, locale_selector=get_locale)

@app.context_processor
def inject_conf():
    return {
        'get_locale': get_locale
    }

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("Loginpage.html")

@app.route("/login", methods=["POST"])
def login_submit():
    if check_user(request.form["username"], request.form["password"]):
        session["user"] = request.form["username"]
        return redirect(url_for("home"))
    return render_template("Loginpage.html", error="Incorrect credentials")

# ------------------ LOAD DASHBOARD DATA ------------------
df = None

def load_df():
    files = [f for f in PREDICTIONS_DIR.iterdir() if f.is_file()]
    if not files:
        return pd.DataFrame(columns=["date", "total_visitors"])
        
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    raw = pd.read_csv(latest_file, low_memory=False)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()

    pivot = raw.pivot_table(
        index="date",
        columns="ticket_name",
        values="predicted_sales",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    pivot.columns = ["date" if c == "date" else f"ticket_{c}" for c in pivot.columns]
    ticket_cols = [c for c in pivot.columns if c.startswith("ticket_")]
    pivot["total_visitors"] = pivot[ticket_cols].sum(axis=1)

    if "temperature" in raw.columns and "total_rain" in raw.columns:
        weather_cols = raw.groupby("date")[["temperature", "total_rain"]].mean().reset_index()
        pivot = pivot.merge(weather_cols, on="date", how="left")

    return pivot.sort_values("date")

def get_df():
    global df
    if df is None:
        df = load_df()
    return df

# ------------------ PAGES ------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("Dashboard.html", _=gettext)

@app.route("/slider")
def slider():
    if "user" not in session:
        return redirect(url_for("login_page"))
    return render_template("Slider.html", _=gettext)

# ------------------ DASHBOARD APIS ------------------
@app.route("/api/visitors")
def visitors():
    df_local = get_df()
    range_param = request.args.get("range", "week")
    start_date_str = request.args.get("date")

    start_date = pd.to_datetime(start_date_str).normalize() if start_date_str else pd.Timestamp.today().normalize()

    if range_param == "week":
        end_date = start_date + pd.Timedelta(days=6)
    elif range_param == "month":
        end_date = (start_date + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    elif range_param == "year":
        end_date = (start_date + pd.DateOffset(years=1)) - pd.Timedelta(days=1)
    else:
        end_date = start_date + pd.Timedelta(days=6)

    mask = (df_local["date"] >= start_date) & (df_local["date"] <= end_date)
    df_filtered = df_local.loc[mask]

    return jsonify({
        "dates": df_filtered["date"].dt.strftime("%Y-%m-%d").tolist(),
        "visitors": df_filtered["total_visitors"].tolist()
    })

@app.route("/api/today")
def today():
    df_local = get_df()
    today_date = pd.Timestamp.today().normalize()
    tomorrow_date = today_date + pd.Timedelta(days=1)

    def get_info(d):
        r = df_local[df_local["date"] == d]
        if r.empty:
            return {"visitors": 0, "temperature": None, "rain": None}

        row = r.iloc[0]
        return {
            "visitors": int(row["total_visitors"]),
            "temperature": round(float(row.get("temperature", 0)), 1),
            "rain": round(float(row.get("total_rain", 0)), 1)
        }

    return jsonify({
        "today": get_info(today_date),
        "tomorrow": get_info(tomorrow_date)
    })

@app.route("/api/day-tickets")
def day_tickets():
    date = pd.to_datetime(request.args["date"]).normalize()
    row = get_df()[lambda x: x["date"] == date]

    if row.empty:
        return jsonify({})

    row = row.iloc[0]
    tickets = {
        c.replace("ticket_", ""): int(row[c])
        for c in row.index if c.startswith("ticket_")
    }

    return jsonify({
        "tickets": tickets,
        "total_visitors": int(row["total_visitors"])
    })

# ------------------ FILE UPLOAD ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if file and allowed_file(file.filename):
        file.save(os.path.join(UPLOAD_FOLDER, secure_filename(file.filename)))
    return redirect(url_for("home"))

# ------------------ SLIDER ML ------------------
@app.route("/api/slider-predict", methods=["POST"])
def slider_predict():
    try:
        payload = request.get_json()
        date_str = payload.get("date")

        date_obj = pd.to_datetime(date_str).normalize()
        df_local = get_df()
        base_row = df_local[df_local["date"] == date_obj]
        baseline = int(base_row["total_visitors"].iloc[0]) if not base_row.empty else 0

        r_morning = float(payload.get("rain_morning", 0.0))
        r_afternoon = float(payload.get("rain_afternoon", 0.0))
        p_morning = float(payload.get("precip_morning", 0.0))
        p_afternoon = float(payload.get("precip_afternoon", 0.0))

        adjusted_total = predict_single_day(
            input_date=date_str,
            temperature=float(payload.get("temperature", 15.0)),
            rain_morning=r_morning,
            rain_afternoon=r_afternoon,
            precip_morning=p_morning,
            precip_afternoon=p_afternoon,
            event_name=payload.get("event_name"),
            holiday_name=payload.get("holiday_name"),
        )

        return jsonify({
            "baseline": baseline,
            "adjusted": adjusted_total
        })

    except Exception as e:
        print(f"Error in slider_predict: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
