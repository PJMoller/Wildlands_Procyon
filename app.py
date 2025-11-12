from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)

# ---- Load CSV once at startup ----
df = pd.read_csv("Wildlands_Procyon/data/predictions/app_predictions_365days.csv", low_memory=False)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

print("CSV MIN DATE:", df['date'].min())
print("CSV MAX DATE:", df['date'].max())


@app.route("/")
def home():
    return render_template("Dashboard.html")


@app.route("/api/visitors")
def get_visitors():
    range_type = request.args.get("range", "week")
    date_str = request.args.get("date", None)

    try:
        selected_date = pd.to_datetime(date_str).normalize()
    except Exception:
        selected_date = df['date'].max().normalize()

    data_df = df.copy()
    data_df['date'] = data_df['date'].dt.normalize()

    # WEEK RANGE (default)
    if range_type == "week":
        start = selected_date - pd.Timedelta(days=selected_date.weekday())
        end = start + pd.Timedelta(days=6)

    # MONTH RANGE
    elif range_type == "month":
        start = selected_date.replace(day=1)
        end = (start + pd.offsets.MonthEnd(1)).normalize()

    # YEAR RANGE
    elif range_type == "year":
        start = selected_date.replace(month=1, day=1)
        end = selected_date.replace(month=12, day=31)

    # fallback
    else:
        start = data_df['date'].max() - pd.Timedelta(days=6)
        end = data_df['date'].max()

    data = data_df[(data_df['date'] >= start) & (data_df['date'] <= end)]
    data = data.sort_values('date')

    return jsonify({
        "dates": data['date'].dt.strftime("%Y-%m-%d").tolist(),
        "visitors": data['total_visitors'].fillna(0).tolist()
    })


@app.route("/api/today")
def today_info():
    today = pd.Timestamp.today().normalize()
    tomorrow = today + pd.Timedelta(days=1)

    data_today = df[df['date'] == today]
    data_tomorrow = df[df['date'] == tomorrow]

    today_visitors = int(data_today['total_visitors'].values[0]) if not data_today.empty else 0
    tomorrow_visitors = int(data_tomorrow['total_visitors'].values[0]) if not data_tomorrow.empty else 0

    return jsonify({
        "today": {
            "date": today.strftime("%A, %b %d"),
            "visitors": today_visitors
        },
        "tomorrow": {
            "date": tomorrow.strftime("%A, %b %d"),
            "visitors": tomorrow_visitors
        }
    })



if __name__ == "__main__":
    app.run(debug=True)
