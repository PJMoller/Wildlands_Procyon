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


# ----------------------------------------------------------
#  API: Expected attendance ranges (week, month, year)
# ----------------------------------------------------------
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

    # WEEK
    if range_type == "week":
        start = selected_date - pd.Timedelta(days=selected_date.weekday())
        end = start + pd.Timedelta(days=6)

    # MONTH
    elif range_type == "month":
        start = selected_date.replace(day=1)
        end = (start + pd.offsets.MonthEnd(1)).normalize()

    # YEAR
    elif range_type == "year":
        start = selected_date.replace(month=1, day=1)
        end = selected_date.replace(month=12, day=31)

    # DAY (used by pie chart)
    elif range_type == "day":
        start = selected_date
        end = selected_date

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


# ----------------------------------------------------------
#  API: Today & Tomorrow widgets
# ----------------------------------------------------------
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


# ----------------------------------------------------------
#  API: Ticket breakdown for selected day (pie chart)
# ----------------------------------------------------------
@app.route("/api/day-tickets")
def day_tickets():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "Missing date parameter"}), 400

    try:
        selected_date = pd.to_datetime(date_str).normalize()
    except:
        return jsonify({"error": "Invalid date format"}), 400

    # Filter row for that day
    row = df[df["date"] == selected_date]
    if row.empty:
        return jsonify({"error": "No data for this date"}), 404

    row = row.iloc[0]

    # Ticket columns = all columns containing "ticket_"
    ticket_columns = [col for col in df.columns if col.startswith("ticket_")]

    ticket_values = {col.replace("ticket_", ""): int(row[col]) if not pd.isna(row[col]) else 0
                     for col in ticket_columns}

    total_visitors = int(row["total_visitors"]) if not pd.isna(row["total_visitors"]) else 0

    return jsonify({
        "date": selected_date.strftime("%Y-%m-%d"),
        "tickets": ticket_values,
        "total_visitors": total_visitors
    })


# ----------------------------------------------------------
#  Run the app
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
