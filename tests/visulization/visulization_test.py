import pandas as pd
import plotly.express as px
from datetime import timedelta
import sys
import os

# === SETTINGS ===
INPUT_FILE = "./data/raw/visitordaily.csv"
CHART_DIR = "./tests/visulization/charts"
os.makedirs(CHART_DIR, exist_ok=True)
# ================

# Read command-line args
PERIOD = sys.argv[1] if len(sys.argv) > 1 else "week"  # week / month / year
START_DATE = pd.to_datetime(sys.argv[2]) if len(sys.argv) > 2 else pd.to_datetime("2017-10-01")

# Load CSV
df = pd.read_csv(INPUT_FILE, sep=";")
df["Date"] = pd.to_datetime(df["Date"])

# Combine all AccessGroupIds into total visitors per day
df_total = df.groupby("Date")["NumberOfUsedEntrances"].sum().reset_index()

# Determine date range
if PERIOD == "week":
    end_date = START_DATE + timedelta(days=6)
elif PERIOD == "month":
    end_date = START_DATE + pd.offsets.MonthEnd(1)
elif PERIOD == "year":
    end_date = START_DATE + pd.offsets.YearEnd(1)
else:
    raise ValueError("PERIOD must be one of: week, month, year")

# Filter for that range
mask = (df_total["Date"] >= START_DATE) & (df_total["Date"] <= end_date)
df_range = df_total.loc[mask]

# Create bar chart
fig = px.bar(
    df_range,
    x="Date",
    y="NumberOfUsedEntrances",
    title=f"Total Visitors ({PERIOD.capitalize()} starting {START_DATE.date()})",
    labels={"NumberOfUsedEntrances": "Total Visitors"},
)

fig.update_layout(
    title={
        'text': f"Total Visitors ({PERIOD.capitalize()} starting {START_DATE.date()})",
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=20, color='black')
    },
    xaxis_title="Date",
    yaxis_title="Total Visitors",
    template="plotly_white",
    autosize=True,
    margin=dict(l=50, r=30, t=80, b=50),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial, sans-serif", size=14, color="black"),  # ✅ main font color
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.1)",
    color="black",
    tickfont=dict(color="black")   # ✅ numbers and labels black
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.1)",
    color="black",
    tickfont=dict(color="black")
)


# ✅ Save chart with dynamic name
filename = f"chart_{PERIOD}_{START_DATE.date()}.html"
output_path = os.path.join(CHART_DIR, filename)
fig.write_html(output_path, include_plotlyjs="cdn", full_html=False)

print(f"✅ Chart saved to: {output_path}")
