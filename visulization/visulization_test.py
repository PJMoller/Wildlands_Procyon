import pandas as pd
import plotly.express as px
import plotly.io as pio
from datetime import timedelta

# Load CSV (it’s huge, so low_memory=False avoids warnings)
df = pd.read_csv("./data/predictions/app_predictions_365days.csv", low_memory=False)

# Keep only the columns we care about
df = df[['date', 'total_visitors']].dropna()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Create folder for charts
import os
os.makedirs("charts", exist_ok=True)

def generate_chart(start_date, period):
    start = pd.to_datetime(start_date)

    if period == "week":
        end = start + timedelta(days=6)
    elif period == "month":
        end = start + pd.DateOffset(months=1)
    elif period == "year":
        end = start + pd.DateOffset(years=1)
    else:
        raise ValueError("Invalid period")

    # Filter data
    mask = (df['date'] >= start) & (df['date'] <= end)
    df_filtered = df.loc[mask]

    if df_filtered.empty:
        print(f"No data for {period} starting {start_date}")
        return

    # Create chart
    fig = px.bar(
        df_filtered,
        x='date',
        y='total_visitors',
        labels={'total_visitors': 'Total Visitors', 'date': 'Date'},
    )

    # Layout styling
    fig.update_layout(
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="black", family="Arial", size=14),
        title=dict(
            text=f"Total Visitors ({period.capitalize()} starting {start.date()})",
            x=0.5,
            font=dict(size=20, color="black")
        )
    )

    fig.update_xaxes(color="black", tickfont=dict(color="black", size=12), title_font=dict(color="black"))
    fig.update_yaxes(color="black", tickfont=dict(color="black", size=12), title_font=dict(color="black"))
    fig.update_traces(marker_line_width=1.5, marker_line_color='white', opacity=0.9)

    # Save as standalone HTML snippet
    output_path = f"charts/chart_{period}_{start_date}.html"
    pio.write_html(fig, file=output_path, full_html=False, include_plotlyjs="cdn")
    print(f"✅ Saved {output_path}")

# Example generation
generate_chart("2025-11-10", "week")
generate_chart("2025-11-01", "month")
generate_chart("2026-01-01", "year")
