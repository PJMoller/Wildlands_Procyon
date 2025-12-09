import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

# === SETTINGS ===
CSV_PATH = "./data/predictions/app_predictions_365days.csv"          # Path to your CSV file
OUTPUT_DIR = "charts"          # Folder for chart files
DATE = "2025-11-04"            # Date to visualize
TOP_N = 7                         # Number of ticket types before grouping
MAX_LABEL_LENGTH = 25          # Max length before shortening

# === PREPARE FOLDER ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("📊 Loading CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)

# === CLEAN DATE ===
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# === FILTER BY DATE ===
target_date = pd.to_datetime(DATE)
row = df[df['date'] == target_date]

if row.empty:
    print(f"❌ No data found for {DATE}")
    exit()

# === SELECT TICKET COLUMNS ===
ticket_cols = [col for col in df.columns if col.startswith("ticket_")]
ticket_data = row[ticket_cols].iloc[0]

# === FILTER NONZERO ===
ticket_data = ticket_data[ticket_data > 0]

if ticket_data.empty:
    print(f"❌ No ticket data for {DATE}")
    exit()

# === SORT AND GROUP ===
ticket_data = ticket_data.sort_values(ascending=False)

if len(ticket_data) > TOP_N:
    top_tickets = ticket_data.head(TOP_N)
    others_sum = ticket_data.iloc[TOP_N:].sum()
    ticket_data = pd.concat([top_tickets, pd.Series({"Overig": others_sum})])

# === CLEAN LABELS ===
ticket_data.index = ticket_data.index.str.replace("ticket_", "", regex=False)
ticket_data.index = ticket_data.index.str.replace("_", " ", regex=False)

# === SHORTEN LONG LABELS ===
def shorten_label(label, max_len=MAX_LABEL_LENGTH):
    return label if len(label) <= max_len else label[:max_len - 3] + "..."

labels = [shorten_label(lbl) for lbl in ticket_data.index]

# === CREATE PIE CHART ===
fig = px.pie(
    names=labels,
    values=ticket_data.values,
    title=f"Ticketverdeling op {DATE}",
    color_discrete_sequence=px.colors.qualitative.Safe
)

# === PIE SETTINGS ===
fig.update_traces(
    textinfo='label+percent',
    textfont_size=13,
    pull=[0.05]*len(ticket_data),
    textposition='inside'
)

# === LAYOUT ===
fig.update_layout(
    title_font=dict(size=22, color="black"),
    font=dict(color="black"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=40, t=80, b=40),
    legend=dict(
        font=dict(size=12, color="black"),
        orientation="v",
        x=1.05,
        y=1
    )
)

# === SAVE ===
output_path = os.path.join(OUTPUT_DIR, f"chart_day_{DATE}.html")
pio.write_html(fig, file=output_path, full_html=False, include_plotlyjs="cdn")

print(f"✅ Saved {output_path}")
