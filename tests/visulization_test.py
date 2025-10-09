import pandas as pd
import plotly.express as px

# Read CSV (note the semicolon separator)
df = pd.read_csv("./data/raw/visitordaily.csv", sep=";")

# Create a line chart of entrances per day per group
fig = px.line(
    df,
    x="Date",
    y="NumberOfUsedEntrances",
    color="Description",
    title="Entrances Over Time",
    markers=True
)

# Export chart as HTML
fig.write_html("chart.html")
