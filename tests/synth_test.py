import pandas as pd

# Load processed data
df = pd.read_csv("../data/processed/processed_merge.csv")
df['date'] = pd.to_datetime(df['date'])

# Analysis 1: Ticket family presence by year
print("=== TICKET FAMILY PRESENCE BY YEAR ===")
family_presence = df.groupby(['ticket_family', df['date'].dt.year]).agg(
    days_available=('is_available', 'sum'),
    days_with_sales=('ticket_num', lambda x: (x > 0).sum()),
    total_sales=('ticket_num', 'sum'),
    avg_daily_sales=('ticket_num', 'mean')
).round(2)

print(family_presence)

# Analysis 2: Individual ticket breakdown
print("\n=== TOP 10 TICKETS BY YEAR ===")
ticket_detail = df.groupby(['ticket_name', 'ticket_family', df['date'].dt.year]).agg(
    days_available=('is_available', 'sum'),
    days_with_sales=('ticket_num', lambda x: (x > 0).sum()),
    total_sales=('ticket_num', 'sum'),
    avg_sales_per_active_day=('ticket_num', lambda x: x[x > 0].mean())
).round(2)

# Show top 10 tickets per year by total sales
for year in [2023, 2024, 2025]:
    print(f"\n{year} Top Tickets:")
    year_data = ticket_detail.xs(year, level='date').sort_values('total_sales', ascending=False).head(10)
    print(year_data[['ticket_family', 'total_sales', 'days_with_sales']])

# Analysis 3: Year-over-year comparison
print("\n=== YEAR-OVER-YEAR COMPARISON ===")
summary = df.groupby(df['date'].dt.year).agg(
    total_days=('date', 'nunique'),
    total_tickets_sold=('ticket_num', 'sum'),
    unique_tickets_active=('ticket_name', lambda x: x[df.loc[x.index, 'ticket_num'] > 0].nunique()),
    avg_sales_per_day=('ticket_num', 'mean')
).round(2)

print(summary)


