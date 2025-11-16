import pandas as pd

PROCESSED_DATA_PATH = "data/processed/processed_merge.csv"
OUTPUT_SEASONALITY_PATH = "data/processed/ticket_seasonality.csv"

# A ticket must have at least this many total sales to get a profile.
MINIMUM_TOTAL_SALES = 50

# A month is considered "active" for a ticket if it accounts for at least
# this percentage of the ticket's total sales
# For a perfect year-round ticket, each month would be ~8.3% (1/12)
SIGNIFICANCE_THRESHOLD = 0.05


def create_ticket_seasonality_profile():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, usecols=['month', 'ticket_num'] + [col for col in pd.read_csv(PROCESSED_DATA_PATH, nrows=0).columns if col.startswith('ticket_')])
    except Exception as e:
        print(f"CRITICAL ERROR during file loading: {e}"); return

    print(f"\ntickets with significance > {SIGNIFICANCE_THRESHOLD:.0%}...")
    
    ticket_cols = [col for col in df.columns if col.startswith('ticket_')]
    seasonality_data = []

    for ticket_col in ticket_cols:
        ticket_name = ticket_col.replace('ticket_', '')
        ticket_df = df[df[ticket_col] == 1]
        
        total_sales = ticket_df['ticket_num'].sum()
        if total_sales < MINIMUM_TOTAL_SALES:
            print(f"  - Skipping '{ticket_name}': Insufficient sales ({int(total_sales)} total).")
            continue
            
        monthly_sales = ticket_df.groupby('month')['ticket_num'].sum()
        
        monthly_distribution = monthly_sales / total_sales
        
        significant_months = monthly_distribution[monthly_distribution > SIGNIFICANCE_THRESHOLD].index.tolist()

        if not significant_months:
            print(f"  - Skipping '{ticket_name}': No single month was significant enough.")
            continue

        significant_months.sort()
        active_months_str = ','.join(map(str, significant_months))
        
        seasonality_data.append({
            "ticket_name": ticket_name,
            "active_months": active_months_str
        })
        print(f"  - Profiled '{ticket_name}': Core months are [{active_months_str}]")

    if not seasonality_data:
        print("\nWARNING: No tickets met the criteria. No output file was created.")
        return

    print(f"\nStep 3: Saving seasonal profile for {len(seasonality_data)} tickets...")
    seasonality_df = pd.DataFrame(seasonality_data)
    seasonality_df.to_csv(OUTPUT_SEASONALITY_PATH, index=False)
    
    print(f"✅ Advanced seasonal profile saved successfully to '{OUTPUT_SEASONALITY_PATH}'")

if __name__ == "__main__":
    create_ticket_seasonality_profile()