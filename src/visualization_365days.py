import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import paths
from src.paths import PREDICTIONS_DIR, IMG_DIR

def visualize_forecast():
    """Create comprehensive visualizations of the 365-day forecast."""

    # Find the most recent forecast file
    forecast_files = sorted(PREDICTIONS_DIR.glob("forecast_365days_*.csv"), reverse=True)
    if not forecast_files:
        print("Error: No forecast files found in predictions directory")
        return

    forecast_file = forecast_files[0]
    # forecast_file = PREDICTIONS_DIR / "forecast_365days_from_20260101_20260113_1725.csv"
    print(f"Visualizing: {forecast_file.name}")

    # Load forecast data
    df = pd.read_csv(forecast_file)
    df['date'] = pd.to_datetime(df['date'])

    # Calculate daily totals
    daily_totals = df.groupby('date')['predicted_sales'].sum().reset_index()
    daily_totals.columns = ['date', 'total_sales']

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # ============= PLOT 1: Daily Sales Over Time =============
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(daily_totals['date'], daily_totals['total_sales'], 
             linewidth=2, color='#2E86AB', alpha=0.8)
    ax1.fill_between(daily_totals['date'], daily_totals['total_sales'], 
                      alpha=0.3, color='#2E86AB')

    # Highlight Christmas period
    christmas_mask = (daily_totals['date'].dt.month == 12) & (daily_totals['date'].dt.day.between(18, 24))
    ax1.scatter(daily_totals[christmas_mask]['date'], 
               daily_totals[christmas_mask]['total_sales'],
               color='red', s=100, zorder=5, label='Christmas Week', alpha=0.7)

    ax1.set_title('Daily Sales Forecast - 365 Days', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Daily Sales', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Add average line
    avg_daily = daily_totals['total_sales'].mean()
    ax1.axhline(y=avg_daily, color='gray', linestyle='--', 
                label=f'Average: {int(avg_daily):,}/day', alpha=0.7)
    ax1.legend()

    # ============= PLOT 2: Monthly Totals =============
    ax2 = plt.subplot(3, 2, 2)
    monthly_totals = df.groupby(df['date'].dt.to_period('M'))['predicted_sales'].sum()
    months = [p.to_timestamp() for p in monthly_totals.index]

    colors = ['#06A77D' if m.month in [7, 8] else '#D00000' if m.month == 12 
              else '#F77F00' if m.month in [4, 5, 10] else '#669BBC' 
              for m in months]

    bars = ax2.bar(months, monthly_totals.values, color=colors, alpha=0.8, width=20)
    ax2.set_title('Monthly Sales Totals', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month', fontsize=11)
    ax2.set_ylabel('Total Sales', fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height/1000)}k',
                ha='center', va='bottom', fontsize=9)

    # ============= PLOT 3: Weekly Moving Average =============
    ax3 = plt.subplot(3, 2, 3)
    daily_totals['7d_avg'] = daily_totals['total_sales'].rolling(window=7, center=True).mean()
    daily_totals['30d_avg'] = daily_totals['total_sales'].rolling(window=30, center=True).mean()

    ax3.plot(daily_totals['date'], daily_totals['7d_avg'], 
             linewidth=2, color='#06A77D', label='7-Day Average', alpha=0.8)
    ax3.plot(daily_totals['date'], daily_totals['30d_avg'], 
             linewidth=2, color='#F77F00', label='30-Day Average', alpha=0.8)

    ax3.set_title('Moving Averages - Trend Analysis', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Average Daily Sales', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # ============= PLOT 4: Day of Week Pattern =============
    ax4 = plt.subplot(3, 2, 4)
    daily_totals['day_of_week'] = daily_totals['date'].dt.dayofweek
    dow_avg = daily_totals.groupby('day_of_week')['total_sales'].mean()

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    colors_dow = ['#2E86AB' if i < 5 else '#06A77D' for i in range(7)]

    bars_dow = ax4.bar(days, dow_avg.values, color=colors_dow, alpha=0.8)
    ax4.set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Day of Week', fontsize=11)
    ax4.set_ylabel('Average Sales', fontsize=11)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars_dow:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9)

    # ============= PLOT 5: Family-Level Breakdown =============
    ax5 = plt.subplot(3, 2, 5)
    family_daily = df.groupby(['date', 'ticket_family'])['predicted_sales'].sum().reset_index()

    families = family_daily['ticket_family'].unique()
    for i, family in enumerate(families):
        family_data = family_daily[family_daily['ticket_family'] == family]
        ax5.plot(family_data['date'], family_data['predicted_sales'], 
                linewidth=2, label=family.title(), alpha=0.8)

    ax5.set_title('Daily Sales by Ticket Family', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Date', fontsize=11)
    ax5.set_ylabel('Daily Sales', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

    # ============= PLOT 6: Key Metrics Summary =============
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')

    # Calculate key metrics
    total_sales = daily_totals['total_sales'].sum()
    avg_daily_sales = daily_totals['total_sales'].mean()
    max_day = daily_totals.loc[daily_totals['total_sales'].idxmax()]
    min_day = daily_totals.loc[daily_totals['total_sales'].idxmin()]

    # Christmas week stats
    christmas_data = daily_totals[christmas_mask]
    christmas_avg = christmas_data['total_sales'].mean() if not christmas_data.empty else 0

    # Low season stats
    low_season_mask = daily_totals['date'].dt.month.isin([1, 2, 11])
    low_season_avg = daily_totals[low_season_mask]['total_sales'].mean()

    # Peak season stats
    peak_season_mask = daily_totals['date'].dt.month.isin([7, 8])
    peak_season_avg = daily_totals[peak_season_mask]['total_sales'].mean()

    metrics_text = f"""KEY FORECAST METRICS

OVERALL STATISTICS
Total Predicted Sales: {int(total_sales):,}
Average Daily Sales: {int(avg_daily_sales):,}

EXTREMES
Highest Day: {max_day['date'].strftime('%Y-%m-%d')}
             {int(max_day['total_sales']):,} tickets

Lowest Day:  {min_day['date'].strftime('%Y-%m-%d')}
             {int(min_day['total_sales']):,} tickets

CHRISTMAS WEEK (Dec 18-24)
Average: {int(christmas_avg):,}


LOW SEASON (Jan, Feb, Nov)
Average: {int(low_season_avg):,}


PEAK SEASON (Jul, Aug)
Average: {int(peak_season_avg):,}
"""

    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle('365-Day Sales Forecast Analysis', 
                fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save the figure
    output_file = IMG_DIR / f"forecast_visualization_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")

    plt.show()

    return output_file


if __name__ == "__main__":
    print("=" * 80)
    print("365-DAY FORECAST VISUALIZATION")
    print("=" * 80)
    visualize_forecast()
    print("\nVisualization complete!")