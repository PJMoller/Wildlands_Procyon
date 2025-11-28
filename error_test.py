import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
PREDICTIONS_DIR = "data/predictions/"
BUDGET_FILE_PATH = "data/raw/budget.xlsx"



def compare_daily_totals():
    """
    Loads the latest forecast and the budget file, compares the total sum of sales
    for each day, and reports the difference, including daily averages.
    Also reports a simple calibrated forecast vs budget using a single scaling factor.
    """
    print("--- Daily Total Sales: Forecast vs. Budget Comparison ---")


    # --- 1. Load and Process Forecast File ---
    print("\nStep 1: Loading and processing the latest forecast file...")
    try:
        list_of_files = glob.glob(os.path.join(PREDICTIONS_DIR, '*.csv'))
        if not list_of_files:
            print(f"ERROR: No forecast files found in '{PREDICTIONS_DIR}'")
            return
        latest_forecast_path = max(list_of_files, key=os.path.getctime)
        print(f"Found latest forecast: {os.path.basename(latest_forecast_path)}")
        forecast_df = pd.read_csv(latest_forecast_path)
        
        forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date
        daily_forecast = (
            forecast_df.groupby('date')['predicted_sales']
            .sum()
            .reset_index()
        )
        daily_forecast.rename(
            columns={'predicted_sales': 'total_forecast_sales'},
            inplace=True
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading forecast file: {e}")
        return


    # --- 2. Load and Process Budget File ---
    print("\nStep 2: Loading and processing the budget file...")
    try:
        budget_df = pd.read_excel(BUDGET_FILE_PATH)
        budget_df['date'] = pd.to_datetime(budget_df['Datum']).dt.date
        daily_budget = (
            budget_df.groupby('date')['Budget']
            .sum()
            .reset_index()
        )
        daily_budget.rename(
            columns={'Budget': 'total_budget_sales'},
            inplace=True
        )
    except FileNotFoundError:
        print(f"CRITICAL ERROR: The budget file was not found at '{BUDGET_FILE_PATH}'")
        return
    except KeyError as e:
        print(f"ERROR: A required column is missing from the budget file: {e}")
        print("Please ensure your budget file has the columns 'Datum' and 'Budget'.")
        return
    except Exception as e:
        print(f"CRITICAL ERROR loading budget file: {e}")
        return


    # --- 3. Merge Daily Totals on Common Days ---
    print("\nStep 3: Finding common days...")
    comparison_df = pd.merge(daily_forecast, daily_budget, on='date', how='inner')

    if comparison_df.empty:
        print("\nERROR: No common dates were found between the forecast and the budget.")
        return
        
    # --- 4. Calculate Raw Differences and MAPE ---
    print("\nStep 4: Calculating and comparing total sales (raw forecast)...")
    comparison_df['difference_raw'] = (
        comparison_df['total_forecast_sales'] - comparison_df['total_budget_sales']
    )

    # Avoid division by zero for days with a budget of 0
    comparison_df['absolute_percentage_error_raw'] = np.where(
        comparison_df['total_budget_sales'] > 0,
        np.abs(comparison_df['difference_raw']) / comparison_df['total_budget_sales'],
        0
    ) * 100

    # Aggregate raw metrics
    total_forecast_raw = comparison_df['total_forecast_sales'].sum()
    total_budget = comparison_df['total_budget_sales'].sum()
    total_difference_raw = comparison_df['difference_raw'].sum()
    average_daily_difference_raw = comparison_df['difference_raw'].mean()
    mape_raw = comparison_df['absolute_percentage_error_raw'].mean()

    if total_budget > 0:
        percentage_diff_raw = (total_difference_raw / total_budget) * 100
    else:
        percentage_diff_raw = float('inf') if total_forecast_raw > 0 else 0

    # --- 4b. Calibrate forecast level to budget (single factor) ---
    if total_forecast_raw > 0:
        k = total_budget / total_forecast_raw
    else:
        k = np.nan

    comparison_df['total_forecast_sales_calibrated'] = (
        comparison_df['total_forecast_sales'] * k
        if not np.isnan(k) else comparison_df['total_forecast_sales']
    )
    comparison_df['difference_calibrated'] = (
        comparison_df['total_forecast_sales_calibrated'] - comparison_df['total_budget_sales']
    )
    comparison_df['absolute_percentage_error_calibrated'] = np.where(
        comparison_df['total_budget_sales'] > 0,
        np.abs(comparison_df['difference_calibrated']) / comparison_df['total_budget_sales'],
        0
    ) * 100

    total_forecast_cal = comparison_df['total_forecast_sales_calibrated'].sum()
    total_difference_cal = comparison_df['difference_calibrated'].sum()
    average_daily_difference_cal = comparison_df['difference_calibrated'].mean()
    mape_cal = comparison_df['absolute_percentage_error_calibrated'].mean()

    if total_budget > 0:
        percentage_diff_cal = (total_difference_cal / total_budget) * 100
    else:
        percentage_diff_cal = float('inf') if total_forecast_cal > 0 else 0

    # --- 5. Report the Results ---
    print("\n--- Comparison Report (RAW FORECAST) ---")
    print(f"\nComparison based on {len(comparison_df)} matching days found in both files.")
    print("--------------------------------------------------")
    print("               OVERALL PERFORMANCE")
    print("--------------------------------------------------")
    print(f"Total Predicted Sales: {int(total_forecast_raw):,}")
    print(f"Total Budgeted Sales:  {int(total_budget):,}")
    print(f"Overall Difference:    {int(total_difference_raw):,}")
    print(f"Percentage Difference: {percentage_diff_raw:+.2f}%")
    print("--------------------------------------------------")
    print("                DAILY AVERAGES")
    print("--------------------------------------------------")
    print(f"Average Daily Difference: {average_daily_difference_raw:+.2f} sales")
    print(f"Mean Absolute Pct Error (MAPE): {mape_raw:.2f}%")
    print("--------------------------------------------------")
    
    if percentage_diff_raw > 0:
        print("\nConclusion: The raw forecast is HIGHER than the budget on average.")
    else:
        print("\nConclusion: The raw forecast is LOWER than the budget on average.")

    print("\n--- Comparison Report (CALIBRATED FORECAST) ---")
    if not np.isnan(k):
        print(f"Calibration factor k (budget / forecast): {k:.2f}")
    else:
        print("Calibration factor k could not be computed (total_forecast_raw = 0).")
    print("--------------------------------------------------")
    print("               OVERALL PERFORMANCE")
    print("--------------------------------------------------")
    print(f"Total Calibrated Forecast: {int(total_forecast_cal):,}")
    print(f"Total Budgeted Sales:      {int(total_budget):,}")
    print(f"Overall Difference:        {int(total_difference_cal):,}")
    print(f"Percentage Difference:     {percentage_diff_cal:+.2f}%")
    print("--------------------------------------------------")
    print("                DAILY AVERAGES")
    print("--------------------------------------------------")
    print(f"Average Daily Difference: {average_daily_difference_cal:+.2f} sales")
    print(f"Mean Absolute Pct Error (MAPE): {mape_cal:.2f}%")
    print("--------------------------------------------------")

    output_path = os.path.join(PREDICTIONS_DIR, 'daily_total_forecast_vs_budget.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"\nA detailed daily comparison (raw + calibrated) has been saved to:\n{output_path}")



# 1) Load the comparison file
df = pd.read_csv(
    "data/predictions/daily_total_forecast_vs_budget.csv",
    parse_dates=["date"]
)

df["ratio_budget_over_forecast"] = (
    df["total_budget_sales"] / df["total_forecast_sales"]
)
# 3) Set date as index for nicer time-series plotting (optional)
df = df.set_index("date")

# 4) Plot the ratio
plt.figure(figsize=(12, 5))
df["ratio_budget_over_forecast"].plot()
plt.axhline(1.0, color="red", linestyle="--", label="ratio = 1.0")
plt.title("Daily ratio: budget / raw forecast")
plt.ylabel("Ratio")
plt.legend()
plt.tight_layout()

# 5) Save the plot to file
plt.savefig("data/predictions/daily_ratio_budget_over_forecast.png", dpi=150)
plt.close()


# --- RUN THE COMPARISON ---
if __name__ == "__main__":
    compare_daily_totals()
