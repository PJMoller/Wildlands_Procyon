import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
PREDICTIONS_DIR = "data/predictions/"
BUDGET_FILE_PATH = "data/raw/budget.xlsx"

def compare_daily_totals_and_calibrate():
    print("--- Daily Total Sales: Forecast vs. Budget Comparison (Day-of-Week Calibration) ---")

    # --- 1. Load and Process Forecast ---
    print("\nStep 1: Loading latest forecast...")
    try:
        list_of_files = glob.glob(os.path.join(PREDICTIONS_DIR, 'forecast_365_days*.csv')) 
        if not list_of_files:
            print(f"ERROR: No forecast files found in '{PREDICTIONS_DIR}'")
            return
        latest_forecast_path = max(list_of_files, key=os.path.getctime)
        print(f"Found latest forecast: {os.path.basename(latest_forecast_path)}")
        
        forecast_df = pd.read_csv(latest_forecast_path)
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        # Group by date to get daily totals
        daily_forecast = (
            forecast_df.groupby('date')['predicted_sales']
            .sum()
            .reset_index()
            .rename(columns={'predicted_sales': 'total_forecast_sales'})
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading forecast: {e}")
        return

    # --- 2. Load and Process Budget ---
    print("\nStep 2: Loading budget file...")
    try:
        budget_df = pd.read_excel(BUDGET_FILE_PATH)
        budget_df['date'] = pd.to_datetime(budget_df['Datum'])
        
        daily_budget = (
            budget_df.groupby('date')['Budget']
            .sum()
            .reset_index()
            .rename(columns={'Budget': 'total_budget_sales'})
        )
    except Exception as e:
        print(f"CRITICAL ERROR loading budget: {e}")
        return

    # --- 3. Merge and Find Common Days ---
    print("\nStep 3: Merging data...")
    comparison_df = pd.merge(daily_forecast, daily_budget, on='date', how='inner')
    
    if comparison_df.empty:
        print("ERROR: No overlapping dates found.")
        return

    # --- 4. Calculate Day-of-Week Calibration Factors ---
    print("\nStep 4: Calculating calibration factors per Day of Week...")
    
    # Add day of week (Monday=0, Sunday=6)
    comparison_df['day_of_week'] = comparison_df['date'].dt.dayofweek
    
    # Group by day of week to find total budget and total forecast for each day
    dow_stats = comparison_df.groupby('day_of_week')[['total_budget_sales', 'total_forecast_sales']].sum()
    
    # Calculate factor k for each day: k_dow = Sum(Budget) / Sum(Forecast)
    dow_stats['k_factor'] = dow_stats['total_budget_sales'] / dow_stats['total_forecast_sales']
    
    # Handle potential division by zero or NaNs
    dow_stats['k_factor'] = dow_stats['k_factor'].fillna(1.0)
    
    print("\nCalculated Calibration Factors (Budget / Forecast):")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dow, row in dow_stats.iterrows():
        print(f"  {days[dow]}: {row['k_factor']:.2f}x multiplier")

    # --- 5. Apply Calibration ---
    # Map the factors back to the daily data
    comparison_df['k_factor_applied'] = comparison_df['day_of_week'].map(dow_stats['k_factor'])
    
    comparison_df['total_forecast_calibrated'] = (
        comparison_df['total_forecast_sales'] * comparison_df['k_factor_applied']
    )

        # --- 6. Calculate Differences & Metrics (WAPE & MAPE) ---
    comparison_df['diff_raw'] = comparison_df['total_forecast_sales'] - comparison_df['total_budget_sales']
    comparison_df['diff_calib'] = comparison_df['total_forecast_calibrated'] - comparison_df['total_budget_sales']
    
    mask_budget_pos = comparison_df['total_budget_sales'] > 0
    
    # --- WAPE Calculation (Sum of Errors / Sum of Budget) ---
    # WAPE is more stable for low-volume days than MAPE
    
    total_abs_error_raw = np.sum(np.abs(comparison_df['diff_raw']))
    total_budget_sum = comparison_df['total_budget_sales'].sum()
    
    wape_raw = (total_abs_error_raw / total_budget_sum) * 100 if total_budget_sum > 0 else np.nan

    total_abs_error_calib = np.sum(np.abs(comparison_df['diff_calib']))
    wape_calib = (total_abs_error_calib / total_budget_sum) * 100 if total_budget_sum > 0 else np.nan

    # --- MAPE Calculation (Mean of Daily % Errors) ---
    mape_raw = np.mean(
        np.abs(comparison_df.loc[mask_budget_pos, 'diff_raw']) 
        / comparison_df.loc[mask_budget_pos, 'total_budget_sales']
    ) * 100
    
    mape_calib = np.mean(
        np.abs(comparison_df.loc[mask_budget_pos, 'diff_calib']) 
        / comparison_df.loc[mask_budget_pos, 'total_budget_sales']
    ) * 100

    # Totals
    total_budget = comparison_df['total_budget_sales'].sum()
    total_fcst_raw = comparison_df['total_forecast_sales'].sum()
    total_fcst_calib = comparison_df['total_forecast_calibrated'].sum()

    print("\n--- Final Results ---")
    print(f"Total Budget:     {total_budget:,.0f}")
    print(f"Raw Forecast:     {total_fcst_raw:,.0f} (WAPE: {wape_raw:.2f}%, MAPE: {mape_raw:.2f}%)")
    print(f"Calib Forecast:   {total_fcst_calib:,.0f} (WAPE: {wape_calib:.2f}%, MAPE: {mape_calib:.2f}%)")
    
    pct_diff_calib = (total_fcst_calib - total_budget) / total_budget * 100
    print(f"Calib vs Budget Total Diff: {pct_diff_calib:+.2f}%")

    # --- 7. Save and Plot ---
    output_path = os.path.join(PREDICTIONS_DIR, 'daily_forecast_dow_calibrated.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed calibration file to: {output_path}")

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(comparison_df['date'], comparison_df['total_budget_sales'], label='Budget', color='black', alpha=0.7)
    plt.plot(comparison_df['date'], comparison_df['total_forecast_sales'], label='Raw Forecast', linestyle='--', alpha=0.5)
    plt.plot(comparison_df['date'], comparison_df['total_forecast_calibrated'], label='Calibrated Forecast', color='green', linewidth=2)
    
    plt.title("Forecast vs Budget: Before and After Day-of-Week Calibration")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(PREDICTIONS_DIR, 'forecast_calibration_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved comparison plot to: {plot_path}")
    plt.close()

if __name__ == "__main__":
    compare_daily_totals_and_calibrate()
