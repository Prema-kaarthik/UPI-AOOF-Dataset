
import pandas as pd
import numpy as np
from datetime import datetime
import os

# --------------------- CONFIG ---------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Example: Assume you downloaded NPCI monthly CSV and saved as "upi_monthly_npci.csv"
# Columns expected: Month, Volume_Mn, Value_Cr, Banks_Live, etc.
# For bank-wise, you may need to merge multiple files.

def load_and_prepare():
    # Load public NPCI-style data (replace filename with your downloaded file)
    df = pd.read_csv(f"{RAW_DIR}/upi_monthly_npci.csv", parse_dates=['Month'])
    df = df.sort_values('Month').reset_index(drop=True)
    
    # Create daily synthetic series for modeling (paper uses daily profiles)
    daily = []
    for _, row in df.iterrows():
        month_start = row['Month']
        days_in_month = pd.date_range(month_start, periods=30, freq='D')  # approx
        base_volume = row['Volume_Mn'] * 1e6 / 30  # daily avg in transactions
        for d in days_in_month:
            # Add realistic volatility + festival spikes
            noise = np.random.normal(0, 0.15 * base_volume)
            vol = max(100_000, base_volume + noise)
            daily.append({'Date': d, 'Volume': vol, 'Value_Cr': vol * 150 / 1e7})  # rough avg txn value
    
    daily_df = pd.DataFrame(daily)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # Temporal split as in paper
    train = daily_df[daily_df['Date'] < '2024-01-01'].copy()
    val   = daily_df[(daily_df['Date'] >= '2024-01-01') & (daily_df['Date'] < '2025-01-01')].copy()
    test  = daily_df[daily_df['Date'] >= '2025-01-01'].copy()
    
    train.to_csv(f"{PROCESSED_DIR}/train_2021-2023.csv", index=False)
    val.to_csv(f"{PROCESSED_DIR}/val_2024.csv", index=False)
    test.to_csv(f"{PROCESSED_DIR}/test_2025.csv", index=False)
    
    print(f"Processed: {len(train)} train, {len(val)} val, {len(test)} test rows")
    return train, val, test

if __name__ == "__main__":
    load_and_prepare()
