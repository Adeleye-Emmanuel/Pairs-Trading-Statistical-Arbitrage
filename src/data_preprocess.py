import os
import argparse
import numpy as np
import pandas as pd

def preprocess_data(input_file: str, output_dir: str):

    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

    df = pd.read_csv(input_file, index_col=0, parse_dates=True) # Read the input CSV file
    df.sort_index(inplace=True) # Sort the dataframe by index (date)

    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    prices_cleaned = df.ffill() # Assuming prices hasn't changed since last know date
    prices_cleaned.dropna(axis=1, how='all', inplace=True) # Drop rows where all values are NaN
    price_cleaned_shape = prices_cleaned.shape
    #df.dropna(inplace=True) # Drop rows where all values are NaN

    log_returns = np.log(prices_cleaned / prices_cleaned.shift(1)).dropna(how="all") # Calculate log returns and drop NaN values
    log_returns_cleaned_shape = log_returns.shape
    print(f"Cleaned prices shape: {price_cleaned_shape}, Log returns shape: {log_returns_cleaned_shape}")

    price_path = os.path.join(output_dir, "cleaned_prices.csv")
    log_returns_path = os.path.join(output_dir, "log_returns.csv")

    df.to_csv(price_path)
    log_returns.to_csv(log_returns_path)

    print(f"💾 Saved clean prices to: {price_path}")
    print(f"💾 Saved log returns to: {log_returns_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ETF data to compute log returns.")
    parser.add_argument("--input_file", type=str, default="data/etfs/combined_etfs.csv", help="Path to the input CSV file with ETF prices.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save the processed files.")
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_dir)    