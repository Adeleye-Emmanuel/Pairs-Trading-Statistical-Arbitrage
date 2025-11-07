import os
import argparse
from typing import List
import pandas as pd
import yfinance as yf

def download_etfs(etf_list: List[str], start_date: str, end_date: str, interval: str, output_dir: str):
    """
    Downloads historical data for a list of ETFs and saves them as CSV files.

    Parameters:
    etf_list (List[str]): List of ETF ticker symbols.
    start_date (str): Start date for historical data in 'YYYY-MM-DD' format.
    end_date (str): End date for historical data in 'YYYY-MM-DD' format.
    interval (str): Data interval (e.g., '1d', '1wk', '1mo').
    output_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) # Ensure the directory exists

    downloaded_etfs = []
    for etf in etf_list:
        print(f"Downloading data for {etf}...")
        
        try:
            data = yf.download(etf, start=start_date, end=end_date, interval=interval) # Download data using yfinance
        except Exception as e:
            print(f"Error downloading {etf}: {e}")
            continue
        downloaded_etfs.append(etf)
        try:
            prices = data['Adj Close']
        except Exception: # Fallback if 'Adj Close' is not available
            if "Close" in data.columns:
                prices = data['Close']
            else:
                prices = data.copy()
        if not prices.empty: # Check if data is not empty
            file_path = os.path.join(output_dir, f"{etf}.csv") # Define file path
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=f"{etf} Close") # Convert Series to DataFrame for consistency       
            prices.index = pd.to_datetime(prices.index) # Ensure index is datetime
            prices.sort_index(inplace=True) # Sort index
            prices.to_csv(file_path) # Save to CSV
            print(f"Saved {etf} data to {file_path}")
        else:
            print(f"No data found for {etf}")

    print(f"Download completed for {len(etf_list)} ETFs. Merging data...")    
    full_df = pd.DataFrame()
    for etf in downloaded_etfs:
        file_path = os.path.join(output_dir, f"{etf}.csv")
        if os.path.exists(file_path):
            etf_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if full_df.empty:
                full_df = etf_data
            else:
                full_df = full_df.join(etf_data, how='outer') # Merge dataframes on index
    
    print(f"Data merging completed. Combined DataFrame shape: {full_df.shape}")

    combined_file_path = os.path.join(output_dir, "combined_etfs.csv")
    full_df.to_csv(combined_file_path) # Save combined dataframe
    print(f"Combined data saved to {combined_file_path}")
    return full_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historical ETF data and save as CSV files.")

    # Define the 100 tickers here:
    etf_tickers = [
        "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IWDA.L", "VXUS", "ACWI", "BND", 
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "ADBE", "CRM", "INTC", 
        "JPM", "BAC", "WFC", "CAT", "GE", "MMM", "BA", "IBM", "HON",
        "JNJ", "PFE", "UNH", "PG", "KO", "PEP", "WMT", "CVS", "MRK", "ABBV",
        "XOM", "CVX", "SLB", "LIN", "BHP", "VALE", "RIO", "ECL", "APD",
        "XLK", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLU",
        "EFA", "EEM", "FXI", "EWJ", "VEU", "VWO", "EWG", "EWH", "EWC", "EWA",
        "AGG", "LQD", "TLT", "SHY", "GLD", "SLV", "USO", "UNG", "DBC", "PALL",
        "MDY", "IJR", "ARKK", "UBER", "DIS", "NFLX", "SBUX", "COST", "TGT", "HD"
        "GILD", "CMCSA", "BILI", "VIZ",
        "EWZ", "INDA", "RSX", "EWW", 
        "HYG", "TIP",
        "DBA", "WOOD",
        "VNQ", "REM"
    ]

    parser.add_argument("--etfs", nargs='+', default=etf_tickers, help="List of ETF ticker symbols.")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date in 'YYYY-MM-DD' format.")
    parser.add_argument("--end", type=str, default=None, help="End date in 'YYYY-MM-DD' format.")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g., '1d', '1wk', '1mo').")
    parser.add_argument("--output_dir", type=str, default="./data/etfs", help="Directory to save the CSV files.")

    args = parser.parse_args()
    
    df = download_etfs(args.etfs, args.start, args.end, args.interval, args.output_dir)
    print("Combined DataFrame:")
    print(df.head())