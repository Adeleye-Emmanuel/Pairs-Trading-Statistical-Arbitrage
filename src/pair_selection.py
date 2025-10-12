import os
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations

from config import ETFS_DIR, PROCESSED_DIR

def load_data():
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    log_returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    return prices, log_returns

def compute_correlations(log_returns):
    return log_returns.corr()

def filter_top_percentile(corr_matrix, percentile: int=95):
    # Get pairs of ETFs with correlation above the specified percentile for all tickers
    threshold = np.percentile(corr_matrix.values.flatten(), percentile)
    print(f"Correlation threshold for top {percentile} percentile: {threshold:.4f}")
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.values[i, j]) 
                       for i in range(corr_matrix.shape[0]) 
                       for j in range(i+1, corr_matrix.shape[1]) 
                       if corr_matrix.values[i, j] >= threshold]
    
    pairs_df = pd.DataFrame(high_corr_pairs, columns=['ETF1', 'ETF2', 'Correlation'])
    pairs_df = pairs_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
    return pairs_df

def perform_cointegration_test(prices, pairs_df, significance_level: float =0.05):
    coint_results = []
    for _, row in pairs_df.iterrows():
        etf1, etf2 = row["ETF1"], row["ETF2"]
        series1, series2 = prices[etf1].dropna(), prices[etf2].dropna()
        common_index = series1.index.intersection(series2.index)
        score, pvalue, _ = coint(series1.loc[common_index], series2.loc[common_index])
        if pvalue < significance_level:
            coint_results.append((etf1, etf2, row["Correlation"], pvalue))
    coint_df = pd.DataFrame(coint_results, columns=['ETF1', 'ETF2', 'Correlation', 'P-Value'])
    return coint_df.sort_values(by='P-Value').reset_index(drop=True)

def run_pair_selection_pipeline(percentile: int =95, significance_level: float =0.05):
    prices, log_returns = load_data()
    corr_matrix = compute_correlations(log_returns)
    top_pairs_df = filter_top_percentile(corr_matrix, percentile=percentile)
    coint_pairs_df = perform_cointegration_test(prices, top_pairs_df, significance_level=significance_level)
    
    print(f"Identified {len(coint_pairs_df)} cointegrated pairs out of {len(top_pairs_df)} high-correlation pairs.")

    return coint_pairs_df

if __name__ == "__main__":
    pairs = run_pair_selection_pipeline(percentile=95, significance_level=0.25)
    print("==== Top 10 Selected Pairs ===")
    print(pairs.head(10))
    output_path = os.path.join(PROCESSED_DIR, "selected_pairs.csv")
    pairs.to_csv(output_path, index=False)
    print(f"💾 Saved selected pairs to: {output_path}")