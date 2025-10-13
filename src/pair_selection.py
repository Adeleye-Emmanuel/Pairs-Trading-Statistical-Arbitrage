import os
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations

from config import ETFS_DIR, PROCESSED_DIR
from copula_utils import copula_dependency_score

def load_data():
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    log_returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    return prices, log_returns

def compute_correlations(log_returns):
    return log_returns.corr()

def filter_top_percentile(corr_matrix, percentile, min_pairs=10, abs_min_corr=0.6):
    # Get pairs of ETFs with correlation above the specified percentile for all tickers
    vals = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    pct_thr = np.percentile(vals, percentile)
    threshold = max(pct_thr, abs_min_corr)
    print(f"Correlation threshold set at: {threshold:.4f} (Percentile: {percentile}th, Abs Min: {abs_min_corr})")

    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.values[i, j]) 
                       for i in range(corr_matrix.shape[0]) 
                       for j in range(i+1, corr_matrix.shape[1]) 
                       if corr_matrix.values[i, j] >= threshold]
    
    pairs_df = pd.DataFrame(high_corr_pairs, columns=['ETF1', 'ETF2', 'Correlation'])
    
    # fallback loop lowering percentile
    while len(high_corr_pairs) <= min_pairs and threshold > abs_min_corr:
        threshold -= 0.05
        high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.values[i, j]) 
                           for i in range(corr_matrix.shape[0]) 
                           for j in range(i+1, corr_matrix.shape[1]) 
                           if corr_matrix.values[i, j] >= threshold]
        print(f"Lowered threshold to {threshold:.4f}, found {len(high_corr_pairs)} pairs.")
        pairs_df = pd.DataFrame(high_corr_pairs, columns=['ETF1', 'ETF2', 'Correlation'])
       
        if len(high_corr_pairs) >= min_pairs:
            break
    
    pairs_df = pairs_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
    return pairs_df

def perform_cointegration_test(prices, pairs_df, significance_level):
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

def add_copula_scores(log_returns, coint_df):
    scores = []
    for _, row in coint_df.iterrows():
        etf1, etf2 = row["ETF1"], row["ETF2"]
        x, y = log_returns[etf1].values, log_returns[etf2].values
        scores.append(copula_dependency_score(x, y))
    coint_df['Copula_Score'] = scores
    coint_df = coint_df.sort_values(by='Copula_Score', ascending=False).reset_index(drop=True)
    return coint_df

def run_pair_selection_pipeline(percentile: int, significance_level: float, use_copula: bool):
    prices, log_returns = load_data()
    corr_matrix = compute_correlations(log_returns)
    top_pairs_df = filter_top_percentile(corr_matrix, percentile=percentile)
    coint_pairs_df = perform_cointegration_test(prices, top_pairs_df, significance_level=significance_level)
    
    print(f"Identified {len(coint_pairs_df)} cointegrated pairs out of {len(top_pairs_df)} high-correlation pairs.")
    if use_copula and not coint_pairs_df.empty:
        print("Computing copula dependency scores...")
        coint_cop_pairs_df = add_copula_scores(log_returns, coint_pairs_df)
        
    # Composite scoring of correlation, cointegration p-value, and copula score
        coint_cop_pairs_df['Composite_Score'] = (
            coint_cop_pairs_df['Correlation'] * 0.4 + 
            (1 - coint_cop_pairs_df['P-Value']) * 0.3 + # to invert p-value (lower is better)
            (coint_cop_pairs_df['Copula_Score']) * 0.3
        )
        pair_selection_scores = coint_cop_pairs_df.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)
    else:
        coint_cop_pairs_df = coint_pairs_df
        coint_cop_pairs_df['Copula_Score'] = np.nan
        coint_cop_pairs_df['Composite_Score'] = (
            coint_cop_pairs_df['Correlation'] * 0.6 + 
            (1 - coint_cop_pairs_df['P-Value']) * 0.4
        )
        pair_selection_scores = coint_cop_pairs_df.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)

    return pair_selection_scores

if __name__ == "__main__":
    pairs = run_pair_selection_pipeline(percentile=65, significance_level=0.25, use_copula=True)
    print("==== Top Selected Pairs ===")
    print(pairs)
    output_path = os.path.join(PROCESSED_DIR, "selected_pairs.csv")
    pairs.to_csv(output_path, index=False)
    print(f"💾 Saved selected pairs to: {output_path}")