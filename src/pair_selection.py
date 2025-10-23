import os
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS, add_constant
from itertools import combinations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import MinMaxScaler

from config import ETFS_DIR, PROCESSED_DIR
from src.copula_model import CopulaModel

class PairSelector():

    def __init__(self, prices_df, returns_df):
        self.prices_df = prices_df
        self.returns_df = returns_df
        self.copula_model = CopulaModel()
        self.selected_pairs = []

    def compute_correlations(self):
        return self.returns_df.corr()

    # This is applied as a first stage filter on the asset universe, later on it could be replaced with a fundamentals filtration function
    # that considers more fundamental metrics as a baseline logic for initial selection before cointegration test
    def filter_top_percentile(self, corr_matrix, percentile, min_pairs=20, abs_min_corr=0.8):
        # Get pairs of ETFs with correlation above the specified percentile for all tickers
        vals = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) == 0:
            raise ValueError("No finite correlation values found.")
        vals = finite_vals
        pct_thr = np.nanpercentile(vals, percentile)
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

    def perform_cointegration_test(self, pairs_df, significance_level):
        coint_results = []
        for _, row in pairs_df.iterrows():
            etf1, etf2 = row["ETF1"], row["ETF2"]
            series1, series2 = self.prices_df[etf1].dropna(), self.prices_df[etf2].dropna()
            common_index = series1.index.intersection(series2.index)
            series1, series2 = series1.loc[common_index], series2.loc[common_index]
            score, pvalue, _ = coint(series1, series2)
            if pvalue < significance_level:
                X = add_constant(series2)
                hr_model = OLS(series1, X).fit()
                hedge_ratio = hr_model.params.iloc[1]

                spread = series1 - hedge_ratio * series2    
                half_life = float("inf")

                try:
                    spread_lag = spread.shift(1).dropna()
                    spread_ret = spread.diff().dropna()
                    common_idx = spread_lag.index.intersection(spread_ret.index)
                    if len(common_idx) > 0:
                        spread_lag = spread_lag.loc[common_idx]
                        spread_lag = add_constant(spread_lag, prepend=True)
                        spread_ret = spread_ret.loc[common_idx]
                        spread_model = OLS(spread_ret, spread_lag).fit()
                        if spread_model.params.iloc[0] < 0:
                            half_life = -np.log(2) / spread_model.params.iloc[0] 

                except Exception as e:
                    print(f"half-life calculation failed for {etf1}-{etf2}: {e}")
                    half_life = float("inf")

                coint_results.append((etf1, etf2, row["Correlation"], pvalue, half_life))
        coint_df = pd.DataFrame(coint_results, columns=['ETF1', 'ETF2', 'Correlation', 'P-Value', 'Half_life'])
        return coint_df.sort_values(by='P-Value').reset_index(drop=True)

    def score_pair(self, coint_result, max_half_life=10000):
        if coint_result["Half_life"] > max_half_life:
            return None
        
        returns_1 = self.returns_df[coint_result["ETF1"]].dropna().values
        returns_2 = self.returns_df[coint_result["ETF2"]].dropna().values
        
        pair_name = f"{coint_result["ETF1"]}_{coint_result["ETF2"]}"
        copula_model = self.copula_model.fit_copula(returns_1, returns_2, pair_name)

        if not copula_model:
            return None
        
        if copula_model["lower_tail_dependence"]>0.3:
            return None
        
        score_vector = {
            'mean_reversion_speed': 1 / (1 + coint_result['Half_life']), 
            'coint_strength': -np.log(coint_result['P-Value']),  
            'copula_fit': copula_model['log_likelihood'],  
            'tail_safety': 1 - copula_model['lower_tail_dependence']  
        }

        pair_score = {
            **coint_result,
            **score_vector,
            'copula_type': copula_model['copula_type'],
            'tail_dependence': copula_model['lower_tail_dependence'],
            'copula_model': copula_model
        }

        return pair_score
    
    def select_pareto_frontier(self, qualified_pairs):

        metrics = np.array([
            [p["mean_reversion_speed"], p["coint_strength"], p["copula_fit"], p["tail_safety"]]
            for p in qualified_pairs
        ])

        metrics_norm = (metrics - metrics.min(axis=0)) / (metrics.max(axis=0) - metrics.min(axis=0))

        # find pareto mask
        pareto_mask = np.ones(len(metrics_norm), dtype=bool)

        for i in range(len(metrics_norm)):
            for j in range(len(metrics_norm)):
                if i!=j and np.all(metrics_norm[j] >= metrics_norm[i]) and np.any(metrics_norm[j] > metrics_norm[i]):
                    pareto_mask[i] = False
                    break
        pareto_pairs = [qualified_pairs[i] for i in range(len(qualified_pairs)) if pareto_mask[i]]
        return pareto_pairs
    
    def get_copula_model(self):
        return self.copula_model

    def run_selection(self):
        corr_matrix = self.compute_correlations()
        candidate_pairs = self.filter_top_percentile(corr_matrix=corr_matrix, percentile=95)
        coint_result = self.perform_cointegration_test(candidate_pairs, 0.05)
        print(coint_result)
        print(f"Found {len(coint_result)} cointegrated pairs")
        qualified_pairs = []
        for _, coint_row in coint_result.iterrows():
            coint_dict = {
                'ETF1': coint_row['ETF1'],
                'ETF2': coint_row['ETF2'],
                'Correlation': coint_row['Correlation'],
                'P-Value': coint_row['P-Value'],
                'Half_life': coint_row['Half_life']
                }
            
            scored_pair = self.score_pair(coint_dict)
            if scored_pair:
                qualified_pairs.append(scored_pair)

        print(f"Found {len(qualified_pairs)} qualified pairs after Copula analysis")       
        if not qualified_pairs:
            print("No qualified pairs found!") 
            return pd.DataFrame()
        
        pareto_pairs = self.select_pareto_frontier(qualified_pairs)
        print(f"Found {len(pareto_pairs)} Pareto Optimal Pairs")
        
        # will possibly create a max pair final object

        final_df = pd.DataFrame(pareto_pairs)
        if "copula_model" in final_df.columns:
            final_df = final_df.drop('copula_model', axis=1)

        self.selected_pairs = pareto_pairs
        print(f"Selected {len(pareto_pairs)} final pairs")

        return self.selected_pairs
    

if __name__ == "__main__":
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    selector = PairSelector(prices, returns)
    selected_pairs = selector.run_selection()
    print(type(selected_pairs))
    print(f"Selected {len(selected_pairs)} pairs")
    print(selected_pairs)
    