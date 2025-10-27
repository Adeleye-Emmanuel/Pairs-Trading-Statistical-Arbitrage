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

    def __init__(self, prices_df, returns_df, train_end_date=None, test_start_date=None):
        self.full_prices_df = prices_df
        self.full_returns_df = returns_df

        # Default split: 60% train, 40% test
        if train_end_date is None:
            split_idx = int(len(prices_df)*0.75)
            self.train_end_date = prices_df.index[split_idx]
        else:
            self.train_end_date = pd.to_datetime(train_end_date)

        if test_start_date is None:
            self.test_start_date = self.train_end_date
        else:
            self.test_start_date = pd.to_datetime(self.test_start_date)
        
        # Splitting data
        self.train_prices_df = prices_df[:self.train_end_date]
        self.train_returns_df = prices_df[:self.train_end_date]
        self.test_prices_df = prices_df[self.test_start_date:]
        self.test_returns_df = prices_df[self.test_start_date:]

        print(f"\nTrain/Test Split:")
        print(f"  Training period: {self.train_prices_df.index[0]} to {self.train_prices_df.index[-1]}")
        print(f"  Training days: {len(self.train_prices_df)}")
        print(f"  Testing period: {self.test_prices_df.index[0]} to {self.test_prices_df.index[-1]}")
        print(f"  Testing days: {len(self.test_prices_df)}")

        self.copula_model = CopulaModel()
        self.selected_pairs = []

    def compute_correlations(self, returns_df):
        return returns_df.corr()

    # This is applied as a first stage filter on the asset universe, later on it could be replaced with a fundamentals filtration function
    # that considers more fundamental metrics as a baseline logic for initial selection before cointegration test
    def filter_top_percentile(self, corr_matrix, percentile, min_pairs=20, abs_min_corr=0.7):
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
        
        pairs_df = pd.DataFrame(high_corr_pairs, columns=['ETF1', 'ETF2', 'Correlation'])
        pairs_df = pairs_df.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
        print(f"  Selected {len(pairs_df)} pairs with correlation >= {threshold:.4f}")
        return pairs_df

    def perform_cointegration_test(self, pairs_df, significance_level=0.01):
        
        print(f"\nCointegration testing (p-value < {significance_level}):")

        coint_results = []
        for _, row in pairs_df.iterrows():
            etf1, etf2 = row["ETF1"], row["ETF2"]
            series1, series2 = self.train_prices_df[etf1].dropna(), self.train_prices_df[etf2].dropna()
            common_index = series1.index.intersection(series2.index)
            series1, series2 = series1.loc[common_index], series2.loc[common_index]
            score, pvalue, _ = coint(series1, series2)

            if pvalue < significance_level:
                X = add_constant(series2)
                hr_model = OLS(series1, X).fit()
                hedge_ratio = hr_model.params.iloc[1]

                spread = series1 - hedge_ratio * series2    

                try:
                    spread_lag = spread.shift(1).dropna()
                    spread_ret = spread.diff().dropna()
                    common_idx = spread_lag.index.intersection(spread_ret.index)
                    if len(common_idx) < 20:
                        return None
                    
                    spread_lag_aligned = spread_lag.loc[common_idx]
                    
                    # AR(1) regression
                    X = add_constant(spread_lag_aligned, prepend=True)
                    spread_ret_aligned = spread_ret.loc[common_idx]
                    spread_model = OLS(spread_ret_aligned, X).fit()
                    beta = spread_model.params.iloc[1]
                    if beta < 0 and beta > -1:
                        half_life = -np.log(2) / beta

                except Exception as e:
                    print(f"half-life calculation failed for {etf1}-{etf2}: {e}")
                    half_life = float("inf")

                coint_results.append((etf1, etf2, row["Correlation"], pvalue, half_life, hedge_ratio))
        coint_df = pd.DataFrame(coint_results, columns=['ETF1', 'ETF2', 'Correlation', 'P-Value', 'Half_life', 'Hedge_ratio'])
        return coint_df.sort_values(by='P-Value').reset_index(drop=True)

    def score_pair(self, coint_result, returns_df):
        
        returns_1 = returns_df[coint_result["ETF1"]].dropna().values
        returns_2 = returns_df[coint_result["ETF2"]].dropna().values
        
        pair_name = f"{coint_result["ETF1"]}_{coint_result["ETF2"]}"
        copula_model = self.copula_model.fit_copula(returns_1, returns_2, pair_name)

        if not copula_model:
            return None
        
        if copula_model["lower_tail_dependence"]>0.5:
            return None
        
        coint_score = np.clip(1- coint_result["P-Value"]/0.01, 0, 1)
        optimal_hl = 60
        hl_score = np.exp(-abs(coint_result["Half_life"] - optimal_hl)/optimal_hl)
        n_obs = len(returns_1)
        ll_per_obs = copula_model["log_likelihood"]/n_obs
        ll_score = np.clip(ll_per_obs, 0, 1) # rough normalisation

        # score_vector = {
        #     'mean_reversion_speed': 1 / (1 + coint_result['Half_life']), 
        #     'coint_strength': -np.log(coint_result['P-Value']),  
        #     'copula_fit': copula_model['log_likelihood'],  
        #     'tail_safety': 1 - copula_model['lower_tail_dependence']  
        # }
        composite_score = (0.5 * coint_score + 0.3 * hl_score + 0.2 * ll_score)

        pair_score = {
            **coint_result,
            "composite_score": composite_score,
            "coint_score": coint_score,
            "half_life_score": hl_score,
            "ll_score": ll_score,
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

    def run_selection(self, max_pairs=15):
        
        print("\n" + "="*60)
        print("PAIR SELECTION (TRAINING PHASE)")
        print("="*60)

        corr_matrix = self.compute_correlations(self.train_returns_df)
        candidate_pairs = self.filter_top_percentile(corr_matrix=corr_matrix, percentile=95)
        coint_result = self.perform_cointegration_test(candidate_pairs, significance_level=0.01)

        print(f"Found {len(coint_result)} cointegrated pairs")
        
        print("\n Copula Fitting and Scoring..")
        qualified_pairs = []
        for _, coint_row in coint_result.iterrows():
            scored_pair = self.score_pair(coint_row.to_dict(), self.train_returns_df)
            if scored_pair:
                qualified_pairs.append(scored_pair)
  
        if not qualified_pairs:
            print("No qualified pairs found!") 
            return []
        
        print(f" {len(qualified_pairs)} pairs successfully fitted")     

        # pareto_pairs = self.select_pareto_frontier(qualified_pairs)
        # print(f"Found {len(pareto_pairs)} Pareto Optimal Pairs")
        
        # # will possibly create a max pair final object

        # final_df = pd.DataFrame(pareto_pairs)
        qualified_pairs.sort(key=lambda x: x["composite_score"], reverse=True)
        selected_pairs = qualified_pairs[:max_pairs]

        print("\n" + "="*60)
        print(f"SELECTED {len(selected_pairs)} PAIRS (Ranked by Score)")
        print("="*60)

        for i, pair in enumerate(selected_pairs, 1):
            print(f"\n{i}. {pair['ETF1']}_{pair['ETF2']}")
            print(f"   Composite Score: {pair['composite_score']:.3f}")
            print(f"   Cointegration: p={pair['P-Value']:.4f}, half-life={pair['Half_life']:.1f} days")
            print(f"   Copula: {pair['copula_type']}, tail_dep={pair['tail_dependence']:.3f}")
            print(f"   Correlation: {pair['Correlation']:.3f}, Hedge Ratio: {pair['Hedge_ratio']:.3f}")
      
        self.selected_pairs = selected_pairs

        return selected_pairs
    
    def get_test_data(self):
        # Return test data for signal generation
        return self.test_prices_df, self.test_returns_df

if __name__ == "__main__":
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    selector = PairSelector(prices, returns)
    selected_pairs = selector.run_selection()
    test_prices, test_returns = selector.get_test_data()
    print(f"\nReady for signal generation on {len(test_prices)} test days")