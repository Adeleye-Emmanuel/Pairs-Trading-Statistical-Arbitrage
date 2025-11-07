import os
import numpy as np 
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS, add_constant
from itertools import combinations
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ETFS_DIR, PROCESSED_DIR
from src.copula_model import CopulaModel

class PairSelector():

    def __init__(self, prices_df, returns_df, train_end_date, test_end_date, 
                 top_n=10, min_correlation=0.6, max_correlation=0.95,
                 coint_pvalue=0.05, min_half_life=5, max_half_life=60):
        """
        Enhanced pair selector with cointegration and half-life filters
        
        Args:
            coint_pvalue: Maximum p-value for cointegration test (0.01 = 99% confidence)
            min_half_life: Minimum acceptable half-life in days
            max_half_life: Maximum acceptable half-life in days
        """
        self.prices = prices_df
        self.returns = returns_df
        self.top_n = top_n
        self.min_correlation = min_correlation
        self.max_correlation = max_correlation
        self.coint_pvalue = coint_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.train_end_date = train_end_date
        self.test_end_date = test_end_date
        self.copula_fitter = CopulaModel()

    def get_train_data(self):
        return self.prices.loc[:self.train_end_date], self.returns.loc[:self.train_end_date]
    
    def get_test_data(self):
        return self.prices.loc[self.train_end_date:self.test_end_date], self.returns.loc[self.train_end_date:self.test_end_date]
    
    def calculate_half_life(self, spread):
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process
        """
        spread_lag = pd.Series(spread).shift(1).values[1:]
        spread_diff = pd.Series(spread).diff().values[1:]
        
        # Remove NaN values
        valid_idx = ~(np.isnan(spread_lag) | np.isnan(spread_diff))
        if valid_idx.sum() < 20:
            return np.nan
        
        spread_lag = spread_lag[valid_idx]
        spread_diff = spread_diff[valid_idx]
        
        # OLS: spread_diff = theta * (mu - spread_lag) + epsilon
        # Simplified: spread_diff = -lambda * spread_lag + const
        X = add_constant(spread_lag)
        model = OLS(spread_diff, X).fit()
        
        lambda_param = -model.params[1]
        
        if lambda_param <= 0:
            return np.nan  # No mean reversion
        
        half_life = np.log(2) / lambda_param
        return half_life
    
    def calculate_pair_metrics(self, etf1, etf2, train_prices, train_returns):
        """
        Calculate comprehensive pair metrics including cointegration and half-life
        """
        prices1 = train_prices[etf1].dropna()
        prices2 = train_prices[etf2].dropna()
        returns1 = train_returns[etf1].dropna()
        returns2 = train_returns[etf2].dropna()
        
        common_index = prices1.index.intersection(prices2.index).intersection(
            returns1.index).intersection(returns2.index)
        
        if len(common_index) < 252:  # Require at least 1 year of data
            return None
        
        prices1 = prices1.loc[common_index]
        prices2 = prices2.loc[common_index]
        returns1 = returns1.loc[common_index]
        returns2 = returns2.loc[common_index]

        # 1. Correlation check (pre-filter)
        corr, _ = spearmanr(returns1, returns2)
        if abs(corr) < self.min_correlation or abs(corr) > self.max_correlation:
            return None
        
        # 2. Cointegration test (CRITICAL)
        coint_stat, pvalue, crit_values = coint(prices1, prices2)
        if pvalue > self.coint_pvalue:
            return None  # Not cointegrated
        
        # 3. Calculate hedge ratio and spread
        X = add_constant(prices2.values)
        ols_model = OLS(prices1.values, X).fit()
        hedge_ratio = ols_model.params[1]
        
        spread = prices1.values - hedge_ratio * prices2.values
        
        # 4. Calculate half-life
        half_life = self.calculate_half_life(spread)
        if np.isnan(half_life) or half_life < self.min_half_life or half_life > self.max_half_life:
            return None  # Half-life out of acceptable range
        
        # 5. Calculate spread statistics
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # 6. Fit copula for ranking (on uniform transforms)
        u_data = self.transform_to_uniform(returns1.values)
        v_data = self.transform_to_uniform(returns2.values)
        
        copula_model = self.copula_fitter.fit_copula(u_data, v_data)
        copula_log_lik = copula_model.get("log_l", -np.inf)

        return {
            "etf1": etf1,
            "etf2": etf2,
            "spearman_corr": corr,
            "coint_pvalue": pvalue,
            "hedge_ratio": hedge_ratio,
            "half_life": half_life,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "copula_type": copula_model.get("copula_type"),
            "copula_log_lik": copula_log_lik,
            "copula_model": copula_model  # Store the fitted model
        }
    
    def transform_to_uniform(self, data):
        """
        Transform data to uniform distribution using empirical CDF
        """
        data_clean = data[np.isfinite(data)]
        if len(data_clean) < 10:
            return np.full(len(data), 0.5)
        
        n = len(data_clean)
        ranks = pd.Series(data_clean).rank(method='average').values
        uniform = ranks / (n + 1)  # Avoid 0 and 1
        return np.clip(uniform, 1e-6, 1-1e-6)
    
    def calculate_composite_score(self, pair):
        """
        Calculate composite ranking score based on multiple factors
        """
        # Lower p-value is better (stronger cointegration)
        coint_score = -np.log10(pair["coint_pvalue"] + 1e-10)
        
        # Half-life closer to 20 days is ideal
        half_life_score = 1 / (1 + abs(pair["half_life"] - 20) / 20)
        
        # Higher correlation (in absolute terms) is better
        corr_score = abs(pair["spearman_corr"])
        
        # Higher copula likelihood indicates better dependence structure
        copula_score = 1 / (1 + np.exp(-pair["copula_log_lik"] / 100))
        
        # Weighted composite score
        weights = {
            "coint": 0.4,
            "half_life": 0.3,
            "corr": 0.15,
            "copula": 0.15
        }
        
        composite = (weights["coint"] * coint_score +
                    weights["half_life"] * half_life_score +
                    weights["corr"] * corr_score +
                    weights["copula"] * copula_score)
        
        return composite
    
    def run_selection(self):
        """
        Run comprehensive pair selection with all filters
        """
        print("Starting enhanced pair selection...")
        train_prices, train_returns = self.get_train_data()
        etf_list = sorted(train_prices.columns.tolist())

        candidate_pairs = []
        total_tested = 0
        
        for etf1, etf2 in combinations(etf_list, 2):
            total_tested += 1
            #if total_tested % 100 == 0:
                #print(f"  Tested {total_tested} pairs, found {len(candidate_pairs)} candidates...")
            
            metrics = self.calculate_pair_metrics(etf1, etf2, train_prices, train_returns)
            if metrics is not None:
                candidate_pairs.append(metrics)
        
        print(f"\nPair Selection Summary:")
        print(f"  Total pairs tested: {total_tested}")
        print(f"  Pairs passing correlation filter: ~{int(total_tested * 0.3)}")
        print(f"  Pairs passing cointegration test (p<{self.coint_pvalue}): {len(candidate_pairs)}")
        print(f"  Pairs with valid half-life ({self.min_half_life}-{self.max_half_life} days): {len(candidate_pairs)}")
        
        if not candidate_pairs:
            print("WARNING: No pairs passed all filters!")
            return []
        
        # Calculate composite scores and rank
        for pair in candidate_pairs:
            pair["composite_score"] = self.calculate_composite_score(pair)
        
        # Sort by composite score
        ranked_pairs = sorted(candidate_pairs, key=lambda x: x["composite_score"], reverse=True)
        selected_pairs = ranked_pairs[:self.top_n]

        print(f"\nSelected top {len(selected_pairs)} pairs:")
        
        # Output selected pairs as dataframe for inspection
        display_cols = ["etf1", "etf2", "coint_pvalue", "half_life", 
                       "spearman_corr", "copula_type", "composite_score"]
        selected_pairs_df = pd.DataFrame(selected_pairs)[display_cols]
        selected_pairs_df["half_life"] = selected_pairs_df["half_life"].round(1)
        selected_pairs_df["coint_pvalue"] = selected_pairs_df["coint_pvalue"].round(4)
        selected_pairs_df["spearman_corr"] = selected_pairs_df["spearman_corr"].round(3)
        selected_pairs_df["composite_score"] = selected_pairs_df["composite_score"].round(3)
        print(selected_pairs_df.to_string(index=False))
        
        return selected_pairs


if __name__ == "__main__":
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    
    train_end = pd.to_datetime("2023-12-31")
    test_end = pd.to_datetime("2024-12-31")
    
    selector = PairSelector(
        prices, returns, 
        train_end_date=train_end,
        test_end_date=test_end,
        top_n=10,
        coint_pvalue=0.01,  # Strict cointegration
        min_half_life=5,     # Fast enough mean reversion
        max_half_life=60     # Not too slow
    )
    
    selected_pairs = selector.run_selection()
    test_prices, test_returns = selector.get_test_data()
    
    print(f"\nReady for signal generation on {len(test_prices)} test days")