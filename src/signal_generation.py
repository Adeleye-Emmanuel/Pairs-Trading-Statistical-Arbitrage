import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import t as student_t, norm
from copulae.elliptical import StudentCopula
from src.copula_model import *
from src.pair_selection import *
from src.config import ETFS_DIR, PROCESSED_DIR

class SignalGenerator:

    def __init__(self, lookback_days=90, entry_threshold=0.95, exit_threshold=0.5, 
                 max_holding_days=32):
        
        self.lookback_days = lookback_days
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_holding_days = max_holding_days

    def fit_marginal_distributions(self, returns):
        """Fit marginal distribution on returns data"""
        try:
            returns_clean = returns[np.isfinite(returns)]
            if len(returns_clean) < 10:
                raise ValueError("Insufficient data")
            
            df, loc, scale = student_t.fit(returns_clean)
            
            # Validate parameters
            if not (np.isfinite(df) and np.isfinite(loc) and np.isfinite(scale) and df > 2):
                raise ValueError("Invalid t-distribution parameters")
            
            def cdf(x):
                return student_t.cdf(x, df, loc=loc, scale=scale)
            
            return {"cdf": cdf, "params": {"df": df, "loc": loc, "scale": scale}}
        
        except:
            # Fallback to normal
            mu, std = np.mean(returns_clean), np.std(returns_clean)
            if std <= 0:
                std = 1e-6
            
            def cdf(x):
                return norm.cdf(x, loc=mu, scale=std)
            
            return {"cdf": cdf, "params": {"mu": mu, "std": std}}
    
    def calculate_conditional_probability_studentt(self, u, v, rho, df):
        """
        Calculate P(U1 <= u | U2 = v) for t-copula
        Returns NaN if calculation fails
        """
        try:

            if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(rho) and np.isfinite(df)):
                return np.nan
            
            if df <= 2 or abs(rho) >= 1:
                return np.nan
            
            t1 = student_t.ppf(u, df)
            t2 = student_t.ppf(v, df)                      
            
            # Check for infinite quantiles
            if not (np.isfinite(t1) and np.isfinite(t2)):
                return np.nan
            
            # Conditional distribution T1|T2 ~ t(df+1, mu_cond, sigma_cond)
            scaling = np.sqrt((df + 1) / (df + t2**2))
            mu_cond = rho * t2 * scaling
            var_cond = ((df + t2**2) / (df + 1)) * (1 - rho**2)
            sigma_cond = np.sqrt(var_cond)
            
            if not np.isfinite(sigma_cond) or sigma_cond <= 0:
                return np.nan
            
            # Calculate conditional CDF
            cond_prob = student_t.cdf(t1, df=df+1, loc=mu_cond, scale=sigma_cond)
            
            if not np.isfinite(cond_prob):
                 return np.nan
             
            return cond_prob
        
        except Exception as e:
            return np.nan
    
    def calculate_conditional_probability_gaussian(self, u, v, rho):
        try:
            if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(rho)):
                return np.nan
            if abs(rho)>=1:
                return np.nan
            
            z1, z2 = norm.ppf(u), norm.ppf(v)
            if not (np.isfinite(z1) and np.isfinite(z1)):
                return np.nan
            # Conditional distribution: Z1|Z2 ~ N(rho*z2, 1-rho^2)
            mu_cond = rho * z2
            sigma_cond = np.sqrt(1-rho**2)

            if not np.isfinite(sigma_cond) or sigma_cond<=0:
                return np.nan
            cond_prob = norm.cdf(z1, loc=mu_cond, scale=sigma_cond)

            if not np.isfinite(cond_prob):
                return np.nan
            
            return cond_prob
        except Exception as e:
            return np.nan
        
    def calulcate_conditional_probability_archimedean(self, u, v, copula):
        """
        Calculate P(U1 <= u | U2 = v) for Archimedean copulas (Frank, Clayton, Gumbel)
        Using numerical derivative: P(U1 <= u | U2 = v) = ∂C(u,v)/∂v
        """
        try:
            if not (np.isfinite(u) and np.isfinite(v)):
                return np.nan
            
            # Small pertubation for numerical derivatives
            epsilon = 1e-6
            v_plus = min(v+epsilon, 1-1e-10)
            v_minus = max(v-epsilon, 1e-10)

            uv = np.array([[u, v_minus], [u, v_plus]])

            try:
                c_minus = copula.cdf(uv[0:1, :])[0]
                c_plus = copula.cdf(uv[1:2, :])[0]
            except:
                c_minus = copula.cdf([u, v_minus])
                c_plus = copula.cdf([u, v_plus])

            cond_prob = (c_plus - c_minus) / (2*epsilon)

            cond_prob = np.clip(cond_prob, 0, 1) # Clipping to valid probability range

            if not np.isfinite(cond_prob):
                return np.nan
            
            return cond_prob
        
        except Exception as e:
            return np.nan
    
    def calculate_conditional_probabilities(self, u, v, copula_model):

        copula = copula_model["copula"]
        copula_type = copula_model["copula_type"]

        if copula_type == "Student_t":
            params_obj = copula.params
            df = float(params_obj.df)
            rho = float(params_obj.rho[0]) if isinstance(params_obj.rho, np.ndarray) else float(params_obj.rho)
            return self.calculate_conditional_probability_studentt(u, v, rho, df)
        
        elif copula_type == 'Gaussian':
            params_obj = copula.params
            if hasattr(params_obj, 'rho'):
                rho = float(params_obj.rho[0]) if isinstance(params_obj.rho, np.ndarray) else float(params_obj.rho)
            else:
                rho = float(copula.params[0]) if hasattr(copula.params, "shape") else 0.5
            return self.calculate_conditional_probability_gaussian(u, v, rho)
        
        elif copula_type in ["Frank", "Clayton", "Gumbel"]:
            return self.calulcate_conditional_probability_archimedean(u,v, copula)
        
        else:
            return np.nan # Unknown copula type    

    def generate_signal(self, prices_df, pair_data, 
                        start_date=None, end_date=None):
        
        pair_name = f"{pair_data['ETF1']}_{pair_data['ETF2']}"
        copula_type = pair_data.get("copula_type", "Unkown")
        print(f"Generating signals for {pair_name} (using {copula_type})...")
        
        hedge_ratio = pair_data.get("hedge_ratio", 1)
        
        # Get static copula model from pair selection
        model = pair_data.get("copula_model")
        if not model:
            print(f"  ERROR: No copula model found for {pair_name}")
            return None

        copula = model["copula"]
        
        if start_date:
            prices_df = prices_df[start_date:]
        if end_date:
            prices_df = prices_df[:end_date]
        
        dates = prices_df.index        
        results = pd.DataFrame(index=dates,
                               columns=[
                                   "position", 
                                   "entry_signal", "exit_signal", 
                                   "cond_prob_1g2",
                                   "spread", "entry_price", "entry_date"
                               ])
        
        position = 0
        entry_idx = None
        entry_date = None
        
        etf1, etf2 = pair_data["ETF1"], pair_data["ETF2"]
        
        successful_calcs = 0
        
        for i in range(self.lookback_days, len(prices_df)):
            current_date = dates[i]
            
            # Get lookback window
            lookback_prices = prices_df.iloc[i-self.lookback_days:i]
            current_prices = prices_df.iloc[i]
            
            # Align series
            series_1 = lookback_prices[etf1].dropna()
            series_2 = lookback_prices[etf2].dropna()
            common_index = series_1.index.intersection(series_2.index)
            
            if len(common_index) < 30:
                continue
            
            lookback_series1 = series_1.loc[common_index]
            lookback_series2 = series_2.loc[common_index]
            
            # Calculate LOG RETURNS on rolling window
            lookback_returns_1 = np.diff(np.log(lookback_series1.values))
            lookback_returns_2 = np.diff(np.log(lookback_series2.values))
            
            lookback_returns_1 = lookback_returns_1[np.isfinite(lookback_returns_1)]
            lookback_returns_2 = lookback_returns_2[np.isfinite(lookback_returns_2)]
            
            if len(lookback_returns_1) < 10 or len(lookback_returns_2) < 10:
                continue
            
            # Calculate current log return
            if not np.isfinite(current_prices[etf1]) or not np.isfinite(current_prices[etf2]):
                continue
                
            current_log_ret_1 = np.log(current_prices[etf1] / lookback_series1.iloc[-1])
            current_log_ret_2 = np.log(current_prices[etf2] / lookback_series2.iloc[-1])
            
            if not (np.isfinite(current_log_ret_1) and np.isfinite(current_log_ret_2)):
                continue
            
            # Fit marginals on rolling window (adapts to changing volatility)
            marginal_1 = self.fit_marginal_distributions(lookback_returns_1)
            marginal_2 = self.fit_marginal_distributions(lookback_returns_2)
            
            # Transform current returns to uniform using fitted marginals
            u_current = marginal_1["cdf"](current_log_ret_1)
            v_current = marginal_2["cdf"](current_log_ret_2)
            
            u_current = np.clip(u_current, 1e-6, 1-1e-6)
            v_current = np.clip(v_current, 1e-6, 1-1e-6)
            
            # Calculate conditional probability using STATIC copula parameters
            cond_prob_1g2 = self.calculate_conditional_probabilities(u_current, v_current, model)
            
            if not np.isfinite(cond_prob_1g2):
                continue
            
            successful_calcs += 1
            
            results.loc[current_date, "cond_prob_1g2"] = cond_prob_1g2
            results.loc[current_date, "spread"] = current_prices[etf1] - hedge_ratio * current_prices[etf2]
            
            # Entry logic (CORRECTED)
            # High cond_prob (>0.95) = ETF1 likely in lower tail = undervalued = LONG spread
            # Low cond_prob (<0.05) = ETF1 likely in upper tail = overvalued = SHORT spread
            if position == 0:
                # ETF1 undervalued relative to ETF2 -> Long spread (buy ETF1, sell ETF2)
                if cond_prob_1g2 > self.entry_threshold:
                    position = 1
                    results.loc[current_date, "entry_signal"] = True
                    results.loc[current_date, "entry_price"] = results.loc[current_date, "spread"]
                    results.loc[current_date, "entry_date"] = current_date
                    entry_idx = i
                    entry_date = current_date
                
                # ETF1 overvalued relative to ETF2 -> Short spread (sell ETF1, buy ETF2)
                elif cond_prob_1g2 < (1 - self.entry_threshold):
                    position = -1
                    results.loc[current_date, "entry_signal"] = True
                    results.loc[current_date, "entry_price"] = results.loc[current_date, "spread"]
                    results.loc[current_date, "entry_date"] = current_date
                    entry_idx = i
                    entry_date = current_date
            
            # Exit logic
            elif position != 0:
                exit_signal = False
                
                # Mean reversion: exit when conditional prob returns toward 0.5
                if abs(cond_prob_1g2 - 0.5) < abs(self.exit_threshold - 0.5):
                    exit_signal = True
                
                # Maximum holding period
                elif entry_idx and (i - entry_idx) >= self.max_holding_days:
                    exit_signal = True
                
                if exit_signal:
                    results.loc[current_date, "exit_signal"] = True
                    results.loc[current_date, "entry_date"] = entry_date
                    position = 0
                    entry_idx = None
                    entry_date = None
            
            results.loc[current_date, "position"] = position
        
        # Clean up results
        results = results.dropna(subset=["cond_prob_1g2"])
        results["entry_signal"] = results["entry_signal"].fillna(False)
        results["exit_signal"] = results["exit_signal"].fillna(False)
        
        num_entries = results["entry_signal"].sum()
        print(f"  Successful calculations: {successful_calcs}")
        print(f"  Generated {num_entries} entry signals")
        
        if num_entries == 0:
            if successful_calcs > 0:
                print(f"  Cond prob range: [{results['cond_prob_1g2'].min():.3f}, {results['cond_prob_1g2'].max():.3f}]")
                print(f"  Entry threshold: {self.entry_threshold} (might be too extreme)")
            else:
                print(f"  ERROR: No successful conditional probability calculations")
        
        return results
    
    def generate_batch_signals(self, prices_df, selected_pairs, start_date=None, end_date=None):
        all_signals = {}
        
        for pair_data in selected_pairs:
            signals = self.generate_signal(prices_df, pair_data, start_date=start_date, end_date=end_date)
            pair_name = f"{pair_data['ETF1']}_{pair_data['ETF2']}"
            all_signals[pair_name] = signals
        
        return all_signals
    
if __name__ == "__main__":
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    selector = PairSelector(prices, returns)
    selected_pairs = selector.run_selection()
    
    test_prices, test_returns = selector.get_test_data()
    signal_gen = SignalGenerator()
    all_signals = signal_gen.generate_batch_signals(test_prices, selected_pairs)
    
    for pair_name, signals in all_signals.items():
        if signals is not None:
            print(f"\n{pair_name}:")
            print(f"  Total signals: {signals['entry_signal'].sum()}")
            if len(signals) > 0:
                print(f"  Avg conditional prob: {signals['cond_prob_1g2'].mean():.3f}")
                print(f"  Cond prob range: [{signals['cond_prob_1g2'].min():.3f}, {signals['cond_prob_1g2'].max():.3f}]")