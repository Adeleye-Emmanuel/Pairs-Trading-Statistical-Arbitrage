import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scipy.stats import t as student_t, norm
from copulae.elliptical import StudentCopula

class SignalGenerator:

    def __init__(self, copula_model, lookback_days=60, entry_threshold=0.95, exit_threshold=0.5, 
                 max_holding_days=63):
        self.copula_model = copula_model
        self.lookback_days = lookback_days
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_holding_days = max_holding_days

    def calculate_normalised_returns(self, current_price, historical_prices):
        
        if len(historical_prices) < 5:
            return 0.0
        
        #recent_prices = historical_prices[-self.lookback_days:]
        returns = np.diff(np.log(historical_prices))

        if len(returns) == 0 or np.std(returns) == 0 :
            return 0.0
        
        current_return = (current_price - historical_prices[-1])/historical_prices[-1]
        normalized_return = (current_return - np.mean(returns))/np.std(returns)

        return normalized_return
    
    def calculate_conditional_probabilities(self, u, v, copula_model):
        try:
            copula = copula_model["copula"]
            rho, df = copula.params[0], copula.params[1]
            t1 = student_t.ppf(u, df)
            t2 = student_t.ppf(v, df)
            mu_cond = rho * t2
            sigma_cond = np.sqrt((df + t2**2) * (1 - rho**2)/(df + 1))

            return student_t.cdf(t1, df+1, loc=mu_cond, scale=sigma_cond)
        
        except:
            return 0.5
    
    def generate_signal(self, prices_df,  pair_data, 
                        start_date=None, end_date=None):
        
        pair_name = f"{pair_data['ETF1']}_{pair_data['ETF2']}"
        print(f"Generating historical signals for {pair_name}...")
        hedge_ratio = pair_data.get("hedge_ratio", 1)
        model = pair_data.get("copula_model")
        if not model:
            print(f"No copula model found for {pair_name}")
            return None
        
        if start_date:
            prices_df = prices_df[start_date:]
        if end_date:
            prices_df = prices_df[:end_date]
        
        dates = prices_df.index        
        results = pd.DataFrame(index=dates,
                               columns=[
                                   "position", 
                                    "entry_signal", "exit_signal", 
                                    "cond_prob_1g2", "cond_prob_2g1"
                                    "spread", "entry_price", "entry_date"
                                        ])
        
        position = 0
        entry_idx = None
        entry_date = None
        
        for i in range(self.lookback_days, len(prices_df)):
            current_date = dates[i]

            lookback_prices = prices_df.iloc[i-self.lookback_days:i]
            current_prices = prices_df.iloc[i]

            # Calculating normalised returns
            norm_return_1 = self.calculate_normalised_returns(current_prices["ETF1"], lookback_prices["ETF1"].values)
            norm_return_2 = self.calculate_normalised_returns(current_prices["ETF1"], lookback_prices["ETF1"].values)

            # Transforming to uniform space using fitted marginals
            u = model["marginal_1"]["cdf"](norm_return_1)
            v = model["marginal_2"]["cdf"](norm_return_2)

            u,v = np.clip(u, 1e-6, 1-1e-6), np.clip(v, 1e-6, 1-1e-6) 

            # Calculating conditional probabilities in both directions
            cond_prob_1g2 = self.calculate_conditional_probabilities(u,v, model)
            cond_prob_2g1 = self.calculate_conditional_probabilities(v,u, model)

            results.loc[current_date, "cond_prob_1g2"] = cond_prob_1g2
            results.loc[current_date, "cond_prob_2g1"] = cond_prob_2g1
            results.loc[current_date, "spread"] = current_prices["ETF1"] - hedge_ratio * current_prices["ETF2"]

            # Entry logic
            if position == 0:
                if cond_prob_1g2 > self.entry_threshold and cond_prob_2g1 < (1-self.entry_threshold):
                    position = 1
                    results.loc[current_date, "entry_signal"] = True
                    results.loc[current_date, "entry_price"] = results.loc[current_date, "spread"]
                    results.loc[current_date, "entry_date"] = current_date
                    entry_idx = i
                    entry_date = current_date

                elif cond_prob_1g2 < (1-self.entry_threshold) and cond_prob_1g2 > self.entry_threshold:
                    position = -1
                    results.loc[current_date, "entry_signal"] = True
                    results.loc[current_date, "entry_price"] = results.loc[current_date, "spread"]
                    results.loc[current_date, "entry_date"] = current_date
                    entry_idx = i
                    entry_date = current_date 

            elif position !=0:
                exit_signal = False

                # Mean reversion exit
                if position == 1 and (cond_prob_1g2<self.exit_threshold or cond_prob_2g1>self.exit_threshold): # ETF1 is not cheap and ETF2 could be considered cheap
                    exit_signal = True
                elif position == -1 and (cond_prob_2g1<self.exit_threshold or cond_prob_1g2>self.exit_threshold): # ETF2 is not cheap and ETF1 could be considered cheap
                    exit_signal = True
                    exit_signal = True
                elif entry_idx and (i-entry_idx) >= self.max_holding_days:
                    exit_signal = True       

                if exit_signal:
                    results.loc[current_date, "exit_signal"] = True
                    results.loc[current_date, "entry_date"] = entry_date
                    position = 0 
                    entry_idx = None
                    entry_date = None
            
            results.loc[current_date, "position"] = position
        
        # Cleaning up results
        results = results.dropna(subset="cond_prob_1g2")
        results["entry_signal"] = results["entry_signal"].fillna(False)
        results["exit_signal"] = results["exit_signal"].fillna(False)

        print(f"Generated {results["entry_signal"].sum()} entry signals")
        return results
    
    def generate_batch_signals(self, prices_df, selected_pairs, start_date=None, end_date=None):

        all_signals = {}
        
        for pair_data in selected_pairs:
            signals = self.generate_signal(prices_df, pair_data, start_date=start_date, end_date=end_date)
            pair_name = f"{pair_data["ETF1"]}_{pair_data["ETF2"]}"
            all_signals[pair_name] = signals

        return all_signals