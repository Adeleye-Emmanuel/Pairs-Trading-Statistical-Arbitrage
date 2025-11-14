import numpy as np
import pandas as pd
from scipy.stats import t as student_t, norm
from statsmodels.api import OLS, add_constant
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.copula_model import CopulaModel
from src.config import ETFS_DIR, PROCESSED_DIR

class OptimizedSignalGenerator:
    
    def __init__(self, 
                 lookback_days=30, # 60
                 # Primary signal: Z-score
                 entry_z_score=2.0, # 2.0
                 exit_z_score=0.35, # 0.35
                 stop_loss_z_score=4.5, # 4.5
                 use_copula_filter=False,
                 copula_veto_threshold=0.8,  # Only veto if copula strongly disagrees
                 # Position management
                 max_holding_days=30,
                 position_scale_by_conviction=True):

        self.lookback_days = lookback_days
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_loss_z_score = stop_loss_z_score
        self.max_holding_days = max_holding_days
        
        self.use_copula_filter = use_copula_filter
        self.copula_veto_threshold = copula_veto_threshold
        self.position_scale_by_conviction = position_scale_by_conviction
        
        print(f"\nOptimized Signal Generator:")
        print(f"  Primary Signal: Z-score > {entry_z_score}")
        print(f"  Copula Role: {'Veto if strongly contradicts' if use_copula_filter else 'Disabled'}")
        print(f"  Position Scaling: {'By conviction' if position_scale_by_conviction else 'Fixed'}")
    
    def calculate_spread_zscore(self, prices, etf1, etf2, hedge_ratio, lookback):

        spread = prices[etf1] - hedge_ratio * prices[etf2]
        spread_mean = spread.rolling(lookback).mean()
        spread_std = spread.rolling(lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-8)
        return spread, z_score
    
    def calculate_copula_signal_strength(self, cond_prob, z_score):

        if z_score > 0:  # Spread is high
            # For short signal, want copula prob to be low (ETF1 overvalued)
            # Low prob (0.0) = agrees, High prob (1.0) = disagrees
            agreement = 1 - 2 * cond_prob  # Maps [0,1] to [1,-1]
        else:  # Spread is low
            # For long signal, want copula prob to be high (ETF1 undervalued)
            # High prob (1.0) = agrees, Low prob (0.0) = disagrees
            agreement = 2 * cond_prob - 1  # Maps [0,1] to [-1,1]
        
        return agreement
    
    def transform_to_uniform(self, data, lookback_data):

        if not np.isfinite(data):
            return 0.5
        
        clean_lookback = lookback_data[np.isfinite(lookback_data)]
        if len(clean_lookback) < 20:
            return 0.5
        
        n = len(clean_lookback)
        rank = np.sum(clean_lookback <= data)
        uniform = (rank + 0.5) / (n + 1)
        return np.clip(uniform, 0.001, 0.999)
    
    def calculate_conditional_prob(self, u, v, copula_model):

        try:
            copula_type = copula_model["copula_type"]
            copula = copula_model["copula"]
            
            if copula_type == "student":
                params = copula.params
                df = float(params.df)
                rho = float(params.rho[0]) if hasattr(params.rho, '__len__') else float(params.rho)
                
                if not (0 < u < 1 and 0 < v < 1) or df <= 2 or abs(rho) >= 1:
                    return 0.5
                
                t1 = student_t.ppf(u, df)
                t2 = student_t.ppf(v, df)
                
                var_scale = (df + t2**2) / (df + 1)
                mu_cond = rho * t2
                sigma_cond = np.sqrt((1 - rho**2) * var_scale)
                
                if sigma_cond <= 0:
                    return 0.5
                
                return student_t.cdf(t1, df=df+1, loc=mu_cond, scale=sigma_cond)
            
            elif copula_type == "gaussian":
                params = copula.params
                rho = float(params[0,1]) if params.ndim > 1 else float(params[0])
                
                if not (0 < u < 1 and 0 < v < 1) or abs(rho) >= 1:
                    return 0.5
                
                z1 = norm.ppf(u)
                z2 = norm.ppf(v)
                
                mu_cond = rho * z2
                sigma_cond = np.sqrt(1 - rho**2)
                
                return norm.cdf(z1, loc=mu_cond, scale=sigma_cond)
            
            # For other copulas, return neutral
            return 0.5
            
        except:
            return 0.5
    
    def generate_signal(self, prices_df, pair_data, start_date=None, end_date=None):
        """Generate trading signals with copula as filter/scaler"""
        
        pair_name = f"{pair_data['etf1']}_{pair_data['etf2']}"
        
        if start_date:
            prices_df = prices_df[start_date:]
        if end_date:
            prices_df = prices_df[:end_date]
        
        etf1, etf2 = pair_data["etf1"], pair_data["etf2"]
        copula_model = pair_data.get("copula_model")
        
        # Calculate spread and z-scores
        spread, z_score = self.calculate_spread_zscore(
            prices_df, etf1, etf2, pair_data["hedge_ratio"], self.lookback_days
        )
        
        # Initialize results
        dates = prices_df.index
        results = pd.DataFrame(index=dates, columns=[
            "position", "position_size", "entry_signal", "exit_signal",
            "spread", "z_score", "cond_prob", "copula_agreement",
            "entry_price", "entry_date"
        ])
        
        position = 0
        position_size = 1.0 
        entry_price = None
        entry_date = None
        entry_idx = None
        
        # Counters
        signals_generated = 0
        signals_vetoed = 0
        signals_scaled_up = 0
        signals_scaled_down = 0
        
        # Generate signals
        for i in range(self.lookback_days, len(prices_df)):
            current_date = dates[i]
            current_z = z_score.iloc[i]
            
            if not np.isfinite(current_z):
                continue
            
            results.loc[current_date, "spread"] = spread.iloc[i]
            results.loc[current_date, "z_score"] = current_z
            
            cond_prob = 0.5
            copula_agreement = 0.0
            
            if self.use_copula_filter and copula_model:

                lookback_prices1 = prices_df[etf1].iloc[max(0, i-self.lookback_days):i]
                lookback_prices2 = prices_df[etf2].iloc[max(0, i-self.lookback_days):i]
                
                if len(lookback_prices1) > 20:
                    lookback_returns1 = np.log(lookback_prices1 / lookback_prices1.shift(1)).dropna().values
                    lookback_returns2 = np.log(lookback_prices2 / lookback_prices2.shift(1)).dropna().values
                    
                    if len(lookback_returns1) > 10 and len(lookback_returns2) > 10:
                        current_ret1 = np.log(prices_df[etf1].iloc[i] / prices_df[etf1].iloc[i-1])
                        current_ret2 = np.log(prices_df[etf2].iloc[i] / prices_df[etf2].iloc[i-1])
                        
                        u = self.transform_to_uniform(current_ret1, lookback_returns1)
                        v = self.transform_to_uniform(current_ret2, lookback_returns2)
                        
                        cond_prob = self.calculate_conditional_prob(u, v, copula_model)
                        copula_agreement = self.calculate_copula_signal_strength(cond_prob, current_z)
                        
                        results.loc[current_date, "cond_prob"] = cond_prob
                        results.loc[current_date, "copula_agreement"] = copula_agreement
            
            # ENTRY LOGIC
            if position == 0:
                base_signal = False
                signal_direction = 0
                
                # Primary signal: Z-score
                if abs(current_z) > self.entry_z_score:
                    base_signal = True
                    signal_direction = -1 if current_z > 0 else 1  # Short if high, long if low
                    signals_generated += 1
                
                if base_signal:
                    # Check copula veto (only if strongly disagrees)
                    if self.use_copula_filter and copula_agreement < -self.copula_veto_threshold:
                        # Copula strongly disagrees, veto the signal
                        signals_vetoed += 1
                        base_signal = False
                    else:
                        # Determine position size based on conviction
                        if self.position_scale_by_conviction:
                            if copula_agreement > 0.5:
                                position_size = 1.5  # Scale up if strong agreement
                                signals_scaled_up += 1
                            elif copula_agreement < -0.25:
                                position_size = 0.5  # Scale down if mild disagreement
                                signals_scaled_down += 1
                            else:
                                position_size = 1.0  # Normal size
                        else:
                            position_size = 1.0
                
                # Execute entry
                if base_signal:
                    position = signal_direction
                    results.loc[current_date, "position_size"] = position_size
                    results.loc[current_date, "entry_signal"] = True
                    entry_price = spread.iloc[i]
                    entry_date = current_date
                    entry_idx = i
            
            # EXIT LOGIC
            elif position != 0:
                exit_signal = False
                
                # Mean reversion exit
                if position == 1 and current_z > -self.exit_z_score:
                    exit_signal = True
                elif position == -1 and current_z < self.exit_z_score:
                    exit_signal = True
                
                # Stop loss
                if abs(current_z) > self.stop_loss_z_score:
                    exit_signal = True
                
                # Time stop
                if entry_idx and (i - entry_idx) >= self.max_holding_days:
                    exit_signal = True
                
                if exit_signal:
                    results.loc[current_date, "exit_signal"] = True
                    results.loc[current_date, "entry_date"] = entry_date
                    position = 0
                    position_size = 1.0
                    entry_price = None
                    entry_date = None
                    entry_idx = None
            
            results.loc[current_date, "position"] = position
        
        # Clean up results
        results = results.dropna(subset=["z_score"])
        results["entry_signal"] = results["entry_signal"].fillna(False)
        results["exit_signal"] = results["exit_signal"].fillna(False)
        results["position_size"] = results["position_size"].fillna(1.0)
        
        # Report statistics
        num_entries = results["entry_signal"].sum()
        num_exits = results["exit_signal"].sum()
        
        # if self.use_copula_filter:
        #     print(f"  Copula impact:")
        #     print(f"    - Signals vetoed: {signals_vetoed}/{signals_generated} ({signals_vetoed/max(signals_generated,1)*100:.1f}%)")
        #     if self.position_scale_by_conviction:
        #         print(f"    - Positions scaled up: {signals_scaled_up}")
        #         print(f"    - Positions scaled down: {signals_scaled_down}")
        
        # Show copula agreement distribution
        if "copula_agreement" in results.columns and num_entries > 0:
            entry_agreements = results[results["entry_signal"] == True]["copula_agreement"].dropna()
            # if len(entry_agreements) > 0:
            #     print(f"    - Avg agreement on entries: {entry_agreements.mean():.2f}")
        
        return results
    
    def generate_batch_signals(self, prices_df, selected_pairs, start_date=None, end_date=None):
        """Generate signals for all pairs"""
        
        all_signals = {}

        total_entries = 0
        total_vetoed = 0
        
        for pair_data in selected_pairs:
            signals = self.generate_signal(prices_df, pair_data, start_date, end_date)
            pair_name = f"{pair_data['etf1']}_{pair_data['etf2']}"
            all_signals[pair_name] = signals
            
            if len(signals) > 0:
                entries = signals[signals["entry_signal"]].shape[0]
                total_entries += entries
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Total entries: {total_entries} across {len(selected_pairs)} pairs")
        print(f"  Average per pair: {total_entries/max(len(selected_pairs),1):.1f}")
        print(f"{'='*60}")
        
        return all_signals


def test_optimized_strategy():
    """Test the optimized strategy"""
    from src.pair_selection import PairSelector
    from src.config import PROCESSED_DIR
    from src.backtesting import ImprovedBacktester
    
    # Load data
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    
    # Run pair selection
    selector = PairSelector(
        prices, returns,
        train_end_date=pd.to_datetime("2022-12-31"),
        test_end_date=pd.to_datetime("2025-10-31"),
        top_n=5,
        coint_pvalue=0.05,
        min_half_life=5,
        max_half_life=90
    )
    
    selected_pairs = selector.run_selection()
    test_prices, test_returns = selector.get_test_data()
    
    # Test configurations
    configs = [
        {"name": "Pure Z-Score", "use_copula": False, "scale": False},
       # {"name": "Z + Copula Veto", "use_copula": True, "scale": False},
       # {"name": "Z + Copula Scaling", "use_copula": True, "scale": True},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"TESTING: {config['name']}") 
        print(f"{'='*80}")
        
        # Generate signals
        signal_gen = OptimizedSignalGenerator(
            entry_z_score=1.8,
            use_copula_filter=config["use_copula"],
            position_scale_by_conviction=config["scale"],
            copula_veto_threshold=0.7  # Only veto if copula strongly disagrees
        )
        
        all_signals = signal_gen.generate_batch_signals(test_prices, selected_pairs)
        
        # Run backtest
        backtester = ImprovedBacktester(
            initial_capital=100_000,
            tcost_bps=5,
            slippage_bps=3
        )
        
        portfolio, metrics = backtester.run_backtest(test_prices, all_signals)
        results[config["name"]] = metrics
    
    # Display comparison
    print(f"\n\n{'='*80}")
    print("STRATEGY COMPARISON")
    print(f"{'='*80}")
    
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 'N/A')}")
        print(f"  Total Return: {metrics.get('Total Return', 'N/A')}")
        print(f"  Max Drawdown: {metrics.get('Maximum Drawdown', 'N/A')}")
        print(f"  Win Rate: {metrics.get('Win Rate (Daily)', 'N/A')}")
    
    return results


if __name__ == "__main__":
    results = test_optimized_strategy()