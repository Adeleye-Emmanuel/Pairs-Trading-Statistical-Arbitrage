import numpy as np
import pandas as pd
from scipy.stats import norm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def debug_signal_conditions(test_prices, selected_pairs, signal_gen):
    """
    Debug why copula-based signals aren't triggering
    """
    
    print("\n" + "="*80)
    print("DEBUGGING COPULA SIGNAL CONDITIONS")
    print("="*80)
    
    for pair_data in selected_pairs[:3]:  # Check first 3 pairs
        etf1, etf2 = pair_data['etf1'], pair_data['etf2']
        pair_name = f"{etf1}_{etf2}"
        
        print(f"\n\nPair: {pair_name}")
        print("-"*40)
        
        # Calculate z-scores
        prices1 = test_prices[etf1].dropna()
        prices2 = test_prices[etf2].dropna()
        
        spread = prices1 - pair_data['hedge_ratio'] * prices2
        spread_mean = spread.rolling(60, min_periods=30).mean()
        spread_std = spread.rolling(60, min_periods=30).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-8)
        
        # Collect statistics
        z_high = (z_score > 2.0).sum()
        z_low = (z_score < -2.0).sum()
        z_extreme = z_high + z_low
        
        print(f"Z-Score Statistics:")
        print(f"  Range: [{z_score.min():.2f}, {z_score.max():.2f}]")
        print(f"  Days with |z| > 2: {z_extreme} ({z_extreme/len(z_score)*100:.1f}%)")
        print(f"  Days with z > 2: {z_high}")
        print(f"  Days with z < -2: {z_low}")
        
        # Now check copula probabilities on high z-score days
        copula_model = pair_data.get('copula_model')
        if copula_model and copula_model['copula_type']:
            print(f"\nCopula: {copula_model['copula_type']}")
            
            # Sample some high z-score days
            high_z_days = z_score[z_score.abs() > 2.0].index[:10]  # First 10 extreme days
            
            if len(high_z_days) > 0:
                print(f"\nAnalyzing {len(high_z_days)} extreme z-score days:")
                print("Date         | Z-Score | Cond Prob | Would Enter?")
                print("-"*50)
                
                for date in high_z_days:
                    if date not in test_prices.index:
                        continue
                        
                    # Get the returns for that day
                    idx = test_prices.index.get_loc(date)
                    if idx > 60:
                        # Calculate returns
                        ret1 = np.log(test_prices[etf1].iloc[idx] / test_prices[etf1].iloc[idx-1])
                        ret2 = np.log(test_prices[etf2].iloc[idx] / test_prices[etf2].iloc[idx-1])
                        
                        # Get lookback returns for ECDF
                        lookback_ret1 = np.log(test_prices[etf1].iloc[idx-60:idx] / test_prices[etf1].iloc[idx-60:idx].shift(1)).dropna()
                        lookback_ret2 = np.log(test_prices[etf2].iloc[idx-60:idx] / test_prices[etf2].iloc[idx-60:idx].shift(1)).dropna()
                        
                        # Transform to uniform
                        u = (lookback_ret1 <= ret1).mean()
                        v = (lookback_ret2 <= ret2).mean()
                        u = np.clip(u, 0.001, 0.999)
                        v = np.clip(v, 0.001, 0.999)
                        
                        # For debugging, calculate what the conditional probability would be
                        # (simplified - actual calculation depends on copula type)
                        z_val = z_score.loc[date]
                        
                        # Check signal conditions
                        z_condition = abs(z_val) > 2.0
                        
                        # Check copula condition (depends on direction)
                        if z_val > 2.0:  # Short signal
                            copula_condition = u < 0.023  # Need extreme upper tail
                            signal = z_condition and copula_condition
                        else:  # Long signal
                            copula_condition = u > 0.977  # Need extreme lower tail
                            signal = z_condition and copula_condition
                        
                        print(f"{date.date()} | {z_val:7.2f} | {u:9.3f} | {'YES' if signal else 'NO'}")
                        
                        if not signal and z_condition:
                            print(f"         -> Z-score triggered but copula didn't (u={u:.3f} not extreme enough)")

# Also create a function to test different threshold combinations
def test_threshold_combinations(test_prices, selected_pairs):
    """
    Test different threshold combinations to find what works
    """
    print("\n" + "="*80)
    print("TESTING DIFFERENT THRESHOLD COMBINATIONS")
    print("="*80)
    
    from src.signal_generation import SignalGenerator
    
    # Test different threshold combinations
    test_configs = [
        {"entry_z": 2.0, "copula_thresh": 0.05, "use_copula": True},
        {"entry_z": 2.0, "copula_thresh": 0.10, "use_copula": True},
        {"entry_z": 1.5, "copula_thresh": 0.10, "use_copula": True},
        {"entry_z": 2.0, "copula_thresh": None, "use_copula": False},  # Pure z-score
    ]
    
    results = []
    
    for config in test_configs:
        if config["use_copula"]:
            # Map z-score to probability thresholds
            thresh_high = 1 - config["copula_thresh"]
            thresh_low = config["copula_thresh"]
            desc = f"Z>{config['entry_z']} + Copula<{config['copula_thresh']}"
        else:
            desc = f"Z>{config['entry_z']} only"
        
        # Count signals (simplified - you'd need to run actual signal generation)
        print(f"\nConfig: {desc}")
        
        # This is a simplified check - in practice you'd run the full signal generation
        total_signals = 0
        for pair_data in selected_pairs[:5]:  # Test on first 5 pairs
            etf1, etf2 = pair_data['etf1'], pair_data['etf2']
            prices1 = test_prices[etf1].dropna()
            prices2 = test_prices[etf2].dropna()
            
            spread = prices1 - pair_data['hedge_ratio'] * prices2
            spread_mean = spread.rolling(60, min_periods=30).mean()
            spread_std = spread.rolling(60, min_periods=30).std()
            z_score = (spread - spread_mean) / (spread_std + 1e-8)
            
            # Count potential signals
            z_signals = (abs(z_score) > config["entry_z"]).sum()
            total_signals += z_signals
            
            print(f"  {etf1}_{etf2}: {z_signals} potential signals")
        
        print(f"  Total: {total_signals} signals across 5 pairs")

if __name__ == "__main__":
    from src.pair_selection import PairSelector
    from src.signal_generation import SignalGenerator
    from src.config import PROCESSED_DIR
    
    # Load data
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    
    # Run pair selection
    selector = PairSelector(
        prices, returns,
        train_end_date=pd.to_datetime("2023-12-31"),
        test_end_date=pd.to_datetime("2024-12-31"),
        top_n=10,
        coint_pvalue=0.01,
        min_half_life=5,
        max_half_life=60
    )
    
    selected_pairs = selector.run_selection()
    test_prices, test_returns = selector.get_test_data()
    
    # Debug why copula signals aren't working
    signal_gen = SignalGenerator(use_dynamic_threshold=True)
    debug_signal_conditions(test_prices, selected_pairs, signal_gen)
    
    # Test different thresholds
    test_threshold_combinations(test_prices, selected_pairs)