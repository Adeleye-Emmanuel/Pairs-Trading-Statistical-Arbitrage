import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import PROCESSED_DIR
from src.pair_selection import PairSelector
from src.signal_generation import SignalGenerator
from src.backtesting import Backtester

# --- Configuration for the Sweep ---
LOOKBACK_DAYS = 60 # Fixed for this round
THRESHOLD_COMBOS = [
    (0.90, 0.55),
    (0.90, 0.65),
    (0.85, 0.65),
    (0.80, 0.70)
]
# -----------------------------------

def run_pipeline(prices, returns, entry_threshold, exit_threshold):
    """Runs the full pipeline for a given set of parameters."""
    
    # 1. Pair Selection (Only needs to run once as it's not a parameter being swept)
    selector = PairSelector(prices, returns, 
                            pd.to_datetime("2024-12-31"), 
                            pd.to_datetime("2025-12-31"), top_n=10)
    selected_pairs = selector.run_selection()
    test_prices, _ = selector.get_test_data()

    # 2. Signal Generation with swept parameters
    signal_gen = SignalGenerator(
        lookback_days=LOOKBACK_DAYS,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold
    )
    all_signals = signal_gen.generate_batch_signals(test_prices, selected_pairs)

    # 3. Backtesting
    backtester = Backtester()
    _, metrics = backtester.run_backtest(test_prices, all_signals)
    
    return metrics


if __name__ == "__main__":
    # Load data once
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)

    print(f"Starting Threshold Sweep (Lookback fixed at {LOOKBACK_DAYS} days)...")
    
    all_results = {}
    
    for entry_t, exit_t in THRESHOLD_COMBOS:
        print(f"\nRunning test: Entry={entry_t}, Exit={exit_t}")
        
        try:
            metrics = run_pipeline(prices, returns, entry_t, exit_t)
            all_results[f"E{entry_t}_X{exit_t}"] = metrics
        except Exception as e:
            print(f"Error encountered for E{entry_t}_X{exit_t}: {e}")
            all_results[f"E{entry_t}_X{exit_t}"] = pd.Series({"Sharpe Ratio": float('nan')})

    # Consolidate and display results
    results_df = pd.DataFrame(all_results).T
    
    print("\n" + "="*50)
    print("✨ Optimization Results: Threshold Sweep ✨")
    print("="*50)
    
    # Sort by Sharpe Ratio to find the best performing combination
    sorted_results = results_df.sort_values(by="Sharpe Ratio", ascending=False)
    
    print(sorted_results[["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"]])
    print("="*50)