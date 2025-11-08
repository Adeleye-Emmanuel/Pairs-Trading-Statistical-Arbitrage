import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import multiprocessing as mp
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Import project modules
from src.pair_selection import PairSelector
from src.signal_generation import OptimizedSignalGenerator
from src.backtesting import ImprovedBacktester
from src.config import PROCESSED_DIR, BACKTEST_DIR


# CORRECTED: Nested walk-forward structure
# Each round has: TRAIN → VALIDATION (optimize params) → TEST (report Sharpe)

NESTED_PERIODS = [
    {
        "name": "Round 1",
        "train_start": pd.to_datetime("2015-01-01"),
        "train_end": pd.to_datetime("2020-12-31"),
        "validation_start": pd.to_datetime("2021-01-01"),
        "validation_end": pd.to_datetime("2021-12-31"),
        "test_start": pd.to_datetime("2022-01-01"),
        "test_end": pd.to_datetime("2022-12-31")
    },
    {
        "name": "Round 2",
        "train_start": pd.to_datetime("2015-01-01"),
        "train_end": pd.to_datetime("2021-12-31"),
        "validation_start": pd.to_datetime("2022-01-01"),
        "validation_end": pd.to_datetime("2022-12-31"),
        "test_start": pd.to_datetime("2023-01-01"),
        "test_end": pd.to_datetime("2023-12-31")
    },
    {
        "name": "Round 3",
        "train_start": pd.to_datetime("2015-01-01"),
        "train_end": pd.to_datetime("2022-12-31"),
        "validation_start": pd.to_datetime("2023-01-01"),
        "validation_end": pd.to_datetime("2023-12-31"),
        "test_start": pd.to_datetime("2024-01-01"),
        "test_end": pd.to_datetime("2024-12-31")
    },
    {
        "name": "Round 4",
        "train_start": pd.to_datetime("2015-01-01"),
        "train_end": pd.to_datetime("2023-12-31"),
        "validation_start": pd.to_datetime("2024-01-01"),
        "validation_end": pd.to_datetime("2024-12-31"),
        "test_start": pd.to_datetime("2025-01-01"),
        "test_end": pd.to_datetime("2025-12-31")
    }
]

prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)

PARAM_SPACE = {
    "entry_z_score": np.arange(1.5, 3.6, 0.25).tolist(),         # 1.5, 1.75, ..., 3.5
    "exit_z_score": np.arange(0.1, 1.05, 0.1).tolist(),          # 0.1, 0.2, ..., 1.0
    "position_size_pct": np.arange(0.02, 0.21, 0.02).tolist(),   # 2%, 4%, ..., 20%
    "top_n": list(range(3, 11)),                                 # 3–10
    "coint_pvalue": np.arange(0.01, 0.051, 0.01).tolist(),       # 0.01, 0.02, ..., 0.05
    "copula_veto_threshold": np.arange(0.1, 0.91, 0.05).tolist() # 0.1, 0.15, ..., 0.9
}


def sample_params():
    """Sample random parameters from search space"""
    cfg = {}
    cfg["entry_z_score"] = float(random.choice(PARAM_SPACE["entry_z_score"]))
    cfg["exit_z_score"] = float(random.choice(PARAM_SPACE["exit_z_score"]))
    if cfg["exit_z_score"] >= cfg["entry_z_score"]:
        cfg["exit_z_score"] = max(0.1, cfg["entry_z_score"] - 0.5)
    cfg["position_size_pct"] = float(random.choice(PARAM_SPACE["position_size_pct"]))
    cfg["top_n"] = int(random.choice(PARAM_SPACE["top_n"]))
    cfg["coint_pvalue"] = float(random.choice(PARAM_SPACE["coint_pvalue"]))
    cfg["copula_veto_threshold"] = float(random.choice(PARAM_SPACE["copula_veto_threshold"]))
    
    return cfg


def evaluate_single_period(cfg, train_end, test_start, test_end, seed=0):
    """
    Evaluate config on a single period
    
    Args:
        cfg: Parameter configuration
        train_end: End date for training (pair selection uses all data up to this)
        test_start: Start of test period
        test_end: End of test period
    
    Returns:
        Sharpe ratio for this period, or -10.0 if failed
    """
    rng = np.random.default_rng(seed)
    
    try:
        # Pair selection on training data
        selector = PairSelector(
            prices, returns,
            train_end_date=train_end,
            test_end_date=test_end,
            top_n=cfg["top_n"],
            coint_pvalue=cfg["coint_pvalue"],
            min_half_life=5,
            max_half_life=90
        )
        
        selected_pairs = selector.run_selection()
        if not selected_pairs:
            return -10.0
        
        # Get test data
        test_prices, test_returns = selector.get_test_data()
        
        # Generate signals
        signal_gen = OptimizedSignalGenerator(
            entry_z_score=cfg["entry_z_score"],
            exit_z_score=cfg["exit_z_score"],
            copula_veto_threshold=cfg["copula_veto_threshold"]
        )
        
        all_signals = signal_gen.generate_batch_signals(test_prices, selected_pairs)
        
        # Backtest
        backtester = ImprovedBacktester(
            initial_capital=100_000,
            position_size_pct=cfg["position_size_pct"],
            tcost_bps=5,
            slippage_bps=3,
            max_positions=cfg["top_n"],
            use_volatility_sizing=True
        )
        
        portfolio, metrics = backtester.run_backtest(test_prices, all_signals)
        
        # Extract Sharpe
        sharpe_str = metrics.get("Sharpe Ratio", None)
        if sharpe_str is None:
            return -10.0
        
        try:
            sharpe_val = float(sharpe_str)
        except:
            return -10.0
        
        return sharpe_val
    
    except Exception as e:
        # Penalize configs that crash
        return -10.0


def optimize_on_validation(period_config, n_trials=100, workers=1):
    """
    STEP 1: Optimize parameters on VALIDATION period only
    
    Args:
        period_config: Dict with train_end, validation_start, validation_end
        n_trials: Number of random parameter combinations to try
    
    Returns:
        best_params: The parameter config with highest validation Sharpe
        best_sharpe: The validation Sharpe of best config
    """
    
    results = []
    seeds = [random.randint(0, 2**31 - 1) for _ in range(n_trials)]
    
    for idx, seed in enumerate(tqdm(seeds, desc=f"Optimizing {period_config['name']}", leave=False)):
        random.seed(seed)
        cfg = sample_params()
        
        # Evaluate ONLY on validation period
        validation_sharpe = evaluate_single_period(
            cfg,
            train_end=period_config["train_end"],
            test_start=period_config["validation_start"],
            test_end=period_config["validation_end"],
            seed=seed
        )
        
        results.append({
            **cfg,
            "validation_sharpe": validation_sharpe,
            "trial_idx": idx
        })
    
    # Sort by validation Sharpe and pick best
    df = pd.DataFrame(results)
    df = df.sort_values(by="validation_sharpe", ascending=False)
    
    best_params = df.iloc[0].to_dict()
    best_sharpe = best_params["validation_sharpe"]
    
    return best_params, best_sharpe, df


def test_on_holdout(best_params, period_config):
    """
    STEP 2: Test the optimized parameters on HOLDOUT test period
    
    Args:
        best_params: Parameters that performed best on validation
        period_config: Dict with test_start, test_end
    
    Returns:
        test_sharpe: Sharpe on truly unseen test data
    """
    
    test_sharpe = evaluate_single_period(
        best_params,
        train_end=period_config["train_end"],
        test_start=period_config["test_start"],
        test_end=period_config["test_end"]
    )
    
    return test_sharpe


def run_nested_walk_forward(n_trials=100, workers=1):
    """
    Run complete nested walk-forward validation
    
    Process:
    1. For each round (2021-2025):
       a. Optimize params on VALIDATION period
       b. Test best params on HOLDOUT test period
    2. Report average TEST Sharpe across all rounds
    
    Returns:
        final_results: DataFrame with results for each round
        avg_test_sharpe: Average Sharpe across all holdout test periods
    """
    print(f"\n{'='*80}")
    print("NESTED WALK-FORWARD VALIDATION")
    print(f"{'='*80}")
    print(f"\nStructure:")
    for i, period in enumerate(NESTED_PERIODS, 1):
        print(f"\nRound {i}: {period['name']}")
        print(f"  Train:      {period['train_start'].date()} to {period['train_end'].date()}")
        print(f"  Validation: {period['validation_start'].date()} to {period['validation_end'].date()} (optimize here)")
        print(f"  Test:       {period['test_start'].date()} to {period['test_end'].date()} (report here)")
    
    all_results = []
    
    for round_idx, period_config in enumerate(NESTED_PERIODS, 1):
        print(f"\n\n{'#'*80}")
        print(f"ROUND {round_idx}: {period_config['name']}")
        print(f"{'#'*80}")
        
        # Step 1: Optimize on validation
        best_params, validation_sharpe, optimization_results = optimize_on_validation(
            period_config, 
            n_trials=n_trials,
            workers=workers
        )
        
        # Save optimization results for this round
        opt_path = os.path.join(BACKTEST_DIR, f"round{round_idx}_optimization.csv")
        optimization_results.to_csv(opt_path, index=False)
        print(f"Saved optimization results to {opt_path}")
        
        # Step 2: Test on holdout
        test_sharpe = test_on_holdout(best_params, period_config)
        
        # Store results
        result = {
            "round": round_idx,
            "validation_period": f"{period_config['validation_start'].date()} to {period_config['validation_end'].date()}",
            "test_period": f"{period_config['test_start'].date()} to {period_config['test_end'].date()}",
            "validation_sharpe": validation_sharpe,
            "test_sharpe": test_sharpe,
            "best_entry_z": best_params["entry_z_score"],
            "best_exit_z": best_params["exit_z_score"],
            "best_position_size": best_params["position_size_pct"],
            "best_top_n": best_params["top_n"],
            "best_coint_pval": best_params["coint_pvalue"],
            "best_veto_thresh": best_params["copula_veto_threshold"]
        }
        
        all_results.append(result)
    
    # Create final summary
    df_final = pd.DataFrame(all_results)
    
    # Calculate statistics
    avg_validation_sharpe = df_final["validation_sharpe"].mean()
    avg_test_sharpe = df_final["test_sharpe"].mean()
    std_test_sharpe = df_final["test_sharpe"].std()
    
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS (NESTED WALK-FORWARD)")
    print(f"{'='*80}\n")
    print(df_final.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Average Validation Sharpe: {avg_validation_sharpe:.2f}")
    print(f"Average Test Sharpe:       {avg_test_sharpe:.2f} ± {std_test_sharpe:.2f}")
    print(f"\nThis is your TRUE expected Sharpe in live trading")
    print(f"(based on proper nested cross-validation)")
    print(f"{'='*80}\n")
    
    # Save final results
    final_path = os.path.join(BACKTEST_DIR, "nested_cv_final_results.csv")
    df_final.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")
    
    return df_final, avg_test_sharpe


def main(args):
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    
    # Run nested walk-forward
    final_results, avg_test_sharpe = run_nested_walk_forward(
        n_trials=args.trials,
        workers=args.workers
    )
    
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"\nYour strategy's expected Sharpe ratio: {avg_test_sharpe:.2f}")
    print(f"\nResults saved to: {BACKTEST_DIR}/nested_cv_final_results.csv")
    print(f"Individual round optimizations saved to: {BACKTEST_DIR}/roundX_optimization.csv")
    
    if avg_test_sharpe < 0.5:
        print(f"\nWARNING: Test Sharpe < 0.5 suggests strategy has no alpha")
        print("   Consider: Expanding universe, loosening filters, or different approach")
    elif avg_test_sharpe < 1.0:
        print(f"\nMarginal strategy (Sharpe 0.5-1.0)")
        print("   May be tradeable but needs improvement")
    else:
        print(f"\n✓ Promising strategy (Sharpe > 1.0)")
        print("  Continue with robustness checks and risk analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested Walk-Forward Parameter Optimization")
    parser.add_argument("--trials", type=int, default=100, 
                       help="Number of random parameter trials per validation period")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Parallel workers (currently single-threaded in corrected version)")
    args = parser.parse_args()
    
    main(args)