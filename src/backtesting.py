import numpy as np
import pandas as pd
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PROCESSED_DIR, BACKTEST_DIR

class ImprovedBacktester:

    def __init__(self,
                 initial_capital=100_000,
                 position_size_pct=0.3,      # 30% of capital per pair
                 tcost_bps=5,                # 5 basis points per leg (20 bps round trip)
                 slippage_bps=3,             # 3 basis points slippage per leg
                 max_positions=5,            # Maximum concurrent positions
                 use_volatility_sizing=True,
                 target_vol=0.05):           # 5% daily volatility target
        """
        Realistic backtester with proper transaction costs and risk management
        
        Total cost per round trip = 4 legs × (tcost + slippage) = 4 × 8 = 32 bps
        """
        
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.tcost = tcost_bps / 10000  # Convert to decimal
        self.slippage = slippage_bps / 10000
        self.max_positions = max_positions
        self.use_volatility_sizing = use_volatility_sizing
        self.target_vol = target_vol
        
        print(f"\nBacktest Configuration:")
        print(f"  Initial Capital: ${initial_capital:,.0f}")
        print(f"  Position Size: {position_size_pct*100:.1f}% per pair")
        print(f"  Transaction Cost: {tcost_bps} bps per leg")
        print(f"  Slippage: {slippage_bps} bps per leg")
        print(f"  Total Round Trip Cost: {4*(tcost_bps+slippage_bps)} bps")
        print(f"  Max Concurrent Positions: {max_positions}")
        print(f"  Volatility Sizing: {use_volatility_sizing}")

    def calculate_position_size(self, spread_vol, current_capital):
        """
        Calculate position size with volatility scaling
        """
        base_size = current_capital * self.position_size_pct
        
        if self.use_volatility_sizing and spread_vol > 0:
            # Scale position inversely to volatility
            vol_scalar = self.target_vol / spread_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Cap leverage
            return base_size * vol_scalar
        
        return base_size

    def calculate_daily_pnl(self, prices_df, signals, pair_name):
        """
        Calculate realistic P&L with proper transaction costs
        """
        etf1, etf2 = pair_name.split("_")
        
        # Merge signals with prices
        data = signals.copy()
        data = data.join(prices_df[[etf1, etf2]], how='inner')
        
        # Forward fill prices for missing data
        data[etf1] = data[etf1].ffill()
        data[etf2] = data[etf2].ffill()
        data = data.dropna(subset=[etf1, etf2])
        
        if len(data) == 0:
            return pd.DataFrame()
        
        # Calculate returns
        data["ret1"] = data[etf1].pct_change()
        data["ret2"] = data[etf2].pct_change()
        
        # Get hedge ratio (from signals or recalculate)
        if "hr_roll" not in data.columns:
            # Use static hedge ratio from pair selection
            data["hr_roll"] = 1.0  # Default if not provided
        else:
            data["hr_roll"] = data["hr_roll"].ffill()
        
        # Calculate spread returns
        data["spread_ret"] = data["ret1"] - data["hr_roll"] * data["ret2"]
        
        # Shift positions to avoid look-ahead bias
        data["position"] = data["position"].ffill().shift(1).fillna(0)
        
        # Calculate spread volatility for position sizing
        spread_vol = data["spread_ret"].rolling(20, min_periods=10).std()
        data["spread_vol"] = spread_vol.fillna(spread_vol.expanding().std())
        
        # Track position changes for transaction costs
        data["pos_change"] = data["position"].diff().fillna(data["position"])
        
        # Calculate transaction costs (4 legs per round trip)
        # Entry: 2 legs (buy/sell each ETF), Exit: 2 legs
        total_cost_per_leg = self.tcost + self.slippage
        data["num_legs"] = np.abs(data["pos_change"]) * 2  # 2 legs per position change
        data["trading_cost_pct"] = data["num_legs"] * total_cost_per_leg
        
        # Calculate gross and net returns
        data["gross_ret"] = data["position"] * data["spread_ret"]
        data["net_ret"] = data["gross_ret"] - data["trading_cost_pct"]
        
        # Track cumulative costs for analysis
        data["cumulative_costs"] = data["trading_cost_pct"].cumsum()
        
        return data
    
    def calculate_portfolio_metrics(self, all_pair_results):
        """
        Combine individual pair results into portfolio with position limits
        """
        if not all_pair_results:
            return pd.DataFrame(), {}
        
        # Align all dataframes to same index
        all_dates = pd.Index([])
        for df in all_pair_results.values():
            if len(df) > 0:
                all_dates = all_dates.union(df.index)
        
        if len(all_dates) == 0:
            return pd.DataFrame(), {}
        
        # Create portfolio dataframe
        portfolio = pd.DataFrame(index=all_dates)
        
        # Track individual pair returns and positions
        for pair_name, pair_data in all_pair_results.items():
            if len(pair_data) == 0:
                continue
            
            portfolio[f"{pair_name}_ret"] = pair_data["net_ret"]
            portfolio[f"{pair_name}_pos"] = pair_data["position"].fillna(0)
            portfolio[f"{pair_name}_cost"] = pair_data["trading_cost_pct"].fillna(0)
        
        # Fill NaN with 0 (no position/return)
        portfolio = portfolio.fillna(0)
        
        # Apply position limits
        position_cols = [c for c in portfolio.columns if c.endswith("_pos")]
        num_active = (portfolio[position_cols] != 0).sum(axis=1)
        
        # Scale down positions if exceeding max
        for idx in portfolio.index:
            if num_active[idx] > self.max_positions:
                active_pairs = portfolio.loc[idx, position_cols][portfolio.loc[idx, position_cols] != 0].index
                scale_factor = self.max_positions / len(active_pairs)
                
                for pos_col in active_pairs:
                    ret_col = pos_col.replace("_pos", "_ret")
                    portfolio.loc[idx, ret_col] *= scale_factor
        
        # Calculate portfolio returns (equal weight among active positions)
        ret_cols = [c for c in portfolio.columns if c.endswith("_ret")]
        portfolio["active_positions"] = (portfolio[position_cols] != 0).sum(axis=1)
        portfolio["active_positions"] = portfolio["active_positions"].replace(0, 1)  # Avoid division by zero
        
        # Equal weight among active positions
        portfolio["portfolio_ret"] = portfolio[ret_cols].sum(axis=1) / portfolio["active_positions"]
        
        # Calculate total costs
        cost_cols = [c for c in portfolio.columns if c.endswith("_cost")]
        portfolio["total_costs"] = portfolio[cost_cols].sum(axis=1) / portfolio["active_positions"]
        
        # Calculate cumulative returns
        portfolio["cumulative_ret"] = (1 + portfolio["portfolio_ret"]).cumprod()
        portfolio["equity_curve"] = self.initial_capital * portfolio["cumulative_ret"]
        
        return portfolio, self.calculate_performance_metrics(portfolio)
    
    def calculate_performance_metrics(self, portfolio, risk_free_rate=0.02):
        """
        Calculate comprehensive performance metrics
        """
        if len(portfolio) == 0:
            return {}
        
        metrics = {}
        returns = portfolio["portfolio_ret"].values
        equity = portfolio["equity_curve"].values
        
        # Basic metrics
        ann_factor = 252
        
        # Returns
        total_return = (equity[-1] / self.initial_capital) - 1
        metrics["Total Return"] = f"{total_return:.2%}"
        
        num_years = len(returns) / ann_factor
        if num_years > 0:
            ann_return = (1 + total_return) ** (1 / num_years) - 1
        else:
            ann_return = 0
        metrics["Annualized Return"] = f"{ann_return:.2%}"
        
        # Volatility
        daily_vol = np.std(returns)
        ann_vol = daily_vol * np.sqrt(ann_factor)
        metrics["Annualized Volatility"] = f"{ann_vol:.2%}"
        
        # Sharpe Ratio
        if ann_vol > 0:
            sharpe = (ann_return - risk_free_rate) / ann_vol
        else:
            sharpe = 0
        metrics["Sharpe Ratio"] = f"{sharpe:.2f}"
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns) * np.sqrt(ann_factor)
            sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        else:
            sortino = np.inf
        metrics["Sortino Ratio"] = f"{sortino:.2f}"
        
        # Maximum Drawdown
        cumulative = portfolio["cumulative_ret"].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        metrics["Maximum Drawdown"] = f"{max_dd:.2%}"
        
        # Calmar Ratio
        if max_dd != 0:
            calmar = ann_return / abs(max_dd)
        else:
            calmar = np.inf
        metrics["Calmar Ratio"] = f"{calmar:.2f}"
        
        # Win Rate
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        if len(returns) > 0:
            win_rate = len(positive_returns) / len(returns[returns != 0])
        else:
            win_rate = 0
        metrics["Win Rate (Daily)"] = f"{win_rate:.1%}"
        
        # Profit Factor
        if len(negative_returns) > 0:
            profit_factor = positive_returns.sum() / abs(negative_returns.sum())
        else:
            profit_factor = np.inf if len(positive_returns) > 0 else 0
        metrics["Profit Factor"] = f"{profit_factor:.2f}"
        
        # Trading Statistics
        avg_positions = portfolio["active_positions"].mean()
        metrics["Avg Active Positions"] = f"{avg_positions:.1f}"
        
        total_costs = portfolio["total_costs"].sum()
        metrics["Total Trading Costs"] = f"{total_costs:.2%} of capital"
        
        # Risk-adjusted metrics
        if daily_vol > 0:
            risk_adj_return = ann_return / ann_vol**2  # Return per unit variance
            metrics["Risk-Adjusted Return"] = f"{risk_adj_return:.2f}"
        
        return metrics
    
    def run_backtest(self, prices_df, all_signals):
        """
        Run comprehensive backtest with all improvements
        """

        all_pair_results = {}
        pair_names = []
        entry_counts = []
        exit_counts = []
        pair_returns = []
        pair_sharpes = []
        # Process each pair
        for pair_name, signals in all_signals.items():
            if signals is None or signals.empty:
                print(f"  Skipping {pair_name}: No signals")
                continue
            
            pair_results = self.calculate_daily_pnl(prices_df, signals, pair_name)
            
            if len(pair_results) > 0:
                all_pair_results[pair_name] = pair_results
                
                # Report pair statistics
                entries = signals[signals["entry_signal"] == True]
                exits = signals[signals["exit_signal"] == True]
                pair_names.append(pair_name)
                entry_counts.append(len(entries))
                exit_counts.append(len(exits))
                if "net_ret" in pair_results.columns:
                    pair_total_ret = (1 + pair_results["net_ret"]).prod() - 1
                    pair_sharpe = (pair_results["net_ret"].mean() * 252) / (pair_results["net_ret"].std() * np.sqrt(252))
                    pair_sharpes.append(pair_sharpe)
                    pair_returns.append(pair_total_ret)
        
        all_pair_stats = pd.DataFrame({
            "Pair": pair_names,
            "Entries": entry_counts,
            "Exits": exit_counts,
            "Total Return": pair_returns,
            "Sharpe Ratio": pair_sharpes}).sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
        
        print("\nIndividual Pair Statistics")
        print(all_pair_stats)            
        
        if not all_pair_results:
            print("\nNo valid pairs to backtest!")
            return pd.DataFrame(), {}
        
        # Combine into portfolio
        portfolio, metrics = self.calculate_portfolio_metrics(all_pair_results)
        
        return portfolio, metrics


def run_complete_backtest():
    """
    Run the complete improved backtest
    """
    from src.pair_selection import PairSelector
    from src.signal_generation import OptimizedSignalGenerator
    
    # Load data
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)

    test_periods = [
        (pd.to_datetime("2021-01-01"), pd.to_datetime("2021-12-31")),
        (pd.to_datetime("2022-01-01"), pd.to_datetime("2022-12-31")),
        (pd.to_datetime("2023-01-01"), pd.to_datetime("2023-12-31")),
        (pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31")),
        (pd.to_datetime("2025-01-01"), pd.to_datetime("2025-12-31")),
    ]
    
    pure_z_sharpe_ratios = []
    pure_z_returns = []
    pure_z_win_rates = []
    pure_z_drawdowns = []

    z_c_veto_sharpe_ratios = []
    z_c_veto_returns = []
    z_c_veto_win_rates = []
    z_c_veto_drawdowns = []

    z_c_scale_sharpe_ratios = []
    z_c_scale_returns = []
    z_c_scale_win_rates = []
    z_c_scale_drawdowns = []

    for start_date, end_date in test_periods:
        print(f"\n\n{'='*80}")
        print(f"TEST PERIOD: {start_date.date()} to {end_date.date()}")
        print(f"{'='*80}")
    
        print("\nPAIR SELECTION")
        print("="*60)
        
        selector = PairSelector(
            prices, returns,
            train_end_date=start_date - pd.DateOffset(days=1),
            test_end_date=end_date,
            top_n=4,
            coint_pvalue=0.05,
            min_half_life=5,
            max_half_life=90
        )
        
        selected_pairs = selector.run_selection()
        
        if not selected_pairs:
            print("No pairs found! Try relaxing filters.")
            return
        
        # Step 2: Signal Generation
        print("\nSIGNAL GENERATION")
        print("="*60)
        
        test_prices, test_returns = selector.get_test_data()
        
    # Test configurations
        configs = [
            {"name": "Pure Z-Score", "use_copula": False, "scale": False},
            {"name": "Z + Copula Veto", "use_copula": True, "scale": False},
            {"name": "Z + Copula Scaling", "use_copula": True, "scale": True},
        ]
        
        results = {}
        
        for config in configs:
            print(f"\n\n{'='*80}")
            print(f"TESTING: {config['name']}")
            print(f"{'='*80}")
            
            # Generate signals
            signal_gen = OptimizedSignalGenerator(
                entry_z_score=2.75,
                exit_z_score=0.4,
                use_copula_filter=config["use_copula"],
                position_scale_by_conviction=config["scale"],
                copula_veto_threshold=0.35  # Only veto if copula strongly disagrees
            )
            
            all_signals = signal_gen.generate_batch_signals(test_prices, selected_pairs)
        
            print("\nBACKTESTING")
            print("="*60)
            
            backtester = ImprovedBacktester(
                initial_capital=100_000,
                position_size_pct=0.1,
                tcost_bps=5,
                slippage_bps=3,
                max_positions=5,
                use_volatility_sizing=True
            )
            
            portfolio, metrics = backtester.run_backtest(test_prices, all_signals)
            
            results[config["name"]] = metrics
        
            # Display comparison
            print(f"\n\n{'='*80}")
            print("STRATEGY COMPARISON")
            print(f"{'='*80}")        
            # Save results
            if len(portfolio) > 0:
                portfolio.to_csv(os.path.join(BACKTEST_DIR, "improved_backtest_results.csv"))
                print(f"\nResults saved to {BACKTEST_DIR}/improved_backtest_results.csv")
            if config["name"] == "Pure Z-Score":
                pure_z_sharpe_ratios.append(metrics.get("Sharpe Ratio", "N/A"))
                pure_z_returns.append(metrics.get("Total Return", "N/A"))
                pure_z_win_rates.append(metrics.get("Win Rate (Daily)", "N/A"))
                pure_z_drawdowns.append(metrics.get("Maximum Drawdown", "N/A"))
            elif config["name"] == "Z + Copula Veto":
                z_c_veto_sharpe_ratios.append(metrics.get("Sharpe Ratio", "N/A"))
                z_c_veto_returns.append(metrics.get("Total Return", "N/A"))
                z_c_veto_win_rates.append(metrics.get("Win Rate (Daily)", "N/A"))
                z_c_veto_drawdowns.append(metrics.get("Maximum Drawdown", "N/A"))
            elif config["name"] == "Z + Copula Scaling":
                z_c_scale_sharpe_ratios.append(metrics.get("Sharpe Ratio", "N/A"))
                z_c_scale_returns.append(metrics.get("Total Return", "N/A"))
                z_c_scale_win_rates.append(metrics.get("Win Rate (Daily)", "N/A"))
                z_c_scale_drawdowns.append(metrics.get("Maximum Drawdown", "N/A"))
        
    # Dataframe of Periods and Sharpe Ratios
    df = pd.DataFrame(test_periods, columns=["Start Date", "End Date"])
    print("PZ = Pure Z-Score, ZCV = Z + Copula Veto, ZCS = Z + Copula Scaling") #, Win_R = Win Rate, Max_D = Max Drawdown")

    df["PZ Sharpe"] = pure_z_sharpe_ratios
    df["ZCV Sharpe"] = z_c_veto_sharpe_ratios if z_c_veto_sharpe_ratios else [np.nan]*len(test_periods)
    df["ZCS Sharpe"] = z_c_scale_sharpe_ratios if z_c_scale_sharpe_ratios else [np.nan]*len(test_periods)

    df["PZ Return"] = pure_z_returns
    df["ZCV Return"] = z_c_veto_returns if z_c_veto_returns else [np.nan]*len(test_periods)
    df["ZCS Return"] = z_c_scale_returns if z_c_scale_returns else [np.nan]*len(test_periods)

    df["PZ Win_R"] = pure_z_win_rates
    df["ZCV Win_R"] = z_c_veto_win_rates
    df["ZCS Win_R"] = z_c_scale_win_rates

    df["PZ Max_D"] = pure_z_drawdowns
    df["ZCV Max_D"] = z_c_veto_drawdowns
    df["ZCS Max_D"] = z_c_scale_drawdowns 
    
    print("\n\nSharpe Ratios by Test Period:")
    print(df)
    return results, portfolio, metrics
if __name__ == "__main__":
    results, portfolio, metrics = run_complete_backtest()