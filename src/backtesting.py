import numpy as np
import pandas as pd
from datetime import datetime
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PROCESSED_DIR, BACKTEST_DIR

class Backtester:

    def __init__(self, signals_dict, prices_df, selected_pairs,
                 initial_capital=100_000,
                 tcost_bps=5,
                 risk_per_trade = 0.02,
                 max_positions=5,
                 lookback_volatility=60):
        """
        Backtest pairs trading strategy with risk-based position sizing
        
        Args:
            signals_dict: Dictionary of {pair_name: signals_df}
            prices_df: Price data for all assets
            selected_pairs: List of pair metadata (includes hedge ratios)
            initial_capital: Starting portfolio value
            transaction_cost_bps: Transaction cost in basis points per side
            risk_per_trade: Target risk per trade as fraction of portfolio (e.g., 0.02 = 2%)
            max_positions: Maximum number of pairs open simultaneously
            lookback_volatility: Days to lookback for spread volatility calculation
        """

        self.signals_dict = signals_dict
        self.prices_df = prices_df
        self.selected_pairs = {f"{p["ETF1"]}_{p["ETF2"]}": p for p in selected_pairs}
        self.initial_capital = initial_capital
        self.tcost_bps = tcost_bps
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.lookback_volatility = lookback_volatility

        self.trades = []
        self.equity_curve = None

    def calculate_spread_series(self, pair_name, pair_data, prices):

        etf1 = pair_data["ETF1"]
        etf2 = pair_data["ETF2"]
        hedge_ratio = pair_data.get("Hedge_ratio", 1)

        spread = prices[etf1] - hedge_ratio * prices[etf2]
        return spread
    
    def calculate_position_size(self, pair_name, pair_data, current_date, portfolio_value):
        """
        Calculate position size based on spread volatility (risk-based sizing)
        
        Returns:
            shares_etf1: Number of shares to trade in ETF1
            shares_etf2: Number of shares to trade in ETF2
            notional_value: Total dollar value of position
        """
        etf1 = pair_data["ETF1"]
        etf2 = pair_data["ETF2"]
        hedge_raio = pair_data.get("Hedge_ratio", 1)

        # Getting historical prices for volatility calculation
        lookback_start = max(0, self.prices_df.index.get_loc(current_date) - self.lookback_volatility)
        historical_prices = self.prices_df.iloc[lookback_start:self.prices_df.index.get_loc(current_date)]

        if len(historical_prices) < 20:
            return None, None, None
        
        # Calculate historical spread and its volatility
        spread = self.calculate_spread_series(pair_name, pair_data, historical_prices)
        spread_vol = spread.std()

        if not np.isfinite(spread_vol) or spread_vol <=0:
            return None, None, None
        
        # Risk-based position sizing
        # Position size such that 1 std move in spread = risk_per_trade of portfolio
        risk_capital = portfolio_value * self.risk_per_trade
        position_size_dollars = risk_capital / spread_vol

        prices_etf1 = self.prices_df.loc[current_date, etf1]
        prices_etf2 = self.prices_df.loc[current_date, etf2]

        shares_etf1 = position_size_dollars / prices_etf1
        shares_etf2 = shares_etf1 * hedge_raio

        notional_value = abs(shares_etf1 * prices_etf1) + abs(shares_etf2*prices_etf2)

        return shares_etf1, shares_etf2, notional_value
    
    def calculate_tcost(self, notional_value):
        return notional_value * (self.tcost_bps/10000)
    
    def backtest_pair(self, pair_name, signals, pair_data):

        pair_trades = []

        etf1 = pair_data["ETF1"]
        etf2 = pair_data["ETF2"]

        entries = signals[signals["entry_signal"]].copy()
        exits = signals[signals["exit_signal"]].copy()

        if len(entries) == 0:
            return pair_trades
        
        for _, entry_row in entries.iterrows():
            entry_date = entry_row.name
            position = entry_row["position"]

            exit_candidates = exits[exits.index > entry_date]
            if len(exit_candidates) == 0:
                continue # Trade never closed
            
            exit_row = exit_candidates.iloc[0]
            exit_date = exit_row.name

            # Get prices at entry and exit
            entry_price_etf1 = self.prices_df.loc[entry_date, etf1]
            entry_price_etf2 = self.prices_df.loc[entry_date, etf2]
            exit_price_etf1 = self.prices_df.loc[exit_date, etf1]
            exit_price_etf2 = self.prices_df.loc[exit_date, etf2]

            if not all(np.isfinite([entry_price_etf1, entry_price_etf2, exit_price_etf1, exit_price_etf2])):
                continue

            # Calculate position size (using entry date portfolio value)
            # For now, use initial capital as proxy (will be updated in portfolio backtest)
            shares_etf1, shares_etf2, notional = self.calculate_position_size(
                pair_name, pair_data, entry_date, self.initial_capital
            )

            if shares_etf1 is None:
                continue

            # Calculate P&L
            if position == 1: # Long spread
                # Long ETF1, Short ETF2
                pnl_etf1 = shares_etf1 * (exit_price_etf1 - entry_price_etf1)
                pnl_etf2 = -shares_etf2 * (exit_price_etf2 - entry_price_etf2)
            else: # Short spread (position == -1)
                # Short ETF1, Long ETF2
                pnl_etf1 = -shares_etf1 * (exit_price_etf1 - entry_price_etf1)
                pnl_etf2 = shares_etf2 * (exit_price_etf2 - entry_price_etf1)

            gross_pnl = pnl_etf1 + pnl_etf2

            # Transaction costs (entry + exit, both sides)
            entry_costs = self.calculate_tcost(notional)
            exit_costs = self.calculate_tcost(notional)
            total_costs = entry_costs + exit_costs

            net_pnl = gross_pnl - total_costs

            # Calculate returns
            net_return = net_pnl / notional if notional > 0 else 0

            trade = {
                "pair":pair_name,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "position": "Long" if position ==1 else "Short",
                "holding_days": (exit_date-entry_date).days,
                "shares_etf1": shares_etf1,
                "shares_etf2": shares_etf2,
                "notional": notional,
                "entry_spread": entry_row["spread"],
                "exit_spread": exit_row["spread"],
                "gross_pnl": gross_pnl,
                "transaction_costs": total_costs,
                "net_pnl": net_pnl,
                "return_pct": net_return * 100
            }

            pair_trades.append(trade)
        
        return pair_trades
    
    def backtest_portfolio(self):
        """
        Backtest entire portfolio with position limit constraints
        """
        print("\n" + "="*60)
        print("BACKTESTING PORTFOLIO")
        print("="*60)
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Risk per Trade: {self.risk_per_trade*100:.1f}%")
        print(f"Max Positions: {self.max_positions}")
        print(f"Transaction Costs: {self.tcost_bps} bps per side")

        all_trades = []

        # Backtest each pair independently (ignoring position limits for now)
        for pair_name, signals in self.signals_dict.items():
            if signals is None or len(signals) == 0:
                continue

            pair_data = self.selected_pairs.get(pair_name)
            if not pair_data:
                continue

            pair_trades = self.backtest_pair(pair_name, signals, pair_data)
            all_trades.extend(pair_trades)

        if len(all_trades) == 0:
            print("\nNo trades executed!")
            return pd.DataFrame(), pd.DataFrame()
        
        trades_df = pd.DataFrame(all_trades)
        trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)

        trades_df["allowed"] = False
        open_positions = {}

        for idx, trade in trades_df.iterrows():
            entry_date = trade["entry_date"]
            exit_date = trade["exit_date"]
            pair = trade["pair"]

            current_open = len(open_positions)

            if current_open < self.max_positions:
                trades_df.at[idx, "allowed"] = True
                open_positions[pair] = exit_date
            
            # Close positions that have exited
            closed_pairs = [p for p, exit_d in open_positions.items() if exit_d <=entry_date]
            for p in closed_pairs:
                del open_positions[p]

        # Filter to allowed trades only
        executed_trades = trades_df[trades_df["allowed"]].copy()

        print(f"\nTotal Trades Generated: {len(trades_df)}")
        print(f"Trades Executed (after position limit): {len(executed_trades)}")
        print(f"Trades Rejected: {len(trades_df) - len(executed_trades)}")

        # Calculate equity curve
        equity_curve = self.calculate_equity_curve(executed_trades)

        self.trades = executed_trades
        self.equity_curve = equity_curve

        return executed_trades, equity_curve
    
    def calculate_equity_curve(self, trades_df):
        if len(trades_df) == 0:
            return pd.DataFrame()
        
        all_dates = pd.date_range(
            start=trades_df["entry_date"].min(),
            end=trades_df["exit_date"].max(),
            freq="D"
        )

        equity = pd.DataFrame(index=all_dates, columns=["equity", "returns"])
        equity["equity"] = self.initial_capital
        equity["returns"] = 0.0

        # Track open positions and their P&L
        for date in all_dates:
            # Realize P&L from trades that closed today
            closed_today = trades_df[trades_df["exit_date"] == date]
            daily_pnl = closed_today["net_pnl"].sum()

            # Update equity
            if date == all_dates[0]:
                equity.loc[date, "equity"] = self.initial_capital
            else:
                prev_equity = equity.loc[:date].iloc[-2]["equity"]
                equity.loc[date, "equity"] = prev_equity + daily_pnl
                equity.loc[date, "returns"] = daily_pnl / prev_equity if prev_equity > 0 else 0
        
        equity["equity"] = equity["equity"].ffill()

        return equity
    
    def calculate_metrics(self):
        if self.trades is None or len(self.trades) == 0:
            print("No trades to analyse")
            return {}
        
        trades = self.trades
        equity = self.equity_curve

        total_trades = len(trades)
        winning_trades = len(trades[trades["net_pnl"] > 0])
        losing_trades = len(trades[trades["net_pnl"] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = trades[trades["net_pnl"]>0]["net_pnl"].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades["net_pnl"]<0]["net_pnl"].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win/avg_loss) if avg_loss !=0 else np.inf

        total_pnl = trades["net_pnl"].sum()
        total_return = (equity["equity"].iloc[-1] - self.initial_capital) / self.initial_capital

        # Time-based metrics
        returns = equity["returns"].dropna()

        if len(returns) > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std>0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Maximum drawdown
        cumulative = (1+ equity["returns"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max)/running_max
        max_drawdown = drawdown.min()

        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown !=0 else 0

        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar_ratio,
            'avg_holding_days': trades['holding_days'].mean(),
            'total_transaction_costs': trades['transaction_costs'].sum()
        }
         
        return metrics
    
    def print_summary(self):
        """Print backtest summary"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print("\nTrade Statistics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Winning Trades: {metrics['winning_trades']} ({metrics['win_rate']*100:.1f}%)")
        print(f"  Losing Trades: {metrics['losing_trades']}")
        print(f"  Average Win: ${metrics['avg_win']:,.2f}")
        print(f"  Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Holding Period: {metrics['avg_holding_days']:.1f} days")
        
        print("\nPortfolio Performance:")
        print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Transaction Costs: ${metrics['total_transaction_costs']:,.2f}")
        
        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        
        # Per-pair breakdown
        print("\n" + "="*60)
        print("PER-PAIR BREAKDOWN")
        print("="*60)
        
        pair_stats = self.trades.groupby('pair').agg({
            'net_pnl': ['count', 'sum', 'mean'],
            'return_pct': 'mean',
            'holding_days': 'mean'
        }).round(2)
        
        pair_stats.columns = ['Trades', 'Total P&L', 'Avg P&L', 'Avg Return %', 'Avg Days']
        print(pair_stats.to_string())

if __name__ == "__main__":
    from src.pair_selection import PairSelector
    from src.signal_generation import SignalGenerator

    # Load data
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col=0, parse_dates=True)
    
    # Run pair selection with train/test split
    selector = PairSelector(prices, returns)
    selected_pairs = selector.run_selection(max_pairs=10)
    
    # Get test data
    test_prices, test_returns = selector.get_test_data()
    
    # Generate signals on test data
    signal_gen = SignalGenerator()
    signals = signal_gen.generate_batch_signals(test_prices, selected_pairs)

    backtester = Backtester(
        signals_dict=signals,
        prices_df=test_prices,
        selected_pairs=selected_pairs,
        initial_capital=100_000,
        tcost_bps=5,
        risk_per_trade=0.05,
        max_positions=5,
        lookback_volatility=90
    )

    trades_df, equity_curve = backtester.backtest_portfolio()
    backtester.print_summary()

    # Saving results
    if len(trades_df) > 0:
        trades_df.to_csv(os.path.join(BACKTEST_DIR, "backtest_trades[2015-2024 on 2025].csv"), index=False)
        equity_curve.to_csv(os.path.join(BACKTEST_DIR, "backtest_equity_curve [2015-2024 on 2025].csv"), index=False)
        print(f"\nResults saved to {BACKTEST_DIR}")