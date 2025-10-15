import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

def estimate_hedge_ratio(data: pd.DataFrame, etf1: str, etf2: str, mode: str = "prices") -> float:
    """
    Estimate the hedge ratio between two ETFs using linear regression.
    """
    print(f"=====================Estimating Hedge Ratio for {etf1} and {etf2}=====================")
    if mode == "prices":
        y = np.log(data[etf1].dropna())
        X = np.log(data[etf2].dropna())
    else:  # default to returns
        y = data[etf1].dropna()
        X = data[etf2].dropna()

    common_index = y.index.intersection(X.index)
    y, X = y.loc[common_index], X.loc[common_index]
    X = add_constant(X)
    model = OLS(y, X).fit()
    print(f"Model Summary for {etf1} ~ {etf2}:\n{model.summary()}")
    beta = model.params.get(etf2, np.nan)
    return float(beta)

def compute_spread(log_returns: pd.DataFrame, etf1: str, etf2: str, hedge_ratio: float) -> pd.Series:
    print(f"=====================Computing Spread for {etf1} and {etf2}=====================")
    if hedge_ratio is None:
        hedge_ratio = estimate_hedge_ratio(log_returns, etf1, etf2)
    
    idx = log_returns[etf1].index.union(log_returns[etf2].index).sort_values()
    y_a = log_returns[etf1].reindex(idx)
    x_a = log_returns[etf2].reindex(idx)
    aligned_data = pd.DataFrame({etf1: y_a, etf2: x_a}).dropna()
    spread = aligned_data[etf1] - hedge_ratio * aligned_data[etf2]
    print(f"Spread NaNs: {spread.isnull().sum()}")

    return spread

def test_stationarity(series: pd.Series, significance_level: float = 0.05) -> Tuple[bool, Dict]:
    print("=====================Performing ADF Test=====================")
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < significance_level
    adf_stats = {
        'ADF Statistic': result[0],
        'p-value': p_value,
        'Used Lag': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4]
    }
    print("Stationarity Tests Completed.")
    return is_stationary, adf_stats

def half_life_ou(spread: pd.Series) -> float:
    """
    Estimating the half life of mean regression for the spread using Ornstein-Uhlenbeck process.
    """
    print("=====================Computing Half-Life of Mean Reversion=====================")
    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()
    common_index = spread_lag.index.intersection(spread_ret.index)
    X = add_constant(spread_lag.loc[common_index])
    model = OLS(spread_ret.loc[common_index], X).fit()
    theta = -model.params.iloc[1]
    if theta <= 0:
        half_life = np.inf
        print("Warning: Non-positive theta estimated, half-life set to infinity.")
    else:
        half_life = np.log(2) / theta
        print(f"Half-Life: {half_life} days")
    
    print("Half-Life Computed.")

    return float(half_life)

def z_score(spread: pd.Series, window: int):
    print("=====================Computing Z-Score=====================")
    mu = spread.rolling(window=window, min_periods=int(window*0.6)).mean()
    sigma = spread.rolling(window=window, min_periods=int(window*0.6)).std(ddof=0)

    z_score = (spread - mu) / sigma
    
    print("Z-Score Computed.")
    print(f"Z-Score Nans: {z_score.isnull().sum()}")
    print("==============================================")

    return z_score

def generate_signals(spread: pd.Series,
                     entry_z: float=2.0,
                     exit_z: float=0.5,
                     window: int=60,
                     max_holding_days: int=252) -> pd.DataFrame:
    
    print("Generating Trading Signals...")
    z_scores = z_score(spread, window=window)
    df = pd.DataFrame({"spread": spread, "z_score": z_scores}).dropna()

    if df.empty:
        print("ERROR: DataFrame is empty. No signals generated after dropping NaN.")
        return pd.DataFrame()
    
    df['position'] = 0
    df["entry"] = False
    df["exit"] = False
    df["entry_price_spread"] = np.nan
    df["entry_date"] = pd.NaT

    position = 0
    entry_idx = None
    entry_date = None

    for i in range(len(df)):
        date = df.index[i]
        zt = df["z_score"].iat[i]
        if position == 0:
            if zt >= entry_z:
                position = -1
                df.at[date, "entry"] = True
                df.at[date, "entry_price_spread"] = df.at[date, "spread"]
                df.at[date, "entry_date"] = date
                entry_idx = i
                entry_date = date
            elif zt <= -entry_z:
                position = 1
                df.at[date, "entry"] = True
                df.at[date, "entry_price_spread"] = df.at[date, "spread"]
                df.at[date, "entry_date"] = date
                entry_idx = i
                entry_date = date

        else:
            # check exit on mean reversion or max holding period
            exit_on_mean_reversion  = abs(zt) <= exit_z
            exit_on_max_holding = (i - entry_idx) >= max_holding_days
            if exit_on_mean_reversion or exit_on_max_holding:
                df.at[date, "exit"] = True
                df.at[date, "entry_date"] = df.at[df.index[entry_idx], "entry_date"]
                position = 0
                entry_idx = None
                entry_date = None

        df.at[date, "position"] = position

    print("Signal Generation Completed.")
    print(df["entry"].value_counts())
    return df

def compute_pnl(df_signals: pd.DataFrame, 
                prices_df: pd.DataFrame, 
                etf1: str, etf2: str, 
                hedge_ratio:float, 
                notional: float=1_000_000,
                tc_bps: float=0.0) -> pd.DataFrame:
    """
    Compute daily PnL for the symmetric pair trade given positions on the spread.
    - notional: dollar exposure per leg sizing (adopted volatility scaling below)
    - tc_bps: round-trip transaction cost in basis points (applied on trade open and close)
    Assumes position column: +1 long spread (long y, short x*beta), -1 short spread (short y, long x*beta)
    """
    print("=====================Computing PnL=====================")
    idx = df_signals.index
    y = prices_df[etf1].reindex(idx).fillna(0)
    x = prices_df[etf2].reindex(idx).fillna(0)

    # Using simple daily returns for PnL calculation
    ret_y = y.pct_change().fillna(0)
    ret_x = x.pct_change().fillna(0)

    # position series
    pos = df_signals["position"].reindex(idx).ffill().fillna(0)
    pos_shift= pos.shift(1).fillna(0) # Avoids lookahead bias

    turnover = pos_shift.diff().abs().fillna(0)
    tc = turnover * (tc_bps/10000.0) * notional * 2  # round-trip cost

    daily_spread_return = ret_y - hedge_ratio * ret_x
    pnl = pos_shift * daily_spread_return * notional * 2 - tc
    cum_pnl = pnl.cumsum()

    summary = {
        "daily_pnl": pnl,
        "cumulative_pnl": cum_pnl,
        "total_return": cum_pnl.iloc[-1] / (notional * 2) if not cum_pnl.empty else 0.0,
        "num_trades": turnover.sum()/2, # Divide by 2 because a full cycle (entry/exit) is 2 turnover points
        "total_tc": tc.sum()         
    }

    print("PnL Computation Completed.")
    return pd.DataFrame(summary)


# Utility: quick run for single pair
def analyze_pair(prices, etf1: str, etf2: str, hedge_ratio: float = None,
                 window: int = 60, entry_z: float = 2.0, exit_z: float = 0.5,
                 notional: float = 1_000_000, tc_bps: float = 2.0, max_holding_days: int = 252,
                 beta_mode: str = "prices") -> Dict:
    """
    Convenience wrapper that computes hedge ratio, spread, diagnostics, signals and pnl summary.
    Returns dict with diagnostics and dataframes (spread, signals, pnl series).
    """
    # estimate hedge ratio if not provided

    beta = hedge_ratio if hedge_ratio is not None else estimate_hedge_ratio(prices, etf1, etf2, mode=beta_mode)

    log_returns = np.log(prices).diff().dropna()
    spread = compute_spread(log_returns, etf1, etf2, hedge_ratio=beta)
    adf = test_stationarity(spread)
    half_life = half_life_ou(spread)
    signals = generate_signals(spread, entry_z=entry_z, exit_z=exit_z, window=window, max_holding_days=max_holding_days)
    pnl = compute_pnl(signals, prices, etf1, etf2, beta, notional=notional, tc_bps=tc_bps)
    
    return {
        "beta": beta,
        "spread": spread,
        "adf": adf,
        "half_life": half_life,
        "signals": signals,
        "pnl": pnl
    }