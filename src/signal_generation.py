import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

def estimate_hedge_ratio(log_returns: pd.DataFrame, etf1: str, etf2: str) -> float:
    """
    Estimate the hedge ratio between two ETFs using linear regression.
    """
    y = log_returns[etf1].dropna()
    X = log_returns[etf2].dropna()
    common_index = y.index.intersection(X.index)
    y, X = y.loc[common_index], X.loc[common_index]
    X = add_constant(X)
    model = OLS(y, X).fit()
    print(f"Model Summary for {etf1} ~ {etf2}:\n{model.summary()}")
    beta = model.params[etf2]
    return float(beta)

def compute_spread(log_returns: pd.DataFrame, etf1: str, etf2: str, hedge_ratio: float) -> pd.Series:
    if hedge_ratio is None:
        hedge_ratio = estimate_hedge_ratio(log_returns, etf1, etf2)
    
    idx = log_returns[etf1].index.union(log_returns[etf2].index).sort_values()
    y_a = log_returns[etf1].reindex(idx).fillna(method='ffill').fillna(method='bfill')
    x_a = log_returns[etf2].reindex(idx).fillna(method='ffill').fillna(method='bfill')
    spread = y_a - hedge_ratio * x_a
    return spread

def test_stationarity(series: pd.Series, significance_level: float = 0.05) -> Tuple[bool, Dict]:
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
    return is_stationary, adf_stats

def half_life_ou(spread: pd.Series) -> float:
    """
    Estimating the half life of mean regression for the spread using Ornstein-Uhlenbeck process.
    """
    spread_lag = spread.shift(1).dropna()
    spread_ret = spread.diff().dropna()
    common_index = spread_lag.index.intersection(spread_ret.index)
    spread_lag, spread_ret = spread_lag.loc[common_index], spread_ret.loc[common_index]
    X = add_constant(spread_lag)
    model = OLS(spread_ret, X).fit()
    theta = -model.params[1]
    if theta <= 0:
        return np.inf
    half_life = np.log(2) / theta
    return half_life

def z_score(spread: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    mu = spread.rolling(window=window, min_periods=int(window*0.6)).mean()
    sigma = spread.rolling(window=window, min_periods=int(window*0.6)).std(ddof=0)
    z_score = (spread - mu) / sigma
    return z_score

def generate_signals(spread: pd.Series,
                     entry_z: float=2.0,
                     exit_z: float=0.5,
                     window: int=60,
                     max_holding_days: int=252) -> pd.DataFrame:
    signals = spread.copy()
    z_scores = z_score(signals, window=window)
    df = pd.DataFrame({"spread": spread, "z_score": z_scores})
    df['position'] = 0
    df["entry"] = False
    df["exit"] = False
    df["entry_price_spread"] = np.nan
    df["entry_index"] = pd.NaT

    position = 0
    entry_idx = None
    entry_i = None

    for i in range(len(df)):
        date = df.index[i]
        zt = df["z"].iat[i]
        if position == 0:
            if zt >= entry_z:
                position = -1
                df.at[date, "entry"] = True
                df.at[date, "entry_price_spread"] = df.at[date, "spread"]
                df.at[date, "entry_index"] = date
                entry_idx = i
                entry_i = date
            elif zt <= -entry_z:
                position = 1
                df.at[date, "entry"] = True
                df.at[date, "entry_price_spread"] = df.at[date, "spread"]
                df.at[date, "entry_index"] = date
                entry_idx = i
                entry_i = date
        else:
            # check exit on mean reversion
            if abs(zt) <= exit_z:
                df.at[date, "exit"] = True
                df.at[date, "entry_index"] = df.at[df.index[entry_idx], "entry_index"]
                position = 0
                entry_idx = None
                entry_i = None
            
            else:
                # check exit on max holding period
                if abs(zt) <= exit_z:
                    df.at[date, "exit"] = True
                    df.at[date, "entry_index"] = df.at[df.index[entry_idx], "entry_index"]
                    position = 0
                    entry_idx = None
                    entry_i = None
                else:
                    if (i-entry_idx) >= max_holding_days:
                        df.at[date, "exit"] = True
                        df.at[date, "entry_index"] = df.at[df.index[entry_idx], "entry_index"]
                        position = 0
                        entry_idx = None
                        entry_i = None
        df.at[date, "position"] = position

    df["position"] = df["position"].ffill().fillna(0).astype(int)
    return df

def compute_pnl(df_signals: pd.DataFrame, log_returns, etf1: str, etf2: str, 
                hedge_ratio:float, 
                notional: float=1_000_000,
                tc_bps: float=0.0) -> pd.DataFrame:
    """
    Compute daily PnL for the symmetric pair trade given positions on the spread.
    - notional: dollar exposure per leg sizing (adopted volatility scaling below)
    - tc_bps: round-trip transaction cost in basis points (applied on trade open and close)
    Assumes position column: +1 long spread (long y, short x*beta), -1 short spread (short y, long x*beta)
    """
    idx = df_signals.index
    y = log_returns[etf1].reindex(idx)
    x = log_returns[etf2].reindex(idx)

    # position series
    pos = df_signals["position"].reindex(idx).ffill().fillna(0)

    # PnL per period for spread position
    # pnl = pos * (notional * y - notional * hedge_ratio * x)
    # Apply tcost when position changes
    trades = pos.diff().fillna(0).abs() > 0
    tc = trades.astype(float) * (tc_bps/10000.0) * notional * 2
    pnl = pos * (notional * y - notional * hedge_ratio * x) - tc
    cum_pnl = pnl.cumsum()
    summary = {
        "daily_pnl": pnl,
        "cumulative_pnl": cum_pnl,
        "total_return": cum_pnl.iloc[-1] / (notional * 2) if not cum_pnl.empty else 0.0,
        "num_trades": trades.sum(),
        "total_tc": tc.sum()        
    }

    return pd.DataFrame(summary)

