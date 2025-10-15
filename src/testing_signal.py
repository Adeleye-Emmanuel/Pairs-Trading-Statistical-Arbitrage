import pandas as pd
import matplotlib.pyplot as plt
import os
from config import PROCESSED_DIR

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.signal_generation import *


prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)

print(prices.head())

print("Analyzing AGG and LQD")
result = analyze_pair(prices, etf1="AGG", etf2="LQD", max_holding_days=252, entry_z=1.5, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
print("Analyzing SPY and QQQ")
result = analyze_pair(prices, etf1="SPY", etf2="QQQ", max_holding_days=252, entry_z=1.5, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
print("Analyzing EFA and EWJ")
result = analyze_pair(prices, etf1="EFA", etf2="EWJ", max_holding_days=252, entry_z=1.5, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
print("Analysing EWJ and VEU")
result = analyze_pair(prices, etf1="EWJ", etf2="VEU", max_holding_days=252, entry_z=1.5, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
