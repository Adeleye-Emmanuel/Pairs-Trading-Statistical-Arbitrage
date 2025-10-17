import pandas as pd
import matplotlib.pyplot as plt
import os
from config import PROCESSED_DIR

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.signal_generation import *


prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)

print(prices.head())


result = analyze_pair(prices, etf1="EWJ", etf2="VEU", max_holding_days=252, entry_z=2, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
result = analyze_pair(prices, etf1="VXUS", etf2="EWJ", max_holding_days=252, entry_z=2, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
result = analyze_pair(prices, etf1="EFA", etf2="EWJ", max_holding_days=252, entry_z=2, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
result = analyze_pair(prices, etf1="XOM", etf2="XLE", max_holding_days=252, entry_z=2, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
result = analyze_pair(prices, etf1="QQQ", etf2="VTI", max_holding_days=252, entry_z=2, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
result = analyze_pair(prices, etf1="VOO", etf2="XLK", max_holding_days=252, entry_z=2, exit_z=0.5, notional=1_000_000, tc_bps=1.0)
print(result)
