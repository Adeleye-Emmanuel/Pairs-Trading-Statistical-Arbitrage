import os
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import coint
from itertools import combinations

from config import ETFS_DIR, PROCESSED_DIR

def load_data():
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "cleaned_prices.csv"), index_col=0, parse_dates=True)
    log_returns = pd.read_csv(os.path.join(PROCESSED_DIR, "log_returns.csv"), index_col==0, parse_dates=True)
    return prices, log_returns

def compute_correlations(log_returns):
    return log_returns.corr()

