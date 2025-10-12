import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.copula_utils import *
import pandas as pd

data = pd.read_csv('data/processed/log_returns.csv', index_col=0, parse_dates=True)
score = copula_dependency_score(data["SPY"].values, data["QQQ"].values)
print(f'Copula Score between SPY and QQQ: {score}')