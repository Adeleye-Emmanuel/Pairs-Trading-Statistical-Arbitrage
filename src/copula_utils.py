import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Clayton
from scipy.stats import kendalltau

def fit_copula(x: np.ndarray, y: np.ndarray):
    

    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    if len(x) < 30:
        return None  # Not enough data to fit a copula
    
    # Convert data to uniform margins using rank transformation
    u = (pd.Series(x).rank(method='average') - 0.5)/len(x)
    v = (pd.Series(y).rank(method='average') - 0.5)/len(y)
    uv = np.column_stack((u, v))   
    
    results = {}

    try:
        # Fit Gaussian copula
        #g_cop = GaussianMultivariate()
        #g_cop.fit(pd.DataFrame(uv, columns=['u', 'v']))
        #log_likelihood_g = np.mean(g_cop.log_likelihood(pd.DataFrame(uv, columns=['u', 'v'])))
        #results["gaussian"] = {"loglik":log_likelihood_g}

        # Fit Clayton copula
        c_cop = Clayton()
        c_cop.fit(pd.DataFrame(uv, columns=['u', 'v']))
        log_likelihood_c = np.mean(c_cop.log_likelihood(pd.DataFrame(uv, columns=['u', 'v'])))
        results["clayton"] = {"loglik":log_likelihood_c}

        # Kendall's tau
        tau, _ = kendalltau(x, y)
        results["kendall_tau"] = tau
    except Exception as e:
        print(f"Error fitting copula: {e}")
        return None
    

    return results

def copula_dependency_score(x: np.ndarray, y: np.ndarray):
    copula_results = fit_copula(x, y)
    if copula_results is None:
        return None

    # Weighted composite score: higher means stronger dependence
    gaussian_weight = 0.6
    clayton_weight = 0.4
    composite_score = (
        #gaussian_weight * copula_results["gaussian"]["loglik"] +
        copula_results["clayton"]["loglik"]
    )

    # Multiply by |Kendalll's Tau| to emphasize monotonic relationships (stronger rank correlation)
    composite_score *= abs(copula_results["kendall_tau"])
    return composite_score