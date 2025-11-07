import numpy as np
import pandas as pd
from copulae.archimedean import ClaytonCopula, GumbelCopula, FrankCopula
from copulae.elliptical import StudentCopula, GaussianCopula
from scipy.stats import t as student_t, norm
import warnings
warnings.filterwarnings("ignore")

def calculate_aic(log_likelihood, num_params):
    return 2 * num_params - 2 * log_likelihood

class CopulaModel:

    def __init__(self, t_dist_df_min=5, initial_guess_rho=0.5):
        self.t_dist_df_min = t_dist_df_min
        self.initial_guess_rho = initial_guess_rho
        self.model_data = None

    def fit_copula(self, u, v, copula_candidates=None):
        if copula_candidates is None:
            copula_candidates = ["gaussian", "student", "clayton", "gumbel", "frank"]
        
        results = []
        data = np.vstack([u, v]).T
        for name in copula_candidates:
            try:
                model = None
                log_l = -np.inf
                k = 0
                if name == "gaussian":
                    model = GaussianCopula()
                    model.fit(data)
                    k = 1
                elif name == "student":
                    model = StudentCopula()
                    model.fit(data)
                    k = 2
                elif name in ["clayton", "gumbel", "frank"]:
                    k = 1
                    if name == "clayton":
                        model = ClaytonCopula()
                    elif name == "gumbel":
                        model = GumbelCopula()
                    elif name == "frank":
                        model = FrankCopula()
                    model.fit(data)
                
                log_l = model.log_lik(data)

                if log_l > -np.inf and k>0 and model is not None:
                    aic = calculate_aic(log_l, k)
                    results.append({
                        "copula_type": name,
                        "copula": model,
                        "log_l": log_l,
                        "AIC": aic,
                        "num_params": k
                    })
            except Exception as e:
                continue
        if not results:
            return {"copula_type": None, "copula": None, "log_l": None, "AIC": None, "num_params": None}
        best_model = min(results, key=lambda x: x["AIC"])
        self.model_data = best_model
        return best_model
    
    def get_log_likelihood(self):
        if self.model_data:
            return self.model_data["log_l"]
        return None
    def get_bic(self, n):
        if self.model_data:
            k = self.model_data["num_params"]
            log_l = self.model_data["log_l"]
            bic = np.log(n) * k - 2 * log_l
            return bic
        return None