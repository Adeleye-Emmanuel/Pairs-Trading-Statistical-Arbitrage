import numpy as np
import pandas as pd
from copulae.archimedean import ClaytonCopula, GumbelCopula, FrankCopula
from copulae.elliptical import StudentCopula, GaussianCopula
from scipy.stats import t as student_t, norm
import warnings
warnings.filterwarnings("ignore")


class CopulaModel:

    def __init__(self):
        self.fitted_models = {}

    def fit_marginal_distributions(self, returns):
        # Fit student's t-distribution to each marginal
        try:
            df, loc, scale = student_t.fit(returns)

            def cdf(x):
                return student_t.cdf(x, df, loc=loc, scale=scale)
            
            def ppf(u):
                return student_t.ppf(u, df, loc=loc, scale=scale)
            
            return {
                "cdf": cdf,
                "ppf": ppf,
                "params": {"df": df, "loc": loc, "scale": scale},
                "type": "student_t"
            }
        
        except Exception as e:
            # Fallback to normal (gaussian) distribution
            mu, std = norm.fit(returns)

            def cdf(x):
                return norm.cdf(x, loc=mu, scale=std)
            
            def ppf(u):
                return norm.ppf(u, loc=mu, scale=std)
            
            return {
                "cdf": cdf,
                "ppf": ppf,
                "params": {"mu": mu, "std": std},
                "type": "gaussian"
            }

    def calculate_tail_dependence(self, copula, copula_type, uv):
        try:
            if copula_type == 'student_t':
                # For Student's t-copula, tail dependence exists and is symmetric
                import math
                from scipy.stats import t as t_dist
                
                # Get parameters - they're stored as a tuple or list
                params = copula.params
                if hasattr(params, '__len__') and len(params) >= 2:
                    rho = params[0]  # correlation parameter
                    df = params[1]   # degrees of freedom
                else:
                    # Handle case where params might be a single value
                    rho = params if isinstance(params, (int, float)) else 0.5
                    df = 4.0  # default degrees of freedom
                
                # Calculate tail dependence for Student's t-copula
                if df > 0 and abs(rho) < 1:
                    tail_dep = 2 * t_dist.cdf(
                        -math.sqrt((df + 1) * (1 - rho) / (1 + rho)), 
                        df=df + 1
                    )
                    return tail_dep, tail_dep  # symmetric
                else:
                    return 0.0, 0.0
                    
            elif copula_type == 'clayton':
                # Clayton has only lower tail dependence
                theta = copula.params
                lower_tail = 2**(-1/theta) if theta > 0 else 0.0
                return lower_tail, 0.0  # lower, upper
                
            elif copula_type == 'gumbel':
                # Gumbel has only upper tail dependence  
                theta = copula.params
                upper_tail = 2 - 2**(1/theta) if theta >= 1 else 0.0
                return 0.0, upper_tail  # lower, upper
                
            else:
                return 0.0, 0.0
                
        except Exception as e:
            print(f"Error calculating tail dependence: {e}")
            return 0.0, 0.0        
    
    
    def fit_copula(self, returns_1, returns_2, pair_name):
        # Fitting complete bivariate copula model for a pair of return series
        try:
            if len(returns_1) < 30 or len(returns_2) < 30:
                raise ValueError("Insufficient data points for copula fitting.")
            
            # align lengths using index intersection
            returns_1 = pd.Series(returns_1)
            returns_2 = pd.Series(returns_2)
            common_index = returns_1.index.intersection(returns_2.index)
            returns_1_a = returns_1.loc[common_index].values
            returns_2_a = returns_2.loc[common_index].values

            marginal_1 = self.fit_marginal_distributions(returns_1_a)
            marginal_2 = self.fit_marginal_distributions(returns_2_a)

            u = marginal_1["cdf"](returns_1_a)
            v = marginal_2["cdf"](returns_2_a)

            u = np.clip(u, 1e-6, 1 - 1e-6)
            v = np.clip(v, 1e-6, 1 - 1e-6)

            uv = np.column_stack((u, v))
            
            copula_models = {}
            copula_classes = {
                "Clayton": (ClaytonCopula, "clayton"),
                "Gumbel": (GumbelCopula, "gumbel"),
                "Frank": (FrankCopula, "frank"),
                "Gaussian": (GaussianCopula, "gaussian"),
                "Student_t": (StudentCopula, "student_t")
                    }

            for name, (cls, short_name) in copula_classes.items():
                try:
                    cop = cls(dim=2)
                    cop.fit(uv)
                    loglik = cop.log_lik(uv)
                    lower_tail, upper_tail = self.calculate_tail_dependence(cop, short_name, uv)
                    copula_models[name] = {
                        "copula": cop,
                        "log_likelihood": loglik,
                        "lower_tail_dep": lower_tail,
                        "upper_tail_dep": upper_tail
                    }

                except Exception as e:
                    print(f"Fitted {pair_name}: {name} copula, LL: {loglik:.2f}, Lower tail: {lower_tail:.3f}")    

            if not copula_models:
                raise ValueError("No copula models were successfully fitted.")
            
            best_copula_name = max(copula_models.keys(), key=lambda x: copula_models[x]["log_likelihood"])
            best_copula = copula_models[best_copula_name]
            
            if not np.isfinite(best_copula["log_likelihood"]):
                raise ValueError(f"Non-finite log-likelihood; {best_copula_name} copula fit likely failed for {pair_name}.")
            
            model = {
                "marginal_1": marginal_1,
                "marginal_2": marginal_2,
                "copula": best_copula["copula"],
                "copula_type": best_copula_name,
                "log_likelihood": float(np.ravel(best_copula["log_likelihood"])[0]),
                "lower_tail_dependence": float(np.ravel(best_copula["lower_tail_dep"])),
                "upper_tail_dependence": float(np.ravel(best_copula["upper_tail_dep"])),
                "historical_returns_1": returns_1,
                "historical_returns_2": returns_2
            }

            self.fitted_models [pair_name] = model


            ll_val = float(np.ravel(best_copula["log_likelihood"])[0])
            lt_val = float(np.ravel(best_copula["lower_tail_dep"]))
            print(f"Fitted {pair_name}: {best_copula_name} copula, LL: {ll_val:.2f}, Lower tail: {lt_val:.3f}")

            
            return model
        except Exception as e:
            print(f"Error fitting copula model for {pair_name}: {e}")
            return None
        

    def get_model(self, pair_name):
        return self.fitted_models.get(pair_name)
    
    def get_all_models(self):
        return self.fitted_models