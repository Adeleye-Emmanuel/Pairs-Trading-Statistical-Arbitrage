# test_copula.py
import numpy as np
from copula_model import CopulaModel
from copulae.archimedean import ClaytonCopula, GumbelCopula, FrankCopula
from copulae.elliptical import StudentCopula, GaussianCopula

# Generate synthetic test data that should work
np.random.seed(42)
n_samples = 1000

# Create correlated data
x = np.random.normal(0, 1, n_samples)
y = 0.7 * x + np.random.normal(0, 0.5, n_samples)

# Transform to uniform via ECDF
def to_ecdf(data):
    sorted_data = np.sort(data)
    return np.interp(data, sorted_data, np.linspace(0, 1, len(sorted_data)))

u = to_ecdf(x)
v = to_ecdf(y)

print(f"Test data - u: [{u.min():.3f}, {u.max():.3f}], v: [{v.min():.3f}, {v.max():.3f}]")

# Test copula fitting
fitter = CopulaModel()
result = fitter.fit_copula(u, v)

print("Test result:", result)