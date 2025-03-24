import numpy as np
from scipy import stats

# Input data
t = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Time (months)
N = np.array([557, 27644, 98078, 1254222, 3680614, 6789073, 11474058,18817443])  # Cumulative cases
K_candidates = [700_000_000, 750_000_000, 800_000_000, 850_000_000, 900_000_000, 950_000_000, 1_000_000_000]  # Candidate K values

best_R2 = -np.inf
best_K = None
best_lambda = None

for K in K_candidates:
    # Transform data: y = ln(N / (K - N))
    y = np.log(N / (K - N))
    
    # Perform linear regression
    slope, intercept, r_value, _, _ = stats.linregress(t, y)
    R2 = r_value ** 2
    
    # Check if this K gives a better R2
    if R2 > best_R2:
        best_R2 = R2
        best_K = K
        best_lambda = slope

# Output results
print(f"Best K: {best_K}")
print(f"Best λ (growth rate): {best_lambda}")
print(f"Best R²: {best_R2}")