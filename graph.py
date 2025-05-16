import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Data
t = np.array([...])  # 0 to 39 (months)
N = np.array([...])  # Cumulative cases

K_candidates = [700_000_000, 750_000_000, 800_000_000, 850_000_000, 900_000_000, 950_000_000, 1_000_000_000, 1_500_000_000, 2_000_000_000]

best_R2 = -np.inf
best_K = None
best_lambda = None
best_intercept = None

for K in K_candidates:
    mask = N < K * 0.9
    if not np.any(mask):
        continue
    y = np.log(N[mask] / (K - N[mask]))
    t_masked = t[mask]
    slope, intercept, r_value, _, _ = stats.linregress(t_masked, y)
    R2 = r_value ** 2
    if R2 > best_R2:
        best_R2 = R2
        best_K = K
        best_lambda = slope
        best_intercept = intercept

print(f"Best K: {best_K}")
print(f"Best λ (growth rate): {best_lambda}")
print(f"Best intercept: {best_intercept}")
print(f"Best R²: {best_R2}")

# Logistic model
def logistic(t, K, N0, lambda_):
    return K / (1 + ((K - N0) / N0) * np.exp(-lambda_ * t))

t_fit = np.linspace(0, 39, 200)
N0 = N[0]
N_t = logistic(t_fit, best_K, N0, best_lambda)

plt.figure(figsize=(10,6))
plt.plot(t_fit, N_t, label="Logistic Growth Model")
plt.scatter(t, N, color='red', label="Actual Data", zorder=5)
plt.xlabel("Time (Months)")
plt.ylabel("Cumulative Cases")
plt.title("Logistic Growth Curve for COVID-19")
plt.xlim(0,39)
plt.legend()
plt.grid()
plt.show()