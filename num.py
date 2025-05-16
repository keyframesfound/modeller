import numpy as np
import matplotlib.pyplot as plt

# Data points (months since first data point, cumulative cases)
t_data = np.array([0, 1, 2, 3, 4, 5])
N_data = np.array([557, 27644, 98078, 1254222, 3680614, 6789073])

# Logistic model parameters found earlier
K = 700_000_000
r = 1.84
N0 = N_data[0]
B = (K - N0) / N0

# Logistic function
def logistic(t):
    return K / (1 + B * np.exp(-r * t))

# Generate smooth curve
t_fit = np.linspace(0, 5, 100)
N_fit = logistic(t_fit)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(t_data, N_data, color='red', label='Actual Data')
plt.plot(t_fit, N_fit, color='blue', label='Logistic Model Fit')
plt.title('COVID-19 Cases vs. Logistic Growth Model')
plt.xlabel('Months since first data point')
plt.ylabel('Cumulative Cases')
plt.yscale('log')  # Use log scale for better visibility, can remove if linear preferred
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()