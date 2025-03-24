import numpy as np
import matplotlib.pyplot as plt

# Define the logistic function
def logistic(t, K, N0, lambda_):
    return K / (1 + ((K - N0) / N0) * np.exp(-lambda_ * t))

# Parameters
K = 700_000_000
N0 = 557
lambda_ = 0.2892016925556748

# Time range (0 to 20 days)
t = np.linspace(0, 80)

# Calculate N(t)
N_t = logistic(t, K, N0, lambda_)

# Plot the curve
plt.plot(t, N_t, label="Logistic Growth Model")
plt.xlabel("Time (Days)")
plt.ylabel("Cumulative Cases")
plt.title("Logistic Growth Curve for COVID-19")
plt.legend()
plt.grid()
plt.show()