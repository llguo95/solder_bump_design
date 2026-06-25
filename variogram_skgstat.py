import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg

# 1. Generate the same dummy data
np.random.seed(42)
n_points = 100
x = np.random.uniform(0, 100, n_points)
y = np.random.uniform(0, 100, n_points)
z = 0.05 * x + 0.02 * y + np.random.normal(0, 2, n_points)
coords = np.column_stack((x, y))

# 2. Create the Variogram object
# scikit-gstat handles the binning, distance calculations, and plotting automatically
V = skg.Variogram(
    coordinates=coords,
    values=z,
    normalize=False,  # Keep raw distances/semivariance
    n_lags=15,
)  # Number of bins to use

# 3. Plot the empirical variogram
V.plot()
plt.title("Empirical Variogram via scikit-gstat")
plt.show()

# You can easily inspect the parameters it calculated
print(f"Number of pairs per lag: {V.np}")
