import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# 1. Generate some dummy spatial data
np.random.seed(42)
n_points = 100
x = np.random.uniform(0, 100, n_points)
y = np.random.uniform(0, 100, n_points)
# Create a spatially correlated Z variable (simple trend + noise)
z = 0.05 * x + 0.02 * y + np.random.normal(0, 2, n_points) 

coords = np.column_stack((x, y))

# 2. Calculate all pairwise distances and squared differences
# pdist calculates the distance between every pair of points
distances = pdist(coords) 
# Calculate the squared difference in Z values for every pair
z_diffs = pdist(z.reshape(-1, 1))**2 

# 3. Define bins for the lag distances
max_distance = np.max(distances)
lag_spacing = 10.0 # Width of each bin
bins = np.arange(0, max_distance + lag_spacing, lag_spacing)

# 4. Calculate empirical semivariance for each bin
bin_centers = (bins[:-1] + bins[1:]) / 2
semivariance = np.zeros(len(bin_centers))

for i in range(len(bin_centers)):
    # Find indices of pairs that fall into the current distance bin
    mask = (distances >= bins[i]) & (distances < bins[i+1])
    
    if np.sum(mask) > 0:
        # Apply the variogram formula: 1/(2N) * sum(diffs^2)
        semivariance[i] = np.mean(z_diffs[mask]) / 2.0
    else:
        semivariance[i] = np.nan # No pairs in this bin

# 5. Plot the empirical variogram
plt.figure(figsize=(8, 5))
plt.scatter(bin_centers, semivariance, color='blue', label='Empirical Semivariance')
plt.xlabel('Lag Distance (h)')
plt.ylabel(r"Semivariance $\gamma$(h)")
plt.title('Empirical Variogram (Calculated via NumPy)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()