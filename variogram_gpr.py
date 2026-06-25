import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import skgstat as skg
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# 1. Load the dataset
df = pd.read_csv("datasets/solder_ball_conc.csv")

# 2. Separate inputs (design parameters) and output (objective)
features = ["d_pad", "t_pad", "d_us", "d_rep1", "t_ubm", "del_d", "h_ball"]
X = df[features].values
y = df["max_conc"].values

# 3. CRITICAL: Standardize the inputs to mean=0, variance=1
# This ensures all 7 dimensions contribute equally to the "distance" calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# PART A: The Empirical Variogram
# ==========================================
# We calculate the variogram in the 7D scaled parameter space.
V = skg.Variogram(coordinates=X_scaled, values=y, normalize=False, n_lags=50)

plt.figure(figsize=(8, 5))
V.plot()
plt.title("Empirical Variogram in 7D Parameter Space")
plt.xlabel("Euclidean Distance in Scaled Parameter Space")
plt.ylabel("Semivariance of max_conc")
plt.grid(True, linestyle="--", alpha=0.6)
# plt.show()

# ==========================================
# PART B: Gaussian Process Regression (GPR)
# ==========================================
y_var = np.var(y)
print("y_var: ", y_var)

# FIX 2: Widen the bounds to prevent ConvergenceWarnings
# Increased length_scale upper bound to 1e4, decreased noise lower bound to 1e-10
kernel = ConstantKernel(constant_value=y_var) * Matern(
    length_scale=np.ones(7), length_scale_bounds=(1e-2, 1e4), nu=1.5
) + WhiteKernel(
    noise_level=1e-6,
    noise_level_bounds="fixed",
    # noise_level_bounds=(1e-10, 0.1),
)

gpr = GaussianProcessRegressor(
    kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, random_state=42
)
gpr.fit(X_scaled, y)

# --- Extracting Engineering Insights (ARD) ---
# FIX 1: Directly access the Matern kernel instead of using a fragile loop.
# The kernel structure is (Constant * Matern) + White.
# Therefore, k1 is the Product, and k1.k2 is the Matern kernel.
matern_kernel = gpr.kernel_.k1.k2
optimized_length_scales = matern_kernel.length_scale

# Create a dataframe to rank parameter sensitivity
sensitivity_df = pd.DataFrame(
    {"Design Parameter": features, "Optimized Length Scale": optimized_length_scales}
).sort_values(by="Optimized Length Scale")

print("\n--- Parameter Sensitivity (ARD Length Scales) ---")
print("(Smaller length scale = max_conc is highly sensitive to this parameter)")
print(sensitivity_df.to_string(index=False))

# Plot GPR Predictions vs Actual FEA results
y_pred, y_std = gpr.predict(X_scaled, return_std=True)

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, alpha=0.7, edgecolors="k", c="blue", label="Data Points")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Perfect Fit")
plt.xlabel("Actual max_conc (FEA)")
plt.ylabel("Predicted max_conc (GPR)")
plt.title("GPR Surrogate Model Accuracy")
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()
