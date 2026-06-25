import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import skgstat as skg
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    WhiteKernel,
    DotProduct,
)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
df = pd.read_csv("datasets/solder_ball_conc.csv")
features = ["d_pad", "t_pad", "d_us", "d_rep1", "t_ubm", "del_d", "h_ball"]
X = df[features].values
y = df["max_conc"].values

# Standardize inputs (CRITICAL for 7D distance calculations)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate Empirical Variogram
V = skg.Variogram(coordinates=X_scaled, values=y, normalize=False, n_lags=15)
lags = np.array(V.bins)
semiv = np.array(V.experimental)

# Fit a theoretical spherical model directly to the empirical points
V.fit("spherical")

# Extract the exact mathematical parameters
emp_range, emp_sill, emp_nugget = V.parameters

print(f"Nugget: {emp_nugget}")
print(f"Sill:   {emp_sill}")
print(f"Range:  {emp_range}")

# Clean up any NaN bins
valid = ~np.isnan(semiv)
lags = lags[valid]
semiv = semiv[valid]

print("--- EMPIRICAL VARIOGRAM DIAGNOSTIC CHECKLIST ---\n")

# ==========================================
# CHECK 1: NUGGET (Y-Intercept)
# ==========================================
# Heuristic: Extrapolate the first 20% of the lags back to distance = 0
n_first = max(2, int(len(lags) * 0.2))
lin_reg_origin = LinearRegression().fit(lags[:n_first].reshape(-1, 1), semiv[:n_first])
nugget_estimate = max(0, lin_reg_origin.predict([[0]])[0])

print(f"1. Estimated Nugget (Y-intercept): {nugget_estimate:.6f}")
if nugget_estimate < 1e-4:
    print("   -> DIAGNOSIS: Nugget is effectively zero (deterministic FEA data).")
    print(
        "   -> ACTION: Fix WhiteKernel to a tiny jitter (1e-6) to prevent singular matrix."
    )
    noise_bounds = "fixed"
    noise_init = 1e-6
else:
    print("   -> DIAGNOSIS: Significant nugget effect detected (noisy data).")
    print("   -> ACTION: Let WhiteKernel optimize to capture noise.")
    noise_bounds = (1e-6, 1.0)
    noise_init = nugget_estimate

# ==========================================
# CHECK 2: SILL (Plateau)
# ==========================================
# Heuristic: Average the semivariance of the last 20% of the lags
n_last = max(2, int(len(lags) * 0.2))
sill_estimate = np.mean(semiv[-n_last:])
print(f"\n2. Estimated Sill (Plateau): {sill_estimate:.6f}")
print("   -> ACTION: Initialize ConstantKernel with this value.")

# ==========================================
# CHECK 3: RANGE (Distance to Sill)
# ==========================================
# Heuristic: Find the first lag where semivariance reaches 95% of the sill
threshold = 0.95 * sill_estimate
range_indices = np.where(semiv >= threshold)[0]
range_estimate = lags[range_indices[0]] if len(range_indices) > 0 else lags[-1]

print(f"\n3. Estimated Range (95% of Sill): {range_estimate:.4f}")
print(f"   -> ACTION: Set length_scale upper bound to ~{range_estimate * 3:.2f}.")

# ==========================================
# CHECK 4: ORIGIN SHAPE (Smooth vs. Sharp)
# ==========================================
# Heuristic: Compare linear vs. quadratic fit on the points BEFORE the Range.
# A linear origin means the function is "rough" (Matern). A parabolic origin means it's "smooth" (RBF).
range_idx = np.searchsorted(lags, range_estimate)
if range_idx < 3:
    range_idx = min(3, len(lags))

subset_lags = lags[:range_idx]
subset_semiv = semiv[:range_idx]

# Linear fit residuals
lin_model = LinearRegression().fit(subset_lags.reshape(-1, 1), subset_semiv)
ss_res_lin = np.sum((subset_semiv - lin_model.predict(subset_lags.reshape(-1, 1))) ** 2)

# Quadratic fit residuals
X_quad = np.column_stack((subset_lags, subset_lags**2))
quad_model = LinearRegression().fit(X_quad, subset_semiv)
ss_res_quad = np.sum((subset_semiv - quad_model.predict(X_quad)) ** 2)

print("\n4. Origin Shape Analysis:")
print(f"   Linear Residuals:    {ss_res_lin:.6f}")
print(f"   Quadratic Residuals: {ss_res_quad:.6f}")

if ss_res_lin < ss_res_quad:
    print("   -> DIAGNOSIS: Origin is linear/sharp.")
    print(
        "   -> ACTION: Use Matern kernel (nu=1.5) for rough/continuous FEA functions."
    )
    kernel_choice = "Matern"
else:
    print("   -> DIAGNOSIS: Origin is parabolic/smooth.")
    print("   -> ACTION: Use RBF (Squared Exponential) kernel.")
    kernel_choice = "RBF"

# ==========================================
# CHECK 5: TREND (Unbounded vs. Bounded)
# ==========================================
# Heuristic: Check if the variogram is still rising steeply at the very end
last_30_idx = int(len(lags) * 0.7)
if last_30_idx < len(lags) - 1:
    slope_last = (
        LinearRegression()
        .fit(lags[last_30_idx:].reshape(-1, 1), semiv[last_30_idx:])
        .coef_[0]
    )
    relative_slope = slope_last / (sill_estimate / lags[-1]) if sill_estimate > 0 else 0

    print(f"\n5. Trend Analysis (Relative slope at end: {relative_slope:.4f}):")
    if relative_slope > 0.2:
        print("   -> DIAGNOSIS: Variogram is unbounded. Global trend exists.")
        print("   -> ACTION: Add a DotProduct kernel to capture the trend.")
        has_trend = True
    else:
        print("   -> DIAGNOSIS: Variogram flattens out. No strong global trend.")
        print("   -> ACTION: Use standard zero-mean GPR.")
        has_trend = False
else:
    has_trend = False

# ==========================================
# BUILD THE RECOMMENDED KERNEL
# ==========================================
print("\n" + "=" * 50)
print("GENERATING RECOMMENDED SCIKIT-LEARN KERNEL...")
print("=" * 50)

# 1. Signal Variance (Sill)
c_kernel = ConstantKernel(constant_value=sill_estimate, constant_value_bounds="fixed")

# 2. Main Spatial Kernel (Range & Shape)
length_scale_init = np.ones(len(features)) * (range_estimate / 2.0)
length_scale_bounds = (1e-3, range_estimate * 3.0)

if kernel_choice == "Matern":
    main_kernel = Matern(
        length_scale=length_scale_init, length_scale_bounds=length_scale_bounds, nu=1.5
    )
else:
    main_kernel = RBF(
        length_scale=length_scale_init, length_scale_bounds=length_scale_bounds
    )

# 3. Noise Kernel (Nugget)
w_kernel = WhiteKernel(noise_level=noise_init, noise_level_bounds=noise_bounds)

# Combine them
if has_trend:
    trend_kernel = DotProduct()
    recommended_kernel = (c_kernel * main_kernel) + trend_kernel + w_kernel
    print("Structure: (Constant * Matern/RBF) + DotProduct (Trend) + White (Noise)")
else:
    recommended_kernel = (c_kernel * main_kernel) + w_kernel
    print("Structure: (Constant * Matern/RBF) + White (Noise)")

print("\nRecommended Kernel Object:")
print(recommended_kernel)

# ==========================================
# PLOT THE DIAGNOSTICS
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(lags, semiv, label="Empirical Semivariance", color="blue", zorder=5)
plt.axhline(
    y=sill_estimate,
    color="r",
    linestyle="--",
    label=f"Estimated Sill ({sill_estimate:.4f})",
)
plt.axvline(
    x=range_estimate,
    color="g",
    linestyle="--",
    label=f"Estimated Range ({range_estimate:.2f})",
)
plt.scatter(
    [0],
    [nugget_estimate],
    color="m",
    s=100,
    marker="*",
    zorder=6,
    label=f"Estimated Nugget ({nugget_estimate:.4f})",
)
plt.xlabel("Lag Distance (Standardized Euclidean)")
plt.ylabel("Semivariance")
plt.title("Empirical Variogram with Automated Diagnostic Estimates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
