import copy
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import gpytorch
import skgstat
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ==========================================
# 0. SET RANDOM SEEDS
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. LOAD DATA & 80-10-10 SPLIT
# ==========================================
df = pd.read_csv("datasets/solder_ball_conc.csv")
features = ["d_pad", "t_pad", "d_us", "d_rep1", "t_ubm", "del_d", "h_ball"]
X_raw = df[features].values
y_raw = df["max_conc"].values

X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=SEED
)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp_raw, y_temp_raw, test_size=0.5, random_state=SEED
)

# ==========================================
# 2. SCALE DATA (FIT ONLY ON TRAINING SET)
# ==========================================
X_scaler = StandardScaler().fit(X_train_raw)
y_scaler = StandardScaler().fit(y_train_raw.reshape(-1, 1))

X_train_scaled = X_scaler.transform(X_train_raw)
X_val_scaled = X_scaler.transform(X_val_raw)
X_test_scaled = X_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

train_x = torch.tensor(X_train_scaled, dtype=torch.float64)
train_y = torch.tensor(y_train_scaled, dtype=torch.float64)
val_x = torch.tensor(X_val_scaled, dtype=torch.float64)
val_y = torch.tensor(y_val_scaled, dtype=torch.float64)
test_x = torch.tensor(X_test_scaled, dtype=torch.float64)
test_y = torch.tensor(y_test_scaled, dtype=torch.float64)

# ==========================================
# 3. EMPIRICAL VARIOGRAM ON TRAINING SET
# ==========================================
V = skgstat.Variogram(
    coordinates=X_train_scaled, values=y_train_scaled, normalize=False, n_lags=10
)
lags = np.array(V.bins)
semiv = np.array(V.experimental)
valid = ~np.isnan(semiv)
lags, semiv = lags[valid], semiv[valid]

n_first = max(2, int(len(lags) * 0.2))
nugget_est = max(
    0,
    LinearRegression()
    .fit(lags[:n_first].reshape(-1, 1), semiv[:n_first])
    .predict([[0]])[0],
)
sill_est = np.mean(semiv[-max(2, int(len(lags) * 0.2)) :])
range_idx = np.where(semiv >= 0.95 * sill_est)[0]
range_est = lags[range_idx[0]] if len(range_idx) > 0 else lags[-1]

print("=" * 60)
print("EMPIRICAL VARIOGRAM ESTIMATES (Training Set)")
print("=" * 60)
print(f"Nugget (Noise): {nugget_est:.6f}")
print(f"Sill (Variance): {sill_est:.6f}")
print(f"Range (Distance): {range_est:.4f}")
print("-> Note: This Range is applied equally to ALL 7 dimensions.\n")


# ==========================================
# 4. DEFINE GPyTorch MODEL CLASS
# ==========================================
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_class):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel_class(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ==========================================
# 5. MODEL A: FIXED VARIOGRAM (NO TRAINING)
# ==========================================
def build_model_A_fixed(x, y):
    """Builds a GP model with parameters strictly fixed to variogram estimates."""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x, y, likelihood, kernel_class=gpytorch.kernels.RBFKernel)

    # 1. Set Outputscale to Sill
    model.covar_module.outputscale = torch.tensor(sill_est)

    # 2. Set Lengthscale to Range (Isotropic assumption across 7D)
    # We divide by 2.0 as the length scale is mathematically shorter than the practical range
    ls_val = max(range_est / 2.0, 1e-3)
    model.covar_module.base_kernel.lengthscale = torch.ones(7) * ls_val

    # 3. Set Noise to Nugget (with a tiny jitter floor to prevent Cholesky crashes)
    noise_val = max(nugget_est, 1.1e-4)
    likelihood.noise = torch.tensor(noise_val)

    # CRITICAL: FREEZE ALL PARAMETERS. No gradients will be calculated.
    for param in model.parameters():
        param.requires_grad = False
    for param in likelihood.parameters():
        param.requires_grad = False

    return model, likelihood


# ==========================================
# 6. MODEL B: TRAINED MLE (FULL OPTIMIZATION)
# ==========================================
def build_and_train_model_B(
    x, y, val_x, val_y_orig, y_scaler, num_restarts=10, training_iter=200
):
    """Builds and fully optimizes a GP model using MLE with random restarts."""
    best_val_r2 = -np.inf
    best_model = None
    best_likelihood = None

    for j in range(num_restarts):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.noise_constraint = gpytorch.constraints.Interval(
            1e-5, 10.0
        )

        model = ExactGPModel(x, y, likelihood, kernel_class=gpytorch.kernels.RBFKernel)
        model.covar_module.outputscale_constraint = gpytorch.constraints.Interval(
            1e-3, 10.0
        )
        model.covar_module.base_kernel.lengthscale_constraint = (
            gpytorch.constraints.Interval(1e-3, 10.0)
        )

        # Initialize Parameters
        if j == 0:
            # SMART INITIALIZATION (Restart 1)
            model.covar_module.outputscale = torch.tensor(sill_est, dtype=torch.float64)
            model.covar_module.base_kernel.lengthscale = torch.ones(
                7, dtype=torch.float64
            ) * (range_est / 2.0)
            likelihood.noise = torch.tensor(
                max(nugget_est, 1.1e-4), dtype=torch.float64
            )
            init_type = "Smart Init"
        else:
            # RANDOM RESTARTS (Restarts 2-10)
            # We sample from a standard normal distribution and use the constraint's
            # transform method to map it smoothly into the valid bounded space.

            # 1. Lengthscale (ARD)
            ls_c = model.covar_module.base_kernel.lengthscale_constraint
            model.covar_module.base_kernel.lengthscale = ls_c.transform(
                torch.randn_like(model.covar_module.base_kernel.lengthscale)
            )

            # 2. Outputscale
            os_c = model.covar_module.outputscale_constraint
            model.covar_module.outputscale = os_c.transform(
                torch.randn(1, dtype=torch.float64)
            )

            # 3. Noise
            n_c = likelihood.noise_covar.noise_constraint
            likelihood.noise = n_c.transform(torch.randn(1, dtype=torch.float64))
            init_type = "Random"

        # Train the model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

        # Evaluate on Validation set to pick the best restart
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_preds_scaled = likelihood(model(val_x)).mean.numpy()
        val_preds_orig = y_scaler.inverse_transform(
            val_preds_scaled.reshape(-1, 1)
        ).flatten()
        val_r2 = r2_score(val_y_orig, val_preds_orig)
        print("validation error calculated between: ", val_y_orig, val_preds_orig)

        print(
            f"  Restart {j + 1}/{num_restarts} ({init_type:<10}) | Val R2: {val_r2:.5f}"
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = copy.deepcopy(model)
            best_likelihood = copy.deepcopy(likelihood)

    return best_model, best_likelihood, best_val_r2


# ==========================================
# 7. EVALUATION FUNCTION
# ==========================================
def evaluate_model(model, likelihood, test_x, test_y_orig, y_scaler):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds_scaled = likelihood(model(test_x)).mean.numpy()
    preds_orig = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    r2 = r2_score(test_y_orig, preds_orig)
    print("test error calculated between:", test_y_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(test_y_orig, preds_orig))
    return r2, rmse, preds_orig


# ==========================================
# 8. RUN EXPERIMENTS
# ==========================================
print("--- EVALUATING MODEL A (Fixed Variogram, No Training) ---")
model_A, likelihood_A = build_model_A_fixed(train_x, train_y)
model_A.eval()
likelihood_A.eval()
# Evaluate on Val
val_preds_A_scaled = likelihood_A(model_A(val_x)).mean.numpy()
val_preds_A_orig = y_scaler.inverse_transform(
    val_preds_A_scaled.reshape(-1, 1)
).flatten()
val_r2_A = r2_score(y_val_raw, val_preds_A_orig)
# Evaluate on Test
test_r2_A, test_rmse_A, pred_A_test = evaluate_model(
    model_A, likelihood_A, test_x, y_test_raw, y_scaler
)
print(f" -> Validation R2: {val_r2_A:.5f}")
print(f" -> Test R2: {test_r2_A:.5f} | Test RMSE: {test_rmse_A:.5f}\n")

print("--- EVALUATING MODEL B (Trained MLE, 10 Random Restarts) ---")
model_B, likelihood_B, val_r2_B = build_and_train_model_B(
    train_x, train_y, val_x, y_val_raw, y_scaler
)
test_r2_B, test_rmse_B, pred_B_test = evaluate_model(
    model_B, likelihood_B, test_x, y_test_raw, y_scaler
)
print(f" -> Best Validation R2 (Selection): {val_r2_B:.5f}")
print(f" -> Test R2: {test_r2_B:.5f} | Test RMSE: {test_rmse_B:.5f}\n")

# Extract final ARD Length Scales for Model B
ls_B = model_B.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
# ==========================================
# 9. PRINT FINAL COMPARISON
# ==========================================
print("=" * 70)
print("FINAL RESULTS: SPARSE 7D DATA (100 POINTS)")
print("=" * 70)
print(
    f"{'Metric':<30} | {'Model A: Fixed Variogram':<20} | {'Model B: Trained MLE':<20}"
)
print("-" * 75)
print(
    f"{'Training Method':<30} | {'None (Frozen)':<20} | {'MLE (Adam + Restarts)':<20}"
)
print(f"{'Validation R2':<30} | {val_r2_A:<20.5f} | {val_r2_B:<20.5f}")
print(f"{'TEST R2 (Unseen Data)':<30} | {test_r2_A:<20.5f} | {test_r2_B:<20.5f}")
print(f"{'TEST RMSE':<30} | {test_rmse_A:<20.5f} | {test_rmse_B:<20.5f}")

print("\n" + "=" * 70)
print("MODEL B OPTIMIZED ARD LENGTH SCALES (Parameter Sensitivity)")
print("=" * 70)
print("(Model A was forced to use the exact same length scale for all 7 parameters)")
for i, feat in enumerate(features):
    print(f"{feat:<10} : {ls_B[i]:.4f}")

# ==========================================
# 10. VISUALIZE TEST PREDICTIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test_raw, pred_A_test, alpha=0.8, edgecolors="k", c="red", s=50)
axes[0].plot(
    [y_test_raw.min(), y_test_raw.max()],
    [y_test_raw.min(), y_test_raw.max()],
    "k--",
    lw=2,
)
axes[0].set_title(f"Model A: Fixed Variogram\nTest $R^2$ = {test_r2_A:.4f}")
axes[0].set_xlabel("Actual max_conc")
axes[0].set_ylabel("Predicted")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_raw, pred_B_test, alpha=0.8, edgecolors="k", c="blue", s=50)
axes[1].plot(
    [y_test_raw.min(), y_test_raw.max()],
    [y_test_raw.min(), y_test_raw.max()],
    "k--",
    lw=2,
)
axes[1].set_title(f"Model B: Trained MLE\nTest $R^2$ = {test_r2_B:.4f}")
axes[1].set_xlabel("Actual max_conc")
axes[1].set_ylabel("Predicted")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
