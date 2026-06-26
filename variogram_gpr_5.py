import copy
import functools
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import gpytorch

import skgstat as skg
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

# Split 80% Train, 20% Temp
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    shuffle=False,
    random_state=SEED,
    # X_raw,
    # y_raw,
    # test_size=0.2,
    # random_state=SEED,
)
# Split Temp into 50% Val (10% total), 50% Test (10% total)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
    X_temp_raw,
    y_temp_raw,
    test_size=0.5,
    shuffle=False,
    random_state=SEED,
    # X_temp_raw,
    # y_temp_raw,
    # test_size=0.5,
    # random_state=SEED,
)

# ==========================================
# 2. SCALE DATA (FIT ONLY ON TRAINING SET)
# ==========================================
# Prevent data leakage by fitting scalers ONLY on the training data
X_scaler = StandardScaler().fit(X_train_raw)
y_scaler = StandardScaler().fit(y_train_raw.reshape(-1, 1))

X_train_scaled = X_scaler.transform(X_train_raw)
X_val_scaled = X_scaler.transform(X_val_raw)
X_test_scaled = X_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.transform(y_train_raw.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val_raw.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

# Convert to PyTorch Tensors
train_x = torch.tensor(X_train_scaled, dtype=torch.float64)
train_y = torch.tensor(y_train_scaled, dtype=torch.float64)
val_x = torch.tensor(X_val_scaled, dtype=torch.float64)
val_y = torch.tensor(y_val_scaled, dtype=torch.float64)
test_x = torch.tensor(X_test_scaled, dtype=torch.float64)
test_y = torch.tensor(y_test_scaled, dtype=torch.float64)

# ==========================================
# 3. GLOBAL VARIOGRAM DIAGNOSTICS (ON TRAIN SET ONLY)
# ==========================================
V = skg.Variogram(
    coordinates=X_train_scaled, values=y_train_scaled, normalize=False, n_lags=20
)
lags = np.array(V.bins)
semiv = np.array(V.experimental)
valid = ~np.isnan(semiv)
lags, semiv = lags[valid], semiv[valid]

n_first = max(10, int(len(lags) * 0.2))
nugget_est = max(
    0,
    LinearRegression()
    .fit(lags[:n_first].reshape(-1, 1), semiv[:n_first])
    .predict([[0]])[0],
)
sill_est = np.mean(semiv[-max(10, int(len(lags) * 0.2)) :])
range_idx = np.where(semiv >= 0.95 * sill_est)[0]
range_est = lags[range_idx[0]] if len(range_idx) > 0 else lags[-1]

print("--- TRAINING SET VARIOGRAM PRIORS ---")
print(f"Nugget: {nugget_est:.6f} | Sill: {sill_est:.6f} | Range: {range_est:.4f}\n")

# ==========================================
# PLOT THE DIAGNOSTICS
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(lags, semiv, label="Empirical Semivariance", color="blue", zorder=5)
plt.axhline(
    y=sill_est,
    color="r",
    linestyle="--",
    label=f"Estimated Sill ({sill_est:.4f})",
)
plt.axvline(
    x=range_est,
    color="g",
    linestyle="--",
    label=f"Estimated Range ({range_est:.2f})",
)
plt.scatter(
    [0],
    [nugget_est],
    color="m",
    s=100,
    marker="*",
    zorder=6,
    label=f"Estimated Nugget ({nugget_est:.4f})",
)
plt.xlabel("Lag Distance (Standardized Euclidean)")
plt.ylabel("Semivariance")
plt.title("Empirical Variogram with Automated Diagnostic Estimates")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ==========================================
# 4. DEFINE GPyTorch MODEL & BUILDERS
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


def build_model_A(x, y):
    """Variogram-Guided Model"""
    # Define the constraint based on the nugget
    if nugget_est < 1e-3:
        noise_constraint = gpytorch.constraints.Interval(1e-6, 1e-4)
    else:
        noise_constraint = gpytorch.constraints.Interval(
            # nugget_est * 0.1, nugget_est * 10.0
            nugget_est * 0.99,
            nugget_est * 1.01,
        )

    # Pass it correctly into the constructor
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint
    )
    model = ExactGPModel(x, y, likelihood, kernel_class=gpytorch.kernels.RBFKernel)

    # ... (rest of your outputscale and lengthscale constraints remain the same) ...
    model.covar_module.outputscale = torch.tensor(sill_est)
    model.covar_module.raw_outputscale_constraint = gpytorch.constraints.Interval(
        # sill_est * 0.1, sill_est * 10.0
        sill_est * 0.99,
        sill_est * 1.01,
    )

    ls_init = torch.ones(7) * (range_est / 2.0)
    model.covar_module.base_kernel.lengthscale = ls_init
    model.covar_module.base_kernel.raw_lengthscale_constraint = (
        # gpytorch.constraints.Interval(1e-3, range_est * 3.0)
        gpytorch.constraints.Interval(range_est * 0.99, range_est * 1.01)
    )

    return model, likelihood


def build_model_B(x, y):
    """Standard MLE Model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # likelihood.raw_noise_constraint = gpytorch.constraints.Interval(1e-5, 10.0)

    model = ExactGPModel(x, y, likelihood, kernel_class=gpytorch.kernels.RBFKernel)
    return model, likelihood


def rsetattr(obj, name, value):
    pre, _, post = name.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, value)


def rgetattr(obj, name, *args):
    def _getattr(obj, name):
        return getattr(obj, name, *args)

    return functools.reduce(_getattr, [obj] + name.split("."))


# ==========================================
# 5. ROBUST LOG-UNIFORM RANDOMIZATION HELPER
# ==========================================
def safe_log_uniform_sample(constraint, shape):
    """Samples hyperparameters log-uniformly within their constraint bounds."""
    lower = max(float(constraint.lower_bound), 1e-10)
    upper = float(constraint.upper_bound)
    if np.isinf(upper):
        upper = lower * 100.0  # Fallback if upper bound is infinite
    log_val = torch.rand(shape) * (np.log(upper) - np.log(lower)) + np.log(lower)
    return torch.exp(log_val)


# ==========================================
# 6. TRAINING & VALIDATION SELECTION LOOP
# ==========================================
def train_and_select_model(
    build_func,
    train_x,
    train_y,
    val_x,
    val_y_orig,
    y_scaler,
    num_restarts=10,
    training_iter=150,
):
    """
    Runs 1 smart initialization + (num_restarts - 1) random restarts.
    Selects the model state that achieves the highest R2 on the VALIDATION set.
    """
    best_val_r2 = -np.inf
    best_model = None
    best_likelihood = None

    base_model, base_likelihood = build_func(train_x, train_y)

    for j in range(num_restarts):
        # Deepcopy to preserve base constraints
        likelihood = copy.deepcopy(base_likelihood)
        model = ExactGPModel(
            train_x, train_y, likelihood, kernel_class=gpytorch.kernels.RBFKernel
        )

        # 1. Apply strict bounds for both models so log-uniform sampling works
        if build_func == build_model_A:
            model.covar_module.raw_outputscale_constraint = (
                # gpytorch.constraints.Interval(sill_est * 0.1, sill_est * 10.0)
                gpytorch.constraints.Interval(sill_est * 0.99, sill_est * 1.01)
            )
            model.covar_module.base_kernel.raw_lengthscale_constraint = (
                # gpytorch.constraints.Interval(1e-3, range_est * 3.0)
                gpytorch.constraints.Interval(range_est * 0.99, range_est * 1.01)
            )
        else:
            pass
            # model.covar_module.raw_outputscale_constraint = (
            #     gpytorch.constraints.Interval(1e-3, 10.0)
            # )
            # model.covar_module.base_kernel.raw_lengthscale_constraint = (
            #     gpytorch.constraints.Interval(1e-3, 10.0)
            # )

        # 2. Initialize Parameters
        if j == 0:
            # SMART INITIALIZATION (Restart 1)
            if build_func == build_model_A:
                model.covar_module.outputscale = torch.tensor(sill_est)
                model.covar_module.base_kernel.lengthscale = torch.ones(7) * (
                    # range_est / 2.0
                    range_est
                )
                if nugget_est < 1e-3:
                    likelihood.noise = torch.tensor(1e-5)
                else:
                    likelihood.noise = torch.tensor(nugget_est)
            init_type = "Smart Init"
        else:
            # RANDOM RESTARTS (Restarts 2-10)
            # We sample log-uniformly across the valid bounds to ensure diverse starting points
            ls_c = model.covar_module.base_kernel.raw_lengthscale_constraint
            model.covar_module.base_kernel.lengthscale = safe_log_uniform_sample(
                ls_c, model.covar_module.base_kernel.lengthscale.shape
            )

            os_c = model.covar_module.raw_outputscale_constraint
            model.covar_module.outputscale = safe_log_uniform_sample(
                os_c, torch.Size([1])
            )

            # FIX: Access the constraint via the noise_covar submodule
            n_c = likelihood.noise_covar.raw_noise_constraint
            # FIX: Set the actual noise property, which automatically updates the raw parameter
            likelihood.noise = safe_log_uniform_sample(n_c, torch.Size([1]))
            init_type = "Random"

        # 3. Train the model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # 4. Evaluate on VALIDATION set
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            val_preds_scaled = likelihood(model(val_x)).mean.numpy()

        val_preds_orig = y_scaler.inverse_transform(
            val_preds_scaled.reshape(-1, 1)
        ).flatten()
        val_r2 = r2_score(val_y_orig, val_preds_orig)

        # Print progress so you can see the 10 restarts happening!
        print(
            f"  Restart {j + 1}/{num_restarts} ({init_type:<10}) | Val R2: {val_r2:.5f}"
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = copy.deepcopy(model)
            best_likelihood = copy.deepcopy(likelihood)

    return best_model, best_likelihood, best_val_r2


print("--- RUNNING 10 RESTARTS WITH VALIDATION SELECTION ---")

# Train Model A
print("\nTraining Variogram-Guided Model (A)...")
model_A, likelihood_A, val_r2_A = train_and_select_model(
    build_model_A, train_x, train_y, val_x, y_val_raw, y_scaler, num_restarts=10
)
print(f" -> Best Validation R2 for Model A: {val_r2_A:.5f}")

# Train Model B
print("\nTraining Standard MLE Model (B)...")
model_B, likelihood_B, val_r2_B = train_and_select_model(
    build_model_B, train_x, train_y, val_x, y_val_raw, y_scaler, num_restarts=10
)
print(f" -> Best Validation R2 for Model B: {val_r2_B:.5f}\n")


# ==========================================
# 7. FINAL TEST EVALUATION
# ==========================================
def evaluate_on_test(model, likelihood, test_x, test_y_orig, y_scaler):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds_scaled = likelihood(model(test_x)).mean.numpy()
    preds_orig = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    r2 = r2_score(test_y_orig, preds_orig)
    rmse = np.sqrt(mean_squared_error(test_y_orig, preds_orig))
    return r2, rmse, preds_orig


test_r2_A, test_rmse_A, pred_A_test = evaluate_on_test(
    model_A, likelihood_A, test_x, y_test_raw, y_scaler
)
test_r2_B, test_rmse_B, pred_B_test = evaluate_on_test(
    model_B, likelihood_B, test_x, y_test_raw, y_scaler
)

# Extract final ARD Length Scales
ls_A = model_A.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
ls_B = model_B.covar_module.base_kernel.lengthscale.detach().numpy().flatten()

# ==========================================
# 8. PRINT RESULTS
# ==========================================
print("=" * 75)
print("FINAL RESULTS: 80-10-10 SPLIT EVALUATION")
print("=" * 75)
print(f"{'Metric':<25} | {'Variogram-Guided (A)':<20} | {'Standard MLE (B)':<20}")
print("-" * 75)
print(f"{'Validation R2 (Selection)':<25} | {val_r2_A:<20.5f} | {val_r2_B:<20.5f}")
print(f"{'TEST R2 (Unseen Data)':<25} | {test_r2_A:<20.5f} | {test_r2_B:<20.5f}")
print(f"{'TEST RMSE (Unseen Data)':<25} | {test_rmse_A:<20.5f} | {test_rmse_B:<20.5f}")

print("\n" + "=" * 75)
print("OPTIMIZED ARD LENGTH SCALES (Parameter Sensitivity)")
print("=" * 75)
print(f"{'Parameter':<10} | {'Variogram-Guided (A)':<20} | {'Standard MLE (B)':<20}")
print("-" * 55)
for i, feat in enumerate(features):
    print(f"{feat:<10} | {ls_A[i]:<20.4f} | {ls_B[i]:<20.4f}")

# ==========================================
# 9. VISUALIZE TEST PREDICTIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_test_raw, pred_A_test, alpha=0.8, edgecolors="k", c="blue", s=50)
axes[0].plot(
    [y_test_raw.min(), y_test_raw.max()],
    [y_test_raw.min(), y_test_raw.max()],
    "r--",
    lw=2,
)
axes[0].set_title(
    f"Variogram-Guided (Test Set)\nTest $R^2$={test_r2_A:.4f} | Val $R^2$={val_r2_A:.4f}"
)
axes[0].set_xlabel("Actual max_conc (FEA)")
axes[0].set_ylabel("Predicted max_conc (GPR)")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_raw, pred_B_test, alpha=0.8, edgecolors="k", c="green", s=50)
axes[1].plot(
    [y_test_raw.min(), y_test_raw.max()],
    [y_test_raw.min(), y_test_raw.max()],
    "r--",
    lw=2,
)
axes[1].set_title(
    f"Standard MLE (Test Set)\nTest $R^2$={test_r2_B:.4f} | Val $R^2$={val_r2_B:.4f}"
)
axes[1].set_xlabel("Actual max_conc (FEA)")
axes[1].set_ylabel("Predicted max_conc (GPR)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
