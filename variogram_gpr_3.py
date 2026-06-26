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
from sklearn.model_selection import KFold

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
# 1. LOAD AND SCALE DATA
# ==========================================
df = pd.read_csv("datasets/solder_ball_conc.csv")
features = ["d_pad", "t_pad", "d_us", "d_rep1", "t_ubm", "del_d", "h_ball"]
X_np = df[features].values
y_np = df["max_conc"].values

scaler = StandardScaler()
X_scaled_np = scaler.fit_transform(X_np)

# Convert to PyTorch Tensors
train_x_full = torch.tensor(X_scaled_np, dtype=torch.float64)
train_y_full = torch.tensor(y_np, dtype=torch.float64)

# ==========================================
# 2. GLOBAL VARIOGRAM DIAGNOSTICS (For Model A Priors)
# ==========================================
V = skg.Variogram(coordinates=X_scaled_np, values=y_np, normalize=False, n_lags=15)
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

print("--- GLOBAL VARIOGRAM PRIORS ---")
print(f"Nugget: {nugget_est:.6f} | Sill: {sill_est:.6f} | Range: {range_est:.4f}\n")


# ==========================================
# 3. DEFINE GPyTorch MODEL & BUILDERS
# ==========================================
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(
#         self, train_x, train_y, likelihood, kernel_type="matern", ard_num_dims=7
#     ):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         base_kernel = (
#             gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
#             if kernel_type == "matern"
#             else gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
#         )
#         self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

#     def forward(self, x):
#         return gpytorch.distributions.MultivariateNormal(
#             self.mean_module(x), self.covar_module(x)
#         )


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
        kernel_class: gpytorch.kernels.RBFKernel,
        uniform: bool = False,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel_class(ard_num_dims=1 if uniform else train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_model_A(x, y):
    """Variogram-Guided Model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.raw_noise_constraint = gpytorch.constraints.Interval(1e-6, 1e-4)
    # likelihood.noise = torch.tensor(1e-5)

    model = ExactGPModel(x, y, likelihood, kernel_class=gpytorch.kernels.RBFKernel)
    model.covar_module.outputscale = torch.tensor(sill_est)
    model.covar_module.raw_outputscale_constraint = gpytorch.constraints.Interval(
        sill_est * 0.1, sill_est * 10.0
    )

    ls_init = torch.ones(7) * (range_est / 2.0)
    model.covar_module.base_kernel.lengthscale = ls_init
    model.covar_module.base_kernel.raw_lengthscale_constraint = (
        gpytorch.constraints.Interval(1e-3, range_est * 3.0)
    )
    return model, likelihood


def build_model_B(x, y):
    """Standard MLE Model"""
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.raw_noise_constraint = gpytorch.constraints.Interval(1e-5, 10.0)

    model = ExactGPModel(x, y, likelihood, kernel_class=gpytorch.kernels.RBFKernel)
    # model.covar_module.base_kernel.raw_lengthscale_constraint = (
    #     gpytorch.constraints.Interval(1e-2, 100.0)
    # )
    return model, likelihood


def rsetattr(obj, name, value):
    """
    Recursive setattr
    """
    pre, _, post = name.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, value)


def rgetattr(obj, name, *args):
    """
    Recursive getattr
    """

    def _getattr(obj, name):
        return getattr(obj, name, *args)

    return functools.reduce(_getattr, [obj] + name.split("."))


def train_and_evaluate(
    model,
    likelihood,
    train_x,
    train_y,
    test_x,
    test_y,
    training_iter=150,
    random_restart=True,
    noisy=True,
    kernel_class=gpytorch.kernels.RBFKernel,
    uniform=False,
):
    # """Trains the model and returns predictions and loss"""
    # model.train()
    # likelihood.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    min_loss_rr = torch.inf
    model_min_loss_rr = None
    likelihood_min_loss_rr = None

    for j in range(10 if random_restart else 1):
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=None
            if noisy
            else gpytorch.constraints.Interval(1e-4, 1e-3)
        )

        model = ExactGPModel(
            train_x,
            train_y,
            likelihood,
            kernel_class=kernel_class,
            uniform=uniform,
        )

        if random_restart:
            if j > 0:
                for name, parameter in model.named_parameters():
                    rsetattr(
                        model, name, torch.nn.Parameter(torch.randn_like(parameter))
                    )

        model_min_loss = copy.deepcopy(model)
        likelihood_min_loss = copy.deepcopy(likelihood)
        # training_iter = 1

        # model.covar_module.base_kernel.raw_lengthscale_constraint = (
        #     gpytorch.constraints.Interval(1.0, 1.001)
        # )

        # model.covar_module.raw_outputscale_constraint = gpytorch.constraints.Interval(
        #     1.0, 1.001
        # )

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.1
        )  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        min_loss = torch.inf
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)

            model.double()

            # Calc loss and backprop gradients
            loss = -mll(output, train_y)

            if loss.item() < min_loss:
                model_min_loss = copy.deepcopy(model)
                likelihood_min_loss = copy.deepcopy(likelihood)

                min_loss = loss.item()

                if min_loss < min_loss_rr:
                    pass
                    # print(j, i, loss.item())
                    # print(list(model.parameters()))

            loss.backward()
            optimizer.step()

            # print(list(model.named_parameters()))

        if min_loss < min_loss_rr:
            # print("model copied")
            model_min_loss_rr = copy.deepcopy(model_min_loss)
            likelihood_min_loss_rr = copy.deepcopy(likelihood_min_loss)
            min_loss_rr = min_loss

    model_min_loss_rr.eval()
    likelihood_min_loss_rr.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood_min_loss_rr(model_min_loss_rr(test_x))

    return preds.mean.numpy(), -mll(preds, test_y).item()


# ==========================================
# 4. 10-FOLD CROSS VALIDATION
# ==========================================
# kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
kf = KFold(n_splits=10)
cv_scores_A = {"r2": [], "rmse": []}
cv_scores_B = {"r2": [], "rmse": []}

print("--- RUNNING 10-FOLD CROSS VALIDATION ---")
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled_np)):
    # print(train_idx, test_idx)

    # Split data for this fold
    fold_train_x = train_x_full[train_idx]
    fold_train_y = train_y_full[train_idx]
    fold_test_x = train_x_full[test_idx]
    fold_test_y = train_y_full[test_idx]

    # --- Model A ---
    model_A, likelihood_A = build_model_A(fold_train_x, fold_train_y)
    pred_A, _ = train_and_evaluate(
        model_A, likelihood_A, fold_train_x, fold_train_y, train_x_full, train_y_full
    )
    cv_scores_A["r2"].append(r2_score(train_y_full.numpy(), pred_A))
    cv_scores_A["rmse"].append(
        np.sqrt(mean_squared_error(train_y_full.numpy(), pred_A))
    )

    # --- Model B ---
    model_B, likelihood_B = build_model_B(fold_train_x, fold_train_y)
    pred_B, _ = train_and_evaluate(
        model_B, likelihood_B, fold_train_x, fold_train_y, train_x_full, train_y_full
    )
    cv_scores_B["r2"].append(r2_score(train_y_full.numpy(), pred_B))
    cv_scores_B["rmse"].append(
        np.sqrt(mean_squared_error(train_y_full.numpy(), pred_B))
    )

    print(
        f"Fold {fold + 1}/10 | Model A R2: {cv_scores_A['r2'][-1]:.5f} | Model B R2: {cv_scores_B['r2'][-1]:.5f}"
    )

# Print CV Summary
print("\n" + "=" * 60)
print("10-FOLD CROSS VALIDATION SUMMARY")
print("=" * 60)
print(f"{'Metric':<20} | {'Variogram-Guided (A)':<20} | {'Standard MLE (B)':<20}")
print("-" * 65)
print(
    f"{'Mean R2':<20} | {np.mean(cv_scores_A['r2']):<20.5f} | {np.mean(cv_scores_B['r2']):<20.5f}"
)
print(
    f"{'Std Dev R2':<20} | {np.std(cv_scores_A['r2']):<20.5f} | {np.std(cv_scores_B['r2']):<20.5f}"
)
print(
    f"{'Mean RMSE':<20} | {np.mean(cv_scores_A['rmse']):<20.5f} | {np.mean(cv_scores_B['rmse']):<20.5f}"
)
print(
    f"{'Std Dev RMSE':<20} | {np.std(cv_scores_A['rmse']):<20.5f} | {np.std(cv_scores_B['rmse']):<20.5f}"
)

# ==========================================
# 5. FINAL TRAINING ON FULL DATASET
# ==========================================
print("\n--- TRAINING FINAL MODELS ON FULL DATASET ---")
model_A_final, likelihood_A_final = build_model_A(train_x_full, train_y_full)
model_B_final, likelihood_B_final = build_model_B(train_x_full, train_y_full)

# Train for more iterations on the full dataset
pred_A_final, lml_A_final = train_and_evaluate(
    model_A_final,
    likelihood_A_final,
    train_x_full,
    train_y_full,
    train_x_full,
    train_y_full,
    training_iter=300,
)
pred_B_final, lml_B_final = train_and_evaluate(
    model_B_final,
    likelihood_B_final,
    train_x_full,
    train_y_full,
    train_x_full,
    train_y_full,
    training_iter=300,
)

# Extract final ARD Length Scales
ls_A_final = (
    model_A_final.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
)
ls_B_final = (
    model_B_final.covar_module.base_kernel.lengthscale.detach().numpy().flatten()
)

print("\n" + "=" * 60)
print("FINAL OPTIMIZED ARD LENGTH SCALES (Full Dataset)")
print("=" * 60)
print(f"{'Parameter':<10} | {'Variogram-Guided (A)':<20} | {'Standard MLE (B)':<20}")
print("-" * 55)
for i, feat in enumerate(features):
    print(f"{feat:<10} | {ls_A_final[i]:<20.4f} | {ls_B_final[i]:<20.4f}")

# ==========================================
# 6. VISUALIZE FINAL PREDICTIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_np, pred_A_final, alpha=0.7, edgecolors="k", c="blue")
axes[0].plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], "r--", lw=2)
axes[0].set_title(f"Variogram-Guided (Full Data)\nLML={lml_A_final:.1f}")
axes[0].set_xlabel("Actual max_conc")
axes[0].set_ylabel("Predicted")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_np, pred_B_final, alpha=0.7, edgecolors="k", c="green")
axes[1].plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], "r--", lw=2)
axes[1].set_title(f"Standard MLE (Full Data)\nLML={lml_B_final:.1f}")
axes[1].set_xlabel("Actual max_conc")
axes[1].set_ylabel("Predicted")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
