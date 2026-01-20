import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel
from sklearn.preprocessing import StandardScaler

from gp import train


def plot_3d(X, X_bounds, y, scaler, model, likelihood):
    x_grid = torch.linspace(0, 1, 50)
    xx, yy = torch.meshgrid(x_grid, x_grid)
    test_x = torch.stack([xx, yy]).reshape(2, -1).T

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], y.flatten())

    ax.set_xlabel("t_lf")
    ax.set_ylabel("t_solder")
    ax.set_zlabel("MidDieStress")

    # ax.plot_surface(xx, yy, observed_pred.mean.reshape(50, 50).detach().numpy(), cmap='viridis', alpha=0.5)
    ax.plot_surface(
        xx * (X_bounds[0, 1] - X_bounds[0, 0]) + X_bounds[0, 0],
        yy * (X_bounds[1, 1] - X_bounds[1, 0]) + X_bounds[1, 0],
        scaler.inverse_transform(observed_pred.mean.detach().numpy()[:, None]).reshape(
            50, 50
        ),
        cmap="viridis",
        alpha=0.5,
    )

    plt.show()


if __name__ == "__main__":
    df_data = pd.read_csv("datasets/Kriging_data.csv", header=[0, 1])

    X = df_data[["t_lf", "t_solder"]].values

    X_bounds = np.array([[0.2, 1.0], [0.02, 0.08]])
    X_scaled = torch.tensor((X - X_bounds[:, 0]) / (X_bounds[:, 1] - X_bounds[:, 0]))

    # y = np.sum(X**2, 1)[:, None]
    y = df_data[("MidDieStress", "bot4")].values[:, None]
    scaler = StandardScaler()
    y_scaled = torch.tensor(scaler.fit_transform(y))

    # df_1 = pd.concat(
    #     (df_data[["t_lf", "t_solder"]], df_data[[("MidDieStress", "top2")]]), axis=1
    # )
    # sns.pairplot(df_1)

    test_idx_list = []
    train_idx_list = []

    for i in range(5):
        test_idx = np.arange(5 * i, 5 * (i + 1))
        train_idx = np.delete(np.arange(25), test_idx)

        test_idx_list.append(test_idx)
        train_idx_list.append(train_idx)

    kernel_class_list = [MaternKernel, RBFKernel, RQKernel]

    for kernel_class in kernel_class_list:
        for fold, (train_idx, test_idx) in enumerate(
            zip(train_idx_list, test_idx_list)
        ):
            model, likelihood = train(
                X_scaled=X_scaled[train_idx],
                y_scaled=y_scaled[train_idx],
                # X_scaled=X_scaled,
                # y_scaled=y_scaled,
                kernel_class=kernel_class,
                ref=True,
            )

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(X_scaled[test_idx]))

            mse = torch.mean((observed_pred.mean - y_scaled[test_idx].flatten()) ** 2)
            print(kernel_class.__name__, fold, mse.item())

            plot_3d(X, X_bounds, y, scaler, model, likelihood)

    for fold, (train_idx, test_idx) in enumerate(zip(train_idx_list, test_idx_list)):
        model, likelihood = train(
            X_scaled=X_scaled[train_idx],
            y_scaled=y_scaled[train_idx],
            # X_scaled=X_scaled,
            # y_scaled=y_scaled,
            kernel_class=RBFKernel,
            training_iter=1,
            ref=True,
        )

        custom_lengthscale = 1 / torch.tensor([[2.832268486, 1.080779145]]) ** 2
        custom_outputscale = torch.tensor([10.0])
        # custom_noise = torch.tensor([1e0])

        model.covar_module.base_kernel.raw_lengthscale_constraint = (
            gpytorch.constraints.Positive()
        )

        model.covar_module.base_kernel.lengthscale = custom_lengthscale
        model.covar_module.outputscale = custom_outputscale
        # model.likelihood.noise_covar.noise = custom_noise

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X_scaled[test_idx]))

        mse = torch.mean((observed_pred.mean - y_scaled[test_idx].flatten()) ** 2)
        # print("Reference", fold, mse.item())

        # plot_3d(X, X_bounds, y, scaler, model, likelihood)

    for fold, (train_idx, test_idx) in enumerate(zip(train_idx_list, test_idx_list)):
        mse = torch.mean(y_scaled[test_idx].flatten() ** 2)
        print("Naive", fold, mse.item())
