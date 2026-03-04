import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from f3dasm.datageneration.functions import *
from f3dasm.datageneration.functions.adapters.pybenchfunction import PyBenchFunction
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel
from scipy.stats.qmc import LatinHypercube
from sklearn.preprocessing import StandardScaler

from gp import train

torch.manual_seed(0)
np.random.seed(0)

rng = np.random.default_rng(seed=0)


class AlpineN2(PyBenchFunction):
    name = "Alpine N. 2"
    continuous = True
    convex = False
    separable = True
    differentiable = True
    multimodal = True
    randomized_term = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (isinstance(d, int) and (not d < 0)), (
            "The dimension d must be None or a positive integer"
        )
        return (d is None) or (d > 0)

    def _set_parameters(self):
        d = self.dimensionality
        self.input_domain = np.array([[0, 10] for _ in range(d)])

    def get_param(self):
        return {}

    def get_global_minimum(self, d):
        X = np.array([7.917 for i in range(d)])
        return (
            self._retrieve_original_input(X),
            self(self._retrieve_original_input(X)),
        )

    def evaluate(self, X):
        res = -np.prod(np.sqrt(X) * np.sin(X))
        return res


if __name__ == "__main__":
    df_data = pd.read_csv("datasets/solder_ball_conc.csv", header=0, index_col=0)

    data_dimensionality = 1
    data_size = 10
    folds = 5

    data_bounds = np.tile([0.0, 1.0], (data_dimensionality, 1))

    data_sampler = LatinHypercube(d=data_dimensionality, scramble=False, rng=rng)
    X = data_sampler.random(n=data_size)
    X_scaled = torch.tensor(
        (X - data_bounds[:, 0]) / (data_bounds[:, 1] - data_bounds[:, 0])
    )

    data_fun = AlpineN2(dimensionality=data_dimensionality, scale_bounds=data_bounds)
    y = data_fun(input_x=X)

    scaler = StandardScaler()
    y_scaled = torch.tensor(scaler.fit_transform(y))

    data_noisy = False
    if data_noisy:
        noise = 0.5 * torch.randn_like(y_scaled)
        y_scaled += noise

    # plt.scatter(X_scaled, y_scaled)
    # plt.show()

    test_idx_list = []
    train_idx_list = []

    n_per_fold = len(X) // folds

    cv = True

    if cv:
        val_idx = np.arange(len(X) - n_per_fold, len(X))

        for fold in range(folds - 1):
            test_idx = np.arange(n_per_fold * fold, n_per_fold * (fold + 1))
            train_idx = np.delete(np.arange(len(X)), np.hstack((test_idx, val_idx)))

            test_idx_list.append(test_idx)
            train_idx_list.append(train_idx)
    else:
        for fold in range(folds):
            test_idx = np.arange(n_per_fold * fold, n_per_fold * (fold + 1))
            train_idx = np.delete(np.arange(len(X)), test_idx)

            test_idx_list.append(test_idx)
            train_idx_list.append(train_idx)

    X_scaled_plot = torch.linspace(0, 1, 100)

    kernel_class_list = [MaternKernel, RBFKernel, RQKernel]

    for model_noisy in [True, False]:
        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(7, 6), sharex=True)
        legend_plotted = False

        for kernel_no, kernel_class in enumerate(kernel_class_list):
            print()
            mse_min = torch.inf
            model_best = None
            train_idx_best = None
            test_idx_best = None
            for fold, (train_idx, test_idx) in enumerate(
                zip(train_idx_list, test_idx_list)
            ):
                if fold != 0:
                    continue
                model, likelihood = train(
                    X_scaled=X_scaled[train_idx],
                    y_scaled=y_scaled[train_idx],
                    # X_scaled=X_scaled,
                    # y_scaled=y_scaled,
                    kernel_class=kernel_class,
                    uniform=False,
                    training_iter=100,
                    noisy=model_noisy,
                    # random_restart=False,
                )

                # Get into evaluation (predictive posterior) mode
                model.eval()
                likelihood.eval()
                # Make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(X_scaled[test_idx]))
                    # observed_pred = likelihood(model(X_scaled))

                mse = torch.mean(
                    (observed_pred.mean - y_scaled[test_idx].flatten()) ** 2
                )
                # mse = torch.mean((observed_pred.mean - y_scaled.flatten()) ** 2)

                if mse < mse_min:
                    print(kernel_class.__name__, fold, mse.item())
                    model_best = model
                    train_idx_best = train_idx
                    test_idx_best = test_idx
                    mse_min = mse.item()

                    if cv:
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            observed_pred = likelihood(model(X_scaled[val_idx]))
                            # observed_pred = likelihood(model(X_scaled))

                        mse_val = torch.mean(
                            (observed_pred.mean - y_scaled[val_idx].flatten()) ** 2
                        )
                        print(kernel_class.__name__, fold, "(val)", mse_val.item())
            for (
                name,
                parameter,
                constraint,
            ) in model_best.named_parameters_and_constraints():
                print(
                    name.rsplit("raw")[-1][1:], constraint.transform(parameter).tolist()
                )

            model_best.eval()
            if cv:
                # X_scaled_val = X_scaled[val_idx]
                # y_scaled_val = y_scaled[val_idx]
                X_scaled_val = X_scaled
                y_scaled_val = y_scaled
            else:
                X_scaled_val = X_scaled
                y_scaled_val = y_scaled

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model_best(X_scaled_val))
                # observed_pred = likelihood(model_best(X_scaled[test_idx]))
                # observed_pred = likelihood(model_best(X_scaled[val_idx]))
                # observed_pred = likelihood(model_best(X_scaled[train_idx_best]))

                observed_pred_plot = likelihood(model_best(X_scaled_plot))

            mse_val = torch.mean((observed_pred.mean - y_scaled_val.flatten()) ** 2)
            # mse_val = torch.mean((observed_pred.mean - y_scaled[test_idx].flatten()) ** 2)
            # mse_val = torch.mean((observed_pred.mean - y_scaled[val_idx].flatten()) ** 2)
            # mse_val = torch.mean(
            #     (observed_pred.mean - y_scaled[train_idx_best].flatten()) ** 2
            # )
            print(
                kernel_class.__name__,
                ["noiseless", "noisy"][model_noisy],
                "total val",
                mse_val.item(),
            )

            # Plot predictions
            with torch.no_grad():
                ax = axs[kernel_no]
                ax.plot(
                    X_scaled_plot,
                    scaler.transform(data_fun(input_x=X_scaled_plot[:, None])),
                    label="Objective",
                    color="black",
                    linestyle="--",
                )

                ax.scatter(
                    X_scaled[train_idx_best],
                    y_scaled[train_idx_best],
                    label="Training data",
                )

                ax.scatter(
                    X_scaled[test_idx_best],
                    y_scaled[test_idx_best],
                    label="Test data",
                    marker="*",
                )

                p = ax.plot(X_scaled_plot, observed_pred_plot.mean, label="Pred. mean")
                color = p[0].get_color()
                ax.fill_between(
                    X_scaled_plot.flatten(),
                    observed_pred_plot.mean - 2 * observed_pred_plot.stddev,
                    observed_pred_plot.mean + 2 * observed_pred_plot.stddev,
                    alpha=0.2,
                    label="Pred. 95% CI",
                    color=color,
                )
                ax.set_ylabel("y")
                if not legend_plotted:
                    ax.legend(loc="upper left", fontsize="x-small", ncols=5)
                    legend_plotted = True
                ax.set_title(f"{kernel_class.__name__}")
                if kernel_no == 2:
                    ax.set_xlabel("x")
        fig.tight_layout()

        fig.savefig(
            f"C:\\Users\\guol\\Documents\\Misc\\PhD\\package_stress\\img\\AlpineN2_{['noiseless', 'noisy'][model_noisy]}.svg"
        )

    # for fold, (train_idx, test_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    #     mse = torch.mean(y_scaled[test_idx].flatten() ** 2)
    #     print("Naive", fold, mse.item())

    plt.show()

    pass

else:
    pass
