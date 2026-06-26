import time

import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel

from sklearn.preprocessing import StandardScaler

from gp import train

MaternKernel, RBFKernel, RQKernel

torch.manual_seed(1)
np.random.seed(1)

if __name__ == "__main__":
    df_data = pd.read_csv("datasets/solder_ball_conc.csv", header=0, index_col=0)

    print(df_data.corr())

    x_vars = [
        "d_pad",
        "t_pad",
        "d_us",
        "d_rep1",
        "t_ubm",
        "del_d",
        "h_ball",
    ]

    g = sns.pairplot(
        df_data,
        x_vars=x_vars,
        y_vars=["max_conc"],
        height=1.5,
        aspect=0.75,
        plot_kws={"s": 10},
    )

    g.set(ylabel="Max. conc.")
    x_labels = [
        "$d_{pad}$",
        "$t_{pad}$",
        "$d_{ubm}$",
        "$d_{rep1}$",
        "$t_{ubm}$",
        "$d_{del}$",
        "$h_{ball}$",
    ]
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_xlabel(x_labels[i])

    # g.savefig("img/solderball_data.svg")

    # g.savefig(
    #     "C:\\Users\\leoli\\Documents\\GitHub\\Dissertation-draft\\img\\applications\\solderball\\solderball_data.svg"
    # )

    # plt.show()

    names = [
        "d_pad",
        "t_pad",
        "d_us",
        "d_rep1",
        "t_ubm",
        "del_d",
        "h_ball",
    ]

    X = df_data[names].values

    X_bounds = np.array(
        [
            [150.0, 180.0],
            [15.0, 30.0],
            [180.0, 220.0],
            [30.0, 160.0],
            [5.0, 15.0],
            [10.0, 40.0],
            [130.0, 165.0],
        ]
    )
    X_scaled = torch.tensor((X - X_bounds[:, 0]) / (X_bounds[:, 1] - X_bounds[:, 0]))

    # y = np.sum(X**2, 1)[:, None]
    y = df_data["max_conc"].values[:, None]
    scaler = StandardScaler()
    y_scaled = torch.tensor(scaler.fit_transform(y))

    # model, likelihood = train(X_scaled=X_scaled, y_scaled=y_scaled, training_iter=500)

    # mse_naive = torch.mean(y_scaled.flatten() ** 2)
    # print("Naive", mse_naive.item())

    test_idx_list = []
    train_idx_list = []

    folds = 10
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

    kernel_class_list = [
        # MaternKernel,
        # RBFKernel,
        RQKernel,
    ]

    for noisy in [
        # True,
        False,
    ]:
        start_total = time.time()

        for kernel_class in kernel_class_list:
            print()
            mse_min = torch.inf
            model_best = None
            train_idx_best = None

            val_error_list = []
            test_error_list = []
            total_error_list = []
            rpd_noise_list = []
            for fold, (train_idx, test_idx) in enumerate(
                zip(train_idx_list, test_idx_list)
            ):
                start_individual = time.time()
                model, likelihood = train(
                    X_scaled=X_scaled[train_idx],
                    y_scaled=y_scaled[train_idx],
                    # X_scaled=X_scaled,
                    # y_scaled=y_scaled,
                    kernel_class=kernel_class,
                    uniform=False,
                    training_iter=100,
                    noisy=noisy,
                    # random_restart=False,
                )
                end_individual = time.time()
                print("Individual time:", end_individual - start_individual)

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
                    model_best = model
                    likelihood_best = likelihood
                    train_idx_best = train_idx
                    mse_min = mse.item()
                print(kernel_class.__name__, fold, mse.item())
                test_error_list.append(mse.item())

                if cv:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        observed_pred = likelihood(model(X_scaled[val_idx]))
                        # observed_pred = likelihood(model(X_scaled))

                    mse_val = torch.mean(
                        (observed_pred.mean - y_scaled[val_idx].flatten()) ** 2
                    )
                    print(kernel_class.__name__, fold, "(val)", mse_val.item())

                    val_error_list.append(mse_val.item())

                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        observed_pred = likelihood(model(X_scaled))

                    mse_total = torch.mean(
                        (observed_pred.mean - y_scaled.flatten()) ** 2
                    )
                    print(kernel_class.__name__, "(total)", mse_total.item())
                    total_error_list.append(mse_total.item())

                    for (
                        name,
                        parameter,
                        constraint,
                    ) in model.named_parameters_and_constraints():
                        # print(
                        #     name.rsplit("raw")[-1][1:],
                        #     constraint.transform(parameter).tolist(),
                        # )
                        if "noise" in name:
                            rpd_noise_val = constraint.transform(parameter).item()
                    rpd_noise_list.append(rpd_noise_val)

            visualize_best_model = True
            if visualize_best_model:
                x_grid = torch.linspace(0, 1, 50)
                y_grid = torch.linspace(0, 1, 50)
                xx, yy = torch.meshgrid(x_grid, y_grid)
                grid_list = torch.stack((xx, yy)).T.reshape(-1, 2)

                # for const_param_value in [
                #     # 0.25,
                #     0.5,
                #     # 0.75,
                # ]:
                #     fig, axs = plt.subplots(ncols=7, nrows=7, figsize=(12, 12))

                #     for i in range(7):
                #         for j in range(7):
                #             if i >= j:
                #                 continue

                #             constant_parameters = torch.tile(
                #                 const_param_value * torch.ones((1, 7)),
                #                 dims=(50 * 50, 1),
                #             )
                #             constant_parameters[:, i] = grid_list[:, 0]
                #             constant_parameters[:, j] = grid_list[:, 1]

                #             model_best.eval()
                #             likelihood_best.eval()
                #             # Make predictions by feeding model through likelihood
                #             with torch.no_grad(), gpytorch.settings.fast_pred_var():
                #                 observed_pred = likelihood_best(
                #                     model_best(constant_parameters)
                #                 )

                #             # observed_pred.mean.reshape(50, 50).numpy()

                #             ax = axs[i, j]
                #             ax.contourf(
                #                 xx.numpy(),
                #                 yy.numpy(),
                #                 observed_pred.mean.reshape(50, 50).numpy(),
                #                 cmap="viridis",
                #                 levels=100,
                #             )
                #             ax.set_xlabel(names[j])
                #             ax.set_ylabel(names[i])
                #     fig.suptitle(
                #         f"{kernel_class.__name__}, {const_param_value}, {['noiseless', 'noisy'][int(noisy)]}"
                #     )
                #     fig.tight_layout()

                constant_parameters = torch.tile(
                    0.5 * torch.ones((1, 7)), dims=(50 * 50, 1)
                )
                constant_parameters[:, 3] = grid_list[:, 0]
                constant_parameters[:, 4] = grid_list[:, 1]

                model_best.eval()
                likelihood_best.eval()
                # Make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood_best(model_best(constant_parameters))

                # observed_pred.mean.reshape(50, 50).numpy()

                # for (
                #     name,
                #     parameter,
                #     constraint,
                # ) in model_best.named_parameters_and_constraints():
                #     print(
                #         name.rsplit("raw")[-1][1:],
                #         constraint.transform(parameter).tolist(),
                #     )

                xx_plot = X_bounds[4, 0] + xx * (X_bounds[4, 1] - X_bounds[4, 0])
                yy_plot = X_bounds[3, 0] + yy * (X_bounds[3, 1] - X_bounds[3, 0])

                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.plot_surface(
                    xx_plot.numpy(),
                    yy_plot.numpy(),
                    scaler.inverse_transform(observed_pred.mean[:, None]).reshape(
                        50, 50
                    ),
                    cmap="viridis",
                )
                # ax.set_xlabel(f"${names[4]}$")
                ax.set_xlabel(r"$t_{ubm}$")
                # ax.set_ylabel(f"${names[3]}$")
                ax.set_ylabel(r"$d_{rep1}$")
                ax.set_zlabel("Max. conc.")
                ax.view_init(20, 45)
                ax.invert_xaxis()
                fig.tight_layout()
                fig.savefig(
                    f"img/solderball_{kernel_class.__name__}_{['noiseless', 'noisy'][int(noisy)]}.svg"
                )

                # plt.show()

            df = pd.DataFrame(
                {
                    "val": test_error_list,
                    "test": val_error_list,
                    "total": total_error_list,
                    "rpd_noise": rpd_noise_list,
                }
            )
            df.to_csv(
                f"datasets/val_test_total_rpdnoise_{kernel_class.__name__}_{['noiseless', 'noisy'][int(noisy)]}.csv"
            )

            # for (
            #     name,
            #     parameter,
            #     constraint,
            # ) in model_best.named_parameters_and_constraints():
            #     if "noise" in name:
            #         print(
            #             name.rsplit("raw")[-1][1:], constraint.transform(parameter).tolist()
            #         )

            # model_best.eval()
            # if cv:
            #     # X_scaled_val = X_scaled[val_idx]
            #     # y_scaled_val = y_scaled[val_idx]
            #     X_scaled_val = X_scaled
            #     y_scaled_val = y_scaled
            # else:
            #     X_scaled_val = X_scaled
            #     y_scaled_val = y_scaled

            # with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #     observed_pred = likelihood(model_best(X_scaled_val))
            #     # observed_pred = likelihood(model_best(X_scaled[test_idx]))
            #     # observed_pred = likelihood(model_best(X_scaled[val_idx]))
            #     # observed_pred = likelihood(model_best(X_scaled[train_idx_best]))

            # mse_val = torch.mean((observed_pred.mean - y_scaled_val.flatten()) ** 2)
            # # mse_val = torch.mean((observed_pred.mean - y_scaled[test_idx].flatten()) ** 2)
            # # mse_val = torch.mean((observed_pred.mean - y_scaled[val_idx].flatten()) ** 2)
            # # mse_val = torch.mean(
            # #     (observed_pred.mean - y_scaled[train_idx_best].flatten()) ** 2
            # # )
            # print(kernel_class.__name__, "val", mse_val.item())

        end_total = time.time()
        print("Total time:", end_total - start_total)

    # for fold, (train_idx, test_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    #     mse = torch.mean(y_scaled[test_idx].flatten() ** 2)
    #     print("Naive", fold, mse.item())

    plt.show()
    pass

else:
    pass
