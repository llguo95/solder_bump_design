import gpytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel
from sklearn.preprocessing import StandardScaler

from gp import train

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    df_data = pd.read_csv("datasets/solder_ball_conc.csv", header=0, index_col=0)

    sns.pairplot(df_data)

    X = df_data[
        [
            "d_pad",
            "t_pad",
            "d_us",
            "d_rep1",
            "t_ubm",
            "del_d",
            "h_ball",
        ]
    ].values

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

    cv = False

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

    kernel_class_list = [MaternKernel, RBFKernel, RQKernel]

    for kernel_class in kernel_class_list:
        print()
        mse_min = torch.inf
        model_best = None
        train_idx_best = None
        for fold, (train_idx, test_idx) in enumerate(
            zip(train_idx_list, test_idx_list)
        ):
            model, likelihood = train(
                X_scaled=X_scaled[train_idx],
                y_scaled=y_scaled[train_idx],
                # X_scaled=X_scaled,
                # y_scaled=y_scaled,
                kernel_class=kernel_class,
                uniform=False,
                training_iter=100,
                noisy=False,
                # random_restart=False,
            )

            # Get into evaluation (predictive posterior) mode
            model.eval()
            likelihood.eval()
            # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(X_scaled[test_idx]))
                # observed_pred = likelihood(model(X_scaled))

            mse = torch.mean((observed_pred.mean - y_scaled[test_idx].flatten()) ** 2)
            # mse = torch.mean((observed_pred.mean - y_scaled.flatten()) ** 2)

            if mse < mse_min:
                print(kernel_class.__name__, fold, mse.item())
                model_best = model
                train_idx_best = train_idx
                mse_min = mse.item()

        for (
            name,
            parameter,
            constraint,
        ) in model_best.named_parameters_and_constraints():
            print(name.rsplit("raw")[-1][1:], constraint.transform(parameter).tolist())

        model_best.eval()
        if cv:
            X_scaled_val = X_scaled[val_idx]
            y_scaled_val = y_scaled[val_idx]
            # X_scaled_val = X_scaled
            # y_scaled_val = y_scaled
        else:
            X_scaled_val = X_scaled
            y_scaled_val = y_scaled

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model_best(X_scaled_val))
            # observed_pred = likelihood(model_best(X_scaled[test_idx]))
            # observed_pred = likelihood(model_best(X_scaled[val_idx]))
            # observed_pred = likelihood(model_best(X_scaled[train_idx_best]))

        mse_val = torch.mean((observed_pred.mean - y_scaled_val.flatten()) ** 2)
        # mse_val = torch.mean((observed_pred.mean - y_scaled[test_idx].flatten()) ** 2)
        # mse_val = torch.mean((observed_pred.mean - y_scaled[val_idx].flatten()) ** 2)
        # mse_val = torch.mean(
        #     (observed_pred.mean - y_scaled[train_idx_best].flatten()) ** 2
        # )
        print(kernel_class.__name__, "val", mse_val.item())

    # plt.show()

    # for fold, (train_idx, test_idx) in enumerate(zip(train_idx_list, test_idx_list)):
    #     mse = torch.mean(y_scaled[test_idx].flatten() ** 2)
    #     print("Naive", fold, mse.item())

    pass

else:
    pass
