import copy
import functools

import gpytorch
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
        kernel_class: gpytorch.kernels.RBFKernel,
        uniform: bool = True,
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


def train(
    X_scaled,
    y_scaled,
    kernel_class=gpytorch.kernels.LinearKernel,
    training_iter=150,
    uniform: bool = True,
    noisy: bool = True,
    random_restart: bool = True,
):
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
            X_scaled,
            y_scaled.flatten(),
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
            output = model(X_scaled)

            model.double()

            # Calc loss and backprop gradients
            loss = -mll(output, y_scaled.flatten().double())

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

    return model_min_loss_rr, likelihood_min_loss_rr
