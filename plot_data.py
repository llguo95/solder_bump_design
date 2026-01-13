import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gpytorch
import torch
import matplotlib.pyplot as plt

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def train(X_scaled, y_scaled):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 1e-3))
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_scaled, y_scaled.flatten(), likelihood)
    training_iter = 150

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_scaled)
        # Calc loss and backprop gradients
        loss = -mll(output, y_scaled.flatten())
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

        # print(list(model.named_parameters()))
    
    return model, likelihood

df_data = pd.read_csv("datasets/Kriging_data.csv", header=[0, 1])

X = df_data[["t_lf", "t_solder"]].values

X_bounds = np.array([[.2, 1.], [.02, .08]])
X_scaled = torch.tensor((X - X_bounds[:, 0]) / (X_bounds[:, 1] - X_bounds[:, 0]))

y = df_data[('MidDieStress', 'top2')].values[:, None]
scaler = StandardScaler()
y_scaled = torch.tensor(scaler.fit_transform(y))

model, likelihood = train(X_scaled=X_scaled, y_scaled=y_scaled)

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
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y.flatten())

ax.set_xlabel('t_lf')
ax.set_ylabel('t_solder')
ax.set_zlabel('MidDieStress')

# ax.plot_surface(xx, yy, observed_pred.mean.reshape(50, 50).detach().numpy(), cmap='viridis', alpha=0.5)
ax.plot_surface(
    xx * (X_bounds[0, 1] - X_bounds[0, 0]) + X_bounds[0, 0],
    yy * (X_bounds[1, 1] - X_bounds[1, 0]) + X_bounds[1, 0], 
    scaler.inverse_transform(observed_pred.mean.detach().numpy()[:, None]).reshape(50, 50), cmap='viridis', alpha=0.5)

plt.show()