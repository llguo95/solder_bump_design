import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gpytorch
import torch
import matplotlib.pyplot as plt

df_data = pd.read_csv("datasets/Kriging_data.csv", header=[0, 1])

X = df_data[["t_lf", "t_solder"]].values

X_bounds = np.array([[.2, 1.], [.02, .08]])
X_scaled = torch.tensor((X - X_bounds[:, 0]) / (X_bounds[:, 1] - X_bounds[:, 0]))

y = np.abs(df_data[('MidDieStress', 'top2')].values[:, None])
scaler = StandardScaler()
y_scaled = torch.tensor(scaler.fit_transform(y))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], y_scaled.flatten())
plt.show()