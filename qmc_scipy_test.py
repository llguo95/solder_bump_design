from scipy.stats import qmc
import matplotlib.pyplot as plt
import numpy as np


for sampler_class in [qmc.LatinHypercube, qmc.Sobol, qmc.Halton]:
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(8, 8), sharex=True, sharey=True)
    for seed in range(9):
        rng = np.random.default_rng(seed=seed)

        sampler = sampler_class(d=2, rng=rng)
        sample = sampler.random(n=64)

        ax = axs[seed % 3, seed // 3]

        ax.scatter(sample[:, 0], sample[:, 1], s=10)
        ax.scatter(sample[:16, 0], sample[:16, 1])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        if seed % 3 == 2:
            ax.set_xlabel("x0")
        if seed // 3 == 0:
            ax.set_ylabel("x1")
        fig.suptitle(f"{sampler_class.__name__} sample")
        fig.tight_layout()
plt.show()
