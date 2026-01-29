__author__ = "Nicolas Chatzikiriakos"
__contact__ = "nicolas.chatzikiriakos@ist.uni-stuttgart.de"
__date__ = "26/01/21"

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
plt.close("all")

data_path = Path("./data")

# ------------------------------------------------------------------
# Specify which experiment to load
# 1: main Example, 2: toy example, 3: random set
# ------------------------------------------------------------------
id = 3

if id == 1:
    data = np.load(data_path / "mainExample_comp.npz")
    nRho = 5
    labels = [
        r"Alg. 1 with $\rho_k \equiv 0$",
        r"Alg. 1 using $\theta_*$ for planning",
        "Wagenmaker Approach",
        "Optimal oracle excitation",
        "Isotropic Random",
    ]
    styles = ["-", "--", "-", "--", "--"]

elif id == 2:
    data = np.load(data_path / "exp_toyExample.npz")
    nRho = 3
    labels = ["Algo with CE", "Algo with Oracle", "Isotropic Gaussian"]
    styles = ["-", "--", "-"]
    
elif id == 3:
    data = np.load(data_path / "mainExample_randomSet.npz")
    nRho = 6
    labels = [
        r"$\rho_k=\frac{1}{1+k}$",
        r"$\rho_k=\frac{1}{(1+k)^2}$",
        "Alg with CE",
        "Wagenmaker with oracle",
        "Alg with Oracle",
        "Gaussian",
    ]
    styles = ["-", "-", "--", "-", "--", "--"]

else:
    raise ValueError("No valid identifier provided. Select id ∈ {1, 2, 3}")

# ------------------------------------------------------------------
# Extract belief trajectories
# ------------------------------------------------------------------
belief_traj = data["belief_traj"]

truePositive = []
for jRho in range(nRho):
    tp = np.squeeze(belief_traj[0, :, :, jRho])
    truePositive.append(tp)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
colors = plt.colormaps["tab10"](np.linspace(0, 1, nRho))

plt.figure()
plt.grid(True)

for i in range(nRho):
    tensor = truePositive[i]  # shape: [episodes, simulations]

    mean_vals = np.mean(tensor, axis=1)
    dev = np.std(tensor, axis=1)

    lower = mean_vals - 0.5 * dev
    upper = mean_vals + 0.5 * dev

    x = np.arange(tensor.shape[0])

    plt.plot(
        x,
        mean_vals,
        linestyle=styles[i],
        color=colors[i],
        linewidth=2,
        label=labels[i],
    )

    plt.fill_between(
        x,
        lower,
        upper,
        color=colors[i],
        alpha=0.1,
    )


plt.xlabel("Epoch")
plt.ylabel(r"$P[\theta_k = \theta_*]$")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
