from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULT = Path(__file__).resolve().parent
REPO = RESULT.parents[1]
SOURCE = REPO / "examples" / "20231203T101131"
data = [pickle.load(open(SOURCE / f"data_{i}.pkl", "rb")) for i in range(3)]

times = np.asarray(data[0]["times"])
settings = data[0]["settings"]
nx = settings["nx"]
x = np.arange(nx + 2) * settings["dx"]
indices = [0, 2, 11, 15, 17]
labels = ["ODE solver", "Ideal quantum simulator (1000 shots)", "IBM Brisbane (1000 shots)"]
colors = ["black", "red", "blue"]

fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
rho = np.asarray(settings["rho"])
mu = np.asarray(settings["mu"])[:nx]
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(x, np.r_[rho[0], rho, rho[-1]], color="blue", label=r"$\rho$")
ax2.plot(x, np.r_[mu[0], mu, mu[-1]], color="red", label=r"$\mu$")
ax.set_xlabel("x [m]")
ax.set_ylabel(r"$\rho$ [kg/m$^3$]", color="blue")
ax2.set_ylabel(r"$\mu$ [Pa]", color="red")
ax.set_title("a.)")
ax.legend(loc="lower right")
ax2.legend(loc="lower right", bbox_to_anchor=(1, 0.17))

for panel, ti in enumerate(indices, start=1):
    ax = axes.flat[panel]
    for solver, (label, color) in enumerate(zip(labels, colors)):
        u = np.asarray(data[solver]["field"]["u"])
        field = np.r_[0.0, u[ti], 0.0]
        if solver == 0:
            ax.plot(x, field, "o-", color=color, markersize=4, label=label)
        else:
            ax.plot(x, field, color=color, linewidth=1.6, label=label)
    ax.set_title(f"{'bcdef'[panel-1]}.)  t = {times[ti]:.4f} s")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"u [$\mu$m]")
    ax.set_ylim(-1.05, 1.05)
    if panel == 1:
        ax.legend(fontsize=7, loc="lower left")

fig.suptitle("Paper Figure 2 reproduced from the archived IBM Brisbane run")
fig.savefig(RESULT / "paper_true_figure2_reproduced.png", dpi=300)
plt.close(fig)

reference = np.asarray(data[0]["field"]["u"])
lines = ["source=examples/20231203T101131"]
for solver, name in ((1, "ideal_1000"), (2, "ibm_brisbane_1000")):
    value = np.asarray(data[solver]["field"]["u"])
    error = np.linalg.norm(value-reference, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-30)
    selected = error[indices]
    lines.append(f"{name},mean_all={error.mean():.8g},max_all={error.max():.8g},selected={selected.tolist()}")
(RESULT / "paper_true_figure2_summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))
