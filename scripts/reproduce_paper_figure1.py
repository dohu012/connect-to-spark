from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.experiment import ForwardExperiment1D


RESULT = Path(__file__).resolve().parent
REPO = RESULT.parents[1]
SOURCE_CONFIG = REPO / "examples" / "20240403T212351" / "configs.json"

configs = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
experiment = ForwardExperiment1D(verbose=2, data_folder=str(RESULT / "figure1_runs"))

for key in sorted(configs, key=int):
    config = dict(configs[key])
    solver = config.pop("solver")
    config.pop("idx", None)
    if solver == "local":
        # Keep all tomography circuits in one Aer job. The repository default
        # creates 24 jobs which each request high internal parallelism and
        # severely oversubscribes the 20-core ARM CPU on DGX Spark.
        config["backend"]["batch_size"] = 100000
    for array_name in ("mu", "rho", "u", "v"):
        config[array_name] = np.asarray(config[array_name], dtype=float)
    experiment.add_solver(solver, **config)

started = time.perf_counter()
data = experiment.run()
elapsed = time.perf_counter() - started

times = np.asarray(data[0]["times"])
settings = data[0]["settings"]
nx = settings["nx"]
x = np.arange(nx + 2)
indices = [0, 2, 11, 15, 17]
labels = ["ODE", "Quantum simulator (20 shots)", "Quantum simulator (1000 shots)"]
colors = ["black", "#ff7f0e", "#d62728"]

fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
rho = np.asarray(settings["rho"])
mu = np.asarray(settings["mu"])
ax = axes[0, 0]
ax2 = ax.twinx()
ax.plot(x, np.r_[rho[0], rho, rho[-1]], color="#1f77b4", label=r"$\rho$")
mu_core = mu[:nx]
ax2.plot(x, np.r_[mu_core[0], mu_core, mu_core[-1]], color="#d62728", label=r"$\mu$")
ax.set_xlabel("Grid coordinate")
ax.set_ylabel(r"Density $\rho$ [kg/m$^3$]", color="#1f77b4")
ax2.set_ylabel(r"Elastic modulus $\mu$ [Pa]", color="#d62728")
ax.set_title("Heterogeneous medium")

for panel, ti in enumerate(indices, start=1):
    ax = axes.flat[panel]
    for solver, (label, color) in enumerate(zip(labels, colors)):
        u = np.asarray(data[solver]["field"]["u"])
        field = np.r_[0.0, u[ti], 0.0]
        if solver == 0:
            ax.scatter(x, field, color=color, s=8, label=label, zorder=3)
        else:
            ax.plot(x, field, color=color, linewidth=1.5, label=label)
    ax.set_title(f"t = {times[ti]:.3f} s")
    ax.set_xlabel("Grid coordinate")
    ax.set_ylabel("Displacement")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.2)
    if panel == 1:
        ax.legend(fontsize=7, loc="lower right")

fig.suptitle("Reproduction of paper Figure 1 on NVIDIA DGX Spark", fontsize=15)
fig.savefig(RESULT / "paper_figure1_reproduced.png", dpi=220)
plt.close(fig)

rows = []
reference = np.asarray(data[0]["field"]["u"])
for solver, shots in ((1, 20), (2, 1000)):
    value = np.asarray(data[solver]["field"]["u"])
    relative = np.linalg.norm(value - reference, axis=1) / np.maximum(
        np.linalg.norm(reference, axis=1), 1e-30
    )
    rows.append((shots, float(relative.mean()), float(relative.max())))

summary = [f"elapsed_seconds={elapsed:.6f}"]
summary.extend(f"shots={shots},mean_time_rl2={mean:.8g},max_time_rl2={maximum:.8g}"
               for shots, mean, maximum in rows)
(RESULT / "paper_figure1_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
print("\n".join(summary))
