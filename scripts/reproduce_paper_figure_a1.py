from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np

from simulation.experiment import ForwardExperiment1D
import utility.circuits as circuits
from utility.circuits import CircuitGen1DA


RESULT = Path(__file__).resolve().parent
REPO = RESULT.parents[1]
configs = json.loads((REPO / "examples" / "20240403T212351" / "configs.json").read_text())
config = dict(configs["1"])
config.pop("idx", None)
solver_name = config.pop("solver")
for name in ("mu", "rho", "u", "v"):
    config[name] = np.asarray(config[name], dtype=float)

experiment = ForwardExperiment1D(verbose=2, data_folder=str(RESULT / "figure2_runs"))
experiment.add_solver(solver_name, **config)
solver = experiment.solvers[0]

# Figure 2 only depicts a representative circuit; no sampling or tomography
# execution is required. Generate the first non-zero time and first observable.
circuits.SIMPLE_CIRCUITS = True
generator = CircuitGen1DA(experiment.logger, backend=None)
groups = generator.tomography_circuits(
    solver.st.get_state(0), solver.tf.h, solver.times[1:2],
    config["backend"]["synthesis"], 100000,
    config["backend"]["optimization"], config["backend"]["seed"], False,
)
circuit = groups[0][0]
figure = circuit.draw(output="mpl", fold=-1)
figure.suptitle("Quantum circuit for the 1-D elastic wave simulation")
figure.tight_layout()
output = RESULT / "paper_figure2_reproduced.png"
figure.savefig(output, dpi=300, bbox_inches="tight")

summary = (
    f"qubits={circuit.num_qubits}\n"
    f"classical_bits={circuit.num_clbits}\n"
    f"depth={circuit.depth()}\n"
    f"operations={dict(circuit.count_ops())}\n"
    f"time={solver.times[1]}\n"
    f"observable={'Z' * circuit.num_qubits}\n"
)
(RESULT / "paper_figure2_summary.txt").write_text(summary, encoding="utf-8")
print(summary)
