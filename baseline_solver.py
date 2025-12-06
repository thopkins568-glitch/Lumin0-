"""
baseline_solver.py
Simple orchestrator to run baseline solvers for comparison.
"""

from lumin0_core import Lumin0Core
from lumin0_force_relax import run_force_relax
from lumin0_matrix_relax import run_matrix_relax
from lumin0_tsp import run_tsp
from lumin0_real import Lumin0Real


def run_baseline_all():
    """
    Returns a dict with all baseline-only results for:
      - particle relaxation
      - matrix relaxation
      - TSP 2-opt
      - real-model mock solver
    """

    core = Lumin0Core()

    return {
        "force_relax": run_force_relax(n=16, steps=200, tension=0.0),
        "matrix_relax": run_matrix_relax(n=64, steps=200, tension=0.0),
        "tsp": run_tsp(cities=32, steps=500, tension=0.0),
        "real": Lumin0Real(dim=64, tension=0.0).run(steps=200, use_tension=False),
        "core_meta": core.describe(),
    }
