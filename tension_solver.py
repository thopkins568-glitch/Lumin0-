"""
tension_solver.py
Runs tension-enabled versions of all Lumin0 solvers.
"""

from lumin0_core import Lumin0Core
from lumin0_force_relax import run_force_relax
from lumin0_matrix_relax import run_matrix_relax
from lumin0_tsp import run_tsp
from lumin0_real import Lumin0Real


def run_tension_all(tension=0.1):
    """
    Returns dict with tension-enabled results across all solvers.
    """

    core = Lumin0Core()

    return {
        "force_relax": run_force_relax(n=16, steps=200, tension=tension),
        "matrix_relax": run_matrix_relax(n=64, steps=200, tension=tension),
        "tsp": run_tsp(cities=32, steps=500, tension=tension),
        "real": Lumin0Real(dim=64, tension=tension).run(steps=200, use_tension=True),
        "core_meta": core.describe(),
    }
