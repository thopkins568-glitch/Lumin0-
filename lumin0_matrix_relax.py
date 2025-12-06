"""
lumin0_matrix_relax.py
Matrix relaxation benchmark: solve A x ≈ b using gradient relaxation
with optional tension coupling toward the origin.
"""

import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS


# -------------------------------------------------
# BASELINE STEP
# -------------------------------------------------
def baseline_matrix_step(A, x, b, lr=0.01):
    """
    Standard relaxation:
        grad = Aᵀ(Ax - b)
        x ← x - lr * grad
    """
    grad = A.T @ (A @ x - b)
    FLOPS.add(A.size * 2)  # gradient FLOP cost approximation
    return x - lr * grad


# -------------------------------------------------
# TENSION STEP
# -------------------------------------------------
def tension_matrix_step(A, x, b, tension=0.1, lr=0.01):
    """
    Same as baseline but adds coupling toward origin:
        coupling = tension * (0 - x)
    """
    grad = A.T @ (A @ x - b)
    FLOPS.add(A.size * 2)

    coupling = -tension * x
    FLOPS.add(x.size)  # tension cost

    return x - lr * grad + coupling


# -------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------
def run_matrix_relax(n=64, steps=200, tension=0.05, seed=None):
    """
    Runs baseline vs tension solver and returns:
       - baseline_err
       - baseline_flops
       - tension_err
       - tension_flops
       - flop_savings
    """

    if seed is not None:
        np.random.seed(seed)

    # Construct system A x ≈ b
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x0 = np.zeros(n)

    # --------------------------
    # BASELINE RUN
    # --------------------------
    x = x0.copy()
    FLOPS.reset()

    for _ in range(steps):
        x = baseline_matrix_step(A, x, b)

    baseline_err = float(np.linalg.norm(A @ x - b))
    baseline_flops = FLOPS.snapshot()

    # --------------------------
    # TENSION RUN
    # --------------------------
    x = x0.copy()
    FLOPS.reset()

    for _ in range(steps):
        x = tension_matrix_step(A, x, b, tension=tension)

    tension_err = float(np.linalg.norm(A @ x - b))
    tension_flops = FLOPS.snapshot()

    # --------------------------
    # RESULT
    # --------------------------
    return {
        "dim": n,
        "steps": steps,
        "baseline_err": baseline_err,
        "baseline_flops": baseline_flops,
        "tension_err": tension_err,
        "tension_flops": tension_flops,
        "flop_savings": baseline_flops - tension_flops,
    }
