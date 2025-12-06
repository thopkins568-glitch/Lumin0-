"""
lumin0_matrix_relax.py
Consistent linear system relaxation benchmark with explicit FLOP accounting.
Matches updated Lumen0 architecture and FLOP model.

We solve A x ≈ b via gradient relaxation:
    grad = Aᵀ(Ax - b)
    x ← x - lr * grad        (baseline)
    x ← x - lr * grad + coupling   (tension)
"""

import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS


# -------------------------------------------------
# FLOP-Accounted Linear Algebra
# -------------------------------------------------

def matvec(A, x):
    """
    Computes A @ x with FLOP accounting:
    For an (n×n) matrix:
        n * n multiplies + n * (n-1) adds
    """
    n = A.shape[0]
    FLOPS.add(n * n)          # multiplies
    FLOPS.add(n * (n - 1))    # adds
    return A @ x


def vec_add(a, b):
    """
    Vector addition with FLOP accounting.
    """
    FLOPS.add(a.size)
    return a + b


def vec_sub(a, b):
    """
    Vector subtraction with FLOP accounting.
    """
    FLOPS.add(a.size)
    return a - b


def scalar_vec_mul(c, v):
    """
    c * v with FLOP accounting.
    """
    FLOPS.add(v.size)
    return c * v


def l2_norm(v):
    """
    ||v||₂ with FLOP accounting.
    """
    # Square: size multiplies
    FLOPS.add(v.size)
    sq = v * v

    # Sum: size adds
    FLOPS.add(v.size - 1)
    s = float(np.sum(sq))

    # sqrt: ~1 FLOP (treat as constant)
    FLOPS.add(1)

    return np.sqrt(s)


# -------------------------------------------------
# GRADIENT COMPUTATION
# -------------------------------------------------

def compute_gradient(A, x, b):
    """
    grad = Aᵀ (A x - b)
    with full FLOP accounting.
    """
    Ax = matvec(A, x)
    r = vec_sub(Ax, b)        # residual
    grad = matvec(A.T, r)
    return grad


# -------------------------------------------------
# SOLVER CLASSES
# -------------------------------------------------

class MatrixRelaxBaseline:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, A, x, b):
        grad = compute_gradient(A, x, b)
        update = scalar_vec_mul(self.lr, grad)
        return vec_sub(x, update)


class MatrixRelaxTension:
    def __init__(self, lr=0.01, tension=0.1):
        self.lr = lr
        self.tension = tension

    def step(self, A, x, b):
        grad = compute_gradient(A, x, b)

        # -lr * grad
        grad_term = scalar_vec_mul(self.lr, grad)
        x_new = vec_sub(x, grad_term)

        # coupling = -tension * x
        coupling = scalar_vec_mul(-self.tension, x)

        # x_new + coupling
        return vec_add(x_new, coupling)


# -------------------------------------------------
# MAIN DRIVER
# -------------------------------------------------

def run_matrix_relax(n=64, steps=200, tension=0.05, seed=None):
    """
    Runs baseline vs tension solver.
    Returns:
        - baseline_err
        - baseline_flops
        - tension_err
        - tension_flops
        - flop_savings
    """

    if seed is not None:
        np.random.seed(seed)

    # Random system A x ≈ b
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x0 = np.zeros(n)

    # --------------------------
    # BASELINE RUN
    # --------------------------
    solver_base = MatrixRelaxBaseline(lr=0.01)
    x = x0.copy()
    FLOPS.reset()

    for _ in range(steps):
        x = solver_base.step(A, x, b)

    baseline_err = l2_norm(matvec(A, x) - b)
    baseline_flops = FLOPS.snapshot()

    # --------------------------
    # TENSION RUN
    # --------------------------
    solver_tens = MatrixRelaxTension(lr=0.01, tension=tension)
    x = x0.copy()
    FLOPS.reset()

    for _ in range(steps):
        x = solver_tens.step(A, x, b)

    tension_err = l2_norm(matvec(A, x) - b)
    tension_flops = FLOPS.snapshot()

    return {
        "dim": n,
        "steps": steps,
        "baseline_err": baseline_err,
        "baseline_flops": baseline_flops,
        "tension_err": tension_err,
        "tension_flops": tension_flops,
        "flop_savings": baseline_flops - tension_flops,
    }
