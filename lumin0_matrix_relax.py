"""
lumin0_matrix_relax.py
Simple matrix relaxation test: solve A*x ≈ b using relaxation + tension.
"""

import numpy as np
from lumin0_core import FLOPS

def baseline_matrix_step(A, x, b, lr=0.01):
    grad = A.T @ (A @ x - b)
    FLOPS.add(A.size * 2)
    return x - lr * grad

def tension_matrix_step(A, x, b, tension=0.1, lr=0.01):
    grad = A.T @ (A @ x - b)
    FLOPS.add(A.size * 2)
    center = np.zeros_like(x)
    coupling = tension * (center - x)
    return x - lr * grad + coupling

def run_matrix_relax(n=64, steps=200, tension=0.05, seed=None):
    if seed:
        np.random.seed(seed)

    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x0 = np.zeros(n)

    # baseline
    x = x0.copy()
    for _ in range(steps):
        x = baseline_matrix_step(A, x, b)
    baseline_err = float(np.linalg.norm(A @ x - b))
    baseline_flops = FLOPS.snapshot()

    # tension
    FLOPS.reset()
    x = x0.copy()
    for _ in range(steps):
        x = tension_matrix_step(A, x, b, tension)
    tension_err = float(np.linalg.norm(A @ x - b))
    tension_flops = FLOPS.snapshot()

    return {
        "dim": n,
        "steps": steps,
        "baseline_err": baseline_err,
        "baseline_flops": baseline_flops,
        "tension_err": tension_err,
        "tension_flops": tension_flops,
        "flop_savings": baseline_flops - tension_flops
    }
