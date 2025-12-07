# lumin0_matrix_relax.py
"""
Matrix relaxation for solving A x ≈ b via simple gradient relaxation,
with optional tension coupling toward origin. Convergence on residual norm.
"""
import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS

def baseline_matrix_step(A: np.ndarray, x: np.ndarray, b: np.ndarray, lr: float = 0.01) -> np.ndarray:
    # grad = A^T (A x - b)
    # FLOPs for Ax: n^2 mult-add ~ 2*n^2; for A^T @ r: another 2*n^2 -> approx 4*n^2
    n = A.shape[0]
    FLOPS.add(int(4 * (n**2)))
    grad = A.T @ (A @ x - b)
    return x - lr * grad

def tension_matrix_step(A: np.ndarray, x: np.ndarray, b: np.ndarray, tension: float = 0.1, lr: float = 0.01) -> np.ndarray:
    n = A.shape[0]
    FLOPS.add(int(4 * (n**2)))
    grad = A.T @ (A @ x - b)
    coupling = -tension * x
    FLOPS.add(x.size)  # coupling ops
    return x - lr * grad + coupling

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r = A @ x - b
    # FLOPs for Ax: ~2*n^2 (we'll count modestly)
    FLOPS.add(int(2 * (A.shape[0]**2)))
    return float(np.linalg.norm(r))

def run_matrix_relax(n: int = 64, steps: int = 200, tension: float = 0.05, tol: float = 0.0, patience: int = 5, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    x0 = np.zeros(n)

    # baseline
    x = x0.copy()
    FLOPS.reset()
    best_res = float("inf")
    stagnant = 0
    iters = 0
    for _ in range(steps):
        x = baseline_matrix_step(A, x, b)
        iters += 1
        res = residual_norm(A, x, b)
        if res < best_res - tol:
            best_res = res
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    baseline_flops = FLOPS.snapshot()
    baseline_err = float(residual_norm(A, x, b))
    baseline_conv = (stagnant >= max(1, patience))

    # tension
    x = x0.copy()
    FLOPS.reset()
    best_res = float("inf")
    stagnant = 0
    t_iters = 0
    for _ in range(steps):
        x = tension_matrix_step(A, x, b, tension=tension)
        t_iters += 1
        res = residual_norm(A, x, b)
        if res < best_res - tol:
            best_res = res
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    tension_flops = FLOPS.snapshot()
    tension_err = float(residual_norm(A, x, b))
    tension_conv = (stagnant >= max(1, patience))

    return {
        "problem": "matrix_relax",
        "params": {"dim": n, "steps": steps, "tension": tension, "tol": tol, "patience": patience, "seed": seed},
        "baseline": {"residual": baseline_err, "flops": int(baseline_flops), "iters": iters, "converged": baseline_conv},
        "tension": {"residual": tension_err, "flops": int(tension_flops), "iters": t_iters, "converged": tension_conv},
        "flop_savings": int(baseline_flops - tension_flops),
    }
