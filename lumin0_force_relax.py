# lumin0_force_relax.py
"""
Particle-spring chain relaxation with optional tension coupling.
Convergence based on change in total chain energy.
"""
import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS

def compute_forces(pos: np.ndarray, k: float = 1.0) -> np.ndarray:
    n = pos.shape[0]
    forces = np.zeros_like(pos)
    for i in range(n - 1):
        diff = pos[i+1] - pos[i]
        forces[i] += k * diff
        forces[i+1] -= k * diff
        FLOPS.add(20)  # heuristic per spring
    return forces

def baseline_force_step(pos: np.ndarray, lr: float = 0.1, k: float = 1.0) -> np.ndarray:
    return pos + lr * compute_forces(pos, k=k)

def tension_force_step(pos: np.ndarray, tension: float = 0.1, lr: float = 0.1, k: float = 1.0) -> np.ndarray:
    center = pos.mean(axis=0)
    coupling = tension * (center - pos)
    FLOPS.add(pos.size * 2)
    return pos + lr * compute_forces(pos, k=k) + coupling

def total_chain_energy(pos: np.ndarray) -> float:
    diffs = pos[1:] - pos[:-1]
    # one square and one add per component approximate
    FLOPS.add(int(diffs.size * 2))
    return float(np.sum(diffs * diffs))

def run_force_relax(n: int = 16, steps: int = 200, tension: float = 0.1, tol: float = 0.0, patience: int = 5, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    pos0 = np.random.randn(n, 2)

    # baseline
    pos = pos0.copy()
    FLOPS.reset()
    best_energy = float("inf")
    stagnant = 0
    iters = 0
    for _ in range(steps):
        pos = baseline_force_step(pos)
        iters += 1
        energy = total_chain_energy(pos)
        if energy < best_energy - tol:
            best_energy = energy
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    baseline_flops = FLOPS.snapshot()
    baseline_energy = float(total_chain_energy(pos))
    baseline_conv = (stagnant >= max(1, patience))

    # tension
    pos = pos0.copy()
    FLOPS.reset()
    best_energy = float("inf")
    stagnant = 0
    t_iters = 0
    for _ in range(steps):
        pos = tension_force_step(pos, tension=tension)
        t_iters += 1
        energy = total_chain_energy(pos)
        if energy < best_energy - tol:
            best_energy = energy
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    tension_flops = FLOPS.snapshot()
    tension_energy = float(total_chain_energy(pos))
    tension_conv = (stagnant >= max(1, patience))

    return {
        "problem": "force_relax",
        "params": {"particles": n, "steps": steps, "tension": tension, "tol": tol, "patience": patience, "seed": seed},
        "baseline": {"energy": baseline_energy, "flops": int(baseline_flops), "iters": iters, "converged": baseline_conv},
        "tension": {"energy": tension_energy, "flops": int(tension_flops), "iters": t_iters, "converged": tension_conv},
        "flop_savings": int(baseline_flops - tension_flops),
    }
