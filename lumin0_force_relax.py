"""
lumin0_force_relax.py
Minimal particle–spring relaxation benchmark with optional tension field.
Uses shared GLOBAL_FLOPS counter from flop_counter.py.
"""

import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS


# -------------------------------------------------
# FORCE COMPUTATION
# -------------------------------------------------
def compute_forces(pos, k=1.0):
    """
    Simple 1D spring chain: force between each pair (i, i+1).
    pos: (n, d)
    """
    n = pos.shape[0]
    forces = np.zeros_like(pos)

    for i in range(n - 1):
        diff = pos[i + 1] - pos[i]
        forces[i] += k * diff
        forces[i + 1] -= k * diff
        FLOPS.add(20)   # force computation cost

    return forces


# -------------------------------------------------
# BASELINE UPDATE
# -------------------------------------------------
def baseline_force_step(pos, lr=0.1, k=1.0):
    return pos + lr * compute_forces(pos, k=k)


# -------------------------------------------------
# TENSION UPDATE
# -------------------------------------------------
def tension_force_step(pos, tension=0.1, lr=0.1, k=1.0):
    center = pos.mean(axis=0)
    coupling = tension * (center - pos)

    FLOPS.add(pos.size * 2)  # tension overhead

    return pos + lr * compute_forces(pos, k=k) + coupling


# -------------------------------------------------
# MAIN SIMULATION
# -------------------------------------------------
def run_force_relax(n=16, steps=200, tension=0.1, seed=None, k=1.0):
    """
    Returns dict with:
      - baseline_energy
      - baseline_flops
      - tension_energy
      - tension_flops
      - flop_savings
    """

    if seed is not None:
        np.random.seed(seed)

    # initial particle positions (2D)
    pos0 = np.random.randn(n, 2)

    # --------------------------
    # BASELINE RUN
    # --------------------------
    pos = pos0.copy()
    FLOPS.reset()

    for _ in range(steps):
        pos = baseline_force_step(pos, lr=0.1, k=k)

    baseline_energy = float(np.sum((pos[1:] - pos[:-1]) ** 2))
    baseline_flops = FLOPS.snapshot()

    # --------------------------
    # TENSION RUN
    # --------------------------
    pos = pos0.copy()
    FLOPS.reset()

    for _ in range(steps):
        pos = tension_force_step(pos, tension=tension, lr=0.1, k=k)

    tension_energy = float(np.sum((pos[1:] - pos[:-1]) ** 2))
    tension_flops = FLOPS.snapshot()

    # --------------------------
    # RESULT
    # --------------------------
    return {
        "particles": n,
        "steps": steps,
        "baseline_energy": baseline_energy,
        "baseline_flops": baseline_flops,
        "tension_energy": tension_energy,
        "tension_flops": tension_flops,
        "flop_savings": baseline_flops - tension_flops,
    }
