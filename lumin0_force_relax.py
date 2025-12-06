"""
lumin0_force_relax.py
Consistent particle–spring relaxation benchmark with explicit FLOP accounting.
Compatible with updated Lumen0 architecture.
"""

import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS


# -------------------------------------------------
# UTILITIES
# -------------------------------------------------

def pairwise_energy(pos):
    """
    Computes Σ (pos[i+1] - pos[i])² with FLOP accounting.
    """
    diffs = pos[1:] - pos[:-1]     # n-1 vector subtractions
    FLOPS.add(diffs.size)          # 1 FLOP per subtraction

    sq = diffs * diffs             # elementwise square
    FLOPS.add(diffs.size)          # 1 FLOP per multiply

    energy = float(np.sum(sq))
    FLOPS.add(sq.size)             # sum is n-1 adds

    return energy


# -------------------------------------------------
# FORCE COMPUTATION
# -------------------------------------------------

def compute_forces(pos, k=1.0):
    """
    Explicit spring forces with FLOP accounting.
    pos: (n, d)
    """
    n, d = pos.shape
    forces = np.zeros_like(pos)

    # For each spring, add +k*diff to i and -k*diff to i+1.
    for i in range(n - 1):
        diff = pos[i + 1] - pos[i]      # d subtractions
        FLOPS.add(d)                    # per-dimension subtracts

        scaled = k * diff               # d multiplies
        FLOPS.add(d)

        forces[i] += scaled             # d adds
        forces[i + 1] -= scaled         # d subtracts
        FLOPS.add(2 * d)

    return forces


# -------------------------------------------------
# SOLVER CLASSES
# -------------------------------------------------

class ForceRelaxBaseline:
    def __init__(self, lr=0.1, k=1.0):
        self.lr = lr
        self.k = k

    def step(self, pos):
        forces = compute_forces(pos, k=self.k)
        # pos + lr * forces
        FLOPS.add(pos.size)  # multiply
        FLOPS.add(pos.size)  # add
        return pos + self.lr * forces


class ForceRelaxTension:
    def __init__(self, lr=0.1, tension=0.1, k=1.0):
        self.lr = lr
        self.k = k
        self.tension = tension

    def step(self, pos):
        forces = compute_forces(pos, k=self.k)

        # coupling = tension * (center - pos)
        center = np.mean(pos, axis=0)
        # mean: n adds and 1 divide, per dimension
        n, d = pos.shape
        FLOPS.add((n * d))     # adds
        FLOPS.add(d)           # divides

        diff = center - pos
        FLOPS.add(pos.size)    # subtracts

        coupling = self.tension * diff
        FLOPS.add(pos.size)    # multiplies

        # total update pos + lr*forces + coupling
        FLOPS.add(pos.size)    # lr*forces multiply
        FLOPS.add(pos.size)    # pos + lr*forces
        FLOPS.add(pos.size)    # add coupling

        return pos + self.lr * forces + coupling


# -------------------------------------------------
# MAIN SIMULATION API
# -------------------------------------------------

def run_force_relax(n=16, steps=200, tension=0.1, seed=None, k=1.0):
    """
    Returns dict:
        - baseline_energy
        - baseline_flops
        - tension_energy
        - tension_flops
        - flop_savings
    """

    if seed is not None:
        np.random.seed(seed)

    pos0 = np.random.randn(n, 2)

    # --------------------------
    # BASELINE
    # --------------------------
    baseline = ForceRelaxBaseline(lr=0.1, k=k)
    pos = pos0.copy()
    FLOPS.reset()

    for _ in range(steps):
        pos = baseline.step(pos)

    baseline_energy = pairwise_energy(pos)
    baseline_flops = FLOPS.snapshot()

    # --------------------------
    # TENSION
    # --------------------------
    tension_solver = ForceRelaxTension(lr=0.1, tension=tension, k=k)
    pos = pos0.copy()
    FLOPS.reset()

    for _ in range(steps):
        pos = tension_solver.step(pos)

    tension_energy = pairwise_energy(pos)
    tension_flops = FLOPS.snapshot()

    return {
        "particles": n,
        "steps": steps,
        "baseline_energy": baseline_energy,
        "baseline_flops": baseline_flops,
        "tension_energy": tension_energy,
        "tension_flops": tension_flops,
        "flop_savings": baseline_flops - tension_flops,
    }
