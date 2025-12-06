"""
lumin0_force_relax.py
Particle-spring system with relaxation + tension.
"""

import numpy as np
from lumin0_core import FLOPS

def compute_forces(pos, k=1.0):
    n = pos.shape[0]
    forces = np.zeros_like(pos)
    for i in range(n - 1):
        d = pos[i + 1] - pos[i]
        forces[i] += k * d
        forces[i + 1] -= k * d
        FLOPS.add(20)
    return forces

def baseline_force_step(pos, lr=0.1):
    return pos + lr * compute_forces(pos)

def tension_force_step(pos, tension=0.1, lr=0.1):
    center = pos.mean(axis=0)
    coupling = tension * (center - pos)
    FLOPS.add(pos.size * 2)
    return pos + lr * compute_forces(pos) + coupling

def run_force_relax(n=16, steps=200, tension=0.1, seed=None):
    if seed:
        np.random.seed(seed)

    pos0 = np.random.randn(n, 2)

    # baseline
    pos = pos0.copy()
    for _ in range(steps):
        pos = baseline_force_step(pos)
    baseline_energy = float(np.sum((pos[1:] - pos[:-1]) ** 2))
    baseline_flops = FLOPS.snapshot()

    # tension
    FLOPS.reset()
    pos = pos0.copy()
    for _ in range(steps):
        pos = tension_force_step(pos, tension=tension)
    tension_energy = float(np.sum((pos[1:] - pos[:-1]) ** 2))
    tension_flops = FLOPS.snapshot()

    return {
        "particles": n,
        "steps": steps,
        "baseline_energy": baseline_energy,
        "baseline_flops": baseline_flops,
        "tension_energy": tension_energy,
        "tension_flops": tension_flops,
        "flop_savings": baseline_flops - tension_flops
    }
