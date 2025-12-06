"""
lumin0_real.py

Mock "real" tension engine for Lumin0.

This module simulates a particle-spring relaxation engine with damping that acts as
a plausible 'real' physical backend to compare FLOP counts against the simple
baseline/tension optimizers.

Public API:
    run_real_relax(n=16, steps=200, tension=0.1, lr=0.05, damping=0.9, seed=None) -> dict

Returned dict contains:
    - particles, steps, tension_strength
    - baseline_energy, baseline_flops
    - tension_energy, tension_flops
    - flop_savings
Notes:
 - Final cost (energy) is computed after stopping FLOP accounting so FLOP totals
   reflect algorithmic work only.
 - FLOP accounting uses GLOBAL_FLOPS from flop_counter.py
"""

import numpy as np
from typing import Dict, Any
from flop_counter import GLOBAL_FLOPS as FLOPS


def _pairwise_spring_forces(positions: np.ndarray, k: float = 1.0) -> np.ndarray:
    """
    Compute nearest-neighbor spring forces for a 1D chain of particles in ND space.
    Heuristic FLOP cost added inside.
    """
    n, d = positions.shape
    forces = np.zeros_like(positions)

    # looped implementation (explicit) so FLOP accounting is straightforward
    for i in range(n - 1):
        # displacement vector
        disp = positions[i + 1] - positions[i]
        # force = k * disp
        f = k * disp
        forces[i] += f
        forces[i + 1] -= f
        # approximate FLOP cost: sub (d), mul (d), add/sub (2*d) -> ~4*d per edge
        FLOPS.add(4 * d)
    return forces


def _compute_energy(positions: np.ndarray) -> float:
    """
    Compute simple spring energy sum over adjacent pairs: 0.5 * k * ||dx||^2.
    We will NOT count this computation toward FLOPs in normal runs (final evaluation only).
    """
    n = positions.shape[0]
    s = 0.0
    for i in range(n - 1):
        dx = positions[i + 1] - positions[i]
        s += 0.5 * np.dot(dx, dx)
    return float(s)


def _baseline_relax_step(pos: np.ndarray, lr: float = 0.05) -> np.ndarray:
    """
    One baseline relaxation step: simple explicit update using spring forces.
    Counts FLOP heuristics for force computation and update.
    """
    forces = _pairwise_spring_forces(pos, k=1.0)  # counts FLOPs inside
    # update: pos += lr * forces  -> mul & add per element
    FLOPS.add(pos.size * 2)
    return pos + lr * forces


def _tension_relax_step(pos: np.ndarray, tension: float, lr: float = 0.05, damping: float = 0.95) -> np.ndarray:
    """
    One tension-augmented relaxation step:
      - baseline spring forces
      - plus coupling term toward centroid scaled by 'tension'
      - optional damping on velocities approximated by a simple shrink (heuristic)
    """
    forces = _pairwise_spring_forces(pos, k=1.0)  # costs counted inside
    n, d = pos.shape

    # compute centroid once (cost ~ n*d)
    centroid = np.mean(pos, axis=0)
    FLOPS.add(n * d)  # mean computation heuristic

    # coupling = tension * (centroid - pos)
    # cost: subtract + mul per element -> ~2 * n * d
    FLOPS.add(2 * n * d)
    coupling = tension * (centroid - pos)

    # update: pos += lr * (forces + coupling)
    FLOPS.add(pos.size * 3)  # add forces+coupling, mul by lr, add to pos
    new_pos = pos + lr * (forces + coupling)

    # damping (heuristic cost)
    FLOPS.add(pos.size)
    new_pos = new_pos * damping

    return new_pos


def run_real_relax(n: int = 16,
                   steps: int = 200,
                   tension: float = 0.1,
                   lr: float = 0.05,
                   damping: float = 0.95,
                   seed: int = None) -> Dict[str, Any]:
    """
    Run a paired baseline vs tension relaxation using the same initial condition.
    Returns a dict describing energies and FLOP counts.
    """

    if seed is not None:
        np.random.seed(seed)

    # initial particle positions in 2D (n x 2)
    pos0 = np.random.randn(n, 2)

    # ----------------- baseline run -----------------
    FLOPS.reset()
    pos = pos0.copy()
    for _ in range(steps):
        pos = _baseline_relax_step(pos, lr=lr)
    baseline_energy = _compute_energy(pos)  # NOT counted toward FLOPS
    baseline_flops = FLOPS.snapshot()

    # ----------------- tension run -----------------
    FLOPS.reset()
    pos = pos0.copy()
    for _ in range(steps):
        pos = _tension_relax_step(pos, tension=tension, lr=lr, damping=damping)
    tension_energy = _compute_energy(pos)
    tension_flops = FLOPS.snapshot()

    return {
        "particles": int(n),
        "steps": int(steps),
        "tension_strength": float(tension),
        "baseline_energy": float(baseline_energy),
        "baseline_flops": int(baseline_flops),
        "tension_energy": float(tension_energy),
        "tension_flops": int(tension_flops),
        "flop_savings": int(baseline_flops - tension_flops),
    }


# Small CLI for quick manual testing
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Run mock 'real' tension engine (particle relaxation).")
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tension", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out = run_real_relax(n=args.particles, steps=args.steps, tension=args.tension, lr=args.lr, damping=args.damping, seed=args.seed)
    print(json.dumps(out, indent=2))
