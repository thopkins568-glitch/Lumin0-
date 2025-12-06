"""
lumin0_tsp.py
TSP baseline vs tension-modified benchmark for Lumin0 FLOP comparison.

This module provides:
- A baseline randomized 2-opt local search.
- A tension-modified 2-opt variant introducing a mild synthetic global constraint.
- A benchmark runner that compares FLOPs between the two.

End-of-run cost measurements are intentionally NOT counted toward FLOPs,
ensuring FLOP comparison reflects *only* the optimization work, not the final evaluation.
"""

import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS
from lumin0_utils import compute_distance_matrix, path_length, random_points


# ---------------------------------------------------------
# Baseline: simple randomized 2-opt
# ---------------------------------------------------------
def baseline_tsp_step(order: np.ndarray, dist: np.ndarray):
    n = len(order)

    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    # ensure a valid non-empty slice
    if i == j:
        return order
    if i > j:
        i, j = j, i

    new_order = order.copy()
    new_order[i:j] = new_order[i:j][::-1]

    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)

    # FLOP cost of this local step (heuristic estimate)
    FLOPS.add(10)

    return new_order if new_cost < old_cost else order


# ---------------------------------------------------------
# Tension-modified 2-opt
# ---------------------------------------------------------
def tension_tsp_step(order: np.ndarray, dist: np.ndarray, tension: float = 0.1):
    n = len(order)

    # synthetic "center" ordering structure
    # this is intentionally simple—representing an abstract pull toward
    # a globally consistent structure
    center = np.arange(n)
    deviation = np.abs(order - center)

    # pick segment start
    i = np.random.randint(0, n)

    # compute segment length based on deviation
    shift = int(1 + tension * np.mean(deviation))
    j = min(n - 1, i + shift)

    # sanity: ensure valid slice
    if i == j:
        return order
    if i > j:
        i, j = j, i

    new_order = order.copy()
    new_order[i:j] = new_order[i:j][::-1]

    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)

    # tension computation / overhead FLOPs
    FLOPS.add(20)

    return new_order if new_cost < old_cost else order


# ---------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------
def run_tsp(cities: int = 32, steps: int = 500, tension: float = 0.1, seed=None):
    """
    Returns a dict:
        {
            cities,
            steps,
            tension_strength,
            baseline_cost,
            baseline_flops,
            tension_cost,
            tension_flops,
            flop_savings
        }

    Final path costs are evaluated AFTER the FLOP counters stop,
    to avoid unfair inflation of FLOP totals.
    """
    if seed is not None:
        np.random.seed(seed)

    # Prepare problem instance
    pts = random_points(cities, dim=2)
    dist = compute_distance_matrix(pts)

    # ---------------- Baseline ----------------
    FLOPS.reset()
    order0 = np.arange(cities)
    baseline = order0.copy()

    for _ in range(steps):
        baseline = baseline_tsp_step(baseline, dist)

    baseline_flops = FLOPS.snapshot()
    baseline_cost = float(path_length(baseline, dist))   # not counted

    # ---------------- Tension Variant ----------------
    FLOPS.reset()
    tension_order = order0.copy()

    for _ in range(steps):
        tension_order = tension_tsp_step(tension_order, dist, tension=tension)

    tension_flops = FLOPS.snapshot()
    tension_cost = float(path_length(tension_order, dist))  # not counted

    return {
        "cities": cities,
        "steps": steps,
        "tension_strength": tension,
        "baseline_cost": baseline_cost,
        "baseline_flops": int(baseline_flops),
        "tension_cost": tension_cost,
        "tension_flops": int(tension_flops),
        "flop_savings": int(baseline_flops - tension_flops)
                     }
