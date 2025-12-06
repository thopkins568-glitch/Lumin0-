"""
lumin0_tsp.py
TSP baseline vs tension-modified benchmark for Lumin0 FLOP comparison.
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

    if i == j:
        return order

    new_order = order.copy()
    new_order[i:j] = new_order[i:j][::-1]

    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)

    FLOPS.add(10)  # heuristic 2-opt overhead

    return new_order if new_cost < old_cost else order


# ---------------------------------------------------------
# Tension-modified 2-opt
# ---------------------------------------------------------
def tension_tsp_step(order: np.ndarray, dist: np.ndarray, tension: float = 0.1):
    n = len(order)

    # synthetic "center" structure for mild global pull
    center = np.arange(n)
    deviation = np.abs(order - center)

    i = np.random.randint(0, n)
    shift = int(1 + tension * np.mean(deviation))
    j = min(n - 1, i + shift)

    new_order = order.copy()
    new_order[i:j] = new_order[i:j][::-1]

    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)

    FLOPS.add(20)  # tension overhead

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
    """
    if seed is not None:
        np.random.seed(seed)

    # generate problem instance
    pts = random_points(cities, dim=2)
    dist = compute_distance_matrix(pts)

    # ---------------- Baseline ----------------
    FLOPS.reset()
    order0 = np.arange(cities)
    baseline = order0.copy()

    for _ in range(steps):
        baseline = baseline_tsp_step(baseline, dist)

    baseline_cost = path_length(baseline, dist)
    baseline_flops = FLOPS.snapshot()

    # ---------------- Tension ----------------
    FLOPS.reset()
    tension_order = order0.copy()

    for _ in range(steps):
        tension_order = tension_tsp_step(tension_order, dist, tension=tension)

    tension_cost = path_length(tension_order, dist)
    tension_flops = FLOPS.snapshot()

    return {
        "cities": cities,
        "steps": steps,
        "tension_strength": tension,
        "baseline_cost": float(baseline_cost),
        "baseline_flops": int(baseline_flops),
        "tension_cost": float(tension_cost),
        "tension_flops": int(tension_flops),
        "flop_savings": int(baseline_flops - tension_flops)
    }
