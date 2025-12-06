"""
lumin0_tsp.py
Tension-based TSP benchmark for Lumin0.
"""

import numpy as np
from lumin0_utils import compute_distance_matrix, path_length, random_points
from lumin0_core import FLOPS

def baseline_tsp_step(order: np.ndarray, dist: np.ndarray):
    """
    Simple 2-opt baseline step.
    """
    n = len(order)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    if i == j:
        return order

    new_order = order.copy()
    new_order[i:j] = new_order[i:j][::-1]

    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)

    FLOPS.add(10)

    if new_cost < old_cost:
        return new_order
    return order

def tension_tsp_step(order: np.ndarray, dist: np.ndarray, tension=0.1):
    """
    Idea:
    - Compute "center of mass" ordering (mean of adjacency structure)
    - Apply 2-opt move influenced by difference from baseline
    """
    n = len(order)

    # synthetic "center" ordering = sorted order for mild bias
    center = np.arange(n)
    deviation = np.abs(order - center)

    i = np.random.randint(0, n)
    j = min(n - 1, i + int(1 + tension * np.mean(deviation)))

    new_order = order.copy()
    new_order[i:j] = new_order[i:j][::-1]

    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)

    FLOPS.add(20)

    return new_order if new_cost < old_cost else order

def run_tsp(cities=32, steps=500, tension=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)

    pts = random_points(cities, dim=2)
    dist = compute_distance_matrix(pts)

    # Baseline
    order0 = np.arange(cities)
    baseline = order0.copy()
    for _ in range(steps):
        baseline = baseline_tsp_step(baseline, dist)
    baseline_cost = path_length(baseline, dist)
    baseline_flops = FLOPS.snapshot()

    # Tension
    FLOPS.reset()
    tension_order = order0.copy()
    for _ in range(steps):
        tension_order = tension_tsp_step(tension_order, dist, tension)
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
        "flop_savings": int(baseline_flops - tension_flops),
    }
