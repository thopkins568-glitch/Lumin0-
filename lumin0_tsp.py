# lumin0_tsp.py
"""
TSP benchmark (baseline 2-opt vs tension-modified 2-opt).
Uses lumin0_utils for distance + path length FLOP accounting.
"""
import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS
from lumin0_utils import compute_distance_matrix, path_length, random_points

def _two_opt_swap(order: np.ndarray, i: int, j: int) -> np.ndarray:
    new = order.copy()
    new[i:j] = new[i:j][::-1]
    return new

def baseline_tsp_step(order: np.ndarray, dist: np.ndarray):
    n = len(order)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    if i == j:
        return order
    if i > j:
        i, j = j, i
    new_order = _two_opt_swap(order, i, j)
    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)
    FLOPS.add(10)  # heuristic overhead for 2-opt step
    return new_order if new_cost < old_cost else order

def tension_tsp_step(order: np.ndarray, dist: np.ndarray, tension: float = 0.1):
    n = len(order)
    center = np.arange(n)
    deviation = np.abs(order - center)
    i = np.random.randint(0, n)
    shift = int(1 + tension * np.mean(deviation))
    j = min(n - 1, i + shift)
    if i == j:
        return order
    if i > j:
        i, j = j, i
    new_order = _two_opt_swap(order, i, j)
    old_cost = path_length(order, dist)
    new_cost = path_length(new_order, dist)
    FLOPS.add(20)
    return new_order if new_cost < old_cost else order

def run_tsp(cities: int = 32, steps: int = 500, tension: float = 0.1, tol: float = 0.0, patience: int = 5, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    pts = random_points(cities, dim=2)
    dist = compute_distance_matrix(pts)

    # baseline
    FLOPS.reset()
    order0 = np.arange(cities)
    baseline = order0.copy()
    best_cost = float("inf")
    stagnant = 0
    iters = 0
    for _ in range(steps):
        baseline = baseline_tsp_step(baseline, dist)
        iters += 1
        cost = path_length(baseline, dist)
        if cost < best_cost - tol:
            best_cost = cost
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    baseline_flops = FLOPS.snapshot()
    baseline_cost = float(path_length(baseline, dist))
    baseline_converged = (stagnant >= max(1, patience))

    # tension
    FLOPS.reset()
    order = order0.copy()
    best_cost = float("inf")
    stagnant = 0
    t_iters = 0
    for _ in range(steps):
        order = tension_tsp_step(order, dist, tension=tension)
        t_iters += 1
        cost = path_length(order, dist)
        if cost < best_cost - tol:
            best_cost = cost
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            break
    tension_flops = FLOPS.snapshot()
    tension_cost = float(path_length(order, dist))
    tension_converged = (stagnant >= max(1, patience))

    return {
        "problem": "tsp",
        "params": {"cities": cities, "steps": steps, "tension": tension, "tol": tol, "patience": patience, "seed": seed},
        "baseline": {"cost": baseline_cost, "flops": int(baseline_flops), "iters": iters, "converged": baseline_converged},
        "tension": {"cost": tension_cost, "flops": int(tension_flops), "iters": t_iters, "converged": tension_converged},
        "flop_savings": int(baseline_flops - tension_flops),
        }
