# lumin0_utils.py
"""
Common utilities with corrected FLOP accounting.

FLOP model notes:
 - Pairwise Euclidean distances for n points in d dims:
   per pair: d subtracts + d squares + (d-1) adds (sum) + 1 sqrt
   total approx: n^2 * (3*d - 1)
 - path_length for m edges: m additions (we count arithmetic additions only)
"""
import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS
from typing import Tuple

def compute_distance_matrix(points: np.ndarray) -> np.ndarray:
    n, d = points.shape
    # FLOP estimate: n^2 * (3*d - 1)
    FLOPS.add(int(n * n * (3 * d - 1)))
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def random_points(n: int, dim: int = 2, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.random.uniform(low, high, size=(n, dim))

def path_length(order: np.ndarray, dist: np.ndarray) -> float:
    m = order.size
    # One addition per edge (sum). We won't count array lookups as FLOPs.
    FLOPS.add(int(m))
    total = 0.0
    for i in range(m):
        j = (i + 1) % m
        total += dist[order[i], order[j]]
    return float(total)
