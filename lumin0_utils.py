"""
lumin0_utils.py
Common utilities for Lumin0 benchmark modules.
"""

import numpy as np

def compute_distance_matrix(points: np.ndarray):
    """
    Compute full pairwise Euclidean distance matrix.
    FLOP heuristic added (rough).
    """
    n = points.shape[0]
    # approx FLOPs: n^2 * d * 3
    from lumin0_core import FLOPS
    FLOPS.add(int(n * n * points.shape[1] * 3))

    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def random_points(n: int, dim: int = 2, low: float = 0.0, high: float = 1.0):
    """Generate random 2D or ND points."""
    return np.random.uniform(low, high, size=(n, dim))

def path_length(order: np.ndarray, dist: np.ndarray):
    """Compute length of a TSP tour given distance matrix."""
    from lumin0_core import FLOPS
    m = order.size
    FLOPS.add(m * 5)
    total = 0.0
    for i in range(m):
        j = (i + 1) % m
        total += dist[order[i], order[j]]
    return total
