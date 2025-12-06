"""
lumin0_utils.py
Common utilities for Lumin0 benchmark modules.
"""

import numpy as np
from lumin0_core import FLOPS


# ---------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------
def compute_distance_matrix(points: np.ndarray):
    """
    Compute full pairwise Euclidean distance matrix.
    """
    n, d = points.shape

    # FLOP model: (subtract, square, sum, sqrt) per pair per dim
    # approx = n^2 * d * 4
    FLOPS.add(int(n * n * d * 4))

    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ---------------------------------------------------------
# Random point generator
# ---------------------------------------------------------
def random_points(n: int, dim: int = 2, low: float = 0.0, high: float = 1.0):
    return np.random.uniform(low, high, size=(n, dim))


# ---------------------------------------------------------
# Path length calculator
# ---------------------------------------------------------
def path_length(order: np.ndarray, dist: np.ndarray):
    """
    Compute length of a TSP tour given a distance matrix.
    """
    m = order.size

    # FLOP model: one lookup + add per edge
    FLOPS.add(int(m * 3))

    total = 0.0
    for i in range(m):
        j = (i + 1) % m
        total += dist[order[i], order[j]]

    return total
