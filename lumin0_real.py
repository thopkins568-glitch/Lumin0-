"""
lumin0_real.py
Physical-style tension solver mock used for FLOP comparison.
Implements a generic relaxation loop with optional tension pull.
"""

import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS


class Lumin0Real:
    """
    Simple mock tension engine that:
      - accepts an initial state vector x
      - computes relaxation steps using a generic gradient (Aᵀ(Ax - b))
      - injects a tunable global tension pull
    """

    def __init__(self, dim=64, tension=0.1, seed=None):
        self.dim = dim
        self.tension = tension

        if seed is not None:
            np.random.seed(seed)

        # random linear system
        self.A = np.random.randn(dim, dim)
        self.b = np.random.randn(dim)
        self.x = np.zeros(dim)

    # ---------------------------------------------------------
    # One baseline update: gradient descent
    # ---------------------------------------------------------
    def baseline_step(self, lr=0.01):
        grad = self.A.T @ (self.A @ self.x - self.b)
        FLOPS.add(self.A.size * 2)
        self.x -= lr * grad
        return self.x

    # ---------------------------------------------------------
    # One tension update: gradient + coupling
    # ---------------------------------------------------------
    def tension_step(self, lr=0.01):
        grad = self.A.T @ (self.A @ self.x - self.b)
        FLOPS.add(self.A.size * 2)

        coupling = -self.tension * self.x
        FLOPS.add(self.x.size)

        self.x -= lr * grad
        self.x += coupling
        return self.x

    # ---------------------------------------------------------
    # Run a full sequence of steps
    # ---------------------------------------------------------
    def run(self, steps=200, use_tension=False):
        FLOPS.reset()
        self.x = np.zeros(self.dim)

        for _ in range(steps):
            if use_tension:
                self.tension_step()
            else:
                self.baseline_step()

        err = float(np.linalg.norm(self.A @ self.x - self.b))
        flops = FLOPS.snapshot()

        return err, flops
