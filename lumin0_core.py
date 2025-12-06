"""
lumin0_core.py

Lumin0 — Core engine for "tension field" optimization experiments.

This module implements:
 - FLOP counting (coarse-grained, heuristic)
 - Two test problems: Rastrigin and Rosenbrock
 - Baseline optimizer (random local search)
 - Tension-augmented optimizer (baseline + coupling toward population mean)
 - Experiment runner that returns structured results and optionally writes JSON

Dependencies: numpy (only)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Tuple
import numpy as np

# ---------------------------
# FLOP counter (coarse)
# ---------------------------
class FlopCounter:
    def __init__(self):
        self.count = 0

    def add(self, n: int):
        self.count += int(n)

    def reset(self):
        self.count = 0

    def snapshot(self) -> int:
        return int(self.count)

# global flop counter instance used inside functions
FLOPS = FlopCounter()

# ---------------------------
# Test functions
# ---------------------------
def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function. Lower is better. Adds coarse FLOP estimate."""
    A = 10.0
    n = x.size
    # cost estimate: a few ops per dimension
    FLOPS.add(n * 6)
    return A * n + np.sum(x * x - A * np.cos(2.0 * np.pi * x))

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function. Lower is better. Adds coarse FLOP estimate."""
    n = x.size
    FLOPS.add(n * 10)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

TEST_FUNCTIONS = {"rastrigin": rastrigin, "rosenbrock": rosenbrock}

# ---------------------------
# Optimizer steps
# ---------------------------
def baseline_step(pop: np.ndarray, func: Callable[[np.ndarray], float], step_size: float = 0.1) -> np.ndarray:
    """
    One iteration of a simple baseline: each particle proposes a random local move;
    if it improves its own objective, accept it.
    """
    new_pop = pop.copy()
    n, d = pop.shape
    for i in range(n):
        delta = np.random.normal(loc=0.0, scale=step_size, size=(d,))
        FLOPS.add(d * 3)  # cost for generating and applying delta (heuristic)
        candidate = pop[i] + delta
        # evaluate candidate vs current (each eval estimated)
        cur_val = func(pop[i])
        cand_val = func(candidate)
        FLOPS.add(20)  # bookkeeping / comparisons (heuristic)
        if cand_val < cur_val:
            new_pop[i] = candidate
    return new_pop

def tension_step(pop: np.ndarray, func: Callable[[np.ndarray], float],
                 tension: float = 0.1, step_size: float = 0.1) -> np.ndarray:
    """
    One iteration of tension-augmented optimizer:
    - baseline random step (delta)
    - plus coupling force toward population mean (tension * (mean - x_i))
    """
    new_pop = pop.copy()
    n, d = pop.shape
    # compute center (mean) once
    center = np.mean(pop, axis=0)
    FLOPS.add(n * d * 2)  # mean computation (heuristic)

    for i in range(n):
        delta = np.random.normal(loc=0.0, scale=step_size, size=(d,))
        FLOPS.add(d * 3)
        coupling = tension * (center - pop[i])
        FLOPS.add(d * 2)
        candidate = pop[i] + delta + coupling
        cur_val = func(pop[i])
        cand_val = func(candidate)
        FLOPS.add(20)
        if cand_val < cur_val:
            new_pop[i] = candidate
    return new_pop

# ---------------------------
# Experiment wrapper
# ---------------------------
@dataclass
class ExperimentResult:
    problem: str
    dim: int
    pop_size: int
    steps: int
    tension_strength: float
    baseline_best: float
    baseline_flops: int
    tension_best: float
    tension_flops: int
    baseline_path: List[float]
    tension_path: List[float]
    wall_time_s: float

    def to_dict(self):
        return asdict(self)

def run_experiment(func_name: str = "rastrigin",
                   dim: int = 8,
                   pop_size: int = 32,
                   steps: int = 100,
                   tension_strength: float = 0.1,
                   seed: int = None,
                   save_json: bool = False,
                   out_path: str = "lumin0_experiment_result.json") -> ExperimentResult:
    """
    Run a pair of experiments:
      - baseline (no tension)
      - tension-augmented (coupling strength = tension_strength)

    Returns an ExperimentResult dataclass. Optionally saves JSON summary.
    """
    if seed is not None:
        np.random.seed(seed)

    if func_name not in TEST_FUNCTIONS:
        raise ValueError(f"Unsupported func_name: {func_name}. Choose from {list(TEST_FUNCTIONS.keys())}")

    func = TEST_FUNCTIONS[func_name]

    # initialize population
    pop0 = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))

    # baseline run
    FLOPS.reset()
    baseline_pop = pop0.copy()
    baseline_path: List[float] = []
    t_start = time.time()
    for _ in range(steps):
        baseline_pop = baseline_step(baseline_pop, func)
        # record the objective of the mean position as a simple aggregate
        baseline_path.append(float(func(baseline_pop.mean(axis=0))))
    t_end = time.time()
    baseline_time = t_end - t_start
    baseline_flops = FLOPS.snapshot()
    baseline_best = float(func(baseline_pop.mean(axis=0)))

    # tension run
    FLOPS.reset()
    tension_pop = pop0.copy()
    tension_path: List[float] = []
    t_start = time.time()
    for _ in range(steps):
        tension_pop = tension_step(tension_pop, func, tension=tension_strength)
        tension_path.append(float(func(tension_pop.mean(axis=0))))
    t_end = time.time()
    tension_time = t_end - t_start
    tension_flops = FLOPS.snapshot()
    tension_best = float(func(tension_pop.mean(axis=0)))

    result = ExperimentResult(
        problem=func_name,
        dim=dim,
        pop_size=pop_size,
        steps=steps,
        tension_strength=tension_strength,
        baseline_best=baseline_best,
        baseline_flops=baseline_flops,
        tension_best=tension_best,
        tension_flops=tension_flops,
        baseline_path=baseline_path,
        tension_path=tension_path,
        wall_time_s=float(baseline_time + tension_time),
    )

    if save_json:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result

# ---------------------------
# Small CLI demo when run directly
# ---------------------------
if __name__ == "__main__":
    import argparse, pprint
    parser = argparse.ArgumentParser(description="Run Lumin0 tension-field experiment (baseline vs tension)")
    parser.add_argument("--problem", type=str, default="rastrigin", choices=list(TEST_FUNCTIONS.keys()))
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tension", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="Save JSON result to disk")
    parser.add_argument("--out", type=str, default="lumin0_result.json", help="JSON output path (if --save)")

    args = parser.parse_args()
    res = run_experiment(func_name=args.problem,
                         dim=args.dim,
                         pop_size=args.pop,
                         steps=args.steps,
                         tension_strength=args.tension,
                         seed=args.seed,
                         save_json=args.save,
                         out_path=args.out)
    pprint.pprint(res.to_dict())
