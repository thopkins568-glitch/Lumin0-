"""
lumin0_core.py

Lumin0 — Core engine for "tension field" optimization experiments.

This file provides:
 - function benchmarks (rastrigin, rosenbrock) with embedded FLOP heuristics
 - consistent use of GLOBAL_FLOPS from flop_counter.py
 - baseline and tension solvers (single evaluation per candidate)
 - run_experiment(...) returning ExperimentResult
 - small CLI demo when run directly
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional
import numpy as np

# Use the shared global FLOP counter used across the repo
from flop_counter import GLOBAL_FLOPS as FLOPS


# ---------------------------
# Benchmark functions
# ---------------------------
def rastrigin(x: np.ndarray) -> float:
    """Rastrigin test function (lower is better). Adds coarse FLOP estimate."""
    n = x.size
    # heuristic: a few ops per dimension (mul, add, cos, etc.)
    FLOPS.add(n * 6)
    A = 10.0
    return A * n + float(np.sum(x * x - A * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock test function (lower is better). Adds coarse FLOP estimate."""
    n = x.size
    FLOPS.add(n * 10)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


FUNCTIONS = {
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}


# ---------------------------
# Result dataclass
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


# ---------------------------
# Internal helper functions
# ---------------------------
def evaluate_population_mean(pop: np.ndarray, func: Callable[[np.ndarray], float]) -> float:
    """Return objective value of the population mean (and count FLOPs via func)."""
    mean_point = np.mean(pop, axis=0)
    # account for computing the mean (heuristic: n*d ops)
    FLOPS.add(pop.size)
    return float(func(mean_point))


# ---------------------------
# Optimizer step implementations
# ---------------------------
def baseline_step(pop: np.ndarray, func: Callable[[np.ndarray], float], step_size: float = 0.1) -> np.ndarray:
    """
    One baseline iteration: for each particle, propose a random delta and greedily accept.
    Counts a small heuristic FLOP cost for RNG + apply.
    """
    new_pop = pop.copy()
    n, d = pop.shape

    for i in range(n):
        # RNG & apply (heuristic)
        FLOPS.add(d * 3)
        delta = np.random.normal(loc=0.0, scale=step_size, size=(d,))

        # evaluate current once
        cur_val = func(pop[i])

        candidate = pop[i] + delta
        cand_val = func(candidate)

        # bookkeeping / comparisons
        FLOPS.add(5)
        if cand_val < cur_val:
            new_pop[i] = candidate

    return new_pop


def tension_step(pop: np.ndarray, func: Callable[[np.ndarray], float], tension: float = 0.1, step_size: float = 0.1) -> np.ndarray:
    """
    One tension iteration: same baseline step plus a coupling term toward population mean.
    """
    new_pop = pop.copy()
    n, d = pop.shape

    # compute center once (heuristic)
    center = np.mean(pop, axis=0)
    FLOPS.add(n * d * 2)

    for i in range(n):
        # RNG & apply (heuristic)
        FLOPS.add(d * 3)
        delta = np.random.normal(loc=0.0, scale=step_size, size=(d,))

        # coupling cost (heuristic)
        FLOPS.add(d * 2)
        coupling = tension * (center - pop[i])

        # evaluate current and candidate once each
        cur_val = func(pop[i])
        candidate = pop[i] + delta + coupling
        cand_val = func(candidate)

        FLOPS.add(5)
        if cand_val < cur_val:
            new_pop[i] = candidate

    return new_pop


# ---------------------------
# High-level runners
# ---------------------------
def run_experiment(
    func_name: str = "rastrigin",
    dim: int = 8,
    pop_size: int = 32,
    steps: int = 100,
    tension_strength: float = 0.1,
    seed: Optional[int] = None,
) -> ExperimentResult:
    """
    Run baseline and tension experiments and return an ExperimentResult.

    Notes:
    - Uses the shared GLOBAL_FLOPS (FLOPS) across runs.
    - Each run resets FLOPS at the start to measure that run's work.
    - The returned 'baseline_path' and 'tension_path' track the objective of
      the population mean at each step (useful for plotting).
    """
    if seed is not None:
        np.random.seed(seed)

    if func_name not in FUNCTIONS:
        raise ValueError(f"Unsupported func_name: {func_name}. Choose from {list(FUNCTIONS.keys())}")

    func = FUNCTIONS[func_name]

    # initial population
    pop0 = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))

    t0 = time.time()

    # Baseline run
    FLOPS.reset()
    baseline_pop = pop0.copy()
    baseline_path: List[float] = []
    for _ in range(steps):
        baseline_pop = baseline_step(baseline_pop, func)
        baseline_path.append(evaluate_population_mean(baseline_pop, func))
    baseline_flops = FLOPS.snapshot()
    baseline_best = min(float(func(ind)) for ind in baseline_pop)

    # Tension run (start from same initial seeds/pop)
    FLOPS.reset()
    tension_pop = pop0.copy()
    tension_path: List[float] = []
    for _ in range(steps):
        tension_pop = tension_step(tension_pop, func, tension=tension_strength)
        tension_path.append(evaluate_population_mean(tension_pop, func))
    tension_flops = FLOPS.snapshot()
    tension_best = min(float(func(ind)) for ind in tension_pop)

    wall = time.time() - t0

    return ExperimentResult(
        problem=func_name,
        dim=dim,
        pop_size=pop_size,
        steps=steps,
        tension_strength=tension_strength,
        baseline_best=baseline_best,
        baseline_flops=int(baseline_flops),
        tension_best=tension_best,
        tension_flops=int(tension_flops),
        baseline_path=baseline_path,
        tension_path=tension_path,
        wall_time_s=float(wall),
    )


# ---------------------------
# CLI demo
# ---------------------------
if __name__ == "__main__":
    import argparse, pprint

    parser = argparse.ArgumentParser(description="Run Lumin0 core experiment (baseline vs tension).")
    parser.add_argument("--problem", type=str, default="rastrigin", choices=list(FUNCTIONS.keys()))
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tension", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    res = run_experiment(
        func_name=args.problem,
        dim=args.dim,
        pop_size=args.pop,
        steps=args.steps,
        tension_strength=args.tension,
        seed=args.seed,
    )
    pprint.pprint(res.to_dict())
