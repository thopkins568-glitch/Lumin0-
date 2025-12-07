# lumin0_core.py
"""
Core experiment engine (baseline vs tension) with convergence detection
and unified ExperimentResult dataclass.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional
import numpy as np
from flop_counter import GLOBAL_FLOPS as FLOPS

# Benchmark functions with FLOP annotations
def rastrigin(x: np.ndarray) -> float:
    n = x.size
    FLOPS.add(n * 6)  # heuristic: a few ops per dim
    A = 10.0
    return float(A * n + np.sum(x * x - A * np.cos(2.0 * np.pi * x)))

def rosenbrock(x: np.ndarray) -> float:
    n = x.size
    FLOPS.add(n * 10)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

FUNCTIONS = {"rastrigin": rastrigin, "rosenbrock": rosenbrock}

@dataclass
class ExperimentResult:
    problem: str
    params: dict
    baseline: dict
    tension: dict
    flop_savings: Optional[int]
    wall_time_s: float

    def to_dict(self):
        return asdict(self)

# internal helpers
def evaluate_population_mean(pop: np.ndarray, func: Callable[[np.ndarray], float]) -> float:
    # mean cost: count mean computation heuristically
    FLOPS.add(pop.size)
    mean_pt = np.mean(pop, axis=0)
    return float(func(mean_pt))

def baseline_step(pop: np.ndarray, func: Callable[[np.ndarray], float], step_size: float = 0.1) -> np.ndarray:
    new_pop = pop.copy()
    n, d = pop.shape
    for i in range(n):
        FLOPS.add(d * 3)  # RNG & apply heuristic
        delta = np.random.normal(0.0, step_size, size=(d,))
        cur_val = func(pop[i])
        candidate = pop[i] + delta
        cand_val = func(candidate)
        FLOPS.add(5)  # comparisons/bookkeeping
        if cand_val < cur_val:
            new_pop[i] = candidate
    return new_pop

def tension_step(pop: np.ndarray, func: Callable[[np.ndarray], float], tension: float = 0.1, step_size: float = 0.1) -> np.ndarray:
    new_pop = pop.copy()
    n, d = pop.shape
    center = np.mean(pop, axis=0)
    FLOPS.add(n * d * 2)  # mean cost heuristic
    for i in range(n):
        FLOPS.add(d * 3)  # RNG & apply
        delta = np.random.normal(0.0, step_size, size=(d,))
        FLOPS.add(d * 2)  # coupling ops heur
        coupling = tension * (center - pop[i])
        cur_val = func(pop[i])
        candidate = pop[i] + delta + coupling
        cand_val = func(candidate)
        FLOPS.add(5)
        if cand_val < cur_val:
            new_pop[i] = candidate
    return new_pop

def run_solver_with_convergence(run_step_fn, pop0: np.ndarray, func: Callable[[np.ndarray], float],
                                steps: int, tol: float, patience: int, **step_kwargs):
    """
    Generic runner: runs step function repeatedly, supports tol+patience convergence.
    Returns final pop, history (mean-objective per step), iters used, converged flag.
    """
    pop = pop0.copy()
    history = []
    best_history = []
    best_val = float("inf")
    stagnant = 0
    iters_used = 0

    for it in range(steps):
        pop = run_step_fn(pop, func, **step_kwargs)
        iters_used += 1
        mean_val = evaluate_population_mean(pop, func)
        history.append(mean_val)
        if mean_val + 0.0 < best_val - tol:
            best_val = mean_val
            stagnant = 0
        else:
            stagnant += 1
        if tol is not None and stagnant >= max(1, patience):
            return pop, history, iters_used, True  # converged
    return pop, history, iters_used, False

def run_experiment(func_name: str = "rastrigin",
                   dim: int = 8,
                   pop_size: int = 32,
                   steps: int = 100,
                   tension_strength: float = 0.1,
                   seed: Optional[int] = None,
                   tol: float = 0.0,
                   patience: int = 5) -> ExperimentResult:
    if seed is not None:
        np.random.seed(seed)

    if func_name not in FUNCTIONS:
        raise ValueError(f"Unsupported func_name: {func_name}")

    func = FUNCTIONS[func_name]
    pop0 = np.random.uniform(-5.0, 5.0, size=(pop_size, dim))

    t0 = time.time()

    # baseline
    FLOPS.reset()
    pop_b, baseline_path, baseline_iters, baseline_conv = run_solver_with_convergence(
        baseline_step, pop0.copy(), func, steps, tol, patience, step_size=0.1
    )
    baseline_flops = FLOPS.snapshot()
    baseline_best = float(min(func(ind) for ind in pop_b))

    # tension
    FLOPS.reset()
    pop_t, tension_path, tension_iters, tension_conv = run_solver_with_convergence(
        tension_step, pop0.copy(), func, steps, tol, patience, tension=tension_strength, step_size=0.1
    )
    tension_flops = FLOPS.snapshot()
    tension_best = float(min(func(ind) for ind in pop_t))

    wall = time.time() - t0

    baseline = {"best": baseline_best, "flops": int(baseline_flops), "iters": baseline_iters, "converged": baseline_conv}
    tension = {"best": tension_best, "flops": int(tension_flops), "iters": tension_iters, "converged": tension_conv}
    flop_savings = int(baseline_flops - tension_flops) if (baseline_flops is not None and tension_flops is not None) else None

    return ExperimentResult(
        problem=func_name,
        params={"dim": dim, "pop_size": pop_size, "steps": steps, "tension_strength": tension_strength, "tol": tol, "patience": patience, "seed": seed},
        baseline=baseline,
        tension=tension,
        flop_savings=flop_savings,
        wall_time_s=float(wall)
    )

# CLI demo when run directly
if __name__ == "__main__":
    import argparse, json, pprint
    parser = argparse.ArgumentParser(description="Run Lumin0 core experiment (baseline vs tension).")
    parser.add_argument("--problem", choices=list(FUNCTIONS.keys()), default="rastrigin")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tension", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    res = run_experiment(func_name=args.problem, dim=args.dim, pop_size=args.pop, steps=args.steps,
                         tension_strength=args.tension, seed=args.seed, tol=args.tol, patience=args.patience)
    pprint.pprint(res.to_dict())
