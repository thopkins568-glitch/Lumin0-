#!/usr/bin/env python3
"""
run_benchmark.py

Unified benchmark runner for Lumin0.
Clean, minimal, and fully compatible with all modules:
 - lumin0_core
 - lumin0_tsp
 - lumin0_matrix_relax
 - lumin0_force_relax
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

# ------------------------------------------------------------
# Core (preferred) — if lumin0_core exists we use it directly
# ------------------------------------------------------------
try:
    from lumin0_core import run_experiment, FLOPS as CORE_FLOPS
    HAS_CORE = True
except Exception:
    run_experiment = None
    CORE_FLOPS = None
    HAS_CORE = False


# ------------------------------------------------------------
# Fallback small functions (rastrigin / rosenbrock)
# ------------------------------------------------------------
def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    n = x.size
    try:
        if CORE_FLOPS is not None:
            CORE_FLOPS.add(n * 6)
    except Exception:
        pass
    return A * n + np.sum(x * x - A * np.cos(2.0 * np.pi * x))


def rosenbrock(x: np.ndarray) -> float:
    n = x.size
    try:
        if CORE_FLOPS is not None:
            CORE_FLOPS.add(n * 10)
    except Exception:
        pass
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


TEST_FUNCTIONS = {
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}


# ------------------------------------------------------------
# Optional modules (TSP, matrix relax, force relax)
# ------------------------------------------------------------
try:
    import lumin0_tsp as tsp_mod
    HAS_TSP = True
except Exception:
    tsp_mod = None
    HAS_TSP = False

try:
    import lumin0_matrix_relax as matrix_mod
    HAS_MATRIX = True
except Exception:
    matrix_mod = None
    HAS_MATRIX = False

try:
    import lumin0_force_relax as force_mod
    HAS_FORCE = True
except Exception:
    force_mod = None
    HAS_FORCE = False


# ------------------------------------------------------------
# Small-function run (when lumin0_core is present)
# ------------------------------------------------------------
def run_small_function_once(
    func_name: str,
    dim: int,
    pop: int,
    steps: int,
    tension: float,
    seed=None,
    save=False,
    outdir="results",
) -> Dict[str, Any]:

    if seed is not None:
        np.random.seed(seed)

    # If lumin0_core is installed, use it exactly
    if HAS_CORE and callable(run_experiment):
        res = run_experiment(
            func_name=func_name,
            dim=dim,
            pop_size=pop,
            steps=steps,
            tension_strength=tension,
            seed=seed,
            save_json=False,
        )
        return res.to_dict()

    # fallback minimal implementation
    func = TEST_FUNCTIONS[func_name]
    pop0 = np.random.uniform(-5.0, 5.0, size=(pop, dim))

    # baseline
    pb = pop0.copy()
    for _ in range(steps):
        for i in range(pop):
            delta = np.random.normal(0, 0.1, size=dim)
            cand = pb[i] + delta
            if func(cand) < func(pb[i]):
                pb[i] = cand
    baseline_best = float(func(pb.mean(axis=0)))

    # tension
    pt = pop0.copy()
    for _ in range(steps):
        center = pt.mean(axis=0)
        for i in range(pop):
            delta = np.random.normal(0, 0.1, size=dim)
            cand = pt[i] + delta + tension * (center - pt[i])
            if func(cand) < func(pt[i]):
                pt[i] = cand
    tension_best = float(func(pt.mean(axis=0)))

    out = {
        "problem": func_name,
        "dim": dim,
        "population": pop,
        "steps": steps,
        "tension_strength": tension,
        "baseline": {"best_value": baseline_best, "flops": None},
        "tension": {"best_value": tension_best, "flops": None},
        "flop_savings": None,
    }

    if save:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with open(Path(outdir) / f"lumin0_{func_name}_{ts}.json", "w") as f:
            json.dump(out, f, indent=2)

    return out


# ------------------------------------------------------------
# Optional problem dispatcher
# ------------------------------------------------------------
def run_optional_problem(name: str, **kw) -> Dict[str, Any]:
    if name == "tsp":
        if not HAS_TSP:
            raise RuntimeError("Add lumin0_tsp.py to enable TSP.")
        return tsp_mod.run_tsp(
            cities=kw.get("cities"),
            steps=kw.get("steps"),
            tension=kw.get("tension"),
            seed=kw.get("seed"),
        )

    if name == "matrix_relax":
        if not HAS_MATRIX:
            raise RuntimeError("Add lumin0_matrix_relax.py to enable matrix relaxation.")
        return matrix_mod.run_matrix_relax(
            n=kw.get("n"),
            steps=kw.get("steps"),
            tension=kw.get("tension"),
            seed=kw.get("seed"),
        )

    if name == "force_relax":
        if not HAS_FORCE:
            raise RuntimeError("Add lumin0_force_relax.py to enable force relaxation.")
        return force_mod.run_force_relax(
            n=kw.get("particles"),
            steps=kw.get("steps"),
            tension=kw.get("tension"),
            seed=kw.get("seed"),
        )

    raise ValueError(f"Unknown problem: {name}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def run_cli():
    p = argparse.ArgumentParser(description="Run Lumin0 benchmark")
    p.add_argument("--problem", type=str, default="rastrigin",
                   choices=["rastrigin", "rosenbrock", "tsp", "matrix_relax", "force_relax"])
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--pop", type=int, default=32)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--tension", type=float, default=0.12)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save", action="store_true")

    # extra for optional modules
    p.add_argument("--cities", type=int, default=32)
    p.add_argument("--matrix_dim", type=int, default=64)
    p.add_argument("--particles", type=int, default=16)

    args = p.parse_args()

    if args.problem in ["rastrigin", "rosenbrock"]:
        out = run_small_function_once(
            func_name=args.problem,
            dim=args.dim,
            pop=args.pop,
            steps=args.steps,
            tension=args.tension,
            seed=args.seed,
            save=args.save,
        )
    else:
        out = run_optional_problem(
            args.problem,
            steps=args.steps,
            tension=args.tension,
            seed=args.seed,
            cities=args.cities,
            n=args.matrix_dim,
            particles=args.particles,
        )

    print(json.dumps(out, indent=2))

    if args.save:
        Path("results").mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        with open(Path("results") / f"lumin0_{args.problem}_{ts}.json", "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    run_cli()
