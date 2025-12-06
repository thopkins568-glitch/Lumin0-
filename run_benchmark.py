#!/usr/bin/env python3
"""
run_benchmark.py

Unified benchmark runner for Lumin0. This script:
 - runs small function benchmarks (rastrigin, rosenbrock)
 - optionally runs larger problems if supporting modules are present:
   - lumin0_tsp (TSP)
   - lumin0_matrix_relax (matrix relaxation)
   - lumin0_force_relax (force relaxation)
 - gracefully handles missing modules and instructs the user how to add them
 - saves JSON results when requested
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

# Try to import lumin0_core.run_experiment if present (preferred)
try:
    from lumin0_core import run_experiment, FLOPS as CORE_FLOPS  # if you have lumin0_core.py
    HAS_CORE = True
except Exception:
    run_experiment = None
    CORE_FLOPS = None
    HAS_CORE = False

# Provide small builtin test functions as fallback
import numpy as np

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    n = x.size
    # best-effort FLOP estimate if a FLOP counter is available
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
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)

# optional larger problem modules (import if present)
HAS_TSP = False
HAS_MATRIX = False
HAS_FORCE = False
try:
    import lumin0_tsp as tsp_mod
    HAS_TSP = True
except Exception:
    tsp_mod = None
try:
    import lumin0_matrix_relax as matrix_mod
    HAS_MATRIX = True
except Exception:
    matrix_mod = None
try:
    import lumin0_force_relax as force_mod
    HAS_FORCE = True
except Exception:
    force_mod = None

TEST_FUNCTIONS = {"rastrigin": rastrigin, "rosenbrock": rosenbrock}

def run_small_function_once(func_name: str, dim: int, pop: int, steps: int, tension: float, seed=None, save=False, outdir="results") -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)

    # If lumin0_core.run_experiment exists, use it directly (preferred)
    if HAS_CORE and callable(run_experiment):
        res = run_experiment(func_name=func_name, dim=dim, pop_size=pop, steps=steps, tension_strength=tension, seed=seed, save_json=False)
        # ExperimentResult has to_dict
        try:
            out = res.to_dict()
        except Exception:
            # fallback mapping
            out = dict(
                problem=func_name,
                dim=dim,
                pop_size=pop,
                steps=steps,
                tension_strength=tension,
                baseline_best=getattr(res, "baseline_best", None),
                baseline_flops=getattr(res, "baseline_flops", None),
                tension_best=getattr(res, "tension_best", None),
                tension_flops=getattr(res, "tension_flops", None),
            )
        return out

    # fallback implementation (uses minimal baseline/tension loops)
    func = TEST_FUNCTIONS[func_name]
    # simple population-based run (mirrors lumin0_core)
    pop0 = np.random.uniform(-5.0, 5.0, size=(pop, dim))

    # baseline
    baseline_pop = pop0.copy()
    # no external FLOPS counter here (user can add lumin0_core to get full accounting)
    for _ in range(steps):
        # simple random-local-search step (fast)
        for i in range(pop):
            delta = np.random.normal(0, 0.1, size=(dim,))
            cand = baseline_pop[i] + delta
            if func(cand) < func(baseline_pop[i]):
                baseline_pop[i] = cand
    baseline_best = float(func(baseline_pop.mean(axis=0)))
    baseline_flops = None

    # tension
    tension_pop = pop0.copy()
    for _ in range(steps):
        center = tension_pop.mean(axis=0)
        for i in range(pop):
            delta = np.random.normal(0, 0.1, size=(dim,))
            cand = tension_pop[i] + delta + tension * (center - tension_pop[i])
            if func(cand) < func(tension_pop[i]):
                tension_pop[i] = cand
    tension_best = float(func(tension_pop.mean(axis=0)))
    tension_flops = None

    result = {
        "problem": func_name,
        "dim": dim,
        "population": pop,
        "steps": steps,
        "tension_strength": tension,
        "baseline": {"best_value": baseline_best, "flops": baseline_flops},
        "tension": {"best_value": tension_best, "flops": tension_flops},
        "flop_savings": None,
    }

    if save:
        p = Path(outdir)
        p.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = p / f"lumin0_{func_name}_{ts}.json"
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)

    return result

def run_optional_problem(name: str, **kwargs) -> Dict[str, Any]:
    """Dispatch to optional problem modules if available."""
    if name == "tsp":
        if not HAS_TSP:
            raise RuntimeError("lumin0_tsp module not found. Add lumin0_tsp.py to enable TSP runs.")
        return tsp_mod.run_tsp(cities=kwargs.get("cities", 32), steps=kwargs.get("steps", 500), tension=kwargs.get("tension", 0.1), seed=kwargs.get("seed", None))
    if name == "matrix_relax":
        if not HAS_MATRIX:
            raise RuntimeError("lumin0_matrix_relax module not found. Add lumin0_matrix_relax.py to enable matrix relaxation runs.")
        return matrix_mod.run_matrix_relax(n=kwargs.get("n", 64), steps=kwargs.get("steps", 200), tension=kwargs.get("tension", 0.05), seed=kwargs.get("seed", None))
    if name == "force_relax":
        if not HAS_FORCE:
            raise RuntimeError("lumin0_force_relax module not found. Add lumin0_force_relax.py to enable force relaxation runs.")
        return force_mod.run_force_relax(n=kwargs.get("n", 16), steps=kwargs.get("steps", 200), tension=kwargs.get("tension", 0.1), seed=kwargs.get("seed", None))
    raise ValueError(f"Unknown optional problem: {name}")

def run_cli():
    parser = argparse.ArgumentParser(description="Run Lumin0 benchmark (minimal runner).")
    parser.add_argument("--problem", type=str, default="rastrigin",
                        choices=["rastrigin", "rosenbrock", "tsp", "matrix_relax", "force_relax"])
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tension", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", action="store_true", help="Save JSON output")
    # extra params for optional problems
    parser.add_argument("--cities", type=int, default=32, help="(TSP) number of cities")
    parser.add_argument("--matrix_dim", type=int, default=64, help="(matrix_relax) dimension")
    parser.add_argument("--particles", type=int, default=16, help="(force_relax) particles")
    args = parser.parse_args()

    problem = args.problem
    if problem in ["rastrigin", "rosenbrock"]:
        out = run_small_function_once(func_name=problem, dim=args.dim, pop=args.pop, steps=args.steps, tension=args.tension, seed=args.seed, save=args.save)
    else:
        try:
            out = run_optional_problem(problem, steps=args.steps, tension=args.tension, seed=args.seed, cities=args.cities, n=args.matrix_dim, particles=args.particles)
        except Exception as e:
            print(f"ERROR running problem {problem}: {e}")
            print("If this module is missing, add the corresponding module file (lumin0_tsp.py, lumin0_matrix_relax.py, or lumin0_force_relax.py).")
            return

    # normalize flop_savings if None
    if out.get("flop_savings") is None and out.get("baseline", {}).get("flops") is not None and out.get("tension", {}).get("flops") is not None:
        out["flop_savings"] = int(out["baseline"]["flops"] - out["tension"]["flops"])

    print("\n=== Lumin0 Result ===")
    print(json.dumps(out, indent=2))

    if args.save and isinstance(out, dict):
        p = Path("results")
        p.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        fname = p / f"lumin0_{problem}_{ts}.json"
        with open(fname, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Saved -> {fname}")

if __name__ == "__main__":
    run_cli()
