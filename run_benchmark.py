#!/usr/bin/env python3
"""
run_benchmark.py
Unified runner for Lumin0 tasks. Uses replaced modules and unified result schema.
"""
import argparse
import json
import time
from pathlib import Path

# import core experiment and optional modules
from lumin0_core import run_experiment as core_run
import lumin0_tsp as tsp_mod
import lumin0_matrix_relax as matrix_mod
import lumin0_force_relax as force_mod

def save_json(data, out_path=None):
    if out_path is None:
        Path("results").mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = f"results/lumin0_run_{ts}.json"
    with open(out_path, "w") as fh:
        json.dump(data, fh, indent=2)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Run Lumin0 benchmark (core and optional problems)")
    parser.add_argument("--problem", choices=["rastrigin", "rosenbrock", "tsp", "matrix_relax", "force_relax"], default="rastrigin")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tension", type=float, default=0.12)
    parser.add_argument("--tol", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--cities", type=int, default=32)
    parser.add_argument("--matrix_dim", type=int, default=64)
    parser.add_argument("--particles", type=int, default=16)
    args = parser.parse_args()

    start = time.time()
    if args.problem in ["rastrigin", "rosenbrock"]:
        res = core_run(func_name=args.problem, dim=args.dim, pop_size=args.pop, steps=args.steps,
                       tension_strength=args.tension, seed=args.seed, tol=args.tol, patience=args.patience)
        out = res.to_dict()
    elif args.problem == "tsp":
        out = tsp_mod.run_tsp(cities=args.cities, steps=args.steps, tension=args.tension, tol=args.tol, patience=args.patience, seed=args.seed)
    elif args.problem == "matrix_relax":
        out = matrix_mod.run_matrix_relax(n=args.matrix_dim, steps=args.steps, tension=args.tension, tol=args.tol, patience=args.patience, seed=args.seed)
    elif args.problem == "force_relax":
        out = force_mod.run_force_relax(n=args.particles, steps=args.steps, tension=args.tension, tol=args.tol, patience=args.patience, seed=args.seed)
    else:
        raise ValueError("Unknown problem")

    out["wall_time_s"] = float(time.time() - start)

    print(json.dumps(out, indent=2))

    if args.save:
        path = args.out if args.out else None
        saved = save_json(out, path)
        print("Saved ->", saved)

if __name__ == "__main__":
    main()
