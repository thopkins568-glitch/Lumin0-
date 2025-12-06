#!/usr/bin/env python3
"""
run_benchmark.py

Thin CLI wrapper for Lumin0 core experiments.
Calls run_experiment() from lumin0_core, saves JSON, prints a short report.
"""

import argparse
import json
import pprint
from datetime import datetime
from lumin0_core import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run Lumin0 benchmark (baseline vs tension).")
    parser.add_argument("--problem", type=str, default="rastrigin", choices=["rastrigin", "rosenbrock"])
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--pop", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tension", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path (auto-generated if empty)")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")

    args = parser.parse_args()

    # Run experiment
    print("Running Lumin0 experiment:")
    print(f"  problem={args.problem} dim={args.dim} pop={args.pop} steps={args.steps} tension={args.tension}")
    result = run_experiment(func_name=args.problem,
                            dim=args.dim,
                            pop_size=args.pop,
                            steps=args.steps,
                            tension_strength=args.tension,
                            seed=args.seed,
                            save_json=False)

    # Human-friendly summary
    summary = {
        "problem": result.problem,
        "dim": result.dim,
        "pop_size": result.pop_size,
        "steps": result.steps,
        "tension_strength": result.tension_strength,
        "baseline_best": result.baseline_best,
        "baseline_flops": result.baseline_flops,
        "tension_best": result.tension_best,
        "tension_flops": result.tension_flops,
        "flop_savings": result.baseline_flops - result.tension_flops,
        "wall_time_s": result.wall_time_s,
    }

    print("\nRESULT SUMMARY")
    pprint.pprint(summary)

    # Optionally save JSON
    if args.save:
        if args.out:
            out_path = args.out
        else:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_path = f"results/lumin0_result_{args.problem}_{ts}.json"
        # ensure parent dir exists
        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"\nSaved full result JSON → {out_path}")

if __name__ == "__main__":
    main()
