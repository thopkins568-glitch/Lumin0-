"""
run_benchmark.py
Unified benchmark runner for all Lumin0 modules (baseline vs tension).
"""

import json
from baseline_solver import run_baseline_all
from tension_solver import run_tension_all


def run_all(tension=0.1):
    """
    Executes:
        - baseline solvers
        - tension solvers
    And returns a unified report dictionary.
    """

    baseline = run_baseline_all()
    tensioned = run_tension_all(tension=tension)

    # compute summary savings for key modules
    summary = {}

    for key in ["force_relax", "matrix_relax", "tsp"]:
        b = baseline[key]
        t = tensioned[key]
        summary[key] = {
            "baseline_flops": b["baseline_flops"],
            "tension_flops": t["tension_flops"],
            "flop_savings": b["baseline_flops"] - t["tension_flops"],
        }

    # "real" solver uses a different return format
    baseline_real_err, baseline_real_flops = baseline["real"]
    tension_real_err, tension_real_flops = tensioned["real"]

    summary["real"] = {
        "baseline_flops": baseline_real_flops,
        "tension_flops": tension_real_flops,
        "flop_savings": baseline_real_flops - tension_real_flops,
    }

    return {
        "baseline": baseline,
        "tension": tensioned,
        "summary": summary,
    }


if __name__ == "__main__":
    result = run_all(tension=0.1)
    print(json.dumps(result, indent=2))
