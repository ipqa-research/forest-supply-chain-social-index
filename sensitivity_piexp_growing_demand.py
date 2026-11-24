import argparse
import os
import time
from datetime import datetime

import pandas as pd
import pyomo.environ as pyo

from model import COST_SCALE, GROWING_DEMAND_DIR, RESULTS_DIR, build_model

RUN_START = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

W1 = 1
W2 = 1
# Normalization constants of SI (see model.py). They only affect the reported
# SI, since this analysis minimizes TPC.
MAX_EV = 6104.679405
MAX_JC = 29.761281

PI_EXP_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Wall-clock time limit per instance, in seconds.
TIME_LIMIT = 8000
MIP_GAP = 0.00

SOLVER_NAME = "gurobi"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "sensitivity_piexp_growing_demand")
LOG_DIR = os.path.join(OUTPUT_DIR, "solver_logs")


def solve(model, pi_exp, time_limit=TIME_LIMIT, mip_gap=MIP_GAP):
    solver = pyo.SolverFactory(SOLVER_NAME)
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = mip_gap
    solver.options["Threads"] = 8
    solver.options["MIPFocus"] = 3
    solver.options["Cuts"] = 2
    solver.options["Seed"] = 12345
    os.makedirs(LOG_DIR, exist_ok=True)
    solver.options["LogFile"] = os.path.join(LOG_DIR, f"gurobi_piexp_{RUN_START}_{pi_exp}.log")

    t0 = time.time()
    results = solver.solve(model, tee=True, load_solutions=False)
    elapsed = time.time() - t0
    has_solution = len(results.solution) > 0
    if has_solution:
        model.solutions.load_from(results)
    return results, elapsed, has_solution


def extract_metrics(model, pi_exp):
    xdisc_total = sum(
        pyo.value(model.Xdisc[k, m, t])
        for k in model.K for m in model.Mnew for t in model.T
        if (m, k) in model.Set_MK
    ) * COST_SCALE

    events_per_plant = {}
    for k in model.K:
        for m in model.Mnew:
            if (m, k) not in model.Set_MK:
                continue
            n_events = sum(1 for c in model.C for t in model.T if pyo.value(model.y[k, m, c, t]) > 0.5)
            if n_events > 0:
                events_per_plant[(k, m)] = n_events

    return {
        "TPC": pyo.value(model.TPC),
        "SI": pyo.value(model.SI),
        "Xdisc_total": xdisc_total,
        "discount_applied_USD": pi_exp * xdisc_total,
        "plants_built": len(events_per_plant),
        "staged_plants": sum(1 for n in events_per_plant.values() if n > 1),
        "total_events": sum(events_per_plant.values()),
    }


def main(pi_exp_values):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    summary_path = os.path.join(OUTPUT_DIR, f"sensitivity_summary_{RUN_START}.xlsx")

    print("Minimizing TPC without a bound on SI, growing demand scenario.")

    for pi_exp in pi_exp_values:
        print("\n" + "=" * 70)
        print(f"  PI_EXP = {pi_exp}")
        print("=" * 70)

        model, save_results, _ = build_model(
            W1, W2, MAX_EV, MAX_JC,
            objective="TPC", sense="minimize",
            pi_exp=pi_exp, solver_name=SOLVER_NAME, solve=False,
            data_dir=GROWING_DEMAND_DIR, output_dir=OUTPUT_DIR,
        )

        try:
            results, elapsed, has_solution = solve(model, pi_exp)
        except Exception as e:
            print(f"  ! Error solving pi_exp = {pi_exp}: {e!r}")
            summary.append({"pi_exp": pi_exp, "status": "error", "termination": "error",
                            "solver_time_s": None, "gap_%": None, "error": repr(e)})
            pd.DataFrame(summary).to_excel(summary_path, index=False)
            continue
        status = str(results.solver.status)
        termination = str(results.solver.termination_condition)

        row = {
            "pi_exp": pi_exp,
            "status": status,
            "termination": termination if has_solution else f"{termination} (no solution)",
            "solver_time_s": round(elapsed, 1),
        }

        if not has_solution:
            row["gap_%"] = None
        elif termination == "optimal":
            row["gap_%"] = 0.0
        else:
            try:
                ub = results.problem.upper_bound
                lb = results.problem.lower_bound
                row["gap_%"] = round(abs(ub - lb) / max(abs(ub), 1e-10) * 100, 4)
            except Exception:
                row["gap_%"] = None

        usable = has_solution and termination in ("optimal", "maxTimeLimit", "feasible")
        if usable:
            try:
                pyo.value(model.TPC)
            except Exception:
                usable = False

        if usable:
            try:
                metrics = extract_metrics(model, pi_exp)
                row.update(metrics)
                print(
                    f"  TPC = {metrics['TPC']:,.2f}  |  SI = {metrics['SI']:.6f}  |  "
                    f"Xdisc total = {metrics['Xdisc_total']:,.2f}  |  "
                    f"discount applied = {metrics['discount_applied_USD']:,.2f}  |  "
                    f"staged plants = {metrics['staged_plants']}/{metrics['plants_built']}"
                )
                save_results(model, filename=os.path.join(OUTPUT_DIR, f"growing_demand_piexp-{pi_exp:.1f}.xlsx"))
            except Exception as e:
                print(f"  ! Could not extract or save results: {e}")
                row["error"] = str(e)
        else:
            print(f"  ! No usable solution (status = {status}, termination = {termination}).")

        summary.append(row)
        pd.DataFrame(summary).to_excel(summary_path, index=False)

    df_summary = pd.DataFrame(summary)
    df_summary.to_excel(summary_path, index=False)
    print("\n" + df_summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensitivity of the minimum-TPC solution to pi_exp under growing demand.")
    parser.add_argument("--pi-exp", type=float, nargs="+", default=PI_EXP_VALUES)
    args = parser.parse_args()
    main(args.pi_exp)
