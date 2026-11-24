import argparse
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

from model import BASE_CASE_DIR, COST_SCALE, EAP_SCALE, RESULTS_DIR, build_model

RUN_START = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

W1 = 1
W2 = 1
PI_EXP = 0.2

# Normalization constants of SI (see model.py).
MAX_EV = 6104.679405
MAX_JC = 29.761281

N_POINTS = 15
TIME_LIMIT_ANCHORS = 15000
TIME_LIMIT_SWEEP = 6200
MIP_GAP_SWEEP = 0.00

OUTPUT_DIR = os.path.join(RESULTS_DIR, "pareto")


def solve(model, time_limit=TIME_LIMIT_SWEEP, mip_gap=MIP_GAP_SWEEP, tag=""):
    solver = pyo.SolverFactory("gurobi")
    solver.options["TimeLimit"] = time_limit
    solver.options["MIPGap"] = mip_gap
    solver.options["Threads"] = 8
    solver.options["MIPFocus"] = 3
    solver.options["Cuts"] = 2
    solver.options["Seed"] = 12345
    if tag:
        log_dir = os.path.join(OUTPUT_DIR, "solver_logs")
        os.makedirs(log_dir, exist_ok=True)
        solver.options["LogFile"] = os.path.join(log_dir, f"gurobi_pareto_{RUN_START}_{tag}.log")
    t0 = time.time()
    result = solver.solve(model, tee=False, load_solutions=True, warmstart=True)
    return result, time.time() - t0


def _final_r_and_jobs(m, k, mm):
    r_val = sum(pyo.value(m.Epsilo[c, mm]) * (1.0 if pyo.value(m.y[k, mm, c, t]) > 0.5 else 0.0)
                for c in m.C for t in m.T)
    if r_val <= 1e-9:
        return 0.0, 0.0
    jobs = pyo.value(m.F_emp[mm]) + pyo.value(m.V_emp[k, mm]) * r_val ** pyo.value(m.b_emp[mm])
    return r_val, jobs


def total_jobs(m):
    return sum(_final_r_and_jobs(m, k, mm)[1] for (k, mm) in m.Emp_set)


def jobs_detail(m):
    rows = []
    for (k, mm) in m.Emp_set:
        levels = [c for c in m.C if any(pyo.value(m.y[k, mm, c, t]) > 0.5 for t in m.T)]
        if not levels:
            continue
        r_val, jobs = _final_r_and_jobs(m, k, mm)
        total_jc = sum(pyo.value(m.JC[k, mm, t]) for t in m.T) / EAP_SCALE
        rows.append({
            "Node": k,
            "Technology": mm,
            "Levels": ", ".join(levels),
            "r_cumulative_capacity": round(r_val, 4),
            "Final_jobs": round(jobs, 2),
            "JC_horizon_total": total_jc,
            "JC_annual_mean": total_jc / len(m.T),
        })
    return pd.DataFrame(rows)


def set_objective(m, expr, sense):
    for o in list(m.component_objects(pyo.Objective, active=True)):
        m.del_component(o)
    m.obj = pyo.Objective(expr=expr, sense=sense)


def write_final_report(ts, folder, anchors_ok, anchor_error, df, SI_max=None, TPC_min=None):
    lines = [f"FINAL REPORT - TPC/SI PARETO SWEEP ({ts})", "=" * 60]
    if not anchors_ok:
        lines += ["", "The anchors were not solved; the sweep was not run.", f"Reason: {anchor_error}"]
    else:
        lines += ["", f"SI_max (anchor)  = {SI_max:.6f}", f"TPC_min (anchor) = {TPC_min:,.2f}", ""]
        optimal = df[df['status'] == 'optimal']
        time_limited = df[df['status'].astype(str).str.startswith('TimeLimit')]
        no_solution = df[(df['status'] != 'optimal') & ~df['status'].astype(str).str.startswith('TimeLimit')]
        lines.append(f"Sweep points: {len(df)} -- {len(optimal)} optimal, "
                     f"{len(time_limited)} stopped by the time limit (saved, gap > 0), "
                     f"{len(no_solution)} without solution.")
        for label, subset in (("Stopped by the time limit:", time_limited), ("Without solution:", no_solution)):
            if len(subset) > 0:
                lines.append(label)
                for _, row in subset.iterrows():
                    lines.append(f"  - Point {int(row['point']):02d} (SI >= {row['eps']:.6f}): "
                                 f"{row['status']}, solver time = {row['solver_time']:.1f} s")
    text = "\n".join(lines)
    with open(os.path.join(folder, f"final_report_{ts}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def pareto_front(min_cost_only=False):
    ts = RUN_START
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building the model...")
    model, save_results, _ = build_model(
        w1=W1, w2=W2, max_EV=MAX_EV, max_JC=MAX_JC,
        objective='SI', sense='maximize', pi_exp=PI_EXP,
        solver_name='gurobi', solve=False,
        data_dir=BASE_CASE_DIR, output_dir=OUTPUT_DIR,
    )

    df = pd.DataFrame()

    print("\nAnchor: minimum TPC...")
    set_objective(model, model.TPC, pyo.minimize)
    res, time_minTPC = solve(model, time_limit=TIME_LIMIT_ANCHORS, mip_gap=0.00, tag="anchor_minTPC")
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        error = f"The minimum-TPC anchor was not solved to optimality: {res.solver.termination_condition}"
        write_final_report(ts, OUTPUT_DIR, False, error, df)
        return
    TPC_min = pyo.value(model.TPC)
    SI_at_TPCmin = pyo.value(model.SI)
    print(f"TPC_min = {TPC_min:,.2f} USD | SI = {SI_at_TPCmin:.6f} | time = {time_minTPC:.1f} s")
    save_results(model, filename=os.path.join(OUTPUT_DIR, f"pareto_{ts}_anchor_minTPC.xlsx"))
    if min_cost_only:
        return

    print("\nAnchor: maximum SI...")
    set_objective(model, model.SI, pyo.maximize)
    res, time_maxSI = solve(model, time_limit=TIME_LIMIT_ANCHORS, mip_gap=0.00, tag="anchor_maxSI")
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        error = f"The maximum-SI anchor was not solved to optimality: {res.solver.termination_condition}"
        write_final_report(ts, OUTPUT_DIR, False, error, df)
        return
    SI_max = pyo.value(model.SI)
    print(f"SI_max = {SI_max:.6f} | TPC = {pyo.value(model.TPC):,.2f} USD | time = {time_maxSI:.1f} s")
    save_results(model, filename=os.path.join(OUTPUT_DIR, f"pareto_{ts}_anchor_maxSI.xlsx"))

    print("\nEpsilon-constraint sweep...")
    epsilons = np.linspace(SI_max, SI_at_TPCmin, N_POINTS)
    set_objective(model, model.TPC, pyo.minimize)

    points = []
    empty = {'file': '', 'SI': None, 'TPC': None, 'jobs': None, 'JC': None, 'EV': None}
    for i, eps in enumerate(epsilons):
        print(f"\nPoint {i + 1}/{N_POINTS}: SI >= {eps:.6f}")
        try:
            model.epsilon_constraint = pyo.Constraint(expr=model.SI >= eps)
            res, solver_time = solve(model, tag=f"point_{i + 1:02d}")
            termination = res.solver.termination_condition
            if termination in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit):
                if termination == pyo.TerminationCondition.optimal:
                    status = 'optimal'
                else:
                    try:
                        ub, lb = res.problem.upper_bound, res.problem.lower_bound
                        status = f"TimeLimit (gap {abs(ub - lb) / max(abs(ub), 1e-10) * 100:.2f}%)"
                    except Exception:
                        status = "TimeLimit (gap unknown)"

                name = f"pareto_point_{i + 1:02d}.xlsx"
                save_results(model, filename=os.path.join(OUTPUT_DIR, name))
                with pd.ExcelWriter(os.path.join(OUTPUT_DIR, name), engine='openpyxl', mode='a') as writer:
                    jobs_detail(model).to_excel(writer, sheet_name="jobs_detail", index=False)

                points.append({
                    'point': i + 1, 'eps': eps, 'file': name,
                    'SI': pyo.value(model.SI),
                    'TPC': pyo.value(model.TPC),
                    'jobs': total_jobs(model),
                    'JC': pyo.value(model.JC_obj) / EAP_SCALE,
                    'EV': pyo.value(model.EV_obj) * COST_SCALE,
                    'status': status, 'solver_time': solver_time,
                })
                print(f"  -> {status}: TPC = {pyo.value(model.TPC):,.2f} USD | SI = {pyo.value(model.SI):.6f}")
            else:
                print(f"  -> No usable solution: {termination}")
                points.append({'point': i + 1, 'eps': eps, **empty,
                               'status': str(termination), 'solver_time': solver_time})
        except Exception as e:
            print(f"  -> Error: {e!r}")
            points.append({'point': i + 1, 'eps': eps, **empty,
                           'status': f"ERROR: {e!r}", 'solver_time': 0})
        finally:
            if hasattr(model, 'epsilon_constraint'):
                model.del_component(model.epsilon_constraint)

    df = pd.DataFrame(points)
    df.to_excel(os.path.join(OUTPUT_DIR, f"pareto_summary_{ts}.xlsx"), index=False)

    ok = df.dropna(subset=['TPC', 'SI']).copy()
    if len(ok) >= 2:
        ok['TPC_plot'] = 1 - (ok['TPC'] - ok['TPC'].min()) / (ok['TPC'].max() - ok['TPC'].min() + 1e-6)
        ok['SI_plot'] = (ok['SI'] - ok['SI'].min()) / (ok['SI'].max() - ok['SI'].min() + 1e-6)
        plt.figure(figsize=(11, 7))
        plt.plot(ok['TPC_plot'], ok['SI_plot'], '--o', color='gray')
        plt.scatter(ok['TPC_plot'], ok['SI_plot'], c=ok['EV'], cmap='viridis', s=120, edgecolors='black', zorder=3)
        plt.xlabel("Total present cost (normalized)", fontsize=13)
        plt.ylabel("Social index (normalized)", fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.colorbar(label="Economic value")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"pareto_{ts}.png"), dpi=300)

    print(df.to_string(index=False))
    write_final_report(ts, OUTPUT_DIR, True, None, df, SI_max=SI_max, TPC_min=TPC_min)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPC/SI Pareto front by the epsilon-constraint method.")
    parser.add_argument("--min-cost-only", action="store_true",
                        help="solve only the minimum-TPC point and stop")
    args = parser.parse_args()
    pareto_front(min_cost_only=args.min_cost_only)
