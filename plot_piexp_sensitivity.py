import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sensitivity_piexp_growing_demand")
RHO = 0.02
OUTPUT = os.path.join(FOLDER, "piexp_sensitivity")
BLUE = "#2a78d6"


def load_summaries(files):
    frames = []
    for order, name in enumerate(files):
        d = pd.read_excel(name).dropna(subset=["TPC"])
        d["file"] = os.path.basename(name)
        d["order"] = order
        frames.append(d[["pi_exp", "TPC", "gap_%", "file", "order"]])
    return (pd.concat(frames)
              .sort_values("order")
              .drop_duplicates("pi_exp", keep="last")
              .sort_values("pi_exp")
              .reset_index(drop=True))


def discounted_capital_cost(pi):
    cc = pd.read_excel(os.path.join(FOLDER, f"growing_demand_piexp-{pi:.1f}.xlsx"), sheet_name="CC")
    return sum(v / (1 + RHO) ** (t - 1) for t, v in zip(cc["T"], cc["Value"]))


def main(files):
    data = load_summaries(files)
    data["TPC_bn"] = data["TPC"] / 1e9
    data["LB_bn"] = data["TPC_bn"] * (1 - data["gap_%"] / 100)
    data["CC_bn"] = [discounted_capital_cost(p) / 1e9 for p in data["pi_exp"]]
    print(data[["pi_exp", "TPC_bn", "LB_bn", "gap_%", "CC_bn", "file"]].round(3).to_string(index=False))

    plt.rcParams.update({"font.size": 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    a.fill_between(data["pi_exp"], data["LB_bn"], data["TPC_bn"], color="0.85", label="Optimality gap")
    a.plot(data["pi_exp"], data["TPC_bn"], "-o", color=BLUE, ms=3, lw=1.5, label="Best solution found")
    a.plot(data["pi_exp"], data["LB_bn"], "--", color="0.45", lw=1, label="Lower bound")
    a.set_title("(a) Total present cost", loc="left", fontsize=9)
    a.set_xlabel(r"Expansion discount $\pi^{exp}$")
    a.set_ylabel(r"$TPC$ (billion USD)")
    a.legend(loc="lower left", fontsize=7, frameon=True)
    a.grid(color="0.9")

    b.plot(data["pi_exp"], data["CC_bn"], "-o", color=BLUE, ms=3, lw=1.5)
    b.set_title("(b) Capital cost", loc="left", fontsize=9)
    b.set_xlabel(r"Expansion discount $\pi^{exp}$")
    b.set_ylabel("Discounted capital cost (billion USD)")
    b.grid(color="0.9")

    fig.tight_layout()
    fig.savefig(OUTPUT + ".pdf")
    fig.savefig(OUTPUT + ".png", dpi=200)
    print(f"Saved: {OUTPUT}.pdf and {OUTPUT}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the sensitivity of TPC and capital cost to pi_exp.")
    parser.add_argument("summaries", nargs="*",
                        help="sensitivity_summary_*.xlsx files; when a pi_exp appears in several, the last file wins "
                             "(default: all summaries in the results folder, oldest first)")
    args = parser.parse_args()
    files = args.summaries or sorted(glob.glob(os.path.join(FOLDER, "sensitivity_summary_*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No sensitivity_summary_*.xlsx in {FOLDER}; run sensitivity_piexp_growing_demand.py first.")
    main(files)
