import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PARETO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pareto")
SIZES = [80, 200, 400]


def latest_summary():
    files = sorted(glob.glob(os.path.join(PARETO_DIR, "pareto_summary_*.xlsx")))
    if not files:
        raise FileNotFoundError(f"No pareto_summary_*.xlsx in {PARETO_DIR}; run pareto_front.py first.")
    return files[-1]


def plot(summary_file, normalized=True, output=None):
    df = pd.read_excel(summary_file).dropna(subset=["TPC", "SI"])

    if normalized:
        df["x"] = 1 - (df["TPC"] - df["TPC"].min()) / (df["TPC"].max() - df["TPC"].min() + 1e-6)
        df["y"] = (df["SI"] - df["SI"].min()) / (df["SI"].max() - df["SI"].min() + 1e-6)
        xlabel, ylabel, legend_loc = "Normalized Total Present Cost", "Normalized Social Index", "lower left"
    else:
        df["x"] = df["TPC"] / 1e9
        df["y"] = df["SI"]
        xlabel, ylabel, legend_loc = "Total Present Cost (billions of USD)", "Social Index", "lower right"

    q1 = df["jobs"].quantile(1 / 3)
    q2 = df["jobs"].quantile(2 / 3)
    df["size"] = df["jobs"].apply(lambda v: SIZES[0] if v <= q1 else (SIZES[1] if v <= q2 else SIZES[2]))

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    scatter = ax.scatter(df["x"], df["y"], s=df["size"], c=df["EV"], cmap="viridis",
                         edgecolors="black", linewidth=1.2)
    ax.plot(df["x"], df["y"], "--", color="gray", alpha=0.5)
    if normalized:
        ax.set_aspect('equal')

    for _, row in df.iterrows():
        radius = row["size"] ** 0.5 / 2
        ax.annotate(f"{int(row['point'])}", xy=(row["x"], row["y"]),
                    xytext=(radius + 4, -4), textcoords="offset points", fontsize=11)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, alpha=0.3)

    labels = [f"[{df['jobs'].min():.0f}, {q1:.0f}]", f"({q1:.0f}, {q2:.0f}]", f"({q2:.0f}, {df['jobs'].max():.0f}]"]
    handles = [ax.scatter([], [], s=s, c="white", edgecolors="black", linewidth=1.2) for s in SIZES]
    leg = ax.legend(handles, labels, title="Total jobs created", loc=legend_loc, frameon=True,
                    fontsize=11, title_fontsize=12, scatterpoints=1, borderpad=1.2, labelspacing=1.4)
    leg.get_frame().set_alpha(0.90)

    cax = fig.add_axes([0.88, 0.15, 0.02, 0.70])
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label("Economic Value", fontsize=14)
    plt.subplots_adjust(left=0.10, right=0.80)

    if output is None:
        stem = os.path.splitext(os.path.basename(summary_file))[0]
        output = os.path.join(PARETO_DIR, f"{stem}_{'normalized' if normalized else 'absolute'}.png")
    plt.savefig(output, dpi=300)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the TPC/SI Pareto front.")
    parser.add_argument("summary", nargs="?", default=None, help="pareto_summary_*.xlsx (default: latest)")
    parser.add_argument("--absolute", action="store_true", help="plot TPC and SI without normalization")
    args = parser.parse_args()
    plot(args.summary or latest_summary(), normalized=not args.absolute)
