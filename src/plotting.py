"""Figures 1-4 of the paper from one or more evaluation_data.csv files.

Produces:
  stance_drift.pdf         per-model L/C/R proportions over rounds   (Figure 1)
  bias_decay.pdf           mean absolute score per round             (Figure 2)
  transition_matrices.pdf  per-model transition heatmaps             (Figure 3)
  stationary_dist.pdf      per-model stationary distributions        (Figure 4)

Example:
  python -m src.plotting \
      --inputs runs/gpt51/evaluation_data.csv runs/gpt4o/evaluation_data.csv \
      --labels GPT5.1 GPT4o --rounds 20 --output figs/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import STATES, stationary_distribution, transition_matrix

COLORS = {"Left": "#1f77b4", "Center": "#d4af37", "Right": "#b22222"}


def round_proportions(df: pd.DataFrame, rounds: int) -> pd.DataFrame:
    rows = []
    for r in range(rounds + 1):
        sub = df[df["round"] == r]["predicted"]
        counts = sub.value_counts().reindex(STATES, fill_value=0)
        total = counts.sum()
        rows.append([r] + (counts / total if total else counts).tolist())
    return pd.DataFrame(rows, columns=["round"] + STATES)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if len(args.inputs) != len(args.labels):
        ap.error("--inputs and --labels must have the same length")

    os.makedirs(args.output, exist_ok=True)
    data = {lb: pd.read_csv(p) for lb, p in zip(args.labels, args.inputs)}
    n = len(data)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    # Figure 1: stance drift -----------------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharey=True)
    for ax, (label, df) in zip(axes.flat, data.items()):
        props = round_proportions(df, args.rounds)
        for s in STATES:
            ax.plot(props["round"], props[s], label=s, color=COLORS[s], lw=1.5)
        ax.set_title(label)
        ax.set_xlabel("Round")
        ax.set_ylim(0, 1)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0][0].set_ylabel("Proportion")
    axes[0][-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Stance Drift Over Paraphrasing Rounds")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "stance_drift.pdf"))
    plt.close(fig)

    # Figure 2: bias decay (MAS) -------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, df in data.items():
        mas = df.groupby("round")["score"].apply(lambda s: s.abs().mean())
        ax.plot(mas.index, mas.values, marker="o", ms=3, lw=1.2, label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Absolute Score")
    ax.set_title("Decay of Bias Intensity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "bias_decay.pdf"))
    plt.close(fig)

    # Figure 3: transition matrices ----------------------------------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows),
                             squeeze=False)
    for ax, (label, df) in zip(axes.flat, data.items()):
        P = transition_matrix(df, args.rounds)
        ax.imshow(P, cmap="YlGnBu", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center",
                        color="white" if P[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), STATES, fontsize=8)
        ax.set_yticks(range(3), STATES, fontsize=8)
        ax.set_title(label)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Transition Probability Matrices")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "transition_matrices.pdf"))
    plt.close(fig)

    # Figure 4: stationary distributions -----------------------------------
    fig, ax = plt.subplots(figsize=(1.2 * n + 3, 4))
    width = 0.25
    x = np.arange(n)
    pis = {lb: stationary_distribution(transition_matrix(df, args.rounds))
           for lb, df in data.items()}
    for k, s in enumerate(STATES):
        ax.bar(x + (k - 1) * width, [pis[lb][k] for lb in data], width,
               label=s, color=COLORS[s])
    ax.axhline(1 / 3, ls="--", c="gray", lw=1, label="Initial (1/3)")
    ax.set_xticks(x, list(data.keys()), fontsize=8)
    ax.set_ylabel("Probability")
    ax.set_title("Predicted Stationary Distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "stationary_dist.pdf"))
    plt.close(fig)

    print(f"Done. Figures saved to {args.output}/")


if __name__ == "__main__":
    main()
