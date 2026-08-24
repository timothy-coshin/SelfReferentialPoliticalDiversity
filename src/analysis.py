"""Per-model analysis (Tables 1-2, Figures 2-4 of the paper).

Reads a long-format evaluation_data.csv (from src.judge) and computes:
  - final L/C/R stance distribution at round R
  - Stuart-Maxwell marginal homogeneity test (round-0 ground truth vs round R)
  - VRR: variance reduction ratio of continuous scores (Eq. 4; sample variance)
  - MAS per round (Eq. 5)
  - pooled transition matrix over rounds 1..R (Figure 3)
  - stationary distribution, political entropy, entropy reduction ratio (Table 2)

Conventions matching the paper:
  - Var(v^0) is a shared baseline measured once on the original seed articles
    (0.3626 in the paper); pass it via --baseline-var so all models share the
    same VRR denominator. If omitted, this run's own round-0 variance is used.
  - Variances are sample variances (ddof=1).
  - Entropy uses the natural log; the initial entropy is computed from the
    ground-truth seed label distribution.

Example:
  python -m src.analysis --input runs/gpt4o/evaluation_data.csv \
      --output runs/gpt4o/analysis --rounds 20 --baseline-var 0.3626
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

STATES = ["Left", "Center", "Right"]


def transition_matrix(df: pd.DataFrame, rounds: int) -> np.ndarray:
    """Pooled single-step transition matrix over transitions into rounds 1..R."""
    idx = {s: i for i, s in enumerate(STATES)}
    counts = np.zeros((3, 3))
    for _, g in df.sort_values("round").groupby("original_id"):
        labels = dict(zip(g["round"], g["predicted"]))
        for r in range(1, rounds + 1):
            a, b = labels.get(r - 1), labels.get(r)
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    rowsum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rowsum, out=np.zeros_like(counts), where=rowsum != 0)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    v = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    return v / v.sum()


def entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def stuart_maxwell(df: pd.DataFrame, rounds: int):
    """Stuart-Maxwell test: round-0 ground-truth labels vs round-R judge labels."""
    from statsmodels.stats.contingency_tables import SquareTable

    base = df[df["round"] == 0][["original_id", "original_stance"]]
    final = df[df["round"] == rounds][["original_id", "predicted"]]
    paired = base.merge(final, on="original_id")
    table = pd.crosstab(
        pd.Categorical(paired["original_stance"], categories=STATES),
        pd.Categorical(paired["predicted"], categories=STATES),
        dropna=False,
    )
    res = SquareTable(table.values).homogeneity()
    return float(res.statistic), int(res.df), float(res.pvalue), len(paired)


def label_distribution(df: pd.DataFrame, round_no: int, column: str = "predicted"):
    sub = df[df["round"] == round_no][column]
    counts = sub.value_counts().reindex(STATES, fill_value=0)
    return (counts / counts.sum()).tolist()


def analyze(df: pd.DataFrame, rounds: int, baseline_var: float | None):
    var0_own = float(df[df["round"] == 0]["score"].var(ddof=1))
    var_r = float(df[df["round"] == rounds]["score"].var(ddof=1))
    var0 = baseline_var if baseline_var is not None else var0_own

    chi2, dof, pvalue, n = stuart_maxwell(df, rounds)
    P = transition_matrix(df, rounds)
    pi = stationary_distribution(P)

    init_counts = (df[df["round"] == 0]
                   .drop_duplicates("original_id")["original_stance"]
                   .value_counts().reindex(STATES, fill_value=0))
    pi0 = (init_counts / init_counts.sum()).values
    h0, h_stat = entropy(pi0), entropy(pi)

    mas = (df.groupby("round")["score"]
             .apply(lambda s: float(s.abs().mean()))
             .reindex(range(rounds + 1)))

    return {
        "n_articles": int(n),
        "initial_distribution": [round(v, 4) for v in pi0],
        "final_distribution": [round(v, 4) for v in label_distribution(df, rounds)],
        "stuart_maxwell": {"chi2": round(chi2, 4), "df": dof, "pvalue": round(pvalue, 4)},
        "variance_round0_baseline": round(var0, 4),
        "variance_round0_this_run": round(var0_own, 4),
        f"variance_round{rounds}": round(var_r, 4),
        "vrr_percent": round((1 - var_r / var0) * 100, 2),
        "transition_matrix": np.round(P, 4).tolist(),
        "stationary_distribution": [round(v, 4) for v in pi],
        "entropy_initial": round(h0, 4),
        "entropy_stationary": round(h_stat, 4),
        "entropy_reduction_percent": round((1 - h_stat / h0) * 100, 2),
        "mas_per_round": {int(r): round(v, 4) for r, v in mas.items() if pd.notna(v)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="evaluation_data.csv")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--baseline-var", type=float, default=None,
                    help="shared Var(v^0) across models (0.3626 in the paper)")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    os.makedirs(args.output, exist_ok=True)
    summary = analyze(df, args.rounds, args.baseline_var)

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(summary["transition_matrix"], index=STATES, columns=STATES).to_csv(
        os.path.join(args.output, "transition_matrix.csv"))
    pd.DataFrame({"probability": summary["stationary_distribution"]}, index=STATES).to_csv(
        os.path.join(args.output, "stationary_distribution.csv"))

    print(json.dumps({k: v for k, v in summary.items() if k != "mas_per_round"}, indent=2))
    print(f"\nDone. Output: {args.output}/")


if __name__ == "__main__":
    main()
