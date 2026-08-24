"""Ecosystem-level Markov analyses (Section 4.2, Figure 5; Appendix G).

Takes the per-model transition matrices produced by src.analysis and computes:
  1. the average transition matrix P-bar (Eq. 8) and its stationary distribution
  2. the average of the per-model stationary distributions
     (the "Average Stationary Distribution" of Figure 5)
  3. the alpha-restart extension (Appendix G): replace each model kernel P with
         P(alpha) = (1 - alpha) P + alpha 1 pi0^T
     where pi0 is the stance distribution of the human-authored seeds, compute
     the per-model stationary distribution, and average across models.

Example:
  python -m src.ecosystem \
      --inputs runs/*/analysis/transition_matrix.csv \
      --pi0 0.3846,0.3333,0.2821 \
      --alphas 0,0.05,0.1,0.2,0.3,0.5 \
      --output runs/ecosystem
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

STATES = ["Left", "Center", "Right"]


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    v = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
    return v / v.sum()


def entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def restart_kernel(P: np.ndarray, alpha: float, pi0: np.ndarray) -> np.ndarray:
    return (1 - alpha) * P + alpha * np.outer(np.ones(3), pi0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-model transition_matrix.csv files (glob patterns ok)")
    ap.add_argument("--pi0", default="0.3846,0.3333,0.2821",
                    help="seed stance distribution L,C,R (paper: 15/13/11 of 39)")
    ap.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    paths = sorted({p for pat in args.inputs for p in glob.glob(pat)})
    if not paths:
        ap.error("no transition matrices matched --inputs")
    kernels = {p: pd.read_csv(p, index_col=0).values for p in paths}
    pi0 = np.array([float(x) for x in args.pi0.split(",")])
    pi0 = pi0 / pi0.sum()
    alphas = [float(x) for x in args.alphas.split(",")]
    os.makedirs(args.output, exist_ok=True)

    print(f"{len(kernels)} model kernels loaded:")
    for p in paths:
        print(f"  - {p}")

    # -- Figure 5 quantities ------------------------------------------------
    P_bar = np.mean(list(kernels.values()), axis=0)
    pi_of_pbar = stationary_distribution(P_bar)
    mean_pi = np.mean([stationary_distribution(P) for P in kernels.values()], axis=0)

    # -- Appendix G: alpha-restart -----------------------------------------
    h0 = entropy(pi0)
    rows = []
    for a in alphas:
        pis = [stationary_distribution(restart_kernel(P, a, pi0))
               for P in kernels.values()]
        mean = np.mean(pis, axis=0)
        rows.append({
            "alpha": a,
            "Left": round(mean[0], 4), "Center": round(mean[1], 4),
            "Right": round(mean[2], 4),
            "entropy": round(entropy(mean), 4),
            "entropy_loss_percent": round((1 - entropy(mean) / h0) * 100, 2),
        })
    restart_df = pd.DataFrame(rows)

    summary = {
        "models": paths,
        "P_bar": np.round(P_bar, 4).tolist(),
        "stationary_of_P_bar": [round(v, 4) for v in pi_of_pbar],
        "mean_of_per_model_stationaries": [round(v, 4) for v in mean_pi],
        "pi0": [round(v, 4) for v in pi0],
    }
    with open(os.path.join(args.output, "ecosystem_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    restart_df.to_csv(os.path.join(args.output, "alpha_restart.csv"), index=False)

    print("\nP_bar:")
    print(pd.DataFrame(P_bar, index=STATES, columns=STATES).round(3))
    print("\nstationary of P_bar:        ",
          dict(zip(STATES, np.round(pi_of_pbar, 3))))
    print("mean of per-model pi (Fig 5):",
          dict(zip(STATES, np.round(mean_pi, 3))))
    print("\nalpha-restart (Appendix G):")
    print(restart_df.to_string(index=False))
    print(f"\nDone. Output: {args.output}/")


if __name__ == "__main__":
    main()
