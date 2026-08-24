"""Per-round stance judging (Section 3.2, Eq. 2-3).

Reads the wide paraphrase CSV produced by src.loop and judges every round
(round 0 = the original article) with the LLM judge at temperature 0.

Output: a long-format evaluation_data.csv with columns
  original_id, round, original_stance, predicted, score, reasoning

The script is resumable: existing (original_id, round) pairs in the output
file are skipped, so an interrupted run can simply be restarted.

Example:
  python -m src.judge --input runs/gpt4o/paraphrase.csv \
      --output runs/gpt4o/evaluation_data.csv --rounds 20
"""

import argparse
import os
import time

import pandas as pd

from .providers import Chat, JUDGE_TEMPERATURE
from . import prompts

API_SLEEP_SECONDS = 0.2


def classify(score: float, threshold: float) -> str:
    if score < -threshold:
        return "Left"
    if score > threshold:
        return "Right"
    return "Center"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="wide paraphrase.csv")
    ap.add_argument("--output", required=True, help="long evaluation_data.csv")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "gemini"])
    ap.add_argument("--judge-model", default="gpt-4o-2024-11-20")
    args = ap.parse_args()

    judge = Chat(args.judge_provider, args.judge_model)
    df = pd.read_csv(args.input)

    done = set()
    results = []
    if os.path.exists(args.output):
        prev = pd.read_csv(args.output)
        results = prev.to_dict("records")
        done = set(zip(prev["original_id"], prev["round"]))
        print(f"Resuming: {len(done)} (article, round) pairs already judged")

    total_calls = 0
    for idx, row in df.iterrows():
        summary = row["Summary_from_the_AllSides_News_Team"]
        for r in range(0, args.rounds + 1):
            if (idx, r) in done:
                continue
            if r == 0:
                title, content = row["Title"], row["Content"]
            else:
                tcol, ccol = f"Round{r}_Title", f"Round{r}_Content"
                if tcol not in row or pd.isna(row[tcol]):
                    break  # article dropped out mid-loop (kept out of all rounds)
                title, content = row[tcol], row[ccol]

            res = judge.complete_json(
                prompts.JUDGE_SYSTEM, prompts.judge_user(summary, title, content),
                temperature=JUDGE_TEMPERATURE,
            )
            if res is None:
                print(f"  ! judge failed for article {idx} round {r}; rerun to resume")
                continue
            score = float(res.get("score", 0.0))
            results.append({
                "original_id": idx,
                "round": r,
                "original_stance": row["Stance"],
                "predicted": classify(score, args.threshold),
                "score": score,
                "reasoning": res.get("reasoning", ""),
            })
            total_calls += 1
            if total_calls % 25 == 0:
                pd.DataFrame(results).to_csv(args.output, index=False,
                                             encoding="utf-8-sig")
                print(f"  [{total_calls} calls] checkpoint saved")
            time.sleep(API_SLEEP_SECONDS)

    pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()
