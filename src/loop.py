"""Recursive paraphrasing loop (Section 3.2 / 4.3 of the paper).

Modes:
  vanilla   : c^r = G(c^{r-1})                       (w/o control)
  scg-gt    : c^r = G(c^{r-1}, y)     with y  = ground-truth stance
  scg-pred  : c^r = G(c^{r-1}, y_hat) with y_hat = judge label of round r-1
  sisp      : chain-of-thought self-identified stance preservation
  mix       : each round randomly picks one of two generators (Figure 6)

Input CSV must contain columns:
  Title, Content, Stance, Summary_from_the_AllSides_News_Team
(see data/README.md for the full schema).

Output: <output_dir>/paraphrase.csv - the input rows plus
  Round{r}_Title, Round{r}_Content            for r = 1..R
  Round{r}_CondLabel                          (scg-pred only)
  Round{r}_Model                              (mix only)

Example:
  python -m src.loop --provider openai --model gpt-4o-2024-11-20 \
      --input data/seeds.csv --rounds 20 --mode vanilla --output-dir runs/gpt4o
"""

import argparse
import json
import os
import random
import time

import pandas as pd

from .providers import Chat, GEN_TEMPERATURE, JUDGE_TEMPERATURE
from . import prompts

API_SLEEP_SECONDS = 0.5


def classify(score: float, threshold: float) -> str:
    if score < -threshold:
        return "Left"
    if score > threshold:
        return "Right"
    return "Center"


def judge_once(judge: Chat, summary: str, title: str, content: str, threshold: float) -> str:
    res = judge.complete_json(
        prompts.JUDGE_SYSTEM, prompts.judge_user(summary, title, content),
        temperature=JUDGE_TEMPERATURE,
    )
    if res is None:
        return "Center"
    return classify(float(res.get("score", 0.0)), threshold)


def generate_once(gen: Chat, mode: str, title: str, content: str, stance: str | None):
    if gen.provider == "local" and gen.prompt_style == "fewshot":
        # Base checkpoints: single few-shot completion prompt (Appendix E).
        return gen.complete_json("", prompts.fewshot_completion_prompt(title, content),
                                 temperature=GEN_TEMPERATURE)
    if mode in ("vanilla", "mix"):
        return gen.complete_json(prompts.PARAPHRASE_SYSTEM,
                                 prompts.paraphrase_user(title, content),
                                 temperature=GEN_TEMPERATURE)
    if mode in ("scg-gt", "scg-pred"):
        return gen.complete_json(prompts.scg_system(stance),
                                 prompts.scg_user(title, content, stance),
                                 temperature=GEN_TEMPERATURE)
    if mode == "sisp":
        return gen.complete_json(prompts.SISP_SYSTEM,
                                 prompts.sisp_user(title, content),
                                 temperature=GEN_TEMPERATURE)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True,
                    choices=["openai", "anthropic", "gemini", "local"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-style", default="chat", choices=["chat", "fewshot"],
                    help="local provider only: 'fewshot' for base checkpoints")
    ap.add_argument("--input", default="data/seeds.csv")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--mode", default="vanilla",
                    choices=["vanilla", "scg-gt", "scg-pred", "sisp", "mix"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="stance classification threshold tau (scg-pred)")
    ap.add_argument("--judge-provider", default="openai", choices=["openai", "gemini"],
                    help="scg-pred only")
    ap.add_argument("--judge-model", default="gpt-4o-2024-11-20",
                    help="scg-pred only")
    # mix mode
    ap.add_argument("--provider2", default=None)
    ap.add_argument("--model2", default=None)
    ap.add_argument("--ratio", type=float, default=0.5,
                    help="mix mode: probability of picking the first generator")
    ap.add_argument("--seed", type=int, default=None,
                    help="mix mode: random seed for model selection")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "paraphrase.csv")
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in vars(args).items()}, f, indent=2)

    gen = Chat(args.provider, args.model, prompt_style=args.prompt_style)
    gen2 = None
    if args.mode == "mix":
        if not (args.provider2 and args.model2):
            ap.error("--mode mix requires --provider2 and --model2")
        gen2 = Chat(args.provider2, args.model2)
        if args.seed is not None:
            random.seed(args.seed)

    judge = None
    if args.mode == "scg-pred":
        judge = Chat(args.judge_provider, args.judge_model)

    df = pd.read_csv(args.input)
    rows = []
    for idx, row in df.iterrows():
        print(f"[{idx + 1}/{len(df)}] {str(row['Title'])[:40]}...")
        data = row.to_dict()
        title, content = row["Title"], row["Content"]
        cond_label = row["Stance"]  # scg-gt uses the ground-truth label every round

        for r in range(1, args.rounds + 1):
            if args.mode == "scg-pred":
                # Condition on the judge's label of the previous round's text.
                cond_label = judge_once(judge, row["Summary_from_the_AllSides_News_Team"],
                                        title, content, args.threshold)
                data[f"Round{r}_CondLabel"] = cond_label

            active = gen
            if args.mode == "mix":
                active = gen if random.random() < args.ratio else gen2
                data[f"Round{r}_Model"] = active.model

            res = generate_once(active, args.mode, title, content, cond_label)
            if not (res and "p_title" in res and "p_content" in res):
                print(f"  ! generation failed at round {r}; stopping this article")
                break
            title, content = res["p_title"], res["p_content"]
            data[f"Round{r}_Title"] = title
            data[f"Round{r}_Content"] = content

            if args.provider != "local":
                time.sleep(API_SLEEP_SECONDS)

        rows.append(data)
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nDone. Output: {out_csv}")


if __name__ == "__main__":
    main()
