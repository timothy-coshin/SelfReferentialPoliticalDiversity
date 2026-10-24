# How Do Self-Referential LLM Loops Contract Political Diversity?

[![EMNLP 2026](https://img.shields.io/badge/EMNLP%202026-Main%20Conference-4b8bbe)](https://aclanthology.org/2026.emnlp-main.876/)
[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-ed1c24)](https://aclanthology.org/2026.emnlp-main.876/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Code for the EMNLP 2026 Main Conference paper
**[How Do Self-Referential LLM Loops Contract Political Diversity?](https://aclanthology.org/2026.emnlp-main.876/)**
(Wooyoung Kim and Wooju Kim, Yonsei University).

The pipeline recursively paraphrases political news articles with an LLM,
judges the political stance of every round with an LLM judge, and analyzes the
resulting stance dynamics as a discrete-time Markov process.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # judge + OpenAI generators
export ANTHROPIC_API_KEY=...     # Claude generators
export GOOGLE_API_KEY=...        # Gemini generators
```

For open-weight generators run locally, additionally install
`torch`, `transformers`, and `accelerate` (see `requirements.txt`).

**Data.** The seed articles are not redistributed (copyright); see
[`data/README.md`](data/README.md) for the schema and collection procedure.
All commands below assume your seed CSV is at `data/seeds.csv`.

## Pipeline

### 1. Recursive paraphrasing loop (Section 3.2)

One run per generator model. 20 rounds, temperature 0.8.

```bash
python -m src.loop --provider openai --model gpt-4o-2024-11-20 \
    --input data/seeds.csv --rounds 20 --mode vanilla --output-dir runs/gpt4o
```

Providers/models used in the paper (Appendix B):

| Short name | `--provider` | `--model` |
|---|---|---|
| GPT5.1 | `openai` | `gpt-5.1-2025-11-13` |
| GPT4o | `openai` | `gpt-4o-2024-11-20` |
| Gemini2.5 | `gemini` | `gemini-2.5-flash` |
| Sonnet4.5 | `anthropic` | `claude-sonnet-4-5-20250929` |
| Exaone3.5 | `local` | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` |
| LLaMa3.1 | `local` | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma3 | `local` | `google/gemma-3-12b-it` |
| Qwen3 | `local` | `Qwen/Qwen3-14B` |

Mitigation modes (Section 4.3): `--mode sisp`, `--mode scg-gt`, and
`--mode scg-pred` (the latter judges the previous round each step and
conditions generation on the predicted label; requires `OPENAI_API_KEY`).

Mixed-model ecosystems (Figure 6):

```bash
python -m src.loop --mode mix \
    --provider openai --model gpt-5.1-2025-11-13 \
    --provider2 openai --model2 gpt-4o-2024-11-20 --ratio 0.5 \
    --input data/seeds.csv --rounds 20 --output-dir runs/mix_50_50
```

### 2. Judge every round (Eq. 2–3)

GPT-4o judge, temperature 0, threshold tau = 0.3. Resumable; rerun the same
command after an interruption.

```bash
python -m src.judge --input runs/gpt4o/paraphrase.csv \
    --output runs/gpt4o/evaluation_data.csv --rounds 20
```

### 3. Per-model analysis (Tables 1–2)

```bash
python -m src.analysis --input runs/gpt4o/evaluation_data.csv \
    --output runs/gpt4o/analysis --rounds 20 --baseline-var 0.3626
```

Reports the final stance distribution, the Stuart–Maxwell test, VRR (Eq. 4,
sample variance with the shared round-0 baseline), MAS per round (Eq. 5),
the pooled transition matrix (Figure 3), and the stationary distribution with
its political entropy (Table 2). `--baseline-var` is the shared `Var(v^0)`
measured once on the original seed articles (0.3626 in the paper); to
reproduce it for your own seeds, judge the seed CSV at round 0 and take the
sample variance of the scores.

### 4. Ecosystem analyses (Figure 5; Appendix G)

```bash
python -m src.ecosystem \
    --inputs "runs/*/analysis/transition_matrix.csv" \
    --pi0 0.3846,0.3333,0.2821 \
    --alphas 0,0.05,0.1,0.2,0.3,0.5 \
    --output runs/ecosystem
```

Computes the average transition matrix and its stationary distribution, the
average of per-model stationary distributions (Figure 5), and the
alpha-restart table with human-content injection (Appendix G).

### 5. Figures

```bash
python -m src.plotting \
    --inputs runs/gpt51/evaluation_data.csv runs/gpt4o/evaluation_data.csv \
    --labels GPT5.1 GPT4o --rounds 20 --output figs/
```

## Base vs. alignment-trained checkpoints (Appendix E)

Base checkpoints have no chat template, so both members of each pair are run
with the same few-shot completion prompt (`--prompt-style fewshot`), isolating
the weights as the only difference within a pair:

```bash
python -m src.loop --provider local --model meta-llama/Llama-3.1-8B \
    --prompt-style fewshot --input data/seeds.csv --rounds 20 \
    --output-dir runs/llama_base

python -m src.loop --provider local --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt-style fewshot --input data/seeds.csv --rounds 20 \
    --output-dir runs/llama_instruct_fs
```

Then judge (step 2) and compare VRR (step 3) using a shared round-0 baseline
judged once from the original seeds.

## Repository layout

```text
src/prompts.py     all prompts, verbatim (paper Appendix K)
src/providers.py   unified OpenAI / Anthropic / Gemini / local-HF client
src/loop.py        recursive paraphrasing (vanilla, SCG, SISP, mix)
src/judge.py       per-round stance judging (long-format output)
src/analysis.py    Tables 1–2 metrics, transition matrix, stationary analysis
src/ecosystem.py   ecosystem average, alpha-restart (Figure 5, Appendix G)
src/plotting.py    Figures 1–4
data/README.md     seed dataset schema and collection procedure
```

## Notes for faithful reproduction

- Generation temperature 0.8; judge temperature 0.0; threshold tau = 0.3;
  R = 20 rounds.
- Generation input content is truncated to 2,000 characters and judge input
  to 3,500 characters, as in the original experiments.
- Variances are sample variances (ddof = 1); all models share the same
  round-0 variance baseline.
- Judge calls are retried with exponential backoff. Do not disable this:
  silently dropped judge calls bias round-level distributions.
- Stochasticity: generation at temperature 0.8 means per-run numbers vary;
  the paper's qualitative split (Preserving vs. Homogenizing) is robust, but
  exact table values will differ across reruns.

## Citation

```bibtex
@inproceedings{kim-kim-2026-self,
    title = "How Do Self-Referential {LLM} Loops Contract Political Diversity?",
    author = "Kim, Wooyoung  and
      Kim, Wooju",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.876/",
    pages = "20246--20263",
    ISBN = "979-8-89176-459-0",
    abstract = "As Large Language Models (LLMs) increasingly consume their own outputs through retrieval-augmented generation and synthetic data pipelines, self-referential loops are becoming prevalent in modern information ecosystems. While prior work has studied variance collapse during recursive training, the dynamic consequences for political diversity during inference remain underexplored. We investigate this by conducting iterative paraphrasing experiments across eight widely used LLMs, and formalize self-referential generation as a discrete-time Markov process to characterize its long-term behavior. Our results reveal a systematic contraction of political diversity, with several models converging toward center-dominant equilibria. Notably, this homogenization is strongly front-loaded: the majority of diversity loss occurs within the earliest iterative rounds, indicating that even a few AI-mediated transformations can disproportionately narrow the political spectrum. We further examine two interventions: Stance-Conditioned Generation (SCG) substantially reduces marginal stance-distribution drift even without ground-truth labels, leaving no statistically detectable shift for three of the four homogenizing models, whereas Self-Identified Stance Preservation (SISP) overshoots into over-polarization; even SCG improves but does not guarantee instance-level stance retention. These findings highlight a structural risk associated with self-referential generation in AI-driven information pipelines. Our code and prompts are publicly available at https://github.com/timothy-coshin/SelfReferentialPoliticalDiversity."
}
```
