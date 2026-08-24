# Seed Dataset

The 39 seed articles used in the paper are **not redistributed** here: they are
copyrighted news articles collected from outlets indexed by
[AllSides](https://www.allsides.com). This document describes the schema and
the collection procedure so the dataset can be reconstructed.

## Schema

`src.loop` and `src.judge` require a CSV with at least these columns
(see `example_seed.csv` for a synthetic one-row example):

| Column | Description |
|---|---|
| `Topic` | One of the seven policy topics (Climate Change, Economy and Jobs, Environment, Healthcare, Immigration, Race and Racism, Terrorism) |
| `Head_Line` | Event headline shared by the L/C/R articles of the same event |
| `Summary_from_the_AllSides_News_Team` | Neutral factual summary of the event, describing the characteristic Left and Right framings. **Shown only to the judge** as a grounding reference ($e_i$ in the paper); never shown to the generator. |
| `Media_Name` | Source outlet |
| `Title` | Article title ($c_i$, round 0) |
| `Content` | Article body text ($c_i$, round 0) |
| `Stance` | Human-labeled stance: `Left`, `Center`, or `Right` ($y_i$) |

## Collection procedure (paper Section 3.1, Appendix A)

1. Select events with pronounced partisan divergence across the seven topics
   (guided by Pew Research Center analyses).
2. For each event, collect coverage from outlets rated Left / Center / Right
   by AllSides; the outlet's AllSides media-bias rating provides the stance
   label, and the AllSides story page provides the neutral event summary.
3. Filter with expert curation: keep only events with clear, robust, and
   consistent partisan divergence in framing (researchers with at least a
   Bachelor's degree in the humanities or social sciences).

The paper's final set: 39 articles (15 Left / 13 Center / 11 Right) covering
13 events across the seven topics.
