# MFRA supplement

Supplementary material for *A Multi-Facet Refinement Architecture for Improving Recall in LLM-based
Zero-shot Target-Aspect-Sentiment Detection of Korean Reviews*.

This repository contains what is needed to recompute the reported results and to re-implement the
method: the prompts, the category set, the conflict and arbitration rules, the evaluator, and
per-sentence count files.

Tables 3–4, Figs. 3–4 and the significance tests can be recomputed from `metrics_by_run.csv` and
`sentence_counts.csv`; Tables 5–6 and Fig. 7(a) from `error_counts.csv` (classification code in
`evaluation/errors.py`). Figs. 5, 6 and 7(b) and the process-level figures in Section IV.2 depend on
intermediate outputs (per-call predictions and stage-2 candidates) that contain corpus text; their
aggregated values are given in `figure_values.csv`. Predictions themselves are not released for the
same reason.

The corpus is not redistributed. No file contains sentence text or spans taken from it; sentences are
referred to by ID only.

## Data

- **Corpus:** National Institute of Korean Language, 2022 Artificial Intelligence Language Ability
  Evaluation Corpus: ABSA (Version 1.0), validation split, 2,794 sentences.
  Apply for access at https://kli.korean.go.kr
- **Evaluation set:** 2,764 sentences. The 30 sentences listed in `splits/excluded_30_ids.txt` were used to
  check prompt outputs during development and are excluded from evaluation.
- To confirm that your copy matches the gold data used in the paper:

  ```
  python evaluation/score.py gold-fingerprint --gold <validation.jsonl>
  ```

## Run settings

| item | value |
|---|---|
| model | `gpt-4o-mini` (OpenAI API) |
| run period | September 2026 |
| decoding | temperature 0, JSON output, API seed not set |
| demonstrations | none (zero-shot) |
| repetitions | 5 independent runs per condition |
| MFRA stage 4 | rule-based; computed deterministically from stage-1 and stage-3 outputs |

## Conditions

| paper | `system` in result files |
|---|---|
| Single-pass zero-shot | `S0` |
| Budget-matched repeated inference (union of 7 calls) | `REPEAT_UNION` |
| Three-step self-refinement | `SR3` |
| MFRA (final output) | `MFRA` |
| Stage-2 candidate pool | `POOL_GE1` |
| ≥2 / ≥3 agreement operating points | `GE2`, `GE3` |
| A-NEG, A-IMP, A-LIT (stage-1 agents) | `P_NEG`, `P_IMP`, `P_LIT` |

## Layout

```
prompts/        prompt pack as sent to the API
  s0_single_pass.txt                 single-pass zero-shot (also used by repeated inference and SR3 call 1)
  sr3_feedback.txt, sr3_refine.txt
  stage1_shared_contract.txt         MFRA stage-1 template
  facet_neg.txt, facet_imp.txt, facet_lit.txt
  stage3_review.txt                  stage-3 review template and per-agent values
  stage4_recheck.txt
  assemble.py, SHA256SUMS.txt        rebuild the agent prompts and verify them
taxonomy/
  categories_21.tsv                  21 Entity#Attribute categories
  reason_codes.tsv                   review reason codes and their arbitration class
rules/
  conflict_definitions.md            conflict types in (t, c, p) notation
  arbitration_table.md               stage-4 decision table and pseudocode
evaluation/
  normalize.py, score.py             normalization, exact match, unit projection, ≥k agreement
  bootstrap_holm.py                  paired bootstrap with Holm correction
  errors.py                          error classification (error types, target-extraction errors)
splits/excluded_30_ids.txt
results/
  metrics_by_run.csv                 sentence_set × system × unit × run: TP/FP/FN, P/R/F1/F2
  sentence_counts.csv                sentence ID × run × system: TP/FP/FN per unit
  error_metrics_by_run.csv           sentence_set × system × run: error-analysis values
  error_counts.csv                   sentence ID × run × system: error flags and counts
  figure_values.csv                  aggregated values for Figs. 5–7(b) and Section IV.2
```

In each prompt file, the text before `---USER---` is the system message and the rest is the user message.
Lower-case placeholders such as `{text}` are filled at run time.
Prompt files keep the internal agent names P-NEG/P-IMP/P-LIT used during the runs (A-NEG/A-IMP/A-LIT in the paper).

`metrics_by_run.csv`, `error_metrics_by_run.csv` and `figure_values.csv` also contain rows for all 2,794
sentences (`sentence_set = all_2794`) for reference.

## Reproducing the results

Requires Python ≥ 3.9 and numpy; `errors.py` also needs scipy.

```
python prompts/assemble.py               # verify prompts
python evaluation/score.py tables        # all systems and units (mean and SD over 5 runs)
python evaluation/bootstrap_holm.py      # significance tests (two families, Holm within each)
python evaluation/errors.py tables       # error analyses (mean and SD over 5 runs)
```

Each `tables` command takes `--all` for the 2,794-sentence set.

To score new predictions (JSON lines, `{"id": ..., "tuples": [{"aspect_term", "aspect_ref", "polarity"}]}`):

```
python evaluation/score.py score --gold <validation.jsonl> --pred <pred.jsonl> --exclude splits/excluded_30_ids.txt
```

Add `--agree k` with stage-1 agent outputs to compute a ≥k operating point.

To classify the errors of new predictions (needs the corpus file, for the sentences and gold offsets):

```
python evaluation/errors.py classify --gold <validation.jsonl> --pred <pred.jsonl> --out errors.csv
```

## Scoring

- Each record is (target t, category c, polarity p) after normalization; an empty t is an implicit target.
  Each sentence's output is treated as a set.
- TASD = (t, c, p), ATSA = (t, p), ACSA = (c, p). TP, FP and FN are summed over sentences (micro).
- Explicit/implicit analysis splits gold and predictions by whether t is empty and matches within each group.
- ≥k agreement is counted separately for each unit (pairs for ATSA and ACSA).
- F2 = 5PR / (4P + R).

## Error analyses

Computed per sentence on the prediction list (duplicates kept) for S0, REPEAT_UNION, SR3 and MFRA.
Full rules are in the docstring of `evaluation/errors.py`.

- Gold and predicted records are aligned: exact matches first, then maximum-weight bipartite matching.
- Error types (multi-label, per sentence): TD (target), ACD (category), SD (sentiment), MOD (multi-opinion,
  only for sentences with ≥ 2 gold records). Rates are the share of sentences with the flag.
- Target-extraction errors, for explicit gold targets: Missed (FR−), Spurious (FR+), Under-extended (FB−),
  Over-extended (FB+), other boundary (FBX), segmentation (FS−, FS+). Rates are per explicit gold record.
- Spurious predictions are split into (a) same category as an implicit gold record in the sentence,
  (b) target string contained in, or containing, an explicit gold target, (c) other.

## Figure values

`figure_values.csv` is in long format: `figure, panel, sentence_set, unit, item, run, run_label, measure, value`.
Values are per run (`run` 1–5) unless `run = mean`.

- **Fig5**: first-k union of the 7 single-pass calls (`repeat_union_k1` … `k7`) and unions of two stage-1
  agents; TP/FP/FN, precision, recall, F1. The other curve points (single agents, all three agents) are
  `P_*` and `POOL_GE1` in `metrics_by_run.csv`.
- **Fig6 a**: per arbitration path (not routed, LLM recheck, Rule 1, Rule 1′, Rule 2, Rule 3; names as in
  `rules/arbitration_table.md`): candidates
  (TASD only), lost TP, removed FP, added FP, recovered TP, from the stage-2 pool to the final output.
- **Fig6 b**: totals of the same changes, and the raw record counts of the stage-2 pool and the final output.
- **Fig6 c**: the same changes by conflict type (a candidate in several flag types is counted in each).
- **Fig7 b**: per entity: gold count, recall of S0 / stage-2 pool / MFRA, missed gold and its share of all
  missed gold (MFRA), and gold missed by the pool in all 5 runs. Mean over runs only (`run = mean`).
- **IV.2.1 / IV.2.2**: TASD gold recovered by one agent only and by all three (share of recovered gold);
  sentences whose 7 repeated outputs are identical; sentences with a structural conflict; stage-2
  candidates routed to arbitration; removed FP per lost TP (per run, and as the ratio of run means).

## License and citation

Released under the MIT License (see `LICENSE`).
If you use this material, please cite the paper (citation to be added on publication).
