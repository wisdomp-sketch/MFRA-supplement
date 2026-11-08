# Structural conflict definitions (MFRA stage 2 → stage 3)

Conflicts are detected by code only; no LLM call is involved. The semantic near-duplicate check was switched
off in all runs (see "What is not a conflict").

## Notation

For one sentence, stage 1 returns three record lists, one per agent (A-NEG, A-IMP, A-LIT).
Stage 2 concatenates them into the candidate pool

  𝒞 = { x_i = (t_i, c_i, p_i, a_i) },  i = 0 … n−1,  tuple_id = `t{i}`

- t = target surface (`aspect_term`, "" if implicit), c = category `E#A` (`aspect_ref`),
  p = polarity, a ∈ {A-NEG, A-IMP, A-LIT} = origin agent.
- Order: all records of A-NEG, then A-IMP, then A-LIT, in each agent's output order.
- **No de-duplication.** If two agents return the same (t, c, p), the pool holds two candidates with
  different tuple_ids. Within-agent duplicates are also kept.
- **No filtering.** Records whose category is outside the runtime validator's 25 pairs are kept; only their
  `confidence` is halved (`invalid_ref_flag`). The same applies to targets containing Latin letters
  or targets that look like opinion words.
- Strings are only trimmed (`strip`) when conflicts are detected. Polarity is compared as returned
  (positive / neutral / negative).

Write E(c) and Attr(c) for the entity and attribute parts of c = E#A.

## The three conflict types

A flag is a set of tuple_ids plus a type. A flag routes candidates to review; it does not say which candidate is wrong.

| # | type (`conflict_type`) | paper name | condition (per sentence) | flagged tuple_ids |
|---|---|---|---|---|
| 1 | `ref_polarity_mismatch` | category-level polarity conflict | for a category c ≠ "": the group 𝒢_c = {x ∈ 𝒞 : c_x = c} contains ≥ 2 distinct non-empty polarities | all of 𝒢_c |
| 2 | `term_polarity_mismatch` | target-level polarity conflict | only among candidates with c = "": for a target t ≠ "", the group 𝒯_t = {x : c_x = "", t_x = t} contains ≥ 2 distinct polarities | all of 𝒯_t |
| 3 | `granularity_overlap_candidate` | entity-scope (granularity) overlap | among candidates with '#' in c and p ≠ "": for a pair (α, p), the group 𝒦_{α,p} = {x : Attr(c_x) = α, p_x = p} contains E = 제품 전체 (upper) **and** at least one other entity (lower: 본품, 패키지·구성품, 브랜드) | all of 𝒦_{α,p} |

Formally:

- (1) ∃ c ≠ "" : |{ p_x : x ∈ 𝒞, c_x = c, p_x ≠ "" }| ≥ 2
- (2) ∃ t ≠ "" : |{ p_x : x ∈ 𝒞, c_x = "", t_x = t, p_x ≠ "" }| ≥ 2
- (3) ∃ (α, p) : 제품 전체 ∈ { E(c_x) : x ∈ 𝒦_{α,p} } and { E(c_x) : x ∈ 𝒦_{α,p} } ∖ {제품 전체} ≠ ∅

Type 2 is effectively unused in the paper runs. The stage-1 contract requires a category on every record,
so candidates with c = "" are rare.

A sentence counts as having a *structural conflict* if at least one flag is raised. This was the case for
16.8% of sentences (5-run mean): 468.0 / 2,794 (16.75%) and 465.6 / 2,764 (16.85%).

## Unit of judgment

- **Unit = candidate (tuple_id), not a unique record.** Each reviewer's action applies to the tuple_ids it
  lists in `target_tuple_ids`. Votes are collected per tuple_id. If a reviewer lists the same tuple_id in
  several actions, its last action counts.
- **Actionable candidates** are the tuple_ids that appear in any flag. All three reviewers are called for every
  sentence (also when there is no flag). The Meta-v1 block in the user message then says
  "No conflict flags. Do not emit review actions."
- **Non-flagged candidates** receive no votes. In stage 4 they take the default KEEP path, so stage 3–4 never
  removes or changes them.
- **Overlapping flags.** A candidate can belong to more than one flag. Stage 4 needs one conflict type per
  candidate, and it takes the type of the *last* flag that contains the candidate. Flags are generated in
  the order type 1 → type 2 → type 3, so a granularity flag overrides a polarity flag. This matters for the
  A-LIT minority hold (Rule 1′) and for the FLAG reason codes (see `arbitration_table.md`).
- The Meta-v1 user block (`{actionable_review_block}` in `prompts/stage3_review.txt`) gives each flag a review
  dimension and fixed fields. This is done by code, not by the LLM:
  - type 1: dimension POLARITY, fixed field `aspect_ref`
  - type 2: dimension POLARITY, fixed field `aspect_term`
  - type 3: dimension ENTITY_SCOPE, fixed fields = the shared attribute and polarity

  The block also lists each flagged key with its unique-agent support count. Agreement counts are given as
  context; the prompt says they are not proof of correctness.

## What is *not* a conflict

- Different targets with the same category and the same polarity.
- The same opinion returned by several agents (duplicates). These are scored once, because evaluation uses sets.
- Opposite polarities on different categories.
- Semantic near-duplicate targets (`semantic_conflict_candidate`). This check was switched off in all runs.
- Validator risks. The `validator_risks` list passed to the reviewers is empty in all runs.
