# Stage 4 — arbitration decision table and pseudocode

Stage 4 is computed deterministically from stage-1 and stage-3 outputs. The only LLM call in stage 4 is the
targeted recheck (`prompts/stage4_recheck.txt`).

## Inputs

- The candidate pool 𝒞 and the conflict flags (see `conflict_definitions.md`).
- Review actions from the three stage-3 reviewers, A-NEG, A-IMP and A-LIT. Each action names the candidates
  it applies to (`target_tuple_ids`) and may carry a `reason_code`.
- Action types allowed by the review and recheck prompts: KEEP, DROP, FLIP (needs `new_value.polarity`),
  MERGE (needs `new_value.normalized_ref`) and FLAG.

**MERGE.** The prompts allow MERGE, and reviewers use it. Stage 4 counts a MERGE vote as KEEP and never
applies the requested `normalized_ref`, so categories are never rewritten. As a KEEP vote, a MERGE vote
counts toward the majority like any other KEEP vote.

## Order of decisions

For each candidate, stage 4 takes the first step that applies:

| Step | Rule | When | Result | Fig. 6(a) path | `figure_values.csv` item |
|---|---|---|---|---|---|
| 0 | — | candidate in no conflict flag | KEEP | not routed (판정 대상 아님) | `not_routed_default_keep` |
| 1 | Recheck | incomplete FLIP/MERGE vote, or entity-scope conflict without a majority | one LLM action | LLM recheck (LLM 의미 재검토) | `llm_recheck` |
| 2 | Rule 1 | majority | majority action | majority (다수결) | `rule1_majority` |
| 2 | Rule 1′ | majority, but A-LIT dissents on an entity-scope conflict | FLAG (= keep) | A-LIT minority hold (관점 소수 유보) | `rule1prime_alit_minority_hold` |
| 3 | Rule 2 | one FLIP, one DROP, one KEEP | FLIP / DROP / FLAG by reason code | three-way split (분할 규칙) | `rule2_three_way_split` |
| 4 | Rule 3 | any other case without a majority | FLAG (= keep) | no-majority hold (다수 미형성 유보) | `rule3_no_majority_hold` |

## Votes per candidate

For each candidate (tuple_id), take each reviewer's vote v_A-NEG, v_A-IMP, v_A-LIT: the reviewer's last action
that names the candidate. A missing vote counts as KEEP, and a MERGE vote counts as KEEP.
Each candidate gets one conflict type: the type of the *last* flag that contains it (granularity overrides
polarity; see `conflict_definitions.md`).

## Step 1 — targeted LLM recheck (before voting)

A candidate is sent to recheck if either condition holds:

- (a) any reviewer's FLIP/MERGE vote is *incomplete*: `target_tuple_ids` is empty, a FLIP has no
  `new_value.polarity`, or a MERGE has no `new_value.normalized_ref`; or
- (b) its conflict type is `granularity_overlap_candidate` (entity-scope overlap) and no action has ≥ 2 votes.

The recheck is keyed by the *first* flag that contains the candidate, and it runs at most once per distinct
tuple-id set. It sees the candidates of that flag, the candidate's conflict type and the three reviewers'
votes, and returns exactly one action. That action replaces the reviewers' votes for every candidate of the
flag. A MERGE returned by the recheck is read as KEEP.

## Steps 2–4 — decision table (candidates without a recheck action)

Let m = the most frequent action among the three votes and n_m = its count.
On an entity-scope conflict (`granularity_overlap_candidate`), A-LIT is the preferred reviewer; on other
conflict types there is no preferred reviewer.

| Rule | condition on the 3 votes | result | reason_code recorded |
|---|---|---|---|
| **1** majority | n_m ≥ 2, and Rule 1′ does not apply | **m** (KEEP / DROP / FLIP / FLAG) | — |
| **1′** A-LIT minority hold | n_m ≥ 2, entity-scope conflict, and A-LIT cast a vote that differs from m | **FLAG** (= keep) | `FACET_MINORITY_SIGNAL` |
| **2** three-way split | {v_A-NEG, v_A-IMP, v_A-LIT} = {FLIP, DROP, KEEP} | the FLIP vote's code ∈ **structural** {NEGATION_SCOPE, CONTRAST_CLAUSE, STRUCTURAL_INCONSISTENT} → **FLIP**; else the DROP vote's code ∈ **removal-justified** {WEAK_EVIDENCE, REDUNDANT_UPPER_REF} → **DROP**; else **FLAG** | for FLAG: `REDUNDANT_REF_UNCERTAIN` if entity-scope, else `TIE_UNRESOLVED` |
| **3** other no majority | n_m = 1 and not Rule 2 (e.g. FLIP + DROP + FLAG) | **FLAG** (= keep) | `REDUNDANT_REF_UNCERTAIN` if entity-scope, else `POLARITY_UNCERTAIN` |

Notes:

- Reason codes change the outcome only in Rule 2. They are matched as exact strings, so a FLIP vote coded
  `POLARITY_INVERSION` is not structural. The standard list is in `taxonomy/reason_codes.tsv`.
- A FLIP result takes the destination polarity from the first FLIP vote, in the order A-NEG, A-IMP, A-LIT,
  that has a `new_value`.
- Candidates in no flag receive no votes (3 × KEEP), so they are always kept.

## Apply

- DROP removes the candidate.
- FLIP sets p := `new_value.polarity`.
- KEEP and FLAG keep the candidate unchanged.
- MERGE is never applied (see above). Categories are never rewritten.

The MFRA output is the remaining candidate list. Duplicates collapse only at evaluation time, where outputs
are treated as sets.

## Pseudocode

```text
REVIEWERS = [A-NEG, A-IMP, A-LIT]

function ARBITRATE(pool, flags, actions[A-NEG], actions[A-IMP], actions[A-LIT]):
    ctype[tid] ← conflict_type of the last flag containing tid          # granularity overrides polarity
    votes[tid][X] ← last action of reviewer X naming tid, X ∈ REVIEWERS  # missing → None
    norm(v) = KEEP if v is None or v = MERGE else v

    for tid in pool:                                                     # Step 1: recheck
        if any v ∈ votes[tid] is incomplete FLIP/MERGE
           or (ctype[tid] = granularity and max_count(norm(votes[tid])) < 2):
            flag ← first flag containing tid;  key ← sorted(flag.tuple_ids)
            if key already rechecked: continue
            r ← LLM_RECHECK(text, ctype[tid], candidates in flag, votes[tid])   # MERGE → KEEP
            if r ≠ None: recheck[t] ← r for all t ∈ flag; mark key

    decisions ← []
    for tid in pool:
        if recheck[tid]: decisions.add(recheck[tid]); continue
        v ← [norm(votes[tid][X]) for X in REVIEWERS]
        (m, n) ← most_common(v)
        if n ≥ 2:                                                        # Step 2
            if ctype[tid] = granularity and votes[tid][A-LIT] ≠ None and norm(votes[tid][A-LIT]) ≠ m:
                decisions.add(FLAG, FACET_MINORITY_SIGNAL)               # Rule 1′ A-LIT minority hold
            else: decisions.add(m)                                       # Rule 1 majority
        elif set(v) = {FLIP, DROP, KEEP}:                                # Step 3: Rule 2 three-way split
            if code(FLIP vote) ∈ STRUCTURAL:        decisions.add(FLIP)
            elif code(DROP vote) ∈ DROP_JUSTIFIED:  decisions.add(DROP)
            else: decisions.add(FLAG, granularity ? REDUNDANT_REF_UNCERTAIN : TIE_UNRESOLVED)
        else:                                                            # Step 4: Rule 3 other no majority
            decisions.add(FLAG, granularity ? REDUNDANT_REF_UNCERTAIN : POLARITY_UNCERTAIN)
    return APPLY(pool, decisions)          # DROP removes; FLIP sets polarity; KEEP/FLAG/MERGE keep
```
