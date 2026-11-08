"""Paired sentence-level bootstrap with Holm correction — the paper's two test families.

    python bootstrap_holm.py [--all] [--counts ../results/sentence_counts.csv] [--out tests.csv]

Procedure (both families, run on the same resamples):
  1. For each system and sentence, average TP/FP/FN over the 5 runs.
  2. Draw B = 10,000 multinomial sentence weights, numpy default_rng(20261020).multinomial(N, 1/N),
     with sentences in the order of sentence_counts.csv.
  3. For each resample, compute micro P/R/F1/F2 from the weighted summed counts of both systems;
     Δ = MFRA − comparison system.
  4. 95% CI = 2.5/97.5 percentiles of Δ; two-sided p = 2·min(P(Δ≤0), P(Δ≥0)); p = 0 is reported as 1/B.
  5. Holm step-down correction inside each family only.

Family 1 — main comparisons (36): MFRA vs {S0, REPEAT_UNION, SR3, POOL_GE1} × {TASD, ATSA, ACSA} × {P, R, F1}
Family 2 — ≥2 operating point (12): MFRA vs GE2 × {TASD, ATSA, ACSA} × {P, R, F1, F2}
           (GE2 = keys produced by ≥2 stage-1 agents, agreement re-counted per unit)
Requires numpy.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import load_counts  # noqa: E402

B = 10_000
SEED = 20261020
UNITS = ("TASD", "ATSA", "ACSA")
FAMILIES = {
    "main_36": {"others": ("S0", "REPEAT_UNION", "SR3", "POOL_GE1"), "metrics": ("P", "R", "F1")},
    "ge2_12": {"others": ("GE2",), "metrics": ("P", "R", "F1", "F2")},
}
HERE = Path(__file__).resolve().parent


def prf(arr):
    tp, fp, fn = arr[..., 0], arr[..., 1], arr[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
        f2 = np.where(4 * p + r > 0, 5 * p * r / (4 * p + r), 0.0)
    return {"P": p, "R": r, "F1": f1, "F2": f2}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=str(HERE.parent / "results" / "sentence_counts.csv"))
    ap.add_argument("--all", action="store_true", help="all 2,794 sentences (default 2,764)")
    ap.add_argument("--out")
    a = ap.parse_args()
    order, data, _ = load_counts(Path(a.counts), a.all)
    N = len(order)
    runs = sorted({r for _, r in data})

    def mean_counts(system, unit):  # (N, 3), averaged over runs
        return np.mean([np.array([data[(system, r)][f"{unit}_{x}"] for x in ("tp", "fp", "fn")], float).T for r in runs], axis=0)

    W = np.random.default_rng(SEED).multinomial(N, np.full(N, 1.0 / N), size=B).astype(np.float64)
    rows = []
    for fam, spec in FAMILIES.items():
        fam_rows = []
        for other in spec["others"]:
            for u in UNITS:
                a_, b_ = mean_counts("MFRA", u), mean_counts(other, u)
                pa, pb = prf(a_.sum(0)), prf(b_.sum(0))
                ba, bb = prf(W @ a_), prf(W @ b_)
                for m in spec["metrics"]:
                    d = ba[m] - bb[m]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    tail = min((d <= 0).mean(), (d >= 0).mean())
                    p = 1.0 / B if tail == 0 else min(1.0, 2 * tail)
                    fam_rows.append({"family": fam, "comparison": f"MFRA vs {other}", "unit": u, "metric": m,
                                     "mfra": float(pa[m]), "other": float(pb[m]), "diff": float(pa[m] - pb[m]),
                                     "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p), "p_floor": bool(tail == 0)})
        for r, h in zip(fam_rows, holm([r["p"] for r in fam_rows])):
            r["p_holm"] = h
        rows += fam_rows

    print(f"N = {N}, B = {B}, seed = {SEED}; Δ = MFRA − other (pooled run-mean counts)")
    print("family,comparison,unit,metric,MFRA,other,diff,ci_lo,ci_hi,p,p_holm")
    for r in rows:
        print(f"{r['family']},{r['comparison']},{r['unit']},{r['metric']},{r['mfra']:.4f},{r['other']:.4f},"
              f"{r['diff']:+.4f},{r['ci_lo']:+.4f},{r['ci_hi']:+.4f},{r['p']:.4g},{r['p_holm']:.4g}")
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
