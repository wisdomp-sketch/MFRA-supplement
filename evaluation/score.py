"""Exact-match scoring used in the paper (standalone, standard library only).

Units (projections of the normalized record (t, c, p)):
  TASD = (t, c, p)  records with empty c are dropped
  ATSA = (t, p)     t = "" for an implicit target
  ACSA = (c, p)     records with empty c are dropped
Per sentence, gold and prediction are SETS of projected keys; TP = |G∩P|, FP = |P−G|, FN = |G−P|.
A sentence whose projected gold set is empty is not scored (its predictions add no FP). In the paper data every
sentence has gold in TASD, ATSA and ACSA, so this applies only to the explicit/implicit groups below.
Micro P/R/F1/F2 over the summed counts; tables report the mean (SD) over the 5 runs.

Explicit/implicit split (TASD, ATSA): gold and prediction keys are split by whether t == "";
matching is done only inside a group; a sentence without gold keys in that group is not scored for it.

≥k agreement: for each unit, project each stage-1 agent's output to that unit, de-duplicate within the
agent, and keep keys produced by ≥k distinct agents (ATSA/ACSA agreement is counted on pairs, not triples).

Usage
  python score.py tables [--all] [--counts ../results/sentence_counts.csv]
      recompute every system × unit row (mean and SD over runs) from the released sentence counts
  python score.py score --gold GOLD.jsonl --pred PRED.jsonl [--exclude ../splits/excluded_30_ids.txt]
      score your own predictions. GOLD: NIKL 2022 ABSA json lines ({"id","sentence_form","annotation"})
      or {"id","tuples":[...]}. PRED: {"id","tuples":[{"aspect_term","aspect_ref","polarity"},...]};
      for ≥k from agent outputs use {"id","agents":{"P_NEG":[...],"P_IMP":[...],"P_LIT":[...]}} and --agree k
  python score.py gold-fingerprint --gold GOLD.jsonl
      check that your gold conversion equals the paper's (compare with GOLD_FINGERPRINT below)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import canonical, nikl_annotation_to_items, records_from_list  # noqa: E402

UNITS = ("TASD", "ATSA", "ACSA")
SPLIT_UNITS = ("TASD", "ATSA")
# sha256 over sorted "id\tc\tt\tp" lines of the normalized gold records, all 2,794 sentences
GOLD_FINGERPRINT = "884a7050667b6d848097dab683b1e030f60cdffada5e91e09619067f5487123c"
HERE = Path(__file__).resolve().parent


def project(unit: str, records: Iterable) -> Set[tuple]:
    out = set()
    for r in records:
        ref, term, pol = canonical(r)
        if unit == "TASD":
            if ref:
                out.add((ref, term, pol))
        elif unit == "ATSA":
            out.add((term, pol))
        elif unit == "ACSA":
            if ref:
                out.add((ref, pol))
        elif unit == "TE":
            out.add(term)
        elif unit == "CE":
            if ref:
                out.add(ref)
        else:
            raise ValueError(unit)
    return out


def key_term(unit: str, key: tuple) -> str:
    return key[1] if unit == "TASD" else key[0]


def counts(G: Set, P: Set) -> Tuple[int, int, int]:
    if not G:
        return (0, 0, 0)
    return (len(G & P), len(P - G), len(G - P))


def split_counts(unit: str, G: Set, P: Set, implicit: bool) -> Tuple[int, int, int]:
    g = {k for k in G if (key_term(unit, k) == "") == implicit}
    p = {k for k in P if (key_term(unit, k) == "") == implicit}
    return counts(g, p)


def agreement(agent_records: Dict[str, Iterable], unit: str, k: int) -> Set[tuple]:
    c = Counter(key for recs in agent_records.values() for key in project(unit, recs))
    return {key for key, n in c.items() if n >= k}


def prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f2 = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
    return {"P": p, "R": r, "F1": f1, "F2": f2}


# ------------------------------------------------------------------ CLI: tables from released counts
def load_counts(path: Path, include_excluded: bool):
    """-> ordered sentence ids, {(system, run): {col: [per-sentence values]}} (rows in file order)."""
    data = defaultdict(lambda: defaultdict(list))
    order: List[str] = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = [c for c in rd.fieldnames if c.endswith(("_tp", "_fp", "_fn"))]
        for row in rd:
            if row["excluded_30"] == "1" and not include_excluded:
                continue
            sid = row["sentence_id"]
            if sid not in seen:
                seen.add(sid)
                order.append(sid)
            d = data[(row["system"], int(row["run"]))]
            for c in cols:
                d[c].append(int(row[c]))
    return order, data, cols


def cmd_tables(a):
    order, data, cols = load_counts(Path(a.counts), a.all)
    units = sorted({c.rsplit("_", 1)[0] for c in cols}, key=lambda u: (len(u), u))
    systems = list(dict.fromkeys(s for s, _ in data))
    print(f"N = {len(order)} sentences; mean (SD) over runs")
    print("system,unit,P,R,F1,F2,SD_P,SD_R,SD_F1,SD_F2")
    for s in systems:
        runs = sorted(r for ss, r in data if ss == s)
        for u in units:
            ms = [prf(*(sum(data[(s, r)][f"{u}_{x}"]) for x in ("tp", "fp", "fn"))) for r in runs]
            mean = {m: statistics.fmean(x[m] for x in ms) for m in ("P", "R", "F1", "F2")}
            sd = {m: statistics.stdev(x[m] for x in ms) if len(ms) > 1 else 0.0 for m in mean}
            print(f"{s},{u}," + ",".join(f"{mean[m]:.4f}" for m in mean) + "," + ",".join(f"{sd[m]:.4f}" for m in sd))


# ------------------------------------------------------------------ CLI: score your own files
def read_gold(path: Path) -> Dict[str, set]:
    out = {}
    for line in open(path, encoding="utf-8-sig"):
        if not line.strip():
            continue
        e = json.loads(line)
        sid = e.get("id") or e.get("uid")
        items = nikl_annotation_to_items(e) if "annotation" in e else (e.get("tuples") or e.get("gold_tuples") or [])
        out[sid] = records_from_list(items)
    return out


def cmd_score(a):
    gold = read_gold(Path(a.gold))
    excl = set()
    if a.exclude:
        excl = {l.strip() for l in open(a.exclude, encoding="utf-8") if l.strip()}
    pred = {}
    for line in open(a.pred, encoding="utf-8-sig"):
        if line.strip():
            e = json.loads(line)
            pred[e.get("id") or e.get("uid")] = e
    tot = defaultdict(lambda: [0, 0, 0])
    n = 0
    for sid, G in gold.items():
        if sid in excl:
            continue
        n += 1
        e = pred.get(sid, {})
        for u in UNITS:
            Gk = project(u, G)
            Pk = agreement({k: records_from_list(v) for k, v in e["agents"].items()}, u, a.agree) if "agents" in e \
                else project(u, records_from_list(e.get("tuples") or []))
            for i, v in enumerate(counts(Gk, Pk)):
                tot[u][i] += v
            if u in SPLIT_UNITS:
                for name, imp in (("explicit", False), ("implicit", True)):
                    for i, v in enumerate(split_counts(u, Gk, Pk, imp)):
                        tot[f"{u}-{name}"][i] += v
    print(f"N = {n} sentences")
    print("unit,tp,fp,fn,P,R,F1,F2")
    for u, (tp, fp, fn) in tot.items():
        m = prf(tp, fp, fn)
        print(f"{u},{tp},{fp},{fn}," + ",".join(f"{m[k]:.4f}" for k in ("P", "R", "F1", "F2")))


def gold_fingerprint(gold: Dict[str, set]) -> str:
    lines = sorted(f"{sid}\t" + "\t".join(canonical(r)) for sid, recs in gold.items() for r in recs)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def cmd_fingerprint(a):
    fp = gold_fingerprint(read_gold(Path(a.gold)))
    print(fp, "MATCH" if fp == GOLD_FINGERPRINT else f"differs from paper ({GOLD_FINGERPRINT})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--counts", default=str(HERE.parent / "results" / "sentence_counts.csv"))
    t.add_argument("--all", action="store_true", help="use all 2,794 sentences (default: 2,764 = without the 30 design-check sentences)")
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--exclude")
    s.add_argument("--agree", type=int, default=2)
    g = sub.add_parser("gold-fingerprint")
    g.add_argument("--gold", required=True)
    a = ap.parse_args()
    {"tables": cmd_tables, "score": cmd_score, "gold-fingerprint": cmd_fingerprint}[a.cmd](a)
