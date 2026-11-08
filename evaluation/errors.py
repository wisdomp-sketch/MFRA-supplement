"""Error classification used in the paper's error analyses (sentence-level error types and target-extraction errors).

Requires numpy and scipy (Hungarian matching). Normalization is the same as in normalize.py, except that a
missing polarity stays empty here instead of becoming "neutral".

Per sentence, the gold list and the prediction list (duplicates kept) are aligned:
  1. exact (t, c, p) matches are paired first;
  2. the remaining gold and predicted records are paired by maximum-weight bipartite matching, with pair score
       +4 same target (implicit-implicit counts as same), or +3 overlapping explicit spans,
          or +2.5 one target string contains the other (both >= 2 characters)
       +2 same category   +1 same polarity
     A pair is kept even at score 0; whatever is left is an unmatched gold or an unmatched prediction.

Sentence-level error types (multi-label flags, after Radi et al. 2025, adapted to this data):
  TD   target error: an unmatched gold, an unmatched prediction, or a pair whose targets differ
  ACD  category error: a pair whose categories differ
  SD   sentiment error: a pair whose polarities differ
  MOD  multi-opinion error, only for sentences with >= 2 gold records: tuple-count collapse, split/duplicate of
       one gold target, cross-category attribution, sentiment spillover, or sentiment swap.
       Not raised when the alignment is ambiguous and none of these patterns is found.
  A sentence has an error if any flag is set; ">= 2 types" if two or more flags are set.

Target-extraction errors (explicit gold targets only, Korean NER error taxonomy of Yu et al. 2021):
  for each explicit gold involved in a TD error, in priority order:
    FR-  Missed: no overlapping predicted span
    FS+  one gold covered by >= 2 predicted spans          FS-  one predicted span covering >= 2 gold spans
    FB-  Under-extended: predicted span inside the gold    FB+  Over-extended: gold span inside the prediction
    FBX  other partial overlap
  for each unmatched explicit prediction:
    FR+  Spurious: overlaps no explicit gold span
  Rates are divided by the number of explicit gold records (implicit gold records are not applicable).

Spurious predictions are further split (in this order):
  (a) its category equals the category of an implicit gold record in the same sentence
  (b) its target and an explicit gold target contain one another after removing whitespace
  (c) other

Usage
  python errors.py classify --gold GOLD.jsonl --pred PRED.jsonl [--out counts.csv]
      GOLD: NIKL 2022 ABSA json lines ({"id","sentence_form","annotation"}); the sentence and the gold
      offsets are needed. PRED: {"id","tuples":[{"aspect_term","aspect_ref","polarity"},...]}.
      Writes one row per sentence with the columns of results/error_counts.csv (without run/system).
  python errors.py tables [--all] [--counts ../results/error_counts.csv] [--by-run OUT.csv]
      recompute the error tables (mean and SD over runs) from the released per-sentence counts
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from normalize import normalize_ref, normalize_term, normalize_polarity  # noqa: E402

NULL_MARKERS = frozenset({"", "null", "NULL", "none", "None"})
NER_COLS = {"FR-": "FR_minus", "FR+": "FR_plus", "FBX": "FBX", "FB-": "FB_minus", "FB+": "FB_plus",
            "FS-": "FS_minus", "FS+": "FS_plus"}
FLAG_COLS = ["TD", "ACD", "SD", "MOD"]
COUNT_COLS = (["n_gold", "n_pred"] + FLAG_COLS + ["n_explicit_gold"] + list(NER_COLS.values())
              + ["spurious_a", "spurious_b", "spurious_c"])


# ------------------------------------------------------------------ records
def norm_target(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        raw = raw.get("term")
        if raw is None:
            return ""
    s = str(raw).strip()
    if s in NULL_MARKERS or s.lower() in {"null", "none"}:
        return ""
    return normalize_term(s)


@dataclass
class Rec:
    target: str
    category: str
    polarity: str
    raw: Dict[str, Any] = field(default_factory=dict)
    index: int = -1
    start: Optional[int] = None
    end: Optional[int] = None

    @property
    def is_implicit(self) -> bool:
        return self.target == ""

    @property
    def key(self):
        return (self.target, self.category, self.polarity)


def to_rec(d: Dict[str, Any], i: int) -> Rec:
    raw_t = d.get("aspect_term")
    if raw_t is None and "target" in d:
        raw_t = d.get("target")
    cat = d.get("aspect_ref")
    if cat is None:
        cat = d.get("category") or d.get("aspect_category")
    pol = d.get("polarity") or d.get("label")
    pol = "" if pol is None or (isinstance(pol, str) and not pol.strip()) else (normalize_polarity(pol, None) or "")
    start = end = None
    if isinstance(d.get("span"), dict):
        start, end = d["span"].get("start"), d["span"].get("end")
    return Rec(norm_target(raw_t), normalize_ref(str(cat).strip() if cat else ""), pol, dict(d), i,
               int(start) if start is not None else None, int(end) if end is not None else None)


# ------------------------------------------------------------------ alignment
@dataclass
class Pair:
    gold: Optional[Rec]
    pred: Optional[Rec]
    status: str  # exact | matched | unmatched_gold | unmatched_pred
    score: float = 0.0
    ambiguous: bool = False


def _span_overlap(a: Rec, b: Rec) -> bool:
    if a.is_implicit or b.is_implicit or None in (a.start, a.end, b.start, b.end):
        return False
    return not (a.end <= b.start or b.end <= a.start)


def same_target(g: Rec, p: Rec) -> bool:
    if g.is_implicit and p.is_implicit:
        return True
    if g.is_implicit or p.is_implicit:
        return False
    return g.target == p.target


def pair_score(g: Rec, p: Rec) -> float:
    s = 0.0
    if same_target(g, p):
        s += 4.0
    elif not g.is_implicit and not p.is_implicit and _span_overlap(g, p):
        s += 3.0
    elif (not g.is_implicit and not p.is_implicit and (g.target in p.target or p.target in g.target)
          and min(len(g.target), len(p.target)) >= 2):
        s += 2.5
    if g.category and p.category and g.category == p.category:
        s += 2.0
    if g.polarity and p.polarity and g.polarity == p.polarity:
        s += 1.0
    return s


def align(golds: Sequence[Rec], preds: Sequence[Rec], sentence: str) -> List[Pair]:
    out: List[Pair] = []
    used_g, used_p = set(), set()
    for i, g in enumerate(golds):
        for j, p in enumerate(preds):
            if j not in used_p and g.key == p.key:
                out.append(Pair(g, p, "exact", 7.0))
                used_g.add(i)
                used_p.add(j)
                break
    g_res = [g for i, g in enumerate(golds) if i not in used_g]
    p_res = [p for j, p in enumerate(preds) if j not in used_p]
    if not g_res or not p_res:
        out += [Pair(None, p, "unmatched_pred") for p in p_res]
        out += [Pair(g, None, "unmatched_gold") for g in g_res]
        return out
    n_g, n_p = len(g_res), len(p_res)
    n = max(n_g, n_p)
    cost = np.zeros((n, n))
    score = np.zeros((n_g, n_p))
    for i, g in enumerate(g_res):
        for j, p in enumerate(p_res):
            score[i, j] = pair_score(g, p)
            cost[i, j] = -score[i, j]
    ag, ap = set(), set()
    for a, b in zip(*linear_sum_assignment(cost)):
        if a >= n_g or b >= n_p:
            continue
        g, p = g_res[a], p_res[b]
        row = score[a]
        top = float(row.max())
        amb = top > 0 and int(np.sum(np.isclose(row, top, atol=1e-6))) > 1
        if not g.is_implicit and sentence and g.target and sentence.lower().count(g.target) > 1:
            amb = True
        ag.add(a)
        ap.add(b)
        out.append(Pair(g, p, "matched", float(score[a, b]), amb))
    out += [Pair(g, None, "unmatched_gold") for i, g in enumerate(g_res) if i not in ag]
    out += [Pair(None, p, "unmatched_pred") for j, p in enumerate(p_res) if j not in ap]
    return out


# ------------------------------------------------------------------ sentence-level error types
def pair_flags(a: Pair) -> Tuple[int, int, int]:
    if a.status == "exact":
        return 0, 0, 0
    if a.status in ("unmatched_gold", "unmatched_pred"):
        return 1, 0, 0
    g, p = a.gold, a.pred
    return int(not same_target(g, p)), int(g.category != p.category), int(g.polarity != p.polarity)


def mod_flag(golds: List[Rec], preds: List[Rec], pairs: List[Pair]) -> int:
    if len(golds) < 2:
        return 0
    reasons = set()
    matched = [a for a in pairs if a.status in ("exact", "matched") and a.gold and a.pred]
    unmatched_g = [a for a in pairs if a.status == "unmatched_gold"]
    if len(preds) < len(golds) and unmatched_g:  # collapse
        for p in preds:
            covers = 0
            for g in golds:
                if g.is_implicit or p.is_implicit:
                    covers += int(bool(g.category) and g.category == p.category)
                    continue
                if g.target and p.target and (g.target in p.target or p.target in g.target):
                    covers += 1
            if covers >= 2:
                reasons.add("collapse")
                break
        if "collapse" not in reasons and len(preds) <= len(golds) - 1:
            reasons.add("count_collapse")
    if len(preds) > len(golds):  # split / duplicate
        for g in golds:
            if g.is_implicit:
                continue
            hits = [p for p in preds if not p.is_implicit and g.target and p.target
                    and (g.target == p.target or g.target in p.target or p.target in g.target)]
            if len(hits) >= 2:
                reasons.add("split")
                break
    for a in matched:  # cross-attribution / spillover
        g, p = a.gold, a.pred
        if g.is_implicit or p.is_implicit:
            continue
        if g.target != p.target and not (g.target in p.target or p.target in g.target):
            continue
        hit = False
        for g2 in golds:
            if g2.index == g.index:
                continue
            if g2.category == p.category and g.category != p.category and g2.category:
                reasons.add("cross_category")
                hit = True
                break
            if g2.polarity == p.polarity and g.polarity != p.polarity and g2.target != g.target:
                if g2.polarity != g.polarity:
                    reasons.add("spillover")
                    hit = True
                    break
        if hit:
            break
    if len(matched) >= 2:  # sentiment swap
        by_t = {a.gold.target: (a.gold.polarity, a.pred.polarity) for a in matched
                if not a.gold.is_implicit and a.gold.target == a.pred.target}
        ts = list(by_t)
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                gi, pi = by_t[ts[i]]
                gj, pj = by_t[ts[j]]
                if gi != gj and pi == gj and pj == gi:
                    reasons.add("swap")
    return int(bool(reasons))


# ------------------------------------------------------------------ target-extraction errors
def _find(sentence: str, term: str) -> List[Tuple[int, int]]:
    if not sentence or not term:
        return []
    hits, i = [], sentence.find(term)
    while i >= 0:
        hits.append((i, i + len(term)))
        i = sentence.find(term, i + 1)
    if hits:
        return hits
    low, t = sentence.lower(), term.lower()
    i = low.find(t)
    while i >= 0:
        hits.append((i, i + len(term)))
        i = low.find(t, i + 1)
    return hits


def span_of(sentence: str, r: Rec) -> Tuple[Optional[int], Optional[int]]:
    """Gold offsets if given; otherwise the first occurrence of the raw (then normalized) target."""
    if r.is_implicit:
        return None, None
    if r.start is not None and r.end is not None and r.end > r.start:
        return r.start, r.end
    raw = r.raw.get("aspect_term") or r.raw.get("target") or ""
    if isinstance(raw, dict):
        raw = raw.get("term") or ""
    hits = _find(sentence, raw)
    if not hits and r.target:
        hits = _find(sentence.lower(), r.target)
    return hits[0] if hits else (None, None)


def _ov(a0, a1, b0, b1) -> bool:
    return not (a1 <= b0 or b1 <= a0)


def _inside(o0, o1, i0, i1) -> bool:
    return o0 <= i0 and o1 >= i1 and (o1 - o0) > (i1 - i0)


def gold_error_type(sentence: str, g: Rec, p: Optional[Rec], golds, preds) -> Optional[str]:
    gs, ge = span_of(sentence, g)
    if gs is None:
        return None  # explicit gold not found in the sentence: not applicable
    if p is None or p.is_implicit:
        return "FR-"
    ps, pe = span_of(sentence, p)
    if ps is None:
        return "FR-"
    cover_p = 0
    for p2 in preds:
        if p2.is_implicit:
            continue
        s2, e2 = span_of(sentence, p2)
        if s2 is not None and (_ov(gs, ge, s2, e2) or (g.target and p2.target and (g.target in p2.target or p2.target in g.target))):
            cover_p += 1
    if cover_p >= 2:
        return "FS+"
    cover_g = 0
    for g2 in golds:
        if g2.is_implicit:
            continue
        s2, e2 = span_of(sentence, g2)
        if s2 is not None and (_ov(s2, e2, ps, pe) or (g2.target and p.target and (g2.target in p.target or p.target in g2.target))):
            cover_g += 1
    if cover_g >= 2:
        return "FS-"
    if not _ov(gs, ge, ps, pe):
        return "FR-"
    if _inside(gs, ge, ps, pe):
        return "FB-"
    if _inside(ps, pe, gs, ge):
        return "FB+"
    return "FBX"


def spurious_type(sentence: str, p: Rec, golds) -> Optional[str]:
    """FR+ if the explicit prediction overlaps no explicit gold span; None if it relates to a gold (not counted)."""
    ps, pe = span_of(sentence, p)
    for g in golds:
        if g.is_implicit:
            continue
        gs, ge = span_of(sentence, g)
        if gs is None or ps is None:
            if g.target and p.target and (g.target == p.target or g.target in p.target or p.target in g.target):
                return None
            continue
        if _ov(gs, ge, ps, pe):
            return None
    return "FR+"


def nows(t: str) -> str:
    return "".join((t or "").split())


# ------------------------------------------------------------------ one sentence
def classify_sentence(sentence: str, gold_items: List[Dict], pred_items: List[Dict]) -> Dict[str, int]:
    golds = [to_rec(d, i) for i, d in enumerate(gold_items)]
    preds = [to_rec(d, i) for i, d in enumerate(pred_items)]
    pairs = align(golds, preds, sentence)
    flags = [pair_flags(a) for a in pairs]
    td_gold = {a.gold.index for a, f in zip(pairs, flags) if f[0] and a.gold is not None}
    row = {"n_gold": len(golds), "n_pred": len(preds),
           "TD": int(any(f[0] for f in flags)), "ACD": int(any(f[1] for f in flags)),
           "SD": int(any(f[2] for f in flags)), "MOD": mod_flag(golds, preds, pairs),
           "n_explicit_gold": sum(1 for a in pairs if a.gold is not None and not a.gold.is_implicit)}
    ner = Counter()
    spurious_targets = []
    seen = set()
    for a in pairs:
        if a.status == "exact":
            continue
        if a.gold is not None and a.gold.index in td_gold and not a.gold.is_implicit:
            t = gold_error_type(sentence, a.gold, a.pred, golds, preds)
            if t:
                ner[t] += 1
        if a.status == "unmatched_pred" and not a.pred.is_implicit:
            k = (a.pred.index, a.pred.target, a.pred.category, a.pred.polarity)
            if k not in seen:
                seen.add(k)
                if spurious_type(sentence, a.pred, golds) == "FR+":
                    ner["FR+"] += 1
                    spurious_targets.append(a.pred.target)
    for t, col in NER_COLS.items():
        row[col] = ner[t]
    # (a)/(b)/(c): the first not-yet-used explicit prediction with the same target stands for each FR+ event
    pool = defaultdict(list)
    for p in preds:
        if not p.is_implicit:
            pool[p.target].append(p)
    impl_cats = {g.category for g in golds if g.is_implicit}
    expl = [nows(g.target) for g in golds if not g.is_implicit and g.target]
    abc = Counter()
    for tgt in spurious_targets:
        p = pool[tgt].pop(0) if pool.get(tgt) else None
        pt = nows(tgt)
        if p is not None and p.category in impl_cats:
            abc["a"] += 1
        elif pt and any(g in pt or pt in g for g in expl):
            abc["b"] += 1
        else:
            abc["c"] += 1
    row.update(spurious_a=abc["a"], spurious_b=abc["b"], spurious_c=abc["c"])
    return row


def nikl_gold(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """NIKL annotation -> gold items with offsets (term = annotated surface, '' if implicit)."""
    out = []
    for ann in entry.get("annotation") or []:
        if len(ann) < 3:
            continue
        span = ann[1] if isinstance(ann[1], (list, tuple)) else [None, 0, 0]
        term = "" if span[0] in (None, "") else str(span[0]).strip()
        item = {"aspect_ref": ann[0], "aspect_term": term, "polarity": ann[2]}
        if len(span) >= 3 and span[1] is not None:
            item["span"] = {"start": int(span[1]), "end": int(span[2])}
        out.append(item)
    return out


# ------------------------------------------------------------------ commands
def cmd_classify(a):
    preds = {}
    for line in open(a.pred, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            preds[str(r["id"])] = r.get("tuples") or []
    out = open(a.out, "w", encoding="utf-8", newline="") if a.out else sys.stdout
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["sentence_id"] + COUNT_COLS)
    for line in open(a.gold, encoding="utf-8"):
        if line.strip():
            e = json.loads(line)
            row = classify_sentence(e.get("sentence_form") or "", nikl_gold(e), preds.get(str(e["id"]), []))
            w.writerow([e["id"]] + [row[c] for c in COUNT_COLS])


def run_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    """Error-table values for one system x run from its per-sentence rows."""
    n = len(rows)
    s = defaultdict(int)
    for r in rows:
        f = [int(r[c]) for c in FLAG_COLS]
        s["any"] += int(any(f))
        s["multi"] += int(sum(f) >= 2)
        for c in COUNT_COLS:
            s[c] += int(r[c])
    out = {"n_sentences": n, "error_rate": s["any"] / n}
    out.update({f"{c}_rate": s[c] / n for c in FLAG_COLS})
    out["multi_type_rate"] = s["multi"] / n
    out["n_explicit_gold"] = s["n_explicit_gold"]
    for col in NER_COLS.values():
        out[f"{col}_n"] = s[col]
        out[f"{col}_per_explicit_gold"] = s[col] / s["n_explicit_gold"]
    for k in ("a", "b", "c"):
        out[f"spurious_{k}_n"] = s[f"spurious_{k}"]
        out[f"spurious_{k}_share"] = s[f"spurious_{k}"] / s["FR_plus"] if s["FR_plus"] else 0.0
    return out


def by_run(counts: Path, include_excluded: bool):
    groups = defaultdict(list)
    for r in csv.DictReader(open(counts, encoding="utf-8")):
        if include_excluded or r["excluded_30"] == "0":
            groups[(r["system"], int(r["run"]), r["run_label"])].append(r)
    return {k: run_metrics(v) for k, v in groups.items()}


def cmd_tables(a):
    res = by_run(Path(a.counts), a.all)
    if a.by_run:
        keys = list(next(iter(res.values())))
        with open(a.by_run, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["system", "run", "run_label"] + keys)
            for (s, run, lab), m in res.items():
                w.writerow([s, run, lab] + [round(m[k], 6) if isinstance(m[k], float) else m[k] for k in keys])
    systems = list(dict.fromkeys(k[0] for k in res))
    show = (["error_rate", "TD_rate", "ACD_rate", "SD_rate", "MOD_rate", "multi_type_rate"]
            + [f"{c}_per_explicit_gold" for c in NER_COLS.values()]
            + ["spurious_a_share", "spurious_b_share", "spurious_c_share"])
    n = next(iter(res.values()))["n_sentences"]
    print(f"N = {n} sentences; mean (SD) over runs")
    print("system,metric,mean,SD")
    for s in systems:
        for m in show:
            xs = [v[m] for k, v in res.items() if k[0] == s]
            print(f"{s},{m},{statistics.fmean(xs):.4f},{statistics.stdev(xs) if len(xs) > 1 else 0:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--gold", required=True)
    c.add_argument("--pred", required=True)
    c.add_argument("--out")
    t = sub.add_parser("tables")
    t.add_argument("--all", action="store_true", help="all 2,794 sentences (default 2,764)")
    t.add_argument("--counts", default=str(HERE.parent / "results" / "error_counts.csv"))
    t.add_argument("--by-run", help="also write per-run values to this CSV")
    a = ap.parse_args()
    {"classify": cmd_classify, "tables": cmd_tables}[a.cmd](a)


if __name__ == "__main__":
    main()
