"""Normalization rules used for every score in the paper (standalone, standard library only).

A record is the triple (aspect_ref, aspect_term, polarity) = (category c, target t, polarity p).
Gold and predictions go through the same functions before exact matching.
These are the rules used to produce the paper's numbers.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

Record = Tuple[str, str, str]  # (aspect_ref, aspect_term, polarity)

_POLARITY_MAP = {
    "pos": "positive", "positive": "positive",
    "neg": "negative", "negative": "negative",
    "neu": "neutral", "neutral": "neutral",
    "mixed": "mixed",
}
NULL_TERMS = frozenset({"", "null", "NULL", "None", "none"})


def normalize_polarity(s: Optional[str], default_missing: Optional[str] = "neutral") -> Optional[str]:
    """pos/neg/neu -> positive/negative/neutral. Missing or unknown -> default_missing."""
    if s is None or (isinstance(s, str) and not s.strip()):
        return default_missing
    key = s.strip().lower()
    return _POLARITY_MAP.get(key, default_missing)


def normalize_term(s: Optional[str]) -> str:
    """Target surface: strip, lower-case, collapse whitespace, strip surrounding punctuation."""
    if s is None:
        return ""
    t = s.strip().lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip(".,;:!?\"'`()[]{}")  # ASCII only, same set as the original evaluator


def normalize_ref(ref: Optional[str]) -> str:
    """Category: 패키지/구성품 -> 패키지·구성품 (corpus vs prompt spelling), collapse whitespace, trim around '#'."""
    if ref is None:
        return ""
    s = str(ref).replace("패키지/구성품", "패키지·구성품").strip()
    s = " ".join(s.split())
    if "#" in s:
        left, right = s.split("#", 1)
        s = left.strip() + "#" + right.strip()
    return s


def is_empty_term(term: Any) -> bool:
    """Implicit target: None, '', 'null', 'None', ... or empty after normalization."""
    if term is None:
        return True
    if not isinstance(term, str):
        return False
    return term.strip() in NULL_TERMS or normalize_term(term) == ""


def _term_text(item: Dict[str, Any]) -> str:
    at = item.get("aspect_term")
    if isinstance(at, dict) and at.get("term") is not None:
        return (at.get("term") or "").strip()
    if isinstance(at, str):
        return at.strip()
    op = item.get("opinion_term")  # legacy field name
    if isinstance(op, dict) and op.get("term") is not None:
        return (op.get("term") or "").strip()
    return ""


def record_from_item(item: Dict[str, Any]) -> Optional[Record]:
    """One JSON record -> (ref, term, polarity). Missing polarity counts as neutral (as in the paper runs)."""
    if item.get("is_implicit") is True:
        term = ""
    else:
        raw = _term_text(item)
        term = normalize_term(raw) if raw else ""
    ref = normalize_ref(item.get("aspect_ref") or item.get("term") or "")
    pol = normalize_polarity(item.get("polarity") or item.get("label"))
    if pol is None:
        return None
    return (ref, term, pol)


def records_from_list(items: Iterable[Dict[str, Any]]) -> Set[Record]:
    """List of JSON records -> set (duplicates inside one output collapse here)."""
    out: Set[Record] = set()
    if not items or not isinstance(items, (list, tuple)):
        return out
    for it in items:
        if not it or not isinstance(it, dict):
            continue
        r = record_from_item(it)
        if r is not None and (r[0] or r[1] or r[2].strip()):
            out.add(r)
    return out


def canonical(r: Record) -> Record:
    """Final key normalization applied right before projection (score_cr_v3_strict.norm_tuple)."""
    ref, term, pol = r
    term_n = "" if is_empty_term(term) else normalize_term(term or "")
    return (normalize_ref(ref or ""), term_n, normalize_polarity(pol or ""))


def nikl_annotation_to_items(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """NIKL 2022 ABSA json entry ({'id','sentence_form','annotation':[[cat,[term,s,e],pol],...]}) -> gold items.

    Same rule used to build the paper's gold file: term = annotated surface; if the surface is null but
    offsets exist, the substring at the offsets; otherwise "" (implicit).
    """
    text = entry.get("sentence_form") or ""
    out = []
    for ann in entry.get("annotation") or []:
        if len(ann) < 3:
            continue
        ref, pol = ann[0], ann[2]
        span = ann[1] if len(ann) > 1 else None
        if span is None or not isinstance(span, (list, tuple)):
            term = ""
        elif span[0] is not None and span[0] != "":
            term = str(span[0]).strip()
        elif len(span) >= 3:
            s, e = int(span[1]), int(span[2])
            term = text[s:e] if 0 <= s <= e <= len(text) else ""
        else:
            term = ""
        out.append({"aspect_ref": ref, "aspect_term": term, "polarity": pol})
    return out
