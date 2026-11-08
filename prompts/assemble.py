"""Rebuild the exact run prompts from the shared templates in this folder.

    python assemble.py            # verify against SHA256SUMS.txt
    python assemble.py --out DIR  # also write the six assembled prompts to DIR

Slots are capitalised or multi-word names, e.g. {Review scope}, {allowed actions}.
Runtime variables filled by the pipeline stay lower_snake_case, e.g. {text}, {candidates_json}.
Values files: a line <<<Slot name>>> starts a value; the value ends at the next <<<...>>> line
(the newline right before that marker is not part of the value).
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

SLOT = re.compile(r"\{([A-Z][A-Za-z ]*|[a-z]+(?: [a-z]+)+)\}")


def parse_values(text: str) -> dict:
    out, name, buf = {}, None, []
    for line in text.split("\n"):
        m = re.fullmatch(r"<<<(.+)>>>", line)
        if m:
            if name is not None:
                out[name] = "\n".join(buf)
            name, buf = (None if m.group(1) == "end" else m.group(1)), []
        elif name is not None:
            buf.append(line)
    return out


def assemble(template: str, values: dict) -> str:
    return SLOT.sub(lambda m: values[m.group(1)], template)


def build_all(folder: Path) -> dict:
    folder = Path(folder)
    rd = lambda n: (folder / n).read_text(encoding="utf-8")  # noqa: E731
    out = {}
    t1 = rd("stage1_shared_contract.txt")
    for k in ("neg", "imp", "lit"):
        out[f"facet_{k}"] = assemble(t1, parse_values(rd(f"facet_{k}.txt")))
    blocks = re.split(r"^=== (.+?) ===\n", rd("stage3_review.txt"), flags=re.M)[1:]
    sec = dict(zip(blocks[0::2], blocks[1::2]))
    t3 = sec["TEMPLATE"][:-1]  # drop the separator newline added between sections
    for r in "ABC":
        key = next(k for k in sec if k.startswith(f"VALUES: Review {r}"))
        out[f"review_{r}"] = assemble(t3, parse_values(sec[key]))
    return out


NAMES = {
    "facet_neg": "perspective_pneg_stage1.md", "facet_imp": "perspective_pimp_stage1.md",
    "facet_lit": "perspective_plit_stage1.md", "review_A": "review_pneg_action.md",
    "review_B": "review_pimp_action.md", "review_C": "review_plit_action.md",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    built = build_all(here)
    sums = {}
    for line in (here / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            f, lf_sha, _ = line.split()
            sums[f] = lf_sha
    for k, txt in built.items():
        ok = hashlib.sha256(txt.encode("utf-8")).hexdigest() == sums[NAMES[k]]
        print(f"{NAMES[k]:32s} {'OK' if ok else 'MISMATCH'}")
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            (Path(a.out) / NAMES[k]).write_text(txt, encoding="utf-8", newline="\n")
