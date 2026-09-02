#!/usr/bin/env python3
"""
compare_conformance.py -- build the backend coverage matrix.

  python3 compare_conformance.py conformance_*.json

Emits a markdown table of keyword vs backend, ready to drop into the article,
plus the two comparisons that matter:

  * typed vs untyped pairs   -- does omitting "type" defeat enforcement?
  * cross-backend divergence -- do backends disagree about the same keyword?

Refuses to include any run marked unusable, and refuses to emit a table at all
if the enum control case failed, because a harness that can't enforce an enum
can't be trusted about anything subtler.
"""

import json
import sys
from collections import OrderedDict

GLYPH = {
    "PREVENTED": "enforced",
    "VIOLATED": "**NOT enforced**",
    "REJECTED": "rejected",
    "MIXED": "**inconsistent**",
    "TRUNCATED": "unsatisfiable",
    "INCONCLUSIVE": "partial data",
    "ERROR": "no data",
}

TYPED_UNTYPED = [("minimum", "untyped_minimum"),
                 ("pattern", "untyped_pattern"),
                 ("maxLength", "untyped_maxLength")]


def load(paths):
    runs = OrderedDict()
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"skipping {p}: {e}", file=sys.stderr)
            continue
        if d.get("usable") is False:
            print(f"skipping {p}: marked unusable (all cases errored)", file=sys.stderr)
            continue
        label = d.get("provenance", {}).get("backend_label") or p
        runs[label] = d
    return runs


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    runs = load(sys.argv[1:])
    if not runs:
        raise SystemExit("no usable conformance files. Nothing to compare.")

    # Control-case gate.
    broken = []
    for label, d in runs.items():
        row = next((c for c in d["cases"] if c["id"] == "enum"), None)
        if row and row["verdict"] != "PREVENTED":
            broken.append(f"{label} (enum={row['verdict']})")
    if broken:
        raise SystemExit(
            "enum control case failed on: " + ", ".join(broken) +
            "\nFix the harness or server config before trusting any row.")

    ids = [c["id"] for c in next(iter(runs.values()))["cases"]]
    labels = list(runs)

    print("### Backend conformance: is the keyword actually enforced?\n")
    print("| Keyword | " + " | ".join(labels) + " |")
    print("|---" * (len(labels) + 1) + "|")
    for cid in ids:
        cells = []
        for lb in labels:
            row = next((c for c in runs[lb]["cases"] if c["id"] == cid), None)
            cells.append(GLYPH.get(row["verdict"], row["verdict"]) if row else "-")
        print(f"| `{cid}` | " + " | ".join(cells) + " |")

    print("\n**Versions**\n")
    for lb, d in runs.items():
        p = d["provenance"]
        libs = ", ".join(f"{k} {p[k]}" for k in
                         ("vllm", "xgrammar", "outlines_core", "llguidance")
                         if p.get(k))
        print(f"- `{lb}`: {libs or 'not captured'}")

    print("\n### The type gate\n")
    any_gate = False
    for typed, untyped in TYPED_UNTYPED:
        for lb in labels:
            cs = {c["id"]: c["verdict"] for c in runs[lb]["cases"]}
            t, u = cs.get(typed), cs.get(untyped)
            if t == "PREVENTED" and u in ("VIOLATED", "MIXED"):
                any_gate = True
                print(f"- `{lb}`: `{typed}` enforced, `{untyped}` {GLYPH[u]}. "
                      f"Omitting `type` defeated the constraint.")
            elif t and u and t != u:
                any_gate = True
                print(f"- `{lb}`: `{typed}` -> {t}, `{untyped}` -> {u}.")
    if not any_gate:
        print("No typed/untyped divergence observed. The type gate did not "
              "change enforcement for these keywords on these backends.")

    print("\n### Cross-backend divergence\n")
    if len(labels) < 2:
        print(f"Only one backend probed (`{labels[0]}`). Nothing to compare. "
              "Run the probe against another backend to fill this in.")
    else:
        div = False
        for cid in ids:
            vs = {lb: next((c["verdict"] for c in runs[lb]["cases"]
                            if c["id"] == cid), None)
                  for lb in labels}
            real = {v for v in vs.values()
                    if v not in (None, "ERROR", "INCONCLUSIVE")}
            if len(real) > 1:
                div = True
                print(f"- `{cid}`: " +
                      ", ".join(f"{lb}={v}" for lb, v in vs.items()))
        if not div:
            print("No divergence: every backend treated every probed keyword "
                  "the same way.")

    print("\n### Outcome breakdown\n")
    print("`rejected` is not a failure: the server refused the schema up front, "
          "so you find out at request time rather than in production. The row "
          "that should worry you is `NOT enforced`, where the schema was "
          "accepted and the constraint did nothing.\n")
    for lb in labels:
        t = {}
        for c in runs[lb]["cases"]:
            t[c["verdict"]] = t.get(c["verdict"], 0) + 1
        n = len(runs[lb]["cases"])
        print(f"- `{lb}` ({n} cases): "
              f"{t.get('PREVENTED',0)} enforced, "
              f"{t.get('REJECTED',0)} rejected up front, "
              f"{t.get('VIOLATED',0)} accepted but NOT enforced, "
              f"{t.get('INCONCLUSIVE',0)+t.get('TRUNCATED',0)+t.get('ERROR',0)+t.get('MIXED',0)} "
              f"without a clean verdict")
        costly = [(c["id"], c["truncation_rate"]) for c in runs[lb]["cases"]
                  if c.get("truncation_rate")]
        if costly:
            print("  truncation cost while enforcing: " +
                  ", ".join(f"`{i}` {r:.0%}" for i, r in costly))


if __name__ == "__main__":
    main()
