#!/usr/bin/env python3
"""
summarize_runs.py -- turn validity_ramp_harness_v4 result files into compact
markdown tables, plus the contention columns that decide how the concurrency
finding can be stated.

  python3 summarize_runs.py ./results/*.json
  python3 summarize_runs.py ./results/*.json --specimens

Prints one table per run file. The columns are chosen so that a reader can see,
in one row, whether validity moved AND whether the server was under any
pressure while it didn't -- because a flat validity rate on an uncontended
server is a much weaker claim than a flat rate on a queueing one, and the two
are indistinguishable if you only print the validity column.

`wait_max` is the load-bearing one. Offered concurrency below the server's
--max-num-seqs cannot queue, so wait_max == 0 means the level tested batching
headroom rather than contention, whatever its concurrency number says.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


def fmt(v, nd=3, dash="-"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def g(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


ROWS = [
    ("c", lambda d: d.get("concurrency"), 0),
    ("n", lambda d: d.get("requests"), 0),
    ("schema", lambda d: d.get("schema_valid_rate"), 4),
    ("semantic", lambda d: d.get("semantic_valid_rate"), 4),
    ("usable", lambda d: d.get("usable_rate"), 4),
    ("trunc", lambda d: d.get("truncation_rate"), 4),
    ("wait_max", lambda d: g(d, "server_metrics", "vllm:num_requests_waiting_max"), 1),
    ("preempt", lambda d: g(d, "server_metrics", "vllm:num_preemptions_total_delta"), 1),
    ("kv_max", lambda d: g(d, "server_metrics", "vllm:kv_cache_usage_perc_max"), 4),
    ("p50", lambda d: d.get("e2e_p50"), 3),
    ("p99", lambda d: d.get("e2e_p99"), 3),
    ("ttft_p50", lambda d: d.get("ttft_p50"), 4),
    ("$/usable", lambda d: d.get("cost_per_usable_usd"), 8),
]


def summarize(path, show_specimens=False):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        runs = json.load(f)
    if not isinstance(runs, list) or not runs or "concurrency" not in runs[0]:
        return
    if not runs:
        print(f"\n### {path}\n\n(empty)")
        return
    head = runs[0]
    meta = [f"mode={head.get('mode')}", f"taskset={head.get('taskset')}",
            f"load={head.get('load_mode')}", f"max_tokens={head.get('max_tokens')}",
            f"temp={head.get('temperature')}", f"seed={head.get('seed')}"]
    if head.get("duration_s"):
        meta.append(f"duration_s={head['duration_s']}")
    if head.get("stream"):
        meta.append("stream=on")
    if head.get("schema_cardinality"):
        meta.append(f"schema_cardinality={head['schema_cardinality']}")
    print(f"\n### {path}")
    print(f"`{' '.join(meta)}` model={head.get('model')}\n")

    cols = [c for c in ROWS if any(c[1](r) is not None for r in runs)]
    print("| " + " | ".join(c[0] for c in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in runs:
        print("| " + " | ".join(fmt(c[1](r), c[2]) for c in cols) + " |")

    print("\nCategories by level:")
    for r in runs:
        cats = r.get("categories") or {}
        subs = r.get("subcategories") or {}
        line = f"  c={r.get('concurrency')}: {cats}"
        if subs:
            line += f"  subs={subs}"
        mt = r.get("multi_turn")
        if mt:
            line += (f"  [multi-turn: {mt['requests']} reqs, "
                     f"mean_turns={mt['mean_turns_completed']}, "
                     f"intermediate_fail={mt['intermediate_turn_failure_rate']}]")
        print(line)

    # Where a conversation died, not just how often. On a five-step task the
    # aggregate rate says nothing actionable: dying at turn 1 is a broken tool
    # schema, at turn 3 broken arithmetic, at turn 5 lost context, and the three
    # take different fixes. Only printed when a run has more than two turns,
    # since on the two-turn task the histogram carries no information the
    # intermediate-failure rate doesn't already give.
    deep = [r for r in runs
            if (max((r.get("multi_turn") or {}).get("turns_completed_hist",
                                                    {"0": 0}).keys(),
                    key=lambda k: int(k), default="0")) not in ("0", "1", "2")]
    if deep:
        print("\nWhere multi-turn conversations ended (turns completed -> n):")
        for r in deep:
            mt = r["multi_turn"]
            print(f"  c={r.get('concurrency')}: {mt['turns_completed_hist']}"
                  f"  failed_at={mt.get('failures_by_turn') or '{}'}")

    # Did the cardinality arm actually offer the cardinality it was configured
    # for? With a rotation longer than the request count the tail of the schema
    # set is never requested, and the run tested a lower number than its
    # filename says.
    card = [r for r in runs if r.get("schema_cardinality")]
    if card:
        print("\nSchema cardinality (offered -> reached):")
        for r in card:
            off = r.get("distinct_schemas_offered")
            used = r.get("distinct_schemas_used")
            flag = "" if off == used else "  <-- ROTATION INCOMPLETE"
            print(f"  c={r.get('concurrency')}: {used}/{off} schemas over "
                  f"{r.get('requests')} requests{flag}")

    # Which task failed, not just how many. Without this, "both arms fail at
    # 25%" cannot be told apart from "the same task fails in both arms, at a
    # different rung" -- and only the second is evidence that enforcement
    # relocates a failure rather than removing it.
    if any(r.get("categories_by_task") for r in runs):
        print("\nCategories by task (per level):")
        for r in runs:
            print(f"  c={r.get('concurrency')}:")
            for task, cats in (r.get("categories_by_task") or {}).items():
                print(f"    {task}: {cats}")

    # Error strings, verbatim. A validity dip made of transport exceptions is a
    # different finding from one made of schema violations, and the category
    # alone does not say which exception fired.
    if any(r.get("errors") for r in runs):
        print("\nErrors by level (verbatim):")
        for r in runs:
            errs = r.get("errors") or {}
            if errs:
                print(f"  c={r.get('concurrency')}: {errs}")

    # Contention verdict. This is the sentence the article needs, derived rather
    # than eyeballed: a level where nothing queued did not test contention.
    contended = [r.get("concurrency") for r in runs
                 if (g(r, "server_metrics", "vllm:num_requests_waiting_max") or 0) > 0]
    preempted = [r.get("concurrency") for r in runs
                 if (g(r, "server_metrics", "vllm:num_preemptions_total_delta") or 0) > 0]
    if any(r.get("server_metrics") for r in runs):
        print("\nContention:")
        print(f"  levels that queued at all: {contended or 'NONE'}")
        print(f"  levels with preemption:    {preempted or 'NONE'}")
        if not contended:
            print("  -> every level fit inside the batch. A flat validity rate")
            print("     here is a claim about batching headroom, not about load.")
        else:
            vals = {r["concurrency"]: r.get("schema_valid_rate") for r in runs
                    if r.get("concurrency") in contended}
            print(f"  schema validity at queueing levels: {vals}")

    if show_specimens:
        seen = set()
        print("\nSpecimens (first occurrence per category across levels):")
        for r in runs:
            specimens = dict(r.get("specimens_by_category") or {})
            if not specimens:
                for record in r.get("records") or []:
                    if record.get("category") == "ok":
                        continue
                    key = record.get("category") or "unknown"
                    if record.get("subcategory"):
                        key = f"{key}/{record['subcategory']}"
                    specimens.setdefault(key, {
                        "task": record.get("task"),
                        "finish_reason": record.get("finish_reason"),
                        "http_status": record.get("http_status"),
                        "error": record.get("error"),
                        "raw": record.get("specimen"),
                    })
            for k, v in specimens.items():
                if k in seen:
                    continue
                seen.add(k)
                raw = (v.get("raw") or "")
                raw = raw if len(raw) < 400 else raw[:400] + " ...[truncated]"
                print(f"\n  [{k}] task={v.get('task')} "
                      f"finish_reason={v.get('finish_reason')} "
                      f"http={v.get('http_status')} error={v.get('error')}")
                print(f"  raw: {raw!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--specimens", action="store_true",
                    help="print one real captured output per failure category")
    a = ap.parse_args()
    paths = []
    for value in a.files:
        p = Path(value)
        if p.is_dir():
            paths.extend(sorted(
                f for f in p.rglob("*")
                if f.is_file() and (f.suffix == ".json" or f.name.endswith(".json.gz"))
            ))
        else:
            paths.append(p)
    for p in paths:
        try:
            summarize(p, a.specimens)
        except Exception as e:
            print(f"\n### {p}\n\n(failed to read: {type(e).__name__}: {e})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
