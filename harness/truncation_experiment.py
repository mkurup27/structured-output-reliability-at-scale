#!/usr/bin/env python3
"""
Truncation recovery: what a repair heuristic does to a cut-off JSON document.

WHAT CHANGED IN v2, AND WHY
---------------------------
v1 of this script reported a "0.0% raw parse rate" across ~350k cut points as
though it were an empirical finding. It is not. Every document here is a
top-level JSON object, so every proper prefix is missing at least its closing
brace and cannot parse *by construction*. That number is a tautology and v2
reports it as one rather than as a result.

v1 also had three real methodology defects, all fixed here:

 1. It cut at every CHARACTER position. Models stop at token boundaries, and
    uniform character sampling over-weights long string fields and long
    documents. v2 cuts at approximate token boundaries and reports results by
    DECILE OF DOCUMENT COMPLETION, which is interpretable, rather than as a
    pooled percentage of cut points, which is not.

 2. It blended three different notions of "correct" into one metric.
    v2 reports them separately:
      parses            - json.loads succeeds after repair
      schema_valid      - validates against the declared schema
      exact             - byte-identical object to the pre-truncation original
      semantic          - passes field-level business rules
    These are nested but NOT equivalent, and the gaps between them are the
    interesting part.

 3. Its semantic checker rejected some of its own generated ground truths
    (only 57.8% of untruncated `nested` documents passed), which made
    "correct" uninterpretable. v2 generates ground truth that satisfies the
    checker by construction and ASSERTS the floor is 1.0 before measuring.

The repair function is NOT "standard bracket closing". It closes unterminated
strings, drops dangling commas and colons, walks back over partial keys, and
then closes open brackets. It is a custom heuristic and is published in full
below so the numbers can be checked against the exact implementation.

Deps: jsonschema. No GPU, no model, no network.
"""

import json
import random
import re
import statistics
import sys
from collections import defaultdict

import jsonschema
from jsonschema import Draft7Validator

jsonschema_version = getattr(jsonschema, "__version__", "unknown")

SEED = 20260813

WORDS = (
    "the customer reported that their droplet became unreachable after a routine kernel "
    "upgrade and the attached block storage volume failed to remount cleanly which caused "
    "the application to return errors for roughly eleven minutes before failover completed "
    "we recommend enabling automated snapshots and reviewing the health check interval"
).split()

INTENTS = ["refund_request", "billing_question", "outage_report", "upgrade_plan", "cancel_account"]
REGIONS = ["nyc3", "sfo3", "ams3", "fra1", "blr1", "tor1"]

SCHEMAS = {}
GENERATORS = {}

# --------------------------------------------------------------------------
# Schemas and generators. Generators are constrained so that every untruncated
# document passes semantic_check() -- verified by assertion in main().
# --------------------------------------------------------------------------

SCHEMAS["flat"] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "confidence": {"type": "number"},
        "customer_id": {"type": "string"},
        "requires_human": {"type": "boolean"},
        "summary": {"type": "string"},
    },
    "required": ["intent", "confidence", "customer_id", "requires_human", "summary"],
    "additionalProperties": False,
}


def gen_flat():
    return {
        "intent": random.choice(INTENTS),
        "confidence": round(random.uniform(0.5, 0.99), 3),
        "customer_id": "cus_%08x" % random.getrandbits(32),
        "requires_human": random.random() < 0.3,
        "summary": " ".join(random.choice(WORDS) for _ in range(random.randint(12, 40))),
    }


SCHEMAS["nested"] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "arguments": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["region", "start_date", "end_date", "limit"],
                },
            },
            "required": ["query", "filters"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["tool", "arguments", "reasoning"],
}


def gen_nested():
    a, b = sorted(random.sample(range(1, 13), 2))  # guarantees start <= end
    return {
        "tool": random.choice(["search_invoices", "get_usage", "list_droplets", "open_ticket"]),
        "arguments": {
            "query": " ".join(random.choice(WORDS) for _ in range(random.randint(4, 12))),
            "filters": {
                "region": random.choice(REGIONS),
                "start_date": "2026-%02d-01" % a,
                "end_date": "2026-%02d-28" % b,
                "limit": random.choice([10, 25, 50, 100]),
            },
        },
        "reasoning": " ".join(random.choice(WORDS) for _ in range(random.randint(20, 60))),
    }


SCHEMAS["enum_heavy"] = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": ["p0", "p1", "p2", "p3", "p4"]},
        "team": {"type": "string", "enum": ["networking", "storage", "compute", "billing", "ml"]},
        "action": {"type": "string",
                   "enum": ["page", "ticket", "auto_remediate", "watch", "close"]},
        "category": {"type": "string",
                     "enum": ["hardware", "software", "config", "capacity", "external", "unknown"]},
        "justification": {"type": "string"},
    },
    "required": ["severity", "team", "action", "category", "justification"],
}


def gen_enum_heavy():
    sev = random.choice(["p0", "p1", "p2", "p3", "p4"])
    # p0 must escalate -- enforced here so ground truth satisfies the checker
    action = (random.choice(["page", "auto_remediate"]) if sev == "p0"
              else random.choice(["page", "ticket", "auto_remediate", "watch", "close"]))
    return {
        "severity": sev,
        "team": random.choice(["networking", "storage", "compute", "billing", "ml"]),
        "action": action,
        "category": random.choice(
            ["hardware", "software", "config", "capacity", "external", "unknown"]),
        "justification": " ".join(random.choice(WORDS) for _ in range(random.randint(15, 45))),
    }


SCHEMAS["array_of_objects"] = {
    "type": "object",
    "properties": {
        "records": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "qty": {"type": "integer"},
                    "unit_price": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["sku", "qty", "unit_price", "note"],
            },
        },
        "total": {"type": "number"},
    },
    "required": ["records", "total"],
}


def gen_array_of_objects():
    recs = [{
        "sku": "SKU-%05d" % random.randint(0, 99999),
        "qty": random.randint(1, 40),
        "unit_price": round(random.uniform(1, 500), 2),
        "note": " ".join(random.choice(WORDS) for _ in range(random.randint(3, 10))),
    } for _ in range(random.randint(3, 10))]
    return {"records": recs,
            "total": round(sum(r["qty"] * r["unit_price"] for r in recs), 2)}


GENERATORS.update({"flat": gen_flat, "nested": gen_nested,
                   "enum_heavy": gen_enum_heavy, "array_of_objects": gen_array_of_objects})


def semantic_check(name, obj):
    """Field-level business rules -- the layer schema validation cannot express."""
    try:
        if name == "flat":
            return (bool(obj["summary"].strip())
                    and 0.0 <= obj["confidence"] <= 1.0
                    and obj["intent"] in INTENTS
                    and obj["customer_id"].startswith("cus_"))
        if name == "nested":
            a = obj["arguments"]; f = a["filters"]
            return (bool(a["query"].strip())
                    and f["region"] in REGIONS
                    and f["start_date"] <= f["end_date"]
                    and 1 <= f["limit"] <= 100)
        if name == "enum_heavy":
            return (bool(obj["justification"].strip())
                    and not (obj["severity"] == "p0"
                             and obj["action"] not in ("page", "auto_remediate")))
        if name == "array_of_objects":
            recs = obj["records"]
            if not recs:
                return False
            total = sum(r["qty"] * r["unit_price"] for r in recs)
            return (abs(total - obj["total"]) <= 0.02
                    and all(r["qty"] > 0 and r["sku"].startswith("SKU-") for r in recs))
    except (KeyError, TypeError, AttributeError):
        return False
    return False


# --------------------------------------------------------------------------
# Approximate token boundaries.
#
# Real truncation happens at BPE token boundaries. Without the target model's
# tokenizer we approximate: structural punctuation is its own token, quoted
# strings break at whitespace, numbers break at digit runs. This is coarser
# than a real BPE vocabulary but it is far closer than every character, and
# critically it stops over-weighting long string fields.
# --------------------------------------------------------------------------

_TOK = re.compile(r'[{}\[\],:]|"|[A-Za-z_]+|\d+|\.|\s+|.')


def token_boundaries(doc):
    """Byte offsets at which a token-aligned generation could have stopped."""
    out, i = [], 0
    for mt in _TOK.finditer(doc):
        i = mt.end()
        if 0 < i < len(doc):
            out.append(i)
    return out


# --------------------------------------------------------------------------
# The repair heuristic. NOT just bracket-closing -- see module docstring.
# --------------------------------------------------------------------------

def repair(prefix: str):
    stack, in_string, escaped = [], False, False
    for ch in prefix:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    out = prefix
    if in_string:                      # close an unterminated string
        if escaped:
            out = out[:-1]
        out += '"'

    s = out.rstrip()
    while s and s[-1] in ",:":         # drop dangling separators and partial keys
        if s[-1] == ":":
            s = s[:-1].rstrip()
            if s.endswith('"'):
                i = len(s) - 2
                while i >= 0 and not (s[i] == '"' and s[i - 1] != "\\"):
                    i -= 1
                s = s[:i].rstrip()
        else:
            s = s[:-1].rstrip()
        s = s.rstrip(",").rstrip()

    for closer in reversed(stack):     # close open brackets
        s += "}" if closer == "{" else "]"
    return s


def try_parse(s):
    try:
        return json.loads(s)
    except Exception:
        return None


# --------------------------------------------------------------------------

def run(n_docs=200):
    random.seed(SEED)
    validators = {k: Draft7Validator(v) for k, v in SCHEMAS.items()}
    results = {}

    for name, gen in GENERATORS.items():
        v = validators[name]
        # Assert the ground-truth floor before measuring anything against it.
        probe = [gen() for _ in range(2000)]
        floor = sum(semantic_check(name, d) for d in probe) / len(probe)
        assert floor == 1.0, f"{name}: ground-truth semantic floor is {floor}, not 1.0"

        by_decile = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        n_cuts = 0

        for _ in range(n_docs):
            original = gen()
            doc = json.dumps(original, separators=(",", ":"))
            cuts = token_boundaries(doc)
            n_cuts += len(cuts)
            for cut in cuts:
                dec = min(9, int(10 * cut / len(doc)))
                by_decile[dec]["n"] += 1
                totals["n"] += 1

                robj = try_parse(repair(doc[:cut]))
                if robj is None:
                    continue
                for key, hit in (
                    ("parses", True),
                    ("schema_valid", v.is_valid(robj)),
                    ("exact", robj == original),
                    ("semantic", semantic_check(name, robj)),
                ):
                    if hit:
                        by_decile[dec][key] += 1
                        totals[key] += 1

        def rate(d, k):
            return round(d[k] / d["n"], 4) if d["n"] else None

        results[name] = {
            "ground_truth_semantic_floor": floor,
            "docs": n_docs,
            "token_cut_points": n_cuts,
            "median_doc_chars": None,
            "pooled": {k: rate(totals, k)
                       for k in ("parses", "schema_valid", "exact", "semantic")},
            "by_decile": {
                f"{d*10}-{d*10+10}%": {k: rate(by_decile[d], k)
                                       for k in ("parses", "schema_valid", "exact", "semantic")}
                for d in sorted(by_decile)
            },
        }
    return results


def bench_ladder(n=2000):
    """
    Cost of each validation rung, per document.

    NOTE: this times a SINGLE loop per operation. It is not a distribution and
    not a median. Timings are strongly environment-dependent -- `jsonschema`
    3.2.0 and 4.26.0 differ by roughly 6x on the same documents -- so report
    your Python and jsonschema versions with any figure taken from here.
    """
    import time
    random.seed(SEED)
    validators = {k: Draft7Validator(v) for k, v in SCHEMAS.items()}
    out = {}
    for name, gen in GENERATORS.items():
        docs = [json.dumps(gen(), separators=(",", ":")) for _ in range(n)]
        objs = [json.loads(d) for d in docs]
        v = validators[name]

        t0 = time.perf_counter()
        for d in docs:
            json.loads(d)
        t_parse = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for o in objs:
            v.is_valid(o)
        t_schema = (time.perf_counter() - t0) / n

        t0 = time.perf_counter()
        for o in objs:
            semantic_check(name, o)
        t_sem = (time.perf_counter() - t0) / n

        out[name] = {
            "median_doc_chars": statistics.median(len(d) for d in docs),
            "parse_us": round(t_parse * 1e6, 2),
            "schema_validate_us": round(t_schema * 1e6, 2),
            "semantic_validate_us": round(t_sem * 1e6, 2),
            "schema_over_parse": round(t_schema / t_parse, 1),
        }
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print("NOTE: raw-prefix parse rate is omitted. Every document here is a top-level")
    print("JSON object, so no proper prefix can parse. That is a tautology, not a result.\n")
    r = run(n)
    for name, res in r.items():
        print("=" * 74)
        print(f"{name}  (ground-truth semantic floor = {res['ground_truth_semantic_floor']:.4f}, "
              f"{res['token_cut_points']} token cut points)")
        print(f"  pooled: {res['pooled']}")
        print(f"  {'decile':>10} | {'parses':>7} | {'schema':>7} | {'exact':>7} | {'semantic':>8}")
        for k, d in res["by_decile"].items():
            print(f"  {k:>10} | {d['parses']:7.4f} | {d['schema_valid']:7.4f} | "
                  f"{d['exact']:7.4f} | {d['semantic']:8.4f}")
    print()
    print("=" * 74)
    print("VALIDATION LADDER (single timed loop per operation, not a median)")
    print(f"python {sys.version.split()[0]}, jsonschema {jsonschema_version}")
    print("=" * 74)
    lad = bench_ladder()
    print(f"  {'family':<18} | {'chars':>6} | {'parse':>8} | {'schema':>9} | "
          f"{'semantic':>9} | {'sch/parse':>9}")
    for k, d in lad.items():
        print(f"  {k:<18} | {d['median_doc_chars']:6.0f} | {d['parse_us']:7.2f}µ | "
              f"{d['schema_validate_us']:8.2f}µ | {d['semantic_validate_us']:8.2f}µ | "
              f"{d['schema_over_parse']:8.1f}×")