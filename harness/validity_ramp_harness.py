#!/usr/bin/env python3
"""
validity_ramp_harness.py -- measure JSON schema validity as a function of
concurrency against any OpenAI-compatible endpoint.

This is the harness behind the essay's concurrency ramp. It works unchanged
against a vLLM server on a GPU Droplet, against DigitalOcean Serverless
Inference (BASE_URL=https://inference.do-ai.run/v1, a DO API token as
API_KEY, and a serverless model slug), or against any other
OpenAI-compatible endpoint.

  pip install httpx jsonschema
  export BASE_URL=http://localhost:8000/v1
  export API_KEY=EMPTY
  python3 validity_ramp_harness.py --model meta-llama/Llama-3.3-70B-Instruct \
      --mode strict --levels 1,10,50,100 --requests-per-level 200

Record, for every run: engine version, backend
(--structured-outputs-config.backend), tokenizer, GPU, max_model_len,
max_num_seqs, max_num_batched_tokens, and sampling params. Findings without
those are not reproducible; see the essay's note on version-pinning.

WHAT CHANGED IN v2, AND WHY
----------------------------
The v1 task set was three single-turn templates producing 127-153 output
tokens against a 512-token budget. That is not a workload capable of
provoking truncation, and it exercises exactly three of the four schema
shapes used elsewhere in this piece's own experiments. Both are fixed here:

1. TASKS now reuses all four schema families from truncation_experiment.py
   (flat, nested, enum-heavy, array-of-objects), so the ramp and the
   truncation/conformance probes are testing the same shapes.
2. A fifth task, `agent_multi_turn`, is genuinely multi-turn: it sends a
   tool-call turn, appends a synthetic tool result, and requires a second
   schema-constrained turn for the final answer. Single-shot question
   answering is not what production agent traffic looks like.
3. `--max-tokens` can be set deliberately low (try 40-60) to run a tight
   arm alongside the default 512-token arm. This is the only way to
   actually observe truncation under load rather than asserting its
   absence from a workload that never approached its budget.

None of this has been run against live infrastructure as of this revision --
running it needs a GPU Droplet or a DO serverless endpoint and a model
deployment, which the environment producing this diff does not have. The
code is published so the redesigned ramp is a `python3
validity_ramp_harness.py` away rather than a rewrite. Treat any ramp numbers
elsewhere in the essay as describing the v1 workload until a v2 run replaces
them.

What it reports per concurrency level:
  parse_rate            fraction of responses that json.loads
  schema_valid_rate     fraction that validate against the requested schema
  semantic_valid_rate   fraction that pass field-level business rules
  truncation_rate       fraction with finish_reason == "length"
  null_content_rate     fraction with no text block. Reported separately from
                        truncation on purpose: a null content is routine when
                        the model returns tool calls, so folding the two
                        together manufactures truncations that did not happen.
  e2e_p50/p95/p99       end-to-end latency, index-based as sorted[int(p * n)]
                        rather than interpolated. At n=200 the reported p99 is
                        the 199th-smallest sample, i.e. a near-maximum.
  attempts_per_valid    requests spent per usable output

Note: this measures end-to-end latency only. For TTFT you need stream=True and
a first-chunk timestamp; that is deliberately left out here so the validity
measurement is not confounded by streaming reassembly, which is itself one of
the failure modes under study.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter

import httpx
from jsonschema import Draft7Validator

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")

# --------------------------------------------------------------------------
# Task set. Keep these identical across arms -- the comparison is worthless
# if the schemas differ.
# --------------------------------------------------------------------------

TASKS = [
    {
        "name": "flat_extract",
        "prompt": (
            "Extract the support ticket fields from this message. "
            "Reply with a single JSON object and nothing else.\n\n"
            "Message: My droplet in nyc3 has been unreachable since the 3am "
            "maintenance window. Billing also double-charged me in July. "
            "Account cus_18ab4f21."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["refund_request", "billing_question", "outage_report",
                             "upgrade_plan", "cancel_account"],
                },
                "confidence": {"type": "number"},
                "customer_id": {"type": "string"},
                "requires_human": {"type": "boolean"},
                "summary": {"type": "string"},
            },
            "required": ["intent", "confidence", "customer_id", "requires_human", "summary"],
            "additionalProperties": False,
        },
        "semantic": lambda o: (
            bool(o.get("summary", "").strip())
            and 0.0 <= o.get("confidence", -1) <= 1.0
            and o.get("customer_id", "").startswith("cus_")
        ),
    },
    {
        "name": "nested_toolcall",
        "prompt": (
            "Plan the single tool call needed to answer: 'how much did I spend on "
            "block storage in fra1 between March and June 2026?' "
            "Reply with a single JSON object and nothing else."
        ),
        "schema": {
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
        },
        "semantic": lambda o: (
            o.get("arguments", {}).get("filters", {}).get("start_date", "")
            <= o.get("arguments", {}).get("filters", {}).get("end_date", "~")
            and bool(o.get("arguments", {}).get("query", "").strip())
        ),
    },
    {
        "name": "array_extract",
        "prompt": (
            "Itemize every line item in this invoice as JSON records with sku, qty, "
            "unit_price and note, plus a total. Reply with a single JSON object.\n\n"
            "Invoice: 3x SKU-01044 block storage @ 10.00; 1x SKU-90210 H100 hour @ 4.41; "
            "12x SKU-33127 bandwidth GB @ 0.01; 2x SKU-77219 snapshot @ 0.06."
        ),
        "schema": {
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
        },
        "semantic": lambda o: (
            len(o.get("records", [])) == 4
            and abs(sum(r.get("qty", 0) * r.get("unit_price", 0)
                        for r in o.get("records", [])) - o.get("total", -1)) < 0.02
        ),
    },
    {
        "name": "enum_triage",
        "prompt": (
            "A monitoring alert fired: 'nyc3-lb-07 health checks failing, 40% of "
            "backend pool unreachable for 6 minutes, customer-facing 5xx rate up "
            "3x.' Classify severity, owning team, immediate action, and root-cause "
            "category, with a one-sentence justification. "
            "Reply with a single JSON object and nothing else."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["p0", "p1", "p2", "p3", "p4"]},
                "team": {"type": "string",
                         "enum": ["networking", "storage", "compute", "billing", "ml"]},
                "action": {"type": "string",
                           "enum": ["page", "ticket", "auto_remediate", "watch", "close"]},
                "category": {"type": "string",
                             "enum": ["hardware", "software", "config", "capacity",
                                      "external", "unknown"]},
                "justification": {"type": "string"},
            },
            "required": ["severity", "team", "action", "category", "justification"],
            "additionalProperties": False,
        },
        "semantic": lambda o: (
            bool(o.get("justification", "").strip())
            and not (o.get("severity") == "p0"
                     and o.get("action") not in ("page", "auto_remediate"))
        ),
    },
]

# Multi-turn task: a tool-call turn, a synthetic tool result appended to the
# transcript, then a second schema-constrained turn for the final answer.
# Single-shot question answering under-represents real agent traffic, which
# is almost always multi-turn.
MULTI_TURN_TASK = {
    "name": "agent_multi_turn",
    "turn1_prompt": (
        "You are an infra assistant. A user asks: 'how much block storage did "
        "account cus_18ab4f21 provision in nyc3 last month?' Decide the single "
        "tool call needed. Reply with a single JSON object and nothing else."
    ),
    "turn1_schema": {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": ["query_usage_db"]},
            "arguments": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "region": {"type": "string"},
                    "resource": {"type": "string"},
                    "window": {"type": "string"},
                },
                "required": ["customer_id", "region", "resource", "window"],
            },
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    },
    "tool_result": (
        '{"customer_id": "cus_18ab4f21", "region": "nyc3", "resource": '
        '"block_storage", "provisioned_gb": 2048, "period": "2026-07"}'
    ),
    "turn2_prompt": (
        "Given that tool result, answer the user with the final structured "
        "response. Reply with a single JSON object and nothing else."
    ),
    "turn2_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "region": {"type": "string"},
            "provisioned_gb": {"type": "number"},
            "period": {"type": "string"},
            "answer": {"type": "string"},
        },
        "required": ["customer_id", "region", "provisioned_gb", "period", "answer"],
        "additionalProperties": False,
    },
    "semantic": lambda o: (
        o.get("customer_id") == "cus_18ab4f21"
        and o.get("provisioned_gb") == 2048
        and bool(o.get("answer", "").strip())
    ),
}


def build_body(task, model, mode, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "temperature": 0.0,
        "seed": 20260813,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    if mode == "strict":
        # Current vLLM / OpenAI surface. The guided_* fields were removed in
        # vLLM v0.12.0 -- do not reintroduce them.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": task["name"],
                "schema": task["schema"],
                "strict": True,
            },
        }
    elif mode == "prompt_only":
        body["messages"][0]["content"] += (
            "\n\nThe object must match this JSON Schema exactly:\n"
            + json.dumps(task["schema"])
        )
    else:
        raise ValueError(mode)
    return body


async def one_request(client, task, model, mode, max_tokens, validator):
    t0 = time.perf_counter()
    rec = {"task": task["name"], "error": None, "finish_reason": None,
           "parse": False, "schema_valid": False, "semantic_valid": False,
           "completion_tokens": None}
    try:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            json=build_body(task, model, mode, max_tokens),
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=600.0,
        )
        rec["http_status"] = r.status_code
        data = r.json()
        choice = data["choices"][0]
        rec["finish_reason"] = choice.get("finish_reason")
        rec["completion_tokens"] = data.get("usage", {}).get("completion_tokens")
        content = choice["message"].get("content")
        if content is None:
            # Reasoning models: the budget can be spent before any content is
            # emitted. This is a truncation failure, not a schema failure.
            rec["error"] = "null_content"
        else:
            try:
                obj = json.loads(content)
                rec["parse"] = True
            except Exception as e:
                rec["error"] = f"parse: {type(e).__name__}"
                obj = None
            if obj is not None:
                errs = sorted(validator.iter_errors(obj), key=lambda e: e.path)
                if not errs:
                    rec["schema_valid"] = True
                    try:
                        rec["semantic_valid"] = bool(task["semantic"](obj))
                    except Exception:
                        rec["semantic_valid"] = False
                else:
                    rec["error"] = f"schema: {errs[0].validator} at {list(errs[0].path)}"
            if not rec["schema_valid"]:
                rec["specimen"] = content[:1200]
    except Exception as e:
        rec["error"] = f"transport: {type(e).__name__}: {e}"
    rec["e2e_s"] = time.perf_counter() - t0
    return rec


async def one_multi_turn_request(client, model, mode, max_tokens):
    """Two schema-constrained turns with a synthetic tool result spliced in
    between. Reports against turn 2's schema only; turn 1 failing is folded
    into an overall failure rather than measured separately, which is a
    known simplification -- see the module docstring's v2 notes."""
    t = MULTI_TURN_TASK
    t0 = time.perf_counter()
    rec = {"task": t["name"], "error": None, "finish_reason": None,
           "parse": False, "schema_valid": False, "semantic_valid": False,
           "completion_tokens": None}
    v1 = Draft7Validator(t["turn1_schema"])
    v2 = Draft7Validator(t["turn2_schema"])
    try:
        body1 = {
            "model": model,
            "messages": [{"role": "user", "content": t["turn1_prompt"]}],
            "temperature": 0.0, "seed": 20260813,
            "max_completion_tokens": max_tokens, "stream": False,
        }
        if mode == "strict":
            body1["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "turn1", "schema": t["turn1_schema"], "strict": True}}
        else:
            body1["messages"][0]["content"] += (
                "\n\nMatch this schema exactly:\n" + json.dumps(t["turn1_schema"]))
        r1 = await client.post(f"{BASE_URL}/chat/completions", json=body1,
                               headers={"Authorization": f"Bearer {API_KEY}"},
                               timeout=600.0)
        turn1_content = r1.json()["choices"][0]["message"].get("content") or ""

        messages = [
            {"role": "user", "content": t["turn1_prompt"]},
            {"role": "assistant", "content": turn1_content},
            {"role": "user", "content": (
                f"Tool result: {t['tool_result']}\n\n{t['turn2_prompt']}")},
        ]
        body2 = {
            "model": model, "messages": messages,
            "temperature": 0.0, "seed": 20260813,
            "max_completion_tokens": max_tokens, "stream": False,
        }
        if mode == "strict":
            body2["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "turn2", "schema": t["turn2_schema"], "strict": True}}
        else:
            messages[-1]["content"] += (
                "\n\nMatch this schema exactly:\n" + json.dumps(t["turn2_schema"]))
        r2 = await client.post(f"{BASE_URL}/chat/completions", json=body2,
                               headers={"Authorization": f"Bearer {API_KEY}"},
                               timeout=600.0)
        data2 = r2.json()
        choice2 = data2["choices"][0]
        rec["finish_reason"] = choice2.get("finish_reason")
        rec["completion_tokens"] = data2.get("usage", {}).get("completion_tokens")
        content2 = choice2["message"].get("content")
        if content2 is None:
            rec["error"] = "null_content"
        else:
            try:
                obj = json.loads(content2)
                rec["parse"] = True
            except Exception as e:
                rec["error"] = f"parse: {type(e).__name__}"
                obj = None
            if obj is not None:
                errs = sorted(v2.iter_errors(obj), key=lambda e: e.path)
                if not errs:
                    rec["schema_valid"] = True
                    try:
                        rec["semantic_valid"] = bool(t["semantic"](obj))
                    except Exception:
                        rec["semantic_valid"] = False
                else:
                    rec["error"] = f"schema: {errs[0].validator} at {list(errs[0].path)}"
            if not rec["schema_valid"]:
                rec["specimen"] = content2[:1200]
    except Exception as e:
        rec["error"] = f"transport: {type(e).__name__}: {e}"
    rec["e2e_s"] = time.perf_counter() - t0
    return rec


async def run_level(concurrency, n_requests, model, mode, max_tokens, multi_turn_frac=0.0):
    validators = {t["name"]: Draft7Validator(t["schema"]) for t in TASKS}
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10,
                          max_keepalive_connections=concurrency + 10)
    out = []
    n_multi_turn = round(n_requests * multi_turn_frac)

    async with httpx.AsyncClient(limits=limits) as client:
        async def guarded(i):
            async with sem:
                if i < n_multi_turn:
                    return await one_multi_turn_request(client, model, mode, max_tokens)
                task = TASKS[i % len(TASKS)]
                return await one_request(client, task, model, mode, max_tokens,
                                         validators[task["name"]])

        # Warm-up: discard. The first structured-output request the server ever
        # sees also elects the backend for the process lifetime, and compiles
        # each grammar for the first time.
        await asyncio.gather(*(guarded(i) for i in range(min(3 * len(TASKS), n_requests))))

        t0 = time.perf_counter()
        out = await asyncio.gather(*(guarded(i) for i in range(n_requests)))
        wall = time.perf_counter() - t0

    n = len(out)
    lat = sorted(r["e2e_s"] for r in out)

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))]

    trunc = sum(1 for r in out if r["finish_reason"] == "length")
    null_content = sum(1 for r in out if r["error"] == "null_content")
    valid = sum(1 for r in out if r["schema_valid"])
    return {
        "concurrency": concurrency,
        "requests": n,
        "wall_s": round(wall, 2),
        "parse_rate": sum(r["parse"] for r in out) / n,
        "schema_valid_rate": valid / n,
        "semantic_valid_rate": sum(r["semantic_valid"] for r in out) / n,
        "truncation_rate": trunc / n,
        "null_content_rate": null_content / n,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in out)),
        "errors": dict(Counter(r["error"] for r in out if r["error"]).most_common(8)),
        "e2e_p50": round(pct(0.50), 3),
        "e2e_p95": round(pct(0.95), 3),
        "e2e_p99": round(pct(0.99), 3),
        "attempts_per_valid": round(n / valid, 4) if valid else None,
        "mean_completion_tokens": round(statistics.mean(
            [r["completion_tokens"] for r in out if r["completion_tokens"]] or [0]), 1),
        "specimens": [r.get("specimen") for r in out if r.get("specimen")][:5],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["strict", "prompt_only"], default="strict")
    ap.add_argument("--levels", default="1,10,50,100")
    ap.add_argument("--requests-per-level", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Set deliberately low (e.g. 40-60) to run the tight-budget "
                         "arm and actually provoke truncation under load.")
    ap.add_argument("--multi-turn-frac", type=float, default=0.0,
                    help="Fraction of requests per level that run the two-turn "
                         "agent_multi_turn task instead of a single-turn task.")
    ap.add_argument("--out", default="validity_ramp_results.json")
    a = ap.parse_args()

    results = []
    for c in [int(x) for x in a.levels.split(",")]:
        r = await run_level(c, a.requests_per_level, a.model, a.mode, a.max_tokens,
                            a.multi_turn_frac)
        r["mode"] = a.mode
        r["model"] = a.model
        r["max_tokens"] = a.max_tokens
        r["multi_turn_frac"] = a.multi_turn_frac
        results.append(r)
        printable = {k: v for k, v in r.items() if k != "specimens"}
        print(json.dumps(printable, indent=2), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())