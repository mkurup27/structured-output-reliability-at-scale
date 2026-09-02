#!/usr/bin/env python3
"""
validity_ramp_harness_managed.py -- measure JSON schema validity as a function of
concurrency against any OpenAI-compatible endpoint.

This is the harness that produced the essay's managed-endpoint arm. It is kept
for provenance only. Use validity_ramp_harness_v4.py for new work: it carries
every flag below under the same name and default, so a command line written for
this file runs there unchanged. Either works against a vLLM server on a GPU
Droplet, against DigitalOcean Serverless Inference, or against any other
OpenAI-compatible endpoint.

  pip install httpx jsonschema
  export BASE_URL=http://localhost:8000/v1
  export API_KEY=EMPTY
  python3 validity_ramp_harness_managed.py --model meta-llama/Llama-3.3-70B-Instruct \
      --mode strict --levels 1,10,50,100 --requests-per-level 200

Record, for every run: engine version, backend
(--structured-outputs-config.backend), tokenizer, GPU, max_model_len,
max_num_seqs, max_num_batched_tokens, and sampling params. Findings without
those are not reproducible; see the essay's note on version-pinning.

What it reports per concurrency level:
  parse_rate            fraction of responses that json.loads
  schema_valid_rate     fraction that validate against the requested schema
  semantic_valid_rate   fraction that pass field-level business rules
  truncation_rate       fraction with finish_reason == "length" or null content
  e2e_p50/p95/p99       end-to-end latency percentiles
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
]

# ---------------------------------------------------------------------------
# Edge-case task set (H3): schemas carrying keywords at/near xgrammar's
# enforcement boundary, including the untyped-fragment case the backend probe
# found xgrammar silently ignores. Selectable via --taskset edge.
# ---------------------------------------------------------------------------
TASKS_EDGE = [
    {
        "name": "untyped_pattern",
        "prompt": (
            "Return a JSON object with a single key 'code'. "
            "Set code to the lowercase word 'hello'. Reply with JSON only."
        ),
        # 'pattern' with NO sibling "type" on the fragment — xgrammar's preflight
        # gate keys off obj.get("type"), so this constraint can fall through.
        # We demand a value (lowercase) that VIOLATES the uppercase pattern:
        # if enforcement works, output is forced uppercase; if it silently
        # fails, the model obeys the prompt and returns lowercase.
        "schema": {
            "type": "object",
            "properties": {
                "code": {"pattern": "^[A-Z]+$"},
            },
            "required": ["code"],
        },
        "semantic": lambda o: isinstance(o.get("code"), str) and o.get("code", "").isupper(),
    },
    {
        "name": "multipleof_edge",
        "prompt": (
            "Return a JSON object with a single integer key 'value'. "
            "Set value to 7. Reply with JSON only."
        ),
        # multipleOf is on xgrammar's rejection list (server refuses the schema)
        # but guidance enforces it. Demanding 7 against multipleOf:10 tests which
        # regime you're in. On xgrammar this whole task should error at request
        # time (constraint-boundary-by-rejection), a distinct outcome worth seeing.
        "schema": {
            "type": "object",
            "properties": {
                "value": {"type": "integer", "multipleOf": 10},
            },
            "required": ["value"],
        },
        "semantic": lambda o: isinstance(o.get("value"), int) and o.get("value", 1) % 10 == 0,
    },
    {
        "name": "typed_control",
        "prompt": (
            "Return a JSON object with a single key 'code'. "
            "Set code to the lowercase word 'hello'. Reply with JSON only."
        ),
        # Typed twin of untyped_pattern: identical request, but WITH "type":"string".
        # xgrammar's gate should catch this one and force uppercase. The contrast
        # between this and untyped_pattern is the whole H3-adjacent finding.
        "schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": "^[A-Z]+$"},
            },
            "required": ["code"],
        },
        "semantic": lambda o: isinstance(o.get("code"), str) and o.get("code", "").isupper(),
    },
]

def build_body(task, model, mode, max_tokens, temperature=0.0, seed=20260813):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        body["seed"] = seed
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


async def one_request(client, task, model, mode, max_tokens, validator, temperature=0.0, seed=20260813):
    t0 = time.perf_counter()
    rec = {"task": task["name"], "error": None, "finish_reason": None,
           "parse": False, "schema_valid": False, "semantic_valid": False,
           "completion_tokens": None}
    try:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            json=build_body(task, model, mode, max_tokens, temperature, seed),
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=600.0,
        )
        rec["http_status"] = r.status_code
        data = r.json()
        if "choices" not in data or not data["choices"]:
            rec["error"] = f"schema_rejected: HTTP {r.status_code}"
            rec["finish_reason"] = "rejected"
            rec["e2e_s"] = time.perf_counter() - t0
            return rec
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


async def run_level(concurrency, n_requests, model, mode, max_tokens, temperature=0.0, seed=20260813):
    validators = {t["name"]: Draft7Validator(t["schema"]) for t in TASKS}
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 10,
                          max_keepalive_connections=concurrency + 10)
    out = []

    async with httpx.AsyncClient(limits=limits) as client:
        async def guarded(i):
            task = TASKS[i % len(TASKS)]
            async with sem:
                return await one_request(client, task, model, mode, max_tokens,
                                         validators[task["name"]], temperature, seed)

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

    trunc = sum(1 for r in out if r["finish_reason"] == "length" or r["error"] == "null_content")
    valid = sum(1 for r in out if r["schema_valid"])
    return {
        "concurrency": concurrency,
        "requests": n,
        "wall_s": round(wall, 2),
        "parse_rate": sum(r["parse"] for r in out) / n,
        "schema_valid_rate": valid / n,
        "semantic_valid_rate": sum(r["semantic_valid"] for r in out) / n,
        "truncation_rate": trunc / n,
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
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260813, 
                    help="fixed seed for determinism; pass -1 for a varying seed (real sampling)")
    ap.add_argument("--out", default="validity_ramp_results.json")
    ap.add_argument("--taskset", choices=["default", "edge"], default="default",
                help="which task set to run: default (3 original) or edge (H3 boundary schemas)")
    a = ap.parse_args()
    global TASKS
    if a.taskset == "edge":
        TASKS = TASKS_EDGE
    print(f"[taskset={a.taskset}, {len(TASKS)} tasks: {[t['name'] for t in TASKS]}]", flush=True)

    results = []
    for c in [int(x) for x in a.levels.split(",")]:
        r = await run_level(c, a.requests_per_level, a.model, a.mode, a.max_tokens,
                            a.temperature, None if a.seed == -1 else a.seed)
        r["mode"] = a.mode
        r["model"] = a.model
        r["max_tokens"] = a.max_tokens
        r["temperature"] = a.temperature
        r["seed"] = None if a.seed == -1 else a.seed
        results.append(r)
        printable = {k: v for k, v in r.items() if k != "specimens"}
        print(json.dumps(printable, indent=2), flush=True)

    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
