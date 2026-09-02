#!/usr/bin/env python3
"""
validity_ramp_harness_v4.py -- schema validity as a function of concurrency
against any OpenAI-compatible endpoint, with the instrumentation the earlier
ramps lacked.

LINEAGE, because there are several of these now and the numbers differ by
version. v1 was the published ramp: three single-turn templates, fixed count,
no instrumentation. v2 added --taskset, --temperature/--seed and the
choiceless-body guard, and produced the tight-budget, edge-schema and
varying-seed arms. v3 added client-side TTFT via streaming. This file is a
single-file consolidation of all of that plus sustained load, server-side
contention signals, a multi-turn agent task and per-request taxonomy tagging,
written as one file rather than a patch series so there is nothing to bisect if
it misbehaves. Every v3 flag exists here with the same name and default, so a
v3 command line runs unchanged. The `default` taskset below was checked against
the v3 file that produced the published arms and is the same three tasks in the
same order, so per-level rates are comparable across the two -- which is not
true of any other taskset here, since each one changes the denominator.

Same contract as v1/v2/v3: point it at a vLLM server on a GPU Droplet, at
DigitalOcean Serverless Inference (BASE_URL=https://inference.do-ai.run/v1, a DO
API token as API_KEY, a serverless model slug), or at any other
OpenAI-compatible endpoint.

  pip install httpx jsonschema
  export BASE_URL=http://localhost:8000/v1 API_KEY=EMPTY
  python3 validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
      --mode strict --levels 1,10,50,100 --requests-per-level 200

WHAT v4 ADDS, AND WHICH OPEN QUESTION EACH ONE ANSWERS
-------------------------------------------------------
v1 and v2 could show *whether* validity moved but not attribute the movement to
anything, and they offered load in fixed-count bursts that never reached steady
state. Four additions, each tied to a specific gap:

1. Duration-based sustained load (--duration-s). A closed loop of `concurrency`
   persistent workers issuing back-to-back requests for a fixed wall-clock
   window, with a separate discarded warm-up window. 200 requests at
   concurrency 100 is two request waves; a 180-second window at concurrency 100
   is sustained load with a populated KV cache and a real queue. This is the
   only mode that can test whether the flat-validity finding survives cache
   pressure rather than outrunning it.

2. Server-side contention signals (--metrics-url). Samples vLLM's /metrics
   endpoint *only across the measured window*, so preemption counts and
   KV-cache occupancy line up with the validity numbers from the same window
   instead of being scraped over an unrelated interval. Reports the preemption
   delta over the window, not the process-lifetime counter.

3. Client-side TTFT (--stream). Streams the response and timestamps the first
   content chunk. Run it as a separate latency pass: streaming changes the
   reassembly path, which is itself one of the failure modes under study, so
   mixing it into the validity arm would confound the thing being measured.

4. Failure-taxonomy tagging (always on). Every request is stamped with exactly
   one category from the essay's five-way taxonomy, by the deterministic rule
   in classify() below. A per-rung boolean tells you a request failed; a
   category tells you which of five different fixes applies.

5. A genuinely multi-turn agent task (--taskset multiturn | agent_mix). A
   tool-call turn, a synthetic tool result spliced into the transcript, then a
   second schema-constrained turn for the final answer, with the intermediate
   turn validated on its own so a broken tool call is distinguishable from a
   broken answer. Single-shot question answering is not what agent traffic
   looks like, and every ramp in the essay so far has been single-shot.

6. A five-step agent task (--taskset deep_multiturn | agent_mix_deep). The
   two-turn task above returned a clean null: adding one turn introduced no
   failure mode. That is a real result with a narrow scope, because the failure
   multi-turn is supposed to produce is context growth pushing later turns
   against the token budget, and two turns is too shallow for the transcript to
   grow. Five turns carry a real prefix: two tool calls, an arithmetic step, a
   planning step with an array, then a final answer that has to still be
   carrying values first seen in turn 1. Every turn is schema-validated, and
   turns with a `semantic` rule are semantically checked too, so a run says
   *which* step lost the thread rather than only that the answer was wrong.

7. Schema cardinality (--schema-cardinality N). Every arm so far offered a
   handful of schemas that compiled once on the warm-up and were served from
   the grammar cache thereafter, which means the compile cost was measured
   exactly once and the cache was never made to evict. This flag synthesizes N
   structurally distinct schemas -- differing in field count, field names,
   types and enum members, so no two compile to the same grammar -- and
   round-robins them, one per request. Round-robin is the worst case for an
   LRU cache on purpose: a schema's next request arrives only after every
   other schema has been compiled, so if the cache holds fewer than N entries
   every repeat is a miss. Past the backend's compiler cache size this puts
   recompilation on the request path, which is the one contention mechanism
   with a plausible route into constrained decoding that request count alone
   cannot reach. Offer at least 2N requests per level or the ramp never reaches
   the schemas at the end of the rotation; the results report
   `distinct_schemas_used` so you can check that it did.

   A first ladder at 64/256/1024/2048 came back flat -- TTFT p50 within 25 ms
   across a 32x cardinality increase, preemption zero -- which most likely
   means the cache never evicted rather than that eviction is free. 512 MiB
   over 2048 entries is 256 KiB each, and these are small flat objects. The
   cache size is the variable to move, not the schema count: restart the server
   with VLLM_XGRAMMAR_CACHE_MB well below the default and the same ladder
   reaches eviction at a cardinality the client can actually offer.

Also folded in from the interim v2 patches: --taskset (default | four | edge),
--temperature / --seed (--seed -1 for independent draws), a guard that turns a
choiceless error body into a labelled `schema_rejected` outcome rather than a
KeyError, measured cost-per-valid-output from actual token usage, and
--dump-records so per-request outputs survive the run.

REPORTING DISCIPLINE
--------------------
Record with every run: engine version, the *resolved* structured-output backend
and its library version, tokenizer, GPU, max_model_len, max_num_seqs,
max_num_batched_tokens, and sampling params. "Validity was X at vLLM 0.27.1
with xgrammar" does not identify a build.

Percentiles are index-based, sorted[int(p * n)], not interpolated. At n=200 the
reported p99 is the 199th-smallest sample, i.e. a near-maximum.
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
# Task sets. Keep these identical across arms -- the comparison is worthless
# if the schemas differ between the arms being compared.
#
# DEFAULT_TASKS is the three-template set the published ramp used, preserved
# verbatim -- verified identical to the v3 file that produced the published
# arms, so per-level rates are comparable across the two versions.
# `four` adds enum_triage,
# which changes every denominator: the published 0.665 semantic rate is
# 2-of-3, and the same run over four tasks would read 0.75 for an unrelated
# reason. Do not compare across task sets.
# --------------------------------------------------------------------------

FLAT_EXTRACT = {
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
}

NESTED_TOOLCALL = {
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
}

ARRAY_EXTRACT = {
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
    # The line items sum to 34.65. This checker is what catches a schema-valid
    # document carrying the wrong number, and it is the only rung that does.
    "semantic": lambda o: (
        len(o.get("records", [])) == 4
        and abs(sum(r.get("qty", 0) * r.get("unit_price", 0)
                    for r in o.get("records", [])) - o.get("total", -1)) < 0.02
    ),
}

ENUM_TRIAGE = {
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
}

# Edge task set: three schemas sitting on xgrammar's enforcement boundary, each
# with a prompt that fights the constraint. The point is not the pass rate, it
# is that the three outcomes are *different kinds* of failure -- a silent
# under-enforcement, an honest request-time rejection, and a clean pass -- and
# that the split does not move with load.
EDGE_UNTYPED_PATTERN = {
    "name": "edge_untyped_pattern",
    # No sibling "type" on `code`. xgrammar's preflight gate tests
    # obj.get("type") == ... before looking at the keyword, so this fragment
    # reaches the compiler unenforced. The local validator still checks it,
    # which is how the violation becomes visible at all.
    "prompt": (
        "Return a JSON object with one field, `code`, set to the lowercase "
        "string 'abc'. Use lowercase letters exactly as written. "
        "Reply with a single JSON object and nothing else."
    ),
    "schema": {
        "type": "object",
        "properties": {"code": {"pattern": "^[A-Z]+$"}},
        "required": ["code"],
    },
    "semantic": lambda o: isinstance(o.get("code"), str) and o["code"].isupper(),
}

EDGE_MULTIPLEOF = {
    "name": "edge_multipleof",
    # multipleOf on an integer is on xgrammar's rejection list: the server
    # refuses the schema outright. That is the honest failure mode, and it
    # arrives as an HTTP 5xx with no `choices` key -- the shape the v2 guard
    # was added to handle.
    "prompt": (
        "Return a JSON object with one field, `qty`, set to 7. "
        "Reply with a single JSON object and nothing else."
    ),
    "schema": {
        "type": "object",
        "properties": {"qty": {"type": "integer", "multipleOf": 5}},
        "required": ["qty"],
    },
    "semantic": lambda o: isinstance(o.get("qty"), int) and o["qty"] % 5 == 0,
}

EDGE_TYPED_CONTROL = {
    "name": "edge_typed_control",
    # Identical to EDGE_UNTYPED_PATTERN except for the sibling "type": "string".
    # That one word is the whole difference between enforced and ignored.
    "prompt": (
        "Return a JSON object with one field, `code`, set to the lowercase "
        "string 'abc'. Use lowercase letters exactly as written. "
        "Reply with a single JSON object and nothing else."
    ),
    "schema": {
        "type": "object",
        "properties": {"code": {"type": "string", "pattern": "^[A-Z]+$"}},
        "required": ["code"],
    },
    "semantic": lambda o: isinstance(o.get("code"), str) and o["code"].isupper(),
}

# Multi-turn agent task. Two schema-constrained turns with a synthetic tool
# result spliced between them, so the second turn carries a real conversation
# prefix rather than a bare instruction. Single-shot question answering
# under-represents production agent traffic, which is almost always multi-turn,
# and it is the workload-realism gap the essay's ramp is open on.
#
# Turn 1 is validated too, and a turn-1 failure is reported as its own outcome
# rather than folded into the overall rate: an agent that fails at the tool-call
# turn never reaches the answer turn, and the two need different fixes.
AGENT_MULTITURN = {
    "name": "agent_multiturn",
    "turns": [
        {
            "prompt": (
                "You are an infra assistant. A user asks: 'how much block "
                "storage did account cus_18ab4f21 provision in nyc3 last "
                "month?' Decide the single tool call needed. Reply with a "
                "single JSON object and nothing else."
            ),
            "schema": {
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
        },
        {
            "prompt": (
                "Given that tool result, answer the user with the final "
                "structured response. Reply with a single JSON object and "
                "nothing else."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "region": {"type": "string"},
                    "provisioned_gb": {"type": "number"},
                    "period": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["customer_id", "region", "provisioned_gb", "period",
                             "answer"],
                "additionalProperties": False,
            },
        },
    ],
    # run_level validates the final turn against this.
    "schema": None,  # filled in below from turns[-1]
    # The tool result says 2048 GB. A schema-valid answer carrying a different
    # number is the multi-turn version of the invoice-arithmetic failure: the
    # model had the right value in its context and did not use it.
    "semantic": lambda o: (
        o.get("customer_id") == "cus_18ab4f21"
        and o.get("provisioned_gb") == 2048
        and o.get("period") == "2026-07"
        and bool(o.get("answer", "").strip())
    ),
}
AGENT_MULTITURN["schema"] = AGENT_MULTITURN["turns"][-1]["schema"]


# Five-step agent task. AGENT_MULTITURN above returned a clean null -- adding a
# single turn introduced no new failure mode -- but two turns cannot exhibit the
# failure multi-turn is supposed to cause, which is the transcript growing until
# later turns are competing with their own context for the token budget. This
# task runs five schema-constrained turns over one continuous transcript:
#
#   1. tool call: look up the account's usage
#   2. tool call: look up the region's rate card
#   3. arithmetic: turn usage x rates into priced line items
#   4. planning: an array of recommended actions
#   5. answer: the final structured response
#
# The arithmetic is the load-bearing part. $204.80 of storage plus $96.00 of
# droplets is $300.80, and turn 5 has to report that total while still carrying
# customer_id and provisioned_gb, both of which it last saw in turn 1's tool
# result. A model that has lost the thread produces a schema-valid answer with a
# wrong number, which is the multi-turn form of the invoice-arithmetic failure.
# Turns 3 and 4 carry their own `semantic` rules so a failure is attributable to
# the step that lost it rather than only visible at the end.
AGENT_DEEP = {
    "name": "agent_deep",
    "turns": [
        {
            "prompt": (
                "You are an infra billing assistant. A user asks: 'what will "
                "account cus_18ab4f21 owe for nyc3 next month at last month's "
                "usage?' Decide the single tool call needed to get their usage "
                "first. Reply with a single JSON object and nothing else."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["query_usage_db"]},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "region": {"type": "string"},
                            "window": {"type": "string"},
                        },
                        "required": ["customer_id", "region", "window"],
                    },
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
            "tool_result": (
                '{"customer_id": "cus_18ab4f21", "region": "nyc3", '
                '"block_storage_gb": 2048, "droplet_count": 2, '
                '"period": "2026-07"}'
            ),
        },
        {
            "prompt": (
                "Now get the prices. Decide the single tool call needed to "
                "fetch the rate card for that region. Reply with a single JSON "
                "object and nothing else."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["query_rate_card"]},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "region": {"type": "string"},
                            "resources": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": ["region", "resources"],
                    },
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            },
            "tool_result": (
                '{"region": "nyc3", "block_storage_usd_per_gb_month": 0.10, '
                '"droplet_usd_per_month": 48.00}'
            ),
        },
        {
            "prompt": (
                "Price the usage against the rate card. Return one line item "
                "per resource, with the quantity, the unit price, and the "
                "subtotal for that line. Reply with a single JSON object and "
                "nothing else."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "enum": ["USD"]},
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "resource": {"type": "string"},
                                "quantity": {"type": "number"},
                                "unit_price_usd": {"type": "number"},
                                "subtotal_usd": {"type": "number"},
                            },
                            "required": ["resource", "quantity",
                                         "unit_price_usd", "subtotal_usd"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["currency", "items"],
                "additionalProperties": False,
            },
            # 2048 GB x $0.10 = $204.80; 2 droplets x $48.00 = $96.00.
            "semantic": lambda o: (
                abs(sum(i.get("subtotal_usd", 0) for i in o.get("items", []))
                    - 300.80) < 0.01
            ),
            "next_user": (
                "Those line items are confirmed. Keep them for the rest of "
                "this conversation."
            ),
        },
        {
            "prompt": (
                "Recommend what the account should do about this bill. Give at "
                "least two concrete actions, each naming the resource it "
                "applies to. Reply with a single JSON object and nothing else."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "projected_total_usd": {"type": "number"},
                    "actions": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string"},
                                "target": {"type": "string"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["action", "target", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["projected_total_usd", "actions"],
                "additionalProperties": False,
            },
            # The projection has to still match the priced total from turn 3,
            # four messages and two tool results back.
            "semantic": lambda o: abs(o.get("projected_total_usd", 0)
                                      - 300.80) < 0.01,
            "next_user": "Understood. Now summarize for the user.",
        },
        {
            "prompt": (
                "Give the final structured answer: who the account is, the "
                "region and period, the block storage they provisioned, the "
                "total owed, and a one-sentence answer. Reply with a single "
                "JSON object and nothing else."
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "region": {"type": "string"},
                    "period": {"type": "string"},
                    "block_storage_gb": {"type": "number"},
                    "total_usd": {"type": "number"},
                    "currency": {"type": "string", "enum": ["USD"]},
                    "answer": {"type": "string"},
                },
                "required": ["customer_id", "region", "period",
                             "block_storage_gb", "total_usd", "currency",
                             "answer"],
                "additionalProperties": False,
            },
        },
    ],
    "schema": None,  # filled in below from turns[-1]
    # customer_id and block_storage_gb were last seen in turn 1's tool result,
    # five turns of transcript ago; total_usd depends on turn 3's arithmetic
    # surviving to the end. Any of the three coming back wrong is a
    # schema-valid answer that lost the conversation.
    "semantic": lambda o: (
        o.get("customer_id") == "cus_18ab4f21"
        and o.get("region") == "nyc3"
        and o.get("period") == "2026-07"
        and o.get("block_storage_gb") == 2048
        and abs(o.get("total_usd", 0) - 300.80) < 0.01
        and bool(o.get("answer", "").strip())
    ),
}
AGENT_DEEP["schema"] = AGENT_DEEP["turns"][-1]["schema"]

TASKSETS = {
    "default": [FLAT_EXTRACT, NESTED_TOOLCALL, ARRAY_EXTRACT],
    "four": [FLAT_EXTRACT, NESTED_TOOLCALL, ARRAY_EXTRACT, ENUM_TRIAGE],
    "edge": [EDGE_UNTYPED_PATTERN, EDGE_MULTIPLEOF, EDGE_TYPED_CONTROL],
    "multiturn": [AGENT_MULTITURN],
    # Agent-shaped mixture: the multi-turn task alongside the single-turn ones,
    # which is closer to real traffic than either alone. Note the denominator
    # changes again -- four tasks, so per-task rates are quarters.
    "agent_mix": [AGENT_MULTITURN, FLAT_EXTRACT, NESTED_TOOLCALL, ARRAY_EXTRACT],
    "deep_multiturn": [AGENT_DEEP],
    # The five-step task against the same single-turn templates, so a
    # depth-driven failure is visible next to a same-run single-turn baseline
    # instead of against numbers from a different run.
    "agent_mix_deep": [AGENT_DEEP, FLAT_EXTRACT, NESTED_TOOLCALL, ARRAY_EXTRACT],
}


# --------------------------------------------------------------------------
# Schema cardinality. Synthesized distinct schemas, for grammar-cache churn.
# --------------------------------------------------------------------------
#
# Everything above offers a handful of schemas. They compile on the warm-up and
# every measured request is a cache hit, so the compile cost appears exactly
# once, off the measured window, and the cache never evicts. That makes the
# ~1000-schema threshold in the essay a code-reading claim rather than a
# measurement.
#
# These schemas have to differ in structure, not just in name: a backend that
# keys its cache on the compiled grammar would serve renamed clones from cache
# and the run would measure nothing. So field count, field names, field types
# and enum members all vary with the index. Generation is a pure function of
# the index, so a rerun offers the identical schema set.
#
# Keep the prompt shape fixed across the set. The variable under test is the
# number of distinct grammars in flight, and if the prompts also varied, output
# length would vary with them and confound the latency reading.

_CARD_NOUNS = ["droplet", "volume", "snapshot", "bucket", "loadbalancer",
               "database", "firewall", "registry", "cluster", "certificate"]


def make_cardinality_tasks(n):
    """n structurally distinct schemas, one task each, generated from the index.

    The task is deliberately a copy: the prompt supplies a concrete short value
    for every field and asks only that they be assembled into an object. The
    first version of this generator asked the model to invent plausible values
    for fields named `registry_00007_f4`, which produced a 37% truncation floor
    at every cardinality and made the arm unreadable. The mechanism is worth
    recording, because it is a real property of constrained decoding rather
    than a quirk of the prompt: an unbounded `string` or `integer` field admits
    an infinitely long legal continuation, since one more character inside an
    open string and one more digit inside a number are always grammatically
    valid. Greedy decoding at temperature 0, given a field name that carries no
    meaning and no value to copy, walks into a repetition loop, and the grammar
    has no way to break it. Two captured specimens: a 400-character digit run
    inside a string field, and `2026070000000...` for ~400 digits in a field
    declared `"type": "integer"`.

    So the fix is on the prompt side, not the schema side. Bounding the string
    needs `maxLength` and bounding that integer needs `maximum` or
    `multipleOf`, and the conformance probe found `multipleOf` silently
    under-enforced on xgrammar and `minimum` without a clean verdict, so the
    schema keywords that would prevent this are exactly the ones the backend
    does not reliably enforce. `maxLength` is added below as a second line of
    defence only -- it is not on xgrammar's preflight rejection list, so it
    cannot fail the request, and if it happens to be unenforced the supplied
    value still keeps the output short. Numeric fields stay unbounded on
    purpose: `maximum` would be relying on the same unverified enforcement, and
    truncation_rate in the results is the check. It should now read 0.0000, and
    a non-zero value at any rung means this failure is back.

    Semantic validity here is an exact-value match against what the prompt
    supplied, which makes it a real check rather than a restatement of the
    schema rung: a model that ignores the instruction fails it. It is still not
    comparable to any other task set's semantic rate, because the task is not
    comparable.
    """
    tasks = []
    for i in range(n):
        noun = _CARD_NOUNS[i % len(_CARD_NOUNS)]
        width = 3 + (i % 5)  # 3-7 fields, so the grammar's shape varies too
        props, required, expected, lines = {}, [], {}, []
        for j in range(width):
            key = f"{noun}_{i:05d}_f{j}"
            if j % 3 == 0:
                props[key] = {"type": "string", "maxLength": 32}
                val = f"{noun}-nyc3-{j}"
            elif j % 3 == 1:
                props[key] = {"type": "integer"}
                val = 100 + j
            else:
                members = [f"{noun}_state{i % 7}", f"{noun}_state{(i + 3) % 7}"]
                props[key] = {"type": "string", "enum": members}
                val = members[0]
            required.append(key)
            expected[key] = val
            lines.append(f"  {key} = {json.dumps(val)}")
        tasks.append({
            "name": f"card_{i:05d}",
            "prompt": (
                "Assemble these field values into a single JSON object, "
                "copying each value exactly as given. Reply with that object "
                "and nothing else.\n\n" + "\n".join(lines)
            ),
            "schema": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
            "semantic": (lambda exp: lambda o: all(
                o.get(k) == v for k, v in exp.items()))(dict(expected)),
        })
    return tasks


def resolve_tasks(cfg):
    """--schema-cardinality overrides --taskset: the two are answering
    different questions and mixing them would leave neither interpretable."""
    if getattr(cfg, "schema_cardinality", None):
        return make_cardinality_tasks(cfg.schema_cardinality)
    return TASKSETS[cfg.taskset]


# --------------------------------------------------------------------------
# Failure taxonomy. One category per request, by a rule you can read.
# --------------------------------------------------------------------------

CLASSIFY_RULE = (
    "ok: schema_valid and semantic_valid. "
    "truncation: finish_reason == 'length' or null content or zero completion "
    "tokens. "
    "constraint_boundary: request-time schema rejection (choiceless 4xx/5xx, "
    "subcategory request_rejected), or a parsed document that violates the "
    "declared schema while strict mode was requested (subcategory "
    "silent_underenforcement). "
    "extraction_parser: a complete, non-truncated response that failed to "
    "json.loads. "
    "semantic: schema_valid but the field-level rules failed. "
    "contention: transport error, timeout, or a 5xx that is not a schema "
    "rejection. "
    "On multi-turn tasks a failure before the final turn keeps its category "
    "and takes the subcategory intermediate_turn, and the error string carries "
    "the turn number that failed. "
    "Evaluated in that order; first match wins."
)


def classify(rec, mode):
    """Assign exactly one taxonomy category, plus a subcategory where the fix
    differs inside a category. Order matters: a truncated response that also
    fails to parse is a truncation, not a parser failure, because raising the
    budget is the fix and re-parsing is not."""
    err = rec.get("error") or ""
    if rec.get("schema_valid") and rec.get("semantic_valid"):
        return "ok", None
    if err.startswith("transport"):
        return "contention", "transport_or_timeout"
    if err.startswith("turn"):
        # A multi-turn task that died before the answer turn. The category is
        # still whatever went wrong; the subcategory says the agent never got
        # far enough to produce a final answer at all.
        if "null_content" in err or rec.get("finish_reason") == "length":
            return "truncation", "intermediate_turn"
        if "schema:" in err:
            return ("constraint_boundary" if mode == "strict"
                    else "schema_noncompliance"), "intermediate_turn"
        if "parse:" in err:
            return "extraction_parser", "intermediate_turn"
        if "semantic" in err:
            return "semantic", "intermediate_turn"
        return "other", "intermediate_turn"
    if rec.get("completion_tokens") == 0:
        return "truncation", "zero_token_completion"
    if err == "null_content":
        return "truncation", "null_content"
    if rec.get("finish_reason") == "length":
        return "truncation", "length"
    if err.startswith("schema_rejected"):
        # The honest failure: the backend refused the schema up front. Fix is
        # to change the schema or the backend, and retrying is pointless.
        return "constraint_boundary", "request_rejected"
    if rec.get("parse") and not rec.get("schema_valid"):
        # In strict mode the backend undertook to make this unreachable, so a
        # schema-invalid parsed document is an enforcement gap -- the silent
        # variant, where only the downstream validator catches it. Under
        # prompt_only nobody undertook anything, so it is model behaviour.
        if mode == "strict":
            return "constraint_boundary", "silent_underenforcement"
        return "schema_noncompliance", None
    if not rec.get("parse"):
        return "extraction_parser", None
    if rec.get("schema_valid") and not rec.get("semantic_valid"):
        return "semantic", None
    return "other", None


# --------------------------------------------------------------------------
# Server-side contention signals, sampled across the measured window only.
# Metric names verified against vLLM 0.27.1 /metrics.
# --------------------------------------------------------------------------

GAUGES = ["vllm:kv_cache_usage_perc", "vllm:num_requests_running",
          "vllm:num_requests_waiting"]
COUNTERS = ["vllm:num_preemptions_total"]
# TTFT is a histogram server-side. The _sum/_count deltas over the window give
# a mean TTFT that cross-checks the client-side --stream measurement.
HIST_PARTS = ["vllm:time_to_first_token_seconds_sum",
              "vllm:time_to_first_token_seconds_count"]
WAITING_BY_REASON = "vllm:num_requests_waiting_by_reason"


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        try:
            val = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        if name in GAUGES or name in COUNTERS or name in HIST_PARTS:
            out[name] = out.get(name, 0.0) + val if name in HIST_PARTS else val
        elif name == WAITING_BY_REASON:
            if 'reason="capacity"' in line:
                out["waiting_capacity"] = val
            elif 'reason="deferred"' in line:
                out["waiting_deferred"] = val
    return out


async def sample_metrics(url, interval, stop_event, sink):
    """Poll /metrics until stop_event is set. Failures are recorded rather than
    raised: a scrape that times out under load is itself a signal, and it must
    not take the ramp down with it."""
    async with httpx.AsyncClient() as client:
        while not stop_event.is_set():
            t = time.time()
            try:
                r = await client.get(url, timeout=5.0)
                sink.append({"ts": t, **parse_metrics(r.text)})
            except Exception as e:
                sink.append({"ts": t, "error": f"{type(e).__name__}: {e}"})
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass


def summarize_metrics(samples):
    """Deltas for counters, distributions for gauges. The preemption number
    that matters is the delta across the measured window, not the counter's
    process-lifetime value."""
    good = [s for s in samples if "error" not in s]
    if not good:
        return {"samples": len(samples), "scrape_errors": len(samples)}
    out = {"samples": len(good), "scrape_errors": len(samples) - len(good)}

    def series(k):
        return [s[k] for s in good if k in s]

    for k in COUNTERS:
        v = series(k)
        if v:
            out[k + "_delta"] = round(v[-1] - v[0], 3)
    for k in GAUGES + ["waiting_capacity", "waiting_deferred"]:
        v = series(k)
        if v:
            out[k + "_mean"] = round(statistics.mean(v), 4)
            out[k + "_max"] = round(max(v), 4)
    s = series("vllm:time_to_first_token_seconds_sum")
    c = series("vllm:time_to_first_token_seconds_count")
    if len(s) >= 2 and len(c) >= 2 and (c[-1] - c[0]) > 0:
        out["server_ttft_mean_s"] = round((s[-1] - s[0]) / (c[-1] - c[0]), 4)
        out["server_requests_in_window"] = int(c[-1] - c[0])
    return out


# --------------------------------------------------------------------------
# Request path
# --------------------------------------------------------------------------

def build_body(messages, schema, name, cfg):
    body = {
        "model": cfg.model,
        "messages": [dict(m) for m in messages],
        "temperature": cfg.temperature,
        "max_completion_tokens": cfg.max_tokens,
        "stream": False,
    }
    if cfg.seed != -1:
        body["seed"] = cfg.seed
    if cfg.mode == "strict":
        # Current vLLM / OpenAI surface. The guided_* fields were removed in
        # vLLM v0.12.0 -- do not reintroduce them.
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True},
        }
    elif cfg.mode == "prompt_only":
        body["messages"][-1]["content"] += (
            "\n\nThe object must match this JSON Schema exactly:\n"
            + json.dumps(schema)
        )
    else:
        raise ValueError(cfg.mode)
    return body


async def read_streaming(client, body, rec, t0):
    """SSE read path. Timestamps the first content chunk for TTFT and
    reassembles the full text so the validation ladder sees the same input it
    would have seen non-streamed."""
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    parts, finish_reason, usage = [], None, {}
    async with client.stream(
        "POST", f"{BASE_URL}/chat/completions", json=body,
        headers={"Authorization": f"Bearer {API_KEY}"}, timeout=600.0,
    ) as r:
        rec["http_status"] = r.status_code
        if r.status_code >= 400:
            await r.aread()
            rec["error"] = f"schema_rejected: HTTP {r.status_code}"
            rec["finish_reason"] = "rejected"
            return None
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except Exception:
                continue
            choices = chunk.get("choices") or []
            if choices:
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    if rec["ttft_s"] is None:
                        rec["ttft_s"] = time.perf_counter() - t0
                    parts.append(piece)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]
    rec["finish_reason"] = finish_reason
    rec["completion_tokens"] = usage.get("completion_tokens")
    rec["prompt_tokens"] = usage.get("prompt_tokens")
    return "".join(parts) if parts else None


async def read_blocking(client, body, rec):
    r = await client.post(
        f"{BASE_URL}/chat/completions", json=body,
        headers={"Authorization": f"Bearer {API_KEY}"}, timeout=600.0,
    )
    rec["http_status"] = r.status_code
    try:
        data = r.json()
    except Exception:
        rec["error"] = f"schema_rejected: HTTP {r.status_code} (unparseable body)"
        rec["finish_reason"] = "rejected"
        return None
    # A rejected schema comes back with no `choices` at all. Reading
    # data["choices"][0] here is what raised KeyError on the edge task set.
    if not data.get("choices"):
        detail = ""
        if isinstance(data.get("error"), dict):
            detail = str(data["error"].get("message", ""))[:200]
        rec["error"] = f"schema_rejected: HTTP {r.status_code} {detail}".strip()
        rec["finish_reason"] = "rejected"
        return None
    choice = data["choices"][0]
    rec["finish_reason"] = choice.get("finish_reason")
    rec["completion_tokens"] = data.get("usage", {}).get("completion_tokens")
    rec["prompt_tokens"] = data.get("usage", {}).get("prompt_tokens")
    return choice["message"].get("content")


async def run_turns(client, task, cfg, rec, t0):
    """Walk a multi-turn task, splicing each turn's synthetic tool result into
    the transcript. Returns the final turn's content, or None if an earlier
    turn failed -- in which case rec carries a `turn<N>:`-prefixed error, so a
    tool-call-turn failure is distinguishable from an answer-turn failure."""
    messages, tokens_in, tokens_out = [], 0, 0
    content = None
    for i, turn in enumerate(task["turns"], start=1):
        messages.append({"role": "user", "content": turn["prompt"]})
        body = build_body(messages, turn["schema"], f"{task['name']}_t{i}", cfg)
        # TTFT is recorded for the final turn only; an earlier turn's first
        # chunk is not the user-visible first token.
        sub = {"error": None, "finish_reason": None, "ttft_s": None,
               "completion_tokens": None, "prompt_tokens": None}
        content = (await read_streaming(client, body, sub, t0)) if cfg.stream \
            else (await read_blocking(client, body, sub))
        tokens_in += sub.get("prompt_tokens") or 0
        tokens_out += sub.get("completion_tokens") or 0
        rec["finish_reason"] = sub["finish_reason"]
        rec["http_status"] = sub.get("http_status")
        rec["ttft_s"] = sub["ttft_s"]
        rec["turns_completed"] = i
        last = i == len(task["turns"])
        if content is None:
            rec["error"] = sub["error"] or "null_content"
            if not last:
                rec["error"] = f"turn{i}: {rec['error']}"
            break
        if not last:
            # Validate the intermediate turn so a broken tool call is visible
            # rather than silently poisoning the next turn's context.
            try:
                obj = json.loads(content)
                errs = list(Draft7Validator(turn["schema"]).iter_errors(obj))
            except Exception as e:
                rec["error"] = f"turn{i}: parse: {type(e).__name__}"
                rec["specimen"] = content[:1200]
                content = None
                break
            if errs:
                rec["error"] = f"turn{i}: schema: {errs[0].validator}"
                rec["specimen"] = content[:1200]
                content = None
                break
            # Intermediate turns may carry their own semantic rule. On a deep
            # task this is what makes a failure attributable: without it, a
            # wrong subtotal in turn 3 only surfaces as a wrong total in turn 5
            # and every step looks equally suspect.
            if turn.get("semantic"):
                try:
                    ok = bool(turn["semantic"](obj))
                except Exception:
                    ok = False
                if not ok:
                    rec["error"] = f"turn{i}: semantic"
                    rec["specimen"] = content[:1200]
                    content = None
                    break
            messages.append({"role": "assistant", "content": content})
            # A turn that made a tool call gets its tool result back. A turn
            # that reasoned instead gets a plain acknowledgement, because
            # labelling its own output "Tool result:" would teach the model
            # that its arithmetic came from somewhere authoritative.
            if turn.get("tool_result") is not None:
                follow = f"Tool result: {turn['tool_result']}"
            else:
                follow = turn.get("next_user", "Acknowledged.")
            messages.append({"role": "user", "content": follow})
    rec["prompt_tokens"] = tokens_in or None
    rec["completion_tokens"] = tokens_out or None
    return content


async def one_request(client, task, cfg, validator):
    t0 = time.perf_counter()
    rec = {"task": task["name"], "error": None, "finish_reason": None,
           "parse": False, "schema_valid": False, "semantic_valid": False,
           "completion_tokens": None, "prompt_tokens": None, "ttft_s": None}
    try:
        if task.get("turns"):
            content = await run_turns(client, task, cfg, rec, t0)
        else:
            body = build_body([{"role": "user", "content": task["prompt"]}],
                              task["schema"], task["name"], cfg)
            content = (await read_streaming(client, body, rec, t0)) if cfg.stream \
                else (await read_blocking(client, body, rec))
        if content is None and rec["error"] is None:
            # Null content with no rejection: the budget went somewhere the
            # response body does not show. Reasoning tokens are the usual
            # answer. This is a truncation, not a schema failure.
            rec["error"] = "null_content"
        elif content is not None:
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
            if not (rec["schema_valid"] and rec["semantic_valid"]):
                rec["specimen"] = content[:1200]
    except Exception as e:
        rec["error"] = f"transport: {type(e).__name__}: {e}"
    rec["e2e_s"] = time.perf_counter() - t0
    rec["category"], rec["subcategory"] = classify(rec, cfg.mode)
    return rec


# --------------------------------------------------------------------------
# Load generation: fixed-count bursts, or duration-based sustained load.
# --------------------------------------------------------------------------

async def run_level(cfg, concurrency):
    tasks = resolve_tasks(cfg)
    validators = {t["name"]: Draft7Validator(t["schema"]) for t in tasks}
    # Connection reuse is a confound at high concurrency, not just a tuning
    # detail. When queue waits exceed the server's keepalive timeout, a pooled
    # connection can be closed server-side between the client checking it out
    # and writing to it, which surfaces as RemoteProtocolError rather than as
    # anything to do with the schema. --no-keepalive removes the pool so that
    # failure mode cannot occur, which is how you tell it apart from a server
    # genuinely dropping requests under load.
    limits = httpx.Limits(
        max_connections=concurrency + 10,
        max_keepalive_connections=0 if cfg.no_keepalive else concurrency + 10)
    counter = {"i": 0}

    def next_task():
        t = tasks[counter["i"] % len(tasks)]
        counter["i"] += 1
        return t

    metric_samples = []
    stop_metrics = asyncio.Event()

    async with httpx.AsyncClient(limits=limits) as client:
        async def issue():
            task = next_task()
            return await one_request(client, task, cfg, validators[task["name"]])

        # Warm-up, always discarded. The first structured-output request the
        # process ever sees also elects the grammar backend for the lifetime of
        # that process, and compiles each grammar for the first time.
        #
        # The cardinality arm needs the opposite of a thorough warm-up. Warming
        # every schema would compile the whole set off the measured window and
        # serve the entire run from cache, which is the condition this arm
        # exists to break. So warm just enough to elect the backend, and let the
        # measured window meet its grammars cold. The rotation counter carries
        # over, so the schemas burned here are not measured twice.
        if cfg.schema_cardinality:
            sem = asyncio.Semaphore(concurrency)

            async def warm_min():
                async with sem:
                    return await issue()

            await asyncio.gather(*(warm_min() for _ in range(min(4, concurrency))))
        elif cfg.duration_s:
            warm_deadline = time.perf_counter() + cfg.warmup_s

            async def warm_worker():
                while time.perf_counter() < warm_deadline:
                    await issue()

            await asyncio.gather(*(warm_worker() for _ in range(concurrency)))
        else:
            sem = asyncio.Semaphore(concurrency)

            async def warm_one():
                async with sem:
                    return await issue()

            await asyncio.gather(*(warm_one()
                                   for _ in range(min(3 * len(tasks),
                                                      cfg.requests_per_level))))

        # Measured window. Metrics sampling starts here and stops here, so the
        # contention signals describe the same interval as the validity rates.
        sampler = None
        if cfg.metrics_url:
            sampler = asyncio.create_task(sample_metrics(
                cfg.metrics_url, cfg.metrics_interval, stop_metrics, metric_samples))

        t0 = time.perf_counter()
        if cfg.duration_s:
            # Closed loop: `concurrency` persistent workers, back-to-back
            # requests, for a fixed wall-clock window. Offered load is
            # sustained rather than delivered in one wave, so the server
            # reaches steady state and the KV cache stays populated.
            deadline = t0 + cfg.duration_s
            out = []

            async def worker():
                while time.perf_counter() < deadline:
                    out.append(await issue())

            await asyncio.gather(*(worker() for _ in range(concurrency)))
        else:
            sem = asyncio.Semaphore(concurrency)

            async def guarded():
                async with sem:
                    return await issue()

            out = list(await asyncio.gather(
                *(guarded() for _ in range(cfg.requests_per_level))))
        wall = time.perf_counter() - t0

        if sampler:
            stop_metrics.set()
            await sampler
            # One final scrape after the window closes. Counter deltas are
            # taken between the first and last sample, so without this the
            # requests completing in the final scrape interval are missing from
            # the histogram delta and server_requests_in_window undercounts
            # against the client's own request total.
            try:
                r = await client.get(cfg.metrics_url, timeout=5.0)
                metric_samples.append({"ts": time.time(), **parse_metrics(r.text)})
            except Exception as e:
                metric_samples.append({"ts": time.time(),
                                       "error": f"{type(e).__name__}: {e}"})

    n = len(out)
    if not n:
        return {"concurrency": concurrency, "requests": 0,
                "note": "no requests completed in the measured window"}
    lat = sorted(r["e2e_s"] for r in out)

    def pct(vals, p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]

    valid = sum(1 for r in out if r["schema_valid"])
    usable = sum(1 for r in out if r["schema_valid"] and r["semantic_valid"])
    in_tok = sum(r["prompt_tokens"] or 0 for r in out)
    out_tok = sum(r["completion_tokens"] or 0 for r in out)
    spend = (in_tok * cfg.price_in + out_tok * cfg.price_out) / 1e6

    res = {
        "concurrency": concurrency,
        "requests": n,
        "wall_s": round(wall, 2),
        "throughput_rps": round(n / wall, 2) if wall else None,
        "parse_rate": sum(r["parse"] for r in out) / n,
        "schema_valid_rate": valid / n,
        "semantic_valid_rate": sum(r["semantic_valid"] for r in out) / n,
        "usable_rate": usable / n,
        "truncation_rate": sum(1 for r in out if r["finish_reason"] == "length") / n,
        "null_content_rate": sum(1 for r in out if r["error"] == "null_content") / n,
        "zero_token_rate": sum(1 for r in out if r["completion_tokens"] == 0) / n,
        # Categories, not just booleans. This is the field to alert on.
        "categories": dict(Counter(r["category"] for r in out).most_common()),
        "subcategories": dict(Counter(
            f"{r['category']}/{r['subcategory']}" for r in out
            if r["subcategory"]).most_common()),
        # Per-task breakdown is what showed failure relocation, so keep it --
        # but the cardinality arm has hundreds of tasks and a per-task map
        # there would be all of the output and none of the signal.
        "categories_by_task": {
            t["name"]: dict(Counter(r["category"] for r in out
                                    if r["task"] == t["name"]).most_common())
            for t in tasks
        } if len(tasks) <= 8 else None,
        "finish_reasons": dict(Counter(r["finish_reason"] for r in out)),
        "errors": dict(Counter(r["error"] for r in out if r["error"]).most_common(8)),
        "e2e_p50": round(pct(lat, 0.50), 3),
        "e2e_p95": round(pct(lat, 0.95), 3),
        "e2e_p99": round(pct(lat, 0.99), 3),
        "attempts_per_schema_valid": round(n / valid, 4) if valid else None,
        "attempts_per_usable": round(n / usable, 4) if usable else None,
        "mean_completion_tokens": round(statistics.mean(
            [r["completion_tokens"] for r in out if r["completion_tokens"]] or [0]), 1),
        # Measured, not modelled: actual tokens billed over actual usable
        # outputs. Compare against the modelled table's assumption of
        # independent retries -- a deterministic semantic failure never
        # converges, so cost per usable output is unbounded for that task class
        # and this figure is a lower bound wherever that is what failed.
        "cost_total_usd": round(spend, 6),
        "cost_per_attempt_usd": round(spend / n, 8),
        "cost_per_schema_valid_usd": round(spend / valid, 8) if valid else None,
        "cost_per_usable_usd": round(spend / usable, 8) if usable else None,
        "prompt_tokens_total": in_tok,
        "completion_tokens_total": out_tok,
        # Offered vs. actually reached. With a rotation longer than the request
        # count, the tail of the schema set is never requested and the arm did
        # not test the cardinality it was configured for.
        "distinct_schemas_offered": len(tasks),
        "distinct_schemas_used": len({r["task"] for r in out}),
    }

    multi = [r for r in out if "turns_completed" in r]
    if multi:
        # A multi-turn agent that fails at the tool-call turn never reaches the
        # answer turn. Reported separately because raising the budget, fixing
        # the tool schema, and fixing the answer schema are three fixes.
        res["multi_turn"] = {
            "requests": len(multi),
            "mean_turns_completed": round(
                statistics.mean(r["turns_completed"] for r in multi), 3),
            "intermediate_turn_failure_rate": round(sum(
                1 for r in multi
                if (r.get("error") or "").startswith("turn")) / len(multi), 4),
            # On a five-step task the aggregate rate is not the finding. Which
            # turn the conversation died on is: a task that always survives to
            # turn 5 and fails there is losing context, and one that dies at
            # turn 3 is failing arithmetic.
            "turns_completed_hist": dict(sorted(
                Counter(r["turns_completed"] for r in multi).items())),
            "failures_by_turn": dict(sorted(Counter(
                (r["error"] or "").split(":")[0] for r in multi
                if (r.get("error") or "").startswith("turn")).items())),
        }

    ttfts = sorted(r["ttft_s"] for r in out if r["ttft_s"] is not None)
    if ttfts:
        res["ttft_p50"] = round(pct(ttfts, 0.50), 4)
        res["ttft_p95"] = round(pct(ttfts, 0.95), 4)
        res["ttft_p99"] = round(pct(ttfts, 0.99), 4)
        res["ttft_n"] = len(ttfts)
    if cfg.metrics_url:
        res["server_metrics"] = summarize_metrics(metric_samples)
    if cfg.dump_records:
        res["records"] = out
    else:
        # One specimen per category/subcategory, not per category: a silent
        # under-enforcement and a request-time rejection are both
        # constraint_boundary failures and take opposite fixes, so keeping only
        # the first one loses the distinction the taxonomy exists to draw.
        res["specimens_by_category"] = {}
        for r in out:
            if r["category"] == "ok":
                continue
            key = f"{r['category']}/{r['subcategory']}" if r["subcategory"] \
                else r["category"]
            if key not in res["specimens_by_category"]:
                res["specimens_by_category"][key] = {
                    "task": r["task"], "finish_reason": r["finish_reason"],
                    "http_status": r.get("http_status"), "error": r["error"],
                    "raw": r.get("specimen"),
                }
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["strict", "prompt_only"], default="strict")
    ap.add_argument("--levels", default="1,10,50,100")
    ap.add_argument("--taskset", choices=sorted(TASKSETS), default="default",
                    help="default = the three templates the published ramp used; "
                         "four adds enum_triage and changes every denominator; "
                         "edge = xgrammar's enforcement boundary; "
                         "multiturn/agent_mix = two-turn agent; "
                         "deep_multiturn/agent_mix_deep = five-step agent.")
    ap.add_argument("--schema-cardinality", type=int, default=None,
                    help="Synthesize this many structurally distinct schemas "
                         "and rotate one per request, to force grammar-cache "
                         "eviction and put compile cost on the request path. "
                         "Overrides --taskset. Offer >=2N requests per level, "
                         "and check distinct_schemas_used in the output.")
    ap.add_argument("--requests-per-level", type=int, default=200,
                    help="Fixed-count mode. Ignored when --duration-s is set.")
    ap.add_argument("--duration-s", type=float, default=None,
                    help="Sustained mode: run a closed loop of `concurrency` "
                         "workers for this many seconds per level instead of a "
                         "fixed request count. Use >=120 to reach steady state.")
    ap.add_argument("--warmup-s", type=float, default=20.0,
                    help="Discarded warm-up window in sustained mode.")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Set deliberately low (e.g. 128) for the tight-budget arm.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20260813,
                    help="-1 omits the seed, so each request is an independent "
                         "draw and the reported rates are proportions rather "
                         "than repeated deterministic completions.")
    ap.add_argument("--stream", action="store_true",
                    help="Stream and record client-side TTFT. Run as a separate "
                         "latency pass: streaming changes the reassembly path.")
    ap.add_argument("--no-keepalive", action="store_true",
                    help="Disable HTTP connection reuse. Use to test whether "
                         "high-concurrency RemoteProtocolError failures are a "
                         "keepalive race rather than the server dropping "
                         "requests.")
    ap.add_argument("--metrics-url", default=None,
                    help="e.g. http://HOST:8000/metrics -- sampled across the "
                         "measured window only.")
    ap.add_argument("--metrics-interval", type=float, default=2.0)
    ap.add_argument("--price-in", type=float, default=0.65,
                    help="USD per 1M input tokens, for measured cost-per-valid.")
    ap.add_argument("--price-out", type=float, default=0.65)
    ap.add_argument("--dump-records", action="store_true",
                    help="Retain every per-request record in the output file. "
                         "The published ramp did not, which is why its tables "
                         "can be reproduced but not replayed.")
    ap.add_argument("--out", default="validity_ramp_v4_results.json")
    cfg = ap.parse_args()

    if cfg.seed == -1 and cfg.temperature == 0.0:
        print("note: --seed -1 at temperature 0 still gives near-identical "
              "completions; raise --temperature for genuinely independent draws",
              flush=True)
    if cfg.schema_cardinality:
        print(f"note: --schema-cardinality {cfg.schema_cardinality} overrides "
              f"--taskset; semantic validity in this arm is field presence "
              f"only and is not comparable to other task sets", flush=True)
        if not cfg.duration_s and cfg.requests_per_level < 2 * cfg.schema_cardinality:
            print(f"warning: {cfg.requests_per_level} requests per level will "
                  f"reach at most that many of "
                  f"{cfg.schema_cardinality} schemas -- raise "
                  f"--requests-per-level to >={2 * cfg.schema_cardinality} or "
                  f"use --duration-s, and check distinct_schemas_used",
                  flush=True)

    results = []
    for c in [int(x) for x in cfg.levels.split(",")]:
        r = await run_level(cfg, c)
        r.update({"mode": cfg.mode, "model": cfg.model,
                  "taskset": ("synthetic_cardinality" if cfg.schema_cardinality
                              else cfg.taskset),
                  "schema_cardinality": cfg.schema_cardinality,
                  "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
                  "seed": cfg.seed, "stream": cfg.stream,
                  "load_mode": "sustained" if cfg.duration_s else "fixed_count",
                  "duration_s": cfg.duration_s, "no_keepalive": cfg.no_keepalive,
                  "classify_rule": CLASSIFY_RULE})
        results.append(r)
        printable = {k: v for k, v in r.items()
                     if k not in ("records", "specimens_by_category", "classify_rule")}
        print(json.dumps(printable, indent=2), flush=True)

    with open(cfg.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    asyncio.run(main())
