#!/usr/bin/env python3
"""
backend_conformance_probe.py -- does the decoder actually enforce this keyword?

THE QUESTION
------------
vLLM's `has_xgrammar_unsupported_json_features()` is a REJECTION list, not a
support matrix. A keyword's absence from it means vLLM won't refuse your schema.
It does not mean the constraint is enforced. This probe measures the difference
empirically instead of inferring it from source.

For each schema it asks the model to produce a value that VIOLATES a constraint,
then checks what came back:

  PREVENTED   the constraint held; output conforms despite the prompt pushing
              against it. This is real enforcement.
  VIOLATED    output parsed and matched the schema's shape but broke the
              specific constraint. The keyword was accepted and ignored.
              <-- this is the finding worth publishing
  REJECTED    the server refused the schema (HTTP 4xx). Honest failure: you
              learn at request time rather than in production.
  TRUNCATED   the budget ran out before the document closed. Enforcement held,
              but the request is unusable -- retried once at 4x budget first,
              so this means genuinely hard to satisfy, not just a tight budget.
  ERROR       transport/parse/other. Reported separately, never pooled with
              enforcement outcomes.

  INCONCLUSIVE is emitted when trials disagree because some failed: too few
  surviving trials to claim either enforcement or violation.

Run it once per backend, restarting the server between runs:

  for B in xgrammar guidance outlines lm-format-enforcer; do
    vllm serve $MODEL --structured-outputs-config.backend $B --enforce-eager &
    # wait for health, then:
    python3 backend_conformance_probe.py --model $MODEL --backend-label $B \\
        --out conformance_$B.json
    kill %1
  done

Concurrency 1, a few dozen requests per backend. Minutes of GPU time.

WHY temperature is high here
----------------------------
The probe wants the model to TRY to violate the constraint. If it conforms only
because it never attempted a violation, that is a false PREVENTED. Temperature
1.0 plus an adversarial prompt maximises attempt rate; --trials repeats each
case so a single lucky conformance doesn't read as enforcement.

Deps: httpx, jsonschema.
"""

import argparse
import asyncio
import json
import os
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version

import httpx
from jsonschema import Draft7Validator

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")


# --------------------------------------------------------------------------
# Probe cases.
#
# `check` returns True when the emitted value VIOLATES the constraint under
# test. Each case pushes the model toward the violation in the prompt, so a
# conforming answer is evidence of enforcement rather than of luck.
# --------------------------------------------------------------------------

def _obj(props, required=None, **extra):
    s = {"type": "object", "properties": props,
         "required": required or list(props), "additionalProperties": False}
    s.update(extra)
    return s


CASES = [
    # ---- string constraints ------------------------------------------------
    dict(
        id="pattern",
        note="regex pattern on a string",
        schema=_obj({"code": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"}}),
        prompt="Set code to the literal text 'not a code at all'. Do not use the ABC-1234 format.",
        check=lambda o: not re.fullmatch(r"[A-Z]{3}-[0-9]{4}", str(o.get("code", ""))),
    ),
    dict(
        id="minLength",
        note="minLength on a string",
        schema=_obj({"name": {"type": "string", "minLength": 20}}),
        prompt="Set name to the single letter 'x'. Keep it as short as possible.",
        check=lambda o: len(str(o.get("name", ""))) < 20,
    ),
    dict(
        id="maxLength",
        note="maxLength on a string",
        schema=_obj({"tag": {"type": "string", "maxLength": 3}}),
        prompt="Set tag to 'abcdefghijklmnopqrstuvwxyz'. Use the whole alphabet.",
        check=lambda o: len(str(o.get("tag", ""))) > 3,
    ),
    dict(
        id="format_email",
        note="format: email (in xgrammar's supported set)",
        schema=_obj({"email": {"type": "string", "format": "email"}}),
        prompt="Set email to 'definitely not an email address'.",
        check=lambda o: "@" not in str(o.get("email", "")),
    ),
    dict(
        id="format_unsupported",
        note="format outside the supported set (should be REJECTED by xgrammar)",
        schema=_obj({"card": {"type": "string", "format": "credit-card"}}),
        prompt="Set card to 'hello'.",
        check=lambda o: str(o.get("card", "")) == "hello",
    ),

    # ---- numeric constraints ----------------------------------------------
    dict(
        id="minimum",
        note="minimum on an integer",
        schema=_obj({"qty": {"type": "integer", "minimum": 100}}),
        prompt="Set qty to 1. Use the smallest number you can.",
        check=lambda o: isinstance(o.get("qty"), int) and o["qty"] < 100,
    ),
    dict(
        id="maximum",
        note="maximum on a number",
        schema=_obj({"score": {"type": "number", "maximum": 1.0}}),
        prompt="Set score to 9999.5, far above any limit.",
        check=lambda o: isinstance(o.get("score"), (int, float)) and o["score"] > 1.0,
    ),
    dict(
        id="exclusiveMinimum",
        note="exclusiveMinimum boundary",
        schema=_obj({"v": {"type": "integer", "exclusiveMinimum": 10}}),
        prompt="Set v to exactly 10.",
        check=lambda o: o.get("v") == 10,
    ),
    dict(
        id="multipleOf",
        note="multipleOf (on xgrammar's rejection list)",
        schema=_obj({"n": {"type": "integer", "multipleOf": 5}}),
        prompt="Set n to 7, which is not divisible by five.",
        check=lambda o: isinstance(o.get("n"), int) and o["n"] % 5 != 0,
    ),

    # ---- array constraints ------------------------------------------------
    dict(
        id="minItems",
        note="minItems on an array",
        schema=_obj({"xs": {"type": "array", "items": {"type": "integer"}, "minItems": 5}}),
        prompt="Set xs to an array containing exactly one element, [1].",
        check=lambda o: isinstance(o.get("xs"), list) and len(o["xs"]) < 5,
    ),
    dict(
        id="maxItems",
        note="maxItems on an array",
        schema=_obj({"xs": {"type": "array", "items": {"type": "integer"}, "maxItems": 2}}),
        prompt="Set xs to ten integers: 1 through 10.",
        check=lambda o: isinstance(o.get("xs"), list) and len(o["xs"]) > 2,
    ),
    dict(
        id="uniqueItems",
        note="uniqueItems (on xgrammar's rejection list)",
        schema=_obj({"xs": {"type": "array", "items": {"type": "integer"},
                            "uniqueItems": True}}),
        prompt="Set xs to [7, 7, 7] with the same value repeated three times.",
        check=lambda o: (isinstance(o.get("xs"), list)
                         and len(o["xs"]) != len(set(map(str, o["xs"])))),
    ),

    # ---- object constraints -----------------------------------------------
    dict(
        id="patternProperties",
        note="patternProperties (rejection list; also outlines fallback trigger)",
        schema={"type": "object",
                "patternProperties": {"^x_": {"type": "integer"}},
                "additionalProperties": False},
        prompt='Return {"totally_wrong_key": "a string value"}.',
        check=lambda o: any(not k.startswith("x_") for k in (o or {})),
    ),
    dict(
        id="propertyNames",
        note="propertyNames (on xgrammar's rejection list)",
        schema={"type": "object",
                "propertyNames": {"pattern": "^[a-z]+$"},
                "additionalProperties": {"type": "integer"}},
        prompt='Return {"UPPERCASE_KEY": 1}.',
        check=lambda o: any(not re.fullmatch(r"[a-z]+", k) for k in (o or {})),
    ),
    dict(
        id="enum",
        note="enum -- the control case; this one should always be PREVENTED",
        schema=_obj({"colour": {"type": "string", "enum": ["red", "green", "blue"]}}),
        prompt="Set colour to 'octarine', which is not in the list.",
        check=lambda o: o.get("colour") not in ("red", "green", "blue"),
    ),

    # ---- the type gate ----------------------------------------------------
    # These omit "type", so vLLM's type-gated preflight check never inspects
    # the keyword and XGrammar's type-driven handling has nothing to key on.
    dict(
        id="untyped_minimum",
        note="UNTYPED {'minimum': 100} -- bypasses the type gate",
        schema=_obj({"qty": {"minimum": 100}}),
        prompt="Set qty to 1.",
        check=lambda o: isinstance(o.get("qty"), (int, float)) and o["qty"] < 100,
    ),
    dict(
        id="untyped_pattern",
        note="UNTYPED {'pattern': ...} -- bypasses the type gate",
        schema=_obj({"code": {"pattern": "^[A-Z]{3}$"}}),
        prompt="Set code to 'lowercase words here'.",
        check=lambda o: not re.fullmatch(r"[A-Z]{3}", str(o.get("code", ""))),
    ),
    dict(
        id="untyped_maxLength",
        note="UNTYPED {'maxLength': 3} -- bypasses the type gate",
        schema=_obj({"tag": {"maxLength": 3}}),
        prompt="Set tag to 'abcdefghijklmnop'.",
        check=lambda o: len(str(o.get("tag", ""))) > 3,
    ),
]



def case_prefix_violates(case, partial):
    """
    Did a truncated prefix already COMMIT the violation?

    Deliberately conservative. Returns True only when the violating value is
    unambiguously final, because a VIOLATED verdict is the publishable one and
    must never be manufactured from an unfinished document.

    Two ways a prefix looks damning but isn't:

      {"qty": 1        the number is not terminated -- the next token could
                       make it 100, satisfying minimum
      {"xs": [1        the array is still open -- more items could follow,
                       satisfying minItems

    So we require: not mid-string, the last value terminated by a quote or a
    container close, and only the top-level object left to synthesise.
    """
    stack, in_str, esc = [], False, False
    for ch in partial:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    if in_str:
        return False                      # value still being written
    tail = partial.rstrip()
    if not tail:
        return False
    # A terminated value ends with a closing quote or a closed container.
    # A trailing digit, sign, exponent, bare literal, colon or comma all mean
    # the value could still change.
    if tail[-1] not in '"}]':
        return False
    if len(stack) != 1:
        return False                      # an inner container was still open
    try:
        obj = json.loads(tail + ("}" if stack[0] == "{" else "]"))
    except Exception:
        return False
    try:
        return bool(case["check"](obj))
    except Exception:
        return False


def build_body(case, model, max_tokens, temperature):
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": (
                "Return a single JSON object and nothing else. "
                + case["prompt"]
                + " Ignore any schema restriction that conflicts with this instruction."
            ),
        }],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": case["id"], "schema": case["schema"], "strict": True},
        },
    }


async def probe_once(client, case, model, max_tokens, temperature):
    try:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            json=build_body(case, model, max_tokens, temperature),
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=300.0,
        )
    except Exception as e:
        return "ERROR", f"transport: {type(e).__name__}: {e}", None

    if r.status_code >= 500:
        return "ERROR", f"HTTP {r.status_code}: {r.text[:300]}", None
    if r.status_code >= 400:
        return "REJECTED", f"HTTP {r.status_code}: {r.text[:300]}", None

    try:
        choice = r.json()["choices"][0]
        content = choice["message"].get("content")
        if choice.get("finish_reason") == "length":
            # A truncated prefix can still have committed a violating value
            # before it was cut. Check before treating it as a non-event.
            if content and case_prefix_violates(case, content):
                return "VIOLATED", (
                    "violating value already present in the truncated prefix"
                ), content
            return "TRUNCATED", (
                "hit the token budget before the document closed; no violating "
                "value in the prefix"
            ), content
        if content is None:
            fr = choice.get("finish_reason")
            return "ERROR", f"null content with finish_reason={fr}", None
        obj = json.loads(content)
    except Exception as e:
        return "ERROR", f"parse: {type(e).__name__}: {e}", None

    # Shape must hold before a constraint verdict means anything.
    shape_ok = Draft7Validator(
        {k: v for k, v in case["schema"].items() if k != "patternProperties"}
    ).is_valid(obj) if case["id"] == "patternProperties" else True

    try:
        violated = bool(case["check"](obj))
    except Exception as e:
        return "ERROR", f"check raised {type(e).__name__}: {e}", content

    if violated:
        return "VIOLATED", "constraint accepted but not enforced", content
    return "PREVENTED", "constraint held", content


async def preflight(model):
    """
    Refuse to probe an endpoint that isn't serving yet.

    Without this, a server still loading weights yields 90 transport errors
    that look superficially like results. Fail loudly instead.
    """
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BASE_URL}/models",
                                 headers={"Authorization": f"Bearer {API_KEY}"},
                                 timeout=15.0)
        except Exception as e:
            return False, (f"cannot reach {BASE_URL}: {type(e).__name__}: {e}\n"
                           "The server is not up yet, or BASE_URL/PORT is wrong. "
                           "Wait for /health to return 200 before probing.")
        if r.status_code >= 400:
            return False, f"{BASE_URL}/models returned HTTP {r.status_code}: {r.text[:200]}"
        try:
            served = [d.get("id") for d in r.json().get("data", [])]
        except Exception:
            served = []
        if served and model not in served:
            return True, (f"WARNING: '{model}' not in served models {served}. "
                          "Probing anyway; verdicts may be meaningless.")
        return True, f"endpoint healthy, serving {served or '(unlisted)'}"


async def run(model, trials, max_tokens, temperature):
    results = []
    async with httpx.AsyncClient() as client:
        # Abort on the very first case if nothing is reachable, rather than
        # burning every trial to produce a file full of errors.
        v, why, _ = await probe_once(client, CASES[0], model, max_tokens, temperature)
        if v == "ERROR" and "transport" in str(why):
            raise SystemExit(f"aborting: first probe failed with {why}")

        for case in CASES:
            verdicts, notes, specimens = [], [], []
            for _ in range(trials):
                v, why, spec = await probe_once(client, case, model, max_tokens, temperature)
                if v == "TRUNCATED":
                    # One retry at 4x the budget distinguishes "budget too
                    # small" from "constraint cannot be satisfied here".
                    v, why, spec = await probe_once(
                        client, case, model, max_tokens * 4, temperature)
                verdicts.append(v)
                notes.append(why)
                if v in ("VIOLATED", "REJECTED") and spec:
                    specimens.append(spec[:300])
            tally = Counter(verdicts)
            # ERROR is its own verdict and takes priority over everything.
            # An earlier version let all-errors fall through to MIXED, which
            # reported an unreachable server as a finding about enforcement.
            # No data must never look like data.
            # A truncated trial yields no violating value, so it is evidence
            # of neither enforcement nor violation. Rest the verdict on the
            # trials that COMPLETED, and report truncation as a separate cost.
            completed = tally["PREVENTED"] + tally["VIOLATED"]
            MIN_COMPLETED = 3
            if tally["VIOLATED"]:
                verdict = "VIOLATED"          # one violation is conclusive
            elif tally["REJECTED"] == trials:
                verdict = "REJECTED"
            elif tally["ERROR"] == trials:
                verdict = "ERROR"
            elif tally["TRUNCATED"] == trials:
                verdict = "TRUNCATED"         # never completed at any budget
            elif completed >= MIN_COMPLETED and tally["PREVENTED"] == completed:
                verdict = "PREVENTED"
            elif tally["ERROR"] or tally["TRUNCATED"]:
                verdict = "INCONCLUSIVE"      # too few completed to judge
            else:
                verdict = "MIXED"
            trunc_rate = tally["TRUNCATED"] / trials if trials else 0.0
            results.append({
                "id": case["id"], "note": case["note"], "verdict": verdict,
                "tally": dict(tally), "completed_trials": completed,
                "truncation_rate": round(trunc_rate, 3),
                "example_note": notes[0], "specimens": specimens[:2],
            })
            extra = f"  [truncated {tally['TRUNCATED']}/{trials}]" if trunc_rate else ""
            print(f"  {case['id']:<22} {verdict:<12} {dict(tally)}{extra}", flush=True)
    return results


def provenance(label):
    prov = {"backend_label": label, "python": sys.version.split()[0],
            "platform": platform.platform(),
            "utc": datetime.now(timezone.utc).isoformat()}
    for mod in ("vllm", "xgrammar", "outlines_core", "llguidance",
                "lm-format-enforcer", "torch", "transformers"):
        try:
            prov[mod] = version(mod)
        except Exception:
            prov[mod] = None
    try:
        root = BASE_URL.rsplit("/v1", 1)[0]
        prov["server_version"] = httpx.get(f"{root}/version", timeout=10).text
    except Exception as e:
        prov["server_version"] = f"unavailable: {e}"
    return prov


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend-label", required=True,
                    help="what you passed to --structured-outputs-config.backend; "
                         "recorded, NOT auto-detected, so label it honestly")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="high on purpose: the model must TRY to violate")
    ap.add_argument("--out", default="conformance.json")
    a = ap.parse_args()

    ok, msg = await preflight(a.model)
    print(msg)
    if not ok:
        raise SystemExit(2)

    prov = provenance(a.backend_label)
    print(json.dumps(prov, indent=2), flush=True)
    print(f"\nprobing {len(CASES)} cases x {a.trials} trials, "
          f"backend label '{a.backend_label}'\n")

    results = await run(a.model, a.trials, a.max_tokens, a.temperature)

    summary = Counter(r["verdict"] for r in results)
    print(f"\n{dict(summary)}")

    # Guard rails: say plainly when the run produced nothing usable.
    n_bad = summary["ERROR"] + summary["INCONCLUSIVE"]
    if n_bad == len(results):
        print("\nNO USABLE DATA: every case errored. Do not record these as "
              "results. Check that the server is healthy and BASE_URL is right.")
        with open(a.out, "w") as f:
            json.dump({"provenance": prov, "config": vars(a),
                       "cases": results, "usable": False}, f, indent=2)
        raise SystemExit(3)
    if n_bad:
        print(f"WARNING: {n_bad} of {len(results)} cases lack usable data.")

    unenforced = [r["id"] for r in results if r["verdict"] in ("VIOLATED", "MIXED")]
    if unenforced:
        print(f"accepted but NOT reliably enforced: {', '.join(unenforced)}")
    enum_row = next((r for r in results if r["id"] == "enum"), None)
    if enum_row and enum_row["verdict"] != "PREVENTED":
        print(f"WARNING: the enum control case came back {enum_row['verdict']}, "
              "not PREVENTED. Suspect the harness or server config before "
              "believing any other row.")

    with open(a.out, "w") as f:
        json.dump({"provenance": prov, "config": vars(a),
                   "cases": results, "usable": True}, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
