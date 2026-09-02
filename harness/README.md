# harness/ — the measurement code

Everything in this directory produces data. Two scripts run offline; the ramp and conformance tools need a live OpenAI-compatible endpoint; the driver runs the current instrumented arms in stage order.

| Script | Needs a server | Runtime | Produces |
|---|---|---|---|
| `truncation_experiment.py` | no | ~6 s | Cut-point recoverability table, validation-ladder timings |
| `cost_of_failure.py` | no | instant | Modelled cost per valid output, retry amplification, cache-cardinality table |
| `validity_ramp_harness_v4.py` | **yes** | minutes to hours | The sustained, streaming, follow-up, agent and cardinality ramps |
| `validity_ramp_harness_burst.py` | **yes** | minutes | The burst arm, kept for provenance only |
| `validity_ramp_harness_managed.py` | **yes** | minutes | The managed-endpoint arm, kept for provenance only |
| `run_gap_closure.sh` | **yes** | ~1 hour for `all` | Runs every arm, in order, into `../results/` |

```bash
pip install -r ../requirements.txt
```

---

## `truncation_experiment.py` — offline

Generates 200 documents per schema family, cuts each at every approximate token boundary, runs the cut through a repair heuristic, and asks four separate questions of the result rather than one: does it parse, does it validate against the schema, does it pass the business rules, and is it the document the model was actually writing.

Past roughly 60% completion the first three all answer yes and the fourth answers no.

```bash
python3 truncation_experiment.py
```

Four families (`flat`, `nested`, `enum_heavy`, `array_of_objects`), 132,737 cut points total, seed 20260813. It also runs the validation-ladder benchmark: per-document cost of parse, schema-validate and semantic-validate.

**Correctness rates are version-independent; the timings are not.** The rates reproduced to four decimal places across `jsonschema` 4.26.0 / Python 3.12.3 and `jsonschema` 3.2.0 / Python 3.10.12 on different hardware. The timings moved by 5.6× to 7.7× between the same two stacks, and most of that is interpreter and hardware rather than the validator library. The `json.loads` control column, which never touches `jsonschema`, moved 5.2× on its own. The script prints its own Python and `jsonschema` versions in the ladder header; quote them with any microsecond figure you report.

Cuts land at regex-approximated token boundaries rather than the target model's BPE vocabulary, so the completion axis is coarser than a real generation. Cut points are pooled per document, so the output is a shape rather than a production failure probability: it describes what happens *given* a cut at a point of completion.

## `cost_of_failure.py` — offline

Pure arithmetic over published prices. Retry amplification and cost-per-valid-output across a range of assumed failure rates, plus the grammar-cache cardinality table. Substitute your own prices; nothing here is measured.

```bash
python3 cost_of_failure.py
```

**The article's *measured* cost figures do not come from this script.** Those come from the ramp harness summing real `prompt_tokens` and `completion_tokens` per level and dividing by outputs that passed every rung, so they need no failure-rate assumption at all. This script is the modelled companion, useful for asking what a failure rate would cost you before you have one.

Its central caveat: it assumes one retry recovers a failure. That is false for deterministic semantic failures, since retrying the invoice task at temperature 0 reproduces the same wrong total forever, so where the failure is deterministic these are lower bounds.

## `validity_ramp_harness_v4.py` — the ramp

Points at any OpenAI-compatible endpoint and measures schema validity as a function of offered load. It produced the instrumented ramps; the historical burst and managed arms retain their original harnesses below.

```bash
export BASE_URL=http://localhost:8000/v1 API_KEY=EMPTY

# the historical burst configuration on the current harness
python3 validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode strict --levels 1,10,50,100 --requests-per-level 200

# sustained load past batch capacity, with server-side contention signals
python3 validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode strict --levels 1,10,50,100,200,400 --duration-s 180 --warmup-s 30 \
  --metrics-url http://localhost:8000/metrics --dump-records

# the schema-cardinality ladder; three requests per schema so the second
# pass reads as a cache hit against the first pass's compile
python3 validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode strict --levels 50 --requests-per-level 6144 \
  --schema-cardinality 2048 --stream --dump-records
```

### What it records per request

Schema validity, semantic validity against ground truth, truncation by termination metadata, exactly one taxonomy category and subcategory, end-to-end p50/p95/p99, client-side TTFT when streaming, measured cost per usable output from real token counts, and server-side contention signals sampled across the measured window.

### Load modes

`--requests-per-level N` offers a fixed-count burst. `--duration-s N` runs a closed loop of persistent workers issuing back-to-back requests for a wall-clock window, with `--warmup-s` discarded separately.

The distinction decides what a result means. Two hundred requests at concurrency 100 is about two request waves; a 180-second window at the same concurrency is sustained load with a populated KV cache and a real queue. **Only the sustained mode can test whether flat validity survives cache pressure rather than outrunning it.**

### Task sets

| `--taskset` | What it is |
|---|---|
| `default` | The three single-turn templates the published burst ramp used. Verified identical in content and order to `validity_ramp_harness_burst.py`, so per-level rates are comparable across versions. |
| `four` | Adds `enum_triage`. **Changes every denominator.** |
| `edge` | Three schemas at xgrammar's enforcement boundary: an untyped `pattern` fragment, a `multipleOf` integer, and a typed-`string` control. |
| `multiturn` | Tool-call turn, synthetic tool result spliced into the transcript, second constrained turn. Intermediate turn validated separately. |
| `agent_mix` | `default` plus the two-turn agent task. This is the set that produced the relocation result. |
| `deep_multiturn` | Five turns: two tool calls, an arithmetic step, a planning step with an array, then a final answer that must still carry values first seen in turn 1. Every turn validated. |
| `agent_mix_deep` | `default` plus the five-step task. |

**No two task sets share a denominator.** Only `default` is comparable to the published numbers.

### Key flags

- `--mode strict` uses `structured_outputs`; `--mode prompt_only` states the schema in prompt text with no grammar enforcement. The prompt-only arm is the baseline everyone starts with, and running both on identical prompts is what distinguishes a failure that enforcement *removed* from one it *moved*.
- `--schema-cardinality N` synthesizes N structurally distinct schemas (differing in field count, names, types and enum members, so no two compile to the same grammar) and round-robins one per request. Round-robin is the worst case for an LRU cache on purpose: a schema's next request arrives only after every other schema has compiled. **Offer at least 2N requests per level** or the ramp never reaches the end of the rotation. The output reports `distinct_schemas_used` so you can check that it did.
- `--stream` records TTFT from the first content chunk. **Run it as a separate pass.** Streaming changes the reassembly path, which is itself one of the failure modes under study.
- `--metrics-url` samples vLLM's `/metrics` only across the measured window, so preemption and KV-cache occupancy line up with the validity numbers from that same window. Preemption is reported as a delta over the window, not the process-lifetime counter.
- `--no-keepalive` disables HTTP connection reuse. This is the control that established the transport failures were the client's pool rather than the server. At a p50 near ten seconds, pooled connections sat idle longer than the server's keepalive timeout.
- `--dump-records` writes per-request records. The published burst ramp did not have this, which is why its tables can be reproduced from the harness but not replayed from that run.

### Reading a result honestly

**Check your offered concurrency against the server's `max_num_seqs` before reading anything into a flat validity line.** Concurrency below batch capacity cannot queue, so a flat result there is a statement about batching headroom rather than about load. That is exactly what the first version of this ramp did: it stopped at concurrency 100 against a 128-slot batch and concluded that concurrency does not affect validity, on a server that was never contended. `summarize_runs.py` flags any level where nothing queued as untested rather than passing.

Also record the *resolved* structured-output backend rather than the one you requested. See the driver's notes on `auto`.

## `validity_ramp_harness_burst.py` — the burst arm, provenance only

The original burst ramp: three single-turn templates, fixed count, no instrumentation. Kept because the published burst arms ran on it. **Use v4 for everything.** Every flag it carries exists in v4 with the same name and default, so an old command line runs unchanged.

`validity_ramp_harness_managed.py` added `--taskset` and `--temperature`/`--seed`, and it's here because the published managed-endpoint arm ran on it. That arm therefore carries no TTFT, no `server_metrics` and no per-request records. v4 is a single-file consolidation of the series rather than a patch on it, and it's what you should use.

## `backend_conformance_probe.py` — is the keyword actually enforced?

Produces the article's 18-keyword enforcement grid. For each keyword it submits a schema plus a prompt that demands a violating value, five trials by default, and records whether the decoder actually prevented the violation. Declared support and enforced support are different things, and this measures the second one.

One backend per run, because the backend is a server-level setting:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --structured-outputs-config.backend xgrammar --enforce-eager

python3 backend_conformance_probe.py --model Qwen/Qwen2.5-7B-Instruct \
  --backend-label xgrammar --out ../results/conformance/conformance_xgrammar.json
```

`--backend-label` is a label only. It records what you claim the server is running and cannot verify it, so pin the backend on the server and keep the two in step.

Temperature defaults to 1.0 here rather than 0. The probe needs the model to *try* to violate the constraint, and greedy decoding makes a single deterministic attempt look like enforcement.

`run_conformance.sh` repeats the above across `BACKENDS`, managing one server per backend. Its header documents the manual path, which is three commands and more reliable.

Read the results with `../analysis/compare_conformance.py`, which refuses to print a grid if the `enum` control case failed, since a backend that can't enforce `enum` isn't measuring anything. On this study's install `outlines` and `lm-format-enforcer` return HTTP 500 on every request, so the guard fires and the published grid covers `xgrammar` and `guidance` only.

## `run_gap_closure.sh` — the driver

Runs the current instrumented arms in stage order, writing to `../results/`. The historical burst and managed artifacts retain their original filenames and harnesses.

```bash
ulimit -n 8192          # 400 concurrent connections exceeds the common 1024 default
export BASE_URL=http://SERVER:8000/v1 API_KEY=EMPTY
export METRICS=http://SERVER:8000/metrics

STAGES=all bash run_gap_closure.sh
```

Environment: `MODEL` (default `Qwen/Qwen2.5-7B-Instruct`), `OUT` (default `../results`), `METRICS` (optional), `STAGES` (default `6,7`), `XGRAMMAR_CACHE_MB` for stage 8, and `DO_BASE_URL` / `DO_API_KEY` / `DO_MODEL` for the managed arm.

**`STAGES` defaults to `6,7`, not to everything.** Rounds one and two were months apart in practice, and running the file top to bottom after round one has landed silently redoes about an hour of finished work and overwrites its result files. Stage numbers match the Run numbers in the script, and a stage covers its lettered sub-runs: stage 1 is Runs 1, 1b and 1c.

| `STAGES=` | Arm | Output |
|---|---|---|
| `1` | Sustained load past batch capacity | `r1_*`, `r1b_*`, `r1c_*` |
| `2` | TTFT streaming pass, both arms | `r2_*`, `r2b_*` |
| `3` | Two-turn agent + `agent_mix` relocation | `r3_*`, `r3b_*` |
| `4` | Tight budget, edge schemas, varying seed | `r4_*`, `r4b_*`, `r4c_*` |
| `5` | Managed endpoint (needs `DO_BASE_URL`; current v4 format) | `r5_managed_rep1.json`, `r5_managed_rep2.json` |
| `6` | Five-step agent, both arms | `r6_*`, `r6b_*`, `r6c_*` |
| `7` | Cardinality ladder, default cache | `r7_cardinality_*`, `r7b_*` |
| `8` | Cardinality ladder, **undersized cache** | reduced-cache files |
| `all` | Everything **except 8** | |

### Stage 8 must run alone

It requires the server running with a deliberately undersized grammar cache, which invalidates every other stage's numbers. It is excluded from `all` deliberately and has to be asked for by number.

```bash
VLLM_XGRAMMAR_CACHE_MB=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --structured-outputs-config.backend xgrammar --max-num-seqs 128 --enforce-eager

# verify in the process environment, not from the launch command
tr '\0' '\n' < /proc/$(pgrep -f '[v]llm serve' | head -1)/environ | grep XGRAMMAR

# on the client host
XGRAMMAR_CACHE_MB=1 STAGES=8 OUT=../results/cardinality/cache-1mib \
  bash run_gap_closure.sh
```

An exported variable left over from a previous shell is exactly how a run ends up mislabelled. Run stage 7 at the default cache first, then restart reduced and run stage 8, then restart at the default and repeat one rung as a control. Without that last step the restart is an alternative explanation for everything the reduced-cache arm shows. Both `pgrep` and the `vllm serve` command run on the **serving** host; everything else runs on the client.

### Serving-host prerequisites

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --structured-outputs-config.backend xgrammar \
  --max-num-seqs 128 --enforce-eager
```

**Pin the backend.** vLLM's `auto` latches on the first structured-output request the process handles and keeps it for the process lifetime, silently, so a ramp run against `auto` cannot say which backend produced its numbers. Record the resolved backend and its library version before starting. vLLM 0.27.1 specifies a *range* for xgrammar rather than an exact pin, so the library version is part of the identifier.

`--max-num-seqs` is load-bearing for the interpretation of every result here rather than a tuning knob. Record its value with the results. Stage 1 deliberately ramps past it.

Capture `pip freeze` on both hosts.

---

## Local modification

`run_gap_closure.sh` was changed for this repository to resolve the harness relative to its own location (`BASH_SOURCE`) instead of the working directory, so the driver works from the split `harness/` + `analysis/` layout. Path plumbing only; no arm, flag, or default changed.
