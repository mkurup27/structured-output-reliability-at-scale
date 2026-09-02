# results/ — run output

**This directory is empty in the published repository.** No result data is included. What follows is what lands here when you run the driver, and what to check before trusting any of it.

```bash
STAGES=all bash ../harness/run_gap_closure.sh
python3 ../analysis/summarize_runs.py *.json
```

## Why it's empty

Two separate reasons, and only one of them is a choice.

**Per-request outputs and server logs were not retained for the earliest vLLM ramp.** That predates `--dump-records`, so those tables can be reproduced from the harness but not replayed from that run. The data does not exist to publish.

**Later runs did retain records**, via `--dump-records`, and those files are not included here.

## What the driver writes

| File | Stage | Arm |
|---|---|---|
| `r1_sustained_strict.json` | 1 | Sustained load past batch capacity, strict |
| `r1b_sustained_prompt_only.json` | 1 | Same, no enforcement |
| `r1c_long_generations.json` | 1 | KV-cache pressure from sequence length rather than request count |
| `r2_ttft_strict.json` | 2 | Streaming TTFT pass, strict |
| `r2b_ttft_prompt_only.json` | 2 | Streaming TTFT pass, prompt-only |
| `r3_multiturn_strict.json` | 3 | Two-turn agent task |
| `r3b_agentmix_strict.json` | 3 | Four-task set, strict (the relocation arm) |
| `r3b_agentmix_prompt_only.json` | 3 | Four-task set, prompt-only (its matched control) |
| `r4_tight_budget.json` | 4 | Truncation observed directly at a tight token budget |
| `r4b_edge_schemas.json` | 4 | Three schemas at xgrammar's enforcement boundary |
| `r4c_varying_seed.json` | 4 | Temperature 0.7, `--seed -1` (the sample-independence control) |
| `r6_deep_multiturn_strict.json` | 6 | Five-step agent conversation |
| `r6b_deep_multiturn_prompt_only.json` | 6 | Same, no enforcement |
| `r6c_agentmix_deep_strict.json` | 6 | Default set plus the five-step task |
| `r7b_cardinality_sustained.json` | 7 | Cardinality ladder under sustained load |

Stage 5 (managed endpoint) and stage 8 (undersized grammar cache) write their own files and are both opt-in.

## What each file contains

Per level: request count, schema and semantic validity, truncation rate by termination metadata, taxonomy category and subcategory counts, p50/p95/p99, TTFT when streaming, `cost_per_usable_usd` measured from real token counts, `distinct_schemas_used`, and `server_metrics` sampled across the measured window when `--metrics-url` was set. Plus `specimens_by_category`: one real captured raw output per failure kind.

With `--dump-records`, per-request records too.

## Before you trust a run

**Check the level's queue depth against the server's `max_num_seqs`.** A concurrency level below batch capacity cannot queue, and flat validity there is a claim about batching headroom rather than about load. `summarize_runs.py` flags these as untested rather than passing, but the flag only helps if you read it.

**Check `distinct_schemas_used` against the cardinality you offered.** Round-robin needs at least 2N requests per level to reach the end of the rotation. A run that fell short is not the cardinality on its label: the sustained reduced-cache arm was labelled 2,048 and reached 1,609.

**Check the resolved backend, not the requested one.** `auto` latches on the first structured-output request the process ever handled.

**Check whether transport failures are in your denominator.** They produce no document to classify and belong on their own dashboard row. In the published ramp every sub-1.00 schema-validity entry was a dropped HTTP connection from the client's own pool, and disabling connection reuse returned validity to 1.00 with the queue still 266 deep.

## Naming

`OUT=` overrides this directory. Keep arms measured under different server configurations in separate directories. A reduced-cache run sitting alongside default-cache runs is how a summary table ends up mixing two servers.
