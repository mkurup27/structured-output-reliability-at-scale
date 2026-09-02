# results/ — run output

Every run behind the article's tables is published here, including per-request records where they were captured. What follows is the layout, what each file holds, and what to check before trusting any of it.

```bash
STAGES=all bash ../harness/run_gap_closure.sh
python3 ../analysis/summarize_runs.py .
```

## Layout

Arms measured under different server configurations sit in separate directories, because a reduced-cache run alongside default-cache runs is how a summary table ends up mixing two servers.

| Directory | What's in it |
|---|---|
| `.` (top level) | The single-server ramps: stages 1, 2, 3, 4 and 6, plus the no-keepalive control |
| `conformance/` | The 18-keyword enforcement probe, one file per backend, four backends |
| `cardinality/default-cache/` | Schema-cardinality ladder at vLLM's default 512 MiB grammar cache |
| `cardinality/cache-1mib/` | The same ladder at `VLLM_XGRAMMAR_CACHE_MB=1` |
| `cardinality/cache-16mib/` | The same ladder at 16 MiB, the run the article leaves unreconciled |
| `cardinality/repeat-control/` | 2,048 rungs re-run at the default cache, the there-and-back control |
| `masktime/` | Per-token decode, strict against prompt-only, three repeats each |
| `onset/` | First-wave TTFT at 16 against 512 schemas, three repeats each |
| `burst-ramp/` | The original bounded-burst ramp, on the burst harness |
| `managed-endpoint/` | DigitalOcean Serverless Inference, on the managed harness |

## Two files are gzipped

The sustained ramps are 16 MB and 21 MB uncompressed, so they ship compressed. `summarize_runs.py` reads them directly:

```bash
python3 ../analysis/summarize_runs.py r1_sustained_strict.json.gz
```

The driver writes plain JSON. From the repository root, create the published copies with:

```bash
gzip -k results/r1_sustained_strict.json results/r1b_sustained_prompt_only.json
```

## Four ramp files carry no per-request records

`burst-ramp/` and `managed-endpoint/` predate `--dump-records`, and they also lack TTFT and `server_metrics`. Their level-aggregate tables are complete and replayable. Their individual requests are not recoverable, and neither are the server logs from those runs.

The v4 ramp files were run with `--dump-records`, so their specimens can be recovered from `records`. Conformance outputs use their own per-case `specimens` arrays.

## What the driver writes

| File | Stage | Arm |
|---|---|---|
| `r1_sustained_strict.json` | 1 | Sustained load past batch capacity, strict; the published copy is gzipped |
| `r1b_sustained_prompt_only.json` | 1 | Same, no enforcement; the published copy is gzipped |
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
| `r7_cardinality_<N>.json` | 7 | Fixed-count cardinality rung |
| `r7b_cardinality_sustained.json` | 7 | Cardinality ladder under sustained load |

Stage 5 (managed endpoint) and stage 8 (undersized grammar cache) are both opt-in. Stage 5 now writes v4-format `r5_managed_rep*.json`; the published `managed-endpoint/results_do_*.json` files retain the historical managed-harness format. The published archive groups managed, cardinality, mask-time and onset runs by server configuration. Set `OUT` to the target directory when rerunning a grouped arm, for example:

```bash
OUT=results/cardinality/default-cache STAGES=7 bash harness/run_gap_closure.sh
XGRAMMAR_CACHE_MB=1 OUT=results/cardinality/cache-1mib STAGES=8 \
  bash harness/run_gap_closure.sh
```

The published 1 MiB fixed-count ladder starts at 64 schemas. Its 16-schema point comes from the three `onset/cache-1mib/onset_card16_rep*.json` runs because a single 48-request rung is dominated by the first concurrent wave.

## What each file contains

Per level: request count, schema and semantic validity, truncation rate by termination metadata, taxonomy category and subcategory counts, p50/p95/p99, TTFT when streaming, `cost_per_usable_usd` measured from real token counts, `distinct_schemas_used`, and `server_metrics` sampled across the measured window when `--metrics-url` was set.

With `--dump-records`, per-request records too.

## Supplemental manual arms

These published supplemental arms are not separate driver stages:

```bash
cd results

# Pooling-disabled control
python3 ../harness/validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode strict --levels 400 --duration-s 180 --warmup-s 30 \
  --metrics-url http://SERVER:8000/metrics --no-keepalive --dump-records \
  --out r6_nokeepalive_c400.json

# Per-token repeats: run three times per arm under each cache configuration
python3 ../harness/validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode strict --levels 10,50,100 --requests-per-level 200 --stream --dump-records \
  --out mask_strict_rep1.json
python3 ../harness/validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode prompt_only --levels 10,50,100 --requests-per-level 200 --stream --dump-records \
  --out mask_prompt_only_rep1.json

# Onset comparison: repeat at cardinality 16 and 512
python3 ../harness/validity_ramp_harness_v4.py --model Qwen/Qwen2.5-7B-Instruct \
  --mode strict --levels 50 --requests-per-level 1536 \
  --schema-cardinality 16 --stream --dump-records --out onset_card16_rep1.json
# Repeat with --schema-cardinality 512.
```

## Before you trust a run

**Check the level's queue depth against the server's `max_num_seqs`.** A concurrency level below batch capacity cannot queue, and flat validity there is a claim about batching headroom rather than about load. `summarize_runs.py` flags these as untested rather than passing, but the flag only helps if you read it.

**Check `distinct_schemas_used` against the cardinality you offered.** Round-robin needs at least 2N requests per level to reach the end of the rotation. A run that fell short is not the cardinality on its label: the sustained reduced-cache arm was labelled 2,048 and reached 1,609.

**Check the resolved backend, not the requested one.** `auto` latches on the first structured-output request the process ever handled.

**Check whether transport failures are in your denominator.** They produce no document to classify and belong on their own dashboard row. In the published ramp every sub-1.00 schema-validity entry was a dropped HTTP connection from the client's own pool, and disabling connection reuse returned validity to 1.00 with the queue still 266 deep.

## Naming

`OUT=` overrides this directory, which is how the per-configuration subdirectories above were produced.
