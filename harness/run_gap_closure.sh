#!/usr/bin/env bash
# Closes the experimental gaps the published ramp left open, in the order that
# puts the result most likely to change the article's headline first.
#
# Runs 1-5 are the first round and have been executed; their results are in the
# article. Runs 6 and 7 are the second round, for the two gaps that first round
# left open because the harness had no task that could reach them: conversation
# depth beyond two turns, and more distinct schemas in flight than the grammar
# cache holds. Neither is a re-analysis of round one -- there was nothing in
# those runs to re-analyse.
#
# Prerequisites on the serving host: vLLM up with the structured-output backend
# PINNED, not left on auto. Record the resolved backend and its library version
# before starting -- `auto` latches on the first structured-output request the
# process handles, so a ramp run against `auto` cannot say which backend
# produced its numbers.
#
#   vllm serve Qwen/Qwen2.5-7B-Instruct \
#     --structured-outputs-config.backend xgrammar \
#     --max-num-seqs 128 --enforce-eager
#
# --max-num-seqs is load-bearing for the interpretation of every result here,
# not just a tuning knob: it is the batch capacity, and offered concurrency
# below it cannot queue. Record its value with the results. Run 1 deliberately
# ramps past it.
#
# On the client host, raise the descriptor limit before Run 1 -- 400 concurrent
# connections plus keepalive exceeds the common 1024 default once httpx and the
# interpreter take their share:
#
#   ulimit -n 8192
#
# Then, on the client host. Note BASE_URL is the SELF-HOSTED vLLM server for
# every run except Run 5; the managed endpoint is a separate opt-in arm with its
# own URL and token, and the two must not be merged into one table.
#   pip install httpx jsonschema
#   export BASE_URL=http://SERVER:8000/v1 API_KEY=EMPTY
#   export METRICS=http://SERVER:8000/metrics
#   bash run_gap_closure.sh
#
# Capture `pip freeze` on both hosts. Every run below writes per-request records
# (--dump-records), which the published ramp did not: its tables can be
# reproduced from the harness but not replayed from the run.

set -euo pipefail

# Resolve the harness next to this script rather than in the caller's working
# directory, so the driver runs from anywhere in the repo.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
METRICS="${METRICS:-}"
H="python3 $HERE/validity_ramp_harness_v4.py --model $MODEL --dump-records"
[ -n "$METRICS" ] && H="$H --metrics-url $METRICS --metrics-interval 2"
OUT="${OUT:-$HERE/../results}"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# Stage selection. Rounds one and two are months apart in practice, and running
# this file top to bottom after round one has landed silently redoes about an
# hour of finished work and overwrites its result files. So pick stages:
#
#   STAGES=7 bash run_gap_closure.sh          # cardinality only
#   STAGES=6,7 bash run_gap_closure.sh        # the whole second round
#   STAGES=all bash run_gap_closure.sh        # everything, round one included
#
# Default is the second round, since round one is already done. Set STAGES=all
# deliberately if you mean to re-run the published arms.
#
# Stage numbers match the Run numbers below, and a stage covers its lettered
# sub-runs: stage 1 is Runs 1, 1b and 1c; stage 7 is Runs 7 and 7b.
#
# Stage 8 is not in the default set and is not in `all`, because it requires the
# server to be running with a deliberately undersized grammar cache, which
# invalidates every other stage's numbers. Run it on its own, then restore the
# server's default cache.
# ---------------------------------------------------------------------------
STAGES="${STAGES:-6,7}"

want() {
  # An explicit stage number always wins.
  case ",$STAGES," in
    *",$1,"*) return 0 ;;
  esac
  # `all` covers every stage except 8, which needs a server running with a
  # reduced grammar cache and so invalidates the arms around it. That one has to
  # be asked for by number.
  case ",$STAGES," in
    *,all,*)
      if [ "$1" = 8 ]; then return 1; fi
      return 0 ;;
  esac
  return 1
}

skip() { echo "-- skipping stage $1 ($2); set STAGES to include it"; }

echo "stages selected: $STAGES"
echo "output dir: $OUT"
echo

# ---------------------------------------------------------------------------
# Run 1 -- sustained load past the batch capacity. The gap that could actually
# move the headline, and the reason the ramp has to be extended rather than
# merely lengthened.
#
# A 12-request probe at concurrency 4 measured KV-cache occupancy of 0.0005
# (0.05%) with num_requests_waiting at 0. The server runs --max-num-seqs 128
# and the published ramp stopped at concurrency 100, so every request the
# published ramp ever offered fit inside a single batch. Nothing queued,
# nothing was preempted, and the KV cache never filled. "Concurrency did not
# change validity" was therefore measured on a server that was never
# contended: scheduler queueing was impossible by construction, not absent by
# observation. Lengthening that same ramp into a sustained window reproduces
# the non-result more thoroughly; it does not test the hypothesis.
#
# So the ramp goes past max_num_seqs. At concurrency 200 and 400 against a
# 128-slot batch, 72 and 272 requests respectively must wait, which is the
# first point in this entire study where the scheduler has to make a decision.
# Watch num_requests_waiting_max going positive -- that is the moment
# contention begins -- and then watch whether schema_valid_rate moves at all.
#
# Both outcomes are worth publishing and they say different things. If
# validity holds at 400 with a real queue, the null is enormously stronger
# than the published version and can be stated as a claim about contention
# rather than about offered load. If it degrades, the folklore was right and
# the published ramp simply never reached the regime where it applies.
# ---------------------------------------------------------------------------
if want 1; then
echo "== Run 1: sustained load past batch capacity, strict =="
$H --mode strict --levels 1,10,50,100,200,400 --duration-s 180 --warmup-s 30 \
   --out "$OUT/r1_sustained_strict.json"

echo "== Run 1b: sustained load past batch capacity, prompt_only baseline =="
$H --mode prompt_only --levels 1,10,50,100,200,400 --duration-s 180 --warmup-s 30 \
   --out "$OUT/r1b_sustained_prompt_only.json"

# Run 1c -- KV-cache pressure from sequence length rather than request count.
# Queueing and cache exhaustion are different contention mechanisms and the
# ramp above only produces the first: 400 short requests still occupy very
# little cache. Long generations at high concurrency are what force eviction
# and recompute, which is the mechanism most likely to interact with a
# grammar-constrained decode. 2048-token budgets at 100-400 concurrent puts
# real pressure on the cache for the first time in this study.
echo "== Run 1c: long generations, sustained =="
$H --mode strict --levels 100,400 --duration-s 180 --warmup-s 30 \
   --max-tokens 2048 --out "$OUT/r1c_long_generations.json"
else skip 1 "sustained load"; fi

# ---------------------------------------------------------------------------
# Run 2 -- TTFT and contention signals. Separate pass, because streaming
# changes the response-reassembly path and that path is itself one of the
# failure modes under study; folding it into the validity arm would confound
# the measurement. Fixed-count here so it is comparable to the published ramp.
#
# This is what lets the article say whether the strict arm's steeper p99 growth
# shows up at first token or only at completion. It still cannot attribute the
# gap to grammar-mask overhead -- that needs mask-time profiling, which stays
# scoped as future work.
# ---------------------------------------------------------------------------
if want 2; then
echo "== Run 2: TTFT pass, strict =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 --stream \
   --out "$OUT/r2_ttft_strict.json"

echo "== Run 2b: TTFT pass, prompt_only =="
$H --mode prompt_only --levels 1,10,50,100 --requests-per-level 200 --stream \
   --out "$OUT/r2b_ttft_prompt_only.json"
else skip 2 "TTFT pass"; fi

# ---------------------------------------------------------------------------
# Run 3 -- multi-turn agent workload, the brief's central workload-realism ask.
# Two schema-constrained turns with a tool result spliced between them. The
# intermediate turn is validated on its own, so multi_turn.
# intermediate_turn_failure_rate separates "the tool call was broken" from
# "the answer was broken" -- two failures that take different fixes and that a
# single-turn ramp cannot tell apart because it never produces the first one.
#
# agent_mix runs the multi-turn task alongside the single-turn set, which is
# closer to real traffic. Note the denominator: four tasks, so per-task rates
# are quarters, and none of these numbers compare directly to the published
# three-template ramp.
# ---------------------------------------------------------------------------
if want 3; then
echo "== Run 3: multi-turn only, strict =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
   --taskset multiturn --out "$OUT/r3_multiturn_strict.json"

echo "== Run 3b: agent mixture, both arms =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
   --taskset agent_mix --out "$OUT/r3b_agentmix_strict.json"
$H --mode prompt_only --levels 1,10,50,100 --requests-per-level 200 \
   --taskset agent_mix --out "$OUT/r3b_agentmix_prompt_only.json"
else skip 3 "two-turn agent"; fi

# ---------------------------------------------------------------------------
# Run 4 -- re-runs of the two follow-up arms under the v4 taxonomy, so the
# specimen table and the per-category counts come from the same instrumented
# code as everything else rather than from three differently-patched harnesses.
# ---------------------------------------------------------------------------
if want 4; then
echo "== Run 4: tight budget (truncation floor) =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
   --max-tokens 128 --out "$OUT/r4_tight_budget.json"

echo "== Run 4b: edge schemas (constraint boundary) =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
   --taskset edge --out "$OUT/r4b_edge_schemas.json"

echo "== Run 4c: varying seed (rates as proportions) =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
   --seed -1 --temperature 0.7 --out "$OUT/r4c_varying_seed.json"
else skip 4 "tight budget, edge schemas, varying seed"; fi

# ---------------------------------------------------------------------------
# Run 6 -- five-step agent conversation. Run 3 came back clean: adding one turn
# introduced no failure mode. That result stands, but two turns cannot exhibit
# the failure multi-turn is supposed to cause, because the transcript never gets
# long enough to compete with itself for the budget. Five turns do: two tool
# calls, an arithmetic step, a planning step, then a final answer that must
# still be carrying customer_id and provisioned_gb from turn 1's tool result and
# the $300.80 total from turn 3's arithmetic.
#
# What to read first in the output is multi_turn.turns_completed_hist and
# multi_turn.failures_by_turn, not the aggregate rate. A task that reaches turn
# 5 every time and fails there is losing context; one that dies at turn 3 is
# failing arithmetic; one that dies at turn 1 has a broken tool schema. Three
# different fixes, and the aggregate rate cannot tell them apart.
#
# Both arms, because relocation is the finding most likely to repeat here: under
# prompt_only a lost thread may surface as a parse failure instead.
#
# Note the budget. Turn 5 prefills the whole transcript, so --max-tokens 512
# governs each turn's completion while the prompt side grows turn over turn. If
# truncation appears only at the last turn, that is the context-growth failure
# this run exists to find, and it should be reported as such rather than as a
# budget misconfiguration.
# ---------------------------------------------------------------------------
#
# Runtime warning. Five sequential turns per request means concurrency 1 costs
# roughly 200 x 5 x the per-request latency, which on the measured ~1.7s is near
# half an hour for that one level, while 10/50/100 together come in under ten
# minutes. R6_C1_REQUESTS trims just that level; 50 conversations is enough for
# an uncontended baseline, which is all the level is for.
# ---------------------------------------------------------------------------
if want 6; then
R6_C1_REQUESTS="${R6_C1_REQUESTS:-200}"

echo "== Run 6: five-step agent, both arms =="
if [ "$R6_C1_REQUESTS" != "200" ]; then
  echo "   (concurrency 1 trimmed to $R6_C1_REQUESTS requests)"
  for M in strict prompt_only; do
    [ "$M" = strict ] && SFX=r6_deep_multiturn_strict || SFX=r6b_deep_multiturn_prompt_only
    $H --mode "$M" --levels 1 --requests-per-level "$R6_C1_REQUESTS" \
       --taskset deep_multiturn --out "$OUT/${SFX}_c1.json"
    $H --mode "$M" --levels 10,50,100 --requests-per-level 200 \
       --taskset deep_multiturn --out "$OUT/${SFX}.json"
  done
else
  $H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
     --taskset deep_multiturn --out "$OUT/r6_deep_multiturn_strict.json"
  $H --mode prompt_only --levels 1,10,50,100 --requests-per-level 200 \
     --taskset deep_multiturn --out "$OUT/r6b_deep_multiturn_prompt_only.json"
fi

echo "== Run 6c: five-step agent alongside the single-turn set =="
$H --mode strict --levels 1,10,50,100 --requests-per-level 200 \
   --taskset agent_mix_deep --out "$OUT/r6c_agentmix_deep_strict.json"
else skip 6 "five-step agent"; fi

# ---------------------------------------------------------------------------
# Run 7 -- schema cardinality, the grammar-cache arm. Every run above offered
# three or four schemas: they compiled once on the warm-up and every measured
# request was a cache hit, which is why compile cost never appeared in any
# latency number and preemption never fired. Ramping request count cannot reach
# this mechanism. Ramping the number of distinct grammars in flight can.
#
# The schemas are synthesized to differ in field count, field names, types and
# enum members, so no two compile to the same grammar and a cache keyed on the
# grammar cannot serve one for another. The warm-up is deliberately cut short
# here for the same reason: warming the whole set would move every compile off
# the measured window, which is exactly the condition being broken.
#
# Read distinct_schemas_used against distinct_schemas_offered first. If they
# differ, the rotation never reached the tail of the set and the arm tested a
# lower cardinality than it was configured for.
#
# The ladder crosses the threshold the essay describes from code-reading. If
# TTFT and p99 are flat across all four levels, the compiler cache is absorbing
# the churn and the ~1000-schema figure is wrong or backend-specific; if they
# step up at one level, that step is the cache size, measured.
#
# FIRST RESULT, 2026-08-27: flat. TTFT p50 was 0.4843 / 0.4826 / 0.4823 / 0.4592
# across 64 / 256 / 1024 / 2048, with the rotation verified complete at every
# rung and preemption at zero. Read that as the cache never evicting rather than
# eviction being free: 512 MiB over 2048 entries is 256 KiB each and these are
# small flat objects, so they almost certainly all stayed resident. Stage 8
# below moves the variable that actually binds.
#
# That run also carried a 37% truncation floor at every rung, from the task
# generator asking the model to invent values for meaningless field names; the
# generator now supplies concrete values. truncation_rate is the check, and it
# should read 0.0000 at every rung. If it does not, that failure is back and
# the validity, cost and kv_max columns are measuring it rather than the server.
# ---------------------------------------------------------------------------
if want 7; then
echo "== Run 7: schema cardinality ladder =="
for CARD in 64 256 1024 2048; do
  # 3x the cardinality, so every schema is requested at least twice and the
  # second pass reads as a cache hit against the first pass's compile.
  $H --mode strict --levels 50 --requests-per-level "$((CARD * 3))" \
     --schema-cardinality "$CARD" --stream \
     --out "$OUT/r7_cardinality_${CARD}.json"
done

echo "== Run 7b: cardinality under sustained load =="
# The ladder above runs at one concurrency to isolate cardinality. This one
# holds cardinality high and sustains load, which is the only configuration in
# the whole suite where cache churn and queueing are present at once -- the
# combination that could actually force preemption.
$H --mode strict --levels 100,400 --duration-s 180 --warmup-s 30 \
   --schema-cardinality 2048 --out "$OUT/r7b_cardinality_sustained.json"
else skip 7 "schema cardinality"; fi

# ---------------------------------------------------------------------------
# Run 8 -- the same ladder against a deliberately undersized grammar cache.
#
# Stage 7 offered 2048 distinct grammars and saw nothing: TTFT flat to within
# 25 ms across a 32x increase. The likely reason is that 2048 small flat schemas
# fit inside the default 512 MiB compiler cache, so the ladder measured cache
# headroom rather than eviction, which is the same mistake the original
# concurrency ramp made against max_num_seqs. Schema count is the variable the
# client can move; cache size is the variable that actually binds. So move that
# one instead and keep the ladder identical, which makes stage 8 a controlled
# comparison against stage 7 rather than a separate experiment.
#
# This needs the SERVER restarted, which this script cannot do for you:
#
#   VLLM_XGRAMMAR_CACHE_MB=16 vllm serve Qwen/Qwen2.5-7B-Instruct \
#     --structured-outputs-config.backend xgrammar \
#     --max-num-seqs 128 --enforce-eager
#
# Then declare the value you used, which gates the stage and lands in the
# filenames so the results are self-describing:
#
#   STAGES=8 XGRAMMAR_CACHE_MB=16 bash run_gap_closure.sh
#
# At roughly 256 KiB per compiled schema, 16 MiB holds on the order of 64, so
# the 16-schema rung should fit and everything above it should thrash. Two
# things to read. A step up in TTFT between two rungs is the cache boundary,
# measured rather than inferred from a source comment. And a flat TTFT even
# here would say the compile itself is cheap for schemas this small, which is a
# different and equally publishable answer -- it would mean the essay's
# ~1000-schema threshold is about memory rather than about latency.
#
# Restore the default cache before running any other stage. A reduced cache
# invalidates every other arm's numbers.
# ---------------------------------------------------------------------------
if want 8; then
  if [ -z "${XGRAMMAR_CACHE_MB:-}" ]; then
    echo "!! stage 8 skipped: set XGRAMMAR_CACHE_MB to the value the SERVER was"
    echo "   restarted with (e.g. XGRAMMAR_CACHE_MB=16). The client cannot set"
    echo "   VLLM_XGRAMMAR_CACHE_MB -- it is read by the serving process at"
    echo "   startup, so an unrestarted server would silently produce a second"
    echo "   copy of stage 7 under a filename claiming a smaller cache."
  else
    echo "== Run 8: cardinality ladder at VLLM_XGRAMMAR_CACHE_MB=$XGRAMMAR_CACHE_MB =="
    for CARD in 16 64 256 1024 2048; do
      $H --mode strict --levels 50 --requests-per-level "$((CARD * 3))" \
         --schema-cardinality "$CARD" --stream \
         --out "$OUT/r8_cache${XGRAMMAR_CACHE_MB}mb_cardinality_${CARD}.json"
    done

    echo "== Run 8b: undersized cache under sustained load =="
    # Eviction and queueing at once, which is the only configuration in either
    # round that could plausibly force preemption. It has read zero everywhere
    # else, including at concurrency 400 with 272 requests queued.
    $H --mode strict --levels 100,400 --duration-s 180 --warmup-s 30 \
       --schema-cardinality 2048 \
       --out "$OUT/r8b_cache${XGRAMMAR_CACHE_MB}mb_sustained.json"
  fi
else skip 8 "undersized grammar cache"; fi

# ---------------------------------------------------------------------------
# Run 5 -- the managed endpoint, arm (c). Opt-in, because it targets a
# DIFFERENT BASE_URL than everything above and needs a real DO token rather
# than API_KEY=EMPTY:
#
#   DO_BASE_URL=https://inference.do-ai.run/v1 DO_API_KEY=dop_v1_... \
#     DO_MODEL=mistral-3-14B bash run_gap_closure.sh
#
# Three deliberate differences from the runs above, none of them cosmetic.
#
# Fixed count, not --duration-s. Sustained mode issues thousands of requests
# per level instead of 200, which on a shared autoscaling endpoint means the
# rate limiter starts shaping the traffic and the concurrency axis stops
# meaning what it says. The published managed arm was specifically able to
# report that no request hit the limiter; that property is worth keeping.
#
# No --metrics-url. There is no /metrics to scrape: the backend, engine
# version, max_num_seqs and batching policy are all provider-internal. That
# opacity is the finding, not an omission -- it is why this arm can show
# validity degrading under load without being able to say what caused it.
#
# 1024 max tokens, not 512. A first run at 512 was confounded by the model's
# natural output length hitting the budget, so truncation there measured the
# budget rather than the endpoint.
# ---------------------------------------------------------------------------
# Gated on the stage as well as the URL, so a second-round run with DO_BASE_URL
# still exported in the shell does not silently re-run the managed arm and
# overwrite its result files.
if want 5 && [ -n "${DO_BASE_URL:-}" ]; then
  echo "== Run 5: managed endpoint (arm c), two independent repeats =="
  for rep in 1 2; do
    BASE_URL="$DO_BASE_URL" API_KEY="${DO_API_KEY:?set DO_API_KEY}" \
      python3 validity_ramp_harness_v4.py \
        --model "${DO_MODEL:-mistral-3-14B}" --dump-records \
        --mode strict --levels 1,10,50,100 --requests-per-level 200 \
        --max-tokens 1024 --out "$OUT/r5_managed_rep${rep}.json"
  done
fi

echo
echo "wrote results to $OUT  (stages: $STAGES)"
echo "Summarize with: python3 $HERE/../analysis/summarize_runs.py $OUT/*.json"
echo "For the article: specimens_by_category in each file is one real captured"
echo "output per failure kind, and cost_per_usable_usd is measured from token"
echo "usage rather than modelled from an assumed failure rate."
echo
echo "Every stage except 5 targeted BASE_URL=${BASE_URL:-unset} (self-hosted vLLM)."
if want 5 && [ -n "${DO_BASE_URL:-}" ]; then
  echo "Run 5 targeted $DO_BASE_URL (managed)."
else
  echo "Run 5 did not run."
fi
echo "Do not merge the two into one table: different model, different serving"
echo "layer, and only the self-hosted side has recorded engine internals."
