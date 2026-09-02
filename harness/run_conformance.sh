#!/usr/bin/env bash
# run_conformance.sh -- probe each grammar backend, one server per backend.
#
# If you would rather not let a script manage the server, don't. The manual
# path is more reliable and only three commands:
#
#   1)  vllm serve $MODEL --structured-outputs-config.backend xgrammar --enforce-eager
#   2)  curl -sf http://localhost:8000/health && echo ready
#   3)  python3 backend_conformance_probe.py --model $MODEL \
#           --backend-label xgrammar --out conformance_xgrammar.json
#
# Repeat with a different backend. The probe refuses to run against an
# endpoint that isn't serving, so step 2 is a courtesy, not a requirement.
#
# Automated use:
#   chmod +x run_conformance.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct ./run_conformance.sh
#
# Env: PORT TRIALS BACKENDS STARTUP_TIMEOUT VLLM_ARGS API_KEY OUT_DIR

# Re-exec under bash if we were started by another shell.
if [ -z "${BASH_VERSION-}" ]; then
  exec bash "$0" "$@"
fi

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults assigned WITHOUT ${VAR:--flag} style substitution. That construct
# tripped at least one bash build with "bad substitution", and when it failed
# under `set -u` every downstream server launch died before it could even
# create its log file. Plain guards are boring and portable.
if [ -z "${MODEL-}" ]; then
  echo "set MODEL, e.g. MODEL=Qwen/Qwen2.5-7B-Instruct $0" >&2
  exit 1
fi
[ -n "${PORT-}" ]            || PORT=8000
[ -n "${TRIALS-}" ]          || TRIALS=5
[ -n "${BACKENDS-}" ]        || BACKENDS="xgrammar guidance outlines lm-format-enforcer"
[ -n "${STARTUP_TIMEOUT-}" ] || STARTUP_TIMEOUT=900
[ -n "${API_KEY-}" ]         || API_KEY=EMPTY
[ -n "${OUT_DIR-}" ]         || OUT_DIR="$HERE/../results/conformance"
if [ -z "${VLLM_ARGS+x}" ]; then
  VLLM_ARGS="--enforce-eager"
fi

BASE_URL="http://localhost:${PORT}/v1"
export BASE_URL API_KEY

command -v vllm >/dev/null 2>&1 || { echo "vllm not on PATH" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not on PATH" >&2; exit 1; }
python3 -c "import httpx, jsonschema" 2>/dev/null \
  || { echo "pip install httpx jsonschema" >&2; exit 1; }
[ -f "$HERE/backend_conformance_probe.py" ] \
  || { echo "backend_conformance_probe.py not found beside this script" >&2; exit 1; }

echo "model      : $MODEL"
echo "port       : $PORT"
echo "backends   : $BACKENDS"
echo "vllm args  : $VLLM_ARGS"
echo "trials     : $TRIALS"

mkdir -p "$OUT_DIR/logs"
SUMMARY=""
SERVER_PID=""

stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  stopping server pid $SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null
    n=0
    while kill -0 "$SERVER_PID" 2>/dev/null && [ "$n" -lt 30 ]; do
      sleep 1; n=$((n + 1))
    done
    kill -9 "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}
trap 'echo; echo interrupted; stop_server; exit 130' INT TERM

for B in $BACKENDS; do
  LOG="$OUT_DIR/logs/server_${B}.log"
  : > "$LOG"          # create it up front so tail/grep always have a target
  echo
  echo "=============================================================="
  echo "backend: $B"
  echo "=============================================================="

  # shellcheck disable=SC2086
  vllm serve "$MODEL" \
      --port "$PORT" \
      --structured-outputs-config.backend "$B" \
      $VLLM_ARGS >>"$LOG" 2>&1 &
  SERVER_PID=$!
  echo "  server pid $SERVER_PID, log $LOG"

  READY=0
  ELAPSED=0
  while [ "$ELAPSED" -lt "$STARTUP_TIMEOUT" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  SERVER EXITED after ${ELAPSED}s. Last 40 log lines:"
      tail -n 40 "$LOG" 2>/dev/null | sed 's/^/    /'
      if grep -qiE 'out of memory|OutOfMemory' "$LOG" 2>/dev/null; then
        echo "  --> OOM. Try VLLM_ARGS=\"--enforce-eager --gpu-memory-utilization 0.85 --max-model-len 4096\""
      fi
      if grep -qiE "invalid choice|unrecognized argument|not a valid|unknown backend" "$LOG" 2>/dev/null; then
        echo "  --> vllm rejected an argument. If it is the backend name, this"
        echo "      build may not offer '$B'; drop it from BACKENDS."
      fi
      break
    fi
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
      READY=1
      echo "  healthy after ${ELAPSED}s"
      break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
  done

  if [ "$READY" -ne 1 ]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  timed out after ${STARTUP_TIMEOUT}s waiting for /health"
      tail -n 20 "$LOG" 2>/dev/null | sed 's/^/    /'
    fi
    SUMMARY="${SUMMARY}
  ${B}: SERVER FAILED (see $LOG)"
    stop_server
    continue
  fi

  OUT="$OUT_DIR/conformance_${B}.json"
  if python3 "$HERE/backend_conformance_probe.py" \
       --model "$MODEL" --backend-label "$B" \
       --trials "$TRIALS" --out "$OUT"; then
    SUMMARY="${SUMMARY}
  ${B}: ok -> $OUT"
  else
    RC=$?
    echo "  probe exited $RC for $B"
    SUMMARY="${SUMMARY}
  ${B}: PROBE FAILED (exit $RC)"
  fi

  stop_server
  sleep 5
done

echo
echo "=============================================================="
printf '%s\n' "$SUMMARY"
echo "=============================================================="
echo
if ls "$OUT_DIR"/conformance_*.json >/dev/null 2>&1; then
  echo "Compare with:  python3 \"$HERE/../analysis/compare_conformance.py\" \"$OUT_DIR\"/conformance_*.json"
else
  echo "No result files were written. Nothing to compare."
fi
