#!/usr/bin/env bash
# usage: IK_BIN=... MODEL_GGUF=... [ROWS_FILE=rows/batch1.psv] [REPS=3] decode-matrix.sh '<regex of row names>'
# row format: name|ENV=VAL ENV2=VAL (or -)|binary (@IK@)|model_path (@MODEL@)|alias|extra args
set -u
# Put your CUDA libs on LD_LIBRARY_PATH before running if they are not on the default path.
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH:-$SCRIPTS/..}
OUT=${OUT:-$BENCH/runs/decode-results.csv}
LOGDIR=${LOGDIR:-$BENCH/runs/logs}
ROWS=${ROWS_FILE:-$BENCH/rows/batch4.psv}
PORT=${PORT:-18090}
REPS=${REPS:-3}
FILTER=${1:-'.+'}
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
if ss -tln | grep -q ":$PORT "; then echo "port $PORT busy, abort"; exit 1; fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
if [ "$used" -gt 2000 ]; then echo "GPUs hold ${used} MiB, stop the served model first, abort"; exit 1; fi
while IFS='|' read -r name envs bin model alias extra; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  echo "$name" | grep -qE "^($FILTER)\$" || continue
  [ "$envs" = "-" ] && envs=""
  bin=${bin//@IK@/${IK_BIN:?set IK_BIN to your ik_llama.cpp llama-server}}
  model=${model//@MODEL@/${MODEL_GGUF:?set MODEL_GGUF to the first GGUF shard}}
  if [ ! -x "$bin" ]; then echo "SKIP $name (no binary)"; continue; fi
  if [ ! -f "$model" ]; then echo "SKIP $name (no model)"; continue; fi
  echo "=== $name $(date +%T) ==="
  TSTART=$(date +%s)
  env $envs "$bin" -m "$model" $extra --host 127.0.0.1 --port "$PORT" --alias "$alias" \
      > "$LOGDIR/$name.log" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 120); do
    sleep 5
    curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && { OK=1; break; }
    kill -0 "$PID" 2>/dev/null || break
  done
  LOAD=$(( $(date +%s) - TSTART ))
  if [ "$OK" = 1 ]; then
    echo "$name,loadinfo,0,$LOAD,0,0,0,0,0,load_s=$LOAD" >> "$OUT"
    for r in $(seq 1 "$REPS"); do
      python3 "$SCRIPTS/smoke-single.py" --url "http://127.0.0.1:$PORT" --model "$alias" \
        --label "$name#r$r" --out "$OUT" || echo "BENCH FAIL $name r$r"
    done
    grep -m4 -i -E "mtp|spec|accept|offload" "$LOGDIR/$name.log" | cut -c1-200
  else
    echo "$name,loadfail,1,,,,,,,see $LOGDIR/$name.log" >> "$OUT"
    echo "NO HEALTH $name"; tail -3 "$LOGDIR/$name.log"
  fi
  kill "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  for i in $(seq 1 60); do ss -tln | grep -q ":$PORT " || break; sleep 1; done
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$u" -lt 2000 ] && break; sleep 3
  done
done < "$ROWS"
echo "ALL DONE $(date +%T)"
