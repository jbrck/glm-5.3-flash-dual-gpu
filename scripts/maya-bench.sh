#!/usr/bin/env bash
# usage: MAYA_DIR=... CFG=... maya-bench.sh <label> [extra maya.py args...]
set -u
LABEL=$1; shift
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH:-$SCRIPTS/..}
OUT=${OUT:-$BENCH/runs/decode-results.csv}
LOG=${LOGDIR:-$BENCH/runs/logs}/$LABEL.log
mkdir -p "$(dirname "$OUT")" "$(dirname "$LOG")"
PORT=8180
cd "${MAYA_DIR:?set MAYA_DIR to your project-maya checkout}" || exit 1
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
[ "$used" -gt 2000 ] && { echo "GPUs hold ${used} MiB, abort"; exit 1; }
ss -tln | grep -q ":$PORT " && { echo "port $PORT busy, abort"; exit 1; }
T0=$(date +%s)
setsid ./.venv/bin/python maya.py --config ${CFG:?set CFG to your Maya config json} --port $PORT "$@" > "$LOG" 2>&1 &
PID=$!
OK=0
for i in $(seq 1 180); do
  sleep 5
  curl -sf -m 4 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { OK=1; break; }
  kill -0 $PID 2>/dev/null || break
done
LOAD=$(( $(date +%s) - T0 ))
if [ "$OK" = 1 ]; then
  MODEL=$(curl -s -m 4 http://127.0.0.1:$PORT/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
  echo "$LABEL,loadinfo,0,$LOAD,0,0,0,0,0,load_s=$LOAD model=$MODEL" >> $OUT
  for r in 1 2 3; do
    python3 $SCRIPTS/smoke-single.py --url http://127.0.0.1:$PORT --model "$MODEL" --label "$LABEL#r$r" --out $OUT || echo "BENCH FAIL r$r"
  done
  grep -E "glm stat|glm spec|vram hit|expert cache" "$LOG" | tail -4 | cut -c1-300
else
  echo "$LABEL,loadfail,1,,,,,,,see $LOG" >> $OUT
  echo "NO HEALTH $LABEL"; tail -8 "$LOG" | cut -c1-250
fi
kill -TERM -- -$PID 2>/dev/null || kill $PID 2>/dev/null
sleep 5
kill -KILL -- -$PID 2>/dev/null
for i in $(seq 1 40); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$u" -lt 2000 ] && break; sleep 3
done
echo "DONE $LABEL $(date +%T)"
