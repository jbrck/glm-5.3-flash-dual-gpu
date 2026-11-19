#!/usr/bin/env bash
# usage: MAYA_DIR=... BASE_CFG=... deep-ladder.sh <ctx1> <ctx2> ...   loads Maya at each ctx, fills ~90%, stops at first failure
set -u
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
BENCH=${BENCH:-$SCRIPTS/..}
P=${MAYA_DIR:?set MAYA_DIR to your project-maya checkout}
BASE_CFG=${BASE_CFG:?set BASE_CFG to your Maya config json}
OUT=${OUT:-$BENCH/runs/decode-results.csv}
mkdir -p "$BENCH/runs/logs" "$(dirname "$OUT")"
PORT=8180
for CTX in "$@"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$used" -gt 2000 ] && { echo "GPUs hold ${used} MiB, abort"; exit 1; }
  ss -tln | grep -q ":$PORT " && { echo "port $PORT busy, abort"; exit 1; }
  CFG=$P/bench-ctx$CTX.json
  python3 - "$CTX" "$CFG" "$BASE_CFG" <<"EOF"
import json, sys
ctx, out = sys.argv[1], sys.argv[2]
c = json.load(open(sys.argv[3]))
a = c["args"]; a[a.index("--max-context") + 1] = ctx
json.dump(c, open(out, "w"), indent=2)
EOF
  LABEL=maya-deep-$CTX
  LOG=$BENCH/runs/logs/$LABEL.log
  echo "=== $LABEL $(date +%T) ==="
  cd $P
  T0=$(date +%s)
  setsid ./.venv/bin/python maya.py --config $CFG --port $PORT > "$LOG" 2>&1 &
  PID=$!
  OK=0
  for i in $(seq 1 240); do
    sleep 5
    curl -sf -m 4 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { OK=1; break; }
    kill -0 $PID 2>/dev/null || break
  done
  LOAD=$(( $(date +%s) - T0 ))
  RC=0
  if [ "$OK" = 1 ]; then
    echo "ready after ${LOAD}s: $(grep -h '^ready' "$LOG" | cut -c1-140)"
    FILL=$(( CTX * 9 / 10 ))
    python3 $SCRIPTS/deep-bench.py --url http://127.0.0.1:$PORT --model glm-5.3-flash --tokens $FILL \
      --label $LABEL --out $OUT || RC=$?
  else
    echo "NO HEALTH $LABEL after ${LOAD}s"; tail -6 "$LOG" | cut -c1-240
    echo "$LABEL,loadfail,1,,,,,,,see $LOG" >> $OUT
    RC=3
  fi
  kill -TERM $PID 2>/dev/null
  pkill -TERM -P $PID 2>/dev/null
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$u" -lt 2000 ] && break; sleep 3
  done
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$u" -gt 2000 ] && { echo "GPUs still hold ${u} MiB after kill, stopping ladder"; exit 4; }
  [ "$RC" != 0 ] && { echo "tier $CTX failed (rc=$RC), stopping ladder"; break; }
done
echo "LADDER DONE $(date +%T)"
