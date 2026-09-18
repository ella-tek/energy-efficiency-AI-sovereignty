#!/bin/bash
# Run ONE measurement: one kernel, one precision, one node.
#
#   bash run_one_measurement.sh <host> <idle_w> <ipu|gpu> <kernel> <n> <dtype> <run>
#
# e.g.  bash run_one_measurement.sh cricket    43.14 gpu stencil 384      fp32 1
#       bash run_one_measurement.sh locust     89.41 gpu triad   68000000 fp16 1
#       bash run_one_measurement.sh mandelbrot 41.6  ipu gemm    4096     fp32 1
#
# Starts the power sampler in a tmux session, runs the kernel, stops the
# sampler. Waits for other users to clear first and takes a per-node lock.
#
# Writes <kernel><dtype>_<host>_run<N>_log.json and a matching _power.csv
# to official/.

set -uo pipefail
HOST=$1; IDLE=$2; KIND=$3; KERN=$4; N=$5; DT=$6; RUN=$7
SECS=${SECS:-300}
MAXWAIT=${MAXWAIT:-900}
ME=$(id -un)

# the kernel names its output from hostname, so a mismatch would split the
# power trace and the timing record across two names
SELF=$(hostname -s 2>/dev/null || hostname | cut -d. -f1)
if [ "$HOST" != "$SELF" ]; then
  echo "refusing: host argument '$HOST' does not match this machine ('$SELF')." >&2
  echo "The power trace and the timing record would be written under different" >&2
  echo "names and the measurement would be dropped by the analysis." >&2
  exit 2
fi

cd ~/scripts || exit 1
mkdir -p official
PROG=official/measurement_log.txt

LOCK="$HOME/.measure_${HOST}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date +%H:%M:%S)] REFUSING: $LOCK held by another measurement" >> $PROG
  exit 3
fi

if [ "$KIND" = ipu ]; then
  SAMP="python3 -u ipu_power_sampler.py"
  case "$KERN" in
    stencil) CMD="GCDA_MONITOR=1 python3 -u ipu_stencil.py --device-iter 1" ;;
    triad)   CMD="GCDA_MONITOR=1 python3 -u ipu_triad.py   --device-iter 1" ;;
    gemm)    CMD="GCDA_MONITOR=1 python3 -u ipu_gemm.py    --device-iter 1" ;;
    *) echo "unknown kernel $KERN" >&2; exit 2 ;;
  esac
else
  SAMP="python3 -u gpu_power_sampler.py"
  case "$KERN" in
    stencil) CMD="python3 -u gpu_stencil.py --b 1" ;;
    triad)   CMD="python3 -u gpu_triad.py   --b 1" ;;
    gemm)    CMD="python3 -u gpu_gemm.py" ;;
    *) echo "unknown kernel $KERN" >&2; exit 2 ;;
  esac
fi

foreign_occupants() {
  if [ "$KIND" = ipu ]; then
    gc-monitor --no-card-info 2>/dev/null | awk -F'|' -v me="$ME" '
      /^\| *[0-9]+ *\|/ {
        pid=$2; usr=$5; gsub(/ /,"",pid); gsub(/ /,"",usr)
        if (usr != "" && usr != me) print usr ":" pid
      }'
  else
    local pid owner
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
      [ -z "$owner" ] && continue
      [ "$owner" = "$ME" ] && continue
      echo "$owner:$pid"
    done
  fi
}

waited=0
while [ $waited -lt $MAXWAIT ]; do
  occ=$(foreign_occupants | tr '\n' ' ')
  [ -z "${occ// /}" ] && break
  echo "[$(date +%H:%M:%S)] waiting on other users: $occ (${waited}/${MAXWAIT}s)" >> $PROG
  sleep 60; waited=$((waited+60))
done

TAG="${KERN}${DT}"
if [ "$DT" = fp32 ]; then TAG="$KERN"; fi
PWRCSV="official/${TAG}_${HOST}_run${RUN}_power.csv"
SESS="redo_${TAG}_r${RUN}"

echo "[$(date +%H:%M:%S)] run$RUN $KERN $DT starting" >> $PROG

rm -f official/RUNNING
tmux kill-session -t "$SESS" 2>/dev/null
tmux new-session -d -s "$SESS" \
  "cd ~/scripts && $SAMP --out $PWRCSV \
   --wait-for official/RUNNING --interval 1.0 --idle-w $IDLE > /dev/null 2>&1"
sleep 4

eval "$CMD --run $RUN --n $N --dtype $DT --seconds $SECS \
      --idle-w $IDLE --marker official/RUNNING --out-dir official" \
      > official/${TAG}_${HOST}_run${RUN}.log 2>&1
RC=$?

tmux send-keys -t "$SESS" C-c 2>/dev/null; sleep 4
tmux kill-session -t "$SESS" 2>/dev/null

TP=$(grep -oE "throughput [0-9.]+" official/${TAG}_${HOST}_run${RUN}.log | head -1)
echo "[$(date +%H:%M:%S)] run$RUN $KERN $DT rc=$RC $TP" >> $PROG

if [ -f "$PWRCSV" ]; then
  ROWS=$(( $(wc -l < "$PWRCSV") - 1 ))
  MIN=$(( SECS * 8 / 10 ))
  [ "$ROWS" -lt "$MIN" ] && echo "[$(date +%H:%M:%S)] WARN power trace short: $ROWS rows" >> $PROG
else
  echo "[$(date +%H:%M:%S)] WARN NO POWER CSV WRITTEN" >> $PROG
fi
[ "$RC" -ne 0 ] && echo "[$(date +%H:%M:%S)] WARN nonzero rc=$RC" >> $PROG
exit $RC
