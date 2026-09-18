#!/bin/bash
# Repeated measurements at the largest batch that fits, for every kernel and
# precision, on one GPU node.
#
#   bash run_batched_rounds.sh <host> <idle_w> <first_run> <last_run>
#
# e.g.  bash run_batched_rounds.sh cricket 43.14 1 20
#
# Finds the largest workable batch once per kernel and precision, then measures
# at that batch for as many runs as asked. Same protocol as the unbatched runs:
# 300 s per measurement, 10 warm-up calls and compile excluded from both the
# timing and the power trace, 60 s between measurements and 300 s between rounds.
#
# The unbatched arm already has its repeats in official/, so only the batched
# arm is measured here. Results go to batchedrounds/.
#
# GPU only. Launch inside tmux and leave alone; nothing runs on the client.

set -uo pipefail
HOST=$1; IDLE=$2; FIRST=$3; LAST=${4:-$FIRST}
SECS=${SECS:-300}
COOL=${COOL:-60}
ROUND_COOL=${ROUND_COOL:-300}
MAXWAIT=${MAXWAIT:-900}
ME=$(id -un)

SELF=$(hostname -s 2>/dev/null || hostname | cut -d. -f1)
[ "$HOST" = "$SELF" ] || { echo "refusing: '$HOST' is not this machine ('$SELF')" >&2; exit 2; }

cd ~/scripts || exit 1
mkdir -p batchedrounds probe
PROG=batchedrounds/progress.log
MAXB=batchedrounds/max_batch_${HOST}.txt

LOCK="$HOME/.measure_${HOST}.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "refusing: another measurement holds $LOCK" >&2; exit 3; }

script_for() {
  case "$1" in
    stencil) echo "gpu_stencil.py" ;;
    triad)   echo "gpu_triad.py" ;;
    gemm)    echo "gpu_gemm_batched.py" ;;
  esac
}
size_for() {
  case "$1" in
    stencil) echo 384 ;;
    triad)   echo 68000000 ;;
    gemm)    echo 4096 ;;
  esac
}
candidates_for() {
  case "$1$2" in
    stencilfp32) echo "64 32 16 8" ;;
    stencilfp16) echo "128 64 32 16" ;;
    triadfp32)   echo "64 32 16 8" ;;
    triadfp16)   echo "128 64 32 16" ;;
    gemmfp32)    echo "128 64 32 16" ;;
    gemmfp16)    echo "256 128 64 32" ;;
  esac
}

foreign() {
  local pid owner
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$owner" ] && continue
    [ "$owner" = "$ME" ] && continue
    echo "$owner:$pid"
  done
}
wait_free() {
  local w=0 o
  while [ $w -lt $MAXWAIT ]; do
    o=$(foreign | tr '\n' ' ')
    [ -z "${o// /}" ] && return 0
    echo "[$(date +%H:%M:%S)] waiting on other users: $o" >> $PROG
    sleep 60; w=$((w+60))
  done
  echo "[$(date +%H:%M:%S)] CONTENDED after ${MAXWAIT}s" >> $PROG
  return 1
}

# largest batch the real script survives, probed at 3 s a go
find_max_batch() {
  local KERN=$1 DT=$2 S N B
  S=$(script_for "$KERN"); N=$(size_for "$KERN")
  for B in $(candidates_for "$KERN" "$DT"); do
    rm -f probe/RUN
    if timeout 300 python3 -u "$S" --run 0 --b "$B" --n "$N" --dtype "$DT" \
         --seconds 3 --idle-w "$IDLE" --marker probe/RUN --out-dir probe \
         > "probe/${KERN}${DT}_B${B}.log" 2>&1 \
       && grep -q throughput "probe/${KERN}${DT}_B${B}.log"; then
      echo "$B"; return 0
    fi
  done
  echo 1
}

measure() {  # kernel dtype batch run
  local KERN=$1 DT=$2 B=$3 RUN=$4
  local S; S=$(script_for "$KERN")
  local TAG="${KERN}${DT}B${B}"
  local PWR="batchedrounds/${TAG}_${HOST}_run${RUN}_power.csv"
  local SESS="br_${TAG}_r${RUN}"

  echo "[$(date +%H:%M:%S)] run$RUN $KERN $DT B=$B starting" >> $PROG
  wait_free || echo "[$(date +%H:%M:%S)] run$RUN $KERN $DT CONTENDED" >> $PROG

  rm -f batchedrounds/RUNNING
  tmux kill-session -t "$SESS" 2>/dev/null
  tmux new-session -d -s "$SESS" \
    "cd ~/scripts && python3 -u gpu_power_sampler.py --out $PWR \
     --wait-for batchedrounds/RUNNING --interval 1.0 --idle-w $IDLE > /dev/null 2>&1"
  sleep 4

  python3 -u "$S" --run "$RUN" --b "$B" --n "$(size_for "$KERN")" --dtype "$DT" \
      --seconds "$SECS" --idle-w "$IDLE" --marker batchedrounds/RUNNING \
      --out-dir batchedrounds > "batchedrounds/${TAG}_${HOST}_run${RUN}.log" 2>&1
  local RC=$?

  tmux send-keys -t "$SESS" C-c 2>/dev/null; sleep 4
  tmux kill-session -t "$SESS" 2>/dev/null

  local TP; TP=$(grep -oE "throughput [0-9.]+" "batchedrounds/${TAG}_${HOST}_run${RUN}.log" | head -1)
  echo "[$(date +%H:%M:%S)] run$RUN $KERN $DT B=$B rc=$RC $TP" >> $PROG
  [ "$RC" -ne 0 ] && echo "[$(date +%H:%M:%S)] WARN run$RUN $KERN $DT rc=$RC" >> $PROG

  if [ -f "$PWR" ]; then
    local ROWS=$(( $(wc -l < "$PWR") - 1 ))
    [ "$ROWS" -lt $(( SECS * 8 / 10 )) ] && \
      echo "[$(date +%H:%M:%S)] WARN run$RUN $KERN $DT power trace short: $ROWS" >> $PROG
  else
    echo "[$(date +%H:%M:%S)] WARN run$RUN $KERN $DT NO POWER CSV" >> $PROG
  fi
  sleep "$COOL"
}

echo "=== batched rounds start $(date -Iseconds) host=$HOST runs $FIRST..$LAST ===" >> $PROG

# sweep once; the answer does not change between rounds
: > "$MAXB"
for KERN in stencil triad gemm; do
  for DT in fp32 fp16; do
    B=$(find_max_batch "$KERN" "$DT")
    echo "$KERN $DT $B" >> "$MAXB"
    echo "[$(date +%H:%M:%S)] max batch $KERN $DT = $B" >> $PROG
  done
done
rm -rf probe
echo "max batch sizes:"; cat "$MAXB"

for RUN in $(seq "$FIRST" "$LAST"); do
  for KERN in stencil triad gemm; do
    for DT in fp32 fp16; do
      B=$(awk -v k="$KERN" -v d="$DT" '$1==k && $2==d {print $3}' "$MAXB")
      [ -z "$B" ] && B=1
      measure "$KERN" "$DT" "$B" "$RUN"
    done
  done
  echo "[$(date +%H:%M:%S)] === run $RUN complete, settling ${ROUND_COOL}s ===" >> $PROG
  sleep "$ROUND_COOL"
done
echo "=== batched rounds finished $(date -Iseconds) ===" >> $PROG
