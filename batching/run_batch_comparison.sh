#!/bin/bash
# Batched versus unbatched, every kernel and precision, on one GPU node.
#
#   bash run_batch_comparison.sh <host> <idle_w>
#
# Answers one question: does giving the GPU more work per call change energy per
# sample? Runs each kernel twice at each precision, once unbatched, once at the
# largest batch that fits, so the only difference between the two arms is the
# batch size. One run per configuration, 12 measurements in total; this is a
# comparison, not a headline result, and the 20-round data already establishes
# that run-to-run variation is under 1%.
#
# GPU only. The IPU is not batched here: its operands are already resident and
# the per-tile memory limit makes large batches infeasible.
#
# Expects the scripts flat in ~/scripts, as run_one_measurement.sh does.
# Writes to ~/scripts/batchcmp/, never to official/.

set -uo pipefail
HOST=$1; IDLE=$2
SECS=${SECS:-300}
COOL=${COOL:-60}
MAXWAIT=${MAXWAIT:-900}
ME=$(id -un)

SELF=$(hostname -s 2>/dev/null || hostname | cut -d. -f1)
if [ "$HOST" != "$SELF" ]; then
  echo "refusing: host argument '$HOST' does not match this machine ('$SELF')." >&2
  exit 2
fi

cd ~/scripts || exit 1
mkdir -p batchcmp probe
PROG=batchcmp/progress.log
MAXB=batchcmp/max_batch_${HOST}.txt

LOCK="$HOME/.measure_${HOST}.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "refusing: another measurement holds $LOCK" >&2; exit 3; }

# stencil and triad batch through the --b flag on the standard scripts; GEMM
# needs the batched variant, which uses torch.bmm for B>1 and plain matmul at
# B=1 so the unbatched arm is identical to the standard script.
script_for() {
  case "$1" in
    stencil) echo "gpu_stencil.py" ;;
    triad)   echo "gpu_triad.py" ;;
    gemm)    echo "gpu_gemm_batched.py" ;;
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

# Largest batch that actually runs. Probes the real scripts for 3 s rather than
# allocating operands in isolation: the scripts hold input and output resident
# while the kernel allocates its own result, so a batch that fits an idealised
# allocation can still exhaust memory in production.
find_max_batch() {
  local KERN=$1 DT=$2; shift 2
  local S; S=$(script_for "$KERN")
  for B in "$@"; do
    rm -f probe/RUN
    if timeout 300 python3 -u "$S" --run 0 --b "$B" --n "$(size_for "$KERN")" \
         --dtype "$DT" --seconds 3 --idle-w "$IDLE" \
         --marker probe/RUN --out-dir probe > "probe/${KERN}${DT}_B${B}.log" 2>&1 \
       && grep -q "throughput" "probe/${KERN}${DT}_B${B}.log"; then
      echo "$B"; return 0
    fi
  done
  echo 1
}

size_for() {
  case "$1" in
    stencil) echo 384 ;;
    triad)   echo 68000000 ;;
    gemm)    echo 4096 ;;
  esac
}

measure() {  # kernel dtype batch
  local KERN=$1 DT=$2 B=$3
  local S; S=$(script_for "$KERN")
  local TAG="${KERN}${DT}B${B}"
  local PWR="batchcmp/${TAG}_${HOST}_power.csv"
  local SESS="bcmp_${TAG}"

  echo "[$(date +%H:%M:%S)] $KERN $DT B=$B starting" >> $PROG
  wait_free || echo "[$(date +%H:%M:%S)] $KERN $DT B=$B CONTENDED" >> $PROG

  rm -f batchcmp/RUNNING
  tmux kill-session -t "$SESS" 2>/dev/null
  tmux new-session -d -s "$SESS" \
    "cd ~/scripts && python3 -u gpu_power_sampler.py --out $PWR \
     --wait-for batchcmp/RUNNING --interval 1.0 --idle-w $IDLE > /dev/null 2>&1"
  sleep 4

  # --run carries the batch size so the two arms land in distinct files
  python3 -u "$S" --run "$B" --b "$B" --n "$(size_for "$KERN")" --dtype "$DT" \
      --seconds "$SECS" --idle-w "$IDLE" --marker batchcmp/RUNNING \
      --out-dir batchcmp > "batchcmp/${TAG}_${HOST}.log" 2>&1
  local RC=$?

  tmux send-keys -t "$SESS" C-c 2>/dev/null; sleep 4
  tmux kill-session -t "$SESS" 2>/dev/null

  local TP; TP=$(grep -oE "throughput [0-9.]+" "batchcmp/${TAG}_${HOST}.log" | head -1)
  echo "[$(date +%H:%M:%S)] $KERN $DT B=$B rc=$RC $TP" >> $PROG
  [ "$RC" -ne 0 ] && echo "[$(date +%H:%M:%S)] WARN $KERN $DT B=$B rc=$RC" >> $PROG
  sleep "$COOL"
}

echo "=== batch comparison $(date -Iseconds) host=$HOST secs=$SECS ===" >> $PROG
: > "$MAXB"

for KERN in stencil triad gemm; do
  for DT in fp32 fp16; do
    case "$KERN$DT" in
      stencilfp32) CAND="64 32 16 8" ;;
      stencilfp16) CAND="128 64 32 16" ;;
      triadfp32)   CAND="64 32 16 8" ;;
      triadfp16)   CAND="128 64 32 16" ;;
      gemmfp32)    CAND="128 64 32 16" ;;
      gemmfp16)    CAND="256 128 64 32" ;;
    esac
    BMAX=$(find_max_batch "$KERN" "$DT" $CAND)
    echo "$KERN $DT $BMAX" >> "$MAXB"
    echo "[$(date +%H:%M:%S)] $KERN $DT max batch $BMAX" >> $PROG

    measure "$KERN" "$DT" 1
    [ "$BMAX" != "1" ] && measure "$KERN" "$DT" "$BMAX"
  done
done

rm -rf probe
echo "=== finished $(date -Iseconds) ===" >> $PROG
echo "max batch sizes:"; cat "$MAXB"
