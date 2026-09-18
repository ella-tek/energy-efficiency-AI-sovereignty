#!/bin/bash
# Chains the batched rounds behind anything currently running on the node.
#
#   bash batched_rounds_supervisor.sh <host> <idle_w> <first_run> <last_run>
#
# Launch detached in tmux. Nothing runs on the client machine, so the session
# survives disconnection.

set -uo pipefail
HOST=$1; IDLE=$2; FIRST=$3; LAST=$4
cd ~/scripts || exit 1
mkdir -p batchedrounds
PROG=batchedrounds/progress.log

echo "[$(date +%H:%M:%S)] supervisor armed on $HOST for runs $FIRST..$LAST" >> $PROG

# not pgrep -f: a tmux server carries the script name in its own command line
# forever, so pgrep would match tmux and wait indefinitely
W=0
while ps -u "$(id -un)" -o cmd --no-headers \
    | grep -v '^tmux' | grep -qE 'run_one_measurement\.sh|run_batch_comparison\.sh'; do
  sleep 30; W=$((W+30))
done
[ "$W" -gt 0 ] && echo "[$(date +%H:%M:%S)] node clear after ${W}s" >> $PROG
sleep 60

exec bash run_batched_rounds.sh "$HOST" "$IDLE" "$FIRST" "$LAST"
