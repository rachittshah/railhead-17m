#!/usr/bin/env zsh
# Worker-count sweep on a fixed workload: test-split states labelled with ONE seed (the cost of a train state)
# into data/labels-bench, so data/labels keeps its split and seed conventions and the sweep never runs out of
# states (1,986 test states are unlabelled). Per worker count: relaunch the labeller, let it warm up WARM
# minutes, then record `rh17m bench labels --since-min WINDOW` as one line of runs/sweep/<name>.jsonl.
#
#   POOL=1 RHS1_SRC=~/railhead-system1-batch/src ENGINE=m4lto-poll scripts/sweep_workers.sh full 14 22 30
#   env: WARM (default 3) and WINDOW (default 10) minutes; the same opt-ins as launch.sh (stack_env.sh)
# Tracing is off so the numbers do not include the exporter. Stops everything first (scripts/stop.sh).
set -u
cd "${0:A:h}/.."
source "${0:A:h}/stack_env.sh"
NAME=${1:?usage: sweep_workers.sh <name> <workers> [<workers> ...]}
shift
WARM=${WARM:-3}
WINDOW=${WINDOW:-10}
OUT=data/labels-bench
mkdir -p runs/sweep data/logs $OUT
for n in "$@"; do
  scripts/stop.sh > /dev/null
  echo "$(date +%H:%M) $n workers: warm $WARM min, window $WINDOW min"
  env RH17M_TRACE=0 RH17M_STAGE=sweep ${=POOLPRE} ${=R} label --workers $n --splits test --seeds 1 \
    --out $OUT --order-seed $RANDOM >> data/logs/sweep.log 2>&1 &
  sleep $(( 60 * (WARM + WINDOW) ))
  win=$(env ${=R} bench labels --labels $OUT --since-min $WINDOW | tail -1)
  echo "{\"name\":\"$NAME\",\"workers\":$n,\"warm_min\":$WARM,\"window_min\":$WINDOW,\"pool\":${POOL:-0},\"engine\":\"${ENGINE:-stock}\",\"pipelined_rhs1\":\"${RHS1_SRC:+yes}\",\"window\":$win}" \
    >> runs/sweep/$NAME.jsonl
  tail -1 runs/sweep/$NAME.jsonl | cut -c1-260
done
scripts/stop.sh > /dev/null
