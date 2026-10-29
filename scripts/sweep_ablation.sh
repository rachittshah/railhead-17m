#!/usr/bin/env zsh
# The CPU-optimisation ablation: the same fixed workload (sweep_workers.sh) under each layer of the stack, 14
# workers, then the full stack at higher worker counts. One window per line of runs/sweep/ablation.jsonl.
#   base      stock engine, rhs1 main (one round trip per dry-run), one OpenTTD process per fork/rollout
#   pipe      + pipelined site search (RHS1_SRC)
#   pipepool  + persistent server pool (POOL=1)
#   full      + faster engine (ENGINE): the whole stack; 14 again at the end to see drift, then 22 and 30
# usage: RHS1_SRC=~/railhead-system1-batch/src ENGINE=pr-check/src scripts/sweep_ablation.sh
set -u
cd "${0:A:h}/.."
: ${RHS1_SRC:?set RHS1_SRC}
: ${ENGINE:?set ENGINE}
S=scripts/sweep_workers.sh
env -u RHS1_SRC -u ENGINE -u POOL $S ablation 14
env -u ENGINE -u POOL $S ablation 14
env -u ENGINE POOL=1 $S ablation 14
env POOL=1 $S ablation 14 22 30
