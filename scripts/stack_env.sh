# Sourced by launch.sh and sweep_workers.sh (cwd = repo root): the opt-in stack variants as a command prefix.
#   RHS1_SRC=~/railhead-system1-batch/src  rhs1 source with the pipelined site search (until its PR merges)
#   ENGINE=m4lto-poll                      data/engines/<name>/build/openttd, or a path to an engine binary
#   POOL=1                                 label stages reuse OpenTTD servers (src/rh17m/pool.py, RH17M_POOL=1)
# Sets PRE (env prefix for every stage), POOLPRE (label stages only) and R (the rh17m command).
PRE=""
POOLPRE=""
[[ ${POOL:-0} == 1 ]] && POOLPRE="RH17M_POOL=1 "
[[ -n ${RHS1_SRC:-} ]] && PRE+="PYTHONPATH=${RHS1_SRC:A} "
if [[ -n ${ENGINE:-} ]]; then
  ENGINE_BIN=$PWD/data/engines/$ENGINE/build/openttd
  [[ -x $ENGINE_BIN ]] || ENGINE_BIN=${ENGINE:A}
  [[ -x $ENGINE_BIN && -f $ENGINE_BIN ]] || { echo "no engine variant $ENGINE"; exit 1; }
  PRE+="RAILHEAD_OPENTTD=$ENGINE_BIN "
fi
R="${PRE}taskpolicy -a .venv/bin/rh17m"  # app resource policy: P-cores (see src/rh17m/sched.py)
# say which rhs1 and engine the stages will use: a silent fallback to the sequential builder measures nothing
echo "rhs1:   $(env ${=PRE} .venv/bin/python -c "import rhs1.builders.air as a; print(a.__file__, 'pipelined' if hasattr(a, '_dry_runs') else 'sequential')")"
echo "engine: ${ENGINE:-stock}"
echo "pool:   ${POOL:-0}"
