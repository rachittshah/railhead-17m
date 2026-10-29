#!/usr/bin/env zsh
# Launch every railhead-17m background stage in tmux session `rh17m` (one window per stage). Every stage is
# resumable, so relaunching after a stop or a crash loses only in-flight work. Stop with scripts/stop.sh.
set -u
cd "${0:A:h}/.."
S=rh17m
# Opt-in variants, applied to every stage through its command prefix (no shared tmux environment needed):
#   RHS1_SRC=~/railhead-system1-batch/src  rhs1 source with the pipelined site search (until its PR merges)
#   ENGINE=m4lto                           data/engines/<name>/build/openttd (scripts/build_engine_variant.sh)
#   POOL=1                                 label stages reuse OpenTTD servers (src/rh17m/pool.py, RH17M_POOL=1)
PRE=""
POOLPRE=""
[[ ${POOL:-0} == 1 ]] && POOLPRE="RH17M_POOL=1 "
[[ -n ${RHS1_SRC:-} ]] && PRE+="PYTHONPATH=${RHS1_SRC:A} "
if [[ -n ${ENGINE:-} ]]; then
  [[ -x $PWD/data/engines/$ENGINE/build/openttd ]] || { echo "no engine variant $ENGINE"; exit 1; }
  PRE+="RAILHEAD_OPENTTD=$PWD/data/engines/$ENGINE/build/openttd "
fi
R="${PRE}taskpolicy -a .venv/bin/rh17m"  # app resource policy: P-cores (see src/rh17m/sched.py)
tmux has-session -t $S 2>/dev/null && { echo "session $S already running"; exit 1; }
DEV=$(ls ~/railhead/tasks | grep -E '^arena-dev-02' | tr '\n' ' ')
mkdir -p data/logs data/phoenix
# say which rhs1 and engine the stages will use: a silent fallback to the sequential builder measures nothing
echo "rhs1:   $(env ${=PRE} .venv/bin/python -c "import rhs1.builders.air as a; print(a.__file__, 'pipelined' if hasattr(a, '_dry_runs') else 'sequential')")"
echo "engine: ${ENGINE:-stock}"
echo "pool:   ${POOL:-0}"
loop() { echo "cd $PWD && while true; do $1; [ -f data/labels/.stop ] && break; sleep 120; done"; }
# Trace viewer first (Arize Phoenix, `uv tool install arize-phoenix`): every stage exports OTLP spans to it
# (src/rh17m/tracing.py); open http://127.0.0.1:6006 for the waterfalls. Stages started before it answers
# re-probe it every minute.
tmux new-session -d -s $S -n traces "cd $PWD && PHOENIX_PORT=6006 PHOENIX_HOST=127.0.0.1 PHOENIX_WORKING_DIR=$PWD/data/phoenix \
  ~/.local/bin/phoenix serve >> data/logs/phoenix.log 2>&1"
for _ in {1..90}; do curl -s -o /dev/null http://127.0.0.1:6006/ && break; sleep 2; done
tmux new-window -d -t $S -n harvest "cd $PWD && [ -f data/states/.harvest-done ] && exit 0; $R states teacher >> data/logs/teacher.log 2>&1; \
  $R states hub-teacher >> data/logs/hub-teacher.log 2>&1; \
  $R states start --tasks $(ls ~/railhead/tasks | grep -E '^arena-(train|dev)-02' | tr '\n' ' ') >> data/logs/start.log 2>&1; \
  $R states selfplay --games 2 --workers 4 >> data/logs/selfplay.log 2>&1; \
  $R states selfplay --tasks hub --policy mix --games 1 --workers 2 >> data/logs/hub-selfplay.log 2>&1; \
  touch data/states/.harvest-done"
tmux new-window -d -t $S -n label-train "$(loop "RH17M_STAGE=label-train ${POOLPRE}$R label --workers ${TRAIN_WORKERS:-11} --splits train --order-seed 3 >> data/logs/label-train.log 2>&1")"
tmux new-window -d -t $S -n label-val "$(loop "RH17M_STAGE=label-val ${POOLPRE}$R label --workers ${VAL_WORKERS:-8} --splits val --order-seed 2 >> data/logs/label-val.log 2>&1")"
tmux new-window -d -t $S -n label-test "$(loop "RH17M_STAGE=label-test ${POOLPRE}$R label --workers ${TEST_WORKERS:-3} --splits test,ood --order-seed 4 >> data/logs/label-test.log 2>&1")"
[[ ${CLIMB:-1} == 1 ]] && tmux new-window -d -t $S -n climb "cd $PWD && ${PRE}scripts/climb_loop.sh"  # CLIMB=0: no GPU stage
tmux new-window -d -t $S -n watch "cd $PWD && $R watch --interval 120 >> data/logs/watch.log 2>&1"
tmux new-window -d -t $S -n trackio "cd $PWD && .venv/bin/trackio show --project railhead-17m --no-footer >> data/logs/trackio.log 2>&1"
tmux list-windows -t $S
