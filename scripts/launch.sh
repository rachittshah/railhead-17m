#!/usr/bin/env zsh
# Launch every railhead-17m background stage in tmux session `rh17m` (one window per stage). Every stage is
# resumable, so relaunching after a stop or a crash loses only in-flight work. Stop with scripts/stop.sh.
set -u
cd "${0:A:h}/.."
S=rh17m
R="taskpolicy -a .venv/bin/rh17m"  # app resource policy: P-cores (see src/rh17m/sched.py)
tmux has-session -t $S 2>/dev/null && { echo "session $S already running"; exit 1; }
DEV=$(ls ~/railhead/tasks | grep -E '^arena-dev-02' | tr '\n' ' ')
mkdir -p data/logs
loop() { echo "cd $PWD && while true; do $1; [ -f data/labels/.stop ] && break; sleep 120; done"; }
tmux new-session -d -s $S -n harvest "cd $PWD && [ -f data/states/.harvest-done ] && exit 0; $R states teacher >> data/logs/teacher.log 2>&1; \
  $R states hub-teacher >> data/logs/hub-teacher.log 2>&1; \
  $R states start --tasks $(ls ~/railhead/tasks | grep -E '^arena-(train|dev)-02' | tr '\n' ' ') >> data/logs/start.log 2>&1; \
  $R states selfplay --games 2 --workers 4 >> data/logs/selfplay.log 2>&1; \
  $R states selfplay --tasks hub --policy mix --games 1 --workers 2 >> data/logs/hub-selfplay.log 2>&1; \
  touch data/states/.harvest-done"
tmux new-window -d -t $S -n label-train "$(loop "$R label --workers ${TRAIN_WORKERS:-11} --splits train --order-seed 3 >> data/logs/label-train.log 2>&1")"
tmux new-window -d -t $S -n label-val "$(loop "$R label --workers ${VAL_WORKERS:-8} --splits val --order-seed 2 >> data/logs/label-val.log 2>&1")"
tmux new-window -d -t $S -n label-test "$(loop "$R label --workers ${TEST_WORKERS:-3} --splits test,ood --order-seed 4 >> data/logs/label-test.log 2>&1")"
tmux new-window -d -t $S -n climb "cd $PWD && scripts/climb_loop.sh"
tmux new-window -d -t $S -n watch "cd $PWD && $R watch --interval 120 >> data/logs/watch.log 2>&1"
tmux new-window -d -t $S -n trackio "cd $PWD && .venv/bin/trackio show --project railhead-17m --no-footer >> data/logs/trackio.log 2>&1"
tmux list-windows -t $S
