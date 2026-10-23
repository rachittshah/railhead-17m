#!/usr/bin/env zsh
# Harvest decision states (teacher replays, task starts, self-play) and simulator-label them as they appear.
# Resumable: every stage skips work already on disk. Logs go to data/logs/.
set -u
cd "${0:A:h}/.."
R=.venv/bin/rh17m
GAMES=${GAMES:-2}
SP_WORKERS=${SP_WORKERS:-4}
LABEL_WORKERS=${LABEL_WORKERS:-8}
mkdir -p data/logs data/states data/labels
rm -f data/states/.harvest-done
(
  $R states teacher >> data/logs/teacher.log 2>&1
  $R states start --tasks ${(f)"$(ls ~/railhead/tasks | grep -E '^arena-(train|dev)-02')"} >> data/logs/start.log 2>&1
  $R states selfplay --games $GAMES --workers $SP_WORKERS >> data/logs/selfplay.log 2>&1
  touch data/states/.harvest-done
) &
while true; do
  $R label --workers $LABEL_WORKERS >> data/logs/label.log 2>&1
  if [[ -f data/states/.harvest-done ]]; then
    $R label --workers $LABEL_WORKERS >> data/logs/label.log 2>&1
    break
  fi
  sleep 30
done
wait
touch data/labels/.label-done
