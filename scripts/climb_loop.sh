#!/usr/bin/env zsh
# Keep hillclimbing as labels arrive: whenever the usable examples grow by GROWTH (default 25%) over the last
# climb, run a climb pass from the incumbent, then publish the incumbent (Hub, if logged in) and push. Runs
# until data/labels/.stop exists. Every pass is a group in Trackio (project railhead-17m).
set -u
cd "${0:A:h}/.."
R=.venv/bin/rh17m
GROWTH=${GROWTH:-1.25}
MIN_VAL=${MIN_VAL:-40}
last=0
count() { .venv/bin/python -I -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from rh17m.dataset import DatasetConfig, load_examples
e = load_examples(Path('data/labels'), DatasetConfig())
print(len(e), sum(x['split'] == 'val' for x in e))"; }
while [[ ! -f data/labels/.stop ]]; do
  read n val <<< "$(count)"
  if (( val >= MIN_VAL )) && (( n >= last * GROWTH )); then
    echo "$(date +%H:%M) climb pass on $n examples ($val val)" | tee -a data/logs/climb_loop.log
    if [[ -f runs/climb/incumbent.json ]]; then
      $R climb --from-incumbent --noise-seeds 2 --max-runs ${MAX_RUNS:-30} >> data/logs/climb.log 2>&1
    else
      $R climb --noise-seeds 3 --max-runs ${MAX_RUNS:-30} >> data/logs/climb.log 2>&1
    fi
    inc=$(.venv/bin/python -I -c "import json; print(json.load(open('runs/climb/incumbent.json'))['model_dir'])" 2>/dev/null)
    if [[ -n "$inc" ]]; then
      $R publish model --run "${inc:h}" >> data/logs/publish.log 2>&1 || echo "$(date +%H:%M) publish skipped (hf auth?)" >> data/logs/climb_loop.log
    fi
    last=$n
  fi
  sleep 600
done
