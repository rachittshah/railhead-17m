#!/usr/bin/env zsh
# Keep hillclimbing as labels arrive: whenever the usable examples grow by GROWTH (default 25%) over the last
# climb, run a climb pass from the incumbent, then publish the incumbent (Hub, if logged in) and push. Runs
# until data/labels/.stop exists. Every pass is a group in Trackio (project railhead-17m).
set -u
cd "${0:A:h}/.."
R=.venv/bin/rh17m
GROWTH=${GROWTH:-1.25}
MIN_VAL=${MIN_VAL:-40}
last=$(cat runs/climb/last_n 2>/dev/null || echo 0)  # persisted, so a restart does not redo a pass
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
      run="${inc:h}"
      # Closed loop on the held-out dev maps, paired with the scripted baselines in runs/play/baselines-w180.
      nice -n 5 $R play --tasks $(ls ~/railhead/tasks | grep -E '^arena-dev-02' | tr '\n' ' ') --policies model \
        --model "$inc" --seeds 1 --wake-days 180 --workers 2 --out "runs/play/${run:t}" >> data/logs/play-model.log 2>&1
      $R play-report --track >> data/logs/climb_loop.log 2>&1
      git add runs/play/summary.json && git commit -q -m "Closed loop: ${run:t} on the dev maps" && git push -q
      $R publish model --run "$run" >> data/logs/publish.log 2>&1 || echo "$(date +%H:%M) publish skipped (hf auth?)" >> data/logs/climb_loop.log
      # DAgger: the incumbent plays every train map once (epsilon-greedy); the train labeller takes these first.
      echo "$(date +%H:%M) dagger self-play with ${run:t}" >> data/logs/climb_loop.log
      nice -n 5 $R states selfplay --tasks train --policy model --model "$inc" --games 1 --max-states 12 \
        --workers 3 --seed0 $(( $(date +%s) % 100000 )) >> data/logs/dagger.log 2>&1
    fi
    last=$n
    echo $n > runs/climb/last_n
  fi
  sleep 600
done
