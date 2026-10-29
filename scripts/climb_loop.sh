#!/usr/bin/env zsh
# Keep hillclimbing as labels arrive: whenever the usable examples grow by GROWTH (default 25%) over the last
# climb, run a climb pass from the incumbent, then publish the incumbent (Hub, if logged in) and push. Runs
# until data/labels/.stop exists. Every pass is a group in Trackio (project railhead-17m) and one trace
# (climb_pass, every rh17m command of the pass nested under it via TRACEPARENT) in the Phoenix viewer.
set -u
cd "${0:A:h}/.."
R=(taskpolicy -a .venv/bin/rh17m)  # app resource policy: P-cores (see src/rh17m/sched.py)
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
    eval "$(.venv/bin/rh17m trace begin)"
    # A fresh Bekko-68M teacher on the current labels (default settings), for the @teacher moves.
    tname="teacher-68m-$(date +%m%d-%H%M)"
    $R train --name "$tname" --set init='"bekko-68m"' group='"teachers"' >> data/logs/teacher68m.log 2>&1 \
      && ln -sfn "../train/$tname/model" runs/teachers/current \
      && echo "$(date +%H:%M) teacher $tname ready" >> data/logs/climb_loop.log
    [[ -f runs/climb/incumbent.json ]] && cp runs/climb/incumbent.json runs/climb/incumbent.prev.json
    if [[ -f runs/climb/incumbent.json ]]; then
      $R climb --from-incumbent --seeds 3 --metric ${CLIMB_METRIC:-exp_regret} --val-continuation ${VAL_CONTINUATION:-expand} --max-runs ${MAX_RUNS:-110} >> data/logs/climb.log 2>&1
    else
      $R climb --seeds 3 --metric ${CLIMB_METRIC:-exp_regret} --val-continuation ${VAL_CONTINUATION:-expand} --max-runs ${MAX_RUNS:-110} >> data/logs/climb.log 2>&1
    fi
    inc=$(.venv/bin/python -I -c "import json; print(json.load(open('runs/climb/incumbent.json'))['model_dir'])" 2>/dev/null)
    prev=$(.venv/bin/python -I -c "import json; print(json.load(open('runs/climb/incumbent.prev.json'))['model_dir'])" 2>/dev/null)
    if [[ -n "$inc" && -n "$prev" && "$inc" != "$prev" ]]; then
      # Closed-loop gate: offline val gains have not always carried to play (an ensemble-distilled model was
      # better on val and significantly worse on the devhub maps), so a new incumbent must not lose in play.
      if nice -n 5 $R gate --candidate "$inc" --reference "$prev" --workers ${GATE_WORKERS:-8} >> data/logs/climb_loop.log 2>&1; then
        echo "$(date +%H:%M) gate accepted ${inc:h:t} over ${prev:h:t}" >> data/logs/climb_loop.log
      else
        echo "$(date +%H:%M) gate rejected ${inc:h:t}; restoring ${prev:h:t}" >> data/logs/climb_loop.log
        cp runs/climb/incumbent.prev.json runs/climb/incumbent.json
        inc="$prev"
      fi
      git add runs/climb/incumbent.json runs/climb/gates.jsonl && git commit -q -m "Closed-loop gate after climb pass" && git push -q
    fi
    if [[ -n "$inc" ]]; then
      $R publish model --run "${inc:h}" >> data/logs/publish.log 2>&1 || echo "$(date +%H:%M) publish skipped (hf auth?)" >> data/logs/climb_loop.log
      # DAgger: the incumbent plays every train map once (epsilon-greedy); the train labeller takes these first.
      echo "$(date +%H:%M) dagger self-play with ${inc:h:t}" >> data/logs/climb_loop.log
      nice -n 5 $R states selfplay --tasks train --policy model --model "$inc" --games 1 --max-states 12 \
        --workers ${DAGGER_WORKERS:-6} --seed0 $(( $(date +%s) % 100000 )) >> data/logs/dagger.log 2>&1
    fi
    .venv/bin/rh17m trace end climb_pass examples=$n val=$val incumbent="${inc:h:t}"
    unset TRACEPARENT RH17M_TRACE_START_NS
    last=$n
    echo $n > runs/climb/last_n
  fi
  sleep 600
done
