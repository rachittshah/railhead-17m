#!/usr/bin/env zsh
# Stop one pipeline stage (e.g. `scripts/stop_stage.sh label`) and everything it started. The pool parent dies
# first so it cannot respawn workers; then its orphaned workers and their OpenTTD servers are killed.
set -u
stage=${1:?usage: stop_stage.sh label|states}
killtree() { local p=$1; for c in $(pgrep -P $p); do killtree $c; done; kill $p 2>/dev/null; }
for p in $(pgrep -f "bin/rh17m $stage"); do kill -9 $p 2>/dev/null; done
sleep 1
for p in $(ps -Ao pid,ppid,command | awk '$2==1 && /railhead-17m\/.venv\/bin\/python3 -c from multiprocessing.spawn/ {print $1}'); do
  killtree $p
done
