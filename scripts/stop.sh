#!/usr/bin/env zsh
# Stop every railhead-17m stage: the stage Python processes first (exact command match, so pool parents
# cannot respawn workers), then their orphaned workers and OpenTTD servers, then the tmux session.
set -u
for p in $(pgrep -f "python3 .venv/bin/rh17m "); do kill -9 $p 2>/dev/null; done
sleep 1
for p in $(ps -Ao pid,ppid,command | awk '$2==1 && /railhead-17m\/.venv\/bin\/python3 -c from multiprocessing.spawn/ {print $1}'); do
  kill -9 $p 2>/dev/null
done
sleep 1
for p in $(ps -Ao pid,ppid,command | awk '$2==1 && /\/openttd -D 127\.0\.0\.1/ {print $1}'); do kill $p 2>/dev/null; done
tmux kill-session -t rh17m 2>/dev/null
rm -rf "${0:A:h}/../data/labels/.claims"
echo "stopped"
