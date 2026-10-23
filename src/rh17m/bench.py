"""Simulator throughput on this machine: load, fork (save + load + reseed) and step costs.

The label engine forks every decision state K x M times and steps each branch H days, so these three
numbers size every labelling run. ``measure`` times a fresh start state; with ``log`` it first replays an
action log to its end, so the timings reflect a grown network (vehicles and stations slow the step).
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from pathlib import Path

from railhead.replay import replay
from railhead.world import World


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def measure(task_dir: Path, log: Path | None = None, days: int = 365, forks: int = 3) -> dict:
    """Time one load, ``forks`` reseeded forks and a ``days`` step on each fork. Returns seconds and
    game-days per second."""
    task_dir = Path(task_dir)
    with tempfile.TemporaryDirectory(prefix="rh17m-bench-") as tmp:
        tmp = Path(tmp)
        if log is None:
            world, load_s = _timed(lambda: World.load(tmp / "base", task_dir / "start.sav"))
        else:
            res, load_s = _timed(lambda: replay(task_dir / "start.sav", Path(log), tmp / "base", keep_open=True))
            if not res.valid or res.world is None:
                raise RuntimeError(f"replay diverged: {res.divergences[:2]}")
            world = res.world
        try:
            vehicles = len(world.call("vehicles"))
            fork_s, step_s = [], []
            for i in range(forks):
                child, s = _timed(lambda i=i: world.fork(tmp / f"fork{i}", reseed=1000 + i))
                fork_s.append(s)
                try:
                    _, s = _timed(lambda child=child: child.step(days=days))
                    step_s.append(s)
                finally:
                    child.close()
        finally:
            world.close()
    step_med = statistics.median(step_s)
    return {"task": task_dir.name, "replayed": log is not None, "vehicles": vehicles, "days": days,
            "load_or_replay_s": round(load_s, 3), "fork_s_median": round(statistics.median(fork_s), 3),
            "step_s_median": round(step_med, 3), "game_days_per_s": round(days / step_med, 1),
            "rollout_s": round(statistics.median(fork_s) + step_med, 3)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="rh17m bench", description=__doc__)
    p.add_argument("task_dir", type=Path)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--forks", type=int, default=3)
    a = p.parse_args(argv)
    print(json.dumps(measure(a.task_dir, a.log, a.days, a.forks)))
    return 0
