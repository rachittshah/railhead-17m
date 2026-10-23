"""rh17m command line: states, labels, bench."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def _states(a: argparse.Namespace) -> int:
    from rh17m import states as S

    root = Path(a.root)
    if a.source == "teacher":
        n = 0
        for task_id, model, log_path in S.seed_games():
            got = S.harvest_teacher(task_id, model, log_path, root)
            n += len(got)
            print(json.dumps({"task": task_id, "model": model, "states": len(got)}), flush=True)
        print(json.dumps({"teacher_states": n}))
    elif a.source == "start":
        for t in a.tasks:
            print(json.dumps(S.start_state(t, root).__dict__))
    elif a.source == "selfplay":
        for t in a.tasks:
            for seed in range(a.seed0, a.seed0 + a.games):
                got = S.selfplay(t, root, seed, policy=a.policy, max_states=a.max_states)
                print(json.dumps({"task": t, "seed": seed, "states": len(got)}), flush=True)
    return 0


def _label(a: argparse.Namespace) -> int:
    from rh17m.labels import LabelConfig, label_all

    cfg = LabelConfig(seeds=a.seeds, chunk_days=a.chunk_days, chunks=a.chunks, max_air=a.max_air)
    only = a.only.split(",") if a.only else None
    print(json.dumps(label_all(Path(a.root), Path(a.out), workers=a.workers, cfg=cfg, only=only)))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="rh17m")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("states", help="harvest decision states")
    s.add_argument("source", choices=["teacher", "start", "selfplay"])
    s.add_argument("--root", default=str(DATA / "states"))
    s.add_argument("--tasks", nargs="*", default=[])
    s.add_argument("--policy", default="mix", choices=["mix", "random", "longest"])
    s.add_argument("--games", type=int, default=1)
    s.add_argument("--seed0", type=int, default=1)
    s.add_argument("--max-states", type=int, default=30)
    s.set_defaults(fn=_states)

    lb = sub.add_parser("label", help="simulator-label states")
    lb.add_argument("--root", default=str(DATA / "states"))
    lb.add_argument("--out", default=str(DATA / "labels"))
    lb.add_argument("--workers", type=int, default=8)
    lb.add_argument("--seeds", type=int, default=3)
    lb.add_argument("--chunk-days", type=int, default=91)
    lb.add_argument("--chunks", type=int, default=8)
    lb.add_argument("--max-air", type=int, default=8)
    lb.add_argument("--only", default="")
    lb.set_defaults(fn=_label)

    b = sub.add_parser("bench", help="simulator throughput")
    b.add_argument("rest", nargs=argparse.REMAINDER)
    b.set_defaults(fn=lambda a: __import__("rh17m.bench", fromlist=["main"]).main(a.rest))

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
