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
    elif a.source == "hub-teacher":
        from rh17m.hub import harvest_hub_teachers

        for r in harvest_hub_teachers(root):
            print(json.dumps(r), flush=True)
    elif a.source == "start":
        for t in a.tasks:
            print(json.dumps(S.start_state(t, root).__dict__))
    elif a.source == "selfplay":
        tasks = a.tasks or sorted(p.name for p in S.TASKS_ROOT.glob("arena-*-02*"))
        if a.tasks == ["hub"]:
            tasks = sorted(p.parent.name for p in S.HUB_TASKS_ROOT.glob("*/start.sav"))
        policies = ["random", "longest", "mix"] if a.policy == "all" else [a.policy]
        for r in S.selfplay_many(tasks, root, list(range(a.seed0, a.seed0 + a.games)), policies,
                                 max_states=a.max_states, workers=a.workers):
            print(json.dumps(r), flush=True)
    return 0


def _tasks(a: argparse.Namespace) -> int:
    from rh17m.hub import fetch_tasks

    ids = fetch_tasks(tuple(a.families.split(",")))
    print(json.dumps({"tasks": len(ids), "first": ids[:3]}))
    return 0


def _label(a: argparse.Namespace) -> int:
    from rh17m.labels import LabelConfig, label_all

    cfg = LabelConfig(seeds=a.seeds, chunk_days=a.chunk_days, chunks=a.chunks, max_air=a.max_air)
    only = a.only.split(",") if a.only else None
    print(json.dumps(label_all(Path(a.root), Path(a.out), workers=a.workers, cfg=cfg, only=only)))
    return 0


def _baselines(a: argparse.Namespace) -> int:
    from rh17m.dataset import DatasetConfig, load_examples
    from rh17m.evaluate import BASELINES, metrics

    exs = load_examples(Path(a.labels), DatasetConfig())
    out: dict = {"examples": {s: sum(e["split"] == s for e in exs) for s in ("train", "val", "test")}}
    for split in ("train", "val", "test"):
        part = [e for e in exs if e["split"] == split]
        if part:
            out[split] = {name: {k: round(v, 4) for k, v in metrics(part, [f(e) for e in part]).items()}
                          for name, f in BASELINES.items()}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def _train(a: argparse.Namespace) -> int:
    from rh17m.train import TrainConfig, run

    cfg = TrainConfig(name=a.name)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k.startswith("data."):
            cfg.data[k[5:]] = json.loads(v)
        else:
            cur = getattr(cfg, k)
            setattr(cfg, k, tuple(json.loads(v)) if isinstance(cur, tuple) else type(cur)(json.loads(v)))
    res = run(cfg, Path(a.labels), Path(a.out) / a.name)
    print(json.dumps({k: res[k] for k in res if not k.endswith("_baselines")}, indent=1))
    return 0


def _play(a: argparse.Namespace) -> int:
    from rh17m.play import tournament

    tasks = a.tasks
    for r in tournament(tasks, a.policies, list(range(a.seed0, a.seed0 + a.seeds)), Path(a.out),
                        wake_days=a.wake_days, workers=a.workers, model_dir=a.model):
        print(json.dumps({k: r[k] for k in ("policy", "task", "seed", "final_nw", "score", "builds", "built_ok")}),
              flush=True)
    return 0


def _climb(a: argparse.Namespace) -> int:
    from rh17m.climb import climb
    from rh17m.train import TrainConfig

    base = TrainConfig(name="base")
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k.startswith("data."):
            base.data[k[5:]] = json.loads(v)
        else:
            setattr(base, k, json.loads(v))
    best = climb(base, Path(a.labels), noise_seeds=a.noise_seeds, max_runs=a.max_runs, push=not a.no_push)
    print(json.dumps({"incumbent": best.name}))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(prog="rh17m")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("states", help="harvest decision states")
    s.add_argument("source", choices=["teacher", "hub-teacher", "start", "selfplay"])
    s.add_argument("--root", default=str(DATA / "states"))
    s.add_argument("--tasks", nargs="*", default=[])
    s.add_argument("--policy", default="all", choices=["all", "mix", "random", "longest"])
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--games", type=int, default=1)
    s.add_argument("--seed0", type=int, default=1)
    s.add_argument("--max-states", type=int, default=12)
    s.set_defaults(fn=_states)

    tk = sub.add_parser("tasks", help="fetch Railhead tasks from the Hub into data/tasks")
    tk.add_argument("--families", default="tycoon")
    tk.set_defaults(fn=_tasks)

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

    bl = sub.add_parser("baselines", help="baseline policies on the labelled decisions")
    bl.add_argument("--labels", default=str(DATA / "labels"))
    bl.add_argument("--out", default="")
    bl.set_defaults(fn=_baselines)

    tr = sub.add_parser("train", help="fine-tune the 17M model on the labelled decisions")
    tr.add_argument("--name", required=True)
    tr.add_argument("--labels", default=str(DATA / "labels"))
    tr.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "runs" / "train"))
    tr.add_argument("--set", nargs="*", default=[], help="TrainConfig overrides key=json, data.<key>=json")
    tr.set_defaults(fn=_train)

    cl = sub.add_parser("climb", help="hillclimb training/data settings on val with a noise floor")
    cl.add_argument("--labels", default=str(DATA / "labels"))
    cl.add_argument("--set", nargs="*", default=[])
    cl.add_argument("--noise-seeds", type=int, default=3)
    cl.add_argument("--max-runs", type=int, default=60)
    cl.add_argument("--no-push", action="store_true")
    cl.set_defaults(fn=_climb)

    pl = sub.add_parser("play", help="closed-loop games: policies x tasks x seeds")
    pl.add_argument("--tasks", nargs="+", required=True)
    pl.add_argument("--policies", nargs="+", default=["hold", "longest", "random"])
    pl.add_argument("--model", default=None)
    pl.add_argument("--seeds", type=int, default=1)
    pl.add_argument("--seed0", type=int, default=1)
    pl.add_argument("--wake-days", type=int, default=90)
    pl.add_argument("--workers", type=int, default=4)
    pl.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "runs" / "play"))
    pl.set_defaults(fn=_play)

    b = sub.add_parser("bench", help="simulator throughput")
    b.add_argument("rest", nargs=argparse.REMAINDER)
    b.set_defaults(fn=lambda a: __import__("rh17m.bench", fromlist=["main"]).main(a.rest))

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
