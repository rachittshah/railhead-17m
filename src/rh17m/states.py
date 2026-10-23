"""Decision states: savegames of a company at a wake-up, plus the metadata the label engine needs.

Every state is ``<root>/<state_id>.sav`` with ``<state_id>.json`` beside it. Three sources:

- ``teacher``: replay a frontier model's action log (Railhead seed games) and save the world after every
  wait, the moments the teacher woke up and decided. Replay checks every recorded checksum, so a state
  comes from a verified game prefix.
- ``selfplay``: a cheap scripted player (random or longest-band air routes, random waits) on any task, so
  states cover maps and situations no teacher played. The policy is recorded with the state.
- ``start``: the task's sealed start state (the first-route decision).

Splits follow the map: a state belongs to its task's split (arena-train-* -> train, arena-dev-* -> dev), so
train and evaluation never share a map.
"""

from __future__ import annotations

import json
import logging
import random
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from railhead.bridge import BridgeError
from railhead.replay import replay
from railhead.task import TaskSpec
from railhead.world import World

log = logging.getLogger(__name__)

SEED_GAME_ROOTS = (
    Path.home() / "railhead-system1" / "runs" / "seed-games",
    Path.home() / "railhead-system1" / "runs" / "seed-games-opus",
)
TASKS_ROOT = Path.home() / "railhead" / "tasks"


@dataclass(frozen=True)
class StateMeta:
    state_id: str
    task_id: str
    split: str
    source: str  # teacher:<model> | selfplay:<policy> | start
    date: int  # OpenTTD date (days)
    ymd: str
    ticks: int
    end_date: int  # last day of the task's horizon (run-off excluded)
    net_worth: int
    vehicles: int


def split_of(task_id: str) -> str:
    return "train" if "-train-" in task_id else "dev" if "-dev-" in task_id else "other"


def _kpis(world: World) -> tuple[dict, int, int]:
    st = world.state()
    c = world.call("company")
    nw = int(c.get("money") or 0) - int(c.get("loan") or 0)
    return st, nw, len(world.call("vehicles"))


def save_state(world: World, root: Path, state_id: str, task: TaskSpec, source: str, end_date: int) -> StateMeta:
    root.mkdir(parents=True, exist_ok=True)
    st, nw, vehicles = _kpis(world)
    world.save(root / f"{state_id}.sav")
    meta = StateMeta(state_id=state_id, task_id=task.task_id, split=split_of(task.task_id), source=source,
                     date=int(st["date"]), ymd=str(st["ymd"]), ticks=int(st["ticks"]), end_date=end_date,
                     net_worth=nw, vehicles=vehicles)
    tmp = root / f".{state_id}.json.tmp"
    tmp.write_text(json.dumps(asdict(meta)))
    tmp.rename(root / f"{state_id}.json")
    return meta


def load_meta(root: Path, state_id: str) -> StateMeta:
    return StateMeta(**json.loads((Path(root) / f"{state_id}.json").read_text()))


def list_states(root: Path) -> list[str]:
    """State ids with both files present, sorted."""
    root = Path(root)
    return sorted(p.stem for p in root.glob("*.json") if (root / f"{p.stem}.sav").exists())


# ---------------------------------------------------------------------- teacher
def seed_games(roots: tuple[Path, ...] = SEED_GAME_ROOTS) -> Iterator[tuple[str, str, Path]]:
    """(task_id, model, actions.jsonl) for every teacher seed game on disk."""
    for root in roots:
        for log_path in sorted(root.glob("*/*/state/run/actions.jsonl")):
            model_dir = log_path.parents[2]
            yield model_dir.parent.name, model_dir.name, log_path


def harvest_teacher(task_id: str, model: str, log_path: Path, root: Path,
                    tasks_root: Path = TASKS_ROOT) -> list[StateMeta]:
    """Replay one teacher game and save a state after each wait. Skips the game if its states exist."""
    task_dir = tasks_root / task_id
    task = TaskSpec.load(task_dir)
    meta_run = json.loads((log_path.parent / "meta.json").read_text())
    end_date = int(meta_run["end_date"])
    prefix = f"{task_id}.teacher-{model}"
    if any(Path(root).glob(f"{prefix}.w*.json")):
        return [load_meta(root, p.stem) for p in sorted(Path(root).glob(f"{prefix}.w*.json"))]
    out: list[StateMeta] = []

    def on_wait(world: World, record: dict) -> None:
        if int(world.state()["date"]) >= end_date:
            return  # the horizon has passed: no decision is left to make
        out.append(save_state(world, Path(root), f"{prefix}.w{len(out):03d}", task, f"teacher:{model}", end_date))

    with tempfile.TemporaryDirectory(prefix="rh17m-harvest-") as tmp:
        res = replay(task_dir / "start.sav", log_path, Path(tmp) / "replay", on_wait=on_wait)
    if not res.valid:
        log.warning("teacher game %s/%s diverged after %d states: %s", task_id, model, len(out),
                    res.divergences[:1])
    return out


# ---------------------------------------------------------------------- self-play
def selfplay(task_id: str, root: Path, seed: int, policy: str = "mix", max_states: int = 30,
             p_build: float = 0.6, tasks_root: Path = TASKS_ROOT) -> list[StateMeta]:
    """Play a scripted company and save a state at every wake-up.

    ``policy``: ``random`` builds a uniformly random air candidate, ``longest`` the longest-band one,
    ``mix`` flips a coin per build. Waits are drawn from 30-365 days. The engine is reseeded with ``seed`` so
    games on the same map diverge."""
    from rhs1.builders.air import AirPlan, build_air_route
    from rhs1.candidates import enumerate_routes
    from rhs1.snapshot import aggregate_routes, take_snapshot

    from rh17m.labels import querier_for

    task_dir = tasks_root / task_id
    task = TaskSpec.load(task_dir)
    rng = random.Random(seed)
    prefix = f"{task_id}.selfplay-{policy}-s{seed}"
    done = sorted(Path(root).glob(f"{prefix}.w*.json"))
    if done:
        return [load_meta(root, p.stem) for p in done]
    out: list[StateMeta] = []
    with tempfile.TemporaryDirectory(prefix="rh17m-selfplay-") as tmp:
        world = World.load(Path(tmp) / "w", task_dir / "start.sav")
        try:
            world.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
            end_date = int(world.state()["date"]) + task.horizon_days
            while len(out) < max_states and int(world.state()["date"]) < end_date:
                out.append(save_state(world, Path(root), f"{prefix}.w{len(out):03d}", task,
                                      f"selfplay:{policy}", end_date))
                if rng.random() < p_build:
                    query = querier_for(world, task, end_date)
                    snap = take_snapshot(query)
                    labels = {c["id"]: c["label"] for c in world.call("cargos")}
                    cands = [c for c in enumerate_routes(snap, aggregate_routes(snap), max_per_mode=8,
                                                         cargo_labels=labels) if c.mode == "air"]
                    if cands:
                        pick_longest = policy == "longest" or (policy == "mix" and rng.random() < 0.5)
                        c = max(cands, key=lambda c: c.length_tiles) if pick_longest else rng.choice(cands)
                        try:
                            build_air_route(world, AirPlan(c.a["id"], c.b["id"], "auto", 2, -1))
                        except BridgeError as e:  # builder reports failures itself; guard the unexpected
                            log.warning("selfplay build raised %s", e)
                world.step(days=rng.choice((30, 60, 90, 180, 365)))
        finally:
            world.close()
    return out


def _selfplay_job(args: tuple) -> dict:
    task_id, root, seed, policy, max_states = args
    try:
        got = selfplay(task_id, Path(root), seed, policy=policy, max_states=max_states)
        return {"task": task_id, "seed": seed, "policy": policy, "states": len(got)}
    except Exception as e:  # a crashed game loses its own states only
        return {"task": task_id, "seed": seed, "policy": policy, "error": repr(e)}


def selfplay_many(task_ids: list[str], root: Path, seeds: list[int], policies: list[str],
                  max_states: int = 30, workers: int = 6) -> Iterator[dict]:
    """Every (task, seed, policy) game, ``workers`` at a time; yields one summary per finished game."""
    import multiprocessing as mp

    jobs = [(t, str(root), s, p, max_states) for s in seeds for p in policies for t in task_ids]
    with mp.get_context("spawn").Pool(workers) as pool:
        yield from pool.imap_unordered(_selfplay_job, jobs)


def start_state(task_id: str, root: Path, tasks_root: Path = TASKS_ROOT) -> StateMeta:
    task_dir = tasks_root / task_id
    task = TaskSpec.load(task_dir)
    sid = f"{task_id}.start"
    if (Path(root) / f"{sid}.json").exists():
        return load_meta(root, sid)
    with tempfile.TemporaryDirectory(prefix="rh17m-start-") as tmp:
        world = World.load(Path(tmp) / "w", task_dir / "start.sav")
        try:
            end_date = int(world.state()["date"]) + task.horizon_days
            return save_state(world, Path(root), sid, task, "start", end_date)
        finally:
            world.close()
