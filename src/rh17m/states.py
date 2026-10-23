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
TASKS_ROOT = Path.home() / "railhead" / "tasks"  # the local checkout: the 512x512 10-year arena maps live here
HUB_TASKS_ROOT = Path(__file__).resolve().parents[2] / "data" / "tasks"  # tasks fetched by ``rh17m tasks``
TASK_ROOTS = (HUB_TASKS_ROOT, TASKS_ROOT)


def find_task(task_id: str) -> Path:
    """The directory of ``task_id`` (task.json + start.sav), Hub copies first, then the local checkout."""
    for root in TASK_ROOTS:
        if (root / task_id / "start.sav").exists():
            return root / task_id
    raise FileNotFoundError(f"task {task_id} is in none of {TASK_ROOTS}; fetch it with `rh17m tasks`")


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
    for split in ("train", "dev", "ood"):
        if f"-{split}-" in task_id:
            return split
    return "other"


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
                    end_date: int | None = None) -> list[StateMeta]:
    """Replay one teacher game and save a state after each wait. Skips the game once it is marked done.
    ``end_date`` defaults to the run's meta.json beside the log."""
    task_dir = find_task(task_id)
    task = TaskSpec.load(task_dir)
    if end_date is None:
        end_date = int(json.loads((log_path.parent / "meta.json").read_text())["end_date"])
    prefix = f"{task_id}.teacher-{model}"
    marker = Path(root) / f".{prefix}.done"
    if marker.exists():
        return [load_meta(root, p.stem) for p in sorted(Path(root).glob(f"{prefix}.w*.json"))]
    for stale in Path(root).glob(f"{prefix}.w*"):  # an interrupted harvest restarts from the beginning
        stale.unlink()
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
    marker.write_text(json.dumps({"states": len(out), "valid": res.valid}))
    return out


# ---------------------------------------------------------------------- self-play
def selfplay(task_id: str, root: Path, seed: int, policy: str = "mix", max_states: int = 30,
             p_build: float = 0.6, epsilon: float = 0.2) -> list[StateMeta]:
    """Play a company and save a state at every wake-up.

    ``policy``: ``random`` builds a uniformly random air candidate, ``longest`` the longest one, ``mix`` flips a
    coin per build; ``model:<model dir>`` asks the trained model at every wake-up (it may hold) and explores
    with a random option with probability ``epsilon`` (DAgger: states from the policy's own distribution).
    Routes are built with the label engine's cash ladder. Waits are 30-365 days (model: 90 or 180, like play).
    The engine is reseeded with ``seed`` so games on the same map diverge."""
    from rhs1.candidates import enumerate_routes
    from rhs1.snapshot import aggregate_routes, take_snapshot

    from rh17m.labels import build_affordable, querier_for

    task_dir = find_task(task_id)
    task = TaskSpec.load(task_dir)
    rng = random.Random(seed)
    model = None
    tag = policy
    if policy.startswith("model:"):
        from rh17m.policy import ModelPolicy

        model_dir = Path(policy.split(":", 1)[1])
        model = ModelPolicy(model_dir)
        tag = f"model-{model_dir.parent.name}"
    prefix = f"{task_id}.selfplay-{tag}-s{seed}"
    marker = Path(root) / f".{prefix}.done"
    if marker.exists():
        return [load_meta(root, p.stem) for p in sorted(Path(root).glob(f"{prefix}.w*.json"))]
    for stale in Path(root).glob(f"{prefix}.w*"):  # an interrupted game is replayed from its seed
        stale.unlink()
    out: list[StateMeta] = []
    with tempfile.TemporaryDirectory(prefix="rh17m-selfplay-") as tmp:
        world = World.load(Path(tmp) / "w", task_dir / "start.sav")
        try:
            world.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
            end_date = int(world.state()["date"]) + task.horizon_days
            while len(out) < max_states and int(world.state()["date"]) < end_date:
                out.append(save_state(world, Path(root), f"{prefix}.w{len(out):03d}", task,
                                      f"selfplay:{tag}", end_date))
                pick = None
                if model is not None:
                    from rh17m.play import wakeup_observation

                    obs, cands = wakeup_observation(world, task, end_date, 8, model_dcfg(model))
                    keys = [k for k, _ in obs["options"]]
                    if rng.random() < epsilon:
                        key = rng.choice(keys)
                    else:
                        probs = model(obs)
                        key = max(probs, key=lambda k: (probs[k], k == "hold"))
                    pick = next((c for c in cands if c.candidate_id == key), None)
                elif rng.random() < p_build:
                    snap = take_snapshot(querier_for(world, task, end_date))
                    labels = {c["id"]: c["label"] for c in world.call("cargos")}
                    cands = [c for c in enumerate_routes(snap, aggregate_routes(snap), max_per_mode=8,
                                                         cargo_labels=labels) if c.mode == "air"]
                    if cands:
                        pick_longest = policy == "longest" or (policy == "mix" and rng.random() < 0.5)
                        pick = max(cands, key=lambda c: c.length_tiles) if pick_longest else rng.choice(cands)
                if pick is not None:
                    try:
                        build_affordable(world, pick.a["id"], pick.b["id"])
                    except BridgeError as e:  # builder reports failures itself; guard the unexpected
                        log.warning("selfplay build raised %s", e)
                waits = (90, 180) if model is not None else (30, 60, 90, 180, 365)
                world.step(days=rng.choice(waits))
        finally:
            world.close()
    marker.write_text(json.dumps({"states": len(out)}))
    return out


def model_dcfg(model):
    """The DatasetConfig the model was trained with (its rh17m.json), so self-play renders the same text."""
    from rh17m.dataset import DatasetConfig

    meta = Path(model.model_dir) / "rh17m.json"
    data = json.loads(meta.read_text()).get("data", {}) if meta.exists() else {}
    return DatasetConfig(**data)


def _selfplay_job(args: tuple) -> dict:
    task_id, root, seed, policy, max_states, epsilon = args
    try:
        got = selfplay(task_id, Path(root), seed, policy=policy, max_states=max_states, epsilon=epsilon)
        return {"task": task_id, "seed": seed, "policy": policy, "states": len(got)}
    except Exception as e:  # noqa: BLE001 - a crashed game loses its own states only
        return {"task": task_id, "seed": seed, "policy": policy, "error": repr(e)}


def selfplay_many(task_ids: list[str], root: Path, seeds: list[int], policies: list[str],
                  max_states: int = 30, workers: int = 6, epsilon: float = 0.2) -> Iterator[dict]:
    """Every (task, seed, policy) game, ``workers`` at a time; yields one summary per finished game."""
    import multiprocessing as mp

    jobs = [(t, str(root), s, p, max_states, epsilon) for s in seeds for p in policies for t in task_ids]
    with mp.get_context("spawn").Pool(workers) as pool:
        yield from pool.imap_unordered(_selfplay_job, jobs)


def start_state(task_id: str, root: Path) -> StateMeta:
    task_dir = find_task(task_id)
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
