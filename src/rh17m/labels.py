"""Simulator labels: play every candidate decision forward and measure what it was worth.

For one state the engine enumerates the air candidates (routes-v1, up to ``max_air``) plus ``hold``. Each
option is applied once on an unseeded fork (the build is deterministic), saved, and then rolled forward
``seeds`` times under the *hold* continuation policy (no further actions), each rollout reseeded with the
same seed list across options (common random numbers, so differences between options are not seed noise).
Net worth (cash minus loan, the arena objective) is recorded at the end of every ``chunk_days`` chunk.

The record keeps the rendered wakeup-v1 state and the candidate rows, so the dataset builder never needs the
simulator again. Writes are atomic (tmp + rename) and a state with a record is skipped, so a run can be
killed and resumed at any point.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import multiprocessing as mp
import random
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from railhead.bridge import BridgeError
from railhead.task import TaskSpec
from railhead.world import World

from rh17m.states import find_task, list_states, load_meta

log = logging.getLogger(__name__)

HOLD = "hold"


@dataclass(frozen=True)
class LabelConfig:
    seeds: int = 3
    chunk_days: int = 91
    chunks: int = 8  # 728 days: about two years, long enough for an air route to pay back
    max_air: int = 8
    planes: int = 2
    version: str = "lbl-v2"


# How a chosen route is built: the first configuration the company can pay for. A large airport with two
# fast planes is the routes-v1 estimate; poorer companies get the cheaper airports (small before 1960-ish,
# commuter later; these exclude big planes) and then a single plane, as a player short of cash would build.
# Only a refusal that spent nothing (no cash, or a type not available this year) moves down the ladder.
BUILD_LADDER: tuple[tuple[str, int], ...] = (
    ("auto", 2), ("small", 2), ("commuter", 2), ("auto", 1), ("small", 1), ("commuter", 1))
_NEXT_RUNG = ("ERR_NOT_ENOUGH_CASH", "ERR_TYPE_NOT_AVAILABLE")


def build_affordable(world: World, a_town: int, b_town: int, ladder=BUILD_LADDER):
    """Try each (airport type, planes) on the ladder until one is not refused for cash or availability.
    Returns the last BuildResult and the configuration used."""
    from rhs1.builders.air import AirPlan, build_air_route

    br, cfg = None, ladder[0]
    for cfg in ladder:
        br = build_air_route(world, AirPlan(a_town, b_town, cfg[0], cfg[1], -1))
        if br.ok or br.cost or br.steps or not (br.failed_step or "").endswith(_NEXT_RUNG):
            break
    return br, cfg


def querier_for(world: World, task: TaskSpec, end_date: int):
    from rhs1.snapshot import world_querier

    return world_querier(world, {"end_date": end_date}, task, [])


def rollout_seeds(state_id: str, n: int) -> list[int]:
    return [int(hashlib.sha256(f"{state_id}:{m}".encode()).hexdigest()[:8], 16) for m in range(n)]


def _net_worth(world: World) -> tuple[int, int]:
    c = world.call("company")
    money = int(c.get("money") or 0)
    return money - int(c.get("loan") or 0), money


def _rollout(save: Path, workdir: Path, seed: int, cfg: LabelConfig) -> dict:
    world = World.load(workdir, save)
    try:
        out = world.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
        if not out or not out[0].startswith("Reseeded"):
            raise RuntimeError(f"engine lacks the reseed patch: {out!r}")
        nw, money = [], []
        for _ in range(cfg.chunks):
            world.step(days=cfg.chunk_days)
            a, b = _net_worth(world)
            nw.append(a)
            money.append(b)
        return {"nw": nw, "money": money}
    finally:
        world.close()


def label_state(root: Path, state_id: str, cfg: LabelConfig = LabelConfig()) -> dict:
    """Label one state (see the module docstring). Returns the record; does not write it."""
    from rhs1.candidates import enumerate_routes
    from rhs1.snapshot import aggregate_routes, render_state, take_snapshot

    meta = load_meta(root, state_id)
    task = TaskSpec.load(find_task(meta.task_id))
    t0 = time.perf_counter()
    seeds = rollout_seeds(state_id, cfg.seeds)
    with tempfile.TemporaryDirectory(prefix="rh17m-label-") as tmp:
        tmp = Path(tmp)
        parent = World.load(tmp / "parent", Path(root) / f"{state_id}.sav")
        try:
            snap = take_snapshot(querier_for(parent, task, meta.end_date))
            routes = aggregate_routes(snap)
            labels = {c["id"]: c["label"] for c in parent.call("cargos")}
            cands = [c for c in enumerate_routes(snap, routes, max_per_mode=cfg.max_air, cargo_labels=labels)
                     if c.mode == "air"][: cfg.max_air]
            rendered = render_state(snap, routes, cands)
            nw0, money0 = _net_worth(parent)
            saves: dict[str, Path] = {HOLD: parent.save(tmp / "opt-hold.sav")}
            builds: dict[str, dict] = {HOLD: {"ok": True, "cost": 0, "failed_step": None}}
            aliases: set[str] = set()  # failed before spending anything: the world is the hold world
            for i, c in enumerate(cands):
                child = parent.fork(tmp / f"build{i}")
                try:
                    try:
                        br, used = build_affordable(child, c.a["id"], c.b["id"])
                        builds[c.candidate_id] = {"ok": br.ok, "cost": br.cost, "failed_step": br.failed_step,
                                                  "airport_type": br.airport_type, "planes": used[1],
                                                  "ladder": list(used)}
                        untouched = not br.ok and not br.cost and not br.steps
                    except BridgeError as e:
                        builds[c.candidate_id] = {"ok": False, "cost": None, "failed_step": f"raise:{e.code}"}
                        untouched = False
                    if untouched:
                        aliases.add(c.candidate_id)
                    else:
                        saves[c.candidate_id] = child.save(tmp / f"opt-{i}.sav")
                finally:
                    child.close()
        finally:
            parent.close()
        t_build = time.perf_counter() - t0
        outcomes: dict[str, Any] = {}
        for key, save in saves.items():
            outcomes[key] = {**builds[key], "rollouts": [
                _rollout(save, tmp / f"roll-{key}-{m}", s, cfg) for m, s in enumerate(seeds)]}
        for key in aliases:  # same world and same seeds as hold: the rollouts would repeat hold's exactly
            outcomes[key] = {**builds[key], "alias": HOLD, "rollouts": outcomes[HOLD]["rollouts"]}
    return {
        "version": cfg.version, "config": asdict(cfg), "state": asdict(meta), "seeds": seeds,
        "nw0": nw0, "money0": money0, "rendered_state": rendered,
        "candidates": [dataclasses.asdict(c) for c in cands], "options": [c.candidate_id for c in cands] + [HOLD],
        "outcomes": outcomes,
        "timing": {"build_s": round(t_build, 2), "total_s": round(time.perf_counter() - t0, 2)},
    }


def _write_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)


def _job(args: tuple) -> dict:
    root, state_id, out_dir, cfg_d = args
    cfg = LabelConfig(**cfg_d)
    dest = Path(out_dir) / f"{state_id}.json"
    if dest.exists():
        return {"state_id": state_id, "skipped": True}
    try:
        rec = label_state(Path(root), state_id, cfg)
        _write_atomic(dest, rec)
        return {"state_id": state_id, "ok": True, "s": rec["timing"]["total_s"], "options": len(rec["options"])}
    except Exception as e:  # noqa: BLE001 - one bad state must not stop the run; the error file lets a rerun retry it
        _write_atomic(Path(out_dir) / f"{state_id}.error.json",
                      {"state_id": state_id, "error": repr(e), "trace": traceback.format_exc()[-4000:]})
        return {"state_id": state_id, "ok": False, "error": repr(e)}


def label_all(root: Path, out_dir: Path, workers: int = 8, cfg: LabelConfig = LabelConfig(),
              only: list[str] | None = None) -> dict:
    """Label every state in ``root`` without a record in ``out_dir``, ``workers`` states at a time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in (only or list_states(root)) if not (out_dir / f"{s}.json").exists()]
    # A fixed shuffle, so every split and source fills at the same rate instead of in name order.
    random.Random(0).shuffle(todo)
    progress = out_dir / "_progress.log"
    done = failed = 0
    t0 = time.perf_counter()
    jobs = [(str(root), s, str(out_dir), asdict(cfg)) for s in todo]
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            done += bool(r.get("ok"))
            failed += r.get("ok") is False
            line = json.dumps({**r, "done": done, "failed": failed, "todo": len(todo),
                               "elapsed_s": round(time.perf_counter() - t0, 1)})
            with progress.open("a") as f:
                f.write(line + "\n")
            log.info(line)
    return {"labelled": done, "failed": failed, "todo": len(todo), "elapsed_s": round(time.perf_counter() - t0, 1)}
