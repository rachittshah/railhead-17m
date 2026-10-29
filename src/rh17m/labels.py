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
import os
import random
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import railhead.server as rh_server
from railhead.actions import to_gs
from railhead.bridge import BridgeError
from railhead.task import TaskSpec
from railhead.world import World

from rh17m import tracing
from rh17m.expand import as_dicts, expand_affordable, expansion_candidates
from rh17m.states import find_task, list_states, load_meta

log = logging.getLogger(__name__)

HOLD = "hold"


@dataclass(frozen=True)
class LabelConfig:
    seeds: int = 3  # val/test/ood states: evaluation needs the tighter values
    train_seeds: int = 1  # train states: CRN keeps option differences tight; more states beat more seeds
    chunk_days: int = 91
    chunks: int = 8  # 728 days: about two years, long enough for an air route to pay back
    max_air: int = 8
    max_expand: int = 3  # expansion options: add planes to the most profitable existing air routes
    # What the company does after the option, inside each rollout. "hold" does nothing, which values only the
    # option's own two years and undervalues new routes (no later planes on them). "expand" adds one plane to
    # the most profitable air route every ``continue_every`` days when cash allows (no site searches: cheap).
    continuation: str = "expand"
    continue_every: int = 182
    planes: int = 2
    version: str = "lbl-v4"  # v2.x: routes only; v3: + expansion options; v4: "expand" continuation in rollouts


# How a chosen route is built: the first configuration the company can pay for. A large airport with two
# fast planes is the routes-v1 estimate; poorer companies get the cheaper airports (small before 1960-ish,
# commuter later; these exclude big planes) and then a single plane, as a player short of cash would build.
# Only a refusal that spent nothing (no cash, or a type not available this year) moves down the ladder.
BUILD_LADDER: tuple[tuple[str, int], ...] = (
    ("auto", 2), ("small", 2), ("commuter", 2), ("auto", 1), ("small", 1), ("commuter", 1))
# ERR_NO_SITE also covers "no affordable site", which a cheaper airport type can fix.
_NEXT_RUNG = ("ERR_NOT_ENOUGH_CASH", "ERR_TYPE_NOT_AVAILABLE", "ERR_NO_SITE")


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


def candidate_extras(snap, cands, labels: dict[int, str]) -> dict[str, dict]:
    """The fastest eligible passenger plane of each air candidate (what routes-v1 prices), by candidate id."""
    from rhs1.candidates import _distance_sq, _passenger_plane

    pass_id = next((k for k, v in labels.items() if v == "PASS"), None)
    out = {}
    for c in cands:
        plane = _passenger_plane(snap, pass_id, _distance_sq(c.a, c.b))
        out[c.candidate_id] = {} if plane is None else {
            "plane": plane.get("name"), "speed": int(plane.get("speed") or 0),
            "capacity": int(plane.get("capacity") or 0), "price": int(plane.get("price") or 0)}
    return out


def querier_for(world: World, task: TaskSpec, end_date: int):
    from rhs1.snapshot import world_querier

    return world_querier(world, {"end_date": end_date}, task, [])


def rollout_seeds(state_id: str, n: int) -> list[int]:
    return [int(hashlib.sha256(f"{state_id}:{m}".encode()).hexdigest()[:8], 16) for m in range(n)]


def _net_worth(world: World) -> tuple[int, int]:
    c = world.call("company")
    money = int(c.get("money") or 0)
    return money - int(c.get("loan") or 0), money


def continue_expand(world: World) -> bool:
    """One continuation step: clone the most profitable air plane (shared orders) into the hangar nearest to
    it. Returns whether a plane was added; a refusal (cash, hangar) is not an error."""
    planes = [v for v in world.call("vehicles") if v.get("vtype") == "air" and int(v.get("profit_last_year") or 0) > 0]
    if not planes:
        return False
    best = max(planes, key=lambda v: (int(v.get("profit_last_year") or 0), int(v.get("profit_this_year") or 0)))
    hangars = [d for d in world.call("depots") if d.get("vtype") == "air"]
    if not hangars:
        return False
    h = min(hangars, key=lambda d: abs(int(d["x"]) - int(best.get("x") or 0)) + abs(int(d["y"]) - int(best.get("y") or 0)))
    try:
        method, args = to_gs("vehicle.clone", {"depot_tile": int(h["tile"]), "vehicle": int(best["id"]),
                                               "share_orders": True})
        vid = int(world.call(method, **args).get("vehicle", -1))
        if vid < 0:
            return False
        method, args = to_gs("vehicle.start", {"vehicle": vid})
        world.call(method, **args)
        return True
    except BridgeError:
        return False


def _rollout(save: Path, workdir: Path, seed: int, cfg: LabelConfig, chunks: int | None = None,
             key: str = "") -> dict:
    with tracing.span("label.rollout", option=key, seed=seed, chunks=cfg.chunks if chunks is None else chunks):
        return _rollout_traced(save, workdir, seed, cfg, chunks)


def _rollout_traced(save: Path, workdir: Path, seed: int, cfg: LabelConfig, chunks: int | None) -> dict:
    world = World.load(workdir, save)
    try:
        out = world.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
        if not out or not out[0].startswith("Reseeded"):
            raise RuntimeError(f"engine lacks the reseed patch: {out!r}")
        nw, money, added = [], [], 0
        every = max(1, cfg.continue_every // cfg.chunk_days)
        n = cfg.chunks if chunks is None else chunks
        for i in range(n):
            world.step(days=cfg.chunk_days)
            a, b = _net_worth(world)
            nw.append(a)
            money.append(b)
            if cfg.continuation == "expand" and (i + 1) % every == 0 and i + 1 < n:
                with tracing.span("label.continue_expand", chunk=i) as sp:
                    ok = continue_expand(world)
                    sp.set_attribute("added", ok)
                added += ok
        return {"nw": nw, "money": money, "continued": added}
    finally:
        world.close()


@contextmanager
def fast_saves(fmt: str = "none"):
    """Uncompressed savegames for the throwaway fork/option saves of one label job: 52 ms vs 1.2 s (LZMA) per
    save of a 148-vehicle state under load, ~3.5 MB each in a temp dir, same future (checksum) as LZMA.
    Harvested states in data/states keep the default format."""
    misc = rh_server.BASE_SETTINGS.setdefault("misc", {})
    old = misc.get("savegame_format")
    misc["savegame_format"] = fmt
    try:
        yield
    finally:
        if old is None:
            misc.pop("savegame_format", None)
        else:
            misc["savegame_format"] = old


def label_state(root: Path, state_id: str, cfg: LabelConfig = LabelConfig()) -> dict:
    """Label one state (see the module docstring). Returns the record; does not write it."""
    with fast_saves():
        return _label_state(root, state_id, cfg)


def _label_state(root: Path, state_id: str, cfg: LabelConfig) -> dict:
    from rhs1.candidates import enumerate_routes
    from rhs1.snapshot import aggregate_routes, render_state, take_snapshot

    meta = load_meta(root, state_id)
    task = TaskSpec.load(find_task(meta.task_id))
    t0 = time.perf_counter()
    gs0 = tracing.bridge_totals()
    starts0 = tracing.server_starts()
    from rh17m.dataset import map_split

    split = map_split(meta.task_id)
    seeds = rollout_seeds(state_id, cfg.train_seeds if split == "train" else cfg.seeds)
    tracing.annotate(split=split, seeds=len(seeds), task=meta.task_id, source=meta.source)
    with tempfile.TemporaryDirectory(prefix="rh17m-label-") as tmp:
        tmp = Path(tmp)
        parent = World.load(tmp / "parent", Path(root) / f"{state_id}.sav")
        try:
            with tracing.span("label.snapshot", bridge=True) as sp:
                snap = take_snapshot(querier_for(parent, task, meta.end_date))
                routes = aggregate_routes(snap)
                labels = {c["id"]: c["label"] for c in parent.call("cargos")}
                cands = [c for c in enumerate_routes(snap, routes, max_per_mode=cfg.max_air, cargo_labels=labels)
                         if c.mode == "air"][: cfg.max_air]
                rendered = render_state(snap, routes, cands)
                extras = candidate_extras(snap, cands, labels)
                nw0, money0 = _net_worth(parent)
                sp.set_attributes({"candidates": len(cands), "nw0": nw0})
            saves: dict[str, Path] = {HOLD: parent.save(tmp / "opt-hold.sav")}
            builds: dict[str, dict] = {HOLD: {"ok": True, "cost": 0, "failed_step": None}}
            aliases: set[str] = set()  # failed before spending anything: the world is the hold world
            for i, c in enumerate(cands):
                with tracing.span("label.build", option=c.candidate_id, kind="route") as sp:
                    child = parent.fork(tmp / f"build{i}")
                    try:
                        try:
                            with tracing.span("label.build_affordable", bridge=True):
                                br, used = build_affordable(child, c.a["id"], c.b["id"])
                            builds[c.candidate_id] = {"ok": br.ok, "cost": br.cost, "failed_step": br.failed_step,
                                                      "airport_type": br.airport_type, "planes": used[1],
                                                      "ladder": list(used)}
                            untouched = not br.ok and not br.cost and not br.steps
                        except BridgeError as e:
                            builds[c.candidate_id] = {"ok": False, "cost": None, "failed_step": f"raise:{e.code}"}
                            untouched = False
                        b = builds[c.candidate_id]
                        sp.set_attributes({"ok": b["ok"], "cost": b["cost"] or 0, "failed_step": b["failed_step"] or "",
                                           "alias": untouched})
                        if untouched:
                            aliases.add(c.candidate_id)
                        else:
                            saves[c.candidate_id] = child.save(tmp / f"opt-{i}.sav")
                    finally:
                        child.close()
            exps = expansion_candidates(snap, routes, parent.call("depots"), cfg.max_expand) if cfg.max_expand else []
            for i, e in enumerate(exps):
                with tracing.span("label.build", option=e.key, kind="expand") as sp:
                    child = parent.fork(tmp / f"expand{i}")
                    try:
                        with tracing.span("label.expand_affordable"):
                            r = expand_affordable(child, e)
                        builds[e.key] = {"ok": r["ok"], "cost": r["cost"], "failed_step": r["failed_step"],
                                         "planes": r["planes"], "kind": "expand"}
                        sp.set_attributes({"ok": r["ok"], "cost": r["cost"] or 0, "planes": r["planes"] or 0,
                                           "alias": not r["planes"] and not r["cost"]})
                        if not r["planes"] and not r["cost"]:
                            aliases.add(e.key)
                        else:
                            saves[e.key] = child.save(tmp / f"opt-x{i}.sav")
                    finally:
                        child.close()
        finally:
            parent.close()
        t_build = time.perf_counter() - t0
        gs_calls, gs_ms = tracing.bridge_since(gs0)  # GameScript round trips of the build phase (not rollouts)
        options = [c.candidate_id for c in cands] + [e.key for e in exps] + [HOLD]
        if len(options) > 1 and len(aliases) == len(options) - 1:
            # Nothing could be built, so every option is the hold world: no signal, no rollouts.
            return {"version": cfg.version, "config": asdict(cfg), "state": asdict(meta), "seeds": seeds,
                    "nw0": nw0, "money0": money0, "rendered_state": rendered,
                    "candidates": [{**dataclasses.asdict(c), **extras.get(c.candidate_id, {})} for c in cands],
                    "expansions": as_dicts(exps), "options": options, "uninformative": "no_build",
                    "builds": builds, "timing": _timing(t_build, t_build, gs_calls, gs_ms,
                                                        tracing.server_starts() - starts0)}
        # No need to simulate past the end of the game (horizon + run-off year), where the arena scores.
        left = meta.end_date + 365 - meta.date
        chunks = max(1, min(cfg.chunks, -(-left // cfg.chunk_days)))
        tracing.annotate(options=len(options), simulated=len(saves), aliases=len(aliases), chunks=chunks)
        outcomes: dict[str, Any] = {}
        for key, save in saves.items():
            outcomes[key] = {**builds[key], "rollouts": [
                _rollout(save, tmp / f"roll-{key}-{m}", s, cfg, chunks, key=key) for m, s in enumerate(seeds)]}
        for key in aliases:  # same world and same seeds as hold: the rollouts would repeat hold's exactly
            outcomes[key] = {**builds[key], "alias": HOLD, "rollouts": outcomes[HOLD]["rollouts"]}
    return {
        "version": cfg.version, "config": asdict(cfg), "state": asdict(meta), "seeds": seeds,
        "nw0": nw0, "money0": money0, "rendered_state": rendered,
        "candidates": [{**dataclasses.asdict(c), **extras.get(c.candidate_id, {})} for c in cands],
        "expansions": as_dicts(exps), "options": options,
        "outcomes": outcomes,
        "timing": _timing(t_build, time.perf_counter() - t0, gs_calls, gs_ms, tracing.server_starts() - starts0),
    }


def _timing(build_s: float, total_s: float, gs_calls: int, gs_ms: float, server_starts: int) -> dict:
    """The record's ``timing``: wall seconds of the build phase and of the job, the GameScript round trips of
    the build phase, and the OpenTTD processes the whole job started (a pooled reload is not a start)."""
    return {"build_s": round(build_s, 2), "total_s": round(total_s, 2), "bridge_calls": gs_calls,
            "bridge_ms": round(gs_ms, 1), "server_starts": server_starts}


def _write_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj))
    tmp.rename(path)


CLAIM_TTL_S = 3600


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(out_dir: Path, state_id: str) -> bool:
    """Atomically claim a state so concurrent labellers never duplicate work. A claim whose worker process is
    gone (a stopped or crashed labeller) or that is older than ``CLAIM_TTL_S`` is taken over."""
    d = Path(out_dir) / ".claims"
    d.mkdir(exist_ok=True)
    path = d / state_id
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(path.read_text() or 0)
            fresh = time.time() - path.stat().st_mtime < CLAIM_TTL_S
        except (OSError, ValueError):
            pid, fresh = 0, False
        if fresh and pid and _alive(pid):
            return False
        path.unlink(missing_ok=True)
        return claim(out_dir, state_id)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def _job(args: tuple) -> dict:
    root, state_id, out_dir, cfg_d, *rest = args  # rest: the labeller run's traceparent
    cfg = LabelConfig(**cfg_d)
    dest = Path(out_dir) / f"{state_id}.json"
    if dest.exists() or not claim(Path(out_dir), state_id):
        return {"state_id": state_id, "skipped": True}
    try:
        with tracing.attach(rest[0] if rest else ""), tracing.span("label.job", state_id=state_id) as sp:
            try:
                rec = label_state(Path(root), state_id, cfg)
                _write_atomic(dest, rec)
                sp.set_attributes({"ok": True, "options": len(rec["options"]),
                                   "uninformative": rec.get("uninformative", "")})
                return {"state_id": state_id, "ok": True, "s": rec["timing"]["total_s"],
                        "options": len(rec["options"])}
            except Exception as e:  # noqa: BLE001 - one bad state must not stop the run; the error file lets a rerun retry it
                sp.record_exception(e)
                sp.set_attributes({"ok": False, "error": repr(e)[:500]})
                _write_atomic(Path(out_dir) / f"{state_id}.error.json",
                              {"state_id": state_id, "error": repr(e), "trace": traceback.format_exc()[-4000:]})
                return {"state_id": state_id, "ok": False, "error": repr(e)}
    finally:
        tracing.flush()


def label_all(root: Path, out_dir: Path, workers: int = 8, cfg: LabelConfig = LabelConfig(),
              only: list[str] | None = None, order_seed: int = 0) -> dict:
    """Label every state in ``root`` without a record in ``out_dir``, ``workers`` states at a time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in (only or list_states(root)) if not (out_dir / f"{s}.json").exists()]
    # A fixed shuffle, so every split and source fills at the same rate instead of in name order; states the
    # model reached in its own play (DAgger) go first, because they are the distribution it must get right.
    random.Random(order_seed).shuffle(todo)
    todo.sort(key=lambda s: ".selfplay-model-" not in s)
    progress = out_dir / "_progress.log"
    done = failed = 0
    t0 = time.perf_counter()
    tracing.annotate(todo=len(todo), workers=workers)
    tp = tracing.traceparent()
    jobs = [(str(root), s, str(out_dir), asdict(cfg), tp) for s in todo]
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(_job, jobs):
            done += bool(r.get("ok"))
            failed += r.get("ok") is False
            if r.get("skipped"):
                continue
            line = json.dumps({**r, "done": done, "failed": failed, "todo": len(todo),
                               "elapsed_s": round(time.perf_counter() - t0, 1)})
            with progress.open("a") as f:
                f.write(line + "\n")
            log.info(line)
    return {"labelled": done, "failed": failed, "todo": len(todo), "elapsed_s": round(time.perf_counter() - t0, 1)}
