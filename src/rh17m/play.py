"""Closed-loop play: a route policy runs a whole company, and the game's own score judges it.

At every wake-up (``wake_days`` apart) the player reads the wake-up snapshot, enumerates the air candidates
exactly as the label engine does, asks the policy for probabilities over the options (the same
``dataset.observation`` text the model was trained on), builds the argmax with ``labels.build_affordable``
unless it is ``hold``, and waits. After the horizon the run-off year passes with no actions, and the final net
worth is scored with Railhead's arena scale (``railhead.rewards.tycoon_log``).

A policy is any ``Callable[[dict], dict[str, float]]`` over an observation; ``evaluate.BASELINES`` (except the
oracle, which needs labels) and the trained model both fit.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from railhead.rewards import tycoon_log
from railhead.task import TaskSpec
from railhead.world import World

from rh17m.dataset import HOLD, DatasetConfig, observation
from rh17m.labels import LabelConfig, build_affordable, querier_for
from rh17m.states import find_task

Policy = Callable[[dict], dict[str, float]]


def wakeup_observation(world: World, task: TaskSpec, end_date: int, max_air: int,
                       dcfg: DatasetConfig) -> tuple[dict, list]:
    from rhs1.candidates import enumerate_routes
    from rhs1.snapshot import aggregate_routes, render_state, take_snapshot

    snap = take_snapshot(querier_for(world, task, end_date))
    routes = aggregate_routes(snap)
    labels = {c["id"]: c["label"] for c in world.call("cargos")}
    cands = [c for c in enumerate_routes(snap, routes, max_per_mode=max_air, cargo_labels=labels)
             if c.mode == "air"][:max_air]
    money0 = int(world.call("company").get("money") or 0)
    cand_dicts = [asdict(c) for c in cands]
    return observation(render_state(snap, routes, cands), cand_dicts, money0, dcfg), cands


def play(task_id: str, policy: Policy, seed: int, wake_days: int = 90, max_air: int = LabelConfig.max_air,
         dcfg: DatasetConfig = DatasetConfig()) -> dict:
    task_dir = find_task(task_id)
    task = TaskSpec.load(task_dir)
    t0 = time.perf_counter()
    decisions, curve = [], []
    with tempfile.TemporaryDirectory(prefix="rh17m-play-") as tmp:
        world = World.load(Path(tmp) / "w", task_dir / "start.sav")
        try:
            world.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
            end_date = int(world.state()["date"]) + task.horizon_days
            policy_s = 0.0
            while int(world.state()["date"]) < end_date:
                obs, cands = wakeup_observation(world, task, end_date, max_air, dcfg)
                t1 = time.perf_counter()
                probs = policy(obs)
                policy_s += time.perf_counter() - t1
                pick = max(probs, key=lambda k: (probs[k], k == HOLD))
                rec = {"date": world.state()["ymd"], "pick": pick, "p": round(probs[pick], 4),
                       "options": len(obs["options"])}
                if pick != HOLD:
                    c = next(c for c in cands if c.candidate_id == pick)
                    br, used = build_affordable(world, c.a["id"], c.b["id"])
                    rec.update(ok=br.ok, cost=br.cost, ladder=list(used), failed=br.failed_step)
                decisions.append(rec)
                days = min(wake_days, end_date - int(world.state()["date"]))
                world.step(days=max(days, 1))
                c = world.call("company")
                curve.append(int(c.get("money") or 0) - int(c.get("loan") or 0))
            world.step(days=task.runoff_days)
            c = world.call("company")
            final = int(c.get("money") or 0) - int(c.get("loan") or 0)
        finally:
            world.close()
    obj = task.objective
    return {"task": task_id, "seed": seed, "final_nw": final,
            "score": tycoon_log(final, obj["w_idle"], obj["w_base"]),
            "builds": sum(1 for d in decisions if d["pick"] != HOLD),
            "built_ok": sum(1 for d in decisions if d.get("ok")), "decisions": decisions, "curve": curve,
            "policy_s": round(policy_s, 2), "wall_s": round(time.perf_counter() - t0, 1)}


def _job(args: tuple) -> dict:
    task_id, policy_name, seed, wake_days, out_dir, model_dir = args
    dest = Path(out_dir) / f"{policy_name}.{task_id}.s{seed}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    policy = make_policy(policy_name, model_dir)
    res = {"policy": policy_name, **play(task_id, policy, seed, wake_days)}
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(dest)
    return res


def make_policy(name: str, model_dir: str | None = None) -> Policy:
    import random

    from rh17m.evaluate import BASELINES

    if name == "model":
        from rh17m.policy import ModelPolicy

        return ModelPolicy(Path(model_dir))
    if name == "random":
        rng = random.Random(0)

        def pick_random(obs: dict) -> dict[str, float]:
            k = rng.choice([k for k, _ in obs["options"]])
            return {kk: float(kk == k) for kk, _ in obs["options"]}

        return pick_random
    return BASELINES[name]


def tournament(task_ids: list[str], policies: list[str], seeds: list[int], out_dir: Path, wake_days: int = 90,
               workers: int = 4, model_dir: str | None = None):
    """Every (task, policy, seed) game, resumable (one JSON per game in ``out_dir``)."""
    import multiprocessing as mp

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(t, p, s, wake_days, str(out_dir), model_dir) for s in seeds for t in task_ids for p in policies]
    with mp.get_context("spawn").Pool(workers) as pool:
        yield from pool.imap_unordered(_job, jobs)
