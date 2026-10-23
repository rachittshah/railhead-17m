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
from rh17m.labels import LabelConfig, build_affordable, candidate_extras, querier_for
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
    extras = candidate_extras(snap, cands, labels)
    cand_dicts = [{**asdict(c), **extras.get(c.candidate_id, {})} for c in cands]
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


def report(play_root: Path) -> dict:
    """Mean arena score and net worth per policy over the tasks every policy has played (paired), from all
    run folders under ``play_root``."""
    import statistics

    games: dict[str, dict[str, list[dict]]] = {}
    for f in Path(play_root).glob("*/*.json"):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text())
        label = r["policy"] if r["policy"] != "model" else f"model:{f.parent.name}"
        games.setdefault(label, {}).setdefault(r["task"], []).append(r)
    if not games:
        return {}
    shared = set.intersection(*(set(v) for v in games.values()))
    out = {"tasks": sorted(shared), "policies": {}}
    for label, by_task in games.items():
        scores = [statistics.mean(g["score"] for g in by_task[t]) for t in sorted(shared)]
        nws = [statistics.mean(g["final_nw"] for g in by_task[t]) for t in sorted(shared)]
        out["policies"][label] = {"mean_score": round(statistics.mean(scores), 4) if scores else None,
                                  "median_nw": round(statistics.median(nws)) if nws else None,
                                  "games": sum(len(by_task[t]) for t in shared)}
    return out


PALETTE = [(47, 107, 79), (201, 130, 27), (58, 100, 168), (196, 90, 44), (122, 82, 160), (90, 90, 90)]


def curves_image(play_root: Path, task: str, path: Path, width: int = 900, height: int = 420) -> Path | None:
    """PNG of net worth over the game for every policy that played ``task`` (asinh scale, so £-50k and £50M
    share an axis). Drawn with PIL, which Trackio already requires."""
    import math

    from PIL import Image, ImageDraw

    series = {}
    for f in sorted(Path(play_root).glob(f"*/*.{task}.s*.json")):
        r = json.loads(f.read_text())
        label = r["policy"] if r["policy"] != "model" else f"model:{f.parent.name}"
        if r.get("curve"):
            series[label] = [math.asinh(v / 1e5) for v in r["curve"]]
    if not series:
        return None
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    pad_l, pad_r, pad_t, pad_b = 60, 180, 30, 40
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    hi = hi if hi > lo else lo + 1
    n = max(len(v) for v in series.values())
    d.rectangle([pad_l, pad_t, width - pad_r, height - pad_b], outline=(200, 205, 200))
    for tick in range(math.floor(lo), math.ceil(hi) + 1):
        y = height - pad_b - (tick - lo) / (hi - lo) * (height - pad_t - pad_b)
        if pad_t <= y <= height - pad_b:
            d.line([pad_l, y, width - pad_r, y], fill=(235, 238, 235))
            d.text((6, y - 6), f"{math.sinh(tick) * 1e5 / 1e6:,.2f}M", fill=(90, 100, 95))
    for i, (label, vals) in enumerate(sorted(series.items())):
        color = PALETTE[i % len(PALETTE)]
        pts = [(pad_l + j / max(1, n - 1) * (width - pad_l - pad_r),
                height - pad_b - (v - lo) / (hi - lo) * (height - pad_t - pad_b)) for j, v in enumerate(vals)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=3)
        d.text((width - pad_r + 12, pad_t + 18 * i), label, fill=color)
    d.text((pad_l, 8), f"{task}: net worth (GBP) at each wake-up, asinh scale", fill=(24, 35, 28))
    img.save(path)
    return Path(path)


def track_report(play_root: Path, summary: dict) -> None:
    """Closed-loop results into Trackio project railhead-17m-play: per-policy and per-task tables, net-worth
    curve images per task, and a Markdown report."""
    import tempfile
    import time

    try:
        from decision_mlx.tracking import Tracker
        if not hasattr(Tracker, "log_table"):
            raise ImportError
    except ImportError:
        from rh17m._tracking import Tracker

    t = Tracker("railhead-17m-play", name=time.strftime("closed-loop-%m%d-%H%M"))
    pol = summary.get("policies", {})
    t.log_table("closed_loop/policies", ["policy", "mean_score", "median_net_worth", "games"],
                [[k, v["mean_score"], v["median_nw"], v["games"]] for k, v in sorted(pol.items(), key=lambda kv: -kv[1]["mean_score"])],
                step=0)
    rows = []
    for f in sorted(Path(play_root).glob("*/*.json")):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text())
        rows.append([r["policy"] if r["policy"] != "model" else f"model:{f.parent.name}", r["task"], r["seed"],
                     round(r["score"], 4), r["final_nw"], r["builds"], r["built_ok"]])
    t.log_table("closed_loop/games", ["policy", "task", "seed", "score", "final_net_worth", "builds", "built_ok"],
                rows, step=0)
    with tempfile.TemporaryDirectory(prefix="rh17m-curves-") as tmp:
        for task in summary.get("tasks", []):
            png = curves_image(play_root, task, Path(tmp) / f"{task}.png")
            if png is not None:
                t.log_image(f"curves/{task}", str(png), caption=f"net worth by policy on {task}", step=0)
    lines = ["# Closed loop on held-out maps", "", f"Tasks: {', '.join(summary.get('tasks', []))}", "",
             "| policy | mean arena score | median final net worth | games |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean_score']:.4f} | £{v['median_nw']:,} | {v['games']} |"
              for k, v in sorted(pol.items(), key=lambda kv: -kv[1]["mean_score"])]
    t.log_report("closed_loop/report", "\n".join(lines), step=0)
    best = max(pol.items(), key=lambda kv: kv[1]["mean_score"])[0] if pol else None
    models = {k: v for k, v in pol.items() if k.startswith("model:")}
    if models and best:
        level = "info" if best.startswith("model:") else "warn"
        t.alert(f"closed loop: best policy is {best}", f"mean scores {json.dumps({k: v['mean_score'] for k, v in pol.items()})}",
                level=level)
    t.finish()
