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
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from railhead.rewards import tycoon_log
from railhead.task import TaskSpec
from railhead.world import World

from rh17m import tracing
from rh17m.dataset import HOLD, DatasetConfig, observation
from rh17m.expand import apply_option, expansion_candidates
from rh17m.labels import LabelConfig, candidate_extras, querier_for
from rh17m.states import find_task

Policy = Callable[[dict], dict[str, float]]


def wakeup_observation(world: World, task: TaskSpec, end_date: int, max_air: int,
                       dcfg: DatasetConfig, max_expand: int = 3) -> tuple[dict, list, list]:
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
    exps = (expansion_candidates(snap, routes, world.call("depots"), max_expand)
            if dcfg.expansions and max_expand else [])
    obs = observation(render_state(snap, routes, cands), cand_dicts, money0, dcfg, [asdict(e) for e in exps])
    return obs, cands, exps


PLAYER_V = "play-v3"  # v2: fall back on no-spend failures, block failed routes; v3: expansion options


def wakeup_options(obs: dict, blocked: set[str]) -> dict:
    """The observation without routes that already failed in this game (a reloaded game would offer them
    again, and a deterministic policy would pick them again)."""
    if not blocked:
        return obs
    return {**obs, "options": [(k, t) for k, t in obs["options"] if k not in blocked],
            "features": {k: f for k, f in obs["features"].items() if k not in blocked}}


def play(task_id: str, policy: Policy, seed: int, wake_days: int = 90, max_air: int = LabelConfig.max_air,
         dcfg: DatasetConfig = DatasetConfig(), max_tries: int = 3, max_wakes: int | None = None) -> dict:
    """Play one game. At each wake-up the options are ranked by the policy's probabilities (ties broken at
    random); the first is built, and when a build fails without spending anything (no site, type not
    available, no cash) the next is tried, up to ``max_tries``, stopping at ``hold``. Failed routes are
    blocked for the rest of the game. Every policy, scripted or learned, plays under these rules.

    A policy that needs the live world (the oracle) has a ``bind`` method, called before it is asked; its
    ``decision_record()`` is added to the wake-up's record and its ``game_meta(wake_days)`` to the game's.
    ``max_wakes`` stops after that many wake-ups (a smoke test): no run-off, no score, ``partial`` set."""
    import random

    task_dir = find_task(task_id)
    task = TaskSpec.load(task_dir)
    rng = random.Random(seed)
    t0 = time.perf_counter()
    decisions, curve = [], []
    blocked: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="rh17m-play-") as tmp:
        world = World.load(Path(tmp) / "w", task_dir / "start.sav")
        try:
            world.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
            end_date = int(world.state()["date"]) + task.horizon_days
            policy_s = 0.0
            while int(world.state()["date"]) < end_date:
                obs, cands, exps = wakeup_observation(world, task, end_date, max_air, dcfg)
                obs = wakeup_options(obs, blocked)
                if hasattr(policy, "bind"):
                    policy.bind(world=world, task=task, task_id=task_id, seed=seed, wake=len(decisions),
                                end_date=end_date, cands=cands, exps=exps)
                t1 = time.perf_counter()
                with tracing.span("play.policy", options=len(obs["options"])):
                    probs = policy(obs)
                policy_s += time.perf_counter() - t1
                order = sorted(probs, key=lambda k: (-probs[k], k != HOLD, rng.random()))
                rec = {"date": world.state()["ymd"], "pick": order[0], "p": round(probs[order[0]], 4),
                       "options": len(obs["options"]), "tries": []}
                if hasattr(policy, "decision_record"):
                    rec.update(policy.decision_record())
                for key in order[:max_tries]:
                    if key == HOLD:
                        break
                    with tracing.span("play.build", option=key, bridge=True) as sp:
                        r = apply_option(world, key, cands, exps)
                        sp.set_attributes({"kind": r["kind"], "ok": r["ok"], "cost": r["cost"] or 0})
                    rec["tries"].append({"key": key, "kind": r["kind"], "ok": r["ok"], "cost": r["cost"],
                                         "ladder": r.get("ladder"), "planes": r.get("planes"),
                                         "failed": r["failed_step"]})
                    if r["ok"] or r["cost"] or r["steps"]:  # built, or spent money: the decision is made
                        rec.update(pick=key, kind=r["kind"], ok=r["ok"], cost=r["cost"], failed=r["failed_step"])
                        if not r["ok"] and r["kind"] == "route":
                            blocked.add(key)
                        break
                    if r["kind"] == "route":  # an expansion refused for cash can succeed later
                        blocked.add(key)
                rec["played"] = rec["pick"] if "kind" in rec else HOLD  # the walk may end at hold
                decisions.append(rec)
                days = min(wake_days, end_date - int(world.state()["date"]))
                world.step(days=max(days, 1))
                c = world.call("company")
                curve.append(int(c.get("money") or 0) - int(c.get("loan") or 0))
                if max_wakes is not None and len(decisions) >= max_wakes:
                    break
            partial = max_wakes is not None and int(world.state()["date"]) < end_date
            if not partial:
                world.step(days=task.runoff_days)
            c = world.call("company")
            final = int(c.get("money") or 0) - int(c.get("loan") or 0)
        finally:
            world.close()
    obj = task.objective
    meta = policy.game_meta(wake_days) if hasattr(policy, "game_meta") else {}
    return {"task": task_id, "seed": seed, "player": PLAYER_V, "wake_days": wake_days, **meta,
            **({"partial": True, "wakes": len(decisions)} if partial else {}), "final_nw": final,
            "score": None if partial else tycoon_log(final, obj["w_idle"], obj["w_base"]),
            "builds": sum(len(d["tries"]) for d in decisions),
            "built_ok": sum(1 for d in decisions if d.get("ok")), "decisions": decisions, "curve": curve,
            "policy_s": round(policy_s, 2), "wall_s": round(time.perf_counter() - t0, 1)}


def game_path(out_dir: Path, policy_name: str, task_id: str, seed: int, wake_days: int,
              max_wakes: int | None = None) -> Path:
    """Where a game's record lives, which is also its cache key. The oracle's name carries every setting its
    result depends on (wake interval, player, label version and config, simulator stack: ``oracle.game_key``),
    so a game played under other settings is never reused; other policies keep their historical names."""
    if policy_name != "oracle":
        return Path(out_dir) / f"{policy_name}.{task_id}.s{seed}.json"
    from rh17m.oracle import game_key, key_tag

    part = f".m{max_wakes}" if max_wakes is not None else ""
    return Path(out_dir) / f"oracle.{task_id}.s{seed}.w{wake_days}{part}.{key_tag(game_key(wake_days))}.json"


def _job(args: tuple) -> dict:
    # rest: the caller's traceparent, then max_wakes
    task_id, policy_name, seed, wake_days, out_dir, model_dir, *rest = args
    max_wakes = rest[1] if len(rest) > 1 else None
    dest = game_path(Path(out_dir), policy_name, task_id, seed, wake_days, max_wakes)
    if dest.exists():
        res = json.loads(dest.read_text())
        if policy_name == "oracle":
            from rh17m.oracle import game_key

            stale = {k: (res.get(k), v) for k, v in game_key(wake_days).items() if res.get(k) != v}
            if stale:
                raise RuntimeError(f"{dest} does not match this run's settings: {stale}")
        return res
    try:
        with tracing.attach(rest[0] if rest else ""), tracing.span(
                "play.game", task=task_id, policy=policy_name, seed=seed, wake_days=wake_days,
                model=model_dir or "") as sp:
            policy = make_policy(policy_name, model_dir)
            if policy_name == "oracle":
                from rh17m.labels import fast_saves

                try:  # one failed oracle game (hours of work for the others in flight) must not stop the run
                    with fast_saves():  # it saves the live world once per option: uncompressed is ~20x faster
                        res = {"policy": policy_name, **play(task_id, policy, seed, wake_days, max_wakes=max_wakes)}
                except Exception as e:  # noqa: BLE001 - recorded beside the game; a rerun retries it
                    import traceback

                    sp.record_exception(e)
                    err = {"policy": policy_name, "task": task_id, "seed": seed, "error": repr(e),
                           "trace": traceback.format_exc()[-4000:]}
                    dest.with_name(dest.name.replace(".json", ".error.json")).write_text(json.dumps(err))
                    return err
            else:
                res = {"policy": policy_name, **play(task_id, policy, seed, wake_days, max_wakes=max_wakes)}
            sp.set_attributes({k: res[k] for k in ("final_nw", "score", "builds", "built_ok", "policy_s")
                               if res[k] is not None})
    finally:
        tracing.flush()
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(dest)
    return res


def make_policy(name: str, model_dir: str | None = None) -> Policy:
    import random

    from rh17m.evaluate import BASELINES

    if name == "oracle":  # the simulator ceiling (not BASELINES["oracle"], which reads offline labels)
        from rh17m.oracle import OraclePolicy

        return OraclePolicy()
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
               workers: int = 4, model_dir: str | None = None, max_wakes: int | None = None):
    """Every (task, policy, seed) game, resumable (one JSON per game in ``out_dir``)."""
    import multiprocessing as mp

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tp = tracing.traceparent()
    jobs = [(t, p, s, wake_days, str(out_dir), model_dir, tp, max_wakes)
            for s in seeds for t in task_ids for p in policies]
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
    import math

    from rh17m.evaluate import bootstrap_diff

    out = {"tasks": sorted(shared), "policies": {}}
    lognw = {label: [statistics.mean(math.asinh(g["final_nw"] / 1e5) for g in by_task[t]) for t in sorted(shared)]
             for label, by_task in games.items()}
    for label, by_task in games.items():
        scores = [statistics.mean(g["score"] for g in by_task[t]) for t in sorted(shared)]
        nws = [statistics.mean(g["final_nw"] for g in by_task[t]) for t in sorted(shared)]
        row = {"mean_score": round(statistics.mean(scores), 4) if scores else None,
               "median_nw": round(statistics.median(nws)) if nws else None,
               "mean_asinh_nw": round(statistics.mean(lognw[label]), 4) if shared else None,
               "games": sum(len(by_task[t]) for t in shared)}
        if "random" in lognw and label != "random" and len(shared) > 1:
            d, lo, hi = bootstrap_diff(lognw[label], lognw["random"])
            row["vs_random_asinh_nw"] = [round(d, 4), round(lo, 4), round(hi, 4)]
        out["policies"][label] = row
    return out


def _days_between(a: str, b: str) -> int:
    from datetime import date

    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def wake_days_of(rec: dict) -> int | None:
    """The record's wake interval: recorded (newer records), else read off its decision dates (the most
    common gap between consecutive wake-ups; the last one may be shorter, the end of the horizon)."""
    if rec.get("wake_days"):
        return int(rec["wake_days"])
    gaps = [_days_between(a["date"], b["date"]) for a, b in zip(rec["decisions"], rec["decisions"][1:])]
    return max(set(gaps), key=gaps.count) if gaps else None


def load_games(run_dirs: list[Path]) -> dict[str, dict[tuple[str, int], dict]]:
    """Game records of the given run folders (one folder = the games of one run, as ``play --out`` writes
    them), by label (the policy; ``model:<folder>`` for a model) and (task, seed). A label playing the same
    (task, seed) twice, or a partial (smoke-test) game, is an error: either would make the pairing ambiguous."""
    games: dict[str, dict[tuple[str, int], dict]] = {}
    for d in run_dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name == "summary.json" or f.name.endswith(".error.json") or f.name.startswith("."):
                continue
            r = json.loads(f.read_text())
            if r.get("partial"):
                raise ValueError(f"{f} is a partial game (max_wakes); it cannot enter a comparison")
            label = r["policy"] if r["policy"] != "model" else f"model:{Path(d).name}"
            key = (r["task"], int(r["seed"]))
            if key in games.setdefault(label, {}):
                raise ValueError(f"{label} played {key} twice (second: {f})")
            games[label][key] = r
    return games


def oracle_decisions(records: list[dict]) -> dict:
    """What the oracle's own wake-ups say about the decision problem: per decision, the spread between the
    best option and the mean option (what a uniform random pick gives up, in asinh net worth) and between the
    best and hold; how often the best option is hold, an expansion or a new route; how often the played
    option built; and whether the fork's build of a played option matched the live build (same ok and cost)."""
    import statistics

    spread, over_hold, kinds, built, checked, mismatched = [], [], {"hold": 0, "expand": 0, "route": 0}, 0, 0, 0
    n = uninformative = played_best = second_tries = 0
    for r in records:
        for d in r["decisions"]:
            v = d.get("values")
            if not v:
                continue
            if d.get("uninformative"):  # nothing buildable: placeholder values, no decision to speak of
                uninformative += 1
                continue
            n += 1
            played_best += d.get("played", d.get("pick")) == d.get("best")
            second_tries += len(d.get("tries", [])) > 1
            best = max(v, key=lambda k: (v[k], k == "hold"))
            spread.append(v[best] - statistics.mean(v.values()))
            over_hold.append(v[best] - v["hold"])
            kinds["hold" if best == "hold" else "expand" if best.startswith("expand-") else "route"] += 1
            built += bool(d.get("ok"))
            for t in d.get("tries", []):
                fb = (d.get("fork_builds") or {}).get(t["key"])
                if fb is not None:
                    checked += 1
                    mismatched += (bool(fb["ok"]), fb["cost"] or 0) != (bool(t["ok"]), t["cost"] or 0)
    if not n:
        return {"decisions": 0, "uninformative": uninformative}

    def q(xs, p):
        return round(sorted(xs)[min(len(xs) - 1, int(p * len(xs)))], 4)

    return {"decisions": n, "best_minus_mean": {"mean": round(statistics.mean(spread), 4), "median": q(spread, 0.5),
                                                "p90": q(spread, 0.9)},
            "best_minus_hold": {"mean": round(statistics.mean(over_hold), 4), "median": q(over_hold, 0.5)},
            "best_is": {k: round(c / n, 4) for k, c in kinds.items()}, "played_built": round(built / n, 4),
            "uninformative": uninformative, "played_best": round(played_best / n, 4),
            # an alias ties hold and loses to it, so a second try means the fork and the live world disagreed
            "second_tries": second_tries,
            "fork_vs_live": {"tries_checked": checked, "mismatched": mismatched}}


def compare(run_dirs: list[Path], refs: list[str] | None = None, labels: list[str] | None = None,
            n_boot: int = 2000) -> dict:
    """Closed-loop comparison across run folders (``rh17m play-report --runs``): games are paired on the
    (task, seed) pairs every compared label played; asinh(final net worth / 100k) is averaged per map over
    its seeds, and every label is compared with every ``ref`` (default: every other label) by the mean
    per-map difference and a 95% bootstrap interval that resamples maps (``evaluate.bootstrap_diff``)."""
    import statistics

    from rh17m.evaluate import bootstrap_diff

    games = load_games(run_dirs)
    if labels:
        missing = [lab for lab in labels if lab not in games]
        if missing:
            raise ValueError(f"no games for {missing}; found {sorted(games)}")
        games = {lab: games[lab] for lab in labels}
    if not games:
        return {}
    keys = sorted(set.intersection(*(set(g) for g in games.values())))
    unpaired = {lab: sorted(f"{t}.s{s}" for t, s in set(g) - set(keys)) for lab, g in games.items()}
    tasks = sorted({t for t, _ in keys})
    per_map = {lab: [statistics.mean(math.asinh(g[k]["final_nw"] / 1e5) for k in keys if k[0] == t) for t in tasks]
               for lab, g in games.items()}
    out: dict = {"maps": len(tasks), "games_per_label": len(keys), "seeds": sorted({s for _, s in keys}),
                 "tasks": tasks, "unpaired": {k: v for k, v in unpaired.items() if v}, "policies": {}, "warnings": []}
    for lab, g in games.items():
        rs = [g[k] for k in keys]
        row = {"mean_asinh_nw": round(statistics.mean(per_map[lab]), 4) if tasks else None,
               "mean_score": round(statistics.mean(r["score"] for r in rs), 4) if rs else None,
               "median_nw": round(statistics.median(r["final_nw"] for r in rs)) if rs else None,
               "games": len(rs), "players": sorted({r.get("player", "?") for r in rs}),
               "wake_days": sorted({wake_days_of(r) or 0 for r in rs}), "vs": {}}
        for ref in (refs or [x for x in games if x != lab]):
            if ref != lab and ref in games and len(tasks) > 1:
                d, lo, hi = bootstrap_diff(per_map[lab], per_map[ref], n=n_boot)
                row["vs"][ref] = {"diff": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        if lab == "oracle":
            row["decisions"] = oracle_decisions(rs)
        out["policies"][lab] = row
    if len({tuple(r["players"]) for r in out["policies"].values()}) > 1:
        out["warnings"].append("labels were played by different player versions")
    if len({tuple(r["wake_days"]) for r in out["policies"].values()}) > 1:
        out["warnings"].append("labels were played at different wake intervals")
    return out


def format_compare(out: dict) -> str:
    """The comparison as a table: one row per label, one column per reference."""
    pol = out.get("policies", {})
    refs = sorted({r for v in pol.values() for r in v["vs"]})
    lines = [(f"{out.get('maps', 0)} maps x seeds {out.get('seeds')}: {out.get('games_per_label', 0)} paired games "
             "per label; diff = mean per-map asinh(nw/100k) difference [95% map bootstrap]")]
    lines.append(" | ".join(["label", "asinh_nw", "score", "median_nw"] + [f"vs {r}" for r in refs]))
    for lab, v in sorted(pol.items(), key=lambda kv: -(kv[1]["mean_asinh_nw"] or 0)):
        cells = [lab, f"{v['mean_asinh_nw']:.3f}", f"{v['mean_score']:.3f}", f"£{v['median_nw']:,}"]
        for r in refs:
            x = v["vs"].get(r)
            cells.append(f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f},{x['ci95'][1]:+.3f}]" if x else "-")
        lines.append(" | ".join(cells))
    dec = pol.get("oracle", {}).get("decisions")
    if dec:
        lines.append(f"oracle decisions: {json.dumps(dec)}")
    lines += [f"unpaired (left out) {lab}: {', '.join(v)}" for lab, v in out.get("unpaired", {}).items()]
    lines += [f"WARNING: {w}" for w in out.get("warnings", [])]
    return "\n".join(lines)


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
        if not hasattr(Tracker, "_call"):  # PRs #7-#11
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


def run_name(model_dir: str) -> str:
    """runs/train/<run>/model -> <run>; a comma-separated ensemble -> 'ens-' + the first run."""
    first = Path(str(model_dir).split(",")[0])
    name = first.parent.name if first.name == "model" else first.name
    return ("ens-" if "," in str(model_dir) else "") + name


def gate(candidate: str, reference: str, tasks: list[str], seeds: list[int], root: Path, wake_days: int = 180,
         workers: int = 3) -> dict:
    """Closed-loop gate: play both models on ``tasks`` x ``seeds`` (games already on disk are reused) and
    compare mean asinh(final net worth) per map, paired, with a bootstrap interval. The candidate is rejected
    only when it is significantly worse (interval entirely below zero)."""
    import math
    import statistics

    from rh17m.evaluate import bootstrap_diff

    per_map = {}
    for model in (candidate, reference):
        out = Path(root) / run_name(model)
        for _ in tournament(tasks, ["model"], seeds, out, wake_days=wake_days, workers=workers, model_dir=model):
            pass
        games: dict[str, list[float]] = {}
        for f in out.glob("model.*.json"):
            r = json.loads(f.read_text())
            if r["task"] in tasks and r["seed"] in seeds:
                games.setdefault(r["task"], []).append(math.asinh(r["final_nw"] / 1e5))
        per_map[model] = {t: statistics.mean(v) for t, v in games.items()}
    shared = sorted(set(per_map[candidate]) & set(per_map[reference]))
    d, lo, hi = bootstrap_diff([per_map[candidate][t] for t in shared], [per_map[reference][t] for t in shared])
    return {"candidate": run_name(candidate), "reference": run_name(reference), "maps": len(shared),
            "seeds": seeds, "diff_asinh_nw": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
            "verdict": "reject" if hi < 0 else "accept"}
