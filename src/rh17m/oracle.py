"""The closed-loop oracle: at every wake-up, value every option with the simulator and play the best one.

It is the ceiling of what imitating the labels can buy. At a wake-up ``play.play`` hands the policy the live
world and the options it offers (``bind``, then the call), exactly as for any other policy. The oracle then
labels that decision with the labeller's own machinery (``labels.build_options`` / ``labels.roll_options``):
every option is built on a fork of the live world (routes through the cash ladder, expansions through
``expand_affordable``), saved, and rolled forward under ``LabelConfig()`` (lbl-v4, ``expand`` continuation)
for 728 days or to the end of the game, whichever is first. Its "probabilities" are the option values,
asinh(net worth / 100k) at the end of the rollout, so the player's own walk plays options in descending value
with its fallback rules and ties going to hold.

Two differences from a label job, both deliberate:
- One rollout seed per wake-up, shared by every option (common random numbers), derived from (task, game
  seed, wake-up index, ``SALT``) and never from the live game's random stream: the oracle does not peek at
  the future the live game will actually have. The live game continues from the state it was in before the
  forks; the forks only read it (save), they never step it.
- The rollout ends at the task's real end (horizon + ``task.runoff_days``), where the labeller assumes a
  365-day run-off (tycoon tasks have 730). On arena maps (365-day run-off) the two are identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

from rh17m.labels import HOLD, LabelConfig, build_options, fast_saves, roll_options, rollout_chunks

ORACLE_V = "oracle-v1"
SALT = "oracle"
SCALE = 100_000.0  # DatasetConfig.scale: values are on the labels' asinh scale


def rollout_seed(task_id: str, game_seed: int, wake: int) -> int:
    """The rollout seed of one wake-up: a hash of who/when, independent of the live game's own seed."""
    return int(hashlib.sha256(f"{task_id}:{game_seed}:{wake}:{SALT}".encode()).hexdigest()[:8], 16)


def option_value(rollouts: list[dict], scale: float = SCALE) -> float:
    """Mean over rollouts of asinh(net worth / scale) at the end of the rollout."""
    return sum(math.asinh(r["nw"][-1] / scale) for r in rollouts) / len(rollouts)


def variant() -> dict:
    """What the simulator stack of this process is: rhs1 source and whether its site search is pipelined,
    the server pool, and the engine (stock unless RAILHEAD_OPENTTD names a variant; ``engine_bin``: the
    resolved binary with its size and mtime, so a rebuilt engine changes the cache key)."""
    from railhead.engine import find_openttd
    from railhead.world import World
    from rhs1.builders import air

    from rh17m import pool

    binary = find_openttd()  # the binary that actually runs, and which build of it
    st = binary.stat()
    return {"rhs1": air.__file__, "pipelined": hasattr(air, "_dry_runs") and hasattr(World, "call_many"),
            "pool": pool.installed(), "engine": os.environ.get("RAILHEAD_OPENTTD") or "stock",
            "engine_bin": f"{binary} {st.st_size} {int(st.st_mtime)}"}


def game_key(wake_days: int, cfg: LabelConfig = LabelConfig()) -> dict:
    """Everything an oracle game's result depends on besides (task, seed): the cache key of its record."""
    from rh17m.play import PLAYER_V

    return {"player": PLAYER_V, "oracle": ORACLE_V, "label_version": cfg.version, "label_config": cfg.__dict__,
            "wake_days": wake_days, "runoff": "task", **variant()}


def key_tag(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:10]


class OraclePolicy:
    """A ``play`` policy that needs the live world: ``play`` calls ``bind`` before every call."""

    def __init__(self, cfg: LabelConfig = LabelConfig()) -> None:
        if os.environ.get("RAILHEAD_OPENTTD"):
            raise RuntimeError("the oracle runs on the stock engine only (unset RAILHEAD_OPENTTD)")
        self.cfg = cfg
        self.ctx: dict = {}
        self.last: dict = {}

    def bind(self, *, world, task, task_id: str, seed: int, wake: int, end_date: int, cands: list,
             exps: list) -> None:
        self.ctx = {"world": world, "task": task, "task_id": task_id, "seed": seed, "wake": wake,
                    "end_date": end_date, "cands": cands, "exps": exps}

    def game_meta(self, wake_days: int) -> dict:
        return game_key(wake_days, self.cfg)

    def __call__(self, obs: dict) -> dict[str, float]:
        c = self.ctx
        if not c:
            raise RuntimeError("OraclePolicy needs play() to bind the live world first")
        t0 = time.perf_counter()
        keys = [k for k, _ in obs["options"]]
        cands = [x for x in c["cands"] if x.candidate_id in keys]  # blocked routes are not offered
        exps = [x for x in c["exps"] if x.key in keys]
        seed = rollout_seed(c["task_id"], c["seed"], c["wake"])
        world, task = c["world"], c["task"]
        date = int(world.state()["date"])
        chunks = rollout_chunks(self.cfg, c["end_date"], date, task.runoff_days)
        with fast_saves(), tempfile.TemporaryDirectory(prefix="rh17m-oracle-") as tmp:
            saves, builds, aliases = build_options(world, Path(tmp), cands, exps)
            uninformative = len(keys) > 1 and len(aliases) == len(keys) - 1
            if uninformative:  # nothing can be built: every option is the hold world
                values = {k: 0.0 for k in keys}
            else:
                outcomes = roll_options(Path(tmp), saves, builds, aliases, [seed], self.cfg, chunks)
                values = {k: option_value(outcomes[k]["rollouts"]) for k in keys}
        best = max(keys, key=lambda k: (values[k], k == HOLD))
        self.last = {"values": {k: round(v, 6) for k, v in values.items()}, "best": best, "rollout_seed": seed,
                     "chunks": chunks, "uninformative": uninformative, "aliases": sorted(aliases),
                     "fork_builds": {k: {f: b.get(f) for f in ("ok", "cost", "failed_step", "planes", "ladder")}
                                     for k, b in builds.items() if k != HOLD},
                     "oracle_s": round(time.perf_counter() - t0, 1)}
        return values

    def decision_record(self) -> dict:
        return self.last
