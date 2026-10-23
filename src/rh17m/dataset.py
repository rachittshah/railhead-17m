"""Training examples from simulator label records.

One example = one decision: the rendered wake-up state, the question "which air route to build now", the
options (candidate routes plus ``hold``) as text, and the simulator's verdict per option.

Value of an option: the mean over seeds of ``asinh(net_worth / scale)`` at the end of the rollout. asinh is
log-like for large values and linear around zero, so a £10M company and a £-100k company sit on one scale and
a 10% gain counts about the same at any size. The training target is a tempered softmax of those values
(``tau`` in asinh units). A state whose options are all within ``min_spread`` of each other carries no signal
and is dropped.

Splits are by map: arena-dev-* maps are the held-out ``test`` split; ``VAL_MAPS`` (three train maps) are the
``val`` split used for model selection and temperature fitting; the remaining train maps are ``train``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

VAL_MAPS = frozenset({"arena-train-0227", "arena-train-0228", "arena-train-0229"})
HOLD = "hold"
INSTRUCTIONS = ("Which air route should the company build now, to maximise its net worth (cash minus loan) "
                "two years from now? Building nothing is an option.")


@dataclass(frozen=True)
class DatasetConfig:
    scale: float = 100_000.0
    tau: float = 0.05
    min_spread: float = 1e-3
    features: bool = True  # describe affordability and the money left after the build in each option
    drop_candidate_table: bool = True  # the options already describe the candidates
    ranks: bool = False  # add each option's rank among the options (length, smaller town, cost)


def map_split(task_id: str) -> str:
    if "-dev-" in task_id:
        return "test"
    if "-ood-" in task_id:
        return "ood"
    return "val" if task_id in VAL_MAPS else "train"


RUNOFF_DAYS = 365


def value_chunk(rec: dict) -> int:
    """Index of the rollout chunk at which the option is valued: the end of the rollout, or the end of the
    game (horizon plus run-off year) when that comes first, because the arena scores net worth there."""
    chunk_days = int(rec.get("config", {}).get("chunk_days", 91))
    n = len(next(iter(rec["outcomes"].values()))["rollouts"][0]["nw"])
    st = rec["state"]
    left = int(st["end_date"]) + RUNOFF_DAYS - int(st["date"])
    return max(0, min(n, math.ceil(left / chunk_days)) - 1)


def option_values(rec: dict, cfg: DatasetConfig) -> tuple[dict[str, float], dict[str, float]]:
    """Mean and standard error over seeds of asinh(net worth / scale) at ``value_chunk``, per option."""
    q, se = {}, {}
    at = value_chunk(rec)
    for key in rec["options"]:
        vals = [math.asinh(r["nw"][at] / cfg.scale) for r in rec["outcomes"][key]["rollouts"]]
        m = sum(vals) / len(vals)
        q[key] = m
        se[key] = (math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
                   if len(vals) > 1 else 0.0)
    return q, se


def soft_target(q: dict[str, float], tau: float) -> dict[str, float]:
    top = max(q.values())
    w = {k: math.exp((v - top) / tau) for k, v in q.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def _k(money) -> int:
    return round(int(money or 0) / 1000)


def describe(c: dict, money0: int, features: bool) -> str:
    a, b = c["a"], c["b"]
    text = (f"air route {a.get('name')} ({a.get('population')} people) to {b.get('name')} "
            f"({b.get('population')} people), {c['length_tiles']} tiles, estimated cost £{_k(c['estimated_cost'])}k")
    if features:
        # The estimate is two large airports and two fast planes; a poorer company builds it smaller (the build
        # ladder), so this is a scale hint, not a yes/no on whether it can be built.
        left = int(money0) - int(c["estimated_cost"])
        text += (f"; £{_k(left)}k cash would remain after that cost" if left >= 0
                 else f"; that cost exceeds the cash by £{_k(-left)}k")
    return text


_ORD = {1: "1st", 2: "2nd", 3: "3rd"}


def _ordinal(n: int) -> str:
    return _ORD.get(n, f"{n}th")


def rank_text(candidates: list[dict], i: int) -> str:
    """Where candidate ``i`` stands among the offered routes: longest, largest smaller town, cheapest."""
    def rank(key, reverse: bool) -> int:
        vals = sorted((key(c) for c in candidates), reverse=reverse)
        return vals.index(key(candidates[i])) + 1

    def pop(c):
        return min(int(c["a"].get("population") or 0), int(c["b"].get("population") or 0))

    n = len(candidates)
    return (f"{_ordinal(rank(lambda c: c['length_tiles'], True))} longest, "
            f"{_ordinal(rank(pop, True))} largest towns, "
            f"{_ordinal(rank(lambda c: c['estimated_cost'], False))} cheapest of {n} routes")


def observation(rendered_state: dict, candidates: list[dict], money0: int, cfg: DatasetConfig) -> dict:
    """What a policy sees at a wake-up: the state text, the question and the options, plus the scripted
    baselines' features. The training examples and the online player both build it here, so the model reads
    the same text offline and in play."""
    options = [(c["candidate_id"], describe(c, money0, cfg.features)) for c in candidates]
    if cfg.ranks and candidates:
        options = [(k, f"{text}; {rank_text(candidates, i)}") for i, (k, text) in enumerate(options)]
    options.append((HOLD, "build nothing now and keep the cash"))
    state = dict(rendered_state)
    if cfg.drop_candidate_table:
        state.pop("candidates", None)
    return {
        "state_json": json.dumps(state, separators=(",", ":")), "instructions": INSTRUCTIONS,
        "options": options, "money0": money0,
        "features": {c["candidate_id"]: {"length": c["length_tiles"], "cost": c["estimated_cost"],
                                         "pop_min": min(int(c["a"].get("population") or 0),
                                                        int(c["b"].get("population") or 0))}
                     for c in candidates},
    }


def build_example(rec: dict, cfg: DatasetConfig) -> dict | None:
    q, se = option_values(rec, cfg)
    if max(q.values()) - min(q.values()) < cfg.min_spread:
        return None
    by_id = {c["candidate_id"]: c for c in rec["candidates"]}
    cands = [by_id[k] for k in rec["options"] if k != HOLD]
    meta = rec["state"]
    return {
        "id": meta["state_id"], "task_id": meta["task_id"], "split": map_split(meta["task_id"]),
        "source": meta["source"], **observation(rec["rendered_state"], cands, rec["money0"], cfg),
        "q": q, "se": se, "target": soft_target(q, cfg.tau), "best": max(q, key=q.get),
    }


def load_examples(labels_dir: Path, cfg: DatasetConfig = DatasetConfig()) -> list[dict]:
    out = []
    for p in sorted(Path(labels_dir).glob("*.json")):
        if p.name.endswith(".error.json") or p.name.startswith("_"):
            continue
        ex = build_example(json.loads(p.read_text()), cfg)
        if ex is not None:
            out.append(ex)
    return out
