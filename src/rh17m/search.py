"""Search at decision time: the model as a prior, the simulator as the judge of a few options.

At a wake-up ``search:<k>`` asks the model for its probabilities over the offered options (the same
observation text as the ``model`` policy), takes its top k, adds ``hold`` when it is not among them, and has
the closed-loop oracle value only those: the inner ``OraclePolicy`` is called with the observation cut down to
the simulated options, so the fork, the build ladder, the lbl-v4 ``expand`` rollouts, the shared rollout seed
``oracle.rollout_seed(task, game seed, wake)`` and the rollout length are the oracle's own code, unchanged.

``search:<k>:random`` is the control: the k options are drawn uniformly from the offered ones (hold included,
the same pool the model ranks) with a seed from (task, game seed, wake) under its own salt; the model is never
asked. If ``search:k`` does no better than its control, the lookahead does the work and the ranking adds nothing.

The policy returns scores for play's unchanged play-v3 walk: a simulated option scores its value; the others
come after every simulated one, in the model's order (the control: a seeded shuffle). Because hold is always
simulated and the walk stops at hold, those fallback options are never tried: the walk can only play a
simulated option that the simulator values above hold (ties go to hold), or fall back along simulated options
whose live build fails without spending.
"""

from __future__ import annotations

import hashlib
import random
import statistics
import time
from pathlib import Path

from rh17m.labels import HOLD, LabelConfig
from rh17m.oracle import OraclePolicy, key_tag

SEARCH_V = "search-v1"
SALT = "search-random"
SOURCES = ("model", "random")
INCUMBENT = Path(__file__).resolve().parents[2] / "runs" / "train" / "1023-1627-base-s1" / "model"


def parse_name(name: str) -> tuple[int, str]:
    """``search:<k>[:model|:random]`` -> (k, source)."""
    parts = name.split(":")
    if len(parts) not in (2, 3) or parts[0] != "search" or not parts[1].isdigit() or int(parts[1]) < 1:
        raise ValueError(f"not a search policy: {name!r} (search:<k> or search:<k>:random)")
    source = parts[2] if len(parts) == 3 else "model"
    if source not in SOURCES:
        raise ValueError(f"unknown candidate source {source!r} in {name!r}: one of {SOURCES}")
    return int(parts[1]), source


def resolve_model(model_dir: str | None, source: str) -> str | None:
    """The model directory a search game uses (and its cache key records): none for the random control, else
    ``model_dir`` or the incumbent, as an absolute path."""
    if source != "model":
        return None
    return str(Path(model_dir or INCUMBENT).resolve())


def random_seed(task_id: str, game_seed: int, wake: int) -> int:
    """The control's draw at one wake-up: a hash of who/when, apart from the rollout seed and the live game."""
    return int(hashlib.sha256(f"{task_id}:{game_seed}:{wake}:{SALT}".encode()).hexdigest()[:8], 16)


def top_k(probs: dict[str, float], keys: list[str], k: int) -> list[str]:
    """The model's k most probable options (ties: hold first, then the observation's order)."""
    return sorted(keys, key=lambda x: (-probs[x], x != HOLD, keys.index(x)))[:k]


def random_k(keys: list[str], k: int, seed: int) -> list[str]:
    return random.Random(seed).sample(keys, min(k, len(keys)))


def simulated_set(chosen: list[str], keys: list[str]) -> list[str]:
    """The chosen options plus hold (the player must be able to decline, and valuing it is free)."""
    return list(chosen) + ([HOLD] if HOLD in keys and HOLD not in chosen else [])


def scores(values: dict[str, float], fallback: list[str]) -> dict[str, float]:
    """Scores for play's walk: simulated options their value; the others below every simulated one, in order."""
    floor = min(values.values()) - 1.0
    return {**values, **{k: floor - i for i, k in enumerate(fallback)}}


def game_key(wake_days: int, k: int, source: str, model_dir: str | None, cfg: LabelConfig = LabelConfig()) -> dict:
    """Everything a search game's result depends on besides (task, seed): the oracle's key (player, label
    version and config, wake interval, run-off, simulator stack) plus k, the candidate source and the model."""
    from rh17m.oracle import game_key as oracle_key

    return {**oracle_key(wake_days, cfg), "search": SEARCH_V, "search_k": k, "search_source": source,
            "search_model": resolve_model(model_dir, source)}


def game_path(out_dir: Path, name: str, task_id: str, seed: int, wake_days: int, max_wakes: int | None = None,
              model_dir: str | None = None) -> Path:
    k, source = parse_name(name)
    part = f".m{max_wakes}" if max_wakes is not None else ""
    tag = key_tag(game_key(wake_days, k, source, model_dir))
    return Path(out_dir) / f"search-k{k}-{source}.{task_id}.s{seed}.w{wake_days}{part}.{tag}.json"


class SearchPolicy:
    """A ``play`` policy that needs the live world (``bind``), like the oracle it wraps."""

    def __init__(self, k: int, source: str = "model", model_dir: str | None = None, prior=None,
                 cfg: LabelConfig = LabelConfig()) -> None:
        if source not in SOURCES or k < 1:
            raise ValueError(f"bad search settings: k={k}, source={source!r}")
        self.k, self.source, self.cfg = k, source, cfg
        self.model_dir = resolve_model(model_dir, source)
        self.oracle = OraclePolicy(cfg)
        self.prior = prior
        if prior is None and source == "model":
            from rh17m.policy import ModelPolicy

            self.prior = ModelPolicy(Path(self.model_dir))
        self.ctx: dict = {}
        self.last: dict = {}

    def bind(self, **ctx) -> None:
        self.ctx = ctx
        self.oracle.bind(**ctx)

    def game_meta(self, wake_days: int) -> dict:
        return game_key(wake_days, self.k, self.source, self.model_dir, self.cfg)

    def __call__(self, obs: dict) -> dict[str, float]:
        t0 = time.perf_counter()
        keys = [k for k, _ in obs["options"]]
        if self.source == "model":
            probs = self.prior(obs)
            rank = top_k(probs, keys, len(keys))
            chosen, model_rank = rank[:self.k], [[k, round(probs[k], 4)] for k in rank]
        else:
            c = self.ctx
            rng = random.Random(random_seed(c["task_id"], c["seed"], c["wake"]))
            chosen = rng.sample(keys, min(self.k, len(keys)))  # == random_k(keys, k, seed)
            rest = [k for k in keys if k not in chosen]
            rng.shuffle(rest)
            rank, model_rank = chosen + rest, None
        sim = simulated_set(chosen, keys)
        values = self.oracle({**obs, "options": [(k, t) for k, t in obs["options"] if k in sim]})
        fallback = [k for k in rank if k not in sim]
        self.last = {**self.oracle.decision_record(), "source": self.source, "model_rank": model_rank,
                     "simulated": sim, "n_simulated": len(sim), "fallback": fallback,
                     "search_s": round(time.perf_counter() - t0, 2)}
        return scores(values, fallback)

    def decision_record(self) -> dict:
        return self.last


def make_search(name: str, model_dir: str | None = None) -> SearchPolicy:
    k, source = parse_name(name)
    return SearchPolicy(k, source, model_dir)


# ---------------------------------------------------------------------- play-report
def oracle_share(lab: list[float], orc: list[float], rnd: list[float], n: int = 2000, seed: int = 0,
                 eps: float = 1e-9) -> dict:
    """Share of the oracle's gain over random that a label recovers, on paired per-map values:
    mean(lab - rnd) / mean(orc - rnd), with a 95% percentile bootstrap that resamples maps (one set of map
    indices for all three series). A resample whose oracle gain is within ``eps`` of zero has no ratio: it is
    left out and counted in ``degenerate``; ``negative_gain`` counts resamples where the oracle lost to random."""
    m = len(rnd)
    gain = [o - r for o, r in zip(orc, rnd, strict=True)]
    got = [x - r for x, r in zip(lab, rnd, strict=True)]
    den = sum(gain) / m
    share = None if abs(den) <= eps else (sum(got) / m) / den
    rng = random.Random(seed)
    ratios, degenerate, negative = [], 0, 0
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        d = sum(gain[i] for i in idx) / m
        if abs(d) <= eps:
            degenerate += 1
            continue
        negative += d < 0
        ratios.append(sum(got[i] for i in idx) / m / d)
    ratios.sort()
    ci = [ratios[int(0.025 * len(ratios))], ratios[max(0, int(0.975 * len(ratios)) - 1)]] if ratios else None
    return {"share": share, "ci95": ci, "degenerate": degenerate, "negative_gain": negative, "resamples": n}


def decision_cost(records: list[dict]) -> dict:
    """Mean options simulated per decision (search: ``n_simulated``; oracle: the options it valued) and mean
    policy seconds per decision (the game's ``policy_s`` over its decisions: every label has it)."""
    sims = [d["n_simulated"] if "n_simulated" in d else len(d["values"])
            for r in records for d in r["decisions"] if "n_simulated" in d or d.get("values")]
    n_dec = sum(len(r["decisions"]) for r in records)
    secs = sum(r.get("policy_s") or 0.0 for r in records)
    return {"options_per_decision": round(statistics.mean(sims), 3) if sims else None,
            "s_per_decision": round(secs / n_dec, 3) if n_dec and all("policy_s" in r for r in records) else None}
