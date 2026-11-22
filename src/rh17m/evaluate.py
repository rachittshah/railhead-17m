"""Offline decision metrics against simulator values, baselines and paired bootstrap comparisons.

Per decision, with ``q`` the option values (asinh net worth, see ``dataset``) and ``p`` a policy's
probabilities:

- ``regret``: q[best] - q[argmax p], in asinh units (0.1 is roughly 10% of net worth at the horizon).
- ``exp_regret``: sum_k p_k (q[best] - q_k), the regret of sampling from p.
- ``capture``: (q[pick] - mean q) / (q[best] - mean q), the share of the best option's edge over a random
  pick that the policy's pick keeps (1 = oracle, 0 = random on average).
- ``hit``: the pick is within ``2 * se`` of the best option (seed noise cannot separate them).
- ``nll``: cross-entropy of the soft target under p. ``ece``: 10-bin calibration error of max p vs hit.
- ``play_regret``: regret of the closed-loop player's outcome (``play_value``): it walks the options in the
  policy's order and skips builds that failed without spending, as play-v2 does.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

Policy = Callable[[dict], dict[str, float]]


def _pick(p: dict[str, float]) -> str:
    return max(p, key=lambda k: (p[k], k == "hold"))


def play_value(ex: dict, p: dict[str, float], max_tries: int = 3) -> float:
    """The value the closed-loop player (play-v2) would get: options in the policy's order, a build that
    failed without spending moves on to the next (up to ``max_tries``), ``hold`` stops the walk."""
    built = ex.get("built") or {}
    for i, k in enumerate(sorted(p, key=lambda k: (-p[k], k != "hold"))):
        if k == "hold" or i >= max_tries:
            return ex["q"]["hold"]
        if built.get(k, True):
            return ex["q"][k]
    return ex["q"]["hold"]


def decision_metrics(ex: dict, p: dict[str, float]) -> dict[str, float]:
    q, se = ex["q"], ex["se"]
    best = max(q.values())
    mean = sum(q.values()) / len(q)
    pick = _pick(p)
    tol = max(1e-3, 2 * math.hypot(se[ex["best"]], se[pick]))
    z = sum(p.values()) or 1.0
    return {
        "regret": best - q[pick],
        "exp_regret": sum((p.get(k, 0.0) / z) * (best - v) for k, v in q.items()),
        "capture": (q[pick] - mean) / (best - mean) if best > mean else 1.0,
        "hit": float(q[pick] >= best - tol),
        "nll": -sum(t * math.log(max(p.get(k, 0.0) / z, 1e-12)) for k, t in ex["target"].items() if t > 0),
        "conf": max(p.values()) / z,
        "pick_hold": float(pick == "hold"),
        "play_regret": best - play_value(ex, p),
    }


def ece(conf: list[float], hit: list[float], bins: int = 10) -> float:
    total, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            total += len(idx) / n * abs(sum(conf[i] for i in idx) / len(idx) - sum(hit[i] for i in idx) / len(idx))
    return total


def metrics(examples: list[dict], probs: list[dict[str, float]]) -> dict[str, float]:
    rows = [decision_metrics(e, p) for e, p in zip(examples, probs, strict=True)]
    n = len(rows)
    out = {k: sum(r[k] for r in rows) / n for k in rows[0]} if rows else {}
    if rows:
        out["ece"] = ece([r["conf"] for r in rows], [r["hit"] for r in rows])
        out["n"] = n
    return out


def per_example(examples: list[dict], probs: list[dict[str, float]], key: str = "regret") -> list[float]:
    return [decision_metrics(e, p)[key] for e, p in zip(examples, probs, strict=True)]


def bootstrap_diff(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over paired examples and its 95% percentile bootstrap interval."""
    d = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def cluster_bootstrap_diff(a: list[float], b: list[float], clusters: list[str], n: int = 2000,
                           seed: int = 0) -> tuple[float, float, float]:
    """Mean of a - b over paired examples with a 95% percentile bootstrap that resamples whole clusters
    (maps): decisions from one map are not independent, so resampling single decisions is too narrow."""
    groups: dict[str, list[float]] = {}
    for x, y, c in zip(a, b, clusters, strict=True):
        groups.setdefault(c, []).append(x - y)
    sums = [(sum(d), len(d)) for d in groups.values()]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        draw = [rng.choice(sums) for _ in sums]
        means.append(sum(s for s, _ in draw) / sum(k for _, k in draw))
    means.sort()
    return sum(s for s, _ in sums) / sum(k for _, k in sums), means[int(0.025 * n)], means[int(0.975 * n) - 1]


# ---------------------------------------------------------------------- baselines
def _onehot(ex: dict, key: str) -> dict[str, float]:
    return {k: float(k == key) for k, _ in ex["options"]}


def _affordable(ex: dict) -> dict[str, dict]:
    return {k: f for k, f in ex["features"].items() if f["cost"] <= ex["money0"]}


def _longest_affordable(ex: dict) -> dict[str, float]:
    aff = _affordable(ex)
    return _onehot(ex, max(aff, key=lambda k: aff[k]["length"]) if aff else "hold")


def _biggest_affordable(ex: dict) -> dict[str, float]:
    aff = _affordable(ex)
    return _onehot(ex, max(aff, key=lambda k: aff[k]["pop_min"]) if aff else "hold")


def _graded(ex: dict, key, top: str | None = None) -> dict[str, float]:
    """Probabilities that rank routes by ``key`` (largest first; ``top`` first if given) and hold last: the
    argmax is the scripted pick, and the order is its fallback when a build fails."""
    keys = sorted(ex["features"], key=lambda k: (k != top, -key(ex["features"][k])))
    n = len(keys)
    raw = {k: float(n - i) for i, k in enumerate(keys)}
    raw["hold"] = 0.0 if n else 1.0
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def _longest(ex: dict) -> dict[str, float]:
    return _graded(ex, lambda f: f["length"]) if ex["features"] else _onehot(ex, "hold")


BASELINES: dict[str, Policy] = {
    "uniform": lambda ex: {k: 1.0 / len(ex["options"]) for k, _ in ex["options"]},
    "hold": lambda ex: _onehot(ex, "hold"),
    "longest": _longest,
    "longest_affordable": _longest_affordable,
    "biggest_affordable": _biggest_affordable,
    "oracle": lambda ex: _onehot(ex, ex["best"]),
}
