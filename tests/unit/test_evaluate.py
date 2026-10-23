import pytest

from rh17m.evaluate import BASELINES, bootstrap_diff, metrics


def _ex(q: dict[str, float], money0: int = 400_000) -> dict:
    feats = {k: {"length": 100 * (i + 1), "cost": 300_000 if k != "c" else 900_000, "pop_min": 500 + i}
             for i, k in enumerate(k for k in q if k != "hold")}
    best = max(q, key=q.get)
    target = {k: (1.0 if k == best else 0.0) for k in q}
    return {"id": "s", "split": "val", "source": "t", "options": [(k, k) for k in q], "q": q,
            "se": {k: 0.0 for k in q}, "target": target, "best": best, "features": feats, "money0": money0}


def test_oracle_has_zero_regret_and_full_capture():
    exs = [_ex({"a": 1.0, "b": 0.5, "hold": 0.0})]
    m = metrics(exs, [BASELINES["oracle"](e) for e in exs])
    assert m["regret"] == pytest.approx(0.0)
    assert m["capture"] == pytest.approx(1.0)
    assert m["hit"] == pytest.approx(1.0)


def test_uniform_expected_regret_is_mean_gap():
    exs = [_ex({"a": 1.0, "b": 0.5, "hold": 0.0})]
    m = metrics(exs, [BASELINES["uniform"](e) for e in exs])
    assert m["exp_regret"] == pytest.approx((0.0 + 0.5 + 1.0) / 3)


def test_longest_affordable_skips_unaffordable_routes():
    e = _ex({"a": 1.0, "b": 0.5, "c": 0.2, "hold": 0.0})
    p = BASELINES["longest_affordable"](e)
    assert max(p, key=p.get) == "b"


def test_bootstrap_diff_detects_a_consistent_gain():
    mean, lo, _ = bootstrap_diff([1.0] * 50, [0.0] * 50)
    assert mean == pytest.approx(1.0) and lo > 0.9


def test_play_regret_skips_options_that_fail_to_build():
    from rh17m.evaluate import decision_metrics

    e = _ex({"a": 1.0, "b": 0.6, "hold": 0.0})
    e["built"] = {"a": False, "b": True, "hold": True}
    m = decision_metrics(e, {"a": 0.7, "b": 0.2, "hold": 0.1})
    assert m["regret"] == pytest.approx(0.0)  # argmax is the best option on paper
    assert m["play_regret"] == pytest.approx(0.4)  # but it cannot be built, so the player gets b
    m = decision_metrics(e, {"a": 0.5, "hold": 0.4, "b": 0.1})
    assert m["play_regret"] == pytest.approx(1.0)  # hold comes before b: the player stops there
