import math

import pytest

from rh17m.dataset import DatasetConfig, build_example, map_split, option_values, soft_target, value_chunk


def _record(values: dict[str, list[int]], money0: int = 500_000, costs: dict[str, int] | None = None) -> dict:
    cands = [k for k in values if k != "hold"]
    return {
        "state": {"state_id": "arena-train-0201.x.w001", "task_id": "arena-train-0201", "split": "train",
                  "source": "teacher:m", "ymd": "2001-01-01", "date": 0, "end_date": 3650},
        "nw0": 100_000, "money0": money0,
        "rendered_state": {"date": "2001-01-01", "finance": {"money": money0}, "candidates": {"cols": "x", "rows": []}},
        "candidates": [{"candidate_id": k, "mode": "air", "a": {"name": f"A{k}", "population": 900},
                        "b": {"name": f"B{k}", "population": 700}, "length_tiles": 300 + i,
                        "cargo": "PASS", "estimated_cost": (costs or {}).get(k, 300_000)} for i, k in enumerate(cands)],
        "options": cands + ["hold"],
        "outcomes": {k: {"ok": True, "cost": 1, "rollouts": [{"nw": [0, v]} for v in vs]} for k, vs in values.items()},
    }


def test_option_values_use_final_net_worth_under_asinh():
    rec = _record({"a": [1_000_000, 1_000_000], "hold": [100_000, 100_000]})
    q, se = option_values(rec, DatasetConfig(scale=100_000))
    assert q["a"] == pytest.approx(math.asinh(10.0))
    assert q["hold"] == pytest.approx(math.asinh(1.0))
    assert se["a"] == pytest.approx(0.0)


def test_soft_target_is_a_tempered_softmax_peaked_on_the_best_option():
    t = soft_target({"a": 1.0, "b": 0.9, "hold": 0.0}, tau=0.1)
    assert sum(t.values()) == pytest.approx(1.0)
    assert t["a"] > t["b"] > t["hold"]
    assert t["a"] / t["b"] == pytest.approx(math.e)


def test_uninformative_state_is_dropped():
    rec = _record({"a": [5, 5], "b": [5, 5], "hold": [5, 5]})
    assert build_example(rec, DatasetConfig()) is None


def test_example_options_describe_affordability_and_hold():
    rec = _record({"a": [2_000_000] * 2, "b": [150_000] * 2, "hold": [100_000] * 2}, money0=200_000,
                  costs={"a": 150_000, "b": 900_000})
    ex = build_example(rec, DatasetConfig(features=True))
    keys = [k for k, _ in ex["options"]]
    assert keys == ["a", "b", "hold"]
    text = dict(ex["options"])
    assert "cash would remain" in text["a"] and "exceeds the cash" in text["b"]
    assert "build nothing" in text["hold"]
    assert ex["best"] == "a"
    assert ex["split"] == "train"


def test_map_split_holds_out_validation_maps():
    assert map_split("arena-dev-0212") == "test"
    assert map_split("arena-train-0228") == "val"
    assert map_split("arena-train-0201") == "train"


def test_late_states_are_valued_at_the_end_of_the_game():
    rec = _record({"a": [1, 2], "hold": [3, 4]})
    for o in rec["outcomes"].values():
        for r in o["rollouts"]:
            r["nw"] = [10, 20, 30, 40, 50, 60, 70, 80]
    rec["config"] = {"chunk_days": 91}
    rec["state"].update(date=1000, end_date=1000)  # only the run-off year is left: 365 days = 5 chunks
    assert value_chunk(rec) == 4
    rec["state"].update(date=0, end_date=5000)
    assert value_chunk(rec) == 7


def test_rank_features_name_the_longest_route():
    rec = _record({"a": [2_000_000] * 2, "b": [150_000] * 2, "hold": [100_000] * 2})
    ex = build_example(rec, DatasetConfig(ranks=True))
    text = dict(ex["options"])
    assert "2nd longest" in text["a"] and "1st longest" in text["b"] and "of 2 routes" in text["b"]
