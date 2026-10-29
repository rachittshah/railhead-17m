import json

from rh17m.bench import compare_label_dirs


def _rec(nw, build_s=10.0):
    return {"state": {"id": "s"}, "options": ["a", "hold"], "builds": {"a": {"ok": True, "cost": 5}},
            "outcomes": {"a": {"rollouts": [{"nw": nw, "money": [1]}]}, "hold": {"rollouts": [{"nw": [1], "money": [1]}]}},
            "timing": {"build_s": build_s, "total_s": build_s + 1, "bridge_calls": 1269}}


def _write(d, name, rec):
    d.mkdir(exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(rec))


def test_records_equal_when_only_timing_differs(tmp_path):
    _write(tmp_path / "a", "s1", _rec([3.0], build_s=60))
    _write(tmp_path / "b", "s1", _rec([3.0], build_s=20))
    r = compare_label_dirs(tmp_path / "a", tmp_path / "b")
    assert r == {"common": 1, "identical": 1, "differing": [], "only_a": [], "only_b": []}


def test_a_changed_outcome_is_reported_with_its_path(tmp_path):
    _write(tmp_path / "a", "s1", _rec([3.0]))
    _write(tmp_path / "b", "s1", _rec([3.5]))
    _write(tmp_path / "a", "s2", _rec([1.0]))
    r = compare_label_dirs(tmp_path / "a", tmp_path / "b")
    assert r["identical"] == 0 and r["only_a"] == ["s2"]
    assert r["differing"] == [{"state": "s1", "first_diff": "outcomes.a.rollouts[0].nw[0]"}]
