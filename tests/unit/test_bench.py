import json
import os
import statistics
import time
from typing import ClassVar

import pytest
from railhead import server as srv

from rh17m import bench

# ---------------------------------------------------------------- bench labels

NOW = 1_800_000_000.0


def _record(seeds, build_s, total_s, rollouts, aliases=0, bridge_calls=None):
    """A label record with ``rollouts`` rollouts per simulated option (hold + one built option) and
    ``aliases`` options that reuse hold's rollouts (those do not count)."""
    outcomes = {"hold": {"rollouts": [{}] * rollouts}, "r1": {"rollouts": [{}] * rollouts}}
    for i in range(aliases):
        outcomes[f"a{i}"] = {"alias": "hold", "rollouts": [{}] * rollouts}
    timing = {"build_s": build_s, "total_s": total_s}
    if bridge_calls is not None:
        timing.update(bridge_calls=bridge_calls, bridge_ms=bridge_calls * 1.5)
    return {"seeds": list(range(seeds)), "outcomes": outcomes, "timing": timing}


def _put(root, name, rec, age_min):
    p = root / name
    p.write_text(json.dumps(rec))
    t = NOW - 60 * age_min
    os.utime(p, (t, t))


@pytest.fixture
def labels_dir(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    # 1-seed train jobs: 2 rollouts per option x 2 simulated options = 2 rollouts... per option set below
    _put(d, "a.json", _record(1, 60, 110, 1, bridge_calls=400), age_min=1)     # 2 rollouts, 25 s each
    _put(d, "b.json", _record(1, 50, 90, 1, bridge_calls=200), age_min=10)     # 2 rollouts, 20 s each
    _put(d, "c.json", _record(1, 70, 130, 1, aliases=3), age_min=24)           # 2 rollouts (aliases skipped), 30 s
    # 3-seed val jobs
    _put(d, "v1.json", _record(3, 60, 200, 3, bridge_calls=600), age_min=5)    # 6 rollouts, 23.3 s each
    _put(d, "v2.json", _record(3, 80, 260, 3), age_min=20)                     # 6 rollouts, 30 s each (no bridge)
    # nothing could be built: no outcomes, no rollouts; still a labelled state
    _put(d, "u.json", {"seeds": [0], "uninformative": "no_build", "timing": {"build_s": 30, "total_s": 30}},
         age_min=3)
    # outside the window, errors and the progress log do not count
    _put(d, "old.json", _record(1, 999, 9999, 1), age_min=26)
    _put(d, "bad.error.json", {"error": "boom"}, age_min=2)
    (d / "_progress.log").write_text("{}\n")
    return d


def test_label_window_counts_what_the_window_holds(labels_dir):
    r = bench.label_window(labels_dir, since_min=25, now=NOW)
    assert r["states"] == 6 and r["states_per_h"] == pytest.approx(6 * 60 / 25)
    assert r["rollouts"] == 2 + 2 + 2 + 6 + 6
    assert r["build_s_median"] == statistics.median([60, 50, 70, 60, 80, 30])
    # per state: (total_s - build_s) / rollouts; the uninformative state has no rollouts
    per = [50 / 2, 40 / 2, 60 / 2, 140 / 6, 180 / 6]
    assert r["s_per_rollout_median"] == pytest.approx(statistics.median(per), abs=0.01)
    assert r["bridge_calls_median"] == statistics.median([400, 200, 600])  # records that carry it
    assert r["bridge_ms_median"] == statistics.median([600, 300, 900])


def test_label_window_splits_by_seed_count(labels_dir):
    by = bench.label_window(labels_dir, since_min=25, now=NOW)["by_seeds"]
    assert set(by) == {1, 3}
    assert by[1]["states"] == 4 and by[1]["rollouts"] == 6  # a, b, c + the uninformative state
    assert by[1]["build_s_median"] == statistics.median([60, 50, 70, 30])
    assert by[1]["s_per_rollout_median"] == pytest.approx(statistics.median([25, 20, 30]), abs=0.01)
    assert by[1]["bridge_calls_median"] == 300
    assert by[3]["states"] == 2 and by[3]["rollouts"] == 12
    assert by[3]["s_per_rollout_median"] == pytest.approx(statistics.median([140 / 6, 30]), abs=0.01)
    assert by[3]["bridge_calls_median"] == 600


def test_label_window_shrinks_with_the_window(labels_dir):
    r = bench.label_window(labels_dir, since_min=6, now=NOW)  # a (1 min), u (3), v1 (5)
    assert r["states"] == 3 and r["states_per_h"] == pytest.approx(30)
    assert bench.label_window(labels_dir, since_min=0.5, now=NOW)["states"] == 0


def test_label_window_without_records_or_bridge_numbers(tmp_path):
    r = bench.label_window(tmp_path, since_min=25, now=NOW)
    assert r["states"] == 0 and r["rollouts"] == 0 and r["bridge_calls_median"] is None
    assert "0 states" in bench.window_line(r)
    _put(tmp_path, "x.json", _record(1, 60, 110, 1), age_min=1)
    r = bench.label_window(tmp_path, since_min=25, now=NOW)
    assert r["bridge_calls_median"] is None
    assert "gs calls" not in bench.window_line(r)


def test_window_line_is_one_line_with_the_headline_numbers(labels_dir):
    line = bench.window_line(bench.label_window(labels_dir, since_min=25, now=NOW))
    assert "\n" not in line
    assert "6 states" in line and "14.4/h" in line and "18 rollouts" in line
    assert "gs calls 400" in line


# ---------------------------------------------------------------- bench determinism

def fake_arm(save, binary, days, seed):
    """What the engine would return: a state hash that depends on the save, the seed and the days stepped."""
    h = hash((str(save), seed, days)) & 0xFFFFFFFF
    return {"checksum": {"date": 700000 + days, "h": h, "ticks": 74 * days, "vehicles": 3}, "ticks": 74 * days,
            "s": 0.0}


def test_compare_flags_every_field_that_differs():
    a = {"checksum": {"h": 1, "ticks": 100}, "ticks": 100, "s": 1.0}
    assert bench.mismatches(a, {"checksum": {"h": 1, "ticks": 100}, "ticks": 100, "s": 2.0}) == []  # wall time is not state
    assert bench.mismatches(a, {"checksum": {"h": 2, "ticks": 100}, "ticks": 100, "s": 1.0}) == ["checksum"]
    assert bench.mismatches(a, {"checksum": {"h": 2, "ticks": 101}, "ticks": 101, "s": 1.0}) == ["checksum", "ticks"]


def test_determinism_passes_when_both_arms_agree(tmp_path):
    saves = [tmp_path / f"s{i}.sav" for i in range(5)]
    rows = bench.measure_determinism(saves, tmp_path / "variant", days=728, arm=fake_arm)
    assert len(rows) == 5 and all(r["ok"] and not r["mismatch"] for r in rows)
    assert {r["save"] for r in rows} == {s.name for s in saves}


def test_determinism_reports_a_different_seed_as_a_mismatch(tmp_path):
    saves = [tmp_path / f"s{i}.sav" for i in range(3)]
    rows = bench.measure_determinism(saves, tmp_path / "variant", days=728, variant_seed=2, arm=fake_arm)
    assert [r["ok"] for r in rows] == [False] * 3
    assert all("checksum" in r["mismatch"] for r in rows)


def test_determinism_cli_exit_code_follows_the_comparison(tmp_path, monkeypatch, capsys):
    variant = tmp_path / "openttd"
    variant.write_text("")
    saves = [str(tmp_path / f"s{i}.sav") for i in range(2)]
    for s in saves:
        open(s, "w").close()
    monkeypatch.setattr(bench, "run_arm", fake_arm)
    monkeypatch.setattr(bench, "stock_engine", lambda: tmp_path / "stock")
    argv = ["determinism", "--binary", str(variant), "--days", "30", "--saves", *saves]
    assert bench.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("OK") for line in out) == 2 and "2/2" in out[-1]
    assert bench.main([*argv, "--variant-seed", "9"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("MISMATCH") for line in out) == 2 and "0/2" in out[-1]


def test_pick_saves_spans_vehicle_counts_and_pins_the_named_state(tmp_path):
    counts = {"early": 0, "e2": 0, "m1": 3, "m2": 22, "m3": 31, "m4": 70, "big": 264, "w": 148, "w2": 149}
    for name, n in counts.items():
        (tmp_path / f"{name}.sav").write_text("")
        (tmp_path / f"{name}.json").write_text(json.dumps({"state_id": name, "vehicles": n}))
    (tmp_path / "nometa.sav").write_text("")  # no metadata: never picked
    picked = bench.pick_saves(tmp_path, n=5, pinned="w")
    names = [p.stem for p in picked]
    assert len(names) == len(set(names)) == 5
    assert "w" in names and "big" in names and names[0] in ("early", "e2")
    assert names == [p.stem for p in bench.pick_saves(tmp_path, n=5, pinned="w")]  # deterministic
    assert len(bench.pick_saves(tmp_path, n=50)) == 8  # at most one per distinct vehicle count


# ---------------------------------------------------------------- bench step

class FakeStepWorld:
    seen: ClassVar[list] = []

    @classmethod
    def load(cls, workdir, savegame):
        server = srv.OpenTTDServer(workdir, srv.ServerConfig())
        cls.seen.append((os.environ.get("RAILHEAD_OPENTTD"), server.command()))
        return cls()

    def step(self, days):
        time.sleep(0.001)

    def close(self):
        pass


def test_measure_step_runs_the_requested_binary_and_prefix(tmp_path, monkeypatch):
    from rh17m import sched

    variant = tmp_path / "openttd-fast"
    variant.write_text("#!/bin/sh\n")
    variant.chmod(0o755)
    monkeypatch.setattr(bench, "World", FakeStepWorld)
    FakeStepWorld.seen = []
    command = srv.OpenTTDServer.command
    pre = list(sched.PREFIX) if getattr(command, "_rh17m_app_policy", False) else []  # darwin only
    bench.measure_step(tmp_path / "x.sav", binary=variant, days=1, n=1)
    bench.measure_step(tmp_path / "x.sav", prefix=("taskpolicy", "-a", "-l", "0"), binary=variant, days=1, n=1)
    (env0, cmd0), (_, cmd1) = FakeStepWorld.seen
    assert env0 == str(variant)
    assert cmd0[: len(pre) + 1] == [*pre, str(variant)]  # sched's prefix kept, binary swapped
    assert cmd1[:5] == ["taskpolicy", "-a", "-l", "0", str(variant)]  # an explicit prefix replaces it
    assert os.environ.get("RAILHEAD_OPENTTD") != str(variant) and srv.OpenTTDServer.command is command


def test_step_ratio_alternates_arms_and_reports_the_speedup(tmp_path, monkeypatch):
    calls = []
    ms = {"stock": iter([10.0, 10.0, 10.0]), "variant": iter([8.0, 9.0, 8.5])}

    def fake_step(savegame, prefix=(), binary=None, days=91, n=3):
        arm = "variant" if binary == tmp_path / "v" else "stock"
        calls.append(arm)
        return {"prefix": "", "binary": str(binary), "ms_per_day": next(ms[arm])}

    monkeypatch.setattr(bench, "measure_step", fake_step)
    monkeypatch.setattr(bench, "stock_engine", lambda: tmp_path / "s")
    res = bench.compare_step(tmp_path / "x.sav", variant=tmp_path / "v", repeat=3)
    assert calls == ["stock", "variant"] * 3
    assert res["stock_ms_per_day"] == 10.0 and res["variant_ms_per_day"] == 8.5
    assert res["ratio"] == pytest.approx(0.85) and res["faster_pct"] == pytest.approx(15.0)
