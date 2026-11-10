"""The closed-loop oracle: seeds that cannot peek, the labeller's machinery, the player's walk, its records."""

import json
import math
from types import SimpleNamespace

import pytest

from rh17m import oracle, play
from rh17m.labels import LabelConfig, rollout_chunks


# ---------------------------------------------------------------------- no peeking
def test_rollout_seeds_never_equal_the_live_games_seed():
    for game_seed in (1, 2):
        seeds = [oracle.rollout_seed("tycoon-dev-0022", game_seed, w) for w in range(50)]
        assert all(s != game_seed & 0xFFFFFFFF for s in seeds)
        assert len(set(seeds)) == 50  # a fresh future at every wake-up
    assert oracle.rollout_seed("t", 1, 3) == oracle.rollout_seed("t", 1, 3)  # deterministic: resumable
    assert oracle.rollout_seed("t", 1, 3) != oracle.rollout_seed("t", 2, 3)
    assert oracle.rollout_seed("t", 1, 3) != oracle.rollout_seed("u", 1, 3)


def test_the_labeller_keeps_its_365_day_runoff_and_the_oracle_uses_the_tasks():
    cfg = LabelConfig()
    assert rollout_chunks(cfg, end_date=365, date=360) == 5  # labeller: horizon + 365 days
    assert rollout_chunks(cfg, end_date=365, date=360, runoff_days=730) == 8  # tycoon game: 728 days fit
    assert rollout_chunks(cfg, end_date=1825, date=1800, runoff_days=365) == 5  # arena: same as the labeller


def test_the_oracle_refuses_an_engine_variant(monkeypatch):
    monkeypatch.setenv("RAILHEAD_OPENTTD", "/tmp/some-variant/openttd")
    with pytest.raises(RuntimeError, match="stock engine"):
        oracle.OraclePolicy()


# ---------------------------------------------------------------------- a whole game on fakes
class FakeWorld:
    def __init__(self, log):
        self.date, self.log, self.money = 0, log, 100_000

    @property
    def admin(self):
        return SimpleNamespace(rcon=lambda cmd: self.log.append(("rcon", cmd)) or ["Reseeded"])

    def state(self):
        return {"date": self.date, "ymd": f"d{self.date}"}

    def step(self, days):
        self.log.append(("step", self.date, days))
        self.date += days

    def call(self, method, **_):
        assert method == "company"
        return {"money": self.money, "loan": 0}

    def close(self):
        self.log.append(("close",))


VALUES = {"r1": 5.0, "r2": 3.0, "expand-a": 4.0, "hold": 4.0}  # r1 best; expand-a ties hold


@pytest.fixture
def fake_game(monkeypatch, tmp_path):
    log: list = []
    world = FakeWorld(log)
    task = SimpleNamespace(horizon_days=365, runoff_days=730, objective={"w_idle": 0.0, "w_base": 1.0})
    cands = [SimpleNamespace(candidate_id="r1"), SimpleNamespace(candidate_id="r2")]
    exps = [SimpleNamespace(key="expand-a")]

    def wakeup_observation(w, t, end_date, max_air, dcfg):
        keys = [c.candidate_id for c in cands] + [e.key for e in exps] + ["hold"]
        return ({"options": [(k, k) for k in keys], "features": {c.candidate_id: {} for c in cands}}, cands, exps)

    def build_options(parent, tmp, cs, es):
        assert parent is world  # the live world is the parent: forks only read it
        log.append(("eval", world.date, [c.candidate_id for c in cs] + [e.key for e in es]))
        keys = [c.candidate_id for c in cs] + [e.key for e in es]
        return ({k: tmp / k for k in keys + ["hold"]}, {k: {"ok": True, "cost": 10} for k in keys}, set())

    def roll_options(tmp, saves, builds, aliases, seeds, cfg, chunks):
        log.append(("roll", world.date, list(seeds), chunks))
        return {k: {"rollouts": [{"nw": [0] * (chunks - 1) + [round(math.sinh(VALUES[k]) * 1e5)]}]} for k in saves}

    def apply_option(w, key, cs, es):
        log.append(("apply", w.date, key))
        if key == "r1":  # the best route fails without spending: the walk moves on
            return {"ok": False, "cost": 0, "failed_step": "site_a:ERR_NO_SITE", "kind": "route", "steps": False}
        return {"ok": True, "cost": 10, "failed_step": None, "kind": "route", "steps": True}

    monkeypatch.setattr(play, "find_task", lambda t: tmp_path)
    monkeypatch.setattr(play.TaskSpec, "load", staticmethod(lambda d: task))
    monkeypatch.setattr(play.World, "load", staticmethod(lambda wd, sav: world))
    monkeypatch.setattr(play, "wakeup_observation", wakeup_observation)
    monkeypatch.setattr(play, "apply_option", apply_option)
    monkeypatch.setattr(play, "tycoon_log", lambda nw, a, b: 0.5)
    monkeypatch.setattr(oracle, "build_options", build_options)
    monkeypatch.setattr(oracle, "roll_options", roll_options)
    monkeypatch.delenv("RAILHEAD_OPENTTD", raising=False)
    return log


def test_the_oracle_plays_descending_value_through_the_players_walk(fake_game):
    log = fake_game
    res = play.play("tycoon-dev-0022", oracle.OraclePolicy(), seed=2, wake_days=180)
    d0, d1 = res["decisions"][:2]
    # wake 0: r1 (best) fails without spending, so the walk goes on; hold ties expand-a and wins the tie
    assert [t["key"] for t in d0["tries"]] == ["r1"] and d0["played"] == "hold"
    assert d0["values"] == pytest.approx(VALUES, abs=1e-5) and d0["best"] == "r1"
    # wake 1: r1 is blocked for the rest of the game, so it is neither offered nor valued; r2 (3.0) < hold
    evals = [e for e in log if e[0] == "eval"]
    assert evals[0][2] == ["r1", "r2", "expand-a"] and evals[1][2] == ["r2", "expand-a"]
    assert d1["tries"] == [] and d1["best"] in ("hold",)


def test_the_oracle_never_moves_the_live_game_and_seeds_each_wake_apart(fake_game):
    log = fake_game
    res = play.play("tycoon-dev-0022", oracle.OraclePolicy(), seed=2, wake_days=180)
    assert ("rcon", "reseed 2") in log
    steps = [e for e in log if e[0] == "step"]
    assert steps == [("step", 0, 180), ("step", 180, 180), ("step", 360, 5), ("step", 365, 730)]  # play's own
    for wake, d in enumerate(res["decisions"]):
        assert d["rollout_seed"] == oracle.rollout_seed("tycoon-dev-0022", 2, wake) != 2
    rolls = [e for e in log if e[0] == "roll"]
    assert [r[2] for r in rolls] == [[d["rollout_seed"]] for d in res["decisions"]]  # one seed, all options
    assert all(r[3] == 8 for r in rolls)  # 728 days: the tycoon game (730-day run-off) outlasts them
    # every valuation happens at the wake-up's date, before the live build of that wake-up
    for e in log:
        if e[0] == "apply":
            assert any(x[0] == "eval" and x[1] == e[1] for x in log[:log.index(e)])


def test_the_oracles_game_record_carries_what_its_result_depends_on(fake_game):
    res = play.play("tycoon-dev-0022", oracle.OraclePolicy(), seed=2, wake_days=180)
    for k in ("wake_days", "player", "label_version", "oracle", "pipelined", "pool", "engine", "engine_bin", "rhs1"):
        assert k in res
    assert res["wake_days"] == 180 and res["label_version"] == LabelConfig().version
    assert res["player"] == play.PLAYER_V and res["engine"] == "stock" and "partial" not in res


def test_max_wakes_makes_a_partial_game_without_runoff(fake_game):
    log = fake_game
    res = play.play("tycoon-dev-0022", oracle.OraclePolicy(), seed=1, wake_days=180, max_wakes=2)
    assert res["partial"] is True and res["score"] is None and len(res["decisions"]) == 2
    assert ("step", 360, 730) not in log and ("step", 365, 730) not in log


def test_nothing_buildable_is_a_hold(fake_game, monkeypatch):
    def nothing(parent, tmp, cs, es):
        keys = [c.candidate_id for c in cs] + [e.key for e in es]
        return {"hold": tmp / "hold"}, {k: {"ok": False, "cost": 0} for k in keys}, set(keys)

    monkeypatch.setattr(oracle, "build_options", nothing)
    res = play.play("tycoon-dev-0022", oracle.OraclePolicy(), seed=1, wake_days=180, max_wakes=1)
    assert res["decisions"][0]["uninformative"] is True and res["decisions"][0]["tries"] == []


# ---------------------------------------------------------------------- cache key
def test_only_the_oracles_cache_key_carries_its_settings(tmp_path):
    assert play.game_path(tmp_path, "random", "t", 1, 180).name == "random.t.s1.json"
    a = play.game_path(tmp_path, "oracle", "t", 1, 180).name
    b = play.game_path(tmp_path, "oracle", "t", 1, 90).name
    assert a.startswith("oracle.t.s1.w180.") and a != b.replace("w90", "w180")
    assert ".m2." in play.game_path(tmp_path, "oracle", "t", 1, 180, max_wakes=2).name


# ---------------------------------------------------------------------- the comparison
def _game(policy, task, seed, nw, **extra):
    return {"policy": policy, "task": task, "seed": seed, "player": "play-v3", "final_nw": nw, "score": 0.5,
            "decisions": [{"date": "1990-02-01", "tries": []}, {"date": "1990-07-31", "tries": []}], **extra}


def _write(d, rec, name=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / (name or f"{rec['policy']}.{rec['task']}.s{rec['seed']}.json")).write_text(json.dumps(rec))


@pytest.fixture
def suite(tmp_path):
    base, model, orc = tmp_path / "baselines", tmp_path / "1023-1627-base-s1", tmp_path / "oracle"
    tasks = ["m0", "m1", "m2", "m3"]
    for i, t in enumerate(tasks):
        for s in (1, 2):
            _write(base, _game("random", t, s, 100_000 * (i + 1)))
            _write(base, _game("longest", t, s, 80_000 * (i + 1)))
            _write(model, _game("model", t, s, 110_000 * (i + 1)))
            dec = [{"date": "1990-02-01", "values": {"r1": 2.0, "expand-a": 1.0, "hold": 0.0}, "ok": True,
                    "best": "r1", "played": "r1", "tries": [{"key": "r1", "ok": True, "cost": 5}],
                    "fork_builds": {"r1": {"ok": True, "cost": 5}}},
                   {"date": "1990-07-31", "values": {"r1": 1.0, "hold": 1.0}, "best": "hold", "played": "hold",
                    "tries": []},
                   {"date": "1991-01-27", "values": {"r1": 0.0, "hold": 0.0}, "uninformative": True,
                    "tries": []}]
            _write(orc, {**_game("oracle", t, s, 300_000 * (i + 1), wake_days=180), "decisions": dec},
                   name=f"oracle.{t}.s{s}.w180.abc.json")
    _write(base, _game("random", "only-random", 1, 5))  # unpaired: dropped
    return [base, model, orc]


def test_compare_pairs_on_task_and_seed_and_bootstraps_over_maps(suite):
    out = play.compare(suite, refs=["random", "longest", "model:1023-1627-base-s1"])
    assert out["maps"] == 4 and out["games_per_label"] == 8 and out["seeds"] == [1, 2]
    pol = out["policies"]
    assert set(pol) == {"random", "longest", "model:1023-1627-base-s1", "oracle"}
    exp = sum(math.asinh(3 * (i + 1)) - math.asinh(i + 1) for i in range(4)) / 4
    vs = pol["oracle"]["vs"]["random"]
    assert vs["diff"] == pytest.approx(exp, abs=1e-4) and vs["ci95"][0] <= vs["diff"] <= vs["ci95"][1]
    assert vs["ci95"][0] > 0 and set(pol["oracle"]["vs"]) == {"random", "longest", "model:1023-1627-base-s1"}
    assert pol["random"]["wake_days"] == [180] and out["warnings"] == []  # read off the decision dates
    dec = pol["oracle"]["decisions"]
    assert dec["decisions"] == 16 and dec["best_is"] == {"hold": 0.5, "expand": 0.0, "route": 0.5}
    assert dec["best_minus_mean"]["mean"] == pytest.approx(0.5) and dec["played_built"] == 0.5
    assert dec["fork_vs_live"] == {"tries_checked": 8, "mismatched": 0}
    assert dec["uninformative"] == 8 and dec["played_best"] == 1.0 and dec["second_tries"] == 0
    assert out["unpaired"] == {"random": ["only-random.s1"]}
    assert "oracle" in play.format_compare(out)


def test_compare_refuses_partial_and_duplicate_games(suite, tmp_path):
    _write(suite[2], {**_game("oracle", "m0", 1, 1, partial=True)}, name="oracle.m0.s1.w180.m2.x.json")
    with pytest.raises(ValueError, match="partial"):
        play.compare(suite)
    (suite[2] / "oracle.m0.s1.w180.m2.x.json").unlink()
    _write(suite[2], _game("oracle", "m0", 1, 1), name="oracle.m0.s1.w90.other.json")
    with pytest.raises(ValueError, match="twice"):
        play.compare(suite)


def test_compare_warns_on_mixed_wake_intervals(suite):
    _write(suite[0], {**_game("random", "m0", 1, 100_000), "wake_days": 90}, name="random.m0.s1.json")
    out = play.compare(suite)
    assert any("wake" in w for w in out["warnings"])
