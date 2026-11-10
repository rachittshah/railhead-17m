"""Search at decision time: the model ranks, the oracle's simulator judges the top k, the player's walk plays."""

import json
import math
from types import SimpleNamespace

import pytest

from rh17m import oracle, play, search
from rh17m.labels import LabelConfig

KEYS = ["r1", "r2", "r3", "expand-a", "hold"]
VALUES = {"r1": 5.0, "r2": 3.0, "r3": 6.0, "expand-a": 4.5, "hold": 4.0}
PRIOR = {"r1": 0.40, "r2": 0.25, "r3": 0.05, "expand-a": 0.20, "hold": 0.10}  # model: r1 > r2 > expand-a > hold > r3


# ---------------------------------------------------------------------- names, seeds, selection
def test_names_parse_and_bad_names_are_refused():
    assert search.parse_name("search:3") == (3, "model")
    assert search.parse_name("search:2:model") == (2, "model")
    assert search.parse_name("search:3:random") == (3, "random")
    for bad in ("search", "search:0", "search:x", "search:3:longest", "search:3:random:1"):
        with pytest.raises(ValueError):
            search.parse_name(bad)


def test_top_k_follows_the_models_probabilities():
    assert search.top_k(PRIOR, KEYS, 1) == ["r1"]
    assert search.top_k(PRIOR, KEYS, 3) == ["r1", "r2", "expand-a"]
    assert search.top_k(PRIOR, KEYS, 10) == ["r1", "r2", "expand-a", "hold", "r3"]
    assert search.top_k({"a": 0.5, "hold": 0.5, "b": 0.0}, ["a", "b", "hold"], 1) == ["hold"]  # ties: hold first


def test_hold_is_always_simulated_and_only_k_or_k_plus_one_are():
    sim = search.simulated_set(search.top_k(PRIOR, KEYS, 3), KEYS)
    assert sim == ["r1", "r2", "expand-a", "hold"]
    assert search.simulated_set(search.top_k(PRIOR, KEYS, 4), KEYS) == ["r1", "r2", "expand-a", "hold"]  # in top 4
    assert search.simulated_set(["hold"], KEYS) == ["hold"]


def test_the_random_control_is_reproducible_and_seeded_apart_from_the_rollouts():
    a = search.random_k(KEYS, 3, search.random_seed("t", 1, 0))
    assert a == search.random_k(KEYS, 3, search.random_seed("t", 1, 0)) and len(set(a)) == 3
    draws = {tuple(search.random_k(KEYS, 3, search.random_seed("t", 1, w))) for w in range(30)}
    assert len(draws) > 3  # varies by wake-up
    assert search.random_seed("t", 1, 0) != oracle.rollout_seed("t", 1, 0)
    assert search.random_seed("t", 1, 0) != search.random_seed("t", 2, 0)


def test_scores_put_simulated_options_first_by_value_then_the_rest_in_the_fallback_order():
    s = search.scores({"r1": -2.0, "hold": -1.0, "r2": -1.0}, ["r3", "expand-a"])
    order = sorted(s, key=lambda k: (-s[k], k != "hold"))  # the player's sort, minus its random tie-break
    assert order == ["hold", "r2", "r1", "r3", "expand-a"]
    assert s["hold"] == -1.0 and s["r1"] == -2.0  # simulated scores are the values themselves


# ---------------------------------------------------------------------- a whole game on fakes
class FakeWorld:
    def __init__(self, log):
        self.date, self.log = 0, log

    @property
    def admin(self):
        return SimpleNamespace(rcon=lambda cmd: self.log.append(("rcon", cmd)) or ["Reseeded"])

    def state(self):
        return {"date": self.date, "ymd": f"d{self.date}"}

    def step(self, days):
        self.date += days

    def call(self, method, **_):
        return {"money": 100_000, "loan": 0}

    def close(self):
        pass


@pytest.fixture
def fake_game(monkeypatch, tmp_path):
    log: list = []
    world = FakeWorld(log)
    task = SimpleNamespace(horizon_days=365, runoff_days=730, objective={"w_idle": 0.0, "w_base": 1.0})
    cands = [SimpleNamespace(candidate_id=k) for k in ("r1", "r2", "r3")]
    exps = [SimpleNamespace(key="expand-a")]

    def wakeup_observation(w, t, end_date, max_air, dcfg):
        return ({"options": [(k, k) for k in KEYS], "features": {}}, cands, exps)

    def build_options(parent, tmp, cs, es):
        assert parent is world
        keys = [c.candidate_id for c in cs] + [e.key for e in es]
        log.append(("eval", world.date, keys))
        return ({k: tmp / k for k in keys + ["hold"]}, {k: {"ok": True, "cost": 10} for k in keys}, set())

    def roll_options(tmp, saves, builds, aliases, seeds, cfg, chunks):
        log.append(("roll", world.date, list(seeds), chunks, sorted(saves)))
        return {k: {"rollouts": [{"nw": [round(math.sinh(VALUES[k]) * 1e5)]}]} for k in saves}

    def apply_option(w, key, cs, es):
        log.append(("apply", w.date, key))
        return {"ok": True, "cost": 10, "failed_step": None, "kind": "route", "steps": True}

    monkeypatch.setattr(play, "find_task", lambda t: tmp_path)
    monkeypatch.setattr(play.TaskSpec, "load", staticmethod(lambda d: task))
    monkeypatch.setattr(play.World, "load", staticmethod(lambda wd, sav: world))
    monkeypatch.setattr(play, "wakeup_observation", wakeup_observation)
    monkeypatch.setattr(play, "apply_option", apply_option)
    monkeypatch.setattr(play, "tycoon_log", lambda nw, a, b: 0.5)
    monkeypatch.setattr(oracle, "build_options", build_options)
    monkeypatch.setattr(oracle, "roll_options", roll_options)
    monkeypatch.setattr(oracle, "variant", lambda: {"rhs1": "x", "pipelined": True, "pool": False,
                                                    "engine": "stock", "engine_bin": "b"})
    monkeypatch.delenv("RAILHEAD_OPENTTD", raising=False)
    return log


def prior(obs):
    return {k: PRIOR[k] for k, _ in obs["options"]}


def no_model(obs):
    raise AssertionError("the random control must not ask the model")


def test_search_simulates_only_the_models_top_k_plus_hold_and_plays_the_best(fake_game):
    log = fake_game
    res = play.play("t", search.SearchPolicy(3, "model", prior=prior), seed=2, wake_days=180, max_wakes=1)
    d = res["decisions"][0]
    evals = [e for e in log if e[0] == "eval"]
    assert evals[0][2] == ["r1", "r2", "expand-a"]  # r3 (the true best, 6.0) is the model's last: not simulated
    assert d["simulated"] == ["r1", "r2", "expand-a", "hold"] and d["n_simulated"] == 4
    assert d["values"] == pytest.approx({k: VALUES[k] for k in d["simulated"]}, abs=1e-5)
    assert d["played"] == "r1" and d["best"] == "r1" and d["source"] == "model"
    assert [k for k, _ in d["model_rank"]] == ["r1", "r2", "expand-a", "hold", "r3"]
    assert d["fallback"] == ["r3"] and d["search_s"] >= 0


def test_search_uses_the_oracles_seed_rule_and_rollout_length(fake_game):
    log = fake_game
    res = play.play("t", search.SearchPolicy(2, "model", prior=prior), seed=2, wake_days=180)
    rolls = [e for e in log if e[0] == "roll"]
    for wake, (d, r) in enumerate(zip(res["decisions"], rolls, strict=True)):
        assert d["rollout_seed"] == oracle.rollout_seed("t", 2, wake) and r[2] == [d["rollout_seed"]]
        assert r[3] == 8


def test_search_with_k_covering_every_option_is_the_oracle(fake_game):
    log = fake_game
    a = play.play("t", oracle.OraclePolicy(), seed=1, wake_days=180)
    rolls_a = [e for e in log if e[0] == "roll"]
    log.clear()
    b = play.play("t", search.SearchPolicy(len(KEYS), "model", prior=prior), seed=1, wake_days=180)
    rolls_b = [e for e in log if e[0] == "roll"]
    assert [r[2:] for r in rolls_a] == [r[2:] for r in rolls_b]  # seeds, length, options (the fake's date runs on)
    for da, db in zip(a["decisions"], b["decisions"], strict=True):
        assert da["values"] == db["values"] and da["played"] == db["played"]
        assert da["rollout_seed"] == db["rollout_seed"] and da["tries"] == db["tries"]
    assert a["final_nw"] == b["final_nw"]


def test_the_random_control_ignores_the_model_and_repeats_itself(fake_game):
    log = fake_game
    r1 = play.play("t", search.SearchPolicy(2, "random", prior=no_model), seed=1, wake_days=180)
    r2 = play.play("t", search.SearchPolicy(2, "random", prior=no_model), seed=1, wake_days=180)
    assert [d["simulated"] for d in r1["decisions"]] == [d["simulated"] for d in r2["decisions"]]
    for wake, d in enumerate(r1["decisions"]):
        drawn = search.random_k(KEYS, 2, search.random_seed("t", 1, wake))
        assert d["simulated"] == search.simulated_set(drawn, KEYS) and d["model_rank"] is None
        assert d["n_simulated"] in (2, 3) and d["source"] == "random"
    assert r1["search_source"] == "random" and r1["search_model"] is None
    assert len([e for e in log if e[0] == "eval"]) == 2 * len(r1["decisions"])


def test_the_game_record_carries_k_source_model_and_the_oracles_variant(fake_game, tmp_path):
    res = play.play("t", search.SearchPolicy(3, "model", model_dir=str(tmp_path / "m"), prior=prior), seed=1,
                    wake_days=180, max_wakes=1)
    for k in ("player", "oracle", "label_version", "label_config", "wake_days", "runoff", "rhs1", "pipelined",
              "pool", "engine", "engine_bin"):
        assert k in res
    assert res["search"] == search.SEARCH_V and res["search_k"] == 3 and res["search_source"] == "model"
    assert res["search_model"] == str((tmp_path / "m").resolve())


def test_the_default_model_is_the_incumbent():
    assert search.resolve_model(None, "model") == str(search.INCUMBENT.resolve())
    assert search.INCUMBENT.parts[-3:] == ("train", "1023-1627-base-s1", "model")
    assert search.resolve_model("/x/m", "random") is None


# ---------------------------------------------------------------------- cache key
def test_the_cache_key_separates_k_source_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle, "variant", lambda: {"engine": "stock"})
    p = {n: play.game_path(tmp_path, n, "t", 1, 180) for n in ("search:2", "search:3", "search:3:random")}
    assert len({x.name for x in p.values()}) == 3 and all(":" not in x.name for x in p.values())
    assert p["search:3"].name.startswith("search-k3-model.t.s1.w180.")
    other = play.game_path(tmp_path, "search:3", "t", 1, 180, model_dir="/elsewhere/model")
    assert other != p["search:3"]
    assert play.game_path(tmp_path, "search:3", "t", 1, 180, model_dir=str(search.INCUMBENT)) == p["search:3"]
    assert ".m2." in play.game_path(tmp_path, "search:3", "t", 1, 180, max_wakes=2).name
    assert play.game_path(tmp_path, "random", "t", 1, 180).name == "random.t.s1.json"  # others unchanged


# ---------------------------------------------------------------------- the share of the oracle's gain
def test_share_is_a_ratio_of_mean_paired_gains_with_a_map_bootstrap():
    rnd = [1.0, 2.0, 3.0, 4.0]
    orc = [2.0, 3.0, 4.0, 5.0]  # +1 everywhere
    lab = [1.5, 2.5, 3.0, 4.6]  # +0.5, +0.5, 0, +0.6
    s = search.oracle_share(lab, orc, rnd, n=500)
    assert s["share"] == pytest.approx(1.6 / 4) and s["ci95"][0] <= s["share"] <= s["ci95"][1]
    assert s["degenerate"] == 0 and s["ci95"][0] >= 0.0 and s["ci95"][1] <= 0.6 + 1e-9


def test_share_skips_and_reports_resamples_whose_gain_is_near_zero():
    rnd = [0.0, 0.0, 0.0, 0.0]
    orc = [1.0, 0.0, 0.0, 0.0]  # the oracle gains on one map only: many resamples have no gain at all
    lab = [0.5, 0.1, 0.0, 0.0]
    s = search.oracle_share(lab, orc, rnd, n=400)
    assert s["share"] == pytest.approx(0.6 / 1.0)
    assert 0 < s["degenerate"] < 400 and all(math.isfinite(x) for x in s["ci95"])
    flat = search.oracle_share([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], n=50)  # no gain at all: no ratio
    assert flat["share"] is None and flat["ci95"] is None and flat["degenerate"] == 50


def test_decision_cost_reads_simulated_counts_and_seconds():
    srch = {"policy_s": 30.0, "decisions": [{"n_simulated": 4, "search_s": 10.0}, {"n_simulated": 3, "search_s": 20.0}]}
    orc = {"policy_s": 60.0, "decisions": [{"values": {"a": 1, "hold": 0}, "oracle_s": 30.0},
                                          {"values": {"a": 1, "b": 2, "hold": 0}, "oracle_s": 30.0}]}
    mdl = {"policy_s": 1.0, "decisions": [{}, {}, {}, {}]}
    assert search.decision_cost([srch]) == {"options_per_decision": 3.5, "s_per_decision": 15.0}
    assert search.decision_cost([orc]) == {"options_per_decision": 2.5, "s_per_decision": 30.0}
    assert search.decision_cost([mdl]) == {"options_per_decision": None, "s_per_decision": 0.25}


def _game(policy, task, seed, nw, **extra):
    return {"policy": policy, "task": task, "seed": seed, "player": "play-v3", "final_nw": nw, "score": 0.5,
            "policy_s": 2.0, "decisions": [{"date": "1990-02-01", "tries": []}, {"date": "1990-07-31", "tries": []}],
            **extra}


def test_compare_reports_the_share_of_the_oracles_gain_for_every_other_label(tmp_path):
    d = tmp_path / "all"
    d.mkdir()
    for i, t in enumerate(["m0", "m1", "m2"]):
        for policy, nw in (("random", 1e5), ("oracle", 3e5), ("search:3", 2e5), ("longest", 0.9e5)):
            extra = {}
            if policy == "search:3":
                extra["decisions"] = [{"date": "1990-02-01", "tries": [], "n_simulated": 4, "search_s": 1.0},
                                      {"date": "1990-07-31", "tries": [], "n_simulated": 4, "search_s": 3.0}]
            rec = _game(policy, t, 1, nw * (i + 1), **extra)
            (d / f"{policy.replace(':', '-')}.{t}.s1.json").write_text(json.dumps(rec))
    out = play.compare([d])
    pol = out["policies"]
    exp = (sum(math.asinh(2 * (i + 1)) - math.asinh(i + 1) for i in range(3))
           / sum(math.asinh(3 * (i + 1)) - math.asinh(i + 1) for i in range(3)))
    assert pol["search:3"]["oracle_share"]["share"] == pytest.approx(exp, abs=1e-4)
    assert pol["longest"]["oracle_share"]["share"] < 0
    assert "oracle_share" not in pol["oracle"] and "oracle_share" not in pol["random"]
    assert pol["search:3"]["cost"] == {"options_per_decision": 4.0, "s_per_decision": 1.0}  # policy_s 2.0 / 2
    text = play.format_compare(out)
    assert "share of oracle gain" in text and "search:3" in text


def test_compare_without_oracle_and_random_has_no_share(tmp_path):
    d = tmp_path / "x"
    d.mkdir()
    for i, t in enumerate(["m0", "m1"]):
        for policy in ("random", "longest"):
            (d / f"{policy}.{t}.s1.json").write_text(json.dumps(_game(policy, t, 1, 1e5 * (i + 1))))
    out = play.compare([d])
    assert all("oracle_share" not in v for v in out["policies"].values())
    assert "share of oracle gain" not in play.format_compare(out)


def test_label_config_is_the_oracles():
    assert search.SearchPolicy(2, "random").oracle.cfg == LabelConfig()
