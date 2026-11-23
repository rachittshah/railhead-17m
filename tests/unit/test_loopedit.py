"""Loop edit: the re-gauge is exact, every tie is a real loop, and the closed-form fits do what they claim."""

import mlx.core as mx
import numpy as np
import pytest

from rh17m import loopedit as L
from rh17m.bekko_mlx import EncodedDecision, collate, random_model
from rh17m.evaluate import cluster_bootstrap_diff

TYPES = ("full_attention", "sliding_attention", "sliding_attention", "full_attention",
         "sliding_attention", "sliding_attention", "full_attention")


def _model(seed=0):
    return random_model(seed, hidden=32, heads=2, intermediate=48, layer_types=TYPES)


def _batch(model, seed=0, count=6, vocab=128):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        k = int(rng.integers(2, 6))
        rows.append(EncodedDecision(
            "choice", tuple(f"o{j}" for j in range(k)), (1, *rng.integers(3, vocab, int(rng.integers(4, 30))).tolist(), 2),
            tuple((*rng.integers(3, vocab, int(rng.integers(1, 8))).tolist(), 2) for _ in range(k))))
    return collate(rows, tasks=model.tasks)


def _logits(model, batch):
    return np.array(model(batch))


def test_walk_matches_the_model():
    model = _model()
    batch = _batch(model)
    logits, taps = L.walk(model, batch, {(6, "att"), (3, "out")})
    np.testing.assert_allclose(np.array(logits), _logits(model, batch), atol=1e-5)
    assert taps[6, "att"][0] is None and taps[3, "out"][0] is not None  # the last prefix update is never made


def test_loops_of_reads_the_schedule():
    assert L.loops_of(TYPES, 3) == 2
    with pytest.raises(ValueError):
        L.loops_of(TYPES, 2)


def test_fold_and_regauge_leave_the_function_unchanged():
    model = _model()
    batch = _batch(model)
    w = L.weights(model)
    teacher_w = L.teacher_weights(w, 3, 2, heads=2)
    np.testing.assert_allclose(_logits(L.rebuild(model, teacher_w), batch), _logits(model, batch), atol=2e-4)
    assert all(np.all(teacher_w[L.key(p, n)] == 1.0) for p in range(1, 7) for n in ("attn_norm", "mlp_norm"))
    assert not np.allclose(teacher_w[L.key(4, "mlp.Wi")], w[L.key(4, "mlp.Wi")])  # the gauge did move


def test_regauge_recovers_a_scrambled_copy():
    """Layers 4-6 set to a scrambled gauge of layers 1-3: aligning must return them to layers 1-3."""
    model = _model(1)
    w = L.fold_norms(L.weights(model), range(1, 7))
    rng = np.random.default_rng(0)
    for j in (1, 2, 3):
        wq, wo = w[L.key(j, "attn.Wqkv")].reshape(3, 2, 16, 32), w[L.key(j, "attn.Wo")].reshape(32, 2, 16)
        q, o = np.empty_like(wq), np.empty_like(wo)
        for a, b in ((0, 1), (1, 0)):  # swap the heads, rotate the value basis, turn the q/k phases
            rot = np.linalg.qr(rng.normal(size=(16, 16)))[0]
            q[2, a], o[:, a, :] = rot @ wq[2, b], wo[:, b, :] @ rot.T
            phase = np.exp(1j * rng.uniform(0, 2 * np.pi, 8))[:, None]
            for t in (0, 1):
                z = phase * L._complex(wq[t, b])
                q[t, a] = np.concatenate([z.real, z.imag])
        w[L.key(j + 3, "attn.Wqkv")], w[L.key(j + 3, "attn.Wo")] = q.reshape(96, 32), o.reshape(32, 32)
        order, sign = rng.permutation(48), rng.choice([-1.0, 1.0], 48)
        wi, wm = w[L.key(j, "mlp.Wi")], w[L.key(j, "mlp.Wo")]
        w[L.key(j + 3, "mlp.Wi")] = np.concatenate([wi[:48][order], sign[:, None] * wi[48:][order]])
        w[L.key(j + 3, "mlp.Wo")] = wm[:, order] * sign[None, :]
        L.align_attention(w, j, j + 3, heads=2)
        L.align_mlp(w, j, j + 3)
        for name in L.LINEARS:
            np.testing.assert_allclose(w[L.key(j + 3, name)], w[L.key(j, name)], atol=1e-8)


@pytest.mark.parametrize("how", ["first", "last", "mean"])
def test_ties_are_loops(how):
    model = _model()
    tied = L.tie(L.teacher_weights(L.weights(model), 3, 2, heads=2), 3, 2, how)
    assert L.is_looped(tied, 3, 2) and not L.is_looped(L.weights(model), 3, 2)
    count = L.unique_parameters(tied, 3)
    assert count["unique_encoder_layers"] < count["encoder_layers"] and count["unique"] < count["all"]


def test_unroll_reproduces_the_tied_model_and_other_depths_run():
    model = _model()
    batch = _batch(model)
    tied = L.tie(L.teacher_weights(L.weights(model), 3, 2, heads=2), 3, 2, "first")
    same, types = L.unroll(tied, TYPES, 3, 2)
    assert types == TYPES
    np.testing.assert_allclose(_logits(L.rebuild(model, same, types), batch), _logits(L.rebuild(model, tied), batch),
                               atol=1e-6)
    for loops, layers in ((1, 4), (3, 10)):
        wn, tn = L.unroll(tied, TYPES, 3, loops)
        assert len(tn) == layers and np.isfinite(_logits(L.rebuild(model, wn, tn), batch)[:, :2]).all()


def test_solve_tied_recovers_weights_that_are_already_tied():
    """On a model whose blocks are equal, the least-squares solve returns the weight it was given."""
    model = _model()
    tied_w = L.tie(L.teacher_weights(L.weights(model), 3, 2, heads=2), 3, 2, "first")
    tied = L.rebuild(model, tied_w)
    batches = [_batch(model, seed, count=8) for seed in range(6)]
    items = [((1, 4), linear) for linear in L.LINEARS]
    for (_, linear), got in zip(items, L.solve_tied(tied, tied, batches, items, ridge=1e-9), strict=True):
        want = tied_w[L.key(1, linear)]
        if linear in L.NORM_OF:  # a layer-normed input has zero mean: the weight is free along the ones vector
            centre = np.eye(want.shape[1]) - 1.0 / want.shape[1]
            got, want = got @ centre, want @ centre
        np.testing.assert_allclose(got, want, atol=5e-3)


def test_activation_ties_stay_closer_to_the_original_than_copying():
    model = _model(2)
    w = L.teacher_weights(L.weights(model), 3, 2, heads=2)
    teacher = L.rebuild(model, w)
    batches = [_batch(model, seed, count=8) for seed in range(8)]
    test = _batch(model, 99, count=8)
    want = _logits(model, test)
    mask = want > -1e8

    def error(weights):
        return float(np.mean((_logits(L.rebuild(model, weights), test) - want)[mask] ** 2))

    fitted = L.regmean(teacher, batches, 3, 2, ridge=1e-4)
    assert L.is_looped(fitted, 3, 2)
    student = L.rebuild(model, fitted)
    L.refine(student, teacher, batches, 3, 2, ridge=1e-4)
    refined = L.weights(student)
    assert L.is_looped(refined, 3, 2)
    assert error(refined) < min(error(L.tie(w, 3, 2, "first")), error(L.tie(w, 3, 2, "last")))


def test_fidelity_and_cluster_bootstrap():
    ref = [{"a": 0.7, "b": 0.2, "c": 0.1}, {"a": 0.1, "b": 0.9}]
    same = L.fidelity(ref, ref)
    assert same["kl"] == pytest.approx(0.0) and same["top1"] == 1.0 and same["spearman"] == pytest.approx(1.0)
    other = L.fidelity(ref, [{"a": 0.1, "b": 0.2, "c": 0.7}, {"a": 0.1, "b": 0.9}])
    assert other["kl"] > 0 and other["top1"] == 0.5
    assert L.match_temperature(ref, ref) == pytest.approx(1.0, abs=0.05)
    diff, lo, hi = cluster_bootstrap_diff([1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 0.0, 0.0], ["m1", "m1", "m2", "m2"])
    assert diff == 2.0 and lo == 1.0 and hi == 3.0  # the interval spans the maps, not the decisions


def test_explainer_attention_walk_reproduces_the_model():
    """The explainer's hand-written softmax attention gives the model's logits, and its weights are distributions."""
    from rh17m import loopexplain as X

    model = _model(4)
    for seed in range(3):
        batch = _batch(model, seed, count=1)
        walk = X.attention_walk(model, batch)
        k = int(np.array(batch.option_mask[0]).sum())
        np.testing.assert_allclose(np.array(walk["logits"])[:k], _logits(model, batch)[0, :k], atol=1e-4)
        assert walk["fused_error"] < 1e-4
        assert len(walk["stages"]) == len(TYPES) + 1 and walk["probe"].shape[0] == len(TYPES) + 1
        np.testing.assert_allclose(np.array(walk["probe"])[-1, :k], np.array(walk["logits"])[:k], atol=1e-6)
        for index, layer in enumerate(walk["layers"]):
            np.testing.assert_allclose(np.array(layer["option_weights"]).sum(-1), 1.0, atol=1e-5)
            assert (layer["prefix_weights"] is None) == (index == len(TYPES) - 1)


def test_prefix_sections_split_instruction_and_state():
    from rh17m.loopexplain import prefix_sections

    pieces = ["[CLS]", "Instruction", ":", " build", "?", "\n", "State", ":", " {}", "[SEP]"]
    assert prefix_sections(pieces) == [{"name": "instruction", "start": 0, "end": 5},
                                       {"name": "state", "start": 5, "end": 10}]
    assert prefix_sections(["a", "b"]) == [{"name": "prefix", "start": 0, "end": 2}]


def test_weights_round_trip():
    model = _model()
    w = L.weights(model)
    again = L.weights(L.rebuild(model, w))
    assert set(again) == set(w) and all(np.array_equal(again[k].astype(np.float32), w[k].astype(np.float32)) for k in w)
    assert isinstance(L.rebuild(model, w).encoder.backbone.layers[1].attn.Wqkv.weight, mx.array)


def test_figures_render_from_results(tmp_path):
    """The results charts and the weight-similarity measurement run on a small model and a results dict."""
    viz = pytest.importorskip("rh17m.loopviz")
    model = _model(3)
    raw, after, names, heads_info, mlp_info = viz.weight_similarity(L.weights(model), heads=2)
    assert raw.shape == after.shape == (3, 7) and len(heads_info) == len(mlp_info) == 3
    assert after.mean() > raw.mean()  # alignment can only make the blocks look more alike
    fid = {"kl": 0.2, "top1": 0.6, "spearman": 0.8, "n": 10}
    metrics = {"regret": 0.3, "play_regret": 0.3, "exp_regret": 0.35, "n": 10}
    diff = {"diff": 0.01, "lo": -0.02, "hi": 0.04}
    results = {
        "holdout": 10, "chosen": "lsq2", "temperature": 2.0, "source_temperature": 2.1,
        "variants_holdout": {"copy_first": {**fid, "kl": 0.6, "data_free": True},
                             "regmean": {**fid, "data_free": False}, "lsq1": {**fid, "data_free": False},
                             "lsq2": {**fid, "kl": 0.1, "data_free": False}},
        "loops": {str(n): {"layers": 1 + 3 * n, "holdout": fid, "val": metrics} for n in (1, 2, 3)},
        "parameters": {"all": 16_798_211, "encoder_layers": 3_902_720, "unique_encoder_layers": 2_230_016},
        **{s: {"maps": 3, "original": metrics, "looped_minus_original": {m: diff for m in
                                                                          ("regret", "play_regret", "exp_regret")}}
           for s in ("val", "test")},
    }
    for theme in viz.THEMES:
        viz.style(theme)
        for fig in (viz.fig_variants, viz.fig_sweeps, viz.fig_loops, viz.fig_regret, viz.fig_parameters):
            fig(results, tmp_path, theme)
        viz.fig_weight_similarity(raw, after, names, tmp_path, theme)
        viz.fig_head_match(heads_info, tmp_path, theme)
    assert len(list(tmp_path.glob("*.svg"))) == 14 and len(list(tmp_path.glob("*-dark.svg"))) == 7
