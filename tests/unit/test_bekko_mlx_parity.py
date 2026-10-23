"""Parity of the MLX port with the published PyTorch runtime (CPU, fp32, SDPA) at the pinned revision.

Needs the ``reference`` extra and the Hub snapshot. Tolerances are the acceptance bars, not fitted values;
the measured maxima are printed (run with ``-s``) and recorded in docs/bekko-mlx.md.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

pytestmark = pytest.mark.integration

PROB_TOL = 1e-4


@pytest.fixture(scope="module")
def model():
    from rh17m.bekko_mlx import load

    return load()


@pytest.fixture(scope="module")
def ref():
    pytest.importorskip("torch")
    from rh17m.bekko_mlx.reference import load_reference

    return load_reference()


@pytest.fixture(scope="module")
def cases(model):
    from rh17m.bekko_mlx.reference import parity_cases

    return parity_cases(model.tokenizer)


@pytest.fixture(scope="module")
def report(model, ref, cases):
    from rh17m.bekko_mlx.reference import parity

    result = parity(model, ref, cases)
    for row in result.rows:
        print(f"\n{row['case']:34s} {row['task']:6s} K={row['options']:2d} P={row['prefix_tokens']:5d} "
              f"tokens_equal={row['tokens_equal']} max|dp|={row['max_abs_diff']:.2e}")
    return result


def test_inputs_are_token_identical_to_the_runtime(report):
    assert all(row["tokens_equal"] for row in report.rows), [r["case"] for r in report.rows if not r["tokens_equal"]]


def test_choice_probabilities_match_runtime(report):
    rows = [row for row in report.rows if row["task"] == "choice"]
    assert len(rows) >= 10 and {2, 12} <= {row["options"] for row in rows}
    assert any(row["prefix_tokens"] >= 800 for row in rows) and any(row["prefix_tokens"] < 200 for row in rows)
    worst = max(row["max_abs_diff"] for row in rows)
    print(f"\nchoice parity: {len(rows)} requests, max |p_mlx - p_torch| = {worst:.3e}")
    assert worst <= PROB_TOL
    assert all(row["argmax_equal"] for row in rows)


def test_noul_and_score_probabilities_match_runtime(report):
    by_task = report.by_task()
    print(f"\nnoul max |dp| = {by_task['noul']:.3e}, score max |dp| = {by_task['score']:.3e}")
    assert by_task["noul"] <= PROB_TOL and by_task["score"] <= PROB_TOL


def test_typed_helpers_agree_with_runtime_outputs(model, ref, cases):
    from rh17m.bekko_mlx import choice_probs, noul_probability, score_prediction

    choice = next(c for c in cases if c.task == "choice")
    expected = ref.predict(choice.request(), prefix_layout=choice.layout, show_progress_bar=False)["decision"]
    got = choice_probs(model, None, choice.state_json, choice.instructions, choice.options, layout=choice.layout)
    assert max(got, key=got.get) == expected["selected_id"]
    noul = next(c for c in cases if c.task == "noul" and dict(c.options).keys() == {"true", "false"})
    expected = ref.predict(noul.request(), prefix_layout=noul.layout, show_progress_bar=False)["decision"]
    options = dict(noul.options)
    p_true = noul_probability(model, None, noul.state_json, noul.instructions, true=options["true"],
                              false=options["false"], layout=noul.layout)
    assert abs(p_true - expected["probability_yes"]) <= PROB_TOL
    score = next(c for c in cases if c.task == "score")
    expected = ref.predict(score.request(), prefix_layout=score.layout, show_progress_bar=False)["decision"]
    got = score_prediction(model, None, score.state_json, score.instructions, score.options, score.values,
                           layout=score.layout)
    assert abs(got["score"] - expected["score"]) <= PROB_TOL * (max(score.values) - min(score.values)) * len(score.values)
    assert abs(got["normalized_score"] - expected["normalized_score"]) <= PROB_TOL * len(score.values)


def test_mlx_save_loads_in_published_runtime(tmp_path, cases):
    """Fine-tuned MLX weights -> save() -> published BekkoSentenceTransformer (strict load) -> same probs."""
    from rh17m.bekko_mlx import decision_probs, load, save
    from rh17m.bekko_mlx.reference import load_reference, reference_probs

    tuned = load()
    mx.random.seed(0)
    tuned.update(tree_map(lambda p: p + 0.02 * mx.random.normal(p.shape) * mx.abs(p).mean(), tuned.parameters()))
    out = save(tuned, tmp_path / "tuned")
    reloaded_ref = load_reference(out)
    reloaded_mlx = load(out)
    worst_torch = worst_mlx = moved = 0.0
    original = load()
    for case in [c for c in cases if c.task == "choice"][:6] + [c for c in cases if c.task != "choice"]:
        encoded = case.encode(tuned.tokenizer)
        got = decision_probs(tuned, [encoded])[0]
        torch_probs = reference_probs(reloaded_ref, case)
        again = decision_probs(reloaded_mlx, [encoded])[0]
        before = decision_probs(original, [encoded])[0]
        worst_torch = max(worst_torch, max(abs(got[k] - torch_probs[k]) for k in got))
        worst_mlx = max(worst_mlx, max(abs(got[k] - again[k]) for k in got))
        moved = max(moved, max(abs(got[k] - before[k]) for k in got))
    print(f"\nround trip: MLX vs torch(saved) {worst_torch:.3e}; MLX vs MLX(saved) {worst_mlx:.3e}; "
          f"perturbation moved probs by up to {moved:.3e}")
    assert moved > 1e-3  # the check is meaningful only if the saved weights differ from the release
    assert worst_torch <= PROB_TOL
    assert worst_mlx == 0.0


def test_bf16_mode_is_close_but_not_a_parity_claim(model, cases):
    """bf16 is an approximate speed mode (the runtime itself warns BF16 can move probabilities);
    the bar here is a coarse sanity bound, documented with the measured value."""
    from rh17m.bekko_mlx import decision_probs, load

    bf16 = load(dtype=mx.bfloat16)
    diffs = []
    for case in cases:
        encoded = case.encode(model.tokenizer)
        a, b = decision_probs(model, [encoded])[0], decision_probs(bf16, [encoded])[0]
        diffs.append(max(abs(a[k] - b[k]) for k in a))
    print(f"\nbf16 vs fp32 max |dp| = {max(diffs):.3e} (median {np.median(diffs):.3e})")
    assert max(diffs) <= 0.05


def test_training_gradients_match_torch_autograd(model, cases):
    """The trainer's soft-target loss and every parameter gradient equal PyTorch autograd through the
    published BekkoInference.forward (mixed choice/noul/score batch, both layouts)."""
    from mlx import nn
    from mlx.utils import tree_flatten

    from rh17m.bekko_mlx import collate, loss_fn
    from rh17m.bekko_mlx.reference import load_reference, reference_loss_and_grads

    chosen = [c for c in cases if c.name in ("800-9", "short-3-state-first", "400-5", "800-12-system",
                                              "noul-800", "score-5")]
    rng = np.random.default_rng(0)
    targets = [list(rng.dirichlet(np.ones(len(c.options)))) for c in chosen]
    torch_loss, torch_grads = reference_loss_and_grads(load_reference(), chosen, targets)
    batch = collate([c.encode(model.tokenizer).with_target(t) for c, t in zip(chosen, targets, strict=True)],
                    tasks=model.tasks, pad_multiple=32)
    loss, grads = nn.value_and_grad(model, loss_fn)(model, batch)
    grads = dict(tree_flatten(grads))
    assert set(grads) == set(torch_grads)
    scale = max(np.abs(g).max() for g in torch_grads.values())
    worst = max(np.abs(np.array(grads[name]) - g).max() for name, g in torch_grads.items())
    print(f"\nloss torch {torch_loss:.7f} mlx {float(loss):.7f}; max |grad diff| {worst:.2e} "
          f"(largest gradient entry {scale:.2e})")
    assert abs(torch_loss - float(loss)) <= 1e-5
    assert worst <= 1e-4 * scale
