"""Differentiable training path: soft-target loss, gradients, AdamW steps, compiled steps."""

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from rh17m.bekko_mlx import (
    EncodedDecision,
    collate,
    kl_divergence,
    loss_fn,
    make_optimizer,
    make_train_step,
    random_model,
    soft_cross_entropy,
)


def _toy(seed=0, count=6, vocab=128):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        k = int(rng.integers(2, 6))
        target = rng.dirichlet(np.ones(k))
        out.append(EncodedDecision(
            "choice", tuple(f"o{j}" for j in range(k)), (1, *rng.integers(3, vocab, int(rng.integers(4, 30))).tolist(), 2),
            tuple((*rng.integers(3, vocab, int(rng.integers(1, 8))).tolist(), 2) for _ in range(k)),
            target=tuple(target / target.sum())))
    return out


def test_soft_cross_entropy_matches_reference_and_kl():
    logits = mx.array([[2.0, -1.0, 0.5, -1e9], [0.0, 1.0, -1e9, -1e9]])
    targets = mx.array([[0.2, 0.3, 0.5, 0.0], [0.9, 0.1, 0.0, 0.0]])
    rows = []
    for row, t in zip(np.array(logits), np.array(targets), strict=True):
        z = row - row.max()
        rows.append(-(t * (z - np.log(np.exp(z).sum()))).sum())
    assert abs(float(soft_cross_entropy(logits, targets)) - np.mean(rows)) < 1e-6
    assert float(kl_divergence(logits, targets)) >= 0
    probs = mx.softmax(logits, axis=-1)
    assert abs(float(kl_divergence(logits, probs))) < 1e-6


def test_gradients_are_finite_and_reach_every_parameter():
    model = random_model(seed=1)
    batch = collate(_toy(1), tasks=model.tasks, pad_multiple=16)
    loss, grads = nn.value_and_grad(model, loss_fn)(model, batch)
    flat = tree_flatten(grads)
    assert np.isfinite(float(loss))
    for name, grad in flat:
        if name.startswith(("heads.noul", "heads.score")):  # unused heads in a choice-only batch
            continue
        assert bool(mx.isfinite(grad).all()), name
        assert float(mx.abs(grad).max()) > 0, name


def test_adamw_steps_lower_the_loss_and_compiled_step_matches_eager():
    batch = collate(_toy(2), pad_multiple=16)
    losses = {}
    for compiled in (False, True):
        model = random_model(seed=2)
        step = make_train_step(model, make_optimizer(1e-3, 1e-3), compile=compiled)
        losses[compiled] = [float(step(batch)) for _ in range(8)]
    assert losses[False][-1] < losses[False][0]
    assert np.abs(np.array(losses[False]) - np.array(losses[True])).max() < 1e-4


@pytest.mark.integration
def test_published_weights_fine_tune_on_soft_targets():
    from rh17m.bekko_mlx import encode_choice, load, synthetic

    model = load()
    tok = model.tokenizer
    rng = np.random.default_rng(0)
    examples = []
    for seed in range(4):
        options = synthetic.options(9, seed)
        target = rng.dirichlet(np.full(len(options), 0.3))
        examples.append(encode_choice(tok, synthetic.state_json(tok, 300, seed), synthetic.INSTRUCTIONS, options)
                        .with_target(target / target.sum()))
    batch = collate(examples, tasks=model.tasks)
    step = make_train_step(model, make_optimizer())
    losses = [float(step(batch)) for _ in range(6)]
    print("\nfine-tune losses:", [round(x, 4) for x in losses])
    assert losses[-1] < losses[0]
