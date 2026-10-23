"""Batching invariance and branch isolation of the MLX Bekko forward pass.

Weight-free tests use a tiny random model with a 2-token sliding half-window so local layers really restrict
reads; the integration test repeats the invariance check on the published 17M weights.
"""

import mlx.core as mx
import numpy as np
import pytest

from rh17m.bekko_mlx import NEG, EncodedDecision, choice_logits, collate, random_model

TOL = 1e-5


def _examples(seed, count, vocab=128):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        options = int(rng.integers(2, 7))
        prefix = (1, *rng.integers(3, vocab, int(rng.integers(3, 40))).tolist(), 2)
        branches = tuple((*rng.integers(3, vocab, int(rng.integers(1, 12))).tolist(), 2) for _ in range(options))
        out.append(EncodedDecision(("choice", "noul", "score")[i % 3], tuple(f"o{j}" for j in range(options)),
                                   prefix, branches))
    return out


def _single(model, examples):
    return [np.array(choice_logits(model, collate([e], tasks=model.tasks)))[0] for e in examples]


@pytest.fixture(scope="module")
def tiny():
    return random_model(seed=3)


def test_batched_logits_equal_single_example_logits(tiny):
    examples = _examples(0, 7)
    batched = np.array(choice_logits(tiny, collate(examples, tasks=tiny.tasks)))
    for row, (example, single) in enumerate(zip(examples, _single(tiny, examples), strict=True)):
        k = len(example.keys)
        assert np.abs(batched[row, :k] - single).max() <= TOL
        assert (batched[row, k:] == NEG).all()


def test_padding_buckets_do_not_change_logits(tiny):
    examples = _examples(1, 5)
    plain = np.array(choice_logits(tiny, collate(examples, tasks=tiny.tasks)))
    padded = np.array(choice_logits(tiny, collate(examples, tasks=tiny.tasks, pad_multiple=64, max_options=11)))
    assert np.abs(padded[:, : plain.shape[1]] - plain).max() <= TOL


def test_option_branches_are_isolated(tiny):
    """Editing one option never moves another option's logit (no cross-branch or option->prefix reads)."""
    (example,) = _examples(2, 1)
    example = EncodedDecision("choice", ("a", "b", "c"), example.prefix_ids, ((5, 6, 7, 2), (8, 9, 2), (10, 2)))
    edited = EncodedDecision("choice", ("a", "b", "c"), example.prefix_ids, ((50, 60, 70, 80, 90, 2), (8, 9, 2), (10, 2)))
    extended = EncodedDecision("choice", ("a", "b", "c", "d"), example.prefix_ids, (*example.option_ids, (11, 12, 2)))
    base, changed, more = (_single(tiny, [e])[0] for e in (example, edited, extended))
    assert abs(base[0] - changed[0]) > 1e-3
    assert np.abs(base[1:] - changed[1:]).max() <= TOL
    assert np.abs(base - more[:3]).max() <= TOL


def test_sliding_window_limits_option_reads_of_the_prefix(tiny):
    """With half-window 2, prefix tokens far from the options reach them only through full-attention layers;
    a model whose layers are all sliding must ignore edits to the start of a long prefix."""
    local = random_model(seed=4, layer_types=("sliding_attention",) * 3)
    prefix = (1, *range(3, 60), 2)
    a = EncodedDecision("choice", ("x", "y"), prefix, ((5, 2), (6, 7, 2)))
    b = EncodedDecision("choice", ("x", "y"), (1, 99, *prefix[2:]), ((5, 2), (6, 7, 2)))
    assert np.abs(_single(local, [a])[0] - _single(local, [b])[0]).max() == 0.0
    assert np.abs(_single(tiny, [a])[0] - _single(tiny, [b])[0]).max() > 1e-6



def test_batch_head_order_must_match_model(tiny):
    (example,) = _examples(5, 1)
    with pytest.raises(ValueError, match="tasks=model.tasks"):
        choice_logits(tiny, collate([example], tasks=("score", "noul", "choice")))

@pytest.mark.integration
def test_batched_logits_equal_single_on_published_weights():
    from rh17m.bekko_mlx import encode_decision, load, synthetic

    model = load()
    tok = model.tokenizer
    examples = [encode_decision(tok, synthetic.state_json(tok, n, seed), synthetic.INSTRUCTIONS,
                                synthetic.options(k, seed), layout=layout)
                for seed, (n, k, layout) in enumerate([(800, 9, "instruction_state"), (120, 2, "state_instruction"),
                                                       (500, 5, "instruction_state"), (950, 12, "state_instruction"),
                                                       (300, 9, "instruction_state"), (60, 3, "instruction_state")])]
    examples.append(encode_decision(tok, synthetic.state_json(tok, 400, 9), "Extra loan?",
                                    [("false", "no"), ("true", "yes")], task="noul"))
    batched = np.array(choice_logits(model, collate(examples, tasks=model.tasks, pad_multiple=32)))
    worst = max(np.abs(batched[row, : len(e.keys)] - single).max()
                for row, (e, single) in enumerate(zip(examples, _single(model, examples), strict=True)))
    print(f"\nbatched-vs-single max |logit diff| on published weights: {worst:.3e}")
    assert worst <= TOL
    mx.clear_cache()
