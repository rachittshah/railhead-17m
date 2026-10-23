"""Soft-target training utilities: the trainer's groupwise cross entropy, AdamW groups and a train step.

``soft_cross_entropy`` is bekko_system_one.loss.distribution_loss_sum / number of decisions: an fp32
cross entropy against a target distribution, with padded options at a large negative logit. Its gradient
equals that of KL(target || model); ``kl_divergence`` subtracts the target entropy for reporting.
"""

from __future__ import annotations

import math
from functools import partial

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn

from .batch import DecisionBatch, choice_logits
from .model import check_tasks


def soft_cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Mean over decisions of -sum_k t_k log softmax(logits)_k (padded options carry t_k = 0)."""
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -(targets * log_probs).sum(axis=-1).mean()


def kl_divergence(logits: mx.array, targets: mx.array) -> mx.array:
    entropy = -(mx.where(targets > 0, targets * mx.log(mx.maximum(targets, 1e-30)), 0.0)).sum(axis=-1).mean()
    return soft_cross_entropy(logits, targets) - entropy


def loss_fn(model, batch: DecisionBatch) -> mx.array:
    if batch.targets is None:
        raise ValueError("training batches need targets")
    return soft_cross_entropy(choice_logits(model, batch), batch.targets)


def recipe_schedule(peak: float, total_steps: int, warmup_ratio: float = 0.1):
    """The v0 recipe's schedule: linear warmup over ceil(10%) of steps, then cosine decay to zero
    (transformers.get_cosine_schedule_with_warmup)."""
    warmup = max(1, math.ceil(total_steps * warmup_ratio))
    return optim.join_schedules([optim.linear_schedule(0.0, peak, warmup),
                                 optim.cosine_decay(peak, max(1, total_steps - warmup))], [warmup])


def make_optimizer(learning_rate=1e-4, head_learning_rate=5e-4, weight_decay=0.01, *,
                   total_steps: int | None = None, warmup_ratio: float = 0.1) -> optim.Optimizer:
    """AdamW as bekko_system_one.training.optimizer_for: encoder 1e-4, task heads 5e-4, decay 0.01 on every
    parameter. With ``total_steps`` both rates follow :func:`recipe_schedule`; rates may also be schedules."""
    if total_steps is not None:
        learning_rate = recipe_schedule(learning_rate, total_steps, warmup_ratio)
        head_learning_rate = recipe_schedule(head_learning_rate, total_steps, warmup_ratio)
    return optim.MultiOptimizer(
        [optim.AdamW(learning_rate=head_learning_rate, weight_decay=weight_decay),
         optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)],
        [lambda path, _: path.startswith("heads.")])


def make_train_step(model, optimizer, *, max_grad_norm: float | None = 1.0, compile: bool = False):
    """Return ``step(batch) -> loss``: value_and_grad, global-norm clipping (1.0, as the v0 trainer), update.

    With ``compile=True`` the step is traced per input shape (bucket shapes with collate's pad_multiple).
    """
    value_and_grad = nn.value_and_grad(model, lambda m, arrays, targets: soft_cross_entropy(
        m.logits_from_arrays(*arrays), targets))

    def raw(arrays, targets):
        loss, grads = value_and_grad(model, arrays, targets)
        if max_grad_norm is not None:
            grads, _ = optim.clip_grad_norm(grads, max_grad_norm)
        optimizer.update(model, grads)
        return loss

    if compile:
        state = [model.state, optimizer.state]
        raw = partial(mx.compile, inputs=state, outputs=state)(raw)

    def step(batch: DecisionBatch) -> mx.array:
        if batch.targets is None:
            raise ValueError("training batches need targets")
        check_tasks(model, batch)
        loss = raw(batch.arrays(), batch.targets)
        mx.eval(loss, model.parameters(), optimizer.state)
        return loss

    return step
