"""Batched, differentiable scoring: collate encoded decisions, get [B, K] logits, interpret probabilities.

Each example is one prefix plus its K options. Prefixes are right-padded to P; the K branches of an example
are packed back to back into one row of width T (no per-option padding). Padding never changes results:
padded keys are masked out and padded options get the NEG logit. ``pad_multiple`` rounds P and T up to
bound the number of distinct shapes (useful with ``mx.compile``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .config import TASKS
from .encoding import BekkoTokenizer, EncodedDecision, encode_decision

PAD_ID = 50283  # ModernBERT [PAD]; any id works because padded tokens are masked.


@dataclass(frozen=True)
class DecisionBatch:
    prefix_ids: mx.array  # [B, P] int32
    prefix_len: mx.array  # [B] int32
    doc_ids: mx.array  # [B, T] int32, option branches packed per row
    doc_seg: mx.array  # [B, T] int32 option index of each token, -1 for padding
    doc_pos: mx.array  # [B, T] int32 position inside its branch (0 for padding)
    option_mask: mx.array  # [B, K] bool
    task_index: mx.array  # [B] int32 index into the model's task heads
    targets: mx.array | None  # [B, K] fp32 soft targets (0 on padded options) or None
    keys: tuple[tuple[str, ...], ...]
    tasks: tuple[str, ...] = TASKS  # head order task_index refers to; the model checks it

    def arrays(self):
        return (self.prefix_ids, self.prefix_len, self.doc_ids, self.doc_seg, self.doc_pos,
                self.option_mask, self.task_index)

    @property
    def size(self) -> int:
        return len(self.keys)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def collate(examples: Sequence[EncodedDecision], *, tasks: Sequence[str] = TASKS, pad_multiple: int = 1,
            pad_id: int = PAD_ID, max_options: int | None = None) -> DecisionBatch:
    """Stack encoded decisions. ``tasks`` must be the model's head order (``model.tasks``)."""
    if not examples:
        raise ValueError("collate needs at least one example")
    with_targets = {e.target is not None for e in examples}
    if len(with_targets) != 1:
        raise ValueError("either every example or none must carry a target")
    batch = len(examples)
    width_p = _round_up(max(len(e.prefix_ids) for e in examples), pad_multiple)
    width_t = _round_up(max(sum(map(len, e.option_ids)) for e in examples), pad_multiple)
    count = max(len(e.option_ids) for e in examples)
    if max_options is not None:
        if max_options < count:
            raise ValueError("an example has more options than max_options")
        count = max_options
    prefix_ids = np.full((batch, width_p), pad_id, dtype=np.int32)
    doc_ids = np.full((batch, width_t), pad_id, dtype=np.int32)
    doc_seg = np.full((batch, width_t), -1, dtype=np.int32)
    doc_pos = np.zeros((batch, width_t), dtype=np.int32)
    option_mask = np.zeros((batch, count), dtype=bool)
    targets = np.zeros((batch, count), dtype=np.float32)
    task_index = np.zeros(batch, dtype=np.int32)
    for row, example in enumerate(examples):
        if example.task not in tasks:
            raise ValueError(f"model has no {example.task} head")
        task_index[row] = tasks.index(example.task)
        prefix_ids[row, : len(example.prefix_ids)] = example.prefix_ids
        offset = 0
        for option, ids in enumerate(example.option_ids):
            doc_ids[row, offset : offset + len(ids)] = ids
            doc_seg[row, offset : offset + len(ids)] = option
            doc_pos[row, offset : offset + len(ids)] = np.arange(len(ids))
            offset += len(ids)
        option_mask[row, : len(example.option_ids)] = True
        if example.target is not None:
            targets[row, : len(example.target)] = example.target
    return DecisionBatch(
        prefix_ids=mx.array(prefix_ids), prefix_len=mx.array([len(e.prefix_ids) for e in examples], dtype=mx.int32),
        doc_ids=mx.array(doc_ids), doc_seg=mx.array(doc_seg), doc_pos=mx.array(doc_pos),
        option_mask=mx.array(option_mask), task_index=mx.array(task_index),
        targets=mx.array(targets) if with_targets == {True} else None,
        keys=tuple(e.keys for e in examples), tasks=tuple(tasks))


def choice_logits(model, batch: DecisionBatch) -> mx.array:
    """Differentiable [B, max_options] fp32 logits (each example's task head; padded options = NEG).

    Use with ``batch.option_mask``; probabilities are ``mx.softmax(logits, axis=-1)``.
    """
    return model(batch)


def decision_probs(model, examples: Sequence[EncodedDecision], *, pad_multiple: int = 1) -> list[dict[str, float]]:
    """Per-decision fp32 softmax over options, keyed by option key, for any mix of tasks."""
    batch = collate(examples, tasks=model.tasks, pad_multiple=pad_multiple)
    probabilities = mx.softmax(choice_logits(model, batch), axis=-1)
    mx.eval(probabilities)
    rows = np.array(probabilities)
    return [{key: float(p) for key, p in zip(keys, row[: len(keys)], strict=True)}
            for keys, row in zip(batch.keys, rows, strict=True)]


def _tokenizer(model, tok):
    tok = tok if tok is not None else model.tokenizer
    if not isinstance(tok, BekkoTokenizer):
        raise TypeError("a BekkoTokenizer is required (load() attaches one as model.tokenizer)")
    return tok


def choice_probs(model, tok: BekkoTokenizer | None, state_json: str, instructions, options, *,
                 layout: str | None = None, system: str = "", **limits) -> dict[str, float]:
    """Probability of each option key for one choice question; ``tok=None`` uses ``model.tokenizer``."""
    encoded = encode_decision(_tokenizer(model, tok), state_json, instructions, options, task="choice",
                              layout=layout, system=system, **limits)
    return decision_probs(model, [encoded])[0]


def noul_probability(model, tok: BekkoTokenizer | None, state_json: str, instructions, *, true: str, false: str,
                     layout: str | None = None, system: str = "", **limits) -> float:
    """P(true) for a binary question with authored descriptions of both meanings."""
    encoded = encode_decision(_tokenizer(model, tok), state_json, instructions,
                              [("false", false), ("true", true)], task="noul", layout=layout, system=system,
                              **limits)
    return decision_probs(model, [encoded])[0]["true"]


def score_prediction(model, tok: BekkoTokenizer | None, state_json: str, instructions, levels, values, *,
                     layout: str | None = None, system: str = "", **limits) -> dict:
    """Ordinal score: probabilities over (key, description) levels and the expected numeric value."""
    encoded = encode_decision(_tokenizer(model, tok), state_json, instructions, levels, task="score",
                              values=values, layout=layout, system=system, **limits)
    probabilities = decision_probs(model, [encoded])[0]
    score = sum(probabilities[key] * value for key, value in zip(encoded.keys, encoded.values, strict=True))
    low, high = min(encoded.values), max(encoded.values)
    return {"score": score, "normalized_score": (score - low) / (high - low), "probabilities": probabilities,
            "values": dict(zip(encoded.keys, encoded.values, strict=True))}
