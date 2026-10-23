"""Native MLX port of Bekko System One v0 17M (inference and training), parity-tested against the
published PyTorch runtime. See docs/bekko-mlx.md.

    model = load()                                  # pinned hotchpotch/bekko-system-one-v0-17m
    probs = choice_probs(model, None, state_json, "Which route?", [("r1", "..."), ("hold", "...")])

    batch = collate([encode_choice(model.tokenizer, s, ins, opts).with_target(t) for ...], tasks=model.tasks)
    loss, grads = nn.value_and_grad(model, loss_fn)(model, batch)
"""

from .batch import (
                    DecisionBatch,
                    choice_logits,
                    choice_probs,
                    collate,
                    decision_probs,
                    noul_probability,
                    score_prediction,
)
from .config import REPO_ID, REVISION, TASKS, BackboneConfig, InferenceSettings
from .encoding import (
                    BekkoTokenizer,
                    EncodedChoice,
                    EncodedDecision,
                    dump_encoded,
                    encode_choice,
                    encode_decision,
                    load_encoded,
                    native_request,
)
from .io import load, load_tokenizer, resolve, save, state_dict
from .model import NEG, BekkoMLX, random_model
from .train import (
                    kl_divergence,
                    loss_fn,
                    make_optimizer,
                    make_train_step,
                    recipe_schedule,
                    soft_cross_entropy,
)

__all__ = [
    "NEG", "REPO_ID", "REVISION", "TASKS", "BackboneConfig", "BekkoMLX", "BekkoTokenizer", "DecisionBatch",
    "EncodedChoice", "EncodedDecision", "InferenceSettings", "choice_logits", "choice_probs", "collate",
    "decision_probs", "dump_encoded", "encode_choice", "encode_decision", "kl_divergence", "load", "load_encoded", "load_tokenizer", "loss_fn",
    "make_optimizer", "make_train_step", "native_request", "noul_probability", "random_model", "recipe_schedule", "resolve", "save",
    "score_prediction", "soft_cross_entropy", "state_dict",
]
