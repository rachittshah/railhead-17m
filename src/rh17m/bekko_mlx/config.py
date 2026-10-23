"""Pinned release identity and validated configuration for the Bekko v0 shared-prefix scorer.

Only the features the published 17M checkpoint uses are implemented. Anything else (dropout, biases,
non-default RoPE, the optional ChoiceInteraction block, task marker tokens with missing vocabulary) is
rejected at load time instead of being silently approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from decision_mlx.bekko_provider import BEKKO_RELEASES

REPO_ID = "hotchpotch/bekko-system-one-v0-17m"
REVISION = BEKKO_RELEASES[REPO_ID]
MODULE_DIR = "0_BekkoInference"
TASKS = ("choice", "noul", "score")
LAYOUTS = ("instruction_state", "state_instruction")
# Files the published SentenceTransformers export needs at its root to load with trust_remote_code.
RUNTIME_FILES = ("inference_v0.py", "modules.json", "config_sentence_transformers.json", "requirements.txt")
SNAPSHOT_PATTERNS = ["*.py", "*.json", "*.md", "*.txt", f"{MODULE_DIR}/*", f"{MODULE_DIR}/tokenizer/*"]


@dataclass(frozen=True)
class BackboneConfig:
    """The ModernBERT fields that determine the computation; ``raw`` is written back verbatim on save."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    norm_eps: float
    layer_types: tuple[str, ...]
    rope_theta: dict[str, float]
    # Inclusive half window of sliding layers: a query at position p reads keys with |p - k| <= window.
    # transformers sets attn.sliding_window = local_attention // 2 + 1 and the runtime subtracts one.
    window: int
    pad_token_id: int
    cls_token_id: int
    sep_token_id: int
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, raw: dict) -> BackboneConfig:
        if raw.get("model_type") != "modernbert":
            raise ValueError("expected a ModernBERT backbone config")
        if raw.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("only the exact (erf) GELU GeGLU activation is implemented")
        for flag in ("attention_bias", "mlp_bias", "norm_bias"):
            if raw.get(flag, False):
                raise ValueError(f"{flag}=true is not implemented")
        for rate in ("attention_dropout", "embedding_dropout", "mlp_dropout"):
            if raw.get(rate, 0.0):
                raise ValueError(f"nonzero {rate} is not implemented")
        if raw["hidden_size"] % raw["num_attention_heads"]:
            raise ValueError("hidden size must divide into attention heads")
        layer_types = tuple(raw["layer_types"])
        if len(layer_types) != raw["num_hidden_layers"] or set(layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError("unsupported layer schedule")
        rope = raw["rope_parameters"]
        thetas = {}
        for kind in set(layer_types):
            if rope[kind].get("rope_type", "default") != "default":
                raise ValueError("only default RoPE is implemented")
            thetas[kind] = float(rope[kind]["rope_theta"])
        return cls(
            vocab_size=raw["vocab_size"], hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"], num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            max_position_embeddings=raw["max_position_embeddings"], norm_eps=float(raw["norm_eps"]),
            layer_types=layer_types, rope_theta=thetas, window=raw["local_attention"] // 2,
            pad_token_id=raw["pad_token_id"], cls_token_id=raw["cls_token_id"],
            sep_token_id=raw["sep_token_id"], raw=dict(raw),
        )


@dataclass(frozen=True)
class InferenceSettings:
    """inference_config.json, resolved exactly as BekkoInference.__init__ resolves it."""

    tasks: tuple[str, ...]
    query_length: int
    document_length: int
    context_length: int
    query_truncation: str
    task_tokens: dict[str, str]
    prefix_layout: str
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(cls, raw: dict, max_positions: int) -> InferenceSettings:
        if raw.get("choice_interaction"):
            raise NotImplementedError("checkpoints with a ChoiceInteraction block are not supported")
        capacity = raw.get("context_length") or max_positions
        if not 5 <= capacity <= max_positions:
            raise ValueError("context length exceeds the backbone positional capacity")
        query = raw.get("query_length") or capacity - 2
        document = raw.get("document_length") or min(3800, capacity - 3)
        if not 3 <= query <= capacity - 2 or not 2 <= document <= capacity - 3:
            raise ValueError("invalid query/document limits for this backbone")
        tasks = tuple(raw["tasks"])
        if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - {*TASKS, "reranker"}:
            raise ValueError("expected distinct supported task heads")
        truncation = raw.get("query_truncation", "balanced")
        layout = raw.get("prefix_layout", "instruction_state")
        if truncation not in {"right", "balanced"} or layout not in LAYOUTS:
            raise ValueError("invalid truncation or prefix layout")
        return cls(tasks=tasks, query_length=query, document_length=document, context_length=capacity,
                   query_truncation=truncation, task_tokens=dict(raw.get("task_tokens") or {}),
                   prefix_layout=layout, raw=dict(raw))
