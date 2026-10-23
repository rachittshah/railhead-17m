"""Native MLX Bekko v0: ModernBERT (Ettin) backbone run as a shared prefix plus isolated option branches.

Mirrors ``_SDPAPrefix`` + ``BekkoInference.forward`` of the published runtime:

* The prefix (instruction + state) is encoded bidirectionally and never reads option tokens.
* At every layer each option token reads the prefix keys/values of that same layer plus its own branch;
  branches never read each other. Option positions continue after their prefix (P + j).
* Sliding layers restrict every read to |q_pos - k_pos| <= window (64 for local_attention=128).
* Option representation = mean of final-normed branch tokens (including the branch's SEP); the task head
  is a Linear(hidden, 1) with bias, evaluated in fp32.

Option branches of one example are packed back to back in a single row (``doc_seg`` names the option of
each token, -1 for padding), so the prefix keys are never replicated per option.
Parameter names match the published ``state_dict`` exactly, so saved weights load in PyTorch strictly.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_map

from .config import BackboneConfig, InferenceSettings

NEG = -1e9  # option padding logit; finite so soft-target losses stay NaN-free (as the trainer's loss)


def _cast(x, dtype):
    """astype that adds no graph node when the dtype already matches (fp32 runs cast nothing)."""
    return x if x.dtype == dtype else x.astype(dtype)


class Linear(nn.Module):
    """Bias-free projection; the weight is cast to the activation dtype (fp32 master, bf16 compute)."""

    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        scale = math.sqrt(1.0 / in_dims)
        self.weight = mx.random.uniform(-scale, scale, (out_dims, in_dims))

    def __call__(self, x):
        return x @ _cast(self.weight, x.dtype).T


class LayerNorm(nn.Module):
    """LayerNorm without bias (norm_bias=false)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.layer_norm(x, _cast(self.weight, x.dtype), None, self.eps)


class Embeddings(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = LayerNorm(config.hidden_size, config.norm_eps)

    def __call__(self, ids, dtype):
        return self.norm(_cast(self.tok_embeddings(ids), dtype))


class Attention(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.Wqkv = Linear(config.hidden_size, 3 * config.hidden_size)
        self.Wo = Linear(config.hidden_size, config.hidden_size)
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        self.scale = self.head_dim**-0.5

    def qkv(self, x, cos, sin):
        batch, length, _ = x.shape
        # Wqkv output is laid out [3, heads, head_dim], as in .view(..., 3, -1, head_dim).
        qkv = self.Wqkv(x).reshape(batch, length, 3, self.heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        return _rotate(qkv[0], cos, sin), _rotate(qkv[1], cos, sin), qkv[2]


def _rotate(x, cos, sin):
    """transformers apply_rotary_pos_emb (rotate_half form), computed in fp32 and cast back."""
    half = x.shape[-1] // 2
    xf = _cast(x, mx.float32)
    rotated = mx.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
    return _cast(xf * cos + rotated * sin, x.dtype)


class MLP(nn.Module):
    """GeGLU: Wo(gelu(a) * gate) with (a, gate) = chunk(Wi(x), 2)."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.Wi = Linear(config.hidden_size, 2 * config.intermediate_size)
        self.Wo = Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        a, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(a) * gate)


class Layer(nn.Module):
    def __init__(self, config: BackboneConfig, index: int):
        super().__init__()
        if index:  # layer 0 uses nn.Identity for attn_norm (no weight in the checkpoint)
            self.attn_norm = LayerNorm(config.hidden_size, config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = LayerNorm(config.hidden_size, config.norm_eps)
        self.mlp = MLP(config)
        self.kind = config.layer_types[index]

    def normed(self, x):
        return self.attn_norm(x) if "attn_norm" in self else x

    def update(self, x, attended):
        batch, heads, length, dim = attended.shape
        x = x + self.attn.Wo(attended.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim))
        return x + self.mlp(self.mlp_norm(x))


class ModernBert(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = [Layer(config, i) for i in range(config.num_hidden_layers)]
        self.final_norm = LayerNorm(config.hidden_size, config.norm_eps)


class Encoder(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.backbone = ModernBert(config)


class _RopeTables:
    """cos/sin per layer kind for every position, bit-matching transformers' fp32 default RoPE.

    Kept outside the module tree so it is neither trained nor saved. inv_freq uses float32 pow (as torch);
    cos/sin of the float32 angle are taken in float64 and rounded, within 1 ulp of torch's float32 cos.
    """

    def __init__(self, config: BackboneConfig):
        dim = config.head_dim
        positions = np.arange(config.max_position_embeddings, dtype=np.float32)[:, None]
        self.tables = {}
        for kind, theta in config.rope_theta.items():
            exponent = np.arange(0, dim, 2).astype(np.float32) / np.float32(dim)
            inv_freq = (np.float32(1.0) / (np.float32(theta) ** exponent)).astype(np.float32)
            angles = (positions * inv_freq[None]).astype(np.float32)
            angles = np.concatenate([angles, angles], axis=-1).astype(np.float64)
            self.tables[kind] = (mx.array(np.cos(angles).astype(np.float32)),
                                 mx.array(np.sin(angles).astype(np.float32)))

    def __call__(self, kind, positions):
        cos, sin = self.tables[kind]
        return cos[positions][:, None], sin[positions][:, None]  # [B, 1, L, head_dim]


class BekkoMLX(nn.Module):
    """Bekko v0 scorer. Call with a :class:`~rh17m.bekko_mlx.batch.DecisionBatch` to get [B, K] logits."""

    def __init__(self, config: BackboneConfig, settings: InferenceSettings):
        super().__init__()
        self.encoder = Encoder(config)
        self.heads = {task: nn.Linear(config.hidden_size, 1) for task in settings.tasks}
        self.config, self.settings = config, settings
        self.compute_dtype = mx.float32
        self.tokenizer = None
        self.source_dir = None
        self._rope = _RopeTables(config)

    @property
    def tasks(self) -> tuple[str, ...]:
        """Head order used by ``DecisionBatch.task_index``."""
        return self.settings.tasks

    def set_compute_dtype(self, dtype) -> BekkoMLX:
        """Encoder activation dtype. Parameters keep their own dtype (fp32 master weights for bf16 compute);
        pooling, heads and softmax always run in fp32."""
        self.compute_dtype = dtype
        return self

    def _masks(self, prefix_len, prefix_width, doc_seg, doc_pos):
        """Boolean attention layouts {sliding: (prefix [B,1,P|1,P], options [B,1,T,P+T])}."""
        window = self.config.window
        keys = mx.arange(prefix_width)
        prefix_valid = keys[None] < prefix_len[:, None]  # [B, P]
        # Padded prefix queries sit at their row's last position (cumsum - 1), so no row is ever empty.
        prefix_pos = mx.minimum(keys[None], prefix_len[:, None] - 1)
        option_pos = doc_pos + prefix_len[:, None]  # [B, T]
        same = (doc_seg[:, :, None] == doc_seg[:, None, :]) & (doc_seg[:, None, :] >= 0)  # [B, T, T]
        full_prefix = mx.broadcast_to(prefix_valid[:, None, :], (*doc_seg.shape, prefix_width))
        layouts = {}
        kinds = set(self.config.layer_types)
        if "full_attention" in kinds:
            layouts["full_attention"] = (prefix_valid[:, None, None, :],
                                         mx.concatenate([full_prefix, same], axis=-1)[:, None])
        if "sliding_attention" in kinds:
            near = mx.abs(prefix_pos[:, :, None] - keys[None, None, :]) <= window
            option_near = mx.abs(option_pos[:, :, None] - keys[None, None, :]) <= window
            branch_near = mx.abs(doc_pos[:, :, None] - doc_pos[:, None, :]) <= window
            layouts["sliding_attention"] = (
                (prefix_valid[:, None, :] & near)[:, None],
                mx.concatenate([full_prefix & option_near, same & branch_near], axis=-1)[:, None])
        return layouts

    def pooled(self, prefix_ids, prefix_len, doc_ids, doc_seg, doc_pos, num_options: int):
        """Mean-pooled final branch states, fp32 [B, K, hidden]."""
        backbone = self.encoder.backbone
        dtype = self.compute_dtype
        prefix_width = prefix_ids.shape[1]
        layouts = self._masks(prefix_len, prefix_width, doc_seg, doc_pos)
        prefix_positions = mx.broadcast_to(mx.arange(prefix_width)[None], prefix_ids.shape)
        option_positions = doc_pos + prefix_len[:, None]
        rotary = {kind: (self._rope(kind, prefix_positions), self._rope(kind, option_positions))
                  for kind in layouts}
        prefix = backbone.embeddings(prefix_ids, dtype)
        hidden = backbone.embeddings(doc_ids, dtype)
        last = len(backbone.layers) - 1
        for index, layer in enumerate(backbone.layers):
            attention = layer.attn
            prefix_mask, option_mask = layouts[layer.kind]
            (pcos, psin), (ocos, osin) = rotary[layer.kind]
            pq, pk, pv = attention.qkv(layer.normed(prefix), pcos, psin)
            q, k, v = attention.qkv(layer.normed(hidden), ocos, osin)
            k, v = mx.concatenate([pk, k], axis=2), mx.concatenate([pv, v], axis=2)
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=attention.scale, mask=option_mask)
            if index < last:  # the last prefix update is never read by an option branch
                prefix = layer.update(prefix, mx.fast.scaled_dot_product_attention(
                    pq, pk, pv, scale=attention.scale, mask=prefix_mask))
            hidden = layer.update(hidden, attended)
        hidden = _cast(backbone.final_norm(hidden), mx.float32)
        onehot = (doc_seg[:, None, :] == mx.arange(num_options)[None, :, None]).astype(mx.float32)
        counts = mx.maximum(onehot.sum(axis=-1, keepdims=True), 1.0)
        return (onehot @ hidden) / counts

    def logits_from_arrays(self, prefix_ids, prefix_len, doc_ids, doc_seg, doc_pos, option_mask, task_index):
        """[B, K] fp32 logits from each example's task head; padded options hold NEG."""
        pooled = self.pooled(prefix_ids, prefix_len, doc_ids, doc_seg, doc_pos, option_mask.shape[1])
        weight = _cast(mx.concatenate([self.heads[t].weight for t in self.tasks], axis=0), mx.float32)
        bias = _cast(mx.concatenate([self.heads[t].bias for t in self.tasks], axis=0), mx.float32)
        every = pooled @ weight.T + bias  # [B, K, tasks]
        index = mx.broadcast_to(task_index[:, None, None], (*option_mask.shape, 1))
        logits = mx.take_along_axis(every, index, axis=-1)[..., 0]
        return mx.where(option_mask, logits, NEG)

    def __call__(self, batch):
        check_tasks(self, batch)
        return self.logits_from_arrays(*batch.arrays())


def check_tasks(model, batch):
    if tuple(batch.tasks) != tuple(model.tasks):
        raise ValueError(f"batch was collated for heads {batch.tasks}, model has {model.tasks}; "
                         "pass tasks=model.tasks to collate")


def random_model(seed: int = 0, *, hidden: int = 32, heads: int = 2, intermediate: int = 48,
                 layer_types=("full_attention", "sliding_attention", "full_attention"), local_attention: int = 4,
                 vocab: int = 128, max_positions: int = 512) -> BekkoMLX:
    """A tiny randomly initialised BekkoMLX with the v0 layout, for weight-free tests of masking,
    batching and training mechanics (small window so sliding layers actually restrict reads)."""
    raw = {"model_type": "modernbert", "vocab_size": vocab, "hidden_size": hidden,
           "intermediate_size": intermediate, "num_hidden_layers": len(layer_types),
           "num_attention_heads": heads, "max_position_embeddings": max_positions, "norm_eps": 1e-5,
           "layer_types": list(layer_types), "local_attention": local_attention,
           "rope_parameters": {kind: {"rope_theta": 160000.0, "rope_type": "default"} for kind in set(layer_types)},
           "pad_token_id": 0, "cls_token_id": 1, "sep_token_id": 2}
    config = BackboneConfig.from_dict(raw)
    settings = InferenceSettings.from_dict({"tasks": ["choice", "noul", "score"]}, max_positions)
    mx.random.seed(seed)
    model = BekkoMLX(config, settings)
    # Norm weights start at one; perturb them so tests exercise every parameter.
    model.update(tree_map(lambda p: p + 0.1 * mx.random.normal(p.shape), model.parameters()))
    mx.eval(model.parameters())
    return model
