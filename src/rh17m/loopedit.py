"""Weight-space edit: turn the Bekko encoder into a looped transformer without a gradient step.

The 17M backbone is ``layer 0`` followed by a block pattern that already repeats: (sliding, sliding, full)
at layers 1-3 and again at 4-6. A looped transformer applies ONE block several times, so the edit makes the
weights of layers 4-6 equal to those of layers 1-3. The export stays unrolled (7 layers whose two blocks are
byte-identical), so the published runtime and the strict loaders read it unchanged; ``unroll`` writes the
same block at any other loop count (1 + block * loops layers is still a valid ModernBERT schedule).

Three kinds of edit, all closed form:

1. Exact re-parameterisations (the function does not change; ``teacher``):
   - fold each LayerNorm gain into the bias-free Linear that reads it;
   - re-gauge the later block onto the first: permute attention heads, rotate each head's value basis
     (orthogonal; undone in ``attn.Wo``), rotate each RoPE pair of the query/key basis by a common phase
     (commutes with the rotary rotation), permute MLP neurons and flip the gate sign (undone in ``mlp.Wo``).
2. Weights-only ties: copy either block over the other, or average the two (raw, or after the re-gauge).
3. Activation-aware ties (calibration inputs from train maps, no labels): ``regmean`` solves every tied
   Linear as the least-squares map that reproduces both layers' outputs on their own inputs
   (W = (sum_p W_p G_p)(sum_p G_p)^-1, G_p the input Gram at depth p); ``lsq`` then re-solves each tied
   Linear on the looped model's own activations so the residual stream tracks the original one
   (Gauss-Seidel sweeps; the output projections absorb the drift the earlier loop left behind).

Variants are ranked by fidelity to the unedited model on held-out train maps (KL of the option
distributions); labels are only read for the final val/test report.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from rh17m import bekko_mlx as B
from rh17m.bekko_mlx.model import NEG, _cast, _rotate

LINEARS = ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo")
NORM_OF = {"attn.Wqkv": "attn_norm", "mlp.Wi": "mlp_norm"}
# Per tied Linear: (captured input, captured target in the teacher, student stream the target is a delta from).
FIT = {
    "attn.Wqkv": ("attn_in", "qkv", None),
    "attn.Wo": ("att", "mid", "x"),
    "mlp.Wi": ("mlp_in", "ag", None),
    "mlp.Wo": ("h", "out", "mid"),
}
DATA_FREE = ("copy_first", "copy_last", "mean_raw", "mean_aligned")


def key(layer: int, name: str) -> str:
    return f"encoder.backbone.layers.{layer}.{name}.weight"


def weights(model) -> dict[str, np.ndarray]:
    """Every parameter under its published name, float64."""
    return {k: np.array(v, dtype=np.float64) for k, v in B.state_dict(model).items()}


def rebuild(source, w: dict[str, np.ndarray], layer_types: tuple[str, ...] | None = None):
    """A BekkoMLX with ``w`` (fp32) and, if given, another layer schedule; tokenizer/runtime from ``source``."""
    raw = dict(source.config.raw)
    if layer_types is not None:
        raw["layer_types"], raw["num_hidden_layers"] = list(layer_types), len(layer_types)
    model = B.BekkoMLX(B.BackboneConfig.from_dict(raw), source.settings)
    model.load_weights([(k, mx.array(v.astype(np.float32))) for k, v in w.items()], strict=True)
    model.tokenizer, model.source_dir = source.tokenizer, source.source_dir
    mx.eval(model.parameters())
    return model.eval()


def loops_of(layer_types: tuple[str, ...], block: int) -> int:
    """How many times the block after layer 0 repeats; raises unless the schedule is layer 0 + block * loops."""
    body = tuple(layer_types[1:])
    loops, rest = divmod(len(body), block)
    if rest or loops < 1 or any(body[i] != body[i % block] for i in range(len(body))):
        raise ValueError(f"layer schedule {layer_types} is not layer 0 plus a repeated block of {block}")
    return loops


# ------------------------------------------------------------------ exact re-parameterisations
def fold_norms(w: dict[str, np.ndarray], layers) -> dict[str, np.ndarray]:
    """Move each LayerNorm gain into the bias-free Linear that reads it (exact); the gains become ones."""
    w = dict(w)
    for layer in layers:
        for linear, norm in NORM_OF.items():
            gain = w[key(layer, norm)]
            w[key(layer, linear)] = w[key(layer, linear)] * gain[None, :]
            w[key(layer, norm)] = np.ones_like(gain)
    return w


def _complex(rows: np.ndarray) -> np.ndarray:
    """RoPE pairs dimension j with j + half: the pair is one complex coordinate."""
    half = rows.shape[0] // 2
    return rows[:half] + 1j * rows[half:]


def align_attention(w: dict[str, np.ndarray], src: int, dst: int, heads: int) -> dict:
    """Re-gauge layer ``dst``'s attention onto ``src`` in place (exact): head order, value basis, q/k phases.
    Returns the head-similarity matrix and the matching (for the figures)."""
    wq_s, wo_s = w[key(src, "attn.Wqkv")], w[key(src, "attn.Wo")]
    wq_d, wo_d = w[key(dst, "attn.Wqkv")], w[key(dst, "attn.Wo")]
    dim = wo_s.shape[0]
    hd = dim // heads
    qs, qd = wq_s.reshape(3, heads, hd, dim), wq_d.reshape(3, heads, hd, dim)
    os_, od = wo_s.reshape(dim, heads, hd), wo_d.reshape(dim, heads, hd)
    sim = np.zeros((heads, heads))
    gauge = {}
    for a in range(heads):
        for b in range(heads):
            # Value basis: the orthogonal R that best maps dst's (value rows, output columns) onto src's.
            vs = np.concatenate([qs[2, a], os_[:, a, :].T], axis=1)
            vd = np.concatenate([qd[2, b], od[:, b, :].T], axis=1)
            u, s, vt = np.linalg.svd(vs @ vd.T)
            # Query/key basis: one unit phase per RoPE pair, shared by q and k (so q.k is unchanged).
            inner = sum((_complex(qs[t, a]) * np.conj(_complex(qd[t, b]))).sum(axis=1) for t in (0, 1))
            mag = np.abs(inner)
            phase = np.where(mag > 1e-12, inner / np.maximum(mag, 1e-12), 1.0)
            qk = math.sqrt(sum(np.sum(qs[t, a] ** 2) for t in (0, 1)) * sum(np.sum(qd[t, b] ** 2) for t in (0, 1)))
            sim[a, b] = s.sum() / (np.linalg.norm(vs) * np.linalg.norm(vd) + 1e-12) + mag.sum() / (qk + 1e-12)
            gauge[a, b] = (u @ vt, phase)
    new_q, new_o = np.empty_like(qd), np.empty_like(od)
    rows, cols = linear_sum_assignment(-sim)
    for a, b in zip(rows, cols, strict=True):
        rot, phase = gauge[a, b]
        new_q[2, a] = rot @ qd[2, b]
        new_o[:, a, :] = od[:, b, :] @ rot.T
        for t in (0, 1):
            z = phase[:, None] * _complex(qd[t, b])
            new_q[t, a] = np.concatenate([z.real, z.imag], axis=0)
    w[key(dst, "attn.Wqkv")] = new_q.reshape(wq_d.shape)
    w[key(dst, "attn.Wo")] = new_o.reshape(wo_d.shape)
    return {"similarity": sim, "match": cols[np.argsort(rows)]}


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def align_mlp(w: dict[str, np.ndarray], src: int, dst: int) -> dict:
    """Re-gauge layer ``dst``'s GeGLU onto ``src`` in place (exact): neuron order and gate sign.
    Returns the neuron cost matrix, the order and the signs (for the figures)."""
    wi_s, wo_s = w[key(src, "mlp.Wi")], w[key(src, "mlp.Wo")]
    wi_d, wo_d = w[key(dst, "mlp.Wi")], w[key(dst, "mlp.Wo")]
    n = wo_s.shape[1]
    # gelu(a) * gate: the gate and the output column can flip sign together; a cannot (gelu is not odd).
    signed = _unit(wi_s[n:]) @ _unit(wi_d[n:]).T + _unit(wo_s.T) @ _unit(wo_d.T).T
    cost = _unit(wi_s[:n]) @ _unit(wi_d[:n]).T + np.abs(signed)
    rows, cols = linear_sum_assignment(-cost)
    order = cols[np.argsort(rows)]
    sign = np.where(signed[np.arange(n), order] < 0, -1.0, 1.0)
    w[key(dst, "mlp.Wi")] = np.concatenate([wi_d[:n][order], sign[:, None] * wi_d[n:][order]], axis=0)
    w[key(dst, "mlp.Wo")] = wo_d[:, order] * sign[None, :]
    return {"cost": cost, "order": order, "sign": sign}


def teacher_weights(w: dict[str, np.ndarray], block: int, loops: int, heads: int, *,
                    align: bool = True) -> dict[str, np.ndarray]:
    """The same function in the gauge the ties are made in: norms folded, later blocks aligned to the first."""
    out = fold_norms(w, range(1, 1 + block * loops))
    if align:
        for r in range(1, loops):
            for j in range(block):
                align_attention(out, 1 + j, 1 + r * block + j, heads)
                align_mlp(out, 1 + j, 1 + r * block + j)
    return out


# ------------------------------------------------------------------ ties
def tie(w: dict[str, np.ndarray], block: int, loops: int, how: str) -> dict[str, np.ndarray]:
    """One block for every loop: the first block, the last block, or the mean of all of them."""
    out = dict(w)
    for j in range(block):
        for name in (*LINEARS, *NORM_OF.values()):
            stack = [w[key(1 + r * block + j, name)] for r in range(loops)]
            value = {"first": stack[0], "last": stack[-1], "mean": sum(stack) / loops}[how]
            for r in range(loops):
                out[key(1 + r * block + j, name)] = value.copy()
    return out


def unroll(w: dict[str, np.ndarray], layer_types: tuple[str, ...], block: int,
           loops: int) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """The first block applied ``loops`` times: weights and schedule of the 1 + block * loops layer model."""
    marker = ".layers."
    out = {k: v for k, v in w.items() if marker not in k or int(k.split(marker)[1].split(".")[0]) == 0}
    for r in range(loops):
        for j in range(block):
            for name in (*LINEARS, *NORM_OF.values()):
                out[key(1 + r * block + j, name)] = w[key(1 + j, name)].copy()
    return out, (layer_types[0], *layer_types[1:1 + block] * loops)


def is_looped(w: dict[str, np.ndarray], block: int, loops: int) -> bool:
    """Every loop's block is bit-identical to the first."""
    return all(np.array_equal(w[key(1 + j, name)], w[key(1 + r * block + j, name)])
               for r in range(1, loops) for j in range(block) for name in (*LINEARS, *NORM_OF.values()))


def unique_parameters(w: dict[str, np.ndarray], block: int) -> dict[str, int]:
    """Parameters that carry information once the blocks are tied (embeddings, layer 0, one block, heads)."""
    marker = ".layers."
    def layer(k):
        return int(k.split(marker)[1].split(".")[0]) if marker in k else -1
    return {"all": sum(v.size for v in w.values()),
            "unique": sum(v.size for k, v in w.items() if layer(k) <= block),
            "encoder_layers": sum(v.size for k, v in w.items() if layer(k) >= 0),
            "unique_encoder_layers": sum(v.size for k, v in w.items() if 0 <= layer(k) <= block)}


# ------------------------------------------------------------------ forward pass with taps
def _merge(heads_out):
    batch, heads, length, dim = heads_out.shape
    return heads_out.transpose(0, 2, 1, 3).reshape(batch, length, heads * dim)


def _split(raw, attention, cos, sin):
    batch, length, _ = raw.shape
    qkv = raw.reshape(batch, length, 3, attention.heads, attention.head_dim).transpose(2, 0, 3, 1, 4)
    return _rotate(qkv[0], cos, sin), _rotate(qkv[1], cos, sin), qkv[2]


def walk(model, batch, want=frozenset()):
    """``model(batch)`` written out sub-layer by sub-layer. Returns (logits, taps) with
    ``taps[(layer, name)] = (prefix tensor or None, option tensor)`` for every pair in ``want``.

    Names: ``x`` stream entering the layer, ``attn_in`` its normed copy, ``qkv`` the Wqkv output before the
    rotary rotation, ``att`` the merged head outputs, ``mid`` the stream after attention, ``mlp_in``, ``ag``
    the Wi output, ``h`` = gelu(a) * gate, ``out`` the stream leaving the layer. The prefix is not updated
    in the last layer (no option reads it), so its taps from ``att`` on are None there.
    """
    prefix_ids, prefix_len, doc_ids, doc_seg, doc_pos, option_mask, task_index = batch.arrays()
    backbone, dtype = model.encoder.backbone, model.compute_dtype
    width = prefix_ids.shape[1]
    layouts = model._masks(prefix_len, width, doc_seg, doc_pos)
    prefix_positions = mx.broadcast_to(mx.arange(width)[None], prefix_ids.shape)
    option_positions = doc_pos + prefix_len[:, None]
    rotary = {kind: (model._rope(kind, prefix_positions), model._rope(kind, option_positions)) for kind in layouts}
    prefix, hidden = backbone.embeddings(prefix_ids, dtype), backbone.embeddings(doc_ids, dtype)
    taps = {}

    def tap(step, name, p, o):
        if (step, name) in want:
            taps[step, name] = (p, o)

    last = len(backbone.layers) - 1
    for step, layer in enumerate(backbone.layers):
        attention, live = layer.attn, step < last
        prefix_mask, option_mask_ = layouts[layer.kind]
        (pcos, psin), (ocos, osin) = rotary[layer.kind]
        tap(step, "x", prefix, hidden)
        p_in, o_in = layer.normed(prefix), layer.normed(hidden)
        tap(step, "attn_in", p_in, o_in)
        p_raw, o_raw = attention.Wqkv(p_in), attention.Wqkv(o_in)
        tap(step, "qkv", p_raw, o_raw)
        pq, pk, pv = _split(p_raw, attention, pcos, psin)
        q, k, v = _split(o_raw, attention, ocos, osin)
        o_att = _merge(mx.fast.scaled_dot_product_attention(
            q, mx.concatenate([pk, k], axis=2), mx.concatenate([pv, v], axis=2), scale=attention.scale,
            mask=option_mask_))
        p_att = _merge(mx.fast.scaled_dot_product_attention(pq, pk, pv, scale=attention.scale,
                                                            mask=prefix_mask)) if live else None
        tap(step, "att", p_att, o_att)
        if live:
            prefix = prefix + attention.Wo(p_att)
        hidden = hidden + attention.Wo(o_att)
        tap(step, "mid", prefix if live else None, hidden)
        p_mlp, o_mlp = (layer.mlp_norm(prefix) if live else None), layer.mlp_norm(hidden)
        tap(step, "mlp_in", p_mlp, o_mlp)
        p_ag, o_ag = (layer.mlp.Wi(p_mlp) if live else None), layer.mlp.Wi(o_mlp)
        tap(step, "ag", p_ag, o_ag)

        def geglu(ag):
            a, gate = mx.split(ag, 2, axis=-1)
            return nn.gelu(a) * gate

        p_h, o_h = (geglu(p_ag) if live else None), geglu(o_ag)
        tap(step, "h", p_h, o_h)
        if live:
            prefix = prefix + layer.mlp.Wo(p_h)
        hidden = hidden + layer.mlp.Wo(o_h)
        tap(step, "out", prefix if live else None, hidden)
    hidden = _cast(backbone.final_norm(hidden), mx.float32)
    count = option_mask.shape[1]
    onehot = (doc_seg[:, None, :] == mx.arange(count)[None, :, None]).astype(mx.float32)
    pooled = (onehot @ hidden) / mx.maximum(onehot.sum(axis=-1, keepdims=True), 1.0)
    weight = mx.concatenate([model.heads[t].weight for t in model.tasks], axis=0).astype(mx.float32)
    bias = mx.concatenate([model.heads[t].bias for t in model.tasks], axis=0).astype(mx.float32)
    every = pooled @ weight.T + bias
    index = mx.broadcast_to(task_index[:, None, None], (*option_mask.shape, 1))
    logits = mx.where(option_mask, mx.take_along_axis(every, index, axis=-1)[..., 0], NEG)
    return logits, taps


# ------------------------------------------------------------------ closed-form fits
def _moments(x, y, mask) -> tuple[np.ndarray, np.ndarray]:
    """Sums over the unmasked tokens of x x^T and y x^T (fp32 on device, returned float64)."""
    xm = (x * mask[..., None].astype(x.dtype)).reshape(-1, x.shape[-1])
    gram, cross = xm.T @ x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]).T @ xm
    mx.eval(gram, cross)
    return np.array(gram, dtype=np.float64), np.array(cross, dtype=np.float64)


def solve_tied(student, teacher, batches, items: list[tuple[tuple[int, ...], str]], ridge: float,
               option_weight: float = 1.0) -> list[np.ndarray]:
    """For each (layers, linear): the one weight that best maps the student's inputs at every listed depth to
    the teacher's targets there (ridge least squares over valid prefix and option tokens; an option token
    counts ``option_weight`` times a prefix token: options are few, and they are what the head reads)."""
    want_s, want_t = set(), set()
    for layers, linear in items:
        source, target, base = FIT[linear]
        for p in layers:
            want_s.add((p, source))
            want_t.add((p, target))
            if base:
                want_s.add((p, base))
    sums: list[list] = [[0.0, 0.0] for _ in items]
    for batch in batches:
        if teacher is student:
            got_s = got_t = walk(student, batch, want_s | want_t)[1]
        else:
            got_s, got_t = walk(student, batch, want_s)[1], walk(teacher, batch, want_t)[1]
        valid = (mx.arange(batch.prefix_ids.shape[1])[None] < batch.prefix_len[:, None], batch.doc_seg >= 0)
        for total, (layers, linear) in zip(sums, items, strict=True):
            source, target, base = FIT[linear]
            for p in layers:
                for side in (0, 1):
                    x = got_s[p, source][side]
                    if x is None:
                        continue
                    y = got_t[p, target][side]
                    if base:
                        y = y - got_s[p, base][side]
                    gram, cross = _moments(x, y, valid[side])
                    scale = option_weight if side else 1.0
                    total[0], total[1] = total[0] + scale * gram, total[1] + scale * cross
    out = []
    for gram, cross in sums:
        dim = gram.shape[0]
        out.append(np.linalg.solve(gram + ridge * np.trace(gram) / dim * np.eye(dim), cross.T).T)
    return out


def _assign(model, layers, linear: str, value: np.ndarray) -> None:
    sub, name = linear.split(".")
    tied = mx.array(value.astype(np.float32))
    for p in layers:
        getattr(getattr(model.encoder.backbone.layers[p], sub), name).weight = tied
    mx.eval(tied)


def _slots(block: int, loops: int) -> list[tuple[int, ...]]:
    return [tuple(1 + r * block + j for r in range(loops)) for j in range(block)]


def regmean(teacher, batches, block: int, loops: int, ridge: float,
            option_weight: float = 1.0) -> dict[str, np.ndarray]:
    """Every tied Linear solved on the teacher's own activations (each layer keeps its own inputs)."""
    items = [(layers, linear) for layers in _slots(block, loops) for linear in LINEARS]
    w = weights(teacher)
    for (layers, linear), value in zip(items, solve_tied(teacher, teacher, batches, items, ridge, option_weight),
                                       strict=True):
        for p in layers:
            w[key(p, linear)] = value.copy()
    return w


def refine(student, teacher, batches, block: int, loops: int, ridge: float, option_weight: float = 1.0) -> None:
    """One Gauss-Seidel sweep, in place: each tied Linear is re-solved on the looped model's own activations,
    in forward order, and written back before the next one is measured."""
    for layers in _slots(block, loops):
        for linear in LINEARS:
            _assign(student, layers, linear,
                    solve_tied(student, teacher, batches, [(layers, linear)], ridge, option_weight)[0])



# ------------------------------------------------------------------ measurement
def fidelity(reference: list[dict[str, float]], probs: list[dict[str, float]]) -> dict[str, float]:
    """How closely ``probs`` reproduces ``reference`` per decision: KL(reference || probs), same top pick,
    Spearman correlation of the option order."""
    kl, top, rho = [], [], []
    for ref, p in zip(reference, probs, strict=True):
        kl.append(sum(v * (math.log(max(v, 1e-30)) - math.log(max(p[k], 1e-30))) for k, v in ref.items() if v > 0))
        top.append(float(max(ref, key=ref.get) == max(p, key=p.get)))
        if len(ref) > 2:
            keys = sorted(ref)
            r = spearmanr([ref[k] for k in keys], [p[k] for k in keys]).statistic
            rho.append(0.0 if math.isnan(r) else float(r))
    n = len(kl)
    return {"kl": sum(kl) / n, "top1": sum(top) / n, "spearman": sum(rho) / max(len(rho), 1), "n": n}


def match_temperature(reference: list[dict[str, float]], probs: list[dict[str, float]]) -> float:
    """The temperature at which ``probs`` is closest (KL) to the already-tempered ``reference``; label-free."""
    from rh17m.train import temper

    grid = [math.exp(x) for x in np.linspace(math.log(0.2), math.log(8.0), 61)]
    return min(grid, key=lambda t: fidelity(reference, [temper(p, t) for p in probs])["kl"])


@dataclass
class LoopConfig:
    model: str
    block: int = 3  # layers per loop; layer 0 stays outside (it has no attention norm)
    calib: int = 384  # calibration decisions (train maps) for the activation-aware ties
    holdout: int = 400  # decisions from held-out train maps that rank the variants
    rounds: int = 3  # Gauss-Seidel sweeps of the lsq tie
    ridge: float = 1e-3  # relative to the mean input-Gram diagonal
    option_weight: float = 1.0  # weight of an option token against a prefix token in the fits
    batch_size: int = 8
    seed: int = 0
    extra_loops: tuple[int, ...] = (1, 3, 4)  # also report the chosen block at these loop counts


def selection(cfg: LoopConfig, examples: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    """Splits, the held-out train-map decisions that rank variants, and the calibration decisions."""
    split = {s: [e for e in examples if e["split"] == s] for s in ("train", "val", "test")}
    # Held-out TRAIN maps rank the variants; the calibration inputs come from the other train maps.
    held_maps = set(sorted({e["task_id"] for e in split["train"]})[::4])
    rng = random.Random(cfg.seed)
    held = [e for e in split["train"] if e["task_id"] in held_maps]
    held = rng.sample(held, min(cfg.holdout, len(held)))
    pool = [e for e in split["train"] if e["task_id"] not in held_maps]
    calib = rng.sample(pool, min(cfg.calib, len(pool)))
    return split, held, calib


def _batches(model, examples: list[dict], batch_size: int):
    from rh17m.train import LAYOUTS, _encoded

    rows = sorted(examples, key=lambda e: len(e["state_json"]))  # similar lengths together: less padding
    return [B.collate([_encoded(model.tokenizer, e, LAYOUTS[0]) for e in rows[i:i + batch_size]],
                      tasks=model.tasks, pad_multiple=64) for i in range(0, len(rows), batch_size)]


def _paired(examples, ours, theirs, metric: str) -> dict[str, float]:
    from rh17m.evaluate import cluster_bootstrap_diff, per_example

    diff, lo, hi = cluster_bootstrap_diff(per_example(examples, ours, metric), per_example(examples, theirs, metric),
                                          [e["task_id"] for e in examples])
    return {"diff": diff, "lo": lo, "hi": hi}


def run(cfg: LoopConfig, labels_dir: Path, out_dir: Path, log=print) -> dict:
    from rh17m.dataset import load_examples
    from rh17m.evaluate import metrics
    from rh17m.train import clear_mlx_cache, limit_mlx_cache, predict, temper

    limit_mlx_cache()
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    source = B.load(cfg.model, revision=None)
    meta = Path(cfg.model) / "rh17m.json"
    base_t = json.loads(meta.read_text())["temperature"] if meta.exists() else 1.0
    types, heads = source.config.layer_types, source.config.num_attention_heads
    loops = loops_of(types, cfg.block)
    if loops < 2:
        raise ValueError("the model has a single block: nothing to tie")

    split, held, calib = selection(cfg, load_examples(Path(labels_dir)))
    log(json.dumps({"examples": {k: len(v) for k, v in split.items()}, "calib": len(calib), "holdout": len(held),
                    "layers": len(types), "block": cfg.block, "loops": loops}))

    w0 = weights(source)
    teacher_w = teacher_weights(w0, cfg.block, loops, heads)
    teacher = rebuild(source, teacher_w)
    ref_held = predict(source, held)
    exact = fidelity(ref_held, predict(teacher, held))
    log(json.dumps({"re-gauge (must be exact)": exact}))
    if exact["kl"] > 1e-6:
        raise RuntimeError(f"the re-gauge changed the function: {exact}")

    variants: dict[str, dict[str, np.ndarray]] = {
        "copy_first": tie(teacher_w, cfg.block, loops, "first"),
        "copy_last": tie(teacher_w, cfg.block, loops, "last"),
        "mean_raw": tie(teacher_weights(w0, cfg.block, loops, heads, align=False), cfg.block, loops, "mean"),
        "mean_aligned": tie(teacher_w, cfg.block, loops, "mean"),
    }
    batches = _batches(source, calib, cfg.batch_size)
    variants["regmean"] = regmean(teacher, batches, cfg.block, loops, cfg.ridge, cfg.option_weight)
    log(json.dumps({"regmean_s": round(time.perf_counter() - started, 1)}))
    student = rebuild(source, variants["regmean"])
    for r in range(1, cfg.rounds + 1):
        refine(student, teacher, batches, cfg.block, loops, cfg.ridge, cfg.option_weight)
        variants[f"lsq{r}"] = weights(student)
        clear_mlx_cache()
        log(json.dumps({"lsq_round": r, "s": round(time.perf_counter() - started, 1)}))

    table = {}
    for name, w in variants.items():
        if not is_looped(w, cfg.block, loops):
            raise RuntimeError(f"{name} is not tied")
        table[name] = {**fidelity(ref_held, predict(rebuild(source, w), held)), "data_free": name in DATA_FREE}
        log(json.dumps({name: table[name]}))
    chosen = min(table, key=lambda k: table[k]["kl"])
    best_free = min(DATA_FREE, key=lambda k: table[k]["kl"])
    model = rebuild(source, variants[chosen])
    temperature = match_temperature([temper(p, base_t) for p in ref_held], predict(model, held))

    # The chosen block at other loop counts: 1 loop is the 4-layer model the tie would be without looping.
    depth = {}
    for n in sorted({loops, *cfg.extra_loops}):
        wn, tn = unroll(variants[chosen], types, cfg.block, n)
        mn = rebuild(source, wn, tn)
        depth[str(n)] = {"layers": len(tn), "holdout": fidelity(ref_held, predict(mn, held)),
                         "val": metrics(split["val"], predict(mn, split["val"]))}
        log(json.dumps({"loops": n, **depth[str(n)]}))

    report = {}
    for name in ("val", "test"):
        exs = split[name]
        if not exs:
            continue
        ref_raw, new_raw = predict(source, exs), predict(model, exs)
        ref_p, new_p = [temper(p, base_t) for p in ref_raw], [temper(p, temperature) for p in new_raw]
        report[name] = {
            "maps": len({e["task_id"] for e in exs}),
            "original": metrics(exs, ref_p), "looped": metrics(exs, new_p),
            "fidelity": fidelity(ref_raw, new_raw),
            "looped_minus_original": {m: _paired(exs, new_p, ref_p, m) for m in ("regret", "play_regret", "exp_regret")},
        }
        if best_free != chosen:
            free_raw = predict(rebuild(source, variants[best_free]), exs)
            report[name]["best_data_free"] = {"variant": best_free, "metrics": metrics(exs, free_raw),
                                              "fidelity": fidelity(ref_raw, free_raw)}
        log(json.dumps({name: report[name]}))

    saved = B.save(model, out_dir / "model", provenance={
        "edit": "rh17m.loopedit", "variant": chosen, "source_model": str(cfg.model),
        "loop": {"block_layers": list(range(1, 1 + cfg.block)), "loops": loops, "layer0_outside": True},
        "temperature": temperature})
    (saved / "rh17m.json").write_text(json.dumps({"temperature": temperature, "data": {}}))
    reloaded = weights(B.load(str(saved), revision=None))
    results = {
        "name": out_dir.name, "source_model": str(cfg.model), "chosen": chosen, "best_data_free": best_free,
        "temperature": temperature, "source_temperature": base_t, "regauge_exactness": exact,
        "variants_holdout": table, "loops": depth, **report,
        "parameters": unique_parameters(reloaded, cfg.block),
        "tied_after_reload": is_looped(reloaded, cfg.block, loops),
        "calib": len(calib), "holdout": len(held), "seconds": round(time.perf_counter() - started, 1),
    }
    if not results["tied_after_reload"]:
        raise RuntimeError("the saved export is not tied")
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results
