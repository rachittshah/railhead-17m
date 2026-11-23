"""Data for the interactive explainer (docs/explainer): recorded activations of real val decisions.

Everything the page shows comes from here, computed by the real models; nothing in the page runs a model.

* ``attention_walk`` is ``loopedit.walk`` with the attention softmax written out, so the weights can be saved.
  Its logits are checked against ``model(batch)`` and its merged head outputs against the fused kernel.
* Per stage (embeddings, then after every layer) it records the option states, the stream norm of every
  token, the size of each attention and MLP update, and a probe: the final norm and the choice head applied
  to that stage's pooled option states. The probe is a reading of the stream, not something the model
  computes before its last layer, and the page labels it that way.
* Attention maps are PNGs, one per (model, layer), four heads side by side. Pixel value = round(255 * sqrt(w)),
  so w = (v / 255) ** 2. Rows are the prefix queries, then the option-token queries; columns are the prefix
  keys, then the option-token keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

from rh17m import bekko_mlx as B
from rh17m import loopedit as L
from rh17m.bekko_mlx.model import NEG, _cast

LOOP_COUNTS = (1, 2, 3, 4, 6, 8)


# ------------------------------------------------------------------ forward pass with explicit attention
def _softmax_attention(q, k, v, mask, scale):
    scores = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)) * scale
    scores = mx.where(mask, scores, -mx.inf)
    weights = mx.softmax(scores, axis=-1)
    return weights, weights @ v.astype(mx.float32)


def attention_walk(model, batch, with_weights: bool = True) -> dict:
    """One forward pass of a single-decision batch, recording what the explainer shows.

    Returns logits [K], per layer the attention weights (prefix [H,P,P] or None in the last layer, options
    [H,T,P+T]), the fused-kernel check, and per stage the option stream [T,H], prefix stream [P,H] (None
    once the prefix stops updating) and the attention/MLP update norms per token.
    """
    prefix_ids, prefix_len, doc_ids, doc_seg, doc_pos, option_mask, task_index = batch.arrays()
    if prefix_ids.shape[0] != 1:
        raise ValueError("attention_walk takes one decision at a time")
    backbone, dtype = model.encoder.backbone, model.compute_dtype
    width = prefix_ids.shape[1]
    layouts = model._masks(prefix_len, width, doc_seg, doc_pos)
    prefix_positions = mx.broadcast_to(mx.arange(width)[None], prefix_ids.shape)
    option_positions = doc_pos + prefix_len[:, None]
    rotary = {kind: (model._rope(kind, prefix_positions), model._rope(kind, option_positions)) for kind in layouts}
    prefix, hidden = backbone.embeddings(prefix_ids, dtype), backbone.embeddings(doc_ids, dtype)
    last = len(backbone.layers) - 1
    out = {"layers": [], "stages": [{"prefix": prefix[0], "options": hidden[0]}], "fused_error": 0.0}
    for step, layer in enumerate(backbone.layers):
        attention, live = layer.attn, step < last
        prefix_mask, option_mask_ = layouts[layer.kind]
        (pcos, psin), (ocos, osin) = rotary[layer.kind]
        pq, pk, pv = L._split(attention.Wqkv(layer.normed(prefix)), attention, pcos, psin)
        q, k, v = L._split(attention.Wqkv(layer.normed(hidden)), attention, ocos, osin)
        keys, values = mx.concatenate([pk, k], axis=2), mx.concatenate([pv, v], axis=2)
        o_w, o_att = _softmax_attention(q, keys, values, option_mask_, attention.scale)
        fused = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attention.scale, mask=option_mask_)
        out["fused_error"] = max(out["fused_error"], float(mx.abs(fused - o_att).max()))
        p_w = None
        if live:
            p_w, p_att = _softmax_attention(pq, pk, pv, prefix_mask, attention.scale)
            p_upd = attention.Wo(L._merge(_cast(p_att, dtype)))
            prefix = prefix + p_upd
        o_upd = attention.Wo(L._merge(_cast(o_att, dtype)))
        hidden = hidden + o_upd
        a, gate = mx.split(layer.mlp.Wi(layer.mlp_norm(hidden)), 2, axis=-1)
        neurons = nn.gelu(a) * gate  # the GeGLU units, [1, T, intermediate]
        o_mlp = layer.mlp.Wo(neurons)
        hidden = hidden + o_mlp
        if live:
            p_mlp = layer.mlp(layer.mlp_norm(prefix))
            prefix = prefix + p_mlp
        out["layers"].append({
            "kind": layer.kind,
            "prefix_weights": p_w[0] if (live and with_weights) else None,
            "option_weights": o_w[0] if with_weights else None,
            "neurons": neurons[0],
            "attn_update": (mx.linalg.norm(p_upd[0], axis=-1) if live else None, mx.linalg.norm(o_upd[0], axis=-1)),
            "mlp_update": (mx.linalg.norm(p_mlp[0], axis=-1) if live else None, mx.linalg.norm(o_mlp[0], axis=-1)),
        })
        out["stages"].append({"prefix": prefix[0] if live else None, "options": hidden[0]})
    count = option_mask.shape[1]
    onehot = (doc_seg[0][None, :] == mx.arange(count)[:, None]).astype(mx.float32)  # [K, T]
    counts = mx.maximum(onehot.sum(axis=-1, keepdims=True), 1.0)
    head = model.heads[model.tasks[int(task_index[0])]]
    weight, bias = head.weight.astype(mx.float32), head.bias.astype(mx.float32)
    final = backbone.final_norm
    pooled, probe = [], []
    for stage in out["stages"]:
        p = (onehot @ _cast(final(stage["options"]), mx.float32)) / counts
        pooled.append(p)
        probe.append((p @ weight.T + bias)[:, 0])
    out["pooled"] = mx.stack(pooled)  # [stages, K, H]
    out["probe"] = mx.stack(probe)  # [stages, K]
    out["logits"] = mx.where(option_mask[0], out["probe"][-1], NEG)
    mx.eval(out)
    return out


# ------------------------------------------------------------------ encoding helpers
def tokens_of(model, ids) -> list[str]:
    """Readable piece for every id; special tokens keep their names."""
    tok = model.tokenizer.tokenizer
    special = {model.tokenizer.cls_id: "[CLS]", model.tokenizer.sep_id: "[SEP]"}
    return [special.get(int(i)) or tok.decode([int(i)]) or tok.id_to_token(int(i)) or "?" for i in ids]


def prefix_sections(pieces: list[str]) -> list[dict]:
    """Instruction and state spans of the prefix, found from its own text (layout instruction_state)."""
    text, starts = "", []
    for piece in pieces:
        starts.append(len(text))
        text += piece
    at = text.find("\nState:")
    if at < 0:
        return [{"name": "prefix", "start": 0, "end": len(pieces)}]
    split = next(i for i, s in enumerate(starts) if s >= at)
    return [{"name": "instruction", "start": 0, "end": split}, {"name": "state", "start": split, "end": len(pieces)}]


def _attention_png(weights_prefix, weights_options, prefix_len: int, path: Path) -> list[int]:
    """[(P+T) x 4(P+T)] grayscale PNG of one layer: v = round(255 sqrt(w)). Returns [height, width]."""
    from PIL import Image

    p_w = None if weights_prefix is None else np.array(weights_prefix)
    o_w = np.array(weights_options)
    heads, tokens, _ = o_w.shape
    size = prefix_len + tokens
    tiles = []
    for h in range(heads):
        grid = np.zeros((size, size), dtype=np.float32)
        if p_w is not None:
            grid[:prefix_len, :prefix_len] = p_w[h, :prefix_len, :prefix_len]
        grid[prefix_len:, :prefix_len] = o_w[h, :, :prefix_len]
        grid[prefix_len:, prefix_len:] = o_w[h, :, o_w.shape[-1] - tokens:]
        tiles.append(grid)
    image = np.round(255.0 * np.sqrt(np.clip(np.concatenate(tiles, axis=1), 0.0, 1.0))).astype(np.uint8)
    Image.fromarray(image, mode="L").save(path, optimize=True)
    return [int(image.shape[0]), int(image.shape[1])]


def _r(x, digits=3):
    return np.round(np.asarray(x, dtype=np.float64), digits).tolist()


def _softmax(x, t=1.0):
    z = np.asarray(x, dtype=np.float64) / t
    z = np.exp(z - z.max())
    return z / z.sum()


# ------------------------------------------------------------------ one decision
def export_decision(models: dict, example: dict, out: Path, temperatures: dict, top_neurons: int = 12) -> dict:
    """Write ``<id>/attn-<model>-<layer>.png`` and return the decision's JSON record.

    ``models``: {"original": model, "looped": model, "loops": {n: model}} with the looped export at 2 loops.
    """
    from rh17m.train import LAYOUTS, _encoded

    src = models["original"]
    enc = _encoded(src.tokenizer, example, LAYOUTS[0])
    batch = B.collate([enc], tasks=src.tasks)
    keys = list(enc.keys)
    k = len(keys)
    prefix_len = len(enc.prefix_ids)
    pieces = tokens_of(src, enc.prefix_ids)
    option_pieces = [tokens_of(src, ids) for ids in enc.option_ids]
    folder = out / example["id"]
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "id": example["id"], "task": example["task_id"], "keys": keys,
        "options": [dict(zip(("key", "text"), o, strict=True)) for o in example["options"]],
        "value": [float(example["q"][key]) for key in keys], "best": example["best"],
        "prefix": {"tokens": pieces, "sections": prefix_sections(pieces)},
        "branches": option_pieces, "models": {},
    }
    walks = {}
    for name in ("original", "looped"):
        model = models[name]
        walk = attention_walk(model, batch)
        reference = np.array(model(batch))[0, :k]
        got = np.array(walk["logits"])[:k]
        if not np.allclose(got, reference, atol=1e-3) or walk["fused_error"] > 1e-3:
            raise AssertionError(f"{name}: explicit attention disagrees with the model "
                                 f"(logits {np.abs(got - reference).max():.2e}, heads {walk['fused_error']:.2e})")
        walks[name] = walk
    # One 2-D projection for both models and every loop count, so positions are comparable.
    loop_walks = {n: attention_walk(m, batch, with_weights=False) for n, m in models["loops"].items()}
    stacks = [np.array(w["pooled"])[:, :k] for w in (*walks.values(), *loop_walks.values())]
    flat = np.concatenate([s.reshape(-1, s.shape[-1]) for s in stacks])
    centre = flat.mean(axis=0)
    basis = np.linalg.svd(flat - centre, full_matrices=False)[2][:2]

    def project(pooled):
        return _r((np.array(pooled)[:, :k] - centre) @ basis.T, 3)

    for name, walk in walks.items():
        layers = []
        for index, layer in enumerate(walk["layers"]):
            shape = _attention_png(layer["prefix_weights"], layer["option_weights"], prefix_len,
                                   folder / f"attn-{name}-{index}.png")
            o_w = np.array(layer["option_weights"])
            # Where each option's tokens look: share of attention on the instruction, the state and their own branch.
            seg = np.array(batch.doc_seg[0])
            sections = record["prefix"]["sections"]
            share = []
            for opt in range(k):
                rows = o_w[:, seg == opt, :]  # [H, n, P+T]
                parts = [rows[:, :, s["start"]:s["end"]].sum(-1).mean(-1) for s in sections]
                parts.append(rows[:, :, prefix_len:].sum(-1).mean(-1))
                share.append(_r(np.stack(parts, axis=-1), 3))  # [H, sections + 1]
            h = np.array(layer["neurons"])  # [T, I]
            per_option = np.stack([h[seg == opt].mean(0) for opt in range(k)])  # [K, I]
            top = np.argsort(-np.abs(per_option).mean(0))[:top_neurons]
            p_attn, o_attn = layer["attn_update"]
            p_mlp, o_mlp = layer["mlp_update"]
            layers.append({
                "kind": layer["kind"], "png": f"{example['id']}/attn-{name}-{index}.png", "png_shape": shape,
                "share": share,
                "entropy": _r([float((-np.where(o_w[hh] > 0, o_w[hh] * np.log(np.maximum(o_w[hh], 1e-30)), 0))
                                      .sum(-1).mean()) for hh in range(o_w.shape[0])], 3),
                "neurons": {"index": top.tolist(), "per_option": _r(per_option[:, top], 3)},
                "attn_update": {"prefix": None if p_attn is None else _r(p_attn, 2), "options": _r(o_attn, 2)},
                "mlp_update": {"prefix": None if p_mlp is None else _r(p_mlp, 2), "options": _r(o_mlp, 2)},
            })
        stages = [{"prefix_norm": None if s["prefix"] is None else _r(mx.linalg.norm(s["prefix"], axis=-1), 2),
                   "options_norm": _r(mx.linalg.norm(s["options"], axis=-1), 2)} for s in walk["stages"]]
        probe = np.array(walk["probe"])[:, :k]
        record["models"][name] = {
            "layers": layers, "stages": stages, "probe": _r(probe, 4), "xy": project(walk["pooled"]),
            "temperature": temperatures[name],
            "probs": _r(_softmax(probe[-1], temperatures[name]), 4),
        }
    record["loops"] = {}
    for n, walk in loop_walks.items():
        probe = np.array(walk["probe"])[:, :k]
        record["loops"][str(n)] = {"probe": _r(probe, 4), "xy": project(walk["pooled"]),
                                   "probs": _r(_softmax(probe[-1], temperatures["looped"]), 4)}
    record["sizes"] = {"prefix": prefix_len, "options": int(sum(map(len, enc.option_ids))),
                       "branch": [len(ids) for ids in enc.option_ids]}
    return record


def pick_decisions(val: list[dict], probs: dict, count: int = 3) -> list[dict]:
    """Short val decisions with 8-10 options: one both models get right, one where they disagree, one hold."""
    pool = sorted((e for e in val if 8 <= len(e["options"]) <= 10), key=lambda e: len(e["state_json"]))

    def top(name, e):
        p = probs[name][e["id"]]
        return max(p, key=p.get)

    rules = [lambda e: top("original", e) == top("looped", e) == e["best"],
             lambda e: top("original", e) != top("looped", e),
             lambda e: e["best"] == "hold"]
    chosen = []
    for rule in rules:
        for e in pool:
            if e not in chosen and rule(e):
                chosen.append(e)
                break
    for e in pool:
        if len(chosen) >= count:
            break
        if e not in chosen:
            chosen.append(e)
    return chosen[:count]


# ------------------------------------------------------------------ the model and the edit
def _matrix_png(m: np.ndarray, path: Path, vmax: float) -> None:
    from PIL import Image

    Image.fromarray(np.round(255 * np.clip(m / vmax, 0, 1)).astype(np.uint8), mode="L").save(path, optimize=True)


def export_meta(src, looped, results: dict, measurements: dict, out: Path) -> dict:
    from rh17m import loopviz as V

    w0 = L.weights(src)
    heads = src.config.num_attention_heads
    raw, after, names, heads_info, mlp_info = V.weight_similarity(w0, heads)
    vmax = float(max(np.percentile(m["cost"], 99.9) for m in mlp_info))
    for j, m in enumerate(mlp_info):
        _matrix_png(m["cost"], out / f"neurons-{j}-raw.png", vmax)
        _matrix_png(m["cost"][:, m["order"]], out / f"neurons-{j}-aligned.png", vmax)
    config = src.config
    layers = [{"index": i, "kind": kind, "parameters": V.layer_parameters(w0, i)}
              for i, kind in enumerate(config.layer_types)]
    embed = int(w0["encoder.backbone.embeddings.tok_embeddings.weight"].size)
    return {
        "config": {"hidden": config.hidden_size, "heads": heads, "head_dim": config.head_dim,
                   "intermediate": config.intermediate_size, "window": config.window,
                   "vocab": config.vocab_size, "layer_types": list(config.layer_types), "layers": layers,
                   "embedding_parameters": embed},
        "parameters": results["parameters"],
        "results": {s: results[s] for s in ("val", "test")},
        "variants": results["variants_holdout"], "chosen": results["chosen"], "loops": results["loops"],
        "temperature": {"original": results["source_temperature"], "looped": results["temperature"]},
        "weight_similarity": {"names": names, "raw": _r(raw, 3), "aligned": _r(after, 3)},
        "heads": [{"similarity": _r(h["similarity"], 3), "match": h["match"].tolist()} for h in heads_info],
        "neurons": [{"raw": f"neurons-{j}-raw.png", "aligned": f"neurons-{j}-aligned.png",
                     "flipped": int((m["sign"] < 0).sum()), "size": len(m["sign"])}
                    for j, m in enumerate(mlp_info)],
        "stream_error": measurements.get("stream_error"), "fixed_point": measurements.get("fixed_point"),
    }


def run(run_dir: Path, out: Path, labels_dir: Path, figures: Path, count: int = 3, log=print) -> dict:
    from rh17m.dataset import load_examples
    from rh17m.train import limit_mlx_cache, predict

    limit_mlx_cache()
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((run_dir / "results.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    cfg = L.LoopConfig(**{**config, "extra_loops": tuple(config["extra_loops"])})
    src = B.load(cfg.model, revision=None)
    looped = B.load(str(run_dir / "model"), revision=None)
    split = L.selection(cfg, load_examples(Path(labels_dir)))[0]
    val = split["val"]
    probs = {"original": dict(zip((e["id"] for e in val), predict(src, val), strict=True)),
             "looped": dict(zip((e["id"] for e in val), predict(looped, val), strict=True))}
    tied = L.weights(looped)
    types = src.config.layer_types
    loops = {n: (looped if n == 2 else L.rebuild(src, *L.unroll(tied, types, cfg.block, n))) for n in LOOP_COUNTS}
    temperatures = {"original": results["source_temperature"], "looped": results["temperature"]}
    measurements = json.loads((Path(figures) / "measurements.json").read_text())
    meta = export_meta(src, looped, results, measurements, out)
    decisions = []
    for example in pick_decisions(val, probs, count):
        record = export_decision({"original": src, "looped": looped, "loops": loops}, example, out, temperatures)
        (out / f"{example['id']}.json").write_text(json.dumps(record, separators=(",", ":")))
        decisions.append({"id": example["id"], "options": len(record["keys"]), "best": record["best"],
                          "tokens": record["sizes"]["prefix"] + record["sizes"]["options"]})
        log(json.dumps(decisions[-1]))
    meta["decisions"] = decisions
    meta["loop_counts"] = list(LOOP_COUNTS)
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return {"out": str(out), "decisions": decisions}
