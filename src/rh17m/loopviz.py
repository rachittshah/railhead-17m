"""Figures for docs/loop-edit.md: charts of the loop-edit results, measurements of the edited model, the
architecture and attention masks drawn from the real config, and an animation of the representation
moving through the loop.

Every figure is written twice, ``<name>.svg`` (light) and ``<name>-dark.svg``, so the markdown can pick one
with ``<picture>``. Colours follow the dataviz reference palette: categorical slots in fixed order (validated
for three series in both modes), one-hue blue ramp for magnitudes, text in ink tokens, hairline grid.

    rh17m loop-figures --run runs/loop/1023-1627-base-s1-loop2 --out docs/figures/loop-edit
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rh17m import bekko_mlx as B
from rh17m import loopedit as L


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, ...]
    ramp: tuple[str, ...]  # sequential blue, surface-adjacent end first


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
              ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"),
              ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"))
DARK = Theme("dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
             ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"),
             ("#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"))
THEMES = (LIGHT, DARK)
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def style(t: Theme) -> None:
    plt.rcParams.update({
        "figure.facecolor": t.surface, "axes.facecolor": t.surface, "savefig.facecolor": t.surface,
        "text.color": t.ink, "axes.labelcolor": t.ink2, "axes.edgecolor": t.axis, "axes.titlecolor": t.ink,
        "xtick.color": t.muted, "ytick.color": t.muted, "xtick.labelcolor": t.ink2, "ytick.labelcolor": t.ink2,
        "grid.color": t.grid, "grid.linewidth": 0.6, "axes.grid": False, "font.family": "sans-serif",
        "font.sans-serif": FONT, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": t.ink2, "svg.fonttype": "none", "lines.linewidth": 2,
        "axes.prop_cycle": matplotlib.cycler(color=list(t.series)),
    })


def ramp(t: Theme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"blue-{t.name}", list(t.ramp))


def save(fig, out: Path, name: str, t: Theme) -> Path:
    path = out / (f"{name}.svg" if t is LIGHT else f"{name}-dark.svg")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    preview = os.environ.get("RH17M_FIG_PREVIEW")  # a directory: also write PNGs there, to look at the layout
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(preview) / f"{path.stem}.png", bbox_inches="tight", pad_inches=0.15, dpi=110)
    plt.close(fig)
    return path


def grid_y(ax) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def grid_x(ax) -> None:
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ result charts (results.json only)
VARIANT_LABEL = {"copy_first": "copy layers 1-3", "copy_last": "copy layers 4-6", "mean_raw": "mean, unaligned",
                 "mean_aligned": "mean, aligned", "regmean": "RegMean", "lsq1": "RegMean + 1 sweep",
                 "lsq2": "+ 2 sweeps", "lsq3": "+ 3 sweeps", "lsq4": "+ 4 sweeps"}


def fig_variants(r: dict, out: Path, t: Theme) -> None:
    table = r["variants_holdout"]
    names = list(table)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(names))[::-1]
    colours = [t.series[0] if table[n]["data_free"] else t.series[1] for n in names]
    for ax, metric, title, fmt in ((axes[0], "kl", "KL to the incumbent (lower is closer)", "{:.3f}"),
                                   (axes[1], "top1", "Top-1 agreement with the incumbent", "{:.2f}")):
        values = [table[n][metric] for n in names]
        ax.barh(y, values, height=0.62, color=colours, edgecolor=t.surface, linewidth=2)
        for yy, v, n in zip(y, values, names, strict=True):
            weight = "bold" if n == r["chosen"] else "normal"
            ax.text(v + max(values) * 0.015, yy, fmt.format(v), va="center", fontsize=9, color=t.ink2,
                    fontweight=weight)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(0, max(values) * 1.18 if metric == "kl" else 1.0)
        grid_x(ax)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [VARIANT_LABEL.get(n, n) + ("  (chosen)" if n == r["chosen"] else "") for n in names])
    handles = [plt.Rectangle((0, 0), 1, 1, color=t.series[0]), plt.Rectangle((0, 0), 1, 1, color=t.series[1])]
    fig.legend(handles, ["weights only (no data)", "uses unlabelled activations"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Every tie, scored on {r['holdout']} decisions from held-out training maps", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", color=t.ink)
    save(fig, out, "variants", t)


def fig_sweeps(r: dict, out: Path, t: Theme) -> None:
    table = r["variants_holdout"]
    stages = ["regmean"] + sorted([k for k in table if k.startswith("lsq") and k[3:].isdigit()],
                                  key=lambda k: int(k[3:]))
    xs = np.arange(len(stages))
    kl = [table[s]["kl"] for s in stages]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(table["copy_first"]["kl"], color=t.muted, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(len(stages) - 1, table["copy_first"]["kl"] + 0.012, "best weights-only tie (copy layers 1-3)",
            ha="right", fontsize=9, color=t.ink2)
    ax.plot(xs, kl, color=t.series[1], marker="o", markersize=7, markeredgecolor=t.surface, markeredgewidth=2)
    for x, v in zip(xs, kl, strict=True):
        ax.text(x, v + 0.022, f"{v:.3f}", ha="center", fontsize=9, color=t.ink2)
    ax.set_xticks(xs, ["RegMean"] + [f"sweep {i}" for i in range(1, len(stages))])
    ax.set_ylim(0, table["copy_first"]["kl"] * 1.12)
    ax.set_ylabel("KL to the incumbent")
    ax.set_title("Gauss-Seidel sweeps on the looped model's own activations")
    grid_y(ax)
    save(fig, out, "sweeps", t)


def fig_loops(r: dict, out: Path, t: Theme) -> None:
    depth = r["loops"]
    loops = sorted(int(k) for k in depth)
    kl = [depth[str(n)]["holdout"]["kl"] for n in loops]
    regret = [depth[str(n)]["val"]["regret"] for n in loops]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.3})
    for ax, values, title, label, fmt in (
            (axes[0], kl, "Closeness to the incumbent (held-out train maps)", "KL to the incumbent", "{:.3f}"),
            (axes[1], regret, "Val regret (398 decisions)", "regret (asinh)", "{:.3f}")):
        ax.plot(loops, values, color=t.series[0], marker="o", markersize=7, markeredgecolor=t.surface,
                markeredgewidth=2)
        for x, v in zip(loops, values, strict=True):
            ax.text(x, v, "  " + fmt.format(v), va="bottom", fontsize=9, color=t.ink2)
        ax.set_xticks(loops, [f"{n}\n({depth[str(n)]['layers']} layers)" for n in loops])
        ax.set_xlabel("times the block is applied")
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10.5)
        grid_y(ax)
    axes[0].set_ylim(0, max(kl) * 1.15)
    axes[1].axhline(r["val"]["original"]["regret"], color=t.muted, linewidth=1.2, linestyle=(0, (4, 3)))
    axes[1].text(loops[-1], r["val"]["original"]["regret"], "incumbent ", ha="right", va="top", fontsize=9,
                 color=t.ink2)
    lo, hi = min(regret + [r["val"]["original"]["regret"]]), max(regret)
    axes[1].set_ylim(lo - 0.02, hi + 0.015)
    fig.suptitle("The same block applied 1-4 times: the second pass does the work", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", color=t.ink, y=1.03)
    save(fig, out, "loops", t)


def fig_regret(r: dict, out: Path, t: Theme) -> None:
    metrics = ("regret", "play_regret", "exp_regret")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, split in zip(axes, ("val", "test"), strict=True):
        rows = r[split]["looped_minus_original"]
        y = np.arange(len(metrics))[::-1]
        for yy, m in zip(y, metrics, strict=True):
            d = rows[m]
            ax.plot([d["lo"], d["hi"]], [yy, yy], color=t.series[0], linewidth=2, solid_capstyle="round")
            ax.plot(d["diff"], yy, "o", color=t.series[0], markersize=8, markeredgecolor=t.surface,
                    markeredgewidth=2)
            ax.text(d["hi"] + 0.006, yy, f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]", va="center",
                    fontsize=9, color=t.ink2)
        ax.axvline(0, color=t.ink2, linewidth=1)
        ax.set_title(f"{split}: {r[split]['original']['n']} decisions, {r[split]['maps']} maps", fontsize=10.5)
        ax.set_xlim(-0.07, 0.15)
        ax.set_xlabel("looped minus incumbent (positive = looped is worse)")
        grid_x(ax)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(metrics))[::-1], ["regret", "play_regret", "exp_regret"])
    fig.suptitle("Decision quality against simulator labels, 95% map-bootstrap intervals", x=0.02, ha="left",
                 fontsize=12, fontweight="bold", color=t.ink, y=1.05)
    save(fig, out, "regret", t)


def fig_parameters(r: dict, out: Path, t: Theme) -> None:
    """Where the parameters are, before and after the tie (unique values)."""
    p = r["parameters"]
    emb, layers, unique_layers = 12_894_464, p["encoder_layers"], p["unique_encoder_layers"]
    rest = p["all"] - emb - layers
    rows = [("original", [emb, layers, 0, rest]), ("looped (unique)", [emb, unique_layers, 0, rest])]
    labels = ["token embeddings", "encoder layers", "", "heads + final norm"]
    fig, ax = plt.subplots(figsize=(9, 2.4))
    for i, (name, parts) in enumerate(rows):
        left = 0
        for j, v in enumerate(parts):
            if not v:
                continue
            ax.barh(i, v, left=left, height=0.55, color=(t.series[0], t.series[1], None, t.series[2])[j],
                    edgecolor=t.surface, linewidth=2)
            if v > 1e6:
                ax.text(left + v / 2, i, f"{v / 1e6:.2f}M", ha="center", va="center", color="#ffffff", fontsize=9)
            left += v
        ax.text(left + 1.5e5, i, f"{left / 1e6:.2f}M", va="center", fontsize=9, color=t.ink2)
    ax.set_yticks([0, 1], [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 18.5e6)
    ax.set_xticks(np.arange(0, 18.1e6, 3e6), [f"{v:.0f}M" for v in range(0, 19, 3)])
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (t.series[0], t.series[1])]
    ax.legend(handles, labels[:2], loc="lower right", ncol=2, bbox_to_anchor=(1, -0.6))
    ax.set_title("Parameters: the tie removes 1.67M of the 3.90M layer weights; embeddings dominate")
    save(fig, out, "parameters", t)


# ------------------------------------------------------------------ measurements on the models
def _split_qkv(w: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return w[:dim], w[dim:2 * dim], w[2 * dim:]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def weight_similarity(w0: dict, heads: int) -> tuple[np.ndarray, np.ndarray, list[str], list, list]:
    """Cosine similarity of each sub-matrix of layer 1+j against layer 4+j: norms folded, then re-gauged."""
    folded = L.fold_norms(w0, range(1, 7))
    aligned = dict(folded)
    heads_info, mlp_info = [], []
    for j in range(3):
        heads_info.append(L.align_attention(aligned, 1 + j, 4 + j, heads))
        mlp_info.append(L.align_mlp(aligned, 1 + j, 4 + j))
    dim = w0[L.key(1, "attn.Wo")].shape[0]
    names = ["Wq", "Wk", "Wv", "attn Wo", "Wi (gelu)", "Wi (gate)", "mlp Wo"]

    def parts(w, layer):
        q, k, v = _split_qkv(w[L.key(layer, "attn.Wqkv")], dim)
        wi = w[L.key(layer, "mlp.Wi")]
        n = wi.shape[0] // 2
        return [q, k, v, w[L.key(layer, "attn.Wo")], wi[:n], wi[n:], w[L.key(layer, "mlp.Wo")]]

    raw = np.array([[_cos(a, b) for a, b in zip(parts(folded, 1 + j), parts(folded, 4 + j), strict=True)]
                    for j in range(3)])
    after = np.array([[_cos(a, b) for a, b in zip(parts(aligned, 1 + j), parts(aligned, 4 + j), strict=True)]
                      for j in range(3)])
    return raw, after, names, heads_info, mlp_info


def fig_weight_similarity(raw, after, names, out: Path, t: Theme) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1), gridspec_kw={"wspace": 0.05})
    vmax = max(raw.max(), after.max())
    for ax, m, title in ((axes[0], raw, "as trained (norms folded)"), (axes[1], after, "after the exact re-gauge")):
        im = ax.imshow(m, cmap=ramp(t), vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.55 * vmax else t.ink)
        ax.set_xticks(range(len(names)), names, rotation=0, fontsize=8.5)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(3), ["layer 1 vs 4", "layer 2 vs 5", "layer 3 vs 6"])
    axes[1].set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("cosine similarity", color=t.ink2)
    cb.outline.set_visible(False)
    fig.suptitle("How alike the two blocks' weights are, before and after alignment (same function both times)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=t.ink, y=1.04)
    save(fig, out, "weight-similarity", t)


def fig_head_match(heads_info, out: Path, t: Theme) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"wspace": 0.25})
    vmax = max(h["similarity"].max() for h in heads_info)
    for j, (ax, h) in enumerate(zip(axes, heads_info, strict=True)):
        sim = h["similarity"]
        ax.imshow(sim, cmap=ramp(t), vmin=0, vmax=vmax)
        for (a, b), v in np.ndenumerate(sim):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if v > 0.55 * vmax else t.ink)
        for a, b in enumerate(h["match"]):
            ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, edgecolor=t.series[1], linewidth=2.5))
        ax.set_xticks(range(sim.shape[1]), [f"h{b}" for b in range(sim.shape[1])])
        ax.set_yticks(range(sim.shape[0]), [f"h{a}" for a in range(sim.shape[0])])
        ax.set_xlabel(f"heads of layer {4 + j}")
        ax.set_ylabel(f"heads of layer {1 + j}")
        ax.set_title(f"layer {1 + j} vs {4 + j}", fontsize=10.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Head matching: similarity after the best value rotation and RoPE phases; boxes = Hungarian match",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=t.ink, y=1.06)
    save(fig, out, "head-match", t)


def fig_neuron_match(mlp_info, out: Path, t: Theme) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), gridspec_kw={"wspace": 0.12, "hspace": 0.3})
    vmax = max(np.percentile(m["cost"], 99.9) for m in mlp_info)
    for j, m in enumerate(mlp_info):
        for row, (mat, title) in enumerate(((m["cost"], "as trained"), (m["cost"][:, m["order"]], "re-ordered"))):
            ax = axes[row, j]
            ax.imshow(mat, cmap=ramp(t), vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"layer {1 + j} vs {4 + j}, {title}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        flipped = int((m["sign"] < 0).sum())
        axes[1, j].set_xlabel(f"{flipped} of {len(m['sign'])} gate signs flipped", color=t.ink2)
    fig.suptitle("GeGLU neurons (384 x 384): matching cost before and after the permutation; "
                 "a diagonal means unit i of one layer has a partner unit i in the other",
                 x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=t.ink, y=0.98)
    save(fig, out, "neuron-match", t)


def _pooled(model, batch, step_out: dict) -> np.ndarray:
    """Per layer: final-normed stream, mean over each option's branch tokens -> [layers, B, K, H]."""
    final = model.encoder.backbone.final_norm
    seg = batch.doc_seg
    k = batch.option_mask.shape[1]
    onehot = (seg[:, None, :] == mx.arange(k)[None, :, None]).astype(mx.float32)
    counts = mx.maximum(onehot.sum(axis=-1, keepdims=True), 1.0)
    rows = []
    for layer in sorted(step_out):
        h = final(step_out[layer]).astype(mx.float32)
        rows.append((onehot @ h) / counts)
    stack = mx.stack(rows)
    mx.eval(stack)
    return np.array(stack)


def stream_error(teacher, students: dict, batches) -> dict[str, list[float]]:
    """Relative error of each student's residual stream (option tokens) against the teacher's, per layer."""
    layers = len(teacher.encoder.backbone.layers)
    want = {(p, "out") for p in range(layers)}
    num = {n: np.zeros(layers) for n in students}
    den = np.zeros(layers)
    for batch in batches:
        ref = L.walk(teacher, batch, want)[1]
        valid = (batch.doc_seg >= 0).astype(mx.float32)[..., None]
        for p in range(layers):
            den[p] += float(((ref[p, "out"][1] ** 2) * valid).sum())
        for name, model in students.items():
            got = L.walk(model, batch, want)[1]
            for p in range(layers):
                num[name][p] += float((((got[p, "out"][1] - ref[p, "out"][1]) ** 2) * valid).sum())
    return {n: list(np.sqrt(num[n] / den)) for n in students}


def fig_stream(errors: dict, out: Path, t: Theme) -> None:
    labels = {"copy_first": "copy layers 1-3 (weights only)", "regmean": "RegMean", "lsq4": "RegMean + 4 sweeps (chosen)"}
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    xs = np.arange(len(next(iter(errors.values()))))
    for i, (name, ys) in enumerate(errors.items()):
        ax.plot(xs, ys, color=t.series[i], marker="o", markersize=6, markeredgecolor=t.surface, markeredgewidth=1.5)
        ax.text(xs[-1] + 0.12, ys[-1], labels.get(name, name), va="center", fontsize=9, color=t.ink2)
    ax.axvspan(0.5, 3.5, color=t.grid, alpha=0.5, linewidth=0)
    ax.axvspan(3.5, 6.5, color=t.grid, alpha=0.25, linewidth=0)
    top = max(max(v) for v in errors.values())
    ax.text(2, top * 1.08, "loop 1 (layers 1-3)", ha="center", fontsize=9, color=t.ink2)
    ax.text(5, top * 1.08, "loop 2 (stored as 4-6)", ha="center", fontsize=9, color=t.ink2)
    ax.set_xticks(xs, [f"after\nlayer {p}" for p in xs])
    ax.set_xlim(-0.3, xs[-1] + 2.6)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("relative error of the stream")
    ax.set_title("Where the looped model drifts from the original: option-token residual stream, layer by layer")
    grid_y(ax)
    save(fig, out, "stream-error", t)


def fixed_point(model, src_probs, held, batches, max_loops: int) -> dict[str, list[float]]:
    """Apply the block 1..max_loops times: change of the pooled option states per extra loop, and KL."""
    from rh17m.train import predict

    w = L.weights(model)
    change, kl, top1 = [], [], []
    prev = None
    for n in range(1, max_loops + 1):
        wn, tn = L.unroll(w, model.config.layer_types, 3, n)
        mn = L.rebuild(model, wn, tn)
        fid = L.fidelity(src_probs, predict(mn, held))
        kl.append(fid["kl"])
        top1.append(fid["top1"])
        last = len(tn) - 1
        pooled = np.concatenate([_pooled(mn, b, {last: L.walk(mn, b, {(last, "out")})[1][last, "out"][1]})[0]
                                 .reshape(-1, model.config.hidden_size) for b in batches])
        if prev is not None:
            change.append(float(np.linalg.norm(pooled - prev) / np.linalg.norm(prev)))
        prev = pooled
    return {"kl": kl, "top1": top1, "change": change}


def fig_fixed_point(fp: dict, out: Path, t: Theme) -> None:
    n = len(fp["kl"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"wspace": 0.3})
    xs = np.arange(1, n + 1)
    axes[0].plot(xs, fp["kl"], color=t.series[0], marker="o", markersize=6, markeredgecolor=t.surface,
                 markeredgewidth=1.5)
    axes[0].axvline(2, color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
    axes[0].text(2.1, max(fp["kl"]) * 0.92, "trained / fitted\nloop count", fontsize=9, color=t.ink2)
    axes[0].set_ylim(0, max(fp["kl"]) * 1.1)
    axes[0].set_title("KL to the incumbent", fontsize=10.5)
    axes[1].plot(xs[1:], fp["change"], color=t.series[0], marker="o", markersize=6, markeredgecolor=t.surface,
                 markeredgewidth=1.5)
    axes[1].set_ylim(0, max(fp["change"]) * 1.15)
    axes[1].set_title("change of the option representation per extra loop", fontsize=10.5)
    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xlabel("times the block is applied")
        grid_y(ax)
    fig.suptitle(f"Running the block past its fitted count (1-{n} loops): it settles rather than diverging",
                 x=0.02, ha="left", fontsize=12, fontweight="bold", color=t.ink, y=1.04)
    save(fig, out, "fixed-point", t)


def fig_logits(ref: list[dict], new: list[dict], out: Path, t: Theme) -> None:
    """Centred log-probabilities, incumbent vs looped, every option of every val decision."""
    xs, ys, best_same = [], [], []
    for a, b in zip(ref, new, strict=True):
        keys = list(a)
        la = np.log(np.maximum([a[k] for k in keys], 1e-12))
        lb = np.log(np.maximum([b[k] for k in keys], 1e-12))
        xs.extend(la - la.mean())
        ys.extend(lb - lb.mean())
        best_same.append(max(a, key=a.get) == max(b, key=b.get))
    xs, ys = np.array(xs), np.array(ys)
    r = np.corrcoef(xs, ys)[0, 1]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = max(np.abs(xs).max(), np.abs(ys).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=t.muted, linewidth=1, linestyle=(0, (4, 3)))
    ax.scatter(xs, ys, s=10, color=t.series[0], alpha=0.35, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("incumbent: centred log-probability")
    ax.set_ylabel("looped: centred log-probability")
    ax.set_title(f"Every option of {len(ref)} val decisions\nPearson r = {r:.2f}, same top pick in "
                 f"{np.mean(best_same):.0%}", fontsize=10.5)
    grid_y(ax)
    grid_x(ax)
    save(fig, out, "logits", t)


# ------------------------------------------------------------------ model visualizations
def layer_parameters(w: dict, layer: int) -> int:
    marker = f".layers.{layer}."
    return sum(v.size for k, v in w.items() if marker in k)


def fig_architecture(model, w: dict, r: dict, out: Path, t: Theme) -> None:
    """Both models drawn from the real config: layer types, widths, parameter counts, the tie."""
    cfg = model.config
    types = cfg.layer_types
    kind_colour = {"full_attention": t.series[0], "sliding_attention": t.series[1]}
    kind_name = {"full_attention": "full attention", "sliding_attention": f"sliding ±{cfg.window}"}
    fig, ax = plt.subplots(figsize=(12.6, 7.4))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    def box(x, y, wdt, h, text, colour, sub="", dashed=False, text_colour=None, alpha=1.0):
        ax.add_patch(FancyBboxPatch((x, y), wdt, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=colour, edgecolor=t.surface if not dashed else t.muted,
                                    linewidth=1.5, linestyle="--" if dashed else "-", alpha=alpha))
        ax.text(x + 0.18, y + h / 2 + (0.11 if sub else 0), text, va="center", fontsize=10,
                color=text_colour or "#ffffff", fontweight="bold")
        if sub:
            ax.text(x + 0.18, y + h / 2 - 0.17, sub, va="center", fontsize=8.5, color=text_colour or "#ffffff")

    def arrow(x0, y0, x1, y1, **kw):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                     color=kw.pop("color", t.ink2), linewidth=1.3, **kw))

    neutral = t.ramp[4]
    emb = w["encoder.backbone.embeddings.tok_embeddings.weight"].size
    for col, (x0, title) in enumerate(((0.5, "Original incumbent: 7 distinct layers"),
                                       (6.6, "Looped: layer 0 + one block applied twice"))):
        ax.text(x0, 9.3, title, fontsize=12.5, fontweight="bold", color=t.ink)
        y = 0.4
        box(x0, y, 4.9, 0.62, "token embeddings + LayerNorm", neutral,
            f"vocab {cfg.vocab_size:,} x {cfg.hidden_size} = {emb / 1e6:.2f}M parameters")
        y += 0.62
        arrow(x0 + 2.45, y + 0.02, x0 + 2.45, y + 0.28)
        y += 0.3
        rows = range(len(types))
        for p in rows:
            tied = col == 1 and p >= 4
            colour = kind_colour[types[p]]
            name = f"layer {p}  ·  {kind_name[types[p]]}"
            sub = f"{layer_parameters(w, p):,} params"
            if p == 0:
                sub += "  ·  no attention norm: stays outside the loop"
            elif col == 1:
                sub = ("same weights as layer " + str(p - 3)) if tied else f"block slot {p}  ·  " + sub
            box(x0, y, 4.9, 0.66, name if not tied else f"loop 2 · slot {p - 3}  ·  {kind_name[types[p]]}",
                colour, sub, dashed=tied, alpha=0.55 if tied else 1.0)
            y += 0.66
            if p < len(types) - 1:
                arrow(x0 + 2.45, y + 0.02, x0 + 2.45, y + 0.2)
                y += 0.22
        arrow(x0 + 2.45, y + 0.02, x0 + 2.45, y + 0.28)
        y += 0.3
        box(x0, y, 4.9, 0.62, "final norm -> mean-pool each option -> choice head", neutral,
            "Linear(256, 1) per option, softmax over options, temperature "
            + (f"{r['source_temperature']:.2f}" if col == 0 else f"{r['temperature']:.2f}"))
        if col == 1:
            # bracket the block and draw the loop arrow
            y1 = 0.4 + 0.62 + 0.3 + 0.88
            y_top = y1 + 3 * 0.88 - 0.22
            ax.add_patch(FancyBboxPatch((x0 - 0.18, y1 - 0.1), 5.26, y_top - y1 + 0.2,
                                        boxstyle="round,pad=0.02,rounding_size=0.15", fill=False,
                                        edgecolor=t.series[2], linewidth=2.2))
            ax.text(x0 + 5.25, y1 + (y_top - y1) / 2 + 0.25, "THE BLOCK\n(one set of\nweights)", fontsize=9.5,
                    color=t.ink, fontweight="bold", va="center")
            ax.add_patch(FancyArrowPatch((x0 + 5.2, y_top + 0.05), (x0 + 5.2, y1 - 0.05),
                                         connectionstyle="arc3,rad=-0.9", arrowstyle="-|>", mutation_scale=16,
                                         color=t.series[2], linewidth=2.2))
            ax.text(x0 + 5.75, y1 - 0.05, "x2", fontsize=13, color=t.series[2], fontweight="bold")
    ax.text(0.5, 8.92, f"width {cfg.hidden_size} · {cfg.num_attention_heads} heads of {cfg.head_dim} · "
            f"GeGLU {cfg.intermediate_size} · RoPE θ 160k · no biases · pre-norm", fontsize=9.5, color=t.ink2)
    p = r["parameters"]
    ax.text(6.6, 8.92, f"encoder layers: {p['encoder_layers'] / 1e6:.2f}M -> {p['unique_encoder_layers'] / 1e6:.2f}M "
            "unique · export stays 7 layers (block written twice)", fontsize=9.5, color=t.ink2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (t.series[0], t.series[1], t.series[2])]
    ax.legend(handles, ["full attention (every token)", f"sliding attention (±{cfg.window} positions)",
                        "the tied block"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    save(fig, out, "architecture", t)


def fig_masks(model, example: dict, out: Path, t: Theme) -> None:
    """The real attention layouts for one val decision: prefix (instruction + state) and option branches."""
    from rh17m.train import LAYOUTS, _encoded

    enc = _encoded(model.tokenizer, example, LAYOUTS[0])
    batch = B.collate([enc], tasks=model.tasks)
    prefix_len = int(batch.prefix_len[0])
    layouts = model._masks(batch.prefix_len, batch.prefix_ids.shape[1], batch.doc_seg, batch.doc_pos)
    seg = np.array(batch.doc_seg[0])
    tokens = int((seg >= 0).sum())
    size = prefix_len + tokens
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"wspace": 0.12})
    cmap = LinearSegmentedColormap.from_list("mask", [t.surface, t.series[0]])
    for ax, kind in zip(axes, ("full_attention", "sliding_attention"), strict=True):
        pm, om = layouts[kind]
        pm, om = np.array(pm)[0, 0], np.array(om)[0, 0]
        grid = np.zeros((size, size))
        grid[:prefix_len, :prefix_len] = np.broadcast_to(pm, (pm.shape[0] if pm.shape[0] > 1 else prefix_len,
                                                              pm.shape[-1]))[:prefix_len, :prefix_len]
        grid[prefix_len:, :prefix_len] = om[:tokens, :prefix_len]
        grid[prefix_len:, prefix_len:] = om[:tokens, om.shape[-1] - om.shape[0]:][:, :tokens]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(prefix_len - 0.5, color=t.ink2, linewidth=0.8)
        ax.axvline(prefix_len - 0.5, color=t.ink2, linewidth=0.8)
        starts = [prefix_len + int(np.argmax(seg == k)) for k in range(int(seg.max()) + 1)]
        for s0 in starts[1:]:
            ax.axhline(s0 - 0.5, color=t.muted, linewidth=0.5)
            ax.axvline(s0 - 0.5, color=t.muted, linewidth=0.5)
        ax.set_title({"full_attention": "full-attention layers (0, 3, 6)",
                      "sliding_attention": f"sliding layers (1, 2, 4, 5): ±{model.config.window} positions"}[kind],
                     fontsize=10.5)
        ax.set_xlabel("key position (prefix | option branches)")
        ax.set_xticks([prefix_len / 2, prefix_len + tokens / 2], [f"prefix ({prefix_len})", f"options ({tokens})"])
        ax.set_yticks([prefix_len / 2, prefix_len + tokens / 2], ["prefix\nqueries", "option\nqueries"])
        ax.tick_params(length=0)
    fig.suptitle(f"Who reads whom, for one real val decision with {len(example['options'])} options: prefix tokens "
                 "read the prefix; each option reads the prefix and its own branch, never another option",
                 x=0.02, ha="left", fontsize=11.5, fontweight="bold", color=t.ink, y=1.0)
    save(fig, out, "attention-masks", t)


# ------------------------------------------------------------------ animation
def trajectory(model, batch) -> np.ndarray:
    """Pooled option states after the embeddings and after every layer: [layers + 1, K, H] for one decision."""
    layers = len(model.encoder.backbone.layers)
    taps = L.walk(model, batch, {(p, "x") for p in range(layers)} | {(layers - 1, "out")})[1]
    outs = {p: taps[p, "x"][1] for p in range(layers)}
    outs[layers] = taps[layers - 1, "out"][1]
    return _pooled(model, batch, outs)[:, 0]


def gif_trajectory(src, looped, example: dict, out: Path, t: Theme) -> Path:
    """Watch one decision's options move through the layers, original vs looped, in a shared 2-D projection."""
    from rh17m.train import LAYOUTS, _encoded

    enc = _encoded(src.tokenizer, example, LAYOUTS[0])
    batch = B.collate([enc], tasks=src.tasks)
    keys = list(enc.keys)
    a, b = trajectory(src, batch)[:, :len(keys)], trajectory(looped, batch)[:, :len(keys)]
    flat = np.concatenate([a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])])
    centre = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - centre, full_matrices=False)
    pa, pb = (a - centre) @ vt[:2].T, (b - centre) @ vt[:2].T
    q = np.array([example["q"][k] for k in keys])
    best = int(np.argmax(q))
    style(t)
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.55, 1, 1], wspace=0.18)
    strip, axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    lim_x = (min(pa[..., 0].min(), pb[..., 0].min()), max(pa[..., 0].max(), pb[..., 0].max()))
    lim_y = (min(pa[..., 1].min(), pb[..., 1].min()), max(pa[..., 1].max(), pb[..., 1].max()))
    pad_x, pad_y = (lim_x[1] - lim_x[0]) * 0.08, (lim_y[1] - lim_y[0]) * 0.08
    colours = ramp(t)(0.25 + 0.75 * (q - q.min()) / max(q.max() - q.min(), 1e-9))
    types = src.config.layer_types
    stages = ["embeddings"] + [f"layer {p}" for p in range(len(types))]

    def draw(frame):
        for ax in (strip, axa, axb):
            ax.clear()
        strip.set_xlim(0, 1)
        strip.set_ylim(-0.5, len(stages) + 0.2)
        strip.axis("off")
        strip.text(0.05, len(stages) - 0.15, "stage (loop = looped model)", fontsize=9, color=t.ink2)
        for i, name in enumerate(stages):
            active = i == frame
            if i == 0:
                colour = t.ramp[4]
            else:
                colour = t.series[0] if types[i - 1] == "full_attention" else t.series[1]
            loop = "" if i <= 1 else (" · loop 1" if i <= 4 else " · loop 2")
            strip.add_patch(FancyBboxPatch((0.05, i - 0.38), 0.9, 0.76, boxstyle="round,pad=0.01,rounding_size=0.08",
                                           facecolor=colour, alpha=1.0 if active else 0.25, edgecolor="none"))
            strip.text(0.12, i, name + loop, va="center", fontsize=9, color="#ffffff" if active else t.ink2,
                       fontweight="bold" if active else "normal")
        for ax, pts, title in ((axa, pa, "original (7 distinct layers)"), (axb, pb, "looped (block x2)")):
            for k in range(len(keys)):
                ax.plot(pts[:frame + 1, k, 0], pts[:frame + 1, k, 1], color=colours[k], linewidth=1, alpha=0.5)
            ax.scatter(pts[frame, :, 0], pts[frame, :, 1], s=70, c=colours, edgecolors=t.surface, linewidths=1.5,
                       zorder=3)
            ax.scatter([pts[frame, best, 0]], [pts[frame, best, 1]], s=190, facecolors="none",
                       edgecolors=t.ink, linewidths=1.6, zorder=4)
            ax.set_xlim(lim_x[0] - pad_x, lim_x[1] + pad_x)
            ax.set_ylim(lim_y[0] - pad_y, lim_y[1] + pad_y)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10.5)
            for s in ax.spines.values():
                s.set_color(t.axis)
        fig.suptitle(f"One val decision, {len(keys)} options, through the encoder ({stages[frame]}). "
                     "Darker = higher simulator value; ring = best option", x=0.02, ha="left", fontsize=11,
                     fontweight="bold", color=t.ink)

    frames = list(range(len(stages))) + [len(stages) - 1] * 4
    anim = animation.FuncAnimation(fig, draw, frames=frames, interval=900)
    path = out / ("trajectory.gif" if t is LIGHT else "trajectory-dark.gif")
    anim.save(path, writer=animation.PillowWriter(fps=1.2), dpi=90, savefig_kwargs={"facecolor": t.surface})
    plt.close(fig)
    return path


# ------------------------------------------------------------------ entry point
def run(run_dir: Path, out: Path, labels_dir: Path, log=print, fixed_loops: int = 8) -> dict:
    from rh17m.dataset import load_examples
    from rh17m.train import limit_mlx_cache, predict, temper

    limit_mlx_cache()
    run_dir, out = Path(run_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads((run_dir / "results.json").read_text())
    cfg = L.LoopConfig(**{**json.loads((run_dir / "config.json").read_text()),
                          "extra_loops": tuple(json.loads((run_dir / "config.json").read_text())["extra_loops"])})
    src = B.load(cfg.model, revision=None)
    looped = B.load(str(run_dir / "model"), revision=None)
    heads = src.config.num_attention_heads
    split, held, calib = L.selection(cfg, load_examples(Path(labels_dir)))
    log(json.dumps({"held": len(held), "calib": len(calib), "val": len(split["val"])}))

    # measurements that need the models
    w0 = L.weights(src)
    raw, after, names, heads_info, mlp_info = weight_similarity(w0, heads)
    teacher = L.rebuild(src, L.teacher_weights(w0, 3, 2, heads))
    calib_batches = L._batches(src, calib, cfg.batch_size)
    students = {"copy_first": L.rebuild(src, L.tie(L.teacher_weights(w0, 3, 2, heads), 3, 2, "first")),
                "regmean": L.rebuild(src, L.regmean(teacher, calib_batches, 3, 2, cfg.ridge, cfg.option_weight)),
                "lsq4": looped}
    held_batches = L._batches(src, held, cfg.batch_size)
    errors = stream_error(src, students, held_batches)
    log(json.dumps({"stream_error_last": {k: round(v[-1], 4) for k, v in errors.items()}}))
    ref_held = predict(src, held)
    fp = fixed_point(looped, ref_held, held, held_batches, fixed_loops)
    log(json.dumps({"fixed_point": fp}))
    val = split["val"]
    ref_val = [temper(p, r["source_temperature"]) for p in predict(src, val)]
    new_val = [temper(p, r["temperature"]) for p in predict(looped, val)]
    # A short val decision with several options shows the masks and the trajectory clearly.
    candidates = [e for e in val if 5 <= len(e["options"]) <= 8]
    example = min(candidates, key=lambda e: len(e["state_json"]))
    measured = {"weight_similarity": {"names": names, "raw": raw.tolist(), "aligned": after.tolist()},
                "head_match": [h["match"].tolist() for h in heads_info],
                "gate_flips": [int((m["sign"] < 0).sum()) for m in mlp_info],
                "stream_error": errors, "fixed_point": fp, "example": example["id"]}
    (out / "measurements.json").write_text(json.dumps(measured, indent=1))

    written = []
    for t in THEMES:
        style(t)
        fig_architecture(src, w0, r, out, t)
        fig_masks(src, example, out, t)
        fig_parameters(r, out, t)
        fig_variants(r, out, t)
        fig_sweeps(r, out, t)
        fig_loops(r, out, t)
        fig_regret(r, out, t)
        fig_weight_similarity(raw, after, names, out, t)
        fig_head_match(heads_info, out, t)
        fig_neuron_match(mlp_info, out, t)
        fig_stream(errors, out, t)
        fig_fixed_point(fp, out, t)
        fig_logits(ref_val, new_val, out, t)
        written.append(str(gif_trajectory(src, looped, example, out, t)))
        log(json.dumps({"theme": t.name, "done": True}))
    written += sorted(str(p) for p in out.glob("*.svg"))
    return {"out": str(out), "files": written, "measurements": measured}
