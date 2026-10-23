"""One training run: Bekko-17M (MLX) fine-tuned on simulator soft targets, selected and calibrated on val.

Augmentation per epoch: the option order is shuffled (the model must not learn positions) and the prefix
layout alternates between instruction-first and state-first (both are Bekko v0 layouts). Model selection
uses val expected regret; the temperature is then fitted on val NLL; test is evaluated once at the end. The
run directory holds ``config.json``, ``history.jsonl``, ``results.json`` and ``model/`` (published
parameter names, loadable by the PyTorch runtime too).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from rh17m import bekko_mlx as B
from rh17m.dataset import DatasetConfig, load_examples
from rh17m.evaluate import BASELINES, metrics, per_example

LAYOUTS = ("instruction_state", "state_instruction")


@dataclass
class TrainConfig:
    name: str = "run"
    data: dict = field(default_factory=dict)  # DatasetConfig overrides
    init: str = "bekko"  # bekko (17M) | bekko-68m | bekko-400m | <path to a saved model dir>
    epochs: int = 8
    batch_size: int = 16
    lr: float = 1e-4
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    shuffle_options: bool = True
    layouts: tuple[str, ...] = LAYOUTS
    seed: int = 0
    pad_multiple: int = 64
    compile: bool = True  # traced train step (shapes bucketed by pad_multiple)
    bf16: bool = True  # bf16 compute with fp32 master weights while training; evaluation runs in fp32
    track: bool = True  # stream metrics to Trackio (project "railhead-17m"; `trackio show` for the dashboard)
    artifact: bool = True  # log the model export as a Trackio artifact (the climb logs only kept models)
    n_traces: int = 20  # val decisions logged as traces
    group: str = ""  # Trackio run group (the climb sets one per climb)
    log_every: int = 5  # steps between train metric logs
    train_splits: tuple[str, ...] = ("train",)
    max_train: int = 0  # 0 = all
    distill_from: str = ""  # teacher (run model dir or release name) that soft-labels unlabelled train states
    distill_max: int = 0  # cap on teacher-labelled examples (0 = all available)
    distill_temperature: float = 1.0  # extra temperature on the teacher's probabilities
    freeze_embeddings: bool = False  # keep the token embeddings at their pretrained values
    freeze_layers: int = 0  # freeze the first N encoder layers (-1: the whole encoder, train the heads only)


def _encode(tok, ex: dict, layout: str, perm: list[int] | None = None) -> B.EncodedDecision:
    opts = ex["options"] if perm is None else [ex["options"][i] for i in perm]
    enc = B.encode_choice(tok, ex["state_json"], ex["instructions"], opts, layout=layout)
    return enc.with_target([ex["target"][k] for k, _ in opts])


def predict(model, examples: list[dict], temperature: float = 1.0, batch_size: int = 32,
            layout: str = LAYOUTS[0]) -> list[dict[str, float]]:
    tok = model.tokenizer
    out: list[dict[str, float]] = []
    for i in range(0, len(examples), batch_size):
        enc = [B.encode_choice(tok, e["state_json"], e["instructions"], e["options"], layout=layout)
               for e in examples[i:i + batch_size]]
        out.extend(B.decision_probs(model, enc, pad_multiple=64))
    return [temper(p, temperature) for p in out]


def temper(p: dict[str, float], t: float) -> dict[str, float]:
    if t == 1.0:
        return p
    w = {k: math.exp(math.log(max(v, 1e-30)) / t) for k, v in p.items()}
    z = sum(w.values())
    return {k: v / z for k, v in w.items()}


def fit_temperature(examples: list[dict], probs: list[dict[str, float]]) -> float:
    """Grid search of T on soft-target NLL (log-spaced 0.2-5)."""
    if not examples:
        return 1.0
    grid = [math.exp(x) for x in np.linspace(math.log(0.2), math.log(5.0), 41)]
    return min(grid, key=lambda t: metrics(examples, [temper(p, t) for p in probs])["nll"])


# Published Bekko v0 releases (decision_mlx.bekko_provider pins the same revisions).
RELEASES = {
    "bekko": ("hotchpotch/bekko-system-one-v0-17m", B.REVISION),
    "bekko-68m": ("hotchpotch/bekko-system-one-v0-68m", "ab7685f23e5edbc1acb12ced4f2c4e12591efa69"),
    "bekko-400m": ("hotchpotch/bekko-system-one-v0-400m", "1960df5602bd93cc8d336fdebd9fb68a30926e13"),
}


def _load(init: str):
    if init in RELEASES:
        repo, revision = RELEASES[init]
        return B.load(repo, revision=revision)
    return B.load(init, revision=None)


TRACK_PROJECT = "railhead-17m"


def _tracker(cfg: TrainConfig, extra: dict):
    """The MLX infra's Trackio tracker (decision_mlx.tracking, decision-mlx PRs #7-#10)."""
    try:
        from decision_mlx.tracking import Tracker
        if not hasattr(Tracker, "_call"):  # PRs #7-#11
            raise ImportError
    except ImportError:  # the local decision-mlx checkout predates them
        from rh17m._tracking import Tracker
    return Tracker(TRACK_PROJECT, name=cfg.name, group=cfg.group or None, config={**asdict(cfg), **extra},
                   enabled=cfg.track, system_interval=2.0)


def _freeze(model, cfg: TrainConfig) -> None:
    """Regularise small-data fine-tuning by keeping the lower encoder at its pretrained weights."""
    backbone = model.encoder.backbone
    if cfg.freeze_layers < 0:
        model.encoder.freeze()
        return
    if cfg.freeze_embeddings and hasattr(backbone, "embeddings"):
        backbone.embeddings.freeze()
    for layer in list(backbone.layers)[: cfg.freeze_layers]:
        layer.freeze()


def _lr(opt, index: int) -> float:
    try:
        return float(opt.optimizers[index].learning_rate)
    except Exception:  # noqa: BLE001
        return float("nan")


def run(cfg: TrainConfig, labels_dir: Path, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    dcfg = DatasetConfig(**cfg.data)
    exs = load_examples(Path(labels_dir), dcfg)
    splits = {s: [e for e in exs if e["split"] == s] for s in ("train", "val", "test", "ood")}
    train = [e for s in cfg.train_splits for e in splits[s]]
    rng = random.Random(cfg.seed)
    if cfg.max_train:
        train = rng.sample(train, min(cfg.max_train, len(train)))
    if not train:
        raise ValueError("no training examples")
    n_distill = 0
    if cfg.distill_from:
        from rh17m.dataset import load_observations

        teacher = _load(cfg.distill_from)
        meta = Path(cfg.distill_from) / "rh17m.json"
        t_temp = (json.loads(meta.read_text())["temperature"] if meta.exists() else 1.0) * cfg.distill_temperature
        unl = load_observations(Path(labels_dir).parent / "obs", dcfg, exclude={e["id"] for e in exs})
        if cfg.distill_max:
            unl = rng.sample(unl, min(cfg.distill_max, len(unl)))
        for e, p in zip(unl, predict(teacher, unl, t_temp), strict=True):
            e["target"] = p
        train = train + unl
        n_distill = len(unl)
        del teacher
    mx.random.seed(cfg.seed)
    model = _load(cfg.init)
    tok = model.tokenizer
    _freeze(model, cfg)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = max(1, steps_per_epoch * cfg.epochs)
    opt = B.make_optimizer(cfg.lr, cfg.head_lr, cfg.weight_decay, total_steps=total, warmup_ratio=cfg.warmup_frac)
    step = B.make_train_step(model, opt, compile=cfg.compile)
    train_dtype = mx.bfloat16 if cfg.bf16 else mx.float32
    val = splits["val"]
    best = {"epoch": -1, "exp_regret": float("inf")}
    best_params = None
    hist = (out_dir / "history.jsonl").open("w")
    tracker = _tracker(cfg, {"n_" + k: len(v) for k, v in splits.items()} | {"n_train_used": len(train),
                                                                              "n_distill": n_distill})
    gstep = 0
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.set_compute_dtype(train_dtype)
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for b in range(0, len(order), cfg.batch_size):
            enc = []
            for i in order[b:b + cfg.batch_size]:
                ex = train[i]
                perm = list(range(len(ex["options"])))
                if cfg.shuffle_options:
                    rng.shuffle(perm)
                enc.append(_encode(tok, ex, rng.choice(cfg.layouts), perm))
            batch = B.collate(enc, tasks=model.tasks, pad_multiple=cfg.pad_multiple)
            ts = time.perf_counter()
            losses.append(float(step(batch)))
            gstep += 1
            if gstep % cfg.log_every == 0:
                tracker.log({"train/loss": losses[-1], "train/lr_encoder": _lr(opt, 1), "train/lr_heads": _lr(opt, 0),
                             "train/examples_per_s": len(enc) / max(time.perf_counter() - ts, 1e-9),
                             "train/epoch": epoch + b / max(1, len(order))}, gstep)
        model.set_compute_dtype(mx.float32)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_s": round(time.perf_counter() - t0, 1)}
        if val:
            vp = predict(model, val)
            m = metrics(val, vp)
            tracker.log_histogram("val/regret_hist", per_example(val, vp, "regret"), step=gstep)
            row.update({f"val_{k}": round(v, 5) for k, v in m.items()})
            if m["exp_regret"] < best["exp_regret"]:
                best = {"epoch": epoch, "exp_regret": m["exp_regret"]}
                best_params = {k: mx.array(v) for k, v in B.state_dict(model).items()}
        hist.write(json.dumps(row) + "\n")
        hist.flush()
        tracker.log({"train/epoch_loss": row["loss"],
                     **{f"val/{k[4:]}": v for k, v in row.items() if k.startswith("val_")}}, gstep)
    hist.close()
    if best_params is not None:
        model.load_weights(list(best_params.items()))
    temperature = fit_temperature(val, predict(model, val)) if val else 1.0
    results = {"name": cfg.name, "examples": {k: len(v) for k, v in splits.items()}, "train_used": len(train),
               "distilled": n_distill,
               "best_epoch": best["epoch"], "temperature": temperature,
               "train_s": round(time.perf_counter() - t0, 1)}
    final_probs: dict[str, list[dict[str, float]]] = {}
    for s in ("val", "test", "ood"):
        part = splits[s]
        if part:
            probs = final_probs[s] = predict(model, part, temperature)
            results[s] = {k: round(v, 5) for k, v in metrics(part, probs).items()}
            results[f"{s}_raw"] = {k: round(v, 5) for k, v in metrics(part, predict(model, part)).items()}
            results[f"{s}_baselines"] = {n: {k: round(v, 5) for k, v in metrics(part, [f(e) for e in part]).items()}
                                         for n, f in BASELINES.items()}
    B.save(model, out_dir / "model", provenance={"run": cfg.name, "temperature": temperature,
                                                 "config": asdict(cfg)})
    (out_dir / "model" / "rh17m.json").write_text(json.dumps({"temperature": temperature, "data": cfg.data}))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    final = {"final/temperature": temperature, "final/best_epoch": best["epoch"]}
    for s in ("val", "test", "ood"):
        for k, v in results.get(s, {}).items():
            final[f"final/{s}_{k}"] = v
        for name in ("longest", "uniform", "hold"):
            for k in ("regret", "exp_regret", "capture"):
                v = results.get(f"{s}_baselines", {}).get(name, {}).get(k)
                if v is not None:
                    final[f"final/{s}_baseline_{name}_{k}"] = v
    tracker.log(final, gstep + 1)
    _log_rich(tracker, cfg, model, splits, final_probs, results, out_dir, temperature, gstep + 1)
    tracker.finish()
    return results


def _log_rich(tracker, cfg: TrainConfig, model, splits: dict, final_probs: dict, results: dict, out_dir: Path,
              temperature: float, step: int) -> None:
    """Every Trackio tab beyond the curves: decisions tables, regret histograms, decision traces, the report,
    an alert when the model beats the strongest scripted policy on val, files and the model artifact."""
    from rh17m.track_rich import DECISION_COLUMNS, decision_rows, decision_trace, report_markdown

    for s, probs in final_probs.items():
        tracker.log_table(f"{s}/decisions", DECISION_COLUMNS, decision_rows(splits[s], probs), step=step)
        tracker.log_histogram(f"{s}/final_regret_hist", per_example(splits[s], probs, "regret"), step=step)
    for e in splits["val"][: cfg.n_traces]:
        t0 = time.perf_counter()
        p = temper(B.choice_probs(model, None, e["state_json"], e["instructions"], e["options"]), temperature)
        tracker.log_trace(f"val/{e['id']}", **decision_trace(e, p, cfg.init, (time.perf_counter() - t0) * 1e3),
                          step=step)
    tracker.log_report("report", report_markdown(results, asdict(cfg)), step=step)
    val, base = results.get("val"), results.get("val_baselines", {}).get("longest")
    if val and base:
        if val["regret"] < base["regret"]:
            tracker.alert(f"{cfg.name} beats always-longest on val",
                          f"regret {val['regret']:.4f} vs {base['regret']:.4f} (n={int(val['n'])})", level="info")
        else:
            tracker.alert(f"{cfg.name} does not beat always-longest on val",
                          f"regret {val['regret']:.4f} vs {base['regret']:.4f} (n={int(val['n'])})", level="warn")
    tracker.save_files(*(str(out_dir / f) for f in ("config.json", "results.json", "history.jsonl")))
    if cfg.artifact:
        tracker.log_artifact(out_dir / "model", name="railhead-17m", type="model", aliases=[cfg.name])
