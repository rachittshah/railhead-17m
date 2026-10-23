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
import mlx.optimizers as optim
import numpy as np

from rh17m import bekko_mlx as B
from rh17m.dataset import DatasetConfig, load_examples
from rh17m.evaluate import BASELINES, metrics

LAYOUTS = ("instruction_state", "state_instruction")


@dataclass
class TrainConfig:
    name: str = "run"
    data: dict = field(default_factory=dict)  # DatasetConfig overrides
    init: str = "bekko"  # bekko (the pinned release) | <path to a saved model dir>
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
    train_splits: tuple[str, ...] = ("train",)
    max_train: int = 0  # 0 = all


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


def _load(init: str):
    return B.load() if init == "bekko" else B.load(init, revision=None)


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
    mx.random.seed(cfg.seed)
    model = _load(cfg.init)
    tok = model.tokenizer
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = max(1, steps_per_epoch * cfg.epochs)
    warm = max(1, int(cfg.warmup_frac * total))

    def sched(peak):
        return optim.join_schedules([optim.linear_schedule(peak * 0.01, peak, warm),
                                     optim.cosine_decay(peak, max(1, total - warm), peak * 0.01)], [warm])

    opt = B.make_optimizer(sched(cfg.lr), sched(cfg.head_lr), cfg.weight_decay)
    step = B.make_train_step(model, opt)
    val = splits["val"]
    best = {"epoch": -1, "exp_regret": float("inf")}
    best_params = None
    hist = (out_dir / "history.jsonl").open("w")
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
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
            losses.append(float(step(batch)))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "elapsed_s": round(time.perf_counter() - t0, 1)}
        if val:
            m = metrics(val, predict(model, val))
            row.update({f"val_{k}": round(v, 5) for k, v in m.items()})
            if m["exp_regret"] < best["exp_regret"]:
                best = {"epoch": epoch, "exp_regret": m["exp_regret"]}
                best_params = {k: mx.array(v) for k, v in B.state_dict(model).items()}
        hist.write(json.dumps(row) + "\n")
        hist.flush()
    hist.close()
    if best_params is not None:
        model.load_weights(list(best_params.items()))
    temperature = fit_temperature(val, predict(model, val)) if val else 1.0
    results = {"name": cfg.name, "examples": {k: len(v) for k, v in splits.items()}, "train_used": len(train),
               "best_epoch": best["epoch"], "temperature": temperature,
               "train_s": round(time.perf_counter() - t0, 1)}
    for s in ("val", "test", "ood"):
        part = splits[s]
        if part:
            probs = predict(model, part, temperature)
            results[s] = {k: round(v, 5) for k, v in metrics(part, probs).items()}
            results[f"{s}_raw"] = {k: round(v, 5) for k, v in metrics(part, predict(model, part)).items()}
            results[f"{s}_baselines"] = {n: {k: round(v, 5) for k, v in metrics(part, [f(e) for e in part]).items()}
                                         for n, f in BASELINES.items()}
    B.save(model, out_dir / "model", provenance={"run": cfg.name, "temperature": temperature,
                                                 "config": asdict(cfg)})
    (out_dir / "model" / "rh17m.json").write_text(json.dumps({"temperature": temperature, "data": cfg.data}))
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results
