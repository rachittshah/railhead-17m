"""Hillclimb: greedy coordinate ascent over training and data settings, judged on val with a noise floor.

The incumbent starts as the base config. Its seed noise is measured first: the base config trained with
``noise_seeds`` seeds gives the spread of val expected regret, and a move must beat the incumbent by more
than that spread AND have a paired bootstrap interval (per-example expected regret, same val examples)
entirely below zero. Each kept move becomes the incumbent; every run, kept or not, is a line in
``runs/climb/ledger.jsonl`` with its config, val metrics and verdict. Both models are always compared on the
current val set, so the comparison stays fair while labels keep arriving.

Moves are config overrides (``{"lr": 3e-4}``, ``{"data": {"tau": 0.1}}``). After a full pass without a kept
move, the step sizes shrink around the incumbent (a second pass of finer moves), then the climb stops.
"""

from __future__ import annotations

import copy
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from rh17m.dataset import DatasetConfig, load_examples
from rh17m.evaluate import bootstrap_diff, metrics, per_example
from rh17m.train import TrainConfig, predict, run

ROOT = Path(__file__).resolve().parents[2]
CLIMB_DIR = ROOT / "runs" / "climb"


def coarse_moves() -> list[dict]:
    return [
        {"data": {"tau": 0.02}}, {"data": {"tau": 0.1}}, {"data": {"tau": 0.2}},
        {"lr": 3e-5}, {"lr": 3e-4},
        {"head_lr": 1e-4}, {"head_lr": 2e-3},
        {"epochs": 4}, {"epochs": 16},
        {"batch_size": 8}, {"batch_size": 32},
        {"data": {"features": False}}, {"data": {"drop_candidate_table": False}}, {"data": {"ranks": True}}, {"data": {"plane": True}},
        {"shuffle_options": False}, {"layouts": ["instruction_state"]},
        {"weight_decay": 0.0}, {"weight_decay": 0.1},
        {"train_splits": ["train", "ood"]},
    ]


def fine_moves(inc: TrainConfig) -> list[dict]:
    tau = inc.data.get("tau", DatasetConfig.tau)
    return [{"data": {"tau": tau * 0.7}}, {"data": {"tau": tau * 1.4}}, {"lr": inc.lr * 0.6}, {"lr": inc.lr * 1.6},
            {"epochs": max(2, int(inc.epochs * 0.75))}, {"epochs": int(inc.epochs * 1.5)}]


def apply(cfg: TrainConfig, move: dict, name: str) -> TrainConfig:
    new = copy.deepcopy(cfg)
    new.name = name
    for k, v in move.items():
        if k == "data":
            new.data = {**new.data, **v}
        else:
            setattr(new, k, tuple(v) if isinstance(getattr(new, k), tuple) else v)
    return new


def _val_scores(model_dir: Path, val: list[dict]) -> tuple[list[float], dict]:
    from rh17m import bekko_mlx as B

    model = B.load(str(model_dir), revision=None)
    t = json.loads((model_dir / "rh17m.json").read_text())["temperature"]
    probs = predict(model, val, t)
    return per_example(val, probs, "exp_regret"), metrics(val, probs)


def _git_push(message: str, paths: list[Path]) -> None:
    rel = [str(p.relative_to(ROOT)) for p in paths if p.exists()]
    if not rel:
        return
    subprocess.run(["git", "add", *rel], cwd=ROOT, check=False)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, check=False)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=False)


def climb(base: TrainConfig, labels_dir: Path, noise_seeds: int = 3, max_runs: int = 60,
          push: bool = True, on_keep=None) -> TrainConfig:
    CLIMB_DIR.mkdir(parents=True, exist_ok=True)
    ledger = CLIMB_DIR / "ledger.jsonl"
    stamp = time.strftime("%m%d-%H%M")

    def log(row: dict) -> None:
        with ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def train(cfg: TrainConfig) -> Path:
        cfg.group = f"climb-{stamp}"
        out = ROOT / "runs" / "train" / cfg.name
        run(cfg, labels_dir, out)
        return out / "model"

    # Noise floor: the same config, different seeds.
    seeds_val = []
    inc_cfg, inc_dir = base, None
    for s in range(noise_seeds):
        cfg = apply(base, {"seed": s}, f"{stamp}-base-s{s}")
        d = train(cfg)
        val = [e for e in load_examples(labels_dir, DatasetConfig(**cfg.data)) if e["split"] == "val"]
        _, m = _val_scores(d, val)
        seeds_val.append(m["exp_regret"])
        log({"run": cfg.name, "kind": "noise", "config": asdict(cfg), "val": m})
        if inc_dir is None or m["exp_regret"] <= min(seeds_val):
            inc_cfg, inc_dir = cfg, d
    noise = max(seeds_val) - min(seeds_val)
    log({"kind": "noise_floor", "exp_regret_by_seed": seeds_val, "noise": noise})
    runs = noise_seeds
    for phase, moves in (("coarse", coarse_moves()), ("fine", None)):
        moves = moves if moves is not None else fine_moves(inc_cfg)
        for i, move in enumerate(moves):
            if runs >= max_runs:
                return inc_cfg
            cfg = apply(inc_cfg, move, f"{stamp}-{phase}{i:02d}")
            cand_dir = train(cfg)
            runs += 1
            val = [e for e in load_examples(labels_dir, DatasetConfig(**cfg.data)) if e["split"] == "val"]
            # Compare on the candidate's val examples; the incumbent is re-scored on exactly these.
            inc_val = [e for e in load_examples(labels_dir, DatasetConfig(**inc_cfg.data)) if e["split"] == "val"]
            ids = {e["id"] for e in val} & {e["id"] for e in inc_val}
            val = [e for e in val if e["id"] in ids]
            inc_val = sorted((e for e in inc_val if e["id"] in ids), key=lambda e: e["id"])
            val.sort(key=lambda e: e["id"])
            a, ma = _val_scores(cand_dir, val)
            b, mb = _val_scores(inc_dir, inc_val)
            diff, lo, hi = bootstrap_diff(a, b)
            keep = diff < -noise and hi < 0
            log({"run": cfg.name, "kind": phase, "move": move, "val": ma, "incumbent_val": mb,
                 "diff_exp_regret": diff, "ci": [lo, hi], "noise": noise, "kept": keep, "n_val": len(val)})
            if keep:
                inc_cfg, inc_dir = cfg, cand_dir
                if on_keep:
                    on_keep(cfg, cand_dir, ma)
                if push:
                    _git_push(f"Climb: keep {json.dumps(move)} (val exp_regret {ma['exp_regret']:.4f}, "
                              f"diff {diff:+.4f} [{lo:+.4f}, {hi:+.4f}])",
                              [ledger, ROOT / "runs" / "train" / cfg.name / "results.json",
                               ROOT / "runs" / "train" / cfg.name / "config.json"])
    if push:
        _git_push("Climb: ledger after a full pass", [ledger])
    return inc_cfg
