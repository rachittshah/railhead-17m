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
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from rh17m.dataset import DatasetConfig, load_examples
from rh17m.evaluate import bootstrap_diff, metrics, per_example
from rh17m.train import TrainConfig, clear_mlx_cache, limit_mlx_cache, predict, run

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
        {"data": {"scale": 1_000_000.0}}, {"data": {"scale": 20_000.0}},
        {"data": {"min_spread": 0.05}}, {"data": {"min_spread": 0.1}},
        # Regularisation for small data: keep the lower encoder pretrained.
        {"freeze_embeddings": True}, {"freeze_embeddings": True, "freeze_layers": 3}, {"freeze_layers": -1},
        {"select_metric": "play_regret"}, {"select_metric": "nll"},
        # Distillation over unlabelled train-map states: from the incumbent itself, and from the latest larger
        # teacher (runs/teachers/current, e.g. Bekko-68M trained on the same labels) when one exists.
        {"distill_from": "@incumbent", "distill_max": 1000},
        {"distill_from": "@teacher", "distill_max": 1000},
        {"distill_from": "@teacher", "distill_max": 4000},
    ]


def fine_moves(inc: TrainConfig) -> list[dict]:
    tau = inc.data.get("tau", DatasetConfig.tau)
    return [{"data": {"tau": tau * 0.7}}, {"data": {"tau": tau * 1.4}}, {"lr": inc.lr * 0.6}, {"lr": inc.lr * 1.6},
            {"epochs": max(2, int(inc.epochs * 0.75))}, {"epochs": int(inc.epochs * 1.5)}]


TEACHER = ROOT / "runs" / "teachers" / "current"


def resolve_teacher(ref: str, incumbent_dir: Path | None) -> str | None:
    """``@incumbent`` -> the incumbent's model dir, ``@teacher`` -> runs/teachers/current (a symlink to a
    trained model dir); anything else is returned unchanged. None when the reference cannot be resolved."""
    if ref == "@incumbent":
        return str(incumbent_dir) if incumbent_dir else None
    if ref == "@teacher":
        return str(TEACHER.resolve()) if TEACHER.exists() else None
    return ref


def apply(cfg: TrainConfig, move: dict, name: str, incumbent_dir: Path | None = None) -> TrainConfig | None:
    new = copy.deepcopy(cfg)
    new.name = name
    if "distill_from" in move:
        teacher = resolve_teacher(move["distill_from"], incumbent_dir)
        if teacher is None:
            return None
        move = {**move, "distill_from": teacher}
    for k, v in move.items():
        if k == "data":
            new.data = {**new.data, **v}
        else:
            setattr(new, k, tuple(v) if isinstance(getattr(new, k), tuple) else v)
    return new


def _run_retrying(cfg: TrainConfig, labels_dir: Path, out: Path, attempts: int = 3) -> dict:
    """``train.run``, retried when Metal drops a command buffer (another process's GPU fault can take this
    one down with it); any other error propagates."""
    for attempt in range(attempts):
        try:
            return run(cfg, labels_dir, out)
        except RuntimeError as e:
            if "METAL" not in str(e) or attempt == attempts - 1:
                raise
            print(f"climb: {cfg.name} hit a Metal error ({e}); retry {attempt + 1}")
            time.sleep(30)
    raise RuntimeError("unreachable")


METRIC = "exp_regret"  # the climb's keep metric; `rh17m climb --metric play_regret` judges closed-loop-like


_PRED_CACHE: dict[tuple[str, str], dict[str, float]] = {}


def _key(e: dict) -> str:
    import hashlib

    return e["id"] + ":" + hashlib.sha1((e["state_json"] + json.dumps(e["options"])).encode()).hexdigest()[:12]


def _val_scores(model_dir: Path, val: list[dict]) -> tuple[list[float], dict]:
    """Per-example ``METRIC`` and aggregate metrics of a saved model on ``val``. Predictions are cached per
    (model, example text): the incumbent is re-scored on every move, but only new val examples cost a pass."""
    from rh17m import bekko_mlx as B

    todo = [e for e in val if (str(model_dir), _key(e)) not in _PRED_CACHE]
    if todo:
        model = B.load(str(model_dir), revision=None)
        t = json.loads((model_dir / "rh17m.json").read_text())["temperature"]
        for e, p in zip(todo, predict(model, todo, t), strict=True):
            _PRED_CACHE[(str(model_dir), _key(e))] = p
        del model
        clear_mlx_cache()
    probs = [_PRED_CACHE[(str(model_dir), _key(e))] for e in val]
    return per_example(val, probs, METRIC), metrics(val, probs)


def _git_push(message: str, paths: list[Path]) -> None:
    rel = [str(p.relative_to(ROOT)) for p in paths if p.exists()]
    if not rel:
        return
    for _ in range(10):  # other stages commit too: retry while the index is locked
        add = subprocess.run(["git", "add", *rel], cwd=ROOT, check=False, capture_output=True, text=True)
        if "index.lock" not in add.stderr:
            break
        time.sleep(3)
    for _ in range(10):
        commit = subprocess.run(["git", "commit", "-q", "-m", message], cwd=ROOT, check=False, capture_output=True,
                                text=True)
        if "index.lock" not in commit.stderr:
            break
        time.sleep(3)
    subprocess.run(["git", "push", "-q"], cwd=ROOT, check=False)


INCUMBENT = CLIMB_DIR / "incumbent.json"


def load_incumbent() -> TrainConfig | None:
    if not INCUMBENT.exists():
        return None
    d = json.loads(INCUMBENT.read_text())["config"]
    cfg = TrainConfig()
    for k, v in d.items():
        setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k, None), tuple) else v)
    return cfg


def save_incumbent(cfg: TrainConfig, model_dir: Path, val: dict, seed_dirs: list[Path] | None = None) -> None:
    """``model_dir`` is the deployed model (the incumbent config's best seed); ``seed_dirs`` all its seeds."""
    INCUMBENT.write_text(json.dumps({"config": asdict(cfg), "model_dir": str(model_dir), "val": val,
                                     "seed_dirs": [str(d) for d in seed_dirs or [model_dir]],
                                     "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def climb(base: TrainConfig, labels_dir: Path, seeds: tuple[int, ...] = (0, 1), max_runs: int = 60,
          push: bool = True, on_keep=None) -> TrainConfig:
    """One climb pass. Every config (the incumbent and each candidate) is trained once per seed in ``seeds``;
    a config's score on a val example is its mean expected regret over those seed models, so candidate and
    incumbent are compared like for like (a best-of-seeds incumbent would be a winner's curse). A move is
    kept when the paired bootstrap interval of the per-example difference is entirely below zero and the
    mean difference exceeds the seed noise of a seed-averaged score (estimated from the incumbent's seeds)."""
    CLIMB_DIR.mkdir(parents=True, exist_ok=True)
    ledger = CLIMB_DIR / "ledger.jsonl"
    stamp = time.strftime("%m%d-%H%M")
    limit_mlx_cache()

    def log(row: dict) -> None:
        with ledger.open("a") as f:
            f.write(json.dumps(row) + "\n")

    try:
        from decision_mlx.tracking import Tracker
        if not hasattr(Tracker, "_call"):  # PRs #7-#11
            raise ImportError
    except ImportError:  # the local decision-mlx checkout predates those PRs
        from rh17m._tracking import Tracker
    tracker = Tracker("railhead-17m-climb", name=f"climb-{stamp}", config={"base": asdict(base),
                      "seeds": list(seeds), "max_runs": max_runs})
    moves_rows: list[list] = []

    def train_seeds(cfg: TrainConfig, name: str) -> list[Path]:
        dirs = []
        for sd in seeds:
            c = copy.deepcopy(cfg)
            c.seed, c.name, c.group, c.artifact = sd, f"{name}-s{sd}", f"climb-{stamp}", False
            _run_retrying(c, labels_dir, ROOT / "runs" / "train" / c.name)
            dirs.append(ROOT / "runs" / "train" / c.name / "model")
        return dirs

    def val_for(cfg: TrainConfig) -> list[dict]:
        return sorted((e for e in load_examples(labels_dir, DatasetConfig(**cfg.data)) if e["split"] == "val"),
                      key=lambda e: e["id"])

    def score(dirs: list[Path], val: list[dict]) -> tuple[list[float], dict, list[float]]:
        per_seed, mets = [], []
        for d in dirs:
            a, m = _val_scores(d, val)
            per_seed.append(a)
            mets.append(m)
        mean = [sum(col) / len(col) for col in zip(*per_seed, strict=True)]
        avg = {k: sum(m[k] for m in mets) / len(mets) for k in mets[0]}
        return mean, avg, [m[METRIC] for m in mets]

    def best_seed(dirs: list[Path], val: list[dict]) -> Path:
        return min(dirs, key=lambda d: _val_scores(d, val)[1][METRIC])

    inc_cfg = copy.deepcopy(base)
    inc_dirs = train_seeds(inc_cfg, f"{stamp}-base")
    val0 = val_for(inc_cfg)
    _, inc_m, by_seed = score(inc_dirs, val0)
    # Std of a difference of two k-seed means ~ seed std * sqrt(2/k); the seed std from the range of k seeds.
    seed_sd = (max(by_seed) - min(by_seed)) / 1.128 if len(by_seed) > 1 else 0.0
    noise = seed_sd * math.sqrt(2 / len(seeds))
    log({"kind": "noise_floor", "run": f"{stamp}-base", "metric": METRIC, "by_seed": by_seed, "noise": noise,
         "val": inc_m, "seeds": list(seeds)})
    tracker.log({"climb/noise_floor": noise, f"climb/base_{METRIC}": inc_m[METRIC]}, step=0)
    tracker.alert(f"climb-{stamp}: base {METRIC} {inc_m[METRIC]:.4f}, noise {noise:.4f}",
                  f"val {METRIC} by seed {[round(v, 4) for v in by_seed]}", level="info")
    deploy = best_seed(inc_dirs, val0)
    save_incumbent(inc_cfg, deploy, inc_m, inc_dirs)
    runs = len(seeds)
    for phase, moves in (("coarse", coarse_moves()), ("fine", None)):
        moves = moves if moves is not None else fine_moves(inc_cfg)
        for i, move in enumerate(moves):
            if runs >= max_runs:
                break
            name = f"{stamp}-{phase}{i:02d}"
            cfg = apply(inc_cfg, move, name, deploy)
            if cfg is None:  # e.g. no teacher trained yet
                log({"run": name, "kind": phase, "move": move, "skipped": "unresolved"})
                continue
            cand_dirs = train_seeds(cfg, name)
            runs += len(seeds)
            # Both configs are scored on the val examples they share (min_spread may drop some).
            cand_val, inc_val = val_for(cfg), val_for(inc_cfg)
            ids = {e["id"] for e in cand_val} & {e["id"] for e in inc_val}
            cand_val = [e for e in cand_val if e["id"] in ids]
            inc_val = [e for e in inc_val if e["id"] in ids]
            a, ma, _ = score(cand_dirs, cand_val)
            b, mb, _ = score(inc_dirs, inc_val)
            diff, lo, hi = bootstrap_diff(a, b)
            keep = diff < -noise and hi < 0
            log({"run": name, "kind": phase, "move": move, "metric": METRIC, "val": ma, "incumbent_val": mb,
                 "diff": diff, "diff_exp_regret": diff,
                 "ci": [lo, hi], "noise": noise, "kept": keep, "n_val": len(ids), "seeds": list(seeds)})
            tracker.log({"climb/candidate_metric": ma[METRIC], "climb/incumbent_metric": mb[METRIC],
                         "climb/candidate_exp_regret": ma["exp_regret"], "climb/candidate_play_regret": ma["play_regret"],
                         "climb/diff": diff, "climb/ci_low": lo, "climb/ci_high": hi, "climb/kept": float(keep),
                         "climb/candidate_regret": ma["regret"], "climb/n_val": len(ids)}, step=runs)
            moves_rows.append([name, phase, json.dumps(move), round(ma[METRIC], 4), round(mb[METRIC], 4),
                               round(diff, 4), round(lo, 4), round(hi, 4), keep, len(ids)])
            if keep:
                inc_cfg, inc_dirs = cfg, cand_dirs
                deploy = best_seed(inc_dirs, cand_val)
                save_incumbent(inc_cfg, deploy, ma, inc_dirs)
                tracker.alert(f"kept {json.dumps(move)}", f"val {METRIC} {ma[METRIC]:.4f}, diff {diff:+.4f} "
                              f"[{lo:+.4f}, {hi:+.4f}] beyond noise {noise:.4f}", level="info")
                tracker.log_artifact(deploy, name="railhead-17m", type="model", aliases=["incumbent", name])
                if on_keep:
                    on_keep(cfg, deploy, ma)
                if push:
                    _git_push(f"Climb: keep {json.dumps(move)} (val {METRIC} {ma[METRIC]:.4f}, "
                              f"diff {diff:+.4f} [{lo:+.4f}, {hi:+.4f}])",
                              [ledger, INCUMBENT, deploy.parent / "results.json", deploy.parent / "config.json"])
    _climb_report(tracker, stamp, moves_rows, inc_cfg, noise, runs)
    tracker.save_files(str(ledger), str(INCUMBENT))
    tracker.finish()
    if push:
        _git_push("Climb: ledger after a full pass", [ledger, INCUMBENT])
    return inc_cfg


MOVE_COLUMNS = ["run", "phase", "move", "candidate", "incumbent", "diff", "ci_low", "ci_high", "kept", "n_val"]


def _climb_report(tracker, stamp: str, rows: list[list], inc_cfg: TrainConfig, noise: float, runs: int) -> None:
    tracker.log_table("climb/moves", MOVE_COLUMNS, rows, step=runs)
    kept = [r for r in rows if r[8]]
    intro = (f"{len(rows)} moves tried, {len(kept)} kept; noise floor {noise:.4f} (val expected regret across "
             "seeds). A move is kept when it beats the incumbent by more than the noise floor and its paired "
             "bootstrap interval is entirely below zero.")
    lines = [f"# Climb {stamp}", "", intro, "",
             "| move | candidate | incumbent | diff | 95% CI | kept |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r[2]}` | {r[3]:.4f} | {r[4]:.4f} | {r[5]:+.4f} | [{r[6]:+.4f}, {r[7]:+.4f}] | "
              f"{'**yes**' if r[8] else 'no'} |" for r in rows]
    lines += ["", f"Incumbent: `{inc_cfg.name}`", "", "```json", json.dumps(asdict(inc_cfg), indent=1), "```"]
    tracker.log_report("climb/report", "\n".join(lines), step=runs)
