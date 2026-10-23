# railhead-17m

A **17M-parameter System-1 decision model** for [Railhead](https://github.com/rachittshah/railhead), the
OpenTTD environment: given a company's wake-up state, it scores which air route to build now (or to hold),
in about a millisecond-scale forward pass on Apple silicon. It starts from
[Bekko System One v0 17M](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) (Ettin/ModernBERT
encoder, shared-prefix option branches), ported to native MLX, and is fine-tuned on **simulator
counterfactual labels**: every option is actually built in a fork of the game and played forward.

Status: work in progress. Results land in `runs/` and in the tables below as they are measured.

## Results so far (2026-10-23, incumbent `1023-1627-base-s1`)

**Closed loop**, 11 held-out 512x512 dev maps (10-year games + run-off), one seed per map, a decision every 180
days, player `play-v3` (falls back to the next option when a build fails without spending; routes and
"add planes" expansions). Arena score is Railhead's `tycoon_log` (saturates at 1.0 on rich maps).

| policy | mean arena score | mean asinh(final net worth / £100k) | model wins (maps) |
|---|---|---|---|
| **railhead-17m** | **0.735** | **4.213** | |
| random option | 0.733 | 4.227 | 6/11 |
| always longest route | 0.665 | 3.656 | 6/11 |
| hold (do nothing) | 0.000 | | |

**Offline**, simulator-valued decisions (asinh units; 0.1 is about 10% of net worth at the horizon):

| split | n | model regret | model play_regret | longest regret | longest play_regret | uniform play_regret |
|---|---|---|---|---|---|---|
| val | 128 | 0.210 | 0.190 | 0.258 | 0.223 | 0.480 |
| test | 82 | 0.316 | 0.295 | 0.635 | 0.568 | 1.070 |

`play_regret` replays the closed-loop player offline (options in the model's order, skipping builds that
failed). Hillclimbing (`runs/climb/ledger.jsonl`) uses seed-matched 3-seed comparisons with an A/A control per
pass; it has kept no move beyond noise at this data size (≈800 decisions); data and the action space
(expansions, continuation policy) have moved results, not settings. Next: `lbl-v4` labels (rollouts keep
adding planes after the option), ensemble distillation, DAgger.

## How the labels are made

```
decision state (savegame)                       options: 8 air candidates (routes-v1) + hold
        │
        ├── fork ─ build option (cash ladder: large/2, small/2, commuter/2, large/1, small/1, commuter/1)
        │            └── save ─┬─ reseed s1 ─ step 8 x 91 days ─ net worth curve
        │                      ├─ reseed s2 ─ ...                 (same seeds for every option:
        │                      └─ reseed s3 ─ ...                  common random numbers)
        └── value(option) = mean_s asinh(net worth at min(728 days, end of game) / £100k)
            target = softmax(value / tau)
```

- **States**: verified replays of frontier-model games (Railhead seed games, Hub Arena games), task starts,
  and scripted self-play (random / longest / mixed policies) on 25 local 512x512 maps and 151 Hub maps.
- **Splits by map**: `arena-dev-*` = test, `arena-train-0227/0228/0229` = val, other train maps = train.
- **Throughput** (M4 Max): fork 0.7 s, a 365-day step 1.0 s on a fresh map and 1.8 s at 148 vehicles;
  ~150-190 s per labelled state with 8 workers in parallel.

## The model

- `src/rh17m/bekko_mlx/`: native MLX Bekko-17M (inference + training), parity-tested against the published
  PyTorch runtime (see `docs/bekko-mlx.md`).
- Input: the wakeup-v1 rendered state (railhead-system1) as JSON, one choice question, options described in
  text (towns, populations, length, cost estimate vs cash). The same `dataset.observation` text is used in
  training and in closed-loop play.
- Training: soft-target cross-entropy, AdamW (encoder 1e-4, heads 5e-4), option-order shuffling and both
  prefix layouts as augmentation, epoch selection on val expected regret, temperature fitted on val NLL.

## Metrics

Offline, against simulator values (asinh units; 0.1 is about 10% of net worth at the horizon):
`regret` (best minus picked), `exp_regret` (under the model's distribution), `capture` (share of the best
option's edge over a random pick), `hit` (pick within seed noise of the best), `nll`, `ece`.
Closed loop: full games on held-out maps, Railhead's arena score (`tycoon_log`).

## Reproduce

```bash
uv venv --python 3.12 .venv && uv pip install -e '.[dev]'
.venv/bin/rh17m tasks                      # Hub tasks -> data/tasks
scripts/launch.sh                          # tmux session rh17m: harvest, label (train/val/test), observe, play, watch, dashboard
scripts/stop.sh                            # stop every stage cleanly (all stages resume on relaunch)
.venv/bin/rh17m baselines                  # scripted policies on the labelled decisions
.venv/bin/rh17m train --name v1            # one training run -> runs/train/v1
.venv/bin/rh17m climb                      # hillclimb with a noise floor -> runs/climb/ledger.jsonl
.venv/bin/rh17m play --tasks arena-dev-0212 --policies longest model --model runs/train/v1/model
```

## Credits

Bekko System One (MIT trainer, model by hotchpotch), Ettin rerankers (Apache-2.0), Railhead and
railhead-system1 (contract, candidates, snapshot, air builder), decision-mlx (MLX decision-model harness).
