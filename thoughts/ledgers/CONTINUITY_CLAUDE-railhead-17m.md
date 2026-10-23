# CONTINUITY: railhead-17m

## Goal
A 17M-parameter System-1 decision model (Bekko-17M init, native MLX) that picks which air route to build in
Railhead/OpenTTD, trained on simulator counterfactual labels, hillclimbed hard, shipped continuously to the
private GitHub repo `rachittshah/railhead-17m` and a private HF model repo (`rachittshah/railhead-17m`, needs
`hf auth login`). Done = the 17M model beats every scripted baseline on held-out maps (offline regret, paired
bootstrap CI) and in closed-loop games, with weights + card on HF.

## Constraints
- Inherit MLX infra: decision-mlx (metrics, Bekko provider), railhead-system1 (contract, candidates, snapshot,
  air builder), Railhead (World, fork/reseed, replay). `uv pip install` only. Push to git often (user: "pushmax").
- Commits via the commit skill: no Claude attribution lines.
- Splits by map: arena-dev-* = test, arena-train-0227/0228/0229 = val, other train maps = train, *-ood-* = ood.
- Hillclimb decisions on val only; report test at milestones.

## Key decisions
- Labels: fork each option (8 air candidates + hold), build via ladder (large/2, small/2, commuter/2,
  large/1, small/1, commuter/1), roll 3 matched seeds x 728 days under hold continuation; value =
  mean asinh(net worth / 100k). Target = softmax(q / tau), tau 0.05. Zero-spend failures alias hold.
- Model: Bekko-17M (MIT trainer, Apache-2.0 Ettin base) ported to MLX by a sub-agent (src/rh17m/bekko_mlx).
- State sources: teacher replays (local seed games + Hub Arena games), task starts, self-play (random,
  longest, mix) on 25 local 512x512 maps and 151 Hub tycoon maps.

## State
- Done:
  - [x] Repo + private GitHub remote
  - [x] Sim bench: fork 0.7 s, 365-day step 1.0 s fresh / 1.8 s at 148 vehicles
  - [x] states.py (teacher, start, self-play, Hub), labels.py (lbl-v2), dataset.py, evaluate.py, hub.py
  - [x] Data pipeline running in tmux `rh17m-data` (window 0: local harvest + label loop; window 1: Hub)
- Now: [→] MLX Bekko port (sub-agent), training loop, first baseline table
- Next: train v1 → eval vs baselines → HF push → hillclimb loop (data, tau, features, lr, epochs, layouts)
- Remaining:
  - [ ] Closed-loop player (17M picks routes in full games) + paired eval on dev maps
  - [ ] Policy iteration: 17M as self-play policy to generate on-policy states
  - [ ] Extra questions (fleet_delta, next_wait) once route choice is solid

## Open questions
- UNCONFIRMED: HF write token (user must run `hf auth login`).
- UNCONFIRMED: label noise floor with 3 seeds; whether 2 seeds suffice.

## Working set
- Repo: ~/railhead-17m (branch main). Data: data/{states,labels,tasks,logs} (git-ignored).
- Pipeline: `scripts/data_pipeline.sh` (tmux rh17m-data). Progress: data/labels/_progress.log.
- Tests: `.venv/bin/python -m pytest -q tests/unit`.
