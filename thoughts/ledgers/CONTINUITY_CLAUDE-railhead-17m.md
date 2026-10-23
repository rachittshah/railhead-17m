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
  - [x] Data pipeline running in tmux `rh17m-data` (window 0: local harvest + label loop; window 1: Hub;
        label2: second labeller, order seed 1; watch: data metrics; trackio: dashboard :7860)
  - [x] MLX Bekko-17M port: parity 1.9e-6 vs PyTorch runtime, 5.4 ms/decision, 116 ex/s compiled bf16
  - [x] train.py (Trackio-tracked), climb.py (noise floor + paired bootstrap), play.py, hub publish
  - [x] Labels lbl-v2.2: NO_SITE ladder step, no-build states skip rollouts, rollouts stop at game end
  - [x] Trackio: decision-mlx PR #7 merged (decision_mlx.tracking.Tracker, `evaluate --track`); rh17m
        uses it (vendored fallback src/rh17m/_tracking.py until the local decision-mlx checkout has #7).
        Dashboard: tmux rh17m-data:trackio, http://127.0.0.1:7860
  - [x] Val = 3 local maps + hashed 1/8 of train maps (17 maps); 4,604 states harvested
  - Early signal (51 ex): best option is rarely the longest (2/50) or hold (1/51); mostly 2nd-4th longest;
    median best-worst spread 0.275 asinh
  - [x] decision-mlx PRs #8/#9 merged: Apple GPU util/memory (ioreg), MLX memory, host CPU/RAM sampled into
        Trackio System Metrics via the Run object (ContextVar bug fixed); rh17m samples every 2 s
  - [x] Closed-loop baselines, 11 dev maps, wake 180 d: random 0.689, longest 0.490, longest_affordable
        0.099, hold 0.000 (runs/play/summary.json)
  - [x] Harvest complete: 5,119 states, all observed (data/obs) for distillation
  - [x] decision-mlx PR #10 merged: Tracker tables/histograms/images/reports/traces/alerts/files/artifacts;
        rh17m fills every Trackio tab (train, climb project railhead-17m-climb, play project railhead-17m-play)
  - [x] Climb pass 1 (238 ex, 59 val): noise floor 0.011, 0 of 23 moves kept -> data-bound
  - [x] Teacher: Bekko-68M on same labels (val exp_regret 0.303 vs 17M ~0.31), runs/teachers/current
  - [x] Labels lbl-v2.4: train states 1 seed (CRN), eval splits 3; labellers train x14, val x5, test+ood x3
  - [x] DAgger: model self-play policy (epsilon 0.2); climb loop runs it after each pass; labelled first
  - Incumbent 1023-1209-base-s2 on arena-dev-0003: £4.69M (0.892) vs longest £4.09M (0.858)
  - [x] Climb fix: seed-matched, seed-averaged comparisons (old rule compared to best-of-seeds incumbent)
  - [x] Player v2: fall back to next option on a no-spend failure, block failed routes (v1 model built OK on
        only 24% of wake-ups; ERR_NO_SITE repeats). v1 games archived in runs/play-v1
  - [x] decision-mlx PR #11: Tracker logs via its own Run (nested trackers); policy inference on CPU; climb
        retries Metal errors (pass-2 crash: 'victim of GPU error/recovery')
  - [x] play_regret metric (offline replay of player v2); climb keep metric = play_regret from pass 3
  - [x] Freezing moves (embeddings 12.9M of 16.8M; +3 layers; head-only) and select_metric moves
  - [x] MLX cache capped at 2 GB + cleared per run: climb had 21.6 GB cached (31.8 GB GPU alloc) -> swap,
        load 200, labels 360-460 s/state; after fix labels recovered (~60/h -> ~157/h and rising)
  - Climb predictions cached per (model, example text)
  - [x] Pass 3 (play_regret, 2 seeds): all moves 'lost' by ~0.04 -> A/A control showed it is noise:
        base seeds differ by 0.041, same config/seeds retrained moved +0.027 [-0.005,+0.063] (training is not
        bit-deterministic run to run). Climb now: 3 seeds, exp_regret keep metric, A/A control each pass
  - [→] Expansion options (lbl-v3, src/rh17m/expand.py): add 2 (or 1) planes to the 3 most profitable air
        routes; labels/play/self-play/observe all execute options via expand.apply_option (player play-v3).
        NOT yet verified end to end; label test running (b794wy2pt); labellers still on code loaded before v3
  - [x] Expansions verified on a real state (3 options, +2 planes, values between routes and hold); lbl-v3
        labelling live; first 101 v3 states: an expansion is the best option in 26
  - [x] Training ~2x faster: cached tokenisation (5.5 ms/ex -> 0), early stopping (patience 3)
  - [x] Ensemble distillation (@ensemble = incumbent's seed models); obs-v3 (with expansions) regenerating
  - [x] Keep rule: noise = max(seed sample-std floor, |A/A control diff|) (pass 5 control +0.0050 > floor 0.0025)
  - Closed loop under play-v3: baselines re-running (runs/play/baselines-v3); v1/v2 games archived
  - [x] CLOSED LOOP (play-v3, 11 dev maps, wake 180 d): incumbent 1023-1627-base-s1 mean arena score 0.735 vs
        random 0.733 vs longest 0.665; asinh net worth 4.213 vs 4.227 vs 3.656 (model ~= random, >> longest).
        Model expands a lot (10-17 of ~20 builds/game). Arena score saturates at 1.0 on rich maps.
  - [x] lbl-v4: rollouts continue with an 'expand' policy (1 plane on best route every 182 d) to fix myopia of
        hold continuation; labellers switched 17:55. DatasetConfig.continuation filter + climb moves
  - [x] Climb loop retrains a Bekko-68M teacher each pass (17:13 teacher: val play_regret 0.270 vs longest 0.248)
  - [x] Ensemble play (comma-separated model dirs); running ens-1023-1627 closed loop
  - [x] devhub closed-loop suite (26 Hub dev maps 256x256, 2 seeds, paired bootstrap on asinh net worth):
        incumbent-1627 vs random +0.171 [-0.231,+0.529], vs longest +0.269 [-0.120,+0.614] (n.s.)
  - [x] KEY FINDING: val gains did not transfer. Pass-7 self-distilled (kept offline) -0.405 [-0.787,-0.036]
        and ensemble-distilled -0.440 [-0.836,-0.074] vs incumbent-1627 in closed loop. Incumbent restored to
        1023-1627-base-s1. 3-seed ensemble of 1627 also worse on 11 local maps (0.650 vs 0.735; noisy)
  - [x] Closed-loop gate in climb loop (rh17m gate, runs/climb/gates.jsonl); climb judges on lbl-v4 val;
        label-generation moves first, distillation last
  - [x] Diagnostic (26 devhub maps): pass-7 base (no distill) vs incumbent-1627 -0.123 [-0.406,+0.179] (data
        change n.s.); self-distill vs pass-7 base -0.282 [-0.634,+0.037]; ens-distill vs base -0.317
        [-0.667,+0.002] -> DISTILLATION ITSELF HURTS closed loop despite val gains. Moves now last + gated.
  - [x] Pass 8 artifact: 'expansions:false' kept because its val had fewer options (lower regret by
        construction). Fix: all configs judged on the same decisions (options/values/targets from the base
        config; only text keys follow the candidate). Incumbent restored again to 1023-1627-base-s1.
- Now: [→] climb pass 9 (22:37, 1356 ex, 247 val; v4 val judging; same-decision judging; closed-loop gate)
- Next: if v4-only training wins offline AND in the gate, the label generation question is settled
- Open: UNCONFIRMED flaky unit test (1 failure once under heavy load, not reproduced in 4 runs)
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
- Pipeline: `scripts/launch.sh` (tmux session rh17m, one window per stage); `scripts/stop.sh`. Logs: data/logs/.
- Tests: `.venv/bin/python -m pytest -q tests/unit`.
