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
- RESUMED 2026-10-24 08:38: climb pass 10 (1369 ex, 251 val) started with teacher retrain; labellers
  train x11, val x8, test x3; watch + Trackio dashboard (:7860). Incumbent still 1023-1627-base-s1.
  - [x] Perf (f27684c): OpenTTD servers under `taskpolicy -a` (daemon policy kept them off P-cores: 95-100 ->
        11 ms/game day), uncompressed temp saves in label jobs (1221 -> 52 ms/save, same checksum).
        Labels ~150/h -> 394/h, 18-20 -> 5.3 s/rollout. Build phase slower under full CPU (31 -> 60 s).
  - [x] E2E OTel tracing (fadbcbe): src/rh17m/tracing.py, Phoenix viewer :6006 (tmux window `traces`)
  - [x] CLIMB=0 launch switch (5d0202c)
- PAUSED 2026-10-24 ~10:15 at user request (fans/CPU): everything stopped via scripts/stop.sh. 2,618 labelled
  states. Incumbent restored to 1023-1627-base-s1 (two aborted passes had overwritten it). Climb passes 10/11
  were aborted (tmux killed by an orphan sweep at 09:20; GPU run stopped by user) -> never pkill ad hoc.
- PERF WORK 2026-10-29 (plan `thoughts/shared/plans/2026-10-29-cpu-hyperoptimize.md`; pipeline stayed paused):
  - [x] Phase 0 instrumentation (dfc9a1b): 1,269 bridge calls per option -> Phase 2 first
  - [x] Phase 2 pipelined site search (614b67c; rhs1 PR #1 still open): build_s 38.8 -> 18.8, 24/24 labels equal
  - [x] Phase 3 server pool code, opt-in RH17M_POOL=1 (d3da588, 614b13b). Reload correctness boxes in the
        plan are unticked; no label in data/ was made with the pool
  - [x] Phase 4 partly: GATE_WORKERS 8 / DAGGER_WORKERS 6 (4b7f254), sweep scripts (70d3cfd, 65c71a1)
  - Sweep, 14 workers, 1-seed jobs, 10-min windows (runs/sweep): 906 states/h base stack vs 1,650/h with
    pool + pipelined rhs1 + the pr-check engine. Launch defaults were NOT changed (still 22 workers, POOL=0).
  - Engine variant (flags + stdin poll, railhead PR #3): 11% faster uncontended; no keep/drop verdict recorded
- AUDIT 2026-11-10 (findings, nothing re-run): the climb keep test resamples single decisions although val is
  398 examples from 10 maps (three maps hold 373), and val also picks the epoch and fits the temperature;
  the closed-loop gate runs on Hub dev maps, which map_split calls `test`; tycoon labels value at 365 run-off
  days where the tasks say 730 (79 states); seven label versions are mixed with no version filter (1,561 of
  2,618 are lbl-v4). No closed-loop oracle (always pick the simulator-best option) has ever been run.
  - [x] ORACLE CEILING 2026-11-10 (src/rh17m/oracle.py, `rh17m play --policies oracle`, `rh17m play-report`):
        26 devhub maps x 2 seeds, paired, map bootstrap: oracle vs random +0.918 [+0.604,+1.213] (25/26 maps),
        vs model +0.747 [+0.452,+1.082] (24/26); model vs random +0.171 (n.s.) = 19% of the oracle's gain.
        -> the labels and action space carry value; the MODEL is the bottleneck; the eval resolves this size.
        Oracle decisions (373): best = route 48%, expand 37%, hold 15%; fork vs live build 316/316 equal.
        arena-dev-0100, 1 seed: model GBP 10.38M, random 7.09M, longest 7.33M (runs/play-arena0100).
  - Branches: audit/2026-11-10-fixes (PR #1 -> main), oracle/closed-loop-ceiling (PR #2 -> audit). The
    checkout is on the oracle branch. FINAL PR #3 = oracle/closed-loop-ceiling -> main (#1 + #2 in one).
    Merge order across repos: railhead-system1 #4, railhead #7, then this #3. Tests pass against both
    of those stacks (153 unit, plus the real-engine label test with the pipelined builder).
  - [x] Local suite, 11 ten-year maps x 1 seed: oracle vs random +1.277 [+0.547,+2.462], vs model +1.291.
  - [x] SEARCH (src/rh17m/search.py; `search:<k>` = model's top k + hold simulated with the oracle's code;
        `search:<k>:random` = control). Devhub 26x2: search:3 vs random +0.598 [+0.290,+0.899], share of
        oracle gain 0.65 at 3.8 options/decision; control 0.57; search:2 0.48; model alone 0.19.
        search:3 vs control +0.077 [-0.147,+0.243], 19/26 maps -> the lookahead does most of the work,
        the model is a weak prior.
- LOOP EDIT 2026-11-21 (branch loop/weight-edit, NOT committed; src/rh17m/loopedit.py, `rh17m loop-edit`,
  tests/unit/test_loopedit.py, evaluate.cluster_bootstrap_diff; 165 unit tests pass):
  - [x] The incumbent is now also a LOOPED transformer by weight edits alone (no gradient step):
        runs/loop/1023-1627-base-s1-loop2/model = layer 0 + one (sliding, sliding, full) block applied twice;
        layers 4-6 are byte-identical to 1-3 after reload, in MLX and in the published PyTorch runtime
        (strict load, 16 parity cases, max |p| diff 1.1e-6). Encoder-layer parameters 3.90M -> 2.23M unique.
  - Method: fold LayerNorm gains into the next Linear and re-gauge layers 4-6 onto 1-3 (head order, value
    basis, RoPE-pair phases, MLP neuron order + gate sign): exact (KL 4e-10). Then tie. Ranked on 400
    decisions from 16 held-out TRAIN maps by KL to the incumbent: copy 1-3 0.677, copy 4-6 0.758, mean 0.885,
    aligned mean 0.837, RegMean 0.395, RegMean + 4 Gauss-Seidel least-squares sweeps on the looped model's own
    activations 0.138 (chosen; 600 calibration decisions from the other train maps, option tokens x6).
    Sweeps plateau after 2; a least-squares head refit did not help (0.152 vs 0.147) and was removed.
  - It is a real loop: the same block once (4 layers) KL 0.737, twice 0.138, three times 0.146, four 0.148.
  - Not the same model: top-1 agreement with the incumbent 0.58 (val) / 0.59 (test), Spearman 0.79 / 0.83.
  - Labels (map bootstrap, looped minus incumbent): val (398 decisions, 10 maps) regret 0.336 vs 0.330,
    +0.007 [-0.046, +0.030]; test (204, 29 maps) 0.443 vs 0.423, +0.020 [-0.024, +0.073]; play_regret
    +0.009 / +0.014, both n.s. The incumbent has little edge to lose, so "n.s." is weak evidence.
  - NOT run: the closed-loop gate (needs the simulator; pipeline is paused). Val/test were printed by the
    tuning runs too; every choice (variant, sweeps, option weight) was made on the held-out train-map KL.
  - Next for this line: retrofit fine-tune from this export (the loop papers' recipe) and compare 3 seeds +
    A/A against the unlooped incumbent; then the gate.
- Now: [→] expert iteration: run the oracle and search policies on the TRAIN maps, train on those decisions,
  then re-run search:3 vs its control: the model's edge over random candidates is the number to move.
  - CORRECTION 2026-11-10: the ~1,300 oracle/search decisions already on disk are all on `-dev-` maps
    (map_split = test, and the closed-loop suites play those maps). Do NOT train on them.
  - `rh17m play` keeps each option's value per decision but not the observation text or a savegame, so
    turning a played decision into a training example needs a small converter (code, no compute).
  - Compute estimate (measured 2026-11-10, M4 Max, 8 games at once, pipelined builder): 31.4 worker-s per
    oracle decision = 896 decisions/h; training 0.028 s per example-epoch (6,100 examples x 8 epochs =
    23 min); search:3 on devhub 12.5 min per model. One game per train map (88 maps: 61 one-year, 16
    five-year, 11 ten-year) is 590 decisions, so a first round of 5 passes is about 3,000 decisions:
    data 3.3 h + 3 seeds 0.8 h + evaluation 2.5 h, about 7 h. Each further 1,000 decisions: 67 min.
    Savings measured on disk the same day: 2,045 of 6,763 oracle options (30%) failed to build with zero
    spend, so a pre-fork filter is worth trying; pruning to the top half of options after 182 days keeps
    the final best in only 61% of 1,341 lbl-v4 states (unsafe), after 364 days in 92% (mean regret 0.008).
    Vast.ai live offers that day: USD 0.005-0.012 per CPU thread-hour (64 threads 0.32-0.67/h, 128 threads
    0.80-1.00/h); thread speed vs an M4 worker assumed 0.2-0.3, UNCONFIRMED (run `rh17m bench` there).
- Next: fix the evaluation (cluster bootstrap by map, gate off the test maps, run-off from the task, label
  version filter) before any further climb pass; then expert iteration (distil the search back into the
  model); widen the action space (loan, wake cadence); HF publish still needs `hf auth login`.

## Open questions
- UNCONFIRMED: HF write token (user must run `hf auth login`).
- UNCONFIRMED: label noise floor with 3 seeds; whether 2 seeds suffice.

## Working set
- Repo: ~/railhead-17m (branch loop/weight-edit off oracle/closed-loop-ceiling = origin/main). Data: data/{states,labels,tasks,logs} (git-ignored).
- Pipeline: `scripts/launch.sh` (tmux session rh17m, one window per stage); `scripts/stop.sh`. Logs: data/logs/.
- Tests: `.venv/bin/python -m pytest -q tests/unit`.
