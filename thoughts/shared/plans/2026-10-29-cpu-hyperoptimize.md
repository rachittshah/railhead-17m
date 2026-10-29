# CPU Hyperoptimization Plan: labelling, closed-loop play and gate on the M4 Max

## Overview

Labelling (simulator counterfactual rollouts) is the binding constraint on the 17M goal: climb passes
stall for lack of new data, and every closed-loop gate costs ~25 min of CPU on the critical path of a pass.
All of it is OpenTTD simulation plus bridge round trips, single-threaded per server, zero GPU. This plan
takes the pipeline from the measured 394 states/h to the most this machine can do, one measured lever at
a time, with a stop rule per lever.

## Current State Analysis (all numbers measured this session, 2026-10-24)

Machine: M4 Max, 10 P-cores + 4 E-cores, 36 GB. Memory is not a lever (22 workers use ~8 GB).

Already shipped (commits f27684c, fadbcbe, 5d0202c):
- OpenTTD servers launch under `taskpolicy -a` (`src/rh17m/sched.py`, installed on package import so pool
  workers get it). Daemon scheduling had kept them off the P-cores: 95-100 ms/game day -> 11 ms.
- Uncompressed temp saves inside label jobs (`labels.fast_saves`): 1221 ms -> 52 ms per save, same
  checksum after 30 days.
- End-to-end OTel tracing to Phoenix (:6006): `label.job > world.load > server.start`, `label.build`,
  `label.build_affordable`, `label.rollout > world.step`, `play.game`, climb spans.

Baseline, 22 workers (train 11 / val 8 / test 3), 25-min window 09:40-10:05:

| metric | value |
|---|---|
| states labelled / h | **394** (was ~150 before the two fixes) |
| rollouts / h | ~5,760 |
| s per rollout (one option x one seed, 8 x 91-day steps) | 5.3 (was 18-20) |
| build phase per state (fork + build every option + save) | **60 s** (was 31 s: got *worse* under full CPU) |
| 1-seed train job | ~110 s = build 60 (55%) + rollouts ~50 (45%) |
| 3-seed val/test job | ~200 s = build 60 (30%) + rollouts ~140 (70%) |

One label job uncontended (traced, state arena-train-0204 w000): 78.9 s total.

| span | count | total | each |
|---|---|---|---|
| label.rollout | 10 | 52.4 s | 5.24 s |
| world.step (91 days) | 80 | 43.7 s | 0.55 s = **6 ms/game day** |
| label.build | 9 | 24.9 s | 2.76 s (fork 0.62 s; the rest is bridge calls) |
| world.load | 20 | 12.8 s | 0.64 s (server.start 0.53 s) |
| continue_expand | 30 | 1.0 s | |
| world.save | 19 | 0.33 s | |

The same state took 368 s under full pipeline load before the fixes: contention is the main cost, and it
lands on the **build phase** (latency-bound: hundreds of ~1 ms bridge round trips), not on rollouts
(throughput-bound: one `step` call per 91 days).

Engine (`~/railhead/.cache/openttd-src/build`, built by `~/railhead/engine/build.sh`): OpenTTD 15.3,
AppleClang 17, Ninja, Release `-O3 -DNDEBUG`, `OPTION_USE_ASSERTS=OFF` (already), **no `-mcpu`, no LTO, no
PGO**. Railhead patches: 0001 gamespeed/reseed, 0002 fast paused frames (1 ms), 0003 large GS JSON.
Runtime selection: `RAILHEAD_OPENTTD=<binary>`; source tree: `RAILHEAD_ENGINE_SRC`.

Other measured facts the plan builds on:
- `sample` of a busy rollout server (daemon policy): 67% of main-thread samples in `__select`
  (`InputWaiting`, `select` with a 1 us timeout, once per tick), 32% in `Tick`. A zero-timeout build was
  ~10% faster, but that A/B ran under the daemon policy where both arms were dominated by it. Unresolved.
- Server start under load 1.3-1.45 s: ~766 ms process to admin port, then Railhead waits (0.3 s settle
  sleep in `_confirm_admin`, 0.1 s poll in `_wait_ready`, up to 1 s hello retry in `Bridge.wait_ready`;
  a 0.05 s retry saved 260 ms).
- Console `load` of a savegame into a running server: **384 ms** under load, checksum identical to a
  fresh `World.load`. Only `checksum` was verified; company binding and builds after reload were not.
- `ServerConfig.debug = "script=2"` by default; nothing parses the log (`wait_ready` uses `hello` over the
  admin port; `tail_log` is only read on errors).
- Builder (`~/railhead-system1/src/rhs1/builders/air.py`): `candidate_sites` (radius 12 -> up to 625
  anchors) calls `air.airport dry_run` per tile until enough usable sites, then `catchment` per usable
  site; `gs_roundtrip_ms` 0.98, `airport_quote_ms` 1.01 uncontended. Calls per build: **not yet counted**.
- Idle paused servers (parent and build child) run 1 ms frames (patch 0002): ~1 kHz wakeups each.
- Gate: 52 games (26 devhub maps x 2 seeds), `workers=3`, ~25 min wall under load (19:37-20:01,
  20:29-20:55). `cli.py gate` takes `--workers`; `climb_loop.sh` does not pass it. DAgger self-play also
  runs with `--workers 3`.
- The worker count (22) has never been swept.

## Desired End State

- Labelling throughput on this Mac is the measured maximum over the levers below; every kept change has a
  measured delta >= 5% in a 25-min contended window, and every dropped one a recorded reason.
- Engine variants are checksum-identical to the stock Railhead engine over >= 5 saves x 728 days.
- Any builder change is result-identical (same chosen tiles, same `BuildResult`) on >= 20 states.
- Gate + DAgger use the full machine when they run.
- `rh17m bench labels` reports the window numbers; the ledger and this plan record the final table.

Plausible compounded outcome (estimate, basis in each phase): 1-seed jobs 110 s -> 45-60 s, 3-seed jobs
200 s -> 110-130 s, i.e. **~1.7-2.3x -> ~700-900 states/h**, and the gate from ~25 min to ~10 min.

## Decisions (settled; no open questions)

1. **Max throughput = max fans.** The user asked for the max level. High Power Mode is in scope (user-run
   sudo step). Reverting is one command.
2. **Railhead, railhead-system1 and the engine patches are the user's repos: branches + PRs, never direct
   edits.** Wrap from rh17m (the `sched.py` pattern) wherever a wrapper suffices.
3. **Never rebuild `~/railhead/.cache/openttd-src/build` with different flags.** Variants are built in a
   sibling tree and selected per process with `RAILHEAD_OPENTTD`.
4. **No game-logic changes** (tile loop, economy, vehicles). Labels must come from a simulator that matches
   the arena engine. Only scheduling, I/O, waiting, process lifetime and compiler flags are touched.
5. **Two measurement regimes.** Uncontended A/B (pipeline stopped) decides engine variants; contended
   25-min windows at a fixed worker count (pipeline running) give the verdict for everything.
6. **Stop rule:** a lever that measures < 5% is dropped and the reason recorded here.
7. **Process control only through `scripts/stop.sh` / `scripts/launch.sh`.** No ad-hoc `pkill` (it took the
   tmux session down twice this session).
8. **Phase 2 vs Phase 3 order is decided by the Phase 0 call count:** >= 300 bridge calls per build ->
   Phase 2 (batched site search) first; otherwise Phase 3 (server pool) first.

## What We're NOT Doing

- GPU, Oracle/cloud offload, RAM upgrades: separate decisions, not CPU optimisation (see the session notes).
- Changing label semantics (seeds, horizon, chunk days, option counts): those are climb moves.
- Shortening the gate (maps/seeds): it is the only closed-loop evidence we have.
- Rewriting the bridge transport (admin-port JSON) or the engine's lockstep model.
- Pinning threads to cores (macOS has no affinity API).

## Implementation Approach

Instrument first (Phase 0), then attack the two halves of a job in the order the call count dictates:
the rollout half with a faster engine (Phase 1), the build half with fewer round trips and fewer process
starts (Phases 2-3), then tune concurrency and the system (Phase 4), then re-baseline (Phase 5). Each phase
ends with a contended 25-min window via `rh17m bench labels` and a one-line verdict in this file.

---

## Phase 0: Instrumentation and gates  [DONE, dfc9a1b]

### Overview
The numbers that decide the rest: calls per build, a determinism gate for engine variants, and the window
report as a CLI command (no more ad-hoc `python -I`).

### Changes Required

#### 1. Bridge-call counter on the build span
**File**: `src/rh17m/tracing.py`, `src/rh17m/labels.py`
**Changes**: a context-local counter incremented by a `World.call` wrapper (installed like `sched.install`,
from rh17m, no Railhead edit); `label.build_affordable` and `label.snapshot` spans get `bridge_calls` and
`bridge_ms` attributes. Also on `play.build` (gate games build the same way).

#### 2. `rh17m bench determinism`
**File**: `src/rh17m/bench.py`, `src/rh17m/cli.py`
**Changes**: `measure_determinism(saves, binary, days=728)`: for each save, `World.load` (stock) and
`World.load` (variant via `RAILHEAD_OPENTTD`), `reseed` the same seed, step `days`, compare `checksum`
(and `state()["ticks"]`). >= 5 saves spanning vehicle counts (pick from `data/states`: an early state,
w031 with 148 vehicles, a late Hub state). Exit non-zero on any mismatch.

#### 3. `rh17m bench labels --since-min N`
**File**: `src/rh17m/bench.py`, `src/rh17m/cli.py`
**Changes**: the window report used by hand this session: states, states/h, rollouts, median build_s,
median s/rollout, split by seeds; reads `data/labels/*.json` mtimes. Also printed by `rh17m watch`.

#### 4. Baseline record
Run 25 min at the launch defaults (22 workers) and record here: states/h, s/rollout, build_s, and the
median `bridge_calls` per build from Phoenix.

### Success Criteria

#### Automated Verification:
- [ ] `.venv/bin/python -m pytest -q tests/unit` passes (new tests: counter attaches to spans with an
      in-memory exporter; `bench labels` on a fixture dir; `bench determinism` returns mismatch on a
      deliberately different seed)
- [ ] `.venv/bin/ruff check src tests` clean
- [ ] `rh17m bench determinism --binary $(readlink -f ~/railhead/.cache/openttd-src/build/openttd)` passes
      against itself (stock vs stock)

#### Manual Verification:
- [ ] Phoenix shows `bridge_calls` on `label.build_affordable`; the median per build is recorded below
- [ ] Baseline row recorded in the Results table

**Decision point:** median `bridge_calls` per build >= 300 -> do Phase 2 before Phase 3.

**Result (measured, one real label job, arena-train-0204 w000, uncontended):** `label.build_affordable` =
**1,269 bridge calls per option** (min = median = max over 8 route options), 1.0-3.4 ms per call
(1.3-4.3 s per option). Whole build phase: 10,173 calls, 18.9 s of 26.8 s (70%). `label.snapshot`: 16
calls. `play.build` (gate games): 1,269 on a first-rung success, 2,404-5,034 when the airport ladder
descends. Cause: `candidate_sites` dry-runs ALL 625 anchors per town (radius 12) with no early stop, 2 towns
per route. -> Phase 2 first.

---

## Phase 1: Engine build variants (no semantic change)

### Overview
Rollouts are 45-70% of a job and are pure engine CPU (6 ms/game day). Four compiler/driver levers, each
A/B'd uncontended with `measure_step` (alternating arms, >= 3 repeats) and gated by `bench determinism`.

### Changes Required

#### 1. Variant build tree
**Files**: Railhead PR: `engine/build.sh` gains `CMAKE_ARGS` passthrough (e.g.
`CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-mcpu=apple-m4 -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"`), honouring the
existing `RAILHEAD_ENGINE_SRC`. Build into `~/railhead/.cache/openttd-fast/` (symlink the OpenGFX tar into
its `baseset/`, as this session's scratch build needed).
**rh17m**: `scripts/launch.sh` exports `RAILHEAD_OPENTTD` when `ENGINE=fast` is set, so the variant is
opt-in per launch.

#### 2. Levers, in order (keep each only if >= 5% on step and checksums identical)
| lever | change | expected (estimate, basis) |
|---|---|---|
| a. `-mcpu=apple-m4` + LTO | CMake flags | 5-15% on step: typical for integer-heavy C++; unverified here |
| b. PGO | `-fprofile-generate` build, profile a real `rh17m label --workers 1 --only <state>` (builds + rollouts), `llvm-profdata merge`, `-fprofile-use` rebuild | further 5-15% (branchy tile/vehicle loops); unverified |
| c. stdin poll | new `0004-stdin-poll.patch`: `InputWaiting` with a 0 timeout (non-blocking) **or** poll stdin every 64 ticks (~5 ms at 81 us/tick; `quit` via stdin keeps working within that) | 0-40%: 67% of busy-server wall samples sat in `__select`; the earlier 10% A/B was confounded by the daemon policy. Re-test under `taskpolicy -a` |
| d. `-d script=2` off | `ServerConfig(debug=None)` from rh17m for label/rollout servers only (`BASE_SETTINGS`-style override, no Railhead edit) | 0-3%; drop if < 2% (the log is useful on failures) |

#### 3. Keep the stock engine for everything that must match the arena
Hub/arena games and `states` harvesting keep the stock binary unless the variant passes the determinism
gate on the Hub maps too (then it is simply the same simulator, faster).

### Success Criteria

#### Automated Verification:
- [ ] `rh17m bench determinism --binary ~/railhead/.cache/openttd-fast/build/openttd` passes on >= 5 saves
- [ ] `rh17m bench step --binary ... --repeat 3` (wrap of `measure_step`) shows the kept variant >= 5%
      faster, alternating arms, pipeline stopped
- [ ] Railhead PR CI green (build.sh still builds stock with no `CMAKE_ARGS`)

#### Manual Verification:
- [ ] 25-min contended window with `ENGINE=fast`: s/rollout improves by the uncontended ratio (+/- 20%)
- [ ] Results table row added; dropped levers recorded with their numbers

**Implementation Note**: pause here and confirm the window numbers before Phase 2/3.

---

## Phase 2: Build-phase latency (the contention lever)  [REDESIGNED: pipelining, no GameScript change]

### Overview
Build went 25 s -> 60 s per state under load while rollouts did not slow: it is latency-bound (Phase 0:
10,173 round trips per state). The first design (a GS-side batched `air.sites`) is replaced: the GameScript's
`HandleEvents` loops `while (GSEventController.IsEventWaiting())`, so it already drains every request that
is waiting at its next tick. Sending the 625 dry-runs back to back therefore costs a few frames, with **no
GameScript or Railhead change**. Also: the plan's "stops at `want` usable sites" was wrong, the Python loop
has no stop condition, so the batch is trivially equivalent (all affordable anchors, same order).

### Changes Required

#### 1. `World.call_many` (rh17m only)
**File**: `src/rh17m/pipeline.py` (installed on package import like `sched.py`; `RH17M_PIPELINE=0` turns it
off): sends requests with a sliding window of 256, matches replies by id (partial `p` pages, `r` finals,
stale ids ignored, events kept), returns one value per call with refusals as `BridgeError` values. Each
request still counts as a bridge call (`tracing.record_calls`), so spans show what a batch replaced.

#### 2. rhs1 PR (branch `perf/pipelined-site-search`, worktree `~/railhead-system1-batch`)
**File**: `src/rhs1/builders/air.py`: `candidate_sites` asks the first anchor alone (an unavailable airport
type is refused on every anchor: still one call), then pipelines the rest when the querier has `.batch`;
`build_air_route`'s `query` gets `.batch` when the world has `call_many`, recording the same quotes.
Without `call_many` the old one-call-at-a-time path runs unchanged. Tests: pipelined == sequential sites,
one probe call for an unavailable type, first non-site error wins, and `build_air_route` returns an
identical `BuildResult` and mutations with and without pipelining (5 scenarios).

#### 3. Secondary knobs (measure, keep if >= 5%)
- Paused frame interval as a console setting (extends patch 0002): 0.25-0.5 ms for build servers.
- Latency tier for build-phase servers: `taskpolicy -a -l 0` on fork/build servers only.

### Measured so far
Real engine, state w031 (148 vehicles), 3 towns, `rh17m bench pipeline`: 626 calls take **~630-660 ms
sequential vs ~20 ms pipelined (27-33x)**, identical site lists (91 / 1 / 158 sites).

### Success Criteria

#### Automated Verification:
- [x] rhs1 unit tests (130 passed incl. 9 new) and ruff clean; rh17m unit tests and ruff clean
- [x] `rh17m bench pipeline`: same sites on 3 real towns
- [ ] Full-record equivalence: 23 real states labelled with `RH17M_PIPELINE=0` vs `1`
      (`rh17m bench compare-labels A B`): options, builds and every rollout net worth identical

#### Manual Verification:
- [ ] 25-min contended window: median build_s per state <= 20 s (from 60), states/h recorded
- [ ] rhs1 PR opened (merge is the user's call)

---

## Phase 3: Persistent server pool (rh17m only)

### Overview
A label job starts ~20 OpenTTD processes (parent, 9-12 build forks, 9-27 rollouts) at 1.3-1.45 s each
under load; a console `load` into a live server costs 384 ms with an identical checksum. Reusing K servers
per worker removes process spawn, dyld, baseset scan and GS compile from every fork and rollout.

### Changes Required

#### 1. `WorldPool`
**File**: `src/rh17m/pool.py` (new)
**Changes**: per worker process, `K` long-lived `OpenTTDServer`s (default 2: one build child, one rollout;
the parent stays as today). `acquire(save, reseed=None) -> World`: rcon `load <save>` (copied into the
server's `save_dir`), `bridge.wait_ready`, `set_speed(0)`, `reseed`; `release(world)` returns the server
instead of `close()`. `fork` and `_rollout` in `labels.py` take the pool via `LabelConfig.pool_servers`
(0 = today's behaviour). Recycle a server after `N` loads (default 50) to bound any state leakage.

#### 2. Reload correctness
Verify, in a unit/integration test, after a reload: `state()["company"]` matches the save, a dry-run
`air.airport` works, `checksum` after 30 days equals a fresh `World.load`, events from the previous game
do not leak (`bridge.events` cleared), and `set_speed` is re-applied (the engine resets it on load).

#### 3. Railhead fixed waits (optional, low priority once the pool lands)
Railhead PR: `_confirm_admin` settle 0.3 s -> 0.1 s, `_wait_ready` poll 0.1 s -> 0.02 s, hello retry
cap 1.0 s -> 0.1 s. Measured value: ~0.5 s per start; paid once per server lifetime with the pool.

### Success Criteria

#### Automated Verification:
- [ ] `tests/unit/test_pool.py` (fake server) and the reload test pass; ruff clean
- [ ] Phoenix: `server.start` count per label job <= 3 (from ~20)

#### Manual Verification:
- [ ] 25-min contended window: build_s and s/rollout both drop; states/h recorded
- [ ] Memory per worker stable over 1 h (`ps` RSS of pooled servers) with recycling on

---

## Phase 4: Concurrency and system tuning

### Overview
22 workers was a guess. With a latency-bound build phase the optimum may be below the core count; with a
batched build it may be above. Also the gate and DAgger are under-parallelised on the climb's critical path.

### Changes Required

#### 1. Worker sweep
Run 25-min windows at total workers {12, 14, 18, 22, 28} (train:val:test in the 11:8:3 ratio), same
label config, same engine; `rh17m bench labels --since-min 25` after each. Pick the max states/h; record
the whole table. Repeat once after Phase 2/3 land (the optimum moves).

#### 2. Gate and DAgger parallelism
**File**: `scripts/climb_loop.sh`: pass `--workers ${GATE_WORKERS:-8}` to `gate` and
`--workers ${DAGGER_WORKERS:-6}` to the DAgger `states selfplay`. Expected: gate ~25 min -> ~10 min
(52 games, 3 -> 8 concurrent; games are independent). Labellers slow for that window; acceptable since
the gate is on the pass's critical path.

#### 3. RAM disk for label temp dirs (optional)
`scripts/launch.sh`: if `RAMDISK=1`, `hdiutil attach -nomount ram://8388608` (4 GB) + `diskutil
erasevolume APFS rh17m-tmp`, export `TMPDIR` for the label windows; detach in `stop.sh`. Covers
22 x ~90 MB of saves per in-flight job. Expected 1-2% (saves/loads of 3.5 MB files); keep only if measured.

#### 4. System (user-run, sudo)
- `! sudo powermetrics -n 1 -i 3000 --samplers cpu_power` at 22 workers: P-cluster frequency vs 4.4 GHz
  shows whether sustained load throttles. If it does, High Power Mode: `! sudo pmset -a powermode 2`
  (revert: `powermode 0`). Verify `pmset -g | grep powermode`.
- Spotlight: add `~/railhead-17m/data` under System Settings > Siri & Spotlight > Privacy
  (`knowledgeconstructiond` and `mds_stores` were seen at 50-80% CPU after a stop).

### Success Criteria

#### Automated Verification:
- [ ] `zsh -n scripts/launch.sh scripts/stop.sh scripts/climb_loop.sh`
- [ ] `tests/unit` + ruff clean

#### Manual Verification:
- [ ] Sweep table recorded; launch defaults updated to the winner
- [ ] One gate timed end to end at the new worker count (from `gates.jsonl` `at` and file mtimes)
- [ ] powermetrics result recorded (throttling yes/no)

---

## Phase 5: Re-baseline and record

- 25-min window at the final configuration; final Results row; ledger State/Next updated; this plan's
  Results table completed with the kept/dropped verdict per lever.
- Memory note for the CLAUDE memory dir: the final launch recipe and the measured ceiling.

## Results (filled in as phases complete)

| phase | change | states/h | s/rollout | build_s | verdict |
|---|---|---|---|---|---|
| baseline | 22 workers, taskpolicy -a, uncompressed temp saves | 394 | 5.3 | 60 | - |

## Testing Strategy

### Unit Tests
- Call counter attaches `bridge_calls`/`bridge_ms` to spans (in-memory exporter, as `test_tracing.py`).
- `bench labels` window maths on a fixture of label JSONs with controlled mtimes.
- `bench determinism` flags a mismatch when seeds differ; passes stock vs stock.
- `WorldPool` acquire/release/recycle with a fake server; reload re-applies speed and clears events.
- `sched.install` with a per-launch prefix (extends `test_sched.py`).

### Integration Tests
- Determinism: variant engine vs stock on >= 5 saves x 728 days (Phase 1 gate).
- Builder equivalence: batched vs per-tile site search on >= 20 states (Phase 2 gate).
- Reload: company/build/checksum after rcon `load` (Phase 3 gate).

### Manual Testing Steps
1. `scripts/stop.sh`; uncontended A/B via `rh17m bench step` (alternating arms, 3 repeats).
2. `scripts/launch.sh` (or `CLIMB=0 ...`); wait 25 min; `rh17m bench labels --since-min 25`.
3. Open Phoenix (:6006) and read the per-span breakdown of a fresh label job.
4. Record the row; apply the stop rule.

## Performance Considerations

- The build phase and the rollout phase compete differently for cores (latency vs throughput); the sweep
  must be re-run after any change that shifts their ratio.
- A faster engine makes the Python side (snapshot, candidates, JSON) a larger fraction; if `label.snapshot`
  or worker CPU exceeds ~10% of a job, profile the worker with `py-spy` before adding workers.
- Idle paused servers at 1 kHz are a small fixed tax per worker (parent + pool servers); the paused-frame
  setting in Phase 2 can raise the interval for idle servers if it shows up.

## Migration Notes

- Everything is opt-in by env (`ENGINE=fast`, `LabelConfig.pool_servers`, `RAMDISK=1`) until measured,
  then promoted to launch defaults. Labels carry `config` and `version`; a changed label config bumps
  `LabelConfig.version` so the dataset filter can tell them apart. Engine variants that pass the
  determinism gate do not change labels and need no version bump.
- Rollback: unset the env / restore the default in `launch.sh`; the stock engine binary is untouched.

## References

- Measurements: this session's transcript (2026-10-24), Phoenix traces (`data/phoenix`), `src/rh17m/bench.py`
  (`measure_rpc`, `measure_save_formats`, `measure_startup`, `measure_reload`, `measure_step`).
- Engine build: `~/railhead/engine/build.sh`, patches `~/railhead/engine/patches/000{1,2,3}-*.patch`.
- Builder: `~/railhead-system1/src/rhs1/builders/air.py:96-150` (`candidate_sites`).
- Server lifecycle: `~/railhead/src/railhead/server.py` (`_confirm_admin`, `_wait_ready`, `stop`),
  `world.py` (`fork`, `save`), `bridge.py` (`wait_ready`, `call`).
- Ledger: `thoughts/ledgers/CONTINUITY_CLAUDE-railhead-17m.md`.
