"""Simulator throughput on this machine: load, fork (save + load + reseed) and step costs.

The label engine forks every decision state K x M times and steps each branch H days, so these three
numbers size every labelling run. ``measure`` times a fresh start state; with ``log`` it first replays an
action log to its end, so the timings reflect a grown network (vehicles and stations slow the step).

Subcommands (``rh17m bench <name>``): ``step`` A/Bs an engine binary against the stock one, ``determinism``
checks a binary reaches the same game state as the stock one, ``labels`` reports the labelling rate of the
last N minutes. Without a subcommand, ``rh17m bench <task_dir>`` runs ``measure``.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from railhead.replay import replay
from railhead.world import World

DATA = Path(__file__).resolve().parents[2] / "data"
PINNED_STATE = "arena-train-0201.teacher-claude-opus-5-5.w031"  # 148 vehicles: the reference mid-game state


def _timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def measure(task_dir: Path, log: Path | None = None, days: int = 365, forks: int = 3) -> dict:
    """Time one load, ``forks`` reseeded forks and a ``days`` step on each fork. Returns seconds and
    game-days per second."""
    task_dir = Path(task_dir)
    with tempfile.TemporaryDirectory(prefix="rh17m-bench-") as tmp:
        tmp = Path(tmp)
        if log is None:
            world, load_s = _timed(lambda: World.load(tmp / "base", task_dir / "start.sav"))
        else:
            res, load_s = _timed(lambda: replay(task_dir / "start.sav", Path(log), tmp / "base", keep_open=True))
            if not res.valid or res.world is None:
                raise RuntimeError(f"replay diverged: {res.divergences[:2]}")
            world = res.world
        try:
            vehicles = len(world.call("vehicles"))
            fork_s, step_s = [], []
            for i in range(forks):
                child, s = _timed(lambda i=i: world.fork(tmp / f"fork{i}", reseed=1000 + i))
                fork_s.append(s)
                try:
                    _, s = _timed(lambda child=child: child.step(days=days))
                    step_s.append(s)
                finally:
                    child.close()
        finally:
            world.close()
    step_med = statistics.median(step_s)
    return {"task": task_dir.name, "replayed": log is not None, "vehicles": vehicles, "days": days,
            "load_or_replay_s": round(load_s, 3), "fork_s_median": round(statistics.median(fork_s), 3),
            "step_s_median": round(step_med, 3), "game_days_per_s": round(days / step_med, 1),
            "rollout_s": round(statistics.median(fork_s) + step_med, 3)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    sub = {"determinism": _determinism_main, "labels": _labels_main, "step": _step_main}
    if argv and argv[0] in sub:
        return sub[argv[0]](argv[1:])
    return _measure_main(argv)


def _measure_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="rh17m bench", description=__doc__)
    p.add_argument("task_dir", type=Path)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--forks", type=int, default=3)
    a = p.parse_args(argv)
    print(json.dumps(measure(a.task_dir, a.log, a.days, a.forks)))
    return 0


def measure_rpc(task_dir: Path, n: int = 200) -> dict:
    """Latency of the pieces a label job is made of: server start (World.load), one GameScript round trip
    (``state``), a dry-run airport quote (what site searches issue by the hundred), a savegame write, and a
    one-day step. Medians in milliseconds."""
    task_dir = Path(task_dir)
    with tempfile.TemporaryDirectory(prefix="rh17m-rpc-") as tmp:
        tmp = Path(tmp)
        world, load_s = _timed(lambda: World.load(tmp / "w", task_dir / "start.sav"))
        try:
            towns = world.call("towns")
            tile = int(towns[0]["tile"])
            from railhead.actions import to_gs

            method, args = to_gs("air.airport", {"tile": tile, "type": "large", "join_station": -1, "dry_run": True})
            rt, quote, step1 = [], [], []
            for _ in range(n):
                rt.append(_timed(lambda: world.call("state"))[1])
            for _ in range(min(n, 100)):
                t0 = time.perf_counter()
                with contextlib.suppress(Exception):  # a refused site still costs a round trip
                    world.call(method, **args)
                quote.append(time.perf_counter() - t0)
            save_s = statistics.median(_timed(lambda i=i: world.save(tmp / f"s{i}.sav"))[1] for i in range(5))
            for _ in range(20):
                step1.append(_timed(lambda: world.step(days=1))[1])
        finally:
            world.close()
    def med(xs):
        return round(1e3 * statistics.median(xs), 2)

    return {"server_start_ms": round(1e3 * load_s, 1), "gs_roundtrip_ms": med(rt), "airport_quote_ms": med(quote),
            "save_ms": round(1e3 * save_s, 1), "step_1day_ms": med(step1)}


def measure_save_formats(task_dir: Path, formats=("lzma", "zlib", "zlib:1", "lzo", "none"), state: Path | None = None) -> list[dict]:
    """Save, size and server start (load) cost per OpenTTD ``savegame_format``. The format does not change the
    game state (replays and checksums are unaffected), only how fast forks write and read their savegames.
    ``state`` is a grown savegame to measure on (default: the task's start)."""
    import railhead.server as S

    src = Path(state) if state else Path(task_dir) / "start.sav"
    out = []
    base = {k: dict(v) for k, v in S.BASE_SETTINGS.items()}
    try:
        for fmt in formats:
            S.BASE_SETTINGS.clear()
            S.BASE_SETTINGS.update({k: dict(v) for k, v in base.items()})
            S.BASE_SETTINGS.setdefault("misc", {})["savegame_format"] = fmt
            with tempfile.TemporaryDirectory(prefix="rh17m-fmt-") as tmp:
                tmp = Path(tmp)
                w = World.load(tmp / "w0", src)
                try:
                    saves = []
                    for i in range(3):
                        saves.append(_timed(w.save, tmp / f"s{i}.sav")[1])
                    size = (tmp / "s0.sav").stat().st_size
                finally:
                    w.close()
                loads, check = [], None
                for i in range(3):
                    w2, s = _timed(World.load, tmp / f"l{i}", tmp / "s0.sav")
                    loads.append(s)
                    try:
                        if i == 0:  # same future from every format: the format only changes the bytes
                            w2.step(days=30)
                            check = w2.call("checksum")
                    finally:
                        w2.close()
            out.append({"format": fmt, "save_ms": round(1e3 * statistics.median(saves)),
                                                "start_from_save_ms": round(1e3 * statistics.median(loads)), "size_kb": size // 1024,
                        "checksum_after_30d": check})
    finally:
        S.BASE_SETTINGS.clear()
        S.BASE_SETTINGS.update(base)
    return out


def measure_startup(savegame: Path, n: int = 5, hello_retry_s: float | None = None) -> dict:
    """Server start phases: process up and admin port answering (OpenTTDServer.start), then the World
    handshake (admin join, GameScript ``hello``, gamespeed). ``hello_retry_s`` replaces the bridge's 1 s
    hello timeout (a hello sent before the GameScript VM starts is dropped and waits out the full timeout)."""
    import railhead.bridge as Bmod
    from railhead.server import OpenTTDServer, ServerConfig

    orig = Bmod.Bridge.wait_ready

    def fast_wait(self, timeout: float = 30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.hello = self.call("hello", timeout=hello_retry_s)
                return self.hello
            except TimeoutError:
                continue
        raise TimeoutError("RailheadBridge GameScript did not answer hello")

    if hello_retry_s is not None:
        Bmod.Bridge.wait_ready = fast_wait
    proc_s, world_s = [], []
    try:
        for i in range(n):
            with tempfile.TemporaryDirectory(prefix="rh17m-start-") as tmp:
                cfg = ServerConfig()
                cfg.savegame = Path(savegame).resolve()
                server, s1 = _timed(OpenTTDServer(Path(tmp) / "w", cfg).start)
                world, s2 = _timed(World, server)
                world.close()
                proc_s.append(s1)
                world_s.append(s2)
    finally:
        Bmod.Bridge.wait_ready = orig
    return {"hello_retry_s": hello_retry_s, "process_ms": round(1e3 * statistics.median(proc_s)),
            "handshake_ms": round(1e3 * statistics.median(world_s)),
            "total_ms": round(1e3 * statistics.median([a + b for a, b in zip(proc_s, world_s, strict=True)]))}


def measure_reload(savegame: Path, n: int = 5) -> dict:
    """Load a savegame into an already running server (console ``load``) instead of starting a new one, and
    check the bridge still answers with the reloaded state. Returns timings and whether the reloaded world
    matches a freshly started one (checksum)."""
    import shutil

    with tempfile.TemporaryDirectory(prefix="rh17m-reload-") as tmp:
        tmp = Path(tmp)
        w = World.load(tmp / "w", savegame)
        try:
            ref = w.call("checksum")
            w.step(days=30)  # move away so a successful reload is visible
            shutil.copy(savegame, w.server.save_dir / "reload_probe.sav")
            times, ok = [], []
            for _ in range(n):
                t0 = time.perf_counter()
                out = w.admin.rcon("load reload_probe.sav")
                w.bridge.wait_ready(timeout=30)
                w.set_speed(0)
                times.append(time.perf_counter() - t0)
                ok.append(w.call("checksum") == ref)
                w.step(days=30)
        finally:
            w.close()
    return {"reload_ms": round(1e3 * statistics.median(times)), "matches_fresh_load": all(ok), "rcon": out[:2]}


@contextlib.contextmanager
def engine(binary: Path | None):
    """Start every server inside the block from ``binary`` (railhead's ``RAILHEAD_OPENTTD``); ``None`` leaves
    the engine selection as it is."""
    if binary is None:
        yield
        return
    old = os.environ.get("RAILHEAD_OPENTTD")
    os.environ["RAILHEAD_OPENTTD"] = str(binary)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RAILHEAD_OPENTTD", None)
        else:
            os.environ["RAILHEAD_OPENTTD"] = old


def stock_engine() -> Path:
    """The engine a plain run uses (railhead's default search), whatever ``RAILHEAD_OPENTTD`` says: the
    reference arm of every A/B must not be the variant that a launch script exported."""
    from railhead.engine import find_openttd

    old = os.environ.pop("RAILHEAD_OPENTTD", None)
    try:
        return find_openttd()
    finally:
        if old is not None:
            os.environ["RAILHEAD_OPENTTD"] = old


def measure_step(savegame: Path, prefix: tuple[str, ...] = (), binary: Path | None = None, days: int = 91,
                 n: int = 3) -> dict:
    """ms per game day for a loaded state, with the server launched under ``prefix`` (e.g.
    ``taskpolicy -a -l 0``; it replaces sched's ``taskpolicy -a``) and/or a different engine ``binary``. The
    dedicated driver polls stdin with a 1 us ``select`` every tick, so how macOS schedules that timer (QoS,
    timer coalescing) shows up directly in the step cost."""
    from railhead import server as srv

    from rh17m import sched

    orig = srv.OpenTTDServer.command

    def command(self):
        cmd = orig(self)
        if prefix:
            if cmd[:len(sched.PREFIX)] == sched.PREFIX:
                cmd = cmd[len(sched.PREFIX):]
            cmd = [*prefix, *cmd]
        return cmd

    srv.OpenTTDServer.command = command
    try:
        with engine(binary), tempfile.TemporaryDirectory(prefix="rh17m-step-") as tmp:
            w = World.load(Path(tmp) / "w", savegame)
            try:
                times = [_timed(lambda: w.step(days=days))[1] for _ in range(n)]
            finally:
                w.close()
    finally:
        srv.OpenTTDServer.command = orig
    return {"prefix": " ".join(prefix), "binary": str(binary or "default"),
            "ms_per_day": round(1e3 * statistics.median(times) / days, 2)}


def compare_step(savegame: Path, variant: Path | None = None, prefix: tuple[str, ...] = (), repeat: int = 3,
                 days: int = 91) -> dict:
    """``measure_step`` of the stock engine vs a variant (another binary and/or a launch ``prefix``), one
    fresh load and one ``days`` step per arm per repeat, arms alternating so drift hits both. ``ratio`` is
    variant over stock ms/day (< 1 = faster)."""
    stock = stock_engine()
    ms: dict[str, list[float]] = {"stock": [], "variant": []}
    for _ in range(repeat):
        ms["stock"].append(measure_step(savegame, binary=stock, days=days, n=1)["ms_per_day"])
        ms["variant"].append(measure_step(savegame, prefix=tuple(prefix), binary=variant or stock, days=days,
                                          n=1)["ms_per_day"])
    a, b = statistics.median(ms["stock"]), statistics.median(ms["variant"])
    return {"save": Path(savegame).name, "days": days, "repeat": repeat, "stock_binary": str(stock),
            "variant_binary": str(variant or stock), "variant_prefix": " ".join(prefix), "stock": ms["stock"],
            "variant": ms["variant"], "stock_ms_per_day": a, "variant_ms_per_day": b, "ratio": round(b / a, 4),
            "faster_pct": round(100 * (1 - b / a), 1)}


def _step_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="rh17m bench step", description=compare_step.__doc__)
    p.add_argument("--binary", type=Path, default=None, help="variant engine (default: the stock engine)")
    p.add_argument("--prefix", default="", help='variant launch prefix, e.g. "taskpolicy -a -l 0"')
    p.add_argument("--save", type=Path, default=DATA / "states" / f"{PINNED_STATE}.sav")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--days", type=int, default=91)
    a = p.parse_args(argv)
    if a.binary is None and not a.prefix:
        p.error("give a variant: --binary and/or --prefix")
    if a.binary is not None and not a.binary.is_file():
        p.error(f"no such engine binary: {a.binary}")
    res = compare_step(a.save, a.binary, tuple(a.prefix.split()), a.repeat, a.days)
    print(f"stock   {res['stock_ms_per_day']:7.2f} ms/day  {res['stock']}")
    print(f"variant {res['variant_ms_per_day']:7.2f} ms/day  {res['variant']}")
    print(f"ratio {res['ratio']:.3f}  ({res['faster_pct']:+.1f}% faster)")
    print(json.dumps(res))
    return 0


# ---------------------------------------------------------------- determinism

def pick_saves(root: Path, n: int = 5, pinned: str | None = PINNED_STATE) -> list[Path]:
    """At most ``n`` saves spanning the vehicle counts in ``root`` (read from the states' json, no loading):
    the ``pinned`` state if present, then evenly spaced distinct counts from the emptiest to the busiest.
    Ordered by vehicle count; the choice among equal counts is the first state id."""
    root = Path(root)
    by_count: dict[int, list[str]] = {}
    for meta in sorted(root.glob("*.json")):
        if meta.name.endswith(".error.json") or not meta.with_suffix(".sav").exists():
            continue
        try:
            v = json.loads(meta.read_text()).get("vehicles")
        except (OSError, ValueError):
            continue
        if isinstance(v, int):
            by_count.setdefault(v, []).append(meta.stem)
    chosen: dict[int, str] = {}
    pin = next((c for c, ids in by_count.items() if pinned in ids), None)
    if pinned is not None and pin is not None:
        chosen[pin] = pinned
    counts = sorted(c for c in by_count if c not in chosen)
    want = n - len(chosen)
    if want >= len(counts):
        picks = counts
    elif want > 1:
        picks = [counts[round(i * (len(counts) - 1) / (want - 1))] for i in range(want)]
    else:
        picks = counts[-1:] if want == 1 else []  # a single pick goes to the busiest state
    for c in picks:
        chosen[c] = by_count[c][0]
    return [root / f"{chosen[c]}.sav" for c in sorted(chosen)]


def run_arm(save: Path, binary: Path, days: int, seed: int) -> dict:
    """Load ``save`` on engine ``binary``, reseed, step ``days``: the game's identity afterwards (checksum and
    tick count) and the wall seconds it took."""
    t0 = time.perf_counter()
    with engine(binary), tempfile.TemporaryDirectory(prefix="rh17m-det-") as tmp:
        w = World.load(Path(tmp) / "w", Path(save))
        try:
            out = w.admin.rcon(f"reseed {seed & 0xFFFFFFFF}")
            if not out or not out[0].startswith("Reseeded"):
                raise RuntimeError(f"engine lacks the reseed patch: {out!r}")
            w.step(days=days, timeout=3600)
            return {"checksum": w.call("checksum"), "ticks": w.state()["ticks"],
                    "s": round(time.perf_counter() - t0, 1)}
        finally:
            w.close()


def mismatches(a: dict, b: dict) -> list[str]:
    """The game-state fields (checksum, ticks) in which two arms differ; wall time is not state."""
    return [k for k in ("checksum", "ticks") if a.get(k) != b.get(k)]


def measure_determinism(saves: list[Path], binary: Path, days: int = 728, seed: int = 1,
                        variant_seed: int | None = None, stock: Path | None = None, arm=None) -> list[dict]:
    """Per save: the stock engine and ``binary`` each load it, reseed (``variant_seed``, default the same
    ``seed``: a different one is the negative control and must show up as a mismatch), step ``days`` and are
    compared. The engine does not change the game, so every row must be ``ok``."""
    arm = arm or run_arm
    stock = stock or stock_engine()
    rows = []
    for save in saves:
        a = arm(save, stock, days, seed)
        b = arm(save, binary, days, seed if variant_seed is None else variant_seed)
        bad = mismatches(a, b)
        meta = Path(save).with_suffix(".json")
        veh = json.loads(meta.read_text()).get("vehicles") if meta.exists() else None
        rows.append({"save": Path(save).name, "vehicles": veh, "days": days, "ok": not bad, "mismatch": bad,
                     "stock": a, "variant": b})
    return rows


def _identity(arm: dict) -> str:
    c = arm["checksum"]  # railhead's checksum is {date, h, ticks, vehicles}
    state = f"h={c['h']} date={c['date']} vehicles={c['vehicles']}" if isinstance(c, dict) else str(c)
    return f"{state} ticks={arm['ticks']}"


def _determinism_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="rh17m bench determinism", description=measure_determinism.__doc__)
    p.add_argument("--binary", type=Path, required=True, help="variant engine")
    p.add_argument("--stock", type=Path, default=None, help="reference engine (default: railhead's stock one)")
    p.add_argument("--days", type=int, default=728)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--variant-seed", type=int, default=None, help="negative control: reseed the variant arm differently")
    p.add_argument("--saves", nargs="*", type=Path, default=None, help="default: spread over data/states by vehicles")
    p.add_argument("--root", type=Path, default=DATA / "states")
    p.add_argument("--count", type=int, default=5, help="saves to pick when --saves is not given")
    p.add_argument("--list", action="store_true", help="print the saves that would be compared and exit")
    a = p.parse_args(argv)
    saves = a.saves if a.saves else pick_saves(a.root, a.count)
    if a.list:
        for s in saves:
            meta = Path(s).with_suffix(".json")
            print(Path(s).name, json.loads(meta.read_text()).get("vehicles") if meta.exists() else "?")
        return 0
    missing = [str(x) for x in [a.binary, *(a.saves or [])] if not Path(x).is_file()]
    if missing or not saves:
        p.error(f"not found: {', '.join(missing) or 'no saves under ' + str(a.root)}")
    stock = a.stock or stock_engine()
    print(f"stock    {stock}\nvariant  {a.binary}", flush=True)
    t0 = time.perf_counter()
    rows = []
    for save in saves:
        r = measure_determinism([save], a.binary, a.days, a.seed, a.variant_seed, stock)[0]
        rows.append(r)
        print(f"{'OK' if r['ok'] else 'MISMATCH':8} {r['save']}  veh={r['vehicles']}  {r['days']}d  "
              + (_identity(r["stock"]) if r["ok"] else
                 f"stock {_identity(r['stock'])}  variant {_identity(r['variant'])}  differs: {','.join(r['mismatch'])}")
              + f"  ({r['stock']['s']} s stock, {r['variant']['s']} s variant)", flush=True)
    good = sum(r["ok"] for r in rows)
    print(f"determinism: {good}/{len(rows)} saves identical over {a.days} days, "
          f"wall {time.perf_counter() - t0:.0f} s")
    return 0 if good == len(rows) else 1


# ---------------------------------------------------------------- labelling rate

def _aggregate(rows: list[dict], since_min: float) -> dict:
    def med(key: str, ndigits: int = 2):
        xs = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(xs), ndigits) if xs else None

    return {"states": len(rows), "states_per_h": round(len(rows) * 60 / since_min, 1),
            "rollouts": sum(r["rollouts"] for r in rows), "build_s_median": med("build_s"),
            "s_per_rollout_median": med("s_per_rollout"), "bridge_calls_median": med("bridge_calls"),
            "bridge_ms_median": med("bridge_ms", 1)}


def label_window(labels: Path, since_min: float = 25, now: float | None = None) -> dict:
    """The labelling rate of the last ``since_min`` minutes, from the records' mtimes: states, states/h,
    rollouts (aliases of hold excluded: they repeat its rollouts), and the medians per state of build_s,
    seconds per rollout ((total_s - build_s) / rollouts; states with no rollouts have none) and the
    GameScript calls of the build phase (records that carry them). ``by_seeds`` splits by the number of
    rollout seeds (1 = train, 3 = val/test)."""
    cutoff = (time.time() if now is None else now) - 60 * since_min
    rows = []
    for path in Path(labels).glob("*.json"):
        if path.name.endswith(".error.json") or path.name.startswith(("_", ".")):
            continue
        try:
            if path.stat().st_mtime < cutoff:
                continue
            rec = json.loads(path.read_text())
        except (OSError, ValueError):  # removed or half-written under us
            continue
        t = rec.get("timing") or {}
        rollouts = sum(len(o.get("rollouts") or []) for o in (rec.get("outcomes") or {}).values()
                       if "alias" not in o)
        build_s, total_s = t.get("build_s"), t.get("total_s")
        rows.append({"seeds": len(rec.get("seeds") or []), "rollouts": rollouts, "build_s": build_s,
                     "s_per_rollout": (total_s - build_s) / rollouts if rollouts and None not in (build_s, total_s)
                     else None, "bridge_calls": t.get("bridge_calls"), "bridge_ms": t.get("bridge_ms")})
    out = {"since_min": since_min, **_aggregate(rows, since_min)}
    out["by_seeds"] = {k: _aggregate([r for r in rows if r["seeds"] == k], since_min)
                       for k in sorted({r["seeds"] for r in rows})}
    return out


def window_line(r: dict) -> str:
    """``label_window`` as one line (the watcher's log and the CLI's headline)."""
    def part(a: dict) -> str:
        s = f"{a['states']} states ({a['states_per_h']:.1f}/h), {a['rollouts']} rollouts"
        if a["build_s_median"] is not None:
            s += f", build {a['build_s_median']:.0f} s"
        if a["s_per_rollout_median"] is not None:
            s += f", {a['s_per_rollout_median']:.1f} s/rollout"
        if a["bridge_calls_median"] is not None:
            s += f", gs calls {a['bridge_calls_median']:.0f}"
        return s

    return f"last {r['since_min']:g} min: {part(r)}" + "".join(
        f" | {k}-seed: {part(a)}" for k, a in r["by_seeds"].items())


def _labels_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="rh17m bench labels", description=label_window.__doc__)
    p.add_argument("--labels", type=Path, default=DATA / "labels")
    p.add_argument("--since-min", type=float, default=25)
    a = p.parse_args(argv)
    r = label_window(a.labels, a.since_min)
    print(window_line(r))
    print(json.dumps(r))
    return 0
