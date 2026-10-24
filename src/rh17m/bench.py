"""Simulator throughput on this machine: load, fork (save + load + reseed) and step costs.

The label engine forks every decision state K x M times and steps each branch H days, so these three
numbers size every labelling run. ``measure`` times a fresh start state; with ``log`` it first replays an
action log to its end, so the timings reflect a grown network (vehicles and stations slow the step).
"""

from __future__ import annotations

import contextlib
import json
import statistics
import tempfile
import time
from pathlib import Path

from railhead.replay import replay
from railhead.world import World


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


def measure_step(savegame: Path, prefix: tuple[str, ...] = (), binary: Path | None = None, days: int = 91,
                 n: int = 3) -> dict:
    """ms per game day for a loaded state, with the server launched under ``prefix`` (e.g. ``taskpolicy -l 0``)
    and/or a different engine ``binary``. The dedicated driver polls stdin with a 1 us ``select`` every tick,
    so how macOS schedules that timer (QoS, timer coalescing) shows up directly in the step cost."""
    from railhead import server as srv

    orig = srv.OpenTTDServer.command
    def command(self):
        cmd = orig(self)
        return [*prefix, str(binary) if binary else cmd[0], *cmd[1:]]

    srv.OpenTTDServer.command = command
    try:
        with tempfile.TemporaryDirectory(prefix="rh17m-step-") as tmp:
            w = World.load(Path(tmp) / "w", savegame)
            try:
                times = [_timed(lambda: w.step(days=days))[1] for _ in range(n)]
            finally:
                w.close()
    finally:
        srv.OpenTTDServer.command = orig
    return {"prefix": " ".join(prefix), "binary": str(binary or "default"),
            "ms_per_day": round(1e3 * statistics.median(times) / days, 2)}
