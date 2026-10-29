import signal
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from railhead.server import ServerConfig
from railhead.world import World

from rh17m import pool


class FakeEngine:
    """What one OpenTTD process does with the console commands the pool uses, including its two traps:
    ``load`` only schedules the switch (it happens before the next command is handled, so a request sent
    right after ``load`` may still reach the old game), and the console's savegame listing is cached until
    ``rm`` invalidates it (a file copied in later is "cannot be found")."""

    def __init__(self, save_dir: Path, game: str) -> None:
        self.save_dir = save_dir
        self.game = game  # contents of the savegame the running game came from
        self.speed = 100
        self.listing: set[str] | None = None
        self.pending: Path | None = None
        self.loads = 0
        self.fail_load = False
        self.log: list[str] = []

    def rcon(self, cmd: str) -> list[str]:
        if self.pending is not None:  # the switch runs at the top of the next main-loop iteration
            self.game, self.speed, self.pending = self.pending.read_text(), 100, None
            self.loads += 1
        self.log.append(cmd)
        word, _, arg = cmd.partition(" ")
        if word == "gamespeed":
            if arg:
                self.speed = int(arg)
            return [f"Game speed: {self.speed}"]
        if word == "load":
            if self.listing is None:
                self.listing = {p.name for p in self.save_dir.glob("*.sav")}
            if arg not in self.listing:
                return [f"'{arg}' cannot be found."]
            if not self.fail_load:
                self.pending = self.save_dir / arg
            return []
        if word == "rm":
            (self.save_dir / arg).unlink(missing_ok=True)
            self.listing = None
            return []
        raise AssertionError(f"unexpected console command {cmd!r}")


class FakeAdmin:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.alive = True
        self.gs_queue: deque = deque()

    def rcon(self, cmd: str, timeout: float = 30.0) -> list[str]:
        if not self.alive:
            raise ConnectionError("admin connection closed")
        return self.engine.rcon(cmd)

    def close(self) -> None:
        self.alive = False


class FakeProc:
    def __init__(self, events: list, name: str) -> None:
        self.events, self.name = events, name
        self.stdin = SimpleNamespace(write=lambda b: events.append(("quit", name)), flush=lambda: None)

    def poll(self):
        return None


class FakeServer:
    made: ClassVar[list] = []
    events: ClassVar[list] = []

    def __init__(self, workdir, config) -> None:
        self.workdir, self.config = Path(workdir), config
        self.running, self.stops, self.admin_error = False, 0, None
        self.engine: FakeEngine | None = None
        self.proc = FakeProc(FakeServer.events, self.workdir.name)
        FakeServer.made.append(self)

    @property
    def save_dir(self) -> Path:
        return self.workdir / "save"

    def start(self):
        self.save_dir.mkdir(parents=True)
        self.engine = FakeEngine(self.save_dir, Path(self.config.savegame).read_text())
        self.running = True
        return self

    def admin(self) -> FakeAdmin:
        if self.admin_error:
            raise self.admin_error
        return FakeAdmin(self.engine)

    def stop(self) -> None:
        FakeServer.events.append(("stop", self.workdir.name))
        self.running = False
        self.stops += 1


class FakeWorld:
    def __init__(self, server, speed: int = 0) -> None:
        self.server = server
        self.admin = server.admin()
        self.bridge = SimpleNamespace(events=deque())
        self.game_at_open = server.engine.game  # which game this connection first talked to
        self.admin.rcon(f"gamespeed {speed}")

    @classmethod
    def load(cls, workdir, savegame, config=None, speed=0):
        cfg = config or ServerConfig()
        cfg.savegame = Path(savegame).resolve()
        return cls(FakeServer(workdir, cfg).start(), speed=speed)

    def close(self) -> None:
        self.admin.close()
        self.server.stop()


@pytest.fixture(autouse=True)
def _reset():
    FakeServer.made = []
    FakeServer.events = []


@pytest.fixture
def saves(tmp_path):
    out = {}
    for name in "abc":
        out[name] = tmp_path / f"{name}.sav"
        out[name].write_text(name)
    return out


@pytest.fixture
def wp(tmp_path):
    p = pool.WorldPool(FakeWorld, FakeWorld.load, base=tmp_path / "pool", recycle=50, fence_timeout=0.3)
    yield p
    p.shutdown()


def test_first_acquire_starts_a_server_in_a_pool_owned_dir(wp, saves, tmp_path):
    caller_dir = tmp_path / "job" / "parent"
    w = wp.acquire(caller_dir, saves["a"])
    assert len(FakeServer.made) == 1 and w.game_at_open == "a"
    # the caller's temp dir goes away with its job; the server must not live in it
    assert not str(w.server.workdir).startswith(str(caller_dir.parent))
    assert str(w.server.workdir).startswith(str(tmp_path / "pool"))
    assert w._rh17m_pool is wp


def test_released_server_is_reloaded_not_restarted(wp, saves, tmp_path):
    w1 = wp.acquire(tmp_path / "w1", saves["a"])
    server = w1.server
    wp.release(w1)
    assert server.running and server.stops == 0
    w2 = wp.acquire(tmp_path / "w2", saves["b"])
    assert w2.server is server and len(FakeServer.made) == 1
    assert server.engine.loads == 1 and w2.game_at_open == "b"
    assert w2 is not w1  # a new World on a new admin connection, not the old object revived


def test_reload_sequence_and_cleanup(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    wp.release(w)
    log = w.server.engine.log
    w = wp.acquire(tmp_path / "w", saves["b"])
    wp.release(w)
    assert log[-1] == "gamespeed 0"  # the new World's own set_speed, after the reload
    first = log[1:-1]
    # speed sentinel (so the engine's reset to 100 proves the load happened), load, fence, rm
    assert first[0] == "gamespeed 0"
    assert first[1].startswith("load ") and first[1].endswith(".sav")
    name = first[1].split()[1]
    assert first[-1] == f"rm {name}"
    assert all(c == "gamespeed" for c in first[2:-1]) and len(first) >= 4
    assert not list(w.server.save_dir.glob("*.sav"))  # the copy is removed after the load
    # the console caches its listing; rm invalidates it, so a second reload of a new file still finds it
    wp.acquire(tmp_path / "w", saves["c"])
    assert w.server.engine.loads == 2


def test_new_world_talks_to_the_new_game_even_though_load_returns_early(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    wp.release(w)
    w2 = wp.acquire(tmp_path / "w", saves["b"])
    # the engine only switches games at its next iteration: the World must not exist before that
    assert w2.game_at_open == "b"


def test_speed_is_reapplied_after_the_engine_resets_it(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    assert w.server.engine.speed == 0
    wp.release(w)
    w = wp.acquire(tmp_path / "w", saves["b"])
    assert w.server.engine.speed == 0  # the load put it back to 100
    wp.release(w)
    w = wp.acquire(tmp_path / "w", saves["b"], speed=50)
    assert w.server.engine.speed == 50


def test_the_next_owner_inherits_no_events_or_replies(wp, saves, tmp_path):
    w1 = wp.acquire(tmp_path / "w", saves["a"])
    w1.bridge.events.append({"t": "ev", "kind": "vehicle_crashed"})
    w1.admin.gs_queue.append({"t": "r", "id": 99, "ok": True})  # a reply nobody collected
    wp.release(w1)
    w2 = wp.acquire(tmp_path / "w", saves["b"])
    assert not w2.bridge.events and not w2.admin.gs_queue


def test_a_released_world_is_dead_and_releasing_twice_is_harmless(wp, saves, tmp_path):
    w1 = wp.acquire(tmp_path / "w", saves["a"])
    wp.release(w1)
    assert not w1.admin.alive
    with pytest.raises(ConnectionError):
        w1.admin.rcon("reseed 1")
    wp.release(w1)  # a stray second close() must not put the server in the pool twice
    a = wp.acquire(tmp_path / "w", saves["b"])
    b = wp.acquire(tmp_path / "w", saves["b"])
    assert a.server is not b.server and len(FakeServer.made) == 2


def test_grows_on_demand_and_keeps_idle_servers(wp, saves, tmp_path):
    a = wp.acquire(tmp_path / "a", saves["a"])
    b = wp.acquire(tmp_path / "b", saves["a"])
    assert a.server is not b.server and len(FakeServer.made) == 2
    wp.release(a)
    wp.release(b)
    c = wp.acquire(tmp_path / "c", saves["b"])
    d = wp.acquire(tmp_path / "d", saves["b"])
    assert {c.server, d.server} == {a.server, b.server} and len(FakeServer.made) == 2
    e = wp.acquire(tmp_path / "e", saves["b"])
    assert len(FakeServer.made) == 3 and e.server not in (a.server, b.server)
    assert wp.stats == {"starts": 3, "reloads": 2, "fallbacks": 0, "recycled": 0}


def test_recycles_a_server_after_n_loads(tmp_path, saves):
    p = pool.WorldPool(FakeWorld, FakeWorld.load, base=tmp_path / "pool", recycle=3, fence_timeout=0.3)
    try:
        servers = []
        for _ in range(4):
            w = p.acquire(tmp_path / "w", saves["a"])
            servers.append(w.server)
            p.release(w)
        # uses 1-3 are one process; its third release retires it, the fourth use starts a new one
        assert servers[0] is servers[1] is servers[2] and servers[3] is not servers[0]
        assert servers[0].stops == 1 and not servers[0].running
        assert p.stats["recycled"] == 1 and p.stats["starts"] == 2
    finally:
        p.shutdown()


def test_a_failed_reload_falls_back_to_a_fresh_server(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    bad = w.server
    wp.release(w)
    bad.engine.fail_load = True  # the engine never switches games: the fence times out
    w2 = wp.acquire(tmp_path / "w", saves["b"])
    assert w2.server is not bad and w2.game_at_open == "b"
    assert bad.stops == 1 and not bad.running
    assert wp.stats["fallbacks"] == 1
    assert not bad.workdir.exists()  # the dropped server's directory is gone with it


def test_a_dead_connection_during_reload_falls_back_too(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    bad = w.server
    wp.release(w)
    bad.admin_error = ConnectionError("admin join refused")
    w2 = wp.acquire(tmp_path / "w", saves["b"])
    assert w2.server is not bad and bad.stops == 1 and wp.stats["fallbacks"] == 1


def test_a_server_that_died_while_idle_is_replaced(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    dead = w.server
    wp.release(w)
    dead.running = False
    w2 = wp.acquire(tmp_path / "w", saves["b"])
    assert w2.server is not dead and w2.game_at_open == "b"


def test_a_missing_savegame_is_the_callers_error_and_costs_no_server(wp, saves, tmp_path):
    w = wp.acquire(tmp_path / "w", saves["a"])
    server = w.server
    wp.release(w)
    with pytest.raises(FileNotFoundError):
        wp.acquire(tmp_path / "w", tmp_path / "nope.sav")
    assert server.running and server.stops == 0 and wp.stats["fallbacks"] == 0
    assert wp.acquire(tmp_path / "w", saves["b"]).server is server  # still pooled


def test_servers_with_different_settings_are_not_interchangeable(wp, saves, tmp_path):
    fast = ServerConfig(settings={"misc": {"savegame_format": "none"}})
    w = wp.acquire(tmp_path / "w", saves["a"], fast)
    wp.release(w)
    other = wp.acquire(tmp_path / "w", saves["a"])  # default settings: needs its own server
    assert other.server is not w.server and len(FakeServer.made) == 2
    again = wp.acquire(tmp_path / "w", saves["b"], ServerConfig(settings={"misc": {"savegame_format": "none"}}))
    assert again.server is w.server


def test_a_failed_first_start_leaves_nothing_behind(wp, saves, tmp_path):
    with pytest.raises(FileNotFoundError):
        wp.acquire(tmp_path / "w", tmp_path / "nope.sav")
    assert wp.stats["starts"] == 0
    assert wp.acquire(tmp_path / "w", saves["a"]).game_at_open == "a"


def test_shutdown_stops_every_server_idle_or_in_use_and_removes_the_dirs(tmp_path, saves):
    p = pool.WorldPool(FakeWorld, FakeWorld.load, base=tmp_path / "pool", recycle=50, fence_timeout=0.3)
    busy = p.acquire(tmp_path / "w", saves["a"])
    idle = p.acquire(tmp_path / "w", saves["a"])
    p.release(idle)
    p.shutdown()
    assert busy.server.stops == 1 and idle.server.stops == 1
    assert not (tmp_path / "pool").exists()
    p.shutdown()  # idempotent
    assert busy.server.stops == 1
    p.release(busy)  # the owner closing its World after shutdown does not resurrect anything
    assert busy.server.stops == 1


def test_shutdown_asks_every_engine_to_quit_before_waiting_for_any(tmp_path, saves):
    p = pool.WorldPool(FakeWorld, FakeWorld.load, base=tmp_path / "pool", recycle=50, fence_timeout=0.3)
    a = p.acquire(tmp_path / "w", saves["a"])
    b = p.acquire(tmp_path / "w", saves["a"])
    p.release(b)
    FakeServer.events.clear()
    p.shutdown()
    kinds = [k for k, _ in FakeServer.events]
    assert kinds.index("stop") > max(i for i, k in enumerate(kinds) if k == "quit")  # all quit first, then reaped
    assert {n for k, n in FakeServer.events if k == "quit"} == {a.server.workdir.name, b.server.workdir.name}


def test_default_is_off_and_install_is_idempotent_and_restorable(monkeypatch):
    monkeypatch.delenv("RH17M_POOL", raising=False)
    was = pool.installed()
    pool.uninstall()
    try:
        load, close = World.__dict__["load"], World.__dict__["close"]
        assert pool.install() is False and not pool.installed()
        assert World.__dict__["load"] is load and World.__dict__["close"] is close
        monkeypatch.setenv("RH17M_POOL", "1")
        monkeypatch.setattr(pool.atexit, "register", lambda fn: None)
        monkeypatch.setattr(pool.signal, "signal", lambda *a: signal.SIG_DFL)
        assert pool.install() is True and pool.installed()
        patched = World.__dict__["load"]
        assert patched is not load
        assert pool.install() is True and World.__dict__["load"] is patched
        pool.uninstall()
        assert World.__dict__["load"] is load and World.__dict__["close"] is close
    finally:
        pool.uninstall()
        if was:
            pool.install()


def test_installed_load_and_close_go_through_the_pool(monkeypatch, tmp_path):
    seen = []

    class StubPool:
        def __init__(self, *a, **k):
            pass

        def acquire(self, workdir, savegame, config=None, speed=0):
            seen.append(("acquire", workdir, savegame, speed))
            return SimpleNamespace(_rh17m_pool=self)

        def release(self, world):
            seen.append(("release", world))
            return True

        def shutdown(self):
            seen.append("shutdown")

    monkeypatch.setenv("RH17M_POOL", "1")
    monkeypatch.setattr(pool, "WorldPool", StubPool)
    monkeypatch.setattr(pool.atexit, "register", lambda fn: None)
    monkeypatch.setattr(pool.signal, "signal", lambda *a: signal.SIG_DFL)
    was = pool.installed()
    pool.uninstall()
    try:
        pool.install()
        w = World.load(tmp_path / "w", tmp_path / "s.sav")
        assert seen == [("acquire", tmp_path / "w", tmp_path / "s.sav", 0)]
        World.close(w)  # a pooled World is handed back, not stopped
        assert seen[-1] == ("release", w)
    finally:
        pool.uninstall()
        if was:
            pool.install()


def test_a_world_the_pool_does_not_own_closes_as_railhead_does(monkeypatch):
    calls = []
    monkeypatch.setenv("RH17M_POOL", "1")
    monkeypatch.setattr(pool.atexit, "register", lambda fn: None)
    monkeypatch.setattr(pool.signal, "signal", lambda *a: signal.SIG_DFL)
    was = pool.installed()
    pool.uninstall()
    try:
        pool.install()
        w = World.__new__(World)
        w.admin = SimpleNamespace(close=lambda: calls.append("admin"))
        w.server = SimpleNamespace(stop=lambda: calls.append("stop"))
        w.close()
        assert calls == ["admin", "stop"]
    finally:
        pool.uninstall()
        if was:
            pool.install()


def test_install_registers_shutdown_at_exit_and_on_sigterm(monkeypatch):
    registered, handlers = [], {}
    monkeypatch.setenv("RH17M_POOL", "1")
    monkeypatch.setattr(pool.atexit, "register", registered.append)
    monkeypatch.setattr(pool.signal, "getsignal", lambda s: signal.SIG_DFL)
    monkeypatch.setattr(pool.signal, "signal", lambda s, h: handlers.setdefault(s, h))
    was = pool.installed()
    pool.uninstall()
    try:
        pool.install()
        assert registered and signal.SIGTERM in handlers and callable(handlers[signal.SIGTERM])
    finally:
        pool.uninstall()
        if was:
            pool.install()


def test_sigterm_stops_servers_then_dies_of_sigterm(monkeypatch):
    calls = []
    monkeypatch.setattr(pool.os, "kill", lambda pid, sig: calls.append(("kill", sig)))
    monkeypatch.setattr(pool.signal, "signal", lambda s, h: calls.append(("signal", s, h)))
    pool._sigterm_handler(lambda: calls.append("shutdown"), signal.SIG_DFL)(signal.SIGTERM, None)
    assert calls == ["shutdown", ("signal", signal.SIGTERM, signal.SIG_DFL), ("kill", signal.SIGTERM)]


def test_a_second_sigterm_during_shutdown_does_not_cut_it_short(monkeypatch):
    calls = []
    monkeypatch.setattr(pool.os, "kill", lambda pid, sig: calls.append(("kill", sig)))
    monkeypatch.setattr(pool.signal, "signal", lambda s, h: calls.append("signal"))
    handler = None

    def shutdown():
        calls.append("shutdown")
        if handler is not None and calls.count("shutdown") == 1:
            handler(signal.SIGTERM, None)  # the same request arriving again while the engines are stopping

    handler = pool._sigterm_handler(shutdown, signal.SIG_DFL)
    handler(signal.SIGTERM, None)
    assert calls == ["shutdown", "signal", ("kill", signal.SIGTERM)]  # one shutdown, one death, at the end


def test_sigterm_chains_to_a_handler_that_was_already_there(monkeypatch):
    calls = []
    prev = lambda s, f: calls.append("previous")
    pool._sigterm_handler(lambda: calls.append("shutdown"), prev)(signal.SIGTERM, None)
    assert calls == ["shutdown", "previous"]


def test_recycle_threshold_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("RH17M_POOL_RECYCLE", raising=False)
    assert pool.recycle_after() == 50
    monkeypatch.setenv("RH17M_POOL_RECYCLE", "7")
    assert pool.recycle_after() == 7
    monkeypatch.setenv("RH17M_POOL_RECYCLE", "soon")
    assert pool.recycle_after() == 50
