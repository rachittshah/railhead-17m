"""A persistent pool of OpenTTD servers: load a savegame into a live server instead of starting a process.

A label job starts ~20 servers (the parent, one fork per build option, one per rollout) at 1.3-1.45 s each
under load; the console ``load`` of a savegame into a running server costs ~0.4 s. ``WorldPool`` keeps the
servers of a worker process alive between uses: ``acquire`` hands out a ``railhead.world.World`` whose game
is the requested savegame, ``release`` (``World.close``) takes the server back. Opt-in with ``RH17M_POOL=1``
(``install`` swaps ``World.load`` / ``World.close`` at package import, like sched.py and pipeline.py), so
every caller of ``World.load`` in a process gets it.

What makes a reload as trustworthy as a fresh start:
- The caller never sees a recycled object. Every acquire builds a new ``World`` on a new admin connection
  (a new Bridge, queue and event list); ``release`` closes the old connection, so a late reply, an event or an
  abandoned request of the previous game cannot reach the next owner, and the previous owner's World is dead.
- ``load`` only *schedules* the switch: the engine performs it at its next main-loop iteration, so a request
  sent right after ``load`` may still be answered by the old game's GameScript. The engine resets
  ``gamespeed`` to 100 when it loads a game, so the pool sets 0 first and waits until it reads 100: that
  proves the switch ran. Only then does the new World connect (it re-applies the speed, as after any load).
- The console caches its savegame listing and cannot find a file copied in later. ``rm`` invalidates the
  cache (and deletes the copy), so every load is followed by one.
- Servers are interchangeable only when started alike (``signature``: settings, game script, engine binary);
  after ``recycle`` uses a server is stopped and replaced, which bounds anything the engine keeps across loads.
- Any failure to reload drops that server and starts a fresh one: the pool never fails a job, it only stops
  saving time. Servers live in a directory the pool owns (a caller's temp dir goes away with its job), and
  are stopped at exit and on SIGTERM (pool workers are terminated, which skips atexit).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from railhead.server import ServerConfig
from railhead.world import World

log = logging.getLogger(__name__)

RECYCLE = 50
FENCE_TIMEOUT_S = 30.0
_SPEED = re.compile(r"Game speed:\s*(\d+)")
_LOAD_REFUSED = re.compile(r"cannot be found|is not a savegame")


def recycle_after() -> int:
    """Loads per server before it is replaced (``RH17M_POOL_RECYCLE``, default 50)."""
    try:
        return max(1, int(os.environ.get("RH17M_POOL_RECYCLE", RECYCLE)))
    except ValueError:
        return RECYCLE


def signature(cfg: ServerConfig) -> str:
    """What a server's start is made of, apart from its ports, password and savegame: two servers with the
    same signature behave alike after a load. Part of it is the environment that picks the engine binary."""
    return json.dumps([cfg.merged_settings(), cfg.game_script, cfg.game_script_settings, cfg.debug, cfg.seed,
                       os.environ.get("RAILHEAD_OPENTTD", ""), os.environ.get("RAILHEAD_SCRIPTS", "")],
                      sort_keys=True, default=str)


@dataclass(eq=False)
class _Slot:
    server: Any
    sig: str
    dir: Path | None
    uses: int = 0
    ctl: Any = None  # a long-lived admin connection (console only: it does not subscribe to the GameScript)


class WorldPool:
    """Long-lived servers for one process. A label job needs at most two live at once (the parent and one
    fork or rollout); the pool grows on demand and keeps what it has. ``fresh_load`` is railhead's own
    ``World.load`` (the unpooled start); ``world_cls`` builds the World around an already running server."""

    def __init__(self, world_cls, fresh_load, base: Path | None = None, recycle: int | None = None,
                 fence_timeout: float = FENCE_TIMEOUT_S) -> None:
        self.world_cls = world_cls
        self._fresh = fresh_load
        self.base = Path(base) if base else None
        self.recycle = recycle or recycle_after()
        self.fence_timeout = fence_timeout
        self.stats = {"starts": 0, "reloads": 0, "fallbacks": 0, "recycled": 0}
        self._lock = threading.RLock()  # re-entrant: the SIGTERM handler may run inside a pool call
        self._idle: list[_Slot] = []
        self._all: list[_Slot] = []
        self._seq = 0
        self._stopping = 0
        self._pid = os.getpid()

    @property
    def stopping(self) -> bool:
        return self._stopping > 0

    # ------------------------------------------------------------------ acquire / release
    def acquire(self, workdir: Path, savegame: Path, config: ServerConfig | None = None, speed: int = 0) -> World:
        """A World loaded from ``savegame`` (same contract as ``World.load``; ``workdir`` only names the
        unpooled fallback, pooled servers live in the pool's own directory)."""
        self._check_pid()
        cfg = config or ServerConfig()
        savegame = Path(savegame)
        if not savegame.is_file():
            return self._fresh(workdir, savegame, cfg, speed)  # the caller's error, raised as it always was
        sig = signature(cfg)
        slot = self._take(sig)
        if slot is not None:
            try:
                world = self._reload(slot, savegame, speed)
            except Exception as e:  # noqa: BLE001 - a pool failure must never fail the job
                log.warning("pool: reload failed (%r); replacing the server", e)
                self.stats["fallbacks"] += 1
                self._discard(slot)
            else:
                self.stats["reloads"] += 1
                return self._adopt(world, slot)
        return self._start(workdir, savegame, cfg, speed, sig)

    def release(self, world: World) -> bool:
        """Take ``world``'s server back (what ``World.close`` does for a pooled World). Returns whether the
        pool owns ``world``; closing twice is harmless."""
        if world.__dict__.get("_rh17m_pool") is not self:
            return False
        slot = world.__dict__.get("_rh17m_slot")
        if slot is None:
            return True
        world._rh17m_slot = None
        with contextlib.suppress(Exception):
            world.admin.close()  # the previous owner's connection dies with its World
        self._check_pid()
        with self._lock:
            if slot not in self._all:  # shut down while this World was out
                return True
            if not slot.server.running:
                self._discard(slot)
            elif slot.uses >= self.recycle:
                self.stats["recycled"] += 1
                self._discard(slot)
            else:
                self._idle.append(slot)
        return True

    def shutdown(self) -> None:
        """Stop every server (idle or lent out) and remove the pool's directory. Idempotent, and safe to
        interrupt: a server stays registered until it is stopped, so a second call (the SIGTERM handler landing
        in the middle of the exit-time call of a worker that was already leaving) finishes what the first
        started, and the process may die right after it. Every engine is asked to quit before the first is
        waited for, so answering a SIGTERM costs one engine's exit time, not the sum."""
        if self._pid != os.getpid():  # a forked child must not stop its parent's servers
            self._check_pid()
            return
        self._stopping += 1
        try:
            with self._lock:
                slots = list(self._all)
                base = self.base
            if slots:
                log.info("pool: stopping %d server(s)", len(slots))
            for slot in slots:
                _ask_quit(slot.server)
            for slot in slots:
                self._discard(slot, close_ctl=False)
            with self._lock:
                self._idle.clear()
                self.base = None
            if base is not None:
                shutil.rmtree(base, ignore_errors=True)
        finally:
            self._stopping -= 1

    # ------------------------------------------------------------------ internals
    def _check_pid(self) -> None:
        if self._pid != os.getpid():  # inherited through fork: those servers belong to the parent
            with self._lock:
                self._idle.clear()
                self._all.clear()
                self.base = None
                self._pid = os.getpid()

    def _take(self, sig: str) -> _Slot | None:
        with self._lock:
            for i in range(len(self._idle) - 1, -1, -1):  # the most recently used server first
                slot = self._idle[i]
                if slot.sig != sig:
                    continue
                del self._idle[i]
                if slot.server.running:
                    return slot
                self._discard(slot)  # died while idle
        return None

    def _new_dir(self) -> Path:
        with self._lock:
            if self.base is None:
                self.base = Path(tempfile.mkdtemp(prefix="rh17m-pool-"))
            self.base.mkdir(parents=True, exist_ok=True)
            self._seq += 1
            return self.base / f"s{self._seq}"

    def _start(self, workdir: Path, savegame: Path, cfg: ServerConfig, speed: int, sig: str) -> World:
        try:
            slot_dir = self._new_dir()
        except OSError as e:
            log.warning("pool: no directory for a pooled server (%r); starting an unpooled one", e)
            return self._fresh(workdir, savegame, cfg, speed)
        try:
            world = self._fresh(slot_dir, savegame, cfg, speed)
        except BaseException:
            shutil.rmtree(slot_dir, ignore_errors=True)
            raise
        slot = _Slot(world.server, sig, slot_dir)
        with self._lock:
            self._all.append(slot)
        self.stats["starts"] += 1
        return self._adopt(world, slot)

    def _adopt(self, world: World, slot: _Slot) -> World:
        slot.uses += 1
        world._rh17m_pool = self
        world._rh17m_slot = slot
        return world

    def _reload(self, slot: _Slot, savegame: Path, speed: int) -> World:
        server = slot.server
        name = f"rh_pool_{slot.uses}.sav"
        copy = server.save_dir / name
        try:
            shutil.copyfile(savegame, copy)
            ctl = self._control(slot)
            out = ctl.rcon("gamespeed 0")
            if not out or not out[0].startswith("Game speed"):
                raise RuntimeError(f"engine lacks the Railhead gamespeed patch: {out!r}")
            out = ctl.rcon(f"load {name}")
            if _LOAD_REFUSED.search(" ".join(out)):
                raise RuntimeError(f"load refused: {out!r}")
            self._await_switch(ctl)
            ctl.rcon(f"rm {name}")  # deletes the copy and drops the console's cached listing
        except BaseException:
            copy.unlink(missing_ok=True)
            raise
        return self.world_cls(server, speed=speed)

    def _control(self, slot: _Slot):
        if slot.ctl is None or not slot.ctl.alive:
            slot.ctl = slot.server.admin()
        return slot.ctl

    def _await_switch(self, ctl) -> None:
        """Wait until the engine reports gamespeed 100 (it was set to 0 before ``load``): the new game exists."""
        deadline = time.monotonic() + self.fence_timeout
        while True:
            out = ctl.rcon("gamespeed", timeout=max(1.0, deadline - time.monotonic()))
            m = _SPEED.search(" ".join(out))
            if m and int(m.group(1)) == 100:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"the engine did not switch games within {self.fence_timeout}s: {out!r}")
            time.sleep(0.005)

    def _discard(self, slot: _Slot, close_ctl: bool = True) -> None:
        """Stop ``slot``'s server and forget it (in that order: see ``shutdown``). Closing the control connection
        sends a packet under its send lock, so ``shutdown`` (which a signal may have interrupted mid-send) leaves
        it: the engine's exit closes the socket."""
        if slot.ctl is not None and close_ctl:
            with contextlib.suppress(Exception):
                slot.ctl.close()
        try:
            slot.server.stop()
        except Exception:
            log.warning("pool: could not stop a server", exc_info=True)
        if slot.dir is not None:
            shutil.rmtree(slot.dir, ignore_errors=True)
        with self._lock:
            if slot in self._all:
                self._all.remove(slot)
            if slot in self._idle:
                self._idle.remove(slot)


def _ask_quit(server) -> None:
    """Send ``quit`` on the engine's stdin without waiting (``stop`` does that, and kills a stuck engine)."""
    proc = getattr(server, "proc", None)
    if proc is not None and proc.poll() is None and proc.stdin:
        with contextlib.suppress(Exception):
            proc.stdin.write(b"quit\n")
            proc.stdin.flush()


# ---------------------------------------------------------------------- install
_STATE: dict[str, Any] = {}


def installed() -> bool:
    return "pool" in _STATE


def fresh_load(workdir: Path, savegame: Path, config: ServerConfig | None = None, speed: int = 0) -> World:
    """``World.load`` as railhead defines it (a new server process), whatever is installed."""
    raw = _STATE.get("load") or World.__dict__["load"]
    return raw.__func__(World, workdir, savegame, config, speed)


def _sigterm_handler(shutdown, previous, busy=None):
    """Stop the pooled servers, then let SIGTERM do what it would have done (the engines run in their own
    sessions, so killing the worker alone would orphan them). A SIGTERM that arrives while a shutdown is
    already running (``busy``: the exit-time call of a worker that was leaving anyway, which Pool.terminate's
    SIGTERM often hits) must not run a second one inside it: the interrupted code may hold a lock the new one
    needs, and killing the process afterwards would cut the first short. It is parked in ``handler.deferred``
    for ``_exit_hook`` to hand back when the first is done."""
    running: list = []
    deferred: list = []

    def handler(signum, frame):
        if running:  # the same request again while the engines are stopping: let the first one finish
            return
        if busy is not None and busy():
            deferred.append(signum)
            return
        running.append(True)
        try:
            shutdown()
        except BaseException:
            log.exception("pool: shutdown on SIGTERM failed")  # the process dies right after: say why
            raise
        finally:
            if callable(previous):
                running.clear()  # that handler decides whether the process lives; a later SIGTERM is a new request
                previous(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

    handler.deferred = deferred
    return handler


def _exit_hook(shutdown, handler):
    """The atexit callback: ``shutdown``, then the SIGTERM that arrived during it (if any) takes effect."""
    def run():
        try:
            shutdown()
        finally:
            if handler is not None and handler.deferred:
                handler.deferred.clear()
                os.kill(os.getpid(), signal.SIGTERM)
    return run


def install() -> bool:
    """Pool every ``World.load`` of this process when ``RH17M_POOL=1`` (idempotent; default off). Returns
    whether the pool is active."""
    if os.environ.get("RH17M_POOL", "0") != "1":
        return False
    if installed():
        return True
    orig_load, orig_close = World.__dict__["load"], World.__dict__["close"]
    wp = WorldPool(World, lambda *a, **k: orig_load.__func__(World, *a, **k))

    def load(cls, workdir, savegame, config=None, speed=0):
        return wp.acquire(workdir, savegame, config, speed)

    def close(self):
        owner = self.__dict__.get("_rh17m_pool")
        if owner is None:
            return orig_close(self)
        owner.release(self)

    World.load = classmethod(load)
    World.close = close
    handler = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)
        if previous is not signal.SIG_IGN:
            handler = _sigterm_handler(wp.shutdown, previous, busy=lambda: wp.stopping)
            signal.signal(signal.SIGTERM, handler)
            _STATE.update(handler=handler, previous=previous)
    hook = _exit_hook(wp.shutdown, handler)
    atexit.register(hook)
    _STATE.update(pool=wp, load=orig_load, close=orig_close, hook=hook)
    return True


def uninstall() -> None:
    """Undo ``install`` (tests): the original World.load / World.close, and the pool's servers stopped."""
    if not installed():
        return
    wp = _STATE["pool"]
    World.load, World.close = _STATE["load"], _STATE["close"]
    atexit.unregister(_STATE["hook"])
    handler = _STATE.get("handler")
    if handler is not None and signal.getsignal(signal.SIGTERM) is handler:
        signal.signal(signal.SIGTERM, _STATE["previous"])
    _STATE.clear()
    wp.shutdown()
