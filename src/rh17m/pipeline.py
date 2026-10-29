"""Pipelined GameScript calls: send many requests, then collect the replies.

``World.call`` is one request, one reply, one game frame: ~1 ms uncontended and 2-4x that when the machine
is busy, and a route build makes ~1,270 of them (the builder dry-runs every anchor around both towns). The
GameScript drains every request waiting at its next tick (``HandleEvents`` loops ``IsEventWaiting``), so
sending a batch back to back costs a few frames, not one per request. Replies carry the request id, so
they are matched by id; a failed request comes back as the ``BridgeError`` in its slot.

``install`` adds ``World.call_many`` (unless railhead grows one). Each request still counts as a round trip
in the call counter, so spans show what a batch replaced.
"""

from __future__ import annotations

import os
import queue
import time
from typing import Any

from railhead.bridge import BridgeError
from railhead.world import World

from rh17m import tracing

WINDOW = 256  # requests in flight at once: bounds the admin socket and the GameScript's event queue


def call_many(world: Any, calls: list[tuple[str, dict]], timeout: float = 60.0, window: int = WINDOW) -> list[Any]:
    """Run ``calls`` ((method, args) pairs) as one pipeline. Returns one result per call, in order; a call the
    engine refused is its ``BridgeError`` (returned, not raised). Raises ``TimeoutError`` / ``ConnectionError``
    for transport failures, like ``Bridge.call``."""
    n = len(calls)
    if not n:
        return []
    bridge = world.bridge
    admin = bridge.admin
    ids = [bridge._new_id() for _ in calls]
    slot = {rid: i for i, rid in enumerate(ids)}
    out: list[Any] = [None] * n
    answered = [False] * n
    pages: dict[int, list] = {}
    sent = done = 0
    t0 = time.perf_counter()
    deadline = time.monotonic() + timeout
    try:
        while done < n:
            while sent < n and sent - done < window:
                method, args = calls[sent]
                admin.send_gs({"id": ids[sent], "m": method, "a": args})
                sent += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{n - done} of {n} pipelined calls unanswered after {timeout}s")
            try:
                msg = admin.recv_gs(timeout=min(remaining, 0.5))
            except queue.Empty:
                if not admin.alive:
                    raise ConnectionError("admin connection closed") from None
                continue
            if not isinstance(msg, dict):
                continue
            kind = msg.get("t")
            if kind == "ev":
                bridge.events.append(msg)
                continue
            i = slot.get(msg.get("id"))
            if i is None or answered[i]:
                continue  # a stale reply from an earlier timed-out call
            if kind == "p":
                pages.setdefault(i, []).extend(msg.get("rows", []))
            elif kind == "r":
                if not msg.get("ok"):
                    out[i] = BridgeError(msg.get("code", "ERR_UNKNOWN"), msg.get("msg", ""), msg.get("data"))
                elif msg.get("rows"):
                    out[i] = pages.pop(i, [])
                else:
                    out[i] = msg.get("v")
                answered[i] = True
                done += 1
    finally:
        tracing.record_calls(sent, time.perf_counter() - t0)
    return out


def install() -> None:
    """Give railhead's World a ``call_many`` (idempotent; a railhead-native one wins). RH17M_PIPELINE=0 leaves
    World without it, so builders fall back to one round trip per call (the equivalence check's other arm)."""
    if os.environ.get("RH17M_PIPELINE", "1") == "0" or hasattr(World, "call_many"):
        return

    def call_many_method(self, calls: list[tuple[str, dict]], **kw: Any) -> list[Any]:
        return call_many(self, calls, **kw)

    call_many_method._rh17m_pipeline = True
    World.call_many = call_many_method
