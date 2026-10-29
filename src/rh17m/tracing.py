"""OpenTelemetry tracing: a waterfall of every label job, climb pass and game, exported over OTLP/HTTP.

Every process (CLI, spawned pool workers) builds its own TracerProvider lazily, on its first span, and sends
batches to ``RH17M_OTLP_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default: the Phoenix viewer that
scripts/launch.sh runs, http://127.0.0.1:6006). Tracing must never break or slow a run: with
``RH17M_TRACE=0``, without the SDK, or when nothing listens on the endpoint (probed once per process), every
span is a no-op. Pool jobs carry the parent's ``traceparent()`` in their args and ``attach()`` it, so a
label job hangs under its labeller run; a CLI command attaches the ``TRACEPARENT`` env var (set by
scripts/climb_loop.sh around a climb pass, see ``begin``/``end``).

Simulator time is traced where it is spent: ``install()`` wraps railhead's World.load/fork/save/step/close
and OpenTTDServer.start (railhead itself stays untouched), at package import like sched.py.

GameScript round trips are counted separately from the spans (``count_calls``): every ``World.call`` adds one
call and its wall time to a context-local running total, whether or not anything is traced. A span opened
with ``bridge=True`` reports the difference over its lifetime as ``bridge_calls`` / ``bridge_ms``, and
``bridge_totals`` / ``bridge_since`` give the same to code that records the numbers itself (label records).
"""

from __future__ import annotations

import functools
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlparse

DEFAULT_ENDPOINT = "http://127.0.0.1:6006"
PROJECT = "rh17m"

_lock = threading.Lock()
_state: dict = {"pid": None, "provider": None, "ids": None, "retry_at": None}
REPROBE_S = 60.0  # an endpoint that was down is probed again this often (the viewer may start later)


class _NoSpan:
    def set_attribute(self, *a, **k) -> None:
        pass

    def set_attributes(self, *a, **k) -> None:
        pass

    def add_event(self, *a, **k) -> None:
        pass

    def record_exception(self, *a, **k) -> None:
        pass

    def end(self, *a, **k) -> None:
        pass


NOOP = _NoSpan()

_bridge: ContextVar[list] = ContextVar("rh17m_bridge")  # [round trips, seconds] since this context began


def endpoint() -> str:
    base = os.environ.get("RH17M_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or DEFAULT_ENDPOINT
    return base.rstrip("/") + "/v1/traces"


def reachable(url: str, timeout: float = 0.25) -> bool:
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", u.port or 80), timeout=timeout):
            return True
    except OSError:
        return False


def _resource():
    import multiprocessing as mp

    from opentelemetry.sdk.resources import Resource

    return Resource.create({"service.name": "rh17m", "openinference.project.name": PROJECT,
                            "rh17m.stage": os.environ.get("RH17M_STAGE", "rh17m"), "process.pid": os.getpid(),
                            "rh17m.worker": mp.current_process().name})


def init(exporter=None) -> bool:
    """Build this process's provider (once per pid). ``exporter`` (tests) replaces the OTLP exporter with a
    synchronous one and forces a rebuild. Returns whether spans are recorded."""
    pid = os.getpid()
    if exporter is None and _state["pid"] == pid:
        return _state["provider"] is not None
    with _lock:
        if exporter is None and _state["pid"] == pid:
            return _state["provider"] is not None
        _state.update(pid=pid, provider=None, retry_at=None)
        if os.environ.get("RH17M_TRACE", "1") == "0":
            return False
        try:
            from opentelemetry.sdk.trace import RandomIdGenerator, TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        except ImportError:
            return False
        if exporter is None:
            url = endpoint()
            if not reachable(url):
                _state["retry_at"] = time.monotonic() + REPROBE_S
                return False
            import logging

            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            # A viewer that goes away mid-run costs dropped spans, not a log line per batch.
            logging.getLogger("opentelemetry.exporter.otlp.proto.http").setLevel(logging.ERROR)
            processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=url, timeout=2), max_queue_size=8192)
        else:
            processor = SimpleSpanProcessor(exporter)

        class Ids(RandomIdGenerator):  # ``end`` emits a span whose ids ``begin`` handed out earlier
            forced: tuple[int, int] | None = None

            def generate_trace_id(self) -> int:
                return self.forced[0] if self.forced else super().generate_trace_id()

            def generate_span_id(self) -> int:
                return self.forced[1] if self.forced else super().generate_span_id()

        ids = Ids()
        provider = TracerProvider(resource=_resource(), id_generator=ids)
        provider.add_span_processor(processor)
        _state.update(provider=provider, ids=ids)
        return True


def reset() -> None:
    """Drop this process's provider (flushing it); the next span re-initialises."""
    p = _state["provider"]
    _state.update(pid=None, provider=None, ids=None, retry_at=None)
    if p is not None:
        p.shutdown()


def _tracer():
    if _state["pid"] != os.getpid() and not init():
        return None
    if _state["retry_at"] is not None and time.monotonic() > _state["retry_at"]:
        _state["pid"] = None
        if not init():
            return None
    p = _state["provider"]
    return p.get_tracer("rh17m") if p is not None else None


def _clean(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if not isinstance(v, (bool, int, float, str)):
            v = json.dumps(v, default=str) if isinstance(v, (dict, list, tuple)) else str(v)
        out[k] = v
    return out


def _running() -> list:
    try:
        return _bridge.get()
    except LookupError:
        acc = [0, 0.0]
        _bridge.set(acc)
        return acc


def bridge_totals() -> tuple[int, float]:
    """GameScript round trips and their wall seconds so far in this context (a running total: take
    ``bridge_since`` of an earlier reading to measure a stretch of code)."""
    calls, seconds = _running()
    return calls, seconds


def bridge_since(before: tuple[int, float]) -> tuple[int, float]:
    """Round trips and milliseconds made since ``before`` (a ``bridge_totals`` reading)."""
    calls, seconds = bridge_totals()
    return calls - before[0], 1e3 * (seconds - before[1])


def count_calls(owner: type, attr: str = "call") -> None:
    """Count ``owner.attr`` (railhead's ``World.call``, the one door every builder, snapshot and step goes
    through) as one GameScript round trip each, refusals included. Idempotent."""
    fn = owner.__dict__[attr]
    if getattr(fn, "_rh17m_counted", False):
        return

    @functools.wraps(fn)
    def counted(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            acc = _running()
            acc[0] += 1
            acc[1] += time.perf_counter() - t0

    counted._rh17m_counted = True
    setattr(owner, attr, counted)


@contextmanager
def span(name: str, *, bridge: bool = False, **attrs):
    """A child of the current span; exceptions are recorded and mark it as an error. ``bridge=True`` sets
    ``bridge_calls`` and ``bridge_ms`` on it: the round trips made inside, nested spans' included."""
    t = _tracer()
    if t is None:
        yield NOOP
        return
    with t.start_as_current_span(name, attributes=_clean(attrs)) as s:
        before = bridge_totals() if bridge else None
        try:
            yield s
        finally:
            if before is not None:
                calls, ms = bridge_since(before)
                s.set_attributes({"bridge_calls": calls, "bridge_ms": round(ms, 1)})


def traced(name: str):
    """Decorator: every call of the function is a span ``name``."""
    def deco(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return inner
    return deco


@contextmanager
def current(span_):
    """Make a ``start``-ed span the parent of the spans opened inside (it is not ended on exit)."""
    if span_ is NOOP or _tracer() is None:
        yield span_
        return
    from opentelemetry import trace

    with trace.use_span(span_, end_on_exit=False, record_exception=False, set_status_on_exception=False):
        yield span_


def start(name: str, **attrs):
    """A span that is not made current (end it with ``.end()``): for loops not worth re-indenting."""
    t = _tracer()
    return NOOP if t is None else t.start_span(name, attributes=_clean(attrs))


def annotate(span_=None, **attrs) -> None:
    """Set attributes on ``span_`` (default: the current span)."""
    if _tracer() is None:
        return
    if span_ is None:
        from opentelemetry import trace

        span_ = trace.get_current_span()
    span_.set_attributes(_clean(attrs))


def traceparent() -> str:
    """The W3C traceparent of the current span ("" when not tracing), to hand to another process."""
    if _tracer() is None:
        return ""
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent", "")


@contextmanager
def attach(tp: str | None):
    """Make the span named by traceparent ``tp`` the current parent (no-op for "" or when not tracing)."""
    if not tp or _tracer() is None:
        yield
        return
    from opentelemetry import context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    token = context.attach(TraceContextTextMapPropagator().extract({"traceparent": tp}))
    try:
        yield
    finally:
        context.detach(token)


def flush(timeout_ms: int = 2000) -> None:
    """Export what is buffered. Pool workers call it after every job: a pool's terminate() skips atexit."""
    p = _state["provider"] if _state["pid"] == os.getpid() else None
    if p is not None:
        p.force_flush(timeout_ms)


def begin() -> dict[str, str]:
    """Ids and start time of a span a shell script will close later with ``end`` (no export happens here)."""
    return {"TRACEPARENT": f"00-{os.urandom(16).hex()}-{os.urandom(8).hex()}-01",
            "RH17M_TRACE_START_NS": str(time.time_ns())}


def end(name: str, tp: str, start_ns: int, **attrs) -> bool:
    """Emit the span ``begin`` opened (same ids, so the children already sent hang under it)."""
    t = _tracer()
    parts = (tp or "").split("-")
    if t is None or len(parts) != 4:
        return False
    from opentelemetry.context import Context

    ids = _state["ids"]
    ids.forced = (int(parts[1], 16), int(parts[2], 16))
    try:
        s = t.start_span(name, context=Context(), attributes=_clean(attrs), start_time=int(start_ns))
    finally:
        ids.forced = None
    s.end()
    flush()
    return True


def wrap(owner: type, attr: str, name: str, attrs=None) -> None:
    """Trace ``owner.attr`` (a method or classmethod) as span ``name``; ``attrs(*args, **kwargs)`` gives its
    attributes. Idempotent."""
    raw = owner.__dict__[attr]
    is_cm = isinstance(raw, classmethod)
    fn = raw.__func__ if is_cm else raw
    if getattr(fn, "_rh17m_traced", False):
        return

    @functools.wraps(fn)
    def traced(*args, **kwargs):
        t = _tracer()
        if t is None:
            return fn(*args, **kwargs)
        try:
            a = _clean(attrs(*args, **kwargs)) if attrs else {}
        except Exception:  # noqa: BLE001 - a bad attribute must not cost the call
            a = {}
        with t.start_as_current_span(name, attributes=a):
            return fn(*args, **kwargs)

    traced._rh17m_traced = True
    setattr(owner, attr, classmethod(traced) if is_cm else traced)


def _name(p) -> str:
    return os.path.basename(str(p))


def install() -> None:
    """Count railhead's GameScript round trips and trace its simulator calls (server start, load, fork, save,
    step, close) in this process. Counting is two ``perf_counter`` reads on a ~1 ms call, so it stays on
    with ``RH17M_TRACE=0``: label records report it either way."""
    from railhead import server as srv
    from railhead import world as wld

    W = wld.World
    count_calls(W)
    if os.environ.get("RH17M_TRACE", "1") == "0":
        return
    wrap(srv.OpenTTDServer, "start", "server.start", lambda self: {"workdir": _name(self.workdir)})
    wrap(W, "load", "world.load", lambda cls, workdir, savegame, *a, **k: {"save": _name(savegame),
                                                                           "workdir": _name(workdir)})
    wrap(W, "fork", "world.fork", lambda self, workdir, reseed=None: {"workdir": _name(workdir)})
    wrap(W, "save", "world.save", lambda self, dest: {"dest": _name(dest)})
    wrap(W, "step", "world.step", lambda self, days=None, ticks=None, **k: {"days": days, "ticks": ticks})
    wrap(W, "close", "world.close")
