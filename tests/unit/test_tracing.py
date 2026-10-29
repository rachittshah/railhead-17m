import time

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rh17m import labels, tracing


@pytest.fixture
def spans(monkeypatch):
    monkeypatch.delenv("RH17M_TRACE", raising=False)
    exp = InMemorySpanExporter()
    tracing.init(exporter=exp)
    yield exp
    tracing.reset()


def _by_name(exp):
    out = {}
    for s in exp.get_finished_spans():
        out.setdefault(s.name, []).append(s)
    return out


class FakeAdmin:
    def rcon(self, cmd):
        return ["Reseeded"] if cmd.startswith("reseed") else []


class FakeWorld:
    """Just enough of railhead's World for labels._rollout."""

    def __init__(self):
        self.admin = FakeAdmin()
        self.money = 1000

    @classmethod
    def load(cls, workdir, savegame):
        return cls()

    def step(self, days):
        self.money += days

    def call(self, method, **args):
        if method == "company":
            return {"money": self.money, "loan": 0}
        return []

    def close(self):
        pass


def test_disabled_is_a_noop(monkeypatch):
    monkeypatch.setenv("RH17M_TRACE", "0")
    tracing.reset()
    assert tracing.init() is False
    with tracing.span("x", a=1) as s:
        s.set_attribute("b", 2)
    assert tracing.traceparent() == ""
    tracing.flush()
    tracing.reset()


def test_unreachable_endpoint_is_a_fast_noop(monkeypatch):
    monkeypatch.delenv("RH17M_TRACE", raising=False)
    monkeypatch.setenv("RH17M_OTLP_ENDPOINT", "http://127.0.0.1:9")  # discard port: nothing listens
    tracing.reset()
    t0 = time.perf_counter()
    assert tracing.init() is False
    with tracing.span("x"):
        pass
    assert time.perf_counter() - t0 < 1.0
    tracing.reset()


def test_spans_nest_and_record_errors(spans):
    with tracing.span("outer", state_id="s1", skip=None):
        with tracing.span("inner", n=3):
            tracing.annotate(ok=True)
        with pytest.raises(ValueError), tracing.span("bad"):
            raise ValueError("boom")
    by = _by_name(spans)
    outer, inner, bad = by["outer"][0], by["inner"][0], by["bad"][0]
    assert inner.parent.span_id == outer.context.span_id
    assert dict(outer.attributes) == {"state_id": "s1"}
    assert inner.attributes["ok"] is True
    assert bad.status.status_code.name == "ERROR"
    assert outer.resource.attributes["service.name"] == "rh17m"


def test_wrap_traces_classmethods_and_methods_once(spans):
    class W:
        @classmethod
        def load(cls, workdir, savegame):
            return cls()

        def step(self, days):
            return days

    tracing.wrap(W, "load", "world.load", lambda cls, workdir, savegame: {"save": str(savegame)})
    tracing.wrap(W, "load", "world.load")  # idempotent
    tracing.wrap(W, "step", "world.step", lambda self, days: {"days": days})
    w = W.load("w", "a.sav")
    assert isinstance(w, W) and w.step(days=91) == 91
    by = _by_name(spans)
    assert len(by["world.load"]) == 1 and by["world.load"][0].attributes["save"] == "a.sav"
    assert by["world.step"][0].attributes["days"] == 91


def test_traceparent_carries_parent_into_a_job(spans):
    with tracing.span("rh17m.label") as run:
        tp = tracing.traceparent()
    assert tp.startswith("00-")
    with tracing.attach(tp), tracing.span("label.job"):
        pass
    job = _by_name(spans)["label.job"][0]
    assert job.parent.span_id == run.get_span_context().span_id
    assert job.context.trace_id == run.get_span_context().trace_id


def test_label_job_spans_hang_under_the_run(spans, monkeypatch, tmp_path):
    monkeypatch.setattr(labels, "label_state", lambda root, sid, cfg: {"options": ["hold"], "timing": {"total_s": 1}})
    with tracing.span("rh17m.label") as run:
        tp = tracing.traceparent()
    r = labels._job((str(tmp_path), "st1", str(tmp_path), {}, tp))
    assert r["ok"] is True
    job = _by_name(spans)["label.job"][0]
    assert job.parent.span_id == run.get_span_context().span_id
    assert job.attributes["state_id"] == "st1" and job.attributes["ok"] is True
    # the old 4-tuple job args still work (a pool respawned mid-run by an older parent)
    (tmp_path / "st1.json").unlink()
    (tmp_path / ".claims" / "st1").unlink()
    assert labels._job((str(tmp_path), "st1", str(tmp_path), {}))["ok"] is True


def test_rollout_spans_chunks_and_continuation(spans, monkeypatch, tmp_path):
    monkeypatch.setattr(labels, "World", FakeWorld)
    tracing.wrap(FakeWorld, "step", "world.step", lambda self, days: {"days": days})
    cfg = labels.LabelConfig(chunks=4, chunk_days=91, continue_every=182)
    out = labels._rollout(tmp_path / "x.sav", tmp_path / "w", 7, cfg, key="hold")
    assert len(out["nw"]) == 4
    by = _by_name(spans)
    roll = by["label.rollout"][0]
    assert roll.attributes["option"] == "hold" and roll.attributes["seed"] == 7
    assert len(by["world.step"]) == 4
    assert all(s.parent.span_id == roll.context.span_id for s in by["world.step"])
    assert len(by["label.continue_expand"]) == 1  # after chunk 2; not after the last chunk


def test_shell_span_parents_cli_commands(spans, monkeypatch, tmp_path):
    from rh17m import cli

    env = tracing.begin()
    monkeypatch.setenv("TRACEPARENT", env["TRACEPARENT"])
    monkeypatch.setenv("RH17M_TRACE_START_NS", env["RH17M_TRACE_START_NS"])
    assert cli.main(["play-report", "--root", str(tmp_path)]) == 0
    cli.main(["trace", "end", "climb_pass", "n=12"])
    by = _by_name(spans)
    cmd, outer = by["rh17m.play-report"][0], by["climb_pass"][0]
    tid, sid = env["TRACEPARENT"].split("-")[1:3]
    assert outer.parent is None and outer.attributes["n"] == "12"
    assert (format(outer.context.trace_id, "032x"), format(outer.context.span_id, "016x")) == (tid, sid)
    assert cmd.parent.span_id == outer.context.span_id
    assert outer.start_time <= cmd.start_time


def test_unreachable_endpoint_is_probed_again_later(monkeypatch):
    monkeypatch.delenv("RH17M_TRACE", raising=False)
    monkeypatch.setenv("RH17M_OTLP_ENDPOINT", "http://127.0.0.1:9")
    tracing.reset()
    assert tracing.init() is False
    probes = []
    monkeypatch.setattr(tracing, "reachable", lambda url: probes.append(url) or False)
    with tracing.span("x"):
        pass
    assert probes == []  # not before REPROBE_S
    monkeypatch.setitem(tracing._state, "retry_at", time.monotonic() - 1)
    with tracing.span("x"):
        pass
    assert len(probes) == 1
    tracing.reset()


class CountedWorld:
    """A World whose ``call`` takes ~2 ms, with railhead's refusal behaviour for one method."""

    def call(self, method, **args):
        time.sleep(0.002)
        if method == "refused":
            raise RuntimeError("ERR_NO_SITE")
        return []


@pytest.fixture
def counted_world():
    tracing.count_calls(CountedWorld)
    return CountedWorld()


def test_bridge_calls_attach_to_spans_without_double_counting(spans, counted_world):
    w = counted_world
    w.call("before")  # outside every span: nobody's
    with tracing.span("outer", bridge=True):
        w.call("a")
        with tracing.span("inner", bridge=True):
            for _ in range(3):
                w.call("b")
        w.call("c")
    with tracing.span("sibling", bridge=True):
        w.call("d")
    with tracing.span("plain"):
        w.call("e")
    by = _by_name(spans)
    outer, inner, sibling = (by[n][0].attributes for n in ("outer", "inner", "sibling"))
    assert (outer["bridge_calls"], inner["bridge_calls"], sibling["bridge_calls"]) == (5, 3, 1)
    assert inner["bridge_ms"] >= 6 and outer["bridge_ms"] >= inner["bridge_ms"] + 4
    assert "bridge_calls" not in by["plain"][0].attributes  # opt-in per span


def test_bridge_calls_count_refusals_and_survive_exceptions(spans, counted_world):
    with pytest.raises(RuntimeError), tracing.span("build", bridge=True):
        counted_world.call("ok")
        counted_world.call("refused")  # a refused build step still costs a round trip
    assert _by_name(spans)["build"][0].attributes["bridge_calls"] == 2


def test_bridge_totals_work_without_a_tracer(monkeypatch, counted_world):
    # The label record carries bridge_calls even with tracing off or no viewer listening.
    monkeypatch.setenv("RH17M_TRACE", "0")
    tracing.reset()
    b0 = tracing.bridge_totals()
    with tracing.span("x", bridge=True) as s:
        counted_world.call("a")
        counted_world.call("b")
    assert s is tracing.NOOP
    calls, ms = tracing.bridge_since(b0)
    assert calls == 2 and ms >= 4


def test_install_counts_world_calls_even_when_tracing_is_off(monkeypatch):
    from railhead.world import World

    monkeypatch.setenv("RH17M_TRACE", "0")
    tracing.install()
    tracing.install()  # idempotent: one round trip is counted once
    assert getattr(World.call, "_rh17m_counted", False)

    class FakeBridge:
        def call(self, method, timeout=30.0, **args):
            return method

    w = World.__new__(World)
    w.bridge = FakeBridge()
    b0 = tracing.bridge_totals()
    assert w.call("state") == "state"
    assert tracing.bridge_since(b0)[0] == 1
