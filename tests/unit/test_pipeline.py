import queue

import pytest

from rh17m import pipeline, tracing


class FakeAdmin:
    """The admin client's GameScript side. ``serve`` turns the requests sent so far into replies, in the
    order given by ``order`` (default: arrival order), so tests can interleave them as the engine may."""

    alive = True

    def __init__(self, handler, order=None):
        self.handler = handler
        self.order = order
        self.sent: list[dict] = []
        self.outstanding = 0
        self.max_outstanding = 0
        self.inbox: queue.Queue = queue.Queue()

    def send_gs(self, obj):
        self.sent.append(obj)
        self.outstanding += 1
        self.max_outstanding = max(self.max_outstanding, self.outstanding)
        # the engine answers everything that is waiting at its next tick: here, once a batch is queued
        for msg in self.handler(obj):
            self.inbox.put(msg)

    def recv_gs(self, timeout=None):
        msg = self.inbox.get(timeout=timeout)
        if isinstance(msg, dict) and msg.get("t") in ("r",):
            self.outstanding -= 1
        return msg


class FakeBridge:
    def __init__(self, admin):
        self.admin = admin
        self._next_id = 1
        self.events = []

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i


class FakeWorld:
    def __init__(self, handler):
        self.bridge = FakeBridge(FakeAdmin(handler))


def echo(req):
    a = req["a"]
    if a.get("fail"):
        return [{"t": "r", "id": req["id"], "ok": False, "code": "ERR_AREA_NOT_CLEAR", "msg": "no"}]
    return [{"t": "r", "id": req["id"], "ok": True, "v": {"tile": a["tile"]}}]


def test_results_follow_request_order_and_errors_are_values():
    w = FakeWorld(echo)
    calls = [("air_airport", {"tile": t, "fail": t == 2}) for t in range(5)]
    out = pipeline.call_many(w, calls)
    assert [o.get("tile") if isinstance(o, dict) else type(o).__name__ for o in out] == [0, 1, "BridgeError", 3, 4]
    assert out[2].code == "ERR_AREA_NOT_CLEAR"
    assert [r["a"]["tile"] for r in w.bridge.admin.sent] == list(range(5))


def test_replies_may_arrive_out_of_order_and_stale_ids_are_ignored():
    w = FakeWorld(lambda req: [])
    admin = w.bridge.admin
    calls = [("m", {"tile": t}) for t in range(3)]
    for rid, t in [(3, 2), (999, 7), (1, 0), (2, 1)]:  # 999: an earlier timed-out call
        admin.inbox.put({"t": "r", "id": rid, "ok": True, "v": {"tile": t}})
    out = pipeline.call_many(w, calls)
    assert [o["tile"] for o in out] == [0, 1, 2]


def test_list_replies_are_assembled_from_partial_pages():
    def pages(req):
        rid = req["id"]
        return [{"t": "p", "id": rid, "rows": [1, 2]}, {"t": "p", "id": rid, "rows": [3]},
                {"t": "r", "id": rid, "ok": True, "rows": True}]

    out = pipeline.call_many(FakeWorld(pages), [("towns", {})])
    assert out == [[1, 2, 3]]


def test_events_are_kept_not_dropped():
    def with_event(req):
        return [{"t": "ev", "kind": "vehicle_crashed"}, *echo(req)]

    w = FakeWorld(with_event)
    pipeline.call_many(w, [("m", {"tile": 0})])
    assert w.bridge.events == [{"t": "ev", "kind": "vehicle_crashed"}]


def test_window_bounds_requests_in_flight():
    w = FakeWorld(echo)
    out = pipeline.call_many(w, [("m", {"tile": t}) for t in range(10)], window=4)
    assert [o["tile"] for o in out] == list(range(10))
    assert w.bridge.admin.max_outstanding <= 4
    assert len(w.bridge.admin.sent) == 10


def test_timeout_when_the_engine_stops_answering():
    w = FakeWorld(lambda req: [])
    with pytest.raises(TimeoutError):
        pipeline.call_many(w, [("m", {})], timeout=0.2)


def test_empty_batch_costs_nothing():
    assert pipeline.call_many(FakeWorld(echo), []) == []


def test_every_call_is_counted_as_a_round_trip():
    w = FakeWorld(echo)
    before = tracing.bridge_totals()
    pipeline.call_many(w, [("m", {"tile": t}) for t in range(7)])
    calls, ms = tracing.bridge_since(before)
    assert calls == 7
    assert ms >= 0


def test_install_adds_call_many_and_can_be_switched_off(monkeypatch):
    from railhead.world import World

    monkeypatch.delattr(World, "call_many", raising=False)
    monkeypatch.setenv("RH17M_PIPELINE", "0")
    pipeline.install()
    assert not hasattr(World, "call_many")
    monkeypatch.delenv("RH17M_PIPELINE")
    pipeline.install()
    assert hasattr(World, "call_many")
    pipeline.install()  # idempotent
    assert World.call_many._rh17m_pipeline
