from rh17m import labels


def test_timing_carries_the_server_starts_of_the_job():
    t = labels._timing(build_s=18.84, total_s=88.123, gs_calls=1269, gs_ms=1500.04, server_starts=3)
    assert t == {"build_s": 18.84, "total_s": 88.12, "bridge_calls": 1269, "bridge_ms": 1500.0, "server_starts": 3}
