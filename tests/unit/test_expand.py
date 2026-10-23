from types import SimpleNamespace

from rh17m.expand import PREFIX, describe_expansion, expansion_candidates


def _route(rid, profit, vehicles, stops):
    return SimpleNamespace(route_id=rid, mode="air", profit_last_year=profit, profit_this_year=profit // 2,
                           vehicle_ids=tuple(vehicles), vehicles=len(vehicles), stops=stops, length_tiles=300)


def test_expansions_rank_routes_by_profit_and_pick_the_nearest_hangar():
    snap = SimpleNamespace(
        info={"map_x": 100},
        vehicles=[{"id": 1, "load": 90, "capacity": 100, "engine": 7, "engine_name": "Darwin"},
                  {"id": 2, "load": 10, "capacity": 100, "engine": 7, "engine_name": "Darwin"},
                  {"id": 3, "load": 50, "capacity": 100, "engine": 7, "engine_name": "Darwin"}],
        engines={"air": [{"id": 7, "price": 40_000}]})
    routes = [_route("low", 10_000, [3], (5 * 100 + 5,)), _route("high", 90_000, [1, 2], (50 * 100 + 50,)),
              SimpleNamespace(route_id="bus", mode="road", profit_last_year=10**9, vehicle_ids=(9,), stops=())]
    depots = [{"vtype": "air", "tile": 1, "x": 6, "y": 6}, {"vtype": "air", "tile": 2, "x": 49, "y": 51},
              {"vtype": "road", "tile": 3, "x": 50, "y": 50}]
    exps = expansion_candidates(snap, routes, depots, max_routes=2)
    assert [e.route_id for e in exps] == ["high", "low"]
    assert exps[0].key == PREFIX + "high" and exps[0].hangar == 2 and exps[0].load_factor == 0.5
    assert exps[0].price == 40_000 and exps[1].hangar == 1
    text = describe_expansion(exps[0].__dict__, 100_000)
    assert "add planes to existing air route high" in text and "50% full" in text and "£20k cash" in text
