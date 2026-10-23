"""Expansion options: add planes to an air route the company already runs.

Opening a new route is one way to invest; the other, and the one strong players use most, is adding planes to
routes that already pay. An expansion candidate is one of the company's air routes (the ``max_routes`` most
profitable last year) with a template plane (one of its vehicles), the hangar nearest to the route's stops,
and the route's load factor (passengers aboard / seats, averaged over its planes): full planes on a paying
route are the signal that more planes would pay too.

``apply_option`` performs any option of the wake-up question, a new route (``labels.build_affordable``) or
an expansion (``expand_affordable``), so labels, play and self-play execute options identically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from railhead.actions import to_gs
from railhead.bridge import BridgeError

EXPAND_V = "expand-v1"
PREFIX = "expand-"
PLANES_LADDER = (2, 1)


@dataclass(frozen=True)
class Expansion:
    key: str
    route_id: str
    vehicles: int
    profit_last_year: int
    profit_this_year: int
    length_tiles: int
    load_factor: float
    template: int
    hangar: int
    plane: str | None
    price: int


def expansion_candidates(snap: Any, routes: list, depots: list[dict], max_routes: int = 3) -> list[Expansion]:
    map_x = int(snap.info.get("map_x") or 1)
    vehicles = {int(v["id"]): v for v in snap.vehicles}
    hangars = [d for d in depots if d.get("vtype") == "air" and int(d.get("tile", -1)) >= 0]
    prices = {int(e["id"]): int(e.get("price") or 0) for e in snap.engines.get("air", [])}
    out = []
    for r in sorted((r for r in routes if r.mode == "air" and r.vehicle_ids),
                    key=lambda r: (-r.profit_last_year, r.route_id))[:max_routes]:
        if not hangars:
            break
        stops = [(t % map_x, t // map_x) for t in r.stops]
        hangar = min(hangars, key=lambda d: min(abs(int(d["x"]) - x) + abs(int(d["y"]) - y) for x, y in stops))
        vs = [vehicles[i] for i in r.vehicle_ids if i in vehicles]
        loads = [int(v.get("load") or 0) / int(v["capacity"]) for v in vs if int(v.get("capacity") or 0) > 0]
        template = vs[0] if vs else {"id": r.vehicle_ids[0]}
        out.append(Expansion(
            key=f"{PREFIX}{r.route_id}", route_id=r.route_id, vehicles=r.vehicles,
            profit_last_year=r.profit_last_year, profit_this_year=r.profit_this_year, length_tiles=r.length_tiles,
            load_factor=round(sum(loads) / len(loads), 3) if loads else 0.0, template=int(template["id"]),
            hangar=int(hangar["tile"]), plane=template.get("engine_name"),
            price=prices.get(int(template.get("engine", -1)), 0)))
    return out


def expand(world: Any, e: Expansion, planes: int) -> dict:
    """Clone the template plane ``planes`` times into the hangar with shared orders and start each copy."""
    cost, added, failed = 0, [], None
    for i in range(planes):
        try:
            method, args = to_gs("vehicle.clone", {"depot_tile": e.hangar, "vehicle": e.template, "share_orders": True})
            reply = world.call(method, **args)
            cost += int(reply.get("cost", 0))
            vid = int(reply.get("vehicle", -1))
            if vid < 0:
                failed = f"clone_{i}:ERR_NO_VEHICLE"
                break
            added.append(vid)
            method, args = to_gs("vehicle.start", {"vehicle": vid})
            world.call(method, **args)
        except BridgeError as exc:
            failed = f"clone_{i}:{exc.code}"
            break
    return {"ok": len(added) == planes, "cost": cost, "failed_step": failed, "planes": len(added), "vehicles": added}


def expand_affordable(world: Any, e: Expansion) -> dict:
    """Two planes, or one when two are refused for cash before anything was bought."""
    res = {}
    for planes in PLANES_LADDER:
        res = expand(world, e, planes)
        if res["ok"] or res["planes"] or not (res["failed_step"] or "").endswith("ERR_NOT_ENOUGH_CASH"):
            break
    return res


def describe_expansion(e: dict, money0: int) -> str:
    left = int(money0) - 2 * int(e.get("price") or 0)
    cash = (f"£{round(left / 1000)}k cash would remain after two planes" if left >= 0
            else f"two planes cost £{round(-left / 1000)}k more than the cash")
    return (f"add planes to existing air route {e['route_id']}: {e['vehicles']} planes ({e.get('plane')}), "
            f"{e['length_tiles']} tiles, profit last year £{round(e['profit_last_year'] / 1000)}k, "
            f"this year £{round(e['profit_this_year'] / 1000)}k, planes {round(100 * e['load_factor'])}% full; "
            f"{cash}")


def apply_option(world: Any, key: str, cands: list, exps: list[Expansion]) -> dict:
    """Perform option ``key`` (a route candidate id or an expansion key) and report what happened."""
    if key.startswith(PREFIX):
        e = next(x for x in exps if x.key == key)
        r = expand_affordable(world, e)
        return {**r, "kind": "expand", "steps": bool(r["planes"])}
    from rh17m.labels import build_affordable

    c = next(c for c in cands if c.candidate_id == key)
    br, used = build_affordable(world, c.a["id"], c.b["id"])
    return {"ok": br.ok, "cost": br.cost, "failed_step": br.failed_step, "airport_type": br.airport_type,
            "ladder": list(used), "kind": "route", "steps": bool(br.steps)}


def as_dicts(exps: list[Expansion]) -> list[dict]:
    return [asdict(e) for e in exps]
