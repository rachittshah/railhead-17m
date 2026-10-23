"""Deterministic OpenTTD-flavoured route-decision inputs for parity tests and benchmarks.

These are not simulator states; they only reproduce the shape of real inputs (compact JSON of towns,
industries and routes; route candidates plus "hold") at a requested token length.
"""

from __future__ import annotations

import json
import random

TOWNS = ["Fenwood", "Ashby Cross", "Little Brindle", "Kettering Vale", "Bürgerstadt", "Ñandú Falls",
         "Oxley Heath", "Port Marrow", "Sundby", "Quillon", "Haverholt", "Eastcombe"]
INDUSTRIES = [("coal_mine", "coal"), ("forest", "wood"), ("oil_wells", "oil"), ("farm", "grain"),
              ("iron_ore_mine", "iron_ore"), ("power_station", None), ("sawmill", None), ("factory", None)]
INSTRUCTIONS = "Pick the single best action for the company this month. Prefer routes that repay their cost soonest."


def _town(rng, index):
    return {"id": index, "name": rng.choice(TOWNS) + ("" if index < len(TOWNS) else f" {index}"),
            "pop": rng.randint(200, 9000), "xy": [rng.randint(0, 255), rng.randint(0, 255)],
            "rating": rng.randint(0, 100)}


def _industry(rng, index):
    kind, cargo = rng.choice(INDUSTRIES)
    item = {"id": index, "type": kind, "xy": [rng.randint(0, 255), rng.randint(0, 255)]}
    if cargo:
        item["prod"] = {cargo: rng.randint(24, 220)}
        item["transported"] = round(rng.random(), 2)
    return item


def _route(rng, index):
    return {"id": f"r{index}", "kind": rng.choice(["rail", "road", "ship"]), "from": rng.randint(0, 60),
            "to": rng.randint(0, 60), "cargo": rng.choice(["coal", "wood", "oil", "grain", "passengers"]),
            "vehicles": rng.randint(1, 6), "profit_ly": rng.randint(-8000, 60000), "age_days": rng.randint(30, 2000)}


def state_json(tok, target_tokens: int, seed: int = 0) -> str:
    """Compact JSON state whose token count is the first to reach ``target_tokens``."""
    rng = random.Random(seed)
    state = {"date": f"19{rng.randint(50, 99)}-{rng.randint(1, 12):02d}-01", "money": rng.randint(-50_000, 900_000),
             "loan": rng.choice([0, 100_000, 300_000]), "company": {"name": "Railhead Transport", "rating": rng.randint(20, 90)},
             "towns": [], "industries": [], "routes": []}
    makers = [("towns", _town), ("industries", _industry), ("routes", _route)]
    while True:
        text = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        if len(tok.ids([text])[0]) >= target_tokens:
            return text
        name, make = makers[rng.randrange(3)]
        state[name].append(make(rng, len(state[name])))


def options(count: int, seed: int = 0, *, structured: bool = False, detail: int = 0) -> list[tuple[str, object]]:
    """``count`` options: route candidates then "hold". ``structured`` uses dict descriptions."""
    rng = random.Random(1000 + seed)
    out = []
    for i in range(count - 1):
        src, dst = rng.sample(TOWNS, 2)
        cargo = rng.choice(["coal", "wood", "oil", "grain", "passengers", "mail"])
        distance, profit, capex = rng.randint(12, 140), rng.randint(-5, 90) * 1000, rng.randint(20, 400) * 1000
        if structured:
            description = {"action": "build_route", "cargo": cargo, "from": src, "to": dst, "distance": distance,
                           "est_annual_profit": profit, "capex": capex, "mode": rng.choice(["rail", "road"])}
        else:
            description = (f"Build a {cargo} route from {src} to {dst}: {distance} tiles, "
                           f"estimated profit {profit} per year, costs {capex}.")
            description += " Note: " + " ".join(rng.choice(TOWNS) for _ in range(detail)) if detail else ""
        out.append((f"route_{i + 1}", description))
    out.append(("hold", "Do nothing this month and keep the cash."))
    return out
