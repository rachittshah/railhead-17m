"""Observations without labels: load a state, render the wake-up text and the options, no rollouts.

A state costs about a second (load + snapshot) instead of minutes (forked rollouts), so every harvested state
gets an observation. Unlabelled observations are what a teacher model soft-labels for distillation into the
17M student. Records go to ``data/obs/<state_id>.json`` in the same shape as a label record minus the
outcomes, so ``dataset.observation`` builds the model input from either.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import tempfile
from pathlib import Path

from railhead.task import TaskSpec
from railhead.world import World

from rh17m.labels import LabelConfig, candidate_extras, querier_for
from rh17m.states import find_task, list_states, load_meta


def observe_state(root: Path, state_id: str, max_air: int = LabelConfig.max_air) -> dict:
    from rhs1.candidates import enumerate_routes
    from rhs1.snapshot import aggregate_routes, render_state, take_snapshot

    meta = load_meta(root, state_id)
    task = TaskSpec.load(find_task(meta.task_id))
    with tempfile.TemporaryDirectory(prefix="rh17m-observe-") as tmp:
        world = World.load(Path(tmp) / "w", Path(root) / f"{state_id}.sav")
        try:
            snap = take_snapshot(querier_for(world, task, meta.end_date))
            routes = aggregate_routes(snap)
            labels = {c["id"]: c["label"] for c in world.call("cargos")}
            cands = [c for c in enumerate_routes(snap, routes, max_per_mode=max_air, cargo_labels=labels)
                     if c.mode == "air"][:max_air]
            extras = candidate_extras(snap, cands, labels)
            money0 = int(world.call("company").get("money") or 0)
            return {"state": dataclasses.asdict(meta), "money0": money0,
                    "rendered_state": render_state(snap, routes, cands),
                    "candidates": [{**dataclasses.asdict(c), **extras.get(c.candidate_id, {})} for c in cands],
                    "options": [c.candidate_id for c in cands] + ["hold"]}
        finally:
            world.close()


def _job(args: tuple) -> dict:
    root, state_id, out_dir = args
    dest = Path(out_dir) / f"{state_id}.json"
    if dest.exists():
        return {"state_id": state_id, "skipped": True}
    try:
        rec = observe_state(Path(root), state_id)
        tmp = dest.with_name(f".{dest.name}.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.rename(dest)
        return {"state_id": state_id, "ok": True, "candidates": len(rec["candidates"])}
    except Exception as e:  # noqa: BLE001 - one bad state must not stop the sweep
        return {"state_id": state_id, "ok": False, "error": repr(e)}


def observe_all(root: Path, out_dir: Path, workers: int = 4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in list_states(root) if not (out_dir / f"{s}.json").exists()]
    with mp.get_context("spawn").Pool(workers) as pool:
        yield from pool.imap_unordered(_job, [(str(root), s, str(out_dir)) for s in todo])
