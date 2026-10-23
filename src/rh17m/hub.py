"""The published Railhead dataset (``rachittshah/railhead`` on the Hub) as a task and teacher-state source.

``fetch_tasks`` writes ``task.json`` + ``start.sav`` for every task of the chosen families into
``data/tasks/`` (``states.find_task`` looks there first), so self-play can run on every Hub map without a
Railhead checkout. ``teacher_games`` yields the model games of the ``trajectories`` config (the Arena games;
the scripted oracle is skipped) for ``states.harvest_teacher``.

The 512x512 10-year arena maps (arena-*-02xx) are not on the Hub yet; they come from the local checkout.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from rh17m.states import HUB_TASKS_ROOT

REPO = "rachittshah/railhead"
TASK_FIELDS = ("task_id", "family", "horizon_days", "runoff_days", "limits", "objective", "scenario", "split")


def _as_obj(v):
    return json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v


def fetch_tasks(families: tuple[str, ...] = ("tycoon",), out: Path = HUB_TASKS_ROOT, repo: str = REPO) -> list[str]:
    """Materialise the Hub tasks of ``families``. Existing task directories are kept. Returns task ids."""
    from datasets import load_dataset

    ds = load_dataset(repo, "tasks")
    ids = []
    for split in ds:
        for row in ds[split]:
            if row["family"] not in families:
                continue
            d = Path(out) / row["task_id"]
            ids.append(row["task_id"])
            if (d / "start.sav").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            spec = {k: _as_obj(row[k]) for k in TASK_FIELDS}
            (d / "task.json").write_text(json.dumps(spec, indent=1))
            tmp = d / ".start.sav.tmp"
            tmp.write_bytes(bytes(row["start_sav"]))
            tmp.rename(d / "start.sav")
    return sorted(ids)


def teacher_games(repo: str = REPO, skip: tuple[str, ...] = ("oracle",)) -> Iterator[tuple[str, str, str]]:
    """(task_id, policy, actions_jsonl text) for every non-oracle trajectory on the Hub."""
    from datasets import load_dataset

    ds = load_dataset(repo, "trajectories")
    for split in ds:
        for row in ds[split]:
            if row["policy"] not in skip and row["actions_jsonl"]:
                yield row["task_id"], row["policy"], row["actions_jsonl"]


def harvest_hub_teachers(root: Path, repo: str = REPO) -> Iterator[dict]:
    """Replay every Hub model game into decision states (``states.harvest_teacher``)."""
    from railhead.task import TaskSpec
    from railhead.world import World

    from rh17m.states import find_task, harvest_teacher

    for task_id, policy, text in teacher_games(repo):
        try:
            task_dir = find_task(task_id)
        except FileNotFoundError:
            yield {"task": task_id, "model": policy, "error": "task not available"}
            continue
        with tempfile.TemporaryDirectory(prefix="rh17m-hubgame-") as tmp:
            log_path = Path(tmp) / "actions.jsonl"
            log_path.write_text(text)
            # The horizon ends horizon_days after the start state's date (the arena runner's end_date).
            with World.load(Path(tmp) / "probe", task_dir / "start.sav") as w:
                end_date = int(w.state()["date"]) + TaskSpec.load(task_dir).horizon_days
            got = harvest_teacher(task_id, policy, log_path, root, end_date=end_date)
        yield {"task": task_id, "model": policy, "states": len(got)}
