"""The trained 17M model as a route policy for closed-loop play (``play.make_policy("model", dir)``)."""

from __future__ import annotations

import json
from pathlib import Path

from rh17m import bekko_mlx as B
from rh17m.train import LAYOUTS, temper


class ModelPolicy:
    def __init__(self, model_dir: Path, layout: str = LAYOUTS[0]):
        self.model_dir = Path(model_dir)
        self.model = B.load(str(model_dir), revision=None)
        meta = Path(model_dir) / "rh17m.json"
        self.temperature = json.loads(meta.read_text())["temperature"] if meta.exists() else 1.0
        self.layout = layout

    def __call__(self, obs: dict) -> dict[str, float]:
        p = B.choice_probs(self.model, None, obs["state_json"], obs["instructions"], obs["options"],
                           layout=self.layout)
        return temper(p, self.temperature)
