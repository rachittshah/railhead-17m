"""The trained 17M model as a route policy for closed-loop play (``play.make_policy("model", dir)``)."""

from __future__ import annotations

import json
from pathlib import Path

from rh17m import bekko_mlx as B
from rh17m.train import LAYOUTS, temper


class ModelPolicy:
    """Plays in its own process (closed loop, self-play). By default on the CPU: a 17M forward pass is cheap
    there, and the GPU stays with training (two processes driving Metal at once can take the trainer down with
    'Command buffer execution failed: ... victim of GPU error/recovery')."""

    def __init__(self, model_dir: Path, layout: str = LAYOUTS[0], device: str | None = None):
        import os

        import mlx.core as mx

        device = device or os.environ.get("RH17M_POLICY_DEVICE", "cpu")
        mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        from rh17m.train import limit_mlx_cache

        limit_mlx_cache()
        # A comma-separated model_dir is an ensemble (e.g. a config's seed models): probabilities are averaged.
        dirs = [Path(d) for d in str(model_dir).split(",") if d]
        self.model_dir = dirs[0]
        self.members = []
        for d in dirs:
            meta = d / "rh17m.json"
            t = json.loads(meta.read_text())["temperature"] if meta.exists() else 1.0
            self.members.append((B.load(str(d), revision=None), t))
        self.model, self.temperature = self.members[0]
        self.layout = layout

    def __call__(self, obs: dict) -> dict[str, float]:
        avg: dict[str, float] = {}
        for model, t in self.members:
            p = temper(B.choice_probs(model, None, obs["state_json"], obs["instructions"], obs["options"],
                                      layout=self.layout), t)
            for k, v in p.items():
                avg[k] = avg.get(k, 0.0) + v / len(self.members)
        return avg
