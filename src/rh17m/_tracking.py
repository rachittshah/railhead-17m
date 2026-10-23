"""Vendored copy of decision_mlx.tracking (decision-mlx PR #7); used only while the local decision-mlx checkout
predates that PR. Optional experiment tracking with Trackio (local-first, wandb-compatible API).

    with Tracker("decision-mlx", name="kev-eval", config={"model": model}) as tracker:
        tracker.log({"train/loss": loss}, step=step)
        tracker.log_summary(result)  # every numeric leaf, flattened to a/b/c keys

Install with ``uv pip install 'decision-mlx[tracking]'`` and open the dashboard with
``trackio show --project decision-mlx``. Without Trackio, or with ``enabled=False``, every call is a no-op.
Tracking never stops a run: a failing Trackio call is reported once on stderr and later calls are skipped.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any, Self


def flatten(value: Any, prefix: str = "", sep: str = "/") -> dict[str, float]:
    """Numeric leaves of nested mappings and sequences as ``{"a/b/0": x}``; booleans become 0/1, non-finite
    numbers and everything else are dropped."""
    out: dict[str, float] = {}
    if isinstance(value, bool):
        out[prefix] = float(value)
    elif isinstance(value, (int, float)):
        if math.isfinite(value):
            out[prefix] = float(value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}{sep}{k}" if prefix else str(k), sep))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flatten(v, f"{prefix}{sep}{i}" if prefix else str(i), sep))
    return out


class Tracker:
    """A Trackio run, or a no-op when tracking is disabled or Trackio is not installed."""

    def __init__(self, project: str, *, name: str | None = None, group: str | None = None,
                 config: Mapping[str, Any] | None = None, enabled: bool = True, **init_kwargs: Any):
        self._trackio = None
        self._failed = False
        if not enabled:
            return
        try:
            import trackio
        except ImportError:
            print("decision-mlx: tracking requested but trackio is not installed "
                  "(uv pip install 'decision-mlx[tracking]')", file=sys.stderr)
            return
        init_kwargs.setdefault("embed", False)
        if not self._guard(lambda: trackio.init(project=project, name=name, group=group,
                                                config=dict(config or {}), **init_kwargs)):
            return
        self._trackio = trackio

    @property
    def active(self) -> bool:
        return self._trackio is not None and not self._failed

    def _guard(self, call) -> bool:
        try:
            call()
            return True
        except Exception as error:  # noqa: BLE001 - tracking must never stop the run
            if not self._failed:
                print(f"decision-mlx: tracking disabled after an error: {error!r}", file=sys.stderr)
            self._failed = True
            return False

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self.active:
            numeric = flatten(metrics)
            if numeric:
                self._guard(lambda: self._trackio.log(numeric, step=step))

    def log_summary(self, result: Mapping[str, Any], prefix: str = "summary", step: int | None = None) -> None:
        self.log({prefix: result} if prefix else result, step=step)

    def finish(self) -> None:
        if self.active:
            self._guard(self._trackio.finish)
        self._trackio = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()
