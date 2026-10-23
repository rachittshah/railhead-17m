"""Vendored copy of decision_mlx.tracking (decision-mlx PRs #7, #8, #9); used only while the local decision-mlx
checkout predates them. Optional experiment tracking with Trackio (local-first, wandb-compatible API).

    with Tracker("decision-mlx", name="kev-eval", config={"model": model}) as tracker:
        tracker.log({"train/loss": loss}, step=step)
        tracker.log_summary(result)  # every numeric leaf, flattened to a/b/c keys

Install with ``uv pip install 'decision-mlx[tracking]'`` and open the dashboard with
``trackio show --project decision-mlx``. Without Trackio, or with ``enabled=False``, every call is a no-op.
Tracking never stops a run: a failing Trackio call is reported once on stderr and later calls are skipped.

``system_interval`` (seconds) samples system metrics on a background thread into Trackio's System Metrics
tab: Apple GPU device/renderer/tiler utilisation and GPU memory (from IOKit's ``PerformanceStatistics`` via
``ioreg``, no sudo), MLX active/peak/cache memory, and host CPU/RAM when psutil is installed. Trackio's own
Apple monitor only records that a GPU exists, so this is what fills the GPU charts on a Mac.
"""

from __future__ import annotations

import math
import platform
import re
import subprocess
import sys
import threading
from collections.abc import Mapping
from typing import Any, Self

_PERF = re.compile(r'"PerformanceStatistics" = \{([^}]*)\}')
_PAIR = re.compile(r'"([^"]+)"=(\d+)')
_GPU_PERCENT = {"Device Utilization %": "gpu/device_utilization",
                "Renderer Utilization %": "gpu/renderer_utilization",
                "Tiler Utilization %": "gpu/tiler_utilization"}
_GPU_BYTES = {"In use system memory": "gpu/in_use_memory_gb", "Alloc system memory": "gpu/alloc_memory_gb"}
_GIB = 2**30


def parse_ioreg_performance(text: str) -> dict[str, float]:
    """GPU utilisation (%) and memory (GiB) from ``ioreg -c IOAccelerator`` output; {} when absent."""
    match = _PERF.search(text)
    if not match:
        return {}
    stats = {k: int(v) for k, v in _PAIR.findall(match.group(1))}
    out = {name: float(stats[k]) for k, name in _GPU_PERCENT.items() if k in stats}
    out.update({name: stats[k] / _GIB for k, name in _GPU_BYTES.items() if k in stats})
    return out


def apple_gpu_metrics() -> dict[str, float]:
    if platform.system() != "Darwin":
        return {}
    try:
        text = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                              capture_output=True, text=True, timeout=2, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_ioreg_performance(text)


def mlx_memory_metrics() -> dict[str, float]:
    try:
        import mlx.core as mx
    except ImportError:
        return {}
    api = mx if hasattr(mx, "get_active_memory") else getattr(mx, "metal", None)
    if api is None:
        return {}
    return {"mlx/active_memory_gb": api.get_active_memory() / _GIB,
            "mlx/peak_memory_gb": api.get_peak_memory() / _GIB,
            "mlx/cache_memory_gb": api.get_cache_memory() / _GIB}


def host_metrics() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {}
    vm = psutil.virtual_memory()
    return {"cpu/utilization": psutil.cpu_percent(interval=None), "memory/used_gb": vm.used / _GIB,
            "memory/percent": vm.percent, "memory/available_gb": vm.available / _GIB}


def system_metrics() -> dict[str, float]:
    """One sample of every available system metric (Apple GPU, MLX memory, host CPU/RAM)."""
    return {**apple_gpu_metrics(), **mlx_memory_metrics(), **host_metrics()}


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
                 config: Mapping[str, Any] | None = None, enabled: bool = True,
                 system_interval: float | None = None, **init_kwargs: Any):
        self._trackio = None
        self._failed = False
        self._sampler: threading.Thread | None = None
        self._run = None
        self._stop = threading.Event()
        if not enabled:
            return
        try:
            import trackio
        except ImportError:
            print("decision-mlx: tracking requested but trackio is not installed "
                  "(uv pip install 'decision-mlx[tracking]')", file=sys.stderr)
            return
        init_kwargs.setdefault("embed", False)
        if system_interval:  # our sampler covers GPU, MLX and host; Trackio's Apple monitor adds only gpu/detected
            init_kwargs.setdefault("auto_log_gpu", False)
            init_kwargs.setdefault("auto_log_cpu", False)
        runs: list = []
        if not self._guard(lambda: runs.append(trackio.init(project=project, name=name, group=group,
                                                            config=dict(config or {}), **init_kwargs))):
            return
        self._trackio = trackio
        # trackio.log_system finds the run through a ContextVar that threads do not inherit, so the sampler
        # thread logs through the Run object itself (as Trackio's own monitors do).
        self._run = runs[0] if runs else None
        if system_interval:
            self._sampler = threading.Thread(target=self._sample, args=(float(system_interval),), daemon=True)
            self._sampler.start()

    def _sample(self, interval: float) -> None:
        log_system = getattr(self._run, "log_system", None)
        while not self._stop.is_set() and self.active:
            try:
                metrics = system_metrics()
                if metrics:
                    if log_system is not None:
                        log_system(metrics)
                    else:
                        self._trackio.log({f"system/{k}": v for k, v in metrics.items()})
            except Exception as error:  # noqa: BLE001 - a sampling hiccup must not end the sampler
                print(f"decision-mlx: system sample skipped: {error!r}", file=sys.stderr)
            self._stop.wait(interval)

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
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=5)
        if self.active:
            self._guard(self._trackio.finish)
        self._trackio = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()
