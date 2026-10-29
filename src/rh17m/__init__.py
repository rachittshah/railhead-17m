"""railhead-17m: a 17M System-1 decision model for Railhead."""

__version__ = "0.1.0"

# Every process (CLI and spawned pool workers) launches its OpenTTD servers on the performance cores, and
# traces its simulator calls (a no-op unless a trace viewer is listening; see tracing.py).
from rh17m import pipeline as _pipeline
from rh17m import pool as _pool
from rh17m import sched as _sched
from rh17m import tracing as _tracing

_sched.install()
_pipeline.install()
_pool.install()  # opt-in (RH17M_POOL=1); before tracing, so world.load / world.close spans wrap the pooled ones
_tracing.install()
