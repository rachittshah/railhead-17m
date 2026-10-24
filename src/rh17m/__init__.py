"""railhead-17m: a 17M System-1 decision model for Railhead."""

__version__ = "0.1.0"

# Every process (CLI and spawned pool workers) launches its OpenTTD servers on the performance cores.
from rh17m import sched as _sched

_sched.install()
