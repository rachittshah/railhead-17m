"""macOS scheduling for simulator servers.

Processes this pipeline spawns (from tmux / a CLI agent, not a GUI app) run under daemon resource policies,
which on Apple Silicon keep them on the efficiency cores: 22 servers shared 4 E-cores while the 10 P-cores
sat ~60% idle. ``taskpolicy -a`` launches a program with the policies given to applications; a 148-vehicle
state stepped at 11 ms/game day under it vs 95-100 ms without (same load). The policy is not inherited
through a parent, so each OpenTTD server is launched under it. ``RH17M_APP_POLICY=0`` turns it off.
"""

from __future__ import annotations

import os
import shutil
import sys

from railhead import server as srv

PREFIX = ["taskpolicy", "-a"]


def install() -> None:
    """Launch every OpenTTD server of this process under ``taskpolicy -a`` (macOS only; idempotent)."""
    if sys.platform != "darwin" or os.environ.get("RH17M_APP_POLICY", "1") == "0" or not shutil.which(PREFIX[0]):
        return
    orig = srv.OpenTTDServer.command
    if getattr(orig, "_rh17m_app_policy", False):
        return

    def command(self) -> list[str]:
        return [*PREFIX, *orig(self)]

    command._rh17m_app_policy = True
    command.__wrapped__ = orig
    srv.OpenTTDServer.command = command
