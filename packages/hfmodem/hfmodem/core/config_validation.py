# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Protocol validators wired by the package initializer, called at config load."""
import math

from collections.abc import Callable
from typing import TYPE_CHECKING

from hfhost.config import ConfigError, choice

if TYPE_CHECKING:
    from .config import ProtocolConfig


PROTOCOL_VALIDATORS: dict[str, Callable[["ProtocolConfig", str, frozenset[str]], None]] = {}


def check_linking_protection(regulatory: str, lp_level: int) -> None:
    if lp_level and regulatory == "part97":
        raise ConfigError("Your chosen part97 profile excludes ALE linking protection; "
                          "set lp_level = 0 for this profile.")


def check_station_rig(station, rig) -> None:
    # Here, not at module level: `import hfmodem` loads this module, and
    # `python -m hfmodem.core.rigs` warns if the package has already loaded .rigs.
    from .rigs import CAT_BACKENDS

    for where, port in (("[station]", station.port), ("[rig]", rig.port)):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ConfigError(f"{where}: port must be an integer in 1..65535")
    choice(rig.cat, tuple(CAT_BACKENDS), "[rig]", "cat")
    if rig.cat == "none" and station.transmit:
        raise ConfigError('[rig]: cat = "none" requires transmit = false')
    if not isinstance(rig.identity, str) or any(ord(c) < 32 or ord(c) == 127 for c in rig.identity):
        raise ConfigError("[rig]: identity must be text without control characters")
    for name, value in (("dial_max_age_s", rig.dial_max_age_s), ("max_watts", rig.max_watts)):
        if value is None and name == "max_watts":
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"[rig]: {name} must be finite and positive")
