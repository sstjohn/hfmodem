# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Protocol validators wired by the package initializer, called at config load."""
from collections.abc import Callable
from typing import TYPE_CHECKING

from hfhost.config import ConfigError

if TYPE_CHECKING:
    from .config import ProtocolConfig


PROTOCOL_VALIDATORS: dict[str, Callable[["ProtocolConfig", str, frozenset[str]], None]] = {}


def check_linking_protection(regulatory: str, lp_level: int) -> None:
    if lp_level and regulatory == "part97":
        raise ConfigError("Your chosen part97 profile excludes ALE linking protection; "
                          "set lp_level = 0 for this profile.")
