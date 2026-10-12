# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Config checks registered by the package root."""
from __future__ import annotations

from typing import TYPE_CHECKING

from hfhost.config import ConfigError, choice

if TYPE_CHECKING:
    from hfmodem.core.config import ProtocolConfig


DEFAULT_MODE = "110 1200L"

def validate(p: ProtocolConfig, ctx: str, provided: frozenset[str] = frozenset()) -> None:
    """A STANAG 5066 node: one address, and a mode the modem has."""
    if p.bandwidth:
        choice(p.bandwidth, ("3000", "3200"), ctx, "bandwidth")
    if not p.enabled or p.host != "s5066":
        return
    from .modem import ModemError
    from .s5066.dpdu import address
    from .s5066.onair import parse_mode
    if len(p.addresses) != 1:
        raise ConfigError(f"{ctx}: host = \"s5066\" takes one address in addresses, "
                          "the node's own (\"0.0.0.1\" and the like)")
    try:
        address(str(p.addresses[0]))
    except ValueError as exc:
        raise ConfigError(f"{ctx}: addresses: {exc}") from None
    try:
        parse_mode(p.mode or DEFAULT_MODE)
    except ModemError as exc:
        raise ConfigError(f"{ctx}: mode: {exc}") from None


def validate_ale(p: "ProtocolConfig", ctx: str, provided: frozenset[str] = frozenset()) -> None:
    """Addresses the standard allows, and channels to scan."""
    if not p.enabled:
        return
    if not p.addresses:
        raise ConfigError(f"{ctx}: addresses is required: the station's own ALE "
                          "address(es), e.g. addresses = [\"W9SSJ\"]")
    for a in p.addresses:
        if not _ale_address(a):
            raise ConfigError(f"{ctx}: addresses: {a!r} is not an ALE address "
                              "(1 to 15 of A-Z and 0-9)")
    if not p.channels:
        raise ConfigError(f"{ctx}: channels is required: the dial frequencies, "
                          "in Hz, the station scans and calls on")
    for f in p.channels:
        if not isinstance(f, (int, float)) or not 1_500_000 <= f <= 30_000_000:
            raise ConfigError(f"{ctx}: channels: {f!r} is not an HF dial frequency in Hz")
    if not 0.5 <= p.scan_rate <= 10:
        raise ConfigError(f"{ctx}: scan_rate {p.scan_rate} is not 0.5 to 10 channels a second "
                          "(A.4.2.1 asks for 2 and 5)")
    import math
    if not math.isfinite(p.sound_s) or (p.sound_s != 0 and p.sound_s < 60):
        raise ConfigError(f"{ctx}: sound_s must be zero or at least 60 seconds")
    if not math.isfinite(p.activity_s) or p.activity_s < 0:
        raise ConfigError(f"{ctx}: activity_s must be finite and nonnegative")
    if p.conventions == "hflink" and (p.sound_s != 0 or p.activity_s != 30 or
                                     provided & {"sound_s", "activity_s"}):
        raise ConfigError(f"{ctx}: hflink sets sound_s and activity_s; remove those overrides")
    if p.conventions != "hflink" and (p.base or p.pilot_channels):
        raise ConfigError(f"{ctx}: base and pilot_channels need conventions = hflink")
    for n in p.nets:
        if not (isinstance(n, dict) and set(n) == {"address", "member", "slot"}):
            raise ConfigError(f"{ctx}: nets: each is {{address, member, slot}}, not {n!r}")
        if not _ale_address(n["address"]) or not _ale_address(n["member"]) \
                or n["member"].upper() not in {a.upper() for a in p.addresses}:
            raise ConfigError(f"{ctx}: nets: {n['address']!r} needs an ALE address, and "
                              "a member address that is one of addresses")
        if not (isinstance(n["slot"], int) and 1 <= n["slot"] <= 32):
            raise ConfigError(f"{ctx}: nets: slot {n['slot']!r} is not 1 to 32")
    if p.conventions not in ("", "hflink"):
        raise ConfigError(f"{ctx}: conventions {p.conventions!r} is not \"\" or \"hflink\"")
    missing = [f for f in p.pilot_channels if f not in p.channels]
    if missing:
        raise ConfigError(f"{ctx}: pilot_channels {missing!r} are not among channels")
    if p.lp_level not in (0, 1, 2):
        raise ConfigError(f"{ctx}: lp_level {p.lp_level!r} is not 0 (off), 1 (AL-1) or 2 (AL-2)")
    if p.lp_level:
        k = p.lp_key
        if not (isinstance(k, str) and len(k) == 14
                and all(c in "0123456789abcdefABCDEF" for c in k)):
            # The key itself is never repeated in a message.
            raise ConfigError(f"{ctx}: lp_key must be 14 hex digits (a 56-bit key) "
                              "when lp_level is set")


def validate_selcall(p: "ProtocolConfig", ctx: str, provided: frozenset[str] = frozenset()) -> None:
    if not p.enabled:
        return
    lens = [len(a) if isinstance(a, str) else 0 for a in p.addresses]
    if not p.addresses or any(not (isinstance(a, str) and a.isdigit() and len(a) in (4, 6))
                              for a in p.addresses) or len(set(lens)) != len(lens):
        raise ConfigError(f"{ctx}: addresses: one selcall address of four digits and/or one "
                          f"of six, e.g. addresses = [\"1234\"], not {list(p.addresses)!r}")
    if not 0.2 <= p.dots_s <= 20:
        raise ConfigError(f"{ctx}: dots_s {p.dots_s} is not 0.2 to 20 seconds")


def _ale_address(a) -> bool:
    """MIL-STD-188-141B A.5.2.4.4: 1 to 15 of A-Z and 0-9."""
    return isinstance(a, str) and 1 <= len(a) <= 15 and all(
        c.isascii() and c.isalnum() for c in a)


