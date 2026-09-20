# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Modem-shaped configuration: frozen dataclasses plus the parsing primitives.

There is deliberately no loader here. A shared loader would have to know one
consumer's file format, and the consumers differ — a bench harness has a
[responder] section an operational daemon has no business carrying. So this
module supplies the sections every consumer shares and the validation helpers
to compose them, and each consumer owns its own top-level load().
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Protocol

from .wire import BANDWIDTHS, COMPRESSION_MODES


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Quirks:
    version_reply: bool = True
    iamalive_s: float = 60.0
    hello_timeout_s: float | None = None   # per-modem override of responder's
    preconnect_queue: bool = True          # spec §7.4; set False for a modem
                                           #   that drops pre-CONNECT writes
    max_idle_recycle_s: float | None = None


DIALECTS = ("vara", "hostapi", "ardop")


@dataclass(frozen=True, slots=True)
class ModemConfig:
    name: str
    cmd_port: int
    data_port: int
    dialect: str = "vara"            # "hostapi" modems ignore data_port

    spawn: tuple[str, ...] | None = None
    cwd: str | None = None
    bandwidth: str = "2300"
    compression: str = "OFF"
    rev: str = ""
    sim_time: bool = False           # tags sessions against simulated air
    spawn_group: str | None = None
    quirks: Quirks = field(default_factory=Quirks)


@dataclass(frozen=True, slots=True)
class SpawnGroupConfig:
    name: str
    spawn: tuple[str, ...]
    cwd: str | None = None


@dataclass(frozen=True, slots=True)
class RigConfig:
    host: str = "127.0.0.1"
    port: int = 4532


class ModemSource(Protocol):
    """What Supervisor needs of a config: the modem inventory and its groups."""

    modems: tuple[ModemConfig, ...]
    spawn_groups: tuple[SpawnGroupConfig, ...]

    def modem(self, name: str) -> ModemConfig: ...


def build_section(cls, table: dict, ctx: str, **extra):
    """Instantiate a config dataclass from a TOML table, rejecting typos."""
    allowed = {f.name for f in fields(cls)} - set(extra)
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{ctx}: unknown key(s) {', '.join(unknown)}")
    try:
        return cls(**table, **extra)
    except TypeError as exc:
        raise ConfigError(f"{ctx}: {exc}") from None


def choice(value: str, allowed: tuple[str, ...], ctx: str, key: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{ctx}: {key} must be one of {', '.join(allowed)} (got {value!r})")


def _spawn_tuple(table: dict, ctx: str) -> None:
    if "spawn" in table:
        spawn = table["spawn"]
        if not (isinstance(spawn, list) and spawn
                and all(isinstance(s, str) for s in spawn)):
            raise ConfigError(f"{ctx}: spawn must be a non-empty list of strings")
        table["spawn"] = tuple(spawn)


def parse_spawn_groups(raw: dict) -> tuple[SpawnGroupConfig, ...]:
    """Parse a [spawn_group.<name>] table-of-tables."""
    groups: list[SpawnGroupConfig] = []
    for name, table in raw.items():
        ctx = f"[spawn_group.{name}]"
        if not isinstance(table, dict):
            raise ConfigError(f"{ctx}: must be a table")
        _spawn_tuple(table, ctx)
        if "spawn" not in table:
            raise ConfigError(f"{ctx}: missing spawn")
        groups.append(build_section(SpawnGroupConfig, table, ctx, name=name))
    return tuple(groups)


def parse_modems(raw: list, groups: tuple[SpawnGroupConfig, ...]
                 ) -> tuple[ModemConfig, ...]:
    """Parse an array of [[modem]] tables, cross-checking them against the
    spawn groups and against each other. An empty result is not an error here:
    whether a consumer can run with no modems is the consumer's rule."""
    group_names = {g.name for g in groups}
    modems: list[ModemConfig] = []
    for i, table in enumerate(raw):
        ctx = f"[[modem]] #{i + 1}"
        if "name" not in table:
            raise ConfigError(f"{ctx}: missing name")
        ctx = f"[[modem]] {table['name']}"
        _spawn_tuple(table, ctx)
        quirks = build_section(Quirks, table.pop("quirks", {}), f"{ctx} quirks")
        modem = build_section(ModemConfig, table, ctx, quirks=quirks)
        choice(modem.dialect, DIALECTS, ctx, "dialect")
        choice(modem.bandwidth, BANDWIDTHS, ctx, "bandwidth")
        choice(modem.compression, COMPRESSION_MODES, ctx, "compression")
        if modem.spawn is not None and modem.spawn_group is not None:
            raise ConfigError(f"{ctx}: spawn and spawn_group are mutually exclusive")
        if modem.spawn_group is not None and modem.spawn_group not in group_names:
            raise ConfigError(f"{ctx}: spawn_group {modem.spawn_group!r} is not defined")
        modems.append(modem)

    seen_names: set[str] = set()
    seen_ports: dict[int, str] = {}
    for m in modems:
        if m.name in seen_names:
            raise ConfigError(f"duplicate modem name {m.name!r}")
        seen_names.add(m.name)
        for port in (m.cmd_port, m.data_port):
            if not port:
                continue        # 0 means "unused": single-socket modems all
                                # leave data_port at 0 and do not collide
            if port in seen_ports:
                raise ConfigError(f"port {port} used by both {seen_ports[port]} and {m.name}")
            seen_ports[port] = m.name
    return tuple(modems)
