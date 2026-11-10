# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Site configuration: TOML -> frozen dataclasses, with eager validation.

The modem-shaped sections and the parsing primitives come from hfhost; what is
here is the shape of a *bench site* — a [responder] that answers whoever calls
and an [initiator] that drives a campaign. Modem-shaped names are re-exported
so callers have one config module to import from.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from hfhost.config import (ConfigError, ModemConfig, Quirks, RigConfig,
                           SpawnGroupConfig, build_section, choice,
                           parse_modems, parse_spawn_groups)

__all__ = ["ConfigError", "ModemConfig", "Quirks", "RigConfig", "SpawnGroupConfig",
           "SiteConfig", "ResponderConfig", "InitiatorConfig", "Config", "load"]


@dataclass(frozen=True, slots=True)
class SiteConfig:
    name: str
    mycall: str
    results_dir: str = "results"


@dataclass(frozen=True, slots=True)
class ResponderConfig:
    pending_timeout_s: float = 30.0
    hello_timeout_s: float = 10.0
    max_session_s: float = 900.0
    on_plain_peer: str = "sink"          # sink | disconnect
    retention_max_age_s: float = 30 * 86400.0
    retention_max_bytes: int = 2 * 1024 ** 3
    retention_interval_s: float = 300.0


@dataclass(frozen=True, slots=True)
class InitiatorConfig:
    connect_timeout_s: float = 90.0
    max_session_s: float = 900.0         # hard cap: a stalling peer must not
                                         # hold a campaign open forever
    on_plain_peer: str = "disconnect"    # disconnect | blind


@dataclass(frozen=True, slots=True)
class Config:
    site: SiteConfig
    responder: ResponderConfig
    initiator: InitiatorConfig
    modems: tuple[ModemConfig, ...]
    spawn_groups: tuple[SpawnGroupConfig, ...] = ()
    rig: RigConfig | None = None

    def modem(self, name: str) -> ModemConfig:
        for m in self.modems:
            if m.name == name:
                return m
        raise KeyError(name)


def load(path: str | Path) -> Config:
    path = Path(path)
    with open(path, "rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from None

    known_sections = {"site", "responder", "initiator", "modem", "spawn_group", "rig"}
    unknown = sorted(set(raw) - known_sections)
    if unknown:
        raise ConfigError(f"{path}: unknown section(s) {', '.join(unknown)}")
    if "site" not in raw:
        raise ConfigError(f"{path}: missing [site] section")

    site = build_section(SiteConfig, raw["site"], "[site]")
    responder = build_section(ResponderConfig, raw.get("responder", {}), "[responder]")
    initiator = build_section(InitiatorConfig, raw.get("initiator", {}), "[initiator]")
    choice(responder.on_plain_peer, ("sink", "disconnect"), "[responder]", "on_plain_peer")
    choice(initiator.on_plain_peer, ("disconnect", "blind"), "[initiator]", "on_plain_peer")

    groups = parse_spawn_groups(raw.get("spawn_group", {}))
    modems = parse_modems(raw.get("modem", []), groups)
    if not modems:
        raise ConfigError(f"{path}: no [[modem]] entries")

    rig = build_section(RigConfig, raw["rig"], "[rig]") if "rig" in raw else None
    return Config(site=site, responder=responder, initiator=initiator,
                  modems=modems, spawn_groups=groups, rig=rig)
