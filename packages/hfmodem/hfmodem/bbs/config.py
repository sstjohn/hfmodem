# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Validated, transport-independent BBS configuration."""
from dataclasses import dataclass, field
from pathlib import Path
import re
import os
import tomllib
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Limits:
    idle_seconds: int = 900
    connection_seconds: int = 3600
    door_seconds_per_day: int = 3600

    def __post_init__(self):
        for name, minimum in (("idle_seconds", 1), ("connection_seconds", 1), ("door_seconds_per_day", 0)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= 86400:
                raise ValueError(f"{name} must be an integer in {minimum}..86400")
        if self.idle_seconds > self.connection_seconds:
            raise ValueError("idle_seconds exceeds connection_seconds")


def normalize_address(address: str, protocol_ssid: bool = True) -> str:
    """Fold only validated AX.25-style numeric SSIDs, preserving portable calls."""
    if not isinstance(address, str):
        raise ValueError("identity must be text")
    value = address.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+(?:/[A-Z0-9]+)*(?:-[0-9]{1,2})?", value) or len(value) > 32:
        raise ValueError("malformed callsign/address")
    if "-" in value:
        base, ssid = value.rsplit("-", 1)
        if int(ssid) > 15 or str(int(ssid)) != ssid:
            raise ValueError("SSID must be canonical 0..15")
        if protocol_ssid:
            value = base
    return value


def validate_message_areas(areas):
    if not isinstance(areas, dict) or len(areas) > 32:
        raise ValueError("message_areas must contain at most 32 areas")
    result = {}
    for slug, title in areas.items():
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", slug) or slug == "sysop":
            raise ValueError("invalid or reserved message area slug")
        if not isinstance(title, str) or not title.strip() or len(title.encode("utf-8")) > 72 or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in title):
            raise ValueError("message area title must be 1..72 UTF-8 bytes without controls")
        result[slug] = title
    return result


@dataclass(frozen=True)
class BBSConfig:
    id: str = "bbs"
    title: str = "BBS"
    state_root: Path = field(default_factory=lambda: Path.home() / ".local/share/hfmodem/bbs/bbs")
    limits: Limits = field(default_factory=Limits)
    callsigns: dict = field(default_factory=dict)
    aliases: dict = field(default_factory=dict)
    quota_timezone: str = "UTC"
    games: dict = field(default_factory=dict)
    export_root: Path | None = None
    sysop: str = ""
    news: str = ""
    message_areas: dict = field(default_factory=lambda: {"general": "General"})
    message_posts_per_day: int = 20

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.id):
            raise ValueError("invalid BBS id")
        if not isinstance(self.limits, Limits):
            raise ValueError("limits must be Limits")
        ZoneInfo(self.quota_timezone)
        object.__setattr__(self, "message_areas", validate_message_areas(self.message_areas))
        if type(self.message_posts_per_day) is not int or not 0 <= self.message_posts_per_day <= 1000:
            raise ValueError("message_posts_per_day must be an integer in 0..1000")
        object.__setattr__(self, "state_root", Path(self.state_root).expanduser().resolve())
        if self.export_root is not None:
            object.__setattr__(self, "export_root", Path(self.export_root).expanduser().resolve())
        aliases = {}
        for source, target in self.aliases.items():
            source, target = normalize_address(source), normalize_address(target)
            if source in aliases and aliases[source] != target:
                raise ValueError("conflicting alias")
            aliases[source] = target
        for source in aliases:
            current, seen = source, set()
            while current in aliases:
                if current in seen:
                    raise ValueError("cyclic alias")
                seen.add(current)
                current = aliases[current]
        object.__setattr__(self, "aliases", aliases)
        overrides = {}
        for address, override in self.callsigns.items():
            canonical = self.resolve(address)
            if isinstance(override, dict):
                unknown = set(override) - set(Limits.__dataclass_fields__)
                if unknown:
                    raise ValueError(f"unknown limit fields: {unknown}")
                override = Limits(**{**self.limits.__dict__, **override})
            if not isinstance(override, Limits):
                raise ValueError("invalid callsign limits")
            if canonical in overrides and overrides[canonical] != override:
                raise ValueError("conflicting callsign/alias policies")
            overrides[canonical] = override
        object.__setattr__(self, "callsigns", overrides)

    def resolve(self, address: str, protocol_ssid: bool = True) -> str:
        result = normalize_address(address, protocol_ssid)
        while result in self.aliases:
            result = self.aliases[result]
        return result

    def effective_limits(self, address: str, protocol_ssid: bool = True) -> Limits:
        return self.callsigns.get(self.resolve(address, protocol_ssid), self.limits)


def configured_path(override=None, *, station_path=None):
    """Explicit CLI path wins; station-relative paths belong to the station file."""
    if override is not None:
        return Path(override).expanduser().absolute() if override else None
    from ..core import config as station_config
    station_path = station_path or os.environ.get(station_config.STATION_ENV)
    if not station_path:
        return None
    value = station_config.load(station_path).bbs.config
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(station_path).absolute().parent / path


def load(path: str | Path) -> BBSConfig:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as source:
        data = tomllib.load(source)
    board = dict(data.get("bbs", {}))
    limits = Limits(**{key: board.pop(key) for key in tuple(board) if key in Limits.__dataclass_fields__})
    board_id = board.get("id", "bbs")
    board.setdefault("state_root", Path.home() / ".local/share/hfmodem/bbs" / board_id)
    for key in ("state_root", "export_root"):
        if board.get(key) is not None:
            value = Path(board[key]).expanduser()
            board[key] = value if value.is_absolute() else path.parent / value
    games = {}
    for name, settings in data.get("games", {}).items():
        settings = dict(settings)
        for key in ("story", "executable", "directory"):
            if key in settings:
                value = Path(settings[key]).expanduser()
                settings[key] = str(value if value.is_absolute() else path.parent / value)
        games[name] = settings
    return BBSConfig(limits=limits, games=games, **board)
