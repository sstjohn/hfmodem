# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Resolve a CW station identity independently of the digital caller.

The command-line entry point validates launcher identity options without
opening devices. Its two output lines are the digital call and the CW call.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
import sys


def validate_call(value: str, source: str) -> str:
    """Accept one ASCII Morse callsign; never rewrite an SSID or punctuation."""
    if not isinstance(value, str) or not value.isascii():
        raise ValueError(f"{source}: CW callsign must be ASCII A-Z, 0-9 and '/'")
    call = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9/]+", call) or not any(c.isalnum() for c in call):
        raise ValueError(f"{source}: CW callsign must be nonempty A-Z, 0-9 and '/' "
                         "without spaces or an SSID; set --idcall for a digital tactical call")
    return call


@dataclass(frozen=True, slots=True)
class CallIdentity:
    digital_call: str
    cw_call: str
    source: str

    @property
    def separate(self) -> bool:
        call = self.digital_call
        return not (isinstance(call, str) and call.isascii()
                    and call.strip().upper() == self.cw_call)

    def closing_required(self, mode: str, *, optional: bool, linked: bool = False) -> bool:
        """Keep each mode's legacy suppression only for the same identity."""
        if mode not in ("pactor", "vara", "ardop"):
            raise ValueError(f"unknown closing identification mode {mode!r}")
        return self.separate or not optional or (mode == "pactor" and not linked)

    def announce(self, *, optional: bool = False) -> None:
        print(f"  identity: digital {self.digital_call}, CW {self.cw_call} "
              f"({self.source})", flush=True)
        if optional and self.separate:
            print("  distinct CW identity: --cwid-optional cannot suppress the closing CW ID",
                  flush=True)


def from_station(station) -> CallIdentity:
    """StationProcess uses only its configuration, never calling-tool overrides."""
    value = station.idcall
    source = "[station] idcall" if value != "" else "[station] mycall"
    from hfhost.config import ConfigError
    try:
        call = validate_call(value if value != "" else station.mycall, source)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return CallIdentity(station.mycall, call, source)


def resolve(digital_call: str, override: str | None = None, *, environ=None) -> CallIdentity:
    """CLI > IDCALL > configured idcall > this attempt's effective digital call."""
    from . import config

    environ = os.environ if environ is None else environ
    if not isinstance(digital_call, str) or any(c in digital_call for c in "\r\n"):
        raise ValueError("digital mycall must be a single line")
    if override is not None:
        value, source = override, "--idcall"
    elif "IDCALL" in environ:
        value, source = environ["IDCALL"], "IDCALL"
    else:
        value, source = digital_call, "digital mycall fallback"
        path = environ.get(config.STATION_ENV)
        if path:
            try:
                station = config.load(path, environ=environ).station
            except (OSError, ValueError) as exc:
                raise config.ConfigError(f"{config.STATION_ENV} names {path}: {exc}") from exc
            if station.idcall != "":
                value, source = station.idcall, "[station] idcall"
    return CallIdentity(digital_call, validate_call(value, source), source)


def options(words: list[str], *, mycall: str, idcall: str | None = None):
    """Read final identity options, including equals forms, among launcher flags."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mycall", default=mycall)
    parser.add_argument("--idcall", default=idcall)
    args, _ = parser.parse_known_args(words)
    return args


def main() -> int:
    args = options(sys.argv[1:], mycall="")
    try:
        identity = resolve(args.mycall, args.idcall)
    except ValueError as exc:
        print(f"identification: {exc}", file=sys.stderr)
        return 2
    print(identity.digital_call)
    print(identity.cw_call)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
