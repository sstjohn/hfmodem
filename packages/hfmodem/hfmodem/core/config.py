# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One file, the whole station.

Before the merge none of the four modems had persistent configuration of any
kind. Every station fact — CAT device, audio device, hamlib model, baud,
callsign, dial — was retyped on each invocation or baked into a shell script,
which is how the same measured codec gain came to be wrong in four places at
once and how three different rig tables acquired three different settle times.

Built on `hfhost.config`'s primitives because they already do the one thing that
matters most here: **an unknown key is an error, not a shrug.** A typo'd key is a
setting that silently did not apply, and on a transmitter that is the difference
between a hot input and a working one.

Everything is validated eagerly at load. The alternative — validating when a
value is first used — means a station comes up, listens for an hour, answers a
call, and only then discovers it cannot key.
"""
from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from hfhost.config import ConfigError, build_section, choice

from hfmodem.core import band, levels
from hfmodem.core.regulatory import Control, Profile, profile
from hfmodem.winlink.session import CLIENT_SID

SCHEMA = 1

#: Names the station file for the on-air tools, which take device flags from
#: `tools/onair.sh` and no file of their own. Same contract as CAT_PORT and
#: HAMLIB_BIN: one export in the operator's profile answers every launcher's
#: children at once, including the ones nobody has written yet.
STATION_ENV = "HFMODEM_STATION"

#: Names the Winlink secure-login password, for every tool here that can run a
#: B2F session. Same contract as STATION_ENV: one export in the operator's
#: profile answers every launcher's children at once — and unlike a flag, it is
#: not in the argv every `ps` on the machine can read.
PASSWORD_ENV = "WINLINK_PASSWORD"

#: The measured working point at this station — FT-891 into a C-Media dongle,
#: rig DATA OUT at 45/100 — is 0.040, holding real traffic at −4 to −8 dBFS peak
#: with nothing railed. The upper bound is 2.5× that, about +8 dB, which puts
#: peaks at the very edge of full scale; anything above it is railing on any
#: chain. It is set here rather than looser specifically to refuse 0.118 and
#: 0.18, two of the four values this station has run at and decoded nothing on.
#:
#: This is a coarse sanity bound and not the real protection. A plausible-looking
#: wrong value passes it, which is why preflight measures the capture and why the
#: standing rule is to verify after setting, every session.
GAIN_RANGE = (0.005, 0.10)


@dataclass(frozen=True, slots=True)
class Station:
    mycall: str = ""
    grid: str = ""
    #: `local` covers local and remote control — a human is at the control point.
    #: `automatic` is the station answering on its own, which is what any
    #: `listen = true` amounts to. No default: several regimes treat the two very
    #: differently and none treats automatic as the easier case.
    control: str = ""
    #: Which rules this station operates under. No default — see core.regulatory.
    regulatory: str = ""
    #: Profile-specific settings, e.g. `licence` for part97.
    regulatory_settings: dict = field(default_factory=dict)
    #: A hard interlock, not a preference. False and nothing can key.
    transmit: bool = False
    id_interval_s: float = 540.0
    id_mode: str = "cw"
    #: What this station calls itself to a Winlink gateway: the name and version
    #: half of the B2F SID, ours by default. The capability letters are not in
    #: here — see `winlink.session._CAPABILITIES`.
    client_sid: str = CLIENT_SID


@dataclass(frozen=True, slots=True)
class Ptt:
    backend: str = "rts"
    port: str = ""
    settle_s: float = 0.04


@dataclass(frozen=True, slots=True)
class Rig:
    model: str = ""
    cat: str = "rigctld"
    host: str = "127.0.0.1"
    port: int = 4532
    mode: str = "PKTUSB"
    #: The CHANNEL CENTRE, as a gateway list publishes it. The dial is derived.
    centre_hz: int = 0
    max_key_s: float = 30.0
    #: PEP the transmitter is set to, in watts. Optional everywhere except a band
    #: that carries its own limit, where §97.313 refuses an emission that does not
    #: state its power — the alternative being to transmit an unknown amount into
    #: a capped band. Without this there was no way to say, so 30 m refused every
    #: protocol and no configuration could fix it.
    power_w: float | None = None
    ptt: Ptt = field(default_factory=Ptt)

    @property
    def dial_hz(self) -> int:
        return band.dial_hz(self.centre_hz)


@dataclass(frozen=True, slots=True)
class Audio:
    input: str = ""
    output: str = ""
    rate: int = 48000
    blocksize: int = 128
    latency: str = "low"
    #: From core.levels, not retyped. A second definition of this number inside
    #: the ratchet's own scan path is how it came to be wrong in four places — and
    #: the ratchet missed this one, because it looks for an ALL-CAPS name.
    input_gain: float = levels.WORKING_POINT
    #: From core.levels for the same reason input_gain is: one transmit level for
    #: one interface, not one per protocol.
    tx_drive: float = levels.TX_DRIVE
    #: Milliseconds of ADC→DAC loop latency the driver does not report, added to
    #: the measured offset on every burst this station schedules. `lat` is the
    #: converter's own `outputBufferDacTime − inputBufferAdcTime`; what it leaves
    #: out is the codec's in-and-out delay, and on 2026-09-14 that put every
    #: emission 20 ms later in the peer's cycle than the scheduler believed —
    #: measured twice over, from the PACTOR-3 witness chain and from the peer's
    #: PACTOR-1 answer position, agreeing to 1 ms. It is a property of the audio
    #: path and not of any protocol, so it is here and not in a modem constant.
    #: Zero until a station has measured its own.
    tx_latency_ms: float = 0.0
    capture_dir: str = ""


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    enabled: bool = False
    host: str = ""
    cmd_port: int = 0
    data_port: int = 0
    port: int = 0
    pty: str = ""
    bandwidth: str = ""
    listen: bool = False


@dataclass(frozen=True, slots=True)
class StationConfig:
    station: Station
    rig: Rig
    audio: Audio
    protocols: dict[str, ProtocolConfig]

    @property
    def control(self) -> Control:
        return Control(self.station.control)

    @property
    def profile(self) -> Profile:
        return profile(self.station.regulatory, **self.station.regulatory_settings)

    @property
    def listens(self) -> bool:
        return any(p.listen for p in self.protocols.values() if p.enabled)


_PROTOCOL_HOSTS = {
    "pactor": ("ptc",),
    "vara": ("vara",),
    "ardop": ("ardop",),
    "sabir": ("hostapi",),
}


def load(path: str | Path) -> StationConfig:
    """Read and validate a station file. Every error names its key."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return parse(raw)


def station_profile() -> Profile | None:
    """The regime this station declared, or None when no station file is named.

    For the tools that key a transmitter and hold no station file of their own.
    None is not a default regime — it is the absence of a declaration, and a
    caller that has to assume something should assume the stricter thing.
    """
    path = os.environ.get(STATION_ENV)
    if not path:
        return None
    try:
        return load(path).profile
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc


def tx_drive(override: float | None = None) -> float:
    """The peak this run leaves at: the flag if one was given, else the station
    file's `[audio] tx_drive`, else the shipped default.

    An override is per-run and deliberate — set at the rig against a meter — so it
    wins over the file. Everything else defers to the file, which is the whole
    point: `tx_drive` was declared, documented and read by nothing, so an operator
    could set it, watch nothing change, and conclude drive was not the problem.
    """
    if override is not None:
        return override
    path = os.environ.get(STATION_ENV)
    if not path:
        return levels.TX_DRIVE
    try:
        return load(path).audio.tx_drive
    except (OSError, ConfigError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc


#: A correction, not a clock. Past this the number is a misreading — the offset
#: the driver DOES report is 26 ms on this station's hardware, and a residual
#: several times that is a broken measurement being handed to the transmitter.
TX_LATENCY_RANGE_MS = (0.0, 60.0)


def tx_latency_ms(override: float | None = None) -> float:
    """Milliseconds to key every burst early for this station's audio path: the
    flag if one was given, else `[audio] tx_latency_ms`, else nothing.

    Read like `tx_drive` and bounded here rather than at each caller, so a flag
    and a file cannot disagree about what is a plausible correction.
    """
    value = override
    if value is None:
        path = os.environ.get(STATION_ENV)
        if not path:
            return 0.0
        try:
            value = load(path).audio.tx_latency_ms
        except (OSError, ConfigError) as exc:
            raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc
    lo, hi = TX_LATENCY_RANGE_MS
    if not lo <= value <= hi:
        raise ConfigError(
            f"tx_latency_ms {value} is outside {lo}–{hi} ms. It is the part of "
            "the ADC→DAC loop the driver does not report, measured by keying "
            "into a loopback and reading where our own burst lands in our own "
            "capture; this station measured 20.")
    return value


def client_sid(override: str = "") -> str:
    """What this run announces itself as to a Winlink gateway: the flag if one
    was given, else the station file's `[station] client_sid`, else ours.

    A gateway may refuse a client type it does not know, and that refusal ends
    the session before any mail can move. What a station announces is a claim
    the operator makes about their own station, so it is the operator's to set
    and this project ships only its own honest name.
    """
    if override:
        return _check_client_sid(override, "--mail-sid")
    path = os.environ.get(STATION_ENV)
    if not path:
        return CLIENT_SID
    try:
        return load(path).station.client_sid
    except (OSError, ConfigError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc


def mail_password(flag: str = "", path: str = "") -> str:
    """The answer to a gateway's ;PQ: challenge, from somewhere that is not argv.

    A password spelled out as a flag argument has two ways out and takes both.
    It is in the process's argv, which every `ps` on the machine can read for
    the life of the run; and it is copied by anything that echoes a command
    line, which is how `tools/onair.sh` came to write its own `ps -o command=`
    into the rig claim and print it back at the next launcher that asked.

    So a file whose mode is the operator's business, or the environment, and the
    flag last and loudly. Two explicit sources are refused rather than ranked:
    at a rig, a setting that silently did not apply is the whole failure mode
    this file's unknown-key rule exists for.
    """
    if flag and path:
        raise ConfigError(
            "`--mail-password` and `--mail-password-file` both name a password, "
            "and nothing here can tell which one this run meant.")
    if path:
        p = Path(path).expanduser()
        try:
            mode = p.stat().st_mode
        except OSError as exc:
            raise ConfigError(f"--mail-password-file names {p}: {exc}") from exc
        if mode & 0o077:
            print(f"!! {p} is readable beyond its owner — chmod 600 it",
                  file=sys.stderr)
        return p.read_text(encoding="utf-8").strip()
    if flag:
        print("!! `--mail-password` puts the password in this process's argv, where "
              f"every `ps` on this machine can read it and every `tee` writes it "
              f"down. Export {PASSWORD_ENV}, or pass --mail-password-file.",
              file=sys.stderr)
    return flag or os.environ.get(PASSWORD_ENV, "")


def grid() -> str:
    """The station file's `[station] grid`, for the B2F comment line.

    Pat writes `; TARGET DE MYCALL (LOCATOR)` and so does every other B2F client
    a gateway sees; a station that declared its grid should say it.
    """
    path = os.environ.get(STATION_ENV)
    if not path:
        return ""
    try:
        return load(path).station.grid
    except (OSError, ConfigError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc


def _check_client_sid(value: str, where: str) -> str:
    # A CR here would not be a malformed SID; it would be a second protocol line
    # of the setter's choosing, injected into the handshake.
    if not value or any(c < " " or c > "~" or c in "[]$" for c in value):
        raise ConfigError(
            f"{where}: client_sid is the NAME and VERSION only, as in "
            f"{CLIENT_SID!r} — printable ASCII, no brackets and no '$'. "
            "hfmodem writes the brackets and appends the B2F capability "
            "letters itself, because those state what this code can do and "
            "are not an identity to choose.")
    return value


def parse(raw: dict) -> StationConfig:
    if raw.get("schema") != SCHEMA:
        raise ConfigError(f"schema must be {SCHEMA} (got {raw.get('schema')!r})")
    body = {k: v for k, v in raw.items() if k != "schema"}
    unknown = sorted(set(body) - {"station", "rig", "audio", "protocols", "logging"})
    if unknown:
        raise ConfigError(f"unknown top-level table(s) {', '.join(unknown)}")

    st_raw = dict(body.get("station", {}))
    settings = {k: st_raw.pop(k) for k in ("licence", "because") if k in st_raw}
    station = build_section(Station, st_raw, "[station]", regulatory_settings=settings)

    if "dial_hz" in body.get("rig", {}):
        raise ConfigError(
            "[rig]: write centre_hz, not dial_hz. A gateway list publishes the "
            "channel CENTRE and the dial is derived as centre − "
            f"{band.DIAL_OFFSET_HZ} Hz. Getting this wrong is silent and kills "
            "both directions.")
    rig_raw = dict(body.get("rig", {}))
    ptt = build_section(Ptt, dict(rig_raw.pop("ptt", {})), "[rig.ptt]")
    rig = build_section(Rig, rig_raw, "[rig]", ptt=ptt)
    audio = build_section(Audio, dict(body.get("audio", {})), "[audio]")

    protocols = {}
    for name, table in body.get("protocols", {}).items():
        if name not in _PROTOCOL_HOSTS:
            raise ConfigError(
                f"[protocols.{name}]: unknown protocol. "
                f"Known: {', '.join(sorted(_PROTOCOL_HOSTS))}")
        p = build_section(ProtocolConfig, dict(table), f"[protocols.{name}]")
        if p.enabled:
            choice(p.host, _PROTOCOL_HOSTS[name], f"[protocols.{name}]", "host")
        protocols[name] = p

    cfg = StationConfig(station, rig, audio, protocols)
    _validate(cfg)
    return cfg


def _validate(cfg: StationConfig) -> None:
    st, audio, rig = cfg.station, cfg.audio, cfg.rig

    if not st.control:
        raise ConfigError(
            "[station]: control is required and has no default — 'local' if an "
            "operator is at the control point, 'automatic' if the station answers "
            "on its own. Several regimes treat the two very differently.")
    choice(st.control, tuple(c.value for c in Control), "[station]", "control")

    if not st.regulatory:
        raise ConfigError(
            "[station]: regulatory is required and has no default. 'part97' for "
            "the US amateur service (add licence = \"general\"|…), or "
            "'unregulated' if you operate under rules hfmodem does not model "
            "(add because = \"…\"), in which case you are answerable for every "
            "emission.")
    cfg.profile  # raises ValueError naming the profile, and validates its settings

    choice(st.id_mode, ("cw", "none"), "[station]", "id_mode")
    _check_client_sid(st.client_sid, "[station]")

    lo, hi = GAIN_RANGE
    if not lo <= audio.input_gain <= hi:
        raise ConfigError(
            f"[audio]: input_gain {audio.input_gain} is outside {lo}–{hi}. The "
            "measured working point at this station is 0.040; it has been wrong "
            "at 0.51, 0.50, 0.18 and 0.118, and a hot input presents as a working "
            "radio that decodes nothing.")

    enabled = {n: p for n, p in cfg.protocols.items() if p.enabled}

    if enabled and not audio.output:
        raise ConfigError(
            "[audio]: output is required when any protocol is enabled. An unset "
            "device means the system default, which is the built-in speakers.")

    seen: dict[int, str] = {}
    for name, p in enabled.items():
        for port in (p.cmd_port, p.data_port, p.port):
            if not port:
                continue
            if port in seen:
                raise ConfigError(
                    f"[protocols.{name}]: port {port} is already used by "
                    f"[protocols.{seen[port]}]")
            seen[port] = name

    # There was a guard here comparing the PTT port against the CAT port. It could
    # never fire: the daemon holds the CAT device and does not tell us which one,
    # so the comparison was always against an empty string. A check that cannot
    # fail reads as protection and is not, so what protects that boundary now is
    # stated where it happens — `hfmodem rig --preflight` prints the device about
    # to be keyed and by what method, and the arm gate refuses a daemon that owns
    # PTT itself.
    if rig.ptt.backend == "rigctld":
        raise ConfigError(
            "[rig.ptt]: backend = \"rigctld\" is not implemented, and this "
            "station would key RTS regardless — a transmitter keyed by a route "
            "other than the configured one. Use backend = \"rts\" with the PTT "
            "interface's own port.")
    choice(rig.ptt.backend, ("rts",), "[rig.ptt]", "backend")

