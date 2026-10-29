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
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

from hfhost.config import ConfigError, build_section, choice

from hfmodem.core import band, identification, levels
from hfmodem.core.config_validation import PROTOCOL_VALIDATORS, check_linking_protection, check_station_rig
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
    port: int = 8737
    mycall: str = ""
    grid: str = ""
    name: str = ""
    qth: str = ""
    antenna: str = ""
    weather: str = ""
    #: `local` covers local and remote control — a human is at the control point.
    #: `automatic` is the station answering on its own, which is what any
    #: `listen = true` amounts to. No default: the operator selects which
    #: control setting the profile's checks use.
    control: str = ""
    #: Which emission limits to apply. No default — see core.regulatory.
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
    #: CW station identity; empty falls back to the effective digital mycall.
    idcall: str = ""


@dataclass(frozen=True, slots=True)
class Ptt:
    backend: str = "rts"
    port: str = ""
    settle_s: float = 0.04


#: The TUNE carrier's RF power when the station file names none.
TUNE_POWER_W = 10.0


@dataclass(frozen=True, slots=True)
class Rig:
    model: str = ""
    cat: str = "yaesu"
    identity: str = ""
    dial_max_age_s: float = 30.0
    max_watts: float | None = None
    host: str = "127.0.0.1"
    port: int = 4532
    mode: str = "PKTUSB"
    #: The CHANNEL CENTRE, as a gateway list publishes it. The dial is derived.
    centre_hz: int = 0
    max_key_s: float = 30.0
    #: PEP the transmitter is set to, in watts. The part97 profile requires it
    #: for segments with a specific power ceiling. Without this setting there
    #: was no way to supply power, so the profile refused every protocol on 30 m.
    power_w: float | None = None
    #: The most RF power, in watts, anything here sets the rig to. A request above it
    #: sets this instead and says so; a rig found above it at session start is lowered.
    power_clamp_w: float | None = None
    #: The RF power of a TUNE carrier, held to `power_clamp_w` like any other.
    tune_power_w: float = TUNE_POWER_W
    #: The receiver's audio passband, lo and hi Hz: what a monitor on this rig
    #: can hear, read off its SSB filter. Nothing sends it to the radio.
    filter_hz: tuple[int, int] = (300, 2900)
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
    tx_latency_ms: float | None = None
    capture_dir: str = ""


@dataclass(frozen=True, slots=True)
class ChainAudio:
    """Selected non-mobile chain. None means unmeasured or unconfigured."""
    input: str = ""
    output: str = ""
    rate: int = 48000
    blocksize: int = 128
    latency: str = "low"
    input_gain: float | None = None
    gain_range: tuple[float, float] | None = None
    tx_drive: float | None = None
    #: The codec output volume the drives were measured at; None reads as 1.0.
    codec_output: float | None = None
    p1_drive: float | None = None
    p3_drive: float | None = None
    p4_drive: float | None = None
    rx_floor_dbfs: float | None = None
    rx_hot_rms: float = 0.30
    rx_silent_rms: float = 0.001
    peak_limit_dbfs: float = -3.0
    peak_target_dbfs: float = -6.0
    tx_latency_ms: float | None = None
    capture_dir: str = ""
    calibration_status: str = "UNMEASURED"
    calibration_evidence: str = ""
    # Signed residual after the existing TX latency correction, scoped to P4.
    p4_answer_residual_ms: float = 0.0
    p4_answer_residual_evidence: str = ""

    def drive_for(self, preset: str = "", override: float | None = None) -> float:
        value = override if override is not None else getattr(self, f"{preset}_drive", None)
        if value is None:
            value = self.tx_drive
        if value is None:
            warn_config("selected audio chain has no tx_drive; using provisional playback drive 0.6")
            value = levels.TX_DRIVE
        return _unit_interval(value, "tx_drive")


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    enabled: bool = False
    host: str = ""
    cmd_port: int = 0
    data_port: int = 0
    # vara only: stock's KISS port, which VarAC's advanced beacons use; 0 leaves it closed.
    kiss_port: int = 0
    # VarAC opt-in export root; relative to station.toml. Empty disables hosting.
    bbs_directory: str = ""
    port: int = 0
    pty: str = ""
    # Private broker credential for a supervised Windows TCP/serial adapter. This is
    # a file path, never the secret itself in configuration/descriptions.
    auth_token_file: str = ""
    bandwidth: str = ""
    listen: bool = False
    # bateleur with host = "s5066": the modem mode the node starts in
    # ("110 1200L" and the like); its address is ``addresses``' one.
    mode: str = ""
    # ALE only ([protocols.ale]): the station's own addresses, the channels
    # it scans as dial frequencies in Hz (as ALE channel lists publish them),
    # how fast it scans, and how long a quiet link lasts (0: until hung up).
    addresses: tuple = ()
    channels: tuple = ()
    scan_rate: float = 2.0
    activity_s: float = 30.0
    sound_s: float = 0.0
    # The nets it belongs to: {address, member, slot} each — the net's
    # address, the station's own address it answers with, its slot.
    nets: tuple = ()
    # Linking protection (MIL-STD-188-141B appendix B): 0 off, 1 AL-1,
    # 2 AL-2, and the 56-bit key as 14 hex digits.
    lp_level: int = 0
    lp_key: str = ""
    # A network's operating conventions over the standard's defaults:
    # "hflink" for the amateur ALE network's (bateleur.ale.hflink: its nets
    # answered, 15 s calls, TWAS sounds, a six-minute idle link). With it,
    # ``base`` sounds each hour on ``pilot_channels`` (dial frequencies in
    # Hz, among ``channels``; empty: all of them).
    conventions: str = ""
    base: bool = False
    pilot_channels: tuple = ()
    # ALE: directories of who is on the air (bateleur.ale.directory), as
    # URLs or files; "default" is the monitoring list. Read in the
    # background at start; the host port then says who a station is.
    directory: tuple = ()
    # Selcall only ([protocols.selcall]): ``addresses`` are its own, one of
    # four digits and/or one of six; the dot pattern ahead of a call (2 s
    # for a radio sitting on the channel, 6 or more for scanning ones); an
    # acknowledgment call answered to calls that ask; and 1870 Hz sent as 0
    # rather than 1 (an inverted sideband somewhere).
    dots_s: float = 6.0
    emergency: bool = False
    acknowledge: bool = False
    invert: bool = False
    options: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Models:
    memory_gb: float = 48.0


@dataclass(frozen=True, slots=True)
class Voice:
    # Models are local paths; setup fetches weights separately from the package.
    worker_python: str = ""
    recognition_model: str = ""
    synthesis_model: str = ""
    synthesis_backend: str = "kokoro"
    synthesis_voice: str = "am_michael"
    synthesis_speaker: int = 5
    language_model: str = ""
    llama_server: str = "llama-server"
    drive: float = 0.6
    compression: str = "soft"
    phonetics: str = "nato"
    speed: float = 1.05
    vad_threshold: float = 0.5
    silence_s: float = 0.8
    preroll_s: float = 0.32
    max_over_s: float = 25.0
    voice_agc: int = 2
    id_mode: str = "voice"
    exchange: tuple[str, ...] = ("report", "location")


@dataclass(frozen=True, slots=True)
class Desk:
    adif_dir: str = ""


@dataclass(frozen=True, slots=True)
class BBSReference:
    config: str = ""

    def __post_init__(self):
        if not isinstance(self.config, str) or '\x00' in self.config:
            raise ConfigError('[bbs] config must be a path string without NUL')


@dataclass(frozen=True, slots=True)
class StationConfig:
    station: Station
    rig: Rig
    audio: Audio | ChainAudio
    protocols: dict[str, ProtocolConfig]
    selected_rig: str = "ft891"
    rig_source: str = "default"
    audio_chain: str = "ft891_external"
    resolved_profile: dict = field(default_factory=dict)
    resolution_sources: dict = field(default_factory=dict)
    audio_calibration: dict = field(default_factory=dict)
    voice: Voice = field(default_factory=Voice)
    models: Models = field(default_factory=Models)
    desk: Desk = field(default_factory=Desk)
    bbs: BBSReference = field(default_factory=BBSReference)

    def require_audio(self, operation: str, *, input=False, output=False,
                      drive=False, floor=False, preset="", drive_override=None) -> None:
        """Warn on missing calibration; require a name only for needed devices."""
        missing = [key for key, needed in (("input", input), ("output", output))
                   if needed and not getattr(self.audio, key, None)]
        advisory = []
        if floor and getattr(self.audio, "rx_floor_dbfs", None) is None:
            advisory.append("rx_floor_dbfs")
        if drive and isinstance(self.audio, ChainAudio):
            if drive_override is not None:
                _unit_interval(drive_override, "tx_drive")
            elif (getattr(self.audio, f"{preset}_drive", None) is None
                  and self.audio.tx_drive is None):
                advisory.append("tx_drive")
        if advisory or missing:
            warn_config(f"{operation}: incomplete {self.selected_rig.upper()} audio configuration "
                        f"[audio.chains.{self.audio_chain}]: missing "
                        + ", ".join(missing + advisory) + "; continuing with provisional calibration")
        if missing:
            raise ConfigError(f"{operation}: device cannot be named: " + ", ".join(missing))

    @property
    def control(self) -> Control:
        return Control(self.station.control)

    @property
    def profile(self) -> Profile:
        return profile(self.station.regulatory, **self.station.regulatory_settings)

    @property
    def full_scale_w(self) -> float:
        """The watts RFPOWER 1.0 stands for: `[rig] max_watts`, else the rig profile's, else 100."""
        return self.rig.max_watts or self.resolved_profile.get("max_watts") or 100.0

    @property
    def declared_power_w(self) -> float | None:
        """`[rig] power_w` held to the clamp: the power this station sets and declares."""
        if self.rig.power_w is None:
            return None
        return clamp_power(self.rig.power_w, self.rig.power_clamp_w, "")[0]

    @property
    def listens(self) -> bool:
        return any(p.listen for p in self.protocols.values() if p.enabled)


_PROTOCOL_HOSTS = {
    "pactor": ("ptc", "ptc-managed"),
    "vara": ("vara", "vara-managed"),
    "ardop": ("ardop", "ardop-managed"),
    "sabir": ("hostapi",),
    "bateleur": ("bateleur", "s5066"),
    "ale": ("ale",),
    "selcall": ("selcall",),
    "rpr": ("kiss", "mail", "tracker", "tracker-managed", "agwpe"),
}


def load(path: str | Path, *, rig_name=None, environ=None) -> StationConfig:
    """Read and validate a station file. Every error names its key."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return parse(raw, rig_name=rig_name, environ=environ)


def station_profile() -> Profile | None:
    """The emission profile this station selected, or None when no station file is named.

    For the tools that key a transmitter and hold no station file of their own.
    None is not a default profile — it is the absence of a declaration, and a
    caller that has to assume something should assume the stricter thing.
    """
    path = os.environ.get(STATION_ENV)
    if not path:
        return None
    try:
        return load(path).profile
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc


def clamp_power(watts: float, clamp_w: float | None, what: str) -> tuple[float, str]:
    """`watts` held to the station's power clamp, and the warning to show when that
    lowered it; an empty warning when it did not."""
    if clamp_w is None or watts <= clamp_w:
        return watts, ""
    return clamp_w, (f"warning: {what} asks {watts:g} W, above the {clamp_w:g} W station clamp "
                     f"([rig] power_clamp_w); setting {clamp_w:g} W")


class PowerLimits(NamedTuple):
    full_scale_w: float
    clamp_w: float | None
    tune_w: float
    tune_carrier: bool


def power_limits() -> PowerLimits:
    """The named station file's RFPOWER full scale, power clamp and TUNE carrier, for
    tools that hold no station file of their own. No file: the selected rig's profile."""
    path = os.environ.get(STATION_ENV)
    if not path:
        from .rigs import RIGS, resolve_rig
        profile = RIGS[resolve_rig()]
        return PowerLimits(profile.get("max_watts") or 100.0, None, TUNE_POWER_W,
                           profile["tune_carrier"])
    try:
        cfg = load(path)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc
    return PowerLimits(cfg.full_scale_w, cfg.rig.power_clamp_w, cfg.rig.tune_power_w,
                       cfg.resolved_profile.get("tune_carrier", True))


def tx_drive(override: float | None = None) -> float:
    """The peak this run leaves at: the flag if one was given, else the station
    file's `[audio] tx_drive`, else the shipped default.

    An override is per-run and deliberate — set at the rig against a meter — so it
    wins over the file. Everything else defers to the file, which is the whole
    point: `tx_drive` was declared, documented and read by nothing, so an operator
    could set it, watch nothing change, and conclude drive was not the problem.
    """
    if override is not None:
        from .rigs import resolve_rig
        if resolve_rig() != "ft891":
            override = _unit_interval(override, "tx_drive")
        return override
    path = os.environ.get(STATION_ENV)
    if not path:
        from .rigs import resolve_rig
        if resolve_rig() != "ft891":
            warn_config("selected audio chain has no tx_drive; using provisional playback drive 0.6")
            return levels.TX_DRIVE
        return levels.TX_DRIVE
    try:
        value = load(path).audio.tx_drive
        if value is None:
            warn_config("selected audio chain has no tx_drive; using provisional playback drive 0.6")
            return levels.TX_DRIVE
        return value
    except (OSError, ConfigError) as exc:
        raise ConfigError(f"{STATION_ENV} names {path}: {exc}") from exc


#: A correction, not a clock. Past this the number is a misreading — the offset
#: the driver DOES report is 26 ms on this station's hardware, and a residual
#: several times that is a broken measurement being handed to the transmitter.
TX_LATENCY_RANGE_MS = (0.0, 60.0)


def tx_latency_ms(override: float | None = None) -> float:
    from .rigs import resolve_tx_timing

    return resolve_tx_timing(override).tx_latency_ms


def client_sid(override: str = "", path: str | None = None) -> str:
    """What this run announces itself as to a Winlink gateway: the flag if one
    was given, else the station file's `[station] client_sid`, else ours.

    A gateway may refuse a client type it does not know, and that refusal ends
    the session before any mail can move. What a station announces is a claim
    the operator makes about their own station, so it is the operator's to set
    and this project ships only its own honest name. `path` names the station
    file when it is not the one in the environment.
    """
    if override:
        return _check_client_sid(override, "--mail-sid")
    path = path or os.environ.get(STATION_ENV)
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


def _station_field(name: str, what: str, hint: str, path: str | None) -> str:
    path = path or os.environ.get(STATION_ENV)
    found = ""
    if path:
        try:
            found = getattr(load(path).station, name)
        except (OSError, ValueError) as exc:
            warn_config(f"{STATION_ENV} names {path}: {exc}")
    if not found:
        warn_config(f"no {what} configured: {hint} in the file {STATION_ENV} names")
    return found


def mycall(path: str | None = None) -> str:
    """The callsign a tool signs with: `$MYCALL`, else `[station] mycall`.

    Empty, with a warning, when neither names one; a tool carries on without it.
    `path` names the station file when it is not the one in the environment.
    """
    return os.environ.get("MYCALL") or _station_field(
        "mycall", "callsign", "export MYCALL, or set [station] mycall", path)


def grid(path: str | None = None) -> str:
    """`[station] grid`, for the B2F comment line and for tools that place the station.

    Pat writes `; TARGET DE MYCALL (LOCATOR)` and so does every other B2F client
    a gateway sees; a station that declared its grid should say it. Empty, with a
    warning, when none is declared.
    """
    return _station_field("grid", "grid", "set [station] grid", path)


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


def warn_config(message):
    print(f"warning: {message}", file=sys.stderr)


def _unit_interval(value, key):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        warn_config(f"{key} must be finite and between 0 and 1; using provisional 0.6")
        return levels.TX_DRIVE
    return value


def _chain_audio(raw: dict, common: dict, chain: str, *, device_default="") -> ChainAudio:
    table = {key: common[key] for key in ("rate", "blocksize", "latency", "capture_dir")
             if key in common}
    table.update(raw)
    defaults = ChainAudio(input=device_default, output=device_default)
    def discard(key, reason):
        warn_config(f"[audio.chains.{chain}]: {key} {reason}; using provisional default")
        table.pop(key, None)
    for key, value in list(table.items()):
        if key not in ChainAudio.__dataclass_fields__:
            discard(key, "is unknown")
        elif key == "p4_answer_residual_ms":
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not -60 <= value <= 60):
                raise ConfigError("p4_answer_residual_ms must be finite in [-60, 60] ms")
        elif key == "gain_range":
            if (not isinstance(value, list) or len(value) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                    or not 0 <= value[0] <= value[1] <= 1):
                discard(key, "must be two finite bounds in [0, 1]")
            else:
                table[key] = tuple(value)
        elif key in ("input_gain", "tx_drive", "p1_drive", "p3_drive", "p4_drive",
                     "codec_output", "rx_floor_dbfs", "tx_latency_ms", "rx_hot_rms", "rx_silent_rms",
                     "peak_limit_dbfs", "peak_target_dbfs"):
            valid = type(value) in (int, float) and math.isfinite(value)
            if key.endswith("drive") or key == "input_gain":
                valid = valid and 0 <= value <= 1
            elif key == "tx_latency_ms":
                valid = valid and 0 <= value <= 60
            elif key.endswith("dbfs"):
                valid = valid and value < 0
            else:
                valid = valid and 0 < value <= 1
            if not valid:
                discard(key, "has an invalid range or non-finite value")
        elif key in ("rate", "blocksize"):
            if type(value) is not int or value <= 0:
                discard(key, "must be a positive integer")
        elif key == "latency":
            if value not in ("low", "high"):
                discard(key, "must be low or high")
        elif key == "calibration_status":
            if value not in ("UNMEASURED", "PROVISIONAL", "INFERRED", "MEASURED"):
                discard(key, "is unknown")
        elif not isinstance(value, str):
            discard(key, "must be a string")
    if table.get("gain_range") and table.get("input_gain") is not None:
        lo, hi = table["gain_range"]
        if not lo <= table["input_gain"] <= hi:
            warn_config(f"[audio.chains.{chain}]: input_gain outside gain_range; retaining operator value")
    if table.get("calibration_status") == "MEASURED" and not table.get("calibration_evidence"):
        warn_config(f"[audio.chains.{chain}]: MEASURED requires calibration_evidence; treating as PROVISIONAL")
        table["calibration_status"] = "PROVISIONAL"
    for lo, hi in (("rx_silent_rms", "rx_hot_rms"), ("peak_target_dbfs", "peak_limit_dbfs")):
        if table.get(lo, getattr(defaults, lo)) >= table.get(hi, getattr(defaults, hi)):
            discard(lo, "conflicts with " + hi)
            discard(hi, "conflicts with " + lo)
    for key in ("input", "output"):
        if not table.get(key):
            table[key] = device_default
    if table.get("p4_answer_residual_ms") and not table.get("p4_answer_residual_evidence", "").strip():
        raise ConfigError("p4_answer_residual_ms requires p4_answer_residual_evidence")
    return ChainAudio(**table)


def _optional_tables(raw, known, where):
    if not isinstance(raw, dict):
        warn_config(f"{where}: expected tables; ignored")
        return {}
    tables = {}
    for key, table in raw.items():
        if key not in known or not isinstance(table, dict):
            warn_config(f"{where}.{key}: unknown or invalid table; ignored")
        else:
            tables[key] = dict(table)
    return tables


def parse(raw: dict, *, rig_name=None, environ=None) -> StationConfig:
    from .rigs import RIGS, profile_cat, resolve_profile
    environ = os.environ if environ is None else environ
    if raw.get("schema") != SCHEMA:
        raise ConfigError(f"schema must be {SCHEMA} (got {raw.get('schema')!r})")
    body = {k: v for k, v in raw.items() if k != "schema"}
    unknown = sorted(set(body) - {"station", "rig", "audio", "protocols", "logging", "voice", "models", "desk", "bbs"})
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
    profiles = _optional_tables(rig_raw.pop("profiles", {}), RIGS, "[rig.profiles]")
    legacy_model = rig_raw.get("model") or None
    name, selected, sources = resolve_profile(rig_name, environ=environ,
                                              legacy_model=legacy_model, profiles=profiles)
    rig_source = ("operator-flag" if rig_name else "HFMODEM_RIG" if environ.get("HFMODEM_RIG")
                  else "[rig] model" if legacy_model else "default")
    rig_raw["model"] = name
    ptt_raw = dict(rig_raw.pop("ptt", {}))
    rig_raw.setdefault("cat", profile_cat(name))
    # The model token also matches Hamlib identities that omit the manufacturer.
    rig_raw.setdefault("identity", "" if name == "generic" else name)
    if name == "generic":
        rig_raw.setdefault("mode", "USB")
        ptt_raw.setdefault("settle_s", selected["settle"])
        if "ptt_port" in selected:
            ptt_raw["port"] = selected["ptt_port"]
    elif name == "ft891":
        if "ptt_port" in profiles.get("ft891", {}) or environ.get("PTT_PORT"):
            ptt_raw["port"] = selected.get("ptt_port", "")
        if "settle" in profiles.get("ft891", {}):
            ptt_raw["settle_s"] = selected["settle"]
    else:
        rig_raw["mode"] = selected["mode"]
        ptt_raw.pop("port", None)  # legacy mobile port never crosses rigs
        ptt_raw["port"] = selected.get("ptt_port", "")
        ptt_raw["settle_s"] = selected["settle"]
    ptt = build_section(Ptt, ptt_raw, "[rig.ptt]")
    if "filter_hz" in rig_raw:
        lo_hi = rig_raw["filter_hz"]
        if not (isinstance(lo_hi, list) and len(lo_hi) == 2
                and all(isinstance(v, int) for v in lo_hi) and 0 <= lo_hi[0] < lo_hi[1]):
            raise ConfigError("[rig]: filter_hz is [lo, hi] in whole Hz, lo below hi")
        rig_raw["filter_hz"] = tuple(lo_hi)
    rig = build_section(Rig, rig_raw, "[rig]", ptt=ptt)
    audio_raw = dict(body.get("audio", {}))
    known_chains = {r.get("audio_chain") for r in RIGS.values()} - {None}
    chains = _optional_tables(audio_raw.pop("chains", {}), known_chains, "[audio.chains]")
    if name == "ft891":
        optional_keys = ("gain_range", "rx_floor_dbfs", "p1_drive", "p3_drive",
                         "p4_drive", "calibration_status", "calibration_evidence")
        calibration = {key: audio_raw.pop(key) for key in optional_keys if key in audio_raw}
        if calibration:
            checked = _chain_audio(calibration, {}, "ft891_external")
            calibration = {key: getattr(checked, key) for key in calibration}
        audio = build_section(Audio, audio_raw, "[audio]")
        chain = "ft891_external"
    else:
        calibration = {}
        # Check flat spelling without constructing the mobile Audio defaults.
        unknown = set(audio_raw) - set(Audio.__dataclass_fields__)
        if unknown:
            warn_config(f"[audio]: unknown key(s) {', '.join(sorted(unknown))}; ignored")
        chain = selected["audio_chain"]
        audio = _chain_audio(chains.get(chain, {}), audio_raw, chain,
                             device_default=selected.get("audio_device", ""))

    protocols = {}
    selected_name = name
    for name, table in body.get("protocols", {}).items():
        if name not in _PROTOCOL_HOSTS:
            raise ConfigError(
                f"[protocols.{name}]: unknown protocol. "
                f"Known: {', '.join(sorted(_PROTOCOL_HOSTS))}")
        p = build_section(ProtocolConfig, dict(table), f"[protocols.{name}]")
        if not isinstance(p.bbs_directory, str) or '\x00' in p.bbs_directory:
            raise ConfigError(f'[protocols.{name}]: bbs_directory must be a path string without NUL')
        if p.bbs_directory and name != 'vara':
            raise ConfigError(f'[protocols.{name}]: bbs_directory is only supported for vara')
        if p.enabled:
            choice(p.host, _PROTOCOL_HOSTS[name], f"[protocols.{name}]", "host")
            if p.host.endswith('-managed') and not p.auth_token_file:
                raise ConfigError(f'[protocols.{name}]: managed host requires auth_token_file')
            if p.auth_token_file and not p.host.endswith('-managed'):
                raise ConfigError(f'[protocols.{name}]: auth_token_file requires a managed host')
            if p.host.endswith('-managed') and (p.cmd_port or p.data_port):
                raise ConfigError(f'[protocols.{name}]: managed host uses port, not cmd_port/data_port')
            if p.host.endswith('-managed') and p.pty:
                raise ConfigError(f'[protocols.{name}]: managed host cannot expose a pty')
        if name in PROTOCOL_VALIDATORS:
            PROTOCOL_VALIDATORS[name](p, f"[protocols.{name}]", frozenset(table))
        protocols[name] = p

    voice_raw = dict(body.get("voice", {}))
    if "exchange" in voice_raw:
        voice_raw["exchange"] = tuple(voice_raw["exchange"])
    voice = build_section(Voice, voice_raw, "[voice]")
    models = build_section(Models, dict(body.get("models", {})), "[models]")
    cfg = StationConfig(station, rig, audio, protocols, selected_name, rig_source, chain,
                        selected, sources, calibration, voice, models,
                        build_section(Desk, dict(body.get("desk", {})), "[desk]"),
                        build_section(BBSReference, dict(body.get("bbs", {})), "[bbs]"))
    _validate(cfg)
    floor = selected.get("min_watts")
    if cfg.rig.power_clamp_w is not None and floor and cfg.rig.power_clamp_w < floor:
        warn_config(f"[rig] power_clamp_w {cfg.rig.power_clamp_w:g} W is below the "
                    f"{cfg.selected_rig.upper()}'s {floor:g} W minimum; clamping at {floor:g} W")
        cfg = replace(cfg, rig=replace(cfg.rig, power_clamp_w=float(floor)))
    return cfg


def _validate(cfg: StationConfig) -> None:
    st, audio, rig = cfg.station, cfg.audio, cfg.rig
    check_station_rig(st, rig)
    if not isinstance(cfg.desk.adif_dir, str) or "\x00" in cfg.desk.adif_dir:
        raise ConfigError("[desk]: adif_dir must be a path string without NUL characters")

    for name, value in (("models.memory_gb", cfg.models.memory_gb),
                        ("voice.speed", cfg.voice.speed), ("voice.silence_s", cfg.voice.silence_s),
                        ("voice.max_over_s", cfg.voice.max_over_s)):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{name} must be finite and positive")
    for name, value in (("drive", cfg.voice.drive), ("vad_threshold", cfg.voice.vad_threshold)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ConfigError(f"voice.{name} must be finite and between 0 and 1")
    if not isinstance(cfg.voice.preroll_s, (float, int)) or not math.isfinite(cfg.voice.preroll_s) or not 0 <= cfg.voice.preroll_s <= 2:
        raise ConfigError("voice.preroll_s must be between 0 and 2 seconds")
    choice(cfg.voice.id_mode, ("voice",), "[voice]", "id_mode")
    choice(cfg.voice.compression, ("off", "soft", "medium", "strong"), "[voice]", "compression")
    choice(cfg.voice.phonetics, ("nato", "alternate", "varied"), "[voice]", "phonetics")
    choice(cfg.voice.synthesis_backend, ("kokoro", "piper"), "[voice]", "synthesis_backend")
    if not isinstance(cfg.voice.synthesis_speaker, int) or cfg.voice.synthesis_speaker < 0:
        raise ConfigError("voice.synthesis_speaker must be a nonnegative speaker ID")
    if any(field not in ("report", "location", "name") for field in cfg.voice.exchange):
        raise ConfigError("[voice] exchange fields: report, location, name")
    if not isinstance(cfg.voice.voice_agc, int) or cfg.voice.voice_agc not in (1, 2, 3, 4, 5):
        raise ConfigError("[voice] voice_agc must be a hamlib AGC setting from 1 to 5")

    if not st.control:
        raise ConfigError(
            "[station]: control is required and has no default — 'local' if an "
            "operator is at the control point, 'automatic' if the station answers "
            "on its own. The selected profile uses this setting.")
    choice(st.control, tuple(c.value for c in Control), "[station]", "control")

    if not st.regulatory:
        raise ConfigError(
            "[station]: regulatory is required and has no default. 'part97' applies "
            "configured emission limits (add licence = \"general\"|…), or "
            "'unregulated' applies no emission limits (add because = \"…\").")
    cfg.profile  # raises ValueError naming the profile, and validates its settings
    ale = cfg.protocols.get("ale")
    if ale is not None and ale.enabled:
        check_linking_protection(st.regulatory, ale.lp_level)

    choice(st.id_mode, ("cw", "none"), "[station]", "id_mode")
    if st.idcall != "":
        try:
            identification.validate_call(st.idcall, "[station] idcall")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    _check_client_sid(st.client_sid, "[station]")

    if isinstance(audio, Audio):
        if not 0 <= audio.tx_drive <= 1:
            raise ConfigError(f"[audio]: tx_drive {audio.tx_drive} must be between 0 and 1")
        lo, hi = cfg.audio_calibration.get("gain_range", GAIN_RANGE)
        if not lo <= audio.input_gain <= hi and "gain_range" in cfg.audio_calibration:
            warn_config("[audio]: input_gain outside configured gain_range; retaining operator value")
        elif not lo <= audio.input_gain <= hi:
            raise ConfigError(
                f"[audio]: input_gain {audio.input_gain} is outside {lo}–{hi}. The "
                "measured working point at this station is 0.040; it has been wrong "
                "at 0.51, 0.50, 0.18 and 0.118, and a hot input presents as a working "
                "radio that decodes nothing.")

    enabled = {n: p for n, p in cfg.protocols.items() if p.enabled}

    if enabled and not audio.output and isinstance(audio, Audio):
        raise ConfigError(
            "[audio]: output is required when any protocol is enabled. An unset "
            "device means the system default, which is the built-in speakers.")

    seen: dict[int, str] = {st.port: "station"}
    for name, p in enabled.items():
        for port in (p.cmd_port, p.data_port, p.kiss_port, p.port):
            if not port:
                continue
            if port in seen:
                raise ConfigError(
                    f"[protocols.{name}]: port {port} is already used by "
                    + ("[station]" if seen[port] == "station" else f"[protocols.{seen[port]}]"))
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
