# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The rigs this station keys, and the hamlib settings each one needs.

Model IDs verified against the local build (`rigctl --list`, Hamlib 5.0).

One table, because there were three and they disagreed: two of them carried the
same model, baud and mode, one of those two also carried the PTT settle time and
the keying line, and a shell launcher carried a fourth copy of the numbers. A
divergence in a table like this is not a lint problem — it decides how a radio
gets keyed.

**`settle`**, the pause between key-down and the first sample, is per rig because
its cause is per rig. The X6100's USB-audio codec re-initialises when it is keyed,
so pushing samples immediately stalls the stream and the head of the burst never
reaches the modulator; it needs 0.40 s. The FT-891 drives an external dongle that
does not re-initialise, and carrying the X6100's figure over held the key open
with dead air at both ends of every burst, which an operator watching the rig will
rightly call out. What settle buys is nothing and what it costs is keyed time: on
PACTOR's 1.25 s cycle, 0.04 leaves 250 ms of clear air per cycle against a
commercial modem's 288. **It is the first number to revisit if a burst comes back
clipped at the head** — in either direction, since too much is as visible as too
little. Raising the FT-891's to 0.10 does not break shrike's schedule; it narrows
the band of peer turnarounds the cycle can service, and `shrike.onair._budget`
prints that band rather than leaving it to be argued about.

The G90's 0.10 is carried rather than measured; nobody has keyed one here.

**`ptt_type` and `ptt_iface`** say how the rig is keyed. hamlib's `RIG` is a CAT
command; `RTS` is a modem line on a second serial interface, which is one ioctl
rather than a blocking CAT transaction — see `core.ptt` for what that buys and
what it costs. A rig with no `ptt_iface` has no second interface and keys over the
port it already has.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import math

# `rigctl -m 1036 --dump-caps`. Everything here shapes or reports the receive
# chain: PREAMP is IPO, IF is the shift, STRENGTH and RAWSTR are the S-meter.
RIG_LEVELS = ('PREAMP', 'ATT', 'AF', 'RF', 'SQL', 'IF', 'NR', 'NOTCHF', 'AGC', 'NB',
              'MONITOR_GAIN', 'RFPOWER', 'STRENGTH', 'RAWSTR', 'ALC', 'SWR')
RIG_FUNCS = ('NB', 'NR', 'ANF', 'APF', 'MN', 'COMP', 'MON', 'TUNER')
# `--dump-caps` gives these a zero-width range, so the rig answers a constant and
# the number means nothing. AGC is the one the clipping question wants, and it has
# to be read off the radio's own display until hamlib can fetch it.
UNREPORTED_LEVELS = ('AGC', 'METER')
# A level is what a control is set to; a func is whether it is doing anything.
# `NR 0.067` and `NOTCHF 1500` read as "noise reduction and the notch are on" and
# mean nothing of the kind when their switch is off, which is how they were first
# misread here on 2026-09-16.
GOVERNED_BY = {'NR': 'NR', 'NB': 'NB', 'NOTCHF': 'MN', 'COMP': 'COMP', 'MONITOR_GAIN': 'MON'}
# Live readings, not settings: they move with the band and the last transmission.
METERS = ('STRENGTH', 'RAWSTR', 'ALC', 'SWR')
# hamlib reports the FT-891's front end as a PREAMP level. 0 is IPO, the preamp
# bypassed, which is this station's setting; nobody should have to know that to
# read the line.
PREAMP_MEANING = {'0': 'IPO, preamp bypassed', '10': 'AMP1', '20': 'AMP2'}

# The front panel's menu, over CAT: `EXggii;` where gg is the group and ii the item,
# answering `EXggiiVALUE;`. Group 08 is DATA, the whole analogue path into the only
# gain control we have. Value digits are what make it a write, so a bare address is
# a read. Only 08-11 is labelled: the operator read 45 off the LCD and CAT answered
# 045, so that one is confirmed. The rest are recorded unlabelled rather than
# guessed -- three readback numbers were misread in one evening by naming them first
# and checking after.
MENU_GROUP = '08'
MENU_ITEMS = tuple(f'{MENU_GROUP}{item:02d}' for item in range(1, 13))
MENU_LABELS = {
    '0801': 'DATA MODE', '0802': 'PSK TONE', '0803': 'OTHER DISP', '0804': 'OTHER SHIFT',
    '0805': 'DATA LCUT FREQ', '0806': 'DATA LCUT SLOPE', '0807': 'DATA HCUT FREQ',
    '0808': 'DATA HCUT SLOPE', '0809': 'DATA IN SELECT', '0810': 'DATA PTT SELECT',
    '0811': 'DATA OUT LEVEL', '0812': 'DATA BFO',
}
# Decoded from the CAT reference, so a reader is not left holding an index.
MENU_VALUES = {
    '0801': {'0': 'PSK', '1': 'OTHERS'},
    '0805': {'00': 'OFF'}, '0807': {'00': 'OFF'},
    '0809': {'0': 'MIC', '1': 'REAR'},
    '0810': {'0': 'DAKY', '1': 'RTS', '2': 'DTR'},
    '0812': {'0': 'USB', '1': 'LSB'},
}

# The controls that decide whether a wideband packet survives are NOT menu items and
# hamlib does not expose them. NARROW caps DATA at 500 Hz whatever the width knob
# reads, which would pass a 2-carrier signal and amputate an 18-carrier one; AGC is
# the level control hamlib reports as a constant 0; CONTOUR's own default is a 15 dB
# notch inside the passband. All are bare-prefix reads: the value digits are what
# would make them writes.
RAW_READS = (
    ('NA0;', 'NARROW', {'0': 'off', '1': 'ON -- DATA caps at 500 Hz'}),
    ('GT0;', 'AGC', {'0': 'OFF', '1': 'FAST', '2': 'MID', '3': 'SLOW',
                     '4': 'AUTO-FAST', '5': 'AUTO-MID', '6': 'AUTO-SLOW'}),
    # SH0 answers on/off then a two-digit index; 17 is 3000 Hz, the widest DATA offers.
    ('SH0;', 'IF WIDTH', {'117': '3000 Hz, the widest DATA offers', '116': '2400 Hz',
                          '115': '2000 Hz', '114': '1700 Hz', '113': '1400 Hz',
                          '112': '1200 Hz', '111': '800 Hz', '110': '500 Hz'}),
    ('IS0;', 'IF SHIFT', {}),
    ('CO00;', 'CONTOUR', {'0': 'off', '1': 'ON'}),
    ('NR0;', 'DNR', {'0': 'off', '1': 'ON'}),
    ('BC0;', 'AUTO NOTCH', {'0': 'off', '1': 'ON'}),
    ('NB0;', 'NOISE BLANKER', {'0': 'off', '1': 'ON'}),
    ('RA0;', 'ATT', {'0': 'off', '1': 'ON (12 dB)'}),
    ('PA0;', 'IPO', {'0': 'IPO, preamp bypassed', '1': 'AMP'}),
    # 000-030 is the whole range on this rig, not 0-100: 030 is wide open.
    ('RG0;', 'RF GAIN', {'030': 'maximum'}),
)
MODE_CODES = {'1': 'LSB', '2': 'USB', '3': 'CW-U', '4': 'FM', '5': 'AM', '6': 'RTTY-L',
              '7': 'CW-L', '8': 'DATA-L', '9': 'RTTY-U', 'A': 'DATA-FM', 'B': 'FM-N',
              'C': 'DATA-U', 'D': 'AM-N', 'E': 'PSK', 'F': 'DATA-FM-N'}

_FT891_READBACK = {
    'rig_levels': RIG_LEVELS,
    'rig_funcs': RIG_FUNCS,
    'unreported_levels': UNREPORTED_LEVELS,
    'governed_by': GOVERNED_BY,
    'meters': METERS,
    'preamp_meaning': PREAMP_MEANING,
    'menu_group': MENU_GROUP,
    'menu_items': MENU_ITEMS,
    'menu_labels': MENU_LABELS,
    'menu_values': MENU_VALUES,
    'raw_reads': RAW_READS,
    'mode_codes': MODE_CODES,
}

# hamlib addresses this rig's controls by VFO: with VFO-B selected it asks RG1;
# GT1; SQ1; and the FTDX10, which has one receiver, refuses or never answers, so
# RF gain and AGC came back as nothing. Those are read raw with P1 0, the only
# value the CAT reference allows, and hamlib keeps the levels that do not ask.
RIG_LEVELS = ('AF', 'NR', 'NOTCHF', 'NB', 'MONITOR_GAIN', 'RFPOWER',
              'STRENGTH', 'RAWSTR', 'ALC', 'SWR')
RIG_FUNCS = ('NB', 'NR', 'ANF', 'APF', 'MN', 'MON', 'TUNER')
PREAMP_MEANING = {}
# `EX P1P2P3;`, CAT reference Table 2: 01-04-xx is RADIO SETTING, MODE PSK/DATA.
MENU_ITEMS = tuple(f'0104{item}' for item in range(13, 19))
MENU_LABELS = {'010413': 'DATA OUT LEVEL', '010414': 'TX BPF SEL',
               '010415': 'DATA MOD SOURCE', '010416': 'REAR SELECT',
               '010417': 'RPORT GAIN', '010418': 'RPTT SELECT'}
MENU_VALUES = {'010414': {'0': '50-3050 Hz', '1': '100-2900 Hz', '2': '200-2800 Hz',
                          '3': '300-2700 Hz', '4': '400-2600 Hz'},
               '010415': {'0': 'MIC', '1': 'REAR'}, '010416': {'0': 'DATA', '1': 'USB'},
               '010418': {'0': 'DAKY', '1': 'RTS', '2': 'DTR'}}
# hamlib's `m` fails outright on this rig (it has no entry for the width index
# the radio answers), so the mode is read here too, for the VFO that VS names.
RAW_READS = (
    ('VS;', 'VFO', {'0': 'A', '1': 'B'}),
    ('MD0;', 'MODE MAIN', MODE_CODES),
    ('MD1;', 'MODE SUB', MODE_CODES),
    ('SH0;', 'WIDTH', {'021': '4000 Hz in PSK/DATA (CAT Table 3)',
                       '023': 'SSB-only index; unverified in DATA'}),
    ('RF0;', 'R. FIL', {'6': '12 kHz', '7': '3 kHz', '9': '500 Hz', 'A': '300 Hz'}),
    ('NA0;', 'NARROW', {'0': 'off', '1': 'ON'}),
    ('GT0;', 'AGC', {'0': 'OFF', '1': 'FAST', '2': 'MID', '3': 'SLOW',
                     '4': 'AUTO-FAST', '5': 'AUTO-MID', '6': 'AUTO-SLOW'}),
    ('RG0;', 'RF GAIN', lambda value: (f'{int(value) / 255:.3f} of 255'
        if value.isdigit() and 0 <= int(value) <= 255 else None)),
    ('PA0;', 'PREAMP', {'0': 'IPO, preamp bypassed', '1': 'AMP1', '2': 'AMP2'}),
    ('RA0;', 'ATT', {'0': 'off', '1': '6 dB', '2': '12 dB', '3': '18 dB'}),
)
_FTDX10_READBACK = {**_FT891_READBACK,
    'menu_group': '0104',
    'rig_levels': RIG_LEVELS,
    'rig_funcs': RIG_FUNCS,
    'preamp_meaning': PREAMP_MEANING,
    'menu_items': MENU_ITEMS,
    'menu_labels': MENU_LABELS,
    'menu_values': MENU_VALUES,
    'raw_reads': RAW_READS,
}

RIGS = {
    "ft891": {"model": 1036, "baud": 38400, "mode": "PKTUSB", "settle": 0.04,
              "tx_latency_ms": 20.0, "tx_latency_status": "INFERRED",
              "ptt_type": "RTS", "ptt_iface": 1, "tr_switch_s": 0.055,
              "tr_recovery_slack_s": 0.015, "audio_device": "USB Audio Device",
              "cat_port": "/dev/cu.usbserial-00A6C71B0",
              "ptt_port": "/dev/cu.usbserial-00A6C71B1",
              "note": "Yaesu FT-891; set CAT rate to 38400 (menu 05-06); external soundcard on DATA jack"},
    "ftdx10": {"rx_width_hz": 4000, "roofing_hz": 12000, "model": 1042, "baud": 38400, "mode": "PKTUSB", "settle": 0.04,
               "tx_latency_ms": 7.0, "tx_latency_status": "MEASURED",
               "ptt_type": "RTS", "ptt_iface": 1, "tr_switch_s": 0.070,
               "tr_recovery_slack_s": 0.015, "audio_chain": "ftdx10_usb",
               "audio_device": "USB AUDIO  CODEC",
               "ptt_port": "/dev/cu.usbserial-01D13C961",
               "endpoint_status": "CONFIRMED",
               "endpoint_evidence": "2026-10-02 operator confirmation on air: CAT and RTS PTT on these ports keyed the FTDX10 cleanly in two PTT-tail runs, and the levels readback named USB AUDIO  CODEC",
               "cat_port": "/dev/cu.usbserial-01D13C960",
               "note": "Yaesu FTDX10; REAR SELECT=USB, RPTT=RTS; settle 40 ms and TR 70 ms provisional, UNMEASURED"},
    "g90":   {"model": 3088, "baud": 19200, "mode": "PKTUSB", "settle": 0.10, "tr_switch_s": 0.055, "tr_recovery_slack_s": 0.015,
              "tx_latency_ms": 0.0, "tx_latency_status": "UNMEASURED", "audio_chain": "g90_external",
              "note": "Xiegu G90; CAT on its serial, audio via external interface"},
    "x6100": {"model": 3087, "baud": 19200, "mode": "PKTUSB", "settle": 0.40, "tr_switch_s": 0.055, "tr_recovery_slack_s": 0.015,
              "tx_latency_ms": 0.0, "tx_latency_status": "UNMEASURED", "audio_chain": "x6100_usb",
              "note": "Xiegu X6100; single USB gives both CAT serial and a soundcard"},
}


# Nominal backend DATA ladders; physical passband qualification is separate.
FT891_DATA_WIDTHS = (50, 100, 150, 200, 250, 300, 350, 400, 450,
                    500, 800, 1200, 1400, 1700, 2000, 2400, 3000)
FTDX10_DATA_WIDTHS = (0, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                     500, 600, 800, 1200, 1400, 1700, 2000, 2400,
                     3000, 3200, 3500, 4000)
_FTDX10_READBACK['menu_items'] = ('030108', '030109', '030110',
    *(f'0104{i:02d}' for i in range(8, 19)), '030401')
_FTDX10_READBACK['menu_labels'] = {**_FTDX10_READBACK['menu_labels'],
    '030108': 'CAT RATE', '030109': 'CAT TIMEOUT', '030110': 'CAT RTS',
    '010408': 'DATA SHIFT (SSB)', '010409': 'DATA LCUT FREQ',
    '010410': 'DATA LCUT SLOPE', '010411': 'DATA HCUT FREQ',
    '010412': 'DATA HCUT SLOPE', '030401': 'HF MAX POWER'}
_FTDX10_READBACK['raw_reads'] = tuple(
    (cmd, label, {**{f'0{i:02d}': f'{hz} Hz in PSK/DATA (CAT Table 3)'
                    for i, hz in enumerate(FTDX10_DATA_WIDTHS) if hz}, **meanings}
     if label == 'WIDTH' else meanings)
    for cmd, label, meanings in _FTDX10_READBACK['raw_reads'])
for _name, _readback, _widths, _maximum in (
        ('ft891', _FT891_READBACK, FT891_DATA_WIDTHS, 30),
        ('ftdx10', _FTDX10_READBACK, FTDX10_DATA_WIDTHS[1:], 255)):
    RIGS[_name].update(readback=_readback, data_widths=_widths,
                      rf_gain_raw_max=_maximum, physical_data_passband_verified=False)
# Empty rows are intentional: no Yaesu decoders or widths for unqualified rigs.
for _name in ('g90', 'x6100'):
    RIGS[_name].update(readback={key: (() if key in ('rig_levels', 'rig_funcs',
        'unreported_levels', 'meters', 'menu_items', 'raw_reads') else {})
        for key in _FT891_READBACK if key != 'menu_group'}, data_widths=())


def resolve_rig(name=None, *, environ=None, legacy_model=None):
    environ = os.environ if environ is None else environ
    selected = name or environ.get("HFMODEM_RIG") or legacy_model or "ft891"
    if selected not in RIGS:
        raise ValueError(f"unknown rig {selected!r}; choose {', '.join(RIGS)}")
    return selected


PROFILE_OVERRIDES = frozenset({"cat_port", "ptt_port", "settle", "tr_switch_s",
                               "tr_recovery_slack_s", "tx_latency_ms",
                               "calibration_status", "calibration_evidence"})


def resolve_profile(name=None, *, environ=None, legacy_model=None, profiles=None,
                    cat_port=None, ptt_port=None):
    """Pure selected-rig facts and endpoint provenance; never probes a device."""
    environ = os.environ if environ is None else environ
    selected = resolve_rig(name, environ=environ, legacy_model=legacy_model)
    from .config import warn_config
    configured = (profiles or {}).get(selected, {})
    if not isinstance(configured, dict):
        warn_config(f"[rig.profiles.{selected}]: expected table; ignored")
        configured = {}
    configured = dict(configured)
    for key in set(configured) - PROFILE_OVERRIDES:
        warn_config(f"[rig.profiles.{selected}]: unknown key {key}; ignored")
        configured.pop(key)
    defaults = RIGS[selected]
    for key, value in list(configured.items()):
        if key in ("settle", "tr_switch_s", "tr_recovery_slack_s", "tx_latency_ms"):
            valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                     and math.isfinite(value) and value >= 0
                     and (key not in ("settle", "tr_switch_s") or value > 0)
                     and (key != "tx_latency_ms" or value <= 60))
        elif key == "calibration_status":
            valid = value in ("UNMEASURED", "PROVISIONAL", "INFERRED", "MEASURED")
        else:
            valid = isinstance(value, str)
        if not valid:
            warn_config(f"[rig.profiles.{selected}] invalid {key}; using rig-profile default")
            configured.pop(key)
    if (configured.get("calibration_status") == "MEASURED"
            and not configured.get("calibration_evidence")):
        warn_config(f"[rig.profiles.{selected}] MEASURED requires calibration_evidence; treating as PROVISIONAL")
        configured["calibration_status"] = "PROVISIONAL"
    values = {**defaults, **configured}
    if "tx_latency_ms" in configured:
        values["tx_latency_status"] = configured.get("calibration_status", "PROVISIONAL")
    sources = {key: "rig-profile" for key in values}
    sources.update({key: "station-toml" for key in configured})
    for key, explicit, envkey in (("cat_port", cat_port, "CAT_PORT"),
                                  ("ptt_port", ptt_port, "PTT_PORT")):
        if explicit:
            values[key], sources[key] = explicit, "operator-flag"
        elif environ.get(envkey):
            values[key], sources[key] = environ[envkey], envkey
    return selected, values, sources


TX_COMMAND_PATHS = ("generic", "flown", "p2-mail", "p4-bench")
P4_FT891_LOGICAL_OFFSET_MS = 17.0


@dataclass(frozen=True, slots=True)
class TxTimingPlan:
    rig: str
    chain: str
    tx_latency_ms: float
    latency_source: str
    calibration_status: str
    command_path: str
    latency_samples: int
    geometry_samples: int
    compatibility_identity: bool
    p4_answer_residual_ms: float = 0.0
    p4_answer_residual_samples: int = 0
    p4_answer_residual_evidence: str = ""
    p4_audio_input: str = ""
    p4_audio_output: str = ""
    p4_answer_base_tx_latency_ms: float | None = None


def resolve_tx_timing(override=None, *, rig=None, station_path=None,
                      command_path="generic", fs=48000, environ=None):
    from . import config

    environ = os.environ if environ is None else environ
    if command_path not in TX_COMMAND_PATHS:
        raise ValueError(f"unknown TX command path {command_path!r}")
    path = station_path if station_path is not None else environ.get(config.STATION_ENV)
    audio = None
    if path:
        try:
            station_cfg = config.load(path, rig_name=rig, environ=environ)
            audio = station_cfg.audio
            rig = station_cfg.selected_rig
        except (OSError, ValueError) as exc:
            raise config.ConfigError(f"{config.STATION_ENV} names {path}: {exc}") from exc
    else:
        rig = resolve_rig(rig, environ=environ)
    empty_input = "default input" if rig == "ft891" else "unresolved input"
    empty_output = "default output" if rig == "ft891" else "unresolved output"
    chain = (f"{path}: {audio.input or empty_input} -> "
             f"{audio.output or empty_output}" if audio else
             f"{rig}: {RIGS[rig].get('audio_device', RIGS[rig].get('audio_chain', 'default audio'))}")
    if override is not None:
        value, source, status = override, "operator", "OPERATOR OVERRIDE"
    elif audio is not None and audio.tx_latency_ms is not None:
        value, source, status = (audio.tx_latency_ms, "station-toml",
                                 "CHAIN OVERRIDE" if rig == "ft891" else
                                 audio.calibration_status if audio.calibration_evidence else "PROVISIONAL")
    else:
        profile = station_cfg.resolved_profile if audio is not None else RIGS[rig]
        value, source, status = (profile["tx_latency_ms"],
                                 station_cfg.resolution_sources.get("tx_latency_ms", "rig-profile")
                                 if audio is not None else "rig-profile",
                                 profile["tx_latency_status"])
    lo, hi = config.TX_LATENCY_RANGE_MS
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise config.ConfigError(f"tx_latency_ms {value} is outside finite {lo}–{hi} ms")
    residual = getattr(audio, "p4_answer_residual_ms", 0.0)
    reference_latency = ((audio.tx_latency_ms if audio.tx_latency_ms is not None
                          else station_cfg.resolved_profile["tx_latency_ms"])
                         if residual else None)
    compatible = rig == "ft891" and source == "rig-profile" and value == 20
    geometry = (round((P4_FT891_LOGICAL_OFFSET_MS if command_path == "p4-bench"
                       else 20.0 if command_path == "generic" else 0.0) * fs / 1000)
                if compatible else 0)
    return TxTimingPlan(rig, chain, float(value), source, status, command_path,
                        round(value * fs / 1000), geometry, compatible,
                        float(getattr(audio, "p4_answer_residual_ms", 0.0)),
                        round(getattr(audio, "p4_answer_residual_ms", 0.0) * fs / 1000),
                        getattr(audio, "p4_answer_residual_evidence", ""),
                        getattr(audio, "input", ""), getattr(audio, "output", ""), reference_latency)


def launcher_defaults(args, *, rig_attr="rig", audio_fields=None,
                      endpoint_fields=None, drive_fields=(), legacy=None):
    """Fill absent launcher flags from the selected chain, without hardware I/O.

    Callers supply their historical FT-891 defaults: entry points intentionally
    differ. FT-891 endpoints remain parser/caller defaults: this bridge must not
    add environment or TOML sources those entry points historically ignored.
    Explicit flags (including zero) always survive this bridge.
    """
    from . import config
    path = os.environ.get(config.STATION_ENV)
    requested = getattr(args, rig_attr, None)
    cfg = config.load(path, rig_name=requested) if path else None
    name = cfg.selected_rig if cfg else resolve_rig(requested)
    profile = cfg.resolved_profile if cfg else RIGS[name]
    setattr(args, rig_attr, name)
    for field, direction in (audio_fields or {}).items():
        if getattr(args, field, None) is None:
            value = (getattr(cfg.audio, direction) if cfg and name != "ft891" else None)
            setattr(args, field, value or ((legacy or {}).get(field) if name == "ft891"
                                         else profile.get("audio_device", "")))
    for field, endpoint in (endpoint_fields or {}).items():
        if name != "ft891" and getattr(args, field, None) is None:
            value = os.environ.get("CAT_PORT" if endpoint == "cat_port" else "PTT_PORT")
            setattr(args, field, value or profile.get(endpoint))
    for field in drive_fields:
        if getattr(args, field, None) is None and name != "ft891":
            setattr(args, field, cfg.audio.drive_for() if cfg else config.ChainAudio().drive_for())
    for field in ("model", "baud"):
        if hasattr(args, field) and getattr(args, field) is None:
            setattr(args, field, profile[field])
    if hasattr(args, "expect_model") and args.expect_model is None and name != "ft891":
        args.expect_model = "FTDX-10" if name == "ftdx10" else name.upper()
    for field, value in (legacy or {}).items():
        if getattr(args, field, None) is None and name == "ft891":
            setattr(args, field, value)
    return args


def model_identity_matches(expected, actual):
    """Hamlib may include punctuation in a model's human-readable identity."""
    normalize = lambda text: ''.join(char for char in text.casefold() if char.isalnum())
    return normalize(expected) in normalize(actual)


def flight_defaults(args):
    path = os.environ.get("HFMODEM_STATION")
    station_cfg = None
    if path:
        from . import config
        station_cfg = config.load(path, rig_name=args.rig)
        args.rig = station_cfg.selected_rig
    else:
        args.rig = resolve_rig(args.rig)
    profile = station_cfg.resolved_profile if station_cfg else RIGS[args.rig]
    args.serial = args.serial or os.environ.get("CAT_PORT") or (
        profile.get("cat_port") if station_cfg or args.rig == "ftdx10" else None)
    if hasattr(args, "ptt_port"):
        args.ptt_port = args.ptt_port or os.environ.get("PTT_PORT") or (
            profile.get("ptt_port") if station_cfg or args.rig == "ftdx10" else None)
    if args.rig in ("ftdx10", "ft891"):
        args.audio_in = args.audio_in or (station_cfg.audio.input if station_cfg else None) or profile.get("audio_device")
        args.audio_out = args.audio_out or (station_cfg.audio.output if station_cfg else None) or profile.get("audio_device")
    return args


if __name__ == "__main__":
    path = os.environ.get("HFMODEM_STATION")
    if path:
        from . import config
        station_cfg = config.load(path)
        name, profile = station_cfg.selected_rig, station_cfg.resolved_profile
    else:
        name, profile = resolve_rig(), RIGS[resolve_rig()]
    print(name)
    for field in ("model", "baud", "mode"):
        print(profile.get(field, ""))
    print(station_cfg.audio.input if path else profile.get("audio_device", ""))
    for field in ("cat_port", "ptt_port"):
        value = profile.get(field, "") if name != "ft891" or (path and
                station_cfg.resolution_sources.get(field) != "rig-profile") else ""
        print(value)
    print(station_cfg.audio.output if path else profile.get("audio_device", ""))
