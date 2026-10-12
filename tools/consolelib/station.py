"""CHECK THE RIG: what this station is set to, read back and said in one voice.

Nothing here writes to the radio. The console reads the whole rig in one CAT
session, prints it the same way whether an arm is flying or the band is quiet,
and records it into every arm's provenance -- none of it is recoverable from the
audio afterwards. The one thing it will write, and only when asked, is the codec
gain, which is the only gain in the whole receive chain.
"""
from __future__ import annotations

import array
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import wave

from . import power, roster
from hfmodem.core import config as station_config
from hfmodem.core.rigs import resolve_profile

ROOT = Path(__file__).resolve().parents[2]
ONAIR = ROOT / 'tools/onair.sh'
KIWI = ROOT / 'tools/kiwi_witness.py'
CODEC_GAIN = ROOT / 'tools/codec_gain.py'
TXWITNESS = ROOT / 'tools/txwitness.py'
VENV = Path(os.environ.get('HFMODEM_PYTHON', str(ROOT / '.venv/bin/python')))

TAP_DEVICE = 'CUBILUX HLMS-C4 Line IN'

HAMLIB_BIN = Path(os.environ.get('HAMLIB_BIN', Path.home() / 'src/radio/hamlib/_install/bin'))


def resolve_selection(rig=None, *, environ=None, allow_missing=False):
    """Resolve one call's rig, endpoints, audio and tables without hardware I/O."""
    environ = os.environ if environ is None else environ
    path = environ.get(station_config.STATION_ENV)
    cfg = station_config.load(path, rig_name=rig, environ=environ) if path and (not allow_missing or Path(path).is_file()) else None
    name, profile = ((cfg.selected_rig, cfg.resolved_profile) if cfg else
                     resolve_profile(rig, environ=environ)[:2])
    cat_port = environ.get('CAT_PORT') or profile.get('cat_port', '')
    ptt_port = environ.get('PTT_PORT') or profile.get('ptt_port', '')
    audio_in = cfg.audio.input if cfg else profile.get('audio_device', '')
    audio_out = cfg.audio.output if cfg else profile.get('audio_device', '')
    rigctl = (str(HAMLIB_BIN / 'rigctl') if (HAMLIB_BIN / 'rigctl').is_file() else 'rigctl',
              '-m', str(profile['model']), '-r', cat_port, '-s', str(profile['baud']))
    return SimpleNamespace(RIG_NAME=name, RIG_PROFILE=profile, CAT_PORT=cat_port,
                           PTT_PORT=ptt_port, AUDIO_IN=audio_in, AUDIO_OUT=audio_out,
                           AUDIO_DEVICE=audio_in, RIGCTL=rigctl, _STATION_CONFIG=cfg,
                           **{key.upper(): value for key, value in profile['readback'].items()})


def warn_receive_only(selection):
    """One line when the selected station file says it does not transmit. The
    operator keys this station, so it is said and never refused."""
    cfg = selection._STATION_CONFIG
    if cfg is not None and not cfg.station.transmit:
        print('warning: the station file says [station] transmit = false, and this keys '
              'the transmitter. Going ahead on your word.', file=sys.stderr, flush=True)


def selection_for_plan(plan):
    environ = dict(os.environ, **plan.get('env', {}))
    # The launcher receives the conventional station-file path even when no
    # file exists. Readback historically works without that optional file.
    path = environ.get(station_config.STATION_ENV)
    if (not os.environ.get(station_config.STATION_ENV)
            and path == str(ROOT / 'station.toml') and not Path(path).is_file()):
        environ.pop(station_config.STATION_ENV)
    return resolve_selection(environ=environ)


# Keep the established module attributes as the default console selection.
# Explicit arms resolve separately and pass their selection through every read;
# they must never mutate these defaults. Reload also replaces all attributes,
# including ones temporarily overridden by a console consumer.
_DEFAULT_FIELDS = ('RIG_NAME', 'RIG_PROFILE', 'CAT_PORT', 'PTT_PORT', 'AUDIO_IN',
                   'AUDIO_OUT', 'AUDIO_DEVICE', 'RIGCTL', '_STATION_CONFIG',
                   'RIG_LEVELS', 'RIG_FUNCS', 'UNREPORTED_LEVELS', 'GOVERNED_BY',
                   'METERS', 'PREAMP_MEANING', 'MENU_GROUP', 'MENU_ITEMS',
                   'MENU_LABELS', 'MENU_VALUES', 'RAW_READS', 'MODE_CODES')
_initial_selection = resolve_selection(allow_missing=True)
globals().update({name: getattr(_initial_selection, name, '')
                  for name in _DEFAULT_FIELDS})
del _initial_selection


def default_selection():
    return SimpleNamespace(**{name: globals()[name] for name in _DEFAULT_FIELDS})


DIAL_TOLERANCE_HZ = 20
PEAK_TARGET_DBFS = -12.0
GAIN_FLOOR_DBFS = -45.0


def cat(*words, joined=False, selection=None):
    """Read the rig. Nothing here writes; the arm's own launcher does the tuning.

    `m` answers on two lines, mode then passband, so a reader that keeps only the
    first silently drops the filter width -- which is the one receive setting the
    readback file already claimed to hold.
    """
    selection = selection or default_selection()
    try:
        result = subprocess.run([*selection.RIGCTL, *words], capture_output=True, text=True, timeout=4)
    except (subprocess.SubprocessError, OSError):
        return None                      # a busy port is not worth an arm
    if result.returncode != 0 or not result.stdout.strip():
        return None
    lines = result.stdout.strip().splitlines()
    return ' '.join(lines) if joined else lines[0]


NO_CONTROL = 'no volume control'


def codec_capability(device=None, scope='input', requested=None, *, selection=None):
    selection = selection or default_selection()
    selected = device or (selection.AUDIO_IN if scope == 'input' else selection.AUDIO_OUT)
    if not selected:
        return dict(status='error', device=selected, scope=scope, value=None,
                    writable=None, error='audio device is unresolved')
    argv = [str(VENV), str(CODEC_GAIN), '--device', selected, '--json', '--scope', scope]
    if requested is not None:
        argv += ['--set', str(requested)]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        if data.get('status') not in ('supported', 'unsupported', 'error'):
            raise ValueError('invalid capability status')
        if data['status'] == 'supported' and (
                not isinstance(data.get('value'), (int, float))
                or not math.isfinite(data['value']) or not 0 <= data['value'] <= 1
                or not isinstance(data.get('writable'), bool)):
            raise ValueError('invalid supported capability scalar or writability')
        return data
    except (OSError, subprocess.SubprocessError, ValueError, AttributeError) as exc:
        return dict(status='error', device=selected, scope=scope, value=None,
                    writable=None, error=str(exc))


def codec_level(device=None, scope='input', *, selection=None):
    """The device's volume scalar, NO_CONTROL when it has none, None when unread."""
    selection = selection or default_selection()
    if selection.RIG_NAME != 'ft891':
        result = codec_capability(device, scope, selection=selection)
        if result['status'] == 'unsupported':
            return NO_CONTROL
        return result.get('value') if result['status'] == 'supported' else None
    result = subprocess.run([str(VENV), str(CODEC_GAIN), '--device', device or selection.AUDIO_DEVICE,
                             '--scope', scope], capture_output=True, text=True, timeout=30)
    found = re.search(r'gain = ([0-9.]+)', result.stdout)
    if found:
        return float(found.group(1))
    return NO_CONTROL if 'NO settable' in result.stdout else None


def read_codec_gain(device=None, *, selection=None):
    selection = selection or default_selection()
    level = codec_level(device, selection=selection)
    return level if isinstance(level, float) else None


def codec_gains(*, selection=None):
    """Both converters. The tap is a separate ADC and carries its own scalar."""
    selection = selection or default_selection()
    if selection.RIG_NAME != 'ft891':
        return {selection.AUDIO_IN: read_codec_gain(selection.AUDIO_IN, selection=selection)} if selection.AUDIO_IN else {}
    return {selection.AUDIO_DEVICE: read_codec_gain(selection.AUDIO_DEVICE, selection=selection),
            TAP_DEVICE: read_codec_gain(TAP_DEVICE, selection=selection)}


def codec_levels(*, selection=None):
    """The FTDX10's own codec, both ways: what reaches us and what drives the rig."""
    selection = selection or default_selection()
    return {scope: codec_level(device, scope, selection=selection) for scope, device in
            (('input', selection.AUDIO_IN), ('output', selection.AUDIO_OUT))}


def gain_advice(peak_dbfs, floor_dbfs, gain):
    """What scalar puts the loudest thing heard at PEAK_TARGET_DBFS.

    The rig runs with AGC off and IPO on (operator, 2026-09-16) and nothing in
    software writes a receive level, so this scalar is the only gain in the whole
    chain. Setting it against the noise floor, as the per-band working points do,
    sets it against the one thing that never clips: what decides clipping is the
    strongest station present, which is not in the window when an arm calibrates.
    Hence a peak target with real headroom, and a floor check so backing off does
    not bury the band noise in the converter's own.
    """
    if peak_dbfs is None or gain in (None, 0):
        return None
    move_db = PEAK_TARGET_DBFS - peak_dbfs
    wanted = min(1.0, gain * 10 ** (move_db / 20))
    floor_after = None if floor_dbfs is None else floor_dbfs + move_db
    return dict(gain_now=gain, peak_dbfs=peak_dbfs, floor_dbfs=floor_dbfs,
                headroom_db=-peak_dbfs, move_db=move_db, gain_wanted=round(wanted, 4),
                floor_after_dbfs=floor_after,
                floor_would_be_too_low=floor_after is not None and floor_after < GAIN_FLOOR_DBFS)


def spell_advice(advice):
    if advice is None:
        return ['  no level advice: the codec gain or the capture did not read']
    lines = [f'  loudest {advice["peak_dbfs"]:.1f} dBFS leaves {advice["headroom_db"]:.1f} dB '
             f'of headroom; band noise {advice["floor_dbfs"]:.1f} dBFS',
             f'  to put the loudest at {PEAK_TARGET_DBFS:.0f} dBFS: gain '
             f'{advice["gain_now"]:.3f} -> {advice["gain_wanted"]:.4f} '
             f'({advice["move_db"]:+.1f} dB), noise would sit at '
             f'{advice["floor_after_dbfs"]:.1f} dBFS']
    if advice['floor_would_be_too_low']:
        lines.append(f'  that would drop the band noise under {GAIN_FLOOR_DBFS:.0f} dBFS, '
                     'where the converter starts to matter. Prefer the rig\'s own level.')
    return [line for line in lines if line]


def menu_number(item):
    """`0811` as the front panel prints it, 08-11; `010413` as 01-04-13."""
    return '-'.join(item[at:at + 2] for at in range(0, len(item), 2))


def spell_environment(env, band_m=None, working_point=None, *, selection=None):
    """The station's state as lines. One writer, so `rig` and an arm read alike.

    An operator comparing a quiet-band dump against what an arm printed needs the
    two to be the same text, or the comparison is between two formatters.
    """
    selection = selection or default_selection()
    gain = env.get('codec_gains', {}).get(selection.AUDIO_DEVICE, env.get('codec_gain'))
    tapgain = env.get('codec_gains', {}).get(TAP_DEVICE)
    rig = env.get('rig') or {}
    levels, funcs = rig.get('levels') or {}, rig.get('funcs') or {}
    def spelt(key, value):
        if key in selection.UNREPORTED_LEVELS:
            return f'{key} {value} (not readable on this rig)'
        if key == 'PREAMP':
            return f'PREAMP {value} ({selection.PREAMP_MEANING.get(value, "unknown setting")})'
        if key == 'ATT' and value in ('0', '0.0'):
            return 'ATT 0 (off)'
        switch = selection.GOVERNED_BY.get(key)
        if switch and switch in funcs:
            return f'{key} {value} ({"ON" if funcs[switch] == "1" else "off"})'
        return f'{key} {value}'

    said = ', '.join(spelt(k, v) for k, v in levels.items()
                     if v is not None and k not in selection.METERS)
    meters = ', '.join(f'{k} {v}' for k, v in levels.items() if v is not None and k in selection.METERS)
    on = [k for k, v in funcs.items() if v == '1' and k not in selection.GOVERNED_BY.values()]
    codec = env.get('codec_levels')
    level = lambda value: ('unread' if value is None else value if value == NO_CONTROL
                           else f'{value:.3f}')
    lines = [(f'  codec input {level(codec.get("input"))}, output {level(codec.get("output"))}'
              if codec else f'  codec gain reads {gain if gain is None else f"{gain:.3f}"}')
             + (f', {band_m} m has worked at {working_point}' if band_m and working_point
                else f', no {band_m} m reading is on file for this hour' if band_m else '')
             + '; the console never sets it.',
             f'  tap gain reads {tapgain if tapgain is None else f"{tapgain:.3f}"} '
             f'on {TAP_DEVICE}; it is a separate converter with its own scale.',
             f'  rig reads {said or "nothing"}',
             f'  rig meters: {meters or "none"}',
             f'  other functions on: {", ".join(on) or "none"}']
    shift = [f'{name} {rig[key]}' for name, key in (('RIT', 'rit'), ('XIT', 'xit'))
             if rig.get(key) not in (None, '0')]
    if rig.get('split') not in (None, '0'):
        shift.append(f'SPLIT on, TX on {rig.get("tx_vfo") or "?"}')
    lines.append('  receive and transmit are on the dial: RIT 0, XIT 0, no split'
                 if not shift and rig.get('rit') is not None else
                 f'  *** {", ".join(shift)} -- the dial is not where we listen or key ***'
                 if shift else '  RIT/XIT/split did not read')
    raw = rig.get('raw') or {}
    if raw:
        said_raw = []
        for _, label, meanings in selection.RAW_READS:
            if label not in raw:
                continue
            value = raw[label]
            if value is None:
                said_raw.append(f'{label} unverified')
                continue
            meaning = (meanings(value) if callable(meanings) else
                       meanings.get(value) or meanings.get(value.lstrip('0') or '0'))
            said_raw.append(f'{label} {value}' + (f' ({meaning})' if meaning else ''))
        lines.append('  ' + '; '.join(said_raw))
    menu = rig.get('menu') or {}
    if menu:
        def menu_said(key, value):
            name = selection.MENU_LABELS.get(key, menu_number(key))
            meaning = (selection.MENU_VALUES.get(key) or {}).get(value)
            return f'{name} {value}' + (f' ({meaning})' if meaning else '')

        lines.append('  menu: ' + ', '.join(menu_said(k, v) for k, v in sorted(menu.items())
                                            if v is not None))
    if rig.get('unread'):
        lines.append(f'  *** some controls did not read ({rig["unread"]}); '
                     'the port was busy or the rig answered short ***')
    if 'rfpower' in env:
        lines.append(f'  RFPOWER reads {env["rfpower"] or "unreadable"}; '
                     'only --p1-watts/--p3-watts write it')
    if rig.get('dial_hz') or rig.get('mode'):
        lines.append(f'  dial {rig.get("dial_hz") or "unreadable"}, '
                     f'mode {rig.get("mode") or "unreadable"}, ptt {rig.get("ptt") or "?"}')
    return lines


def listen(seconds, device=None):
    """Record the band and return its peak and its floor, in dBFS. Writes nothing."""
    selection = default_selection()
    out = Path(tempfile.gettempdir()) / f'level-{int(time.time())}.wav'
    result = subprocess.run(
        [str(VENV), str(TXWITNESS), 'record', str(seconds), str(out),
         '--device', device or selection.AUDIO_DEVICE],
        capture_output=True, text=True, timeout=seconds + 120)
    from hfmodem.core import wav

    out = wav.stored(out)
    if not out.is_file():
        raise RuntimeError(f'the card gave nothing back in {seconds:g} s: '
                           f'{result.stdout.strip() or result.stderr.strip()}')
    try:
        with wave.open(wav.source(out), 'rb') as w:
            frames, width = w.getnframes(), w.getsampwidth()
            raw = w.readframes(frames)
        kind = {2: 'h', 4: 'i'}[width]
        full = float(2 ** (8 * width - 1))
        data = array.array(kind)
        data.frombytes(raw)
        peak = max(abs(v) for v in data) / full
        rms = (sum(v * v for v in data) / len(data)) ** 0.5 / full
    finally:
        out.unlink(missing_ok=True)
    to_db = lambda v: 20 * math.log10(v) if v > 0 else None
    return to_db(peak), to_db(rms)


def gain_report(seconds, apply, device):
    """Measure the band and say what scalar the loudest thing heard wants."""
    selection = default_selection()
    device = device or selection.AUDIO_DEVICE
    if not device:
        raise ValueError(f"[audio.chains.{selection.RIG_PROFILE.get('audio_chain', '')}] input is required for capture")
    capability = codec_capability(device) if selection.RIG_NAME != 'ft891' else None
    if capability and (capability['status'] != 'supported' or not capability['writable']):
        print(f"  codec gain {capability['status']}: {capability.get('error') or 'no writable volume control'}")
        return 1 if capability['status'] == 'error' else 0
    gain = capability['value'] if capability else read_codec_gain(device)
    print(f'  listening {seconds:g} s on {device}, gain reads '
          f'{gain if gain is None else f"{gain:.3f}"}', flush=True)
    peak, floor = listen(seconds, device)
    advice = gain_advice(peak, floor, gain)
    for line in spell_advice(advice):
        print(line)
    if not apply or advice is None:
        print('  nothing was written. Pass --set to apply it.' if advice else '')
        return 0
    if advice['floor_would_be_too_low']:
        print('  NOT SET: see above.')
        return 0
    if selection.RIG_NAME != 'ft891':
        result = codec_capability(device, requested=advice['gain_wanted'])
        print(f"  requested {advice['gain_wanted']:.4f}; observed {result.get('value')}; {result['status']}")
        return 0 if result['status'] == 'supported' else 1
    subprocess.run([str(VENV), str(CODEC_GAIN), '--device', device,
                    '--set', str(advice['gain_wanted'])], check=True, timeout=30)
    print(f'  set {device} to {advice["gain_wanted"]:.4f}; '
          f'it reads {read_codec_gain(device)}', flush=True)
    return 0


def rig_report(mode, as_json, against, watch, band_m=None):
    """The same readback an arm takes, on a quiet band, so drift has a baseline."""
    selection = default_selection()
    first = read_environment()
    if as_json:
        print(json.dumps(first, indent=2))
    else:
        now = datetime.now(timezone.utc)
        print(f'{now.isoformat(timespec="seconds")}  {selection.AUDIO_DEVICE}')
        for line in spell_environment(first, band_m,
                                      band_m and selection.RIG_NAME == 'ft891' and roster.gain_hint(mode, band_m, now)):
            print(line)
    if against:
        record = json.loads((Path(against) if Path(against).is_file()
                             else mode.arms / against / 'provenance.json').read_text())
        for label in ('readings', 'readings_at_end'):
            if label in record:
                changed = differences(record[label], first)
                print(f'  vs {record["arm"]} {label}: '
                      + (', '.join(changed) if changed else 'no change'))
    while watch:
        time.sleep(watch)
        now = read_environment()
        changed = differences(first, now)
        stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
        print(f'{stamp}  ' + (', '.join(changed) if changed else 'no change'), flush=True)
        first = now if changed else first
    return 0


def differences(before, after):
    """Every readback that moved, named `WHERE.WHAT old -> new`."""
    selection = default_selection()
    said = []
    for where in ('codec_gains', 'codec_levels'):
        for key in sorted(set(before.get(where) or {}) | set(after.get(where) or {})):
            a, b = (before.get(where) or {}).get(key), (after.get(where) or {}).get(key)
            if a != b:
                said.append(f'{key} {a} -> {b}')
    for where in ('levels', 'funcs'):
        a_all = (before.get('rig') or {}).get(where) or {}
        b_all = (after.get('rig') or {}).get(where) or {}
        for key in sorted(set(a_all) | set(b_all)):
            if a_all.get(key) != b_all.get(key):
                said.append(f'{key} {a_all.get(key)} -> {b_all.get(key)}')
    a_all = (before.get('rig') or {}).get('menu') or {}
    b_all = (after.get('rig') or {}).get('menu') or {}
    for key in sorted(set(a_all) | set(b_all)):
        if a_all.get(key) != b_all.get(key):
            said.append(f'{selection.MENU_LABELS.get(key, key)} {a_all.get(key)} -> {b_all.get(key)}')
    for key in ('dial_hz', 'mode', 'ptt'):
        a, b = (before.get('rig') or {}).get(key), (after.get('rig') or {}).get(key)
        if a != b:
            said.append(f'{key} {a} -> {b}')
    return said


def read_environment(*, selection=None):
    """The station's own state, read at the start of every arm and again at the end.

    A later reader cannot recover any of this from the audio, and until 2026-09-16
    no arm recorded it: three separate investigations into unread packets had to
    reason around the receive chain because nobody had written down what it was
    set to.
    """
    selection = selection or default_selection()
    try:
        if selection.RIG_NAME == 'ftdx10':
            capabilities = {scope: codec_capability(device, scope, selection=selection) for scope, device in
                            (('input', selection.AUDIO_IN), ('output', selection.AUDIO_OUT))}
            levels = {scope: (NO_CONTROL if c['status'] == 'unsupported' else
                             c.get('value') if c['status'] == 'supported' else None)
                      for scope, c in capabilities.items()}
            return dict(codec_gains={selection.AUDIO_IN: levels['input'] if isinstance(levels['input'], float) else None}
                        if selection.AUDIO_IN else {}, codec_levels=levels,
                        codec_capabilities=capabilities, rig=rig_settings(selection=selection))
        return dict(codec_gains=codec_gains(selection=selection), rig=rig_settings(selection=selection))
    except Exception as trouble:        # a readback may never cost an arm its record
        return dict(error=f'{type(trouble).__name__}: {trouble}')


def rig_settings(*, selection=None):
    """Every control this rig will report, receive and transmit, in one CAT session.

    One `rigctl` with the whole list on stdin, not one process per control: the port
    is shared with the arm itself, and twenty-four opens are twenty-four chances to
    collide. That is not hypothetical -- reading them one at a time timed out inside
    a live arm's teardown on 2026-09-16 and cost it its record.

    Levels and funcs come back as bare `Level Value:` / `Func:` lines in the order
    asked, so a short reply would silently shift every reading onto the wrong name.
    Hence the count check: on any mismatch the whole group reads unread rather than
    wrong. Names are from `rigctl -m 1036 --dump-caps`.
    """
    selection = selection or default_selection()
    script = ''.join(f'{c}\n' for c in
                     ['f', 'm', 't', 'j', 'z', 's',
                      *(f'l {name}' for name in selection.RIG_LEVELS),
                      *(f'u {name}' for name in selection.RIG_FUNCS),
                      *(f'w EX{item};' for item in selection.MENU_ITEMS),
                      *(f'w {command}' for command, _, _ in selection.RAW_READS), 'q'])
    try:
        result = subprocess.run(list(selection.RIGCTL), input=script, capture_output=True,
                                text=True, timeout=25)
    except (subprocess.SubprocessError, OSError) as trouble:
        return dict(levels={}, funcs={}, unread=f'{type(trouble).__name__}')
    said = {}
    values, flags = [], []
    for line in result.stdout.splitlines():
        label, _, value = line.partition(':')
        value = value.strip()
        if label == 'Level Value':
            values.append(value)
        elif label == 'Func':
            flags.append(value)
        elif label in ('Frequency', 'Mode', 'Passband', 'PTT', 'RIT', 'XIT', 'Split', 'TX VFO'):
            said[label] = value
    replies = ' '.join(re.findall(r'Reply: (.*)', result.stdout))
    def menu_payload(item):
        matches = re.findall(rf'\bEX{item}([^\s;]+);', replies)
        return matches[0] if len(matches) == 1 else None
    menu = ({item: (found.group(1) if (found := re.search(rf'EX{item}(\S+?);', replies))
                    else None) for item in selection.MENU_ITEMS} if selection.RIG_NAME == 'ft891' else
            {item: menu_payload(item) for item in selection.MENU_ITEMS})
    silent = [menu_number(item) for item, value in menu.items() if value is None]
    raw = {}
    raw_unread = []
    for command, label, _ in selection.RAW_READS:
        prefix = command.rstrip(';')
        if selection.RIG_NAME == 'ft891':
            found = re.search(rf'\b{prefix}(\S*?);', replies)
            if found and found.group(1):
                raw[label] = found.group(1)
        else:
            matches = re.findall(rf'\b{prefix}([^\s;]*);', replies)
            if len(matches) == 1 and matches[0]:
                raw[label] = matches[0]
            else:
                raw[label] = None
                reason = 'ambiguous replies' if len(matches) > 1 else 'did not answer'
                raw_unread.append(f'{label} ({command}) {reason}')
    levels = (dict(zip(selection.RIG_LEVELS, values)) if len(values) == len(selection.RIG_LEVELS) else {})
    funcs = (dict(zip(selection.RIG_FUNCS, flags)) if len(flags) == len(selection.RIG_FUNCS) else {})
    mode = ' '.join(x for x in (said.get('Mode'), said.get('Passband')) if x)
    operating = raw.get('MODE SUB' if raw.get('VFO') == '1' else 'MODE MAIN')
    if not mode and operating:
        mode = f'{selection.MODE_CODES.get(operating, operating)} (read raw; hamlib gave none)'
    return dict(levels=levels, funcs=funcs, menu=menu, raw=raw,
                dial_hz=said.get('Frequency'), mode=mode or None, ptt=said.get('PTT'),
                rit=said.get('RIT'), xit=said.get('XIT'),
                split=said.get('Split'), tx_vfo=said.get('TX VFO'),
                unread='; '.join(
                    ([] if levels and funcs else
                     [f'{len(values)}/{len(selection.RIG_LEVELS)} levels, {len(flags)}/{len(selection.RIG_FUNCS)} funcs'])
                    + ([f'menu {", ".join(silent)} did not answer'] if silent else [])
                    + raw_unread) or None)


# Exact settings already named and decoded by the shared Yaesu readback tables.
# Output level and audio gain remain provisional; they are not guessed here.
_MAIL_DATA_SETTINGS = {
    'ftdx10': {'010414': '0', '010415': '1', '010416': '1', '010418': '1'},
    'ft891': {'0801': '1', '0805': '00', '0807': '00', '0809': '1', '0810': '1', '0812': '0'},
}


def prepare_radio(plan):
    """Under the arm lease, configure data routing and verify it before keying."""
    selection = selection_for_plan(plan)
    cfg = selection._STATION_CONFIG
    if cfg is None or selection.RIG_NAME not in _MAIL_DATA_SETTINGS:
        raise ValueError('Guided radio preparation requires a saved FTDX10 or FT-891 station.')
    settings = _MAIL_DATA_SETTINGS[selection.RIG_NAME]
    print(f'Preparing {selection.RIG_NAME.upper()} data audio and {cfg.rig.power_w:g} W power…', flush=True)
    commands = ['V VFOA', 'w MD0C;',
                *(f'w EX{item}{value};' for item, value in settings.items()),
                f'L RFPOWER {cfg.rig.power_w / 100:g}', 'q']
    try:
        result = subprocess.run(list(selection.RIGCTL), input='\n'.join(commands) + '\n',
                                capture_output=True, text=True, timeout=25)
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError('The radio is not responding. Check its USB control connection and set CAT rate to 38400 baud. Nothing was keyed.') from exc
    if result.returncode or re.search(r'RPRT -\d+|error =', result.stdout + result.stderr):
        raise ValueError('The radio did not accept its data settings. Check its USB control connection and 38400-baud CAT setting; nothing was keyed.')
    reading = rig_settings(selection=selection)
    mismatches = [selection.MENU_LABELS.get(item, item) for item, value in settings.items()
                  if (reading.get('menu', {}).get(item) or '').lstrip('0') != value.lstrip('0')]
    if any(reading.get('menu', {}).get(item) is None for item in settings):
        raise ValueError('The radio did not report its data settings. Check the USB control connection; nothing was keyed.')
    if mismatches:
        raise ValueError('Radio settings were not verified: ' + ', '.join(mismatches) + '. Nothing was keyed.')
    mode = reading.get('mode') or ''
    if not mode.startswith(('PKTUSB', 'DATA-U')):
        raise ValueError('DATA USB mode was not verified; nothing was keyed.')
    try:
        power = float(reading.get('levels', {}).get('RFPOWER', '')) * 100
    except (ValueError, TypeError):
        raise ValueError('The radio did not report its RF power; nothing was keyed.') from None
    if abs(power - cfg.rig.power_w) > 1:
        raise ValueError(f'Radio power reads {power:g} W instead of {cfg.rig.power_w:g} W; nothing was keyed.')
    print('Data audio and RF power verified.', flush=True)


def check_rig(plan):
    """Read what the rig is on. Nothing here writes to it."""
    selection = selection_for_plan(plan)
    dial = cat('f', selection=selection)
    if dial is None or not dial.isdigit():
        raise RuntimeError('rigctl read no frequency back. The CAT link is down or held; '
                           f'check `{shlex.join([*selection.RIGCTL, "f"])}` before spending the slot.')
    off_by = abs(int(dial) - plan['dial_hz'])
    if off_by > DIAL_TOLERANCE_HZ and not plan['adjustments']['retune']:
        print(f'  the rig reads {dial} and this arm is {plan["dial_hz"]} ({off_by} Hz off), '
              f'and --no-retune leaves it there. Set it with '
              f'`{shlex.join([*selection.RIGCTL, "F", str(plan["dial_hz"])])}` if that is not what you meant.',
              flush=True)
    return dict(dial_hz=int(dial), dial_off_by_hz=off_by,
                rfpower=cat('l', 'RFPOWER', selection=selection),
                codec_gain=read_codec_gain(selection=selection),
                **read_environment(selection=selection))


# A RADIO THAT STOPPED ANSWERING CAT, AND THE ONE WAY BACK FROM IT.
#
# THE DETECTOR IS THE LAUNCHER'S OWN AND THERE IS NO SECOND ONE. kestrel_connect
# reads the dial before it keys anything, and prints this line when the daemon is
# up and the radio behind it answers nothing. Nothing here polls the rig looking
# for the fault: a CAT-polling watchdog was considered and ruled out.
#
# WHAT BRINGS IT BACK IS MAINS. The ATU's RFI is what took the interface down and
# no CAT-side reset reaches it; the plug cycles the rig and the tuner keeps its
# stored matches across the cut. See power.py for why this is a standing
# condition of the station and not something an operator can tune around.
CAT_SILENT = re.compile(r'^RIG NOT ANSWERING: .*$', re.M)

CAT_RETURN_S = 60.0     # a ceiling with slack; CAT has answered inside 15
CAT_POLL_S = 2.0


def cat_fault(directory):
    """The launcher's own words for a radio that stopped answering CAT, or None."""
    log = Path(directory or '.') / 'arm.log'
    if not log.is_file():
        return None
    found = CAT_SILENT.search(log.read_text(errors='replace'))
    return found.group(0).strip() if found else None


def cat_returns(ceiling_s=CAT_RETURN_S):
    """Poll a real dial query until the radio answers one, and say how long that took.

    Not a fixed sleep. The front panel takes 5-10 s to come back and CAT has
    answered inside 15, but a sleep either spends the worst case on every recovery
    or fires early and reads a booting radio as a dead one -- which is exactly the
    fault it is recovering from. What the elapsed figure is for is the trend: a
    rig taking longer to come back every time is a fact nothing else here notices.
    """
    started = time.monotonic()
    while True:
        dial = cat('f')
        spent = time.monotonic() - started
        if dial and dial.isdigit():
            return int(dial), spent
        if spent >= ceiling_s:
            return None, spent
        time.sleep(CAT_POLL_S)


def recover_cat(say=print):
    """Cycle the rig's mains and wait for a real dial query to answer.

    WHAT MAKES THIS SAFE IS WHERE IT RUNS. The arm this recovers died before it
    keyed anything -- the launcher refused it over the dial read -- so the finals
    are cold when the relay opens. The meter's own refusal in `power.cycle` is a
    second line and not the first: it witnesses a carrier, and a carrier is not
    the same fact as a key.

    Nothing here keys anything and nothing is written to the radio.
    """
    measured = []
    for line in power.cycle():      # said as each step measures, not collected and dumped
        measured.append(line)
        say(line)
    dial, spent = cat_returns()
    return dict(power=measured, dial_hz=dial, cat_back_s=round(spent, 1),
                recovered=dial is not None)


@contextmanager
def winlink_password(named=None):
    """Put a credential where the launcher will find it, for one arm, and no longer.

    `onair.sh` turns `--mail-password-file` into WINLINK_PASSWORD before any verb
    reads it, so the environment is the one route that is always open and the one
    that keeps the secret out of argv and out of every transcript that quotes the
    line. A password the operator named on the command line is already on the
    line; everything else arrives here.

    An unattended run with no credential anywhere is refused before the rig is
    opened, rather than at the mail stage with the slot half spent.
    """
    resolved = None
    if named:
        if not Path(named).expanduser().read_text().strip():
            raise ValueError('Winlink password file is empty; nothing was opened.')
        yield
        return
    if os.environ.get('WINLINK_PASSWORD'):
        yield
        return
    path = os.environ.get('WINLINK_PASSWORD_FILE')
    if path:
        held = Path(path).expanduser()
        resolved = held.read_text().strip()
        if not resolved:
            raise ValueError('Winlink password file is empty; nothing was opened.')
    if resolved is None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                'No Winlink credential: pass --mail-password-file PATH, or export '
                'WINLINK_PASSWORD. Nothing was opened. A run with no terminal cannot '
                'be asked for one.')
        resolved = getpass.getpass('Winlink password (hidden, this arm only): ').strip()
        if not resolved:
            raise ValueError('No Winlink password entered; nothing was opened.')
    os.environ['WINLINK_PASSWORD'] = resolved
    del resolved
    try:
        yield
    finally:
        os.environ.pop('WINLINK_PASSWORD', None)


def start_tap(plan, directory, *, writer=None, say=None, spawned=None, stop=None):
    """The rig's own monitor and the receive audio, on a converter the arm never mutes.

    `stream.wav` is the data jack and it is muted while we key; the tap is the only
    recording that holds both sides of a cycle, and the only one that can answer
    "the peer was audible but we did not read it" after the fact. It records on
    every arm because the question is never known in advance.
    """
    say = say or (lambda line: print(line, flush=True))
    selection = selection_for_plan(plan)
    tap_device = os.environ.get('HFMODEM_TAP_DEVICE')
    if not tap_device and not (plan['adjustments'].get('prepare_radio') or os.environ.get('HFMODEM_OPERATOR_INSTALLED')):
        tap_device = TAP_DEVICE
    if selection.RIG_NAME != 'ft891' or not tap_device:
        return None
    log = directory / 'tap.log'
    ownership = writer.reserve('tap.log') if writer else None
    extra = (['--capture-id', plan['capture_id'], '--capture-root', str(directory),
              '--capture-role', 'tap'] if writer else [])
    process = subprocess.Popen(
        [str(VENV), str(TXWITNESS), 'record', str(plan['adjustments']['witness_seconds']),
         str(directory / 'tap.wav'), '--device', tap_device, *extra],
        stdout=ownership.open() if ownership else log.open('w'), stderr=subprocess.STDOUT)
    if spawned:
        spawned(process)
    pause = stop.wait if stop else time.sleep
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if 'recording' in log.read_text(errors='replace'):
            say(f'  tap recording, pid {process.pid} -> {directory / "tap.wav"}')
            return process
        if process.poll() is not None:
            say(f'  NO TAP: {log} says why. The arm goes on.')
            return None
        if pause(0.5):
            return process
    return process


def place_kiwi(plan, directory, *, writer=None, say=None, spawned=None, stop=None):
    """A witness that is not recording before key-down is a null [witness-must-be-placed-before-the-arm].

    `spawned` is handed the process the moment it exists, and a set `stop` ends the
    wait for readiness at once, for a caller that may be torn down mid-hunt; both
    recorders take them."""
    say = say or (lambda line: print(line, flush=True))
    from .operator import witness_inventory
    inventory = witness_inventory()
    if not inventory.is_file():
        say(f'  NO WITNESS: configure witness_candidates ({inventory}).')
        return None
    log = directory / 'witness.log'
    ownership = writer.reserve('witness.log') if writer else None
    extra = (['--capture-id', plan['capture_id'], '--capture-root', str(directory),
              '--capture-role', 'console-witness'] if writer else [])
    process = subprocess.Popen(
        [str(VENV), str(KIWI), '--dial', f'{plan["dial_hz"] / 1000:.3f}',
         '--seconds', str(plan['adjustments']['witness_seconds']),
         '--out', str(directory / 'witness.wav'),
         '--candidates', str(inventory),
         *(['--peer', plan['call']] if plan['call'] else []),
         *(['--prefer', plan['adjustments']['kiwi']] if plan['adjustments']['kiwi'] else []),
         *extra],
        stdout=ownership.open() if ownership else log.open('w'), stderr=subprocess.STDOUT)
    if spawned:
        spawned(process)
    pause = stop.wait if stop else time.sleep
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if 'WITNESS PLACED' in log.read_text(errors='replace'):
            say(f'  witness placed, pid {process.pid} -> {directory / "witness.wav"}')
            return process
        if process.poll() is not None:
            say(f'  NO WITNESS: {log} says why. The arm goes on.')
            return None
        if pause(1.0):
            return process
    say(f'  witness still hunting a receiver after 120s; going anyway ({log}).')
    return process
