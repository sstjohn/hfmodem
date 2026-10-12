"""The rig as a monitor source: its receive audio, its filter, and its dial.

Receive-only. The dial and mode come from one read-only CAT read. The one write
anywhere here is `tune`, asked for with `--freq`, made once before listening,
read back and refused on a mismatch. It sets only frequency: campaign tuning
also sets receive controls, which a monitoring operator already chose.
Nothing here keys.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from hfhost.audio import WavReplaySource, Window

from . import station

DEFAULT_FILTER_HZ = (300, 2900)


@dataclass(frozen=True)
class Dial:
    hz: int | None
    mode: str | None
    source: str


def station_file(explicit=None):
    from hfmodem.core import config
    return Path(explicit or os.environ.get(config.STATION_ENV)
                or station.ROOT / 'station.toml').expanduser()


def station_config(path=None):
    from hfmodem.core import config
    found = station_file(path)
    return config.load(found) if found.is_file() else None


def read_dial(cat=None):
    """`rigctl f m` in one session, which answers '7100000 PKTUSB 3000'."""
    said = (cat or station.cat)('f', 'm', joined=True)
    words = (said or '').split()
    if not words or not words[0].isdigit():
        return Dial(None, None, 'cat unreadable')
    # hamlib's FTDX10 width table has no entry for the DATA-U width the rig
    # reports (SH0023), so `m` answers 'error = ...' while `f` still reads.
    mode = words[1] if len(words) > 1 and words[1] != 'error' else None
    return Dial(int(words[0]), mode, 'cat')


def input_device(cfg=None):
    """Use the selected chain; a losing station-file rig cannot supply audio."""
    if cfg and cfg.selected_rig == station.RIG_NAME:
        return cfg.audio.input
    return station.AUDIO_DEVICE


def tune(dial_hz, run=None):
    """Set only the frequency through the selected rig's CAT interface."""
    done = (run or subprocess.run)([*station.RIGCTL, 'F', str(int(dial_hz))],
                                   text=True, capture_output=True, timeout=4)
    if done.returncode != 0:
        said = (done.stderr or done.stdout).strip()
        raise RuntimeError(said or f'tune {dial_hz} failed ({done.returncode})')
    return done.stdout.strip()


def window(dial, cfg=None):
    lo, hi = cfg.rig.filter_hz if cfg else DEFAULT_FILTER_HZ
    mode = dial.mode or (cfg.rig.mode if cfg else 'USB')
    where = f'{dial.hz / 1e3:.3f} kHz' if dial.hz else 'dial unknown'
    return Window.sideband(float(lo), float(hi), f'rig {mode} {where}', dial.hz, mode)


def source(device=None, *, freq_hz=None, dial_hz=None, path=None,
           cat=None, run=None):
    """``(CommandSource, Dial)`` for the station's receive device.

    `freq_hz` tunes first; `dial_hz` labels a run whose CAT is down and is
    believed over the read.
    """
    from creance.monitor import device_source
    cfg = station_config(path)
    if freq_hz is not None:
        tune(freq_hz, run=run)
    dial = Dial(int(dial_hz), None, '--dial') if dial_hz is not None else read_dial(cat)
    if freq_hz is not None:
        from hfmodem.core.band import QSY_TOLERANCE_HZ
        if dial.hz is None or abs(dial.hz - freq_hz) > QSY_TOLERANCE_HZ:
            raise RuntimeError(f'DIAL MISMATCH: asked {freq_hz} Hz, read {dial.hz}; '
                               'the monitor has not opened audio.')
    selected_input = device or input_device(cfg)
    if not selected_input:
        raise ValueError(f"[audio.chains.{station.RIG_PROFILE.get('audio_chain', '')}] input is required for capture")
    audio = device_source(selected_input)
    audio.window = window(dial, cfg)
    return audio, dial


def wav_source(recording, *, dial_hz=None, span=None, realtime=False, path=None):
    """A recording, windowed to this station's filter unless `span` (lo, hi) says otherwise."""
    cfg = station_config(path)
    filter_hz = cfg.rig.filter_hz if cfg else DEFAULT_FILTER_HZ
    mode = cfg.rig.mode if cfg else 'USB'
    window = None if span is None else Window.sideband(
        float(span[0]), float(span[1]), f'wav {Path(recording).name}', dial_hz, mode)
    return WavReplaySource(recording, realtime=realtime, dial_hz=dial_hz, mode=mode,
                           window=window, filter_hz=filter_hz)
