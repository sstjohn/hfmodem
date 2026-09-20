# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One station, one drive, and the knob that actually sets it.

Four modems shared one radio and one interface and disagreed about level by 18 dB:
shrike normalised to 0.6, kestrel to 0.8, sabir to 0.1, and besra to nothing at
all — whatever its modulator happened to sum to, resampled to 48 kHz with no
limit, which put 1.9% of a 4FSK.2000.600 frame past the rail. An operator sets the
rig's input gain once, against whichever modem is running, and is overdriving on
the next.

`[audio] tx_drive` was declared, documented and shipped in the example, and was
read by exactly one thing — the unified `TxArbiter`, which no on-air path goes
through. So the knob could be set, nothing would change, and drive would be ruled
out as the cause of a weak signal. That is what these tests are for: the number
reaches every program that can key this station, a run may still override it at
the rig, and nothing any of them composes leaves past full scale.

Off-air entirely. No serial port, no sound card, no rig: the play calls are
stubbed and the waveforms are the real renderers' own output.
"""
from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from hfhost.config import ConfigError

from hfmodem.core import config, levels

_REPO = Path(__file__).resolve().parents[5]
_TOOLS = _REPO / "tools"
sys.path.insert(0, str(_TOOLS))

#: `tools/` and `examples/` are the repo's, not the wheel's, so everything reached
#: through them is absent when the published suite runs inside a built
#: distribution. Named rather than skipped silently: a drive that stops reaching
#: kestrel is exactly what this file exists to catch.
_off_tree = pytest.mark.skipif(
    not (_TOOLS / "kestrel_connect.py").exists(),
    reason=f"{_TOOLS} is not in this tree -- kestrel_connect ships from the "
           "repo rather than the package, so its drive cannot be read here")

#: Far enough from every value in the tree that a test passing on it cannot be
#: passing on a leftover default.
_SET = 0.42
_OVERRIDE = 0.25


@pytest.fixture
def station(tmp_path, monkeypatch):
    """The example station file, at a drive nothing else in the tree uses, named
    to the on-air tools the way an operator's profile names it."""
    def _write(drive: float = _SET) -> Path:
        raw = (_REPO / "examples" / "station.toml").read_text(encoding="utf-8")
        out = tmp_path / "station.toml"
        out.write_text(re.sub(r"(?m)^tx_drive.*$", f"tx_drive = {drive}", raw),
                       encoding="utf-8")
        monkeypatch.setenv(config.STATION_ENV, str(out))
        return out
    return _write


# -- the resolver ---------------------------------------------------------

def test_without_a_station_file_the_drive_is_the_shipped_one(monkeypatch):
    monkeypatch.delenv(config.STATION_ENV, raising=False)
    assert config.tx_drive() == levels.TX_DRIVE


def test_the_station_file_sets_the_drive(station):
    station()
    assert config.tx_drive() == _SET


def test_a_run_may_still_overrule_the_station(station):
    """The overrides are how the on-air tools are actually driven — a level
    raised at the rig against a meter, for that run only."""
    station()
    assert config.tx_drive(_OVERRIDE) == _OVERRIDE


def test_a_station_file_that_will_not_load_names_itself(tmp_path, monkeypatch):
    """Four transmit tools read this on the way to the key. A bare traceback out
    of one of them reads as a broken tool rather than as a station file."""
    monkeypatch.setenv(config.STATION_ENV, str(tmp_path / "nothing.toml"))
    with pytest.raises(ConfigError, match=config.STATION_ENV):
        config.tx_drive()


# -- every program that can key -------------------------------------------

def _watch(monkeypatch) -> dict:
    """What the resolver answered on this run. One module object behind all four
    imports, so one patch sees whichever of them is being driven."""
    seen: dict = {}
    real = config.tx_drive
    monkeypatch.setattr(config, "tx_drive",
                        lambda o=None: seen.setdefault("d", real(o)))
    return seen


def _shrike(argv, monkeypatch):
    from hfmodem.shrike import onair
    seen = _watch(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["onair", *argv])
    with pytest.raises(SystemExit):          # no --dxcall; after the resolve
        onair.main()
    return seen["d"]


def _kestrel(argv, monkeypatch):
    import kestrel_connect
    seen = _watch(monkeypatch)
    monkeypatch.setattr(kestrel_connect, "connect", lambda *a, **k: False)
    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", "W1AW",
                                      "--mycall", "W9SSJ", "--dry-run", *argv])
    kestrel_connect.main()
    return seen["d"]


def _besra(argv, monkeypatch):
    """Straight off the built link, which is the strongest form of this: the
    number that reached the transmitter, not the number the resolver returned."""
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server
    from hfmodem.tests.besra.test_cli import _FakeRig

    monkeypatch.setattr(radio, "Rig", _FakeRig)
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    monkeypatch.setattr("signal.signal", lambda s, h: None)
    args = run_server.parser().parse_args(
        ["--radio", "ft891", "--rigctld", "localhost:4532", "--dial", "7099000",
         "--quiet", "--no-record", *argv])
    modem, link = run_server._radio_modem(args)
    try:
        return link.drive
    finally:
        modem.stop()


def _sabir(argv, monkeypatch):
    from hfmodem.sabir import onair
    seen = _watch(monkeypatch)
    # --transmit with no keying line: refused by `_refusals`, after the resolve
    assert onair.main(["--mycall", "W9SSJ", "--transmit", *argv]) == 2
    return seen["d"]


#: Every program in this repository that can key the transmitter, and the flag
#: each spells its drive with. A default left on the flag shadows the station
#: file silently, which is how three of these came to carry a private level.
KEYERS = [
    pytest.param(_shrike, "--tx-drive", id="shrike.onair"),
    pytest.param(_kestrel, "--amplitude", id="kestrel_connect", marks=_off_tree),
    pytest.param(_besra, "--tx-drive", id="besra.run_server"),
    pytest.param(_sabir, "--gain", id="sabir.onair"),
]


@pytest.mark.parametrize("entry,flag", KEYERS)
def test_the_configured_drive_reaches_every_on_air_path(entry, flag, station,
                                                        monkeypatch):
    station()
    assert entry([], monkeypatch) == _SET, (
        f"{flag} was left off, so this program transmits at a level of its own "
        f"and [audio] tx_drive reaches nothing")


@pytest.mark.parametrize("entry,flag", KEYERS)
def test_a_flag_beats_the_station_file(entry, flag, station, monkeypatch):
    station()
    assert entry([flag, str(_OVERRIDE)], monkeypatch) == _OVERRIDE


# -- what leaves the card -------------------------------------------------

def _pactor1():
    from hfmodem.shrike import pactor1
    return pactor1.connect_signal("W1AW")


def _vara_cr():
    from hfmodem.kestrel.vara import vara_frames as VF
    from hfmodem.kestrel.vara import vara_mfsk as MK
    return MK.synth_burst("W1AW", VF.CR)


def _sabir_beacon():
    from hfmodem.sabir import onair
    return onair.presence_burst("W9SSJ").audio


def _shrike_played(audio, drive, monkeypatch, tmp_path):
    from hfmodem.shrike import onair

    class _Rig:
        def key_failure(self): return None

    played = []
    monkeypatch.setattr(onair.ota, "_play",
                        lambda a, dev, mk, rig, settle=0.1: (played.append(a), (0.0, 0))[1])
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=None, outdir=tmp_path,
                       drive=drive)
    tx._tx(audio, "level check")
    return played[0]


def _kestrel_played(audio, drive, monkeypatch):
    import kestrel_connect

    fake = types.ModuleType("sounddevice")
    fake.query_devices = lambda dev: {"max_output_channels": 2}
    fake.InputStream = lambda **kw: types.SimpleNamespace(
        start=lambda: None, stop=lambda: None, close=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    played = []
    monkeypatch.setattr(kestrel_connect, "play_drained",
                        lambda block, fs, dev: played.append(np.asarray(block)))
    io = kestrel_connect.AudioVaraIO("stub", "stub", drive)
    io.tx(audio)
    return played[0]


def _besra_played(frame_type, drive, monkeypatch):
    from hfmodem.tests.besra.test_radio_audio_path import _Sink, _stub_sounddevice
    import hfmodem.besra.radio as R
    from hfmodem.besra.phy import modulator as M

    played = _stub_sounddevice(monkeypatch)
    link = R.RadioLink(_Sink(), rig=None, out_device="stub", drive=drive)
    link._transmit(M.render_frame(frame_type, payload=bytes(range(24)),
                                  session_id=0x5A))
    return played[-1][1]                        # the burst; `warm_output` wrote first


# One burst per modem per envelope family, plus the frames that were over the rail
# before anything normalised them: every 1000 and 2000 Hz besra mode reached 1.07
# to 1.12 at 48 kHz, clipping between 0.1% and 1.9% of its samples depending on
# payload. ardopcf's own soft clip leaves the 12 kHz array at 0.9979 and the
# interpolation to 48 kHz overshoots it, so composed scale was never the
# protection. 4FSK.500.100S is here as the frame this station's ARQ actually
# sends, which cleared the rail on its own and must still leave at the drive.
def _bursts(monkeypatch, tmp_path, drive):
    yield "shrike PACTOR-1 connect", _shrike_played(_pactor1(), drive,
                                                    monkeypatch, tmp_path)
    if (_TOOLS / "kestrel_connect.py").exists():
        yield "kestrel MFSK connect-request", _kestrel_played(_vara_cr(), drive,
                                                              monkeypatch)
    for name, ft in (("4FSK.500.100S", 0x4C), ("4FSK.2000.600", 0x7A),
                     ("4FSK.2000.600S", 0x7C), ("16QAM.2000.100", 0x74)):
        yield f"besra {name}", _besra_played(ft, drive, monkeypatch)


def test_no_burst_leaves_this_station_past_full_scale(monkeypatch, tmp_path):
    from hfmodem.sabir import radio as sabir_radio
    for what, audio in _bursts(monkeypatch, tmp_path, levels.TX_DRIVE):
        peak = float(np.abs(audio).max())
        assert peak == pytest.approx(levels.TX_DRIVE, abs=1e-4), (
            f"{what} left at peak {peak:.4f}, not {levels.TX_DRIVE}")
    # sabir plays what `levelled` returns and nothing else touches it.
    peak = float(np.abs(sabir_radio.levelled(_sabir_beacon())).max())
    assert peak == pytest.approx(levels.TX_DRIVE, abs=1e-4)


def test_a_configured_drive_is_the_peak_that_goes_out(monkeypatch, tmp_path):
    """Not merely bounded by it — the renderers compose 22 dB apart, so a limiter
    would leave the quiet ones quiet and only the loud ones at the drive."""
    for what, audio in _bursts(monkeypatch, tmp_path, _OVERRIDE):
        peak = float(np.abs(audio).max())
        assert peak == pytest.approx(_OVERRIDE, abs=1e-4), (
            f"{what} left at peak {peak:.4f}, not {_OVERRIDE}")
