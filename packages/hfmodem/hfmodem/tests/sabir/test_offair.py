# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Off-air beacon tool: TX render -> a simulated WebSDR/KiwiSDR capture
(CFO + AWGN + Kiwi-rate resample) -> decode. Validates the one-way chain
offline so a real on-air run only adds a real channel, not new code paths.

The simulated capture is flat -- constant level, stationary noise -- and a real
one is not, so the recordings of the 2026-08-29 transmission are read here too;
`clips.py` says what they are and what they caught that this could not."""

import argparse

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import resample, resample_poly

from hfmodem.sabir import offair
from hfmodem.sabir.arq import wire
from hfmodem.sabir.floor import beacon as wspr
from hfmodem.sabir.floor.mfsk import FloorModem
from hfmodem.tests.sabir import clips


def _presence_audio(call="W1AW"):
    bc = wire.Beacon.build(wire.capabilities(range(3, 7 + 1), wire.FASTCTL), call, profile=1)
    return offair._lead_silence(offair.to_real(FloorModem().transmit(bc.pack())))


def _add_noise(x, snr_db, rng):
    active = x[np.abs(x) > 1e-6]
    sig = float(np.mean(active ** 2))
    n = rng.standard_normal(x.size) * np.sqrt(sig / 10 ** (snr_db / 10))
    return x + n


def _cfo(x, hz):
    z = offair.to_analytic(x)
    t = np.arange(z.size) / offair.FS
    return offair.to_real(z * np.exp(2j * np.pi * hz * t))


def test_wav_roundtrip_48k(tmp_path):
    x = _presence_audio()
    p = str(tmp_path / "b.wav")
    offair.wav_write(p, x)
    y = offair.wav_read(p)
    assert abs(y.size - x.size) <= 1                   # same length at 48 kHz
    # and the shape is preserved up to the write's peak-normalisation (not just
    # the length -- a x1000 or wrong-dtype scaling would pass a length check)
    xs, ys = x / np.abs(x).max(), y / np.abs(y).max()
    assert np.corrcoef(xs, ys)[0, 1] > 0.999
    assert 0.5 < np.abs(y).max() < 1.0                 # normalised into range


def test_cmd_decode_reports_the_beacon(tmp_path, capsys):
    """The CLI decode path (untested before) names a rendered presence beacon."""
    x = _presence_audio("K6XYZ")
    p = str(tmp_path / "c.wav")
    offair.wav_write(p, x)
    offair.cmd_decode(argparse.Namespace(wav=p))
    assert "K6XYZ" in capsys.readouterr().out


def test_wav_read_resamples_kiwi_rate(tmp_path):
    x = _presence_audio()
    kiwi = resample_poly(x, 12000, offair.FS)           # a 12 kHz Kiwi capture
    p = str(tmp_path / "k.wav")
    wavfile.write(p, 12000,
                  (kiwi / np.abs(kiwi).max() * 0.9 * 32767).astype(np.int16))
    y = offair.wav_read(p)                              # -> 48 kHz float
    assert abs(y.size - x.size) <= 2 and y.dtype == np.float64


def test_presence_beacon_through_simulated_websdr(tmp_path):
    """+30 Hz dial offset, 8 dB peak SNR, round-tripped through a 12 kHz Kiwi
    int16 WAV -- the realistic off-air path -- still decodes byte-exact."""
    rng = np.random.default_rng(0)
    x = _cfo(_presence_audio("K6XYZ"), 30.0)
    x = _add_noise(x, 8.0, rng)
    kiwi = resample_poly(x, 12000, offair.FS)
    p = str(tmp_path / "cap.wav")
    wavfile.write(p, 12000,
                  (kiwi / np.abs(kiwi).max() * 0.9 * 32767).astype(np.int16))
    z = offair.to_analytic(offair.wav_read(p))
    raw, _ = FloorModem().receive(z, wire.CONNECTIONLESS_BYTES)
    blk = wire.Control.unpack(raw) if raw is not None else None
    assert isinstance(blk, wire.Beacon) and blk.call == "K6XYZ"


def test_wspr_beacon_pipeline_under_noise(tmp_path):
    """The narrow-tone beacon renders, survives noise, and decodes -- the tool
    path; test_m6b owns the deep-sensitivity numbers."""
    rng = np.random.default_rng(1)
    pl = wspr.BeaconPayload(callsign="W1AW", grid="FN31", status=0)
    x = offair._lead_silence(
        offair.to_real(wspr.send_beacon(pl, wspr.BEACON_GEARS["beacon_deep"])))
    x = _add_noise(x, -5.0, rng)
    z = offair.to_analytic(x)
    got, _ = wspr.recv_beacon(z, wspr.BEACON_GEARS["beacon_deep"])
    assert got is not None and got.callsign == "W1AW" and got.grid == "FN31"


# -- the real channel, which none of the above has ----------------------------

@pytest.mark.skipif(not clips.SLOT.is_dir(),
                    reason=f"{clips.SLOT} is not on this machine")
@pytest.mark.parametrize("clip,rung", sorted(clips.HEARD.items()))
def test_the_first_transmission_decodes_off_both_receivers(clip, rung, capsys):
    """Eight recordings, two receivers, four waveforms, one blind decode each.

    `beacon_short` decoded in simulation and missed on both KiwiSDRs, because
    the coarse sync accumulated received energy: a selective fade swings the
    level 10-15 dB inside a burst, so a lag whose Costas windows landed on the
    fade peaks outbid the true lag by 1.9x while holding a third of its tone
    selectivity. The rung the decode names is asserted, not just that something
    decoded -- naming it is the whole reason a rung is announced.
    """
    offair.cmd_decode(argparse.Namespace(wav=str(clips.SLOT / f"{clip}.wav")))
    out = capsys.readouterr().out
    want = ("presence beacon: station 'W9SSJ'" if rung == "presence"
            else f"({rung}): W9SSJ EN63")
    assert want in out, out


@pytest.mark.parametrize("gear", ["beacon_short", "beacon_med", "beacon_deep"])
def test_a_recorder_whose_sample_clock_is_not_an_integer(gear, tmp_path):
    """Both KiwiSDRs ran ~92 ppm slow -- 11998.893 and 11998.871 Hz under a
    12000 Hz header, which is the rate `wav_read` resamples on. Every rung
    decodes through that and keeps decoding out to ~1%, so the sample clock had
    two orders of magnitude in hand and is not what `beacon_short` missed on.
    Pinned because the simulated capture above is an exact integer clock and a
    receiver never is."""
    x = offair.render_wspr("W9SSJ", "EN63", gear)
    kiwi = resample_poly(x, 12000, offair.FS)               # the recorder's rate
    slow = resample(kiwi, round(kiwi.size * 11998.893 / 12000))
    p = str(tmp_path / "kiwi.wav")
    wavfile.write(p, 12000,                                 # ... and its header
                  (slow / np.abs(slow).max() * 0.9 * 32767).astype(np.int16))
    got, _ = wspr.recv_beacon(offair.to_analytic(offair.wav_read(p)),
                              wspr.BEACON_GEARS[gear])
    assert got is not None and got.callsign == "W9SSJ" and got.grid == "EN63"


def test_deep2_opens_with_a_whole_deep_burst(tmp_path, capsys):
    """`beacon_deep2` tiles the coded stream twice, so its first 100 symbols are
    a bit-exact `beacon_deep` burst -- and the deep plan's four Costas blocks sit
    inside a deep2 burst's seven at four different lags, so a deep decode of a
    deep2 transmission succeeds whenever sync happens to pick the first. That is
    a correct decode of a burst that really is on the air, under the wrong rung's
    name, and which of the two the operator is told is a coin toss. The rungs are
    told apart on the whole burst and never on the prefix, so `cmd_decode` scans
    deepest first: the outermost rung that fits is the one transmitted."""
    pl = wspr.BeaconPayload("W9SSJ", "EN63", 0)
    deep = wspr.send_beacon(pl, wspr.BEACON_GEARS["beacon_deep"])
    deep2 = wspr.send_beacon(pl, wspr.BEACON_GEARS["beacon_deep2"])
    ramp = deep.size - wspr.GUARD_TAIL - wspr.EDGE       # deep's own closing ramp
    assert np.array_equal(deep[:ramp], deep2[:ramp])

    x = offair._lead_silence(offair.to_real(deep2), 0.25)   # a lag deep does take
    assert wspr.recv_beacon(offair.to_analytic(x),
                            wspr.BEACON_GEARS["beacon_deep"])[0] == pl
    p = str(tmp_path / "deep2.wav")
    offair.wav_write(p, x)
    offair.cmd_decode(argparse.Namespace(wav=p))
    assert "(beacon_deep2): W9SSJ EN63" in capsys.readouterr().out


def test_to_analytic_decodes_the_same_at_awkward_lengths():
    """`hilbert` never pads, so a capture whose length has a large prime factor
    costs ~5x. Padding to a fast length changes the boundary convention -- the
    analytic transform is non-local -- so the guarantee that matters is not
    numerical equality but that the burst still decodes byte-exact. Lengths are
    nudged off round numbers on purpose."""
    m = FloorModem()
    blk = wire.Control.ident(wire.ID, 0x81, "W9SSJ").pack()
    rng = np.random.default_rng(0)
    tx = offair.to_real(m.transmit(blk))
    for trim in (0, 1, 7, 13, 101):
        x = np.concatenate([np.zeros(4096), tx, np.zeros(4096)])
        if trim:
            x = x[:-trim]
        x = x + rng.standard_normal(x.size) * 0.05
        got, _ = m.receive(offair.to_analytic(x), len(blk))
        assert got == blk, f"failed at length {x.size}"


# -- and it keys nothing ------------------------------------------------------
#
# This module used to carry a `transmit` verb that opened a rig by serial port,
# keyed it over CAT and played a WAV, with no arm gate, no keying line, no dial
# readback and nothing listened to first — an improvised keying path beside a
# decoder whose whole job is to be run at a desk. It moved to `sabir.onair`,
# where `tests/sabir/test_onair.py` pins each of those four. What is left here
# renders and decodes, and the gate below is that it stays that way.

def test_nothing_here_can_key_a_transmitter():
    from pathlib import Path
    text = Path(offair.__file__).read_text()
    for reach in ('add_parser("transmit"', "ptt", "rigctl", "sounddevice",
                  "sabir import radio"):
        assert reach not in text, (
            f"offair reaches for {reach!r}; putting a signal on the air is "
            "sabir.onair's, and one keying path per modem is the rule")


def test_analytic_rejects_multichannel_audio():
    """A sound card hands you (frames, channels); `hilbert` transforms the last
    axis, so 2-D input silently returned a wrongly shaped array."""
    from hfmodem.sabir.phy.modem import analytic
    with pytest.raises(ValueError):
        analytic(np.zeros((479, 2)))
    analytic(np.zeros(479))


def test_wav_read_handles_the_formats_a_far_end_recorder_produces():
    """The far end of an on-air test records with whatever it has. All of these
    must read and decode, and 8-bit must not carry its unsigned DC pedestal --
    the floor decoder ignores DC but `_snr_estimate` squares it."""
    import subprocess
    from hfmodem.sabir.arq import wire
    from hfmodem.sabir.floor.mfsk import FloorModem
    m = FloorModem()
    blk = wire.Beacon.build(wire.capabilities(range(3, 7 + 1), wire.FASTCTL), "W9SSJ", profile=1).pack()
    src = "/tmp/_farend_ref.wav"
    offair.wav_write(src, offair._lead_silence(offair.to_real(m.transmit(blk))))
    cases = {
        "44.1k stereo s16": ["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"],
        "12k mono s16": ["-ar", "12000", "-ac", "1", "-c:a", "pcm_s16le"],
        "8k mono u8": ["-ar", "8000", "-ac", "1", "-c:a", "pcm_u8"],
    }
    for name, args in cases.items():
        out = "/tmp/_farend_case.wav"
        if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                           *args, out]).returncode != 0:
            pytest.skip("ffmpeg unavailable")
        x = offair.wav_read(out)
        assert abs(x.mean()) < 0.05, f"{name}: DC offset {x.mean():+.3f}"
        got, _ = m.receive(offair.to_analytic(x), wire.CONNECTIONLESS_BYTES)
        b = wire.Control.unpack(got) if got is not None else None
        assert isinstance(b, wire.Beacon) and b.call == "W9SSJ", name
