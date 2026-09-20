# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The acquisition funnel a receive session reports, and what it must not be read as.

ardopcf keeps this bookkeeping per session and prints it at teardown —
`ARQ.c:2612 LogStats()`: `LeaderDetects`, `LeaderSyncs`, `FrameSyncs`,
`Good Frame Type Decodes`, `Failed Frame Type Decodes`. We kept none of it, and
that absence is why the operator had to notice by ear that ARDOP arrivals go
unanswered. They do, and that is the protocol: every path from leader detected to
no header returns silently to `SearchingForLeader` (`SoundInput.c:937-1046`) and
`ARQ.c:1306` says it outright — no reply if no correct decode. So the answer to
the observation is a readout, not a NAK.

**No row here is "a frame arrived we could not read", and the readout may not
imply it.** A failed frame-type decode is one step of the acquisition walk over a
channel that mostly holds noise: `test_an_empty_channel_still_fills_the_funnel`
pins that on audio with no frame in it at all. The funnel is worth having for its
shape across a session and across bands, and the corpus test below is what makes
a number in it reproducible.

It reaches three surfaces, and the one that matters mid-slot is the monitor's
close-out summary: a host STATUS line needs a client attached to read it and a
session log is read afterwards, while `LiveMonitor.close` prints to the terminal
the operator is watching.
"""

from __future__ import annotations

import logging
import wave

import numpy as np
import pytest

from hfmodem.besra import monitor
from hfmodem.besra import radio as R
from hfmodem.besra.arq.modem import BesraModem
from hfmodem.besra.phy import modulator as M
from hfmodem.besra.sim.echo import _EchoObserver
from hfmodem.besra.phy.demodulator import SAMPLE_RATE, Demodulator
from hfmodem.core.resample import to_card
from hfmodem.tests import evidence
from .test_demod_robustness import _tune   # single-sideband tuning offset, as a rig has

_BLOCK = int(0.1 * R.FS_RADIO)          # the sound card's own block

#: 44 s of 7102 kHz recorded 2026-07-24 06:54:57Z, the corpus file besra's rolling
#: window was designed against (see `RollingDecoder`). A real ARDOP channel: a
#: run of ConReqs and a great deal of nothing.
_CORPUS = evidence.CORPUS / "7102k_065457.wav"

#: LeaderDetects, Good/Failed Frame Type Decodes, Frames reported over that file,
#: replayed through the live path 2026-08-27. Only the middle two moved when
#: `detect._CFO_WINDOW` stopped reading past the leader: thirteen header scans that
#: used to name a type on a shifted grid now fail to, and the five frames the file
#: gives up are the same five.
_MEASURED = (479, 46, 6263, 5)


class _Sink:
    """Stands in for the modem: takes decoded frames and host STATUS lines."""

    def __init__(self):
        self.audio_out = None
        self.frames: list = []
        self.status_lines: list[str] = []

    def receive_frames(self, frames):
        self.frames.extend(frames)

    def expected_session(self):
        return None

    def rx_epoch(self):
        return 0

    def status(self, text: str) -> None:
        self.status_lines.append(text)


def _at_card_rate(au: np.ndarray) -> np.ndarray:
    return to_card(au, SAMPLE_RATE).astype(np.float32)


def _replay(card: np.ndarray, rx: R.RollingDecoder) -> None:
    """Feed a 48 kHz capture through the live receive path, block by block."""
    for i in range(0, card.size, _BLOCK):
        rx.push(card[i:i + _BLOCK])
        rx.pump()
    rx.flush()


def _rolling(card: np.ndarray) -> R.RollingDecoder:
    rx = R.RollingDecoder(Demodulator(), lambda pos, f: None)
    _replay(card, rx)
    return rx


def test_one_real_frame_walks_the_whole_funnel():
    """A leader holds, a header decodes, one frame is reported — every row moves."""
    au = M.render_frame(0x4A, payload=b"funnel", session_id=0x51)
    card = _at_card_rate(np.concatenate([np.zeros(2400, dtype="<i2"), au]))
    funnel = _rolling(card).funnel

    assert funnel.frames == 1
    assert funnel.good_frame_types >= 1
    assert funnel.leader_detects >= 1
    assert funnel.capture_s == pytest.approx(card.size / R.FS_RADIO, abs=0.01)


def test_an_empty_channel_still_fills_the_funnel():
    """The reason no row may be read as frames we missed: 8 s of noise fails 629
    header scans — 4700 a minute — with nothing on the channel to miss."""
    rng = np.random.default_rng(2026)
    card = rng.normal(0.0, 0.01, int(8.0 * R.FS_RADIO)).astype(np.float32)
    funnel = _rolling(card).funnel

    assert funnel.frames == 0
    assert funnel.failed_frame_types > 400


def test_the_readout_speaks_the_reference_vocabulary():
    """`LogStats`' own row labels, so a besra funnel reads against an ardopcf one —
    and the caveat travels with them, because both surfaces show this one string."""
    line = str(_rolling(np.zeros(int(0.5 * R.FS_RADIO), dtype=np.float32)).funnel)

    for label in ("LeaderDetects", "Good Frame Type Decodes",
                  "Failed Frame Type Decodes", "Frames reported"):
        assert label in line, line
    assert "search bookkeeping" in line
    assert "\n" not in line               # one line: it goes out as a host STATUS


def test_the_session_reports_its_funnel_to_the_host_and_the_log(caplog):
    """Both surfaces, at the end of the receive session — `RadioLink.close`, which
    is where a besra run stops listening."""
    au = M.render_frame(0x4A, payload=b"both surfaces", session_id=0x51)
    link = R.RadioLink(_Sink(), rig=None)
    _replay(_at_card_rate(np.concatenate([np.zeros(2400, dtype="<i2"), au])), link._rx)

    with caplog.at_level(logging.INFO, logger="hfmodem.besra.radio"):
        link.close()

    assert link.modem.status_lines == [str(link._rx.funnel)]
    assert str(link._rx.funnel) in caplog.text
    assert "Frames reported=1" in caplog.text


def test_the_modem_carries_a_status_line_the_session_never_saw():
    """The seam the funnel crosses to reach the host: the receive path is the
    radio backend's, not the ARQ session's, so the modem takes a line from outside
    it — and says nothing at all before a host is attached."""
    said: list[str] = []

    class _Host(_EchoObserver):
        def modem_status(self, text):
            said.append(text)

    modem = BesraModem()
    modem.status("before the host")
    modem.start(_Host(), threaded=False)
    modem.status("RX funnel over 0.0 s")

    assert said == ["RX funnel over 0.0 s"]




def test_the_leader_row_hears_a_leader_off_the_nominal_bins():
    """The row is a leader detector across the ±200 Hz a receiver must tolerate
    (spec §4.1/§7), not a detector of leaders that happen to be on frequency.

    Read on the nominal 1475/1525 Hz bins alone it is blind exactly where the
    operator most needs it: a mistuned channel decodes frames — the acquisition
    walk estimates the offset per candidate — while the leader row reads 0, and
    nothing in the readout says why. Shifted 195 Hz low, this frame's leader
    holds nowhere near 1475/1525 and the row must still see it."""
    au = np.concatenate([np.zeros(2400, dtype="<i2"),
                         M.render_frame(0x4A, payload=b"195 Hz low", session_id=0x51)])
    # Half scale first: the analytic shift overshoots and would clip through int16.
    funnel = _rolling(_at_card_rate(_tune(au.astype(float) * 0.5, -195.0))).funnel

    assert funnel.frames == 1
    assert funnel.leader_detects >= 1


def test_a_mistuned_channel_names_its_offset():
    """What `LeaderDetects=0` under a non-zero frame row used to mean, said out
    loud. Retuning is the one thing the operator can do and the modem cannot, so
    the readout names the offset the leaders held at — to the 25 Hz the detection
    bin resolves — rather than leaving it to be inferred from a zero."""
    au = np.concatenate([np.zeros(2400, dtype="<i2"),
                         M.render_frame(0x4A, payload=b"195 Hz low", session_id=0x51)])
    funnel = _rolling(_at_card_rate(_tune(au.astype(float) * 0.5, -195.0))).funnel

    assert funnel.leader_offset == -200
    assert "-200 Hz" in str(funnel)


def test_an_on_tune_channel_names_no_offset():
    """The other half of that: a leader on the nominal bins is not an offset to
    correct, and a readout that names one on every session names nothing."""
    au = np.concatenate([np.zeros(2400, dtype="<i2"),
                         M.render_frame(0x4A, payload=b"on tune", session_id=0x51)])
    funnel = _rolling(_at_card_rate(au)).funnel

    assert funnel.leader_detects >= 1
    assert funnel.leader_offset is None
    assert "Hz" not in str(funnel)


def test_the_monitor_summary_carries_the_funnel(tmp_path, capsys):
    """The third surface, and the one an operator actually watches during a slot:
    `LiveMonitor.close` prints the per-session summary. The host STATUS line goes
    to a client that may not be attached and the session log is read afterwards.

    And the frame count in it is the decoder's own. The monitor kept a second
    counter of the same quantity, which is how two counters for one number begin
    to disagree."""
    au = np.concatenate([np.zeros(2400, dtype="<i2"),
                         M.render_frame(0x4A, payload=b"watched live", session_id=0x51)])
    card = _at_card_rate(au)
    mon = monitor.LiveMonitor(Demodulator(), tmp_path / "cap.wav", source="fake")
    for i in range(0, card.size, _BLOCK):
        mon.push(card[i:i + _BLOCK])
        mon.pump()
    mon.close()

    out = capsys.readouterr().out
    assert "frames 1" in out
    assert str(mon._rolling.funnel) in out
    assert "Frames reported=1" in out and "search bookkeeping" in out



@pytest.mark.skipif(not _CORPUS.exists(), reason=f"{_CORPUS} absent")
def test_the_funnel_over_a_named_recording():
    """The corpus as the oracle: this file, through the live path, every time.

    Five `ConReq2000M`, KC3OWM to K4PAR-2 on the 3.59 s repeat grid, and two
    things about the shape worth reading off them. The decode rows run to thousands
    where the frame row is 5 — 8500 failed header scans a minute — which is what
    "not frames missed" means in practice.

    And the capture is mistuned: its three measured leaders read −195, −159 and
    −198 Hz (`detect.estimate_leader_cfo`), which is why the row read 0 while the
    file gave up five clean frames — the acquisition walk estimates the offset per
    candidate and decodes them, and a leader row on the nominal bins alone reads
    0.131/0.122/0.232 there against a 0.25 floor where the same leaders read
    0.927/0.450/0.958 at the estimated shift. The row searches the offsets now, so
    the mistuning is a number the operator can act on instead of a zero they have
    to interpret, and the acquisition rows are untouched by the search: the walk
    still triggers on the nominal bins.
    """
    with wave.open(str(_CORPUS)) as w:
        assert w.getframerate() == R.FS_RADIO
        card = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0

    funnel = _rolling(card).funnel

    assert funnel.capture_s == pytest.approx(44.0, abs=0.05)
    assert (funnel.leader_detects, funnel.good_frame_types,
            funnel.failed_frame_types, funnel.frames) == _MEASURED
    assert funnel.leader_offset == -200          # measured -195, -159, -198 Hz
