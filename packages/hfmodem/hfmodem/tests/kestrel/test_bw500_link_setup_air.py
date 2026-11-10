# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW500 behind our link-setup, off the air: the NAK, the ack, and the first over.

Two evening calls on 40 m, 2026-09-11. K5DAT-13 answered our first link-setup
with the seed-289 position-1 NAK 0.32 s after it ended and was silent through the
two resends that followed 13 s apart. K5FIT keyed the connected-ack and then its
greeting. The whole-call replay now uses the surviving 0924 K5FIT tap.

Over every BW500 link-setup tail on file (K5DAT-13, K5FIT, KC9GHZ and AA5AF of
09-20) the called station's 289 stream scores 3.53 at position 1 behind
K5DAT-13's first link-setup, and no more than 1.30 at any position anywhere else
away from the connect offers.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.tests import evidence
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel import test_bw500_answer as answer
from hfmodem.tests.kestrel.test_link_setup_nak_stream import _awaiting, _resent

FIXTURES = Path(__file__).parent / "fixtures" / "linksetup-nak-500-0911"
evidence.require_input(FIXTURES / "manifest.json", allow_module_level=True)
TAPES = {m["called"]: m for m in json.loads((FIXTURES / "manifest.json").read_text())}


def _tail(called):
    return corpora.wav_mono(FIXTURES / TAPES[called]["file"]).astype(float)


def _stream(hs, x):
    for i in range(0, len(x), answer.CHUNK):
        hs.on_rx_stream(x[i:i + answer.CHUNK])


def test_k5dat_nak_resends_the_link_setup():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, called="K5DAT-13", bw="500")
    _stream(hs, _tail("K5DAT-13"))
    assert _resent(hs, io)
    assert any("link-setup NAK" in m and "K5DAT-13" in m for m in io.log_lines)


@pytest.mark.parametrize("called", ["K5FIT", "N0XYZ"])
def test_the_nak_names_its_station(called):
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, called=called, bw="500")
    _stream(hs, _tail("K5DAT-13"))
    assert hs.step == VA._I_LINKSETUP_SENT and not io.sent


def test_k5fit_connected_ack_is_not_a_nak():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, called="K5FIT", bw="500")
    _stream(hs, _tail("K5FIT"))
    assert hs.state == VA.VaraState.CONNECTED
    assert not any("link-setup NAK" in m for m in io.log_lines)


def test_k5fit_first_over_is_read_and_acknowledged(monkeypatch):
    """The whole call replayed: connect, link-setup, ack, and the greeting.

    The 0924 tap brings the link up at 57.6 s. Its first over delivers the same
    43-byte greeting and draws a continue, without a keepalive before it.
    """
    monkeypatch.setattr(answer._RecordingIO, "data", lambda self, b: None, raising=False)
    x = corpora.c_samples("k5fit-0924-whole-link.flac", external=True)
    io, hs, _ = answer._replay(x / np.abs(x).max(), "K5FIT")
    first = next(i for i, m in enumerate(io.logs) if m.startswith("rx DATA over #1"))
    assert any(m.startswith("rx connected-ack") for m in io.logs[:first])
    assert io.logs[first + 1] == "delivered 43 payload bytes to the host"
    assert "continue burst" in io.logs[first + 2]
    assert not any("keepalive" in m for m in io.logs[:first + 3])
