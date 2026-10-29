# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A station listening at BW2750 answers a BW2300 request and runs the session at
2300, as stock does; BW500 and BW2300 answer only their own  [spec 04 §4.2.2].

The requests are stock VARA's own, off the 2026-07-21 loopback: AAAA1 calls BBBB2
at BW2750 at 0.597 s and at BW2300 at 56.034 s."""
from functools import cache

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.corpora import BW2750_CAPTURE, requires_bw2750_capture
from hfmodem.tests.kestrel.test_stock_caller_connect import IO

REQUEST_2750, REQUEST_2300 = (0.1, 2.5), (55.5, 58.0)


class HostIO(IO):
    def __init__(self):
        super().__init__()
        self.host = []

    def pending(self):
        self.host.append("PENDING")

    def connected(self, caller, called, bw):
        self.host.append(f"CONNECTED {caller} {called} {bw}")


@cache
def _tape():
    # The harness never closed this file: its RIFF size is 0, the samples start
    # at byte 44.
    a = np.fromfile(str(BW2750_CAPTURE), dtype=np.float32, offset=44).astype(float)
    return a / np.abs(a).max()


def _request(span):
    return _tape()[int(span[0] * MK.FS):int(span[1] * MK.FS)]


def _listen(bw, **kw):
    io = HostIO()
    hs = VA.VaraStationHandshake(["BBBB2"], io, bw=bw, **kw)
    hs.listen(True)
    return hs, io


def _stream(hs, x):
    for at in range(0, len(x), 1024):
        hs.on_rx_stream(x[at:at + 1024])


@requires_bw2750_capture
@pytest.mark.parametrize("route", ["stream", "bracket"])
def test_a_2750_listener_answers_the_stock_2300_request_at_2300(route):
    hs, io = _listen("2750")
    x = _request(REQUEST_2300)
    _stream(hs, x) if route == "stream" else hs.on_rx_audio(x)
    assert len(io.keyed) == 1, io.lines
    np.testing.assert_allclose(
        io.keyed[0], MK.synth_burst("BBBB2", VF.connect_response("2300")))
    assert hs.bw == "2300"
    assert io.host == ["PENDING"]


@requires_bw2750_capture
def test_the_host_sees_the_session_at_2300_and_it_keys_there():
    hs, io = _listen("2750")
    _stream(hs, _request(REQUEST_2300))
    assert hs._take_link_setup(VF.link_setup_frame("AAAA1", bw="2300", high_speed=False,
                                                   compression="OFF"))
    assert io.host == ["PENDING", "CONNECTED AAAA1 BBBB2 2300"]
    keyed = len(io.keyed)
    hs.send(b"hello at 2300\r")
    # The caller's confirmation grants the first answering turn at every HF
    # bandwidth; the write waits for it and goes out as a 2300 DATA over.
    assert len(io.keyed) == keyed and hs._awaiting_caller_confirm
    confirm = MK.synth_burst("BBBB2", VF.for_bw(VF.SESSION_CONFIRM, "2300"))
    _stream(hs, np.concatenate([np.zeros(4800), confirm, np.zeros(24000)]))
    assert hs.turn == VA._TURN_OURS and hs._tx_pending is not None, io.lines
    assert len(io.keyed) == keyed + 1 and hs.bw == "2300"
    assert io.lines[-1].startswith("tx DATA over #1"), io.lines


@pytest.mark.parametrize("bw", ["2300", "2750"])
@pytest.mark.parametrize("route", ["stream", "burst"])
def test_mfsk_only_responder_requests_the_turn_for_host_data(bw, route):
    hs, io = _listen("2750", mfsk_only=True)
    request = MK.synth_burst("BBBB2", VF.connect_request(bw))
    _stream(hs, request) if route == "stream" else hs.on_rx_audio(request)
    assert io.host == ["PENDING", f"CONNECTED UNKNOWN BBBB2 {bw}"]
    assert not hs._awaiting_caller_confirm and hs._released_form is None
    keyed = len(io.keyed)
    hs.send(b"hello from the responder\r")
    assert len(io.keyed) == keyed + 1, io.lines
    assert hs.turn == VA._TURN_ASKED and hs._tx_pending is None
    np.testing.assert_allclose(
        io.keyed[-1], MK.synth_burst(hs.caller, VF.for_bw(VF.SESSION_TURN_REQUEST, bw)))


@requires_bw2750_capture
def test_relistening_returns_to_2750():
    hs, io = _listen("2750", tx_level=2)
    _stream(hs, _request(REQUEST_2300))
    assert hs.bw == "2300" and hs.tx_level is None
    hs.state = VA.VaraState.DISCONNECTED
    hs.listen(True)
    assert (hs.bw, hs.tx_level) == ("2750", 2)
    _stream(hs, _request(REQUEST_2750))
    assert hs.bw == "2750" and len(io.keyed) == 2
    np.testing.assert_allclose(
        io.keyed[-1], MK.synth_burst("BBBB2", VF.connect_response("2750")))


@requires_bw2750_capture
@pytest.mark.parametrize("bw,span", [("500", REQUEST_2300), ("2300", REQUEST_2750)])
def test_narrower_listeners_answer_only_their_own(bw, span):
    hs, io = _listen(bw)
    x = _request(span)
    _stream(hs, x)
    hs.on_rx_audio(x)
    assert not io.keyed and hs.state is VA.VaraState.LISTENING, io.lines
    assert hs.bw == bw
