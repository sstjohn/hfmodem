# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The seed-289 lattice-position-1 frame behind our link-setup is the peer's NAK.

K0SI on 80 m keyed it at +0.12 s behind our link-setup, on 09-16, 09-19 and 09-23,
and ignored the retries that came 8-12 s later. It was read as a connect
confirmation, and the links it brought up carried nothing. Stock keys the same
payload on a CRC failure, so it is read where a link-setup of ours is outstanding
and answered by keying the link-setup again at once.
"""
from dataclasses import replace
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_response_by_stream import _IO


class AirIO(_IO):
    """`_IO` that can also see the air: what was keyed and what went out."""

    def __init__(self):
        super().__init__()
        self.keyups = 0
        self.sent: list[np.ndarray] = []

    def key(self, on):
        if on:
            self.keyups += 1

    def tx(self, samples):
        self.sent.append(np.asarray(samples, float))

    def tx_went_out(self):
        return True


def _awaiting(step, called="K0SI", state=VA.VaraState.CONNECTING, bw="2300"):
    io = AirIO()
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw=bw)
    hs.role, hs.called, hs.caller = "initiator", called, "W9SSJ"
    hs.state, hs.step = state, step
    if step == VA._I_LINKSETUP_SENT:
        hs._linksetup_tx = 1
    return hs, io


def _nak_window(called="K0SI", bw="2300"):
    burst = MK.synth_burst(called, VF.for_bw(VF.SESSION_LINK_SETUP_NAK, bw))
    return np.concatenate([np.zeros(int(0.3 * MK.FS)), burst,
                           np.zeros(int(0.12 * MK.FS))])


def _resent(hs, io):
    return (hs.state == VA.VaraState.CONNECTING and hs.step == VA._I_LINKSETUP_SENT
            and hs._linksetup_tx == 2 and io.keyups == len(io.sent) == 1)


def test_descriptor_is_position_one_on_the_connect_response_stream():
    kind = VF.SESSION_LINK_SETUP_NAK
    assert kind.seed_off == VF.CONNECT_RESPONSE.seed_off      # the 289 stream
    assert len(kind.preamble) + kind.n_payload == 16
    assert VF.payload_position(VF.payload_bins("K0SI", kind), "K0SI", kind) == 1
    # calibration: the connect-response itself is position 17.
    assert (VF.CONNECT_RESPONSE.preadv - 1) // VF.lattice_step(VF.CONNECT_RESPONSE) == 17


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_the_nak_resends_the_link_setup(bw):
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, bw=bw)
    assert hs._link_setup_nacked(_nak_window(bw=bw))
    assert _resent(hs, io)
    assert any("link-setup NAK" in m for m in io.log_lines)
    assert not any("link is up" in m for m in io.log_lines)


def test_a_queued_block_does_not_bring_the_link_up():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    hs._txq = [b"a block the host queued while connecting"]
    assert hs._link_setup_nacked(_nak_window())
    assert _resent(hs, io) and hs._txq
    assert not any("turn-request" in m for m in io.log_lines)


def test_the_nak_payload_collides_with_bw2300_lattice_position_one():
    """Why `_ANSWER_LATTICE_BY_BW["2300"]` must never be widened down to 1.

    The NAK IS connect-response lattice position 1, so at BW2300 the two are the
    same fifteen tones and only the bandwidth separates what they mean.
    """
    kind = VF.SESSION_LINK_SETUP_NAK
    resp = VF.CONNECT_RESPONSE
    position_1 = replace(resp, preadv=1 + VF.lattice_step(resp) * 1)
    for call in ("K0SI", "KB5LZK", "W1AW"):
        assert VF.payload_bins(call, kind) == VF.payload_bins(call, position_1)
    assert 1 not in VA._ANSWER_LATTICE_BY_BW["2300"]
    assert VA._ANSWER_LATTICE_BY_BW["2750"] == (0, 1, 2)


def test_both_new_frames_are_in_the_burst_registry():
    assert VF.BURSTS[VF.SESSION_LINK_SETUP_NAK.name] is VF.SESSION_LINK_SETUP_NAK
    assert VF.BURSTS[VF.SESSION_OVER_NAK_RESPONDER.name] is VF.SESSION_OVER_NAK_RESPONDER


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
def test_a_foreign_nak_is_not_taken(bw):
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, bw=bw)
    assert not hs._link_setup_nacked(_nak_window("N0XYZ", bw))
    assert hs.state == VA.VaraState.CONNECTING and not io.sent


def test_a_stock_connect_response_is_not_a_nak():
    # Position 17 keyed to K0SI is the stock offer, not this route's position 1.
    kind = VF.for_bw(VF.CONNECT_RESPONSE, "2300")
    burst = MK.synth_burst("K0SI", kind)
    window = np.concatenate([np.zeros(int(0.3 * MK.FS)), burst,
                             np.zeros(int(0.12 * MK.FS))])
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    assert not hs._link_setup_nacked(window)
    assert not io.sent


def test_the_route_is_gated_to_a_sent_link_setup():
    # Not before a link-setup is out: the connecting branch of _stream_connect_ask
    # only runs at _I_LINKSETUP_SENT.
    hs, io = _awaiting(VA._I_CR_SENT)
    for at in range(0, len(_nak_window()), 4800):
        hs.on_rx_stream(_nak_window()[at:at + 4800])
    assert hs.state == VA.VaraState.CONNECTING and not io.sent


def test_a_turn_request_still_brings_the_link_up_and_asks_for_the_queue():
    """The turn-request-before-connected route is a connect, and leaves a debt.

    With a block queued the request is declined — "retaining the turn" — but the
    link came up in the PEER's turn, so retaining it retains nothing. The 0920
    K0SI tap starts here at 67 s, after the connect-response; replaying that
    response in LINKSETUP_SENT would first resend our link-setup.
    """
    x = corpora.c_samples("k0si-0920-turn-request-67-72.flac")
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    hs.tx_level = 3
    hs._txq = [b"a block the host queued while connecting"]
    for i in range(0, len(x), 4800):
        hs.on_rx_stream(x[i:i + 4800])
        if hs.state == VA.VaraState.CONNECTED:
            break
    assert hs.state == VA.VaraState.CONNECTED
    assert any("turn-request from K0SI" in m and "link is up" in m
               for m in io.log_lines)
    assert hs.turn == VA._TURN_ASKED, "the queued block was stranded"
    assert io.keyups == 1 and len(io.sent) == 1


# --------------------------------------------------------------------------- #
# Which route claims a position-1 payload, and why the answer differs by bandwidth.
@pytest.mark.parametrize("bw,collides", [("2300", True), ("2750", False)])
def test_the_nak_collides_with_lattice_position_one_only_at_bw2300(bw, collides):
    """`for_bw` swaps the alphabet, not the SEED_OFF: the NAK stays on stream 289
    while BW2750's connect-response answers on 579."""
    cr = VF.connect_response(bw)
    nak = VF.for_bw(VF.SESSION_LINK_SETUP_NAK, bw)
    pos_1 = replace(cr, preadv=1 + VF.lattice_step(cr) * 1)
    assert nak.seed_off == VF.CONNECT_RESPONSE.seed_off == 289
    same = VF.payload_bins("K0SI", nak) == VF.payload_bins("K0SI", pos_1)
    assert same is collides
    if collides:
        assert 1 not in VA._ANSWER_LATTICE_BY_BW[bw]     # never widen it down to 1
    else:
        assert 1 in VA._ANSWER_LATTICE_BY_BW[bw]         # safe: a different stream


def _drive_stream(hs, tones):
    seg = np.concatenate([np.zeros(int(0.3 * MK.FS)), MK.synth_tones(tones),
                          np.zeros(int(0.9 * MK.FS))])
    for i in range(0, len(seg), 4800):
        hs.on_rx_stream(seg[i:i + 4800])


@pytest.mark.parametrize("step", [VA._I_CR_SENT, VA._I_LINKSETUP_SENT])
def test_the_nak_route_is_reached_only_after_a_link_setup(step):
    """The NAK is read in one step only, and `_stream_connect_ask` runs ahead of
    the connect-response search, so there it takes precedence."""
    hs, io = _awaiting(step, bw="2300")
    _drive_stream(hs, VF.handshake_tones(
        "K0SI", replace(VF.CONNECT_RESPONSE,
                        preadv=1 + VF.lattice_step(VF.CONNECT_RESPONSE))))
    assert hs.state == VA.VaraState.CONNECTING
    assert not hs.answers                            # never a below-offer report
    assert len(io.sent) == (step == VA._I_LINKSETUP_SENT)


@pytest.mark.parametrize("step", [VA._I_CR_SENT, VA._I_LINKSETUP_SENT])
def test_bw2750_position_one_is_the_below_offer_report_in_either_step(step):
    """At BW2750 the two readings are different streams, so the report stands
    whether or not a link-setup is outstanding."""
    hs, _ = _awaiting(step, bw="2750")
    cr = VF.connect_response("2750")
    _drive_stream(hs, VF.handshake_tones(
        "K0SI", replace(cr, preadv=1 + VF.lattice_step(cr))))
    assert [a.position for a in hs.answers] == [1]
    assert hs.state == VA.VaraState.CONNECTING       # a report, never a connect


def test_k0si_nak_of_the_first_link_setup_is_read_on_the_stream():
    x = corpora.c_samples("k0si-0920-whole-setup-nak.flac", external=True)
    hs, io = _awaiting(VA._I_CR_SENT)
    for i in range(0, len(x), 4800):
        hs.on_rx_stream(x[i:i + 4800])
        if len(io.sent) == 2 or hs.state != VA.VaraState.CONNECTING:
            break
    assert hs.state == VA.VaraState.CONNECTING and hs._linksetup_tx == 2
    assert [m.split(" asking")[0] for m in io.log_lines
            if m.startswith("tx link-setup")] == ["tx link-setup caller=W9SSJ level=4"] * 2
    assert any(m.startswith("rx link-setup NAK (289/31) from K0SI") for m in io.log_lines)


@pytest.mark.parametrize("chunk", [960, 4096, 4800])
def test_a_prompt_recorded_nak_is_answered_before_it_goes_stale(chunk):
    # Exact receive cursor after the first setup, including its 100 ms TX guard.
    path = Path(__file__).with_name("fixtures") / "connect-confirm-0920" / "k0si-040853.wav"
    if not path.is_file():
        pytest.skip(f"optional recording is not included: {path}")
    with wave.open(str(path)) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), "<i2") / 32768.0
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    keyed = None
    for at in range(0, len(audio), chunk):
        hs.on_rx_stream(audio[at:at + chunk])
        if io.sent:
            keyed = min(at + chunk, len(audio)) / MK.FS
            break
    assert keyed is not None and 0.71 <= keyed <= 0.96
    assert _resent(hs, io)


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
@pytest.mark.parametrize("delay", [0.0, 0.035, 0.3, 1.4])
def test_the_nak_cadence_covers_early_and_late_replies(bw, delay):
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, bw=bw)
    burst = MK.synth_burst("K0SI", VF.for_bw(VF.SESSION_LINK_SETUP_NAK, bw))
    audio = np.r_[np.zeros(round(delay * MK.FS)), burst, np.zeros(MK.FS // 4)]
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
        if io.sent:
            assert at + 960 >= round(delay * MK.FS) + len(burst)
            break
    assert _resent(hs, io)


def test_a_stale_nak_is_not_taken_from_a_delayed_poll():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    burst = MK.synth_burst("K0SI", VF.SESSION_LINK_SETUP_NAK)
    hs.on_rx_stream(np.r_[burst, np.zeros(2 * MK.FS)])
    assert hs.state == VA.VaraState.CONNECTING
    assert not io.sent
