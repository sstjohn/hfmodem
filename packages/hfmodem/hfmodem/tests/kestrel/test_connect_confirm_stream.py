# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A Trimode responder's seed-289 lattice-position-1 connect confirmation.

K0SI on 80 m answered our FIRST link-setup at +0.12 s with a 16-symbol frame on
the connect-response stream at lattice position 1, then ignored our retries. It
confirms the link the way the turn-request-before-connected route does, and a
stock responder never keys it — so it is gated exactly like that route and never
relaxes the connect gate the ack and turn-request routes hold.
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
    return hs, io


def _confirm_window(called="K0SI"):
    kind = VF.for_bw(VF.SESSION_CONNECT_CONFIRM, "2300")
    burst = MK.synth_burst(called, kind)
    return np.concatenate([np.zeros(int(0.3 * MK.FS)), burst,
                           np.zeros(int(0.12 * MK.FS))])


def test_descriptor_is_position_one_on_the_connect_response_stream():
    kind = VF.SESSION_CONNECT_CONFIRM
    assert kind.seed_off == VF.CONNECT_RESPONSE.seed_off      # the 289 stream
    assert len(kind.preamble) + kind.n_payload == 16
    assert VF.payload_position(VF.payload_bins("K0SI", kind), "K0SI", kind) == 1
    # calibration: the connect-response itself is position 17.
    assert (VF.CONNECT_RESPONSE.preadv - 1) // VF.lattice_step(VF.CONNECT_RESPONSE) == 17


def test_confirmed_link_is_taken_and_keys_nothing():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    assert hs._connect_confirmed(_confirm_window())
    assert hs.state == VA.VaraState.CONNECTED
    assert hs.step == VA._I_CONNECTED
    assert not any("session-confirm" in m for m in io.log_lines)   # confirm=False
    assert any("connect-confirm" in m and "link is up" in m for m in io.log_lines)


def test_a_queue_waiting_on_the_link_leaves_on_an_ask():
    """A host block queued before the link came up must not sit there.

    The turn starts the peer's, so the confirmation brings the link up owing the
    queue a channel; without an ask the session closed with the block still
    queued and not one request keyed.
    """
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    hs.tx_level = 3
    hs._txq = [b"a block the host queued while connecting"]
    assert hs._connect_confirmed(_confirm_window())
    assert hs.state == VA.VaraState.CONNECTED
    assert hs.turn == VA._TURN_ASKED, "the queue was never asked for a channel"
    assert io.keyups == 1 and len(io.sent) == 1     # the ask reached the air
    assert any("turn-request" in m for m in io.log_lines)
    assert hs._txq, "the block is still owed until the turn comes back"


def test_an_empty_queue_leaves_the_turn_with_the_peer():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    assert not hs._txq
    assert hs._connect_confirmed(_confirm_window())
    assert hs.state == VA.VaraState.CONNECTED
    assert hs.turn == VA._TURN_PEER          # the gateway's own request follows
    assert io.keyups == 0 and not io.sent    # nothing was keyed at all


def test_the_confirm_payload_collides_with_bw2300_lattice_position_one():
    """Why `_ANSWER_LATTICE_BY_BW["2300"]` must never be widened down to 1.

    The confirmation IS connect-response lattice position 1, so at BW2300 the two
    are the same fifteen tones and only the bandwidth separates what they mean.
    """
    kind = VF.SESSION_CONNECT_CONFIRM
    resp = VF.CONNECT_RESPONSE
    position_1 = replace(resp, preadv=1 + VF.lattice_step(resp) * 1)
    for call in ("K0SI", "KB5LZK", "W1AW"):
        assert VF.payload_bins(call, kind) == VF.payload_bins(call, position_1)
    assert 1 not in VA._ANSWER_LATTICE_BY_BW["2300"]
    assert VA._ANSWER_LATTICE_BY_BW["2750"] == (0, 1, 2)


def test_both_new_frames_are_in_the_burst_registry():
    assert VF.BURSTS[VF.SESSION_CONNECT_CONFIRM.name] is VF.SESSION_CONNECT_CONFIRM
    assert VF.BURSTS[VF.SESSION_OVER_NAK_RESPONDER.name] is VF.SESSION_OVER_NAK_RESPONDER


def test_a_foreign_confirmation_is_not_taken():
    hs, _ = _awaiting(VA._I_LINKSETUP_SENT)
    assert not hs._connect_confirmed(_confirm_window("N0XYZ"))
    assert hs.state == VA.VaraState.CONNECTING


def test_a_stock_connect_response_does_not_confirm_the_link_here():
    # Position 17 keyed to K0SI is the stock offer, not this route's position 1.
    kind = VF.for_bw(VF.CONNECT_RESPONSE, "2300")
    burst = MK.synth_burst("K0SI", kind)
    window = np.concatenate([np.zeros(int(0.3 * MK.FS)), burst,
                             np.zeros(int(0.12 * MK.FS))])
    hs, _ = _awaiting(VA._I_LINKSETUP_SENT)
    assert not hs._connect_confirmed(window)
    assert hs.state == VA.VaraState.CONNECTING


def test_the_route_is_gated_to_a_sent_link_setup():
    # Not before a link-setup is out: the connecting branch of _stream_connect_ask
    # only runs at _I_LINKSETUP_SENT.
    hs, _ = _awaiting(VA._I_CR_SENT)
    for at in range(0, len(_confirm_window()), 4800):
        hs.on_rx_stream(_confirm_window()[at:at + 4800])
    assert hs.state == VA.VaraState.CONNECTING       # a connect-confirm cannot fire here


@corpora.requires_onair_seed289_confirm
def test_a_queue_is_not_stranded_by_the_turn_request_route_either():
    """The turn-request-before-connected route leaves the same debt.

    With a block queued the request is declined — "retaining the turn" — but the
    link came up in the PEER's turn, so retaining it retains nothing: on this
    tape the link was up at 2.9 s and the block was still queued at the close,
    through six keepalives and not one request keyed.
    """
    x = corpora.wav_mono(corpora.ONAIR_SEED289_CONFIRM[1])   # 034443Z
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
def test_the_confirm_collides_with_lattice_position_one_only_at_bw2300(bw, collides):
    """`for_bw` swaps the alphabet, not the SEED_OFF: the confirmation stays on
    stream 289 while BW2750's connect-response answers on 579."""
    cr = VF.connect_response(bw)
    cf = VF.for_bw(VF.SESSION_CONNECT_CONFIRM, bw)
    pos_1 = replace(cr, preadv=1 + VF.lattice_step(cr) * 1)
    assert cf.seed_off == VF.CONNECT_RESPONSE.seed_off == 289
    same = VF.payload_bins("K0SI", cf) == VF.payload_bins("K0SI", pos_1)
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
def test_the_confirm_route_is_reached_only_after_a_link_setup(step):
    """The confirmation is read in one step only, and `_stream_connect_ask` runs
    ahead of the connect-response search, so there it takes precedence."""
    hs, _ = _awaiting(step, bw="2300")
    _drive_stream(hs, VF.handshake_tones(
        "K0SI", replace(VF.CONNECT_RESPONSE,
                        preadv=1 + VF.lattice_step(VF.CONNECT_RESPONSE))))
    if step == VA._I_LINKSETUP_SENT:
        assert hs.state == VA.VaraState.CONNECTED    # the confirmation claims it
        assert not hs.answers                        # never a below-offer report
    else:
        assert hs.state == VA.VaraState.CONNECTING   # no link-setup, no confirm
        assert not hs.answers                        # and 1 is not in 2300's table


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


@corpora.requires_onair_seed289_confirm
def test_k0si_first_link_setup_is_confirmed_on_the_stream():
    x = corpora.wav_mono(corpora.ONAIR_SEED289_CONFIRM[0])   # 034218Z
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    for i in range(0, len(x), 4800):
        hs.on_rx_stream(x[i:i + 4800])
        if hs.state == VA.VaraState.CONNECTED:
            break
    assert hs.state == VA.VaraState.CONNECTED
    line = next(m for m in io.log_lines if "connect-confirm" in m)
    assert "K0SI" in line and "link is up" in line


@pytest.mark.parametrize("chunk", [960, 4096, 4800])
@pytest.mark.parametrize("queued", [False, True])
def test_prompt_recorded_confirmation_is_seen_before_it_goes_stale(chunk, queued):
    # Exact receive cursor after the first setup, including its 100 ms TX guard.
    # Previously the first search waited for a 32-symbol request; this shorter
    # reply had already fallen outside the confirmation freshness window.
    path = Path(__file__).with_name("fixtures") / "connect-confirm-0920" / "k0si-040853.wav"
    if not path.is_file():
        pytest.skip(f"optional recording is not included: {path}")
    with wave.open(str(path)) as wav:
        audio = np.frombuffer(wav.readframes(wav.getnframes()), "<i2") / 32768.0
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    if queued:
        hs.tx_level = 3
        hs._txq = [b"mail proposal queued during setup"]
    accepted = None
    for at in range(0, len(audio), chunk):
        hs.on_rx_stream(audio[at:at + chunk])
        if hs.state == VA.VaraState.CONNECTED:
            accepted = min(at + chunk, len(audio)) / MK.FS
            break
    assert accepted is not None
    assert 0.71 <= accepted <= 0.96
    assert any("connect-confirm" in m for m in io.log_lines)
    assert io.keyups == len(io.sent) == int(queued)
    assert hs.turn == (VA._TURN_ASKED if queued else VA._TURN_PEER)
    if queued:
        assert hs._txq == [b"mail proposal queued during setup"]


@pytest.mark.parametrize("bw", ["500", "2300", "2750"])
@pytest.mark.parametrize("delay", [0.0, 0.035, 0.3, 1.4])
def test_short_confirmation_cadence_covers_early_and_late_replies(bw, delay):
    hs, io = _awaiting(VA._I_LINKSETUP_SENT, bw=bw)
    burst = MK.synth_burst("K0SI", VF.for_bw(VF.SESSION_CONNECT_CONFIRM, bw))
    audio = np.r_[np.zeros(round(delay * MK.FS)), burst, np.zeros(MK.FS // 4)]
    for at in range(0, len(audio), 960):
        hs.on_rx_stream(audio[at:at + 960])
        if hs.state == VA.VaraState.CONNECTED:
            assert at + 960 >= round(delay * MK.FS) + len(burst)
            break
    assert hs.state == VA.VaraState.CONNECTED
    assert not io.sent


def test_a_stale_short_confirmation_is_not_taken_from_a_delayed_poll():
    hs, io = _awaiting(VA._I_LINKSETUP_SENT)
    burst = MK.synth_burst("K0SI", VF.SESSION_CONNECT_CONFIRM)
    hs.on_rx_stream(np.r_[burst, np.zeros(2 * MK.FS)])
    assert hs.state == VA.VaraState.CONNECTING
    assert not io.sent
