# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Lost and rejected overs must not advance the outbound mail stream."""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_vara_giveup import _connected


def _nak():
    return np.concatenate((np.zeros(12000),
                           MK.synth_tone_pairs(VF.NAK_RESPONDER_2300),
                           np.zeros(12000)))


def _sending():
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b"A" * phy.payload_size("2300") + b"second block")
    return hs, io


@pytest.mark.parametrize("stream", (False, True))
def test_nak_repeats_the_same_over_before_the_next_mail_block(stream):
    hs, io = _sending()
    original = io.sent[0].copy()
    audio = _nak()
    assert hs._peer_nak(audio)
    assert not hs._peer_over_continue(audio)
    if stream:
        for i in range(0, len(audio), VA._STREAM_BLOCK):
            hs.on_rx_stream(audio[i:i + VA._STREAM_BLOCK])
        hs.on_rx_audio(audio)  # The bracket must not retry the stream's NAK twice.
    else:
        hs.on_rx_audio(audio)
    assert len(io.sent) == 2
    np.testing.assert_array_equal(io.sent[-1], original)
    assert hs._over == 1 and hs._txq == [b"second block"]
    hs._took_control_burst()
    assert hs._over == 2 and not hs._txq
    assert hs._tx_pending is not None
    hs._took_control_burst()
    assert hs._tx_pending is None and hs.turn == VA._TURN_PEER


def test_a_missing_final_answer_queries_without_repeating_the_block():
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b"only block")
    hs.idle_keepalive()
    assert not np.array_equal(io.sent[0], io.sent[1])
    assert hs._final_query_attempts == 1 and hs._tx_pending is not None
    assert hs.turn == VA._TURN_OURS and hs._over == 1


def test_a_damaged_nak_is_not_promoted_to_a_continue():
    hs, io = _sending()
    pairs = list(VF.NAK_RESPONDER_2300)
    pairs[3] = (40, 90)
    audio = np.concatenate((np.zeros(12000), MK.synth_tone_pairs(pairs),
                            np.zeros(12000)))
    assert not hs._peer_nak(audio)
    assert not hs._peer_over_continue(audio)
    hs.on_rx_audio(audio)
    assert len(io.sent) == 1 and hs._over == 1
    hs.idle_keepalive()
    assert hs._intermediate_query_for == hs._tx_pending and hs._tx_retries == 0
    assert not np.array_equal(io.sent[0], io.sent[1])


def test_host_writes_wait_behind_the_unacknowledged_block():
    hs, io = _sending()
    hs.send(b"third block")
    assert len(io.sent) == 1
    assert hs._txq == [b"second block", b"third block"]


def test_refused_retry_preserves_the_frame_and_retry_budget():
    hs, io = _sending()
    original = io.sent[0].copy()
    io.transmits = False
    hs.on_rx_audio(_nak())
    assert hs._tx_retries == 0 and hs._over == 1
    assert hs._txq == [b"second block"]
    io.transmits = True
    hs.on_rx_audio(_nak())  # A newly decoded NAK licenses the retry.
    assert hs._tx_retries == 1
    np.testing.assert_array_equal(io.sent[-1], original)


@pytest.mark.parametrize("nak", (False, True))
def test_unanswered_retries_are_bounded(nak):
    hs, io = _sending()
    for _ in range(VA._OVER_RETRY_MAX + 1):
        if nak:
            hs.on_rx_audio(_nak())
        else:
            hs._intermediate_query_at -= 4.6
            hs.idle_keepalive()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert hs._over == 1 and hs._txq == [b"second block"]
    assert hs._tx_pending is not None  # No acknowledgment was invented.


@pytest.mark.parametrize("turn", (VA._TURN_OURS, VA._TURN_PEER))
def test_a_refused_release_is_retried_when_the_transport_recovers(turn):
    hs, io = _connected(False)
    hs.turn = turn
    assert not hs._release_turn()
    assert hs.turn == turn and hs._release_owed
    assert not hs._released and not hs._handed_over
    io.transmits = True
    hs.idle_keepalive()
    expected = MK.synth_burst(hs.called, VF.SESSION_TURN_RELEASE)
    np.testing.assert_array_equal(io.sent[-1], expected)
    assert hs.turn == VA._TURN_PEER and hs._released and hs._handed_over
    assert not hs._release_owed


def test_a_new_session_does_not_replay_an_old_unacknowledged_frame():
    hs, io = _sending()
    hs.originate("K7ABC")
    assert hs._tx_pending is None and hs._tx_retries == 0


@pytest.mark.parametrize("fault", ("nak",))
def test_mail_finishes_byte_exact_after_an_outbound_fault(fault):
    from hfmodem.station.mail import VaraLoopback
    from hfmodem.winlink import B2FSession, MailExchange, compose

    message = compose("W9SSJ", "SMTP:op@example.net", "retry proof",
                      np.random.default_rng(8).bytes(350))
    session = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         outbox=[message])
    transport = VaraLoopback("W9SSJ", "K7ABC")
    on_air = transport.peer.on_air
    overs = []
    rejected = []

    def reject_once(samples):
        if (transport.peer.hs.state == VA.VaraState.CONNECTED
                and len(samples) >= VA._DATA_OVER_MIN):
            overs.append(samples.copy())
            if len(overs) == 3:
                rejected.append(samples.copy())
                transport._to_station.append(_nak())
                return
            if len(overs) == 4:
                np.testing.assert_array_equal(samples, rejected[0])
        on_air(samples)

    transport.peer.on_air = reject_once
    report = MailExchange(session, transport, max_steps=200).run()
    assert rejected
    assert session.done and not session.failure, report
    assert session.sent_mids == [message.mid]
    assert [m.render() for m in transport.rms.inbox] == [message.render()]
