# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Lost and rejected overs must not advance the outbound mail stream."""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_vara_giveup import _connected


def _nak(called="KC9GHZ"):
    return np.concatenate((np.zeros(12000),
                           MK.synth_tone_pairs(VF.nak(called, "2300")[1]),
                           np.zeros(12000)))


def _sending():
    """A short over with a second host write behind it. A NAK of it is a DATA
    NACK like a full over's, and the short over is its own delivery boundary
    [doc 28]."""
    hs, io = _connected()
    hs.turn = VA._TURN_OURS
    hs.send(b"first block")
    hs.send(b"second block")
    return hs, io


def _unconfirmed(hs):
    body = hs._tx_pending[0]
    return (phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            + b"".join(hs._txq))


def _resent_a_record_lower(hs):
    body, over, level = hs._tx_pending
    return (over == 1 and level == phy.lower_level("2300", phy.base_level("2300"))
            and phy.vara_payload(body, caller=hs.caller, body_len=len(body))
            == b"first block")


@pytest.mark.parametrize("stream", (False, True))
def test_a_nak_resends_the_over_a_record_lower_before_the_next_mail_block(stream):
    hs, io = _sending()
    audio = _nak()
    assert hs._peer_nak_at(audio) is not None
    assert not hs._peer_over_continue(audio)
    if stream:
        for i in range(0, len(audio), VA._STREAM_BLOCK):
            hs.on_rx_stream(audio[i:i + VA._STREAM_BLOCK])
        hs.on_rx_audio(audio)  # The bracket must not retry the stream's NAK twice.
    else:
        hs.on_rx_audio(audio)
    assert len(io.sent) == 2 and _resent_a_record_lower(hs)
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


def _damaged_nak(hs, damaged):
    hs.turn = VA._TURN_OURS
    hs.send(b"A" * phy.payload_size("2300") + b"second block")
    pairs = list(VF.nak("KC9GHZ", "2300")[1])
    for i in damaged:
        pairs[i] = (40, 90)
    return np.concatenate((np.zeros(12000), MK.synth_tone_pairs(pairs),
                           np.zeros(12000)))


def test_a_damaged_nak_is_not_promoted_to_a_continue():
    hs, io = _connected()
    audio = _damaged_nak(hs, (3, 5))
    assert hs._peer_nak_at(audio) is None and hs._held_nak_at(audio) is None
    assert not hs._peer_over_continue(audio)
    hs.on_rx_audio(audio)
    assert len(io.sent) == 1 and hs._over == 1
    hs.idle_keepalive()
    assert hs._intermediate_query_for == hs._tx_pending and hs._tx_retries == 0
    assert not np.array_equal(io.sent[0], io.sent[1])


def test_a_nak_with_one_pair_off_is_read_whole_as_the_nak():
    # K5FIT 2026-09-27: seven of eight pairs exact, soft 2.75
    hs, io = _connected()
    audio = _damaged_nak(hs, (3,))
    assert hs._peer_nak_at(audio) is None and hs._held_nak_at(audio) is not None
    assert not hs._peer_over_continue(audio)
    hs.on_rx_audio(audio)
    assert len(io.sent) == 2 and hs._over == 1 and hs._tx_retries == 1
    assert hs._tx_pending[2] == phy.lower_level("2300", phy.base_level("2300"))


def test_host_writes_wait_behind_the_unacknowledged_block():
    hs, io = _sending()
    hs.send(b"third block")
    assert len(io.sent) == 1
    assert hs._txq == [b"second block", b"third block"]


def test_refused_retry_preserves_the_frame_and_retry_budget():
    hs, io = _sending()
    original = hs._tx_pending
    io.transmits = False
    hs.on_rx_audio(_nak())
    assert hs._tx_retries == 0 and hs._over == 1
    assert hs._tx_pending == original and hs._txq == [b"second block"]
    io.transmits = True
    hs.on_rx_audio(_nak())  # A newly decoded NAK licenses the retry.
    assert hs._tx_retries == 1 and _resent_a_record_lower(hs)


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
    assert hs._over == 1 and hs._tx_pending is not None
    assert _unconfirmed(hs) == b"first block" + b"second block"


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
                transport._to_station.append(_nak("K7ABC"))
                return
        on_air(samples)

    transport.peer.on_air = reject_once
    report = MailExchange(session, transport, max_steps=200).run()
    assert rejected
    assert session.done and not session.failure, report
    assert session.sent_mids == [message.mid]
    assert [m.render() for m in transport.rms.inbox] == [message.render()]
