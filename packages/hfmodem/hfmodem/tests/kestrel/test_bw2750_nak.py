# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Measured BW2750 unread-over recovery: silence, peer idle, generated request.

Stock caller W9SSJ asked called station KC9GHZ with SESSION_OVER_NAK after an
unread greeting. These state tests do not invent an eight-pair NAK or a responder
role. Native measurement and stock recovery qualify the waveform separately.
"""
import numpy as np
import pytest
from pathlib import Path

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_vara_giveup import _IO

STOCK_NAK = Path(__file__).with_name("fixtures") / "bw2750-nak-0916" / "stock-caller-idle-nak.wav"


class KeyIO(_IO):
    def __init__(self, transmits=True):
        super().__init__(transmits)
        self.keys = []

    def key(self, on):
        self.keys.append(on)


def waiting(*, role="initiator", caller="W9SSJ", transmits=True):
    io = KeyIO(transmits)
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2750")
    hs.role, hs.called, hs.caller = role, "KC9GHZ", caller
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_PEER
    hs.allow_data_retries = False
    return hs, io


def partial_window(hs):
    hs._held_answer = (False, False)
    hs._release_held_answer(VA._ANSWER_HOLD_MAX)


def idle(called="KC9GHZ"):
    return np.concatenate((MK.synth_burst(called, VF.for_bw(VF.SESSION_RESPONDER_IDLE, "2750")),
                           np.zeros(int(.2 * MK.FS))))


def assert_request(audio, called="KC9GHZ"):
    kind = VF.for_bw(VF.SESSION_OVER_NAK, "2750")
    np.testing.assert_array_equal(audio, MK.synth_burst(called, kind))
    assert MK.demod_burst(audio, kind, MK.band_for("2750")) == VF.handshake_tones(called, kind)


def test_generated_recovery_does_not_invent_an_eight_pair_nak():
    hs, io = waiting()
    assert hs._has_idle_over_nak()
    assert VF.nak("W9SSJ", "2750") is None
    assert not hs._tx_nak()
    assert not io.sent and not io.keys
    assert not hs._peer_nak(idle())


@pytest.mark.skipif(not STOCK_NAK.exists(),
                    reason="the stock caller idle-NAK recording requires the source checkout "
                           "(fixtures/bw2750-nak-0916)")
def test_generated_request_matches_all_symbols_of_native_stock_recovery():
    from scipy.io import wavfile
    fs, audio = wavfile.read(STOCK_NAK)
    assert fs == MK.FS
    kind = VF.for_bw(VF.SESSION_OVER_NAK, "2750")
    expected = np.asarray(VF.handshake_tones("KC9GHZ", kind))
    heard = VA._payload_alignment(audio, kind, expected,
                                   band=MK.band_for("2750"))
    assert len(expected) == 32
    np.testing.assert_array_equal(heard, expected)


def test_initial_unread_over_keys_nothing_even_with_data_retries_disabled():
    hs, io = waiting()
    assert not hs.allow_data_retries
    assert hs._undecoded_over()
    assert not io.sent and not io.keys and not io.delivered
    assert hs._answer_owed == VA._OWED_OVER and hs._owed_block and hs._owed_recovery
    assert hs._reacks == hs._since_progress == hs.idle_keyed == 0
    assert hs.turn == VA._TURN_PEER and hs.state == VA.VaraState.CONNECTED


@pytest.mark.parametrize("stream", [False, True])
def test_fresh_peer_idle_draws_called_keyed_generated_request(stream):
    hs, io = waiting()
    hs._undecoded_over()
    cue = np.concatenate((np.zeros(MK.FS // 2), idle(), np.zeros(MK.FS // 2)))
    if stream:
        for at in range(0, len(cue), 960):
            hs.on_rx_stream(cue[at:at + 960])
        hs.on_rx_audio(cue)  # The bracket cannot spend the same idle twice.
    else:
        hs.on_rx_audio(idle())
    assert len(io.sent) == 1
    assert_request(io.sent[0])
    assert hs._owed_block and hs._owed_recovery
    assert hs._reacks == hs._since_progress == hs.idle_keyed == 1
    assert not io.delivered


def test_foreign_peer_idle_does_not_solicit_our_recovery():
    hs, io = waiting()
    hs._undecoded_over()
    hs.on_rx_audio(idle("N0XYZ"))
    assert not io.sent and hs._reacks == 0
    assert hs._owed_block


def test_raw_idle_without_complete_tail_and_guard_keys_nothing_yet():
    hs, io = waiting()
    hs._undecoded_over()
    tone = MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_RESPONDER_IDLE, "2750"))
    cue = np.concatenate((np.zeros(MK.FS // 2), tone[:-MK.HOP]))
    for at in range(0, len(cue), 960):
        hs.on_rx_stream(cue[at:at + 960])
    assert not io.sent and hs._reacks == 0
    tail = np.concatenate((tone[-MK.HOP:], np.zeros(MK.FS // 2)))
    for at in range(0, len(tail), 960):
        hs.on_rx_stream(tail[at:at + 960])
    assert len(io.sent) == 1
    assert_request(io.sent[0])


def test_stale_raw_idle_is_not_a_current_transmit_gap():
    hs, io = waiting()
    hs._undecoded_over()
    tone = MK.synth_burst(hs.called, VF.for_bw(VF.SESSION_RESPONDER_IDLE, "2750"))
    late = np.concatenate((np.zeros(MK.FS // 4), tone, np.zeros(int(1.5 * MK.FS))))
    hs.on_rx_stream(late)
    assert not io.sent and hs._reacks == 0
    assert hs._owed_block


@pytest.mark.parametrize("role,caller", [("responder", "W9SSJ"), ("initiator", "N0XYZ")])
def test_unmeasured_local_role_or_caller_stays_unsupported(role, caller):
    hs, io = waiting(role=role, caller=caller)
    assert not hs._has_idle_over_nak()
    assert not hs._tx_nak()
    assert not io.sent and not io.keys
    assert hs._undecoded_over()
    assert not io.sent and not io.keys
    assert hs._owed_block and not hs._owed_recovery and hs._reacks == 0
    assert any("no NAK is measured" in msg for msg in io.msgs)
    assert not hs._reack()
    if role == "initiator":
        assert hs.state == VA.VaraState.DISCONNECTED
    else:
        # Existing disconnect waveforms are initiator-only too; do not invent
        # responder behavior merely to satisfy a broader close assertion.
        assert not io.sent and not io.keys and hs._owed_block
    assert hs._reacks == 0


def test_refused_generated_request_preserves_debt_and_does_not_spend_budget():
    hs, io = waiting(transmits=False)
    hs._undecoded_over()
    for _ in range(12):
        assert not hs._reack()
    assert not io.sent
    assert hs._reacks == hs._since_progress == hs.idle_keyed == 0
    assert hs._owed_block and hs._owed_recovery
    assert hs.state == VA.VaraState.CONNECTED
    io.transmits = True
    assert hs._reack()
    assert len(io.sent) == 1
    assert_request(io.sent[0])
    assert hs._reacks == hs._since_progress == hs.idle_keyed == 1
    assert hs._owed_block


def test_partial_window_timeout_can_request_its_missing_suffix():
    hs, io = waiting()
    partial_window(hs)
    assert not io.sent and hs._held_answer is None
    assert hs._owed_block
    assert hs._reack()
    assert len(io.sent) == 1
    assert_request(io.sent[0])
    assert hs._owed_block


def test_unread_recovery_repeats_only_generated_request_and_stops_at_budget():
    hs, io = waiting()
    hs._undecoded_over()
    for n in range(1, VA._REACK_MAX + 1):
        assert hs._reack()
        assert len(io.sent) == n and hs._reacks == n
        assert_request(io.sent[-1])
        assert hs._owed_block and not io.delivered
    assert not hs._reack()
    assert hs.state == VA.VaraState.DISCONNECTED
    assert sum("tx NAK" in msg for msg in io.msgs) == VA._REACK_MAX
    assert any("unread window still missing" in msg for msg in io.msgs)
    before = len(io.sent)
    hs.idle_keepalive()
    assert len(io.sent) == before


def test_an_unread_resend_does_not_reset_the_successful_ask_count():
    hs, io = waiting()
    hs._undecoded_over()
    assert hs._reack()
    hs._undecoded_over()
    assert len(io.sent) == 1 and hs._reacks == 1
    assert hs._owed_block and hs._owed_recovery
    assert hs._reack()
    assert hs._reacks == 2 and len(io.sent) == 2
    assert_request(io.sent[-1])


def test_peer_controls_cannot_clear_unread_debt_or_restart_progress():
    hs, _ = waiting()
    hs._undecoded_over()
    hs._since_progress = 2
    before = hs.progress
    hs._took_stall_answer()
    assert hs._owed_block and hs._answer_owed == VA._OWED_OVER
    assert hs._since_progress == 2 and hs.progress == before


def test_peer_gap_and_idle_clock_do_not_charge_the_same_request_twice():
    hs, io = waiting()
    hs._undecoded_over()
    assert hs._reack()
    hs.idle_keepalive()
    assert len(io.sent) == 1 and hs._since_progress == hs.idle_keyed == 1
    hs.idle_keepalive()
    assert len(io.sent) == 2 and hs._since_progress == hs.idle_keyed == 2
    assert_request(io.sent[-1])


def test_partial_window_recovery_delivers_only_missing_suffix_once():
    hs, io = waiting()
    a, b, c = (phy.vara_body(x * 89, "W9SSJ") for x in (b"A", b"B", b"C"))
    hs._deliver([a], hold=True)
    partial_window(hs)
    assert hs._reack()
    assert_request(io.sent[-1])
    hs._deliver([a], hold=True)
    hs._deliver([b], hold=False)
    hs._key_over_answer(last=False, owes_release=False)
    assert io.delivered == [b"A" * 89, b"B" * 89]
    assert not hs._owed_block
    hs._deliver([a, b], hold=False)
    assert io.delivered == [b"A" * 89, b"B" * 89]
    hs._deliver([a], hold=True)
    hs._deliver([c], hold=False)
    assert io.delivered == [b"A" * 89, b"B" * 89, b"A" * 89, b"C" * 89]


def test_explicit_unread_second_frame_cancels_held_positive_ack_of_prefix():
    """Peer idle cannot turn a known unread suffix into an accepted window.

    The first frame was already delivered while the peer was still keying.
    A CRC failure in its second frame makes the held positive answer unsafe;
    that prefix must survive recovery without reaching the host twice.
    """
    hs, io = waiting()
    a, b = (phy.vara_body(x * 89, "W9SSJ") for x in (b"A", b"B"))
    hs._deliver([a], hold=True)
    hs._held_answer = (False, False)
    assert io.delivered == [b"A" * 89] and not io.sent

    # Explicit unread second frame, before the old held-answer timeout.
    hs._undecoded_over()
    assert hs._held_answer is None, "known unread DATA left a positive ACK armed"
    assert hs._owed_block and hs._owed_recovery
    assert not io.sent and io.delivered == [b"A" * 89]

    hs.on_rx_audio(idle())
    assert len(io.sent) == 1
    assert_request(io.sent[0])
    assert hs._owed_block and hs._answer_owed == VA._OWED_OVER

    # Even after the former timeout, the idle cannot justify acknowledging A
    # as if the entire window had ended cleanly.
    hs._release_held_answer(VA._ANSWER_HOLD_MAX)
    assert len(io.sent) == 1 and hs._owed_block
    assert io.delivered == [b"A" * 89]

    # The retransmission includes the known prefix. Only its missing suffix
    # is newly delivered, and only now may the window receive a positive ACK.
    hs._deliver([a], hold=True)
    hs._deliver([b], hold=False)
    assert io.delivered == [b"A" * 89, b"B" * 89]
    hs._key_over_answer(last=False, owes_release=False)
    assert len(io.sent) == 2 and not hs._owed_block


def test_unread_request_preserves_outbound_ownership_and_echo_history():
    hs, io = waiting()
    body = phy.vara_body(b"outbound", "W9SSJ")
    pending = (body, 7, 103)
    hs._tx_pending = pending
    hs._txq = [b"next outbound"]
    hs._keyed_bodies.add(body)
    hs._undecoded_over()
    assert hs._reack()
    assert hs._tx_pending is pending and hs._txq == [b"next outbound"]
    assert hs._keyed_bodies == {body}
    assert hs.turn == VA._TURN_PEER and not io.delivered
    assert len(io.sent) == 1
    assert_request(io.sent[0])


def test_request_transport_exception_still_unkeys_without_charging_recovery():
    hs, io = waiting()
    hs._undecoded_over()

    def fail(_samples):
        raise RuntimeError("synthetic transport failure")

    io.tx = fail
    with pytest.raises(RuntimeError, match="synthetic transport failure"):
        hs._reack()
    assert io.keys == [True, False]
    assert hs._owed_block and hs._reacks == hs._since_progress == 0
