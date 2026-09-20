# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The 288/1117 responder over-NAK at BW2750, and the bounded resend it licenses.

KC9GHZ answered every final-answer query of ours on the 2026-09-16 BW2750 tape
with SESSION_OVER_NAK_RESPONDER — the responder counterpart of the NAK this
station keys during recovery. A keyed NAK is a positive ask, so it resends the
outstanding over once even where host retries are disabled; the retry budget
still bounds it.
"""
import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel import corpora
from hfmodem.tests.kestrel.test_vara_giveup import _IO


class KeyIO(_IO):
    def __init__(self, transmits=True):
        super().__init__(transmits)
        self.keys = []
        self.now = 0.0          # tape clock, set by the live driver
        self.keyups: list[float] = []

    def key(self, on):
        self.keys.append(on)
        if on:
            self.keyups.append(self.now)


def sending(*, allow_retries=False, transmits=True):
    io = KeyIO(transmits)
    hs = VA.VaraStationHandshake(["W9SSJ"], io, bw="2750")
    hs.role, hs.called, hs.caller = "initiator", "KC9GHZ", "W9SSJ"
    hs.state, hs.step = VA.VaraState.CONNECTED, VA._I_CONNECTED
    hs.turn = VA._TURN_OURS
    hs.allow_data_retries = allow_retries
    hs.tx_level = 103
    hs._tx_pending = (phy.vara_body(b"outbound", "W9SSJ"), 3, 103)
    return hs, io


def nak_burst(called="KC9GHZ"):
    kind = VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750")
    return np.concatenate((MK.synth_burst(called, kind), np.zeros(int(.3 * MK.FS))))


def test_frame_is_the_responder_offset_of_the_caller_nak():
    assert VF.SESSION_OVER_NAK_RESPONDER.seed_off == VF.SESSION_OVER_NAK.seed_off + 228
    assert VF.SESSION_OVER_NAK_RESPONDER.preadv == VF.SESSION_OVER_NAK.preadv
    assert len(VF.SESSION_OVER_NAK_RESPONDER.preamble) + \
        VF.SESSION_OVER_NAK_RESPONDER.n_payload == 32


#: Every other 32-symbol frame drawn from the responder's own SEED_OFF 288. The
#: NAK has to be separable from each of them, because they arrive in the same
#: turnarounds and only this one licenses a resend.
SIBLINGS_288 = (VF.SESSION_INTERMEDIATE_QUERY_ANSWER, VF.SESSION_RESPONDER_OVER_IDLE,
                VF.SESSION_RESPONDER_IDLE, VF.SESSION_DRAINED_RESPONDER,
                VF.SESSION_RESPONDER_OVER_ANSWER)


def test_recognizer_takes_the_synthesised_frame_and_rejects_a_foreign_one():
    hs, _ = sending()
    assert hs._peer_responder_nak(nak_burst("KC9GHZ"))
    assert not hs._peer_responder_nak(nak_burst("N0XYZ"))


@pytest.mark.parametrize("kind", SIBLINGS_288, ids=lambda k: k.name)
def test_no_sibling_of_the_288_family_reads_as_the_nak(kind):
    hs, _ = sending()
    assert kind.seed_off == VF.SESSION_OVER_NAK_RESPONDER.seed_off
    burst = np.concatenate((MK.synth_burst("KC9GHZ", VF.for_bw(kind, "2750")),
                            np.zeros(int(.3 * MK.FS))))
    assert not hs._peer_responder_nak(burst)


@pytest.mark.parametrize("kind", [VF.SESSION_OVER_NAK, VF.SESSION_TURN_IDLE],
                         ids=lambda k: k.name)
def test_the_callers_own_frames_do_not_read_as_the_responder_nak(kind):
    hs, _ = sending()
    for call in ("KC9GHZ", "W9SSJ"):
        burst = np.concatenate((MK.synth_burst(call, VF.for_bw(kind, "2750")),
                                np.zeros(int(.3 * MK.FS))))
        assert not hs._peer_responder_nak(burst)


def test_nak_resends_the_outstanding_over_under_disabled_retries():
    hs, io = sending(allow_retries=False)
    assert not hs.allow_data_retries
    assert hs._stream_answer(nak_burst())
    assert len(io.sent) == 1 and hs._tx_retries == 1
    assert hs._tx_pending is not None
    assert any("responder NAK" in m and "repeating" in m for m in io.msgs)
    assert hs.state == VA.VaraState.CONNECTED


def test_resend_is_bounded_by_the_retry_budget():
    hs, io = sending(allow_retries=False)
    for n in range(1, VA._OVER_RETRY_MAX + 1):
        hs._reset_answer_search()
        assert hs._stream_answer(nak_burst())
        assert hs._tx_retries == n and len(io.sent) == n
    # Budget spent: the next NAK closes rather than keying a fourth over.
    hs._reset_answer_search()
    assert hs._stream_answer(nak_burst())
    assert hs._tx_retries == VA._OVER_RETRY_MAX      # no fourth over
    assert hs.state == VA.VaraState.DISCONNECTED


def test_nak_with_nothing_pending_keys_nothing():
    hs, io = sending(allow_retries=False)
    hs._tx_pending = None
    assert hs._stream_answer(nak_burst())
    assert not io.sent
    assert any("nothing outstanding" in m for m in io.msgs)


@corpora.requires_onair_responder_nak_2750
@pytest.mark.parametrize("at", [69.312, 81.093, 92.837])
def test_frame_is_recognized_on_the_kc9ghz_tape(at):
    x = corpora.wav_mono(corpora.ONAIR_RESPONDER_NAK_2750)
    kind = VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750")
    span = VA._span(kind)
    a = int((at - 0.2) * MK.FS)
    seg = x[a:a + span + int(0.6 * MK.FS)]
    tones = np.asarray(VF.handshake_tones("KC9GHZ", kind), dtype=np.int32)
    heard = VA._payload_alignment(seg, kind, tones, band=MK.band_for("2750"))
    assert VF.recognize([int(t) for t in heard], "KC9GHZ", kind)


#: Where the three NAKs start on the tape, and so where each one's last symbol is.
TAPE_NAKS = (69.312, 81.093, 92.837)


def _drive_live(x, t0, t1, block=0.125):
    """The tape through `_stream_answer` exactly as the receive path feeds it."""
    hs, io = sending(allow_retries=False)
    n = int(block * MK.FS)
    for a in range(int(t0 * MK.FS), int(t1 * MK.FS), n):
        io.now = a / MK.FS + block
        hs._stream_answer(x[a:a + n])
    return hs, io


@corpora.requires_onair_responder_nak_2750
@pytest.mark.parametrize("start", TAPE_NAKS)
def test_the_resend_is_keyed_after_the_naks_last_symbol(start):
    """The over answering a NAK must not go out on top of the NAK.

    The arm opens at `_SESSION_MIN` (0.726 s) and the frame is 1.366 s, so a fit
    named off its first two thirds used to key 0.616 s inside the burst — a 4.4 s
    over across a half-duplex peer still transmitting, and two rungs of the retry
    budget spent on one ask.
    """
    x = corpora.wav_mono(corpora.ONAIR_RESPONDER_NAK_2750)
    last_symbol = start + VA._span(VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750")) / MK.FS
    hs, io = _drive_live(x, start - 1.5, start + 4.0)
    assert io.keyups, "the NAK drew no resend at all"
    assert len(io.keyups) == 1, f"one ask, one resend: {io.keyups}"
    assert io.keyups[0] >= last_symbol, (
        f"keyed {last_symbol - io.keyups[0]:.3f} s inside the peer's NAK")
    assert hs._tx_retries == 1
    assert 0 <= hs._nak_held <= VA._GRANT_FRESH_S


def _stalled_station():
    hs, io = sending(allow_retries=False)
    hs.turn = VA._TURN_PEER
    hs._tx_pending = None
    hs._answer_owed = VA._OWED_OVER          # the ladder owes the peer an answer
    hs._reack_frame = VF.SESSION_OVER_RESPONSE.name
    return hs, io


def test_a_nak_out_of_turn_re_acks_and_keys_no_data():
    """Out of turn what the peer could not read is our acknowledgement.

    So the ladder's rung goes out and no DATA does — and crucially the DEBT
    SURVIVES. Routing this to `_took_stall_answer` cleared `_answer_owed`, and a
    station owing four ladder rungs was left with none: the very answer the peer
    had just said it could not read would never be keyed again.
    """
    hs, io = _stalled_station()
    assert hs._stalled()
    assert hs._stream_answer(nak_burst())
    assert hs._tx_retries == 0                      # no DATA over out of turn
    assert hs.turn == VA._TURN_PEER
    assert hs._answer_owed == VA._OWED_OVER, "the owe was cleared"
    assert hs._reacks == 1 and len(io.sent) == 1    # the rung reached the air
    assert any("did not read our answer" in m for m in io.msgs)
    # And the rest of the ladder is still there.
    rungs = 1
    while hs._reack():
        rungs += 1
    assert rungs == VA._REACK_MAX


def test_the_ladder_is_the_same_length_with_and_without_the_nak():
    """The NAK spends one rung, it does not spend the ladder."""
    hs, io = _stalled_station()
    baseline = 0
    while hs._reack():
        baseline += 1
    hs, io = _stalled_station()
    assert hs._stream_answer(nak_burst())
    after = 1
    while hs._reack():
        after += 1
    assert after == baseline == VA._REACK_MAX


def test_a_stale_nak_is_reported_rather_than_read_as_silence():
    hs, io = sending()
    kind = VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750")
    stale = np.concatenate((MK.synth_burst("KC9GHZ", kind),
                            np.zeros(int((VA._GRANT_FRESH_S + 0.4) * MK.FS))))
    assert not hs._peer_responder_nak(stale)
    assert any("listening gap has gone" in m for m in io.msgs), io.msgs
    # Said once, not once per scan.
    hs._peer_responder_nak(stale)
    assert sum("listening gap has gone" in m for m in io.msgs) == 1


def test_a_nak_still_arriving_is_not_taken_yet():
    hs, _ = sending()
    kind = VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750")
    whole = MK.synth_burst("KC9GHZ", kind)
    assert not hs._peer_responder_nak(whole[:-3 * MK.HOP])   # last symbols missing
    assert hs._peer_responder_nak(np.concatenate(
        (whole, np.zeros(int(0.2 * MK.FS)))))


def test_a_stale_nak_is_past_the_peers_listening_gap():
    hs, _ = sending()
    kind = VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750")
    stale = np.concatenate((MK.synth_burst("KC9GHZ", kind),
                            np.zeros(int((VA._GRANT_FRESH_S + 0.4) * MK.FS))))
    assert not hs._peer_responder_nak(stale)


@corpora.requires_onair_responder_nak_2750
def test_no_responder_nak_in_a_quiet_stretch_of_the_tape():
    x = corpora.wav_mono(corpora.ONAIR_RESPONDER_NAK_2750)
    hs, _ = sending()
    span = VA._span(VF.for_bw(VF.SESSION_OVER_NAK_RESPONDER, "2750"))
    for at in (3.0, 40.0):
        a = int(at * MK.FS)
        assert not hs._peer_responder_nak(x[a:a + span + int(0.6 * MK.FS)])


def _asked_with_nothing_owed():
    """Where a spent ladder leaves a station: `_reack_spent` clears the debt and
    asks for the turn, so nothing is owed and the turn is `_TURN_ASKED` — the
    other half of `_stalled`, and the half the out-of-turn arm was written
    without."""
    hs, io = sending(allow_retries=False)
    hs._tx_pending = None
    hs.turn = VA._TURN_ASKED
    hs._answer_owed = None
    hs._released = True
    return hs, io


def test_an_out_of_turn_nak_is_not_progress():
    """A frame saying "I could not read you" has read nothing of ours.

    `_progressed` zeroes the give-up budget and moves the pair
    `kestrel_connect.mail_session` restarts its keepalive deadline on: measured,
    one NAK took (progress, idle_keyed, _since_progress) from (0, 0, 5) to
    (1, 1, 1). What a rung costs is `_reack`'s own charge, and that stands.
    """
    hs, io = _stalled_station()
    hs._since_progress = 5
    assert hs._stream_answer(nak_burst())
    assert hs._reacks == 1 and len(io.sent) == 1, "the rung stopped going out"
    assert hs.progress == 0, "the NAK was counted as progress"
    assert hs._since_progress == 6, "the NAK zeroed the give-up budget"
    assert hs.idle_keyed == 1, "the rung was not charged"


def test_a_nak_while_our_turn_request_is_outstanding_acts_on_nothing():
    """`_stalled` is true in `_TURN_ASKED` too, and there the NAK answers no
    acknowledgement of ours — there is none owed. It is an observation."""
    hs, io = _asked_with_nothing_owed()
    hs._since_progress = 5
    assert hs._stalled()
    assert hs._stream_answer(nak_burst())
    assert not io.sent, "keyed at a NAK with nothing owed"
    assert (hs.progress, hs.idle_keyed, hs._since_progress) == (0, 0, 5)
    assert hs._released, "cleared the release the peer's own poll follows"
    assert hs.turn == VA._TURN_ASKED
    assert any("turn-request is outstanding" in m for m in io.msgs), io.msgs
    assert not any("did not read our answer" in m for m in io.msgs), io.msgs


def test_a_nak_per_tick_does_not_hold_a_spent_link_open():
    """The live-lock: a responder keying one NAK per tick interval against a
    station whose ladder is spent. Every NAK zeroed the budget, nothing was
    keyed, and the link stayed CONNECTED — 13 in a row, measured."""
    hs, io = _asked_with_nothing_owed()
    read = 0
    for tick in range(1, 40):
        hs._reset_answer_search()
        read += hs._stream_answer(nak_burst())
        hs.idle_keepalive()
        if hs.state is not VA.VaraState.CONNECTED:
            break
    else:
        raise AssertionError("a NAK per tick held the link open for ever")
    assert read, "the NAKs reached nothing at all"
    assert tick <= VA._MAX_WITHOUT_PROGRESS + 2, (
        f"the link closed at tick {tick}, not on the give-up budget's schedule")
