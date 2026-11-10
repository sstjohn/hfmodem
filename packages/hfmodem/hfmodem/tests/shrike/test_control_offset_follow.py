# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""PACTOR-3 is keyed on the peer's carrier raster, not on our own.

2026-09-13 measured one gateway's PACTOR-3 at -5 Hz in the 12:57 arm and +64 Hz
in the 13:16 arm, on three instruments, while our own transmitter sat at -5 Hz
in both with zero spread over 65 bursts and the peer's own FSK stayed within
3 Hz of nominal. `p3acquire` accepts a control within about 8 Hz of true, so the
second arm's CS1 was never read and the gateway repeated its changeover 31 times
and signed off. VE3KPG the same day read our entry at +50 Hz -- two carriers its
own sweep can find -- then answered 38 of our fourteen-carrier SL3 packets with
CS1 and accepted none.

What is proved here is the SIGN, on our own receiver: a burst rendered for a
peer at +NN Hz is one that a receiver correcting by +NN Hz reads, and one a
receiver at nominal does not.
"""
import json

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3acquire, p3rx, placement, spec
from hfmodem.tests.shrike import test_entry_answer as entry
from hfmodem.tests.shrike import test_p3_reply_placement as reply

FS = onair.FS
PEER = reply.A5_PEER
PAYLOAD = (b"fourteen carriers, 120 Hz apart, 50 Hz off its raster........"
           [:placement.SPEED_PATHS[3].crc_bytes - 3])


class _Sessrx:
    """Only what the transmitter asks a session receiver for."""

    def __init__(self, hz: float):
        self.p3_receive_offset_hz = hz

    def skip(self, seconds: float) -> None:
        pass

    def bridge(self, audio: np.ndarray) -> None:
        pass


def bench(tmp_path, monkeypatch, hz, follow="all", peer=True):
    """An IRS holding one CRC packet from a peer measured at `hz`.

    `peer=False` reverses the same comb to ISS and notes no packet: the
    acknowledgement guard has a whole cycle to place an 877 ms packet in, which
    is what an ISS holds and what an answer slot does not.
    """
    g = reply.irs_grid()
    tx = reply.driver(g, tmp_path)
    tx.sessrx = _Sessrx(hz)
    tx.p3_follow_offset = follow
    if peer:
        g.note_p3_packet(PEER, reply.PACKET_N, reply.CYCLE, swapped=False,
                         identity=reply.IDENTITY)
        slot = (PEER + reply.REPLY_N - g.anchor) // reply.CYCLE
    else:
        g.reverse(to_iss=True)
        slot = PEER // reply.CYCLE
    emitted = []
    transmit = tx.live.transmit
    monkeypatch.setattr(tx.live, "transmit",
                        lambda audio, **kw: (emitted.append(audio.copy()),
                                             transmit(audio, **kw))[1])
    return g, tx, slot, emitted


def key(tx, g, slot, send):
    """Aim, put the clock one settle in front of the key, and send."""
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - round(.04 * FS)
    return send()


def keyed(emitted):
    """One emission, back at unit scale, as a float64 the readers accept."""
    assert len(emitted) == 1
    x = np.asarray(emitted[0], np.float64)
    return np.pad(x / max(abs(x)) * .5, (4800, 4800))


def read_control(audio, **kw):
    got = p3acquire.control_signal(audio, **kw)
    return None if got is None else (got.event.cs, got.offset_hz)


# -- the sign -------------------------------------------------------------

@pytest.mark.parametrize("hz", [60.0, -60.0, 25.0, -75.0, 75.0])
def test_a_control_keyed_for_a_displaced_peer_reads_at_that_displacement(
        tmp_path, monkeypatch, hz):
    """The whole claim, through the production emit: key for +NN, read at +NN.

    `p3acquire.compensate(x, hz)` shifts a spectrum DOWN by `hz` and receive
    calls it as `compensate(seg, +offset)` to bring a high-arriving peer back to
    nominal. A control rendered for a peer at +60 must therefore be found by the
    same reader at +60 -- and NOT at zero, which is where every arm before this
    one keyed it.
    """
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, hz)
    assert reply.key_ack(tx, g, slot) is None and not tx.refused
    audio = keyed(emitted)
    assert read_control(audio) == (arq.CS_ACK, hz)
    assert read_control(audio, offsets=(0.0,)) is None


def test_an_sl3_packet_for_a_peer_at_plus_50_decodes_at_plus_50_and_not_at_zero(
        tmp_path, monkeypatch):
    """VE3KPG's arm: fourteen carriers at 120 Hz spacing, 50 Hz off its raster."""
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 50.0, peer=False)
    assert key(tx, g, slot, lambda: tx.send_packet(3, PAYLOAD, 0x5b)) \
        != arq.REFUSED
    audio = keyed(emitted)

    def fields(correction):
        return [(p.sl, p.payload) for p in p3rx.decode_p3_packets(
            p3acquire.compensate(audio, correction)).packets]

    assert fields(50.0) == [(3, PAYLOAD)]
    assert fields(0.0) == []
    # ...and the nominal render is what a receiver at zero reads, so the
    # decoder above is not simply deaf to this packet at every correction.
    nominal = np.pad(placement.link_packet(3, PAYLOAD, 0x5b), (4800, 4800))
    assert [(p.sl, p.payload)
            for p in p3rx.decode_p3_packets(nominal).packets] == [(3, PAYLOAD)]


# -- the negative controls ------------------------------------------------

def rendered_nominal(tx, drive_from):
    """Today's samples: the untouched render, trimmed and driven as `_tx` does."""
    x = onair._trim_silence(placement.historical_control_signal(arq.CS_ACK))
    return (x * (tx.drive / max(abs(x)))).astype(drive_from.dtype)


def test_a_session_that_has_measured_nothing_keys_todays_samples(
        tmp_path, monkeypatch):
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 0.0)
    assert reply.key_ack(tx, g, slot) is None
    np.testing.assert_array_equal(emitted[0], rendered_nominal(tx, emitted[0]))
    assert tx.tx_pulse_offsets == (852, 852)
    assert read_control(keyed(emitted)) == (arq.CS_ACK, 0.0)


def test_follow_none_keys_a_displaced_peer_at_nominal(tmp_path, monkeypatch):
    """`--p3-follow-offset none`, sample for sample the arm that was not read."""
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 60.0, follow="none")
    assert reply.key_ack(tx, g, slot) is None
    np.testing.assert_array_equal(emitted[0], rendered_nominal(tx, emitted[0]))
    assert tx.tx_pulse_offsets == (852, 852)
    assert read_control(keyed(emitted)) == (arq.CS_ACK, 0.0)


def test_follow_control_moves_the_codeword_and_leaves_the_data_packet(
        tmp_path, monkeypatch):
    """The narrower experiment stays reachable: codewords only."""
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 50.0, follow="control")
    assert reply.key_ack(tx, g, slot) is None
    assert read_control(keyed(emitted)) == (arq.CS_ACK, 50.0)

    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 50.0, follow="control",
                                 peer=False)
    assert key(tx, g, slot, lambda: tx.send_packet(3, PAYLOAD, 0x5b)) \
        != arq.REFUSED
    assert [(p.sl, p.payload) for p in p3rx.decode_p3_packets(
        keyed(emitted)).packets] == [(3, PAYLOAD)]


def test_a_stored_offset_cannot_key_us_past_the_readers_own_sweep(
        tmp_path, monkeypatch):
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 400.0)
    assert reply.key_ack(tx, g, slot) is None
    bound = max(p3acquire.CONTROL_OFFSETS_HZ)
    assert read_control(keyed(emitted)) == (arq.CS_ACK, bound)


# -- what the burst carries with it ---------------------------------------

def test_the_applied_offset_is_on_the_tx_line_and_in_the_sidecar(
        tmp_path, monkeypatch, capsys):
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, -15.0)
    assert reply.key_ack(tx, g, slot) is None
    assert "TX[1] CS1 ACK  TX -15 Hz  (" in capsys.readouterr().out
    sidecar = json.loads((tmp_path / "tx_01.json").read_text())
    assert sidecar["tx_offset_hz"] == -15.0
    # BESIDE THE LABEL, NOT IN IT. Scenes elsewhere read `control` to name the
    # codeword a sidecar belongs to; the frequency is its own field.
    assert sidecar["control"] == "CS1 ACK"
    assert sidecar["pulse_offsets"] == list(tx.tx_pulse_offsets)


def test_a_pactor1_burst_carries_no_offset_at_all(tmp_path, monkeypatch, capsys):
    """Only PACTOR-3 moves. The peer's FSK was on frequency in both arms, and a
    dial that satisfied its P3 would break the P1 link that works."""
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 60.0, peer=False)
    key(tx, g, slot, lambda: tx.send_p1_cs(arq.CS_ACK))
    said = capsys.readouterr().out
    assert len(emitted) == 1
    assert "TX[1] P1 CS1 LSB  (" in said and " Hz" not in said


def test_the_reply_instant_and_the_comb_are_where_they_were(
        tmp_path, monkeypatch):
    """Placement is not what moved: the audio start is still the answer slot."""
    due = PEER + reply.REPLY_N
    for hz in (0.0, 60.0, -60.0):
        g, tx, slot, emitted = bench(tmp_path, monkeypatch, hz)
        assert reply.key_ack(tx, g, slot) is None
        assert g.boundary(slot) == due
        assert tx.tx_audio_start == due
        assert g._p3_controls[-1][0] == due


# -- the pulse lead is measured on what is keyed --------------------------

def test_the_transforms_padding_leaves_the_burst_its_own_length(
        tmp_path, monkeypatch):
    """The fast-length pad is trimmed straight back off, on every render.

    It differs from the bare transform only in the wrap the padding removes,
    and that lives in the skirt the driver's 2% trim drops: the two agree to
    4e-5 of a 0.5 peak across everything that is keyed.
    """
    _, tx, _, _ = bench(tmp_path, monkeypatch, 60.0)
    for raw in (placement.historical_control_signal(arq.CS_ACK),
                placement.link_packet(3, PAYLOAD, 0x5b),
                placement.changeover_packet(PAYLOAD[:3], 0)):
        moved = tx._p3_offset(raw, control=True)
        assert len(moved) == len(raw)
        bare = p3acquire.compensate(np.asarray(raw, float), -60.0)
        on = np.flatnonzero(abs(moved) > .02 * abs(moved).max())
        np.testing.assert_allclose(moved[on[0]:on[-1] + 1],
                                   bare[on[0]:on[-1] + 1], rtol=0, atol=4e-5)


def test_the_lead_helper_reproduces_every_cached_codeword_lead():
    for cs in range(6):
        assert placement.pulse_lead(placement.historical_control_signal(cs)) \
            == placement.historical_control_pulse_lead(cs)
        for swapped in (False, True):
            assert placement.pulse_lead(
                placement.control_signal(cs, swapped=swapped)) \
                == placement.control_pulse_lead(cs, swapped=swapped)


def test_a_moved_control_reports_its_own_pulse_centers(tmp_path, monkeypatch):
    """CS6 at -60 Hz crosses the driver's 2% trim 47 samples later than CS1.

    The trim is an instantaneous test and the carriers are what moved, so the
    codeword's cached lead is no longer this burst's. The sidecar has to
    describe the samples that went out.
    """
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, -60.0)
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = tx.key_instant(g, slot) - round(.04 * FS)
    assert tx._send_p3_control(5) is None
    audio = np.asarray(emitted[0], np.float64)
    lead = placement.pulse_lead(audio)
    assert lead != placement.historical_control_pulse_lead(5)
    assert tx.tx_pulse_offsets == (lead, lead)


# -- and what may MOVE it: one frame is a candidate, not a correction -------

SESSION_HZ = -25.0
"""Where WS8EOC sat for the whole of `captures/onair-0913-2152`: -24.5 Hz, and
this is the reader's own grid point for it."""

ALIAS_HZ = 75.0
"""...and where one acquisition of that arm landed. `[p3] frame @ 3288704 ...
primary RX +75.2 Hz` moved the session, and `tx_43` went out at +75.0 -- a
hundred hertz from the station it was answering, with nineteen keyings on the
peer's own raster in front of it."""


def acquiring(*offsets: float):
    """A real `_SessionRx` run over cycles of a peer's bare codeword.

    `_p3_cs` acquires during entry: it corroborates a body-less word on a second
    distinct cycle at the same frequency and phase, so each acquisition here
    costs two cycles of the peer's, and every one that lands reaches
    `_follow_p3_offset` the way the arm's did.
    """
    n = round(spec.CYCLE_SHORT_S * FS)
    anchor = round(.60 * FS)
    at = anchor - round(entry.ANSWER_LEAD_S * FS)
    sess = entry._Session(role=arq.ISS, entry_pending=True)
    for k, hz in enumerate(offsets):
        sess.rx.new_cycle()
        sess.rx._p3_cs(entry._answered_cycle(entry._answer(arq.CS_ACK, hz),
                                             at, n), anchor, seg_start=k * n)
    return sess


def keys_at(tmp_path, monkeypatch, sess):
    """What that session's next acknowledgement actually puts on the air."""
    g, tx, slot, emitted = bench(tmp_path, monkeypatch, 0.0)
    tx.sessrx = sess.rx
    assert reply.key_ack(tx, g, slot) is None and not tx.refused
    return read_control(keyed(emitted))


def test_a_session_with_no_estimate_takes_its_first_acquisition_whole():
    """There is nothing to corroborate a first reading against, and a peer on
    its own raster is the case the whole follow exists for."""
    sess = acquiring(ALIAS_HZ, ALIAS_HZ)
    assert sess.rx.p3_receive_offset_hz == ALIAS_HZ
    assert sess.rx._p3_offset_candidate is None


def test_a_neighbouring_grid_point_is_the_same_peer_and_is_taken():
    """Inside one coarse step the two hypotheses are neighbours: a peer that
    has drifted reads at one of them and the follow has to keep up."""
    sess = acquiring(SESSION_HZ, SESSION_HZ, -50.0, -50.0)
    assert sess.rx.p3_receive_offset_hz == -50.0
    assert sess.rx._p3_offset_candidate is None


def test_one_frame_a_hundred_hertz_off_leaves_the_transmitter_where_it_was(
        tmp_path, monkeypatch):
    """`onair-0913-2152`, and the keying it cost."""
    sess = acquiring(SESSION_HZ, SESSION_HZ, ALIAS_HZ, ALIAS_HZ)
    assert sess.rx.p3_receive_offset_hz == SESSION_HZ
    assert sess.rx._p3_offset_candidate == ALIAS_HZ
    assert keys_at(tmp_path, monkeypatch, sess) == (arq.CS_ACK, SESSION_HZ)


def test_a_second_acquisition_that_agrees_is_what_moves_it(
        tmp_path, monkeypatch):
    """The guard is corroboration, not a ceiling: a peer that really has moved
    a hundred hertz is followed on the second frame that says so."""
    sess = acquiring(SESSION_HZ, SESSION_HZ, ALIAS_HZ, ALIAS_HZ,
                     ALIAS_HZ, ALIAS_HZ)
    assert sess.rx.p3_receive_offset_hz == ALIAS_HZ
    assert sess.rx._p3_offset_candidate is None
    assert keys_at(tmp_path, monkeypatch, sess) == (arq.CS_ACK, ALIAS_HZ)


def test_the_held_read_is_still_taken(capsys):
    """A word that decoded is a reading of the air, and the peer's answer clock
    is measured from it. What the guard withholds is the transmitter alone."""
    held = acquiring(SESSION_HZ, SESSION_HZ, ALIAS_HZ, ALIAS_HZ)
    moved = acquiring(SESSION_HZ, SESSION_HZ, ALIAS_HZ, ALIAS_HZ,
                      ALIAS_HZ, ALIAS_HZ)
    assert held.rx._p3_answer_at == moved.rx._p3_answer_at \
        - 2 * round(spec.CYCLE_SHORT_S * FS)
    assert "held as a candidate" in capsys.readouterr().out


def test_a_link_that_drops_forgets_the_estimate_and_the_candidate():
    sess = acquiring(SESSION_HZ, SESSION_HZ, ALIAS_HZ, ALIAS_HZ)
    sess.host.arq._to(arq.State.DISCONNECTED)
    sess.rx.new_cycle()
    assert sess.rx.p3_receive_offset_hz == 0.0
    assert sess.rx._p3_offset_candidate is None
    assert not sess.rx._p3_offset_fixed
