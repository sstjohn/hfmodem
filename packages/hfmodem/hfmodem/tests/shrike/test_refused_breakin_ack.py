# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A refused CS3 cannot erase the ACK owed to the packet it would answer."""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_control_profiles import _received_reply
from hfmodem.tests.shrike.test_entry_candidate_body import META

FS = onair.FS


def test_recorded_repeat_gets_ack_when_historical_phase_refuses_cs3(
        tmp_path, monkeypatch, capsys):
    session, tx, live, grid, slot, emitted = _received_reply(
        tmp_path, monkeypatch, "historical", "audio-start")
    tx.emit_pending_cs()
    assert len(emitted) == 1
    session.host.arq.on_host_data(b"ABCDEF")
    session.host.arq.on_host_breakin()
    shift = 2 * grid.cycle_n
    origin = META["first_sample"] + shift
    crop = recorded_pcm(META)
    live.audio[origin:origin+len(crop)] = crop
    head = session.rx._p3_delivered_at + shift
    end = head + round(.83 * FS)
    live.pos = live.now = end
    grid.cycles += 2
    session.rx.new_cycle()
    tx.aim(grid, slot + 2)
    onair._scan_frame(session.rx, live.audio[origin:end].copy(), origin)
    assert len(session.packets) == 2
    assert session.packets[-1].packet[2] == b"RMS"
    assert session.host.arq._breakin_armed
    assert tx._pending_p3_cs is None  # The attempted CS3 replaces this ACK.
    timing = grid._p3_reply_timing()
    assert timing is not None
    # The audio-start profile used to manufacture this refusal by recording its
    # pulse center as the emitted reference. It records the aim now, so the comb
    # is put in front of the peer's read instant by the same 17.75 ms here --
    # what the fallback this test is about has to fall back from.
    grid.anchor -= placement.historical_control_pulse_lead(arq.CS_ACK)
    old_anchor, old_seq = grid.anchor, session.host.arq._next_seq
    session.host.tick()
    trace = capsys.readouterr().out
    assert "CHANGEOVER NOT PLACED" in trace and "18 ms" in trace
    assert len(emitted) == 1  # The CS3 never reached the duplex device.
    assert session.host.arq.role == arq.IRS and not grid.sending
    assert grid.anchor == old_anchor
    assert session.host.arq._inflight is None
    assert session.host.arq._next_seq == old_seq
    assert bytes(session.host.arq._outbuf) == b"ABCDEF"
    assert session.host.arq._buffer_raw == 6
    assert tx._pending_p3_cs == arq.CS_ACK
    tx.emit_pending_cs()
    assert len(emitted) == 2 and not tx.refused
    expected = onair._trim_silence(placement.historical_control_signal(arq.CS_ACK))
    expected = (expected * (tx.drive / max(abs(expected)))).astype(np.float32)
    np.testing.assert_allclose(emitted[-1], expected, rtol=0, atol=2e-12)
    assert tx.rig.edges == [True, False, True, False]
    assert len(tx.keyed) == 2
    assert tx.slots_used[-1] == slot + 2
    assert tx.tx_audio_start == grid.boundary(slot + 2)
    assert session.host.arq.rx_seq == 0
    assert session.host.arq._buffer_raw == 6


class RefusingTurn(arq.ArqIO):
    def __init__(self):
        self.events = []
        self.refuse_ack = False
        self.refuse_packet = True
        self.delivered = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.events.append(("packet", bytes(payload), status, breakin))
        return arq.REFUSED if self.refuse_packet else len(payload)

    def send_cs(self, index):
        self.events.append(("cs", index))
        return arq.REFUSED if self.refuse_ack else None

    def deliver(self, data):
        self.delivered.append(data)


def receiving():
    io = RefusingTurn()
    a = arq.PactorArq(io)
    a.role = arq.IRS
    a._enter_connected()
    a.breakin_bytes_override = 3
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    io.events.clear()
    a.on_host_data(b"ABCDEF")
    a.on_host_breakin()
    return a, io


@pytest.mark.parametrize("refuse_ack", [False, True])
def test_refused_turn_restores_bytes_and_attempts_only_owed_ack(refuse_ack):
    a, io = receiving()
    io.refuse_ack = refuse_ack
    a._silent_cycles = 4
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    assert a._breakin_armed and not io.events
    seq, accepted = a._next_seq, a._rx_accepted
    a.on_cycle()
    assert [e[0] for e in io.events] == ["packet", "cs"]
    assert io.events[-1] == ("cs", arq.CS_ACK)
    assert io.events[0][1] == b"ABC" and io.events[0][3]
    assert a.role == arq.IRS and a.state == arq.State.CONNECTED
    assert a._inflight is None and a._next_seq == seq
    assert bytes(a._outbuf) == b"ABCDEF" and a._buffer_raw == 6
    assert io.delivered == [b"RMS"] and a._rx_accepted == accepted
    assert not a._breakin_armed and not a._rx_this_cycle
    assert not a._refused_burst
    # A fallback the seam also refused put nothing on the air, so it cannot
    # forgive the cycle it was owed for.
    assert a._silent_cycles == (4 if refuse_ack else 0)
    a.on_cycle()
    # ...and the quiet cycle behind it is charged to the peer only if its own
    # repeat request reached the air. A seam that refused it left the far end
    # nothing to answer.
    assert a._silent_cycles == (4 if refuse_ack else 1)
    io.refuse_ack = False
    a.on_cycle()
    assert a._silent_cycles == (5 if refuse_ack else 2)


def test_refused_fallback_ack_is_not_accounted_as_emitted():
    a, io = receiving()
    io.refuse_ack = True
    emitted = []
    a.on_cs_emitted = lambda cs: emitted.append(cs)
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    assert io.events[-1] == ("cs", arq.CS_ACK) and not emitted
    io.refuse_ack = False
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    assert io.events[-1] == ("cs", arq.CS_ACK) and emitted == [arq.CS_ACK]


@pytest.mark.parametrize("why", ["quiet_reclaim", "quiet_qrt", "armed_qrt"])
def test_refused_quiet_or_teardown_turn_does_not_invent_an_ack(why):
    a, io = receiving()
    if why == "quiet_reclaim":
        a._peer_receiving = arq.RECLAIM_CODEWORDS
    else:
        a.on_host_disconnect()
        if why == "armed_qrt":
            a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                           protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    assert len(io.events) == 2 and io.events[0][0] == "packet"
    # THE DISPLACED CONTROL STILL KEYS, and it is the repeat request: a
    # teardown settles nothing at the far end on its way out, and a reclaim has
    # no packet at the far end to settle. Keying nothing instead withheld the
    # emitted control phase the next placement is measured from -- see
    # `test_breakin_prompt_placement`.
    assert io.events[1] == ("cs", arq.CS_REQUEST)
    assert a.role == arq.IRS
    assert bytes(a._outbuf) == b"ABCDEF" and a._buffer_raw == 6


def test_a_keyed_changeover_takes_the_link_and_the_peers_ack_settles_it():
    """The other side of the same seam: the changeover that DOES key.

    0913-1550 never reached this line -- ten placements refused for one to
    seven milliseconds -- so the turn the mail client had already built its
    reply for was never asked for. Keyed, the link is ours from the next cycle
    and the peer's acknowledgement settles the packet the CS3 head carried.
    """
    a, io = receiving()
    io.refuse_packet = False
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    assert a._breakin_armed
    a.on_cycle()
    assert [e[0] for e in io.events] == ["packet"]
    assert io.events[0][1] == b"ABC" and io.events[0][3]
    # THE TURN IS TAKEN BY THE PACKET, not by a codeword behind it.
    assert a.role == arq.ISS and a.state == arq.State.CONNECTED
    assert not a._breakin_armed and not a._breakin_pending
    assert a._inflight is not None and a._inflight.payload == b"ABC"
    assert bytes(a._outbuf) == b"DEF"
    # ...and the peer's answer settles it and moves the rest of the reply.
    a.on_rx_cs(arq.CS_ACK)
    a.on_cycle()
    assert a.role == arq.ISS
    assert io.events[-1][0] == "packet" and io.events[-1][1] == b"DEF"
    assert not io.events[-1][3]        # ...and no second changeover behind it


def test_a_changeover_seam_that_keys_nothing_ends_the_link():
    """0913-1556: 87 refusals, no carrier, no retry, no progress.

    The arm took the turn on its first keyed changeover and then lost the peer.
    Every cycle after that printed `CHANGEOVER NOT PLACED -- ... no fresh
    corroborated packet/control phase pair`, which spends no retry by design --
    and with nothing else keying in those cycles the link stood up with no
    carrier on it until the hold's 96 idle cycles ran out in cycle 347, mail
    stage still at "their turn".
    """
    a, io = receiving()
    io.refuse_ack = True
    for cycle in range(a.cfg.max_retries):
        a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                       protocol=spec.Protocol.PACTOR3)
        a.on_cycle()
        assert io.events[-1] == ("cs", arq.CS_ACK)
        assert not a._qrt_pending, cycle
    a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                   protocol=spec.Protocol.PACTOR3)
    a.on_cycle()
    # SIGNED OFF, not vanished: the budget ends the link the way every other
    # one does, and the goodbye then rides its own decision clock.
    assert a._refused_cycles > a.cfg.max_retries and a._qrt_pending
    a.on_cycle()
    assert io.events[-2][0] == "packet" and io.events[-2][3]
    assert io.events[-1] == ("cs", arq.CS_REQUEST)


def test_a_fallback_that_reaches_the_air_keeps_the_link_up():
    """NEGATIVE CONTROL, and it is 0913-1550: ten refusals, ten ACKs, no teardown.

    The rule the budget above must not break. A refused changeover whose
    fallback keys is a cycle the peer was answered in, and that station is not
    off the air -- it is holding a link with a codeword, which is what the
    reverse channel IS.
    """
    a, io = receiving()
    for cycle in range(3 * a.cfg.max_retries):
        a.on_rx_packet(1, b"RMS", 0, True, breakin=True,
                       protocol=spec.Protocol.PACTOR3)
        a.on_cycle()
        assert io.events[-1] == ("cs", arq.CS_ACK)
        assert a.state == arq.State.CONNECTED and a.role == arq.IRS, cycle
