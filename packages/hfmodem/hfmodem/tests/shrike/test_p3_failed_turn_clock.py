# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A failed local CS3 must not rotate the unchanged peer stint a second time."""
from types import SimpleNamespace

import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_grid import _Bench, _Rig


IDENTITY = (True, 1, 0, b"RMS")


def observed_1337():
    """Logged phases and actually emitted CS1 references, no fitted geometry."""
    g = onair._MasterGrid(2635, 60000, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.d_n, g.d_ref_n = 4412, 46080
    g.keyed_slot = 12
    g.keying(spec.Protocol.PACTOR3, entry_pending=True, entry_variant="template")
    g.reverse(to_iss=False)
    g.note_p3_packet(828693, 38880, 60000, swapped=False, identity=IDENTITY)
    g.note_p3_control(991435)
    g.note_p3_control(1051435)
    g.note_p3_packet(1008795, 38880, 60000, swapped=True, identity=IDENTITY)
    assert g.peer_read_gap == 3760
    return g


def observed_b3():
    """`working/onair-0912-2349`: the peer's stint comes back ten periods on.

    Its changeover train was bounded at eight cycles and refused the ninth, so
    the returning packet at 2568924 is 600053 samples -- ten periods, 53 samples
    of peer drift -- past the corroborated 1968871.
    """
    g = onair._MasterGrid(2635, 60000, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.d_n, g.d_ref_n = 4412, 46080
    g.keyed_slot = 12
    g.keying(spec.Protocol.PACTOR3, entry_pending=True, entry_variant="template")
    g.reverse(to_iss=False)
    g.note_p3_packet(1908871, 38880, 60000, swapped=False, identity=IDENTITY)
    g.note_p3_control(1908871 + round(onair.P3_REPLY_S * onair.FS))
    g.note_p3_packet(1968871, 38880, 60000, swapped=False, identity=IDENTITY)
    return g


def attempted(g, emitted_phase=1111435):
    """An emitted CS3 and the ISS role it claims, neither of them answered yet."""
    g.remember_p3_turn(emitted_phase)
    assert g._p3_turn is not None
    g.turn_accepted = False
    g.reverse(to_iss=True)


def test_returning_stint_ten_periods_on_restores_the_irs_clock():
    g = observed_b3()
    anchor, timing = g.anchor, g._p3_timing
    attempted(g, 1968871 + round(onair.P3_REPLY_S * onair.FS))
    g.note_p3_packet(2568924, 38880, 60000, swapped=False, identity=IDENTITY)
    assert g.recover_p3_turn(2568924, IDENTITY)
    assert not g.sending and g.anchor == anchor
    assert g._p3_timing == timing and not g._p3_reply_phase_invalid
    # The reply comb is restored; the changeover placement is not, because
    # `_p3_reply_timing` ages the measurement itself at ONSET_MAX_CYCLES and the
    # returning packet is ten periods past it. Its own evidence, its own bound.
    assert g._p3_reply_timing() is None
    assert g.p3_control_refusal(g.rx_slot(2568924) + 1) is None


def event(phase=1188708, *, payload=b"RMS", status=0):
    return rxfront.Event(phase / onair.FS, "packet", "recorded RMS coordinates",
                         protocol=spec.Protocol.PACTOR3,
                         packet=(1, status, payload, True), breakin=True,
                         carrier_swapped=False)


def test_recorded_repeat_restores_prior_irs_phase_and_timing(tmp_path):
    g = observed_1337()
    prior_timing = g._p3_reply_timing()
    attempted(g)
    # This is the logged unchanged raster, three cycles less87samples.
    phase = 1188708
    assert phase - 1008795 - 3 * 60000 == -87
    g.note_p3_packet(phase, 38880, 60000, swapped=False, identity=IDENTITY)
    tx = onair.RadioTx(None, transmit=False, settle=.040, outdir=tmp_path)
    tx.aim(g, 21)
    tx.host = SimpleNamespace(arq=SimpleNamespace(unconfirmed_breakin=True))
    assert tx.recover_p3_turn(event())
    assert not g.sending and g.anchor == 31435 and g.rx_ref_n == 17280
    assert g._p3_reply_timing() == prior_timing
    assert g._p3_peer[0] == phase and g._p3_peer_swap is False
    assert g.key_refusal(g.boundary(21) - 1920 - 677, 1920 + 11677) is None
    # A second application cannot undo another real turn.
    assert not tx.recover_p3_turn(event())


@pytest.mark.parametrize("change", ["different_phase", "different_payload",
                                   "different_status", "too_old", "before_tx",
                                   "different_width", "different_cycle", "duplicate"])
def test_new_or_unproven_stint_cannot_restore_old_clock(change):
    g = observed_1337()
    attempted(g)
    phase, width, period, identity = 1188708, 38880, 60000, IDENTITY
    if change == "different_phase":
        phase += 28800
    elif change == "different_payload":
        identity = (True, 1, 0, b"NEW")
    elif change == "different_status":
        identity = (True, 1, 1, b"RMS")
    elif change == "too_old":
        # Seventeen actual peer periods. Nine used to be refused here, which is
        # the stint B3 came back on: the changeover train is bounded by
        # ONSET_MAX_CYCLES, so a returning stint always arrives past it and
        # recovery could never fire. TURN_RECOVERY_CYCLES is what ages now.
        phase += 14 * period
    elif change == "before_tx":
        phase = 1068795
    elif change == "different_width":
        width += 480
    elif change == "different_cycle":
        period *= 3
    elif change == "duplicate":
        phase = 1008795
    g.note_p3_packet(phase, width, period, identity=identity)
    assert not g.recover_p3_turn(phase, identity)
    assert g.sending and g.anchor == 31435
    g.reverse(to_iss=False)
    # No turnaround happened at either end, so neither comb moved. Rotating the
    # way back out puts the comb at 60235, where every ACK lands inside the
    # packet it answers.
    assert g.anchor == 31435 and g._p3_reply_timing() is None


def test_settled_breakin_cannot_reuse_snapshot(tmp_path):
    g = observed_1337()
    attempted(g)
    g.note_p3_packet(1188708, 38880, 60000, identity=IDENTITY)
    tx = onair.RadioTx(None, transmit=False, settle=.04, outdir=tmp_path)
    tx.aim(g, 21)
    tx.host = SimpleNamespace(arq=SimpleNamespace(unconfirmed_breakin=False))
    assert not tx.recover_p3_turn(event())
    assert g._p3_turn is None and g.sending


@pytest.mark.parametrize("reset", ["protocol", "cycle", "bare_control"])
def test_snapshot_does_not_outlive_geometry_change(reset):
    g = observed_1337()
    attempted(g)
    if reset == "protocol":
        g.keying(spec.Protocol.PACTOR1)
    elif reset == "cycle":
        g.regear(True)
        g.regear(False)
    else:
        g.note_peer_codeword(1188708, onair.P3_CS_N, "ACK", "WS8EOC",
                             protocol=spec.Protocol.PACTOR3)
    assert g._p3_turn is None


@pytest.mark.parametrize("emitted", [True, False])
def test_snapshot_is_taken_only_after_successful_cs3_emission(emitted, tmp_path, monkeypatch):
    g = observed_1337()
    tx = onair.RadioTx(None, transmit=False, settle=.04, outdir=tmp_path)
    tx.aim(g, 18)

    def emit(audio, what, **kw):
        tx.refused = not emitted
        if emitted:
            tx.tx_audio_start = g.boundary(18) - kw["lead_n"]
            tx.tx_end = tx.tx_audio_start + 40000

    monkeypatch.setattr(tx, "_tx", emit)
    tx.send_packet(3, b"ABC", 0, breakin=True)
    assert (g._p3_turn is not None) == emitted
    if emitted:
        first = g._p3_turn
        assert first.emitted_phase == g.boundary(18)
        g.reverse(to_iss=True)
        tx.aim(g, 19)
        tx.send_packet(3, b"ABC", 0, breakin=True)
        assert g._p3_turn is first


def test_quiet_reply_requires_fresh_clock_even_after_collision_evidence_expires(tmp_path, monkeypatch):
    g = observed_1337()
    g.cycles += 2
    assert g._peer_air() is None
    assert g.p3_control_refusal(20) is None  # A short fade keeps the valid clock.
    assert "no fresh packet clock" in g.p3_control_refusal(25)
    tx = onair.RadioTx(None, transmit=False, settle=.04, outdir=tmp_path)
    tx.aim(g, 25)
    monkeypatch.setattr(tx, "_tx", lambda *args, **kw: pytest.fail("stale reply keyed"))
    assert tx._send_p3_control(arq.CS_ACK) == arq.REFUSED


def test_known_bad_phase_never_becomes_valid_by_aging(tmp_path):
    g = observed_1337()
    attempted(g)
    g.note_p3_packet(1188708, 38880, 60000, identity=IDENTITY)
    g.reverse(to_iss=False)
    # The return from an unanswered changeover does not rotate, so the
    # overlapping comb this test needs is made here rather than fallen into.
    assert g.anchor == 31435
    g.anchor += g.data_n - g.cs_n
    tx = onair.RadioTx(None, transmit=False, settle=.04, outdir=tmp_path)
    tx.aim(g, 21)
    assert tx._refused(g.boundary(21) - 1920 - 677, 1920 + 11677, "CS1 ACK")
    assert g._p3_reply_phase_invalid
    g.cycles += 2
    assert g._peer_air() is None
    assert "remains invalid" in g.p3_control_refusal(22)
    g.keying(spec.Protocol.PACTOR1)
    assert g.p3_control_refusal(22) is None


def test_a_fresh_crc_packet_clears_the_invalid_reply_phase(tmp_path):
    """The positive control: the latch may not outlive the reading it describes."""
    g = observed_1337()
    attempted(g)
    g.note_p3_packet(1188708, 38880, 60000, identity=IDENTITY)
    g.reverse(to_iss=False)
    g.anchor += g.data_n - g.cs_n
    tx = onair.RadioTx(None, transmit=False, settle=.04, outdir=tmp_path)
    tx.aim(g, 21)
    assert tx._refused(g.boundary(21) - 1920 - 677, 1920 + 11677, "CS1 ACK")
    assert g._p3_reply_phase_invalid
    assert "remains invalid" in g.p3_control_refusal(22)
    g.note_p3_packet(1248708, 38880, 60000, identity=IDENTITY)
    assert not g._p3_reply_phase_invalid
    # ...and the guard still re-decides against the new frame at the key.
    assert g.p3_control_refusal(22) is None


@pytest.mark.parametrize("same_stint", [True, False])
def test_session_host_and_radio_preserve_only_the_unchanged_peer_stint(same_stint, tmp_path):
    g = observed_1337()
    s = _Session(role=arq.IRS)
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path, settle=.04)
    tx.host = s.host
    s.host.peer = tx
    tx.live = bench = _Bench()
    # The restored boundary is checked against the leading pulse below, which
    # is the pulse-center experiment's reference rather than the default's.
    tx.p3_control_placement = "pulse-center"
    tx.defer_p3_cs = True
    tx.breakin_due = True
    tx.aim(g, 18)
    bench.now = bench.pos = tx.key_instant(g, 18) - round(.04 * onair.FS)
    a = s.host.arq
    a.on_host_data(b"ABCDEF")
    a.on_host_breakin()
    s.rx._on(event(1008795))  # Production dedup, clock observer and PTC dispatch.
    prior_rx = (a.rx_seq, a._expected_seq, a._last_rx_field)
    s.host.tick()
    assert a.unconfirmed_breakin and a._inflight.payload == b"ABC"
    assert len(bench.emissions) == 1 and g._p3_turn is not None
    onair._grid_reversal(g, s.host)
    assert g.sending

    phase = 1188708 + (0 if same_stint else 28800)
    s.rx.new_cycle()
    s.rx._on(event(phase))
    onair._grid_reversal(g, s.host)
    assert a.role == arq.IRS and not g.sending
    if not same_stint:
        # A new stint keeps the comb the first reversal rotated; only the ARQ
        # state differs from the recovered case. A second rotation here is what
        # put 0912-2349's replies inside the peer.
        assert g.anchor == 31435
        assert a._buffer_raw == 3 and a._outbuf == b"DEF"
        assert g._p3_reply_timing() is None
        return

    assert a._buffer_raw == 6 and a._outbuf == b"ABCDEF"
    assert g.anchor == 31435
    assert (a.rx_seq, a._expected_seq, a._last_rx_field) == prior_rx
    assert a._turn_ack_owed and tx._pending_p3_cs == arq.CS_ACK
    s.host.app = SimpleNamespace(done=False)
    s.host.app_turns()  # Pending mail cannot replace this answer with another CS3.
    tx.aim(g, 21)
    bench.now = bench.pos = tx.key_instant(g, 21) - round(.04 * onair.FS)
    s.host.tick()
    assert len(bench.emissions) == 1 and a._turn_ack_owed
    tx.emit_pending_cs()
    assert len(bench.emissions) == 2 and not a._turn_ack_owed
    assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == g.boundary(21)
    assert a.role == arq.IRS and a._buffer_raw == 6


def test_bare_head_keeps_snapshot_and_inflight_until_body_classifies(tmp_path):
    g = observed_1337()
    s = _Session(role=arq.ISS)
    tx = onair.RadioTx(None, transmit=False, settle=.04, outdir=tmp_path)
    tx.host = s.host
    s.host.peer = tx
    tx.aim(g, 18)
    a = s.host.arq
    a.on_host_data(b"ABCDEF")
    # Prepare an actual ARQ inflight packet with a recording-only seam, then
    # attach the driver's emitted-turn evidence separately; no radio opens.
    a._start_next_packet(breakin=True)
    g.remember_p3_turn(1111435)
    saved = g._p3_turn
    g.reverse(to_iss=True)
    assert a.unconfirmed_breakin and saved is not None
    s.rx._on(rxfront.Event(1188708 / onair.FS, "cs", "body not yet decoded",
                          protocol=spec.Protocol.PACTOR3, cs=arq.CS_BREAKIN))
    assert a.unconfirmed_breakin and a._buffer_raw == 6
    assert g._p3_turn is saved and g.sending
    before = tx.n
    s.host.tick()
    assert tx.n == before  # One receive opportunity for the body, no retry TX.
    tx.defer_p3_cs = True
    s.rx._on(event())
    onair._grid_reversal(g, s.host)
    assert a.role == arq.IRS and a._buffer_raw == 6
    assert g.anchor == 31435 and not g.sending and a._turn_ack_owed


def test_contact_exit_clears_failed_turn_snapshot_and_invalid_phase():
    g = observed_1337()
    attempted(g)
    g._p3_reply_phase_invalid = True
    host = SimpleNamespace(protocol=spec.Protocol.PACTOR1,
                           arq=SimpleNamespace(state=arq.State.LISTENING, role=None))
    onair._grid_reversal(g, host)
    assert g._p3_turn is None and g._p3_reply_timing() is None
    assert not g._p3_peer_confirmed and not g._p3_reply_phase_invalid
