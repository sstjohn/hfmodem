# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""P3 break-in uses observed P3 phase timing, without borrowing P1 turnaround."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.tests.shrike.test_cs6_stalls import ElapsedPCM
from hfmodem.tests.shrike.test_entry_answer import _Session

FS = 48000


def transitioned():
    # 11:19 WS8EOC log: original anchor2634, P1 d4413. These are independent
    # recorded coordinates, never obtained by inverting peer_read_gap.
    g = onair._MasterGrid(2634, 60000, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.d_n, g.d_ref_n = 4413, 46080
    g.acquired = g.corroborated = True
    g.keyed_slot = 36
    g.keying(spec.Protocol.PACTOR3, entry_pending=True, entry_variant="template")
    g.reverse(to_iss=False)
    g.note_p3_packet(2388632, 38880, 60000)
    return g


def tx_at(g, slot, tmp_path):
    tx = onair.RadioTx(None, transmit=False, settle=.040, outdir=tmp_path)
    tx.aim(g, slot)
    return tx


def observed_reply(g):
    # Current driver's leading control phase is on its boundary. This is an
    # explicit counterfactual to the baseline's uncorrected filter-skirt onset;
    # the peer CRC coordinates themselves come from the 11:19 log.
    g.note_p3_control(2551434)
    g.note_p3_control(2611434)  # Newer TX precedes the delayed decode, not its RF.
    g.note_p3_packet(2568704, 38880, 60000)


def test_recorded_upgrade_keeps_p1_read_anchor_but_learns_p3_reply(tmp_path):
    g = transitioned()
    g.note_p3_packet(2568704, 38880, 60000)
    tx = tx_at(g, 44, tmp_path)
    end = g.peer_packet_end(44)
    assert end.at_slot == 2667584
    assert g.peer_read_gap == 6627
    assert end.at_slot + g.peer_read_gap - tx.boundary == 2777
    assert "58 ms" in tx._clamp_refusal(end)
    assert "no fresh corroborated" in g.breakin_refusal(end)

    g = transitioned()
    observed_reply(g)
    assert g.d == 4413 and g.rx_ref_n == 17280  # P1 grant tracking is retained.
    assert g._p3_reply_timing()[0:2] == (2551434, 2568704)
    assert g._p3_reply_timing()[4] == 7190  # RX phase minus actual CS phase/end.
    assert g.peer_read_gap == 3850
    tx = tx_at(g, 44, tmp_path)
    assert tx._clamp_refusal(g.peer_packet_end(44)) is None
    assert tx._place_breakin() and tx.placed
    assert tx.boundary == 2671434


def test_independent_scs_reverse_changeover_matches_prior_control_phase(tmp_path):
    # PIII_Complete_1, pactor3.md §17.1: actual CS3 at7.7681s follows CS5
    # at6.5194s and a CRC packet at6.8200s. Prior CRC CS3 at5.5637s establishes
    # the peer's phase. These are observed phase coordinates, not envelope feet.
    n = lambda seconds: round(seconds * FS)
    g = onair._MasterGrid(n(6.5194), 60000, 0, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.d_n = 0.105 * FS  # Unrelated P1 value must not choose the P3 reply.
    g.note_p3_packet(n(5.5637), 38880, 60000)
    g.note_p3_control(n(6.5194))
    g.note_p3_packet(n(6.8200), 39120, 60000)
    tx = tx_at(g, 1, tmp_path)
    assert g._p3_reply_timing() is not None
    assert tx._place_breakin()
    assert abs(tx.boundary - n(7.7681)) < round(.002 * FS)
    # This independently accepted reverse-direction head is not at the
    # forward direction's packet+890ms slot.
    assert tx.boundary - n(6.8200) > round(.940 * FS)


@pytest.mark.parametrize("problem", ["no_tx", "only_one_frame", "wrong_phase",
                                     "wrong_cycle", "impossible_gap"])
def test_unproven_p3_position_remains_refused(problem, tmp_path):
    g = transitioned()
    if problem == "only_one_frame":
        g._p3_peer = None
    if problem != "no_tx":
        g.note_p3_control(2551434 if problem != "impossible_gap" else 2564000)
    phase = 2568704 + (1200 if problem == "wrong_phase" else 0)
    g.note_p3_packet(phase, 38880, 180000 if problem == "wrong_cycle" else 60000)
    assert g._p3_reply_timing() is None
    tx = tx_at(g, 44, tmp_path)
    assert tx._place_breakin() == ""
    assert "no fresh corroborated" in tx.unplaceable


@pytest.mark.parametrize("transition", ["role", "protocol", "cycle", "age"])
def test_p3_timing_cannot_outlive_its_evidence(transition):
    g = transitioned()
    observed_reply(g)
    assert g._p3_reply_timing() is not None
    if transition == "role":
        g.reverse(to_iss=True)
        g.reverse(to_iss=False)
    elif transition == "protocol":
        g.keying(spec.Protocol.PACTOR1)
    elif transition == "cycle":
        g.regear(True)
        g.regear(False)
    else:
        g.cycles += onair.ONSET_MAX_CYCLES + 1
    assert g._p3_reply_timing() is None


def test_a_changeover_retry_keeps_its_original_p3_placement(tmp_path):
    g = transitioned()
    observed_reply(g)
    tx = tx_at(g, 44, tmp_path)
    assert tx._place_breakin()
    first = tx.boundary
    g.reverse(to_iss=True)
    tx.aim(g, 45)
    assert tx._place_breakin()
    assert tx.boundary == first + 60000
    tx.host = SimpleNamespace(protocol=spec.Protocol.PACTOR3,
                             arq=SimpleNamespace(role=arq.ISS, entry_pending=False))
    tx.breakin_due = True
    reserve = onair._p3_breakin_lead_bound(placement.PROTOCOL_RISE,
                                          placement.CASE0_STAGGER)
    assert tx.key_instant(g, 45) == tx.boundary - reserve
    tx.breakin_due = False
    assert tx.key_instant(g, 45) == tx.boundary  # Ordinary ISS data has no CS3.


@pytest.mark.parametrize("rise,stagger", [(True, True), (True, False),
                                         (False, True), (False, False)])
def test_head_bound_covers_payload_and_status_without_full_filter_reserve(
        rise, stagger, monkeypatch):
    monkeypatch.setattr(placement, "PROTOCOL_RISE", rise)
    monkeypatch.setattr(placement, "CASE0_STAGGER", stagger)
    bound = onair._p3_breakin_lead_bound(rise, stagger)
    center = int(np.argmax(placement.protocol_config().pulse()))
    assert 0 < bound < center
    for payload, status in [(b"", 0), (b"abc", 0xff), (b"\xff\x00\xaa", 1)]:
        for swapped in (False, True):
            audio = placement.changeover_packet(payload, status, swapped=swapped)
            trim = int(np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0])
            assert center - trim <= bound


def test_reserved_cs3_lead_still_reads_the_current_recorded_frame():
    # The morning hold14 frame coordinates also used by test_p3_reply_timing.
    # Reserving a CS3 skirt must not silently select the preceding frame.
    rx = SimpleNamespace(_p3_row0=round(43.6025 * FS),
                         _p3_span=35760, _p3_cycle_n=60000)
    phase = 2131430
    lead = onair._p3_breakin_lead_bound(True, True)
    deadline = onair._p3_decode_deadline(SimpleNamespace(key_notice=1536),
                                        phase - lead, 1920)
    ready = onair._p3_frame_ready(rx, deadline)
    assert ready == rx._p3_row0 + rx._p3_span + 240  # Half-symbol alignment slack.
    assert ready + 1536 + round(.006 * FS) < phase - lead


def test_observed_tx_phase_cannot_be_replaced_with_requested_boundary(tmp_path):
    g = transitioned()
    # An emission actually 18ms late is not evidence for an on-time answer.
    g.note_p3_control(2551434 + 864)
    g.note_p3_packet(2568704, 38880, 60000)
    tx = tx_at(g, 44, tmp_path)
    assert g._p3_reply_timing() is not None
    assert "18 ms" in tx._clamp_refusal(g.peer_packet_end(44))
    assert tx._place_breakin() == "" and tx.unplaceable


def test_skipped_wall_slots_expire_the_actual_control_reference():
    g = transitioned()
    observed_reply(g)
    assert g._p3_reply_timing() is not None
    # No update calls: a single driver iteration can jump many wall slots.
    g.note_p3_packet(2568704 + 10 * 60000, 38880, 60000)
    assert g._p3_reply_timing() is None


def test_long_cycle_reply_uses_measured_frame_extent(tmp_path):
    # A constructed long-cycle pair uses the same independently supplied
    # reverse gap as the short recorded case; no desired reply boundary is
    # inverted to produce either received coordinate.
    g = onair._MasterGrid(1000000, 60000, 8880, packet_n=46080,
                          cs_n=5760, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(True)
    phase, width = 1017270, 158880
    g.note_p3_packet(phase - 180000, width, 180000)
    g.note_p3_control(1000000)
    g.note_p3_packet(phase, width, 180000)
    assert g._p3_reply_timing().reverse_gap == 7190
    assert g.peer_read_gap == 3850
    tx = tx_at(g, 3, tmp_path)
    assert tx._place_breakin()
    assert tx.boundary == 1180000


@pytest.mark.parametrize("reference", ["audio-start", "pulse-center"])
def test_real_phase_recording_occurs_only_after_emission(reference, tmp_path,
                                                        monkeypatch):
    g = transitioned()
    tx = tx_at(g, 42, tmp_path)
    tx.p3_control_placement = reference
    aimed = g.boundary(42)

    def refuse(audio, what, **kwargs):
        tx.refused = True

    monkeypatch.setattr(tx, "_tx", refuse)
    assert tx._send_p3_control(0) == arq.REFUSED
    assert not g._p3_controls

    def emit(audio, what, **kwargs):
        tx.n += 1
        tx.refused = False
        tx.tx_audio_start = aimed - kwargs["lead_n"]
        tx.tx_pulse_offsets = (684, 924)

    monkeypatch.setattr(tx, "_tx", emit)
    tx._send_p3_control(0)
    # THE REFERENCE THE BURST WAS AIMED BY, in either placement: the whole
    # reply-timing chain compares this against `boundary`, and the physical
    # pulse center is a lead away from it.
    assert g._p3_controls == [(aimed, 60000)]


@pytest.mark.parametrize("reference", ["audio-start", "pulse-center"])
def test_an_ack_train_cannot_manufacture_the_18_ms_changeover_refusal(
        reference, tmp_path, monkeypatch):
    """Either placement's own control, read back through the reply-timing chain."""
    g = transitioned()
    tx = tx_at(g, 42, tmp_path)
    tx.p3_control_placement = reference
    lead = placement.historical_control_pulse_lead(0)

    def emit(audio, what, **kwargs):
        tx.n += 1
        tx.refused = False
        tx.tx_audio_start = tx.boundary - kwargs["lead_n"]
        tx.tx_pulse_offsets = (lead, lead)

    monkeypatch.setattr(tx, "_tx", emit)
    tx._send_p3_control(0)
    control = g._p3_controls[-1][0]
    assert control == g.boundary(42)
    g.note_p3_packet(control + 17280, 38880, 60000)
    later = tx_at(g, 44, tmp_path)
    assert later._clamp_refusal(g.peer_packet_end(44)) is None


def test_recording_the_pulse_center_is_what_refused_every_changeover(tmp_path,
                                                                    monkeypatch):
    """The negative control for the case above, at the same coordinates."""
    g = transitioned()
    tx = tx_at(g, 42, tmp_path)
    lead = placement.historical_control_pulse_lead(0)

    def emit(audio, what, **kwargs):
        tx.n += 1
        tx.refused = False
        tx.tx_audio_start = tx.boundary
        tx.tx_pulse_offsets = (lead, lead)

    monkeypatch.setattr(tx, "_tx", emit)
    tx._send_p3_control(0)
    boundary = g.boundary(42)
    g._p3_controls[:] = [(boundary + lead, 60000)]  # What 0912-2300 recorded.
    g.note_p3_packet(boundary + 17280, 38880, 60000)
    later = tx_at(g, 44, tmp_path)
    assert "18 ms" in later._clamp_refusal(g.peer_packet_end(44))


@pytest.mark.parametrize("swapped", [False, True])
def test_breakin_head_phase_replaces_control_phase_after_trim(swapped, tmp_path, monkeypatch):
    g = transitioned()
    observed_reply(g)
    tx = tx_at(g, 44, tmp_path)
    tx.host = SimpleNamespace(protocol=spec.Protocol.PACTOR3,
                             arq=SimpleNamespace(role=arq.IRS, entry_pending=False))
    tx.breakin_due = True
    monkeypatch.setattr(tx, "_flip", lambda: swapped)
    calls = []

    def capture(audio, what, **kwargs):
        assert tx.unplaceable is None
        calls.append((audio, tx.boundary, kwargs))
        tx.refused = False

    monkeypatch.setattr(tx, "_tx", capture)
    tx.send_packet(3, b"abc", 0, breakin=True)
    audio, phase, args = calls[0]
    trim = np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0]
    # Independent filter geometry: source FIR has 31 taps at8samples/symbol.
    # Resampling maps tap15 to sample900; output padding is not group delay.
    center = 15 * (FS // 100 // 8)
    assert args["lead_n"] == center - trim
    assert phase - args["lead_n"] + center - trim == g.boundary(44)
    assert abs(args["pulse_offsets"][0] - args["pulse_offsets"][1]) == 240
    tx.breakin_due = True
    assert tx.key_instant(g, 44) <= phase - args["lead_n"]


def test_old_clamp_and_collision_guards_still_refuse(tmp_path):
    g = transitioned()
    observed_reply(g)
    tx = tx_at(g, 44, tmp_path)
    # Actual CRC body at2568704: taking the link cannot overlap the packet
    # being answered, even with corroborated reply timing.
    assert g.key_refusal(2569000, 40000, changeover=True) is not None
    g.anchor -= 1000
    assert tx._place_breakin() == "" and "boundary sits" in tx.unplaceable


@pytest.mark.parametrize("late", [False, True])
def test_duplex_breakin_reaches_phase_or_refuses_before_emission(late, tmp_path, monkeypatch):
    g = transitioned()
    observed_reply(g)
    tx = tx_at(g, 44, tmp_path)
    tx.host = SimpleNamespace(protocol=spec.Protocol.PACTOR3,
                             arq=SimpleNamespace(role=arq.IRS, entry_pending=False,
                                                 entry_variant=None, answering=False))
    tx.breakin_due = True
    # All output is intercepted by this arithmetic stream and dummy key seam.
    tx.rig = SimpleNamespace(ptt=lambda state: None, key_failure=lambda: None)
    tx.transmit = True
    clock = ElapsedPCM(np.empty(0), notice=1536)
    tx.live = clock
    clock.samples = clock.pos = g.boundary(44) - (100 if late else 5000)

    def forbidden(*args, **kwargs):
        pytest.fail("placement test attempted per-burst hardware playback")

    monkeypatch.setattr(onair.ota, "_play", forbidden)
    rendered = tx.send_packet(3, b"abc", 0, breakin=True)
    if late:
        assert rendered == arq.REFUSED and tx.refused
        assert not clock.emissions and not tx.keyed
    else:
        assert rendered == 3 and not tx.refused
        assert len(clock.emissions) == len(tx.keyed) == 1
        assert tx.tx_audio_start < g.boundary(44)
        assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == g.boundary(44)
        # PTT is after the current peer's nominal end; it can overlap only the
        # next packet that a successfully decoded CS3 would cancel.
        assert tx.tx_key_up >= g.peer_packet_end(44).at_slot


@pytest.mark.parametrize("older", [False, True])
def test_grid_crc_watermark_protects_timing_and_command_epoch(older):
    g = transitioned()
    observed_reply(g)
    timing, peer = g._p3_timing, g._p3_peer
    g._p3_command_slot, g._p3_command_row0 = 44, 2700000
    g.note_p3_packet(peer[0] - (60000 if older else 0), 38880, 60000)
    assert g._p3_timing == timing and g._p3_peer == peer
    assert (g._p3_command_slot, g._p3_command_row0) == (44, 2700000)


@pytest.mark.parametrize("replay_path", ["rolling", "tracked"])
def test_role_change_does_not_redeliver_old_crc_or_rewind_timing(replay_path):
    s = _Session(role=arq.IRS)
    g = transitioned()
    g._p3_peer = None
    s.host.peer.raster = g
    s.rx._scan_origin = 0
    first = rxfront.Event(t=(2388632 + 4320) / FS, kind="packet", text="first",
                          protocol=spec.Protocol.PACTOR3,
                          packet=(1, 0, b"one", True), start=2388632 + 4320,
                          carrier_swapped=False)
    second = replace(first, t=(2568704 + 4320) / FS, start=2568704 + 4320,
                     text="second")
    for event in (first, second):
        s.rx.new_cycle()
        if event is second:
            g.note_p3_control(2551434)
        s.rx._scan(np.zeros(FS), [lambda audio: (event, "test decoder")])
    assert len(s.packets) == 2
    timing, peer = g._p3_timing, g._p3_peer
    assert timing is not None
    s.host.arq.role = arq.ISS
    g.reverse(to_iss=True)
    s.rx.new_cycle()
    for old in (second, first):
        if replay_path == "rolling":
            s.rx._on(old)
        else:
            s.rx._scan(np.zeros(FS), [lambda audio: (old, "old decoder")])
    assert len(s.packets) == 2
    assert g._p3_timing == timing and g._p3_peer == peer
    # A distinct later physical changeover still reaches the real host.
    future = replace(second, t=2628704 / FS, start=2628704, breakin=True,
                     text="new changeover")
    s.rx._on(future)
    assert len(s.packets) == 3 and s.host.arq.role == arq.IRS
    assert g._p3_peer[0] == 2628704


def test_leaving_p3_releases_delivery_watermark():
    s = _Session(role=arq.IRS)
    s.rx._p3_delivered_at = 123456
    s.host.protocol = spec.Protocol.PACTOR1
    s.rx.new_cycle()
    assert s.rx._p3_delivered_at is None


@pytest.mark.parametrize("evidence", ["new_p3", "older_p3", "p1_alias",
                                     "unknown_protocol", "cs3_head", "energy",
                                     "overlaps_our_key"])
def test_only_new_decoded_p3_bare_control_invalidates_packet_timing(evidence, tmp_path):
    g = transitioned()
    observed_reply(g)
    timing = g._p3_timing
    at = 2620000 if evidence != "older_p3" else 2550000
    if evidence in ("new_p3", "overlaps_our_key"):
        ev = rxfront.Event(t=at / FS, kind="cs", text="REQ",
                           protocol=spec.Protocol.PACTOR3, cs=arq.CS_REQUEST)
        tx = SimpleNamespace(keyings=[(at-1, at+1)] if evidence == "overlaps_our_key" else [],
                             host=SimpleNamespace(arq=SimpleNamespace(dxcall="WS8EOC")),
                             tx_key_up=None)
        onair._forecast_next_key(SimpleNamespace(cs_log=[ev]), tx, g, at-10)
    elif evidence == "energy":
        g.peer_onset = at
    else:
        protocol = {"p1_alias": spec.Protocol.PACTOR1,
                    "unknown_protocol": None}.get(evidence, spec.Protocol.PACTOR3)
        name = {"p1_alias": "CS2/ack", "cs3_head": "BREAK-IN"}.get(evidence, "REQ")
        g.note_peer_codeword(at, 10080, name, "WS8EOC", protocol=protocol)
    if evidence == "new_p3":
        assert g._p3_timing is None and not g._p3_controls
        tx = tx_at(g, 44, tmp_path)
        assert tx._place_breakin() == ""
        assert "no fresh corroborated" in tx.unplaceable
    else:
        assert g._p3_timing == timing


# -- 0913-1550, WS8EOC 40 m: the break-in that was never keyed ---------------
#
# The Winlink greeting arrived whole over PACTOR-3 and the mail client built its
# reply, and then ten consecutive cycles refused the changeover that would have
# taken the turn. Every one of them printed BOTH gates: `SLOT ... IS GONE` at
# +1.4 to +11.4 ms from the grid, and `LATE TO THE KEY ... that instant went
# +1.2 ms ago` from the emission path -- overruns of one to seven milliseconds
# against a reader that forgives five. Twenty-three slots went that way, our
# cadence fell to two slots against a gateway keying every one, and the peer
# stopped being readable long before the link came down.
#
# The coordinates below are that arm's: the placement at +82 ms past the peer's
# packet end, and the overruns the transcript measured.

from hfmodem.tests.shrike.test_grid import _Bench, _Rig            # noqa: E402
from hfmodem.tests.shrike import test_p3_reply_placement as reply  # noqa: E402

SETTLE_N = round(.04 * FS)

# WHERE THE OVERRUN CAN STAND. `_Bench` hands the admission guard one 128-frame
# block at a time -- the quantisation round 14 measured -- so a scene's lateness
# lives on a lattice 2.67 ms apart, and these are its three points either side
# of the tolerance. 1.67 and 4.33 ms are the arm's +1.4 to +5.4 band; 7.00 ms is
# its +7.2.
LATE_N = (80, 208)          # inside what the reader forgives
PAST_N = 336                # ...and past it


def host_stub(role=arq.IRS):
    return SimpleNamespace(
        protocol=spec.Protocol.PACTOR3,
        arq=SimpleNamespace(role=role, entry_pending=False, entry_variant=None,
                            answering=False, rx_seq=0, _rx_seen=True))


def duplex(g, tmp_path, seconds=200.0):
    """A real transmitter on the bench stream, with PTT going nowhere."""
    tx = onair.RadioTx(_Rig(), transmit=True, out_dev=0, outdir=tmp_path,
                       settle=.04)
    tx.live = _Bench(seconds=seconds)
    tx.raster = g
    tx.host = host_stub()
    return tx


PAYLOAD, STATUS = b"ABC", 3


def changeover_lead(swapped=False):
    """Where this burst's CS3 phase reference sits inside its own samples.

    The renderer's own two lines, because the carrier comes up a lead in FRONT
    of the placed instant and that is the instant a scene has to stand against.
    """
    audio = placement.changeover_packet(PAYLOAD, STATUS, swapped=swapped)
    trim = int(np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0])
    return int(np.argmax(placement.protocol_config().pulse())) - trim


def breakin_cycle(tmp_path, late_n, slot=44):
    """The arm's cycle: a placed changeover whose instant has just gone by."""
    g = transitioned()
    g.note_p3_control(2551434)
    g.note_p3_control(2611434)
    g.note_p3_packet(2568704, 38880, 60000,
                     swapped=False, identity=(True, 1, 0, b"RMS"))
    tx = duplex(g, tmp_path)
    tx._flip = lambda: False
    tx.breakin_due = True
    tx.aim(g, slot)
    key = tx._breakin_key(g, slot)
    carrier = key - changeover_lead()
    # Where the stream stands when the changeover reaches `_tx`: the earliest
    # sample the converter can still be handed is `late` past the instant this
    # burst's carrier has to come up at.
    tx.live.now = tx.live.pos = carrier + late_n - tx.live.key_notice
    return g, tx, key, tx.live.clamp_late(carrier)


def keyed_at(tx):
    return tx.live.emissions[-1][0] if tx.live.emissions else None


@pytest.mark.parametrize("late_n", LATE_N)
def test_a_sub_symbol_overrun_keys_the_changeover_into_its_placed_instant(
        late_n, tmp_path, capsys):
    """0913-1550's own cycle, with the forgiveness the controls already had."""
    g, tx, key, late = breakin_cycle(tmp_path, late_n)
    assert late == late_n <= round(onair.BREAKIN_CLAMP_TOL_S * FS)
    slot, end = tx.slot, g.peer_packet_end(tx.slot)
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) != arq.REFUSED
    said = capsys.readouterr().out
    assert onair.KEYED_LATE in said and onair.LATE_KEY not in said
    assert len(tx.live.emissions) == 1 and not tx.refused
    # NOT MOVED, ONLY OCCUPIED LATE. The slot is the one the placement was made
    # in and the carrier is still past the packet it answers -- a slip in the
    # only direction a changeover has room for.
    assert tx.slot == slot
    assert tx.boundary == key - changeover_lead()
    assert 0 < keyed_at(tx) - end.at_slot
    assert 0 < keyed_at(tx) - tx.boundary <= late
    # ...and the turn it asked for is snapshotted, which a refusal never does.
    assert g._p3_turn is not None


@pytest.mark.parametrize("late_n", LATE_N)
def test_reverting_the_forgiveness_reproduces_the_arms_refusal(
        late_n, tmp_path, monkeypatch, capsys):
    """NEGATIVE CONTROL: the exclusion as it flew, on the same cycle.

    `BREAKIN_CLAMP_TOL_S` at zero is exactly what `_tx` did with a placed burst
    before this round -- any overrun at all, and the cycle is given up.
    """
    monkeypatch.setattr(onair, "BREAKIN_CLAMP_TOL_S", 0.0)
    g, tx, key, _ = breakin_cycle(tmp_path, late_n)
    slot = tx.slot
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) == arq.REFUSED
    said = capsys.readouterr().out
    assert onair.LATE_KEY in said and "re-placed on the next cycle" in said
    assert tx.refused and not tx.live.emissions
    assert tx.slot == slot          # and never re-aimed at a boundary of ours
    assert g._p3_turn is None


def test_past_the_forgiveness_the_changeover_is_still_refused_and_never_moved(
        tmp_path, capsys):
    """AND IT IS A FORGIVENESS, NOT A REMOVAL.

    Past half a symbol the placement is one the peer cannot read, and a
    changeover has no second-best boundary to be stepped to: the cycle is given
    up and the FSM places it again against the next reading.
    """
    g, tx, key, late = breakin_cycle(tmp_path, PAST_N)
    assert late == PAST_N > round(onair.BREAKIN_CLAMP_TOL_S * FS)
    slot = tx.slot
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) == arq.REFUSED
    said = capsys.readouterr().out
    assert onair.LATE_KEY in said and onair.KEYED_LATE not in said
    assert tx.refused and not tx.live.emissions and tx.slot == slot


def test_a_refused_changeover_spends_nothing_of_the_reply_clock(tmp_path):
    """The clock that places the NEXT one is untouched by this one's refusal.

    0913-1550 refused ten changeovers in a row and placed an eleventh, so the
    refusal has to be free: the peer's packet, the emitted control phases and
    the corroborated pair between them are all evidence about the far end and
    about bursts that did go out, and a burst that did not is none of it.
    """
    g, tx, key, _ = breakin_cycle(tmp_path, PAST_N)
    before = (g._p3_reply_timing(), list(g._p3_controls), g._p3_peer,
              g._p3_reply_phase_invalid, g.anchor, g.sending)
    assert tx.send_packet(3, PAYLOAD, STATUS, breakin=True) == arq.REFUSED
    assert (g._p3_reply_timing(), list(g._p3_controls), g._p3_peer,
            g._p3_reply_phase_invalid, g.anchor, g.sending) == before
    # ...and the next cycle places it again, on the same evidence.
    tx.breakin_due = True
    tx.aim(g, tx.slot + 1)
    assert tx._place_breakin() and tx.placed and tx.unplaceable is None


# -- the grid's half of the same cycle ---------------------------------------

class _Sessrx:
    """Only what `_regrid` and the emission path ask a session receiver for."""

    frame_seen = False
    p3_receive_offset_hz = 0.0

    def flush(self) -> None:
        pass

    def bridge(self, audio) -> None:
        pass

    def skip(self, seconds: float) -> None:
        pass


# The cycle's own work, charged in front of the GRID's admission check where
# round 14 established the arm pays it -- and 512 is that round's own sawtooth
# charge. It puts the overrun between +0.0 and +2.7 ms across every phase of
# the callback block's residue, which leaves the changeover's FIRST render room
# inside the same tolerance behind it. A cycle with no room for its own render
# is a cycle the changeover cannot be built in, and that is what the round's
# caching of the render is for rather than a looser gate.
PREKEY_N = 512

# ...and what a CHANGEOVER cycle then spends before the emission path's own
# check that an ordinary one does not: `changeover_packet` at 0.97 ms and the
# 2% trim at 0.69, measured on this bench. Paid on the FIRST build of a given
# packet and free on every repeat, which is what `RadioTx._p3_breakin` keeps --
# so this scene charges the worst case, the cycle that builds one.
CHANGEOVER_WORK_N = 80

PEER, CYCLE, PACKET_N = reply.A5_PEER, reply.CYCLE, reply.PACKET_N
IDENTITY = reply.IDENTITY


def stint(tmp_path, cycles=40, prompt_at=None):
    """An IRS answering a CRC packet every cycle, through grid and transmitter.

    Each cycle is placed, admitted and keyed the way the hold loop does it:
    the reply comb onto the peer's packet, `_keyable_slot` with the
    changeover's own lead in hand, the cycle's work charged in front of the
    GRID's check, `_regrid`, the render charged in front of the EMISSION path's
    check, and the burst.
    """
    g = reply.irs_grid()
    tx = duplex(g, tmp_path, seconds=300.0)
    tx._flip = lambda: False
    tx.sessrx = sessrx = _Sessrx()
    tx.defer_p3_cs = True
    live = tx.live
    seg, seg_start = np.zeros(0, np.float32), 0
    slot = (PEER + reply.REPLY_N - g.anchor) // CYCLE
    keyed: list[tuple[int, str]] = []
    for k in range(cycles):
        g.note_p3_packet(PEER + k * CYCLE, PACKET_N, CYCLE, swapped=False,
                         identity=IDENTITY)
        tx.breakin_due = k == prompt_at
        tx.aim(g, slot)
        onair._p3_place_reply(g, tx, slot)
        early = g.boundary(slot) - tx.key_instant(g, slot)
        slot = onair._keyable_slot(live, g, slot, SETTLE_N + early)
        tx.aim(g, slot)
        # Where the cycle's last read leaves the clock, plus the work it does
        # between that read and the grid's gate.
        live.now = live.pos = tx.key_instant(g, slot) - SETTLE_N
        live.spend(PREKEY_N)
        slot, seg, seg_start = onair._regrid(live, g, tx, host_stub(), sessrx,
                                             slot, seg, seg_start, SETTLE_N)
        live.spend(CHANGEOVER_WORK_N if tx.breakin_due else 0)
        before = len(live.emissions)
        if tx.breakin_due:
            tx.send_packet(3, PAYLOAD, STATUS, breakin=True)
        else:
            tx._send_p3_control(arq.CS_ACK)
        if len(live.emissions) > before:
            keyed.append((tx.slot, "BREAKIN" if k == prompt_at else "CS"))
        g.cycles += 1
        slot = tx.slot + 1
    return g, tx, keyed


def test_a_forty_cycle_irs_stint_answers_every_packet_and_loses_no_slot(
        tmp_path, capsys):
    """0913-1550 gave away twenty-three of them. This is the same cycle."""
    g, tx, keyed = stint(tmp_path)
    said = capsys.readouterr().out
    assert [w for _, w in keyed] == ["CS"] * 40
    slots = [s for s, _ in keyed]
    assert set(b - a for a, b in zip(slots, slots[1:])) == {1}
    assert onair.SLOT_GONE not in said and onair.LATE_KEY not in said
    # ...and the cycle really was over its key instant, or the scene asks
    # nothing of the gate it is about.
    assert said.count(onair.SLOT_KEPT) + said.count(onair.KEYED_LATE) >= 40


@pytest.mark.parametrize("prompt_at", [12, 13])
def test_the_turn_is_taken_in_the_answer_slot_of_the_first_eligible_cycle(
        prompt_at, tmp_path, capsys):
    """The mail client asks at cycle k and the changeover keys in slot k.

    Both parities, because the changeover is aimed at the peer's packet and the
    slot's own shift is what a burst holding a rendered polarity would have to
    step in twos to keep.
    """
    g, tx, keyed = stint(tmp_path, prompt_at=prompt_at)
    said = capsys.readouterr().out
    assert [w for _, w in keyed] == ["CS"] * prompt_at + ["BREAKIN"] + \
        ["CS"] * (39 - prompt_at)
    slots = [s for s, _ in keyed]
    assert set(b - a for a, b in zip(slots, slots[1:])) == {1}
    assert onair.SLOT_GONE not in said and onair.LATE_KEY not in said
    # THE CHANGEOVER CARRIES THE CYCLE'S EXTRA WORK, and the grid allowed for it
    # rather than handing the slot to a burst the emission path then refused.
    assert tx.breakin_cost_n >= CHANGEOVER_WORK_N > tx.prekey_cost_n
    assert g._p3_turn is not None


def test_the_grid_keeps_a_changeover_cycles_slot_on_its_own_cost(tmp_path):
    """`_clamp_forgives`, which excluded the changeover and spent the cycle.

    The exclusion read `_tx`'s "nowhere to be moved to" as a rule about the
    forgiveness. It is a rule about the STEP: handing the slot back does not
    re-place a changeover, it spends the cycle the placement was for.
    """
    g, tx, _, _ = breakin_cycle(tmp_path, LATE_N[0])
    tx.prekey_cost_n, tx.breakin_cost_n = 0, CHANGEOVER_WORK_N
    tol = round(onair.BREAKIN_CLAMP_TOL_S * FS)
    assert tx.cycle_cost_n() == CHANGEOVER_WORK_N
    # ON THE WHOLE TOLERANCE, because `_prekey_lead` now stands that same cost
    # off the drain: charged in both places the gate shuts for good. `_tx`
    # measures the interval before it asks the clamp, so a REFUSED changeover
    # records its cost too, and a cost past the tolerance then leaves `room`
    # negative for every changeover behind the first -- which is the seven
    # refusals behind `arm-v23-A-40-ws8eoc`'s first.
    assert onair._clamp_forgives(tx.live, tx, tol) == tol
    assert onair._clamp_forgives(tx.live, tx, tol + 1) == 0
    tx.breakin_cost_n = 2 * tol
    assert onair._clamp_forgives(tx.live, tx, tol) == tol
    # ...while an ordinary cycle's own cost still comes off its tolerance: no
    # drain of its stands it off, so the grid owes it here.
    tx.breakin_due = False
    tx.prekey_cost_n = CHANGEOVER_WORK_N
    room = round(onair.KEY_CLAMP_TOL_S * FS) - CHANGEOVER_WORK_N
    assert onair._clamp_forgives(tx.live, tx, room) == room
    assert onair._clamp_forgives(tx.live, tx, room + 1) == 0


def test_a_stream_with_no_transmitter_forgives_a_changeover_nothing(tmp_path):
    """The one exclusion that stands: there the shortfall is the rig's lead."""
    g, tx, _, _ = breakin_cycle(tmp_path, LATE_N[0])
    tx.breakin_due = True
    assert onair._clamp_forgives(tx.live, tx, 1)
    tx.live.transmit = None
    assert onair._clamp_forgives(tx.live, tx, 1) == 0


def test_a_repeated_changeover_is_rendered_once(tmp_path, monkeypatch):
    """The render and its trim come off the window they were being refused in.

    A changeover is re-placed every cycle until one keys and the packet does
    not change while it is being retried. Round 14 took the frequency shift off
    this window; the render behind it is 0.97 ms and the 2% trim 0.69, measured
    on this bench, against a cycle whose whole margin is single-digit.

    ONCE PER POLARITY, AND ON THE GRID'S OWN `_flip`, which is what the air has.
    `_MasterGrid.shift` is slot parity, so a changeover re-placed on consecutive
    slots alternates the `swapped` the ident carries: `arm-v23-A-40-ws8eoc` put
    eight re-placements on slots 199, 204, 209, 214, 217, 220, 225 and 230 and a
    one-entry cache missed every one of them.
    """
    built: list[tuple[int, bool]] = []
    bare = placement.changeover_packet

    def counted(payload, status, *, swapped=False):
        built.append((len(payload), swapped))
        return bare(payload, status, swapped=swapped)

    g, tx, key, _ = breakin_cycle(tmp_path, LATE_N[0])
    del tx._flip                  # the grid's parity, not the scene's pin
    monkeypatch.setattr(placement, "changeover_packet", counted)
    leads, idents = {}, []
    for cycle in range(6):
        tx.breakin_due = True
        tx.aim(g, 44 + cycle)
        tx.live.now = tx.live.pos = key + cycle * CYCLE - round(.3 * FS)
        tx.send_packet(3, PAYLOAD, STATUS, breakin=True)
        ident = next(reversed(tx._p3_breakin))
        idents.append(ident)
        leads.setdefault(ident, set()).add(tx._p3_breakin[ident][2])
    assert {i[-1] for i in idents} == {False, True}, idents
    assert sorted(built) == sorted({(len(PAYLOAD), s) for s in (False, True)})
    assert all(len(v) == 1 for v in leads.values()) and len(leads) == 2
    # ...and a packet the peer has never seen still pays for itself, once.
    tx.breakin_due = True
    tx.aim(g, 51)
    tx.send_packet(3, b"RMS", STATUS, breakin=True)
    assert len(built) == 3 and built[-1][0] == 3
