# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""September 9 P3 receive windows and replies, without hardware or wall sleeps."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3rx, placement, rxfront, spec
from hfmodem.tests.shrike.archive import P3_FIXTURES, requires_ws8eoc_p3
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_gateway_acquisition import audio

FS = onair.FS
INDEX = P3_FIXTURES / "ws8eoc-0909-p3.json"
# Read by `parametrize` at collection, which is before any skip can fire, so an
# index that does not cross the publication boundary has to be a value here.
RECORDS = json.loads(INDEX.read_text()) if INDEX.exists() else []


def grid(long=False):
    g = onair._MasterGrid(31430, 60000, 8880,
                          packet_n=46080, cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.sending = False
    g.regear(long)
    return g


@requires_ws8eoc_p3
@pytest.mark.parametrize("row", RECORDS or [None],
                         ids=lambda r: r["file"] if r else "no recordings")
def test_recorded_packets_are_short_despite_long_request(row):
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = row["correction_hz"]
    if row["seq"] == 0:
        s.rx._p3_changeover_pending = True
    onair._scan_frame(s.rx, audio(row["file"]), row["start"])
    ev = s.packets[-1]
    assert ev.packet[:3] == (row["sl"], row["seq"] | (0x20 if row["seq"] else 0),
                              row["payload"].encode())
    assert ev.cycle_long is False
    assert ev.t > row["start"] / FS


@requires_ws8eoc_p3
def test_absolute_lock_survives_skipped_cycles_and_changed_capture_origin():
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -25
    first, second = RECORDS[1:3]
    onair._scan_frame(s.rx, audio(first["file"]), first["start"])
    tracked = s.rx.sync.tracked
    s.rx.new_cycle()
    # The next recovered window opens 350 ms earlier relative to its packet,
    # and five actual short cycles have passed since the last decoded packet.
    prefix = round(.350 * FS)
    moved = np.pad(audio(second["file"]), (prefix, 0))
    onair._scan_frame(s.rx, moved, second["start"] - prefix, tracked_only=True)
    assert s.rx.sync.tracked == tracked + 1
    assert s.packets[-1].packet[2] == b"ode 1"
    assert s.rx.p3_receive_offset_hz == -25


@requires_ws8eoc_p3
def test_same_recording_cannot_count_as_another_clean_repetition():
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -25
    row = RECORDS[1]
    x = audio(row["file"])
    onair._scan_frame(s.rx, x, row["start"])
    sent = list(s.host.peer.sent)
    s.rx.new_cycle()
    onair._scan_frame(s.rx, x, row["start"])
    assert s.host.peer.sent == sent
    assert len(s.packets) == 1


@requires_ws8eoc_p3
def test_early_recovery_allows_a_newer_tracked_frame_before_deferred_reply():
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -25
    first, second = RECORDS[1:3]
    onair._scan_frame(s.rx, audio(first["file"]), first["start"])
    # Same hold iteration: early recovery has already set frame_seen.
    onair._scan_frame(s.rx, audio(second["file"]), second["start"],
                      tracked_only=True)
    onair._scan_frame(s.rx, audio(second["file"]), second["start"],
                      tracked_only=True)
    assert [ev.packet[2] for ev in s.packets] == [b" Trim", b"ode 1"]


def test_prekey_miss_does_not_launch_a_blind_scan(monkeypatch):
    s = _Session(role=arq.IRS)
    monkeypatch.setattr(s.rx.sync, "packet", lambda x, **kw: None)
    def blind(*args, **kwargs):
        pytest.fail("unbounded blind decoder ran at the reply deadline")
    monkeypatch.setattr(rxfront, "decode_expected_packet", blind)
    onair._scan_frame(s.rx, np.zeros(FS), 100000, tracked_only=True)
    assert not s.packets
    assert not s.rx._tracked_only  # Scoped; the next early scan may acquire.


@requires_ws8eoc_p3
def test_recorded_correction_sensitive_repeat_preserves_entry_correction():
    row = RECORDS[-1]
    x = audio(row["file"])
    from hfmodem.shrike import p3acquire
    assert rxfront.decode_expected_packet(p3acquire.compensate(x, -25)) is None
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -25
    onair._scan_frame(s.rx, x, row["start"])
    assert s.packets[-1].packet[2] == b"ode 1"
    assert s.rx.p3_receive_offset_hz == -25


@requires_ws8eoc_p3
@pytest.mark.parametrize("kind", ["noise", "truncated"])
def test_cold_correction_fallback_rejects_incomplete_or_unframed_audio(kind):
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -25
    x = (np.random.default_rng(909).normal(0, .1, FS)
         if kind == "noise" else audio(RECORDS[-1]["file"])[:FS//2])
    onair._scan_frame(s.rx, x, RECORDS[-1]["start"])
    assert not s.packets


def test_recorded_frame_span_leaves_time_for_the_measured_tracked_decode():
    # hold_14's frame: phase 43.5125, row 0 43.6025; the corresponding
    # short reply on the measured grid is 44.40479. The patched tracked
    # reader measured 16.2 ms, which cannot fit after boundary-minus-40 ms
    # once the duplex DAC's 32 ms notice is reserved.
    rx = SimpleNamespace(_p3_row0=round(43.6025*FS),
                         _p3_span=35760, _p3_cycle_n=60000)
    key = 2131430
    ready = onair._p3_frame_ready(rx, key-1920)
    assert ready >= rx._p3_row0 + rx._p3_span
    assert ready + round(.0162*FS) + 1536 < key
    assert key - 1920 + round(.0162*FS) + 1536 > key


def test_latest_crc_packet_is_selected_from_a_recovered_window(monkeypatch):
    old = p3rx.P3Packet(1, 1, b"old", 10000, 1)
    new = p3rx.P3Packet(2, 2, b"new", 130000, 2, long_cycle=True)
    monkeypatch.setattr(p3rx, "decode_headed", lambda *a, **k:
                        (SimpleNamespace(packets=[old, new]), 2))
    ev = rxfront.decode_expected_packet(np.zeros(180000))
    assert ev.packet[2] == b"new"
    assert ev.start == 130000
    assert ev.cycle_long is True


def test_complete_long_frame_survives_a_window_ending_into_the_next_cycle():
    s = _Session(role=arq.IRS)
    packet = placement.link_packet(2, b"long frame", 0x20, long_cycle=True)
    x = np.pad(packet, (4800, FS))
    s.rx.deep_scan(x)
    assert s.packets[-1].packet[2] == b"long frame"
    assert s.packets[-1].cycle_long is True


@requires_ws8eoc_p3
def test_recovered_changeover_window_keeps_its_absolute_origin():
    s = _Session(role=arq.IRS)
    s.rx.p3_receive_offset_hz = -25
    s.rx._p3_changeover_pending = True
    row = RECORDS[0]
    prefix = 5 * FS
    x = np.pad(audio(row["file"]), (prefix, 0))
    onair._scan_frame(s.rx, x, row["start"] - prefix)
    assert s.packets[-1].packet[2] == b"RMS"
    assert abs(s.packets[-1].t - 29.765) < .01


@pytest.mark.parametrize("was,long,last_slot,next_slot", [
    (False, True, 32, 35), (True, False, 35, 36),
    (True, True, 35, 38), (False, False, 32, 33),
])
def test_cycle_switch_preserves_the_last_reply_epoch(was,long,last_slot,next_slot):
    g = grid(was)
    advanced = g.next_slot(last_slot)
    result, _ = onair._regear_next_slot(g, advanced, long)
    assert result == next_slot
    assert g.boundary(result) - g.boundary(last_slot) == (180000 if long else 60000)


def test_duplicate_cycle_request_is_acknowledged_without_repeating_cs6():
    s = _Session(role=arq.IRS)
    s.host.arq.cfg.long_cycle = True
    s.host.arq.cfg.speed_up_after = 1  # A duplicate must still be acknowledged.
    for _ in range(2):
        s.host.on_rx_event(rxfront.Event(0, "packet", "short frame requests long",
            # SL3 supports both lengths, isolating duplicate-request handling.
            protocol=spec.Protocol.PACTOR3, packet=(3, 0x21, b" Trim", True),
            cycle_long=False))
    assert s.host.peer.sent == [("cs", arq.CS_CYCLE_TOG), ("cs", arq.CS_ACK + 1)]
    assert s.host.arq.cycle_request is None
    assert bytes(s.host.channel(s.host.ptchn).rx) == b" Trim"
    s.host.arq.cfg.speed_up_after = 100
    s.host.on_rx_event(rxfront.Event(0, "packet", "physical long frame",
        protocol=spec.Protocol.PACTOR3, packet=(2, 0x22, b"next", True),
        cycle_long=True))
    assert s.host.peer.sent[-1] == ("cs", 0)  # Even counter acknowledged by CS1.
    assert s.host.arq.cycle_long


def test_p3_answers_are_queued_until_the_loop_emits(tmp_path, monkeypatch):
    s = _Session(role=arq.IRS)
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path, settle=.04)
    tx.attach(s.host)
    emitted = []
    def capture(audio, what, **timing):
        assert min(timing["pulse_offsets"]) == timing["lead_n"]
        emitted.append(what)
        # A stub that advances nothing reads as a refusal, which a refused
        # answer is now held for -- so this one reports the air it used.
        tx.tx_end = (tx.tx_end or 0) + len(audio)
    monkeypatch.setattr(tx, "_tx", capture)
    # The aim lead only equals the pulse offset under pulse-center placement.
    tx.p3_control_placement = "pulse-center"
    tx.defer_p3_cs = True
    tx.send_cs(0)
    tx.send_cs(1)
    assert not emitted
    tx.emit_pending_cs()
    tx.emit_pending_cs()
    assert emitted == ["CS2 REPEAT/await seq=1"]


def test_collision_guard_uses_received_length_and_expires():
    g = grid(long=True)  # Our local command has not established peer geometry.
    g.note_p3_packet(100000, 39120, 60000)
    # Next short packet is at 160000. A reply overlapping it must be refused,
    # even though our locally assumed 3.75-second cycle would leave it clear.
    assert g.key_refusal(159000, 12000)
    assert g.key_refusal(142000, 12000) is None
    g.update([], np.zeros(1), 100000)
    assert g.peer_onset == 100000
    g.update([], np.zeros(1), 200000)
    assert g.peer_onset is None
    assert g._peer_air() is None


def test_p3_observation_cannot_guard_a_p1_link():
    g = grid()
    g.note_p3_packet(100000, 39120, 60000)
    g.protocol = spec.Protocol.PACTOR1
    g.peer_onset = None
    assert g._peer_air() is None


def test_fresh_crc_collision_evidence_cannot_be_overridden_by_retries(tmp_path):
    g = grid(long=True)
    g.note_p3_packet(100000, 39120, 60000)
    tx = onair.RadioTx(rig=None, transmit=False, outdir=tmp_path, settle=.04)
    tx.raster = g
    for _ in range(onair.GUARD_MAX_DROPS + 2):
        assert tx._refused(159000, 12000, "CS6")
    assert not tx._refused(142000, 12000, "CS6")


def test_long_reply_recovery_does_not_repeat_a_blind_scan_at_each_key(tmp_path):
    from hfmodem.tests.shrike.test_grid import _Bench, _Rig
    g = grid(long=True)
    bench = _Bench(seconds=20)
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    host = SimpleNamespace(protocol=spec.Protocol.PACTOR3, arq=SimpleNamespace(
        state=arq.State.CONNECTED, role=arq.IRS, cycle_long=True,
        entry_pending=False, cycle_request=None, cycle_command_emitted=False))

    class Receiver:
        def bridge(self, chunk):
            pass

        def deep_scan(self, chunk):
            assert self._tracked_only
            bench._advance(bench.now + round(.004 * FS))

        def flush(self):
            pytest.fail("rolling scan charged against the recovered P3 key")

        def skip(self, seconds):
            pass

    receiver = Receiver()
    receiver.host = host
    tx.live, tx.sessrx = bench, receiver
    tx.defer_p3_cs = True
    tx.aim(g, 1)
    old = bench.read(g.boundary(1) + round(.100 * FS))
    slot, _, _ = onair._regrid(bench, g, tx, host, receiver, 1, old, 0, 1920)
    assert slot == 4  # Exactly the next long-cycle reply, no 14.7 s chase.
    assert bench.clamp_late(tx.key_instant(g, slot)) == 0
