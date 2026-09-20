# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Unexpected long headers buy bounded silence; only CRC adopts their cycle."""
from dataclasses import replace

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, p3frame, p3rx, placement, rxfront
from hfmodem.tests.shrike.test_cs6_driver import _arm

FS = onair.FS
SHORT = round(1.25 * FS)
PAYLOAD = b" WS8EOC >\r"


def scene(tmp_path, *, long_cycle=True, swapped=False, erase_body=False):
    audio = np.pad(placement.link_packet(
        3, PAYLOAD, 0x03, long_cycle=long_cycle, swapped=swapped), (4800, 4800))
    row0 = (4800 + (placement.protocol_config().pulse().size - 1) // 2
            + p3frame.DATA_OFFSET * rxfront.SPS)
    if erase_body:
        audio[row0 + rxfront.SyncedRx.WIDEBAND_HEADER_BODY_N:] = 0
    s, tx, grid, clock = _arm(tmp_path, audio, sl=3, expected_seq=3)
    s.host.arq.cfg.long_cycle = False
    s.host.arq._cycle_request = None
    s.host.arq._cycle_command_emitted = False
    s.host.arq._rx_seen = True
    s.rx._p3_row0 = row0 - SHORT
    s.rx._p3_delivered_at = row0 - SHORT
    s.rx._p3_cycle_n = SHORT
    s.rx._p3_span = rxfront._frame_span(placement.SPEED_PATHS[3])
    s.rx._p3_clock_role = arq.IRS
    s.rx.sync.packet_level = 3
    s.rx.p3_receive_offset_hz = 0.
    tx.sessrx = s.rx
    # The caller reaches this helper at its ordinary short-packet deadline,
    # not immediately after the prefix-only header API becomes available.
    initial_end = (row0 + rxfront._packet_span(s.rx._p3_span)
                   - rxfront.SyncedRx.WIDEBAND_EARLY_N + rxfront.SPS // 2)
    clock.pos = clock.samples = clock.end = initial_end
    initial = clock.audio[:initial_end].copy()
    tx._pending_p3_cs = arq.CS_REQUEST
    return s, tx, grid, clock, initial, row0


def run(scene, slot=1, audio=None, origin=0):
    s, tx, grid, clock, initial, _ = scene
    return onair._p3_unexpected_long_window(
        clock, grid, tx, s.host, s.rx, slot,
        initial if audio is None else audio, origin, round(.04 * FS))


def test_header_cancels_reply_and_collects_only_one_bounded_window(tmp_path, monkeypatch):
    sc = scene(tmp_path)
    s, tx, grid, clock, initial, row0 = sc
    calls = []
    collect = onair._collect

    def observed_collect(*args):
        calls.append(args[-1])
        assert tx._pending_p3_cs is None
        assert not s.host.arq.cycle_long
        assert s.rx._p3_row0 == row0 - SHORT
        assert not s.packets
        return collect(*args)

    monkeypatch.setattr(onair, "_collect", observed_collect)
    monkeypatch.setattr(p3rx, "decode_at", lambda *a, **k: None)
    slot, audio, origin = run(sc)
    assert slot == 3
    assert len(calls) == 1
    path = placement.LONG_PATHS[3]
    expected_end = (row0 + rxfront._frame_span(path)
                    - rxfront.UNREAD_TAIL_N + 1)
    delay = (rxfront._matched_filter().size - 1) // 2
    assert expected_end == (row0 + (path.n_symbols - 1) * rxfront.SPS
                            + max(path.clock_offsets(rxfront.SPS)) + delay + 1)
    assert calls == [expected_end + clock.holdback]
    assert clock.pos == expected_end
    assert origin == 0 and len(audio) == expected_end
    assert s.rx._p3_long_window_checked_at == row0
    assert tx._pending_p3_cs is None
    assert not tx.keyed and not s.packets
    assert not s.host.arq.cycle_long and grid.ticks == 1
    assert s.rx._p3_cycle_n == SHORT
    assert s.rx._p3_row0 == s.rx._p3_delivered_at == row0 - SHORT
    assert not onair._p3_current_long_crc(s.rx, slot)


def test_reading_the_same_header_again_cannot_extend_or_recancel(tmp_path, monkeypatch):
    sc = scene(tmp_path, erase_body=True)
    s, tx, grid, clock, initial, row0 = sc
    slot, _, _ = run(sc)
    assert slot == 3 and not s.packets
    pos = clock.pos
    tx._pending_p3_cs = arq.CS_REQUEST
    monkeypatch.setattr(onair, "_collect", lambda *a, **k: pytest.fail("duplicate extended window"))
    monkeypatch.setattr(s.rx.sync, "wideband_header_at", lambda *a, **k: pytest.fail("duplicate header read"))
    again, audio, origin = run(sc, slot=slot)
    assert again == slot and audio is initial and origin == 0
    assert clock.pos == pos
    assert tx._pending_p3_cs == arq.CS_REQUEST


@pytest.mark.parametrize("kind", ["quiet", "weak-header", "short-header"])
def test_uncredible_or_short_header_never_stands_down(tmp_path, monkeypatch, kind):
    sc = scene(tmp_path, long_cycle=kind != "short-header")
    s, tx, grid, clock, initial, row0 = sc
    if kind == "quiet":
        initial[:] = 0
    elif kind == "weak-header":
        read = p3rx.header_of

        def weak(*args, **kwargs):
            h = read(*args, **kwargs)
            assert h is not None
            return replace(h, fit=p3rx.anchor_gate(placement.SPEED_PATHS[3]) - .001)

        monkeypatch.setattr(p3rx, "header_of", weak)
    monkeypatch.setattr(onair, "_collect", lambda *a, **k: pytest.fail("unjustified standdown"))
    slot, audio, origin = run(sc)
    assert slot == 1 and audio is initial and origin == 0
    assert tx._pending_p3_cs == arq.CS_REQUEST
    assert s.rx._p3_long_window_checked_at is None
    assert not s.host.arq.cycle_long and not s.packets and not tx.keyed


@pytest.mark.parametrize("swapped", [False, True])
def test_crc_valid_rendered_long_body_reaches_host_under_short_preference(tmp_path, swapped):
    sc = scene(tmp_path, swapped=swapped)
    s, tx, grid, clock, initial, row0 = sc
    slot, audio, origin = run(sc)
    assert slot == 3
    assert len(s.packets) == 1
    ev = s.packets[0]
    assert ev.packet == (3, 0x03, PAYLOAD, True)
    assert ev.cycle_long and ev.carrier_swapped == swapped
    assert abs(ev.start - row0) <= rxfront.SPS // 4
    assert round(ev.t * FS) == ev.start
    assert s.rx._p3_long_window_checked_at == ev.start
    assert s.host.arq.cfg.long_cycle is False
    assert s.host.arq.cycle_long and grid.ticks == 3
    assert s.rx._p3_cycle_n == 3 * SHORT
    assert s.rx._p3_row0 == s.rx._p3_delivered_at == ev.start
    assert s.rx.frame_seen
    assert not tx.keyed
    assert s.rx._p3_long_crc_reply == (ev.start, slot)
    assert onair._p3_current_long_crc(s.rx, slot)


def test_prefix_with_destroyed_body_does_not_deliver_or_adopt_cycle(tmp_path):
    sc = scene(tmp_path, erase_body=True)
    s, tx, grid, clock, initial, row0 = sc
    slot, _, _ = run(sc)
    assert slot == 3
    assert not s.packets and not s.rx.frame_seen
    assert not s.host.arq.cycle_long and grid.ticks == 1
    assert s.rx._p3_cycle_n == SHORT
    assert s.rx._p3_row0 == row0 - SHORT
    assert tx._pending_p3_cs is None and not tx.keyed
    assert s.rx._p3_long_crc_reply is None
    assert not onair._p3_current_long_crc(s.rx, slot)


@pytest.mark.parametrize("guard", ["iss", "entry", "already-long", "no-clock", "already-delivered"])
def test_out_of_scope_session_cannot_cancel_reply(tmp_path, monkeypatch, guard):
    sc = scene(tmp_path)
    s, tx, grid, clock, initial, row0 = sc
    if guard == "iss":
        s.host.arq.role = arq.ISS
    elif guard == "entry":
        s.host.arq.entry_pending = True
    elif guard == "already-long":
        s.host.arq._cycle_long = True
    elif guard == "no-clock":
        s.rx._p3_row0 = None
    else:
        s.rx._p3_delivered_at = row0
    monkeypatch.setattr(s.rx.sync, "wideband_header_at", lambda *a, **k: pytest.fail("out-of-scope header read"))
    assert run(sc)[0] == 1
    assert tx._pending_p3_cs == arq.CS_REQUEST
    assert not tx.keyed


@pytest.mark.parametrize("change", ["no-proof", "old-row", "old-slot",
                                   "new-delivery", "iss", "p1", "closed"])
def test_current_long_crc_proof_is_scoped_to_packet_slot_and_link(tmp_path, change):
    sc = scene(tmp_path)
    s, tx, grid, clock, initial, row0 = sc
    slot, _, _ = run(sc)
    delivered = s.rx._p3_delivered_at
    assert onair._p3_current_long_crc(s.rx, slot)
    if change == "no-proof":
        s.rx._p3_long_crc_reply = None
    elif change == "old-row":
        s.rx._p3_long_crc_reply = (delivered - SHORT, slot)
    elif change == "old-slot":
        slot += 1
    elif change == "new-delivery":
        s.rx._p3_delivered_at += 3 * SHORT
    elif change == "iss":
        s.host.arq.role = arq.ISS
    elif change == "p1":
        s.host.protocol = onair.Protocol.PACTOR1
    else:
        s.host.arq.state = onair.State.DISCONNECTED
    assert not onair._p3_current_long_crc(s.rx, slot)


def test_helper_entry_clears_old_proof_even_when_guarded_out(tmp_path):
    sc = scene(tmp_path)
    s, tx, grid, clock, initial, row0 = sc
    s.rx._p3_long_crc_reply = (s.rx._p3_delivered_at, 1)
    assert onair._p3_current_long_crc(s.rx, 1)
    s.host.arq.entry_pending = True
    assert run(sc)[0] == 1
    assert s.rx._p3_long_crc_reply is None
    assert not onair._p3_current_long_crc(s.rx, 1)


def test_crc_without_delivered_watermark_advance_cannot_skip_control_reader(tmp_path, monkeypatch):
    sc = scene(tmp_path)
    s, tx, grid, clock, initial, row0 = sc
    decoded = []
    monkeypatch.setattr(s.rx, "_on", lambda ev: decoded.append(ev))
    slot, _, _ = run(sc)
    assert len(decoded) == 1 and decoded[0].packet[3]
    assert s.rx._p3_delivered_at == row0 - SHORT
    assert s.rx._p3_long_crc_reply is None
    assert not onair._p3_current_long_crc(s.rx, slot)
