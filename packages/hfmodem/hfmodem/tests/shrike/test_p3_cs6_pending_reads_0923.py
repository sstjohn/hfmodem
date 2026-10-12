# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A pending CS6 reads the long frame alone in front of the long reply's key.

0923-1948 WS8EOC-80: CS6 at slot 21, the peer went long on the next row, and
its long replies were due at slots 24, 27, 30 and 33. Each walk read the long
frame (9.66 ms) and then a short read of the whole grown window (37.58 ms) in
front of the endpoint's key, against 23.0 ms of room: SLOT 24 IS GONE +24.2 ms,
then 27, 28, 29, 33. Here the grid's slot 0 is the arm's slot 21.
"""
import numpy as np

from hfmodem.shrike import arq, onair, p3frame, rxfront
from hfmodem.tests.shrike.test_cs6_driver import _arm
from hfmodem.tests.shrike.test_cs6_stalls import _duplex

FS = 48000
LONG_MS, SHORT_MS, ROOM_MS = 9.66, 37.58, 23.0


def _walks(tmp_path, monkeypatch):
    tape = np.zeros(18 * FS)
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 1536)
    clock.holdback = 660  # the FT-891's, which the arm's room was measured on
    reads = []
    scan = onair._scan_frame

    def counted(sessrx, audio, origin, **kw):
        reads.append((clock.pos, origin, audio.size, kw.get("p3_row0")))
        ms = LONG_MS if kw.get("p3_row0") is not None else SHORT_MS
        clock.samples += round(ms * FS / 1e3)
        return scan(sessrx, audio, origin, **kw)

    monkeypatch.setattr(onair, "_scan_frame", counted)
    audio, origin = np.zeros(0), 0
    walks = []
    for start in (1, 4, 7, 10):
        tx.aim(g, start)
        until = onair._p3_frame_ready(
            s.rx, onair._p3_decode_deadline(clock, tx.key_instant(g, start), 1920))
        audio, origin, _ = onair._collect(clock, s.rx, audio, origin, until)
        del reads[:]
        end, audio, origin = onair._p3_transition_window(
            clock, g, tx, s.host, s.rx, start, audio, origin, 1920)
        walks.append((start, end, list(reads)))
        tx.aim(g, end)
        tx._send_p3_control(arq.CS_REQUEST)
    return s, tx, g, walks


def test_each_long_reply_slot_is_reached_with_one_read_in_front_of_it(
        tmp_path, monkeypatch):
    s, tx, g, walks = _walks(tmp_path, monkeypatch)
    assert [end for _, end, _ in walks] == [3, 6, 9, 12]  # the arm's 24..33
    assert s.host.arq.cycle_command_emitted and not s.host.arq.cycle_long
    for start, end, reads in walks:
        gate = [r for r in reads if r[0] > tx.key_instant(g, end - 1)]
        assert len(gate) == 1 and gate[0][3] is not None
        spent = sum(LONG_MS if r[3] is not None else SHORT_MS for r in gate)
        assert spent <= ROOM_MS


def test_probe_reads_take_only_the_frame_they_can_decode(tmp_path, monkeypatch):
    s, _, _, walks = _walks(tmp_path, monkeypatch)
    span, cycle_n = onair._p3_receive_geometry(s.rx)
    lead = p3frame.DATA_OFFSET * rxfront.SPS + 2400
    for _, _, reads in walks:
        for _, origin, size, row in reads:
            if row is None:
                assert size <= cycle_n + rxfront._packet_span(span) + lead
            else:
                assert origin == row - lead


def test_the_transmitter_keys_every_long_reply_slot(tmp_path, monkeypatch):
    tx = _walks(tmp_path, monkeypatch)[1]
    assert tx.slots_used == [3, 6, 9, 12] and not tx.refused
