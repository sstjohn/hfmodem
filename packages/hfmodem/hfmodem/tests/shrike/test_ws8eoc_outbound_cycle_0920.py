# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The 00:46 UTC WS8EOC grant must emit in its first available slot."""
import numpy as np
import pytest

from hfmodem.shrike import arq, onair, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_breakin_timing import duplex
from hfmodem.tests.shrike.test_ws8eoc_long_recovery_0920 import recording

FS, SHORT, LONG = 48000, 60000, 180000


def sender(tmp_path):
    g = onair._MasterGrid(32590, SHORT, 0, packet_n=38880, cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, True
    g.acquired = g.corroborated = True
    g.d_n, g.d_ref_n = 4800, 46320
    g.reply_clock = ReplyClock(entry_phase=364492, pulse_epoch=393442)
    s = _Session(role=arq.ISS)
    tx = duplex(g, tmp_path)
    tx.reply_clock = g.reply_clock
    s.host.peer = tx
    tx.attach(s.host)
    tx.sessrx = s.rx
    a = s.host.arq
    a.speed_level = 1
    a._inflight = arq._Packet(status=0x21, payload=b': W9S', sl=1)
    a._next_seq = 2
    a._outbuf.extend(b'X' * 100)
    a._buffer_raw = 105
    a.payload_ledger.queue(a._inflight.payload + bytes(a._outbuf))
    s.rx._p3_answer_at = 2477538
    g.keyed_slot = 41
    g._p3_keyed_reply = (41, 2492774, 2533241, SHORT)
    tx.slots_used.append(41)
    return s, tx, g


@pytest.mark.parametrize('already_aimed', [False, True])
def test_recorded_grant_keys_first_long_on_old_next_boundary(tmp_path, already_aimed):
    s, tx, g = sender(tmp_path)
    tx.aim(g, 42 if already_aimed else 41)
    tx.live.now = tx.live.pos = 2537598 + 11000
    pcm = recording('outbound-cs6.wav')[:13400]
    heard, _ = s.rx.control_signal_in(pcm, 2537598 - 2400, g)
    assert heard == arq.CS_CYCLE_TOG
    assert s.host.arq.cycle_long and g.cycle_long
    assert tx.slots_used == [41, 42] and not tx.refused
    assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == 2553442
    assert len(tx.live.emissions) == 1
    assert len(s.host.arq._inflight.payload) == 36
    assert s.host.arq._inflight.seq == 2
    # The loop must neither add two phantom slots nor reapply the period shift.
    slot, line = onair._regear_next_slot(g, g.next_slot(tx.slot), True)
    assert slot == 45 and line is None
    # A repeat request retains both the long period and the unacknowledged data.
    packet = s.host.arq._inflight
    tx.aim(g, slot)
    tx.live.now = tx.live.pos = g.boundary(slot) - 10000
    s.host.arq.on_rx_cs(arq.CS_REQUEST)
    assert tx.slots_used[-1] == 45 and not tx.refused
    assert s.host.arq._inflight is packet
    assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == 2553442 + LONG


def test_cycle_commit_does_not_disable_overlap_guard(tmp_path):
    s, tx, g = sender(tmp_path)
    tx.aim(g, 42)
    tx.live.now = tx.live.pos = 2537598 + 11000
    # A peer burst at our actual carrier instant still forbids transmission.
    g.note_peer_codeword(2553442 - 2600, 10080, 'CYCLE-TOG', 'WS8EOC',
                         protocol=spec.Protocol.PACTOR3)
    tx.send_long_packet(1, b'X' * 36, 0x22)
    assert g.cycle_long and tx.refused
    assert not tx.live.emissions


def test_long_silence_recovers_once_without_broad_decoding(tmp_path, monkeypatch):
    s, tx, g = sender(tmp_path)
    g.regear(True)
    tx.aim(g, 45)
    tx.live.now = tx.live.pos = g.boundary(45) + 5000
    calls = []
    def scan(sessrx, audio, origin, *, tracked_only=False):
        calls.append(('scan', tracked_only))
        # The failed arm's scans spent ~90 ms. Charge only bounded recovery
        # 5 ms; broad scans + flush must not be on the recovered deadline.
        tx.live.now += round((.005 if tracked_only else .095) * FS)
    def control(*args):
        calls.append(('control', True))
        tx.live.now += round(.003 * FS)
        return None, None
    def flush(*args):
        calls.append(('flush', False))
        tx.live.now += round(.10 * FS)
    monkeypatch.setattr(onair, '_scan_frame', scan)
    monkeypatch.setattr(s.rx, 'control_signal_in', control)
    monkeypatch.setattr(s.rx, 'flush', flush)
    slot, _, _ = onair._regrid(tx.live, g, tx, s.host, s.rx, 45,
                              np.zeros(0, np.float32), tx.live.pos, 1920)
    assert slot == 48
    assert calls == [('control', True), ('scan', True)]
    assert not tx.live.clamp_late(tx.key_instant(g, slot))
    tx.send_long_packet(1, b'X' * 36, 0x22)
    assert tx.slots_used[-1] == 48 and not tx.refused


@pytest.mark.parametrize('already_aimed', [False, True])
def test_return_to_short_uses_old_long_next_boundary(tmp_path, already_aimed):
    s, tx, g = sender(tmp_path)
    g.regear(True)
    a = s.host.arq
    a.cycle_long = True
    a._inflight = arq._Packet(status=0x22, payload=b'X' * 36, sl=1)
    a._next_seq = 3
    g.keyed_slot = 42
    g._p3_keyed_reply = (42, 2552774, 2712281, LONG)
    tx.slots_used[:] = [42]
    tx.aim(g, 45 if already_aimed else 42)
    phase = 2537598 + LONG
    tx.live.now = tx.live.pos = phase + 11000
    s.rx._p3_answer_at = 2537598
    heard, _ = s.rx.control_signal_in(recording('outbound-cs6.wav')[:13400],
                                      phase - 2400, g)
    assert heard == arq.CS_CYCLE_TOG
    assert not a.cycle_long and not g.cycle_long and not tx.refused
    assert tx.slots_used == [42, 45]
    assert tx.tx_audio_start + min(tx.tx_pulse_offsets) == 2733442
    assert len(a._inflight.payload) == 5 and a._inflight.seq == 3
    assert g.next_slot(45) == 46
