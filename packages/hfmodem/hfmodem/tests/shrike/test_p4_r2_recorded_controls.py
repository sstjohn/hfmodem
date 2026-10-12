# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""R2 unit 2 recording controls, expected to pass before and after.

These bounded tails prove KB5LZK's word and absolute onset only. They do not
predict a peer response after changing local emissions or invent association.
"""
import pytest
from hfmodem.shrike import p4oracle, onair, p1rx, pactor1
from hfmodem.tests.shrike.p4_r2_recordings import (
    FS, KB5LZK_TAILS, RECOVERY_1029_CS_DUE, recording,
)


@pytest.mark.parametrize('start,end,onset', KB5LZK_TAILS)
def test_r2_kb5lzk_recorded_takeover_control(start, end, onset):
    tape = recording('0926-1342')
    origin, audio, gaps = tape.snippet(start, end)
    assert origin == start and len(audio) == end-start and gaps == ()
    assert not tape.own_tx_mask_available
    hit, = p4oracle.word_tail(audio, FS, origin=origin)
    assert hit.word == 'Y27' and hit.score > .99
    # Frozen measured decoder onset, in original absolute stream samples.
    assert abs(hit.stream_sample-onset) <= 1
    with pytest.raises(ValueError, match='mask unavailable'):
        tape.snippet(start, end, require_mask=True)


def test_r2_1029_recording_geometry_and_own_tx_mask():
    tape = recording('1001-1029')
    assert '--p1-robust-stall=12' in tape.options
    assert 'CS due @ 1793346; captured 3.985 s' in tape.log
    assert 'slot 26 p4-hold=42.20ms' in tape.log
    assert 'slot 27 p4-hold=216.75ms' in tape.log
    assert 'slot 30 p4-hold=577.31ms' in tape.log
    assert RECOVERY_1029_CS_DUE == 1793346
    assert tape.own_tx_mask_available
    start, end = tape.own_tx[0]
    origin, audio, gaps = tape.snippet(start, end, require_mask=True)
    assert origin == start and gaps == ((start, end),)
    assert not audio.any()
    assert tape.audio[start:end].any()  # masking never mutates the capture


def test_r2_kb5lzk_hold_driver_control():
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx
    tape = recording('0926-1342')
    start, end, onset = KB5LZK_TAILS[0]
    _, audio, _ = tape.snippet(start, end)
    live = Stream(tape.audio)
    live.pos = end
    tx = Tx(live)
    tx.slot, tx.tx_end = 31, start
    tx.p4_probe.requested = False
    tx.p4_probe.emissions.clear()
    host = tx.p4_link_host
    host.p4_after_first_ack = True
    tx.sessrx.host = host
    grid = onair._MasterGrid(2729, 60000, 8880, packet_n=46080,
                             cs_n=5760, d_max_n=6240)
    heard = onair._p4_hold_word(tx, host, live, grid, audio, start, start)
    assert heard == 29 and tx.p4_probe.peer_breakin
    assert tx.p4_probe.receiver is tx.sessrx.p4 and not tx.slots_used
    assert any(o.word == 'Y27' and abs(o.stream_sample-onset) < FS//100
               for o in tx.sessrx.p4.observations)


def test_r2_2157_p1_ack_recorded_control():
    tape = recording('0930-2157')
    start, end = round(29.75*FS), round(30.03*FS)
    _, audio, _ = tape.snippet(start, end, require_mask=True)
    cs = p1rx.cs_anchored(audio, (round(29.87*FS)-start)/FS)
    assert (cs.index, cs.errors, cs.sense) == (pactor1.CS_ACK_A, 0, 0)
    assert tape.probe['trigger_word'] == 'P1 ACK'
