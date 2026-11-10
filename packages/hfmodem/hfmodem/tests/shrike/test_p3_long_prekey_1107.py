# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A long packet's word comes from that packet, read in front of its answer.

Each recording is the window one 11-07 hold had when its pre-key reads ran,
and each was answered on air with the cycle before's word: K4MSU's long SL1
at a header that faded below its gate (15:31 hold 13) or sat a quarter symbol
off its frame (hold 11), and WS8EOC's first long SL2 packet after our CS4
(15:19 hold 11).
"""
import hashlib
import json
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session

FIXTURE = Path(__file__).with_name('fixtures') / 'p3-long-prekey-1107'
if not (FIXTURE / 'metadata.json').is_file():
    pytest.skip('11-07 PACTOR-3 recordings are not included in this distribution',
                allow_module_level=True)
ROWS = json.loads((FIXTURE / 'metadata.json').read_text())


def recorded(row):
    path = FIXTURE / row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    fs, raw = wavfile.read(path)
    assert fs == onair.FS and raw.ndim == 1
    return raw.astype(float) / 32768


def receiver(row):
    s = _Session(role=arq.IRS)
    s.host.arq.cycle_long = True
    s.host.arq._rx_seen = True
    s.host.arq._expected_seq = row['status'] & spec.STATUS_SEQ
    s.host.arq._p3_rx_sl = 1
    s.rx._p3_row0 = s.rx._p3_delivered_at = row['previous_row0']
    s.rx._p3_clock_role = arq.IRS
    s.rx._p3_cycle_n = round(spec.CYCLE_LONG_S * onair.FS)
    s.rx._p3_span = rxfront._frame_span(placement.LONG_PATHS[1])
    s.rx.sync.packet_level = 1
    s.rx.p3_receive_offset_hz = row['offset_hz']
    if row['sl'] == 2:
        s.rx.note_p3_speedup_emitted(1, 2)
    return s


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['file'])
def test_long_packet_is_answered_from_its_own_cycle(row, monkeypatch):
    """The bounded readers alone, as the pre-key read takes them: the general
    tracked decoder behind them is not where a long reply's word may come from."""
    s = receiver(row)
    monkeypatch.setattr(s.rx, '_read_p3_packet',
                        lambda *a, **k: pytest.fail('the general reader opened'))
    onair._scan_frame(s.rx, recorded(row), row['start'], tracked_only=True)
    assert [ev.packet[:2] for ev in s.packets] == [(row['sl'], row['status'])]
    (ev,) = s.packets
    assert ev.cycle_long
    assert ev.packet[2].startswith(row['prefix'].encode())
    assert s.rx._p3_delivered_at == row['row0']
    # A long packet whose bit 5 has fallen is answered with the way back,
    # which acknowledges it too; one still asking for the long cycle, by parity.
    word = (arq.CS_CYCLE_TOG if not row['status'] & spec.STATUS_LONG_CYCLE
            else arq.CS_REQUEST if row['status'] & 1 else arq.CS_ACK)
    assert s.host.peer.sent[-1] == ('cs', word)


@pytest.mark.parametrize('sl,climbs', [(1, True), (2, False)])
def test_long_cycle_speed_up_stops_at_the_level_read_before_its_answer(sl, climbs):
    """SL1 -> SL2 on the long cycle is read in front of its answer; SL3 is not."""
    from hfmodem.tests.shrike.test_changeover_progression import _linked
    link, air = _linked()
    link.role = arq.IRS
    link.cycle_long = True
    for seq, payload in [(1, b'a'), (2, b'b'), (3, b'c')]:
        link.on_rx_packet(sl, payload, spec.status_byte(seq, long_cycle_request=True), True,
                          protocol=spec.Protocol.PACTOR3, cycle_long=True)
    assert air.controls == [arq.CS_ACK, arq.CS_ACK,
                            arq.CS_SPEED_UP if climbs else arq.CS_ACK]
