# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A gateway's first changeover packet is acknowledged in its first slot.

The window in front of that slot ends short of the 0.81 s packet, and every
first changeover on the 11-07 tapes was answered with the seeded CS2 although
its body was already in that window: K4MSU 15:31 hold 5 and 04:51 hold 5, and
WS8EOC's empty 0x40 changeover at 15:19 hold 42.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, placement, rxfront

FIXTURE = Path(__file__).with_name('fixtures') / 'p3-first-changeover-1107'
if not (FIXTURE / 'metadata.json').is_file():
    pytest.skip('11-07 PACTOR-3 recordings are not included in this distribution',
                allow_module_level=True)
ROWS = json.loads((FIXTURE / 'metadata.json').read_text())


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['file'])
def test_first_changeover_body_is_acknowledged_in_its_first_slot(tmp_path, monkeypatch, row):
    from hfmodem.tests.shrike.test_cs6_stalls import _buffered_listen
    from hfmodem.tests.shrike.test_p3_first_changeover_ack import arm
    s, tx, live, g, seg = arm(tmp_path, row, fixtures=FIXTURE)
    delivered = []
    s.host.deliver = delivered.append
    seg = onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0, np.zeros(0), 0,
                                    1920, reserve_s=onair.PREKEY_RESERVE_S)
    assert seg.size == s.rx.cs_at + s.rx.CHANGEOVER_READ_N
    assert [ev.packet[2] for ev in s.packets] == [row['payload'].encode()]
    assert b''.join(delivered) == row['payload'].encode()
    assert tx._pending_p3_cs == arq.CS_ACK
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot, _, _ = onair._regrid(live, g, tx, s.host, s.rx, 0, seg, 0, 1920)
    s.host.tick()
    tx.emit_pending_cs()
    assert not tx.refused and tx.slots_used == [slot] == [0]
    assert tx.tx_audio_start + row['origin'] == row['first_ack_audio']


class _ArmedCapture:
    """`_LiveInput.arm` on the harness clock: an armed burst keys on its own
    instant once the commit is past the DAC's notice."""
    _armed = None

    @property
    def armed_at(self):
        return None if self._armed is None else self._armed[0]

    def arm(self, audio, *, at, settle, key=None, max_key=40.0):
        late = self.clamp_late(at)
        if late:
            raise onair._MissedTxSlot(late)
        self._armed = (at, np.asarray(audio, np.float32), settle)

    def disarm(self):
        if self._armed is None:
            return True
        at, _, settle = self._armed
        if self.samples >= at - round(settle * onair.FS):
            return False
        self._armed = None
        return True

    def abort(self):
        self._armed = None

    def transmit(self, audio, *, at, settle, key, max_key, admit=None):
        armed, self._armed = self._armed, None
        self.missed = self.clamp_late(at) if armed is not None else 0
        if not self.missed:
            return super().transmit(audio, at=at, settle=settle, key=key,
                                    max_key=max_key, admit=admit)
        if admit is not None:
            admit()
        at, audio, settle = armed
        self.keyed_at = at - round(settle * onair.FS)
        self.keyed_s = (at + len(audio) - self.keyed_at) / onair.FS
        self.emissions.append((at, at + len(audio), self.keyed_at))
        self.samples = max(self.samples, at + len(audio))
        return at, self.samples


@pytest.mark.parametrize('overrun', [0, 4800])
@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['file'])
def test_the_first_answer_is_armed_before_the_body_decode(tmp_path, monkeypatch, row, overrun):
    """A body decode that runs long cannot hand the first answer's slot on:
    the reply is armed once the turn makes us the IRS, ahead of the decode."""
    from hfmodem.tests.shrike.test_cs6_stalls import _buffered_listen
    from hfmodem.tests.shrike.test_p3_first_changeover_ack import Capture, arm
    s, tx, live, g, seg = arm(tmp_path, row, fixtures=FIXTURE)
    live.__class__ = type('Armed', (_ArmedCapture, Capture), {})
    decode = rxfront._cs_event

    def slow_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        live.samples += overrun
        return result
    monkeypatch.setattr(rxfront, '_cs_event', slow_decode)
    seg = onair._receive_changeover(live, g, tx, s.host, s.rx, 0, seg, 0, np.zeros(0), 0,
                                    1920, reserve_s=onair.PREKEY_RESERVE_S)
    armed = tx._p3_armed
    assert armed is not None and armed.slot == 0
    monkeypatch.setattr(onair, '_listen_until_answer', _buffered_listen)
    slot, _, _ = onair._regrid(live, g, tx, s.host, s.rx, 0, seg, 0, 1920)
    s.host.tick()
    tx.emit_pending_cs()
    assert tx.slots_used[:1] == [0] and len(live.emissions) == 1
    assert live.emissions[0][0] + row['origin'] == row['first_ack_audio']
    assert tx._p3_emitted_cs == (arq.CS_ACK if not overrun else armed.cs)
