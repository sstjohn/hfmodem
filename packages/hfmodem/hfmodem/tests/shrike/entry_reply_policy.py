# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Shared entry-policy assertions for the slot and deadline regression tests.

Source helper, included with the tests rather than supplied by recording inputs.
On 2026-10-03, WS8EOC's nine repeated P1 grants were hidden by shortened
entry receive windows. Every entry slot keys: the reply's read stops where the
next entry keys, and a truncated read is not peer silence. Owned grants use the
finite fourteen-grant campaign and neither acknowledge P3 data nor confirm
entry. Actual keys must still meet admission, and refused slots must be counted.
"""
from contextlib import contextmanager
from unittest.mock import patch

from hfmodem.shrike import arq, onair


@contextmanager
def observe_entry_policy():
    evidence = dict(refused=set(), partials={})
    read, regrid = onair._read_held_window, onair._regrid
    repeat = onair._SessionRx._entry_repeat_read

    def partial(self, seg, origin, raster, extent, at):
        result = repeat(self, seg, origin, raster, extent, at)
        proof = self._entry_repeat_result
        if proof['match'] is not None and not proof['complete']:
            assert proof['credited'] and self.host.arq._entry_window_truncated
            assert not self.host.arq._entry_p1_current
            assert proof['match'].observed_bits >= 9
            evidence['partials'][self.host.peer.slot] = (
                raster._p3_keyed_reply, extent, proof['match'])
        return result

    def held(live, raster, tx, sessrx, slot, audio, origin, **kw):
        result = read(live, raster, tx, sessrx, slot, audio, origin, **kw)
        evidence['host'] = sessrx.host
        return result

    def grid(live, raster, tx, host, sessrx, slot, audio, origin, settle):
        result = regrid(live, raster, tx, host, sessrx, slot, audio, origin, settle)
        evidence['refused'].update(s for s in range(slot, result[0])
                                   if s not in tx.slots_used)
        return result

    with (patch.object(onair, '_read_held_window', held),
          patch.object(onair, '_regrid', grid),
          patch.object(onair._SessionRx, '_entry_repeat_read', partial)):
        yield evidence


def assert_entry_policy(got, *, bounded=False):
    evidence = got['entry_policy']
    entries = [b for b in got['bursts'] if 'ENTRY' in b['what'] and not b['refused']]
    assert entries
    slots = [b['slot'] for b in entries]
    assert slots == sorted(set(slots))
    omitted = set(range(slots[0], slots[-1] + 1)) - set(slots)
    assert omitted <= evidence['refused'], omitted
    for b in entries:
        assert 0 <= b['first'] - b['boundary'] <= round(onair.KEY_CLAMP_TOL_S * onair.FS)
    assert not set(slots) & evidence['refused']
    assert 'LISTENING CYCLE' not in got['log']
    assert all(rendered is None
               for b, rendered in zip(got['bursts'], got['rendered'])
               if 'ENTRY' in b['what'] and not b['refused'])
    if bounded:
        host = evidence['host']
        assert host.arq._upgrade_requests == arq.ENTRY_GRANT_CYCLES
        # The first entry and one for each grant of the campaign: no slot of
        # it is given to reading a reply.
        assert len(entries) == arq.ENTRY_GRANT_CYCLES + 1
        assert 'repeated-grant rule' in got['log']
    return entries
