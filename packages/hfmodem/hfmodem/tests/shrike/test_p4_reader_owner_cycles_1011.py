# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Delegate actual driver cycle/unread calls; retain raw counts and call order."""
import pytest
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.shrike.p4turn import P4Turn
from hfmodem.tests.shrike.test_p4_second_delta_parent_1011 import Recorder
from hfmodem.tests.shrike.test_p4_second_delta_ordinary_driver_1011 import test_c4_real_driver_source_and_hold_unread as drive
from hfmodem.tests.shrike.reader_owner_support_1011 import record


@pytest.mark.parametrize('variant', ['repeat', 'unread_refinement'])
def test_c4_actual_completed_cycle_prefix_and_raw_snapshot_order(variant, monkeypatch):
    calls, prefix, aliases = [], [], {}
    finish, peer, snapshot = P4Receiver.finish_cycle, P4Turn.peer_cycle, Recorder.snapshot
    def physical(at, rx):
        # Constructed PCM has a known 180000-sample physical raster. Preserve
        # both nominal and genuinely refined read positions in its identity map.
        slot = round((at-480000)/180000)
        nominal = 480000+slot*180000
        assert abs(at-nominal) <= 1680
        values = aliases.setdefault(slot, set())
        values.add(at)
        values.update(o.stream_sample for o in rx.observations if abs(o.stream_sample-nominal) <= 1680)
        return slot
    def counted(rx, at):
        before = dict(raw=rx.silent_cycles, counted=rx._counted_cycle,
                      counted_live=rx._counted_live_onset, live=list(rx.live_onsets),
                      masked=rx._masked(at), provisional_placed_long=[o.stream_sample for o in rx.observations
                          if o.kind == 'long' and o.provisional and o.placed and abs(o.stream_sample-at) < rx.fs//10])
        slot = physical(at, rx)
        result = finish(rx, at)
        consumed = before['counted'] is None or at > before['counted']+rx.fs//10
        row = dict(order=len(calls), stage='finish_cycle', at=at, slot=slot,
                   before=before, result=result, after_counted=rx._counted_cycle,
                   after_live=rx._counted_live_onset, consumed=consumed,
                   support_end=getattr(rx, '_r1_support_end', None), prefix_before=list(prefix))
        if consumed:
            assert slot not in prefix
            prefix.append(slot)
        else:
            assert result == before['raw']
        row['prefix_after'] = list(prefix)
        calls.append(row)
        record('cycle_call', case=variant, **row)
        return result
    def charged(m, silent, at):
        row = dict(order=len(calls), stage='peer_cycle', at=at, silent=silent,
                   state_before=m.state, prefix=list(prefix))
        result = peer(m, silent, at)
        row.update(state_after=m.state, reason=m.reason)
        calls.append(row)
        record('cycle_call', case=variant, **row)
        return result
    def boundary(r, label, reg=None):
        snapshot(r, label, reg)
        record('cycle_raw_snapshot', case=variant, order=len(calls), prefix=list(prefix),
               raw=r.rx.silent_cycles, row=r.rows[-1])
    def support(name):
        original = getattr(P4Receiver, name)
        def read(rx, audio, origin, **kw):
            rx._r1_support_end = origin+len(audio)
            return original(rx, audio, origin, **kw)
        return read
    monkeypatch.setattr(P4Receiver, 'finish_cycle', counted)
    monkeypatch.setattr(P4Turn, 'peer_cycle', charged)
    monkeypatch.setattr(Recorder, 'snapshot', boundary)
    for name in ('read_partial', 'read_window', 'read_chirp'):
        monkeypatch.setattr(P4Receiver, name, support(name))
    drive(variant)
    record('completed_cycles', case=variant, calls=calls,
           physical_aliases={str(k):sorted(v) for k,v in aliases.items()}, prefix=prefix)
    consumed = [r for r in calls if r['stage'] == 'finish_cycle' and r['consumed']]
    assert consumed and len({r['slot'] for r in consumed}) == len(consumed)
    assert all(r['support_end'] is not None and r['support_end'] >= r['at']+157440 for r in consumed)
