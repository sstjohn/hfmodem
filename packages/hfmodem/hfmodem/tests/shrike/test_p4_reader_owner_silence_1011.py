# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Finite physical accounting and masking; no test-added finish in the driver."""
from unittest.mock import patch
import numpy as np
from hfmodem.shrike import onair, p4turn
from hfmodem.shrike.p4receiver import P4Receiver, Clock, Expect, Emission, Role
from hfmodem.tests.shrike.test_p4_link_0927 import Tx, FS
from hfmodem.tests.shrike.test_p4_part_a_driver_1011 import AbsoluteStream
from hfmodem.tests.shrike.test_p4_opus_driver_1011 import GRID
from hfmodem.tests.shrike.test_p4turn import word
from hfmodem.tests.shrike.reader_owner_support_1011 import record


def test_c4_real_driver_exact_sixteen_completed_packet_cycles_then_down():
    onset, cycle = 480000, 180000
    # Frame-less established PACKET context: silence is actual zero PCM,
    # with no fabricated classification, source, response or decoder result.
    live = AbsoluteStream(np.zeros(17*cycle, np.float32), onset-24000)
    tx = Tx(live)
    tx.settle = .025
    tx.p4_probe.emissions, tx.p4_probe.peer_breakin = [], True
    tx.p4_link_options.p4_ladder, tx.p4_link_options.p4_data = 'none', 'body'
    tx.p4_link_options.p4_link_timeout = len(live.audio)/FS
    rx = tx.sessrx.p4
    rx.clock = Clock(Expect.PACKET, onset, .330, 3.75, 0.)
    rx.expected = True
    cycles, machines = [], []
    original = rx.finish_cycle
    def finish(at):
        before = rx.silent_cycles
        counted = rx._counted_cycle
        result = original(at)
        cycles.append((at, counted, live.pos, before, result))
        return result
    rx.finish_cycle = finish
    cls = p4turn.P4Turn
    def factory(*a, **kw):
        kw['role'] = 'irs'
        m = cls(*a, **kw)
        machines.append(m)
        return m
    with patch.object(p4turn, 'P4Turn', factory):
        onair._run_p4_link(tx, live, GRID)
    m = machines[0]
    record('finite_silence_driver', context='frame-less quiet established PACKET', cycles=cycles,
           state=m.state, reason=m.reason, slots=tx.slots_used, events=tx.p4_probe.link_events)
    assert len(cycles) == 16
    assert cycles[0][0] == onset
    # With no admitted physical frame, the ordinary default V27 response
    # installs the next packet clock at its real IRS key+.330. Record that
    # actual raster; do not substitute a nominal placed packet history.
    assert [r[0] for r in cycles[1:]] == [e['audio_start']+15840 for e in tx.p4_probe.emissions[:-1]]
    assert [r[4] for r in cycles] == list(range(1, 17))
    assert all(end >= at+157440 for at, counted, end, before, after in cycles)
    assert m.state == p4turn.State.DOWN and m.reason == 'peer silent'
    assert rx.silent_cycles == 16
    keyed = [e for e in tx.p4_probe.link_events if e['event'] == 'Keyed']
    assert len(keyed) == len(tx.slots_used) == len(tx.p4_probe.emissions) == 16
    assert all(e['kind'] == p4turn.Kind.WORD and e['state'] == p4turn.State.IRS_ANSWER and e['word'] == 'V27' for e in keyed)
    assert tx.p4_link_host.arq.tx_acked_bytes == 0


def test_c4_semantic_mask_and_duplicate_finish_use_actual_emission_and_registration():
    rx = P4Receiver(FS, admitted=True)
    rx.enable_hold()
    first = 480000
    rx.note_emission(Emission(p4turn.Kind.SHORT, Role.ISS, first-44880, .758))
    reg = rx.register(word('Z27', first, extent=.245).observation)
    assert reg.responds_to == first-44880 and rx.peer_onset == first
    assert rx.finish_cycle(first) == 0
    masked = first+60000
    rx.note_emission(Emission(p4turn.Kind.SHORT, Role.ISS, masked, .758))
    assert rx._masked(masked)
    before = rx._counted_live_onset, rx.silent_cycles
    assert rx.finish_cycle(masked) == 0
    counted = rx._counted_cycle
    assert rx.finish_cycle(masked) == 0
    assert rx._counted_cycle == counted and before == (rx._counted_live_onset, rx.silent_cycles)
    record('semantic_mask_duplicate', first=first, masked=masked, actual_emissions=rx.emissions,
           counted=counted, count=rx.silent_cycles, liveness=rx.live_onsets,
           label_context='separate receiver component; ordinary driver finish recorder adds no calls')
