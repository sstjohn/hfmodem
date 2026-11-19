# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""11-17 80 m mail arms: normal packets decided before their key.

Every normal unread at decision on 11-17 was 5-8 ms late: the robust and chirp
probes ran ahead of the normal, the decoder stopped at the decision instant
although the key stayed admissible for 18 ms more, and a whole-window read the
answer did not need came after it. The replays run the station's clock line
under the declared sample clock, with the costs the arms measured charged to it.
"""

from contextlib import ExitStack
import json
import time
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core import wav
from hfmodem.shrike import onair, p4chirprx, p4decode, p4host, p4oracle, p4short
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.evidence import OPERATOR_DATA, require_input
from hfmodem.tests.shrike.test_p4_arm_replay_1112 import declared, preset
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

CAPTURES = {
    'k4spb': 'onair-20261117T053511891283Z-6820f821-e2c6-4071-b2bf-d57116dc1b2e',
    'wm4rb': 'onair-20261117T052356031821Z-d8322e6d-d20f-4159-9ab9-b0e6ca541139',
}
# The arms' FTDX10 clock line: DAC notice 1476, 336 samples of station latency,
# a 40 ms settle -- the last read 49.0 ms ahead of the key, admission 18.2 ms later.
NOTICE, LATENCY, SETTLE = 1476, 336, .040
INPUT_LAG = round(.008*FS)
# The arms at load 2.3-3.9: one turbo iteration; inside read_partial the word
# probe, the normal's placement, the robust field and the chirp header and sweep
# (K4SPB 28.516 began its decode 28 ms in); the whole-window read, all told.
COSTS = dict(iteration=.0033, _word=.004, place_short=.012, robust_at=.008,
             acquire_header=.0026, acquire_sweep=.0024, read_partial=.003, read_window=.0096)


def tape(name, stop_s):
    capture = OPERATOR_DATA/'captures'/CAPTURES[name]
    metadata = json.loads(require_input(capture/'stream.json').read_text())
    recorded = json.loads(require_input(capture/'p4-probe.json').read_text())
    audio = np.zeros(metadata['samples'], np.float32)
    for segment in metadata['segment_artifacts'].values():
        with require_input(capture/segment['wav']).open('rb') as source, wav.decoded(source) as decoded:
            rate, pcm = wavfile.read(decoded)
        assert rate == FS
        start = segment['start_stream_sample']
        audio[start:start+len(pcm)] = (pcm.astype(np.float32)/-np.iinfo(pcm.dtype).min)[:len(audio)-start]
    return audio[:round(stop_s*FS)], recorded


def replay(name, stop_s):
    audio, recorded = tape(name, stop_s)
    intervals = [(e['audio_start'], e['audio_end']) for e in recorded['emissions']]
    for start, end in intervals:
        audio[start:end] = 0
    opening = recorded['emissions'][0]
    stream = Stream(audio)
    stream.pos = stream.wall = opening['audio_end']
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.transmit = True
    stream.key_notice, stream._blk, stream.tx_latency_n = NOTICE, 128, LATENCY
    stream._callback_timing = [(0, 128, 0., 0), (128, 128, .0001, 0), (256, 128, .0002, 0)]
    stream.clamp_late = lambda at: max(0, stream.sample_now()+stream.key_notice-at)
    tx = Tx(stream)
    tx.settle = SETTLE
    tx.p4_probe.emissions = [opening.copy()]
    tx.p4_probe.values = recorded.get('values') or (7, 38)
    for key, value in preset().items():
        setattr(tx.p4_link_options, key, value)
    tx.p4_link_options.p4_link_timeout = (len(audio)-stream.pos)/FS
    schedule, seen = [], set()
    for emission in recorded['emissions']:
        payload = emission.get('payload')
        if payload and payload not in seen:
            seen.add(payload)
            schedule.append([emission['audio_start']-round(1.3*FS), bytes.fromhex(payload)])
    take = stream.take_until
    rx = tx.sessrx.p4

    def charge(seconds):
        stream.wall = stream.sample_now()+round(seconds*FS)

    def feed(end):
        for row in schedule:
            if row[1] is not None and end >= row[0]:
                tx.p4_link_host.arq.on_host_data(row[1])
                row[1] = None
        start = stream.pos
        waits = min(end, len(audio)) > stream.sample_now()
        part = take(end)
        if waits:
            stream.wall = stream.pos+INPUT_LAG
        for left, right in intervals:
            if start < right and stream.pos > left:
                rx.note_gap(max(left, start), min(right, stream.pos))
        return part

    stream.take_until = feed
    partial = []
    for method in ('read_partial', 'read_window'):
        read = getattr(rx, method)

        def timed(*args, _read=read, _method=method, **kwargs):
            partial.append(_method == 'read_partial')
            try:
                result = _read(*args, **kwargs)
            finally:
                partial.pop()
            charge(COSTS[_method])
            return result

        setattr(rx, method, timed)
    key = tx._tx

    def emit(audio, what):
        key(audio, what)
        stream.pos = max(stream.pos, tx.tx_end)
        stream.wall = max(stream.wall, stream.pos)

    tx._tx = emit
    iterations = p4decode.turbo_iterations

    def costly_iterations(*args, **kwargs):
        for result in iterations(*args, **kwargs):
            charge(COSTS['iteration'])
            yield result

    def costly(module, name):
        call = getattr(module, name)

        def wrapped(*args, **kwargs):
            if partial and partial[-1]:
                charge(COSTS[name])
            return call(*args, **kwargs)
        return patch.object(module, name, wrapped)

    adapter = p4host.P4Host
    with ExitStack() as stack:
        declared(stack, stream)
        stack.enter_context(patch.object(p4decode, 'turbo_iterations', costly_iterations))
        stack.enter_context(costly(P4Receiver, '_word'))
        stack.enter_context(costly(p4short, 'place_short'))
        stack.enter_context(costly(p4oracle, 'robust_at'))
        stack.enter_context(costly(p4chirprx, 'acquire_header'))
        stack.enter_context(costly(p4chirprx, 'acquire_sweep'))
        stack.enter_context(patch.object(p4host, 'P4Host', lambda host, **kw: adapter(host, terminal=True)))
        onair._run_p4_link(tx, stream, onair._MasterGrid(opening['audio_start'] % 60000,
            60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return tx.p4_probe


def decisions(probe, lo, hi):
    keyed = {e['at']: e for e in probe.link_events if e['event'] == 'Keyed'}
    return [(e, keyed.get(e['proposed_at'])) for e in probe.link_events
            if e['event'] == 'Decision' and e['expected'] == 'packet' and lo < e['at']/FS < hi]


def admissible(row):
    """Keyable: sampled before the key's admission check."""
    return row['now_sample'] <= row['nominal_due']-NOTICE


def test_k4spb_c2_is_answered_x27_on_its_own_slot():
    probe = replay('k4spb', 31.)
    (row, keyed), = decisions(probe, 28.4, 28.6)
    assert keyed is not None and keyed['word'] == 'X27'
    assert round(keyed['at']/FS, 3) == 29.436 and keyed['purpose'] != 'repeat-request'
    assert admissible(row) and 'read_window' not in row['work_ms']
    assert any(h['kind'] == 'normal' and h.get('status') == 0x02 and 28.4 < h['onset_sample']/FS < 28.6
               for h in probe.heard)


def test_wm4rb_unreadable_normals_are_still_answered_in_their_slots():
    probe = replay('wm4rb', 37.)
    rows = decisions(probe, 28.4, 36.1)
    unread = [(row, keyed) for row, keyed in rows if row['reason'] == 'body unread at decision']
    assert {round(row['at']/FS, 2) for row, _ in unread} >= {28.52, 29.77, 32.27, 34.77, 36.02}
    for row, keyed in rows:
        assert admissible(row) and keyed is not None, row
    assert all(keyed['purpose'] == 'repeat-request' for _, keyed in unread)
    assert {round(row['at']/FS, 2) for row, _ in rows if row['reason'] == 'proposal'} == {31.02, 33.52}


def test_a_placed_normal_is_read_before_the_robust_and_chirp_probes(monkeypatch):
    audio, _ = tape('k4spb', 30.)
    at = round(28.516*FS)
    start = at-round(.5*FS)
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=.85)
    rx.enable_hold()
    rx.chirp = rx.expected = True
    rx._had_irs_emission = True
    rx.clock = Clock(Expect.PACKET, at, .330, 1.25, .85)
    rx.peer_onset = at-60000
    rx.acked_counter = 1
    rx.normal_deadline = lambda onset: time.monotonic()+1.
    probes = []
    for module, name in ((p4oracle, 'robust_at'), (p4chirprx, 'acquire_header'),
                         (p4chirprx, 'acquire_sweep')):
        call = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _call=call, _name=name, **kw:
                            probes.append(_name) or _call(*a, **kw))
    records = rx.read_partial(audio[start:at+round(.81*FS)], start, deadline=at+round(.81*FS))
    normal, = [r.observation for r in records if r.observation.kind == 'normal']
    assert normal.frame is not None and normal.frame.status.byte == 0x02
    assert probes == []


@pytest.mark.parametrize('knee', [p4short.NORMAL_COHERENCE_KNEE, .20])
def test_a_weak_robust_short_on_a_normal_slot_stays_a_short(monkeypatch, knee):
    # 2031 64.754: the v53 short (robust head 0.245) places as a normal at
    # 0.236-0.270, just under the knee. A knee below that stands in for a
    # stronger copy: an unread normal must not swallow the short.
    from hfmodem.tests.shrike.test_p4_robust_admission_1001 import capture
    monkeypatch.setattr(p4short, 'NORMAL_COHERENCE_KNEE', knee)
    monkeypatch.setattr(p4short, 'NORMAL_ACQUIRE_KNEE', knee)
    audio = capture('2031')
    at, cfo = round(64.754*FS), 2.63
    origin, early = at-round(.05*FS), at+round(.81*FS)
    rx = P4Receiver(FS, admitted=True, breakin=True, cfo_hz=cfo)
    rx.enable_hold()
    rx._had_irs_emission = True
    rx.clock = Clock(Expect.PACKET, at, .33, 1.25, cfo)
    rx.peer_onset = at-60000
    obs = rx._packet(audio[origin:early], origin, partial=True)
    assert obs is not None and obs.kind == 'short' and obs.value == 53
