# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Three 2026-11-14 mail arms replayed under the declared sample clock: K4MSU
05:05Z (skipped data slots, a body over the peer's turn, a close keyed into its
ISS raster), K4SPB 05:11Z (an IRS close that keyed nothing) and WS8EOC 04:48Z
(an uninvited break-in on a repeat we had acked)."""

from contextlib import ExitStack
import json
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from hfmodem.core import wav
from hfmodem.shrike import onair, p4host, p4turn
from hfmodem.tests.evidence import OPERATOR_DATA, require_input
from hfmodem.tests.shrike.test_p4_arm_replay_1112 import declared, preset
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

CAPTURES = {
    'k4msu': 'onair-20261114T050519328936Z-d7f28db7-e0ff-47f0-bd94-a4b765a5b009',
    'k4spb': 'onair-20261114T051147493217Z-5e953a6c-b9f9-4a64-b262-a275de796517',
    'ws8eoc': 'onair-20261114T044831098622Z-7dae9d3f-c71b-4bf6-bae9-12d1df929ef6',
}
_cache = {}


def tape(capture):
    metadata = json.loads(require_input(capture / 'stream.json').read_text())
    assert metadata['samplerate'] == FS
    audio = np.zeros(max(s['end_stream_sample'] for s in metadata['segment_artifacts'].values()),
                     np.float32)
    for segment in metadata['segment_artifacts'].values():
        with (capture / segment['wav']).open('rb') as source, wav.decoded(source) as decoded:
            rate, pcm = wavfile.read(decoded)
        assert rate == FS
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        part = pcm.astype(np.float32)
        if np.issubdtype(pcm.dtype, np.integer):
            part /= -np.iinfo(pcm.dtype).min
        audio[segment['start_stream_sample']:segment['start_stream_sample'] + len(part)] = part
    return audio


def replay(name, *, read_window=.004, read_partial=.003, fail_at=None, queued=None, **options):
    key = (name, read_window, read_partial, fail_at, queued, tuple(sorted(options.items())))
    if key in _cache:
        return _cache[key]
    capture = OPERATOR_DATA / 'captures' / CAPTURES[name]
    recorded = json.loads(require_input(capture / 'p4-probe.json').read_text())
    audio = tape(capture)
    intervals = [(e['audio_start'], e['audio_end']) for e in recorded['emissions']]
    for start, end in intervals:
        audio[start:end] = 0
    opening = recorded['emissions'][0]
    stream = Stream(audio)
    stream.pos = stream.wall = opening['audio_end']
    stream.sample_now = lambda: max(stream.pos, stream.wall)
    stream.transmit = True
    stream.key_notice, stream._blk, stream.tx_latency_n = 1796, 128, 144
    stream._callback_timing = [(0, 128, 0., 0), (128, 128, .0001, 0), (256, 128, .0002, 0)]
    stream.clamp_late = lambda at: max(0, stream.sample_now() + stream.key_notice - at)
    tx = Tx(stream)
    tx.settle = .037
    tx.p4_probe.emissions = [opening.copy()]
    tx.p4_probe.values = recorded.get('values') or (7, 38)
    for option, value in {**preset(), **options}.items():
        setattr(tx.p4_link_options, option, value)
    tx.p4_link_options.p4_link_timeout = (len(audio) - stream.pos) / FS
    schedule, seen = [], {queued[1].hex()} if queued is not None else set()
    for emission in recorded['emissions']:
        payload = emission.get('payload')
        if payload and payload not in seen:
            seen.add(payload)
            schedule.append([emission['audio_start'] - round(1.3 * FS), bytes.fromhex(payload)])
    if queued is not None:
        schedule.append([round(queued[0] * FS), queued[1]])
    take = stream.take_until
    rx = tx.sessrx.p4

    def feed(end):
        for row in schedule:
            if row[1] is not None and end >= row[0]:
                tx.p4_link_host.arq.on_host_data(row[1])
                row[1] = None
        start = stream.pos
        part = take(end)
        for left, right in intervals:
            if start < right and stream.pos > left:
                rx.note_gap(max(left, start), min(right, stream.pos))
        return part

    stream.take_until = feed
    for method, cost in (('read_partial', read_partial), ('read_window', read_window)):
        read = getattr(rx, method)

        def timed(*args, _read=read, _cost=cost, **kwargs):
            result = _read(*args, **kwargs)
            stream.wall = stream.sample_now() + round(_cost * FS)
            return result

        setattr(rx, method, timed)
    key = tx._tx

    def emit(audio, what):
        key(audio, what)
        stream.pos = max(stream.pos, tx.tx_end)
        stream.wall = max(stream.wall, stream.pos)

    tx._tx = emit
    adapter = p4host.P4Host
    with ExitStack() as stack:
        declared(stack, stream)
        if fail_at is not None:
            # The live arm's mail session failed here; the replay has no client.
            decide = p4turn.P4Turn.decide

            def failing(machine, now):
                if now >= fail_at * FS and not machine.app_done:
                    machine.finish_application(now, failure='gateway banner ended without a SID')
                return decide(machine, now)

            stack.enter_context(patch.object(p4turn.P4Turn, 'decide', failing))
        stack.enter_context(patch.object(p4host, 'P4Host', lambda host, **kw: adapter(host, terminal=True)))
        onair._run_p4_link(tx, stream, onair._MasterGrid(opening['audio_start'] % 60000,
            60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    _cache[key] = tx.p4_probe
    return tx.p4_probe


def keyed(probe, start=0., end=1e9):
    return [e for e in probe.link_events if e['event'] == 'Keyed' and start <= e['at'] / FS < end]



def test_k4msu_peer_turn_in_our_close_is_answered_with_words_and_asks_before_closing():
    probe = replay('k4msu')
    grant = keyed(probe, 82.4, 82.7)
    assert [(e['kind'], e['word'], e['purpose']) for e in grant] == [('word', 'V27', 'close-grant')]
    after = keyed(probe, 83.)
    assert after and all(e['kind'] == 'word' for e in after)
    assert not any(e['state'] == 'ISS_CLOSE' for e in after)
    asks = [e for e in after if e['state'] == 'IRS_BREAKIN']
    assert len(asks) >= 3 and all(e['word'] == 'Y27' for e in asks)
    assert probe.reason != 'close silence'


def test_k4msu_slow_ack_read_keys_the_armed_repeat_in_every_slot():
    probe = replay('k4msu', read_window=.025)
    # Eight tries of the c0 body, then the close its exhausted budget owes.
    data = [e['at'] for e in keyed(probe) if e['kind'] == 'body']
    assert len(data) == 9 and keyed(probe, 74.)[0]['status'] == 0x20
    assert all(b - a == round(1.25 * FS) for a, b in zip(data, data[1:]))
    assert any(e['event'] == 'armed-repeat' for e in probe.link_events)


def test_k4msu_first_body_after_a_bare_y27_turn_carries_c1_only_when_asked():
    for counter, status in (('c0', 0), ('c1', 1)):
        probe = replay('k4msu', p4_y27_first_body=counter)
        first = next(e for e in keyed(probe) if e['kind'] == 'body' and e.get('payload'))
        assert first['status'] == status


def test_k4spb_close_after_mail_failure_asks_in_every_long_slot():
    probe = replay('k4spb', fail_at=153.509)
    asks = keyed(probe, 153.5)
    assert len(asks) == 4 and all((e['word'], e['purpose']) == ('Y27', 'close-request') for e in asks)
    assert all(abs((b['at'] - a['at']) - round(3.75 * FS)) <= FS // 50 for a, b in zip(asks, asks[1:]))
    # The break-in bound ends it, never "peer closed" over registering longs.
    assert probe.reason == 'break-in unanswered'


def test_ws8eoc_repeat_of_an_acked_greeting_is_acked_again_not_broken_into():
    # The mail client queued its answer to the greeting read at 31.013.
    probe = replay('ws8eoc', queued=(31.9, b'[Pat-1.0.0-B2FHM$]\r; WS8EOC DE W9SSJ\rFF\r'))
    after = keyed(probe, 32.5)
    assert after and all(e['kind'] == 'word' for e in after)
    assert after[0]['word'] == 'V27' and round(after[0]['at'] / FS, 3) == 33.183
    assert probe.reason != 'break-in unanswered'
