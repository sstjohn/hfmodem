# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Whole recorded mail arms under the declared sample clock."""

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.core import wav
from hfmodem.shrike import onair, p4hold, p4host, p4longdecode, p4longnative
from hfmodem.shrike import p4receiver, p4shortlevels, p4turn
from hfmodem.tests.evidence import RECORD, TREE, require_input
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

CAPTURES = {
    've3kpg': 'onair-20261013T023405134222Z-fbba7fb6-ec76-4336-a6c6-7a0e1bad7601',
    'ws8eoc': 'onair-20261027T063407663476Z-3435ae7d-f8b1-4b34-b8ea-b1b8e916e6f1',
    'ws8eoc-1114-breakin': 'onair-20261114T023519566555Z-0f62a9ea-5514-43b2-893f-bc27f6e530cc',
    'ws8eoc-1114-hold': 'onair-20261114T023645733294Z-715111ea-1590-405d-a579-e73cb71ff177',
}


def preset():
    tools = TREE / 'tools'
    if not tools.is_dir():
        pytest.skip('tools/ not present (installed-wheel run)')
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import console
    from consolelib import modes, pactor
    args = console.parser_for(modes.resolve('pactor')).parse_args([
        'fly', 'OFFLINE', '--sense', '--dry-run', '--p4-mail', '--tx-drive', '1', '--p1-drive', '1'])
    args.client_sid = ''
    words, _ = pactor.flags(None, dict(p1_watts=None, p3_watts=None), args, Path('.'), None)
    resolved = onair.build_parser().parse_args(words)
    onair._connect_defaults(resolved)
    onair._p4_entry_defaults(resolved)
    return {k: v for k, v in vars(resolved).items() if k.startswith('p4_')
            and k not in ('p4_arm', 'p4_answer_residual_ms', 'p4_link_timeout')}


def declared(stack, stream):
    now = lambda: stream.sample_now() / FS
    turn_init = p4turn.P4Turn.__init__
    short_init = p4shortlevels.ShortDecode.__init__
    qam_init = p4shortlevels.QamShortDecode.__init__

    def turn(self, *args, **kwargs):
        turn_init(self, *args, **{**kwargs, 'wall_clock': now, 'managed_application': True})

    def short(self, *args, **kwargs):
        short_init(self, *args, **{**kwargs, 'now': now})

    def qam(self, *args, **kwargs):
        qam_init(self, *args, **{**kwargs, 'now': now})

    stack.enter_context(patch.object(p4longnative, 'backend', None))
    stack.enter_context(patch.object(p4longdecode, '_python_step_seconds', lambda *args: 0.))
    stack.enter_context(patch.object(p4longdecode, '_python_check_seconds', lambda _: 0.))
    stack.enter_context(patch.object(time, 'monotonic', now))
    stack.enter_context(patch.object(time, 'perf_counter', now))
    for module in (p4receiver, p4hold, p4shortlevels, p4turn):
        if hasattr(module, 'monotonic'):
            stack.enter_context(patch.object(module, 'monotonic', now))
    stack.enter_context(patch.object(p4turn.P4Turn, '__init__', turn))
    stack.enter_context(patch.object(p4shortlevels.ShortDecode, '__init__', short))
    stack.enter_context(patch.object(p4shortlevels.QamShortDecode, '__init__', qam))


def replay(name, **options):
    capture = RECORD / 'operator-data/captures' / CAPTURES[name]
    metadata = json.loads(require_input(capture / 'stream.json').read_text())
    recorded = json.loads(require_input(capture / 'p4-probe.json').read_text())
    assert metadata['samplerate'] == FS and metadata.get('first_stream_sample', 0) == 0
    segment = metadata['segments'][0] if 'segments' in metadata else 'stream.flac'
    with (capture / segment).open('rb') as source, wav.decoded(source) as decoded:
        rate, pcm = wavfile.read(decoded)
    assert rate == FS
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    audio = pcm.astype(np.float32)
    if np.issubdtype(pcm.dtype, np.integer):
        audio /= -np.iinfo(pcm.dtype).min
    assert len(audio) == metadata['samples']
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
    for key, value in {**preset(), **options}.items():
        setattr(tx.p4_link_options, key, value)
    tx.p4_link_options.p4_link_timeout = (len(audio) - stream.pos) / FS
    schedule, seen = [], set()
    for emission in recorded['emissions']:
        payload = emission.get('payload')
        if payload and payload not in seen:
            seen.add(payload)
            schedule.append([emission['audio_start'] - round(1.3 * FS), bytes.fromhex(payload)])
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
    for method, cost in (('read_partial', .003), ('read_window', .004)):
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
        stack.enter_context(patch.object(p4host, 'P4Host', lambda host, **kw: adapter(host, terminal=True)))
        onair._run_p4_link(tx, stream, onair._MasterGrid(opening['audio_start'] % 60000,
            60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    return tx.p4_probe


def test_ve3kpg_greeting_opens_chain_and_reads_long_arm():
    probe = replay('ve3kpg')
    keys = [e for e in probe.link_events if e['event'] == 'Keyed']
    greeting = [e for e in keys if 40 < e['at'] / FS < 41]
    assert len(greeting) == 1
    assert greeting[0]['word'] == 'X27' and greeting[0]['ack_identity'] is not None
    assert greeting[0]['purpose'] != 'repeat-request'
    assert any(h['kind'] == 'normal' and h.get('data') and 41 < h['onset_sample'] / FS < 43
               for h in probe.heard)
    assert sum(h['kind'] == 'long' and h.get('speedlevel') == 'SL4-long' and bool(h.get('data'))
               for h in probe.heard) >= 20


def test_ws8eoc_unread_normal_long_word_and_ledger_onsets():
    probe = replay('ws8eoc')
    keys = [e for e in probe.link_events if e['event'] == 'Keyed' and 24 < e['at'] / FS < 32]
    assert len(keys) == 3
    assert [e['word'] for e in keys] == ['T27'] * 3
    assert all(e['purpose'] == 'repeat-request' and e['ack_identity'] is None for e in keys)
    ledger = json.loads(require_input(Path(__file__).with_name('fixtures') /
                                      'p4-long-ledger-1111/frames.json').read_text())['frames']
    rows = [r for r in ledger if CAPTURES['ws8eoc'] in r['tape'] and r['form'] == 'NORMAL']
    assert rows
    for row in rows:
        nearest = min((h for h in probe.heard if h['kind'] == 'long' and h.get('data')),
                      key=lambda h: abs(h['onset_sample'] - row['onset_sample']))
        assert nearest['status'] == row['status']
        assert abs(nearest['onset_sample'] - row['onset_sample']) <= 48, row
