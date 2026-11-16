# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import wav
from hfmodem.shrike import arq, onair, p3frame, placement, rxfront, spec
from hfmodem.shrike.p3trial import CYCLE, ROTATION, ReplyClock
from hfmodem.tests import evidence
from hfmodem.tests.shrike.onair_support import _CallbackBench, _FakeCodec, _Rig, _Session


CAPTURES = (
    'onair-20261108T191306999322Z-0a2264a0-0a23-4e74-80a7-299a02e8965e',
    'onair-20261108T191546308486Z-1ee12558-dc82-4ac4-b5af-a3ef3c5c5131',
    'onair-20261108T193518175243Z-849e1e66-2259-41a9-8251-1c7629cced6b',
    'onair-20261109T001917618714Z-a5e0244e-332a-40d6-8094-63af19d8be8b',
)
FS = onair.FS
BASELINE = Path(__file__).with_name('p3_control_driver_peer_1111.json')


def control_runs(capture):
    cap = evidence.RECORD / 'operator-data' / 'captures' / capture
    identity_path = cap / 'capture-identity.json'
    evidence.require_input(identity_path)
    identity = json.loads(identity_path.read_text())
    log = (evidence.RECORD / 'operator-data' / 'working' / capture
           / identity['transcript']['path'])
    evidence.require_input(log)
    evidence.require_input(next(cap.glob('stream-*.flac'), cap / 'stream.wav'))
    frame = packet = own = None
    rows = []
    for line in log.read_text().splitlines():
        if match := re.search(r'\[p3\] frame @ (\d+) \((.*?)\), (short|long)', line):
            frame = (int(match[1]), match[2], match[3] == 'long')
        if match := re.search(r'\[traffic\] RX P3 SL(\d) DATA .*status=0x([0-9a-f]+)', line):
            if frame is not None and 'changeover' not in frame[1]:
                packet = (frame[0] - p3frame.DATA_OFFSET * rxfront.SPS,
                          int(match[1]), int(match[2], 16), frame[2])
        elif '[traffic] RX P3 CHANGEOVER' in line:
            packet = None
        if not line.startswith('tx_event '):
            continue
        event = json.loads(line[9:])
        if event.get('event') != 'completed' or event.get('outcome') != 'sent':
            continue
        label = event['label']
        if match := re.match(r'SL(\d) (?:LONG )?(?:pkt|ENTRY).*status=0x([0-9a-f]+)', label):
            own = (event['audio_start_sample'], int(match[1]), int(match[2], 16),
                   'LONG' in label)
            packet = None
        elif label.startswith('P3 BREAK-IN'):
            own = (event['audio_start_sample'], 1, 0, False)
            packet = None
        elif (match := re.match(r'CS([12456])\b', label)) and packet is not None and own is not None:
            phase, level, status, long_cycle = packet
            rows.append(dict(phase=phase, level=level, status=status, long=long_cycle,
                             word=int(match[1]) - 1, own=own, slot=event['slot'],
                             start=event['audio_start_sample'], hz=event.get('tx_offset_hz') or 0.0))
    assert rows, capture
    runs = []
    for row in rows:
        if (not runs or len(runs[-1]) == 6 or row['own'] != runs[-1][-1]['own']
                or row['long'] != runs[-1][-1]['long']
                or row['slot'] - runs[-1][-1]['slot'] != (3 if row['long'] else 1)):
            runs.append([])
        runs[-1].append(row)
    return cap, runs


def replay_controls(cap, rows, tmp_path, monkeypatch, request, *, option, waveform, realtime):
    origin = max(0, (rows[0]['phase'] - CYCLE) // 128 * 128)
    period = CYCLE * (3 if rows[0]['long'] else 1)
    stop = rows[-1]['start'] + 2 * period
    recorded_end = json.loads((cap / 'stream.json').read_text())['samples']
    audio = wav.read(cap / 'stream.wav', start=origin, stop=min(stop, recorded_end)).astype(np.float32)
    audio = np.pad(audio, (0, stop - origin - audio.size))
    s = _Session(role=arq.IRS)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.long_cycle = False
    s.host.arq._cycle_long = rows[0]['long']
    s.host.arq.dxcall = 'RECORDED'
    g = onair._MasterGrid(0, CYCLE, 8880, packet_n=38880,
                          cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(rows[0]['long'])
    own_start, own_level, own_status, own_long = rows[0]['own']
    # The transcript omits the packet's foot; use an explicit code-frame seed
    # so replay placement does not depend on estimating a pulse from the tape.
    own_lead = min(placement.pulse_lead(placement.link_packet(
        own_level, b'', own_status, swapped=foot, long_cycle=own_long))
        for foot in (False, True))
    own = own_start + own_lead - origin
    g._p3_yield_pulse = own + ROTATION
    answer = round((onair.P3_LONG_ANSWER_S if rows[0]['long'] else onair.P3_ANSWER_S) * FS)
    g.anchor = rows[0]['phase'] - origin + answer - 852 - CYCLE
    codec = (_FakeCodec(in_lat=788 / FS, out_lat=480 / FS, audio=audio)
             if realtime else None)
    live = codec.live if codec else _CallbackBench(seconds=0, holdback=916, lat_in=788)
    if codec:
        request.addfinalizer(codec.close)
        monkeypatch.setattr(live, 'close', codec.close)
        read = live.read

        def bounded_read(count):
            assert live.pos + count <= audio.size, 'driver replay exceeded its sample budget'
            return read(count)

        monkeypatch.setattr(live, 'read', bounded_read)
    else:
        live.LAT = live._lat = 1268
        live.audio, live.limit = audio, audio.size
    live.tx_latency_n, live.geometry_n = 960, 960
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path,
                       settle=.04, record_audio=False)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.sessrx, tx.p4_probe = live, s.rx, None
    tx.reply_clock = ReplyClock(entry_phase=own, pulse_epoch=own + ROTATION)
    tx.p3_control_instant = option
    tx.p3_control_waveform = waveform
    tx.p3_control_stagger = 'lead-12' if waveform == 'current' else 'off'
    tx.p3_follow_offset = 'all'
    tx.defer_p3_cs = True
    tx.aim(g, 1)
    live.flush_to(max(0, rows[0]['phase'] - origin - 2000))
    emissions = []
    cycle = s.rx.new_cycle
    tick = s.host.tick
    render = tx._render_p3_control

    def next_cycle():
        cycle()
        if len(emissions) >= len(rows):
            s.host.arq.on_host_abort()
            return
        row = rows[len(emissions)]
        path = (placement.LONG_PATHS if row['long'] else placement.SPEED_PATHS)[row['level']]
        width = rxfront._packet_span(rxfront._frame_span(path)) + p3frame.DATA_OFFSET * rxfront.SPS
        g.note_p3_packet(row['phase'] - origin, width, period, swapped=None,
                         identity=(False, row['level'], row['status'], b'recorded'))
        s.rx.p3_receive_offset_hz = row['hz']

    def owed(**kwargs):
        tick(**kwargs)
        if len(emissions) < len(rows):
            tx._pending_p3_cs = rows[len(emissions)]['word']

    def emitted(word, **kwargs):
        before = tx.n
        result = render(word, **kwargs)
        if tx.n != before:
            if codec:
                assert live.emitted == ('sent', tx.tx_audio_start, tx.tx_end)
            emissions.append(dict(slot=tx.slot, start=tx.tx_audio_start,
                                  pulse=tx.tx_audio_start + min(tx.tx_pulse_offsets),
                                  word=word, epoch=tx.reply_clock.pulse_epoch,
                                  own=g._p3_yield_pulse))
        return result

    monkeypatch.setattr(s.host.arq, 'on_host_connect', lambda *args: None)
    monkeypatch.setattr(onair, '_MasterGrid', lambda *args, **kwargs: g)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *args, **kwargs: None)
    monkeypatch.setattr(onair, '_read_ahead', lambda *args, **kwargs: None)
    monkeypatch.setattr(s.rx, 'new_cycle', next_cycle)
    monkeypatch.setattr(s.host, 'tick', owed)
    monkeypatch.setattr(tx, '_render_p3_control', emitted)
    for name in ('feed', 'deep_scan', 'upgrade_scan', 'flush', 'acquire_ahead'):
        monkeypatch.setattr(s.rx, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(s.rx, 'control_signal_in', lambda *args, **kwargs: (None, None))
    args = onair.build_parser().parse_args(['--mycall', 'W9SSJ', '--hold', '12'])
    identity = SimpleNamespace(cw_call='W9SSJ', closing_required=lambda *args, **kwargs: False)
    try:
        onair._drive_session(args, tx=tx, host=s.host, sessrx=s.rx, live=live,
            rig=None, dx='RECORDED', reply=None, outdir=tmp_path, identity=identity)
    finally:
        if codec:
            codec.close()
            assert not codec._thread.is_alive()
    return emissions


@pytest.mark.parametrize('capture', CAPTURES)
@pytest.mark.parametrize('waveform', ['historical', 'current'])
@pytest.mark.parametrize('option', ['peer', 'raster'])
@pytest.mark.parametrize('realtime', [False, pytest.param(True, marks=pytest.mark.wall_clock)],
                         ids=['arithmetic', 'live'])
def test_recorded_data_controls_keep_every_driver_slot(capture, waveform, option, realtime,
                                                      tmp_path, monkeypatch, request, record_property):
    cap, runs = control_runs(capture)
    baseline = json.loads(evidence.require_input(BASELINE).read_text())['scenarios'][capture][waveform]
    assert len(runs) == len(baseline)
    count = 0
    observed = []
    for index, rows in enumerate(runs):
        directory = tmp_path / str(index)
        directory.mkdir()
        with monkeypatch.context() as patch:
            emitted = replay_controls(cap, rows, directory, patch, request, option=option,
                                      waveform=waveform, realtime=realtime)
        assert len(emitted) == len(rows), (capture, index)
        assert [row['word'] for row in emitted] == [row['word'] for row in rows]
        step = 3 if rows[0]['long'] else 1
        assert [row['slot'] for row in emitted] == list(range(1, 1 + len(rows) * step, step))
        actual = [[r['slot'], r['start'], r['pulse'], r['epoch']] for r in emitted]
        if option == 'peer' or rows[0]['long']:
            assert actual == baseline[index]
        else:
            assert [r['epoch'] for r in emitted] == [r[3] for r in baseline[index]]
            for row in emitted:
                error = (row['pulse'] - row['own'] + CYCLE // 2) % CYCLE - CYCLE // 2
                assert abs(error) <= FS // 1000, (capture, index, row, error)
        observed.append(emitted)
        count += len(emitted)
    assert count == sum(map(len, runs))
    record_property('emissions', json.dumps(observed))
