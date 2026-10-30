#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline Kestrel/Xara interop. Run from any directory; no radio or sound I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import queue
import threading
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
import traceback
import wave
from unittest.mock import patch
from collections import deque
from contextlib import ExitStack
from functools import lru_cache

from paths import (HERE, ROOT, XARA, PROBE, PIN, XARA_URL, git, sha256, selection,
                   validate_probe, validate_kestrel_imports, git_metadata, source_identity)
validate_kestrel_imports()

import numpy as np
from channel import Channel, Condition, calibrate, RATE
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.tx import varahf500_tx as tx500, varahf2300_tx as txwide

HARNESS_HASHES = {name: sha256(HERE / name) for name in
                  ('run.py', 'paths.py', 'cli.py', 'prepare.py', 'check.sh',
                   'checks.py', 'checks_ipc.py', 'channel.py', 'src/main.rs', 'build.rs', 'Cargo.toml.in', 'Cargo.lock')}


def invoke(*args):
    validate_probe()
    return subprocess.run([str(PROBE), *map(str, args)], check=True,
                          capture_output=True, text=True, timeout=60).stdout


def provenance():
    chosen = selection()
    metadata = git_metadata(ROOT)
    identity = source_identity(ROOT)
    return {'xara_commit': git(XARA, 'rev-parse', 'HEAD'),
            'xara_requested_ref': chosen.get('requested_ref', PIN),
            'xara_resolved_commit': git(XARA, 'rev-parse', 'HEAD'),
            'xara_source_url': chosen.get('source_url', XARA_URL),
            'xara_dirty': git(XARA, 'status', '--porcelain'),
            'kestrel_source_root': str(ROOT),
            'kestrel_source_kind': 'git-checkout' if metadata is not None else 'archive',
            'kestrel_commit': metadata['commit'] if metadata is not None else None,
            'kestrel_tracked_changes': metadata['tracked_changes'] if metadata is not None else None,
            'kestrel_status': metadata['status'] if metadata is not None else None,
            'kestrel_source_identity': identity,
            'harness_root': str(HERE), 'xara_source_root': str(XARA),
            'platform': sys.platform, 'python': sys.version,
            'process_nice': os.getpriority(os.PRIO_PROCESS, 0),
            'harness_sha256': HARNESS_HASHES,
            'argv': sys.argv,
            'thread_caps': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                             'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')},
            'probe_sha256': hashlib.sha256(PROBE.read_bytes()).hexdigest()}


def render(frames, bw, level):
    lead = txwide.VB6Rnd()
    if level > 4:
        return phy.render_level([f[:-2] for f in frames], str(bw), level, lead)
    if bw == 500:
        if level == 4:
            return tx500.synth_burst(frames, lead=lead)
        # This API takes host levels; wideband synth_frame takes record keys.
        return tx500.synth_l3_burst(frames, level=level)
    assert len(frames) == 1
    return txwide.synth_frame(frames[0], level=phy.level_key(str(bw), level), over=lead)


def channel(x):
    x = np.asarray(x, dtype=float)
    # Scale both renderers to the same useful RMS; retain waveform shape.
    x = x * (0.12 / np.sqrt(np.mean(x*x)))
    return np.pad(x, (24000, 24000)).astype('<f8')


def phy_channel(x, args, bw, direction):
    if args.condition.snr == 'clean':
        return channel(x), {'condition': asdict(args.condition)}
    audio, calibration = calibrate(x, args.condition.rms)
    horizon = getattr(args, 'horizon', min(3600, len(audio)/RATE+5))
    wire = Channel(args.condition, seed=args.seed, bw=bw, caller='kestrel', replicate=args.replicate,
                   direction=direction, horizon=horizon)
    wire.register(RATE+calibration['first_nonzero'], calibration['samples'], 'phy')
    tail = wire.delay+sum(v['taps'] for v in wire.filter_info.values())+RATE//2
    padded = np.pad(audio, (RATE, tail))
    out = wire.process(padded, 0)
    return out, dict(calibration=calibration, channel=wire.evidence())


def phy_cases(args):
    """CRC-clean arbitrary frames, exact bytes both ways, every normal HF rung."""
    rows = []
    for line in invoke('inventory').splitlines():
        bw, level, size, bps, waveform = line.split(',', 4)
        bw, level, size = int(bw), int(level), int(size)
        if args.bw and bw not in args.bw or args.level and level not in args.level:
            continue
        counts = [1, 2] if args.pairs and 'Ofdm' in waveform and (
            (bw == 500 and level >= 12) or (bw == 2300 and level >= 15) or (bw == 2750 and level >= 16)) else [1]
        for count in counts:
            frames = []
            for j in range(count):
                body = bytes((i*73 + 19 + 37*j) % 256 for i in range(size-3)) + b'\x82'
                frames.append(body + crc16_genibus(body).to_bytes(2, 'big'))
            tag = f'bw{bw}-l{level:02d}-n{count}'
            directory = args.out / tag
            directory.mkdir(parents=True, exist_ok=True)
            row = {'bw': bw, 'level': level, 'frame_bytes': size, 'frames': count,
                   'bps': float(bps), 'waveform': waveform, 'results': {}}
            for direction in ['kestrel-to-xara', 'xara-to-kestrel']:
                start = time.monotonic()
                try:
                    pcm = directory / f'{direction}.f64'
                    if direction == 'kestrel-to-xara':
                        x, evidence = phy_channel(render(frames, bw, level), args, bw, 'A->B')
                        x.astype('<f8').tofile(pcm)
                        got = [bytes.fromhex(h) for h in invoke('rx', bw, level, pcm, count).splitlines()]
                        want = frames
                    else:
                        invoke('tx', bw, level, ','.join(f.hex() for f in frames), pcm)
                        x, evidence = phy_channel(np.fromfile(pcm, dtype='<f8'), args, bw, 'B->A')
                        x.tofile(pcm)
                        read = phy.decode_level(x, str(bw), level, expect_pair=count == 2)
                        got = list(read.frames) if read else []
                        want = [f[:-2] for f in frames]
                    ok = got == want
                    detail = {'ok': ok, 'expected': [f.hex() for f in want],
                              'decoded': [f.hex() for f in got],
                              'evidence': evidence,
                              'pcm_sha256': hashlib.sha256(pcm.read_bytes()).hexdigest()}
                    corrupt = len(got) > len(want) or any(f != want[i] for i, f in enumerate(got[:len(want)]))
                    detail['outcome'] = 'corruption' if corrupt else 'pass' if ok else 'frame-miss'
                    detail['corruption_receiver'] = ('b' if direction == 'kestrel-to-xara' else 'a') if corrupt else None
                    native = directory/f'{direction}-control.f64'
                    invoke('tx', bw, level, ','.join(f.hex() for f in frames), native)
                    control, control_evidence = phy_channel(np.fromfile(native, '<f8'), args, bw,
                        'A->B' if direction == 'kestrel-to-xara' else 'B->A')
                    control.astype('<f8').tofile(native)
                    baseline = [bytes.fromhex(h) for h in invoke('rx', bw, level, native, count).splitlines()]
                    detail['control'] = dict(ok=baseline == frames, decoded=[f.hex() for f in baseline],
                                             evidence=control_evidence)
                    bad_control = len(baseline) > len(frames) or any(f != frames[i] for i, f in enumerate(baseline[:len(frames)]))
                    detail['control'].update(outcome='corruption' if bad_control else 'pass' if baseline == frames else 'frame-miss',
                        corruption_receiver='b' if bad_control else None)
                    detail['comparison'] = paired(detail, detail['control'])
                    if not args.keep_audio and baseline == frames:
                        native.unlink()
                    if not args.keep_audio and ok:
                        pcm.unlink()
                except Exception as error:
                    detail = {'ok': False, 'error': repr(error)}
                detail['elapsed_seconds'] = time.monotonic() - start
                row['results'][direction] = detail
                print(f'{tag} {direction}: {"PASS" if detail["ok"] else "FAIL"}', flush=True)
            rows.append(row)
            (directory / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
    assert rows, 'no selected cases'
    result = {'provenance': args.provenance, 'kind': 'phy', 'cases': rows,
              'ok': all(v['ok'] for r in rows for v in r['results'].values())}
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.out / 'summary.txt').write_text('\n'.join(
        f'{r["bw"]} L{r["level"]} {d} ok={v["ok"]} baseline={v.get("control", {}).get("ok")}'
        for r in rows for d, v in r['results'].items())+'\n')
    return result['ok']


def read_exact(pipe, n, timeout=30, *, deadline=None):
    end, chunks = deadline if deadline is not None else time.monotonic() + timeout, []
    while n:
        left = end - time.monotonic()
        if left <= 0 or not select.select([pipe], [], [], max(0, left))[0]:
            raise TimeoutError('Xara adapter did not reply')
        chunk = os.read(pipe.fileno(), n)
        if not chunk:
            raise EOFError('Xara adapter exited mid-response')
        chunks.append(chunk)
        n -= len(chunk)
    return b''.join(chunks)


def write_all(pipe, blob, timeout=30, *, deadline=None):
    """Bound the entire write, including a peer that stops consuming stdin."""
    end = deadline if deadline is not None else time.monotonic() + timeout
    fd = pipe.fileno()
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        remaining = memoryview(blob)
        while remaining:
            left = end - time.monotonic()
            if left <= 0 or not select.select([], [pipe], [], max(0, left))[1]:
                raise TimeoutError('Xara adapter did not consume its command')
            try:
                written = os.write(fd, remaining)
            except BlockingIOError:
                continue
            if not written:
                raise EOFError('Xara adapter stopped consuming its command')
            remaining = remaining[written:]
    finally:
        os.set_blocking(fd, blocking)


class Xara:
    def __init__(self, bw, registered, comp, trace, call='W1AW', remote='W9SSJ', *, timeout=30):
        validate_probe()
        self.process = subprocess.Popen([str(PROBE), 'station', str(bw), call,
                                         remote, str(registered).lower(), comp],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        bufsize=0)
        self.trace, self.got, self.events, self.state = trace, bytearray(), [], ''
        self.n = 0
        self.timeout, self._closed = timeout, False

    def command(self, line, samples=None):
        if self._closed:
            raise ValueError('Xara adapter is closed')
        deadline = time.monotonic() + self.timeout
        try:
            write_all(self.process.stdin, line.encode() + b'\n', deadline=deadline)
            if samples is not None:
                write_all(self.process.stdin, np.asarray(samples, dtype='<f8').tobytes(),
                          deadline=deadline)
                self.n += len(samples)
            size, = struct.unpack('<I', read_exact(self.process.stdout, 4, deadline=deadline))
            if size >= 1_000_000:
                raise ValueError('Xara adapter returned an oversized event response')
            for event in read_exact(self.process.stdout, size, deadline=deadline).decode().splitlines():
                if event.startswith('STATE:'):
                    self.state = event[6:]
                else:
                    self.events.append(event)
                    self.trace('xara', event)
                    if event.startswith('DATA:'):
                        self.got.extend(bytes.fromhex(event[5:]))
            if samples is not None:
                return np.frombuffer(read_exact(self.process.stdout, len(samples)*8,
                                                deadline=deadline), dtype='<f8').copy()
        except BaseException:
            self.close()
            raise

    def step(self, samples):
        return self.command(f'STEP {len(samples)}', samples)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            try:
                self.process.stdin.close()
            except OSError:
                pass  # A dead child's broken input must not bypass reaping it.
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.process.stdout.close()


class Timeline:
    """Schedule synchronous Kestrel TX without advancing the sample channel."""
    def __init__(self):
        self.n = self.blocked_until = 0
        self.tx = deque()

    def clock(self):
        return 1000 + max(self.n, self.blocked_until) / 48000

    def emit(self, samples):
        start = max(self.n, self.blocked_until)
        x = np.asarray(samples, float).copy()
        self.blocked_until = start + len(x)
        self.tx.append((start, x))

    def output(self, start, n):
        out = np.zeros(n)
        while self.tx:
            at, x = self.tx[0]
            if at >= start + n:
                break
            lo, hi = max(start, at), min(start+n, at+len(x))
            if hi > lo:
                out[lo-start:hi-start] += x[lo-at:hi-at]
            if at+len(x) > start+n:
                break
            self.tx.popleft()
        return out


@lru_cache(maxsize=1)
def inventory():
    return [dict(zip(('bw', 'level', 'size', 'bps', 'waveform'),
                     (int(b), int(l), int(s), float(p), w)))
            for b, l, s, p, w in (line.split(',', 4) for line in invoke('inventory').splitlines())]


def payloads(size):
    return tuple((text*((size+len(text)-1)//len(text)))[:size] for text in
                 (b'Hello Xara from Kestrel. 0123456789\r\n', b'Hello Kestrel from Xara. abcdefghij\r\n'))


class DeliveryError(Exception):
    pass


def check_prefix(actual, expected):
    if len(actual) > len(expected) or not expected.startswith(bytes(actual)):
        raise DeliveryError('corrupt or duplicate application delivery')


def structured(event):
    match = re.search(r'Bitrate \{ level: (\d+), bps: (\d+), tx: (true|false) \}', event)
    if match:
        return dict(version=1, kind='level', level=int(match[1]), bps=int(match[2]), tx=match[3] == 'true')
    match = re.search(r'tx L(\d+)x(\d+)', event)
    if match:
        return dict(version=1, kind='data-render', level=int(match[1]), frames=int(match[2]))
    match = re.search(r'tx c(\d+) f(\d+)(\+?)', event)
    if match:
        code = int(match[1])
        extra = (9 if 26 <= code <= 40 else 7) if match[3] else 0
        return dict(version=1, kind='control-render', code=code, fmt=int(match[2]), extra=extra)
    match = re.search(r'rx .*?(?:at )?level (\d+)', event)
    if match and 'failed' not in event and 'connect-response' not in event:
        return dict(version=1, kind='level', level=int(match[1]), tx=False)
    if event.startswith('retry DATA over'):
        return dict(version=1, kind='data-retry')
    if event.startswith('DATA:'):
        return dict(version=1, kind='delivery', bytes=len(event[5:])//2)
    return dict(version=1, kind='raw')


class Metrics:
    def __init__(self):
        self.levels = {'tx': [], 'rx': []}
        self.data_retries = 0
        self.events = []

    def add(self, event):
        item = structured(event)
        self.events.append(item)
        if item['kind'] == 'level':
            self.levels['tx' if item['tx'] else 'rx'].append(item['level'])
        if item['kind'] == 'data-retry':
            self.data_retries += 1
        return item

    def result(self, kestrel=False):
        return dict(levels={side: dict(highest=max(values) if values else None,
            final=values[-1] if values else None, reason=None if values else 'no data level observed')
            for side, values in self.levels.items()},
            data_retries=self.data_retries if kestrel else None, control_retries=None,
            retry_reason='control repeats lack unique frame identity; pinned Xara exposes no retry identity' if not kestrel
                         else 'data retry trace counted; control identity unavailable')


class Calibration:
    def __init__(self, bw, condition, directory):
        self.bw, self.condition, self.directory = bw, condition, directory
        self.table, self.bursts, self.pending = {}, [], deque()
        self.active = None
        self.completed = False

    def entry(self, kind, level, count=1, extra=0):
        key = (kind, level, count, extra)
        if key not in self.table:
            path = self.directory / 'calibration.f64'
            if kind == 'data':
                size = next(r['size'] for r in inventory() if r['bw'] == self.bw and r['level'] == level)
                body = bytes((i*73+19) % 256 for i in range(size-3))+b'\x82'
                frame = body+crc16_genibus(body).to_bytes(2, 'big')
                invoke('tx', self.bw, level, ','.join([frame.hex()]*count), path)
                audio = np.fromfile(path, '<f8')
                audio = np.rint(np.clip(audio, -1, 1)*32767)/32767
            else:
                invoke('tx-control', self.bw, count, level, extra, path)
                audio = np.fromfile(path, '<f8')
            path.unlink()
            _, detail = calibrate(audio, self.condition.rms)
            self.table[key] = dict(detail, total_samples=len(audio), kind=kind, level=level,
                                   frames=count if kind == 'data' else None, native_drive_db=-5)
        return self.table[key]

    def observe(self, event, at):
        if event == 'Ptt(false)':
            self.completed = True
            return
        item = structured(event)
        if item['kind'] == 'data-render':
            detail = self.entry('data', item['level'], item['frames'])
        elif item['kind'] == 'control-render':
            detail = self.entry('control', item['code'], item['fmt'], item['extra'])
        elif re.search(r'\btx\s', event):
            entries = [self.entry('data', r['level']) for r in inventory() if r['bw'] == self.bw]
            rms = float(np.mean([r['native_rms'] for r in entries]))
            detail = dict(native_rms=rms, gain=self.condition.rms/rms, calibrated_rms=self.condition.rms,
                          total_samples=None, first_nonzero=0,
                          kind='cw' if 'tx cw' in event else 'unknown', fallback='data-level mean')
        else:
            return
        self.pending.append(dict(detail, rendered_at=at))

    def process(self, samples, at):
        out = samples.copy()
        start = 0
        while start < len(out):
            if self.active is None:
                nz = np.flatnonzero(out[start:])
                if not len(nz):
                    break
                if not self.pending:
                    raise ValueError('uncalibrated Xara burst: no render trace before PCM')
                self.active = self.pending.popleft()
                start = max(start, start+int(nz[0])-self.active['first_nonzero'])
                self.active.update(start_sample=at+start, consumed=0)
                self.bursts.append(self.active)
            entry = self.active
            remaining = entry['total_samples']
            if remaining is None:
                end = len(out)
            else:
                end = min(len(out), start+remaining-entry['consumed'])
            if remaining is None:
                nz = np.flatnonzero(out[start:end])
                if len(nz):
                    entry['last_nonzero'] = entry['consumed']+int(nz[-1])
            lo = max(start, start+entry['first_nonzero']-entry['consumed'])
            hi = min(end, start+entry.get('last_nonzero', entry['consumed']+end-start-1)+1-entry['consumed'])
            hi = max(lo, hi)
            entry['native_energy'] = entry.get('native_energy', 0.)+float(out[lo:hi]@out[lo:hi])
            entry['native_count'] = (entry.get('last_nonzero', -1)+1 if remaining is None else
                                     entry.get('native_count', 0)+max(0, hi-lo))
            out[start:end] *= entry['gain']
            entry['consumed'] += end-start
            entry['realized_native_rms'] = float(np.sqrt(entry['native_energy']/max(1, entry['native_count'])))
            entry['realized_calibrated_rms'] = entry['realized_native_rms']*entry['gain']
            entry['end_sample'] = at+end
            start = end
            if remaining is not None and entry['consumed'] == remaining:
                self.active = None
        if self.completed and self.active is not None and self.active['total_samples'] is None:
            self.active.update(samples=self.active['native_count'], total_samples=self.active['consumed'])
            self.active = None
        self.completed = False
        return out


class ImmediateQueue(queue.Queue):
    def get(self, *args, **kwargs):
        return super().get(block=False)


from hfmodem.kestrel.host.station_runtime import StationTransport, CHUNK, ECHO_GUARD
from hfmodem.kestrel.monitor import Segmenter


class ReceivePath(StationTransport):
    def __init__(self, clock=lambda: 0):
        self.clock = clock
        self.contact = SimpleNamespace(receive_boundary=(0, 0))
        self.frames = ImmediateQueue()
        self.stop, self.fault = threading.Event(), threading.Event()
        self._rx_generation = self._skip_before = 0
        self._segmenter, self._bursts = Segmenter(), deque()
        self._hs = None
        self._freshness, self._channel_boundary = (0, 0), (0, 0)
        self._seen_flush = 0

    @property
    def link(self):
        return SimpleNamespace(station=SimpleNamespace(audio=SimpleNamespace(samples=self.clock())))


class Kestrel:
    def __init__(self, args, bw, trace, resources, *, replay=False):
        from hfmodem.kestrel.vara import vara_arq as arq
        from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
        from hfmodem.kestrel.host.server import HostSession
        self.t, self.got, self.messages = Timeline(), bytearray(), []
        self.clean = args.condition.snr == 'clean'
        self.replay = replay
        self.trace, self.calibration, self.metadata = trace, [], {}
        self.history, self.burst, self.quiet = deque(maxlen=50), [], 0
        self.pieces, self.piece_at = [], 0
        self.rx = ReceivePath(lambda: self.t.n)
        self.replay_acc, self.replay_on = [], False
        owner = self

        class Transport(arq.VaraIO):
            @property
            def receive_epoch(self):
                return owner.rx.receive_epoch
            @property
            def hs(self):
                return owner.rx.hs
            @hs.setter
            def hs(self, value):
                owner.rx.hs = value
            @property
            def receiving(self):
                if replay:
                    return bool(owner.replay_acc)
                return bool(owner.burst) if owner.clean else owner.rx.receiving
            def tx_metadata(self, **details):
                owner.metadata = {k: v if v is None or isinstance(v, (str, int, float, bool)) else str(v)
                                  for k, v in details.items()}
                level = details.get('speed_level')
                if level is not None and details.get('packet_kind') == 'data':
                    trace('a', f'Bitrate {{ level: {phy.host_level(bw, level)}, bps: 0, tx: true }}')
            def tx(self, samples):
                samples = np.asarray(samples, float)
                if owner.clean:
                    rms = np.sqrt(np.mean(samples*samples))
                    audio = samples*(.12/rms) if rms else samples
                else:
                    audio, detail = calibrate(samples, args.condition.rms)
                    owner.calibration.append(dict(detail, metadata=owner.metadata,
                        start_sample=max(owner.t.n, owner.t.blocked_until)))
                owner.t.emit(audio)
                if not owner.clean or replay:
                    owner.rx._reset_rx(owner.t.n)
                    owner.rx._skip_before = max(owner.rx._skip_before, owner.t.blocked_until+ECHO_GUARD)
            def log(self, text): trace('a', text)
            def next_rx_burst(self, timeout, hs):
                if owner.clean or replay:
                    return None
                burst = owner.rx.next_rx_burst(timeout=timeout, hs=hs)
                if burst is not None:
                    trace('a', f'Segmenter burst samples={len(burst)} at={owner.t.n}')
                return burst

        class Sink:
            def __init__(self, data=False): self.data = data
            def sendall(self, blob):
                if self.data:
                    owner.got.extend(blob)
                    trace('a', 'DATA:'+blob.hex())
                else:
                    for line in blob.decode('ascii').split('\r'):
                        if line:
                            owner.messages.append(line)
                            trace('a', line)

        self.wire = Transport()
        self.runtime = AudioHostRuntime(self.wire, high_speed=args.registered, clock=self.t.clock)
        self.modem = self.runtime.modem_factory()
        self.host = HostSession(Sink(), Sink(True), self.modem)
        resources.callback(self.modem.stop)
        self.modem.start(self.host)
        self.rx.hs = self.modem.station
        self.arq = arq
        for line in ['MYCALL W9SSJ', f'BW{bw}', f'COMPRESSION {args.compression}', 'LISTEN ON']:
            self.host._dispatch(line)
            self.modem.poll()

    @property
    def connected(self):
        return any(s.startswith('CONNECTED ') for s in self.messages)

    @property
    def disconnected(self):
        return 'DISCONNECTED' in self.messages

    def command(self, line):
        if line.startswith('SEND '):
            self.modem.transmit(bytes.fromhex(line[5:]))
        else:
            self.host._dispatch('CONNECT W9SSJ W1AW' if line == 'CONNECT' else line)
            self.modem.poll()

    def step(self, heard, at, replay_on=False):
        t = self.t
        t.n = at+len(heard)
        self.t.output(at, len(heard))
        if self.clean and not self.replay:
            if t.n < t.blocked_until:
                self.history.clear()
                self.burst, self.quiet = [], 0
                return
            emitting = bool(np.max(np.abs(heard)) > .001)
            if emitting:
                if not self.burst:
                    self.burst = list(self.history)
                self.quiet = 0
            if self.burst:
                self.burst.append(heard.copy())
                self.quiet = 0 if emitting else self.quiet+len(heard)
            self.history.append(heard.copy())
            self.modem.station.on_rx_stream(heard)
            if self.burst and self.quiet >= 12000 and t.n >= t.blocked_until:
                self.modem.station.on_rx_audio(np.concatenate(self.burst))
                self.burst, self.quiet = [], 0
            if t.n >= t.blocked_until:
                self.runtime.step(0)
            return
        self.pieces.append(np.asarray(heard, np.float32))
        if sum(map(len, self.pieces)) < CHUNK:
            return
        samples = np.concatenate(self.pieces)
        self.pieces.clear()
        start = t.n-len(samples)
        if self.replay:
            if t.n < t.blocked_until:
                self.replay_acc.clear()
                return
            samples = samples[max(0, self.rx._skip_before-start):]
            if not len(samples):
                self.replay_acc.clear()
                return
            generation = self.rx._rx_generation
            self.modem.station.on_rx_stream(samples)
            if generation != self.rx._rx_generation:
                self.replay_acc.clear()
                return
            if replay_on:
                self.replay_acc.append(samples)
            elif self.replay_acc:
                self.modem.station.on_rx_audio(np.concatenate(self.replay_acc))
                self.replay_acc.clear()
            if t.n >= t.blocked_until:
                self.runtime.step(0)
        else:
            self.rx.feed(start, samples)
            generation = self.rx._rx_generation
            while True:
                before = self.rx.receiving
                self.runtime.step(1)
                if self.rx.receiving != before:
                    self.trace('a', f'Segmenter inb={self.rx.receiving} at={start}')
                if generation != self.rx._rx_generation or (self.rx.frames.empty() and not self.rx._bursts):
                    break


def endpoint_connected(endpoint):
    return endpoint.connected if isinstance(endpoint, Kestrel) else any(e.startswith('Host(Connected ') for e in endpoint.events)


def endpoint_disconnected(endpoint):
    return endpoint.disconnected if isinstance(endpoint, Kestrel) else 'Host(Disconnected)' in endpoint.events


def classify(error, phase, connected, corruption):
    if corruption:
        return 'corruption'
    if error is None:
        return 'pass'
    if not isinstance(error, DeliveryError):
        return 'invalid'
    if 'disconnected' in str(error):
        return 'early-disconnect'
    if phase == 'disconnect':
        return 'missing-orderly-close'
    return 'transfer-timeout' if connected else 'no-connect'


def case_horizon(args, bw):
    min_bps = min((r for r in inventory() if r['bw'] == bw), key=lambda r: r['level'])['bps']
    transfer = 60+3*8*args.bytes/min_bps
    return transfer, (100+7500+2*(int(np.ceil(transfer*100))+round(args.application_gap*100))+500+6000+500)/100


def run_case(args, bw, caller, *, control=False):
    tag = f'bw{bw}-{caller}-{"registered" if args.registered else "limited"}'
    directory = args.out/tag
    directory.mkdir(parents=True, exist_ok=True)
    transfer, horizon = case_horizon(args, bw)
    horizon = getattr(args, 'horizon', horizon)
    pa, pb = payloads(args.bytes)
    now, phase, error, corruption = 0, 'connect', None, None
    metrics = {side: Metrics() for side in ('a', 'b')}
    calibrators = {side: Calibration(bw, args.condition, directory) for side in ('a', 'b')}
    channels = [Channel(args.condition, seed=args.seed, bw=bw, caller=caller, replicate=args.replicate,
                        direction=d, horizon=horizon) for d in ('A->B', 'B->A')]
    phases, commands, ptt, intervals = {}, [], False, []
    primary_start = time.monotonic()
    row = dict(bw=bw, caller=caller, registered=args.registered, compression=args.compression,
               condition=asdict(args.condition), seed=args.seed, replicate=args.replicate,
               control=control, ok=False, phases=phases, trace=str(directory/'trace.jsonl'))
    registered_bursts = [{}, {}]
    with ExitStack() as resources:
        log = resources.enter_context((directory/'trace.jsonl').open('w'))
        capture = resources.enter_context((directory/'replay.f32').open('wb')) if not control else None
        recordings = []
        if args.record_audio:
            for name in ('a-to-b', 'b-to-a'):
                wav = resources.enter_context(wave.open(str(directory/f'{name}.wav'), 'wb'))
                wav.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
                recordings.append(wav)
        def trace(side, event):
            nonlocal ptt
            item = metrics[side].add(event)
            log.write(json.dumps(dict(sample=now, seconds=now/RATE, side=side, event=event,
                structured=item, driver_seconds=a.t.clock()-1000 if side == 'a' and isinstance(a, Kestrel) else now/RATE))+'\n')
            if (control or side == 'b') and args.condition.snr != 'clean':
                calibrators[side].observe(event, now)
            if side == 'b' and event == 'Ptt(true)':
                intervals.append([now, None])
                ptt = True
            if side == 'b' and event == 'Ptt(false)':
                if intervals:
                    intervals[-1][1] = now+480
                ptt = False
        a = None
        try:
            if control:
                a = Xara(bw, args.registered, args.compression, lambda _, e: trace('a', e), 'W9SSJ', 'W1AW')
                resources.callback(a.close)
            else:
                a = Kestrel(args, bw, trace, resources)
            b = Xara(bw, args.registered, args.compression, lambda _, e: trace('b', e))
            resources.callback(b.close)

            def command(endpoint, line):
                if endpoint is a:
                    commands.append((now, line))
                endpoint.command(line)

            def register(bursts, index):
                wire, seen = channels[index], registered_bursts[index]
                for i, entry in enumerate(bursts):
                    if i not in seen:
                        seen[i] = wire.register(entry['start_sample']+entry.get('first_nonzero', 0),
                                                entry.get('samples'), 'ab'[index])
                    elif seen[i]['count'] is None and entry.get('samples') is not None:
                        wire.finish(seen[i], entry['samples'])

            def step():
                nonlocal now, corruption
                heard_a, heard_b = channels[1].next_block(), channels[0].next_block()
                if control:
                    oa = a.step(heard_a)
                    if args.condition.snr != 'clean':
                        oa = calibrators['a'].process(oa, now)
                else:
                    oa = a.t.output(now, 480)
                    if args.condition.snr != 'clean':
                        capture.write(np.asarray(heard_a, '<f4').tobytes())
                        a.step(heard_a, now)
                bursts = a.calibration if isinstance(a, Kestrel) else calibrators['a'].bursts
                register(bursts, 0)
                channels[0].process(oa, now)
                ob = b.step(heard_b)
                if args.condition.snr != 'clean':
                    ob = calibrators['b'].process(ob, now)
                register(calibrators['b'].bursts, 1)
                channels[1].process(ob, now)
                for wav, pcm in zip(recordings, (oa, ob)):
                    wav.writeframesraw((np.clip(pcm, -1, 1)*32767).round().astype('<i2').tobytes())
                if not control and args.condition.snr == 'clean':
                    capture.write(np.asarray(heard_a, '<f4').tobytes())
                    a.step(heard_a, now)
                now += 480
                for endpoint, expected, receiver in ((b, pa, 'b'), (a, pb, 'a')):
                    try:
                        check_prefix(endpoint.got, expected)
                    except DeliveryError:
                        corruption = receiver
                        raise

            def wait(done, name, seconds):
                nonlocal phase
                phase = name
                end = now+round(seconds*RATE)
                while now < end:
                    if done():
                        phases[name] = now/RATE
                        return
                    if name not in ('connect', 'disconnect', 'post-close') and (
                            endpoint_disconnected(a) or endpoint_disconnected(b)):
                        raise DeliveryError('cleanly disconnected before full delivery')
                    step()
                raise DeliveryError(f'{name} deadline')

            driver = patch.object(a.arq, 'time', SimpleNamespace(monotonic=a.t.clock)) if not control else ExitStack()
            with driver:
                for _ in range(100): step()
                phases['connect-submitted'] = now/RATE
                command(a if caller == 'kestrel' else b, 'CONNECT')
                wait(lambda: endpoint_connected(a) and endpoint_connected(b), 'connect', 75)
                for _ in range(round(args.application_gap*100)): step()
                phases['a-send'] = now/RATE
                command(a, 'SEND '+pa.hex())
                wait(lambda: bytes(b.got) == pa, 'a-to-b', transfer)
                for _ in range(round(args.application_gap*100)): step()
                phases['b-send'] = now/RATE
                command(b, 'SEND '+pb.hex())
                wait(lambda: bytes(a.got) == pb, 'b-to-a', transfer)
                for _ in range(500): step()
                command(a, 'DISCONNECT')
                wait(lambda: endpoint_disconnected(a) and endpoint_disconnected(b), 'disconnect', 60)
                phase = 'post-close'
                for _ in range(500): step()
                row['ok'] = True
        except Exception as exc:
            error = exc
            row['error'] = repr(exc)
            (directory/'exception.txt').write_text(traceback.format_exc())
        row['outcome'] = classify(error, phase, a is not None and endpoint_connected(a), corruption)
        row['corruption_receiver'] = corruption
        row['corruption_implementation'] = ('Kestrel' if corruption == 'a' and not control else 'Xara') if corruption else None
        row['corruption_direction'] = ('B->A' if corruption == 'a' else 'A->B') if corruption else None
        row['failed_phase'] = phase if error is not None else None
        row['received'] = {}
        for side, endpoint, want in (('a', a, pb), ('b', locals().get('b'), pa)):
            got = bytes(endpoint.got) if endpoint is not None else b''
            row['received'][side] = dict(expected_length=len(want), received_length=len(got),
                expected_sha256=hashlib.sha256(want).hexdigest(), received_sha256=hashlib.sha256(got).hexdigest())
        if capture and now:
            capture.flush()
            try:
                row['ptt-replay'] = ptt_replay(args, bw, directory, commands, intervals, now,
                                                   channels[1], pb, row, a)
            except Exception as exc:
                row['ptt-replay'] = dict(outcome='invalid', error=repr(exc))
                row['outcome'], row['ok'] = 'invalid', False
        elif capture:
            row['ptt-replay'] = dict(outcome='invalid', reason='case did not start')
        row['metrics'] = {side: value.result(not control and side == 'a') for side, value in metrics.items()}
        row['calibration'] = dict(a=a.calibration if isinstance(a, Kestrel) else calibrators['a'].bursts,
                                  b=calibrators['b'].bursts)
        row['channels'] = [c.evidence() for c in channels]
        if isinstance(a, Kestrel):
            row['driver_seconds'] = a.t.clock()-1000
    if not control:
        (directory/'replay.f32').unlink(missing_ok=True)
    row.update(simulated_seconds=now/RATE, elapsed_seconds=time.monotonic()-primary_start,
               transfer_deadline_seconds=transfer)
    row['timing'] = {}
    if 'connect-submitted' in phases:
        row['timing']['connect_seconds'] = phases.get('connect', now/RATE)-phases['connect-submitted']
    for direction, start in (('a-to-b', 'a-send'), ('b-to-a', 'b-send')):
        duration = phases.get(direction, now/RATE)-phases[start] if start in phases else None
        delivered = row['received']['b' if direction == 'a-to-b' else 'a']['received_length']
        row['timing'][direction] = dict(seconds=duration, delivered_bytes=delivered,
            goodput_bps=delivered*8/duration if duration else None)
    (directory/'result.json').write_text(json.dumps(row, indent=2)+'\n')
    print(f'{tag} {"control" if control else "session"}: {row["outcome"]}', flush=True)
    return row


def recovered_phases(primary, replay, size):
    return [phase for phase, recovered in (
        ('connect', not primary['connected'] and replay['connected']),
        ('a-to-b', primary['confirmed'] < size <= replay['confirmed']),
        ('b-to-a', not primary['received'] and replay['received']),
        ('disconnect', not primary['disconnected'] and replay['connected'] and replay['disconnected']
         and replay['close_received'])) if recovered]


def ptt_replay(args, bw, directory, commands, intervals, end, wire, expected, row, primary):
    log = (directory/'ptt-replay-trace.jsonl').open('w')
    now, command_index = 0, 0
    close_requested, close_handover, close_received = False, False, False
    def state(endpoint, close_received=False):
        return dict(connected=endpoint.connected, disconnected=endpoint.disconnected,
                    received=bytes(endpoint.got) == expected, confirmed=endpoint.modem.confirmed_total,
                    close_received=close_received)
    primary_state = state(primary)
    with ExitStack() as resources:
        resources.callback(log.close)
        def trace(side, event):
            nonlocal close_handover, close_received
            if close_requested:
                if event.startswith(('rx turn grant — ', 'rx session-turn-release-responder — ',
                                     'rx session-drained-responder — ', 'rx caller handover — ',
                                     'turn granted (peer answered with ')):
                    close_handover = True
                if (close_handover and event.startswith('close keyed to ') or
                        event.startswith(('rx disconnect-request — session closed',
                                          'rx session-disconnect-request-responder — ',
                                          'rx session-disconnect-request-responder-short — ',
                                          'rx receiver-disconnect — ')) or
                        event.endswith("'s CW ident — it has closed the link")):
                    close_received = True
            log.write(json.dumps(dict(sample=now, side=side, event=event))+'\n')
        endpoint = Kestrel(args, bw, trace, resources, replay=True)
        audio = np.memmap(directory/'replay.f32', '<f4', mode='r')
        support = 4800
        windows = [wire.window(lo, (hi if hi is not None else end)-lo, output=True) for lo, hi in intervals]
        windows = [(max(0, lo-support), hi+support) for lo, hi in windows]
        phases = {}
        with patch.object(endpoint.arq, 'time', SimpleNamespace(monotonic=endpoint.t.clock)):
            for now in range(0, end, 480):
                while command_index < len(commands) and commands[command_index][0] <= now:
                    if commands[command_index][1] == 'DISCONNECT':
                        close_requested = True
                    endpoint.command(commands[command_index][1])
                    command_index += 1
                on = any(lo < now+480 and hi > now for lo, hi in windows)
                endpoint.step(audio[now:now+480], now, replay_on=on)
                check_prefix(endpoint.got, expected)
                current = state(endpoint)
                for phase, done in (('connect', current['connected']), ('a-to-b', current['confirmed'] >= args.bytes),
                                    ('b-to-a', current['received']), ('disconnect', current['disconnected'])):
                    if done:
                        phases.setdefault(phase, (now+480)/RATE)
        got = bytes(endpoint.got)
        replay_state = state(endpoint, close_received)
        recovered = recovered_phases(primary_state, replay_state, args.bytes)
        del audio
        return dict(outcome='recovered' if row['failed_phase'] in recovered else 'incomplete' if row['failed_phase'] else 'observed',
                    recovered_phases=recovered, phases=phases, primary=primary_state, replay=replay_state,
                    received_length=len(got), expected_length=len(expected), windows=windows,
                    trace=str(directory/'ptt-replay-trace.jsonl'),
                    meaning='same received PCM and application commands; diagnostic TX is discarded')


def session_case(args, bw, caller):
    return run_case(args, bw, caller)


def summary(result, out):
    def number(value):
        return '-' if value is None else f'{value:.2f}'
    def level(row, side):
        levels = row.get('metrics', {}).get(side, {}).get('levels', {}).get('tx', {})
        return f'{levels.get("highest") or "-"}/{levels.get("final") or "-"}'
    def table(rows):
        widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
        return ['  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    rows = [('condition', 'bw', 'caller', 'direction', 'Kestrel outcome', 'replay', 'level high/final',
             'retries', 'transfer s', 'goodput bps', 'Xara/Xara outcome', 'level high/final', 'transfer s', 'classification')]
    traces = []
    for pair in result['cases']:
        row = pair.get('session', pair)
        if row is None:
            continue
        baseline = pair.get('control') if isinstance(pair.get('control'), dict) else {}
        primary = row
        if row.get('control') is True:
            primary, baseline = {}, row
        c = row.get('condition', {})
        condition = c.get('snr', '-')+('dB' if c.get('snr') not in ('clean', 'inf') else '')
        if c.get('fading', 'none') != 'none':
            condition += '/'+c['fading']
        for name in ('offset', 'drift', 'clock_ppm', 'qrn'):
            if c.get(name) not in (None, 0, 'off'):
                condition += f'/{name}={c[name]}'
        condition += f' s={row.get("seed")} r={row.get("replicate")} '+('registered' if row.get('registered') else 'limited')
        for direction, side in (('a-to-b', 'a'), ('b-to-a', 'b')):
            timing = primary.get('timing', {}).get(direction, {})
            metrics = primary.get('metrics', {}).get(side, {})
            retries = '/'.join(str(metrics.get(k)) if metrics.get(k) is not None else '-'
                               for k in ('data_retries', 'control_retries'))
            rows.append((condition, row['bw'], row['caller'], direction, primary.get('outcome', '-'),
                primary.get('ptt-replay', {}).get('outcome', '-'), level(primary, side), retries,
                number(timing.get('seconds')), number(timing.get('goodput_bps')), baseline.get('outcome', '-'),
                level(baseline, side), number(baseline.get('timing', {}).get(direction, {}).get('seconds')),
                pair.get('comparison', '-')))
        traces.append(f'{condition} bw={row["bw"]} caller={row["caller"]}: '+
                      ' '.join(str(v['trace']) for v in (primary, baseline) if v.get('trace')))
    lines = table(rows)
    lines += ['', 'Levels are transmitted high/final; retries are data/control; - means unavailable.']
    for threshold in result.get('thresholds', []):
        mode = 'registered' if threshold['registered'] else 'limited'
        lines += ['', f'Threshold: bw={threshold["bw"]} fading={threshold["fading"]} mode={mode}',
                  '  rungs dB: '+', '.join(map(str, threshold['rungs'])),
                  f'  Kestrel: {threshold["kestrel_threshold"]}; Xara/Xara: {threshold["xara_threshold"]}; {threshold["verdict"]}']
    lines += ['', 'Traces:']+traces
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')


def sessions(args, control=False):
    rows = []
    start = time.monotonic()
    frozen = [(bw, caller) for bw in args.bw or [500, 2300, 2750] for caller in args.caller or ['kestrel', 'xara']]
    args.horizon = max(case_horizon(args, bw)[1] for bw, _ in frozen)
    print(json.dumps(dict(inventory=frozen)), flush=True)
    for bw, caller in frozen:
        rows.append(run_case(args, bw, caller, control=control))
        if not rows[-1]['ok'] and not args.keep_going:
            break
    result = dict(provenance=args.provenance, kind='control' if control else 'session', cases=rows,
                  inventory=frozen, complete=len(rows) == len(frozen), ok=all(r['ok'] for r in rows),
                  elapsed_seconds=time.monotonic()-start)
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    summary(result, args.out)
    return result['ok']


def controls(args):
    return sessions(args, control=True)


def conditions(full=False):
    rows = [Condition(), Condition('inf')]+[Condition(str(s)) for s in (10, 3, 0, -5, -10, -15)]
    rows += [Condition('3', p) for p in ('moderate', 'poor')]
    rows += [Condition('10', 'moderate', 50, .1, 50)]
    if full:
        rows += [Condition(str(s), p) for p in ('good', 'moderate', 'poor') for s in (3, -5, -10)
                 if not (s == 3 and p in ('moderate', 'poor'))]
        rows += [Condition('10', **{field: v}) for field, size in (('offset', 50), ('drift', .1), ('clock_ppm', 50)) for v in (size, -size)]
        rows += [Condition('0', qrn='standard')]
    return rows


def paired(session, control):
    if session.get('corruption_receiver') == 'a':
        return 'kestrel-corruption'
    if 'invalid' in (session['outcome'], control['outcome']):
        return 'invalid'
    if session.get('corruption_receiver') == 'b':
        return 'peer-finding' if control.get('corruption_receiver') in ('a', 'b') else 'interop-corruption'
    if control.get('corruption_receiver') in ('a', 'b'):
        return 'peer-finding'
    if session['ok'] and control['ok']:
        return 'both-pass'
    if not session['ok'] and not control['ok']:
        return 'both-fail-cleanly'
    if not session['ok']:
        return 'rx-gate-shortfall' if session.get('failed_phase') in session.get('ptt-replay', {}).get('recovered_phases', []) else 'cell-shortfall'
    return 'kestrel-only-pass'


def thresholds(cases, seeds, seed=None):
    groups = {}
    for pair in cases:
        row = pair['session']
        c = row['condition']
        if c['snr'] in ('clean', 'inf') or c['offset'] or c['drift'] or c['clock_ppm'] or c['qrn'] != 'off':
            continue
        key = (row['bw'], c['fading'], row['registered'])
        groups.setdefault(key, {}).setdefault(float(c['snr']), []).append(pair)
    result = []
    for key, rungs in groups.items():
        ordered = sorted(rungs, reverse=True)
        master = seed if seed is not None else cases[0]['session']['seed']
        expected = {(caller, master, replicate) for caller in ('kestrel', 'xara') for replicate in range(seeds)}
        def identity(row):
            return row['caller'], row['seed'], row['replicate']
        def coverage(rows):
            return (len(rows) == len(expected) and
                    all({identity(p[side]) for p in rows} == expected for side in ('session', 'control')) and
                    all(identity(p['session']) == identity(p['control']) for p in rows))
        reached = {}
        for side in ('session', 'control'):
            passes = [i for i, snr in enumerate(ordered) if coverage(rungs[snr]) and all(p[side]['ok'] for p in rungs[snr])]
            reached[side] = max(passes) if passes else None
        k, x = reached['session'], reached['control']
        shortfall = x is not None and (k is None or x-k > 1)
        covered = all(coverage(rows) for rows in rungs.values())
        result.append(dict(bw=key[0], fading=key[1], registered=key[2], rungs=ordered,
            kestrel_threshold=ordered[k] if k is not None else None,
            xara_threshold=ordered[x] if x is not None else None,
            verdict='incomplete' if not covered else 'smoke' if seeds < 3 else
                    'shortfall' if shortfall else 'no-baseline-threshold' if x is None else 'within-one-rung'))
    return result


def ladder(args):
    expanded = []
    for c in [args.condition] if args.custom else conditions(args.ladder == 'full'):
        modes = [False, True] if args.ladder == 'full' and (c.snr in ('clean', 'inf') or
            (c.snr in ('10', '3') and c.fading == 'none' and not (c.offset or c.drift or c.clock_ppm) and c.qrn == 'off') or
            c == Condition('3', 'good')) else [False if args.ladder == 'full' else args.registered]
        for mode in modes:
            for replicate in range(1 if c.snr == 'clean' or args.ladder != 'full' else 3):
                for bw in args.bw or [500, 2300, 2750]:
                    for caller in args.caller or ['kestrel', 'xara']:
                        expanded.append(dict(condition=asdict(c), registered=mode, replicate=replicate, bw=bw, caller=caller))
    horizon = max(case_horizon(args, bw)[1] for bw in args.bw or [500, 2300, 2750])
    for item in expanded:
        Channel(Condition(**item['condition']), horizon=horizon)
        item['status'] = 'unrun'
    print(json.dumps(dict(inventory=expanded, horizon_seconds=horizon), indent=2), flush=True)
    phy_rows = [args.condition] if args.custom else (
        [Condition(), Condition('3')] if args.ladder == 'default' else
        [Condition(), Condition('inf')]+[Condition(str(s)) for s in (10, 3, 0, -5, -10, -15)]+
        [Condition(str(s), 'moderate') for s in (3, -5, -10)])
    phy_inventory = [dict(condition=asdict(c), levels=args.level or ([1, 4] if args.ladder == 'default' else None),
                          bandwidths=args.bw or [500, 2300, 2750], pairs=args.ladder == 'full' or args.pairs)
                     for c in phy_rows]
    for item in phy_inventory:
        item['status'] = 'unrun'
        item['probes'] = [dict(record, frames=count, directions=['A->B', 'B->A'], seed=args.seed, replicate=0)
            for record in inventory() if record['bw'] in item['bandwidths'] and
            (item['levels'] is None or record['level'] in item['levels'])
            for count in ([1, 2] if item['pairs'] and 'Ofdm' in record['waveform'] and
                record['level'] >= {500: 12, 2300: 15, 2750: 16}[record['bw']] else [1])]
    print(json.dumps(dict(phy_inventory=phy_inventory), indent=2), flush=True)
    result = dict(provenance=args.provenance, kind='ladder', inventory=expanded, phy_inventory=phy_inventory,
                  cases=[], phy=[], complete=False, ok=False)
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    start = time.monotonic()
    for index, item in enumerate(expanded):
        pair = {}
        for side in ('session', 'control'):
            directory = args.out/f'{index:04d}'/side
            child = SimpleNamespace(**(vars(args) | item | dict(out=directory, condition=Condition(**item['condition']), horizon=horizon)))
            pair[side] = run_case(child, item['bw'], item['caller'], control=side == 'control')
        pair['comparison'] = paired(pair['session'], pair['control'])
        result['cases'].append(pair)
        item['status'] = 'complete'
        (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        if (not pair['session']['ok'] or not pair['control']['ok']) and not args.keep_going:
            break
    result['complete'] = len(result['cases']) == len(expanded)
    if result['complete']:
        for index, item in enumerate(phy_inventory):
            directory = args.out/'phy'/f'{index:02d}'
            directory.mkdir(parents=True)
            child = SimpleNamespace(**(vars(args) | dict(out=directory, condition=Condition(**item['condition']),
                level=item['levels'], pairs=item['pairs'], horizon=horizon, replicate=0)))
            phy_cases(child)
            result['phy'].append(json.loads((directory/'result.json').read_text()))
            item['status'] = 'complete'
    result['thresholds'] = thresholds(result['cases'], 3 if args.ladder == 'full' else 1, args.seed)
    result['ok'] = result['complete'] and not any(p['comparison'] in
        ('invalid', 'kestrel-corruption', 'interop-corruption') for p in result['cases']) and not any(
        t['verdict'] in ('shortfall', 'incomplete') for t in result['thresholds'])
    result['ok'] = result['ok'] and not any('error' in v or v.get('comparison') in ('kestrel-corruption', 'interop-corruption')
        for r in result['phy'] for c in r['cases'] for v in c['results'].values())
    result['elapsed_seconds'] = time.monotonic()-start
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    summary(result, args.out)
    return result['ok']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['phy', 'session', 'control', 'ladder'])
    parser.add_argument('--bw', type=int, choices=[500, 2300, 2750], action='append')
    parser.add_argument('--level', type=int, action='append')
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--keep-audio', action='store_true')
    parser.add_argument('--caller', choices=['kestrel', 'xara'], action='append')
    parser.add_argument('--registered', action='store_true')
    parser.add_argument('--compression', choices=['OFF', 'TEXT', 'FILES'], default='OFF')
    parser.add_argument('--bytes', type=int, default=256)
    parser.add_argument('--record-audio', action='store_true')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--application-gap', type=float, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ladder', choices=['default', 'full'])
    parser.add_argument('--snr')
    parser.add_argument('--fading', choices=['none', 'good', 'moderate', 'poor'])
    parser.add_argument('--offset', type=float)
    parser.add_argument('--drift', type=float)
    parser.add_argument('--clock-ppm', type=float)
    parser.add_argument('--qrn', choices=['off', 'standard'])
    parser.add_argument('--seed', type=int, default=20261025)
    argv = sys.argv[1:] if argv is None else argv
    if any(a == '--noise' or a.startswith('--noise=') for a in argv):
        parser.error('--noise was receiver RMS, not SNR; migrate to --snr DB (SNR2500), or --snr clean')
    args = parser.parse_args(argv)
    for name in ('bw', 'level', 'caller'):
        values = getattr(args, name)
        if values:
            setattr(args, name, list(dict.fromkeys(values)))
    fields = ('snr', 'fading', 'offset', 'drift', 'clock_ppm', 'qrn')
    args.custom = any(getattr(args, name) is not None for name in fields)
    if args.ladder and (args.kind != 'ladder' or args.custom):
        parser.error('--ladder is only for ladder and conflicts with explicit channel flags')
    args.ladder = args.ladder or 'default'
    try:
        args.condition = Condition(**{name: getattr(args, name) for name in fields if getattr(args, name) is not None})
    except (ValueError, OverflowError) as exc:
        parser.error(str(exc))
    if args.bytes < 1 or not np.isfinite(args.application_gap) or args.application_gap < 0:
        parser.error('--bytes must be positive and --application-gap finite and nonnegative')
    args.replicate = 0
    args.out = args.out.expanduser().resolve()
    return args


def main():
    args = parse_args()
    if args.out.exists():
        raise RuntimeError('--out must be a fresh directory')
    validate_probe()
    args.out.mkdir(parents=True)
    args.provenance = provenance()
    return 0 if {'phy': phy_cases, 'session': sessions, 'control': controls, 'ladder': ladder}[args.kind](args) else 1


if __name__ == '__main__':
    raise SystemExit(main())
