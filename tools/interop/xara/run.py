#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Offline Kestrel/Xara interop. Run from any directory; no radio or sound I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from paths import (HERE, ROOT, XARA, PROBE, PIN, XARA_URL, git, sha256, selection,
                   validate_probe, validate_kestrel_imports, git_metadata, source_identity)
validate_kestrel_imports()

import numpy as np
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.coding.crc import crc16_genibus
from hfmodem.kestrel.tx import varahf500_tx as tx500, varahf2300_tx as txwide

HARNESS_HASHES = {name: sha256(HERE / name) for name in
                  ('run.py', 'paths.py', 'cli.py', 'prepare.py', 'check.sh',
                   'checks.py', 'checks_ipc.py', 'src/main.rs', 'build.rs', 'Cargo.toml.in', 'Cargo.lock')}


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
                        channel(render(frames, bw, level)).tofile(pcm)
                        got = [bytes.fromhex(h) for h in invoke('rx', bw, level, pcm, count).splitlines()]
                        want = frames
                    else:
                        invoke('tx', bw, level, ','.join(f.hex() for f in frames), pcm)
                        x = channel(np.fromfile(pcm, dtype='<f8'))
                        x.tofile(pcm)
                        read = phy.decode_level(x, str(bw), level, expect_pair=count == 2)
                        got = list(read.frames) if read else []
                        want = [f[:-2] for f in frames]
                    ok = got == want
                    detail = {'ok': ok, 'expected': [f.hex() for f in want],
                              'decoded': [f.hex() for f in got],
                              'pcm_sha256': hashlib.sha256(pcm.read_bytes()).hexdigest()}
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


def session_case(args, bw, caller):
    with ExitStack() as resources:
        return _session_case(args, bw, caller, resources)


def _session_case(args, bw, caller, resources):
    from hfmodem.kestrel.vara import vara_arq as arq
    from hfmodem.kestrel.host.audio_runtime import AudioHostRuntime
    from hfmodem.kestrel.host.server import HostSession

    tag = f'bw{bw}-{caller}-{args.compression.lower()}-{"registered" if args.registered else "limited"}'
    directory = args.out / tag
    directory.mkdir(parents=True, exist_ok=True)
    t = Timeline()
    log = resources.enter_context((directory / 'trace.jsonl').open('w'))
    recordings = []
    if args.record_audio:
        for name in ('kestrel-to-xara', 'xara-to-kestrel'):
            wav = resources.enter_context(wave.open(str(directory / f'{name}.wav'), 'wb'))
            wav.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            recordings.append(wav)

    def trace(side, event):
        log.write(json.dumps({'seconds': t.n/48000, 'side': side, 'event': event,
                              'driver_seconds': t.clock()-1000 if side.startswith('kestrel') else t.n/48000}) + '\n')

    class Transport(arq.VaraIO):
        receiving = False
        def tx(self, samples):
            # Kestrel's renderers use arbitrary codec scales. Model the output
            # gain here; Xara already applies its native -5 dB drive law.
            samples = np.asarray(samples, float)
            rms = np.sqrt(np.mean(samples*samples))
            t.emit(samples * (0.12/rms) if rms else samples)
        def log(self, text): trace('kestrel', text)
        def next_rx_burst(self, timeout, hs): return None

    class Sink:
        def __init__(self, is_data=False):
            self.messages, self.data, self.is_data = [], bytearray(), is_data
        def sendall(self, blob):
            if self.is_data:
                self.data.extend(blob)
                trace('kestrel', 'DATA:' + blob.hex())
            else:
                for line in blob.decode('ascii').split('\r'):
                    if line:
                        self.messages.append(line)
                        trace('kestrel-host', line)

    wire = Transport()
    runtime = AudioHostRuntime(wire, high_speed=args.registered, clock=t.clock)
    modem = runtime.modem_factory()
    command, data = Sink(), Sink(True)
    host = HostSession(command, data, modem)
    resources.callback(modem.stop)
    modem.start(host)
    xara = Xara(bw, args.registered, args.compression, trace)
    resources.callback(xara.close)
    rng = np.random.default_rng(20261025)
    # Fixed 50 ms cable delay in each direction; sample clock is shared.
    delay = deque([np.zeros(480) for _ in range(5)])
    xdelay = deque([np.zeros(480) for _ in range(5)])
    history = deque(maxlen=50)
    burst, quiet = [], 0
    phases = {}

    def step():
        nonlocal burst, quiet
        n = t.n
        k_out = t.output(n, 480)
        delay.append(k_out)
        x_input = delay.popleft()
        if args.noise:
            x_input = x_input + rng.normal(0, args.noise, len(x_input))
        x_out = xara.step(x_input)
        for wav, pcm in zip(recordings, (k_out, x_out)):
            wav.writeframesraw((np.clip(pcm, -1, 1)*32767).round().astype('<i2').tobytes())
        xdelay.append(x_out)
        heard = xdelay.popleft()
        t.n += 480
        if args.noise:
            heard = heard + rng.normal(0, args.noise, len(heard))
        if t.n < t.blocked_until:
            history.clear()
            burst, quiet = [], 0
            return
        emitting = bool(np.max(np.abs(heard)) > max(0.001, args.noise*6))
        if emitting:
            if not burst:
                burst = list(history)
            quiet = 0
        if burst:
            burst.append(heard.copy())
            quiet = 0 if emitting else quiet + 480
        history.append(heard.copy())
        wire.receiving = emitting or bool(burst)
        modem.station.on_rx_stream(heard)
        if burst and quiet >= 12000 and t.n >= t.blocked_until:
            modem.station.on_rx_audio(np.concatenate(burst))
            burst, quiet = [], 0
            wire.receiving = False
        if t.n >= t.blocked_until:
            runtime.step(0)

    def until(predicate, phase, seconds):
        end = t.n + int(seconds*48000)
        while t.n < end:
            if predicate():
                phases[phase] = t.n/48000
                return
            step()
        raise TimeoutError(f'{phase}: kestrel={modem.station.state}/{modem.station.turn}, xara={xara.state}')

    start = time.monotonic()
    result = {'bw': bw, 'caller': caller, 'registered': args.registered,
              'compression': args.compression, 'noise_rms': args.noise, 'ok': False, 'phases': phases}
    result['channel'] = {'rate': 48000, 'block_samples': 480, 'delay_samples': 2400,
                         'kestrel_tx_rms': .12, 'xara_drive_db': -5,
                         'application_gap_seconds': args.application_gap}
    try:
        # Patch only the driver module's clock; never modify the process-wide time module.
        from types import SimpleNamespace
        with patch.object(arq, 'time', SimpleNamespace(monotonic=t.clock)):
            for line in ['MYCALL W9SSJ', f'BW{bw}', f'COMPRESSION {args.compression}', 'LISTEN ON']:
                host._dispatch(line)
                modem.poll()
            for _ in range(100): step()
            if caller == 'kestrel':
                host._dispatch('CONNECT W9SSJ W1AW')
                modem.poll()
            else:
                xara.command('CONNECT')
            until(lambda: any(s.startswith('CONNECTED ') for s in command.messages)
                  and any(s.startswith('Host(Connected ') for s in xara.events), 'connect', 75)
            for _ in range(round(args.application_gap*100)): step()
            pa = (b'Hello Xara from Kestrel. 0123456789\r\n' * 10)[:args.bytes]
            pb = (b'Hello Kestrel from Xara. abcdefghij\r\n' * 10)[:args.bytes]
            if args.bytes > len(pa):
                pa = (pa * (args.bytes//len(pa)+1))[:args.bytes]
                pb = (pb * (args.bytes//len(pb)+1))[:args.bytes]
            modem.transmit(pa)
            until(lambda: len(xara.got) >= len(pa), 'kestrel-to-xara', 60 + args.bytes/6)
            assert bytes(xara.got) == pa, 'Kestrel-to-Xara payload mismatch'
            for _ in range(round(args.application_gap*100)): step()
            xara.command('SEND ' + pb.hex())
            until(lambda: len(data.data) >= len(pb), 'xara-to-kestrel', 60 + args.bytes/6)
            assert bytes(data.data) == pb, 'Xara-to-Kestrel payload mismatch'
            # Flush all host bytes before orderly close, then observe both ends.
            for _ in range(500): step()
            host._dispatch('DISCONNECT')
            until(lambda: 'DISCONNECTED' in command.messages and 'Host(Disconnected)' in xara.events,
                  'disconnect', 60)
            assert bytes(xara.got) == pa and bytes(data.data) == pb, 'duplicate data after close'
            result['ok'] = True
            result['bytes_each_direction'] = args.bytes
    except Exception as error:
        result['error'] = repr(error)
        (directory/'exception.txt').write_text(traceback.format_exc())
    result.update(elapsed_seconds=time.monotonic()-start, simulated_seconds=t.n/48000,
                  kestrel_received=len(data.data), xara_received=len(xara.got))
    (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'{tag}: {"PASS" if result["ok"] else "FAIL"} {result.get("error", "")}', flush=True)
    return result


def sessions(args):
    rows = []
    for bw in args.bw or [500, 2300, 2750]:
        for caller in args.caller or ['kestrel', 'xara']:
            row = session_case(args, bw, caller)
            rows.append(row)
            if not row['ok'] and not args.keep_going:
                # Evidence first: do not disguise a failed endpoint with fixes.
                break
        if not rows[-1]['ok'] and not args.keep_going: break
    result = {'provenance': args.provenance, 'kind': 'session', 'cases': rows,
              'ok': all(r['ok'] for r in rows)}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result['ok']


def controls(args):
    """Xara/Xara through the same compiled endpoint and binary PCM IPC."""
    rows = []
    for bw in args.bw or [500, 2300, 2750]:
        with ExitStack() as resources:
            rows.append(_control_case(args, bw, resources))
    result = {'provenance': args.provenance, 'kind': 'xara-control', 'cases': rows,
              'ok': all(r['ok'] for r in rows)}
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result['ok']


def _control_case(args, bw, resources):
    directory = args.out / f'bw{bw}'
    directory.mkdir(parents=True, exist_ok=True)
    log = resources.enter_context((directory/'trace.jsonl').open('w'))
    now = 0
    def trace(side, event):
        log.write(json.dumps({'seconds': now/48000, 'side': side, 'event': event})+'\n')
    a = Xara(bw, args.registered, args.compression, lambda _, e: trace('a', e), 'W9SSJ', 'W1AW')
    resources.callback(a.close)
    b = Xara(bw, args.registered, args.compression, lambda _, e: trace('b', e))
    resources.callback(b.close)
    qa, qb = (deque([np.zeros(480) for _ in range(5)]) for _ in range(2))
    phases = {}
    def step():
        nonlocal now
        oa, ob = a.step(qb.popleft()), b.step(qa.popleft())
        qa.append(oa)
        qb.append(ob)
        now += 480
    def until(done, phase, seconds=120):
        end = now+seconds*48000
        while now < end and not done(): step()
        if not done(): raise TimeoutError(f'{phase}: {a.state}/{b.state}')
        phases[phase] = now/48000
    row = {'bw': bw, 'ok': False, 'phases': phases}
    try:
        for _ in range(100): step()
        a.command('CONNECT')
        until(lambda: all(any(e.startswith('Host(Connected ') for e in x.events) for x in (a, b)), 'connect')
        pa, pb = b'Control A\r\n'*23, b'Control B\r\n'*23
        a.command('SEND '+pa.hex())
        until(lambda: len(b.got) >= len(pa), 'a-to-b')
        assert bytes(b.got) == pa
        b.command('SEND '+pb.hex())
        until(lambda: len(a.got) >= len(pb), 'b-to-a')
        assert bytes(a.got) == pb
        for _ in range(500): step()
        a.command('DISCONNECT')
        until(lambda: all('Host(Disconnected)' in x.events for x in (a, b)), 'disconnect')
        assert bytes(b.got) == pa and bytes(a.got) == pb
        row['ok'] = True
    except Exception as error:
        row['error'] = repr(error)
        (directory/'exception.txt').write_text(traceback.format_exc())
    print(f'Xara/Xara BW{bw}: {"PASS" if row["ok"] else "FAIL"}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['phy', 'session', 'control'])
    parser.add_argument('--bw', type=int, choices=[500, 2300, 2750], action='append')
    parser.add_argument('--level', type=int, action='append')
    parser.add_argument('--pairs', action='store_true')
    parser.add_argument('--keep-audio', action='store_true')
    parser.add_argument('--caller', choices=['kestrel', 'xara'], action='append')
    parser.add_argument('--registered', action='store_true')
    parser.add_argument('--compression', choices=['OFF', 'TEXT', 'FILES'], default='OFF')
    parser.add_argument('--bytes', type=int, default=256)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--record-audio', action='store_true')
    parser.add_argument('--keep-going', action='store_true', help='record every independent session result, including failures')
    parser.add_argument('--application-gap', type=float, default=0,
                        help='explicit delay after connect and before reverse transfer (default: immediate)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    if args.out.exists():
        parser.error('--out must be a fresh directory')
    validate_probe()
    args.out.mkdir(parents=True)
    if args.bytes < 1 or args.noise < 0 or args.application_gap < 0:
        parser.error('--bytes must be positive; --noise and --application-gap must be nonnegative')
    args.provenance = provenance()
    return 0 if {'phy': phy_cases, 'session': sessions, 'control': controls}[args.kind](args) else 1


if __name__ == '__main__':
    raise SystemExit(main())
