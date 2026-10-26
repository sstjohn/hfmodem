#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-only PACTOR corpus watch, ranked by completed Winlink sessions."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import time
import wave

import numpy as np

from corpuslib import feed, inventory
from corpuslib.detector import fsk_offset
from corpuslib.policy import Hold
from hfmodem.core import claim, config
from hfmodem.core.native import spawn_native
from hfcapture import (Capture, RATE, portaudio, sound_input, device_claim,
                       device_opened, stop_native, close_native, drain_native)

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--preview', action='store_true', help='rank channels without radio access')
    action.add_argument('--execute', action='store_true', help='tune and record; receive only')
    action.add_argument('--replay', type=Path, help='exercise detector/hold policy on a mono 48 kHz PCM16 WAV')
    action.add_argument('--inventory', type=Path, nargs='?', const=Path('latest'), metavar='DIRECTORY',
                        help='print the latest capture inventory, or name a run directory; no radio access')
    p.add_argument('--grid', default=None,
                   help="receiver locator for distance ranking (default: [station] grid; "
                        "without one, sessions are ranked without distance)")
    p.add_argument('--feed-file', type=Path, help='saved Winlink PropagationRecords JSON; no network')
    p.add_argument('--centers', type=int, nargs='+', help='explicit center frequencies in Hz, ranked in given order')
    p.add_argument('--hours', type=int, default=24)
    p.add_argument('--reach-km', type=float, default=4200)
    p.add_argument('--min-hz', type=int, default=3000000)
    p.add_argument('--max-hz', type=int, default=30000000)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--refresh', type=float, default=900, help='refresh completed-session ranking every N seconds')
    p.add_argument('--dwell', type=float, default=60)
    p.add_argument('--quiet', type=float, default=120, help='release after this many seconds without useful evidence (default 120)')
    p.add_argument('--candidate', type=float, default=15, help='one bounded candidate extension per visit')
    p.add_argument('--max-hold', type=float, default=1800)
    p.add_argument('--max-offset', type=float, default=60, help='maximum fitted FSK offset in Hz')
    p.add_argument('--visits', type=int, default=0, help='stop after N channel visits; 0 runs until interrupted')
    p.add_argument('--device', default=None)
    p.add_argument('--cat-port', default=None,
                   help='defaults to CAT_PORT or the console station port')
    p.add_argument('--model', type=int, default=None)
    p.add_argument('--baud', type=int, default=None)
    p.add_argument('--rigctld', help='use an existing daemon at host:port instead of starting one')
    p.add_argument('--out', type=Path, help='new output directory')
    p.add_argument('--min-free-gb', type=float, default=2)
    from hfmodem.core.rigs import RIGS
    p.add_argument("--rig", choices=tuple(RIGS))
    return p


def utc():
    return datetime.now(timezone.utc).isoformat()


def collection_home():
    from consolelib.operator import data_root
    return data_root() / 'collections'


def latest_run():
    """Locate recorded runs, including custom output and older captures/ runs."""
    home = collection_home()
    manifests = set(home.glob('runs/*/collector.json'))
    manifests.update((collection_home().parent / 'captures').glob('*/collector.json'))
    try:
        pointer = json.loads((home / 'latest.json').read_text())
        manifests.add(Path(pointer['directory']) / 'collector.json')
    except (OSError, ValueError, KeyError, TypeError):
        pass
    candidates = []
    for manifest in manifests:
        try:
            metadata = json.loads(manifest.read_text())
            if metadata.get('arguments', {}).get('replay'):
                continue
            started = datetime.fromisoformat(metadata['started_utc']).timestamp()
            candidates.append((started, str(manifest.parent.resolve())))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    if not candidates:
        raise ValueError('No collector run found yet. Start one with: tools/console pactor collect --execute')
    return Path(max(candidates)[1])


def save(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def candidates(args):
    if args.centers:
        return [dict(center_hz=c, dial_hz=c-1500, source='explicit')
                for c in dict.fromkeys(args.centers)], {'source': 'explicit'}
    data = json.loads(args.feed_file.read_text()) if args.feed_file else feed.fetch(args.hours)
    channels, tally = feed.rank(data, args.grid, hours=args.hours, reach_km=args.reach_km,
                                min_hz=args.min_hz, max_hz=args.max_hz)
    return channels[:args.channels], dict(tally, source='completed-Winlink-sessions', loaded_utc=utc(),
                                         input=str(args.feed_file) if args.feed_file else 'live-feed')


def request(address, command):
    # CAT PTT support is needed to *read* transmit state. Restrict this client's
    # commands independently: it can query/tune, never issue a PTT write.
    if command not in ('f', 'm', 't', 'M USB 3000') and not re.fullmatch(r'F [0-9]+', command):
        raise ValueError(f'command is not permitted by the receive-only collector: {command}')
    with socket.create_connection(address, timeout=3) as connection:
        connection.settimeout(3)
        connection.sendall((command + '\n').encode('ascii'))
        reader = connection.makefile('r')
        count = 2 if command == 'm' else 1
        rows = []
        for _ in range(count):
            rows.append(reader.readline().strip())
            if rows[-1].startswith('RPRT ') or not rows[-1]:
                break
    if not rows[0] or any(r.startswith('RPRT ') and r != 'RPRT 0' for r in rows):
        raise RuntimeError(f'rigctld {command.split()[0]} failed: {rows}')
    return rows


@contextmanager
def receiver(args, out):
    if not args.cat_port:
        raise ValueError('--cat-port or CAT_PORT is required for the shared rig claim')
    proc, log, readers = None, None, []
    with device_claim("corpus_watch", devices=claim.Devices(
            cat_port=args.cat_port, capture=args.device,
            cat="rigctld" if args.rigctld else "serial")) as held:
        try:
            if args.rigctld:
                host, port = args.rigctld.rsplit(':', 1)
                address = (host, int(port))
            else:
                # A private port: never kill, adopt, or silently connect to another
                # rigctld. The shared CAT claim precedes opening the serial port.
                address = ('127.0.0.1', 4537)
                with socket.socket() as check:
                    check.bind(address)
                exe = shutil.which('rigctld') or str(Path(os.environ.get(
                    'HAMLIB_BIN', Path.home() / 'src/radio/hamlib/_install/bin')) / 'rigctld')
                log = (out / 'rigctld.log').open('w')
                proc = spawn_native([exe, '-m', str(args.model), '-r', args.cat_port,
                    '-s', str(args.baud), '-t', str(address[1]), '-T', address[0],
                    '--set-conf=ptt_type=None'], claim_fd=held.claim,
                    long_lived=True, device=args.cat_port)
                readers = drain_native(proc, log)
            for _ in range(20):
                if proc is not None and not proc.alive():
                    raise RuntimeError('collector rigctld exited; see rigctld.log')
                try:
                    request(address, 'f')
                    break
                except OSError:
                    time.sleep(.2)
            else:
                raise RuntimeError('rigctld did not become ready')
            device_opened()
            yield address
        finally:
            if proc is not None:
                stop_native(proc)
                for reader in readers:
                    reader.join(timeout=2)
                    if reader.is_alive():
                        raise claim.ClaimError("collector diagnostic reader did not stop")
                close_native(proc, stopped=True)
            if log is not None:
                log.close()


def tune(address, center):
    # Only read PTT. No PTT command, audio output stream, or modem TX path.
    if request(address, 't')[0] != '0':
        raise RuntimeError('rig reports PTT active; refusing to tune')
    request(address, 'M USB 3000')
    request(address, f'F {center - 1500}')
    dial = int(request(address, 'f')[0])
    mode, width = request(address, 'm')
    if dial != center - 1500 or mode != 'USB' or int(width) < 2700:
        raise RuntimeError(f'receiver readback differs: {dial} {mode} {width}')
    return dict(dial_hz=dial, mode=mode, passband_hz=int(width))


def monitor(args, events):
    from creance.monitor.driver import ModemMonitor, MonitorSpec
    import sys
    spec = MonitorSpec('pactor-corpus', sys.executable,
                      str(ROOT / 'tools/corpuslib/detector.py'), str(ROOT),
                      (str(args.max_offset),))
    mon = ModemMonitor(spec, events, backlog_s=12)
    mon.start()
    return mon


def visit(args, path, channel, stopped, readback):
    path.mkdir()
    metadata = dict(started_utc=utc(), channel=channel, readback=readback,
                    receive_only=True, sample_rate=RATE, policy=dict(
                    dwell=args.dwell, quiet=args.quiet, candidate=args.candidate,
                    maximum=args.max_hold, max_offset_hz=args.max_offset))
    save(path / 'session.json', metadata)
    incoming, events = queue.Queue(maxsize=200), queue.Queue()
    capture = Capture(path)
    hold = Hold(args.dwell, args.quiet, args.candidate, args.max_hold)
    mon = monitor(args, events)
    fatal = []
    started = time.monotonic()
    state = None
    tail = np.zeros(0)
    next_screen = RATE
    reason = 'interrupted'
    gaps = 0
    analyzed_until = 0.
    pending_hop = None

    def callback(data, frames, stamp, status):
        try:
            incoming.put_nowait((bytes(data), float(stamp.inputBufferAdcTime), str(status), time.time()))
        except queue.Full:
            fatal.append('recording queue overflow; capture incomplete')

    with (path / 'events.jsonl').open('w', buffering=1) as log:
        def emit(obj):
            log.write(json.dumps(dict(wall_utc=utc(), **obj)) + '\n')

        def detections():
            nonlocal analyzed_until
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    break
                if event.kind == 'ANALYZED':
                    analyzed_until = max(analyzed_until, event.t)
                    emit(dict(event='analysis-progress', through_s=analyzed_until))
                    continue
                action = hold.observe(event.t, event.kind)
                emit(dict(event='detection', **asdict(event), action=action))
                print(f'  {event.t:7.1f}s {event.kind}: {action} {event.detail}', flush=True)

        try:
            sd = portaudio()
            device, name = sound_input(sd, args.device)
            metadata['device'] = name
            with device_claim("corpus_watch input", devices=claim.Devices(capture=args.device)):
                with sd.RawInputStream(device=device, samplerate=RATE, channels=1,
                                       dtype='int16', blocksize=4800, callback=callback):
                    device_opened()
                    while not stopped():
                        if fatal:
                            raise RuntimeError(fatal[0])
                        if not mon.alive:
                            raise RuntimeError('PACTOR detector exited; channel is unclassified, not quiet')
                        if mon.dropped_samples:
                            raise RuntimeError('PACTOR detector fell behind and dropped samples; '
                                               'capture retained, timing evidence is incomplete')
                        try:
                            pcm, adc, status, arrived = incoming.get(timeout=3)
                        except queue.Empty:
                            raise RuntimeError('receiver audio stalled') from None
                        at = capture.frames
                        if at == 0:
                            metadata['first_block'] = dict(adc_time=adc, received_unix=arrived,
                                                           samples=len(pcm)//2)
                        gap = capture.feed(pcm, adc)
                        if gap or status:
                            gaps += 1
                            emit(dict(event='audio-discontinuity', sample=at, gap_s=gap, status=status))
                        mon.feed(at, pcm)
                        tail = np.concatenate((tail, np.frombuffer(pcm, '<i2').astype(float) / 32768))[-RATE:]
                        if capture.frames >= next_screen:
                            next_screen += RATE
                            offset = fsk_offset(tail)
                            if offset is not None and abs(offset) <= args.max_offset:
                                action = hold.observe(capture.seconds, 'CENTERED-FSK-CANDIDATE')
                                if action != 'no-hold-evidence':
                                    emit(dict(event='candidate', t=capture.seconds, offset_hz=offset, action=action))
                        detections()
                        stay, why = hold.decision(capture.seconds)
                        if not stay and capture.seconds < args.max_hold:
                            # Finish classifying the audio that led to this hop
                            # before releasing the channel. Continue recording while
                            # the decoder catches up, rather than waiting with deaf IO.
                            if pending_hop is None:
                                pending_hop = capture.seconds
                            if analyzed_until + .001 < pending_hop:
                                if capture.seconds - pending_hop > 12:
                                    raise RuntimeError('detector did not classify the hop boundary within 12 seconds')
                                stay, why = True, 'waiting-for-detector-before-hop'
                        elif stay:
                            pending_hop = None
                        if why != state:
                            print(f'  {capture.seconds:.1f}s {why}', flush=True)
                            emit(dict(event='decision', t=capture.seconds, reason=why, stay=stay))
                            state = why
                        if not stay:
                            reason = why
                            break
                        if shutil.disk_usage(path).free < args.min_free_gb * 1e9:
                            raise RuntimeError('minimum free disk space reached')
        except BaseException as exc:
            reason = f'error: {type(exc).__name__}: {exc}'
            raise
        finally:
            capture.close()
            elapsed = time.monotonic() - started
            drained = mon.stop(grace=5)
            detections()
            metadata.update(ended_utc=utc(), reason=reason, audio_seconds=capture.seconds,
                elapsed_seconds=elapsed, frames=capture.frames, gaps=capture.gaps,
                audio_status_events=gaps, queue_errors=fatal, detector_drained=drained,
                detector_dropped_samples=mon.dropped_samples, hold=asdict(hold),
                analyzed_through_s=analyzed_until,
                continuous=capture.continuous(elapsed) and not gaps and not fatal,
                files=[p.name for p in capture.parts])
            # Infinite sentinels are internal policy state, not JSON numbers.
            metadata['hold'] = {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
                                for k, v in metadata['hold'].items()}
            save(path / 'session.json', metadata)


def replay(args, out):
    from corpuslib.detector import analyze, BLOCK, LOOKBACK
    from hfmodem.core.wav import source
    hold = Hold(args.dwell, args.quiet, args.candidate, args.max_hold)
    seen = {}
    with wave.open(source(args.replay), 'rb') as wav, (out / 'replay.jsonl').open('w') as log:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise ValueError('replay requires mono PCM16 WAV at 48000 Hz')
        tail, base = np.zeros(0), 0
        while raw := wav.readframes(BLOCK):
            audio = np.concatenate((tail, np.frombuffer(raw, '<i2').astype(float) / 32768))
            for event in analyze(audio, args.max_offset):
                event['t'] += base / RATE
                key = (event['protocol'], event['kind'])
                if event['t'] <= seen.get(key, -1e9) + .75:
                    continue
                seen[key] = event['t']
                event['action'] = hold.observe(event['t'], event['kind'])
                log.write(json.dumps(event) + '\n')
                print(f"{event['t']:.1f}s {event['kind']} {event['action']}", flush=True)
            now = (base + len(audio)) / RATE
            stay, reason = hold.decision(now)
            log.write(json.dumps(dict(t=now, stay=stay, reason=reason)) + '\n')
            cut = max(0, len(audio) - LOOKBACK)
            tail, base = audio[cut:], base + cut
    print(out / 'replay.jsonl')


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    from hfmodem.core.rigs import launcher_defaults
    from consolelib import station
    launcher_defaults(args, audio_fields={'device': 'input'},
                      endpoint_fields={'cat_port': 'cat_port'},
                      legacy={'cat_port': station.resolve_selection(args.rig).CAT_PORT,
                              'device': 'USB Audio Device'})
    if args.inventory:
        inventory.write(latest_run() if args.inventory == Path('latest') else args.inventory)
        return 0
    if args.grid is None:
        args.grid = config.grid()
    if args.grid and not re.fullmatch(r'[A-Ra-r]{2}\d{2}([A-Xa-x]{2})?', args.grid):
        p.error('--grid must be a four- or six-character Maidenhead locator')
    positive = ('hours', 'reach_km', 'channels', 'refresh', 'dwell', 'quiet',
                'candidate', 'max_hold', 'max_offset', 'min_free_gb')
    if any(not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0 for name in positive):
        p.error('timing, count, offset and capacity options must be positive and finite')
    if args.max_hold < args.dwell or args.visits < 0:
        p.error('--max-hold must cover --dwell; --visits cannot be negative')
    if args.max_offset > 100:
        p.error('--max-offset must be at most 100 Hz; this watch does not chase adjacent channels')
    if not 1800000 <= args.min_hz <= args.max_hz <= 30000000:
        p.error('--min-hz and --max-hz must define an HF range in Hz')
    if args.centers and any(not 1800000 <= c <= 30000000 for c in args.centers):
        p.error('--centers must be HF center frequencies in Hz')
    if args.replay:
        channels, census = [], {}
    else:
        channels, census = candidates(args)
        print(json.dumps(dict(census=census, channels=channels), indent=2))
        if not channels:
            raise RuntimeError('No eligible PACTOR session channels; widen range or supply --centers')
    if args.preview:
        return 0
    home = collection_home()
    out = args.out or home / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *sorted((ROOT / 'tools/corpuslib').glob('*.py'))]
    save(out / 'collector.json', dict(started_utc=utc(), receive_only=True,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sources}))
    if args.replay:
        replay(args, out)
        return 0
    home.mkdir(parents=True, exist_ok=True)
    save(home / 'latest.json', dict(directory=str(out.resolve()), started_utc=utc()))
    print(f'Recording into {out.resolve()}; Ctrl-C finishes and prints the inventory.', flush=True)
    stop = False

    def halt(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)
    save(out / 'ranking-0000.json', dict(census=census, channels=channels))
    count, refreshed = 0, time.monotonic()
    with inventory_on_exit(out), receiver(args, out) as address:
        while not stop and (not args.visits or count < args.visits):
            for channel in channels:
                if stop or (args.visits and count >= args.visits):
                    break
                if shutil.disk_usage(out).free < args.min_free_gb * 1e9:
                    raise RuntimeError('minimum free disk space reached')
                readback = tune(address, channel['center_hz'])
                time.sleep(.5)
                print(f"Listening {channel['center_hz']} Hz center / {channel['dial_hz']} Hz USB dial", flush=True)
                visit(args, out / f"{count:04d}-{channel['center_hz']}", channel, lambda: stop, readback)
                inventory.write(out, display=False)
                count += 1
            if not args.centers and not stop and time.monotonic() - refreshed >= args.refresh:
                # Refresh only between rounds, never abandon a held exchange.
                # Failure is explicit and retains last known channels; timestamps
                # in ranking artifacts make that fallback visible.
                try:
                    updated, census = candidates(args)
                    if not updated:
                        raise RuntimeError('refresh returned no eligible channels')
                    channels = updated
                    save(out / f'ranking-{count:04d}.json', dict(census=census, channels=channels))
                except Exception as exc:
                    print(f'Feed refresh failed: {exc}; retaining previous ranking', flush=True)
                    save(out / f'feed-error-{count:04d}.json', dict(utc=utc(), error=str(exc)))
                refreshed = time.monotonic()
    return 0


@contextmanager
def inventory_on_exit(out):
    status = dict(status='finished', ended_utc=None)
    try:
        yield
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # Also inventory completed visits when a later tune/capture fails.
        # A reporting failure must not replace the original collection error.
        try:
            status['ended_utc'] = utc()
            save(out / 'run-status.json', status)
            inventory.write(out)
        except (OSError, ValueError) as exc:
            print(f'Could not write capture inventory: {exc}; recordings remain in {out}', flush=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
