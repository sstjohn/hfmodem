# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import struct
import sys
import threading

from hfmodem.core import coexist
from hfmodem.tests.coexist import shim


def parse_args(argv):
    result = dict(rate=48000, channels=1, duration=None, format=None,
                  backend=None, device=None, output=None, listing=False)
    values = {'-f', '-i', '-ac', '-ar', '-t', '-c:a', '-acodec', '-loglevel',
              '-list_devices', '-thread_queue_size', '-sample_rate', '-channels'}
    flags = {'-hide_banner', '-y', '-nostdin', '-vn', '-nostats', '-n'}
    args = iter(argv)
    for arg in args:
        if arg in flags:
            continue
        if arg in values:
            value = next(args, None)
            if value is None:
                raise ValueError(f'missing value for {arg}')
            if arg == '-f':
                result['backend' if result['device'] is None else 'format'] = value
            elif arg == '-i':
                result['device'] = value
            elif arg in ('-ar', '-sample_rate'):
                result['rate'] = int(value)
            elif arg in ('-ac', '-channels'):
                result['channels'] = int(value)
            elif arg == '-t':
                result['duration'] = float(value)
            elif arg == '-list_devices':
                result['listing'] = value == 'true'
            elif arg in ('-c:a', '-acodec') and value != 'pcm_s16le':
                raise ValueError(f'unsupported codec: {value}')
        elif arg == '-' or not arg.startswith('-'):
            if result['output'] is not None:
                raise ValueError('multiple outputs')
            result['output'] = arg
        else:
            raise ValueError(f'unsupported ffmpeg option: {arg}')
    if result['backend'] not in ('avfoundation', 'pulse', 'alsa', 'dshow'):
        raise ValueError('unsupported fake capture backend')
    if result['listing']:
        return result
    if result['device'] is None or result['output'] is None:
        raise ValueError('capture input and output required')
    result['format'] = result['format'] or ('s16le' if result['output'] in ('-', 'pipe:1') else 'wav')
    if result['format'] not in ('s16le', 'wav'):
        raise ValueError('unsupported output format')
    duration = result['duration']
    if (result['rate'] <= 0 or result['channels'] not in (1, 2)
            or (duration is not None and (not math.isfinite(duration) or duration < 0))):
        raise ValueError('invalid capture dimensions')
    return result


def wav_header(rate, channels, size=0):
    # Match ffmpeg's LIST/INFO/ISFT layout, including the even-byte chunk padding.
    encoder = b'hfmodem stub\0'
    info = b'INFO' + struct.pack('<4sI', b'ISFT', len(encoder)) + encoder
    info += b'\0' * (len(encoder) % 2)
    metadata = struct.pack('<4sI', b'LIST', len(info)) + info
    return (struct.pack('<4sI4s4sIHHIIHH', b'RIFF', 36 + len(metadata) + size, b'WAVE',
                        b'fmt ', 16, 1, channels, rate, rate * channels * 2,
                        channels * 2, 16)
            + metadata + struct.pack('<4sI', b'data', size))


def main(argv=None):
    if os.environ.get('HFMODEM_COEXIST_HARNESS') != '1':
        print('ffmpeg stub requires HFMODEM_COEXIST_HARNESS=1', file=sys.stderr)
        return 2
    options = parse_args(sys.argv[1:] if argv is None else argv)
    if options['listing']:
        print('[AVFoundation indev] AVFoundation audio devices:\n'
              '[AVFoundation indev] [0] Coexist fake audio', file=sys.stderr)
        return 1
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    output = None
    is_pipe = options['output'] in ('-', 'pipe:1')
    size = 0
    try:
        coexist.checkpoint('ffmpeg_before_open')
        with shim.RawInputStream(device=options['device'], samplerate=options['rate'],
                                 channels=options['channels'], dtype='int16') as stream:
            if is_pipe:
                output = sys.stdout.buffer
            else:
                path = Path(options['output']).resolve()
                if Path(os.environ['HFMODEM_STATE_DIR']).resolve() not in path.parents:
                    raise ValueError('fake capture output must be under HFMODEM_STATE_DIR')
                output = path.open('wb')
            if options['format'] == 'wav':
                header = wav_header(options['rate'], options['channels'])
                output.write(header)
                output.flush()
                coexist.event('ffmpeg_header', size=len(header))
            coexist.checkpoint('ffmpeg_before_output')
            stopped.wait(float(os.environ.get('HFMODEM_COEXIST_FFMPEG_DELAY', '0')))
            if diagnostic := os.environ.get('HFMODEM_COEXIST_FFMPEG_STDERR'):
                print(diagnostic, file=sys.stderr, flush=True)
            remaining = (None if options['duration'] is None else
                         round(options['duration'] * options['rate']))
            while not stopped.is_set() and (remaining is None or remaining > 0):
                frames = 256 if remaining is None else min(256, remaining)
                data, _ = stream.read(frames)
                output.write(data)
                output.flush()
                size += len(data)
                if remaining is not None:
                    remaining -= frames
            if options['format'] == 'wav' and not is_pipe:
                output.seek(0)
                output.write(wav_header(options['rate'], options['channels'], size))
    except BrokenPipeError:
        return 1
    finally:
        if output is not None and not is_pipe:
            output.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
