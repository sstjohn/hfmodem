# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import codecs
import io
import locale
from pathlib import Path
import tempfile
import time

from . import claim


def output(process, paths, timeout, *, encoding=None, errors='strict'):
    deadline = time.monotonic() + timeout
    decoders = [io.IncrementalNewlineDecoder(
        codecs.getincrementaldecoder(encoding or locale.getpreferredencoding(False))(errors=errors), translate=True)
        for _ in paths]
    parts = [[] for _ in paths]
    code = None
    with paths[0].open('rb') as stdout, paths[1].open('rb') as stderr:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                code = None
                break
            code = process.poll()
            available = False
            for stream, decoder, output in zip((stdout, stderr), decoders, parts):
                block = stream.read(65536)
                available |= bool(block)
                output.append(decoder.decode(block))
            if not available:
                if code is not None:
                    break
                time.sleep(min(.05, remaining))
    for decoder, output in zip(decoders, parts):
        output.append(decoder.decode(b'', final=True))
    text = ''.join(''.join(output) for output in parts)
    if code is None:
        text += '\nNOT CONFIRMED: cleanup deadline expired\n'
    else:
        claim.record_exit(process._claim_child)
    return code, text


def run(descriptor, argv, timeout, **options):
    encoding, errors = options.get('encoding'), options.get('errors', 'strict')
    directory = claim.hardware.state_dir() / 'cleanup-logs'
    directory.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='down-', dir=directory))
    paths = (directory / 'stdout.log', directory / 'stderr.log')
    # The child's inherited file handles outlive every console-side reference.
    with paths[0].open('wb') as stdout, paths[1].open('wb') as stderr:
        process = claim.spawn(descriptor, argv, role='cleanup', text=True,
            stdout=stdout, stderr=stderr, **options)
    return output(process, paths, timeout, encoding=encoding, errors=errors)

