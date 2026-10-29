# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Hardware-free, stdlib terminal entrance and escaped transcript harness."""
import argparse
from dataclasses import asdict
import json
import fcntl
import io
import os
from pathlib import Path
import select
import sys
import time

from ..model import BusyError, Presentation
from ..presentation import plain


class _FDStreams:
    """Take raw-byte ownership of descriptors without using buffered readline.

    Capture *all* original flags before changing either descriptor: duplicated
    stdin/stdout descriptors may refer to the same open-file description.
    StringIO is the only supported stream without a file descriptor.
    """
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
        self.input_fd = self._fd(incoming)
        self.output_fd = self._fd(outgoing)
        self.flags = {}
        for fd in (self.input_fd, self.output_fd):
            if fd is not None and fd not in self.flags:
                self.flags[fd] = fcntl.fcntl(fd, fcntl.F_GETFL)

    @staticmethod
    def _fd(stream):
        if isinstance(stream, io.StringIO):
            return None
        try:
            fd = stream.fileno()
            if not isinstance(fd, int) or fd < 0:
                raise ValueError()
            return fd
        except (AttributeError, OSError, ValueError) as error:
            raise ValueError('local streams require a file descriptor or StringIO') from error

    def __enter__(self):
        try:
            for fd, flags in self.flags.items():
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self):
        first_error = None
        for fd, flags in self.flags.items():
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL, flags)
            except OSError as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def __exit__(self, *_):
        self.restore()

    def read(self):
        if self.input_fd is None:
            return self.incoming.read(1).encode('utf-8')
        return os.read(self.input_fd, 256)

    def write(self, data):
        if self.output_fd is None:
            self.outgoing.write(data.decode('utf-8'))
            return len(data)
        return os.write(self.output_fd, data)

    def wait(self, *, reading, writing):
        reads = [self.input_fd] if reading and self.input_fd is not None else []
        writes = [self.output_fd] if writing and self.output_fd is not None else []
        if reads or writes:
            select.select(reads, writes, [], .01)
        else:
            time.sleep(.01)


def run(core, peer, *, incoming=None, outgoing=None, transcript=None, ansi=False):
    """Pump core deadlines while terminal input/output is incomplete or blocked.

    Descriptor I/O is bounded, nonblocking UTF-8. This function owns the raw
    descriptors during the session; callers must not concurrently use buffered
    wrappers around them. Unknown potentially blocking non-fd streams are
    rejected. Blank native Enter means MORE. EOF forces cleanup and never
    fabricates game input. Hard close discards unwritten terminal output.

    Optional JSONL/plain transcripts are synchronous regular-file evidence;
    blocked transcript storage is outside the nonblocking terminal guarantee.
    Records describe queued events, with delivery dispositions recorded after
    complete writes or cancellation; completion does not assert human receipt.
    """
    incoming = sys.stdin if incoming is None else incoming
    outgoing = sys.stdout if outgoing is None else outgoing
    streams = _FDStreams(incoming, outgoing)
    presentation = Presentation('ansi16' if ansi else 'plain', page_bytes=2048, ansi_capable=ansi)
    evidence = readable = None
    sid = None
    pending = b''
    pending_event = None
    input_buffer = bytearray()
    eof = False
    counter = 0

    def record(direction, value):
        if evidence:
            evidence.write(json.dumps({'direction': direction, **value}, ensure_ascii=True, default=str) + '\n')
            evidence.flush()
        if readable and 'text' in value:
            readable.write(f'{direction}: {plain(value["text"])}\n')
            readable.flush()

    def cancelled():
        if pending_event is not None:
            record('delivery', {'output_id': pending_event.output_id, 'disposition': 'cancelled'})

    try:
        with streams:
            if transcript:
                evidence = Path(transcript).open('w', encoding='utf-8')
                readable = Path(str(transcript) + '.txt').open('w', encoding='utf-8')
            sid = core.open(peer, presentation, mode='local')
            while True:
                core.tick()
                s = core.session
                if s is None or s.id != sid or s.state == 'closed':
                    cancelled()
                    pending = b''
                    return 0
                if pending:
                    try:
                        written = streams.write(pending)
                    except BlockingIOError:
                        written = 0
                    except (BrokenPipeError, ConnectionError, OSError):
                        cancelled()
                        core.close(sid, 'local output failed')
                        return 0
                    pending = pending[written:]
                    if not pending and pending_event is not None:
                        event, pending_event = pending_event, None
                        core.output_done(sid, event.output_id, 'complete')
                        record('delivery', {'output_id': event.output_id, 'disposition': 'complete'})
                        if core.session.state != 'closed' and core.session.paused:
                            pending = b'[Enter] Continue  [R] Repeat  [Q] Stop reading\n'
                    if pending:
                        streams.wait(reading=False, writing=True)
                    continue
                events = core.poll(sid)
                if events:
                    event = events[0]
                    record('out', asdict(event))
                    if event.kind in ('close', 'exit'):
                        return 0
                    pending_event = event
                    pending = (event.text + '\n').encode('utf-8')
                    continue
                s = core.session
                if s.state in ('loading', 'probing', 'starting', 'admitting', 'stopping') or core.message_ui.busy:
                    streams.wait(reading=False, writing=False)
                    continue
                if b'\n' in input_buffer:
                    raw, separator, tail = bytes(input_buffer).partition(b'\n')
                    input_buffer[:] = tail if separator else b''
                    if raw.endswith(b'\r'):
                        raw = raw[:-1]
                    try:
                        action = raw.decode('utf-8', 'strict')
                    except UnicodeError:
                        core.close(sid, 'local malformed UTF-8 input')
                        return 0
                    if len(raw) > 256:
                        core.close(sid, 'local input exceeded command limit')
                        return 0
                    counter += 1
                    if not action:
                        action = 'BBS MORE'
                    record('in', {'input_id': str(counter), 'text': action})
                    core.input(sid, str(counter), action)
                    continue
                if eof:
                    core.close(sid, 'local EOF')
                    return 0
                try:
                    data = streams.read()
                except BlockingIOError:
                    data = None
                except (ConnectionError, OSError):
                    core.close(sid, 'local input failed')
                    return 0
                if data == b'':
                    eof = True
                elif data:
                    input_buffer.extend(data)
                    # At most one bounded read beyond the current line. Reject
                    # an unfinished overlong command rather than buffering it.
                    first = bytes(input_buffer).split(b'\n', 1)[0]
                    if len(first) > 257:  # allow CR before LF at the byte bound
                        core.close(sid, 'local input exceeded command limit')
                        return 0
                else:
                    streams.wait(reading=True, writing=False)
    except KeyboardInterrupt:
        if sid is not None:
            core.close(sid, 'local interrupt')
        return 130
    finally:
        if evidence:
            evidence.close()
        if readable:
            readable.close()
        if sid is not None:
            core.shutdown()
            stop = time.monotonic() + 6
            while not core.quiescent and time.monotonic() < stop:
                core.tick()
                time.sleep(0.01)
            core.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hfmodem bbs local', description=__doc__)
    parser.add_argument('--config', help='BBS config; defaults to HFMODEM_STATION [bbs] config')
    parser.add_argument('--peer', default='LOCAL')
    parser.add_argument('--transcript', help='write escaped JSONL and a readable .txt companion')
    parser.add_argument('--ansi16', action='store_true', help='explicitly opt in to safe color on a terminal')
    args = parser.parse_args(argv)
    try:
        from ..service import open_service
        from ..config import configured_path
        path = configured_path(args.config)
        if path is None:
            raise ValueError('supply --config or set [bbs] config in HFMODEM_STATION')
        with open_service(path) as service:
            return run(service.core, args.peer, transcript=args.transcript,
                       ansi=args.ansi16 and sys.stdout.isatty())
    except (OSError, ValueError, RuntimeError, BusyError) as error:
        print(f'BBS local: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
