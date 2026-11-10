# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded Tracker byte stream over an authenticated supervisor attachment."""
import select
import time

from hfmodem.core.managed_host import ManagedHost


def tracker_broker(runtime, *, secret, port=0):
    def serve(stream):
        attachment = runtime.attach('tracker')
        output = bytearray()
        progress = time.monotonic()
        try:
            stream.setblocking(False)
            while attachment.alive:
                readable, writable, _ = select.select(
                    [stream], [stream] if output else [], [], .02)
                if readable:
                    try:
                        data = stream.recv(8192)
                    except BlockingIOError:
                        data = None
                    if data == b'':
                        return
                    if data:
                        reply = runtime.feed(attachment, data)
                        if len(output) + len(reply) > 65536:
                            raise ConnectionAbortedError('Tracker stream output overflow')
                        if reply and not output:
                            progress = time.monotonic()
                        output += reply
                if writable and output:
                    try:
                        written = stream.send(output)
                    except BlockingIOError:
                        written = 0
                    if written:
                        del output[:written]
                        progress = time.monotonic()
                if output and time.monotonic() - progress >= 5.:
                    raise ConnectionAbortedError('Tracker stream output timed out')
            raise ConnectionAbortedError(attachment.failure or 'Tracker attachment retired')
        except ValueError:
            # Closing a descriptor from the lease revoker can invalidate select.
            # Other backend errors still fail the broker closed.
            if stream.fileno() >= 0:
                raise
        finally:
            # This is an owner command and must finish before the broker lets
            # a replacement lease attach. Do not swallow a failed retirement.
            runtime.detach_and_wait(attachment)

    return ManagedHost(serve, secret=secret, port=port, stream_roles=(3,),
                       start_owner=runtime.start, stop_owner=runtime.stop)
