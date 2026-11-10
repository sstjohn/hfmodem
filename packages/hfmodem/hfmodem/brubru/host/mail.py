# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The station's B2F byte pipe on the connected-packet owner thread."""
from collections import deque
import time

from hfmodem.ax25.connected import State
from hfmodem.winlink.client import MailClient, summarize, write_inbox


STOP_CLOSE_S = 2.
RETIRE_MARGIN_S = 1.


class MailHost:
    def __init__(self, engine, session):
        self.engine = engine
        self.pending = deque()
        self.closing = self.connected = self.down = False
        self.alive = True
        self.client = MailClient(session, self.send)
        engine.state_changed = self._state
        engine.receive = self.client.on_link_data
        engine.can_accept = lambda: self.alive
        engine.listen = False

    def send(self, data):
        self.pending.append(bytes(data))

    def _state(self, state, remote, reason):
        if state is State.CONNECTED:
            self.connected = True
            print(f'  CONNECTED {remote}', flush=True)
            self.client.link_up()
        elif state is State.DISCONNECTED:
            self.down = True
            print('  link down', flush=True)

    def disconnect(self):
        self.closing = True

    def service(self):
        if self.down or not self.alive:
            return
        if self.engine.state is State.CONNECTED:
            room = self.engine.buffer_limit - self.engine.pending_bytes
            while self.pending and room > 0:
                data = self.pending.popleft()
                chunk = data[:room]
                self.engine.write(chunk)
                room -= len(chunk)
                if len(chunk) < len(data):
                    self.pending.appendleft(data[len(chunk):])
        if self.closing and not self.pending:
            self.closing = False
            self.engine.disconnect()

    def attachment_lost(self):
        self.alive = False
        self.pending.clear()
        self.engine.listen = False
        self.engine.abort('mail attachment lost')


def run_exchange(runtime, session, gateway, *, timeout=1200, stopped=lambda: False,
                 outdir=None, close_timeout=30., burst_limit=30.04):
    """Connect, exchange, close and retire before releasing the station lane."""
    attachment = runtime.attach('mail', session=session)
    host = attachment.host
    outbound = len(session._out)
    deadline = time.monotonic() + timeout
    ended = 'the run was cut short before the link was decided'
    ok = False

    def status():
        return runtime._call(lambda: (host.connected, host.down, session.done,
                                      session.failure, len(session.sent_mids)))

    try:
        print(f'mail: {session.mycall.upper()} -> {gateway.upper()} — connecting '
              f'as {session.sid}', flush=True)
        runtime._call(lambda: runtime.engine.connect(gateway.upper()))
        while True:
            connected, down, done, failure, sent = status()
            stopping = stopped()
            if (stopping or down or not attachment.alive or done
                    or time.monotonic() >= deadline):
                break
            time.sleep(.05)
        if stopping:
            ended = 'the run was stopped and the link was closed'
        elif not connected:
            ended = 'no link — nothing answered the connect request'
            print('mail: NO LINK', flush=True)
        elif failure or attachment.failure:
            ended = 'the exchange failed and the link was closed'
            print('mail: exchange failed — closing the link', flush=True)
        elif done:
            ok = sent == outbound
            ended = ('the exchange finished and the link was closed' if ok else
                     'not every outbound message was sent')
        elif down:
            ended = 'the link went down mid-exchange'
        else:
            ended = f'out of time after {timeout:.0f} s and the link was closed'
            print('mail: out of time — closing the link', flush=True)
    finally:
        try:
            if attachment.alive:
                runtime._call(host.disconnect)
                end = time.monotonic() + close_timeout
                settled = False
                while attachment.alive:
                    if stopped():
                        end = min(end, time.monotonic() + STOP_CLOSE_S)
                    settled = runtime._call(lambda: host.down and not runtime.active)
                    if settled or time.monotonic() >= end:
                        break
                    time.sleep(.05)
                _, down, done, failure, sent = status()
                drained = runtime._call(lambda: not host.pending and not runtime.engine.discarded
                                         and runtime.engine.pending_bytes == 0)
                if not attachment.alive or attachment.failure:
                    ended = 'the packet attachment failed before the link closed'
                elif not settled:
                    ended = 'link close did not finish before the deadline'
                    runtime._call(lambda: runtime.engine.abort('mail close deadline'))
                elif not drained:
                    ended = 'the link closed with undelivered mail bytes'
                elif failure or attachment.failure:
                    ended = 'the exchange failed and the link was closed'
                elif stopped():
                    ended = 'the run was stopped and the link was closed'
                ok = (ok and settled and down and done and not failure and sent == outbound and drained
                      and attachment.alive and not attachment.failure and not stopped())
            else:
                ok = False
                ended = 'the packet attachment failed before the link closed'
        except BaseException:
            ok = False
            ended = 'link close failed before validation'
            raise
        finally:
            try:
                # Abort cancels queued work; a committed burst must finish on air.
                runtime.detach_and_wait(attachment, timeout=burst_limit + RETIRE_MARGIN_S)
            except BaseException:
                ok = False
                ended = 'packet RF work did not retire successfully'
                raise
            finally:
                print('mail: exchange complete — closing' if ok else
                      'mail: exchange failed — ' + ended, flush=True)
                print(f'session ended: {ended}', flush=True)
                print(summarize(session), flush=True)
                if outdir is not None:
                    for path in write_inbox(session, outdir):
                        print(f'mail: wrote {path}', flush=True)
    return 0 if ok else 1
