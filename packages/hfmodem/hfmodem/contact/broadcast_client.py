# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Explicit unlinked broadcast workflow; never a connected application message."""
import secrets
from .broadcast import Broadcast, encode_ui


class BroadcastClient:
    def __init__(self, session):
        self.session = session
        self.selected = None
        self.active = False
        self.phase = None
        self.cancelled = False
        self.token = None
        self.result = None

    def help(self):
        self.session.say('/all cf selects the calling frequency; /all slot N selects a slot without transmitting; '
                         '/all TEXT broadcasts there; /all status shows state; '
                         '/all off returns to CF. Unlinked, no reception acknowledgement; '
                         '/stop cancels a pending send after any frame already in flight.')

    def status(self):
        place = ('no slot selected' if self.selected is None else
                 f'{self.selected / 1000:.3f} kHz USB dial')
        self.session.say(f'Broadcast: {self.phase or self.result or "idle"}; {place}; '
                         'remote reception is unconfirmed')

    def idle(self):
        from .varac import HostError
        s, c = self.session, self.session.client
        if (self.active or s.steps is not None or s._closing or s._quitting
                or c.closed or c.connected or c.pending or c.ptt or s.contact is not None
                or s.line_state() is not False or s._band_fault):
            raise HostError('Broadcast requires an idle, unlinked, verified-unkeyed session')
        if c.buffer not in (None, 0):
            raise HostError('Broadcast refused while connected-data bytes remain queued')

    def command(self, text):
        from .varac import HostError, State, slot_frequency
        s, c = self.session, self.session.client
        words = text.split()
        if not words or words == ['help']:
            self.help()
            return
        if words == ['status']:
            self.status()
            return
        self.idle()
        if words == ['cf'] or words[:1] == ['slot']:
            if words != ['cf'] and len(words) != 2:
                raise ValueError('usage: /all slot N')
            target = (s.calling_hz if words == ['cf'] else
                      slot_frequency(s.flow.band, int(words[1]), s.flow.codec))
            place = 'calling frequency' if words == ['cf'] else f'slot {words[1]}'
            c.request('LISTEN OFF')
            self.selected = None
            s.flow._state(State.CALLING)
            s.flow.tune(target)
            self.selected = target
            s.arm(True)
            self.result = None
            s.say(f'Broadcast {place} selected: {target / 1000:.3f} kHz USB dial; '
                  'no RF sent. Auto-answer and periodic beacons paused here; '
                  '/all TEXT sends, /all off returns to monitoring.')
            return
        if words == ['off']:
            self.selected = None
            s.monitor()
            s.say('Broadcast slot hold ended; normal monitoring and beacon schedule resumed')
            return
        if self.selected is None or self.selected != s.flow.dial:
            raise HostError('Select a channel first with /all cf or /all slot N; no broadcast sent')
        if c.busy is True:
            raise HostError('Broadcast not sent: channel is busy; no automatic retry')
        payload = text.encode('utf-8')
        if not text.isprintable():
            raise ValueError('Broadcast text must be one printable line')
        raw = encode_ui(c.mycall, 'ALL', payload, command=1)
        # Same qualified capacity/identity check as the host, before any request.
        from ..kestrel.rx.broadcast import packetize
        packetize(Broadcast(1, c.mycall, 'ALL', payload))
        self.active, self.cancelled, self.result = True, False, None
        self.phase = 'checking host broadcast capability; nothing queued'
        self.token = secrets.token_hex(8)
        s._run(self._send(raw, self.selected, c.link_generation))

    def cancel(self):
        if self.active:
            self.cancelled = True
            self.phase = 'cancelling; an already keyed frame cannot be undone'
            self.session.say(self.phase)
            return True
        return False

    def _valid(self, dial, generation):
        c = self.session.client
        return (not c.closed and not c.connected and not c.pending
                and c.link_generation == generation and self.session.flow.dial == dial)

    def _wait(self, after, predicate, timeout, dial, generation):
        from .varac import HostError
        c = self.session.client
        end = c.clock() + timeout
        while True:
            if self.cancelled:
                raise InterruptedError('broadcast cancelled')
            if not self._valid(dial, generation):
                raise HostError('Link or frequency changed during broadcast operation')
            for line in c.since(after):
                if line == 'WRONG':
                    raise HostError('Host rejected broadcast; no retry sent')
                if predicate(line):
                    return line
            if c.clock() >= end:
                raise TimeoutError('Broadcast host response timed out; reception unconfirmed')
            yield .1

    def _settle(self):
        """Cancel a potentially admitted frame; keep command/UI polling alive."""
        from .varac import HostError
        s, c = self.session, self.session.client
        s._closing = True
        after = c.send_command('VERSION')
        c.send_command('ABORT')
        end = c.clock() + 10
        while True:
            barrier = next((seq for seq, line in c._lines
                            if seq > after and line.startswith('VERSION ')), None)
            lines = c.since(barrier) if barrier is not None else []
            if 'WRONG' in lines:
                raise HostError('Host rejected broadcast cancellation; cannot resume')
            if 'OK' in lines and not c.ptt and s.line_state() is False:
                s._closing = False
                return
            if c.closed or c.clock() >= end:
                raise HostError('Broadcast cancellation/unkey unconfirmed; terminal must end')
            yield .1

    def _send(self, raw, dial, generation):
        from .varac import HostError
        s, c = self.session, self.session.client
        sent_command = False
        settling = False
        try:
            after = c.send_command('BROADCAST CAPS')
            caps = yield from self._wait(after, lambda line: line.startswith('BROADCAST CAPS '),
                                        10, dial, generation)
            if '1' not in caps.split()[2:]:
                raise HostError('Host does not support command-1 broadcasts; nothing sent')
            if c.ptt or s.line_state() is not False or c.busy is True:
                raise HostError('Channel/keying state changed; broadcast not sent')
            self.phase = 'waiting for host admission'
            sent_command = True  # A write failure can still leave uncertain admission.
            after = c.send_command(f'BROADCAST {self.token} 1 {raw.hex()}')
            prefix = f'BROADCAST {self.token} '
            queued = yield from self._wait(after, lambda line: line.startswith(prefix),
                                          10, dial, generation)
            if queued != prefix + 'QUEUED':
                raise HostError('Broadcast completion arrived without matching admission')
            s.say(f'Broadcast queued on {dial / 1000:.3f} kHz USB dial; not yet counted as transmitted')
            s.emit('broadcast_tx', token=self.token, state='queued', dial_hz=dial,
                   command=1, payload_hex=raw.hex())
            self.phase = 'queued; waiting for actual RF completion and unkey'
            result = yield from self._wait(after, lambda line: line.startswith(prefix)
                                            and line != prefix + 'QUEUED', 60, dial, generation)
            if result == prefix + 'SENT':
                if c.ptt or s.line_state() is not False:
                    raise HostError('Broadcast SENT arrived while keying remains unsettled')
                self.result = 'transmitted; remote reception unconfirmed'
                s.say(f'Broadcast transmitted on {dial / 1000:.3f} kHz USB dial; '
                      'remote reception is unconfirmed')
                s.emit('broadcast_tx', token=self.token, state='sent', dial_hz=dial)
            elif result.startswith(prefix + 'FAILED '):
                self.result = result[len(prefix):]
                s.say('Broadcast ' + self.result + '; no automatic retry')
                s.emit('broadcast_tx', token=self.token, state='failed', detail=self.result,
                       dial_hz=dial)
                if c.ptt or s.line_state() is not False:
                    settling = True
                    yield from self._settle()
            else:
                raise HostError('Unknown broadcast terminal response')
        except (HostError, TimeoutError, OSError) as error:
            self.result = str(error)
            if (sent_command and not settling and c.link_generation == generation
                    and not (c.connected or c.pending)):
                yield from self._settle()
            if isinstance(error, InterruptedError):
                s.say('Broadcast wait cancelled; any transmitted frame remains unacknowledged')
            else:
                raise
        finally:
            self.active, self.phase = False, None

    def received(self, message, *, dial, via='current channel'):
        s = self.session
        try:
            preview = repr(message.text)
        except UnicodeDecodeError:
            preview = f'{len(message.info)} opaque bytes, hex={message.info.hex()}'
        s.say(f'[broadcast {dial / 1000:.3f} kHz {via}] '
              f'{message.source} → {message.destination}: {preview} '
              '(RF-declared identity; no connected session)')
        s.emit('broadcast_rx', source=message.source, destination=message.destination,
               command=message.command, payload_hex=message.info.hex(), dial_hz=dial, via=via)
        s.mail_collection.notice(message, dial=dial)
