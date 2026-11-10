# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Explicit unlinked broadcast workflow; never a connected application message."""
from ..kestrel.broadcast import encode_ui
from ..kestrel.ui import UIFrame
from ..kestrel.tx.ui import packetize


class BroadcastClient:
    def __init__(self, session):
        self.session = session
        self.selected = None
        self.active = False
        self.phase = None
        self.cancelled = False
        self.possibly_outstanding = 0
        self._retune_command = None
        self._arm_restore_pending = False
        self.result = None

    def help(self):
        self.session.say('/all cf selects the calling frequency; /all slot N selects a slot without transmitting; '
                         '/all TEXT broadcasts there; /all status shows state; '
                         '/all off returns to CF; /all clear acknowledges possibly outstanding frames. '
                         'Unlinked, no reception acknowledgement; /stop stops waiting; '
                         'the host may still key this frame.')

    def status(self):
        place = ('no slot selected' if self.selected is None else
                 f'{self.selected / 1000:.3f} kHz USB dial')
        self.session.say(f'Broadcast: {self.phase or self.result or "idle"}; {place}; '
                         f'{self.possibly_outstanding} possibly outstanding; remote reception is unconfirmed')

    def reconnected(self):
        if self.possibly_outstanding:
            self.session.say(f'{self.possibly_outstanding} frames written to the host may still key '
                             'after reconnect; /all clear acknowledges them')

    def confirm_retune(self, command):
        return self.session.prepare_retune(command)

    def _confirm_retune(self, command):
        if not self.possibly_outstanding:
            return True
        if self._retune_command != command:
            self._retune_command = command
            self.session.say(f'{self.possibly_outstanding} frames written to the host may still key '
                             'on the new frequency. Proceed? /all yes acknowledges and proceeds; '
                             '/all no stays here')
        return False

    def warn_retune(self):
        """Automatic moves warn without acknowledging the host's outstanding queue."""
        if self.possibly_outstanding:
            self.session.say(f'{self.possibly_outstanding} frames written to the host may still key '
                             'on the new frequency. /all clear acknowledges them')

    def idle(self, *, arming=False):
        from .varac import HostError
        s, c = self.session, self.session.client
        for blocked, why in (
                (self.active, 'a broadcast operation is already in progress'),
                (s.steps is not None, f'{s.intent or "another operation"} is in progress'),
                (s._closing, 'the session is closing'),
                (s._quitting, 'the session is quitting'),
                (c.closed, 'the host connection is closed'),
                (c.connected, 'a link is up'),
                (c.pending, 'the host has a connection pending'),
                (c.ptt, 'the host reports PTT on'),
                (s.contact is not None, 'a contact is still open'),
                (s._band_fault, 'band setup is unverified')):
            if blocked:
                raise HostError(f'Broadcast refused: {why}')
        if arming and s.line_state() is None:
            # A session that is not answering holds no keyer, so its line reads only once armed.
            s.arm(True)
            if s.line_state() is not False:
                s.arm(s._auto_answer)
        line = s.line_state()
        if line is not False:
            raise HostError('Broadcast refused: the keying line is '
                            + ('HIGH' if line else 'unverified (no keyer held)'))
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
        if words == ['clear']:
            self.possibly_outstanding = 0
            self._retune_command = None
            s.say('Possibly outstanding frames acknowledged; the host queue is unchanged')
            return
        if words in (['yes'], ['no']):
            command, self._retune_command = self._retune_command, None
            if command is None:
                s.say('No broadcast retune acknowledgement is pending')
            elif words == ['yes']:
                self.possibly_outstanding = 0
                if callable(command):
                    command()
                else:
                    s.command(command)
            else:
                s.say('Retune cancelled; possibly outstanding frames retained')
            return
        selecting = words == ['cf'] or words[:1] == ['slot']
        self.idle(arming=selecting or words == ['off'])
        if selecting:
            if words != ['cf'] and len(words) != 2:
                raise ValueError('usage: /all slot N')
            target = (s.calling_hz if words == ['cf'] else
                      slot_frequency(s.flow.band, int(words[1]), s.flow.codec))
            place = 'calling frequency' if words == ['cf'] else f'slot {words[1]}'
            if target != self.selected and not self.confirm_retune('/all ' + text):
                return
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
            if not self.confirm_retune('/all off'):
                return
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
        self._start('ALL', payload, self.selected)

    def directed(self, recipient, payload=b'<SMP>', *, valid=lambda: True,
                 before_admit=lambda: None):
        from .varac import HostError
        s, c = self.session, self.session.client
        def admit():
            if c.busy is not False or c._writes or c.buffer != 0:
                raise HostError('Directed broadcast deferred: channel or data drain is unverified')
            before_admit()
        try:
            self.idle(arming=True)
            if c.busy is not False or c._writes or c.buffer != 0:
                raise HostError('Directed broadcast deferred: channel or data drain is unverified')
            if not valid():
                raise HostError('Directed broadcast is no longer eligible')
            c.request('LISTEN OFF')
            self._start(recipient, payload, s.flow.dial, valid=valid,
                        before_admit=admit, directed=True)
        except (HostError, ValueError, OSError):
            if s.steps is None and not self.active:
                self._restore_arm()
            raise

    def _restore_arm(self):
        s, c = self.session, self.session.client
        if c.connected or c.closed or s._closing or s._quitting:
            self._arm_restore_pending = False
            return
        if c.pending or c.ptt:
            self._arm_restore_pending = True
            return
        self._arm_restore_pending = False
        c.request('LISTEN CQ' if s._auto_answer else 'LISTEN OFF')
        s.arm(s._auto_answer)

    def restore_when_settled(self):
        """Finish a directed notice's cleanup without keeping its wait alive."""
        s, c = self.session, self.session.client
        if not self._arm_restore_pending:
            return
        if (c.connected or c.closed or s._closing or s._quitting
                or self.selected is not None or s._follow_tuned):
            self._arm_restore_pending = False
        elif not self.active and s.steps is None:
            self._restore_arm()

    def _start(self, recipient, payload, dial, *, valid=lambda: True,
               before_admit=lambda: None, directed=False):
        s, c = self.session, self.session.client
        frame = UIFrame.from_ax25(encode_ui(c.mycall, recipient, payload, command=1), command=1)
        packetize(frame)
        if c.kiss_socket is None and c._kiss_connect is None:
            from .varac import HostError
            raise HostError('the host has no KISS port; enable KISS in VARA')
        self.active, self.cancelled, self.result = True, False, None
        self.phase = 'preparing KISS write; nothing written'
        s._run(self._send(frame, dial, c.link_generation, valid=valid,
                          before_admit=before_admit, directed=directed))

    def cancel(self):
        if self.active:
            self.cancelled = True
            self.phase = 'stopping wait; the host may still key this frame'
            self.session.say(self.phase)
            return True
        return False

    def _valid(self, dial, generation):
        c = self.session.client
        return (not c.closed and not c.connected and not c.link_pending
                and c.link_generation == generation and self.session.flow.dial == dial)

    def _settle(self):
        self.session.say('Broadcast wait stopped; the host may still key this frame')
        yield from ()

    def _send(self, frame, dial, generation, *, valid=lambda: True,
              before_admit=lambda: None, directed=False):
        from .varac import HostError
        s, c = self.session, self.session.client
        attempted = False
        def writing():
            nonlocal attempted
            attempted = True
            self.possibly_outstanding += 1
            self._retune_command = None
        def written():
            self.phase = self.result = 'written; no matching transmission observed yet'
            s.say('Broadcast ' + self.phase)
        try:
            yield .1
            if self.cancelled:
                raise InterruptedError('broadcast wait cancelled')
            if not self._valid(dial, generation) or not valid():
                raise InterruptedError('broadcast is no longer eligible')
            if c.pending or c.ptt or s.line_state() is not False or c.busy is True:
                raise HostError('Channel/keying state changed; broadcast not sent')
            before_admit()
            if not self._valid(dial, generation) or not valid():
                raise InterruptedError('broadcast is no longer eligible after admission reservation')
            if (c.pending or c.ptt or s.line_state() is not False
                    or (c.busy is not False if directed else c.busy is True)
                    or directed and (c._writes or c.buffer != 0)):
                raise HostError('Channel/keying or drain state changed during admission reservation')
            wait = c.queued_kiss_steps(frame, cancelled=lambda: self.cancelled,
                valid=lambda: self._valid(dial, generation) and valid(),
                writing=writing, written=written)
            try:
                for delay in wait:
                    yield delay
                    # A queued frame cannot key while the ID holds the modem
                    # lock. Pause this wait after the short submission hold,
                    # then resume waiting for the same frame without resending.
                    if (not self.cancelled and self._valid(dial, generation) and valid()
                            and s._periodic_due(broadcast_wait=True)):
                        identification = s._identify_steps()
                        try:
                            for hint in identification:
                                yield hint
                                if self.cancelled:
                                    raise InterruptedError('broadcast wait cancelled')
                        finally:
                            identification.close()
            finally:
                wait.close()
            self.result = 'a host transmission matching this frame followed'
            s.say('Broadcast: ' + self.result + '; remote reception is unconfirmed')
        except InterruptedError as error:
            self.result = str(error)
            yield from self._settle()
        except (HostError, OSError) as error:
            self.result = str(error)
            if attempted:
                s.say('Broadcast write or wait failed; the host may still key this frame')
            raise HostError(str(error)) from error
        finally:
            self.active, self.phase = False, None
            if directed:
                self._restore_arm()

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
