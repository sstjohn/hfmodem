# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""One modulo-8 AX.25 connection, independent of the packet waveform.

Implements SABM (AX.25 2.0), implicit REJ, RR/RNR, cumulative acknowledgement,
and DISC/UA. Optional explicit-path repeating uses AX.25 2.2 section 3.12.4;
it has separate bounded output and earns no connected-host credit.
Extended sequencing and DAMA are not advertised. Sources: TAPR AX.25 2.2
sections 3, 4 and 6 and the SCS Tracker 1.7
F/N/O/@I/@T2/@T3 descriptions. Only the runtime owner calls these methods.

``transmit`` accepts an FCS-free frame and returns True when its transmission
is accepted. A deferred/refused frame must return False; timers then do not
pretend it reached the channel. The station binding must report completion
before accepting another frame from this engine.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum
import time

from .frame import Frame, address, parse, ui_frame

SABM, UA, DISC, DM, FRMR = 0x2F, 0x63, 0x43, 0x0F, 0x87
RR, RNR, REJ = 1, 5, 9


def callsign(raw):
    name = ''.join(chr(c >> 1) for c in raw[:6]).rstrip()
    ssid = (raw[6] >> 1) & 15
    return f'{name}-{ssid}' if ssid else name


class State(Enum):
    DISCONNECTED = 'disconnected'
    CONNECTING = 'connecting'
    CONNECTED = 'connected'
    DISCONNECTING = 'disconnecting'


@dataclass
class Offered:
    sequence: int
    payload: bytes
    sent_at: float
    retried: bool = False


class Connection:
    def __init__(self, mycall, transmit, *, state_changed=lambda *_: None,
                 receive=lambda *_: None, confirmed=lambda *_: None,
                 can_accept=lambda: True, clock=time.monotonic):
        address(mycall)
        self.mycall, self.remote, self.via = mycall, '', ()
        self.transmit, self.state_changed = transmit, state_changed
        self.receive, self.confirmed, self.can_accept = receive, confirmed, can_accept
        self.clock = clock
        self.admit_contact = lambda: True
        self.state = State.DISCONNECTED
        self.listen = True
        self.window, self.paclen, self.retry_limit = 2, 128, 10
        self.max_info = 256
        self.rtt, self.t2, self.t3, self.ipoll = 5.0, 1.5, 180.0, 60
        self.buffer_limit = 65536
        self.accepted = self.delivered = self.discarded = 0
        self.last_unconfirmed = b''
        self.retired = deque(maxlen=16)
        self.generation = 0
        self.pending, self.offered, self.controls = deque(), deque(), deque()
        self.repeat_enabled, self.repeat_alias = False, 'SCSTNC'
        self.repeats = deque()
        self.vs = self.vr = self.retries = 0
        self.peer_busy = self.local_busy = False
        self._reject = False
        self._close = False
        self._retry = deque()
        self._deadline = self._ack_at = None
        self._last_activity = clock()
        self._polling = False
        self._transmitting = False
        self._transmit_started = None

    @property
    def pending_bytes(self):
        return sum(map(len, self.pending)) + sum(len(x.payload) for x in self.offered)

    @property
    def outstanding_frames(self):
        return len(self.pending) + len(self.offered)

    @property
    def active(self):
        return self.state is not State.DISCONNECTED

    def _state(self, state, reason=''):
        self.state = state
        self.state_changed(state, self.remote, reason)

    def _fresh(self):
        self.generation += 1
        self.accepted = self.delivered = self.discarded = 0
        self.last_unconfirmed = b''
        self.pending.clear()
        self.offered.clear()
        self.controls.clear()
        self._retry.clear()
        self.vs = self.vr = self.retries = 0
        self.peer_busy = self.local_busy = self._reject = self._close = False
        self._deadline = self._ack_at = None
        self._last_activity = self.clock()
        self._polling = False

    def connect(self, remote, via=()):
        if self.active:
            raise ValueError('connection already active')
        address(remote)
        if len(via) > 8:
            raise ValueError('at most eight repeaters')
        for call in via:
            address(call)
        if not self.admit_contact():
            raise ValueError('STATION CHANNEL OWNER BUSY')
        self._fresh()
        self.remote, self.via = remote, tuple(via)
        self._state(State.CONNECTING)
        self._control(SABM | 0x10, True)

    def write(self, payload):
        payload = bytes(payload)
        if self.state is not State.CONNECTED or self._close:
            raise ValueError('connection not accepting payload')
        if self.pending_bytes + len(payload) > self.buffer_limit:
            raise BufferError('AX.25 host buffer full')
        self.pending.extend(payload[i:i + self.paclen] for i in range(0, len(payload), self.paclen))
        self.accepted += len(payload)

    def disconnect(self):
        if self.state is State.DISCONNECTED:
            return
        if self._close or self.state in (State.CONNECTING, State.DISCONNECTING):
            self.abort('forced host disconnect')
            return
        self._close = True
        self._start_close()

    def _start_close(self):
        if self._close and not self.pending and not self.offered and self.state is State.CONNECTED:
            self._deadline = None
            self.retries = 0
            self._state(State.DISCONNECTING)
            self._control(DISC | 0x10, True)

    def abort(self, reason='host abort'):
        self.repeats.clear()
        if self.active:
            # A local abort is explicit loss, never peer confirmation.
            self._finish(reason)
        else:
            self.controls.clear()
            self._retry.clear()

    def _finish(self, reason):
        self.last_unconfirmed = b''.join(x.payload for x in self.offered) + b''.join(self.pending)
        self.retired.append({'generation': self.generation, 'remote': self.remote,
                             'accepted': self.accepted, 'confirmed': self.delivered,
                             'unconfirmed': self.last_unconfirmed, 'reason': reason})
        self.discarded += len(self.last_unconfirmed)
        self.pending.clear()
        self.offered.clear()
        self.controls.clear()
        self._retry.clear()
        self._deadline = self._ack_at = None
        self._close = self._polling = False
        self._state(State.DISCONNECTED, reason)

    def unproto(self, destination, payload, via=()):
        if len(payload) > self.paclen:
            raise ValueError('UI payload exceeds packet length')
        self.controls.append((ui_frame(self.mycall, destination, bytes(payload), tuple(via)), False))

    def _frame(self, control, command, payload=b'', *, remote=None, via=None):
        calls = (remote or self.remote, self.mycall, *(self.via if via is None else via))
        addresses = tuple(address(call, last=i == len(calls) - 1,
                                  flag=(command if i == 0 else not command if i == 1 else False))
                          for i, call in enumerate(calls))
        return Frame(addresses, control, 0xF0 if control & 1 == 0 else None, payload).encode()

    def _control(self, control, command=False, *, remote=None, via=None):
        if len(self.controls) >= 32:
            self.abort('AX.25 control output overflow')
            return
        waits = command and bool(control & 0x10)
        item = (self._frame(control, command, remote=remote, via=via), waits)
        if item not in self.controls:
            self.controls.append(item)

    def _rr(self, final=False, command=False):
        self._control((RNR if self.local_busy else RR) | self.vr << 5 | int(final) << 4, command)
        self._ack_at = None

    def set_busy(self, on):
        if self.local_busy != bool(on):
            self.local_busy = bool(on)
            if self.state is State.CONNECTED:
                self._rr()

    def _ack(self, nr):
        first = self.offered[0].sequence if self.offered else self.vs
        count = (nr - first) % 8
        if count > len(self.offered):
            self._finish('invalid peer acknowledgement')
            return False
        now = self.clock()
        for _ in range(count):
            frame = self.offered.popleft()
            if not frame.retried:
                self.rtt = max(.1, .875 * self.rtt + .125 * (now - frame.sent_at))
            self.delivered += len(frame.payload)
            self.confirmed(frame.payload)
        if count:
            self.retries = 0
            self._retry.clear()
            self._deadline = now + 2 * self.rtt if self.offered or self._polling else None
        return True

    def on_frame(self, raw):
        try:
            frame = parse(raw)
        except ValueError:
            return
        dest, source = map(callsign, frame.addresses[:2])
        self._repeat(frame)
        # An unfinished path cannot reach the local endpoint merely because a
        # copy was decoded before its next repeater.
        if dest != self.mycall or any(not a[6] & 0x80 for a in frame.addresses[2:]):
            return
        command = bool(frame.addresses[0][6] & 0x80) and not frame.addresses[1][6] & 0x80
        response = not frame.addresses[0][6] & 0x80 and bool(frame.addresses[1][6] & 0x80)
        if not command and not response:
            return                 # not an AX.25 2.x command/response
        if not self.active and not self.admit_contact():
            return                 # another station owner; no competing DM/UA
        control, pf = frame.control, bool(frame.control & 0x10)
        kind = control & 0xEF
        route = tuple(callsign(a) for a in reversed(frame.addresses[2:]))
        if kind in (SABM, 0x6F):
            same = self.active and source == self.remote
            duplicate_setup = (same and self.state is State.CONNECTED
                               and self.vs == self.vr == self.accepted == 0)
            if kind == SABM and command and duplicate_setup:
                self._control(UA | int(pf) << 4)
                return
            if kind == SABM and command and same and self.state is State.CONNECTED:
                # A reset after traffic is an explicit failed contact. Do not
                # rebind accepted bytes or hide loss behind a zero frame count.
                # A subsequent setup waits for the host's old output to drain.
                self._finish('peer reset after contact traffic')
                self._control(DM | int(pf) << 4, remote=source, via=route)
                return
            eligible = (self.listen or same and self.state is State.CONNECTING) and self.can_accept()
            if kind != SABM or not command or not eligible:
                if command:
                    self._control(DM | int(pf) << 4, remote=source, via=route)
                return
            if self.active and source != self.remote:
                self._control(DM | int(pf) << 4, remote=source, via=route)
                return
            self._fresh()
            self.remote, self.via = source, route
            self._control(UA | int(pf) << 4)
            self._state(State.CONNECTED)
            return
        if source != self.remote or self.state is State.DISCONNECTED:
            if command and pf:
                self._control(DM | 0x10, remote=source, via=route)
            return
        self._last_activity = self.clock()
        if kind == UA and response and pf:
            if self.state is State.CONNECTING:
                self.controls.clear()
                self._deadline = None
                self.retries = 0
                self._state(State.CONNECTED)
            elif self.state is State.DISCONNECTING:
                self._finish('peer confirmed disconnect')
            return
        if kind in (DM, FRMR) and response:
            self._finish('peer refused or reset connection')
            return
        if kind == DISC and command:
            # Retain the reply after local retirement; old payload is disposed.
            self._finish('peer disconnect')
            self._control(UA | int(pf) << 4)
            return
        if self.state is not State.CONNECTED:
            return
        if control & 3 in (0, 2, 1):
            if not self._ack(control >> 5):
                return
            if response and pf and self._polling:
                self._polling = False
                self.retries = 0
                self._deadline = self.clock() + 2 * self.rtt if self.offered else None
        if control & 1 == 0:
            if not command or frame.pid != 0xF0 or len(frame.info) > self.max_info:
                return
            ns = (control >> 1) & 7
            if self.local_busy:
                self._rr(pf)
            elif ns == self.vr:
                self.vr = (self.vr + 1) % 8
                self._reject = False
                self.receive(frame.info)
                if pf:
                    self._rr(True)
                elif self._ack_at is None:
                    self._ack_at = self.clock() + self.t2
            elif ns == (self.vr - 1) % 8:
                self._rr(pf)        # duplicated RF copy, no host delivery
            elif not self._reject or pf:
                self._reject = True
                self._control(REJ | self.vr << 5 | int(pf) << 4)
        elif control & 3 == 1:
            code = control & 15
            if code not in (RR, RNR, REJ):
                self._finish('unsupported supervisory frame')
                return
            was_busy = self.peer_busy
            self.peer_busy = code == RNR
            if code == REJ or was_busy and not self.peer_busy:
                self._retry = deque(x.sequence for x in self.offered)
            if command and pf:
                self._rr(True)
        self._start_close()

    def _repeat(self, frame):
        if not self.repeat_enabled or not self.listen or len(frame.info) > self.max_info:
            return
        path = frame.addresses[2:]
        for index, raw in enumerate(path, 2):
            if not raw[6] & 0x80:
                # Only the first uncompleted hop may act. A noncontiguous H
                # sequence is malformed and must not create a forwarding loop.
                if (callsign(raw) not in (self.mycall, self.repeat_alias)
                        or any(a[6] & 0x80 for a in frame.addresses[index+1:])):
                    return
                addresses = list(frame.addresses)
                addresses[index] = raw[:6] + bytes([raw[6] | 0x80])
                output = Frame(tuple(addresses), frame.control, frame.pid, frame.info).encode()
                if len(self.repeats) < 32 and output not in self.repeats:
                    if self.admit_contact():
                        self.repeats.append(output)
                return

    def _send_repeat(self):
        if not self.repeat_enabled or not self.listen:
            self.repeats.clear()
            return
        if self.repeats and self.transmit(self.repeats[0]):
            self.repeats.popleft()

    def transmission_started(self):
        """A station backend accepted a frame but has not finished emitting it."""
        self._transmitting = True
        self._transmit_started = self.clock()

    def transmission_finished(self, okay):
        """Resume T1 from actual emission completion, or retain explicit loss."""
        if not self._transmitting:
            return
        self._transmitting = False
        if not okay:
            self.abort('packet transmission failed')
        elif self._deadline is not None:
            wait = max(0, self._deadline - self._transmit_started)
            self._deadline = self.clock() + wait
        self._transmit_started = None

    def tick(self):
        if self._transmitting:
            return
        now = self.clock()
        if self._ack_at is not None and now >= self._ack_at:
            self._rr()
        if self._deadline is not None and now >= self._deadline:
            if self.retry_limit and self.retries >= self.retry_limit:
                self._finish('retry limit exhausted')
                return
            self.retries += 1
            self._deadline = None
            if self.state in (State.CONNECTING, State.DISCONNECTING):
                self._control((SABM if self.state is State.CONNECTING else DISC) | 0x10, True)
            elif self.state is State.CONNECTED:
                self._polling = True
                self._rr(True, True)
                if not self.peer_busy:
                    self._retry = deque(x.sequence for x in self.offered)
        elif (self.state is State.CONNECTED and self._deadline is None and self.t3
              and now - self._last_activity >= self.t3):
            self._polling = True
            self._rr(True, True)
        if self.controls:
            raw, waits = self.controls[0]
            if self.transmit(raw):
                self.controls.popleft()
                self._last_activity = now
                if waits:
                    self._deadline = now + min(120, 2 * self.rtt * 2 ** min(self.retries, 5))
            return
        if self.state is not State.CONNECTED or self.peer_busy:
            self._send_repeat()
            return
        if self._retry:
            sequence = self._retry[0]
            frame = next((x for x in self.offered if x.sequence == sequence), None)
            if frame is None:
                self._retry.popleft()
                return
            if self.transmit(self._frame(frame.sequence << 1 | self.vr << 5, True, frame.payload)):
                frame.retried = True
                self._retry.popleft()
                self._deadline = now + 2 * self.rtt
            return
        if self.pending and len(self.offered) < self.window:
            blob = self.pending[0]
            poll = len(blob) <= self.ipoll and (len(self.pending) == 1 or len(self.offered) + 1 == self.window)
            control = self.vs << 1 | self.vr << 5 | int(poll) << 4
            if self.transmit(self._frame(control, True, blob)):
                self.pending.popleft()
                self.offered.append(Offered(self.vs, blob, now))
                self.vs = (self.vs + 1) % 8
                if self._deadline is None:
                    self._deadline = now + 2 * self.rtt
                self._ack_at = None     # N(R) piggybacks the pending ACK
                self._last_activity = now
            return
        self._start_close()
        if not self.controls:
            self._send_repeat()
