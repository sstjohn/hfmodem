# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""SCS Tracker terminal and non-CRC JHOST1 presentation of AX.25.

This adapter is serviced by the packet runtime owner, not by the RF poll or a
serial read clock. Unlike JHOST4, JHOST1 has no alternate-bit retry cache.
Required RF parameters go through ``configure`` before committing their values.
The supported profile is one R300 connected channel with optional explicit-path
digipeating. DAMA is not supported.
"""
from collections import deque
import re
import time

from hfmodem.ax25.connected import State
from hfmodem.ax25.frame import address
from hfmodem.core import hostmode as hm
from .tracker_clock import TrackerClock


class TrackerHost:
    def __init__(self, engine, *, configure=None, clock=time.monotonic):
        self.engine, self.configure, self.clock = engine, configure, clock
        self._calendar = TrackerClock(clock)
        self.mode = 0
        self.channel = 0
        self._decoder = hm.Decoder(crc=False)
        self._line = bytearray()
        self._command = self._overlong = False
        self.events, self.rx = deque(), deque()
        self.failed = None
        self._drain_deadline = None
        self.values = {'A': 1, 'E': 1, 'F': 500, 'M': 'N', 'N': 10, 'O': 2,
                       'P': 32, 'T': 25, '%XR': 200, 'W': 10, 'R': 0, '@R': 'SCSTNC',
                       'Y': 1, 'Z': 0, '@D': 0, '@F': 0, '@I': 60,
                       '@T2': 150, '@T3': 18000, '@U': 0, '@V': 0,
                       '%A': 0, '%BR': 300, '%F': 1700, '%L': 1500, 'K': 0}
        # SCS 1.7 %F belongs to legacy 300-Bd FSK, not to RPR. Retain that
        # inactive parameter separately; the native RPR producer is fixed at
        # the documented %L default of 1500 Hz and cannot retune yet.
        self.ui_path = ('CQ', ())
        # Attachment defaults do not inherit forwarding from a retired client.
        self.engine.repeat_enabled = False
        self.engine.repeat_alias = self.values['@R']
        self.engine.repeats.clear()
        self.engine.state_changed = self._state
        self.engine.receive = self._receive
        self.engine.can_accept = self._can_accept

    def _can_accept(self):
        return not self.failed and not self.events and not self.rx

    def _state(self, state, remote, reason):
        if state in (State.CONNECTED, State.DISCONNECTED):
            if len(self.events) >= 16:
                self.failed = 'Tracker link output overflow'
                return
            self.events.append((state is State.DISCONNECTED,
                                f'(1) {state.value.upper()} to {remote}',
                                self._calendar.now()))
            if state is State.DISCONNECTED:
                self._drain_deadline = self.clock() + 30

    def _receive(self, payload):
        blocks = [payload[i:i + 128] for i in range(0, len(payload), 128)]
        if len(self.rx) + len(blocks) > 8:
            self.failed = 'Tracker received output overflow'
            self.engine.abort(self.failed)
            return
        self.rx.extend(blocks)
        if len(self.rx) >= 6:
            self.engine.set_busy(True)

    def service(self):
        if (self._drain_deadline is not None and self.clock() >= self._drain_deadline
                and (self.events or self.rx)):
            self.failed = 'Tracker closed contact output not drained'
        if self.failed and self.engine.active:
            self.engine.abort(self.failed)

    def attachment_lost(self):
        self.engine.listen = False
        self.engine.repeat_enabled = False
        self.engine.abort('Tracker attachment lost')
        self.events.clear()
        self.rx.clear()
        self.engine.set_busy(False)
        self.failed = None
        self._drain_deadline = None
        self.mode = 0
        self._line.clear()
        self._command = self._overlong = False
        self._decoder = hm.Decoder(crc=False)

    def feed(self, data):
        self.service()
        out = bytearray()
        for byte in data:
            if not self.mode and self.values['E']:
                # SCS 1.7 E: terminal echo replaces nonprinting input with
                # '.', except BELL/TAB. Hostmode bytes are always opaque.
                out.append(byte if 32 <= byte < 127 or byte in (7, 9) else ord('.'))
            if self.mode:
                for packet in self._decoder.feed(bytes([byte])):
                    out.extend(self._host_packet(packet))
            elif byte == 27:
                self._line.clear()
                self._command = True
                self._overlong = False
            elif byte == 13:
                if self._command:
                    if self._overlong:
                        text = 'INVALID COMMAND: line too long'
                    else:
                        try:
                            text = self.command(self._line.decode('ascii'))
                        except (ValueError, UnicodeError, BufferError) as exc:
                            text = f'INVALID COMMAND: {exc}'
                    if not self.mode:
                        out.extend(('\r' + text + '\r' + ('\n' if self.values['A'] else '')).encode('ascii'))
                self._line.clear()
                self._command = self._overlong = False
            elif self._command and byte != 10:
                if len(self._line) >= 256:
                    self._overlong = True
                elif not self._overlong:
                    self._line.append(byte)
        return bytes(out)

    def _host_packet(self, packet):
        ch = packet.channel
        try:
            if ch not in (0, 1):
                raise ValueError('only channels 0 and 1 supported')
            if not packet.is_command:
                if self.failed:
                    raise ValueError(self.failed)
                if ch == 1:
                    self.engine.write(packet.data)
                else:
                    destination, via = self.ui_path
                    self.engine.unproto(destination, packet.data, via)
                return hm.unwrap(hm.reply(ch, hm.OK))
            text = packet.text.upper().strip()
            if text in ('G', 'G0', 'G1'):
                if ch == 1:
                    if (text != 'G0' and self.events
                            and (not self.events[0][0] or not self.rx)):
                        _, event, stamp = self.events.popleft()
                        if self.values['K']:
                            event += ' - ' + self._calendar.format(stamp)
                        if not self.events and not self.rx:
                            self._drain_deadline = None
                        return hm.unwrap(hm.reply_text(ch, hm.LINK, event))
                    if text != 'G1' and self.rx:
                        body = self.rx.popleft()
                        if len(self.rx) < 4:
                            self.engine.set_busy(False)
                        return hm.unwrap(hm.reply(ch, hm.DATA, body))
                return hm.unwrap(hm.reply(ch, hm.OK))
            if text == 'L':
                # WA8DED: link messages, receive frames, unsent frames,
                # unacknowledged frames, retries, link state. Trimode indexes
                # field four; Express indexes field three. Never merge them.
                state = {State.DISCONNECTED: 0, State.CONNECTING: 1,
                         State.DISCONNECTING: 2, State.CONNECTED: 4}[self.engine.state]
                body = f'{len(self.events)} {len(self.rx)} {len(self.engine.pending)} {len(self.engine.offered)} {self.engine.retries} {state}'
                return hm.unwrap(hm.reply_text(ch, hm.MSG, body))
            result = self.command(text, channel=ch)
            return hm.unwrap(hm.reply_text(ch, hm.MSG, result) if result else hm.reply(ch, hm.OK))
        except (ValueError, UnicodeError, BufferError) as exc:
            return hm.unwrap(hm.reply_text(ch, hm.FAIL, str(exc)))

    def command(self, text, *, channel=None):
        text = text.upper().strip()
        ch = self.channel if channel is None else channel
        if text in ('JHOST0', 'JHOST1'):
            self.mode = int(text[-1])
            self._decoder = hm.Decoder(crc=False)
            return ''
        match = re.fullmatch(r'(%BR|%XR|%F|%L|%A|@T2|@T3|@[DFIRUV]|[A-Z])(.*)', text)
        if not match:
            raise ValueError('unsupported command')
        key, arg = match.groups()
        arg = arg.strip()
        if key == 'S':
            if arg:
                value = int(arg)
                if value not in (0, 1):
                    raise ValueError('only channels 0 and 1 supported')
                self.channel = value
            return f'CHANNEL {self.channel}'
        if key == 'I':
            if not arg:
                return self.engine.mycall
            address(arg)
            if self.engine.active or self.engine._transmitting:
                raise ValueError('identity change during contact')
            self.engine.repeats.clear()
            self.engine.mycall = arg
            self.engine.listen = self.values['Y'] == 1 and arg != 'XXXXX'
            return ''
        if key == 'C':
            if not arg:
                return self.engine.remote
            calls = arg.split()
            if len(calls) > 1 and calls[1] in ('V', 'VIA'):
                del calls[1]
            for call in calls:
                address(call)
            if ch == 0:
                self.ui_path = (calls[0], tuple(calls[1:]))
            else:
                if not self._can_accept():
                    raise ValueError('prior contact output not drained')
                self.engine.connect(calls[0], calls[1:])
            return ''
        if key == 'D':
            if arg:
                raise ValueError('D takes no argument')
            self.engine.disconnect()
            return ''
        if key == 'V' and not arg:
            return 'hfmodem Tracker R300 (modulo-8 AX.25)'
        if key not in self.values:
            raise ValueError('unsupported command')
        if key == 'K':
            if not arg:
                return f"{self.values['K']} {self._calendar.format(self._calendar.now())}"
            if arg.isascii() and arg.isdecimal():
                value = int(arg)
                if value not in (0, 1, 2):
                    raise ValueError('K timestamp mode must be 0..2')
                self.values['K'] = value
            else:
                self._calendar.set(arg)
            return ''
        if not arg:
            return str(self.values[key])
        if key == '@R':
            address(arg)
            if self.engine.active or self.engine._transmitting:
                raise ValueError('repeater identity change during contact')
            self.engine.repeats.clear()
            self.engine.repeat_alias = arg
            self.values[key] = arg
            return ''
        if key == 'M':
            if arg != 'N':
                raise ValueError('monitor profile supports MN only')
            self.values[key] = arg
            return ''
        value = int(arg)
        limits = {'A': (0, 1), 'E': (0, 1), 'F': (1, 1500), 'N': (0, 127),
                  'O': (1, 7), 'P': (0, 255), 'T': (0, 500), '%XR': (30, 3000),
                  'W': (0, 127), 'R': (0, 1), 'Y': (0, 1), 'Z': (0, 0),
                  '@D': (0, 0), '@F': (0, 0), '@I': (1, 256), '@T2': (0, 100000),
                  '@T3': (0, 100000), '@U': (0, 0), '@V': (0, 0), '%A': (0, 0),
                  '%BR': (300, 300), '%F': (1000, 3000), '%L': (1500, 1500)}
        lo, hi = limits[key]
        if not lo <= value <= hi:
            raise ValueError(f'{key}: supported range {lo}..{hi}')
        if self.engine.active and key in ('F', 'N', 'O', '@I', '@T2', '@T3', '%BR', '%XR', 'T', 'P', 'W'):
            raise ValueError('configuration change during contact')
        if key in ('%BR', '%XR', 'T', 'P', 'W'):
            if self.configure is None:
                raise ValueError('RF control backend unavailable')
            self.configure(key, value)
        if key == 'F':
            self.engine.rtt = value * (.5 if value < 16 else .01)
        elif key == 'N':
            self.engine.retry_limit = value
        elif key == 'O':
            self.engine.window = value
        elif key == '@I':
            self.engine.ipoll = value
        elif key == '@T2':
            self.engine.t2 = value * .01
        elif key == '@T3':
            self.engine.t3 = value * .01
        elif key == 'Y':
            self.engine.listen = value == 1 and self.engine.mycall != 'XXXXX'
            if not self.engine.listen:
                self.engine.repeats.clear()
        elif key == 'R':
            self.engine.repeat_enabled = bool(value)
            if not value:
                self.engine.repeats.clear()
        self.values[key] = value
        return ''
