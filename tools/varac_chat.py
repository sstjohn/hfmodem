#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Kestrel chat terminal using the VarAC protocol, launched under the console claim."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
import io
import json
import os
from pathlib import Path
import queue
import re
import select
import sys
import threading
import time

from hfmodem.contact.varac import (BANDWIDTHS, CALLING_HZ, DEFAULT_BW, Client, Flow, HostError, Identification,
                                   Mailbox, Session, emit_tag, utc)
from hfmodem.core import posix
from hfmodem.station.contactlog import ContactLog

termios = posix.module("termios")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ui', choices=('auto', 'line'), default='auto')
    p.add_argument('--mycall', type=str.upper, required=True,
                   help='the digital call, as the fly path takes it: a tactical call such as X-W9SSJ is fine')
    p.add_argument('--idcall', help='CW call for the closing ID; default IDCALL, [station] idcall, then --mycall')
    p.add_argument('--band', type=int, choices=tuple(CALLING_HZ), required=True)
    p.add_argument('--dial', type=int, help="this session's calling frequency in place of the band's")
    p.add_argument('--bw', type=int, choices=BANDWIDTHS, default=DEFAULT_BW,
                   help=f'VARA bandwidth for monitoring, /cq and /call (default {DEFAULT_BW})')
    for field in ('name', 'qth', 'loc', 'services'):
        p.add_argument('--' + field)
    p.add_argument('--bbs-directory', help='serve this directory and advertise BBS; overrides station.toml')
    p.add_argument('--no-bbs', action='store_true', help='disable the configured BBS export')
    p.add_argument('--bbs-config', type=Path, help='shared BBS config; overrides station.toml [bbs] config')
    p.add_argument('--no-bbs-service', action='store_true', help='disable interactive BBS service')
    p.add_argument('--pskreporter', action='store_true',
                   help='opt in to public PSKReporter CQ/beacon uploads; requires --loc GRID')
    p.add_argument('--monitor-wide', action='store_true',
                   help='set the dial off each channel worked by the offset that fits the most VarAC '
                        'channels in the passband, so the slot monitor hears more of them; our audio '
                        'is moved to match both ways')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--rigctld', default='127.0.0.1:4532')
    p.add_argument('--expect-model', required=True)
    p.add_argument('--ptt-device', required=True)
    p.add_argument('--tx-device', required=True)
    p.add_argument('--rx-device', required=True)
    from hfmodem.core.recording import add_recording_argument
    add_recording_argument(p)
    p.add_argument('--record', type=Path)
    p.add_argument('--no-record', action='store_true')
    return p


def dial_offset(args):
    from hfmodem.contact.slots import wide_offset
    return wide_offset() if args.monitor_wide else 0


def validate(args):
    from hfmodem.core import identification
    if getattr(args, 'pskreporter', False):
        from hfmodem.station.pskreporter import Config
        Config(args.mycall, args.loc or '', enabled=True)
    args.identity = identification.resolve(args.mycall, args.idcall)
    if args.dial is not None:
        # Every calling-frequency decision reads this table, so the session moves as one.
        CALLING_HZ[args.band] = args.dial
    args.identity.announce()
    for field in ('name', 'qth', 'loc', 'services'):
        value = getattr(args, field, None)
        if value is not None:
            emit_tag('S' if field == 'services' else field.upper(), value)


class _ConnectionLine(str):
    def __new__(cls, value, recording_id):
        line = super().__new__(cls, value)
        line.recording_id = recording_id
        return line


def chat_server(recording_events=None):
    """A host that reports CQ frames whatever its listen state, so the session can
    monitor with auto-answer off; its port bytes are otherwise VARA's."""
    from hfmodem.kestrel.host.server import HostSession, SessionState, VaraServer

    class ReportingSession(HostSession):
        transmission_count = 0

        def modem_ptt(self, on):
            self.transmission_count += int(on)
            super().modem_ptt(on)

        def modem_connected(self, src, dst, bw):
            self.state = SessionState.CONNECTED
            cid = getattr(getattr(self._modem, 'station', None), '_recording_id', None)
            line = f'CONNECTED {src} {dst}' + (f' {bw}' if bw else '')
            self._event('command', _ConnectionLine(line, cid))

        def _write_event(self, kind, value):
            if recording_events is not None and isinstance(value, _ConnectionLine):
                recording_events.append(value.recording_id)
            super()._write_event(kind, value)

        def modem_cqframe(self, call, bw, sn):
            self.modem_bitrate(1, 18, False)
            self.modem_snr(sn)
            self._event('command', f'CQFRAME {call} {bw}')

    class ChatServer(VaraServer):
        session_class = ReportingSession

    return ChatServer


#: How long a peer's closing ident may hold ours back. A VARA HF station keys its
#: ident 0.09 s behind its close and holds it 3.25 s for a six-letter call at 28 WPM.
PEER_IDENT_S = 8.0
PEER_IDENT_GAP_S = 0.5
IDENT_TIMEOUT_S = 20.0


class _IdentificationSteps:
    """Keep the final receipt available even when generator cancellation discards its return."""

    def __init__(self, host, call, bandwidth, after_peer, say):
        self.result = None
        self._steps = host._identify_steps(call, bandwidth, after_peer, say, self)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._steps)

    def close(self):
        self._steps.close()


class Host:
    def __init__(self, transport, *, runtime_factory=None, server_factory=None, release=None):
        from kestrel_host import AudioHostRuntime
        self.transport = transport
        self._recording_events = deque()
        self.release = release
        self.runtime = (runtime_factory or AudioHostRuntime)(transport)
        self.server_factory = server_factory or chat_server(self._recording_events)
        self.server = None
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.stepping = threading.Event()
        self.stepping.set()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def stamp_event(self, item):
        if item.kind == 'host' and item.raw.startswith('CONNECTED '):
            cid = self._recording_events.popleft() if self._recording_events else None
            return replace(item, recording_id=cid)
        return item

    def _run(self):
        try:
            while not self.stop.is_set():
                with self.lock:
                    self.runtime.step()
                self.stepping.wait()
        except BaseException as error:
            self.error = error
            self.stop.set()

    def _host_log(self, kind, detail):
        if kind == 'rejected':
            self.transport.log(f'host command rejected: {detail}')

    def __enter__(self):
        try:
            self.server = self.server_factory(host='127.0.0.1', cmd_port=0, data_port=0, kiss_port=0,
                                              modem_factory=self.runtime.modem_factory,
                                              log=self._host_log)
            self.transport.start()
            self.server.start_background()
            self.thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def identify(self, call, bandwidth, after_peer=False, say=print, *, cancelled=None):
        """Key ours only once the channel is clear of the peer's closing ID.

        On VARA HF the closer IDs first and the other side waits it out, so when the
        peer closed, ours waits for a clear gap; when we closed, a clear channel
        keys at once. Clear means PEER_IDENT_GAP_S without reception, longer
        than a Morse word space at the peer's speed; PEER_IDENT_S bounds the wait.
        """
        import kestrel_connect
        cancelled = cancelled or threading.Event()
        self.stepping.clear()
        try:
            with self.lock:
                start = clear_since = time.monotonic()
                waiting = after_peer
                while (now := time.monotonic()) - start < PEER_IDENT_S:
                    if self.transport.receiving:
                        waiting, clear_since = True, None
                    elif clear_since is None:
                        clear_since = now
                    if clear_since is not None and (not waiting or now - clear_since >= PEER_IDENT_GAP_S):
                        break
                    if cancelled.wait(.05):
                        break
                sent = False
                if cancelled.is_set():
                    from hfmodem.core.cwid import unsent
                    unsent(call, 'identification cancelled', say)
                else:
                    sent = kestrel_connect.identify(self.transport, call, bandwidth, say, cancelled=cancelled)
                    if kestrel_connect._wedged is not None:
                        raise kestrel_connect._wedged
                unkey = getattr(self.transport, 'unkey_confirmed', None)
                confirmed = unkey is not None and unkey() is True
                if sent is True and not confirmed:
                    from hfmodem.core.cwid import unsent
                    unsent(call, 'CW unkey not confirmed', say)
                    self.stop.set()
                    raise HostError('CW unkey not confirmed; stopping runtime')
                if confirmed:
                    return Identification(True, self.runtime.modem.observer.transmission_count,
                                          succeeded=sent is True)
                return False
        finally:
            self.stepping.set()

    def identify_steps(self, call, bandwidth, after_peer=False, say=print):
        return _IdentificationSteps(self, call, bandwidth, after_peer, say)

    def _identify_steps(self, call, bandwidth, after_peer, say, attempt):
        cancelled = threading.Event()
        failures = []
        results = []
        end = time.monotonic() + IDENT_TIMEOUT_S
        def identify():
            try:
                result = self.identify(call, bandwidth, after_peer, say, cancelled=cancelled)
                results.append((result, time.monotonic()))
            except BaseException as error:
                self.stop.set()
                failures.append(error)
        worker = threading.Thread(target=identify, daemon=True)
        worker.start()
        try:
            while worker.is_alive() and time.monotonic() < end:
                yield .05
        finally:
            cancelled.set()
            worker.join(3)
            if worker.is_alive():
                self.stop.set()
                raise HostError('CW identification did not stop')
            if failures:
                raise HostError(f'CW identification failed: {failures[0]}') from failures[0]
            result = results[0][0] if results else False
            if isinstance(result, Identification) and result.succeeded is True and results[0][1] >= end:
                from hfmodem.core.cwid import unsent
                unsent(call, 'identification deadline expired', say)
                result = replace(result, succeeded=False)
            attempt.result = result
        return attempt.result

    def arm(self, on):
        # The runtime's unkey fence reads this line; mid-open it reads unconfirmed.
        with self.lock:
            self.transport.rig.armed = on

    def check(self):
        if self.error is not None:
            raise HostError(f'Kestrel host stopped: {self.error}') from self.error

    def close(self):
        self.stop.set()
        self.stepping.set()
        try:
            if self.thread.ident is not None:
                self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise HostError('Kestrel runtime did not stop')
            try:
                if self.server is not None:
                    self.server.stop()
            finally:
                if self.runtime.modem:
                    self.runtime.modem.stop()
                    self.runtime.modem.poll()
        finally:
            try:
                # Recording transports separate device teardown from storage waits.
                # Stop audio before retiring the keyer, including on failed closes.
                getattr(self.transport, 'stop_audio', self.transport.stop)()
            finally:
                try:
                    if self.release is not None:
                        self.release()
                finally:
                    stop_recording = getattr(self.transport, 'stop_recording', None)
                    if stop_recording is not None:
                        stop_recording()

    def __exit__(self, *exc):
        self.close()


def tune(rig, dial, bandwidth, say=print):
    from hfmodem.core.band import QSY_TOLERANCE_HZ
    from hfmodem.core.rigs import RIGS, mode_passband, resolve_rig
    if rig.keyed:
        raise HostError('Cannot QSY a keyed transmitter')
    try:
        rig.set_freq(dial)
        rig.set_mode('PKTUSB', mode_passband(RIGS[resolve_rig()], 3000))
        state = rig.state()
    except OSError as error:
        raise HostError(f'CAT failed during QSY to {dial} Hz: {error}') from error
    if (state['frequency_hz'] is None or abs(state['frequency_hz'] - dial) > QSY_TOLERANCE_HZ
            or state['mode'] != 'PKTUSB' or (state['passband_hz'] or 0) < 500):
        raise HostError(f'QSY did not take: requested {dial} Hz PKTUSB, read {state}')
    if state['passband_hz'] < bandwidth:
        say(f'warning: the rig passband reads {state["passband_hz"]} Hz, narrower than the '
            f'{bandwidth} Hz session')


def change_band(rig, dial, bandwidth, say=print, settle=1.0):
    """The rig to another band's calling frequency. The FTDX10's band stack brings back
    that band's own mode, width and roofing, so it gets the arm's whole `tune` again,
    read back as the arm reads it; another rig gets a QSY. The ATU's match memory is not
    on the CAT and a tune keys a carrier, so the operator is asked to press TUNE."""
    from hfmodem.core import ftdx10
    from hfmodem.core.band import QSY_TOLERANCE_HZ
    from hfmodem.core.rigs import resolve_rig
    if resolve_rig() != 'ftdx10':
        return tune(rig, dial, bandwidth, say)
    if rig.keyed:
        raise HostError('Cannot change band on a keyed transmitter')
    try:
        reply = ftdx10.normal(rig.batch(ftdx10.session_batch(dial)))
        got = ftdx10.session_frequency(reply, daemon=True)
        if not got or abs(int(got) - dial) > QSY_TOLERANCE_HZ:
            raise HostError(f'band change did not take: asked {dial} Hz, the rig reads {got or "no dial"}')
        say(f'  dial {got} verified')
        if not ftdx10.session_report(reply, daemon=True, say=say):
            raise HostError('band change setup unverified; transmitter remains disarmed')
        # A reply inside the batch that set the filters answers before the rig has settled.
        time.sleep(settle)
        if not ftdx10.readback_report(ftdx10.normal(rig.batch(ftdx10.READBACK)), daemon=True, say=say):
            raise HostError('band change filters unverified; transmitter remains disarmed')
        on = ftdx10.tuner(ftdx10.normal(rig.batch(ftdx10.TUNER)))
    except OSError as error:
        raise HostError(f'CAT failed during the band change to {dial} Hz: {error}') from error
    if on is not False:
        say(f'tuner {"on" if on else "unread"}: if the ATU holds no match for {dial / 1e6:.3f} MHz, '
            'press TUNE on the rig; the session keys no tune carrier')
    return dict(verified=True, tuner_on=on, atu_match_known=False)


class Log(io.TextIOWrapper):
    def fileno(self):
        # input() edits a line only when sys.stdout claims fd 1, which now holds the terminal.
        return 1


class Terminal:
    """The operator's line, edited by readline in a reader thread; screen lines print above it.

    The console pipes stdout to arm.log, so that pipe moves behind sys.stdout and the
    terminal takes fds 0 and 1, where readline reads and draws."""
    PROMPT = '> '

    def __init__(self, held, history):
        tty = os.open('/dev/tty', os.O_RDWR)
        held.callback(os.close, tty)
        held.callback(termios.tcsetattr, tty, termios.TCSANOW, termios.tcgetattr(tty))
        sys.stdout.flush()
        log = os.dup(1)
        os.dup2(tty, 0)
        os.dup2(tty, 1)
        sys.stdout = Log(io.FileIO(log, 'w'), line_buffering=True)
        self.screen = held.enter_context(open(os.dup(tty), 'w', buffering=1))
        self.history, self.lines, self.editing = history, queue.Queue(), False
        import readline  # libedit settles at import whether fds 0 and 1 are a terminal
        self.edit = readline
        if readline.backend == 'editline':
            readline.parse_and_bind('bind ^R em-inc-search-prev')
        readline.set_history_length(1000)
        try:
            readline.read_history_file(history)
        except OSError:
            pass
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            self.editing = True
            try:
                line = input(self.PROMPT) + '\n'
            except EOFError:
                line = ''
            self.editing = False
            if line.strip():
                self.edit.write_history_file(self.history)
            self.lines.put(line)
            if not line:
                return

    def readline(self):
        """The next operator line, '' at EOF, or None when nothing waits."""
        try:
            return self.lines.get_nowait()
        except queue.Empty:
            return None

    def write(self, text):
        if self.editing:
            # libedit keeps the line just sent, newline and all, as its buffer until the next keystroke.
            typed = self.edit.get_line_buffer()
            text = '\r\x1b[K' + text + self.PROMPT + ('' if typed.endswith('\n') else typed)
        self.screen.write(text)

    def write_bbs(self, text):
        """Render a filtered BBS page without a callsign prefix or styled editor."""
        self.write('\x1b[0m' + text + '\x1b[0m\n')

    def flush(self):
        self.screen.flush()


class WindowsTerminal(Terminal):
    """A cancellable console record reader with an independent edit buffer."""

    def __init__(self, held, history):
        from hfmodem.core import winconsole as win
        from consolelib.claim import defer_interrupts
        from contextlib import ExitStack
        self.held = ExitStack()
        self.thread = None
        self.exit = None
        self.lock = threading.RLock()
        self.lines = queue.Queue()
        self.typed = ''
        self.surrogate = ''
        self.error = None
        self.history = []
        self.position = 0
        self.draft = ''
        self.win = win
        self.api = win.api()
        held.callback(self.close)
        try:
            if self.api.ReadConsoleInputExW is None:
                raise RuntimeError('Windows line UI unavailable: ReadConsoleInputExW is missing on this build')
            self.console = self.held.enter_context(win.Console())
            self.screen = self.console
            self.vt = self.console.enable_vt()
            self.exit = win.checked(self.api.CreateEventW(None, True, False, None), 'CreateEventW')
            self.held.callback(self.api.CloseHandle, self.exit)
            self.console.write(self.PROMPT)
            self.thread = threading.Thread(target=self._read, name='chat-console-reader')
            # Finish the native-to-Python startup handoff before interruption can unwind it.
            with defer_interrupts():
                self.thread.start()
        except BaseException:
            self.close()
            raise

    def close(self):
        from consolelib.claim import defer_interrupts
        with defer_interrupts():
            if self.exit is not None:
                self.win.checked(self.api.SetEvent(self.exit), 'SetEvent')
            if self.thread is not None and self.thread.ident is not None:
                self.thread.join()
            try:
                self.held.close()
            finally:
                self.exit = None

    def _read(self):
        import ctypes as C
        from ctypes import wintypes as W
        handles = (W.HANDLE * 2)(self.exit, self.console.input)
        records = (self.win.InputRecord * 64)()
        try:
            while True:
                ready = self.api.WaitForMultipleObjects(2, handles, False, self.win.SLICE_MS)
                if ready == 0:
                    return
                if ready == 258:
                    continue
                if ready != 1:
                    raise OSError('WaitForMultipleObjects failed for the line UI')
                count = W.DWORD()
                self.win.checked(self.api.ReadConsoleInputExW(
                    self.console.input, records, len(records), C.byref(count),
                    self.win.CONSOLE_READ_NOWAIT), 'ReadConsoleInputExW')
                with self.lock:
                    for record in records[:count.value]:
                        if record.kind == 1 and record.event.key.down:
                            key = record.event.key
                            for _ in range(key.repeat):
                                self._key(key.vk, key.char.UnicodeChar)
        except BaseException as error:
            self.error = error
            self.lines.put('')

    def _redraw(self):
        self.console.write(('\r\x1b[K' if self.vt else '\n') + self.PROMPT + self.typed)

    def _key(self, vk, char):
        pending, self.surrogate = self.surrogate, ''
        code = ord(char) if len(char) == 1 else 0
        if 0xd800 <= code <= 0xdbff:
            self.surrogate = char
            return
        if pending and 0xdc00 <= code <= 0xdfff:
            char = chr(0x10000 + ((ord(pending) - 0xd800) << 10) + code - 0xdc00)
        if vk == 13:
            line = self.typed
            self.console.write('\r\n' + self.PROMPT)
            self.lines.put(line + '\n')
            if line.strip():
                self.history.append(line)
                del self.history[:-1000]
            self.typed, self.draft = '', ''
            self.position = len(self.history)
        elif vk == 8:
            if self.typed:
                self.typed = self.typed[:-1]
                if self.vt:
                    self._redraw()
                else:
                    self.console.write('\b \b')
        elif vk in (38, 40):
            if self.position == len(self.history):
                self.draft = self.typed
            self.position = min(len(self.history), max(0, self.position + (-1 if vk == 38 else 1)))
            self.typed = self.history[self.position] if self.position < len(self.history) else self.draft
            self._redraw()
        elif char and char.isprintable():
            self.typed += char
            self.console.write(char)

    def readline(self):
        if self.error is not None:
            raise self.error
        return super().readline()

    def write(self, text):
        with self.lock:
            if self.vt:
                self.console.write('\r\x1b[K' + text + self.PROMPT + self.typed)
            else:
                text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
                self.console.write('\n' + text + ('' if text.endswith('\n') else '\n')
                                   + self.PROMPT + self.typed)


def line_terminal(held, history, ui='auto'):
    if ui not in ('auto', 'line'):
        raise ValueError('Textual UI is unavailable on this branch; use --ui line')
    return (WindowsTerminal if os.name == 'nt' else Terminal)(held, history)


class Diagnostics:
    """Bounded runtime-to-terminal mailbox; audio threads never write the screen."""
    def __init__(self):
        self.events = queue.Queue(maxsize=256)
        self.recovery = queue.Queue(maxsize=16)
        self.dropped = 0

    def put(self, kind, **details):
        if kind in ('link_recovery', 'capture_rate'):
            # Keep recovery/closing notices despite a flood of verbose frames.
            # If the operator thread stalls, retain the newest bounded history.
            try:
                self.recovery.put_nowait((kind, details))
            except queue.Full:
                try:
                    self.recovery.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.recovery.put_nowait((kind, details))
                except queue.Full:
                    pass
            return
        try:
            self.events.put_nowait((kind, details))
        except queue.Full:
            self.dropped += 1

    def drain(self, session):
        for _ in range(16):
            try:
                kind, details = self.recovery.get_nowait()
            except queue.Empty:
                break
            session.diagnostic(kind, **details)
        if self.dropped:
            dropped, self.dropped = self.dropped, 0
            session.diagnostic('dropped', count=dropped)
        for _ in range(256):
            try:
                kind, details = self.events.get_nowait()
            except queue.Empty:
                break
            session.diagnostic(kind, **details)


def terminal(session, tty):
    """Operator lines go to the session between its ticks; EOF or /quit ends it."""
    failure = None
    try:
        while True:
            session.tick()
            line = tty.readline()
            if line is None:
                continue
            if not line or not session.command(line):
                break
    except KeyboardInterrupt:
        pass
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            session.quit()
        except BaseException as cleanup_error:
            if failure is not None and cleanup_error is not failure:
                # Preserve the initiating fault while retaining the close failure
                # in the traceback. The outer runtime/keyer teardown still runs.
                raise failure from cleanup_error
            raise


def received_retune_adapter(runtime, client, tuner, unkeyed, *, completed=None, quitting=lambda: False):
    """Bridge cooperative application work to the sole modem owner's boundary."""
    def retune_steps(target, generation):
        modem = runtime.modem
        if modem is None:
            raise HostError('Receiver retune has no modem owner')
        valid = lambda: (client.link_generation == generation
                         and client.connected and not client.closed and not quitting())
        try:
            request = runtime.request_retune(
                lambda: tuner(target), modem=modem, station=modem.station,
                receive_cursor=modem.received_cursor, unkeyed=unkeyed, valid=valid)
        except (ValueError, queue.Full) as error:
            raise HostError(f'Receiver retune cannot be queued: {error}') from error
        try:
            while not request.done:
                if not valid():
                    raise HostError('Link changed during receiver retune')
                yield .05
            if request.error is not None:
                raise HostError(f'Receiver retune failed: {request.error}') from request.error
            if request.cancelled:
                raise HostError('Receiver retune cancelled')
        finally:
            request.cancel(timeout=.1 if quitting() else -1)
            if request.done and not request.cancelled and request.error is None:
                if completed is not None:
                    completed(target)
    return retune_steps


def close_reporter(reporter, say, emit):
    before = reporter.status()['submitted']
    result = reporter.close()
    emit('pskreporter_shutdown', **result)
    say(f"PSKReporter shutdown: {result['submitted'] - before} final UDP reports submitted; "
        f"{result['unsent_at_shutdown']} unsent, {result['inflight_at_shutdown']} in flight/unknown. "
        "Server receipt is unconfirmed.")


ESCAPES = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|\a')


def logging_say(screen, log):
    """The operator's screen, each line also kept UTC-stamped in the session's console.log."""
    def say(text):
        screen.write(text + '\n')
        screen.flush()
        for line in ESCAPES.sub('', text).split('\n'):
            log.write(f'{utc()} {line}\n')
        log.flush()

    return say


def run(args, rig, admission, bbs_service=None):
    with ExitStack() as held:
        radio = ExitStack()
        radio.callback(rig.shutdown)
        held.callback(radio.close)
        import numpy as np
        from hfmodem.contact.slots import SlotWatch
        from hfmodem.core import config, identification
        idcall = identification.resolve(args.mycall, args.idcall).cw_call
        station_path = os.environ.get(config.STATION_ENV)
        try:
            station = config.load(station_path, environ=os.environ).station if station_path else config.Station()
        except (OSError, ValueError) as exc:
            raise config.ConfigError(f'{config.STATION_ENV} names {station_path}: {exc}') from exc
        from kestrel_connect import AudioVaraIO
        from hfmodem.contact.bbs import HostCatalog
        from hfmodem.contact.bbs_server import configured_directory

        offset = dial_offset(args)
        scale_w, clamp_w = config.power_limits()[:2]
        root = configured_directory('' if args.no_bbs else args.bbs_directory)
        catalog = bbs_service.catalog if bbs_service is not None else None
        if bbs_service is None and root:
            catalog = HostCatalog()
            held.callback(catalog.close)
            catalog.enable(root)
        tty = line_terminal(held, args.session.parent.parent / 'history', getattr(args, 'ui', 'auto'))
        transcript = held.enter_context((args.session / 'host.jsonl').open('w', encoding='utf-8'))
        events = held.enter_context((args.session / 'events.jsonl').open('a', encoding='utf-8'))
        console = held.enter_context((args.session / 'console.log').open('a', encoding='utf-8'))
        contacts = held.enter_context((args.session / 'contacts.jsonl').open('a', encoding='utf-8'))
        session = None
        reporter = None
        if getattr(args, 'pskreporter', False):
            from hfmodem.station.pskreporter import Config, Reporter
            reporter = Reporter(Config(args.mycall, args.loc, enabled=True))
            held.callback(lambda: close_reporter(reporter, say, emit))

        say = logging_say(tty, console)
        say(f'session records: {args.session}')

        def bbs_say(text):
            tty.write_bbs(text)
            tty.flush()

        def emit(kind, **record):
            if kind == 'heard' and reporter is not None:
                reporter.accept_heard(record, live=True)
            line = json.dumps(dict(kind=kind, utc=utc(), **record), ensure_ascii=False) + '\n'
            for stream in (events, contacts) if kind == 'contact' else (events,):
                stream.write(line)
                stream.flush()
            if kind == 'contact':
                recorder = getattr(audio, 'rec', None)
                contact_log.update(record, sample=getattr(recorder, 'position', None), offset_hz=offset,
                                   recording_id=record.get('recording_id'))
            if kind == 'band':
                # arm.log's reader, the console, moves the KiwiSDR witness on this line.
                print(f'BAND {record["band_m"]} m {record["dial_hz"]} Hz', flush=True)

        def event(item):
            stamp = getattr(host, 'stamp_event', None)
            if stamp is not None:
                item = stamp(item)
            transcript.write(json.dumps(dict(utc=utc(), kind=item.kind, raw=item.raw),
                                        ensure_ascii=False) + '\n')
            transcript.flush()
            if session is not None:
                session.handle(item)

        audio = AudioVaraIO(args.tx_device, args.rx_device, config.tx_drive(), rig,
                            record=args.session / 'receive.wav', offset_hz=offset,
                            resolved=getattr(args, 'recording_selection', None))
        contact_log = ContactLog(args.session, mycall=args.mycall,
                                recording=getattr(getattr(audio, 'rec', None), 'session', None))
        diagnostics = Diagnostics()
        audio.diagnostic = diagnostics.put
        calling = CALLING_HZ[args.band]
        slots = SlotWatch(calling, calling, offset=offset, mycall=args.mycall)
        radio.callback(slots.close)
        # Our own transmission comes back through the codec; the watch hears silence then.
        audio.taps.append(lambda frame: slots.put(np.zeros_like(frame) if rig.keyed else frame))
        host_owner = Host(audio)
        host_owner.release = radio.close
        host = held.enter_context(host_owner)
        if offset:
            say(f'monitor-wide: the dial sits {abs(offset)} Hz {"below" if offset < 0 else "above"} '
                f'each channel; we key and hear the channel at {1500 - offset} Hz audio')

        def closing():
            if (rec := getattr(audio, 'rec', None)) is not None:
                say(f'recording {rec.session.policy}: {rec.session.directory}')
        held.callback(closing)
        admission.opened()
        from hfmodem.core import claim
        from hfmodem.core.pipeio import Owner
        owner_fd = (claim.extra('owner', optional=True) if os.name == 'nt' else
                    int(os.environ['HFMODEM_OWNER_FD']) if 'HFMODEM_OWNER_FD' in os.environ else None)
        owner = Owner(owner_fd) if owner_fd is not None else None
        if owner is not None:
            held.callback(owner.close)

        def ready(sockets, timeout):
            host.check()
            if session is not None:
                diagnostics.drain(session)
            while not rig.meter_reports.empty():
                emit('tx_meters', **rig.meter_reports.get())
            if rig.retired:
                raise HostError('Station session was retired')
            if owner is not None and owner.gone():
                rig.shutdown()
                raise HostError('Console owner exited')
            timeout = min(timeout, .1) if owner else timeout
            if not sockets:
                time.sleep(timeout)
                return []
            return select.select(sockets, [], [], timeout)[0]

        with Client.open(command_port=host.server.cmd_port, data_port=host.server.data_port,
                         kiss_port=getattr(host.server, "kiss_port", None), event=event, ready=ready) as client:
            client.initialize(args.mycall, args.bw)
            flow = Flow(client, lambda hz: tune(rig, hz + offset, args.bw, say), band=args.band,
                        profile={field.upper(): getattr(args, field) for field in ('name', 'qth', 'loc')},
                        services=args.services)
            flow.received_retune_steps = received_retune_adapter(
                host.runtime, client, flow.tuner,
                lambda: not rig.keyed and rig.line_state() is False,
                completed=flow.received_tuned, quitting=lambda: session is not None and session._quitting)
            flow.dial = CALLING_HZ[args.band]
            def report(line):
                print(line, flush=True)
                say(line.strip())

            def identify(bandwidth, after_peer):
                return host.identify_steps(idcall, bandwidth, after_peer, report)

            session = Session(client, flow, say=say, emit=emit, arm=host.arm,
                              line_state=rig.line_state, bandwidth=args.bw, reporter=reporter,
                              identify=identify, idcall=idcall, ident_report=report,
                              id_interval_s=station.id_interval_s, closing_cwid=station.closing_cwid,
                              power=partial(rig.power_w, scale_w=scale_w), power_clamp_w=clamp_w,
                              caller=lambda: getattr(host.runtime.modem, 'attempt_caller', None),
                              mailbox=Mailbox(args.session.parent.parent / 'vmail'), slots=slots,
                              band_tuner=lambda hz: change_band(rig, hz + offset, args.bw, say),
                              bbs_catalog=None if args.no_bbs else catalog,
                              bbs_core=bbs_service.core if bbs_service is not None else None,
                              bbs_say=bbs_say,
                              bbs_reload=bbs_service.reload if bbs_service is not None else None)
            session.say(session.parking.summary())
            if catalog is not None and not args.no_bbs:
                export = bbs_service.config.export_root if bbs_service is not None else root
                say(f'BBS hosting enabled: {export}; advertised in advanced beacons and opening reports')
            if bbs_service is not None:
                say(f'Interactive BBS enabled: {bbs_service.config.title}; peer enters with BBS')
            say('PSKReporter ' + ('on: validated CQ/beacon decodes, five-minute batches; '
                                    '/pskreporter status or off' if reporter is not None else 'off'))
            say('commands: ' + ', '.join(Session.COMMANDS) + '; anything else is sent when linked')
            try:
                session.monitor()
                terminal(session, tty)
            except BaseException as error:
                if isinstance(error, Exception):
                    say(f'error: {error}')
                session.identification_unsent()
                raise
            finally:
                session.parking.store.close()
        return 0


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    from hfmodem.core.recording import resolve_recording
    try:
        args.recording_selection = resolve_recording(args.recording, no_record=args.no_record, record=args.record)
    except ValueError as exc:
        p.error(str(exc))
    print(f'Recording: {args.recording_selection.policy} ({args.recording_selection.source})')
    if args.record is not None:
        args.session = args.record
    validate(args)
    from hfmodem.bbs.config import configured_path
    args.bbs_config = None if args.no_bbs_service else configured_path(args.bbs_config)
    # Validate dependencies and acquire the BBS owner before touching the rig.
    with ExitStack() as held:
        service = None
        if args.bbs_config and not args.no_bbs_service:
            from hfmodem.bbs.service import open_service
            from hfmodem.contact.bbs_server import configured_directory
            legacy = configured_directory(args.bbs_directory)
            service = held.enter_context(open_service(args.bbs_config, legacy_root=legacy))
        return claimed_main(args, service)


def claimed_main(args, bbs_service=None):
    from hfmodem.core.claim import received
    from hfmodem.core.ptt import arming_refusal
    from rig_client import Rig, _install_unkey_handlers

    if refusal := arming_refusal(args.ptt_device):
        raise ValueError(refusal)
    admission = received(launch=True)
    rig = None
    try:
        rig = Rig(args.rigctld, False, print, ptt_device=args.ptt_device, line_ptt=True)
        _install_unkey_handlers(rig)
        model, dial = rig.identify()
        if args.expect_model.lower() not in model.lower() or not dial.isdigit():
            raise HostError(f'Rig identification failed: {model!r}, dial {dial!r}')
        tune(rig, CALLING_HZ[args.band] + dial_offset(args), args.bw)
        return run(args, rig, admission, bbs_service)
    finally:
        try:
            if rig is not None:
                rig.shutdown()
                print('PTT VERDICT: ' + ('keying line handed back LOW' if rig.released_low else
                      'CANNOT PROVE PTT IS DOWN'), flush=True)
        finally:
            admission.failed()


if __name__ == '__main__':
    from kestrel_connect import exit_status
    raise SystemExit(exit_status(main))
