# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Keyboard-to-keyboard RTTY on the station rig: the session, with no screen.

The operator keys every over. Nothing here keys on its own, holds traffic for an
ID or caps an over; the watchdog's drop is its only unkey of its own. Transmit
tones are fixed, and AFC never walks the receive pair more than `AFC_BOUND_HZ`
from them.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import queue
import re
import select
import signal
import socket
import sys
import threading
import time
import wave

import numpy as np

from hfmodem.core import tx_events
from hfmodem.core.audio import CARD_RATE_HZ, BurstInvalidated, StreamLane
from hfmodem.lanner import fsk, ita2, rtty

FS = CARD_RATE_HZ
CENTRE_HZ = 2210.0
#: About one character at 45.45 Bd: a typed character is on the air within two.
QUEUE_S = 0.15
CAPACITY_S = 1.0
STARVED_S = 1.0
STEP_S = 0.01
RX_BLOCK_S = 0.1
ACQUIRE_S = 3.0
ACQUIRE_SPAN_HZ = 250.0
AFC_EVERY_S = 2.0
AFC_STEP_HZ = 5.0
AFC_BOUND_HZ = 100.0
NUDGE_HZ = 10.0
SQUELCH_FRAMES = 16
SQUELCH_OPEN = 0.8
SQUELCH_DROP = 0.5
GAP_CAP_S = 1.0
DIAL_EVERY_S = 2.0
RETRY_S = 5.0
OWNER_ENV = 'HFMODEM_OWNER_FD'
SET_ASIDE = 'a contact set aside earlier'
PIPE_LINES = 2000
DISK_JOBS = 1000
RST = '599'

MACROS = (
    ('CQ', 'CQ CQ DE {MYCALL} {MYCALL} K'),
    ('answer', '{CALL} DE {MYCALL}'),
    ('exchange', '{CALL} DE {MYCALL} UR {RST} {RST} QTH {GRID} {GRID} BK'),
    ('contest', '{CALL} {RST} {SERIAL} {SERIAL}'),
    ('73', '{CALL} 73 DE {MYCALL} SK'),
    ('ID', 'DE {MYCALL}'),
    ('QRZ', 'QRZ DE {MYCALL} K'),
)

CALLSIGN = re.compile(r'\b(?:[A-Z]{1,2}|\d[A-Z]|[A-Z]\d)\d[A-Z]{1,4}(?:/[A-Z0-9]{1,4})?\b')


class _Tape(StreamLane):
    """On the FTDX10 the monitored transmit audio is the record of our own side of the
    QSO, so the half-duplex flush must not reach the recording."""

    def flush_to(self, card_index: int) -> None:
        pass


class _Writer:
    """One sink's writes, in order, on a thread of its own behind a bounded queue. A
    write that finds the queue full or the writer closed is refused at once, so a
    stalled pipe or disk costs that sink its backlog and never holds the caller or
    grows memory. `SimpleQueue.put` is re-entrant, so a signal handler that logs
    while its own thread is mid-put cannot deadlock; the bound is approximate."""

    def __init__(self, limit, warn):
        self._jobs = queue.SimpleQueue()
        self._limit = limit
        self._warn = warn
        self.closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def put(self, job, *args, force=False) -> bool:
        if self.closed or not force and self._jobs.qsize() >= self._limit:
            return False
        self._jobs.put((job, args))
        return True

    def _run(self):
        while (item := self._jobs.get()) is not None:
            job, args = item
            try:
                job(*args)
            except Exception as exc:
                self._warn(f'could not write ({job.__name__}): {exc}; the session goes on')

    def flush(self, timeout=None) -> bool:
        done = threading.Event()
        return self.put(done.set, force=True) and done.wait(timeout)

    def close(self, timeout) -> bool:
        if not self.closed:
            self._jobs.put(None)
        self.closed = True
        self._thread.join(timeout)
        return not self._thread.is_alive()


class _Contacts:
    """ADIF records on a thread of their own, never behind stdout or the recording.
    A record stays held until it is written, retried while the session runs; one
    still held at the end is set aside beside the log, where the next session
    counts its serial and writes it."""

    def __init__(self, log, note):
        self.log = log
        self._note = note
        self._held: deque[tuple[str, str]] = deque((SET_ASIDE, record) for record in log.kept())
        self._warned = None
        self._passes = [0, 0]
        self._kept = None
        self._done = threading.Condition()
        self._wake = threading.Event()
        self._running = True
        self._wake.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, call, record):
        self._held.append((call, record))
        self._wake.set()

    def _run(self):
        while self._running:
            self._wake.wait(RETRY_S)
            self._wake.clear()
            with self._done:
                self._passes[0] += 1
                started = self._passes[0]
            self._write()
            with self._done:
                self._passes[1] = started
                self._done.notify_all()
        left = [record for _, record in self._held]
        self._kept = left and (str(self.log.held) if self.log.keep(left) else '')

    def flush(self, timeout=None) -> bool:
        """Wait for one attempt at everything held, begun after this call."""
        with self._done:
            after = self._passes[0] + 1
            self._wake.set()
            return self._done.wait_for(lambda: self._passes[1] >= after, timeout)

    def _write(self):
        while self._held:
            call, record = self._held[0]
            try:
                if (call == SET_ASIDE or self._warned == record) and self.log.logged(record):
                    self.log.sync()
                else:
                    self.log.write(record)
            except OSError as exc:
                if self._warned != record:
                    self._warned = record
                    self._note(f'NOT LOGGED {call} yet: {exc}. Held and retried every '
                               f'{RETRY_S:g}s. The record: {record.strip()}')
                return
            self._held.popleft()
            if call == SET_ASIDE and not self.log.keep(
                    [kept for label, kept in self._held if label == SET_ASIDE]):
                self._note(f'could not update {self.log.held}; the next session will find '
                           f'{call} already in the log and skip it')
            self._note(f'logged {call} to {self.log.path}')

    def close(self, timeout) -> tuple[list[str], str]:
        """The records still unwritten, and where they were set aside: nowhere when
        the set-aside write failed or the disk outlasted `timeout`."""
        self._running = False
        self._wake.set()
        self._thread.join(timeout)
        left = [record for _, record in list(self._held)]
        return left, ('' if self._thread.is_alive() else self._kept or '')


class Session:
    def __init__(self, audio, keyer, log, *, rate=rtty.AMATEUR, centre_hz=CENTRE_HZ,
                 dial_hz=None, drive=0.5, settle_s=0.04, tr_switch_s=0.07,
                 mycall='', grid='', directory=None, out=None, variant='us',
                 usos=True, dial_reader=None, warnings=(), shout=None, owner=None, parent=None):
        self.audio = audio
        self.keyer = keyer
        self.log = log
        self.rate = rate
        self.dial_hz = dial_hz
        self._dial_reader = dial_reader
        self.tx_centre = centre_hz
        self.rx_centre = centre_hz
        self.reverse = False
        self.drive = self.drive_ceiling = drive
        self.mycall = mycall.upper()
        self.grid = grid
        self.variant = variant
        self.state = 'rx'
        self.ptt_unconfirmed = False
        self.alarm_text = ''
        self.warnings = tuple(warnings)
        self.on_screen = False
        self.squelch = SQUELCH_OPEN
        self.typeahead: list[str] = []
        self._flight: deque[list] = deque()
        self.pieces: deque[tuple[str, str, bool]] = deque()
        self.lane = audio.subscribe(StreamLane(FS))
        self.stream = rtty.Stream(baud=rate.baud, mark_hz=self.rx_mark, space_hz=self.rx_space,
                                  variant=variant, fs=FS, unshift_on_space=usos)
        self._out = out or sys.stdout
        self._shout = shout or (lambda line: print(line, file=sys.stderr, flush=True))
        warn = lambda text: self.pieces.append(('note', text, False))
        self._pipe = _Writer(PIPE_LINES, warn)
        self._disk = _Writer(DISK_JOBS, warn)
        self._lost: dict[str, list[int]] = {}
        self._contacts = _Contacts(log, self.note)
        self._lock = threading.RLock()
        self._settle_n = round(settle_s * FS)
        self._tr_n = round(tr_switch_s * FS)
        self._tx = None
        self._tx_stream = None
        self._key_index = None
        self._keying_since = 0.0
        self._halted = None
        self._failed = None
        self._owner, self._parent = owner, parent
        self._halts = 0
        self._cut = None
        self._beat = time.monotonic()
        self._mute_until = 0
        self._rx_next = None
        self._rx_pending: list[tuple[np.ndarray, np.ndarray]] = []
        self._rx_pending_n = 0
        self._recent: deque[np.ndarray] = deque()
        self._recent_n = 0
        self._acquire = (int(audio.samples) + round(ACQUIRE_S * FS), self.tx_centre)
        self._afc_due = 0
        self._frames: deque[bool] = deque(maxlen=SQUELCH_FRAMES)
        self._heard = ''
        self._sent_tail = ''
        self.id_at = None
        self._lines = {'RX': '', 'TX': ''}
        self._events = tx_events.TxEvents('rtty')
        self._tx_id = 0
        self._threads: list[threading.Thread] = []
        self._running = False
        directory = Path(directory) if directory else None
        self._news = directory / 'recorders.log' if directory else None
        self._news_read = 0
        self._transcript = (directory / 'transcript.txt').open('a', buffering=1) if directory else None
        self._tape = None
        if directory:
            self._tape_lane = audio.subscribe(_Tape(FS))
            self._tape = wave.open(str(directory / 'stream.wav'), 'wb')
            self._tape.setnchannels(1)
            self._tape.setsampwidth(2)
            self._tape.setframerate(FS)
        self._annotate(f'{rate}, mark {self.tx_mark:g} space {self.tx_space:g} Hz audio'
                       + (f', mark RF {self.mark_rf_hz} Hz on dial {dial_hz}' if dial_hz else '')
                       + f', drive {drive:.3f}')
        for warning in self.warnings:
            self.alarm(warning, current=False)

    @property
    def tx_mark(self):
        return self.tx_centre + self.rate.shift / 2

    @property
    def tx_space(self):
        return self.tx_centre - self.rate.shift / 2

    @property
    def rx_mark(self):
        return self.rx_centre + (-1 if self.reverse else 1) * self.rate.shift / 2

    @property
    def rx_space(self):
        return self.rx_centre - (-1 if self.reverse else 1) * self.rate.shift / 2

    @property
    def mark_rf_hz(self):
        return None if self.dial_hz is None else round(self.dial_hz + self.tx_mark)

    def _retune(self, centre, why):
        self.rx_centre = centre
        self.stream.retune(self.rx_mark, self.rx_space)
        self.say(f'rtty: receive pair {self.rx_centre:.1f} Hz '
                 f'({self.rx_centre - self.tx_centre:+.1f} from ours), {why}')

    def _put(self, writer, kind, job, *args, at=None, size=1):
        lost = self._lost.get(kind)
        if not writer.put(job, *args):
            if lost:
                lost[1] += size
            else:
                self._lost[kind] = [at, size]
            return
        if lost and self._lost.pop(kind, None):
            self._lost_note(kind, *lost)

    def _lost_note(self, kind, at, size):
        what = (f'{size} samples ({size / FS:.1f}s) of {kind} from card {at}' if at is not None
                else f'{size} {kind} lines')
        self.note(f'recording gap: {what} not written; the writer fell behind')

    def say(self, line):
        stamp = datetime.now(timezone.utc).strftime('%H:%M:%S.%f')[:-3]
        line = f'{stamp}Z {line}'
        if self._pipe.closed:
            self._shout(line)
        else:
            self._put(self._pipe, 'event log', self._print, line)

    def _print(self, line):
        try:
            print(line, file=self._out, flush=True)
        except BrokenPipeError:
            if self.state != 'rx' and self._halted is None:
                self._drop('the console stopped reading the event log')
            raise

    def flush(self, timeout=None):
        return all([self._contacts.flush(timeout), self._disk.flush(timeout),
                    self._pipe.flush(timeout)])

    def note(self, text):
        self.pieces.append(('note', text, False))
        self.say(f'rtty: {text}')

    def alarm(self, text, current=True):
        if current:
            self.alarm_text = text
        self.pieces.append(('alarm', text, False))
        self.say(f'rtty: ALARM {text}')
        if not self.on_screen and not self._pipe.closed:
            self._shout(f'!! {text}')

    def _show(self, kind, text, dim=False):
        self.pieces.append((kind.lower(), text, dim))
        line = self._lines[kind] + text
        *done, self._lines[kind] = line.split('\n')
        for finished in done:
            self._write_line(kind, finished)

    def _annotate(self, text):
        self.say(f'rtty: {text}')
        self._transcribe(f'-- {text}')

    def _write_line(self, kind, text):
        self.say(f'{kind.lower()}: {text}')
        self._transcribe(f'{kind} {text}')

    def _transcribe(self, text):
        if self._transcript:
            stamp = datetime.now(timezone.utc).strftime('%H:%M:%S')
            self._put(self._disk, 'transcript', self._transcript.write, f'{stamp}Z {text}\n')

    def type(self, ch: str) -> bool:
        """Append to the typeahead; False for a character ITA2 cannot send."""
        with self._lock:
            ch = ch.upper()
            if ita2.Encoder(self.variant).feed(ch)[1]:
                return False
            self.typeahead.append(ch)
            return True

    def backspace(self):
        with self._lock:
            if self.typeahead:
                self.typeahead.pop()

    def macro(self, index: int):
        with self._lock:
            name, text = MACROS[index]
            text = self.log.substitute(text.replace('{MYCALL}', self.mycall)
                                       .replace('{GRID}', self.grid).replace('{RST}', RST))
            if '{CALL}' in MACROS[index][1] and self.log.current is None:
                self.note('no {CALL} yet: Ctrl-K takes it from what was received')
            for ch in f'\n{text}\n':
                self.type(ch)
            if index == 0 and self.state == 'rx':
                self.key()

    def _audio_trouble(self):
        if getattr(self.keyer, 'must_retire', False):
            return ('the keying line could not be confirmed down earlier this session; '
                    'restart the terminal to key again')
        if self._failed is None and (gone := self._orphaned()):
            self._failed = gone
        if self._failed:
            return self._failed
        if self.audio.failure is not None:
            return f'the audio callback failed: {self.audio.failure}'
        if self.audio.callback_age() > STARVED_S:
            return f'the audio callbacks stopped {self.audio.callback_age():.1f}s ago'
        if self._running and time.monotonic() - self._beat > STARVED_S:
            return f'the session thread stopped {time.monotonic() - self._beat:.1f}s ago'
        return None

    def halt(self, why):
        """PTT down from any thread, at once: no lock, no pipe, no disk. Whoever holds
        the session next settles the over's books against the instant taken here."""
        cut = self.audio.sample_now()
        if self._cut is None:
            self._cut = cut
        self._halted = why
        self._halts += 1
        if not self.keyer.key(False):
            self.ptt_unconfirmed = True
            return False
        return True

    def key(self):
        with self._lock:
            if self.state != 'rx':
                return
            if trouble := self._audio_trouble():
                self.note(f'{trouble}; nothing transmitted')
                return
            self._halted, self._cut, halts = None, None, self._halts
            self.alarm_text, self._sent_tail = '', ''
            try:
                self.state, self._keying_since = 'keying', time.monotonic()
                if not self.keyer.key(True, 'rtty over'):
                    raise _Unconfirmed('PTT did not confirm on key-up')
                if self._halts != halts:
                    raise _Overtaken(self._halted)
                now = int(self.audio.sample_now())
                self._key_index = now
                self._tx = rtty.Transmitter(self.rate, self.tx_mark, self.tx_space,
                                            pull=self._pull, variant=self.variant, fs=FS,
                                            amplitude=self.drive)
                self._tx_stream = self.audio.arm_stream(now + self._settle_n,
                                                        round(CAPACITY_S * FS))
                self.state = 'tx'
                self._annotate(f'key-down at card {now}, audio from {self._tx_stream.at}, '
                               + (f'mark RF {self.mark_rf_hz} Hz, ' if self.dial_hz else '')
                               + f'drive {self.drive:.3f}')
                self._pump()
            except BaseException as exc:
                self._abandon(exc)
                if not isinstance(exc, Exception):
                    raise

    def _abandon(self, exc):
        down = self.halt(f'key-up failed: {exc}')
        if self._tx_stream is not None:
            try:
                self.audio.finish_burst()
            except Exception:
                pass
        self._settle(down)
        if isinstance(exc, _Overtaken) or isinstance(exc, _Unconfirmed) and down:
            return
        why = exc if isinstance(exc, Exception) else 'interrupted'
        self.alarm(f'key-up failed ({why}); PTT ' + ('dropped' if down else
                   'DOWN NOT CONFIRMED -- unkey the rig by hand'))

    def turn_over(self):
        with self._lock:
            if self.state == 'rx':
                if not self.typeahead:
                    return
                self.key()
                if self.state == 'rx':
                    return
            if self.state == 'tx':
                self.typeahead.append('\n')
                self._tx.finish()
                self.state = 'turnover'

    def abort(self):
        if self.state == 'rx':
            if self.ptt_unconfirmed and self.keyer.key(False):
                self.ptt_unconfirmed = False
                self.alarm_text = ''
                self.note('PTT reads down now')
            return
        self.halt('stopped by the operator')
        with self._lock:
            if self.state != 'rx':
                self._unkey('stopped by the operator')

    def set_call(self):
        with self._lock:
            calls = [c for c in CALLSIGN.findall(self._heard) if c != self.mycall]
            if not calls:
                self.note('no callsign heard yet')
                return None
            self.log.start(calls[-1])
            self.note(f'QSO with {calls[-1]}, serial {self.log.serial:03d}')
            return calls[-1]

    def commit(self):
        with self._lock:
            if self.log.current is None:
                self.note('no QSO in progress to log (Ctrl-K starts one)')
                return None
            qso, record = self.log.record(mode='RTTY', freq_hz=self.mark_rf_hz or 0,
                                          rst_sent=RST, rst_rcvd=RST)
            self._contacts.add(f'{qso.call} serial {qso.serial:03d}', record)
            return qso

    def toggle_reverse(self):
        with self._lock:
            self.reverse = not self.reverse
            self._retune(self.rx_centre, 'reverse ' + ('on' if self.reverse else 'off'))

    def toggle_usos(self):
        with self._lock:
            self.stream.unshift_on_space = not self.stream.unshift_on_space
            self.note('USOS ' + ('on' if self.stream.unshift_on_space else 'off'))

    def reacquire(self):
        with self._lock:
            self._acquire = (0, self.rx_centre)

    def nudge(self, sign: int):
        with self._lock:
            self._retune(self.rx_centre + sign * NUDGE_HZ, 'nudged')

    def adjust_drive(self, db: float):
        with self._lock:
            self.drive = min(1.0, self.drive * 10 ** (db / 20))
            if self._tx is not None:
                self._tx.amplitude = self.drive
            self.say(f'rtty: drive {self.drive:.3f}'
                     + (f', above the chain tx_drive {self.drive_ceiling:.3f}'
                        if self.drive > self.drive_ceiling else ''))

    def adjust_squelch(self, step: float):
        with self._lock:
            self.squelch = min(1.0, max(SQUELCH_DROP, self.squelch + step))

    @property
    def unsent(self) -> str:
        with self._lock:
            return ''.join(ch for ch, _ in self._flight) + ''.join(self.typeahead)

    def _pull(self):
        if self._flight and self._flight[-1][1] is None:
            self._flight[-1][1] = self._tx.scheduled
        if not self.typeahead:
            return ''
        self._flight.append([self.typeahead.pop(0), None])
        return self._flight[-1][0]

    def _sent(self, progress, cut):
        """Characters whose last frame left the card before capture index `cut`."""
        while self._flight and self._flight[0][1] is not None:
            done = progress.card_end(self._flight[0][1])
            if done is None or done > cut:
                return
            ch = self._flight.popleft()[0]
            self._show('TX', ch)
            self._sent_tail = (self._sent_tail + ch)[-32:]
            if self.mycall and self._sent_tail.endswith(self.mycall):
                self.id_at = done

    def _pump(self):
        tx, stream = self._tx, self._tx_stream
        if stream.closed:
            return
        want = round(QUEUE_S * FS) - stream.queued
        if want <= 0:
            return
        start = tx.position
        chunk = tx.render(want)
        if tx.end is not None:
            chunk = chunk[:max(0, tx.end - start)]
        stream.write(chunk)
        if tx.end is not None and tx.position >= tx.end:
            stream.close()

    def _transmit(self):
        if self.state == 'rx':
            return
        if self._halted is not None:
            self._unkey(self._halted)
            return
        self._pump()
        stream, now = self._tx_stream, self.audio.sample_now()
        if self._cut is not None:
            now = min(now, self._cut)
        progress = stream.progress
        self._sent(progress, now)
        if stream.drained_at(progress) and now >= progress.end:
            self._unkey('turned over')

    def _unkey(self, why):
        cut = self._cut if self._cut is not None else self.audio.sample_now()
        down = self.keyer.key(False)
        stream, outcome = self._tx_stream, 'sent'
        if stream is None:
            if not down:
                self.alarm(f'PTT DOWN NOT CONFIRMED ({why}) -- unkey the rig by hand')
            self._settle(down)
            return
        progress = stream.progress
        try:
            end = self.audio.finish_burst()
        except BurstInvalidated as exc:
            end, outcome = stream.end, 'error'
            self.note(f'the over had a hole in it: {exc}')
        self._sent(progress, cut)
        self._tx_id += 1
        self._events.record('completed', self._tx_id, outcome=outcome, start_index=stream.at,
                            end_index=end, key_index=self._key_index, drive=self.drive,
                            rate=str(self.rate), mark_hz=self.tx_mark, space_hz=self.tx_space,
                            mark_rf_hz=self.mark_rf_hz)
        for line in self._events.lines():
            self._put(self._pipe, 'event log', self._print, line)
        if down:
            self.say(f'rtty: unkeyed at card {int(cut)}, audio {stream.at}-{end}: {why}')
        else:
            self.alarm(f'PTT DOWN NOT CONFIRMED at card {int(cut)} ({why}) '
                       '-- unkey the rig by hand')
        self._mute_until = end + self._tr_n
        self._acquire = (self._mute_until + round(ACQUIRE_S * FS), self.tx_centre)
        self._settle(down)

    def _settle(self, down):
        self.ptt_unconfirmed = not down
        self.typeahead[:0] = [ch for ch, _ in self._flight]
        self._flight.clear()
        self._tx = self._tx_stream = None
        self._halted = self._cut = None
        self.state = 'rx'

    def watchdog(self):
        """Lock-free, and `Keyer` never waits: the thread that would hold the lock is
        the one this exists to outlive."""
        state, stream = self.state, self._tx_stream
        if self._failed is None and (gone := self._orphaned()):
            self._failed = gone
            if state == 'rx':
                self.alarm(f'{gone}; keying is declined')
            else:
                self._drop(f'WATCHDOG -- {gone}')
            return
        if state == 'rx' or self._halted is not None:
            return
        if state == 'keying':
            held = time.monotonic() - self._keying_since
            why = f'key-up still unfinished after {held:.1f}s' if held > STARVED_S else None
        else:
            why = self._audio_trouble()
        if why is None and state != 'keying' and time.monotonic() - self._beat > STARVED_S:
            why = f'the transmit thread stopped {time.monotonic() - self._beat:.1f}s ago'
        if (why is None and stream is not None and stream.starved_at is not None
                and self.audio.samples - stream.starved_at >= STARVED_S * FS):
            why = f'transmit audio starved for {(self.audio.samples - stream.starved_at) / FS:.1f}s'
        if why is not None:
            self._drop(f'WATCHDOG -- {why}')

    def screen_failed(self, exc):
        failed = f'the screen failed ({type(exc).__name__}: {exc})'
        down = self.halt(failed)
        self.alarm(f'{failed}; ' + ('unkeyed and left' if down else
                                    'PTT DOWN NOT CONFIRMED -- unkey the rig by hand'))
        return 1

    def _orphaned(self):
        """Why nobody supervises this session any more: the console that owns it
        has gone (its end of the ownership pipe closed, as it does however the
        console ended), or the launcher that started it has."""
        if self._owner is not None:
            try:
                if select.select([self._owner], [], [], 0)[0] and not os.read(self._owner, 64):
                    return 'the console that owns this session is gone'
            except OSError as exc:
                return f'the console that owns this session is gone ({exc})'
        if self._parent is not None and os.getppid() != self._parent:
            return 'the launcher that started this session is gone'
        return None

    def _drop(self, why):
        if self.halt(why):
            self.alarm(f'{why}; PTT dropped')
        else:
            self.alarm(f'{why}; PTT DOWN NOT CONFIRMED -- unkey the rig by hand')

    def _receive(self):
        for chunk in self.lane.poll():
            at, samples = chunk.at, chunk.samples
            if self._rx_next is not None and at > self._rx_next:
                gap = min(at - self._rx_next, round(GAP_CAP_S * FS))
                self._queue(np.zeros(gap, np.float32), np.zeros(gap, bool))
            self._rx_next = at + len(samples)
            index = at + np.arange(len(samples))
            keyed_from = self._key_index if self.state != 'rx' else math.inf
            valid = (index >= self._mute_until) & (index < keyed_from)
            for fault in chunk.faults:
                if fault.cause != 'flush':
                    valid[min(max(fault.index - at, 0), len(valid) - 1)] = False
            heard = np.where(valid, samples, 0).astype(np.float32)
            self._queue(heard, valid)
            self._remember(heard)
        if self._rx_pending_n >= RX_BLOCK_S * FS:
            audio = np.concatenate([a for a, _ in self._rx_pending])
            valid = np.concatenate([v for _, v in self._rx_pending])
            self._rx_pending, self._rx_pending_n = [], 0
            self.stream.feed(audio, valid=valid)
            self._decoded(self.stream.events)

    def _queue(self, audio, valid):
        self._rx_pending.append((audio, valid))
        self._rx_pending_n += len(audio)

    def _remember(self, audio):
        self._recent.append(audio)
        self._recent_n += len(audio)
        while self._recent_n - len(self._recent[0]) >= ACQUIRE_S * FS:
            self._recent_n -= len(self._recent.popleft())

    def recent(self, seconds: float) -> np.ndarray:
        if not self._recent:
            return np.zeros(0, np.float32)
        return np.concatenate(self._recent)[-round(seconds * FS):]

    def _decoded(self, events):
        for event in events:
            self._frames.append(event.accepted)
            if not (event.accepted and event.text):
                continue
            read = sum(self._frames) / len(self._frames)
            if read < self.squelch - (SQUELCH_OPEN - SQUELCH_DROP):
                continue
            self._heard = (self._heard + event.text)[-2000:]
            self._show('RX', event.text, read < self.squelch or not event.shift_known)

    def _track(self):
        if self.state != 'rx':
            return
        due, near = self._acquire
        if due is not None and self.audio.samples >= due and self._recent_n >= ACQUIRE_S * FS / 2:
            self._acquire = (None, None)
            centre, db = fsk.tone_pair(self.recent(ACQUIRE_S), FS, shift=self.rate.shift,
                                       near=near, span=ACQUIRE_SPAN_HZ)
            if db > 10:
                self._retune(centre, f'acquired at {db:.0f} dB')
        if self.audio.samples < self._afc_due:
            return
        self._afc_due = self.audio.samples + round(AFC_EVERY_S * FS)
        if len(self._frames) < SQUELCH_FRAMES or sum(self._frames) / len(self._frames) < self.squelch:
            return
        centre = self._fit(self.recent(AFC_EVERY_S))
        if (abs(centre - self.rx_centre) > AFC_STEP_HZ
                and abs(centre - self.tx_centre) <= AFC_BOUND_HZ):
            self._retune(centre, 'AFC')

    def _fit(self, audio):
        """The centre within ±20 Hz that puts the most energy in a unit-wide band at each tone."""
        power = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2
        freqs = np.fft.rfftfreq(len(audio), 1 / FS)
        total = np.concatenate(([0.0], np.cumsum(power)))

        def band(hz):
            lo, hi = np.searchsorted(freqs, [hz - self.rate.baud / 2, hz + self.rate.baud / 2])
            return total[hi] - total[lo]

        offsets = self.rx_centre + np.arange(-20, 21)
        energy = [band(c + self.rate.shift / 2) + band(c - self.rate.shift / 2) for c in offsets]
        return float(offsets[int(np.argmax(energy))])

    def spectrum(self, width: int):
        """A strip of ±400 Hz around our centre, and the columns of our mark and space."""
        with self._lock:
            audio = self.recent(0.5)
        if len(audio) < FS // 10:
            audio = np.zeros(FS // 2)
        power = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) ** 2 + 1e-12
        freqs = np.fft.rfftfreq(len(audio), 1 / FS)
        edges = np.linspace(self.tx_centre - 400, self.tx_centre + 400, width + 1)
        cells = np.array([power[(freqs >= lo) & (freqs < hi)].max(initial=1e-12)
                          for lo, hi in zip(edges[:-1], edges[1:])])
        db = 10 * np.log10(cells / np.median(cells))
        shades = ' .:-=+*#'
        strip = ''.join(shades[int(np.clip(d / 4, 0, len(shades) - 1))] for d in db)

        def column(hz):
            return int(np.clip((hz - edges[0]) / 800 * width, 0, width - 1))

        return strip, column(self.tx_mark), column(self.tx_space)

    def step(self):
        self._beat = time.monotonic()
        with self._lock:
            self._transmit()
            self._receive()
            self._track()
            if self._tape is not None:
                for chunk in self._tape_lane.poll():
                    self._put(self._disk, 'stream.wav', self._record, chunk.samples,
                              at=chunk.at, size=len(chunk.samples))

    def _record(self, samples):
        self._tape.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())

    def refresh_dial(self):
        """Admission, asked just before the request is sent, rejects a read when it
        sees keying; a dispatch can race that transition, and a late reply is
        discarded. PTT is the RTS line, which a CAT read cannot hold or delay."""
        if self.state != 'rx':
            return
        hz = self._dial_reader(lambda: self.state == 'rx')
        with self._lock:
            if hz and hz != self.dial_hz and self.state == 'rx':
                self.dial_hz = hz
                self._annotate(f'dial {hz}, mark RF {self.mark_rf_hz} Hz')

    def _follow_dial(self):
        while self._running:
            time.sleep(DIAL_EVERY_S)
            self.refresh_dial()

    def start(self):
        self._running = True
        for target in (self._work, self._guard) + ((self._follow_dial,) if self._dial_reader else ()):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    def read_news(self):
        try:
            with self._news.open('rb') as said:
                said.seek(self._news_read)
                text = said.read()
        except OSError:
            return
        whole = text[:text.rfind(b'\n') + 1]
        self._news_read += len(whole)
        for line in whole.decode(errors='replace').splitlines():
            self.note(line)

    def _work(self):
        try:
            self.audio.breathe()
            news_due = 0.0
            while self._running:
                self.step()
                if self._news is not None and time.monotonic() >= news_due:
                    news_due = time.monotonic() + 1.0
                    self.read_news()
                time.sleep(STEP_S)
        except Exception as exc:
            self._failed = (f'the session thread failed ({type(exc).__name__}: {exc}); receive '
                            'has stopped')
            self.alarm(f'{self._failed} and keying is declined')
            raise
        finally:
            if self.state != 'rx':
                self._drop('the session thread stopped while keyed')

    def _guard(self):
        while self._running:
            self.watchdog()
            time.sleep(0.1)

    def close(self):
        """Unkey and stop the threads. The writers stay up, so an alarm from handing
        the hardware back still has somewhere to go; `finish` ends them."""
        if self.state != 'rx':
            self.halt('session closed')
        elif self.ptt_unconfirmed:
            self.abort()
        with self._lock:
            if self.state != 'rx':
                self._unkey('session closed')
            self._running = False
            for kind, text in self._lines.items():
                if text:
                    self._write_line(kind, text)
            for kind in ('transcript', 'stream.wav'):
                if lost := self._lost.pop(kind, None):
                    self._lost_note(kind, *lost)
        for thread in self._threads:
            thread.join(timeout=2)

    def finish(self, timeout=10):
        """Close the records and give back, as lines for the screen, anything that
        could not be written: above all, contacts."""
        deadline = time.monotonic() + timeout
        left, kept = self._contacts.close(timeout)
        for done in (self._transcript, self._tape):
            if done is not None:
                self._disk.put(done.close, force=True)
        disk = self._disk.close(max(0.1, deadline - time.monotonic()))
        pipe = self._pipe.close(max(0.1, deadline - time.monotonic()))
        said = [f'NOT LOGGED: {record.strip()}' for record in left]
        if left:
            said.append(f'those {len(left)} contact(s) are set aside in {kept}, and the next '
                        'session writes them to the log' if kept else
                        f'those {len(left)} contact(s) are NOT in {self.log.path}: copy them in by hand')
        if not disk:
            said.append('the transcript and stream.wav writer had not finished; their ends may be missing')
        if not pipe:
            said.append('the event log (stdout) had not drained; arm.log may end early')
        return said


class _Overtaken(Exception):
    pass


class _Unconfirmed(Exception):
    pass


def parser():
    ap = argparse.ArgumentParser(prog='python -m hfmodem.lanner.qso',
                                 description='keyboard-to-keyboard RTTY on the station rig; '
                                             'the operator keys every over')
    ap.add_argument('--rate', default=str(rtty.AMATEUR), metavar='BAUD/SHIFT[/STOP]')
    ap.add_argument('--centre', type=float, default=CENTRE_HZ, metavar='HZ',
                    help='audio centre of our tones; mark is centre + shift/2')
    ap.add_argument('--dial', type=int, metavar='HZ', help='the dial, for the mark RF figure')
    ap.add_argument('--rig', help='rig profile; default HFMODEM_RIG, else ft891')
    ap.add_argument('--audio-in', required=True)
    ap.add_argument('--audio-out', required=True)
    ap.add_argument('--ptt-device', required=True, help='the serial line whose RTS keys the rig')
    ap.add_argument('--mycall', default='', help='default [station] mycall')
    ap.add_argument('--contest', help='contest id for serials and the ADIF record')
    ap.add_argument('--adif', required=True, help='the QSO log, appended as ADIF')
    ap.add_argument('--session', type=Path, help='directory for transcript.txt and stream.wav')
    ap.add_argument('--drive', type=float, help='default the chain tx_drive')
    ap.add_argument('--launcher', type=int, metavar='PID',
                    help='the shell that started this terminal, which must stay its parent')
    ap.add_argument('--warn', action='append', default=[], metavar='TEXT',
                    help="the launcher's warning, held on the status line all session")
    return ap


def rigctld_dial(admit=lambda: True, *, port=4532):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=0.5) as cat:
            if not admit():
                return None
            cat.sendall(b'f\n')
            return int(cat.recv(64).split()[0])
    except (OSError, ValueError, IndexError):
        return None


def main(argv=None) -> int:
    from hfmodem.core import config, levels, rigs
    from hfmodem.core.audio import StationAudio
    from hfmodem.core.devices import find_device
    from hfmodem.core.ptt import LineKeyer
    from hfmodem.lanner import terminal
    from hfmodem.station.qsolog import QsoLog

    args = parser().parse_args(argv)
    rate = rtty.Rate.parse(args.rate)
    path = os.environ.get(config.STATION_ENV)
    cfg = config.load(path, rig_name=args.rig) if path and Path(path).is_file() else None
    profile = cfg.resolved_profile if cfg else rigs.resolve_profile(args.rig)[1]
    timing = rigs.resolve_tx_timing(rig=args.rig, station_path=path if cfg else '')
    mycall = (args.mycall or (cfg.station.mycall if cfg else '')).upper()
    grid = cfg.station.grid if cfg else ''
    if args.drive is not None:
        drive = args.drive
    elif cfg and isinstance(cfg.audio, config.ChainAudio):
        drive = cfg.audio.drive_for()
    else:
        drive = cfg.audio.tx_drive if cfg else levels.TX_DRIVE
    if not mycall:
        print('warning: no callsign: [station] mycall is unset and --mycall was not given, '
              'so {MYCALL} sends nothing', file=sys.stderr)
    if not grid:
        print('warning: no grid: [station] grid is unset, so {GRID} sends nothing',
              file=sys.stderr)
    if args.session:
        args.session.mkdir(parents=True, exist_ok=True)
    out = terminal.claim_tty()
    session = None

    later = []

    def shout(line):
        """Held until the hardware is handed back, so a terminal or pipe that will not
        take a write cannot hold teardown."""
        later.append(line)

    def tell(lines):
        for line in lines:
            for sink in (sys.__stdout__, sys.stderr):
                try:
                    print(line, file=sink, flush=True)
                except (OSError, ValueError):
                    pass

    def log(line):
        if session is None:
            tell([line])
        else:
            session.say(line)

    def alarm(line):
        if session is None:
            tell([line])
        else:
            session.alarm(line)

    keyer = LineKeyer(args.ptt_device, log, alarm)
    audio = StationAudio(input_device=find_device(args.audio_in, 'in', required=True),
                         output_device=find_device(args.audio_out, 'out', required=True),
                         tx_latency_n=timing.latency_samples)

    stopping = False

    def down(signum, _frame):
        nonlocal stopping
        first = not stopping
        # A group signal also reaches the launcher's forwarding trap. Repeated
        # delivery still unkeys, but must not interrupt the cleanup already begun.
        stopping = True
        if session is not None:
            session.halt(f'signal {signum}')
        if first:
            raise SystemExit(128 + signum)

    def halt_only(signum, _frame):
        if session is not None:
            session.halt(f'signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, down)
    try:
        keyer.arm()
        audio.open()
        session = Session(audio, keyer, QsoLog(args.adif, mycall, grid=grid, contest=args.contest),
                          rate=rate, centre_hz=args.centre, dial_hz=args.dial, drive=drive,
                          settle_s=profile['settle'], tr_switch_s=profile['tr_switch_s'],
                          mycall=mycall, grid=grid, directory=args.session, out=out,
                          dial_reader=rigctld_dial if args.dial else None,
                          warnings=args.warn, shout=shout,
                          owner=int(os.environ[OWNER_ENV]) if os.environ.get(OWNER_ENV) else None,
                          parent=args.launcher)
        try:
            return terminal.run(session)
        except Exception as exc:
            return session.screen_failed(exc)
    finally:
        stopping = True
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, halt_only)
        steps = [session.close] if session else []
        for step in steps + [audio.close, keyer.hand_back]:
            try:
                step()
            except Exception as exc:
                shout(f'!! shutdown: {type(exc).__name__}: {exc}')
        if session is not None:
            later.extend(f'!! {line}' for line in session.finish())
        tell(later)


if __name__ == '__main__':
    raise SystemExit(main())
