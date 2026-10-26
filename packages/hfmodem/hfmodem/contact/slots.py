# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-only watch over every VarAC channel the receiver's passband holds.

Each channel inside the passband is mixed to 1500 Hz and read by the CQ and beacon
detector, the connect-request scanner and the BW500 frame reader. Keyed idle
replies are read against known endpoints; an in-channel energy check against
the gaps between channels catches what none of them decode.
Nothing here keys.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import multiprocessing
import os
import queue
import re
import threading
import time

import numpy as np
from scipy.signal import firwin, hilbert, lfilter, oaconvolve, remez

from ..kestrel.arq import phy
from ..kestrel.coding.crc import crc16_genibus
from ..kestrel.monitor import HandshakeScanner
from ..kestrel.rx import beacon as B, broadcast as UI
from .broadcast import Broadcast
from ..kestrel.rx.bw500_locator import Receiver
from ..kestrel.rx.cq_stream import CQDetector
from ..kestrel.vara import compression, vara_frames as VF
from ..monitor.cw import CwReader
from .passive import HighSpeed, IdleReplies
from .varac import BEACON_TEXT, DEFAULT_CODEC, SLOT_WAIT_S, label

FS = 48000
CENTRE_HZ = 1500
HALF_HZ = 250
BLOCK = FS // 2
#: The passband is measured, as the median noise spectrum of the receive audio: for
#: MEASURE_S once the session starts and after every retune, then every REMEASURE_S.
#: A channel counts when its whole 500 Hz band sits inside the EDGE_DB points.
MEASURE_S = 30
REMEASURE_S = 300
EDGE_DB = 8
LEVELS_DB = (3, 6, EDGE_DB, 10)
BIN_HZ = 10
SPAN_HZ = 6000
#: Reported again when an edge moves this far or the channels change.
MOVED_HZ = 50
#: The receive audio falls off a cliff below 100 Hz whatever the filter: -6 dB at
#: 94 Hz and -10 dB at 82 at both IF shifts measured on 10-23.
CUTOFF_HZ = 100
#: Until the session has measured its own: the FTDX10 at WIDTH 4000 and IF SHIFT +500,
#: -6 dB at 94 and 3023 Hz (core.rigs records the measurement).
PASSBAND_HZ = (CUTOFF_HZ, 3023)
#: TX BPF SEL 1, as the arm's menu readout reads it.
TX_BPF_HZ = (100, 2900)
#: How long a VARA link reads busy after the last thing read from it. Stock polls an
#: idle link about every 12 s, so a link whose two sides are both heard is never silent
#: for 30 s; but often one side is all the footprint holds. On the 10-24 tape slot 1
#: carried AI7EQ's side of a chat with KD5LEE, and went 183 s between decoded frames
#: while KD5LEE, unheard, sent.
LINK_HOLD_S = 240
#: ...and once a close is heard (<DISC>, a CW ident), only this.
CLOSED_HOLD_S = 30
#: A link with no frame read for this long is idle, polling, or the side we cannot hear;
#: AI7EQ's chat on the 10-24 tape went up to 80 s between decoded frames.
IDLE_S = 60
#: A CQ naming a slot names the called station of a link that starts there this soon.
CQ_WAIT_S = SLOT_WAIT_S + 30
IDENT_STEP = 8
#: Energy: a 0.5 s block counts when the channel stands this far over its own floor,
#: the floor a low percentile of its last ten minutes against the inter-channel gaps.
SIGNAL_DB = 3.0
FLOOR_BLOCKS = 1200
FLOOR_PERCENTILE = 10
#: ...and the channel shows a signal while a third of its last 20 s counted, once that
#: has held STABLE_S, and for SIGNAL_HOLD_S after that last held. The slot 11 session
#: on the 10-23 tape decoded no frame. Energy never holds a link: the calling
#: frequency's beacons would keep one busy for good.
WINDOW_BLOCKS = 40
SIGNAL_SHARE = 1 / 3
STABLE_S = 5
SIGNAL_HOLD_S = 60
CALL = re.compile(r'(?<![A-Z0-9/])(?=[A-Z0-9]{4})(?:[A-Z]{1,2}|[0-9][A-Z]|[A-Z][0-9])[0-9][A-Z]{1,4}'
                  r'(?:-[0-9]{1,2})?(?![A-Z0-9])')
TAG = re.compile(r'<([A-Z]{1,6})(?::[^<>]*)?>')
# Only a standalone opening report identifies its declared sender. A callsign
# mentioned in chat, a directory, quoted text or a message body is not an endpoint.
SIGNED = re.compile(r'(?:[0-9]+ )?de ([A-Z0-9/-]+) <R[+-][0-9]{2}>'
                    r'(?: <S:[A-Z]+>)?')
TEXT_KEEP = 160


def channels(dial, calling, *, offset=0, codec=DEFAULT_CODEC, passband=PASSBAND_HZ):
    """`[(slot, audio Hz)]` for the calling frequency (slot None) and each slot whose
    500 Hz band the passband holds, heard with the rig `offset` Hz off `dial`."""
    rig = dial + offset
    grid = [(None, 0)] + [(slot, codec.offset_hz(slot)) for slot in codec.slots]
    if dial != calling and all(calling + hz != dial for _, hz in grid):
        grid.append((f'hz:{dial}', dial - calling))
    held = [(slot, CENTRE_HZ + calling + hz - rig) for slot, hz in grid]
    return sorted(((slot, audio) for slot, audio in held
                   if passband[0] <= audio - HALF_HZ and audio + HALF_HZ <= passband[1]),
                  key=lambda item: item[1])


def wide_offset(passband=PASSBAND_HZ, tx=TX_BPF_HZ, codec=DEFAULT_CODEC):
    """The dial offset, in 50 Hz steps, that holds the most channels inside `passband`
    from the calling frequency while our own, sent and heard at 1500 - offset, stays
    inside the TX bandpass `tx`; of those, the one with the most room at the passband's
    edges, then the nearest the channel."""
    def held(offset):
        return channels(0, 0, offset=offset, codec=codec, passband=passband)

    def score(offset):
        audio = [hz for _, hz in held(offset)]
        room = min(min(hz - HALF_HZ - passband[0], passband[1] - hz - HALF_HZ) for hz in audio)
        return len(audio), room, -abs(offset)

    ours = [offset for offset in range(-1500, 1501, 50)
            if tx[0] <= CENTRE_HZ - offset - HALF_HZ and CENTRE_HZ - offset + HALF_HZ <= tx[1]
            and (None, CENTRE_HZ - offset) in held(offset)]
    return max(ours, key=score, default=0)


@dataclass(frozen=True)
class Passband:
    """The receive audio's edges, `{dB down: (low Hz, high Hz)}`."""
    edges: dict

    @property
    def usable(self):
        lo, hi = self.edges[EDGE_DB]
        return max(lo, CUTOFF_HZ), hi

    def moved(self, other):
        return other is None or any(abs(a - b) > MOVED_HZ for level in LEVELS_DB
                                    for a, b in zip(self.edges[level], other.edges[level]))

    def __str__(self):
        return ', '.join(f'-{level} dB {lo:.0f}-{hi:.0f}' for level, (lo, hi) in self.edges.items()
                         if level != EDGE_DB) + ' Hz'


def measure(spectra):
    """The Passband of BLOCK-sample power spectra of receiver noise: their median per
    BIN_HZ, against the median of the bins within 10 dB of the top. The noise is not
    flat across the passband, and its 95th percentile alone put the -3 dB points of
    30 s windows of the 10-23 tape anywhere from 1042 to 1943 Hz."""
    per = BIN_HZ * BLOCK // FS
    n = SPAN_HZ // BIN_HZ
    binned = np.asarray(spectra)[:, :n * per].reshape(len(spectra), n, per).mean(axis=2)
    curve = np.maximum(np.median(binned, axis=0), 1e-30)
    top = np.percentile(curve, 95)
    db = 10 * np.log10(curve / np.median(curve[curve >= top / 10]))
    f = (np.arange(n) * per + (per - 1) / 2) * FS / BLOCK
    return Passband({level: _edges(f, db, -level) for level in LEVELS_DB})


def _edges(f, db, level):
    """Where the longest run at or above `level` crosses it, interpolated between bins."""
    above = np.concatenate(([False], db >= level, [False]))
    starts, ends = np.flatnonzero(above[1:] & ~above[:-1]), np.flatnonzero(above[:-1] & ~above[1:]) - 1
    i, j = max(zip(starts, ends), key=lambda run: run[1] - run[0])

    def cross(a, b):
        return f[a] + (level - db[a]) / (db[b] - db[a]) * (f[b] - f[a])
    return (cross(i - 1, i) if i else f[i]), (cross(j + 1, j) if j + 1 < len(f) else f[j])


@dataclass(frozen=True)
class Measured:
    """What the monitor says when the passband it measures changes: the edges, the
    channels it now watches, and the offset `wide_offset` would pick from them."""
    passband: Passband
    slots: tuple
    offset: int
    wide: int

    def line(self):
        said = f'passband {self.passband}; watching {", ".join(label(slot) for slot in self.slots)}'
        if self.offset and self.wide != self.offset:
            side = 'below' if self.wide < 0 else 'above'
            said += f' (--monitor-wide would sit the dial {abs(self.wide)} Hz {side} the channel)'
        return said


def _named(result):
    """A handshake burst that regenerated a callsign, read on the BW500 lattice, or
    locked its whole preamble. Six of eight preamble tones with no payload came up
    twice on slot 11 over 1540 s of the 10-23 tape and are not taken."""
    pre = re.search(r'(\d+)/(\d+) preamble', result.quality)
    return bool(result.gateway or 'BW500' in result.info or (pre and pre[1] == pre[2]))


@dataclass(frozen=True)
class Activity:
    """One channel as last judged: `busy` is a VARA link, held LINK_HOLD_S after the
    last thing read from it; `signal` is energy or a CQ or beacon. `caller` and
    `called` name the link's stations where something said which is which, the caller
    with a `?` while only its frames' trailer names it; `one_side` is whether all that
    was heard keying came from one of them. Times are seconds of monitored audio, `at`
    the judging's."""
    slot: int | str | None
    audio_hz: int
    state: str = 'quiet'
    heard: float | None = None
    calls: tuple = ()
    level: int | None = None
    tags: tuple = ()
    home: bool = False
    caller: str = ''
    called: str = ''
    stage: str = ''
    since: float | None = None
    at: float = 0.
    sn: float | None = None
    one_side: bool = False

    @property
    def name(self):
        return label(self.slot)

    def who(self):
        """`KD5LEE? ↔ AI7EQ`, `NC3Z ↔ DL4VCW`, `AI7EQ ↔ ?`."""
        others = [c for c in self.calls if c not in (self.caller.rstrip('?'), self.called)]
        if self.caller or self.called:
            return ', '.join([f'{self.caller or "?"} ↔ {self.called or "?"}', *others])
        if len(others) == 1:
            return f'{others[0]} ↔ ?'
        return ' ↔ '.join(others) if len(others) == 2 else ', '.join(others)


def payload_text(payload, *, text_encoded=False):
    """Classify already CRC-validated bytes, without assuming Latin-1 is chat."""
    original = payload
    error = None
    if text_encoded:
        decoded = compression.decode_text(payload)
        payload, error = decoded.data, decoded.error
    if error is None:
        try:
            text = payload.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            if text and all(c.isprintable() or c in '\r\n\t' for c in text):
                return text, None
    preview_bytes = original[:32]
    try:
        preview = repr(preview_bytes.decode('utf-8'))
    except UnicodeDecodeError:
        preview = repr(preview_bytes)[1:]  # Byte escapes, without pretending Latin-1 is text.
    detail = (f'decoded payload bytes ({len(original)} bytes); '
              'text encoding/compression not resolved; preview ' + preview
              + ('…' if len(original) > 32 else '') + '; hex ' + preview_bytes.hex()
              + ('…' if len(original) > 32 else ''))
    return None, detail


@dataclass(frozen=True)
class Said:
    """One line of a channel's QSO: a `stage` (connect request, link up, idle or
    polling, QSY, closing, CW ident, quiet), a `text` line its frames carried, or a
    new `level`. `station` sent it, where the text is signed."""
    slot: int | str | None
    at: float
    kind: str
    text: str
    station: str = ''


@dataclass
class _Evidence:
    since: float | None = None
    active: float = -1e9
    frame: float = -1e9
    closed: bool = False
    signal: float = -1e9
    signal_since: float = -1e9
    heard: float | None = None
    caller: str = ''
    called: str = ''
    keyed: str = ''
    calls: list = field(default_factory=list)
    sent: set = field(default_factory=set)
    level: int | None = None
    sn: float | None = None
    tags: list = field(default_factory=list)
    text: str = ''
    stage: str = ''
    last_chunk: tuple | None = None


class Mixer:
    """One channel, band-limited to +-375 Hz and moved to 1500 Hz real audio."""
    TAPS = 801

    def __init__(self, audio_hz):
        n = np.arange(self.TAPS) - self.TAPS // 2
        self.h = 2 * firwin(self.TAPS, 375, fs=FS) * np.exp(2j * np.pi * audio_hz * n / FS)
        self.tail = np.zeros(self.TAPS - 1, complex)
        self.hz = CENTRE_HZ - audio_hz
        self.n = 0

    def __call__(self, x):
        joined = np.concatenate((self.tail, x))
        self.tail = joined[-(self.TAPS - 1):]
        y = oaconvolve(joined, self.h, mode='valid')
        k = self.n + np.arange(len(y))
        self.n = (self.n + len(y)) % FS
        return np.real(y * np.exp(2j * np.pi * self.hz * k / FS))


class _Channel:
    def __init__(self, slot, audio_hz, calls, home):
        self.slot, self.audio_hz, self.home = slot, audio_hz, home
        self.mixer = Mixer(audio_hz)
        self.cq = CQDetector(FS)
        self.requests = HandshakeScanner(calls)
        self.frames = Receiver(observe_pairs=True)
        self.high = HighSpeed()
        self.idles = IdleReplies()
        # Stock keys its closing ident on a mark at the channel's centre; the space,
        # 750 Hz below, is outside the mix and leaves the mark on-off keyed. The mix
        # holds nothing above 1900 Hz, so it is read at an eighth of the rate.
        self.ident = CwReader(FS // IDENT_STEP, CENTRE_HZ)
        self.ratios = deque(maxlen=FLOOR_BLOCKS)
        self.hot = deque(maxlen=WINDOW_BLOCKS)
        self.excess = deque(maxlen=WINDOW_BLOCKS // 2)


class SlotMonitor:
    """Feed it the receive audio as heard at `dial` (the dial the modem works; the rig
    sits `offset` Hz off it). `feed` returns `('state', slot, Activity)` when a
    channel's state or its named stations change, `('qso', slot, Said)` for each line
    of a channel's QSO, and `('cq' | 'beacon', slot, result)` for each frame heard off
    the modem's own channel, which the host already reports.
    `('passband', None, Measured)` says what a measurement of the passband changed;
    `passband` is the one watched until the first."""

    def __init__(self, dial, calling, *, offset=0, mycall='', codec=DEFAULT_CODEC, passband=PASSBAND_HZ):
        self.calling, self.offset, self.codec, self.passband = calling, offset, codec, passband
        self._initial_passband = passband
        self.mycall = mycall.upper()
        self.calls = [mycall.upper()] if mycall else []
        self.samples = 0
        self._reported = None
        self._evidence: dict = {}
        self._judged: dict = {}
        self._cqs: dict = {}
        self.retune(dial)

    @property
    def now(self):
        return self.samples / FS

    def retune(self, dial, calling=None):
        """A new `calling` frequency is another band: what was read of the last one is
        dropped, and the passband measured there is reported again once it is measured here."""
        if calling is not None and calling != self.calling:
            self.calling, self._reported = calling, None
            self.passband = self._initial_passband
            self.calls = [self.mycall] if self.mycall else []
            self._evidence, self._judged, self._cqs = {}, {}, {}
        self.dial = dial
        self._high_cursor = 0
        self._high_credit = .2
        self._channels = []
        self._place()
        self._pending = np.empty(0)
        self._spectra, self._measure_from = [], self.now

    def _place(self):
        """Watch the channels the passband holds, keeping the readers of those it held."""
        rig = self.dial + self.offset
        kept = {(ch.slot, ch.audio_hz): ch for ch in self._channels}
        self._channels = [kept.get((slot, audio)) or
                          _Channel(slot, audio, self.calls, audio == CENTRE_HZ + self.dial - rig)
                          for slot, audio in channels(self.dial, self.calling, offset=self.offset,
                                                      codec=self.codec, passband=self.passband)]
        grid = [CENTRE_HZ + self.calling + hz - rig for hz in range(-4500, 4501, 750)]
        lo, hi = self.passband
        self._gaps = [(g + 300, g + 450) for g in grid if lo + 50 <= g + 300 and g + 450 <= hi - 50]
        return set(kept) != {(ch.slot, ch.audio_hz) for ch in self._channels}

    def _measure(self, spectrum):
        # Keyed, the watch is fed silence, which says nothing about the receiver.
        if self.now < self._measure_from or not spectrum.any():
            return []
        self._spectra.append(spectrum[:SPAN_HZ * BLOCK // FS])
        if len(self._spectra) < MEASURE_S * FS // BLOCK:
            return []
        measured = measure(self._spectra)
        self._spectra, self._measure_from = [], self.now + REMEASURE_S
        self.passband = measured.usable
        if not (self._place() or measured.moved(self._reported)):
            return []
        self._reported = measured
        return [('passband', None, Measured(measured, tuple(ch.slot for ch in self._channels),
                                            self.offset, wide_offset(measured.usable, codec=self.codec)))]

    def table(self):
        return [self._judge(ch) for ch in self._channels]

    def heard(self, call, slot=None):
        """A station heard anywhere; a CQ's `slot` is where it waits for an answer."""
        call = call.upper()
        if call and call not in self.calls:
            self.calls.append(call)
            for ch in self._channels:
                ch.requests.calls.append(call)
        if call and slot is not None:
            self._cqs[slot] = (call, self.now)

    def feed(self, samples):
        x = np.concatenate((self._pending, np.asarray(samples, float).reshape(-1)))
        out = []
        while len(x) >= BLOCK:
            out += self._block(x[:BLOCK])
            x = x[BLOCK:]
        self._pending = x
        return out

    def _block(self, x):
        self.samples += len(x)
        spectrum = abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / FS)
        out = self._measure(spectrum)
        gap = np.mean([spectrum[(f >= a) & (f < b)].mean() for a, b in self._gaps]) if self._gaps else 0
        for ch in self._channels:
            ev = self._evidence.setdefault(ch.slot, _Evidence())
            band = spectrum[(f >= ch.audio_hz - HALF_HZ) & (f < ch.audio_hz + HALF_HZ)].mean()
            if gap > 0:
                self._energy(ch, ev, 10 * np.log10(max(band, 1e-30) / gap))
            out += self._read(ch, ev, ch.mixer(x))
            out += self._idle(ch, ev)
            judged = self._judge(ch)
            before = self._judged.get(ch.slot)
            if before is None and judged.state == 'quiet':
                self._judged[ch.slot] = judged
            elif (before is None or before.state != judged.state
                  or (judged.state != 'quiet' and set(judged.calls) - set(before.calls))):
                self._judged[ch.slot] = judged
                if before is not None and before.state == 'busy' and judged.state != 'busy':
                    out.append(self._said(ch, 'stage', 'quiet; the link is over'))
                out.append(('state', ch.slot, judged))
        # One bounded emission across the entire watch per half-second block.
        # Rotate priority so simultaneous busy channels do not starve the last.
        # Elapsed-time debt limits high-speed work to 10% of audio time over
        # time, with 0.2s burst credit. One indivisible codec call may overshoot;
        # pending candidates then wait or expire with an explicit gap notice.
        self._high_credit = min(.2, self._high_credit + len(x) / FS * .10)
        for i in range(len(self._channels)):
            if self._high_credit < 0:
                break
            j = (self._high_cursor + i) % len(self._channels)
            ch = self._channels[j]
            if ch.high.ready:
                self._high_cursor = (j + 1) % len(self._channels)
                ev = self._evidence[ch.slot]
                began = time.monotonic()
                frames = ch.high.read()
                self._high_credit -= time.monotonic() - began
                coverage = self._coverage(ch, ev)
                out += coverage
                for frame in frames:
                    if coverage:
                        ev.text, ev.last_chunk = '', None
                    out += self._frame(ch, ev, frame)
                if coverage:
                    ev.text, ev.last_chunk = '', None
                break
        return out

    def _coverage(self, ch, ev):
        notices = ch.high.drain_notices()
        if notices:
            ev.text, ev.last_chunk = '', None
        return [self._said(ch, 'coverage', notice) for notice in notices]

    def _said(self, ch, kind, text, station=''):
        return 'qso', ch.slot, Said(ch.slot, self.now, kind, text, station)

    def _lit(self, ev, decoded=False):
        if self.now - ev.signal >= SIGNAL_HOLD_S:
            ev.signal_since = self.now - STABLE_S * decoded
        ev.signal = ev.heard = self.now

    def _energy(self, ch, ev, ratio):
        ch.ratios.append(ratio)
        if len(ch.ratios) < WINDOW_BLOCKS:
            return
        excess = ratio - np.percentile(ch.ratios, FLOOR_PERCENTILE)
        ch.excess.append(excess)
        ch.hot.append(excess > SIGNAL_DB)
        if ch.hot[-1] and sum(ch.hot) >= SIGNAL_SHARE * WINDOW_BLOCKS:
            self._lit(ev)

    def _link(self, ch, ev):
        """Whether this is a new link: one that starts on a slot a CQ named soon before
        is that CQ's answer."""
        ev.active = ev.heard = self.now
        if ev.since is not None:
            return False
        ev.since = self.now
        call, at = self._cqs.get(ch.slot, ('', -1e9))
        if self.now - at <= CQ_WAIT_S and not ev.called:
            ev.called = call
        return True

    def _read(self, ch, ev, y):
        out = []
        for call in ch.idles.feed(y, (self.mycall, ev.caller, ev.called, ev.keyed)):
            self._link(ch, ev)
            self._lit(ev, decoded=True)
            ev.stage = 'idle or polling'
            out.append(self._said(ch, 'control', f'idle reply addressed to {call}'))
        ch.high.feed(y)
        out += self._coverage(ch, ev)
        for result in ch.cq.feed(y):
            kind = 'broadcast' if isinstance(result, Broadcast) else 'cq' if hasattr(result, 'call') else 'beacon'
            beacon = BEACON_TEXT.fullmatch(result.text) if kind == 'beacon' else None
            name = result.source if kind == 'broadcast' else result.call if kind == 'cq' else beacon[3] if beacon else ''
            if name and name.split('-')[0] == self.mycall.split('-')[0]:
                continue
            self._lit(ev, decoded=True)
            if name:
                slot = None
                if kind == 'cq':
                    try:
                        slot = self.codec.decode(name)[1]
                    except ValueError:
                        pass
                self.heard(name.split('-')[0], slot)
            if not ch.home:
                out.append((kind, ch.slot, result))
        for _, result in ch.requests.push(y):
            if not _named(result):
                continue
            self._link(ch, ev)
            ev.stage = 'connecting'
            if result.kind == 'CR':
                ev.sent.add('caller')
                ev.called = result.gateway or ev.called
                out.append(self._said(ch, 'stage', f'connect request: {ev.caller or "?"} calling {ev.called or "?"}'))
            else:
                ev.sent.add('called')
                ev.caller = ev.caller or result.gateway
                out.append(self._said(ch, 'stage', f'connect response to {ev.caller or "?"}'))
        for word in ch.ident.push(y[::IDENT_STEP]):
            call = word.get('station', '')
            if call and (word['grade'] == 'confirmed' or call in (ev.caller, ev.called, ev.keyed, *ev.calls)):
                self._link(ch, ev)
                ev.closed, ev.stage = True, 'closing'
                ev.sent.add(call)
                if call not in ev.calls:
                    ev.calls.append(call)
                self.heard(call.split('-')[0])
                out.append(self._said(ch, 'stage', f'CW ident {call}'))
        losses = ch.frames.overflows + ch.frames.expired
        try:
            overs = ch.frames.feed(y)
        except RuntimeError:
            ch.frames, overs = Receiver(observe_pairs=True), []
        gap = ch.frames.overflows + ch.frames.expired > losses
        if gap:
            ev.text, ev.last_chunk = '', None
            out.append(self._said(ch, 'coverage',
                'passive base-data gap: candidate budget exhausted; validated frames retained'))
        for over in overs:
            for frame in over.frames:
                if frame.crc_ok:
                    ch.high.note_base(frame.start, over.end)
                    out += self._frame(ch, ev, frame)
            if gap:
                ev.text, ev.last_chunk = '', None
        return out

    def _frame(self, ch, ev, frame):
        body = frame.frame_bytes
        if B.parse_payload(body) is not None or UI.is_broadcast_frame(frame):
            self._lit(ev, decoded=True)
            return []
        new = self._link(ch, ev)
        ev.frame = self.now
        if frame.is_connect or VF.is_link_setup(body):
            caller = VF.caller_from_link_setup(body)
            if CALL.fullmatch(caller):
                ev.caller = caller
                ev.sent.add('caller')
                self.heard(caller.split('-')[0])
            ev.stage, ev.level = 'link up', frame.level
            return [self._said(ch, 'stage', f'link up: {ev.caller or "?"} calling {ev.called or "?"}, '
                                            f'level {frame.level}')]
        sn = max(ch.excess, default=None)
        out = []
        if new:
            out.append(self._said(ch, 'stage', f'link heard at level {frame.level}'))
        elif frame.level != ev.level:
            out.append(self._said(ch, 'level', f'level {frame.level}' + (f', {sn:+.0f} dB' if sn is not None else '')))
        ev.level, ev.sn = frame.level, sn
        body = body[:-2]
        # A full frame's kind01 has no trailer. Compressed bytes can contain
        # a literal14, so delimiter-only extraction would truncate clean text.
        payload = (body[:-1] if body[-1] & 3 == 1 else
                   phy.vara_payload(body, caller=ev.caller or ev.keyed or None, body_len=len(body)))
        text, detail = payload_text(payload, text_encoded=bool(body[-1] & compression.TEXT_FLAG))
        if text is None:
            ev.text = ''  # Never join text across an unresolved binary/compressed gap.
            out.append(self._said(ch, 'payload', detail))
            return out
        if ev.last_chunk is not None:
            previous, level, at = ev.last_chunk
            if (self.now - at < 30 and text and previous and
                    (text == previous or frame.level < level and previous.startswith(text))):
                out.append(self._said(ch, 'coverage',
                    'possible retransmission; showing this decoded chunk separately'))
        ev.last_chunk = (text, frame.level, self.now)
        self._keyed(ev, body[:-2])
        ev.text = (ev.text + text)[-TEXT_KEEP:]
        signed = SIGNED.fullmatch(text.strip())
        station = signed[1] if signed and CALL.fullmatch(signed[1]) else ''
        if ev.caller and ev.called and station not in (ev.caller, ev.called):
            station = ''  # An embedded declaration cannot add a third endpoint.
        if station:
            ev.sent.add(station)
            # Of a link's two stations, one that signs and is not the known one is the other.
            if not ev.caller and ev.called and station != ev.called:
                ev.caller, ev.keyed = station, ''
            elif not ev.called and station != (ev.caller or ev.keyed):
                ev.called = station
            if station not in ev.calls:
                ev.calls.append(station)
                self.heard(station.split('-')[0])
        if ev.stage not in ('closing', 'QSY'):
            ev.stage = 'chatting'
        out += [self._said(ch, 'text', line.strip(), station)
                for line in text.replace('\r', '\n').split('\n') if line.strip()]
        for tag in TAG.findall(ev.text):
            if tag in ev.tags:
                continue
            ev.tags = (ev.tags + [tag])[-6:]
            if tag == 'DISC':
                ev.closed, ev.stage = True, 'closing'
                out.append(self._said(ch, 'stage', 'closing <DISC>'))
            elif tag in ('Q', 'QF'):
                ev.stage = 'QSY'
                out.append(self._said(ch, 'stage', 'QSY ' + re.search(rf'<{tag}[:>][^<>]*>?', ev.text)[0]))
        return out

    def _keyed(self, ev, body):
        """The caller, by the trailer a short frame closes with: 0x14 and the high byte
        of the caller's callsign CRC-16/GENIBUS (kestrel.arq.phy.vara_body), whichever
        side sent it. Taken from the link's evidenced endpoints, else those heard
        anywhere, when one alone fits; one in 256 others fits too, hence the `?`."""
        tail = body.rstrip(b'\0')
        if ev.caller or ev.keyed or len(tail) < 2 or tail[-2] != phy._VARA_TRAILER:
            return
        mine = self.mycall.split('-')[0]
        for pool in (ev.calls, self.calls):
            fits = [c for c in dict.fromkeys(pool)
                    if c not in (ev.called, mine) and crc16_genibus(c.encode()) >> 8 == tail[-1]]
            if fits:
                ev.keyed = fits[0] if len(fits) == 1 else ''
                return

    def _idle(self, ch, ev):
        if (ev.since is None or ev.stage in ('idle or polling', 'closing')
                or self.now - max(ev.frame, ev.since) < IDLE_S):
            return []
        ev.stage = 'idle or polling'
        return [self._said(ch, 'stage', 'idle or polling')]

    def _judge(self, ch):
        ev = self._evidence.setdefault(ch.slot, _Evidence())
        now = self.now
        if ev.since is not None and now - ev.active >= (CLOSED_HOLD_S if ev.closed else LINK_HOLD_S):
            ev = self._evidence[ch.slot] = _Evidence(heard=ev.heard)
        if ev.since is not None:
            state = 'busy'
        elif now - ev.signal < SIGNAL_HOLD_S and ev.signal - ev.signal_since >= STABLE_S:
            state = 'signal'
        else:
            state = 'quiet'
        caller = ev.caller or ev.keyed
        calls = tuple(dict.fromkeys([c for c in (caller, ev.called) if c] + ev.calls))
        sides = {'caller' if s == caller else 'called' if s == ev.called else s for s in ev.sent}
        return Activity(ch.slot, ch.audio_hz, state, ev.heard, calls, ev.level, tuple(ev.tags), ch.home,
                        ev.caller or (f'{ev.keyed}?' if ev.keyed else ''), ev.called, ev.stage, ev.since,
                        now, ev.sn, len(sides) == 1)


# Passive decoder shutdown must leave room inside the launcher's two-second quit contract.
WATCH_CLOSE_GRACE_S = .1
WATCH_CLOSE_REAP_S = .2


class SlotWatch:
    """A SlotMonitor in a child process at nice 10, so its reading never holds the
    modem's interpreter while an answer is due. `put` takes audio from the capture
    callback without blocking it, `poll` hands the session what changed, `table`
    the channels as last judged."""

    def __init__(self, dial, calling, **monitor):
        self._calling, self._generation = calling, 0
        context = multiprocessing.get_context('spawn')
        self._in, self._out = context.Queue(maxsize=64), context.Queue(maxsize=512)
        self._dropped = 0
        self._hints_lost = 0
        self._commands = deque()
        self._command_lock = threading.Lock()
        self._table, self._now, self._found = SlotMonitor(dial, calling, **monitor).table(), 0., []
        self._process = context.Process(target=_watch, args=(self._in, self._out, dial, calling, monitor),
                                        daemon=True)
        self._process.start()

    def _flush_controls(self):
        # Called with the parent lock held; every queue operation is nonblocking.
        while self._commands:
            try:
                self._in.put_nowait(self._commands[0])
            except queue.Full:
                return False
            self._commands.popleft()
        if getattr(self, '_hints_lost', 0):
            try:
                self._in.put_nowait(('hint-gap', self._hints_lost))
            except queue.Full:
                return False
            self._hints_lost = 0
        return True

    def _controls(self):
        if not hasattr(self, '_commands'):
            self._commands, self._command_lock = deque(), threading.Lock()
        if not self._command_lock.acquire(blocking=False):
            return False
        try:
            return self._flush_controls()
        finally:
            self._command_lock.release()

    def put(self, frame):
        # Capture never waits. Hold one lock across pending controls, gap and
        # audio so a UI retune cannot slip between them and mislabel old audio.
        if not self._command_lock.acquire(blocking=False):
            self._dropped += len(frame)
            return
        try:
            if not self._flush_controls():
                self._dropped += len(frame)
                return
            try:
                if self._dropped:
                    self._in.put_nowait(('gap', self._dropped))
                    self._dropped = 0
                self._in.put_nowait(frame)
            except queue.Full:
                self._dropped += len(frame)
        finally:
            self._command_lock.release()

    def retune(self, dial, calling=None):
        if not hasattr(self, '_commands'):
            self._commands, self._command_lock = deque(), threading.Lock()
        with self._command_lock:
            self._generation += 1
            self._found, self._table = [], []
            if calling is not None and calling != self._calling:
                self._calling, self._table = calling, []
            # The latest tune carries authoritative band state, even when an
            # unsent band change was followed by a slot-only tune.
            self._commands.clear()
            self._commands.append(('tune', dial, self._calling, self._generation))
            self._flush_controls()

    def heard(self, call, slot=None):
        if not hasattr(self, '_commands'):
            self._commands, self._command_lock = deque(), threading.Lock()
        with self._command_lock:
            command = ('heard', call, slot)
            if command not in self._commands:
                if len(self._commands) < 64:
                    self._commands.append(command)
                else:
                    self._hints_lost = getattr(self, '_hints_lost', 0) + 1
            self._flush_controls()

    def close(self):
        if getattr(self, '_closed', False):
            return
        try:
            try:
                self._in.put_nowait(None)
            except (queue.Full, ValueError):
                pass
            self._process.join(timeout=WATCH_CLOSE_GRACE_S)
            if self._process.is_alive():
                # _watch owns only CPU decoders and these queues: no rig, audio,
                # recording or CAT resource needs a graceful hardware release.
                self._process.terminate()
                self._process.join(timeout=WATCH_CLOSE_REAP_S)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=WATCH_CLOSE_REAP_S)
            if self._process.is_alive():
                raise RuntimeError('Passive slot decoder did not stop after kill')
            self._closed = True
        finally:
            # A stalled reader must not make Queue's feeder flush part of quit.
            for channel in (self._in, self._out):
                channel.cancel_join_thread()
                channel.close()

    def _drain(self):
        self._controls()
        for _ in range(512):
            try:
                generation, record = self._out.get_nowait()
            except queue.Empty:
                return
            if generation != self._generation and record[0] != 'error':
                continue
            if record[0] == 'table':
                self._table, self._now = record[1], record[2]
            else:
                if len(self._found) >= 512:
                    self._found = [('qso', a.slot, Said(a.slot, self._now, 'coverage',
                        'passive transcript gap: output backlog exceeded 512 events'))
                        for a in self._table]
                self._found.append(record)

    def poll(self):
        self._drain()
        found, self._found = self._found, []
        return found

    def table(self):
        self._drain()
        return self._table, self._now


def _watch(inbox, outbox, dial, calling, monitor):
    os.nice(10)
    generation = 0
    try:
        watch = SlotMonitor(dial, calling, **monitor)
        while (item := inbox.get()) is not None:
            if isinstance(item, tuple):
                if item[0] == 'tune':
                    _, dial, calling, generation = item
                    watch.retune(dial, calling)
                    outbox.put((generation, ('table', watch.table(), watch.now)))
                elif item[0] == 'hint-gap':
                    for ch in watch._channels:
                        outbox.put((generation, watch._said(ch, 'coverage',
                            f'passive station hints skipped: {item[1]}; identities may remain unknown')))
                elif item[0] == 'gap':
                    watch.samples += item[1]
                    watch.retune(watch.dial)
                    for ev in watch._evidence.values():
                        ev.text, ev.last_chunk = '', None
                    for ch in watch._channels:
                        outbox.put((generation, watch._said(ch, 'coverage',
                            f'passive audio gap: {item[1] / FS:.2f} seconds skipped; readers reset')))
                else:
                    watch.heard(*item[1:])
                continue
            before = watch.now
            for record in watch.feed(item):
                outbox.put((generation, record))
            if watch.now != before:
                outbox.put((generation, ('table', watch.table(), watch.now)))
    except Exception as error:
        outbox.put((generation, ('error', None, repr(error))))


class AudioShift:
    """Real audio moved by `hz`, streamed: a Hilbert pair, then a complex mix. Group
    delay 127 samples (2.6 ms), the mix's phase kept with it, so the output is the
    moved input delayed; from 300 Hz up the image is 43 dB down and the gain within
    0.12 dB."""
    TAPS = 255

    def __init__(self, hz):
        self.hz = hz
        self.h = -remez(self.TAPS, [200, FS / 2 - 200], [1], type='hilbert', fs=FS)
        self.zi = np.zeros(self.TAPS - 1)
        self.delay = np.zeros(self.TAPS // 2)
        self.n = 0

    def __call__(self, x):
        x = np.asarray(x, float)
        quad, self.zi = lfilter(self.h, 1, x, zi=self.zi)
        joined = np.concatenate((self.delay, x))
        real, self.delay = joined[:len(x)], joined[len(x):]
        k = self.n + np.arange(len(x)) - self.TAPS // 2
        self.n = (self.n + len(x)) % FS
        return np.real((real + 1j * quad) * np.exp(2j * np.pi * self.hz * k / FS)).astype(np.float32)


def shift_burst(samples, hz):
    """A whole transmit burst moved by `hz`, through its analytic signal."""
    x = np.asarray(samples, float)
    pad = np.zeros(4096)
    analytic = hilbert(np.concatenate((pad, x, pad)))[len(pad):len(pad) + len(x)]
    return np.real(analytic * np.exp(2j * np.pi * hz * np.arange(len(x)) / FS))
