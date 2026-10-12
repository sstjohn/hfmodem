# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Streaming 170 Hz RTTY and SITOR-B/NAVTEX monitor; s16le in, JSON out.

The acquisition manager owns the cut's bounded history and each transmission's
frequency, power reference, activity and lifetime. Candidates need pair-local
alternation; strong framing or phasing can rescue interference from neighbours.
Carrier lifetimes require evidence on their own pair, including a coherent
RTTY mark carrier during idle. Idle requires a steady mark envelope; fresh
keying on a different pair can replace a fading lane even when the pairs share
a tone. Replay begins at that pair's carrier onset and requires activity since
the preceding search; completed bursts are decoded
once. Lanes retain framing, interpreted case and FEC through fades of up to two
seconds. Erased RTTY frames and unresolved SITOR pairs revoke shift certainty
until the next accepted shift. Power references and carrier fits belong to
each acquired emission; onset and endpoint estimates can overlap by the
envelope resolution at a short handoff. Longer silence ends the transmission.
A new transmission starts with fresh framing and power and unknown RTTY shift
until LTRS or FIGS. Periodic fits preserve decoder state.
Material carrier moves bypass the refit timer and replay their onset from
bounded framing checkpoints; already published records remain committed by
absolute frame position. Replay may revise start-bit edges within a frame,
but cannot republish that frame.

Lanes only decode assigned audio. The synchronous modem owns clock recovery
and termination; its time marks arbitrate RTTY and identify NAVTEX emissions
across alternative tuning cursors. Incomplete NAVTEX parsers can survive a fade
without keeping a carrier active. The manager reports that distinction to the
channelizer so track retirement cannot truncate unfinished traffic.
Before phasing, a 100 Bd clock or sustained constant-ratio word grid makes
overlapping async text tentative. It does not establish SITOR diversity parity;
the synchronous path keeps watching for phasing. RTTY unshift-on-space defaults
on and is selected once per runner with --rtty-usos / --no-rtty-usos.
Async grading also retains the last 32 frame attempts across line and shift
boundaries; an accidental shift cannot erase evidence of rejected stops.
Four consecutive accepted frames at the async character cadence can restore
framing confidence after corruption; synchronous evidence still takes priority.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hfmodem.lanner.sitor import Resume

from hfmodem.monitor import FS

BW_HZ = 700.0
NOMINAL_HZ = 1500.0
READ_BYTES = 9600
STEP_S = 0.25
RETUNE_S = 30.0
PAIR_TOL_HZ = 60.0
MATERIAL_HZ = 20.0
PAIR_DB = 15.0
BLOCK_S = 20.0
LOOKBACK_S = 12.0
EMISSION_S = 600.0
TEXT_WAIT_S = 600.0
TEXT_LIMIT = 10000
FADE_S = 2.0
PAUSE_S = 0.75
LINE_LIMIT = 80
ARBITRATE_S = 1.0
REJECT_RATE = 0.2
CONFIRMED, TENTATIVE = "confirmed", "tentative"


def normalize(t: float, protocol: str, text: str, offset_hz: float,
              *, station: str = "", detail: str = "45.45 Bd 170 Hz",
              grade: str = CONFIRMED) -> dict:
    """A decoded stretch in the common monitor record shape."""
    return {"t": round(t, 3), "modem": "lanner", "protocol": protocol,
            "kind": "MESSAGE" if protocol == "NAVTEX" else "TEXT",
            "text": text, "station": station, "detail": detail, "grade": grade,
            "offset_hz": round(offset_hz, 1),
            "centre_hz": round(NOMINAL_HZ + offset_hz, 1),
            **({"role": "sender"} if protocol == "NAVTEX" else {})}


def _emit(record: dict) -> None:
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


class _SitorAudio:
    """Delay-compensated continuous mixing and decimation from 48 kHz to 6 kHz."""

    def __init__(self, bw, centre=NOMINAL_HZ):
        import numpy as np
        from scipy.signal import firwin
        from hfmodem.monitor.channel import Channel

        self.mixer = Channel(centre, 1700.0, bw / 2)
        # Same anti-alias filter and zero-phase alignment as resample_poly(1, 8).
        self.h = firwin(161, 1 / 8, window=("kaiser", 5.0))
        self.zi = np.zeros(160)
        self.n = 0
        self.skip = 10

    def push(self, block, *, final=False):
        import numpy as np
        from scipy.signal import lfilter

        moved = self.mixer.push(block)
        if final:
            moved = np.concatenate((moved, self.mixer.flush(), np.zeros(80)))
        if not len(moved):
            return np.empty(0)
        filtered, self.zi = lfilter(self.h, [1.0], moved, zi=self.zi)
        low = filtered[(-self.n) % 8::8]
        self.n += len(moved)
        cut = min(self.skip, len(low))
        self.skip -= cut
        return low[cut:]


class _History:
    """Fixed storage avoids retaining a different malloc size on every push.

    Consumers see chronological views. Compaction occurs only when the write
    head reaches the allocation's end; dropping history itself is allocation-free.
    """

    def __init__(self, capacity):
        import numpy as np
        self.storage = np.empty(capacity)
        self.start = self.end = 0

    @property
    def samples(self):
        return self.storage[self.start:self.end]

    def append(self, block):
        size = self.end - self.start
        if self.end + len(block) > len(self.storage):
            self.storage[:size] = self.samples
            self.start, self.end = 0, size
        if self.end + len(block) > len(self.storage):
            raise RuntimeError('decoder failed to advance bounded audio history')
        self.storage[self.end:self.end + len(block)] = block
        self.end += len(block)
        return self.samples

    def drop(self, count):
        self.start += count
        return self.samples


@dataclass
class _Transmission:
    start: int
    end: int | None
    midpoint: float
    decoded: bool = False


@dataclass
class _SitorState:
    """One tuning hypothesis: cursor, FEC/shift, parser and decoded boundaries."""

    pending: str = ""
    certainty: list[bool] = field(default_factory=list)
    origin: tuple[float, float] | None = None
    marks: list[tuple[int, float]] = field(default_factory=list)
    lengths: dict[tuple[int, float], int] = field(default_factory=dict)
    end: int = 0
    lost_at: int | None = None
    resume: Resume | None = None
    resume_start: int | None = None
    transmissions: list[_Transmission] = field(default_factory=list)


class _SitorDecoder:
    """Synchronous decoder with owned state and borrowed, bounded audio."""

    def __init__(self, chan, state=None):
        self.chan = chan
        self.sitor = state if state is not None else _SitorState()
        self.sitor_capture = None
        self.sitor_centre = NOMINAL_HZ
        self.output = lambda record: _emit(record)
        self.n = self.sitor_base = 0

    def offset(self, midpoint):
        hz = midpoint - NOMINAL_HZ
        return self.chan.offset(hz) if self.chan else hz

    def sitor_text(self, text, *, times=(), certainty=(), complete=False, final=False):
        """Feed characters and their on-air time marks to the NAVTEX scanner."""
        from hfmodem.lanner import navtex

        if text:
            start = len(self.sitor.pending)
            self.sitor.marks.append((start, times[0]))
            # Shift and CR words consume slots without printing characters.
            # Start another 140 ms character run after those gaps.
            self.sitor.marks.extend((start + i, times[i]) for i in range(1, len(times))
                                    if abs(times[i] - times[i - 1] - 0.14) > 0.02)
        self.sitor.certainty.extend(certainty)
        joined = self.sitor.pending + text
        if not joined.strip() or self.sitor.origin is None:
            self.sitor.pending = joined
            return
        origin, midpoint = self.sitor.origin

        def time_at(index):
            at, mark = next((at, mark) for at, mark in reversed(self.sitor.marks)
                            if at <= index)
            return mark + (index - at) * 0.14

        def trim(count):
            if not count:
                return
            mark = time_at(count)
            self.sitor.marks = [(0, mark)] + [
                (at - count, t) for at, t in self.sitor.marks if at > count]

        messages, remainder = navtex.scan(joined)
        for at, until, message in messages:
            self.emit_sitor(normalize(time_at(at), "NAVTEX", message.text, self.offset(midpoint),
                            station=message.station,
                            detail=f"{message.subject} {message.serial:02d}",
                            grade=CONFIRMED if all(self.sitor.certainty[at:until]) else TENTATIVE))
        if messages:
            consumed = messages[-1][1]
            trim(consumed)
            joined = joined[consumed:]
            self.sitor.certainty = self.sitor.certainty[consumed:]
        self.sitor.pending = joined
        expired = ((complete and self.n / FS - origin >= TEXT_WAIT_S)
                   or len(joined) >= TEXT_LIMIT)
        if final or expired or (complete and not remainder):
            if joined.strip():
                self.emit_sitor(normalize(time_at(0), "SITOR-B", joined, self.offset(midpoint),
                                detail="100 Bd 170 Hz",
                                grade=CONFIRMED if all(self.sitor.certainty) else TENTATIVE))
            self.sitor.pending = ""
            self.sitor.certainty = []
            self.sitor.origin = None
            self.sitor.marks = []
        elif complete and remainder:
            trim(len(joined) - len(remainder))
            self.sitor.certainty = self.sitor.certainty[len(joined) - len(remainder):]
            self.sitor.pending = remainder

    def emit_sitor(self, record):
        if self.sitor_capture is None:
            self.output(record)
        else:
            self.sitor_capture.append(record)

    def _decode_sitor(self, final=False):
        if not getattr(self, 'sitor_enabled', True):
            return
        from hfmodem.lanner import sitor

        for tr in self.sitor.transmissions:
            start, end, midpoint = tr.start, tr.end, tr.midpoint
            if tr.decoded:
                continue
            # Keep lookback because the watcher may acquire partway through phasing.
            rate = FS // 8
            if self.sitor.resume_start != start:
                self.sitor.resume = sitor.Resume(max_fade_s=FADE_S)
                self.sitor.resume.sample = max(self.sitor_base, start // 8 - round(LOOKBACK_S * rate))
                self.sitor.resume_start = start
            lo = round(self.sitor.resume.sample)
            if self.sitor.resume.origin is not None:
                lo -= round(LOOKBACK_S * rate)
            lo = max(self.sitor_base, lo)
            hi = min(self.sitor_base + len(self.sitor_samples),
                     self.n // 8 if end is None else round(end / 8 + 0.2 * rate))
            audio = self.sitor_samples[lo - self.sitor_base:hi - self.sitor_base]
            complete = end is not None or final
            if not len(audio):
                continue
            sitor.transmissions(
                audio, fs=rate, resume=self.sitor.resume, origin=lo,
                final=complete, offset_hz=midpoint - self.sitor_centre)
            self.sitor.lost_at = (round(self.sitor.resume.emission_end * 8)
                                  if not self.sitor.resume.locked
                                  and self.sitor.resume.emission_end is not None else None)
            if end is None and (final or (self.sitor.resume.emission_finished
                                          and not self.sitor.resume.locked)):
                tr.end = end = (min(self.n, round(self.sitor.resume.emission_end * 8))
                               if self.sitor.resume.emission_end is not None else self.n)
                complete = True
            for lock, (sample, value) in enumerate(self.sitor.resume.texts):
                if sample / rate < self.sitor.end / FS - 0.2:
                    continue
                key = (start, sample)
                previous = self.sitor.lengths.get(key, 0)
                delta = value[previous:]
                self.sitor.lengths[key] = len(value)
                if delta:
                    if self.sitor.origin is None:
                        self.sitor.origin = (sample / rate, midpoint)
                    self.sitor_text(delta, times=[sample / rate for sample in
                        self.sitor.resume.char_samples[lock][previous:]],
                        certainty=self.sitor.resume.char_shift_known[lock][previous:])
                    if self.sitor.resume.lock_finished(lock):
                        self.sitor_text("", complete=True)
            if complete:
                self.sitor_text("", complete=True, final=final)
                self.sitor.lengths = {key: length for key, length in self.sitor.lengths.items()
                                      if key[0] != start}
            if end is not None or final:
                tr.decoded = True
                self.sitor.end = self.n if end is None else end
        if self.sitor.pending and (final or (self.sitor.origin is not None
                and not any(tr.end is None for tr in self.sitor.transmissions)
                and self.n / FS - self.sitor.origin[0] >= TEXT_WAIT_S)):
            self.sitor_text("", complete=True, final=final)


class _DecoderLane(_SitorDecoder):
    def __init__(self, chan, bw):
        super().__init__(chan)
        self.bw = bw
        self.output = lambda record: _emit(record)
        self.rtty_enabled = self.sitor_enabled = True
        # Resume retains three seconds of FEC plus up to 56 phasing units;
        # also leave room for the next quarter-second before decoding it.
        capacity = round((BLOCK_S + LOOKBACK_S + 4) * FS)
        self.history = _History(capacity)
        self.low_history = _History(capacity // 8 + 1)
        self.audio = self.history.samples
        self.base = self.n = self.cursor = 0
        self.midpoint = None
        self.reader = None
        self.reader_origin = 0
        self.next_block = round(BLOCK_S * FS)
        self.line = ""
        self.line_t = self.last_char = 0.0
        self.last_char_sample = -1
        self.good = self.bad = 0
        self.recent_framing = deque(maxlen=32)
        self.async_run = 0
        self.frame_sample = None
        self.shift_known = False
        self.line_unknown = False
        self.pending = []
        self.reader_power = 0.0
        self.block_energy = None
        self.rtty_checkpoints = deque(maxlen=16)
        self.published_through = -1
        self.rtty_live = True
        self.sitor_lanes = []
        self.sitor_capture = None
        self.sitor_samples = self.low_history.samples
        self.sitor_base = 0
        self.sitor_stream = None
        self.sitor_signatures = deque(maxlen=32)

    def suppressed(self, start, end):
        return any(start <= (tr.end if tr.end is not None else self.sitor.lost_at if self.sitor.lost_at is not None else self.n) / FS
                   and end >= tr.start / FS - 0.15
                   for tr in self.sitor.transmissions)

    def finish_line(self):
        if self.line.strip():
            recent_bad = sum(not accepted for accepted in self.recent_framing)
            grade = (TENTATIVE if self.line_unknown
                     or self.bad / max(1, self.good + self.bad) > REJECT_RATE
                     or (self.async_run < 4
                         and recent_bad / max(1, len(self.recent_framing)) > REJECT_RATE)
                     else CONFIRMED)
            self.pending.append((self.last_char, self.last_char_sample, normalize(
                self.line_t, "RTTY", self.line, self.offset(self.midpoint), grade=grade)))
        self.line = ""
        self.line_unknown = False
        self.good = self.bad = 0

    def consume(self):
        from hfmodem.lanner import rtty

        for event in self.reader.events:
            frame_sample = self.reader_origin + event.sample
            if frame_sample <= self.published_through:
                # A replayed edge can move within the committed frame. Its
                # start still precedes that frame's published stop sample.
                self.line = ""
                self.line_unknown = False
                self.good = self.bad = 0
                self.note_frame(frame_sample, event.accepted)
                continue
            end_sample = frame_sample + round(6.5 * FS / rtty.BAUD)
            t = end_sample / FS
            value, accepted = event.text, event.accepted
            if self.suppressed(frame_sample / FS, t):
                self.line = ""
                self.good = self.bad = 0
                continue
            if event.shift_known != self.shift_known:
                # Keep the known prefix confirmed when later text becomes tentative.
                self.finish_line()
                self.shift_known = event.shift_known
            self.note_frame(frame_sample, accepted)
            self.good += int(accepted)
            self.bad += int(not accepted)
            for ch in value:
                if not self.line:
                    self.line_t = t
                if len(self.line) >= LINE_LIMIT and ch != "\n":
                    self.finish_line()
                    self.line_t = t
                self.line_unknown |= not self.shift_known
                self.line += ch
                self.last_char = t
                self.last_char_sample = end_sample
                if ch == "\n":
                    self.finish_line()
        self.reader.events.clear()

    def note_frame(self, sample, accepted):
        """Keep stop evidence independently of shift certainty and text lines.

        Normal 1.5 or 2-unit stops put async starts 7.5 or 8 units apart.
        Allow discriminator edge jitter, but not the neighbouring 100 Bd
        grid's 160/170/180 ms intervals. A clean run can restore confidence
        promptly without letting a single accidental shift do so.
        """
        from hfmodem.lanner import rtty

        unit = FS / rtty.BAUD
        delta = (sample - self.frame_sample) / unit if self.frame_sample is not None else 0
        consecutive = min(abs(delta - 7.5), abs(delta - 8)) <= .12
        self.async_run = (self.async_run + 1 if consecutive else 1) if accepted else 0
        self.frame_sample = sample
        self.recent_framing.append(accepted)

    def release(self, final=False):
        kept = []
        for ready, end, rec in self.pending:
            if self.suppressed(rec["t"], end / FS):
                continue
            if end <= self.published_through:
                continue
            if final or self.n / FS - ready >= ARBITRATE_S:
                if any(rec['t'] <= stop / FS and end >= start
                       for start, stop in self.sitor_signatures):
                    rec['grade'] = TENTATIVE
                self.output(rec)
                self.published_through = end
            else:
                kept.append((ready, end, rec))
        self.pending = kept

    def watch(self):
        """Recognise synchronous evidence within RTTY's arbitration delay.
        Traffic signatures affect grading; only phasing opens a transmission.
        The decoder owns termination and word sync.
        """
        if not self.sitor_enabled:
            return
        import numpy as np
        from hfmodem.lanner import fsk, sitor

        if any(tr.end is None for tr in self.sitor.transmissions):
            return
        active = None
        midpoint = self.midpoint
        if midpoint is None:
            return
        rate = FS // 8
        low_lo = max(self.sitor_base, self.n // 8 - 3 * rate)
        lo = low_lo * 8
        audio = self.sitor_samples[low_lo - self.sitor_base:]
        if len(audio) < rate // 2:
            return
        soft = fsk.discriminate(audio, f_hi=sitor.F_B + midpoint - self.sitor_centre,
                                f_lo=sitor.F_Y + midpoint - self.sitor_centre,
                                baud=sitor.BAUD, fs=rate)
        centres = fsk.track(soft, baud=sitor.BAUD, fs=rate)
        if sitor.traffic_signature(soft, fs=rate, centres=centres):
            signature_start = lo
            if self.sitor_signatures and lo <= self.sitor_signatures[-1][1]:
                signature_start = self.sitor_signatures.pop()[0]
            self.sitor_signatures.append((signature_start, self.n))
        if not len(centres):
            return
        hard = soft[np.rint(centres).astype(int)] > 0
        pos = 0
        while active is None:
            found = sitor.find_phasing(hard, pos)
            if found is None:
                break
            start = lo + round(centres[found[2]] * 8)
            if not any(tr.start - FS // 5 <= start <= tr.end
                       for tr in self.sitor.transmissions if tr.end is not None):
                active = _Transmission(start, None, midpoint)
                if self.sitor.origin is None:
                    self.sitor.origin = (start / FS, midpoint)
                self.sitor.transmissions.append(active)
                self.sitor.lost_at = None
            pos = found[2] + 14

    def decode_sitor(self, final=False):
        """Keep alternate tuning cursors only after the modem reports errors.

        A wrong spectral skirt can decode good words but corrupt a shift word.
        Trials therefore start from the previous cursor, before that error,
        and retain their own FEC, shift and NAVTEX parser state thereafter.
        All lanes share the bounded audio; clean signals use only one cursor.
        """
        if not self.sitor_enabled:
            return
        from copy import deepcopy
        active = next((tr for tr in self.sitor.transmissions if not tr.decoded), None)
        checkpoint = (deepcopy(self.sitor)
                      if active is not None and not self.sitor_lanes else None)
        previous = sum(len(text) for _, text in self.sitor.resume.texts) if self.sitor.resume else 0
        records = []
        self.sitor_capture = records
        self._decode_sitor(final)
        self.sitor_capture = None
        if checkpoint is not None and self.sitor.resume is not None:
            text = ''.join(value for _, value in self.sitor.resume.texts)[previous:]
            degraded = ('_' in text or (len(self.sitor.resume.recent) >= 20
                        and sum(self.sitor.resume.recent) < .9 * len(self.sitor.resume.recent)))
            if degraded:
                for delta in (-10, 10):
                    lane = _SitorDecoder(self.chan, deepcopy(checkpoint))
                    lane.sitor_centre = self.sitor_centre
                    for tr in lane.sitor.transmissions:
                        if not tr.decoded:
                            tr.midpoint += delta
                    if lane.sitor.origin is not None:
                        lane.sitor.origin = (lane.sitor.origin[0], lane.sitor.origin[1] + delta)
                    self.sitor_lanes.append(lane)
        for lane in self.sitor_lanes:
            lane.n = self.n
            lane.sitor_base, lane.sitor_samples = self.sitor_base, self.sitor_samples
            lane.sitor_capture = []
            # Silence, a hard bound, or a decoded end also bounds the trials.
            for tr in lane.sitor.transmissions:
                parent = next((p for p in self.sitor.transmissions if p.start == tr.start), None)
                if parent is not None and parent.end is not None and tr.end is None:
                    tr.end = parent.end
            lane._decode_sitor(final)
            records.extend(r for r in lane.sitor_capture if r['protocol'] == 'NAVTEX')
            lane.sitor_capture = None
        navtex_records = [r for r in records if r['protocol'] == 'NAVTEX']
        chosen = []
        def agreement(record):
            # Among equally complete/error-marked hypotheses, prefer the text
            # supported by the other tunings over a lone valid-word mistake.
            return sum(sum(a == b for a, b in zip(record['text'], other['text']))
                       for other in navtex_records if self.sitor_twin(record, other))

        for record in sorted(navtex_records, key=lambda r: (
                r['grade'] == TENTATIVE, r['text'].count('_') / max(1, len(r['text'])),
                -len(r['text']), -agreement(r))):
            if not any(self.sitor_twin(record, old) for old in chosen):
                chosen.append(record)
        for record in records:
            if record['protocol'] != 'NAVTEX' and not any(
                    abs(record['t'] - msg['t']) < .8 for msg in chosen):
                self.output(record)
        for record in sorted(chosen, key=lambda r: r['t']):
            self.output(record)
        if final or not any(not tr.decoded for tr in self.sitor.transmissions):
            self.sitor_lanes = []

    @staticmethod
    def sitor_twin(a, b):
        return abs(a['t'] - b['t']) < .8

    _RTTY_FIELDS = ('line', 'line_t', 'last_char', 'last_char_sample', 'good', 'bad',
                    'shift_known', 'line_unknown', 'pending', 'recent_framing',
                    'async_run', 'frame_sample')

    def checkpoint_rtty(self):
        from copy import deepcopy
        self.rtty_checkpoints.append((self.cursor, deepcopy(self.reader),
            deepcopy([getattr(self, name) for name in self._RTTY_FIELDS])))

    def retune(self, midpoint, onset):
        """Replay a moved carrier's onset from bounded async checkpoints.

        Reconstruct the prefix at the old tuning before changing tones. Keep
        published records committed while replacing uncommitted framing/text.
        The synchronous cursor retains its independent FEC and parser state.
        """
        if not self.rtty_enabled:
            self.midpoint = midpoint
            return
        import numpy as np
        from hfmodem.lanner import rtty

        onset = min(self.n, max(self.base, onset))
        checkpoint = next((c for c in reversed(self.rtty_checkpoints)
                           if c[0] <= onset), None)
        if checkpoint is not None:
            self.cursor, self.reader, values = checkpoint
            for name, value in zip(self._RTTY_FIELDS, values):
                setattr(self, name, value)
            if self.last_char_sample <= self.published_through:
                self.line = ""
                self.line_unknown = False
                self.good = self.bad = 0
            if self.cursor < onset:
                samples = self.audio[self.cursor - self.base:onset - self.base]
                valid = None
                if len(samples) >= round(FS / rtty.BAUD) and self.reader_power:
                    power = _AcquisitionManager.tone_energy(samples, self.midpoint)
                    valid = power >= self.reader_power * .01
                    samples = np.where(valid, samples, 0.)
                self.reader.feed(samples, valid=valid)
                self.cursor = onset
                self.consume()
            self.rtty_checkpoints.clear()
        self.reader.retune(midpoint + 85, midpoint - 85)
        self.midpoint = midpoint

    def advance_rtty_clock(self, upto):
        """Skip proven synchronous traffic without allocating a long DSP block."""
        if not self.rtty_enabled:
            self.cursor = upto
            return
        import numpy as np
        while self.cursor < upto:
            size = min(round(STEP_S * FS), upto - self.cursor)
            if self.reader is not None:
                self.reader.feed(np.zeros(size))
                self.reader.events.clear()
            self.cursor += size

    def wait(self, count, *, final=False):
        """Park a completed lane's parser, advancing only expiry and arbitration."""
        self.n += count
        self.block_energy = None
        self.rtty_checkpoints.clear()
        if len(self.history.storage):
            self.history = _History(0)
            self.low_history = _History(0)
            self.audio = self.history.samples
            self.sitor_samples = self.low_history.samples
            self.sitor_stream = None
            self.sitor_lanes = []
            self.sitor.resume = None
            self.sitor.transmissions = []
        self.base = self.cursor = self.n
        self.sitor_base = self.n // 8
        self._decode_sitor(final)
        self.release(final)

    def push(self, block, final=False):
        import numpy as np
        from hfmodem.lanner import rtty

        if self.sitor_stream is None:
            self.sitor_stream = _SitorAudio(self.bw, self.sitor_centre)
            self.sitor_base = self.n // 8
        low = self.sitor_stream.push(block, final=final)
        self.sitor_samples = self.low_history.append(low)
        self.audio = self.history.append(block)
        self.n += len(block)
        if self.reader is not None or (self.sitor_enabled and not self.rtty_enabled):
            self.watch()
        else:
            self.cursor = self.n
        if final or self.n >= self.next_block or any(
                tr.end is not None and not tr.decoded for tr in self.sitor.transmissions):
            self.decode_sitor(final=final)
            self.next_block = self.n + round(BLOCK_S * FS)
        if any(tr.end is None for tr in self.sitor.transmissions):
            # Audio already consumed by the synchronous decoder cannot contain
            # following RTTY. Advance its async clock without framing traffic.
            if self.sitor.resume is not None:
                drop_to = max(self.cursor, min(self.n, self.sitor.lost_at if self.sitor.lost_at is not None
                                               else round(self.sitor.resume.sample * 8)))
                if drop_to > self.cursor:
                    self.advance_rtty_clock(drop_to)
        if not self.rtty_enabled:
            self.cursor = self.n
        if self.rtty_enabled and self.reader is not None and (self.rtty_live or final or self.sitor.lost_at is not None) and (final or not any(
                tr.end is None for tr in self.sitor.transmissions) or self.sitor.lost_at is not None):
            while self.cursor < self.n:
                upto = min(self.n, self.cursor + round(STEP_S * FS))
                self.checkpoint_rtty()
                samples = self.audio[self.cursor - self.base:upto - self.base]
                valid = None
                if len(samples) >= round(FS / rtty.BAUD) and self.reader_power:
                    cached = self.block_energy
                    power = (cached[2] if cached is not None
                             and cached[0] == self.cursor and cached[1] == self.midpoint
                             and len(cached[2]) == len(samples) else
                             _AcquisitionManager.tone_energy(samples, self.midpoint))
                    valid = power >= self.reader_power * 0.01
                    samples = np.where(valid, samples, 0.0)
                self.reader.feed(samples, valid=valid)
                self.cursor = upto
                self.consume()
        if self.n / FS - self.last_char >= PAUSE_S:
            self.finish_line()
        if final and self.rtty_enabled and self.reader is not None:
            self.reader.flush()
            self.consume()
            self.finish_line()
        self.release(final)
        self.block_energy = None
        keep = max(self.base, self.n - round((BLOCK_S + LOOKBACK_S) * FS))
        keep = min(keep, self.cursor)
        if keep > self.base:
            self.audio = self.history.drop(keep - self.base)
            self.base = keep
        low_keep = max(self.sitor_base, self.n // 8 - round((BLOCK_S + LOOKBACK_S) * FS / 8))
        if any(not tr.decoded for tr in self.sitor.transmissions) and self.sitor.resume is not None:
            low_keep = min(low_keep, max(self.sitor_base,
                round(self.sitor.resume.sample) - round(LOOKBACK_S * FS / 8)))
        if low_keep > self.sitor_base:
            self.sitor_samples = self.low_history.drop(low_keep - self.sitor_base)
            self.sitor_base = low_keep
        self.sitor.transmissions = [tr for tr in self.sitor.transmissions
                              if tr.end is None or not tr.decoded or tr.end >= self.base]


@dataclass
class _Carrier:
    power: float
    last_sample: int
    refit_at: int
    fading: bool = False


class _AcquisitionManager:
    """Single owner of acquisition, transmission boundaries and decoder slots."""

    def __init__(self, chan, bw, *, rtty_usos=True, protocol=None, search_span=None):
        import numpy as np
        self.chan, self.bw = chan, bw
        self.rtty_usos = rtty_usos
        self.protocol = protocol
        self.search_span = search_span
        self.n = 0
        self.audio = np.empty(0)
        self.decoder_lanes = []
        self.active: dict[_DecoderLane, _Carrier] = {}
        self.next_search = round(.75 * FS)
        self.searched_through = 0
        self.claimed = []
        self.said = deque(maxlen=1000)

    def publish(self, record):
        # SITOR-B and NAVTEX share the same synchronous receiver. A SITOR
        # monitor also shows framed bulletins; NAVTEX asks for bulletins only.
        allowed = {'RTTY': ('RTTY',), 'SITOR-B': ('SITOR-B', 'NAVTEX'),
                   'NAVTEX': ('NAVTEX',)}
        if self.protocol and record['protocol'] not in allowed[self.protocol]:
            return
        # The on-air header position and pair identify an emission across
        # tuning trials. Equal bulletin bodies at different times remain distinct.
        if record['protocol'] == 'NAVTEX':
            if any(abs(record['centre_hz'] - old['centre_hz']) < PAIR_TOL_HZ
                   and _DecoderLane.sitor_twin(record, old) for old in self.said):
                return
            self.said.append({key: record[key] for key in ('t', 'centre_hz')})
        _emit(record)

    @staticmethod
    def tone_energy(samples, midpoint):
        """Four-unit Hann power separates the pair from neighbouring tones."""
        from hfmodem.lanner import fsk, rtty
        return fsk.tone_energy(samples, f_hi=midpoint + 85,
                               f_lo=midpoint - 85, baud=rtty.BAUD, fs=FS, units=4)

    def evidence(self, recent, midpoint, *, validate=True):
        """Find a carrier run beyond the preceding discovery watermark.

        Fresh candidates require validation. A just-validated pair can use its
        coherent envelope alone to recover onset before a short search window.
        """
        import numpy as np
        from hfmodem.lanner import fsk
        energy = self.tone_energy(recent, midpoint)
        level = float(np.max(energy)) if len(energy) else 0.0
        if level <= 1e-12:
            return None
        off = energy < level * .1
        width, hop = FS // 20, FS // 200
        quiet = np.convolve(off[::hop].astype(int),
                            np.ones(width // hop, dtype=int), mode='valid') == width // hop
        # Runs are split only by 50 ms of silence, not symbol transitions.
        edges = np.flatnonzero(np.diff(np.r_[True, quiet, True]))
        for begin, end in zip(edges[::2][::-1], edges[1::2][::-1]):
            onset = max(0, int(begin * hop))
            stop = min(len(recent), int((end + width // hop - 1) * hop))
            absolute_end = self.n - len(recent) + stop
            if absolute_end <= self.searched_through:
                continue
            current = recent[onset:stop]
            if len(current) >= FS // 3 and (not validate or any(fsk.keyed_pair(
                    current[-FS:], FS, f_hi=midpoint + 85, f_lo=midpoint - 85,
                    baud=rate) for rate in (45.45, 1000 / 15))):
                return onset, stop, level
        return None

    def end_carrier(self, lane):
        """Release async framing and power, preserving only unfinished text."""
        state = self.active.get(lane)
        for tr in lane.sitor.transmissions:
            if tr.end is None:
                tr.end = (lane.sitor.lost_at if lane.sitor.lost_at is not None
                          else state.last_sample if state is not None else lane.n)
        if any(not tr.decoded for tr in lane.sitor.transmissions):
            lane.decode_sitor()
        if lane.rtty_enabled and lane.reader is not None:
            lane.reader.flush()
            lane.consume()
            lane.finish_line()
        lane.reader = None
        lane.rtty_checkpoints.clear()
        lane.reader_power = 0.0
        lane.cursor = lane.n
        state = self.active.pop(lane, None)
        if state is not None:
            self.claimed.append((lane.midpoint, state.last_sample))

    def assign(self, midpoint, recent, onset, stop, level):
        from hfmodem.lanner import rtty
        start = self.n - len(recent) + onset
        end = self.n - len(recent) + stop
        if any(abs(midpoint - hz) <= PAIR_TOL_HZ and start < until
               for hz, until in self.claimed):
            return None
        # Association applies to live carriers, never to old tuning history.
        lane = min(self.active, key=lambda p: abs(p.midpoint - midpoint), default=None)
        if lane is not None and abs(lane.midpoint - midpoint) <= PAIR_TOL_HZ:
            state = self.active[lane]
            if state.fading and end <= state.last_sample + round(.05 * FS):
                return None
            # The 50 ms envelope boundary can straddle a short handoff.
            # Fresh weaker traffic owns a new level even when its estimated
            # onset slightly precedes the previous carrier's endpoint.
            if (state.fading and start >= state.last_sample - round(.05 * FS)
                    and (level < state.power * .1 or (
                        abs(lane.midpoint - midpoint) > 20
                        and not any(tr.end is None for tr in lane.sitor.transmissions)))):
                self.end_carrier(lane)
            else:
                lane.rtty_live = True
                state.fading = False
                state.last_sample = end
                if self.n < state.refit_at and abs(lane.midpoint - midpoint) <= MATERIAL_HZ:
                    return None
                if abs(lane.midpoint - midpoint) > MATERIAL_HZ:
                    import numpy as np
                    tail = recent[-round(.75 * FS):]
                    new_power = float(np.mean(self.tone_energy(tail, midpoint)))
                    old_power = float(np.mean(self.tone_energy(tail, lane.midpoint)))
                    if new_power < 2 * old_power:
                        if self.n < state.refit_at:
                            return None
                        if lane.reader is not None:
                            lane.reader.retune(midpoint + 85, midpoint - 85)
                        lane.midpoint = midpoint
                    else:
                        # Coherent power dates the moved carrier's arrival.
                        energy = self.tone_energy(recent, midpoint)
                        arrival = np.flatnonzero(energy >= level * .25)
                        moved_start = self.n - len(recent) + int(arrival[0]) if len(arrival) else start
                        lane.retune(midpoint, max(lane.base, moved_start))
                else:
                    if lane.reader is not None:
                        lane.reader.retune(midpoint + 85, midpoint - 85)
                    lane.midpoint = midpoint
                for tr in lane.sitor.transmissions:
                    if not tr.decoded:
                        tr.midpoint = midpoint
                if lane.sitor.origin is not None:
                    lane.sitor.origin = (lane.sitor.origin[0], midpoint)
                state.refit_at = self.n + round(RETUNE_S * FS)
                return None
        if len(self.active) >= 4:
            return None
        # A parser waiting through a fade may be reused, but its reader and
        # level belong to the new carrier and are always constructed afresh.
        previous = next((p for p in self.decoder_lanes if p not in self.active
                         and abs(p.midpoint - midpoint) <= PAIR_TOL_HZ), None)
        lane = _DecoderLane(self.chan, self.bw)
        lane.rtty_enabled = self.protocol in (None, 'RTTY')
        lane.sitor_enabled = self.protocol != 'RTTY'
        if self.search_span is not None:
            # A 4 kHz input cannot fit unshifted into the real 6 kHz SITOR
            # decoder. Move each acquired pair to its nominal 1700 Hz before
            # decimation; keep original midpoints for reporting and refits.
            lane.sitor_centre = midpoint
        lane.output = self.publish
        lane.n = lane.base = lane.cursor = start
        self.decoder_lanes.append(lane)
        if previous is not None and previous.sitor.pending:
            for field in ('pending', 'certainty', 'origin', 'marks'):
                setattr(lane.sitor, field, getattr(previous.sitor, field))
            previous.sitor.pending = ''
            previous.sitor.certainty = []
            previous.sitor.origin = None
            previous.sitor.marks = []
        lane.midpoint = midpoint
        if lane.rtty_enabled:
            lane.reader = rtty.Stream(mark_hz=midpoint + 85, space_hz=midpoint - 85,
                                     fs=FS, unshift_on_space=self.rtty_usos)
        lane.reader_origin = start
        lane.reader_power = level
        self.active[lane] = _Carrier(level, end, self.n + round(2.25 * FS))
        return lane, recent[start - (self.n - len(recent)):]

    def push(self, block, final=False):
        import numpy as np
        from hfmodem.lanner import fsk
        self.n += len(block)
        self.audio = np.concatenate((self.audio, block))[-3 * FS:]
        # Endpoints are measured before discovery so quiet lanes cannot consume
        # the cap or lend their power reference to a new transmission.
        for lane, state in list(self.active.items()):
            energy = self.tone_energy(block, lane.midpoint)
            lane.block_energy = (self.n - len(block), lane.midpoint, energy)
            synchronous = any(tr.end is None for tr in lane.sitor.transmissions)
            threshold = state.power * (.01 if synchronous else .1)
            present = len(energy) and float(np.mean(energy)) >= threshold
            above = np.flatnonzero(energy >= threshold)
            arrival = self.n - len(block) + int(above[0]) if len(above) else self.n
            # Narrow power masks fades; keying, framing or coherent mark idle
            # owns the lifetime. Neighbour leakage cannot refresh the endpoint.
            rate = 100 if synchronous else 45.45
            own_activity = present and fsk.keyed_pair(
                self.audio[-round(.75 * FS):], FS, f_hi=lane.midpoint + 85,
                f_lo=lane.midpoint - 85, baud=rate)
            if present and rate == 100 and not own_activity:
                own_activity = fsk.framed_pair(
                    self.audio[-round(.75 * FS):], FS, f_hi=lane.midpoint + 85,
                    f_lo=lane.midpoint - 85, sitor_locked=True)
            if present and not synchronous and not own_activity:
                own_activity = fsk.mark_idle(
                    block, FS, mark_hz=lane.midpoint + 85,
                    space_hz=lane.midpoint - 85, power=state.power)
            if present and lane.last_char >= self.n / FS - .5:
                own_activity = True
            if own_activity:
                state.fading = False
                lane.rtty_live = True
                state.last_sample = self.n - len(block) + int(above[-1]) + 1
            else:
                state.fading = True
                lane.rtty_live = bool(present)
                self.next_search = min(self.next_search, self.n)
                if any(tr.end is None for tr in lane.sitor.transmissions):
                    lane.decode_sitor()
                # A timely return needs enough samples to commit a complete
                # post-fade frame before retirement can decide its continuity.
                if (not own_activity and self.n - state.last_sample > FADE_S * FS
                        and not (present and arrival - state.last_sample <= FADE_S * FS
                                 and self.n - state.last_sample <= (FADE_S + 2 * STEP_S) * FS)):
                    for tr in lane.sitor.transmissions:
                        if tr.end is None:
                            tr.end = state.last_sample
                    self.end_carrier(lane)
            for tr in lane.sitor.transmissions:
                if tr.end is None and self.n - tr.start >= EMISSION_S * FS:
                    tr.end = self.n
        new = {}
        if self.n >= self.next_search or final:
            self.next_search = self.n + round((1.0 if self.active else STEP_S) * FS)
            assigned = []
            for seconds in (3, 1.5, .75):
                recent = self.audio[-round(seconds * FS):]
                candidates = fsk.tone_pairs(
                    recent, FS, shift=170, near=NOMINAL_HZ,
                    span=(self.search_span if self.search_span is not None else
                          max(0, self.bw / 2 - 85) if self.chan else PAIR_TOL_HZ),
                    tol=1, max_s=3, keyed=True, limit=4) if len(recent) else []
                for midpoint, ratio in candidates:
                    # A pair assembled from two transmitters can look keyed
                    # briefly. It cannot own a tone already assigned to a lane.
                    if (ratio < PAIR_DB
                            or any(abs(midpoint - m) < PAIR_TOL_HZ for m in assigned)
                            or any(abs(abs(midpoint - p.midpoint) - 170) < 12
                                   and not state.fading for p, state in self.active.items())):
                        continue
                    proof = self.evidence(recent, midpoint)
                    if proof is None:
                        continue
                    # Fresh pair-local keying supersedes a fading owner of a
                    # shared tone. Its power and framing belong to a new lane.
                    for old, state in list(self.active.items()):
                        if state.fading and abs(abs(midpoint - old.midpoint) - 170) < 12:
                            self.end_carrier(old)
                    onset, stop, level = proof
                    if onset:
                        # The search window's Hann taper buries a run that has
                        # only just begun, so place the pair on the run itself.
                        # Evidence has already proven its keying there; a 15 ms
                        # envelope retest of a short, mostly idle burst passes
                        # or fails by hundredths of correlation.
                        fitted, strength = fsk.tone_pair(
                            recent[onset:stop], FS, shift=170, near=midpoint,
                            span=25, tol=1, max_s=3)
                        if strength >= PAIR_DB:
                            midpoint = fitted
                    associated = min(self.active, key=lambda p: abs(p.midpoint - midpoint), default=None)
                    if (associated is None or abs(associated.midpoint - midpoint) > PAIR_TOL_HZ
                            or (abs(associated.midpoint - midpoint) <= MATERIAL_HZ
                                and self.n >= self.active[associated].refit_at
                                and not any(tr.end is None for tr in associated.sitor.transmissions))):
                        # Fit the validated carrier run so a louder predecessor
                        # in the lookback cannot displace a weak successor.
                        midpoint = fsk.fit_pair(recent[onset:stop], FS, midpoint=midpoint)
                    result = self.assign(midpoint, recent, onset, stop, level)
                    assigned.append(midpoint)
                    if result is not None:
                        lane, _ = result
                        # This pair has fresh keying/framing evidence. Extend
                        # replay to its coherent-power onset in bounded history.
                        proof = self.evidence(self.audio, midpoint, validate=False)
                        if proof is not None and lane.n == lane.base:
                            onset, _, _ = proof
                            lane.n = lane.base = lane.cursor = self.n - len(self.audio) + onset
                            lane.reader_origin = lane.n
                            result = lane, self.audio[onset:]
                        new[result[0]] = result[1]
            self.searched_through = self.n
            self.claimed = [(hz, end) for hz, end in self.claimed if end > self.n - len(self.audio)]
        for lane in self.decoder_lanes:
            if (lane not in self.active and lane.reader is None
                    and all(tr.decoded for tr in lane.sitor.transmissions)):
                lane.wait(len(block), final=final)
            else:
                lane.push(new.get(lane, block), final=final)
        # Dormant parsers do not occupy acquisition slots; discard them after
        # their bounded wait, once arbitration has released their last line.
        self.decoder_lanes = [p for p in self.decoder_lanes if p in self.active or p.pending
                      or p.sitor.pending or any(not tr.decoded for tr in p.sitor.transmissions)]
        if self.chan:
            heard = self.n / FS - BLOCK_S - LOOKBACK_S
            for p in self.decoder_lanes:
                for tr in p.sitor.transmissions:
                    if not tr.decoded:
                        heard = min(heard, tr.start / FS)
                if p.sitor.pending and p.sitor.origin is not None:
                    heard = min(heard, p.sitor.origin[0])
            _emit({"heard": round(heard, 3)})
            _emit({"t": round(self.n / FS, 3), "modem": "lanner",
                   "protocol": "UNKNOWN", "kind": "emission-status",
                   "detail": "open" if any(
                       any(not tr.decoded for tr in p.sitor.transmissions) or p.sitor.pending.strip()
                       or (not final and p in self.active)
                       for p in self.decoder_lanes) else "closed"})


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    from hfmodem.monitor.channel import tuned

    ap = argparse.ArgumentParser()
    ap.add_argument("--centre", type=float)
    ap.add_argument("--bw", type=float, default=BW_HZ)
    ap.add_argument("--rtty-usos", action=argparse.BooleanOptionalAction, default=True,
                    help="unshift RTTY to letters on space (default: on)")
    ap.add_argument('--protocol', choices=('RTTY', 'SITOR-B', 'NAVTEX'),
                    help='decode only the selected async or synchronous protocol')
    ap.add_argument('--search-span', type=float,
                    help='search this far either side of 1500 Hz without a track')
    args = ap.parse_args(argv)
    chan = tuned(args.centre, args.bw)
    runner = _AcquisitionManager(chan, args.bw, rtty_usos=args.rtty_usos,
                                 protocol=args.protocol, search_span=args.search_span)
    stdin = sys.stdin.buffer
    carry = b""
    pending = np.empty(0)
    step = round(STEP_S * FS)
    while True:
        raw = stdin.read(READ_BYTES)
        if not raw:
            break
        raw = carry + raw
        carry = raw[-1:] if len(raw) % 2 else b""
        raw = raw[:len(raw) - len(carry)]
        x = np.frombuffer(raw, "<i2").astype(float) / 32768.0
        pending = np.concatenate((pending, chan.push(x) if chan else x))
        while len(pending) >= step:
            runner.push(pending[:step])
            pending = pending[step:]
    if chan:
        pending = np.concatenate((pending, chan.flush()))
    while len(pending) > step:
        runner.push(pending[:step])
        pending = pending[step:]
    runner.push(pending, final=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
