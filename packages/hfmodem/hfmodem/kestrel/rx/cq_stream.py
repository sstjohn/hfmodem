# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded continuous CQ acquisition; turbo runs only after header/reference lock."""
from dataclasses import dataclass, field, replace
import logging

import numpy as np
from scipy.signal import hilbert, resample_poly

from . import beacon as B
from . import broadcast as UI
from . import varahf500 as _rx
from .cq import HEADER, REFERENCES, REFERENCE_BINS, decode_power, parse_frame
from .ui_spans import MAX_HORIZON, SpanPublished, SpanResolved, UISpan, UISpanLedger

FS = 12000
SYMBOL = 512
HOP = 128
SEARCH_HISTORY = 4 * (32 - 1)
SEARCH_WAIT = 2048 + SYMBOL
RESAMPLE_DELAY = 24
# Some stations open the body well after the header ends, not 32 ms after:
# AB4EJ ~130 ms, KB4RA and KO4UXZ up to ~0.5 s on air. A candidate is first
# tried once EARLY samples past the nominal body are in, and only a body that
# never reached turbo waits for a full second.
EARLY = 3072
LATE = 12000
# Weak on-air CQs lock at header 20/32 and 14/24 reference hits; CRC and the
# canonical re-encode in parse_frame decide acceptance.
HEADER_MIN = 20
REFERENCE_MIN = 14
# The CQ frame and the KISS broadcast frame (VarAC's advanced beacon) share the
# header grid and differ in their 32 tones; each is scored against its own.
HEADERS = {'cq': HEADER, 'beacon': B.HEADER, 'ui1': UI.HEADERS[1], 'ui4': UI.HEADERS[4]}
BAND = {kind: (int(h.min()), int(h.max()) + 1) for kind, h in HEADERS.items()}
# Header shifts searched, in bins of FS / SYMBOL (23.4 Hz). A VARA HF station decodes a CQ or
# a beacon keyed 140 Hz off the centre and neither at 150 (2026-10-23).
# Six bins take both to 155 Hz; five stop at 135.
SHIFTS = range(-6, 7)
# Samples of 12 kHz audio each body needs past its header before a decode is tried.
BODY = {'cq': 384 + 94 * SYMBOL,
        'beacon': (B.RECORD.lead + B.RECORD.ncols) * B.RECORD.dw // 4,
        'ui1': 2 * 126 * 2048 // 4 + 384,
        'ui2': 2 * (B.RECORD.lead + B.RECORD.ncols) * B.RECORD.dw // 4 + 384,
        'ui4': 2 * 403 * 512 // 4 + 384}


def quality(x, center) -> float | None:
    if len(x) < 1024 or not np.all(np.isfinite(x)):
        return None
    windows = np.lib.stride_tricks.sliding_window_view(x, 1024)[::512]
    w = np.hanning(1024)
    p = np.mean(abs(np.fft.fft(windows * w, axis=-1)) ** 2, axis=0)
    f = np.fft.fftfreq(1024, 1 / FS) - center
    noise = p[((f >= -1200) & (f < -500)) | ((f > 500) & (f <= 1400))].mean()
    band = (f >= -500) & (f <= 500)
    signal = float(p[band].sum() - noise * band.sum()) * FS / 1024
    # A decoded frame need not have a measurable excess over the flank noise
    # (for example, with adjacent-channel interference). A numerical floor
    # here used to turn a non-positive estimate into a fictitious -345 dB SN.
    if signal <= 0 or noise <= 0 or not np.isfinite(signal + noise):
        return None
    # VARA's CQ SN refers the noise to the whole 12 kHz baseband: its
    # noise-limited reports sit 10*log10(12000/2900) below the 2900 Hz SNR.
    return float(10 * np.log10(signal / max(noise * FS, 1e-30)))


def beacon_quality(samples, body_start) -> float | None:
    """SN from the beacon's known reference tones, on aligned 48 kHz audio.

    Measure noise on unoccupied bins in the 500 Hz comb, not the adjacent
    channels. Hann windows suppress leakage from those channels and from the
    symbol boundaries. Integrate the reference tone and its two neighbours;
    leave another bin on each side out of the noise estimate to tolerate the
    decoder's residual frequency error. Keep the existing 12 kHz noise
    reference (this is not SNR in a 500 Hz bandwidth).
    """
    r = B.RECORD
    refs, tones = _rx._l3_tables(B.LEVEL)[:2]
    indices = body_start + refs[:, None] * r.dw + np.arange(r.dw)
    if indices.min() < 0 or indices.max() >= len(samples):
        return None
    blocks = samples[indices]
    if not np.all(np.isfinite(blocks)):
        return None
    return _reference_quality(blocks, tones,
                              np.arange(r.first_bin, r.first_bin + r.span), _rx.FS)


def _reference_quality(blocks, tones, bins, sample_rate) -> float | None:
    """Known-tone power versus unused in-band bins, referred to 12 kHz noise.

    Bin coordinates may straddle DC for centered IQ or be negative for the
    mirrored real-audio spectrum. Wrap only for FFT indexing, not distances.
    """
    if not np.all(np.isfinite(blocks)):
        return None
    width = blocks.shape[1]
    windowed = blocks * np.hanning(width)
    power = abs(np.fft.fft(windowed, axis=1)) ** 2
    neighbours = tones[:, None] + [-1, 0, 1]
    rows = np.arange(len(tones))[:, None]
    peak = power[rows, neighbours % width].mean(axis=0)
    if not np.all(peak > 0):
        return None
    # Refine the decoder's coarse frequency correction with the shared
    # reference peak. Otherwise residual CFO leaks strong signals into the
    # nominally empty bins and imposes an artificial high-SN ceiling.
    left, middle, right = np.log(peak)
    curvature = left - 2 * middle + right
    delta = (np.clip(.5 * (left - right) / curvature, -.5, .5)
             if curvature < 0 and middle > max(left, right) else 0.0)
    rotation = np.exp(-2j * np.pi * delta * np.arange(width) / width)
    power = abs(np.fft.fft(windowed * rotation, axis=1)) ** 2
    unused = abs(bins[None, :] - tones[:, None]) > 2
    # Equal weight per reference, including references at the comb edges.
    noise = (np.sum(power[:, bins % width] * unused, axis=1) / unused.sum(axis=1)).mean()
    lit = power[rows, neighbours % width]
    signal = lit.sum(axis=1).mean() - 3 * noise
    if signal <= 0 or noise <= 0 or not np.isfinite(signal + noise):
        return None
    return float(10 * np.log10(signal / noise * (sample_rate / width) / FS))


@dataclass
class _UIGroup:
    candidate: tuple
    level: int
    deadline: int
    frames: list = field(default_factory=list)
    failed: bool = False
    sn: float | None = None
    hz: float | None = None
    offset: int = 0  # Measured slot correction at 12 kHz; deadlines stay header-relative.

    @property
    def sample(self):
        return self.candidate[1]

    @property
    def slot(self):
        nominal = self.sample + (67072 + len(self.frames) * UI.PERIOD[self.level]) // 4
        return nominal + self.offset

    @property
    def extended(self):
        return self.deadline > UI.deadline(4 * self.sample, self.level, 2) // 4


class CQDetector:
    def __init__(self, sample_rate=48000, *, ui_header=None, ui_deadline=None,
                 sample_start=0, tune_epoch=0):
        if not np.isfinite(sample_rate) or sample_rate < 8000:
            raise ValueError('CQ input rate must be at least 8000 Hz')
        self.ui_header = ui_header
        self.ui_deadline = ui_deadline
        self.sample_rate = float(sample_rate)
        self._span_ledger = UISpanLedger(tune_epoch,
            route_retention=round(MAX_HORIZON * self.sample_rate / 48000))
        self._span_events = {}
        self._next_span_id = 0
        self.reset(sample_start=sample_start)

    @property
    def tune_epoch(self):
        return self._span_ledger.tune_epoch

    @property
    def input_clock(self):
        return self._anchor + self._source_count

    def _input_sample(self, sample):
        delay = RESAMPLE_DELAY if self.sample_rate > 24000 else 0
        return self._anchor + round(sample * self.sample_rate / FS) - delay

    @property
    def header_frontier(self):
        """Earliest header grid point a subsequent search may still accept."""
        history = 0 if self._spectra is None else len(self._spectra)
        return self._input_sample(max(0, self._scan - history * HOP))

    @property
    def classification_watermark(self):
        first = round(67072 * self.sample_rate / 48000)
        tolerance = round(16384 * self.sample_rate / 48000)
        return self.header_frontier + first - tolerance

    def take_span_events(self):
        """Drain coalesced coverage changes separately from delivered messages.

        Undrained changes live for the maximum horizon plus bounded route
        retention. Consumers must advance their ledger on the same input clock.
        """
        events = [event for changes in self._span_events.values() for event in changes]
        self._span_events.clear()
        return events

    def reset(self, *, sample_start=None, tune_epoch=None):
        current = getattr(self, 'input_clock', None)
        anchor = (0 if current is None else current) if sample_start is None else sample_start
        if current is not None and anchor < current:
            raise ValueError('CQ reset moves input clock backwards')
        if tune_epoch is not None and tune_epoch != self.tune_epoch:
            self._span_ledger.retune(tune_epoch)
            self._span_events.clear()
        else:
            for span in self._span_ledger.spans:
                if span.resolution is None:
                    self._resolve_span(span, 'truncated')
        self._anchor = anchor
        self._source_count = 0
        self._output_count = 0
        self._previous = None
        self._filter_tail = np.zeros(48)
        n = np.arange(49) - 24
        cutoff = min(.45, 5000 / self.sample_rate)
        self._filter = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(49)
        self._filter /= self._filter.sum()
        self._buffer = np.empty(0)
        self._base = 0
        self._scan = 0
        self._spectra = None
        self._candidates = []
        self._groups = []
        self._ui_enabled = True
        self._last_attempt = -100000
        self._iq = None
        self.turbo_attempts = 0

    def gap(self, sample_start):
        if sample_start < self.input_clock:
            raise ValueError('CQ input gap moves backwards')
        self.reset(sample_start=sample_start)

    def _publish_span(self, span):
        self._span_ledger.publish(span)
        event = SpanPublished(span)
        changes = self._span_events.setdefault(span.id, [])
        first = next((e for e in changes if isinstance(e, SpanPublished)), None)
        changes[:] = [first, event] if first is not None and first != event else [event]

    def _ensure_span(self, candidate):
        kind = candidate[5]
        if kind != 'beacon' and not kind.startswith('ui'):
            return None
        level = 2 if kind == 'beacon' else int(kind[-1])
        origin = self._input_sample(candidate[1])
        reach = round(1024 * self.sample_rate / FS)
        previous = next((s for s in self._span_ledger.spans
                         if s.level == level and abs(s.header_origin - origin) <= reach), None)
        if previous is not None:
            return previous
        span = UISpan(self._next_span_id, self.tune_epoch, level, origin,
                      sample_rate=self.sample_rate)
        self._next_span_id += 1
        self._publish_span(span)
        return span

    def _refine_span(self, candidate, raw):
        span = self._ensure_span(candidate)
        if span.resolution in ('group', 'beacon', 'singleton'):
            return span
        count = UI.packet_count(raw[1])
        refined = replace(span, possible_packet_count=count,
                          ambiguous_2_or_18=not raw[1] & 15 and bool(raw[1] & 16))
        if refined != span:
            self._publish_span(refined)
        return refined

    def _resolve_span(self, span, outcome, *, validated_end=None, count=None):
        before = self._span_ledger.spans
        self._span_ledger.resolve(span.id, outcome, validated_end=validated_end,
                                  count=count, tune_epoch=span.tune_epoch)
        if before != self._span_ledger.spans:
            changes = self._span_events.setdefault(span.id, [])
            changes[:] = [e for e in changes if isinstance(e, SpanPublished)]
            changes.append(SpanResolved(span.id, span.tune_epoch, outcome, validated_end, count))

    def _body_end(self, candidate, level, count=1, offset=0):
        end = candidate[1] + (67072 + (count - 1) * UI.PERIOD[level]
                              + UI.BODY_SAMPLES[level]) // 4 + offset
        return self._input_sample(end)

    def _resample(self, samples):
        x = np.asarray(samples).reshape(-1)
        if self.sample_rate > 24000:
            joined = np.concatenate((self._filter_tail, x))
            self._filter_tail = joined[-48:]
            x = np.convolve(joined, self._filter, mode='valid')
        start = self._source_count
        self._source_count += len(x)
        if self._previous is not None:
            x = np.concatenate(([self._previous], x))
            start -= 1
        self._previous = x[-1]
        stop = int(np.floor((self._source_count - 1) * FS / self.sample_rate)) + 1
        positions = np.arange(self._output_count, stop) * self.sample_rate / FS - start
        self._output_count = stop
        return np.interp(positions, np.arange(len(x)), x)

    def feed(self, samples, *, ui=True, sample_start=None):
        if sample_start is not None and sample_start != self.input_clock:
            self.gap(sample_start)
        self._ui_enabled = ui
        if not ui:
            self._groups.clear()
            self._candidates = [c for c in self._candidates if not c[5].startswith('ui')]
        samples = np.asarray(samples)
        if not samples.size:
            return []
        if self._iq is None:
            self._iq = np.iscomplexobj(samples)
        elif self._iq != np.iscomplexobj(samples):
            raise ValueError('CQ input cannot change between real audio and centered IQ')
        results = []
        for start in range(0, len(samples), max(1, int(self.sample_rate // 4))):
            x = self._resample(samples[start:start + max(1, int(self.sample_rate // 4))])
            self._buffer = np.concatenate((self._buffer, x))
            end = self._base + len(self._buffer)
            if end - self._scan >= SEARCH_WAIT:
                self._search()
            ready = [c for c in self._candidates if end >= c[1] + 32*SYMBOL + BODY[c[5]] + c[4]]
            for candidate in ready:
                self._candidates.remove(candidate)
                late = candidate[4] == LATE
                if not late and candidate[5] != 'ui2':
                    if candidate[1] <= self._last_attempt + 1024:
                        continue
                    self._last_attempt = candidate[1]
                attempts = self.turbo_attempts
                decoder = self._decode if candidate[5] == 'cq' else self._decode_ui if candidate[5].startswith('ui') else self._decode_beacon
                result = decoder(candidate)
                if result is not None:
                    results.append(result)
                elif not late and not candidate[5].startswith('ui') and self.turbo_attempts == attempts:
                    self._add_candidate(candidate[:4] + (LATE, candidate[5]))
            for group in list(self._groups):
                result = self._advance_ui(group, end)
                if result is not None:
                    results.append(result)
            keep = max(0, min(self._scan - self._base,
                              len(self._buffer) - (SYMBOL + 2048 if self._groups else 8 * FS)))
            needed = [c[1] - 512 for c in self._candidates]
            needed += [g.slot - 384 for g in self._groups if not g.failed]
            if needed:
                keep = min(keep, max(0, min(needed) - self._base))
            self._buffer = self._buffer[keep:]
            self._base += keep
            retained = self.input_clock - self._span_ledger.route_retention
            self._span_ledger.set_frontier('events', retained)
            self._span_ledger.advance(self.input_clock, self.classification_watermark)
            live_ids = {s.id for s in self._span_ledger.spans}
            self._span_events = {i: events for i, events in self._span_events.items() if i in live_ids}
        return results

    def _search(self):
        at = self._scan - self._base
        frames = np.lib.stride_tricks.sliding_window_view(self._buffer[at:], SYMBOL)[::HOP]
        power = abs(np.fft.fft(frames, axis=1)) ** 2
        first = self._scan // HOP
        self._scan += len(frames) * HOP
        if self._spectra is not None:
            first -= len(self._spectra)
            power = np.concatenate((self._spectra, power))
        self._spectra = power[-SEARCH_HISTORY:]
        count = len(power) - SEARCH_HISTORY
        if count <= 0:
            return
        center = 0 if self._iq else 64
        variants = [(kind, sign, shift) for kind in HEADERS
                    if self._ui_enabled or not kind.startswith('ui')
                    for sign in (1, -1) for shift in SHIFTS]
        scores = []
        indices = np.arange(count)[:, None] + 4 * np.arange(32)
        for kind, sign, shift in variants:
            bins = (center + sign * (HEADERS[kind] - 64) + shift) % SYMBOL
            band = (center + sign * (np.arange(*BAND[kind]) - 64) + shift) % SYMBOL
            peak = np.maximum(power[:, band].max(1), 1e-30)
            scores.append((power[indices, bins] / peak[indices]).sum(1))
        scores = np.asarray(scores)
        best = scores.argmax(0)
        hits = np.flatnonzero(scores[best, np.arange(count)] >= HEADER_MIN)
        for group in np.split(hits, np.flatnonzero(np.diff(hits) > 4) + 1):
            if not len(group):
                continue
            i = group[np.argmax(scores[best[group], group])]
            kind, sign, shift = variants[best[i]]
            candidate = (float(scores[best[i], i]), int((first + i) * HOP), sign, shift, EARLY, kind)
            self._ensure_span(candidate)
            previous = next((c for c in self._candidates if abs(c[1] - candidate[1]) <= 1024), None)
            if previous is not None:
                if previous[0] >= candidate[0]:
                    continue
                self._candidates.remove(previous)
            if candidate[1] > self._last_attempt + 1024:
                self._add_candidate(candidate)

    def _add_candidate(self, candidate):
        self._ensure_span(candidate)
        while len(self._candidates) + len(self._groups) >= 4:
            if not self._candidates:
                return
            self._candidates.remove(min(self._candidates, key=lambda c: c[1]))
        self._candidates.append(candidate)
        kind = candidate[5]
        if (self.ui_header is not None and candidate[4] == EARLY
                and (kind.startswith('ui') or kind == 'beacon')):
            self.ui_header(self._input_sample(candidate[1]), 2 if kind == 'beacon' else int(kind[-1]))

    def _decode(self, candidate):
        sample, sign, shift = candidate[1:4]
        x = self._buffer.conj() if sign == -1 else self._buffer
        center = 0 if self._iq else sign * 64
        guess = sample - self._base + 34 * SYMBOL + 384
        bins = (np.arange(54, 75) - 64 + center) % SYMBOL

        def lock(offsets):
            offsets = offsets[(offsets >= 0) & (offsets + 92 * SYMBOL <= len(x))]
            if not len(offsets):
                return None
            chunks = x[offsets[:, None, None] + REFERENCES[None, :, None] * SYMBOL + np.arange(SYMBOL)]
            best = None
            for hz in shift * sign * FS / SYMBOL + np.arange(-15, 16, 3):
                rotation = np.exp(-2j * np.pi * hz * np.arange(SYMBOL) / FS)
                power = abs(np.fft.fft(chunks * rotation, axis=-1)[..., bins]) ** 2
                score = (power[:, np.arange(24), REFERENCE_BINS - 54]
                         / np.maximum(power.max(-1), 1e-30)).sum(1)
                k = int(score.argmax())
                if best is None or score[k] > best[0]:
                    best = (float(score[k]), int(offsets[k]), hz)
            return best

        if candidate[4] == EARLY:
            best = lock(np.arange(guess - 640, guess + EARLY + 1, 16))
        else:
            # The reference score is a symbol wide in time: a coarse pass over
            # the whole late window, then a fine pass around its peak.
            best = lock(np.arange(guess - 640, guess + LATE + 1, 64))
            best = best and lock(np.arange(best[1] - 48, best[1] + 49, 16))
        if best is None:
            return None
        score, offset, hz = best
        if score < REFERENCE_MIN:
            return None
        body = x[offset:offset + 92 * SYMBOL]
        rotation = np.exp(-2j * np.pi * hz * np.arange(SYMBOL) / FS)
        power = abs(np.fft.fft(body.reshape(92, SYMBOL) * rotation, axis=1)[:, bins]) ** 2
        if np.count_nonzero(power[REFERENCES].argmax(1) == REFERENCE_BINS - 54) < REFERENCE_MIN:
            return None
        self.turbo_attempts += 1
        frame = decode_power(power)
        if frame is None:
            return None
        sn = self._cq_quality(x, offset, hz, center)
        return parse_frame(frame, sample=sample, sn=sn)

    @staticmethod
    def _cq_quality(x, offset, hz, center):
        # Acquisition accepts a symbol-wide plateau of reference hits. That
        # is enough to decode, but mixed neighbouring symbols look like noise
        # to an SN estimator. Refine timing by reference energy concentration.
        offsets = offset + np.arange(-SYMBOL // 2, SYMBOL // 2 + 1, 16)
        offsets = offsets[(offsets >= 0) & (offsets + 92 * SYMBOL <= len(x))]
        chunks = x[offsets[:, None, None] + REFERENCES[None, :, None] * SYMBOL
                   + np.arange(SYMBOL)]
        tones = REFERENCE_BINS - 64 + center
        bins = np.arange(54, 75) - 64 + center
        best = None
        for frequency in hz + np.arange(-18, 19, 3):
            rotation = np.exp(-2j * np.pi * frequency * np.arange(SYMBOL) / FS)
            power = abs(np.fft.fft(chunks * rotation, axis=-1)) ** 2
            signal = power[:, np.arange(len(tones)), tones % SYMBOL].sum(axis=1)
            total = power[..., bins % SYMBOL].sum(axis=(1, 2))
            scores = signal / np.maximum(total, 1e-30)
            k = int(scores.argmax())
            if best is None or scores[k] > best[0]:
                best = scores[k], chunks[k] * rotation
        return _reference_quality(best[1], tones, bins, FS)

    def _decode_beacon(self, candidate):
        """The record-1 body behind a beacon header, read by the BW500 receiver at
        48 kHz: the window is moved to 1500 Hz real audio, the header's offset
        taken out, and the body's own grid, lead and frequency searched over the
        candidate's early or late window."""
        span = self._ensure_span(candidate)
        sample, sign, shift, window = candidate[1:5]
        x = self._buffer.conj() if sign == -1 else self._buffer
        dw = B.RECORD.dw // 4
        guess = sample - self._base + 32 * SYMBOL
        lo = max(0, guess - 2 * dw)
        hi = min(len(x), guess + BODY['beacon'] + window + 2 * dw)
        if hi - lo < (B.RECORD.lead + B.RECORD.ncols) * dw:
            self._resolve_span(span, 'failed')
            return None
        n = np.arange(hi - lo)
        analytic = x[lo:hi] if self._iq else hilbert(x[lo:hi])
        offset = (0 if self._iq else 1500) + shift * sign * FS / SYMBOL
        best = None
        for hz in (-12, 0, 12):
            seg = np.real(analytic * np.exp(2j * np.pi * (1500 - offset - hz) * n / FS))
            x48 = resample_poly(seg, 4, 1)
            first = 4 * (guess - lo) - B.RECORD.dw
            for start in range(first, first + 4 * window, B.RECORD.lead * B.RECORD.dw):
                stop = start + (B.RECORD.lead + B.RECORD.ncols + _rx._L3_LEAD_MAX) * B.RECORD.dw
                if stop > len(x48):
                    break
                aligned = _rx._l3_align(x48, start, stop, B.LEVEL)
                if best is None or aligned[2] > best[0][2]:
                    best = (aligned, x48, start, stop)
        if best is None or best[0][2] < _rx._L3_GUARD_MIN // 2:
            self._resolve_span(span, 'failed')
            return None
        aligned, x48, start, stop = best
        self.turbo_attempts += 1
        frame = _rx._l3_frames(start, stop, *aligned, B.LEVEL)[0]
        if not frame.crc_ok:
            self._resolve_span(span, 'failed')
            if self._ui_enabled:
                self._start_ui(candidate, 2, None, None)
            return None
        raw = bytes(frame.frame_bytes)
        body = aligned[0] + (aligned[1] - B.RECORD.lead) * B.RECORD.dw
        body_end = self._input_sample(self._base + lo + (body + UI.BODY_SAMPLES[2]) / 4)
        if UI.ui_type(raw[0]):
            if not self._ui_enabled:
                return None
            if raw[0] & 128:
                # Keep the CRC-clean body's timing when leaving the shared
                # beacon path; later slots follow each decoded packet's grid.
                onset = self._base + lo + body // 4
                group = self._start_ui(candidate, 2, raw, quality(x[lo:hi], offset),
                                       offset=onset - sample - 67072 // 4)
                if group is not None:
                    return self._advance_ui(group, self._base + len(self._buffer))
                return None
            if self.ui_deadline is not None:
                self.ui_deadline(span.header_origin, self._body_end(candidate, 2)
                                 + round(26048 * self.sample_rate / 48000))
            result = UI.parse_packets([raw], level=2, sample=sample, sn=quality(x[lo:hi], offset))
            self._resolve_span(span, 'singleton' if result is not None else 'failed',
                               validated_end=body_end if result is not None else None,
                               count=1 if result is not None else None)
            return result
        result = B.parse_payload(frame.frame_bytes, sample=sample,
                                 sn=beacon_quality(x48, aligned[0] + aligned[1] * B.RECORD.dw))
        if result is not None and self.ui_deadline is not None:
            self.ui_deadline(span.header_origin, None)
        self._resolve_span(span, 'beacon' if result is not None else 'failed',
                           validated_end=body_end if result is not None else None,
                           count=1 if result is not None else None)
        return result


    def _ui_slot(self, candidate, level, index, hz=None, offset=0):
        sample, sign, shift = candidate[1:4]
        nominal = sample + (67072 + index * UI.PERIOD[level]) // 4
        guess = nominal + offset - self._base
        lo = max(0, guess - 384)
        hi = min(len(self._buffer), guess + (UI.BODY_SAMPLES[level] + 2048) // 4)
        if guess < 0 or hi <= lo:
            return None, None, None, offset
        x = self._buffer[lo:hi]
        analytic = x if self._iq else hilbert(x)
        if sign == -1:
            analytic = analytic.conj()
        carrier = (0 if self._iq else sign * 1500) + sign * shift * FS / SYMBOL
        n = np.arange(len(analytic))

        def attempted():
            self.turbo_attempts += 1

        def aligned(start):
            nonlocal offset
            offset = self._base + lo + start // 4 - nominal

        for frequency in ((0, -12, 12) if hz is None else (hz,)):
            real = np.real(analytic * np.exp(2j * np.pi * (1500 - carrier - frequency) * n / FS))
            x48 = resample_poly(real, 4, 1)
            raw = UI.decode_packet(x48, 4 * (guess - lo), level=level,
                                   on_attempt=attempted, on_alignment=aligned)
            if raw is not None:
                return raw, quality(x, carrier), frequency, offset
        return None, None, None, offset

    def _start_ui(self, candidate, level, raw, sn, hz=None, offset=0):
        span = self._ensure_span(candidate)
        n = UI.packet_count(raw[1]) if raw is not None else 18
        group = _UIGroup(candidate, level, UI.deadline(4 * candidate[1], level, n) // 4,
                         [raw] if raw is not None else [], raw is None, sn, hz, offset)
        if group.extended and any(g.extended for g in self._groups):
            logging.getLogger(__name__).debug('Overlapping extended UI group dropped')
            self._resolve_span(span, 'failed')
            return None
        if raw is not None:
            span = self._refine_span(candidate, raw)
        else:
            self._resolve_span(span, 'failed')
        self._groups.append(group)
        if self.ui_deadline is not None:
            self.ui_deadline(span.header_origin, self._input_sample(group.deadline))
        return group

    def _advance_ui(self, group, end):
        span = self._ensure_span(group.candidate)
        while not group.failed and end >= group.slot + (UI.BODY_SAMPLES[group.level] + 2048) // 4:
            raw, _, _, offset = self._ui_slot(group.candidate, group.level, len(group.frames),
                                            group.hz, group.offset)
            if raw is None or not UI.ui_type(raw[0]) or raw[0] & 0x7F != group.frames[0][0] & 0x7F:
                group.failed = True
                self._resolve_span(span, 'failed')
                break
            group.frames.append(raw)
            group.offset = offset
            if not raw[0] & 128:
                result = UI.parse_packets(group.frames, level=group.level,
                                          sample=group.sample, sn=group.sn)
                if result is not None:
                    self._resolve_span(span, 'group', count=len(group.frames),
                        validated_end=self._body_end(group.candidate, group.level,
                                                      len(group.frames), group.offset))
                    self._groups.remove(group)
                    return result
                group.failed = True
                self._resolve_span(span, 'failed')
                break
            if len(group.frames) == 18:
                group.failed = True
                self._resolve_span(span, 'failed')
                break
            if len(group.frames) >= span.possible_packet_count:
                span = replace(span, possible_packet_count=18)
                self._publish_span(span)
            new_deadline = UI.deadline(4 * group.sample, group.level, len(group.frames) + 1) // 4
            if new_deadline > group.deadline:
                if not group.extended and any(g is not group and g.extended for g in self._groups):
                    group.failed = True
                    self._resolve_span(span, 'failed')
                    break
                group.deadline = new_deadline
                if self.ui_deadline is not None:
                    self.ui_deadline(span.header_origin, self._input_sample(group.deadline))
        if end >= group.deadline:
            self._resolve_span(span, 'truncated')
            self._groups.remove(group)
        return None

    def _decode_ui(self, candidate):
        span = self._ensure_span(candidate)
        level = int(candidate[5][-1])
        raw, sn, hz, offset = self._ui_slot(candidate, level, 0)
        if raw is not None and not UI.ui_type(raw[0]):
            self._resolve_span(span, 'failed')
            return None
        if raw is not None and not raw[0] & 128:
            if self.ui_deadline is not None:
                self.ui_deadline(span.header_origin, self._body_end(candidate, level)
                                 + round(26048 * self.sample_rate / 48000))
            result = UI.parse_packets([raw], level=level, sample=candidate[1], sn=sn)
            self._resolve_span(span, 'singleton' if result is not None else 'failed',
                validated_end=self._body_end(candidate, level, offset=offset) if result is not None else None,
                count=1 if result is not None else None)
            return result
        group = self._start_ui(candidate, level, raw, sn, hz, offset)
        if group is not None:
            return self._advance_ui(group, self._base + len(self._buffer))
        return None
