# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Bounded, receive-only DATA candidates and idle replies for the passive watch.

The ordinary level 1–4 receiver remains independent. This reader tries the tone
and OFDM families only once per energy-delimited emission, never runs ARQ, and
returns only CRC-validated frames. A decode call can search multiple rows within
its family. Twelve-second emissions and two pending candidates bound storage;
the caller schedules at most one candidate across all channels per half second.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..kestrel.arq import phy
from ..kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK

FS = 48000
STEP = 960
QUIET = 4 * STEP
PAD = 12000
MAX_SPAN = 12 * FS
MAX_PENDING = 2


class IdleReplies:
    """Bounded BW500 idle evidence, without attributing a transmitter or running ARQ.

    Only explicitly supplied call keys are tested. A key names the addressee,
    not the station transmitting it. Single-tone replies need a complete tail
    and twenty clear matching tones; two-tone replies must beat the competing
    control family on their weaker carriers. Energy alone is not a reply.
    """
    def __init__(self):
        self.raw = np.empty(0)
        self.samples = 0
        self.scanned = 0
        self.last = {}
        self.tones = {}
        self.pairs = {}

    def feed(self, samples, calls):
        self.samples += len(samples)
        self.raw = np.concatenate((self.raw, samples))[-2 * FS:]
        if len(self.raw) < int(1.5 * FS) or self.samples - self.scanned < FS // 4:
            return []
        self.scanned = self.samples
        kind = VF.for_bw(VF.SESSION_TURN_IDLE_RESPONDER, '500')
        band = MK.band_for('500')
        track = VA._top3_track(self.raw, band=band)
        out = []
        calls = tuple(dict.fromkeys(c for c in calls if c))
        self.tones = {c: t for c, t in self.tones.items() if c in calls}
        self.pairs = {c: p for c, p in self.pairs.items() if c in calls}
        self.last = {c: at for c, at in self.last.items() if c in calls}
        power = None
        for call in calls:
            if call not in self.tones:
                self.tones[call] = np.asarray(VF.handshake_tones(call, kind))
            tones = self.tones[call]
            heard, at = VA._payload_fit(self.raw, kind, tones, track, band)
            clear = heard[1:] >= 0
            if (clear.sum() < 20 or not clear[-1]
                    or np.any(heard[1:][clear] != tones[1:][clear])):
                # The same idle/ACK state also has an eleven-symbol two-tone
                # form (K5CG after QSY). A single-tone-only reader hid it.
                if call not in self.pairs:
                    self.pairs[call] = (VF.control_bursts(call, '500', 1)[1],
                                       VA._control_family(call, call, '500'))
                pairs, family = self.pairs[call]
                if power is None:
                    power = VA._band_power(self.raw, band)
                if VA._frame_fit(power, pairs, band)[0] < 2.0:
                    continue
                at = VA._soft_pairs(power, pairs, family, band)
                if at is None or at + len(pairs) * MK.HOP > len(self.raw):
                    continue
            start = self.samples - len(self.raw) + at
            if start - self.last.get(call, -FS * 10) < FS:
                continue
            self.last[call] = start
            out.append(call)
        return out


@dataclass(frozen=True)
class Frame:
    start: int
    end: int
    level: int
    frame_bytes: bytes
    crc_ok: bool = True
    is_connect: bool = False


class HighSpeed:
    def __init__(self):
        self.samples = 0
        self.raw = np.empty(0)
        self.origin = 0
        self.floor = deque(maxlen=3000)
        self.start = self.last_hot = None
        self.too_long = False
        self.pending = deque()
        self.base = deque(maxlen=16)
        self.notices = []
        self.attempts = self.skipped = 0

    def _skip(self, why):
        self.skipped += 1
        self.notices.append('passive high-speed gap: ' + why)

    def note_base(self, start, end):
        self.base.append((start, end))

    def feed(self, samples):
        x = np.asarray(samples, float)
        before = self.samples
        self.samples += len(x)
        self.raw = np.concatenate((self.raw, x))[-(MAX_SPAN + 2 * PAD):]
        self.origin = self.samples - len(self.raw)
        floor = (float(np.percentile(self.floor, 20)) if self.floor
                 else float(np.mean(x * x)) if len(x) else 0.)
        for i in range(0, len(x), STEP):
            part = x[i:i + STEP]
            power = float(np.mean(part * part)) if len(part) else 0.
            hot = power > max(floor * 3.2, 1e-14)
            self.floor.append(power)
            at = before + i
            if hot:
                if self.start is None:
                    self.start, self.too_long = at, False
                self.last_hot = at + len(part)
                if self.last_hot - self.start > MAX_SPAN and not self.too_long:
                    self._skip('emission exceeds 12 seconds')
                    self.too_long = True
            elif self.start is not None and at + len(part) - self.last_hot >= QUIET:
                if not self.too_long and self.last_hot - self.start >= 2 * FS:
                    if len(self.pending) == MAX_PENDING:
                        self.pending.popleft()
                        self._skip('candidate queue full')
                    self.pending.append((self.start, self.last_hot))
                self.start = self.last_hot = None
                self.too_long = False
        while self.pending and self.pending[0][0] - PAD < self.origin:
            self.pending.popleft()
            self._skip('candidate expired before decode')

    @property
    def ready(self):
        return bool(self.pending and self.pending[0][1] + PAD <= self.samples)

    def read(self):
        if not self.ready:
            return []
        start, end = self.pending.popleft()
        # A base frame from this actual emission was already passed upstream.
        # Padded candidate windows may overlap a different adjacent emission.
        if any(start - STEP <= a < end for a, _ in self.base):
            return []
        a, b = max(self.origin, start - PAD), min(self.samples, end + PAD)
        x = self.raw[a - self.origin:b - self.origin]
        self.attempts += 1
        for expected in (6, 7):
            receiver = phy._family('500', expected)[0]
            found = receiver.decode_burst(x, 500, expected)
            if found.frames:
                located = (1 if expected == 6 else
                           found.quality.raw_bits // receiver.ROWS[500, found.level].coded)
                if located > len(found.frames):
                    self._skip(f'partial DATA emission: {len(found.frames)} of {located} frames validated')
                return [Frame(start, end, found.level, body) for body in found.frames]
        return []

    def drain_notices(self):
        notices, self.notices = self.notices, []
        return notices
