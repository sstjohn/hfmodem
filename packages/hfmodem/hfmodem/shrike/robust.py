# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The bounded P2 exchange between a robust answer and P3 entry.

The caller's raster remains the clock. Acquisition measures a P2 reference;
FSK onset is only a search hint, and no prediction is a measured P3 answer.
"""
from dataclasses import dataclass, field
import numpy as np
from . import compress, p1rx, p2rx, p3acquire, pactor2


@dataclass(frozen=True)
class Control:
    entry: int
    at: int
    swapped: bool
    errors: int = 0


def acquire(audio, at: int, *, radius: int, fs: int = 48000,
            preferred_swap: bool = False, lower: int | None = None) -> Control | None:
    """Read heard P2 words over one bounded bracket, sharing its transform.

    `at` and the returned instant name the upper virtual lane's reference,
    as in p2rx.control_signal_at. Choose the exact alignment nearest the
    prediction; its ordinary +/-2.5 ms tolerance is included by the caller.
    """
    z = p2rx.bin_phasors(audio, fs)
    entries = p1rx.ROBUST_P2_ENTRIES
    table = p2rx.cs_table(True)[list(entries)]
    frame = round(at / fs * p2rx.ACQ_RATE) + p2rx.CS_FRAME_LAG
    slack = round(radius / fs * p2rx.ACQ_RATE)
    candidates = []
    for d in range(-slack, slack + 1):
        if lower is not None and d < lower:
            continue
        for swap in (preferred_swap, not preferred_swap):
            soft = p2rx.cs_softs(z, p2rx.CENTRE_BIN_PAIR, frame+d, swap)
            if soft is None or not np.any(soft):
                continue
            dist = np.count_nonzero(table != (soft < 0), axis=1)
            found = np.flatnonzero(dist == 0)
            if len(found) == 1:
                candidates.append((abs(d), swap != preferred_swap, d, swap,
                                   entries[int(found[0])]))
    if not candidates:
        return None
    _, _, d, swap, entry = min(candidates)
    return Control(entry, round((frame+d-p2rx.CS_FRAME_LAG)/p2rx.ACQ_RATE*fs), swap)


@dataclass
class Bridge:
    mycall: str
    fsk_phase: int
    text: str = 'pmc'
    max_cycles: int = 10
    force_after: int | None = None
    offset_hz: float = 0.0
    marker_flag: str = 'parity'
    opening: str = 'none'
    fs: int = 48000
    fsk_sense: int | None = None
    fsk_slot: int | None = None
    phase: int | None = None
    candidate: tuple[int, int] | None = None
    silent: int = 0
    cycles: int = 0
    heard: list[dict] = field(default_factory=list)

    @property
    def payload(self) -> bytes:
        line = self.mycall.lower() + '\r'
        if self.text == 'ascii':
            return line.encode('ascii')
        n = pactor2.PATHS[1].crc_bytes - 3
        return compress.pmc(line, field_bytes=n)

    @property
    def status(self) -> int:
        return 0x19 if self.text == 'pmc' else 0x01

    @property
    def marker(self) -> int:
        flag = (self.status & 1) ^ (self.marker_flag == 'inverted')
        return pactor2.marker_index(pactor2.PATHS[1].level, False, flag)

    def read(self, audio, origin: int, boundary: int, slot: int) -> tuple[bool, str]:
        """Consume this keyed cycle once. Return (confirmed grant, report)."""
        self.cycles += 1
        acquiring = self.phase is None
        # Search FSK-40..+10 ms, preferring the recorded -20 ms separation.
        low = self.fsk_phase - round(.040*self.fs)
        high = self.fsk_phase + round(.010*self.fs)
        phase = self.phase if not acquiring else self.fsk_phase-round(.020*self.fs)
        # Crop once, with the acquisition filter's support and whole word.
        lo = max(0, boundary+low-origin-round(.045*self.fs))
        hi = min(len(audio), boundary+high-origin+round(.240*self.fs))
        x = audio[lo:hi]
        if not len(x):
            self.heard.append(dict(slot=slot, form=None, entry=None, unread=True))
            return False, (f'no audio at the answer instant: read {len(audio)} samples '
                           f'from {origin}, answer due at {boundary+self.fsk_phase}')
        if self.offset_hz:
            x = p3acquire.compensate(x, self.offset_hz)
        at = boundary+phase-origin-lo
        # An asymmetric bracket expressed by a centre and radius; preferred
        # -20 ms is used for the subsequent tracked reads.
        got = acquire(x, at, radius=round((.030 if acquiring else .0025)*self.fs),
                      fs=self.fs, preferred_swap=bool((self.cycles-1)%2),
                      lower=-16 if acquiring else None)
        grant = False
        if got is not None:
            measured = origin+lo+got.at-boundary
            if (self.phase is None and self.candidate is not None
                    and slot == self.candidate[0]+1
                    and abs(measured-self.candidate[1]) <= round(.0025*self.fs)):
                self.phase = measured
            self.candidate = (slot, measured)
            self.silent = 0
            grant = got.entry == 6 and self.phase is not None
            report = f'P2 entry {got.entry} phase={measured/self.fs:.6f}s errors=0 confirmed={self.phase is not None}'
            row = dict(slot=slot, form='P2', entry=got.entry, onset=origin+lo+got.at,
                       phase=measured/self.fs, offset_hz=self.offset_hz, confirmed=self.phase is not None)
        else:
            # The held answer is read where the first one was measured, in the
            # polarity the slot's parity gives it: a peer still answering in FSK
            # is heard, not silent, so it never starts a re-acquisition.
            sense = (None if self.fsk_sense is None or self.fsk_slot is None
                     else self.fsk_sense ^ ((slot - self.fsk_slot) & 1))
            fsk = p1rx.read_held_answer(audio, boundary+self.fsk_phase-origin, self.fs,
                                        sense=sense, offset_hz=self.offset_hz)
            if fsk is not None:
                self.silent = 0
                report = f'FSK entry {fsk.entry} errors={fsk.errors}'
                row = dict(slot=slot, form='FSK', entry=fsk.entry,
                           onset=origin+round(fsk.at*self.fs), offset_hz=fsk.offset_hz,
                           errors=fsk.errors, tone=fsk.tone)
            else:
                self.silent += 1
                if self.silent >= 2:
                    self.phase = self.candidate = None
                report = 'silence'
                row = dict(slot=slot, form=None, entry=None)
        self.heard.append(row)
        return grant, report
