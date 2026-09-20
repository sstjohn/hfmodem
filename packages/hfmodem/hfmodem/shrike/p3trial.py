# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""P3 reply clocks and timing trials. Coordinates are capture samples.

A retains peer-relative audio-start placement. B preserves the emitted entry
phase plus a 600 ms rotation. This state never grants permission to transmit;
the driver's ordinary admission, freshness and collision guards still apply.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

FS = 48000
CYCLE = 60000
ROTATION = 28800
LIMIT = 12
SECONDS = 20
DEFAULT_ENTRY_DELAY_MS = 4.875
"""Restore the 234 samples trimmed before the entry pulse at 48 kHz.

Stock P1-to-P3 clocks and the September 19 VE3KPG/K0NTS trials support this
entry epoch. Greeting reply placement is a separate timing decision.
"""


@dataclass
class ReplyClock:
    """Mail receive clock, independent of the initial-turn experiment limits.

    Start on the successful B +3.125 ms pulse epoch. After that, accepted
    role changes rotate this epoch with the grid; a failed break-in does not.
    This object neither grants TX permission nor ends an application turn.
    """
    entry_phase: int | None = None
    pulse_epoch: int | None = None

    def entry(self, phase):
        if self.pulse_epoch is None:
            self.entry_phase = phase

    def target(self, lead):
        if self.entry_phase is None:
            return None
        if self.pulse_epoch is None:
            self.pulse_epoch = self.entry_phase + ROTATION + 150
        return self.pulse_epoch - lead

    def rotate(self, samples):
        if self.pulse_epoch is not None:
            self.pulse_epoch += samples


@dataclass
class TimingTrial:
    arm: str
    entry_phase: int | None = None
    opened: int | None = None
    first_reply: int | None = None
    reason: str | None = None
    closing: bool = False
    opportunities: int = 0
    entries: list = field(default_factory=list)
    packets: list = field(default_factory=list)
    replies: list = field(default_factory=list)
    progress: list = field(default_factory=list)
    unbounded: bool = False
    reply_delay_ms: float = 0.0

    @property
    def filename(self):
        return 'p3-timing-trial.json'

    def __post_init__(self):
        if self.arm not in ('A', 'B'):
            raise ValueError('P3 timing trial must be A or B')
        if self.reply_delay_ms not in (0.0, 3.125):
            raise ValueError('P3 reply delay must be 0 or 3.125 ms')
        if self.reply_delay_ms and self.arm != 'B':
            raise ValueError('P3 reply delay requires timing trial B')

    @property
    def active(self):
        return self.opened is not None and self.reason is None

    def finish(self, reason):
        if self.reason is None:
            self.reason = reason
            print(f'    [p3 trial] {self.arm} stopped: {reason}; '
                  f'{self.opportunities} reply opportunities, '
                  f'{len(self.progress)} advancing ordinary fields', flush=True)

    def entry(self, phase, tx_number):
        if self.opened is not None or self.reason is not None:
            return
        if self.entry_phase is not None:
            error = (phase-self.entry_phase+CYCLE//2) % CYCLE-CYCLE//2
            if abs(error) > round(.005*FS):
                self.finish('entry retries have ambiguous phase')
                return
        self.entry_phase = phase
        self.entries.append(dict(phase=phase, tx=tx_number))
        print(f'    [p3 trial] {self.arm} emitted entry phase {phase}', flush=True)

    def packet(self, phase, status, payload, *, breakin, long_cycle):
        if self.reason is not None:
            return
        if self.opened is None:
            if not breakin:
                return
            if self.entry_phase is None or phase <= self.entry_phase:
                self.finish('no emitted entry establishes the timing origin')
                return
            self.opened = phase
            print(f'    [p3 trial] {self.arm} opened on CRC changeover at {phase}', flush=True)
        if self.packets and phase <= self.packets[-1]['phase']:
            return
        if self.check(phase):
            return
        self.packets.append(dict(phase=phase, seq=status & 3, status=status,
                                 payload_hex=payload.hex(), breakin=breakin,
                                 long_cycle=bool(long_cycle)))
        if long_cycle:
            self.finish('peer changed to long cycle')
        elif status & 0x80:
            self.finish('peer QRT')
        elif not breakin and status & 0x40:
            self.finish('peer requested role reversal')
        elif breakin and self.progress:
            self.finish('new peer turn')
        elif not breakin and payload and (status & 3) == (len(self.progress)+1) % 4:
            # Extended reception wraps the counter and can legitimately carry
            # identical bytes in successive packets. Duplicates keep their seq.
            if self.unbounded or all(p['payload_hex'] != payload.hex() for p in self.progress):
                self.progress.append(self.packets[-1])
                if len(self.progress) == 3 and not self.unbounded:
                    self.finish('inbound progression: ordinary counters 1, 2, 3')

    def target(self, peer_phase, lead):
        """Return an AUDIO-START epoch; L is the exact outgoing pulse lead."""
        if self.opened is None:
            raise ValueError('trial has no CRC changeover')
        pulse = (peer_phase+round(.890*FS)+lead if self.arm == 'A'
                 else self.entry_phase+ROTATION+round(self.reply_delay_ms*FS/1000))
        # Place the first opportunity after the initial packet, not after the
        # latest duplicate. Its deadline cannot be extended by decoding retries.
        if self.first_reply is None:
            minimum = self.opened+round(.81*FS)
            first = self.opened+round(.890*FS)+lead if self.arm == 'A' else pulse
            self.first_reply = first + max(0, (minimum-first+CYCLE-1)//CYCLE)*CYCLE
        return pulse-lead

    def check(self, now):
        if self.active:
            if self.first_reply is not None and now >= self.first_reply:
                elapsed = (now-self.first_reply)//CYCLE
                self.opportunities = max(self.opportunities,
                    elapsed if self.unbounded else min(LIMIT, elapsed))
            if self.unbounded:
                return self.reason
            if now >= self.opened+SECONDS*FS:
                self.finish('20 second trial limit')
            elif self.first_reply is not None and now >= self.first_reply+LIMIT*CYCLE:
                self.opportunities = LIMIT
                self.finish('12 reply opportunity limit')
        return self.reason

    def attempt(self, pulse):
        self.check(pulse)
        if self.reason is not None:
            return False
        if self.first_reply is None:
            raise ValueError('reply must be placed before attempting it')
        self.opportunities = max(self.opportunities,
            max(1, round((pulse-self.first_reply)/CYCLE)+1))
        if self.opportunities > LIMIT and not self.unbounded:
            self.finish('12 reply opportunity limit')
            return False
        return True

    def verdict(self):
        counters = ','.join(str(p['seq']) for p in self.progress) or 'none'
        delay = f' reply +{self.reply_delay_ms:g} ms' if self.reply_delay_ms else ''
        return (f'P3 TIMING TRIAL {self.arm}{delay}: ordinary counters {counters}; '
                f'{self.reason or "incomplete"}; local CRC evidence')

    def write(self, directory):
        result = asdict(self)
        result['limits'] = dict(opportunities=None if self.unbounded else LIMIT,
                               seconds=None if self.unbounded else SECONDS,
                               ordinary_fields=None if self.unbounded else 3)
        result['sample_rate'] = FS
        result['target_entry_rotation_ms'] = (600+self.reply_delay_ms if self.arm == 'B' else None)
        result['changeover_exited'] = bool(self.progress)
        result['progression'] = len(self.progress) >= 3
        Path(directory, self.filename).write_text(json.dumps(result, indent=2)+'\n')


@dataclass
class EntryTimingTrial(TimingTrial):
    """A: historical zero delay. B: the default 234-sample entry delay.

    Score entry acquisition only. Deadline starts at the first actual emission,
    never resets on grants, and stops before trying to advance the greeting.
    """

    @property
    def filename(self):
        return 'p3-entry-timing-trial.json'

    @property
    def delay_ms(self):
        return 0.0 if self.arm == 'A' else DEFAULT_ENTRY_DELAY_MS

    def finish(self, reason):
        if self.reason is None:
            self.reason = reason
            print(f'    [p3 entry trial] {self.arm} stopped: {reason}; '
                  f'{len(self.entries)} emitted entries', flush=True)

    def check(self, now):
        if self.entries and now >= self.entries[0]['phase']+SECONDS*FS:
            self.finish('20 second entry-acquisition limit')
        return self.reason

    def packet(self, phase, status, payload, *, breakin, long_cycle):
        if self.reason is not None or not self.entries:
            return
        if phase <= self.entries[0]['phase']:
            return
        # A packet starting inside our entry is not inbound evidence. Our
        # local TX pickup must not make a trial declare itself acquired.
        if any(e['phase']-round(.02*FS) <= phase < e['phase']+round(.84*FS)
               for e in self.entries):
            return
        if self.check(phase):
            return
        self.opened = phase
        self.packets.append(dict(phase=phase, seq=status & 3, status=status,
            payload_hex=payload.hex(), breakin=breakin, long_cycle=bool(long_cycle)))
        self.finish('CRC-valid inbound P3 packet')

    def verdict(self):
        return (f'P3 ENTRY TIMING TRIAL {self.arm}: {len(self.entries)} entries; '
                f'{self.reason or "incomplete"}; greeting progression not tested')

    def write(self, directory):
        result = asdict(self)
        result.update(sample_rate=FS, entry_delay_ms=self.delay_ms,
            limits=dict(seconds_from_first_entry=SECONDS),
            p3_crc_verified=bool(self.packets), progression_tested=False,
            entries_before_first_p3=(sum(e['phase'] < self.opened for e in self.entries)
                                     if self.opened is not None else None))
        Path(directory, self.filename).write_text(json.dumps(result, indent=2)+'\n')
