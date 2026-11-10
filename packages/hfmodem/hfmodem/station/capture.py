# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
from pathlib import Path
import sys
import uuid

from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.core.contactlog import ContactLog
from hfmodem.core.recording import ContactBoundaries, RecordingSession, publish_policy, resolve_recording


class StationCapture:
    def __init__(self, directory, source, *, audio=None, resolved=None):
        root = Path(directory).expanduser()
        self.directory = root / f'capture-{uuid.uuid4().hex}'
        self.audio = audio
        selected = resolved or resolve_recording()
        self.session = RecordingSession(selected.policy, self.directory, source=selected.source)
        self.contact_log = ContactLog(self.directory, protocol=None, application='Station',
                                      recording=self.session)
        self.rx = self.session.stream('rx', rate=CARD_RATE_HZ, clock='card')
        self.tx = self.session.stream('submitted-tx', rate=CARD_RATE_HZ, clock='dac')
        self.tx.row['source_anchor'] = dict(clock=self.rx.clock, sample=0, position=0)
        self.boundaries = ContactBoundaries(self.session, run=self.directory.name)
        self.rx.row.update(input=source, missing_samples=0, audio_lost_samples=0,
                           audio_overflows=0, audio_underruns=0)
        publish_policy(self.session)
        self._next = None
        self.missing = 0

    def positions(self):
        return {lane.clock: lane.position for lane in self.session.lanes.values()}

    def push(self, block):
        if self._next is not None:
            self.missing += max(0, block.n0 - self._next)
        self._next = block.n0 + len(block.samples)
        self.rx.push(block.n0, block.samples,
                     timing=dict(monotonic_origin=block.adc_time - block.n0 / self.rx.rate,
                                 clock_uncertainty_s=0.))

    def push_tx(self, first_sample, pcm):
        self.tx.push(first_sample, pcm)

    def flush_to(self, index):
        pass

    def resync(self, index):
        pass

    def drain(self):
        pass

    def close(self):
        self.boundaries.close_owner(self.positions(), 'capture closed')
        self.rx.row['missing_samples'] = self.missing
        if self.audio is not None:
            self.rx.row.update(audio_lost_samples=getattr(self.audio, 'lost', 0),
                               audio_overflows=getattr(self.audio, 'overflows', 0),
                               audio_underruns=getattr(self.audio, 'underruns', 0))
            if limitation := getattr(self.audio, 'timestamp_limitation', None):
                lost = (self.missing or self.rx.row['audio_lost_samples']
                        or self.rx.row['audio_overflows'])
                self.rx.row.update(
                    timestamp_limitation=limitation,
                    estimated_lost_samples=getattr(self.audio, 'estimated_lost', 0),
                    continuous=False if lost else None)
        try:
            self.session.close()
        except Exception as exc:
            print(f'Recording close: {exc}', file=sys.stderr, flush=True)
