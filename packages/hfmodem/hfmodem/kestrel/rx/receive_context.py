# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Owner-thread UI classification on the continuous transport clock."""
from .cq_stream import CQDetector, FS, HOP, SEARCH_HISTORY, SEARCH_WAIT, RESAMPLE_DELAY
from .ui_spans import UISpanLedger

INPUT_STEP = 4800
# Search retains 124 spectra, can wait 2048 + SYMBOL at 12 kHz, and
# receives at most INPUT_STEP at 48 kHz. Include filter delay and rounding.
# Coverage gains 67072 - 16384 from header-to-training geometry. Requiring
# coverage past the conservative frame end (rather than training) leaves
# this same lag behind that end. Round up to a complete transport step.
_SCANNER_LAG = (SEARCH_HISTORY * HOP + SEARCH_WAIT) * (48000 // FS)
HOLD_SAMPLES = ((_SCANNER_LAG + INPUT_STEP + RESAMPLE_DELAY + 4 - (67072 - 16384)
                 + INPUT_STEP - 1) // INPUT_STEP) * INPUT_STEP


class ReceiveContext:
    def __init__(self, *, sample_start=0, tune_epoch=0):
        self.detector = CQDetector(sample_start=sample_start, tune_epoch=tune_epoch)
        self.ledger = UISpanLedger(tune_epoch)

    @property
    def input_clock(self):
        return self.detector.input_clock

    @property
    def watermark(self):
        return self.detector.classification_watermark

    def publish(self):
        for event in self.detector.take_span_events():
            self.ledger.apply(event)

    def feed(self, samples, sample_start, *, sink=None):
        skip = max(0, self.input_clock - sample_start)
        samples, sample_start = samples[skip:], sample_start + skip
        if not len(samples):
            return
        for at in range(0, len(samples), INPUT_STEP):
            results = self.detector.feed(samples[at:at + INPUT_STEP],
                                         sample_start=sample_start + at)
            self.publish()
            if sink is not None:
                for result in results:
                    sink(result)

    def advance(self, frontiers):
        self.publish()
        for route, sample in frontiers.items():
            self.ledger.set_frontier(route, sample)
        self.ledger.advance(self.input_clock, self.watermark)

    def gap(self, sample_start):
        self.detector.gap(max(sample_start, self.input_clock))
        self.publish()

    def retune(self, tune_epoch, sample_start):
        self.detector.reset(tune_epoch=tune_epoch,
                            sample_start=max(sample_start, self.input_clock))
        self.ledger.retune(tune_epoch)
