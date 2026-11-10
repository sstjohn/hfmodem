# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Runtime UI occupancy, independent of packet assembly."""
from ..monitor import FS, SEG_FRAME
from ..rx.broadcast import deadline
from ..rx.cq_stream import CQDetector


class UIChannel:
    def __init__(self, segmenter, receive_context=None):
        self.segmenter = segmenter
        self.headers = {}
        self.threshold = None
        self.quiet = 0
        segmenter.preserve_floor_history = False
        segmenter.level_observer = self._level
        self.receive_context = receive_context
        self.detector = (receive_context.detector if receive_context is not None else
                         CQDetector())
        self.detector.ui_header = self._header
        self.detector.ui_deadline = self._deadline

    @property
    def active(self):
        return bool(self.headers) or self.segmenter.inb

    @property
    def qualified(self):
        return bool(self.headers)

    def _local(self, sample):
        if self.receive_context is None:
            return sample
        return sample - getattr(self.segmenter, "capture_start", 0)

    def _header(self, sample, level):
        sample = self._local(sample)
        if any(abs(sample - old) <= 4096 for old in self.headers):
            return
        if not self.headers:
            floor = self.segmenter.nf
            self.threshold = (floor or 1e-12) * self.segmenter.exit
            self.quiet = 0
            self.segmenter._floor_brms.clear()
            self.segmenter._floor_ibrms.clear()
            self.segmenter.preserve_floor_history = True
        if len(self.headers) < 4:
            self.headers[sample] = deadline(sample, level)
        self.segmenter.floor_frozen = bool(self.headers)

    def _deadline(self, sample, end):
        sample = self._local(sample)
        end = None if end is None else self._local(end)
        sample = next((old for old in self.headers if abs(sample - old) <= 4096), sample)
        if sample in self.headers:
            if end is None:
                del self.headers[sample]
            else:
                self.headers[sample] = end
        self.segmenter.floor_frozen = bool(self.headers)

    def _level(self, sample, rms):
        self.headers = {start: end for start, end in self.headers.items() if sample < end}
        if self.headers:
            self.quiet = self.quiet + SEG_FRAME if rms < self.threshold else 0
            if self.quiet >= FS // 2:
                self.headers.clear()
        self.segmenter.floor_frozen = bool(self.headers)

    def push(self, samples, *, sink=None, ui=True, activity=None):
        bursts = []
        for at in range(0, len(samples), FS // 10):
            block = samples[at:at + FS // 10]
            if sink is not None and self.receive_context is None:
                for result in self.detector.feed(block, ui=ui):
                    sink(result)
            bursts.extend(self.segmenter.push(block))
            if activity is not None:
                activity(block)
        return bursts
