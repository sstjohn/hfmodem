# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""History witness: peer reads/end evidence authorize credit before transport."""
from dataclasses import dataclass


@dataclass
class Saved:
    source: int | None
    kind: str
    phase: int
    sent: bool = False
    effect: tuple | None = None
    retry_double: bool = False
    retry_level: int | None = None


class DecisionHistory:
    def __init__(self):
        self.stream = b''
        self.anchor = None
        self.replay = None
        self.alt, self.trusted = 0, True
        self.saved = Saved(None, 'nak', 0)
        self.close = None
        self.blocks, self.pending = (), b''
        self.expected = None
        self.basis = None
        self.span = None
        self.new = b''
        self.partial = False
        self.retry_double = False
        self.retry_level = None

    def reject(self, reason, rejected=b'', code='R8', *, blocks=None, pending=None):
        start = self.replay[0] if self.replay else self.anchor[0] if self.anchor else len(self.stream)
        self.close = (code, reason, start, len(self.stream), rejected,
                      self.blocks if blocks is None else blocks,
                      self.pending if pending is None else pending)
        self.expected = None

    def negative(self, *, arm=False, phase=None):
        effect = None
        if arm and self.anchor:
            if self.anchor[2]:
                self.reject('POSSIBLE single recovery backstop')
                return
            effect = self.anchor[0], self.replay[2] if self.replay else len(self.stream)
        self.saved = Saved(None, 'nak', self.alt if phase is None else phase, effect=effect)
        self.expected, self.basis = 'nak', 'unread'

    def recovery(self, *, phase, inside, silent, phaseless, poll, explicit=True):
        self.close = None
        self.expected = self.basis = None
        a = self.saved
        if a.kind == 'nak' and not a.sent:
            self.expected, self.basis = 'nak', 'debt'
            return
        if a.kind == 'final':
            self.expected, self.basis = 'final', 'final'
            return
        if silent:
            if a.kind == 'nak':
                self.expected, self.basis = 'nak', 'debt'
            elif not inside:
                self.negative(arm=True)
            return
        proof = (self.trusted and phase in (0, 1) and not phaseless
                 and not (poll and a.phase == 1) and (a.phase, phase) != (0, 0))
        if a.kind == 'frame10' and (proof and a.phase == phase or inside):
            self.expected, self.basis = 'frame10', 'phase' if proof and a.phase == phase else 'bound'
        elif proof:
            if a.phase == phase:
                self.expected, self.basis = a.kind, 'phase'
            else:
                self.negative()
        elif a.kind in ('continue', 'frame10'):
            if inside:
                self.expected, self.basis = a.kind, 'bound'
            else:
                self.negative(arm=True)
        else:
            self.expected, self.basis = 'nak', 'debt'

    def window(self, attempt, statuses, verdict, limited, *, single, ceiling=False, envelope='inconclusive'):
        self.close = None
        self.expected = self.basis = None
        self.span = None
        read = [(i, f) for i, (f, s) in enumerate(zip(attempt.frames, statuses)) if s == 'read']
        observed = [s for s in statuses if s != 'invisible']
        if not observed:
            return
        if self.retry_double and attempt.level <= self.retry_level:
            if attempt.level < self.retry_level:
                verdict = 'NONE'
            elif attempt.level > 1 or envelope != 'single':
                verdict = 'POSSIBLE'
            else:
                verdict = 'NONE'
        self.retry_double = False
        if read and read[0][1].compressed:
            verdict = 'NONE'
        short_head = bool(read and not read[0][1].full and not read[0][1].last
                          and not read[0][1].compressed)
        certain = verdict == 'CERTAIN' or short_head
        tail_only = verdict == 'CERTAIN' and len(observed) == 1 and read and read[0][1].last
        complete = (not ceiling and 'X' not in observed and 'erased' not in observed
                    and not tail_only and (len(observed) == 2 or not certain and single
                                          and (verdict == 'NONE' or envelope != 'double')))
        self.partial = bool(not complete and not tail_only and limited and statuses[0] == 'read'
                            and (len(statuses) == 1 or statuses[1] != 'read'))
        if not complete and not self.partial:
            self.negative()
            self.saved.retry_double = bool(certain or envelope == 'double' or len(observed) == 2 and not all(
                s == 'read' for s in observed))
            self.saved.retry_level = attempt.level
            return
        credited = read[:1] if self.partial else read
        data = b''.join(f.replay for _, f in credited)
        start, end_replay = len(self.stream), self.replay
        if self.replay:
            low, frontier, cursor, other = self.replay
            if verdict == 'POSSIBLE' and len(credited) != 2:
                self.reject('POSSIBLE single inside replay', data)
                return
            start = low if cursor is None else cursor
            overlap = min(len(data), frontier-start)
            matches = data[:overlap] == self.stream[start:start+overlap]
            if other is not None:
                n = min(len(data), frontier-other)
                alternate = data[:n] == self.stream[other:other+n]
                if matches == alternate:
                    self.reject('nested candidates both agree' if matches else
                                'nested candidates neither agrees', data)
                    return
                if alternate:
                    start = other
                matches = True
            if not matches:
                if cursor is not None:
                    self.reject('replay cursor disagrees', data)
                    return
                start, end_replay = len(self.stream), None
            else:
                end_replay = (low, frontier, start+len(data), None) if start+len(data) < frontier else None
        self.new = data[max(0, len(self.stream)-start):]
        self.span = start, start+len(data)
        self.next_replay = end_replay
        terminal = not self.partial and credited[-1][1].last
        self.expected = 'frame10' if self.partial else 'final' if terminal else 'continue'
        self.basis = 'head' if self.partial else 'complete'
        self.next_anchor = start, start+len(data), verdict == 'POSSIBLE' and len(credited) == 1
        self.next_saved = Saved(attempt.number, self.expected, self.alt)

    def committed(self, *, parsed=None):
        self.stream += self.new
        self.anchor = self.next_anchor
        self.replay = None if self.expected == 'final' else self.next_replay
        self.saved = self.next_saved
        if parsed:
            blocks = self.blocks + parsed.blocks
            split = len(blocks) if self.expected == 'final' else next(
                (i for i, b in enumerate(blocks) if b.stored), len(blocks))
            self.blocks, self.pending = blocks[split:], parsed.pending

    def sent(self, air):
        a = self.saved
        if air and not a.sent:
            a.sent = True
            self.alt ^= 1
            if a.kind in ('nak', 'frame10'):
                self.trusted = False
            if a.kind == 'nak':
                self.retry_double = a.retry_double
                self.retry_level = a.retry_level
            if a.kind == 'nak' and a.effect:
                low, other = a.effect
                self.replay = (low, len(self.stream), None,
                               other if other < len(self.stream) else None)
            if a.kind == 'final':
                self.alt, self.trusted, self.replay = 0, True, None

    def verify_close(self, report):
        assert self.close is not None, ('unjustified close', report)
        code, reason, start, frontier, rejected, blocks, pending = self.close
        assert (report.code, report.reason, report.start, report.frontier,
                report.rejected, report.blocks, report.pending) == (
                    code, reason, start, frontier, rejected, blocks, pending)
