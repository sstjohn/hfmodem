# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Pure receive-window decisions: one window at a time, delivered at its answer.

``event`` consumes G/F/X/E/Q/R/C/T. A Ready action requests preparation, not
keying. The adapter extracts payloads and decodes flagged TEXT frames only
then, calls ``placement`` to preview the new replay bytes, previews its FILES
parser on those bytes, and calls ``commit``. All previews are non-mutating.
Actions are ordered; ``answer_before`` controls Key versus Emit ordering.
Call ``sent`` for each Key before supplying another event, even on refusal.
Timing, hint counts, waveform generation and codecs belong to the adapter.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Frame:
    body: bytes
    full: bool = True
    last: bool = False
    compressed: bool = False
    hint: int = 0
    ambiguous: bool = False


@dataclass(frozen=True)
class Prepared:
    data: bytes
    error: str | None = None


@dataclass
class Window:
    phase: int
    level: int
    verdict: str
    expected: frozenset[int]
    frames: tuple[Frame | None, ...]
    limited_pair: bool = False
    single: bool = False
    held: bool = False
    ready: bool = False
    partial: bool = False
    envelope: str = 'inconclusive'
    retry_floor: bool = False
    learn_single: bool = False


@dataclass
class Answer:
    attempt: int
    phase: int
    kind: str
    step: int = 0
    original: str = 'original'
    sent: bool = False
    effect: tuple[int, int] | None = None
    reanswers: int = 0
    retry_double: bool = False
    retry_level: int | None = None
    retry_single: bool = False


@dataclass(frozen=True)
class Anchor:
    start: int
    end: int
    level: int
    frames: tuple[Frame, ...]
    possible_single: bool


@dataclass(frozen=True)
class Replay:
    start: int
    frontier: int
    cursor: int | None = None
    alternative: int | None = None


@dataclass(frozen=True)
class Placement:
    start: int
    data: bytes
    new: bytes
    replay: Replay | None
    error: str | None = None


@dataclass(frozen=True)
class Block:
    """One complete FILES block, validated by the adapter's preview parser."""
    data: bytes
    stored: bool


@dataclass(frozen=True)
class FilesParse:
    """Preview on placement.new, starting from the last committed parser state.

    blocks are newly completed blocks, pending is the entire unparsed wire
    suffix. The adapter installs its preview decoder state only on Commit.
    A deflate block must satisfy EOF, exact input consumption and 32500 limit.
    """
    blocks: tuple[Block, ...] = ()
    pending: bytes = b''
    error: str | None = None


@dataclass(frozen=True)
class FilesHold:
    blocks: tuple[Block, ...] = ()
    pending: bytes = b''

    def preview(self, parsed: FilesParse, terminal: bool):
        blocks = self.blocks + parsed.blocks
        error = parsed.error
        if any(len(b.data) > 32500 for b in blocks):
            error = error or 'FILES block exceeds 32500'
        if terminal and parsed.pending:
            error = error or 'delivery ends inside a block'
        if error:
            return self, b'', Report('fault', error, blocks, parsed.pending)
        split = next((i for i, b in enumerate(blocks) if b.stored), len(blocks))
        if terminal:
            split = len(blocks)
        if sum(len(b.data) for b in blocks[split:]) > 128 * 1024:
            return self, b'', Report('R10', 'FILES hold exhausted', blocks, parsed.pending)
        emitted = b''.join(b.data for b in blocks[:split])
        return FilesHold(blocks[split:], parsed.pending), emitted, None


@dataclass(frozen=True)
class Ready:
    partial: bool = False


@dataclass(frozen=True)
class Key:
    attempt: int
    kind: str
    form: str
    step: int
    basis: str
    reanswer: bool = False
    budgeted: bool = True


@dataclass(frozen=True)
class Commit:
    start: int
    end: int
    new: bytes
    progressed: bool


@dataclass(frozen=True)
class Emit:
    data: bytes


@dataclass(frozen=True)
class Report:
    """Receiver-known unconfirmed replay span, rejected bytes and held blocks.

    A receive machine cannot count erased peer bytes. The adapter adds those
    from its session accounting; [start, frontier) is conservatively uncertain.
    """
    code: str
    reason: str
    blocks: tuple[Block, ...] = ()
    pending: bytes = b''
    start: int = 0
    frontier: int = 0
    rejected: bytes = b''


@dataclass(frozen=True)
class Close:
    report: Report


@dataclass(frozen=True)
class ResetDecoder:
    pass


class ReceiveWindow:
    """W/A/K/T ownership; no I/O, clocks, codecs or waveform imports.

    E.single means §3 single-frame evidence; E without it means the companion
    slot ended. F/X.envelope supplies measured extent; E may refine it when
    the burst ends (the floor retry's verdict then follows that measurement).
    Q/R.inside is the adapter's _answer_unheard onset comparison.
    R.recognized identifies a fitted idle/query; explicit selects query32.
    budget=False is the adapter's exhausted session/negative recovery budget.
    session_budget=False also closes a pre-greeting window whose NAKs are free.
    G.greeting=False reconstructs unread debt rather than granting a new turn.
    T is an abort unless a committed final A exists; new_session resets trust.
    """
    def __init__(self, mode='OFF', *, high_speed=True):
        if mode not in ('OFF', 'TEXT', 'FILES'):
            raise ValueError(mode)
        self.mode = mode
        self.high_speed = high_speed
        self.state = 'IDLE'
        self.W = self.A = self.K = self.T = None
        self.alt = 0
        self.trusted = True
        self.R = b''
        self.files = FilesHold()
        self.closed = False
        self._attempt = 0
        self._pending_key = None
        self._retry_double = False
        self._retry_level = None
        self._retry_single = False
        self.no_doubles = False
        self.before_data = False

    @property
    def S(self):
        return len(self.R)

    def _answer(self, phase, kind, *, effect=None, step=0, original='original'):
        self._attempt += 1
        self.A = Answer(self._attempt, phase, kind, step, original, effect=effect)

    def _key(self, *, basis, explicit=True, reanswer=False, budgeted=True):
        a = self.A
        if a.kind == 'continue':
            form = 'query32' if explicit or a.reanswers else a.original
            step = a.step if self.high_speed or form != 'query32' else 0
        else:
            form, step = a.kind, a.step
        key = Key(a.attempt, a.kind, form, step, basis, reanswer, budgeted)
        self._pending_key = key
        return [key]

    def _report(self, code, reason, *, rejected=b'', blocks=None, pending=None):
        start = self.T.start if self.T else self.K.start if self.K else self.S
        return Report(code, reason, self.files.blocks if blocks is None else blocks,
                      self.files.pending if pending is None else pending,
                      start, self.S, rejected)

    def _close(self, report):
        self.closed = True
        self.no_doubles = False
        self._retry_single = False
        self.before_data = False
        self.W = self.T = None
        self.state = 'IDLE'
        self.files = FilesHold()
        return [Close(report), ResetDecoder()]

    def _negative(self, phase, *, arm=False, key=True, retry_double=False, retry_level=None,
                  retry_single=False):
        effect = None
        if arm and self.K is not None:
            if self.K.possible_single:
                return self._close(self._report('R8', 'POSSIBLE single recovery backstop'))
            effect = self.K.start, self.T.cursor if self.T else self.S
        self._answer(phase, 'nak', effect=effect)
        self.A.retry_double = retry_double
        self.A.retry_level = retry_level
        self.A.retry_single = retry_single
        return self._key(basis='unread') if key else []

    def event(self, kind, *, frame=None, level=1, verdict='NONE', envelope=None, limited_pair=False,
              peer_keying=False, single=False, phase=None, phaseless=False,
              bw500_poll=False, inside=False, recognized=True, explicit=True,
              budget=True, session_budget=True, greeting=True, new_session=False):
        if self._pending_key is not None:
            raise ValueError('record the keying result before the next event')
        if kind == 'T':
            final = self.A is not None and self.A.kind == 'final'
            out = []
            if not final and self.mode == 'FILES':
                out = [self._report('abort', 'delivery aborted'), ResetDecoder()]
            self.W = self.T = self.K = None
            self._retry_double = False
            self._retry_level = None
            self._retry_single = False
            self.no_doubles = False
            self.before_data = False
            self.state = 'IDLE'
            self.files = FilesHold()
            if not final or new_session:
                self.A = None
            self.alt = 0
            self.trusted = bool(new_session or final and self.A.sent)
            if new_session:
                self.closed = False
            return out
        if self.closed:
            return []
        if self.state == 'IDLE' and kind == 'G':
            self.before_data = greeting
            self._answer(self.alt, 'nak')
            return []
        if self.state == 'IDLE' and kind in ('F', 'X'):
            self.before_data = False
            if verdict not in ('NONE', 'CERTAIN', 'POSSIBLE'):
                raise ValueError(verdict)
            envelope = 'inconclusive' if envelope is None else envelope
            if envelope not in ('single', 'double', 'inconclusive'):
                raise ValueError(envelope)
            retry_floor = self._retry_double and level == self._retry_level == 1
            if self._retry_double and level <= self._retry_level:
                verdict = ('NONE' if level < self._retry_level or
                           level == 1 and envelope == 'single' else 'POSSIBLE')
            learn_single = self._retry_single and level == self._retry_level and kind == 'F'
            self._retry_double = False
            self._retry_single = False
            if self.no_doubles:
                verdict = 'NONE'
            expected = {'NONE': frozenset((1,)), 'CERTAIN': frozenset((2,)),
                        'POSSIBLE': frozenset((1, 2))}[verdict]
            if kind == 'F':
                if frame is None:
                    raise ValueError('F needs a frame')
                if frame.ambiguous:
                    frame = replace(frame, last=verdict == 'NONE')
                if frame.compressed:
                    verdict, expected = 'NONE', frozenset((1,))
                elif not frame.full and not frame.last and not frame.ambiguous:
                    expected = frozenset((2,))
            self.W = Window(self.alt, level, verdict, expected,
                            (frame if kind == 'F' else None,), limited_pair,
                            envelope=envelope, retry_floor=retry_floor and not (
                                frame is not None and frame.compressed),
                            learn_single=learn_single)
            if kind == 'F' and frame.last and verdict == 'CERTAIN' and not frame.ambiguous:
                self.W.frames = (None, frame)
            self.state = 'OPEN'
            return []
        if self.state == 'OPEN' and kind == 'F':
            self._append(frame)
            return []
        if self.state == 'OPEN' and kind == 'X':
            self._append(None)
            return []
        if self.state == 'OPEN' and kind in ('E', 'Q', 'C'):
            w = self.W
            if envelope is not None:
                if envelope not in ('single', 'double', 'inconclusive'):
                    raise ValueError(envelope)
                w.envelope = envelope
                if w.retry_floor:
                    w.verdict = 'NONE' if envelope == 'single' else 'POSSIBLE'
                    w.expected = frozenset((1,)) if w.verdict == 'NONE' else frozenset((1, 2))
                    if any(f is not None and not f.full and not f.last and
                           not f.ambiguous for f in w.frames):
                        w.expected = frozenset((2,))
            w.single = single
            if (kind != 'C' and w.learn_single and len(w.frames) == 1
                    and w.frames[0] is not None and not peer_keying
                    and w.envelope == 'single' and 1 in w.expected):
                self.no_doubles = True
                # The lesson changes expectation, never placement.
            complete = (kind != 'C' and all(f is not None for f in w.frames)
                        and (len(w.frames) == 2 or 1 in w.expected and
                             (w.verdict == 'NONE' and not peer_keying or
                              single and w.envelope != 'double')))
            self.state = 'ENDED-COMPLETE' if complete else 'ENDED-INCOMPLETE'
            if complete and len(w.frames) == 1 and w.frames[0].ambiguous:
                w.frames = (replace(w.frames[0], last=True),)
        elif self.state == 'ENDED-COMPLETE' and kind == 'F' and self.W.held:
            self._append(frame)
            self.W.held = self.W.ready = False
            self.state = 'OPEN'
            return []
        elif self.state == 'ENDED-COMPLETE' and kind == 'X' and self.W.held:
            self._append(None)
            self.W.ready = False
            self.state = 'ENDED-INCOMPLETE'
        if self.state == 'ENDED-COMPLETE':
            if peer_keying and kind != 'C':
                self.W.held = True
                return []
            self.W.ready = True
            return [Ready()]
        if self.state == 'ENDED-INCOMPLETE':
            w = self.W
            if (w.limited_pair and w.frames[0] is not None
                    and (len(w.frames) == 1 or w.frames[1] is None)):
                w.partial = w.ready = True
                return [Ready(True)]
            self.W = None
            self.state = 'IDLE'
            return self._negative(w.phase, key=not peer_keying,
                                  retry_double=w.expected == frozenset((2,)) or
                                  len(w.frames) == 2 and None in w.frames or
                                  w.envelope == 'double',
                                  retry_level=w.level,
                                  retry_single=(kind != 'C' and w.verdict == 'CERTAIN'
                                      and len(w.frames) == 1 and w.frames[0] is not None
                                      and not peer_keying))
        if self.state != 'IDLE' or kind not in ('Q', 'R') or self.A is None:
            return []
        a = self.A
        if not session_budget:
            return self._close(self._report('budget', 'pre-greeting session give-up cap exhausted'))
        if a.kind == 'nak' and self.before_data and phase is None and recognized:
            return self._key(basis='debt', reanswer=True, budgeted=False)
        if not budget:
            return self._close(self._report('budget', 're-answer budget exhausted'))
        if a.kind == 'nak' and not a.sent:
            return self._key(basis='debt', reanswer=True)
        if a.kind == 'final':
            return self._key(basis='final', reanswer=True)
        silent = kind == 'R' and not recognized
        if silent:
            if a.kind == 'nak':
                return self._key(basis='debt', reanswer=True)
            return [] if inside else self._negative(self.alt, arm=True)
        proof = (self.trusted and phase in (0, 1) and not phaseless
                 and not (bw500_poll and a.phase == 1)
                 and not (a.phase == phase == 0) and not silent)
        names_a = proof and a.phase == phase
        if a.kind == 'frame10' and (names_a or inside):
            return self._key(basis='phase' if names_a else 'bound', reanswer=True)
        if proof:
            if names_a:
                return self._key(basis='phase', explicit=explicit, reanswer=True)
            return self._negative(self.alt)
        if not silent and a.kind in ('continue', 'frame10') and inside:
            return self._key(basis='bound', explicit=explicit, reanswer=True)
        if a.kind in ('continue', 'frame10') and not inside:
            return self._negative(self.alt, arm=True)
        if a.kind == 'nak':
            return self._key(basis='debt', reanswer=True)
        return []

    def _append(self, frame):
        if len(self.W.frames) >= 2:
            raise ValueError('a window has at most two frames')
        if self.W.frames[0] is not None and self.W.frames[0].ambiguous:
            self.W.frames = (replace(self.W.frames[0], last=False),)
        self.W.frames += (frame,)

    def _agrees(self, data, start):
        overlap = min(len(data), self.S - start)
        return data[:overlap] == self.R[start:start + overlap]

    def placement(self, prepared):
        """Preflight only; prepared entries correspond to read frames in W."""
        w = self.W
        if w is None or not w.ready:
            raise ValueError('no ended window ready for commit')
        frames = w.frames[:1] if w.partial else w.frames
        if len(prepared) != len(frames) or any(f is None for f in frames):
            raise ValueError('preparation does not cover the credited frames')
        data = b''
        for p in prepared:
            data += p.data
            if p.error:
                break
        start, replay = self.S, self.T
        if replay:
            if w.verdict == 'POSSIBLE' and len(frames) != 2:
                return Placement(start, data, b'', replay, 'POSSIBLE single inside replay')
            start = replay.start if replay.cursor is None else replay.cursor
            agrees = self._agrees(data, start)
            if replay.alternative is not None:
                other = self._agrees(data, replay.alternative)
                if agrees == other:
                    return Placement(start, data, b'', replay, 'nested candidates both agree' if agrees
                                     else 'nested candidates neither agrees')
                if other:
                    start = replay.alternative
                agrees = True
            if not agrees:
                if replay.cursor is not None:
                    return Placement(start, data, b'', replay, 'replay cursor disagrees')
                start, replay = self.S, None
            else:
                cursor = start + len(data)
                replay = replace(replay, cursor=cursor, alternative=None) if cursor < replay.frontier else None
        return Placement(start, data, data[max(0, self.S - start):], replay)

    def commit(self, prepared, *, files=None, step=0, original='original',
               answer_before=False):
        """Preflight every rejection before returning Key/Commit/Emit actions.

        TEXT Prepared.error carries its verified plaintext prefix. OFF/FILES
        preparation errors are malformed frames: use X, never this interface.
        FilesParse is mandatory in FILES, even when no new bytes are placed.
        """
        if self._pending_key is not None:
            raise ValueError('keying result is pending')
        p = self.placement(prepared)
        w = self.W
        if p.error:
            return self._close(self._report('R8', p.error, rejected=p.data))
        fault = next((v.error for v in prepared if v.error), None)
        if fault and self.mode != 'TEXT':
            raise ValueError('only prepared TEXT may carry a fault prefix')
        frames = w.frames[:1] if w.partial else w.frames
        terminal = not w.partial and frames[-1].last
        next_files, emitted = self.files, p.new
        if self.mode == 'FILES':
            if files is None:
                raise ValueError('FILES needs a parser preflight')
            next_files, emitted, report = self.files.preview(files, terminal)
            if report:
                return self._close(replace(report, start=self.K.start if self.K else self.S,
                                           frontier=self.S, rejected=p.new))
        replay_advanced = False
        if self.T is not None:
            previous = (self.T.cursor if self.T.cursor is not None else
                        self.T.alternative if self.T.alternative is not None else self.T.start)
            replay_advanced = p.start + len(p.data) > previous
        self.R += p.new
        self.T = None if terminal else p.replay
        self.files = next_files
        self.K = Anchor(p.start, p.start + len(p.data), w.level, frames,
                        w.verdict == 'POSSIBLE' and len(frames) == 1)
        self.W = None
        self.state = 'IDLE'
        progressed = bool(p.new or replay_advanced)
        actions = [Commit(p.start, p.start + len(p.data), p.new, progressed)]
        if emitted:
            actions.append(Emit(emitted))
        if fault:
            return actions + self._close(self._report('fault', fault, rejected=p.data))
        self._answer(w.phase, 'frame10' if w.partial else 'final' if terminal else 'continue',
                     step=step, original=original)
        keys = self._key(basis='head' if w.partial else 'complete', explicit=False)
        return keys + actions if answer_before else actions + keys

    def sent(self, key, reached_air):
        """Apply a transport result, never the peer's decode result."""
        if key != self._pending_key:
            raise ValueError('key does not match the pending decision')
        self._pending_key = None
        if not reached_air:
            return []
        a = self.A
        if key.reanswer:
            a.reanswers += 1
        if a.sent:
            return []
        a.sent = True
        self.alt ^= 1
        if a.kind in ('nak', 'frame10'):
            self.trusted = False
        if a.kind == 'nak':
            self._retry_double = a.retry_double
            self._retry_level = a.retry_level
            self._retry_single = a.retry_single
        if a.kind == 'nak' and a.effect is not None:
            start, other = a.effect
            self.T = Replay(start, self.S, alternative=other if other < self.S else None)
        if a.kind == 'final':
            self.alt, self.trusted, self.T = 0, True, None
        return []
