# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""SITOR mode B, collective only: the time-diversity FEC of ITU-R M.625 §4.

Even slots are DX and odd slots are RX; each word is sent again in the RX
stream five slots later. These are the audio tones for USB with the dial
1.7 kHz below the assigned frequency.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from hfmodem.lanner import ccir476, fsk, ita2

BAUD = 100.0
F_B = 1785.0
F_Y = 1615.0
FS = fsk.FS
_IDLE_BETA = ccir476.word([c == "B" for c in ccir476.SERVICE_SIGNALS["idle_beta"]])


def words(text, *, phasing_pairs=16, variant="us") -> list[int]:
    dx = [ccir476.PHASING_2] * phasing_pairs
    dx += [ccir476.from_ita2(c) for c in ita2.encode(text, variant)]
    dx += [ccir476.PHASING_1] * 3      # §4.6.7: at least three idle alphas in DX
    slots = []
    for k, w in enumerate(dx):
        slots.append(w)
        delayed = dx[k - 2] if k >= 2 else ccir476.PHASING_1
        slots.append(ccir476.PHASING_1 if delayed == ccir476.PHASING_2 else delayed)
    return slots


def modulate(text, *, phasing_pairs=16, fs=FS, variant="us") -> np.ndarray:
    return _tx(words(text, phasing_pairs=phasing_pairs, variant=variant), fs=fs)


# Word-grid changes require a 0.3 gain in the valid-word fraction. Soft
# combining requires the best candidate's summed score to exceed the runner-up by 0.3.
_SYNC_MARGIN = 0.3
_SOFT_MARGIN = 0.3
_SYNC_EVERY = 20        # slots between word-sync checks
_LOCK_WINDOW = 40       # slots the loss-of-lock fraction covers


def demodulate(audio, *, fs=FS, variant="us") -> str:
    return "\n".join(t for _, t in transmissions(audio, fs=fs, variant=variant))


class Resume:
    """Optional transmission cursor, in absolute input samples at ``fs``.

    Keep lookback before ``sample`` when supplying the next audio window.
    ``phasing_samples`` and ``char_samples`` give fitted absolute sample
    positions for each lock and each decoded character, parallel to ``texts``.
    ``char_shift_known`` records each character's case certainty. An unresolved
    diversity pair revokes certainty until a recovered LTRS or FIGS.
    ``max_fade_s`` retains lock through bounded runs of erased symbols without
    treating their missing words as evidence of a different word grid.
    Text snapshots include completed locks and the current lock, so callers
    can replace their previous snapshot without joining partial words twice.
    """

    def __init__(self, *, max_fade_s=0.0):
        self.max_fade_words = round(max_fade_s * BAUD / 7)
        self.fade_words = 0
        self.fade_start = None
        self.sample = 0.0
        self.symbol = 0
        self.origin = None
        self.locked = False
        self.texts = []
        self.phasing_samples = []
        self.char_samples = []
        self.char_shift_known = []
        self.shift_known = False
        self.power = 0.0
        self.decoder = None
        self.recent = deque(maxlen=_LOCK_WINDOW)
        self.idle = self.since = 0
        self._ends = []
        self._end_reasons = []


    @property
    def emission_end(self):
        """Absolute sample of the latest termination or lock loss, else None.

        The decoder's recovered word grid owns this boundary. It remains
        available after scanning resumes in search of the next phasing run.
        """
        return self._ends[-1] if self._ends else None

    @property
    def emission_finished(self):
        """Whether the latest released lock ended with idle alphas."""
        return bool(self._end_reasons and self._end_reasons[-1] == "idle")

    def lock_finished(self, index):
        """Whether snapshot lock ``index`` ended with idle alphas."""
        return (0 <= index < len(self._end_reasons)
                and self._end_reasons[index] == "idle")


def transmissions(audio, *, fs=FS, variant="us", resume=None, origin=0,
                  final=True, offset_hz=0.0, sample_positions=False) -> list[tuple[int | float, str]]:
    """Return (phasing position, decoded text) for each transmission.

    With a :class:`Resume`, ``origin`` locates this overlapping audio window
    in absolute input samples. Include acquisition lookback before its cursor;
    ``final=False`` leaves three seconds for FEC and word-sync lookahead. The
    returned snapshot includes previous locks (their relative indices can be
    negative). ``offset_hz`` tunes both tones without another audio mixer.
    ``sample_positions=True`` returns fitted absolute phasing sample positions
    instead of nominal decision indices. Without a cursor, decode the entire
    supplied audio. At end of input, complete DX words without a received
    RX copy are decoded only when they are valid traffic words.
    """
    if resume is None:
        resume = Resume()

    def snapshot():
        positions = (resume.phasing_samples if sample_positions else
                     [round((sample - origin) * BAUD / fs) for sample, _ in resume.texts])
        return [(position, text) for position, (_, text) in zip(positions, resume.texts) if text]

    soft = fsk.discriminate(audio, f_hi=F_B + offset_hz, f_lo=F_Y + offset_hz,
                            baud=BAUD, fs=fs)
    power = None
    if resume.max_fade_words and len(soft):
        # Four-unit Hann windows separate a neighbour's nearest tone even
        # when it is only 70 Hz away. Broadband power would mask the fade.
        power = fsk.tone_energy(audio, f_hi=F_B + offset_hz,
                                f_lo=F_Y + offset_hz, baud=BAUD, fs=fs, units=4)
    centres = fsk.track(soft, baud=BAUD, fs=fs)
    decisions = soft[np.rint(centres).astype(np.int64)]
    hard = decisions > 0
    if not centres.size:
        return snapshot()
    # Nonfinal windows retain real audio for FEC and word-sync lookahead.
    stop = len(hard) if final else max(0, len(hard) - round(3 * BAUD))

    def faded():
        local_soft = np.where(power >= resume.power * .01, soft, 0)
        local_centres = fsk.track(local_soft, baud=BAUD, fs=fs)
        local_decisions = local_soft[np.rint(local_centres).astype(np.int64)]
        return local_centres, local_decisions, local_decisions > 0

    def nearest(track, sample):
        return int(np.argmin(abs(track - sample)))

    # The faded track and the full one can count a different number of symbols
    # across a fade, since a normalised discriminator reads full-scale noise
    # off near silence. Symbol indices therefore never cross between them; a
    # position moves from one to the other by its sample.
    local = faded() if power is not None and resume.locked else (centres, decisions, hard)
    pos = nearest(local[0], resume.sample - origin)
    if resume.origin is None:
        resume.origin = origin
        resume.symbol = pos
    symbol_base = resume.symbol - pos
    while pos < stop:
        if resume.locked:
            off, j0 = pos, 0
        else:
            found = find_phasing(hard[:stop], pos)
            if found is None:
                pos = max(pos, stop - 56)
                break
            off, j0, index = found
            resume.texts.append([resume.origin + (symbol_base + index) * fs / BAUD, ""])
            resume.phasing_samples.append(float(origin + centres[index]))
            resume.char_samples.append([])
            resume.char_shift_known.append([])
            resume.shift_known = False
            if power is not None:
                # Phasing belongs to this emission, even when lookback contains
                # a much louder predecessor. Keep its reference through fades.
                lo = round(centres[index])
                hi = min(len(power), lo + round(.56 * fs))
                resume.power = float(np.max(power[lo:hi]))
            resume.decoder = ita2.Decoder(variant, unshift_on_space=False)
            resume.recent.clear()
            resume.idle = resume.since = resume.fade_words = 0
            resume.fade_start = None
            resume.locked = True
            if power is not None:
                local = faded()
                off += nearest(local[0], centres[index]) - index
        local_centres, local_decisions, local_hard = local
        text, end = _decode(local_decisions, local_hard, off, j0,
                            resume=resume, stop=min(stop, len(local_centres)), centres=local_centres, origin=origin,
                            final=final)
        if not resume.locked:
            # Include the remaining idle-alpha tail prescribed by M.625.
            tail = 28 * fs / BAUD if resume.idle >= 2 else 0
            boundary = float(origin + local_centres[min(end, len(local_centres) - 1)] + tail)
            if resume.idle < 2 and resume.fade_start is not None:
                boundary = resume.fade_start
            resume._ends.append(boundary)
            resume._end_reasons.append("idle" if resume.idle >= 2 else "loss")
        resume.texts[-1][1] += text
        pos = end
        if not resume.locked and local_centres is not centres and end < len(local_centres):
            pos = nearest(centres, local_centres[end])
            local = centres, decisions, hard
        if resume.locked:
            centres = local_centres
            break
    if pos < len(centres):
        resume.sample = origin + centres[pos]
    else:
        resume.sample = origin + len(audio)
    resume.symbol = symbol_base + pos
    return snapshot()


def traffic_signature(soft, *, fs=FS, centres=None) -> bool:
    """Evidence against asynchronous RTTY, without asserting SITOR DX parity.

    Mid-stream traffic can reveal a 100 Bd transition grid or a sustained
    seven-unit constant-ratio word grid. Neither establishes the diversity
    phase required to acquire text; acquisition still waits for phasing.
    ``centres`` can supply an already recovered 100 Bd symbol grid.
    """
    soft = np.asarray(soft)
    if len(soft) < .7 * fs:
        return False
    edges = np.flatnonzero(np.diff(soft > 0))
    # Ignore incomplete discriminator windows at the buffer edges.
    unit = fs / BAUD
    edges = edges[(edges >= unit) & (edges < len(soft) - unit)]
    if len(edges) >= 16:
        coherence = abs(np.mean(np.exp(2j * np.pi * edges / unit)))
        if coherence >= .85:
            return True
    if centres is None:
        centres = fsk.track(soft, baud=BAUD, fs=fs)
    decisions = soft[np.rint(centres).astype(int)]
    for offset in range(7):
        size = (len(decisions) - offset) // 7 * 7
        if size < 12 * 7:
            continue
        rows = decisions[offset:offset + size].reshape(-1, 7)
        valid = (np.min(np.abs(rows), axis=1) >= .5) & ((rows > 0).sum(axis=1) == 4)
        if np.mean(valid) >= .85:
            return True
    return False


def word_activity(decisions) -> bool:
    """Constant-ratio traffic evidence for an already acquired SITOR lane.

    Search word offsets on the recovered symbol grid; ignore erased words.
    Acquisition still requires phasing, since traffic cannot establish DX parity.
    """
    from scipy.special import bdtrc

    for offset in range(7):
        size = (len(decisions) - offset) // 7 * 7
        if size <= 0:
            continue
        rows = decisions[offset:offset + size].reshape(-1, 7)
        rows = rows[np.max(np.abs(rows), axis=1) >= .1]
        valid = int(np.count_nonzero((rows > 0).sum(axis=1) == 4))
        # For independent equiprobable units, 35 of the 128 possible words
        # contain exactly four B units.
        if len(rows) >= 4 and valid / len(rows) >= .6 and bdtrc(valid - 1, len(rows), 35 / 128) < .05:
            return True
    return False


def find_phasing(hard: np.ndarray, pos: int) -> tuple[int, int, int] | None:
    """Earliest qualifying phasing run at or past `pos`: (grid offset, first
    DX row, symbol index). At least four consecutive phasing-2/phasing-1
    pairs at one of seven word-grid offsets and either DX parity. ``hard``
    contains symbol decisions (True is B); return None if no run qualifies."""
    ph1, ph2 = ccir476.PHASING_1, ccir476.PHASING_2
    pw = 1 << np.arange(7)
    # Overlap enough rows to retain a run straddling a chunk boundary.
    # Seven offsets with both DX parities cover every alignment from pos.
    chunk_rows = 256
    while hard.size - pos >= 8 * 7:
        best = None
        for o in range(pos, pos + 7):
            nrows = min(chunk_rows, (hard.size - o) // 7)
            if nrows < 8:
                continue
            wnum = ((~hard[o:o + 7 * nrows].reshape(nrows, 7)) * pw).sum(1)
            for parity in (0, 1):
                run = start = 0
                for j in range(parity, nrows - 1, 2):
                    if wnum[j] == ph2 and wnum[j + 1] == ph1:
                        if run == 0:
                            start = j
                        run += 1
                        if run >= 4:
                            index = o + 7 * start
                            if best is None or index < best[2]:
                                best = (o, start, index)
                            break
                    else:
                        run = 0
        if best is not None:
            return best
        pos += 7 * (chunk_rows - 8)
    return None


def _decode(decisions: np.ndarray, hard: np.ndarray, off: int, j0: int,
            *, resume: Resume, stop: int, centres: np.ndarray, origin: int,
            final: bool) -> tuple[str, int]:
    """One transmission, from its phasing run to end of emission or lock loss;
    returns its text and the symbol index where scanning should resume."""
    ph1, ph2 = ccir476.PHASING_1, ccir476.PHASING_2
    decoder = resume.decoder
    out: list[str] = []
    recent = resume.recent
    idle, since = resume.idle, resume.since
    j = j0
    while off + 7 * (j + 1) <= hard.size and off + 7 * j < stop:
        has_rx = off + 7 * (j + 6) <= hard.size
        if not has_rx and not final:
            break
        dx_start = off + 7 * j
        erased = np.max(np.abs(decisions[dx_start:dx_start + 7])) < .1
        if resume.max_fade_words and erased and not resume.fade_words:
            resume.fade_start = float(origin + centres[dx_start])
        resume.fade_words = resume.fade_words + 2 if erased else 0
        if resume.max_fade_words and resume.fade_words > resume.max_fade_words:
            resume.locked = False
            break
        fading = erased and resume.fade_words <= resume.max_fade_words
        dx = _wordint(hard, off, j)
        rx = _wordint(hard, off, j + 5) if has_rx else None
        if dx == ph1:
            idle += 1
            if idle >= 2:
                resume.locked = False
                break
        else:
            idle = 0
            if dx != ph2 or (has_rx and rx != ph1):
                since += 2
                recent.append(ccir476.valid(dx))
                if has_rx:
                    recent.append(ccir476.valid(rx))
                    value = _combine(decisions, off, j, dx, rx, decoder, resume=resume)
                else:
                    code = ccir476.to_ita2(dx) if ccir476.valid(dx) else None
                    if code is None:
                        resume.shift_known = False
                    elif code in (0b11111, 0b11011):
                        resume.shift_known = True
                    value = decoder.feed(code) if code is not None else ""
                out.append(value)
                if value:
                    resume.char_shift_known[-1].extend([resume.shift_known] * len(value))
                    resume.char_samples[-1].extend(
                        [float(origin + centres[off + 7 * j])] * len(value))
            recovering = resume.fade_start is not None and not fading
            valid_pair = ccir476.valid(dx) or (has_rx and ccir476.valid(rx))
            if (recovering and len(recent) == _LOCK_WINDOW
                    and sum(recent) < .4 * _LOCK_WINDOW and not valid_pair):
                resume.locked = False
                break
            if since >= _SYNC_EVERY and not fading:
                since = 0
                off = _realign(hard, off, j)
                recovered = recovering and valid_pair
                if len(recent) == _LOCK_WINDOW and sum(recent) < 0.4 * _LOCK_WINDOW and not recovered:
                    resume.locked = False
                    break
        if not erased and sum(recent) >= .4 * _LOCK_WINDOW:
            resume.fade_start = None
        j += 2
    resume.idle, resume.since = idle, since
    return "".join(out), off + 7 * j


def _wordint(hard: np.ndarray, off: int, j: int) -> int:
    return sum((0 if b else 1) << i for i, b in enumerate(hard[off + 7 * j:off + 7 * j + 7]))


def _combine(decisions, off, j, dx, rx, decoder, *, resume=None) -> str:
    def unresolved():
        if resume is not None:
            resume.shift_known = False
        return "_"

    def recovered(code):
        if resume is not None and code in (0b11111, 0b11011):
            resume.shift_known = True
        return decoder.feed(code)

    if ccir476.valid(dx) and ccir476.valid(rx) and dx != rx:
        return unresolved()
    if dx == rx == _IDLE_BETA:
        return ""
    codes = sorted({ccir476.to_ita2(w) for w in (dx, rx)
                    if ccir476.valid(w) and ccir476.to_ita2(w) is not None})
    if len(codes) == 1:
        return recovered(codes[0])
    if ccir476.valid(dx) or ccir476.valid(rx):
        return unresolved()
    v = (decisions[off + 7 * j:off + 7 * j + 7]
         + decisions[off + 7 * (j + 5):off + 7 * (j + 5) + 7])
    words_ = ccir476.TRAFFIC_WORDS
    scores = sorted((float(v @ np.where(np.array(ccir476.units(w), dtype=bool), 1.0, -1.0)),
                    ccir476.to_ita2(w)) for w in words_)
    if scores[-1][0] - scores[-2][0] >= _SOFT_MARGIN:
        return recovered(scores[-1][1])
    return unresolved()


def _realign(hard: np.ndarray, off: int, j: int) -> int:
    """Shift the word grid one unit toward whichever neighbour has clearly
    more constant-ratio words in the next 40 slots, if a neighbour does."""
    def frac(o: int) -> float:
        start = o + 7 * j
        n = min(_LOCK_WINDOW, (hard.size - start) // 7)
        if start < 0 or n <= 0:
            return 0.0
        return (hard[start:start + 7 * n].reshape(n, 7).sum(1) == 4).mean()
    f = {o: frac(o) for o in (off - 1, off, off + 1) if o + 7 * j >= 0}
    best = max(f, key=f.get)
    return best if f[best] - f[off] >= _SYNC_MARGIN else off


def _tx(words: list[int], *, fs: int = FS) -> np.ndarray:
    units = [u for w in words for u in ccir476.units(w)]
    return fsk.modulate(units, baud=BAUD, f_hi=F_B, f_lo=F_Y, fs=fs)
