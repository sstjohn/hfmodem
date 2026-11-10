# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The PACTOR-4 control words this station has heard, as despread symbols.

EVERY CONTROL-LIKE EMISSION ON FILE IS A SEQUENCE OF SF-16 SPREAD SYMBOLS, and
none of them is a Chu sequence. Locking the chip grid on the published
`scsSpread16` despreading coherence and reading the symbols out turns a burst
into a short word, and the words repeat: four distinct ones across four years of
recordings, two gateways and three tapes, where a Chu19 fit over the same
material sits at 0.64 against a matched null of 0.637 mean and 0.687 max. So the
Chu root `p4sig.header_score` answers is not what names these bursts, and this
is: a bank of measured words, matched with a free complex scale.

Three of the four are resolved well enough to be literals here.

BREAKIN, 27 symbols. Two independent instances -- the w18 recording's break-in
header at despreading coherence 0.975, and PACTORIV_audio2's first burst --
correlate at 0.988. This is the word the two agree on.

CALLER20, 20 symbols. The header of the robust short packet PACTORIVaudio1's
CALLER keys three times, at 4.9, 6.1 and 7.4 s. The three agree at 0.994-0.997,
which is what a header that carries no counter looks like.

GATEWAY, 18 symbols. What WS8EOC answered with on 2026-08-28 and KB5LZK on
2026-08-29, each stacked coherently over its own session's cycles and the two
stacked together: 0.979 between the stations. Tail-anchored, because the
receiver's mute release censors the head of every one of them -- the length is
the clean tail and not the word's own length.

X27 AND Y27, 27 symbols, are what a gateway answers robust short packets
with, in its own slot 0.935 s after the caller's boundary: X27 after the first
short, Y27 after every later one, and once the caller stops keying each Y27
carries a normal-mode break-in body. VE3KPG answered our shorts with them on
2026-09-26 and N9VC answered its caller with them on the 09-25 Kiwi recording.
Measured as phase steps, the way `p4robusttx.HEADER_STEPS` is: each instance
locked on a render of the running mean and the steps read coherently symbol by
symbol, 8 instances of X27 and 20 of Y27. The two stations' means agree within
13 degrees a step on X27 and 10 on Y27, and every step's concentration is
0.99-1.00. GATEWAY is V27's last eighteen symbols, at 0.995: the August
answers were this word with its head censored.

THE FOURTH IS NOT HERE. PACTORIVaudio1's three answering bursts despread at
coherence 0.32-0.37 and do not resolve to a stable word at that signal level.
They are a fourth word by length and place, and nothing more can be said.

WHY THE FREQUENCY SEARCH IS WIDE. The chip grid is locked on despreading
coherence, and that statistic is blind to a carrier offset: a 16-chip block
spans 8.9 ms, so tens of hertz cost it nothing and the offset it picks is
arbitrary within that. The residual lands whole in the symbol phases -- 20 Hz
across a 27-symbol word is four and a half turns -- so it is taken out here
instead, over +/-20 Hz. Read that way PACTORIVaudio1's three caller headers all
name CALLER20 at 0.998-0.999, where +/-1 Hz names only two of the three.

A WIDE SEARCH NEEDS ITS EDGES WATCHED, and that is `interior`. The fit that
named PACTOR-IV_112-5_DQPSK.WAV's third robust-short CALLER20 at 0.96 sat on
BOTH boundaries, shift -2 and +20.0 Hz: walking its onset 10 ms swung it
0.48-0.96 and narrowing the tuning to 5 Hz collapsed that file's four bursts to
0.34-0.57. None of them is a word we hold -- the four are two headers
alternating, {0,2} against {1,3}. A maximum on the edge is the search running
out of room, so no word is named on one. The three genuine caller headers sit
at shift 0 and 3-7 Hz, nowhere near an edge. A header read at its true carrier
from its first chip sits two symbols in: the N9VC link on the 08-02 K5DAT-13
recording names CALLER20 at 0.98-0.99 on its three header-A shorts at shift
-1 and -2, and a clean header-A render at shift -2, so the slide reaches three.

Two nulls bound the knee, both taken as `match` names: the best INTERIOR fit
of the three words. Over 1200 random symbol words it runs 0.51 mean, 0.68 at
the 99th and 0.79 at its maximum; end to end -- through the despreading lock
that argmaxes sixteen phases, fifty-one offsets and sixteen chip starts before
this sees anything -- over 80 synthetic spread words and noise windows it runs
0.56 mean and 0.71 at its maximum. A genuine word lands at 0.998-1.000 and
audio1's unresolved answering bursts at 0.51-0.56. Pairwise the three in the
bank sit at 0.42-0.60 under this search; they are different words, and the wide
frequency freedom lifts every floor together.

Provenance: the symbol recovery is `working/`'s header-taxonomy pass --
despreading-coherence grid lock, tail-anchored stacking, and the pairwise
matrix that separated the four. Quantised int16 re/im over 32767, as
`p4sig.SPREAD16` and `p4shorttx.HEADER` are.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

#: The symbol rate a spread symbol arrives at: 1800 chips/s over 16 chips.
SYMBOL_RATE = 112.5

#: A match. Above the maximum of BOTH nulls in the header -- 0.79 on random
#: symbol words and 0.66 end to end through the despreading lock -- and far
#: below the 0.998-1.000 a word measured twice lands at.
MATCH_KNEE = 0.88

#: How far the bank is slid, and the residual frequency it is slid under.
MATCH_SHIFT = 3
MATCH_HZ = 20.0
MATCH_STEP_HZ = 0.5

_BREAKIN_IQ = (
      8820,  31558,  31245,   9869,   3855,  32539, -32579,  -3507,
     24611, -21633,  14077,  29589, -26244,  19620, -32414,   4795,
    -24857,  21350, -24446,  21820, -17516,  27693, -10170,  31149,
    -26245,  19619, -21055,  25107, -32755,   -893, -27775, -17385,
    -17867,  27467,  30112, -12921,  13527, -29845,  -3643, -32564,
    -23488,  22847, -30555,  11834, -32513,   4070, -11115,  30824,
    -24645,  21594, -22101,  24192,  14787,  29240,
)
_CALLER20_IQ = (
     -8357,  31683,  16049,  28568,  27210, -18256, -13308, -29943,
      2006, -32706, -32098,   6587,  32676,  -2444, -26277, -19576,
    -32405,  -4857,  10436,  31061,  -5445, -32311, -23291,  23048,
    -28622,  15953,   5263,  32341,  31187, -10052,  18647, -26944,
    -11557,  30661, -25735,  20283, -32525,  -3978, -14480, -29394,
)
_GATEWAY_IQ = (
    -32749,   1082,  25692, -20338,  21514,  24715, -21791,  24471,
    -32765,    398, -31583,  -8727,  32651,  -2749,  32145,  -6356,
       -17, -32767,  32404,  -4862,   9203,  31448, -18836,  26812,
      7908,  31798, -25513,  20562,  16987,  28020, -13726, -29753,
     11448, -30702,  16437,  28346,
)

#: The words a gateway answers robust shorts with, as phase steps in degrees.
STEPS = {
    # VE3KPG 0926 (23 Z27s), N9VC sessions 0926 (V/U/T); design §4.1.
    'Z27': (-169, -171, -69, 174, 51, -178, -116, 72, -133, -79, 54, -169,
            -70, -121, -52, 92, 11, 98, -6, -57, -21, 138, 53, -40, -101, 167),
    'V27': (-61, -16, 156, -139, -46, -23, 145, -29, -142, 15, -83, 170,
            -44, 146, 169, 90, 178, -77, 37, -174, 59, -91, -171, 145, -118, -11),
    'U27': (139, -162, 86, -12, 26, -93, -14, 136, 125, 99, -166, 76,
            -179, -160, 168, -113, -26, -26, 173, 4, -178, -61, 133, -28, -177, 41),
    'T27': (-16, 141, -35, -29, -83, -6, 157, -86, -42, -57, -85, -44, 131,
            -93, -70, -108, 41, -133, -74, 56, 139, 79, -30, 92, 112, 135),
    'X27': (-73, -67, 70, 176, -150, 105, -141, 175, 39, 135, 78, 70, 29,
            2, 147, -25, -96, 61, 79, 34, -66, 36, -87, 168, 32, 107),
    'Y27': (59, 15, -178, 77, 131, 17, 32, -111, 98, -132, 87, -22, -1,
            82, -25, -155, -147, 50, -123, -135, 168, -145, 9, 32, -50, -96),
    # W27: WS8EOC 0930 1509 witness, six copies answering our first data body
    # (captures/onair-0930-1510/witness.wav 43.38-49.64 s), pairwise r 1.00.
    'W27': (146, 72, -71, 37, 156, 180, 32, 60, -51, 142, -58, -102, 179, 25,
            64, 34, 129, -146, -162, -172, -36, 22, -146, -103, 53, -50),
}


#: Measured word carrier minus body carrier; shared with the stock renderer.
CARRIER_HZ = {'X27': 3.65, 'V27': -1.45, 'Y27': 1.40, 'U27': -3.30, 'T27': 3.80}


def _word(raw: tuple[int, ...]) -> np.ndarray:
    values = np.array(raw, np.float64) / 32767.0
    word = values[0::2] + 1j * values[1::2]
    return (word / np.abs(word)).astype(np.complex128)


def phasors(degrees) -> np.ndarray:
    """Differential phase steps as absolute symbols, from a phase-0 start."""
    return np.exp(1j * np.radians(np.concatenate([[0], np.cumsum(degrees)])))


#: The bank, longest first so a tie goes to the word with more evidence in it.
BANK = {'BREAKIN': _word(_BREAKIN_IQ),
        **{name: phasors(steps) for name, steps in STEPS.items()},
        'CALLER20': _word(_CALLER20_IQ), 'GATEWAY': _word(_GATEWAY_IQ)}


@lru_cache(maxsize=64)
def _carrier_grid(n):
    hz = np.arange(-MATCH_HZ, MATCH_HZ + 1e-9, MATCH_STEP_HZ)
    return hz, np.exp(2j * np.pi * np.arange(n)[:, None] / SYMBOL_RATE * hz)


def correlate(a: np.ndarray, b: np.ndarray) -> tuple[float, int, float]:
    """(|corr|, shift, Hz) of the best fit: tail-anchored, free scale, tuned.

    Free complex scale because neither the carrier phase nor the gain is ours
    to know; slid because the segmenter's onset is a 5 ms grid and a symbol is
    8.9 ms; tuned widely because the grid lock ahead of it cannot see a carrier
    offset at all (see the module header).

    The shift and the hertz come back because WHERE THE FIT SITS IS PART OF
    WHETHER IT IS ONE. A maximum on the edge of the search says the real
    optimum is outside it, which is a fit that has run out of room rather than
    one that has found something -- `match` will not name a word on it.
    """
    best = (0.0, 0, 0.0)
    for shift in range(-MATCH_SHIFT, MATCH_SHIFT + 1):
        x, y = a[max(0, shift):], b[max(0, -shift):]
        n = min(x.size, y.size)
        if n < min(a.size, b.size) - MATCH_SHIFT:
            continue
        x, y = x[:n], y[:n]
        scale = np.linalg.norm(x) * np.linalg.norm(y)
        hz, grid = _carrier_grid(n)
        scores = abs((x.conj()*y) @ grid) / max(scale, 1e-30)
        k = int(scores.argmax())
        if scores[k] > best[0]:
            best = (float(scores[k]), shift, float(hz[k]))
    return best


def interior(shift: int, hz: float) -> bool:
    """Whether a fit found its own maximum rather than the search's edge."""
    return abs(shift) < MATCH_SHIFT and abs(hz) < MATCH_HZ - MATCH_STEP_HZ / 2


def match(symbols: np.ndarray) -> tuple[str | None, float]:
    """(the bank word this is, or None, and the score it was named on)."""
    heard = np.asarray(symbols, complex)
    if heard.size < 12:
        return None, 0.0
    heard = heard / np.maximum(np.abs(heard), 1e-30)
    best, score, placed = None, 0.0, False
    for name, word in BANK.items():
        r, shift, hz = correlate(word, heard[:word.size + MATCH_SHIFT])
        if r > score:
            best, score, placed = name, r, interior(shift, hz)
    return (best if placed and score >= MATCH_KNEE else None), score


def match_placed(
    symbols,
    *,
    cfo_hz: float,
    hz_span: float = 3.0,
    step: float = 0.5,
    shifts=(-1, 0, 1),
    carrier_offsets=None,
):
    """Named 27-symbol word at a known body carrier: bounded residual refinement.

    Input symbols are already tuned by the despreader at cfo_hz. The parameter
    records that contract; each word searches its offset +/-hz_span. A supplied
    carrier_offsets map retains the reference of unmeasured words.
    """
    if not np.isfinite(cfo_hz) or hz_span <= 0 or step <= 0:
        raise ValueError("finite carrier and positive search dimensions required")
    offsets = CARRIER_HZ if carrier_offsets is None else carrier_offsets
    heard = np.asarray(symbols, complex)
    if heard.size < 12:
        return None, 0.0, 0
    heard = heard / np.maximum(abs(heard), 1e-30)
    carriers = np.arange(-hz_span, hz_span + step / 2, step)
    best = (None, 0.0, 0, 0.0)
    for name, word in BANK.items():
        if len(word) != 27:
            continue
        for shift in shifts:
            x, y = word[max(0, shift) :], heard[max(0, -shift) :]
            n = min(len(x), len(y))
            if n < 12:
                continue
            turns = np.arange(n) / SYMBOL_RATE
            scores = (
                abs(
                    (x[:n].conj() * y[:n])
                    @ np.exp(-2j * np.pi * turns[:, None] * (carriers + offsets.get(name, 0.)))
                )
                / n
            )
            k = int(scores.argmax())
            if scores[k] > best[1]:
                best = name, float(scores[k]), shift, float(carriers[k])
    name, score, shift, hz = best
    return (
        name if score >= MATCH_KNEE and abs(hz) < hz_span - step / 2 else None,
        score,
        shift,
    )


@lru_cache(maxsize=64)
def _placed_templates(count, acquisition=False, carrier_offsets=None):
    offsets = CARRIER_HZ if carrier_offsets is None else dict(carrier_offsets)
    entries, templates = [], []
    span = MATCH_HZ if acquisition else 3.0
    carriers = np.arange(-span, span + MATCH_STEP_HZ/2, MATCH_STEP_HZ)
    for name, word in BANK.items():
        if len(word) != 27:
            continue
        for shift in (-1, 0, 1):
            lo = max(0, -shift)
            reference = word[max(0, shift):]
            n = min(len(reference), count - lo)
            for hz in carriers:
                frequency = hz + (0. if acquisition else offsets.get(name, 0.))
                row = np.zeros(count, complex)
                row[lo:lo+n] = reference[:n].conj() * np.exp(
                    -2j * np.pi * np.arange(n) / SYMBOL_RATE * frequency) / n
                templates.append(row)
                entries.append((name, shift, hz))
    return entries, np.array(templates)


def placed_scores(symbols, *, acquisition=False, carrier_offsets=None):
    """Rank words at body CFO + word offset; acquisition uses absolute residual Hz."""
    heard = np.asarray(symbols, complex)
    offsets = None if carrier_offsets is None else tuple(sorted(carrier_offsets.items()))
    entries, templates = _placed_templates(heard.shape[1], acquisition, offsets)
    heard = heard / np.maximum(abs(heard), 1e-30)
    scores = abs(np.einsum('ij,kj->ik', heard, templates, optimize=False))
    best = scores.argmax(axis=1)
    return [entries[i] for i in best], scores[np.arange(len(best)), best]
