# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station's own callsign, as Morse audio for the transmit path.

One caller: the identification interlock, which owes a callsign at the cadence
the station's regulatory profile names, and owes it in a form that reads as a
callsign to an operator listening and to every decoder ever written — unlike a
callsign inside a data frame, which under §97.309(b) is an unspecified digital
code and reads as nothing to anyone not running that modem.

The rig's arm gate is deliberately not a caller. An earlier version keyed the
callsign in Morse on the RTS line during the PTT proof and claimed that
discharged §97.119; it did not, because keying a sideband transmitter with no
audio produces no RF, so nothing identifiable went out. The proof now says what
it is — a keying test that identifies nothing (`Rig._prove_ptt`).

Timing is PARIS: a dit is 1.2 s / WPM, a dah is three dits, one dit separates the
elements of a character, three separate characters, seven separate words. The tone
is keyed with a raised-cosine rise and fall, because a hard-edged one splashes
clicks either side of the frequency, which is the last thing an identification
should do.

besra's `send_id()` is not a caller and must not become one: it transmits an
ARDOP IDFrame, frame type 0x30, which is a digital frame carrying the callsign as
data. Different obligation, different mechanism, and wiring this in beside it
would put two identifications on the air for one.
"""
from __future__ import annotations

import numpy as np

# ITU-R M.1677-1. A callsign is letters, digits and '/' for a portable indicator;
# nothing else belongs in an identification, and an unknown character must be
# refused rather than guessed at — a mis-keyed callsign identifies someone else.
MORSE = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    "/": "-..-.",
}

#: §97.119(b)(1): an identification sent by an automatic device may not exceed
#: 20 WPM. Refused rather than clamped — a caller asking for 25 has a wrong idea
#: of what it is allowed to send, and quietly sending 20 leaves it holding that
#: idea while its next transmission is somebody else's problem.
MAX_WPM = 20.0

WPM = 20.0
TONE_HZ = 700.0

#: What a keyed tone occupies, for the regulatory gate. A 20 WPM CW envelope with a
#: 5 ms raised-cosine edge is a few tens of hertz wide; 150 is generous and errs
#: toward charging the emission for more than it uses.
BANDWIDTH_HZ = 150.0
FS = 48000
RAMP_S = 0.005


def dit_seconds(wpm: float) -> float:
    """PARIS: the word ``PARIS `` is 50 dits long, so a dit is 1.2 s / WPM."""
    if not 0.0 < wpm <= MAX_WPM:
        raise ValueError(f"{wpm} WPM: an identification runs at 0 < WPM <= {MAX_WPM:g} "
                         "(§97.119(b)(1))")
    return 1.2 / wpm


def _code(ch: str) -> str:
    try:
        return MORSE[ch]
    except KeyError:
        raise ValueError(f"no Morse code for {ch!r} — a callsign is A-Z, 0-9 and '/'") from None


def pattern(text: str) -> str:
    """The dit-dah rendering, for an operator to read before it goes on the air."""
    return "   ".join(" ".join(_code(c) for c in word) for word in text.upper().split())


def keying(text: str, wpm: float = WPM) -> list[tuple[bool, float]]:
    """The keying sequence as ``(key down, seconds)``, with no lead or trail."""
    dit = dit_seconds(wpm)
    units: list[tuple[bool, int]] = []
    for w, word in enumerate(text.upper().split()):
        if w:
            units.append((False, 7))
        for c, ch in enumerate(word):
            if c:
                units.append((False, 3))
            for e, element in enumerate(_code(ch)):
                if e:
                    units.append((False, 1))
                units.append((True, 3 if element == "-" else 1))
    return [(on, n * dit) for on, n in units]


def duration(text: str, wpm: float = WPM) -> float:
    return sum(secs for _, secs in keying(text, wpm))


def audio(text: str, wpm: float = WPM, tone_hz: float = TONE_HZ, fs: int = FS,
          amplitude: float = 0.5, ramp_s: float = RAMP_S) -> np.ndarray:
    """The identification as float samples, ready to play into the transmit path."""
    seq = keying(text, wpm)
    if not seq:
        raise ValueError("nothing to identify with")
    n_ramp = max(1, int(round(min(ramp_s, dit_seconds(wpm) / 4) * fs)))
    rise = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, n_ramp, endpoint=False))
    env = np.zeros(sum(int(round(secs * fs)) for _, secs in seq))
    at = 0
    for on, secs in seq:
        n = int(round(secs * fs))
        if on:
            env[at:at + n] = 1.0
            env[at:at + n_ramp] = rise
            env[at + n - n_ramp:at + n] = rise[::-1]
        at += n
    phase = 2 * np.pi * tone_hz * np.arange(len(env)) / fs
    return (amplitude * env * np.sin(phase)).astype(np.float32)
