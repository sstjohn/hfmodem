# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The Morse station identification, character by character and dit by dit.

A mis-keyed callsign is worse than no callsign: it identifies someone else. So
nothing here is checked against the generator's own tables. The character set is
compared with an independently transcribed ITU-R M.1677-1 one, the timing with
the PARIS standard, and the audio is read back through an envelope detector that
knows only what a receiver would know.
"""
import numpy as np
import pytest

from hfmodem.core import cwid as station_id

# ITU-R M.1677-1, transcribed here on its own so a slip in either table shows up
# as a disagreement rather than as agreement with itself.
ITU = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....",
    "6": "-....", "7": "--...", "8": "---..", "9": "----.", "0": "-----",
    "/": "-..-.",
}

def test_character_table_is_itu():
    assert station_id.MORSE == ITU

def test_unknown_characters_are_refused():
    """Silence or a guess would both put the wrong callsign on the air."""
    for bad in ("W9SSJ!", "W9 SSJ.", "ÅA1AA"):
        with pytest.raises(ValueError):
            station_id.audio(bad)

def test_dit_is_paris():
    """'PARIS ' is 50 dits; the word alone, with no trailing gap, is 43."""
    assert station_id.dit_seconds(20.0) == pytest.approx(0.06)
    assert station_id.duration("PARIS", 20.0) == pytest.approx(43 * 0.06)
    assert station_id.duration("PARIS PARIS", 20.0) == pytest.approx((43 + 7 + 43) * 0.06)

def test_gaps_are_one_three_seven():
    seq = station_id.keying("EE E", 20.0)
    dit = station_id.dit_seconds(20.0)
    assert seq == [(True, dit), (False, 3 * dit), (True, dit),
                   (False, 7 * dit), (True, dit)]
    assert station_id.keying("A", 20.0) == [(True, dit), (False, dit), (True, 3 * dit)]

def test_audio_length_matches_the_keying():
    a = station_id.audio("W9SSJ", wpm=18.0)
    assert len(a) / station_id.FS == pytest.approx(station_id.duration("W9SSJ", 18.0), abs=1e-3)

def test_tone_is_where_it_was_asked_for():
    a = station_id.audio("W9SSJ", tone_hz=650.0)
    spectrum = np.abs(np.fft.rfft(a.astype(float)))
    peak = np.fft.rfftfreq(len(a), 1 / station_id.FS)[int(np.argmax(spectrum))]
    assert peak == pytest.approx(650.0, abs=5.0)

def test_keying_does_not_click():
    """Rise and fall shaped: the waveform never steps further than the tone itself.

    A hard-keyed element steps the full amplitude in one sample, and that edge is
    what splashes onto the neighbours.
    """
    a = station_id.audio("W9SSJ", tone_hz=700.0, amplitude=0.5).astype(float)
    tone_step = 2 * np.pi * 700.0 / station_id.FS * 0.5
    assert np.abs(np.diff(a)).max() < 1.2 * tone_step
    assert a[0] == 0.0 and a[-1] == 0.0
    assert np.abs(a).max() <= 0.5 + 1e-6

def _decode(a: np.ndarray, wpm: float, fs: int = station_id.FS) -> str:
    """Recover the text from the audio, knowing only the speed. No inside view."""
    inverse = {code: ch for ch, code in ITU.items()}
    win = int(0.002 * fs)
    env = np.convolve(np.abs(a.astype(float)), np.ones(win) / win, "same")
    on = env > 0.25 * env.max()
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    runs = [(bool(on[s]), e - s) for s, e in
            zip(np.r_[0, edges], np.r_[edges, len(on)], strict=True)]
    dit = station_id.dit_seconds(wpm) * fs
    text, code = "", ""
    for keyed, n in runs:
        units = round(n / dit)
        if keyed:
            code += "." if units <= 2 else "-"
        elif units >= 3:
            text += inverse[code] + (" " if units >= 7 else "")
            code = ""
    return text + inverse[code]

@pytest.mark.parametrize("call", ["W9SSJ", "W9SSJ/P", "K0A", "VE3ABC/QRP", "8P9AA"])
@pytest.mark.parametrize("wpm", [18.0, 20.0])
def test_a_receiver_reads_back_the_callsign(call, wpm):
    assert _decode(station_id.audio(call, wpm=wpm), wpm) == call

def test_a_receiver_reads_back_two_words():
    assert _decode(station_id.audio("DE W9SSJ", wpm=18.0), 18.0) == "DE W9SSJ"
