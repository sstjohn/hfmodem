# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""ARDOP identification audio at the modem's native 12 kHz rate.

The published ARDOP specification requires Morse after an enabled IDFrame.
The functional reference uses 768-sample dits and 1609.375/1390.625 Hz
mark/space tones. ONOFF replaces the space tone with silence. This renderer
uses the station's standard Morse alphabet, not the reference's packed table.
"""
import numpy as np

from hfmodem.core.cwid import keying

from .modulator import SAMPLE_RATE, _TxFilter

DIT_SAMPLES = 768
MARK_HZ = 1609.375
SPACE_HZ = 1390.625


def render_cwid(call: str, *, on_off=False) -> np.ndarray:
    # The digital identity includes its SSID; the Morse identity names the
    # station's base call, as the ARDOP implementations do.
    base = call.upper().split('-', 1)[0]
    if not base or not base.isascii() or not base.isalnum() or len(base) > 7:
        raise ValueError('CW identification requires an ASCII base callsign')
    sequence = [(False, 6 * DIT_SAMPLES)]
    sequence.extend((on, round(seconds * SAMPLE_RATE))
                    for on, seconds in keying('DE ' + base,
                                               wpm=1.2*SAMPLE_RATE/DIT_SAMPLES))
    sequence.append((False, 6 * DIT_SAMPLES))
    n = sum(length for _, length in sequence)
    marks = np.zeros(n, dtype=bool)
    at = 0
    for on, length in sequence:
        marks[at:at+length] = on
        at += length
    # Both frequencies complete an integer number of cycles per dit. A
    # continuous phase accumulator also preserves phase at each transition.
    frequency = np.where(marks, MARK_HZ, SPACE_HZ)
    phase = 2*np.pi/SAMPLE_RATE * np.concatenate(([0.], np.cumsum(frequency[:-1])))
    signal = 0.9 * 26000 * np.sin(phase)
    if on_off:
        signal[~marks] = 0
    # Use the existing 500 Hz transmit filter for the FSK transitions. Padding
    # retains its tail and compensates its documented 60-sample delay.
    return _TxFilter(500).render(np.pad(np.rint(signal), (0, 60)))
