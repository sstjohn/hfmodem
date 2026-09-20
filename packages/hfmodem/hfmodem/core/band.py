# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where the dial goes. Arithmetic only — the rules live in `core.regulatory`.

A published channel names its *centre*; on upper sideband the dial is that minus
1500 Hz. Getting it wrong is silent — the radio keys, the waveform is correct,
and nothing is heard in either direction — and every modem in this project has
had it wrong at some point. `DIAL_OFFSET_HZ` is defined here and nowhere else.

It happens to equal the audio passband centre used by the modulators
(`shrike/spec.py`, `sabir/phy/preamble.py`, `besra/dsp/templates.py`). That is a
property of the sideband convention rather than a shared constant, and the two
must not be collapsed: change the audio centre of one waveform and the dial
convention does not move with it.

Nothing here knows about any regulator. Which emissions are permitted is a
question for whichever `core.regulatory` profile the station declares, and this
module is used identically by a station under an amateur licence, a commercial
one, an experimental authorisation, or none at all on a bench.
"""
from __future__ import annotations

#: A published channel centre minus its dial, on upper sideband.
DIAL_OFFSET_HZ = 1500

#: How far the dial may sit from what was asked for and still be that channel.
#: Measured rather than chosen: over the 24 QSYs this station made through
#: rigctld on 2026-08-13/14 the FT-891 read back the dial it was given exactly,
#: 24 of 24, while the closest two channels called here sit 500 Hz apart. The bar
#: is wider than any readback error seen and far narrower than any wrong channel.
#:
#: It was written down at four transmit paths and one of them said 10 -- a number
#: with no measurement behind it, silently stricter than the other three on the
#: same rig. Which is the argument for one definition rather than for either
#: value.
QSY_TOLERANCE_HZ = 20


def dial_hz(centre_hz: float) -> int:
    """The dial for a published channel centre."""
    return int(round(centre_hz)) - DIAL_OFFSET_HZ


def centre_hz(dial: float) -> int:
    """The channel centre a dial setting corresponds to."""
    return int(round(dial)) + DIAL_OFFSET_HZ


# Actual emission placement is core.regulatory.Emission.span, using the
# modulator's measured audio passband in core.occupied. A nominal audio center
# alone does not specify carrier allocation, filter skirts or occupied width.
