# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Where the dial goes. Arithmetic only — emission profiles live in `core.regulatory`.

A published channel names its *centre*; on upper sideband the dial is that minus
1500 Hz. Getting it wrong is silent — the radio keys, the waveform is correct,
and nothing is heard in either direction — and every modem in this project has
had it wrong at some point. `DIAL_OFFSET_HZ` is defined here and nowhere else.

It happens to equal the audio passband centre used by the modulators
(`shrike/spec.py`, `sabir/phy/preamble.py`, `besra/dsp/templates.py`). That is a
property of the sideband convention rather than a shared constant, and the two
must not be collapsed: change the audio centre of one waveform and the dial
convention does not move with it.

This arithmetic is independent of the operator's selected limits. The station
checks emissions against its chosen `core.regulatory` profile separately.
"""
from __future__ import annotations

#: A published channel centre minus its dial, on upper sideband.
DIAL_OFFSET_HZ = 1500

#: HF band-plan ranges in MHz, with band names in metres.
AMATEUR_HF_BANDS = (
    (1.8, 2.0, 160), (3.5, 4.0, 80), (5.3, 5.41, 60), (7.0, 7.3, 40),
    (10.1, 10.15, 30), (14.0, 14.35, 20), (18.068, 18.168, 17),
    (21.0, 21.45, 15), (24.89, 24.99, 12), (28.0, 29.7, 10),
)


def amateur_hf(hz: float) -> bool:
    return any(lo * 1e6 <= hz <= hi * 1e6 for lo, hi, _ in AMATEUR_HF_BANDS)

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
