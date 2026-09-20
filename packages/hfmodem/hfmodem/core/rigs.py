# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The rigs this station keys, and the hamlib settings each one needs.

Model IDs verified against the local build (`rigctl --list`, Hamlib 5.0).

One table, because there were three and they disagreed: two of them carried the
same model, baud and mode, one of those two also carried the PTT settle time and
the keying line, and a shell launcher carried a fourth copy of the numbers. A
divergence in a table like this is not a lint problem — it decides how a radio
gets keyed.

**`settle`**, the pause between key-down and the first sample, is per rig because
its cause is per rig. The X6100's USB-audio codec re-initialises when it is keyed,
so pushing samples immediately stalls the stream and the head of the burst never
reaches the modulator; it needs 0.40 s. The FT-891 drives an external dongle that
does not re-initialise, and carrying the X6100's figure over held the key open
with dead air at both ends of every burst, which an operator watching the rig will
rightly call out. What settle buys is nothing and what it costs is keyed time: on
PACTOR's 1.25 s cycle, 0.04 leaves 250 ms of clear air per cycle against a
commercial modem's 288. **It is the first number to revisit if a burst comes back
clipped at the head** — in either direction, since too much is as visible as too
little. Raising the FT-891's to 0.10 does not break shrike's schedule; it narrows
the band of peer turnarounds the cycle can service, and `shrike.onair._budget`
prints that band rather than leaving it to be argued about.

The G90's 0.10 is carried rather than measured; nobody has keyed one here.

**`ptt_type` and `ptt_iface`** say how the rig is keyed. hamlib's `RIG` is a CAT
command; `RTS` is a modem line on a second serial interface, which is one ioctl
rather than a blocking CAT transaction — see `core.ptt` for what that buys and
what it costs. A rig with no `ptt_iface` has no second interface and keys over the
port it already has.
"""
from __future__ import annotations

RIGS = {
    "ft891": {"model": 1036, "baud": 38400, "mode": "PKTUSB", "settle": 0.04,
              "ptt_type": "RTS", "ptt_iface": 1,
              "note": "Yaesu FT-891; set CAT rate to 38400 (menu 05-06); external soundcard on DATA jack"},
    "g90":   {"model": 3088, "baud": 19200, "mode": "PKTUSB", "settle": 0.10,
              "note": "Xiegu G90; CAT on its serial, audio via external interface"},
    "x6100": {"model": 3087, "baud": 19200, "mode": "PKTUSB", "settle": 0.40,
              "note": "Xiegu X6100; single USB gives both CAT serial and a soundcard"},
}
