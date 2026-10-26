# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The sound card a hermetic test has unless it put a scripted one ahead of
this: any use of it is a TRIP, so no test reaches the station's codec."""
from hermetic_guard import trip


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    trip(f"sounddevice.{name}")
