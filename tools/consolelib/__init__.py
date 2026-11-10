# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The station console: one substrate, one profile per radio mode.

find a host   roster.py    peers, channel resolution, dial arithmetic, hours
check the rig station.py    CAT readback, menu, codec gains, tap, witness
claim the rig claim.py      the arm lock, spacing, what this gateway last did
power the rig power.py      the smartplug on the mains; the CAT reset runs on it
hear the rig  monitor.py    what the receiver hears, to the terminal and events.jsonl
call a host   arm.py        build, launch, tail, provenance, verdict, status
what it loads residency.py  what the interpreter resolved, and --qualify's refusal
a sequence    campaign.py   arms with dispositions, previewed before it keys
across modes  multimode.py  one band, every station in the mode that is right for it

What differs between modes is `modes.Mode`, which is mostly data.
"""
