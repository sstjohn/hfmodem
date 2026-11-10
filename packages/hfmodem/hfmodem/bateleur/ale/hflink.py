# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The amateur ALE network's conventions, as HFLINK publishes them, for a
`Config`: which nets a station answers, how long a call runs, how it sounds,
how long an idle link is kept.

- Nets: HFN (texting and sounding, the data channels) and HFL (voice), each
  with three member slots, one's own callsign always member 1 and the other
  two six-character place holders; QRZ (text keyboarding) answered the same
  way; GPR (position reports) and RPT (status and bulletins) carry AMD ended
  TWAS, which no one answers. A net call is the amateur CQ: HFN on data
  channels, HFL on voice ones.
- A call, and a sound, runs 15 seconds; a tune time of 3 seconds.
- Sounds end TWAS, not TIS; a base station sounds each hour on its pilot
  channels, a mobile or portable one not at all. No LQA exchange.
- A link with no traffic is let go after six minutes.

The channel list itself is not here: HFLINK publishes it, and a station
loads it from its own channel file.
"""
from __future__ import annotations

from . import words as W
from .controller import Config

#: Each net's member slots after member 1 (one's own callsign).
NET_SLOTS = {"HFN": ("345678", "876543"), "HFL": ("123456", "654321")}
#: Nets answered: in slot 1, with one's own callsign.
NETS = ("HFN", "HFL", "QRZ", "GPR", "RPT")
CALL_S = 15.0
TUNE_S = 3.0
SOUND_S = 3600.0
LINK_IDLE_S = 360.0


def config(callsign: str, channels: list[str], *, base: bool = False,
           pilot_channels: list[int] | None = None, **kw) -> Config:
    """A `Config` on the amateur network's conventions. ``base``: sound each
    hour, on ``pilot_channels`` (indices into ``channels``; default all)."""
    me = W.check_address(callsign.upper())
    c = Config(addresses=[me], channels=list(channels),
               nets={n: (me, 1) for n in NETS},
               call_s=CALL_S, tune_s=TUNE_S, activity_s=LINK_IDLE_S,
               sound_s=SOUND_S if base else 0.0, sound_channels=pilot_channels,
               sound_terminator=W.TWAS)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def net_members(net: str, callsign: str) -> list[str]:
    """The member list a net call to ``net`` is timed by: oneself first."""
    return [W.check_address(callsign.upper()), *NET_SLOTS.get(net.upper(), ())]
