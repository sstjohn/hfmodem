# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Operational PACTOR modem over the radio -- shrike's own receiver in the loop.

Where `ota.py` probes a gateway and hands the reply to an external reference
decoder, this runs a real ARQ session: shrike transmits a connect, then decodes
the reply with its OWN receiver (`shrike.rxfront`) and drives `PactorArq`, so the
state machine reacts on the air. Half-duplex on the 1.25 s cycle grid -- our cycle
transmits, the peer's cycle is recorded and decoded -- reusing the exact receive
path the monitor and `shrike.live` run, so a connect that works here is the same
decode proven off-air.

TRANSMIT IS OFF BY DEFAULT. Without ``--transmit`` this is a dry run: every TX
burst is rendered to a WAV (and is decodable by shrike's own receiver, which
proves the waveform) but the rig is never keyed, and the reply is read from
``--reply-wav`` so the whole FSM loop is exercised with no hardware. ``--transmit``
keys the rig and requires a licensed operator on frequency; keep power <=50 W
unless the operator is present.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
from collections import deque
import itertools
import json
import math
import os
import queue
import re
import shlex
import signal
import sys
import threading
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.fft import next_fast_len

from . import (live, ota, p1rx, p2rx, p3acquire, p3frame, p3rx, p4chirp, p4sig, pactor1, pactor2,
               placement, rxfront, session, spec, traffic)
from ..core import band, config, cwid, gil, levels, rates, wav
from ..core.devices import find_device
from ..core.ptt import PttError
from ..core.rig import FATAL_SIGNALS, arm_deadman
from ..core.rigs import RIGS
from ..winlink import CLIENT_SID, progress_to_stdout
from .arq import (ArqConfig, CS_ACK, CS_BREAKIN, CS_CYCLE_TOG, GOODBYE_CYCLES, IRS, ISS, LONG_TICKS,
                  REFUSED, SEQ_MOD, State, check_entry_ladder)
from .ptc import GRANT_ENTRY_SL, PtcHost
from .spec import Protocol
from .p3trial import DEFAULT_ENTRY_DELAY_MS, EntryTimingTrial, ReplyClock, TimingTrial
from .p4probe import EntryProbe

FS = ota.FS
# Below this the capture cannot carry a decodable signal, so a quiet window says
# more about our input gain than about the far end. Real off-air traffic recorded
# through this rig sits at ~0.1-0.3 RMS; the deliberately-attenuated corpus
# setting produced ~0.002.
RX_DEAF_RMS = 0.01
# The station's drive, not one of shrike's own -- the four modems share one
# interface and one radio. What it is for and how to raise it is in `core.levels`;
# `--tx-drive` overrides it for a run, and `[audio] tx_drive` sets it for the
# station.
TX_DRIVE = levels.TX_DRIVE
# Measured on the FT-891: receive audio stays muted for about 85 ms after PTT
# drops, so a listening window has that much dead time at its head.
#
# Measured 2026-07-28 from the sessions themselves: the median level across the 46
# listening windows of the 2026-07-27 and 2026-07-28 WS8EOC sessions sits at -50 dB
# for the first 55 ms after the carrier drops and reaches the window median by 60.
# Those recordings are working-record material and do not ship, so the figure is
# stated here rather than the path.
# 0.085 stood here before, and 0.10 before that, both asserted rather than read off
# a capture. This one is worth re-measuring per rig: `--preflight` does it in
# fifteen seconds and prints what it found.
TR_SWITCH_S = 0.055
# The two acknowledging codewords, as indices into spec.P1_CS_NAMES. They carry
# their meaning in the CHANGE from one to the other rather than in either word --
# a repeated CS1 is a repeat request exactly as a repeated CS4 is -- so the
# end-of-session verdict looks for the alternation across them, not for a value.
P1_ACKS = (0, 1)
# What an UPGRADE arm announces in the status byte's bits 4-5 unless told
# otherwise. Both recorded PACTOR-1 -> PACTOR-3 completions' callers announce with
# both bits set (W4DNA 0x31, DL6MAA 0x35), every granted session in this station's
# record announced at 3, and no grant has ever arrived at 0, 1 or 2 -- so 3 is the
# honest declaration for a station that implements the upgrade, and 0 declares a
# PACTOR-1 ceiling. Every granted session on record was drawn at 0x31 or 0x35;
# none at 0x21 or clear.
# A mail arm announces 0 and the same flag is its opt-out: see `_arm_defaults`,
# where the measurement for that half is.
P1_STATUS_ANNOUNCE = 3
# The states in which a link is up and owes the far end a cycle. DISCONNECTING is
# one of them: the QRT rides a packet, so a station saying goodbye is transmitting.
LINKED = (State.CONNECTED, State.DISCONNECTING)
# How much audio to capture after hearing a peer's reply start. Its link-setup
# reply is ~350-405 ms -- three copies of a 12-bit control signal at 100 Bd --
# but ONE copy decodes, so waiting for all three buys nothing and spends the
# cycle budget the acknowledgement has to fit inside.
#
# Measured on three independent bursts (both corpus anchors and WS8EOC's real
# answer on 7100 kHz): 120 ms of post-onset audio fails, 140 ms decodes at zero
# errors, and nothing above 140 improves it. The cliff sits at one codeword, as
# it should. 180 gives 40 ms over the proven figure and still returns 270 ms of
# turnaround to the 1.25 s cycle.
P1_BURST_S = 0.18
# The session decoder slides far faster than the band monitor's, because ARQ has
# to answer inside the cycle rather than merely notice eventually. The window
# cannot shrink -- it must still hold a 3.75 s data cycle whole -- so the slide
# does the work. Decoding costs 0.021x real time on real off-air PACTOR-3
# (measured: a 3.75 s cycle in 77 ms), which puts a 4 s window every 0.25 s at
# about a third of one core.
RX_WINDOW_S = 4.0
RX_SLIDE_S = 0.25
# How much of the cycle the end-of-cycle flush re-decodes. See _SessionRx.flush.
FLUSH_CONTEXT_S = 0.75
PREKEY_RESERVE_S = 0.030
"""What the listen window keeps back for the decode that runs before the key.

ONE DECODE FITS HERE AND IT IS THE FRAME SCAN, because that is the one the
cycle's own answer hangs on: an IRS that has not read the peer's packet by its
own boundary has nothing to acknowledge. MEASURED on this machine, this repo:
20.3 ms over a 1.30 s window with nothing in it, 1.8 ms when the packet is there.
With the holdback in front of it this reserve covers that three times over.

FIXED AT BOTH ENDS, which is why it is 30 ms and not more. Below the scan's cost
the decode runs past the key instant -- the bridge runs TO that instant -- and
the cycle hands its slot back to the grid, which is 2.5 s between packets and
stalls a link on its own (docs/protocols/pactor/pactor1-timing.md §3, §8.1).
Above it the window closes early enough to put the PEER'S OWN CONTROL SIGNAL
outside the audio the rolling decoder is fed: on the 1.25 s raster that codeword
ends 65 ms before our boundary and this window closes 43 ms before it, so tens of
milliseconds are all that separate them. Widening the reserve to 55 ms moved the
codeword across that line, and both the held link and the changeover in
`hfmodem.tests.shrike.test_silence` and `test_breakin` stopped working.

So the end-of-cycle flush -- 26.7 ms measured, and the second decode this reserve
used to carry -- runs BEHIND OUR OWN CARRIER instead. It is a re-decode of audio
the rolling decoder has already been fed, where `control_signal` reads the peer's
codeword at the instant the grid predicts for 0.3 ms; what moving it costs is
that a codeword only the flush can find reaches the state machine a cycle late.

NO AUDIO IS GIVEN UP FOR ANY OF IT. The bridge still runs to the key instant and
`read_ready` collects everything the reserve did not read, so the window handed
to the anchored read is the same window as before -- which matters, because a
changeover head needs 155 ms past its anchor and truncating the window to the
peer's burst instead cost the 840 ms behind one, whole.
"""
TX_ADMIT_RESERVE_S = 0.002
"""What a keyed cycle keeps in hand between its LAST READ and the admission check.

TWO DEADLINES, AND NOBODY HAD MEASURED THE GAP BETWEEN THEM. The cycle's last
read is scheduled on the PTT instant, `boundary - settle_n`; the transmitter's
admission check (`_LiveInput.clamp_late`) is taken `key_notice` in front of the
boundary, which at the measured 1152-sample ADC-to-DAC latency and a 128-frame
block is 32 ms of the FT-891's 40 ms settle. So everything the cycle does after
that read -- the anchored codeword read, the tick, the render, the final drain --
had 8 ms: a rig property minus a device property, sized against nothing.

MEASURED AGAINST THE WORK, 2026-09-11. The arm's own printed figures put 7.0 to
16.8 ms in that window over 29 keyings, median 9.1, and eight transmissions were
refused for overrunning it -- every one of them on a slot with `slot % 4 == 1`,
which is the 128-frame callback grid beating against 60000 = 468.75 blocks a
slot. The guard was adjudicating a budget the cycle could not meet, and it was
right to refuse: one of those bursts would otherwise have been clamped 29.4 ms
into the peer's raster. What was wrong was the budget.

AND IT IS CAPPED BY THE PEER'S OWN ANSWER, which is why it is 2 ms and not the
6 the measurement asks for. This reserve comes out of the cycle's LAST READ, and
the last read is what the codeword reader is given: `_budget` puts the top of the
readable turnaround at `cycle - settle - packet - P1_CS_S - ACQUIRE_TAIL_S`,
which at a 40 ms settle is 109 ms, and every millisecond taken here comes off
that. `D_NOMINAL_S` -- the turnaround this station bootstraps from, the one
`--tx-offset` assumes and the one every bench scene runs at -- is 105. So the
whole slack over the nominal is 4 ms, the peer's changeover packet wants 2 of
them (`tests.shrike.test_grid.the_reversed_link_answers_out_of_channel_time`
loses the link at 3), and 2 ms is what the schedule has to give. The rest of the
deficit cannot be bought here at all: it is the rig's T/R settle, the 141 ms the
codeword search wants past our carrier, or the work itself.

What it does buy is the threshold. The guard trips when the pre-key work passes
`settle - key_notice` plus the input latency and whatever the callback phase
leaves, and the arm sat on that threshold -- so it refused ordinary cycles one in
four as well as the one that mattered. Moving it by two milliseconds moves the
whole sawtooth off the mean. It costs 2 ms of channel time out of 1250 per keyed
cycle and it does not shorten the PTT lead, it lengthens it: the drain finishes
earlier, so `transmit` has the whole settle left to sleep through.
"""
# What a recovered slot keeps in hand for the decode that runs before its key.
#
# `_regrid` hands back a slot the cycle overran, listens through it, and decodes
# what it hears -- and that decode spends wall clock while consuming no samples,
# which is the exact mechanism that lost the slot in the first place. Doing it
# after the bridge, as the cycle does, makes the re-test fail on the same
# geometry every iteration: a loop with no progress term.
#
# MEASURED on this machine, this repo: `_SessionRx.flush` 26.7 ms over its 0.75 s
# cap, `deep_scan` 20.3 ms on a 1.30 s window with nothing in it. 150 ms carries
# both with room for the station Pi, and it comes off the front of a ~1.05 s
# recovered window rather than off the key.
REGRID_RESERVE_S = 0.150
# ...and how many times `_regrid` may try before it gives the burst to the
# emission path's backstop. A re-grid that can run forever occupies nothing and
# reports nothing wrong, which is worse than a burst that goes out late: with the
# decode charged where it happens, three iterations of a non-converging loop cost
# four slots and then say so.
REGRID_TRIES = 3
# How much of a listening window may go through the ROLLING decoder, in slots.
# Everything else in the window is collected and held for the flush.
#
# THE ROLLING DECODE COSTS MORE THAN THE AUDIO LASTS, and every window is as long
# as the loop is late. Those two together are what took the 2026-08-02 session
# from a 1.25 s cadence to minutes: a cycle that overran collected a longer
# window, a longer window cost more to decode, and the cost was paid in the same
# wall clock that decides which slot is still reachable. The session's captures
# are in `captures/onair-0802-1609` and `hfmodem.tests.shrike.test_grid`
# reproduces it from the inside.
#
# MEASURED ON THAT SESSION'S OWN AUDIO -- hold_08.wav, the 59.8 s window it
# collected -- pushed through `_SessionRx`'s own rolling parameters, a 4 s window
# re-decoded every 0.25 s slide, so each second of channel is looked at sixteen
# times. CPU seconds against seconds of audio:
#
#     window   0.25 s   1.25 s   2.50 s   5.00 s   10.0 s
#     decode    0.00 s   2.99 s  15.52 s  68.68 s   187 s
#     per s      0.0      2.4      6.2     13.7      18.7
#
# The design's own window -- what a keyed cycle has between the peer's answer and
# its next key -- is 250 ms, and at 250 ms this is free: the buffer never reaches
# the half second `RollingRx` will decode below. One slot in is where it starts
# costing more than it takes, and it climbs from there.
#
# The same measurement on synthetic noise reads 0.11, 0.27 and 0.81 over the same
# lengths -- under one throughout, and a third of what the air costs at 10 s.
# That is why nothing offline had ever seen this: an empty channel gives the
# decoder nothing to run its scans on, and real audio is what makes it dear.
# `perf/shrike-rtf.py` says the same thing from the other side, and its budget
# for this path is 0.1.
#
# ONE SLOT, because that is the protocol's own answer time. "PACTOR arbeitet als
# bitsynchrones System mit einem festen Zeitraster": a station answers inside the
# cycle, so audio further back than a cycle belongs to a turn that has already
# gone, and hearing it a slide sooner buys nothing that can still be acted on.
# Nothing is discarded -- the held audio is in the same buffer, reaches the flush
# and both frame scans, and `_collect` has always treated the last of a cycle
# exactly this way.
FEED_MAX_SLOTS = 1
# Cycles the hold loop may spend CLOSING the link once its budget is out. The
# disconnect is a protocol exchange rather than a state change: the QRT bit
# rides a data packet, the peer acknowledges it, and a station that is receiving
# when its budget expires must first take the link back with a break-in packet
# to say anything at all -- two to four cycles end to end. The loop leaves as
# soon as the state machine reports the link down, so this is a cap and not a
# duration.
QRT_CYCLES = 4
# The furthest out a hold can ever be pushed, whatever the peer is saying. See
# `_HoldBudget`: the deadline moves with the bytes, and a peer that never stops
# sending them would otherwise hold a shared HF channel for as long as it liked.
#
# 480 cycles is ten minutes. At PACTOR-1's 100 Bd a packet carries 8 bytes of
# field every 1.25 s, so this ceiling is about 3.8 kB of payload -- more than a
# Winlink greeting, proposal and short message together, and rather more than the
# 68.9 s of the longest session this station has on disk.
HOLD_MAX_CYCLES = round(600 / spec.CYCLE_SHORT_S)
# A mail exchange IS a held link: two minutes of no-payload cycles to give the
# gateway room to greet, propose and turn the channel round, every cycle that
# moves bytes buying another N. Without it the connect branch disconnects on the
# very cycle the gateway would have greeted us.
MAIL_HOLD_CYCLES = 96
# What the read of a changeover packet keeps past the frame's last bit. The
# packet behind a CS3 head spans exactly `at + packet_n`, and a buffer cut at
# that very sample puts the CRC's final bits under the demodulator's own
# integration window, where they read as nothing: hold_14 of onair-0803-225309
# carries WS8EOC's changeover packet with every bit before the tail intact and
# lost its CRC to exactly this edge. Four 100 Bd bit-times, and the guard is the
# first thing the caller gives up: it caps the read at its own key instant less
# `PREKEY_RESERVE_S`, so what the reserve takes comes out of the guard and the
# packet is read whole whatever is left.
BK_TAIL_GUARD_S = 0.040
# The operator lines a gate can be written against. Pinned here so that rewording
# one cannot silently take a test's assertion with it.
SLOT_GONE = "IS GONE"
LATE_KEY = "LATE TO THE KEY"
# ...and the opposite verdict on the same overrun: the burst went out on the
# boundary it was aimed at rather than a later one. Its own words because it is
# its own fact -- `LATE_KEY` says a cycle was moved or refused, and a scene that
# counts one against the other must not read this as either. See
# `KEY_CLAMP_TOL_S`.
KEYED_LATE = "KEYED INTO ITS BOUNDARY"
# ...and the grid's own half of that verdict, taken a tick and a render earlier:
# the slot is NOT handed back, because the emission path can still key this
# carrier inside the boundary. See `_clamp_forgives`.
SLOT_KEPT = "IS KEPT"
# ...and the render seam's own half, one tick later: the aim is re-tested before
# a polarity is baked in, and a slot given up THERE is as spent as one `_regrid`
# handed back. Its own words so that a scene counting refusals sees this one
# too. See `RadioTx._advance_aim`.
SLOT_UNRENDERED = "GIVEN UP BEFORE THE RENDER"
SHORT_OF_THE_FLOOR = "SHORT OF THE LISTEN FLOOR"
BEHIND_THE_GRID = "BEHIND THE GRID"
# ...and the cycle an ISS spends listening for the answer to its changeover
# rather than re-keying one nothing can answer. See `arq.breakin_listen_due`.
LISTENING_FOR_THE_CEDE = "LISTENING FOR THE CEDE"
# The floor every reader on this stream has: the rolling decoder's own, which is
# also `RollingRx.flush`'s half second. Named here because the cycle geometry is
# decided against it.
MIN_DECODE_S = live.MIN_DECODE_S
# Where our data bits sit relative to the peer's control signal, DURING
# ACQUISITION ONLY. The protocol's own arithmetic, from the turnaround gap `d`:
#
#     offset = cycle - packet - d = 1250 - 960 - d = 290 - d
#
#     d =  40 ms  (a shipped responder implementation)      -> 250 ms
#
# 85 ms stood here, on the grounds that it was the protocol's own value and the
# fixed point of the changeover involution. Retracted 2026-07-28: the involution
# holds for ANY d, a map being an involution selects nothing, and its fixed point
# 85 + p moves with the path anyway. No source states 85. What sources there are:
# the 1990 description mandates no offset at all and has the master SEARCH the
# window, hf's responder answers at packet_end + txdelay + 10 ms, and WS8EOC measured
# 172-219 ms. So 40 ms is used here not because it is nominal but because it is
# the only value a shipped implementation that interoperated actually transmitted.
#
# 40 is nonetheless not what goes here, because this rig cannot hear it: receive
# audio is still muted 55 ms after the carrier drops, so a peer answering at 40
# loses the head of its codeword to our own T/R.
#
# What goes here is MEASURED, between two commercial modems on a working link.
# rf-corpus/7101k_234600.wav carries both stations of a W4DNA -> KE5YTA exchange:
# anchored on the caller's own 1.2505 s raster, its answers sit at packet_end +
# 105, 106, 106 and 102 ms, and 968 + 105 + 120 + 58 closes the cycle at 1251 ms.
# Four cycles inside 4 ms of each other is a station's turnaround, not a spread.
# It also sits mid-band for the schedule below (d 55-130), so it costs nothing to
# take the measurement over the midpoint that stood here before it existed.
#
# It is a bootstrap and nothing more. `d` is a property of the peer, it spans at
# least 40-180 ms across the two stations we have measured, and the moment an
# answer is heard `_MasterGrid` holds the measured one instead.
#
# ACQUISITION ONLY, and that qualifier is the more important half. A master does
# not derive its transmit instant from the peer at all -- it holds a free-running
# grid and the peer comes to it. Referencing the peer here is how we FIND the
# link; continuing to reference it once running would import the peer's jitter
# into our grid and feed it back, a loop the protocol deliberately does not have.
# `_MasterGrid` spends this constant exactly once, when it places the grid.
TX_OFFSET_S = 0.185

# Where to look for the first control signal, before one has ever been heard. Only
# `rx_due` reads it, and only until `d_n` exists -- it aims the receive window, it
# never places a transmission. The same bootstrap TX_OFFSET_S is 290 - d of --
# WRITTEN AS THAT ARITHMETIC, because the two are one measurement stated twice
# and only the derivation keeps the window we watch and the grid we key on
# agreeing about where the answer should be when the offset is ever retuned.
# 1.25 - 0.96 - 0.185 = 0.105 s today.
D_NOMINAL_S = (spec.CYCLE_SHORT_S - spec.P1_PACKET_S) - TX_OFFSET_S

# What we transmit once the link has upgraded, and so where the receive window
# hangs off on an upgraded link. See `_MasterGrid.data_n`.
P3_PACKET_N = round(placement.PACKET_S * FS)

# ...and the same packet on the 3.75 s cycle: one phase reference, the header
# block, and `placement.LONG_PATHS`' 320 rows instead of 72. 3.290 s against
# 0.810, which is 2.480 s where the cycle grows 2.500 -- the long cycle carries
# 20 ms MORE slack behind its packet than the short one does, and that is the
# whole of why `_MasterGrid.regear` moves the receive anchor by the CYCLE's
# growth rather than by the packet's. Measured on `rf-corpus/PIII_Complete_1`:
# the reference IRS answers a long packet 3390 ms after its phase reference,
# against 890 for a short one -- 2500 apart, both 360 ms before the next
# boundary.
P3_LONG_PACKET_N = round(
    (placement.PACKET_S
     + (placement.LONG_ROWS - placement.FRAME_SYMBOLS) / spec.SYMBOL_RATE_BD)
    * FS)

# Nominal control interval for the grid's 600 ms role rotation. The physical
# renderer additionally includes stagger and pulse runout; see _MasterGrid.cs_n.
P3_CS_N = round((spec.CS_BITS_PER_TONE + 1) / spec.SYMBOL_RATE_BD * FS)

# PACTOR-2's, counted the same way P3_PACKET_N is -- the GRID length, one symbol
# short of the keying. `pactor2.frame` puts a nine-pulse marker and 72 (or 320)
# data pulses on the air, 81 or 329 in all, and `pactor2.cs_slot` answers a
# `TURNAROUND_S` of 70 ms behind the last of them: 0.880 s and 3.360. This is
# that answer instant less the 80 ms the grid's own `d` carries on a settled
# link, which is where P3_PACKET_N sits against 0.890 for the same reason -- the
# 82nd symbol every recorded PACTOR-3 station keys is inside the measurement and
# outside the figure.
#
# [SCS] s2 prints the same 0.800 outright: the standard packet is "shortened to
# 0.8 seconds in order not to shorten the maximum possible propagation delay,
# which is thus still 170 milliseconds", and the phase reference pulse in front
# of it is described separately (`spec.PULSE_SLOT_S`). 3.280 is the same
# arithmetic over `pactor2.PATHS_LONG`' 320 pulses, and [SCS] s2 prints that one
# too: "the length of these data packets is 3.28 seconds".
P2_PACKET_N = round((pactor2.MARKER_SYMBOLS + pactor2.PATHS[0].n_symbols - 1)
                    * spec.PULSE_SLOT_S * FS)
P2_LONG_PACKET_N = round(
    (pactor2.MARKER_SYMBOLS + pactor2.PATHS_LONG[0].n_symbols - 1)
    * spec.PULSE_SLOT_S * FS)

# ...and PACTOR-2's codeword, which is the same twenty bits and one phase
# reference at the same 100 Bd: 210 ms, arrived at through PACTOR-2's own
# constant rather than through PACTOR-3's. See `_MasterGrid.cs_n`.
P2_CS_N = round(p2rx.CS_SYMBOLS * spec.PULSE_SLOT_S * FS)

# Samples every PACTOR-2 keying goes out in FRONT of its own boundary.
#
# A SHAPED BURST DOES NOT START AT ITS OWN PHASE REFERENCE. The raised cosine
# puts the leading tail of the first pulse ahead of it, so sample 0 of what
# `pactor2` renders is 18.75 ms before the instant Figure 1's raster is written
# in -- and a station that keys such a burst FLUSH with the boundary has put its
# whole comb that late, which is nearly two symbols at 100 Bd. Nothing
# downstream recovers it: `_MasterGrid.rx_due` names the phase reference,
# `pactor2.cs_slot` counts from it, and the peer's answer window is aimed at
# where it thinks ours was.
#
# Spent at the boundary rather than by padding, for `RadioTx._breakin_key`'s
# reason: the audio is finished by the time `_tx` holds it and where the carrier
# comes up is the one thing still free. PACTOR-3 controls and CS3 heads need
# their own trimmed lead: trimming to 2% retains part of the filter skirt.
P2_KEY_LEAD_N = pactor2.pulse_lead()
P3_CS_KEY_LEAD_N = max(placement.control_pulse_lead(cs, swapped=swap)
                       for cs in range(6) for swap in (False, True))
"""Conservative receive-window reserve for the control's first pulse center."""


@lru_cache(maxsize=4)
def _p3_breakin_lead_bound(rise: bool, stagger: bool) -> int:
    """Bound the trimmed CS3 skirt using its payload-independent head.

    The first 200 ms precede the field and its causal filter response. Their
    peak cannot exceed the complete burst's peak, so their 2% crossing cannot
    be later than the actual trim. This bounds the lead without reserving the
    whole filter delay and losing a nearly complete peer frame. Both carrier
    arrangements are included; the arguments key the two renderer switches.
    """
    assert (rise, stagger) == (placement.PROTOCOL_RISE, placement.CASE0_STAGGER)
    center = int(np.argmax(placement.protocol_config().pulse()))
    leads = []
    for swapped in (False, True):
        head = placement.changeover_packet(b"", 0, swapped=swapped)[:FS // 5]
        first = int(np.flatnonzero(abs(head) > .02 * np.max(abs(head)))[0])
        leads.append(center - first + 1)  # One sample covers FFT roundoff.
    return max(leads)


# The `p2sl1` entry's last symbol past the boundary: every pulse of the short
# frame after the reference one, plus the half symbol the lagging carrier ends
# late by (`pactor2._stagger`). 81.5 symbol times: 815 ms.
P2_ENTRY_END_N = round((pactor2.MARKER_SYMBOLS + pactor2.PATHS[0].n_symbols + 0.5)
                       / pactor2.SYMBOL_RATE * FS)

# Where our own ENTRY PACKET's last symbol falls, samples after the slot
# boundary, and the other half of what `_MasterGrid.answer_position` reads an
# answer against. `P3_PACKET_N` cannot serve: it is counted from the packet's
# phase reference and the boundary is not on it. Measured off the render the
# transmitter keys, at 48 kHz: the boundary leads the first symbol by 13.9 ms of
# skirt that survives `_trim_silence`'s 2 % threshold (26.8 with the generic
# raised cosine `placement.PROTOCOL_RISE` retired: eight symbols of kernel where
# the protocol pulse spans three), and the comb behind it is
# 82.5 symbol times -- 82 symbols where `placement.PACKET_S` counts 81, the 82nd
# being `placement.ENTRY_TRAILER`, and the half because `placement.CASE0_STAGGER`
# leads channel 5 by T/2 so the two clocks do not end together. 13.9 + 825.0 =
# 838.9 ms, against 960.0 ms for PACTOR-1's last bit.
#
# THE STAGGER MOVED BOTH ENDS AND THEY DID NOT CANCEL: the tail grew 240 samples
# and the quieter leading half-symbol pushed the trim's crossing 375 samples
# later, so the keyed extent came DOWN 2.8 ms. `tests.shrike.test_p3_upgrade`
# holds this to the renderer at half a symbol.
ENTRY_END_N = 666 + round(82.5 / spec.SYMBOL_RATE_BD * FS)

# What `--p3-entry-delay` may ask for, and the ceiling is arithmetic rather than
# taste: the entry runs `ENTRY_END_N` into its slot and a slot is 1.25 s, so a
# delay past this walks the packet's tail out of the cycle the peer counts.
# 200 ms is six times the figure the flag exists to fly and still a quarter of
# the room; anything larger is a different experiment and wants its own name.
MAX_ENTRY_DELAY_MS = 200.0

# The shortest gap `_acquire` will believe is a turnaround, and the floor the
# tracker may not walk `d` under. 40 ms is the turnaround of a shipped responder
# implementation and nothing in the corpus goes below it; what sits under it is
# our own echo -- 28 of the 51 out-of-band gaps in the rig corpus, and the
# 39.1 ms latch that cost the 2026-08-10 link.
D_MIN_S = 0.040

# WHERE A CHANGEOVER PACKET KEYS IS WHERE THE PEER READS, and the two questions
# that look alike are not: where a reference modem keys ITS changeover at us, and
# where the station we are calling reads a codeword of ours. Only the second
# places a burst, and it is `_MasterGrid.peer_read_gap` -- our own free-running
# boundary, `170 ms - d`, measured from the end of the packet we are taking the
# channel from rather than from our own comb.
#
# MEASURED AT THE READER, KB5LZK: twelve codewords of ours over four arms, two
# bands and both speeds are acknowledged when their audio begins 88.5-98.0 ms
# after the peer's packet ends (median 94.9; the six the peer's packet counter
# advanced across sit 91.5-98.0). `M + d` closes on the 170 ms turnaround budget
# in every arm -- 94.9 + 78.9, 96.0 + 72.4, 91.5 + 71.1 -- and every one of those
# keyings sat at `rot` between 0.0 and +1.9 ms, which is the boundary.
#
# THE ONE CHANGEOVER PACKET A REAL MODEM HAS ACCEPTED FROM US sits in the middle
# of that window: 2026-08-22, KB5LZK, 100 Bd, `TX[28] P1 BREAK-IN #0` at +94.9 ms
# and `rot +1.9`. The gateway cancelled its own next 960 ms packet -- the raster
# it had been holding puts one at 43.2996 and the tape has nothing there --
# answered CS2 at +90.5 ms, took our `pkt#1 BK`, and sent its banner.
#
# BREAKIN_LEAD_S IS THE CONTROL, AND IT IS THE OTHER MEASUREMENT. 71.3 and
# 71.9 ms is where WS8EOC's own changeover packets arrive at US, on both arms of
# 2026-08-30, and +71.2 is where the reference peer answers DL6MAA's entry
# packet. Flown as a placement it put eighteen keyings at +68 to +70 ms on
# 2026-09-04, 21-30 ms in front of the earliest instant this gateway has been
# shown to read at, and none was answered -- against the same arms' own
# codewords, read at +96.0 in the cycle before. `p1rx.CS_ANCHOR_S` records why
# the miss is total rather than degraded: the one written reference reader takes
# twelve bits from one latched instant with no search, 100% inside 3 ms and
# nothing at 6.
#
# THE PACKET ITSELF IS A NULL AT EVERY OTHER PROPERTY -- the twelve on-air bytes
# are identical to `pactor1.breakin_packet`'s at both speeds at three gateways,
# 21 of 21, head at zero bit errors -- so this instant is the only measured
# difference there is to fly.
BREAKIN_LEAD_S = 0.072
# How far the boundary clamp may pull a changeover in front of the peer's read
# instant before the placement is refused instead: past the 1.6-3.0 ms measured
# where comb and onset disagreed on 2026-09-04's acknowledged arms.
BREAKIN_CLAMP_TOL_S = 0.005

# How far INTO its own boundary a carrier may come up and still be keyed there,
# rather than the cycle being given away.
#
# THE COST OF GIVING ONE AWAY IS TWO SLOTS, not one: the audio in hand is already
# rendered in one polarity, so the next boundary it fits is the one after next.
# VE3KPG, 2026-09-13, keyed 16 of the 32 cycles of its PACTOR-3 stint on exactly
# that -- overruns of 1.6 to 5.6 ms, every one of them inside half a symbol.
#
# AND THE OVERRUN IS THE PROTOCOL'S GEOMETRY RATHER THAN A BUDGET THIS PROCESS
# CAN TRIM. A PACTOR-3 codeword is twenty symbols at 100 Bd, 200 ms where
# PACTOR-1's is 120, so a cycle keying 0.87 s of packet and answered a turnaround
# behind it has the peer's last symbol about 1.18 s into the 1.25 s cycle -- and
# the settle and the converter's own DAC notice want the 74 ms in front of the
# next boundary. What is left between the two is single-digit milliseconds,
# whatever the tick, the render and the drain cost.
#
# Five, which is `BREAKIN_CLAMP_TOL_S`'s number for the same reason: half a
# symbol at 100 Bd, and what the reference decoder forgives. A burst later than
# this keeps the old answer and steps to the boundary its shift fits.
KEY_CLAMP_TOL_S = 0.005

# How many PACTOR-3 bursts keep their frequency-corrected render. A cycle holds
# at most an entry or a packet and a codeword, and an unacknowledged packet is
# re-sent against the same codeword, so four covers every repeat a link makes
# while a burst nothing repeats is evicted before it can hold memory. See
# `RadioTx._p3_offset`.
P3_SHIFT_KEEP = 4

# How far a burst may sit from where the grid expects it and still be taken as the
# peer's control signal. A CAPTURE RANGE, not a pull on our own timing: everything
# inside it moves the receive window, and nothing anywhere moves the transmit
# anchor.
#
# This was 0.050, justified as "wider than the protocol's tolerance because this is
# acquisition and we do not yet know the phase". Acquisition is now its own path --
# a search over the whole turnaround span, ending in one decision -- so the only
# thing left inside this limit is TRACKING, where the phase is known and 50 ms is
# just an invitation. Measured on a --replay of the 62 s W6IDS recording named
# under `BLIND_CYCLES` below, which does not ship: over ten
# tracking cycles the detector twice offered a burst 21 and 45 ms out, and each
# was accepted and moved the window 2.6 and 5.6 ms, while every genuine one sat
# inside 6 ms. Two bits, which covers the detector's own 5 ms grid and the +/-5 ms
# the protocol calls the edge, and rejects both of those.
MAX_PULL_S = 0.020

# HOW FAR A READING OF THE PEER'S RASTER MAY BE PROJECTED, in the peer's own
# cycles, before the changeover placement standing on it is refused instead of
# keyed. `_MasterGrid.peer_at` is the last onset this station READ, and until
# 2026-09-04 it was carried without limit: on the evening's three KB5LZK arms
# one codeword decoded at sample 4032796 placed ten consecutive changeover
# packets, the last of them 12.5 s later.
#
# EIGHT, WHICH IS THE OLDEST READING WITH ON-AIR EVIDENCE BEHIND IT. Arm 4's
# TX[55] was placed on an onset eight cycles back, landed at +98 ms -- inside
# the 88-98 ms window KB5LZK's packet counter advances across -- and the
# gateway answered `CS1/at anchor` at zero bit errors. Nothing older than that
# has ever been shown to place a burst a station read, and the arms' ages of 9
# and 10 all went out at the refuted instant.
#
# THE PROJECTION ITSELF DOES NOT EXPIRE, and that is the measurement rather
# than the reason for the bound. Over the 66 pairs of consecutive decoded
# onsets in those three arms the residual against a 1.25 s comb is inside
# 2.0 ms at two cycles and shows no growth with the gap -- the peer's grid runs
# 1249.99 +/- 0.44 ms, so the drift term is 0.1 ms in ten cycles and the
# spread is the detector's own 5 ms grid. What ages is the EVIDENCE that the
# peer is still transmitting on that raster at all, and this is where this
# station stops assuming it.
ONSET_MAX_CYCLES = 8

# WHERE THE IRS ANSWER SLOT IS, in the peer's own coordinates: a control signal's
# phase reference sits this far past the answered packet's (pactor3.md, "The
# answer slot" -- 889.4-891.9 ms over sixteen measured cycles, and 3.390 s for
# the same geometry over the long packet). `_MasterGrid.p3_reply_shift` is the
# only thing that puts our comb there; until it existed the IRS reply was keyed
# at a free-running comb carried over from the connect train.
P3_REPLY_S, P3_LONG_REPLY_S = 0.890, 3.390
# What a read keeps in hand for the decode that has to finish inside it, on top
# of the DAC's own notice. `_p3_decode_deadline` spends it on every tracked read
# in the loop and `_SessionRx._p3_acquisition_fits` on the one read that was
# asking `clamp_late` bare, which allows `key_notice` and nothing else.
P3_DECODE_RESERVE_S = 0.006

# How long a saved IRS clock outlives its own changeover train. The train is
# bounded by ONSET_MAX_CYCLES, so sharing that bound makes recovery unreachable
# in exactly the case it exists for: the last retry goes out at 8 cycles and the
# peer's returning stint necessarily arrives later. WS8EOC 2026-09-12 came back
# 10 periods past the corroborated packet.
TURN_RECOVERY_CYCLES = 2 * ONSET_MAX_CYCLES

# How far a CORROBORATED peer raster may be projected with nothing decoding.
# ONSET_MAX_CYCLES is what a single reading buys. Three consecutive CRC frames
# each landing within MAX_PULL_S of the projection from one origin is a measured
# comb rather than one reading, and the peer keeps keying under it:
# A5 2026-09-12 advanced its counter across a nine-cycle no-decode gap, and the
# witness fold has the peer radiating a full packet in essentially every cycle.
RASTER_PROJECT_CYCLES = 16

# The shortest burst that may re-origin the peer's raster. A PACTOR-1 station
# transmits a 120 ms codeword or a 960 ms packet and nothing between; the
# 2026-09-04 evening arms measured its codewords at 118-130 ms through both the
# burst timer and the reply test, and `rxfront.CS_BURST_MS`'s 60 ms floor is
# below anything that is one. See `_MasterGrid.note_peer_bursts`.
ONSET_MIN_MS = 100.0

# Fraction of the receive-window error taken per cycle. An eighth, which is the
# reference implementation's: its estimator low-passes the last eight sub-bit
# deviations and applies their mean, a first-order loop of gain 1/8 and time
# constant ~10 s. That number is not a tuning choice, it is the authority the far
# end has over its own clock, and a link is two of these loops meeting -- so
# running ours faster than the protocol's does not converge sooner, it converges
# somewhere the peer cannot follow.
#
# This was 0.3, applied to the TRANSMIT anchor, which is the defect this file was
# rewritten to remove. See `_MasterGrid`.
TIMING_GAIN = 0.125


class _MissedTxSlot(RuntimeError):
    """The DAC queue cannot place an explicitly scheduled burst any more."""


class RadioTx:
    """The ARQ FSM's transmit side, on a real rig (or a dry-run WAV sink).

    Implements the peer seam `PtcHost` hands its PHY-bound calls to. It renders
    each outbound burst and either keys the rig and plays it, or -- in a dry run --
    stashes it to a WAV. The reply never comes back through here: it arrives over
    the air and is recorded and decoded by the run loop, unlike the SimPeer.
    """

    def __init__(self, rig=None, *, transmit: bool, out_dev=None,
                 outdir: Path, max_key: float = 40.0, settle: float = 0.10,
                 drive: float = TX_DRIVE, p1_drive: float = 1.0):
        self.rig, self.transmit = rig, transmit
        self.out_dev, self.outdir, self.max_key = out_dev, outdir, max_key
        self.settle, self.drive = settle, drive
        # The PACTOR-1 leg's share of the drive. `at_drive` normalises every
        # burst to the same peak, which puts constant-envelope FSK ~6 dB ABOVE
        # the speed-level-1 entry packet in average power -- where both modems of
        # the reference session key it the other way round: DL6MAA's entry
        # carries +7.0 dB over its own PACTOR-1, and the PTC-II answering it
        # holds the same ratio. 1.0 keeps the shipped behaviour; below 1.0 keys
        # only the PACTOR-1 renderings at drive * p1_drive so an arm can fly the
        # reference's convention without touching what the entry packet sends.
        self.p1_drive = p1_drive
        # Explicit rig experiment; normal sessions retain reply-derived phase.
        self.p1_setup_phase = "reply"
        self.live = None
        self.sessrx = None
        self.timing_trial = None
        self.reply_clock = None
        self.last_dur = 0.0
        self.reply_at = None
        self.tx_end = None       # capture-stream index of our carrier dropping
        self.tx_audio_start = None   # ...of our first modulated sample
        self.tx_pulse_offsets = None  # phase centers relative to trimmed audio
        self.tx_key_up = None        # ...and of PTT, a settle in front of that
        self.boundary = None     # the slot this transmission is aimed at
        self.raster = None       # ...and the grid it came off, so it can be re-aimed
        self.slot = 0
        self.host: Optional[PtcHost] = None
        self.guard_drops = 0     # consecutive refusals at the key, either role
        self.keyings: list[tuple[int, int]] = []   # carrier up/down, capture samples
        self.n = 0
        self.traffic = traffic.TrafficLog()
        # The PACTOR-1 packet counter of every data packet that went out, in
        # order. Whether it ever moved is half the end-of-session verdict, and
        # the FSM's own `_next_seq` cannot answer it: a speed change requeues the
        # packet and winds that counter back, so the only record of what was on
        # the air is kept where the air is.
        self.p1_seq: list[int] = []
        # ...and how many of them carried information the peer had not been
        # offered before, which is what `--p1-status-from` counts off. See
        # `_p1_nth`.
        self.p1_packets = 0
        self._p1_bits45_keyed = False
        # Which entries of `p1_seq` were break-ins. The changeover packet resets
        # the counter, and a reset out of #3 lands on #0 -- arithmetically the
        # same step as the wrap an acknowledged #3 makes, and the only thing that
        # tells them apart is which packet carried it.
        self.p1_breakin_at: set[int] = set()
        # THE SAME RECORD FOR EVERY PROTOCOL, and it is a separate list because
        # `_p1_nth` asks a PACTOR-1 question of the one above. The end-of-session
        # counter account read `p1_seq`, so a session that spent its whole link
        # in PACTOR-3 reported the two PACTOR-1 packets it opened with -- `packet
        # counters sent: #1` and `our packet counter never advanced` over a
        # thirty-six-cycle PACTOR-3 train that was numbered 3 throughout.
        self.seq_sent: list[int] = []
        self.breakin_at: set[int] = set()
        # "Die Shiftlage der FSK-Aussendung wird einmalig beim Verbindungsaufbau
        # fixiert. Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage
        # invertiert."
        #
        # THE SHIFT IS A PROPERTY OF THE CYCLE, NOT OF THE TRANSMISSION, and the
        # difference is what stalled the WS8EOC link on 2026-07-27. The cycle is
        # the grid's business, so `_MasterGrid.shift` counts it and `aim` carries
        # the answer here for the burst about to go out; everything in one cycle
        # shares it, which is what §7 says both directions do.
        #
        # This counted transmissions, and then counted SAMPLES since the call --
        # each in turn correct until something moved underneath it. Slots are what
        # survive both: the never-transmit-late guard skipping one, and `reverse`
        # sliding the anchor a rotation at a changeover.
        self.shift_inverted = False    # dry-run fallback: no grid, no cycle count
        self.invert: Optional[bool] = None
        # The polarity the audio now in flight was RENDERED in, which is not the
        # same fact as the polarity the grid currently wants. See `_flip`.
        self._sent_invert: Optional[bool] = None
        # A keyed P3 changeover establishes our data-carrier order relative
        # to local slot parity. Keep that order through the following data.
        self._p3_packet_swap_bias = False
        # Actual CS3 emission, scoped to the outstanding ARQ object. A peer
        # can replace its packet with CS2 before acknowledging this field;
        # that invalidates IRS geometry, not the emitted ISS pulse raster.
        self._p3_emitted_turn = None
        # Slot parity at the FIRST control keyed under `--p3-control-stagger`,
        # so the flag names the foot of burst one rather than of whichever slot
        # the session happened to open on. See `_p3_stagger_foot`.
        self._p3_stagger_anchor: Optional[bool] = None
        # The slot each burst actually went out on, appended where the carrier is
        # scheduled. The loop's own local records the slot it INTENDED, and the
        # backstop below can move a burst after that local has been read: the
        # session that put half its bursts in the next slot printed "median slot
        # increment 1 = 1.25 s" off exactly that.
        self.slots_used: list[int] = []
        # WHAT ACTUALLY WENT ON THE AIR: (slot, seconds of carrier) for every
        # burst the rig was keyed for, appended after the burst and only once
        # `Rig.key_failure` finds the keying channel still standing. It sat
        # where PTT went UP until 2026-08-10, which made it a record of
        # intentions: 23 of these were appended against a serial device that
        # did not exist -- the key-downs were pipe writes into rigctls already
        # dying on it -- and the summary summed them into "22.1 s of carrier"
        # while the operator heard nothing.
        #
        # `self.n` is not that number and never was. It counts RENDERED bursts --
        # it is the TX[n] label and the dry run's filename, and in a dry run
        # nothing is transmitted at all -- and it counts a 120 ms control signal
        # and a 960 ms data packet as one apiece. The session summary printed it
        # against the cycle count, and every duty-cycle argument since has been
        # quoted off that pair: the 2026-08-06 witness session reported
        # "transmissions 22" over 97 cycles, which is true of bursts (14 packets
        # and 8 control signals, from the capture sidecars) and says nothing about
        # how much of the air we held.
        self.keyed: list[tuple[int, float]] = []
        # The PTT lead of every duplex burst, short or not; what makes a gridded
        # one short is worked out at the read in `_tx`. The session summary quotes
        # the worst of these once; the per-burst line speaks only for a new worst,
        # because a line printed on every burst of a whole slot is one the
        # operator learns to read past.
        self.leads: list[float] = []
        # Whether the transmission this cycle may be a changeover packet, which
        # is the one burst not keyed on the boundary. Read from the FSM at the
        # top of the cycle (`ptc.PtcHost.breakin_due`) because the window in
        # front of the key has to close early enough to reach the instant.
        self.breakin_due = False
        # ...and whether the burst in hand was actually placed against the
        # peer's transmission, which is what `_tx` reads to know that its
        # boundary is not a slot boundary and cannot be traded for a later one.
        self.placed = False
        # Whether the LAST burst handed to `_tx` was refused rather than keyed.
        # The packet seams read it back as `arq.REFUSED` so the retry budget
        # counts air rather than intentions. A test seam that overrides `_tx`
        # models no guard and so never sets it, which is the right answer for
        # one: nothing there can refuse.
        self.refused = False
        # ...and why the placement in hand could not be made against the peer's
        # transmission, which `_tx` refuses on. See `_MasterGrid.breakin_refusal`.
        self.unplaceable: Optional[str] = None
        self.defer_p3_cs = False
        self._pending_p3_cs: Optional[int] = None
        self._pending_p3_cs_p1 = False
        self._entry_render: Optional[tuple] = None
        # A PACTOR-3 burst the peer's raster has already been applied to, kept
        # by the burst's own identity. See `_p3_offset`.
        self._p3_shifted: dict[tuple, tuple[float, np.ndarray]] = {}
        # The changeover packets a retry repeats, with the lead measured on the
        # samples each will key: ident -> (raw render, shifted render, lead).
        # KEYED LIKE `_p3_shifted` AND NOT LIKE A SINGLE SLOT, because the ident
        # carries `swapped` and `_MasterGrid.shift` is slot parity: a changeover
        # re-placed on consecutive slots alternates it, and one entry missed on
        # all eight of `arm-v23-A-40-ws8eoc`'s re-placements (slots 199, 204,
        # 209, 214, 217, 220, 225, 230). That is 0.96 ms of `changeover_packet`
        # and 0.69 of the 2% trim, every cycle, out of the 8.2 ms such a cycle
        # spends between the grid's check and the emission path's.
        self._p3_breakin: dict[tuple, tuple] = {}

    def attach(self, host: PtcHost) -> None:
        self.host = host
        self.traffic = traffic.for_host(host)

    def _log_p3_packet(self, sl, payload, status, *, kind="DATA", long_cycle=False):
        # After the TX admission guard: an unkeyed slot is not a packet retry.
        if self.refused:
            return
        path = (placement.CHANGEOVER if kind == "CHANGEOVER" else
                (placement.LONG_PATHS if long_cycle else placement.SPEED_PATHS)[sl])
        field = placement.build_field(
            placement.field_info(payload, path.crc_bytes - 3, status), path)
        lines = self.traffic.packet("TX", sl, status, payload[:path.crc_bytes-3],
                                   kind=kind, long_cycle=long_cycle, field=field)
        self._log_p3_emission(lines)

    def _log_p3_emission(self, lines):
        if not self.transmit:
            # Only the direction label; "TX " may occur inside a payload.
            lines = [line.replace("TX ", "TX(dry-run) ", 1)
                     if line.startswith(("    [traffic] TX ", "    [speed] TX "))
                     else line for line in lines]
        traffic.emit(lines)

    def aim(self, raster: "_MasterGrid", slot: int) -> int:
        """Point the next transmission at `slot`: when it keys, and in which shift.

        Both together, because they are one fact about one cycle and a station
        that keys on the right boundary in the wrong shift is unreadable to a peer
        that counts cycles -- and unreadable in a way that looks exactly like a
        dead band.
        """
        self.raster, self.slot = raster, slot
        raster.reply_clock = self.reply_clock
        self.boundary = raster.boundary(slot)
        self.invert = raster.shift(slot)
        return self.boundary

    breakin_at_boundary = True
    """Whether the changeover packet keys where the peer READS: `170 ms - d`.

    On by default. `--breakin-lead-72` is the control and flies
    `BREAKIN_LEAD_S` instead -- where a reference modem keys its own changeover
    at us, which is 21-30 ms in front of the window this station's codewords are
    acknowledged in. Either way the instant is taken from the peer's decoded
    transmission and never from our own comb, which two reversals can leave
    840 ms along.

    BOTH PROTOCOLS, because the gap is the grid's own cycle arithmetic --
    `_MasterGrid.peer_read_gap` -- and not a constant of the waveform. It was
    `p1_breakin_lead`, then `breakin_lead`; neither of the old spellings is
    aliased, because the setting they named is not the one this holds.
    """

    def _breakin_reference(self, raster: "_MasterGrid", slot: int) -> int:
        """CS3 replaces the emitted control's pulse, not its audio skirt."""
        boundary = raster.boundary(slot)
        clock = getattr(self, "reply_clock", None)
        if clock is not None and raster.protocol == Protocol.PACTOR3:
            phase = clock.pulse_epoch
            if phase is not None:
                offset = (phase - boundary) % raster.slot_n
                if offset > raster.slot_n // 2:
                    offset -= raster.slot_n
                return boundary + offset
        return boundary

    def _breakin_key(self, raster: "_MasterGrid", slot: int) -> int:
        """Where a changeover packet keys: the peer's read instant, past its packet.

        THE BOUNDARY, TAKEN FROM THE PEER RATHER THAN FROM OUR COMB, and those
        are two halves of one 2026-09-04 finding. `peer_read_gap` is
        `170 ms - d` -- our own free-running boundary, and the window KB5LZK's
        packet counter advances across: twelve codewords at 88.5-98.0 ms past
        its packet end, and the one changeover packet a real modem has accepted
        from us at +94.9 with `rot +1.9`. `BREAKIN_LEAD_S` is the control and
        put eighteen keyings at +68 to +70, none answered.

        AND NEVER LATER THAN OUR OWN BOUNDARY. The two are the same instant
        while the tracker's prediction and the peer's decoded onset agree, and
        1.6-3.0 ms apart on 2026-09-04's arms where they do not -- 98.4 ms
        against a measured window whose top is 98.0, at a reader that is a cliff
        inside a few milliseconds. The boundary is where every one of those
        twelve acknowledged codewords sat, at `rot` 0.0 to +1.9 ms, so where the
        rule would spend the residual the boundary is taken instead. The
        2026-09-03 misplacement was `peer_end` computed off the comb, not the
        comb: a grid through both reversals puts `boundary - at_slot` at
        `170 ms - d` exactly, which `test_grid` holds.

        AND NEVER EARLIER THAN THE TRANSMITTER CAN MAKE, which outranks the
        clamp: the changeover rides behind a packet that decoded, so the frame
        scan `PREKEY_RESERVE_S` and the PTT settle are both owed between that
        packet ending and the audio. The FT-891's 40 ms leaves the reader's own
        instant reachable with 25 ms in hand; the g90's 100 does not and the
        x6100's 400 is not in the same country, and a rig that cannot make it
        keys at the earliest instant it can, which is still inside the peer's
        290 ms gap for any floor under 170 -- the codeword head has to be whole
        before the peer's next transmission, and 130 + 120 leaves 40.

        AND THE FLOOR IS NOT A PLACEMENT. Where it binds because the RIG is
        slow it is the best instant that transmitter can make; where it binds
        because the GAP collapsed it is our own comb wearing the placement's
        name, which is 2026-09-04's +70 ms. `_MasterGrid.breakin_refusal`
        separates them, and `_place_breakin` refuses ahead of this.

        WITH NOTHING READ THERE IS NOTHING TO PLACE AGAINST, and the burst keys
        where it always did.

        AND IT NAMES ONE INSTANT, WHICH IS THE ONE THE CYCLE IS BUILT AROUND.
        It used to step a peer cycle forward until the transmitter could reach
        the answer, which reads as prudence and is not: the step runs at RENDER
        time, after the window in front of the key has already closed against
        the instant being abandoned, so the cycle is spent deaf -- 1.25 s in
        which KB5LZK's whole next packet went unread, nine times an arm, on
        2026-09-04's two mail arms. Eighteen keyings of eighteen took it, every
        one logged `+1322 ms past the peer's packet`, and the packet then went
        out 72 ms past the NEXT one, which the tape agrees with to 2 ms and the
        log could not say.

        So the instant stands. `RadioTx._tx` refuses a placement that has gone
        (`LATE_KEY`, `arq.REFUSED`, no retry spent) and the FSM places the
        changeover again on the next cycle -- against that cycle's own reading,
        which is what the window closing at this instant is what buys.
        """
        retry = self._p3_retry_phase(raster, slot)
        if retry is not None:
            return retry
        end = raster.peer_packet_end(slot)
        if end is None:
            return raster.boundary(slot)
        gap = (raster.peer_read_gap if self.breakin_at_boundary
               else round(BREAKIN_LEAD_S * FS))
        key = min(self._breakin_reference(raster, slot), end.at_slot + gap)
        return max(key, end.at_slot
                   + round((self.settle + PREKEY_RESERVE_S) * FS))

    def _p3_retry_phase(self, raster: "_MasterGrid", slot: int) -> Optional[int]:
        """A fresh repeat request can retain this emitted, pending CS3 clock.

        The obsolete IRS packet/control pair is deliberately not revived. The
        evidence is an actual CS3 and a decoded control in its off-transmit
        answer aperture. Silence, a new pending field, and an unrelated phase
        cannot acquire this path. The final timing and collision guards still
        judge the waveform that will actually be emitted.
        """
        saved, host = self._p3_emitted_turn, self.host
        if (saved is None or self.reply_clock is None or host is None
                or host.protocol != Protocol.PACTOR3
                or raster.protocol != Protocol.PACTOR3 or not raster.sending
                or not getattr(host.arq, "unconfirmed_breakin", False)):
            return None
        owner, packet, identity, pulse, start, end, period = saved
        if (owner is not raster or host.arq._inflight is not packet
                or identity != (packet.status, packet.payload)
                or raster.cycle_n != period):
            return None
        cs = raster.peer_cs
        if (cs is None or cs.protocol != Protocol.PACTOR3
                or cs.width != P3_CS_N or cs.name not in ("REQ", "NAK")
                or not 0 <= raster.cycles - cs.cycle <= ISS_GUARD_CYCLES
                or cs.at < end
                or any(lo < cs.at + cs.width and cs.at < hi
                       for lo, hi in self.keyings)):
            return None
        periods = (cs.at - start) // period
        if not end + periods * period <= cs.at <= start + (periods + 1) * period - cs.width:
            return None
        reference = self._breakin_reference(raster, slot)
        target = pulse + round((reference - pulse) / period) * period
        if (abs(target - reference) > round(BREAKIN_CLAMP_TOL_S * FS)
                or not cs.width <= target - cs.at <= ISS_GUARD_CYCLES * period):
            return None
        return target

    def _place_breakin(self) -> str:
        """Aim the changeover packet at the peer's transmission; say where.

        The aim is moved after the audio is rendered because `_flip` can still
        step the slot under it, and both protocols' changeover seams come
        through here for it.

        THE LINE NAMES THE INSTANT THE BURST ACHIEVED, and it names it against
        the packet ending it was placed on rather than against the reading that
        ending was projected from. Those are the same number only while the
        projection is fresh, and 2026-09-04's eighteen changeovers printed
        `+1322 ms` -- one peer cycle plus the lead -- for keyings the tape puts
        72 ms past a packet end. The onset and the cycle count go out with it so
        a capture can be walked back to the same instant.
        """
        retry = (None if self.raster is None else
                 self._p3_retry_phase(self.raster, self.slot))
        if retry is not None:
            self.unplaceable = None
            self.boundary, self.placed = retry, True
            return " on the emitted CS3 clock, corroborated by the peer's repeat request"
        end = (None if self.raster is None else
               self.raster.peer_packet_end(self.slot))
        if end is None:
            return ""
        self.unplaceable = (self.raster.breakin_refusal(end)
                            or self._clamp_refusal(end))
        if self.unplaceable is not None:
            return ""
        self.boundary = self._breakin_key(self.raster, self.slot)
        self.placed = True
        return (" +%.0f ms past the peer's packet end (its onset read at "
                "sample %d, +%.0f ms, %d cycle(s) on)"
                % ((self.boundary - end.at_slot) / FS * 1e3, end.at,
                   (end.end - end.at) / FS * 1e3, end.cycles))

    def _clamp_refusal(self, end: "_PeerEnd") -> Optional[str]:
        """The boundary clamp pulling the key in front of the reader.

        `_breakin_key` takes our own boundary over the read instant, and that
        is a placement only while the two agree -- 1.6 to 3.0 ms apart on
        2026-09-04's arms, at a reader that is a cliff inside a few ms. KB5LZK
        2026-09-07 21:57 CDT, TX[130] and TX[131]: `d` held at a 74 ms
        candidate put the read at +96 and the comb at +70, and both went out
        at +70, unread. `breakin_refusal` covers the released turnaround; this
        is the same rule with `d` held and the comb wrong instead.
        """
        gap = (self.raster.peer_read_gap if self.breakin_at_boundary
               else round(BREAKIN_LEAD_S * FS))
        early = end.at_slot + gap - self._breakin_reference(self.raster, self.slot)
        if early <= round(BREAKIN_CLAMP_TOL_S * FS):
            return None
        return (f"the peer reads {gap / FS * 1e3:.0f} ms past its packet and our "
                f"own boundary sits {early / FS * 1e3:.0f} ms in front of that, "
                f"past the {BREAKIN_CLAMP_TOL_S * 1e3:.0f} ms the reader forgives "
                f"-- the comb and the peer's ending disagree, and the boundary "
                f"is not a placement")

    def key_instant(self, raster: "_MasterGrid", slot: int) -> int:
        """The earliest sample this cycle's transmission may key at.

        The boundary for every burst but one. A changeover packet is placed
        against the peer's transmission and can want the channel up to 58 ms
        sooner, and which of the two the tick will key is not settled until the
        peer's packet has decoded -- inside the window this bounds. So the
        window and the collect in front of the key are measured from the
        earlier of the two, and the burst that turns out to be a control signal
        still keys on its own boundary, unmoved.
        """
        boundary = (self._breakin_reference(raster, slot) if self.breakin_due
                    else raster.boundary(slot))
        # P2's shaped pulse begins before its phase-reference boundary. Reserve
        # that lead in the receive window too: subtracting it only inside _tx
        # loses an otherwise reachable slot on every compatibility entry.
        pulse_lead = 0
        if self.host is not None and (
                self.host.protocol == spec.Protocol.PACTOR2 or
                (self.host.arq.entry_pending and
                 self.host.arq.entry_variant == "p2sl1")):
            pulse_lead = P2_KEY_LEAD_N
        elif self.host is not None and self.host.protocol == Protocol.PACTOR3 \
                and (self.host.arq.role == IRS or self.breakin_due):
            # CS3 replaces a control at its phase reference too. Its head uses
            # the case-0 pulse, including retries after ARQ becomes ISS.
            if self.breakin_due:
                pulse_lead = _p3_breakin_lead_bound(placement.PROTOCOL_RISE,
                                                   placement.CASE0_STAGGER)
            elif getattr(self, "p3_control_placement", "audio-start") == "audio-start":
                pulse_lead = 0
            else:
                tail, staggered = self._p3_control_shape()
                pulse_lead = max(
                    placement.control_burst_pulse_lead(cs, tail=tail, swapped=foot)
                    for cs in range(6)
                    for foot in ((False, True) if staggered else (None,)))
        boundary -= pulse_lead
        if not self.breakin_due:
            return boundary
        return min(boundary, self._breakin_key(raster, slot) - pulse_lead)

    # -- ArqIO peer seam --------------------------------------------------
    def _advance_aim(self, *, check_late: bool = False,
                     before_render: bool = False) -> None:
        """Point at the NEXT slot where the one we are aimed at is already spent.

        A second burst in one cycle belongs to the cycle after it. `arq.on_rx_cs`
        answers a control signal in the cycle it arrived in, and the flush that
        decodes one runs behind `host.tick()`, so that answer is rendered against
        an aim still naming the slot the tick has just keyed. `check_late`
        also checks unused slots, for a final ID after the session loop ends.
        """
        if self.raster is None or self.live is None:
            return
        spent = self.slots_used and self.slots_used[-1] >= self.slot
        # A decode can also consume an UNUSED slot. Catch the actual DAC clamp
        # before choosing polarity, while the next slot's shift is still free.
        # Do not demand the full settle here: normal holdback keys can have
        # 28–33 ms of PTT lead while their audio still lands on the boundary.
        # A placed break-in must instead be refused/re-placed against the peer.
        late = (self.live.clamp_late(self.boundary)
                if before_render and not spent and not self.breakin_due
                and self.boundary is not None else 0)
        # THE QUESTION THE GRID ALREADY ASKED OF THIS INSTANT. `_regrid` ran the
        # same overrun through `_clamp_forgives` a tick ago and kept the slot;
        # a bare `clamp_late` answered the other way in the very next line of
        # the log and took the cadence to two. Asking again is the conservative
        # half of the pair: the gate deducts the whole of `cycle_cost_n` a
        # second time although part of it has been spent since, so it cannot
        # allow here what `_tx` will refuse on the instant the carrier occupies.
        if late and _clamp_forgives(self.live, self, late):
            late = 0
        if spent or check_late or late:
            gone, gave = self.boundary, self.slot
            slot = self.raster.next_slot(self.slot) if spent or late else self.slot
            self.aim(self.raster, _keyable_slot(
                self.live, self.raster, slot, round(self.settle * FS), listen=False))
            if late:
                print(f"    [grid] SLOT {gave} {SLOT_UNRENDERED}. The unrendered "
                      f"burst would miss boundary {gone} by "
                      f"{late / FS * 1e3:+.1f} ms, past the "
                      f"{KEY_CLAMP_TOL_S * 1e3:.0f} ms the reader forgives less "
                      f"the {self.cycle_cost_n() / FS * 1e3:.1f} ms this cycle "
                      f"still owes the tick and the render; aiming at slot "
                      f"{self.slot} before choosing its shift.", flush=True)

    def _flip(self) -> bool:
        """The shift polarity for this transmission, as the grid has aimed it.

        THE AIM IS ADVANCED FIRST WHERE ITS SLOT IS ALREADY SPENT
        (`_advance_aim`), because otherwise the shift comes off the spent slot,
        and `_tx`'s backstop, holding finished audio, can only
        step in TWOS to keep the polarity it was handed: the packet lands one
        whole cycle after the one the peer is timing us against, every cycle,
        never drifting off it. The 2026-08-19 forced run is that
        session -- slots 10, 14, 18, 22, 26, 30 against a gateway keying every
        slot, twenty-one packets sent and the counter never past #1.

        Falls back to a per-transmission toggle only where there is no grid to
        count cycles on (a dry run), where nothing is listening.

        The answer is REMEMBERED as well as returned. Every renderer bakes the
        polarity into its samples here, while the burst is being built, and `_tx`
        is handed the finished audio -- so a re-aim inside `_tx` can move where
        the carrier comes up but cannot move which shift it comes up in. Keeping
        the rendered polarity is what lets the emission path pick a boundary that
        agrees with the audio it is holding.
        """
        self._advance_aim(before_render=True)
        if self.invert is not None:
            inv = self.invert
        else:
            inv = self.shift_inverted
            self.shift_inverted = not inv
        self._sent_invert = inv
        return inv

    def connect_burst(self, mycall: str, dxcall: str) -> None:
        # The PACTOR-1 connect names the *called* station
        # (docs/protocols/pactor/pactor-connect-frames.md).
        #
        # Every sync packet goes out in the SAME shift position. "Die Shiftlage
        # der FSK-Aussendung wird einmalig beim Verbindungsaufbau FIXIERT" -- it
        # is established once, at link setup, and only then does "mit jedem neuen
        # Paket oder Kontrollsignal wird die Shiftlage invertiert" take effect.
        # Alternating the calls themselves means the station being called sees a
        # different polarity on every attempt while it is still trying to lock,
        # and it is the thing that has to lock onto it. Doing so took answers
        # from 36-92 bursts a session to zero, twice running.
        #
        # Pinned, and then CONSUMED. Assigning the polarity without advancing it
        # left the connect and the first data packet in the same shift position,
        # which is the one transition the alternation is not allowed to skip:
        # off-air, five consecutive bursts from a real station invert strictly,
        # every cycle, with no exception at the handover into the data phase.
        #
        # The first call goes out BEFORE the grid exists -- it is the thing the
        # anchor is measured from -- so it takes the dry-run toggle and is slot 0
        # by construction, non-inverted. Every call after it is aimed like any
        # other transmission. Pinning an epoch here as well made the first two
        # calls share a shift, and a peer that locked to the first then read every
        # later transmission one cycle out of phase, forever.
        # The host's connect operator is read HERE, as the burst is rendered, the
        # way `entry_variant` is: `C %CALL` keys the branch-B robust frame and
        # every retry resends the call that was asked for. A host-less RadioTx is
        # a bench rig and keys the ordinary call.
        variant = "normal" if self.host is None else self.host.arq.connect_variant
        self._tx(pactor1.connect_signal(dxcall, invert=self._flip(),
                                        variant=variant),
                 f"connect->{dxcall}", drive=self.drive * self.p1_drive)

    p3_follow_offset = "all"
    """Which PACTOR-3 transmissions follow the peer's measured carrier offset.

    `--p3-follow-offset`, and it ships on. WS8EOC's PACTOR-3 carriers sat 64 Hz
    above ours on 2026-09-13 at 13:16 while its own FSK stayed within 3 Hz of
    nominal, so this is not a dial: a 40 m gateway's PACTOR-3 raster is its own,
    and it has been measured between -75 and +64 Hz across five arms of one day.
    `p3acquire` accepts a control only within about 8 Hz of true, and our reader
    finds a displaced peer solely because `CONTROL_OFFSETS_HZ` sweeps +-75 Hz.
    A peer sitting on its own raster has no such sweep to spare, so our
    nominal-keyed CS1 arrived 64 Hz out and was never read; the gateway repeated
    its changeover 31 times and signed off. A 14-carrier SL3 packet on 120 Hz
    spacing survives it even less well -- VE3KPG at +50 Hz answered 38 of our
    data packets with CS1 and accepted none.

    "control" is the narrower experiment: codewords follow, packets stay at
    nominal. "none" is the negative control that keys everything where every arm
    before 2026-09-13 keyed it. All 267 header lines recorded
    before 2026-09-16 flew "all"; the other two have never been on
    the air, and the assessed launcher's `follow-offset` profile variable is
    there to pair them against it at one gateway in one hour.

    WHATEVER THIS SAYS, `_p3_offset` STILL OWES THE PACTOR-1 CROSS-CHECK. A
    measured raster and a per-symbol phase convention are the same reading to a
    differential acquisition, and only the FSK leg of the same link can tell them
    apart -- `_SessionRx._follow_p3_offset`.
    """

    p3_keep_slots = "all"
    """Which cycles the grid may leave a sub-symbol overrun to the emission path.

    `--p3-keep-slots`, and it ships on "all", which is `_clamp_forgives` as
    rounds 14 and 16 left it. The two narrower values are the flyable halves of
    that pair, because the runtime that keys nearly every slot is also the one
    WS8EOC stops acknowledging at SL1 seq=1 while the runtime that gave slots
    away delivered the whole greeting: "controls" restores round 16's exclusion
    so a changeover cycle gives its slot away, and "none" is the gate off
    entirely -- every overrun spends its slot, which is every arm up to v9.
    """

    _tx_offset_hz: Optional[float] = None
    """Offset the burst in front of `_tx` was rendered at, or None for a
    transmission this correction does not apply to. Consumed by `_tx` the way
    `_sent_invert` is, so a refused render cannot label the next burst."""

    prekey_cost_n: int = 0
    """What the cycle last spent between the GRID's admission check and the
    EMISSION path's own -- the tick, the render, the channel guard and the
    drain -- on the stream's own clock.

    Measured rather than assumed, because it is what `_regrid` has to allow for
    before it may leave a sub-symbol overrun to `_tx` (`_clamp_forgives`), and
    because it is the one figure that says whether work taken off the pre-key
    path actually came off it.
    """

    breakin_cost_n: int = 0
    """The same interval, measured on a CHANGEOVER cycle, kept apart from it.

    A changeover cycle does everything an ordinary one does and then renders a
    56-row packet and measures a 2% trim on the samples it actually keys: 0.97
    and 0.69 ms on this bench against a codeword whose render is cached and
    free, and 8.2 ms on `arm-v23-A-40-ws8eoc`, where the tick behind it
    assembles the B2F login. `_prekey_lead` stands this off the drain on a
    changeover cycle, so the interval the cycle needs is one somebody measured
    rather than the remainder of the settle.
    """

    _admitted_at: Optional[int] = None
    """Where the stream stood when the grid last admitted this cycle. Consumed
    by `_tx`, so a slot the grid handed back cannot charge its whole listening
    window to the render."""

    def cycle_cost_n(self) -> int:
        """What THIS cycle still owes between the grid's check and the key.

        The larger of the two measurements on a changeover cycle, because such
        a cycle does everything an ordinary one does and then some, and a figure
        that under-counts is the one that spends two slots.
        """
        return (max(self.prekey_cost_n, self.breakin_cost_n)
                if self.breakin_due else self.prekey_cost_n)

    def _p3_offset(self, audio: np.ndarray, *, control: bool,
                   ident: Optional[tuple] = None) -> np.ndarray:
        """Move a rendered PACTOR-3 burst onto the peer's own carrier raster.

        `p3acquire.compensate(x, hz)` shifts a spectrum DOWN by `hz`; it is used
        on receive as `compensate(seg, +offset)` to bring a high-arriving peer
        back to nominal. Keying high is therefore its negation, and the whole
        correction is self-cancelling -- a peer on frequency moves nothing.

        WHAT MAY MOVE THE TRANSMITTER. `p3_receive_offset_hz` is written in two
        places and both have already paid for it: `_p3_cs` stores a control read
        that either carried a CRC-valid body or repeated at the same frequency
        and cycle phase on a distinct cycle, and `_p3_changeover_packet` stores
        only a CRC-valid changeover. A bare first head is held in
        `_p3_head_candidate` and never reaches here, which matters because the
        coarse grid's +-100 Hz alias also decodes: a control keyed onto an alias
        is worse than one keyed at nominal. Clamped to the span the reader
        itself sweeps, so a stored value from anywhere cannot key us off band.

        AND THAT WAS NOT ENOUGH ON ITS OWN. Both writers go through
        `_SessionRx._follow_p3_offset`, because a CRC-valid decode still lands
        on the alias sometimes: `captures/onair-0913-2152` read one frame at
        +75.2 Hz against a peer that sat at -24.5 all arm, and `tx_43` went out
        at +75.0. A jump of more than one coarse step is now held as a candidate
        until a second frame agrees with it, so what reaches this transform is
        an offset two decodes have named.

        AND IT IS `p3_transmit_offset_hz` THAT IS READ HERE, NOT THE RECEIVE
        RASTER. The two differ only where a PACTOR-1 codeword read this session
        contradicts the PACTOR-3 acquisition past `p3acquire.P1_CROSS_CHECK_HZ`,
        which is the one case where following costs the reverse channel and buys
        nothing -- see `_SessionRx._follow_p3_offset`. A session object that
        carries no such attribute keys at its receive offset, as before.

        KEPT UNDER `ident`, WHICH IS THE BURST'S IDENTITY. A granted station
        keys the SAME entry packet every cycle until the peer reads one, and an
        ISS holding a link repeats a codeword and re-sends an unacknowledged
        packet, so the transform below ran again on byte-identical samples in
        the milliseconds in front of every one of those keys. Named by what it
        was built from, it runs once per burst per offset and every repeat of it
        is free -- and the first build of a burst the peer has never seen is the
        only one left on the critical path.
        """
        if self.p3_follow_offset == "none" or (self.p3_follow_offset == "control"
                                               and not control):
            self._tx_offset_hz = 0.0
            return audio
        bound = max(p3acquire.CONTROL_OFFSETS_HZ)
        hz = getattr(self.sessrx, "p3_transmit_offset_hz", None)
        if hz is None:
            hz = getattr(self.sessrx, "p3_receive_offset_hz", 0.0)
        hz = max(-bound, min(bound, float(hz or 0.0)))
        self._tx_offset_hz = hz
        if not hz:
            return audio
        kept = self._p3_shifted.get(ident) if ident is not None else None
        if kept is not None and kept[0] == hz:
            return kept[1]
        # PADDED TO A LENGTH THE TRANSFORM LIKES, because a burst nothing has
        # keyed before still pays for it here. An SL3 packet is 41219 samples
        # and 41219 is 47 x 877, so the analytic signal costs 2.2 ms of the
        # roughly 8 ms a cycle has between its last read and the admission
        # check -- more than the render it follows. At 41250 it costs 0.9. The
        # tail is trimmed straight back off, so what is keyed is the same
        # samples either way.
        n = len(audio)
        moved = p3acquire.compensate(
            np.pad(audio, (0, next_fast_len(n) - n)), -hz)[:n]
        if ident is not None:
            self._p3_shifted[ident] = (hz, moved)
            while len(self._p3_shifted) > P3_SHIFT_KEEP:
                del self._p3_shifted[next(iter(self._p3_shifted))]
        return moved

    def _p3_control_shape(self) -> tuple[bool, bool]:
        """(trailing repeat symbol, staggered carriers) for this station's controls.

        `--p3-control-waveform current` is the whole measured template and
        carries both; `--p3-control-tail` and `--p3-control-stagger` add either
        to the synchronous default, which is what makes them separable on the
        air. The two contribute independently to fit against real emitters --
        0.741-0.763 with neither, 0.760-0.784 with the tail alone, 0.866-0.887
        with the stagger alone, 0.887-0.908 with both on the right foot.
        """
        current = getattr(self, "p3_control_waveform", "historical") == "current"
        return (current or getattr(self, "p3_control_tail", "off") == "repeat",
                current or getattr(self, "p3_control_stagger", "off") != "off")

    def _p3_stagger_foot(self, grid_swap: bool) -> bool:
        """This cycle's carrier arrangement, GENERATED rather than inferred.

        Real emitters alternate the leading carrier strictly every ARQ cycle --
        24 consecutive alternations on one station with no exception -- so the
        arrangement needs no evidence from the peer, only a starting foot and a
        cycle count. `grid_swap` is slot parity, which is that count; the first
        control keyed anchors it, so `--p3-control-stagger` names the foot of
        burst one and every burst after it follows from the grid.

        THE FOOT IS THE WHOLE RISK. A synchronous burst sits at most half a
        symbol from either arrangement, but the wrong arrangement is a FULL
        symbol out on one tone -- 6.4 raw errors of 20 against a code that
        corrects 5, and 84 of 486 modelled trials landing on the wrong codeword.
        That is why it is a settable binary and why every burst prints it.

        The anchor is committed by the EMISSION and not by this call, because a
        burst the guards decline still spends its slot: anchoring on a refused
        render leaves the next parity in front of us and the first burst that
        reaches the air on the foot the flag did not name.
        """
        anchor = (grid_swap if self._p3_stagger_anchor is None
                  else self._p3_stagger_anchor)
        return (getattr(self, "p3_control_stagger", "off") == "lead-12") ^ (
            grid_swap != anchor)

    def _p3_control_label(self, cs_index: int, swapped: Optional[bool],
                          p1: bool = False) -> str:
        """The TX[n] label, and with it the burst sidecar's `control` field.

        Which foot a staggered burst went out on is not recoverable from a tape
        without this, because the alternation is ours and there is nothing to
        compare it against. CS1/CS2 also name the packet they acknowledge;
        their static ACK/REQ names alone misdescribe the odd-counter ACK.
        """
        name = f"CS{cs_index + 1} {spec.CS_NAMES[cs_index]}"
        if p1:
            return f"{name} [pactor-1]"
        if self.host is not None and cs_index in (0, 1):
            name = (f"CS{cs_index + 1} "
                    f"{traffic.TrafficLog.control_meaning('TX', cs_index, self.host.arq)}")
        shape = []
        if getattr(self, "p3_control_stagger", "off") != "off":
            shape.append("lead-12" if swapped else "lead-5")
        if getattr(self, "p3_control_tail", "off") == "repeat":
            shape.append("tail")
        return f"{name} [{', '.join(shape)}]" if shape else name

    def p3_reply_shift(self, slot: int, cs_index=None):
        """Place both timing arms before sizing RX, and again before emission."""
        r, trial = self.raster, self.timing_trial
        target = None
        if (trial is not None and trial.opened is not None and not trial.closing
                and trial.reason is None
                and r.protocol == Protocol.PACTOR3 and not r.sending):
            ci = self._pending_p3_cs if cs_index is None else cs_index
            ci = CS_ACK if ci is None else ci
            raw = placement.control_burst(ci, tail=False, swapped=None)
            audio = self._p3_offset(raw, control=True, ident=("cs", ci, False, None))
            lead = placement.pulse_lead(audio)
            if r._p3_peer is not None and trial.entry_phase is not None:
                phase = r._peer_raster_position(r._p3_peer[0], r.cycle_n) or r._p3_peer[0]
                epoch = trial.target(phase, lead)
                if trial.arm == "B":
                    target = epoch
        if (self.reply_clock is not None and r.protocol == Protocol.PACTOR3
                and not r.sending and r._p3_peer is not None):
            ci = self._pending_p3_cs if cs_index is None else cs_index
            ci = CS_ACK if ci is None else ci
            raw = placement.control_burst(ci, tail=False, swapped=None)
            audio = self._p3_offset(raw, control=True, ident=("cs", ci, False, None))
            target = self.reply_clock.target(placement.pulse_lead(audio))
        return (r.p3_reply_shift(slot) if target is None else
                r.p3_reply_shift(slot, target=target))

    def _timing_trial_refusal(self, delay=0):
        trial = self.timing_trial
        if trial is None or trial.closing:
            return False
        _entry_trial_control_confirmation(self, self.host)
        now = self.live.sample_now() if self.live is not None else 0
        scheduled = ((self.boundary or now) + delay + min(self.tx_pulse_offsets or (0,)))
        trial.check(max(now, scheduled))
        if trial.reason is None:
            return False
        self.refused = True
        print(f"    [p3 trial] scored TX refused: {trial.reason}", flush=True)
        return True

    def _p3_reply_carrier_order(self, phase: int, fallback: bool) -> bool:
        """Project the peer's CRC-confirmed arrangement onto a reply's cycle.

        PIII_Complete_1 has the caller's CS3 at 7.7681 s opposite the preceding
        peer packet, and the answerer's CS3 at 64.9160 s retaining its peer's
        order. Ordinary staggered controls follow the same origin-dependent
        rule. Packet counters and our arbitrary local slot parity cannot set it.
        """
        r = self.raster
        if r is None or r._p3_peer is None or r._p3_peer_swap is None:
            return fallback
        at, _, cycle_n, _ = r._p3_peer
        cycles = max(0, (phase - at) // cycle_n)
        swapped = r._p3_peer_swap ^ bool(cycles & 1)
        if self.host is not None and not self.host.arq.answering:
            swapped = not swapped
        return swapped

    def _send_p3_control(self, cs_index: int, *, p1_codeword: bool = False):
        r = self.raster
        if r is not None:
            # THE RESIDUE, NOT THE PLACEMENT. Both loops place the comb where
            # the cycle's window is sized (`_p3_place_reply`, at the top of the
            # cycle and again in `_regrid`), so what is left here is sub-slot
            # and forward: a backward move of any size at this point names a
            # boundary the clock has already passed. Kept because it is the
            # last thing in front of the render -- `_flip` settles the boundary
            # and the polarity this burst is built for -- and because a driver
            # can be keyed without a loop around it. The shift is counted in
            # slots, so moving the anchor cannot change the polarity below.
            moved = self.p3_reply_shift(self.slot, cs_index)
            if moved is not None:
                print(f"    [grid] {moved}", flush=True)
                self.aim(r, self.slot)
        grid_swap = self._flip()  # Re-aim before choosing the waveform's order.
        trial = self.timing_trial
        if trial is not None and trial.active:
            raw = placement.control_burst(cs_index, tail=False, swapped=None)
            moved = self._p3_offset(raw, control=True,
                                     ident=("cs", cs_index, False, None))
            if not trial.attempt(r.boundary(self.slot) + placement.pulse_lead(moved)):
                self.refused = True
                return REFUSED
        swapped = grid_swap
        if r is not None:
            why = r.p3_control_refusal(self.slot)
            if why is not None:
                self.refused = True
                print(f"    !! P3 REPLY CLOCK -- NOT KEYING CS{cs_index + 1}: "
                      f"{why}.", flush=True)
                return REFUSED
        foot = getattr(self, "p3_control_stagger", "off")
        if p1_codeword:
            # THE QUESTION IS THE CODEWORD, NOT THE EMISSION. A gateway reads
            # this renderer: WS8EOC has obeyed our PACTOR-3 CS4 sixteen times,
            # fifteen of the sixteen peer speed rises on record follow one of
            # ours within six cycles, every rise exactly +1 and to the level
            # asked for -- and on all sixteen its mod-4 counter stepped by
            # exactly one, which is a gear command carrying that cycle's
            # acknowledgement (pactor3.md, "The acknowledgement is the
            # alternation"). What a peer stuck at seq=0 is not advancing on is
            # our PACTOR-3 CS1: same carriers, same modulation, different
            # codeword.
            #
            # SO ASK IT WITH THE CODEWORD THAT HAS A RECORD. A counter-0
            # changeover carrying `RMS Tri` is acknowledged CS1 and a repeat is
            # asked for with CS2 (`pactor1-control-signals.md` 4.2), and three
            # arms keyed that answer as a PACTOR-1 codeword: KB5LZK 2026-08-22
            # CS1 and the whole banner crossed, counters 0,1,2,3,0; WS8EOC
            # 2026-08-30 arms 12/13 CS1 and the peer advanced to counter 1;
            # VE1YZ 2026-09-02 CS2 and the peer repeated the identical packet 45
            # more times. So the link stays at PACTOR-3 and the alternation
            # stays the packet counter's; only the twenty DBPSK symbols become
            # twelve FSK bits, in the PACTOR-3 answer slot the peer is
            # listening in.
            #
            # AND IT IS OUT OF SPEC, which is why it is off by default:
            # pactor3.md sec 6 has every control signal DBPSK on channels 5 and
            # 12, and the PACTOR-1 codewords as a separate adjacent table.
            #
            # AND AT NOMINAL, without `_p3_offset`. The far end reads PACTOR-1
            # by correlating fixed 1400/1600 Hz lines, where a few hertz of a
            # 200 Hz shift is nothing, and the transform is clamped to +-75 Hz
            # -- 60 of which survive the PACTOR-1 cross-check -- which would put
            # MARK a third of the way to SPACE.
            audio = pactor1.control_signal(cs_index, msb_first=self.p1_ack_msb,
                                           invert=grid_swap)
            swapped, lead, offsets = None, 0, None
            drive = self.drive * self.p1_drive
            self._tx_offset_hz = 0.0
        else:
            drive = None
            tail, staggered = self._p3_control_shape()
            if foot != "off":
                swapped = self._p3_stagger_foot(grid_swap)
            elif not staggered:
                swapped = None
            elif r is not None:
                swapped = self._p3_reply_carrier_order(r.boundary(self.slot), grid_swap)
            audio = placement.control_burst(cs_index, tail=tail, swapped=swapped)
            lead = placement.control_burst_pulse_lead(cs_index, tail=tail,
                                                      swapped=swapped)
            if swapped is None:
                # Both tones are identical: a later slot needs no carrier swap.
                self._sent_invert = None
                offsets = (lead, lead)
            else:
                half = FS // 200
                offsets = (lead + half, lead) if swapped else (lead, lead + half)
            moved = self._p3_offset(
                audio, control=True, ident=("cs", cs_index, tail, swapped))
            if moved is not audio:
                # MEASURED ON WHAT IS KEYED. The 2% trim is an instantaneous test
                # and the carriers have just moved under it, so the codeword's
                # cached lead can name a sample the driver will not trim to --
                # CS6 by 47 of them at -60 Hz. Re-measuring keeps the stagger and
                # keeps the sidecar's pulse centers describing this burst.
                audio, shift = moved, placement.pulse_lead(moved) - lead
                lead += shift
                offsets = tuple(n + shift for n in offsets)
        # Keep measured pulse bookkeeping independent of the experimental
        # scheduling reference. Audio-start placement must not report its
        # padding edge as the emitted symbol center.
        aim_lead = (0 if p1_codeword
                    or getattr(self, "p3_control_placement", "audio-start")
                    == "audio-start" else lead)
        before = self.n
        self._tx(audio, self._p3_control_label(cs_index, swapped, p1_codeword),
                 drive=drive, lead_n=aim_lead, pulse_offsets=offsets)
        if self.refused or self.n == before:
            return REFUSED
        if not p1_codeword and cs_index == 3 and self.sessrx is not None:
            self.sessrx.note_p3_speedup_emitted()
        if not p1_codeword:
            self._log_p3_emission(self.traffic.control(
                "TX", cs_index, None if self.host is None else self.host.arq))
        if trial is not None and trial.opened is not None and not trial.closing:
            trial.replies.append(dict(tx=self.n, slot=self.slot, cs=cs_index+1,
                audio_start=self.tx_audio_start, pulse_offsets=list(offsets or ()),
                entry_phase=trial.entry_phase,
                reply_delay_ms=trial.reply_delay_ms,
                peer_phase=r._p3_peer[0] if r._p3_peer else None))
        if not p1_codeword and foot != "off" and self._p3_stagger_anchor is None:
            self._p3_stagger_anchor = grid_swap
        if r is not None and self.tx_audio_start is not None:
            # THE REFERENCE THE BURST WAS AIMED BY, which is what every consumer
            # compares against `boundary`: `_observe_p3_reply_timing` ->
            # `peer_read_gap` -> `_breakin_key` -> `_clamp_refusal`, at a 5 ms
            # tolerance. Recording the pulse center under audio-start placement
            # inflates the read instant by the 17.75 ms lead, and 0912-2300
            # then refused every changeover "about 18 ms before the calculated
            # position".
            # Mail's CS3 placement compares against the control pulse epoch;
            # both ends of that measurement must use the emitted pulse. The
            # legacy policy above still uses its audio-start reference.
            reference_lead = (min(offsets) if getattr(self, "reply_clock", None)
                              is not None and not p1_codeword and offsets
                              else aim_lead)
            r.note_p3_control(self.tx_audio_start + reference_lead)
        if cs_index == CS_CYCLE_TOG and r is not None:
            # Preserve the command's actual phase through missed probes and
            # regrids: it dates the long-frame hypothesis `_p3_transition_window`
            # retains and reads. It grants no reply boundary and refuses none --
            # the comb belongs to the peer's packets (`_keyable_slot`).
            r._p3_command_slot = self.slot
            if r._p3_peer is not None:
                at, _, cycle_n, _ = r._p3_peer
                cycles = max(0, (r.boundary(self.slot) - at) // cycle_n)
                r._p3_command_row0 = (at + (cycles + 1) * cycle_n
                                     + p3frame.DATA_OFFSET * rxfront.SPS)
        return REFUSED if self.refused else None

    def send_cs(self, cs_index: int, *, p1_codeword: bool = False):
        if self.defer_p3_cs:
            # Receiving a frame may update ARQ while we are still collecting
            # this cycle. Emitting here would wait through that collection and
            # flush it under our carrier. The hold loop emits the final answer.
            self._pending_p3_cs = cs_index
            self._pending_p3_cs_p1 = p1_codeword
            return
        return self._send_p3_control(cs_index, p1_codeword=p1_codeword)

    def cancel_pending_cs(self) -> None:
        self._pending_p3_cs = None

    def recover_p3_turn(self, ev) -> bool:
        """Restore the old IRS clock before ARQ handles a failed turn attempt.

        A repeated peer CS3 on its old raster cannot acknowledge our own CS3.
        The host owns that unconfirmed-packet fact; this driver supplies the
        independently measured phase and the saved pre-emission clock.
        """
        r = self.raster
        if (r is None or self.host is None
                or not self.host.arq.unconfirmed_breakin):
            if r is not None:
                r._p3_turn = None
            return False
        if (ev.protocol != Protocol.PACTOR3 or not ev.breakin
                or ev.packet is None or not ev.packet[3]):
            return False
        identity = (ev.breakin, *ev.packet[:3])
        return r.recover_p3_turn(round(ev.t * FS), identity)

    def emit_pending_cs(self) -> None:
        cs, self._pending_p3_cs = self._pending_p3_cs, None
        if cs is None or self.host.protocol != Protocol.PACTOR3 \
                or self.host.arq.role != IRS:
            return
        previous_end = self.tx_end
        self._send_p3_control(cs, p1_codeword=self._pending_p3_cs_p1)
        if not self.refused and self.tx_end != previous_end:
            self.host.on_cs_emitted(cs)
            return
        # HELD, NOT DROPPED. A refusal names one instant, and the answer it
        # declined is still owed: the IRS owes the ISS a codeword every cycle.
        # Clearing the queue ahead of the emit lost that cycle's answer with no
        # line naming it. ARQ overwrites this the moment it has a newer word.
        self._pending_p3_cs = cs
        print(f"    [grid] CS{cs + 1} {spec.CS_NAMES[cs]} was not placed this "
              f"cycle; held for the next keyable reply boundary.", flush=True)

    p1_ack_msb = False
    p1_status_bits45 = 0
    p1_status_from = 1

    def send_p1_cs(self, index: int) -> None:
        # Measure from the BURST that stopped the listen, which is what the cycle
        # is timed against; measuring from the window close hides a break that
        # fired late in it.
        #
        # In samples where there are samples: the onset and our own carrier drop
        # are both indices in one continuous stream, so the interval between them
        # is exact. The clock reading is the fallback for a dry run, and it
        # carries the scheduler's jitter with it.
        #
        # THE INTERVAL IS EXACT AND ITS ENDPOINT IS NOT, so the line says which is
        # which. `p1_reply_starting` set this stamp, and that test asks only
        # whether the 1400/1600 pair is up right now, at `P1_STARTING_EXCESS`
        # = 1.4x the guard bands -- deliberately looser than the thresholds that
        # gate a line an operator reads, because its own job is to stop listening
        # and a false stop costs one window. It reads no bits and names no
        # station. Printed as "after the peer's burst" it offered that looseness
        # as a measurement of a peer, which is the failure its own constant is
        # commented against.
        at = getattr(self.host, "_burst_at", None) if self.host else None
        n = getattr(self.host, "_burst_sample", None) if self.host else None
        loose = "onset by energy at 1.4x, unattributed"
        if n is not None and self.live is not None:
            d = (self.live.samples - n) / FS
            since_tx = (f", {(n - self.tx_end) / FS:.3f} s after our carrier "
                        f"dropped" if self.tx_end is not None else "")
            print(f"    [timing] ACK keyed {d:.3f} s after the tone pair came up "
                  f"({'INSIDE' if d < 1.25 else 'OUTSIDE'} the 1.25 s cycle)"
                  f"{since_tx} -- {loose}", flush=True)
        elif at is not None:
            d = time.time() - at
            print(f"    [timing] ACK keyed {d:.3f} s after the tone pair came up "
                  f"({'INSIDE' if d < 1.25 else 'OUTSIDE'} the 1.25 s cycle) "
                  f"-- {loose}, on the wall clock", flush=True)
        elif getattr(self, "reply_at", None) is not None:
            print(f"    [timing] ACK keyed {time.time() - self.reply_at:.3f} s "
                  f"after the window closed; burst time unknown", flush=True)
        # The PACTOR-1 link-layer acknowledgement. A peer answers our SELCALL with
        # one of these and expects one back before the link is up; jumping to a
        # PACTOR-3 data packet here -- which is what shrike did on its first
        # connect -- leaves the far end still waiting on the P1 handshake.
        self._tx(pactor1.control_signal(index, msb_first=self.p1_ack_msb,
                                        invert=self._flip()),
                 f"P1 CS{index + 1} {'MSB' if self.p1_ack_msb else 'LSB'}",
                 drive=self.drive * self.p1_drive)

    def _p1_nth(self, packet_count: int, *, breakin: bool = False) -> int:
        """Which DISTINCT data packet this is, 1-based -- a repeat keeps the last.

        The mod-4 counter changing is the air's own record of new information:
        "bei jedem Paket, das neue Information enthaelt, wird das Bitmuster
        invertiert", and the header is that counter's low bit. `arq` builds a
        status byte once in `_start_next_packet` and re-sends the packet in
        flight unchanged on a NAK, so a repeat arrives here under the counter it
        repeats. A break-in RESETS the counter and can land back on the value we
        last keyed, so it says outright that it is new.
        """
        if breakin or not self.p1_seq or self.p1_seq[-1] != (packet_count & 3):
            self.p1_packets += 1
        return self.p1_packets

    def _note_counter(self, packet_count: int, *, breakin: bool = False) -> None:
        """Record one data packet's counter, whatever protocol carried it.

        The end-of-session account is about the LINK, and the link changes
        protocol under it. A changeover is marked because it restarts the
        counter, so the step it makes is not an acknowledgement's.
        """
        if breakin:
            self.breakin_at.add(len(self.seq_sent))
        self.seq_sent.append(packet_count & spec.STATUS_SEQ)

    def send_p1_packet(self, payload: bytes, baud: int, packet_count: int, *,
                       header: int | None = None,
                       changeover_request: bool = False,
                       qrt: bool = False) -> Optional[int]:
        """A PACTOR-1 data packet -- the caller's answer to a control signal."""
        nth = self._p1_nth(packet_count)
        bits45 = self.p1_status_bits45 if nth >= self.p1_status_from else 0
        if bits45 and not self._p1_bits45_keyed:
            self._p1_bits45_keyed = True
            status = pactor1.status_byte(packet_count & 3, bits45=bits45,
                                         changeover_request=changeover_request,
                                         qrt=qrt)
            print(f"    [status] packet {nth} of the session is the first with "
                  f"bits 4-5 set: status 0x{status:02x}", flush=True)
        self.p1_seq.append(packet_count)
        self._note_counter(packet_count)
        flags = ("" if not changeover_request else " BK") + ("" if not qrt else " QRT")
        self._tx(
            pactor1.packet_signal(payload, baud, packet_count=packet_count,
                                  header=header, bits45=bits45,
                                  changeover_request=changeover_request,
                                  qrt=qrt, invert=self._flip()),
            f"P1 pkt#{packet_count} {baud}Bd {len(payload)}B{flags}",
            drive=self.drive * self.p1_drive)
        return REFUSED if self.refused else None

    def send_p1_breakin(self, payload: bytes, baud: int, packet_count: int, *,
                        qrt: bool = False) -> Optional[int]:
        """The changeover packet: CS3 as its head, 840 ms of our own data behind.

        Rendered here rather than as a control signal because that is what it is.
        A bare CS3 burst went out for shrike's whole on-air history, and a station
        that heard one had nothing to switch to receive FOR -- it is the packet
        that carries the break-in, and the codeword only marks its first 120 ms.

        AND IT IS THE ONE BURST NOT KEYED ON THE BOUNDARY. See `_breakin_key`
        and `_place_breakin`.
        """
        self._p1_nth(packet_count, breakin=True)
        self.p1_breakin_at.add(len(self.p1_seq))
        self.p1_seq.append(packet_count)
        self._note_counter(packet_count, breakin=True)
        audio = pactor1.breakin_signal(payload, baud, packet_count=packet_count,
                                       qrt=qrt, invert=self._flip())
        placed = self._place_breakin()
        self._tx(audio,
                 f"P1 BREAK-IN #{packet_count} {baud}Bd {len(payload)}B"
                 f"{'' if not qrt else ' QRT'}{placed}",
                 drive=self.drive * self.p1_drive)
        self.breakin_due = False
        return REFUSED if self.refused else None

    entry_delay_n = round(DEFAULT_ENTRY_DELAY_MS * FS / 1000)
    """Samples the entry packet is keyed PAST its slot boundary: 234 by default.

    The 4.875 ms delay restores the time removed by trimming the entry waveform.
    Stock transitions place the pulse around 19 ms after the P1 cycle boundary;
    our zero-delay entries were around 13 ms. The September 19 delayed trials
    reached P3 controls after two entries at both VE3KPG and K0NTS,
    with the pulse positions checked against captured PCM.
    `--p3-entry-delay 0` retains the historical epoch for comparison.

    A delay is a NEGATIVE lead: `_tx`'s `lead_n` moves the carrier in front of
    the boundary, so the entry hands it the delay's negation and every guard
    downstream -- the admission reserve, `key_refusal`, `clamp_late` and the
    slot search after a late key -- reasons about the instant that is actually
    keyed. Nothing is relaxed to make room for it; a delay only ever moves the
    carrier later, which is the direction all of them already permit.
    """

    def send_entry_packet(self, sl: int, payload: bytes, status: int, *,
                          acquire: bool = False) -> int:
        """The first PACTOR-3 packet a granted link keys, in the one arrangement
        a peer has ever been recorded reading.

        NOT `_flip()`'s. The carrier swap alternates cycle by cycle on an
        established link, and taking the entry packet's from the same parity
        meant this station keyed two different packets at four grants -- variable
        header 0 in one cycle and header 1 with the carriers exchanged in the
        next, where DL6MAA keyed one entry, unswapped, at variable header 0.
        `p3frame.variable_header` makes that header the swap: speed level 1,
        short cycle, unswapped IS header 0, so pinning the one pins the other.

        The aim still advances (`_advance_aim`) because a spent slot is spent
        whatever the burst is, but nothing is remembered in `_sent_invert`: this
        packet bakes no polarity in, so a late key is free to take the next
        boundary rather than the next but one.

        AND IT ENDS ITS TRELLIS WHERE THE REFERENCE ENDS ITS OWN. This is the one
        packet whose every symbol a granting peer can predict, so the eight bits
        after the field are not slack -- `placement.ENTRY_FLUSH` carries what was
        measured off DL6MAA's and what a zero flush cost.

        `entry_delay_n` restores the leading time removed by the transmit trim;
        it changes the scheduled start without changing the entry samples.
        """
        self._advance_aim()
        self._note_counter(status)
        audio, field = self._entry_burst(sl, payload, status, acquire)
        audio = self._p3_offset(audio, control=False,
                                ident=("entry",) + self._entry_render[0])
        # This marker belongs to the exact frequency-corrected, trimmed entry.
        # Its audio-start boundary differs from a control's by several ms.
        trial = self.timing_trial
        entry_lead = None
        if trial is not None or self.reply_clock is not None:
            trim = int(np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0])
            entry_lead = int(np.argmax(placement.protocol_config().pulse())) - trim
        previous_end = self.tx_end
        late = (f" +{self.entry_delay_n / FS * 1e3:.1f}ms" if self.entry_delay_n
                else "")
        self._tx(
            audio,
            f"SL{sl} ENTRY {len(payload)}B{' +burst' if acquire else ''}{late} "
            f"P3 status=0x{status:02x} field={field.hex()}",
            lead_n=-self.entry_delay_n,
            pulse_offsets=None if entry_lead is None else
            (entry_lead, entry_lead + (FS//200 if placement.CASE0_STAGGER else 0)))
        if (trial is not None and not self.refused and self.tx_end != previous_end
                and self.tx_audio_start is not None):
            trial.entry(self.tx_audio_start + entry_lead, self.n)
        if (self.reply_clock is not None and not self.refused
                and self.tx_end != previous_end and self.tx_audio_start is not None):
            self.reply_clock.entry(self.tx_audio_start + entry_lead)
            if self.raster is not None:
                # Entry is always physically unswapped, regardless of this
                # local grid's parity. Ordinary ISS traffic must alternate
                # from that actually emitted arrangement. A later emitted
                # entry renews the reference; a refused attempt cannot.
                self._p3_packet_swap_bias = self.raster.shift(self.slot)
        self._log_p3_packet(sl, payload, status, kind="ENTRY")
        return (REFUSED if self.refused else
                placement.SPEED_PATHS[sl].crc_bytes - 3)

    def _entry_burst(self, sl: int, payload: bytes, status: int,
                     acquire: bool) -> tuple[np.ndarray, bytes]:
        """The entry waveform and its field, built once and kept.

        A granted station keys the SAME packet until the peer reads one or the
        budget runs out -- `status=0x1a field=0f8f87c7c31a6689` eight times on
        2026-09-10, byte-identical every cycle -- and it was being encoded,
        walked, staggered and pulse-shaped again each time, inside the window in
        front of the key, alongside a second `build_field` kept only for the log
        line. One slot of the campaign is one keying, so one build is all the
        campaign owes. The module switches are in the key because they are read
        at render time and a command line sets them.
        """
        key = (sl, payload, status, acquire,
               placement.CASE0_STAGGER, placement.PROTOCOL_RISE)
        if self._entry_render is None or self._entry_render[0] != key:
            path = placement.SPEED_PATHS[sl]
            self._entry_render = (
                key,
                placement.link_packet(sl, payload, status, swapped=False,
                                      acquire=acquire,
                                      flush=placement.ENTRY_FLUSH),
                placement.build_field(
                    placement.field_info(payload, path.crc_bytes - 3, status),
                    path))
        return self._entry_render[1], self._entry_render[2]

    def send_p4_entry_packet(self, payload: bytes, status: int) -> Optional[int]:
        """The PACTOR-4 chirp entry -- answering the grant in the protocol the
        granting station may actually be commanding (`p4chirp`).

        3307.5 ms of audio against a 1.25 s cycle, so `_keyable_slot` steps it
        to one slot in three and says so; the 442.5 ms left before the third
        boundary clears the 210 ms listen floor, because PACTOR-4's own chirp
        cycle is 3.75 s -- three of these slots exactly. Like the PACTOR-3
        entry it is pinned to one arrangement and bakes no polarity in.
        """
        probe = getattr(self, "p4_probe", None)
        if probe is not None:
            probe.request(int(self.live.sample_now()) if self.live is not None else 0,
                          payload, status)
            # The grant callback only queues the probe. No rendering/scanning
            # on the old P3 pre-key deadline, and no ARQ retry owns this burst.
            return REFUSED
        self._advance_aim()
        self._tx(p4chirp.entry_packet(payload, status),
                 f"P4 SL1 CHIRP ENTRY {len(payload)}B "
                 f"{p4chirp.PACKET_S * 1e3:.0f}ms")
        return REFUSED if self.refused else None

    def send_packet(self, sl: int, payload: bytes, status: int,
                    breakin: bool = False) -> int:
        # "the digital data stream that constitutes a specific virtual carrier is
        # swapped to a different tone with every ARQ cycle" -- the same per-cycle
        # parity the grid already counts for the FSK shift, so it comes from the
        # one place that knows which cycle this is. CS3 replaces a reply and
        # starts on the peer-derived order; following data retains that phase.
        #
        # A CHANGEOVER PACKET IS NOT KEYED AT A SPEED LEVEL. `placement.CHANGEOVER`
        # is two carriers and 56 rows whatever the traffic behind it runs at, so
        # `sl` names the level this station will resume at rather than the one on
        # the air here. It takes part in the swap like any other packet.
        grid_swap = self._flip()
        swapped = grid_swap ^ self._p3_packet_swap_bias
        self._note_counter(status, breakin=breakin)
        if breakin:
            if self.raster is not None:
                swapped = self._p3_reply_carrier_order(
                    self._breakin_reference(self.raster, self.slot), swapped)
            # CS3 replaces our control's pulse reference. P3's own measured
            # packet/control pair places it; the reference's two directions
            # have different gaps (pactor3.md §17.1). Aim after rendering
            # because `_flip` can still step the slot under this packet.
            # BUILT ONCE, WHICH IS WHAT THE CYCLE HAS ROOM FOR. Round 14 took
            # the frequency shift off this window and named the render as the
            # next thing on it; this is that. A changeover is re-placed every
            # cycle until one keys -- ten of them on 0913-1550 -- on a
            # byte-identical packet, and `changeover_packet` measures 0.97 ms
            # against a codeword whose render is cached and free, with the 2%
            # trim behind it at 0.69. That 1.66 ms comes out of the same
            # single-digit margin the placement is refused for missing.
            ident = ("bk", payload, status, swapped)
            kept = self._p3_breakin.get(ident)
            raw = (kept[0] if kept is not None else
                   placement.changeover_packet(payload, status, swapped=swapped))
            audio = self._p3_offset(raw, control=True, ident=ident)
            if kept is not None and kept[1] is audio:
                lead = kept[2]
            else:
                # The same phase-reference instant as the control being
                # replaced, not the first sample of its filter skirt. The
                # payload can affect peak normalization, so measure the trim on
                # this actual waveform -- which is why the measurement is kept
                # against the SHIFTED samples and re-taken when they move.
                trim = int(np.flatnonzero(abs(audio) > .02 * max(abs(audio)))[0])
                lead = int(np.argmax(placement.protocol_config().pulse())) - trim
            self._p3_breakin[ident] = (raw, audio, lead)
            while len(self._p3_breakin) > P3_SHIFT_KEEP:
                del self._p3_breakin[next(iter(self._p3_breakin))]
            half = FS // 200 if placement.CASE0_STAGGER else 0
            offsets = (lead + half, lead) if swapped else (lead, lead + half)
            placed = self._place_breakin()
            previous_end = self.tx_end
            self._tx(audio, f"P3 BREAK-IN {len(payload)}B -> SL{sl}{placed}",
                     lead_n=lead, pulse_offsets=offsets)
            if not self.refused and self.tx_end != previous_end:
                self._p3_packet_swap_bias = swapped ^ grid_swap
            if (not self.refused and self.tx_end != previous_end
                    and self.raster is not None
                    and self.tx_audio_start is not None):
                packet = (None if self.host is None else
                          getattr(self.host.arq, "_inflight", None))
                if (self.reply_clock is not None and packet is not None
                        and packet.breakin
                        and (packet.status, packet.payload) == (status, payload)):
                    saved = self._p3_emitted_turn
                    identity = (status, payload)
                    if (saved is None or saved[0] is not self.raster
                            or saved[1] is not packet or saved[2] != identity
                            or saved[6] != self.raster.cycle_n):
                        self._p3_emitted_turn = (
                            self.raster, packet, identity,
                            self.tx_audio_start + min(offsets),
                            self.tx_audio_start, self.tx_end,
                            self.raster.cycle_n)
                self.raster.remember_p3_turn(
                    self.tx_audio_start + min(offsets))
            self.breakin_due = False
            self._log_p3_packet(sl, payload, status, kind="CHANGEOVER")
            return (REFUSED if self.refused else
                    placement.CHANGEOVER.crc_bytes - 3)
        audio = self._p3_offset(
            placement.link_packet(sl, payload, status, swapped=swapped),
            control=False, ident=("pkt", sl, payload, status, swapped))
        self._send_p3_ordinary_audio(
            audio, f"SL{sl} pkt {len(payload)}B status=0x{status:02x}", swapped)
        if self.refused:
            return REFUSED
        self._log_p3_packet(sl, payload, status)
        # Report the actual fixed field size to ARQ truncation accounting.
        return placement.SPEED_PATHS[sl].crc_bytes - 3

    def send_p3_terminal(self, header_bit: int = 1) -> int:
        """Experimental stock close marker on the established short ISS clock."""
        swapped = self._flip() ^ self._p3_packet_swap_bias
        self._note_counter(header_bit)
        audio = self._p3_offset(
            placement.terminal_packet(swapped=swapped, header_bit=header_bit),
            control=False, ident=("terminal", swapped, header_bit))
        self._send_p3_ordinary_audio(
            audio, f"P3 TERMINAL marker VH{header_bit} (await CS{header_bit+1})", swapped)
        if not self.refused:
            self._log_p3_emission(self.traffic.terminal(header_bit))
        return REFUSED if self.refused else 0

    def _send_p3_ordinary_audio(self, audio, label: str, swapped: bool) -> None:
        """Place an ordinary-header waveform at the retained transmit pulse."""
        lead, offsets = 0, None
        clock = self.reply_clock
        phase = (None if clock is None else
                 clock.pulse_epoch if clock.pulse_epoch is not None else
                 clock.entry_phase)
        if phase is not None and self.raster is not None:
            # The ordinary header follows the CS3 on the same symbol clock.
            # Their filter skirts trim differently; keeping their audio starts
            # fixed moves SL1's phase almost 4 ms ahead of the preceding ACK.
            # A bare CS1 can instead accept entry while we remain ISS. Retain
            # that emitted entry's pulse, including its delay, without opening
            # the IRS epoch until a real changeover arrives.
            boundary = self.raster.boundary(self.slot)
            offset = (phase-boundary) % self.raster.slot_n
            if offset > self.raster.slot_n // 2:
                offset -= self.raster.slot_n
            self.boundary = boundary+offset
            lead = placement.pulse_lead(audio)
            half = FS // 200 if placement.CASE0_STAGGER else 0
            offsets = (lead + half, lead) if swapped else (lead, lead + half)
        self._tx(audio, label, lead_n=lead, pulse_offsets=offsets)

    def send_long_packet(self, sl: int, payload: bytes, status: int) -> int:
        """A data packet on the 3.75 s cycle -- three grid slots to the packet.

        The FSM's `cycle_long` routes here (`ptc.PtcHost.send_packet`), and the
        grid places it: `_MasterGrid.ticks` is 3 while long, so `_flip`'s aim
        steps the comb three slots and the burst keys on a cycle boundary rather
        than wherever the acknowledgement that granted it happened to decode.
        The answer is then due 3.390 s past our phase reference rather than
        0.890 (`_MasterGrid.regear`).

        NOT YET ON THE AIR. It is driven end to end on the bench --
        `tests.shrike.test_longcycle` keys one through this grid and reads the
        placement, the answer instant and the peer's own long packet inside our
        listen window -- and no arm has keyed one at a gateway.
        `--no-long-cycle` declines the length for a slot.
        """
        self._note_counter(status)
        swapped = self._flip() ^ self._p3_packet_swap_bias
        audio = self._p3_offset(
                placement.link_packet(sl, payload, status, swapped=swapped,
                                      long_cycle=True), control=False,
                ident=("long", sl, payload, status, swapped))
        self._send_p3_ordinary_audio(
            audio, f"SL{sl} LONG pkt {len(payload)}B status=0x{status:02x}", swapped)
        self._log_p3_packet(sl, payload, status, long_cycle=True)
        return (REFUSED if self.refused else
                placement.LONG_PATHS[sl].crc_bytes - 3)

    # -- PACTOR-2 --------------------------------------------------------
    def _cycle_swap(self) -> bool:
        """This ARQ cycle's carrier arrangement, WITHOUT consuming a toggle.

        `_flip` is the transmit-side alternation and every packet spends one;
        a codeword does not, because it is not a second cycle. The arrangement a
        PACTOR-2 answer has to go out in is the one the packet it answers went
        out in -- the two virtual carriers exchange tones once per ARQ cycle
        (`pactor2.data_burst`), and a fresh flip here would put our
        acknowledgement on the arrangement the NEXT packet will use, which is
        the one the peer is not listening on. On a grid that is
        `_MasterGrid.shift` for our own slot; in a dry run there is no cycle to
        count and the standing toggle is read rather than stepped.
        """
        self._advance_aim()
        return self.shift_inverted if self.invert is None else self.invert

    def send_p2_cs(self, index: int) -> None:
        """PACTOR-2's twenty-bit codeword: DBPSK on both of its carriers.

        A HYPOTHESIS ON THE AIR, and the one thing in a PACTOR-2 link that is.
        `pactor2.control_signal` says at length what it is built out of -- the
        six codewords are PACTOR-3's, measured on a stranger's tape, and their
        keying onto twenty pulses is PACTOR-3's too, put on PACTOR-2's two
        staggered carriers. Nothing outside this package has ever graded one,
        because grading a reverse channel needs a station that answers and a
        monitor never does.
        """
        self._tx(pactor2.control_signal(index, swapped=self._cycle_swap()),
                 f"P2 CS{index + 1} {spec.CS_NAMES[index]}",
                 lead_n=P2_KEY_LEAD_N)

    def send_p2_packet(self, sl: int, payload: bytes, status: int) -> int:
        """A PACTOR-2 data packet on the 1.25 s cycle: marker, then 72 pulses."""
        return self._p2_burst(pactor2.PATHS[sl - 1], payload, status)

    def send_p2_long_packet(self, sl: int, payload: bytes, status: int) -> int:
        """The same on the 3.75 s cycle: 320 data pulses, three grid slots.

        `_MasterGrid.ticks` steps the comb by three exactly as it does in
        PACTOR-3, and the answer moves with the PACKET -- 3.360 s rather than
        0.880, which is `pactor2.cs_slot` at either length (`_MasterGrid.regear`,
        `data_n`). The frame length rides the marker, so a receiver is told which
        of the two it is looking at by the burst itself."""
        return self._p2_burst(pactor2.PATHS_LONG[sl - 1], payload, status)

    def _p2_burst(self, path, payload: bytes, status: int, *,
                  swapped: Optional[bool] = None, what: Optional[str] = None) -> int:
        """One PACTOR-2 cycle's transmission, keyed on the raster.

        `placement.field_info` builds the field for both protocols, because both
        fill it the same way and two ends of one link must agree byte for byte:
        the template where there is nothing to say, IDLE behind a part-filled
        payload, and the status byte last. `pactor2.build_field` puts the CRC
        behind that.

        A payload longer than the path holds is CUT and the return says how much
        went out, which is what stops `arq.PactorArq._sent` settling bytes the
        air never carried.
        """
        n = path.crc_bytes - 3
        self._note_counter(status)
        self._tx(pactor2.data_burst(
            pactor2.build_field(placement.field_info(payload, n, status), path),
            path, swapped=self._flip() if swapped is None else swapped),
            what or f"P2 {path.name} pkt {len(payload)}B", lead_n=P2_KEY_LEAD_N)
        return REFUSED if self.refused else n

    def send_p2_entry_packet(self, payload: bytes, status: int) -> int:
        """The `p2sl1` rung: a PACTOR-2 SL1 short frame where the entry goes.

        Pinned like the PACTOR-3 entry -- unswapped, no polarity remembered --
        because it is the one packet a granting peer has to acquire cold. The
        phase-reference pulse lands on the boundary (`P2_KEY_LEAD_N`), so the
        last symbol falls `P2_ENTRY_END_N` past it, and the answer-position
        instrument is told that extent rather than PACTOR-3's.
        """
        self._advance_aim()
        if self.raster is not None:
            self.raster._entry_extent_override = P2_ENTRY_END_N
        path = pactor2.PATHS[0]
        field = pactor2.build_field(
            placement.field_info(payload, path.crc_bytes - 3, status), path)
        try:
            return self._p2_burst(
                path, payload, status, swapped=False,
                what=f"P2 SL1 COMPAT ENTRY {len(payload)}B "
                     f"{P2_ENTRY_END_N / FS * 1e3:.0f}ms 1400/1600 Hz "
                     f"status=0x{status:02x} field={field.hex()}")
        finally:
            if self.raster is not None:
                self.raster._entry_extent_override = None

    def _ack_gap_line(self, *, changeover: bool = False) -> None:
        """Where our codeword's first symbol landed against the instant the peer
        reads at. Its carrier is up a settle in front of that (`tx_key_up`); what
        the peer's latched read anchor is measured against is the audio.

        `rot` is the distance from our own free-running boundary, which IS that
        instant, and nothing the peer does can move it; `key-dataend` is the
        same quantity seen from the peer's side, off its last measured burst.
        A `rot` outside `p1rx.CS_ANCHOR_S` is an acknowledgement the reference
        decoder refuses at any signal level, so it is said out loud rather than
        left to be read off a column.
        """
        r = self.raster
        if r is None or r.sending or self.tx_audio_start is None or changeover:
            return

        def signed(n: int) -> int:
            n %= r.slot_n
            return n - r.slot_n if n > r.slot_n // 2 else n

        phase_lead = min(self.tx_pulse_offsets) if self.tx_pulse_offsets else 0
        rot = signed(self.tx_audio_start + phase_lead - r.boundary(self.slot))
        where = ("free-running -- no tracked onset this cycle"
                 if r.peer_onset is None else
                 "key-dataend %+7.1f ms"
                 % (signed(self.tx_audio_start - r.peer_onset - r.data_n)
                    / FS * 1e3))
        if r.protocol == Protocol.PACTOR3 and r._p3_peer is not None \
                and r.cycles - r._p3_peer[3] <= 1:
            at, width, cycle_n, _ = r._p3_peer
            elapsed = self.tx_audio_start - at - width
            cycles, gap = divmod(elapsed, cycle_n)
            where = (f"P3 key-dataend {gap / FS * 1e3:+7.1f} ms "
                     f"(CRC frame @ {at}); elapsed {elapsed / FS * 1e3:+.1f} ms, "
                     f"{cycles} whole cycle(s)")
        placed = (self.boundary is not None and self.tx_pulse_offsets is None
                  and self.boundary != r.boundary(self.slot))
        missed = (
            "  -- off the boundary ON PURPOSE: the changeover packet is placed "
            "against the peer's transmission, not read at a latched instant"
            if placed else
            "" if abs(rot) <= p1rx.CS_ANCHOR_S * FS else
            f"  !! OUTSIDE THE {p1rx.CS_ANCHOR_S * 1e3:.0f} ms READ ANCHOR: "
            f"the peer reads twelve bit periods from one latched instant and "
            f"this word starts outside them")
        if r.protocol == Protocol.PACTOR3 and self.tx_pulse_offsets is not None:
            missed = " (measured P3 pulse offset; no P1 read-anchor verdict)"
        print(f"    [ack] {where} | rot {rot / FS * 1e3:+7.1f} ms{missed}",
              flush=True)
        if self.tx_pulse_offsets is not None:
            centers = tuple(self.tx_audio_start + n for n in self.tx_pulse_offsets)
            peer = r._p3_peer
            age = "unknown" if peer is None else f"{(self.tx_audio_start-peer[0])/FS:.3f} s"
            print(f"    [ack] P3 pulse centers ch5={centers[0]} ch12={centers[1]}; "
                  f"audio start={self.tx_audio_start}, end={self.tx_end}; "
                  f"last decoded peer frame age={age}", flush=True)

    def pump(self) -> None:      # the far end is real; nothing to settle
        pass

    def cycle(self) -> None:
        pass

    observing = False
    """The link is gone and the run is listening out the rest of its hold.

    A latch on the transmitter rather than a rule in the loop above it, because
    the loop is not the only thing that keys: the FSM answers a decode from
    `on_rx_event`, and a station that calls US on the frequency we are listening
    to is answered out of LISTENING with a codeword. An observation the channel
    can talk into keying is not an observation.
    """

    def answered_refusal(self) -> Optional[str]:
        """Why nothing may go on the air on this link, or None.

        `_MasterGrid` is the CALLER'S grid: its anchor is placed from our own
        first call and free-runs from there, and every instant it offers is
        right only because the station reading them is the one we called. A
        PACTOR-1 ISS reads its control signal twelve bit periods from one
        latched instant and never searches, and which instant that is depends
        on which end of the call it sits at -- its own packet end plus `d` as
        the caller, plus `slot - data - cs - d` as the called station
        (`tests.shrike.test_ackplace._reference`, off pactor.c:745, :1745,
        :1122). On a link we answered the peer is the caller, so it reads
        `2d - (slot - data - cs)` from where this grid keys: 70 ms, seven bit
        periods, at a 50 ms turnaround, and the reference decoder refuses that
        at any signal level.

        The placement a called station needs is not a second `boundary` -- the
        asymmetry is entirely in where the anchor goes. The reference latches
        the responder's transmit clock to the caller's connect, at `10 ms + our own
        txdelay` past its end (:1744/:1745), and free-runs it by whole cycles
        from there like any other. Nothing here places that anchor, because
        nothing here answers a call: `run` takes a `--dxcall` and calls. Until
        something does both, a burst keyed off the caller's grid is one the
        station that called us cannot read, and saying so is worth more than
        a form.
        """
        if self.observing:
            return ("the link is down and this run is only listening out the "
                    "rest of its hold")
        if self.host is None or not self.host.arq.answering:
            return None
        return ("this link was ANSWERED, not placed. The transmit grid is a "
                "caller's and the station that called us reads at its own "
                "instant; no anchor for the called side exists to key on")

    qrm_guard = True
    """Whether a decoded codeword may refuse a transmission of our own.

    On by default and off in one token (`--no-qrm-guard`), because the other
    arrangement is how `--p1-act-on-grant` came to default off and discarded
    every grant for a week. Every line the guard prints names the flag.
    """

    listening = False
    """This cycle is a deliberate silence: render, refuse, and key nothing.

    `--listen-every N`, set by the hold loop for one cycle in N. Refusing at the
    seam rather than skipping the FSM's tick is what makes the silence cost
    nothing: the packet is re-placed next cycle, the retry budget counts air and
    charges none of it (`arq.REFUSED`), and the cycle the peer would have spent
    answering our carrier is a whole 1.25 s of open receiver instead of the
    0.24 s a keyed cycle leaves.
    """

    def _refused(self, carrier: int, air_n: int, what: str, *,
                 changeover: bool = False, strict: bool = False) -> bool:
        """Say why this burst is not going out, or answer False.

        ONE DOOR, ONE RULE, TWO NAMES. What each role loses by dropping differs
        -- as IRS the burst is a control signal and the peer repeats, as ISS it
        is a cycle of a grant -- but neither may refuse for ever, and the
        receiving side used to. Its phase is `_d_max_n` less a turnaround that
        does not move inside a session, so a first refusal there is not one
        cycle: it is the whole link, silently, at the one moment a gateway is
        trying to hand us mail. Both stand down after `GUARD_MAX_DROPS`.

        THE CHANGEOVER PACKET IS THE THIRD CASE AND IT NEVER STANDS DOWN. The
        stand-down is for a grid out of step with the peer's raster, where a
        refusal repeats on evidence that cannot be acted on; this burst is
        placed on that raster by construction (`_breakin_key`), so a refusal
        means only that the placement could not be made -- and keying anyway
        puts the packet inside the transmission it is answering, which is the
        one thing it must never do. On 2026-09-03 the release let seven of them
        out into exactly that, at 656-667 ms into the peer's packet, and the
        gateway answered none. A refused changeover costs the cycle and no
        retry (`arq.REFUSED`), so nothing deadlocks on it.

        THE LINE IS READ ONCE, MID-SLOT, AND HAS TO STAND ALONE. An operator
        seeing it needs to tell a refusal from a fault without leaving the
        terminal: what was heard, from whom, what was therefore declined, how
        many in a row, and the flag that turns it off.

        A DROP IS ALSO RECORDED, in `refused`, because the FSM is owed it: see
        `_tx`.
        """
        refusal = self.raster.key_refusal(carrier, air_n, changeover=changeover)
        if refusal is None:
            self.guard_drops = 0
            return False
        probe = getattr(self, "p4_probe", None)
        if probe is not None and probe.emitting:
            if not self.qrm_guard:
                return False
            print(f"    !! P4 PROBE GUARD -- NOT KEYING {what}: {refusal}; "
                  "probe stops; no automatic stand-down.", flush=True)
            self.refused = True
            return True
        if strict:
            print(f"    !! CW ID GUARD -- NOT KEYING {what}: {refusal}.", flush=True)
            self.refused = True
            return True
        if changeover:
            print(f"    !! CHANGEOVER NOT PLACED -- NOT KEYING {what}: "
                  f"{refusal}. The packet that takes the link may key over the "
                  f"peer's NEXT transmission, which is the one it cancels, and "
                  f"never over the one it answers.", flush=True)
            self.refused = True
            return True
        if self.raster.protocol == Protocol.PACTOR3 and not self.raster.sending \
                and self.raster._p3_peer is not None \
                and self.raster.cycles - self.raster._p3_peer[3] <= 1:
            # A fresh CRC-valid frame is stronger evidence than the historical
            # energy-only guard. Repeated refusals cannot make its airtime free.
            self.raster._p3_reply_phase_invalid = True
            print(f"    !! P3 ACK GUARD -- NOT KEYING {what}: {refusal}.",
                  flush=True)
            self.refused = True
            return True
        # `--no-qrm-guard` is the SENDING side's, and its help says so: the
        # receiving side's refusal predates the flag and is not what the operator
        # reaches for mid-slot.
        guard = "QRM GUARD" if self.raster.sending else "ACK GUARD"
        flag = "; --no-qrm-guard turns this guard off" if self.raster.sending else ""
        if self.raster.sending and not self.qrm_guard:
            print(f"    !! {guard} OFF -- KEYING {what} ANYWAY: {refusal}. "
                  f"Drop --no-qrm-guard from the arm line to have this refuse.",
                  flush=True)
            return False
        self.guard_drops += 1
        if self.guard_drops > GUARD_MAX_DROPS:
            self.guard_drops = 0
            print(f"    !! {guard} STOOD DOWN -- KEYING {what} after "
                  f"{GUARD_MAX_DROPS} refusals in a row: {refusal}. A "
                  f"refusal every cycle is our grid out of step with the peer's "
                  f"raster, not the peer, and a guard that never lifts hands "
                  f"over the channel.", flush=True)
            return False
        print(f"    !! {guard} -- NOT KEYING {what}: {refusal}. Burst dropped "
              f"({self.guard_drops} of {GUARD_MAX_DROPS} in a row{flag}).",
              flush=True)
        self.refused = True
        return True

    # -- transmit or stash ------------------------------------------------
    def identify(self, mycall: str) -> None:
        """Identify a normally completed unanswered call through the TX guards.

        The connect burst names only the called station. Our callsign is sent
        in link setup after an answer, so an unanswered call has transmitted
        without identifying its caller. CW supplies that missing identification.

        A refusal leaves identification to the operator. This is never called
        while unwinding an exception, or for a dry run or replay. Ctrl-C during
        identification stops playback and is consumed here so device cleanup,
        the session summary, and mail writes can finish; it never retries CW.
        """
        if not (self.transmit and self.rig is not None and self.keyed):
            return
        try:
            audio = cwid.audio(mycall)
            if len(audio) / FS > self.max_key:
                raise ValueError("CW identification exceeds --max-key")
            if self.live is None or self.raster is None or self.boundary is None:
                raise ValueError("no receive clock for guarded CW identification")
            # Completion can follow an unused slot whose boundary has passed.
            self._advance_aim(check_late=True)
            before = len(self.keyed)
            self._tx(audio, f"ID {mycall.upper()}", strict_guard=True)
            if len(self.keyed) == before:
                raise ValueError("transmit guard refused identification")
        except BaseException as exc:  # Cleanup must survive SystemExit from playback.
            print(f"  !! IDENTIFICATION NOT CONFIRMED ({exc!r}) -- send "
                  f"{mycall.upper()} by hand", flush=True)

    def _tx(self, audio: np.ndarray, what: str,
            drive: float | None = None, lead_n: int = 0,
            *, strict_guard: bool = False,
            pulse_offsets: tuple[int, int] | None = None) -> None:
        """Key `audio`, or say why not.

        `lead_n` moves the carrier that many samples IN FRONT of the boundary,
        and it survives the backstop below -- a re-aimed burst takes the same
        lead onto its new slot, because a shaped PACTOR-2 comb keyed flush with a
        boundary is 18.75 ms late wherever that boundary is (`P2_KEY_LEAD_N`).

        WHETHER IT WENT OUT IS LEFT IN `refused`, which the packet seams hand
        back to the FSM as `arq.REFUSED`: a cycle nothing was keyed in spends no
        retry, because the budget counts air.
        """
        probe = getattr(self, "p4_probe", None)
        if probe is not None and probe.requested and not probe.emitting:
            self.refused = True
            return
        self.tx_pulse_offsets = pulse_offsets
        # Consumed here, like `_sent_invert` below: a render the guards decline
        # must not label the next burst with a frequency it was not keyed at.
        # BESIDE THE LABEL AND NOT IN IT -- `what` is the burst's identity, and
        # the sidecars and the scenes that read one are entitled to keep it.
        offset_hz, self._tx_offset_hz = self._tx_offset_hz, None
        tuned = "" if offset_hz is None else f"  TX {offset_hz:+g} Hz"
        placed, self.placed = self.placed, False
        why, self.unplaceable = self.unplaceable, None
        self.refused = False
        if self._timing_trial_refusal():
            return
        if self.listening:
            print(f"    !! LISTENING CYCLE -- NOT KEYING {what}: this cycle is "
                  f"given to the receiver (--listen-every). No retry spent; the "
                  f"packet goes out on the next one.", flush=True)
            self.refused = True
            return
        if why is not None:
            # NOT KEYED AND NOT MOVED, and the cycle is what pays. A changeover
            # has one instant and it is the peer's; where that instant cannot be
            # computed there is no second-best placement, only our own comb --
            # which is what 2026-09-04 keyed 17 times at a refuted +70 ms while
            # every line it printed read healthy. The retry budget counts air,
            # so this spends none of it and the FSM places the packet again on
            # the next reading. `_refused`'s third case says why this one never
            # stands down.
            print(f"    !! CHANGEOVER NOT PLACED -- NOT KEYING {what}: {why}. "
                  f"Not keyed and no retry spent; the changeover is re-placed "
                  f"on the next reading.", flush=True)
            self.refused = True
            return
        answered = self.answered_refusal()
        if answered is not None:
            # NOT `refused`, and the difference is what the FSM can do about it.
            # A guard's drop is one cycle of a placement that can be made again;
            # this station cannot key AT ALL on this link, and a budget that
            # never spends leaves it up for ever with nothing on the air.
            print(f"    !! NOT KEYING {what}: {answered}.", flush=True)
            return
        self.n += 1
        # Consume the rendered arrangement so it cannot constrain a later burst.
        inv, self._sent_invert = self._sent_invert, None
        # The renderers pick amplitudes to suit an offline decode from a file --
        # the PACTOR-1 connect comes out at peak 0.11, nearly 20 dB down -- and on
        # a data interface the level IS the transmit power, so that shipped 5 W
        # where the rig was set for 50.
        audio = levels.at_drive(audio, self.drive if drive is None else drive)
        # Key only for the RF. The renderers pad silence around a burst so a
        # file-fed decoder has room to settle -- the PACTOR-1 connect carries
        # 0.5 s at each end of a 1.98 s file for 0.98 s of signal -- and
        # transmitting that padding holds the key open with dead air, which is
        # both bad practice and half the duration of a PACTOR cycle wasted.
        audio = _trim_silence(audio)
        if lead_n and self.boundary is not None:
            self.boundary -= lead_n
        # HOW FAR INTO ITS BOUNDARY THIS CARRIER COMES UP, and it is carried
        # apart from `boundary` on purpose: the boundary is the grid's and every
        # instrument in this file measures against it, so a burst keyed a
        # millisecond into one has to go on reading as a millisecond late rather
        # than as a boundary that moved. See `KEY_CLAMP_TOL_S`.
        forgiven = 0
        dur = len(audio) / FS
        self.last_dur = dur
        unheard = round(dur * FS)     # ...until a capture stream says otherwise
        # EVERY transmission lands on the grid, not only the ones the cycle tick
        # starts. An acknowledgement decoded mid-window used to key the instant the
        # decoder emitted it -- measured on a replay, 250 ms before the boundary --
        # because the FSM answers from `on_rx_event` and only `host.tick()` was
        # standing on the raster. That is the whole collision the cycle grid exists
        # to prevent, arriving through the other door: the peer's answer is due, and
        # we key into it because our own decode happened to finish
        # early. The wait is on SAMPLES and is a no-op when the tick already waited.
        if self.live is not None and self.boundary is not None:
            # THE BACKSTOP, and it guarantees strictly less than `_regrid` does.
            # The run loop gives a slot back before the audio for it is
            # collected, which is the right answer where there is a window left
            # to spend; this catches everything else -- the transmissions the
            # loop does not schedule, a burst the FSM keys straight out of a
            # decode -- and all it can do for those is move the carrier to a
            # boundary that will still take it. Said out loud every time, because
            # after the carrier every figure is measured from the same late
            # position and reads healthy.
            settle_n = round(self.settle * FS)
            # REFUSE, OUT LOUD, rather than key into the peer -- and ask BEFORE
            # the placement decision rather than after it. This is channel
            # analysis; what follows it is arithmetic on a clock that keeps
            # running, and the backstop's approval below is only good until the
            # next callback hands a block over. Paid here it comes out of channel
            # time. Paid between the approval and the re-check it came out of the
            # admission budget, which is where eight of 2026-09-11's thirty
            # transmissions went. Asked on the whole interval the rig will be
            # keyed for -- the settle and the trimmed audio behind it, which is
            # what the far end shares the channel with.
            if self.raster is not None and self._refused(
                    self.boundary - settle_n, settle_n + unheard, what,
                    changeover=placed, strict=strict_guard):
                return
            # WHAT THE GRID HAD TO ALLOW FOR, closed here because this is the
            # other end of it: the tick, the render, the level, the trim and the
            # channel guard all sit between the grid's admission check and this
            # one. `_regrid` cannot forgive an overrun without knowing this
            # number, and nothing could say whether work taken off the pre-key
            # path had actually come off it. Consumed, so a slot the grid handed
            # back charges its listening window to nobody.
            admitted, self._admitted_at = self._admitted_at, None
            if admitted is not None:
                # ...to the path the GRID charged it against, which is the other
                # end of the same interval: `_clamp_forgives` keyed its question
                # on `breakin_due` and so does this answer. See `breakin_cost_n`.
                cost = max(0, int(self.live.sample_now()) - admitted)
                if self.breakin_due:
                    self.breakin_cost_n = cost
                else:
                    self.prekey_cost_n = cost
            # Against the sample, with nothing allowed for. `clamp_late` is what
            # the clamp would actually cost this burst, and a carrier that cannot
            # come up on the boundary is not on the grid. See `_regrid`.
            # Per-burst playback and replay put audio a full settle after now;
            # only duplex can schedule its audio independently of the PTT lead.
            # Guard the onset this path actually emits/models, not merely now.
            late = self.live.clamp_late(self.boundary)
            if getattr(self.live, "transmit", None) is None:
                # A simulated clock may itself impose DAC notice. Notice and
                # settle overlap, so enforce both floors without summing them.
                late = max(late, int(self.live.sample_now() + settle_n
                                     - self.boundary))
            # THE READER'S OWN NUMBER FOR THIS BURST. Both are five milliseconds
            # -- half a symbol at 100 Bd -- and they are separate constants
            # because they answer about different instants: our comb, and the
            # peer's packet ending.
            tol_s = BREAKIN_CLAMP_TOL_S if placed else KEY_CLAMP_TOL_S
            if (0 < late <= round(tol_s * FS)
                    and getattr(self.live, "transmit", None) is not None):
                # KEYED WHERE IT CAN BE, which is what the peer is waiting for.
                # See `KEY_CLAMP_TOL_S`: the cycle is not late because anything
                # here was slow, and the slot after this one is not a cycle the
                # peer will read us in.
                #
                # WITH A TRANSMITTER TO MOVE, and only then. `transmit` asserts
                # PTT a settle in front of the audio it is given, so a burst
                # handed a later instant keeps its whole lead. A stream with no
                # output device has no such instant: there the shortfall is
                # measured against the settle itself (above), and forgiving one
                # would spend the rig's lead rather than the burst's place.
                #
                # ASKED AGAIN ON THE INSTANT IT WILL ACTUALLY OCCUPY. The
                # channel guard above was answered about a carrier on the
                # boundary, and this one comes up behind it: a burst that fits
                # its cycle by less than the forgiveness would otherwise be
                # shifted into the peer's next transmission by this very line.
                #
                # A PLACED CHANGEOVER TOO, and the exclusion this replaced read
                # one sentence into two. "Nowhere to be moved to" is true of the
                # STEP below -- a later boundary of ours is not that burst placed
                # elsewhere -- and it is not true here, where the same instant is
                # merely occupied `late` samples into itself. The direction is
                # the safe one: the changeover sits PAST the packet it answers,
                # so a slip can only widen that gap, never walk back into the
                # transmission. 0913-1550 refused ten of them at +1.2 to +7.2 ms
                # against a peer whose read window this station has measured at
                # +88.5 to +98.0 -- every one of those a cycle given away for a
                # slip the reader does not notice, and the lost slots behind
                # them are what took the link down.
                if self.raster is not None and self._refused(
                        self.boundary + late - settle_n, settle_n + unheard,
                        what, changeover=placed, strict=strict_guard):
                    return
                forgiven = late
                print(f"    !! {KEYED_LATE}: {what} comes up "
                      f"{late / FS * 1e3:+.1f} ms into its boundary rather than "
                      f"on it -- inside the "
                      f"{tol_s * 1e3:.0f} ms the reader forgives, so "
                      f"the cycle is keyed rather than given away.", flush=True)
                late = 0
            if late and placed:
                # PAST THE FORGIVENESS A CHANGEOVER PACKET HAS NOWHERE TO BE
                # MOVED TO. Its instant is the peer's packet ending plus a lead,
                # so a later boundary of ours is not the same burst placed
                # elsewhere -- it is the changeover keyed at whatever our comb
                # happens to hold, which on 2026-09-03 was the middle of the
                # packet it was answering. The cycle is given up instead, and
                # the FSM re-places it against the next reading: the intent is
                # standing by then, so the window in front of the key closes
                # early enough to make it.
                print(f"    !! {LATE_KEY}: {what} is placed against the peer's "
                      f"transmission and that instant went "
                      f"{late / FS * 1e3:+.1f} ms ago. Not keyed; the changeover "
                      f"is re-placed on the next cycle.", flush=True)
                self.refused = True
                return
            if late:
                gone = self.boundary
                # IN THE SHIFT THE AUDIO IS ALREADY RENDERED IN. `shift(slot)` is
                # slot parity, so the next placeable boundary flips it -- and a
                # burst on a correct boundary in the wrong shift is unreadable to
                # a station counting cycles, in a way that looks exactly like a
                # dead band. Holding the polarity means stepping in twos, which
                # costs at most one more slot than not caring would.
                #
                # The search starts one slot on because the one we are standing on
                # has demonstrably gone; that is also what makes it terminate.
                # With the settle in hand, not merely placeable: placeability
                # alone accepts a boundary `key_notice` ahead, and `transmit` then
                # finds the PTT instant already past and keys immediately, which
                # on the x6100 hands the rig 32 ms of a 400 ms settle.
                phase_offset = 0
                if (self.reply_clock is not None
                        and (self.reply_clock.pulse_epoch is not None
                             or self.reply_clock.entry_phase is not None)
                        and self.raster.protocol == Protocol.PACTOR3):
                    # Mail data is aimed at the established symbol pulse;
                    # the raster itself still names the control audio start.
                    # Re-aiming after a late render must preserve that offset
                    # as well as the already-rendered carrier arrangement.
                    phase_offset = (self.boundary + lead_n
                                    - self.raster.boundary(self.slot))
                self.aim(self.raster, _keyable_slot(
                    self.live, self.raster, self.raster.next_slot(self.slot),
                    settle_n, listen=False, shift=inv))
                self.boundary += phase_offset - lead_n
                print(f"    !! {LATE_KEY}: the carrier would come up "
                      f"{late / FS * 1e3:+.1f} ms into slot boundary {gone} "
                      f"rather than on it. Aiming at slot {self.slot} instead, "
                      f"which is the next one this burst's own shift fits.",
                      flush=True)
                # ON THE CARRIER INSTANT OF THE FINAL BOUNDARY. A burst that
                # moved is held to the same floor as one that did not, so the
                # guard is asked again here -- and only here, which is what keeps
                # it off the critical path on every cycle that keys where it
                # aimed.
                if self.raster is not None and self._refused(
                        self.boundary - settle_n, settle_n + unheard, what,
                        changeover=placed, strict=strict_guard):
                    return
            # NOT DEAF ACROSS THE SLOT WE JUST GAVE UP. Moving a burst to a later
            # boundary leaves a whole slot of the peer's channel in front of the
            # key, and `flush_to` after the carrier takes every sample of it --
            # the same loss as emptying the capture queue after a transmission,
            # which threw away the far end's opening symbols and then recorded
            # that nobody had answered.
            #
            # TAKE FIRST, THEN SLEEP OUT WHAT THE BLOCK BOUNDARY LEAVES, and the
            # order is the whole of the PTT lead. `read` takes the capture queue
            # one 128-frame block at a time and concatenates -- 4.4 ms to drain a
            # 20 s backlog, 10.2 ms for 60 s, measured -- and it BLOCKS per block,
            # so a take that runs to the key instant drains the queue as the codec
            # fills it and finishes with nothing left to do. Behind the wait
            # instead, the whole drain lands between the bridge and the key, where
            # every millisecond of it is spent out of the rig's settle rather than
            # out of the cycle. That is what "PTT LEAD ERODED to 29 ms of the 40"
            # was on 2026-08-02: the backstop had just re-aimed 20 slots on, so
            # the wait was 24 s long and the queue behind it was 24 s deep.
            #
            # A replay reads the same samples either way. Its `wait_until` is an
            # advance through the file and stashes what it passed for the next
            # `read_ready`; taken first, the same span comes back here instead and
            # is bridged here instead, and the wait that follows then has nothing
            # left to advance over. What must not happen on a replay is reading
            # PAST this instant -- that consumed the next window from under the
            # loop and lost the 840 ms behind a changeover head, whole -- and
            # `take_until` is bounded by the same index the wait is.
            #
            # HELD, never fed. `_SessionRx.feed` decodes, the FSM answers a decode
            # from `on_rx_event`, and that would key from inside a transmission
            # that has not gone out yet. The next flush decodes it.
            # A large output latency can require more DAC notice than the PTT
            # settle. Do not spend that notice in this final buffer drain after
            # the placeability guard has already passed -- and keep the measured
            # reserve behind it, so what separates the approval from the re-check
            # is an interval somebody measured rather than the remainder of one
            # callback block. See `TX_ADMIT_RESERVE_S`.
            wait_lead = _prekey_lead(self.live, settle_n,
                                     self.cycle_cost_n() if self.breakin_due
                                     else 0)
            waiting = self.live.take_until(self.boundary - wait_lead)
            if waiting.size and self.sessrx is not None:
                self.sessrx.bridge(waiting)
            self.live.wait_until(self.boundary - wait_lead)
            # Draining/bridging can itself stall. The already rendered waveform
            # must not be silently clamped late by the duplex transmitter.
            lost = (self.live.clamp_late(self.boundary)
                    if getattr(self.live, "transmit", None) is not None else 0)
            if 0 < lost <= round(tol_s * FS):
                # The same forgiveness as above, on the far side of the drain,
                # and the same question asked of the channel first: the burst is
                # rendered, the slot is this cycle's, and half a symbol of
                # lateness is not a reason to spend the cycle. A changeover on
                # its own tolerance for the same reason it has one above --
                # forgiving it occupies its placed instant a shade later, and
                # nothing here moves it to a boundary of ours.
                if self.raster is not None and self._refused(
                        self.boundary + lost - settle_n, settle_n + unheard,
                        what, changeover=placed, strict=strict_guard):
                    return
                forgiven, lost = lost, 0
            if lost:
                # WITH THE MAGNITUDE. Eight of these were printed on 2026-09-11
                # and not one of them carried a number, so nothing could tell the
                # burst that would merely have been 0.2 ms late from the one that
                # would have landed 29 ms inside the peer's raster -- which is
                # the whole question a refusal raises.
                print(f"    !! {LATE_KEY}: {what} lost its DAC notice while "
                      f"draining the receive buffer -- the earliest audio this "
                      f"stream can still schedule is "
                      f"{lost / FS * 1e3:+.1f} ms past the boundary; not keyed.",
                      flush=True)
                self.refused = True
                return
        if probe is not None and probe.emitting and not probe.room(
                max(int(self.live.sample_now()), self.boundary),
                len(audio) / FS + probe.listen_seconds, FS):
            self.refused = True
            probe.reason = "deadline leaves no complete chirp and receive window"
            return
        def record_slot():
            if self.boundary is not None:
                self.slots_used.append(self.slot)
                if self.raster is not None:
                    self.raster.keyed_slot = self.slot
                    if self.host is not None:
                        # Admission evidence belongs to the burst that really
                        # keyed, not the next requested or skipped slot. Retain
                        # the waveform end and this emission's cycle geometry.
                        self.raster._p3_keyed_reply = (
                            (self.slot, self.boundary,
                             self.boundary + len(audio), self.raster.cycle_n)
                            if self.host.protocol == Protocol.PACTOR3 else None)
                        if probe is not None and probe.emitting:
                            return  # P4 probe geometry is recorded independently of P3.
                        line = self.raster.keying(
                            self.host.protocol, extent_n=len(audio),
                            entry_pending=self.host.arq.entry_pending,
                            entry_variant=self.host.arq.entry_variant)
                        if line:
                            print(f"    [grid] {line}", flush=True)
        # The duplex stream's own transmitter, where there is one. A replay has
        # no output device, so it keeps the per-burst playback and the modelled
        # carrier times below.
        duplex = getattr(self.live, "transmit", None)
        # A duplex enqueue may still refuse after a scheduling/logging stall.
        # Publish its slot and waveform geometry only after actual transmission.
        if duplex is None or not (self.transmit and self.rig is not None):
            record_slot()
        if duplex is None and self.live is not None and self.boundary is not None:
            # MODELLED, not measured: a replay has no carrier, so where one would
            # have come up is the best this path can say. The duplex path below
            # reports the sample it actually came up on instead.
            off = (self.live.sample_now() + self.settle * FS - self.boundary) / FS
            print(f"    [grid] RF would start {off * 1e3:+.1f} ms from the slot "
                  f"boundary", flush=True)
            self.tx_audio_start = int(self.live.sample_now() + self.settle * FS)
            self.tx_key_up = self.tx_audio_start - round(self.settle * FS)
            # ...and where it would have dropped. Overwritten below by the
            # measured instant wherever there is one, and it has to exist even
            # where there is not: the connect search is aimed at the turnaround
            # after our carrier, so a replay with no `tx_end` searches nowhere
            # and reads exactly like a band with nobody on it.
            self.tx_end = self.tx_audio_start + round(dur * FS)
            self._ack_gap_line(changeover=placed)
        if self.transmit and self.rig is not None:
            print(f"  TX[{self.n}] {what}{tuned}  ({dur:.1f}s)  -- keying", flush=True)
            if duplex is not None:
                # Where our audio and our carrier actually landed, on the capture
                # stream's own clock, so a peer burst found later can be placed
                # against them. Every figure this file printed until the stream
                # went duplex was measured against a boundary or a window edge,
                # and those are our intentions rather than our emissions -- three
                # separate analyses of the same collision reached three different
                # answers because the one interval that decides it, peer burst
                # against our carrier, was never recorded.
                #
                # THE TWO ARE A SETTLE APART AND ARE NOT INTERCHANGEABLE. `at`
                # names the audio, so `transmit`'s first return is the boundary
                # itself whenever the burst was placeable and says nothing about
                # the carrier. `keyed_at` is the instant PTT was asserted, read
                # off the same converter the peer's onsets are read off.
                try:
                    if self._timing_trial_refusal(forgiven):
                        return
                    self.tx_audio_start, self.tx_end = duplex(
                        audio,
                        at=None if self.boundary is None
                        else self.boundary + forgiven, settle=self.settle,
                        key=self.rig.ptt, max_key=self.max_key)
                except _MissedTxSlot as exc:
                    self.refused = True
                    print(f"    !! {LATE_KEY}: {what} missed the DAC enqueue "
                          f"deadline by {int(exc.args[0]) / FS * 1e3:.1f} ms; "
                          "not keyed.", flush=True)
                    return
                self.tx_key_up = self.live.keyed_at
                record_slot()
                # Discard only what was captured while the carrier was up. The
                # far end starts answering 70-190 ms after it drops, measured on
                # a third-party receiver, and dropping the whole queue here threw
                # those opening symbols away -- after which our own log recorded
                # that nobody had answered.
                #
                # WHAT IT DROPS IS WHAT THE DECODER'S CLOCK OWES, and it is the
                # settle as well as the burst: the PTT lead is captured audio
                # nobody heard. Read here rather than assumed, so the two clocks
                # cannot drift apart by a term neither of them names -- see
                # `_SessionRx.cs_at`, which indexes the capture stream with one.
                unheard = self.tx_end - self.live.pos
                self.live.flush_to(self.tx_end)
                where = ("%+.1f ms from the slot boundary, its audio %+.1f"
                         % ((self.tx_key_up - self.boundary) / FS * 1e3,
                            (self.tx_audio_start - self.boundary) / FS * 1e3)
                         if self.boundary is not None
                         and self.tx_key_up is not None else
                         "with no slot yet to aim at")
                # What the rig actually got before the audio, against what it was
                # configured to need. `transmit` cannot key earlier than the
                # stream can schedule, so a boundary accepted with too little
                # notice comes out as a short settle rather than as a late burst
                # -- silent, and on the x6100 it is 32 ms of 400. Printed rather
                # than inferred from the two numbers either side of it.
                #
                # THE SHORTFALL IS THE HOLDBACK, and it is arithmetic rather than
                # jitter. `take_until` above reads to the same sample `wait_until`
                # then aims at, and `read` returns only once the codec has
                # DELIVERED it -- one block plus one input latency later, which is
                # the definition of `holdback`. The wait behind it has nothing
                # left to sleep, so the key lands that far inside the settle:
                # 13 ms on this machine, and 40 - 13 is the 27 the logs floor at.
                # Corpus-wide, 2026-08-14, 89 session logs: all 89 first connects
                # keyed 39-40 on the `at=None` path, which does not go through
                # here, and all 746 gridded bursts keyed 26-34, median 30. No
                # burst on either path crosses into the other's range.
                #
                # NOT the keying command, which this line charged it to until
                # 2026-08-14. `Rig.ptt(True)` is two stats and one non-blocking
                # write into rigctl's stdin and never reads a reply -- 0.04 ms
                # median with a callback contending at 375 Hz, measured -- so this
                # is the lead itself and not a floor under it.
                #
                # WHAT IT COSTS ON THE AIR IS ZERO, measured off the KiwiSDR
                # witness of `captures/onair-0803-*`: our own bursts arrive square
                # at 103 mi, within 1.6 dB of body one millisecond after turn-on,
                # bit 0 the strongest of the head, nine connects at zero bit
                # errors. `tests/shrike/test_txhead.py` carries that, the 8 ms of
                # head a connect can actually lose, and why the missing term is
                # not simply added back.
                lead = self.live.keyed_s - dur
                # Spoken when it first erodes and when it worsens, and otherwise
                # left to the summary: on 2026-08-13 this suffix printed on
                # every burst after the first, all night, both bands, every
                # gateway, and fifty repeats of one fact bury the rest of the
                # log. The lead itself is still on every burst's line, as PTT
                # keyed minus audio.
                worse = (lead < self.settle - 1e-3
                         and (not self.leads or lead < min(self.leads)))
                self.leads.append(lead)
                # WHAT THE CORRECTION ACTUALLY DID, on the burst it did it to.
                # It moves the audio and the key together and is invisible in
                # the two instants either side of it, so a run that flew with it
                # and one that flew without read identically without this.
                early = getattr(self.live, "tx_latency_n", 0)
                corrected = ("" if not early else
                             f", both {early / FS * 1e3:.1f} ms early for this "
                             f"station's uncompensated loop latency")
                short = ("" if not worse else
                         f" -- PTT LEAD SHORT at {lead * 1e3:.0f} ms of the "
                         f"{self.settle * 1e3:.0f} this rig is set for (the "
                         f"holdback, spent reading up to the key instant; costs "
                         f"no head on this rig -- see test_txhead)")
                print(f"    [grid] RF started {where}; PTT keyed "
                      f"{self.live.keyed_s:.3f} s for {dur:.3f} s of audio"
                      f"{corrected}{short}", flush=True)
                self._ack_gap_line(changeover=placed)
            else:
                _dur, unkey = ota._play(audio, self.out_dev, self.max_key,
                                        self.rig, settle=self.settle)
                if self.live is not None:
                    was = self.live.pos
                    self.live.flush(before=unkey)
                    self.tx_end = self.live.pos
                    unheard = self.tx_end - was
            # The carrier is CLAIMED only here, with the burst over and the
            # channel that keyed it still inspectable. Refusing to go on is the
            # point: every later line of the session -- cycles keyed, carrier
            # seconds, the verdict itself -- is quoted off this list, and a
            # burst nothing keyed must not be able to reach any of them.
            fail = self.rig.key_failure()
            if fail is not None:
                raise PttError(f"TX[{self.n}] {what} has no keyed carrier "
                               f"behind it: {fail}")
            self.keyed.append((self.slot, dur))
            if pulse_offsets is not None:
                # Exact DAC reference, not an RF recording. Preserve level and
                # absolute timing for comparison with pickup or independent RF.
                wav.write(self.outdir / f"tx_{self.n:02d}.wav", audio, FS,
                          source=("PACTOR-3 entry DAC reference" if " ENTRY " in what
                                  else "PACTOR-3 control DAC reference"), control=what,
                          slot=self.slot, audio_start=self.tx_audio_start,
                          audio_end=self.tx_end, ptt_up=self.tx_key_up,
                          pulse_offsets=list(pulse_offsets),
                          tx_offset_hz=offset_hz,
                          timing_reply_delay_ms=(None if self.timing_trial is None else
                                                 self.timing_trial.reply_delay_ms),
                          timing_trial=(None if self.timing_trial is None else
                                        self.timing_trial.arm),
                          p3_mail_reply_epoch=(None if self.reply_clock is None else
                                               self.reply_clock.pulse_epoch),
                          entry_phase=(self.tx_audio_start + min(pulse_offsets)
                                       if " ENTRY " in what else
                                       None if self.timing_trial is None else
                                       self.timing_trial.entry_phase))
            if self.sessrx is not None:
                self.sessrx.skip(unheard / FS)   # ...do not splice it in either
        else:
            wav_path = str(self.outdir / f"tx_{self.n:02d}.wav")
            session.write_wav(wav_path, audio)
            print(f"  TX[{self.n}] {what}{tuned}  ({dur:.1f}s)  -- dry run -> {wav_path}")
        # WHERE OUR CARRIER ACTUALLY WAS, kept rather than re-derived. The
        # sending-side guard needs to know that a codeword it decoded cannot
        # have arrived under one of these; a modular projection of the raster
        # would answer the same question with its own premise. See
        # `_forecast_next_key`.
        if self.tx_key_up is not None and self.tx_end is not None:
            self.keyings.append((self.tx_key_up, self.tx_end))
            del self.keyings[:-KEYINGS_KEPT]


class _LiveInput:
    """A continuously-open duplex stream -- the session's master clock, and its
    transmitter.

    Opening a fresh recording after unkeying costs stream-startup latency on top
    of the rig's T/R switch, and a PACTOR peer answers inside its 1.25 s cycle --
    about 0.29 s after our burst ends. That window was being spent opening the
    device, so an answer the operator could hear on the speaker never reached the
    capture at all. Keeping the stream open removes the startup entirely: audio is
    always flowing, we simply discard what arrives while we are transmitting.

    It is also what the cycle grid is MADE OF. The codec emits 48000 samples a
    second whatever the OS is doing, so a boundary named by sample index is
    immune to the scheduler; a boundary named by `time.monotonic()` is not.
    Measured on this machine under load, sleeping to an absolute deadline lands
    within 2 ms at the median and 70 ms at the tail -- and 70 ms is a quarter of
    the 0.29 s window a peer's answer has to fit inside, arriving without
    warning. Waiting for sample N to be delivered has no such tail: it asks the
    OS only to keep a buffer fed, which is a latency requirement with tens of
    milliseconds of slack rather than a precision one. The last hop onto a
    boundary is a short sleep, and `holdback` is how far short of it a reader
    has to stop for that sleep to have anything to bridge.

    So readers ask for absolute sample POSITIONS (`take_until`), and `pos` --
    the index of the next sample not yet handed out -- is where the session
    thinks it is on the raster.

    THE OUTPUT SIDE OF THE SAME STREAM IS THE TRANSMITTER. `transmit` schedules
    a burst on the output frame counter, which is the input counter -- one stream
    means one counter -- so the capture index our carrier came up on is an
    arithmetic fact rather than an estimate, and PTT is timed against the DAC
    clock instead of a sleep. `out_device=None` opens capture only, for the
    listening tools that key through `ota._play`.

    `record` ARMS THE SESSION'S ONE UNBROKEN RECORDING, in front of the discard
    floor: the whole capture stream reaches a file while the per-window captures
    go on starting at our own data end. Two questions the windows cannot answer.

    WHETHER A DRIFT IS THE PEER'S OR OURS, which needs the stretch between two
    windows and not just the windows. Concatenating them splices out every span
    we were keyed, so a walk across them is observable and not attributable.

    AND WHERE A BURST ALREADY RUNNING AT A WINDOW'S FIRST SAMPLE BEGAN.
    `rxfront.p1_burst_onsets` will not call such a run an onset -- no rising edge
    was seen -- and drops the first half-window (10 ms) besides, so from a window
    alone "the peer started under our carrier" and "the peer started inside the
    55 ms our receiver is still muted for" (`TR_SWITCH_S`) are one observation.
    They are two in the file: the muted span is IN it at its measured -50 dB, so
    a run either rises inside that span or steps up as the mute lifts, and the
    profile is continuous across the window's start either way.

    Same timebase as the windows by construction rather than by correlation: the
    file is fed from the callback before `_floor` is consulted, so its sample k
    is capture-stream sample k, the `end_stream_sample` a window's sidecar
    carries indexes straight into it, and the `(first, end)` `transmit` returns
    say which stretches of it were our own carrier without a log to correlate
    against.
    """

    def __init__(self, device, out_device=None, fs: int = FS, *,
                 blocksize: int = 128, latency="low", record=None,
                 tx_latency_n: int = 0):
        import queue
        import sounddevice as sd
        self._q: queue.Queue = queue.Queue()
        self.fs = fs
        self.samples = 0          # delivered by the codec, not by the clock
        self.pos = 0              # index of the next sample a reader will get
        self.xruns = 0
        self.xrun_at: int | None = None
        self.underruns = 0        # output starved -- a hole in our OWN emission
        self.lost = 0             # captured, timestamped, never handed to Python
        self.holdback = 0         # see below: how far short of a deadline to read
        self._rest = np.zeros(0, np.float32)     # tail of a part-consumed block
        self._floor = 0           # capture below this is ours, and is dropped
        self._fit: list[tuple[float, int]] = []  # (ADC instant, samples), thinned
        self._ahead: float | None = None   # converter timeline minus samples, s
        self._last = (time.monotonic(), 0)       # (ADC instant, its sample index)
        # Bounded callback-entry history, sampled by the scheduler before keying.
        # Keep wall delivery distinct from the ADC's uniform sample clock.
        self._callback_timing = deque(maxlen=256)
        self._dac = (time.monotonic(), 0)        # ...and the same for the DAC
        self._duplex = out_device is not None
        self._lat: int | None = None   # samples from a sample's ADC to its DAC
        #: ...AND THE PART OF THE SAME PATH THE DRIVER DOES NOT REPORT, from
        #: `[audio] tx_latency_ms` or `--tx-latency-ms`. `_lat` is the
        #: converter's own timestamp difference; the codec's in-and-out delay is
        #: not in it, so every burst reached the air that much later than the
        #: index it was scheduled at. `arm-v23-A-40-ws8eoc` put it at 19.8 ms
        #: from the peer's PACTOR-1 answer position and at ~20 ms from what the
        #: PACTOR-3 witness chain needs to close -- the same number by two
        #: routes. It belongs to the audio path, so it is spent on every burst
        #: of every protocol and lives in no protocol's constant.
        self.tx_latency_n = int(tx_latency_n)
        self._tx: Optional[np.ndarray] = None    # armed burst, cleared when sent
        self._tx_at = 0                          # output frame it starts on
        self._tx_done = threading.Event()
        self._tx_end_time = 0.0                  # when its last sample leaves
        self.keyed_s = 0.0        # how long PTT was up for the last transmission
        self.keyed_at: int | None = None   # ...and the capture sample it went up on
        # ARMED BEFORE THE STREAM STARTS, so the file's sample 0 is the capture
        # stream's sample 0 and `end_stream_sample` in a window's sidecar is an
        # offset into it. Fed from the callback, ahead of `_floor`, which the
        # reader never sees below.
        self._rec = (wav.CaptureRecorder(record, rate=fs, source=str(device))
                     if record is not None else None)
        if self._rec is not None:
            self._rec.facts["first_stream_sample"] = self.samples
        self._rec_stop = threading.Event()
        self._rec_thread: threading.Thread | None = None
        self._rec_fault: str | None = None

        def _fill(out, frames, n0, dac):
            buf = self._tx
            out.fill(0)
            if buf is None:
                return
            a, b = max(n0, self._tx_at), min(n0 + frames, self._tx_at + buf.size)
            if b > a:
                out[a - n0:b - n0, 0] = buf[a - self._tx_at:b - self._tx_at]
            if b >= self._tx_at + buf.size:
                self._tx = None
                self._tx_end_time = dac + (self._tx_at + buf.size - n0) / fs
                self._tx_done.set()

        def _cb(ind, out, frames, t, status):
            # An overflow does not accumulate and does not announce itself: lose
            # 480 samples and the cycle grid shifts 10 ms permanently, silently,
            # with none of the walk that made the drift visible. A capture taken
            # across one is measuring a different grid than the log says, so the
            # index it happened at is kept and the session is called invalid.
            #
            # Tested by flag rather than on the status word as a whole, which is
            # what an input-only stream could afford: a duplex stream also raises
            # `priming_output` as it starts, and an output underrun is a hole in
            # what we transmitted rather than a shift in what we heard.
            if status.input_overflow or status.input_underflow:
                self.xruns += 1
                if self.xrun_at is None:
                    self.xrun_at = self.samples
            if status.output_underflow:
                self.underruns += 1
            now = time.monotonic()
            # The stream clock and the system clock need not share a base, so the
            # ADC and DAC instants come across as differences from `currentTime`,
            # which is read at this same instant. Measured here they agree to
            # 82 us, but that is a property of this host API and not a promise.
            base = now - t.currentTime
            n0 = self.samples
            self._callback_timing.append((n0, frames, now, t.inputBufferAdcTime))
            # A rate fit downstream reads what this counts as a crystal
            # instead, which is what -4860 ppm was.
            ahead = t.inputBufferAdcTime - n0 / fs
            self.lost += rates.lost_step(ahead, self._ahead, blocksize, fs)
            self._ahead = ahead
            blk = ind[:, 0].copy()
            self._q.put((t.inputBufferAdcTime + base, n0, blk))
            if self._rec is not None:
                self._rec.push(blk)
            self.samples = n0 + frames
            # Anchored on the ADC instant of sample n0, not on callback entry.
            # Callback entry carries the driver's delivery jitter -- 3 ms rms
            # here, a whole PACTOR bit -- while the converter's own timestamps
            # step by exactly one block, forever.
            self._last = (t.inputBufferAdcTime + base, n0)
            self._dac = (t.outputBufferDacTime + base, n0)
            if self._duplex and self._lat is None and not status.priming_output:
                self._lat = round((t.outputBufferDacTime
                                   - t.inputBufferAdcTime) * fs)
            # WITHOUT `base`, unlike every other consumer above -- see
            # `clock_ppm`.
            if not self._fit or t.inputBufferAdcTime - self._fit[-1][0] >= 1.0:
                self._fit.append((t.inputBufferAdcTime, n0))
            if out is not None:
                _fill(out, frames, n0, t.outputBufferDacTime + base)

        # ONE STREAM OWNS THE DEVICE, and the small buffer is downstream of that.
        #
        # The dongle's default hands over 4096 frames at a time (85.3 ms), and
        # 1.25 s is 14.65 of those, so a boundary's phase inside a block rotates
        # and the instant we can ACT on one walks a sawtooth -- measured, median
        # 21.7 ms and max 42.9 ms. Asking for a small buffer fixes that (median
        # 0.06 ms over 460 slots) and, on the air, once stopped the capture dead
        # 20 ms in: the read position froze at sample 960 and eleven connect
        # bursts went out into a receiver that was not listening.
        #
        # What it had in common with the other audio failure of that week -- a
        # persistent OutputStream dropping capture RMS 16x -- is TWO PortAudio
        # streams on one CoreAudio device. Two streams contend for it; two
        # streams in one process draw err -50 outright. A duplex stream cannot
        # contend with itself, and it is also the only way to know when a sample
        # leaves the DAC, which is what the per-burst `sd.play` cost 143 ms of
        # keyed dead air for.
        #
        # The failure it replaces is silent and total -- the grid arithmetic
        # stays perfectly self-consistent while `pos` never moves -- so this
        # wants a supervised on-air test, and `tests/shrike/test_duplex.py` is
        # what can be checked without one.
        if out_device is None:
            self._stream = sd.InputStream(
                device=device, channels=1, samplerate=fs, dtype="float32",
                blocksize=blocksize, latency=latency,
                callback=lambda ind, frames, t, status: _cb(
                    ind, None, frames, t, status))
        else:
            self._stream = sd.Stream(
                device=(device, out_device), channels=(1, 1), samplerate=fs,
                dtype="float32", blocksize=blocksize, latency=latency,
                callback=_cb)
        self._stream.start()
        if self._rec is not None:
            # Off the callback, and off the session loop: the loop is inside
            # `transmit` for most of a keyed cycle, so draining from it would
            # hold a whole burst's audio in memory and then write it in the gap
            # a transmission is scheduled out of.
            self._rec_thread = threading.Thread(target=self._record,
                                                name="shrike-rec", daemon=True)
            self._rec_thread.start()
        # Do not hand back a stream that has not started. The session's first act
        # is the connect burst, and it arrives before the first callback does:
        # nothing had been captured, so there was no ADC/DAC pair to schedule the
        # carrier against and the run stopped on its first transmission. It takes
        # about 15 ms here. A stream that never starts is the silent-total
        # failure this class exists to make loud, so it says so rather than
        # letting the session call into a receiver that is not listening.
        deadline = time.monotonic() + 2.0
        while self.samples == 0 or (self._duplex and self._lat is None):
            if time.monotonic() > deadline:
                self.close()
                raise SystemExit(
                    f"the audio stream on device {device!r} started but delivered "
                    f"nothing in 2 s. Transmitting now would be calling into a "
                    f"channel we cannot hear.")
            time.sleep(0.002)
        # How far short of a deadline a reader has to stop for `wait_until` to
        # have anything left to bridge. A block, PLUS the input latency: the
        # grid is anchored on the converter's own timestamps, so a sample exists
        # one latency before the driver hands it over, and stopping only a block
        # short means every read returns after the instant it was aiming at.
        # Measured before this term was here: PTT went up 0 to 8 ms late,
        # walking cycle to cycle because 1.25 s is 468.75 blocks and the phase
        # rotates. It comes out of the decode window, where 13 ms is affordable
        # and the 85 ms the device's default block cost was not.
        #
        # AND IT IS WHERE THE INPUT LATENCY CEILING IS. `take_until`/`wait_until`
        # return one holdback late, so the sleep in front of `key(True)` has
        # `settle - holdback` left and the PTT lead is what remains. Measured
        # 2026-08-26 at a 40 ms settle: 10.7 ms of latency leaves a 26.6 ms lead,
        # 21 ms leaves 16.3, 40 ms leaves -2.7 and 85.3 ms leaves -48. A negative
        # lead is the carrier coming up after its own audio has started, so the
        # latency has to stay under about 27 ms. It cannot be bought back with a
        # larger settle: the workable turnaround is
        # `cycle - packet - cs - settle`, which is 130 ms at a 40 ms settle and
        # 45 ms at the ~125 ms an 85 ms holdback would need -- under all but three
        # of the turnarounds this station has acquired, and under the median of
        # `PEER_TURNAROUND_S`.
        lat = self._stream.latency
        self._blk = self._stream.blocksize or 128
        self.holdback = self._blk + round(
            (lat[0] if isinstance(lat, tuple) else lat) * fs)
        self._breathing = contextlib.ExitStack()
        self._breathing.enter_context(gil.breathing())

    def _record(self) -> None:
        """Drain to the disk until the session ends, or until the disk refuses.

        A write that fails is the one way this recording loses its tail, and it
        loses it in the shape the whole file exists to avoid -- a shorter file
        that reads as a complete one. So the thread stops at the first fault and
        leaves the reason where `stream_report` will print it; a transmission in
        progress outranks a recording, and a raise here would take the session's
        unkey with it.
        """
        while not self._rec_stop.wait(0.1):
            try:
                self._rec.drain()
            except Exception as exc:              # noqa: BLE001
                self._rec_fault = str(exc)
                print(f"  !! session stream stopped writing -- {exc}", flush=True)
                return

    def stream_report(self) -> str:
        """What the unbroken recording of this session cost, after `close`.

        48 kHz mono int16 is 5.8 MB a minute, and the arms of an A/B are bounded
        at 200 s apiece by the launcher that runs them, so one comparison costs
        under 40 MB against the several gigabytes of recordings this station
        already keeps. Nothing here truncates. A cap that stopped the recorder
        mid-session would cut exactly the span being measured, which is the shape
        of the loss this station measured out of ffmpeg's avfoundation input:
        0.904 of real time in samples, nothing logged at -loglevel error, and
        nine capture windows short by 9% in the middle because of it.
        """
        if self._rec is None:
            return "no continuous recording was taken"
        if self._rec_fault is not None:
            return (f"session stream: {self._rec.path} is INCOMPLETE after "
                    f"{self._rec.seconds:.1f} s -- {self._rec_fault}. What is in "
                    f"it is still on the session's sample clock; what is missing "
                    f"is everything after that.")
        # The whole path, not the basename: the line an operator reads at three
        # in the morning is the one they paste into the analysis.
        return (f"session stream: {self._rec.path} -- "
                f"{self._rec.seconds:.1f} s, "
                f"{self._rec.path.stat().st_size / 1e6:.1f} MB, on the sample "
                f"clock every capture beside it is indexed on")

    def clock_ppm(self) -> float:
        """Error of the capture clock against the system clock, in ppm.

        The one term a sample-locked grid does not remove, and nobody had
        measured it: the audio device is a separate USB dongle, so this is a
        cheap part's crystal rather than the transceiver's.

        The fit points go in WITHOUT `base`, unlike every other consumer in the
        callback. `base` is how late Python was to that callback, and the readers
        need it because they compare against `time.monotonic()`; a rate fit does
        not, and cannot afford it. Idle it is 90 us and harmless, but it is the
        one term there that grows without bound under a contended interpreter,
        and half a second of it over a minute is -8000 ppm of fiction on a
        crystal this station measures at +4 to +7.
        """
        return rates.rate_fit(self._fit, self.fs)[0]

    def clock_report(self) -> str:
        return rates.clock_report(
            samples=self.samples, lost=self.lost, xruns=self.xruns,
            underruns=self.underruns, points=self._fit, blocksize=self._blk,
            fs=self.fs)

    def sample_now(self) -> float:
        """Where the capture stream stands at this instant, between callbacks.

        `samples` only moves once a block, and even 2.7 ms of quantisation is a
        tenth of the per-cycle timing budget, so this interpolates from the last
        block's ADC timestamp at the nominal rate -- good to a microsecond over a
        block, since the rate is right to 1.4 ppm.

        The CONVERTER's position, which leads the delivered count by the input
        latency: sample `sample_now()` exists but is still in the driver. That is
        a constant, and it is the SAME constant for the grid boundaries and the
        reply onsets measured against it, so differences here are exact and only
        the absolute offset to the air carries it.
        """
        t, n = self._last
        return n + (time.monotonic() - t) * self.fs

    def wait_until(self, index: int) -> float:
        """Sleep out the wall-clock remainder to sample `index`; returns ms slept.

        The last hop of the raster, and the only place a clock is still allowed.
        A read cannot return before the block CONTAINING its last sample lands,
        so waiting on samples alone can only ever arrive late, by up to a block.
        The caller therefore stops a block short and sleeps the rest, which is a
        sub-block sleep to an instant computed off the sample grid: it inherits
        the scheduler's jitter for that last hop but none of its drift, because
        the instant itself is derived from the samples and not from the last
        cycle.

        To the sample's CAPTURE, so it returns one input latency before that
        sample can be read. Nothing downstream keys off this any more -- the
        carrier is scheduled on the DAC clock by `transmit` -- so what the
        difference costs is that much of the window's tail, which the transmit
        would have discarded anyway.
        """
        t, n = self._last
        deadline = t + (index - n) / self.fs
        d = deadline - time.monotonic()
        self._sleep_until(deadline)
        return max(0.0, d) * 1e3

    # -- the output half ---------------------------------------------------
    @property
    def key_notice(self) -> int:
        """Capture samples of notice the transmitter needs before an aim point.

        The measured input-to-output latency plus three blocks -- the callback
        that fills the buffer after next cannot be pre-empted out from under us
        at 2.7 ms a block. `transmit` schedules with the same two terms, and
        with `tx_latency_n`, which moves the enqueue deadline forward by exactly
        what it moves the audio.
        """
        return (self._lat or 0) + self.tx_latency_n + 3 * self._blk

    def clamp_late(self, at: int) -> int:
        """How late the earliest schedulable audio would be relative to `at`.

        The old transmitter silently clamped to `samples + key_notice`, moving
        the burst while its own timing reports remained self-consistent. The
        driver uses this advance check to recover a later slot; `transmit` also
        refuses an expired explicit deadline at enqueue, before arming output.
        """
        return max(0, self.samples + self.key_notice - at)

    def _dac_time(self, index: int) -> float:
        """When output frame `index` reaches the DAC, on the system clock."""
        t, n = self._dac
        return t + (index - n) / self.fs

    def _sleep_until(self, t: float) -> None:
        """Sleep to a deadline the raster and the PTT edges are measured against.

        In hops of half a millisecond, never one long sleep. `time.sleep` is good
        to 60 us over 200 us here and lands up to 9 MS LATE over 20 ms --
        measured, both, on this machine -- because the OS coalesces long timers.
        That was two thirds of the observed PTT jitter and a third of the
        transmit slot. Short sleeps rather than a spin because the audio callback
        is Python and wants the GIL back every 2.7 ms.
        """
        while True:
            d = t - time.monotonic()
            if d <= 0:
                return
            time.sleep(min(d, 1e-3) / 2)

    def transmit(self, audio: np.ndarray, *, at: Optional[int] = None,
                 settle: float = 0.0, key=None,
                 max_key: float = 40.0) -> tuple[int, int]:
        """Send `audio` so its first sample reaches the DAC as capture sample
        `at` reaches the ADC. Returns (first, end) capture-stream indices.

        `end` is exclusive -- the first sample after our carrier dropped -- so
        `flush_to(end)` starts the next read on real peer audio and the reply
        onset is counted from a sample rather than from a block boundary.

        An explicit `at` that no longer leaves DAC notice raises `_MissedTxSlot`
        before output or PTT is armed. An ungridded `at=None` still takes the
        earliest time that leaves the configured settle.

        The two counters are one counter (one stream, one callback), so the
        capture index a transmitted sample lands on is the output frame plus a
        fixed latency. That latency is MEASURED from the callback's own ADC and
        DAC timestamps rather than assumed: 1152 samples on this machine, with a
        standard deviation of zero over 22208 callbacks.

        PTT is asserted `settle` before the burst's DAC instant and released at
        its last sample's, so the keyed window is `settle` plus the audio and
        nothing else. It was 144 ms more than that: per-burst stream startup put
        40 ms of dead key at the front and CoreAudio's stream teardown -- not
        buffered audio -- another 103 ms at the back, of a 289 ms gap the far end
        answers in.

        `key` is called with True and then False and is unkeyed again from the
        `finally`, whatever happens above it, plus once more by a watchdog: a rig
        left transmitting is the one failure that must not happen. How long it
        was actually asserted is left in `keyed_s` -- measured, because that is
        the number the cycle budget is spent out of.
        """
        if not self._duplex:
            raise SystemExit("this stream was opened for capture only -- there is "
                             "no output device to transmit through")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        dur = audio.size / self.fs
        # Refuse rather than key and cut short. Truncating a legitimate
        # transmission is worse than not sending it -- a clipped WSPR frame
        # (110.6 s, against a 40 s default) once read as a dead antenna and cost
        # a real hardware misdiagnosis.
        if dur > max_key:
            raise SystemExit(f"refusing to transmit: {dur:.1f}s of audio exceeds "
                             f"--max-key {max_key:.0f}s. Raise --max-key to at "
                             f"least {dur + 5:.0f} for this transmission.")
        # Three blocks of notice: the callback that fills the buffer after next
        # cannot be pre-empted out from under us at 2.7 ms a block.
        earliest = self.samples + 3 * self._blk
        if at is None:
            # No slot to aim at -- the connect burst, sent before there is a
            # grid. Take the whole settle rather than the earliest instant the
            # stream could manage: that burst is the only thing a station being
            # called has to lock onto, and at the earliest instant the rig would
            # get a quarter of its settle and put out a short one.
            t, n = self._dac
            start = max(earliest,
                        n + round((time.monotonic() + settle - t) * self.fs))
        else:
            # EARLIER BY THE WHOLE LOOP, not by the half the driver reports. The
            # contract in the first line of this docstring was not being kept:
            # the burst reached the air `tx_latency_n` after the index it names.
            # Corrected here rather than in a protocol constant -- `P3_REPLY_S`
            # 0.890 is measured from SCS traffic and is right -- so the same
            # correction is spent on PACTOR-1, PACTOR-2, ARDOP and an ID alike.
            # PTT follows it: the key is `settle` in front of the burst's DAC
            # instant, which has just moved, and `_prekey_lead` moves the drain
            # in front of it by the same amount, so the lead is untouched.
            start = at - self._lat - self.tx_latency_n
            if start < earliest:
                # This is the last admission check before arming the callback.
                # An explicit sample is a protocol deadline, not permission to
                # slide the burst into the peer's packet after a Python stall.
                raise _MissedTxSlot(earliest - start)
        first = start + self._lat + self.tx_latency_n
        self.keyed_at = None
        self._tx_done.clear()
        self._tx_at = start
        self._tx = audio                  # ...last: it is what arms the callback
        stop = threading.Event()
        limit = settle + dur + max(2.0, 0.25 * dur)

        def watchdog():
            if not stop.wait(limit):
                self._tx = None
                try:
                    if key:
                        key(False)
                except Exception:
                    pass
                print(f"!! PTT watchdog fired at {limit:.0f}s -- force unkey")
        threading.Thread(target=watchdog, daemon=True).start()
        up = None
        try:
            self._sleep_until(self._dac_time(start) - settle)
            if key:
                # An explicit refusal only: a stub key returns None, which says
                # nothing about a rig. `Rig.ptt` returns False when the command
                # did not reach rigctl (and raises when the device itself is
                # gone), and audio into an unkeyed transmitter is a cycle spent
                # talking to nobody -- or worse, half of it keys late and lands
                # on top of the far end. `PttError` rather than a bare exit so
                # the session can end saying WHY in its summary.
                if key(True) is False:
                    raise PttError("PTT did not reach the rig -- not "
                                   "transmitting into an unknown keying state")
                up = time.monotonic()
                # On the converter's own timeline, which is the one the peer's
                # burst onsets are read off, so the interval between the two is
                # a subtraction. `settle` says what was intended; this says what
                # the rig got, and the holdback ahead of the key means they are
                # not the same number.
                self.keyed_at = int(self.sample_now())
            if not self._tx_done.wait(limit):
                raise SystemExit("the output stream never sent the burst -- "
                                 "unkeyed and stopping")
            self._sleep_until(self._tx_end_time)
        finally:
            self._tx = None
            if key:
                try:
                    key(False)
                except Exception:
                    pass
                self.keyed_s = time.monotonic() - up if up else 0.0
            stop.set()
        return first, first + audio.size

    def flush_to(self, index: int) -> None:
        """Drop capture up to `index` -- our own transmission, to the sample.

        `flush(before=...)` can only land on a block boundary, because a wall
        clock reading is all the per-burst playback could report about its own
        unkeying. The duplex stream knows the capture index its last sample left
        the DAC on, so the next window starts exactly there.
        """
        self._rest = np.zeros(0, np.float32)
        self._floor = self.pos = index

    def flush(self, before: float | None = None) -> None:
        """Drop audio captured BEFORE `before` -- our own transmission.

        Emptying the queue outright is what this used to do, and it threw away
        the beginning of the peer's answer along with our burst. The queue is
        drained after `_play` returns, by which point the rig has been unkeyed
        for some milliseconds and the far end -- measured on a third-party
        receiver at 70-190 ms after our carrier drops -- may already be
        transmitting. Its opening symbols were sitting in that queue, and we
        deleted them, then recorded that nobody answered.

        Blocks carry the ADC instant of their first sample, so "discard what we
        were sending" and "discard what arrived after we stopped" are different
        things. This is the one place a wall-clock time enters the sample domain,
        and it is here for the listening tools that still key through
        `ota._play`, which can only report its unkeying as a clock reading. The
        session's own transmit path knows the sample and uses `flush_to`.
        """
        import queue as _q
        keep = []
        while True:
            try:
                item = self._q.get_nowait()
            except _q.Empty:
                break
            if before is not None and item[0] >= before:
                keep.append(item)
        self._rest = np.zeros(0, np.float32)
        self._floor = self.pos = keep[0][1] if keep else self.samples
        for item in keep:
            self._q.put(item)

    def read(self, count: int) -> np.ndarray:
        """The next `count` samples, blocking until the codec has delivered them.

        Blocks below the discard floor are skipped rather than handed out. A
        `flush_to` past the end of the queue is the ordinary case -- our last
        sample leaves the DAC before the capture of that instant has been
        delivered -- and without the floor those in-flight blocks would come back
        as if they were the peer's answer, shifted by their whole length.
        """
        import queue
        got: list[np.ndarray] = []
        n = 0
        if self._rest.size:
            got.append(self._rest[:count])
            n = got[0].size
            self._rest = self._rest[n:]
        while n < count:
            try:
                _t, i, blk = self._q.get(timeout=count / self.fs + 2)
            except queue.Empty:
                break
            if i + blk.size <= self._floor:
                continue
            if i < self._floor:
                blk = blk[self._floor - i:]
            if n + blk.size > count:
                blk, self._rest = blk[:count - n], blk[count - n:]
            got.append(blk)
            n += blk.size
        self.pos += n
        return np.concatenate(got) if got else np.zeros(0, np.float32)

    def take(self, seconds: float) -> np.ndarray:
        return self.read(int(seconds * self.fs))

    def read_ready(self) -> np.ndarray:
        """Everything already delivered, without waiting for more.

        The window stops a block short of the boundary so the bridge sleep has
        something to bridge, and the audio in that gap is real peer audio that
        `flush(before=unkey)` would throw away the moment we key -- so it is
        collected here instead, right before the key goes up. What is left
        behind is only what the codec has not handed over yet, which is nobody's
        to have.
        """
        return self.read(max(0, self.samples - self.pos))

    def take_until(self, index: int) -> np.ndarray:
        """Audio from the read position up to absolute sample `index`.

        Returns nothing if we are already past it -- a cycle that overran is
        absorbed here rather than displacing every cycle after it, exactly as an
        absolute deadline absorbs one, but without the scheduler in the loop.
        """
        return self.read(max(0, index - self.pos))

    def close(self) -> None:
        self._breathing.close()
        try:
            self._stream.stop(); self._stream.close()
        except Exception:
            pass
        # After the stream, so no callback can push past the last drain, and the
        # writing thread joined before the file is closed -- closing a WAV under
        # a drain in flight races it. The counters are the ones every window
        # sidecar carries, and they condemn a file and cannot clear one: see the
        # `grid_loss_seen` note in `_save_capture` for what neither of them sees.
        self._rec_stop.set()
        if self._rec_thread is not None:
            self._rec_thread.join(timeout=2.0)
            self._rec_thread = None
        if self._rec is not None:
            self._rec.facts |= {"lost_samples": self.lost, "xruns": self.xruns,
                                "grid_loss_seen": bool(self.xruns or self.lost),
                                "truncated_by": self._rec_fault}
            try:
                self._rec.close()
            except Exception as exc:              # noqa: BLE001
                # The sidecar is what says how long this file is, so a close that
                # fails leaves one nobody can grade. Said out loud here and again
                # by `stream_report`; the rig is already down by now and stays down.
                self._rec_fault = self._rec_fault or str(exc)
                print(f"  !! session stream NOT closed -- {exc}", flush=True)


class _ReplayInput:
    """A recorded capture standing in for the sound card, same interface.

    The session loop is the part of shrike that cannot be exercised offline --
    every claim about whether it holds an ARQ cycle needed a radio and a willing
    gateway, so it was tested by transmitting at strangers. Replaying a capture
    through the identical path makes the loop's timing and the FSM's responses
    reproducible on the corpus. It is not a substitute for the air: the far end is
    a recording and cannot react to us. It answers the questions that do not need
    a reactive peer -- does an event reach the FSM, and how late.
    """

    def __init__(self, path: str, fs: int = FS, *, realtime: bool = False):
        self.audio = _load(path)
        self.path = path
        self.pos = 0
        self.fs = fs
        self.realtime = realtime
        self.xruns = 0
        self.xrun_at = None
        # A file is handed over entire. Whatever the session that made it lost is
        # already missing from the samples and is not this replay's to count.
        self.lost = 0
        self.holdback = 0         # a file has no blocks to hold back for
        self._bridged = np.zeros(0, np.float32)

    @property
    def samples(self) -> int:
        """A file delivers exactly what is read from it, so the two agree."""
        return self.pos

    def clock_ppm(self) -> float:
        """A recording has no clock of its own -- it has the one it was made on."""
        return float("nan")

    def clock_report(self) -> str:
        return f"replay: {self.pos} samples read; a file has no clock to measure"

    def stream_report(self) -> str:
        """The unbroken recording of a replay is the recording it was driven from."""
        return f"session stream: not taken -- this session was driven from {self.path}"

    def flush(self, before: float | None = None) -> None:
        """No queue to drop; a replay has no live backlog to fall behind on."""

    def read(self, count: int) -> np.ndarray:
        seg = self.audio[self.pos:self.pos + count]
        self.pos += len(seg)
        if self.realtime and len(seg):
            time.sleep(len(seg) / self.fs)
        return seg

    def take(self, seconds: float) -> np.ndarray:
        return self.read(int(seconds * self.fs))

    def take_until(self, index: int) -> np.ndarray:
        return self.read(max(0, index - self.pos))

    def read_ready(self) -> np.ndarray:
        """Whatever the bridge below skipped over.

        A live input holds that span in its queue and hands it back here; a file
        has to be told to keep it, and dropping it would lose the peer audio
        between an early break and the slot boundary -- which is most of the
        cycle, and the part a link-setup answer arrives in.
        """
        seg, self._bridged = self._bridged, np.zeros(0, np.float32)
        return seg

    def sample_now(self) -> float:
        return float(self.pos)

    key_notice = 0

    def clamp_late(self, index: int) -> int:
        """A file has no transmitter and no wall clock to be late against: the
        only instant it cannot key on is one the reader has already gone past."""
        return max(0, self.pos - index)

    def wait_until(self, index: int) -> float:
        """A recording has no clock to sleep against -- so the file IS the clock,
        and waiting for sample `index` means advancing to it.

        This used to do nothing at all, on the reasoning that a file is already
        where it is. The read position then stayed wherever the last window ended
        while the grid marched on, and after an early break -- which truncates a
        window to 0.43 s of a 2.5 s slot -- the two diverged by seconds and never
        recovered: measured on a replayed session, -1.1 s at the first break and
        -10.7 s eleven cycles later. The recording it was measured on is not on
        disk under any name, so the figures are what survives of it. A replay in that state exercises the state
        machine over audio with no relation to the slot it is supposedly in, and
        the off-grid figure it prints describes the harness rather than the
        session. Nothing sleeps here; the file's own samples are the clock.
        """
        self._bridged = self.read(max(0, index - self.pos))
        return 0.0

    def close(self) -> None:
        pass


class _SessionRx:
    """One rolling decoder for the whole session, fed every chunk we take.

    The loop used to decode in three separate shapes -- the listen window, the
    hold window, and the early-break test -- each starting from a clean buffer, so
    a burst straddling two of them was seen by none, and the FSM learned nothing
    until a window ended. That is survivable for catching a connect answer, since
    a gateway retries, and fatal for a multi-cycle QSO, where every turnaround has
    its own answer that is not repeated. A single rolling decoder sees one
    continuous stream and delivers events a slide after the audio carrying them.
    """

    def __init__(self, host, tag: str = "RX"):
        self.host, self.tag = host, tag
        # Retires the optional call-phase experiment after the first payload
        # settles, a role change, or departure from initial P1 setup. This is
        # session-local: later retransmissions/counter wraps cannot re-arm it.
        self.p1_setup_finished = False
        self.p1_setup_phase = getattr(
            getattr(host, "peer", None), "p1_setup_phase", "reply")
        self.rx = live.RollingRx(self._on, window_s=RX_WINDOW_S,
                                 keep_s=RX_WINDOW_S - RX_SLIDE_S,
                                 p3_packets=lambda: not (
                                     self.host.protocol == Protocol.PACTOR1
                                     and self.host.arq.state == State.CONNECTING))
        self.cs_seen = False         # a control signal already reached the FSM
        self.frame_seen = False      # ...and a data frame. One of each per cycle
        # This cycle's PACTOR-3 codeword, where a reader has already decoded one
        # at an instant another will ask for: the stream sample it was AIMED at,
        # the stream sample it was FOUND on, and the event.
        self._p3_word_read: Optional[tuple[int, int, rxfront.Event]] = None
        # Stream time this station was not listening to -- our own carrier, and
        # audio taken out of the stream without being fed. Owed to the clock, and
        # paid at the next decode rather than at the moment it is declared; see
        # `skip`.
        self._unheard = 0.0
        self.count = 0
        # Every control signal the FSM was actually given, kept in order for the
        # end-of-session summary. The session's own decodes and nothing else --
        # a summary rebuilt from the saved captures would report a receiver that
        # is not the one that ran.
        self.cs_log: list = []
        # Where in the session's clock a twelve-bit word has already been read,
        # ANY word: `cs_seen` counts only the four with a meaning, and the two
        # unassigned ones are still one transmission apiece. See
        # `_read_codeword_at_bursts`, which is the second reader of the same
        # audio and the only thing that asks.
        self.words_at: list[float] = []
        # ...and where the ANCHORED read has delivered one, for the session and
        # not the cycle. See `_on`, which is what reads it.
        self._anchored_at: list[float] = []
        # PACTOR-1 has no FEC, so the peer's retransmissions are the only
        # redundancy it sends and combining them is the only way to spend it.
        # One per link and one per direction: the memory groups by consecutive
        # failure, and both a disconnect and a changeover end the run of copies
        # it is holding. See `p1rx.PacketMemory` and `new_cycle`.
        self.p1_memory = p1rx.PacketMemory()
        # PACTOR-3's, on the same rule and for the same reason it is here rather
        # than inside the decode: `p3rx.decode_p3_packets` builds one per call,
        # so a receiver that scans once a cycle held exactly one copy and could
        # never combine two. Not fed from the rolling decoder -- its windows
        # overlap, so the same failed frame would enter the sum four times and
        # push the peer's real repeats out.
        self.p3_memory = p3rx.FieldMemory()
        # Tracks where in the cycle the peer's data frame lands. Fed only from
        # `deep_scan`, never from the rolling decoder: a lock is an offset into the
        # buffer it was found in, and `RollingRx` slides its buffer out from under
        # one. So the ISS's acknowledgements, which arrive through the rolling
        # scan, stay on the blind path here; the IRS's data frames, which arrive on
        # the cycle's own audio, do not.
        #
        # ON THE SESSION'S OWN MEMORY, because the peer sends one run of copies
        # and the two readers see it a cycle at a time between them: the tracked
        # read holds the cycles the lock answers and the sweep the ones it
        # misses, so two memories each hold a fraction of one field and the sum
        # is in neither. What clears the session's clears this read's, which is
        # what the delivery, the role change and the speed level's own key check
        # already do.
        self.sync = rxfront.SyncedRx(memory=self.p3_memory)
        # Which 25 Hz carrier pair the peer's PACTOR-2 frames armed on, kept
        # because the codeword read cannot find its own: `p2rx.find_markers`
        # reports the pair off every burst and `p2rx.control_signal_at` has to
        # be told one. `CENTRE_BIN_PAIR` is 1500 Hz, which is where a link that
        # has not yet decoded a frame has to assume the peer is.
        self.p2_bin_pair = p2rx.CENTRE_BIN_PAIR
        self._p1_role = host.arq.role
        self.p3_receive_offset_hz = 0.0
        # Whether a PACTOR-1 codeword read this session contradicts that. See
        # `_follow_p3_offset`, which is the only writer, and
        # `p3_transmit_offset_hz`, which is what the transmitter reads.
        self._p3_offset_contradicts_p1 = False
        self._p3_offset_fixed = False
        self._p3_offset_candidate = None
        self._p3_corrected = None
        self._p3_changeover_pending = False
        self._p3_acquisition_cycle = 0
        self._p3_head_candidate = None
        self._p3_entry_body_candidate = None
        self._p3_entry_body_checked_at = None
        self._p3_answer_at = None
        self._final_connect_listen = False
        self._scan_origin: Optional[int] = None
        self._tracked_only = False
        self._p3_row0: Optional[int] = None
        self._p3_clock_role = None
        self._p3_cycle_n = round(spec.CYCLE_SHORT_S * FS)
        self._p3_span = rxfront._frame_span(placement.SPEED_PATHS[1])
        self._p3_delivered_at: Optional[int] = None
        self._p3_repeat_changeover = False
        self._p3_sl2_expected_until = None

        self.p3_wideband_prekey = False
        self._p3_long_window_checked_at = None
        self._p3_long_crc_reply = None
        self._p3_prekey_crc = None
        self._p3_stage_ms = {}
        self._p3_control_collect_plan = None

    def wideband_prekey_active(self) -> bool:
        """The opt-in SL3 upgrade path, only with a CRC-established clock."""
        a = self.host.arq
        return (self.p3_wideband_prekey and self.host.protocol == Protocol.PACTOR3
                and a.state in LINKED and a.role == IRS and not a.entry_pending
                and not a.cycle_long and not a.cycle_command_emitted
                and self._p3_cycle_n == round(spec.CYCLE_SHORT_S * FS)
                and self.sync.packet_level in (None, 1, 2, 3)
                and self._p3_row0 is not None)

    def note_p3_speedup_emitted(self) -> None:
        """Reserve an SL2 read for the next packets, without claiming an SL2 lock."""
        if (self.host.protocol == Protocol.PACTOR3 and self.host.arq.role == IRS
                and self.sync.packet_level == 1 and self._p3_row0 is not None):
            self._p3_sl2_expected_until = self._p3_row0 + 3 * self._p3_cycle_n

    def sl2_prekey_expected(self, target: int) -> bool:
        return (self.sync.packet_level == 2
                or (self._p3_sl2_expected_until is not None
                    and target <= self._p3_sl2_expected_until))

    def feed(self, chunk: np.ndarray) -> None:
        if not chunk.size:
            return
        self._settle()
        # Waiting on a link-setup answer is the one moment the receiver has a
        # strong prior about what is arriving and when, and that prior is what
        # pays for the control-signal code's error-correcting radius. Without it
        # the far end came up and we never did: a rendered ACK through any noise
        # at all decodes to the right codeword at 2 bit errors of 20, and the
        # sweeping rule throws that away.
        self.rx.cs_max_errors = (
            rxfront.CS_EXPECTED_MAX_ERRORS
            if self.host.arq.state in (State.CONNECTING, State.CONNECTED)
            else rxfront.CS_MAX_ERRORS)
        # CONNECTING only, and it is a different prior from the one above: not
        # "an answer is due" but "we do not know where in the window it sits".
        # The PACTOR-1 control signal is read at one instant off the burst
        # envelope, which is right for a station holding the grid and wrong for
        # one still looking for it -- measured on real gateway replies recorded
        # with our transmitter off, reading a point finds 1 of WS8EOC's bursts
        # and 0 of W6IDS's where the search finds 17 and 19. It must go off again
        # on CONNECTED: 224 alignments per burst against a two-word alphabet is
        # affordable once per acquisition and is buying false accepts for a grid
        # we already have.
        self.rx.acquiring = self.host.arq.state == State.CONNECTING
        self.rx.push(chunk)

    def bridge(self, chunk: np.ndarray) -> None:
        """Take the bridge's audio into the stream without decoding it here.

        `flush` decodes it a moment later, capped, which is the one decode the
        transmit slot is budgeted for. See `live.RollingRx.hold`.
        """
        self._settle()
        self.rx.hold(chunk)

    def new_cycle(self) -> None:
        """A cycle begins. The peer owes us ONE control signal in it, and one frame.

        Both are one-shot for the same reason. The reverse channel carries a
        single answer per cycle and the state machine emits a single answer to it
        -- so a second delivery of the same transmission sends a second control
        signal on top of the first, and acknowledging a packet twice advances the
        sequence past one the peer never received. Three paths can reach a frame
        in one cycle now (the scan before the key, the packet behind a changeover,
        and a slot the grid handed back), and they overlap in the audio they are
        given.

        AND THE SOFT MEMORY, which is one-shot on the link rather than on the
        cycle. Its copies are grouped by consecutive failure, so what ends a run
        is the peer no longer repeating the packet: a link that has gone down,
        and a changeover, after which the station accumulating copies is the one
        transmitting them. Neither can produce a wrong field -- the header and
        CRC gates refuse a mixed sum -- but both spend cycles summing audio that
        cannot combine.
        """
        self.cs_seen = self.frame_seen = False
        self._p3_word_read = None
        self._p3_prekey_crc = None
        self._p3_control_collect_plan = None
        self._p3_acquisition_cycle += 1
        self.words_at.clear()
        role = self.host.arq.role
        if role != self._p1_role or self.host.protocol != Protocol.PACTOR3:
            self._p3_answer_at = None
            self._p3_head_candidate = None
            self._p3_entry_body_candidate = None
        if self.host.arq.state not in (State.CONNECTING, *LINKED):
            self.p3_receive_offset_hz = 0.0
            self._p3_offset_contradicts_p1 = False
            self._p3_offset_fixed = False
            self._p3_offset_candidate = None
            self._p3_head_candidate = None
            self._p3_answer_at = None
            self._p3_changeover_pending = False
        if (self.host.protocol != Protocol.PACTOR3
                or self.host.arq.state not in (State.CONNECTING, *LINKED)):
            self._p3_delivered_at = None
            self._p3_entry_body_candidate = None
            self._p3_entry_body_checked_at = None
        # Our own CS3 is a proposed turn, not proof that the peer stopped
        # sending. Keep its receive-only packet clock while that turn awaits
        # acknowledgement. In onair-0914-1322 the peer repeated SL4 after our
        # CS3; dropping this clock left readable retries to blind acquisition.
        keep_peer_clock = (self.host.protocol == Protocol.PACTOR3
                           and role == ISS
                           and self.host.arq.unconfirmed_breakin
                           and self._p3_clock_role == IRS
                           and self._p3_row0 is not None)
        # A confirmed/cancelled turn can leave the local role unchanged. Retire
        # its old clock then too, unless a fresh peer frame has already assigned
        # the clock to the current role. No retained clock authorizes a TX.
        old_turn_ended = (role == ISS and self._p3_clock_role == IRS
                          and not keep_peer_clock)
        if role != self._p1_role or old_turn_ended or \
                self.host.arq.state not in (State.CONNECTING, *LINKED):
            self.p1_memory.clear()
            self.p3_memory.clear()
            # A CRC-valid changeover can establish the incoming clock in the
            # very event that changes our role. Keep that fresh clock; discard
            # one belonging to the previous direction or a closed session.
            if (role != self._p3_clock_role and not keep_peer_clock) or \
                    self.host.arq.state not in (State.CONNECTING, *LINKED):
                self._p3_row0 = None
                # The incoming clock changes direction, but the recording's
                # absolute time does not. Keep the delivery watermark so a
                # rolling old frame cannot undo our new role or rewind timing.
                self._p3_clock_role = None
                self.sync.packet_at = None
                self._p3_repeat_changeover = False
                self._p3_sl2_expected_until = None
        self._p1_role = role

    @property
    def cs_heard(self) -> Optional[int]:
        """The codeword the state machine was given this cycle, whoever read it.

        NOT `control_signal`'s return, which answers a narrower question: whether
        the ANCHORED read produced one. That read declines outright the moment any
        other path has already delivered a codeword (see `cs_seen`), and for a
        break-in that is the ordinary case rather than the exception -- a station
        taking the channel transmits where it decides to, not on our grid, so the
        anchor cannot be pointed at it and the sweeping decoder is what reads it.

        MEASURED, K4MSU 3595 kHz on 2026-08-19 22:10 (captures/onair-0819-2210).
        Three CS3s reached the FSM; one came from the anchored read and its field
        was taken, and the two the sweep found reversed the role and were never
        read at all. A caller acting on the changeover acts on this.
        """
        return self._last_cs.cs if self.cs_seen else None

    @property
    def cs_at(self) -> Optional[int]:
        """...and the CAPTURE-STREAM sample it sits on, which is where its
        packet begins: the clock `seg_start`, `live.take_until` and every
        sidecar index are on. See `tests.shrike.test_session_clock`, which is
        what holds the reader's clock to that one."""
        return round(self._last_cs.t * FS) if self.cs_seen else None

    @property
    def _last_cs(self):
        """The last event in the log that carries a CODEWORD, which is not the
        last event in the log: `cs_log` also keeps the unassigned words, so a
        cycle that read `0x59A` after a CS1 would otherwise hand both properties
        above an event whose `cs` is None."""
        return next(ev for ev in reversed(self.cs_log) if ev.cs is not None)

    def expect_frame(self) -> None:
        """A SECOND transmission from the peer inside one cycle.

        The packet behind a changeover head, and the only thing that arrives that
        way: the peer has taken the link and sent 840 ms of data after the
        codeword we were still answering. The one-frame rule above counts
        transmissions rather than cycles, so this one is owed its own scan even if
        the window before the key had already produced a frame.
        """
        self.frame_seen = False

    def changeover_body(self, audio: np.ndarray, origin: int, head: int) -> None:
        """Finish an already decoded P3 CS3 without reacquiring the channel."""
        if (self.host.protocol != Protocol.PACTOR3
                or self.cs_heard != CS_BREAKIN or self.cs_at != head
                or head < origin
                or (self._p3_delivered_at is not None
                    and head <= self._p3_delivered_at + rxfront.SPS)
                or origin + audio.size < head + round(placement.PACKET_S * FS)
                or not self._p3_acquisition_fits(self.CHANGEOVER_BODY_RESERVE_S)):
            return
        # Keep the established head and frequency. The body decoder checks
        # both carrier arrangements, its CRC, and neighbouring alignments.
        # A failed body leaves the normal post-key acquisition path available;
        # it must not launch a frequency/time search inside this reply slot.
        lo = max(0, head - origin - 8 * rxfront.SPS)
        hi = min(audio.size, head - origin + round(placement.PACKET_S * FS)
                 + 8 * rxfront.SPS)
        started = time.perf_counter()
        ev = rxfront._cs_event(self._corrected(audio[lo:hi]), CS_BREAKIN, 0,
                              head - origin - lo, head / FS, "")
        self._p3_stage_ms["changeover_body"] = (time.perf_counter() - started) * 1e3
        if ev.packet is None:
            return
        before = self._p3_delivered_at
        self._on(replace(ev, start=head), anchored=True)
        if self._p3_delivered_at == before:
            return
        self.frame_seen = True
        self._p3_changeover_pending = False
        self._p3_clock_role = self.host.arq.role
        self.p3_memory.clear()
        self._p3_prekey_crc = (head, getattr(self.host.peer, "slot", None))

    def control_bridge_until(self, until: int, final_until: int,
                             now: int, raster) -> int:
        """Do not let the first bridge consume a cold reply's decode budget."""
        early = self.control_collect_until(final_until, now, raster)
        if early < final_until:
            self._p3_control_collect_plan = (
                self._p3_acquisition_cycle, raster._p3_keyed_reply,
                final_until, early)
        return min(until, early) if early < final_until else until

    def control_collect_until(self, until: int, now: int, raster) -> int:
        """Allocate cold P3 acquisition time without weakening key admission.

        A settle-only close leaves eight milliseconds beyond DAC notice,
        less than acquisition plus the shared decode reserve. Collect a little
        less trailing audio while acquiring an ISS reply, never while reading
        IRS data or an established control clock. The bounded coherent reader
        still requires the complete control word; an earlier cutoff cannot
        authorize a partial word or an answer to an unkeyed slot.
        """
        a = self.host.arq
        keyed = getattr(raster, "_p3_keyed_reply", None)
        if (self.host.protocol != Protocol.PACTOR3 or a.state not in LINKED
                or a.role != ISS or not raster.sending
                or not (a.entry_pending or self._p3_answer_at is None)
                or keyed is None or keyed[0] != raster.keyed_slot):
            self._p3_control_collect_plan = None
            return until
        plan = self._p3_control_collect_plan
        if plan is not None:
            if plan[:3] == (self._p3_acquisition_cycle, keyed, until):
                # This deadline was selected before the FIRST bridge. Reaching
                # it does not authorize collecting onward to the old deadline.
                # Any work already spent is still charged by decode/key guards.
                return plan[3]
            self._p3_control_collect_plan = None
        _, started, emitted_end, cycle_n = keyed
        earlier = until - round(self.P3_ANSWER_ACQUIRE_RESERVE_S * FS)
        if (earlier <= now or earlier < emitted_end + round(.23 * FS)
                or earlier > started + cycle_n):
            return until
        candidate = self._p3_head_candidate
        if candidate is not None:
            periods = (emitted_end - candidate[1] + cycle_n - 1) // cycle_n
            head = candidate[1] + periods * cycle_n
            if head + P3_CS_N > earlier:
                return until
        return earlier

    def control_signal_in(self, seg: np.ndarray, seg_start: int, raster):
        """Read one actual keyed reply, with independent P1 and P3 admission.

        A released FSK tracker must not disable a learned P3 clock or cold
        entry acquisition. Conversely, a periodic clock cannot invent a reply
        to a skipped transmission. The extra P3 path is restricted to the
        captured portion of the last physically keyed P3 reply interval.
        An outstanding CS3 is different: the peer can keep acknowledging that
        same packet through our listening/skipped cycles. Read one later reply
        interval while that actual break-in remains unconfirmed.
        """
        if self.cs_seen or self.host.arq.state not in LINKED:
            return None, None
        end = seg_start + seg.size
        p1_at = raster.rx_due_in(seg_start, end)
        keyed = getattr(raster, "_p3_keyed_reply", None)
        if (self.host.protocol == Protocol.PACTOR3 and keyed is not None
                and keyed[0] == raster.keyed_slot):
            _, started, emitted_end, cycle_n = keyed
            periods = 0
            if raster.sending and self.host.arq.unconfirmed_breakin:
                # Select at most one captured interval, with enough trailing
                # audio for the existing cold acquisition. This projects an
                # answer to the SAME emitted CS3, not an ACK to an unkeyed new
                # packet. The original one-slot admission resumes when ARQ
                # settles that break-in. It also excludes our keyed waveform.
                periods = max(0, (end - emitted_end - round(.23 * FS)) // cycle_n)
            lo = max(seg_start, emitted_end + periods * cycle_n)
            hi = min(end, started + (periods + 1) * cycle_n)
            nominal = raster.rx_due(raster.keyed_slot) + periods * cycle_n
            at = None
            if hi - lo >= P3_CS_N:
                if self._p3_answer_at is not None:
                    # Project onto this emitted slot's reply interval, never
                    # the latest periodic instant anywhere in a recovered tape.
                    periods = (lo - self._p3_answer_at + cycle_n - 1) // cycle_n
                    projected = self._p3_answer_at + periods * cycle_n
                    # A saved phase remains a bounded receive hypothesis for
                    # this actually keyed packet, even after an outage. The
                    # age limit protects TX placement, not listening: applying
                    # it here strands the non-None anchor beyond eight misses
                    # while also excluding cold acquisition. Only a decoded
                    # answer refreshes the anchor; silence cannot move it.
                    if (periods >= 1
                            and lo <= projected
                            and projected + P3_CS_N <= hi):
                        at = projected
                if (at is None and (self.host.arq.entry_pending
                                    or self._p3_answer_at is None)
                        and hi - lo >= round(.23 * FS)):
                    # The cold sweep has its own 0.46-s bracket and deadline.
                    # The nominal P1 position is a search centre only here;
                    # it need not contain a complete P1 word, and may not be
                    # reused as an independently authorized P1 read.
                    at = max(lo, min(nominal, hi - P3_CS_N))
            if at is not None:
                heard = self.control_signal(seg[lo - seg_start:hi - seg_start],
                                            lo, at, p1_allowed=False)
                if heard is not None:
                    return heard, at
        if p1_at is not None:
            return (self.control_signal(seg, seg_start, p1_at,
                                        p3_allowed=False), p1_at)
        return None, None

    def control_signal(self, seg: np.ndarray, seg_start: int,
                       at: int, *, p1_allowed: bool = True,
                       p3_allowed: bool = True) -> Optional[int]:
        """The peer's control signal, read at the instant the grid predicts it.

        THE ONLY DECODE A ONE-SLOT KEYED CYCLE CAN AFFORD, and for two windows in
        that cycle the only one there is. A sending station's listening window is
        `cycle - packet - settle - holdback` = 237 ms on the 13 ms holdback a
        128-frame block buys, and 250 ms with the bridge's block collected -- it
        read 165 ms while the block was the device's 85 ms default. Either way it
        is below `RollingRx`'s half-second floor, below the flush's,
        and below `deep_scan`'s, all three of which are right to refuse -- a
        sweeping decoder given a quarter second is a false-alarm generator. So the
        ISS decoded NOTHING in a one-slot cycle, and the peer's acknowledgement,
        which is the whole reverse channel, could not reach the state machine.

        A station holding a link does not sweep. It knows a control signal is due
        and where, so it reads twelve bits there and spends the distance-8 margin
        on one trial -- which is what the reference implementation does, with no
        search at all.

        A FALLBACK, never a second voice: the rolling decoder and the flush have
        both already had this cycle's audio, and either one delivering a control
        signal takes this one out. One transmission may drive the FSM once --
        acknowledging the same packet twice advances the sequence past a packet
        the peer never received.

        Two positioners, in the order a station holding a link expects them, and
        the second is reached only when the first found nothing. `cs_anchored`
        places its window by requiring the two slots AFTER the word to be quiet,
        which is what a bare control signal ends with; a changeover packet has
        840 ms of data there instead, and `cs_head` mirrors that term onto the
        silence in front.

        THE TAIL TERM ORDERS THE ALIGNMENTS AND DOES NOT GATE THEM, which is not
        what this said until 2026-08-28. A changeover head reaches the anchored
        path whenever it is the only zero-error alignment in the bracket, and
        both the rendered packet and K4MSU's real one do, at every anchor across
        the bracket. So the anchored path cannot tell a head from a bare burst,
        and `_p1_cs` refuses a CS3 from it rather than reporting one.

        `cs_anchored` and not `decode_control_signal`, and the difference is the
        one WS8EOC's link stalled on: the latter spends `CS_SEARCH_S` looking for
        the burst, which is right when a detector pointed at it and wrong when the
        grid did. Over the 24 receive windows of 2026-07-30 it landed 20-45 ms off
        the word in 22 of them, read the answer in none, and read the CS3 that
        every CS4 casts two bits along in two. The anchored reader takes all 20
        of the answers those windows carry, against 58 accepts in 16 201
        twelve-bit reads of quiet audio from nine recordings -- 3.6 per 1000, the
        price of searching `p1rx.CS_SEARCH_HALF_S` rather than reading at a
        point.

        PACTOR-1 ALWAYS, THE UPGRADED PROTOCOL ONLY ONCE WE TRANSMIT IT, and the
        asymmetry is the protocol's rather than a caution. The reverse channel
        answers what it decoded, and the IRS follows the ISS -- so a peer cannot
        answer above PACTOR-1 before it has decoded something above PACTOR-1 from
        us, and while we are still keying PACTOR-1 such an answer is not a thing
        the far end can produce. Once we have upgraded, both are possible and both
        must be read: a peer that could not follow keeps answering in PACTOR-1,
        and that answer is the evidence the upgrade is reversed on
        (`ptc.PtcHost._follow_peer`).

        What running the PACTOR-3 reader in the PACTOR-1 phase would cost is
        MEASURED rather than argued, because the arithmetic overstates it badly.
        It searches 128 alignments at radius 1 of a six-word code, which on
        uniform random words would be worth a manufactured codeword every seventy
        calls; offered real waveforms it manufactures none -- 0 accepts across 200
        PACTOR-1 control signals in both shift senses, 200 PACTOR-1 data packets
        and 400 windows of white noise. So the reason it is not run there is that
        an answer above PACTOR-1 cannot arrive there, not that it is dangerous.

        PACTOR-2 HAS AN ENTRY IN THE TABLE BELOW as of 2026-09-02, and it is the
        last thing a PACTOR-2 link needed. Its data frame was already read
        (`_p2_packet`); its control signal is the same twenty-bit codeword
        PACTOR-3 uses on a different physical layer -- two DPSK carriers rather
        than tones 5 and 12 -- and `p2rx.control_signal_at` is that reader. A
        peer's bare acknowledgement in PACTOR-2 reaches the FSM now, which is
        what makes us its ISS rather than a station it can only talk at.

        AND IT IS THE HALF NOTHING OUTSIDE THIS PACKAGE HAS GRADED. The reader
        closes against `pactor2.control_signal`, which is a hypothesis for the
        reasons stated there, and a receive-only monitor cannot settle it because
        a monitor never answers. Where the codewords a real PACTOR-2 station
        keys actually sit is unmeasured -- the corpus holds none -- so this reads
        at `pactor2.cs_slot`'s instant and reads nothing anywhere else.
        """
        # LINKED, AND SO DISCONNECTING TOO. A PACTOR goodbye is an exchange: the
        # QRT bit rides a data packet and the peer acknowledges it, and
        # `arq.PactorArq._on_ack` reaching `_finish_disconnected` on STATUS_QRT is
        # the only path a link closes cleanly by. A receiver that shuts the moment
        # the goodbye goes out cannot hear one acknowledged at all, so every
        # session spends its teardown budget and reports the peer silent whatever
        # the peer sent.
        #
        # MEASURED, K4MSU on 3595 kHz, 2026-08-19 22:07. The cycle after the QRT
        # went out reported nothing decoded, and the window it wrote to disk
        # carries a CS2 at zero bit errors -- read alike by a free sweep over the
        # whole window and by this read at the anchor -- against a CS1 the cycle
        # before. In PACTOR-1 that change IS the acknowledgement, and the session
        # ended "the peer never acknowledged the goodbye", which was the most
        # common end-of-session line of that evening across three gateways.
        #
        # What a codeword does once admitted here is bounded: CS2 and CS4
        # acknowledge and close the link, CS1 asks for the goodbye again, and CS3
        # yields the role while preserving DISCONNECTING. A local QRT remains
        # owed; a received QRT retains the role until its final ACK emits.
        # `QRT_CYCLES` caps the whole teardown.
        if self.cs_seen or self.host.arq.state not in LINKED:
            return None
        readers = [self._p1_cs] if p1_allowed else []
        above = {Protocol.PACTOR2: self._p2_cs,
                 Protocol.PACTOR3: self._p3_cs}.get(self.host.protocol)
        if above is not None and (above != self._p3_cs or p3_allowed):
            readers.insert(0, above)
        for read in readers:
            ev = (read(seg, at - seg_start, seg_start=seg_start)
                  if read == self._p3_cs else read(seg, at - seg_start))
            if ev is not None:
                # ON THE SESSION'S CLOCK, which is the one every other line of
                # the log is on. A reader positions itself inside the window it
                # was handed and times its event there; `_summary` prints that
                # figure in the same column as `RollingRx`'s, which counts from
                # the session's first sample. See `tests.shrike.test_session_clock`.
                self._on(replace(ev, t=ev.t + seg_start / FS), anchored=True)
                return ev.cs
        return None

    @staticmethod
    def _p1_cs(seg: np.ndarray, at: int):
        """PACTOR-1's twelve-bit codeword at the grid's instant, `at` samples in.

        Two positioners, in the order a station holding a link expects them, and
        the second is reached only when the first found nothing. `cs_anchored`
        places its window by requiring the two slots AFTER the word to be quiet,
        which is what a bare control signal ends with; a changeover packet has
        840 ms of data there instead, and `cs_head` mirrors that term onto the
        silence in front.

        THE TAIL TERM ORDERS THE ALIGNMENTS AND DOES NOT GATE THEM, which is not
        what this said until 2026-08-28. A changeover head reaches the anchored
        path whenever it is the only zero-error alignment in the bracket, and
        both the rendered packet and K4MSU's real one do, at every anchor across
        the bracket. So the anchored path cannot tell a head from a bare burst
        and the refusal below is not a formality.

        SO A CS3 FROM THAT PATH IS AMBIGUOUS AND IS NEVER REPORTED AS ONE. Two
        different things put a bare-looking CS3 at the anchor -- a real head,
        which the tail term does not reject, and a CS4 read two bit periods late,
        which `p1rx.CS_ALIAS_BITS` names -- and the anchored read cannot say
        which, because it decides the word before it weighs what surrounds it.
        The break-in is the one codeword whose cost is the channel, so it is read
        again by the positioner aimed at a head and taken only from there.
        Refusing the class costs a real changeover nothing: `cs_head` reads it at
        every anchor across the bracket, rendered and off the air both. What it
        costs is the 22.35 s cycle of `captures/onair-0828-1838`, where two
        milliseconds of anchor motion turned a speed request into a handover and
        both ends spent sixteen cycles receiving.
        """
        t0 = at / FS
        got = p1rx.cs_anchored(seg, t0)
        name = "at anchor"
        if got is None or got[1] or got[0] == pactor1.CS_CHANGEOVER:
            # The changeover packet, and nothing else, gets the second trial: its
            # head is a CS3 in the control-signal slot with the new sender's first
            # packet behind it, the burst detector rejects it on length, and the
            # frame scan cannot be given a whole one in time. Restricting the
            # accept to the one codeword is what keeps a second read at an
            # ungated instant from doubling the false-accept rate.
            got = p1rx.cs_head(seg, t0)
            name = "break-in head"
            if got is None or got[1] or got[0] != pactor1.CS_CHANGEOVER:
                return None
        if got.unassigned:
            # NAMED, AND STILL NOT A CODEWORD. `control_signal` returns `ev.cs`,
            # which is None here, so the cycle counts as unanswered by every
            # caller that reads a control signal -- the word is not one. Which
            # word it is travels in `spare`, and one of the two has a measured
            # meaning: `ptc.PtcHost` takes the 0x59A grant off this event and
            # nothing takes 0x6A9. What it cost to throw the name away silently
            # is the DL6MAA capture's 4.407 s burst, logged as a six-error CS1
            # for as long as the table held four words.
            return rxfront.Event(t0, "unassigned",
                                 f"{spec.P1_CS_NAMES[got[0]]} at anchor "
                                 f"(0 bit errors, PACTOR-1, shift "
                                 f"{'inverted' if got.sense else 'normal'})",
                                 protocol="PACTOR-1", spare=got[0],
                                 sense=got.sense)
        return rxfront.Event(t0, "cs",
                             f"CS{got[0] + 1}/{name} (0 bit errors, PACTOR-1, "
                             f"shift {'inverted' if got.sense else 'normal'})",
                             protocol="PACTOR-1", cs=got[0], sense=got.sense)

    def _p3_cs(self, seg: np.ndarray, at: int, *, seg_start: int = 0):
        """Read the P3 answer at the grid anchor, acquiring one where there is none.

        Established answers prefer the learned peer clock, then the caller's
        anchor. Where the anchored read finds nothing and this link has never
        measured a P3 answer instant, a missed read searches all control words
        at the new answer phase and frequency; two distinct cycles corroborate
        a word without a CRC body.

        `_p3_answer_at is None` AND NOT `entry_pending`, which is what
        `captures/onair-0913-1837` cost. An IRS stint needs no peer codeword to
        read a greeting, so a link that came up, took the whole greeting and
        then reversed reaches ISS having measured no P3 answer at all -- and the
        gate as it stood switched the acquiring search off the moment entry was
        confirmed. That session's first ISS packet was aimed by the PACTOR-1
        turnaround of 22.36 s, `breakin_refusal` declined it, and 88 consecutive
        cycles printed CHANGEOVER NOT PLACED with 0 PACTOR-3 control signals
        decoded in 377 s. Entry keeps its own term: during entry the search runs
        against a measured instant too, because the peer's changeover can arrive
        at a phase the entry answer did not.

        Both halves want the last block of the window: the anchored reads
        because the codeword ends there, and the search because its bracket is
        centred on the same instant. Neither can be moved in front of the
        second `_collect`, which is why the ladder had to get cheaper instead.

        AND BOTH ARE ANCHORED, so a recovered window is transformed only where
        the anchor is (`P3_TRACKED_KEEP_S`). The whole window stands wherever the
        anchor is not inside that tail, which is what several slots given back
        produce and where a trimmed read would be lost outright.
        """
        keep = self._p3_tracked_keep_n()
        if keep is not None and seg.size > keep and at >= seg.size - keep:
            lo = seg.size - keep
            seg, seg_start, at = seg[lo:], seg_start + lo, at - lo
        corrected = self._corrected(seg)
        cfg = self.host.arq.cfg
        cycle_n = round((cfg.data_cycle_s if self.host.arq.cycle_long
                         else cfg.cycle_s) * FS)
        details = self._p3_acquisition_fits(self.CHANGEOVER_BODY_RESERVE_S)
        ev = None
        if self._p3_answer_at is not None:
            # Track the peer's actual answer clock. The P1 FSK tracker can
            # release its old reference while these P3 replies remain readable.
            cycles = round((seg_start + at - self._p3_answer_at) / cycle_n)
            projected = self._p3_answer_at + cycles * cycle_n - seg_start
            ev = (self._p3_read_again(seg_start, projected)
                  or self.sync.control_signal_tracked(corrected, projected,
                                                      details=details))
        if ev is None:
            ev = (self._p3_read_again(seg_start, at)
                  or self.sync.control_signal_at(corrected, at, details=details))
        if ev is not None:
            if not self._p3_bodyless_head_supported(seg, ev):
                return None
            if (self._p3_answer_at is not None
                    and seg_start + ev.start <= self._p3_answer_at + rxfront.SPS):
                return None
            if self._p3_answer_at is not None:
                elapsed = seg_start + ev.start - self._p3_answer_at
                gap = round(elapsed / cycle_n)
                if gap > ONSET_MAX_CYCLES:
                    displacement_ms = (elapsed - gap * cycle_n) / FS * 1000
                    print(f"    [p3] answer recovered after {gap} cycles: "
                          f"{spec.CS_NAMES[ev.cs]}, "
                          f"{displacement_ms:+.1f} ms from projected answer",
                          flush=True)
            self._p3_answer_at = seg_start + ev.start
        if ev is not None or not (self.host.arq.entry_pending
                                  or self._p3_answer_at is None):
            return ev
        # The first P3 answer can precede the old P1 anchor by ~100 ms.
        # WS8EOC 2026-09-08 also needed receive-frequency acquisition: its
        # CS3 heads and CRC-valid RMS greeting were present while this read
        # returned nothing. VE3KPG/KB5LZK on 2026-09-10 instead answered with
        # bare CS1. Two physical cycles corroborate an acquired word; a complete
        # validated body suffices on its own. Only CS3 changes the link role.
        # A skipped key can leave several cycles in this buffer. Acquisition
        # concerns the current reply and must still fit the pre-key deadline --
        # ASKED, not assumed, which is the rule its two siblings already keep.
        if not self._p3_acquisition_fits(self.P3_ANSWER_ACQUIRE_RESERVE_S):
            return None
        return self._acquire_answer(seg, seg_start, at)

    def tracked_answer_position_ms(self, boundary: int,
                                   slot_n: int) -> Optional[float]:
        """Where the peer answered, folded onto its slot -- the CODEWORD'S instant.

        `_MasterGrid` asks the same question of an envelope, and the two do not
        agree. On KB5LZK's 40 m arm of 2026-09-15 the onset reading moved 140 ms
        over five cycles -- 1015.3, 1044.0, 938.1, 1033.0, 904.0 -- while the
        matched filter put every clean answer of that arm within 0.6 ms of a
        straight line through them. An envelope is looking for where the channel
        got louder; a codeword filter is looking for the word. Only the second
        is the answer's position, and this receiver has already measured it to
        the sample by the time anything asks.

        None unless a PACTOR-III control signal was tracked INSIDE THE SLOT the
        caller is asking about. A folded instant would let the previous cycle's
        answer stand in for this one's, which is the quiet cycle reporting a
        position it never measured -- and an instrument reports a measurement or
        it reports nothing.
        """
        at = self._p3_answer_at
        return (None if at is None or not boundary <= at < boundary + slot_n
                else (at - boundary) / FS * 1e3)

    def _p3_bodyless_head_supported(self, audio: np.ndarray, ev) -> bool:
        """A word inside IRS data is not automatically a new peer head.

        The 1731 capture's ordinary SL3 field produced a zero-error CS3
        alignment but failed the coherent acquisition/extent test. Promoting
        it replaced packet geometry and suppressed three replies. Keep CRC
        packets and ISS answers unchanged; require the existing stronger
        head detector for a body-less CS3 while already receiving P3 data.
        """
        a = self.host.arq
        if (self.host.protocol != Protocol.PACTOR3 or a.role != IRS
                or a.state not in LINKED or a.entry_pending
                or ev.cs != CS_BREAKIN or ev.packet is not None):
            return True
        if not self._p3_acquisition_fits(self.P3_ANSWER_ACQUIRE_RESERVE_S):
            return False
        lo = max(0, ev.start - round(.08 * FS))
        hi = min(audio.size, ev.start + round(.31 * FS))
        got = p3acquire.changeover(
            audio[lo:hi], offsets=(self.p3_receive_offset_hz,))
        return (got is not None
                and abs(lo + got.event.start - ev.start) <= rxfront.SPS)

    def _corrected(self, audio: np.ndarray) -> np.ndarray:
        """This cycle's window on the peer's carrier, transformed once.

        `_read_p3_packet` and `_p3_cs` are handed the same buffer at the same
        offset in the same cycle and each ran the transform: 0.89 ms of a 0.94 s
        window and 1.99 of a 2.19 s one, twice, out of the 1.8 ms the
        `onair-0914-0850` geometry leaves between the frame-end close and
        `_regrid`'s admission check.

        KEYED ON THE BUFFER ITSELF, not on the caller's origin, because the two
        readers name the same samples differently -- `deep_scan` passes a
        trimmed view and its absolute origin, the anchored read passes the whole
        window and the segment's. The address and length identify the samples
        under both; holding the source beside the result keeps that address from
        being reused under the key while the entry stands.
        """
        hz = self.p3_receive_offset_hz
        key = audio.ctypes.data, audio.size, hz
        held = self._p3_corrected
        if held is None or held[0] != key:
            held = key, audio, p3acquire.compensate(audio, hz)
            self._p3_corrected = held
        return held[2]

    def _p3_read_again(self, seg_start: int, at: int):
        """This cycle's codeword, where a reader has already decoded it there.

        `deep_scan`'s tracked-changeover branch and this anchored read are aimed
        at the same instant, in the same cycle, at the same audio -- and when the
        word is CS3 each of them demodulates the 0.84 s frame behind it on both
        carrier arrangements. MEASURED on `captures/onair-0913-2152`:
        `rxfront._cs_event` ran 111 times over 93 cycles, and that decode is
        7.6 ms median of a pre-key budget that is 7.6 ms whole. The second call
        cannot learn anything the first did not, so it is answered out of what
        the first one returned rather than run again.
        """
        if self._p3_word_read is None:
            return None
        aim, found, ev = self._p3_word_read
        start = found - seg_start
        if abs(aim - seg_start - at) > rxfront.SPS or start < 0:
            return None
        return replace(ev, start=start, t=start / FS)

    def _acquire_answer(self, seg: np.ndarray, seg_start: int, at: int):
        """`_p3_cs`'s cold half: the answer's phase and frequency, searched.

        ITS OWN ENTRY CONDITION, AND `cs_seen` IS NOT IN IT. `control_signal`
        tries this reader first and the PACTOR-1 one only behind it, so on a
        PACTOR-3 link a successful acquisition is what the cycle answers with
        and the anchored PACTOR-1 read never runs. Asking `cs_seen` again here
        would let a reader that has not run yet cancel one that has -- and on
        an upgraded link with a live PACTOR-1 onset clock, the anchored
        twelve-bit read is the one whose false-accept rate is priced in its own
        docstring.
        """
        if self.host.arq.state not in LINKED:
            return None
        cfg = self.host.arq.cfg
        cycle_n = round((cfg.data_cycle_s if self.host.arq.cycle_long
                         else cfg.cycle_s) * FS)
        lo = max(0, at - round(.22 * FS))
        hi = min(len(seg), at + round(.24 * FS))
        previous = self._p3_head_candidate
        preferred = (self.p3_receive_offset_hz if self._p3_offset_fixed else
                     previous[2] if previous is not None
                     and 1 <= self._p3_acquisition_cycle - previous[0] <= 2
                     else None)
        got = p3acquire.control_signal(
            seg[lo:hi], offsets=p3acquire.ENTRY_CONTROL_OFFSETS_HZ,
            preferred_hz=preferred)
        if got is None:
            return None
        got.event = replace(got.event, start=got.event.start + lo,
                            t=got.event.t + lo / FS)
        if got.event.cs == placement.BREAKIN_CS and got.event.packet is None:
            # The short search window can omit an available changeover body.
            # Decode that one acquired position from the full segment so a
            # valid CRC can deliver its payload immediately, without waiting
            # for a second head or widening the frequency search.
            got.event = rxfront._cs_event(
                p3acquire.compensate(seg, got.offset_hz), got.event.cs, 0,
                got.event.start, got.event.t,
                f", acquisition RX {got.offset_hz:+g} Hz")
        absolute = seg_start + got.event.start
        now = self._p3_acquisition_cycle
        previous = self._p3_head_candidate
        self._p3_head_candidate = (now, absolute, got.offset_hz)
        elapsed = 0 if previous is None else absolute - previous[1]
        periods = round(elapsed / cycle_n)
        if got.event.packet is None and (previous is None
                or not 1 <= now-previous[0] <= 2
                or periods < 1
                or abs(elapsed - periods * cycle_n) > round(.020*FS)
                or abs(got.offset_hz-previous[2]) > 25):
            if (got.event.cs == placement.BREAKIN_CS
                    and (self._p3_entry_body_checked_at is None
                         or absolute > self._p3_entry_body_checked_at + rxfront.SPS)):
                # A coherent first head is not yet a role change, but its body
                # can extend past the next entry key. Finish that one bounded
                # receive window before deciding to transmit again.
                self._p3_entry_body_candidate = (absolute, got.offset_hz)
            print(f"    [p3] candidate {spec.CS_NAMES[got.event.cs]} "
                  f"at {absolute / FS:.3f} s, RX {got.offset_hz:+g} Hz; "
                  "awaiting a distinct corroborating cycle", flush=True)
            return None
        self._follow_p3_offset(got.offset_hz)
        self._p3_answer_at = absolute
        self._p3_changeover_pending = (got.event.cs == placement.BREAKIN_CS
                                       and got.event.packet is None)
        self.p3_memory.clear()
        self._p3_head_candidate = None
        self._p3_entry_body_candidate = None
        return got.event

    def _p2_cs(self, seg: np.ndarray, at: int):
        """PACTOR-2's twenty bits, at the instant the grid predicts them.

        `_p3_cs`'s counterpart and the same read: twenty DBPSK bits at one point,
        no search, `p2rx.CS_MAX_ERRORS` of slack -- which is zero. What it takes
        that PACTOR-3's does not is the two things a PACTOR-2 cycle carries and
        PACTOR-3 has no equivalent of.

        THE ARRANGEMENT, because a PACTOR-2 codeword rides the same two virtual
        carriers the data field does and they exchange tones every ARQ cycle. Our
        own grid counts that parity for the shift and the carrier swap alike
        (`_MasterGrid.shift`), so the arrangement the peer's answer is in is the
        arrangement our own packet went out in -- and a reader pinned to the home
        one is deaf on every other cycle, which is what put a real recording's
        markers on a 2.5 s spacing (`p2rx.find_markers`).

        THE CARRIER PAIR, because PACTOR-2 has no fixed tone plan the way
        PACTOR-3's channels 5 and 12 are fixed: the frame is 400 Hz wide wherever
        the pair sits, and the acquisition correlator reports which of 24 bins it
        armed on. `p2_bin_pair` is the last one a frame decoded at, so the
        codeword is read where the peer's data was rather than where a 1500 Hz
        centre would put it.
        """
        swapped = bool(getattr(self.host.peer, "invert", False))
        got = p2rx.control_signal_at(seg, at, bin_pair=self.p2_bin_pair,
                                     swapped=swapped)
        if got is None:
            return None
        cs, errors = got
        return rxfront.Event(
            at / FS, "cs",
            f"CS{cs + 1}/{spec.CS_NAMES[cs]} ({errors} bit errors, PACTOR-2, "
            f"bin pair {self.p2_bin_pair}, "
            f"{'swapped' if swapped else 'home'} arrangement)",
            protocol=Protocol.PACTOR2, cs=cs)

    def flush(self) -> None:
        """Decode the last of this cycle NOW -- and only the last of it.

        BEHIND OUR OWN CARRIER, in the 0.96 s where the cost is free, and not in
        front of the key where it used to be: the cycle leaves 43 ms between the
        window closing and the PTT instant, the frame scan has first claim on
        those, and 26.7 ms measured does not fit in what is left. What that costs
        is stated where the reserve is; what it buys is a key on time.

        `RollingRx.flush` decodes its whole buffer, which by
        the end of a cycle is the whole cycle, and `rxfront.decode_events` is
        superlinear in the length: 8 ms over 0.5 s, 12 over 0.75, 18 over 1.0,
        37 over the 1.33 s a 1.25 s cycle plus its tail actually holds. That 37
        was measured as the near-constant 39-45 ms by which the log's "RF starts"
        trailed its own "off-grid" reading, every cycle, and `wait_until` cannot
        give back a deadline that has already gone.

        The whole buffer was never needed. The listen loop has already decoded
        everything up to its last slide with a full window of history behind it;
        what is left for the flush is the block held back for the bridge. Three
        quarters of a second covers that with six times a control signal's length
        of context, and stays above the half second `RollingRx` refuses to decode
        below.

        AND IT IS NOT WHERE A CODEWORD IS FOUND. What the trim gives up is a
        SWEEP over audio the listen loop held, and a sweep is the dearest
        instrument in the receiver for the cheapest thing in the protocol: 24 ms
        over 0.75 s, 110 over 2.09, 434 over 3.89, against 0.3 ms for one
        anchored read of a 120 ms codeword. Every one of those milliseconds is
        also a sample the capture may not survive: measured 2026-08-26, 23 of 23
        capture-loss intervals sat inside a `RollingRx` decode holding the
        interpreter. So the codewords a held window carries are read where the
        burst detector already found them (`_read_codeword_at_bursts`), and this
        stays the short decode a keyed cycle can afford.

        NOR IS IT WHERE A DATA FRAME IS FOUND, and the trim is not what stops it.
        A short-cycle PACTOR-3 packet is 0.89-0.90 s of air at every speed level,
        so 0.75 s of context delivers none of them -- measured, 0 packets at
        levels 2, 3 and 5 against 1 apiece at 1.05 s. Widening it is still the
        wrong trade four times over. `deep_scan` has already read that same audio
        this cycle, header-anchored, and `new_cycle`'s one-frame rule would
        refuse a second delivery of it anyway. The sweep costs 117-124 ms over
        1.05 s against `deep_scan`'s 74-84, and 46 ms rather than 30 on the
        cycles that hold nothing -- paid every keyed cycle, inside the decode
        the capture-loss measurement above indicts. It cannot reach the long
        cycle at any affordable length: a 3.37 s packet needs 3.6 s of context
        and 499 ms to sweep it, so `deep_scan` would remain the only reader
        there whatever this constant said. And the rolling decoder's windows
        overlap, so it is the one path the session's `p3_memory` must not be
        handed -- four slides of one failed frame would push the peer's real
        repeats out of the sum.
        """
        keep = int(FLUSH_CONTEXT_S * self.rx.fs)
        if len(self.rx.buf) > keep:
            drop = len(self.rx.buf) - keep
            self.rx.buf = self.rx.buf[drop:]
            self.rx.t0 += drop / self.rx.fs      # ...so events keep their time
        had = len(self.rx.buf)
        self.rx.flush()
        # `RollingRx.flush` empties the buffer without advancing past what it
        # decoded, which no other caller notices because it decodes once, at the
        # end of a stream. Here the stream goes on.
        self.rx.t0 += (had - len(self.rx.buf)) / self.rx.fs
        # And only now is our own carrier stepped over -- the whole point of
        # holding the debt (see `skip`).
        self._settle()

    def deep_scan(self, audio: np.ndarray) -> None:
        """Once per cycle, look for a frame the way that actually finds one.

        The rolling decoder's pilot gate cannot be trusted for data frames -- a
        packet decodes on a clean channel and at no signal-to-noise ratio below it
        -- and cannot be repaired cheaply, because the pilot peak's offset from the
        true row 0 scatters once noise is present. The CRC scan does find them, and
        it is unforgeable, so it needs no gate ahead of it. It runs here, once per
        cycle on the cycle's own audio, rather than four times a second.

        Once ONE cycle has produced a frame, the rest are tracked rather than
        scanned. PACTOR is cycle-synchronous, so the next row 0 lands a cycle later
        within the grid's jitter, and demodulating at that alignment costs a
        fortieth of the scan on the station Pi. A miss falls straight back to the
        scan, so a lost cycle costs a retransmit and not the lock.

        THE LINK'S OWN PROTOCOL, AND BEFORE THE KEY, because this is the decode
        the cycle's answer hangs on: an IRS that has not read the peer's packet by
        its own boundary has nothing to acknowledge, and a repeat request goes out
        instead of an acknowledgement. It runs on the audio up to the peer's
        turnaround and is paid for out of `PREKEY_RESERVE_S`. MEASURED over a
        1.30 s window: 1.8 ms when the packet is there, 20.3 ms when the channel
        is empty.

        WHERE "up to the peer's turnaround" IS depends on which station we are,
        and the hold loop calls this at a different point in the cycle for each.
        Sending, the peer's codeword ends well before `boundary - d` and the
        mid-cycle buffer holds it whole. Receiving, the involution keeps each
        side's own gap across a changeover, so the peer's PACKET ends about `d`
        before our boundary -- the very sample the mid-cycle buffer stops at --
        and a scan run there is handed a frame missing its last bits, which
        `tones.fits` rejects without a word. Measured on onair-0803-225309:
        WS8EOC's changeover packet arrived bit-perfect (96/96 against the
        rendered frame) in two consecutive receive windows, decodes CRC-valid
        from the saved captures, and reached the live state machine in neither.
        The session asked for a repeat, was given one, could not read that
        either, and asked again until both stations gave up. So the receiving
        station scans after the final bridge, when the tail is in hand -- and
        ONLY WHILE THE SCAN IS FINDING FRAMES, because that spot has 8 ms past
        `key_notice` in it: room for the 1.8 ms a frame that is there costs,
        none for the 20.3 ms of a sweep over a channel that is not decoding.
        Against a gateway whose packets never resolved, the unconditional
        pre-key sweep put every IRS cycle of 2026-08-13 at its key 11-15 ms
        late and lost four slots in 23 cycles to `SLOT GONE` -- each one a
        missing acknowledgement. A cycle that comes up empty therefore moves
        the next cycle's sweep to the top of the cycle, in front of a whole
        slot of listening, where a frame it does find still keys on that
        cycle's own boundary -- one cycle late, the flush's bargain -- and
        arms the pre-key scan again.

        EVERY OTHER PROTOCOL WE CAN READ IS `upgrade_scan`'s, and it runs in the
        transmit slot rather than not at all.

        THE WINDOW HAS TO HOLD THE WHOLE PACKET, which on the long cycle is
        3.37 s rather than 0.90. Nothing here can widen it -- a frame missing
        its last rows is one `tones.fits` rejects without a word -- so the
        cycle length reaches the readers through the grid, which takes it once
        a cycle (`_MasterGrid.regear`) and sizes every window in front of the
        key from it.

        Unlike the control-signal pair, neither reader can be talked into an
        accept -- a CRC-16 plus `p3rx.confirmed` on one side, and a CRC plus the
        header and eye gates on the other -- so neither needs a gate in front of
        it, and where they run is a question of the clock alone.
        """
        self._scan(audio, self._readers()[:1])

    def upgrade_scan(self, audio: np.ndarray) -> None:
        """The same scan, for the protocol the peer might be leading into.

        A data packet is where an upgrade actually arrives. The ISS decides to
        stop transmitting PACTOR-1 and the IRS learns of it from the first packet
        in the new protocol -- so a receiver pinned to what IT is transmitting
        cannot follow a peer that leads. That is this scan's own history read from
        the other side: pinned to PACTOR-3, a station that had just transmitted a
        correct PACTOR-1 data phase went looking for the peer's answer with the
        wrong demodulator, on the wrong tones, every cycle, and there was no path
        by which a peer's data packet could reach the FSM at all. Doing the same
        to a peer that starts sending PACTOR-2 would be that fault in a third
        direction.

        SO THIS IS A QUESTION OF WHEN, NOT WHETHER. The blind PACTOR-3 scan
        measures 48.0 ms against the PACTOR-1 scan's 20.3 on the same window, and
        the whole of what a keyed cycle has in front of its PTT is the holdback
        plus `PREKEY_RESERVE_S` -- 43 ms, which the reserve cannot be widened to
        change without taking the peer's own codeword out of the fed window. Both
        readers do not fit, and a cycle that overruns hands its slot back to the
        grid: 2.5 s between packets, which stalls a link on its own. Our own
        carrier is 0.96 s of a 1.25 s cycle with nothing else to do behind it, so
        the speculative reader runs THERE, and an upgrade is followed a cycle
        later than the first packet carrying it. The peer repeats a packet nobody
        acknowledged; that is what ARQ is for.

        The reader list is what shrike can DEMODULATE, which is not the same set
        as what it can transmit and is the reason it is a list rather than a pair.
        PACTOR-2 is now in it, and is the case that proves the distinction:
        `ptc.TRANSMITTABLE` does not hold it and will not until the frame marker is
        built, so a peer leading into it is heard, acknowledged and answered in
        PACTOR-1 -- `ptc.PtcHost._follow_peer`'s last branch, which nothing had
        ever reached. Hearing a protocol we cannot key is worth the slot it costs:
        the alternative is the `onair-0803-214416-over` session, where the ARQ data
        phase ran, the changeover handed the channel to the gateway, and what came
        back had no reader at all.

        WHAT THE SECOND READER ADDS TO THE SLOT, on the same 1.30 s window as the
        48.0 ms above: 14.5 ms with nothing there, which is the marker correlation
        and nothing behind it; 40 ms median on the cycles of a real PACTOR-2 link;
        344 ms once in the 32 cycles of `hb9ak_055246_c1500.wav`, where a marker
        armed and no alignment of the 256 the rotation search offers produced a
        CRC. Against 0.96 s of our own carrier the worst of those still leaves
        half the slot -- and the floor is what every cycle actually pays.
        """
        self._scan(audio, self._readers()[1:])

    def _readers(self) -> list:
        """Every protocol we can demodulate a frame in, the link's own first.

        Keyed by protocol rather than ordered by hand, because the rule is about
        the link and not about the ladder: the head of the list is whatever we are
        transmitting -- it runs before the key, on the audio this cycle's answer
        hangs on -- and the rest run behind our own carrier in whatever order they
        are declared in. Which of them is even a candidate is not a question this
        station gets to answer; see `upgrade_scan`.
        """
        readers = {Protocol.PACTOR3: self._p3_packet,
                   Protocol.PACTOR2: self._p2_packet,
                   Protocol.PACTOR1: self._p1_packet}
        return [readers.pop(self.host.protocol), *readers.values()]

    def _scan(self, audio: np.ndarray, readers: list) -> None:
        # CONNECTING TOO, because a frame is one of the two ways a call is
        # answered and the only unforgeable one. `arq.on_rx_packet` has carried a
        # CONNECTING branch for it -- a Winlink RMS that answers by sending its
        # greeting rather than a codeword -- and no audio could ever reach it: the
        # CRC scan is the only path a data frame arrives by at real SNR, and this
        # guard closed it for exactly the phase the branch exists for.
        #
        # AND DISCONNECTING, for `control_signal`'s reason and one of its own: a
        # peer that answers our goodbye by breaking in sends a CS3-headed packet,
        # and `arq.on_rx_packet` yields to it and delivers its field in the same
        # call. Refused here, the last thing a gateway says is dropped -- and the
        # yield it is owed is the only thing that can tell this station the peer
        # is not finished.
        # A deferred P3 answer can still be replaced by a newer physical frame
        # in this hold iteration. Absolute-position deduplication below keeps
        # overlapping tracked reads from counting the same transmission twice.
        newer_p3 = (self._tracked_only and self._scan_origin is not None
                    and self.host.protocol == Protocol.PACTOR3)
        if (audio.size < FS // 2 or (self.frame_seen and not newer_p3)
                or self.host.arq.state not in (State.CONNECTING, *LINKED)):
            return
        for read in readers:
            got = read(audio)
            if got is not None:
                ev, how = got
                if ev.protocol == Protocol.PACTOR3 and self._scan_origin is not None:
                    absolute = self._scan_origin + ev.start
                    if not self._note_p3_frame(ev, absolute):
                        return
                    ev = replace(ev, t=ev.t + self._scan_origin / FS)
                    print(f"    [p3] frame @ {absolute} ({how}), "
                          f"{'long' if ev.cycle_long else 'short'}, "
                          f"primary RX {self.p3_receive_offset_hz:+g} Hz", flush=True)
                self.count += 1
                self.frame_seen = True
                print(f"    {self.tag} ({how}) {ev.text}", flush=True)
                traffic.received(self.host, ev)
                self.host.on_rx_event(ev)
                if ev.protocol == Protocol.PACTOR3 and self._scan_origin is not None:
                    self._p3_clock_role = self.host.arq.role
                return

    def _seed_p3_changeover_clock(self, phase: int) -> None:
        """Move phase to the changeover; retain the measured cycle and span."""
        self._p3_row0 = phase + p3frame.DATA_OFFSET * rxfront.SPS
        self._p3_delivered_at = phase
        # Until an ordinary header arrives the peer may repeat this CS3-headed
        # frame. It has no variable header for SyncedRx.packet to track.
        self._p3_repeat_changeover = True

    def _p1_packet(self, audio: np.ndarray):
        # A recovered slot may contain several cycles. The expected-packet
        # decoder returns the first valid frame, but the acknowledgement we
        # are about to key belongs to the latest cycle. Reading an older frame
        # can request the wrong counter or replay a superseded changeover.
        # PACTOR-1 always has a 1.25-second cycle, at either baud rate.
        offset = max(0, audio.size - round(spec.CYCLE_SHORT_S * FS))
        ev = rxfront.decode_expected_p1_packet(audio[offset:], self.p1_memory)
        if ev is not None and ev.start is not None:
            ev = replace(ev, start=ev.start + offset)
        return None if ev is None else (ev, "P1 scanned")

    def _p3_packet(self, audio: np.ndarray):
        # A requested SL2 must be decoded before its own ACK. preferred_only
        # intentionally stays on the last validated level, and the existing
        # wideband trial is SL3-only; neither can make this transition in time.
        if (self._tracked_only and self._scan_origin is not None
                and self._p3_row0 is not None and self.host.arq.role == IRS
                and self._p3_cycle_n == round(spec.CYCLE_SHORT_S * FS)
                and not self.host.arq.cycle_command_emitted):
            origin = self._scan_origin
            span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[2]))
            latest = origin + audio.size - span
            target = self._p3_row0 + ((latest - self._p3_row0) // self._p3_cycle_n) * self._p3_cycle_n
            target = getattr(self, "_p3_target_row0", None) or target
            fresh = self._p3_delivered_at is None or target > self._p3_delivered_at + rxfront.SPS
            if (fresh and self.sl2_prekey_expected(target)
                    and self._p3_acquisition_fits(self.P3_SL2_RESERVE_S)):
                lo = max(0, target - origin - p3frame.DATA_OFFSET * rxfront.SPS - 2400)
                ev = self.sync.sl2_packet_at(self._corrected(audio[lo:]), target - origin - lo)
                if ev is not None:
                    self._p3_changeover_pending = self._p3_repeat_changeover = False
                    self._p3_sl2_expected_until = None
                    self.p3_memory.clear()
                    return replace(ev, start=ev.start + lo, t=ev.t + lo / FS), "SL2 pre-key"
        # Read the current SL1 packet before spending time on a repeated CS3
        # or speculative wideband body. The stock peer can advance immediately
        # after our ACK; keeping the previous CHANGEOVER's counter for one more
        # cycle guarantees a wrong reply on every such advance.
        if (self._tracked_only and self._scan_origin is not None
                and self._p3_row0 is not None and self.sync.packet_level in (None, 1, 2)
                and self._p3_cycle_n == round(spec.CYCLE_SHORT_S * FS)
                and not self.host.arq.cycle_command_emitted
                and self._p3_acquisition_fits(self.P3_SL1_RESERVE_S)):
            origin = self._scan_origin
            span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[1]))
            latest = origin + audio.size - span
            target = self._p3_row0 + ((latest - self._p3_row0) // self._p3_cycle_n) * self._p3_cycle_n
            target = getattr(self, "_p3_target_row0", None) or target
            lo = max(0, target - origin - p3frame.DATA_OFFSET * rxfront.SPS - 2400)
            # Share the retained-window correction with the following tracked
            # codeword reader if this CRC misses. The sparse decoder itself
            # still samples only this packet's header and body.
            lo = min(lo, max(0, audio.size - (self._p3_tracked_keep_n() or audio.size)))
            fresh = self._p3_delivered_at is None or target > self._p3_delivered_at + rxfront.SPS
            if fresh and target - origin >= p3frame.DATA_OFFSET * rxfront.SPS + rxfront.SPS:
                ev = self.sync.sl1_packet_at(self._corrected(audio[lo:]), target - origin - lo)
                if ev is not None:
                    self._p3_changeover_pending = self._p3_repeat_changeover = False
                    self.p3_memory.clear()
                    return replace(ev, start=ev.start + lo, t=ev.t + lo / FS), "SL1 pre-key"
        # A repeated changeover has a CS3 head, not a variable header, and
        # its shortened case-0 body is not an ordinary SL1 frame. Project the
        # head on its retained clock before the wideband-only branch can
        # decline an unknown packet_level. The body decoder still requires
        # its own CRC and neighbouring-alignment confirmation; a head alone
        # never delivers data or changes the reply clock here.
        #
        # Project on available HEAD support, not the ordinary packet's tail.
        # The latter projected an entire cycle backwards on hold_05..07 of
        # onair-0914-1731 despite all three windows containing readable RMS.
        if (self._tracked_only and self._scan_origin is not None
                and self._p3_repeat_changeover and self._p3_row0 is not None
                and self._p3_acquisition_fits(self.CHANGEOVER_BODY_RESERVE_S)):
            base = self._p3_row0 - p3frame.DATA_OFFSET * rxfront.SPS
            end = self._scan_origin + audio.size
            phase = base + ((end - base - P3_CS_N) // self._p3_cycle_n) * self._p3_cycle_n
            at = phase - self._scan_origin
            if at >= 0:
                lo = max(0, at - 8 * rxfront.SPS)
                hi = min(audio.size, at + round(placement.PACKET_S * FS)
                         + 8 * rxfront.SPS)
                ev = self.sync.control_signal_at(
                    self._corrected(audio[lo:hi]), at - lo, details=True)
                if ev is not None and ev.breakin and ev.packet is not None:
                    self.p3_memory.clear()
                    return (replace(ev, start=ev.start + lo, t=ev.t + lo / FS),
                            "changeover pre-key")
        # A recovered window can span many cycles. Only the latest complete
        # cycle can inform the next reply; bound the blind search accordingly.
        # Include a preceding cycle too: a window ending partway through the
        # next long frame must still contain the last complete frame's head.
        blind = round((2 * spec.CYCLE_LONG_S + .15) * FS)
        keep = (self._p3_tracked_keep_n() or blind) if self._tracked_only else blind
        trim = max(0, len(audio) - keep)
        if (self._tracked_only and self._scan_origin is not None
                and self._p3_row0 is not None):
            # A trailing 1.4 s is not necessarily a complete short packet:
            # the window may already contain most of the NEXT cycle. Preserve
            # the latest complete frame's header before bounding the reader,
            # rather than projecting a head that the crop has just discarded.
            # This retains at most one cycle plus its frame and header lead;
            # it does not open a blind search over the whole held window.
            span, cycle_n = _p3_receive_geometry(self)
            latest = (self._scan_origin + len(audio)
                      - rxfront._packet_span(span))
            target = self._p3_row0 + ((latest - self._p3_row0) // cycle_n) * cycle_n
            head = (target - self._scan_origin
                    - p3frame.DATA_OFFSET * rxfront.SPS - 2400)
            trim = min(trim, max(0, head))
        audio = audio[trim:]
        if (self._tracked_only and self._scan_origin is not None
                and self.wideband_prekey_active()):
            origin = self._scan_origin + trim
            target = _p3_wideband_target(self, origin + len(audio))
            if target is not None and target >= origin + p3frame.DATA_OFFSET * rxfront.SPS:
                if not self._p3_acquisition_fits(self.P3_WIDEBAND_RESERVE_S):
                    return None
                lo = max(0, target - origin - p3frame.DATA_OFFSET * rxfront.SPS - 2400)
                corrected = self._corrected(audio[lo:])
                ev, owned = self.sync.wideband_packet_at(
                    corrected, target - origin - lo,
                    allow_short_fallback=True,
                    can_decode=lambda: self._p3_acquisition_fits(
                        self.P3_WIDEBAND_RESERVE_S))
                if ev is not None:
                    self._p3_changeover_pending = self._p3_repeat_changeover = False
                    return (replace(ev, start=ev.start + trim + lo,
                                    t=ev.t + (trim + lo) / FS), "wideband pre-key")
                if owned or (self.sync.packet_level or 0) >= 3:
                    return None
                # The narrow path is still owed its own processing reserve.
                if not self._p3_acquisition_fits(self.P3_WIDEBAND_RESERVE_S):
                    return None
            # This guard belongs only to the experimental short-cycle path.
            # Existing CS6 acquisition depends on the no-preference ladder.
            if self.sync.packet_level is None:
                return None
        if self._scan_origin is not None and self._p3_row0 is not None:
            origin = self._scan_origin + trim
            span, cycle_n = _p3_receive_geometry(self)
            # AGAINST THE PACKET'S OWN END, not the decoder's trailing margin.
            # Two of `_frame_span`'s rows are `rx.decode_frame`'s margin and
            # carry no signal -- `_sampled_baseband` reads past the buffer as
            # silence -- so what has to be inside the window is the packet, and
            # it was: a median 16.8 ms inside on all 182 IRS cycles of
            # `captures/onair-0913-2320`, where the margin overran on 125 of them
            # by a median 152 samples. Floored against the margin the index
            # dropped a whole cycle, `at` went negative and no read ran at all.
            # STILL A FLOOR: a window that runs on past the frame -- a recovered
            # slot, a long listen -- must not aim at a cycle the peer has not
            # keyed yet. `SyncedRx._candidates` applies the same end.
            latest = origin + len(audio) - rxfront._packet_span(span)
            cycles = (latest - self._p3_row0) // cycle_n
            at = self._p3_row0 + cycles * cycle_n - origin
            self.sync.packet_at = at if at >= p3frame.DATA_OFFSET * rxfront.SPS else None
        target = getattr(self, "_p3_target_row0", None)
        if target is not None and self._scan_origin is not None:
            self.sync.packet_at = target - self._scan_origin - trim
        got = self._read_p3_packet(
            audio,
            None if self._scan_origin is None else self._scan_origin + trim)
        if got is None:
            return None
        ev, how = got
        return replace(ev, start=ev.start + trim, t=ev.t + trim / FS), how

    def _read_p3_packet(self, audio: np.ndarray, origin: int | None = None):
        # A granted entry answer and a body behind an acquired CS3 have no
        # variable header. Searching for normal data first cost 85 ms on the
        # recorded RMS reply, exceeding the changeover's pre-key reserve.
        changeover_first = (self.host.arq.entry_pending
                            or self._p3_changeover_pending
                            or self._p3_repeat_changeover)
        corrected = self._corrected(audio)
        if self._p3_repeat_changeover and self.sync.packet_at is not None:
            at = self.sync.packet_at - p3frame.DATA_OFFSET * rxfront.SPS
            ev = self.sync.control_signal_at(
                corrected, at,
                details=self._p3_acquisition_fits(self.CHANGEOVER_BODY_RESERVE_S))
            if ev is not None and origin is not None:
                self._p3_word_read = origin + at, origin + ev.start, ev
            if ev is not None and ev.breakin and ev.packet is not None:
                self.p3_memory.clear()
                # AT THE INSTANT IT WAS AIMED AT, because this read confirms a
                # head and cannot locate one. A twenty-bit word at mutual
                # distance twelve decodes at zero bit errors across a plateau
                # -- measured on `ws8eoc-0910/first-rms.wav`, fifteen
                # alignments from 300 samples early to 120 late -- and
                # `rxfront._best_cs` resolves that by taking the earliest
                # admissible one, so its answer is the leading edge of its own
                # capture range rather than the packet's phase. The
                # acquisition's coherent argmax puts the same head at 7200,
                # which is where the crop's manifest and the fixture's PCM
                # both have it; the tracked read reports 6900 and stays there,
                # and in `test_p3_morning_loop`'s scene that step reached the
                # transmit comb through `p3_reply_shift`.
                #
                # The clock still closes: a peer that has moved off the
                # projection by more than the reader's four symbols misses
                # here and falls through to `_p3_changeover_packet`, which
                # measures phase. What the projection accumulates meanwhile is
                # the peer's cycle against our nominal 60000 -- 2.3 samples a
                # cycle on `onair-0913-0014`, 2.5 on `onair-0912-2321` --
                # against the 300-sample bias it removes.
                return replace(ev, start=at, t=at / FS), "changeover tracked"
        if changeover_first and not self._tracked_only:
            got = self._p3_changeover_packet(audio)
            if got is not None:
                return got
        ev = self.sync.packet(corrected, preferred_only=self._tracked_only)
        if ev is not None:
            # A single-shot decode ends the run of copies, `decode_expected_p1_packet`'s
            # rule: the field is delivered and the peer stops repeating it.
            self.p3_memory.clear()
            self._p3_changeover_pending = False
            self._p3_repeat_changeover = False
            return ev, "tracked"
        if self._tracked_only:
            # THE TRACKED READ OWNED THE CYCLE AND NEVER PAID FOR IT. Aiming at
            # the projected head is a 3 ms read where the acquisition is a 10,
            # so a cycle that has just delivered spends the cheap one and keys.
            # Over `onair-0913-0014` and `onair-0912-2321` that read delivered
            # nothing at all, while every cycle it owned -- every cycle
            # immediately after a delivery, 12 of them -- held a CRC-valid copy
            # the acquisition below reads off the same window. Run the tracked
            # reads first, which is what they are cheap for, and fall through
            # rather than spend the cycle on them.
            if not self._p3_acquisition_fits(self.P3_ACQUIRE_RESERVE_S):
                return None
            # The CS3 acquisition and nothing behind it: the blind
            # variable-header sweep is the 85 ms one named above, it is bounded
            # by nothing, and a reply deadline is not where it may run --
            # `test_prekey_miss_does_not_launch_a_blind_scan`. Every P3
            # delivery of both arms came from this acquisition anyway.
            return self._p3_changeover_packet(audio)
        ev = rxfront.decode_expected_packet(corrected, self.p3_memory)
        how = "scanned"
        if ev is None and self.p3_receive_offset_hz:
            # WS8EOC's 64.760 s repeat passes at zero correction but not at
            # the -25 Hz acquired on CS3. Try the unshifted recording once,
            # only in the early scan. Do not combine the same physical frame
            # twice or discard the acquisition correction needed by others.
            ev = rxfront.decode_expected_packet(audio)
            how = "scanned RX 0 Hz"
            if ev is not None:
                self.p3_memory.clear()
        if ev is None:
            # A changeover has no normal variable-header block. In the long
            # listen gaps of the failed WS8EOC arm, complete CRC-valid packets
            # were therefore still missed after the short head was lost.
            return None if changeover_first else self._p3_changeover_packet(audio)
        self._p3_changeover_pending = False
        self._p3_repeat_changeover = False
        self.sync.observe(ev)
        return ev, how

    P3_FINE_STEPS_HZ = (5.0, -5.0)
    """The step `p3acquire.OFFSETS_HZ` has no room for.

    The changeover search sweeps 25 Hz apart while `p3acquire.control_signal`
    reads the same carriers on `CONTROL_OFFSETS_HZ`, 5 Hz apart. WS8EOC's
    cycle 13 of `onair-0913-0014` is a complete CRC-valid `b'RMS'` in the live
    receive window that reads at +5 Hz and at no offset the coarse grid holds.
    """

    P3_ACQUIRE_RESERVE_S = .025
    """What a cycle must have left before it may spend one on the acquisition.

    Measured over the thirty-five receive windows of `onair-0913-0014`, single
    worker on this machine: `p3acquire.changeover` on the coarse list costs
    3.4 ms on a slot holding nothing and 9.2-17.9 ms on one holding a head to
    validate at full rate. The three `ws8eoc-0913-*` crops of that arm, which
    are the same windows cut to disk, read 10.7, 13.1 and 14.3 ms median.

    NOT THE MEASUREMENT ITSELF. 17.9 ms was the worst case of a quiet machine,
    and this is admission rather than accounting: the read that overruns costs
    the transmit slot, so the reserve carries the measurement plus the spread
    a loaded host adds to it. The numbers above stay what they are.
    """

    CHANGEOVER_BODY_RESERVE_S = .015
    """What the frame behind a CS3 head costs, and what a cycle must have spare.

    `rxfront._cs_event` reads the changeover packet on both carrier
    arrangements: over the 55 linked PACTOR-3 cycles of
    `captures/onair-0913-2152`, 7.6 ms median, 14.4 at the ninetieth percentile
    and 56.5 at worst -- against the 7.6 ms a cycle holds between its last read
    and `_regrid`'s admission check. It was the only read in the loop that never
    asked whether it fit, and 21 of the 37 windows in which a changeover
    codeword decoded at the tracked anchor lost their transmit slot.

    Declining it costs the three bytes the new sender puts behind the head for
    one cycle: the codeword still yields the link, and `_p3_changeover_packet`
    reads the body out of the same buffer behind the key.
    """

    P3_ANSWER_ACQUIRE_RESERVE_S = .012
    """What `_acquire_answer`'s frequency sweep costs, and what it must have.

    The entry path now searches forty-one hypotheses over the same bounded
    bracket, covering KB5LZK's recorded -86-Hz replies. Twelve milliseconds is
    a conservative admission allocation, not a measured complete-loop bound;
    capture/deadline replay must qualify it before another RF candidate.

    Previously `p3acquire.control_signal` searched thirty-one hypotheses over the 0.46 s
    bracket around the anchor: 5.9 ms median on this machine over the receive
    windows of `captures/onair-0914-0850`, and `p3acquire._bands` prices the
    same shape at 13.9. It is the cold half, so it runs on every cycle of a link
    whose peer never sends a codeword -- an IRS stint under a gateway -- and on
    that arm's geometry the window closes 1.8 ms in front of the check. Ungated
    it spent 23 of 56 slots and took the cadence to two, which is what a gateway
    reads as a station that has stopped answering.
    """

    P3_TRACKED_KEEP_S = spec.CYCLE_SHORT_S + .15
    """How much of a recovered window a TRACKED read may transform.

    Both tracked readers aim at ONE instant and both sit in the last cycle --
    `_p3_packet` at the projected row 0, `_p3_cs` at the grid's own codeword
    position -- while `p3acquire.compensate` runs over the whole window: 0.88 ms
    of the 0.95 s a one-slot cycle collects and 4.18 of the 4.71 s the third
    recovered slot of `arm-v23-A-40-ws8eoc` collected. That growth is what makes
    a lost slot self-sustaining, the recovered cycle being dearer than the one
    that lost it, and none of it is read.

    THE BLIND LADDER BEHIND A MISS KEEPS THE WHOLE WINDOW: it has a cycle to
    spend and it is what finds a changeover.

    AND SO DOES ANY CYCLE A LONG FRAME MAY ARRIVE IN (`_p3_tracked_keep_n`).
    This is the SHORT cycle's number and a long frame is 3.37 s on its own, so
    a link holding the long cycle -- or one with a cycle command out and no
    answer behind it yet -- would have its answer trimmed away entirely. The
    saving is what the short cycle's budget needs and the long cycle's does
    not: its window is 3.75 s wide before a slot is ever given back.
    """

    P3_FINE_RESERVE_S = .035
    """...and what the 5 Hz refinement behind it costs: 1.9 ms median over the
    same windows, 26.7 ms where several fine hypotheses each accept a head, and
    29.9 ms for the coarse list and the refinement together on the three crops.
    Spent against the same headroom rule as the coarse reserve above."""

    P3_COLD_RESERVE_S = .150
    """Cold changeover frequency coverage, in an early scan only.

    The 2323 WS8EOC recording needs about -15 Hz, between the coarse grid
    and the +/-5 Hz neighbours of zero. Completing the 5 Hz grid costs
    55-64 ms on its first held window and 85-101 ms on four-second cuts
    on this host. Keep a separate reserve; this is not a pre-key reader.
    """

    P3_FOLLOW_TOL_HZ = 6.0
    """How far a fresh acquisition may sit from a candidate and still be it.

    The reader accepts a control within about 8 Hz of true and refines by up to
    `p3acquire.FINE_LIMIT_HZ`, so two reads of one transmitter land within a few
    hertz of each other and two reads of different ones do not."""

    P3_FOLLOW_STEP_HZ = 2 * p3acquire.FINE_LIMIT_HZ
    """...and how far it may move the session on its own word: one coarse grid
    step, which is what `FINE_LIMIT_HZ` is half of. Inside that the two
    hypotheses are neighbours and either could be the same peer drifting; past
    it they are different grid points and one of them is an alias."""

    @property
    def p3_transmit_offset_hz(self) -> float:
        """The receive raster, where the PACTOR-1 leg does not contradict it.

        DERIVED AND NOT STORED. `p3_receive_offset_hz` is written from several
        places -- the session reset, `_listen_through_breakin_body`'s bounded
        body read, a test placing a peer -- and a second copy of it went stale
        at every one of them. `_follow_p3_offset` sets the flag and nothing
        else does, so a raster that arrives any other way is followed exactly
        as it was before this check existed."""
        return 0.0 if self._p3_offset_contradicts_p1 else self.p3_receive_offset_hz

    def _follow_p3_offset(self, hz: float) -> None:
        """Move the session's receive raster onto a decoded offset -- or not.

        WHAT THIS GUARDS IS THE TRANSMITTER. `RadioTx._p3_offset` keys every
        PACTOR-3 burst at whatever this holds, so one acquisition that lands on
        the wrong 25 Hz grid point takes the whole reverse channel off the
        peer's raster with it. MEASURED, `captures/onair-0913-2152`: the peer
        sat at -24.5 Hz for the arm and one frame read +75.2; `tx_43` went out
        at +75.0, a hundred hertz from the station it was answering.

        SO A JUMP IS A CANDIDATE, NOT A CORRECTION. Inside one coarse step the
        read is the same peer as the session already has and is taken. Past it,
        the first frame only nominates -- the second one that agrees with the
        nomination is what moves us, and a read the session takes in the
        ordinary way clears the nomination with it. A session that has acquired
        nothing yet has no estimate to corroborate against, so its first read
        establishes one.

        The frame itself is delivered either way: it decoded, and the offset it
        decoded at is a reading of the air, not of the far end's identity.

        AND THE TRANSMITTER IS CROSS-CHECKED AGAINST PACTOR-1, WHICH THE RECEIVER
        IS NOT. `p3acquire._refined` is decision-directed off a differential
        codeword, so a constant per-symbol phase convention is a frequency to it
        and there is no reading it can take that separates the two: a synthetic
        control keyed at exactly 0 Hz with a +45 degree per-symbol convention
        acquires at +12.6 Hz, and +270 degrees at -25.1. The PACTOR-1 reader has
        no such ambiguity -- it correlates FSK lines at a fixed 1400/1600 Hz and
        searches time only -- so a zero-error codeword bounds the peer to
        `p3acquire.P1_CROSS_CHECK_HZ` of nominal. Where the two disagree past
        that bound the PACTOR-1 read wins and the transmitter stays put; the
        RECEIVE raster still follows, because whatever the reading means, the
        peer's frames demonstrably decode there and a tracked read at 0 Hz would
        lose them. KB5LZK on 40 m, 2026-09-15: P1 tones within 0.4 Hz of nominal,
        P3 control acquired at +74.6, and every data packet keyed there unread.
        """
        hz = float(hz)
        if (not self._p3_offset_fixed
                or abs(hz - self.p3_receive_offset_hz) <= self.P3_FOLLOW_STEP_HZ
                or (self._p3_offset_candidate is not None
                    and abs(hz - self._p3_offset_candidate) <= self.P3_FOLLOW_TOL_HZ)):
            self.p3_receive_offset_hz = hz
            self._p3_offset_fixed = True
            self._p3_offset_candidate = None
            words = sum(ev.protocol == Protocol.PACTOR1
                        and (ev.cs is not None or ev.spare is not None)
                        and _cs_errors(ev) == 0 for ev in self.cs_log)
            bound = p3acquire.P1_CROSS_CHECK_HZ
            agreed = not words or abs(hz) <= bound
            self._p3_offset_contradicts_p1 = not agreed
            if not words:
                verdict = "has read nothing this session -- unchecked, TX follows"
            else:
                verdict = (f"reads 0 +-{bound:g} Hz over {words} zero-error "
                           f"codeword{'' if words == 1 else 's'} -- "
                           + ("agreed, TX follows" if agreed else
                              "CONTRADICTED, TX stays at +0 Hz"))
            print(f"    [p3] follow RX {hz:+g} Hz; PACTOR-1 {verdict}", flush=True)
            return
        self._p3_offset_candidate = hz
        print(f"    [p3] RX {hz:+g} Hz is {hz - self.p3_receive_offset_hz:+.1f} Hz "
              f"off the session's {self.p3_receive_offset_hz:+g} Hz; held as a "
              "candidate, awaiting a second frame that agrees", flush=True)

    def _p3_coarse_offsets(self) -> tuple[float, ...]:
        # The first complete WS8EOC changeover in onair-0914-2252 validates
        # near -96 Hz, outside the shared +/-75 Hz grid. A later repeat also
        # validates on its biased near-zero branch, which leaves ordinary
        # SL3 off the receive filters. Add only the two cold edge hypotheses;
        # _p3_changeover_packet still commits an offset only with a body CRC.
        return (self.p3_receive_offset_hz,
                *(hz for hz in (*p3acquire.OFFSETS_HZ, -100.0, 100.0)
                  if hz != self.p3_receive_offset_hz))

    def _p3_fine_offsets(self) -> tuple[float, ...]:
        """The 5 Hz neighbours of the acquired correction, and of zero.

        Both, because the tracked reads carry the session's own correction and
        the coarse list is led by it, so a peer that has drifted since the
        acquisition sits beside one of the two and not beside the grid.
        """
        bound = max(p3acquire.CONTROL_OFFSETS_HZ)
        tried = set(self._p3_coarse_offsets())
        out: list[float] = []
        for base in (self.p3_receive_offset_hz, 0.0):
            for step in self.P3_FINE_STEPS_HZ:
                hz = float(base + step)
                if abs(hz) <= bound and hz not in tried and hz not in out:
                    out.append(hz)
        return tuple(out)

    def _p3_cold_offsets(self) -> tuple[float, ...]:
        """Cover the holes left by coarse acquisition and local refinement.

        A tracked CS3 head may change role without acquiring its body's
        frequency. Until a body establishes row zero, neither that head nor
        an offset learned from it is evidence that the coarse grid suffices.
        Keep the existing near-offset trials first, then cover the remainder
        of the entry receiver's +/-100 Hz range at 5 Hz spacing.
        """
        tried = set(self._p3_coarse_offsets()) | set(self._p3_fine_offsets())
        return tuple(sorted(
            (hz for hz in p3acquire.ENTRY_CONTROL_OFFSETS_HZ if hz not in tried),
            key=lambda hz: (abs(hz - self.p3_receive_offset_hz), hz)))

    P3_SL1_RESERVE_S = .004
    """One narrow header fit and CRC, before the shared admission reserve."""

    P3_SL2_RESERVE_S = .006
    """One short six-carrier field, at most two physical arrangements."""

    P3_WIDEBAND_RESERVE_S = .012
    """Bounded header/body trial, plus the shared 6 ms admission reserve."""

    def _p3_tracked_keep_n(self) -> Optional[int]:
        """Samples a tracked read may transform, or None for the whole window."""
        arq = self.host.arq
        if arq.cycle_long or arq.cycle_command_emitted:
            return None
        return round(self.P3_TRACKED_KEEP_S * FS)

    def _p3_acquisition_fits(self, reserve_s: float) -> bool:
        """Whether `reserve_s` of decode still fits in front of the key.

        The loop's own admission arithmetic -- `_regrid` asks
        `live.clamp_late(tx.key_instant(...))` of these same two objects -- with
        the decode this is about to spend held back from it, so a read that
        would push the carrier past its slot declines instead. A stream with no
        transmitter has no key to be late for and gets no bound, which is
        `_p3_decode_deadline`'s rule for the same reason.

        AND WITH `P3_DECODE_RESERVE_S` ON TOP, which is what every other read in
        the loop keeps (`_p3_decode_deadline`). `clamp_late` allows `key_notice`
        and nothing else, so asking it bare spent the acquisition into the
        decoder reserve the rest of the cycle holds: `P3_ACQUIRE_RESERVE_S` was
        the measured worst case exactly, the read was admitted with about
        0.1 ms in hand, and an overrun costs the transmit slot.
        """
        tx = getattr(self.host, "peer", None)
        live, raster = getattr(tx, "live", None), getattr(tx, "raster", None)
        slot = getattr(tx, "slot", None)
        if (live is None or raster is None or slot is None
                or getattr(live, "transmit", None) is None):
            return True
        return not live.clamp_late(tx.key_instant(raster, slot)
                                   - round(reserve_s * FS)
                                   - round(P3_DECODE_RESERVE_S * FS))

    def _p3_changeover_packet(self, audio: np.ndarray):
        trim = max(0, len(audio) - 4 * FS)
        window = audio[trim:]
        got = p3acquire.changeover(window, offsets=self._p3_coarse_offsets())
        if got is None or got.event.packet is None:
            fine = self._p3_fine_offsets()
            if fine and self._p3_acquisition_fits(self.P3_FINE_RESERVE_S):
                got = p3acquire.changeover(window, offsets=fine)
        if ((got is None or got.event.packet is None)
                and self._p3_row0 is None and not self._tracked_only
                and self._p3_acquisition_fits(self.P3_COLD_RESERVE_S)):
            # A head alone cannot authorize an ACK. Acquire the missing body
            # while there is time, then let the normal tracked read and reply
            # clock checks place the acknowledgement.
            got = p3acquire.changeover(window, offsets=self._p3_cold_offsets())
        if got is None or got.event.packet is None:
            return None
        self._follow_p3_offset(got.offset_hz)
        self._p3_changeover_pending = False
        self.p3_memory.clear()
        ev = replace(got.event, start=got.event.start + trim,
                     t=got.event.t + trim / FS)
        return ev, "P3 changeover acquired"

    def _p2_packet(self, audio: np.ndarray):
        """PACTOR-2's data frame, anchored on the frame marker it arrived with.

        Not tracked across cycles the way PACTOR-3 is, and it needs no equivalent:
        the marker IS the anchor, it is carried by every burst, and reading it
        costs 14.5 ms of the 1.30 s window against the 48.0 ms of a blind
        PACTOR-3 scan on the same window. There is nothing left over for a `SyncedRx` to
        buy -- the same reason `decode_expected_p1_packet` has no counterpart.

        The field layout is PACTOR-3's -- data, status byte, CRC -- and that is
        READ rather than carried over. Byte `crc_bytes - 3` holds the mod-4 packet
        counter in its low two bits, and over the two HB9AK recordings it steps by
        exactly one between adjacent cycles 25 times and holds still 8 times, with
        nothing else happening at all. A byte that walks a stranger's ARQ sequence
        in a recording this project did not transmit is in the place the sequence
        is read from.

        THE SPEED LEVEL IS THE FRAME'S OWN, as of 2026-09-02. It was
        `arq.P1_SPEED_LEVEL` here, on the grounds `_p1_packet_event` reports it:
        `arq._gear_cs` turns a clean run into CS4 -- "send the next level up" --
        and while the codeword that would carry the request was PACTOR-1's, in
        which the same index is Speedchange, a station that could not key
        PACTOR-2 had no gear to ask a PACTOR-2 peer for. It keys one now
        (`ptc.TRANSMITTABLE`), the request goes out as `pactor2.control_signal`,
        and `arq.P2_LADDER` is the ladder it is counted against -- four rungs,
        climbing to three. So the level travels, and the protocol travels beside
        it because the two answer different questions (`arq.on_rx_packet`): the
        status byte's layout is the protocol's, and the data type is three bits
        wide in both PACTOR-2 and PACTOR-3 where PACTOR-1 spends two.

        `spec.field_payload` and not a bare rstrip, for the same reason
        `_p1_packet_event` uses it: a field that is nothing but the walking
        template carries no payload at all, and delivering one hands the host a
        decompressor's reading of `spec.TEMPLATE`.
        """
        got = p2rx.decode_expected_burst(audio)
        if got is None:
            return None
        t, path, field, bin_pair = got
        self.p2_bin_pair = bin_pair
        info = field[:path.crc_bytes - 2]
        st = info[-1]
        data = spec.field_payload(info[:-1])
        return rxfront.Event(
            t, "packet",
            f"P2 {path.name} {len(data)}B status=0x{st:02x} "
            f"(seq={st & spec.STATUS_SEQ}) {data[:20]!r}  [CRC-VALID]",
            protocol=Protocol.PACTOR2,
            packet=(path.level + 1, st, data, True)), "P2 marked"

    def skip(self, seconds: float) -> None:
        """Stream time nobody here heard: our own transmission, or audio taken
        out of the stream and handed to a scan instead of to the decoder.

        OWED, NOT SETTLED HERE. `RollingRx.skip` empties the buffer as well as
        advancing the clock, and `RadioTx._tx` calls this the instant the carrier
        comes down -- three statements ahead of the `flush()` in the same cycle,
        whose entire input is that buffer (see `flush`, and `RollingRx.hold`,
        which holds the bridge's audio back expressly *for* it). So the one
        decode the transmit slot budgets for ran on an empty buffer on every
        keyed cycle, and the audio it was budgeted to read reached the WAV on
        disk and nothing else. It costs most where the buffer is longest: an IRS
        hold cycle listens through the peer's whole 840 ms packet and feeds none
        of it -- `_listen_until_answer` holds every slice while we are receiving
        -- so the flush was the only decoder that audio was ever going to reach.

        Settling it at the next decode instead keeps what the drop was for. The
        clock still steps over the burst before anything captured after it is
        fed, so the two sides of a transmission are never spliced into one
        window; the audio in hand when the key went up is simply read first.
        """
        self._unheard += seconds

    def _settle(self) -> None:
        """Pay the debt before any new audio joins the buffer: what is still in
        it belongs to the far side of the transmission and cannot be decoded in
        one window with what comes next."""
        if self._unheard:
            self.rx.skip(self._unheard)
            self._unheard = 0.0

    def _note_p3_frame(self, ev, absolute: int) -> bool:
        """Accept one CRC-valid physical frame across tracked and rolling reads."""
        if self._p3_delivered_at is not None and absolute <= self._p3_delivered_at + rxfront.SPS:
            return False
        raster = getattr(self.host.peer, "raster", None)
        after_command = getattr(raster, "_p3_command_row0", None)
        if (not ev.breakin and after_command is not None
                and absolute < after_command - rxfront.SPS):
            # A late backstop can emit after an unread earlier frame. That
            # newly decoded historical frame is not an answer to this CS6.
            return False
        self._p3_delivered_at = absolute
        trial = getattr(self.host.peer, "timing_trial", None)
        if trial is not None:
            phase = absolute if ev.breakin else absolute - p3frame.DATA_OFFSET * rxfront.SPS
            trial.packet(phase, ev.packet[1], ev.packet[2],
                         breakin=ev.breakin, long_cycle=ev.cycle_long)
        if ev.breakin:
            self._seed_p3_changeover_clock(absolute)
        else:
            self._p3_row0 = absolute
            self._p3_cycle_n = round((spec.CYCLE_LONG_S if ev.cycle_long
                                     else spec.CYCLE_SHORT_S) * FS)
            path = (placement.LONG_PATHS if ev.cycle_long
                    else placement.SPEED_PATHS)[ev.packet[0]]
            self._p3_span = rxfront._frame_span(path)
        raster = getattr(self.host.peer, "raster", None)
        if raster is not None:
            phase = absolute if ev.breakin else absolute - p3frame.DATA_OFFSET * rxfront.SPS
            width = round(placement.PACKET_S * FS) if ev.breakin else (
                rxfront._packet_span(self._p3_span)
                + p3frame.DATA_OFFSET * rxfront.SPS)
            raster.note_p3_packet(phase, width, self._p3_cycle_n,
                                 swapped=ev.carrier_swapped,
                                 identity=(ev.breakin, *ev.packet[:3]))
        return True

    def _on(self, ev, *, anchored: bool = False,
            connect_confirmed: bool = False) -> None:
        # The wide final listen can let the rolling decoder revisit a single
        # candidate from the last call. Only the existing narrow-band,
        # multi-cycle acquisition may promote P1 control words to a connection
        # here. CRC-valid DATA still follows its normal connection path.
        if (self._final_connect_listen and self.host.arq.state == State.CONNECTING
                and ev.protocol == Protocol.PACTOR1 and ev.kind == "cs"
                and not connect_confirmed):
            return
        # ONE TRANSMISSION, ONE EVENT, ACROSS THE CYCLE BOUNDARY TOO.
        # `new_cycle`'s one-shot flags hold that within a cycle and cannot hold it
        # across one, and the two readers do not always reach a burst in the same
        # cycle: the anchored read is aimed at an instant and takes the peer's
        # answer in the cycle it arrives, while the sweeping decoder needs
        # `MIN_DECODE_S` behind it and can reach the same burst a cycle later.
        # Searching `p1rx.CS_SEARCH_HALF_S` made that reachable -- the connect
        # answer of a peer turning around 5 ms inside the nominal is read on time
        # now -- and the second copy of a codeword is a REQUEST in PACTOR-1, so
        # the packet counter stopped advancing.
        #
        # Asked of the SWEEP and not of the anchor, which is the asymmetry that
        # makes it safe: the anchored read is aimed once per cycle at an instant
        # the grid owns, so two of its own reads are two cycles by construction,
        # where the sweep re-decodes a rolling buffer and is the one that can say
        # the same thing twice.
        if not anchored and ev.kind in ("cs", "unassigned") and any(
                abs(ev.t - t) < 2 * spec.P1_CS_S for t in self._anchored_at):
            return
        if anchored and ev.kind in ("cs", "unassigned"):
            self._anchored_at.append(ev.t)
        p3_frame = (ev.protocol == Protocol.PACTOR3 and ev.kind == "packet"
                    and ev.packet is not None and ev.packet[3]
                    and (ev.breakin or ev.carrier_swapped is not None))
        if p3_frame and not self._note_p3_frame(ev, round(ev.t * FS)):
            return
        self.count += 1
        if ev.kind in ("cs", "unassigned"):
            self.words_at.append(ev.t)
            # BOTH KINDS ARE TRANSMISSIONS THE PEER MADE, and the summary counts
            # transmissions. Only `cs` used to be kept, so a session's tally of
            # what came back excluded the unassigned words entirely -- and on
            # 2026-08-26 that printed `3 CS1/CS2, 0 other` over ten `0x59A` at
            # zero bit errors, a peer commanding PACTOR-3 once a cycle scored as
            # an empty channel. `cs_seen` still counts only the four words with a
            # meaning, because that gate is about what the FSM may act on.
            self.cs_log.append(ev)
        if ev.kind == "cs":
            self.cs_seen = True
        print(f"    {self.tag} {ev.t:6.2f}  {ev.kind:8s} {ev.text}",
              flush=True)
        traffic.received(self.host, ev)
        sent_before = self.host.sent_total
        was_p1_connecting = (self.host.protocol == Protocol.PACTOR1
                             and self.host.arq.state == State.CONNECTING)
        # A mail ACK can synchronously emit the next packet from the ARQ.
        # Give its TX guard the decoded control's geometry before that callback,
        # rather than the previous peer data packet's occupied interval. The
        # normal end-of-cycle forecast repeats this observation for reporting.
        peer = self.host.peer
        raster = getattr(peer, "raster", None)
        if (getattr(peer, "reply_clock", None) is not None and raster is not None
                and ev.protocol == Protocol.PACTOR3 and ev.kind == "cs"):
            phase = round(ev.t * FS)
            if not any(lo <= phase <= hi for lo, hi in getattr(peer, "keyings", ())):
                raster.note_peer_codeword(phase, P3_CS_N, _cs_name(ev),
                                          self.host.arq.dxcall,
                                          protocol=ev.protocol)
        self.host.on_rx_event(ev)
        if was_p1_connecting and not (self.host.protocol == Protocol.PACTOR1
                                     and self.host.arq.state == State.CONNECTING):
            self.rx.enable_p3_packets_after(ev.t)
        if p3_frame:
            self.frame_seen = True
            self._p3_clock_role = self.host.arq.role
        if self.p1_setup_phase == "call" and not self.p1_setup_finished and (
                self.host.sent_total > sent_before
                or self.host.arq.role != ISS
                or self.host.protocol != Protocol.PACTOR1
                or self.host.arq.state not in (State.CONNECTING, State.CONNECTED)):
            self.p1_setup_finished = True


def _report_collision(bursts: list[tuple[int, int]], tx) -> None:
    """Where a burst at the FSK tones sat against OUR carrier, and where the next lands.

    The operator hears the answer to this instantly -- "as we unkey the other
    station is already responding" -- and the program had no way to say it. Every
    number it printed was relative to a slot boundary or a window edge, which are
    where we MEANT to transmit; none was relative to the carrier itself. Three
    analyses of one collision reached three different conclusions from that gap,
    two of them confidently wrong, so this prints the interval instead of leaving
    it to be reconstructed afterwards from capture files.

    THE INTERVAL IS OURS TO STATE; THE SENDER IS NOT. `bursts` comes from
    `p1_bursts`, an energy-shape test on the 1400/1600 bins -- measured on
    the corpus, that family also fires on VARA, on PACTOR-2 and on a 500 Hz-class
    ARQ station, and nothing here reads a bit or a callsign. So these lines time a
    burst and decline to say whose, which matters most exactly where they print:
    live, while the operator is deciding whether the peer answered under our
    carrier. Said as "their burst" it was an identification the code never made.

    BACKWARD ONLY, AND THAT IS THE CHANGE. This also printed a forecast about
    the next cycle -- `[predict]`, built by adding one slot to `max(onsets)` and
    assuming the source repeats a 120 ms codeword there. It is gone from here;
    `_forecast_next_key` is what replaced it, and the difference is that the
    forecast now has to name a codeword this station decoded.

    BOTH ENDS OF OUR KEYING ARE THE CARRIER'S. `tx_key_up` and `tx_end`, never
    `tx_audio_start`: the settle is on the air like any other part of the
    keying, and an overlap measured from the audio cannot see the interval this
    exists to catch. The backward gap runs to whichever end of ours the burst is
    outside of, which is why it names that end.

    AND IT IS STRUCTURALLY BLIND TO THE COLLISIONS THAT MATTER MOST. The rig
    mutes this receiver while we key, so a burst under our own carrier is in no
    recording of ours and produces no onset: 2026-08-26 printed `[collide]`
    zero times in a session a third party's receiver measured twenty codewords
    transmitted over. `[clear]` is a statement about the bursts we could hear
    and never about the channel.
    """
    if not bursts or tx.tx_key_up is None or tx.tx_end is None:
        return
    for a, burst in bursts:
        overlap = min(a + burst, tx.tx_end) - max(a, tx.tx_key_up)
        if overlap > 0:
            print(f"    [collide] a burst at the FSK tones overlapped our carrier "
                  f"by {overlap / FS * 1e3:.0f} ms of its "
                  f"{burst / FS * 1e3:.0f} ms; not attributed", flush=True)
        else:
            after = a >= tx.tx_end
            gap = (a - tx.tx_end if after else a + burst - tx.tx_key_up) / FS
            where = ("after our carrier dropped" if after else
                     "before our carrier came up")
            print(f"    [clear] a burst at the FSK tones "
                  f"{'started' if after else 'ended'} "
                  f"{abs(gap) * 1e3:.0f} ms {where} "
                  f"({burst / FS * 1e3:.0f} ms of it); not attributed",
                  flush=True)


def _peer_bursts(seg: np.ndarray, seg_start: int) -> list[tuple[int, int]]:
    """Capture-stream sample and length of every PACTOR-1 burst in this window.

    `seg` must END at the reader's current position -- the caller's
    `live.pos - seg.size` is `seg_start` -- so the indices land on the same
    continuous stream as the grid boundaries. That is the whole reason this is
    done in samples: the phase between our raster and the peer's becomes a
    subtraction of two integers rather than a difference of two clock readings.
    """
    return [(seg_start + int(t * FS), round(d * FS))
            for t, d in rxfront.p1_bursts(seg)]


ACQUIRE_T0 = p1rx.ACQUIRE_LEAD_S
"""Earliest offset into a capture the codeword search can be aimed at: it reads
`t0 - ACQUIRE_LEAD_S` and returns None rather than clipping.

One number for one reason, and it was two copies of a literal: the floor here is
the search's lead, so a band aimed at the floor is the earliest one the search
will accept at all."""

ACQUIRE_FLOOR_S = 0.055
"""The earliest turnaround the connect search may read, seconds after our carrier
drops. MEASURED AT 14 ms AND HELD AT 55, which is the whole of what this constant
is for: the two numbers are different questions and both are written down here.

THE RIG'S MUTE IS 14 ms, NOT 55. `TR_SWITCH_S`'s 55 came off the WS8EOC sessions
of 2026-07-28 -- "-50 dB for the first 55 ms, window median by 60" -- and this
station does not behave that way now. Over 34 keyed slots of 2026-09-16, three
arms and two bands (`pactor-current-kb5lzk-30-pounce-20260916T155343Z`,
`...ws8eoc-80-sense-20260916T150542Z`, `...ws8eoc-80-pounce-20260916T132318Z`),
the receive chain reads -48 to -65 dB for the first 12 ms and is back within 3 dB
of that window's own floor at d = 13-15 ms, median 14, on every slot. So the band
between 20 and 55 ms is audio we can hear and do not search, and there really is
peer traffic in it: KB5LZK answers the raster at d = 20-46 ms, ten zero-error
0x59A in the sixteen in-link slots of that 30 m call.

AND IT STAYS AT 55 ANYWAY, because the corpus prices the 35 ms. Over the same
32239 negative cycle-windows from 1188 recordings, one draw of the search per
window at the read tail:

      floor   accepting windows   with `ANSWER_CODEWORD_X`   false links (gated)
        55      1161  (3.60%)          65  (0.20%)                   0
        20      1766  (5.48%)         104  (0.32%)                   1

A wider band is more alignments and `p1rx.acquire_null` grows with the span, so
the ungated rate rises by half; what matters is that the level gate no longer
holds the line -- 20 ms puts a phantom link back into the corpus, on
`rf-corpus/offair/ws8eoc_witness_20260806/kiwi_witness.wav`, and the phantoms of
2026-08-06 are exactly what the rule was bought against.

WHAT THE 35 ms WOULD BUY IS NOT IN THE TAPES EITHER. On the KB5LZK 30 m call the
floor at 20 reads nothing the floor at 55 does not: the connect answers of slots
21-23 are found at both (d = 52.5/52.5/56.2 against 55.0/55.0/56.2), and the
in-link 0x59A at 20-46 ms are outside `p1rx.ACQUIRE_ALPHABET` -- a connect answer
is CS1 or CS4 -- so no floor reaches them. They are missed by the onset detector,
which wants 4.0x and reads them at 2.0-3.7x, and that is where their recovery
lives rather than here.

Re-measure the mute per rig -- `--preflight` does it in fifteen seconds -- and
re-price this against the corpus before moving it. Lowering it is a trade of
false links for band, and today the band is empty."""

ACQUIRE_TAIL_S = spec.P1_CS_S + 0.021
"""Audio the ANSWER SLOT reserves past the top of the band it is given.

A codeword starting at the top edge runs `P1_CS_S`, and 20 ms beyond that is the
smoothing tail `acquire_control_signal` reads out into -- plus a millisecond of
rounding. This is the schedule's number: `_budget` refuses to key a cycle whose
whole band falls inside it, and `_AnswerBand` scores its level over the band plus
a codeword, which is this same reach. `ANSWER_OCCUPIED_DB`'s 7 dB knee is priced
on exactly that slot, over 201 arms, so the slot is not free to move.

Named because two places need the same number and they had drifted apart by
exactly this much: `_budget` was passing schedules whose whole band
`_acquisition_window` then refused, which is how the g90 came to be a rig that
calls, reports "nothing decoded" every cycle, and cannot hear an answer at all.
"""

ACQUIRE_READ_TAIL_S = spec.P1_CS_S + 0.002
"""...and what the SEARCH needs, which is less, because it clips.

The smoother's 20 ms are taken where the capture has them and not required where
it does not -- `acquire_control_signal` clamps its own slice -- so what a read
actually costs past the top of its band is one codeword and two milliseconds of
rounding. THE SLOT AND THE READ ARE DIFFERENT QUESTIONS and they were one number
until the difference was charged to the peer: a keyed 1.25 s cycle captures
235-241 ms, which put the readable top at 94-100 ms and made it dither with the
capture length. VE3KPG's turnaround is 96-107 ms, so the top sat inside the
peer's own spread and cut it in half: over the fifteen calls to that gateway of
2026-09-13, 357 listen windows, the sessions read 42 zero-error codewords where
the same audio holds 95.

The top is 113-119 ms on those same captures now, which holds the whole of that
station's spread, and it costs the search 2.70% of negative cycle-windows against
3.60% -- 870 accepts to 1161 over 32239 windows of real off-air energy addressed
to nobody. What that buys is priced where it is spent: `_ConnectEvidence` fires
on ONE of those 1012 recordings either way.
"""

PEER_TURNAROUND_S = (0.087, 0.096, 0.134)
"""Where a real PACTOR-1 responder's answer STARTS: (min, median, max), seconds.

Measured with the ENERGY detector, which has no band and no codeword -- 69 burst
onsets over 114 keyed windows of the seven WS8EOC sessions of 2026-07-30. The
gate has to be open to measure what goes through it, and this is the number that
was got wrong the other way round: the search used to be handed a window ending
at 100 ms, it reported answers at 92-99, and that cluster was written down as the
peer's turnaround. It is the window's edge.

The distribution is not tight. 28% of those onsets start past 100 ms and the tail
reaches 134, so a caller with a slow PTT does not lose an edge case -- it loses
answers. What does NOT move is a station's turnaround inside one session:
the tightest three accepts in an eight-cycle span run 2.5 ms at their widest over
those sessions, which is what `_ConnectEvidence.TOL_S` is bought against.

IT IS NOT IN THE SEARCH'S UNITS AND DID NOT MOVE WITH IT. These are burst ONSETS
from the energy detector; the codeword search's `d` is a different instrument,
and the two were checked against each other on 2026-09-16 rather than assumed to
agree -- `p1rx.ACQUIRE_LEAD_S` has that comparison, which came out at 0.1 ms on
the answer the force arm connected on.

WS8EOC ON 80 m ANSWERS FASTER THAN THIS, AND KB5LZK FASTER AGAIN. The three
sessions of 2026-09-16 that corroborated a peer did it at 71-75 ms (WS8EOC, twice)
and 55-61 (KB5LZK), against the 87 ms floor these seven 40 m sessions of one
station put on it. So the distribution is a band-and-station reading rather than
the protocol's, and nothing may be gated on its lower edge: the floor the search
is actually held to is `TR_SWITCH_S`, our own mute.

The turnaround analysis that re-derives all of it is in the project's working
notes, which are not part of this distribution.
"""


def _d_max_n(cycle: float, settle: float) -> int:
    """The latest turnaround we could still hear out before we have to key, samples.

    Not a tolerance: past it a control signal is still going when our carrier comes
    up, and nobody's implementation can do anything with one. This is
    `_MasterGrid.d_max_n`, the bound an onset is rejected against, and the upper
    edge `_acquisition_window` searches to.

    A named function rather than an expression at its one call site, because it had
    a second call site that was not a call: `test_p1cs` carried the arithmetic
    verbatim and fed the answer back in, so replacing this with `round(cycle * FS)`
    -- search the whole cycle, which is the one thing the band exists to prevent --
    was byte-identical across all 145 shrike tests. A bench that recomputes the
    production expression cannot disagree with it.
    """
    return round((cycle - settle - spec.P1_PACKET_S - spec.P1_CS_S) * FS)


def _acquisition_window(seg_n: int, since_tx: int, d_max_n: int,
                        tail_s: float = ACQUIRE_TAIL_S) -> tuple[float, float]:
    """Where in this capture a connect answer could be: `(t0, span)`, seconds.

    AN ANSWER IS DUE `d` AFTER OUR OWN CARRIER DROPS, and `d` is a turnaround --
    bounded below by the rig coming out of transmit and above by the last
    codeword we could hear out before we key again. `since_tx` is how far this
    window's first sample sits past that drop, so the band lands on the capture;
    a window that holds none of it gets no search at all.

    `tail_s` is how much audio the caller needs past the band's top. It defaults
    to the SLOT's reach, which is what `_budget` schedules and what `_AnswerBand`
    scores a level over; the codeword search asks for `ACQUIRE_READ_TAIL_S`, 19 ms
    less, because it clips its own slice rather than refusing. One argument rather
    than two functions: the band is the same band, and only its top edge is being
    asked a different question.

    That is the whole gate, and three separate things fall out of it rather than
    being asserted one by one:

      * A HUSH IS NOT SEARCHED PAST ITS FIRST CYCLE, and the first one is, which
        is not a concession: a keyed cycle's window closes before its own key, so
        the band belonging to the last call of a run lands in the cycle after it,
        hushed or not. From the second cycle on our last carrier is a whole cycle
        back and the band falls off the front of the window on its own. 7 of the
        74 candidates on file were accepted in a first hush cycle, and
        `_MasterGrid.update` takes `since_tx` for the same reason: `_place` is
        built on nothing we hear being an answer to us, which is true of the rest
        of a hush and false of the cycle that opens it.
      * NEITHER IS A SLOT HANDED BACK. `_regrid` extends the window past the
        boundary it could not key on, and five of the fifteen windows the three
        2026-08-06 calls put through this search were over a second long for that
        reason. Six of those fifteen accepted across the whole window against one
        across the band, and one of the three phantom links came from an accept
        833 ms in. The band is a property of our transmission, not of how much
        audio the cycle happened to leave.
      * AND AN IMPLAUSIBLE OFFSET IS NOT AN ANSWER. Receive audio is muted
        while our own T/R changes over, so a station answering inside that loses
        the head of its codeword -- a zero-error match there is a match on noise.
        That took the second phantom, an accept 13 ms in. How long that lasts is
        `ACQUIRE_FLOOR_S`, MEASURED, and it is not `TR_SWITCH_S`.

    MEASURED over 32239 cycle-windows -- real off-air energy addressed to nobody,
    from 1012 recordings: everything in rf-corpus and this station's own ARDOP
    captures. The search accepts in 5.1% of them across the window it used to be
    handed, 2.70% across the band at the slot's tail, and 3.60% at the read tail.
    Real answers pay almost nothing for that: of 57 accepts across the seven
    WS8EOC sessions of 2026-07-30 the band drops five, one of them the 993 ms
    outlier the session itself refused.

    THE BAND'S UPPER EDGE IS NOT A CHOICE, and neither is it the peer's
    turnaround. It is where our own schedule runs out of audio. See `_budget`.

    The project's working notes, not part of this distribution, re-derive all of it.
    """
    lo = max(ACQUIRE_T0 * FS, ACQUIRE_FLOOR_S * FS - since_tx)
    hi = min(d_max_n - since_tx, seg_n - tail_s * FS)
    return lo / FS, max(0.0, (hi - lo) / FS)


ANSWER_CODEWORD_X = 1.4
"""How far a candidate codeword has to stand over its own window's passband
median before it is a transmission rather than a reading of noise.

THE CODEWORD SEARCH HAS NO ENERGY TEST IN IT. It bandpasses 1200-1800 Hz, takes
the sign of the instantaneous frequency and matches twelve bits, so it accepts
band noise at whatever rate the alignments allow -- `p1rx.acquire_null`'s
arithmetic, and 6.6% of the windows of the WS8EOC tape of 2026-09-16 in practice.
That is the rate `_ConnectEvidence` is priced against, and it is also why five
"answers" stood in that session's log with a witness receiver 160 miles from the
gateway showing nothing in the gaps.

The level is `rxfront._cs_profile`'s p1 excess -- the mean of the two FSK tone
bins over the median of the passband, 20 ms windows -- taken as the MEDIAN across
the candidate's own 120 ms, which is its plateau rather than its best moment.
The same instrument `cs_evidence` prints, so the log line and the gate cannot
drift apart, and deliberately far below the 4.0x the onset detector fires at:
this search exists to read bursts that detector rejects.

MEASURED ON BOTH POPULATIONS, and the threshold is where they separate.
Positives are the nine accepts the three sessions of 2026-09-16 that produced a
link corroborated on -- WS8EOC 80 m twice (2.79/2.66/2.81 and 3.22/4.48/4.61) and
KB5LZK 40 m (1.87/3.26/1.73), the last of which the peer then answered in session
with two 0x59A and a CS2. Negatives are the 699 accepts of
`tests/shrike/fixtures/connect-ceiling-negatives.json`, every accept in every
recording of that corpus able to hold three: median 1.06, p90 1.27, p99 2.11.
The late CS1 of 2026-09-13 that `_ConnectTail` exists for is a sixth positive and
was not used to choose the threshold: it reads 3.14 in its own window.

      gate    negative accepts    false links    positives kept
      none         699                 1              9 of 9
      1.4           26                 0              9 of 9
      1.5           18                 0              9 of 9
      1.7           10                 0              9 of 9
      2.3            6                 0              7 of 9

1.4 keeps every accept a link was ever built on, 24% under the weakest of them,
takes 96% of the corpus's accepts off the rule's input, and closes the one false
link the rule still had. THE MARGIN IS A FLOOR AND NOT A CEILING: every positive
above was measured with this station's own post-key RFI in the window -- the tap
of 2026-09-16 has the receive floor 7.6 dB up from the first key-down -- and this
is a ratio against that floor, so a real answer's excess is depressed by exactly
the thing the margin is protecting it from. Find the emitter and these numbers
improve. Past 1.73 the gate starts costing real connects, and the accepts left in
the corpus above 2.3 are stations rather than noise -- this gate cannot tell a transmission addressed to somebody else from one
addressed to us, and it is not asked to.

It costs one pass of the same profile `cs_evidence` prints on every cycle that
decodes nothing -- 3.5 ms over a keyed cycle's window, 19 ms over a hush's --
paid only on an accept and in the dead time after the key.

The floor moves with the receiver, so it is a RATIO against the same window and
not a level: the tap analysis of that morning has this station's own RFI lifting
the receive floor 7.6 dB from the first key-down, which moves both terms."""


def _candidate_excess(seg: np.ndarray, at: float) -> float:
    """A candidate codeword's in-band excess over its own window, `x median`."""
    p1, _wide, ts = rxfront._cs_profile(seg)
    sel = (ts >= at) & (ts <= at + spec.P1_CS_S)
    return float(np.median(p1[sel])) if sel.any() else 0.0


INLINK_READ_BAND = (0.020, 0.130)
"""Where a linked peer's codeword is read when the onset detector found nothing.

THE DETECTOR'S THRESHOLD WAS TAKING DOWN ANSWERED LINKS. `_p1_runs` wants 4.0x
over the window median before a burst is offered to a decoder at all, and a
gateway that stands under that is read as silence -- which the retry budget
spends and `_give_up` acts on. `pactor-current-kb5lzk-30-pounce-20260916T155343Z`
is the case with the numbers on it: ten of that call's sixteen in-link slots hold
a 0x59A at ZERO bit errors, read at 1.99-3.71x, and the session logged "no
control codewords decoded inside the link", spent `--link-retries`, and signed off
with a QRT over a station that answered two cycles in three.

SO THE GRID IS ASKED INSTEAD OF THE DETECTOR. One read per alignment across the
band an answer is due in, zero bit errors, the level the candidate gate already
prices (`ANSWER_CODEWORD_X`) in front of it. That is the same trade
`p1rx.acquire_control_signal` makes during a call, and it is made here for the
same reason: the codeword's own exactness is a stronger gate than the shape test
in front of it, and this station cannot afford to hear only the loud half of a
gateway.

20 TO 130 ms IS BOTH ENDS OF THE RECORD AND THE WINDOW'S OWN EDGE. KB5LZK
answers this station at d = 20-46 ms on 30 m (band-power onset, n = 15, median
38) and `PEER_TURNAROUND_S` puts WS8EOC at 87-134 on 40 m, so a band that holds
one and not the other is a per-station reading rather than a protocol one. The
top is where a keyed cycle runs out of audio rather than where a peer stops
answering -- 130 plus a codeword is the whole of a 1.25 s cycle's listen window
-- which is `_d_max_n`'s argument in another place. Unlike the connect search
this may run BELOW `ACQUIRE_FLOOR_S`: the mute is 14 ms on this rig (that
constant has the measurement), a linked peer is a station we have already
identified, and a false read here delays a teardown by a cycle rather than
inventing a link.

THE OFFSET IT REPORTS IS WHERE THE READ STARTED, not where the peer keyed.
`p1rx.decode_control_signal` locks the burst inside the slice it is handed, so
the first alignment that reads is the earliest one holding the whole word, and on
a burst at 38 ms that is the floor of the band. The turnaround a linked session
wants is the grid's, which `_MasterGrid` keeps; this line says a word arrived and
what it stood at.

MEASURED on the 32239 negative cycle-windows of
`tests/shrike/fixtures/connect-ceiling-negatives.json`, real off-air energy
addressed to nobody: the read accepts in 839 of them (2.602%) and 65 of those
clear the level (0.202%), one cycle in about 500. What a false one costs is one
cycle of a budget that is eight; what the eight bought on the KB5LZK call was a
QRT over an answering gateway.

WHAT IT CANNOT DO FROM HERE is credit the budget for an UNASSIGNED word. An
assigned codeword reaches `arq.on_rx_cs` and the acknowledgement clears
`_inflight.retries` -- eight silent cycles against none, driven through the real
FSM in `tests/shrike/test_inlink_anchored_read_0916.py`. `0x59A` reaches
`note_upgrade_unread`, which only counts while an upgrade window is open, so on a
plain PACTOR-1 link the word is logged, delivered and charged a retry anyway.
Closing that is one line in `arq.py` -- the reset it already performs when an
upgrade is asked for again -- and it is not this module's to make."""

INLINK_READ_HOP_S = 0.002
"""The step between the alignments the anchored in-link read tries: a fifth of
`p1rx.CS_BIT_S`, which is what the read tolerates without losing a bit."""


def _anchored_answer(seg: np.ndarray, since_tx: int):
    """The peer's codeword in a linked cycle, read at the grid rather than at an
    onset: `(CS, seconds into seg, level)` at zero bit errors, or None."""
    lo = max(0.0, INLINK_READ_BAND[0] - since_tx / FS)
    hi = min(INLINK_READ_BAND[1] - since_tx / FS,
             (seg.size - spec.P1_CS_S * FS) / FS)
    at = lo
    while at <= hi:
        got = p1rx.decode_control_signal(seg, at, spec.P1_CS_S)
        if got is not None and got[1] == 0:
            level = _candidate_excess(seg, at)
            if level >= ANSWER_CODEWORD_X:
                return got, at, level
        at += INLINK_READ_HOP_S
    return None


def _grid_answer(sessrx, seg: np.ndarray, seg_start: int,
                 since_tx: int) -> Optional[str]:
    """`_anchored_answer`, delivered: the word and the operator's line for it.

    ONE BODY, BOTH LOOPS. This read went in wired to the setup loop alone, under
    a guard that reads CONNECTED -- which that loop is for the single iteration
    the link comes up in, after which it leaves for the hold loop and never
    returns. So the read that exists for a linked cycle ran in no linked cycle at
    all: across the 123 arm logs this station had flown when it was found, the line
    below appears zero times, the KB5LZK call it was written off included.

    Delivered the way the detector's own word is delivered, and no further: what
    a codeword buys the link -- an ack, a grant, a cycle the silence budget may
    not charge -- is the host's to decide, and it decides it the same way
    whichever instrument found the burst.

    EXCEPT THE CHANGEOVER, WHICH THIS READER MAY NOT DELIVER. A reversal owes
    the peer's packet: the head read rotates the grid AND collects the 840 ms
    behind the word, which is the whole of what the changeover is carrying.
    Twelve bits at the grid have none of that, and a grid rotated on them alone
    yields to a station whose packet was then never read -- the role desync the
    head read exists to prevent, arrived at by a different door. So it is
    reported and dropped, which also tells the operator what the head read
    missed.
    """
    got = _anchored_answer(seg, since_tx)
    if got is None:
        return None
    cs, at, level = got
    name = spec.P1_CS_NAMES[cs[0]]
    d_ms = (at + since_tx / FS) * 1e3
    if cs[0] == CS_BREAKIN:
        return (f"RX {name} read at the grid from d = {d_ms:.0f} ms and NOT "
                f"acted on -- a changeover carries a packet this read has not "
                f"got, and the grid may not turn over without it")
    sessrx._on(rxfront.Event(
        seg_start / FS + at,
        "unassigned" if cs.unassigned else "cs",
        f"{name}  (0 bit errors, PACTOR-1, read at the grid, "
        f"shift {'inverted' if cs.sense else 'normal'})",
        protocol="PACTOR-1", sense=cs.sense,
        **({"spare": cs[0]} if cs.unassigned else {"cs": cs[0]})))
    return (f"RX {name} read at the grid from d = {d_ms:.0f} ms -- 0 bit "
            f"errors, {level:.1f}x in-band over the codeword from there; the "
            f"onset detector offered nothing")


def _answer_origins(keyed_at: int, boundary: int, slot_n: int, data_n: int,
                    projected: bool) -> tuple[int, int]:
    """Carrier ends an answer in this window could be answering, best first.

    `keyed_at` is our last carrier that really dropped; the other is the SLOT'S
    OWN projection -- the call this window's slot would have carried, ending
    `data_n` into it. The peer answers the raster rather than the PTT, so a cycle
    we did not key still has a band and it hangs off the slot, which is why a
    hush and the final listen take the projection first.

    TWO, BECAUSE THE CYCLE THAT ENDS A HUSH FITS NEITHER DESCRIPTION. It keys --
    `hush` is already False in it -- while `keyed_at` still names the carrier
    from before the hush, a run of cycles back, so `_acquisition_window` was
    handed a `since_tx` well past the band and gave back nothing to search at
    all. Cycles 11, 20 and 29 of the WS8EOC call of 2026-09-16
    (`pactor-current-ws8eoc-80-sense-20260916T150542Z`) went unread that way, one
    at the end of each of that arm's three hushes: each of those windows holds
    46-52 ms of searchable band off its own slot, and was handed none.

    A FALLBACK AND NOT A TEST ON THE ANCHOR, which is the form that survives the
    schedule. "`keyed_at` is more than a slot back" looks like the same question
    and is not: a skipped slot, a re-aimed key or a hand-back moves `boundary`
    without any hush at all, and reading those as stale sends the search off our
    own carrier and into a band the peer is not in -- 50 entry campaigns of
    `test_entry_slots_kept` decoded nothing that way. So the window itself
    decides: the second origin is tried only where the first leaves nothing to
    search, and a window with a real band keeps it.
    """
    slot = boundary - slot_n + data_n
    return (slot, keyed_at) if projected else (keyed_at, slot)


#: How far the answer slot has to stand over the quietest this session has heard
#: that same slot before energy in it is reported as somebody transmitting.
#:
#: MEASURED BOTH WAYS, and it is a knee rather than a margin. The negative
#: population is 2096 scored listen windows from 73 real arms in which no PACTOR-1
#: control signal was ever decoded -- this station's own rig, its own transmit
#: grid, its own windows -- with the `answered` gate below forced open so the
#: level test alone is what is being measured. Against the three sessions where a
#: peer really did fill the slot:
#:
#:      dB   null arms      WS8EOC 0828   KB5LZK arm 6   KB5LZK arm 7b
#:       6      3.39%          14/15          10/15           1/1
#:       7      1.43%          14/15           6/15           1/1
#:       8      0.81%          14/15           5/15           1/1
#:      10      0.29%          12/15           1/15           1/1
#:
#: Seven halves the accepts on a channel nobody answered on for four of arm 6's
#: fifteen cycles, and 1.4% a cycle is under the 2.7% the connect codeword search
#: is already affordable at. In deployment the rate on that population is zero
#: rather than 1.4%: nothing ever answered in any of those 73 arms, so no grid
#: acquired and the gate below never opened. The 1.4% is the level test's own.
ANSWER_OCCUPIED_DB = 7.0

#: Listen windows the floor is built from before it may be stood on. A floor from
#: one window is that window, and the first window of a session is as likely to
#: hold the peer as any other.
ANSWER_FLOOR_WINDOWS = 4

#: How far the window has to RISE inside itself before the level above is allowed
#: to mean somebody answered. `rxfront.answer_onset`, in dB.
#:
#: Set from the 28 listen windows of `captures/onair-0904-1659`, which is the arm
#: the level test hung a link up on: the five cycles WS8EOC answered in step +3.8,
#: +3.8, +4.1, +5.7 and +6.1, and the two the arm flagged step +0.1 and +0.7. Two
#: sits 1.3 dB over the louder flag and 1.8 dB under the weaker answer. Two other
#: unread windows of that session reach +3.0 and +4.1 with no answer in them, and
#: they are why this is asked ONLY of a window the level test already stood up:
#: the onset says the energy began here, the level says there is energy, and
#: neither is an occupied slot on its own.
ANSWER_ONSET_DB = 2.0

#: ...and the other way a level may be believed: the same reading, again, in the
#: cycle before. An emission already under way when our carrier drops has no edge
#: to find and is the shape both 2026-08-28's WS8EOC and 2026-08-29's KB5LZK
#: arrived in -- fifteen consecutive windows apiece -- so a run of them is the
#: evidence a single window cannot carry.
#:
#: The previous window has to stand this far over the floor for the run to count.
#: Five, from the same 28 windows: the loudest window adjacent to either flag
#: stands +4.2 dB, and neither flag has a neighbour within 2.9 dB of itself.
ANSWER_PERSIST_DB = 5.0

#: ...and in the same place in the passband. The centroid of the excess over the
#: session's own quietest profile, `rxfront.ONSET_BAND_HZ` resolution; two
#: unrelated bursts in consecutive cycles are not one emission persisting. The
#: fifteen filled windows of `captures/onair-0828-1844` walk 1199-1664 Hz and
#: never move more than 400 between consecutive cycles.
ANSWER_BAND_TOL_HZ = 400.0


class _SlotReading(NamedTuple):
    """What one listen window had to say, and whether it is a finding.

    `occupied` is the only thing that moves anything downstream. A reading that
    is not occupied is a line for the operator and nothing else.
    """

    line: str
    occupied: bool


class _AnswerBand:
    """How loud the slot this link's answer is due in has been, cycle by cycle.

    THE ONE THING THE ONSET DETECTOR CANNOT REPORT. `_peer_bursts` is shaped like
    PACTOR-1 -- two tone bins over the passband median, with a veto on wideband
    energy -- so a peer that leaves PACTOR-1 raises neither half of it, and a
    peer that has changed waveform prints the line an EMPTY CHANNEL prints. The
    codeword path does not misread the burst; it never fires on it at all. On
    2026-08-28 that spent a retry budget against WS8EOC over fifteen consecutive
    cycles it transmitted in -- twelve of them printed as `NO CONTROL SIGNAL --
    nothing heard` -- on the cleanest receive path this project has recorded, and
    then signed off. The same twelve lines, over the same signature, came off
    KB5LZK on another band eight days later.

    IT NAMES NOBODY, AND CANNOT. A level in a slot separates a transmission from
    an empty channel and nothing else: sliced blind across 30923 cycle-windows of
    off-air energy addressed to nobody -- 902 corpus recordings, most of them of
    other people transmitting -- this accepts 36% of them, because that is what
    was in them. What buys the claim is not the level but where it is asked and
    what has answered there before, and the two gates below are the whole of it.
    So the line says `not attributed`, and the only thing it moves is whether a
    cycle counted as silent -- the same bound `arq.PactorArq.note_peer_heard`
    carries, for the same reason: the peer and an occupant want opposite things
    and this cannot tell them apart.

    ENERGY IS NOT A CODEWORD, and this is deliberately unable to become one. It
    yields a line and a level; it produces no onset, so nothing here reaches
    `_read_codeword_at_bursts`, `_forecast_next_key` or the grid tracker, and the
    only thing it is allowed to move downstream is whether the cycle was silent
    (`arq.PactorArq.note_unreadable_answer`). A burst nobody read may not
    acknowledge a packet, advance a counter or authorise a changeover.

    POSITIONALLY ANCHORED, which is what makes a bare level test affordable at
    all. It is asked once a cycle, across `_acquisition_window`'s band and no
    wider -- the span in which an answer to OUR OWN transmission has to lie, plus
    the codeword length so a burst starting at the far edge is inside it. A cycle
    we did not key has no answer due and gets no band, and so is never asked.

    AND THE FLOOR IS THIS SESSION'S OWN. What the excess is measured against is
    the quietest that same slot has been since the call went out, which is the
    only reference that survives an AGC, a band change and a codec gain the
    operator set that morning. It also states its own bound: a link whose peer
    has been transmitting through every window it ever collected has never heard
    the slot quiet, so its floor is the peer and this says nothing. That is the
    right direction to fail in -- the claim is that the channel got louder than
    this station has known it be, and a station that has known it no other way
    has not measured anything.

    AND A LEVEL IS NOT A FINDING, which is 2026-09-04's whole lesson and the
    reason `_SlotReading` has a second field. On that arm the top two windows of a
    28-window distribution spanning 8.6 dB were called occupied and the link was
    signed off on the second; the tape carries no PACTOR of any kind in either,
    the first fired 5.1 s BEFORE the peer granted PACTOR-3, the second 9.9 s after
    the peer's last codeword, and windows recorded with the link down and nobody
    on the air stand within 1.1 dB of the same threshold. What was in them was
    co-channel energy already up in the first bin the receiver could hear.
    So the level is reported as what it is -- a channel reading -- and an OCCUPIED
    SLOT needs one of the two things a channel reading has not got: an ONSET
    inside the window (`ANSWER_ONSET_DB`; an answer has a turnaround, and every
    one of that arm's own answers steps 10-13 dB at it), or the SAME READING IN
    THE CYCLE BEFORE at the same place in the passband (`ANSWER_PERSIST_DB`,
    `ANSWER_BAND_TOL_HZ`; an emission already under way when our carrier drops
    has no edge to find, and both sessions where a peer really did fill the slot
    filled fifteen consecutive windows of it).

    THE FIRST WINDOW OF A RUN IS THEREFORE ONLY A READING, and that is the price:
    against the three sessions a peer genuinely filled the slot in, this keeps 12
    of 15, 4 of 15 and 0 of 1 where the bare level kept 14, 6 and 1. It is
    affordable because of what it is spent on -- see `_report_answer_band`, where
    the finding now holds a cycle and nothing else. A window an unread emission
    covers for one cycle only, at the end of an arm, is the case that is lost.

    AND A READ WINDOW SEEDS NOTHING. The cycle before is evidence of an emission
    persisting only if nothing explained its energy: a codeword raises the level
    as surely as an occupant does, so a window the reader took is remembered as
    no window at all.
    """

    def __init__(self) -> None:
        self.floor: Optional[float] = None
        self.windows = 0
        self.bands: Optional[np.ndarray] = None
        self.last: Optional[tuple[float, Optional[float]]] = None

    def _centre(self, onset: Optional[rxfront.SlotOnset]) -> Optional[float]:
        """Where this window's excess sits, against the quietest profile so far.

        The level test's own reference, per band rather than over the whole of
        one: what is wanted is the place the extra energy is, and the running
        minimum is what makes that a place rather than the shape of the passband.
        """
        if onset is None:
            return None
        floor, self.bands = self.bands, (onset.bands if self.bands is None
                                         else np.minimum(self.bands, onset.bands))
        if floor is None:
            return None
        excess = np.maximum(onset.bands - floor, 0.0)
        if not excess.sum():
            return None
        hz = (rxfront.OCCUPANCY_BAND_HZ[0]
              + rxfront.ONSET_BAND_HZ * (np.arange(excess.size) + 0.5))
        return float((excess * hz).sum() / excess.sum())

    def sight(self, seg: np.ndarray, since_tx: int, d_max_n: int,
              *, read: bool, answered: bool,
              peer_owes_a_packet: bool = False,
              offset_hz: float = 0.0) -> Optional[_SlotReading]:
        """Fold one listen window in; a line where the slot is occupied, else None.

        `read` is whether PACTOR-1 was found in this window -- a decode, or an
        onset at the tones. EVERY window feeds the floor and only an unread one
        can produce a line, and the asymmetry is the whole design: a control
        signal raises this measure as surely as anything else does, so a cycle
        the reader took is a cycle with nothing left to report, while the channel
        underneath that codeword is exactly where the floor comes from. The
        lowest reading of the WS8EOC session is `hold_01`, a cycle that decoded.

        `answered` is `_MasterGrid.acquired`: has this grid ever heard the peer
        inside its own receive window. IT IS WHAT MAKES THE SLOT A SLOT. Until
        something has answered there, "where our answer is due" is a schedule and
        not a measurement, and a level test over a schedule is a level test over
        the channel -- which fires on every station that comes up on it. It
        latches, so the cycles this exists for -- the ones after the peer stops
        being readable -- are on the right side of it.

        `offset_hz` is where the session's receiver has the peer's PACTOR-III
        raster, and it reaches only the comb flag below -- the level, the width
        and the verdict are the measurements they were.
        """
        if seg is None or not seg.size:
            return None
        t0, span = _acquisition_window(seg.size, since_tx, d_max_n)
        if span <= 0:
            return None
        t1 = t0 + span + spec.P1_CS_S
        level = rxfront.quiet_level_db(seg, t0, t1)
        if level is None:
            # Missing/digitally held PCM measures no RF floor and cannot join
            # two otherwise separate observations into a persistent emission.
            self.last = None
            return None
        self.windows += 1
        onset = rxfront.answer_onset(seg, t0, t1)
        centre = self._centre(onset)
        floor, self.floor = self.floor, (level if self.floor is None
                                         else min(self.floor, level))
        before, self.last = self.last, (None if read or floor is None
                                        else (level - floor, centre))
        if read or not answered or floor is None \
                or self.windows <= ANSWER_FLOOR_WINDOWS:
            return None
        excess = level - floor
        if excess < ANSWER_OCCUPIED_DB:
            return None
        window = seg[int(t0 * FS):int(t1 * FS)]
        bw = rxfront.occupied_bw(window)
        # Width cannot say what raised the level and this can: whether both of
        # PACTOR-III's own channels stand over the comb. It grades nothing --
        # the verdict above and below is the same verdict it was -- and it is
        # the number that separates a gateway answering off PACTOR-1 from the
        # band filling up, which no line here could say before.
        comb = rxfront.p3_comb_db(window, offset_hz=offset_hz)
        # The slice is the slot plus one header length: the signature is a
        # 168.9 ms template and a burst starting at the slot's far edge has to
        # fit inside what is scored, or the score measures truncation.
        hi = min(seg.size, int((t1 + p4sig.HEADER_CHIPS / p4sig.CHIP_RATE) * FS))
        p4 = p4sig.spread_score(seg[int(t0 * FS):hi])
        named = (f" -- and it carries the 1800/16 chip-spreading signature of "
                 f"PACTOR-4 robust-mode signaling (r {p4:.2f}, knee "
                 f"{p4sig.SPREAD_KNEE:.2f})" if p4 >= p4sig.SPREAD_KNEE else "")
        stepped = onset is not None and onset.step_db >= ANSWER_ONSET_DB
        held = (before is not None and before[0] >= ANSWER_PERSIST_DB
                and centre is not None and before[1] is not None
                and abs(centre - before[1]) <= ANSWER_BAND_TOL_HZ)
        measured = (f"d {t0 * 1e3:.0f}-{t1 * 1e3:.0f} ms stands {excess:+.1f} dB "
                    f"over the quietest this session has heard that slot "
                    f"({floor:.1f} dB over {self.windows} windows), across "
                    f"{bw:.0f} Hz, PACTOR-III's own pair {comb:+.1f} dB over the "
                    f"channel comb (knee {rxfront.P3_COMB_KNEE_DB:.1f}) so "
                    f"{'off' if comb < rxfront.P3_COMB_KNEE_DB else 'on'}"
                    f"-carrier, "
                    + ("no onset in the window"
                       if onset is None else
                       f"onset {onset.step_db:+.1f} dB at the turnaround "
                       f"(knee {ANSWER_ONSET_DB:.1f})")
                    + (f", centred {centre:.0f} Hz" if centre is not None else ""))
        if not (stepped or held):
            return _SlotReading(
                f"ANSWER SLOT LEVEL -- {measured}. The channel got louder than "
                f"this station has known that slot be, and nothing STARTED in "
                f"it: no edge at the turnaround, and the cycle before was quiet "
                f"there. A CHANNEL READING and not an answer slot occupied -- "
                f"nothing is spent, nothing is held, and the link stays up",
                False)
        if peer_owes_a_packet:
            # THE OCCUPANT IS THE STATION WE ARE ANSWERING. An IRS holding a
            # PACTOR-3 link owes a codeword and is owed a data packet, so what
            # this window measures is not an answer slot with a stranger in it:
            # it is where the peer's own burst begins, and a burst too weak for
            # the CRC still stands over the floor. On WS8EOC's 80 m arm of
            # 2026-09-13 all seven prints fell on cycles a witness 160 miles
            # from the gateway decoded `SL2 DATA 23B` in.
            #
            # AND AN ISS WITH AN UNCONFIRMED CHANGEOVER IS IN THE SAME POSITION,
            # which the role alone cannot say. Between keying a changeover and
            # reading its answer this end holds the sending role while the peer
            # holds the channel -- it is still keying its own packet on its own
            # raster, which is the whole reason the changeover was sent. Asked
            # by role only, the 2320 arm printed `ANSWER SLOT OCCUPIED` at hold
            # 210 against the station it was answering.
            return _SlotReading(
                f"PEER'S PACKET UNREAD -- {measured}, and no reader took any of "
                f"it. This is where the PACTOR-3 station we are acknowledging "
                f"owes us its data packet{named}, and something is transmitting "
                f"there: attributed to the peer by the link's own role, unread "
                f"rather than unidentified. The cycle is held and the link "
                f"stays up", True)
        return _SlotReading(
            f"ANSWER SLOT OCCUPIED -- {measured}, and no reader took any of it. "
            f"Somebody is transmitting where our answer is due in something "
            f"that is not PACTOR-1{named}; "
            + ("it begins inside the window, at the turnaround"
               if stepped else
               "it was standing there in the cycle before, in the same place "
               "in the passband")
            + "; not attributed, not a codeword, and the link stays up", True)


def _report_answer_band(band: _AnswerBand, host, raster: "_MasterGrid", seg,
                        since_tx: int, d_max_n: int, *, read: bool) -> None:
    """Say what the slot carried, and hold the cycle where it carried a station.

    The two halves are one call because they must not come apart: a line an
    operator reads and a cycle the strand budget does not spend rest on the same
    finding, and the version of this that printed without telling `arq` is the
    version that hung up on WS8EOC while saying in the log that it could hear it.
    `note_unreadable_answer` is deliberately the only thing reached from here.

    AND ONLY AN OCCUPIED SLOT REACHES IT. A channel reading is printed and goes
    no further -- the operator gets the measurement, the cycle counts as silent,
    and the retry budget runs exactly as it would have on an empty channel. A
    level that moves nothing is the whole answer to 2026-09-04, where the two
    loudest windows of a session's own noise held cycles and then ended a link
    that four times over had granted us PACTOR-3.
    """
    sessrx = getattr(raster, "sessrx", None)
    reading = band.sight(seg, since_tx, d_max_n, read=read,
                         answered=raster.acquired,
                         peer_owes_a_packet=(host.protocol == Protocol.PACTOR3
                                             and host.arq.state in LINKED
                                             and (host.arq.role == IRS
                                                  or host.arq.unconfirmed_breakin)),
                         offset_hz=getattr(sessrx, "p3_receive_offset_hz", 0.0))
    if reading is None:
        return
    print(f"    !! {reading.line}", flush=True)
    if reading.occupied:
        host.arq.note_unreadable_answer()


class _ConnectTail:
    """Give late exact candidates time to corroborate, with a fixed ceiling.

    WS8EOC at 00:03 on September 13 first decoded in the final listening
    interval. Closing on the original deadline prevented further evidence.
    A candidate spends the whole ceiling. It does not itself authorize a
    connection.

    IT USED TO BUY THREE CYCLES FROM WHERE IT LANDED, and the ceiling was then
    an upper bound nothing ever reached: the WS8EOC call of 2026-09-16
    (`pactor-current-ws8eoc-80-sense-20260916T150542Z`) took its first candidate
    early enough that three cycles ran to 46.350 s against a ceiling of 50.080,
    and the session closed with three of the eight authorised cycles unspent
    while the peer was answering in searched windows 29 and 31. Corroboration is
    `_ConnectEvidence.N` accepts inside `SPAN` searched windows; cycles it never
    listens through are draws it never takes, and holding some back buys
    nothing. `MAX_CYCLES` remains the only bound, and the reason it is one: past
    it we owe the channel an identification.

    AND THE LATE CYCLES ARE THE ONES WORTH HAVING on this station as it stands.
    The tap analysis of 2026-09-16 (arm `...80-sense-20260916T145959Z`) measures
    the receive floor rising 7.6 dB broadband from the first sustained key-down
    until 0.64 s after our last transmission -- a comb at exactly 1000.000 Hz,
    the USB frame rate, with a 3.5-6 kHz hash pedestal, flat across all 22 gaps,
    the same on both jacks and only ~12 dB down through the rig's own T/R mute,
    so it is this station's RFI entering ahead of the mute. Every window after
    our first call is searched ~8 dB deafer than the pre-key floor, worst at the
    top of the passband, which is the obvious reading of a peer that became
    decodable only in searched window 29 of 34.
    """

    INITIAL_CYCLES = 2
    MAX_CYCLES = 8

    def __init__(self, start: int, slot_n: int):
        self.slot_n = slot_n
        self.end = start + self.INITIAL_CYCLES * slot_n
        self.limit = start + self.MAX_CYCLES * slot_n

    def candidate(self, now: int) -> bool:
        """Spend the rest of the ceiling; False once there is nothing left to spend.

        `now` is where the candidate landed, which the extension no longer turns
        on -- the ceiling is absolute and a later candidate cannot reach past it.
        """
        if self.limit <= self.end:
            return False
        self.end = self.limit
        return True


class _ConnectEvidence:
    """The candidate answers a call has collected, and whether they are a station.

    ONE ZERO-ERROR CODEWORD IS NOT AN ANSWER. The connect search reads twelve
    bits against two legal codewords in two shift senses, so four words of 4096
    match at every alignment it tries -- one accept in every 37 cycles of the band
    it now searches, which over a call of any length is a coin toss rather than a
    decode. The sessions of 2026-08-06 are what that costs: three calls, three
    gateways, one accept each, three links reported to stations that were never
    there, and an operator listening live who heard all three.

    What separates a station from the floor is not the strength of one accept but
    the AGREEMENT of several. A peer answers every cycle until the caller's first
    data packet decodes, and it answers at its own turnaround, which is a property
    of its hardware and does not move WITHIN a session -- across the seven WS8EOC
    sessions of 2026-07-30 its bursts start anywhere from 87 to 134 ms
    (`PEER_TURNAROUND_S`), but the tightest three inside any eight cycles of one
    session span 2.5 ms at their widest. Noise has no turnaround, so its accepts
    scatter across the whole band and each is independent of the last.

    So the rule is three accepts inside `SPAN` SEARCHED WINDOWS whose offsets
    agree to `TOL_S`, and every term in it is MEASURED over 31996 cycle-windows
    of real off-air energy addressed to nobody -- 11.3 hours from 1005
    recordings, all of rf-corpus and this station's own ARDOP captures, of which
    416 hold at least one accept:

      * THREE, and ANY three rather than the last three. The longest run of
        ADJACENT accepts agreeing to even 20 ms in that population is two, so the
        third is what the rule is bought with and adjacency buys nothing on top of
        it. Adjacency does cost: `onair-0730-2036` is a session a real gateway
        answered and its accepts fall on cycles 3, 5 and 7. So does taking the
        last three, which is the same mistake wearing a sleeve -- see `offer`.
      * INSIDE TWELVE SEARCHED WINDOWS, and the bound is what the rule stands
        on: three accepts agreeing anywhere in a whole recording fires on 34 of
        the 416, bounded to 12 windows it fires on 1, and bounded to 8 on none.

        WINDOWS AND NOT CYCLES, because a cycle we did not call in has no answer
        due and is never offered to the search. `note_search` counts the draws
        this rule is exposed to, and the negative population is 31996
        consecutive such draws. A hush costs six cycles in ten and the band
        falls off the front of the window from its second cycle on, so
        `onair-0911-2332` searched 13 of its 25 cycles: an eight-CYCLE bound
        spent four of the twelve windows it was priced for, and the rule ran at
        twice the strictness it was measured at.

        TWELVE AND NOT EIGHT, bought against a real peer rather than a
        preference. WS8EOC answered that call three times at 98, 100 and 99 ms
        -- a 2 ms group, against the 2.5 ms the seven 2026-07-30 sessions put on
        `TOL_S` -- in searched windows 5, 7 and 13, and the whole group was
        discarded while the operator listened to the gateway answering on a
        monitor receiver. Nine windows end to end is what this station's own
        4-call/6-hush rotation costs when a peer answers the last call of each
        run: three answers from three runs cannot land inside eight. The price
        is the one the line above already measures, one false link over the 416
        against none at eight, and it is paid in the currency the 2026-08-06
        phantoms were: 1005 recordings, one of which now fires.
      * TEN MILLISECONDS, which is half the grid's own capture range
        (`MAX_PULL_S`) and four times the widest real group. Twenty fires on 3 of
        the 416; ten fires on none.

    AND A LEVEL NOW STANDS IN FRONT OF IT. `ANSWER_CODEWORD_X` refuses an accept
    that does not stand over its own window, which takes 97.6% of this
    population's accepts off this rule's input and closes the one recording it
    still fired on. Everything priced here is the rule against the UNGATED
    search, which is the harder population and the one it has to keep standing
    on: a gate is a level, and a level is the first thing a night of poor
    conditions takes away.

    RE-PRICED WHEN THE SEARCH'S CEILING ROSE, because a band 19 ms wider is 19 ms
    more chance as well as 19 ms more peer. Over the same population re-swept --
    32239 windows from 1012 recordings -- the search accepts in 2.70% of them at
    the slot's reach and 3.60% at the read's, and this rule closes on ONE
    recording either way, the same one. So nothing here moved: the accepts the
    wider band adds are spread across recordings rather than stacked three deep
    inside twelve windows of one, which is the difference the rule exists to
    read. `tests/shrike/fixtures/connect-ceiling-negatives.json` is that sweep.

    Against the seven 2026-07-30 sessions this fires on the four that ran long
    enough to hold three answers, at cycles 7, 5, 4 and 5 -- two to four cycles
    later than the single accept did, which is two to four cycles of a call the
    peer was going to repeat anyway. It fires on none of the three sessions of
    2026-08-06, and on no eight-cycle span of the 31996 negative windows.

    THE SEARCH'S OWN COARSENESS IS THE REMAINING COST, and it is measured rather
    than left implied: at a 1.25 ms hop over a 10 ms bit, five of the 57 accepts
    those seven sessions hold fall between trial alignments, and `onair-0730-2037`
    never reaches three because of it. Halving the hop recovers them and takes the
    per-window false-accept rate from 2.7% to 4.9%, at which this rule fires on
    four of the negative recordings. Four fictional links is not a price worth
    paying for one real session recognised, so the hop stays.

    The project's working notes, not part of this distribution, re-derive all of it.
    """

    N, SPAN, TOL_S = 3, 12, 0.010

    def __init__(self) -> None:
        # cycle, offset, and which search of the session drew it
        self.candidates: list[tuple[int, float, int]] = []
        self.at: Optional[int] = None      # the cycle the evidence closed in
        self.searched = 0
        self.discarded = 0                 # accepts the level gate refused
        self.by_chance = 0.0

    def note_search(self, span: float) -> None:
        """One cycle offered to the search, across a band `span` seconds wide.

        Counted at the SEARCH and not at the accept, because the count exists to
        put the accepts in proportion: two candidates out of four cycles and two
        out of forty are different readings of the same channel, and the line
        used to print only the two. `by_chance` accumulates per cycle rather than
        being a rate times the total, because the band is not the same width
        every cycle -- `_acquisition_window` gives back what our own schedule and
        the capture leave, so a run of long cycles and a run of short ones have
        different nulls and neither is the constant somebody would otherwise
        write down here.

        AND IT CARRIES `SPAN`, which is a count of draws rather than of cycles:
        a call/hush rotation searches four or five cycles in ten, so a bound in
        cycles is a different rule on every schedule. See `offer`.
        """
        self.searched += 1
        self.by_chance += p1rx.acquire_null(span)

    def note_discard(self) -> None:
        """One accept the level gate refused. Counted, because the null above is
        the UNGATED rate and the operator is owed the difference."""
        self.discarded += 1

    def offer(self, cycle: int, d: float) -> bool:
        """Take one accepted codeword; True when the evidence now names a peer.

        ANY `N` of the live candidates, not the last `N`. Taking the last N is
        strictly stronger than the rule this class argues for, and it fails the
        way the rule is written not to: one chance accept landing between a peer's
        answers undoes them, however many of them agree.

        The seven real sessions do not show it, and that is luck rather than a
        property -- `onair-0730-2036` holds an accept at 74 ms among answers at
        100, 99, 98 and 95, but it arrives in cycle 20, twelve cycles after the
        rule has already closed. 23% of eight-cycle spans of occupied HF hold a
        spurious accept, and priced against the measured accept rate and the
        measured answer offsets, taking the last three recognises a peer later in
        3.7% of calls -- 5.9% when it answers every other cycle -- by three
        cycles. Those are cycles of a connect budget that is six.

        LIVE MEANS DRAWN WITHIN `SPAN` SEARCHES, counted off `note_search`,
        which the loop calls immediately in front of every search and nowhere
        else -- so a caller driving `offer` directly drives that too.
        """
        self.candidates.append((cycle, d, self.searched))
        live = sorted(x[1] for x in self.candidates
                      if self.searched - x[2] < self.SPAN)
        if any(live[i + self.N - 1] - live[i] <= self.TOL_S
               for i in range(len(live) - self.N + 1)):
            if self.at is None:
                self.at = cycle
            return True
        return False

    def closest_pair(self) -> Optional[float]:
        """The tightest two candidate offsets of the session, seconds.

        How near the evidence came, in the rule's own currency: `TOL_S` is a
        spread between accepts, so the best pair says whether a session missed
        by a millisecond or scattered across the whole band. None until two
        accepts exist -- one codeword has no spread, which is the whole of what
        is wrong with reading one as an answer.
        """
        ds = sorted(d for _, d, _ in self.candidates)
        return min((b - a for a, b in zip(ds, ds[1:])), default=None)

    def report(self) -> str:
        """What the search offered and what became of it. Never a bare count.

        "control signals decoded 1" is what the three phantom sessions printed,
        and it reads as a weak answer rather than as a coin landing heads. The
        candidates and their offsets say which it was: a station repeats itself
        at one turnaround, and chance does not.

        SO THE NULL IS PRINTED BESIDE THE COUNT, because a count alone still
        reads as evidence about the channel and here it is very nearly none.
        `p1rx.acquire_null` is the search's own arithmetic -- four words of 4096
        at every alignment, no energy test anywhere in it -- and an operator
        deciding at three in the morning whether to spend another arm on this
        frequency needs "2, and chance alone gives about 2" to be one glance
        rather than a calculation. It is not a verdict and it does not touch one:
        the corroboration rule below is what separates a station from the floor,
        and it fires on none of this.

        The figure is approximate -- see `acquire_null` -- so it is hedged where
        it is printed rather than dressed up in a decimal it has not earned.

        WHAT WAS SEARCHED, said rather than implied. This claimed "any cycle we
        called in", and the first cycle of a hush is searched too: a keyed
        cycle's window closes before its own key, so the band belonging to the
        last call of a run lands in the cycle after it. Over the record 7 of the
        74 candidates on file were accepted in one. `_acquisition_window` has the
        rest of it.
        """
        if not self.searched:
            return ("connect candidates: no cycle was searched -- no window held "
                    "the band an answer to our call was due in")
        cut = (f", {self.discarded} more read out of the band's own noise and "
               f"discarded" if self.discarded else "")
        tally = (f"{len(self.candidates) or 'none'} of {self.searched} cycles "
                 f"searched{cut}, chance alone gives about {self.by_chance:.1f} "
                 f"before the level gate")
        if not self.candidates:
            return (f"connect candidates: {tally} -- no zero-error codeword in "
                    f"the band an answer to our call was due in")
        where = ", ".join(f"cycle {c} at d = {d * 1e3:.0f} ms (search {w})"
                          for c, d, w in self.candidates)
        verdict = (f"corroborated in cycle {self.at}" if self.at is not None else
                   f"none corroborated -- an answer is {self.N} inside "
                   f"{self.SPAN} searched windows agreeing to "
                   f"{self.TOL_S * 1e3:.0f} ms")
        return f"connect candidates: {tally} ({where}) -- {verdict}"


class _TurnaroundEvidence:
    """The gaps a caller's bursts have implied, and whether one of them is a peer.

    ONE BURST IN THE BAND IS NOT A TURNAROUND, for `_ConnectEvidence`'s reason:
    what separates a station from the floor is not one reading but the AGREEMENT
    of several. A peer answers every cycle at its own turnaround, which is a
    property of its hardware and does not move within a session. An echo, a
    burst of QRM and another link's raster each land where they land, and the
    next cycle puts them somewhere else.

    `_acquire` used to take the FIRST onset of a cycle whose gap fell in the
    band, so ARRIVAL ORDER decided between candidates rather than evidence.
    Benched 2026-08-14 over all 899 (spurious 41-69 ms, real 90-120 ms) pairs
    with both bursts in every cycle: the spurious one won all 899, and the
    receive window then opened on an echo while the peer answered 40 ms away.
    `key_refusal` cannot see that either -- it measures phase against the same
    latched burst -- and three more cycles of the real peer answering did not
    undo it: once `d` is set the tracker only looks inside `MAX_PULL_S`, and an
    answer that far off is never offered to the search again.

    So the rule is TWO candidates from different cycles inside `SPAN` agreeing to
    `TOL_N`, AND NO SECOND GAP THE SAME COULD BE SAID OF. Corroboration answers
    "is anything there"; it does not by itself answer "which of these is
    answering us", and where the bench puts a recurring burst either side of the
    band it is the second question that decides the placement. Two witnessed
    turnarounds in one window is a thing a master cannot resolve from timing, so
    it says so and stays on the rotation. Every term is measured over the 106 rig
    sessions of 2026-08-13/14 -- 816 cycles that reached this search, 101 of
    which held a burst at all and 50 a burst inside the band, ending in 16
    acquisitions across 11 sessions:

      * TWO, not `_ConnectEvidence`'s three. That search reads twelve bits
        against four legal words and accepts on 2.7% of windows; this one is
        offered an in-band envelope onset in 6% of cycles and the two rates are
        not the same kind of number. Three corroborates 5 of the 16 acquisitions
        against two's 9, and the four it drops include WM4RB's +71.4 ms
        placement, which is a real peer.
      * INSIDE EIGHT CYCLES, the same bound. The widest corroborating pair in
        the corpus is 3 cycles apart and 4, 8 and 16 corroborate the same 9, so
        eight is what a peer answering every OTHER cycle needs with room over --
        KB5LZK on 2026-08-14 is that peer, answering in cycles 11 and 13 of
        `rig-session-20260814-005758` and in neither 12. Sixteen would buy one
        placement back, W6IDS's +58.6 ms, by letting a re-acquisition lean on
        the same session's earlier answers 13 cycles back. It is not taken:
        span is the term `_ConnectEvidence` measured a corroboration rule's
        false fires against -- unbounded 34 of 416, twelve cycles 1, eight 0 --
        and this search has no such measurement of its own to widen on.
      * `MAX_PULL_S`, because the tracker already calls any burst inside it the
        same station, and an agreement tighter than that would refuse what
        `_track` would then accept. The nine corroborating pairs in the corpus
        spread 0.8, 0.9, 1.0, 1.0, 2.0, 4.0, 4.4, 12.0 and 19.4 ms, so ten --
        the connect search's tolerance, bought against codeword alignments
        rather than an envelope detector's 5 ms grid -- drops the last two.
      * AND ONE WITNESSED GAP, which costs nothing measured and is the half of
        the rule the bench needs: with the spurious burst recurring at a fixed
        gap rather than scattering, two cycles of agreement corroborate BOTH
        bursts and sorted order hands the band back to whichever sits earlier --
        all 899 again, and the clamped aim back inside the real packet. No
        window of the corpus ever holds two witnessed gaps: 50 in-band bursts
        over 106 sessions, never two stations in one eight-cycle span.

    WHAT THIS IS NOT is a codeword witness. It reads gaps, so what it separates
    is a station from bursts that do not repeat, and a station alone from a band
    with two things in it. `D_MIN_S` still carries the near edge, where every
    spurious burst the corpus actually holds sits -- 28 of the 51 out-of-band
    gaps are under 40 ms and the other 23 are past 130.
    """

    N, SPAN, TOL_N = 2, 8, round(MAX_PULL_S * FS)

    def __init__(self) -> None:
        self.candidates: list[tuple[int, float]] = []   # cycle, gap in samples
        self.at: Optional[int] = None      # the cycle the evidence closed in
        self.witnessed = 0                 # gaps `N` cycles agreed on, last offer

    def offer(self, cycle: int, gaps: list[float]) -> Optional[float]:
        """Take one cycle's in-band gaps; return the corroborated one, or None.

        ANY `N` of the live candidates, for `_ConnectEvidence.offer`'s reason,
        and grouped by LINKAGE rather than by a fixed window: a peer `_track`
        walks a couple of milliseconds a cycle has to stay one gap here, or an
        eight-cycle span of it splits in two and reads as two stations.

        AND THIS CYCLE HAS TO BE IN IT. A placement hangs off a burst heard in
        the cycle it is aimed from, so a group made only of candidates the span
        has not yet dropped names a turnaround with nothing to key against --
        and one group ageing out from under a second would otherwise hand this
        cycle's unrelated burst a corroboration it never earned.
        """
        self.candidates += [(cycle, g) for g in gaps]
        live = sorted((g, c) for c, g in self.candidates if cycle - c < self.SPAN)
        groups: list[list[tuple[float, int]]] = []
        for x in live:
            if groups and x[0] - groups[-1][-1][0] <= self.TOL_N:
                groups[-1].append(x)
            else:
                groups.append([x])
        # DISTINCT CYCLES, because two bursts of one cycle are two readings of
        # one moment. A station repeating itself is the whole claim.
        agreed = [g for g in groups if len({c for _, c in g}) >= self.N]
        self.witnessed = len(agreed)
        if len(agreed) != 1:
            return None
        lo, hi = agreed[0][0][0], agreed[0][-1][0]
        if not any(lo <= g <= hi for g in gaps):
            return None
        if self.at is None:
            self.at = cycle
        return (lo + hi) / 2


def _edge_dev(seg: Optional[np.ndarray], seg_start: int, at: int) -> Optional[float]:
    """Sub-bit timing error, in samples, of the control signal starting at `at`.

    Positive means the burst sits LATE of `at`. `None` when there is no audio in
    hand or nothing edge-like inside the burst -- a missing measurement, which the
    loop treats as a measurement of nothing rather than as a zero.
    """
    if seg is None or not seg.size:
        return None
    dev = p1rx.cs_time_dev(seg, (at - seg_start) / FS)
    return None if dev is None else dev * FS


class _PeerEnd(NamedTuple):
    """Where the peer's packet ends, and the reading that says so.

    ONE RECORD BECAUSE THE INSTANT IS USELESS WITHOUT ITS ORIGIN. The
    changeover packet keys `BREAKIN_LEAD_S` past `at_slot`, and until
    2026-09-04 the log named that instant against `end` -- the reading's own
    packet ending, whole peer cycles behind it -- so eighteen keyings printed
    `+1322 ms past the peer's packet` for a burst that went out 72 ms past one.
    The tape settles which is which only if the line carries the onset the
    projection stands on and how many of the peer's cycles it was carried.
    """
    at: int                 # the decoded onset the projection stands on
    end: int                # that reading's own packet ending
    at_slot: int            # `end` carried onto the cycle the aimed slot names
    cycles: int             # cycles of the peer's between the two


class _P3ReplyTiming(NamedTuple):
    control_phase: int
    packet_phase: int
    packet_width: int
    cycle_n: int
    reverse_gap: int
    observed_cycle: int


class _P3TurnSnapshot(NamedTuple):
    emitted_phase: int
    anchor: int
    d_ref_n: Optional[int]
    d_n: Optional[float]
    peer: tuple[int, int, int, int]
    identity: tuple
    timing: _P3ReplyTiming
    controls: tuple[tuple[int, int], ...]


class _MasterGrid:
    """The caller's transmit cycle -- free-running -- and the receive window on it.

    A PACTOR master's transmit instants are a local 1.25 s grid, referenced to
    nothing the far end does: "Der SLAVE-Takt wird auf den MASTER-Takt
    synchronisiert (Auswertung der Flankenwechsel)", and a reference
    implementation applies its timing correction to the receive anchor
    unconditionally and to the transmit anchor only when it is NOT the master.
    See docs/protocols/pactor/pactor1-timing.md §1.

    This class used to do the opposite: it pulled the transmit anchor towards the
    peer's bursts every cycle, at a gain of 0.3. That is not a loud failure -- on
    the air it tracked to within +/-8 ms and a link did come up once -- which is
    exactly why it needs stating plainly. It imports the peer's jitter into our
    grid and feeds it back, and the station on the other end is a follower with
    about an eighth of a bit of authority per cycle: it can trim a static offset,
    slowly, and it cannot chase a moving grid. Two followers and no clock is not a
    link, and nothing on the air says so -- the peer's control signals go on
    sounding perfectly healthy while every packet fails.

    So there are two anchors here and they are not equal:

      * `anchor` -- our own data-bit instant. Advanced by exactly one cycle,
        every cycle, and moved by NOTHING once a control signal has been heard.
      * `d` -- the turnaround gap, which puts the receive window at
        `anchor + rx_ref_n + d`, 960 ms + d on the link it was measured on.
        Searched for until a second cycle agrees with it
        (`_TurnaroundEvidence`), then corrected from sub-bit edge statistics at
        `TIMING_GAIN` of the error per cycle, for good: `tx_next = tx_prev +
        1250 ms` regardless.

    The one thing that can move `anchor` is placing the grid in the first place,
    and it happens at most while cold and off the air. See `_place`.
    """

    MAX_MISSES = 3
    """Cycles with no control signal where the grid predicts one before `d` goes
    back to being searched for. THE TRANSMIT ANCHOR IS NOT TOUCHED -- only the
    receive window re-opens.

    A deliberate divergence, and it is a divergence: "neither station ever
    re-acquires timing after connect" (§4), and the reference implementation can
    afford that because it acquires on a zero-bit-error 12-bit correlation and a
    wrong `d` costs it a link rather than a session. Ours comes off an envelope
    detector on a 5 ms grid, which QRM satisfies, so acquiring wrong is a real
    outcome here and has to be recoverable."""

    BLIND_CYCLES = 4
    """Cycles hearing nothing before the session stops transmitting.

    Our own transmission is what is most likely to be covering the peer: on the
    one-slot cadence our RF plus the T/R recovery either side of it occupy about
    1.1 s of the 1.25 s cycle, so a grid placed at the wrong phase misses the
    peer's burst -- and, because both grids are stable to a fraction of a
    millisecond a cycle, misses it identically for the rest of the session. There
    is no drift to walk us out of it. Silence is the only move that changes the
    geometry, and for a master it is the ONLY move: it may not retime itself
    towards the peer.

    Four, because a station has to be given a call or two before its not
    answering means anything.

    AND ONLY WHILE NOTHING HAS EVER BEEN HEARD ON THIS GRID -- see `_blind`."""

    HUSH_CYCLES = 6
    """Cycles spent off the air once BLIND_CYCLES have gone by with nothing heard.

    Call, then stop and listen: the working record does this by hand, and the
    premise is measured. A gateway keeps calling for tens of its own cycles after
    the caller goes quiet -- two recordings, 77 s and 62 s, hold one doing exactly
    that, a provoked gateway and W6IDS, bursts 1.25 s apart with our transmitter
    off, and `rxfront.p1_burst_onsets` finds 31 and 38 of them. Six of our slots is
    7.5-15 s depending on the cadence, comfortably inside that, and short enough
    that a peer which simply is not there gets called again rather than
    abandoned."""

    def __init__(self, anchor: int, slot_n: int, offset_n: int, *,
                 packet_n: int, cs_n: int, d_max_n: int):
        self.anchor, self.slot_n, self.offset_n = anchor, slot_n, offset_n
        self.p1_data_n, self.p1_cs_n = packet_n, cs_n
        # Protocol geometry follows keying and an accepted mode/role change.
        # A late changeover must update it before rotation, even when no reply
        # in that protocol has keyed yet. `data_n` and `cs_n` read it.
        self.protocol = Protocol.PACTOR1
        # ...and how long a cycle of it is. NOT set at the key, unlike the
        # protocol: the cycle length decides the window in FRONT of the key and
        # the slot the key lands on, so the loop reads it off the FSM at the top
        # of the cycle instead. See `regear`.
        self.cycle_long = False
        self.sending = True          # ISS: our transmission is the data packet
        self.d_max_n = d_max_n       # the latest turnaround we could still hear
        self.d_n: Optional[float] = None     # turnaround gap; None = not acquired
        # ...and what it is a gap FROM. See `rx_ref_n`, which is the only thing
        # that reads it.
        self.d_ref_n: Optional[int] = None
        self.misses = 0
        self.blind = 0               # consecutive cycles with nothing heard
        self.hush_left = 0           # cycles still owed to listening in the clear
        self.cycles = 0              # updates folded, which is cycles collected
        # Whether a second cycle has agreed with the gap `d_n` came off. A
        # reading of one burst opens the receive window; only a corroborated one
        # is tracked rather than searched for again. See `_TurnaroundEvidence`.
        self.corroborated = False
        self.evidence = _TurnaroundEvidence()
        # ...and whether a turnaround was EVER measured on this grid. `d_n` goes
        # back to None after MAX_MISSES; this does not. See `_blind`.
        self.acquired = False
        # The last zero-error codeword read in the band an answer to our own
        # call was due in: (cycle, codeword index, offset in seconds). Evidence
        # that the peer is answering and nothing at all about where to aim --
        # see `update` and `_ConnectEvidence`, which are the two halves this
        # used to be one of.
        self.answered_word: Optional[tuple[int, int, float]] = None
        # Slot 0 is the call that fixed the shift, so `shift_slot` starts there
        # and `align` is the only thing that ever moves it.
        self.shift_slot = 0
        # What the session is doing, for the two lines that report what it
        # heard. Display only; see `_scheduler_reading`.
        self.reading = ""
        # The peer's last tracked burst onset, capture-stream samples: where
        # its packet ended, which is what `key_refusal` holds our carrier clear
        # of and what the `[ack]` line measures a keyed burst against. It lives
        # ONE cycle -- the first miss releases it, ahead of the window's
        # three-cycle grace -- because it is one reading of where one packet sat.
        self.peer_onset: Optional[int] = None
        self._p3_peer: Optional[tuple[int, int, int, int]] = None
        self._p3_peer_confirmed = False
        self._p3_peer_swap: Optional[bool] = None
        self._p3_peer_identity: Optional[tuple] = None
        self._p3_turn: Optional[_P3TurnSnapshot] = None
        self._p3_reply_phase_invalid = False
        # Has the peer agreed to the role this station is holding? A changeover
        # packet makes us the ISS the moment it is built, and the peer is still
        # transmitting on its own raster until it answers -- so a stint it never
        # agreed to is one no turnaround happened in, at either end, and the
        # return from it owes no rotation. `_grid_reversal` samples it every
        # cycle the role stands, because the state that carries the agreement is
        # gone by the time the peer's resumed stint has put the role back.
        self.turn_accepted = True
        self.reply_clock = None
        # The peer's last transmission of ANY kind on its own raster, decoded or
        # merely heard. `p3_control_refusal` ages the reply against this; the
        # boundary does not depend on a decode.
        self._p3_heard_at: Optional[int] = None
        # The peer's raster: the frame every later one has projected from, and
        # how many have. The reply comb rides this rather than each decode, so
        # the frame search's own quantum does not walk our transmit phase.
        self._p3_raster_origin: Optional[int] = None
        self._p3_raster_run = 0
        self._p3_command_slot: Optional[int] = None
        self._p3_command_row0: Optional[int] = None
        # Actual emitted control phase references, not requested slot boundaries.
        # A rolling CRC read can arrive after a newer control has already keyed.
        self._p3_controls: list[tuple[int, int]] = []
        self._p3_timing: Optional[_P3ReplyTiming] = None
        # Placement can project a previously observed raster for a bounded
        # number of cycles. Keep that origin separate from the one-cycle
        # collision observation, so a miss cannot resurrect an older codeword.
        self._peer_raster_at: Optional[int] = None
        # The peer's last DECODED codeword, which `key_refusal` projects on the
        # peer's own raster while we are the ISS. `peer_onset` above is an
        # energy onset and guards the other role; this is a word, and it is the
        # only thing allowed to refuse a transmission of our own.
        self.peer_cs: Optional[_PeerCodeword] = None
        # The slot our carrier last actually came up in, set where the burst is
        # scheduled (`RadioTx._tx`). The receive anchor is OUR OWN DATA END plus
        # `d`, so a slot we spent hushed or gave up has no anchor at all -- see
        # `rx_due_in`, which is the only thing that reads this.
        self.keyed_slot: Optional[int] = None
        self._p3_keyed_reply: Optional[tuple[int, int, int, int]] = None
        # The soonest turnaround this cycle's bursts implied, whatever became of
        # them. A record for the layer that has to say what the far end was doing
        # and cannot see the channel: see `arq.PactorArq.note_burst`.
        self.nearest_gap_n: Optional[int] = None
        # The three terms of `answer_position`: the fold the first entry packet
        # was keyed in, the answer position last measured before it, and one
        # (fold, sample, ms) per cycle since. Every one of them is the onset
        # `nearest_gap_n` is taken from, read from the boundary instead of from
        # our data end; nothing here is a second measurement.
        self.entry_at: Optional[int] = None
        self._entry_variant: Optional[str] = None
        self._first_entry_key_n: Optional[int] = None
        # Where our entry's last symbol falls past the boundary; the answer
        # position a reading peer is predicted at is taken off it. PACTOR-3's
        # by default, and a rung with another extent says so when it keys.
        self.entry_end_n = ENTRY_END_N
        self._entry_extent_override: Optional[int] = None
        # ...and the SAMPLE that fold's boundary sits on, which is what decides
        # which population a reading joins. The fold cannot: the window in front
        # of a key is read after it, so the cycle the entry is keyed in reports
        # an onset that arrived BEFORE the entry went out. On 2026-09-03 both
        # arms' first `ENTRY ANSWER` was the previous slot's codeword, 187 ms in
        # front of the entry's own boundary, and "19 of 19 at the PACTOR-1
        # position" carried one cycle that could not have read one.
        self.entry_key_n: Optional[int] = None
        self.answer_unread_ms: Optional[float] = None
        self.entry_answers: list[tuple[int, int, float]] = []
        # ...and the matched filter's instant for whichever of those cycles the
        # receiver tracked a PACTOR-III control in, by fold. An envelope finds
        # where the channel got louder and a codeword filter finds the word: on
        # KB5LZK's 40 m arm of 2026-09-15 the first reading moved 140 ms over
        # five cycles while the second held 902 ms to a millisecond. Only the
        # cycles that carry a word get one, so the series stays a measurement
        # rather than a quiet cycle borrowing the last one's answer.
        self.entry_tracked: dict[int, float] = {}
        # The receiver that took them. Set by the session loop where the grid is
        # built; without it the position falls back to the onset, which is what
        # every reading was before this.
        self.sessrx: Optional[_SessionRx] = None

    @property
    def data_n(self) -> int:
        """How long our own data packet is, in the protocol we are keying.

        PACTOR-1's 960 ms stood here whatever the link was in, and that is the
        length of the one waveform this station stops transmitting the moment it
        upgrades: `key_refusal` holds our carrier clear of the peer's data by it,
        `_keyable_slot` measures the cycle's listening floor from it, and
        `reverse`'s rotation is it less a codeword.

        NOT THE RECEIVE WINDOW, and that half was this property's own overreach.
        `rx_due` hung off `packet_n` and so followed this the moment the link
        upgraded, walking the window 150 ms earlier while the peer went on
        answering exactly where it had. `rx_ref_n` is where the receive side
        stands now, and the measurement is under it.

        MEASURED, `rf-corpus/PIII_Complete_1.wav`, the only real PACTOR-3 link in
        the corpus: over fifteen short cycles the IRS answers 889.4-891.9 ms
        after the packet's phase reference, and `placement.PACKET_S` accounts for
        810 of that. The turnaround left over is 80 ms -- beside the 100-105 ms
        the same station pair spends in PACTOR-1, and inside the band `_acquire`
        searches. The recording's long cycles answer at 3390 ms, which is that
        same geometry over 320 rows, and nothing here transmits one.
        `tests.shrike.test_p3_upgrade._window_arm` reads all fifteen back through
        the production reader at the production tolerance.

        THAT IS A SETTLED PACTOR-3 LINK and it is not where an upgrade starts.
        Both stations there had the entry packet behind them, so the whole cycle
        carries the protocol's own 810 + 80 + 210 + 150 geometry. A peer that has
        not yet acknowledged the entry packet still holds a PACTOR-1 grid and
        still answers where PACTOR-1 put it -- see `rx_ref_n`, which is measured
        on four such arms.

        OUR CARRIER COMES UP AHEAD OF THE PHASE REFERENCE, and by more than this
        length: the shaped pulse puts 40 ms of transmit filter either side of the
        eighty-one symbols and `_trim_silence` keeps what clears 2% of the peak,
        so the boundary leads the first symbol by 35.0 ms at speed level 1 -- the
        entry packet's own level -- and 22.5 to 27.5 ms at the others, and our
        data ends 35.0 ms later than this says. It is left out of the figure
        because it is a property of our own envelope and the trim threshold
        rather than of the protocol, and it costs the receive window nothing now
        that the window no longer hangs off here.

        AND THE RENDER KEYS ONE SYMBOL PAST IT. `placement.data_packet` appends
        `ENTRY_TRAILER` at every level, so what goes out is 82 symbols where
        `placement.PACKET_S` counts 81. THIS FIGURE DOES NOT FOLLOW IT, and that
        is deliberate: 890 ms is where a peer ANSWERS, measured against a station
        that already keys the trailer -- DL6MAA keys 82 and its IRS answers at
        889.4-891.9 -- so moving the reference to 820 would aim the window 10 ms
        past a measured instant to account for a symbol the measurement already
        contains. What the extra symbol belongs to is the KEYED EXTENT, and every
        place that needs one measures the render: `_tx` hands `key_refusal` the
        trimmed audio's own length, and `_keyable_slot`'s listen floor is taken
        from the capture position rather than from here. The one reading left
        short is `_peer_air`'s standing-in width as the IRS, by that same symbol.

        AND THE LENGTH IS THE CYCLE'S, not the protocol's. `P3_LONG_PACKET_N` is
        3.290 s of packet on the 3.75 s cycle, four times this one, and it stood
        here at 0.810 for either length -- so `key_refusal`'s fit test could
        never pass a long burst, `peer_end` put the peer's packet ending 2.5 s
        early and `rx_due` aimed the codeword read at 0.890 s where the answer
        was at 3.390. See `regear` and `cycle_n`.

        PACTOR-2 IS 10 MS SHORTER AT BOTH LENGTHS and answers 10 ms sooner:
        0.800 and 3.280 of packet against a codeword at 0.880 and 3.360
        (`pactor2.cs_slot`). Its header is a nine-pulse frame marker where
        PACTOR-3 spends nine on a phase reference and a header block, so the two
        combs are the same shape and one pulse apart -- and the difference lands
        on the turnaround rather than on the grid, because `d` is the same 80 ms
        either way. Not measured: no recording in the corpus carries a PACTOR-2
        answer slot at all, and `pactor2.TURNAROUND_S` is where that stands.
        """
        if self.protocol is Protocol.PACTOR2:
            return P2_LONG_PACKET_N if self.cycle_long else P2_PACKET_N
        if self.protocol is not Protocol.PACTOR3:
            return self.p1_data_n
        return P3_LONG_PACKET_N if self.cycle_long else P3_PACKET_N

    @property
    def ticks(self) -> int:
        """Grid slots to one ARQ cycle: `arq.LONG_TICKS` while long, else one.

        ONE RASTER SERVES BOTH LENGTHS. The reference's long packets sit 3.750 s
        apart on the same phase comb its short ones set -- three of these slots
        exactly -- so the cycle length changes without a resynchronisation and
        the comb is what a station holds through it. That is why this is a slot
        COUNT and not a second `slot_n`: `boundary`, `shift` and the tracker all
        go on counting the 1.25 s grid, and what changes is how many of its slots
        one turn of the link occupies.

        The shift alternates per ARQ cycle rather than per slot, and it still
        does: three is odd, so stepping the comb by `ticks` flips `shift`
        exactly once a cycle at either length.
        """
        return LONG_TICKS if self.cycle_long else 1

    @property
    def cycle_n(self) -> int:
        """One turn of the link, in samples: `slot_n` times `ticks`.

        The modulus everything about the FAR END is periodic in -- it answers
        once a cycle, not once a slot -- so `key_refusal` projects on this and
        `peer_end` steps back by it. `slot_n` remains the modulus of our own
        transmit comb.
        """
        return self.slot_n * self.ticks

    def next_slot(self, slot: int) -> int:
        """The comb position one ARQ cycle past `slot`."""
        return slot + self.ticks

    def regear(self, long: bool) -> Optional[str]:
        """Take the cycle length the FSM is on; returns a line, or None.

        READ AT THE TOP OF THE CYCLE, not at the key, and that is the whole
        difference from `keying`. The protocol is a record of what went out; the
        cycle length decides the window in front of the key, the slot the key
        lands on and how many ticks the FSM is owed -- so it has to be known
        before any of the three is computed. One bit, read once a cycle, in
        `update`'s `linked` shape: the grid still knows nothing about the ARQ.

        THE PEER ANSWERS AT A FIXED INSTANT IN THE CYCLE, so what the anchor
        moves by depends on which of the peer's two transmissions we are waiting
        for. Measured on all ten long cycles and all fifteen short ones of
        `rf-corpus/PIII_Complete_1`, `rxfront._best_cs` swept over the
        acquisition band from each packet's phase reference:

          * the answer to our PACKET moves with the CYCLE -- 889.4-891.9 ms on
            the short cycles, 3389.4-3392.5 on the long ones, 2500 apart to
            within a symbol. The packet itself grows only 2480
            (`P3_LONG_PACKET_N`), so a station that moved this anchor by its own
            packet's growth would aim 20 ms early, at the far edge of
            `MAX_PULL_S`. The 20 ms is the packet's, not the turnaround's: every
            recorded station keys one symbol more than we render (H3), and two
            of them is exactly it.
          * the peer's own PACKET, which is what an IRS waits for, does not move
            at all: from our codeword to the next packet's phase reference is
            358-359 ms at both lengths in the same recording.

        So the shift is the cycle's growth while we are the ISS and nothing
        while we are the IRS -- `sending` is the whole of the difference, the
        same way it is in `packet_n`.

        Before a turnaround is acquired there is no anchor to move and
        `rx_ref_n` falls through to `packet_n`, which follows `data_n`; the
        first answer heard replaces it.

        PACTOR-2 MOVES BY ITS PACKET, and the difference is one of measurement
        rather than of protocol. The 20 ms above is a reading of one real
        station: a PACTOR-3 IRS answers 2500 apart on a packet that grows 2480,
        and every recorded station keys one symbol more than we render. Nothing
        has ever read a PACTOR-2 answer slot at either length, so importing that
        20 ms here would put our receive window 20 ms away from the instant our
        own transmitter answers a peer at (`pactor2.cs_slot`) on no evidence at
        all. Held to `cs_slot`, the two agree: 0.880 s short and 3.360 long.
        """
        if bool(long) == self.cycle_long:
            return None
        self._clear_p3_reply_timing()
        was, was_data = self.cycle_n, self.data_n
        self.cycle_long = bool(long)
        if not self.sending:
            grew = 0
        elif self.protocol is Protocol.PACTOR2:
            grew = self.data_n - was_data
        else:
            grew = self.cycle_n - was
        if self.d_ref_n is not None:
            self.d_ref_n += grew
        return (f"cycle length -> {'LONG 3.75 s' if long else 'short 1.25 s'}: "
                f"{self.ticks} grid slot{'' if self.ticks == 1 else 's'} to the "
                f"cycle, {self.data_n / FS * 1e3:.0f} ms a packet, the receive "
                f"window {grew / FS * 1e3:+.0f} ms to "
                f"{(self.rx_ref_n + self.d) / FS * 1e3:.0f} ms past the key")

    @property
    def cs_n(self) -> int:
        """Nominal codeword length for cycle rotation and receive prediction.

        The P3 reference cycle rotates by 810 - 210 = 600 ms on a role change
        (PIII_Complete_1; pactor3.md §7). Its physical control waveform includes
        pulse shaping, a trailing repeat and a 5 ms carrier stagger; those are
        tracked by the transmitter's actual audio extent and pulse centers.
        They do not change this nominal cycle geometry. P2 has its own constant.
        """
        if self.protocol is Protocol.PACTOR2:
            return P2_CS_N
        return P3_CS_N if self.protocol is Protocol.PACTOR3 else self.p1_cs_n

    @property
    def packet_n(self) -> int:
        """How long our own transmission is: a data packet as ISS, a control
        signal as IRS. The receive window hangs off its end, so the two roles put
        it a whole rotation apart -- 840 ms in PACTOR-1, 600 in PACTOR-3 -- which
        is the changeover seen from the receiving side."""
        return self.data_n if self.sending else self.cs_n

    @property
    def rx_ref_n(self) -> int:
        """The transmission of ours `d` is a gap from. `packet_n` until pinned.

        `d` IS A READING OF THE RASTER, TAKEN THROUGH OUR OWN DATA END, and the
        two are the same number only for as long as our packet keeps its length.
        The peer holds a free-running cycle grid -- the same one this class holds,
        and for the same reason `_place` exists -- so its answer sits at a fixed
        instant in the cycle. It does not move when our packet gets shorter. So
        the instant is what is held here, and the gap is left on the transmission
        it was measured against.

        MEASURED, and against the peers' own transmissions rather than ours: the
        four upgraded arms of 2026-08-26, two stations, 40 m and 80 m,
        `captures/onair-0825-21{01,05,08,11}`. Each acquired a PACTOR-1 turnaround
        of 73.0, 77.0, 96.5 and 91.2 ms, upgraded, and keyed four
        150 ms-shorter PACTOR-3 packets. `rxfront.p1_burst_onsets` puts the peer's
        burst 1031-1043, 1030-1047, 1054-1066 and 1048-1101 ms after our slot
        boundary in the PACTOR-1 cycles either side of the upgrade -- and
        1033.9-1035.6, 1035.6, 1054.0 and 1051.7-1087.9 ms after it INSIDE the
        upgraded ones. The same instant, to a few milliseconds, across a packet
        that changed length by 150.

        What that costs when it is not carried is measured on the same audio.
        `p1rx.cs_anchored` at `boundary + P3_PACKET_N + d` -- the anchor 97a826b
        left -- reads nothing in 15 of the 16 post-upgrade cycles and, in the
        sixteenth, a two-error CS3: a changeover codeword, which would have
        reversed the grid on a phantom. At the carried anchor the same reader
        takes `pactor1.CS_59A` at ZERO bit errors in 13 of the 16, shift sense
        alternating cycle by cycle, and the three it misses are cycles no burst
        was detected in at all.

        NOT A CONTRADICTION OF `data_n`'S 0.890, and the difference between them
        is READING the entry packet. That figure is the gap from a PACTOR-3
        packet's PHASE REFERENCE to the answer, and the sixteen cycles of
        `PIII_Complete_1` it comes off include the entry packet's own: that peer
        answered at 5.5638 s to a phase reference at 4.673 -- 890.8 ms -- where
        its two PACTOR-1 answers, at 3.156 and 4.406 s, put the next one at
        5.656. It moved 92 ms, in one cycle, on the first packet it read. The four
        arms above never gave their peers one to read: each went on repeating the
        grant once a cycle, at zero bit errors, for as long as we keyed at it. So
        one rule with the same subject in both: the answer is a property of the
        raster, and the raster moves when the LINK changes protocol -- which is
        when the far end reads the entry packet, not when our transmitter changes
        waveform.

        WHAT IT COSTS IF THE PEER DOES FOLLOW is three cycles, and they are
        `MAX_MISSES`. The answer moves to the PACTOR-3 slot, the carried anchor
        misses it, `_miss` releases both the gap and this reference, and the
        search reopens against what we are now keying -- where the same answers
        sit 115 ms out (890 from the phase reference, which our boundary leads by
        35.0 ms, less the 810 ms packet), comfortably inside the 40-130 ms band
        `_acquire` searches. `tests.shrike.test_p3_upgrade._window_arm` reads
        that. Three cycles is what the release costs anywhere; it is not paid
        blind, because `_SessionRx.upgrade_scan` sweeps the whole window for the
        peer's PACTOR-3 packets throughout and that is how the corpus's own
        answer -- a changeover packet, not a bare codeword -- arrives.

        PINNED AT THE PROTOCOL CHANGE AND NOT ONLY AT THE FIRST ANSWER. A grant
        arrives before a turnaround has to have been measured: the anchored
        codeword reader runs at the nominal `d` and reads the peer whether or not
        the envelope detector ever offered `_acquire` a burst to corroborate, and
        it is a grant read there that upgrades the link at all. With nothing
        pinned the fall-through then hands the window straight back to `packet_n`
        for the whole upgraded stint -- the one failure this property exists
        against, arrived at by the one route it did not cover. So `keying` pins
        the transmission the window was standing on when the protocol changed,
        whichever way it changes and in either role: the peer's instant is no
        more ours to move at a fallback than at an upgrade, and nothing about
        either is PACTOR-3's.

        `d` IS NOT TOUCHED THERE, and that is the same hazard the paragraph above
        names from the other side. It is a gap from the pinned transmission, so
        moving it by the packet's 150 ms as well would count one change twice and
        aim 150 ms the other way -- at neither position.

        WS8EOC, 2026-09-11 22:14, is where the number comes from: ten entry
        packets at SL1 drew thirteen answers at 1052.3-1059.3 ms past our slot
        boundary, which is our 960 ms PACTOR-1 packet plus the 95-97 ms
        turnaround that session tracked, every one of them a `0x59A` grant read
        at anchor at zero bit errors. That grid had corroborated a turnaround in
        cycle 11 and so held its instant through the upgrade. The same arm with
        the corroboration missed aims at 810 + `d`, and 905 ms reads none of
        them: the answers are 147 ms away, seven times `MAX_PULL_S`.

        `reverse` still moves it, and must: a changeover is a role change both
        stations make, and the rotation is theirs as much as ours.
        """
        return self.packet_n if self.d_ref_n is None else self.d_ref_n

    def keying(self, protocol: Protocol,
               extent_n: Optional[int] = None, *,
               entry_pending: bool = False,
               entry_variant: Optional[str] = None) -> Optional[str]:
        """Record the waveform that went out; returns a line, or None.

        Set where the burst is scheduled (`RadioTx._tx`) because it is a record of
        what went on the air rather than of what the loop meant to key. `data_n`
        and `cs_n` are what read it. `rx_ref_n`, deliberately, does not follow
        it: a protocol change PINS that instead, at the transmission the receive
        window was already standing on.

        `extent_n` is the trimmed extent the transmitter keyed, and the entry
        position is measured against it rather than against the banked SL1
        template: an uninvited `SL3 pkt` runs 25.8 ms longer than `ENTRY_END_N`,
        which is past `MAX_PULL_S`, so a peer that read one was reported OFF
        BOTH POSITIONS.
        """
        override, self._entry_extent_override = self._entry_extent_override, None
        if protocol is Protocol.PACTOR3 and (entry_pending or self.entry_at is None):
            if self.entry_at is None or (entry_pending and
                                          entry_variant != self._entry_variant):
                self.entry_at = self.cycles
                self.entry_key_n = self.boundary(self.keyed_slot)
                if self._first_entry_key_n is None:
                    self._first_entry_key_n = self.entry_key_n
                self._entry_variant = entry_variant
                self.entry_answers.clear()
                self.entry_tracked.clear()
            if override is not None or extent_n is not None:
                self.entry_end_n = override if override is not None else extent_n
        if protocol is self.protocol:
            return None
        was, ref = self.data_n, self.packet_n
        self.protocol = protocol
        self._p3_peer_confirmed = False
        self._p3_reply_phase_invalid = False
        self._p3_heard_at = None
        self._p3_raster_origin = None
        self._p3_raster_run = 0
        self._p3_command_slot = self._p3_command_row0 = None
        self._clear_p3_reply_timing()
        # ...and the instant is pinned here when nothing has pinned it yet, so a
        # link that upgraded on a grant it read at anchor keeps the position it
        # read that grant at. See `rx_ref_n`.
        if self.d_ref_n is None:
            self.d_ref_n = ref
        return (f"keying {protocol} at {self.data_n / FS * 1e3:.0f} ms a "
                f"packet, {(self.data_n - was) / FS * 1e3:+.0f} ms on the last: "
                f"the receive window HOLDS ITS INSTANT at "
                f"{(self.rx_ref_n + self.d) / FS * 1e3:.0f} ms past the key, "
                f"because the peer answers on the cycle and not on our packet")

    def reverse(self, *, to_iss: bool) -> str:
        """Rotate the grid through a direction change. One cycle, both stations.

        The constant is the 1990 Level-1 description's, which places the answer to
        a changeover packet "eine CS-Laenge (0.12 sec) vor Ende des alten eigenen
        TX-Blocks" -- one control signal before the end of the station's own old
        packet. It goes on the RECEIVE anchor of the station becoming ISS and on
        the TRANSMIT anchor of the station becoming IRS, which are one instant
        seen from the two ends. IT IS DURATIONS AND NOT BIT COUNTS: every control
        signal is sent at 100 Bd whatever the packet's speed, so 960 - 120 = 840
        ms at either PACTOR-1 speed, where the 200 Bd packet's own bits would say
        900; and in PACTOR-3 the same rule reads 810 - 210 = 600 ms off `data_n`
        and `cs_n`. Measured off the air both ways and at both speeds: against
        WS8EOC on 7101.5 kHz, 2026-08-18, the peer's turnaround runs continuously
        through the reversal -- 92.4 ms either side of the first and 93.0 either
        side of the second, with 18 cycles at anchor at zero bit errors after it,
        all 200 Bd.

        Here the receive anchor is held as `rx_ref_n`, the transmission `d` was
        measured from, so the move on that side is to rotate it between the packet
        and the codeword -- which is what swapping `packet_n` used to do for it,
        back when the two were the same thing. A ROLE CHANGE IS NOT A PROTOCOL
        CHANGE: both stations make this one and both move, so the anchor follows
        it, where a change of our own waveform alone leaves the peer where it was.
        The transmit anchor is stored too, so the station yielding moves it
        explicitly -- and that is the term with teeth. Without it the new IRS keys
        a whole rotation early, which on any turnaround lands squarely inside the
        packet it has just been told to listen to, every cycle, deterministically.

        `d` is untouched, and that is not an omission. Solving the grid either side
        of the reversal maps d -> 170 - d + 2p, an involution, and each station's
        own gap comes out the same number it was: our packet ends and the peer
        answers d later, before and after.
        """
        self._p3_command_slot = self._p3_command_row0 = None
        if not to_iss:
            self._clear_p3_reply_timing()
            self._p3_reply_phase_invalid = False
        # Returning to ISS leaves our transmit phase unchanged. An unanswered
        # CS3 retry still needs the observed position of its original head.
        self.sending = to_iss
        rot = self.data_n - self.cs_n
        if self.d_ref_n is not None:
            self.d_ref_n += rot if to_iss else -rot
        # EVERY TURNAROUND THE PEER ACCEPTED, AND ONLY THOSE. The rotation is
        # not a thing the anchor carries once and keeps: a station's transmit
        # grid moves when IT becomes the IRS and stands when it becomes the ISS,
        # so the two ends move alternately and the PEER's comb moves across our
        # ISS stint. A comb that rotates on the first yield and holds through
        # every later one comes back a whole rotation behind, which is what
        # `captures/onair-0916-2253` recorded: WS8EOC's comb +848 ms across our
        # stint, ours +0, and every acknowledgement of the last 50 s keyed
        # 147 ms into the 960 ms packet it was answering. Thirty codewords at
        # zero bit errors and not one packet.
        #
        # An ISS role we only CLAIMED is the exception, and the one the latch
        # this replaces was really for. Until the peer answers our changeover it
        # is still the ISS on its own raster, and when it simply carries on
        # (`arq._resume_peer_stint`) the role comes back with no turnaround
        # having happened at either end. Rotating on that return leaves the comb
        # +1200 ms where +600 is owed: 0912-2349 keyed every ACK inside the
        # packet it was answering, from the second reversal on.
        moved = rot if to_iss or self.turn_accepted else 0
        if not to_iss:
            self.anchor += moved
            if self.reply_clock is not None:
                self.reply_clock.rotate(moved)
            self.turn_accepted = True
        unanswered = "" if moved else (" -- the changeover we keyed was never "
                                       "answered, so neither comb moved")
        return (f"GRID REVERSED -> {'ISS' if to_iss else 'IRS'}: "
                f"{'receive' if to_iss else 'transmit'} anchor "
                f"{moved / FS * 1e3:+.0f} ms{unanswered}; we now send "
                f"{self.packet_n / FS * 1e3:.0f} ms a cycle")

    @property
    def locked(self) -> bool:
        """Is the RECEIVE window acquired? The transmit grid is never anything else."""
        return self.d_n is not None

    def boundary(self, slot: int) -> int:
        """Where the transmission aimed at `slot` keys. The free rotation, and
        there is no other: a PACTOR-1 ISS reads its control signal twelve bit
        periods from one latched instant with no timing search, and against a
        Winlink RMS we are always the caller, so that instant is its own packet
        end plus `slot_n - data_n - cs_n - d`. Which is this, for every `d`.
        See `tests.shrike.test_ackplace`.
        """
        return self.anchor + slot * self.slot_n

    def shift(self, slot: int) -> bool:
        """Is the FSK shift inverted in `slot`? Whole cycles since the call.

        "Die Shiftlage der FSK-Aussendung wird einmalig beim Verbindungsaufbau
        fixiert. Mit jedem neuen Paket oder Kontrollsignal wird die Shiftlage
        invertiert" -- and the reference implementation's per-cycle epilogue
        inverts BOTH senses unconditionally, whether or not that station
        transmitted or received anything in that cycle
        (docs/protocols/pactor/pactor1-data-packets.md §7). So it is counted in
        SLOTS: the never-transmit-late guard skips some, and a skipped slot
        advances the shift exactly like any other.

        Counted in slots rather than in samples since the call, which is what
        this did. A slot number survives the two things that move the anchor
        under it: `_place` correcting the grid's phase by up to half a cycle
        after a hush, and `reverse` sliding it a rotation per changeover, which
        accumulate. Under a sample count either one rotates the phase into
        whatever the floor division lands on, silently, mid-link.
        """
        return bool((slot - self.shift_slot) & 1)

    def align(self, slot: int, sense: int) -> Optional[str]:
        """Adopt the shift the PEER used in `slot`. Returns a line, or None.

        Within one cycle the two directions share the shift -- the sense a
        station sends its control signal in is the sense it expects the caller's
        packet in, in that same cycle (§7). So the peer's control signal is a
        direct reading of the phase it is counting on, and it is the only one
        available: which of our call bursts it locked to, and therefore where its
        own count began, is not observable from this end.

        A coherent alternating call train has the same extrapolated phase
        regardless of which call the peer acquired. Thus delayed acquisition
        alone does not establish a phase error. KI0BK 2026-09-09 has opposite
        call/reply senses on the nominal same-cycle grid; the P1 setup experiment
        in `_align_shift` tests whether adopting that reply phase caused its
        first-block stall. The remote receiver's expected phase is unobserved.
        """
        if self.shift(slot) == bool(sense):
            return None
        self.shift_slot ^= 1
        return (f"SHIFT REALIGNED to the peer: slot {slot} is "
                f"{'inverted' if sense else 'not inverted'}, and our next "
                f"transmission follows it")

    def rx_slot(self, at: int) -> int:
        """Which slot the control signal at stream sample `at` answers.

        `rx_due` places it a packet plus `d` past that slot's boundary; the grid
        is periodic, so the slot comes back by division. Rounded, because `at` is
        where a burst was heard and the grid tolerates it wandering tens of
        milliseconds inside a 1.25 s slot.
        """
        return round((at - self.rx_due(0)) / self.slot_n)

    def boundary_after(self, at: int) -> int:
        """The first boundary at or after sample `at`, whatever slot that is."""
        return self.anchor + -(-(at - self.anchor) // self.slot_n) * self.slot_n

    def note_peer_codeword(self, at: int, width: int, name: str,
                           who: Optional[str], *,
                           protocol: Optional[Protocol] = None) -> None:
        """Record the peer's last decoded codeword for the sending-side guard."""
        self.peer_cs = _PeerCodeword(at, width, name, who, self.cycles, protocol)
        if (self.protocol == Protocol.PACTOR3 and protocol == Protocol.PACTOR3
                and width == P3_CS_N and name in ("ACK", "REQ", "SPEED-UP", "NAK", "CYCLE-TOG")
                and self._p3_peer is not None
                and at >= self._p3_peer[0] + self._p3_peer[1]):
            # The peer now sent a bare control after its last CRC packet. That
            # changes the duration/role evidence: the old packet/control pair
            # cannot place a reclaim against this different transmission.
            self._clear_p3_reply_timing()

    def key_refusal(self, carrier: int, air_n: int, *,
                    changeover: bool = False) -> Optional[str]:
        """Why the carrier must NOT come up at `carrier`, or None.

        ONE QUESTION IN BOTH ROLES: does the air we are about to occupy --
        `carrier` to carrier-down, `air_n` samples of it -- overlap the air the
        peer is about to occupy, projected one cycle on the raster? Nothing
        else. Checked modulo the cycle, because the burst in hand may be any
        number of slots old.

        AND THE CHANGEOVER PACKET ASKS HALF OF IT. Taking the link IS keying
        into the slot the peer would have transmitted in: the reference modems'
        break-in begins 71-72 ms after our packet ends, on a 1.25 s cycle where
        our next packet is due 290 ms after it, so their 960 ms lies across 742
        of ours -- and does not collide, because a station that reads the CS3
        head in its own answer window cancels that transmission. The fit against
        the NEXT projection is therefore not a question to ask of this burst.
        What remains is: is the peer still transmitting the packet we are
        answering? That refuses, and nothing overrides it.

        ASKED OF THE CARRIER, NOT OF THE BOUNDARY, and the two are a settle
        apart. A caller passing its boundary is asking about an instant its own
        transmitter reaches 40 ms later.

        THE ROLES DIFFER IN THEIR EVIDENCE, NOT IN THEIR RULE, and `_peer_air`
        is where that difference lives: as ISS a twelve-bit word read at zero
        errors, as IRS an energy onset and the packet a station of our own
        protocol sends under it.

        NO COURTESY MARGIN ON EITHER SIDE, and the margin is the whole cost.
        The condition is overlap, so the band is exactly `_d_max_n` -- 1250 -
        960 - 120 - 40 = 130 ms at a 40 ms settle -- and no turnaround this grid
        was willing to acquire can be refused here. The IRS side carried a 20 ms
        `ACK_GUARD_S` until 2026-08-28, which closed its band at d <= 110: over
        the 753 acknowledgements this station has keyed as IRS across 57 sessions
        that refuses nine, and eight of them are every IRS cycle of the WM4RB
        link of 2026-08-14, which was answering us at 121.7 ms. What the margin was bought against
        (0d9eb95, "a bad onset estimate can now cost a cycle instead of somebody
        else's over") is real and is paid for elsewhere now: a bad onset is
        wrong by far more than 20 ms and is caught by the fit below, and by the
        corroboration, the `D_MIN_S` floor and the one-cycle life that decide
        what `peer_onset` may be at all.
        """
        air = self._peer_air()
        if air is None:
            return None
        at, width, whose = air
        if changeover and self._bare_peer_width(at) is None:
            # A CODEWORD WE READ IS THE HEAD OF A PACKET here, and that is the
            # whole difference: the station we are taking the link from is the
            # ISS and sends 960 ms, of which `peer_cs` names the first 120. Our
            # own answer window reads that head; the peer's reads ours.
            width = max(width, self.data_n)
        cycle_n = (self._p3_peer[2] if self.protocol == Protocol.PACTOR3
                   and not self.sending and self._p3_peer is not None
                   and at == self._p3_peer[0] else self.cycle_n)
        phase = (carrier - at) % cycle_n
        # FIT, not merely start clear: our whole keying has to be down before the
        # peer's next burst, or its tail lies across the head of that one.
        #
        # ON THE CYCLE AND NOT THE SLOT, because a station transmits once a cycle
        # and that is what a projection projects. Asked against `slot_n` a long
        # burst could not fit by construction -- 3.41 s of keying into 1.25 --
        # so every cycle with anything in `_peer_air` was refused three times
        # and then keyed anyway by the stand-down, over the peer. The two are
        # the same number on the short cycle, which is every arm flown.
        if width <= phase and (changeover or phase + air_n <= cycle_n):
            return None
        where = (f" at sample {at}, repeated on its own "
                 f"{cycle_n / FS:.3f} s raster,")
        if phase < width:
            return (f"{whose}{where} is still on the air "
                    f"{(width - phase) / FS * 1e3:.0f} ms after our carrier "
                    f"comes up")
        return (f"{whose}{where} begins "
                f"{(phase + air_n - cycle_n) / FS * 1e3:.0f} ms before our "
                f"{air_n / FS:.3f} s transmission would end")

    def _peer_air(self) -> Optional[tuple[int, int, str]]:
        """The peer's next transmission -- onset, width, and what says so.

        THE SENDING SIDE HAD NO GUARD AT ALL until cb534dd. On 2026-08-26 this
        station transmitted over WS8EOC's control signal twenty times in one
        session -- measured off a KiwiSDR at Empire, Michigan to +/-5 ms, with
        both stations on one clock -- while `key_refusal` returned on its first
        line, because we were the ISS for every keyed cycle of that grant. Our
        own recording structurally cannot hold the overlap: the rig mutes the
        receiver while we key, and `[collide]` printed zero times.

        A DECODED WORD, PROJECTED ON THE PEER'S OWN RASTER. That witness put the
        gateway on 1249.99 +/- 0.44 ms across twenty-five consecutive control
        signals with none missing, which is what makes a one-cycle projection
        worth resting a refusal on -- and it is still a projection, which is what
        `GUARD_MAX_DROPS` and `ISS_GUARD_CYCLES` are for.

        AND THE WORD HAS TO SURVIVE ITS OWN EVIDENCE, which `_forecast_next_key`
        settles before anything reaches here: a codeword we read is a codeword we
        did not transmit over.

        THE RECEIVING SIDE'S IS WEAKER AND SAYS SO. An energy onset identifies
        no station, and `data_n` is what a packet of our own protocol runs rather
        than a length anything measured. It is bounded the other way instead:
        `peer_onset` lives one cycle, only a burst inside the acquisition band
        becomes it, and `_track` will not move it further than `MAX_PULL_S`.
        """
        if self.protocol == Protocol.PACTOR3 and self._p3_peer_confirmed:
            cs = self._peer_codeword_for_geometry()
            packet = self._p3_peer
            packet_fresh = packet is not None and self.cycles - packet[3] <= 1
            cs_fresh = cs is not None and self.cycles - cs.cycle <= ISS_GUARD_CYCLES
            if packet_fresh and (not cs_fresh or packet[0] >= cs.at):
                return packet[0], packet[1], "the CRC-valid P3 packet we decoded"
            if cs_fresh:
                width = self._bare_peer_width(cs.at) or self.data_n
                return cs.at, width, f"the {cs.name} we decoded"
            return None
        if not self.sending:
            if self.protocol == Protocol.PACTOR3 and self._p3_peer is not None \
                    and self.cycles - self._p3_peer[3] <= 1:
                at, width, _, _ = self._p3_peer
                return at, width, "the CRC-valid P3 packet we decoded"
            if self.peer_onset is None:
                return None
            bare = self._bare_peer_width(self.peer_onset)
            if bare is not None:
                return (self.peer_onset, bare,
                        f"the {self.peer_cs.name} we decoded")
            return (self.peer_onset, self.data_n,
                    f"the {self.data_n / FS * 1e3:.0f} ms packet we heard")
        cs = self.peer_cs
        if cs is None or self.cycles - cs.cycle > ISS_GUARD_CYCLES:
            return None
        return (cs.at, cs.width,
                f"the {cs.name} we decoded from {cs.who}" if cs.who else
                f"the {cs.name} we decoded")

    def note_p3_packet(self, at: int, width: int, cycle_n: int, *,
                       swapped: Optional[bool] = None,
                       identity: Optional[tuple] = None) -> None:
        """Carry a decoded packet's sample position separately from FSK energy."""
        if self._p3_peer is not None and at <= self._p3_peer[0]:
            return
        self._p3_peer_confirmed = True
        self._p3_command_slot = self._p3_command_row0 = None
        self._observe_p3_reply_timing(at, width, cycle_n)
        if self._peer_raster_position(at, cycle_n) is None:
            self._p3_raster_origin, self._p3_raster_run = at, 1
        else:
            self._p3_raster_run += 1
        # THE LATCH DESCRIBES THE GEOMETRY OF AN OLDER READING. The guard
        # re-decides against this frame at the key, and a latch that outlives its
        # evidence turned one refused cycle into a dead link: 0912-2349 printed
        # forty `remains invalid` with a CRC-valid frame in hand every cycle.
        self._p3_reply_phase_invalid = False
        self._p3_peer = (at, width, cycle_n, self.cycles)
        self._p3_peer_swap = swapped
        self._p3_peer_identity = identity
        self.peer_onset = at
        self._peer_raster_at = self._p3_heard_at = at

    def _peer_raster_position(self, at: int, cycle_n: int) -> Optional[int]:
        """`at` snapped to the peer's established raster, or None if it is off it.

        The peer holds a free-running cycle grid -- 1249.99 +/- 0.44 ms over
        twenty-five witnessed transmissions, and 0.048 ms per cycle of fitted
        drift over 33 cycles off air -- while our own frame search quantises
        each reading to about 1.25 ms and can step further on a re-lock. So the
        raster is the origin projected, and a single frame's coordinate is a
        reading OF it rather than a new one.
        """
        origin, peer = self._p3_raster_origin, self._p3_peer
        if origin is None or peer is None or peer[2] != cycle_n:
            return None
        periods = round((at - origin) / cycle_n)
        projected = origin + periods * cycle_n
        if periods < 1 or abs(at - projected) > round(MAX_PULL_S * FS):
            return None
        return projected

    @property
    def _p3_raster_corroborated(self) -> bool:
        """Three CRC frames in a row on one comb: a raster, not one reading."""
        return self._p3_raster_run >= 3

    def _clear_p3_reply_timing(self) -> None:
        self._p3_controls.clear()
        self._p3_timing = None
        self._p3_turn = None

    def remember_p3_turn(self, emitted_phase: int) -> None:
        """Snapshot a credible IRS clock only after its outgoing CS3 emitted."""
        if self.sending:
            return  # A retry must not overwrite the original IRS snapshot.
        timing = self._p3_reply_timing()
        peer = self._p3_peer
        identity = self._p3_peer_identity
        self._p3_turn = None
        if (timing is None or peer is None or identity is None
                or not identity[0] or self._p3_reply_phase_invalid
                or not 0 < emitted_phase - peer[0] <= ONSET_MAX_CYCLES * peer[2]):
            return
        self._p3_turn = _P3TurnSnapshot(
            emitted_phase, self.anchor, self.d_ref_n, self.d_n, peer,
            identity, timing, tuple(self._p3_controls))

    def recover_p3_turn(self, phase: int, identity: tuple) -> bool:
        """Recognize the unchanged peer stint; never fit a new transmit phase."""
        saved = self._p3_turn
        peer = self._p3_peer
        if (saved is None or self.protocol != Protocol.PACTOR3
                or not self.sending or peer is None or phase != peer[0]
                or identity != saved.identity
                or peer[1:3] != saved.peer[1:3]
                or peer[2] != self.cycle_n
                or not saved.emitted_phase < phase
                or not 0 < phase - saved.peer[0] <= TURN_RECOVERY_CYCLES * peer[2]
                or not 0 < phase - saved.timing.packet_phase <= TURN_RECOVERY_CYCLES * peer[2]):
            return False
        elapsed = phase - saved.peer[0]
        periods = round(elapsed / peer[2])
        if (periods < 1
                or abs(elapsed - periods * peer[2]) > round(MAX_PULL_S * FS)):
            return False
        self.anchor, self.d_ref_n, self.d_n = saved.anchor, saved.d_ref_n, saved.d_n
        self.sending = False
        # The role is back where the saved comb belongs, and this recovery is
        # the proof the changeover was never accepted -- so the next claim
        # starts clean rather than carrying this one's refusal.
        self.turn_accepted = True
        self._p3_controls[:] = saved.controls
        self._p3_timing = saved.timing
        self._p3_turn = None
        self._p3_reply_phase_invalid = False
        self._p3_command_slot = self._p3_command_row0 = None
        print("    [grid] peer repeated its old P3 stint after our unconfirmed "
              "CS3; restored the prior IRS reply clock", flush=True)
        return True

    def p3_control_refusal(self, slot: int) -> Optional[str]:
        """A quiet channel does not establish a valid P3 reply clock."""
        if self.protocol != Protocol.PACTOR3 or self.sending or not self._p3_peer_confirmed:
            return None
        if self._p3_reply_phase_invalid:
            return "the IRS reply phase overlapped a decoded peer packet and remains invalid"
        peer = self._p3_peer
        if peer is None or peer[2] != self.cycle_n:
            return "no fresh packet clock supports this PACTOR-3 reply"
        # THE BOUNDARY DOES NOT DEPEND ON THE DECODE -- it is the comb, held on
        # the peer's own raster -- so what ages here is the evidence that the
        # peer is still transmitting. An undecoded burst on that raster is that
        # evidence (`note_peer_bursts`), and a raster three CRC frames have
        # agreed on carries further than one reading of it: 0913-0014 stopped
        # answering after five `no fresh packet clock` with the peer still keying.
        heard = max(peer[0], self._p3_heard_at or 0)
        limit = (RASTER_PROJECT_CYCLES if self._p3_raster_corroborated
                 else ONSET_MAX_CYCLES) * peer[2]
        if not 0 < self.boundary(slot) - heard <= limit:
            return "no fresh packet clock supports this PACTOR-3 reply"
        return None

    def p3_reply_shift(self, slot: int, *, target: Optional[int] = None) -> Optional[str]:
        """Put the reply comb back on the decoded packet. Returns a line, or None.

        The default policy places the answer a fixed delay past the packet.
        The opt-in timing trial can supply an audio-start epoch derived from
        our emitted entry instead. `update` returns early for a PACTOR-3 IRS,
        so `_place`, `_acquire` and `_track` are all unreachable for the whole
        stint: without this the anchor is whatever the connect train left plus
        `reverse`'s rotation, and a comb wrong by any amount stays wrong with a
        CRC-valid frame in hand every cycle.

        ON THE COMB'S OWN PERIOD, `slot_n`, and not on the peer's cycle. The
        boundary steps by a slot, so reducing the error modulo anything longer
        moves every other boundary to correct this one. Which slot the reply is
        keyed in is `_keyable_slot`'s question and stays there.

        AND ON THE RASTER RATHER THAN ON THE READING (`_peer_raster_position`):
        the comb would otherwise carry every step of our own frame search into
        our transmit phase, a cycle at a time.

        THE LONG CYCLE NEEDS NO TERM OF ITS OWN AND `P3_LONG_REPLY_S` SUPPLIES
        NONE. 3.390 - 0.890 is 2.500 s, which is exactly two slots, so the two
        branches below reduce to the same anchor and the constant is a statement
        of where the long answer sits rather than an instruction to move there.
        What actually puts the codeword 3.390 s past the packet is the SLOT:
        `regear` makes `ticks` three and `_regear_next_slot` steps the comb by
        that, so the reply keys two slots further on with the anchor unmoved.
        Naming both is the point -- a reader asking where the long answer goes
        finds the number here -- and a geometry in which the difference stopped
        being a whole number of slots would need this line to fire.
        """
        peer = self._p3_peer
        if (self.protocol != Protocol.PACTOR3 or self.sending or peer is None
                or peer[2] != self.cycle_n
                or self.cycles - peer[3] > ONSET_MAX_CYCLES):
            return None
        at = self._peer_raster_position(peer[0], peer[2]) or peer[0]
        due = at + round((P3_LONG_REPLY_S if self.cycle_long
                          else P3_REPLY_S) * FS)
        if target is not None:
            due = target
        err = (self.boundary(slot) - due) % self.slot_n
        if err > self.slot_n // 2:
            err -= self.slot_n
        if not err:
            return None
        self.anchor -= err
        if target is not None:
            policy = "P3 MAIL" if self.reply_clock is not None else "P3 TRIAL B"
            return (f"{policy} reply moved {-err / FS * 1e3:+.3f} ms "
                    f"onto the emitted entry/turn clock (audio epoch {target})")
        return (f"P3 REPLY COMB re-placed {-err / FS * 1e3:+.1f} ms onto the "
                f"peer's raster (CRC packet @ {peer[0]}, on the raster at "
                f"{at}, answered {(due - at) / FS * 1e3:.0f} ms past its "
                f"phase reference)")

    def note_p3_control(self, phase: int) -> None:
        """Record an emitted IRS control's leading phase-reference sample.

        P1's measured turnaround survives entry for reading repeated grants.
        It cannot also describe P3's different packet/control geometry. A P3
        measurement needs its own transmitted reference, on the same sample
        clock as a subsequent CRC frame, rather than the boundary we intended.
        """
        if self.protocol != Protocol.PACTOR3 or self.sending:
            return
        if self._p3_controls and phase <= self._p3_controls[-1][0]:
            return
        self._p3_controls.append((phase, self.cycle_n))
        del self._p3_controls[:-ONSET_MAX_CYCLES]

    def _observe_p3_reply_timing(self, at: int, width: int, cycle_n: int) -> None:
        """Measure RX phase after actual TX; two distinct cycles corroborate it.

        The independent SCS exchange in pactor3.md §17.1 has different reply
        phases in its two directions: CS3 replaces the existing codeword slot.
        No universal 70 ms gap, and no old P1 turnaround, specifies that slot.
        Measure the reverse gap from an emitted P3 control instead. Whole-cycle
        projection is explicit and bounded, so a late decode can use the older
        control that preceded it. Nothing is fitted to the desired break-in.

        MEASURING THE PAIR BELONGS TO THE IRS; CARRYING IT DOES NOT. A new
        reverse gap still needs an emitted control of our own, and an ISS emits
        none -- but the changeover an ISS has to place wants the same pair, and
        with the measurement gated on the role it died `ONSET_MAX_CYCLES` cycles
        after the reversal every time. MEASURED, 0913-2143: the last
        corroboration at hold 102, `breakin_refusal`'s `is None` branch from
        hold 111 on, nine times, while the peer kept a CRC-valid packet on its
        own raster in every one of those cycles. So `_reanchor_p3_reply_timing`
        carries it.
        """
        if self.protocol != Protocol.PACTOR3:
            return
        if self.sending:
            self._reanchor_p3_reply_timing(at, cycle_n)
            return
        if cycle_n != self.cycle_n:
            self._p3_timing = None
            return
        refs = [(phase, period) for phase, period in self._p3_controls
                if period == cycle_n and 0 < at - phase <= ONSET_MAX_CYCLES * cycle_n]
        previous = self._p3_peer
        if not refs or previous is None:
            self._p3_timing = None
            return
        control, _ = refs[-1]
        after_control = (at - control) % cycle_n
        gap = after_control - self.cs_n
        room = cycle_n - width - self.cs_n
        minimum = round(D_MIN_S * FS)
        if not minimum <= gap <= room - minimum:
            self._p3_timing = None
            return
        elapsed = at - previous[0]
        periods = round(elapsed / cycle_n)
        # The established frame tracker allows this phase bracket. The actual
        # placement remains bounded separately by BREAKIN_CLAMP_TOL_S.
        tolerance = round(MAX_PULL_S * FS)
        # HOW FAR THE PEER'S OWN COMB REACHES, which is the question
        # `p3_control_refusal` already answers and this asked differently. What
        # `periods` corroborates is that this packet lies a whole number of the
        # peer's cycles from the last one; three CRC frames agreeing on one comb
        # make that a measured raster rather than a single reading, and
        # `RASTER_PROJECT_CYCLES` is how far such a raster carries. Bounded by
        # ONSET_MAX_CYCLES here instead, the FIRST frame back from a deaf stretch
        # can never corroborate a pair -- and that is the cycle the changeover
        # wants one. 0913-2050: the peer's greeting resumed 14 cycles after the
        # last decode, on the raster to 2 ms, and the reply position was refused
        # anyway until the goodbye ran out of clock. The 20 ms agreement below
        # binds whatever the reach is.
        reach = (RASTER_PROJECT_CYCLES if self._p3_raster_corroborated
                 else ONSET_MAX_CYCLES)
        agrees = (previous[2] == cycle_n
                  and 1 <= periods <= reach
                  and abs(elapsed - periods * cycle_n) <= tolerance)
        if not agrees:
            self._p3_timing = None
            return
        first = self._p3_timing is None
        # Control phase, packet phase, width, cycle, measured turnaround, age.
        self._p3_timing = _P3ReplyTiming(control, at, width, cycle_n, gap, self.cycles)
        if first:
            print(f"    [grid] P3 reply timing corroborated: emitted control "
                  f"phase {control}, CRC packet phase {at}; reverse gap "
                  f"{gap / FS * 1e3:.2f} ms, forward gap "
                  f"{(room - gap) / FS * 1e3:.2f} ms. "
                  "Timing evidence; peer acceptance is not inferred.", flush=True)

    def _reanchor_p3_reply_timing(self, at: int, cycle_n: int) -> None:
        """Move the IRS-measured pair onto this frame, while we hold the link.

        WHAT THE ROLE CHANGES AND WHAT IT DOES NOT. `control_phase`,
        `reverse_gap`, `packet_width` and `cycle_n` are the link's geometry --
        where the peer answers a control, how long its packet is, what raster
        both ends run on -- and a reversal does not touch any of them. What goes
        stale is only WHEN the pair was read, and that is exactly what a fresh
        CRC-valid peer packet supplies. So the age moves and the measurement
        does not; nothing here fits a new gap, which still needs a control of
        our own and still belongs to the IRS.

        ON THE CORROBORATED RASTER AND NO FURTHER. This is the projection
        `p3_control_refusal` and the IRS branch above already bound: three CRC
        frames agreeing on one comb make a raster, `RASTER_PROJECT_CYCLES` is
        how far one carries, and a frame off that comb by more than the tracker
        calls the same transmission re-anchors nothing. A peer that stops
        sending therefore ages out on `ONSET_MAX_CYCLES` as before.
        """
        measured, peer = self._p3_timing, self._p3_peer
        if (measured is None or peer is None or not self._p3_raster_corroborated
                or cycle_n != self.cycle_n or measured.cycle_n != cycle_n
                or peer[2] != cycle_n):
            return
        elapsed = at - measured.packet_phase
        periods = round(elapsed / cycle_n)
        if not 1 <= periods <= RASTER_PROJECT_CYCLES \
                or abs(elapsed - periods * cycle_n) > round(MAX_PULL_S * FS):
            return
        self._p3_timing = measured._replace(packet_phase=at,
                                            observed_cycle=self.cycles)

    def _p3_reply_timing(self) -> Optional[_P3ReplyTiming]:
        measured = self._p3_timing
        if (self.protocol == Protocol.PACTOR3
                and measured is not None and measured.cycle_n == self.cycle_n
                and self._p3_peer is not None
                and 0 <= self._p3_peer[0] - measured.packet_phase <= ONSET_MAX_CYCLES * measured.cycle_n
                and self.cycles - measured.observed_cycle <= ONSET_MAX_CYCLES):
            return measured
        return None

    @property
    def d(self) -> int:
        """The turnaround gap: how long after our carrier drops the peer answers.

        Measured once acquired; the nominal until then.
        """
        return int(round(self.d_n if self.d_n is not None else D_NOMINAL_S * FS))

    def rx_due(self, slot: int) -> int:
        """Where this cycle's control signal is expected: `rx_ref_n`, plus `d`."""
        return self.boundary(slot) + self.rx_ref_n + self.d

    @property
    def peer_at(self) -> Optional[int]:
        """Where a transmission of the peer's began, as a sample we READ.

        The decoded codeword or the tracked onset, whichever is the more recent
        reading of the same raster -- neither is derived from our own anchor,
        which is the whole point of them here.
        """
        cs = self._peer_codeword_for_geometry()
        if self.protocol == Protocol.PACTOR3 and self._p3_peer_confirmed:
            return max((at for at in (None if self._p3_peer is None else self._p3_peer[0],
                                      None if cs is None else cs.at)
                        if at is not None), default=None)
        seen = [at for at in (self._peer_raster_at, self.peer_onset,
                              None if cs is None else cs.at)
                if at is not None]
        return max(seen, default=None)

    def _peer_codeword_for_geometry(self) -> Optional["_PeerCodeword"]:
        """Confirmed P3 geometry consumes decoded P3 words, never FSK aliases."""
        cs = self.peer_cs
        if (self.protocol == Protocol.PACTOR3 and self._p3_peer_confirmed
                and cs is not None and cs.protocol != Protocol.PACTOR3):
            return None
        return cs

    def _bare_peer_width(self, at: int) -> Optional[int]:
        """Duration supported by a recent bare codeword on this peer raster.

        During a lost turn change both hosts can believe they are IRS. A CS2
        still lasts 120 ms in P1 (210 ms in P3); assigning it a data packet's
        duration refuses the answer and places recovery 840/600 ms late.
        CS3 is a packet head and must retain the full-packet guard. Energy
        alone, stale decodes, and onsets off the decoded raster cannot shorten
        that guard, including bursts truncated by fading or interference.
        """
        cs = self._peer_codeword_for_geometry()
        if (cs is None or self.cycles - cs.cycle > ISS_GUARD_CYCLES
                or cs.name not in ("CS1/ack", "CS2/ack", "CS4/100Bd",
                                   "ACK", "REQ", "SPEED-UP", "NAK", "CYCLE-TOG")):
            return None
        tolerance = round(MAX_PULL_S * FS)
        age = at - cs.at
        if not -tolerance <= age <= ISS_GUARD_CYCLES * self.cycle_n + tolerance:
            return None
        phase = age % self.cycle_n
        if min(phase, self.cycle_n - phase) > tolerance:
            return None
        return cs.width

    @property
    def peer_read_gap(self) -> int:
        """Where the peer reads a codeword of ours, past the end of its packet.

        The cycle closing on itself: the peer's packet, our turnaround, our own
        120 ms, and the turnaround back. `170 ms - d` in PACTOR-1 on the short
        cycle, which is our own free-running boundary seen from the peer's
        transmission instead of from our comb -- and the comb is what two
        reversals can leave 840 ms along, which is why this is computed and not
        read off `boundary`.

        MEASURED AT KB5LZK, four arms and twelve codewords: acknowledged at
        88.5-98.0 ms past its packet end, median 94.9, every one of them keyed
        at `rot` 0.0 to +1.9 ms. `M + d` comes to 162.6-173.8 over those arms,
        so the budget is the shape of the answer and `d` is what makes it a
        number: the grid tracks it to 0.1-0.4 ms within a session and it differs
        by 7 ms between gateways.

        `cs_n` AND NOT `packet_n`, because the changeover packet's head IS a
        codeword and the peer reads it in the codeword's slot. `packet_n` is
        960 ms the moment we turn ISS, which is one cycle into the very sequence
        this places.
        """
        measured = self._p3_reply_timing()
        if measured is not None:
            return (measured.cycle_n - measured.packet_width - self.cs_n
                    - measured.reverse_gap)
        return self.cycle_n - self.data_n - self.cs_n - self.d

    def peer_packet_end(self, slot: int) -> Optional[_PeerEnd]:
        """Where the peer's last read transmission ends, projected on ITS raster.

        NOT WALKED BACK FROM `rx_due`, and that is the 2026-09-03 finding.
        `rx_due` hangs off our own anchor, and `reverse` moves that anchor by
        `data_n - cs_n` on the way to IRS while the way back moves only
        `d_ref_n` -- correctly, because each rotation is an instant seen from
        one end. A grid through BOTH reversals therefore carries the 840 ms on
        its boundary comb, which is -410 ms on a 1.25 s cycle, and the derived
        packet ending lands 550 ms INSIDE the packet it names. On KB5LZK that
        cycle, seven changeover packets keyed 656-667 ms into the peer's own
        transmission and not one was answered.

        A READING SURVIVES EVERY ROTATION. The peer holds a free-running cycle
        grid -- 1249.99 +/- 0.44 ms over twenty-five consecutive transmissions,
        witness-measured -- so its own onset projected forward on `cycle_n` is
        the instant, whatever our anchor has been through since.

        A recent bare codeword on this raster supplies its decoded duration.
        Otherwise our protocol's full packet length stands in for theirs,
        including CS3 packet heads and bursts identified only by energy.

        `slot` ONLY CHOOSES WHICH CYCLE, and choosing it late costs the cycle
        and nothing else: an instant already gone is refused by `RadioTx._tx`
        rather than keyed at our comb, and the burst is placed again on the next
        reading.
        """
        at = self.peer_at
        if at is None:
            return None
        measured = self._p3_reply_timing()
        if (measured is not None and self._p3_peer is not None
                and self._bare_peer_width(at) is None):
            phase, width, period, seen = self._p3_peer
            if (period == measured.cycle_n and width == measured.packet_width
                    and self.cycles - seen <= ONSET_MAX_CYCLES):
                end = phase + width
                cycles = (self.boundary(slot) - end) // period
                return _PeerEnd(phase, end, end + cycles * period, cycles)
        end = at + (self._bare_peer_width(at) or self.data_n)
        cycles = (self.boundary(slot) - end) // self.cycle_n
        return _PeerEnd(at, end, end + cycles * self.cycle_n, cycles)

    def peer_phase(self, at: int) -> int:
        """`at` against the PEER's own raster, signed. Not against our comb."""
        ph = (at - self.peer_at) % self.cycle_n
        return ph - self.cycle_n if ph > self.cycle_n // 2 else ph

    def note_peer_bursts(self, bursts: list[tuple[int, int]]) -> Optional[str]:
        """Re-origin the peer's raster on a burst we HEARD and could not read.

        THE ONE READING THAT SURVIVES THE ROLE WE ARE IN. Everything else that
        moves `peer_at` needs a decode or a turnaround: `note_peer_codeword`
        wants a codeword off the reader, and `_track` measures against `d_n` and
        our own anchor. Holding the link we key 960 ms of every 1.25 s cycle, so
        the peer's transmissions fall under our carrier and neither fires --
        on 2026-09-04's three KB5LZK arms one onset then placed ten consecutive
        changeover packets while the burst timer was printing 118-130 ms runs in
        the gaps and the reply test was calling them `p1reply`.

        SO IT IS MEASURED AGAINST THE PROJECTION AND NOT AGAINST `d`. The peer's
        raster is `peer_at` plus whole cycles, which is the same instant
        whatever our anchor has been through, and a burst on it is that raster
        re-read. Nothing here claims WHOSE burst it is -- `p1_bursts` is an
        energy shape and fires on VARA and on PACTOR-2 as well -- so the two
        conditions carry the whole of it:

          * ON THE RASTER, inside `MAX_PULL_S`. Twenty milliseconds is what the
            tracker already calls the same transmission, so an accepted burst
            moves the origin by at most that much in a cycle and a burst that is
            somebody else's is off the comb by far more than a placement can
            spend. It cannot walk: each acceptance is bounded against the
            projection the last one left.
          * AND LONG ENOUGH TO BE A TRANSMISSION, `ONSET_MIN_MS`. The detector's
            own floor is 60 ms, which admits the tail of a fade.

        Returns a line where it moved, and None where nothing qualified.
        """
        if self.peer_at is None:
            return None
        wide = round(ONSET_MIN_MS * FS / 1e3)
        near = [(a, w) for a, w in bursts
                if w >= wide and abs(self.peer_phase(a)) <= round(MAX_PULL_S * FS)]
        if not near:
            return None
        at, width = min(near, key=lambda b: abs(self.peer_phase(b[0])))
        moved = self.peer_phase(at)
        self.peer_onset = self._p3_heard_at = at
        return (f"the peer's raster re-read on a burst nothing decoded: "
                f"{width / FS * 1e3:.0f} ms at sample {at}, "
                f"{moved / FS * 1e3:+.1f} ms on its own comb")

    def breakin_refusal(self, end: _PeerEnd) -> Optional[str]:
        """Why the changeover cannot be placed against `end`, or None.

        A PLACEMENT THAT CANNOT BE MADE IS REFUSED, NEVER APPROXIMATED, and
        2026-09-04's evening arms are what that sentence cost. 35 changeover
        packets over three KB5LZK arms: 18 keyed at +94 to +98 ms past the
        peer's packet end, inside the window its packet counter advances across,
        and 17 at +70 -- the instant eighteen keyings refuted the same day.
        Every one of the 17 followed the line `receive window released after 3
        cycles`, in all three arms, and the arithmetic is not the projection:
        `peer_read_gap` is `170 ms - d`, `d` falls back to `D_NOMINAL_S`'s 105 ms
        the moment the turnaround is released, and 1250 - 960 - 120 - 105 is
        65 ms -- under the settle and reserve the transmitter owes, so the floor
        takes over and prints +70 whatever the peer is doing. A released
        turnaround is this grid saying it no longer knows where the peer reads,
        and a placement is exactly the thing that cannot be made without it.

        AND THE READING MAY NOT OUTLIVE ITS EVIDENCE -- `ONSET_MAX_CYCLES`.
        That bound refuses nothing on these arms that the turnaround does not
        refuse first, and it is not there for them: it stops one onset placing
        bursts indefinitely through a deaf stretch the turnaround happens to
        survive, which arm 4 came within a cycle of at ages 7 and 8.

        `RadioTx._tx` spends no retry on it (`arq.REFUSED`), and the cycle it
        gives up is the one that re-acquires: off the air the listen window is
        the whole 1.25 s instead of the 240 ms a keyed cycle leaves, which is
        where the peer's transmission is heard again.
        """
        measured = self._p3_reply_timing()
        if self.protocol == Protocol.PACTOR3 and self._p3_peer_confirmed and measured is None:
            return ("the PACTOR-3 reply position has no fresh corroborated "
                    "packet/control phase pair; the retained PACTOR-1 "
                    "turnaround cannot place this changeover")
        if not self.locked and measured is None:
            return (f"the turnaround was released, so the read instant falls "
                    f"back to a nominal {D_NOMINAL_S * 1e3:.0f} ms `d` and "
                    f"{self.peer_read_gap / FS * 1e3:.0f} ms past the peer's "
                    f"packet -- under the transmitter's own floor, which is the "
                    f"instant 2026-09-04 refuted")
        if end.cycles > ONSET_MAX_CYCLES:
            return (f"the peer's packet ending is projected {end.cycles} cycles "
                    f"({end.cycles * self.cycle_n / FS:.1f} s) from an onset "
                    f"read at sample {end.at}, past the {ONSET_MAX_CYCLES} this "
                    f"station has ever placed a burst a peer read from")
        return None

    def rx_due_in(self, lo: int, hi: int) -> Optional[int]:
        """The receive instant this window holds, if it holds one whole.

        `rx_due` names one slot's, and the window the loop has just collected
        belongs to whichever slot it last transmitted in -- which is not `slot`
        minus one, because the never-transmit-late guard skips slots. So it is
        `keyed_slot`'s instant or none, and a slot we never keyed is the second
        case: an answer to a packet that did not go out is not a thing the
        channel can hold.

        THE GRID IS PERIODIC AND THAT USED TO BE ENOUGH. The instant was found
        from the period -- the first one at or after `lo`, whatever slot it
        belonged to -- which is the same answer whenever the window opens just
        behind our own carrier, and a different one whenever it does not. A
        hushed cycle collects to the boundary rather than to the key, so its
        window runs about 2 s and opens BEFORE the instant of a slot spent off
        the air; the walk then aims the anchored reader at where an answer to a
        transmission we never made would have been, and `cs_anchored` reads
        whatever twelve bits are there.

        MEASURED, 2026-08-14, both sessions that reversed on a decode nothing
        was answering. rig-session-20260814-211608 read `rx_due(5)` 1.06 s into
        the 2.045 s captures/onair-0814-2116/hold_01, and -212300 read
        `rx_due(7)` 1.05 s into the 1.192 s onair-0814-2123/hold_01; both of
        those slots print `HUSHED, not keying`, and both reads came out at zero
        bit errors -- CS3 against a peer whose turnaround measured 81-88 ms,
        CS4 against one at 87-134. The 1.05 and 1.06 are OFFSETS INTO THE
        SEGMENT and were never turnarounds. That is why they agree across two
        gateways: a hushed window opens on the slot boundary rather than after
        our carrier, so the instant sits a whole 0.96 s packet plus the 105 ms
        nominal `d` into it -- 1.065 s, for any peer, on any night. A keyed
        cycle's window opens where the carrier dropped and the same instant
        lands at `d`, which is the 0.09 the answered cycles print.

        Swept at 1 ms over those same captures, `cs_anchored` returns a
        zero-error codeword in 1.62% and 2.71% of reads (216 of 13323, 367 of
        13549) and in 0.07% of the empty-channel audio N3HYM-10 and W9OTR
        collected the same night (19 of 28221). One read a cycle at an instant
        our own transmission defines is what that rate is affordable at, and it
        is the whole of what pays for it. The session captures are not empty --
        a 1 ms sweep walks over the peer's own bursts in them -- so the reader's
        own figure is the empty-channel one, which searching
        `p1rx.CS_SEARCH_HALF_S` rather than reading at a point moved from 0.05%.

        The legitimate reversal of that night survives unchanged:
        rig-session-20260814-210924 read `rx_due(49)` 0.09 s into
        onair-0814-2109/hold_06, and slot 49 is where TX[26] keyed.

        WHAT IT COSTS is the cycle after any slot we did not fill -- a hush, a
        slot the never-transmit-late guard skipped, an acknowledgement
        `key_refusal` dropped. That cycle's codeword goes unread, and the peer
        repeats it: a station that got no answer asks again, which is the whole
        of the reverse channel. The window keeps tracking through it either way,
        because `update` folds the cycle's onsets whether or not anything was
        read at the anchor.

        WHOLE MEANS THE PEER'S CODEWORD AND NOT OURS. `cs_n` is the length of what
        WE key, and an upgraded link is exactly where the two part company: until
        the far end has acknowledged the entry packet it is still answering in
        PACTOR-1, and on the raster instant it answers at a 210 ms PACTOR-3
        codeword could not finish before our own next key anyway. Measured over
        the sixteen post-upgrade cycles of 2026-08-26: the window runs from our
        emission end to 48 ms before the boundary, 333 ms, and the answer sits
        168 ms before its end -- room for the 120 ms word that was there, 42 ms
        short of the one we were keying. Asked for ours, this refused every one of
        them and the anchored read never ran.
        """
        if self.keyed_slot is None:
            return None
        at = self.rx_due(self.keyed_slot)
        return at if lo <= at and at + self.p1_cs_n <= hi else None

    def _gap(self, at: int) -> int:
        """The turnaround this burst implies: samples from our own data ending.

        Modulo the cycle, because the grid is periodic and a burst names the same
        gap whichever slot it was heard in.

        On `rx_ref_n` rather than on `packet_n`, so that a gap offered to the
        search, the gap the tracker corrects and the gap `rx_due` aims at are all
        one measurement. The band `_acquire` tests against is then still the band
        the turnaround was acquired in, which is where an upgraded link's peer
        goes on answering.
        """
        return (at - self.anchor - self.rx_ref_n) % self.slot_n

    def _signed_gap(self, at: int) -> int:
        """`_gap` about the cycle rather than around it: a burst 55 ms in FRONT of
        our data ending reads -55 and not 1195.

        The modulus is right for the band test -- an answer is a gap, and a
        negative one is not a gap -- and wrong for the line that reports the miss.
        WS8EOC's answer instant walked 133 ms earlier across the session of
        2026-08-26 until it arrived before our own audio ended, and the log then
        read `1 burst(s) heard, nearest at 1212 ms` on a codeword that was 38 ms
        EARLY and clean. A near miss on the near edge was printed as the far edge
        of the cycle, every cycle, for as long as the walk lasted.
        """
        gap = self._gap(at)
        return gap - self.slot_n if gap > self.slot_n // 2 else gap

    def _err(self, at: int) -> float:
        """Signed samples from where the grid predicts a control signal to `at`."""
        err = (self._gap(at) - self.d_n) % self.slot_n
        return err - self.slot_n if err > self.slot_n / 2 else err

    def _phase(self, at: int) -> int:
        """Signed samples between where we would key and where this burst says to."""
        err = (at + self.offset_n - self.anchor) % self.slot_n
        return err - self.slot_n if err > self.slot_n // 2 else err

    # -- one cycle's worth of evidence ------------------------------------
    def update(self, onsets: list[int], seg: Optional[np.ndarray] = None,
               seg_start: int = 0, *, hushed: bool = False,
               since_tx: int = 0, linked: bool = False,
               answered: Optional[tuple[int, float]] = None,
               reading: str = "") -> Optional[str]:
        """Fold one cycle's bursts into the grid; returns a line for the log.

        `seg` is the audio those onsets were found in, so the sub-bit edge
        statistic can be taken where the burst actually is. Without it the loop
        still runs on the detector's 5 ms grid alone, a decimated version of the
        same measurement.

        `since_tx` is how far this window's first sample sits past our own
        carrier dropping -- `_acquisition_window`'s argument, asked here for the
        opposite answer. That window searches for an answer wherever one could
        be; `_place` may run only where one could NOT, and a hush does not put
        our carrier a cycle back until its second cycle.

        `linked` is whether the session holds a link this cycle. The grid has no
        business knowing about the ARQ and does not: it is one bit, read once a
        cycle. A link ends setup listening and forbids rephasing the TX grid.

        `answered` is the codeword this cycle's connect search read at zero bit
        errors in the band an answer to our call was due in, `(index, d)`, or
        None. It is the OTHER half of `_TurnaroundEvidence`'s question and it is
        kept apart from it on purpose: this one says the peer is answering, that
        one says where to aim, and only the second may move a grid.

        `reading` is the same arrangement and the same discipline: a string,
        carried once a cycle, printed by the two diagnostics and read by
        nothing. See `_scheduler_reading`.
        """
        self.reading = reading
        self.cycles += 1
        if answered is not None:
            # A ZERO-ERROR CODEWORD IN OUR OWN ANSWER BAND REFUTES THE HUSH.
            # `_blind` goes quiet on two premises together -- our carrier may be
            # covering the peer's burst, and nothing on this grid says anything
            # is there -- and twelve bits decoded at zero errors where an answer
            # to our own call was due refutes both, whatever the onset detector
            # made of the same audio. WS8EOC answered `onair-0911-2332` in cycle
            # 5 and the session went off the air over it in the same cycle,
            # three times over, on a grid the peer was answering.
            #
            # It buys keeping the air, and nothing else: `_place` is reached
            # only from a hush, and the corroboration that aims a receive window
            # is still three of these. See `_ConnectEvidence`.
            self.answered_word = (self.cycles, *answered)
            self.blind, self.hush_left = 0, 0
        if linked:
            # A decoder can connect while a hush is already in progress.
            # KB5LZK and N5TW did so on 2026-09-09; placing the grid afterward
            # moved their first DATA by -30/+11 ms relative to our call train.
            # Keep that train's timing and resume transmitting on its next slot.
            self.hush_left = 0
        if self.protocol == Protocol.PACTOR3 and not self.sending:
            self.nearest_gap_n = None
            if self._p3_peer is not None and self.cycles - self._p3_peer[3] <= 1:
                self.peer_onset = self._p3_peer[0]
                return "P3 receive position from CRC-valid frame"
            self.peer_onset = None
            return "P3 receive position unconfirmed this cycle"
        # SIGNED, like the line `_acquire` prints from the same measurement. Its
        # one consumer is `arq.note_burst`, which asks whether the answer
        # arrived where a receiver could have it, and a burst 38 ms in front of
        # our own audio ending reported as 1212 ms answers that question with the
        # wrong end of the cycle. A negative gap is the one reading that says our
        # carrier was over the peer.
        near = min(onsets, key=lambda a: abs(self._signed_gap(a)), default=None)
        self.nearest_gap_n = None if near is None else self._signed_gap(near)
        # ...and the same onset from the BOUNDARY, which is where the entry
        # packet's two answer positions are separated. `rx_ref_n + nearest_gap_n`
        # is this number decomposed -- it is what the `keying` line prints -- and
        # the undecomposed form is what survives the reference being released,
        # which is exactly what a peer moving 105 ms costs the tracker.
        #
        # Not on a hush, which keys nothing for anything to be an answer to. The
        # PRE-ENTRY reading additionally waits for a MEASURED turnaround, because
        # until there is one a burst in the window is not yet the peer's. That
        # used to read `d_ref_n`, which said the same thing until `keying` began
        # pinning one with no measurement behind it.
        if near is not None and not hushed:
            at_ms = (near - self.anchor) % self.slot_n / FS * 1e3
            # ON THE SAMPLE AND NOT ON THE FOLD (`entry_key_n`): a burst that
            # arrived before the entry was keyed is a reading of the position
            # BEFORE it, whichever cycle the loop got round to noting it in.
            if self.entry_key_n is not None and near >= self.entry_key_n:
                if self.sessrx is not None:
                    tracked = self.sessrx.tracked_answer_position_ms(
                        near - (near - self.anchor) % self.slot_n, self.slot_n)
                    if tracked is not None:
                        self.entry_tracked[self.cycles] = tracked
                self.entry_answers.append((self.cycles, near, at_ms))
            elif (self.d_n is not None or self.acquired) and (
                    self._first_entry_key_n is None
                    or near < self._first_entry_key_n):
                # A late read of the preceding rung is neither a baseline P1
                # answer nor an answer to the entry currently being measured.
                self.answer_unread_ms = at_ms
        if not onsets:
            return self._nothing_heard(linked)
        # The tracker takes over at corroboration, not at the first reading:
        # until then every cycle re-searches the whole band. `_track` looks only
        # inside `MAX_PULL_S`, so a first reading that landed on the wrong burst
        # would otherwise deafen the search to the right one for the life of the
        # acquisition -- which is what the 899-pair bench measured.
        if self.corroborated:
            return self._track(onsets, seg, seg_start)
        # ...and a grid that has been answered is never re-placed, which used to
        # follow from `d_n` being set and now has to be said: a reading that has
        # not been corroborated still keeps `_place` off the anchor it was
        # measured against.
        if hushed and since_tx >= self.d_max_n and not self.acquired and not linked:
            # Off the air AND out of our own answer band, so nothing we can hear
            # is an answer to us: this is a station running its own raster, and
            # the only thing to do with it is decide where to put ours.
            return self._place(onsets[0])
        return self._acquire(onsets, seg, seg_start, linked)

    def _nothing_heard(self, linked: bool = False) -> str:
        if self.d_n is not None:
            return self._miss("nothing heard")
        # SAY SO, EVERY CYCLE. This returned None, so a session with no reference
        # logged no grid line at all and read exactly like one that was tracking:
        # thirteen cycles of a live session went out against a phase nothing on the
        # channel agreed with and the log never once mentioned that it had nothing
        # to steer by. Same class of fault as CAPTURE STREAM DEAD -- from the
        # inside the arithmetic stays self-consistent while the premise under it
        # is gone.
        return self._blind(linked=linked)

    def _miss(self, why: str) -> str:
        # The onset goes first, on the FIRST miss, while the window keeps its
        # grace. `d` is a property of the peer and worth holding through a fade;
        # the onset is one cycle's reading of where one packet sat, and a cycle
        # that failed to corroborate it is a cycle it no longer describes.
        # Released with the window instead, it survived two uncorroborated
        # cycles -- and on 2026-08-10 that dropped a live link: a mis-acquired
        # d put the derived boundary inside the packet, `key_refusal` dropped
        # the ack, and the silence it bought was itself the next miss, so
        # three cycles running refused on one dead burst @ 2686915 before the
        # third miss broke the loop. Without an onset the placement stands on
        # the free boundary, which keys, and the guard has nothing stale to
        # measure against.
        self._peer_raster_at = self.peer_at
        self.peer_onset = None
        self.misses += 1
        if self.misses < self.MAX_MISSES:
            return (f"held the grid -- no control signal where it is due, {why} "
                    f"({self.misses} of {self.MAX_MISSES})")
        # The corroboration goes with `d`, not with the onset: it is a claim
        # about the same number, and a gap nothing is left to describe is not a
        # corroborated one. The candidates behind it stay -- they age out on the
        # evidence's own span, so a peer that fades for a cycle or two and comes
        # back at the same gap is corroborated by its own earlier answers.
        # `d` GOES AND THE REFERENCE STAYS, because they are different claims.
        # The gap is a reading of the peer that nothing has corroborated for
        # three cycles. The reference is OUR OWN TRANSMISSION that gap was
        # measured from, and it stops describing the peer only when the far end
        # reads our entry packet and moves to the new raster -- which is the far
        # end's doing, and shows up as a burst rather than as silence. Discarded
        # here it fell back to `packet_n`, 150 ms shorter the moment we keyed
        # PACTOR-3: on 2026-09-11 that put the anchored reader at boundary
        # +915 ms against seven sample-indexed answers at +1052 to +1065, and the
        # thirteen consecutive grants behind it were never read again. `_acquire`
        # swaps the anchor when a burst says to.
        self.d_n, self.misses, self.corroborated = None, 0, False
        self.evidence.at = None
        return (f"receive window released after {self.MAX_MISSES} cycles ({why}) "
                f"-- searching for the turnaround again; the transmit grid has "
                f"not moved and will not")

    def _place(self, at: int) -> str:
        """Phase the grid on a station heard while our own transmitter is off.

        A master creates the grid rather than acquiring one, and a reference
        implementation simply starts it 200 ms after the local clock. That is
        fine when the call is answered and a lottery when it is not: both rasters
        are stable to a fraction of a millisecond a cycle, so a phase that puts
        our 0.96 s packet over the peer's 0.12 s burst puts it there for the whole
        session, with no drift to walk us out of it. That happened, and HUSH_CYCLES
        exists because of it.

        So the grid is placed from what we can hear, and only from a cycle spent
        off the air, where nothing on the channel can be an answer to us. A
        burst heard in our own receive window is a different thing entirely: it is
        the peer answering, it sets `d`, and it must not move this.

        WHICH IS NOT THE SAME AS "not keying this cycle", and reading it that way
        cost the master its own rule. A hush arms after the cycle it was decided
        in, so the first cycle of one opens where our last carrier dropped and
        carries that call's whole answer band -- `_acquisition_window` searches it
        as exactly that. On the 25 placements on file the two readings disagree
        10 times; of the 9 the logs pin the geometry of, 6 phased the transmit
        grid on a burst 53-76 ms past our own carrier and the other 3 within
        38 ms of it -- the master retiming towards a peer whose timing is
        already derived from ours. `update` gates this on `since_tx` for that
        reason, and a first-hush burst in the band goes to `_acquire`, which is
        where an answer belongs.

        A decoded connect answer also forbids placement, even later in a hush:
        the peer can repeat its response on the original call grid while we
        listen. Time since our last transmission does not make it an independent
        station once the host has accepted that answer.
        """
        err = self._phase(at)
        self.anchor += err
        self.blind = self.hush_left = 0
        return (f"GRID PLACED from a burst at sample {at}: phase "
                f"{err / FS * 1e3:+.1f} ms, our packet now falls "
                f"{self.offset_n / FS * 1e3:.0f} ms after one of theirs")

    def _acquire(self, onsets: list[int], seg, seg_start: int,
                 linked: bool = False) -> str:
        """Learn `d` from the control signals our own window can hold.

        The master's whole flexibility about the turnaround, and it is spent here:
        it opens a window at its own data ending and searches it. The span is
        bounded above by the instant a 120 ms control signal must be finished for
        us to key on time -- which is also, to a millisecond, where a reference
        implementation's search loop stops, and that is not a coincidence: a peer
        that turns around later than that gets transmitted over whatever anyone
        does about it.

        And below by `D_MIN_S`, because a gap no station could have turned around
        in is not a peer. The rejected onset costs the search nothing -- the rest
        of the cycle's bursts are still tried, and a cycle that offers only
        impossible ones is blind rather than acquired.

        WHAT COMES OUT OF THE BAND IS A READING, NOT YET A PEER. Every in-band
        gap goes to `_TurnaroundEvidence`, and the session's turnaround is the one
        two cycles agree on. Until they do, the newest reading still opens the
        receive window and still gives the decoder an anchor, and only the
        TRACKER waits: an uncorroborated gap is searched for again next cycle
        rather than pulled towards.

        Withholding `d` as well was benched on 2026-08-14 against every rig
        session on file and is the wrong rule: five of the eleven that acquired
        did so off a burst no later cycle ever agreed with, three of them in the
        last cycle of the call, and gating the window on agreement leaves those
        five with nothing to steer by at all. Gating only the anchor costs 1 to
        3 cycles, median 1, over the nine acquisitions that do corroborate --
        paid on the rotation, which is where the 2026-08-14 peer counter
        advances came from.
        """
        d_min_n = round(D_MIN_S * FS)
        band = [at for at in onsets if d_min_n <= self._gap(at) <= self.d_max_n]
        if not band and self.d_ref_n not in (None, self.packet_n):
            # AND THE OTHER ANCHOR, WHICH IS THE PEER HAVING FOLLOWED US. The
            # carried reference survives a release (`_miss`), so the search opens
            # where the answers have been landing; a peer that has since read the
            # entry packet answers a rotation earlier instead, off the packet we
            # are keying now. Which of the two is right is the far end's to say,
            # and it is said here by where a burst actually falls -- never by our
            # own transmitter having changed waveform. Free when the carry is
            # right: the band under it was not empty.
            held, self.d_ref_n = self.d_ref_n, None
            band = [at for at in onsets
                    if d_min_n <= self._gap(at) <= self.d_max_n]
            if not band:
                self.d_ref_n = held
        if not band:
            near = min(onsets, key=lambda a: abs(self._signed_gap(a)))
            # UNDER THE FLOOR IS NOT "NOTHING HEARD", and after a changeover it
            # is the one reading that could be an answer we are rejecting: the
            # yielding station's own turnaround has never been measured, so a
            # station that came back 30 ms after our audio ended would be
            # refused here exactly like the packet tail this floor is for. Named
            # so the two can be told apart in the log rather than inferred from
            # a gap column. `D_MIN_S` is unmoved; this only says what it cost.
            under = [self._gap(at) for at in onsets if 0 <= self._gap(at) < d_min_n]
            floor = "" if not under else (
                f"; {len(under)} UNDER THE FLOOR, nearest "
                f"{min(under) / FS * 1e3:+.0f} ms -- a transmission where only "
                f"a turnaround faster than any on record could put one")
            return self._blind(f"{len(onsets)} burst(s) heard, nearest at "
                               f"{self._signed_gap(near) / FS * 1e3:+.0f} ms "
                               f"where an answer has to fall between "
                               f"{D_MIN_S * 1e3:.0f} and "
                               f"{self.d_max_n / FS * 1e3:.0f}{floor}",
                               linked=linked)
        agreed = self.evidence.offer(self.cycles, [self._gap(at) for at in band])
        at = (band[0] if agreed is None else
              min(band, key=lambda a: abs(self._gap(a) - agreed)))
        dev = _edge_dev(seg, seg_start, at)
        self.d_n = self._gap(at) + (dev or 0.0)
        # ...on whatever `_gap` just measured it against, which is our own
        # transmission unless a carry is standing. See `rx_ref_n`.
        self.d_ref_n = self.rx_ref_n
        self.corroborated = agreed is not None
        self.acquired = True
        self.peer_onset = at
        self.misses = 0
        self.blind = self.hush_left = 0
        edges = '' if dev is None else f' (edges {dev / FS * 1e3:+.1f} ms)'
        if not self.corroborated:
            why = (f"{self.evidence.witnessed} turnarounds are witnessed in this "
                   f"window and timing cannot say which is answering us"
                   if self.evidence.witnessed > 1 else
                   f"no second cycle inside {self.evidence.SPAN} has agreed to "
                   f"{self.evidence.TOL_N / FS * 1e3:.0f} ms yet")
            return (f"turnaround CANDIDATE @ sample {at}: d = "
                    f"{self.d_n / FS * 1e3:.1f} ms after our data ends{edges} "
                    f"-- {why}, so the receive window opens on it and the "
                    f"tracker waits")
        return (f"TURNAROUND ACQUIRED @ sample {at}: d = "
                f"{self.d_n / FS * 1e3:.1f} ms after our data ends{edges}, "
                f"corroborated in cycle {self.evidence.at}; the transmit grid "
                f"is unchanged")

    def _track(self, onsets: list[int], seg, seg_start: int) -> str:
        """Correct the RECEIVE window, and only it, by an eighth of the error.

        Two terms, and they are one measurement at two resolutions: the detector
        places the burst on its 5 ms grid inside MAX_PULL_S, and the edge statistic
        resolves inside that. A reference implementation's estimator
        searches +/- one bit around each expected transition and has no capture
        range at all beyond it, because it reads at a fixed instant and we search;
        composing the two gives the same quantity over a wider range. The GAIN is
        the protocol's and is not ours to choose -- see TIMING_GAIN.
        """
        best = min(onsets, key=lambda a: abs(self._err(a)))
        err = self._err(best)
        if abs(err) > round(MAX_PULL_S * FS):
            return self._miss(f"nearest {err / FS * 1e3:+.0f} ms")
        self.misses = 0
        self.peer_onset = best
        dev = _edge_dev(seg, seg_start, best)
        total = err + (dev or 0.0)
        self.d_n += TIMING_GAIN * total
        late = "" if self.d_n <= self.d_max_n else (
            f" -- LATER THAN WE CAN HEAR IT OUT ({self.d_max_n / FS * 1e3:.0f} ms); "
            f"we key over the end of it")
        early = "" if self.d_n >= round(D_MIN_S * FS) else (
            f" -- UNDER THE {D_MIN_S * 1e3:.0f} ms TURNAROUND FLOOR, which no "
            f"station could have answered in: the reading is no longer a "
            f"turnaround and nothing may be measured against it")
        return (f"d {self.d_n / FS * 1e3:.1f} ms @ sample {best}: error "
                f"{total / FS * 1e3:+.1f} ms"
                f"{'' if dev is None else f' (edges {dev / FS * 1e3:+.1f})'}, "
                f"window {TIMING_GAIN * total / FS * 1e3:+.1f} ms{late}{early}")

    # -- having nothing to steer by, which is its own state ----------------
    def _p3_control_heard(self) -> Optional[str]:
        """This cycle's PACTOR-3 codeword at the sending end, where there is one.

        The ISS's own half of what `note_p3_control` records for the receiving
        end. `_forecast_next_key` hands the decoded word over before the fold,
        so the word in hand is the answer to the packet we keyed in this cycle;
        a word read under our own carrier never gets that far.
        """
        cs = self.peer_cs
        if (self.protocol != Protocol.PACTOR3 or not self.sending
                or cs is None or cs.protocol != Protocol.PACTOR3
                or self.cycles - cs.cycle > 1):
            return None
        return (f"the {cs.name} we decoded at sample {cs.at} answered this "
                f"cycle -- the onset detector found nothing where the PACTOR-1 "
                f"clock puts an answer, and a PACTOR-3 peer does not answer "
                f"there")

    def _blind(self, why: str = "nothing heard", *, linked: bool = False) -> str:
        """One cycle with nothing to steer by, and whether to go quiet over it.

        THE HUSH IS A LINK-SETUP MOVE AND NOTHING ELSE. It buys one thing: a
        grid whose phase has never been checked against anything gets placed by
        what it hears instead of by where it started. That deadlock needs both
        halves -- our own transmission covering the peer's burst, AND no evidence
        on this grid that anything is there -- and the second half stops being
        true the moment a turnaround is measured or a link comes up.

        So neither state arms one. A grid that has acquired a `d` heard the peer
        INSIDE ITS OWN RECEIVE WINDOW, with our carrier up in the same cycle: the
        phase is demonstrably not one that hides it, and `d` going back to None
        after MAX_MISSES does not un-hear it, which is why `acquired` latches
        where `locked` does not. A session holding a link is the same argument
        from the other end, and it is the one with teeth: an IRS times its
        acknowledgement against a 1.25 s raster and is gone several cycles before
        a six-cycle hush is spent, so all the hush can do there is drop a station
        that IS answering. Measured on 2026-08-06 over four sessions against three
        gateways: the setup phase transmitted 4 cycles in every 10 and the hold
        phase kept the same rhythm the whole way down, on a raster that never
        once acquired.

        An in-flight hush drains while setup remains unanswered. `update`
        cancels its remaining cycles as soon as a link comes up.
        """
        # WHAT THE ONSET DETECTOR SAW, AND ONLY THAT. Every branch below the
        # credit counts bursts; none of them asks a decoder anything, and on an
        # upgraded link the two are looking in different places -- this line said
        # NO CONTROL SIGNAL through fourteen decoded PACTOR-3 acknowledgements on
        # 2026-09-10, because the clock it hangs on is the PACTOR-1 onset one
        # and that had already been released. So it says which clock it is on.
        said = f" -- {self.reading}" if self.reading else ""
        # AND A DECODED CODEWORD IS A HEARD CYCLE. As PACTOR-3 ISS the detector
        # is searching a clock the peer has left, so a station answering every
        # single cycle counted as silence: VE3KPG's 35 consecutive CS1 at zero
        # bit errors on 2026-09-13 ran this count from 1 to 49 while every one
        # of our packets was being answered. The credit is freshness and nothing
        # else -- it places no window and moves no anchor.
        heard = self._p3_control_heard()
        if heard is not None:
            self.blind = 0
            return f"{heard}{said}"
        self.blind += 1
        if self.answered_word is not None and self.answered_word[0] == self.cycles:
            _, cs, d = self.answered_word
            return (f"ANSWERED, NOT PLACED -- {why} at the onset detector, and "
                    f"CS{cs + 1} decoded at zero bit errors {d * 1e3:.0f} ms "
                    f"into the band an answer to our call was due in. The peer "
                    f"is answering and our carrier is not covering it, so no "
                    f"hush is armed; one codeword aims nothing{said}")
        if not self.hush_left and (self.acquired or linked):
            since = ("this grid has been answered before" if self.acquired
                     else "the link is up")
            return (f"NO CONTROL SIGNAL -- {why} for {self.blind} cycle(s); "
                    f"staying on the air, because {since} and silence cannot "
                    f"place the grid better than that{said}")
        if self.hush_left:
            self.hush_left -= 1
            if not self.hush_left:
                # The hush is spent. Start the count again rather than falling
                # straight back into it -- a peer that is simply not on frequency
                # should be called, not listened to forever.
                self.blind = 0
                return ("NO CONTROL SIGNAL -- the hush heard nothing either; "
                        f"back on the air and calling{said}")
        elif self.blind >= self.BLIND_CYCLES:
            self.hush_left = self.HUSH_CYCLES
        if self.hush_left:
            return (f"NO CONTROL SIGNAL after {self.blind} cycles ({why}) -- OFF "
                    f"THE AIR, {self.hush_left} cycle(s) of listening in the clear "
                    f"left to find a raster to call on{said}")
        return (f"NO CONTROL SIGNAL -- {why} for {self.blind} cycle(s); calling on "
                f"a grid nothing has answered yet "
                f"({self.BLIND_CYCLES - self.blind} more before going quiet)"
                f"{said}")

    # -- did the peer READ the entry packet -------------------------------
    @property
    def entry_read_ms(self) -> Optional[float]:
        """Where a peer that read our entry packet answers, or None.

        THE SPLIT IS OURS AND NOT THE REFERENCE'S. `PIII_Complete_1`'s IRS moved
        92 ms on the first entry it read, and that number is the reference pair's
        transmitter and turnaround; ours is the difference between the two
        transmissions of OUR OWN that a peer can time its answer off -- 960.0 ms
        of PACTOR-1 packet from the boundary against `entry_end_n`, which is
        where the entry we actually keyed ends (838.9 ms for the SL1 template
        `ENTRY_END_N` predicts) -- whatever the turnaround happens to be. So the
        turnaround cancels and does not have to be known twice: the position
        measured before the entry was keyed carries it, and reading the entry
        subtracts the difference of the two packets from it.

        THE OTHER ROUTE TO THE SAME NUMBER, and it is why the difference is taken
        off the last SYMBOL rather than off our carrier dropping 24.7 ms later:
        `PIII_Complete_1`'s IRS answers 890.8 ms after the packet's phase
        reference, and our boundary leads that reference by 31.8 ms: the 26.8 ms
        of skirt `ENTRY_END_N` accounts for, plus the half symbol
        `placement.CASE0_STAGGER` puts channel 5 ahead of the rest, which is
        where `ENTRY_END_N`'s extent is measured from and where the reference's
        is not. 31.8 + 890.8 is 922.6 ms, which is the same instant this
        arithmetic reaches from our own 71 ms turnaround.

        A KEYED EXTENT CARRIES THE PULSE'S RING-OUT behind that last symbol --
        4.4 ms of it on the template entry -- which is inside the half symbol
        `tests.shrike.test_p3_upgrade` already holds `ENTRY_END_N` to, so no
        ring-out term is taken off it here.
        """
        return (None if self.answer_unread_ms is None else
                self.answer_unread_ms - (self.p1_data_n - self.entry_end_n) / FS * 1e3)

    def _population(self, ms: float) -> tuple[str, float]:
        """Which of the two positions `ms` is, and how far off it sits.

        `MAX_PULL_S` is the tolerance because it is already this grid's answer to
        "is that the same instant" -- the capture range the tracker calls a burst
        the peer's control signal. It is a fifth of the split, so nothing about
        the verdict turns on the choice.
        """
        unread, read = self.answer_unread_ms, self.entry_read_ms
        tol = MAX_PULL_S * 1e3
        if abs(ms - unread) <= tol:
            return "AT THE PACTOR-1 POSITION", ms - unread
        if abs(ms - read) <= tol:
            return "AT THE ENTRY POSITION", ms - read
        return "OFF BOTH POSITIONS", min(ms - unread, ms - read, key=abs)

    def answer_position(self) -> Optional[str]:
        """Where this cycle's answer sat, if an entry packet is what it answers.

        The measurement three slots have taken by hand off the `[grid] d` series
        and the `at anchor` tags, said once, in the cycle it was measured in.
        Nothing new is measured: the position is `rx_ref_n` -- our own PACTOR-1
        data end, held across the upgrade for the reason `rx_ref_n` gives -- plus
        the cycle's nearest gap, which is the same reading `arq.note_burst` is
        fed and the same one `[grid] d` tracks.
        """
        if not self.entry_answers or self.answer_unread_ms is None:
            return None
        cycle, at, ms = self.entry_answers[-1]
        if cycle != self.cycles:
            return None
        where, _ = self._population(ms)
        return (f"ENTRY ANSWER {where}: {ms:.1f} ms after our slot boundary "
                f"@ sample {at} in cycle {cycle}, "
                f"{ms - self.answer_unread_ms:+.1f} ms off where it answered "
                f"before the entry ({self.answer_unread_ms:.1f}) and "
                f"{ms - self.entry_read_ms:+.1f} ms off where reading one puts "
                f"it ({self.entry_read_ms:.1f})")

    def entry_verdict(self) -> str:
        """The run's account of the answer position. One paragraph, always said.

        SAID EVEN WHEN THERE IS NOTHING TO SAY, which is the failure mode this
        exists against: an arm that drew no grant keys no entry, so where the
        peer answered decides nothing, and a line that reports a position without
        saying so reads as a result. Arms have been spent on that reading.
        """
        if self.entry_at is None:
            return ("NO ENTRY PACKET WAS KEYED IN THIS ARM: nothing went out for "
                    "the peer to read, so where it answered decides nothing "
                    "about entry detection and no position is reported")
        if self.answer_unread_ms is None:
            return (f"NO ANSWER POSITION BEFORE THE ENTRY: an entry packet was "
                    f"keyed in cycle {self.entry_at} and no cycle before it "
                    f"measured where the peer answered, so the two positions "
                    f"cannot be placed and nothing here is a reading")
        if not self.entry_answers:
            # ...WHICH IS A STATEMENT ABOUT ONSETS. `entry_answers` is fed from
            # the nearest onset and from nothing else, so a link whose answers
            # are decoded but whose bursts never registered reads as silence
            # here. What the decoders made of the same cycles is beside it.
            return (f"NO ANSWER TO THE ENTRY PACKET: one was keyed in cycle "
                    f"{self.entry_at} and no cycle since carried a burst to time. "
                    f"The peer answered at {self.answer_unread_ms:.1f} ms after "
                    f"our slot boundary before it and has not answered since"
                    f"{f'. {self.reading}' if self.reading else ''}")
        placed = [(self.entry_tracked.get(cycle, ms), cycle in self.entry_tracked)
                  for cycle, _, ms in self.entry_answers]
        series = ", ".join(f"{ms:.1f}{' (tracked)' if word else ''}"
                           for ms, word in placed)
        instrument = ("" if not self.entry_tracked else
                      ". The positions marked (tracked) are the codeword's own "
                      "instant and the rest are the onset detector's")
        first, last = self.entry_answers[0][0], self.entry_answers[-1][0]
        cycles = (f"cycle {first}" if first == last else
                  f"cycles {first}-{last}")
        seen = [self._population(ms) for ms, _ in placed]
        held = [w for w, _ in seen]
        worst = max(abs(o) for _, o in seen)
        verdict = {
            frozenset({"AT THE PACTOR-1 POSITION"}): "THE ANSWER DID NOT MOVE",
            frozenset({"AT THE ENTRY POSITION"}): "THE ANSWER MOVED TO THE ENTRY POSITION",
            frozenset({"OFF BOTH POSITIONS"}): "THE ANSWER IS AT NEITHER POSITION",
        }.get(frozenset(held), "THE READINGS DO NOT AGREE WITH EACH OTHER")
        return (f"ENTRY ANSWER POSITION: {series} ms after our slot boundary over "
                f"{len(seen)} cycle(s) ({cycles}) since the entry was keyed, against "
                f"{self.answer_unread_ms:.1f} ms measured before it. Our "
                f"PACTOR-1 packet's last bit falls "
                f"{self.p1_data_n / FS * 1e3:.1f} ms after the boundary and the "
                f"entry we keyed ends {self.entry_end_n / FS * 1e3:.1f} ms past "
                f"the same boundary, so a "
                f"peer that read the entry answers at "
                f"{self.entry_read_ms:.1f}. {held.count('AT THE PACTOR-1 POSITION')} "
                f"of {len(seen)} sit at the PACTOR-1 position and "
                f"{held.count('AT THE ENTRY POSITION')} at the entry position, "
                f"worst {worst:.1f} ms off. {verdict}. The entry position is where "
                f"our own keyed geometry puts a peer that read one; it is a "
                f"prediction, and this is the measurement against it{instrument}")


def _scheduler_reading(sessrx, host, raster: _MasterGrid,
                       answered: Optional[tuple[int, float]] = None) -> str:
    """What the run is doing, for the two lines that report what it heard.

    DISPLAY ONLY. Nothing reads this back and no decision turns on it; it is
    appended to `_blind`'s line and to `entry_verdict`'s, both of which count
    onsets and neither of which can see a decoder. On 2026-09-10 that pair
    reported NO CONTROL SIGNAL and NO ANSWER TO THE ENTRY PACKET while the
    tracked PACTOR-3 reader in the same run decoded fourteen acknowledgements,
    and the run's headline verdict contradicted its own control-signal tally.
    Both lines were true about the onset detector and false about the link.

    Four clauses, because those are the four things that were not recoverable
    from the transcript: which mode the link is in, what the last control word
    literally was, what the ARQ made of it, and which clock the scheduler is
    keeping time on -- the PACTOR-1 onset one these lines hang off, the tracked
    PACTOR-3 answer clock, or neither.
    """
    role = host.arq.role or "no role"
    entry = ", entry pending" if host.arq.entry_pending else ""
    mode = (f"mode {host.protocol.value} {role} in {host.arq.state.name}"
            f"{entry}")
    if sessrx.cs_log:
        ev = sessrx.cs_log[-1]
        word = f"last control {ev.text} @ {ev.t:.2f} s"
    elif answered is not None or raster.answered_word is not None:
        # NOT "NOTHING HAS BEEN READ". `cs_log` holds what a decoder was given,
        # and the connect search is not one: an uncorroborated accept never
        # reaches `_on`, so a session that had read three zero-error codewords
        # printed this line under every one of them and sent the operator after
        # a receiver fault that was not there. `onair-0911-2332`, 2026-09-11.
        read, d = answered if answered is not None else raster.answered_word[1:]
        when = ("this cycle" if answered is not None
                else f"cycle {raster.answered_word[0]}")
        word = (f"no control word at a decoder; CS{read + 1} read at zero bit "
                f"errors in {when}'s answer band, d {d * 1e3:.0f} ms")
    else:
        word = "no control word has been read this session"
    cs = sessrx.cs_heard
    took = (f"the ARQ was given CS{cs + 1} {spec.CS_NAMES[cs]}" if cs is not None
            else "nothing reached the ARQ as a codeword this cycle")
    if sessrx._p3_answer_at is not None:
        clock = ("the tracked PACTOR-3 answer clock @ sample "
                 f"{sessrx._p3_answer_at}")
    elif sessrx._p3_row0 is not None:
        clock = f"the tracked PACTOR-3 frame clock @ sample {sessrx._p3_row0}"
    elif raster.d_n is not None:
        clock = f"the PACTOR-1 onset clock, d {raster.d_n / FS * 1e3:.1f} ms"
    else:
        clock = "the free-running transmit grid alone, no turnaround held"
    return f"{mode}; {word}; {took}; scheduler on {clock}"


def _grid_reversal(raster: _MasterGrid, host) -> Optional[str]:
    """Rotate the grid if the link's direction is not the grid's, else nothing.

    Asked around each cycle's decoders and tick. A yield found before the key
    must move the grid before transmission; taking the link is decided inside
    the tick; and a yield found by any post-key decoder must move the grid before
    calculating the next receive window. These checks only compare directions
    and apply the rotation arithmetic when needed.

    THE GRID IS THE REFERENCE, not a snapshot of the role. Snapshotted at the top
    of the cycle, a CS3 the post-key flush delivered changed the role after the
    snapshot and before the next one, and the next cycle read "role unchanged"
    over a grid still phased for sending: KB5LZK 2026-09-07 21:57 CDT, CS3 at
    223.65 s, `changeover -> IRS` with no `GRID REVERSED`, and every reply after
    it 0.5 s outside the window.
    """
    if (getattr(host, "protocol", raster.protocol) != Protocol.PACTOR3
            or getattr(host.arq, "state", None) not in LINKED):
        if raster._p3_peer_confirmed:
            raster._clear_p3_reply_timing()
        raster._p3_peer_confirmed = False
        raster._p3_reply_phase_invalid = False
    elif not getattr(host.arq, "entry_pending", True):
        # Bare P3 controls can confirm entry without a CRC packet. This runs
        # after every RX route, even when accepting the answer kept our role.
        raster._p3_peer_confirmed = True
    role = host.arq.role
    if role == ISS:
        # SAMPLED WHILE IT IS STILL TRUE, every cycle the role stands, because
        # the peer's agreement to a changeover we keyed can arrive on any cycle
        # of the stint and the state that says so is gone by the time its
        # resumed stint has put the role back. See `_MasterGrid.reverse`.
        raster.turn_accepted = not getattr(host.arq, "unconfirmed_breakin", False)
    if role is None or raster.sending == (role == ISS):
        return None
    # A decoded changeover may also change protocol before anything has keyed
    # in it. Rotate using that accepted protocol, not the last emitted waveform:
    # P1's 960-120 ms rotation puts a P3 ACK 240 ms late (810-210 ms).
    raster.protocol = host.protocol
    return raster.reverse(to_iss=role == ISS)


def _align_shift(raster: _MasterGrid, sessrx, tx, slot: int, at: int) -> None:
    """Take the peer's FSK shift for the cycle its control signal belongs to.

    Normally the peer's reading sets our phase. The opt-in P1-only setup
    experiment retains the call train's phase until the first payload settles.
    KI0BK's recorded reply polarity disagreed with that coherent call train;
    this switch tests causality without changing the normal alignment policy.

    Re-aims whatever transmission the loop is currently holding, so a correction
    decided before the key reaches that burst rather than the one after it.
    """
    ev = sessrx.cs_log[-1] if sessrx.cs_log else None
    if ev is None or ev.sense is None:
        return
    if (sessrx.host.protocol == Protocol.PACTOR3
            and not sessrx.host.arq.entry_pending and ev.protocol == Protocol.PACTOR1):
        # Repeated P1 grants still place unconfirmed entry. Once the peer has
        # answered in P3, a lower-protocol word cannot change P3 carrier order.
        return
    if (getattr(tx, "p1_setup_phase", "reply") == "call"
            and sessrx.host.stay_in_pactor1
            and sessrx.host.protocol == Protocol.PACTOR1
            and sessrx.host.arq.role == ISS
            and sessrx.host.arq.state in (State.CONNECTING, State.CONNECTED)
            and not sessrx.p1_setup_finished):
        if raster.shift(raster.rx_slot(at)) != bool(ev.sense):
            print("    [grid] P1 setup experiment: retaining call phase "
                  "despite opposite peer CS sense", flush=True)
        return
    line = raster.align(raster.rx_slot(at), ev.sense)
    if line:
        tx.aim(raster, slot)
        print(f"    [grid] {line}", flush=True)


def _reverse_before_key(raster: _MasterGrid, host, tx, slot: int) -> Optional[str]:
    """Rotate the grid, and re-aim the transmission at the slot it now lands in.

    The re-aim is the half that has teeth. `reverse` moves the anchor; nothing
    keys off the anchor directly, and `tx.boundary` -- which `_tx` waits on --
    was read from it before the rotation. Left stale, a station that has just
    yielded keys a whole rotation early, into the packet it was told to listen to.
    """
    line = _grid_reversal(raster, host)
    if line:
        tx.aim(raster, slot)
        print(f"    [grid] {line}", flush=True)
    return line


def _receive_changeover(live, raster, tx, host, sessrx, slot: int,
                        seg: np.ndarray, seg_start: int,
                        prev: np.ndarray, prev_start: int, settle_n: int,
                        *, reserve_s: float = 0.0) -> np.ndarray:
    """Collect and decode the CS3 body before its first reply opportunity."""
    if (live is None or sessrx.cs_heard != CS_BREAKIN
            or not _reverse_before_key(raster, host, tx, slot)):
        return seg
    head, boundary = sessrx.cs_at, tx.boundary
    p3 = host.protocol == Protocol.PACTOR3
    if p3:
        # P1's 960 ms body extended collection to the PTT settle, then a
        # cold P3 scan spent 125-131 ms and lost VE3KPG's first ACK. P3 needs
        # its own 810 ms extent plus filter support. take_until blocks until
        # that audio is delivered, so its holdback and callback batch belong
        # in the deadline budget, not in the requested sample extent.
        ready = (head + round(placement.PACKET_S * FS)
                 + rxfront.MATCHED_DELAY_N + rxfront.SPS // 2)
        deadline = (_p3_decode_deadline(live, tx.key_instant(raster, slot), settle_n)
                    - round(sessrx.CHANGEOVER_BODY_RESERVE_S * FS)
                    - live.holdback - _callback_delivery_n(live))
        until = min(ready, deadline)
    else:
        end = head + round(spec.P1_PACKET_S * FS)
        until = min(end + round(BK_TAIL_GUARD_S * FS), boundary - settle_n,
                    max(end, boundary - settle_n - round(reserve_s * FS)))
    rest = live.take_until(until)
    if rest.size:
        sessrx.skip(rest.size / FS)
        seg = np.concatenate([seg, rest])
    if p3 or rest.size or head < seg_start:
        joined = _breakin_audio(head, seg, seg_start, prev, prev_start)
        origin = seg_start + seg.size - joined.size
        if p3:
            sessrx.changeover_body(joined, origin, head)
        else:
            sessrx.expect_frame()
            _scan_frame(sessrx, joined, origin)
    return seg


def _breakin_audio(head: int, seg: np.ndarray, seg_start: int,
                   prev: np.ndarray, prev_start: int) -> np.ndarray:
    """The stream from a changeover head forward, across the join between windows.

    A changeover packet is the one transmission that does not sit on our grid, so
    its 960 ms can straddle the edge between two of our windows and neither holds
    a frame. That edge is a seam in the bookkeeping and not in the audio whenever
    the windows are contiguous, so the scan is handed both.

    AND ONLY THEN. What separates two windows that are not contiguous is our own
    carrier: a packet spanning that gap was transmitted over, and no reader
    recovers it. The contiguity test is the physics rather than a caution -- and
    on a cycle we keyed in it is never true, so the loss is SAID rather than
    left to be inferred from a scan that found nothing.
    """
    if head >= seg_start or head < prev_start:
        return seg
    gap = seg_start - (prev_start + prev.size)
    if gap:
        print(f"    [grid] the changeover head at sample {head} fell in the "
              f"window before this one and our own carrier holds the "
              f"{gap / FS * 1e3:.0f} ms between them -- the packet behind it "
              f"is not in any audio we have", flush=True)
        return seg
    return np.concatenate([prev[head - prev_start:], seg])


def _loss_seen(live) -> Optional[str]:
    """What a capture source is known to be missing, or None if nothing yet.

    Two counters, one consequence, and the one that has never fired here is the
    one every grader used to ask. `xruns` is the driver's flag: zero on all 68
    capture-clock readings in this station's record, 31 of which were short of
    the air, and zero across the 450 windows that lost audio on the 34 sessions
    binned for the raster. `lost` is `rates.lost_step` -- the converter's own
    timestamps against the delivered count -- which is the loss this station
    actually has: measured inside 10% of the stream clock's own shortfall at
    every stall from 60 to 600 ms, and against a second receiver at 313 ms to
    the KiwiSDR's 298.

    So the count leads and the flag follows it, and everything that grades this
    capture asks here. It used to be asked separately in three places, all three
    of them on the flag alone, which is how a preflight cleared a starved stream
    and a session logged nothing while a fifth of its windows were spliced.
    """
    said = []
    if live.lost:
        said.append(f"{live.lost} samples ({live.lost / live.fs * 1e3:.0f} ms) "
                    f"the converter timestamped and we were never handed")
    if live.xruns:
        said.append(f"{live.xruns} driver xrun(s), first at sample {live.xrun_at}")
    return ", ".join(said) or None


class _CycleEvidence:
    """One cycle's window: written to disk, and what it says about our own
    receiver said out loud. ONE BODY, BOTH LOOPS -- every check here stood in
    the setup phase alone, and a hold runs to `HOLD_MAX_CYCLES`."""

    def __init__(self, live, tx, raster: "_MasterGrid", outdir: Path):
        self.live, self.tx, self.raster, self.outdir = live, tx, raster, outdir
        self.xruns = self.lost = 0

    def record(self, name: str, seg: np.ndarray,
               seg_start: int) -> list[tuple[int, int]]:
        live = self.live
        if not seg.size:
            return []
        _save_capture_async(self.outdir / f"{name}.wav", seg, live.xruns,
                            end=seg_start + seg.size, lost=live.lost)
        if live.lost > self.lost or live.xruns > self.xruns:
            new, first = live.lost - self.lost, not (self.lost or self.xruns)
            self.lost, self.xruns = live.lost, live.xruns
            print(f"    !!!! CAPTURE LOSS: {new} samples "
                  f"({new / live.fs * 1e3:.0f} ms) in this window -- "
                  f"{_loss_seen(live)} for the session so far", flush=True)
            if first:
                print("    !!!! Every sample index from here on is offset from "
                      "the air by at least that much and growing: the cycle "
                      "grid has shifted permanently, and no timing measurement "
                      "from this session is usable. SESSION INVALID for timing "
                      "-- restart it.", flush=True)
        # AGAINST FULL SCALE, and against the window's own peak until now, which
        # is the measurement the sidecar beside it warns about: relative to the
        # peak, every capture rails at its own maximum and a real one hides
        # inside a threshold. 2026-08-29's clipping arm put 0.006% to 0.173% of
        # each window on the rail and this line printed on none of the 25, at 2%
        # of a peak that was itself the rail.
        railed = levels.railed_ppm(seg) / 1e4
        if railed:
            # THE RIG'S AF KNOB IS NOT IN THIS PATH, and saying so cost an
            # operator a recalibration on 2026-08-22. This station feeds the
            # modems from `USB Audio Device` off the data port; `AF` drives the
            # speaker output, which is the separate `KT USB Audio` tap and is
            # calibrated there at 0.243 -- 0.271 rails 17% of that tap's samples
            # while still sounding fine on the speakers, because the tap sees the
            # AF stage ahead of the operator's volume control. Name the controls
            # that reach THIS converter.
            print(f"    !! RX {railed:.3f}% SATURATED -- lower the input "
                  f"gain on the modem's own capture device: "
                  f"`tools/codec_gain.py --device <name> --set`, or [audio] "
                  f"input_gain in the station file. NOT the rig's AF, which "
                  f"drives the speaker tap and not this feed. Clipping can "
                  f"impair decoding; this reading does not establish peer silence.",
                  flush=True)
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < RX_DEAF_RMS:
            # ONE BAND DOWN IS NOT THE CODEC'S FAULT, and this line sent the
            # operator to it anyway. The capture gain is shared across bands: on
            # 2026-08-20 a 20 m arm read 0.0046 RMS while 40 m read 0.055-0.065
            # on that same setting, so raising it to lift 20 m would have
            # overdriven 40 m. RF gain is the per-band lever and recovered 18 dB.
            # The sentence lives in `core.levels`, so this and `rx_verdict`
            # cannot drift apart.
            print(f"    !! RX level {rms:.4f} RMS -- receiver may be DEAF. "
                  f"{levels.QUIET_CONTROLS} A silent window here is NOT "
                  f"evidence the gateway stayed quiet", flush=True)
        # In the dead time behind our own carrier: 7 ms on a 1.25 s window,
        # against a 28 ms transmit slot.
        bursts = _peer_bursts(seg, seg_start)
        _report_collision(bursts, self.tx)
        # THE LENGTHS GO OUT WITH THE ONSETS. `_MasterGrid.note_peer_bursts` is
        # the caller that needs them, and dropping them here is what left it
        # nothing to re-origin a raster on while our own carrier was over the
        # peer's transmissions.
        return bursts


def _budget(cycle: float, offset: float, settle: float) -> tuple[bool, str]:
    """Does one cycle hold their answer, our packet and both turnarounds?

    Stated rather than assumed, because every term in it has been wrong at least
    once and the failure is silent: a schedule that does not close still runs, it
    just transmits over the far end. Reported at startup with the numbers, so a
    change to the settle or the cycle argues with the arithmetic instead of with a
    later recording.
    """
    keyed = settle + spec.P1_PACKET_S
    clear = cycle - keyed
    # The band of peer turnarounds this schedule can actually SEARCH. Too early and
    # the rig is still deaf coming out of transmit; too late and the answer is still
    # running when we key. A commercial modem keys 961 ms of 1250 and so services
    # 55-169; the number to watch is how much of that band we give away.
    #
    #
    # TWO TOPS, and the 21 ms between them is the whole reason this is stated
    # here. `hearable` is where a codeword is still complete before we key --
    # `_MasterGrid.d_max_n`, the bound an onset is rejected against. `latest` is
    # where one can be READ, which the codeword search needs `ACQUIRE_TAIL_S` more
    # audio for. The gate belongs on the second: while it sat on the first the
    # g90 -- settle 0.100 -- passed here with a 15 ms band and was then handed no
    # band at all by `_acquisition_window`. It called, it printed "RX (nothing
    # decoded)" every cycle, and it could not have read an answer that was there.
    earliest = TR_SWITCH_S
    # ...AND `latest` PAYS THE ADMISSION RESERVE, where `hearable` does not. The
    # reserve does not move when we KEY, so a codeword is complete before the
    # carrier exactly as it was; it moves when we stop READING, and a codeword
    # still running inside `TX_ADMIT_RESERVE_S` reaches no reader however
    # complete it was. At a 40 ms settle that is the readable band going from
    # 55-109 ms to 55-107, which still holds `D_NOMINAL_S`, `PEER_TURNAROUND_S`'s
    # 96 ms median and the 93-100 this station measured at WS8EOC -- and it is
    # the term that CAPS the reserve. See `TX_ADMIT_RESERVE_S`.
    hearable = clear - spec.P1_CS_S
    latest = clear - ACQUIRE_TAIL_S - TX_ADMIT_RESERVE_S
    implied = cycle - spec.P1_PACKET_S - offset       # the `d` this offset assumes
    # A millisecond of slack: the packet is nominal, the T/R figure is a median over
    # 46 windows, and `cycle` may be the peer's MEASURED period rather than 1.250
    # exactly. Comparing these as exact reals once refused to key over a 30
    # MICROSECOND overrun, which is precision the inputs do not have.
    slack = 1e-3
    # The gate is on the SCHEDULE -- whether any peer turnaround can be serviced --
    # not on `implied`, which is a bootstrap the first measured answer replaces. A
    # bootstrap outside the band is said out loud and is not a reason to refuse
    # to key.
    fits = latest >= earliest - slack
    note = "" if earliest - slack <= implied <= latest + slack else (
        f" -- OUTSIDE that band, so the first answer is "
        f"{'clipped by T/R' if implied < earliest else 'still running when we key'} "
        f"until the measured d replaces it")
    # ...and against where a peer actually answers, which is neither of those
    # numbers and is the one the operator wants. A band that stops short of the
    # median is a receiver that will hear half its answers at best, and it fails
    # by printing "nothing decoded" -- indistinguishable from a quiet frequency.
    lo_d, mid_d, hi_d = PEER_TURNAROUND_S
    short = "" if latest >= mid_d else (
        f" -- SHORT OF THE MEDIAN ANSWER: this rig cannot read a peer at "
        f"{mid_d * 1e3:.0f} ms, and a station that answers reads as silence")
    if hearable <= earliest - slack:
        # The x6100 case: the top of the band is NEGATIVE, and printed as a
        # range it read "d 55--230 ms" -- a phantom 55-230 an operator takes
        # for a band. An empty band is a fact, and the line states it as one.
        band = ("the acquisition band is EMPTY -- this settle leaves no d at "
                "which an answer fits between T/R and our next key-down")
    else:
        band = (
            f"holds an answer whole for d {earliest * 1e3:.0f}-"
            f"{hearable * 1e3:.0f} ms, "
            + (f"of which {earliest * 1e3:.0f}-{latest * 1e3:.0f} can be read"
               if fits else "NONE of which is long enough to read"))
    return fits, (
        f"cycle budget: settle {settle:.3f} + packet {spec.P1_PACKET_S:.3f} "
        f"= {keyed:.3f} keyed of {cycle:.3f} s, leaving {clear * 1e3:.0f} ms clear; "
        f"{band}, against a measured peer at {lo_d * 1e3:.0f}-{hi_d * 1e3:.0f} "
        f"(median {mid_d * 1e3:.0f}); "
        f"TX offset {offset:.3f} assumes d {implied * 1e3:.0f} ms{note}{short}")


def _trim_silence(audio: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    """Drop leading/trailing near-silence so PTT covers only the RF."""
    env = np.abs(audio)
    on = np.where(env > thresh * (env.max() or 1.0))[0]
    return audio if on.size == 0 else audio[on[0]:on[-1] + 1]


class _PeerCodeword(NamedTuple):
    """A twelve-bit word read at zero errors, and where it sat.

    The sending side's whole evidence base. Not an onset and not an energy
    shape: `_read_codeword_at_bursts` decodes, `_forecast_next_key` records,
    and nothing else writes here. The narrator that stood on `max(onsets)`
    alarmed on one cycle a third party's receiver puts clear by 113 ms and said
    nothing about twenty real collisions.
    """
    at: int                 # capture-stream sample of its onset
    width: int              # samples it occupies
    name: str               # which codeword, as the log names it
    who: Optional[str]      # the station this link called, where one is up
    cycle: int              # the grid's fold count when it was read
    protocol: Optional[Protocol] = None


#: Cycles a decoded codeword may still refuse a key. Past it the projection is
#: an assumption about a station last heard from four cycles ago.
ISS_GUARD_CYCLES = 3

#: Keying intervals kept for the contradiction test in `_forecast_next_key`.
#: The forecast reads a codeword out of the window just closed, so it never has
#: to look further back than the cycles that window spans.
KEYINGS_KEPT = 8

#: Consecutive refusals, in either role, before one burst goes anyway. A peer
#: stuck transmitting on our own raster keeps its codeword fresh every cycle, so
#: the refusal would never lift and the guard would hand over the channel it
#: exists to protect. The receiving side reaches the same place by a shorter
#: road: its phase is `_d_max_n` less the turnaround, which does not move inside
#: a session, so one refused acknowledgement is every acknowledgement of that
#: link. Three drops then a key bounds either at quarter rate rather than at
#: zero, and the line that keys says which it is.
GUARD_MAX_DROPS = 3


def tune_carrier(rig, out_dev, seconds: float, drive: float) -> None:
    """Key a steady carrier so the ATU can match a new band.

    A single tone, not a modulated burst: an ATU needs constant amplitude to
    settle. Power is held down two independent ways -- RFPOWER at the rig and a
    low soundcard drive -- because on a data interface the audio level sets the
    output, so lowering only RFPOWER would not do it. RFPOWER is restored after.
    """
    tone = drive * np.sin(2 * np.pi * 1500.0 * np.arange(int(seconds * FS)) / FS)
    print(f"  TUNE: {seconds:.0f}s carrier at 1500 Hz, drive {drive:.2f}, "
          f"RFPOWER 0.10 -- match the ATU now", flush=True)
    rig.set_power(0.10)
    try:
        ota._play(tone, out_dev, seconds + 5, rig, settle=0.10)
    finally:
        rig.ptt(False)
        rig.set_power(0.50)
        print("  TUNE done, RFPOWER restored to 0.50", flush=True)


#: The top of this rig's RFPOWER scale in watts. The level is linear in watts
#: over a hundred on the FT-891 -- 0.6 reads back as 60 W, 0.5 as 50 -- so a
#: watts figure is a level and there is nothing above 1.00 to ask for.
MAX_WATTS = 100.0


class _PhasePower:
    """The rig's RF power, moved at the link's protocol boundaries.

    THE AUDIO DRIVE CANNOT DO THIS, which is the whole reason the class exists.
    On the FT-891 the ALC holds down the two-tone PACTOR-3 entry by its 3 dB
    crest, and passes a constant-envelope PACTOR-1 packet at whatever RFPOWER
    allows: 0.52 drive still put FSK out above 50 W on 2026-09-13 with the same
    setting keeping the entry packet well under it. So an arm that wants 15 W of
    PACTOR-1 and 60 W of PACTOR-3 has to write the rig's own level, and
    `--p1-drive` -- which shapes the audio and is left exactly as it was --
    cannot stand in for it.

    ONE WRITE A TRANSITION, AND NEVER IN FRONT OF A KEY. Every write rides a
    phase change read out of a receive window: the grant arrives a turnaround
    ahead of the entry packet's boundary (`ptc.PtcHost._take_grant`, which calls
    in before `arq.on_rx_grant` queues the entry), and a fallback or a teardown
    has no burst behind it at all. The write itself is a pipe write into the
    long-lived rigctl and the CAT transaction rides behind it, so what the cycle
    spends is tens of microseconds and what the rig spends is its own; neither is
    charged to the few milliseconds between a cycle's last read and its key. A
    level already held is not written again, which is what keeps a grant repeated
    every cycle, or a second fallback, off the port.

    A WRITE THAT DOES NOT GO IS NOT A REASON TO STOP TRANSMITTING. The level is
    the experiment's variable; the session is the experiment. A refusal says so
    in the transcript and the arm goes on at whatever the rig is holding.
    """

    def __init__(self, rig, *, p1: Optional[float], p3: Optional[float]):
        self.rig = rig
        self.levels = {Protocol.PACTOR1: None if p1 is None else p1 / MAX_WATTS,
                       Protocol.PACTOR3: None if p3 is None else p3 / MAX_WATTS}
        #: What this arm last put on the rig, and None until it has put anything
        #: there -- which is also what says whether there is a level to restore.
        self.level: Optional[float] = None
        self.baseline = self._read_back()

    def _read_back(self) -> Optional[float]:
        """What the rig was on before the arm touched it, or None if it did not
        say. Read through a one-shot before the first key, where closing and
        reopening the CAT port costs a session nothing."""
        try:
            level = float(self.rig.get_power())
        except (TypeError, ValueError):
            print("  RFPOWER: the rig did not answer `l RFPOWER`, so the level "
                  "it started on is unknown and nothing is restored at "
                  "teardown", flush=True)
            return None
        print(f"  RFPOWER {level:.2f} -> {level * MAX_WATTS:.0f} W at arm "
              f"start; restored at teardown", flush=True)
        return level

    def select(self, protocol: Protocol, why: str) -> None:
        """Put the rig on this protocol's level. [ptc.PtcHost phase hooks]"""
        level = self.levels.get(protocol)
        if level is None or level == self.level:
            return
        if self._write(level, f"{protocol}, {why}"):
            self.level = level

    def restore(self) -> None:
        """Give the rig back the level it was found on, after the key is down.

        Through a one-shot and AFTER `ota.Rig.stop`: stop closes the live rigctl
        and latches the door behind it, and nothing that runs while the
        transmitter may still be up is allowed to hold the port the unkey ladder
        needs.
        """
        if self.baseline is None or self.level is None:
            return
        if self.rig.set_power_once(self.baseline):
            print(f"  RFPOWER {self.baseline:.2f} -> "
                  f"{self.baseline * MAX_WATTS:.0f} W restored (the level the "
                  f"arm started on)", flush=True)
        else:
            print(f"  !! RFPOWER NOT RESTORED: the rig is left at "
                  f"{self.level * MAX_WATTS:.0f} W, not the "
                  f"{self.baseline * MAX_WATTS:.0f} W it started on", flush=True)

    def _write(self, level: float, what: str) -> bool:
        t0 = time.perf_counter()
        took = self.rig.set_power(level)
        ms = (time.perf_counter() - t0) * 1e3
        if not took:
            held = ("the level it is on" if self.level is None
                    else f"{self.level * MAX_WATTS:.0f} W")
            print(f"  !! RFPOWER {level:.2f} ({what}) DID NOT REACH THE RIG -- "
                  f"the arm goes on at {held}", flush=True)
            return False
        print(f"  RFPOWER {level:.2f} -> {level * MAX_WATTS:.0f} W ({what}), "
              f"handed to rigctl in {ms:.2f} ms", flush=True)
        return True


def _phase_power(args, rig) -> Optional[_PhasePower]:
    """The arm's power plan, or None when the operator named no levels.

    None is what pins the old behaviour: a session that was not asked to move
    RFPOWER never opens the subject, and the rig keeps whatever the launcher
    read back.
    """
    if rig is None or (args.p1_watts is None and args.p3_watts is None):
        return None
    return _PhasePower(rig, p1=args.p1_watts, p3=args.p3_watts)


def _preflight(in_dev, out_dev, rig, settle: float, *, key: bool,
               blocksize: int = 128) -> int:
    """The station facts a bench cannot settle, measured in fifteen seconds.

    The duplex transmit path was proved against a loopback device, which is not a
    USB codec sharing one clock with a radio. Four things stayed open, and each of
    them is the kind that shows up as a bad QSO rather than as an error:

      * whether this codec tolerates a persistently open duplex stream. A
        persistent OUTPUT-ONLY stream starved its input 16x on 2026-07-28 and was
        reverted; the argument that one stream owning the device cannot contend
        with itself is structural, and structural arguments are what this check is
        for.
      * whether it runs at 128 frames rather than its own 4096 default.
      * the ADC-to-DAC offset, which is derived at runtime and has only ever been
        seen on a loopback.
      * how long the rig stays deaf after unkeying, which sets the earliest peer
        answer we can hear at all and is a term in the cycle budget.

    Keying happens with NO modulation. On a data interface the audio IS the drive,
    so PTT with silence puts no power into the antenna -- it exercises the T/R
    path without transmitting. That is the only reason this is allowed to key
    without an ATU match or a listen-first.
    """
    print("== preflight ==", flush=True)
    ref = _LiveInput(in_dev)
    try:
        time.sleep(2.0)
        quiet = ref.read_ready()
    finally:
        ref.close()
    rms_ref = float(np.sqrt(np.mean(quiet ** 2))) if quiet.size else 0.0
    print(f"  capture-only : rms {rms_ref:.5f} over {quiet.size / FS:.1f}s, "
          f"{_loss_seen(ref) or 'nothing seen to go missing'}", flush=True)

    live = _LiveInput(in_dev, out_dev, blocksize=blocksize)
    ok = True
    try:
        time.sleep(2.0)
        seg = live.read_ready()
        rms_dup = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
        lat = live._lat
        print(f"  duplex       : rms {rms_dup:.5f} over {seg.size / FS:.1f}s, "
              f"{_loss_seen(live) or 'nothing seen to go missing'}, "
              f"{live.underruns} underruns, "
              f"block {live._blk}, ADC->DAC {lat} samples"
              f"{f' ({lat / FS * 1e3:.1f} ms)' if lat else ''}", flush=True)
        # The regression this exists to catch was 16x down, so half is a wide gate
        # that still cannot be passed by a starved input.
        if rms_ref <= 1e-5:
            print("  -- reference is silent, so the starvation gate did not run. "
                  "Re-run with the rig on a live band before trusting this.",
                  flush=True)
        starved = rms_ref > 1e-5 and rms_dup < 0.5 * rms_ref
        if starved:
            ok = False
            print(f"  !! INPUT STARVED: duplex rms is {rms_ref / max(rms_dup, 1e-9):.1f}x "
                  f"down on capture-only. Do not transmit.", flush=True)
        # ON THE COUNT, NOT ON THE DRIVER'S FLAG. This gate read `live.xruns`
        # alone, and that flag is zero on every capture-clock reading in the
        # record -- including the 31 short of the air. A preflight that asks it
        # alone clears a starved stream every time it is run, which is the
        # failure `_loss_seen` exists to end.
        lost = _loss_seen(live)
        if lost:
            ok = False
            print(f"  !! CAPTURE IS LOSING SAMPLES at block {live._blk}: {lost}. "
                  "Every sample index this session computes would be off the air "
                  "by that much and growing. Do not transmit.", flush=True)
        if live.underruns:
            ok = False
            print(f"  !! {live.underruns} output underruns -- the holes are in "
                  f"our OWN emission at this blocksize", flush=True)

        if key and rig is not None:
            # Which wire this is about to key, and how. The PTT port is derived
            # from --serial rather than given, and the two device names differ in
            # one character, so this is the line that confirms the assignment
            # instead of leaving it assumed.
            print("  keying       : "
                  + (f"CAT command on {rig.serial}" if rig.ptt_type == "RIG"
                     else f"{rig.ptt_type} on {rig.ptt_port} (CAT is {rig.serial})"),
                  flush=True)
            recov, actuate, reached = [], [], True
            for i in range(5):
                base = live.read_ready()
                floor = float(np.median(np.abs(base))) if base.size else 0.0
                # `rig.ptt` is a pipe write to a long-lived rigctl and returns in
                # microseconds, so timing the CALL measures nothing. What `settle`
                # has to cover is the rig ACTUATING, and that is observable from
                # here: keying mutes our own receiver, so the sample at which the
                # band noise collapses is the sample the rig went to transmit.
                keyed_at = live.sample_now()
                reached &= rig.ptt(True) is not False
                time.sleep(0.25)
                reached &= rig.ptt(False) is not False
                at = live.sample_now()
                time.sleep(0.6)
                # `after` starts at the reader's position, which is back at the
                # key-DOWN -- the whole keyed interval is still in front of it.
                # Recovery is measured from the unkey, so index into it.
                pos0 = live.pos
                after = live.read_ready()
                lvl = np.abs(after)
                # Down to a tenth of the quiet-band level is unambiguous: the mute
                # measured -50 dB, and band noise does not fade 20 dB in one block.
                if floor > 1e-5:
                    gone = np.nonzero(lvl[max(0, int(keyed_at - pos0)):]
                                      <= 0.1 * floor)[0]
                    actuate.append(gone[0] / FS * 1e3 if gone.size else float("nan"))
                # Where the level climbs back to the quiet-band median it had
                # before we keyed. Below that the receiver is still muted, and a
                # peer answering inside this window is one we cannot hear.
                env = lvl[max(0, int(at - pos0)):]
                back = np.nonzero(env >= max(floor, 1e-6))[0]
                recov.append(back[0] / FS * 1e3 if back.size else float("nan"))
            if not reached:
                ok = False
                print("  !! a PTT command did not reach rigctl -- keying is not "
                      "under this program's control", flush=True)
            if actuate and not np.all(np.isnan(actuate)):
                act = float(np.nanmedian(actuate))
                print(f"  PTT actuation: {act:.0f} ms median of {len(actuate)} "
                      f"(settle is {settle * 1e3:.0f} ms)", flush=True)
                if act > settle * 1e3:
                    ok = False
                    print(f"  !! the rig keys AFTER the audio starts -- the head of "
                          f"every burst is being cut. Raise this rig's settle to at "
                          f"least {np.ceil(act / 10) * 10 / 1e3:.2f}", flush=True)
            else:
                print("  PTT actuation: not measurable -- band too quiet to see the "
                      "receiver mute", flush=True)
            tr = float(np.nanmedian(recov))
            print(f"  T/R recovery : {tr:.0f} ms median of 5 "
                  f"(TR_SWITCH_S is {TR_SWITCH_S * 1e3:.0f} ms)", flush=True)
            if not np.isnan(tr) and tr > TR_SWITCH_S * 1e3 + 15:
                ok = False
                print(f"  !! slower than the budget assumes -- raise TR_SWITCH_S to "
                      f"{np.ceil(tr / 5) * 5 / 1e3:.3f} and re-read the budget line",
                      flush=True)
    finally:
        live.close()
        if rig is not None:
            rig.stop()
    print(f"== preflight {'OK' if ok else 'FAILED'} ==", flush=True)
    return 0 if ok else 1


_WRITER = None
# How long the end of a session will wait for the writer. Long enough for a
# queue of ordinary windows to land, short enough that a disk which has stopped
# answering costs a finished session a pause and not an evening.
CAPTURE_DRAIN_S = 10.0


class _CaptureWriter:
    """The thread the deferred WAV writes run on, and the tally that says so.

    The thread is a daemon because the alternative is a session that cannot be
    stopped while a write is stuck. That makes the shutdown someone else's job:
    an interpreter exiting does not wait for a daemon, so everything still
    queued goes over the side without a word. The tally here is what lets the
    end of a session say whether that happened.
    """

    def __init__(self) -> None:
        self._q = queue.Queue()
        self._idle = threading.Condition()
        self.queued = 0
        self.written = 0
        self.lost: list[str] = []
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def pending(self) -> int:
        return self.queued - self.written - len(self.lost)

    def put(self, item: tuple) -> None:
        with self._idle:
            self.queued += 1
        self._q.put(item)

    def _run(self) -> None:
        while True:
            path, *rest = self._q.get()
            note = None
            try:
                _save_capture(path, *rest)
            except Exception as exc:
                # Still swallowed -- a transmission in progress outranks a
                # recording, and a raise here would take the session with it.
                # But said out loud, on this thread, which is by construction
                # the one nothing is timed against.
                note = f"{path.name}: {exc}"
                print(f"  !! capture NOT written -- {note}", flush=True)
            with self._idle:
                if note is None:
                    self.written += 1
                else:
                    self.lost.append(note)
                self._idle.notify_all()

    def drain(self, timeout: float) -> int:
        """Wait for the queue to empty; answer with what never made it."""
        deadline = time.monotonic() + timeout
        with self._idle:
            while self.pending:
                if not self._idle.wait(max(0.0, deadline - time.monotonic())):
                    break
            return self.pending

    def report(self) -> str:
        line = f"captures written {self.written}"
        if self.lost:
            line += f", {len(self.lost)} LOST: " + "; ".join(self.lost)
        if self.pending:
            line += (f", {self.pending} still unwritten -- the writer ran out of "
                     f"time and these captures are not on the disk")
        return line if self.lost or self.pending else line + ", all on disk"


def _drain_captures(timeout: float = CAPTURE_DRAIN_S) -> str:
    """Wait for the queued captures to reach the disk, and say whether they did.

    Called from the session's own teardown -- AFTER the rig is unkeyed, never
    before -- so the operator gets the account beside the rest of the summary,
    and again from `atexit` for the exits that never reach that `finally`.
    Draining twice is free: the second call finds nothing pending.
    """
    if _WRITER is None:
        return "captures written 0, none were taken"
    _WRITER.drain(timeout)
    return _WRITER.report()


def _drain_at_exit() -> None:
    """The backstop, for an exit the session teardown did not run.

    A signalled session unwinds through `run`'s `finally` and has already
    drained by the time this fires. One that dies somewhere else -- a raise
    before the loop, a `sys.exit` from another entry point -- has not, and its
    captures are the ones worth the most.
    """
    if _WRITER is not None and _WRITER.pending:
        print("\n" + _drain_captures(), flush=True)


def _save_capture_async(path: Path, seg: np.ndarray, xruns: int = 0,
                        end: Optional[int] = None, lost: int = 0) -> None:
    """Queue a capture for writing instead of writing it here.

    A WAV write sits between the listening window closing and the next key-down.
    That gap used to be argued at 28 ms from a budget since retracted; it is really
    the whole 250 ms of clear air less the peer's answer, so there is more room
    than was thought. It is still not the place for a synchronous write: the file
    is wanted for analysis afterwards, not for the decision being made now, and a
    disk stall does not get to move a transmission.

    The levels still have to be measured on THIS thread, before the array can be
    touched, because they are the pre-normalisation numbers and the writer
    normalises. Only the file write is deferred.
    """
    global _WRITER
    if _WRITER is None:
        _WRITER = _CaptureWriter()
        atexit.register(_drain_at_exit)
    _WRITER.put((path, np.asarray(seg).copy(), xruns, end, lost))


def _save_capture(path: Path, seg: np.ndarray, xruns: int = 0,
                  end: Optional[int] = None, lost: int = 0) -> None:
    """Write a capture, and beside it the levels it had BEFORE normalisation.

    `session.write_wav` scales every file to 0.8 peak, so a capture that was
    hard-clipped off the air measures 0.00% railed once saved and looks pristine.
    The phase is already gone by then; the one number that explains a failed
    session is the one number the recording does not contain. That is not
    hypothetical -- two separate readers, including me, measured saved captures
    and concluded there had been no clipping, and were wrong both times.

    So the levels are taken here, on the audio as it came off the codec, and
    written to a sidecar. Analysis later reads the sidecar, not the WAV.
    """
    seg = np.asarray(seg)
    peak = float(np.abs(seg).max()) if seg.size else 0.0
    stats = {
        "samples": int(seg.size),
        "seconds": round(seg.size / FS, 3),
        "peak": round(peak, 5),
        "rms": round(float(np.sqrt(np.mean(seg ** 2))), 5) if seg.size else 0.0,
        # Against full scale, NOT against this capture's own peak: clipping is a
        # property of the converter, and measuring it relative to the peak makes
        # every capture look 100% railed at its maximum.
        "railed_pct": round(float((np.abs(seg) > 0.995).mean()) * 100, 3)
        if seg.size else 0.0,
        "normalised_on_write": True,
        # THE TWO WAYS THIS STREAM LOSES SAMPLES, and the driver only knows one,
        # so the count leads and the flag follows it. `lost` is what the
        # converter timestamped and a starved interpreter never accepted, which
        # is the loss this station actually has -- 31 of the 68 capture-clock
        # readings on this record are short of the air, one of them by 47.7 s of
        # its own 86.8. `xruns` is what the driver DECLARED, and it is zero on
        # all 68 and on all 450 windows that lost audio; read first, it says
        # clean over every one of them. `lost` is the WHOLE stream's total at
        # this window's close: once the stream is spliced every later index is
        # off the air too, so a window with a clean interior is not on the air's
        # clock.
        "lost_samples": lost,
        "xruns": xruns,
        # Both counters condemn and neither acquits, so this is named for what was
        # SEEN and not for a valid grid -- which is what it used to claim, as
        # `grid_valid`, over 45 spliced recordings. `lost` is anchored on
        # `t.inputBufferAdcTime` and cannot see loss upstream of it. Only a
        # second receiver could, and the one witnessed arm does not: corrected
        # for the aligner's 8300 ppm framing error it reads +260 ms where this
        # counter justified 295.
        "grid_loss_seen": bool(xruns or lost),
        # Where this window's LAST sample sits on the capture stream -- the same
        # clock the grid boundaries and every logged burst sample live on. The
        # 2026-08-03 post-mortem had to reconstruct this from the collect
        # arithmetic before a single burst could be timed; one integer makes
        # every capture self-pinning instead.
        "end_stream_sample": end,
    }
    session.write_wav(str(path), seg)
    path.with_suffix(".json").write_text(json.dumps(stats) + "\n")


def _assert_capturing(seg: np.ndarray, live, n0: int) -> None:
    """Stop the session if the capture stream has stopped delivering.

    A window that returned no audio AND no new samples is not a quiet band, it
    is a dead receiver, and from the inside the two are indistinguishable
    without this check: the grid arithmetic stays perfectly self-consistent
    while `pos` never moves, so the boundaries march off into the future and
    every window reads empty while the log keeps saying "nothing decoded".

    It has happened, and it cost a session: the input stalled 20 ms after it
    opened and eleven connect bursts went out into a receiver that was not
    listening, on a frequency where the same station had answered minutes
    earlier. Deaf-and-transmitting is the worst state this program can be in --
    it is unusable to the far end and it occupies a channel -- so it stops
    rather than carrying on politely.
    """
    if seg.size or live.samples != n0:
        return
    raise SystemExit(
        f"CAPTURE STREAM DEAD: no audio and no new samples in a whole window "
        f"(stalled at sample {live.samples}, {live.xruns} xruns). The receiver "
        f"is not listening, so transmitting again would be calling into a "
        f"channel we cannot hear. Stopping.\n"
        f"  It has happened once, when the capture stream asked the device for "
        f"a non-default buffer size; see the note in _LiveInput.")


def _listen_until_answer(live, want: int, host, sessrx, feed_n: int,
                         slice_s: float = 0.25, *, forget_burst: bool = True):
    """Collect `want` SAMPLES of audio, stopping the instant a reply is decoded.

    Counted in samples, not seconds, because this is what holds the raster: the
    caller passes the sample index of the next grid boundary and the window ends
    when the codec has delivered it. Nothing here sleeps or reads a clock.

    Every slice is handed to the session decoder as it arrives, so an event is
    delivered a slide after the audio carrying it rather than after the window --
    for the last `feed_n` samples of the window, which is as far back as an event
    can still be answered in this cycle. What is in front of that is collected and
    HELD, for the reason and at the cost the loop below states.

    `slice_s` looks like free latency and is not. Polling costs a mean 210 ms to
    notice a burst at 250 ms and 65 ms at 100 ms, so shrinking it is tempting --
    but it was measured across the negative corpus and it is a bad trade twice
    over. A 100 ms WINDOW fires on pure noise 13 times in 252 slices where a
    250 ms window fires 0 in 101: less audio is worse statistics, not just less
    delay. And holding the window at 250 ms while hopping every 100 ms still
    triples false fires, because more tests means more threshold crossings.
    A false fire breaks the listen early, truncating the capture to ~0.35 s --
    and a real reply arrives 0.49-0.85 s in, so we would miss it and send another
    connect instead of the acknowledgement. Missing the reply costs a whole
    cycle; the polling delay costs 145 ms. Leave it at 250 ms.
    """
    # Forget any earlier cycle's burst. The stamp is what the ACK latency is
    # measured against, and the cheap start-of-burst test can fire on a cycle
    # that never produces an acknowledgement -- so a stamp left lying around
    # makes the next cycle's ACK look a whole cycle late when it was not. That
    # is a broken instrument, and it reported 2.489 s against a 1.25 s budget.
    #
    # ONCE PER CYCLE, THOUGH. `_regrid` listens again inside the same cycle, and
    # clearing here would wipe the onset the main window recorded -- degrading
    # the ACK-latency reading on exactly the cycles that misbehaved.
    if forget_burst:
        host._burst_at = None
        host._burst_sample = None
    if 0 < feed_n < want:
        # Not an alarm, and not a loss. This used to read "only the 1.25 s in
        # front of the key can be decoded as it arrives. The rest goes to the
        # flush -- ... every cycle it could have answered has gone", and on
        # 2026-08-11 a live session read exactly that as the receive window
        # discarding the gateway's answer. The excess is the stale sliver at
        # the FRONT of the window, and holding moves a decode rather than
        # dropping a sample: held slices ride the same rolling
        # buffer as the fed ones and are decoded with the first fed slice, the
        # reply test watches every slice either way, and the end-of-cycle scans
        # read the whole window.
        print(f"    [grid] {BEHIND_THE_GRID}: this window holds "
              f"{want / FS:.2f} s of channel, "
              f"{(want - feed_n) / FS * 1e3:.0f} ms more than the cycle a "
              f"station answers in. The excess at the front is held, not "
              f"dropped: the frame scan reads the whole window, the reply test "
              f"watches every slice, and the anchored read is aimed inside it. "
              f"Held audio is let go of at the crossing to the fed tail, so a "
              f"slide never re-reads it; holding moves a decode and discards "
              f"nothing. "
              f"A window longer than the cycle means the last key did not "
              f"land on the next slot -- {SHORT_OF_THE_FLOOR} and "
              f"{SLOT_GONE} say which.",
              flush=True)
    slice_n = int(slice_s * FS)
    got, n = [], 0
    while n < want:
        chunk = live.read(min(slice_n, want - n))
        if chunk.size == 0:
            break
        got.append(chunk)
        if sessrx is not None:
            # FED ONLY WITHIN `feed_n` OF THE KEY, and this is what stops a slow
            # cycle from becoming a stopped one. The rolling decoder is what makes
            # an event arrive a slide after the audio carrying it, and that is
            # worth its cost only while there is still time to answer in: it
            # re-decodes a whole 4 s window every 0.25 s slide, which on real
            # off-air audio costs more than the audio lasts (`FEED_MAX_SLOTS`).
            # Audio further back than a cycle cannot be answered this cycle
            # whatever it holds, so it is HELD instead -- in the stream, in the
            # same buffer, read by the frame scan over the whole window, at a
            # cost the cycle can pay. The vocabulary is `_collect`'s and so is
            # the reasoning: held, never fed.
            #
            # ...AND LET GO OF WHERE THE FEED BEGINS, once. Held audio rides
            # the buffer the rolling decoder reads, so left in front of the fed
            # tail it is re-decoded on every 0.25 s slide: 1026 ms of rolling
            # feed over the 2.507 s window a handed-back slot builds, against
            # 290 ms over the 1.250 s one, measured on the 2026-09-13 turn. The
            # scans have the whole window either way -- `_scan_frame` below and
            # the anchored read in the caller -- and `live.RollingRx.push` drops
            # what is in front of it.
            #
            # `feed_n` of zero holds the whole window, which is what `_regrid`
            # asks for: see there.
            if want - n <= feed_n:
                sessrx.feed(chunk)
            else:
                sessrx.bridge(chunk)
        n += chunk.size
        # Test the SLICE, not the accumulation, and with the cheap burst-under-way
        # test: the full one cannot pass on a short slice, so using it here meant
        # the loop never broke and the acknowledgement went out ~1.5 s after the
        # peer's burst -- measured -- against a 1.25 s cycle.
        #
        # EVERY PHASE, and only the early break below is the connect's. The stamp
        # is what `RadioTx.send_p1_cs` measures the acknowledgement against, and
        # taking it only while CONNECTING left a held link with no stamp at all --
        # so the line fell through to `reply_at`, which the hold loop never sets,
        # and reported the seconds since the CONNECT phase ended. Three sessions of
        # 2026-08-09 read "the ack is a median 11 s late" off that; the numbers step
        # by exactly one cycle each, which is all they were counting. It costs
        # 0.07 ms a slice.
        if (getattr(host, "_burst_sample", None) is None
                and rxfront.p1_reply_starting(chunk)):
            host._burst_at = time.time()       # when the tone pair came up
            # ...and WHERE, which is the better instrument. An energy test on a
            # clock has the whole scheduler in it; a sample index is exact, and
            # our own transmission's end is a sample index too, so reply latency
            # is a subtraction rather than an estimate.
            host._burst_sample = live.pos
            if host.arq.state == State.CONNECTING:
                # Heard it start -- now capture the REST of the burst before
                # stopping. Breaking dead here leaves too little audio for the
                # full test to confirm, so the FSM never sees the reply and sends
                # another connect instead of the acknowledgement.
                #
                # NOT PAST THE END OF THE WINDOW. `want` is where the caller has
                # to key, and reading beyond it walks straight through a deadline
                # the bridge below cannot then give back -- measured on a replay
                # at exactly +180 ms, P1_BURST_S, on every cycle where the burst
                # landed in the last slice. Nothing is lost by stopping: the
                # caller bridges to the boundary and pushes what arrives in the
                # meantime through the same decoder, so the only audio this cap
                # gives up is the audio after our own PTT goes up, which a keyed
                # rig could not have heard anyway.
                tail = live.read(min(int(P1_BURST_S * FS), want - n))
                if tail.size:
                    got.append(tail)
                    if sessrx is not None:
                        sessrx.feed(tail)
                break
            # A held link listens the window out: its early break belongs to the
            # connect, where the thing being waited for is one short burst.
    return np.concatenate(got) if got else np.zeros(0, np.float32)


def _read_codeword_at_bursts(sessrx, seg: np.ndarray, seg_start: int,
                             onsets: list[int]) -> None:
    """Read the peer's codeword where the detector actually found a burst.

    THE LAST WAY A CODEWORD REACHES THE FSM, and the cheap one. Three paths run
    ahead of it and each has a cycle it cannot cover: the rolling feed only sees
    the audio the listen loop fed, which is none of it while we are receiving;
    the anchored read needs an instant, and a grid with no `d` has none to aim
    at; the flush keeps `FLUSH_CONTEXT_S` of the buffer and a long window is
    mostly in front of that.

    So on the cycles the grid goes blind -- which are exactly the cycles a peer
    is trying to be heard on -- nothing read a codeword at all. WS8EOC's
    `hold_23` of 2026-08-26: `0x59A` at zero bit errors 969 ms into a 2.088 s
    held window, 17-29 dB over its own guard bands, with the burst detector
    putting an onset on it to the millisecond, and the session logged the cycle
    quiet. Eleven of the sixteen grants our own recordings hold sit in windows
    of that shape.

    ON THE ONSETS THE CYCLE ALREADY HAS, which is what makes it affordable and
    what keeps it honest. `evidence_of.record` runs the burst detector once per
    cycle for the fold below; this adds 0.2-0.6 ms of `cs_anchored` per window
    on top, against 24-434 ms for a sweep of the same audio -- and every one of
    those milliseconds is a sample the capture may not survive (measured
    2026-08-26: 23 of 23 capture-loss intervals sat inside a decode holding the
    interpreter). It is also aimed rather than swept: a read
    at a detected burst, at zero bit errors, with the two slots after the word
    required quiet, and no alignment search.

    ONE CYCLE LATE, which is the flush's own bargain and for the flush's own
    reason: the onsets do not exist until the window is closed, and the key has
    gone by. The peer repeats a codeword nobody answered; that is what the
    reverse channel is.
    """
    for at in onsets:
        # ...and not the burst the flush or the anchor has already read. They
        # decode the same audio and `cs_seen` does not cover the unassigned
        # words, so without this a grant reaches the host twice in one cycle and
        # the count of how many times the peer asked is a count of our readers.
        if any(abs(at / FS - t) < 2 * spec.P1_CS_S for t in sessrx.words_at):
            continue
        sessrx.control_signal(seg, seg_start, at)


def _forecast_next_key(sessrx, tx, raster, seg_start: int) -> None:
    """Where our next carrier lands against the peer's next codeword.

    THE FORECAST HAS TO NAME SOMETHING WE READ. This was `_report_collision`'s
    `[predict]`, and it stood on `max(onsets)` -- an energy shape on the
    1400/1600 bins that the corpus has firing on VARA, on PACTOR-2 and on a
    500 Hz-class ARQ station, extrapolated one cycle forward on the assumption
    that whatever it was repeats on our raster. That is three claims about a
    source it declined to identify, printed live, while the operator is deciding
    whether to stop the slot.

    MEASURED AGAINST A THIRD PARTY'S RECEIVER, 2026-08-26. It printed on 22 of
    that session's cycles and alarmed on exactly one of them: `we key -11 ms
    after it ends -- INTO IT`, about TX[19],
    a cycle the KiwiSDR at Empire puts clear by 113 ms in front and 184 ms
    behind. Wrong by 124 ms, and it named the one rung of that session that never
    collided; on the same cycle the grid logged `no control signal where it is
    due`, so its own conditions did not corroborate it. Of the twenty codewords
    the same recording shows we DID transmit over it said nothing, and it could
    not have -- the rig mutes this receiver while we key, so those bursts are in
    no capture of ours and produce no onset. A narrator that can only speak about
    the cycles nothing went wrong on has to be right about those.

    So it prints only where an event carrying a twelve-bit word read at zero
    errors sits in the window just closed. `nxt + cs_n` is then the peer's own
    codeword rather than an assumed one, and the source is named.

    AND THE LEAD IS MEASURED. `boundary - tx_key_up` is where the carrier
    actually came up on the last keying, off the same converter the onsets are
    counted on; the nominal settle it used to subtract is 12 ms further out than
    this station keys, which is the whole of the -11 ms it printed.

    Still a forecast: it assumes the peer repeats on its own raster, which is
    what a reverse channel does and not what it promises. The line narrates; the
    WORD it read is handed to `_MasterGrid.note_peer_codeword`, and the refusal
    that stands on it is `_iss_refusal`, bounded there for exactly this reason.
    """
    ev = next((e for e in reversed(sessrx.cs_log)
               if round(e.t * FS) >= seg_start), None)
    if ev is None:
        return
    host = getattr(tx, "host", None)
    if (host is not None and getattr(host, "protocol", None) == Protocol.PACTOR3
            and not host.arq.entry_pending and ev.protocol == Protocol.PACTOR1):
        # Keep the lower-protocol event in the transcript, but do not let an
        # ignored word replace confirmed P3 duration or phase evidence.
        return
    # The PEER's codeword, which is what `raster.cs_n` is not: that one is ours,
    # and on an upgraded link the two are 210 ms and 120.
    burst = raster.p1_cs_n if ev.protocol == Protocol.PACTOR1 else P3_CS_N
    at = round(ev.t * FS)
    # A WORD WE DECODED WAS NOT UNDER OUR OWN CARRIER. The rig mutes this
    # receiver while we key, so a codeword read at zero errors is proof the
    # transmitter was off while it was on the air. Where the grid puts one inside
    # our own last keying, the grid is out of step with the file rather than the
    # peer being early -- one session lost 298 ms of capture and walked the
    # answer instant 133 ms against a grid whose own off-grid line read +/-2 ms
    # every cycle -- and a projection off that instant is not evidence about the
    # next one. The narration below still runs; only the refusal declines to
    # stand on it. `captures/onair-0819-2210` is the recording this is measured
    # on: every codeword there decodes cleanly and the grid places all of them
    # 124 ms inside our own packet.
    #
    # RECORDED AHEAD OF THE TWO INSTANTS BELOW, because the refusal is about the
    # word and the narration is about the lead: a session that has not keyed yet
    # has no measured lead and still has a peer.
    if not any(lo <= at <= hi for lo, hi in getattr(tx, "keyings", ())):
        arq = getattr(getattr(tx, "host", None), "arq", None)
        raster.note_peer_codeword(at, burst, _cs_name(ev),
                                  getattr(arq, "dxcall", None), protocol=ev.protocol)
    if tx.tx_key_up is None or tx.boundary is None:
        return
    nxt = round(ev.t * FS) + raster.slot_n
    # ...and the nominal settle where that pair cannot be a lead. A replay keys
    # nothing, so its two instants are an arithmetic identity rather than a
    # measurement, and a figure outside one settle-to-a-cycle is not one either.
    lead_n = tx.boundary - tx.tx_key_up
    if not 0 < lead_n < raster.slot_n:
        lead_n = round(tx.settle * FS)
    key = raster.boundary_after(nxt) - lead_n
    lead = (key - (nxt + burst)) / FS
    print(f"    [predict] the {_cs_name(ev)} we read at {ev.t:.2f} s, repeated "
          f"on its own raster, ends {abs(lead) * 1e3:.0f} ms "
          f"{'before' if lead >= 0 else 'AFTER'} our carrier comes up -- "
          f"{'clear' if lead >= 0 else 'INTO IT'}", flush=True)


def _collect(live, sessrx, seg: np.ndarray, seg_start: int,
             until: int) -> tuple[np.ndarray, int, float]:
    """Wait for capture sample `until`, take what the holdback left, and hold it.

    THE BRIDGE, and a keyed cycle spends it twice. Once where the PEER stopped
    transmitting, which is `d` before our boundary and is the last instant the
    frame scan can learn anything -- everything after it is our own turnaround.
    Once at the key instant itself, for the anchored control-signal read, which
    wants the codeword's last block and costs 0.3 ms to run. What goes between
    the two is the cycle's decode; what goes after the second is arithmetic.

    HELD, never fed. `_SessionRx.feed` decodes, the FSM answers a decode from
    `on_rx_event`, and that would key from inside a window the loop has not
    finished with. The next flush reads it.

    NOT CONCATENATED ACROSS A TRANSMISSION. Anything above may have reached the
    FSM and keyed, and `_tx` empties the capture queue for the length of our own
    burst -- so a tail collected after one belongs to a different window, and
    splicing it on would put every offset taken from `seg_start`, the anchored
    read among them, a whole burst out.
    """
    ms = live.wait_until(until)
    tail = live.read_ready()
    if not tail.size:
        return seg, seg_start, ms
    sessrx.bridge(tail)
    origin = live.pos - tail.size
    if seg.size and origin == seg_start + seg.size:
        return np.concatenate([seg, tail]), seg_start, ms
    return tail, origin, ms


def _scan_frame(sessrx, audio: np.ndarray, origin: int, *,
                tracked_only: bool = False, upgrade: bool = False,
                p3_row0: Optional[int] = None) -> None:
    """Give the cycle decoder the origin of this particular contiguous window.

    Keep deep_scan's one-argument seam for existing replay clocks. The origin
    is scoped to this call; a rolling decoder must never inherit it.
    """
    probe = getattr(getattr(getattr(sessrx, "host", None), "peer", None), "p4_probe", None)
    if probe is not None and probe.requested:
        return  # Full receive audio is retained for offline P4 analysis.
    old = (getattr(sessrx, "_scan_origin", None),
           getattr(sessrx, "_tracked_only", False),
           getattr(sessrx, "_p3_target_row0", None))
    sessrx._scan_origin, sessrx._tracked_only = origin, tracked_only
    sessrx._p3_target_row0 = p3_row0
    before = getattr(sessrx, "_p3_delivered_at", None)
    started = time.perf_counter()
    try:
        if upgrade:
            sessrx.upgrade_scan(audio)
        else:
            sessrx.deep_scan(audio)
    finally:
        sessrx._scan_origin, sessrx._tracked_only, sessrx._p3_target_row0 = old
        stages = getattr(sessrx, "_p3_stage_ms", None)
        if stages is not None:
            name = "upgrade" if upgrade else "tracked" if tracked_only else "scan"
            stages[name] = stages.get(name, 0.0) + (time.perf_counter() - started) * 1e3
    delivered = getattr(sessrx, "_p3_delivered_at", None)
    if tracked_only and delivered is not None and delivered != before:
        tx = getattr(sessrx.host, "peer", None)
        slot = getattr(tx, "slot", None)
        if slot is not None:
            sessrx._p3_prekey_crc = (delivered, slot)


def _p3_current_packet_crc(sessrx, raster, slot: int) -> bool:
    """Fresh pre-key CRC for this physical packet and this reply opportunity.

    A previous-window delivery or a recovered slot cannot spend this proof.
    In particular, frame_seen alone is not evidence about the current packet.
    The peer's measured packet must end before this reply and belong to this
    cycle. No header-only reading can suppress the long-body standdown.
    """
    host = sessrx.host
    delivered = getattr(sessrx, "_p3_delivered_at", None)
    proof = getattr(sessrx, "_p3_prekey_crc", None)
    if (proof is None or proof != (delivered, slot)
            or host.protocol != Protocol.PACTOR3 or host.arq.role != IRS
            or host.arq.state not in LINKED or host.arq.entry_pending):
        return False
    peer = getattr(raster, "_p3_peer", None)
    if peer is None or peer[0] not in (
            delivered, delivered - p3frame.DATA_OFFSET * rxfront.SPS):
        return False
    gap = raster.boundary(slot) - peer[0]
    return peer[1] <= gap < peer[2]


def _window_swept(host, flowing: bool) -> bool:
    """Whether last cycle's window has already been read to its end.

    NOT `flowing`, which says a frame reached the FSM last iteration, and the
    two were one predicate. On PACTOR-3 they are different facts: the pre-key
    scan is `tracked_only` and its window is closed by `_p3_frame_ready` on a
    projected head, so it confirms a frame rather than reading the window out.
    A delivery in the free top-of-cycle sweep sets `frame_seen` for the cycle
    that follows, and that cycle's OWN window then never reached a sweep --
    which is why no live PACTOR-3 delivery in `onair-0913-0014` or
    `onair-0912-2321` lands on two consecutive cycles against a peer keying
    every one of them: 5->6, 8->9, 10->11, 12->13, 18->19, 20->21, 25->26,
    28->29, 30->31, 33->34, and A5's 2->3 and 29->30, twelve windows holding a
    CRC-valid copy apiece.

    A PACTOR-1 pre-key scan IS the window's complete read -- `tracked_only`
    means nothing to `decode_expected_p1_packet` -- and P1 has no
    absolute-position watermark to refuse a second delivery of one packet, so
    there the sweep stays suppressed exactly as it was.
    """
    return flowing and host.protocol != Protocol.PACTOR3


def _scan_previous_window(sessrx, audio: np.ndarray, origin: int, *,
                          sending: bool, flowing: bool) -> bool:
    """Read retained audio before the next wait, including an unaccepted turn.

    A pending local CS3 does not make the peer an IRS. Its old data packets
    can occupy the whole listening window, not our predicted control slot.
    Give that window the same early scan an IRS gets; do not put a blind
    packet search into the final pre-key reserve or change any TX permission.
    """
    host = sessrx.host
    pending_turn = (host.protocol == Protocol.PACTOR3
                    and host.arq.unconfirmed_breakin)
    if (not audio.size or (sending and not pending_turn)
            or _window_swept(host, flowing)):
        return False
    _scan_frame(sessrx, audio, origin)
    return sessrx.frame_seen


def _p3_cs6_pending(host) -> bool:
    """Is a long-cycle command out and unanswered? Then a probe may be owed.

    THE CALLER STILL READS ITS OWN SLOT OUT. `_regrid` used to hand its recovered
    window straight here unscanned, on the grounds that the helper reads each
    probe itself -- but the helper skips the read at its FIRST probe, which is
    the caller's own slot, precisely because the caller is expected to have done
    it. Between the two the recovered slot was never decoded at all, so the
    helper always advanced at least one probe and the peer's packet in that slot
    answered a boundary one further on. There is no duplicate to save: the read
    lands at the same point in the cycle either way.
    """
    arq = host.arq
    return (host.protocol == Protocol.PACTOR3 and arq.role == IRS
            and arq.cycle_request is not None and arq.cycle_command_emitted)


def _p3_unexpected_long_window(live, raster, tx, host, sessrx, slot, audio,
                              origin, settle_n):
    """Respect an observed SCS long body without assuming a negotiated veto.

    A header buys one bounded listen, not an ACK or a new clock. Only the
    ordinary CRC delivery path can adopt its geometry. Re-reading the same
    candidate cannot extend this window, and the HOLD budget counts its slots.
    """
    sessrx._p3_long_crc_reply = None
    arq = host.arq
    short = round(spec.CYCLE_SHORT_S * FS)
    if (host.protocol != Protocol.PACTOR3 or arq.role != IRS
            or arq.state not in LINKED or arq.entry_pending or arq.cycle_long
            or getattr(sessrx, "_p3_cycle_n", None) != short
            or getattr(sessrx, "_p3_row0", None) is None or not audio.size):
        return slot, audio, origin
    if _p3_current_packet_crc(sessrx, raster, slot):
        return slot, audio, origin
    target = _p3_wideband_target(sessrx, origin + audio.size)
    delivered = getattr(sessrx, "_p3_delivered_at", None)
    checked = getattr(sessrx, "_p3_long_window_checked_at", None)
    if (target is None or (delivered is not None and target <= delivered + rxfront.SPS)
            or (checked is not None and target <= checked + rxfront.SPS)):
        return slot, audio, origin
    lead = p3frame.DATA_OFFSET * rxfront.SPS
    lo = max(0, target - origin - lead - 2400)
    at = target - origin - lo
    if at < lead + rxfront.SPS // 4:
        return slot, audio, origin
    started = time.perf_counter()
    prefix = audio[lo:min(audio.size, target - origin + round(.05 * FS))]
    header = sessrx.sync.wideband_header_at(sessrx._corrected(prefix), at)
    stages = getattr(sessrx, "_p3_stage_ms", None)
    if stages is not None:
        stages["long_header"] = stages.get("long_header", 0.0) + (time.perf_counter() - started) * 1e3
    if header is None or not header.long_cycle:
        return slot, audio, origin
    first = origin + lo + header.at + lead
    sessrx._p3_long_window_checked_at = first
    levels = [sl for sl in header.levels if sl >= 3]
    # The header already fixes the alignment. Retain the last sparse matched-
    # filter sample (inclusive), not the unused tracking margin after it.
    end = first + max(rxfront._frame_span(p3rx.path_for(sl, header))
                      - rxfront.UNREAD_TAIL_N + 1 for sl in levels)
    # The candidate belongs to the current short reply. At most two further
    # short slots are withheld, independently of subsequent header guesses.
    endpoint = slot + 2
    tx.cancel_pending_cs()
    print(f"    [p3] long header at {first / FS:.3f} s, fit {header.fit:.3f}; "
          f"withholding short reply through slot {endpoint}, CRC still required",
          flush=True)
    audio, origin, _ = _collect(live, sessrx, audio, origin, end + live.holdback)
    # Callback delivery can be coarser than the configured block/holdback.
    # Read the remaining sample count explicitly, rather than guessing another
    # delivery quantum. A short read leaves CRC gated below; never join a gap.
    if live.pos < end:
        tail_start = live.pos
        tail = live.take_until(end)
        if tail.size:
            sessrx.bridge(tail)
            if audio.size and origin + audio.size == tail_start:
                audio = np.concatenate([audio, tail])
            else:
                audio, origin = tail, tail_start
    tx.aim(raster, endpoint)
    start = first - origin
    if origin <= first - lead and origin + audio.size >= end:
        # One retained header, at most two header-aliased speed levels. No
        # blind position ladder, no zero-padding of a missing long body.
        crop = max(0, start - lead - 2400)
        window = sessrx._corrected(audio[crop:end - origin])
        start -= crop
        h = replace(header, at=start - lead)
        delay = (rxfront._matched_filter().size - 1) // 2
        started = time.perf_counter()
        for sl in levels:
            path = p3rx.path_for(sl, h)
            idx = np.unique(np.add.outer(
                start + (np.arange(path.n_symbols + 1) - 1) * rxfront.SPS + delay,
                path.clock_offsets(rxfront.SPS)).ravel())
            Z = rxfront._sampled_baseband(window, path.tones, idx)
            rot, _ = p3rx.field_rotation(Z, start, path, h.rot)
            packet = p3rx.decode_at(window, start, sl, Z=Z, header=replace(h, rot=rot))
            if packet is not None:
                ev = rxfront._packet_event(packet, t=first / FS, tag=", retained long")
                previous = sessrx._p3_delivered_at
                sessrx._on(replace(ev, start=first))
                if sessrx._p3_delivered_at == first and previous != first:
                    sessrx._p3_long_crc_reply = (first, endpoint)
                sessrx.sync.packet_level = sl
                sessrx.sync.rotation = rot
                sessrx.p3_memory.clear()
                break
        if stages is not None:
            stages["long_body"] = stages.get("long_body", 0.0) + (time.perf_counter() - started) * 1e3
    line = raster.regear(arq.cycle_long)
    if line:
        print(f"    [grid] {line}", flush=True)
    return endpoint, audio, origin


def _p3_current_long_crc(sessrx, slot: int) -> bool:
    """The retained body delivered THIS reply's packet, not a previous scan.

    Its CRC includes the packet/turn status. A second control-word search in
    this same data window cannot add an owed answer, but can spend its deadline.
    The proof expires each HOLD iteration and names both physical row and slot.
    """
    proof = getattr(sessrx, "_p3_long_crc_reply", None)
    host = sessrx.host
    return (proof is not None and host.protocol == Protocol.PACTOR3
            and host.arq.role == IRS and host.arq.state in LINKED
            and proof == (sessrx._p3_delivered_at, slot))


def _p3_transition_window(live, raster, tx, host, sessrx, slot, audio, origin,
                          settle_n):
    """Resolve an emitted LONG command without transmitting inside its answer.

    Keep the confirmed short raster until a packet proves otherwise. Check each
    short opportunity for an old-cycle retry; if none decodes, retain the first
    possible long header and listen through its full frame. At most two extra
    short slots are collected. No header hypothesis alone authorizes a key.

    ONE WALK PER RECOVERY, AND `_regrid` IS WHAT COUNTS IT. Each probe reads to
    `P3_DECODE_RESERVE_S` of its own key and decodes behind that, so a walk taken
    inside a recovery leaves the slot it lands on late by exactly that decode.
    While the reply comb was masked to multiples of three this could not happen:
    `_keyable_slot` handed back to the endpoint and the walk was a no-op. Off the
    mask it is real, and taken again on every try it is a loop with no progress
    term -- three tries, twelve slots, and a `--hold` budget counted in cycles
    that the session never reaches.

    THE ONE PLACE A REPLY WAITS ON THE CYCLE LENGTH, and it waits on a READ. The
    first probe is the caller's own slot, so a peer still keying 1.25 s frames is
    answered where it is listening and nothing is given up: the loop leaves on the
    packet, whichever length its header carries, and `regear` then follows
    `arq.cycle_long` -- which `observe_peer_cycle` moves off a CRC-valid header
    and nothing else moves off our own CS6. Only a slot that came up EMPTY is
    spent walking to the endpoint, which is where a long body would have ended.
    """
    arq = host.arq
    waiting = lambda: _p3_cs6_pending(host)
    if not waiting():
        slot, audio, origin = _p3_unexpected_long_window(
            live, raster, tx, host, sessrx, slot, audio, origin, settle_n)
        if host.protocol == Protocol.PACTOR3 and arq.role == IRS:
            # The caller's initial scan may already have resolved the command.
            # Keep this reply slot, but reconcile the next-slot step before a
            # late-key recovery can advance inside a now-confirmed long frame.
            line = raster.regear(arq.cycle_long)
            if line:
                print(f"    [grid] {line}", flush=True)
        return slot, audio, origin
    cycle = round(spec.CYCLE_SHORT_S * FS)
    command_slot = raster._p3_command_slot
    if command_slot is None:
        # A simulated driver may supply ARQ state without a physical TX epoch.
        # Real RadioTx always records the epoch when CS6 leaves the seam.
        command_slot = slot - 1
    group = max(0, (slot - command_slot - 1) // 3)
    endpoint = command_slot + (group + 1) * 3
    first = raster._p3_command_row0
    if first is not None:
        first += group * 3 * cycle
    elif sessrx._p3_row0 is not None:
        first_slot = command_slot + group * 3 + 1
        first = sessrx._p3_row0 + max(
            0, (raster.boundary(first_slot) - sessrx._p3_row0) // cycle) * cycle
    for probe in range(slot, endpoint + 1):
        fresh = probe != slot
        if fresh:
            tx.aim(raster, probe)
            deadline = _p3_decode_deadline(live, tx.key_instant(raster, probe), settle_n)
            # TO THE FRAME'S OWN END, and it may not be shortened: this window
            # exists to read a body out, and a read that stops early inside one
            # returns nothing at all. What the scans behind it cost is bounded
            # by `_regrid`'s slot budget instead -- see `REGRID_TRIES`.
            audio, origin, _ = _collect(
                live, sessrx, audio, origin,
                _p3_frame_ready(sessrx, deadline, live.holdback))
        # Try the retained long header before a putative short header inside it.
        if probe == endpoint and first is not None:
            _scan_frame(sessrx, audio, origin, tracked_only=True, p3_row0=first)
        # ...and the short read only where this probe added audio to look in.
        # The caller hands in a window it has just read out at the same aim, so
        # on the first probe this is that read repeated: 16.2 ms of the 0909
        # overloaded-reader session's own budget, spent behind the collect and
        # charged to the keying settle, for a decode that cannot find anything
        # the first one did not.
        if (fresh or first is None) and waiting():
            _scan_frame(sessrx, audio, origin, tracked_only=True)
        if not waiting():
            break
    slot = probe
    line = raster.regear(arq.cycle_long)
    if line:
        print(f"    [grid] {line}", flush=True)
    print(f"    [grid] CS6 receive window ended at slot {slot}; "
          f"peer cycle {'long' if arq.cycle_long else 'short/unconfirmed'}", flush=True)
    return slot, audio, origin


def _regear_next_slot(raster, slot: int, long: bool) -> tuple[int, Optional[str]]:
    """A next-slot index was advanced using the previous cycle's length."""
    old_ticks = raster.ticks
    line = raster.regear(long)
    return slot + raster.ticks - old_ticks, line


def _p3_receive_geometry(sessrx) -> tuple[int, int]:
    """Observed geometry, or the short probe while a CS6 answer is unresolved."""
    span, cycle_n = sessrx._p3_span, sessrx._p3_cycle_n
    arq = getattr(getattr(sessrx, "host", None), "arq", None)
    if arq is not None and getattr(arq, "cycle_command_emitted", False):
        if cycle_n == round(spec.CYCLE_LONG_S * FS):
            span -= (placement.LONG_ROWS - placement.FRAME_SYMBOLS) * rxfront.SPS
        cycle_n = round(spec.CYCLE_SHORT_S * FS)
    return span, cycle_n


def _p3_wideband_target(sessrx, end: int) -> Optional[int]:
    """Latest short wideband row whose allowed partial tail is available."""
    row0 = getattr(sessrx, "_p3_row0", None)
    if row0 is None:
        return None
    span = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[3]))
    cycles = (end - row0 - span + rxfront.SyncedRx.WIDEBAND_EARLY_N) // round(spec.CYCLE_SHORT_S * FS)
    return row0 + cycles * round(spec.CYCLE_SHORT_S * FS)


def _p3_frame_ready(sessrx, deadline: int, holdback: int = 0) -> int:
    """Close a tracked receive window when its complete frame is DELIVERED.

    Waiting all the way to the PTT deadline wastes the packet's remaining
    turnaround. Keep the decoder's whole-frame span plus half a symbol of
    alignment slack, and spend the rest on the bounded tracked decode.

    ON WHAT THE CODEC HAS HANDED OVER, not on where the converter stands.
    `_collect` waits for a CAPTURE instant and a read returns once the block
    containing it has been DELIVERED, one `holdback` later -- which is that
    term's whole definition, and this window was sized without it. The audio the
    scan received therefore stopped 8.2 ms short of the span it had waited for,
    on every IRS cycle of `captures/onair-0913-2320`.

    AND WHAT THAT COST WAS THE ANCHOR AND NOT THE DECODE. `UNREAD_TAIL_N` of that
    span is past every instant the tracked grid and its matched filter touch, so
    a window stopping inside it loses nothing a read would have used -- all 182
    of that arm's cycles deliver the same 104 frames with this term paid in full,
    in part, or not at all, and where the shortfall did land is `_p3_packet`'s
    projection. So the term is a FLOOR rather than an extension: a rig whose
    holdback is deeper than the slack and the unread tail together would starve
    the reader, and one whose holdback is not waits exactly as long as it did.
    Listening past what the reader reads is not free -- it comes out of
    `P3_DECODE_RESERVE_S`, and a host already over that reserve loses the slot.
    """
    row0 = getattr(sessrx, "_p3_row0", None)
    if row0 is None:
        return deadline
    span, cycle_n = _p3_receive_geometry(sessrx)
    packet_span = rxfront._packet_span(span)
    target = row0 + ((deadline - row0 - packet_span) // cycle_n) * cycle_n
    if (cycle_n == round(spec.CYCLE_SHORT_S * FS)
            and getattr(sessrx, "sl2_prekey_expected", lambda _: False)(target)
            and not sessrx.host.arq.cycle_command_emitted):
        # The bounded SL2 reader needs the packet, not the old tracked
        # decoder's two trailing margin rows. Spend those 20 ms on its CRC
        # before the reply instead. Pay capture holdback and alignment slack.
        return min(deadline, target + packet_span + holdback + rxfront.SPS // 2)
    end = row0 + span
    cycles = (deadline - end) // cycle_n
    end += cycles * cycle_n
    # Alignment slack may use spare turnaround, but must not turn a complete
    # current packet into a request to revisit the previous physical cycle.
    ready = min(deadline, max(end + rxfront.SPS // 2,
                              end - rxfront.UNREAD_TAIL_N + holdback))
    if getattr(sessrx, "wideband_prekey_active", lambda: False)():
        target = _p3_wideband_target(sessrx, deadline)
        packet = rxfront._packet_span(rxfront._frame_span(placement.SPEED_PATHS[3]))
        # AND THE FILTER'S OWN REACH PAST THE LAST ROW. A matched filter reads
        # the window of audio ENDING at its grid instant, so a window closed on
        # the packet's last sample hands `_sampled_baseband` its group delay of
        # zeros and the last row or two are demodulated out of the padding
        # rather than off the air. The K0NTS holds of 2026-09-15 stopped 613-901
        # samples inside the field's last row and lost 28-49 ms that way -- worth
        # a few bytes a cycle on a field the CRC was refusing by a few bytes.
        # STILL BOUNDED by `WIDEBAND_EARLY_N` and by the deadline: this asks for
        # the support where the turnaround has it and reads short where it does
        # not, exactly as before.
        partial = (target + packet + rxfront.MATCHED_DELAY_N
                   - rxfront.SyncedRx.WIDEBAND_EARLY_N
                   + holdback + rxfront.SPS // 2)
        ready = max(ready, min(deadline, partial))
    return ready


def _callback_delivery_n(live) -> int:
    """Largest recent contiguous batch delivered together by the audio driver.

    The September 19 duplex stream supplied three 128-frame callbacks every
    8 ms. A blocking read can therefore overshoot by 384 samples even though
    the stream's negotiated block is 128. Count adjoining sample ranges whose
    wall timestamps are less than a quarter block apart; a lone delayed
    callback is not evidence of a larger delivery batch.
    """
    block = getattr(live, "_blk", 0)
    largest, batch, previous = block, 0, None
    for record in tuple(getattr(live, "_callback_timing", ())) [-32:]:
        start, frames, wall, _adc = record
        if (previous is not None and start == previous[0] + previous[1]
                and 0 <= wall - previous[2] < block / (4 * FS)):
            batch += frames
        else:
            batch = frames
        largest = max(largest, batch)
        previous = record
    return largest


def _prekey_lead(live, settle_n: int, owed_n: int = 0) -> int:
    """How far in front of the key instant a cycle's LAST READ stops.

    AND WHAT THIS PARTICULAR CYCLE STILL OWES, because the interval this leaves
    past `key_notice` is 7.6 ms on the FT-891 and a changeover cycle spends 8.2
    in it -- `changeover_packet` and its trim in front of a tick that assembles
    the B2F login, against a tick and a codeword render that measure 1.1 to 1.6.
    `arm-v23-A-40-ws8eoc` drained to this lead, came out of `_regrid` 1.1 ms
    late and reached `RadioTx._tx` at +9.1: eight changeovers refused at +5.1 to
    +14.8 ms while the gateway repeated ` via WS8EOC >` 28 times.
    `RadioTx.breakin_cost_n` is that interval measured on the stream's own
    clock, and it was already being kept -- for `_clamp_forgives` alone, which
    is the gate rather than the window.

    ONE NUMBER FOR THE READ DEADLINE AND FOR THE DRAIN, because the admission
    guard is the difference between this and `key_notice`. Both loops read to it,
    `RadioTx._tx` and `_regrid` drain to it, and what separates it from the
    guard's own deadline is `TX_ADMIT_RESERVE_S` plus the block the delivered
    count is quantised to. Computed apart -- the read on the PTT instant, the
    check on the DAC notice -- the gap was whatever the rig's settle happened to
    leave over the device's latency, which is not a measurement of anything.

    A STREAM WITH NO TRANSMITTER GETS NEITHER TERM. A replay has no DAC to give
    notice to and no clock to lose it on -- `_ReplayInput.clamp_late` asks only
    whether the reader has gone past the instant -- so the reserve buys nothing
    there and the settle alone is the deadline, exactly as it was.
    """
    if getattr(live, "transmit", None) is None:
        return settle_n
    # THE SETTLE CARRIES THE TRANSMIT CORRECTION AND SO DOES THE NOTICE, so what
    # this leaves past `key_notice` -- the budget for the tick, the render and
    # the drain -- is the same interval whether the correction is set or not.
    # Without it here the audio moves early and the drain does not, and the
    # station spends the correction out of its own PTT lead.
    return (max(settle_n + getattr(live, "tx_latency_n", 0),
                getattr(live, "key_notice", 0) + _callback_delivery_n(live))
            + round(TX_ADMIT_RESERVE_S * FS) + owed_n)


def _clock_line(live, settle_n: int) -> str:
    """The numbers every transmit deadline in this file is built from.

    PRINTED UNCONDITIONALLY, because the one arm that needed them ran without a
    preflight and the preflight was the only place they appeared. The 2026-09-11
    reconstruction had to infer `_blk` from the four-slot period of the refusals
    and quote every figure that depended on `_lat` at a documented value the
    session itself never recorded.
    """
    blk = getattr(live, "_blk", 0)
    lat = getattr(live, "_lat", 0) or 0
    notice = getattr(live, "key_notice", 0)
    hold = getattr(live, "holdback", 0)
    lead = _prekey_lead(live, settle_n)
    extra = getattr(live, "tx_latency_n", 0)
    return (f"clocks: block {blk}, ADC->DAC {lat} samples "
            f"({lat / FS * 1e3:.1f} ms) + {extra} samples "
            f"({extra / FS * 1e3:.1f} ms) of station transmit latency the "
            f"driver does not report, DAC notice {notice} "
            f"({notice / FS * 1e3:.1f} ms), holdback {hold} "
            f"({hold / FS * 1e3:.1f} ms -- the part of the PTT lead the key does "
            f"not get, because a read returns once the codec has DELIVERED its "
            f"last block). The cycle's last read stops {lead / FS * 1e3:.1f} ms "
            f"in front of the key and the admission check is taken "
            f"{(lead - notice) / FS * 1e3:.1f} ms later; that interval is the "
            f"whole budget for the tick, the render and the drain.")


def _p3_decode_deadline(live, audio_at: int, settle_n: int) -> int:
    """Latest tracked read before the DAC notice and a small decoder reserve.

    AND THE HOLDBACK IS ALREADY ON THIS SIDE OF IT, which is why it is not added
    here. `_collect` waits on the CAPTURE index and `_LiveInput.wait_until`
    "returns one input latency before that sample can be read", so the audio a
    read hands back stops a `holdback` SHORT of the deadline and `clamp_late` is
    taken on that same delivered count. The reserve a tracked read actually
    keeps in front of the admission check is `P3_DECODE_RESERVE_S` plus the
    holdback -- 19.8 ms on the FT-891's 660 -- and charging the holdback again
    costs the frame rather than buying room: on
    `fixtures/prekey-read-0913` it puts the deadline 13 samples inside the
    packet's own end, `_p3_frame_ready` floors a whole cycle back, and two of
    that arm's seven pre-key deliveries arrive in the next cycle's sweep
    instead. See `test_the_decode_reserve_already_holds_the_holdback`.

    PTT settle is a target, not the audio scheduling limit. A first pulse placed
    on time leaves slightly less settle on this rig; moving a whole read to the
    previous packet to preserve that target loses the current acknowledgement.
    The post-decode clamp still refuses an overrun rather than shifting the RF.
    """
    reserve = getattr(live, "key_notice", 0) + round(P3_DECODE_RESERVE_S * FS)
    return audio_at - reserve


def _p3_place_reply(raster, tx, slot: int) -> None:
    """Put the IRS reply comb on the packet it answers, while the cycle can move.

    WHERE THE GRID IS SETTLED FOR THE CYCLE, AND NOT AT THE EMIT. The answer
    slot is a fixed delay past the peer's packet (`_MasterGrid.p3_reply_shift`),
    so a comb standing LATE of it has to move BACKWARDS -- and by the time
    `RadioTx._send_p3_control` runs, the window in front of the key has been
    sized against the old comb and the clock is standing at
    `key_instant(slot) - settle`. A backward move larger than
    `settle + key_notice`, about 48 ms, then names a boundary already in the
    past: `_advance_aim` prints `the unrendered burst would miss boundary ...
    aiming at slot N+1`, the codeword goes out a whole cycle late, and the
    answer slot the ISS is reading stays empty. Roughly half the residue space
    is that direction, and the two placement scenes on file happened to need a
    +1.25 ms forward move apiece.

    Here the move costs nothing. Nothing downstream has measured a window yet,
    so the whole cycle -- the listen, the pre-key read, `_keyable_slot`'s
    search and the key itself -- is sized against the comb that was placed
    rather than against the one it replaced. The emit's own call then finds a
    residue rather than a placement.

    A no-op off the PACTOR-3 IRS path: `p3_reply_shift` returns None for every
    other protocol, for a sending grid, and for a peer packet older than
    `ONSET_MAX_CYCLES`.
    """
    shift = getattr(tx, "p3_reply_shift", raster.p3_reply_shift)
    moved = shift(slot)
    if moved is None:
        return
    print(f"    [grid] {moved}", flush=True)
    tx.aim(raster, slot)


def _keyable_slot(live, raster: _MasterGrid, slot: int, lead_n: int, *,
                  listen: bool = True, shift: Optional[bool] = None) -> int:
    """The first slot at or after `slot` we can still key on properly.

    THE ONE PLACE THE KEYABLE-SLOT ARITHMETIC LIVES. It stood in four spellings
    -- both loops' never-transmit-late guards, `_regrid`'s search and the emission
    path's backstop -- and they disagreed about the holdback, which is how the
    2026-07-27 session came to transmit every 2.5 s.

    Two terms. The carrier has to be placeable on the boundary with `lead_n`
    samples of our own still to spend first, and -- for a caller with a receive
    window to spend before it -- the boundary has to be far enough ahead to leave
    a window a control signal fits in. Placeability alone can land on a slot with
    nothing left to listen in, and the audio in front of a key is what `flush_to`
    takes.

    `raster.cs_n` grows to 210 ms on an upgraded link, and the slot grows with it:
    a PACTOR-3 ISS drops its carrier 810 ms into the cycle rather than 960, so
    where PACTOR-1 leaves 190 ms of listening against a 120 ms requirement,
    PACTOR-3 leaves 340 against 210. The floor is further from firing after the
    upgrade than before it.

    THAT ARITHMETIC IS ON `data_n`, AND WHAT GOES OUT IS THE RENDER. Nothing
    measures one against the other, so a burst longer than the cycle budgets for
    it -- the entry ladder's `burst` rung is a packet behind a 200 ms acquisition
    preamble, 1.074 s of audio leaving 136 ms against a 210 ms floor -- takes the
    cadence to one slot in two. A step taken for the listen floor SAYS SO; a step
    taken for placeability is already `SLOT_GONE` and `LATE_KEY` at the call
    sites.

    Placeable is `clamp_late(boundary - lead_n) == 0` -- the clamp run forwards,
    with the caller's own lead spent first. `lead_n` is the PTT settle still
    owed the rig plus whatever work is left, and it is what separates "the DAC
    can still emit here" from "we can still key here properly": without it the
    search accepts a boundary `key_notice` ahead, `transmit` finds the PTT
    instant already past, and the rig gets 32 ms of whatever settle it was
    configured for.

    MEASURE THE WINDOW ON THE AUDIO THAT IS ACTUALLY DECODED. `live.holdback` is
    held back from `_listen_until_answer` and then collected by the bridge,
    concatenated and handed to the decoder -- so it IS read, and subtracting it
    here asks for a control signal to fit in a window one block shorter than the
    one the decoder gets. On the one-slot cadence that is 105 ms against a 120 ms
    requirement, so the guard fired EVERY cycle and shrike transmitted every
    2.5 s for the whole 2026-07-27 session (measured off the capture lengths: a
    1.36 s listen window is unreachable on a one-slot cadence). §8.1 of
    docs/protocols/pactor/pactor1-timing.md says why that alone stalls a link:
    the peer reads at a fixed offset, an empty slot is a failed packet, and its
    timing loop runs `soft_time_dev` on the noise it finds there and drags its
    read anchor off ours.

    `shift` is for a caller holding audio that is already rendered. The FSK
    polarity is baked into the samples before the transmitter ever sees them, so
    a slot whose parity disagrees with it is not a slot this burst can go out in.

    AND THE SEARCH WALKS THE CYCLE'S COMB, `raster.ticks` at a step. On the long
    cycle every third slot is a cycle boundary and the two between them are
    inside the packet; a burst stepped onto one of those would key 1.25 s off the
    phase comb the peer is counting on, which is the desynchronisation the whole
    of `ticks` exists to prevent. One at a time on the short cycle, which is
    every arm flown.

    THE COMB IS THE PEER'S, AND AN OUTSTANDING CS6 IS NOT EVIDENCE ABOUT IT. The
    search used to mask every slot whose distance from our own COMMAND slot was
    not a multiple of three, on the theory that the other two might lie inside a
    long packet the peer had yet to send. WS8EOC on 2026-09-13 sent no such
    packet: across nineteen CS6 keyings the receiver read nineteen CRC-valid
    frames, every one of them short and every one on the unmoved 1.25 s raster,
    and the log holds no `cycle length ->` line at all. The mask meanwhile turned 43 of that stint's 71
    overruns into three-slot steps -- 80 to 83 to 86 to 89, four of them off an
    overrun under 5 ms -- handing 86 slots to a gateway that was keying a packet
    in each of them, and exhausting `REGRID_TRIES` thirteen times. `ticks`
    already keeps the search on the comb once a CRC-valid long frame has moved
    it, and until one has there is no long packet to step around:
    `_p3_transition_window` is the one place a reply waits on the cycle length,
    and it waits on a read rather than on our own request.
    """
    said = False
    while True:
        boundary = raster.boundary(slot)
        room = boundary - lead_n - live.pos
        if live.clamp_late(boundary - lead_n) == 0:
            if not listen or room >= raster.cs_n:
                if shift is None or raster.shift(slot) == shift:
                    return slot
            elif not said:
                said = True
                print(f"    [grid] {SHORT_OF_THE_FLOOR}: slot {slot} leaves "
                      f"{room / FS * 1e3:.0f} ms of channel in front of the key "
                      f"where a control signal is "
                      f"{raster.cs_n / FS * 1e3:.0f} ms, so this burst keys a "
                      f"slot on. What we key is longer than the cycle budgets "
                      f"for it.", flush=True)
        slot = raster.next_slot(slot)


def _finish_entry_candidate(live, raster, tx, host, sessrx, slot,
                            seg, seg_start, settle_n):
    """Finish one first-CS3 body before another entry can cover it.

    WS8EOC 2026-09-12 13:10: the head was in the normal answer window,
    its CRC-valid RMS body in the recovered slot. Scanning only the latter
    dropped the head; another entry then covered the next repetition.
    This hold is bounded by the candidate's own short-frame end. A false head
    changes no ARQ state and cannot extend the hold by being read again.
    """
    candidate = getattr(sessrx, "_p3_entry_body_candidate", None)
    if candidate is None:
        return seg, seg_start
    head, offset_hz = candidate
    sessrx._p3_entry_body_candidate = None
    sessrx._p3_entry_body_checked_at = head
    if (host.protocol != Protocol.PACTOR3 or not host.arq.entry_pending
            or host.arq.role != ISS or host.arq.state not in LINKED):
        return seg, seg_start
    # THE BODY ENDS AT THE PACKET, and the read is bounded by our own answer
    # instant -- measured from the same head, because the slot that answer keys
    # in is not settled until this body's CRC has reversed the grid. The 20 ms of
    # slack came out of the reply's 23 ms pre-key window: 0912-2349 lost slot 30
    # by +33.3 ms and 0913-0014 slot 57 by +35.4, each arm's first IRS reply slot
    # and, in 0913-0014, its only missed slot all session.
    end = min(head + round(placement.PACKET_S * FS),
              _p3_decode_deadline(live, head + round(P3_REPLY_S * FS), settle_n))
    if seg_start > head or head >= seg_start + seg.size:
        return seg, seg_start  # No joining across a transmission or lost audio.
    if live.pos < end:
        print("    [p3] first BREAK-IN head: listening through its body "
              "before another entry", flush=True)
        fresh_start = live.pos
        fresh = _listen_until_answer(live, end - live.pos, host, sessrx, 0,
                                     forget_burst=False)
        if fresh.size:
            if fresh_start != seg_start + seg.size:
                return fresh, fresh_start
            seg = np.concatenate([seg, fresh])
    # Preserve the head and its filter history while bounding the decoder's
    # input. The normal frame-delivery gate owns deduplication and timing.
    lo = max(0, head - round(.08 * FS) - seg_start)
    hi = min(seg.size, end - seg_start)
    # Retain the head's fine frequency hypothesis for this bounded body read.
    # The broad changeover search otherwise steps over it in 25 Hz increments.
    # An unvalidated candidate must not replace the established receive offset.
    old_offset = sessrx.p3_receive_offset_hz
    old_delivery = sessrx._p3_delivered_at
    sessrx.p3_receive_offset_hz = offset_hz
    try:
        _scan_frame(sessrx, seg[lo:hi], seg_start + lo)
    finally:
        if sessrx._p3_delivered_at == old_delivery:
            sessrx.p3_receive_offset_hz = old_offset
    _reverse_before_key(raster, host, tx, slot)
    return seg, seg_start


def _clamp_forgives(live, tx, late: int) -> int:
    """Whether the emission path will key this overrun INTO its own boundary.

    `KEY_CLAMP_TOL_S` is `RadioTx._tx`'s rule, and `_tx` is the only place that
    can APPLY it: it holds the finished audio, it knows the instant the carrier
    will actually occupy, and it asks the channel guard again on that instant.
    `_regrid` stands a whole tick and render in front of that and took the same
    overrun as its own verdict -- against the sample, with nothing allowed for
    -- so every sub-symbol overrun was spent as a slot BEFORE the forgiveness
    could see one. `arm-v9-A-40-ws8eoc-b`, 2026-09-13 14:42, granted-entry
    phase: nineteen `SLOT ... IS GONE` at +1.6 to +5.6 ms and not a single
    `KEYED INTO ITS BOUNDARY` in the whole arm, on a duplex stream where every
    one of those keyings was inside the tolerance.

    AND THE ALTERNATION IS THAT DECISION'S OWN ECHO. A slot handed back takes
    the cadence to two, and the admission guard is taken on a DELIVERED count
    quantised to the callback block: 60000 samples to the slot is 96 past a
    128-frame block, so a one-slot cadence walks the residue through four values
    and a two-slot cadence through exactly two (120000 mod 128 = 64). That is
    the +1.6/+5.6 ms alternation in the log -- the residue, plus a block of the
    cycle's own jitter -- and it is self-sustaining: the worse of the two phases
    is past the gate every time it comes round, so the first lost slot costs
    every other slot for the rest of the stint.

    WITH THE CYCLE'S OWN REMAINING WORK ALLOWED FOR, because this is the earlier
    gate. `RadioTx.prekey_cost_n` is what the interval between the two checks
    measured last cycle, on the stream's own clock; forgiving without it would
    hand `_tx` an overrun already past the tolerance, and a burst with a
    polarity baked into its samples then steps TWO slots where this gate would
    have spent one.

    A CHANGEOVER IS FORGIVEN ON ITS OWN TOLERANCE AND ITS OWN COST, and the
    exclusion that stood here cost 0913-1550 the link. It read `_tx`'s "nowhere
    to be moved to" as a rule about the forgiveness, when that sentence is about
    the STEP: handing the slot back does not re-place a changeover, it spends
    the cycle the placement was for and ages the peer's packet ending by another
    period. The arm is what that is worth -- ten break-in cycles, every one both
    `SLOT ... IS GONE` at +1.4 to +11.4 ms and `LATE TO THE KEY` at +1.2 to
    +7.2, twenty-three slots given away in all, and a gateway that stopped being
    readable because our cadence had gone to two slots under it.

    ITS COST IS ITS OWN because a changeover cycle does strictly more work than
    the cycle whose figure it would otherwise borrow (`breakin_cost_n`) -- and
    that cost is now STOOD OFF THE DRAIN rather than deducted here
    (`_prekey_lead`), which is why this asks a changeover for the whole
    tolerance. Charged in both places the gate shuts for good: `RadioTx._tx`
    measures the interval before it asks the clamp, so a refused changeover
    records its cost too, and `arm-v23-A-40-ws8eoc`'s first refusal wrote 8.2 ms
    against a 5 ms tolerance and left `room` negative for the seven behind it.
    An ordinary cycle still pays here, because nothing stands ITS cost off: a
    figure that under-counts hands `_tx` an overrun already past the tolerance,
    and a burst with a polarity baked into its samples then steps TWO slots
    where this gate would have spent one.

    A STREAM WITH NO TRANSMITTER IS NOT FORGIVEN: there the shortfall is
    measured against the PTT settle rather than against an instant a converter
    can be handed, and forgiving one would spend the rig's lead rather than the
    burst's place.

    AND THE WHOLE GATE IS SWITCHABLE (`RadioTx.p3_keep_slots`), because the two
    runtimes it separates are the ones the air disagrees about: "controls" is
    this function with round 16's changeover exclusion back, "none" is it off.
    """
    if tx.p3_keep_slots == "none" or (tx.p3_keep_slots == "controls"
                                      and tx.breakin_due):
        return 0
    if getattr(live, "transmit", None) is None:
        return 0
    tol = BREAKIN_CLAMP_TOL_S if tx.breakin_due else KEY_CLAMP_TOL_S
    # A changeover cycle's cost is STOOD OFF THE DRAIN and so is not deducted
    # here as well (`_prekey_lead`). Charged in both places the gate shuts for
    # good: `RadioTx._tx` measures the interval before it asks the clamp, so a
    # REFUSED changeover records its cost too, and 8.2 ms against a 5 ms
    # tolerance leaves `room` negative for every changeover behind it. That is
    # the seven refusals behind the first on `arm-v23-A-40-ws8eoc`.
    room = round(tol * FS) - (0 if tx.breakin_due else tx.cycle_cost_n())
    return late if 0 < late <= room else 0


def _regrid(live, raster: _MasterGrid, tx, host, sessrx, slot: int,
            seg: np.ndarray, seg_start: int,
            settle_n: int) -> tuple[int, np.ndarray, int]:
    """Give up a slot the cycle has already overrun, and spend it listening.

    THE GRID OUTRANKS THE WINDOW, and until this ran it did not: the
    never-transmit-late guard is evaluated once, at the top of the cycle, against
    the READ position, and everything between it and the key costs wall-clock
    time while consuming no samples. `hfmodem.tests.shrike.test_grid` carries the
    on-air measurement and reproduces it; the numbers live there rather than here
    because that is where they can be re-run.

    So a slot the carrier can no longer be placed in is given up rather than
    keyed into -- the same decision the top-of-cycle guard makes on the same
    evidence -- and the re-aim goes through `aim`, which carries the shift with
    it. That is safe HERE and only here: this runs before `host.tick()` renders,
    so nothing has yet baked a polarity in. `RadioTx._tx`'s backstop is holding
    finished audio and cannot.

    GIVEN UP AGAINST THE SAMPLE, bar half a symbol. A carrier that cannot come
    up ON the boundary is not on the grid -- but one the emission path can still
    key INSIDE it is, and that decision belongs to `_tx`, which holds the audio
    and the instant the carrier will occupy. See `_clamp_forgives`, which is
    what stops this gate spending a slot on an overrun the reader forgives.

    It tolerated `d - TR_SWITCH_S` of lateness once, on the grounds that the
    cycle is genuinely over budget on the FT-891 and that a strict rule would
    hand a slot back every cycle -- the 2.5 s cadence, which stalls a link on its
    own. The premise was false: the tracked path measures 18 ms a frame, a blind
    scan of both protocols 65.5 ms, and the cycle is 1250. Nothing here is
    compute-bound. What was true is that the work was being done in the one window
    where it costs a slot, between the bridge -- which runs to the key instant --
    and the key. Moved off it (the frame scan in front of the bridge, the flush
    and the speculative scan behind our own carrier) that window holds a 0.3 ms
    anchored read against the 8 ms the FT-891's 40 ms settle leaves past
    `key_notice`, and the tolerance has nothing left to absorb.

    The recovered slot is spent LISTENING rather than slept through. Waiting it
    out and then keying would leave a whole slot of the peer's channel sitting in
    the queue for `flush_to` to drop -- the same loss as emptying the capture
    queue after a transmission, which threw away the far end's opening symbols
    and then recorded that nobody had answered.

    ITS OWN DECODE RUNS INSIDE THE SLOT, NOT ON TOP OF THE KEY. `REGRID_RESERVE_S`
    comes off the front of the recovered window so the flush and the frame scan
    are paid for out of channel time rather than out of the settle. Without that
    the re-test faces the identical geometry every iteration -- a loop with no
    progress term, and on the FT-891's 40 ms settle it does not converge at all:
    31 slots given up and one burst emitted in 40 s of bench clock, with
    `host.tick()` never reached, so the ARQ never advances and `--max-cycles`
    never decrements. Bounded as well as fixed, because a re-grid that can run
    forever occupies nothing and reports nothing wrong.
    """
    # ON THE KEY INSTANT, WHICH IS THE BOUNDARY FOR EVERY BURST BUT ONE. A
    # changeover packet is placed against the peer's transmission and wants the
    # channel `early` sooner -- 21-23 ms at KB5LZK's 72 ms turnaround -- and
    # everything here reserved the settle in front of the BOUNDARY instead. That
    # leaves `settle - early` for the tick and the render where a boundary
    # keying gets the whole settle: 14-18 ms of the 40 on 2026-09-04, against a
    # render that measures 9-13 on the same arm's ordinary cycles, and all
    # eighteen changeovers of the two mail arms lost the race.
    seg, seg_start = _finish_entry_candidate(
        live, raster, tx, host, sessrx, slot, seg, seg_start, settle_n)
    # AND THE REPLY COMB BEFORE THE SEARCH BELOW, because a slot handed back is
    # chosen against `boundary`, and on a PACTOR-3 IRS the boundary that matters
    # is the answered packet's answer slot. Both loops reach the key through
    # here, so this is where a comb the cycle's own decode moved gets on it.
    _p3_place_reply(raster, tx, slot)
    early = raster.boundary(slot) - tx.key_instant(raster, slot)
    lead_n = settle_n + round(REGRID_RESERVE_S * FS) + early
    # THE PROGRESS TERM FOR THE TRANSITION WINDOW, and the loop above says why it
    # needs one. The window walks to the end of a possible long body and decodes
    # behind each probe, so the slot it hands back is late by its own read --
    # taken once that is a slot spent listening for the answer to our CS6, taken
    # on every try it is the same overrun three times over and no slot at the end
    # of it. One walk buys the evidence; the tries after it are for the grid.
    walked = False
    for _ in range(REGRID_TRIES):
        if tx.slots_used and tx.slots_used[-1] >= slot:
            # ALREADY SPENT. The FSM answers from `on_rx_event`, so a decode
            # inside this cycle's own window can key -- `_tx` puts that burst on
            # the grid itself and the tick below is suppressed
            # (`arq.on_cycle`'s `_rx_this_cycle`). There is nothing left to aim,
            # and a boundary a whole burst behind us reads exactly like an
            # overrun: handing the slot back here moved every following
            # transmission a slot on and took the cadence to two.
            return slot, seg, seg_start
        late = live.clamp_late(tx.key_instant(raster, slot))
        kept = _clamp_forgives(live, tx, late)
        if kept:
            print(f"    [grid] SLOT {slot} {SLOT_KEPT} -- the cycle overran its "
                  f"own key instant by {kept / FS * 1e3:+.1f} ms, which is "
                  f"inside the "
                  f"{(BREAKIN_CLAMP_TOL_S if tx.breakin_due else KEY_CLAMP_TOL_S) * 1e3:.0f} "
                  f"ms the reader forgives with the "
                  f"{tx.cycle_cost_n() / FS * 1e3:.1f} ms this cycle still owes "
                  f"the tick and the render. The emission path keys it into the "
                  f"boundary rather than the grid giving the slot away.",
                  flush=True)
        if not late or kept:
            # WHERE THE CYCLE WAS ADMITTED, for `_tx` to close against. A slot
            # handed back below is stamped with nothing: its recovered window is
            # a whole slot of listening and charging that to the render would
            # shut the forgiveness for the cycle after it.
            tx._admitted_at = int(live.sample_now())
            tx._callback_at_admit = tuple(getattr(live, "_callback_timing", ())) [-16:]
            return slot, seg, seg_start
        tx._admitted_at = None
        gave = slot
        slot = _keyable_slot(live, raster, slot, lead_n)
        tx.aim(raster, slot)
        key = tx.key_instant(raster, slot)
        print(f"    [grid] SLOT {gave} {SLOT_GONE} -- the cycle overran its own "
              f"key instant by {late / FS * 1e3:+.1f} ms, so the carrier could "
              f"not come up on it. Keying on slot {slot} instead and listening "
              f"until then; the grid has not moved.", flush=True)
        # HELD, NEVER FED, and this is the whole of what makes handing a slot back
        # cheaper than keeping it. `REGRID_RESERVE_S` is sized for the two decodes
        # named in it -- the flush at 26.7 ms and the frame scan at 20.3 -- and
        # the rolling feed was never in that sum: it re-decodes a 4 s window every
        # 0.25 s slide, which on real off-air audio costs 2.4 times the length of
        # the audio and rises from there (`FEED_MAX_SLOTS`). A slot handed back
        # then cost three seconds to save one and a half, four times a cycle, and
        # the 2026-08-02 session ran fifteen seconds to the slot. Held, the
        # recovered window costs the two decodes the reserve was measured for.
        #
        # LISTENED TO ALL THE SAME, which is the constraint this must not break.
        # The audio is in the stream and in the same buffer: `deep_scan` below
        # reads it for a frame over its whole length, the flush below decodes the
        # last of it, and the anchored control-signal read in the caller reads the
        # instant the grid says a codeword is due -- which is how a station
        # holding a link is supposed to find one anyway, rather than by sweeping.
        #
        # Its origin first, read forwards -- `_collect`'s rule; see the merge
        # below for why it cannot be recovered afterwards.
        more_start = live.pos
        more = _listen_until_answer(
            live, key - lead_n - live.holdback - live.pos, host, sessrx, 0,
            forget_burst=False)
        # The scan that finds a data frame, on the recovered slot's own audio.
        # Audio that reaches only `feed` and `flush` has been seen by the rolling
        # decoder alone, whose pilot gate cannot be trusted for data frames
        # (`_SessionRx.deep_scan`) -- so a peer packet arriving in the slot we
        # just handed back would be collected, fed, and never looked at by the
        # decoder that would have found it.
        p3_reply = (getattr(tx, "defer_p3_cs", False)
                    and host.protocol == Protocol.PACTOR3 and not raster.sending)
        pending_turn = (host.protocol == Protocol.PACTOR3 and raster.sending
                        and getattr(host.arq, "unconfirmed_breakin", False)
                        and getattr(sessrx, "_p3_row0", None) is not None)
        tracked_read = p3_reply or pending_turn
        if tracked_read:
            # A recovered P3 slot owes the same complete-frame read as an
            # ordinary slot. The generic regrid reserve ends inside its body.
            #
            more, more_start, _ = _collect(
                live, sessrx, more, more_start,
                _p3_frame_ready(sessrx, _p3_decode_deadline(live, key, settle_n),
                                live.holdback))
        heard = None
        if pending_turn and more.size:
            # Our CS3 has not established the peer's new role. It can answer
            # with a bare ACK or continue its old packet. Read the bounded
            # control first; its ARQ callback can settle the turn and emit.
            heard, _ = sessrx.control_signal_in(more, more_start, raster)
            _reverse_before_key(raster, host, tx, slot)
            key = tx.key_instant(raster, slot)
        delivered_before = getattr(sessrx, "_p3_delivered_at", None)
        if more.size and heard is None:
            _scan_frame(sessrx, more, more_start, tracked_only=tracked_read)
        if pending_turn:
            _reverse_before_key(raster, host, tx, slot)
            _p3_place_reply(raster, tx, slot)
            key = tx.key_instant(raster, slot)
        if p3_reply and not walked:
            walked = True
            slot, more, more_start = _p3_transition_window(
                live, raster, tx, host, sessrx, slot, more, more_start, settle_n)
            key = tx.key_instant(raster, slot)
        if not tracked_read:
            sessrx.flush()
        peer = getattr(raster, "_p3_peer", None)
        pending_crc = (pending_turn and peer is not None
                       and getattr(sessrx, "_p3_delivered_at", None) != delivered_before
                       and peer[1] <= key - peer[0] < peer[2])
        if (heard is not None or pending_crc
                or p3_reply and _p3_current_packet_crc(sessrx, raster, slot)):
            # Do not wait away a current-packet CRC's saved turnaround for
            # another callback batch. Recheck admission immediately; the
            # transmitter retains its final drain and lateness guard.
            if more.size:
                if seg.size and more_start == seg_start + seg.size:
                    seg = np.concatenate([seg, more])
                else:
                    seg, seg_start = more, more_start
            continue
        # Everything from here to the return is arithmetic: the reserve above is
        # what the two decodes were spent out of, and the re-test at the top of
        # the loop has to face a boundary that has not moved under them.
        #
        # ...bar the drain, which is why it goes in front of the wait rather than
        # behind it. `read` blocks a block at a time, so a take that runs to the
        # key instant empties the queue as the codec fills it; behind the wait the
        # same samples come out in one lump standing on the settle. It does not
        # get the whole settle back: reading TO the wait's own deadline can only
        # return after it, by the holdback, and that is what `RadioTx._tx` prints
        # as a short PTT lead. Measured, and left alone, in test_txhead.
        tail_start = live.pos
        wait_lead = _prekey_lead(live, settle_n, tx.cycle_cost_n())
        tail = live.take_until(key - wait_lead)
        if tail.size:
            sessrx.bridge(tail)      # held; the next cycle's flush decodes it
        live.wait_until(key - wait_lead)
        # Each piece carries its own origin, taken where it was read. The scans
        # between the two reads can key, and `_tx`'s flush moves `pos` -- so a
        # single origin summed from sizes afterwards is built from the very
        # quantity the flush invalidates, and splicing `more` and `tail` over
        # the hole puts every offset taken from `seg_start` -- the capture
        # sidecar's end and the onset fold among them -- a whole burst out.
        for fresh, start in ((more, more_start), (tail, tail_start)):
            if not fresh.size:
                continue
            if seg.size and start == seg_start + seg.size:
                seg = np.concatenate([seg, fresh])
            else:
                seg, seg_start = fresh, start
    over = live.clamp_late(tx.key_instant(raster, slot))
    if over and not _clamp_forgives(live, tx, over):
        print(f"    !! REGRID GAVE UP after {REGRID_TRIES} tries: slot {slot} "
              f"is late already -- the cycle is overrunning slots faster than "
              f"the grid can hand them back. This burst goes to the emission "
              f"path's backstop.", flush=True)
    return slot, seg, seg_start


def _load(path: str) -> np.ndarray:
    return session.load_wav(path, FS)


class _Link(NamedTuple):
    """What a hold cycle reads off the link to decide whether it is still going."""
    sent: int
    rcvd: int
    role: Optional[str]

    @classmethod
    def of(cls, host) -> "_Link":
        return cls(host.sent_total, host.rcvd_total, host.arq.role)


class _HoldBudget:
    """The cycle a hold stops on, which is a condition rather than a duration.

    `--hold N` IS AN IDLE TIMEOUT: the deadline sits N cycles past the last cycle
    that moved payload. N is the right answer for a peer that has gone and the
    wrong one for a peer that is mid-sentence, and a count of cycles cannot tell
    those apart. K4MSU, 3595 kHz, 2026-08-19 22:10, `captures/onair-0819-2210`:
    twelve cycles of codewords, and in cycle 13 the gateway took the channel with
    a CS3-headed packet carrying `RMS Tri` -- a full 7-byte break-in field, the
    opening of "RMS Trimode". Cycle 13 is the cycle a deadline of 12 expires in.
    `hold_13` and `hold_15` both decode that same packet under packet counter 0,
    which is an ISS repeating what its IRS never acknowledged: this station spent
    the exchange's one moving cycle queueing a goodbye instead of an answer.

    Against a peer that has genuinely gone the first cycle is the last one that
    moved anything, so the hold ends on N -- and never reaches it, because the
    ARQ gives up on an unanswered packet after `cfg.max_retries` and takes the
    link down three cycles inside a 12-cycle budget.

    BYTES BUY A CYCLE, AND SO DOES THE CHANGEOVER THAT ASKS FOR ONE. An
    acknowledgement buys nothing: a link doing nothing at all still draws a
    codeword every cycle, and the 22:07 session's `#1 x18` is what a peer
    answering forever looks like. Nor does a packet on its own: the idle fill an
    ISS with an empty buffer transmits decodes to nothing and is never delivered.
    A CS3 is the exception because it is the only codeword a peer sends when it
    has something to say -- taking the transmit direction to say it -- and a hold
    that accepts the changeover and then breaks in with a goodbye before the
    peer's first packet lands is the same fault this class was made against, one
    turn later. `_Link` is what the loop reads once a cycle.

    Both directions, because it is one fault with the arrow reversed: a station
    uploading a message holds a link the peer is only acknowledging, and cutting
    that off mid-message is the same thing done to our own bytes.

    `ceiling` is what keeps a grant from becoming a lease. Without it a peer that
    never stops sending holds this transmitter on a shared channel for as long as
    it likes, and that is not the peer's decision to make. It binds FROM THE
    FIRST CYCLE and not only where the deadline is pushed: a `--hold` past the
    ceiling is a ceiling-length hold, which is why the deadline starts as though
    cycle 0 had moved payload. Clamped in `moved` alone the class ran backwards
    above the ceiling -- `--hold 1000` gave a peer that said nothing the whole
    1000 cycles, and the first byte it sent pulled the deadline in to 480.

    Which of the two bounds is holding the deadline is not the operator's to
    infer, so it is recorded rather than reconstructed: against a peer that keeps
    sending, the idle timeout never expires at all and saying it did names a
    budget that had nothing to do with the ending.

    THE CEILING IS TIME AND THE IDLE TIMEOUT IS TURNS, and the long cycle is
    where they part company. A hold cycle is one turn of the link whatever the
    link is worth in seconds, which is what `--hold` means; the ceiling is a
    promise about how long this transmitter holds a shared channel, and 480 of
    them at 3.75 s is half an hour rather than the ten minutes it says. So the
    grid slots a cycle spent are charged (`spend`), and the ceiling binds on
    those.
    """

    def __init__(self, cycles: int, ceiling: int = HOLD_MAX_CYCLES, *, unbounded=False):
        self.idle, self.ceiling, self.closed = cycles, ceiling, False
        self.unbounded = unbounded
        self.slots, self.ticks = 0, 1
        self.moved(0)

    def spend(self, slots: int) -> None:
        """Charge one cycle's grid slots against the ceiling."""
        self.slots += slots
        self.ticks = slots

    def cycle(self, h: int, was: "_Link", now: "_Link") -> None:
        if self.closed:
            return
        if (now.sent, now.rcvd) != (was.sent, was.rcvd) or (
                now.role == IRS and was.role != IRS):
            self.moved(h)

    def moved(self, h: int) -> None:
        if self.unbounded:
            self.at_ceiling, self.deadline = False, float('inf')
            return
        # The ceiling in the unit the deadline is kept in: the cycles the slots
        # left over it will buy, at what this cycle cost. On the short cycle
        # that is `h + (ceiling - h)` -- the ceiling itself, an absolute cycle
        # number, exactly as it flies. A cycle spends at least one slot, so a
        # caller that never charges any is charged the short cycle's.
        spent = max(self.slots, h)
        cap = h + -(-(self.ceiling - spent) // self.ticks)
        self.at_ceiling = h + self.idle > cap
        self.deadline = min(h + self.idle, cap)

    def close(self, h: int) -> None:
        """Spend no more of it: the exchange is over and the goodbye is owed."""
        self.closed = True
        self.deadline = min(self.deadline, h)

    @property
    def ran_out(self) -> str:
        """The bound that closed the hold, named, for the session's own ending."""
        return (f"the {self.ceiling}-slot ceiling on one station's hold of a "
                f"shared channel was reached ({self.slots} spent)"
                if self.at_ceiling else
                f"the hold's {self.idle} idle cycles ran out")


def _listen_due(every: int, keyed: int, last: int) -> bool:
    """Is this the cycle `--listen-every` gives to the receiver.

    COUNTED IN KEYINGS AND NOT IN CYCLES, which is what makes the schedule mean
    what the flag says: a hold spends cycles on hushes, on refused changeovers
    and on the listen cycles themselves, and counting those would put two
    silences together and hand the peer a gap it can only read as a station that
    has gone. `last` is the keying count the previous listen cycle was taken at,
    and it is what stops a cycle that keyed nothing repeating the decision for
    ever off the same number.
    """
    return bool(every) and keyed > 0 and keyed % every == 0 and keyed != last


def _drain_host_log(host) -> None:
    """Print what the host layer had to say about this cycle, and forget it.

    `PtcHost.log` writes the link-layer's own account -- why a codeword was not
    taken as an answer to our call, that a CS4 dropped the link to 100 Bd and
    requeued the packet, that the retry budget is gone -- into `log_lines`, which
    nothing read. So the one explanation an operator needs when a station answers
    and the link still does not hold was being written and thrown away every
    cycle, and the session log showed only the codewords it was written about.
    """
    for line in host.log_lines:
        print(f"    [host] {line}", flush=True)
    del host.log_lines[:]


def _cs_errors(ev) -> int:
    """Bit errors behind a control-signal decode, read off the event's own text.

    The acquisition path carries no count because it reports nothing but an exact
    match -- `p1rx.acquire_control_signal` requires zero bit errors in twelve --
    so an event without a number is a zero-error event.
    """
    m = re.search(r"(\d+) bit errors", str(ev.text))
    return int(m.group(1)) if m else 0


def _cs_name(ev) -> str:
    """PACTOR-1's four codewords are a different set from PACTOR-3's six.

    ...and its two unassigned words are a third set, which is why `spare` is
    read here: they arrive with no `cs`, and a tally that indexed on `cs` alone
    could only leave them out."""
    if ev.cs is None:
        return spec.P1_CS_NAMES[ev.spare]
    names = spec.P1_CS_NAMES if ev.protocol == Protocol.PACTOR1 else spec.CS_NAMES
    return names[ev.cs]


def _held_answers(held: list) -> str:
    """What the peer said INSIDE the link, which is what a hold's ending left out.

    Six sessions of the 2026-08-19 evening ended down the same branch, and they
    were two entirely different evenings. WS8EOC 20:20, KB5LZK 20:41 and K4MSU
    20:48 heard a gateway the whole way through -- `captures/onair-0819-2048`
    reads CS1 unbroken across `hold_05` to `hold_16`, twelve consecutive cycles
    of a strong, clean "send that again". KO4HJO 20:53, W9OTR 21:08 and VE3WLR
    21:16 heard nothing inside the link at all. Both groups ended `the peer never
    acknowledged the goodbye`, which is true of both and describes neither.

    Counted from the cycle the link came up, not from the session's first sample.
    The connect answer is a codeword too, and a count that includes it says "a
    peer answered" about a session in which nothing answered once we were linked.

    A repeated codeword is a REQUEST -- in PACTOR-1 the acknowledgement is the
    alternation, so an unbroken CS1 is a peer asking twelve times for the packet
    it already has. It is still an answer, which is why the retry budget never
    spends against it and the hold, not the ARQ, is what ends such a session.

    Which is why the alternation is counted over `P1_ACKS` and not over every
    codeword that arrived: a link that went CS1, CS4, CS1 changed its codeword
    twice while acknowledging nothing at all.

    AND OVER PACTOR-1's OWN CODEWORDS, AT ZERO ERRORS, which is `_summary`'s
    filter exactly -- the two verdicts are printed in one output and must not
    disagree about what alternated. This had neither. `P1_ACKS` is a pair of
    INDICES, and an upgraded link's reverse channel is PACTOR-3, where index 1 is
    CS2/req: a repeat request counted as the other half of an acknowledgement,
    manufacturing an alternation out of a peer asking twice for the same packet.
    The error filter is the same argument one bit lower down -- a codeword the
    demodulator was unsure of is not evidence that the peer's answer changed.
    """
    cs = [ev for ev in held if ev.cs is not None or ev.spare is not None]
    if not cs:
        return "no control codewords decoded inside the link (data frames counted separately)"
    heard: dict[str, int] = {}
    for name in map(_cs_name, cs):
        heard[name] = heard.get(name, 0) + 1
    what = ", ".join(f"{n} x{k}" if k > 1 else n for n, k in sorted(heard.items()))
    acks = [ev.cs for ev in cs if ev.protocol == Protocol.PACTOR1
            and ev.cs in P1_ACKS and _cs_errors(ev) == 0]
    changed = sum(1 for a, b in zip(acks, acks[1:]) if a != b)
    if not changed:
        # A REPEAT REQUEST IS A CLAIM ABOUT CS1 AND CS2 AND ABOUT NOTHING ELSE.
        # The alternation is the acknowledgement, so an unbroken run of one of
        # them is a peer asking again -- but `0x59A` is not in that alternation
        # at all, and calling a run of the PACTOR-3 grant a repeat request reads
        # a peer commanding a waveform as a peer stuck on a packet.
        #
        # A HELD CS4 IS A REQUEST TOO, by a clause of its own: after a CS4
        # connect answer the first 100 Bd block is acknowledged CS1 or CS3, and
        # "CS4 in Folge (ohne zwischenzeitliches richtiges CS) werden als
        # 'Request' interpretiert" (pactor1-control-signals.md sec 4.3). KB5LZK
        # spent its whole 30-cycle budget on one, 2026-09-06, and this line
        # called those 26 zero-error PACTOR-1 codewords unreadable.
        speed = sum(1 for ev in cs if ev.protocol == Protocol.PACTOR1
                    and ev.cs == pactor1.CS_SPEED and _cs_errors(ev) == 0)
        other = len(cs) - len(acks) - speed
        counted = []
        if acks:
            counted.append(f"{len(acks)} a repeat request")
        if speed:
            counted.append(f"{speed} a held CS4, which is the request for the "
                           f"first 100 Bd block")
        if other:
            counted.append(f"{other} the alternation cannot count: another "
                           f"protocol's codeword, a word PACTOR-1 assigns no "
                           f"meaning, or a read with bit errors in it")
        return (f"the peer answered {len(cs)} held cycles ({what}) and never "
                f"alternated -- "
                + (", and ".join(counted) if speed or other
                   else "every one a repeat request"))
    return (f"the peer answered {len(cs)} held cycles ({what}), alternating "
            f"{changed} times")


def _observe_out_the_hold(live, raster: _MasterGrid, tx, sessrx, host,
                          evidence_of: _CycleEvidence, budget: _HoldBudget,
                          first: int, slot: int) -> int:
    """Spend the rest of the hold RECEIVING, with the transmitter latched off.

    The link going down and the channel going quiet are two different events, and
    stopping at the first one gives up the recording that says which happened. A
    peer whose acknowledgements we stopped reading goes on asking for its packet;
    a gateway that dropped us signs off, or calls the next station. None of that
    is visible from a session that ends where the ARQ gave up: WS8EOC, 2026-08-28
    (captures/onair-0828-1844), was still transmitting on our own raster 26 s
    after we signed off, and the only receiver that knew was a separate tap.

    So the loop keeps its receive half and none of its transmit half: no tick, no
    goodbye, no aim. The grid is read for where the next window closes and
    nothing is keyed on it, and `RadioTx.observing` holds that whatever the audio
    turns out to contain.

    Nothing here reports an answer band or forecasts the next key. Both measure
    the channel against our own carrier, and there is no longer one to measure
    against; a band drawn around a transmission that did not happen is an
    instrument reading its own arithmetic.

    Returns the cycles observed.
    """
    tx.observing = True
    for h in itertools.count(first):
        if h > budget.deadline:
            return h - first
        before = sessrx.count
        sessrx.new_cycle()
        # Whatever slot the stream is standing in, rather than the one the hold
        # left off at: the last cycle keyed, and a boundary already behind the
        # read position is a window of no samples that never advances.
        while raster.boundary(slot) <= live.pos:
            slot += 1
        boundary = raster.boundary(slot)
        # Boundary to boundary, and held rather than fed for the receiving hold
        # cycle's reason: the rolling decoder costs more than the audio lasts, and
        # a cycle spent inside it is a cycle the capture stream can lose samples
        # in. The flush and the burst reader below have the whole window.
        whole = _listen_until_answer(live, boundary - live.pos, host, sessrx, 0)
        whole, seg_start, bridge = _collect(live, sessrx, whole,
                                            live.pos - whole.size, boundary)
        sessrx.flush()
        if whole.size:
            _scan_frame(sessrx, whole, seg_start, upgrade=True)
        onsets = [at for at, _ in
                  evidence_of.record(f"hold_{h:02d}", whole, seg_start)]
        _read_codeword_at_bursts(sessrx, whole, seg_start, onsets)
        _drain_host_log(host)
        print(f"    [grid] hold {h} slot {slot}; captured "
              f"{whole.size / FS:.3f} s, bridged {bridge:.0f} ms "
              f"-- OBSERVING, not keying", flush=True)
        if sessrx.count == before:
            print(f"    HOLD {h} RX (quiet)", flush=True)
        # Off the air, and further past our own last carrier with every cycle:
        # nothing heard here can be an answer to us, which is what `update`
        # reads `since_tx` to decide.
        line = raster.update(onsets, whole, seg_start, hushed=True,
                             reading=_scheduler_reading(sessrx, host, raster),
                             since_tx=seg_start - (tx.tx_end or 0))
        if line:
            print(f"    [grid] {line}", flush=True)


def _duty(keyed: list[tuple[int, float]], cycles: int, cycle_s: float) -> str:
    """How much of the session we held the channel for, two ways.

    KEYED CYCLES, because that is what a witness receiver counts and what a peer
    experiences: a cycle we were absent from is a cycle it got no packet in,
    whatever we did in the others. Distinct SLOTS rather than bursts -- the
    emission path's backstop moves a burst the loop did not schedule onto a later
    boundary, so bursts and cycles are not the same count -- and read off the
    keying branch of `RadioTx._tx`, so a dry run reports none rather than
    reporting its filenames as transmissions.

    AND SECONDS, because the two burst lengths are eight to one. The 2026-08-06
    session's 22 bursts were 14 data packets and 8 control signals: 23% of its
    cycles and 12% of its air. One number cannot say both, and a duty cycle
    quoted off the wrong one is out by a factor of two.
    """
    if not keyed:
        return "keyed 0 cycles -- nothing went on the air"
    slots, air = {s for s, _ in keyed}, sum(d for _, d in keyed)
    return (f"keyed {len(slots)} of {cycles} cycles ({100 * len(slots) / cycles:.0f}%) "
            f"in {len(keyed)} burst(s), {air:.1f} s of carrier "
            f"({100 * air / (cycles * cycle_s):.0f}% of the air)")


def _summary(cs_log: list, tx_slots: list[int], cycles: int,
             keyed: list[tuple[int, float]], cycle_s: float,
             seq_sent: list[int], evidence: "_ConnectEvidence",
             ended: str, captures: str = "",
             leads: tuple[float, ...] | list[float] = (),
             settle: float = 0.0,
             breakin_at: frozenset[int] | set[int] = frozenset()) -> None:
    """What the session decoded, and whether the link ever moved.

    The per-cycle log runs to dozens of lines a minute and the operator is at the
    radio while it scrolls, so the one thing a connect attempt is run for -- did
    a packet ever get through -- was recoverable only by post-processing the
    captures afterwards. Everything here comes from the events the FSM was handed
    and from the counters we put on the air; nothing re-reads the audio, so a
    signal missing from this list is a signal the session genuinely did not
    decode.

    TWO measurements, and the pair is the point. An acknowledgement in PACTOR-1
    is the ALTERNATION of CS1 and CS2 -- a repeated codeword, whichever one it
    is, is a repeat request -- so the peer's side of the question is whether the
    acknowledging codeword ever changed. Our side is whether the packet counter
    left the value the first data packet carries. A link that is working shows
    both; the WS8EOC sessions of 2026-07-30 show neither.

    This asked whether any codeword other than CS4 arrived, and answered YES on
    every one of those sessions -- the connect answer is a CS1, so the test was
    satisfied before the first data packet was even sent, and "Prediction met"
    printed over a whole session of a peer asking for the same packet again.
    """
    # AND THE UNASSIGNED WORDS ARE IN IT. A twelve-bit word read at zero of
    # twelve is the peer transmitting on our raster whether or not PACTOR-1
    # publishes a meaning for it, and counting only the four with meanings is
    # what let `3 CS1/CS2, 0 other` print over ten zero-error `0x59A`.
    p1 = [ev for ev in cs_log
          if ev.protocol == Protocol.PACTOR1
          and (ev.cs is not None or ev.spare is not None)
          and _cs_errors(ev) == 0]
    # An upgraded link's reverse channel is PACTOR-3, and this counted only
    # PACTOR-1 -- so a session that upgraded and then ran perfectly reported zero
    # control signals, which reads exactly like a dead band.
    p3 = [ev for ev in cs_log
          if ev.protocol == Protocol.PACTOR3 and ev.cs is not None]
    print("\n-- session summary --", flush=True)
    print(f"session ended: {ended}")
    print(_duty(keyed, cycles, cycle_s))
    # THE PTT LEAD, once and with the worst value seen, where fifty per-burst
    # lines stood in the record on 2026-08-13. Stated even when nothing eroded,
    # because this figure is what any rig-side settle measurement will be
    # argued from. The erosion has a zero-cost check by ear, and the
    # operator ran it first: the silent gap behind a burst is the cycle minus
    # the PTT-held time, so every millisecond lost off the front of the lead is
    # a millisecond added to the pause after the burst.
    if leads:
        worst = min(leads)
        eroded = sum(1 for x in leads if x < settle - 1e-3)
        if eroded:
            print(f"PTT lead: SHORT on {eroded} of {len(leads)} bursts, worst "
                  f"{worst * 1e3:.0f} ms of the {settle * 1e3:.0f} this rig is "
                  f"set for -- the holdback, on every burst aimed at a boundary; "
                  f"audible as the pause after such a burst running up to "
                  f"{(settle - worst) * 1e3:.0f} ms long")
        else:
            print(f"PTT lead: the full {settle * 1e3:.0f} ms on all "
                  f"{len(leads)} bursts")
    # Beside the cycles keyed, because the analysis afterwards is done off these
    # files: a session whose tail never reached the disk is one whose last
    # cycles cannot be measured, and the operator has to learn that here rather
    # than tomorrow.
    if captures:
        print(captures)
    # What the search saw, not just what survived it. "control signals decoded 1"
    # is what three sessions that connected to nobody printed on 2026-08-06, and
    # it reads as a faint peer rather than as a coin landing heads.
    print(evidence.report())
    # SECONDS INTO THE SESSION, said rather than left to be inferred. The column
    # carried two clocks -- the rolling decoder's session time and the anchored
    # read's offset into its own window -- and a 0.09 among figures of 38.16 read
    # as a codeword in the first tenth of a second.
    print(f"control signals decoded {len(cs_log)} ({len(p1)} PACTOR-1 at zero "
          f"errors, {len(p3)} PACTOR-3), at seconds into the session")
    for ev in p3:
        print(f"  {ev.t:7.2f}  {spec.CS_NAMES[ev.cs]}  (PACTOR-3)")
    # The transmit cadence, MEASURED. A prediction that fails on the two-slot
    # regression says something entirely different from one that fails on the
    # protocol's own 1.25 s cadence, and the difference is not visible anywhere
    # else in a scrolling log.
    #
    # Read off `RadioTx.slots_used`, which is appended where the carrier is
    # scheduled rather than where the loop decided to schedule one. The loop's
    # own local is read before the emission path's backstop can move a burst, so
    # it records the intention: "median slot increment 1 = 1.25 s" is what the
    # session that put half its bursts in the next slot printed.
    inc = np.diff(tx_slots)
    if inc.size:
        med = float(np.median(inc))
        how = ("the one-slot cadence" if med == 1 else
               "NOT one slot -- the peer was called at the wrong rate")
        print(f"transmit cadence: median slot increment {med:g} "
              f"= {med * cycle_s:.2f} s, {how}")
    else:
        print("transmit cadence: fewer than two transmissions, nothing to measure")
    for ev in p1:
        print(f"  {ev.t:7.2f}  {_cs_name(ev)}")
    acks = [ev for ev in p1 if ev.cs in P1_ACKS]
    alternated = next((ev for ev, prev in zip(acks[1:], acks) if ev.cs != prev.cs),
                      None)
    runs = [(n, len(list(g))) for n, g in itertools.groupby(seq_sent)]
    print("packet counters sent: "
          + (", ".join(f"#{n}" + (f" x{k}" if k > 1 else "") for n, k in runs)
             or "none -- no data packet was sent"))
    # An ADVANCE is a +1 step mod 4 -- the thing only an acknowledgement can
    # produce. `set(seq_sent)` was the test here, and a BREAK-IN satisfied it:
    # the changeover packet resets the counter to 0 (arq: "the counter resets
    # [spec]"), so "#1 x9, #0" -- nine unacknowledged repeats and a yield --
    # read as the counter moving, and one session's verdict printed
    # ACKNOWLEDGED over a link that carried a single packet and then died.
    # Testing the step arithmetically left the same hole one counter narrower:
    # a reset out of #3 also lands on #0. So the break-in is excluded by WHICH
    # PACKET IT WAS, which keeps the honest #3 -> #0 -- the wrap an acknowledged
    # #3 makes, indistinguishable from the reset by arithmetic alone.
    advanced = next((f"#{a} -> #{b}"
                     for i, (a, b) in enumerate(zip(seq_sent, seq_sent[1:]))
                     if (b - a) % SEQ_MOD == 1 and i + 1 not in breakin_at), None)
    if not p1 and p3:
        # The PACTOR-1 alternation test does not apply to a link that left
        # PACTOR-1: PACTOR-3 has an explicit NAK, so its acknowledgement is the
        # codeword and not the toggle.
        print(f"verdict: the link ran in PACTOR-3 -- {len(p3)} control signals "
              f"decoded and our counter reached "
              f"#{seq_sent[-1] if seq_sent else '?'}.", flush=True)
    elif not p1 and evidence.candidates:
        # A SILENT CHANNEL AND AN UNCORROBORATED ONE ARE DIFFERENT NIGHTS. This
        # line reads off what the ARQ was handed, and an accept the rule did not
        # close on is never handed to it -- so the WS8EOC call of 2026-09-16
        # printed "no PACTOR-1 control signal decoded" with five zero-error
        # codewords standing in its own log, two of them 1.5 ms apart. The
        # operator's next move differs: nothing decoded is a band question, and
        # accepts that did not corroborate is a question about this station's
        # receive floor and the peer's schedule. The rule itself does not move
        # here; the verdict stops contradicting the log above it.
        spread = evidence.closest_pair()
        near = (f"closest pair {spread * 1e3:.1f} ms apart"
                if spread is not None else "a single accept, which has no spread")
        print(f"verdict: {len(evidence.candidates)} zero-error codeword(s) "
              f"accepted in the answer band and none corroborated -- {near}, "
              f"against {evidence.N} inside {evidence.SPAN} searched windows "
              f"agreeing to {evidence.TOL_S * 1e3:.0f} ms. Nothing was delivered "
              f"to the link, so the prediction was not tested.", flush=True)
    elif not p1:
        print("verdict: no PACTOR-1 control signal decoded -- the prediction was "
              "not tested.", flush=True)
    elif alternated is None or advanced is None:
        why = []
        if alternated is None:
            # NAMED, NOT COUNTED. "0 other" was the whole of what this line had
            # to say about a gateway that spent forty-five seconds commanding
            # PACTOR-3, and the bucket was empty because the words were never
            # collected -- so the one line an operator reads for the verdict said
            # the channel was quiet. What arrived is now spelled, because "10
            # other" and "10 x 0x59A/unassigned" call for different next slots.
            other: dict[str, int] = {}
            for ev in p1:
                if ev.cs not in P1_ACKS:
                    n = _cs_name(ev)
                    other[n] = other.get(n, 0) + 1
            spelled = ", ".join(f"{n} x{k}" if k > 1 else n
                                for n, k in sorted(other.items()))
            why.append(f"the peer never alternated its acknowledgement "
                       f"({len(acks)} CS1/CS2, {len(p1) - len(acks)} other"
                       + (f": {spelled}" if spelled else "") + ")")
        if advanced is None:
            why.append("our packet counter never advanced")
        print(f"verdict: {'; '.join(why)}. Prediction NOT met.", flush=True)
    else:
        print(f"verdict: ACKNOWLEDGED -- the peer alternated to "
              f"{spec.P1_CS_NAMES[alternated.cs]} at {alternated.t:.2f} s and our "
              f"counter advanced {advanced}. Prediction met.", flush=True)


def _mail_arm(args) -> bool:
    """Whether this arm is carrying Winlink mail, which is a different arm."""
    return bool(args.mail_send or args.mail_fetch)


def _p3_mail_defaults(args) -> None:
    """Opt-in mail preset using the measured initial IRS pulse placement."""
    if not getattr(args, "p3_mail", False):
        if getattr(args, "p3_qrt_confirm", False):
            raise SystemExit("--p3-qrt-confirm requires --p3-mail")
        if getattr(args, "mail_wait_greeting", False) is None:
            args.mail_wait_greeting = False
        return
    incompatible = ("p3_timing_trial", "p3_entry_timing_trial", "p3_timing_unbounded",
                    "p4_entry", "pactor1_only", "pactor3_only", "offer_pactor2",
                    "p3_uninvited", "decline_grant", "message", "listen_every",
                    "p3_changeover_cs5", "p3_changeover_p1_cs")
    if (any(getattr(args, k, False) for k in incompatible)
            or getattr(args, "p3_timing_reply_delay", None) is not None):
        raise SystemExit("--p3-mail requires granted P3 entry without timing trials, "
                         "forced modes or alternate controls")
    if (args.p3_control_waveform != "historical"
            or args.p3_control_placement != "audio-start"
            or args.p3_control_tail != "off" or args.p3_control_stagger != "off"
            or not args.p3_entry_stagger or not args.p3_rise
            or args.p3_entry_delay != DEFAULT_ENTRY_DELAY_MS
            or args.cycle != 1.25 or tuple(args.p3_entry) != ("template",)
            or args.p1_setup_phase != "reply"):
        raise SystemExit("--p3-mail requires historical audio-start controls and "
                         "the staggered shaped template entry at +4.875 ms")
    args.p1_grant_only = True
    args.p3_repeat_gear = 0
    args.p3_entry_sl = args.p3_traffic_sl = 1
    args.no_p3_fallback = True
    args.over, args.observe_after_link_down = True, False
    args.mail_fetch = True
    if args.mail_wait_greeting is None:
        args.mail_wait_greeting = True
    if not args.hold:
        args.hold = MAIL_HOLD_CYCLES


def _timing_trial_defaults(args) -> None:
    """An opt-in timing preset; reject unrelated experiments before opening IO."""
    entry_arm = getattr(args, "p3_entry_timing_trial", None)
    unbounded = getattr(args, "p3_timing_unbounded", False)
    reply_delay = getattr(args, "p3_timing_reply_delay", None)
    if reply_delay is not None and (getattr(args, "p3_timing_trial", None) != 'B' or entry_arm):
        raise SystemExit("--p3-timing-reply-delay requires --p3-timing-trial B")
    if unbounded and (not getattr(args, "p3_timing_trial", None) or entry_arm):
        raise SystemExit("--p3-timing-unbounded requires --p3-timing-trial A or B")
    if entry_arm and getattr(args, "p3_timing_trial", None):
        raise SystemExit("entry timing and reply timing are separate experiments")
    if not getattr(args, "p3_timing_trial", None) and not entry_arm:
        return
    incompatible = ("pactor1_only", "pactor3_only", "offer_pactor2", "p3_uninvited",
                    "decline_grant", "mail_send", "mail_fetch", "message",
                    "p3_changeover_cs5", "p3_changeover_p1_cs", "listen_every")
    if any(getattr(args, k, False) for k in incompatible):
        raise SystemExit("--p3-timing-trial needs a granted P3 entry without mail, "
                         "forced modes, alternate controls or listening cycles")
    if (args.p3_control_waveform != "historical"
            or args.p3_control_placement != "audio-start"
            or args.p3_control_tail != "off" or args.p3_control_stagger != "off"
            or not args.p3_entry_stagger or not args.p3_rise
            or args.p3_entry_delay not in (0.0, DEFAULT_ENTRY_DELAY_MS)
            or args.cycle != 1.25
            or tuple(args.p3_entry) != ("template",)):
        raise SystemExit("--p3-timing-trial requires the historical audio-start "
                         "control and staggered shaped template entry on a 1.25 s cycle")
    args.p1_grant_only = True
    args.p3_speed_up, args.p3_repeat_gear = "hold", 0
    args.long_cycle, args.no_p3_fallback = False, True
    args.over, args.observe_after_link_down = True, False
    args.hold = 32
    if entry_arm:
        args.p3_entry_delay = EntryTimingTrial(entry_arm).delay_ms
        print(f"  P3 ENTRY TIMING TRIAL {entry_arm}: entry delay {args.p3_entry_delay:g} ms; "
              "no mail or speed commands; stop on P3 acquisition or 20 seconds "
              "from the first emitted entry", flush=True)
    else:
        limit = ("unbounded initial receive turn; no experiment or hold cutoff; "
                 "peer termination, turn changes and normal link-loss rules still apply"
                 if unbounded else "12 reply opportunities / 20 seconds after CRC changeover")
        print(f"  P3 TIMING TRIAL {args.p3_timing_trial}: no mail or speed commands; "
              f"{limit}", flush=True)
        if reply_delay is not None:
            print(f"  P3 REPLY DELAY +{reply_delay:g} ms: target pulse at emitted entry "
                  f"+{600+reply_delay:g} ms modulo 1250 ms; RX and PTT use the shifted grid", flush=True)


def _entry_trial_control_confirmation(tx, host):
    trial = getattr(tx, "timing_trial", None)
    if (isinstance(trial, EntryTimingTrial) and trial.entries and trial.reason is None
            and host is not None and host.protocol == Protocol.PACTOR3
            and host.arq.state == State.CONNECTED and not host.arq._qrt_pending
            and host.arq._disconnect_ticks is None
            and not host.arq.entry_pending):
        trial.finish("entry state confirmed; no inbound CRC yet")


def _timing_trial_close(tx, host, now):
    """End scoring and request the existing guarded teardown exactly once."""
    trial = getattr(tx, "timing_trial", None)
    if trial is None or trial.closing:
        return None
    _entry_trial_control_confirmation(tx, host)
    trial.check(now)
    if trial.active and (host.protocol != Protocol.PACTOR3
                         or host.arq.role != IRS or host.arq.state not in LINKED):
        trial.finish("link or role changed")
    if trial.reason is None:
        return None
    trial.closing = True
    # A peer's QRT still deserves its queued final ACK. Otherwise discard the
    # scored reply before requesting our ordinary bounded disconnect sequence.
    if not host.arq._rx_close_pending:
        tx.cancel_pending_cs()
        host.arq.on_host_disconnect()
    return f"P3 timing trial {trial.arm}: {trial.reason}"


def _mail_app_turns(host, mail, *, wait_greeting: bool = False) -> None:
    """Optionally wait for actual mail output before taking the peer's turn.

    WS8EOC B3 on September 12 requeued setup announcement bytes on the grant.
    They provoked a break-in after RMS, before B2F had produced any reply.
    A calling session emits its first text only after the greeting prompt;
    receiving a banner fragment or SID alone must not release this gate.
    """
    host.app_turns(allow_breakin=not wait_greeting or bool(mail.session.sent_text))


UPGRADE_DOORS = ("pactor3_only", "offer_pactor2", "p1_grant_only",
                 "p3_uninvited")
"""The flags of the `level` group that lead OUT of PACTOR-1.

The one place a door is named. `--pactor1-only` is the closed door and
`--decline-grant` refuses the one thing an announcement can draw, so neither is
here; anything else added to that group and not added here turns a mail arm into
a PACTOR-1 run behind the operator's back, which is what `--unannounced` did to
three arms on 2026-09-17."""


def _run_p4_probe(tx, live, raster) -> str:
    """Own the post-grant clock; retain RX without running connected P3 ARQ.

    A full four-second RX interval follows each burst. This is an experiment
    cadence on the inherited phase grid, not a claim about the peer's P4 state.
    """
    probe = tx.p4_probe
    if live is None or raster is None:
        probe.reason = "probe requires a sample-clock capture stream"
        return probe.reason
    audio = p4chirp.entry_packet(probe.payload, probe.status)
    # Render once, before choosing a future boundary. No scan/render is owed
    # between the final admission check and the chirp's key instant.
    slot = tx.slot
    receiver, tx.sessrx = tx.sessrx, None
    tx.breakin_due = tx.listening = False
    probe.reason = "probe interrupted"

    def capture(until, kind="post-chirp"):
        begin = live.pos
        until = min(until, probe.deadline(FS))
        try:
            while live.pos < until:
                if not probe.room(int(live.sample_now()), 0, FS):
                    probe.reason = "post-grant deadline during receive window"
                    break
                part = live.take_until(min(until, live.pos + FS // 4))
                if not part.size:
                    probe.reason = "capture ended before receive window completed"
                    break
        finally:
            probe.windows.append(dict(kind=kind, start=begin, end=live.pos, target=until))
        return live.pos >= until

    try:
        for attempt in range(probe.attempts):
            now = max(live.pos, int(live.sample_now()))
            # Keep half a second for admission, or longer for a slow rig/DAC.
            reserve = max(FS // 2, _prekey_lead(live, round(tx.settle * FS)) + FS // 4)
            while raster.boundary(slot) < now + reserve:
                slot = raster.next_slot(slot)
            boundary = tx.aim(raster, slot)
            span = (boundary - now) / FS + len(audio) / FS + probe.listen_seconds
            if not probe.room(now, span, FS):
                probe.reason = "post-grant deadline; no room for a full chirp and receive window"
                break
            # Capture early waiting time without invoking the rolling decoder.
            if not capture(boundary - reserve, "prekey"):
                break
            before = len(tx.slots_used)
            probe.emitting = True
            try:
                tx._tx(audio, f"P4 PROBE CHIRP {attempt + 1}/{probe.attempts}")
            finally:
                probe.emitting = False
            if tx.refused or len(tx.slots_used) == before:
                probe.reason = "chirp refused; stopping without automatic retry or guard stand-down"
                capture(min(probe.deadline(FS), live.pos + round(probe.listen_seconds * FS)),
                        "after-refusal")
                break
            probe.emissions.append(dict(slot=tx.slot, audio_start=tx.tx_audio_start,
                                        audio_end=tx.tx_end))
            print(f"    [p4 probe] chirp {len(probe.emissions)}/{probe.attempts}; "
                  f"receiving continuously for {probe.listen_seconds:g} seconds; "
                  "P3 scans and ARQ replies suspended", flush=True)
            if live.pos < tx.tx_end:  # Dry replay has no duplex TX flush.
                live.take_until(tx.tx_end)
            if not capture(tx.tx_end + round(probe.listen_seconds * FS)):
                break
            slot = raster.next_slot(tx.slot)
        else:
            probe.reason = "chirp limit reached; final receive window complete"
    finally:
        tx.sessrx = receiver
        probe.emitting = False
    print(f"    [p4 probe] STOP: {probe.reason}", flush=True)
    return probe.reason


def _p1_response_defaults(args) -> None:
    """Select the alternate P1 announcement/response, independently of entry mode."""
    if not getattr(args, "p1_6a9", False):
        return
    if args.p1_status_bits45 not in (None, 1):
        raise SystemExit("--p1-6a9 requires --p1-status-bits45 1")
    if getattr(args, "p3_uninvited", False):
        raise SystemExit("--p1-6a9 conflicts with --p3-uninvited")
    args.p1_status_bits45 = 1


def _p4_entry_defaults(args) -> None:
    """Resolve the explicit P4 entry probe before any hardware is opened."""
    if not getattr(args, "p4_entry", False):
        return
    incompatible = ("pactor1_only", "pactor3_only", "offer_pactor2", "p3_uninvited",
                    "decline_grant", "mail_send", "mail_fetch", "message",
                    "p3_timing_trial", "p3_entry_timing_trial")
    if any(getattr(args, k, False) for k in incompatible):
        raise SystemExit("--p4-entry requires a granted entry probe without mail, "
                         "forced modes or P3 timing trials")
    if tuple(args.p3_entry) not in (("template",), ("p4chirp",)):
        raise SystemExit("--p4-entry cannot be combined with a different --p3-entry ladder")
    if not 1 <= args.p4_entry_attempts <= 4:
        raise SystemExit("--p4-entry-attempts must be between 1 and 4")
    if not 0 < args.p4_entry_timeout <= 60:
        raise SystemExit("--p4-entry-timeout must be positive and at most 60 seconds")
    args.p3_entry = ("p4chirp",)
    args.p1_grant_only = True
    args.hold = 12
    args.observe_after_link_down = False


def _arm_defaults(args) -> None:
    """The PACTOR-1 settings a mail arm does not share with an upgrade arm.

    Every other arm this modem flies is an experiment about reaching PACTOR-3,
    and its defaults are chosen for that: bits 4-5 announce the capability,
    because every 0x59A grant on record was drawn at 3, and the link takes the
    upgrade off the first acknowledged packet with traffic behind it. A mail arm
    wants neither, and the measurements say so rather than the taste.

    BITS 4-5. WS8EOC 3596500 on 2026-08-30, same
    gateway, band, hour, drive and gain: at 3, three arms, the counter frozen at
    #1 and nothing acknowledged; at 0, two arms, `#1 -> #2 -> #3` and
    ACKNOWLEDGED both times. Bit 4 is the top of the data type, so 0x3n declares
    PMC German over an ASCII field and the peer stops reading PACTOR-1 at all.
    A mail arm needs the data path, not the grant.

    THE UPGRADE. `arq._on_ack` offers one on every acknowledgement with payload
    queued, and during a mail exchange that is every acknowledgement -- so the
    link leaves PACTOR-1 uninvited on the login packet, discards the gateway's
    PACTOR-1 codewords for `ptc.UPGRADE_SILENCE_CYCLES` and spends those cycles
    against `max_retries`. With bits 4-5 clear no grant can be drawn either, so
    it is uninvited PACTOR-3, and this station's record says nothing at all about
    what a peer does with one: all 27 uninvited entries on tape are VOID, every
    one keyed at speed level 3 rather than `ptc.GRANT_ENTRY_SL`, 18 of them with
    our acknowledgement inside the peer's own packet, and 24 abandoning the
    waveform after one or two packets. What costs the mail arm its data path is
    the cycles, which are spent whatever the peer would have done.

    THE ANNOUNCEMENT FOLLOWS THE DOOR, BUT NOT EVERY DOOR. A mail arm that names
    a way into PACTOR-3 keeps the upgrade -- the whole of `UPGRADE_DOORS` -- and
    the three that a grant can come through, `--pactor3-only`, `--offer-pactor2`
    and `--p1-grant-only`, announce at 3 because a grant cannot be drawn without
    it. `--p3-uninvited` is the fourth and it announces nothing: it asks what a
    gateway does with a waveform it was never told about, and bits 4-5 at 3 is
    the telling. `--decline-grant` is not a door at all: it refuses the grant the
    announcement exists to draw, so it keeps 0 alongside `--pactor1-only`.
    `--p1-status-bits45` always wins.
    """
    if _mail_arm(args):
        if not args.pactor1_only \
                and not any(getattr(args, door) for door in UPGRADE_DOORS):
            args.pactor1_only = True
        if args.p1_status_bits45 is None and not (
                args.pactor3_only or args.offer_pactor2 or args.p1_grant_only):
            args.p1_status_bits45 = 0
        if not args.hold:
            args.hold = MAIL_HOLD_CYCLES
    if args.p1_status_bits45 is None:
        args.p1_status_bits45 = P1_STATUS_ANNOUNCE
    if getattr(args, "p1_setup_phase", "reply") == "call" and not args.pactor1_only:
        raise SystemExit("--p1-setup-phase call requires a PACTOR-1-only arm")


def _redact_argv(argv) -> tuple[str, ...]:
    """Retain typed tokens, hiding only the password value."""
    said = list(argv)
    hide_next = False
    for i, token in enumerate(said):
        if hide_next:
            said[i] = "***"
            hide_next = False
        elif token == "--mail-password":
            hide_next = True
        elif token.startswith("--mail-password="):
            said[i] = "--mail-password=***"
    return tuple(said)


def _note_host_free(host) -> None:
    """The Free Signal the launcher's channel sense read, off the environment.

    `tools/lib/attempts.sh` exports `HOST_FREE_IDENT`/`HOST_FREE_AT` where the
    sense found the channel's occupant free rather than busy, and clears them
    before every attempt so one arm cannot inherit another's reading.

    BOTH HALVES OR NEITHER. An ident with no instant is a station list; what
    makes it a go-ahead is that the burst was recent, and only the pair says so.
    A launcher that exported one of them badly costs the arm a transcript line
    and not its slot.
    """
    ident = os.environ.get("HOST_FREE_IDENT")
    try:
        at = float(os.environ["HOST_FREE_AT"]) if ident else None
    except (KeyError, ValueError):
        at = None
    host.host_free_ident, host.host_free_at = (ident, at) if at else (None, None)
    if at:
        print(f"  host free: {ident} last burst {time.time() - at:.1f} s "
              f"before start", flush=True)


@progress_to_stdout()
def run(args) -> int:
    # Written before any guard can refuse the arm, because a transcript whose
    # command line has to be reconstructed afterwards cannot settle what flew.
    # The password never reaches a transcript.
    typed = getattr(args, "_typed_argv", None)
    print("  argv: " + (shlex.join(typed) if typed is not None else
                        "unavailable (run received a namespace)"), flush=True)
    args.tx_drive = config.tx_drive(args.tx_drive)
    args.tx_latency_ms = config.tx_latency_ms(args.tx_latency_ms)
    _p1_response_defaults(args)
    _p4_entry_defaults(args)
    _p3_mail_defaults(args)
    _timing_trial_defaults(args)
    _arm_defaults(args)
    said = []
    for name, value in sorted(vars(args).items()):
        if name.startswith("_"):
            continue
        # Compare the resolved field name exactly: the password-file path is
        # useful provenance and contains no password.
        value = "***" if name == "mail_password" else value
        said.append(f"--{name.replace('_', '-')}={value!s}")
    print("  resolved args: " + shlex.join(said), flush=True)
    if getattr(args, "p4_entry", False):
        print("  EXPERIMENTAL P4 ENTRY: chirp after grant; gateway acceptance and "
              "connected P4 ARQ are unvalidated. QRM guard "
              + ("enabled" if args.qrm_guard else "explicitly disabled") + ".", flush=True)
    if not 0.0 <= args.p3_entry_delay <= MAX_ENTRY_DELAY_MS:
        # A NEGATIVE delay is a lead, and a lead on the entry packet is a key in
        # front of the boundary -- the one direction the admission reserve
        # cannot absorb. The ceiling is the slot with the packet in it.
        raise SystemExit(f"--p3-entry-delay {args.p3_entry_delay} is not in "
                         f"[0, {MAX_ENTRY_DELAY_MS:.0f}] ms")
    if args.p3_repeat_gear < 0:
        # A run cannot be shorter than none, and a negative bound fires on the
        # first packet of every link -- CS4 every cycle, which is a gear command
        # and not an experiment.
        raise SystemExit(f"--p3-repeat-gear {args.p3_repeat_gear} is negative")
    if not 0.0 < args.p1_drive <= 1.0:
        # Above 1.0 the P1 leg would clip past the calibrated peak; the rail is
        # the entry packet's, not a headroom pool.
        raise SystemExit(f"--p1-drive {args.p1_drive} is not in (0, 1]")
    for flag, watts in (("--p1-watts", args.p1_watts),
                        ("--p3-watts", args.p3_watts)):
        if watts is None:
            continue
        if not 0.0 < watts <= MAX_WATTS:
            raise SystemExit(f"{flag} {watts} is not in (0, {MAX_WATTS:.0f}] W")
        if not args.transmit:
            # There is no CAT connection in a dry run -- the rig object is what
            # holds it -- so the flag would be silently nothing rather than the
            # power it names.
            raise SystemExit(f"{flag} writes RFPOWER over CAT; pass --transmit")
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    # The connect burst carries the CALLED station's callsign, so a default here
    # would put a real, uninvolved station on the air. Only the modes that render
    # a connect need one; a watch that just listens does not.
    if not args.dxcall and not (args.listen_only or args.preflight or args.tune):
        raise SystemExit("--dxcall is required: the station this will call")

    # Only --transmit consumes a frequency -- the QSY before keying. A dry run
    # renders TX to WAV and tunes nothing, and bare --preflight/--listen-only
    # have no rig at all; demanding a dial there is what broke them the day the
    # dxcall default (whose gateway entry had quietly supplied one) went away.
    dial = args.dial or 0
    if args.transmit and not dial:
        by_call = ota.GATEWAYS.get(args.dxcall.upper(), {}) if args.dxcall else {}
        center = args.center or by_call.get(args.band)
        if center is None:
            raise SystemExit("no freq: pass --dial/--center or a known --dxcall/--band")
        dial = ota.center_to_dial(center)

    settle = RIGS[args.rig].get("settle", 0.10)
    fits, budget = _budget(args.cycle, args.tx_offset, settle)
    print(budget, flush=True)
    if not fits and args.transmit:
        # Refuse BEFORE the rig is opened, and refuse rather than shave a term.
        # Every number in that line has been wrong at least once, and the way a
        # schedule that does not close fails is by transmitting over the far end
        # while reporting that it stayed quiet -- so this hands the arithmetic to
        # the operator instead of quietly widening something to make the sum work.
        raise SystemExit("the cycle does not close at this offset -- not keying")

    rig = None
    if args.transmit:
        print("!! TRANSMIT ARMED -- a licensed operator must be on frequency; keep <=50 W\n")
        r = RIGS[args.rig]
        # The arm gate. `ptt_device` refuses a keying path that does not exist
        # -- which is what let --serial's placeholder default through a whole
        # 27-cycle session on 2026-08-10 -- and refusing HERE, before the rig
        # object exists, is what makes "the session never started" the failure
        # mode instead of "the session finished and lied". An explicit
        # --ptt-port bypasses the derivation and is checked at every key-down
        # instead (`Rig.ptt`).
        if not args.serial:
            raise SystemExit("NOT KEYING: --serial is required with --transmit "
                             "-- `ls /dev/cu.*` (macOS) or "
                             "`ls /dev/serial/by-id/*` (Linux) to find yours")
        try:
            ptt_port = args.ptt_port or ota.ptt_device(args.rig, args.serial)
        except PttError as exc:
            raise SystemExit(f"NOT KEYING: {exc}") from None
        if getattr(args, "rx_assessment", None):
            from ..core.rxreadiness import read_assessment
            try:
                read_assessment(args, args.serial)
            except Exception as exc:
                raise SystemExit(f"NOT KEYING: invalid RX assessment before CAT: {exc}") from exc
        rig = ota.Rig(r["model"], args.serial, args.baud or r["baud"],
                      ptt_type=args.ptt_type or r.get("ptt_type", "RIG"),
                      ptt_port=ptt_port)
        if not getattr(args, "rx_assessment", None):
            rig.set_freq(dial)
        # Read the dial back and refuse to key if it did not take. A sweep QSYs
        # between targets, and a set_freq that silently failed would spend the
        # whole next attempt calling one gateway on another one's frequency --
        # which looks exactly like "nobody answered".
        if getattr(args, "rx_assessment", None):
            from ..core.rxreadiness import verify_assessed_receiver_startup
            assessed = verify_assessed_receiver_startup(rig, args=args, serial=args.serial,
                expected_mode=r["mode"], model=r["model"], dial=dial, output_dir=outdir)
            got = str(assessed["actual"]["frequency_hz"])
        else:
            got = rig.get_freq()
        if got and abs(int(got) - dial) > band.QSY_TOLERANCE_HZ:
            raise SystemExit(f"QSY FAILED: asked {dial} Hz, rig reports {got} Hz "
                             f"-- not keying")
        # An empty readback is not a weaker mismatch, it is no answer at all: the
        # frequency this would transmit on is unknown. This printed a warning and
        # keyed anyway, which is the 2026-08-13 shape exactly -- four attempts on
        # a stale dial, some across the PACTOR channels, caught by the operator
        # watching the rig rather than by anything here. besra and sabir both
        # refuse an unreadable answer; this was the one that did not.
        if not got:
            raise SystemExit("QSY UNVERIFIED: the rig returned no frequency, so "
                             "the dial this would key on is unknown -- not keying. "
                             "CAT contention is the usual cause; nothing else may "
                             "hold the port while a transmit tool is starting.")
        # Mode AFTER frequency: the FT-891 keeps a mode per band, so a jump from
        # 80 m to 40 m can restore whatever that band was last left in.
        from ..core.rxreadiness import verify_receiver_startup
        if not getattr(args, "rx_assessment", None):
            verify_receiver_startup(rig, expected_mode=r["mode"], model=r["model"],
                                    pactor1_only=args.pactor1_only, output_dir=outdir)
        print(f"rig dial {dial} Hz  (signal centre ~{dial + int(spec.CENTER_FREQ_HZ)} Hz)"
              f"  TX drive {args.tx_drive:.2f} peak, PTT settle "
              f"{RIGS[args.rig].get('settle', 0.10):.2f}s, "
              f"cycle {args.cycle:.2f}s, transmit latency correction "
              f"{args.tx_latency_ms:.1f} ms")
        print(f"   P1 control signals sent "
              f"{'LSB' if args.p1_ack_lsb else 'MSB'}-first; which codeword goes "
              f"out is the protocol's alternation, not a setting", flush=True)
    else:
        print("dry run -- rig not keyed; TX rendered to WAV, RX from --reply-wav\n")

    if args.listen_only:
        # One unbroken recording, no transmitter involved.
        #
        # The session captures are per-cycle -- 1.25 s kept, then a gap while we
        # key and process -- and concatenating them splices those gaps out. That
        # is fine for a 120 ms control signal and useless for anything whose
        # meaning is carried in timing across seconds. Measured on synthetic
        # Morse: a continuous recording of WS8EOC reads WS8EOC, while the same
        # audio chopped into 1.25 s windows reads WS?EGC, WE8ON or W8MC as the
        # dropped gap grows. The last is the dangerous one -- a well-formed
        # callsign that is not the right callsign. A gateway signs in CW, and
        # that signature is the only identification in a session that does not
        # depend on shrike's own DSP being correct, so it is worth a mode.
        import sounddevice as sd
        in_dev = find_device(args.audio_in, "in", required=True)
        path = outdir / "listen.wav"
        print(f"recording {args.listen_only:.0f}s continuously -> {path}", flush=True)
        # Recorded here rather than through ota._record so the audio is in hand
        # BEFORE it is written: _record writes and returns a path, and the file it
        # returns is already normalised, which is exactly the trap the sidecar
        # exists to close. The levels have to be taken off the codec's own array.
        rec = sd.rec(int(args.listen_only * FS), samplerate=FS, channels=1,
                     dtype="float32", device=in_dev)
        sd.wait()
        audio = rec[:, 0]
        _save_capture(path, audio)
        for ev in rxfront.decode_events(audio):
            print(f"  {ev.t:7.2f}  {ev.kind:8s} {ev.text}", flush=True)
        return 0

    out_dev = find_device(args.audio_out, "out", required=True) if args.transmit else None
    if args.preflight:
        # The output device is resolved here whether or not --transmit was passed:
        # the point of this mode is the duplex stream, and opening one keys nothing.
        # --transmit gates only the PTT half.
        return _preflight(find_device(args.audio_in, "in", required=True),
                          find_device(args.audio_out, "out", required=True),
                          rig if args.transmit else None, settle,
                          key=args.transmit, blocksize=args.audio_block)
    if args.tune:
        if not args.transmit:
            raise SystemExit("--tune keys the rig; pass --transmit")
        # The session loop below gets its unkey from a `finally`; the tune path
        # sat outside one, so a Ctrl-C during the carrier reached `_play`'s and
        # `tune_carrier`'s unkeys but never the belt-and-braces one-shot in
        # `Rig.stop`. The one operation that keys the rig for ten seconds at a
        # time is the last one that should have the weaker guarantee.
        try:
            tune_carrier(rig, out_dev, args.tune, args.tune_drive)
        finally:
            rig.stop()
        return 0
    # Every mode that may run without a --dxcall has returned by here.
    dx = args.dxcall.upper()
    in_dev = find_device(args.audio_in, "in", required=True) if args.transmit else None

    tx = RadioTx(rig, transmit=args.transmit, out_dev=out_dev, outdir=outdir,
                 max_key=args.max_key, drive=args.tx_drive,
                 p1_drive=args.p1_drive, settle=settle)
    if getattr(args, "p3_mail", False):
        tx.reply_clock = ReplyClock()
        print("  P3 MAIL: starts at SL1, entry +603.125 ms initial reply pulse; accepted "
              "turn changes rotate the clock; normal mail completion and "
              "link/hold limits apply", flush=True)
    if getattr(args, "p3_timing_trial", None):
        tx.timing_trial = TimingTrial(args.p3_timing_trial,
            unbounded=getattr(args, "p3_timing_unbounded", False),
            reply_delay_ms=getattr(args, "p3_timing_reply_delay", None) or 0.0)
    elif getattr(args, "p3_entry_timing_trial", None):
        tx.timing_trial = EntryTimingTrial(args.p3_entry_timing_trial)
    tx.p3_control_waveform = getattr(args, "p3_control_waveform", "historical")
    tx.p3_control_placement = getattr(args, "p3_control_placement", "audio-start")
    tx.p3_control_tail = getattr(args, "p3_control_tail", "off")
    tx.p3_control_stagger = getattr(args, "p3_control_stagger", "off")
    tx.p3_follow_offset = getattr(args, "p3_follow_offset", "all")
    tx.p3_keep_slots = getattr(args, "p3_keep_slots", "all")
    print(f"  P3 control profile: waveform={tx.p3_control_waveform}, "
          f"placement={tx.p3_control_placement}, "
          f"tail={tx.p3_control_tail}, stagger={tx.p3_control_stagger}, "
          f"follow-offset={tx.p3_follow_offset}, "
          f"keep-slots={tx.p3_keep_slots}", flush=True)
    if tx.p3_control_stagger != "off":
        print(f"  --p3-control-stagger {tx.p3_control_stagger}: the leading "
              "carrier alternates every ARQ cycle from this foot, generated "
              "rather than projected from the peer. Each burst prints the "
              "arrangement it went out on.", flush=True)
    tx.p4_probe = (EntryProbe(args.p4_entry_attempts, args.p4_entry_timeout)
                   if getattr(args, "p4_entry", False) else None)
    host = PtcHost(peer=tx, mycall=args.mycall)
    _note_host_free(host)
    host.stay_in_pactor1 = args.pactor1_only
    host.stay_in_pactor3 = args.pactor3_only
    host.no_p3_fallback = args.no_p3_fallback
    host.p3_changeover_cs5 = getattr(args, "p3_changeover_cs5", False)
    host.p3_qrt_confirm = getattr(args, "p3_qrt_confirm", False)
    if host.p3_qrt_confirm:
        print("  P3 QRT CONFIRM experiment: after the QRT ACK, send the stock "
              "terminal pattern with opposite ACK parity; teardown remains bounded.")
    if host.p3_changeover_cs5:
        print("  --p3-changeover-cs5: experimental CS5 reply to decoded P3 "
              "changeovers; reply timing unchanged.", flush=True)
    host.p3_changeover_p1_cs = getattr(args, "p3_changeover_p1_cs", False)
    if host.p3_changeover_p1_cs:
        print("  --p3-changeover-p1-cs: the CS1/CS2 answering a P3 changeover "
              "goes out as a 12-bit PACTOR-1 codeword, keyed in the PACTOR-3 "
              "answer slot at nominal frequency. OUT OF SPEC -- a level-3 link "
              "answers in DBPSK on channels 5 and 12. Labelled [pactor-1] in "
              "TX[n].", flush=True)
    if args.no_p3_fallback:
        print("  --no-p3-fallback: after entering P3, transmit only P3 "
              "until this contact ends; hold and teardown limits remain active.",
              flush=True)
    host.p1_act_on_grant = not args.decline_grant
    host.p1_6a9 = getattr(args, "p1_6a9", False)
    if host.p1_6a9:
        if tx.p4_probe is not None:
            tx.p4_probe.trigger_word = "0x6A9"
        print("  P1 RESPONSE EXPERIMENT: bits45=1 (0x11 at counter 1); "
              "selected entry response 0x6A9. Entry waveform and P1-only "
              "policy unchanged.", flush=True)
    host.p1_grant_only = args.p1_grant_only
    host.offer_pactor2 = args.offer_pactor2
    host.announce_lower = args.announce_lower
    host.arq.cfg.entry_ladder = args.p3_entry
    host.arq.cfg.entry_sl = args.p3_entry_sl
    if args.p3_entry_sl != ArqConfig.entry_sl:
        print(f"  --p3-entry-sl {args.p3_entry_sl}: an uninvited upgrade opens "
              f"at SL{args.p3_entry_sl}; a granted one still opens at "
              f"SL{GRANT_ENTRY_SL}.", flush=True)
    host.arq.cfg.traffic_sl = args.p3_traffic_sl
    if args.p3_traffic_sl != ArqConfig.traffic_sl:
        print(f"  --p3-traffic-sl {args.p3_traffic_sl}: an answered entry "
              f"packet runs traffic at SL{args.p3_traffic_sl}; the peer's CS4 "
              "still climbs from there.", flush=True)
    host.arq.cfg.speed_up = args.p3_speed_up
    host.arq.cfg.p3_max_try = getattr(args, "p3_max_try", ArqConfig.p3_max_try)
    host.arq.cfg.p3_max_down = getattr(args, "p3_max_down", ArqConfig.p3_max_down)
    print(f"  P3 adaptation: MAXTry={host.arq.cfg.p3_max_try} total speed-up "
          f"trial transmissions; MAXDown={host.arq.cfg.p3_max_down} consecutive "
          "receive errors before CS5.", flush=True)
    host.p3_tx_gear_hold = getattr(args, "p3_tx_gear_hold", False)
    if host.p3_tx_gear_hold:
        print("  P3 TX GEAR HOLD trial: repeated CS4/CS5 retain the pending "
              "packet until an ACK or different command; MAXTry bounds "
              "the speed-up trial.", flush=True)
    if args.p3_speed_up == "hold":
        print("  --p3-speed-up hold: no CS4 from either gear seam; the link "
              "stays at the level the entry opened it on and the peer's own "
              "gear commands are all that move it.", flush=True)
    elif args.p3_mail:
        print("  P3 MAIL speed-up auto: request the next level after three "
              "consecutive cycles delivering new payload; repeats and unread "
              "cycles reset the run.", flush=True)
    host.arq.cfg.repeat_gear = args.p3_repeat_gear
    if args.p3_repeat_gear and args.p3_speed_up == "auto":
        print(f"  --p3-repeat-gear {args.p3_repeat_gear}: a run of identical "
              f"packets answered identically draws CS4 on the "
              f"{args.p3_repeat_gear + 1}th, which acknowledges the same "
              "counter; the alternation resumes behind it.", flush=True)
    host.arq.cfg.long_cycle = args.long_cycle
    tx.p1_ack_msb = not args.p1_ack_lsb
    tx.p1_status_bits45 = args.p1_status_bits45
    tx.p1_setup_phase = getattr(args, "p1_setup_phase", "reply")
    if tx.p1_setup_phase == "call":
        print("P1 SETUP PHASE EXPERIMENT: preserve call phase until first "
              "payload ACK or role change; later alignment is normal", flush=True)
    tx.p1_status_from = args.p1_status_from
    placement.CASE0_STAGGER = args.p3_entry_stagger
    placement.PROTOCOL_RISE = args.p3_rise
    placement.DATA_FLUSH = {"entry": placement.ENTRY_FLUSH,
                            "reference": placement.REFERENCE_FLUSH,
                            }.get(args.p3_data_flush)
    if args.p3_data_flush == "entry":
        print("  --p3-data-flush entry: every data packet ends its trellis on "
              "the entry packet's measured flush instead of zeros.", flush=True)
    elif args.p3_data_flush == "reference":
        print("  --p3-data-flush reference: levels 2 to 6 end their trellis on "
              "the tail DL6MAA's own level-3 packets carry; speed level 1 keys "
              "zeros, which is what WS8EOC keys.", flush=True)
    tx.entry_delay_n = round(args.p3_entry_delay / 1e3 * FS)
    tx.qrm_guard = args.qrm_guard
    tx.breakin_at_boundary = args.breakin_at_boundary
    # SAID AT THE TOP OF THE ARM, because both of these are settings whose whole
    # failure mode is nobody remembering they are off. A grant cannot be drawn
    # without an announcement, and nothing refuses a key over the peer without
    # the guard; neither absence produces a line of its own later.
    if _mail_arm(args) or not args.p1_status_bits45:
        door = "PACTOR-1 only" if args.pactor1_only else "upgrades enabled"
        hold = f"; hold {args.hold} cycles (--hold)" if _mail_arm(args) else ""
        print(f"  {'MAIL ' if _mail_arm(args) else ''}ARM: {door}{hold}.", flush=True)
        if args.pactor1_only:
            doors = " ".join(f"--{d.replace('_', '-')}" for d in UPGRADE_DOORS)
            print(f"  NO ENTRY PACKET WILL BE KEYED: nothing leaves PACTOR-1 in "
                  f"this arm. The doors are {doors}, and a mail arm that names "
                  f"none of them takes this one.", flush=True)
        print(f"  Announcement: --p1-status-bits45 {args.p1_status_bits45}. "
              + ("Use --p1-status-bits45 3 to request a grant."
                 if not args.p1_status_bits45 else "Grant requests enabled."),
              flush=True)
    if not args.qrm_guard:
        print("  QRM GUARD OFF: nothing will refuse a key over the peer's own "
              "codeword. --qrm-guard puts it back.", flush=True)
    if not args.long_cycle:
        print("  SHORT CYCLE PREFERRED: local policy does not request the "
              "3.75 s cycle. A credible unexpected wideband long header "
              "can withhold a short reply; only a CRC-valid packet adopts "
              "the peer's observed cycle length.", flush=True)
    if not args.breakin_at_boundary:
        print(f"  BREAK-IN AT {BREAKIN_LEAD_S * 1e3:.0f} ms: the changeover "
              f"packet keys where a reference modem keys its own at us, which "
              f"is 21-30 ms in front of the window KB5LZK acknowledges our "
              f"codewords in. This is the control arm; drop the flag to key at "
              f"the peer's read instant.", flush=True)
    if args.p3_entry_delay:
        print(f"  P3 ENTRY DELAY {args.p3_entry_delay:+g} ms after the slot "
              f"boundary (default {DEFAULT_ENTRY_DELAY_MS:g} ms). "
              "Use --p3-entry-delay 0 for the historical zero-delay comparison.",
              flush=True)
    if args.p1_drive != 1.0:
        print(f"  P1 DRIVE {args.p1_drive:.2f}: the PACTOR-1 legs key at "
              f"{20 * math.log10(args.p1_drive):+.1f} dB against the PACTOR-3 "
              f"peak. The entry packet and every other PACTOR-3 burst are "
              f"untouched.", flush=True)
    # TWO COUNTERS, AND --retries ONLY EVER REACHED THE FIRST. `max_connect_retries`
    # bounds the CALL; `max_retries` bounds the live link and is what `_give_up`
    # spends into a QRT. They are deliberately separate -- the connect budget is
    # sized against `_ConnectEvidence`, and `RECLAIM_CODEWORDS` and
    # `UPGRADE_SILENCE_CYCLES` are sized to fall below the link one -- so the
    # second gets its own flag rather than a share of the first. What was wrong
    # was silence: WS8EOC, 2026-09-11, was flown with `--retries 20` on every arm
    # and ended on the default 8, and nothing the operator could read said which
    # number had been moved.
    if args.retries:
        host.arq.cfg.max_connect_retries = args.retries
    if args.link_retries:
        host.arq.cfg.max_retries = args.link_retries
    if args.retries or args.link_retries:
        print(f"  RETRY BUDGETS: connect {host.arq.cfg.max_connect_retries} "
              f"cycles (--retries); live link {host.arq.cfg.max_retries} cycles "
              f"(--link-retries), and that is the one that signs a session off "
              f"with a QRT.", flush=True)

    # Winlink mail rides the held link: a B2F session attached as the host's
    # application, fed by `deliver` and answering through the ARQ, with
    # `app_turns` working the changeover per cycle. The exchange's turn order
    # past the greeting follows the published FBB protocol, and a live session
    # has run the whole of it -- over ARDOP, not over this transport. WW2MI's CMS
    # on 2026-08-19 proposed a message, was answered FS, sent the SOH/STX/EOT
    # blocks and closed clean; the message parsed and was written out. See
    # `winlink/client.py`, which is where that turn order lives. What this flag
    # exists to change is the PACTOR half: on this transport the exchange has
    # reached a break-in field and no further (`tests/shrike/test_holdbudget.py`).
    mail = None
    if _mail_arm(args):
        from ..winlink import B2FSession, MailClient, load_outbound
        outbox = [load_outbound(p, args.mycall, to=args.mail_to,
                                subject=args.mail_subject)
                  for p in (args.mail_send or [])]
        mail = MailClient(
            B2FSession(args.mycall, role="calling", target=args.dxcall,
                       password=args.mail_password, outbox=outbox,
                       grid=config.grid(),
                       client_sid=config.client_sid(args.mail_sid)),
            host.arq.on_host_data)
        host.app = mail
        print(f"mail: announcing as {mail.session.sid}", flush=True)
        if getattr(args, "mail_wait_greeting", False):
            print("mail: --mail-wait-greeting: ACK the greeting; defer automatic "
                  "break-in until the mail client has a reply", flush=True)

    reply = _load(args.reply_wav) if args.reply_wav else None
    sessrx = _SessionRx(host)
    sessrx.p3_wideband_prekey = getattr(args, "p3_wideband_prekey", False)
    if sessrx.p3_wideband_prekey:
        print("P3 bounded wideband pre-key read enabled (short IRS cycles)", flush=True)
    tx.sessrx = sessrx
    if args.replay:
        live = _ReplayInput(args.replay, realtime=args.replay_realtime)
        print(f"replay: driving the session from {args.replay} "
              f"({live.audio.size / FS:.1f} s)"
              f"{' in real time' if args.replay_realtime else ''}", flush=True)
    else:
        # Unconditional: what this file answers is not recoverable from the
        # windows afterwards at any price, and a session that was not recorded
        # cannot be run again -- the far end and the band have both moved on.
        live = (_LiveInput(in_dev, out_dev, blocksize=args.audio_block,
                           record=outdir / "stream.wav",
                           tx_latency_n=round(args.tx_latency_ms * FS / 1e3))
                if args.transmit else None)
    if live:
        tx.live = live
        tx.defer_p3_cs = live is not None
        print(_clock_line(live, round(tx.settle * FS)), flush=True)
    cycles_run = 0
    code = 0
    evidence = _ConnectEvidence()
    # Out here with the other things the summary is entitled to, because the
    # summary runs from a `finally` and the grid is placed several dozen lines
    # into the `try`.
    raster: Optional[_MasterGrid] = None
    # Why the session stopped, in its own words. A hold that ran out and a hold
    # the link fell out from under are the same length of log and the same
    # silence on the speaker; only this tells them apart.
    ended = "the cycle budget ran out"
    linked = False
    # The rig's own power, moved at the link's protocol boundaries; the host
    # layer calls in from the transitions and this end owns the CAT.
    power = _phase_power(args, rig)
    host.phase_power = power
    try:
        print(f"connecting {args.mycall} -> {dx} ...")
        if power is not None:
            # BEFORE the call and not from `connect_burst`, which is the seam
            # that keys it: a level written a pipe write in front of the carrier
            # is a level the rig may not have taken yet when the carrier goes up.
            power.select(Protocol.PACTOR1, "before the first call")
        host.arq.on_host_connect(args.mycall, dx)          # ISS: sends the connect
        # THE RASTER IS COUNTED IN SAMPLES. Cycle n begins at capture-stream
        # sample `anchor + n * slot_n`, and cycle zero is where our connect
        # burst ended -- which is the instant the far end starts timing us from.
        #
        # This replaces an absolute wall-clock deadline, which was itself a fix
        # for a relative sleep. The deadline killed the accumulation (12-19 ms a
        # cycle, walking the peer's answer out of the window in about twenty)
        # but not the jitter: measured on this machine, sleeping to a 100 ms
        # deadline under load lands within 2 ms at the median and 70 ms at the
        # worst, and 70 ms is a quarter of the answer window, delivered without
        # warning. Waiting on the codec instead cannot have that tail -- audio
        # arrives at 48 kHz whatever the scheduler is doing, and nothing below
        # is allowed to sleep or read a clock.
        slot_n = round(args.cycle * FS)
        settle_n = round(tx.settle * FS)
        # The floor on a listen window, and it is the CONTROL SIGNAL, not the
        # protocol's answer window. `spec.CS_WINDOW_S` is 0.29 s because
        # 0.96 + 0.29 = 1.25 -- it is the whole span from our carrier dropping to
        # the next boundary, and the keying settle and the block held back for the
        # bridge are spent INSIDE it. Demanding 0.29 s of reading on top of them
        # asks for 0.415 s of a 0.29 s window, which no one-slot cadence can ever
        # supply: it leaves 165 ms, so the guard below skipped a slot every cycle
        # and the schedule could not run at the peer's own rate at all. What the
        # window has to be long enough for is the thing it is listening for.
        cs_burst_n = round(spec.P1_CS_S * FS)
        offset_n = round(args.tx_offset * FS)
        packet_n = round(spec.P1_PACKET_S * FS)
        # THE ANCHOR IS OUR OWN DATA STARTING, AND IT NEVER MOVES AGAIN.
        #
        # `tx_end` is where our carrier dropped, so the connect's data began one
        # packet earlier; anchoring on the drop itself put the second call a whole
        # packet -- 0.96 s -- later than one cycle after the first. That never
        # showed, because the old raster re-anchored on the peer's first burst and
        # swallowed it. A free-running master has nothing to swallow it with: the
        # first cycle has to be right, and every one after is this plus 1.25 s.
        anchor = (tx.tx_end - round(tx.last_dur * FS) if tx.tx_end is not None
                  else (live.samples if live is not None else 0))
        d_max_n = _d_max_n(args.cycle, tx.settle)
        raster = _MasterGrid(anchor, slot_n, offset_n, packet_n=packet_n,
                             cs_n=cs_burst_n, d_max_n=d_max_n)
        raster.sessrx = sessrx
        print(f"master grid: anchor @ sample {anchor}, cycle {args.cycle:.4f} s, "
              f"free-running; a turnaround up to "
              f"{d_max_n / FS * 1e3:.0f} ms is searchable", flush=True)
        onsets: list[int] = []             # the peer's bursts, from last cycle
        bursts: list[tuple[int, int]] = []      # ...and how long each one ran
        # The window that closed last cycle, kept for `_breakin_audio`: a
        # transmission that is not on our grid can begin in one window and finish
        # in the next, and the two loops below share the pair for the same reason
        # they share the raster -- the hold's first window follows the setup
        # phase's last one on the same stream.
        prev, prev_start = np.zeros(0, np.float32), 0
        evidence_of = _CycleEvidence(live, tx, raster, outdir)
        # ONE FLOOR FOR THE WHOLE SESSION, both loops, for the raster's reason:
        # the connect phase's windows are the same channel the hold phase listens
        # to, and a floor that started again at CONNECTED would start again in the
        # cycles a peer is most likely to be filling.
        answer_band = _AnswerBand()
        slot = 1
        # A last call still owns an answer window. Keep the decoder and the
        # CONNECTING state alive for two full receive-only cycles before CW or
        # close. The partial window after the last carrier plus those cycles
        # can contain the first candidate; a bounded extension then gives it
        # further distinct cycles to satisfy the unchanged corroboration rule.
        connect_tail = None
        for c in range(1, args.max_cycles + _ConnectTail.MAX_CYCLES + 3):
            if tx.p4_probe is not None and tx.p4_probe.requested:
                ended = _run_p4_probe(tx, live, raster)
                break
            if c > args.max_cycles and (live is None or linked):
                break
            if (live is not None and not linked and connect_tail is None
                    and host.arq.state == State.CONNECTING
                    and (c > args.max_cycles
                         or (host.arq._connect_retries >= host.arq.cfg.max_connect_retries
                             and not host.arq._peer_heard))):
                connect_tail = _ConnectTail(max(live.pos, tx.tx_end or 0), slot_n)
                sessrx._final_connect_listen = True
                print("  FINAL CALL LISTEN: two receive-only cycles before "
                      "identification/close; a decoded answer continues the session",
                      flush=True)
            cycles_run += 1
            sessrx.new_cycle()
            # Where our last carrier dropped, read before this cycle can key over
            # it: the window collected below sits after the PREVIOUS transmission,
            # and it is that one an answer in it would be answering.
            keyed_at = tx.tx_end if tx.tx_end is not None else 0
            # THE ESCAPE FROM NEVER BEING ANSWERED. A master may not retime itself
            # towards the peer, so when our own transmission is what covers the
            # peer's burst there is nothing in the protocol that breaks the
            # deadlock: both grids are stable, so a phase that hides the peer hides
            # it forever. A grid that has spent BLIND_CYCLES hearing nothing stops
            # transmitting for HUSH_CYCLES and only listens, and what it hears then
            # is the one thing allowed to place the grid. Only meaningful with an
            # input to listen with -- a dry run has no channel and no peer to find,
            # and only while nothing has ever been heard on this grid, which is
            # `_MasterGrid._blind`'s business and is where that rule lives.
            final_listen = connect_tail is not None
            hush = live is not None and (raster.hush_left > 0 or final_listen)
            # Gate on having an INPUT, not on transmitting: a replay is a live
            # input the rig is not attached to, and keying off --transmit meant
            # the replay was opened and never read.
            if live is not None:
                # ONE raster for the whole session, in both phases. Link setup
                # used to run on a ~2.3 s cycle (a 1 s burst plus a 1.2 s listen)
                # and then switch to 1.25 s on connecting, so the cadence audibly
                # changed mid-session -- the operator heard it change after the
                # fifth transmission and called it before any measurement did.
                #
                # The long setup window was justified by replies "arriving
                # 0.49-0.85 s after the window opens", which looked like a spread
                # too wide for a 0.29 s slot. It was not a spread: reply onset
                # across consecutive cycles walked monotonically -- 620, 560, 500,
                # 445, 385, 325, 265 ms -- as our long cycle dragged the answer
                # through the window. Those two numbers were the same walk sampled
                # at opposite ends of one run. So the window is the description's.
                #
                # `settle_n` comes off the end because PTT has to be up before
                # the audio starts -- the window closes early by exactly the
                # keying delay so the burst itself lands ON the boundary.
                boundary = raster.boundary(slot)
                # NEVER TRANSMIT LATE. A free-running grid stays ahead of the
                # reader on its own, but placing it -- the one move that shifts the
                # anchor, and only from a hush -- can pull the boundary back behind
                # where the reader already stands, by up to half a cycle. The
                # window below would then clamp to its floor and read straight past
                # it, which is keying into the burst we just placed ourselves
                # against. Skip to the next slot and leave the channel alone for a
                # cycle instead. The working record's burst-lock replay learnt the
                # same rule the audible way: a schedule that transmits whenever it
                # is late runs its transmissions back to back.
                # The margin is a WHOLE CONTROL SIGNAL, and it is not slack.
                #
                # It looks like a "never transmit late" guard being needlessly
                # strict -- skipping a slot whenever the loop comes within 120 ms
                # of the boundary, which on a zero-slack budget is often, and the
                # irregular cadence that produces is audible. Loosening it to
                # `settle + block` was tried ON THE AIR and is much worse: with
                # only keying-time required, the listen window is whatever happens
                # to be left, and it collapsed to its 85 ms floor four cycles
                # running. Measured windows across that run:
                #
                #     1.20 1.28 0.08 0.26 1.20 1.28 1.28 1.28 0.08 0.08 0.08 0.08
                #
                # 85 ms cannot hold a 120 ms control signal, so those cycles were
                # deaf by construction, and back-to-back transmission is what the
                # operator heard. The margin guarantees a window big enough to
                # HEAR THE ANSWER IN, which is the point of having a window.
                #
                # The irregular cadence is therefore a symptom of the budget
                # having no slack, not of this rule. Fix it by making the cycle
                # cheaper, not by removing the guarantee.
                #
                # The arithmetic, and the holdback term it must not have, are in
                # `_keyable_slot`.
                slot = _keyable_slot(live, raster, slot, settle_n)
                boundary = tx.aim(raster, slot)
                # Stop a block SHORT of the boundary and sleep the rest: a read
                # cannot return until the block holding its last sample lands,
                # so reading right up to the boundary always arrives late by an
                # arbitrary fraction of a block. The held-back samples are not
                # lost -- they are still queued, and the next window starts from
                # `pos`, so the audio stays continuous and the grid stays
                # absolute.
                if hush:
                    # Nothing keys, so nothing has to be early for the key: run
                    # the window right up to the boundary. With `slot += 1` below
                    # that makes the hush one unbroken recording across cycles,
                    # which is what the provoked-listen procedure of `BLIND_CYCLES`
                    # produces by hand.
                    want = boundary - live.pos
                else:
                    # No floor here. The guard above has already established that
                    # this is at least a control signal long, and clamping a short
                    # window UP is reading past the boundary -- which is the one
                    # thing the guard exists to prevent.
                    want = (round(args.listen * FS) if args.listen else
                            boundary - settle_n - live.holdback
                            - round(PREKEY_RESERVE_S * FS) - live.pos)
                n0 = live.samples
                # Listen in slices and act the moment a peer answers. Taking the
                # whole window first and decoding after put our acknowledgement
                # 1.5 s late -- outside the 1.25 s cycle the peer is timing us
                # against, which is no acknowledgement at all.
                seg = _listen_until_answer(live, want, host, sessrx,
                                           FEED_MAX_SLOTS * raster.slot_n)
                if not args.replay:
                    _assert_capturing(seg, live, n0)
                tx.reply_at = time.time()
            else:
                seg = reply if reply is not None else np.zeros(int(args.listen * FS), np.float32)
                seg_start = 0             # a dry run has no stream to be placed on
                sessrx.feed(seg)          # a dry run has no listen loop to feed it
                # No input means no sample clock. Nothing in a dry run is timing
                # critical -- there is no far end to be late for -- so the wall
                # clock paces it, and only here.
                time.sleep(args.cycle)
            # ONLY WHAT THE DECISION NEEDS GOES BETWEEN THE DEADLINE AND THE KEY.
            #
            # Everything from here to `host.tick()` is spent between the peer's
            # transmission ending and our PTT, so what is here is exactly what the
            # FSM has to answer from: the last of the audio, the decode of it, and
            # the frame scan. The diagnostics, the capture write and the onset
            # measurement used to be here too and now sit below the key, where the
            # time is free.
            before = sessrx.count
            # ...and separately, what a DECODER took, which is what the connect
            # search below is gated on. `count` moves for every event kind the
            # front end raises, and `detect`, `fsk` and `p1reply` are shape lines
            # `rxfront`'s own contract forbids reading as a station -- so a burst
            # too weak to decode consumed the search of the cycle it landed in.
            # `captures/onair-0913-2157` is what that costs: WS8EOC answered from
            # cycle 6 at a turnaround repeatable to 0.12 ms, the three cycles
            # holding its strongest bursts each raised a `p1reply` and were never
            # searched, and the link came up in cycle 30 instead of 24.
            #
            # `words_at` is every twelve-bit word a decoder took, `cs` and
            # `unassigned` alike, and `rcvd_total` is payload a CRC-valid packet
            # delivered. The third station kind, `connect`, leaves CONNECTING,
            # which the search's own guard below reads directly.
            decoded_before = (len(sessrx.words_at), host.rcvd_total)
            # The frame scan goes in front of the bridge, because the bridge runs
            # to the key instant and everything after it is spent out of the rig's
            # settle -- 8 ms of it on the FT-891, against a scan that measures
            # 20.3. In front, it is spent out of `PREKEY_RESERVE_S`, which is
            # channel time the bridge gives back below.
            if live is not None:
                # Where this window sits on the stream, fixed BEFORE anything can
                # key: every decode below can reach the FSM, and `_tx` flushes the
                # capture queue and moves `pos`, after which the window's own
                # origin can no longer be recovered from it.
                seg_start = live.pos - seg.size
                # Read to where the PEER stopped transmitting rather than to our
                # own key: its packet ends `d` before our boundary, so nothing
                # after that instant can carry any of it, and the scan below has
                # `d - settle` to run in -- 65 ms at the measured turnaround.
                seg, seg_start, bridge = _collect(
                    live, sessrx, seg, seg_start,
                    sessrx.control_bridge_until(
                        boundary - max(raster.d, settle_n),
                        boundary - _prekey_lead(live, settle_n),
                        max(live.pos, int(live.sample_now())), raster))
                if seg.size:
                    _scan_frame(sessrx, seg, seg_start)
                # ...and now the rest of it, up to the key. This is not spare
                # audio: on a locked raster the peer's control signal ends 65 ms
                # before the boundary and the window closes 43 ms before it, so
                # the last of the thing we are answering is in here -- and a
                # changeover head needs 155 ms past its anchor, which is why it is
                # collected before the anchored read rather than left to the next
                # cycle.
                seg, seg_start, tailed = _collect(
                    live, sessrx, seg, seg_start,
                    sessrx.control_collect_until(
                        boundary - _prekey_lead(live, settle_n),
                        max(live.pos, int(live.sample_now())), raster))
                bridge += tailed
            heard, at = (sessrx.control_signal_in(seg, seg_start, raster)
                         if live is not None else (None, None))
            if heard is not None:
                _align_shift(raster, sessrx, tx, slot, at)
            seg = _receive_changeover(
                live, raster, tx, host, sessrx, slot, seg, seg_start,
                prev, prev_start, settle_n)
            boundary = tx.boundary
            if _reverse_before_key(raster, host, tx, slot):
                boundary = tx.boundary
            if final_listen and host.arq.state in LINKED:
                hush = False
            if live is not None and not hush:
                # LAST, so nothing below it can spend the slot it just checked.
                slot, seg, seg_start = _regrid(live, raster, tx, host, sessrx,
                                               slot, seg, seg_start, settle_n)
                boundary = tx.boundary
                # The hand-back's own scan can yield too, and it ran after the
                # check above.
                if _reverse_before_key(raster, host, tx, slot):
                    boundary = tx.boundary
            if live is not None:
                # Where we actually stand when the key is about to go up. This
                # is the claim of the design and it is printed every cycle
                # rather than inferred later -- and it is measured against the
                # sample grid, so it cannot accumulate the way a sleep does.
                # Taken LAST, after the decode: taken before it, the number said
                # nothing about the instant the carrier would actually appear.
                #
                # Against the instant THIS cycle was aimed at. A hushed cycle
                # keys nothing and reads to the boundary itself, so measuring
                # it against the key instant branded every hush ~+40 ms off a
                # grid it stood exactly on -- onair-0811-1215 printed a walk to
                # +133.8 ms that way, and it was read live as the grid
                # drifting when every sample of it was settle plus decode time.
                off = (live.sample_now()
                       - (boundary if hush else boundary - settle_n)) / FS
                print(f"    [grid] slot {slot} boundary @ sample {boundary}, "
                      f"CS due @ {raster.rx_due(slot)}"
                      f"{'' if raster.locked else ' (nominal, not acquired)'}; "
                      f"captured {seg.size / FS:.3f} s, bridged {bridge:.0f} ms, "
                      f"off-grid {off * 1e3:+.1f} ms"
                      f"{' -- HUSHED, not keying' if hush else ''}", flush=True)
            # Nothing to wait out: the listen above ran until the codec had
            # delivered the boundary, so we are already standing on it. Advance
            # by one slot, or by two while the link is still being set up.
            #
            # ONE SLOT, ALWAYS. There is no slow setup regime.
            #
            # A working PACTOR-1 implementation sends the call packet every 1.25 s
            # from the first cycle, in every cycle, and listens in the gap; a
            # skipped cycle presents the responder with an empty packet slot, and
            # it reads at a fixed offset without looking elsewhere for it. See
            # docs/protocols/pactor/pactor1-timing.md §3 and §8.1.
            #
            # This used to widen to two slots whenever we were not locked, on the
            # reasoning that a station which has not synchronised to us needs
            # somewhere wider to land its first answer. That reasoning had the
            # roles backwards: the RESPONDER synchronises to the MASTER's clock, and we
            # are the master whenever we placed the call. Giving it a wider cadence
            # to aim at does not help a receiver that is aiming at ours.
            slot += 1
            if not hush:
                host.tick()                                 # advance the cycle grid
                tx.emit_pending_cs()
            # ...and the other half of the reversal: taking the link is decided
            # inside the tick, by the changeover packet the tick just sent.
            line = _grid_reversal(raster, host)
            if line:
                print(f"    [grid] {line}", flush=True)
            _drain_host_log(host)
            # DEAD TIME. Nothing below gates the key, so all of it runs after the
            # transmission rather than in front of it -- the capture stream keeps
            # filling its queue meanwhile, so no audio is lost by reading late.
            #
            # Which is where the two decodes the cycle's own answer does NOT hang
            # on belong: the end-of-cycle flush, and the scan for a protocol we
            # are not in. 26.7 ms and 48.0 ms measured, against the 43 the window
            # leaves before the key and 0.96 s of our own carrier after it. See
            # `PREKEY_RESERVE_S` and `_SessionRx.upgrade_scan` -- an upgrade is
            # followed a cycle later, never not at all.
            sessrx.flush()
            # ...which is the third place a direction is decided: the flush
            # reads the last of the peer's burst, and a CS3 in it yields now.
            line = _grid_reversal(raster, host)
            if line:
                print(f"    [grid] {line}", flush=True)
            if seg.size:
                _scan_frame(sessrx, seg, seg_start, upgrade=True)
            # The alternate-protocol scan can yield after the flush did not.
            # Rotate before updating the receive grid for the next window.
            line = _grid_reversal(raster, host)
            if line:
                print(f"    [grid] {line}", flush=True)
            answered = None
            if ((len(sessrx.words_at), host.rcvd_total) == decoded_before
                    or (final_listen and host.arq.state == State.CONNECTING)):
                # The onset detector rejected whatever is here, so nothing above
                # ever offered it to a decoder. On 2026-07-30 that discarded a
                # CS1/ack from WS8EOC at zero bit errors: the live session
                # reported nothing decoded and `--acquire` found it in the very
                # capture that session had just written.
                #
                # So ask the codeword directly. This is the acquisition search
                # used exactly as its own contract requires -- once per cycle,
                # over the window where an answer is due, and never once
                # connected -- and its accept is zero bit errors in twelve at
                # mutual distance eight, which is a stronger gate than the shape
                # test that rejected the burst.
                found, t0 = None, 0.0
                # A CYCLE WE DID NOT KEY STILL HAS AN ANSWER BAND, and it hangs
                # off the slot we would have called on. The peer keeps answering
                # the raster through a hush and through the final listen alike --
                # it is synchronised to our clock, not to our PTT -- so the band
                # is the same narrow turnaround, one per slot, measured from the
                # carrier end the slot projects.
                #
                # `keyed_at` freezes where the hush began, and a window measured
                # from it falls off the front of the band: four decodable CS4s of
                # `onair-0912-2329` went unread that way while the session sat in
                # CONNECTING. The hush is where this search is needed most -- it
                # is the phase in which our own carrier is provably not covering
                # the peer -- so `_answer_origins` says what this window could
                # be following, and it has the rest of it.
                search_end = keyed_at
                # `live is not None` is what says `boundary` exists: a dry run
                # never reaches the grid and has no window to search anyway.
                if (live is not None and seg is not None
                        and host.arq.state != State.CONNECTED):
                    for search_end in _answer_origins(
                            keyed_at, boundary, raster.slot_n, raster.p1_data_n,
                            final_listen or hush):
                        t0, span = _acquisition_window(
                            seg.size, seg_start - search_end, d_max_n,
                            ACQUIRE_READ_TAIL_S)
                        if span > 0:
                            evidence.note_search(span)
                            found = p1rx.acquire_control_signal(
                                seg, t0, spec.P1_CS_S, span=span)
                            break
                # WHERE THE CODEWORD IS IN THIS WINDOW, and the lead is NOT
                # subtracted here: the search counts alignments from
                # `ACQUIRE_LEAD_S` in front of `t0` and returns the first that
                # reads, which runs early by about the same amount, so the two
                # cancel to a millisecond. `p1rx.ACQUIRE_LEAD_S` carries the
                # bench and the air measurement that say so.
                at = t0 + found[1] if found is not None else 0.0
                # ...AND THE TURNAROUND IS MEASURED FROM THE CARRIER, in one
                # expression rather than two cases of one. The window's own
                # distance from that carrier is what placed the band -- it is the
                # `since_tx` above -- so the same term belongs in the answer,
                # projected slot or real one.
                candidate_d = (at + (seg_start - search_end) / FS
                               if found is not None else 0.0)
                # AND A CODEWORD IS NOT AN ANSWER UNLESS SOMETHING TRANSMITTED
                # IT. The search matches twelve bits with no energy test behind
                # it; the level this stands at in its own window is the missing
                # half. Taken here rather than inside the search because it is
                # the CANDIDATE that has to clear it -- the search is also how a
                # burst the onset detector threw away gets read.
                if found is not None:
                    level = _candidate_excess(seg, at)
                    if level < ANSWER_CODEWORD_X:
                        evidence.note_discard()
                        print(f"    RX codeword at d = {candidate_d * 1e3:.0f} ms "
                              f"DISCARDED -- {level:.1f}x in-band against the "
                              f"{ANSWER_CODEWORD_X}x a transmission stands at; "
                              f"read out of this window's own noise", flush=True)
                        found = None
                # AND THE LINKED CYCLE ASKS IT AT THE GRID. Same block, because
                # it is the same finding -- nothing reached a decoder this cycle
                # -- and the opposite conclusion: a station we have already
                # identified is answering somewhere in the band it owes us, and
                # only the onset detector's threshold stood between that word and
                # the retry budget. `INLINK_READ_BAND` has the KB5LZK call it was
                # read off. PACTOR-1 ONLY, because that is the waveform this reads
                # and the one it was measured on: an upgraded link answers in
                # PACTOR-3 tones, where `p3acquire` is the instrument and a
                # PACTOR-1 read would be reading the noise between them.
                inlink = None
                if (live is not None and seg is not None and seg.size
                        and host.arq.state == State.CONNECTED
                        and host.protocol is Protocol.PACTOR1):
                    inlink = _grid_answer(sessrx, seg, seg_start,
                                          seg_start - keyed_at)
                if (found is not None and final_listen
                        and connect_tail.candidate(live.pos)):
                    print("  FINAL CALL LISTEN: exact candidate extends listening "
                          f"to {connect_tail.end / FS:.3f}s on the capture clock "
                          f"(fixed ceiling {connect_tail.limit / FS:.3f}s)", flush=True)
                if inlink is not None:
                    print(f"    {inlink}", flush=True)
                elif found is None:
                    print(f"    RX (nothing decoded) -- {rxfront.cs_evidence(seg)}",
                          flush=True)
                elif not evidence.offer(c, candidate_d):
                    # SAY SO AND KEEP CALLING. A candidate is a codeword and not
                    # yet a station; the peer repeats its answer every cycle
                    # until our first data packet decodes, so the corroboration
                    # costs nothing a real link will not pay back.
                    #
                    # ...but it does cost cycles, and the connect budget is four
                    # of them. A candidate holds that budget open exactly as a
                    # bare burst does -- twelve bits at a plausible turnaround is
                    # the stronger evidence of the two -- and without it the rule
                    # can be waiting for a third accept the ARQ will not give it:
                    # `onair-0730-2036` answered on cycles 3, 5 and 7.
                    host.arq.note_peer_heard()
                    # ...and the grid is told, because the hush is decided there
                    # and a decoded answer is exactly what it may not be armed
                    # over. Where to aim is still `evidence`'s to say.
                    answered = (found[0], candidate_d)
                    print(f"    RX candidate CS{found[0] + 1} at d = "
                          f"{candidate_d * 1e3:.0f} ms -- not yet corroborated"
                          f" ({len(evidence.candidates)} so far); still "
                          f"{'listening' if final_listen else 'calling'}",
                          flush=True)
                else:
                    answered = (found[0], candidate_d)
                    # Deliver it the way the anchor path delivers its own decode.
                    # Reporting it and dropping it is what the previous run did,
                    # and six acknowledgements went past while the session sat in
                    # CONNECTING waiting for one.
                    sessrx._on(rxfront.Event(
                        seg_start / FS + at, "cs",
                        f"CS{found[0] + 1}/codeword search (0 bit errors, "
                        f"PACTOR-1, shift "
                        f"{'inverted' if found[2] else 'normal'})",
                        protocol="PACTOR-1", cs=found[0], sense=found[2]),
                        connect_confirmed=final_listen)
                    # The connect answer is the FIRST reading of the peer's phase,
                    # and this cycle's call has already gone out -- so what it
                    # corrects is the next cycle's transmission, which is the
                    # first data packet. A link that takes it here never sends a
                    # data packet in the wrong shift at all.
                    _align_shift(raster, sessrx, tx, slot,
                                 seg_start + int(at * FS))
            if live is not None:
                onsets = [at for at, _ in
                          evidence_of.record(f"rx_{c:02d}", seg, seg_start)]
                _report_answer_band(answer_band, host, raster, seg,
                                    seg_start - keyed_at, d_max_n,
                                    read=bool(onsets) or sessrx.count != before)
            # ...and fold them into the grid HERE, in the same dead time, with the
            # audio they were found in still in hand -- the edge statistic has to
            # be taken on the burst itself.
            #
            # This used to run at the top of the next cycle, on the argument that a
            # phase applied mid-window would move the boundary out from under a
            # window already half collected. That argument belonged to a raster
            # whose anchor moved. This one's does not: everything below corrects
            # the RECEIVE window, and the only thing that can shift the transmit
            # boundary is placing the grid, which happens after a hushed cycle in
            # which we did not transmit and the guard above catches anyway.
            line = raster.update(
                onsets, seg, seg_start, hushed=hush,
                reading=_scheduler_reading(sessrx, host, raster, answered),
                since_tx=seg_start - keyed_at, answered=answered,
                linked=host.arq.state in LINKED)
            onsets = []
            prev, prev_start = seg, seg_start
            if line:
                print(f"    [grid] {line}", flush=True)
            where = raster.answer_position()
            if where:
                print(f"    [entry] {where}", flush=True)
            st = host.arq.state
            linked = linked or st in LINKED
            print(f"  cycle {c}: state {st}", flush=True)
            if st == State.CONNECTED and args.hold:
                # Stay up and keep listening. The whole question after an
                # acknowledgement is what the peer does NEXT, and disconnecting
                # the moment we reach CONNECTED throws exactly that away.
                hold_description = ("without an experiment/hold cutoff"
                    if getattr(args, "p3_timing_unbounded", False)
                    else f"{args.hold} idle cycles")
                print(f"** CONNECTED to {host.arq.dxcall or dx} ** holding "
                      f"{hold_description}, listening", flush=True)
                if args.message:
                    # `--message` used to be queued only on the branch below --
                    # connect, send, disconnect -- so a held session was the one
                    # arrangement that could carry a message across a real link
                    # and the one that never had a message to carry. The link
                    # layer does not transmit it here: it queues, and the hold
                    # loop's ticks put it on the air a packet a cycle.
                    host.arq.on_host_data(args.message.encode())
                if args.over:
                    # THE PEER CANNOT SEND WHILE WE HOLD THE CHANNEL. PACTOR ARQ
                    # gives the caller the transmit direction and keeps it there
                    # until a changeover hands it back, so a session that only
                    # ever sends is a session the gateway can only acknowledge --
                    # which is every session this station has run. Requesting the
                    # changeover as the buffer drains is what lets a gateway
                    # answer with data rather than with a codeword.
                    host.arq.on_host_over()
                sessrx.tag = "HOLD RX"
                whole = np.zeros(0, np.float32)
                budget, goodbye = _HoldBudget(args.hold,
                    unbounded=getattr(args, "p3_timing_unbounded", False)), None
                # The experiment adds a distinct bounded phase after the QRT
                # ACK. The outer loop must allow it to run; VE3KPG answered
                # the QRT near the end of the original four-cycle allowance.
                qrt_cycles = QRT_CYCLES + (GOODBYE_CYCLES if host.p3_qrt_confirm else 0)
                # Where the link's own codewords start in the log, so the ending
                # can count what the peer said once we were UP rather than what
                # the session decoded in total.
                held_from = len(sessrx.cs_log)
                # Which budget closed the hold, filled in WHERE IT CLOSES rather
                # than written out here in advance: against a peer that keeps
                # sending bytes the idle timeout never expires at all -- the
                # ceiling holds the deadline and the loop stops on that -- so a
                # sentence fixed in front of the loop names a bound that had
                # nothing to do with the ending.
                stopped = None
                listened_at = 0
                previous_cycle_slot = tx.slot
                for h in itertools.count(1):
                    if tx.p4_probe is not None and tx.p4_probe.requested:
                        ended = _run_p4_probe(tx, live, raster)
                        break
                    trial_end = _timing_trial_close(
                        tx, host, live.sample_now() if live is not None else 0)
                    if trial_end is not None and goodbye is None:
                        stopped, goodbye = trial_end, h
                    terminal_before = (host.arq._disconnect_ticks is not None,
                                       host.arq._rx_close_pending, host.arq.said_goodbye,
                                       host.arq.terminal_confirm_pending)
                    if goodbye is None and host.arq.state not in LINKED:
                        # THE HOLD IS A LINK'S, NOT A BUDGET'S. The ARQ gives up
                        # on a peer that has stopped answering -- retries spent,
                        # or a disconnect from the far end -- and the loop's only
                        # exit test was `h > args.hold`, so it went on spinning
                        # the rest of its budget with the link already gone:
                        # ninety cycles is close to two minutes, and on
                        # 2026-08-06 the operator heard every one of them as
                        # "long periods of silence, on the order of minutes".
                        #
                        # There is no goodbye to send FROM HERE.
                        # `on_host_disconnect` below is for a link that is still
                        # up; this one is not, and a QRT queued on a DISCONNECTED
                        # state machine goes nowhere while the raster keeps
                        # calling. The goodbye on this path belongs to the FSM,
                        # which owns the budget that ended the link and is still
                        # the ISS-or-IRS when it runs out -- `arq._give_up`.
                        sent = ("the link was signed off with a QRT"
                                if host.arq.said_goodbye else
                                "goodbye could not be placed within its deadline"
                                if host.arq.goodbye_unplaceable else
                                "no goodbye was sent")
                        ended = (f"the link went down in hold cycle {h} of "
                                 f"{budget.deadline} ({host.arq.state}); "
                                 f"{_held_answers(sessrx.cs_log[held_from:])}; "
                                 f"{sent}")
                        # The rest of the budget is spent one of two ways, and
                        # never on the air either way: stop, or listen. See
                        # `_observe_out_the_hold` for what the second is for.
                        watch = args.observe_after_link_down and live is not None
                        print(f"** LINK DOWN ** {ended} -- "
                              + ("listening out the rest of the hold, "
                                 "transmitting nothing" if watch else
                                 "stopping rather than holding a channel "
                                 "nothing is on; --observe-after-link-down "
                                 "would spend the rest of it listening"),
                              flush=True)
                        if watch:
                            watched = _observe_out_the_hold(
                                live, raster, tx, sessrx, host, evidence_of,
                                budget, h, slot)
                            cycles_run += watched
                            ended += (
                                f"; the run then observed {watched} further "
                                f"cycles without transmitting"
                                + (f", to hold cycle {h + watched - 1}"
                                   if watched else ""))
                        break
                    if goodbye is not None:
                        # BOTH ENDINGS ARE A GOODBYE'S, and neither is reachable
                        # without one having gone out. Only 7 PACTOR sessions in
                        # the whole record ever transmitted a QRT, so a verdict
                        # about whether the peer answered one is about nothing at
                        # all on every other session that reaches this loop.
                        #
                        # AND THE GOODBYE'S FATE IS NOT THE SESSION. Which of the
                        # two competing clocks stopped it, what the peer said
                        # while the link was up, and what became of the QRT are
                        # three different questions, and the line answered only
                        # the third -- so a gateway heard on every cycle and a
                        # channel nobody was on printed the same sentence. See
                        # `_held_answers`.
                        heard = _held_answers(sessrx.cs_log[held_from:])
                        if host.arq.state in (State.DISCONNECTED,
                                              State.LISTENING):
                            # A closed link is not an answered one. The state
                            # machine ends its own teardown on `GOODBYE_CYCLES`
                            # whether or not anything came back, so reading the
                            # close as the acknowledgement credits the peer with
                            # a codeword it may never have sent.
                            was = ("acknowledged" if host.arq.goodbye_acked else
                                   f"unanswered, and the link closed on its own "
                                   f"{GOODBYE_CYCLES}-cycle teardown")
                            ended = (f"{stopped} in cycle {goodbye}; {heard}; "
                                     f"the goodbye was {was}")
                            break
                        if h > goodbye + qrt_cycles:
                            ended = (f"{stopped} in cycle {goodbye}; {heard}; "
                                     f"the goodbye went unacknowledged for "
                                     f"{qrt_cycles} cycles")
                            break
                    # Listen for the REST of the cycle, not a whole one, and take
                    # it in slices so each turnaround's answer is decoded as it
                    # arrives -- on a held link none of them is repeated, so an
                    # event noticed a cycle late is an event missed. Our own
                    # transmission occupies most of the 1.25 s raster (a packet is
                    # 0.96 s, and PTT settles before it), so listening a whole
                    # cycle on top of that made our turn 2.3 s against the peer's
                    # 1.25: we keyed while it answered and listened while it
                    # waited. The floor is the control-signal window, because a
                    # window shorter than the thing it is meant to catch catches
                    # nothing.
                    #
                    # "PACTOR arbeitet als bitsynchrones System mit einem festen
                    # Zeitraster" -- the peer is not timing each turn against our
                    # last one, it is running a clock, and reply onset across a
                    # hold used to WALK rather than scatter (620 -> 560 -> 500 ->
                    # ... -> 265 ms in one run) because ours ran slow. It is the
                    # same grid as the setup phase, one slot further on.
                    cycles_run += 1
                    before = sessrx.count
                    receive_opportunity = host.arq.begin_receive_opportunity()
                    sessrx._p3_stage_ms = {}
                    sessrx._p3_long_crc_reply = None
                    # What has crossed the link so far, against which this
                    # cycle's own reading is compared below.
                    carried = _Link.of(host)
                    # Whether the last cycle put a frame through the FSM, read
                    # before the one-shot resets: it decides which side of the
                    # key this cycle's frame scan can afford to run on.
                    flowing = sessrx.frame_seen or host.arq.cycle_command_emitted
                    sessrx.new_cycle()
                    # Connection cancels a hush in flight. A linked session
                    # must keep its cadence: six cycles of silence cannot rescue
                    # its 1.25 s raster and can drop a station still answering.
                    hush = live is not None and raster.hush_left > 0
                    # ...and the other way a cycle goes unkeyed, which is this
                    # station's own decision rather than the grid's: one cycle in
                    # N given to the receiver. `hold_15` of onair-0904-1659 was
                    # the single most informative window of that arm -- 1.256 s
                    # with the receiver open where every other window was a
                    # 0.24 s peephole between our carriers -- and it existed only
                    # because a teardown decision happened to cost a cycle. The
                    # FSM still runs: the burst is refused at the seam, which
                    # spends no retry and re-places the packet next cycle.
                    #
                    # ...and the other reason a cycle is given to the receiver,
                    # which is not a schedule but the link asking for it: after
                    # a changeover the peer has not answered, the 365 ms a keyed
                    # cycle leaves cannot hold its 815 ms packet, so re-keying
                    # the changeover is the deafness that produced the silence.
                    # See `arq.breakin_listen_due`.
                    cede = (live is not None and not hush
                            and host.arq.breakin_listen_due)
                    listen = cede or (live is not None and not hush
                                      and _listen_due(args.listen_every, tx.n,
                                                      listened_at))
                    if listen and not cede:
                        listened_at = tx.n
                    if cede:
                        print(f"    [grid] {LISTENING_FOR_THE_CEDE} -- the "
                              f"changeover went unanswered, and a keyed cycle "
                              f"leaves less window than any reader needs "
                              f"({MIN_DECODE_S * 1e3:.0f} ms) to take the "
                              f"peer's answer. This cycle keys nothing so its "
                              f"packet can be read whole; the changeover goes "
                              f"out again on the next one.", flush=True)
                    tx.listening = listen
                    quiet = hush or listen
                    # Read before the aim: a changeover packet is placed against
                    # the peer's transmission rather than on our boundary, and
                    # everything in front of the key is measured from the
                    # earlier of the two. See `RadioTx.key_instant`.
                    tx.breakin_due = host.breakin_due and not quiet
                    # ...and before the aim for a second reason: on the long
                    # cycle the next boundary is three slots on, the window in
                    # front of it is a whole 3.75 s turn, and the peer's packet
                    # inside it is 3.37 s. Read here rather than at the key,
                    # which is where the protocol is recorded: the length
                    # decides the window and the slot, and both are computed
                    # below. See `_MasterGrid.regear`.
                    # A post-key scan can confirm P3 after entry fallback.
                    # Queue its answer even when this iteration began in P1;
                    # the next pre-key pass applies protocol and role first.
                    tx.defer_p3_cs = live is not None
                    probing_cycle = (host.protocol == Protocol.PACTOR3
                                     and host.arq.role == IRS
                                     and host.arq.cycle_command_emitted)
                    slot, geared = _regear_next_slot(
                        raster, slot, False if probing_cycle else host.arq.cycle_long)
                    if geared:
                        print(f"    [grid] {geared}", flush=True)
                    boundary = tx.aim(raster, slot)
                    key_at = boundary
                    if live is not None:
                        # The setup loop's rule, and a held link needs it
                        # just as much: a nudge is small but the slot it lands
                        # in may already be all but spent. Measured on the
                        # decoded window, not on the early-return slice; see
                        # `_keyable_slot`.
                        # With the changeover's lead in hand, because it is the
                        # same for every slot: asked without it, a slot whose
                        # boundary is placeable but whose key instant is not
                        # leaves the burst to `_tx`'s backstop and a whole cycle.
                        # THE COMB FIRST OF ALL. Every number below is measured
                        # from `boundary`, and a PACTOR-3 IRS reply boundary is
                        # the answered packet's answer slot rather than whatever
                        # the last cycle left. See `_p3_place_reply`.
                        _p3_place_reply(raster, tx, slot)
                        early = raster.boundary(slot) - tx.key_instant(raster, slot)
                        slot = _keyable_slot(live, raster, slot, settle_n + early)
                        boundary = tx.aim(raster, slot)
                        key_at = tx.key_instant(raster, slot)
                        if whole.size:
                            # THE STALLED LINK'S FRAME SCAN, one cycle late and
                            # free. The last window went unscanned before its
                            # key (see below); swept here, the 20.3 ms an empty
                            # channel costs land in front of a whole cycle of
                            # listening instead of inside the keying settle. A
                            # frame found now answers from `on_rx_event` and
                            # keys ON the boundary just aimed -- the grid path
                            # a mid-window decode has always taken -- and puts
                            # the scan back in front of the key next cycle.
                            #
                            # `or`, because this sweep now also runs on a cycle
                            # that was already flowing: what it finds can add to
                            # that fact and must not replace it.
                            found = _scan_previous_window(
                                sessrx, whole, prev_start,
                                sending=raster.sending, flowing=flowing)
                            flowing = flowing or found
                        win = (boundary - live.pos if quiet else
                               key_at - settle_n - live.holdback
                               - round(PREKEY_RESERVE_S * FS) - live.pos)
                        n0 = live.samples
                        # The same listener the setup phase uses. Its early break
                        # is gated on CONNECTING, so on a held link it simply
                        # collects the window -- one code path for both phases
                        # rather than two that can disagree about the grid.
                        #
                        # HELD, NOT FED, WHILE THE PEER HOLDS THE LINK. A sending
                        # station's window is 0.24 s and rides under `RollingRx`'s
                        # half-second floor, so feeding it is free. A receiving
                        # station's window is the peer's whole 0.96 s packet, and
                        # feeding it engages the rolling regime inside the wall
                        # clock that decides whether our acknowledgement can key:
                        # 140 ms measured on one real receive window of
                        # onair-0803-225309 on this machine -- more than twice
                        # the budget between the peer's packet ending and our
                        # key, before the station Pi is asked. That is where the
                        # 2026-08-03 reversal lost its cadence: acknowledgements
                        # on slots 15, 16, 19, 23, 26, 28, 30, 32, against a peer
                        # that keys every slot and hears an IRS that misses one
                        # as a station going deaf. The frame the window holds is
                        # read by `deep_scan` below for single milliseconds, and
                        # the flush still re-decodes the tail behind our carrier.
                        whole = _listen_until_answer(
                            live, win, host, sessrx,
                            FEED_MAX_SLOTS * raster.slot_n if raster.sending
                            else 0)
                        if not args.replay and win > 0:
                            # A zero-length window proves nothing about the
                            # stream -- and the scan above may have keyed,
                            # which spends the window before it opens.
                            _assert_capturing(whole, live, n0)
                    else:
                        whole = np.zeros(0, np.float32)
                        time.sleep(args.cycle)
                    if live is not None:
                        # The setup loop's order, for its reasons: the origin
                        # first, then the frame scan out of channel time, and the
                        # bridge to the key last of all.
                        seg_start = live.pos - whole.size
                        whole, seg_start, bridge = _collect(
                            live, sessrx, whole, seg_start,
                            sessrx.control_bridge_until(
                                key_at - max(raster.d, settle_n),
                                key_at - _prekey_lead(live, settle_n),
                                max(live.pos, int(live.sample_now())), raster))
                        if whole.size and raster.sending:
                            _scan_frame(sessrx, whole, seg_start)
                        until = sessrx.control_collect_until(
                            key_at - _prekey_lead(live, settle_n),
                            max(live.pos, int(live.sample_now())), raster)
                        if (host.protocol == Protocol.PACTOR3 and tx.defer_p3_cs
                                and not raster.sending
                                and (flowing or sessrx.wideband_prekey_active())):
                            until = _p3_frame_ready(
                                sessrx,
                                _p3_decode_deadline(live, key_at, settle_n),
                                live.holdback)
                        whole, seg_start, tailed = _collect(
                            live, sessrx, whole, seg_start, until)
                        bridge += tailed
                        # READ HERE AND NOT AFTER THE KEY. This window follows the
                        # PREVIOUS transmission, and by the time the cycle's own
                        # evidence is folded in below `tx.tx_end` has moved on to
                        # this cycle's carrier -- which would put the answer band
                        # a whole cycle away from where the answer is.
                        held_at = tx.tx_end if tx.tx_end is not None else 0
                        # The window is closed; the fallback the ACK-latency line
                        # falls back to when the peer was never heard measures from
                        # here. The setup loop sets it every cycle and this one
                        # never did, so on a held link that line was reporting the
                        # age of the connect phase.
                        tx.reply_at = time.time()
                        if (whole.size and not raster.sending
                                and (flowing or sessrx.wideband_prekey_active())):
                            # AFTER the last bridge, because the involution keeps
                            # each side's own turnaround: the peer's packet ends
                            # about `d` before OUR boundary, which is the exact
                            # sample the mid-cycle buffer stops at. Scanned there,
                            # the frame's last bits are never in the audio and it
                            # cannot decode -- see `deep_scan` for the session
                            # that wedged on it.
                            #
                            # ONLY WHILE THE SCAN IS EARNING SAME-CYCLE
                            # ACKNOWLEDGEMENTS. Between here and the key there
                            # are 8 ms past `key_notice`, which holds the
                            # 1.8 ms a frame that is there costs and not the
                            # 20.3 ms of a sweep that finds nothing -- the
                            # price every IRS cycle of the 2026-08-13 session
                            # paid, 11-15 ms late at its key with four slots
                            # lost outright, against a gateway whose packets
                            # were not decoding at all. So a cycle that came up
                            # empty moves the NEXT cycle's scan to the top of
                            # the cycle, where the sweep is free; a frame found
                            # there still reaches the FSM before that cycle's
                            # key, one cycle late -- the flush's own bargain.
                            _scan_frame(sessrx, whole, seg_start,
                                        tracked_only=tx.defer_p3_cs)
                        if host.protocol == Protocol.PACTOR3 and host.arq.role == IRS:
                            slot, whole, seg_start = _p3_transition_window(
                                live, raster, tx, host, sessrx, slot, whole,
                                seg_start, settle_n)
                            boundary = tx.boundary
                            key_at = tx.key_instant(raster, slot)
                    # Before the key, for the setup loop's reason: a station that
                    # has just yielded owes the peer the rest of its packet, and
                    # reads it in the 840 ms the rotation has just bought.
                    control_started = time.perf_counter()
                    heard, at = (sessrx.control_signal_in(whole, seg_start, raster)
                                 if live is not None
                                 and not _p3_current_long_crc(sessrx, slot)
                                 and not _p3_current_packet_crc(sessrx, raster, slot)
                                 else (None, None))
                    sessrx._p3_stage_ms["control"] = (time.perf_counter() - control_started) * 1e3
                    if heard is not None:
                        _align_shift(raster, sessrx, tx, slot, at)
                    whole = _receive_changeover(
                        live, raster, tx, host, sessrx, slot, whole, seg_start,
                        prev, prev_start, settle_n, reserve_s=PREKEY_RESERVE_S)
                    boundary = tx.boundary
                    if _reverse_before_key(raster, host, tx, slot):
                        boundary = tx.boundary
                    if live is not None and not quiet:
                        # The setup loop's rule, and a held link is where it was
                        # caught: see `_regrid`.
                        slot, whole, seg_start = _regrid(
                            live, raster, tx, host, sessrx, slot, whole,
                            seg_start, settle_n)
                        boundary, key_at = tx.boundary, tx.key_instant(raster, slot)
                        if _reverse_before_key(raster, host, tx, slot):
                            boundary, key_at = tx.boundary, tx.key_instant(raster, slot)
                    if live is not None:
                        # The setup loop's reference, for its reason: a hush
                        # reads to the boundary, a keyed cycle to the key -- and
                        # the changeover's key is not the boundary, which is what
                        # made this line read healthy on the one cycle it was
                        # 22 ms of settle short.
                        off = (live.sample_now()
                               - (boundary if quiet else key_at - settle_n)) / FS
                        print(f"    [grid] hold {h} slot {slot}, CS due @ "
                              f"{raster.rx_due(slot)}; captured "
                              f"{whole.size / FS:.3f} s, bridged {bridge:.0f} ms, "
                              f"off-grid {off * 1e3:+.1f} ms"
                              f"{' -- HUSHED, not keying' if hush else ''}"
                              f"{' -- LISTENING CYCLE, not keying' if listen else ''}",
                              flush=True)
                    # ONE ARQ CYCLE, which is `ticks` grid slots. The comb is
                    # the same at either length -- the reference's long packets
                    # sit 3.750 s apart on the phase comb its short ones set --
                    # so stepping by one while long would key the next burst
                    # inside the packet it is supposed to follow.
                    elapsed_ticks = max(1, slot - previous_cycle_slot)
                    previous_cycle_slot = slot
                    slot = raster.next_slot(slot)
                    # AFTER THE LISTENING AND BEFORE THE KEY. A decision taken at
                    # the top of the cycle cannot have the last thing the peer
                    # said in evidence, and that is the margin the whole question
                    # turns on: K4MSU's greeting packet reached the FSM through
                    # this cycle's frame scan and its bytes were on the host a
                    # few hundred milliseconds before the key that answered it.
                    budget.spend(elapsed_ticks)
                    budget.cycle(h, carried, _Link.of(host))
                    trial_end = _timing_trial_close(
                        tx, host, live.sample_now() if live is not None else 0)
                    if trial_end is not None and goodbye is None:
                        stopped, goodbye = trial_end, h
                    if h > budget.deadline and goodbye is None:
                        # The QRT still has to be TRANSMITTED. Falling out of the
                        # loop straight to PTT-off queued it on a state machine
                        # that never got another tick, so no session ever said
                        # goodbye -- and a peer holds a channel it believes is
                        # up: on 2026-08-03 WS8EOC spent 34 cycles asking a
                        # vanished station for its next packet and then signed
                        # off in CW. An IRS needs these cycles most of all,
                        # because QRT rides a packet and it must break in to
                        # send one.
                        # Unless something has already said why: `close` leaves
                        # its own reason where the mail exchange is what ended
                        # the hold, and the budget's words are for a hold that
                        # nothing else stopped.
                        if stopped is None:
                            stopped = budget.ran_out
                        host.arq.on_host_disconnect()
                        goodbye = h
                    if not hush:
                        # Age the elapsed wall slots, but execute only this
                        # present opportunity. Replaying past ticks can overwrite
                        # a fresh deferred ACK or transmit retries without RX.
                        # A disconnect requested just above did not exist during
                        # the listening interval and must not age retroactively.
                        terminal_now = (host.arq._disconnect_ticks is not None,
                                        host.arq._rx_close_pending, host.arq.said_goodbye,
                                        host.arq.terminal_confirm_pending)
                        closing_now = any(now and not before for before, now
                                          in zip(terminal_before, terminal_now))
                        host.tick(elapsed_ticks=1 if closing_now else elapsed_ticks,
                                  cycle_ticks=raster.ticks)
                        tx.emit_pending_cs()
                    if host.protocol == Protocol.PACTOR3:
                        timings = ", ".join(f"{name}={ms:.2f}ms" for name, ms
                                            in sessrx._p3_stage_ms.items())
                        callbacks = getattr(tx, "_callback_at_admit", ())
                        delivery = [(n, frames, round((wall - callbacks[0][2]) * 1e3, 3),
                                     round((adc - callbacks[0][3]) * 1e3, 3))
                                    for n, frames, wall, adc in callbacks]
                        print(f"    [p3 timing] hold {h}, RX end {seg_start + whole.size}; "
                              f"{timings}; callbacks(sample,frames,wall_ms,adc_ms)={delivery}",
                              flush=True)
                    if mail is not None and not hush:
                        _mail_app_turns(host, mail, wait_greeting=getattr(
                            args, "mail_wait_greeting", False))
                        # A session writes the body and its close-out in the
                        # one `feed` and sets `done` on the same call, so the
                        # flag says the exchange is over and not that it is on
                        # the air. `test_b2f_loopback_send_ws8eoc` scene 11
                        # leaves the whole message in `_txbuf` behind it.
                        if (mail.done and goodbye is None
                                and (mail.session.failure or not host._txbuf)):
                            # The exchange is over: spend none of the
                            # remaining hold. Pulling the deadline in hands the
                            # QRT to the path above, which is the one place a
                            # goodbye is actually transmitted.
                            failure = mail.session.failure
                            verdict = (f"exchange failed: {failure}" if failure
                                       else "exchange complete")
                            print(f"    [mail] {verdict} -- closing", flush=True)
                            budget.close(h)
                            stopped = (f"the mail exchange failed: {failure}" if failure
                                       else "the mail exchange completed")

                    line = _grid_reversal(raster, host)
                    if line:
                        print(f"    [grid] {line}", flush=True)
                    _drain_host_log(host)
                    # In the dead time behind our own carrier, for the setup
                    # loop's reasons: see `PREKEY_RESERVE_S` and
                    # `_SessionRx.upgrade_scan`.
                    sessrx.flush()
                    line = _grid_reversal(raster, host)
                    if line:
                        print(f"    [grid] {line}", flush=True)
                    if whole.size:
                        _scan_frame(sessrx, whole, seg_start, upgrade=True)
                    line = _grid_reversal(raster, host)
                    if line:
                        print(f"    [grid] {line}", flush=True)
                    if live is not None:
                        bursts = evidence_of.record(f"hold_{h:02d}", whole,
                                                    seg_start)
                        onsets = [at for at, _ in bursts]
                        _read_codeword_at_bursts(sessrx, whole, seg_start,
                                                 onsets)
                        # AND NOW THE GRID, which is the reader of last resort and
                        # the one a held link was always the case for. Everything
                        # above starts from an onset, and `_p1_runs` wants 4.0x
                        # before it offers one: K0NTS answered 40 m at 1.9-5.3x
                        # from a turnaround 40 ms off the nominal answer slot, so
                        # neither the detector nor `cs_anchored` could reach it
                        # and the session spent its retry budget on a station
                        # answering every cycle. `INLINK_READ_BAND` carries the
                        # trade this is made on.
                        #
                        # ONLY WHERE THE CYCLE OWES US A CODEWORD. `held_at` is
                        # OUR carrier dropping, and the band hangs off it: a
                        # cycle we did not key measures from a carrier a cycle or
                        # more back, and a cycle we spent as the IRS is owed the
                        # peer's packet rather than its answer. Both would put
                        # the search somewhere the peer is not.
                        #
                        # PACTOR-1 ONLY, because that is the waveform this reads
                        # and the one it was measured on: an upgraded link
                        # answers in PACTOR-3 tones, where `p3acquire` is the
                        # instrument and a PACTOR-1 read would be reading the
                        # noise between them.
                        if (whole.size and raster.sending and not quiet
                                and host.protocol is Protocol.PACTOR1
                                and not sessrx.words_at):
                            line = _grid_answer(sessrx, whole, seg_start,
                                                seg_start - held_at)
                            if line:
                                print(f"    {line}", flush=True)
                        # This last decoder can find a changeover missed above.
                        # Correct the grid before forecasting or folding onsets;
                        # next cycle's pre-key check is too late for its listen.
                        line = _grid_reversal(raster, host)
                        if line:
                            print(f"    [grid] {line}", flush=True)
                        # AFTER the codeword read and never in front of it: this
                        # asks only what the reader left, and a window it took
                        # has nothing here to report.
                        _report_answer_band(
                            answer_band, host, raster, whole,
                            seg_start - held_at, d_max_n,
                            read=bool(onsets) or sessrx.count != before)
                        # ...and only now, because the forecast rests on a
                        # codeword and the onset read is the last path that can
                        # deliver one.
                        _forecast_next_key(sessrx, tx, raster, seg_start)
                    if sessrx.count == before:
                        print(f"    HOLD {h} RX (quiet)", flush=True)
                    # Same fold as the setup loop, in the same place and for the
                    # same reason: the receive window is corrected where the audio
                    # is, and the transmit grid keeps running whatever it says.
                    line = raster.update(onsets, whole, seg_start, hushed=quiet,
                                         reading=_scheduler_reading(
                                             sessrx, host, raster),
                                         linked=host.arq.state in LINKED)
                    if raster.nearest_gap_n is not None:
                        # In-band on the same two bounds `_acquire` accepts a
                        # turnaround inside, because that is the whole of what
                        # "where an answer is due" means -- and the ARQ, which
                        # cannot see the channel, spends its link-dead budget on
                        # the answer.
                        host.arq.note_burst(
                            raster.nearest_gap_n / FS * 1e3,
                            at_anchor=(round(D_MIN_S * FS)
                                       <= raster.nearest_gap_n
                                       <= raster.d_max_n))
                    # AFTER the fold, because a cycle with no codeword where one
                    # is due drops the onset it was holding and this is the
                    # reading that replaces it: while we hold the link there is
                    # no other, and the changeover's placement stands on it.
                    reread = raster.note_peer_bursts(bursts)
                    if reread:
                        print(f"    [grid] {reread}", flush=True)
                        # ...and the ARQ, which cannot see the channel and is
                        # about to spend a refusal on the silence this is the
                        # refutation of. See `arq.note_peer_raster_burst`.
                        host.arq.note_peer_raster_burst()
                    host.arq.finish_receive_opportunity(receive_opportunity)
                    onsets, bursts = [], []
                    prev, prev_start = whole, seg_start
                    if line:
                        print(f"    [grid] {line}", flush=True)
                    where = raster.answer_position()
                    if where:
                        print(f"    [entry] {where}", flush=True)
                break
            if st == State.CONNECTED:
                print(f"** CONNECTED to {host.arq.dxcall or dx} **")
                if args.message:
                    host.arq.on_host_data(args.message.encode())
                    host.tick()
                host.arq.on_host_disconnect()
                break
            if st == State.DISCONNECTED:
                ended = "the ARQ gave up calling"
                print("link down."); break
            if connect_tail is not None and live.pos >= connect_tail.end:
                ended = "the final-call receive-only interval ended without an accepted answer"
                print(f"  {ended}", flush=True)
                break
            if live is None and reply is None:
                print("  (dry run, no --reply-wav: nothing to receive -- stopping)"); break
            if args.replay and live.pos >= live.audio.size:
                ended = "the replay ran out of audio"
                print("  (replay exhausted)"); break
    except KeyboardInterrupt:
        # Caught so the summary is the last thing on the screen. Uncaught, the
        # traceback prints AFTER the `finally` and scrolls the verdict away --
        # and an operator who stops a run early wants the verdict most. The rig
        # is still unkeyed below; nothing about the shutdown path changes.
        ended = "the operator interrupted it"
        print("\ninterrupted.", flush=True)
    except PttError as exc:
        # The mid-session twin of the arm gate: the same silence begins with a
        # device unplugged, or a rigctl dying under a session already running,
        # and no check at start can see either. Caught rather than propagated
        # so the summary below is the account -- `keyed` holds only confirmed
        # bursts, this string is why there are no more of them, and the exit
        # code says the session is not a success to any script reading it.
        ended = f"KEYING FAILED -- {exc}"
        code = 1
        print(f"\n!! KEYING FAILED: {exc}", flush=True)
    else:
        # Only ordinary completion may start another transmission. SIGTERM,
        # Ctrl-C and failures go straight to unkeying and closing the devices.
        if not linked:
            tx.identify(args.mycall)
    finally:
        try:
            if live is not None:
                # The last term in the error budget, reported rather than assumed.
                # A sample-locked raster leaves only the mismatch between this codec
                # and the far end's TNC; at 100 ppm each way that is ~250 us per
                # cycle, and it sets how long a link holds before the raster needs
                # re-syncing. It is also the one number that says whether the grid
                # above is measuring seconds or measuring something else.
                print(rates.host_report())
                print(live.clock_report())
                live.close()
                print(live.stream_report())
        finally:
            if rig is not None:
                rig.stop()
                print("PTT off.")
                if power is not None:
                    power.restore()
        # AFTER the key is down and not one line before it. The captures are
        # analysed tomorrow and can wait the few hundred milliseconds; a
        # transmitter still asserted cannot wait for anything.
        if tx.timing_trial is not None:
            if tx.timing_trial.reason is None:
                tx.timing_trial.finish(ended or "session ended before trial completed")
            tx.timing_trial.write(outdir)
            print(f"P3 timing trial results: {outdir / tx.timing_trial.filename}",
                  flush=True)
        captures = _drain_captures()
        if tx.p4_probe is not None:
            if ("pending" in tx.p4_probe.reason or tx.p4_probe.emitting
                    or tx.p4_probe.reason == "probe interrupted"):
                tx.p4_probe.reason = ended or "probe interrupted"
            result = tx.p4_probe.result()
            result["mode"] = ("live" if args.transmit and rig is not None else
                              "replay" if args.replay else "dry-run")
            (outdir / "p4-probe.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"P4 probe: {len(result['emissions'])}/{result['burst_limit']} "
                  f"chirps; {result['reason']}. Receive windows retained for offline analysis.",
                  flush=True)
        if raster is not None and tx.p4_probe is None:
            print(f"    [entry] {raster.entry_verdict()}", flush=True)
        # Last, so it survives a scrolling terminal.
        if tx.p4_probe is None:
            _summary(sessrx.cs_log, tx.slots_used, cycles_run, tx.keyed, args.cycle,
                     tx.seq_sent, evidence, ended, captures,
                     leads=tx.leads, settle=tx.settle, breakin_at=tx.breakin_at)
        else:
            print(f"session ended: {ended}; {len(sessrx.cs_log)} controls decoded "
                  "before probe takeover", flush=True)
            print(captures, flush=True)
            print("verdict: P4 probe capture only; acceptance not evaluated live", flush=True)
        if tx.timing_trial is not None:
            # Console selects the last verdict. A successful P1 connection is
            # not the outcome of an experiment about inbound P3 progression.
            print('verdict: ' + tx.timing_trial.verdict(), flush=True)
        if mail is not None:
            from ..winlink import summarize, write_inbox
            print(summarize(mail.session), flush=True)
            for path in write_inbox(mail.session, args.mail_out):
                print(f"mail: wrote {path}", flush=True)
    return code


def _entry_ladder(spec_str: str) -> tuple[str, ...]:
    """A ladder off the command line, or the refusal `arq` states for it.

    The grounded rungs are refused here rather than warned about, because the
    warning was a docstring and the docstring shipped the rung on by default.
    """
    try:
        return check_entry_ladder(r.strip() for r in spec_str.split(",") if r.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def main() -> int:
    p = argparse.ArgumentParser(description="shrike operational PACTOR modem (RX in the loop)")
    p.add_argument("--mycall", default="N0CALL")
    p.add_argument("--dxcall", help="the station to call -- no default, this "
                                    "callsign goes on the air")
    p.add_argument("--band", default="40m")
    p.add_argument("--center", type=int, help="channel centre Hz (dial = centre-1500)")
    p.add_argument("--dial", type=int, help="explicit dial Hz (overrides --center/table)")
    p.add_argument("--transmit", action="store_true", help="ARM the rig (licensed op only)")
    p.add_argument("--rig", choices=list(RIGS), default="ft891")
    # No default on purpose: a shipped device path is right on one machine and
    # silently wrong at the next rig, and the placeholder that stood here was
    # what the 2026-08-10 arm-gate incident let through. Required under
    # --transmit only -- dry runs and --replay never open a port.
    p.add_argument("--serial",
                   help="CAT serial port, required with --transmit -- "
                        "`ls /dev/cu.*` to find yours")
    p.add_argument("--baud", type=int, default=0)
    p.add_argument("--ptt-type", choices=["RTS", "DTR", "RIG"],
                   help="how to key: a serial line, or a CAT command (default: "
                        "the rig's own, RTS on the FT-891)")
    p.add_argument("--ptt-port", help="the device whose RTS/DTR keys the rig "
                                      "(default: derived from --serial)")
    p.add_argument("--audio-out")
    p.add_argument("--audio-in")
    p.add_argument("--reply-wav", help="dry run: feed this WAV as the recorded reply")
    p.add_argument("--message", help="send this text once connected")
    p.add_argument("--mail-send", action="append", metavar="FILE",
                   help="Winlink mail: a rendered .b2f message, or a body file "
                        "with --mail-to (repeatable). Attaches a B2F session "
                        "to the link in place of raw --message text")
    p.add_argument("--mail-fetch", action="store_true",
                   help="Winlink mail: collect whatever the gateway holds")
    p.add_argument("--mail-wait-greeting", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="defer automatic application break-in until the mail "
                        "client has a reply to the gateway greeting; preserve "
                        "queued bytes and normal teardown")
    p.add_argument("--mail-to", default="",
                   help="recipient for a --mail-send body file")
    p.add_argument("--mail-subject", default="")
    p.add_argument("--mail-password", default="",
                   help=f"secure-login answer to the gateway's ;PQ: challenge. "
                        f"Spelled out here it is in this process's argv, which "
                        f"every `ps` can read: prefer --mail-password-file or "
                        f"${config.PASSWORD_ENV}")
    p.add_argument("--mail-password-file", default="",
                   help="read the ;PQ: answer from this file instead, so it "
                        "never reaches argv (default: "
                        f"${config.PASSWORD_ENV})")
    p.add_argument("--mail-sid", default="",
                   help="the client type this station announces: the name and "
                        "version half of the B2F SID, no brackets and no "
                        "capability letters. Defaults to [station] client_sid "
                        f"from the station file named by {config.STATION_ENV}, "
                        f"else {CLIENT_SID}")
    p.add_argument("--mail-out", default="logs/mail",
                   help="where received messages are written")
    p.add_argument("--over", action="store_true",
                   help="hand the channel over once the message drains, so the "
                        "peer can send data instead of only acknowledging")
    p.add_argument("--preflight", action="store_true",
                   help="check the audio path and, with --transmit, the T/R path "
                        "against this rig; keys with NO modulation, so no RF")
    # The escape hatch for a codec that will not run at 128 frames. Raising it
    # costs keyed time and nothing else -- measured 2026-07-28, the excess over the
    # audio is 4.5 ms at 128, 8.5 at 256, 18 at 512, against 250 ms of clear air --
    # so a dongle that xruns is a --audio-block 256 away from working rather than a
    # session lost to an edit at the radio.
    p.add_argument("--audio-block", type=int, default=128, metavar="FRAMES",
                   help="duplex stream blocksize (default 128; raise to 256 or 512 "
                        "if --preflight reports capture loss)")
    p.add_argument("--tune", type=float, default=0.0, metavar="SECONDS",
                   help="key a steady low-power carrier for SECONDS so the ATU "
                        "can match a new band, then exit without calling anyone")
    p.add_argument("--listen-only", type=float, default=0.0, metavar="SECONDS",
                   help="record SECONDS of UNBROKEN audio without transmitting, "
                        "for analysis a cycle-by-cycle capture cannot support")
    p.add_argument("--tune-drive", type=float, default=0.10,
                   help="soundcard drive for --tune (default 0.10, about 5 W)")
    p.add_argument("--hold", type=int, default=0, metavar="N",
                   help="after connecting, stay up and listen instead of "
                        "disconnecting -- what the peer does after our "
                        "acknowledgement is the thing worth seeing. N is how "
                        "many cycles with no payload either way to hold for, "
                        "not the length of the session: every cycle that moves "
                        "bytes buys another N, to a ceiling of "
                        f"{HOLD_MAX_CYCLES} cycles "
                        f"({HOLD_MAX_CYCLES * spec.CYCLE_SHORT_S / 60:.0f} "
                        "minutes on a shared channel). It is a bound and not a "
                        "promise: the ARQ gives up on a packet the peer stops "
                        "answering, and the run stops with the link unless "
                        "--observe-after-link-down is flown")
    p.add_argument("--listen-every", type=int, default=0, metavar="N",
                   help="on a held link, give every Nth cycle to the receiver: "
                        "render the packet, key nothing, and take the whole "
                        "1.25 s instead of the 0.24 s a keyed cycle leaves. Off "
                        "unless flown, because it changes what goes on the air "
                        "-- the peer gets a cycle with no packet in it. The "
                        "cycle costs no retry (the burst is refused at the seam) "
                        "and one slot of --hold")
    p.add_argument("--observe-after-link-down", action="store_true",
                   help="when the ARQ gives up mid-hold, spend what is left of "
                        "--hold LISTENING: receive, decode and log, and key "
                        "nothing at all. Off by default, because the ordinary "
                        "answer to a link that is gone is to stop rather than "
                        "to sit on a shared channel")
    p.add_argument("--no-p3-fallback", action="store_true",
                   help="after attempting P3, never return to P1 during this "
                        "contact; preserve upgrade confirmation, hold and "
                        "teardown limits (does not change when P3 is offered)")
    p.add_argument("--max-cycles", type=int, default=8)
    p.add_argument("--p3-mail", action="store_true",
                   help="experimental mail starting at SL1: entry +603.125 ms initial reply "
                        "pulse, retained through normal turn changes; fetch mail "
                        "and optionally send --mail-send; no timing-trial stop")
    p.add_argument("--p3-qrt-confirm", action="store_true",
                   help="with --p3-mail, try the measured terminal marker after "
                        "the QRT ACK and await opposite ACK parity within the teardown limit")
    p.add_argument("--p3-changeover-cs5", action="store_true",
                   help="experiment: reply CS5 to CRC-valid P3 changeovers, "
                        "following observed SCS exchanges; retain ordinary "
                        "packet ACKs, reply timing and final QRT ACKs")
    p.add_argument("--p3-changeover-p1-cs", action="store_true",
                   help="experiment: acknowledge a PACTOR-3 changeover packet "
                        "with a PACTOR-1 codeword, in the PACTOR-3 answer slot "
                        "and at nominal frequency. The link, the counter and "
                        "the alternation stay PACTOR-3; only CS1/CS2 changes "
                        "renderer, and only on the changeover cycle. Out of "
                        "spec: a level-3 link answers DBPSK on channels 5 "
                        "and 12")
    p.add_argument("--p3-control-waveform", choices=("current", "historical"),
                   default="historical", help="the synchronous waveform every "
                   "arm that advanced a WS8EOC counter keyed; \"current\" is the "
                   "staggered SCS-template experiment")
    p.add_argument("--p3-timing-trial", choices=("A", "B"),
                   help="bounded initial P3 receive experiment: A keeps peer-relative "
                        "timing; B rotates the emitted entry phase by 600 ms. "
                        "Holds speed, disables long cycles and ends after three "
                        "ordinary counters, 12 opportunities or 20 seconds; no mail")
    p.add_argument("--p3-entry-timing-trial", choices=("A", "B"),
                   help="bounded entry acquisition experiment: A uses zero delay; "
                        "B uses the default 4.875 ms delay to restore the trimmed pulse epoch. "
                        "Stop on P3 acquisition or 20 seconds from first entry; no mail")
    p.add_argument("--p3-timing-reply-delay", type=float, choices=(0.0, 3.125),
                   help="with --p3-timing-trial B, deliberately delay the reply grid "
                        "by this many ms; 3.125 targets entry +603.125 ms. RX/PTT "
                        "deadlines follow the grid; entry timing is unchanged")
    p.add_argument("--p3-timing-unbounded", action="store_true",
                   help="with --p3-timing-trial A/B, remove the opportunity, elapsed-time, "
                        "three-field and hold cutoffs for the initial receive turn. "
                        "No mail; peer QRT/turn changes, link loss and operator stop still end it")
    # 0909, 0911 and A5 advanced WS8EOC from audio-start; over the 43 controls
    # those arms keyed, pulse-center advanced nothing.
    p.add_argument("--p3-control-placement", choices=("pulse-center", "audio-start"),
                   default="audio-start", help="put the trimmed audio start on "
                   "the reply boundary, the placement that advanced WS8EOC; "
                   "\"pulse-center\" aims the leading pulse there and is the "
                   "experiment")
    p.add_argument("--p3-control-tail", choices=("off", "repeat"),
                   default="off", help="add the twenty-second symbol every "
                   "measured SCS control sends -- a full-amplitude repeat of "
                   "the twentieth -- to the synchronous default waveform; off "
                   "keys the 21 symbols every arm that advanced a counter flew")
    p.add_argument("--p3-control-stagger", choices=("off", "lead-5", "lead-12"),
                   default="off", help="key the half-symbol (5 ms) carrier "
                   "stagger real emitters send, alternating the leading carrier "
                   "every ARQ cycle from the named starting foot; \"lead-5\" "
                   "opens with channel 5 in front, \"lead-12\" with channel 12. "
                   "Off keys both tones on one clock. THE FOOT IS THE RISK: the "
                   "wrong arrangement puts one tone a FULL symbol out, which is "
                   "worse than synchronous, so fly one arm on each and compare")
    p.add_argument("--p3-follow-offset", choices=("all", "control", "none"),
                   default="all", help="key PACTOR-3 on the carrier offset the "
                   "peer's own PACTOR-3 arrives at, once a corroborated read "
                   "has measured it; \"control\" moves codewords only and "
                   "\"none\" keys everything at nominal, which left 31 WS8EOC "
                   "changeovers and 38 VE3KPG data packets unread on 0913")
    p.add_argument("--p3-wideband-prekey", action="store_true",
                   help="experiment: bounded header-selected short SL3 "
                        "receive before the IRS reply, including an explicitly "
                        "limited partial tail; preserve transmit placement")
    p.add_argument("--p3-keep-slots", choices=("all", "controls", "none"),
                   default="all", help="which cycles the grid may leave a "
                   "sub-symbol overrun to the emission path rather than "
                   "handing the slot back: \"all\" keys nearly every slot and "
                   "is what ships, \"controls\" gives a changeover cycle's "
                   "slot away as v10 did, and \"none\" turns the forgiveness "
                   "off entirely, which is v9 and every arm before it")
    p.add_argument("--retries", type=int, default=0,
                   help="cycles with no sign of the peer before a CONNECT "
                        "attempt is abandoned (0 = FSM default). It governs "
                        "nothing once the link is up -- --link-retries is that "
                        "budget")
    p.add_argument("--link-retries", type=int, default=0,
                   help="unanswered cycles a LIVE link spends before the ARQ "
                        "signs off with a QRT (0 = FSM default). This is the "
                        "counter that ends sessions; several fallback budgets "
                        "are calibrated to sit below it, so raising it lengthens "
                        "the session and lowering it reorders them")
    p.add_argument("--replay", metavar="WAV",
                   help="drive the session from a recorded capture instead of "
                        "the sound card -- exercises the whole loop and FSM on "
                        "the corpus, with no radio")
    p.add_argument("--replay-realtime", action="store_true",
                   help="pace --replay at wall-clock speed, so the loop's timing "
                        "is measured rather than assumed")
    p.add_argument("--listen", type=float, default=0.0,
                   help="RX window per cycle (s). 0 = the rest of the PACTOR "
                        "cycle after the burst, which is what a peer expects.")
    p.add_argument("--cycle", type=float, default=spec.CYCLE_SHORT_S,
                   help="ARQ cycle length (s); the call repeats on this grid")
    p.add_argument("--tx-offset", type=float, default=TX_OFFSET_S,
                   help="seconds from a PEER burst starting to our RF starting, "
                        "used ONCE to place the free-running grid when a hush "
                        "finds a station to call on. It is 'cycle - packet - d' "
                        "for the 105 ms turnaround this default implies, which "
                        "is the one measured between two commercial modems; "
                        "after the grid is "
                        "placed nothing the peer does moves our transmit timing.")
    p.add_argument("--max-key", type=float, default=40.0)
    # ONE DECISION -- which door a session may reach a protocol above PACTOR-1
    # by -- so the parser takes one flag of the group. A session cannot both
    # refuse the upgrade and be pinned to it, it cannot hold the link in
    # PACTOR-1 and also key the waveform on a grant, and it cannot refuse the
    # uninvited upgrade and offer an uninvited PACTOR-2 rung behind it.
    p.add_argument(
        "--p1-setup-phase", choices=("reply", "call"), default="reply",
        help="initial P1 transmit phase: reply adopts the peer CS sense "
             "(normal); call experimentally preserves the alternating call "
             "phase until first payload ACK or role change. Requires a "
             "PACTOR-1-only arm; does not enable P3 or change packet bytes.")
    level = p.add_mutually_exclusive_group()
    level.add_argument(
        "--pactor1-only", action="store_true",
        help="never offer the PACTOR-3 upgrade, so the whole session stays on "
             "the connect's 1400/1600 Hz. It is also what entitles the "
             "launcher's channel sense to judge 1200-1800 Hz instead of the "
             "380-2620 an upgraded link can fill (core.occupied), so the two "
             "are set from one flag. A mail arm takes this by default; naming "
             "any other flag of this group is how an operator opts out.")
    level.add_argument(
        "--pactor3-only", action="store_true",
        help="take the upgrade on the first acknowledged packet, with or "
             "without traffic behind it, and hold the link there: neither the "
             "peer answering in PACTOR-1 nor a run of unanswered cycles brings "
             "it back down. The link still OPENS in PACTOR-1, because a "
             "connect burst is PACTOR-1. For a rig test that wants the upgrade "
             "as its one variable rather than as emergent behaviour.")
    level.add_argument(
        "--p1-grant-only", "--p1-act-on-grant", action="store_true",
        dest="p1_grant_only",
        help="make the peer's grant the ONLY door into PACTOR-3: the uninvited "
             "upgrade off the next acknowledged packet is refused, so an arm "
             "asking what a gateway does with an INVITED PACTOR-3 has one "
             "variable. --pactor1-only cannot do this -- it declines the grant "
             "too -- which is why the parser refuses that pair. Answering the "
             "grant itself needs no flag: the old --p1-act-on-grant spelling "
             "still names this restriction, which is the other half of what it "
             "always meant. Pair with --p1-status-bits45, which is what asks.")
    level.add_argument(
        "--p3-uninvited", action="store_true",
        help="go up without being invited and advertise nothing: leave the "
             "upgrade open so it is taken off the next acknowledged PACTOR-1 "
             "packet with bytes behind it, and do not restore the announcement "
             "a mail arm clears. It is the only door that asks whether a "
             "gateway gates on a capability we DECLARE or simply reads what we "
             "key, and it needs no flag on an upgrade arm -- that is the "
             "default there. On a mail arm it does, because a mail arm holds "
             "itself in PACTOR-1 unless a door is named, and dropping "
             "--p1-grant-only without naming this one is what silently turned "
             "three 2026-09-17 arms into PACTOR-1 runs. The PACTOR-3 phase it "
             "opens starts at --p3-entry-sl carrying the host's own bytes, not "
             "at the granted door's speed level 1 behind a template.")
    level.add_argument(
        "--decline-grant", action="store_true",
        help="DECLINE the peer's 0x59A -- name the upgrade grant in the log and "
             "stay in PACTOR-1, refusing a capability the far end has just said "
             "it has. The link otherwise goes as far as both ends allow, so this "
             "is the opt-out and not the default: a gateway that grants may serve "
             "nothing below PACTOR-2, and the grant is the only place our own "
             "PACTOR-3 has ever been invited.")
    level.add_argument(
        "--offer-pactor2", action="store_true",
        help="offer PACTOR-2 uninvited, AFTER a peer has answered our PACTOR-3 "
             "in PACTOR-1 and ruled it out for the link. Off, because the "
             "corpus holds no PACTOR-1-to-PACTOR-2 transition at all -- both "
             "graded third-party completions go 1 -> 3 in one step -- and our "
             "PACTOR-2 codeword keying is graded by nothing outside this "
             "package. What it keys is a speed-level-1 short packet, DBPSK on "
             "both carriers behind the frame marker that names it. FOLLOWING a "
             "peer that leads into PACTOR-2 needs no flag: there the far end "
             "has already keyed the waveform.")
    p.add_argument("--p4-entry-attempts", type=int, default=2, metavar="N",
                   help="P4 probe chirp cap, 1..4 (default 2); requires --p4-entry")
    p.add_argument("--p4-entry-timeout", type=float, default=60.0, metavar="SECONDS",
                   help="P4 probe deadline after grant, at most 60 seconds; requires --p4-entry")
    p.add_argument("--p1-6a9", action="store_true",
                   help="announce P1 bits45=1 and select 0x6A9 as the entry response; "
                        "does not change the entry waveform or P1-only policy")
    p.add_argument("--p4-entry", action="store_true",
                   help="experimental PACTOR-4 SL1 chirp entry after a grant; "
                        "off by default. Selects the p4chirp entry and grant-only "
                        "upgrade. No connected P4 ARQ or frame-type selection. "
                        "Keeps the QRM guard; --no-qrm-guard is a separate override.")
    p.add_argument("--p3-entry", type=_entry_ladder, metavar="RUNGS",
                   default=",".join(ArqConfig.entry_ladder),
                   help="the entry packets a granted upgrade keys, in order, "
                        "four cycles each: 'template' is DL6MAA's shape -- an "
                        "empty field, so the walking template fills it, at the "
                        "data type the reference declares, unswapped; 'burst' is "
                        "the same packet behind the acquisition preamble; 'data' "
                        "is the user text at type 0 this station keyed through "
                        "2026-08-26 and no gateway took. A granted peer asks 13 "
                        "to 17 times, so a ladder spends cycles it was already "
                        "offering. 'burst' IS REFUSED: it renders 1.074 s and "
                        "keys one slot in two against a 210 ms listen floor, so "
                        "it is offered on half the cadence of 'template' and the "
                        "two cannot be scored against each other until the rung "
                        "or the cycle changes (arq.ENTRY_RUNGS_GROUNDED). "
                        "'p4chirp' answers the grant with PACTOR-4 speed level "
                        "1's two-tone chirp entry instead of any PACTOR-3 "
                        "packet -- 3.3 s of audio keyed one slot in three, for "
                        "a gateway whose grant may be commanding its own "
                        "maximum rather than ours. Default template.")
    p.add_argument("--p3-entry-sl", type=int, choices=range(1, 7),
                   default=ArqConfig.entry_sl, metavar="SL",
                   help="open an UNINVITED PACTOR-3 phase at this speed level "
                        "instead of the default 3. A GRANTED one opens at "
                        "ptc.GRANT_ENTRY_SL whatever this says: the peer that "
                        "sent 0x59A is waiting for the entry packet and both "
                        "reference recordings key that one at level 1, on tones "
                        "5 and 12 -- the pair every level shares, which is what "
                        "makes it acquirable off a PACTOR-1 raster. Uninvited it "
                        "has gone out at 3 every time this station has keyed one: "
                        "all 27 on record, at stations never shown to read even "
                        "the level-1 entry behind a grant, which is why that null "
                        "is void rather than negative. --p3-entry-sl 1 keys the "
                        "acquirable packet at a peer that was not told to expect "
                        "anything, which is the question the 27 were meant to ask.")
    p.add_argument("--p3-traffic-sl", type=int, choices=range(1, 7),
                   default=ArqConfig.traffic_sl, metavar="SL",
                   help="run the traffic at this speed level once the peer has "
                        "answered the entry packet, instead of the default 3. "
                        "The entry packet is speed level 1 whatever this says, "
                        "and a peer that reads it has proved it acquired THAT "
                        "level: VE3KPG took the entry twice on 2026-09-13 and "
                        "then answered 32 and 17 speed-level-3 packets with the "
                        "same codeword, accepting none. --p3-traffic-sl 1 holds "
                        "the traffic where the peer was last read and leaves the "
                        "climb to the peer's own CS4, which is the only gear "
                        "command the protocol gives it.")
    p.add_argument("--p3-speed-up", choices=("auto", "hold"),
                   default=ArqConfig.speed_up,
                   help="whether this station may ask the peer for a higher "
                        "speed level. 'auto' (the default) keys CS4 once three "
                        "consecutive cycles have each delivered a new packet "
                        "-- a repeat, or a cycle nothing was read in, ends the "
                        "run; 'hold' keys none at all, from "
                        "either gear seam, and the link stays at the level the "
                        "entry opened it on. For a path already known to be "
                        "marginal: WS8EOC on 80 m 2026-09-13 21:35 was asked "
                        "for SL2 and went from 38%% of the peer's cycles read "
                        "to 3%%, the gateway repeating one SL2 block 33 times "
                        "while a witness heard every one of them.")
    p.add_argument("--p3-tx-gear-hold", action="store_true",
                   help="experimental outbound P3 gear handling: treat a held "
                        "CS4/CS5 as a repeat request, preserving the pending "
                        "bytes and counter until an ACK or different command; "
                        "MAXTry still bounds the speed-up trial")
    p.add_argument("--p3-max-try", type=int, choices=range(1, 10),
                   default=ArqConfig.p3_max_try, metavar="N",
                   help="SCS MAXTry: total transmissions at a trial speed before "
                        "returning to the previous speed (1..9, default 2)")
    p.add_argument("--p3-max-down", type=int, choices=range(2, 31),
                   default=ArqConfig.p3_max_down, metavar="N",
                   help="SCS MAXDown: consecutive receive errors before requesting "
                        "one speed level down with CS5 (2..30, default 6)")
    p.add_argument("--p3-repeat-gear", type=int, default=ArqConfig.repeat_gear,
                   metavar="N",
                   help="after N identical packets answered with one identical "
                        "codeword, key CS4 -- the speed-up request, which also "
                        "acknowledges -- instead of another alternation word, "
                        "then resume the alternation. 0 (the default) is "
                        "today's behaviour. The ordinary climb spends ACCEPTED "
                        "traffic packets and a repeat is not one, so a peer "
                        "that repeats produces no clean run and the link never "
                        "leaves the level it stalled on: WS8EOC repeated "
                        "status=0x21 seq=1 thirty-six times on 2026-09-13 and "
                        "that arm keyed no CS4 at all, while both arms that "
                        "delivered the greeting keyed one within three packets.")
    p.add_argument("--p3-entry-stagger", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="lead the entry and changeover packets' lower carrier "
                        "by half a symbol, which is what DL6MAA keys: its "
                        "channel 5 runs T/2 ahead of channel 12 and shrike put "
                        "both on one clock, which is the render four granted "
                        "arms keyed and none was ever acknowledged. It governs "
                        "the two-carrier keyings only -- the entry packet and "
                        "the changeover packet -- and nothing else on the link. "
                        "--no-p3-entry-stagger is the one-clock control.")
    p.add_argument("--p3-rise", "--p3-entry-rise",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="shape every PACTOR-III keying with the protocol's own "
                        "symbol pulse rather than the generic raised cosine, "
                        "which put 40 ms of kernel in front of the first symbol "
                        "where DL6MAA keys 18. Whole-waveform correlation "
                        "against the reference entry 0.809 to 0.877. The data "
                        "packets at speed levels 2 to 6 were the last keyings "
                        "still on the generic filter, and it cost them 20 ms of "
                        "energy ahead of the packet and a phase reference 27.8 "
                        "ms past the boundary where the entry's is at 13.9. "
                        "--no-p3-rise is the flown-kernel control; "
                        "--p3-entry-rise is the old name for this flag.")
    p.add_argument("--p3-data-flush", choices=("zeros", "entry", "reference"),
                   default="zeros",
                   help="what an ordinary data packet ends its trellis on. "
                        "'zeros' is the terminated code every receiver assumes "
                        "and every arm has flown; 'entry' keys the eight bits "
                        "measured off DL6MAA's entry packet "
                        "(placement.ENTRY_FLUSH) on the data packets as well, "
                        "which is the remaining difference between the entry a "
                        "peer read in five keyings and the speed-level-1 data "
                        "packets the same peer answered CS1 thirty-five times; "
                        "'reference' keys the six bits all six of that "
                        "recording's clean level-3 data packets end on "
                        "(placement.REFERENCE_FLUSH), levels 2 to 6 only. The "
                        "entry packet keys its own either way.")
    p.add_argument("--p3-entry-delay", type=float, nargs="?", metavar="MS",
                   const=DEFAULT_ENTRY_DELAY_MS, default=DEFAULT_ENTRY_DELAY_MS,
                   help="delay P3 entry audio after its slot boundary "
                        f"(default, also with a bare flag: {DEFAULT_ENTRY_DELAY_MS:g} ms). "
                        "Restores the leading time removed by the waveform trim. "
                        "Use 0 for historical timing. Samples and transmit guards "
                        "are unchanged.")
    p.add_argument("--announce-lower", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="send the callsign announcement lowercase (1w9ssj), "
                        "as both completions on tape do. --no-announce-lower "
                        "is the uppercase form every arm flew before "
                        "2026-09-08.")
    p.add_argument("--p1-drive", type=float, default=1.0,
                   help="scale ONLY the PACTOR-1 legs (connect, data packets, "
                        "control signals, break-in) to tx-drive times this, "
                        "leaving PACTOR-3 at full drive. Both modems of the "
                        "reference session key PACTOR-1 well below their "
                        "PACTOR-3 -- DL6MAA's SL1 entry carries +7.0 dB over "
                        "its own PACTOR-1 average, ours -5.8 dB under it at "
                        "1.0, a 13 dB convention gap that cannot be closed "
                        "upward (the amplifier is at its rail). 0.35 "
                        "reproduces the reference ratio. Default 1.0: "
                        "unchanged behaviour.")
    p.add_argument("--p1-watts", type=float, default=None,
                   help="RF POWER, in watts, for the PACTOR-1 phase -- written "
                        "to the rig as RFPOWER over the arm's own CAT "
                        "connection, before the first call and again whenever "
                        "the link falls back or ends. The audio drive cannot "
                        "set this: the ALC holds the two-tone PACTOR-3 entry "
                        "down by its crest and passes a constant-envelope FSK "
                        "packet at whatever RFPOWER allows. Default: the level "
                        "the rig is already on, untouched.")
    p.add_argument("--p3-watts", type=float, default=None,
                   help="RF POWER, in watts, for the PACTOR-3 phase -- written "
                        "the instant the peer's 0x59A grant is read, a "
                        "turnaround ahead of the entry packet and never inside "
                        "a pre-key window. The level the rig read back at arm "
                        "start comes back at teardown. See --p1-watts.")
    p.add_argument("--p1-status-bits45", type=int, default=None,
                   choices=(0, 1, 2, 3),
                   help="announcement status bits 4-5, as a 2-bit value: 0 none, "
                        "1 bit 4 only (0x11), 2 bit 5 only (0x21), 3 both (0x31). "
                        "Default 3 on an upgrade arm: the value both recorded "
                        "PACTOR-3 completions' callers announced (W4DNA 0x31, "
                        "DL6MAA 0x35) and the only one that has ever drawn a "
                        "0x59A grant here -- every granted session on record "
                        "announced at 3, and none at 0, 1 or 2. Default 0 on a "
                        "mail arm, where the measured cost of 3 is the data path "
                        "itself: the peer's counter freezes at #1 and nothing is "
                        "acknowledged either way.")
    p.add_argument("--qrm-guard", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="refuse to key where the peer's last DECODED codeword, "
                        "projected one cycle on its own raster, would still be "
                        "on the air or would start inside our own burst. This "
                        "is the SENDING side's guard and there was none: on "
                        "2026-08-26 this station transmitted over WS8EOC 20 "
                        "times in one session and its own capture could not "
                        "show it, because the rig mutes the receiver while we "
                        "key. It rests on a twelve-bit word read at zero "
                        "errors, never on an energy forecast; its band is "
                        "`_d_max_n`, so it refuses no turnaround the grid would "
                        "acquire; and it stands down for one burst after 3 "
                        "refusals in a row, so it cannot deadlock a link. "
                        "--no-qrm-guard turns it off mid-slot and every line it "
                        "prints names that flag.")
    p.add_argument("--p1-status-from", type=int, default=1, metavar="K",
                   help="hold --p1-status-bits45 back until the Kth data packet "
                        "of the session, 1-based; the packets before it go out "
                        "with bits 4-5 clear. Default 1, from the first packet. "
                        "Every station on record sets them on the FIRST packet "
                        "of a link and never mid-link, so whether a peer reads "
                        "them once at connect or on every packet is untested -- "
                        "switching on an established, advancing link is the arm "
                        "that tells the two apart. A retransmission is the same "
                        "packet and does not count towards K.")
    p.add_argument("--long-cycle", "--long-cycles", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="let the link climb onto the 3.75 s cycle -- status "
                        "bit 5 while more than a long field is queued, and CS6 "
                        "granting a peer's ask. It is the sustained exchange "
                        "rather than an optimisation: the reference session "
                        "spends 17.9-51.6 s of its traffic phase there and its "
                        "IRS reaches for CS6 on the seventh data cycle. "
                        "--no-long-cycle declines it for the slot, which is one "
                        "raster and one packet length for the whole session; a "
                        "CS6 that arrives anyway is still followed, because a "
                        "station holding the short raster through one "
                        "desynchronises the link. Sending CS6 records a pending "
                        "change; a fresh decoded packet determines the peer's "
                        "actual length. Unreadable answers retain possible-long "
                        "receive protection.")
    p.add_argument("--breakin-lead-72", dest="breakin_at_boundary",
                   action="store_false", default=True,
                   help="THE CONTROL. Key the changeover packet "
                        f"{BREAKIN_LEAD_S * 1e3:.0f} ms after the peer's packet "
                        "ends -- where the reference modems key THEIRS at us "
                        "(71.3-71.9 ms off WS8EOC, both arms of 2026-08-30) -- "
                        "instead of where this station's codewords are read. "
                        "Flown on 2026-09-04 it put 18 keyings at +68 to +70 ms "
                        "against a reader whose acknowledged window is "
                        "88.5-98.0, and drew nothing. Default is 170 ms less "
                        "the tracked turnaround, past the peer's packet, which "
                        "is our own boundary and is where the one accepted "
                        "changeover on record sat (+94.9 ms, KB5LZK, "
                        "2026-08-22).")
    p.add_argument("--p1-ack-lsb", action=argparse.BooleanOptionalAction, default=True,
                   help="send the CS least-significant-bit first, which is what "
                        "the air carries -- WS8EOC's answer train reads CS1 then "
                        "CS4 this way round and CS2 then CS3 the other, and "
                        "neither of those is a legal connect answer. "
                        "--no-p1-ack-lsb is for comparison only.")
    p.add_argument("--tx-latency-ms", type=float, default=None,
                   help="key every burst this many milliseconds early, for the "
                        "part of the ADC->DAC loop the driver does not report. "
                        "Measured by keying into a loopback and reading where "
                        "our own burst lands in our own capture; this station "
                        "measures 20. It is the audio path's, not any "
                        "protocol's, so it applies to every emission and PTT "
                        "moves with it. Defaults to [audio] tx_latency_ms from "
                        f"the station file named by {config.STATION_ENV}, else "
                        "0 -- unset, nothing moves.")
    p.add_argument("--tx-drive", type=float, default=None,
                   help="soundcard peak for TX (0..1); the audio level sets RF "
                        "power on a data interface. Raise against a watched "
                        f"meter. Defaults to [audio] tx_drive from the station "
                        f"file named by {config.STATION_ENV}, else {TX_DRIVE}.")
    p.add_argument("--outdir", default="captures/onair")
    typed = sys.argv[1:]
    p.add_argument("--rx-assessment", help="Fresh guarded-launcher RX/CCA receipt; readback-only startup")
    p.add_argument("--rx-claim-fd", type=int, default=9, help="Existing inherited station claim descriptor")
    args = p.parse_args(typed)
    _p1_response_defaults(args)
    _p4_entry_defaults(args)
    _p3_mail_defaults(args)
    args._typed_argv = _redact_argv(typed)
    args.mail_password = config.mail_password(args.mail_password, args.mail_password_file)
    return run(args)


# What the unwind gets before the interpreter is taken down under it. Every other
# deadman in this tree arms with the key ALREADY DOWN -- `Rig.panic` unkeys first
# -- and can afford `PANIC_BUDGET_S`. Here the key is still up when the handler
# runs and the unkey is three `finally` blocks below: a burst already on the air
# runs out its cycle, `live.close()` puts the card and the session recording away,
# and `ota.Rig.stop` then walks its whole ladder -- the live pipe, up to five
# seconds taking rigctl off the serial port, and the one-shot `T 0` that is the
# unkey which holds. Firing before that ladder finishes would REMOVE the last
# software path to a dead transmitter rather than provide one.
#
# Bounded above by what reaps this modem: `tools/onair.sh` and `onair_session.py`
# both give a child 12 s to unkey itself and then send SIGKILL, which ends the
# handler and the deadman with it.
DEADMAN_BUDGET_S = 10.0


def _unkey_on_signal() -> None:
    """Turn a termination signal into the exception the `finally` blocks expect.

    Every unkey in this module hangs off a `finally`, and a `finally` runs when
    Python unwinds -- not when the process is killed. So a session that was
    holding the key when it received SIGTERM died with the transmitter still
    asserted, and stayed that way until someone reached the radio. That happened
    on 2026-08-04, on 10.1 MHz, and the operator had to power the rig down.

    `timeout(1)` sends SIGTERM. So does a supervisor, a harness cancelling a
    tool call, and `kill` with no argument -- which is to say the ordinary ways
    an unattended run is stopped are exactly the ways that left it keyed.
    Raising here puts the unwind back on the normal path: the `finally` in
    `RadioTx._tx` drops the line, the rig wrapper's own `finally` drops it
    again, and the watchdog is the third.

    AND THE UNWIND ITSELF IS BOUNDED, because three `finally` blocks are three
    places to wedge: the PortAudio teardown and the WAV write below us are both
    capable of it, and a SIGTERM spent inside one is a signal the operator has
    already used up. The deadman is under all of it -- not instead of the unkey
    above, which is what actually lowers the line, but so that a session which
    cannot finish dying stops holding the ports it is dying with.

    SIGINT is answered with `KeyboardInterrupt` and the other two with
    `SystemExit`, because Ctrl-C is the one that has a reader waiting. CPython's
    own SIGINT handler raises `KeyboardInterrupt`, `run` catches it so the
    summary is the last thing on the screen rather than the first thing a
    traceback scrolls away, and the exit status stays the session's own. A
    `SystemExit` here would walk past that clause and spend the verdict to buy a
    backstop the operator already gets underneath it.

    SIGKILL cannot be caught and never will be. The only defence left against it
    is the hardware side -- a serial line that falls when the port closes -- and
    on this station that is a hope and not a mechanism: unmeasured on the adapter
    (`core.ptt`), and on 2026-08-04 it did not happen. Which is why anything that
    reaps a modem here sends SIGTERM first and gives it a moment to reach this
    handler, and proves the line down afterwards rather than trusting the signal.
    """
    def _die(signum, _frame):
        arm_deadman(DEADMAN_BUDGET_S)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(f"signal {signum} -- unkeying and stopping")
    for sig in FATAL_SIGNALS:
        try:
            signal.signal(sig, _die)
        except (ValueError, OSError):
            pass            # not the main thread, or the platform lacks it


if __name__ == "__main__":
    _unkey_on_signal()
    sys.exit(main())
