#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Attempt a kestrel -> VARA connect over a real audio path (or the Wine loopback).

Drives the VARA connect handshake (kestrel/vara/vara_arq.VaraStationHandshake):
CR -> connect-response -> link-setup -> connected-ack -> CONNECTED, over a
half-duplex audio transport keyed on the PTT line. Run it at the rig once the
antenna is tuned; it reports whether the gateway answered and where it stalled.

Every attempt records what it heard, to a timestamped WAV under ``logs/onair``
(``--record`` elsewhere, ``--no-record`` not at all). An attempt nobody recorded
cannot be diagnosed afterwards, and every advance this receive chain has made came
out of a recording.

    # against a real gateway on the rig:
    python3 tools/kestrel_connect.py --gateway W9XYZ --mycall MYCALL \
        --tx-device "USB Audio Device" --rx-device "USB Audio Device" \
        --rigctld 127.0.0.1:4532

    # bench validation against a Wine VARA (no rig, no PTT):
    python3 tools/kestrel_connect.py --gateway W1AW --mycall MYCALL \
        --tx-device "BlackHole 16ch" --rx-device "BlackHole 64ch"
"""
from __future__ import annotations

import argparse
import atexit
import csv
import datetime as _dt
from functools import partial
import os
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "hfmodem"))
from vara_monitor import (GATEWAY_LIST, SEG_FRAME, Segmenter, Session, classify,
                          emit, load_gateways)
from vara_rig_bridge import Rig, _install_unkey_handlers

from hfmodem.core import band, config, cwid, identification, levels, rates
from hfmodem.core.audio import (WARM_S, drop_output, hold_output, play_drained,
                                warm_output)
from hfmodem.core.busy import (DEAF_DBFS, MUTE_DROP_DB, POUNCE_WINDOW_S, level_range,
                               receiver_fault)
from hfmodem.core.busy import is_busy as _is_busy
from hfmodem.core.occupied import FILTER_HZ, clipped, occupied_hz
from hfmodem.core.ptt import arming_refusal
from hfmodem.core.rigs import RIGS, mode_passband, resolve_rig
from hfmodem.core.wav import store_later
from hfmodem.kestrel.host.busy_gate import BandGate
from hfmodem.kestrel.vara.vara_arq import (
    _I_CR_SENT,
    _I_LINKSETUP_SENT,
    _LINKSETUP_MAX_TX,
    _TURN_ASKED,
    OVER_CONTINUE_ANSWERS,
    OVER_CONTINUE_BURST,
    VaraIO,
    VaraState,
    VaraStationHandshake,
)
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.winlink import CLIENT_SID, progress_to_stdout

_ROOT = Path(__file__).resolve().parents[1]
FS = 48000
# The receiver's own health is `core.busy`'s judgement, not a second opinion held
# here: the same two levels, the same thresholds, and the same sentence the other
# two listening tools print. What that module keeps a muted receiver out of is its
# occupancy score; what `receiver_fault` keeps it out of is a key-up.

# How long PTT stays up past the end of a burst, once the audio is out. It used to
# cover two different things under one number — waiting for the transmission to
# actually leave the device, and then holding the key idle — and could not tell
# them apart, so shortening it to uncover a peer's answer also risked cutting the
# transmission short. `core.audio.play_drained` now waits on the device rather than
# on a constant, and what is left here is only the idle part.
#
# Why the idle part costs anything: a peer answers from 0.10 s after our last
# transmitted sample, and this station's input stays dead for the whole hold plus
# its own mute recovery. At a 0.25 s hold the input was measured dead for 0.42 s
# after every transmission, so a connected-ack's four-symbol preamble landed
# entirely inside the gap and no connect could complete. At 0.05 the input is dead
# for 0.088-0.244 s past our last transmitted sample — median 0.182 over the 56
# changeovers in the two KC9GHZ session recordings, and 0.175 over the 104 of the
# 2026-08-23 session recordings (0.161-0.190 at the tenth and ninetieth
# percentiles), read off the recordings and now off the line below too. The hold
# is 0.05 of that and PTT release and the receiver's own T/R recovery are the
# rest, so this is the only part of the blackout the software holds. On 2026-08-06
# against KC9GHZ the connect-response went from 9/15 tones the receiver was live
# for to 14/15, the connected-ack was heard for the first time, and the session
# CONNECTED.
#
# WHAT THE REMAINING 0.05 BUYS A PEER IS ONE SYMBOL, and that is why it stays.
# Cutting it to zero would leave ~0.125 s of blackout against a turnaround that
# opens at 0.10, so the head of an answer is lost either way — and the head is
# affordable: KB3AC-10's answers to the 2026-08-23 turn-requests score 30/30 and
# 30/31 payload tones off that session's recording with anything from 0.00 to
# 0.60 s of their heads cut away  [see _payload_alignment, which scores a symbol
# outside the audio as not comparable rather than as wrong].
#
# 0.05 rather than zero because what the transmit stream can report ends at
# PortAudio: the USB codec's own buffering and the rig's audio path are below it
# and report nothing. `tools/ptt_tail_check.py` is what measures the whole path
# from the outside, and it is what should set this number — the one run held
# (2026-08-06, at the old 0.25 hold) has the full 3.66 s of audio radiating and
# the key coming down 0.42 s after it. No run has been made at 0.05.
TX_IDLE_HOLD_S = 0.05

# Receive audio discarded after every transmission, on top of the idle hold above.
# The rig monitors its own transmit into the RX codec and the input stream delivers
# its frames a codec buffer behind the present, so the samples recorded either side
# of the unkey are ours whatever the band is doing.
#
# Measured from the hold rather than fixed beside it, because PTT is up for the
# whole of the hold and those samples are ours. At the flat 0.1 s this replaces,
# `--tx-tail 0` still discarded 0.1 s of receive audio: the knob shortened the
# keying and not the blackout it is documented against.
#
# It reads as the thing that takes the head off a gateway's answer, and it is not.
# The hold and the guard come to 0.100 s, KC9GHZ answers a turn-request from
# 0.100 s, and the two landing on one number is the whole of that appearance. What
# the recordings say is that the receiver is already gone: over 56 changeovers in
# the two KC9GHZ sessions the input sits at the codec floor for 0.088-0.244 s past
# our last transmitted sample, so 55 times out of 56 the guard ends while the
# receiver is still deaf and what it discarded was 40 dB under the band noise.
# Shrinking it buys silence, and letting a recogniser reach back into it buys the
# same silence at the price of our own transmission being in reach.
#
# There is no echo tail to size it against either: our own burst is 35 dB down in
# the first two-millisecond frame past its last sample, 45 dB down in the next and
# at the codec floor after that, on both sessions. What is covered is not decay
# but the offset between the delivered sample count `tx` reads and where our audio
# really ended in the stream — the USB codec's buffering and the rig's audio path,
# which report nothing, the same gap the idle hold covers from the transmit side.
#
# So it stays, and PEER_ANSWERS_FROM_S is what bounds it: the cursor may not skip
# past the instant a peer can start answering. What does cost a gateway's answer
# its opening symbols — up to 0.14 s of it, three of the 32 a frame carries, which
# is why one arrived at 29-31 tones — is the radio's blackout, and TX_IDLE_HOLD_S
# is the only 0.05 of that this file can move.
RX_ECHO_GUARD_S = 0.05

# When a peer's answer starts, measured off air and taken as the earliest of
# everything the recordings hold. KC9GHZ keys its drained-responder 0.100-0.105 s
# after our turn-request's last sample — three occurrences over two sessions
# (2026-08-17, and twice on 2026-08-22), 32/32 tones on an independent receiver
# 103 mi away — and its idle-response 0.112-0.150 s after a turn-idle over ten
# answers. NS0A opens its connect-response 0.121-0.158 s over six attempts, and
# answered a real VARA's link-setup 0.171 s after that burst's last sample.
#
# So the floor over every measurement is the turn-request answer, and this is that
# observation exactly rather than a bound under it: the number is reported and
# never gated on, and padding it would only make the blackout printed beside it
# read better than it is. Every number about the changeover is meaningless without
# it, which is why the line carries it.
PEER_ANSWERS_FROM_S = 0.10

# Audio measured after each transmission to see how much of the peer's answer we
# were awake for. Long enough to contain the latest a connected-ack can end
# (0.17 s turnaround plus its 0.48 s), short enough to end before the next thing
# either station keys — past that the window fills with the answer itself and with
# our own next burst, and the measurement stops being about the gap.
GAP_WINDOW = int(0.8 * FS)

# The resolution everything about the changeover is read at. Long enough that the
# quietest frame inside a keyed burst still stands 25 dB clear of DEAF_DBFS, short
# enough that the unkey lands inside one frame of where it happened.
_MUTE_FRAME_S = 0.02

# How long the input has to stay up to be the BAND coming back rather than our own
# transmission on its way out. What follows our last sample is not a mute and then
# the band: it is our burst decaying through a receiver that is already muted, and
# it crosses DEAF_DBFS from above. Over the seven keyings of the two 2026-09-11
# KC9GHZ arms that decay holds over the threshold for at most two frames (-47 dB in
# the first, -46 in the second, past -70 by the fifth) before the deep mute, where
# the band once back stands at -11 to -14 dB and stays there for the rest of the
# window. So the first frame above the threshold read the ramp and reported the
# blackout as 0.00-0.04 s against a real 0.16-0.19; the first frame of a run this
# long reports 0.16-0.18 on the same seven. It costs nothing to wait for — the
# window is already GAP_WINDOW long and nothing is gated on the answer.
_LIVE_RUN_S = 0.10


def _live_frames(audio: np.ndarray, f: int) -> np.ndarray:
    """One flag per whole frame of F samples: was the receiver delivering audio?"""
    n = len(audio) // f
    rms = np.sqrt((audio[:n * f].reshape(n, f) ** 2).mean(1)) + 1e-15
    return 20 * np.log10(rms) > DEAF_DBFS


def _tx_end_in_capture(audio: np.ndarray, earliest: int) -> int:
    """Where our own transmission ended, read off the capture rather than the clock.

    `play_drained` returns at a wall-clock instant, and the capture buffer's length
    at that instant is a place in the bookkeeping and not a place in our burst:
    whatever the transmit path spends between our last sample and that return is
    charged to the channel. Over the three 2026-08-29 arms it is 0.00-0.06 s on 48
    ordinary turnarounds and 0.62 s on the three that follow the 11-symbol control
    burst — and that is the turnaround where a gateway keys its own release, so we
    skipped 0.58 s of a 0.71 s frame three times out of three and answered a
    handover we never saw with keepalives.

    The rig monitors our transmit into the receive codec, so the burst is in this
    stream at -20 dBFS with the unkey 50 dB under it: the edge from loud to dead is
    our own last sample, to a frame. Taken as the LAST such edge below the buffer's
    end rather than the first, which is what makes it safe — the first is the
    key-up mute whenever the output stream's open outlasts the burst it opened for,
    and a cursor set there hands our own transmission to the decoder. Over every
    keyed run in the three recordings, at every lead-in from 0.02 to 0.80 s, the
    last edge is the burst's end and nothing else.

    `earliest` is where our last sample could soonest have been captured, which is
    the buffer's length when we started writing plus the burst. Below it there is
    nothing of ours to find, and above it the only edge that can precede the burst's
    end is the key-up mute the rule already steps over.

    WITH NO EDGE AT ALL THE DRAIN HAS NOT COME BACK EARLY, whatever this returning
    `len(audio)` then makes `late` say. The burst does not step under DEAF_DBFS: it
    decays out of the muted receiver at about half a decibel a millisecond, so a
    -20 dBFS monitor takes 60 ms to cross, and whether 60 ms of decay fits inside
    the buffer is what the missing edge is really deciding. Read off the whole
    capture against the burst's own level instead, over 551 turnarounds of
    2026-09-17/18, the drain measures 86 ms past our last monitored sample on the
    230 this found an edge for and 79 ms on the 321 it did not — one population.
    With that absolute-only detector `late` read 0 on 58% of turnarounds, the idle hold was slept whole, and the
    key stayed down 43 ms longer than it was owed in the September turnaround recording.
    The relative AC edge below avoids interpreting DC settling as live audio;
    the absolute detector remains the fallback when that edge is unavailable.
    """
    # A muted codec can carry a slowly decaying DC offset long after the
    # transmitted audio ended. Differencing removes that offset; a sustained
    # drop relative to our own recent audio locates the end without waiting for
    # an absolute noise-floor crossing. Keep the original conservative fallback
    # when there is no complete, well-supported edge in the available capture.
    ac_end = _tx_ac_end(audio, earliest)
    if ac_end is not None:
        return ac_end
    f = int(_MUTE_FRAME_S * FS)
    n = (len(audio) - earliest) // f
    if n < 2:
        return len(audio)
    dead = ~_live_frames(audio[earliest:earliest + n * f], f)
    edge = np.flatnonzero(dead[1:] & ~dead[:-1])
    return earliest + (edge[-1] + 1) * f if len(edge) else len(audio)


def _tx_ac_end(audio: np.ndarray, earliest: int) -> int | None:
    """Locate a sustained end of monitored audio, ignoring the codec's DC tail.

    The earliest possible end still bounds the search. Require 12 ms of live
    audio followed by 8 ms at least 20 dB below the recent monitor level; an
    unfinished capture or an isolated low OFDM symbol is not an end marker.
    Return the last supported edge, as the absolute detector does, to avoid
    treating the key-up mute as the end of our transmission.
    """
    frame = int(.002 * FS)
    lo = max(0, earliest - int(.1 * FS))
    n = (len(audio) - lo) // frame
    if earliest < int(.1 * FS) or n < 10:
        return None
    x = np.asarray(audio[lo:lo + n * frame], dtype=float)
    ac = np.diff(x, prepend=x[0])
    rms = np.sqrt(np.mean(ac.reshape(n, frame) ** 2, axis=1))
    reference = np.percentile(rms[:min(50, n)], 90)
    if reference <= 1e-8:
        return None
    live = rms >= reference * .1
    edges = np.flatnonzero(live[:-1] & ~live[1:]) + 1
    for at in edges[::-1]:
        if (lo + at * frame >= earliest and at >= 6 and at + 4 <= n
                and np.all(live[at - 6:at]) and not np.any(live[at:at + 4])):
            return lo + int(at) * frame
    return None


def _tx_last_sample(audio: np.ndarray, end: int) -> int:
    """The same edge to a tenth of a frame, for what is measured FROM it.

    ``end`` is where our burst crossed :data:`DEAF_DBFS` on its way out of a muted
    receiver, and that is not where the burst ended: the decay is 35 dB down in the
    first two milliseconds past the last sample and 45 dB down in the next, so it
    stays over the threshold for two frames of the 2026-09-11 KC9GHZ recordings and
    the crossing then lands wherever the frame grid falls. The cursor can be that
    approximate — it only has to sit past our audio, which this makes it sit
    further past — but the changeover measurement is quoted from it, and 0.02-0.04 s
    of float is a quarter of the blackout being reported. Both arms printed the same
    changeover as anything from +0.04 to +0.16 s.

    The burst's own level is the reference, taken over the tenth of a second behind
    the crossing — steady MFSK, and at least 0.06 s of it is real audio however the
    decay lands — and ``MUTE_DROP_DB`` under it is the same rule `core.busy` uses to
    tell this rig transmitting from a live band, with 10 dB to spare against the
    first sub-frame of the decay.
    """
    f = max(1, int(_MUTE_FRAME_S * FS) // 10)
    n = min(end, int(0.1 * FS)) // f
    if n < 2:
        return end
    lo = end - n * f
    rms = np.sqrt((audio[lo:end].reshape(n, f) ** 2).mean(1)) + 1e-15
    db = 20 * np.log10(rms)
    ours = np.flatnonzero(db > db.max() - MUTE_DROP_DB)
    return lo + (ours[-1] + 1) * f


# One connect-request on the air, first tone to last: 41 symbols on the hop, and
# the last one whole. Read off the waveform rather than off a stopwatch, because a
# stopwatch reading of it has already been wrong once — `--cr-interval` carried
# "the burst itself is about 6.25 s" from a third-party receiver on 2026-08-03,
# where the 2026-08-23 session recordings, which hold our own transmit monitor
# between the key-up mute and the unkey blackout, put the audio at 1.770-1.775 s
# and the keyed span at 1.85.
CR_BURST_S = ((len(VF.CR.preamble) + VF.CR.n_payload - 1) * MK.HOP + MK.STRIDE) / MK.FS


# The longest spacing between connect-requests that has ever drawn an answer from
# this station. The two arms that connected on 2026-08-23 — KB3AC-10 and KE8LVA,
# the first VARA links this station has held — asked for 10 s and put 12.7 s
# between key-ups; past that the ladder is only keeping a gateway waiting, since a
# gateway answers a request it hears 0.10-0.16 s later and what makes it take 35-48
# s to answer at all is which request it is free for, not how long it has to think.
_CR_INTERVAL_MAX_S = 10.0


# What an answer that is not the connect-response buys: the gateway is listening
# this second, so the next request goes out now rather than on the cadence — but
# not on top of a second answer keyed behind the first. On the 2026-09-09 bench a
# stock responder answers each request exactly once and never repeats behind it,
# and outside a live session the tightest gap between two of its keyings anywhere
# in those captures is 0.89 s  in the September lattice recordings.
_CR_AFTER_ANSWER_S = 1.1

#: What a re-keyed request goes out as, by ``--cr-retry-form``.
_CR_FORMS = {"full": "full 41-symbol form", "stock": "stock 32-symbol form"}


def _over_continue(name: str) -> str:
    """The 8-symbol burst was named `captured` while it was a copied tail, and
    flight plans on file still say so."""
    return OVER_CONTINUE_BURST if name == "captured" else name


def cr_cadence(timeout: float, max_cr: int) -> float:
    """Seconds of listening between connect-requests, so the ladder covers the
    whole attempt rather than the front of it.

    The interval is dead time AFTER a request — `connect` re-arms it when
    `originate` returns — so a rung costs :data:`CR_BURST_S` plus this, and
    ``max_cr`` rungs at the flat 4.6 s default were spent in 53 s whatever the
    operator asked ``--timeout`` for. On 2026-08-23 that is exactly what arm 7
    did: eight requests inside the first 37 s of a 180 s window, then 143 s with
    nothing left to call with, against a gateway that answered the same station's
    calls at 40, 46 and 48 s on the arms either side of it.

    So the ladder is stretched to the window instead of the window being spent on
    the ladder, and raising ``--timeout`` now buys what an operator expects it to.

    Floored at the burst, which is the one interval the waveform forbids going
    under: below it a repeat would key before the previous request had left the
    air and the gateway would be answering into our own next transmission. Capped
    at :data:`_CR_INTERVAL_MAX_S`, past which spreading the ladder further only
    delays the request a gateway is waiting to be free for.
    """
    return min(_CR_INTERVAL_MAX_S,
               max(CR_BURST_S, timeout / max_cr - CR_BURST_S))


RECORD_DIR = _ROOT / "logs" / "onair"

#: How long a stream may take past its own audio to drain, stop and close. A
#: healthy per-burst stop returns 0.09-0.20 s past the last sample.
STOP_BUDGET_S = 2.0

#: How far off its nominal rate the capture may run, over two consecutive
#: windows, before the operator is told. Two, so that the one block a first
#: output stream costs [see AudioVaraIO._warm_transmit_path] is not a warning.
PACE_TOLERANCE = .02
PACE_WINDOW_S = 1.0


class AudioWedged(RuntimeError):
    """A stream's stop or close did not come back within its budget.

    `Pa_StopStream` can deadlock inside CoreAudio: `AudioOutputUnitStop` holds the
    unit and waits on the HAL's IO context, while the IO thread holds that context
    and waits on the unit to answer PortAudio's IsRunning listener. Nothing in
    Python can interrupt it, and every later PortAudio call queues behind the same
    mutex, so from here on this process makes none.
    """


_wedged: AudioWedged | None = None


def _bounded(budget_s: float, *calls, cancelled=None, delivered=None) -> None:
    """Run `calls` in order on a helper thread, raising `AudioWedged` if they
    have not all returned within `budget_s`.

    `delivered` counts the samples the device has handed over. It is what tells
    a device that is merely slow from one that is stuck: on 2026-10-28 an FT-891
    codec back from re-enumerating ran at half rate, a 6.35 s beacon took longer
    than its 8.35 s to drain, and the line it left said CoreAudio was deadlocked.
    """
    global _wedged
    if _wedged is not None:
        raise _wedged
    failed: list[BaseException] = []
    began, before = time.monotonic(), delivered() if delivered is not None else 0

    def run() -> None:
        try:
            for call in calls:
                call()
        except BaseException as exc:
            failed.append(exc)

    worker = threading.Thread(target=run, name="portaudio", daemon=True)
    worker.start()
    if cancelled is None:
        worker.join(budget_s)
    else:
        end = time.monotonic() + budget_s
        while worker.is_alive() and time.monotonic() < end:
            if cancelled.is_set():
                budget_s = .5
                end = min(end, time.monotonic() + budget_s)
            worker.join(min(.05, max(0, end - time.monotonic())))
    if worker.is_alive():
        import sounddevice as sd
        # Its exit handler is Pa_Terminate, which aborts every open stream and
        # would hang the interpreter's exit on the same mutex.
        atexit.unregister(sd._exit_handler)
        moved = delivered() - before if delivered is not None else 0
        if moved > 0:
            share = moved / FS / (time.monotonic() - began)
            why = (f"the audio device delivered {share:.0%} of real time; "
                   f"the stream did not drain within {budget_s:.1f} s")
        else:
            why = (f"a PortAudio stream did not stop within {budget_s:.1f} s; "
                   f"CoreAudio is deadlocked in this process")
        _wedged = AudioWedged(f"AUDIO WEDGED: {why} and no stream is touched again")
        raise _wedged
    if failed:
        raise failed[0]


def exit_status(main, *args):
    """What `main(*args)` would exit with, or the end of the process if audio wedged.

    A wedged stream is never stopped, so CoreAudio's IO thread goes on calling its
    Python callback, and interpreter finalization frees what that callback runs:
    on 2026-10-28 the chat host unkeyed, reported and handed back the line, then
    died SIGSEGV in `Py_Finalize`. Once a stream has wedged, `main`'s own teardown
    and the atexit handlers (the recording's store, the unkey hook, logging's
    flush) are everything owed, and the process leaves through `os._exit` with the
    status the interpreter would have given it.
    """
    try:
        ending = SystemExit(main(*args))
    except BaseException as raised:
        if _wedged is None:
            raise
        ending = raised
    if _wedged is None:
        return ending.code
    if not isinstance(ending, SystemExit):
        sys.excepthook(type(ending), ending, ending.__traceback__)
        status = 130 if isinstance(ending, KeyboardInterrupt) else 1
    elif ending.code is None or isinstance(ending.code, int):
        status = ending.code or 0
    else:
        print(ending.code, file=sys.stderr)
        status = 1
    print(f"{_wedged}: exit {status} without interpreter teardown", file=sys.stderr)
    atexit._run_exitfuncs()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


# Shared with onair.sh: only an immediate occupancy refusal is retryable.
# An expired pounce budget spends 3; the receiver's own faults remain ordinary
# errors (1). The VARA launcher delegates sensing here after codec startup, so
# its clear evidence is fresh when the ready caller originates.
CHANNEL_BUSY = 4

# A pre-key refusal that touched real state -- rigctld unreachable, the wrong
# radio, a tune that did not take, a gateway listed at a waveform this call
# cannot use -- and never got near argparse. `refuse_usage_error` in onair.sh
# reads exit 2 as "the launcher passed kestrel a bad argument", which used to be
# every one of these too: a transient CAT dropout came back exit 2 and onair.sh
# reported it as its own bug rather than the ordinary refusal it was. Nothing
# was transmitted either way, so this is not CHANNEL_BUSY's retryable occupancy
# refusal -- just a code of its own, off the one argparse actually owns.
PREKEY_REFUSED = 7


class ChannelBusy(Exception):
    """The occupancy check found the channel taken, and nothing was keyed."""


class ChannelWaitExpired(Exception):
    """The bounded occupancy wait ended without three consecutive clear windows."""

# Listed modes a station can be called at, which is not the same question as
# which waveforms kestrel can key.
#
# The connect request names the bandwidth it asks for: the three requests share
# their ten-tone preamble and differ in the generator state behind it, and the
# BW500 one in its alphabet as well  [vara_frames, connect_request]. Against a
# real VARA over the loopback a responder in BW2750 answers the BW2300 request
# and brings the session up as BW2300, which its host reports — so a station
# listed VARA 2750 is callable at 2300, and that is the only cross-bandwidth
# answer measured. A BW2300 responder has never been offered the BW2750 request,
# and BW500 answers nobody else's.
#
# The three pairs share their symbol counts — 41 for the request, 23 for the
# response — so what a received burst is can only be resolved with the session
# bandwidth beside the count, which is what ``vara_arq._kind_for`` takes.
SPEAKABLE = {"2300": ("VARA", "VARA 2750"), "2750": ("VARA 2750",),
             "500": ("VARA 500",)}


# How far a listed centre may sit from the dial asked about and still be that
# channel, and it is bounded from both sides.
#
# TOO NARROW and the launcher's rounding falls outside it. `onair.sh` builds the
# dial with `bc`, which TRUNCATES: at the `scale=1` it once used, a centre of
# 7102.85 kHz arrived as a dial of 7101.3 rather than 7101.35, and a window of 0.05
# admitted none of it — ten of the 1144 published channels then matched no row and
# fell back to the station-wide answer, which is how KB3PCY 7107.25 (listed VARA 500
# only, and refused on its true dial) became a channel this tool would have called.
# The launcher passes the whole dial now; the two are checked against each other in
# `test_the_eligibility_guard_answers_the_same_on_the_dial_the_launcher_builds`, so
# this window does not have to be guessed against a rounding rule that has moved
# once already.
#
# TOO WIDE and it reaches the row next door, which is the failure that costs a rig
# window rather than a lookup. The tightest two channels any one station lists are
# 0.5 kHz apart, so at 0.6 a plain-VARA neighbour vouches for a VARA 500 channel and
# the refusal never happens. 0.15 admits the intended row and can never admit a
# second.
_CHANNEL_TOL_KHZ = 0.15


def unspeakable(csv_path: Path | None, call: str,
                dial_khz: float | None = None, bw: str = "2300") -> list[str]:
    """The listed VARA modes, when kestrel can call the station at none of them.

    Empty when it is reachable, and empty when the list says nothing about it —
    an absent station is unknown, not unreachable.

    Reachability is a property of the channel, not the station: K5DAT-13 listens
    at VARA 500 on 40 m and plain VARA on 20 m, and the 20 m listing must not
    vouch for a 40 m call. With ``dial_khz`` the question is asked of the listed
    channel at that frequency (the CSV publishes the centre; dial is 1500 Hz
    below), and of the whole station only when no channel there is listed.
    """
    if csv_path is None or not csv_path.exists():
        return []
    rows = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("Callsign") or "").strip().upper() != call.upper():
                continue
            centre = (row.get("Frequency") or "").replace(",", "").split()[:1]
            try:
                centre_khz = float(centre[0]) if centre else None
            except ValueError:
                centre_khz = None
            rows.append(((row.get("Mode") or "").strip(), centre_khz))
    if dial_khz is not None:
        here = [(m, c) for m, c in rows
                if c is not None and abs(c - (dial_khz + 1.5)) < _CHANNEL_TOL_KHZ]
        rows = here or rows
    listed = sorted({m for m, _ in rows if m.startswith("VARA")
                     and not m.startswith("VARA FM")})
    return [] if not listed or any(m in SPEAKABLE[bw] for m in listed) else listed


# Longest burst the gate hands over in one piece. This was set to 6 s to get the
# gateway's answer out of a gate that a pinned noise floor was holding open
# indefinitely, and that is no longer what it is for: with the floor tracked as a
# rolling quantile the real NS0A connect-response (offair/NS0A_2300, 9.79-10.77 s)
# reaches the handshake at 11.0 s with no cap at all, against 17.2 s before.
#
# 6 s cleared the 4.4 s wideband over with slop, and a single over is not what a
# gateway sends. W0LON 7101200 on 2026-09-04 keyed 10.0-11.3 s in every one of
# fifteen turnarounds across two connects, +12.8 to +14.8 dB, and this gate
# force-closed all of them mid-burst — 8 of 8 on the cleaner session, against the
# 1 in 13 the 6 s was priced at. Nothing decoded and every fragment surfaced as
# session-responder-idle. The comment above the old value had already named the
# case: back-to-back gateway overs bracket as one 11.4 s region.
#
# 12 s is the monitor's own cap, which reads those same bursts whole, and it
# clears the longest one measured with 0.7 s of slop. It is not the handshake's
# concern either way — `on_rx_stream` finds the connect-response and this gate
# arms MONITOR_HELD_MIN_S into a stream, so what moves here is the connected
# state, which is where a run of overs lives.
#
# It also un-breaks KA_DEFER_MAX_S below, which is this value: a due keepalive
# waited 6 s for the channel to go quiet and then keyed anyway, which put our
# carrier 4-5 s inside the peer's own over. Answering across a peer's tail is
# measured to deliver nothing where answering after its unkey delivers 65 of 65.
SEG_MAX_S = 12.0


class _BracketSegmenter(Segmenter):
    """The monitor's hysteresis gate, handing the handshake the whole bracket.

    :class:`vara_monitor.Segmenter` brackets a keyed region with a two-threshold
    gate over a tracked noise floor and then re-cuts that region with
    ``find_bursts``, which thresholds on the region's global maximum — so a fade
    inside one burst splits it again, exactly the split the hysteresis exists to
    ride through. Exact boundaries are what the traffic monitor wants; an
    unbroken burst is what the handshake wants. Measured here over a 225-case
    fade sweep (15 fade positions x 3 durations x 5 depths inside a synthetic
    connected-ack), counting how often the ack survives as one piece and how
    often the connect completes:

        single-threshold cut, first falling edge   99/225 whole   171/225 connect
        Segmenter as written (gate + find_bursts) 132/225 whole   165/225 connect
        the gate alone, bracket handed over       225/225 whole   225/225 connect

    The gap widened when the connected-ack was re-measured (spec 04 §4.2C): its
    four fixed symbols are at its head, so a cut that starts inside the burst
    loses it outright rather than costing a few tones.

    The force-close is inherited (see ``SEG_MAX_S``), so a continuous carrier
    still ends a burst rather than growing one without bound.

    It also opens lower than the monitor does, and that is the whole difference
    between hearing a gateway and not. Measured on the 2026-07-26 recording, where
    KB9MMT answered twice (15/15 both times, at 56.02 s and 63.98 s) while kestrel
    reported no answer: those replies sit 6.2 dB over the tracked floor and the
    band noise around them reaches 6.7 dB at its 99th percentile, so there is no
    threshold that takes one without the other. At 4.0x the handshake is handed
    nothing at all; at 2.0x it is handed 15 bursts from 134 s, and one of them
    decodes as the connect-response. That is the right trade here and the wrong
    one for the traffic log: the handshake demodulates every burst and discards
    what does not match a callsign, so a false open costs a demod, while a missed
    one costs the connection. That discard is what makes the trade safe, and it
    was measured rather than assumed — every bracket the three off-air recordings
    hand over at 2.0x, scored against 196 callsigns that are not on them, is
    9097 attempts whose best score is 0.33 against an ACCEPT of 0.8.
    """

    enter = 2.0
    exit = 1.4

    # The second way in, and the burst it exists for is the peer's turnaround
    # answer: an 11-symbol two-tone control burst 0.49 s long, which a real VARA
    # keys 2.4 dB QUIETER than everything else it sends (0.2041 against 0.2691,
    # off the bench tapes). Against the peak test's +6.02 dB over the floor that
    # burst needs +6.7 dB of channel, and it is the first read to go: measured on
    # the clean bench tape with band-limited noise added, all nine of them are
    # recognised on a clear cable, four to six of nine at +6 dB SNR, and none at
    # +4 dB.
    #
    # Nothing is wrong with the recogniser. Hand `_peer_control_burst` those same
    # nine bursts with this gate's own boundaries and no gate in front of them and
    # it takes 27 of 27 down to -3 dB SNR and 26 of 27 at -3.8 dB. THE READER IS
    # ABOUT EIGHT DECIBELS MORE SENSITIVE THAN THE GATE THAT FEEDS IT, and this
    # closes about two of those eight: all nine bursts are recognised down to
    # +5 dB with it and only on a clear cable without it.
    #
    # 1.68, which is the monitor's own calibrated figure, and LOWER IS NOT BETTER
    # HERE even though a bench tape says it is. Against 21 real off-air and
    # negative recordings 1.35 looked free — 569 brackets became 1528 for 0 frames
    # falsely named and 0 `_peer_wants_turn` accepts — because that arm counts
    # false ACCEPTS. What it does not count is a true positive lost to a bracket
    # boundary, and that is what a lower bar costs: on the 2026-07-26 KB9MMT
    # recording the gate hands over 15 bursts at 1.68 and the connect-response
    # decodes 15/15, against 45 bursts at 1.50 and 77 at 1.35 where the best any
    # of them matches is 3/15. A fake cable has no band on it, so a ladder run
    # over one cannot show this; `test_the_gateway_that_answered_on_air_reaches_
    # the_handshake` is what does.
    #
    # Lowering `enter` instead was measured against the same corpus and is the
    # wrong lever for a different reason: it opens 6 brackets on a 200 Hz carrier
    # and 34 on the PACTOR-3 phantom fixture where this opens 0 and 5, because a
    # peak test cannot tell a static crash from an over and a median over a
    # quarter-second can.
    #
    # The hold is 0.25 s rather than MONITOR_HOLD_S because a median over half a
    # second of a 0.49 s burst is never more than half burst: at 0.5 s the nine
    # control bursts survive to +4 dB, at 0.25 s to +2 dB.
    #
    # It arms MONITOR_HELD_MIN_S (7.5 s) into a stream, so it is not what finds a
    # connect-response — `on_rx_stream` owns that and needs no gate. What it is
    # for is the connected state, which is minutes long.
    held = 1.68
    hold_s = 0.25

    def __init__(self):
        super().__init__(max_s=SEG_MAX_S)

    def _crop(self, chunk, base):
        """The whole bracket, uncut — see the class docstring.

        It carries the gate's pre-roll ahead of the burst and its hangover
        behind, and both earn their keep: the pre-roll is the head of a burst the
        gate opened late on, and the hangover is the trailing window
        ``demod_tones`` needs for the last symbol of a burst that ends at the
        bracket's edge. Naming a burst by dividing this length is what read every
        32-symbol frame as 37; that belongs to the reader, and
        :meth:`VaraStationHandshake.on_rx_audio` does it by recognition instead.
        """
        return [(base, chunk)]

    def _close(self, out, *, sustained: bool = False) -> None:
        """A force-closed bracket that never stood clear of the floor supersedes it.

        The floor is a quantile over SEG_FLOOR_S of between-bracket frames, so it
        cannot follow a STEP in what the receiver delivers, and it stops following
        anything at all while a bracket is open — nothing inside one feeds it. Put
        those together and the gate latches: a step up opens a bracket on band
        noise, the frozen floor keeps it open to ``max_s``, and the frames the
        force-close contributes are outnumbered in a full window by the band that
        no longer exists, so the next frame opens the next bracket.

        MEASURED on `logs/onair/20260911T043103Z-W9SSJ-K5FIT.wav`, where the
        receiver's own level stepped +3.5 dB broadband across our third
        connect-request (-17.5 dBFS before 20.80 s, -13.7 after, no carrier and no
        band edges in it — `analysis/k5fit-burst/spectra.py`). The floor stood at
        -18.1, the gate opened at -12.1, and 35.72 s of empty 40 m band came back
        as two 12.011 s brackets and twelve minimum-length ones, none of which the
        operator could hear. Every one of them deferred the next request.

        What separates that from the thing this gate exists to protect — a run of
        gateway overs long enough to force-close — is not the duration, which is
        `max_s` for both. It is that the band's own frames never reach the level
        that opens the gate: those two brackets ran +3.6 and +4.0 dB over the
        floor against the +6.0 the peak test asks for, where a VARA over's median
        frame sits 13 dB over its band and W0LON's measured +12.8 to +14.8. So a
        bracket whose MEDIAN never reached its own entry threshold was the
        channel, and the history from before it describes a channel that has been
        superseded rather than one to average with — the reasoning
        :meth:`Segmenter._disown_mute` already applies to a mute. The in-band
        history goes with it, and refills; there is no level in it this can
        re-test. Replayed over that recording, 14 brackets become 1.
        """
        if (sustained and self.brms and self.nf
                and float(np.median(self.brms)) <= self.nf * self.enter):
            self.hist.clear()          # super() refills it from this bracket
            self.ihist.clear()
        super()._close(out, sustained=sustained)

    def restart(self) -> None:
        """Forget the audio in flight, keep the noise floor.

        Called after our own transmission, where what is buffered is the rig
        monitoring itself. The floor is a property of the channel rather than of
        the burst, so keeping it leaves the gate armed for a reply that arrives
        before the first quiet frame."""
        self.buf = np.zeros(0)
        self.t0 = self.pos = 0
        self.inb, self.acc, self.brms, self.ring, self.quiet = False, [], [], [], 0
        self.iheld.clear()      # audio in flight, like the pre-roll ring


class Recorder:
    """Every received sample, streamed to a WAV as it arrives.

    An attempt that was not recorded cannot be diagnosed afterwards, and the only
    reason the 2026-07-26 session could be — it turned out to hold two answers from
    KB9MMT that this software walked past — is that the operator happened to have a
    recorder running by hand. So it is on by default and it is not optional
    equipment.

    Written from its own thread, off a queue, in int16. The PortAudio callback must
    not block on a file, and a session's audio must not be accumulated in memory to
    be written at the end: 150 s of 48 kHz float is 29 MB and a session is several
    of those, on a machine that has already been taken down by memory pressure.
    """

    def __init__(self, path: Path):
        self.path = path
        self.store_log = path.with_suffix(".store.log")
        self.samples = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._wav = wave.open(str(path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(FS)
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._enqueue_lock = threading.Lock()
        self._queued_samples = 0
        self._closed = False
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        with self._enqueue_lock:
            self._q.put(frame)
            self._queued_samples += len(frame)

    @property
    def position(self):
        with self._enqueue_lock:
            return self._queued_samples

    def _drain(self) -> None:
        while (frame := self._q.get()) is not None:
            if callable(frame):
                # A contact cutout runs behind all already queued PCM. Flush the
                # WAV header and bytes before its reader opens the ongoing file.
                self._wav._file.flush()
                frame()
                continue
            pcm = np.clip(np.asarray(frame, np.float32), -1.0, 1.0) * 32767.0
            self._wav.writeframes(pcm.astype("<i2").tobytes())
            self.samples += len(frame)

    def after_pending(self, action):
        """Schedule file work off the radio/audio threads; failures reach the caller."""
        from concurrent.futures import Future
        result = Future()
        def run():
            try:
                result.set_result(action())
            except Exception as error:
                result.set_exception(error)
        with self._enqueue_lock:
            if self._closed:
                raise RuntimeError('Recorder is closed')
            self._q.put(run)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._thread.join(timeout=10)
        self._wav.close()
        store_later(self.path, self.store_log)


def recording_path(directory: Path, mycall: str, gateway: str) -> Path:
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(directory) / f"{stamp}-{mycall.upper()}-{gateway.upper()}.wav"


PROGRESS_ENV = "KESTREL_PROGRESS_FILE"


class ProgressWatch:
    """Forward progress, on a side channel a launcher can supervise.

    The file named by ``KESTREL_PROGRESS_FILE`` gets one line per milestone and
    nothing else. Silence in it is the finding: an idle, a repeated control, a
    retry and a payload already delivered all leave it untouched, so a launcher
    holding a monotonic clock over this file can end an arm that is keying on a
    cadence and moving nothing — which on 2026-09-11 took a human 78 s and an
    externally approved signal to do  in the K5FIT BW500 on-air recording.

    ``inflight`` is not progress. It says a frame is on the air or arriving and
    when it should be over, so the launcher can let it finish rather than cut it
    mid-burst. With no file named nothing here writes or costs anything.
    """

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None
        self._seen = dict(connected=0, offer=0, outbound_acked=0, host_bytes=0,
                          mail_stage="", mail_sent=0, mail_done=0)
        self._last_payload = b""

    def emit(self, verb: str, kind: str, **fields) -> None:
        if self.path is None:
            return
        line = " ".join([verb, kind, f"mono={time.monotonic():.6f}",
                         *(f"{k}={v}" for k, v in fields.items())])
        try:
            with self.path.open("a") as stream:
                stream.write(line + "\n")
        except OSError:
            # A side channel never ends an attempt. The launcher reads the
            # silence as a stall and stops the arm, which is the safe direction.
            self.path = None

    def _advance(self, kind: str, value) -> None:
        if value != self._seen[kind]:
            self._seen[kind] = value
            self.emit("progress", kind, value=value)

    def link(self, hs) -> None:
        """Milestones only the handshake knows: the offer, the link, an over of
        ours the peer acknowledged.

        Acknowledged, not keyed: ``_over`` counts DATA overs that went out and a
        retry re-keys the same one, so the difference between it and the frame
        still outstanding rises only when the peer retires a block.

        Read off whatever the handshake keeps, and nothing demanded of it: a
        stand-in that models the link and not the delivery reports the link.
        """
        if hs.state is VaraState.CONNECTED:
            self._advance("connected", 1)
        if getattr(hs, "_linksetup_tx", 0):
            self._advance("offer", 1)
        self._advance("outbound_acked", getattr(hs, "_over", 0)
                      - (getattr(hs, "_tx_pending", None) is not None))

    def mail(self, session) -> None:
        if session is None:
            return
        self._advance("mail_stage", session.stage.replace(" ", "-"))
        self._advance("mail_sent", len(session.sent_mids))
        self._advance("mail_done", int(session.done))

    def delivered(self, payload: bytes) -> None:
        """Host bytes off the link — new ones. A block delivered twice is the
        same block, and a link repeating itself is not moving."""
        if payload and payload != self._last_payload:
            self._last_payload = bytes(payload)
            self._advance("host_bytes", self._seen["host_bytes"] + len(payload))

    def inflight(self, kind: str, seconds: float) -> None:
        self.emit("inflight", kind, ends_in=f"{max(0.0, seconds):.3f}")

    def inflight_done(self, kind: str) -> None:
        self.emit("inflight-done", kind)


# The watch a transport that never named a file gets — including every double a
# test builds, which owes this module no attribute it did not ask for.
NO_PROGRESS = ProgressWatch(None)


def progress_watch() -> ProgressWatch:
    return ProgressWatch(os.environ.get(PROGRESS_ENV))


class _MailSeam:
    """The Winlink mail client's two edges on a live transport: the connected
    edge is `link_up`, and payload decoded off the peer's DATA overs feeds the
    session in arrival order. With no client attached a delivery falls through
    to `VaraIO.data`'s unrouted report — never consumed in silence, because
    silence is indistinguishable from a gateway that sent nothing."""

    client = None                       # a winlink MailClient, or None
    progress = NO_PROGRESS              # replaced per run; disabled by default

    def connected(self, caller: str, called: str, bw: str) -> None:
        self.connected_info = (caller, called, bw)
        if self.client is not None:
            self.client.link_up()

    def data(self, payload: bytes) -> None:
        self.progress.delivered(payload)
        if self.client is None:
            super().data(payload)
        else:
            self.client.on_link_data(bytes(payload))


class AudioVaraIO(_MailSeam, VaraIO):
    """Half-duplex audio transport for the handshake.

    TX bursts are played into ``tx_device`` (the interface fills the device's
    channel count, signal on ch0+ch1 — a mono stream is inaudible to a Wine VARA
    on a multi-channel virtual device). RX audio from ``rx_device`` is recorded
    continuously; :meth:`next_rx_burst` hands the driver the next keyed burst.
    A ``rig``, if given, keys around every TX; without one nothing is keyed and
    the audio goes wherever ``tx_device`` points (the Wine loopback).

    ``record`` is a WAV path that every received sample is written to, ours
    included — the rig monitors its own transmit into the receive codec, so the
    file holds both sides of the attempt.

    ``offset_hz`` is how far the rig's dial sits from the dial the modem works: the
    modem hears its channel moved back to 1500 Hz and keys it where the channel
    sits. The recording and every ``taps`` callable get the audio as received.
    """

    def __init__(self, tx_device: str, rx_device: str,
                 amplitude: float = levels.TX_DRIVE,
                 rig: Rig | None = None, record: Path | None = None,
                 tx_tail: float = TX_IDLE_HOLD_S, warm: bool = False,
                 trace_timing: bool = False, offset_hz: int = 0):
        import sounddevice as sd
        self.tx_device = tx_device
        self.rx_device = rx_device
        self.amp = amplitude
        self.tx_tail = tx_tail
        self.rig = rig
        self._warm = warm
        from hfmodem.core.tx_events import TxEvents
        self._tx_events = TxEvents("VARA")
        self._tx_event_seq = 0
        self._tx_metadata = {}
        self._tx_key_confirmed = False
        self._tx_key_sample = None
        self.bursts_sent = 0               # played under a key-up the rig confirmed
        self.burst_level: tuple[float, float] | None = None   # (sounding RMS, peak drive)
        self._trace_timing_enabled = trace_timing
        self._timing_events = []
        self._timing_key_active = False
        self._timing_key_seq = 0
        self._timing_tx_seq = 0
        self.tx_channels = int(sd.query_devices(tx_device, "output")["max_output_channels"])
        #: Whether our own transmission comes back through the capture. It does
        #: when the rig monitors one codec both ways and it cannot when the two
        #: devices are different, which is what sizes the echo guard and what
        #: decides whether the output stream may be held  [see tx, start].
        self._monitored = tx_device == rx_device
        self._buf: list[np.ndarray] = []   # unconsumed RX frames (front-compacted)
        self._consumed = 0                 # RX samples already scanned, into _buf
        self._rx_generation = 0           # advances when TX invalidates pending RX
        self._lock = threading.Lock()      # _buf is shared with the recorder thread
        self._seg = _BracketSegmenter()    # scanned audio -> keyed bursts
        self._band = BandGate()            # scanned audio -> BUSY at BW500
        self._rx_inflight = False          # a region the gate holds open, declared
        self._bursts: list[np.ndarray] = []            # closed, not yet handed over
        self._gap: np.ndarray | None = None            # audio under measurement
        self._gap_from: int | None = None              # where that audio starts, into _buf
        self._gap_lead = 0                             # of it, the part before the cursor
        self._refused = False                          # the rig would not key this burst
        self.hs = None                                 # connect() sets it; log() reads the turn
        self._run = threading.Event(); self._run.set()
        self._stream = sd.InputStream(device=rx_device, channels=1, samplerate=FS,
                                      dtype="float32", callback=self._on_rx)
        # The recording's own timebase, taken where the recording is written.
        # Until this was here no VARA capture on the record carried a reading of
        # it, on the same interpreter and the same PortAudio that cost a shrike
        # arm 293 ms with `input_overflow` false on every callback that ran.
        self.samples = 0
        self.xruns = 0
        self.lost = 0             # captured, timestamped, never handed to Python
        self.warm_lost = 0        # ...and what `_warm_transmit_path` cost, kept apart
        self._ahead: float | None = None
        self._fit: list[tuple[float, int]] = []   # (ADC instant, samples), thinned
        # The stream is opened without a `blocksize`, so PortAudio picks one and
        # the callback's own `frames` is the only place it is stated. Until the
        # first one arrives there is no loss to measure against it.
        self._blk = 128
        self._paced: tuple[float, int] | None = None   # (callback instant, samples)
        self._slow = False
        self.rec = Recorder(record) if record is not None else None
        self.connected_info = None
        self.taps = []
        self.offset_hz = offset_hz
        if offset_hz:
            from hfmodem.contact.slots import AudioShift
            self._heard_shift = AudioShift(offset_hz)

    def _on_rx(self, indata, frames, t, status):
        n0 = self.samples
        self.samples = n0 + frames
        self._paced = (time.monotonic(), self.samples)
        self._blk = frames
        if status.input_overflow or status.input_underflow:
            self.xruns += 1
        ahead = t.inputBufferAdcTime - n0 / FS
        self.lost += rates.lost_step(ahead, self._ahead, frames, FS)
        self._ahead = ahead
        # One point a second is ample over the minutes an attempt runs, and keeps
        # a list that lives as long as the session from growing into a leak.
        if not self._fit or t.inputBufferAdcTime - self._fit[-1][0] >= 1.0:
            self._fit.append((t.inputBufferAdcTime, n0))
        if self._run.is_set():
            frame = indata[:, 0].copy()
            if self.rec is not None:
                self.rec.write(frame)
            for tap in self.taps:
                tap(frame)
            if self.offset_hz:
                frame = self._heard_shift(frame)
            with self._lock:
                self._buf.append(frame)

    def clock_report(self) -> str:
        # No TX underruns to report: transmit goes out through `play_drained` on
        # a stream this object does not hold, so the field is the shared line's
        # and stays zero here rather than being dropped and taking the whole
        # reading out of the record's ranking.
        return rates.clock_report(
            samples=self.samples, lost=self.lost, xruns=self.xruns, underruns=0,
            points=self._fit, blocksize=self._blk, fs=FS)

    def _compact(self) -> None:
        """Drop whole leading frames already scanned, so _buf stays bounded to
        the unconsumed tail rather than the whole session (O(1) amortised)."""
        with self._lock:
            drop = 0
            while self._buf and drop + len(self._buf[0]) <= self._consumed:
                drop += len(self._buf.pop(0))
        self._consumed -= drop

    def start(self):
        self._stream.start()
        threading.Thread(target=self._watch_pace, name="capture-pace", daemon=True).start()
        if self._warm:
            self._warm_transmit_path()
        self._hold_transmit_path()

    def _watch_pace(self) -> None:
        """Say so, loudly, while the capture runs off its rate against `time.monotonic()`.

        A codec can come back from a USB re-enumeration clocking far from 48 kHz
        and still read LIVE on every startup check: on 2026-10-28 the FT-891's
        delivered 368640 samples in 13 s and nothing said so until a beacon's
        drain overran. A playback on the same codec runs on that clock, so this
        is the line to read before the first key-up. It warns and nothing else.
        """
        marks: list[tuple[float, int]] = []
        while self._run.is_set():
            time.sleep(PACE_WINDOW_S)
            if self._paced is None or (marks and self._paced[0] == marks[-1][0]):
                continue
            marks = (marks + [self._paced])[-3:]
            if len(marks) < 3:
                continue
            shares = [(b[1] - a[1]) / FS / (b[0] - a[0]) for a, b in zip(marks, marks[1:])]
            off = [abs(share - 1) > PACE_TOLERANCE for share in shares]
            share = (marks[-1][1] - marks[0][1]) / FS / (marks[-1][0] - marks[0][0])
            if all(off) and not self._slow:
                self._slow = True
                text = (f"!! CAPTURE RATE: {self.rx_device} is delivering {share:.0%} of "
                        f"real time ({share * FS:.0f} samples/s against {FS}); re-plug the "
                        f"radio's USB cable or power-cycle the radio to restart its USB audio")
            elif not any(off) and self._slow:
                self._slow = False
                text = f"capture rate: {self.rx_device} is back at {share:.0%} of real time"
            else:
                continue
            print(f"  {text}", flush=True)
            self._diagnostic("capture_rate", text=text, device=self.rx_device,
                             share=round(share, 4), slow=self._slow)

    def _hold_transmit_path(self) -> None:
        """Leave the transmit stream running, where the capture does not pay for it.

        `core.audio.hold_output` carries what it buys — 0.150-0.192 s of CoreAudio
        starting the device, in front of every answer, against 0.011-0.021 s once
        the stream is up — and the one condition it cannot check for itself: the
        capture stream this object holds is the other PortAudio stream that would
        contend for the device. So the transmit path is held open only when it is
        a different device, which is what two one-way cables are and what a
        station driving one codec is not.

        `TX_IDLE_HOLD_S` is what covers the difference in the drain. A held
        stream's write returns when the ring is empty rather than when the device
        has finished with it, which measured -0.011 s against the last sample on
        the cable, where a per-burst stream's `stop` returned +0.14 to +0.20 s
        past it. The key is down for `TX_IDLE_HOLD_S` past the return either way.
        """
        if self._monitored:
            return
        hold_output(self.tx_device, FS, self.tx_channels)

    def _warm_transmit_path(self) -> None:
        """Pay the transmit device's first stream open here, before the first key-up.

        `core.audio.warm_output` carries why that open is the expensive one and
        why no later one is; what is local is what it costs the *capture*.
        Opening the first output stream on the CoreAudio device the input stream
        already holds stops the input dead. Measured on this station's codec: the
        callback entry gap goes to 320 ms while `inputBufferAdcTime` advances
        15360 samples for one 4096-frame block, so 11264 samples -- 22 of the
        device's 512-frame host buffers, 235 ms -- were timestamped and never
        handed over. It is real loss and not a miscount: over the same run the
        delivered sample count falls the same 11264 short of `time.monotonic()`.

        Unpaid, it lands under the first connect-request, and every VARA attempt
        on this record then prints the identical `11264 samples ... THE STREAM IS
        SPLICED` however long it ran or however many bursts it keyed -- 22 of the
        32 readings on disk, with 11776 (one more host buffer) on the other ten.
        A splice detector that prints a constant on every session teaches an
        operator to skip the line, and the line is how genuine starvation is
        found. Paid here it costs a fifth of a second of the pre-session settle,
        the recording is unbroken from the first request onward, and what the
        capture-clock line reports afterwards is loss the session actually took.

        Only an armed session asks for it [`warm` in `main`]. The open is worth
        moving out of a keyed window, and a session that keys nothing has no
        keyed window to move it out of -- so nothing that is not about to key
        writes to the transmit device at all. What is written is silence, and it
        is written before anything is keyed: nothing reaches the band.
        """
        before, at = self.lost, self.samples
        _bounded(WARM_S + STOP_BUDGET_S,
                 lambda: warm_output(self.tx_device, self.tx_channels))
        # The stall is charged to the first block delivered after the device
        # settles, which can land either inside `warm_output` or just past it.
        deadline = time.monotonic() + 1.0
        while self.samples < at + 2 * self._blk and time.monotonic() < deadline:
            time.sleep(0.005)
        self.warm_lost, self.lost = self.lost - before, before

    def stop(self):
        # Before the stream is torn down and the lanes let go, because that is
        # the state whose thread count is the one the capture ran under.
        print(f"  {rates.host_report()}", flush=True)
        self._run.clear()
        try:
            _bounded(STOP_BUDGET_S, drop_output, self._stream.stop, self._stream.close,
                     delivered=lambda: self.samples)
        except AudioWedged as wedged:
            print(f"  {wedged}", flush=True)
        except Exception:
            pass
        if self.rec is not None:
            self.rec.close()
            print(f"  recorded {self.rec.samples / FS:.0f}s of receive audio to "
                  f"{self.rec.path}", flush=True)
            print(f"  receive audio is being stored in the background (log: {self.rec.store_log})",
                  flush=True)
        if self.warm_lost:
            print(f"  transmit warm-up: {self.warm_lost} samples "
                  f"({self.warm_lost / FS * 1e3:.0f} ms) of capture, the price of "
                  f"this device's first output stream. Charged at start, ahead of "
                  f"the first request, and not in the line below.", flush=True)
        print(f"  {self.clock_report()}", flush=True)

    def trace_timing_event(self, event: str, **fields) -> None:
        """Buffer a software timestamp, without formatting/printing while keyed.

        These are host events, not measured PTT edges or acoustic boundaries.
        Capture counts include callback accounting; buffer cursors are relative
        to the current compacted buffer, not absolute WAV sample indices.
        """
        if not getattr(self, "_trace_timing_enabled", False):
            return
        self._timing_events.append((event, time.monotonic(), {
            "key_seq": self._timing_key_seq, "tx_seq": self._timing_tx_seq,
            **fields}))

    def _flush_timing(self) -> None:
        if not getattr(self, "_trace_timing_enabled", False):
            return
        events, self._timing_events = self._timing_events, []
        lines = []
        for event, stamp, fields in events:
            def value(v):
                return f"{v:.9f}" if isinstance(v, float) else str(v)
            rest = " ".join(f"{key}={value(val)}" for key, val in fields.items())
            lines.append(f"timing event={event} mono_s={stamp:.9f} {rest}")
        if lines:
            print("\n".join(lines), flush=True)

    def _diagnostic(self, kind, **details):
        """Optional nonblocking observer; diagnostics must never interrupt radio IO."""
        callback = getattr(self, 'diagnostic', None)
        if callback is not None:
            try:
                callback(kind, **details)
            except Exception:
                pass

    def tx_metadata(self, **metadata):
        self._tx_metadata = metadata

    def _flush_tx_events(self, unkey_confirmed=None):
        journal = getattr(self, "_tx_events", None)
        if journal is not None:
            for event in journal.pending:
                event["unkey_confirmed"] = unkey_confirmed
                event["ptt_end_sample"] = getattr(self, "samples", None) if unkey_confirmed else None
            lines = journal.lines()
            if lines:
                print("\n".join(lines), flush=True)

    def unkey_confirmed(self):
        """Explicit completion fence for unlinked sends; never infer from audio end."""
        if self._timing_key_active:
            return False
        return (self.rig is None or (not self.rig.retired and not self.rig.keyed
                                     and self.rig.line_state() is False))

    # -- VaraIO ----------------------------------------------------------------
    def retuned(self):
        """Owner-only receive boundary after CAT; keep recording and ARQ debt."""
        with self._lock:
            self._buf.clear()
            self._consumed = 0
            self._rx_generation += 1
            self._seg.restart()
            self._band.restart()
            self._bursts.clear()
            self._gap = self._gap_from = None
            self._gap_lead = 0
            self._rx_inflight = False
        self.progress.inflight_done('rx')

    def key(self, on: bool) -> None:
        unwinding = sys.exc_info()[1]
        if getattr(self, "_trace_timing_enabled", False) and on:
            self._timing_key_seq += 1
        if on:
            self._tx_key_confirmed = False
            self._tx_key_sample = None
            self._timing_key_active = True
        self.trace_timing_event("key_request", on=int(on), rig=int(self.rig is not None))
        try:
            keyed = self.rig.key(on) if self.rig is not None else None
        except BaseException as error:
            self.trace_timing_event("key_error", on=int(on), error=type(error).__name__)
            # Do not add stdout work while physical key state is uncertain.
            # A later successful key(False) can flush this error as well.
            self._timing_key_active = True
            if (not on and unwinding is not None
                    and not isinstance(unwinding, Exception)):
                raise unwinding
            raise
        self.trace_timing_event("key_return", on=int(on),
                                ok="none" if keyed is None else int(bool(keyed)))
        if on:
            self._tx_key_confirmed = keyed is not None and bool(keyed)
            self._tx_key_sample = getattr(self, "samples", None)
        if on and self.rig is not None:
            # A disarmed rig intentionally refuses PTT while loopback audio still
            # plays. Armed refusals suppress only this keyed region's waveform.
            self._refused = self.rig.armed and not keyed
        if not on:
            self._timing_key_active = self.rig is not None and not keyed
            if not self._timing_key_active:
                self._flush_timing()
                self._flush_tx_events(unkey_confirmed=bool(keyed))

    def tx(self, samples: np.ndarray, drive: float | None = None, *,
           identification: bool = False, cancelled=None) -> None:
        if getattr(self, "_trace_timing_enabled", False):
            self._timing_tx_seq += 1
        self.trace_timing_event("tx_entry", capture_samples=getattr(self, "samples", None))
        outcome = "returned"
        metadata, self._tx_metadata = getattr(self, "_tx_metadata", {}), {}
        self._tx_event_seq = getattr(self, "_tx_event_seq", 0) + 1
        audio_start = audio_end = waveform_samples = None
        try:
            if self._refused:
                outcome = "key_refused"
                # A burst into a transmitter that is down is silence on the band and a
                # transmission in the log, and the attempt then grinds through every
                # remaining burst the same way. Playing nothing leaves the handshake to
                # end it on its own timeout and retry ladder.
                self.log("NOT TRANSMITTING — the key-up was refused")
                return
            if self.rig is not None and self.rig.retired:
                outcome = "retired"
                # An unkey that never confirmed leaves a transmitter that may still be on
                # the air, and the answer to that is not another burst into it. On
                # 2026-07-28 this went on sending connect-requests to a rig whose unkey
                # rigctld had swallowed.
                self.log("NOT TRANSMITTING — the transmitter has been taken down")
                return
            peak = self.amp if drive is None else drive
            if getattr(self, "offset_hz", 0):
                from hfmodem.contact.slots import shift_burst
                samples = shift_burst(samples, -self.offset_hz)
            s = levels.at_drive(samples, peak)
            waveform_samples = len(s)
            block = np.zeros((len(s), self.tx_channels), dtype=np.float32)
            block[:, 0] = s
            if self.tx_channels > 1:
                block[:, 1] = block[:, 0]
            with self._lock:
                playing_from = sum(len(b) for b in self._buf)
            audio_start = getattr(self, "samples", None)
            self.trace_timing_event("play_start", waveform_samples=len(s),
                                    capture_samples=getattr(self, "samples", None),
                                    buffer_samples=playing_from, cursor=getattr(self, "_consumed", None))
            self.progress.inflight("tx", len(s) / FS + self.tx_tail)
            self._diagnostic('tx_start', identification=identification,
                             keyed=bool(getattr(self, '_tx_key_confirmed', False)), **metadata)
            playback = {'cancelled': cancelled.is_set} if cancelled is not None else {}
            waiting = {'cancelled': cancelled} if cancelled is not None else {}
            if getattr(self, "_monitored", False):
                # One codec both ways: its capture count is the playback's clock.
                waiting['delivered'] = lambda: self.samples
            _bounded(len(block) / FS + STOP_BUDGET_S,
                     lambda: play_drained(block, FS, self.tx_device, **playback), **waiting)
            audio_end = getattr(self, "samples", None)
            if self._tx_key_confirmed:
                self.bursts_sent += 1
                if not identification:
                    self.burst_level = (levels.sounding_rms(s), peak)
            self.trace_timing_event("play_return", capture_samples=getattr(self, "samples", None))
            # The rig monitors its own transmit into the RX codec, so our burst
            # echoes into the input. Skip everything recorded through the end of our
            # TX, plus the keyed idle hold and a codec-latency margin
            # (``RX_ECHO_GUARD_S``) — so we hear the gateway, not ourselves.
            #
            # WHERE THE END OF OUR TX IS, is read out of the capture and not off the
            # clock  [see _tx_end_in_capture]. The buffer's length when the drain came
            # back stood in for it and is not it.
            #
            # The cursor is taken here, before the idle hold below, and deliberately:
            # advancing it afterwards would skip a further TX_IDLE_HOLD_S of the
            # channel, and a gateway answering into that window would arrive with its
            # head missing. What that costs instead is a little more of our own receive
            # mute reaching the gate, which is the case the floor tracker is measured on.
            #
            # AND THE GUARD IS ONLY OWED WHERE THERE IS AN ECHO TO SKIP. It is the
            # monitor's, so a transmit device that is not the receive device has no
            # burst of ours in its capture and every sample the guard steps over is
            # the peer's. That went unfelt while our answers were 1.37 s long and the
            # peer's next over came a second later; against the 0.34 s continue burst
            # the peer keys 0.06-0.10 s after our last sample [vara_arq,
            # OVER_CONTINUE_ANSWERS] and the 0.10 s guard lands inside the head of its
            # frame — 24 of 24 reference columns became 19, twice in four fetches, on
            # a clean cable with nothing else wrong.
            with self._lock:
                audio = np.concatenate(self._buf) if self._buf else np.zeros(0)
                end = _tx_end_in_capture(audio, playing_from + len(s))
                guard = (self.tx_tail + RX_ECHO_GUARD_S) if self._monitored else 0.0
                self._consumed = end + int(guard * FS)
                self._rx_generation += 1
                self._seg.restart()            # what it has half-bracketed is our own burst
                self._band.restart()
                self._bursts.clear()
                self._gap = np.zeros(0)        # measure how long the input stays dead
                self._gap_from = (_tx_last_sample(audio, end) if end < len(audio)
                                  else end)    # ...our own last sample, not the cursor
                self._gap_lead = self._consumed - self._gap_from
                # HOW LATE THE DRAIN CAME BACK, in the capture's own currency: our
                # last radiated sample is at `_gap_from` and the buffer reaches
                # `len(audio)` at the instant `play_drained` returned, so the
                # difference is the elapsed time between the two with the capture
                # path's latency cancelled out of both. A per-burst `stop` returns
                # 0.14-0.20 s past the last sample on the cable and ~0.09 s past it
                # on the rig, and the key was held TX_IDLE_HOLD_S past THAT rather
                # than past the transmission.
                late = (len(audio) - self._gap_from) / FS if self._monitored else 0.0
                self.trace_timing_event("capture_cursor", capture_samples=getattr(self, "samples", None),
                                        buffer_samples=len(audio), tx_end_in_buffer=end,
                                        cursor=getattr(self, "_consumed", None), guard_samples=int(guard * FS))
            # Hold the key idle past the audio (see TX_IDLE_HOLD_S). It happens here,
            # inside the burst, because the handshake keys down the moment this returns
            # and PTT is not ours to hold from anywhere else.
            #
            # The hold is what is OWED, not what is added: the key stays down
            # max(late, TX_IDLE_HOLD_S) past our last radiated sample, and the sleep is
            # only the part of that the drain has not already spent. It cannot clip a
            # tail however wrong the measurement is: the shortest the sleep goes is
            # zero, and zero is the unkey landing where it landed before there was a
            # hold at all — on a return `play_drained` does not make until the device
            # has played every sample.
            # What a mis-measurement can spend is the one symbol TX_IDLE_HOLD_S buys a
            # peer, and only where the capture says the drain has already spent it.
            # A transmit path that is not the receive device is unmeasurable here and
            # returns EARLY (-0.011 s, held stream), so it keeps the whole hold.
            tail = max(0.0, self.tx_tail - late)
            self.trace_timing_event("tail_start", requested_s=self.tx_tail,
                                    drain_late_s=late, sleep_s=tail)
            if cancelled is not None:
                if cancelled.wait(tail):
                    raise InterruptedError('identification cancelled')
            else:
                time.sleep(tail)
            self.trace_timing_event("tail_end", capture_samples=getattr(self, "samples", None))
        except BaseException as error:
            outcome = "error"
            self.trace_timing_event("tx_error", error=type(error).__name__)
            raise
        finally:
            self.progress.inflight_done("tx")
            journal = getattr(self, "_tx_events", None)
            result = ("refused" if outcome in ("key_refused", "retired") else
                      "error" if outcome == "error" else
                      "dry_run" if self.rig is None or not self.rig.armed else
                      "sent" if getattr(self, "_tx_key_confirmed", False) else "unconfirmed")
            self._diagnostic('tx_end', outcome=result, identification=identification, **metadata)
            if journal is not None:
                recorder = getattr(self, "rec", None)
                journal.record("completed", self._tx_event_seq, outcome=result,
                    recording=str(recorder.path) if recorder is not None else None,
                    sample_rate=FS, waveform_samples=waveform_samples,
                    ptt_start_sample=getattr(self, "_tx_key_sample", None),
                    audio_start_sample=audio_start, audio_end_sample=audio_end,
                    timing_basis="host callback counters at playback calls; not measured RF",
                    **metadata)
            self.trace_timing_event("tx_return", outcome=outcome,
                                    capture_samples=getattr(self, "samples", None))
            # Standalone no-key bench writes also produce a complete trace.
            # Normal handshake calls flush only after key(False) returns.
            if not getattr(self, "_timing_key_active", False):
                self._flush_timing()
                self._flush_tx_events()

    def _report_rx_gap(self, fresh: np.ndarray) -> None:
        """Report when the input came back after the last transmission.

        A peer answers from ``PEER_ANSWERS_FROM_S``, and the connected-ack is the
        one burst in the handshake whose recognisable part — its four fixed
        preamble symbols — comes first. An input still dead 0.4 s later therefore
        loses the ack whole, while a connect-response, whose fifteen predictable
        tones are at the back, comes through and the attempt reads as a gateway
        that answered and then went quiet. Nothing else in an attempt makes that
        visible, and it is a property of the audio path rather than of the band,
        so it is measured every time.

        Measured FROM THE UNKEY, echo guard included, which is the change that
        makes the number mean anything. The measurement used to start where the
        guard ended and count only what was still dead beyond it, so the
        2026-08-18 KC9GHZ log reported six gaps of 0.02-0.06 s against a real
        blackout of 0.12-0.16 s — small enough to read as nothing worth chasing,
        where the true figure lands on the instant a peer starts answering. The
        audio inside the guard is discarded rather than dead, but it is equally
        gone from every search that runs on this stream, and the sum of the two is
        what a peer's answer has to survive.

        Measured from OUR OWN LAST TRANSMITTED SAMPLE, which is a place in the
        audio and not a place in the bookkeeping. The clock used to start where the
        delivered sample count did when `tx` took the cursor, and that sits past
        the end of our burst by however far the codec and the rig's audio path run
        behind — so the 2026-08-23 session printed +0.10 to +0.18 s on changeovers
        whose recordings hold 0.16-0.23 s of dead input. The window now opens on
        our last sample, read off the burst's own edge  [see _tx_last_sample] rather
        than off the cursor the same edge sets to a frame, and the blind run is the
        leading one for the same reason.

        This is a level threshold, not a peer detector or proof that the band is
        back. Even the sustained run [see _LIVE_RUN_S] can catch our own transmit
        tail decaying through a muted receiver: the September 11 night KC9GHZ
        recording returned to the band around +0.17 s while this line printed
        +0.06 s. It also prints after requests that receive no reply. Use the
        protocol classifiers or native audio to establish a peer transmission.

        The part of it behind the cursor is therefore the guard and nothing else,
        which is why the line states it rather than measuring it. It used to be
        measured, against a fixed look-back, and it saturated: the three 2026-08-29
        arms printed 0.60 s to the centisecond on exactly the turnarounds where the
        reading had gone wrong, and a clipped number reads as a precise one.

        What the figure is worth knowing: 0.161-0.190 s at the tenth and ninetieth
        percentiles of the 104 changeovers on the 2026-08-23 recordings, median
        0.175, and :data:`TX_IDLE_HOLD_S` is 0.05 of it. A peer answers from
        :data:`PEER_ANSWERS_FROM_S`, so a burst opening into the turnaround loses
        its first symbol or two and no more: KB3AC-10's turn grant scored 30 of 31
        payload tones off that session's recording with 0.6 s of its head cut away.
        The printed reference reply time is a calibration constant, never an
        observation that this peer answered. A threshold not reached within the
        observation window is reported as censored rather than as an exact delay.
        """
        if self._gap is None:
            return
        self._gap = np.concatenate([self._gap, fresh])
        if len(self._gap) < self._gap_lead + GAP_WINDOW:
            return
        f = int(_MUTE_FRAME_S * FS)
        n = (self._gap_lead + GAP_WINDOW) // f
        gap, self._gap = self._gap[:n * f], None
        run = max(1, round(_LIVE_RUN_S / _MUTE_FRAME_S))
        live = _live_frames(gap, f)
        back = np.flatnonzero(np.convolve(live, np.ones(run, int), "valid") == run)
        if len(back) and not back[0]:
            print("  post-TX level check: no below-threshold interval detected "
                  "at the start; this is not a detected peer reply", flush=True)
            return
        if not len(back):
            print(f"  post-TX level check: no sustained threshold crossing within "
                  f"{n * f / FS:.2f} s (the cursor stands at "
                  f"+{self._gap_lead / FS:.2f}); this does not classify a peer reply",
                  flush=True)
            return
        else:
            # Where inside that frame the band arrived, at a tenth of it: the frame
            # the edge falls in reads live from the band it holds, and the 0.02 s
            # this is all read at would round the answer down by up to a frame.
            at = back[0] * f
            blind = (at + int(np.argmax(_live_frames(gap[at:at + f], f // 10)))
                     * (f // 10)) / FS
        print(f"  post-TX level threshold from +{blind:.2f} s after our last transmitted sample "
              f"(the cursor stands at +{self._gap_lead / FS:.2f}); reference reply "
              f"timing +{PEER_ANSWERS_FROM_S:.2f} s, observation window "
              f"{GAP_WINDOW / FS:.1f} s — this is not a detected peer reply", flush=True)

    def pending(self) -> None:
        print("  (responder PENDING)")

    def log(self, msg: str) -> None:
        if msg.startswith('link-recovery '):
            state, _, text = msg[len('link-recovery '):].partition(': ')
            self._diagnostic('link_recovery', state=state, text=text)
        if msg.startswith('rx '):
            self._diagnostic('rx_protocol', text=msg[3:])
        # The traffic classifier labels every burst; a bracket the handshake could
        # not read is not a finding on a live band. The second clause here used to
        # name the disconnect-request specifically, because a 17-symbol noise
        # bracket sized as one — the handshake now declines to name any burst it
        # has not matched to a callsign, so one line covers it.
        #
        # It stops being noise the moment we are waiting on an answer we asked for.
        # KC9GHZ answered all three turn-requests of the 2026-08-22 session and the
        # gate, riding the band noise open, handed each 1.387 s frame over inside a
        # 6.016 s bracket; all three arrived here and all three were dropped, so
        # the log the operator read showed unbroken silence while the gateway was
        # transmitting. A miss on the turn is worth a line even when it is noise.
        # So is one while a caller's link setup is due: on 10-23 at 11:13Z a weak
        # station's setup went unread and the log said nothing between PENDING and
        # CANCELPENDING.
        if ("not an MFSK handshake burst" in msg and not (
                self.hs is not None and (self.hs.turn == _TURN_ASKED or (
                    self.hs.role == "responder" and self.hs.state is VaraState.CONNECTING)))):
            return
        print(f"  {msg}", flush=True)

    def channel_busy(self, secs: float, bw: str = "") -> tuple[bool, float]:
        """Listen on the RX stream we already run, and say whether the part of the
        frequency ``bw`` will fill is in use. Transmitting without looking is not
        acceptable operating practice, and this tool had no check at all — it went
        straight to keying."""
        time.sleep(secs)
        with self._lock:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(1)
        take = int(secs * FS)
        busy, _per_sec, margin = _is_busy(audio[-take:] if len(audio) > take else audio,
                                          FS, occupied_hz("vara", bw))
        return busy, margin

    def rx_level_range(self, secs: float) -> tuple[float, float]:
        """`core.busy.level_range` over the last ``secs`` of RX audio.

        Both ends matter, which is why the measurement is shared: a floor across
        the whole window means the input is not hearing the radio at all, and a
        floor reached only intermittently means something else is keying this rig.
        Neither shows up as a PortAudio status flag — the stream is perfectly
        healthy, the audio simply is not there.
        """
        with self._lock:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(1)
        take = int(secs * FS)
        return level_range(audio[-take:] if len(audio) > take else audio, FS)

    def prime_floor(self) -> None:
        """Measure the band on the listen window, before anything is keyed.

        The bracket gate opens at a multiple of a quantile over its own frame
        history, and that history is fed only from :meth:`next_rx_burst`, which
        does not run until the first connect-request has gone out. So the first
        frames the gate ever sees are the gateway's connect-response, and a
        quantile taken over a second of it sits at the level of the signal rather
        than of the band: the gate then holds at twice the response's own
        amplitude, and the connected-ack behind it never opens it. The attempt
        ends as "NOT connected" against a station that answered everything.

        Measured against a real VARA over the loopback at 20 dB SNR: without
        this, kestrel completes 2 connects in 6, and the four failures are
        exactly the four where the gateway answered the *first* request — the
        healthy case, which leaves the least band audio in the history before the
        response arrives. The floor recovers on its own about a second after the
        ack has gone by, which is why the attempts that needed a second request
        all succeeded and why this reads as intermittent rather than broken.
        """
        with self._lock:
            audio = np.concatenate(self._buf) if self._buf else np.zeros(0)
        fresh = audio[self._consumed:]
        if len(fresh):
            self._consumed += len(fresh)
            self._seg.push(fresh)      # brackets discarded: nothing is expected yet
            self._band.push(fresh)
            self._bursts.clear()
            self._compact()

    # -- RX burst segmentation -------------------------------------------------
    @property
    def receiving(self) -> bool:
        """Whether the gate has a region open, or the newest audio would open one.

        The gate opens at 2.0x the tracked floor because a gateway's over can sit
        4 dB over the band noise (see :class:`_BracketSegmenter`), so this says
        "somebody may be keying" and not "somebody is". It is read where the cost
        of the two errors is that way round: waiting out a static crash costs a
        bounded delay, keying into the peer's over costs the over.

        AS OF NOW, and not as of the last audio scanned. Scanning happens in
        :meth:`next_rx_burst`, and one pass of it can hold for as long as the host
        takes: the mail client's answer to a gateway's greeting runs 1.4 s, so an
        ask composed inside that call read a channel that had been quiet when it
        was last looked at. Across 32 bench fetches 26 of 182 turn-requests were
        keyed on top of a peer that was already transmitting. So the buffer's
        newest frame is put to the gate's own law here, against the gate's own
        tracked floor, and nothing is consumed by the asking.
        """
        if self._seg.inb:
            return True
        if self._seg.nf is None:
            return False
        with self._lock:
            n, tail = 0, []
            for b in reversed(self._buf):
                tail.append(b)
                n += len(b)
                if n >= SEG_FRAME:
                    break
            if n < SEG_FRAME:
                return False
            frame = np.concatenate(tail[::-1])[-SEG_FRAME:]
        return float(np.sqrt((frame ** 2).mean())) > self._seg.nf * self._seg.enter

    def channel_active(self, bw: str) -> bool:
        """What BUSY reports: the BW500 passband gate at 500 (`busy_gate`), the
        bracket gate at the wider bandwidths, whose stock BUSY is not measured."""
        return self._band.active if str(bw) == "500" else self.receiving

    def busy_detail(self) -> str:
        """What the gate read at a BUSY edge, so a live session's BUSY can be replayed."""
        seg = self._seg
        db = lambda v: f"{20 * np.log10(max(v, 1e-12)):.1f}" if v else "none"
        with self._lock:
            tail = np.concatenate(self._buf)[-SEG_FRAME:] if self._buf else np.zeros(0)
        frame = float(np.sqrt((tail ** 2).mean())) if len(tail) else None
        held = f"{float(np.median(seg.iheld)):.2f}" if seg.iheld else "none"
        return (f"floor {db(seg.nf)} dBFS, frame {db(frame)} dBFS, held {held}, "
                f"open {bool(seg.inb)}, {self._band.detail()}, epoch {self._rx_generation}, "
                f"sample {self.samples}")

    def next_rx_burst(self, timeout: float, hs=None):
        """Wait for the responder's next keyed burst after the last TX, or None.

        Every sample recorded since the last scan goes through the gate, which
        hands back each keyed region once it has closed — on trailing silence, or
        on the maximum burst length. A burst that fades in the middle stays one
        burst: this used to cut at the first falling edge of a single-threshold
        envelope, so one fade split the gateway's answer into pieces too short to
        decide and the attempt ended as "NOT connected", which reads exactly like
        a gateway that never answered.

        The same samples also go straight to ``hs.on_rx_stream``, ahead of the gate
        and independent of it, which is how the gateway's answer is found at all: in
        the SSB passband KB9MMT's reply sits 1.6 dB *below* the band noise, and what
        opens an energy gate on it is out-of-band splatter from an overdriven
        receive chain — a receiver that is set up properly hands the gate nothing.
        The stream search takes the connect-response from there (see
        ``VaraStationHandshake.on_rx_stream``); the brackets keep everything else.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                audio = np.concatenate(self._buf) if self._buf else np.zeros(0)
            fresh = audio[self._consumed:]
            if len(fresh):
                # The blackout measurement, and only it, is shown the audio the
                # cursor skipped: it needs the end of our own burst to have
                # anything to time from  [see _report_rx_gap].
                if self._gap_from is not None:
                    lead = audio[self._gap_from:self._consumed]
                    self._gap_from = None
                    self._report_rx_gap(np.concatenate([lead, fresh]))
                else:
                    self._report_rx_gap(fresh)
                self._consumed += len(fresh)
                generation = self._rx_generation
                if hs is not None:
                    hs.on_rx_stream(fresh)
                if generation != self._rx_generation:
                    # Stream recognition can transmit. tx() already discarded
                    # the old brackets and advanced past that transmission;
                    # feeding its pre-TX snapshot back to the gate would replay
                    # the request that was just answered.
                    self._compact()
                    return None
                self._bursts += [b for _, b in self._seg.push(fresh)]
                self._band.push(fresh)
                self._compact()
                # A region the gate still holds open is a peer keying now, and
                # SEG_MAX_S is the longest one it can be  [_BracketSegmenter].
                if bool(self._seg.inb) != self._rx_inflight:
                    self._rx_inflight = not self._rx_inflight
                    if self._rx_inflight:
                        self.progress.inflight("rx", SEG_MAX_S)
                    else:
                        self.progress.inflight_done("rx")
            if self._bursts:
                return self._bursts.pop(0)
            if time.time() >= deadline:
                return None
            # The grid a turnaround is actually decided on: the over route's own
            # scan step is finer than this, so what a peer's over waits between
            # ending and being answered is this poll and not that step.
            time.sleep(0.02)


class _DryIO(_MailSeam, VaraIO):
    """The audio transport with the audio removed: every surface `connect` and
    `mail_session` drive, no device opened, no rig, nothing keyed. What a
    ``--dry-run`` proves is the wiring from flags to loop — the same radioless
    rehearsal shrike's onair runs without ``--transmit`` — so a typo in a mail
    flag dies here instead of at the radio."""

    rig = None
    rx_device = "none (dry run)"

    def __init__(self):
        self.connected_info = None
        self.bursts_sent = 0
        self.burst_level = None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def key(self, on: bool) -> None: ...
    def tx(self, samples, drive=None, *, identification=False) -> None: ...
    def prime_floor(self) -> None: ...

    def pending(self) -> None:
        print("  (responder PENDING)")

    def log(self, msg: str) -> None:
        print(f"  {msg}", flush=True)

    def channel_busy(self, secs: float, bw: str = "") -> tuple[bool, float]:
        return False, -99.0

    def rx_level_range(self, secs: float) -> tuple[float, float]:
        return -30.0, -40.0             # alive, and not being keyed over

    def next_rx_burst(self, timeout: float, hs=None):
        time.sleep(min(timeout, 0.05))
        return None


# How long a due transmission — a keepalive, or a connect-request on its cadence
# — waits for the channel to go quiet before it keys anyway: one bracket's worth,
# since SEG_MAX_S is the longest region the gate can hold open. Past it the
# channel is the band rather than the peer's over.
KA_DEFER_MAX_S = SEG_MAX_S


def mail_session(hs: VaraStationHandshake, io, client, timeout: float,
                 keepalive_s: float = 10.0) -> bool:
    """Pump the held link until the B2F exchange closes or the clock runs out.

    The gateway speaks first — its SID greeting arrives as an ordinary DATA
    over, is delivered through the mail seam, and everything the session
    answers rides ``hs.send``: queued, the turn asked for, drained an over at a
    time by the turn law. Received bursts drive the handshake exactly as they
    did during the connect; with nothing PROGRESSING, the idle cadence keeps the
    link alive [spec 05 §5.5]. Returns True only after a successful exchange
    and acknowledgement of its queued output, including the final FQ.

    Progressing, and not arriving. This clock used to restart on any burst the
    gate bracketed, which is a cadence a live band can hold open indefinitely:
    replaying the 102 s after the 2026-08-20 NAK hands it 42 brackets at a longest
    gap of 8.50 s, every one of them audio the handshake could not name. Nothing
    was ever keyed into that, so `_MAX_WITHOUT_PROGRESS` — the budget that closes
    a link nobody is answering on — never spent a tick, and the session was still
    CONNECTED when the operator stopped it. It fired the other way too, and that
    is what cost the session: ten seconds after the loop started, with the modem
    having answered an over in between, it keyed a keepalive 3.7 s into the
    gateway's next over  [see ``VaraStationHandshake._key``].

    ``hs.progress`` is the handshake's own count of the link moving forward, so
    both readings come from the one definition instead of from a burst arriving.
    Due is not the same as clear, though: a keepalive that comes due while the
    channel is keyed waits for it, up to ``KA_DEFER_MAX_S``, and the deadline it
    waited past is not moved by the waiting.
    """
    deadline = time.time() + timeout
    watch = getattr(io, "progress", NO_PROGRESS)
    seen, next_ka = (hs.progress, hs.idle_keyed), time.time() + keepalive_s
    query_due_in = getattr(hs, "intermediate_query_due_in", lambda: None)
    retry_due_in, retry_due_at = getattr(hs, "data_retry_due_in", lambda: None), None
    poll_due_in = getattr(hs, "idle_poll_due_in", lambda: None)

    def unfinished():
        return not client.done or getattr(hs, "data_pending", False)

    while time.time() < deadline:
        if getattr(getattr(client, "session", None), "failure", ""):
            return False
        if hs.state is not VaraState.CONNECTED:
            # The link layer has given the session up — overs that would not
            # decode, to their budget  [vara_arq._undecoded_over].
            # Without a link this loop keys nothing, so staying in it spends the
            # rest of the mail timeout in silence and reports "out of time".
            print("  mail abandoned — the link layer closed the session",
                  flush=True)
            return False
        if not unfinished():
            return True
        if io.rig is not None and io.rig.retired:
            print("  mail abandoned — the transmitter has been taken down",
                  flush=True)
            return False
        # `hs=`, because the gateway's overs are found on the receive stream and
        # not by the gate: on 2026-08-06 KC9GHZ's greeting sat 4.1 dB over the
        # band noise, opened nothing, and this loop spun for 227 s waiting to be
        # spoken to while the answer was already in the audio. The connect loop
        # had passed the stream in from the start; this one had not, so the search
        # stopped at the moment the link came up.
        # A solicited intermediate answer and an unacknowledged over each have
        # their own measured deadline; neither may wait for the ordinary
        # ten-second keepalive cadence, nor be quantised by a one-second poll.
        poll_s = min(1.0, deadline - time.time(), next_ka - time.time())
        for delay in (query_due_in(), retry_due_in(), poll_due_in()):
            if delay is not None:
                poll_s = min(poll_s, delay)
        burst = io.next_rx_burst(timeout=max(.02, poll_s), hs=hs)
        if burst is not None:
            hs.on_rx_audio(burst)
        watch.link(hs)
        watch.mail(getattr(client, "session", None))
        if (not unfinished()
                or getattr(getattr(client, "session", None), "failure", "")):
            continue
        if (hs.progress, hs.idle_keyed) != seen:
            seen, next_ka = (hs.progress, hs.idle_keyed), time.time() + keepalive_s
        query_delay, retry_delay, poll_delay = query_due_in(), retry_due_in(), poll_due_in()
        if retry_delay is None or retry_delay > 0:
            retry_due_at = None
        elif retry_due_at is None:
            retry_due_at = time.time()
        if (query_delay is not None and query_delay <= 0
                and hs.state is VaraState.CONNECTED and unfinished()
                and not io.receiving):
            hs.query_intermediate_answer()
            seen, next_ka = (hs.progress, hs.idle_keyed), time.time() + keepalive_s
        elif (retry_due_at is not None
                and hs.state is VaraState.CONNECTED and unfinished()
                and not (io.receiving
                         and time.time() < retry_due_at + KA_DEFER_MAX_S)):
            # Recover the outstanding answer on its measured deadline. The
            # state machine solicits feedback or closes with bytes unconfirmed;
            # silence never authorizes DATA retransmission. Defer for a busy
            # channel on the same terms as a due keepalive.
            hs.retry_data_over()
            seen, next_ka = (hs.progress, hs.idle_keyed), time.time() + keepalive_s
        elif poll_delay is not None and poll_delay <= 0 and not io.receiving:
            # The turn is ours and empty: stock's poll runs on its own clock.
            hs.idle_keepalive()
            seen, next_ka = (hs.progress, hs.idle_keyed), time.time() + keepalive_s
        elif time.time() >= next_ka and not (
                io.receiving and time.time() < next_ka + KA_DEFER_MAX_S):
            hs.idle_keepalive()
            seen, next_ka = (hs.progress, hs.idle_keyed), time.time() + keepalive_s
    return (hs.state is VaraState.CONNECTED and not unfinished()
            and not getattr(getattr(client, "session", None), "failure", ""))


def attach_mail(args, hs: VaraStationHandshake, io):
    """Build the mail client on the transport, or (None, None) without flags.

    The two wires this makes — payload deliveries routed to the client through
    the transport's seam, and the session's answers queued on ``hs.send`` —
    are the whole difference between mail moving and mail lost, so they are
    made in one place a test can hold.
    """
    if not (args.mail_send or args.mail_fetch):
        return None, None
    from hfmodem.winlink import B2FSession, MailClient, load_outbound
    outbox = [load_outbound(p, args.mycall, to=args.mail_to,
                            subject=args.mail_subject)
              for p in (args.mail_send or [])]
    session = B2FSession(args.mycall, role="calling", target=args.gateway,
                         password=args.mail_password, outbox=outbox,
                         grid=config.grid(), comment=False,
                         client_sid=config.client_sid(args.mail_sid))
    mail = MailClient(session, hs.send)
    io.client = mail
    print(f"mail: announcing as {session.sid}", flush=True)
    return mail, session


def disconnect_session(hs: VaraStationHandshake, io, tries: int = 3,
                       wait: float = 5.0) -> bool:
    """Drive the close [spec 05 §5.6] and say whether it completed.

    `hs.disconnect` keys the 32-symbol disconnect burst. An acknowledgement can
    be lost to the channel, so it is re-keyed up to ``tries`` times — and a close
    that goes unanswered is reported False, never claimed.

    THE WAIT AND THE RETRIES ARE ON A HANDSHAKE THAT MAY NOT EXIST. The three-burst
    close this was written against was read off a loopback tape, and on 2026-08-26
    two stock 4.9.0 instances on one cable produced nothing of the kind: the closing
    station keys the burst ONCE, immediately ahead of its own CW ident, and the only
    thing that follows is the peer's ident. So `False` here may mean the peer closed
    exactly as a 4.9.0 does and this function had nothing to hear. The loop is left
    as it stands because re-keying an unanswered close is harmless and the verdict is
    read by `onair_session`; what must not happen is reading `False` as a peer that
    ignored us.
    """
    for _ in range(tries):
        if io.rig is not None and io.rig.retired:
            return False
        hs.disconnect()                     # keys (or re-keys) the request
        deadline = time.time() + wait
        while time.time() < deadline and hs.state == VaraState.DISCONNECTING:
            burst = io.next_rx_burst(timeout=1.0)
            if burst is not None:
                hs.on_rx_audio(burst)
        if hs.state == VaraState.DISCONNECTED:
            return True
    return False


def wait_clear_channel(io, bw: str, listen_first: float, force: bool,
                       pounce_wait: float = 0.0) -> bool:
    """Judge fresh audio with the caller ready; never reuse a shell clear verdict.

    A pounce retains the shell gate's three consecutive three-second windows,
    restarting the clear count when an occupant returns. The one wait budget
    covers the whole sequence; it does not restart on a disagreeing last window.
    The already-open session recorder retains every window, including refusals.
    """
    if listen_first <= 0 and pounce_wait <= 0:
        return True
    budget = pounce_wait if not force else 0.0
    seconds = POUNCE_WINDOW_S if budget else listen_first
    if seconds <= 0:  # Direct callers can disable ordinary sensing, not a pounce.
        seconds = POUNCE_WINDOW_S
    started = time.monotonic()
    clear = 0
    print(f"  caller ready; occupancy check starts at "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
          f"(window {seconds:g}s, wait budget {budget:g}s)", flush=True)
    while True:
        # Do not accept a shortened window just because the budget is ending.
        if budget and time.monotonic() - started + seconds > budget:
            print(f"  no clear channel after {time.monotonic() - started:.3f}s "
                  f"of the {budget:g}s wait budget; insufficient time for "
                  f"another {seconds:g}s window", flush=True)
            raise ChannelWaitExpired
        busy, margin = io.channel_busy(seconds, bw)
        fault = receiver_fault(*io.rx_level_range(seconds),
                               seconds=seconds, device=io.rx_device)
        if fault is not None:
            print(f"  {fault.reason}", flush=True)
            return False
        clear = 0 if busy else clear + 1
        print(f"  channel sense: {margin:+.1f} dB against the occupancy thresholds "
              f"({'OCCUPIED' if busy else 'clear'}); "
              f"elapsed {time.monotonic() - started:.3f}s, "
              f"clear windows {clear}/{3 if budget else 1}", flush=True)
        # Consume each window while waiting, so floor initialization cannot
        # accumulate a whole pounce budget of DSP work after the clear decision.
        io.prime_floor()
        if budget and time.monotonic() - started > budget:
            print(f"  occupancy wait exceeded its {budget:g}s budget", flush=True)
            raise ChannelWaitExpired
        if force or (not busy and (not budget or clear >= 3)):
            if busy:
                print("  --force: calling anyway over an occupant", flush=True)
            print("  occupancy decision complete; caller ready to originate", flush=True)
            trace = getattr(io, "trace_timing_event", None)
            if trace is not None:
                trace("occupancy_ready", forced=int(force), window_s=seconds,
                      clear_windows=clear)
            return True
        if not budget:
            print("  channel in use — NOT transmitting (pass --force to override)",
                  flush=True)
            raise ChannelBusy


def connect(gateway: str, mycall: str, bw: str, io: AudioVaraIO,
            timeout: float = 60.0, cr_interval: float | None = None,
            max_cr: int = 8, calls: list = None, listen_first: float = 8.0,
            force: bool = False, ls_interval: float = 8.0,
            hs: VaraStationHandshake | None = None,
            pounce_wait: float = 0.0, admission=None) -> bool:
    """Originate a connect, re-sending the CR until the window or the budget ends.

    A single connect-request is easily missed (QSB, timing, a briefly-busy
    gateway), so like real VARA we repeat it every ``cr_interval`` seconds — by
    default :func:`cr_cadence` of the window — up to ``max_cr`` times until a
    connect-response comes back, then run the link-setup.
    The link-setup itself is repeated every ``ls_interval`` seconds while no
    connected-ack comes back — modest, because the over is 4.4 s of transmit and
    the whole operating window is fifteen minutes.

    Spending that budget with the gateway silent is not the end of the attempt: it
    goes back to the connect-request exchange and calls again, which is the burst
    a fade lets through  [see ``vara_arq._CONNECT_MAX_RESTARTS``, which bounds it].
    ``max_cr`` covers every request the attempt puts on the air, restarts included,
    and the attempt ends the moment neither ladder has anything left rather than
    listening out the rest of its window.

    Every received burst is run through the traffic classifier (the same demod
    that drives the handshake) and logged — so a connect is also a live monitor
    of everything else on the channel.
    """
    hs = hs or VaraStationHandshake([mycall], io, bw=bw, panel=calls or ())
    if cr_interval is None:
        cr_interval = cr_cadence(timeout, max_cr)
    watch = getattr(io, "progress", NO_PROGRESS)
    io.hs = hs
    io.start()
    if admission is not None:
        admission.opened()
    time.sleep(0.3)
    mon_calls = list(dict.fromkeys([gateway.upper(), mycall.upper()] + (calls or [])))
    session = Session()
    if not wait_clear_channel(io, bw, listen_first, force, pounce_wait):
        return False
    print(f"originating: {mycall} -> {gateway} (BW{bw}) — up to {max_cr} requests "
          f"of {CR_BURST_S:.2f} s, {cr_interval:.1f} s apart", flush=True)
    cr_count = 1 if hs.originate(gateway) else 0         # CR #1, if it went out
    unread = False
    next_cr = time.time() + cr_interval
    ls_count, next_ls = 0, 0.0
    t0 = time.time()
    deadline = t0 + timeout
    while hs.state != VaraState.CONNECTED and time.time() < deadline:
        watch.link(hs)
        if io.rig is not None and io.rig.retired:
            print("  attempt abandoned — the transmitter has been taken down",
                  flush=True)
            break
        # Drain reception before a timer can erase it by transmitting. A closed
        # bracket may be followed by more queued audio, and an open region may
        # still be the peer's reply. Waiting never extends the attempt deadline
        # or spends/replenishes either transmit budget.
        # Stop polling when the next transmission falls due. A received burst
        # can shift the poll's phase; a fixed 1.5 s wait then delays a clear
        # channel unnecessarily. The occupancy defer below still applies.
        due = next_cr if hs.step == _I_CR_SENT else next_ls
        now = time.time()
        burst = io.next_rx_burst(
            timeout=max(0.02, min(1.5, deadline - now,
                                  due - now if due else 0.0)), hs=hs)
        if burst is not None:
            try:
                emit(time.time() - t0,
                     classify(burst, mon_calls, decode_wideband=False), session)
            except Exception:
                pass
            hs.on_rx_audio(burst)
        if hs.state == VaraState.CONNECTED or time.time() >= deadline:
            break
        if hs.answer_retry:
            # The gateway has just said it heard a request addressed to it and
            # could not read it, which is a verdict on our transmitter and an
            # invitation: it is listening now  [vara_arq._ANSWER_LATTICE].
            hs.answer_retry = False
            if cr_count < max_cr:
                unread = True
                print(f"  {gateway.upper()} could not read that request — re-keying "
                      f"at once rather than waiting the cadence", flush=True)
                next_cr = min(next_cr, time.time() + _CR_AFTER_ANSWER_S)
        # A burst may be the answer and an open gate may be the peer composing
        # one, so a due transmission waits for the channel — and, as in the
        # keepalive cadence, not for ever. Past ``KA_DEFER_MAX_S`` the channel is
        # the band rather than the peer's over, and nothing bounded this side of
        # it: calling K5FIT on 2026-09-11 the gate latched open on a receiver
        # level step [see `_BracketSegmenter._close`] and requests 4 and 5 waited
        # 35.72 s on a channel the operator heard nothing on, against a 3.0 s
        # cadence inside a 65 s attempt. The clock is the deferred transmission's
        # own due time, so waiting for the channel does not move it.
        due = next_cr if hs.step == _I_CR_SENT else next_ls
        if (burst is not None or io.receiving) and (
                not due or time.time() < due + KA_DEFER_MAX_S):
            continue
        # no connect-response yet -> resend the CR on cadence
        if hs.step == _I_CR_SENT and cr_count < max_cr and time.time() >= next_cr:
            # The budget counts requests that reached the air. A rig refusing the
            # key-up spends the interval — the cadence is what keeps us off a
            # transmitter we cannot key — but not one of the attempts.
            if hs.originate(gateway, retry=True):
                cr_count += 1
                why = ("answered but could not read us" if unread else
                       "no connect offer" if hs.answers else "no answer")
                print(f"  {why} — resending CR ({cr_count}/{max_cr}, "
                      f"{_CR_FORMS[hs.cr_retry_form]})", flush=True)
            unread = False
            next_cr = time.time() + cr_interval
        elif hs.step == _I_LINKSETUP_SENT:
            if hs._linksetup_tx != ls_count:          # the handshake sent one itself
                ls_count, next_ls = hs._linksetup_tx, time.time() + ls_interval
            elif time.time() >= next_ls:
                if ls_count < hs.max_link_setups:
                    print(f"  no connected-ack — resending link-setup "
                          f"({ls_count + 1}/{hs.max_link_setups})", flush=True)
                    hs.resend_link_setup()
                    ls_count, next_ls = hs._linksetup_tx, time.time() + ls_interval
                # Step 4 spent with the gateway silent: the fade that eats a 4.36 s
                # over is one the 1.77 s request rides through, so the attempt calls
                # again instead of dying here. The CR train above keys it and counts
                # it, which is what keeps one budget over every request on the air.
                elif cr_count < max_cr and hs.link_setup_unanswered():
                    print(f"  no connected-ack for {ls_count} link-setups — calling "
                          f"again", flush=True)
                    ls_count, next_cr = 0, 0.0
                else:
                    print("  no connected-ack, and nothing left to call with — "
                          "abandoning the attempt", flush=True)
                    break
    watch.link(hs)
    return hs.state == VaraState.CONNECTED


def identify(io, mycall: str, bw: str, say=partial(print, flush=True), *, cancelled=None) -> None:
    """Our callsign in CW straight after the session, whatever VARA's frames
    carried: the tail of our own transmission, so it senses no channel."""
    with cwid.confirming(mycall, say):
        if io.burst_level is None:
            raise ValueError("no completed data burst for identification level")
        audio = cwid.audio(mycall.upper(), fs=FS, tone_hz=sum(occupied_hz("vara", bw)) / 2)
        drive = levels.drive_for_rms(audio, *io.burst_level)
        before = io.bursts_sent
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError('identification cancelled')
        io.key(True)
        try:
            io.tx(audio, drive, identification=True,
                  **({'cancelled': cancelled} if cancelled is not None else {}))
        finally:
            unwinding = sys.exc_info()[1]
            try:
                io.key(False)
            except BaseException:
                if unwinding is not None and not isinstance(unwinding, Exception):
                    raise unwinding
                raise
        if io.bursts_sent == before:
            raise RuntimeError("the transmitter refused identification")


def close_verdict(closed: bool, gateway: str) -> str:
    """Report local closure without inventing a remote acknowledgement.

    `disconnect()` closes locally once its burst is transmitted. Stock VARA
    does not acknowledge that burst, so True is not evidence of peer receipt.
    Keep the marker consumed by onair_session, and state its limited meaning.
    """
    if closed:
        return (f"DISCONNECTED: clean close — local session ended; close sent to "
                f"{gateway.upper()}, peer receipt unconfirmed")
    return (f"NOT DISCONNECTED: local close to {gateway.upper()} did not complete; "
            "peer state is unknown and may persist until its inactivity timeout.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", required=True, help="called (gateway) callsign")
    ap.add_argument("--mycall", required=True, help="our digital (caller) callsign")
    ap.add_argument("--idcall", help="CW station callsign, independent of --mycall; "
                    "default: IDCALL, [station] idcall, then --mycall")
    ap.add_argument("--bw", default="2300", choices=["2300", "2750", "500"],
                    help="session bandwidth. A VARA 2750 listing answers 2300 as "
                         "well as its own; VARA and VARA 500 listings answer only "
                         "theirs")
    ap.add_argument("--tx-device", default="BlackHole 16ch")
    ap.add_argument("--rx-device", default="BlackHole 64ch")
    ap.add_argument("--rigctld", help="host:port of the rigctld that tunes the "
                                      "radio; it never keys (omit for BlackHole)")
    ap.add_argument("--ptt-device", metavar="DEV",
                    help="required with --arm: the serial line whose RTS is the PTT")
    ap.add_argument("--line-ptt", action="store_true",
                    help="required with --arm: key by driving that line. rigctld "
                         "only tunes, which it does before there is any RF about")
    ap.add_argument("--freq", type=float,
                    help="dial frequency in kHz to tune before connecting, e.g. 7107.0 "
                         f"(asks for PKTUSB / {FILTER_HZ} Hz too, and refuses a "
                         "filter narrower than the emission; needs --rigctld)")
    # A PEAK, and the waveform is peak-normalised, so an OFDM over averages far
    # below a handshake tone at the same figure. That is why a steady carrier at 0.6
    # reaches the rig's ceiling while an over at 0.5 does not: the number bounds
    # the peak, and the peak-to-average of the two signals is nothing alike.
    #
    # MEASURED at the station, 2026-08-09, against the operator's wattmeter:
    # --amplitude 0.8 peaks the FT-891 at its full 50 W setting with the ALC
    # still. That is the figure to raise the whole station to when a meter is on
    # it -- one line in the station file -- rather than the figure for VARA
    # alone, which is how this modem came to go out 2.5 dB above the other three.
    # Raise it only against a meter, and stop where the ALC starts to move: IMD
    # splatter costs more than the decibel buys.
    ap.add_argument("--amplitude", type=float, default=None,
                    help="soundcard peak for TX (0..1). Defaults to [audio] "
                         f"tx_drive from the station file named by "
                         f"{config.STATION_ENV}, else {levels.TX_DRIVE}")
    ap.add_argument("--trace-timing", action="store_true",
                    help="log monotonic software key/playback/drain/tail events; "
                         "key requests are not measured physical PTT edges")
    ap.add_argument("--tx-tail", type=float, default=TX_IDLE_HOLD_S,
                    help="seconds PTT is held past the last transmitted sample; "
                         "see TX_IDLE_HOLD_S, and the 'input live from' lines it "
                         "moves. The drain is not in here — play_drained waits on "
                         "the device for that")
    ap.add_argument("--arm", action="store_true",
                    help="permit keying the transmitter; without it nothing is "
                         "keyed. Needs --expect-model (never key a radio you "
                         "cannot name) and --ptt-device (never key one you "
                         "cannot unkey)")
    ap.add_argument("--cwid-optional", action="store_true",
                    help="skip the closing CW ID only when digital and CW callsigns are equal")
    ap.add_argument("--expect-model", default=None,
                    help="refuse to key unless the rig identifies as this model")
    ap.add_argument("--gateways", type=Path, default=GATEWAY_LIST,
                    help="gateway CSV used to name stations heard on the air, and "
                         "to check the target is listed at a bandwidth kestrel "
                         f"can call (default {GATEWAY_LIST.name})")
    ap.add_argument("--listen-first", type=float, default=8.0,
                    help="seconds to listen before the first CR (0 disables)")
    ap.add_argument("--pounce-wait", type=float, default=0.0,
                    help="maximum seconds to await three consecutive clear 3s "
                         "windows after audio startup; --force uses one normal listen")
    ap.add_argument("--force", action="store_true",
                    help="transmit even if the channel sounds occupied, or if the "
                         "gateway list says the target's channel cannot answer")
    ap.add_argument("--record", type=Path, default=RECORD_DIR,
                    help="directory to write the attempt's receive audio into "
                         f"(default {RECORD_DIR})")
    ap.add_argument("--no-record", dest="record", action="store_const", const=None,
                    help="do not record — an attempt nobody recorded cannot be "
                         "diagnosed afterwards")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--mail-send", action="append", metavar="FILE",
                    help="Winlink mail: a rendered .b2f message, or a body file "
                         "with --mail-to (repeatable). Runs a B2F exchange over "
                         "the held link once connected")
    ap.add_argument("--no-data-retries", action="store_true",
                    help="compatibility option: missing answers now use bounded state "
                         "queries; DATA is repeated only for a decoded NAK")
    ap.add_argument("--compression", choices=("OFF", "TEXT", "FILES"), default="OFF",
                    help="link compression preference (default OFF); TEXT calls remain "
                         "OFF until its receive tables and interop checks are complete")
    ap.add_argument("--low-speed-only", action="store_true",
                    help="key the free link-setup (body[7] 0x80, ask 4) and stay at "
                         "level 4 instead of the default high speed capable "
                         "gear-shift: levels 4 to the bandwidth's top, stepped by "
                         "the receiving station's continue frames")
    ap.add_argument("--tx-level", type=int, choices=(1, 2, 3, 4),
                    help="BW2750: outgoing DATA target host speed level. Each "
                         "delivery enters at level 4 and steps down one record "
                         "per full frame until it holds here; link setup and "
                         "receive speed are unaffected")
    ap.add_argument("--probe-intermediate-query", action="store_true",
                    help="compatibility option: intermediate-answer queries are now "
                         "enabled by default at BW2300 and BW2750")
    ap.add_argument("--mail-fetch", action="store_true",
                    help="Winlink mail: collect whatever the gateway holds")
    ap.add_argument("--mail-to", default="",
                    help="recipient for a --mail-send body file")
    ap.add_argument("--mail-subject", default="")
    ap.add_argument("--mail-password", default="",
                    help=f"secure-login answer to the gateway's ;PQ: challenge. "
                         f"Spelled out here it is in this process's argv, which "
                         f"every `ps` can read: prefer --mail-password-file or "
                         f"${config.PASSWORD_ENV}")
    ap.add_argument("--mail-password-file", default="",
                    help="read the ;PQ: answer from this file instead, so it "
                         "never reaches argv (default: "
                         f"${config.PASSWORD_ENV})")
    ap.add_argument("--mail-sid", default="",
                    help="the client type this station announces: the name and "
                         "version half of the B2F SID, no brackets and no "
                         "capability letters. Defaults to [station] client_sid "
                         f"from the station file named by {config.STATION_ENV}, "
                         f"else {CLIENT_SID}")
    ap.add_argument("--mail-out", default="logs/mail",
                    help="where received messages are written")
    ap.add_argument("--mail-timeout", type=float, default=600.0,
                    help="seconds for the mail exchange after the connect "
                         "(default 600)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no audio, no rig, nothing keyed: run the whole "
                         "flag-to-loop wiring radiolessly and report the stage "
                         "it reached")
    ap.add_argument("--max-cr", type=int, default=8,
                    help="connect-requests the attempt may put on the air, "
                         "restarts included (default 8)")
    ap.add_argument("--cr-retry-form", choices=tuple(_CR_FORMS), default="full",
                    help="what the second and later connect-requests key: `full` "
                         "is this station's 41-symbol request (the default), "
                         "`stock` the 32-symbol one a real VARA caller re-keys — "
                         "one preamble tone and the same 31 payload tones. Read "
                         "off the bench, and flown on 2026-09-17: all 16 arms of "
                         "that campaign that transmitted re-keyed `stock`, the "
                         "KB8AY session that reached an accepted login among "
                         "them")
    ap.add_argument("--max-link-setups", type=int, default=_LINKSETUP_MAX_TX,
                    help="link-setup transmissions per connect exchange, including "
                         "the first (default 3); also bounded by --timeout")
    ap.add_argument("--over-continue", choices=OVER_CONTINUE_ANSWERS,
                    type=_over_continue, default=None,
                    help="answer to full DATA overs during the peer's first "
                         "delivery: `short` (16 symbols) at BW2300/BW2750, "
                         "`burst` (8) at BW500 by default; `generated` (32) "
                         "remains selectable. Explicit selection overrides the "
                         "bandwidth choice. Every form is drawn from the called "
                         "station's identity; `captured` is accepted for `burst`. "
                         "See --over-continue-after for later deliveries")
    ap.add_argument("--over-continue-after", choices=OVER_CONTINUE_ANSWERS,
                    type=_over_continue, default=None,
                    help="the same choice for every delivery after the turn "
                         "has been ours and gone back; unset, `short`. A "
                         "gateway that read the generated frame through its "
                         "greeting idled at it on the first over it keyed after "
                         "the handover, five arms at three gateways on "
                         "2026-09-06, and a stock responder re-keys 1.37-1.43 s "
                         "after its unkey in that phase, inside the 1.366 s "
                         "frame. `short` and `burst` both end early and both "
                         "carried six consecutive post-handover overs at the "
                         "bench byte-exact; `short` is the lattice family a "
                         "gateway reads, and the burst as one link's copy is "
                         "what VE3KPG refused. Nothing has been keyed at a "
                         "gateway after a handover but the generated frame")
    ap.add_argument("--cr-interval", type=float, default=None,
                    help="seconds of listening between connect-requests. Derived "
                         "from --timeout and --max-cr so the ladder covers the "
                         "window rather than its front; see cr_cadence")
    args = ap.parse_args()
    try:
        identity = identification.resolve(args.mycall, args.idcall)
    except ValueError as exc:
        ap.error(str(exc))
    identity.announce(optional=args.cwid_optional)
    if args.tx_level is not None and args.bw != "2750":
        ap.error("--tx-level requires --bw 2750")
    if not np.isfinite(args.pounce_wait) or args.pounce_wait < 0:
        ap.error("--pounce-wait must be finite and nonnegative")
    if args.max_link_setups < 1:
        ap.error("--max-link-setups must be at least 1")
    args.amplitude = config.tx_drive(args.amplitude)
    args.mail_password = config.mail_password(args.mail_password, args.mail_password_file)

    if args.dry_run and args.arm:
        print("--dry-run keys nothing; it cannot be combined with --arm")
        return 2

    # The unkey of last resort — clearing RTS on the line itself, no daemon in the
    # loop — can only run if it knows the line, so --arm requires it, for the same
    # reason --expect-model is required: refusing to key beats keying with a safety
    # disabled. The gate is `core.ptt.arming_refusal` for every transmit tool alike;
    # what is local is the exit convention.
    if args.arm and (refusal := arming_refusal(args.ptt_device)):
        print(refusal)
        return 2
    if args.arm and not args.line_ptt:
        print("--arm requires --line-ptt: hamlib may tune but may not key")
        return 2

    # Everything that can be checked without hardware is checked before the rig
    # is touched, so a bad flag costs seconds and not a rig slot: a mail file
    # that will not load, or a target whose listed channel kestrel cannot call.
    if args.mail_send or args.mail_fetch:
        from hfmodem.winlink import load_outbound
        from hfmodem.winlink.message import MessageError
        for p in args.mail_send or []:
            try:
                load_outbound(p, args.mycall, to=args.mail_to,
                              subject=args.mail_subject)
            except (OSError, MessageError) as e:
                print(f"MAIL REFUSED before anything keys: --mail-send {p}: {e}")
                return PREKEY_REFUSED
    if modes := unspeakable(args.gateways, args.gateway, args.freq, args.bw):
        where = (f"at {args.freq + 1.5:.1f} kHz centre" if args.freq
                 else "on every listed channel")
        print(f"  {args.gateway.upper()} listens {where} only as "
              f"{', '.join(modes)} — this call is BW{args.bw}, and that station "
              f"answers a different waveform. It cannot answer.",
              flush=True)
        if not args.force:
            print("  NOT calling a channel that cannot answer "
                  "(pass --force if the gateway list is wrong)", flush=True)
            return PREKEY_REFUSED

    # One rig object for tuning AND keying. This used to build a disarmed Rig for
    # tuning and route PTT around it through a bare socket helper, which discarded
    # every protection the Rig already implements: the arm gate, positive model
    # identification before keying, in-band RPRT checking on each key, the retry,
    # the independent watchdog that unkeys if a transmission runs long, and the
    # atexit/SIGINT unkey that a `finally` cannot provide when the process is
    # interrupted mid-over. The transmitting tool opted out of all of it in a
    # one-line comment; that is how a stuck transmitter happens.
    from hfmodem.core.claim import ClaimError, inherited

    try:
        admission = inherited(9, launch=True) if not args.dry_run else None
    except ClaimError as exc:
        print(f"CLAIM NOT ADMITTED: {exc}", flush=True)
        return 2
    try:
        return _run(args, identity, admission=admission)
    finally:
        if admission is not None:
            admission.failed()


def _run(args, identity, *, admission=None) -> int:
    rig = None
    if args.rigctld and not args.dry_run:
        host, _, port = args.rigctld.partition(":")
        rig = Rig(f"{host}:{int(port or 4532)}", args.arm, print,
                  ptt_device=args.ptt_device, line_ptt=args.line_ptt)
        try:
            model, dial = rig.identify()
        except OSError as e:
            print(f"CANNOT REACH rigctld at {args.rigctld} ({e}) — nothing keyed. "
                  f"Start the daemon first; onair.sh does.")
            return PREKEY_REFUSED
        print(f"  rigctld: model={model!r} dial={dial}")
        if not dial.isdigit():
            # A daemon that takes the command and answers nothing (or an RPRT
            # error) is a radio the daemon cannot talk to, not a wrong radio.
            print(f"RIG NOT ANSWERING: rigctld at {args.rigctld} returned "
                  f"{dial!r} for the dial — the daemon is up but the radio is "
                  f"not answering CAT. Check the rig is on and the daemon owns "
                  f"its port; nothing keyed.")
            return PREKEY_REFUSED
        if args.arm:
            # never key a radio you cannot name — the same gate the bridge applies
            if not args.expect_model:
                print("  --arm requires --expect-model"); return PREKEY_REFUSED
            if args.expect_model.lower() not in model.lower():
                print(f"  REFUSING TO ARM: rig reports {model!r}, expected "
                      f"{args.expect_model!r} — probably the wrong radio"); return PREKEY_REFUSED
            print(f"  rig identified as {model} — arming")
        if args.freq:
            want = int(args.freq * 1000)
            try:
                rig.set_freq(want)
                rig.set_mode("PKTUSB", mode_passband(RIGS[resolve_rig()], FILTER_HZ))
                st = rig.state()
            except OSError as e:
                print(f"TUNE FAILED: rigctld at {args.rigctld} went away while "
                      f"tuning ({e}) — nothing keyed.")
                return PREKEY_REFUSED
            # Read the dial back rather than trusting the set: a QSY that did
            # not take is silent, and loses both directions at once.
            # Against the station's tolerance, not exact equality. This was the
            # one dial check in the tree testing `!=`, roughly twenty times
            # stricter than the shared fact every other transmit path uses --
            # a rig reporting one hertz off would refuse here and pass there.
            # Twenty hertz cannot reach another channel: they are 500 apart.
            if (abs(st["frequency_hz"] - want) > band.QSY_TOLERANCE_HZ
                    or st["mode"] != "PKTUSB"):
                print(f"TUNE DID NOT TAKE: asked for {want} Hz PKTUSB, rig reads "
                      f"back {st['frequency_hz']} Hz {st['mode']} — not keying a "
                      f"radio on the wrong frequency.")
                return PREKEY_REFUSED
            if narrow := clipped(st["passband_hz"], occupied_hz("vara", args.bw)):
                print(f"TUNE DID NOT TAKE: {narrow} — not keying a radio that "
                      f"cuts the emission down on its way out.")
                return PREKEY_REFUSED
            print(f"tuned {args.freq} kHz dial, PKTUSB / {st['passband_hz']} Hz")
        if not args.arm:
            print("  NOT ARMED — pass --arm to key the transmitter", flush=True)
    elif args.freq and not args.dry_run:
        # A dry run ignores the rig flags entirely, so a pasted live command
        # plus --dry-run rehearses instead of dying on a false diagnosis.
        print("--freq needs --rigctld"); return 2

    if rig is not None:
        _install_unkey_handlers(rig)

    calls = (load_gateways(args.gateways, None, False)
             if args.gateways and not args.dry_run else [])
    if args.dry_run:
        print("  DRY RUN — no audio devices, no rig, nothing keyed", flush=True)
        io = _DryIO()
    else:
        wav = (recording_path(args.record, args.mycall, args.gateway)
               if args.record else None)
        if wav is not None:
            print(f"  recording receive audio to {wav}", flush=True)
        print(f"  TX drive {args.amplitude:.2f} peak", flush=True)
        io = AudioVaraIO(args.tx_device, args.rx_device, args.amplitude, rig,
                         record=wav, tx_tail=args.tx_tail, warm=args.arm,
                         trace_timing=args.trace_timing)
    io.progress = progress_watch()
    hs = VaraStationHandshake([args.mycall], io, bw=args.bw, panel=calls,
                              over_continue=args.over_continue,
                              over_continue_after=args.over_continue_after,
                              max_link_setups=args.max_link_setups,
                              cr_retry_form=args.cr_retry_form,
                              allow_data_retries=not args.no_data_retries,
                              probe_intermediate_query=True,
                              tx_level=args.tx_level,
                              high_speed=not args.low_speed_only,
                              compression=args.compression)
    identified = False
    id_owed = (identity.closing_required("vara", optional=args.cwid_optional) and not args.dry_run
               and rig is not None and args.arm)
    with progress_to_stdout():
        mail, session = attach_mail(args, hs, io)
        try:
            ok = connect(args.gateway, args.mycall, args.bw, io, args.timeout,
                         calls=calls, cr_interval=args.cr_interval,
                         max_cr=args.max_cr, listen_first=args.listen_first,
                         force=args.force, hs=hs, pounce_wait=args.pounce_wait,
                         admission=admission)
            if ok and mail is not None:
                print(f"  mail: {args.mycall.upper()} -> {args.gateway.upper()} over "
                      "the held link", flush=True)
                done = mail_session(hs, io, mail, args.mail_timeout)
                # `done` is `_state == _DONE`, and `_fail` sets that state too, so a
                # session that died reports done. besra printed "exchange complete"
                # over a failed KY4RY exchange on 2026-08-14 for exactly this, and
                # the exit code below has always known better -- it reads
                # `session.done and not session.failure`. Say what it says.
                if session is not None and session.failure:
                    print(f"  mail: exchange failed -- {session.failure}", flush=True)
                elif not done and hs.state is not VaraState.CONNECTED:
                    # "out of time" would name the clock for a session the link layer
                    # ended, which is the one thing here that is not the clock's doing.
                    print("  mail: the link closed before the exchange finished",
                          flush=True)
                else:
                    print("  mail: exchange complete" if done
                          else "  mail: out of time", flush=True)
            closed = False
            if ok:
                print("  closing the session (one disconnect burst)", flush=True)
                closed = disconnect_session(hs, io)
            identified = True
            if id_owed:
                identify(io, identity.cw_call, args.bw)
        except ChannelWaitExpired:
            return 3  # Same terminal wait-budget status as the shell gate.
        except AudioWedged as wedged:
            print(f"\n{wedged}. The attempt ends here and says nothing about "
                  f"{args.gateway.upper()} or the band.", flush=True)
            return 1
        except ChannelBusy:
            # Refused ahead of key-down, so there is no verdict to print below: this
            # attempt says nothing about the gateway or the band, and the line it would
            # otherwise reach claims it does.
            return CHANNEL_BUSY
        finally:
            unwinding = sys.exc_info()[1]
            # The transmitter first, and only then the audio. `io.stop()` closes the
            # input and joins a recorder thread that is given ten seconds; an exception
            # escaping the session with PTT asserted — reachable, since a failed unkey
            # returns False rather than raising — used to hold the key through all of
            # it, with only the atexit hook and MAX_KEY_S behind that. A retired rig has
            # already had that ladder run and reported, and running it again would buy
            # nothing but bury the reason the attempt ended.
            #
            # The unkey is itself nested, so the ordering is true rather than probable:
            # an unkey that DOES raise would otherwise leak the input device and the
            # recorder join it is sequenced ahead of.
            try:
                try:
                    if rig is not None and rig.keyed and not rig.retired:
                        rig.unkey_hard("before the audio teardown")
                finally:
                    io.stop()
            except BaseException:
                if unwinding is not None and not isinstance(unwinding, Exception):
                    raise unwinding
                raise
            finally:
                # Every flown session owes its ID, even if its first burst failed.
                if not identified and id_owed:
                    cwid.unsent(identity.cw_call, "the attempt ended before its closing ID")
    # "No response" is a claim about the gateway and the band, and an attempt that
    # lost its transmitter cannot make one. `connect` breaks out of its loop the
    # moment the rig retires, so what follows is one burst and silence — which
    # reads identically to a gateway that never answered, and this station's whole
    # record is read off these verdicts. `onair_session` shows the operator the
    # last ten lines of the attempt, and the abandonment that says why is four
    # lines above that window; this line is inside it.
    if ok:
        print(f"\nCONNECTED to {args.gateway}")
        print(close_verdict(closed, args.gateway), flush=True)
    elif rig is not None and rig.retired:
        print(f"\nNOT connected — the transmitter has been taken down. This attempt "
              f"ended on its own rig and says nothing about {args.gateway.upper()} "
              f"or the band.")
    else:
        print("\nNOT connected (no/!=expected response)")
        if hs.answers:
            frames = ", ".join(str(a.position) for a in hs.answers)
            print(f"  {args.gateway.upper()} sent {len(hs.answers)} confirmed "
                  f"BW{hs.bw} connect offers (frames {frames}); "
                  "setup was not confirmed", flush=True)
    # "No response" and "an answer we could not put a name to" are different
    # findings about the band, and the second used to read as the first: both
    # routes that accept an answer are keyed to the callsign we dialled, so a
    # connect-response addressed to anybody else reached nothing and the attempt
    # reported silence. It happened on both KB3AC-10 arms of 2026-08-26 and was
    # found by ear. Printed under either verdict, because a connect that came up
    # on the fourth request says nothing about who answered the second.
    for a in hs.unattributed:
        print(f"  a connect-response answered this call and names nobody: "
              f"{a.preamble} preamble tones exact, {a.heard} payload tones read, "
              f"and the payload is VARA's own generator output for a callsign that "
              f"is not {args.gateway.upper()} and is not on the panel "
              f"(nearest {a.best_call} at {a.best_tones}). Somebody transmitted; "
              f"the recording is the evidence.", flush=True)
    if mail is not None:
        from hfmodem.winlink import summarize, write_inbox
        print(summarize(session), flush=True)
        for p in write_inbox(session, args.mail_out):
            print(f"mail: wrote {p}", flush=True)
        sent_all = len(session.sent_mids) == len(args.mail_send or [])
        return 0 if ok and done and not session.failure and sent_all else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(exit_status(main))
