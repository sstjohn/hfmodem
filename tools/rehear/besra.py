# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""besra: what the half-duplex mute took off the decoder, measured off the capture.

`RadioLink._capture` records every block the card delivers and pushes only the
unmuted ones at `RollingDecoder`, so a besra capture holds our own keyed intervals
in full and the recorder sits UPSTREAM of the only drop the live path makes. That
makes the two arms exact:

live
    `RollingDecoder(Demodulator(), on_frame, resample=True)` fed the recording as
    0.1 s float32 blocks at 48 kHz, with every block overlapping a keyed interval
    withheld exactly as the mute withholds it, `pump()` after each block and
    `flush()` at the end. This is the live receive path, not a model of it.

ungated
    The same `RollingDecoder` over the same blocks, withholding nothing, less the
    frames that land inside our own keyed intervals — this station's transmissions
    leaking into its own muted receiver, which no receiver anywhere heard.

Both arms are given whatever the session's own log holds — the session id, the
mute, the memory-ARQ epoch — because none of the three is in the audio and the live
decoder had all of them. Without a log the arms are deaf against the station they
are replaying, and `Rehearing.deaf` says so out loud. See `sessionlog`.

One detector, one variable. A whole-capture `Demodulator().decode` was the ungated
arm first, and it is the blinder of the two: its silence gate is a fraction of the
*capture's* peak, so on `20260809T180900Z` it reads three of our six ConReq2000M
transmissions where a 6.25 s window reads five. `_keyed_intervals` is built on what
the ungated arm found, so a blinder ungated arm withholds fewer of our own
transmissions than the live arm then decodes out of them — on that recording, two
ConReq2000M of ours and three bodyless controls minted from their bodies, all five
reported as heard only live, none of them a finding, and a corpus sweep of that
arrangement ran to three figures. Two passes of the same detector cannot do it: a
window artefact appears in both and cancels, and `only_live` is left meaning the one
thing it should, that the live path made something out of the splice.

The trap is the timebase. `RollingDecoder` counts only the samples it was *pushed*,
so the position it hands `on_frame` indexes audio-delivered-to-the-decoder and runs
earlier than the recording by the whole of every withheld block. `_on_the_recording`
adds them back, and what that is worth is not subtle: on the 2026-08-05 KE8LVA
session, which withholds 26.7 s, the four frames the live path finds arrive at
21.54, 46.91, 122.10 and 169.30 s and read at 7.14, 20.61, 95.80 and 141.80
unadjusted — 14.4 to 27.5 s early, twenty-eight to fifty-five slack widths, each
reported at once as a frame discarded and a frame heard only live on a session where
the live path lost nothing. Corrected, all four land on the ungated arm's own
positions to the hundredth of a second.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hfmodem.besra.dsp import detect
from hfmodem.besra.phy.demodulator import (SAMPLE_RATE, _SHORT_CONTROL, DecodedFrame,
                                           Demodulator, frame_span)
from hfmodem.besra.radio import FS_RADIO, RollingDecoder
from hfmodem.core.resample import from_card
from hfmodem.winlink import mask_pr

from . import UPSTREAM, Heard, Rehearing, sessionlog

PROTOCOL = "besra"

_BLOCK = int(0.1 * FS_RADIO)             # the sound card's own block, and the mute's grain

_ZERO = "first sample of the capture WAV (sidecar started_utc)"

_UNCOVERED = (
    "header_only sightings, a guess at a type rather than a frame, which the live "
    "path reports and neither arm counts — they are graded against the log instead; "
    "the bodyless controls that ARE counted, marked `bare`, whose ok is a crispness "
    "floor and a leader-abutment test and never RS or CRC; and ConReq/IDFrame, whose "
    "ok is RS but whose callsign is only checked for length and for holding no space, "
    "so noise can carry one")

#: What a replay hears less of than the station did, and by how much. Both figures
#: are measured on `20260826T133925Z-besra-7060000.wav` against the 51 frames that
#: session logged: hint-less, 28; hinted but muted off the capture's own brackets
#: rather than off the log's edges, 48; both from the log, 51.
_NO_LOG = (
    "no session log names this recording, so both arms run with no session hint — "
    "every bodyless control faces the unsolicited 9.5 crispness floor and "
    "`_scan_header` mints no header-only sighting at all. On 20260826T133925Z that "
    "reading reproduces 28 of the 51 frames the live path reported, losing the ConAck "
    "that opened the session, the END that closed it, eight DATAACKs, three BREAKs and "
    "ten data frames. "
    "Honest for a recording of somebody else's QSO; for one of ours it is a replay of "
    "about half of it")

_NO_ID = (
    "{log} names this recording but logged no frame to take a session id from, so both "
    "arms run with no hint and every bodyless control faces the unsolicited 9.5 "
    "crispness floor — about half a session, measured on 20260826T133925Z")

_NO_ALIGNMENT = (
    "{log} carries the session hint, but its clock would not align to this capture's "
    "own key-downs, so the mute is grown off the capture's silence brackets instead of "
    "read off the log's TX/PTT-OFF edges. On 20260826T133925Z the brackets withhold "
    "64.5 s where the station withheld 50.6, miss five keyings outright, and lose three "
    "more frames with them")


def claims(path: Path) -> bool:
    """This station's own besra captures, ``logs/onair/<stamp>-besra-<freq>.wav``.

    Narrow on purpose: the arms above need a 48 kHz card capture whose keyed
    intervals are still in the file, which is what `CaptureRecorder` writes.
    shrike's session stream is now the same shape and is left unclaimed on
    purpose — see `shrike.py` — and nothing else in the corpus is.
    """
    return path.suffix in (".wav", ".flac") and "-besra-" in path.name


def rehear(path: Path, log=None, *, selection=None) -> Rehearing:
    selection = selection or getattr(log, 'selection', None) or sessionlog.recording_for(path)
    if selection is None:
        return _rehear(path, log)
    chosen = sessionlog.select(selection, path)
    result = Rehearing(path, PROTOCOL, 0., _ZERO, UPSTREAM)
    if not chosen.complete:
        result.note = 'Context incomplete: ' + str(chosen.reason)
        return result
    from copy import deepcopy
    from hfmodem.core import wav
    lo, hi = chosen.bounds
    result.seconds = (hi-lo)/chosen.rate
    for span in chosen.spans:
        rec = deepcopy(selection)
        rec['lanes'][chosen.lane]['selection'] = list(span.bounds)
        part = sessionlog.select(rec, path)
        audio = wav.read(path, FS_RADIO, start=span.bounds[0], stop=span.bounds[1])
        decoded = _rehear(path, part.log, selection=rec, _audio=audio)
        result.append(decoded, (span.bounds[0]-lo)/chosen.rate)
    return result


def _rehear(path: Path, log: sessionlog.SessionLog | None = None, *, selection=None, _audio=None) -> Rehearing:
    card = _audio if _audio is not None else _capture(path)
    au = from_card(card, SAMPLE_RATE)
    seconds = card.size / FS_RADIO
    # The capture's own key edges are read before either arm runs, because the log's
    # clock has to be put on this recording's timebase before the arms can be told
    # anything the log holds — and unlike the intervals below, the edges owe nothing
    # to what was decoded.
    edges, quiet, level = _key_edges(au)
    started = _started(path)
    where = (sessionlog.Alignment(log.selection_offset, 0, 0., 0, 0)
             if log is not None and log.selection_offset is not None else
             None if log is None else sessionlog.align(
                 log, [a for a, _ in edges], _quieter(level, log.keyed),
                 None if started is None else log.predicted(started)))
    offset = 0.0 if where is None else where.offset

    reported = _rolling(card, (), log, offset)
    heard = [(at, f) for at, f in reported if _decoded(f)]
    if where is None:
        keyed = _keyed_intervals(edges, quiet, heard)
        mute = "silence brackets grown off the capture"
    else:
        keyed = _merge(sorted(_on_this_recording(log.keyed, offset, seconds)))
        mute = f"the log's own TX/PTT-OFF edges, {where}"

    told = _rolling(card, keyed, log, offset)
    # Our own transmissions, off the ungated side: this station's audio leaking into
    # its own muted receiver, which no receiver heard. On 20260803T151721Z that is
    # all eleven of them, W9SSJ > KE8LVA on the ConReq repeat grid 0.61 s after each
    # key-down, and counting them as audio the live path threw away would put eleven
    # imaginary losses on a recording that lost nothing. A frame whose header lands
    # in a block *before* the key-down block stays: keying on top of a station
    # already transmitting is a loss, and one this instrument exists to find.
    ungated = [Heard(at, f.name, _detail(f)) for at, f in heard if not _withheld(at, keyed)]
    # By time alone: two carriers of one frame report from the same block, and
    # `DecodedFrame` has no ordering to break the tie with.
    inbound = [(at, f) for at, f in sorted(told, key=lambda p: p[0])
               if not _withheld(at, keyed)]
    return Rehearing(
        recording=path, protocol=PROTOCOL, seconds=seconds,
        zero=_ZERO, recorder=UPSTREAM,
        live=sorted(Heard(at, f.name, _detail(f)) for at, f in inbound if _decoded(f)),
        ungated=sorted(ungated),
        ours=_spoken(heard, keyed),
        # A withheld block is whole, so a frame at a keyed edge loses up to 0.1 s of
        # itself and the header scan then pins it anywhere within a leader (0.24 s)
        # of where the ungated arm reads it. Against that, the tightest repeat of one
        # frame kind in these recordings is KE8LVA's IDLE at 2.46 s, so half a second
        # separates a match from its own neighbour with room either way.
        slack_s=0.5,
        uncovered=_UNCOVERED,
        deaf=_deaf(log, where),
        against_log=_against(log, offset, inbound),
        note=f"mute from {mute}: {len(keyed)} intervals, "
             f"{sum(b - a for a, b in keyed):.1f} s, holding "
             f"{len(heard) - len(ungated)} frames of our own. "
             f"{_hint(log)}{_unbracketed(keyed) if where is None else ''}",
    )


def _hint(log: sessionlog.SessionLog | None) -> str:
    if log is None or log.session is None:
        return "No session hint."
    return (f"Both arms hinted with session 0x{log.session:02X} and {len(log.epochs)} "
            f"memory-ARQ resets, read from {log.path.name}.")


def _unbracketed(keyed: Sequence[tuple[float, float]]) -> str:
    """How many of the intervals are shorter than anything that can be transmitted
    in one — a key edge whose transmission neither the quiet nor a frame closed
    around. Whatever our own audio decoded to there lands on the ungated side and is
    counted as a loss it never was."""
    n = sum(b - a < 0.44 for a, b in keyed)
    return f" {n} of the intervals are a key edge this pass could not bracket."


def _deaf(log: sessionlog.SessionLog | None,
          where: sessionlog.Alignment | None) -> str:
    """What this replay could not hear that the station could.

    A run without the log is a legitimate reading and the only one available for
    somebody else's QSO. It is not a reading of what this station heard, and the
    difference is half the frames — so it is said in the report rather than left for
    the reader to know.
    """
    if log is None:
        return _NO_LOG
    if log.session is None:
        return _NO_ID.format(log=log.path.name)
    if where is None:
        return _NO_ALIGNMENT.format(log=log.path.name)
    return ""


def _against(log: sessionlog.SessionLog | None, offset: float,
             inbound: Sequence[tuple[float, DecodedFrame]]) -> tuple[int, int] | None:
    """This replay's gated arm against the frames the session itself logged.

    Everything the arm reported goes in, header-only sightings included: the live
    path logged those and a replay that reproduces 46 of 51 by declining to count
    the five it was never asked about has reproduced nothing.
    """
    if log is None or not log.frames:
        return None
    got = [(at, f.name) for at, f in inbound]
    frames = log.frames
    if getattr(log, 'selection', None):
        frames = getattr(log, 'grading', None)
        if frames is None:
            return None
    return len(sessionlog.reproduced(frames, got, offset)), len(frames)


def _on_this_recording(keyed: Sequence[tuple[float, float]], offset: float,
                       seconds: float) -> list[tuple[float, float]]:
    """The log's intervals on the recording's timebase, clipped to it. A session
    whose first key-down precedes its own capture is ordinary — the recorder opens
    after the rig is keyed for the channel check."""
    out = []
    for a, b in ((a - offset, b - offset) for a, b in keyed):
        if b > 0 and a < seconds:
            out.append((max(a, 0.0), min(b, seconds)))
    return out


def _spoken(heard: Sequence[tuple[float, DecodedFrame]],
            keyed: Sequence[tuple[float, float]]) -> list[Heard]:
    """The payload this station transmitted, read back off its own muted receiver.

    ARQ repeats collapse on the frame's own even/odd tag, which is what the far
    end dedupes on: on `20260819T024321Z` the third handshake frame goes out
    three times and the gateway reads it once. Masking then runs over the joined
    stream rather than over a frame, because a 16-byte frame splits `;PR:` down
    the middle — `...\r;P` ends one and `R: <digits>` opens the next — and a
    per-frame mask would print the digits and miss. `mask_pr` is the transcript's
    own rule, and it strikes out character for character, so the chunks below still
    fall on the frame boundaries the join was made from.

    The fold judges the frames itself rather than trusting the caller to have done
    it. `_rolling` reports what did not decode alongside what did, and a frame the
    channel destroyed still carries the bytes RS could not correct — so one landing
    between two repeats takes the turn of the frame it shadows and drops it from the
    transcript. Nothing raises; the answer just comes back one frame short.
    """
    said = [(at, f) for at, f in sorted(heard, key=lambda p: p[0])
            if _decoded(f) and f.payload and _withheld(at, keyed)]
    fresh: list[tuple[float, DecodedFrame]] = []
    for at, f in said:
        if not (fresh and fresh[-1][1].name[-1] == f.name[-1]):
            fresh.append((at, f))
    text = mask_pr("".join(f.payload.decode("latin-1") for _, f in fresh))
    out, cut = [], 0
    for at, f in fresh:
        chunk = text[cut:cut + len(f.payload)]
        cut += len(f.payload)
        out.append(Heard(at, f.name, chunk.encode("unicode_escape").decode("ascii")))
    return out


def _capture(path: Path) -> np.ndarray:
    """The recording as the sound card delivered it: mono float32 at 48 kHz.

    `CaptureRecorder` writes int16 of the card's float block times 32768 and
    nothing else — no normalisation, the sidecar says so — so dividing back
    reconstructs the block `RadioLink._capture` was handed, sample for sample.
    """
    from hfmodem.core import wav
    from hfhost.recordings import intervals
    for _, segment in intervals(path):
        import wave
        with wav.opened_source(segment) as stream, wave.open(stream) as source:
            params = source.getparams()
            if (params.framerate, params.nchannels, params.sampwidth) != (FS_RADIO, 1, 2):
                raise ValueError(f"{path.name}: not a 48 kHz mono int16 besra capture")
    first, last = wav.analysis_bounds(path)
    return wav.read(path, start=first, stop=last).astype(np.float32)


def _started(path: Path) -> dt.datetime | None:
    """When the recorder opened this capture, from the sidecar it wrote beside it.

    The recording's own zero, and half of what puts a session log on its timebase.
    Absent for a capture whose sidecar never landed — a run killed mid-flight — and
    then the key-downs vote alone.
    """
    try:
        stamp = json.loads(path.with_suffix(".json").read_text())["started_utc"]
    except (OSError, ValueError, KeyError):
        return None
    return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")


def _key_edges(au: np.ndarray) -> tuple[list[tuple[float, float]],
                                        list[tuple[float, float]], np.ndarray]:
    """The capture's own key transitions, the quiet stretches they sit in, and the
    level trace both were read off.

    Every transmission is bracketed in the capture by two runs of digital silence:
    from key-down until the first transmitted sample (`Rig.settle`) and from the
    last one until the mute clears. They read 35-63 dB under the band's own median
    and the next quietest thing in any of these files is 13 dB above that, so the
    brackets are unmistakable. What lies between them is not: this station's
    captures hold their own transmissions anywhere from 22 dB under the band (a rig
    muting its own receiver, and the leak is all that reaches the file) to 13 dB
    *over* it (one whose capture the key does not mute at all). Both are in
    ``logs/onair``, so no level rule finds our own audio and the brackets have to.

    The edges are also what a session log's clock is aligned against, which is why
    they are read apart from the intervals below: the edges owe nothing to what was
    decoded, and the alignment has to be settled before either arm runs.
    """
    if not au.size:                                      # a run that died before its first block
        return [], [], np.zeros(0)
    n = SAMPLE_RATE // 100                               # 10 ms
    x = au.astype(np.float64)                            # int16 squared is int16, and wraps
    rms = np.sqrt(detect.boxcar(x * x, n)[::n] / n)
    med = float(np.median(rms))
    # A tenth of a second of level, on the 10 ms grid. A lone 10 ms frame of band
    # reaches down into the muted leak's range between symbols; a tenth of a second
    # of band does not. Padded with the median so a stretch cannot grow off an end.
    smooth = detect.boxcar(np.r_[np.full(4, med), rms, np.full(5, med)], 10) / 10
    seconds = n / SAMPLE_RATE
    return ([(a * seconds, b * seconds) for a, b in _runs(rms < 0.05 * med)],
            [(q * seconds, r * seconds) for q, r in _runs(smooth < 0.35 * med)],
            20 * np.log10(np.maximum(rms, 1e-9)))


#: The level trace's own grain, and `_key_edges`' — a hundredth of a second.
_LEVEL_S = 0.01

#: How far under the band either side of it the capture has to sit inside one of the
#: station's own keyings before that keying corroborates an alignment. The rig mutes
#: its own receiver while we transmit, so a keying laid over the audio in the right
#: place reads quieter than the band it interrupts. On the 2026-08-29 15:15 W4UC call
#: every one of the six keyings drops 4.1-4.4 dB, and the key-up comb one
#: transmission out reads 3.5-3.9 dB the *other* way on every one of them.
_MUTE_DB = 1.0

#: How much band to read either side of a keying. Long enough to median away a
#: symbol and short enough to stay inside the 2.5 s that separates transmissions.
_FLANK_S = 0.5

#: What a key edge costs at each end of a keying: the settle bracket at the start and
#: the PTT tail at the end are digital silence, not our audio, and neither is band.
_EDGE_S = 0.05


def _quieter(level: np.ndarray,
             keyed: Sequence[tuple[float, float]]) -> Callable[[float], int]:
    """How many of the station's own keyings the capture's level agrees with.

    The discriminator `sessionlog.align` settles a tied vote on, and the only thing
    in the capture that tells a key-down bracket from a key-up one: the receiver is
    muted for the whole of a transmission of ours and hears the band on both sides
    of it, so at the right offset every keying the log records sits in a trough and
    at an offset one transmission out every keying straddles one.

    What it costs where the mute is shallow is a narrower margin, keying by keying.
    On `20260826T133925Z` our own leader reaches the file at band level and only the
    frame behind it is down, and 55 of the 56 keyings still read as a trough. On
    `20260814T000846Z`, where the rig mutes least of anything in the corpus, four of
    six do and the winner leads 4 to 0. Where none do — a capture holding our own
    audio at or over the band, which this project has — every candidate scores zero,
    nothing outranks anything, and `align` raises rather than choosing. That is the
    cost, and it is the right one: the reading it declines to give is a coin toss.
    """
    def span(a: float, b: float) -> np.ndarray:
        return level[max(int(a / _LEVEL_S), 0):min(int(b / _LEVEL_S) + 1, level.size)]

    def trough(a: float, b: float) -> bool:
        inside = span(a + _EDGE_S, b - _EDGE_S)
        band = np.r_[span(a - _FLANK_S, a - _EDGE_S), span(b + _EDGE_S, b + _FLANK_S)]
        if inside.size < 10 or not band.size:
            return False
        return float(np.median(band) - np.median(inside)) >= _MUTE_DB

    def at(offset: float) -> int:
        return sum(trough(a - offset, b - offset) for a, b in keyed)

    return at


def _keyed_intervals(edges: Sequence[tuple[float, float]],
                     quiet: Sequence[tuple[float, float]],
                     heard: Sequence[tuple[float, DecodedFrame]]) -> list[tuple[float, float]]:
    """When ``RadioLink.muted`` was set, in seconds — grown off the capture alone.

    This is the model for a recording with no session log beside it. The log's own
    `TX`/`PTT OFF` lines are the mute itself and are preferred wherever they are to
    hand: on `20260826T133925Z` this model withholds 64.5 s where the station
    withheld 50.6, misses five keyings outright, and costs the replay three frames.

    A bracket is grown over the quiet stretch it sits in, which recovers the whole
    interval wherever the rig muted the capture; where it grows into nothing,
    `_bracketed` closes it around the frame instead. Pairing each bracket with the
    next one regardless was tried and is worse: it needs the brackets to alternate
    key-down and key-up, one missed edge inverts every interval after it, and an
    inverted interval withholds the band and hides the answer that arrived in it.

    `test_offair_ke8lva_greeting._keyed_intervals` reads the key-down bracket alone,
    with a 0.2 s floor it can afford because besra keyed on a hardcoded 0.35 s
    settle when that recording was made. The settle is the rig's own 0.04 s now and
    the same rule finds 0.06 s brackets, so it reports **no keyed intervals at all**
    on every capture after 2026-08-09 and the live arm silently stops being gated.
    """
    keyed: list[tuple[float, float]] = []
    unexplained: list[tuple[float, float]] = []
    for a, b in edges:
        # Overlapping, not containing: the smoothed level clears the threshold a
        # frame or two into the bracket, where the band the bracket interrupts is
        # still in the window, and containment loses every key-down for that frame.
        lo, hi = next(((q, r) for q, r in quiet if q < b and a < r), (a, b))
        # The shortest transmission there is runs 0.44 s — a bodyless control behind
        # its leader — so a quiet stretch no longer than the bracket that anchors it
        # is a key the capture was not muted for, and our audio is somewhere louder.
        (keyed if hi - lo >= b - a + 0.44 else unexplained).append((lo, hi))

    return _merge(sorted(keyed + _bracketed(unexplained, heard)))


def _bracketed(edges: Sequence[tuple[float, float]],
               heard: Sequence[tuple[float, DecodedFrame]]) -> list[tuple[float, float]]:
    """Key edges the quiet did not explain, closed around the frame between them.

    Where the key does not mute the capture, the only marks a transmission leaves
    are its two edges and the frame between them, which the ungated arm then reads at
    full strength: 44 of the 47 frames in one 164 s session on 2026-08-14 are our
    own, and nothing in their level tells them from the gateway's.

    Reading them off the ungated arm rather than off a whole-capture pass is what
    makes the two arms agree. The whole-capture pass finds three of our six
    transmissions on `20260809T180900Z` and the rolling arm finds five, so anchoring
    on the former left three transmissions unwithheld for the live arm to decode out
    of — two ConReq2000M of ours reported as heard-only-live, and three bodyless
    controls minted from their bodies with them.

    The geometry does, and not loosely. Our leader begins where the key-down edge
    ends and runs 0.24 s to the header; the frame's last sample is `frame_span` past
    that, and the key-up edge follows on the PTT's own tail. Over those 44 the two
    distances read 0.25-0.32 s and 0.03 s, every one — against 0.39 s for the
    gateway data frame in the same file that the second window keeps out. A remote
    frame has to have arrived inside our PTT cycle, on a receiver that was muted for
    it, to fit.
    """
    out = []
    for at, f in heard:
        ends = at + frame_span(f.type) / SAMPLE_RATE
        down = next((e for e in edges if 0.10 <= at - e[1] <= 0.35), None)
        up = next((e for e in edges if -0.05 <= e[0] - ends <= 0.15), None)
        if down is not None and up is not None and down[1] < up[0]:
            out.append((down[0], up[1]))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edge = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1).tolist(), np.flatnonzero(edge == -1).tolist()))


def _merge(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


@dataclass(slots=True)
class _Party:
    """What the ARQ layer told the live decoder, at the point in the recording the
    decoder has reached.

    `Demodulator` polls `expect_session` and `rx_epoch` once per pass, and both
    answer for *now* — so a replay that returns one constant for a 265 s file is
    modelling a session that never began and never ended. The log holds both
    schedules: the first `TX` line to the `[DISC]` marker for the id
    (`ArqSession.expected_session` is None outside its own state), and the `[IRS]`
    and `[DISC]` markers for the epoch (`ArqSession._set_state`).
    """
    session: int | None = None
    span: tuple[float, float] = (0.0, math.inf)
    epochs: tuple[float, ...] = ()
    at: float = 0.0
    selection: dict | None = None
    offset: float = 0.0

    @classmethod
    def of(cls, log: sessionlog.SessionLog | None, offset: float) -> "_Party":
        if log is None:
            return cls()
        selection = getattr(log, 'selection', None)
        if selection is not None:
            return cls(selection=selection, offset=offset)
        lo, hi = log.session_span
        return cls(log.session, (lo - offset, hi - offset),
                   tuple(e - offset for e in log.epochs))

    def state(self):
        if not self.selection or self.selection.get('complete') is False:
            return {}
        ctx = self.selection.get('context', {})
        lane = self.selection.get('lane', 'rx')
        ref = self.selection.get('lanes', {}).get(lane, {})
        mapping = ctx.get('lanes', {}).get(lane, {})
        anchor = mapping.get('anchor', {})
        rate = ref.get('rate', 1)
        sample = anchor.get('sample', 0) + (self.at + self.offset - anchor.get('transcript_s', 0)) * rate
        schedule = ctx.get('lane_states', {}).get(lane, {}) if lane != 'rx' and 'rx' in self.selection.get('lanes', {}) else ctx
        state = dict(schedule.get('initial', {}))
        for change in schedule.get('changes', []):
            if change.get('clock_generation') == ref.get('clock_generation') and change['sample'] <= sample:
                state.update(change)
        return state

    def expect(self) -> int | None:
        if self.selection is not None:
            return self.state().get('expected_session')
        return self.session if self.span[0] <= self.at <= self.span[1] else None

    def epoch(self) -> int:
        if self.selection is not None:
            return self.state().get('arq_epoch', 0)
        return sum(e <= self.at for e in self.epochs)


def _rolling(card: np.ndarray, keyed: Sequence[tuple[float, float]],
             log: sessionlog.SessionLog | None,
             offset: float) -> list[tuple[float, DecodedFrame]]:
    """The live receive path over the recording, withholding what ``keyed`` covers.

    Both arms are this, told the same things by the same log, and the ungated one
    passes no intervals at all — so the two differ in which blocks reached the
    decoder and in nothing else.
    """
    got: list[tuple[int, DecodedFrame]] = []
    arq = _Party.of(log, offset)
    rx = RollingDecoder(Demodulator(expect_session=arq.expect, rx_epoch=arq.epoch),
                        lambda pos, f: got.append((pos, f)), resample=True)
    pushed: list[int] = []                   # the recording block each pushed block came from
    for i in range(0, card.size, _BLOCK):
        arq.at = (i + _BLOCK) / FS_RADIO
        if not _muted(i / FS_RADIO, (i + _BLOCK) / FS_RADIO, keyed):
            rx.push(card[i:i + _BLOCK])
            pushed.append(i // _BLOCK)
        # The pump thread runs on its own cadence whether or not the last block was
        # pushed, so a muted stretch still gets the windows its elapsed time earns.
        rx.pump()
    rx.flush()
    return [(_on_the_recording(pos, pushed), f) for pos, f in got]


def _muted(t0: float, t1: float, keyed: Sequence[tuple[float, float]]) -> bool:
    return any(a < t1 and t0 < b for a, b in keyed)


def _withheld(at: float, keyed: Sequence[tuple[float, float]]) -> bool:
    """Was the block carrying this frame's header kept from the decoder?

    On the block, not on the interval: the mute drops whole 0.1 s blocks, so the
    ungated arm has to exclude on the same grain or the two disagree about a frame
    whose header lands in the block a keyed interval opens in but a few thousandths
    of a second ahead of the interval itself. Three ConReq2000M and ConAck2000 of
    ours on `20260814T051359Z` sit 4.3, 2.2 and 0.1 ms before their own key-down —
    kept by a point test, never seen by the live arm, and three of the corpus's nine
    discards. Their neighbours in the same file clear the interval start by 0.5 ms
    and are excluded, which is how close a point test runs to a coin toss here.
    """
    block = int(at * FS_RADIO) // _BLOCK
    return _muted(block * _BLOCK / FS_RADIO, (block + 1) * _BLOCK / FS_RADIO, keyed)


def _on_the_recording(pos: int, pushed: Sequence[int]) -> float:
    """A `RollingDecoder` position, in seconds from the recording's first sample.

    `pos` is a 12 kHz index into the audio the decoder was *given*, which is the
    recording minus every withheld block, so it has to be un-compressed by the
    blocks withheld before it or every frame after our first transmission reads
    seconds early — and, at 0.5 s slack, reads as a frame the live path discarded.
    """
    n, off = divmod(pos * (FS_RADIO // SAMPLE_RATE), _BLOCK)
    return (pushed[min(n, len(pushed) - 1)] * _BLOCK + off) / FS_RADIO


def _decoded(f: DecodedFrame) -> bool:
    """A frame this receiver read, as against one it guessed at.

    `ok` is an integrity verdict wherever there is content to check — RS and payload
    CRC for data, RS over a 12+4 block for ConReq/ID/Ping, majority over three
    repeats for ConAck/PingAck. Two classes are weaker than that reads, and the
    corpus has caught both, so neither is worth a line in a report unqualified.

    A bodyless control has no body at all: `ok` is True the moment ten tones parse
    as a type, and all that stands behind it is a crispness floor and
    `_leader_backed`'s abutment test. Which floor depends on whether the session id
    is to hand — 8.0 for one this end is party to, 9.5 for one from nobody — and the
    detail line says `bare` either way.

    A ConReq or ID is RS-backed but not callsign-backed: `callsign.unpack_callsign`
    checks the unpacked field for length and for having no space in it and for
    nothing else, so an RS-corrected block of noise can carry one. Two survive the
    whole corpus — `ConReq1000M  FC<C155-5 > ;_0Q5@@` on 20260819T024321Z and
    `IDFrame  *?V"*JJ-E` on 20260814T003231Z — and both read ok=True.

    Neither is filtered here. Reading a frame out is this module's job and re-judging
    the demodulator's verdict is not; `uncovered` says what the verdict is worth.

    `header_only` is the same shortfall without the floor: a type claimed off ten
    tones when nothing validated, which for a bodyless control comes back ok=True
    because there is no body to fail. Never counted.
    """
    return f.ok and not f.header_only


def _detail(f: DecodedFrame) -> str:
    bits = [f"sess 0x{f.session_id:02X}"]
    if f.caller:
        bits.append(f"{f.caller} > {f.target}" if f.target else f.caller)
    if f.grid:
        bits.append(f.grid)
    if f.conack_timing_ms is not None:
        bits.append(f"leader {f.conack_timing_ms} ms")
    if f.pingack_sn_db is not None:
        bits.append(f"S/N {f.pingack_sn_db} dB")
    if f.payload:
        bits.append(f"{len(f.payload)} B")
    if f.name in _SHORT_CONTROL:
        bits.append("bare")
    return "  ".join(bits)
