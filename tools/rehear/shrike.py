# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Replay a shrike session capture through the live receive path, and ungated.

THE RECORDER IS DOWNSTREAM OF THE DROPS, and nothing else here means anything
until that is read. `_save_capture_async` writes the segment the session READ,
off `seg_start = live.pos - seg.size`, and `_LiveInput.flush_to(self.tx_end)`
has already walked the capture stream's floor past every sample taken while our
own carrier was up. Named rather than cited by line: the four `onair.py` numbers
that stood here read 4860, 4674, 804 and 1319 against code at 4871, 4685, 807 and
1322, and the rest of this docstring is de-cited for the same reason. So a shrike
WINDOW capture cannot hold a sample the decoder was denied, and no amount of
replaying one will measure a sample drop. The other two adapters record in front
of their gating; this one does not, and says so rather than reporting a zero.

The samples the floor walked past are not lost — they are in `stream.wav`, which
`_LiveInput` writes from the callback ahead of the floor, on the same index the
windows' `end_stream_sample` counts. That file is deliberately NOT claimed here:
it holds our own transmissions, and this adapter's live arm would report each of
them as something heard.

WHICH IS ALSO WHERE `ours` IS, and why it is empty here rather than unreadable. A
window capture starts at `tx_end` by construction, so no window can hold a sample
of our own transmission and this adapter has nothing to put in the field the other
two fill. `stream.wav` has every one of them, back through the rig at about
-42 dBFS against a -88 dBFS mute: `tools/txwitness.py edges` reads the anatomy of
each keying off it, and `tests/shrike/test_txhead.py` reads the bits.

What IS in front of the recording is the decoder-side gating, which is the larger
half of the loss anyway:

  * `RollingRx._decode` forces `p3_envelope=False` on every stream decode --
    846.7 ms per call over 0.85 s of audio is why. The PACTOR-3
    envelope fallback has never run on a live stream in this modem's history, so
    a `packet` it finds here is one the live path is structurally unable to reach.
  * `RollingRx` decodes nothing below `MIN_DECODE_S`, half a second, and a calling
    cycle's listening window is 0.24 s. The rolling decoder is SILENT for the
    whole connect phase. See `uncovered`: that is not the same as deaf.
  * `_SessionRx.flush` cuts the buffer to `FLUSH_CONTEXT_S` (0.75 s) before the
    end-of-cycle decode, so an IRS hold cycle -- where `_listen_until_answer`
    holds every slice and feeds none of it -- has its last 0.75 s decoded once,
    with no slide behind it, and the rest of the peer's packet not at all.
  * `cs_max_errors` is NOT one of them, whatever `_SessionRx.feed` looks like.
    `CS_EXPECTED_MAX_ERRORS` is `CS_MAX_ERRORS` is 0 in `rxfront`: the relaxation
    was withdrawn and the choice now selects between two equal numbers. The live
    arm still makes it, as the live path does.

`TR_SWITCH_S` (0.055 s) is not a drop and is not paid here. `_acquisition_window`
spends it as the LOWER EDGE of the band a connect answer is searched in --
receive audio is still muted that long after our carrier, so a
zero-error match inside it is a match on noise, and one of the 2026-08-06 phantom
links came from an accept 13 ms in. Nothing discards those samples. They are in
the capture and the ungated arm reads them.

SO THIS ADAPTER REPORTS NO DISCARDS, AND DID NOT UNTIL 2026-09-03. It printed a
`discarded` column against a live arm that was `RollingRx` alone -- one of the
session's four readers, and the one that is silent by construction on a 0.24 s
calling window -- and its own caveat called the result "an upper bound on what
was lost and not the loss". A column that is an upper bound on nothing is not an
instrument, and it was read as a receive-gating lead twice in one week and
withdrawn twice: `onair-0902-2342` printed `live 6, ungated 20, discarded 14`
against a session log naming 52 answered cycles, and `onair-0903-1107` printed
`discarded 4 cs CS1` for four CS1 that log names itself.

What replaced it is the session's own reader, run at the session's own instant.
`_SessionRx.control_signal`'s PACTOR-1 half is `_p1_cs`, a point read at the
answer instant the transmit grid holds, and the instant is in the arm's log --
`[grid] CS due @ <sample>`, on the same stream clock a sidecar's
`end_stream_sample` counts. So a held cycle's window can be laid against the grid
and read the way the station read it, and each ungated hit gets one of three
readings rather than a count:

  * SEEN -- the session's own log names the same codeword at the same instant, or
    the replayed anchored read takes it. Not a loss in any sense.
  * IN A PACKET -- it sits inside a transmission of the peer's that this capture
    holds whole, which for a changeover is `P1_CS_S + P1_PACKET_S` of 100 Bd data
    behind the head. A false accept by construction: an ISS keys no control
    signals, and the session's own one-per-cycle rule (`_SessionRx.cs_seen`)
    means no reader of ours was even offered it. Ten of `onair-0902-2342`'s
    fourteen are these, 0.18-0.68 s into the peer's own changeover packet.
  * UNSEEN -- what is left, reported with its offset from the answer slot. The
    only class that can be a loss, and the only one worth a rig slot.

WITHOUT A LOG THERE IS NO ANCHOR AND SO NO READING, and that is `deaf` rather
than a column of zeros: the grid's answer instant is a property of when WE keyed
and no capture holds it.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from hfmodem.shrike import live, onair, rxfront, spec

from . import (DOWNSTREAM, IN_PACKET, SEEN, UNSEEN, Heard, Rehearing, Verdict,
               _near, sessionlog)

PROTOCOL = "shrike"

FS = rxfront.FS

# What a station acts on, and the line `hfmodem.tests.shrike.test_corpus` draws:
# `detect`, `fsk` and `p1reply` state that something is at the tones without
# claiming to have read a bit of it, so counting them would let an ungated pass
# "recover" events that were never decodes. A `packet` whose CRC failed is not a
# frame either, and is dropped below rather than counted at a discount.
CONFIRMABLE = ("connect", "cs", "packet")

# `_listen_until_answer`'s `slice_s`, which is a default argument rather than a
# constant and is the size the session actually hands the decoder. It is also
# `RollingRx`'s slide, so a replay that pushed the whole capture in one call would
# cross the slide once instead of four times a second and decode over a history
# the session never had.
SLICE_S = 0.25

_CAPTURE = re.compile(r"(rx|hold)_\d+")

#: A HELD cycle's window, which is the one the anchored point read runs in.
#: `_SessionRx.control_signal` refuses outside `LINKED`, and `rx_NN` is what the
#: calling loop writes -- so replaying the anchor there would credit the live arm
#: with a decode the session's connect phase reached by a different reader, or by
#: none.
_HELD = re.compile(r"hold_\d+")

#: Two reads this close are one transmission. `_SessionRx._on` uses `2 * P1_CS_S`
#: for the same judgement across its own readers; one codeword is enough here
#: because both arms are looking at one window, and the session log prints its
#: times to two decimals, which is 5 ms of quantisation to clear.
_SAME_S = spec.P1_CS_S

#: How long the peer holds the channel after a changeover head: the head itself
#: and the packet behind it. `pactor1.breakin_signal` renders 1.06 s of that.
_BREAKIN_S = spec.P1_CS_S + spec.P1_PACKET_S

_ZERO = (
    "sample 0 is the first sample the session read after `flush_to(tx_end)`: the "
    "first capture-stream sample after our own carrier dropped, which is neither "
    "a cycle boundary nor a wall-clock instant. `end_stream_sample` in the sidecar "
    "puts the LAST sample on the session's absolute stream clock and is the only "
    "key back to it; captures written before that field was added carry no such "
    "key and every time in them is capture-relative only")

_UNCOVERED = (
    "sample drops of every kind -- the recorder is behind them. And one of the "
    "session's four readers: the connect-phase codeword search "
    "(`p1rx.acquire_control_signal`, banded by `_acquisition_window`), whose band "
    "is set by how far this window's first sample sits past our own carrier and "
    "which no capture records. A connect-phase hit is therefore read against the "
    "arm's LOG rather than replayed -- the log names both the codeword and the "
    "sample -- and never counted as a loss on the strength of the rolling arm's "
    "silence, which on a 0.24 s window is `MIN_DECODE_S` and not a gate. And the "
    "two unassigned words: `CONFIRMABLE` holds only the four codewords PACTOR-1 "
    "gives a meaning, so an arm the peer spent commanding PACTOR-3 with `0x59A` "
    "reports a `live` of almost nothing beside a log full of decodes. Both arms "
    "are blind to them alike, so the comparison stands and the count does not")

_NOTE = (
    "the live arm pushes every slice, where a receiving cycle holds them, so it "
    "decodes at least as much as the session's rolling decoder did, and on a held "
    "cycle it also reads `_SessionRx.control_signal`'s PACTOR-1 point read at "
    "every instant the arm's own grid aimed one at. Both are a replay and not the "
    "session's own log, which is read separately and is what SEEN stands on.")

_DEAF = (
    "{why}, so the transmit grid's answer instant is "
    "unknown and `_SessionRx.control_signal` -- the point read that delivered 44 "
    "of `onair-0902-2342`'s 52 answered cycles -- cannot be run at all. What is "
    "left is the rolling decoder, which decodes nothing below `live.MIN_DECODE_S` "
    "and so hears nothing whatever on a 0.24 s calling window: 6 of that arm's 52. "
    "Every ungated hit here is UNCLASSIFIED rather than lost, and the summary "
    "counts it in none of the three")


def claims(path: Path) -> bool:
    """A shrike session capture: `rx_NN.wav` or `hold_NN.wav`, with its sidecar.

    The sidecar is part of the identity rather than a convenience. `write_wav`
    normalises every capture to 0.8 peak, so the WAV alone cannot answer one
    level question about the audio that was received.
    """
    return (path.suffix in (".wav", ".flac") and _CAPTURE.fullmatch(path.stem) is not None
            and path.with_suffix(".json").exists())


def rehear(path: Path, log=None, *, selection=None) -> Rehearing:
    selection = selection or getattr(log, 'selection', None) or sessionlog.recording_for(path)
    if selection is None:
        return _rehear(path, log)
    chosen = sessionlog.select(selection, path)
    result = Rehearing(path, PROTOCOL, 0., _ZERO, DOWNSTREAM)
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
        audio = wav.read(path, FS, start=span.bounds[0], stop=span.bounds[1])
        decoded = _rehear(path, part.log, selection=rec, _audio=audio)
        result.append(decoded, (span.bounds[0]-lo)/chosen.rate)
    return result


def _rehear(path: Path, log=None, *, selection=None, _audio=None) -> Rehearing:
    """Replay one window, and say what each ungated hit is.

    The log is not read for a session id -- a shrike arm has none -- but for the
    two things the audio cannot hold: the instants this arm's transmit grid aimed
    a control signal at, and what the station read there. Without it the anchored
    reader has nothing to aim and the hits are left unclassified; see `_DEAF`.
    """
    selection = selection or getattr(log, 'selection', None)
    selected = sessionlog.select(selection, path) if selection is not None else None
    if _audio is not None:
        audio = _audio
    elif selected is not None:
        from hfmodem.core import wav
        audio = wav.read(path, FS, start=selected.bounds[0], stop=selected.bounds[1]) if selected.bounds else np.empty(0)
    else:
        audio = rxfront.load_wav(str(path))
    stats = json.loads(path.with_suffix(".json").read_text())
    end = stats.get("end_stream_sample")
    where = (f"ends at stream sample {end}"
             if end is not None else
             "unplaceable on the session clock -- written before `end_stream_sample`")
    # An older sidecar carries `grid_valid`, which was `xruns == 0` -- the
    # driver's silence and not the capture's, since the loss this station has is
    # blocks a starved interpreter dropped and those never reach that counter. It
    # is not evidence of a whole capture and is not read as one here.
    timing = {}
    if limitation := stats.get('timestamp_limitation'):
        timing = {k: stats[k] for k in ('timestamp_limitation', 'estimated_lost_samples',
                  'lost_samples', 'xruns', 'grid_loss_seen', 'continuous') if k in stats}
        continuity = 'failed' if stats.get('continuous') is False else 'unknown'
        loss = (f", and continuity {continuity}: {limitation}; "
                f"uncertain estimate {stats.get('estimated_lost_samples', 'unknown')} "
                f"timestamp sample(s), separate from {stats.get('lost_samples', 'unknown')} "
                f"confirmed lost sample(s) and {stats.get('xruns', 'unknown')} xrun(s); "
                "timestamp estimates cannot establish converter loss or clear the grid")
    elif "lost_samples" not in stats:
        loss = (", and its sidecar predates the lost-block counter -- what it is "
                "missing is `tools/clock_shortfall.py` against the session log, "
                "or `tools/grid_phase.py` over the directory, which needs no log "
                "because the peer's own transmitter is the reference")
        marked = path.parent / "clock_shortfall.json"
        if marked.exists():
            loss = (f", and its sidecar predates the lost-block counter -- the "
                    f"`clock_shortfall.json` beside it puts the session at least "
                    f"{json.loads(marked.read_text())['lost_ms']:.0f} ms short of "
                    f"the air")
    elif stats["grid_loss_seen"]:
        loss = (f", and {stats['lost_samples']} sample(s) plus {stats['xruns']} "
                f"xrun(s) put every index in it off the air -- by at least that, "
                f"since neither counter sees loss upstream of the converter")
    else:
        loss = (", and no loss reached either counter, which is as far as this "
                "capture can clear itself from the inside -- `tools/grid_phase.py` "
                "reads the peer's bursts against our own cycle and so sees loss "
                "upstream of the converter that neither counter can, and "
                "`tools/witness_align.py` against a second receiver sees the "
                "whole hole")
    zero = end - stats["samples"] if end is not None else 0
    if selected is not None and selected.bounds:
        zero, end = selected.bounds
    source_rate = selected.rate if selected is not None else FS
    grid = ((sessionlog.shrike_grid(log.path, source_rate, selection=selection, lane=selected.lane) if selection is not None
             else sessionlog.shrike_grid(log.path, FS))
            if log is not None and end is not None else None)
    aimed_at, said = (grid.within(zero, end, source_rate) if grid is not None else ([], []))
    rolling = _rolling(audio)
    anchored = (_anchored(audio, [round((a - zero)*FS/source_rate) for a in aimed_at])
                if _HELD.fullmatch(path.stem) or selected is not None else [])
    ungated = _ungated(audio)
    return Rehearing(
        recording=path, protocol=PROTOCOL, seconds=len(audio) / FS,
        zero=_ZERO, recorder=DOWNSTREAM,
        live=sorted(rolling + anchored), ungated=ungated,
        aimed=set(anchored),
        verdicts=(_verdicts(ungated, rolling, anchored,
                            [_said(w, zero*FS/source_rate) for w in said],
                            [(a - zero) / source_rate for a in aimed_at])
                  if grid is not None else None),
        deaf="" if grid is not None else _DEAF.format(
            why="this capture predates `end_stream_sample` and cannot be put on the "
                "session clock at all" if end is None else
                "no session log names this arm"),
        uncovered=_UNCOVERED, capture_timing=timing,
        note=f"{_NOTE} This one {where}; off the codec it peaked "
             f"{stats['peak']:.3f} at {stats['rms']:.3f} rms{loss}")


def _said(word: sessionlog.Word, zero: int) -> sessionlog.Word:
    """One of the log's own decodes, moved from the stream clock onto this window."""
    return replace(word, at=word.at - zero / FS)


def _anchored(audio: np.ndarray, anchors: list[int]) -> list[Heard]:
    """`_SessionRx.control_signal`'s PACTOR-1 read, at the instants the grid held.

    THE SESSION'S OWN METHOD, called rather than restated. `_p1_cs` reads no
    attribute of the session it hangs off -- it is `p1rx.cs_anchored` with
    `cs_head` behind it and the CS3 refusal in front -- so a replay that recreated
    its two positioners would be a second implementation of the one thing this
    tool exists to hold the first to.

    ONE TRANSMISSION, ONE EVENT, which is `_SessionRx._on`'s own rule: a cycle's
    audio carries two grid instants a millisecond apart -- the prediction and the
    measurement the same cycle wrote down -- and the peer keyed one codeword.
    AGAINST ITS OWN READS ONLY, though, and not against the rolling arm's:
    `cs_seen` would gate this in the session, but the rolling arm HERE is
    deliberately over-permissive -- it pushes every slice where a hold cycle holds
    them -- so letting it suppress the anchor makes the replay deafer than the
    station in a way the station never was. `hold_40` of `onair-0902-2342` is
    that: a sweep accept 147 ms off the slot taking out the changeover head the
    session's log names at the slot.
    """
    out: list[Heard] = []
    for at in anchors:
        if any(abs(at / FS - h.at) < 2 * spec.P1_CS_S for h in out):
            continue
        ev = onair._SessionRx._p1_cs(audio, at)
        if ev is not None and _confirmed(ev):
            out.append(_heard(ev))
    return out


def _verdicts(ungated: list[Heard], rolling: list[Heard], anchored: list[Heard],
              said: list[sessionlog.Word],
              anchors: list[float]) -> dict[Heard, Verdict]:
    """What each ungated hit is, against the station's own record of the instant.

    In this order, and the order is the argument.

    WHAT THE STATION WROTE DOWN BEATS EVERYTHING, and it is the only thing this
    calls SEEN outright. `discarded` used to be its complement, and on
    `onair-0903-1107` that made four CS1 the arm's own log names at the same
    instant into four losses.

    THE PEER'S OWN PACKET COMES SECOND, ahead of both replays, because a sending
    station keys no control signals: a codeword the sweep finds 150 ms inside a
    changeover packet is twelve bits of somebody's data whatever any reader of
    ours would have done with it.

    THE ANCHORED READ COMES THIRD AND THE SWEEP DOES NOT COUNT AT ALL. The
    anchored read is aimed at the instant the grid held and is the session's own
    method run on the session's own number; the rolling arm here pushes every
    slice where a hold cycle holds them, so it decodes strictly more than the
    session did and an accept of its own is not evidence the station heard
    anything. Where it is all there is, the hit is UNSEEN and the reason says so.
    """
    packets = _peer_packets([*rolling, *anchored, *ungated])
    out: dict[Heard, Verdict] = {}
    for h in ungated:
        where = _placed(h, anchors)
        word = next((w for w in said
                     if w.what == h.what and abs(w.at - h.at) <= _SAME_S), None)
        inside = next((t for t in packets if t < h.at <= t + _BREAKIN_S), None)
        if word is not None:
            out[h] = Verdict(SEEN, f"{where}; the arm's log names it there too "
                                   f"({word.text})")
        elif inside is not None:
            out[h] = Verdict(
                IN_PACKET,
                f"{where}; {h.at - inside:.3f} s into the peer's own transmission "
                f"at {inside:.3f} s -- a false accept inside a packet, not a word "
                f"the peer keyed")
        elif _near(h, anchored, _SAME_S):
            out[h] = Verdict(SEEN, f"{where}; the anchored point read, replayed at "
                                   f"the instant this arm's grid held, takes it")
        else:
            out[h] = Verdict(UNSEEN, f"{where}; not in the arm's log" + (
                ", though the rolling replay reaches it -- and that replay is fed "
                "more than the session's was" if _near(h, rolling, _SAME_S) else ""))
    return out


def _peer_packets(heard: list[Heard]) -> list[float]:
    """Where in this window the peer held the channel with a data packet.

    A changeover head is the one that matters and the one the corpus is full of:
    CS3 in the control-signal slot with `P1_PACKET_S` of 100 Bd behind it. An ISS
    sends no control signals, so every codeword-shaped accept in there is the
    reader finding twelve bits of somebody's data -- ten of `onair-0902-2342`'s
    fourteen. A CRC-valid frame either arm decoded is the same statement made
    without the head.
    """
    return sorted({h.at for h in heard
                   if "break-in" in h.detail or h.what.startswith("packet")})


def _placed(h: Heard, anchors: list[float]) -> str:
    """Where a hit sits against the instant the peer's answer was due."""
    if not anchors:
        return f"{h.at:.3f} s in, against no answer slot this window holds"
    at = min(anchors, key=lambda a: abs(a - h.at))
    return f"{(h.at - at) * 1e3:+.0f} ms from the answer slot at {at:.3f} s"


def _rolling(audio: np.ndarray) -> list[Heard]:
    """`_SessionRx`'s rolling decoder, at its settings and fed the way it is fed.

    The sweeping half of the live arm, and the more permissive half: a receiving
    cycle HOLDS its slices and this pushes them, so what comes back is at least
    what the session's rolling decoder had and generally more. `_anchored` is the
    other half and is the one aimed where the station aimed.
    """
    heard: list[Heard] = []

    def on_event(ev) -> None:
        if _confirmed(ev):
            heard.append(_heard(ev))

    rx = live.RollingRx(on_event, window_s=onair.RX_WINDOW_S,
                        keep_s=onair.RX_WINDOW_S - onair.RX_SLIDE_S)
    # The link-up value of the two `_SessionRx.feed` chooses between, which is the
    # permissive one and today is the same integer as the sweeping one.
    rx.cs_max_errors = rxfront.CS_EXPECTED_MAX_ERRORS
    slice_n = int(SLICE_S * FS)
    for i in range(0, len(audio), slice_n):
        rx.push(audio[i:i + slice_n])
    _flush(rx)
    return sorted(heard)


def _flush(rx: live.RollingRx) -> None:
    """`_SessionRx.flush`, which on a receiving cycle is the only decode there is.

    An IRS listens through the peer's whole 0.96 s packet with every slice held
    rather than fed, so the buffer reaches the end of the cycle undecoded -- and
    then only its last `FLUSH_CONTEXT_S` is looked at.
    """
    keep = int(onair.FLUSH_CONTEXT_S * rx.fs)
    if len(rx.buf) > keep:
        drop = len(rx.buf) - keep
        rx.buf = rx.buf[drop:]
        rx.t0 += drop / rx.fs
    rx.flush()


def _ungated(audio: np.ndarray) -> list[Heard]:
    """Every pass rxfront has, over the whole capture at once.

    `acquiring` is on because this is not a stream and cannot key over anybody;
    `p3_envelope` is on because the reason it is off live is a wall clock this
    has none of. Both only ever add, which is the point -- but the accept gate
    does not move: zero bit errors on a control signal and a valid CRC on a frame.
    """
    return sorted(_heard(ev) for ev in rxfront.decode_events(
        audio, cs_max_errors=rxfront.CS_MAX_ERRORS, acquiring=True,
        p3_envelope=True) if _confirmed(ev))


def _confirmed(ev) -> bool:
    if ev.kind not in CONFIRMABLE:
        return False
    return ev.kind != "packet" or bool(ev.packet and ev.packet[3])


def _heard(ev) -> Heard:
    return Heard(round(ev.t, 3), _key(ev), ev.text)


def _key(ev) -> str:
    """The diff key: what was decoded, never how it was worded.

    The wording carries the pass that found it -- "at anchor", "codeword search",
    "100Bd" -- and the two arms take different passes to the same burst by
    construction, so keying on the text would report every recovered control
    signal as a discard and a live-only both.
    """
    if ev.kind == "cs":
        return f"cs CS{ev.cs + 1}" if ev.cs is not None else "cs"
    if ev.kind == "connect":
        return f"connect {ev.connect.callsign}" if ev.connect else "connect"
    sl, status, _payload, _crc = ev.packet
    return f"packet sl{sl}/{status:#04x}"
