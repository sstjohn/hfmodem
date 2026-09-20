# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""One live audio source, fanned to every modem monitor, merged into one view.

The loop is deliberately plain: pull a frame from the source, hand the same
bytes to every monitor, drain whatever detections have come back, and every so
often print the structured summary. The audio clock paces everything — a live
source blocks at real time, a WAV replay paces itself — so no timing lives here.

On a live source, handing a frame to a monitor never waits on it (see
:class:`ModemMonitor`): a runner that falls behind drops audio at its own
backlog rather than stalling this loop, because a stall here backs the source up
and the audio is then discarded upstream, out of reach of anything that could
count it. Over a replay the same stall costs nothing but time, so there the feed
waits and the sweep stays complete. What a pass lost, at the capture and at each
backlog, is printed when it ends — a monitor that reports nothing is read as a
quiet channel, so the seconds it never heard have to sit next to that verdict.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Iterable

from hfhost.audio import FS, AudioSource, WavReplaySource

from .activity import (STREAM_CLOCK, WALL_CLOCK, ActivityView, Detection,
                       format_detection, render_summary)
from .driver import ModemMonitor, MonitorSpec

_CLOCK_NOTE = {
    WALL_CLOCK: "local wall clock, when the audio arrived",
    STREAM_CLOCK: "seconds of audio into the stream, not wall time",
}


def run(source: AudioSource, specs: Iterable[MonitorSpec], *,
        window_s: float = 20.0, summary_every_s: float = 10.0,
        drain_timeout: float = 120.0, backlog_s: float = ModemMonitor.BACKLOG_S,
        live: bool | None = None, stop: threading.Event | None = None,
        out=sys.stdout) -> ActivityView:
    """Monitor ``source`` with every spec in ``specs`` until the source ends or
    the operator interrupts, printing a live confidence-graded log and periodic
    summaries. Returns the final :class:`ActivityView` (handy for tests)."""
    # A monitor that cannot run and a band with nothing on it produce the same
    # summary, so the set that was asked for and the set that is listening are
    # reported apart. Narrowing this silently is how a pass reports a modem quiet
    # on a channel it never had an ear on.
    asked = [(s, s.unavailable()) for s in specs]
    for spec, why in asked:
        if why:
            print(f"# {spec.name} is not listening on this pass: {why}",
                  file=sys.stderr, flush=True)
    specs = [s for s, why in asked if not why]
    if not specs:
        print("no modem monitor is available", file=sys.stderr)
        return ActivityView(window_s=window_s)
    if live is None:
        # One question decides both the stamp and the backlog policy: is the
        # source producing this audio right now? If it is, waiting loses samples
        # and the wall clock is the only position that survives them. A replay
        # loses nothing while it waits, and its wall clock would record when the
        # file was read rather than when the audio happened.
        live = not isinstance(source, WavReplaySource)
    clock = WALL_CLOCK if live else STREAM_CLOCK

    detections: "queue.Queue[Detection]" = queue.Queue()
    monitors = [ModemMonitor(s, detections, backlog_s=backlog_s, lossy=live)
                for s in specs]
    view = ActivityView(window_s=window_s)

    for m in monitors:
        m.start()
    print("# creance monitor  |  " + ", ".join(s.name for s in specs)
          + f"  |  window {window_s:g}s  |  position: {_CLOCK_NOTE[clock]}",
          file=out, flush=True)

    next_summary = summary_every_s
    stream_t = 0.0
    started = time.monotonic()
    warned_dropping = False
    interrupted = False
    try:
        for t0, pcm in source.frames():
            stream_t = (t0 + len(pcm) // 2) / FS
            for m in monitors:
                m.feed(t0, pcm)
            _drain(detections, view, clock, out)
            if not warned_dropping and any(m.dropped_samples for m in monitors):
                warned_dropping = True
                print("# WARNING: a monitor is behind the stream and is dropping "
                      "audio it will never decode — see the accounting at the end "
                      "of this run", file=sys.stderr, flush=True)
            if stream_t >= next_summary:
                next_summary = stream_t + summary_every_s
                _emit_summary(view, stream_t, out)
            if not any(m.alive for m in monitors):
                print("# all monitors stopped", file=sys.stderr)
                break
            if stop is not None and stop.is_set():
                print("# stopped", file=sys.stderr)
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n# stopped", file=sys.stderr)
    finally:
        # The runners are behind the stream by however much decode they owe, so
        # the end of the stream is not the end of the detections. Wait for them
        # rather than for a fixed interval -- especially under --fast, where the
        # backlog is largest and the bulk corpus sweeps run.
        grace = 0.5 if interrupted else drain_timeout
        short = [m.spec.name for m in monitors if not m.stop(grace)]
        _drain(detections, view, clock, out)
        if short and not interrupted:
            print(f"# WARNING: {', '.join(short)} still had audio to decode after "
                  f"{grace:g}s and was killed — detections at the end of this "
                  "stream are missing", file=sys.stderr)
        _emit_summary(view, stream_t, out)
        _report_losses(monitors, stream_t, time.monotonic() - started, live)
    return view


def _drain(q: "queue.Queue[Detection]", view: ActivityView, clock: str,
           out) -> None:
    while True:
        try:
            d = q.get_nowait()
        except queue.Empty:
            return
        view.add(d)
        print(format_detection(d, clock=clock), file=out, flush=True)


def _emit_summary(view: ActivityView, t_now: float, out) -> None:
    print("  == " + render_summary(view.summary(t_now)) + " ==",
          file=out, flush=True)


def _report_losses(monitors: list[ModemMonitor], audio_s: float, wall_s: float,
                   live: bool) -> None:
    """What this pass did not hear, said out loud.

    Audio goes missing in two places and neither used to leave a trace. The
    capture drops it before this process sees it — and a runner that falls behind
    drops it at its own backlog. Both make a channel look quieter than it was,
    and this project has already published one "no handshake captured" that was
    really "no handshake in the fraction we kept".

    This line is also what caught the capture itself. Six live passes on three
    days read 87.0-87.6% while every session recording beside them was whole,
    which is what moved the card off ffmpeg's avfoundation input and onto
    PortAudio; the reading is kept because no capture library retires the
    question.
    """
    lines = []
    lossy = False
    # Comparing audio against the wall clock only measures the capture when the
    # source is paced by real time; a --fast replay outruns it by design.
    if live and wall_s > 0:
        kept = audio_s / wall_s
        lines.append(f"# capture: {audio_s:.1f} s of audio over {wall_s:.1f} s "
                     f"wall = {kept:.1%} of real time")
        if kept < 0.99:
            lossy = True
            lines.append(f"#   {wall_s - audio_s:.1f} s of this channel never "
                         "reached the decoders — the input device dropped it "
                         "(see creance/monitor/runners/capture_runner.py)")
    for m in monitors:
        lost = m.dropped_samples / FS
        if lost:
            lossy = True
            share = lost / audio_s if audio_s else 0.0
            lines.append(f"# {m.spec.name}: dropped {lost:.1f} s of audio at a "
                         f"full feed backlog ({share:.1%} of the stream) — it "
                         "was behind and decoded none of it")
    if lossy:
        lines.append("# a quiet result from this pass is quiet on the audio that "
                     "arrived, not on the channel")
    for line in lines:
        print(line, file=sys.stderr, flush=True)
