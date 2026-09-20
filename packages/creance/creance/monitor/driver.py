# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Driving a modem's monitor decoder as an isolated subprocess.

Every decode is numpy work and creance itself stays stdlib-only, so a monitor
runs as a subprocess: creance feeds it raw s16le audio on stdin and reads
normalised detection JSON, one object per line, on stdout. numpy never enters
this process.

The runner on the far end of that pipe (``creance/monitor/runners/<modem>_runner.py``)
is the *only* per-modem code. It is the whole adapter: when a modem's monitor
interface shifts, the edit is to that one file. Everything here is
modem-agnostic.

A monitor that dies must not take the aggregator with it: a broken pipe marks
that monitor dead and the rest keep listening, the same resilience the harness
gives a modem that fails to come up. Neither may a monitor that merely runs slow
take the *stream* with it, which is what the bounded backlog below is for.
"""

from __future__ import annotations

import bisect
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from hfhost.audio import FS, SAMPLE_WIDTH, CommandSource

from .activity import Detection

RUNNERS = Path(__file__).resolve().parent / "runners"
#: source tree holding both the modem distribution and the shared field tools
#: the kestrel runner reads its segmenter from
MODEM_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True, slots=True)
class MonitorSpec:
    """How to launch one modem's runner: which interpreter, which runner file,
    and the working directory it runs in."""
    name: str
    interpreter: str          # python that can import this modem's stack
    runner: str               # runner script path
    cwd: str                  # tree the runner resolves its inputs against
    args: tuple[str, ...] = ()

    def available(self) -> bool:
        return not self.unavailable()

    def unavailable(self) -> str:
        """Why this monitor cannot run, or ``""`` if it can.

        The question is whether the runner RUNS, and only starting it answers
        that. Layout cannot: a runner file is on disk in every tree carrying
        creance, while what it imports is the modem's own business and need not
        be there — one of the three reads its decode path out of a directory the
        publication boundary denies, so a distribution holds its file and not its
        import.

        Answered by layout, such a runner is launched, dies on its first import,
        and is retired, and the pass then reports that modem quiet rather than
        deaf. The resilience above is what makes that silent, so the cost of
        asking early is one short subprocess per modem.
        """
        if not Path(self.cwd).is_dir():
            return f"{self.cwd} is not a directory"
        if not Path(self.runner).exists():
            return f"no runner at {self.runner}"
        if not Path(self.interpreter).exists():
            return f"no interpreter at {self.interpreter}"
        return _probe(self)


#: how long the probe waits for a runner to answer definitively. Only a runner
#: still going at the bound costs the whole wait, and that one is answered
#: `available` anyway, so the bound can never report a working modem missing —
#: it only decides how long a definitive answer is waited for. The three shipped
#: runners settle in 0.06-1.4 s on this station: a failed import is the fastest
#: outcome of the three, since it happens before numpy is loaded.
PROBE_S = 15.0


@lru_cache(maxsize=None)
def _probe(spec: "MonitorSpec") -> str:
    """Start ``spec``'s runner on an empty stream and report what stopped it.

    An empty stdin is immediate EOF, so the runner performs every import it owns,
    builds whatever it holds, flushes and exits — the whole startup path and none
    of the decoding.

    A runner still running at the bound has already done the part being asked
    about, so it passes. Blocking past EOF is a different fault with a different
    remedy: the driver kills it on ``stop`` and the pass is reported short, which
    a probe reading it as a missing modem would mask rather than mend.

    Cached because availability is asked once per spec per process and the answer
    costs a subprocess.
    """
    try:
        proc = subprocess.Popen([spec.interpreter, spec.runner, *spec.args],
                                cwd=spec.cwd, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        return f"runner could not be started: {exc}"
    try:
        err = proc.communicate(timeout=PROBE_S)[1] or b""
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return ""
    if not proc.returncode:
        return ""
    lines = err.decode("utf-8", "replace").strip().splitlines()
    return (f"runner exited {proc.returncode}: "
            f"{lines[-1] if lines else 'no diagnostic'}")


def default_specs(base: str | Path | None = None,
                  deep: bool = False) -> list[MonitorSpec]:
    """The modems creance knows how to monitor. The set is a plain list — adding
    sabir later is one entry.

    The runners use the interpreter creance itself runs under. The Kestrel
    runner additionally needs the development tree's unshipped monitoring tools;
    the startup probe reports it unavailable when those tools are absent.
    ``CREANCE_MODEM_PYTHON`` points them at another interpreter — an external checkout of a modem, or a build with a
    different numpy — and ``base`` (or ``CREANCE_MODEM_ROOT``) moves the tree
    they resolve against.

    ``deep`` asks each runner for its expensive decode (kestrel's wideband over,
    which costs seconds of turbo decoding per burst). It is offline-only: on live
    audio the cost backs the stream up, so the CLI refuses it outside --wav.
    """
    root = Path(base or os.environ.get("CREANCE_MODEM_ROOT") or MODEM_ROOT)
    return [MonitorSpec(name, _interpreter(), str(RUNNERS / f"{name}_runner.py"),
                        str(root),
                        ("--deep",) if deep and name == "kestrel" else ())
            for name in ("kestrel", "shrike", "besra")]


def _interpreter() -> str:
    return os.environ.get("CREANCE_MODEM_PYTHON", sys.executable)


def device_source(device: str) -> CommandSource:
    """Live rig audio, off a runner that opens the card through PortAudio.

    Same arrangement as a decode runner and for the same reason: the library and
    the device table live with the modems, not here. What it replaced was ffmpeg
    opening the card from inside creance, which cost an eighth of every live pass
    — see :mod:`creance.monitor.runners.capture_runner` for the six that measured
    it.
    """
    return CommandSource([_interpreter(),
                          str(RUNNERS / "capture_runner.py"), device])


def _spawn_thread(target) -> threading.Thread:
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


class _Clock:
    """Where a runner's own sample count sits on the source's timeline.

    A runner counts only the samples it was handed, so the moment the feed drops
    any the two clocks separate and stay separated — and the runner has no way
    to know it. Every frame that actually reaches the runner is marked here with
    the source position and the wall time it arrived at, and a runner's
    timestamps are read back against those marks. That is what keeps a
    detection's position on the stream's clock instead of the runner's.
    """

    #: marks kept before the oldest are trimmed; 0.1 s frames, so ~10 minutes.
    #: A detection lags its audio by a decode window, never by minutes.
    KEEP = 6000

    def __init__(self) -> None:
        self.fed = 0                     # samples handed to the runner so far
        self._marks: list[tuple[int, int, float]] = []   # (fed, source_t0, wall)

    def mark(self, t0: int, wall: float, samples: int) -> None:
        self._marks.append((self.fed, t0, wall))
        self.fed += samples
        if len(self._marks) > 2 * self.KEEP:
            del self._marks[:self.KEEP]

    def locate(self, runner_t: float) -> tuple[float, float]:
        """A runner timestamp -> (seconds into the source stream, unix time)."""
        pos = round(runner_t * FS)
        i = bisect.bisect_right(self._marks, pos, key=lambda m: m[0]) - 1
        if i < 0:
            wall = self._marks[0][2] if self._marks else time.time()
            return runner_t, wall
        fed, t0, wall = self._marks[i]
        into = (pos - fed) / FS
        return t0 / FS + into, wall + into


class ModemMonitor:
    """A running monitor subprocess: write audio to it, read detections back.

    Detections land on ``out`` (a shared queue the aggregator drains). Stderr is
    forwarded to this process's stderr so a runner's own diagnostics stay
    visible without polluting the detection stream.

    Audio reaches the runner through a bounded backlog and a writer thread, not
    a direct write. Until 2026-08-14 ``feed`` wrote straight to the runner's
    stdin: a runner that fell behind filled the pipe buffer, the blocking write
    stopped the aggregate loop reading its source, ffmpeg's stdout backed up
    behind it and avfoundation discarded the audio at the top of the chain.
    Measured that day, the live monitor kept 74.6% of real time where the
    recorder on the same input kept 91.0%, and none of the missing 16 points
    appeared anywhere in the output — they were lost in a process that had no
    idea they existed. A bounded backlog cannot conjure that audio back, but it
    moves the loss to the one place able to count it: ``dropped_samples`` here,
    reported by the aggregator at the end of every pass.

    That trade only pays where waiting costs audio, so ``lossy`` says whether it
    does. Over a recording the reader can wait as long as the runner needs and
    lose nothing, and a `--fast` sweep feeds far quicker than any runner decodes:
    dropping there would throw away most of the file to no purpose at all.
    """

    #: how far behind the stream a runner may fall before its backlog is full.
    #: Well past any runner's normal decode lag (0.04-0.4 s to flush at EOF) and
    #: short enough that a monitor never answers "what is on this frequency now"
    #: out of a channel that has since moved on.
    BACKLOG_S = 8.0

    #: how long a lossless feed waits for room before it gives up and drops. Only
    #: a runner that has stopped reading altogether gets that far, and dropping
    #: beats hanging a sweep on it.
    STALL_S = 30.0

    def __init__(self, spec: MonitorSpec, out: "queue.Queue[Detection]", *,
                 backlog_s: float = BACKLOG_S, lossy: bool = True) -> None:
        self.spec = spec
        self.out = out
        self.proc: subprocess.Popen | None = None
        self.alive = False
        #: whether a full backlog drops audio or waits for room; see :meth:`feed`
        self.lossy = lossy
        #: audio this monitor never saw because its backlog was full
        self.dropped_samples = 0
        self._cap = int(backlog_s * FS) * SAMPLE_WIDTH
        self._cv = threading.Condition()
        self._pending: deque[tuple[int, float, bytes]] = deque()
        self._queued = 0
        self._closing = False
        self._clock = _Clock()
        self._reader: threading.Thread | None = None
        self._errpump: threading.Thread | None = None
        self._writer: threading.Thread | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.spec.interpreter, self.spec.runner, *self.spec.args],
            cwd=self.spec.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0)
        self.alive = True
        self._reader = _spawn_thread(self._read)
        self._errpump = _spawn_thread(self._pump_err)
        self._writer = _spawn_thread(self._write)

    def feed(self, t0: int, pcm: bytes) -> None:
        """Queue one audio frame for the runner.

        ``t0`` is the frame's sample index in the source stream, which is the
        clock every detection is reported on.

        What a full backlog does is the whole point, and it turns on one question
        the caller answers with ``lossy``: can the source lose audio while this
        waits? On a live capture it can and does, so the oldest frame is dropped
        and counted — a real loss, reported as one, where waiting would instead
        stall the reader and push a *larger* loss upstream into a device that
        counts nothing. Over a recording nothing is at stake but time, so the
        feed waits for room and the sweep stays complete.
        """
        if not self.alive:
            return
        arrived = time.time()
        with self._cv:
            if not self.lossy:
                self._cv.wait_for(
                    lambda: (self._queued + len(pcm) <= self._cap
                             or not self._pending or not self.alive),
                    timeout=self.STALL_S)
            while self._pending and self._queued + len(pcm) > self._cap:
                _, _, old = self._pending.popleft()
                self._queued -= len(old)
                self.dropped_samples += len(old) // SAMPLE_WIDTH
            self._pending.append((t0, arrived, pcm))
            self._queued += len(pcm)
            self._cv.notify_all()

    def _write(self) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        stdin = self.proc.stdin
        while True:
            with self._cv:
                while not self._pending and not self._closing:
                    self._cv.wait()
                if not self._pending:
                    break
                t0, arrived, pcm = self._pending.popleft()
                self._queued -= len(pcm)
                self._clock.mark(t0, arrived, len(pcm) // SAMPLE_WIDTH)
                self._cv.notify_all()          # a lossless feed may be waiting
            try:
                stdin.write(pcm)
                stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._retire()
                return
        try:
            stdin.close()          # EOF: the runner flushes what it still owes
        except OSError:
            pass

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue                       # a stray print is not a detection
            obj.setdefault("modem", self.spec.name)
            try:
                d = Detection.from_json(obj)
            except (KeyError, ValueError, TypeError):
                continue
            with self._cv:
                stream_t, wall = self._clock.locate(d.t)
            self.out.put(replace(d, t=stream_t, wall=wall))
        self._retire()

    def _retire(self) -> None:
        """Mark this monitor dead and wake anything waiting on it, so a feed
        holding out for backlog room does not wait on a runner that has gone."""
        with self._cv:
            self.alive = False
            self._cv.notify_all()

    def _pump_err(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        prefix = f"[{self.spec.name}] "
        for line in self.proc.stderr:
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                print(prefix + text, file=sys.stderr, flush=True)

    def stop(self, grace: float = 0.0) -> bool:
        """Drain the backlog, send EOF, and give the runner ``grace`` seconds in
        total to finish on its own.

        A runner is always behind the stream by its own decode window, and
        anything it has not yet emitted when it is killed is lost. That loss is
        not a random sample: it is the end of the capture, which for an ARQ
        recording is the answering station's reply, the data phase and the
        disconnect. What it needs is only the tens of milliseconds to flush
        that window on EOF — measured at 0.04-0.4 s for all three runners on a
        five-minute file — so ``grace`` is a backstop for a wedged runner, not
        an expected wait.

        The writer thread owns stdin and closes it once the backlog is out, so
        the deadline covers the drain and the flush together; closing it from
        here instead would race a thread that may be blocked writing to it.

        Returns False if the deadline ran out and the runner had to be killed,
        so the caller can report a short sweep instead of letting it read as a
        quiet band.
        """
        self.alive = False
        if self.proc is None:
            return True
        deadline = time.monotonic() + grace
        with self._cv:
            self._closing = True
            self._cv.notify_all()
        if self._writer:
            self._writer.join(timeout=max(0.0, deadline - time.monotonic()))
        clean = True
        try:
            self.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            clean = False
            self.proc.terminate()          # also unblocks a writer mid-write
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._reader:
            self._reader.join(timeout=2)   # everything it read is on the queue
        return clean
