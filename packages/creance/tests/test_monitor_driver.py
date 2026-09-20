# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Driver and fan-out, exercised with a stub runner subprocess.

The stub stands in for a modem's runner: it reads s16le on stdin and emits
detection JSON on stdout, exactly the contract the real runners honour, so the
subprocess plumbing, JSON parsing, fan-out and merge are all tested without
numpy or a modem present.
"""

import io
import queue
import struct
import sys
import threading
import time
import wave

from hfhost.audio import FS, AudioSource

from creance.monitor import default_specs
from creance.monitor.activity import CONFIRMED, Detection
from creance.monitor.aggregate import run
from creance.monitor.driver import ModemMonitor, MonitorSpec, _Clock

FRAME = 4800                            # 0.1 s, as hfhost hands frames out


STUB = r'''
import sys, json
seen = 0
fired = set()
stdin = sys.stdin.buffer
while True:
    raw = stdin.read(9600)
    if not raw:
        break
    seen += len(raw)
    for thresh, det in ((20000, ("shrike", "PACTOR-3", "HEADER", "confirmed")),
                        (60000, ("kestrel", "VARA", "CR", "confirmed"))):
        if seen >= thresh and thresh not in fired:
            fired.add(thresh)
            modem, proto, kind, grade = det
            sys.stdout.write(json.dumps(
                {"t": seen / 96000.0, "modem": modem, "protocol": proto,
                 "kind": kind, "grade": grade, "station": "W1AW"}) + "\n")
            sys.stdout.flush()
'''

DEAD = "import sys; sys.exit(0)"

# A runner slower than real time -- the load case that used to stall the reader.
# It reports at its own sample count, which is what the driver maps back onto
# the source's clock.
SLOW = r'''
import sys, json, time
seen = 0
stdin = sys.stdin.buffer
while True:
    raw = stdin.read(9600)
    if not raw:
        break
    time.sleep(0.05)                    # 0.1 s of audio costs 0.05 s of decode
    seen += len(raw) // 2
    sys.stdout.write(json.dumps(
        {"t": seen / 48000.0, "modem": "shrike", "protocol": "PACTOR-3",
         "kind": "HEADER", "grade": "confirmed"}) + "\n")
    sys.stdout.flush()
'''

# Reports its own sample count after every frame, so a detection's position can
# be checked against the source position the driver was handed.
TICKER = r'''
import sys, json
seen = 0
stdin = sys.stdin.buffer
while True:
    raw = stdin.read(9600)
    if not raw:
        break
    seen += len(raw) // 2
    sys.stdout.write(json.dumps(
        {"t": seen / 48000.0, "modem": "shrike", "protocol": "PACTOR-3",
         "kind": "HEADER", "grade": "confirmed"}) + "\n")
    sys.stdout.flush()
'''

# A runner that is behind the stream: it decodes nothing until EOF, then emits
# what it owes. Real runners lag by however much decode they are carrying, and
# under --fast that is most of the file -- so the tail of a capture is emitted
# after the audio has run out, not before.
LAGGARD = r'''
import sys, json, time
sys.stdin.buffer.read()                 # everything, then EOF
time.sleep(0.4)
sys.stdout.write(json.dumps(
    {"t": 9.0, "modem": "shrike", "protocol": "PACTOR-3", "kind": "HEADER",
     "grade": "confirmed", "station": "W1AW"}) + "\n")
sys.stdout.flush()
'''

# Wedges on the audio rather than on startup, which is the fault being modelled:
# a runner that still owes decode when the stream ends and never gives it up. On
# the empty stream an availability probe hands it, it has nothing owing and exits.
WEDGED = "import sys, time\nif sys.stdin.buffer.read(): time.sleep(60)\n"

# A runner whose file is present and whose imports are not. The file existing is
# what `available()` used to answer with, and it is the wrong question: this one
# passes that check, starts, dies before it reads a byte, and leaves the pass
# reporting a quiet channel for the modem it was the only ear on.
UNIMPORTABLE = "import a_module_this_tree_does_not_have\n"


def _write_wav(path, seconds, rate=48000):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", 1000) for _ in range(n)))


def _stub_spec(tmp_path, name, body=STUB):
    runner = tmp_path / f"{name}_runner.py"
    runner.write_text(body)
    return MonitorSpec(name, sys.executable, str(runner), str(tmp_path))


def test_modem_monitor_reads_detections_off_stdout(tmp_path):
    out: "queue.Queue[Detection]" = queue.Queue()
    m = ModemMonitor(_stub_spec(tmp_path, "shrike"), out)
    m.start()
    for i in range(10):
        m.feed(i * 4800, b"\x00" * 9600)
    got = []
    try:
        while len(got) < 2:
            got.append(out.get(timeout=5))
    finally:
        m.stop()
    kinds = {d.kind for d in got}
    assert "HEADER" in kinds
    assert all(d.grade == CONFIRMED for d in got)


def test_dead_runner_is_retired_without_raising(tmp_path):
    out: "queue.Queue[Detection]" = queue.Queue()
    m = ModemMonitor(_stub_spec(tmp_path, "shrike", DEAD), out)
    m.start()
    time.sleep(0.3)
    m.feed(0, b"\x00" * 9600)       # must not raise even though the pipe is gone
    assert m.alive is False
    m.stop()


def test_end_to_end_fan_out_and_merge(tmp_path):
    from hfhost.audio import WavReplaySource

    wav = tmp_path / "cap.wav"
    _write_wav(wav, 1.0)
    specs = [_stub_spec(tmp_path, "shrike"), _stub_spec(tmp_path, "kestrel")]
    out = io.StringIO()
    view = run(WavReplaySource(wav, realtime=False), specs,
               summary_every_s=1000, out=out)      # no periodic summary noise
    s = view.summary(1.0)
    # both stub modems' detections merged onto one view
    assert set(s["protocols"]) == {"PACTOR-3", "VARA"}
    assert s["overall"] == CONFIRMED.replace("confirmed", "active_confirmed")
    text = out.getvalue()
    assert "HEADER" in text and "CR" in text


def test_a_lagging_runner_keeps_its_tail(tmp_path):
    """The end of a capture is where an ARQ exchange resolves — the reply, the
    data phase, the disconnect. Killing a runner at EOF drops exactly that, and
    a truncated sweep is indistinguishable from a quiet band."""
    from hfhost.audio import WavReplaySource

    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.2)
    out = io.StringIO()
    view = run(WavReplaySource(wav, realtime=False),
               [_stub_spec(tmp_path, "shrike", LAGGARD)],
               summary_every_s=1000, out=out)
    assert set(view.summary(9.0)["protocols"]) == {"PACTOR-3"}
    assert "HEADER" in out.getvalue()


def test_a_wedged_runner_is_killed_and_reported_short(tmp_path, capsys):
    """The backstop, and it must be loud: a sweep that lost its tail has to say
    so rather than report a short answer as a complete one."""
    from hfhost.audio import WavReplaySource

    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.2)
    run(WavReplaySource(wav, realtime=False),
        [_stub_spec(tmp_path, "shrike", WEDGED)],
        summary_every_s=1000, drain_timeout=0.5, out=io.StringIO())
    assert "WARNING" in capsys.readouterr().err


def test_no_available_monitor_is_handled(tmp_path, capsys):
    out = io.StringIO()
    missing = MonitorSpec("ghost", sys.executable,
                          str(tmp_path / "nope.py"), str(tmp_path))
    from hfhost.audio import WavReplaySource
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.2)
    view = run(WavReplaySource(wav, realtime=False), [missing], out=out)
    assert view.summary(1.0)["overall"] == "quiet"


# -- backpressure --------------------------------------------------------------

class _PacedSource(AudioSource):
    """Frames on demand, optionally paced, remembering how long the consumer
    took to take them. Stands in for a live capture without opening a device."""

    def __init__(self, frames, *, period: float = 0.0):
        self.n = frames
        self.period = period
        self.read_s = 0.0

    def frames(self):
        start = time.monotonic()
        for i in range(self.n):
            if self.period:
                time.sleep(self.period)
            yield i * FRAME, b"\x00" * (FRAME * 2)
        self.read_s = time.monotonic() - start


def test_a_slow_runner_does_not_stall_the_reader(tmp_path):
    """The defect, in the small.

    A runner slower than its input used to block ``feed``, which stopped the
    aggregate loop reading its source, which backed ffmpeg up, which made
    avfoundation discard audio nobody could count. Measured 2026-08-14 on a live
    pass: 74.6% of real time kept against the recorder's 91.0%. So the reader
    has to run at the source's pace no matter how far behind a runner falls.
    """
    src = _PacedSource(40)              # 4 s of audio, free to read
    run(src, [_stub_spec(tmp_path, "shrike", SLOW)], backlog_s=0.5, live=True,
        summary_every_s=1000, drain_timeout=2.0, out=io.StringIO())
    # the runner owes 40 * 0.05 = 2 s of decode; the reader must not have waited
    # for any of it
    assert src.read_s < 1.0, f"the reader was paced by the runner ({src.read_s:.2f}s)"


def test_audio_a_slow_runner_never_saw_is_counted_and_reported(tmp_path, capsys):
    """A drop here is a real loss and says so. The point is not that it is
    smaller than the loss it replaces — it is that it lands somewhere countable
    instead of upstream in a capture device that reports nothing."""
    run(_PacedSource(40), [_stub_spec(tmp_path, "shrike", SLOW)], backlog_s=0.3,
        summary_every_s=1000, drain_timeout=2.0, live=True, out=io.StringIO())
    err = capsys.readouterr().err
    assert "dropped" in err and "full feed backlog" in err
    assert "quiet on the audio that arrived" in err


def test_a_slow_runner_offline_is_waited_for_rather_than_starved(tmp_path, capsys):
    """Dropping is only right when waiting would lose more. Over a recording it
    would lose nothing at all — there is no capture device to discard anything
    while the reader pauses — so a `--fast` sweep waits and stays complete."""
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 4.0)
    from hfhost.audio import WavReplaySource

    out = io.StringIO()
    run(WavReplaySource(wav, realtime=False),
        [_stub_spec(tmp_path, "shrike", SLOW)], backlog_s=0.3,
        summary_every_s=1000, drain_timeout=10.0, out=out)
    assert "dropped" not in capsys.readouterr().err
    # SLOW reports once per 0.1 s frame it reads, so every frame of the file
    # reached the decoder or the count is short
    assert sum(ln.startswith("[") for ln in out.getvalue().splitlines()) == 40


def test_a_live_pass_always_reports_what_it_kept(tmp_path, capsys):
    """Unconditionally, loss or no loss: a figure printed only when it is bad is
    a figure nobody has ever seen good, and so cannot judge."""
    run(_PacedSource(5), [_stub_spec(tmp_path, "shrike")], summary_every_s=1000,
        drain_timeout=2.0, live=True, out=io.StringIO())
    err = capsys.readouterr().err
    assert "# capture:" in err and "% of real time" in err


def test_a_stop_event_ends_the_pass_and_still_reports_the_capture(tmp_path, capsys):
    """The event `timeout`'s SIGTERM now sets, in place of the default
    disposition that used to kill the process before the summary printed."""
    stop = threading.Event()
    threading.Timer(0.15, stop.set).start()
    run(_PacedSource(100, period=0.05), [_stub_spec(tmp_path, "shrike")],
        summary_every_s=1000, drain_timeout=2.0, live=True, stop=stop,
        out=io.StringIO())
    err = capsys.readouterr().err
    assert "# stopped" in err
    assert "# capture:" in err and "% of real time" in err


def test_a_lossy_capture_is_named_as_a_capture_loss(tmp_path, capsys):
    """Half the audio never arrives: the pass must not read as a quiet channel."""
    run(_PacedSource(10, period=0.2), [_stub_spec(tmp_path, "shrike")],
        summary_every_s=1000, drain_timeout=2.0, live=True, out=io.StringIO())
    err = capsys.readouterr().err
    assert "never reached the decoders" in err
    assert "quiet on the audio that arrived" in err


# -- what the printed position is ----------------------------------------------

def test_a_detection_sits_at_its_source_position_not_the_runners_own_count(tmp_path):
    """A runner counts only the samples it was handed and reports against that.
    The driver puts every detection back on the source's clock, so the number
    that reaches the log is a position in the stream this process received."""
    out: "queue.Queue[Detection]" = queue.Queue()
    m = ModemMonitor(_stub_spec(tmp_path, "shrike", TICKER), out)
    m.start()
    for i in range(5):                  # the stream is already 1.0 s old
        m.feed(FS + i * FRAME, b"\x00" * (FRAME * 2))
    got = [out.get(timeout=5) for _ in range(5)]
    m.stop(2.0)
    assert [round(d.t, 3) for d in got] == [1.1, 1.2, 1.3, 1.4, 1.5]
    assert all(d.wall > 0 for d in got)


def test_the_clock_carries_dropped_audio_across_to_the_source_position():
    """Dropped audio shortens the runner's stream and nothing tells the runner
    so. The mark table is what keeps its timestamps meaning what they say."""
    clock = _Clock()
    for i in range(10):
        if i in (3, 4, 5):
            continue                    # dropped at a full backlog
        clock.mark(i * FRAME, 1000.0 + i * 0.1, FRAME)
    # the runner has been fed 7 frames and calls the last of them 0.6 s
    assert clock.fed == 7 * FRAME
    stream_t, wall = clock.locate(0.6)
    assert round(stream_t, 3) == 0.9    # frame 9 of the source, not frame 6
    assert round(wall, 3) == 1000.9
    # and before the gap the two clocks still agree
    assert round(clock.locate(0.2)[0], 3) == 0.2


def test_default_specs_are_named_and_share_one_interpreter(tmp_path):
    specs = {s.name: s for s in default_specs(tmp_path)}
    assert set(specs) == {"kestrel", "shrike", "besra"}
    # one interpreter for all three, since one distribution carries them all
    assert len({s.interpreter for s in specs.values()}) == 1
    # a base that is not there -> nothing available
    assert not any(s.available() for s in default_specs(tmp_path / "empty"))


def test_a_runner_that_cannot_import_is_not_available(tmp_path):
    """Availability is whether the runner RUNS, not whether its file is on disk.

    Answered by layout, this returns True for a runner that dies on its first
    import, and the aggregator then drops the dead monitor and reports the
    channel as quiet for that modem. A monitor that cannot run and a band with
    nothing on it produce the same summary, which is the one confusion a monitor
    may never cause.
    """
    assert not _stub_spec(tmp_path, "ghost", UNIMPORTABLE).available()
    assert _stub_spec(tmp_path, "shrike").available()


def test_a_runner_that_cannot_import_is_named_rather_than_dropped(tmp_path, capsys):
    """The loss is reported the way every other loss in this pass is."""
    from hfhost.audio import WavReplaySource
    wav = tmp_path / "cap.wav"
    _write_wav(wav, 0.5)
    run(WavReplaySource(wav, realtime=False),
        [_stub_spec(tmp_path, "shrike"), _stub_spec(tmp_path, "ghost", UNIMPORTABLE)],
        out=io.StringIO())
    # The report has to be creance's own statement of the loss, not the runner's
    # traceback arriving on the stderr pump — that only appears for a monitor
    # that got far enough to be launched, which after this fix it does not.
    said = [ln for ln in capsys.readouterr().err.splitlines()
            if "not listening" in ln]
    assert len(said) == 1 and "ghost" in said[0]
    assert "a_module_this_tree_does_not_have" in said[0]
