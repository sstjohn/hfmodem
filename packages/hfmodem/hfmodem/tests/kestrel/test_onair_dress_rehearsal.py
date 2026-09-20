# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The whole runbook, armed, against a simulated station.

``tools/onair_session.py`` is what the operator runs at the radio, and until now the
only exercise it had ever had was ``--dry-run`` against the real rig. Since that
rehearsal the demodulator, the burst segmenter, the ARQ state machine and the
response-recovery path have all moved. This runs the tool itself — as a subprocess,
through its own command line, with nothing imported and nothing monkeypatched inside
it — against two stand-ins:

  * :class:`~hfmodem.tests.kestrel.fake_rigctld.FakeRigctld` on an ephemeral loopback port,
    answering ``\\dump_caps``, ``f``, ``F``, ``m``, ``M``, ``t`` and ``T`` the way the
    FT-891 does, and misbehaving on request. **Nothing here may touch 4533**: there
    is a real transmitter on the end of it.
  * ``kestrel/tests/kestrel/fakeaudio/sounddevice.py``, put at the front of the child's
    ``PYTHONPATH``, which gives the session a channel whose contents the test picks —
    band noise, a steady carrier, a real off-air recording, or a listening kestrel
    standing in for the gateway.

The far end is a real :class:`~hfmodem.kestrel.vara.vara_arq.VaraStationHandshake`, so a
connect completes here only when the tool synthesises a real connect-request, the
station demodulates it, and the tool recovers the station's real answer through its
own gate, preamble locator and state machine. Nothing in the harness can say
"connected" on the tool's behalf, and
:func:`test_a_channel_that_never_answers_is_an_honest_no_connect` is the check that
it cannot: the same audio path, the same flags, a station that does not answer, and a
session that must say so.

What the exit codes mean, and where each is pinned:

    0  connected                          the gateway answered and the link came up
    1  no connect                         nothing answered, or the operator interrupted
    2  refused before anything transmits  no targets, no keying line, wrong rig,
                                          mode not confirmed
    3  PTT was already up                 the session stops rather than call over it
    4  the attempt never reached the air  our software failed, not a silent gateway
    5  the receiver cannot be trusted     deaf or muted input; no target can work
    6  the rig has been retired           an unkey never confirmed — not a
                                          per-target condition, so the run stops

Every case asserts where PTT ended up, read off the fake daemon rather than off the
session's own printout — except the one where the daemon stops answering, which
cannot be read and where what is asserted instead is that the session says so loudly.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora, fake_rigctld
from hfmodem.tests.kestrel.fake_rigctld import FakeRigctld
from hfmodem.tests.kestrel.test_station_id import _decode
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK

S = corpora.harness("onair_session")
KC = corpora.harness("kestrel_connect")
RIG = corpora.harness("vara_rig_bridge")
from hfmodem.core import cwid as SID

_TOOLS = corpora.TOOLS
_FAKE_AUDIO = Path(__file__).resolve().parent / "fakeaudio"

pytestmark = pytest.mark.skipif(not (_FAKE_AUDIO / "sounddevice.py").exists(),
                                reason="test harness not present (installed-wheel run)")

MYCALL = "W9SSJ"
#: The keying line every run here names, because an armed session refuses to start
#: without one: the unkey of last resort — RTS cleared on the line itself, no daemon
#: in the loop — has nothing to pull down otherwise. The gate stats the path and
#: never opens it, so /dev/null stands in for the adapter without a port being held.
#: It is not a line that keys: the ioctl comes back ENOTTY, which is why the deaf
#: daemon below still ends in TRANSMITTER MAY BE STUCK.
PTT_LINE = "/dev/null"
#: A line that moves no pin is also why every run here asks for the daemon. The
#: session's own default is the opposite — each attempt drives the keying line
#: and rigctld only tunes, the shape the 2026-08-13 wire records argue for — but
#: that shape wants a line with a pin on the end of it, and `LineKeyer.arm`
#: refuses /dev/null before the first burst. What is rehearsed below is the
#: session; the shape the operator gets is pinned as text by
#: :func:`test_the_attempt_is_armed_on_the_keying_line`, which is what keeps this
#: opt-out from quietly becoming the default again.
DAEMON_PTT = "--no-line-ptt"
# Two 40 m channels whose emission stays inside the data segment once the 1500 Hz
# dial offset is applied: 7101.0 -> dial 7099.5, 7103.5 -> dial 7102.0.
TARGETS = (("W1AAA", 7101.0), ("W1BBB", 7103.5))


def _csv(tmp_path: Path, *rows) -> Path:
    p = tmp_path / "gateways.csv"
    p.write_text("Callsign,Frequency,GridSquare,Hours,Mode\n"
                 + "".join(f"{c},{f},EN62,,VARA\n" for c, f in (rows or TARGETS)))
    return p


class Session:
    """One run of ``onair_session.py``, and everything it did on the way."""

    def __init__(self, proc: subprocess.Popen, server: FakeRigctld, events: Path,
                 record: Path):
        self.proc, self.server, self._events = proc, server, events
        self.record = record
        self.out = ""
        self.rc: int | None = None

    @property
    def recorded(self) -> list[Path]:
        """Everything the session left behind, under its one timestamped directory."""
        dirs = sorted(p for p in self.record.glob("*") if p.is_dir())
        assert len(dirs) <= 1, f"a session made more than one directory: {dirs}"
        return sorted(dirs[0].iterdir()) if dirs else []

    def wait(self, timeout: float) -> Session:
        try:
            self.out = self.proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            self.out = corpora.stop_tool(self.proc)
            raise AssertionError(f"the session never finished:\n{self.out[-4000:]}") from None
        self.rc = self.proc.returncode
        return self

    def events(self, kind: str | None = None, pid: int | None = None) -> list[dict]:
        if not self._events.exists():
            return []
        rows = [json.loads(ln) for ln in self._events.read_text().splitlines() if ln.strip()]
        return [e for e in rows
                if (kind is None or e["event"] == kind) and (pid is None or e["pid"] == pid)]

    @property
    def cat(self) -> list[str]:
        return list(self.server.commands)

    def __repr__(self) -> str:
        return f"exit {self.rc}\n{self.out}"


@pytest.fixture
def rigctld():
    yield from fake_rigctld.serving()


@pytest.fixture
def session(tmp_path):
    """Launch the tool. Anything still running when the test ends is stopped, and
    stopped with the attempt it spawned rather than around it."""
    started: list[Session] = []

    def start(server: FakeRigctld, *, audio: dict | None = None, gateways=(),
              wait: float | None = 240.0, ptt_device: str | None = PTT_LINE,
              **flags) -> Session:
        events = tmp_path / "audio-events.jsonl"
        cfg = {"log": str(events), **(audio or {})}
        env = {**os.environ,
               "PYTHONPATH": os.pathsep.join([str(_FAKE_AUDIO), str(corpora.PKG_ROOT)]),
               "KESTREL_FAKE_AUDIO": json.dumps(cfg)}
        # Recording is on by default and writes under `logs/` — which is the point
        # of it, and is not somewhere a test may write, so every run here is pointed
        # at its own tmp_path.
        record = tmp_path / "recordings"
        argv = [sys.executable, str(_TOOLS / "onair_session.py"),
                "--mycall", MYCALL, "--band", "40", "--grid", "EN63",
                "--gateways", str(_csv(tmp_path, *gateways)),
                "--rigctld", f"127.0.0.1:{server.port}",
                "--device", "fake", "--listen", "5", "--record", str(record),
                DAEMON_PTT,
                *(("--ptt-device", ptt_device) if ptt_device else ())]
        for k, v in flags.items():
            argv.append("--" + k.replace("_", "-"))
            if v is not True:
                argv.append(str(v))
        s = Session(corpora.launch(argv, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env,
                                   cwd=str(corpora.REPO)), server, events, record)
        started.append(s)
        return s.wait(wait) if wait else s

    yield start
    for s in started:
        if s.proc.poll() is None:
            corpora.stop_tool(s.proc)


def _keyed(server: FakeRigctld) -> int:
    return server.seen("T 1")


def _qsy_targets(cat: list[str]) -> list[int]:
    return [int(c.split()[1]) for c in cat if c.startswith("F ")]


# -- the harness itself ------------------------------------------------------------

def test_the_rehearsal_cannot_reach_a_real_audio_device(tmp_path):
    """The one assumption everything else rests on.

    If the substitution ever stopped taking, this module would open the station's
    codec — which another modem may be holding — and every result below would be
    about a channel nobody chose. So the child's ``import sounddevice`` is resolved
    with the same environment the session gets, and checked to land here.
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(_FAKE_AUDIO), str(corpora.PKG_ROOT)]),
           "KESTREL_FAKE_AUDIO": "{}"}
    r = subprocess.run([sys.executable, "-c",
                        "import sounddevice; print(sounddevice.__file__)"],
                       capture_output=True, text=True, env=env, cwd=str(corpora.REPO),
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert Path(r.stdout.strip()).parent == _FAKE_AUDIO, r.stdout


_STANDIN_SESSION = """\
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))
subprocess.Popen([sys.executable, "-c", sys.argv[1]])
print("SESSION UP", flush=True)
time.sleep(3600)
"""
_STANDIN_ATTEMPT = """\
import signal, sys, time
signal.signal(signal.SIGTERM, {on_term})
print("ATTEMPT UP", flush=True)
time.sleep(3600)
"""


def _spawned_by(pid: int, timeout: float = 30.0) -> int:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
        if found.stdout.split():
            return int(found.stdout.split()[0])
        time.sleep(0.05)
    raise AssertionError(f"pid {pid} never spawned anything")


def _gone(pid: int, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize("on_term,unkeys", [
    ("lambda *a: (print('ATTEMPT UNKEYED', flush=True), sys.exit(0))", True),
    ("signal.SIG_IGN", False),
], ids=["attempt-unkeys", "attempt-ignores-sigterm"])
def test_stopping_a_session_is_bounded_and_reaches_the_attempt(tmp_path, on_term, unkeys):
    """The stop every run in this file ends on, against a session that spawns an
    attempt the way the real one does.

    A bare ``kill()`` reaches the session and leaves whatever it started: an attempt
    keying with nobody reading it, and — for as long as it holds any of its parent's
    output — a ``communicate()`` waiting on a pipe that will never close. That is not
    a failing test but a run with no children, no output and no end. So the group is
    signalled, SIGTERM before SIGKILL, and every read is bounded.
    """
    tool = tmp_path / "session.py"
    tool.write_text(_STANDIN_SESSION)
    proc = corpora.launch([sys.executable, str(tool),
                           _STANDIN_ATTEMPT.format(on_term=on_term)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    attempt = _spawned_by(proc.pid)

    t0 = time.monotonic()
    out = corpora.stop_tool(proc, grace=2.0)
    took = time.monotonic() - t0

    assert took < 10.0, f"the stop took {took:.1f} s, which is a run waiting on a pipe"
    assert _gone(attempt), "the attempt outlived the session and is still keying"
    assert ("ATTEMPT UNKEYED" in out) == unkeys, (
        f"SIGTERM did not reach the attempt itself:\n{out}")


def test_the_attempt_is_armed_on_the_keying_line():
    """The argv the session hands the connect tool, read as text — which is the
    one thing the runs below cannot show, because they opt out of it.

    On 2026-08-13 every armed VARA attempt at this station keyed ~8.4 s of
    unintended carrier, and the forensics were one shape twelve times over: the
    socket connects in under a millisecond, ``T 0`` is accepted, and nothing
    comes back — while the keying line, tried afterwards, read LOW instantly,
    eight of eight. An attempt is the transmission that must work while the
    antenna is radiating, so it drives the line itself and rigctld is left to
    tune. The default carries that, rather than a flag an operator can forget:
    a daemon-keyed armed session here is 0-for-N by construction.
    """
    src = (_TOOLS / "onair_session.py").read_text()
    cmd = src.split('"kestrel_connect.py"')[1].split("]")[0]
    for flag in ("--arm", "--expect-model", "--ptt-device", "--line-ptt"):
        assert flag in cmd, f"the attempt is no longer launched with {flag}"
    decl = [ln for ln in src.splitlines() if 'add_argument("--line-ptt"' in ln]
    assert decl and "default=True" in decl[0], (
        "--line-ptt no longer defaults on, so an armed session keys through the "
        "daemon whenever a flag is forgotten")
    assert "ptt_type=RTS" not in src and "ptt_pathname" not in src, (
        "the session hands rigctld the keying line as well — two owners of one "
        "transmitter is how it keys unasked")


# -- the sequence ------------------------------------------------------------------

def test_the_rehearsal_walks_every_target_without_keying(rigctld, session):
    """A dry run, end to end, over two targets — and the transmitter stays cold.

    This is the sequence the runbook promises, read off the wire rather than off the
    session's own narration: tune to the published centre less 1500 Hz, read the dial
    back, assert the data mode *after every QSY* (a Yaesu restores the mode last used
    on the band it is tuned to, so a mode confirmed on one channel says nothing about
    the next), check PTT is down, and listen before deciding anything.
    """
    server = rigctld()
    s = session(server, dry_run=True)
    assert s.rc == 1, s

    assert s.cat == ["\\dump_caps", "M PKTUSB 2400", "m",
                     "F 7099500", "f", "M PKTUSB 2400", "m", "t",
                     "F 7102000", "f", "M PKTUSB 2400", "m", "t",
                     # ... and the transmitter is read back twice on the way out:
                     # once by the shutdown, once for the line the operator reads.
                     "t", "t"], s

    # The dial, not the published channel. Getting this wrong is silent and kills
    # both directions: our bursts above where the gateway listens, its answer under
    # the SSB filter.
    assert _qsy_targets(s.cat) == [S.dial_hz(7101000), S.dial_hz(7103500)]
    assert "7101.000" in s.out and "7099.500" in s.out, "both numbers must be printed"

    assert len(s.events("rec")) == 2, "the channel was not sensed once per target"
    assert all(e["secs"] == pytest.approx(5.0, abs=0.1) for e in s.events("rec"))
    assert s.out.count("channel sense:") == 2
    assert s.out.count("(would call here)") == 2
    # Once, at the end — not once per target. Morse between connect requests is a
    # transmission out of turn.
    assert s.out.count("would identify: W9SSJ in Morse") == 1

    assert _keyed(server) == 0, "a dry run reached the key line"
    assert not s.events("play"), "a dry run put audio into the transmitter"
    assert server.ptt == 0


def test_a_channel_it_cannot_call_over_is_skipped(rigctld, session):
    """Armed, so the only thing between the tool and the air is its own verdict.

    A steady carrier is somebody else's transmission: skipped, on every target, and
    nothing keyed.
    """
    server = rigctld()
    s = session(server, audio={"bed": "carrier",
                               "station": {"call": "W1AAA", "bw": "2300"}},
                arm=True, expect_model="FT-891", per_target=20)
    assert s.rc == 1, s
    assert s.out.count("BUSY, skipping") == 2, s
    assert "RECEIVER" not in s.out, "called a live occupied channel a broken receiver"
    assert _keyed(server) == 0, "keyed on a channel it had just refused to call"
    assert server.ptt == 0


def test_a_deaf_receiver_stops_the_session_instead_of_reading_as_a_busy_band(rigctld,
                                                                             session):
    """The most dangerous input the sense has, and until now the quietest failure.

    Silence below the codec floor is not a quiet band — it is a receiver that is not
    hearing the radio. The occupancy check has no verdict to give on it, which the
    bias resolves as busy with a margin of exactly ``+0.0`` dB, and that prints the
    same line a genuinely occupied frequency prints. The dress rehearsal spent a whole
    run that way: ``channel sense: +0.0 dB against the occupancy thresholds -> BUSY,
    skipping`` on every target, which
    reads as an unlucky band and is really a codec that is not attached.
    ``kestrel_connect`` has the check and never gets to run it, because the session
    skips each target before the child is ever started.

    So the session makes the call itself, says which of the two it is, and stops —
    every remaining target would fail identically and take the window with them.
    """
    server = rigctld()
    s = session(server, audio={"bed": "quiet",
                               "station": {"call": "W1AAA", "bw": "2300"}},
                arm=True, expect_model="FT-891", per_target=20)
    assert s.rc == 5, s
    assert "RECEIVER DEAF" in s.out, s
    assert "BUSY, skipping" not in s.out, "still reporting a dead codec as a busy band"
    assert len(_qsy_targets(s.cat)) == 1, "carried on to a target that could not work"
    assert _keyed(server) == 0 and server.ptt == 0


def test_a_deaf_receiver_is_not_something_force_can_overrule(rigctld, session):
    """``--force`` is the operator saying the band sounds workable to *him*. It is a
    judgement about the channel, and a receiver at the codec floor is not a channel:
    nothing he can hear says our input is attached, and an attempt into a dead
    receiver cannot hear the answer it is waiting for."""
    server = rigctld()
    s = session(server, audio={"bed": "quiet"}, arm=True, expect_model="FT-891",
                per_target=20, force=True)
    assert s.rc == 5, s
    assert "RECEIVER DEAF" in s.out and "--force does not make a deaf receiver hear" in s.out
    assert _keyed(server) == 0 and server.ptt == 0


# -- a gateway that answers, and one that does not ---------------------------------

def test_a_gateway_that_answers_produces_a_connect(rigctld, session):
    """The armed runbook, both targets, to the first connect and no further.

    W1AAA is called and does not answer, because the only station on the channel is
    listening as W1BBB — so the first target exercises call, no answer, next target.
    W1BBB then answers: CR out over the audio path, connect-response back,
    link-setup out, connected-ack back, CONNECTED reported, stop.

    Nothing in the harness decides that. The far end is a listening
    :class:`VaraStationHandshake` which demodulates what the tool really transmitted
    and answers with real synthesised bursts; the tool recovers them through its own
    gate, preamble locator and state machine.
    """
    server = rigctld()
    s = session(server, audio={"station": {"call": "W1BBB", "bw": "2300"}},
                arm=True, expect_model="FT-891", per_target=30, wait=400)
    assert s.rc == 0, s
    assert "W1AAA did not answer" in s.out, s
    assert "*** CONNECTED to W1BBB ***" in s.out, s

    # Both targets tuned, in order, and nothing beyond the one that answered.
    assert _qsy_targets(s.cat) == [S.dial_hz(7101000), S.dial_hz(7103500)]

    # The far end really heard us and really answered.
    assert [e["n"] for e in s.events("station_answers")][:2] == [1, 2], s
    assert s.events("station_connected"), "the simulated station never came up"

    # The session must never end without the operator being told which way the
    # link went. Here the close burst goes out and the local session ends, so
    # the verdict is the clean close, said in as many words — by the connect
    # tool and by the session — and never the left-open line, whose arm is
    # pinned by the deaf-peer counterexample in test_mail.
    #
    # What it may not say is that the far end answered it. Stock 4.9.0 does not
    # acknowledge a disconnect burst (spec 05 §5.6, measured 2026-08-26), so
    # `close_verdict` reports local closure and leaves peer receipt unclaimed;
    # this line used to read "the close was answered and W1BBB released the
    # session", which the simulated responder here cannot establish either.
    assert "DISCONNECTED: clean close" in s.out, s
    assert "Local session closed; disconnect sent to W1BBB" in s.out, s
    assert "Peer receipt is unconfirmed" in s.out, s
    assert "NOT DISCONNECTED" not in s.out, s

    # And the link being down, the §97.119 identification is keyed rather than
    # withheld: Morse into a live ARQ session is a transmission out of turn,
    # but this session no longer holds one.
    assert "NOT identifying" not in s.out, s
    assert s.events("play", pid=s.proc.pid), (
        "no identification was keyed after a cleanly closed session")

    assert _keyed(server) > 0, "a connect that never keyed anything"
    assert server.ptt == 0, "left the transmitter keyed after a successful connect"


def test_a_session_records_everything_it_heard(rigctld, session):
    """The change with the most to say for itself, because the last session proved it.

    2026-07-26 is diagnosable only because the operator happened to be recording by
    hand — and what that recording turned out to hold was a gateway answering us
    twice while the tool reported silence. Neither tool wrote a byte of audio, and
    the session printed the last ten lines of each attempt and dropped the rest.

    So: the listen window of every target, the receive audio of every attempt, and
    every line each attempt printed, under one timestamped directory the session
    names on its way out.
    """
    server = rigctld()
    s = session(server, gateways=(TARGETS[0],), arm=True, expect_model="FT-891",
                per_target=20, wait=300)
    assert s.rc == 1, s

    kept = {p.name: p for p in s.recorded}
    listens = [p for n, p in kept.items() if n.endswith("-listen.wav")]
    attempts = [p for n, p in kept.items() if n.endswith(".wav") and p not in listens]
    logs = [p for n, p in kept.items() if n.endswith(".log")]
    assert len(listens) == 1, f"the listen window was not kept: {sorted(kept)}"
    assert len(attempts) == 1, f"the attempt's receive audio was not kept: {sorted(kept)}"
    assert len(logs) == 1, f"the attempt's output was not kept: {sorted(kept)}"

    with wave.open(str(listens[0])) as w:
        assert w.getframerate() == 48000
        assert w.getnframes() / 48000 == pytest.approx(5.0, abs=0.2)
    with wave.open(str(attempts[0])) as w:
        assert w.getnframes() / 48000 > 10.0, "an attempt's audio stops before it does"

    # The transcript is the whole of what the child said, not the tail the session
    # printed — that is the part that was being thrown away.
    said = logs[0].read_text().splitlines()
    printed = [ln.strip() for ln in s.out.splitlines()]
    assert len(said) > 10
    assert sum(ln.strip() not in printed for ln in said) > 0, (
        "the transcript holds nothing the session had not already printed")
    assert str(logs[0].parent) in s.out, "never said where any of it went"


def test_no_record_is_the_operator_asking_for_nothing_to_be_kept(rigctld, session):
    """It has to be possible, and it has to be loud, because it is the wrong default."""
    server = rigctld()
    s = session(server, gateways=(TARGETS[0],), no_record=True, arm=True,
                expect_model="FT-891", per_target=20, wait=300)
    assert s.rc == 1, s
    assert "NOT RECORDING" in s.out, s
    assert not s.record.exists() or not list(s.record.iterdir()), (
        f"wrote something anyway: {list(s.record.iterdir())}")


# The least the transmitter may stay up past the last sample — a literal, where the
# ceiling below is derived. The hold is what the ceiling is a tolerance around, so
# the ceiling moving with it is right; the floor moving with it means there is
# nothing checking the hold happens at all. Setting `TX_IDLE_HOLD_S` to 0.0 left this
# file and all 25 of `test_ptt_tail_check` green, because every reference to the hold
# in the test tree was derived from it. Measured off the fake daemon's own PTT line,
# four bursts and an identification per run:
#
#     TX_IDLE_HOLD_S = 0.05    PTT drops 0.060-0.066 s past the last sample
#     TX_IDLE_HOLD_S = 0.0     PTT drops 0.0035-0.0086 s past it
#
# so 0.03 s sits between the two populations, and it is a lower bound rather than a
# derivation: a build that stops holding the key idle fails here.
_TAIL_HOLD_MIN_S = 0.03

# ...and the ceiling on the hold ITSELF, which nothing held either. The per-burst
# ceiling below is `KC.TX_IDLE_HOLD_S + _TAIL_MARGIN_S` — a tolerance around the
# hold, so it moves with it, and at a hold of 1.0 s this file stays green with a
# full second of unmodulated carrier after every burst. What bounds the hold is not
# politeness: KC9GHZ answers a turn-request 0.100 s after our last transmitted
# sample, three occurrences over two off-air sessions and the earliest start any
# recording holds (`kestrel_connect.PEER_ANSWERS_FROM_S` carries the whole set).
# A hold that reaches into that is a transmitter keyed over the answer it is
# waiting for, and it fails by looking exactly like a deaf receiver — which is how
# it was read for an evening. A literal for the same reason the floor above is:
# derived from the tool's own constant it would move whenever that did, and stop
# bounding anything.
_EARLIEST_ANSWER_S = 0.100


def test_the_idle_hold_ends_before_a_gateway_can_answer():
    assert _TAIL_HOLD_MIN_S <= KC.TX_IDLE_HOLD_S < _EARLIEST_ANSWER_S, (
        f"an idle hold of {KC.TX_IDLE_HOLD_S:.2f} s is outside "
        f"[{_TAIL_HOLD_MIN_S:.2f}, {_EARLIEST_ANSWER_S:.3f}) s")

# How far past the last sample the transmitter may still be up. The hold itself is
# deliberate and is `KC.TX_IDLE_HOLD_S`; this is what the machinery around it may add.
# Measured against the real station on 2026-07-28: `sd.play(blocking=True)` returns
# 0.20 s past the last sample, a `T 0` round trip to the FT-891 is 0.27 s, and the
# whole `AudioVaraIO.tx` costs 2.17-2.37 s for a 1.750 s burst. So the honest budget
# is ~0.5 s and one second of it is generous — three times what that rig's own
# receive mute showed off air (0.42 s past the audio on each of two clean bursts,
# in logs/onair/20260728T234742Z), and six times under the 7.84 s the third burst of
# that same session held the transmitter up for.
_TAIL_MARGIN_S = 1.0

# And how much unmodulated carrier may precede the first sample. The key-up is one
# `T 1` and nothing else: against this station's FT-891 that is a 0.27 s round trip,
# and both transmit paths build their audio before they key. So 1.0 s is threefold
# generous — and it is half of what was actually there, because `Cat.key` used to
# command PTT and then read it back through a fixed 1.1 s settling wait per command,
# putting a measured 2.21 s of dead carrier on a shared band before the first dit of
# every identification.
_CARRIER_LEAD_S = 1.0

#: Below this, a play carried nothing: `fakeaudio` scores a block of zeros at
#: -240 dBFS, and the quietest thing either process modulates is far above it.
_SILENT_DBFS = -100.0


def _transmissions(s: Session) -> list[dict]:
    """The plays that carried something, and the check that the rest is one thing.

    An armed session writes one block of silence to the transmit device before it
    keys anything, to pay that device's first stream open outside a keyed window
    [`kestrel_connect.AudioVaraIO._warm_transmit_path`]. It reaches the card and
    it is not a transmission: nothing modulated, nothing keyed, nothing on the
    band — so the brackets below, which are about what a transmitter carried, are
    not its to answer.

    Leaving it out is only honest while it stays one silent block ahead of the
    first key-up, so that is asserted here rather than assumed. A warm-up that
    grew modulation is caught by the caller's bracket; one that drifted into the
    keyed region, or a burst that reached the card having lost its modulation, is
    caught right here.
    """
    played = s.events("play")
    silent = [e for e in played if e["dbfs"] <= _SILENT_DBFS]
    assert len(silent) <= 1, (
        f"{len(silent)} silent writes reached the transmit device — the warm-up is "
        f"one block, and anything else silent on the card is a burst that lost its "
        f"modulation on the way out")
    keyed = s.server.stamped("T 1")
    for e in silent:
        assert not keyed or e["t"] < keyed[0], (
            f"silence was written {e['t'] - keyed[0]:.3f} s after the transmitter "
            f"first came up — the warm-up is allowed to reach the card only "
            f"because it is over before the key ever moves")
    return [e for e in played if e["dbfs"] > _SILENT_DBFS]


def test_the_key_brackets_the_audio_closely_at_both_edges(rigctld, session):
    """Both edges of the keyed region, and each of them has cost something.

    The tail is a window. Too short: CoreAudio is still playing when
    ``sd.play(blocking=True)`` returns, so unkeying the moment it returns takes the
    tail of the burst off the air with it. The 1.75 s connect-request survives that —
    gateways have answered it — but the 4.4 s link-setup is a frame under a CRC, and
    a frame missing its tail is discarded by the gateway without a word. Too long: a
    transmitter still up seconds after the audio has stopped is a transmitter nobody
    is modulating, sitting on a shared band. That happened, and this bound is what
    would have caught it.

    The identification is in the window on the same terms as a burst, and it is the
    transmission with the least slack: the drain is ~0.20 s and a dah at the default
    20 WPM is 0.18 s, so an unkey on the return takes the last element of the callsign
    with it. That is not a shortened identification, it is a different one — measured
    here, ``W9SSJ`` went out as ``W9SSW`` — and worse than the hand-sent call it
    replaces.

    The lead has no window at all, only a ceiling: every millisecond between the key
    and the first sample is an unmodulated carrier and none of it is wanted. It is
    checked on the identification as well as on the connect requests, because the
    identification is where it was — 2.21 s of carrier before the first dit, on every
    call this station sent.

    Read off the transport and the daemon rather than off the tool: every burst
    either process played, against when the daemon's PTT line actually moved. And the
    burst itself is untouched — the tail is held, not padded, so nothing at either
    end has to recognise a burst longer than the one it was taught.
    """
    server = rigctld()
    s = session(server, gateways=(TARGETS[0],), arm=True, expect_model="FT-891",
                per_target=20, wait=300)
    assert s.rc == 1, s

    played = _transmissions(s)
    ident = [e for e in played if e["pid"] == s.proc.pid]
    bursts = [e for e in played if e["pid"] != s.proc.pid]
    assert bursts, "the attempt never played anything"
    assert ident, "the session never identified, so the lead is unproven where it was"
    for e in played:
        up = server.raised_before(e["t"])
        assert up is not None, f"audio at {e['t']} went out over an unkeyed transmitter"
        assert e["t"] - up <= _CARRIER_LEAD_S, (
            f"{e['t'] - up:.3f} s of unmodulated carrier before {e['secs']:.2f} s of "
            f"audio, more than {_CARRIER_LEAD_S:.2f} s — a keyed transmitter carrying "
            f"nothing, on a shared band, every time")

    for e in played:
        end = e["t"] + e["secs"]
        down = server.dropped_after(end)
        assert down is not None, f"audio ending at {end} left the transmitter keyed"
        assert down - end >= _TAIL_HOLD_MIN_S, (
            f"PTT dropped {down - end:.4f} s past the last sample, less than the "
            f"{_TAIL_HOLD_MIN_S:.2f} s a deliberate idle hold puts there — either the "
            f"key is being dropped on the device's heels or the hold is gone")
        assert down - end <= KC.TX_IDLE_HOLD_S + _TAIL_MARGIN_S, (
            f"PTT was held {down - end:.3f} s past the last sample, more than "
            f"{KC.TX_IDLE_HOLD_S + _TAIL_MARGIN_S:.2f} s — an unmodulated transmitter "
            f"on a shared band")

    cr = len(MK.synth_burst(TARGETS[0][0], VF.CR)) / 48000
    assert all(e["secs"] == pytest.approx(cr, abs=0.01) for e in bursts), (
        f"the connect-request is {cr:.3f} s and what the transmitter carried was "
        f"{sorted({round(e['secs'], 3) for e in bursts})} s — either the burst was "
        "padded rather than the key held, or its end never left the device")


def test_the_callsign_the_transmitter_carried_is_the_whole_callsign(rigctld, session):
    """§97.119 asks for the callsign, not for most of it.

    The bracket test above says the key was held long enough; this says what came out
    of the other end. The device is the one that knows: the fake card is a fifth of a
    second behind, waits that out on ``stop()`` and throws it away without one, so
    what it parks is what the transmitter carried rather than what the tool handed
    over. Read back through the same envelope detector ``test_station_id`` uses,
    which knows only the keying speed, a clipped final element shows up here as the
    wrong character rather than as a number of milliseconds.

    The drain used to be modelled on this side, as a constant subtracted from a card
    that had no latency at all — so the test could only ever confirm its own
    arithmetic. It is measured on the transport now, which is why `handed_s` is
    asserted too: a transmit path that drains emits everything it was given.
    """
    server = rigctld()
    s = session(server, gateways=(TARGETS[0],), arm=True, expect_model="FT-891",
                per_target=20, wait=300)
    assert s.rc == 1, s

    ident = [e for e in s.events("play") if e["pid"] == s.proc.pid]
    assert len(ident) == 1, f"the session did not identify exactly once: {ident}"
    e = ident[0]
    assert server.dropped_after(e["t"] + e["secs"]) is not None, (
        "the identification left the transmitter keyed")
    assert e["secs"] == pytest.approx(e["handed_s"], abs=0.001), (
        f"{e['handed_s'] - e['secs']:.3f} s of the identification never left the "
        "device — the transmit path unkeyed without draining it")

    carried = np.fromfile(e["audio"], dtype=np.float32)
    assert _decode(carried, SID.WPM) == MYCALL, (
        f"the transmitter carried {_decode(carried, SID.WPM)!r} of {MYCALL!r} — "
        f"a part-sent callsign identifies somebody else")


def test_an_unkey_the_daemon_swallows_is_retried_and_ends_the_attempt(rigctld,
                                                                      session):
    """2026-07-28, and the reason the operator stopped the session at the radio.

    rigctld took the third connect-request's ``T 0`` and never answered it. The unkey
    then spent its whole six-second budget on that one command — three seconds
    waiting for a reply that never came, three more on the readback — and gave up
    having asked exactly once. Measured off that session's own receive recording, the
    FT-891 stayed keyed 7.84 s past the last sample of a 1.75 s burst, against the
    0.42 s the two bursts before it showed.

    Two things have to be true afterwards. The budget has to buy repeated ``T 0``s
    rather than one long wait — the write is what drops the transmitter and the read
    is only evidence — and nothing may go back on the air over a transmitter nobody
    has confirmed is down. The session then transmitted a fourth connect-request into
    exactly that.

    ``deaf_after`` is set so the child's key-up is answered and its unkey is not:
    eight commands carry the parent through its QSY, mode and PTT checks, and three
    more carry the child through ``\\dump_caps``, ``f`` and ``T 1``.
    """
    server = rigctld(deaf_after=11)
    s = session(server, gateways=(TARGETS[0],), arm=True, expect_model="FT-891",
                per_target=40, wait=300)

    keyed = server.stamped("T 1")
    assert keyed, "the attempt never reached the air"
    tries = [t for t in server.stamped("T 0") if t < keyed[0] + 8.0]
    assert len(tries) >= 3, (              # four by design; three leaves scheduling slack
        f"the six-second unkey budget bought {len(tries)} attempt(s) at T 0 — a "
        f"budget spent waiting on one blocked read is a single try in a retry loop's "
        f"clothes")

    assert "MAY BE STUCK" in s.out, (
        "the stuck transmitter was not reported to the operator while it was stuck")
    assert "the transmitter has been taken down" in s.out, s
    assert "NOT connected (no/!=expected response)" not in s.out, (
        "the verdict reads as a gateway that stayed silent, over a transmitter "
        "this attempt had already lost")
    bursts = [e for e in _transmissions(s) if e["pid"] != s.proc.pid]
    assert len(bursts) == 1, (
        f"{len(bursts)} bursts went out over a transmitter whose unkey was never "
        f"confirmed")

    # The parent's own verdict, not just the child's: a dead transmitter is not
    # a gateway that stayed silent, and this station's operating record is read
    # off exactly this line.
    assert s.rc == 6, s
    assert "W1AAA did not answer" not in s.out, (
        "a transmitter that was never keyed at again was filed as a silent gateway")


def test_a_retired_transmitter_stops_the_whole_run_not_just_the_target(rigctld,
                                                                       session):
    """A rig that cannot key does not get better at the next channel.

    Same defect as the test above — rigctld swallows the third connect-request's
    unkey — but with a second target waiting behind the first. The old behaviour
    moved on to it, over a transmitter it had already lost, and would have logged
    it silent too; every target after the first would have compounded the same
    false verdict.
    """
    server = rigctld(deaf_after=11)
    s = session(server, gateways=TARGETS, arm=True, expect_model="FT-891",
                per_target=40, wait=300)
    assert s.rc == 6, s
    assert "the transmitter has been taken down" in s.out, s
    assert "did not answer" not in s.out, s
    assert len(_qsy_targets(s.cat)) == 1, (
        "called a second target over a transmitter that cannot key")


@pytest.mark.parametrize("station,why", [
    (None, "nobody on the frequency"),
    ({"call": "W1AAA", "bw": "2300", "max_tx": 1}, "answered once, then went quiet"),
], ids=["silent", "no-ack"])
def test_a_channel_that_never_answers_is_an_honest_no_connect(rigctld, session,
                                                              station, why):
    """The tool must not be able to report a connect that did not happen.

    Two ways of not connecting that look alike from the operator's chair, and the
    second is the failure of 2026-07-26: a gateway that sends its connect-response
    and never acknowledges the link-setup. The link-setup really goes out there, so
    it is the path with the most to say for itself and the most to gain from lying.
    """
    server = rigctld()
    audio = {"station": station} if station else {}
    s = session(server, gateways=(TARGETS[0],), audio=audio,
                arm=True, expect_model="FT-891", per_target=25, wait=300)
    assert s.rc == 1, s
    assert "W1AAA did not answer" in s.out, s
    assert "*** CONNECTED" not in s.out, f"claimed a connect ({why}):\n{s}"
    assert "no connect." in s.out
    assert _keyed(server) > 0, "the attempt never reached the air, so this proves nothing"
    assert server.ptt == 0

    # §97.119: the session transmitted and no link was left open, so the callsign is
    # keyed — once, at the end of the communication, after everything else has stopped.
    ident = s.events("play", pid=s.proc.pid)
    assert len(ident) == 1 and ident[0]["secs"] > 1.0, (
        f"the identification did not go out at the end of the session: {ident}")
    assert s.out.count("identifying W9SSJ") == 1
    assert ident[0]["t"] > max(e["t"] for e in s.events("play") if e["pid"] != s.proc.pid), (
        "the callsign was keyed before the last connect request, not after it")


@pytest.mark.parametrize("bed,occupied", [
    ({"wav": corpora.CLEAR_CHANNEL}, False),
    ({"wav": corpora.MONITORED_SESSIONS[1], "start": 8.0, "stop": 28.0}, True),
], ids=["verified-clear", "another-station-mid-session"])
def test_the_channel_sense_decides_on_real_receiver_audio(rigctld, session,
                                                          bed, occupied):
    """Both verdicts, off real off-air recordings rather than off a stand-in.

    A gate that always says clear is a gate that transmits over people, and a gate
    that always says busy never lets the session call at all — so both directions are
    pinned, and they are pinned on audio through this rig and this codec.

    The occupied side is 7096.5 kHz at 16:34 UTC on 2026-08-14, monitored with this
    station's transmitter off: a VARA 500 exchange 600 Hz wide on the registered
    channel, whose two callsigns and mode a second source names. It is the occupant
    this gate exists for — narrower than the 610-2405 Hz our own BW2300 call would
    fill, so a session that judged only its own centre would key straight over it —
    and it reads `shape` 15.06 dB against a threshold of 6.0 in the five seconds
    the session judges.

    Every recording tried before it asked the gate to be certain about band noise
    instead, in three different ways:

      * 2026-07-26 is occupied by nothing but kestrel's own eight transmissions and
        the receiver muting around them. Our own bursts are not an occupant.
      * 7102.0 kHz on 2026-08-09 carries a real station, and it is a narrowband
        feature at 2613 Hz — outside what BW2300 fills, so an armed VARA session is
        entitled to walk past it and does. `test_channel_busy` pins that waiver.
      * 7103.5 kHz of that same slot carries a real station too, and it is under this
        gate's floor: a carrier at 806.8 Hz through the first 2.2 s of the only eight
        seconds the recording can offer, 17.2 dB below the in-band noise. `tone` finds
        it at 808.6 Hz and scores 2.26 against its 5.0, on a clear side that already
        reaches 3.33. What refused the window was `burst` 7.86, which was not that
        carrier at all: 0.25 s of dense atmospheric impulses, lifting the passband
        6.9 dB and 3250-4000 Hz by 21 — above the receiver's SSB filter, where nothing
        on the channel can reach. A crack is not an occupant either.
    """
    if not bed["wav"].exists():
        pytest.skip(f"off-air recording not present ({bed['wav']})")
    server = rigctld()
    s = session(server, gateways=(("KB9MMT", 7101.0),),
                audio={"bed": {**bed, "wav": str(bed["wav"]), "gain": 0.5}},
                arm=True, expect_model="FT-891", per_target=12, wait=300)
    assert s.rc == 1, s
    assert ("BUSY, skipping" in s.out) == occupied, s
    assert (_keyed(server) > 0) == (not occupied), "keyed against its own verdict"
    assert "*** CONNECTED" not in s.out, "manufactured a connect out of band noise"
    assert server.ptt == 0


def test_force_calls_a_channel_the_sense_reports_busy(rigctld, session):
    """The operator's ears outrank the detector, and until now he had no way to say so.

    The occupancy gate is biased toward busy on purpose, and on a noisy evening it
    can skip every target in the window — it reads the frequency of 2026-07-26 as
    busy in every window of the recording. A session with no override spends the
    slot printing verdicts, which has already happened once. So there is an
    override, and it says so at the top of the run and again on the target it is
    overruling, because a session that quietly transmits over somebody is worse
    than one that quietly refuses.
    """
    server = rigctld()
    s = session(server, gateways=(TARGETS[0],),
                audio={"bed": "carrier", "station": {"call": "W1AAA", "bw": "2300"}},
                arm=True, expect_model="FT-891", per_target=25, force=True, wait=300)
    assert "--force IS IN EFFECT" in s.out, s
    assert "BUSY, but --force is set — calling anyway" in s.out, s
    assert "BUSY, skipping" not in s.out
    assert _keyed(server) > 0, "--force is set and it still did not call"
    assert server.ptt == 0


def test_named_targets_are_called_in_the_order_they_are_named(rigctld, session):
    """Distance is a proxy for "might hear us"; a station that has answered is not.

    KB9MMT is the only station that has demonstrably heard this one and answered,
    and it ranks sixth by distance — so with four targets it was never called at
    all. Naming them puts the operator's own evidence ahead of the ranking, and the
    order is his too: here the *further* of the two goes first, which no ranking
    would produce.
    """
    server = rigctld()
    s = session(server, calls="W1BBB,W1AAA", dry_run=True)
    assert s.rc == 1, s
    assert _qsy_targets(s.cat) == [S.dial_hz(7103500), S.dial_hz(7101000)], s
    assert "as named" in s.out


def test_a_named_target_that_is_not_on_the_band_is_said_out_loud(rigctld, session):
    """A typo in a callsign must not quietly become a different session."""
    server = rigctld()
    s = session(server, calls="W1AAA,W9ZZZ", dry_run=True)
    assert s.rc == 1, s
    assert "W9ZZZ is not a 40 m gateway" in s.out, s
    assert _qsy_targets(s.cat) == [S.dial_hz(7101000)]


# -- the arm gate ------------------------------------------------------------------

def test_arm_is_refused_when_there_is_no_keying_line_to_unkey_with(rigctld, session):
    """2026-08-10, and the reason every run in this file now names a line.

    A session was armed with no keying line behind it. rigctld took the unkey and
    never answered; the last resort — RTS cleared on the line itself, which needs no
    daemon and no reply — had no line to clear, and the transmitter sat keyed with no
    modulation while the operator watched it. The rehearsal armed the same way in
    every test here and none of them noticed, because a fallback that is never
    exercised looks exactly like one that works.

    So it is refused where ``--expect-model`` is refused, and for the same reason:
    at the flags, before a rig is identified or a channel is sensed, and never with
    the fallback quietly disabled.
    """
    server = rigctld()
    s = session(server, arm=True, expect_model="FT-891", ptt_device=None, wait=60)
    assert s.rc == 2, s
    assert "--ptt-device" in s.out, s
    assert server.commands == [], f"talked to the rig anyway: {server.commands}"
    assert _keyed(server) == 0 and server.ptt == 0


def test_arm_is_refused_when_the_rig_is_not_the_expected_model(rigctld, session):
    """4533 is the FT-891 here and 4532 is a different radio. Naming the rig is what
    stands between a session and keying the wrong transmitter."""
    server = rigctld(model="IC-7300")
    s = session(server, arm=True, expect_model="FT-891", wait=60)
    assert s.rc == 2, s
    assert "REFUSING TO ARM" in s.out and "IC-7300" in s.out
    assert _keyed(server) == 0 and server.ptt == 0


@pytest.mark.parametrize("kw", [
    pytest.param({"refuse": ("M",)}, id="mode-refused"),
    pytest.param({"passbands": (500, 1800)}, id="filter-too-narrow"),
])
def test_arm_is_refused_when_the_mode_is_not_confirmed(rigctld, session, kw):
    """In a voice mode an FT-891 routes the microphone rather than the USB codec, so
    an unconfirmed mode means every burst would key an unmodulated carrier. The
    narrow-filter case is the other half: a rig that took the mode and gave back a
    1.8 kHz filter cannot pass a 2300 Hz emission."""
    server = rigctld(**kw)
    s = session(server, arm=True, expect_model="FT-891", wait=60)
    assert s.rc == 2, s
    assert "REFUSING TO ARM" in s.out and "mode not confirmed" in s.out
    assert _keyed(server) == 0 and server.ptt == 0


def test_without_arm_the_key_line_is_never_reached(rigctld, session):
    """No ``--arm`` and no ``--dry-run`` either: it still senses and still refuses to
    transmit, which is what makes the flag load-bearing rather than decorative.

    It also asks for a filter the rig does not have. A rig offers a ladder and answers
    with the one it selected, so 3 kHz comes back as the FT-891's widest PKTUSB and
    that is a correct set, not a failed one — the session only requires the filter to
    be no narrower than the emission.
    """
    server = rigctld()
    s = session(server, per_target=20, passband=3000)
    assert s.rc == 1, s
    assert "NOT ARMED" in s.out
    assert s.out.count("(would call here)") == 2
    assert "M PKTUSB 3000" in s.cat and "mode: PKTUSB / 2400 Hz" in s.out, s
    assert _keyed(server) == 0 and server.ptt == 0


# -- the ways a session ends -------------------------------------------------------

def test_a_transmitter_already_keyed_stops_the_session(rigctld, session):
    """PTT up before we have called anything is not a thing to transmit through."""
    server = rigctld()
    server.ptt = 1
    s = session(server, arm=True, expect_model="FT-891", per_target=20, wait=120)
    assert s.rc == 3, s
    assert "PTT IS UP" in s.out
    assert server.ptt == 0, "found a keyed transmitter and left it keyed"
    assert len(_qsy_targets(s.cat)) == 1, "carried on to the next target"


def test_an_attempt_that_never_reaches_the_air_stops_with_the_slot_intact(rigctld,
                                                                          session):
    """The audio device is gone, so the connect tool dies before it keys anything.

    "The gateway stayed silent" and "our own process never started" are opposite
    conclusions and only the first is worth carrying to the next target: the second
    repeats identically on every one of them and takes the slot with it.
    """
    server = rigctld()
    s = session(server, audio={"input_fails": True},
                arm=True, expect_model="FT-891", per_target=20, wait=180)
    assert s.rc == 4, s
    assert "never reached the air" in s.out, s
    assert len(_qsy_targets(s.cat)) == 1, "burnt a second target on the same defect"
    assert "identifying" not in s.out, "identified for a transmission that never happened"
    assert server.ptt == 0


def test_losing_the_input_device_costs_a_target_and_not_the_slot(rigctld, session):
    """This station runs a second modem against the same codec.

    A read that fails is one target skipped, not a session ended and not a frequency
    transmitted on blind — the sense has to happen before the call, so no sense means
    no call.
    """
    server = rigctld()
    s = session(server, audio={"rec_fails": True},
                arm=True, expect_model="FT-891", per_target=20, wait=180)
    assert s.rc == 1, s
    assert s.out.count("cannot listen on 'fake'") == 2, s
    assert _keyed(server) == 0, "called a channel it never listened to"
    assert server.ptt == 0


def test_ctrl_c_during_an_attempt_drops_the_transmitter(rigctld, session):
    """An interrupt is an emergency stop, and it stops at the transmitter.

    The signal is sent once the attempt is demonstrably on the air — the child's
    first ``T 1`` — which is the moment a ``finally:`` would not save us.
    """
    server = rigctld()
    s = session(server, gateways=(TARGETS[0],),
                audio={"station": {"call": "W1AAA", "bw": "2300"}},
                arm=True, expect_model="FT-891", per_target=60, wait=None)
    assert server.wait_for("T 1", timeout=180), "the attempt never reached the air"
    s.proc.send_signal(signal.SIGINT)
    s.wait(60)

    assert s.rc == 1, s
    assert "*** interrupted" in s.out, s
    assert "*** CONNECTED" not in s.out
    assert server.ptt == 0, "interrupted with the transmitter keyed"

    # And nothing of ours is left holding it: no further traffic reaches the daemon.
    settled = len(server.commands)
    time.sleep(2.0)
    assert len(server.commands) == settled, (
        f"something outlived the session and is still talking to rigctld: "
        f"{server.commands[settled:]}")
    assert server.ptt == 0


def test_a_rigctld_that_stops_answering_is_shouted_about_not_assumed_away(rigctld,
                                                                          session):
    """The daemon goes quiet partway through — which is how the last rehearsal ended.

    Six commands are answered, which carries the session into the first target's
    mode check, and then nothing is. The session must finish rather than hang, must
    not talk itself into a verdict it has no evidence for, and must tell the operator
    in plain words that it can no longer see the transmitter — in the same words the
    attempt uses, now that both processes drop PTT through the same code.
    """
    server = rigctld(deaf_after=6)
    s = session(server, arm=True, expect_model="FT-891", per_target=20, wait=180)
    assert s.rc == 1, s
    assert "MAY BE STUCK" in s.out, s
    assert "PTT down: False" in s.out, "reported a transmitter it cannot see as down"
    assert "*** CONNECTED" not in s.out
    assert _keyed(server) == 0


def test_no_targets_is_refused_before_the_rig_is_touched(rigctld, session):
    """Nothing on the band means nothing to do, and it costs no CAT at all."""
    server = rigctld()
    s = session(server, gateways=(("W1AAA", 14100.0),), arm=True,
                expect_model="FT-891", wait=60)
    assert s.rc == 2, s
    assert "no gateways match" in s.out
    assert server.commands == [], f"talked to the rig anyway: {server.commands}"


# -- the child, and the parent that owns the transmitter after it ------------------

@pytest.fixture
def cat(rigctld):
    """An armed :class:`onair_session.Cat` against a fake daemon, stood down after."""
    made = []

    def make(server, wait: float = 0.05):
        c = S.Cat(f"127.0.0.1:{server.port}", wait=wait, armed=True)
        made.append(c)
        return c

    yield make
    for c in made:
        c.rig.retire()


@pytest.mark.parametrize("body,expect_kill", [
    ("import time; time.sleep(600)", False),
    ("import signal, time\n"
     "signal.signal(signal.SIGTERM, lambda *a: None)\n"
     "while True: time.sleep(0.05)", True),
], ids=["hangs", "ignores-sigterm"])
@pytest.mark.realtime
def test_an_attempt_that_overruns_is_terminated_and_the_transmitter_dropped(
        monkeypatch, tmp_path, rigctld, cat, body, expect_kill):
    """``subprocess.run(timeout=)`` would send SIGKILL, the one signal a child cannot
    unkey through: its handlers and its watchdog die with it, and if it was keyed at
    that instant the radio stays keyed. So the child gets SIGTERM and time to unkey
    itself, and the parent drops PTT afterwards whatever happened — it cannot know
    where in an over the child stopped.

    Driven in process rather than through the command line, because
    ``run_attempt``'s window is the per-target timeout plus ninety seconds and this
    is about the escalation, not about the clock.
    """
    monkeypatch.setattr(S, "_CHILD_GRACE_S", 1.0)
    server = rigctld()
    c = cat(server)
    c.key(True, "standing in for a child mid-over")
    assert server.ptt == 1

    src = tmp_path / "child.py"
    src.write_text(body)
    t0 = time.monotonic()
    rc, _out = S.run_attempt([sys.executable, str(src)], 1.0, c)
    assert time.monotonic() - t0 < 15.0, "the parent waited on a child that never ends"
    assert rc == -(signal.SIGKILL if expect_kill else signal.SIGTERM), (
        f"the child ended with {rc}, not the escalation this path promises")
    assert S._child is None
    assert server.ptt == 0, "the parent left the transmitter up after killing the child"


# What a key-up may cost when rigctld will not confirm it — a number the operator can
# be given, because he is standing in front of a transmitter in an unknown state while
# it runs. At most `_KEY_BUDGET_S` deciding the key-up has failed, then at most
# `_UNKEY_BUDGET_S` putting the rig back down in case rigctld keyed the radio and
# failed only to say so. It used to be bounded by neither: three attempts at a
# three-second socket timeout, twice over, measured at 15.07 s before anything was
# said.
_KEY_GIVE_UP_S = RIG._KEY_BUDGET_S + RIG._UNKEY_BUDGET_S


@pytest.mark.parametrize("server_kw", [
    pytest.param({"refuse": ("T",)}, id="refuses"),
    pytest.param({"answer": False}, id="says-nothing"),
])
def test_a_key_up_that_cannot_be_confirmed_gives_up_fast_and_loudly(rigctld, cat,
                                                                    capsys, server_kw):
    """A rig that will not confirm a key-up is not a rig to keep asking.

    Whatever it is doing, nothing is going out — and the answer to that is to say so
    while the operator is still in front of the radio, not to spend a quarter of a
    minute finding out. The two ways rigctld declines are both here: an in-band
    ``RPRT -1``, which is an answer, and silence, which is not. Either way the key-up
    ends inside a stated bound, reports itself, and puts PTT down on the way past —
    rigctld may have keyed the radio and failed only to say so.
    """
    server = rigctld(**server_kw)
    c = cat(server)
    t0 = time.monotonic()
    assert c.key(True, "station ID") is False, "keyed against a daemon that refused"
    held = time.monotonic() - t0
    assert held <= _KEY_GIVE_UP_S + 1.0, (
        f"a key-up nobody could confirm took {held:.1f} s to give up, past the "
        f"{_KEY_GIVE_UP_S:.1f} s this path promises")
    said = capsys.readouterr().out
    assert "PTT ON NOT CONFIRMED" in said and "NOT TRANSMITTING" in said, said
    assert server.seen("T 0"), "gave up on a key-up without dropping PTT to be sure"
    assert server.ptt == 0


@pytest.mark.realtime
def test_a_daemon_that_goes_quiet_while_keyed_is_never_read_as_unkeyed(rigctld, cat,
                                                                       capsys):
    """The worst state this program has, in isolation: keyed, and no way to see the rig.

    An empty reply is not a confirmation. The shutdown must finish — an operator is
    waiting on it — and it must say what it could not do. This is the shape the CLI
    case above ends in, without the session around it.
    """
    server = rigctld()
    c = cat(server)
    assert c.key(True, "test") and server.ptt == 1
    server.answer = False

    t0 = time.monotonic()
    c.shutdown()
    assert time.monotonic() - t0 < 20.0, "shutdown never returned"
    said = capsys.readouterr().out
    assert "MAY BE STUCK" in said, said
    assert server.seen("T 0") >= 3, "gave up after one unanswered write"
    assert c.ptt_state() is None, "an unanswered rig must read unknown, never down"
