# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The instrument that says whether the key comes down, checked before it is believed.

`tools/ptt_tail_check.py` exists because every keying figure this tree holds was
measured against a fake rigctld and a fake sound device, and the first contact with
real hardware left the transmitter up for 7.84 s. It is the first thing that keys
when privileges return, so what it says has to be worth something.

Three claims, tested three ways:

  * the measurement. Synthetic recordings with a mute interval put there on purpose,
    and then the real one — the capture that ended the 2026-07-28 session, where the
    instrument has to find 7.84 s on the third transmission and 0.42 s on the first.
  * the bounds. They are the dress rehearsal's bounds, asserted here to be the same
    numbers, so that a pass on the bench and a pass on the radio are one claim.
  * the safety. The arm gate, a rig that reads back keyed, a rig that is not the one
    named, a receiver too deaf to measure anything, and a Ctrl-C mid-transmission —
    each against a fake daemon on an ephemeral port. **Nothing here may touch 4533**:
    there is a real transmitter on the end of it.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core.ptt import arming_refusal
from hfmodem.tests.kestrel import corpora, fake_rigctld
from hfmodem.tests.kestrel.fake_rigctld import FakeRigctld

T = corpora.harness("ptt_tail_check")
KC = corpora.harness("kestrel_connect")

_ROOT = Path(corpora.ROOT)
_TOOLS = corpora.TOOLS
_FAKE_AUDIO = Path(__file__).resolve().parent / "fakeaudio"
_TOOL = _TOOLS / "ptt_tail_check.py"

# The recording the tool was written for: three transmissions through a receiver this
# rig mutes itself, the last of them held 7.84 s past its final sample.
_INCIDENT = _ROOT / "logs" / "onair" / "20260728T234742Z" / "20260728T234805Z-W9SSJ-K9WRA.wav"

MYCALL = "W9SSJ"
PTT_LINE = "/dev/null"            # a character device the arm gate accepts, nothing opens
FS = 48000


def _synth(*, lead=0.20, audio=3.70, tail=0.42, before=7.0, after=1.5, band=-2.0,
           floor=-69.0, monitor=-24.0, monitor_hz=None, seed=3) -> np.ndarray:
    """A recording of one transmission through a receiver that mutes itself.

    The three levels are the ones this station measured off the air on 2026-07-28:
    band audio, the muted codec while the key is up and nothing is modulating it, and
    the transmit audio coming back through that same muted input.

    ``monitor_hz`` puts that transmit audio on a tone over the codec floor instead of
    spreading it across the whole spectrum as noise. Noise was the shape this file
    used everywhere, and it is the one shape that cannot expose a detector reading
    total level: it is broadband, so it darkens the top of the passband along with
    everything else, and it was always set below the band, so the keyed region looked
    quiet however it was measured. Neither is true of a rig's MONITOR — this
    station's emission is a 700 Hz identification and every modem it hosts works
    below 2 kHz, and an operator sets the gain by ear, which puts it over the band.
    """
    rng = np.random.default_rng(seed)

    def part(secs: float, dbfs: float) -> np.ndarray:
        return rng.standard_normal(int(round(secs * FS))) * 10 ** (dbfs / 20)

    def emission(secs: float, dbfs: float) -> np.ndarray:
        if monitor_hz is None:
            return part(secs, dbfs)
        t = np.arange(int(round(secs * FS))) / FS
        return part(secs, floor) + np.sin(2 * np.pi * monitor_hz * t) * 10 ** (dbfs / 20) * np.sqrt(2)

    return np.concatenate([part(before, band), part(lead, floor), emission(audio, monitor),
                           part(tail, floor), part(after, band)])


def _wav(path: Path, samples: np.ndarray) -> Path:
    from scipy.io import wavfile

    wavfile.write(str(path), FS, samples.astype(np.float32))
    return path


# -- the measurement ---------------------------------------------------------------

@pytest.mark.parametrize("lead,audio,tail", [
    (0.20, 3.70, 0.42),          # what a healthy transmission looked like off the air
    (0.05, 1.75, 0.25),          # the deliberate hold alone, on a connect-request
    (0.16, 1.80, 7.84),          # the stuck one
    (0.90, 3.70, 1.10),          # both edges wide, and both inside their bounds
])
def test_a_known_mute_interval_is_measured_back(lead, audio, tail):
    """The analysis, on audio whose answer is known to the frame.

    A frame is 20 ms and each edge can lose one to the frame that straddles it, so
    50 ms is the tolerance the resolution allows.
    """
    trace = T.measure(_synth(lead=lead, audio=audio, tail=tail), FS)
    assert trace is not None
    assert trace.lead == pytest.approx(lead, abs=0.05)
    assert trace.audio_s == pytest.approx(audio, abs=0.05)
    assert trace.tail == pytest.approx(tail, abs=0.05)
    assert trace.keyed_s == pytest.approx(lead + audio + tail, abs=0.05)
    assert trace.regions == 1


def test_a_quiet_band_is_measured_the_same_way():
    """The thresholds are drops, not levels: a band 30 dB quieter reads the same.

    Absolute thresholds are how a check calibrated on one evening's receiver becomes
    a check that reports nothing on the next.
    """
    trace = T.measure(_synth(band=-32.0, floor=-95.0, monitor=-54.0), FS)
    assert trace is not None
    assert trace.tail == pytest.approx(0.42, abs=0.05)


@pytest.mark.parametrize("tail,code,word", [(0.42, 0, "PASS"), (7.84, 1, "FAIL")])
def test_a_monitor_loud_enough_to_hear_does_not_hide_the_keying(tail, code, word):
    """The rig's MONITOR set where an operator actually sets it: transmit audio over
    the band, not under it. A monitor too quiet to hear is a monitor that cannot
    report a lead, so this is the working case and not the awkward one.

    Read on TOTAL level the keying comes apart. It stops looking quiet the instant
    the audio starts, so all that is left below the band is the unmodulated carrier
    at each end, and the tool measures one of those stretches instead of the
    transmission — here it takes the tail, finds no audio in it and reports nothing,
    where on 2026-08-15 off the air it took the lead, found the head of an ARDOP
    burst in it and called that a 0.28 s transmission with a 0.220 s tail. A pass.
    `txwitness` read the same 100 s recording as eight keyings of about 1.9 s. Which
    way it goes wrong depends only on what happened to be in the stretch it kept,
    which is why the keyed region is read across the top of the passband instead.
    """
    trace = T.measure(_synth(lead=0.16, audio=1.80, tail=tail, band=-12.0,
                             monitor=-10.0, monitor_hz=700.0), FS)
    assert trace is not None
    assert trace.audio_dbfs is not None, "the transmit audio was not found at all"
    assert trace.lead == pytest.approx(0.16, abs=0.05)
    assert trace.audio_s == pytest.approx(1.80, abs=0.05)
    assert trace.tail == pytest.approx(tail, abs=0.05)
    got, message = T.verdict(trace)
    assert got == code, message
    assert word in message


def test_a_receiver_that_never_mutes_measures_nothing_and_says_so():
    """The failure that matters most: no verdict may come out of a recording with no
    trace of a transmitter in it."""
    rng = np.random.default_rng(1)
    assert T.measure(rng.standard_normal(int(12 * FS)) * 10 ** (-2 / 20), FS) is None
    code, message = T.verdict(None)
    assert code == 4 and "INCONCLUSIVE" in message


def test_a_keyed_region_with_no_audio_in_it_is_not_a_tail():
    """A key that went up and down over nothing at all places no last sample, so
    there is nothing to measure the tail from — and 4.3 s of it must not read as a
    4.3 s pass."""
    trace = T.measure(_synth(lead=2.0, audio=0.0, tail=2.3), FS)
    assert trace is not None
    assert trace.keyed_s == pytest.approx(4.3, abs=0.05)
    assert trace.audio_start is None and trace.tail is None
    code, message = T.verdict(trace)
    assert code == 4 and "INCONCLUSIVE" in message


def test_the_bounds_are_the_ones_the_rehearsal_asserts():
    """One claim, not two. If the dress rehearsal's margins move, this moves with
    them or the radio is being held to a standard the bench is not."""
    from hfmodem.tests.kestrel import test_onair_dress_rehearsal as rehearsal

    assert T.TAIL_MARGIN_S == rehearsal._TAIL_MARGIN_S
    assert T.CARRIER_LEAD_S == rehearsal._CARRIER_LEAD_S
    assert T.TAIL_BOUND_S == KC.TX_IDLE_HOLD_S + T.TAIL_MARGIN_S


@pytest.mark.parametrize("lead,tail,code", [
    (0.2, T.TAIL_BOUND_S - 0.1, 0),
    (0.2, T.TAIL_BOUND_S + 0.3, 1),
    (T.CARRIER_LEAD_S + 0.5, 0.42, 1),
])
def test_the_verdict_follows_the_bounds_at_both_edges(lead, tail, code):
    trace = T.measure(_synth(lead=lead, tail=tail), FS)
    got, message = T.verdict(trace)
    assert got == code, message
    assert ("PASS" if code == 0 else "FAIL") in message


# -- the real recording ------------------------------------------------------------

incident = pytest.mark.skipif(not _INCIDENT.exists(),
                              reason="the 2026-07-28 capture is not in this tree")


@incident
def test_the_stuck_transmitter_is_found_in_the_recording_of_it():
    """The measurement against the only ground truth there is: a real transmitter,
    a real receive mute, and a tail somebody has already read off it by hand."""
    audio, fs = T.read_wav(_INCIDENT)
    trace = T.measure(audio, fs)
    assert trace.tail == pytest.approx(7.84, abs=0.02)
    assert trace.regions == 3, "the other two transmissions of that session went missing"
    assert T.verdict(trace)[0] == 1


@incident
def test_the_clean_transmissions_of_that_same_session_pass():
    """A check that only ever fails is not a check. The first transmission of the
    same recording is the healthy case, off the air, at 0.42 s.

    Read across the top of the passband this comes out at 0.44 rather than the 0.42
    read off total level, and the frame in between is why: at 2.74 s the whole band
    is already back to −9 dBFS while the top of it is still at −41, because band
    audio returns from the bottom up as the receiver un-mutes. Both readings put the
    edge inside the same 20 ms frame, so the tolerance here is a frame — it was 0.02,
    finer than the instrument resolves, and only held while the edge happened to
    round the other way. The direction is asserted instead, and it is the safe one:
    reading the mute alone rather than the mute plus whatever the monitor is doing
    brackets the keyed region a frame wider at each end, so the tail is never
    reported shorter than it was.
    """
    audio, fs = T.read_wav(_INCIDENT)
    trace = T.measure(audio[:int(5 * fs)], fs)
    assert trace.tail == pytest.approx(0.42, abs=0.05)
    assert trace.tail >= 0.42, "the tail is being reported shorter than it was measured"
    assert trace.lead == pytest.approx(0.18, abs=0.05)
    assert T.verdict(trace)[0] == 0


# -- the tool, as the operator runs it ---------------------------------------------

@pytest.fixture
def rigctld():
    yield from fake_rigctld.serving()


@pytest.fixture
def run(tmp_path):
    """The tool as a subprocess, with a fake daemon and a fake codec under it.

    The codec substitution is the assumption everything below rests on: without it
    this module would open the station's own sound card, which another modem may be
    holding.
    """
    if not (_FAKE_AUDIO / "sounddevice.py").exists():
        pytest.skip("test harness not present (installed-wheel run)")
    started: list[subprocess.Popen] = []

    def go(server: FakeRigctld | None = None, *, audio: dict | None = None,
           wait: float | None = 180.0, **flags):
        env = {**os.environ,
               "PYTHONPATH": os.pathsep.join([str(_FAKE_AUDIO), str(corpora.PKG_ROOT)]),
               "KESTREL_FAKE_AUDIO": json.dumps({**(audio or {})})}
        argv = [sys.executable, str(_TOOL), "--record", str(tmp_path / "rec")]
        if server is not None:
            argv += ["--rigctld", f"127.0.0.1:{server.port}"]
        # An `--analyze` run takes none of the station flags: it never opens a device
        # and never asks a radio anything.
        # 20 WPM, not the 30 this used to pass to run faster: §97.119(b)(1) caps a
        # CW identification there and core.cwid refuses above it. A test that asks
        # the tool to send something the rule forbids is testing the wrong tool.
        # --no-line-ptt: the bench keys through the fake daemon, because no
        # test machine has a line whose pin moves (TIOCMBIC on /dev/null or a
        # pty is ENOTTY) and a run asking for one refuses before the first
        # burst. The operator's line-keyed default is pinned as text by
        # test_the_operators_check_keys_the_line_not_the_daemon.
        live = {} if "analyze" in flags else {"mycall": MYCALL, "device": "fake",
                                              "listen": 5, "wpm": 20,
                                              "no_line_ptt": True}
        # Arming names the keying line, because the tool refuses without one: the
        # last-resort unkey has to have something to pull when rigctld stops
        # answering. The gate stats and never opens, so /dev/null satisfies it
        # without a serial port anywhere near this suite.
        if flags.get("arm") and "ptt_device" not in flags:
            live["ptt_device"] = PTT_LINE
        for k, v in {**live, **flags}.items():
            if v is None:                 # the way a test asks for a flag to be absent
                continue
            argv.append("--" + k.replace("_", "-"))
            if v is not True:
                argv.append(str(v))
        proc = corpora.launch(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env, cwd=str(corpora.REPO))
        started.append(proc)
        if wait is None:
            return proc
        out = proc.communicate(timeout=wait)[0]
        return subprocess.CompletedProcess(argv, proc.returncode, out, "")

    yield go
    for p in started:
        if p.poll() is None:
            corpora.stop_tool(p)


def _sessions(tmp_path: Path) -> list[Path]:
    return sorted(p for p in (tmp_path / "rec").glob("*") if p.is_dir())


def test_the_dry_run_cannot_reach_the_transmitter(rigctld, run):
    """The default is a rehearsal, and a rehearsal that keys is not one."""
    server = rigctld()
    r = run(server, dial=7101.0)
    assert r.returncode == 2, r.stdout
    assert not server.seen("T 1"), server.commands
    assert "DISARMED" in r.stdout
    assert "nothing is established" in r.stdout


def test_an_armed_run_keys_once_and_leaves_the_transmitter_down(rigctld, run, tmp_path):
    """The whole sequence, armed: one key-up, one transmission, PTT confirmed down.

    The fake codec does not mute itself while it plays, which is the point of running
    it here: the tool has to come back with no verdict rather than with a pass it
    cannot support, and its own account of the transmission has to be printed anyway.
    """
    server = rigctld()
    r = run(server, arm=True, expect_model="FT-891", dial=7101.0)
    assert r.returncode == 4, r.stdout
    assert server.seen("T 1") == 1, server.commands
    assert server.ptt == 0
    keys = [c for c in server.commands if c.startswith("T ")]
    assert keys[0] == "T 1" and keys[-1] == "T 0", keys
    assert "INCONCLUSIVE" in r.stdout
    assert "off the air" in r.stdout and "software" in r.stdout

    sessions = _sessions(tmp_path)
    assert len(sessions) == 1, sessions
    kept = {p.name for p in sessions[0].iterdir()}
    assert {"recording.wav", "timeline.json"} <= kept, kept
    stamps = json.loads((sessions[0] / "timeline.json").read_text())
    # In order, and adjacent statements are allowed to share a clock tick — the
    # gap that has to be real is the transmission's, which the line below sizes
    # against the audio rather than against the resolution of `time.time()`.
    order = ["key_ok", "play_start", "play_return", "unkey_ok"]
    stamped = [stamps[k] for k in order]
    assert stamped == sorted(stamped), dict(zip(order, stamped))
    assert stamps["play_return"] - stamps["play_start"] >= stamps["played_s"] - 0.1


def test_the_dial_and_the_mode_are_read_back_before_anything_is_keyed(rigctld, run):
    """Never key a radio you have not asked where it is. A ``F``/``M`` rigctld
    accepted is not a rig that moved."""
    server = rigctld()
    r = run(server, arm=True, expect_model="FT-891", dial=7101.0)
    assert "7101.000" in r.stdout, r.stdout
    assert server.freq == 7101000 and server.mode == "PKTUSB"
    assert "7101.700" in r.stdout, "never said where the emission would land"


def test_the_preparation_asks_the_rig_only_what_it_uses(rigctld, run):
    """The CAT sequence, read off the wire — the dress rehearsal pins its own for
    the same reason.

    `Rig.model` was split out of `identify` because the frequency read that comes
    with it is a command spent on an answer nobody looks at, and an extra ``f`` on
    this path has broken a keying test before. Nothing pinned the sequence here, so
    the tool went on calling `identify`; this is that pin.
    """
    server = rigctld()
    r = run(server, dial=7101.0)
    assert r.returncode == 2, r.stdout
    assert server.commands == ["\\dump_caps", "t", "F 7101000", "M PKTUSB 2400",
                               ";f", ";m"], server.commands


def test_a_rig_that_reads_back_keyed_stops_the_check(rigctld, run):
    """This instrument measures the PTT line. One that is already up is somebody
    else's transmission, and calling over it is exactly the fault being hunted."""
    server = rigctld()
    server.ptt = 1
    r = run(server, arm=True, expect_model="FT-891")
    assert r.returncode == 3, r.stdout
    assert not server.seen("T 1"), server.commands
    assert server.ptt == 0, "found a keyed transmitter and walked away from it"


def test_the_wrong_radio_is_refused_before_anything_transmits(rigctld, run):
    server = rigctld(model="IC-7300")
    r = run(server, arm=True, expect_model="FT-891")
    assert r.returncode == 2, r.stdout
    assert not server.seen("T 1"), server.commands
    assert "REFUSING TO ARM" in r.stdout


def test_arming_without_naming_the_radio_is_refused(run):
    r = run(None, arm=True)
    assert r.returncode == 2, r.stdout
    assert "--expect-model" in r.stdout


def test_arming_without_a_keying_line_is_refused(rigctld, run):
    """This tool exists because a transmitter once stayed up 7.84 s past its last
    sample, so it may not arm with the one path that could have brought it down left
    unnamed. Refused at the flags: the fake daemon takes no command.

    The refusal is `core.ptt.arming_refusal`'s, word for word. This tool used to
    keep its own copy, and the copy had already drifted."""
    server = rigctld()
    r = run(server, arm=True, expect_model="FT-891", ptt_device=None)
    assert r.returncode == 2, r.stdout
    assert arming_refusal(None) in r.stdout, r.stdout
    assert server.commands == [], server.commands


def test_arming_on_a_keying_line_with_no_device_behind_it_is_refused(rigctld, run,
                                                                    tmp_path):
    """A path that exists but is not a character device is the placeholder failure:
    it looks like a keying line right up until nothing happens."""
    dud = tmp_path / "not-a-serial-port"
    dud.write_text("")
    server = rigctld()
    r = run(server, arm=True, expect_model="FT-891", ptt_device=str(dud))
    assert r.returncode == 2, r.stdout
    assert arming_refusal(str(dud)) in r.stdout, r.stdout
    assert server.commands == [], server.commands


def test_the_operators_check_keys_the_line_not_the_daemon():
    """The shape the operator gets, pinned as text — the runs in this file
    cannot show it, because they opt out of it: no bench has a line whose pin
    moves (`TIOCMBIC` on /dev/null or a pty is ENOTTY), so the bench keys
    through the fake daemon the way the dress rehearsal does.

    On air the daemon path is 0-for-N: rigctld answered every command until it
    keyed the rig and then answered nothing, twelve failed unkeys out of twelve
    on 2026-08-13, while the line itself read LOW instantly, eight of eight.
    This tool is the one the README sends an operator to before any connect
    attempt — it exists because a transmitter once stayed up 7.84 s past its
    last sample — and it was the last armed tool still keying through the
    daemon, with no way to ask for the line at all.
    """
    src = _TOOL.read_text()
    assert "line_ptt=args.line_ptt" in src, (
        "the Rig is built without the keying line, so every armed run keys "
        "through the daemon")
    decl = [ln for ln in src.splitlines() if 'add_argument("--line-ptt"' in ln]
    assert decl and "default=True" in decl[0], (
        "--line-ptt is not the default, so the operator's first act on a new "
        "station keys through the daemon whenever a flag is forgotten")


def _printed_commands(text: str) -> list[str]:
    """Every ptt_tail_check command a document prints, continuations joined."""
    joined = re.sub(r"\\\s*\n\s*", " ", text)
    return [ln.strip() for ln in joined.splitlines()
            if "ptt_tail_check.py" in ln and "python" in ln.split("#")[0]]


@pytest.mark.parametrize("doc", ["README", "docstring"])
def test_every_printed_example_is_a_command_the_tool_accepts(doc):
    """The README presents this check as the thing to run before any connect
    attempt, so the command it prints must start. As printed until now, the
    armed example was refused by the arm gate before the tool did anything:
    it carried no --ptt-device."""
    text = ((_TOOLS / "README.md").read_text() if doc == "README"
            else _TOOL.read_text())
    cmds = _printed_commands(text)
    assert cmds, f"the {doc} no longer shows how to run the check"
    armed = [c for c in cmds if "--arm" in c]
    assert armed, f"the {doc} no longer shows the armed run"
    src = _TOOL.read_text()
    for c in armed:
        assert "--ptt-device" in c, (
            f"the {doc} prints an armed command the arm gate refuses: {c}")
    for c in cmds:
        for flag in set(re.findall(r"--[a-z][a-z-]*", c)):
            assert f'"{flag}"' in src, (
                f"the {doc} prints {flag}, which the tool does not take: {c}")


def test_a_deaf_receiver_is_not_transmitted_over(rigctld, run):
    """A receiver below the codec floor has no mute to show, so keying into it would
    spend the RF and answer nothing."""
    server = rigctld()
    r = run(server, audio={"bed": "quiet"}, arm=True, expect_model="FT-891")
    assert r.returncode == 5, r.stdout
    assert not server.seen("T 1"), server.commands


def test_ctrl_c_mid_transmission_puts_the_transmitter_down(rigctld, run):
    """The first thing that keys after privileges return is the first thing that has
    to survive an operator reaching for Ctrl-C."""
    server = rigctld()
    proc = run(server, arm=True, expect_model="FT-891", wait=None)
    assert server.wait_for("T 1", timeout=60), "the tool never keyed"
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    out = proc.communicate(timeout=30)[0]
    assert proc.returncode != 0, out
    assert server.ptt == 0, out
    assert server.seen("T 0"), server.commands


# -- the verdict off a recording ---------------------------------------------------

def _session(tmp_path: Path, **kw) -> Path:
    """A session directory as a live run leaves one, with the software's own account
    of the transmission it recorded."""
    out = tmp_path / "session"
    out.mkdir(exist_ok=True)
    lead, audio, tail = kw.get("lead", 0.2), kw.get("audio", 3.7), kw.get("tail", 0.42)
    _wav(out / "recording.wav", _synth(**kw))
    started = 1_000_000.0
    before = kw.get("before", 7.0)
    (out / "timeline.json").write_text(json.dumps({
        "started": started, "played_s": audio, "tail_hold_s": KC.TX_IDLE_HOLD_S,
        "key_cmd": started + before - 0.1, "key_ok": started + before,
        "play_start": started + before + lead, "play_return": started + before + lead + audio,
        "unkey_cmd": started + before + lead + audio + tail,
        "unkey_ok": started + before + lead + audio + tail,
    }))
    return out


@pytest.mark.parametrize("tail,code,word", [(0.42, 0, "PASS"), (7.84, 1, "FAIL")])
def test_analyze_reaches_the_same_verdict_off_a_kept_recording(run, tmp_path, tail, code,
                                                               word):
    """The tool's own command line, over a recording whose answer is known — the path
    that puts a session under the same bounds afterwards as it ran under."""
    r = run(None, analyze=str(_session(tmp_path, tail=tail)))
    assert r.returncode == code, r.stdout
    assert word in r.stdout
    assert f"{tail:.3f} s" in r.stdout


def test_analyze_shows_where_the_two_accounts_disagree(run, tmp_path):
    """A recording whose transmitter came up half a second before the software says
    it asked: the disagreement has to be on the page, not averaged away."""
    out = _session(tmp_path, lead=0.2)
    stamps = json.loads((out / "timeline.json").read_text())
    stamps["key_ok"] -= 0.5
    (out / "timeline.json").write_text(json.dumps(stamps))
    r = run(None, analyze=str(out))
    assert r.returncode == 0, r.stdout
    assert "+0.500 s" in r.stdout, r.stdout


@incident
def test_analyze_reads_the_recording_of_the_incident(run):
    """Pointed at the session that started all this, the instrument calls it."""
    r = run(None, analyze=str(_INCIDENT.parent))
    assert r.returncode == 1, r.stdout
    assert "7.840 s" in r.stdout
    assert "3 keyed regions" in r.stdout
