# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The narrator has to catch, live, what two slots only caught the next day.

Everything asserted here is a thing an operator missed while it was happening:

  * **2026-08-18, WW2MI attempt 2.** The gateway rejected the login challenge and
    sat at its prompt. Our side answered its IDLE frames for seventy-one seconds
    — twenty-six keyings on a shared band — and stopped only when the operator
    sent SIGTERM. The refusal itself was invisible until teardown, because the
    peer's words are printed by `summarize()` at the end of the session and not
    when they arrive. So the deadlock must be caught structurally, from the frame
    traffic, and it must be caught while there is still time to act on it.
  * **2026-08-18, WW2MI attempt 1.** Authenticated, was offered a 496-byte
    message, and died with `expected SOH, got byte 0xb1`. Two HEADER-ONLY frames
    landed on our own session, the second six seconds before the failure, among
    two more on foreign sessions that had nothing to do with us. Distinguishing
    those four is the whole value of the frame lines.
  * **2026-08-16, KC9GHZ.** VARA connected, the peer went away, and the modem
    sent three turn-requests and twenty-five keepalives into an empty channel.
  * **2026-08-16, WS8EOC.** PACTOR held a clean link and sent packet #1 thirty-
    eight times without advancing.

And three things it said that the stream does not support, each one worse than
saying nothing:

  * **2026-08-19, the ack-placement A/B.** `never keyed -- this run transmitted
    nothing`, over a launcher stream whose two arms keyed seventy times between
    them into WS8EOC. `ackab` runs the modem through lib/attempts.sh as a child
    with its output redirected, so the keying is real and the stream is blind to
    it. An agent believing that would have retried a launcher that worked.
  * **`transmitting into an empty channel`**, over a PACTOR run that had decoded
    seventeen inbound frames -- thirteen of them control signals at zero bit
    errors, the last of them twenty-seven lines earlier -- and over a VARA call
    three lines above the gateway answering at 15 of 15 tones. The count of
    unanswered transmissions was right both times. Who else was on the air was
    never ours to say, and it is the half an operator reads.
  * **`channel in use, waiting`**, on refusals that ended the run.

The narrator reads text and only text. There is no path from it to the key, and
the last test here is what keeps it that way.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hfmodem.station.narrate import Event, Narrator, narrate
from hfmodem.tests import evidence

REPO = evidence.TREE
LOGS = evidence.WORKING
TOOL = REPO / "tools" / "narrate.py"
#: The A/B whose two arms this stream cannot see, and the directory it wrote.
ACKAB = evidence.CAPTURES / "ackab-WS8EOC-0819-0058"

pytestmark = pytest.mark.skipif(not LOGS.is_dir(),
                                reason="session logs not present (installed-wheel run)")


def read(name: str) -> tuple[list[Event], Narrator]:
    return narrate(lines_of(LOGS / name))


def lines_of(path: Path) -> list[str]:
    if not path.is_file():
        pytest.skip(f"{path} not present")
    return path.read_text(errors="replace").splitlines()


def alarms(events: list[Event]) -> list[str]:
    return [e.text for e in events if e.alarm]


def test_the_deadlock_is_called_out_a_minute_before_the_operator_stopped_it():
    events, _ = read("mailfetch-ardop-ww2mi-4.log")
    stalls = [e for e in events if e.alarm and "stalled" in e.text]
    assert stalls, "the IDLE/DATAACK deadlock produced no alarm"
    # SIGTERM landed at 21:50:37; anything after ~21:50 is a post-mortem, not a
    # warning the operator could have keyed off.
    assert stalls[0].at < "21:50:00", f"first stall alarm was late: {stalls[0]}"
    assert "no data either way" in stalls[0].text


def test_a_login_refusal_that_names_the_remaining_budget_is_an_alarm():
    events, n = read("mailfetch-ardop-ww2mi-4.log")
    refusals = [t for t in alarms(events) if "2 attempts remaining" in t]
    assert refusals, "the gateway's rejection was not raised"
    assert "Invalid login challenge response" in refusals[0]
    assert n.outcome == "nothing moved"


def test_every_word_the_peer_said_is_surfaced():
    events, _ = read("mailfetch-ardop-ww2mi-4.log")
    said = [e.text for e in events if e.tag == "peer"]
    assert "Login [931]:" in said
    assert "W9SSJ has 1437 daily minutes remaining with WW2MI (EN82KR)" in said


def test_a_failed_stage_is_reported_with_the_line_that_failed():
    events, n = read("mailfetch-ardop-ww2mi-3.log")
    assert any("expected SOH, got byte 0xb1" in t for t in alarms(events))
    assert n.stage == "failed"


def test_the_message_the_gateway_offered_is_marked_as_the_answer():
    events, _ = read("mailfetch-ardop-ww2mi-3.log")
    offered = [e for e in events if e.tag == "MAIL"]
    assert offered, "the ;PM: offer was not distinguished from the rest of the greeting"
    assert "LJ2AJE2IHO9B" in offered[0].text


def test_bad_frames_on_our_session_are_told_apart_from_foreign_traffic():
    events, n = read("mailfetch-ardop-ww2mi-3.log")
    assert n.session == "0xf3"
    assert (n.rx_bad_ours, n.rx_bad_foreign) == (2, 2)
    reported = [e.text for e in events if e.tag == "frame"]
    assert len(reported) == 2, reported
    assert all("on our session" in t for t in reported)
    assert "q=69" in reported[0] and "q=64" in reported[1]


def test_transmissions_nobody_answered_are_an_alarm_and_stay_a_measurement():
    events, n = read("s2-vara-kc9ghz-force.log")
    unanswered = [t for t in alarms(events) if "nothing decoded back" in t]
    assert unanswered, "twenty-five keepalives with no reply raised nothing"
    # KC9GHZ's connect-response at 15 of 15 tones and three CRC-clean DATA overs
    # are among the nine decodes behind this alarm, so the channel was not empty
    # and the alarm may not say it was.
    assert "9 inbound decodes earlier" in unanswered[0], unanswered[0]
    assert n.target == "KC9GHZ"
    assert n.link_up, "the link was never torn down, and the state must say so"


def test_forcing_a_call_over_an_occupant_is_never_silent():
    events, _ = read("s2-vara-kc9ghz-force.log")
    assert any("--force" in t for t in alarms(events))


def test_a_packet_number_that_never_advances_is_an_alarm():
    events, _ = read("s2-pactor-ws8eoc-mail-2.log")
    stuck = [t for t in alarms(events) if "packet #1" in t]
    assert stuck, "forty-five sends of packet #1 raised nothing"
    assert "10 times" in stuck[0], "the first alarm must land at ten, not at forty"


def test_a_pactor_run_that_keyed_is_not_called_silent():
    """PACTOR keys from its own transmit grid and logs no `PTT ON` line at all.

    Reading keyings off the PTT lines alone reported `never keyed` for a run that
    put twenty-one bursts on the air, which is the most dangerous thing this tool
    could get backwards.
    """
    events, n = read("pounce-pactor-ws8eoc.log")
    assert n.keyings == 21
    assert not any("keying visible" in e.text for e in events)
    assert any("nothing decoded inbound on this stream yet" in t
               for t in alarms(events)), (
        "twenty-one connect bursts and no control signal decoded is the case "
        "this alarm exists for, and all of it that the stream establishes")


def test_the_ab_whose_arms_keyed_seventy_times_is_not_called_silent():
    """The worst thing this tool has said, and the rule that keeps it unsaid.

    `never keyed -- this run transmitted nothing` over the ack-placement A/B of
    2026-08-19, whose arms keyed 22 and 48 times. A confident false negative
    spends the transmitter: an agent reading it retries a launcher that worked.
    What this stream carries is two attempt verdicts and no line of what either
    attempt did, so that -- and the directory holding what it could not see --
    is the whole of what it may say.
    """
    events, n = read("t6-ackab-ws8eoc.log")
    assert n.keyings == 0, "the arms keyed in a child process, not on this stream"
    said = [e.text for e in events]
    assert not any("transmitted nothing" in t or "never keyed" in t for t in said)
    blind = [t for t in said if "no keying visible on this stream" in t]
    assert blind, said
    assert "2 attempts ran as a child process" in blind[0]
    assert blind[0].endswith("(under " + str(ACKAB) + ")"), blind[0]
    assert not any(e.alarm and "keying" in e.text for e in events), (
        "not seeing a thing is not the same as the thing not happening, and an "
        "alarm is what says otherwise")


def test_the_attempts_a_launcher_made_are_reported_with_their_verdicts():
    events, n = read("t6-ackab-ws8eoc.log")
    verdicts = [e.text for e in events if e.tag == "run"]
    assert any("PACTOR/clamped WS8EOC  3596500  CHANGEOVER" in t for t in verdicts)
    assert any("PACTOR/closed WS8EOC  3596500  DATA_PHASE" in t for t in verdicts)
    # The gate's own verdicts are shown and are not attempts: neither started a
    # child, so counting them would overstate what ran out of view.
    assert any("FORCED" in t for t in verdicts)
    assert len(n.attempts) == 2, n.attempts


def test_the_arms_are_narrated_from_the_logs_the_stream_could_not_carry():
    keyed = {}
    for arm in ("clamped", "closed"):
        _, n = narrate(lines_of(ACKAB / f"pactor-WS8EOC-3596500-{arm}.log"))
        keyed[arm] = n.keyings
    assert keyed == {"clamped": 22, "closed": 48}, keyed


def test_a_refusal_that_ends_the_run_is_not_reported_as_a_wait():
    """`waiting` was printed on a refusal after which nothing waited.

    The narrator has no way to know whether a budget is being spent or the pass
    is over, and the operator's next move differs. What the line establishes is
    that we did not key.
    """
    events, _ = narrate(["  channel sense: +3.0 dB -> OCCUPIED",
                         "!! channel in use — listen first, or pass --force"])
    said = [e.text for e in events]
    assert "channel in use, not keyed (refusal 1)" in said, said
    assert not any("waiting" in t for t in said)
    assert not any(e.alarm for e in events), (
        "a launcher that stayed off a busy channel is the gate working, and an "
        "alarm over it is one more red line to learn to ignore")


def test_a_transmitter_left_keyed_is_an_alarm():
    events, _ = narrate([
        "2026-08-18 21:47:52,444 INFO PTT ON -> line",
        "2026-08-18 21:48:41,000 INFO RX IDLE sess=0xf3 ok=True",
    ])
    assert any("keyed for" in t for t in alarms(events))


def test_foreign_traffic_is_not_the_peer_answering():
    """A stranger's frame on another session must not reset the silence count.

    Otherwise a busy band hides the case this tool exists for: ours is the only
    session whose frames say anybody is still listening to us.
    """
    lines = ["2026-08-18 21:47:55,520 INFO RX ConAck200 sess=0xf3 ok=True q=92"]
    lines += [f"2026-08-18 21:48:0{i},000 INFO TX IDLE 0.47s" for i in range(8)]
    lines.insert(4, "2026-08-18 21:48:02,500 INFO RX 16QAM.500.100.O sess=0x00 ok=False"
                    " HEADER-ONLY q=5")
    events, n = narrate(lines)
    assert any("nothing decoded back" in t for t in alarms(events))
    assert (n.rx_bad_ours, n.rx_bad_foreign) == (0, 1)


def test_the_unanswered_alarm_says_what_we_decoded_not_who_was_on_the_air():
    """Seventeen inbound frames off this channel, then `an empty channel`.

    The count of transmissions nobody answered is a measurement and is worth an
    alarm. Whether anyone was there is a guess, it is the half an operator
    reads, and a station whose logs said "X never answers on 40 m" was wrong.
    """
    events, _ = read("t6-pactor-ws8eoc-force.log")
    unanswered = [t for t in alarms(events) if "nothing decoded back" in t]
    assert unanswered, "six unanswered transmissions raised nothing"
    assert "17 inbound decodes earlier on this stream" in unanswered[0], unanswered[0]


def test_an_answer_arriving_three_lines_later_indicts_the_conclusion_only():
    """The same alarm over a gateway that was about to answer.

    Nothing had been decoded when it fired, which is the whole of what our
    receiver could report and is worth raising. Three lines below, KC9GHZ's
    connect-response landed at 15 of 15 tones. A measurement that a later line
    completes is still a measurement; `an empty channel` was a verdict on the
    band that the next three lines overturned.
    """
    events, _ = read("t5-vara-kc9ghz-4.log")
    unanswered = [e for e in events if e.alarm and "nothing decoded back" in e.text]
    assert unanswered, "six unanswered connect-requests raised nothing"
    assert unanswered[0].text.endswith("nothing decoded inbound on this stream yet")
    assert unanswered[0].at == "L56", unanswered[0]
    answered = lines_of(LOGS / "t5-vara-kc9ghz-4.log")[58]
    assert "connect-response for KC9GHZ" in answered and "15/15 tones" in answered


def test_no_line_in_the_corpus_names_a_cause_the_stream_cannot_establish():
    """The narrator is quoted into working documents, so its vocabulary is the
    thing under test: an audit of 2026-08-18 found thirty-two claims across four
    of them that the logs do not support, several load-bearing enough to have
    steered an investigation. These four phrasings were three of those claims.

    Only what the narrator composes is under test. A launcher line it quotes
    verbatim -- `waiting for 7106.5 kHz to clear` -- is the launcher's word for
    what the launcher is about to do, and passing it through unaltered is the
    narrator declining to interpret it.
    """
    retired = ("empty channel", "transmitted nothing", "never keyed",
               "channel in use, waiting")
    for log in sorted(LOGS.glob("*.log")):
        events, n = narrate(lines_of(log))
        for text in [e.text for e in events] + n.state():
            for phrase in retired:
                assert phrase not in text, f"{log.name}: {text}"


def test_the_state_block_answers_where_the_session_got_to():
    _, n = read("mailfetch-ardop-ww2mi-4.log")
    block = "\n".join(n.state())
    assert "UP to WW2MI" in block
    assert "nothing moved" in block
    assert "2 attempts remaining" in block


def test_the_narration_is_shorter_than_the_log_by_an_order_of_magnitude():
    """Silence is the feature. A narrator that reprints the stream is no use."""
    for name in ("mailfetch-ardop-ww2mi-3.log", "mailfetch-ardop-ww2mi-4.log",
                 "s2-pactor-ws8eoc-mail-2.log", "s2-vara-kc9ghz-force.log"):
        lines = (LOGS / name).read_text(errors="replace").splitlines()
        events, _ = narrate(lines)
        assert len(events) * 5 < len(lines), f"{name}: {len(events)} of {len(lines)}"


def tail_on(path: Path):
    return _tool().Tail(os.open(path, os.O_RDONLY), path)


def test_a_half_written_line_is_held_back_until_its_newline_arrives(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("2026-08-18 21:47:52,444 INFO PTT ON -> line\nRX Con")
    tail = tail_on(log)
    assert tail.lines() == ["2026-08-18 21:47:52,444 INFO PTT ON -> line"]
    assert tail.lines() == [], "half a line is not a line"
    with log.open("a") as f:
        f.write("Ack200 sess=0xf3 ok=True q=92\n")
    assert tail.lines() == ["RX ConAck200 sess=0xf3 ok=True q=92"]


def test_a_log_that_shrinks_under_us_is_read_again_from_the_top(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("  CONNECTED WW2MI @ 200 Hz\n")
    tail = tail_on(log)
    assert tail.lines() == ["  CONNECTED WW2MI @ 200 Hz"]
    log.write_text("  link down\n")
    assert tail.lines() is None, "a restarted log must reset the narration"
    assert tail.lines() == ["  link down"]


def test_a_pipe_that_is_merely_quiet_is_not_a_pipe_that_has_finished():
    """Reading nothing right now is what a live session looks like most of the
    time; reading nothing ever again is the launcher exiting. Confusing the two
    either stops the narrator sixty seconds into a run or hangs it after."""
    read_fd, write_fd = os.pipe()
    tail = tail_on_fd(read_fd)
    assert tail.lines() == [] and not tail.eof
    os.write(write_fd, b"  CONNECTED WW2MI @ 200 Hz\n")
    assert tail.lines() == ["  CONNECTED WW2MI @ 200 Hz"]
    assert not tail.eof
    os.close(write_fd)
    assert tail.lines() == [] and tail.eof
    os.close(read_fd)


def tail_on_fd(fd: int):
    return _tool().Tail(fd)


def test_the_tool_reads_the_arm_logs_the_stream_names_at_the_end():
    log = LOGS / "t6-ackab-ws8eoc.log"
    if not log.is_file() or not ACKAB.is_dir():
        pytest.skip(f"{log} or {ACKAB} not present")
    r = subprocess.run([sys.executable, str(TOOL), str(log), "--no-color"],
                       capture_output=True, text=True, timeout=120, cwd=REPO)
    assert "no keying visible on this stream" in r.stdout, r.stderr[-1500:]
    assert "[clamped]   keying  down, 22 keyings" in r.stdout, r.stdout[-2500:]
    assert "[closed]   keying  down, 48 keyings" in r.stdout, r.stdout[-2500:]


@pytest.mark.realtime
def test_an_arm_is_narrated_while_it_is_still_running(tmp_path):
    """An alarm that arrives after the run it was meant to interrupt is not one.

    The whole value of this tool is intervening while there is still time, and
    an `ackab` arm can hold the transmitter for two hundred seconds with its
    output going to a log the launcher's own stream never carries.
    """
    run, arms = tmp_path / "run.log", tmp_path / "arms"
    arms.mkdir()
    run.write_text("== ack-aim A/B: W9SSJ -> WS8EOC, centre 3596500 (dial 3595000)\n")
    seen = tmp_path / "narration"
    with seen.open("w") as out:
        child = subprocess.Popen(
            [sys.executable, str(TOOL), str(run), "-f", "--no-color",
             "--poll", "0.05", "--arm", str(arms / "*.log")],
            stdout=out, stderr=subprocess.STDOUT, cwd=REPO)
        try:
            arm = arms / "pactor-WS8EOC-3596500-clamped.log"
            arm.write_text("  TX[1] connect->WS8EOC  (1.0s)  -- keying\n")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if "[clamped]" in seen.read_text():
                    break
                time.sleep(0.05)
            told = seen.read_text()
        finally:
            child.terminate()
            child.wait(timeout=30)
    assert "[clamped]" in told and "first key-down" in told, told
    assert "-- narrating" in told


def _tool():
    import importlib.util

    spec = importlib.util.spec_from_file_location("narrate_cli", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_narrator_cannot_key():
    """The one thing that must stay true of this tool, enforced structurally.

    It runs during a live slot, beside a session that owns the rig. Nothing it
    imports may be able to reach a serial port, an audio device or a peer, so the
    import list is the test: a reader of text has no business with any of them.
    """
    forbidden = {"serial", "sounddevice", "socket", "socketserver", "http",
                 "urllib", "asyncio", "subprocess", "hfmodem.core.ptt",
                 "hfmodem.core.rig", "hfmodem.station.air"}
    for path in (TOOL, REPO / "packages/hfmodem/hfmodem/station/narrate.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                assert name not in forbidden and root not in forbidden, \
                    f"{path.name} imports {name}"


def test_the_tool_runs_over_a_real_log_and_reports_alarms_in_its_exit_code():
    log = LOGS / "mailfetch-ardop-ww2mi-4.log"
    if not log.is_file():
        pytest.skip(f"{log} not present")
    r = subprocess.run([sys.executable, str(TOOL), str(log), "--no-color"],
                       capture_output=True, text=True, timeout=120, cwd=REPO)
    assert r.returncode == 1, r.stderr[-1500:]
    assert "CONNECTED WW2MI" in r.stdout
    assert "ALARM" in r.stdout
    assert "-- state at 21:50:37" in r.stdout


def test_a_utc_stamped_mail_line_is_read_without_moving_the_clock():
    """`winlink.client` stamps its lines UTC — that is what lets `rehear` put a
    session beside the tape — and the rest of the stream is the local clock.
    Reading one as the other would carry the whole narration an offset away."""
    events, n = narrate([
        "2026-09-08 21:00:00,000 INFO CONNECTED KE8LVA @ 1500 Hz",
        "2026-09-09 02:00:01,100Z mail: stage awaiting greeting",
        "2026-09-08 21:00:02,000 INFO TX DATA 1.2s",
    ])
    assert n.stage == "awaiting greeting"
    assert [e.at for e in events if e.at] == ["21:00:00", "21:00:00", "21:00:02"]
