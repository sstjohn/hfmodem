# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The transmitter: the refusals, the kill paths, and what a radio-free bench
can and cannot prove.

Provable here, against a fake rigctld and a recording keying line: every refusal,
the order of key-up and key-down, that no CAT command runs while keyed, that the
watchdog deadline is derived from the burst, that the panic path holds no lock and
never reopens, that an unconfirmed unkey retires the rig, and that the regulatory
gate is consulted on the dial the radio reports.

Not provable here, and marked `hardware` where it is attempted at all: that a pin
moves, that HUPCL drops the line when the process dies, actuation latency, and the
FT-891's response to an RTS edge from a foreign fd.
"""
from __future__ import annotations

import threading
import time

import pytest

from hfmodem.core.regulatory import Control, Emission, Unregulated, centred
from hfmodem.core.regulatory.part97 import Licence, Part97
from hfmodem.core.rig import Cat, KeyToken, Rig, RigError, StuckTransmitter
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld

BENCH = Unregulated(because="unit test, no radio present")


@pytest.fixture
def fake():
    f = FakeRigctld()
    yield f
    f.close()


@pytest.fixture(autouse=True)
def _no_leaked_timers():
    """A rig left holding a watchdog leaks a Timer into whatever runs next.

    That is not hypothetical: it killed a whole pytest run, because a stale
    watchdog re-entered the panic path and the re-entry guard called os._exit.
    The guard is now a return, and rigs get closed anyway.
    """
    made: list[Rig] = []
    Rig._made = made
    yield made
    for r in made:
        try:
            r.close()
        except Exception:      # noqa: BLE001 — teardown must not mask a failure
            pass


def make(fake, *, ptt=None, profile=BENCH, transmit=True, armed=True,
         control=Control.LOCAL, mycall="N0CALL", **kw) -> Rig:
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=ptt or FakePtt(),
            profile=profile, control=control, mycall=mycall, transmit=transmit, **kw)
    if armed:
        r.cat.open()
        r._armed = True
        r._dial_hz = fake.freq
    getattr(Rig, "_made", []).append(r)
    return r


def burst(dial=None, bw=500.0, **kw) -> Emission:
    return centred(dial if dial is not None else 7_100_000, bw, **kw)


# --- refusals: nothing keys until everything is true --------------------------

def test_transmit_false_is_a_hard_interlock(fake):
    """Not a preference. The station file's default, and what to set while
    working on the code."""
    r = make(fake, transmit=False)
    with pytest.raises(RigError, match="transmit is disabled"):
        r.key(burst(), why="test", duration_s=1.0)


def test_an_unarmed_rig_refuses(fake):
    r = make(fake, armed=False)
    r.cat.open()
    with pytest.raises(RigError, match="arm"):
        r.key(burst(), why="test", duration_s=1.0)


def test_a_retired_rig_refuses_forever(fake):
    """A transmitter nobody has confirmed is down is not one this program may
    key again."""
    r = make(fake)
    r._retire("an earlier unkey was never confirmed")
    assert r.retired
    with pytest.raises(RigError, match="retired"):
        r.key(burst(), why="test", duration_s=1.0)


def test_keying_twice_is_refused(fake):
    r = make(fake)
    r.key(burst(), why="first", duration_s=0.1)
    with pytest.raises(RigError, match="already keyed"):
        r.key(burst(), why="second", duration_s=0.1)
    r.unkey()


# --- the gate cannot be bypassed ---------------------------------------------

def test_the_regulatory_profile_is_consulted_before_the_line_goes_up(fake):
    """An earlier design had a good gate that nothing called. Making the emission
    an argument to key() means forgetting is a type error."""
    ptt = FakePtt()
    r = make(fake, ptt=ptt, profile=Part97(licence=Licence.GENERAL),
             control=Control.AUTOMATIC)
    fake.freq = 14_078_500          # 14.080 centre: outside §97.221(b)
    with pytest.raises(RigError, match=r"97\.221"):
        r.key(burst(bw=2300.0), why="wide and unattended", duration_s=1.0)
    assert ptt.line is False, "the line went up before the check"


def test_the_check_uses_the_dial_the_radio_reports_not_the_one_we_asked_for(fake):
    """The dial does not stay where you left it."""
    r = make(fake, profile=Part97(licence=Licence.GENERAL), control=Control.AUTOMATIC)
    fake.freq = 13_000_000          # somewhere it may not transmit at all
    with pytest.raises(RigError, match=r"97\.301"):
        r.key(burst(dial=7_100_000), why="stale dial", duration_s=1.0)


def test_a_permitted_emission_keys(fake):
    ptt = FakePtt()
    r = make(fake, ptt=ptt, profile=Part97(licence=Licence.GENERAL))
    fake.freq = 7_100_000
    token = r.key(burst(), why="ok", duration_s=0.1)
    assert isinstance(token, KeyToken) and ptt.line is True
    r.unkey(token)
    assert ptt.line is False


# --- no CAT while keyed ------------------------------------------------------

def test_no_cat_command_runs_while_keyed(fake):
    """The single point every account of a stuck transmitter here agrees on."""
    r = make(fake)
    token = r.key(burst(), why="test", duration_s=0.5)
    with pytest.raises(RigError, match="while keyed"):
        r.cat.freq()
    with pytest.raises(RigError, match="while keyed"):
        r.tune(7_101_500)
    r.unkey(token)
    r.cat.freq()        # fine again


def test_dial_does_not_query_the_radio_while_keyed(fake):
    r = make(fake)
    token = r.key(burst(), why="test", duration_s=0.5)
    assert r.dial() == r._dial_hz       # the cached value, no CAT traffic
    r.unkey(token)


# --- the watchdog is per burst ----------------------------------------------

@pytest.mark.parametrize("dur,expect", [
    (0.1, 2.1),         # slack floor dominates a short burst
    (1.25, 3.25),       # PACTOR's cycle
    (60.0, 30.0),       # clamped by max_key_s
])
def test_the_deadline_comes_from_the_burst_not_a_flat_ceiling(fake, dur, expect):
    """A flat 30 s cap lets a 1.25 s burst whose audio never arrives hold an
    unmodulated carrier for twenty-nine seconds of someone else's channel."""
    r = make(fake, max_key_s=30.0)
    assert r._deadline(dur, 0.0) == pytest.approx(expect, rel=1e-6)


def test_the_watchdog_brings_the_key_down_and_retires_the_rig(fake):
    ptt = FakePtt()
    r = make(fake, ptt=ptt, max_key_s=0.15)
    r.key(burst(), why="a caller that never returns", duration_s=0.01)
    assert ptt.line is True
    deadline = time.monotonic() + 3.0
    while ptt.line and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ptt.line is False, "the watchdog never fired"
    assert r.retired and "keyed past" in r.retired_why


def test_a_clean_unkey_cancels_the_watchdog(fake):
    ptt = FakePtt()
    r = make(fake, ptt=ptt, max_key_s=0.2)
    r.unkey(r.key(burst(), why="test", duration_s=0.01))
    time.sleep(0.4)
    assert not r.retired, "the watchdog fired after a clean unkey"


# --- the panic path ---------------------------------------------------------

def test_panic_releases_the_line_and_never_reopens(fake):
    ptt = FakePtt()
    r = make(fake, ptt=ptt)
    r.key(burst(), why="test", duration_s=1.0)
    r.panic_unkey("bench")
    assert ptt.released and ptt.line is False
    assert "release" in ptt.calls
    assert not any(c.startswith("assert up") for c in ptt.calls[ptt.calls.index("release"):]), (
        "something asserted the line after release — a reopen re-raises RTS")


def test_panic_has_no_keyed_guard(fake):
    """A guard that can suppress the panic is not a safety mechanism."""
    ptt = FakePtt()
    r = make(fake, ptt=ptt)
    assert not r.keyed
    r.panic_unkey("not keyed as far as we know")
    assert ptt.released


def test_panic_confirms_through_an_independent_connection(fake):
    r = make(fake)
    r.key(burst(), why="test", duration_s=1.0)
    r.cat.close()                       # the persistent socket is gone
    r.panic_unkey("bench")              # must still reach rigctld
    assert "T 0" in fake.log and "t" in fake.log


def test_an_unconfirmable_unkey_raises_and_retires(fake):
    """Loud on purpose, and `retired` is set before the raise so the program has
    already stopped being able to key even if nobody catches this.

    Both witnesses have to be blind for this to be the right outcome: a driver
    that will not answer TIOCMGET, and a daemon that will not report PTT down.
    With either one talking, the alarm would be a lie.
    """
    r = make(fake, ptt=FakePtt(unreadable=True))
    r.key(burst(), why="test", duration_s=1.0)
    fake.lie_ptt = True                 # rigctld will never report it down
    with pytest.raises(StuckTransmitter, match="remove power"):
        r.panic_unkey("bench")
    assert r.retired


def test_the_keying_line_confirms_what_the_daemon_cannot(fake, capfd):
    """`ptt_type=None` is the only configuration the arm gate accepts, and on it
    both `T 0` and `t` come back ENAVAIL. So the independent CAT path carries no
    evidence on the station as it is actually run.

    That used to mean every panic ended in "CHECK THE RADIO AND REMOVE POWER" —
    on every watchdog and every Ctrl-C — which is exactly how an operator learns
    to stop reading the loudest message in the program.

    The keying line is asked instead, and it is the better witness anyway: it is
    what keys this transmitter, established by asserting each line in turn and
    watching the receiver go deaf on RTS and not on DTR. `release()` reads it back
    between the clear and the close, which is the only moment it can.
    """
    fake.ptt_type = "None"
    r = make(fake)
    assert r.cat.ptt_state() is None, "a daemon with no PTT cannot report one"
    r.key(burst(), why="test", duration_s=1.0)
    r.panic_unkey("bench")               # no alarm: the line answered
    assert r.retired
    assert "keying line reads low" in capfd.readouterr().err


def test_panic_takes_no_lock(fake):
    """The hazard: the arbiter holds its lock, calls key(), blocks in rigctld; the
    watchdog fires and waits behind it. Deadlock, transmitter keyed."""
    r = make(fake)
    r.key(burst(), why="test", duration_s=5.0)
    r._lock.acquire()                   # simulate a caller mid-key
    done = threading.Event()

    def panic():
        try:
            r.panic_unkey("while the lock is held")
        except StuckTransmitter:
            pass
        done.set()

    threading.Thread(target=panic, daemon=True).start()
    assert done.wait(3.0), "panic_unkey blocked on a lock"
    r._lock.release()


def test_a_keying_line_that_refuses_to_assert_retires_rather_than_pretending(fake):
    r = make(fake, ptt=FakePtt(refuse_assert=True))
    with pytest.raises(RigError, match="cannot key"):
        r.key(burst(), why="test", duration_s=0.1)
    assert r.retired
    assert not r.keyed


def test_a_line_that_reads_high_after_deassert_escalates(fake):
    """It panics and retires, but does not raise — because the radio, asked
    independently, says it is not transmitting.

    That distinction is the point: a stuck line *sensor* is not a stuck
    transmitter, and the radio's own report is better evidence about the air than
    our view of a serial pin. So the rig is retired — something is wrong with the
    keying path and a human should look before it keys again — while the
    unrecoverable case is left to mean what it says.
    """
    r = make(fake, ptt=FakePtt(stick=True))
    token = r.key(burst(), why="test", duration_s=0.1)
    r.unkey(token)
    assert r.retired and "reads high" in r.retired_why
    assert not r.keyed


# --- CAT hygiene ------------------------------------------------------------

def test_an_in_band_negative_rprt_is_an_error_not_a_success(fake):
    """rigctld reports failure by returning `RPRT <negative>` rather than by
    raising. An unchecked reply is an accepted write that did nothing."""
    r = make(fake)
    fake.refuse = True
    with pytest.raises(RigError, match="refused"):
        r.cat.set_freq(7_100_000)


def test_every_command_asks_for_a_framed_reply(fake):
    """`+` is the whole of the framing. A plain get answers with the bare value and
    no terminator, so a reader that waits for `RPRT ` waits for the next command's
    reply — or forever."""
    cat = Cat("127.0.0.1", fake.port)
    cat.open()
    try:
        cat.freq()
        cat.set_freq(7_100_000)
        cat.model_name()
        assert fake.log and all(c.startswith("+") for c in fake.log), fake.log
        assert not any("extended_resp" in c for c in fake.log), (
            "hamlib 5 has no such setting: it answers `RPRT -1` and closes the "
            "connection, and the framing was never in it to begin with")
    finally:
        cat.close()


def test_a_daemon_that_hangs_up_during_setup_fails_loudly(fake):
    """Refusing a setup command and dropping the connection are one event, and the
    refusal arrives first.

    Swallowing the refusal returned a Cat whose socket was already gone, so every
    later command said "CAT is not open" — from a caller that had done nothing
    wrong, hours of debugging away from the daemon that hung up.
    """
    fake.no_extended = True
    cat = Cat("127.0.0.1", fake.port, timeout=1.0)
    with pytest.raises(RigError, match="closed the connection"):
        cat.open()


@pytest.mark.realtime
def test_a_reply_with_no_terminator_times_out_rather_than_hanging(fake):
    """The other half of the same discovery: a daemon that ignores the `+` answers
    `f` with `7097500` and nothing else. There is no terminator to read to, so the
    read has to end at its own deadline and say what happened."""
    fake.no_terminator = True
    cat = Cat("127.0.0.1", fake.port, timeout=0.3)
    cat.open()                  # a *set* still answers `RPRT 0`, so setup passes
    try:
        t0 = time.monotonic()
        with pytest.raises(RigError, match="did not finish a reply"):
            cat.freq()
        assert time.monotonic() - t0 < 2.0
    finally:
        cat.close()


def test_no_rigctld_says_so_plainly():
    cat = Cat("127.0.0.1", 1)           # nothing listens on port 1
    with pytest.raises(RigError, match="no rigctld"):
        cat.open()


def test_tune_believes_the_readback(fake):
    r = make(fake)
    r.tune(7_101_500)
    assert r.dial() == 7_100_000        # centre - 1500


def test_tune_refuses_while_keyed(fake):
    r = make(fake)
    token = r.key(burst(), why="test", duration_s=0.5)
    with pytest.raises(RigError, match="while keyed"):
        r.tune(7_101_500)
    r.unkey(token)


# --- the arm gate -----------------------------------------------------------

def test_arm_refuses_the_wrong_radio(fake):
    fake.model = "Icom IC-7300"
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=FakePtt(),
            profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    with pytest.raises(RigError, match="wrong radio"):
        r.arm(prove_ptt=False)


def test_arm_refuses_an_implausible_dial(fake):
    fake.freq = 42
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=FakePtt(),
            profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    with pytest.raises(RigError, match="not an HF dial"):
        r.arm(prove_ptt=False)


def test_arm_refuses_a_radio_that_is_already_transmitting(fake):
    fake.lie_ptt = True
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=FakePtt(),
            profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    with pytest.raises(RigError, match="already transmitting"):
        r.arm(prove_ptt=False)


def test_a_dry_arm_reports_ptt_unproven(fake):
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=FakePtt(),
            profile=BENCH, control=Control.LOCAL, mycall="N0CALL", transmit=True)
    report = r.arm(prove_ptt=False)
    assert report.ptt_proven is False
    assert "UNPROVEN" in str(report)
    assert report.dial_hz == fake.freq


# --- the arm gate's transmit path -------------------------------------------
#
# These exist because a review traced coverage line by line and found that
# `_prove_ptt` and the ownership check — the two functions in this module that put
# RF on the air — had none at all, because every test set `_armed = True` directly.
# Three of the findings it reported lived exactly there.

def full(fake, **kw):
    """A rig that has not been hand-armed, so arm() really runs.

    `ptt_type = None` is the only daemon configuration the arm gate accepts, and
    it is the one the bench daemon reports. It is also the reason the proof below
    cannot end in PTT PROVEN: a rig with no PTT of its own answers `t` with
    ENAVAIL, so CAT has nothing to say about a line it does not own.
    """
    ptt = kw.pop("ptt", None) or FakePtt()
    fake.ptt_type = kw.pop("ptt_type", "None")
    fake.ptt_line = ptt
    r = Rig(model="ft891", cat=Cat("127.0.0.1", fake.port), ptt=ptt,
            profile=kw.pop("profile", BENCH), control=Control.LOCAL,
            mycall=kw.pop("mycall", "N0CALL"), transmit=kw.pop("transmit", True), **kw)
    getattr(Rig, "_made", []).append(r)
    return r, ptt


def test_proving_the_line_is_a_transmit_path_and_obeys_transmit_false(fake):
    """arm(prove_ptt=True) keys. The module claims nothing keys while transmit is
    false; that was true of key() and false of arm()."""
    r, _ = full(fake, transmit=False)
    with pytest.raises(RigError, match="transmit = false"):
        r.arm(prove_ptt=True)
    r.arm(prove_ptt=False)                  # the non-transmitting path is fine


def test_proving_the_line_refuses_on_a_retired_rig(fake):
    r, _ = full(fake)
    r._retire("an earlier unkey was never confirmed")
    with pytest.raises(RigError, match="retired"):
        r.arm(prove_ptt=True)


def test_the_proof_needs_no_callsign_because_it_identifies_nothing(fake):
    """The proof keys a sideband transmitter with no audio, which emits no RF.
    An earlier version required station.mycall on the claim that it sent the
    callsign and discharged §97.119; nothing was sent, so the requirement
    enforced an identification that never happened."""
    r, ptt = full(fake, mycall="")
    r.arm(prove_ptt=True)
    assert ptt.line is False and not r.keyed


def test_the_proof_keys_briefly_and_leaves_the_line_down(fake):
    r, ptt = full(fake)
    report = r.arm(prove_ptt=True)
    assert ptt.line is False
    assert not r.keyed and not r.retired
    assert "assert up" in ptt.calls and ptt.calls[-1] == "assert down"
    assert report.ptt_owner_checked is True


def test_the_proof_cannot_be_proven_by_the_daemon_this_station_requires(fake):
    """PTT PROVEN is unreachable, and the report has to say which kind of silence
    it met.

    The arm gate accepts `ptt_type=None` and nothing else, because two owners of
    one transmitter will key it when nobody asked. A rig with no PTT of its own
    answers `t` with ENAVAIL — measured on the bench daemon — so on the only
    configuration this station will run, CAT can neither confirm the line nor deny
    it. That is a missing measurement, not a failed one, and a report that
    conflated them would fail a correctly wired station.
    """
    r, _ = full(fake)
    report = r.arm(prove_ptt=True)
    assert report.ptt_proven is False
    assert "UNPROVEN" in str(report)
    assert r.cat.ptt_readable is False
    assert any("ENAVAIL" in n for n in report.notes), report.notes
    assert not any("does not report" in n for n in report.notes), report.notes


@pytest.mark.realtime
def test_the_proof_finishes_well_inside_its_own_watchdog(fake):
    """It armed 2.0 s and then sent a callsign taking 4.38 s at 20 WPM, so the
    watchdog fired mid-proof, retired the rig, and left the loop keying a released
    port. The deadline now comes from the proof's own length."""
    r, ptt = full(fake)
    t0 = time.monotonic()
    r.arm(prove_ptt=True)
    assert time.monotonic() - t0 < 1.5
    assert not r.retired, "the watchdog fired during its own proof"


def test_the_proof_never_asks_the_radio_to_key(fake):
    """Finding out who owns the line by sending `T 1` is a CAT write, which this
    module forbids — and on a daemon configured for CAT PTT it genuinely keys an
    unidentified carrier with no watchdog armed."""
    r, _ = full(fake)
    r.arm(prove_ptt=True)
    # The log keeps the `+` a framed command arrives with, so the guard has to
    # look past it or a framed `T 1` would walk straight through.
    assert not any(c.lstrip("+").startswith("T 1") for c in fake.log), fake.log
    assert any("get_conf" in c for c in fake.log)


def test_a_daemon_that_keys_the_radio_itself_is_refused(fake):
    r, _ = full(fake, ptt_type="RTS")
    with pytest.raises(RigError, match="key the radio itself"):
        r.arm(prove_ptt=True)


def test_a_daemon_that_will_not_disable_its_cache_says_so_in_the_report(fake):
    """`t` then reports hamlib's memory of its own last command rather than the
    radio, and nothing downstream may treat it as evidence."""
    fake.no_cache = True
    r, _ = full(fake)
    report = r.arm(prove_ptt=False)
    assert any("cache" in n for n in report.notes), report.notes


def test_a_rig_that_retires_while_arming_raises_rather_than_reporting_success(fake):
    """The proof's teardown can panic, and an ArmReport that reads like success
    for a rig which will refuse every burst is worse than an exception.

    The line sticks up after deassert; `lie_ptt` is deliberately NOT set, because
    that would make the arm gate refuse at its already-transmitting check and the
    proof would never run.
    """
    r, ptt = full(fake, ptt=FakePtt(stick=True))
    with pytest.raises(RigError):
        r.arm(prove_ptt=True)
    assert r.retired, "arming reported success for a rig that had already retired"


def test_a_radio_that_will_not_name_itself_is_refused(fake):
    fake.model = ""
    r, _ = full(fake)
    with pytest.raises(RigError, match="would not say what radio"):
        r.arm(prove_ptt=False)


def test_a_second_panic_repeats_the_first_ones_verdict(fake):
    """`panic()` starts from `confirmed = True` and only clears it by catching
    this, so a re-entry that merely returned laundered an unconfirmed unkey into
    a clean exit. The operator's Ctrl-C after a stuck watchdog is exactly that
    sequence, and exit 0 is what invites a supervisor to restart into a keyed
    radio."""
    r = make(fake, ptt=FakePtt(unreadable=True))
    r.key(burst(), why="test", duration_s=1.0)
    fake.lie_ptt = True
    with pytest.raises(StuckTransmitter):
        r.panic_unkey("watchdog")
    with pytest.raises(StuckTransmitter, match="an earlier panic"):
        r.panic_unkey("signal")


def test_a_second_panic_after_a_confirmed_one_stays_quiet(fake):
    r = make(fake, ptt=FakePtt())
    r.key(burst(), why="test", duration_s=1.0)
    r.panic_unkey("watchdog")
    r.panic_unkey("signal")             # confirmed down; nothing to re-raise
