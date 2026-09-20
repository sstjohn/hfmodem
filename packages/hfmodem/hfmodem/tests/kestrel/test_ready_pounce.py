# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Fresh occupancy decisions with a ready caller; no codec, RF, or DSP work."""
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from . import corpora

_LAUNCHER = Path(__file__).resolve().parents[5] / "tools/onair.sh"
requires_launcher = pytest.mark.skipif(
    not _LAUNCHER.exists(), reason="tools/onair.sh not present (installed-wheel run)")


@pytest.fixture
def ready(monkeypatch):
    kc = corpora.harness("kestrel_connect")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(kc.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(kc.time, "sleep", lambda secs: setattr(clock, "now", clock.now + secs))

    class Io:
        rx_device = "mock receiver"
        rig = None

        def __init__(self, busy):
            self.busy = iter(busy)
            self.events = []
            self.levels = (-30.0, -40.0)

        def start(self):
            self.events.append((clock.now, "ready"))

        def channel_busy(self, seconds, bw):
            self.events.append((clock.now, "sense", seconds, bw))
            clock.now += seconds
            occupied = next(self.busy)
            return occupied, 5.0 if occupied else -2.0

        def rx_level_range(self, _seconds):
            return self.levels

        def prime_floor(self):
            self.events.append((clock.now, "consume"))

    return kc, clock, Io


def test_returning_occupant_restarts_clear_count_in_ready_caller(ready):
    kc, clock, Io = ready
    io = Io([False, False, True, False, False, False])
    hs = SimpleNamespace(state=kc.VaraState.CONNECTED)
    hs.originate = lambda _gw: io.events.append((clock.now, "originate")) or True
    assert kc.connect("W1AAA", "W9SSJ", "500", io, hs=hs, pounce_wait=60)
    assert [e[1] for e in io.events] == ["ready"] + ["sense", "consume"] * 6 + ["originate"]
    # No second 8s check, caller startup, or added sleep after the final clear.
    assert io.events[-1][0] == io.events[-2][0] == pytest.approx(18.3)
    assert all(e[2:] == (3.0, "500") for e in io.events if e[1] == "sense")


def test_pounce_budget_does_not_restart_or_accept_partial_window(ready):
    kc, clock, Io = ready
    io = Io([False, False, True, False, False])
    with pytest.raises(kc.ChannelWaitExpired):
        kc.wait_clear_channel(io, "2750", 8, False, 14)
    assert clock.now == 12
    assert len([e for e in io.events if e[1] == "sense"]) == 4


def test_processing_cannot_make_an_expired_decision_authorize_tx(ready):
    kc, clock, Io = ready
    io = Io([False, False, False])
    original = io.prime_floor

    def slow_floor():
        original()
        clock.now += 0.5

    io.prime_floor = slow_floor
    with pytest.raises(kc.ChannelWaitExpired):
        kc.wait_clear_channel(io, "2750", 8, False, 10.25)
    assert clock.now == 10.5


def test_normal_arm_uses_one_eight_second_window(ready):
    kc, clock, Io = ready
    io = Io([False])
    assert kc.wait_clear_channel(io, "2300", 8, False)
    assert clock.now == 8
    assert io.events == [(0, "sense", 8, "2300"), (8, "consume")]


def test_normal_busy_arm_refuses_with_existing_status(ready):
    kc, _, Io = ready
    with pytest.raises(kc.ChannelBusy):
        kc.wait_clear_channel(Io([True]), "2750", 8, False)
    assert kc.CHANNEL_BUSY == 4


def test_force_records_one_window_and_bypasses_wait_not_dead_receiver(ready):
    kc, clock, Io = ready
    assert kc.wait_clear_channel(Io([True]), "500", 8, True, 60)
    assert clock.now == 8
    io = Io([True])
    io.levels = (-99, -99)
    assert not kc.wait_clear_channel(io, "500", 8, True, 60)
    assert io.events == [(8, "sense", 8, "500")]


def shell_gate(modem, *args):
    function = re.search(r"(?ms)^gate_channel\(\).*?^}", _LAUNCHER.read_text()).group()
    script = '''set -euo pipefail
POUNCE_WAIT=60
AUDIO=mock
CHANNEL_BUSY=4
die() { echo "refused: $*"; exit 1; }
refuse_busy() { exit 4; }
tune() { echo "tune $*"; }
gate_rx_level() { echo "receiver $KEY_FORCE"; }
sense_channel() { echo "shell-sense $*"; }
# The gate reports where its pre-gate seconds went, from marks this slice does not
# carry. Stubbed like every other collaborator here, so the extraction stays a test
# of the gate's own decisions rather than of what happens to sit above it.
before_the_gate() { :; }
''' + function + '''
gate_channel 7103800 "$@"
printf 'force=%s\\n' "$KEY_FORCE"
printf 'arg=%s\\n' "${KEY_ARGS[@]}"
'''
    return subprocess.run(["bash", "-c", script, "test-gate", modem, "500", *args],
                          capture_output=True, text=True, timeout=5)


@pytest.mark.parametrize("force", [False, True])
@requires_launcher
def test_vara_shell_defers_occupancy_but_retains_tune_receiver_and_force(force):
    result = shell_gate("vara", "--listen-first", "0", *(["--force"] if force else []))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "tune 7103800"
    assert lines[1] == "receiver " + ("--force" if force else "")
    assert not any("shell-sense" in line for line in lines)
    assert lines[-4:] == ["arg=--pounce-wait", "arg=" + ("0" if force else "60"),
                          "arg=--listen-first", "arg=8"]
    assert "force=" + ("--force" if force else "") in lines


@requires_launcher
def test_other_modems_keep_their_shell_occupancy_gate():
    result = shell_gate("pactor", "--max-calls", "3")
    assert result.returncode == 0
    assert result.stdout.splitlines()[:2] == ["shell-sense 7103800 pactor 500 60", "receiver "]
    assert "pounce-wait" not in result.stdout


def test_pounce_cannot_be_disabled_by_listen_first_zero(ready):
    kc, clock, Io = ready
    assert kc.wait_clear_channel(Io([False] * 3), "500", 0, False, 60)
    assert clock.now == 9


@pytest.mark.parametrize("budget", ["-1", "nan", "inf"])
def test_invalid_pounce_budget_is_refused_before_hardware(monkeypatch, budget):
    import sys
    kc = corpora.harness("kestrel_connect")
    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", "W1AAA",
                                      "--mycall", "W9SSJ", "--pounce-wait", budget])
    with pytest.raises(SystemExit) as exc:
        kc.main()
    assert exc.value.code == 2


def test_exhausted_wait_reaches_shell_and_closes_audio(monkeypatch):
    import sys
    kc = corpora.harness("kestrel_connect")
    stopped = []
    monkeypatch.setattr(kc, "AudioVaraIO", lambda *a, **kw: SimpleNamespace(
        stop=lambda: stopped.append(True)))
    monkeypatch.setattr(kc, "VaraStationHandshake", lambda *a, **kw: object())
    monkeypatch.setattr(kc, "attach_mail", lambda *a, **kw: (None, None))

    def expired(*a, **kw):
        assert kw["pounce_wait"] == 60
        raise kc.ChannelWaitExpired

    monkeypatch.setattr(kc, "connect", expired)
    monkeypatch.setattr(sys, "argv", ["kestrel_connect", "--gateway", "W1AAA",
                                      "--mycall", "W9SSJ", "--pounce-wait", "60"])
    assert kc.main() == 3
    assert stopped == [True]
