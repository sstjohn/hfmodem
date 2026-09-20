# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Every tools/ entry point must start.

The tool that drives the transmitter had no test at all, which is how tool
defects become on-air incidents. `--help` is a low bar, but it reaches argparse
and therefore catches a bad import, a syntax error in a rarely-taken path, and a
flag that references a name that no longer exists — the class of breakage that
otherwise surfaces at the radio with the operator waiting.

These are receive-only invocations: `--help` exits before any device, socket or
transmitter is touched.
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.tests.kestrel import corpora

_ROOT = Path(__file__).resolve().parents[5]
_TOOLS = _ROOT / "tools"

pytestmark = pytest.mark.skipif(not _TOOLS.is_dir(),
                                reason="tools/ not present (installed-wheel run)")

_SKIP = {"channel_busy.py"}          # a module, not a CLI


def _clis() -> list[Path]:
    return sorted(p for p in _TOOLS.glob("*.py") if p.name not in _SKIP)


def _kestrel_connect():
    """The connect tool as a module. It lives in tools/, not in the package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kestrel_connect_cli", _TOOLS / "kestrel_connect.py")
    kc = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kc
    sys.path.insert(0, str(_TOOLS))
    try:
        spec.loader.exec_module(kc)
    finally:
        sys.path.remove(str(_TOOLS))
    return kc


@pytest.mark.parametrize("tool", _clis(), ids=lambda p: p.stem)
def test_tool_starts(tool: Path):
    r = subprocess.run([sys.executable, str(tool), "--help"], env=corpora.child_env(),
                       capture_output=True, text=True, timeout=120, cwd=_ROOT)
    assert r.returncode == 0, f"{tool.name} --help failed:\n{r.stderr[-1500:]}"
    assert r.stdout.strip(), f"{tool.name} --help printed nothing"


def test_the_connect_tool_unkeys_on_termination_signals(monkeypatch):
    """SIGTERM, SIGHUP and SIGINT must each reach the rig's shutdown before the
    process exits: every other unkey hangs off a `finally`, and a `finally`
    runs when Python unwinds, not when the process is killed. A session that
    was holding the key when it was stopped from outside kept transmitting on
    2026-08-04 until the operator cut power — and timeout(1), a supervisor and
    a dropped terminal are exactly the ordinary ways an unattended run stops."""
    import signal
    import types

    from . import corpora
    kc = corpora.harness("kestrel_connect")

    handlers: dict = {}
    monkeypatch.setattr(signal, "signal",
                        lambda sig, h: handlers.__setitem__(sig, h))
    monkeypatch.setattr("atexit.register", lambda f: None)
    panics = []
    kc._install_unkey_handlers(types.SimpleNamespace(
        panic=panics.append, shutdown=lambda: None))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        assert sig in handlers, f"no handler installed for {sig!r}"
        with pytest.raises(SystemExit):
            handlers[sig](sig, None)
    # panic, not a plain unkey: it retires the rig first and arms the deadman,
    # which are exactly what a wedged unwind cannot be trusted to do.
    assert panics == ["SIGTERM", "SIGHUP", "SIGINT"]


def test_transmitting_tools_require_an_explicit_arm():
    """Every tool that can key must not do so by default."""
    for name in ("kestrel_connect.py", "vara_rig_bridge.py", "onair_session.py",
                 "ptt_tail_check.py"):
        text = (_TOOLS / name).read_text()
        assert '"--arm"' in text, f"{name} has no --arm gate"
        assert 'action="store_true"' in text, f"{name}'s --arm is not opt-in"


def test_the_connect_tool_offers_only_the_bandwidth_it_speaks():
    """``--bw`` offers a value only when a whole session can be driven at it.

    It once offered 500 while ``VaraStationHandshake`` keyed the same BW2300 burst
    whatever it said, so the flag described a choice nobody had. Then the waveforms
    arrived — ``VF.CR500``, ``VF.CONNECT_RESPONSE_500``, and the BW500 link-setup a
    real VARA HF v4.9.0 answered with CONNECTED — and the flag still stayed out,
    because a RECEIVED burst was resolved to its kind by symbol count alone and the
    two bandwidths' pairs share theirs: 41 for the request, 23 for the response.

    The counts still collide and always will, at all three bandwidths. What
    changed is that the resolution takes the session bandwidth alongside the
    count, so a session reads what it hears at its own bandwidth. Both halves are
    asserted: the collision, because it is what makes the bandwidth argument
    necessary, and the resolver, because the flag is only honest while it is
    there.

    2750 stayed out longest, on the reading that a BW2750 responder answers the
    BW2300 request and brings the session up at 2300 — true, and beside the
    point: the 2026-07-21 loopback's BW2750 request is a different generator
    state from the BW2300 one on the same tape, and only that one brings a
    session up at 2750  [see VF.CR2750].
    """
    text = (_TOOLS / "kestrel_connect.py").read_text()
    assert 'choices=["2300", "2750", "500"]' in text, (
        "--bw no longer offers the bandwidths this tool can drive a session at")
    for other, wide in ((VF.CR500, VF.CR), (VF.CR2750, VF.CR),
                        (VF.CONNECT_RESPONSE_500, VF.CONNECT_RESPONSE),
                        (VF.CONNECT_RESPONSE_2750, VF.CONNECT_RESPONSE)):
        assert (len(other.preamble) + other.n_payload
                == len(wide.preamble) + wide.n_payload), (
            f"{other.name} no longer collides with {wide.name} on symbol count — "
            "the session bandwidth may no longer be needed to tell them apart")
    for bw, req, resp in (("500", VF.CR500, VF.CONNECT_RESPONSE_500),
                          ("2750", VF.CR2750, VF.CONNECT_RESPONSE_2750),
                          ("2300", VF.CR, VF.CONNECT_RESPONSE)):
        assert VA._kind_for(41, bw) is req
        assert VA._kind_for(23, bw) is resp


def test_a_target_listed_only_at_a_bandwidth_kestrel_cannot_call_is_said_out_loud(
        tmp_path):
    """Winlink lists stations at VARA 500, and calling one is 15 minutes of slot
    spent on a station that was never going to reply. VARA 2750 is not that case:
    a responder in BW2750 accepts kestrel's BW2300 request and brings the session
    up at 2300, so it stays callable at 2300 and stays quiet; called at 2750, only
    a VARA 2750 listing has been measured to answer.
    """
    kc = _kestrel_connect()
    csv_path = tmp_path / "gateways.csv"
    csv_path.write_text(
        "Callsign,Mode\n"
        "N0AAA,VARA 500\n"
        "N0BBB,VARA 2750\n"
        "N0CCC,VARA\n"
        "N0DDD,VARA 500\nN0DDD,VARA 2750\n"
        "N0EEE,VARA FM 9600\n")
    assert kc.unspeakable(csv_path, "N0AAA") == ["VARA 500"]
    assert kc.unspeakable(csv_path, "N0BBB") == []
    assert kc.unspeakable(csv_path, "N0CCC") == []
    assert kc.unspeakable(csv_path, "N0DDD") == []   # listed at both; 2750 is callable
    assert kc.unspeakable(csv_path, "N0EEE") == []   # a different waveform entirely
    assert kc.unspeakable(csv_path, "N0ZZZ") == []   # unlisted is unknown, not unreachable
    assert kc.unspeakable(tmp_path / "absent.csv", "N0AAA") == []

    # Reachability is per channel when the dial is known: a station callable
    # somewhere is not callable everywhere, and the 20 m plain-VARA listing must
    # not vouch for a 40 m channel listed VARA 500 only (K5DAT-13, 2026-08-03).
    chan = tmp_path / "channels.csv"
    chan.write_text(
        "Callsign,Frequency,Mode\n"
        'N0FFF,"7,104.700 KHz",VARA 500\n'
        'N0FFF,"14,108.000 KHz",VARA\n')
    assert kc.unspeakable(chan, "N0FFF") == []                    # station-wide: callable
    assert kc.unspeakable(chan, "N0FFF", 7103.2) == ["VARA 500"]  # this channel is not
    assert kc.unspeakable(chan, "N0FFF", 14106.5) == []
    assert kc.unspeakable(chan, "N0FFF", 3595.0) == []            # no listing at that dial:
    assert kc.unspeakable(csv_path, "N0AAA", 7103.2) == ["VARA 500"]  # station-wide again


def test_the_eligibility_guard_answers_the_same_on_the_dial_the_launcher_builds():
    """The published list, every row, asked both ways.

    ``unspeakable`` matches a listed centre against ``dial + 1500 Hz``, and the dial
    it is given comes from ``onair.sh``, which builds it with ``bc`` — and ``bc``
    truncates at whatever scale the launcher asks for. At ``scale=1`` a centre of
    7102.850 kHz is asked about as a dial of 7101.3 and the row it belongs to sits
    50 Hz away; against a window of 50 Hz that matched nothing and the guard fell
    back to the station-wide answer, which is a different question: KB3PCY 7107.250
    is listed VARA 500 only and refused on its true dial, and called on the
    launcher's.

    The scale is READ FROM THE LAUNCHER rather than restated, because that is the
    number that moves: it was 1 when this was written and is 3 now, and a copy of it
    here would have gone on describing a rounding the launcher had stopped doing.
    So the guard is asked about all 1144 published channels twice — on the exact dial
    and on the one the launcher would build — and the two verdicts have to agree. It
    is a real list rather than a fixture because the failure is a collision between a
    rounding rule and the distribution of published frequencies, and neither is
    inventable.
    """
    gateways = _ROOT / "winlink-vara-gateways.csv"
    if not gateways.exists():
        pytest.skip("the published gateway list is not in this tree")
    launcher = _TOOLS / "onair.sh"
    if not launcher.exists():
        pytest.skip("onair.sh not present")
    kc = _kestrel_connect()

    scale = re.search(r'echo "scale=(\d+); \(\$centre - 1500\)/1000" \| bc',
                      launcher.read_text())
    assert scale, "onair.sh no longer builds --freq with bc"
    step = 10 ** max(0, 3 - int(scale.group(1)))      # Hz the launcher truncates to

    rows = list(csv.DictReader(gateways.open(newline="")))
    assert len(rows) > 500, f"only {len(rows)} channels — the list has been truncated"
    disagreed = []
    for row in rows:
        call = row["Callsign"].strip()
        centre_hz = int(round(float(row["Frequency"].replace(",", "").split()[0]) * 1000))
        exact = (centre_hz - 1500) / 1000.0
        truncated = ((centre_hz - 1500) // step) * step / 1000.0
        if kc.unspeakable(gateways, call, exact) != kc.unspeakable(gateways, call,
                                                                   truncated):
            disagreed.append((call, centre_hz, exact, truncated))
    assert not disagreed, (
        f"{len(disagreed)} published channels get a different answer on the dial "
        f"onair.sh builds than on their exact one: {disagreed[:6]}")


def test_the_channel_window_can_never_reach_a_second_row(tmp_path):
    """The other side of that tolerance, which nothing held.

    Widening `_CHANNEL_TOL_KHZ` is the obvious repair for any channel that fails to
    match, and the test above only ever gets happier as it grows — every value from
    0.6 to 5.0 passed it. What a wide window costs is on the air: the tightest two
    channels any one station publishes are 0.5 kHz apart, so at 0.6 the row next
    door is admitted too, its plain-VARA listing vouches for a VARA 500 channel,
    and the tool calls a gateway that cannot hear a BW2300 connect-request. That is
    the exact refusal the per-channel question was added to make.
    """
    kc = _kestrel_connect()
    chan = tmp_path / "adjacent.csv"
    chan.write_text("Callsign,Frequency,Mode\n"
                    'N0FFF,"7,104.700 KHz",VARA 500\n'
                    'N0FFF,"7,105.200 KHz",VARA\n')
    assert kc.unspeakable(chan, "N0FFF", 7103.2) == ["VARA 500"]
    assert kc.unspeakable(chan, "N0FFF", 7103.7) == []


def _busy_io(margin: float = 7.5):
    """Enough of `AudioVaraIO` for the re-sense, answering OCCUPIED on a healthy
    receiver: the levels are the ones `receiver_fault` calls ordinary."""
    class Io:
        rx_device = "BlackHole 64ch"
        hs = None

        def start(self): ...
        def stop(self): ...
        def prime_floor(self): ...
        def channel_busy(self, seconds, bw): return True, margin
        def rx_level_range(self, seconds): return -42.0, -18.0
    return Io()


def test_a_late_busy_refusal_is_the_one_thing_that_spends_channel_busy(monkeypatch):
    """`pounce` keys again on exactly one status, and until now the refusal its
    own text describes could not reach it.

    The launcher's gate senses before the modem starts; kestrel senses AGAIN in
    the last moment before key-down, so an occupant who came back in between is
    refused here and nowhere else — and that refusal used to spend 1, which is
    also what a failed connect, a deaf receiver and an unconfigured station spend.
    Widening what counts as busy is not the fix: read that way, those three are an
    invitation to key the same transmitter five more times, and once were.
    """
    from . import corpora
    kc = corpora.harness("kestrel_connect")
    monkeypatch.setattr(kc.time, "sleep", lambda _s: None)

    with pytest.raises(kc.ChannelBusy):
        kc.connect("W1AAA", "W9SSJ", "2300", _busy_io(), listen_first=8.0)

    assert kc.CHANNEL_BUSY == 4, "kestrel and the launcher no longer agree on the status"


def test_a_deaf_receiver_on_a_busy_channel_is_still_an_ordinary_refusal(monkeypatch):
    """The constraint the retry ladder was corrected on. `receiver_fault` is read
    BEFORE the occupancy verdict and answers first, so the one arrangement that
    could quietly widen the new status — an input delivering nothing on a channel
    that also reads occupied — must still end the run rather than call again. No
    number of windows turns a gain up."""
    from . import corpora
    kc = corpora.harness("kestrel_connect")
    monkeypatch.setattr(kc.time, "sleep", lambda _s: None)

    io = _busy_io()
    io.rx_level_range = lambda seconds: (-99.0, -99.0)
    assert kc.connect("W1AAA", "W9SSJ", "2300", io, listen_first=8.0) is False


def test_the_late_busy_status_reaches_the_shell(monkeypatch):
    """And survives `main`'s teardown, which is where every other exit runs."""
    from . import corpora
    kc = corpora.harness("kestrel_connect")

    class Rig:
        keyed = False
        retired = False

        def __init__(self, *_a, **_kw):
            pass

        def identify(self):
            return "Yaesu FT-891", "7100000"

    stopped = []

    class Io:
        def __init__(self, *_a, **_kw):
            pass

        def stop(self):
            stopped.append(True)

    def refused(*_a, **_kw):
        raise kc.ChannelBusy

    monkeypatch.setattr(kc, "Rig", Rig)
    monkeypatch.setattr(kc, "AudioVaraIO", Io)
    monkeypatch.setattr(kc, "VaraStationHandshake", lambda *a, **kw: object())
    monkeypatch.setattr(kc, "attach_mail", lambda *a, **kw: (None, None))
    monkeypatch.setattr(kc, "_install_unkey_handlers", lambda rig: None)
    monkeypatch.setattr(kc, "connect", refused)
    monkeypatch.setattr(sys, "argv",
                        ["kestrel_connect.py", "--gateway", "W1AAA",
                         "--mycall", "W9SSJ", "--rigctld", "127.0.0.1:65000"])

    assert kc.main() == kc.CHANNEL_BUSY
    assert stopped, "the audio was left open by the refusal's early return"


def test_pounce_keys_again_on_that_status_and_on_no_other():
    """The shell half of the same convention, read out of the launcher itself."""
    text = (_TOOLS / "onair.sh").read_text()
    body = text[text.index("\npounce)"):text.index("\nlisten)")]
    assert '[ "$rc" -eq "$CHANNEL_BUSY" ] || exit "$rc"' in body, (
        "pounce no longer exits on everything but the busy status")
    assert '[ "$rc" -eq 3 ] && die' in body, (
        "a spent wait budget is no longer separated from an occupied channel")
