# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""sabir's on-air entry point: the refusals, and the shape of the one burst.

These pin the gates rather than the feature. sabir had no way to key a
transmitter at all until this existed, and what has to be true of the first one
it makes is not that a beacon goes out but that nothing goes out unarmed, on an
unverified dial, over somebody else's session, or for longer than it was meant
to. Every refusal below is a defect this station has already paid for once.

Nothing here opens a device. `arming_refusal` is a `stat`, the keying line is a
recorder, and the CAT calls and the channel sense are intercepted where the
entry point calls them.
"""
from __future__ import annotations

import atexit
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hfmodem.core import band, busy, cwid, regulatory
from hfmodem.core.ptt import PttError, arming_refusal
from hfmodem.sabir import onair, radio

#: `/dev/null` stats as a character device and drives no modem lines, so it gets
#: past the arm gate and fails at the ioctl — which is how a test proves every
#: gate in front of the line opened without needing a line.
FAKE_DEV = "/dev/null"

DIAL = 7_102_000
CHANNEL = DIAL + 1500

#: A phone segment. No licence class carries data there, so it is what the
#: regulatory gate refuses when nothing else about the run is wrong.
PHONE_CHANNEL = 7_200_000

#: The shipped profile rather than a stub, because a gate proved against a stub
#: is a gate proved against itself. 7.102 MHz sits well inside the General 40 m
#: data segment.
RULES = regulatory.profile("part97", licence="general")

#: What the presence beacon really occupies on this dial. The bounds tests below
#: play square arrays rather than beacons, but there is no such thing as audio
#: that goes out without describing itself, so they describe this.
EMISSION = onair.presence_burst("W9SSJ").emission(DIAL)


class RecordingKeyer:
    """What `LineKeyer` is to `radio.Rig`, minus the ioctl."""

    def __init__(self, *, confirms: bool = True) -> None:
        self.confirms = confirms
        self.must_retire = False
        self.events: list[str] = []

    def arm(self) -> None:
        self.events.append("arm")

    def key(self, on: bool, note: str = "") -> bool:
        self.events.append("key" if on else "unkey")
        return self.confirms if on else True

    def hand_back(self) -> None:
        self.events.append("hand_back")


def rig(ptt_device: str = FAKE_DEV, **kw) -> radio.Rig:
    r = radio.Rig("ft891", ptt_device, profile=RULES, control=onair.CONTROL,
                  rigctld="127.0.0.1:4532")
    r.dial_hz = DIAL                    # as a QSY would have left it
    r._keyer = RecordingKeyer(**kw)
    return r


def _qsy(offset: int = 0):
    """A rig that lands `offset` hertz from where it was asked.

    It records what it read back, as the real `qsy` does, because that — and not
    what was asked for — is the dial the emission is checked against.
    """
    def qsy(self, hz: int) -> int:
        self.dial_hz = hz + offset
        return self.dial_hz
    return qsy


def cli(*extra: str, channel: int = CHANNEL) -> list[str]:
    """A complete armed command line, for tests that remove one thing from it.

    Audio devices as indices rather than names: an index resolves without asking
    PortAudio anything, and nothing in this file may open a card.
    """
    return ["--mycall", "W9SSJ", "--channel", str(channel), "--transmit",
            "--ptt-device", FAKE_DEV, "--line-ptt",
            "--rigctld", "127.0.0.1:4532",
            "--regulatory", "part97", "--licence", "general",
            "--audio-in", "1", "--audio-out", "2", *extra]


@pytest.fixture
def radio_answers(monkeypatch):
    """A rig that identifies, tunes where it is told, and a clear channel."""
    monkeypatch.setattr(radio.Rig, "identify", lambda self: "Yaesu FT-891")
    monkeypatch.setattr(radio.Rig, "transmitting", lambda self: None)
    monkeypatch.setattr(radio.Rig, "qsy", _qsy())
    monkeypatch.setattr(radio.Rig, "set_mode", lambda self: None)
    monkeypatch.setattr(onair, "listen",
                        lambda *_a, **_k: onair.Sense("clear", -3.0))
    keyers: list[RecordingKeyer] = []

    def keyer(device, log, alarm=None):
        k = RecordingKeyer()
        keyers.append(k)
        return k
    monkeypatch.setattr(radio, "LineKeyer", keyer)
    monkeypatch.setattr(radio, "play_drained", lambda audio, fs, device: None)
    return keyers


#: Which dispositions `onair._install_signals` takes over, and this fixture gives
#: back.
_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


@pytest.fixture(autouse=True)
def _no_handler_outlives_its_run(monkeypatch):
    """`onair.run` takes over three signals and registers an atexit hook for the
    whole process. That is right for a run that owns a transmitter and wrong to
    leave behind in a test runner, and the difference cost a suite: sabir's
    orphaned SIGTERM handler caught a signal meant for pytest twenty minutes
    later, inside besra's decoder, printed `unkey: SIGTERM` for a rig long gone
    and raised SystemExit — and from there on Ctrl-C no longer interrupted a
    session attached to a radio.

    The atexit hook is recorded and dropped rather than registered: nothing here
    owns a transmitter at interpreter exit, and a hook closing over a
    `RecordingKeyer` would run during shutdown against a rig that does not
    exist. The assertion on the way in is what fails, loudly and in the next
    test, if this ever stops restoring.
    """
    monkeypatch.setattr(atexit, "register", lambda fn, *a, **kw: fn)
    before = {s: signal.getsignal(s) for s in _SIGNALS}
    escaped = [s.name for s, h in before.items()
               if getattr(h, "__qualname__", "").startswith("_install_signals")]
    assert not escaped, (
        f"a previous test left {', '.join(escaped)} bound to sabir's unkey "
        f"handler; process-global disposition does not belong to one test")
    yield
    for sig, handler in before.items():
        signal.signal(sig, handler)


# -- the arm gate ------------------------------------------------------------

def test_arming_without_a_keying_line_is_refused(capsys):
    """A session was once launched armed with no line named: rigctld took the
    unkey and never answered, and the last-resort unkey had nothing to pull
    down. What reaches the operator is the shared gate's own sentence, whole —
    a tool that says something else of its own is the drift that guards
    against."""
    rc = onair.main(["--mycall", "W9SSJ", "--channel", str(CHANNEL), "--transmit"])
    assert rc == 2
    assert arming_refusal(None) in capsys.readouterr().out


def test_arming_on_a_keying_line_that_is_not_there_is_refused(capsys):
    """The placeholder default: a device path that exists on no machine rode
    through a full session that logged 22.1 s of carrier while never touching
    the radio."""
    ghost = "/dev/cu.usbserial-XXXXB1"
    rc = onair.main(["--mycall", "W9SSJ", "--channel", str(CHANNEL),
                     "--transmit", "--ptt-device", ghost])
    assert rc == 2
    assert arming_refusal(ghost) in capsys.readouterr().out


def test_arming_through_the_daemon_is_refused(capsys):
    """rigctld answered every command until the first key-down of a session
    under RF and then answered nothing, twelve unkeys out of twelve. The daemon
    tunes; it does not key."""
    rc = onair.main(["--mycall", "W9SSJ", "--channel", str(CHANNEL),
                     "--transmit", "--ptt-device", FAKE_DEV])
    assert rc == 2
    assert "--line-ptt" in capsys.readouterr().out


def test_arming_without_a_callsign_is_refused(capsys):
    """The callsign inside a beacon is a digital code. What identifies this
    station is the Morse, and it needs a call to send."""
    rc = onair.main(["--channel", str(CHANNEL), "--transmit",
                     "--ptt-device", FAKE_DEV, "--line-ptt"])
    assert rc == 2
    assert "--mycall" in capsys.readouterr().out


def test_arming_with_everything_named_gets_past_the_gates(radio_answers, capsys):
    """A gate that also refuses the armed-correctly case is just the
    transmitter switched off."""
    assert onair.main(cli()) == 0
    out = capsys.readouterr().out
    assert "dial 7.102000 MHz verified" in out
    assert radio_answers[0].events == ["arm", "key", "unkey", "hand_back"]


# -- the dial ----------------------------------------------------------------

def test_no_channel_and_no_dial_is_refused(capsys):
    """The dial would be whatever the rig was last left on. On 2026-08-13 four
    attempts went out that way, some across the PACTOR channels, and the
    operator caught it at the rig where no log would have."""
    rc = onair.main(["--mycall", "W9SSJ", "--transmit", "--ptt-device", FAKE_DEV,
                     "--line-ptt", "--rigctld", "127.0.0.1:4532"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "NOT KEYING" in out and "--channel or --dial" in out


def test_a_dial_that_reads_back_wrong_refuses_and_keys_nothing(
        radio_answers, monkeypatch, capsys):
    monkeypatch.setattr(radio.Rig, "qsy", _qsy(500))
    assert onair.main(cli()) == 2
    assert "QSY FAILED" in capsys.readouterr().out
    assert not radio_answers[0].events, "the line was touched after a failed QSY"


def test_a_dial_that_reads_back_unreadable_refuses(radio_answers, monkeypatch,
                                                   capsys):
    """Unlike shrike, an unreadable answer refuses: the frequency read passes
    through the daemon, so nothing readable means nobody is talking to the
    radio."""
    monkeypatch.setattr(radio.Rig, "qsy", lambda self, hz: None)
    assert onair.main(cli()) == 2
    assert "nothing readable" in capsys.readouterr().out


def test_a_rig_something_else_is_holding_refuses(radio_answers, monkeypatch,
                                                 capsys):
    """One transceiver, several agents, no station-wide PTT owner: keying into
    another modem's transmission doses both receivers and corrupts both sets of
    measurements."""
    monkeypatch.setattr(radio.Rig, "transmitting", lambda self: True)
    assert onair.main(cli()) == 2
    assert "ALREADY transmitting" in capsys.readouterr().out
    assert not radio_answers[0].events


def test_a_daemon_that_cannot_report_ptt_is_not_a_refusal(radio_answers,
                                                          monkeypatch):
    """A rigctld started `ptt_type=None` -- the only configuration this station
    accepts, so the keying line has one owner -- answers `t` with ENAVAIL. An
    unreadable PTT and a PTT that reads busy mean opposite things."""
    monkeypatch.setattr(radio.Rig, "transmitting", lambda self: None)
    assert onair.main(cli()) == 0


def test_a_readback_inside_the_tolerance_is_accepted(radio_answers, monkeypatch):
    """The bar is wider than any readback error this rig has made and far
    narrower than any wrong channel."""
    monkeypatch.setattr(radio.Rig, "qsy", _qsy(band.QSY_TOLERANCE_HZ))
    assert onair.main(cli()) == 0


def test_the_tolerance_has_one_definition_for_every_transmit_path():
    """It was written down at four of them and one said 10 — silently stricter
    than the other three on the same rig, and nothing said which was meant."""
    assert not hasattr(radio, "QSY_TOLERANCE_HZ")
    assert band.QSY_TOLERANCE_HZ == 20


# -- the regulatory gate -----------------------------------------------------

def test_a_beacon_outside_the_licence_never_reaches_the_key(radio_answers, capsys):
    """The gate this path had none of. 7.200 MHz carries phone and no data for
    any class, and every other thing about this run is right — the line is
    named, the dial reads back, the channel is clear — so the only thing that
    can refuse it is the profile."""
    assert onair.main(cli(channel=PHONE_CHANNEL)) == 2
    out = capsys.readouterr().out
    assert "refused by part97" in out and "97.301" in out
    assert "key" not in radio_answers[0].events


def test_the_key_cannot_be_reached_without_an_emission():
    """Forgetting is a type error rather than an omission, at both doors: the
    rig will not take a profile it was not given, and neither `key` nor
    `transmit` has a way through without saying what goes out."""
    with pytest.raises(TypeError):
        radio.Rig("ft891", FAKE_DEV, rigctld="127.0.0.1:4532")
    r = rig()
    with pytest.raises(TypeError):
        r.key()
    with pytest.raises(TypeError):
        radio.transmit(r, np.ones(int(0.1 * radio.FS)))
    assert not r._keyer.events


def test_the_check_is_against_the_dial_the_rig_is_on():
    """`key()` re-places the emission on the rig's own dial rather than trusting
    the one it was handed. An emission built for a channel it may use does not
    become legal on a radio sitting somewhere else."""
    r = rig()
    r.dial_hz = PHONE_CHANNEL
    with pytest.raises(regulatory.NotPermitted, match=r"97\.301"):
        r.key(EMISSION)
    assert not r._keyer.events


def test_a_rig_that_has_read_back_no_dial_will_not_key():
    """There is nothing to place the emission on, and a frequency this process
    has not verified is exactly the one the whole dial gate exists to refuse."""
    r = rig()
    r.dial_hz = None
    with pytest.raises(regulatory.NotPermitted, match="not verified"):
        r.key(EMISSION)
    assert not r._keyer.events


def test_a_run_that_names_no_rules_is_refused(capsys):
    """`core.regulatory` ships no default profile, and this path had no way to
    name one at all — which is the same thing as operating under none."""
    args = [a for a in cli() if a not in ("--regulatory", "part97")]
    assert onair.main(args) == 2
    out = capsys.readouterr().out
    assert "--regulatory" in out and "has not said enough" in out


def test_part97_without_a_licence_class_is_refused(capsys):
    """Which HF segments carry data depends on the class, so a guess in the
    permissive direction is the one that transmits where it may not."""
    args = [a for a in cli() if a not in ("--licence", "general")]
    assert onair.main(args) == 2
    assert "--licence" in capsys.readouterr().out


def test_an_unregulated_station_still_has_to_say_why(capsys):
    """It is the override flag, so it costs a stated reason — a station running
    unchecked should not look, in its logs, like one that chose."""
    args = [a for a in cli() if a not in ("--licence", "general")]
    args[args.index("part97")] = "unregulated"
    assert onair.main(args) == 2
    assert "requires a stated reason" in capsys.readouterr().out


def test_a_band_with_a_power_limit_refuses_an_undeclared_emission(
        radio_answers, capsys):
    """30 m is capped at 200 W for everyone. Declaring nothing there is asking
    to put an unknown amount of power into a limited band, so `--power` is what
    answers it rather than a default nobody measured."""
    thirty = 10_142_000
    assert onair.main(cli(channel=thirty)) == 2
    assert "does not state its power" in capsys.readouterr().out
    assert onair.main(cli("--power", "100", channel=thirty)) == 0
    assert onair.main(cli("--power", "400", channel=thirty)) == 2
    assert "97.313" in capsys.readouterr().out


# -- the audio devices -------------------------------------------------------

def test_operator_controlled_launch_needs_no_power_declaration(radio_answers, capsys):
    args = cli('--because', 'operator-controlled Sabir RF test', channel=10_126_500)
    args[args.index('part97')] = 'unregulated'
    at = args.index('--licence')
    del args[at:at + 2]
    assert '--power' not in args
    assert onair.main(args) == 0
    assert 'key' in radio_answers[0].events
    output = capsys.readouterr().out
    assert 'NOT KEYING' not in output
    assert 'transmitted ' in output

@pytest.mark.parametrize("flag", ["--audio-out", "--audio-in"])
def test_transmitting_without_a_named_audio_device_is_refused(radio_answers, flag):
    """An unset device is the system default, which is the laptop's own speakers
    and microphone: the beacon goes into the room and the channel sense judges
    the room. Every symptom of that looks like a radio fault."""
    args = cli()
    at = args.index(flag)
    del args[at:at + 2]
    with pytest.raises(SystemExit, match="required when transmitting"):
        onair.main(args)
    assert not radio_answers, "the refusal came before a rig was even built"


def test_sabir_keeps_no_private_device_lister():
    """The pair that lived here could only drift from `core.devices`, and had
    already drifted in the direction that matters: neither could refuse, so the
    resolution beside them could not either."""
    assert not hasattr(radio, "output_devices")
    assert not hasattr(radio, "input_devices")


# -- listening before transmitting -------------------------------------------

def test_an_occupied_channel_stops_the_transmission(radio_answers, monkeypatch,
                                                    capsys):
    monkeypatch.setattr(onair, "listen",
                        lambda *_a, **_k: onair.Sense("busy", 2.1))
    assert onair.main(cli()) == 1
    assert "--force" in capsys.readouterr().out
    assert not radio_answers[0].events


def test_the_operator_may_overrule_an_occupied_channel(radio_answers, monkeypatch):
    """HF is a shared medium of robust signals and the operator's ears outrank
    the detector."""
    monkeypatch.setattr(onair, "listen",
                        lambda *_a, **_k: onair.Sense("busy", 2.1))
    assert onair.main(cli("--force")) == 0
    assert "key" in radio_answers[0].events


@pytest.mark.parametrize("verdict", ["deaf", "intermittent"])
def test_a_receiver_that_cannot_judge_is_not_overridable(radio_answers,
                                                         monkeypatch, verdict):
    """Not a judgement about the band, so there is nothing for `--force` to
    weigh: a station that cannot hear a channel may not call on it."""
    monkeypatch.setattr(onair, "listen",
                        lambda *_a, **_k: onair.Sense(verdict, 0.0))
    assert onair.main(cli("--force")) == 2
    assert not radio_answers[0].events


def test_a_window_with_no_verdict_reads_busy():
    """`core.busy` is biased toward busy and this must not undo it: silence
    scores nothing, and no evidence is not evidence of a clear channel."""
    heard = onair.listen(1.0, None, log=lambda *_: None)
    assert heard.verdict == "deaf" and not heard.overridable


def _receiver(monkeypatch, audio: np.ndarray) -> None:
    """A card that hands `listen` the window it is given, so the verdict under
    test is `core.busy`'s own rather than a `Sense` a monkeypatch wrote."""
    class _sd:
        @staticmethod
        def rec(frames, **kw):
            return audio[:frames].reshape(-1, 1).astype(np.float32)

    monkeypatch.setitem(sys.modules, "sounddevice", _sd)


#: Eight seconds of receiver noise, which is the window every threshold in
#: `core.busy` was calibrated against.
def _noise(seconds: float = 9.0, level: float = 0.05) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(int(seconds * onair.FS)) * level


def test_a_live_window_with_nobody_on_it_reads_clear(monkeypatch):
    """The verdict the other three branches are the exception to, and the only
    one that lets a transmission happen — so it is the one worth reaching
    through the real scorer rather than around it."""
    _receiver(monkeypatch, _noise())
    heard = onair.listen(8.0, None, log=lambda *_: None)
    assert heard.verdict == "clear"
    assert heard.margin_db < 0.0, "a clear channel has headroom, not a margin"


def test_a_carrier_on_the_channel_reads_busy_and_is_overridable(monkeypatch):
    """A steady unmodulated carrier moves neither the burst nor the shape score,
    which is why `core.busy` carries a tone detector at all. It is a judgement
    about the band, so it is the operator's to overrule."""
    t = np.arange(int(9.0 * onair.FS)) / onair.FS
    _receiver(monkeypatch, _noise() + 0.05 * np.sin(2 * np.pi * 1500.0 * t))
    heard = onair.listen(8.0, None, log=lambda *_: None)
    assert heard.verdict == "busy" and heard.overridable
    assert heard.margin_db > 0.0


def test_a_receiver_that_drops_out_mid_window_reads_intermittent(monkeypatch):
    """Something else is keying this radio, and the samples that come back are
    beautifully steady — the most dangerous failure `core.busy` can have, and
    the reason liveness is decided before occupancy. Not overridable: nothing
    has judged the band."""
    audio = _noise()
    audio[int(4.0 * onair.FS):] *= 0.003
    _receiver(monkeypatch, audio)
    said = []
    heard = onair.listen(8.0, None, "KT USB", log=said.append)
    assert heard.verdict == "intermittent" and not heard.overridable
    # The operator's line, not just the verdict: the verdict is a word he never
    # sees, and this is what tells him the rig is being keyed by something else.
    assert "RECEIVER INTERMITTENT" in said[0]
    assert float(said[0].split(" dB")[0].split()[-1]) >= busy.MUTE_DROP_DB


@pytest.fixture(autouse=True)
def _silent_receiver(monkeypatch):
    """`listen` records; nothing in this file may open a card. The default is
    digital silence, which `core.busy` reads as deaf."""
    class _sd:
        @staticmethod
        def rec(frames, **kw):
            return np.zeros((frames, 1), dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sounddevice", _sd)


# -- the transmission itself -------------------------------------------------

def test_a_dry_run_describes_the_burst_and_keys_nothing(capsys):
    assert onair.main(["--mycall", "W9SSJ", "--channel", str(CHANNEL)]) == 0
    out = capsys.readouterr().out
    assert "NOT ARMED" in out
    assert "7.103" in out and "s keyed" in out


def test_the_burst_carries_the_callsign_in_morse():
    """A callsign inside a data frame is an unspecified digital code that reads
    as nothing to anyone not running this modem, so it cannot be what identifies
    the station."""
    call = "W9SSJ"
    burst = onair.presence_burst(call)
    beacon_s = burst.seconds - onair._ID_GAP_S - cwid.duration(call)
    assert beacon_s == pytest.approx(5.5, abs=0.1), "the beacon's own length"
    tail = burst.audio[-int(cwid.duration(call) * onair.FS):]
    keyed = np.abs(tail) > 0.1 * np.abs(tail).max()
    assert 0.3 < keyed.mean() < 0.8, "the tail is not a keyed Morse envelope"


def test_the_described_span_covers_the_identification_too():
    """The Morse sits at 700 Hz of a 1500 Hz-centred passband in its own right;
    a span quoted off the beacon alone would understate what goes out."""
    narrow = onair.wspr_burst("W9SSJ", "EN63", "beacon_deep")
    lo, hi = onair._ID_SPAN_HZ
    said = narrow.describe(DIAL)
    assert f"{lo:.0f} - {hi:.0f} Hz" in said


def test_the_key_goes_up_after_the_audio_device_is_warm(monkeypatch):
    """Everything that stuck a set of finals once is designed out: the device is
    warmed before PTT so the key never opens onto dead air, and the rig's own
    settle sits between the key and the first sample."""
    r = rig()
    order: list[str] = []
    monkeypatch.setattr(radio, "play_drained",
                        lambda audio, fs, device: order.append(
                            "warm" if audio.size < r.settle * radio.FS + 9600
                            else "audio"))
    monkeypatch.setattr(r._keyer, "key",
                        lambda on, note="": order.append("key" if on else "unkey") or True)
    radio.transmit(r, np.ones(int(2.0 * radio.FS)), EMISSION, max_key_s=10.0)
    assert order == ["warm", "key", "audio", "unkey"]


def test_the_key_comes_down_when_the_audio_raises(monkeypatch):
    def boom(audio, fs, device):
        if audio.size > 48000:
            raise RuntimeError("the card went away mid-burst")
    r = rig()
    monkeypatch.setattr(radio, "play_drained", boom)
    with pytest.raises(RuntimeError):
        radio.transmit(r, np.ones(int(2.0 * radio.FS)), EMISSION, max_key_s=10.0)
    assert r._keyer.events == ["key", "unkey"]


def test_a_key_up_the_line_will_not_confirm_plays_no_audio(monkeypatch):
    """`LineKeyer.key` has already deasserted to be sure. Playing into a rig
    that may not be keyed spends the slot proving nothing."""
    r = rig(confirms=False)
    played: list[int] = []
    monkeypatch.setattr(radio, "play_drained",
                        lambda audio, fs, device: played.append(audio.size))
    assert radio.transmit(r, np.ones(int(2.0 * radio.FS)), EMISSION,
                          max_key_s=10.0) == 0.0
    assert len(played) == 1, "the warm-up only; nothing was transmitted"


def test_audio_longer_than_the_ceiling_is_refused_before_the_key(monkeypatch):
    """Refused up front rather than truncated or watchdogged: a clipped
    transmission is a worse artefact than a missing one, and a watchdog that
    fires every time is the ceiling being wrong."""
    r = rig()
    monkeypatch.setattr(radio, "play_drained", lambda audio, fs, device: None)
    with pytest.raises(ValueError, match="key ceiling"):
        radio.transmit(r, np.ones(int(30.0 * radio.FS)), EMISSION, max_key_s=10.0)
    assert not r._keyer.events


def test_the_watchdog_unkeys_and_retires_when_the_play_overruns(monkeypatch):
    """The one bound that does not depend on the play returning."""
    import time
    r = rig()
    calls: list[int] = []

    def stalls(audio, fs, device):
        calls.append(audio.size)
        if len(calls) > 1:                       # past the warm-up
            time.sleep(0.5)
    monkeypatch.setattr(radio, "play_drained", stalls)
    radio.transmit(r, np.ones(int(0.1 * radio.FS)), EMISSION, max_key_s=0.2)
    time.sleep(0.1)
    assert r.retired
    assert r._keyer.events.count("unkey") >= 2


def test_a_retired_rig_refuses_key_up_and_never_key_down():
    """After a signal-time unkey a racing thread must not put the line back up
    in the window before the interpreter exits."""
    r = rig()
    r.retire()
    assert r.key(EMISSION) is False
    assert r.unkey() is True


# -- the line is the ladder's, not sabir's -----------------------------------

def test_sabir_keeps_no_copy_of_the_ladder():
    """Load-bearing safety text and the drop rung live in `core.ptt.Keyer`. Three
    modems each grew a private answer to the same four questions and the same
    stuck-key defect was found and fixed in each separately; sabir is the fourth
    caller and adds no fifth answer."""
    for mod in (radio, onair):
        text = Path(mod.__file__).read_text()
        assert "drop_rts(" not in text, (
            f"{mod.__name__} reaches for the drop rung itself; the ladder owns it")
        for phrase in ("CONFIRMED DOWN", "MAY BE STUCK"):
            assert phrase not in text, (
                f"{mod.__name__} spells {phrase!r} itself")
    assert "arming_refusal" in Path(onair.__file__).read_text()


def test_the_entry_point_starts():
    """`--help` and exit 0: the module imports, its parser is well formed, and
    nothing at import time needs a radio."""
    r = subprocess.run([sys.executable, "-m", "hfmodem.sabir.onair", "--help"],
                       capture_output=True, text=True, timeout=120,
                       cwd=Path(radio.__file__).resolve().parents[3])
    assert r.returncode == 0, r.stderr[-1500:]


def test_the_keying_line_is_opened_by_the_shared_class():
    """`/dev/null` is a character device that drives no modem lines, so arming
    on it reaches `RtsPtt` and fails there — which is the proof that sabir opens
    the line through `core.ptt` and not with an ioctl of its own."""
    with pytest.raises(PttError):
        radio.Rig("ft891", FAKE_DEV, profile=RULES, control=onair.CONTROL,
                   rigctld="127.0.0.1:4532").arm()


def test_a_line_that_will_not_drive_stops_the_run(monkeypatch, capsys):
    """Past every gate, the next thing touched is the keying line itself, and
    what the operator gets there is a sentence rather than a traceback."""
    monkeypatch.setattr(radio.Rig, "identify", lambda self: "Yaesu FT-891")
    monkeypatch.setattr(radio.Rig, "transmitting", lambda self: None)
    monkeypatch.setattr(radio.Rig, "qsy", _qsy())
    monkeypatch.setattr(radio.Rig, "set_mode", lambda self: None)
    monkeypatch.setattr(onair, "listen",
                        lambda *_a, **_k: onair.Sense("clear", -3.0))
    assert onair.main(cli()) == 2
    assert "NOT KEYING" in capsys.readouterr().out
