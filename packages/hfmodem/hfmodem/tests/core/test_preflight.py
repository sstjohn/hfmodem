# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The station check, and what a bench with no radio on it can prove about one.

Provable here: that the verb exists and reaches the right code, that
`transmit = false` refuses before anything can key, that a station with no radio
attached reports the lines it could not measure instead of a clean sheet, that a
tripped gate exits non-zero, and the arithmetic both keying edges are found by.

Not provable here at all: the actuation and recovery figures themselves, the
starvation comparison, and the ADC->DAC offset. Every one of those is a
measurement of a particular card and a particular radio, which is the whole
reason this check exists as something to run rather than something to assert.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hfmodem import cli
from hfmodem.core import config
from hfmodem.core.cat import CatDesync, QueryRefused
from hfmodem.core.rig import ArmReport
from hfmodem.core.rates import CARD_RATE_HZ
from hfmodem.station import preflight
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld
from hfmodem.tests.hermetic import fake_claim_audio_metadata  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_claim_audio_metadata")

CONFIG = """\
schema = 1

[station]
mycall     = "N0CALL"
control    = "local"
regulatory = "unregulated"
because    = "unit test, no radio present"
transmit   = {transmit}

[rig]
model     = "ft891"
centre_hz = 7101500
host      = "127.0.0.1"
port      = {port}

[rig.profiles.ft891]
cat_port = '{cat_port}'

[rig.ptt]
port     = '{ptt}'
settle_s = 0.04

[audio]
input      = "{device}"
output     = "{device}"
input_gain = 0.040
"""

#: A name no sound card can match, so nothing on this machine is ever opened.
NO_DEVICE = "hfmodem preflight test — no such device"


def station_file(tmp_path, *, transmit: bool, device: str = NO_DEVICE, port: int = 4532,
                 cat: str | None = None):
    path = tmp_path / "station.toml"
    path.write_text(CONFIG.format(transmit=str(transmit).lower(), port=port,
                                  ptt=tmp_path / "no-such-tty", cat_port=tmp_path / "no-such-cat",
                                  device=device), encoding="utf-8")
    if cat:
        path.write_text(path.read_text().replace("[rig]\n", f'[rig]\ncat = "{cat}"\n'))
    return path


class FakeLine(FakePtt):
    """A keying line that can be opened, which `FakePtt` has no reason to be.

    Substituted for `RtsPtt` rather than driven through one: Darwin returns
    ENOTTY for TIOCMGET on both ends of a pty, so there is no serial device on
    this bench whose modem lines can be read back at all.
    """

    port = "/dev/fake-keying-line"

    def open(self) -> None:
        return None


class QuietCard:
    """Enough of `StationAudio` for the keying loop: a clock, and a live stream."""

    def __init__(self) -> None:
        self.samples = 0

    def alive(self) -> None:
        return None

    def sample_now(self) -> float:
        self.samples += 128
        return float(self.samples)


class KeyedCard(QuietCard):
    """A card whose capture shows the receiver mute, standing in for its own tap.

    The keying loop reads three times per keying — a drop, the quiet band it takes
    its floor from, then the capture spanning the key — and asks the clock twice,
    for the key-up and the unkey. Both cycles repeat, so every keying sees the
    same band go quiet 10 ms after the line goes up and come back 20 ms after it
    comes down.
    """

    FLOOR = 0.05
    KEY_AT, UNKEY_AT = 4800, 16800
    ACTUATION_MS, RECOVERY_MS = 10.0, 20.0

    def __init__(self) -> None:
        self._clock = itertools.cycle((self.KEY_AT, self.UNKEY_AT))
        self._reads = itertools.cycle((self._nothing, self._band, self._capture))

    def sample_now(self) -> float:
        return float(next(self._clock))

    def poll(self):
        return next(self._reads)()

    def _nothing(self):
        return []

    def _band(self):
        return [(0, np.full(CARD_RATE_HZ // 10, self.FLOOR, np.float32))]

    def _capture(self):
        ms = CARD_RATE_HZ / 1e3
        return [(0, synthetic(mute_at=self.KEY_AT + int(self.ACTUATION_MS * ms),
                              back_at=self.UNKEY_AT + int(self.RECOVERY_MS * ms),
                              floor=self.FLOOR))]


def armed(tmp_path, monkeypatch, line: FakePtt):
    """A `Preflight` whose rig is armed against a fake rigctld and `line`."""
    rigctld = FakeRigctld(ptt_line=line, ptt_type="None")
    monkeypatch.setattr(preflight, "RtsPtt", lambda port: line)
    cfg = config.load(station_file(tmp_path, transmit=True, port=rigctld.port, cat="rigctld"))
    p = preflight.Preflight(cfg)
    return p, p._rig(), rigctld


@pytest.fixture
def cfg(tmp_path):
    return config.load(station_file(tmp_path, transmit=True))


def test_wasapi_rig_spread_reports_nominal_offset_unknown(cfg, monkeypatch, capsys):
    from hfmodem.tests.core.test_audio import _fix23_station

    audio, streams = _fix23_station(monkeypatch)
    audio.open()
    try:
        check = preflight.Preflight(cfg)
        check._offset(audio)
        line, = check.lines
        assert line.name == "ADC->DAC" and line.verdict == preflight.UNKNOWN
        nominal = round(sum(streams[0].latency) * CARD_RATE_HZ)
        assert f"nominal {nominal} samples" in line.detail
        for reason in ("cannot measure", "WASAPI timestamps carry no converter time",
                       "PortAudio's reported buffering", "transmit indices are estimates",
                       "tx_latency_ms is measured on this host"):
            assert reason in line.detail
        assert "ADC->DAC" in capsys.readouterr().out
    finally:
        audio.close()


@pytest.mark.parametrize("lat,verdict", [(None, preflight.STOP), (-352, preflight.STOP),
                                       (128, preflight.OK)])
def test_measured_offset_keeps_its_verdict(cfg, lat, verdict):
    check = preflight.Preflight(cfg)
    check._offset(NS(lat=lat, lat_provenance="measured"))
    line, = check.lines
    assert line.verdict == verdict
    if verdict == preflight.OK:
        assert line.detail == ("128 samples (2.7 ms), median over the settling callbacks "
                               "with the spread asserted")


@pytest.mark.parametrize("host", ["Windows WASAPI", "Core Audio", "ALSA"])
def test_stamp_model_keying_timings_are_estimates_only_on_wasapi(cfg, monkeypatch, host):
    from hfmodem.core import audio as audio_module
    from hfmodem.tests.core.test_audio import _fix23_stamps, _fix23_station

    stamps = list(_fix23_stamps(varying=False))
    if host != "Windows WASAPI":
        # Measured-host control: actual converter anchors and a real 50/60 ms
        # rig defect, rather than WASAPI's 40 ms timestamp bias.
        stamps = [NS(currentTime=t.currentTime,
                     inputBufferAdcTime=100 + i * 128 / CARD_RATE_HZ,
                     outputBufferDacTime=t.outputBufferDacTime)
                  for i, t in enumerate(stamps)]
    card, _ = _fix23_station(monkeypatch, host=host, stamps=stamps)
    card.open()
    try:
        cfg = replace(cfg, rig=replace(cfg.rig, ptt=replace(cfg.rig.ptt, settle_s=.030)))
        check = preflight.Preflight(cfg)
        monkeypatch.setattr(preflight, "TR_RECOVERY_S", .030)
        monkeypatch.setattr(preflight, "TR_RECOVERY_SLACK_S", 0)
        tap = KeyedCard()
        if host != "Windows WASAPI":
            tap.ACTUATION_MS, tap.RECOVERY_MS = 50, 60
        check.tap = tap
        now = {"time": 0.0}
        monkeypatch.setattr(audio_module.time, "monotonic", lambda: now["time"])
        monkeypatch.setattr(preflight.time, "sleep", lambda _: None)
        # The supplied 480/128 model's constant age is 1440 frames. Raw ADC
        # stamps lead true capture by age + packet length = 1920 frames (40 ms).
        n0 = card.samples - card.blocksize
        bias = .040 if host == "Windows WASAPI" else 0
        assert card._last[0] - (100 + n0 / CARD_RATE_HZ) == pytest.approx(bias)
        reads = 0
        read = check._read

        def capture():
            nonlocal reads
            if reads % 3 == 0:
                now["time"] = 100 + tap.KEY_AT / CARD_RATE_HZ
            reads += 1
            return read()

        keys = []

        def key(rig):
            keys.append(card.sample_now())
            now["time"] = 100 + tap.UNKEY_AT / CARD_RATE_HZ

        monkeypatch.setattr(check, "_read", capture)
        monkeypatch.setattr(check, "_key", key)
        rig = NS(retired=False, line_state="low")
        # WASAPI ground truth is 10/20 ms; its raw anchors give 50/60 ms.
        a, r = check._one_keying(card, rig)
        assert (a, r) == pytest.approx((50, 60), abs=.1)
        check.lines.append(preflight.Line("keying", preflight.UNKNOWN, "unproven"))
        check._keying(card, rig)
        assert len(keys) == preflight.KEYINGS + 1
        timings = [ln for ln in check.lines if ln.name in ("PTT actuation", "T/R recovery")]
        assert len(timings) == 2
        if host == "Windows WASAPI":
            assert all(ln.verdict == preflight.UNKNOWN for ln in timings)
            for line, ms in zip(timings, (50, 60)):
                assert f"estimate {ms} ms" in line.detail
                assert "WASAPI timestamps carry no converter time" in line.detail
                assert not line.note
            # The biased timing cannot promote an unproven line to measured OK.
            assert check.lines[0].verdict == preflight.UNKNOWN
            assert not any(ln.verdict == preflight.STOP for ln in check.lines)
        else:
            assert all(ln.verdict == preflight.STOP for ln in timings)
            assert "Raise [rig.ptt] settle_s" in timings[0].note
            assert "Raise TR_RECOVERY_S" in timings[1].note
    finally:
        card.close()


@pytest.mark.parametrize("reported_loss", [0, 128])
def test_wasapi_exact_clock_with_changing_delivery_age_is_unknown(cfg, monkeypatch, reported_loss):
    from hfmodem.tests.core.test_audio import _fix23_stamps, _fix23_station

    # Exact 48 kHz converters; capture packets arrive 2 ms sooner each second.
    ages = [1440 - round(packet * 480 * .002) for packet in range(1000)]
    card, _ = _fix23_station(monkeypatch, stamps=_fix23_stamps(ages=ages, packets=1000))
    card.open()
    try:
        assert card.failure is None and card.lost == 0
        ppm, _ = card.clock_ppm()
        assert 1900 < ppm < 2200
        # Even the raw-stamp loss branch must not turn this fit into a measured
        # converter verdict or offer machine-load advice on this host.
        card.lost = reported_loss
        check = preflight.Preflight(cfg)
        check._clock(card)
        line, = check.lines
        assert line.verdict == preflight.UNKNOWN
        assert f"estimate {ppm:+.1f} ppm" in line.detail
        assert "WASAPI timestamps carry no converter time" in line.detail
        assert not line.note
    finally:
        card.close()


def test_wasapi_missing_edges_and_short_clock_fit_give_the_timestamp_reason(cfg, monkeypatch):
    from hfmodem.tests.core.test_audio import _fix23_station

    card, _ = _fix23_station(monkeypatch, callbacks=1)
    card.open()
    try:
        check = preflight.Preflight(cfg)
        keyings = []

        def no_edges(audio, rig):
            keyings.append(True)
            return float("nan"), float("nan")

        monkeypatch.setattr(check, "_one_keying", no_edges)
        check._keying(card, NS(retired=False, line_state="low"))
        check._clock(card)
        assert len(keyings) == preflight.KEYINGS
        assert [ln.name for ln in check.lines] == ["PTT actuation", "T/R recovery", "card clock"]
        for line in check.lines:
            assert line.verdict == preflight.UNKNOWN
            assert card.timestamp_limitation in line.detail
            assert not line.note
            assert "estimate nan" not in line.detail
    finally:
        card.close()


@pytest.mark.parametrize("provenance,prefix", [("nominal", "nominal "), ("measured", "")])
def test_station_description_labels_nominal_offset(cfg, tmp_path, monkeypatch, provenance, prefix):
    from hfmodem.station.process import Station

    monkeypatch.setenv("HFMODEM_DATA_DIR", str(tmp_path / "data"))
    station = Station(cfg)
    station.audio = NS(lat=1771, lat_provenance=provenance)
    assert f"audio     card, {prefix}offset 1771 samples" in station.describe()


# --- the verb -----------------------------------------------------------------

def test_the_rig_verb_takes_preflight(tmp_path):
    args = cli.build_parser().parse_args(
        ["rig", str(station_file(tmp_path, transmit=True)), "--preflight"])
    assert args.preflight is True
    assert args.fn is cli.cmd_rig


def test_preflight_runs_the_station_check_and_returns_its_answer(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cfg):
        seen["mycall"] = cfg.station.mycall
        return 0

    monkeypatch.setattr(preflight, "run", fake_run)
    rc = cli.main(["rig", str(station_file(tmp_path, transmit=True)), "--preflight"])
    assert rc == 0
    assert seen["mycall"] == "N0CALL"


@pytest.mark.parametrize("cat,backend", [(None, "serial"), ("rigctld", "rigctld")])
@pytest.mark.parametrize("verb", ["--arm", "--preflight"])
def test_the_rig_claim_checks_the_station_cat_backend(tmp_path, monkeypatch, cat, backend, verb):
    from hfmodem.core import claim
    path = station_file(tmp_path, transmit=True, cat=cat)
    seen = []

    def refuse(kind, operation, *, devices):
        seen.append(devices)
        raise claim.ClaimBusy("rig claimed: test")

    monkeypatch.setattr(claim, "acquire", refuse)
    assert cli.main(["rig", str(path), verb]) == 5
    assert [d.cat for d in seen] == [backend]


@pytest.mark.parametrize("error,rc", [("ClaimBusy", 5), ("ClaimError", 1)])
def test_a_refused_claim_is_printed_not_a_traceback(tmp_path, monkeypatch, capsys, error, rc):
    from hfmodem.core import claim

    def refuse(kind, operation, *, devices):
        raise getattr(claim, error)("rig claimed: Holder(pid=4242)")

    monkeypatch.setattr(claim, "acquire", refuse)
    assert cli.main(["rig", str(station_file(tmp_path, transmit=True)), "--arm"]) == rc
    assert capsys.readouterr().err == "hfmodem rig: rig claimed: Holder(pid=4242)\n"


@pytest.fixture
def drivers(monkeypatch):
    """The CAT drivers the station would build, as (driver, endpoint)."""
    from hfmodem.station import process
    built = []

    def driver(name):
        return type(name, (), {"__init__": lambda self, *args, **kw: built.append((name, args[0]))})

    monkeypatch.setattr(process, "YaesuCat", driver("YaesuCat"))
    monkeypatch.setattr(process, "RigctldCat", driver("RigctldCat"))
    return built


@pytest.mark.parametrize("cat,driver", [("yaesu", "YaesuCat"), ("rigctld", "RigctldCat"),
                                        ("civ", "RigctldCat"), ("none", None)])
def test_rig_arm_drives_the_station_cat_backend(tmp_path, monkeypatch, capsys, drivers, cat, driver):
    from hfmodem.core import ptt, rig

    class FakeRig:
        def __init__(self, **kwargs):
            self.cat = kwargs["cat"]

        def arm(self, *, prove_ptt):
            return f"armed through {type(self.cat).__name__}"

        def close(self):
            pass

    monkeypatch.setattr(rig, "Rig", FakeRig)
    monkeypatch.setattr(ptt, "RtsPtt", lambda port: FakeLine())
    path = station_file(tmp_path, transmit=cat != "none", cat=cat)
    rc = cli.main(["rig", str(path), "--arm"])
    if driver is None:
        assert rc == 2 and drivers == []
        return
    endpoint = str(tmp_path / "no-such-cat") if cat == "yaesu" else "127.0.0.1"
    assert (rc, drivers) == (0, [(driver, endpoint)])
    assert f"armed through {driver}" in capsys.readouterr().out


@pytest.mark.parametrize("cat,driver,wire", [("yaesu", "YaesuCat", "CAT on "),
                                             ("rigctld", "RigctldCat", "CAT via rigctld at ")])
def test_preflight_drives_the_station_cat_backend(tmp_path, monkeypatch, drivers, cat, driver, wire):
    def unarmable(**kwargs):
        raise preflight.RigError("not this test's concern")

    monkeypatch.setattr(preflight, "Rig", unarmable)
    p = preflight.Preflight(config.load(station_file(tmp_path, transmit=True, cat=cat)))
    assert p._rig() is None
    assert [name for name, _ in drivers] == [driver]
    assert wire in p.lines[-1].detail


LAUNCHED = """
import sys
from hfmodem import cli
from hfmodem.core import claim
from hfmodem.station import preflight

def fresh_claim(*args, **kwargs):
    raise AssertionError("took a claim of its own under the launcher's")

def ran(cfg):
    print("PREFLIGHT RAN", flush=True)
    return 0

claim.acquire, preflight.run = fresh_claim, ran
sys.exit(cli.main(["rig", sys.argv[1], "--preflight"]))
"""


@pytest.mark.posix_only("onair.sh hands the verb its claim on fd 9")
@pytest.mark.parametrize("cat", ["rigctld", "yaesu"])
def test_rig_under_a_launchers_claim_runs_on_that_claim(tmp_path, monkeypatch, cat):
    """`onair.sh preflight` runs this verb under claim_run: the claim is the
    launcher's, on fd 9, and the verb's own acquire would be refused by it."""
    from hfmodem.core import claim
    (tmp_path / "state").mkdir()
    monkeypatch.setenv("HFMODEM_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HFMODEM_COEXIST_HARNESS", "1")
    station = station_file(tmp_path, transmit=True, cat=cat)
    descriptor = claim.acquire("shell", "preflight")
    try:
        run = subprocess.run([sys.executable, "-c", LAUNCHED, str(station)],
                             env=dict(os.environ, RIG_CLAIMED=str(claim.claim_path())),
                             preexec_fn=lambda: os.dup2(claim._fd(descriptor), 9),
                             close_fds=False, capture_output=True, text=True, timeout=60)
        assert (run.returncode, run.stdout) == (0, "CLAIM ADMITTED\nPREFLIGHT RAN\n"), run.stderr
        assert claim.holder(descriptor).operation == "preflight"
    finally:
        claim.release(descriptor)


@pytest.mark.posix_only("the fake Yaesu is a pty, and Windows has none")
@pytest.mark.parametrize("transmit", [False, True])
@pytest.mark.parametrize("ptt_set", [False, True])
def test_cat_only_rig_opens_no_ptt(tmp_path, monkeypatch, capsys, transmit, ptt_set):
    from hfmodem.core import claim, ptt
    from hfmodem.core.cat import YaesuCat
    from hfmodem.station import process
    from hfmodem.tests.core.fakeyaesu import FakeYaesu, PtySerial

    fake = FakeYaesu()
    monkeypatch.setattr(process, "cat_driver",
                        lambda cfg: YaesuCat(fake.port, serial_factory=PtySerial))
    path = station_file(tmp_path, transmit=transmit, cat="yaesu")
    if not ptt_set:
        path.write_text(path.read_text().replace(f"port     = '{tmp_path / 'no-such-tty'}'", ''))
    devices = []
    monkeypatch.setattr(claim, "acquire", lambda *a, **kw: devices.append(kw["devices"]))
    monkeypatch.setattr(claim, "release", lambda descriptor: None)
    monkeypatch.setattr(claim, "close", lambda descriptor: None)
    monkeypatch.setattr(ptt, "RtsPtt", lambda port: pytest.fail("constructed a PTT line"))
    try:
        assert cli.main(["rig", str(path)]) == 0
        assert len(devices) == 1 and devices[0].ptt_port is None
        assert "PTT proof skipped" in capsys.readouterr().out
        assert fake.log == ['AI0;', 'ID;', 'FA;', 'TX;']
    finally:
        fake.close()


@pytest.mark.parametrize("flag", ["--arm", "--preflight"])
def test_keying_rig_requires_a_ptt_port(tmp_path, capsys, flag):
    path = station_file(tmp_path, transmit=True)
    path.write_text(path.read_text().replace(f"port     = '{tmp_path / 'no-such-tty'}'", ''))
    assert cli.main(["rig", str(path), flag]) == 2
    assert "[rig.ptt] port is unset" in capsys.readouterr().err


def test_arm_is_untouched_by_the_new_flag(tmp_path, capsys):
    """--arm on a transmit = false station still refuses, in its own words."""
    rc = cli.main(["rig", str(station_file(tmp_path, transmit=False)), "--arm"])
    assert rc == 2
    assert "proving the PTT line" in capsys.readouterr().err


# --- nothing keys without transmit = true -------------------------------------

def test_transmit_false_refuses_before_anything_opens(tmp_path, capsys, monkeypatch):
    def never(cfg):
        raise AssertionError("preflight ran on a station that may not transmit")

    monkeypatch.setattr(preflight, "run", never)
    rc = cli.main(["rig", str(station_file(tmp_path, transmit=False)), "--preflight"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "transmit = false" in err and "key the radio" in err


# --- a station with no radio on it --------------------------------------------

def test_no_rig_and_no_card_cannot_measure_and_is_not_a_pass(tmp_path, capsys, monkeypatch):
    def no_card(name, kind):
        raise preflight.AudioError("this test has no card")

    monkeypatch.setattr(preflight, "_device", no_card)
    rc = cli.main(["rig", str(station_file(tmp_path, transmit=True)), "--preflight"])
    out = capsys.readouterr().out
    assert rc != 0
    assert "== preflight OK ==" not in out
    assert "INCOMPLETE" in out
    # Every line, named, said it could not be measured — not a silent pass and
    # not a blanket refusal that says nothing about which measurement is missing.
    for name in ("capture-only", "duplex", "xruns", "ADC->DAC", "keying",
                 "PTT actuation", "T/R recovery", "card clock"):
        assert any(ln.name == name and ln.verdict == preflight.UNKNOWN
                   for ln in _lines(out)), name


def _lines(out: str):
    """The printed report, back as `Line`s, so a test reads what an operator does."""
    got = []
    for raw in out.splitlines():
        if not raw.startswith("  ") or ":" not in raw:
            continue
        name, _, rest = raw.strip().partition(":")
        verdict = rest.strip()
        for word in (preflight.UNKNOWN, preflight.STOP, preflight.OK):
            if verdict.startswith(word):
                got.append(preflight.Line(name.strip(), word, verdict[len(word):]))
                break
    return got


# --- the verdict --------------------------------------------------------------

def test_a_stop_exits_non_zero(cfg, capsys):
    p = preflight.Preflight(cfg)
    p.say("duplex", preflight.OK, "rms 0.00412")
    p.say("xruns", preflight.STOP, "3 underrun(s)")
    assert p.verdict() == 1
    out = capsys.readouterr().out
    assert "== preflight FAILED ==" in out
    assert "stopped on: xruns" in out


def test_a_line_nobody_could_measure_is_not_a_pass(cfg, capsys):
    p = preflight.Preflight(cfg)
    p.say("duplex", preflight.OK, "rms 0.00412")
    p.say("PTT actuation", preflight.UNKNOWN, "nothing was keyed")
    assert p.verdict() == 1
    assert "== preflight OK ==" not in capsys.readouterr().out


def test_every_line_measured_and_ok_is_the_only_pass(cfg, capsys):
    p = preflight.Preflight(cfg)
    for name in ("capture-only", "duplex", "xruns", "ADC->DAC", "keying",
                 "PTT actuation", "T/R recovery", "card clock"):
        p.say(name, preflight.OK, "measured")
    assert p.verdict() == 0
    assert "== preflight OK ==" in capsys.readouterr().out


# --- the gates ----------------------------------------------------------------

def test_a_starved_duplex_input_is_a_stop(cfg, capsys):
    p = preflight.Preflight(cfg)
    p._duplex(np.full(CARD_RATE_HZ, 0.001, np.float32), reference=0.1)
    assert p.lines[-1].verdict == preflight.STOP
    assert "INPUT STARVED" in capsys.readouterr().out


def test_a_duplex_stream_at_the_reference_level_passes(cfg):
    p = preflight.Preflight(cfg)
    p._duplex(np.full(CARD_RATE_HZ, 0.1, np.float32), reference=0.1)
    assert p.lines[-1].verdict == preflight.OK


def test_actuation_past_the_configured_settle_is_a_stop(cfg, capsys):
    p = preflight.Preflight(cfg)
    p._actuation([180.0, 175.0, 182.0])         # settle_s is 0.04 in the file
    assert p.lines[-1].verdict == preflight.STOP
    assert "settle_s to at least 0.18" in capsys.readouterr().out


def test_actuation_inside_the_settle_passes(cfg):
    p = preflight.Preflight(cfg)
    p._actuation([21.0, 19.0, 22.0])
    assert p.lines[-1].verdict == preflight.OK


def test_a_band_too_quiet_to_time_is_not_an_actuation_of_zero(cfg):
    p = preflight.Preflight(cfg)
    p._actuation([float("nan")] * preflight.KEYINGS)
    assert p.lines[-1].verdict == preflight.UNKNOWN


def test_recovery_past_the_turnaround_budget_is_a_stop(cfg):
    p = preflight.Preflight(cfg)
    p._recovery([120.0, 118.0, 125.0])
    assert p.lines[-1].verdict == preflight.STOP


def test_recovery_inside_the_turnaround_budget_passes(cfg):
    p = preflight.Preflight(cfg)
    p._recovery([48.0, 52.0, 50.0])
    assert p.lines[-1].verdict == preflight.OK


# --- keying, against a fake radio ---------------------------------------------

@pytest.fixture(autouse=True)
def _brisk(monkeypatch):
    """The dwells are what a rig needs, not what the arithmetic needs."""
    monkeypatch.setattr(preflight, "HOLD_S", 0.01)
    monkeypatch.setattr(preflight, "FLOOR_S", 0.0)
    monkeypatch.setattr(preflight, "RECOVER_S", 0.0)


def test_the_keying_line_is_named_before_anything_is_timed(
        tmp_path, monkeypatch, capsys):
    """Which wire, on which port, before a single number is measured.

    The verdict is UNKNOWN rather than OK because the daemon this station requires
    has no PTT of its own to report, so its silence about the line is not a
    finding. Calling that a failure would fail a correctly wired station; the
    keying measurement below is what can tell.
    """
    line = FakeLine()
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        assert rig is not None
        assert p.lines[-1].verdict == preflight.UNKNOWN
        out = capsys.readouterr().out
        assert "RTS on" in out and "PTT UNPROVEN" in out
        assert "line ownership verified" in out
    finally:
        rig.close()
        rigctld.close()


def test_every_keying_brings_the_line_back_down(tmp_path, monkeypatch):
    line = FakeLine()
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        p._keying(QuietCard(), rig)
        # The arm gate's own proof, then one per keying, and down after each.
        assert line.calls.count("assert up") == 1 + preflight.KEYINGS
        assert line.calls.count("assert down") == 1 + preflight.KEYINGS
        assert line.line is False
    finally:
        rig.close()
        rigctld.close()


class AfterUnkeys(FakeLine):
    """A line that misbehaves after its `good`th deassert: it reads back `reads`,
    or the driver refuses the deassert."""

    def __init__(self, good: int, *, reads=None, refuse: bool = False) -> None:
        super().__init__()
        self.good, self.reads, self.refuse = good, reads, refuse

    def _bad(self) -> bool:
        return self.calls.count("assert down") > self.good

    def assert_(self, on: bool) -> None:
        super().assert_(on)
        if self.refuse and not on and self._bad():
            raise preflight.PttError("the fake refuses to deassert")

    def sense(self):
        return self.reads if self._bad() and not self.released else super().sense()


class Deadmen:
    """`threading` as preflight sees it, recording every deadman it starts."""

    def __init__(self) -> None:
        self.started = []

    def Timer(self, interval, function, args=()):
        timer = NS(interval=interval, daemon=False, cancel=lambda: None)
        timer.start = lambda: self.started.append(timer)
        return timer


@pytest.mark.parametrize("good", [1, preflight.KEYINGS], ids=["first", "last"])
def test_an_unconfirmed_unkey_stops_the_keying_and_fences_the_rig(
        tmp_path, monkeypatch, capsys, good):
    """I4: after an unkey nobody could read back, nothing keys until one reads low.
    On the last keying there is no next key to refuse, so the loop says so itself."""
    line = AfterUnkeys(good)
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        assert rig.line_state == "low"
        p._keying(QuietCard(), rig)
        # The arm gate's proof, then each preflight keying up to the unread unkey.
        assert line.calls.count("assert up") == 1 + good
        assert rig.line_state == "unconfirmed" and not rig.retired
        stopped = [ln for ln in p.lines if ln.verdict == preflight.STOP]
        assert [ln.name for ln in stopped] == ["PTT actuation", "T/R recovery"]
        assert all(ln.detail == f"after {good} of {preflight.KEYINGS} keyings"
                   for ln in stopped)
        assert all("the keying line is unconfirmed after the preflight unkey" in ln.note
                   for ln in stopped)
        assert p.verdict() == 1
        assert "== preflight FAILED ==" in capsys.readouterr().out

        deadmen = Deadmen()
        monkeypatch.setattr(preflight, "threading", deadmen)
        with pytest.raises(preflight.RigError, match="keying line is unconfirmed"):
            p._key(rig)
        assert deadmen.started == [], "a refused key armed a deadman"
        assert line.calls.count("assert up") == 1 + good
        assert not rig.retired
    finally:
        rig.close()
        rigctld.close()


def test_preflight_keys_only_a_line_that_reads_low(tmp_path, monkeypatch):
    line = FakeLine()
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        rig.line_state = "high"
        with pytest.raises(preflight.RigError, match="keying line is high"):
            p._key(rig)
        assert line.calls.count("assert up") == 1
    finally:
        rig.close()
        rigctld.close()


@pytest.mark.parametrize("fault, why", [
    ({"reads": True}, "the keying line reads high after the preflight deassert"),
    ({"refuse": True}, "the keying line refused to deassert"),
], ids=["reads-high", "refused-deassert"])
def test_an_unkey_that_fails_outright_retires_the_rig(tmp_path, monkeypatch, fault, why):
    line = AfterUnkeys(1, **fault)
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        p._keying(QuietCard(), rig)
        assert rig.retired and why in rig.retired_why
        assert line.calls.count("assert up") == 2
        stopped = [ln for ln in p.lines if ln.verdict == preflight.STOP]
        assert [ln.name for ln in stopped] == ["PTT actuation", "T/R recovery"]
        assert all(ln.note.startswith("the rig retired: ") and why in ln.note
                   for ln in stopped)
        assert p.verdict() == 1
    finally:
        rig.close()
        rigctld.close()


def test_a_quiet_band_reports_nothing_measured_rather_than_a_pass(
        tmp_path, monkeypatch):
    line = FakeLine()
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        p._keying(QuietCard(), rig)
        assert [ln.verdict for ln in p.lines[-2:]] == [preflight.UNKNOWN] * 2
        assert p.verdict() == 1
    finally:
        rig.close()
        rigctld.close()


def test_a_measured_actuation_settles_the_keying_line_cat_could_not_read(
        tmp_path, monkeypatch, capsys):
    """The receiver muting on every keying is the line reaching the radio.

    CAT cannot say so on the daemon this station requires, so `keying` goes by as
    UNKNOWN before anything is timed. The measurement that follows is the evidence
    that line was waiting for, and a station that keys must not read as one that
    could not be measured.
    """
    line = FakeLine()
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        assert p.lines[-1] == preflight.Line("keying", preflight.UNKNOWN,
                                             p.lines[-1].detail, p.lines[-1].note)
        card = KeyedCard()
        p.tap = card
        p._keying(card, rig)
        keying = [ln for ln in p.lines if ln.name == "keying"]
        assert len(keying) == 1, "the report must not carry two keying verdicts"
        assert keying[0].verdict == preflight.OK
        assert f"{preflight.KEYINGS} of {preflight.KEYINGS}" in keying[0].detail
        assert "RTS on" in keying[0].detail
        assert p.verdict() == 0
        assert "== preflight OK ==" in capsys.readouterr().out
    finally:
        rig.close()
        rigctld.close()


def test_an_unmeasurable_keying_leaves_the_line_unproven(
        tmp_path, monkeypatch, capsys):
    """The direction that matters: nothing muted, so nothing is claimed."""
    line = FakeLine()
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        p._keying(QuietCard(), rig)
        keying = [ln for ln in p.lines if ln.name == "keying"]
        assert [ln.verdict for ln in keying] == [preflight.UNKNOWN]
        assert p.verdict() == 1
        assert "INCOMPLETE" in capsys.readouterr().out
    finally:
        rig.close()
        rigctld.close()


def test_a_line_that_will_not_come_down_never_reaches_the_measurement(
        tmp_path, monkeypatch):
    """The line sticks up, so the arm gate's own proof retires the rig."""
    line = FakeLine(stick=True)
    p, rig, rigctld = armed(tmp_path, monkeypatch, line)
    try:
        assert rig is None      # the arm gate's own proof never came down
        assert p.lines[-1].verdict == preflight.UNKNOWN
        p._keying(QuietCard(), None)
        assert p.verdict() == 1
    finally:
        if rig is not None:
            rig.close()
        rigctld.close()


def test_a_fault_at_rig_construction_is_a_finding_not_a_crash(cfg, monkeypatch):
    """The three constructors are lazy today, and nothing but this test says so.
    If one of them starts touching hardware, the fault must land as a 'cannot
    measure' line — reporting on absent hardware without crashing is the job."""
    def eager(port):
        raise preflight.PttError(f"cannot open PTT port {port}: no such tty")

    monkeypatch.setattr(preflight, "RtsPtt", eager)
    p = preflight.Preflight(cfg)
    assert p._rig() is None
    assert p.lines[-1].name == "keying"
    assert p.lines[-1].verdict == preflight.UNKNOWN
    assert "no such tty" in p.lines[-1].detail
    assert p.verdict() == 1


def test_a_rig_that_could_not_be_armed_is_not_left_under_the_signal_guard(
        tmp_path, monkeypatch):
    """`unkey_on_signal` reads `self.rig` when the signal arrives, and what it finds
    here is a rig this run already closed: `ptt.open()` never succeeded, so
    `released_low` is None and a Ctrl-C during the card clock spends the whole CAT
    ladder and prints UNKEY NOT CONFIRMED for a station that never keyed."""
    line = FakeLine()
    rigctld = FakeRigctld(ptt_line=line, ptt_type="None", refuse=True)
    monkeypatch.setattr(preflight, "RtsPtt", lambda port: line)
    p = preflight.Preflight(config.load(
        station_file(tmp_path, transmit=True, port=rigctld.port, cat="rigctld")))
    try:
        assert p._rig() is None
        assert p.rig is None, "the guard was left holding a closed rig"
    finally:
        rigctld.close()


# --- the arithmetic both edges come out of ------------------------------------

def synthetic(*, mute_at: int, back_at: int, floor: float = 0.05) -> np.ndarray:
    """A capture that goes quiet where a rig keys and comes back where it stops."""
    seg = np.full(CARD_RATE_HZ, floor, np.float32)
    seg[mute_at:back_at] = floor * 1e-3
    return seg


def test_the_edges_are_where_the_receiver_mutes_and_comes_back():
    keyed_at, unkeyed_at = 480, 12480         # 10 ms and 260 ms into the window
    seg = synthetic(mute_at=1440, back_at=13440)
    actuation, recovery = _edges(seg, keyed_at, unkeyed_at, floor=0.05)
    assert actuation == pytest.approx(20.0, abs=0.1)     # 960 samples after key-up
    assert recovery == pytest.approx(20.0, abs=0.1)      # 960 after the unkey


def test_a_silent_band_measures_nothing_rather_than_zero():
    seg = np.zeros(CARD_RATE_HZ, np.float32)
    actuation, recovery = _edges(seg, 0, 100, floor=0.0)
    assert np.isnan(actuation) and np.isnan(recovery)


def test_a_receiver_that_never_comes_back_is_nan_not_the_end_of_the_window():
    seg = synthetic(mute_at=480, back_at=CARD_RATE_HZ)
    _, recovery = _edges(seg, 480, 24000, floor=0.05)
    assert np.isnan(recovery)


def _edges(seg, keyed_at, unkeyed_at, *, floor):
    return preflight._edges(seg, 0, keyed_at, unkeyed_at, floor)


# --- the clock line, and what a large ppm on this station actually means --------

class _Clock:
    """Only the three things `_clock` reads off a card."""

    blocksize = 128

    def __init__(self, ppm: float, *, lost: int, samples: int = 90 * CARD_RATE_HZ):
        self._ppm, self.lost, self.samples = ppm, lost, samples

    def clock_ppm(self):
        return self._ppm, 0.5


def test_a_spliced_capture_is_not_reported_as_a_wrong_sample_rate(cfg):
    """The card measures +3.8 to +6.7 ppm on this station and read −867251 under
    eight pure-Python threads: 87% of a 90 s stream never reached Python, with
    `input_overflow` false on every callback that did run. Every reading past
    ±20 ppm in the whole record is negative, which a crystal has no reason to be.
    So the refusal may not name the crystal — the loss counter says what it is."""
    p = preflight.Preflight(cfg)
    p._clock(_Clock(-867_251.0, lost=87 * 90 * CARD_RATE_HZ // 100))
    line = p.lines[-1]
    said = f"{line.detail} {line.note}"
    assert line.verdict == preflight.STOP
    assert "never reached" in said, said
    assert str(CARD_RATE_HZ) not in said, (
        f"the operator is told the card runs at the wrong rate: {said}")


def test_a_card_at_the_wrong_rate_still_says_so(cfg):
    """With nothing lost, a large ppm is the rate itself — a device delivering
    44100 where 48000 was asked for reads about −81000."""
    p = preflight.Preflight(cfg)
    p._clock(_Clock(-81_000.0, lost=0))
    line = p.lines[-1]
    assert line.verdict == preflight.STOP
    assert str(CARD_RATE_HZ) in f"{line.detail} {line.note}"


def test_a_healthy_crystal_passes(cfg):
    p = preflight.Preflight(cfg)
    p._clock(_Clock(7.4, lost=0))
    assert p.lines[-1].verdict == preflight.OK


@pytest.mark.parametrize('host', ['Windows WASAPI', 'Core Audio', 'ALSA'])
def test_standalone_preflight_delivery_age_timing(monkeypatch, capsys, host):
    """Healthy 10/20 ms edges appear as 40/60 ms from biased key anchors."""
    from hfmodem.core.audio import timestamp_limitation
    from hfmodem.shrike import onair

    class Live:
        fs = onair.FS
        lost = xruns = underruns = estimated_lost = 0
        _lat, _blk = 1771, 128
        lat_provenance = 'nominal' if host == 'Windows WASAPI' else 'measured'
        def __init__(self, *args, **kw):
            self.timestamp_limitation = timestamp_limitation(host)
            self.pos = self.samples = self.reads = self.keys = 0
        def read_ready(self):
            self.reads += 1
            if self.reads >= 3 and self.reads % 2:
                # True key at 100 ms, unkey at 350 ms; mute at 110,
                # recovery at 370. Raw key anchors say 70 and 310 ms.
                data = np.ones(round(.8*self.fs), np.float32)*.1
                data[round(.11*self.fs):round(.37*self.fs)] = 0
            else:
                data = np.ones(4800, np.float32)*.1
            self.pos += len(data)
            self.samples = self.pos
            return data
        def sample_now(self):
            self.keys += 1
            return self.pos + round((.07 if self.keys % 2 else .31)*self.fs)
        def close(self):
            pass
    calls = []
    rig = NS(ptt_port='mock-ptt', serial='mock-cat',
             ptt=lambda keyed: calls.append(keyed), stop=lambda: None)
    monkeypatch.setattr(onair, '_LiveInput', Live)
    monkeypatch.setattr(onair, 'TR_SWITCH_S', .03)
    monkeypatch.setattr(onair.time, 'sleep', lambda _: None)
    result = onair._preflight('in', 'out', rig, .03, key=True)
    out = capsys.readouterr().out
    assert calls == [True, False]*5
    assert '40 ms' in out and '60 ms' in out
    if host == 'Windows WASAPI':
        assert result == 0
        assert out.count('UNKNOWN') >= 2 and 'estimate' in out
        assert 'Raise' not in out and 'raise TR_SWITCH_S' not in out
        assert 'FAILED' not in out
    else:
        assert result == 1 and 'FAILED' in out
        assert "Raise this rig's settle" in out and 'raise TR_SWITCH_S' in out


@pytest.mark.parametrize("windows,enabled,warned", [
    (True, False, True), (True, True, False), (False, False, False)])
def test_long_paths_off_is_one_warning_and_preflight_continues(cfg, monkeypatch, capsys,
                                                               windows, enabled, warned):
    reads, measured = [], []
    monkeypatch.setattr(preflight, "_WINDOWS", windows)
    monkeypatch.setattr(preflight, "_long_paths_enabled", lambda: reads.append(1) or enabled)
    p = preflight.Preflight(cfg)
    monkeypatch.setattr(p, "_reference", lambda: None)
    monkeypatch.setattr(p, "_card", lambda reference: None)
    monkeypatch.setattr(p, "_rig", lambda: None)
    monkeypatch.setattr(p, "_keying", lambda audio, rig: measured.append("keying"))
    monkeypatch.setattr(p, "_clock", lambda audio: measured.append("clock"))
    monkeypatch.setattr(p, "verdict", lambda: 0)
    assert p.run() == 0
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if "LongPathsEnabled" in line] == (
        ["  !! Windows long paths are off (LongPathsEnabled=0): other programs may not open "
         "recordings deeper than 260 characters"] if warned else [])
    assert measured == ["keying", "clock"]
    assert reads == ([1] if windows else [])


class FlakyCatRig:
    """Arms after the CAT faults scripted for it, one per construction."""

    faults: list = []
    built = 0

    def __init__(self, **kwargs):
        type(self).built += 1
        self.ptt = FakeLine()
        self.cat = type("Cat", (), {"cache_free": True, "reads_ptt": True})()
        self.closed = False

    def arm(self, *, prove_ptt):
        if self.faults:
            raise self.faults.pop(0)
        return ArmReport("FT-891", 7_100_000, True, True, ())

    def close(self):
        self.closed = True


@pytest.mark.parametrize("fault", [QueryRefused("radio refused CAT query"),
                                   CatDesync("unsolicited or ambiguous CAT reply '?;'")])
def test_a_transient_cat_read_does_not_skip_the_keying_proof(cfg, monkeypatch, capsys, fault):
    monkeypatch.setattr(preflight, "Rig", FlakyCatRig)
    monkeypatch.setattr(preflight, "CAT_RETRY_S", 0)
    monkeypatch.setattr(FlakyCatRig, "faults", [fault, fault])
    monkeypatch.setattr(FlakyCatRig, "built", 0)
    p = preflight.Preflight(cfg)
    rig = p._rig()
    assert rig is p.rig and FlakyCatRig.built == 3
    assert p.lines[-1].verdict == preflight.OK and "PTT proven" in p.lines[-1].detail
    assert capsys.readouterr().out.count("retrying on a fresh open") == 2


def test_a_cat_read_that_stays_ambiguous_still_fails(cfg, monkeypatch):
    monkeypatch.setattr(preflight, "Rig", FlakyCatRig)
    monkeypatch.setattr(preflight, "CAT_RETRY_S", 0)
    monkeypatch.setattr(FlakyCatRig, "faults", [CatDesync("unsolicited")] * 3)
    monkeypatch.setattr(FlakyCatRig, "built", 0)
    p = preflight.Preflight(cfg)
    assert p._rig() is None and p.rig is None
    assert FlakyCatRig.built == preflight.CAT_READ_TRIES
    assert p.lines[-1].verdict == preflight.UNKNOWN and "unsolicited" in p.lines[-1].detail


def test_a_cat_timeout_is_not_retried(cfg, monkeypatch):
    monkeypatch.setattr(preflight, "Rig", FlakyCatRig)
    monkeypatch.setattr(FlakyCatRig, "faults", [preflight.RigError("CAT reply timed out")])
    monkeypatch.setattr(FlakyCatRig, "built", 0)
    p = preflight.Preflight(cfg)
    assert p._rig() is None and FlakyCatRig.built == 1
    assert "timed out" in p.lines[-1].detail
