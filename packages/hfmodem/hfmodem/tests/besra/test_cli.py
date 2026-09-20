# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The host-server CLI entry point, exercised end to end.

`run_server.main()`'s mode dispatch (loopback / besra / radio) is wiring no other
test runs. These boot the actual `besra-modem` process in each hardware-free mode
and confirm it answers, and unit-test the radio wiring with the sound card stubbed
out."""

from __future__ import annotations

import re
import socket
import subprocess
import sys

import pytest


def _boot(*extra: str):
    # The child binds port 0 and its startup banner names the ports it got —
    # so there is no pick-then-rebind window for another run to land in.
    proc = subprocess.Popen(
        [sys.executable, "-m", "hfmodem.besra.host.run_server",
         "--control-port", "0", *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        banner = proc.stdout.readline()          # EOF if the server dies first
        m = re.search(r"control :(\d+)\s+data :(\d+)", banner)
        if not m:
            proc.wait(timeout=10)
            raise AssertionError(
                f"no port banner, got {banner!r}: {proc.stderr.read()}")
        cport = int(m.group(1))
        c = socket.create_connection(("127.0.0.1", cport), timeout=5)
        c.sendall(b"VERSION\r")
        c.settimeout(5)
        reply = c.recv(200).decode()
        c.close()
        return reply
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_help_exits_clean():
    r = subprocess.run([sys.executable, "-m", "hfmodem.besra.host.run_server", "--help"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "--radio" in r.stdout


def test_loopback_mode_boots_and_answers():
    assert _boot().startswith("VERSION besra-")


@pytest.mark.slow
def test_besra_mode_boots_and_answers():
    assert _boot("--modem", "besra").startswith("VERSION besra-")


def _mail_argv(*extra: str) -> list[str]:
    return ["--modem", "besra", "--call", "KE8LVA", "--mycall", "W9SSJ", *extra]


def _refused(monkeypatch, argv: list[str], capsys) -> str:
    """Run `main()` on ``argv`` and return the usage error it exits 2 with."""
    from hfmodem.besra.host import run_server

    monkeypatch.setattr(sys, "argv", ["besra-modem", *argv])
    with pytest.raises(SystemExit) as exc:
        run_server.main()
    assert exc.value.code == 2
    return capsys.readouterr().err


@pytest.mark.parametrize("extra,wanted", [
    (["--mail-send", "/nonexistent/body.txt"], "--mail-send"),
    (["--mail-send", "@BODY@"], "--mail-to"),
])
def test_a_mail_run_is_refused_before_anything_is_built(monkeypatch, capsys,
                                                        tmp_path, extra, wanted):
    """Every outbound message is loaded while a bad path or a body file with
    nowhere to go still costs a usage line. Later — after the modem is up and, on
    the `--radio` path, after the rig is tuned — the same mistake came back as a
    traceback with the transmitter already armed. kestrel's identical guard has
    three tests; this one had none, and an `ap.error` that is never exercised is
    a refusal nobody has seen refuse.
    """
    body = tmp_path / "body.txt"
    body.write_text("mail for the far end\n")
    argv = _mail_argv(*[str(body) if a == "@BODY@" else a for a in extra])

    err = _refused(monkeypatch, argv, capsys)

    assert wanted in err and "Traceback" not in err


def test_a_call_with_nothing_to_send_is_refused(monkeypatch, capsys):
    """`--call` alone connects a gateway and has nothing to say to it."""
    err = _refused(monkeypatch, _mail_argv(), capsys)
    assert "nothing to do" in err


def _radio_args(**over):
    """The parser's own defaults, with what a test cares about overridden.

    Hand-built namespaces here drifted from the parser and broke the moment a flag
    was added — an argument these tests never make and cannot see. Taking the
    defaults from the parser makes them the same object the CLI passes.
    """
    from hfmodem.besra.host import run_server

    args = run_server.parser().parse_args([])
    for k, v in over.items():
        setattr(args, k, v)
    return args


def test_radio_wiring_builds_without_hardware(monkeypatch):
    # Exercise run_server._radio_modem's arg dispatch with the sound card stubbed.
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)

    args = _radio_args(radio="ft891", quiet=True, record=None)
    modem, link = run_server._radio_modem(args)
    try:
        assert isinstance(link, radio.RadioLink)
        assert modem.audio_out is not None      # RadioLink bound the TX seam
    finally:
        modem.stop()


class _FakeRig:
    """Records what run_server does to the rig, without spawning rigctl."""
    reported = "Yaesu FT-891"
    #: What `qsy` answers: "echo" reads back the dial that was asked for, an int
    #: is a rig that stayed where it was, None is a rig that will not say.
    reports_freq = "echo"

    def __init__(self):
        self.calls = []

    @classmethod
    def named(cls, name, serial, rigctl="rigctl", rigctld=None,
              ptt_device=None, line_ptt=False):
        r = cls(); r.calls.append(("named", name, serial, rigctl, rigctld))
        r.line_ptt = line_ptt
        return r

    def identify(self):
        self.calls.append("identify"); return self.reported

    def set_mode(self, m): self.calls.append(("mode", m))

    def qsy(self, f):
        self.calls.append(("freq", f))
        return f if self.reports_freq == "echo" else self.reports_freq

    def unkey(self, why=""): self.calls.append(("unkey", why))
    def retire(self): self.calls.append("retire")


def test_radio_wiring_drives_the_rig(monkeypatch, capsys):
    """The --radio path (station.sh call): identify before arming, set mode + dial,
    bind the rig to RadioLink, and install a SIGTERM/SIGHUP unkey — the ordinary
    ways an unattended run is stopped, which no `finally` survives. Never runs
    with real hardware, so stub the rig and the sound card."""
    import signal
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    monkeypatch.setattr(radio, "Rig", _FakeRig)
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    handlers = {}
    monkeypatch.setattr(signal, "signal", lambda s, h: handlers.__setitem__(s, h))

    args = _radio_args(radio="ft891", serial="/dev/cu.x", rigctl="/opt/rigctl",
                       audio_in="USB Audio Device", audio_out="USB Audio Device",
                       dial=7103500, quiet=True, record=None)
    modem, link = run_server._radio_modem(args)
    try:
        rig = link.rig
        assert ("named", "ft891", "/dev/cu.x", "/opt/rigctl", None) in rig.calls
        assert "identify" in rig.calls              # positive ID before arming
        assert ("mode", "PKTUSB") in rig.calls
        assert ("freq", 7103500) in rig.calls       # --dial set verbatim
        assert "WARNING" not in capsys.readouterr().out   # FT-891 matches, no mismatch warn

        for sig in (signal.SIGTERM, signal.SIGHUP):   # each retires, unkeys, exits
            with pytest.raises(SystemExit):
                handlers[sig](sig, None)
            assert ("unkey", signal.Signals(sig).name) in rig.calls
        # Retire precedes the unkey: a racing transmit thread must not re-key
        # in the window between the handler and interpreter exit.
        assert rig.calls.index("retire") < rig.calls.index(("unkey", "SIGTERM"))
    finally:
        modem.stop()


def test_radio_warns_on_model_mismatch(monkeypatch, capsys):
    """Keying the wrong radio is the expensive mistake: a rig reporting a different
    model than --radio must warn."""
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    monkeypatch.setattr(_FakeRig, "reported", "Xiegu G90")   # not the ft891 we asked for
    monkeypatch.setattr(radio, "Rig", _FakeRig)
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    monkeypatch.setattr("signal.signal", lambda s, h: None)

    args = _radio_args(radio="ft891", serial="/dev/cu.x", dial=7103500,
                       quiet=True, record=None)
    modem, link = run_server._radio_modem(args)
    try:
        assert "WARNING" in capsys.readouterr().out
    finally:
        modem.stop()


def _rigged(monkeypatch):
    from hfmodem.besra import radio

    monkeypatch.setattr(radio, "Rig", _FakeRig)
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    monkeypatch.setattr("signal.signal", lambda s, h: None)


def test_a_qsy_the_rig_did_not_take_is_never_keyed_over(monkeypatch):
    """PACTOR's refusal, now besra's. On 2026-08-13 four ARDOP attempts went out
    on whatever dial the previous run left — across the PACTOR channels — while
    the log printed the dial besra *wanted*. The operator caught it by watching
    the rig; nothing in any log would have. So the dial is read back after the
    set, and a rig that reports a different frequency is not keyed, in the words
    shrike's QSY gate uses."""
    from hfmodem.besra.host import run_server

    _rigged(monkeypatch)
    monkeypatch.setattr(_FakeRig, "reports_freq", 14000000)   # the set did not take
    args = _radio_args(radio="ft891", rigctld="localhost:4532", dial=7100000,
                       quiet=True, record=None)
    with pytest.raises(SystemExit,
                       match=r"QSY FAILED: asked 7100000.*14000000.*not keying"):
        run_server._radio_modem(args)


def test_a_dial_that_cannot_be_read_back_is_not_keyed_either(monkeypatch):
    """An unreadable rig is not a verified dial. The 2026-08-13 log line —
    "model via rigctld not readable (daemon owns the rig — ok)" — is exactly the
    situation this closes: unreadable was treated as fine, and besra keyed."""
    from hfmodem.besra.host import run_server

    _rigged(monkeypatch)
    monkeypatch.setattr(_FakeRig, "reports_freq", None)       # rig will not say
    args = _radio_args(radio="ft891", rigctld="localhost:4532", channel=7103500,
                       quiet=True, record=None)
    with pytest.raises(SystemExit, match=r"QSY FAILED.*not keying"):
        run_server._radio_modem(args)


def test_a_rig_with_no_dial_named_is_not_keyed_either(monkeypatch):
    """The same defect with the gate taken away: no dial asked for is no dial to
    verify, and the modem would key on whatever the rig was last left on — which
    is precisely what happened on 2026-08-13. Both ARDOP verbs in
    `tools/onair.sh` pass --channel, so nothing that works today loses."""
    from hfmodem.besra.host import run_server

    _rigged(monkeypatch)
    args = _radio_args(radio="ft891", rigctld="localhost:4532",
                       quiet=True, record=None)
    with pytest.raises(SystemExit, match=r"NOT KEYING.*no --channel or --dial"):
        run_server._radio_modem(args)


def test_a_verified_dial_proceeds_and_mode_follows_frequency(monkeypatch):
    """The healthy path through the gate — and mode set AFTER frequency, because
    the FT-891 keeps a mode per band, so a band jump can restore whatever the
    new band was last left in (shrike learned this first)."""
    from hfmodem.besra.host import run_server

    _rigged(monkeypatch)
    args = _radio_args(radio="ft891", rigctld="localhost:4532", dial=7100000,
                       quiet=True, record=None)
    modem, link = run_server._radio_modem(args)
    try:
        calls = link.rig.calls
        assert ("freq", 7100000) in calls
        assert calls.index(("freq", 7100000)) < calls.index(("mode", "PKTUSB"))
    finally:
        modem.stop()


def test_line_ptt_reaches_the_rig(monkeypatch):
    """--line-ptt is the flag that decides whether a burst's key travels the link
    RF disrupts. A flag the parser accepts and nothing acts on is worse than no
    flag: the launcher passes it, the log says nothing, and every key goes back
    through the daemon."""
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    monkeypatch.setattr(radio, "Rig", _FakeRig)
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    monkeypatch.setattr("signal.signal", lambda s, h: None)

    args = _radio_args(radio="ft891", rigctld="localhost:4532", dial=7100000,
                       ptt_device="/dev/whatever", line_ptt=True,
                       quiet=True, record=None)
    modem, link = run_server._radio_modem(args)
    try:
        assert link.rig.line_ptt is True
    finally:
        modem.stop()


def test_line_ptt_without_a_device_is_refused(monkeypatch, capsys):
    """It names no line to key, and the fallback would be the daemon."""
    # On the message, not on "--ptt-device": argparse prints the whole usage line
    # on any error, so every flag it defines appears in `err` whatever went wrong.
    err = _refused(monkeypatch, _mail_argv("--line-ptt"), capsys)
    assert "needs --ptt-device" in err, err


def test_radio_wiring_through_rigctld(monkeypatch):
    """The mandated path: --rigctld with no --serial still builds the rig, routed
    through the shared daemon."""
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    monkeypatch.setattr(radio, "Rig", _FakeRig)
    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    monkeypatch.setattr("signal.signal", lambda s, h: None)

    args = _radio_args(radio="ft891", rigctld="localhost:4532", dial=7099000,
                       quiet=True, record=None)
    modem, link = run_server._radio_modem(args)
    try:
        assert link.rig is not None                 # daemon address alone arms the rig
        assert ("named", "ft891", None, "rigctl", "localhost:4532") in link.rig.calls
        assert ("freq", 7099000) in link.rig.calls
    finally:
        modem.stop()


def test_the_default_is_to_record():
    """A --radio run that nobody asked to record still records. On 2026-07-30 this
    path logged two frame detections on a live channel and kept no audio, so neither
    could be shown to be a signal rather than the noise floor. Recording is what
    makes an on-air claim checkable, so it defaults on in the parser rather than
    depending on the operator remembering a flag."""
    from hfmodem.besra.host import run_server

    assert run_server.RECORD_DIR.parts[-2:] == ("logs", "onair")
    on_air = ["--radio", "ft891", "--rigctld", "127.0.0.1:4532", "--channel", "7103500"]
    assert run_server.parser().parse_args(on_air).record == run_server.RECORD_DIR
    assert run_server.parser().parse_args(on_air + ["--no-record"]).record is None

    text = subprocess.run(
        [sys.executable, "-m", "hfmodem.besra.host.run_server", "--help"],
        capture_output=True, text=True, timeout=30).stdout
    assert "--record" in text and "--no-record" in text


def test_radio_wiring_records_receive_audio(monkeypatch, tmp_path, capsys):
    """The wiring, not just the flag: `_radio_modem` hands RadioLink a recorder
    whose path names the dial it was listening on, and says so on stdout."""
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    monkeypatch.setattr(radio.RadioLink, "start", lambda self: None)
    args = _radio_args(radio="ft891", audio_in="USB Audio", audio_out="USB Audio",
                       channel=7103500, quiet=False, record=tmp_path)
    modem, link = run_server._radio_modem(args)
    try:
        assert link.recorder is not None, "an on-air link with no recorder"
        # --channel 7103500 tunes the dial 1500 Hz low, and the recording is named
        # for the dial so it can be lined up against a log without guessing.
        assert link.recorder.path.name.endswith("-besra-7102000.wav")
        assert link.recorder.path.parent == tmp_path
        assert str(link.recorder.path) in capsys.readouterr().out
    finally:
        link.close()
        modem.stop()
    assert link.recorder.path.exists() and link.recorder.path.with_suffix(".json").exists()
