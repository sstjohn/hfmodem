# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Mail over besra's modulated air: a B2F exchange between two BesraModems.

Every byte both ways is rendered to 12 kHz audio, flown across `StepAir`, and
demodulated — the full stack under the mail layer, with the turn reversals
ARDOP's own BREAK law drives. What this cannot prove is a real ardopcf's
opinion or the RF; it proves everything nearer than those.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.station.mail import ArdopLoopback, ArdopMail
from hfmodem.winlink import B2FSession, MailClient, MailExchange, compose


def test_a_message_crosses_the_air_byte_exact_both_ways():
    out = compose("W9SSJ", "SMTP:op@example.net", "ardop loopback proof",
                  b"a message carried over besra's modulated air.\r\n")
    back = compose("K7ABC", "W9SSJ", "return mail",
                   b"and one the other way.\r\n")
    session = B2FSession("W9SSJ", role="calling", target="K7ABC",
                         outbox=[out])
    t = ArdopLoopback("W9SSJ", "K7ABC", rms_outbox=[back])
    report = MailExchange(session, t, max_steps=300).run()

    assert session.done and not session.failure, report
    assert t.rms.done and not t.rms.failure
    assert [m.render() for m in t.rms.inbox] == [out.render()]
    assert [m.render() for m in session.inbox] == [back.render()]
    assert "closed" in report
    # And the link came down afterwards, whatever the session thought.
    assert not t.modem.connected and not t.far.connected


def test_only_arq_payload_feeds_the_session():
    """The kind filter is a wire that must not be simplified away: an ERR
    block is a decode the link itself rejected, and feeding it to the session
    would fail a transfer the link never broke. IDF and FEC are not session
    bytes either."""
    fed = []
    obs = ArdopMail(MailClient.__new__(MailClient))
    obs.client.session = None
    obs.client.on_link_data = fed.append          # observe routing only
    obs.modem_data_received("ERR", b"\x01garbled")
    obs.modem_data_received("FEC", b"broadcast")
    obs.modem_data_received("IDF", b"ID W9SSJ")
    assert fed == []
    obs.modem_data_received("ARQ", b"[SID]\r")
    assert fed == [b"[SID]\r"]


def test_cli_mail_verb_runs_the_ardop_rehearsal(tmp_path, capsys):
    from hfmodem import cli
    rc = cli.main(["mail", "--gateway", "K7ABC", "--protocol", "ardop",
                   "--mycall", "W9SSJ", "--fetch",
                   "--out", str(tmp_path / "in")])
    outtext = capsys.readouterr().out
    assert rc == 0, outtext
    assert "stage closed" in outtext
    assert len(list((tmp_path / "in").glob("*.b2f"))) == 1
    assert "ardop-mail K7ABC" in outtext, "the on-air command is spelled out"
    # And the launcher answers to the verb the printed line names.
    from pathlib import Path
    launcher = Path(__file__).resolve().parents[5] / "tools" / "onair.sh"
    if launcher.exists():
        assert "ardop-mail)" in launcher.read_text()


def test_the_printed_command_carries_the_channel_it_rehearsed(tmp_path, capsys):
    """The rig slot flies the printed line verbatim, so the line must name the
    channel the rehearsal used — centre AND bandwidth.

    KE8LVA answered at BW2000 and nothing below it has ever been answered; a
    line that silently said 500 would fly a mode the gateway was never heard
    in, and the difference does not show up until the rig is keyed.
    """
    from hfmodem import cli
    rc = cli.main(["mail", "--gateway", "KE8LVA", "--protocol", "ardop",
                   "--mycall", "W9SSJ", "--fetch", "--freq", "7103500",
                   "--bandwidth", "2000", "--out", str(tmp_path / "in")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "ardop-mail KE8LVA 7103500 2000" in out
    assert "--mail-send" not in out, "a fetch composes nothing in anyone's name"


# --------------------------------------------------------------------------- #
# The on-air mail mode: `run_server --call` runs the exchange over the live
# link's own observer seam. The far end here is a real answering session wired
# where the radio would be, so the whole of `_run_mail` — connect, the observer
# routing, teardown, the verdict — runs exactly as it does at the rig.

class _WiredModem:
    """A `ModemCore`-shaped stand-in whose far end is a real answering RMS
    session: connect succeeds at once and greets, and every transmitted byte
    feeds the RMS, whose answer comes back through the observer as delivered
    ARQ payload — the same seam the radio uses."""

    def __init__(self, rms):
        self.rms = rms
        self.obs = None
        self.connected = False

    def set_mycall(self, call):
        pass

    def start(self, obs):
        self.obs = obs

    def stop(self):
        pass

    def connect(self, target, repeats=5):
        self.connected = True
        self.obs.modem_connected(target, 500)
        self._deliver(self.rms.start())

    def transmit(self, blob):
        self._deliver(self.rms.feed(blob))

    def _deliver(self, reply):
        if reply:
            self.obs.modem_data_received("ARQ", reply)

    def disconnect(self):
        self.connected = False
        self.obs.modem_disconnected()

    def abort(self):
        self.connected = False


def _mail_args(tmp_path, **over):
    import types
    args = dict(call="K7ABC", mycall="W9SSJ", mail_send=None, mail_fetch=True,
                mail_to="", mail_subject="", mail_password="", mail_sid="",
                mail_out=str(tmp_path / "in"), mail_timeout=5.0)
    args.update(over)
    return types.SimpleNamespace(**args)


def test_run_mail_carries_an_exchange_over_the_live_observer_seam(tmp_path, capsys):
    from hfmodem.besra.host import run_server
    back = compose("K7ABC", "W9SSJ", "held", b"carried gateway to station.\r\n")
    rms = B2FSession("K7ABC", role="answering", target="W9SSJ", outbox=[back])
    rc = run_server._run_mail(_mail_args(tmp_path), _WiredModem(rms))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "mail: exchange complete" in out
    assert "mail: stage closed" in out
    assert "session ended: the exchange finished and the link was closed" in out
    assert rms.done and not rms.failure
    written = list((tmp_path / "in").glob("*.b2f"))
    assert len(written) == 1 and written[0].read_bytes() == back.render()


def test_a_run_that_never_linked_still_says_how_it_ended(tmp_path, capsys):
    """The two ARDOP arms of the 2026-08-20 slot ended here — six connect
    requests, no answer — and printed no verdict in the shape every PACTOR arm
    of the same slot printed one. `mail: NO LINK` is the moment; `session
    ended:` is the verdict, and it is the line the session reports agree on."""
    from hfmodem.besra.host import run_server

    class _Deaf(_WiredModem):
        def connect(self, target, repeats=5):
            self.obs.modem_status(f"CONNECT TO {target} FAILED!")

    rc = run_server._run_mail(_mail_args(tmp_path, mail_timeout=1.0),
                              _Deaf(B2FSession("K7ABC", role="answering",
                                               target="W9SSJ")))
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "session ended: no link — nothing answered the connect request" in out


def test_a_swallowed_delivery_leaves_the_live_exchange_dead(tmp_path, capsys):
    """The planted counterexample: sever the observer seam for one delivery and
    the greeting never reaches the session, so the run must end red with the
    stage named — not report success around the hole."""
    from hfmodem.besra.host import run_server

    class _Swallows(_WiredModem):
        dropped = False

        def _deliver(self, reply):
            if reply and not self.dropped:
                self.dropped = True
                return
            super()._deliver(reply)

    rms = B2FSession("K7ABC", role="answering", target="W9SSJ",
                     outbox=[compose("K7ABC", "W9SSJ", "held", b"never\r\n")])
    rc = run_server._run_mail(_mail_args(tmp_path, mail_timeout=1.0),
                              _Swallows(rms))
    out = capsys.readouterr().out
    assert rc == 1, "the run claimed success past a lost delivery"
    assert "mail: stage awaiting greeting" in out
    assert "mail: nothing moved" in out


def test_a_failed_exchange_never_reports_itself_complete(tmp_path, capsys):
    """A session that fails is finished — same _DONE state a clean FQ reaches —
    so "done" alone cannot tell the operator which happened. The KY4RY run on
    2026-08-14 printed `exchange complete — closing` and then, nine seconds
    later at teardown, `stage failed`; the operator read the pair as a
    contradiction and could not account for it. Only one of them may print.
    """
    from hfmodem.besra.host import run_server

    class _BreaksItsOwnBlock:
        """An RMS whose greeting is ordinary and whose turn is not: a line that
        is no command, between an FC and the F> that closes it."""

        def start(self):
            return (b"RMS Trimode 1.4.2.0 Welcome to K7ABC\r"
                    b"[WL2K-5.0-B2FWIHJM$]\r;PQ: 55734713\rCMS via K7ABC >\r")

        def feed(self, blob):
            return b"FC EM ABCDEF123456 100 50 0\rnot a command at all\r"

    rc = run_server._run_mail(_mail_args(tmp_path),
                              _WiredModem(_BreaksItsOwnBlock()))
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "mail: exchange failed" in out
    assert "mail: exchange complete" not in out
    assert "mail: stage failed — unexpected line inside a proposal block" in out
    assert out.index("mail: exchange failed") < out.index("mail: stage failed")


def test_onair_mail_dry_run_reports_its_stage(tmp_path):
    """The keying tool's mail flags, end to end on its radioless path: the
    subprocess parses them, attaches the session, and the summary names the
    stage — so a typo dies here instead of at the radio."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "hfmodem.besra.host.run_server",
         "--modem", "loopback", "--call", "K7ABC", "--mycall", "W9SSJ",
         "--mail-fetch", "--mail-timeout", "2",
         "--mail-out", str(tmp_path / "mail")],
        capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "DRY RUN" in r.stdout, (
        "a radioless mail run must say so — its log is otherwise "
        "line-for-line what a real on-air failure prints")
    assert "mail: stage awaiting greeting" in r.stdout
    assert "mail: nothing moved" in r.stdout


def test_a_radio_mail_run_refuses_to_start_unarmed(tmp_path):
    """--call with --radio initiates on its own the moment it starts, so it
    follows the station rule every keying tool does: no key-up without an
    explicit --transmit. The refusal happens before any hardware is touched."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "-m", "hfmodem.besra.host.run_server",
         "--radio", "ft891", "--call", "K7ABC", "--mycall", "W9SSJ",
         "--mail-fetch"],
        capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert r.returncode == 2
    assert "pass --transmit" in r.stderr


# The closing identification at the rig: the real `RadioLink` with its audio
# devices replaced, each burst it plays written down in order with the modem's
# own edges.

class _Wired(_WiredModem):
    expected_session = rx_epoch = None

    def __init__(self, rms, events):
        super().__init__(rms)
        self.events = events

    def connect(self, target, repeats=5):
        from hfmodem.besra.phy.modulator import render_frame
        self.events.append("connect")
        self.audio_out(render_frame(0x50, bytes(range(64)), session_id=1).astype(np.float32))
        super().connect(target, repeats)

    def disconnect(self):
        self.events.append("disconnect")
        if self.fail:
            raise RuntimeError("modem fault")
        super().disconnect()

    def stop(self):
        self.events.append("stop")
        if getattr(self, "stop_failure", False):
            raise RuntimeError("modem stop failed")

    def status(self, text):
        pass

    fail = False


def _closing(monkeypatch, tmp_path, *flags, played=None, interrupt=False, fail=False,
             first_failure=None, cleanup=None, refuse=False, diagnostics=None):
    import sys
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server
    events = []
    modem = _Wired(B2FSession("K7ABC", role="answering", target="W9SSJ"), events)
    modem.fail = fail
    modem.stop_failure = cleanup == "modemstop"
    edges = []

    class Rig:
        line_ptt, settle = False, 0.0

        def ptt(self, on):
            edges.append("key" if on else "unkey")
            if not on and cleanup == "unkey":
                raise RuntimeError("unkey failed")
            return not (on and refuse)

        def stop(self):
            edges.append("rigstop")
            if cleanup == "stop":
                raise RuntimeError("rig stop failed")

    link = radio.RadioLink(modem, rig=Rig() if diagnostics is not None else None)
    if diagnostics is not None:
        diagnostics.update(link=link, edges=edges, events=events)

    def play(audio, fs, device):
        if first_failure is not None and "stop" not in events:
            raise first_failure
        if interrupt and "stop" in events:
            raise KeyboardInterrupt
        events.append(("play", len(audio)))
        if played is not None:
            played.append(audio)

    monkeypatch.setattr(radio, "play_drained", play)
    monkeypatch.setattr(radio, "warm_output", lambda device: None)
    monkeypatch.setattr(run_server, "_radio_modem", lambda args: (modem, link))
    if cleanup == "report":
        def failed_report():
            raise RuntimeError("host report failed")
        monkeypatch.setattr(run_server.rates, "host_report", failed_report)
    monkeypatch.setattr(sys, "argv", ["besra", "--radio", "ft891", "--transmit",
        "--call", "K7ABC", "--mycall", "W9SSJ", "--mail-fetch",
        "--mail-out", str(tmp_path), *flags])
    run_server.main()
    return events


def _id_samples():
    from hfmodem.besra import radio
    from hfmodem.core import cwid
    return round(len(cwid.audio("W9SSJ", fs=radio.SAMPLE_RATE)) * radio.FS_RADIO
                 / radio.SAMPLE_RATE)


def test_an_ardop_mail_run_identifies_straight_after_the_modem_stops(monkeypatch, tmp_path):
    events = _closing(monkeypatch, tmp_path)
    assert events[0] == "connect" and events[1][0] == "play"
    assert events[events.index("stop") + 1:] == [("play", _id_samples())]
    assert events.index("disconnect") < events.index("stop")


def test_cwid_optional_leaves_identification_to_ardop(monkeypatch, tmp_path):
    events = _closing(monkeypatch, tmp_path, "--cwid-optional")
    assert events[-1] == "stop"


@pytest.mark.parametrize('optional', [False, True])
def test_tactical_call_and_distinct_closing_cw(monkeypatch, tmp_path, optional):
    from hfmodem.besra.frame.callsign import pack_callsign, unpack_callsign
    from hfmodem.tests.kestrel.test_station_id import _decode
    from hfmodem.besra import radio
    from hfmodem.core import cwid
    calls, played = [], []
    original = _WiredModem.set_mycall
    def set_mycall(modem, call):
        calls.append(unpack_callsign(pack_callsign(call)))
        return original(modem, call)
    monkeypatch.setattr(_WiredModem, 'set_mycall', set_mycall)
    flags = ['--mycall=TACTIC', '--idcall=K1ABC'] + (['--cwid-optional'] if optional else [])
    _closing(monkeypatch, tmp_path, *flags, played=played)
    assert calls == ['TACTIC']
    assert _decode(played[-1], cwid.WPM, fs=radio.FS_RADIO) == 'K1ABC'


def test_distinct_optional_unwind_names_cw_call(monkeypatch, tmp_path, capsys):
    with pytest.raises(RuntimeError, match='modem fault'):
        _closing(monkeypatch, tmp_path, '--mycall=TACTIC', '--idcall=K1ABC',
                 '--cwid-optional', fail=True)
    output = capsys.readouterr().out
    assert output.count('send K1ABC by hand') == 1 and 'send TACTIC by hand' not in output


def test_bad_cw_call_refuses_before_ardop_hardware(monkeypatch):
    import sys
    from hfmodem.besra.host import run_server
    monkeypatch.setattr(run_server, '_radio_modem', lambda *a: pytest.fail('modem opened'))
    monkeypatch.setattr(sys, 'argv', ['besra', '--radio', 'ft891', '--transmit',
                                    '--call', 'K7ABC', '--mycall', 'TACTIC',
                                    '--idcall=ß', '--mail-fetch'])
    with pytest.raises(SystemExit) as exc:
        run_server.main()
    assert exc.value.code == 2


def test_an_interrupted_ident_says_identify_by_hand_and_still_ends_the_run(
        monkeypatch, tmp_path, capsys):
    with pytest.raises(KeyboardInterrupt):
        _closing(monkeypatch, tmp_path, interrupt=True)
    assert "send W9SSJ by hand" in capsys.readouterr().out


def test_a_run_that_fails_after_keying_says_identify_by_hand(monkeypatch, tmp_path, capsys):
    with pytest.raises(RuntimeError, match="modem fault"):
        _closing(monkeypatch, tmp_path, fail=True)
    assert capsys.readouterr().out.count("send W9SSJ by hand") == 1


def test_an_identification_that_does_not_play_says_identify_by_hand(
        monkeypatch, capsys):
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    class _Refusing:
        def ptt(self, on):
            return False if on else True
        line_ptt, settle = False, 0.0

    monkeypatch.setattr(radio, "warm_output", lambda device: None)
    link = radio.RadioLink(_Wired(None, []), rig=_Refusing())
    link.bursts_sent, link.burst_level = 1, (.3, .6)
    run_server._identify(link, "W9SSJ", 500)
    assert link.bursts_sent == 1
    assert "send W9SSJ by hand" in capsys.readouterr().out


def test_the_closing_id_sounds_at_the_channel_centre_and_the_data_level(
        monkeypatch, tmp_path):
    from hfmodem.besra import radio
    from hfmodem.core import levels
    played = []
    diagnostics = {}
    _closing(monkeypatch, tmp_path, played=played, diagnostics=diagnostics)
    data, ident = played[0], played[-1]
    spectrum = np.abs(np.fft.rfft(ident))
    assert np.fft.rfftfreq(ident.size, 1 / radio.FS_RADIO)[spectrum.argmax()] == \
        pytest.approx(1500, abs=2)
    assert levels.sounding_rms(ident) == pytest.approx(levels.sounding_rms(data), rel=.02)
    assert np.abs(ident).max() < np.abs(data).max()
    assert diagnostics["link"].burst_level[0] == pytest.approx(levels.sounding_rms(data))


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit("SIGTERM"),
                                    RuntimeError("first playback failed")])
@pytest.mark.parametrize("cleanup", [None, "stop", "report", "modemstop", "unkey"])
def test_first_ardop_burst_unwind_warns_after_unkey_and_preserves_signals(
        monkeypatch, tmp_path, capsys, failure, cleanup):
    diagnostics = {}
    expected = type(failure) if not isinstance(failure, Exception) or cleanup is None \
        else RuntimeError
    with pytest.raises(expected) as stopped:
        _closing(monkeypatch, tmp_path, first_failure=failure, cleanup=cleanup,
                 diagnostics=diagnostics)
    if not isinstance(failure, Exception) or cleanup is None:
        assert stopped.value is failure
    link = diagnostics["link"]
    assert diagnostics["edges"] == ["key", "unkey", "rigstop"]
    assert link.bursts_sent == 0 and link.burst_level is None
    assert capsys.readouterr().out.count("send W9SSJ by hand") == 1


@pytest.mark.parametrize("cleanup", [None, "stop"])
def test_the_host_report_keeps_session_threads_but_prints_after_radio_close(
        monkeypatch, tmp_path, capsys, cleanup):
    import threading
    from hfmodem.besra import radio
    from hfmodem.besra.host import run_server

    retired = threading.Event()
    worker = threading.Thread(target=retired.wait, name="test-session-rx")
    worker.start()
    sample = run_server.rates.host_report
    close = radio.RadioLink.close
    reports = []

    def assembled_report():
        assert worker.is_alive(), "session threads retired before the host sample"
        report = sample()
        reports.append(report)
        return report

    def retire(link):
        assert len(reports) == 1
        assert reports[0] not in capsys.readouterr().out
        try:
            close(link)
        finally:
            retired.set()
            worker.join(timeout=1)
            assert not worker.is_alive()
            print("radio teardown finished")

    monkeypatch.setattr(run_server.rates, "host_report", assembled_report)
    monkeypatch.setattr(radio.RadioLink, "close", retire)
    try:
        if cleanup is None:
            _closing(monkeypatch, tmp_path, diagnostics={})
        else:
            with pytest.raises(RuntimeError, match="rig stop failed"):
                _closing(monkeypatch, tmp_path, cleanup=cleanup, diagnostics={})
        out = capsys.readouterr().out
        assert out.count(reports[0]) == 1
        assert out.index("radio teardown finished") < out.index(reports[0])
    finally:
        retired.set()
        worker.join(timeout=1)


def test_an_ardop_session_with_only_refused_keys_warns_without_keying_id(
        monkeypatch, tmp_path, capsys):
    diagnostics = {}
    _closing(monkeypatch, tmp_path, refuse=True, diagnostics=diagnostics)
    assert diagnostics["link"].bursts_sent == 0
    assert diagnostics["edges"] == ["key", "unkey", "rigstop"]
    out = capsys.readouterr().out
    assert "no completed data burst for identification level" in out
    assert out.count("send W9SSJ by hand") == 1
