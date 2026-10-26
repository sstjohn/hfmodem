# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CW identification obeys TX guards and never runs during shutdown unwinding."""
import inspect
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import cwid, levels
from hfmodem.shrike import onair, ota, pactor1, ptc
from hfmodem.tests.shrike.test_armdefaults import BASE, _parsed
from hfmodem.tests.shrike.test_beacon import LADDER, every_opcode, ladder_rig
from hfmodem.tests.shrike.test_beacon import stop_handler  # noqa: F401 -- a fixture
from hfmodem.tests.shrike.test_connected_hush import grid
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

CALL = "W9SSJ"


class Clock(_Bench):
    def __init__(self, failure=None):
        super().__init__(seconds=30)
        self.failure = failure
        self.sent = []
        self.closed = False

    def transmit(self, audio, **kwargs):
        if self.failure is not None:
            raise self.failure
        self.sent.append(audio)
        return super().transmit(audio, **kwargs)

    def close(self):
        self.closed = True


def transmitter(tmp_path, *, failure=None, max_key=40):
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, drive=.3,
                      max_key=max_key)
    tx.live = Clock()
    held, skipped = [], []
    tx.sessrx = SimpleNamespace(bridge=lambda a: held.append(len(a)), skip=skipped.append)
    tx._tx(pactor1.control_signal(pactor1.CS_SPEED), "P1 data")
    assert tx.keyed and tx.burst_level is not None
    tx.live.sent.clear()
    tx.rig.edges.clear()
    held.clear()
    skipped.clear()
    tx.live.failure = failure
    tx.aim(grid(0), 1)
    return tx, held, skipped


def test_dry_runs_and_radioless_calls_do_not_emit_or_write_an_ident(tmp_path, capsys):
    for transmit, rig in ((False, _Rig()), (True, None)):
        tx = onair.RadioTx(rig, transmit=transmit, outdir=tmp_path)
        tx.identify(CALL)
    assert not list(tmp_path.iterdir())
    assert not capsys.readouterr().out


def test_an_armed_session_without_a_completed_burst_warns_without_keying(tmp_path, capsys):
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path)
    tx.identify(CALL)
    assert not tx.rig.edges and tx.burst_level is None
    out = capsys.readouterr().out
    assert "no completed data burst for identification level" in out
    assert out.count(f"send {CALL} by hand") == 1


def test_ident_uses_session_drive_and_records_airtime_and_receiver_clock(tmp_path):
    tx, held, skipped = transmitter(tmp_path)
    tx.identify(CALL)
    assert tx.rig.edges == [True, False]
    assert np.isclose(np.abs(tx.live.sent[0]).max(), .3)
    assert len(tx.keyed) == 2 and tx.slots_used == [1]
    assert tx.keyed[-1][1] == pytest.approx(cwid.duration(CALL), abs=.01)
    assert tx.tx_end == tx.live.pos == tx.live.emissions[-1][1]
    assert tx.keyings[-1] == (tx.tx_key_up, tx.tx_end)
    assert held and skipped


def test_ident_advances_past_the_last_used_slot_before_rendering(tmp_path, monkeypatch):
    tx, _, _ = transmitter(tmp_path)
    tx.slots_used.append(1)
    tx.live.now = tx.boundary + 48000
    aims = []
    emit = tx._tx

    def record_aim(*args, **kwargs):
        aims.append(tx.boundary)
        return emit(*args, **kwargs)

    monkeypatch.setattr(tx, "_tx", record_aim)
    tx.identify(CALL)
    assert aims == [tx.tx_audio_start] and tx.slot > 1


def test_ident_waits_out_our_peers_projected_frame_and_then_keys(tmp_path, capsys):
    """A codeword read in the link's last cycle is projected forward on the
    peer's raster; the ID moves past it, aged on the sample clock, and keys
    within `ISS_GUARD_CYCLES` slots rather than refusing for ever."""
    tx, _, _ = transmitter(tmp_path)
    at = tx.boundary + 50000 - tx.raster.cycle_n
    tx.raster.note_peer_codeword(at, 5760, "CS1/ack", "K7ABC")
    tx.identify(CALL)
    assert len(tx.keyed) == 2 and tx.slot > 1
    assert tx.tx_key_up - at > onair.ISS_GUARD_CYCLES * tx.raster.cycle_n + 5760
    out = capsys.readouterr().out
    assert "waits a slot past our peer" in out and "by hand" not in out


def test_a_peer_projection_inside_its_life_still_moves_the_ident(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    tx.raster.note_peer_codeword(tx.boundary + 50000, 5760, "CS1/ack", "K7ABC")
    last = tx.boundary + 50000 + onair.ISS_GUARD_CYCLES * tx.raster.cycle_n + 5760
    tx.raster.forget_stale(last)
    assert tx.raster.peer_cs is not None
    tx.raster.forget_stale(last + 1)
    assert tx.raster.peer_cs is None
    tx.identify(CALL)
    assert len(tx.keyed) == 2


def test_ident_answers_to_none_of_the_links_data_admissions(tmp_path, capsys):
    """After a P4 probe its request stays set, a P1 hold can still hold audio,
    and an observed hold (after-connect listen or chirp, an exhausted P2
    bridge, an unanswered robust entry, --observe-after-link-down) leaves
    `observing` set; all of them refuse link bursts, and none may refuse the ID."""
    tx, _, _ = transmitter(tmp_path)
    tx.p4_probe = SimpleNamespace(requested=True, emitting=False)
    tx.p4_hold_audio = SimpleNamespace(pending=True)
    tx.host = ptc.PtcHost(peer=tx, mycall=CALL)
    assert tx.host.protocol == onair.Protocol.PACTOR1
    tx.listening = True
    tx.observing = True
    tx._tx(np.ones(4800, np.float32), "P1 pkt#1")
    assert tx.refused and len(tx.keyed) == 1
    tx.identify(CALL)
    assert len(tx.keyed) == 2
    assert "CONFIRMED" not in capsys.readouterr().out


def test_ident_reaims_a_late_slot_through_the_normal_backstop(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    tx.live.now = tx.boundary + 2000
    tx.identify(CALL)
    assert tx.slot > 1
    assert tx.tx_audio_start == tx.boundary


def test_playback_failure_says_identify_by_hand(tmp_path, capsys):
    tx, _, _ = transmitter(tmp_path, failure=RuntimeError("device gone"))
    tx.identify(CALL)
    assert f"send {CALL} by hand" in capsys.readouterr().out
    assert len(tx.keyed) == 1


@pytest.mark.parametrize("failure", [SystemExit("SIGTERM"), KeyboardInterrupt()])
def test_an_interrupted_ident_says_identify_by_hand_and_still_ends_the_run(
        tmp_path, capsys, failure):
    tx, _, _ = transmitter(tmp_path, failure=failure)
    with pytest.raises(type(failure)):
        tx.identify(CALL)
    assert f"send {CALL} by hand" in capsys.readouterr().out
    assert len(tx.keyed) == 1


def test_short_max_key_refuses_before_playback(tmp_path, capsys):
    tx, _, _ = transmitter(tmp_path, max_key=1)
    tx.identify(CALL)
    assert not tx.live.sent
    assert "exceeds --max-key" in capsys.readouterr().out


@pytest.mark.parametrize("ending", [SystemExit("SIGTERM"), KeyboardInterrupt(), None])
def test_session_shutdown_closes_devices_and_writes_mail_without_keying_id(
        tmp_path, monkeypatch, capsys, ending):
    from hfmodem import winlink
    clock = Clock()
    rig = _Rig()
    stopped, mail_written, ids = [], [], []
    monkeypatch.setattr(rig, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(onair.ota, "Rig", lambda *a, **kw: rig)
    monkeypatch.setattr(onair.ota, 'ptt_device', lambda *a, **kw: None)
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair, "_LiveInput", lambda *a, **kw: clock)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(winlink, "write_inbox", lambda *a: mail_written.append(True) or [])
    original = onair.RadioTx.identify

    def identify(tx, call):
        ids.append(call)
        return original(tx, call)

    monkeypatch.setattr(onair.RadioTx, "identify", identify)
    if ending is not None:
        def interrupted(_host):
            raise ending
        monkeypatch.setattr(onair.PtcHost, "tick", interrupted)
    args = _parsed("--transmit", "--serial", "/dev/null", "--dial", "7100000",
                   "--outdir", str(tmp_path / 'capture'), "--mail-fetch", "--max-key", "1",
                   "--max-cycles", "1")
    if ending is not None:
        with pytest.raises(type(ending)):
            onair.run(args)
    else:
        onair.run(args)
    assert clock.closed and stopped and mail_written
    assert bool(ids) == (ending is None)
    assert all(len(a) / onair.FS < 1 for a in clock.sent)
    log = capsys.readouterr().out
    assert "PTT off." in log and "transmit cadence:" in log
    if ending is not None:
        assert log.count(f"send {args.mycall} by hand") == 1


def _ctrl_c(_host):
    raise KeyboardInterrupt


def _interrupted(tmp_path, monkeypatch, *, stop, summary, tick=_ctrl_c):
    """A flown session stopped by Ctrl-C, run through the shipped entry point."""
    rig = _Rig()
    events = []
    monkeypatch.setattr(rig, "stop", lambda: stop(events))
    monkeypatch.setattr(rig, "ptt", lambda on: events.append(on) or True)
    monkeypatch.setattr(onair.ota, "Rig", lambda *a, **kw: rig)
    monkeypatch.setattr(onair.ota, 'ptt_device', lambda *a, **kw: None)
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair, "_LiveInput", lambda *a, **kw: Clock())
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)

    monkeypatch.setattr(onair.PtcHost, "tick", tick)
    monkeypatch.setattr(onair, "_summary", lambda *a, **kw: summary(events))
    monkeypatch.setattr(sys, "argv", BASE + [
        "--transmit", "--serial", "/dev/null", "--dial", "7100000",
        "--outdir", str(tmp_path / 'capture'), "--max-key", "1", "--max-cycles", "1"])
    return onair.main_refusing(), events


def test_a_second_ctrl_c_during_the_verdict_leaves_the_rig_down(
        tmp_path, monkeypatch, capsys):
    def again(events):
        events.append("verdict")
        raise KeyboardInterrupt

    code, events = _interrupted(tmp_path, monkeypatch, stop=lambda e: e.append("stop"),
                                summary=again)
    assert code == 130
    assert events[-1] == "verdict" and True in events[:events.index("stop")]
    assert True not in events[events.index("stop"):]
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.usefixtures("stop_handler")
def test_a_second_ctrl_c_inside_the_unkey_waits_for_the_power_restore(
        tmp_path, monkeypatch, capsys):
    ladder = []
    rig, deadmen = ladder_rig(monkeypatch, ladder)
    monkeypatch.setattr(onair, "_phase_power", lambda *_a: SimpleNamespace(
        restore=lambda: ladder.append("restore")))
    code, _ = _interrupted(tmp_path, monkeypatch, stop=lambda _e: rig.stop(),
                           summary=lambda _e: None)
    assert code == 130
    assert ladder == LADDER + ["restore"]
    assert deadmen[0] == ota.DEADMAN_BUDGET_S
    assert ota._depth == 0 and ota._pending is None
    assert "PTT off." in capsys.readouterr().out


def test_the_rest_of_a_stopped_unwind_gets_what_closing_and_draining_wait(
        tmp_path, monkeypatch):
    """What the teardown's hold hands the deadman must cover the waits that
    follow it: the card's close joining the recording thread, and the drain
    waiting on queued WAV writes for as long as `run` lets it."""
    handed = []
    held = onair.signals_held

    def recorded(after_s=None):
        handed.append(after_s)
        return held(after_s)

    monkeypatch.setattr(onair, "signals_held", recorded)
    code, _ = _interrupted(tmp_path, monkeypatch, stop=lambda _e: None,
                           summary=lambda _e: None)
    drain = inspect.signature(onair._drain_captures).parameters["timeout"].default
    assert code == 130 and handed, handed
    assert handed[0] >= onair.REC_JOIN_S + drain, (
        f"{handed[0]} s handed over for a {onair.REC_JOIN_S} s join and a {drain} s drain")


@pytest.mark.usefixtures("stop_handler")
def test_a_second_signal_between_finally_and_the_hold_still_stops_the_rig(
        tmp_path, monkeypatch):
    """The first signal starts the unwind; a second one, at any opcode between
    `_drive_session`'s `finally:` and the hold it opens, must not cost `Rig.stop`."""
    run = inspect.unwrap(onair._drive_session)
    source, first = inspect.getsourcelines(run)
    begin = next(i for i, line in enumerate(source)
                 if line.strip() == "unwinding = sys.exc_info()[1]") - 1
    end = next(i for i in range(begin, len(source)) if "with signals_held(" in source[i])
    regions = {run.__code__: set(range(first + begin, first + end + 1))}

    def first_signal(_host):
        signal.raise_signal(signal.SIGINT)

    class Trial:
        trials = 0

        def run(self):
            self.trials += 1
            where = tmp_path / str(self.trials)
            where.mkdir()
            self.code, self.events = _interrupted(
                where, monkeypatch, stop=lambda e: e.append("stop"),
                summary=lambda _e: None, tick=first_signal)

        def check(self, _outcome, n, _real):
            assert "stop" in self.events, f"opcode {n} skipped Rig.stop"
            assert self.code == 130, f"opcode {n}: {self.code}"
            monkeypatch.setattr(ota, "_pending", None)

    assert every_opcode(regions, signal.SIGINT, Trial()) > 5


@pytest.mark.parametrize("ending,status", [
    (KeyboardInterrupt, 130),
    (lambda: onair.Signalled(signal.SIGTERM), 143),
    (lambda: onair.Signalled(signal.SIGHUP), 129)])
def test_a_teardown_failure_under_a_stop_signal_is_still_said(
        tmp_path, monkeypatch, capsys, ending, status):
    def stopped(_host):
        raise ending()

    def fails(_events):
        raise RuntimeError("rigctl did not answer the unkey")

    code, _ = _interrupted(tmp_path, monkeypatch, stop=fails, summary=lambda e: None,
                           tick=stopped)
    assert code == status
    assert "RuntimeError: rigctl did not answer the unkey" in capsys.readouterr().err


def test_an_exception_the_session_handled_is_not_printed_as_a_teardown_failure(
        tmp_path, monkeypatch, capsys):
    def ctrl_c_while_handling(_host):
        try:
            raise ValueError("the session handled this")
        except ValueError:
            raise KeyboardInterrupt

    code, _ = _interrupted(tmp_path, monkeypatch, stop=lambda _e: None,
                           summary=lambda _e: None, tick=ctrl_c_while_handling)
    assert code == 130
    assert "the session handled this" not in capsys.readouterr().err


@pytest.mark.parametrize("linked,optional,ids", [
    (False, False, 1), (True, False, 1), (False, True, 1), (True, True, 0)])
def test_a_linked_session_closes_with_cw_unless_cwid_is_optional(linked, optional, ids):
    from unittest.mock import Mock
    from hfmodem.tests.shrike.test_p4_rx_exception_0344 import run_reporting
    tx = SimpleNamespace(identify=Mock())
    run_reporting(Mock(), Mock(), tx, linked=linked,
                  args=SimpleNamespace(mycall=CALL, p4_link=True, cwid_optional=optional))
    assert tx.identify.call_count == ids


def test_cwid_optional_is_off_by_default():
    assert not _parsed().cwid_optional
    assert _parsed("--cwid-optional").cwid_optional


@pytest.mark.parametrize('linked', [False, True])
def test_distinct_cw_survives_optional_link_suppression(linked):
    from unittest.mock import Mock
    from hfmodem.tests.shrike.test_p4_rx_exception_0344 import run_reporting
    tx = SimpleNamespace(identify=Mock())
    run_reporting(Mock(), Mock(), tx, linked=linked,
                  args=SimpleNamespace(mycall='TACTIC', idcall='K1ABC', p4_link=True,
                                       cwid_optional=True))
    tx.identify.assert_called_once_with('K1ABC')


def test_pactor_tactical_digital_call_and_distinct_cw_audio(tmp_path, monkeypatch):
    from hfmodem.tests.kestrel.test_station_id import _decode
    clock, rig, calls = Clock(), _Rig(), []
    monkeypatch.setattr(onair.ota, 'Rig', lambda *a, **kw: rig)
    monkeypatch.setattr(onair.ota, 'ptt_device', lambda *a, **kw: None)
    monkeypatch.setattr(onair, 'find_device', lambda *a, **kw: 0)
    monkeypatch.setattr(onair, '_LiveInput', lambda *a, **kw: clock)
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    original = onair.PtcHost
    def host(*a, **kw):
        instance = original(*a, **kw)
        calls.append((kw['mycall'], instance.p1_setup_payload()))
        return instance
    monkeypatch.setattr(onair, 'PtcHost', host)
    args = _parsed('--mycall=TACTIC', '--idcall=K1ABC', '--cwid-optional',
                   '--transmit', '--serial', '/dev/null', '--dial', '7100000',
                   '--outdir', str(tmp_path / 'capture'), '--max-cycles', '1')
    onair.run(args)
    assert calls[0][0] == 'TACTIC' and calls[0][1].upper().endswith(b'TACTIC\r')
    assert _decode(clock.sent[-1], cwid.WPM) == 'K1ABC'


def test_pactor_invalid_cw_identity_stops_before_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(onair.ota, 'Rig', lambda *a, **kw: pytest.fail('rig opened'))
    monkeypatch.setattr(onair, 'find_device', lambda *a, **kw: pytest.fail('audio opened'))
    args = _parsed('--idcall=ß', '--transmit', '--serial', '/dev/null',
                   '--outdir', str(tmp_path))
    with pytest.raises(SystemExit, match='ASCII'):
        onair.run(args)


def test_the_closing_id_sounds_at_the_channel_centre_and_the_data_level(tmp_path):
    from hfmodem.core.occupied import occupied_hz
    tx, _, _ = transmitter(tmp_path)
    t = np.arange(round(.96 * onair.FS)) / onair.FS
    phases = np.random.default_rng(2).uniform(0, 2 * np.pi, 18)
    data = sum(np.sin(2 * np.pi * (500 + 100 * k) * t + p) for k, p in enumerate(phases))
    tx._tx(data.astype(np.float32), "P3 data")
    tx.identify(CALL)
    sent, ident = tx.live.sent
    spectrum = np.abs(np.fft.rfft(ident))
    assert np.fft.rfftfreq(ident.size, 1 / onair.FS)[spectrum.argmax()] == \
        pytest.approx(sum(occupied_hz("pactor")) / 2, abs=2)
    assert levels.sounding_rms(ident) == pytest.approx(levels.sounding_rms(sent), rel=.01)
    assert np.abs(ident).max() < np.abs(sent).max()


def test_a_guard_that_never_lifts_gives_up_after_its_bound_and_says_so(tmp_path, capsys):
    tx, _, _ = transmitter(tmp_path)
    aimed = []
    tx.raster.key_refusal = lambda *a, **k: aimed.append(tx.boundary) or "always busy"
    tx.identify(CALL)
    assert len(aimed) == onair.ISS_GUARD_CYCLES + 2 == len(set(aimed))
    assert len(tx.keyed) == 1 and not tx.live.sent
    assert not tx.rig.edges
    out = capsys.readouterr().out
    assert out.count("waits a slot past our peer") == onair.ISS_GUARD_CYCLES + 2
    assert out.count(f"IDENTIFICATION NOT CONFIRMED (ValueError('transmit guard "
                     f"refused identification: always busy')) -- send {CALL} by hand") == 1


def test_a_p3_packet_and_an_irs_onset_age_out_by_their_one_cycle_life(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    r = tx.raster
    width = 4800
    r._p3_peer = (tx.boundary - width, width, r.cycle_n, r.cycles)
    r.peer_onset = tx.boundary - width
    r.forget_stale(tx.boundary - width + r.cycle_n + width)
    assert r._p3_peer is not None
    r.forget_stale(tx.boundary - width + r.cycle_n + width + 1)
    assert r._p3_peer is None and r.peer_onset is not None
    r.forget_stale(tx.boundary - width + r.cycle_n + r.data_n)
    assert r.peer_onset is not None
    r.forget_stale(tx.boundary - width + r.cycle_n + r.data_n + 1)
    assert r._p3_peer is None and r.peer_onset is None


@pytest.mark.parametrize("long", [False, True])
def test_p3_breakin_projection_keeps_its_packet_width_through_settle(
        tmp_path, monkeypatch, long):
    tx, _, _ = transmitter(tmp_path)
    r = tx.raster
    r.protocol = onair.Protocol.PACTOR3
    r._p3_peer_confirmed = True
    r.cycle_long = long
    at = tx.boundary - onair.ISS_GUARD_CYCLES * r.cycle_n - r.data_n
    r.note_peer_codeword(at, onair.P3_CS_N, "BREAK-IN", "K7ABC",
                         protocol=onair.Protocol.PACTOR3)
    assert r._peer_air()[1] == r.data_n
    seen = []
    forget = r.forget_stale

    def age(carrier):
        forget(carrier)
        seen.append((carrier, r.peer_cs))

    monkeypatch.setattr(r, "forget_stale", age)
    tx.identify(CALL)
    assert seen[0][0] == r.boundary(1) - round(tx.settle * onair.FS)
    assert seen[0][1] is not None
    assert seen[-1][1] is None
    assert tx.tx_key_up - at > onair.ISS_GUARD_CYCLES * r.cycle_n + r.data_n
    assert len(tx.keyed) == 2


def test_p3_packet_age_uses_the_cadence_that_the_guard_projects(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    r = tx.raster
    r.protocol = onair.Protocol.PACTOR3
    r.sending = False
    at, width, cadence = tx.boundary, 4800, 3 * r.cycle_n
    r._p3_peer = (at, width, cadence, r.cycles)
    r.forget_stale(at + r.cycle_n + width + 1)
    assert r._p3_peer is not None
    r.forget_stale(at + cadence + width)
    assert r._p3_peer is not None
    assert r.key_refusal(at + cadence + width - 1, 1) is not None
    r.forget_stale(at + cadence + width + 1)
    assert r._p3_peer is None


@pytest.mark.parametrize("failure", [SystemExit("SIGTERM"), KeyboardInterrupt(),
                                    RuntimeError("first playback failed")])
@pytest.mark.parametrize("teardown", [None, "report", "close", "stop"])
def test_first_burst_failure_warns_after_unkey_and_preserves_stop_signals(
        tmp_path, monkeypatch, capsys, failure, teardown):
    events = []

    def stage(name):
        events.append(name)
        if name == teardown:
            raise RuntimeError("cleanup failed")
        return name

    class FirstBurst(Clock):
        def transmit(self, audio, **kwargs):
            kwargs["key"](True)
            try:
                raise failure
            finally:
                kwargs["key"](False)

        def close(self):
            stage("close")

    live, rig = FirstBurst(), _Rig()
    monkeypatch.setattr(rig, "stop", lambda: stage("stop"))
    monkeypatch.setattr(onair.ota, "Rig", lambda *a, **kw: rig)
    monkeypatch.setattr(onair.ota, "ptt_device", lambda *a, **kw: None)
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair, "_LiveInput", lambda *a, **kw: live)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(onair.rates, "host_report", lambda: stage("report"))
    monkeypatch.setattr(onair.RadioTx, "identify", lambda *a: events.append("id"))
    args = _parsed("--transmit", "--serial", "/dev/null", "--dial", "7100000",
                   "--outdir", str(tmp_path / "capture"), "--max-cycles", "1")
    with pytest.raises(type(failure) if not isinstance(failure, Exception) or teardown is None
                       else RuntimeError):
        onair.run(args)
    assert events[0] == "stop" and "close" in events and "id" not in events
    assert rig.edges == [True, False]
    assert capsys.readouterr().out.count(f"send {CALL} by hand") == 1
