# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""CW identification obeys TX guards and never runs during shutdown unwinding."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.core import cwid
from hfmodem.shrike import onair
from hfmodem.tests.shrike.test_armdefaults import _parsed
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
    tx.live = Clock(failure)
    tx.keyed.append((0, .96))  # A call was actually transmitted.
    held, skipped = [], []
    tx.sessrx = SimpleNamespace(bridge=lambda a: held.append(len(a)), skip=skipped.append)
    tx.aim(grid(0), 1)
    return tx, held, skipped


def test_dry_runs_replays_and_unkeyed_calls_do_not_emit_or_write_an_ident(tmp_path, capsys):
    for transmit in (False, True):
        tx = onair.RadioTx(_Rig(), transmit=transmit, outdir=tmp_path)
        tx.identify(CALL)
    assert not list(tmp_path.iterdir())
    assert not capsys.readouterr().out


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


@pytest.mark.parametrize("guard_enabled", [True, False])
def test_ident_never_stands_down_or_bypasses_an_occupied_reply_slot(tmp_path, guard_enabled):
    tx, _, _ = transmitter(tmp_path)
    tx.qrm_guard = guard_enabled
    tx.raster.note_peer_codeword(tx.boundary + 50000, 5760, "CS1/ack", "K7ABC")
    for _ in range(onair.GUARD_MAX_DROPS + 2):
        tx.identify(CALL)
    assert not tx.live.sent and not tx.rig.edges
    assert len(tx.keyed) == 1


def test_ident_reaims_a_late_slot_through_the_normal_backstop(tmp_path):
    tx, _, _ = transmitter(tmp_path)
    tx.live.now = tx.boundary + 2000
    tx.identify(CALL)
    assert tx.slot > 1
    assert tx.tx_audio_start == tx.boundary


@pytest.mark.parametrize("failure", [RuntimeError("device gone"), SystemExit("max-key"), KeyboardInterrupt()])
def test_playback_failure_cannot_escape_identification(tmp_path, capsys, failure):
    tx, _, _ = transmitter(tmp_path, failure=failure)
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
                   "--outdir", str(tmp_path), "--mail-fetch", "--max-key", "1",
                   "--max-cycles", "1")
    if isinstance(ending, SystemExit):
        with pytest.raises(SystemExit, match="SIGTERM"):
            onair.run(args)
    else:
        onair.run(args)
    assert clock.closed and stopped and mail_written
    assert bool(ids) == (ending is None)
    assert all(len(a) / onair.FS < 1 for a in clock.sent)
    log = capsys.readouterr().out
    assert "PTT off." in log and "transmit cadence:" in log
