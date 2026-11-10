# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A slot a pending P4 hold word gives up is logged as held, not as an overrun.

`_regrid` gives up the slot while hold audio is pending whether or not it is
late, and logged that as `IS GONE ... +0.0 ms`. Keying the on-time slot after an
early hold read instead fails `test_p4_peer_breakin_0927`'s hold route: the
early read's budget clears `pending` before the Y27 is confirmed and PACTOR-1
keys over the break-in.
"""
from types import SimpleNamespace

from hfmodem.shrike import onair
from hfmodem.shrike.p4holdwork import HoldAudio
from hfmodem.tests.shrike.test_p4_regrid_handoff_0929 import recovery_scene

FS = onair.FS
SETTLE_N = round(.040 * FS)


def _scene(tmp_path, monkeypatch):
    live, raster, tx, host, rx, seg, _ = recovery_scene(tmp_path, monkeypatch, None)
    tx.p4_hold_audio = HoldAudio(FS)
    tx.p4_hold_audio.pending = True
    monkeypatch.setattr(onair, '_p4_read_hold', lambda *a, **kw: None)
    return SimpleNamespace(live=live, raster=raster, tx=tx, host=host, rx=rx, seg=seg)


def _regrid(s, slot):
    return onair._regrid(s.live, s.raster, s.tx, s.host, s.rx, slot, s.seg, 0, SETTLE_N)


def test_an_on_time_slot_given_to_the_p4_word_is_logged_held(tmp_path, monkeypatch, capsys):
    s = _scene(tmp_path, monkeypatch)
    assert not onair._late_at_key(s.live, s.tx.key_instant(s.raster, 2), s.tx.cycle_cost_n())
    slot, _, _ = _regrid(s, 2)
    out = capsys.readouterr().out
    assert slot > 2 and not s.live.emissions
    assert f"SLOT 2 {onair.SLOT_HELD}" in out
    assert f"SLOT 2 {onair.SLOT_GONE}" not in out


def test_a_late_slot_is_given_up_on_its_real_lateness(tmp_path, monkeypatch, capsys):
    s = _scene(tmp_path, monkeypatch)
    assert onair._late_at_key(s.live, s.tx.key_instant(s.raster, 1),
                              s.tx.cycle_cost_n()) > round(.001 * FS)
    _regrid(s, 1)
    out = capsys.readouterr().out
    gone = [line for line in out.splitlines() if f"SLOT 1 {onair.SLOT_GONE}" in line]
    assert gone and "by +0.0 ms" not in gone[0], gone
    assert f"SLOT 1 {onair.SLOT_HELD}" not in out
