# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

import ast
import inspect
from types import SimpleNamespace

import pytest

from hfmodem.shrike import onair
from hfmodem.tests.shrike.p3_cycle_support import emit, grant, put_packet, scene


def hold_receive(s, tx, g, live, *, flowing=True, whole=None, prev_start=None):
    # Execute the driver's scan, listen and bridge statements themselves: copying
    # their intended order here would conceal the bridge regression.
    tree = ast.parse(inspect.getsource(onair._drive_session))
    listen = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and any(isinstance(child, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'win'
                                  for t in child.targets) for child in node.body))
    start = next(i for i, node in enumerate(listen.body)
                 if any(isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name)
                        and child.func.id == '_scan_previous_window'
                        for child in ast.walk(node)))
    bridge = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and node.body and isinstance(node.body[0], ast.Assign)
                  and isinstance(node.body[0].targets[0], ast.Name)
                  and node.body[0].targets[0].id == 'seg_start'
                  and any(isinstance(child, ast.Call)
                          and isinstance(child.func, ast.Name)
                          and child.func.id == '_p3_transition_window'
                          for child in ast.walk(node)))
    code = ast.Module(body=listen.body[start:] + bridge.body, type_ignores=[])
    scope = dict(vars(onair), live=live, raster=g, tx=tx, host=s.host, sessrx=s.rx,
                 slot=tx.slot, boundary=tx.boundary, key_at=tx.key_instant(g, tx.slot),
                 whole=live.audio[:0] if whole is None else whole,
                 prev_start=live.pos if prev_start is None else prev_start,
                 settle_n=1920, flowing=flowing, quiet=False, listen=False,
                 args=SimpleNamespace(replay=True))
    exec(compile(code, onair.__file__, 'exec'), scope)
    return scope


@pytest.mark.parametrize('flowing', [False, True])
def test_previous_window_delayed_revert_keys_every_short_boundary(
        tmp_path, monkeypatch, flowing):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    emitted = [grant(s, tx, g, live, False)]
    row = put_packet(s, g, live, 2, False, 0x20, opportunity=2)
    prev_start = live.pos
    whole = live.read(g.boundary(3) - 5000 - live.pos)
    emitted.append(emit(s, tx, g, live, 3))
    assert s.host.arq.cycle_command_emitted
    tx.aim(g, 6)
    tx.arm_p3_reply()
    scan = s.rx.deep_scan

    def charged(audio):
        live.spend(round(.010 * onair.FS))
        return scan(audio)

    monkeypatch.setattr(s.rx, 'deep_scan', charged)
    deadlines = []
    for _ in range(3):
        s.rx.new_cycle()
        result = hold_receive(s, tx, g, live, flowing=flowing,
                              whole=whole, prev_start=prev_start)
        deadlines.append((result['slot'], result['key_at'], live.now))
        emitted.append(emit(s, tx, g, live))
        whole, prev_start = result['whole'], result['seg_start']
        tx.aim(g, g.next_slot(tx.slot))
        tx.arm_p3_reply()

    assert abs(s.rx._p3_delivered_at - row) <= 120
    assert not s.host.arq.cycle_command_emitted
    assert emitted == [(0, 5), (3, 0), (4, 0), (5, 0), (6, 0)]
    assert tx.slots_used == [0, 3, 4, 5, 6]
    assert [at for at, _ in live.submitted] == [
        g.boundary(slot) for slot in [0, 3, 4, 5, 6]]
    for slot, key_at, finished in deadlines:
        assert key_at == tx.key_instant(g, slot)
        assert finished < key_at - live.key_notice


@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize('committed', [False, True])
def test_listen_header_evidence_preserves_owed_burst(
        tmp_path, monkeypatch, pending, committed):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    row = put_packet(s, g, live, 1, True, 0x21, header_only=True)
    if not pending:
        s.host.arq._cycle_request = None
        s.host.arq._cycle_command_emitted = False
        g.p3_interval(False, 0)
    slot = 3 if pending else 1
    tx.aim(g, slot)
    tx.arm_p3_reply()
    burst, armed = live._armed, tx._p3_armed
    if committed:
        live.spend(g.boundary(slot) - 1000 - live.now)
        assert not live.disarm()
    origin = live.pos
    audio = onair._listen_until_answer(
        live, row + 10000 - live.pos, s.host, s.rx, 0,
        ahead=lambda audio, origin: onair._read_ahead(s.rx, g, audio, origin))
    assert any(key[0] == 'header' and value is not None and value.long_cycle
               for key, value in s.rx._p3_ahead.items())
    if not committed and not pending:
        assert burst.stop.is_set() and not burst.keying
        slot = 3
        assert tx.slot == tx._p3_armed.slot == slot
    else:
        assert tx._p3_armed is armed
        assert live._armed is burst and not burst.stop.is_set()

    def slow_decode(audio):
        live.spend(g.boundary(slot) + 100 - live.now)

    monkeypatch.setattr(s.rx, 'deep_scan', slow_decode)
    onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert emit(s, tx, g, live) == (slot, 0)
    assert any(at == g.boundary(slot) for at, _ in live.submitted)


@pytest.mark.parametrize('flowing', [False, True])
@pytest.mark.parametrize('level', [2, 3, 4, 5, 6])
@pytest.mark.parametrize('opportunity', [0, 1])
def test_hold_loop_reads_short_duplicate_before_long_bridge(
        tmp_path, monkeypatch, opportunity, level, flowing):
    s, tx, g, live = scene(tmp_path, monkeypatch)
    grant(s, tx, g, live, False)
    row = put_packet(s, g, live, level, False, 0x20, opportunity=opportunity)
    tx.aim(g, 3)
    tx.arm_p3_reply()
    scan = s.rx.deep_scan

    def charged(audio):
        live.spend(round(.010 * onair.FS))
        return scan(audio)

    monkeypatch.setattr(s.rx, 'deep_scan', charged)
    result = hold_receive(s, tx, g, live, flowing=flowing)
    assert abs(s.rx._p3_delivered_at - row) <= 120
    assert not s.host.arq.cycle_command_emitted
    assert result['slot'] == opportunity + 1
    assert result['key_at'] == tx.key_instant(g, opportunity + 1)
    assert live.now < g.boundary(opportunity + 1) - live.key_notice
    assert emit(s, tx, g, live) == (opportunity + 1, 0)


@pytest.mark.parametrize('level', [3, 4, 5, 6])
@pytest.mark.parametrize('old_long, confirm', [(False, False), (True, True), (True, False), (False, True)])
def test_pending_high_level_crc_confirms_or_reverts(
        tmp_path, monkeypatch, level, old_long, confirm):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=old_long)
    grant(s, tx, g, live, old_long)
    long = not old_long if confirm else old_long
    row = put_packet(s, g, live, level, long,
                     (1 if confirm else 0) | (0 if old_long else 0x20))
    monkeypatch.setattr(s.rx, '_read_p3_packet',
                        lambda *a, **k: pytest.fail('unbounded packet reader'))
    slot = 3 if long or not old_long else 1
    s.rx.new_cycle()
    tx.aim(g, slot)
    tx.arm_p3_reply()
    onair._p3_transition_window(live, g, tx, s.host, s.rx, slot,
                               live.audio[live.pos:live.pos], live.pos, 1920)
    assert abs(s.rx._p3_delivered_at - row) <= 120
    assert s.packets[-1].packet[:3] == (level, (1 if confirm else 0)
                                      | (0 if old_long else 0x20), b'X')
    assert s.host.arq.cycle_long == long
    assert not s.host.arq.cycle_command_emitted
    assert emit(s, tx, g, live) == (3 if long else 1, 1 if confirm else 0)


@pytest.mark.parametrize('level, long', [(4, False), (3, True), (6, True)])
@pytest.mark.parametrize('admitted', [False, True])
def test_pending_high_level_crc_miss_keeps_held_word_within_budget(
        tmp_path, monkeypatch, level, long, admitted):
    s, tx, g, live = scene(tmp_path, monkeypatch, old_long=True)
    grant(s, tx, g, live, True)
    put_packet(s, g, live, level, long, 0, header_only=True)
    slot = 3 if long else 1
    tx.aim(g, slot)
    tx.arm_p3_reply()
    before = s.rx._p3_delivered_at
    calls = []
    decode = onair.p3rx.decode_at

    def read(*args, **kwargs):
        if kwargs["header"].long_cycle == long:
            calls.append(live.now)
            live.spend(round(s.rx.P3_WIDEBAND_RESERVE_S * onair.FS))
        return decode(*args, **kwargs)

    monkeypatch.setattr(onair.p3rx, 'decode_at', read)
    monkeypatch.setattr(s.rx, '_read_p3_packet',
                        lambda *a, **k: pytest.fail('unbounded packet reader'))
    if not admitted:
        live.spend(g.boundary(slot) - live.key_notice - 280 - live.now)
    found, _, _, _ = onair._p3_transition_window(
        live, g, tx, s.host, s.rx, slot,
        live.audio[live.pos:live.pos], live.pos, 1920)
    # Spending to the key notice commits PTT on slot 1; a committed reply keeps its slot.
    expected = 3 if long else 1
    assert found == expected
    assert bool(calls) == admitted
    assert len(calls) <= 1
    assert s.rx._p3_delivered_at == before
    assert s.host.arq.cycle_command_emitted
    assert live.now <= g.boundary(expected) - live.key_notice
    assert emit(s, tx, g, live) == (expected, 0)
