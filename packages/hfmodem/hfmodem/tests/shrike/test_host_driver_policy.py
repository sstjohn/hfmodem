# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""External host budgets on the unchanged native sample-clock caller; no RF."""
from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair
from hfmodem.shrike.arq import State
from hfmodem.shrike.ptc import PtcHost
from hfmodem.tests.shrike.test_grid import _Bench, _Rig


class Clock(_Bench):
    def read(self, count):
        assert max(self.pos, self.floor) + max(0, int(count)) <= self.limit, \
            'host-controlled driver missed its fixture abort/timeout'
        return super().read(count)


def contact(tmp_path, monkeypatch, *, maximum, stop_after=None, heard=False):
    clock = Clock(seconds=50)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.live, tx.p4_probe = clock, None
    host = PtcHost(tx, mycall='N0CALL')
    host.set_listen(True)
    host._store('MYLEVEL', '1')
    host._store('MAXERROR', str(maximum))
    owner = host.enable_owner(defer_connect=True)
    owner.request('connect', 'N0CALL', 'N0DX')
    endpoint = SimpleNamespace(host=host, closed=False, pending_bytes=0)
    if stop_after is not None or heard:
        def peer(clock, end):
            if heard:
                host.arq.note_peer_heard()
            if len(clock.emissions) == stop_after:
                host._request_host('abort')
        clock.peer = peer
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL', '--max-cycles', '1'])
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    identity = SimpleNamespace(cw_call='N0CALL', closing_required=lambda *a, **kw: False)
    code = onair._drive_session(args, tx=tx, host=host, sessrx=rx, live=clock,
        rig=None, dx='N0DX', reply=None, outdir=tmp_path, identity=identity,
        start_admitted=lambda: owner.start_connect(endpoint), host_controlled=True)
    return code, host, tx, clock, args


def test_host_maxerror_call_budget_is_not_cut_off_by_cli_experiment_limits(tmp_path, monkeypatch):
    code, host, tx, clock, args = contact(tmp_path, monkeypatch, maximum=30)
    assert code == 0 and len(tx.keyed) == host.arq._host_sync_packets == 30
    assert args.max_cycles == 1 and not args.p3_timing_unbounded
    assert set(np.diff([start for start, _ in clock.emissions])) == {60000}
    assert host.arq.state == State.LISTENING and not host._owner.contact_active
    assert clock.pos - clock.emissions[-1][1] >= 2*60000


def test_disabled_host_call_timeout_still_obeys_a_later_owner_abort(tmp_path, monkeypatch):
    code, host, tx, clock, args = contact(tmp_path, monkeypatch, maximum=255, stop_after=12)
    assert code == 0 and len(tx.keyed) == host.arq._host_sync_packets == 12
    assert args.max_cycles == 1 and not args.p3_timing_unbounded
    assert host.arq.state == State.LISTENING
    assert not host._owner.contact_active and not host._owner.connect_pending
    assert len(clock.emissions) == 12


@pytest.mark.parametrize('invalid', ['no_owner', 'no_timeout', 'internal_app', 'message', 'over'])
def test_host_controlled_driver_requires_external_application_ownership(tmp_path, invalid):
    clock = _Bench(seconds=2)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path)
    tx.live = clock
    host = PtcHost(tx)
    if invalid != 'no_timeout':
        host._store('MAXERROR', '30')
    if invalid != 'no_owner':
        host.enable_owner()
    args = onair.build_parser().parse_args([])
    if invalid == 'internal_app':
        host.app = object()
    elif invalid == 'message':
        args.message = 'internal payload'
    elif invalid == 'over':
        args.over = True
    with pytest.raises(ValueError, match='external application'):
        onair._drive_session(args, tx=tx, host=host, sessrx=None, live=clock,
            rig=None, dx='N0DX', reply=None, outdir=tmp_path, identity=None,
            host_controlled=True)
    assert not clock.emissions


def test_unidentified_presence_cannot_erase_the_final_host_budget_receive_interval(tmp_path, monkeypatch):
    code, host, tx, clock, _ = contact(tmp_path, monkeypatch, maximum=30, heard=True)
    assert code == 0 and len(tx.keyed) == 30
    assert host.arq._connect_retries == 0  # presence bought the legacy retry counter
    assert clock.pos - clock.emissions[-1][1] >= 2*60000
    assert host.arq.state == State.LISTENING


def test_external_host_holds_a_connected_contact_without_cli_idle_or_hold_cutoff(tmp_path, monkeypatch):
    from hfmodem.shrike import p1rx
    clock = Clock(seconds=30)
    clock.read(p1rx.CALL_N)
    tx = onair.RadioTx(_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    tx.live, tx.p4_probe = clock, None
    host = PtcHost(tx, mycall='N0CALL')
    host._store('MYLEVEL', '1')
    host._store('MAXERROR', '255')
    host.set_listen(True)
    owner = host.enable_owner(defer_connect=True)
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    args = onair.build_parser().parse_args(['--mycall', 'N0CALL', '--hold', '0', '--max-cycles', '1'])
    ticks = []
    tick = host.tick
    def host_tick(**options):
        ticks.append((host.arq.state, host.arq._qrt_pending))
        if len(ticks) == 5:
            host._request_host('abort')
        return tick(**options)
    host.tick = host_tick
    monkeypatch.setattr(onair, '_save_capture_async', lambda *a, **kw: None)
    identity = SimpleNamespace(cw_call='N0CALL', closing_required=lambda *a, **kw: False)
    code = onair._drive_session(args, tx=tx, host=host, sessrx=rx, live=clock,
        rig=None, dx='', reply=None, outdir=tmp_path, identity=identity,
        called=(0, p1rx.Connect('Normal', 'N0CALL', False)), host_controlled=True)
    assert code == 0 and len(ticks) == 5
    assert all(state == State.CONNECTED and not qrt for state, qrt in ticks)
    assert host.arq.state == State.LISTENING and not owner.contact_active
    assert args.hold == 0 and not args.p3_timing_unbounded
