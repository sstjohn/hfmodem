# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""September 10: recorded late P3 confirmation must reach a safe P3 ACK."""
import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import arq, onair, rxfront, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike.test_granted_entry_retry import granted, repeat
from hfmodem.tests.shrike.test_grid import _Bench, _Rig

FIXTURES = Path(__file__).with_name('fixtures') / 'ws8eoc-0910'
METADATA = FIXTURES / 'metadata.json'


@cache
def rows():
    if not METADATA.exists():
        pytest.skip(f'WS8EOC morning recordings absent: {METADATA}')
    return {r['file']: r for r in json.loads(METADATA.read_text())['fixtures']}


def recorded(name):
    row = rows()[name]
    return row, recorded_pcm({'file': 'ws8eoc-0910/' + name, 'sha256': row['pcm_sha256']})


def fallback():
    host, _ = granted()
    for _ in range(arq.ENTRY_GRANT_CYCLES):
        repeat(host)
    assert host.protocol == spec.Protocol.PACTOR1 and host.arq.role == arq.ISS
    return host


def harness(tmp_path, *, slot=48):
    host = fallback()
    tx = onair.RadioTx(rig=_Rig(), transmit=True, outdir=tmp_path, settle=.04)
    host.peer = tx
    tx.attach(host)
    rx = onair._SessionRx(host)
    tx.sessrx = rx
    bench = _Bench(seconds=100)
    tx.live = bench
    # M06: slot44's stale IRS boundary2682952 minus44*60000 minus40320.
    grid = onair._MasterGrid(2632, 60000, 8880,
                            packet_n=46080, cs_n=5760, d_max_n=6240)
    grid.d_n, grid.d_ref_n = .094 * onair.FS, 46080
    grid.keyed_slot = slot - 1
    tx.aim(grid, slot)
    # These cases measure the staggered SCS control's own geometry, which is
    # the experiment now that the defaults are historical/audio-start.
    tx.p3_control_waveform, tx.p3_control_placement = "current", "pulse-center"
    # The recorded peer carries its own -50 Hz error and the geometry below is
    # the nominal staggered control's. Keying onto the peer's raster is
    # `test_control_offset_follow`'s subject, not this file's.
    tx.p3_follow_offset = "none"
    tx.defer_p3_cs = True
    return host, rx, tx, grid, bench


@pytest.mark.parametrize('reader', ['upgrade', 'anchored'])
def test_recorded_fallback_changes_protocol_before_render_and_rotates_before_key(tmp_path, reader):
    host, rx, tx, grid, bench = harness(tmp_path)
    row, audio = recorded('guard-rms.wav')
    bench._advance(row['end_sample'])
    bench.pos = bench.now
    if reader == 'upgrade':
        # Production speculative scan; original sample origin belongs to it too.
        onair._scan_frame(rx, audio, row['start_sample'], upgrade=True)
    else:
        # A confirmed P3 event may also arrive from the rolling reader.
        from hfmodem.shrike import p3acquire
        ev = p3acquire.changeover(audio, offsets=(-50,)).event
        rx._on(onair.replace(ev, t=ev.t + row['start_sample'] / onair.FS))
    assert host.protocol == spec.Protocol.PACTOR3
    assert host.arq.role == arq.IRS
    assert bytes(host.channel(host.ptchn).rx) == b'RMS'
    assert not bench.emissions, 'the receive event must not emit before grid reversal'
    assert tx._pending_p3_cs == arq.CS_ACK
    assert abs(grid._p3_peer[0] - row['onset_sample']) <= rxfront.SPS
    onair._reverse_before_key(grid, host, tx, 48)
    # Independent P3 short rotation:810ms data minus210ms control =600ms.
    assert grid.anchor == 2632 + 28800
    assert tx.boundary == 2911432
    tx.emit_pending_cs()
    assert len(bench.emissions) == 1 and not tx.refused
    begin, end = bench.emissions[0]
    phase = grid._p3_peer[0]
    peer = phase + ((begin - phase) // 60000) * 60000
    # Actual trimmed waveform and PTT settle, not nominal CS duration.
    assert begin - round(.04 * onair.FS) > peer + 38880
    assert end < peer + 60000
    # The staggered control ends 11000 samples after its leading pulse
    # (independent SCS waveform), rather than after its trimmed audio onset.
    assert end == tx.boundary + min(tx.tx_pulse_offsets) + 11000
    # 44 samples wider than the rotation alone left it: the comb is held on the
    # answer slot of the packet it is acknowledging, not on the connect train's
    # phase carried through the reversal.
    assert peer + 60000 - end == 6280
    assert grid.boundary(48) - phase == 60000 + round(onair.P3_REPLY_S * onair.FS)
    assert onair._grid_reversal(grid, host) is None
    assert grid.anchor == 31476


@pytest.mark.parametrize('name', ['ve3-m02-grant.wav', 've3-m08-grant.wav'])
def test_morning_p1_grant_contrasts_cannot_resurrect_p3(tmp_path, name):
    host, rx, tx, grid, bench = harness(tmp_path)
    row, audio = recorded(name)
    onair._scan_frame(rx, audio, row['start_sample'], upgrade=True)
    assert host.protocol == spec.Protocol.PACTOR1
    assert host.arq.role == arq.ISS
    assert not bench.emissions and not tx._pending_p3_cs


def test_repeated_recorded_changeovers_keep_one_payload_and_one_rotation(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path, slot=42)
    for name, slot in [('first-rms.wav', 42), ('repeat-rms.wav', 45),
                       ('guard-rms.wav', 48), ('next-rms.wav', 50),
                       ('alternate-rms.wav', 51)]:
        row, audio = recorded(name)
        rx.new_cycle()
        tx.aim(grid, slot)
        count = rx.count
        onair._scan_frame(rx, audio, row['start_sample'],
                          upgrade=host.protocol == spec.Protocol.PACTOR1,
                          tracked_only=host.protocol == spec.Protocol.PACTOR3)
        assert rx.count == count + 1
        onair._reverse_before_key(grid, host, tx, slot)
        tx.emit_pending_cs()
        assert not tx.refused
        # ONE rotation, still: the comb only ever moves by the answer slot's own
        # residual off the packet it is acknowledging, never by another 600 ms.
        assert abs(grid.anchor - 31432) <= round(onair.MAX_PULL_S * onair.FS)
        at = grid._peer_raster_position(grid._p3_peer[0], 60000) or grid._p3_peer[0]
        assert (grid.boundary(slot) - at) % 60000 == round(onair.P3_REPLY_S * onair.FS)
        assert host.arq._clean_run == 0
        assert bytes(host.channel(host.ptchn).rx) == b'RMS'
        before = rx.count
        rx.new_cycle()
        # Overlapping readers of one physical frame owe no second ACK/event.
        onair._scan_frame(rx, np.pad(audio, (17, 0)), row['start_sample'] - 17)
        assert rx.count == before
    assert len(bench.emissions) == 5


def test_long_changeover_uses_long_rotation_without_replacing_received_clock(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path, slot=48)
    host.protocol = spec.Protocol.PACTOR3
    host.arq.cycle_long = True
    grid.cycle_long = True
    rx._p3_cycle_n, rx._p3_span = 180000, 160000
    row, audio = recorded('guard-rms.wav')
    onair._scan_frame(rx, audio, row['start_sample'])
    onair._reverse_before_key(grid, host, tx, 48)
    assert grid.anchor == 2632 + round(3.080 * onair.FS)
    assert (rx._p3_cycle_n, rx._p3_span) == (180000, 160000)
    tx.emit_pending_cs()
    assert len(bench.emissions) == 1 and not tx.refused


@pytest.mark.parametrize('long,rotation', [(False, 28800), (True, 147840)])
def test_accepted_protocol_controls_reversal_before_its_first_key(tmp_path, long, rotation):
    host, rx, tx, grid, bench = harness(tmp_path)
    # Event acceptance changed both mode and role; last keyed geometry is P1.
    host.protocol, host.arq.role = spec.Protocol.PACTOR3, arq.IRS
    grid.cycle_long = host.arq.cycle_long = long
    onair._reverse_before_key(grid, host, tx, 48)
    assert tx.boundary == 2632 + 48 * 60000 + rotation
    assert grid.protocol == spec.Protocol.PACTOR3
