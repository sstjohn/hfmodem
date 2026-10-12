# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Original recorded fades, current attribution and separate CFO controls."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from hfmodem.shrike import arq, onair, p3acquire, rxfront
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_p3_giveup_1002 import _Session
from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid
from hfmodem.tests.shrike.p3_receive_only_support import ChargedReturn, DeferredReplay
from hfmodem.tests.shrike.test_p3_presence_progress_1003 import presence_session

PIN_PATH = Path(__file__).with_name('fixtures') / 'p3_receive_only_recorded_1003.json'
if not PIN_PATH.is_file():
    pytest.skip('private recorded-fade metadata is absent from this distribution', allow_module_level=True)
PINS = json.loads(PIN_PATH.read_text())


@pytest.fixture(scope='module', params=PINS, ids=lambda p: p['arm'])
def recorded(request):
    p = request.param
    path = evidence.require_input(evidence.CAPTURES/f"onair-1003-{p['arm']}/stream.flac")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == p['recording_sha256']
    log = evidence.require_input(evidence.CAPTURES.parent/p['original_tx_log'])
    assert hashlib.sha256(log.read_bytes()).hexdigest() == p['original_tx_log_sha256']
    provenance = evidence.require_input(evidence.RECORD/p['original_provenance'])
    assert hashlib.sha256(provenance.read_bytes()).hexdigest() == p['original_provenance_sha256']
    assert p['long_cycle_enabled'] == ('--no-long-cycle' not in json.loads(provenance.read_text())['command'])
    return p, rxfront.load_wav(str(path))


@pytest.mark.parametrize('original_config', [True, False], ids=['recorded-policy', 'historical-observer-policy'])
def test_stateful_fade_continues_current_form_without_blind_keys(recorded, capsys, tmp_path, original_config):
    p, audio = recorded
    s = _Session(role=arq.IRS)
    a, rx = s.host.arq, s.rx
    seam = s.host.peer = DeferredReplay(s.host)
    seam.raster = grid = irs_grid(anchor=0)
    a.cfg.speed_up, a.cfg.repeat_gear = 'hold', 0
    # The prior design observer accidentally used the default long policy.
    # Preserve its exact outcome separately; it is not actual-arm provenance.
    a.cfg.long_cycle = p['long_cycle_enabled'] if original_config else True
    rx.p3_receive_offset_hz, rx._p3_offset_fixed = p['cfo'], True
    rx.sync.packet_level = p['level']
    # Quiet replay never turns originally transmitted/missing samples into RX.
    rx._p3_presence_gaps = p['original_tx_masks']
    if p['arm'] in ('1047', '1140'):
        rx._p3_row0 = p['reads'][0]['row']
        rx._p3_cycle_n = 60000
    rows, previous, flowing = {}, None, False
    for frozen in p['windows']:
        h, origin, end = frozen['hold'], frozen['origin'], frozen['end']
        meta = evidence.CAPTURES/f"onair-1003-{p['arm']}/hold_{h:02d}.json"
        assert hashlib.sha256(meta.read_bytes()).hexdigest() == frozen['metadata_sha256']
        assert frozen['no_completed_tx_overlap']
        assert all(b >= end or e <= origin for b, e in p['original_tx_masks'])
        window = audio[origin:end]
        rx.new_cycle()
        seam.slot = h
        # The original delivered slice ends in this physical opportunity.
        # The grid's peer tuple still comes only from actual decoded events.
        grid.boundary = lambda slot, end=end: end
        token = a.begin_receive_opportunity()
        if previous is not None and not flowing:
            onair._scan_frame(rx, *previous)
        charged = original_config and (p['arm'], h) in (('1043', 61), ('1047', 59))
        if charged:
            # Original captured PTT minus the final available sample. No future
            # samples, extra search domain, or late-key permission is supplied.
            pcm = np.rint(window*32768).astype('<i2')
            assert hashlib.sha256(pcm.tobytes()).hexdigest() == p['charged_return']['window_pcm16_sha256']
            charge = ChargedReturn(s, seam, p['original_tx_masks'], end, tmp_path)
        onair._scan_frame(rx, window, origin, tracked_only=True)
        if charged:
            charge.mark('scan')
        flowing = a._receive_opportunity.crc
        onair._p3_qualify_presence(rx, grid, seam, window, origin)
        source = a._receive_opportunity.presence_source
        before = len(seam.sent)+len(seam.frames)
        s.host.tick()
        if charged:
            charge.mark('settle')
            assert source == 'current CRC'
            assert rx._p3_delivered_at == p['charged_return']['row']
            metric = charge.emit(grid, h)
            with capsys.disabled():
                print('RECORDED_RETURN', p['arm'], h, metric)
        else:
            seam.emit_pending_cs()
        rows[h] = dict(quiet=a.p3_receive_only, source=source,
                       current_crc=source == 'current CRC',
                       keyed=len(seam.sent)+len(seam.frames)-before,
                       row=rx._p3_delivered_at, crc_ticks=a._p3_crc_wait_ticks)
        a.finish_receive_opportunity(token)
        assert a.state == arq.State.CONNECTED and not a._qrt_pending
        if a.p3_receive_only:
            assert rows[h]['keyed'] == 0
        previous = window, origin
    if p['arm'] == '1047':
        assert all(rows[h]['quiet'] for h in range(54, 59))
        assert all(rows[h]['source'] == 'unsupported' for h in range(49, 59))
        assert not rows[59]['quiet'] and rows[59]['current_crc']
    elif p['arm'] == '1043':
        if original_config:
            assert not rows[61]['quiet'] and rows[61]['current_crc']
            assert rows[61]['row'] == 4253554
            assert all(rows[h]['quiet'] for h in (67, 68))
        else:
            assert all(rows[h]['quiet'] for h in range(61, 69))
        assert rows[68]['source'] == 'unsupported'
        assert not rows[69]['quiet'] and rows[69]['current_crc']
    else:
        assert not any(r['quiet'] for r in rows.values())
    capsys.readouterr()


def test_actual_tail_attribution_never_creates_sixth_response(recorded, capsys):
    p, audio = recorded
    s = presence_session(p['last_crc_row'], cfo=p['cfo'])
    a, rx = s.host.arq, s.rx
    rx.sync.packet_level = p['level']
    rx._p3_presence_form = (a._link_generation, p['level'])
    keys = 0
    for index, row in enumerate(p['tail']):
        token = a.begin_receive_opportunity()
        tx = SimpleNamespace(slot=row['hold'], live=SimpleNamespace(lost=row['lost_samples']))
        a._receive_opportunity.capture_lost = row['lost_samples']
        grid = SimpleNamespace(boundary=lambda slot: row['target']+37000)
        onair._p3_qualify_presence(rx, grid, tx,
                                  audio[row['origin']:row['end']], row['origin'])
        assert not a.peer_qualified and not row['masks']
        before = len(s.host.peer.sent)
        s.host.tick()
        a.finish_receive_opportunity(token)
        keys += len(s.host.peer.sent)-before
        assert a.state == arq.State.CONNECTED
        assert a.p3_receive_only == (index >= 5)
        assert keys == min(index+1, 5)
    assert len(p['tail']) == {'1043': 48, '1047': 43, '1140': 14}[p['arm']]
    capsys.readouterr()


def test_tracked_and_nominal_crc_controls_keep_their_distinct_results(recorded):
    p, audio = recorded
    for pin in p['reads']:
        corrected = p3acquire.compensate(audio[pin['origin']:pin['end']], pin['cfo'])
        reader = rxfront.SyncedRx()
        ev = getattr(reader, f"sl{p['level']}_packet_at")(corrected, pin['row']-pin['origin'])
        crc = bool(ev is not None and ev.packet and ev.packet[3])
        assert crc == pin['crc']
        if crc:
            assert ev.packet[1] == pin['status']
            assert hashlib.sha256(ev.packet[2]).hexdigest() == pin['digest']
