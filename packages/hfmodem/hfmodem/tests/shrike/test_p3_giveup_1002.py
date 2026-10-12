# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from hfmodem.shrike import arq, onair, placement, rxfront, spec
from hfmodem.tests.shrike.test_entry_answer import _Session, _head_at
from hfmodem.tests.shrike.test_p3level_session import _Seam
from hfmodem.tests.shrike.test_p3_input_loss import corroborated
from hfmodem.tests.shrike.test_v27_arq_emission_opportunity import linked, packet
from hfmodem.winlink import B2FSession, MailClient, compose


@pytest.mark.parametrize('payload', [b'', b'FS Y\r'])
@pytest.mark.parametrize('settle_role', [arq.IRS, arq.ISS])
def test_crc_resets_silence_before_tick_or_role_change(payload, settle_role):
    a, io = linked()
    a._silent_cycles = a.cfg.max_retries
    a._dark_windows = arq.P3_DARK_WINDOWS - 1
    a._p3_crc_wait_ticks = arq.P3_CRC_WAIT_TICKS - 1
    token = a.begin_receive_opportunity()
    a.on_cycle()
    packet(a, payload=payload)
    assert a._silent_cycles == 0
    # Current-opportunity attribution is settled before the response, not by
    # a potentially historical receive callback. Silence still resets above.
    assert a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS - 1
    a.role = settle_role
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == a._dark_windows == a._p3_crc_wait_ticks == 0
    assert not a._qrt_pending and io.delivered == payload


@pytest.mark.parametrize('evidence', ['crc_idle', 'occupied', 'carrier'])
def test_109_second_proposal_wait_cannot_spend_silence_on_peer(evidence):
    a, io = linked()
    a._silent_cycles = a.cfg.max_retries
    for _ in range(88):
        token = a.begin_receive_opportunity()
        a.on_cycle()
        if evidence == 'crc_idle':
            packet(a, payload=b'')
        elif evidence == 'occupied':
            a.note_burst(100, at_anchor=True)
        else:
            a.note_peer_carrier()
        a.finish_receive_opportunity(token)
        assert a.state == arq.State.CONNECTED and not a._qrt_pending
    token = a.begin_receive_opportunity()
    a.on_cycle()
    packet(a, seq=2, payload=b'FS Y\r')
    a.finish_receive_opportunity(token)
    assert a._silent_cycles == 0 and io.delivered == b'FS Y\r'


@pytest.mark.parametrize('evidence', ['none', 'heard', 'off_anchor'])
def test_dead_peer_goes_quiet_at_six_then_ends_at_crc_lifetime(evidence):
    a, io = linked()
    for index in range(arq.P3_CRC_WAIT_TICKS):
        token = a.begin_receive_opportunity()
        a.on_cycle()
        if evidence == 'heard':
            a.note_peer_heard()
        elif evidence == 'off_anchor':
            a.note_burst(-17, at_anchor=False)
        a.finish_receive_opportunity(token)
        if arq.P3_DARK_WINDOWS <= index+1 < arq.P3_CRC_WAIT_TICKS:
            assert a.p3_receive_only
    assert a.state == arq.State.DISCONNECTED and not a._qrt_pending
    assert not io.packets


def test_carrier_does_not_forgive_a_failed_recovery():
    a, _ = linked()
    a._silent_cycles = a.cfg.max_retries
    a._recovering = True
    token = a.begin_receive_opportunity()
    a.on_cycle()
    a.note_peer_carrier()
    a.finish_receive_opportunity(token)
    assert a.p3_receive_only and not a._qrt_pending
    assert a._recovering  # Weak occupancy did not repair the link.


@pytest.mark.parametrize('size', [0, 32, 60000])
@pytest.mark.parametrize('template', [False, True])
def test_zero_samples_are_dark_even_without_a_template(size, template):
    peer = onair._PeerSpectrum()
    if template:
        t = np.arange(60000) / onair.FS
        peer.learn(np.sin(2 * np.pi * 1600 * t))
    assert not peer.verdict(np.zeros(size, np.float32), 60000, decoded=False)


def test_missing_template_silence_ends_the_reviewers_unbounded_reproduction():
    a, _ = linked()
    peer, grid = onair._PeerSpectrum(), corroborated()
    budget = onair._HoldBudget(32, unbounded=True)
    silence = np.zeros(60000, np.float32)
    for h in range(600):
        token = a.begin_receive_opportunity()
        assert grid.p3_control_refusal(32 + h) is None
        if peer.verdict(silence, 60000, decoded=False):
            a.note_peer_carrier()
        a.on_cycle()
        a.finish_receive_opportunity(token)
        budget.spend(1)
        if a.state == arq.State.DISCONNECTED:
            break
    assert h + 1 == arq.P3_CRC_WAIT_TICKS
    assert budget.deadline == float('inf')


def sl1_window(*, muted=False):
    raw = placement.link_packet(1, b'', 0)
    seg = np.pad(raw, (max(0, 60000 - len(raw)), 0))[-60000:]
    seg += np.random.default_rng(2026).normal(0, .0001, 60000)
    if muted:
        seg[:12000] = 0
    return seg


@pytest.mark.parametrize('muted', [False, True])
def test_missing_template_sl1_survives_reviewers_six_windows(muted):
    a, _ = linked()
    peer = onair._PeerSpectrum()
    seg = sl1_window(muted=muted)
    assert rxfront.occupied_bw(seg) < peer.WIDE_HZ
    if muted:
        assert rxfront.quiet_level_db(seg) is None
    for _ in range(arq.P3_DARK_WINDOWS):
        token = a.begin_receive_opportunity()
        if peer.verdict(seg, 60000, decoded=False):
            a.note_peer_carrier()
        a.on_cycle()
        a.finish_receive_opportunity(token)
    assert not peer.decoded
    assert a.state == arq.State.CONNECTED and not a._qrt_pending
    assert a._dark_windows == arq.P3_DARK_WINDOWS and a.p3_receive_only
    assert a._p3_crc_wait_ticks == arq.P3_DARK_WINDOWS


def test_cs3_head_enters_irs_without_template_and_survives_mute_release(capsys):
    s = _Session(role=arq.ISS)
    peer = onair._PeerSpectrum()
    audio = np.pad(placement.changeover_packet(b'', 0), (4800, 4800))
    ci, errors, at = _head_at(audio, 4800)
    assert (ci, errors) == (arq.CS_BREAKIN, 0)
    ev = s.rx.sync.control_signal_tracked(audio[:at + 12000], at, details=False)
    assert ev is not None and ev.cs == arq.CS_BREAKIN and ev.packet is None
    s.rx._on(ev, anchored=True)
    a = s.host.arq
    assert a.role == arq.IRS and not s.packets and not peer.decoded
    for h in range(88):
        seg = sl1_window(muted=h < arq.P3_DARK_WINDOWS)
        token = a.begin_receive_opportunity()
        assert peer.verdict(seg, 60000, decoded=False)
        a.note_peer_carrier()
        s.host.tick()
        a.finish_receive_opportunity(token)
        assert a.state == arq.State.CONNECTED and not a._qrt_pending
    assert a._p3_crc_wait_ticks == 88
    token = a.begin_receive_opportunity()
    packet(a, seq=2, payload=b'FS Y\r')
    a.finish_receive_opportunity(token)
    assert a._p3_crc_wait_ticks == a._silent_cycles == a._dark_windows == 0
    assert not a._qrt_pending


@pytest.mark.parametrize('kind', ['sl1', 'muted', 'tone', 'broadband'])
@pytest.mark.parametrize('ticks', [1, arq.LONG_TICKS])
def test_unknown_template_grace_cannot_extend_unbounded_receive_budget(kind, ticks):
    a, _ = linked()
    a.cycle_long = ticks == arq.LONG_TICKS
    peer = onair._PeerSpectrum()
    seg = sl1_window(muted=kind == 'muted')
    if kind == 'tone':
        seg = np.sin(2 * np.pi * 1600 * np.arange(60000) / onair.FS)
    elif kind == 'broadband':
        seg = np.random.default_rng(2026).normal(0, .1, 60000)
    budget = onair._HoldBudget(32, unbounded=True)
    for h in range(arq.P3_CRC_WAIT_TICKS // ticks):
        token = a.begin_receive_opportunity()
        assert peer.verdict(seg, 60000, decoded=False)
        a.note_peer_carrier()
        a.on_cycle(elapsed_ticks=ticks, cycle_ticks=ticks)
        a.finish_receive_opportunity(token)
        budget.spend(ticks)
        assert a._dark_windows == h+1
        if arq.P3_DARK_WINDOWS <= h+1 < arq.P3_CRC_WAIT_TICKS // ticks:
            assert a.p3_receive_only
        if h < arq.P3_CRC_WAIT_TICKS // ticks - 1:
            assert a.state == arq.State.CONNECTED and not a._qrt_pending
    assert not peer.decoded and budget.deadline == float('inf')
    assert a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS
    assert a.state == arq.State.DISCONNECTED and not a._qrt_pending


def finish_goodbye(a, ticks=1):
    for _ in range(arq.GOODBYE_CYCLES * arq.LONG_TICKS + 2):
        token = a.begin_receive_opportunity()
        a.on_cycle(elapsed_ticks=ticks, cycle_ticks=ticks)
        a.finish_receive_opportunity(token)
        if a.state == arq.State.DISCONNECTED:
            return
    pytest.fail('the finite receive budget did not finish its goodbye')


@pytest.mark.parametrize('evidence', ['carrier', 'occupied', 'alternating'])
@pytest.mark.parametrize('ticks', [1, arq.LONG_TICKS, 11])
def test_crc_free_receive_budget_is_finite_despite_weak_evidence(evidence, ticks):
    a, _ = linked()
    a.cycle_long = ticks == arq.LONG_TICKS
    budget = onair._HoldBudget(32, unbounded=True)
    for h in range(600):
        token = a.begin_receive_opportunity()
        if evidence == 'carrier' or (evidence == 'alternating' and h % 2):
            a.note_peer_carrier()
        else:
            a.note_burst(100, at_anchor=True)
        a.on_cycle(elapsed_ticks=ticks, cycle_ticks=ticks)
        a.finish_receive_opportunity(token)
        budget.spend(ticks)
        if a.state == arq.State.DISCONNECTED:
            break
        assert not a._qrt_pending
    assert a.state == arq.State.DISCONNECTED and not a._qrt_pending
    assert h + 1 == (arq.P3_CRC_WAIT_TICKS + ticks - 1) // ticks
    assert budget.deadline == float('inf')


@pytest.mark.parametrize('kind', ['template', 'broadband'])
def test_recorded_template_and_broadband_qrm_cannot_extend_unbounded_hold(kind):
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'p3-giveup-1002.json').read_text())
    cap = Path(__file__).resolve().parents[5] / 'captures' / fixture['arm']
    if not (cap / 'stream.flac').is_file():
        pytest.skip(f'local recording unavailable: {cap}')
    audio = rxfront.load_wav(str(cap / 'stream.flac'))
    frozen = fixture['holds']['64']
    template = audio[frozen['origin']:frozen['end']][-60000:]
    peer = onair._PeerSpectrum()
    peer.learn(template)
    assert peer.decoded
    peer.hear(np.random.default_rng(32).normal(0, 0.0001, 60000).astype(np.float32))
    qrm = (template if kind == 'template' else
           np.random.default_rng(1002).normal(0, 0.1, 60000).astype(np.float32))
    a, _ = linked()
    grid = corroborated()
    budget = onair._HoldBudget(32, unbounded=True)
    for h in range(600):
        token = a.begin_receive_opportunity()
        assert peer.verdict(qrm, 60000, decoded=False)
        if kind == 'broadband':
            assert peer.contrast_db(qrm) <= peer.KNEE_DB
            assert peer.excess >= peer.WIDE_DB
        assert grid.p3_control_refusal(32 + h) is None
        a.note_peer_carrier()
        a.on_cycle()
        a.finish_receive_opportunity(token)
        budget.spend(1)
        if a.state == arq.State.DISCONNECTED:
            break
        assert not a._qrt_pending
    assert a.state == arq.State.DISCONNECTED and not a._qrt_pending
    assert h + 1 == arq.P3_CRC_WAIT_TICKS
    assert budget.deadline == float('inf')


@pytest.mark.parametrize('payload', [b'', b'FS Y\r'])
def test_crc_on_the_final_budget_window_renews_the_full_wait(payload):
    a, _ = linked()
    for _ in range(2):
        for _ in range(arq.P3_CRC_WAIT_TICKS - 1):
            token = a.begin_receive_opportunity()
            a.note_peer_carrier()
            a.on_cycle()
            a.finish_receive_opportunity(token)
            assert not a._qrt_pending
        token = a.begin_receive_opportunity()
        a.on_cycle()
        packet(a, payload=payload)
        a.finish_receive_opportunity(token)
        assert a._p3_crc_wait_ticks == a._silent_cycles == a._dark_windows == 0
        assert a.state == arq.State.CONNECTED and not a._qrt_pending


def test_failed_crc_cannot_renew_the_independent_receive_budget():
    a, _ = linked()
    a.cfg.max_retries = arq.P3_CRC_WAIT_TICKS
    for _ in range(arq.P3_CRC_WAIT_TICKS):
        token = a.begin_receive_opportunity()
        packet(a, crc=False)
        a.note_peer_carrier()
        a.on_cycle()
        a.finish_receive_opportunity(token)
    assert a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS
    assert a.state == arq.State.DISCONNECTED and not a._qrt_pending


@pytest.mark.parametrize('long', [False, True])
def test_receive_windows_without_a_tick_still_spend_the_carrier_bound(long):
    a, _ = linked()
    a.cycle_long = long
    ticks = arq.LONG_TICKS if long else 1
    for _ in range(arq.P3_CRC_WAIT_TICKS // ticks):
        token = a.begin_receive_opportunity()
        a.note_peer_carrier()
        a.finish_receive_opportunity(token)
    assert a._p3_crc_wait_ticks == arq.P3_CRC_WAIT_TICKS
    assert a.state == arq.State.DISCONNECTED and not a._qrt_pending


class ReplaySeam(_Seam):
    def __init__(self):
        super().__init__()
        self.frames = []

    def send_packet(self, sl, payload, status, breakin=False):
        self.frames.append((sl, payload, status, breakin))
        return len(payload)


def _k9kdj_recorded_recovery(capsys, expected_hold, *, tmp_path=None):
    fixture = json.loads((Path(__file__).parent / 'fixtures' /
                          'p3-giveup-1002.json').read_text())
    cap = Path(__file__).resolve().parents[5] / 'captures' / fixture['arm']
    if not (cap / 'stream.flac').is_file():
        pytest.skip(f'local recording unavailable: {cap}')
    assert hashlib.sha256((cap / 'stream.flac').read_bytes()).hexdigest() == fixture['stream_sha256']
    audio = rxfront.load_wav(str(cap / 'stream.flac'))
    windows = {}
    for key, frozen in fixture['holds'].items():
        meta = cap / f'hold_{int(key):02d}.json'
        assert hashlib.sha256(meta.read_bytes()).hexdigest() == frozen['metadata_sha256']
        windows[int(key)] = audio[frozen['origin']:frozen['end']], frozen['origin']
    s = _Session(role=arq.IRS)
    seam = s.host.peer = ReplaySeam()
    a = s.host.arq
    if tmp_path is not None:
        from hfmodem.tests import evidence
        from hfmodem.tests.shrike.p3_receive_only_support import ChargedReturn, DeferredReplay
        from hfmodem.tests.shrike.test_p3_reply_placement import irs_grid
        pins = json.loads(Path(__file__).with_name('fixtures').joinpath(
            'p3_receive_only_k9kdj_1003.json').read_text())
        for path, digest in (('provenance', 'provenance_sha256'),
                             ('original_tx_log', 'original_tx_log_sha256')):
            assert hashlib.sha256((evidence.RECORD/pins[path]).read_bytes()).hexdigest() == pins[digest]
        assert '--no-long-cycle' in json.loads((evidence.RECORD/pins['provenance']).read_text())['command']
        a.cfg.long_cycle = pins['long_cycle_enabled']
        seam = s.host.peer = DeferredReplay(s.host)
        seam.raster = recorded_grid = irs_grid()
        s.rx._p3_presence_gaps = pins['original_tx_masks']
    a.cfg.repeat_gear = 0
    a.cfg.speed_up = 'hold'
    s.rx.p3_receive_offset_hz = -2
    s.rx._p3_offset_fixed = True
    session = B2FSession('W9SSJ', target='K9KDJ', outbox=[compose(
        'W9SSJ', 'SMTP:test@example.net', 'giveup regression', 'accepted body')])
    mail = MailClient(session, a.on_host_data)
    s.host.app = mail
    mail.link_up()
    mail.on_link_data(b'[RMS Trimode-1.3.14.0-B2FHM$]\rK9KDJ >\r')
    assert session.stage == 'awaiting proposal answer'
    a._outbuf.clear()
    a._buffer_raw = 0
    s.host.buffer(0)
    flowing = False
    trace = []
    accepted_hold = None
    for h in range(64, 85):
        window, origin = windows[h]
        s.rx.new_cycle()
        token = a.begin_receive_opportunity()
        if not flowing and h > 64:
            onair._scan_frame(s.rx, *windows[h - 1])
        if h == 72:
            s.rx.sync.packet_level = 2
        if tmp_path is not None:
            assert all(b >= origin+len(window) or e <= origin for b, e in pins['original_tx_masks'])
            seam.slot = h
            recorded_grid.boundary = lambda slot: origin+window.size
            if h == 83:
                pcm = np.rint(window*32768).astype('<i2')
                assert hashlib.sha256(pcm.tobytes()).hexdigest() == pins['return_window_pcm16_sha256']
                charge = ChargedReturn(s, seam, pins['original_tx_masks'], origin+len(window), tmp_path)
        onair._scan_frame(s.rx, window, origin, tracked_only=True)
        flowing = a._receive_opportunity.crc
        from types import SimpleNamespace
        # Actual current-window structure replaces occupancy. The frozen end
        # selects this physical opportunity; a prior-window CRC is insufficient.
        grid = (recorded_grid if tmp_path is not None else
                SimpleNamespace(boundary=lambda slot: origin+window.size))
        seam.slot = h
        onair._p3_qualify_presence(s.rx, grid, seam, window, origin)
        s.host.tick()
        if tmp_path is not None:
            if h == 83:
                charge.mark('receive-settle')
                assert a._receive_opportunity.presence_source == 'current CRC'
                metric = charge.emit(grid, h)
                with capsys.disabled():
                    print('RECORDED_RETURN K9KDJ 83', metric)
            else:
                seam.emit_pending_cs()
        onair._mail_app_turns(s.host, mail, wait_greeting=True)
        qualified = a.peer_qualified
        a.finish_receive_opportunity(token)
        trace.append((h, a._silent_cycles))
        if h in (80, 81):
            assert a.p3_receive_only
        if h == 82:
            assert qualified and not a.p3_receive_only
        if session.sent_mids and accepted_hold is None:
            accepted_hold = h
        assert a.state == arq.State.CONNECTED and not a._qrt_pending, trace
        assert not any(status & spec.STATUS_QRT for _, _, status, _ in seam.frames)
    assert session.sent_mids and session._out[0].answer == 'Y'
    assert accepted_hold == expected_hold
    assert any(ev.packet[2] == b' oct 2\rFS Y\r' for ev in s.packets)
    assert a._buffer_raw > 0 and a._outbuf
    if expected_hold == 84:
        assert a._breakin_pending
    else:
        # The current-copy FS acceptance also moves the ordinary break-in one
        # opportunity earlier. Its field remains unconfirmed modem ownership.
        assert a._inflight is not None and a._inflight.breakin
        assert a._inflight.payload and a._outbuf
        assert a._buffer_raw == len(a._inflight.payload)+len(a._outbuf)
    assert not any('discarded' in line for line in s.host.log_lines)
    capsys.readouterr()


def test_k9kdj_recording_delivers_fs_y_and_keeps_the_accepted_body(capsys, monkeypatch):
    # Preserve the original strict84 pin under the original reader eligibility.
    monkeypatch.setattr(onair._SessionRx, 'sl2_prekey_expected',
                        lambda rx: rx.sync.packet_level == 2 or 2 in rx._p3_expected())
    _k9kdj_recorded_recovery(capsys, 84)


def test_k9kdj_retained_reader_delivers_current_fs_y_at_83(capsys, tmp_path):
    # Approved established-form retention permits the actual current copy;
    # the original reader waits for its previous-window scan in hold84.
    _k9kdj_recorded_recovery(capsys, 83, tmp_path=tmp_path)
