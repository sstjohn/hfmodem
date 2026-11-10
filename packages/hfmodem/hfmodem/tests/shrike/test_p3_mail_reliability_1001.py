# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""W6IDS 1001-0215 carrier/turn pins and VE1YZ 0923-2310 close discipline."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from hfmodem.shrike import arq, onair, p3acquire, rxfront, spec
from hfmodem.tests import evidence
from hfmodem.tests.shrike.test_entry_answer import _Session
from hfmodem.tests.shrike.test_p3_reply_placement import driver, irs_grid
from hfmodem.winlink import B2FSession, MailClient

CAP = evidence.CAPTURES / 'onair-1001-0215'
TRACE = Path(__file__).parent / 'fixtures' / 'p3-mail-reliability-1001.json'
if not (TRACE).exists():
    pytest.skip('p3-mail-reliability-1001.json is not in this tree', allow_module_level=True)
FIXTURE = json.loads(TRACE.read_text())
INPUTS = FIXTURE['nomination_inputs']


def held(n, arm='onair-1001-0215'):
    path = evidence.CAPTURES / arm / f'hold_{n:02d}.flac'
    if not path.exists():
        pytest.skip(f'local recording unavailable: {path}')
    meta = json.loads(path.with_suffix('.json').read_text())
    frozen = INPUTS[str(Path('captures') / arm / path.name)]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen['sha256']
    assert hashlib.sha256(path.with_suffix('.json').read_bytes()).hexdigest() == frozen['metadata_sha256']
    assert meta['end_stream_sample']-meta['samples'] == frozen['origin']
    assert meta['samples'] == frozen['samples']
    return rxfront.load_wav(str(path)), meta['end_stream_sample']-meta['samples']


def radio(s, tmp_path):
    g = irs_grid()
    tx = driver(g, tmp_path)
    tx.attach(s.host)
    tx.sessrx = s.rx
    s.host.peer = tx
    tx.p3_follow_offset = 'all'
    tx.defer_p3_cs = True
    tx.aim(g, 23)
    return tx, g


@pytest.mark.parametrize('n', [15, 16, 17])
def test_first_tracked_changeover_nominates_measured_carrier(n, tmp_path, capsys):
    audio, origin = held(n)
    got = p3acquire.changeover(audio, offsets=p3acquire.CONTROL_OFFSETS_HZ)
    assert got is not None and got.event.packet == (1, 0x20, b'RMS', True)
    assert -10 < got.offset_hz < -9
    s = _Session(entry_pending=True)
    tx, g = radio(s, tmp_path)
    tx.aim(g, 23+n-15)
    s.rx.new_cycle()
    assert s.rx.control_signal(audio, origin, 1368552+(n-15)*60000) == arq.CS_BREAKIN
    assert s.packets[-1].packet == got.event.packet
    assert not s.rx._p3_offset_fixed
    onair._p3_place_reply(g, tx, 23+n-15)
    tx.emit_pending_cs()
    assert tx.n == 1 and not tx.refused
    assert json.loads((tmp_path/'tx_01.json').read_text())['tx_offset_hz'] == 0
    assert s.rx._p3_offset_fixed
    assert s.rx.p3_receive_offset_hz == pytest.approx(got.offset_hz, abs=.2)
    assert 'follow RX -9.' in capsys.readouterr().out


def mail_link():
    s = _Session(role=arq.IRS)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    session = B2FSession('W9SSJ', target='W6IDS')
    mail = MailClient(session, s.host.arq.on_host_data)
    s.host.app = mail
    mail.link_up()
    # The captured login ends in )\rFF\r, acknowledged before the empty hand-back.
    mail.on_link_data(b'[RMS Trimode-1.3.14.0-B2FHM$]\rW6IDS>\r')
    assert bytes(s.host.arq._outbuf).endswith(b'FF\r')
    s.host.arq._outbuf.clear()
    s.host.arq._buffer_raw = 0
    s.host.buffer(0)
    return s, session


def packet(s, status, payload=b'', *, breakin=False):
    s.host.on_rx_event(rxfront.Event(0., 'packet', 'recorded',
        protocol=spec.Protocol.PACTOR3, packet=(1, status, payload, True),
        breakin=breakin))


def test_w6ids_empty_handback_takes_one_turn_with_fq():
    s, session = mail_link()
    packet(s, 0x40, breakin=True)
    # 0215's advancing invitations are 0x41,42,43,40...; duplicates do not
    # advance the counter and must not shorten the six-invitation allowance.
    for i in range(1, 6):
        packet(s, 0x40 | i % 4)
        packet(s, 0x40 | i % 4)
        assert not s.host.arq._breakin_pending
        assert not session.done
    packet(s, 0x42)
    assert session.done and session.our_fq and not session.remote_ff
    assert bytes(s.host.arq._outbuf) == b'FQ\r'
    assert s.host.arq._breakin_pending and s.host.arq._breakin_armed
    s.host.tick()
    assert s.host.arq.role == arq.ISS
    assert s.host.arq._inflight.breakin
    assert s.host.arq._inflight.payload == b'FQ\r'



def test_recorded_handback_train_queues_fq_on_the_sixth_new_packet():
    trace = json.loads((Path(__file__).parent / 'fixtures' /
                        'p3-mail-reliability-1001.json').read_text())
    rows = trace['empty_handbacks']
    assert len(rows) == 37  # 33 advancing counters and four physical repeats.
    s, session = mail_link()
    packet(s, trace['changeover']['status'], breakin=True)
    close_at = None
    previous = 0
    invitations = 0
    for row in rows:
        _, status, field = row['decoded']
        assert not field and status & 0x40
        if status & 3 != previous:
            invitations += 1
            previous = status & 3
        packet(s, status)
        if session.done and close_at is None:
            close_at = row['phase']
            assert invitations == 6
    assert sum(ours and said == 'FQ' for ours, said, _ in session.exchange) == 1
    assert bytes(s.host.arq._outbuf) == b'FQ\r'
    assert close_at == 11002002
    assert (close_at-trace['changeover']['phase'])/48000 < 11


def test_ve1yz_three_empty_changeovers_then_four_quiet_cycles_wait_for_peer_fq():
    s, session = mail_link()
    for _ in range(3):
        packet(s, 0x00, breakin=True)
    for _ in range(4):
        s.host.tick()
    assert not session.done and not s.host.arq._breakin_pending
    packet(s, 0x41, b'FQ\r')
    assert session.done and not session.our_fq
    assert not s.host.arq._outbuf


@pytest.mark.parametrize('data', [b'FC EM ', b'FC EM ABC 12 12 0\r'])
def test_partial_or_complete_proposal_prevents_empty_handback_close(data):
    s, session = mail_link()
    for i in range(1, 6):
        packet(s, 0x40 | i % 4)
    packet(s, 0x02, data)
    for i in range(3, 15):
        packet(s, 0x40 | i % 4)
    assert not session.done and not session.our_fq
    assert not s.host.arq._outbuf


def test_no_ff_no_empty_handback_close():
    s = _Session(role=arq.IRS)
    session = B2FSession('W9SSJ')
    s.host.app = MailClient(session, s.host.arq.on_host_data)
    for i in range(34):
        packet(s, 0x40 | i % 4)
    assert not session.done and not s.host.arq._breakin_pending


def test_changeover_rendered_before_slot_and_preview_does_not_consume_bytes(tmp_path, monkeypatch):
    s, _ = mail_link()
    s.host.arq.on_host_data(b';FW: W9SSJ\r')
    s.host.arq.on_host_breakin()
    g = irs_grid()
    tx = driver(g, tmp_path)
    tx.attach(s.host)
    tx.sessrx = s.rx
    s.host.peer = tx
    tx.p3_follow_offset = 'all'
    s.rx.p3_receive_offset_hz = -9.6
    tx.aim(g, 138)  # 0215's first GIVEN UP break-in slot.
    before = (bytes(s.host.arq._outbuf), s.host.arq._next_seq,
              s.host.arq._over_offered, s.host.arq._inflight)
    tx.prepare_p3_breakin()
    onair._p3_place_reply(g, tx, 138)
    assert before == (bytes(s.host.arq._outbuf), s.host.arq._next_seq,
                      s.host.arq._over_offered, s.host.arq._inflight)
    assert {key[-1] for key in tx._p3_breakin} == {False, True}
    def late_render(*args, **kwargs):
        pytest.fail('changeover rendered inside the key deadline')
    monkeypatch.setattr(onair.placement, 'changeover_packet', late_render)
    # Exercise the tick's real send method, while isolating placement from the
    # cached render: this pin does not claim the recorded peer accepts sooner.
    monkeypatch.setattr(tx, '_tx', lambda *args, **kwargs: None)
    monkeypatch.setattr(tx, '_place_breakin', lambda: '')
    tx.send_packet(1, b';FW', 0, breakin=True)


def first_stint(tmp_path):
    s = _Session(entry_pending=True)
    s.host.arq.cfg.speed_up = 'hold'
    s.host.arq.cfg.repeat_gear = 0
    g = irs_grid()
    tx = driver(g, tmp_path)
    tx.attach(s.host)
    tx.sessrx = s.rx
    s.host.peer = tx
    tx.p3_follow_offset = 'all'
    tx.defer_p3_cs = True
    tx.live.limit = 200 * onair.FS
    audio, origin = held(15)
    tx.aim(g, 23)
    s.rx.new_cycle()
    s.rx.control_signal(audio, origin, 1368552)
    fixed_before_key = s.rx._p3_offset_fixed
    onair._p3_place_reply(g, tx, 23)
    tx.emit_pending_cs()
    assert tx.n == 1
    assert s.rx._p3_offset_fixed
    # The recorded RMS copies precede the ordinary greeting. The grid
    # and frequency come from the reader; only the held capture windows are
    # supplied by this test. No peer reply is synthesized.
    reads = []
    for n in range(16, 120):
        audio, origin = held(n)
        end = origin + len(audio)
        tx.live.now = tx.live.pos = end
        slot = (end-g.anchor)//60000+1
        tx.aim(g, slot)
        s.rx.new_cycle()
        before = len(s.packets)
        onair._scan_frame(s.rx, audio, origin, tracked_only=True)
        if len(s.packets) > before:
            reads.append(n)
        slot = onair._p3_place_reply(g, tx, slot)
        tx.emit_pending_cs()
    return s, fixed_before_key, reads


def test_first_stint_prekey_reads_and_controls_follow_the_first_frame(tmp_path, capsys):
    _, fixed_before_key, reads = first_stint(tmp_path)
    assert not fixed_before_key
    log = capsys.readouterr().out
    (tmp_path/'replay.log').write_text(log)
    assert 'SL1 pre-key' in log
    assert reads
    ordinary_log = log[log.index('SL1 pre-key'):]
    assert '1 whole cycle(s)' not in ordinary_log
    assert '0 whole cycle(s)' in log
    controls = [meta for meta in (json.loads(p.read_text())
                for p in sorted(tmp_path.glob('tx_*.json')))
                if meta.get('control','').startswith('CS')]
    assert controls[0]['tx_offset_hz'] == 0
    assert len(controls) > 1
    assert all(meta['tx_offset_hz'] == pytest.approx(-9.6, abs=.25)
               for meta in controls[1:])


def test_first_stint_delivery_is_preserved(tmp_path):
    s, _, _ = first_stint(tmp_path)
    delivered = []
    for ev in s.packets:
        row = [ev.packet[1], ev.packet[2].hex()]
        if not delivered or delivered[-1] != row:
            delivered.append(row)
    assert delivered == FIXTURE['first_stint_delivery']['packets']


def test_empty_handback_fq_drains_before_confirmed_qrt():
    from hfmodem.tests.shrike.test_p3_terminal_confirm_arq import Peer, wire
    s, session = mail_link()
    peer = Peer()
    s.host.peer = peer
    s.host.p3_qrt_confirm = True
    packet(s, 0x40, breakin=True)
    for i in range(1, 7):
        packet(s, 0x40 | i % 4)
    s.host.tick()
    assert session.done and s.host._txbuf == 3
    assert s.host.arq._inflight.payload == b'FQ\r'
    assert not s.host.arq._qrt_pending
    wire(s.host, s.host.arq.tx_seq & 1)
    assert s.host._txbuf == 0
    # This is the main mail loop's drained-buffer close condition.
    s.host.arq.on_host_disconnect()
    s.host.tick()
    assert s.host.arq.said_goodbye
    assert s.host.arq._inflight.status & spec.STATUS_QRT
    previous = s.host.arq.tx_seq & 1
    wire(s.host, previous)
    assert s.host.arq.terminal_confirm_pending
    s.host.tick()
    assert peer.markers_emitted == 1
    wire(s.host, previous ^ 1)
    assert s.host.arq.goodbye_acked
    # This session fixture listens again after a completed connection.
    assert s.host.arq.state == arq.State.LISTENING


def test_unacknowledged_ff_does_not_start_empty_handback_close():
    s, session = mail_link()
    s.host.arq._buffer_raw = 3  # FF has left the queue but has not been ACKed.
    s.host.buffer(3)
    for i in range(1, 10):
        packet(s, 0x40 | i % 4)
    assert not session.done and not s.host.arq._breakin_pending


@pytest.mark.parametrize('repeat', [False, True])
def test_tracked_packet_reader_nominates_after_control(tmp_path, monkeypatch, repeat):
    audio, origin = held(16)
    s = _Session(role=arq.IRS)
    tx, _ = radio(s, tmp_path)
    at = 1428552-origin
    ev = s.rx.sync.control_signal_at(audio, at, details=True)
    assert ev is not None and ev.packet == (1, 0x20, b'RMS', True)
    s.rx._tracked_only = True
    s.rx.sync.packet_at = at+4320
    s.rx._p3_repeat_changeover = repeat
    if not repeat:
        # Force the other tracked return using a real CRC-decoded frame;
        # the frequency nomination itself still reads the recorded waveform.
        monkeypatch.setattr(s.rx.sync, 'packet', lambda *a, **kw: ev)
    got = s.rx._read_p3_packet(audio, origin)
    assert got is not None and got[0].packet == ev.packet
    assert not s.rx._p3_offset_fixed
    tx.send_cs(arq.CS_ACK)
    tx.emit_pending_cs()
    assert tx.n == 1 and not tx.refused
    assert s.rx._p3_offset_fixed
    assert s.rx.p3_receive_offset_hz == pytest.approx(-9.4, abs=.1)


@pytest.mark.parametrize('search_ms', [20, 28])
@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('reader', ['control', 'repeat', 'packet'])
@pytest.mark.parametrize('n,hz', [(15, -9.6), (16, -9.4), (17, -9.6)])
def test_tracked_nomination_cost_is_after_key(
        tmp_path, monkeypatch, reader, deferred, search_ms, n, hz):
    audio, origin = held(n)
    s = _Session(entry_pending=reader == 'control',
                 role=arq.ISS if reader == 'control' else arq.IRS)
    tx, g = radio(s, tmp_path)
    tx.aim(g, 23+n-15)
    tx.defer_p3_cs = deferred
    s.rx.new_cycle()
    phase = 1368552+(n-15)*60000
    at = phase-origin
    ev = s.rx.sync.control_signal_at(audio, at, details=True)
    assert ev is not None and ev.packet == (1, 0x20, b'RMS', True)
    head = at if reader == 'repeat' else ev.start
    g.note_p3_packet(origin+head, round(onair.placement.PACKET_S*onair.FS),
                    60000, swapped=ev.carrier_swapped,
                    identity=(True, *ev.packet[:3]))
    slot = onair._p3_place_reply(g, tx, 23+n-15)
    due = tx.key_instant(g, slot)
    decode_n = round(.0105*onair.FS)  # 0215 hold 15's logged control/body cost.
    search_n = round(search_ms/1000*onair.FS)
    # 53 ms before the boundary leaves 2.5 ms after decode and the 40 ms
    # settle. Either search cost would exceed the 5 ms clamp forgiveness.
    tx.live.now = tx.live.pos = due-round(.053*onair.FS)
    trace = []
    decode = s.rx.sync.control_signal_at
    def charged_decode(*args, **kwargs):
        result = decode(*args, **kwargs)
        tx.live.spend(decode_n)
        trace.append(('decode', tx.live.now))
        return result
    monkeypatch.setattr(s.rx.sync, 'control_signal_at', charged_decode)
    acquire = p3acquire.changeover
    def charged_search(*args, **kwargs):
        trace.append(('search', tx.live.now))
        tx.live.spend(search_n)
        return acquire(*args, **kwargs)
    monkeypatch.setattr(p3acquire, 'changeover', charged_search)
    transmit = tx.live.transmit
    def keyed(*args, **kwargs):
        if tx.n == 1:
            assert [n for _, n in tx.live.spent] == [decode_n]
        trace.append(('key', tx.live.now))
        return transmit(*args, **kwargs)
    monkeypatch.setattr(tx.live, 'transmit', keyed)
    if reader == 'control':
        s.rx.control_signal(audio, origin, phase)
    else:
        s.rx.sync.packet_at = at+4320
        s.rx._p3_repeat_changeover = reader == 'repeat'
        if reader == 'packet':
            def packet_decode(*args, **kwargs):
                tx.live.spend(decode_n)
                trace.append(('decode', tx.live.now))
                return ev
            monkeypatch.setattr(s.rx.sync, 'packet', packet_decode)
        onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    if deferred:
        assert [name for name, _ in trace] == ['decode']
        tx.emit_pending_cs()
    assert [name for name, _ in trace] == ['decode', 'key', 'search']
    assert trace[2][1] >= tx.tx_end
    assert tx.live.now == trace[2][1]+search_n
    assert tx.n == 1 and not tx.refused and tx.slot == slot
    assert tx.tx_audio_start == due
    assert json.loads((tmp_path/'tx_01.json').read_text())['tx_offset_hz'] == 0
    assert s.rx.p3_receive_offset_hz == pytest.approx(hz, abs=.2)

    tx.aim(g, slot + 1)
    next_due = tx.key_instant(g, slot + 1)
    assert tx.live.now + round((tx.settle + onair.P3_DECODE_RESERVE_S)*onair.FS) < next_due
    tx.send_cs(arq.CS_ACK)
    if deferred:
        tx.emit_pending_cs()
    assert tx.n == 2 and not tx.refused and tx.slot == slot + 1
    assert tx.tx_audio_start == next_due
    assert [name for name, _ in trace] == ['decode', 'key', 'search', 'key']
    assert json.loads((tmp_path/'tx_02.json').read_text())['tx_offset_hz'] == pytest.approx(hz, abs=.2)


def test_failed_nomination_retries_only_after_the_next_key(tmp_path, monkeypatch):
    s = _Session(entry_pending=True)
    tx, g = radio(s, tmp_path)
    acquire = p3acquire.changeover
    searched = []
    def miss_once(*args, **kwargs):
        searched.append(tx.n)
        assert tx.n == len(searched)
        assert tx.live.now >= tx.tx_end
        tx.live.spend(round(.028*onair.FS))
        return None if len(searched) == 1 else acquire(*args, **kwargs)
    monkeypatch.setattr(p3acquire, 'changeover', miss_once)
    for n in (15, 16):
        audio, origin = held(n)
        s.rx.new_cycle()
        tx.aim(g, 23+n-15)
        s.rx.control_signal(audio, origin, 1368552+(n-15)*60000)
        assert len(searched) == n-15
        onair._p3_place_reply(g, tx, 23+n-15)
        tx.emit_pending_cs()
        assert tx.n == n-14 and not tx.refused
        assert s.rx._p3_offset_fixed == (n == 16)
    assert searched == [1, 2]
    assert s.rx.p3_receive_offset_hz == pytest.approx(-9.4, abs=.2)


def test_refused_control_does_not_spend_nomination_cost(tmp_path, monkeypatch):
    s = _Session(entry_pending=True)
    tx, g = radio(s, tmp_path)
    audio, origin = held(15)
    s.rx.new_cycle()
    s.rx.control_signal(audio, origin, 1368552)
    assert not s.rx._p3_offset_fixed
    onair._p3_place_reply(g, tx, 23)
    def refuse(*args, **kwargs):
        tx.refused = True
    with monkeypatch.context() as refused:
        refused.setattr(tx, '_tx', refuse)
        refused.setattr(p3acquire, 'changeover',
                        lambda *a, **kw: pytest.fail('search after refused key'))
        tx.emit_pending_cs()
    assert tx.n == 0 and not s.rx._p3_offset_fixed
    assert tx._pending_p3_cs is not None
    tx.emit_pending_cs()
    assert tx.n == 1 and not tx.refused
    assert s.rx.p3_receive_offset_hz == pytest.approx(-9.6, abs=.2)


@pytest.mark.parametrize('reader,n,hz', [('control', 15, -9.6),
                                        ('repeat', 16, -9.4),
                                        ('packet', 17, -9.6)])
@pytest.mark.parametrize('retry', ['deferred', 'refused', 'failed'])
def test_deferred_nomination_owns_audio_across_cycle(
        tmp_path, monkeypatch, reader, n, hz, retry):
    audio, origin = held(n)
    s = _Session(entry_pending=reader == 'control',
                 role=arq.ISS if reader == 'control' else arq.IRS)
    tx, g = radio(s, tmp_path)
    tx.aim(g, 23+n-15)
    s.rx.new_cycle()
    at = 1368552+(n-15)*60000
    if reader == 'control':
        s.rx.control_signal(audio, origin, at)
    else:
        ev = s.rx.sync.control_signal_at(audio, at-origin, details=True)
        s.rx.sync.packet_at = at-origin+4320
        s.rx._p3_repeat_changeover = reader == 'repeat'
        if reader == 'packet':
            monkeypatch.setattr(s.rx.sync, 'packet', lambda *a, **kw: ev)
        onair._scan_frame(s.rx, audio, origin, tracked_only=True)
    assert not s.rx._p3_offset_fixed
    assert tx._pending_p3_cs is not None
    slot = onair._p3_place_reply(g, tx, tx.slot)
    audio[:] = 0  # The capture collector reuses its storage before the key.
    s.rx.new_cycle()
    acquire = p3acquire.changeover
    searches = []
    def search(*args, **kwargs):
        searches.append(tx.n)
        assert tx.live.now >= tx.tx_end
        tx.live.spend(round(.028*onair.FS))
        return None if retry == 'failed' and len(searches) == 1 else acquire(*args, **kwargs)
    monkeypatch.setattr(p3acquire, 'changeover', search)
    if retry == 'refused':
        def refuse(*args, **kwargs):
            tx.refused = True
        with monkeypatch.context() as refused:
            refused.setattr(tx, '_tx', refuse)
            tx.emit_pending_cs()
        assert not searches and tx.n == 0
        s.rx.new_cycle()
    tx.emit_pending_cs()
    assert searches == [1]
    assert json.loads((tmp_path/'tx_01.json').read_text())['tx_offset_hz'] == 0
    if retry == 'failed':
        assert not s.rx._p3_offset_fixed
        s.rx.new_cycle()
        tx.aim(g, slot+1)
        tx.send_cs(arq.CS_ACK)
        tx.emit_pending_cs()
        assert searches == [1, 2]
    assert not tx.refused
    assert s.rx._p3_offset_fixed
    assert s.rx.p3_receive_offset_hz == pytest.approx(hz, abs=.2)


def test_postkey_nomination_cannot_spend_the_next_control_reserve(tmp_path, monkeypatch):
    s = _Session(entry_pending=True)
    tx, g = radio(s, tmp_path)
    audio, origin = held(15)
    s.rx.control_signal(audio, origin, 1368552)
    slot = onair._p3_place_reply(g, tx, 23)
    due = tx.key_instant(g, slot+1)
    transmit = tx._tx
    def delayed_return(*args, **kwargs):
        transmit(*args, **kwargs)
        # Independent post-key wall-clock jump leaves time for the next ACK,
        # but not for a 20-28 ms search and its settle/decoder reserve.
        tx.live.now = due-round(.055*onair.FS)
    with monkeypatch.context() as delayed:
        delayed.setattr(tx, '_tx', delayed_return)
        delayed.setattr(p3acquire, 'changeover',
                        lambda *a, **kw: pytest.fail('search stole next key reserve'))
        tx.emit_pending_cs()
    assert tx.n == 1 and not s.rx._p3_offset_fixed
    audio[:] = 0
    s.rx.new_cycle()
    tx.aim(g, slot+1)
    tx.send_cs(arq.CS_ACK)
    tx.emit_pending_cs()
    assert tx.n == 2 and not tx.refused and tx.tx_audio_start == due
    assert s.rx.p3_receive_offset_hz == pytest.approx(-9.6, abs=.2)
    assert json.loads((tmp_path/'tx_02.json').read_text())['tx_offset_hz'] == 0


@pytest.mark.parametrize('change', ['role', 'call', 'down', 'protocol', 'packet',
                                    'same-packet-data'])
def test_deferred_nomination_expires_with_its_context(tmp_path, monkeypatch, change):
    s = _Session(entry_pending=True)
    tx, g = radio(s, tmp_path)
    audio, origin = held(15)
    s.rx.control_signal(audio, origin, 1368552)
    if change == 'role':
        s.host.arq.role = arq.ISS
    elif change == 'call':
        s.host.arq.dxcall = 'K4MSU'
    elif change == 'down':
        s.host.arq.state = arq.State.LISTENING
    elif change == 'protocol':
        s.host.protocol = spec.Protocol.PACTOR1
    else:
        field = s.packets[-1].packet if change == 'same-packet-data' else (1, 0x01, b'new', True)
        ev = replace(s.packets[-1], packet=field, breakin=False)
        s.rx._note_p3_frame(ev, 1428552)
    s.rx.new_cycle()
    monkeypatch.setattr(p3acquire, 'changeover',
                        lambda *a, **kw: pytest.fail('search for expired nomination'))
    onair._p3_place_reply(g, tx, 23)
    tx.emit_pending_cs()
    assert not s.rx._p3_offset_fixed


def test_k4msu_0321_tracked_control_preserves_delivery(tmp_path):
    s = _Session(entry_pending=True)
    tx, g = radio(s, tmp_path)
    for n in (14, 15):
        audio, origin = held(n, 'onair-1001-0321')
        s.rx.new_cycle()
        tx.aim(g, n+8)
        s.rx.control_signal(audio, origin, 1308708+(n-14)*60000)
        assert s.packets[-1].packet == (1, 0, b'RMS', True)
        assert not s.rx._p3_offset_fixed
        onair._p3_place_reply(g, tx, n+8)
        tx.emit_pending_cs()
        assert tx.n == n-13 and not tx.refused
        # Hold 14's tracked CRC delivers RMS even though acquisition fails;
        # hold 15's repeat supplies the first usable carrier measurement.
        assert s.rx._p3_offset_fixed == (n == 15)
    assert s.rx.p3_receive_offset_hz == pytest.approx(-14.8, abs=.2)
    assert json.loads((tmp_path/'tx_01.json').read_text())['tx_offset_hz'] == 0
