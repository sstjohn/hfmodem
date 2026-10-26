# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import json
from pathlib import Path

import numpy as np
import pytest

from hfmodem.kestrel.arq.modem import AudioChannel, KestrelModem
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.host.modem_core import LoopbackModem
from hfmodem.kestrel.host.server import HostSession, SessionState
from hfmodem.kestrel.host.vara_modem import VaraHostModem
from hfmodem.kestrel.rx import varahf500 as RX
from hfmodem.kestrel.tx import varahf500_tx as TX
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as MK
from . import test_idle_poll as idle
from .test_compression import IO

GOLDEN = json.loads((Path(__file__).parent / 'fixtures/chat-p0-host.json').read_text())
SN_PAIRS = json.loads((Path(__file__).parent / 'fixtures/chat-sn-pairs.json').read_text())


class Wire:
    def __init__(self): self.data = bytearray()
    def sendall(self, blob): self.data.extend(blob)


def host(bandwidth='500'):
    cmd, data = Wire(), Wire()
    modem = VaraHostModem(IO(), bandwidth=bandwidth)
    session = HostSession(cmd, data, modem)
    modem.start(session)
    modem.poll()
    cmd.data.clear()
    return session, modem, cmd, data


@pytest.mark.parametrize('record', GOLDEN['listen'])
def test_p0_listen_transcripts(record):
    session, modem, cmd, _ = host()
    for command in record['commands']:
        session._dispatch(command)
        modem.poll()
    assert cmd.data == ''.join(r + '\r' for r in record['replies']).encode()
    last_listen = 'OFF'
    for command in record['commands']:
        if command.startswith('LISTEN '): last_listen = command.split()[1]
        if command == 'CHAT ON' and last_listen == 'OFF': last_listen = 'ON'
        if command == 'CHAT OFF' and last_listen == 'CQ': last_listen = 'ON'
    assert session.listening == modem._listen == (last_listen != 'OFF')
    cq = False
    for command in record['commands']:
        if command.startswith('LISTEN '): cq = command == 'LISTEN CQ'
        if command.startswith('CHAT '): cq = command == 'CHAT ON'
    assert session.listen_cq == cq
    assert modem.station.chat == session.chat


@pytest.mark.parametrize('commands,reported', [
    (['LISTEN ON'], False), (['LISTEN CQ'], True), (['CHAT ON'], True),
    (['CHAT ON', 'CHAT OFF'], False), (['CHAT ON', 'LISTEN ON'], False),
    (['CHAT ON', 'LISTEN OFF'], False),
])
def test_cq_notifications_follow_listen_mode(commands, reported):
    session, modem, cmd, _ = host()
    for command in commands:
        session._dispatch(command)
        modem.poll()
    cmd.data.clear()
    session.modem_cqframe('W9SSJ-9', '2300', 18.6)
    assert cmd.data == (b'BITRATE (1)  18 bps RX\rSN 18.6\rCQFRAME W9SSJ-9 2300\r' if reported else b'')


@pytest.mark.parametrize('record', GOLDEN['clean'])
@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('compression', ['OFF', 'TEXT', 'FILES'])
def test_p0_clean_replies_and_queue_effects(record, chat, compression):
    session, modem, cmd, _ = host()
    hs = modem.station
    hs.state = A.VaraState.DISCONNECTED if record['t'] < 1 else A.VaraState.CONNECTED
    if record['t'] >= 1:
        session.state = SessionState.CONNECTED
    hs.role = 'initiator'
    hs._start_compression(compression)
    hs.chat = chat
    hs._answering = True
    hs.turn = A._TURN_OURS if 'FAILED' in record['reply'] else A._TURN_PEER
    if 'BUFFEREMPTY' not in record['reply']:
        modem.transmit(b'queued text')
        modem.poll()
    before = tuple(hs._txq)
    cmd.data.clear()
    session._dispatch('CLEANTXBUFFER')
    # Stock answers at once; only a clear that succeeds waits for the owner.
    assert (cmd.data == b'') == (record['reply'] == 'CLEANTXBUFFER OK')
    modem.poll()
    expected = record['reply'] + '\r'
    if record['reply'] == 'CLEANTXBUFFER OK':
        expected += 'BUFFER 0\r'
    assert cmd.data == expected.encode()
    assert tuple(hs._txq) == (before if 'FAILED' in record['reply'] else ())
    queued = len(b'queued text') if 'FAILED' in record['reply'] else 0
    assert hs.payload_ledger.pending == queued


def test_clean_unsupported_modem_replies_wrong():
    modem = KestrelModem(AudioChannel().a)
    cmd, data = Wire(), Wire()
    session = HostSession(cmd, data, modem)
    modem._observer = session
    session._dispatch('CLEANTXBUFFER')
    session._dispatch('CHAT ON')
    assert cmd.data == b'WRONG\rWRONG\r'


def test_loopback_clean_reports_empty_buffer():
    modem = LoopbackModem()
    cmd, data = Wire(), Wire()
    session = HostSession(cmd, data, modem)
    modem._observer = session
    modem.transmit(b'queued text')
    assert cmd.data == b''
    session._dispatch('CLEANTXBUFFER')
    assert cmd.data == b'CLEANTXBUFFER OK\r'
    assert not modem._preconnect and modem._buffer == 0


def test_clean_cannot_discard_unacknowledged_over():
    _, modem, _, _ = host()
    hs = modem.station
    hs.state, hs.role, hs.turn = A.VaraState.CONNECTED, 'initiator', A._TURN_OURS
    hs.send(b'outstanding')
    assert hs._tx_pending is not None
    pending = hs._tx_pending
    hs.turn = A._TURN_PEER
    assert hs.clean_tx_buffer() == 'FAILED '
    assert hs._tx_pending == pending


def test_clean_cannot_clear_bytes_waiting_for_a_link():
    """Stock answers FAILED on a disconnected modem with bytes queued (bench 3
    r3c gate 3); the bytes stay for the next link."""
    _, modem, cmd, _ = host()
    modem.transmit(b'send this')
    modem.clean_tx_buffer()
    modem.set_listen(True)
    modem.poll()
    hs = modem.station
    assert hs.preconnect_data == (b'send this',)
    assert hs.payload_ledger.pending == len(b'send this')
    assert modem.discarded_total == 0
    assert b'CLEANTXBUFFER FAILED \r' in cmd.data


def test_no_buffer_line_while_no_link_is_up():
    """Bench 2 4b gate3-stock-ref, stock A: a clear answered BUFFEREMPTY, 26
    bytes written while disconnected and two clears answered FAILED, with no
    BUFFER line; its first came after CONNECTED, the drain's BUFFER 0."""
    cmd, data = Wire(), Wire()
    modem = VaraHostModem(IO(), bandwidth='500')
    session = HostSession(cmd, data, modem)
    modem.start(session)
    session._dispatch('CLEANTXBUFFER')
    modem.poll()
    modem.transmit(b'queued while disconnected\r')
    modem.poll()
    for _ in range(2):
        session._dispatch('CLEANTXBUFFER')
        modem.poll()
    assert cmd.data.split(b'\r') == [b'CLEANTXBUFFER BUFFEREMPTY', b'REGISTERED HFMODEM',
                                     b'CLEANTXBUFFER FAILED ', b'CLEANTXBUFFER FAILED ', b'']
    cmd.data.clear()
    hs = modem.station
    hs.state, hs.role, hs.turn = A.VaraState.CONNECTED, 'initiator', A._TURN_PEER
    hs.io.connected('W9SSJ', 'WI9YZA', '500')
    hs.retire_payload()
    modem.poll()
    assert cmd.data == b'CONNECTED W9SSJ WI9YZA 500\rBUFFER 0\r'


def test_chat_survives_station_replacement():
    session, modem, _, _ = host()
    session._dispatch('CHAT ON')
    modem.poll()
    old = modem.station
    modem.abort()
    modem.poll()
    assert modem.station is not old and modem.station.chat
    session._dispatch('CHAT OFF')
    modem.poll()
    assert not modem.station.chat and modem._listen


def test_chat_off_host_callback_bytes_match_baseline_except_sn_and_bitrate_spacing():
    session, modem, cmd, data = host()
    baseline = GOLDEN['chat_off_baseline']
    for command in baseline['commands']:
        session._dispatch(command)
        modem.poll()
    session.modem_pending()
    session.modem_connected('W9SSJ', 'WI9YZA', '500')
    session.modem_bitrate(4, 512, False)
    session.modem_snr(-9.3)
    session.modem_buffer(8)
    session.modem_ptt(True)
    session.modem_data_received(b'hello\x00\xff')
    session.modem_ptt(False)
    session.modem_buffer(0)
    session.modem_disconnected()
    lines = cmd.data.split(b'\r')
    assert b'\r'.join(x.replace(b')  ', b') ') if x.startswith(b'BITRATE ') else x
                      for x in lines if not x.startswith(b'SN ')) == bytes.fromhex(
        baseline['command_bytes_hex'])
    assert lines.count(b'SN -9.3') == 1
    assert data.data == bytes.fromhex(baseline['data_bytes_hex'])


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('chat', [False, True])
def test_disconnect_keys_the_roles_frame(role, chat):
    hs, io = idle._station('500')
    hs.role, hs.chat, hs.turn = role, chat, A._TURN_OURS
    hs.disconnect()
    kind = F.SESSION_DISCONNECT_REQ if role == 'initiator' else F.SESSION_DISCONNECT_REQ_RESPONDER
    np.testing.assert_array_equal(io.sent[-1], MK.synth_burst(idle.CALLED, F.for_bw(kind, '500')))
    assert hs.state == A.VaraState.DISCONNECTED


def test_chat_polls_match_p0_cadence_and_hold_twelve_minutes(monkeypatch):
    monkeypatch.setattr(A, 'time', idle._Clock)
    idle._Clock.t = 1000.
    hs, io = idle._station('500')
    hs.chat, hs.turn = True, A._TURN_OURS
    hs._hold_idle()
    for _ in range(60):
        hs._took_idle_response()
        idle._Clock.t += hs.idle_poll_due_in()
        hs.idle_keepalive()
    assert hs.state == A.VaraState.CONNECTED
    assert io.at[-1] - io.at[0] >= 720
    assert all(idle._is_poll(hs, x) for x in io.sent)
    measured = np.median(np.diff(GOLDEN['idle']['poll_starts']))
    assert io.at[1] - io.at[0] == pytest.approx(11.15)
    assert np.allclose(np.diff(io.at)[1:], measured, atol=.08)
    hs.chat = False
    idle._Clock.t += hs.idle_poll_due_in()
    hs.idle_keepalive()
    assert hs.state == A.VaraState.DISCONNECTED


@pytest.mark.parametrize('turn', [A._TURN_OURS, A._TURN_PEER])
def test_chat_still_closes_when_peer_stops_answering(monkeypatch, turn):
    monkeypatch.setattr(A, 'time', idle._Clock)
    idle._Clock.t = 1000.
    hs, _ = idle._station('500')
    hs.chat, hs.turn = True, turn
    if turn == A._TURN_OURS: hs._hold_idle()
    for _ in range(A._MAX_WITHOUT_PROGRESS + 2):
        idle._Clock.t += 12.06
        hs.idle_keepalive()
    assert hs.state == A.VaraState.DISCONNECTED


def test_chat_answers_peer_polls_with_existing_keepalive(monkeypatch):
    monkeypatch.setattr(A, 'time', idle._Clock)
    idle._Clock.t = 1000.
    hs, io = idle._delivered_and_drained()
    hs.chat = True
    idle._peer_polls(hs, io, 8)
    assert idle._keepalives(io, F.SESSION_KEEPALIVE_B) == 8
    assert hs.state == A.VaraState.CONNECTED


@pytest.mark.parametrize('rms,expected', [(0, 20.0), (.01, 15.3), (.03, 12.5),
    (.06, 6.9), (.10, 4.9), (.15, -.6), (.20, -2.2), (.30, -6.55), (.45, -9.3)])
def test_500_quality_matches_p0_noise_ladder(rms, expected):
    from scipy.signal import butter, sosfilt
    x = TX.synth_burst(TX.build_frame(bytes(range(43)), 0x81), onset=3000)
    x = x[5000:-5000]
    x = x * (.237 / np.sqrt(np.mean(x*x)))
    noise = sosfilt(butter(8, 2900, fs=RX.FS, output='sos'),
                    np.random.default_rng(1018).normal(size=len(x)))
    noise *= rms / np.sqrt(np.mean(noise*noise))
    assert RX.quality(x + noise) == pytest.approx(expected, abs=2)


@pytest.mark.parametrize('raw,stock', [
    (46.7, 23.0), (46.6, 22.9), (45., 21.2), (45., 21.2), (19.9, 13.8),
    (9.5, 4.9), (3.5, -3.4), (-.1, -3.9),
    (45., 20.5), (45., 19.5), (27.5, 15.3), (18., 12.5),
    (11.9, 6.9), (7.5, 4.1), (7.5, 5.7), (4., -.6),
    (1.5, -2.2), (-2., -6.2), (-2., -6.9), (-5.6, -9.3),
])
def test_500_sn_recorded_p0_and_gate_pairs(raw, stock):
    assert RX.stock_sn(raw) == pytest.approx(stock, abs=2)


@pytest.mark.parametrize('pair', [p for p in SN_PAIRS['pairs'] if p['level'] not in (7, 12)])
def test_tone_ofdm_sn_recorded_pairs(pair):
    assert phy.host_sn(pair['bw'], pair['level'], pair['raw']) == pytest.approx(
        pair['stock'], abs=2)


@pytest.mark.parametrize('bw,level', [
    (bw, level) for bw in ('2300', '2750') for level in range(1, phy.TOP[bw] + 1)])
@pytest.mark.parametrize('raw', [-5., 39.7, 61.2266090186779, 63.01, 80.])
def test_uncalibrated_host_sn_preserves_decoder_scale(bw, level, raw):
    assert phy.host_sn(bw, level, raw) == raw


@pytest.mark.parametrize('level,raw,stock', [(12, 58.2, (20.3, 31.5)), (9, 62.8, (29.0, 30.5)),
                                           (7, 63.01, (30.1, 31.9))])
def test_500_upper_level_sn_lands_on_stocks_scale(level, raw, stock):
    """Bench 3 gate 3 reported 58.2 at L12 and 62.8 at L9; stock reads 20.3-31.5
    at L12 (median 28.3), 29.7 at L9 and 30.1-31.9 at L7 on the same cables."""
    assert stock[0] <= phy.host_sn('500', level, raw) <= stock[1]


@pytest.mark.parametrize('raw', [59.54667484852187, 59.55, 63.01, 80.])
def test_500_l13_host_sn_stays_on_plateau_above_top_knot(raw):
    assert phy.host_sn('500', 13, raw) == 27.7


@pytest.mark.parametrize('raw', [39.7, None])
@pytest.mark.parametrize('level,bps', [(1, 18), (4, 88), (5, 177), (6, 270),
    (7, 441), (8, 588), (10, 884), (11, 1060), (12, 1286), (13, 1543)])
def test_recorded_rx_bitrate_bytes_and_raw_rate_quality(monkeypatch, level, bps, raw):
    _, modem, cmd, _ = host()
    hs = modem.station
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'WI9YZA'
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._start_compression('OFF')
    body = phy.vara_body(b'hello', hs.caller, body_len=phy.body_size('500', 4))

    def received(samples):
        hs._rx_read = (level, raw)
        return [body]

    monkeypatch.setattr(hs, '_peer_data_over', received)
    assert hs._answer_data_over([])
    if raw is None:
        assert b'SN ' not in cmd.data
        assert b'BITRATE ' not in cmd.data
        assert hs._rx_read == (level, None)
        return
    expected = f'SN {phy.host_sn("500", level, raw):.1f}\rBITRATE ({level})  {bps} bps RX\r'
    assert cmd.data.startswith(expected.encode())
    assert cmd.data.count(b'BITRATE ') == 1
    assert hs._rx_read == (level, raw)


def test_decode_level_keeps_raw_tone_quality(monkeypatch):
    from types import SimpleNamespace
    raw = 39.7
    monkeypatch.setattr(phy._tone, 'decode_burst', lambda *a: SimpleNamespace(
        level=6, frames=(b'body\x00\x00',), quality=SimpleNamespace(snr_db=raw)))
    read = phy.decode_level(np.zeros(300000), '500', 6)
    assert read.quality == raw
    assert phy.host_sn('500', read.level, read.quality) != raw


def test_gate_wideband_rx_bitrate_and_uncalibrated_sn(monkeypatch):
    _, modem, cmd, _ = host('2300')
    hs = modem.station
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'WI9YZA'
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._start_compression('OFF')
    hs._rx_read = (6, 39.7)
    body = phy.vara_body(b'hello', hs.caller, body_len=phy.body_size('2300', 4))
    monkeypatch.setattr(hs, '_peer_data_over', lambda samples: [body])
    assert hs._answer_data_over([])
    assert cmd.data.startswith(b'SN 39.7\rBITRATE (6)  363 bps RX\r')


def test_500_quality_reaches_host_once_per_received_over():
    session, modem, cmd, data = host()
    hs = modem.station
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'WI9YZA'
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._start_compression('OFF')
    frame = TX.build_frame(bytes(range(43)), 0x81)
    x = TX.synth_burst(frame, onset=3000)
    assert hs._answer_data_over(x)
    sn = [line for line in cmd.data.split(b'\r') if line.startswith(b'SN ')]
    assert len(sn) == 1 and float(sn[0][3:]) == pytest.approx(20, abs=2)
    lines = cmd.data.split(b'\r')
    assert lines[lines.index(sn[0]) + 1] == b'BITRATE (4)  88 bps RX'
    assert len(data.data) == 43


def test_500_link_setup_reports_quality_but_controls_do_not():
    _, modem, cmd, _ = host()
    hs = modem.station
    hs.role, hs.called = 'responder', 'WI9YZA'
    hs.state = A.VaraState.CONNECTING
    x = np.concatenate([A.OF.link_setup_tx('W9SSJ', bw='500'), np.zeros(4800)])
    assert hs._rx_link_setup(x)
    lines = cmd.data.split(b'\r')
    assert sum(line.startswith(b'SN ') for line in lines) == 1
    assert b'CONNECTED W9SSJ WI9YZA 500' in lines
    assert not any(line.startswith(b'BITRATE ') for line in lines)
    cmd.data.clear()
    hs.role = 'initiator'
    hs.on_rx_audio(MK.synth_burst(hs.called, F.for_bw(F.SESSION_IDLE_RESPONSE, '500')))
    assert b'SN ' not in cmd.data
    assert b'BITRATE ' not in cmd.data


def test_two_500_frames_in_one_over_report_sn_once():
    _, modem, cmd, data = host()
    hs = modem.station
    hs.role, hs.caller, hs.called = 'initiator', 'W9SSJ', 'WI9YZA'
    hs.state, hs.turn = A.VaraState.CONNECTED, A._TURN_PEER
    hs._start_compression('OFF')
    frames = [TX.build_frame(bytes([n]) * 43, 0x81) for n in (3, 4)]
    hs._peer_over_state, hs._peer_paired_500 = (4, 0x81), True
    assert hs._answer_data_over(TX.synth_burst(frames, onset=3000))
    assert sum(line.startswith(b'SN ') for line in cmd.data.split(b'\r')) == 1
    assert cmd.data.count(b'BITRATE (4)  88 bps RX\r') == 1
    assert data.data == bytes([3]) * 43 + bytes([4]) * 43


def test_data_delivery_retains_blocking_backpressure():
    import threading
    session, modem, _, _ = host()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class BlockedWire:
        def sendall(self, blob):
            entered.set()
            assert release.wait(5)

    session._data = BlockedWire()

    def deliver():
        modem.station.io.data(b'held by host')
        finished.set()

    worker = threading.Thread(target=deliver)
    worker.start()
    try:
        assert entered.wait(5)
        assert not finished.is_set()
    finally:
        release.set()
        worker.join(5)
    assert finished.is_set()


def test_an_unasked_queue_clears_but_not_peer_reception():
    _, modem, _, _ = host()
    hs = modem.station
    hs.state, hs.role, hs.turn = A.VaraState.CONNECTED, 'initiator', A._TURN_PEER
    hs._answering = True
    hs.send(b'discard me')
    hs._ask_owed = True
    hs._tx_recovery_levels = [0]
    assert hs.clean_tx_buffer() == 'OK'
    assert not hs.data_pending and not hs._ask_owed
    assert not hs._tx_recovery_levels
    assert hs.turn == A._TURN_PEER
    assert hs.state == A.VaraState.CONNECTED


def test_buffer_clear_refuses_a_turn_we_hold_even_while_asking():
    _, modem, _, _ = host()
    hs = modem.station
    hs.state, hs.role, hs.turn = A.VaraState.CONNECTED, 'initiator', A._TURN_OURS
    hs._answering = True
    hs.send(b'keep me')
    hs.turn, hs._poll_at = A._TURN_ASKED, 1.0
    before = tuple(hs._txq)
    assert hs.clean_tx_buffer() == 'FAILED '
    assert tuple(hs._txq) == before


@pytest.mark.parametrize('level', [1, 2, 3, 4])
@pytest.mark.parametrize('sn', [-9.3, 20.0])
def test_host_sn_does_not_change_500_rate_selection(level, sn):
    hs, _ = idle._station('500')
    hs._link_high_speed = True
    hs._rx_read = (level, None)
    previous = hs._irs_step()
    hs._rx_read = (level, sn)
    assert hs._irs_step() == previous


def test_called_close_waits_for_successful_transmission(monkeypatch):
    session, modem, cmd, _ = host()
    hs = modem.station
    hs.role, hs.state, hs.turn = 'responder', A.VaraState.CONNECTED, A._TURN_OURS
    modem._connected = True
    sent = []
    monkeypatch.setattr(hs, '_send_burst', lambda kind: sent.append(kind))
    session._dispatch('DISCONNECT')
    modem.poll()
    assert hs.state == A.VaraState.DISCONNECTING and modem._closing
    assert b'DISCONNECTED' not in cmd.data
    assert sent == [F.SESSION_DISCONNECT_REQ_RESPONDER]
    monkeypatch.setattr(hs, '_send_burst', lambda kind: True)
    modem.poll()
    assert hs.state == A.VaraState.DISCONNECTED and not modem._closing
    assert cmd.data.count(b'DISCONNECTED\r') == 1
