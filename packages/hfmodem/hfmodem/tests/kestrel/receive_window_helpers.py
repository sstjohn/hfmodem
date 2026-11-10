# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Receive-window station, waveform and query helpers."""
import numpy as np

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.rx import varahf2300 as rx, varahf500 as rx500
from hfmodem.kestrel.tx import varahf2300_tx as tx, varahf500_tx as tx500
from hfmodem.kestrel.tx.varahf2300_tx import VB6Rnd
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_data_over_gate import _connected
from .test_idle_poll import clock, _Clock, _IO, _feed  # noqa: F401

_payload_oracles = []


def station(bw, role, caller="W9SSJ"):
    h, io = _connected(bw=bw, role=role, io=_IO())
    h.caller = caller
    h.turn = A._TURN_PEER
    h._start_compression('OFF')
    transmit = io.tx

    def timed_tx(samples):
        transmit(samples)
        _Clock.t += len(samples) / M.FS

    io.tx = timed_tx
    h._key(False)
    return h, io



def full_body(payload, caller, size=90, field=0x81, *, tail=None, body_len=None):
    field = field if tail is None else tail
    size = size if body_len is None else body_len
    body = phy.vara_body(payload, caller, tail=field, body_len=size)
    extracted = phy.vara_payload(body, caller=caller, body_len=size)
    assert extracted == payload
    _payload_oracles.append({'payload_hex': payload.hex(), 'body_hex': body.hex(),
                             'extracted_hex': extracted.hex()})
    return body


def window(bw, *payloads, level=4, caller="W9SSJ"):
    if bw == '2300' and level == 4 and caller == 'W9SSJ':
        from .test_two_over_window import _window
        return _window(*payloads)
    record = phy.level_key(bw, level)
    r = rx.RECORDS[record]
    span = r.ncols * r.dw50
    bursts = [tx.synth_burst(full_body(p, caller, phy.body_size(bw, record)),
                             record, i) for i, p in enumerate(payloads)]
    return np.concatenate([bursts[0]] + [b[-span:] for b in bursts[1:]])



def index_window(bw, level, bodies):
    record = phy.level_key(bw, level)
    r = rx.RECORDS[record]
    bursts = [tx.synth_burst(body, record, i) for i, body in enumerate(bodies)]
    return np.concatenate([bursts[0]] + [b[-r.ncols * r.dw50:] for b in bursts[1:]])


def psk_window(bw, level, bodies, seed=43):
    """DOUBLE-LAYOUT P10: four shared lead symbols, 149 symbols per body."""
    from hfmodem.kestrel.coding.crc import crc16_genibus
    from hfmodem.kestrel.rx import varaofdm as ofdm
    from hfmodem.kestrel.tx import varaofdm_tx as ofdm_tx

    r = ofdm.ROWS[int(bw), level]
    assert not r.qam and len(bodies) in (1, 2)
    frames = [b + crc16_genibus(b).to_bytes(2, 'big') for b in bodies]
    rnd = VB6Rnd(seed)
    for _ in frames:
        lead = [rnd.draw(r.bpc) for _ in range(r.lead * r.carriers)]
    head = r.lut[np.array(lead)].reshape(r.carriers, r.lead).T
    cells = np.concatenate([head, *(ofdm_tx._frame_cells(f, r) for f in frames)])
    spec = np.zeros((len(cells), ofdm.TU), complex)
    spec[:, r.bins] = cells
    symbols = np.fft.ifft(spec, axis=1).real
    ramp = np.sin(np.pi * (np.arange(32) + 0.5) / 64) ** 2
    span = np.arange(-ofdm.CP, ofdm.TU + len(ramp))
    shape = np.ones(len(span))
    shape[:len(ramp)], shape[-len(ramp):] = ramp, ramp[::-1]
    out = np.zeros(len(cells) * ofdm.SYM + len(ramp))
    for i, symbol in enumerate(symbols):
        out[i * ofdm.SYM + ofdm.CP + span] += shape * symbol[span % ofdm.TU]
    out = out[:len(cells) * ofdm.SYM]
    out[-len(ramp):] *= ramp[::-1]
    return out * (ofdm_tx.OUT_RMS / np.sqrt(np.mean(out ** 2)))


def tone_window(bw, level, bodies, seed=43):
    """DOUBLE-LAYOUT T7: shared lead, fresh reference phases for each body."""
    from hfmodem.kestrel.coding.crc import crc16_genibus
    from hfmodem.kestrel.rx import varatone as tone
    from hfmodem.kestrel.tx import varatone_tx as tone_tx

    r = tone.ROWS[int(bw), level]
    assert len(bodies) in (1, 2)
    rnd = VB6Rnd(seed)
    phases = tone.GRAY[np.array([rnd.draw(2) for _ in range((len(bodies) + tone.C0) * r.nb)])
                       .reshape(-1, r.nb).T]
    lead = (1j ** phases[:, len(bodies):]
            * np.exp(-2j * np.pi * tone.grid(r.nb)[:, :tone.C0] / 128))
    shift = (r.nsym - tone.C0) * tone.H
    single = tone.H * (r.nsym - 1) + len(r.taps) + max(r.stagger)
    if (int(bw), level) == (2300, 7):
        single = 187152
    out = np.zeros(single + (len(bodies) - 1) * shift)
    for i, body in enumerate(bodies):
        frame = body + crc16_genibus(body).to_bytes(2, 'big')
        cells = np.concatenate([lead if i == 0 else np.zeros_like(lead),
                                tone_tx.body(frame, r, phases[:, i])], axis=1)
        wave = tone_tx.render(cells, r)
        out[i * shift:i * shift + len(wave)] += wave
    return out * (tone_tx.RMS / np.sqrt(np.mean(out ** 2)))


def feed(h, *audio):
    _feed(h, np.concatenate([*audio, np.zeros(M.FS // 2)]))



def query(h, *, delay):
    wait = max(0, round((h._unkeyed_at + delay - _Clock.t) * M.FS))
    _feed(h, np.zeros(wait))
    if h.role == 'initiator':
        kind = F.SESSION_RESPONDER_IDLE
    else:
        kind = F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0]
    feed(h, M.synth_burst(h.called, F.for_bw(kind, h.bw)))



def assert_missing_nak(h, io, before):
    assert len(io.sent) == before + 1, io.msgs
    if h.role == 'responder':
        kinds = (F.SESSION_OVER_NAK_RESPONDER, F.SESSION_OVER_NAK_RESPONDER_SHORT)
        expected = [M.synth_burst(h.called, F.for_bw(kind, h.bw)) for kind in kinds]
        expected.append(M.synth_tone_pairs(F.nak(h.called, h.bw)[1]))
    else:
        expected = [M.synth_tone_pairs(F.nak(h.called, h.bw)[0]),
                    M.synth_burst(h.called, F.for_bw(F.SESSION_OVER_NAK, h.bw)),
                    M.synth_burst(h.called, F.for_bw(F.SESSION_OVER_NAK_SHORT, h.bw))]
    assert any(np.array_equal(io.sent[-1], x) for x in expected), io.msgs


def assert_bracket_recovery_nak(h, io, before, *, committed=b''):
    assert b''.join(io.host) == committed
    if h.bw == '2750' and h.role == 'initiator':
        assert len(io.sent) == before and h._owed_block
        assert h._rx_window.W is None
        assert h._rx_window.A.kind == 'nak' and not h._rx_window.A.sent
        kind = F.for_bw(F.SESSION_RESPONDER_IDLE, h.bw)
        idle = np.r_[M.synth_burst(h.called, kind), np.zeros(M.FS // 10)]
        _Clock.t += len(idle) / M.FS
        assert h._peer_responder_idle(idle) and h._idle_complete_exact
        assert h._a_gap()
        assert h._reack_over()
        np.testing.assert_array_equal(io.sent[-1], M.synth_burst(
            h.called, F.for_bw(F.SESSION_OVER_NAK, h.bw)))
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert b''.join(io.host) == committed



def burst(level, *values):
    bodies = [(value + b'\x95' if len(value) == rx500._PAYLOAD[level] else
               full_body(value, 'W9SSJ', rx500._PAYLOAD[level] + 1))
              for value in values]
    for value, body in zip(values, bodies):
        assert phy.vara_payload(body, caller='W9SSJ', body_len=len(body)) == value
    frames = [(tx500.build_frame(body[:-1], body[-1]) if level == 4 else
               tx500.build_l3_frame(body[:-1], body[-1], level)) for body in bodies]
    return tx500.synth_burst(frames, level=level, onset=24000)



def query500(h, *, delay):
    wait = max(0, round((h._unkeyed_at + delay - _Clock.t) * M.FS))
    _feed(h, np.zeros(wait))
    kind = (F.SESSION_RESPONDER_IDLE if h.role == 'initiator' else
            F.SESSION_INTERMEDIATE_ANSWER_QUERIES[0])
    feed(h, M.synth_burst(h.called, F.for_bw(kind, '500')))



def slot(level):
    rec = rx500.INDEX_RECORDS.get(level)
    return rec.ncols * rec.dw if rec else rx500.NSYM * rx500.H



def replay_window(bw, *payloads, level=4):
    record = phy.level_key(bw, level)
    r = rx.RECORDS[record]
    span = r.ncols * r.dw50
    bursts = [tx.synth_burst(full_body(p, 'W9SSJ', phy.body_size(bw, record), 0x99), record, i)
              for i, p in enumerate(payloads)]
    return np.concatenate([bursts[0]] + [b[-span:] for b in bursts[1:]])



def receive(h, data, route):
    if route == 'stream':
        feed(h, data)
    else:
        _Clock.t += len(data) / M.FS
        assert h._answer_data_over(data)



def top_window(h, *payloads):
    level = phy.TOP[h.bw]
    size = phy.body_size(h.bw, phy.level_key(h.bw, level))
    bodies = tuple(full_body(p, h.caller, size) for p in payloads)
    return np.r_[np.zeros(M.FS // 2), phy.render_level(bodies, h.bw, level, VB6Rnd(43))]



def top_station(role, predicted=False):
    h, io = station('500', role)
    h._link_high_speed = True
    h._expect_level = phy.TOP[h.bw]
    h._lv_armed = True
    if predicted:
        size = phy.body_size('500', phy.level_key('500', phy.TOP['500']))
        for i in range(1, 7):
            body = full_body(bytes([i]) * (size - 1), h.caller, size, 0x85)
            h._level_read = phy.LevelRead(phy.TOP['500'], (body,), None, 1, end=0)
            try:
                assert h._answer_data_over(np.zeros(0))
            finally:
                h._level_read = None
            assert len(io.sent) == i and h._rx_window.A.step == 0
        h._lv_armed = True
        assert h._peer_doubles()
        assert b''.join(io.host) == b''.join(bytes([i]) * (size - 1) for i in range(1, 7))
    return h, io



def answered_metadata(h, *, kind='continue', sent=True):
    from hfmodem.kestrel.arq.rxwindow import Answer

    h._rx_window.A = Answer(1, 0, kind, original=A.OVER_CONTINUE_BURST, sent=sent)
    h._rx_window._attempt = 1
    h._answer_owed = A._OWED_OVER
    h._reack_frame = A.OVER_CONTINUE_BURST
    h._over_answer = ('continue', A.OVER_CONTINUE_BURST)




import pytest


@pytest.fixture(autouse=True)
def rx_evidence(monkeypatch, request, record_property):
    from dataclasses import asdict, is_dataclass
    import json
    from hfmodem.kestrel.arq import rxwindow as RW

    stations, history = [], []
    _payload_oracles.clear()
    init = A.VaraStationHandshake.__init__
    event, sent = RW.ReceiveWindow.event, RW.ReceiveWindow.sent
    perform = getattr(A.VaraStationHandshake, '_perform_rx_actions', None)

    def constructed(h, *args, **kwargs):
        init(h, *args, **kwargs)
        stations.append(h)

    def snapshot(machine):
        return {'state': machine.state, 'R': machine.R, 'closed': machine.closed,
                'alt': machine.alt, 'trusted': machine.trusted,
                **{name: getattr(machine, name) for name in ('W', 'A', 'K', 'T')}}

    def observed(machine, kind, **kwargs):
        entry = {'event': kind, 'inputs': kwargs, 'before': snapshot(machine)}
        actions = event(machine, kind, **kwargs)
        entry.update(after=snapshot(machine), actions=actions)
        history.append(json.loads(json.dumps(entry, default=encoded)))
        return actions

    def aired(machine, key, reached_air):
        history.append({'sent': key, 'reached_air': reached_air})
        return sent(machine, key, reached_air)

    def performed(h, actions, **kwargs):
        history.extend({'action': type(action).__name__, 'value': action} for action in actions)
        return perform(h, actions, **kwargs)

    def encoded(value):
        if isinstance(value, bytes):
            return {'hex': value.hex()}
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if is_dataclass(value):
            return asdict(value)
        return repr(value)

    monkeypatch.setattr(A.VaraStationHandshake, '__init__', constructed)
    monkeypatch.setattr(RW.ReceiveWindow, 'event', observed)
    monkeypatch.setattr(RW.ReceiveWindow, 'sent', aired)
    if perform is not None:
        monkeypatch.setattr(A.VaraStationHandshake, '_perform_rx_actions', performed)
    yield
    record_property('window_evidence', json.dumps({
        'history': history,
        'parameters': getattr(getattr(request.node, 'callspec', None), 'params', {}),
        'fixture_payload_oracles': list(_payload_oracles),
        'stations': [{'machine': snapshot(h._rx_window) if hasattr(h, '_rx_window') else None,
                      'host_hex': b''.join(getattr(h.io, 'host', ())).hex(),
                      'messages': getattr(h.io, 'msgs', ()),
                      'reports': getattr(h, '_receive_reports', ())} for h in stations],
    }, default=encoded))


def commit_bodies(h, bodies, *, verdict=None):
    h._deliver(bodies, verdict=verdict)
    h._perform_rx_actions(h._rx_window.event('E', single=True, envelope='single'))


def arm_replay(h, io, *, query=query):
    before = len(io.sent)
    query(h, delay=12.)
    assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert h._rx_window.T is not None
    assert (h._rx_window.T.start, h._rx_window.T.frontier) == (h._rx_window.K.start, h._rx_window.S)


def assert_r8(h, io, before, surviving, rejected):
    assert h._rx_window.closed and h.state == A.VaraState.DISCONNECTED
    assert len(io.sent) == before
    assert b''.join(io.host) == surviving
    report = h._receive_reports[-1]
    assert report.code == 'R8' and report.rejected == rejected
    assert report.frontier == len(surviving) and report.start == 0


def terminal_retry(h, io, terminal, expected, *, unextended=False):
    assert b''.join(io.host) == expected[:-len(terminal)]
    if h._rx_window.W is not None:
        feed(h, np.zeros(A._ANSWER_HOLD_MAX))
    if h._rx_window.A.kind != 'nak' or not h._rx_window.A.sent:
        before = len(io.sent)
        query(h, delay=.2)
        assert_missing_nak(h, io, before)
    assert h._rx_window.A.kind == 'nak' and h._rx_window.A.sent
    assert_missing_nak(h, io, len(io.sent) - 1)
    assert b''.join(io.host) == expected[:-len(terminal)]
    assert h._rx_window.T is None
    if not unextended:
        feed(h, window(h.bw, terminal, level=2, caller=h.caller))
        assert b''.join(io.host) == expected
