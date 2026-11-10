# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import p4chirprx, p4frame, p4oracle, p4words
from hfmodem.shrike.p4observation import Registered, RegisteredObservation
from hfmodem.shrike.p4receiver import Kind, Observation
from hfmodem.shrike.p4turn import State
from hfmodem.tests.shrike.corpora import n9vc_0802
from hfmodem.tests.shrike.test_p4turn import machine


def chirp(frame, fs, at):
    return Observation(Kind.CHIRP, fs, 0, at, at, 3.28, 0., .8, frame=frame)


def answer(frame, fs=48000, at=48000):
    m, arq = machine()
    m.state = State.IRS_ANSWER
    m.first_body = False
    m.observe(RegisteredObservation(chirp(frame, fs, at), Registered.NEW),
              now=at+round(3.30*fs))
    return m.decide(at+round(3.400*fs)-m.lead), arq


def test_stock_n9vc_protocol3_chirp_draws_x27():
    fs, pcm = n9vc_0802()
    base = 6.
    audio = pcm[round(base*fs):round(10.6*fs)]/32768
    audio = (hilbert(audio)*np.exp(-2j*np.pi*400*np.arange(len(audio))/fs)).real
    onset = round((6.660-base)*fs)
    received = p4chirprx.decode_packet(audio, fs=fs, start_sample=onset)
    assert received.frame.status.byte == 0x61
    assert received.frame.status.protocol == p4frame.ProtocolStatus.SPEED_UP
    fraction, symbols, stock = p4oracle.placed_despread(audio, fs, 10.090-base, -4., count=27)
    # This isolated word observation does not establish an accepted chirp ack.
    assert p4words.match(symbols)[0] == 'U27'
    proposal, arq = answer(received.frame, fs, onset)
    assert proposal is not None and proposal.kind == Kind.WORD and proposal.word == 'X27'
    assert proposal.stream_sample == onset+round(3.400*fs)
    assert abs(proposal.stream_sample-stock) < round(.035*fs)
    assert proposal.ack_identity == (0x61, received.frame.data)
    assert arq._buffer_raw == 0


@pytest.mark.parametrize('status,kind,word', [(0x00, Kind.WORD, 'V27'), (0x21, Kind.WORD, 'V27'),
                                              (0x41, Kind.WORD, 'X27')])
def test_other_chirp_protocols_keep_their_answers(status, kind, word):
    data = b'[BPQ-6.0.25.1-B2FWIHJM'
    frame = p4frame.decode_field(p4frame.encode_field(data, p4frame.Status.from_byte(status)),
                                 data_bytes=len(data))
    proposal, _ = answer(frame)
    assert (proposal.kind, proposal.word) == (kind, word)


@pytest.mark.parametrize('kind,delay,cycle', [(Kind.BODY, .920, 1.25), (Kind.CHIRP, 3.400, 3.75)])
@pytest.mark.parametrize('refused', [False, True])
def test_protocol3_exchange_answers_repeats_after_refused_keys(kind, delay, cycle, refused):
    from hfmodem.tests.shrike.test_p4turn import body
    from dataclasses import replace
    m, _ = machine(role='irs')
    first = body(48000, b'gateway', 0x61)
    first = replace(first, observation=replace(first.observation, kind=kind))
    due = 48000+round(delay*48000)
    m.observe(first, now=due-m.lead-1)
    p = m.decide(due-m.lead)
    expected = 'X27' if kind == Kind.CHIRP else 'U27'
    assert p and p.word == expected and p.stream_sample == due
    if refused:
        m.refused('deadline')
        assert m._last_ack is None
    else:
        m.keyed()
    repeat = 48000+round(cycle*48000)
    m.observe(replace(first, observation=replace(first.observation, stream_sample=repeat)),
              now=repeat+round(delay*48000)-m.lead-1)
    p = m.decide(repeat+round(delay*48000)-m.lead)
    assert p.word == expected and p.stream_sample == repeat+round(delay*48000)
    m.keyed()
    assert m.adapter.last == (0x61, first.observation.frame.data)


@pytest.mark.parametrize('lead', [.044, .074])
@pytest.mark.parametrize('cost_ms', [0, 40])
def test_recorded_protocol3_chirp_through_causal_driver(monkeypatch, lead, cost_ms):
    from scipy.signal import resample_poly
    from hfmodem.shrike import onair, p4turn
    from hfmodem.shrike.p4receiver import Clock, Expect
    from hfmodem.tests.shrike.test_p4_link_0927 import Stream, Tx, FS
    fs, pcm = n9vc_0802()
    piece = pcm[round(6.160*fs):round(10.0*fs)]/32768
    piece = (hilbert(piece)*np.exp(-2j*np.pi*400*np.arange(len(piece))/fs)).real
    audio = np.zeros(10*FS)
    piece = resample_poly(piece, FS, fs)
    audio[:len(piece)] = piece
    live = Stream(audio)
    live.wall = 0
    live.sample_now = lambda: max(live.pos, live.wall)
    tx = Tx(live)
    tx.settle = lead
    tx.p4_probe.emissions.clear()
    tx.p4_link_options.p4_link_timeout = 9.
    receiver = tx.sessrx.p4
    receiver.chirp = receiver.expected = True
    receiver.cfo_hz = 0.
    receiver.clock = Clock(Expect.PACKET, FS//2, .330, 3.75, 0.)
    turn = p4turn.P4Turn
    monkeypatch.setattr(p4turn, 'P4Turn', lambda *a, **kw: turn(*a, **dict(kw, role='irs')))
    read = receiver.read_window
    def charged(*a, **kw):
        result = read(*a, **kw)
        if any(r.observation.frame is not None for r in result):
            live.wall = live.sample_now()+round(cost_ms*FS/1000)
        return result
    monkeypatch.setattr(receiver, 'read_window', charged)
    onair._run_p4_link(tx, live, onair._MasterGrid(
        0, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    heard = [r for r in tx.p4_probe.heard if r['kind'] == Kind.CHIRP and 'status' in r]
    assert heard and heard[0]['status'] == 0x61
    requests = {e['at'] for e in tx.p4_probe.link_events
                if e['event'] == 'Keyed' and not e.get('current_slot_accepted', True)}
    sent = [e for e in tx.p4_probe.emissions if e['audio_start'] not in requests]
    assert len(sent) == 1 and sent[0]['word'] == 'X27'
    due = heard[0]['onset_sample']+round(3.4*FS)
    assert sent[0]['audio_start'] in (due, due+180000)
    assert heard[0]['capture_end_sample'] >= heard[0]['onset_sample']+round(3.3075*FS)
    assert sent[0]['audio_start'] >= heard[0]['received_at_sample']+round(lead*FS)
