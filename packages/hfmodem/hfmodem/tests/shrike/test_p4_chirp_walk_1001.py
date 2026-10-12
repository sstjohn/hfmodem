# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""D1: o0802, 2206, 2153 and 2213, plus bounded scripted continuations."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.tests.evidence import require_input
from hfmodem.core import wav
from hfmodem.shrike import onair, p4chirp, p4chirprx, p4oracle, p4robusttx, p4words
from hfmodem.shrike.p4receiver import P4Receiver
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4turn import P4Turn, State
from hfmodem.tests.shrike.corpora import n9vc_0802
from hfmodem.tests.shrike.test_p4turn import machine, word, FS

ROOT = Path(__file__).resolve().parents[5]
SHORT = round(1.25*FS)
LONG = round(3.75*FS)
MAIL = b';FW: W9SSJ\r[Pat-1.0.0-B2FHM$]\r;PR: 48753069\r'


def answer(m, name):
    at = m.last.stream_sample + round((3.4 if m.last.kind == Kind.CHIRP else .92)*FS)
    m.observe(word(name, at, responds_to=m.last.stream_sample), now=at+round(.245*FS))
    return m.keyed()


def silence(m):
    due = m.last.stream_sample + SHORT
    m.silence(due-m.lead)
    return m.keyed()


def ladder():
    m, a = machine(ladder='chirp', data='chirp')
    a.on_host_data(MAIL)
    m.start(0)
    m.keyed()
    return m, a


def admitted(m):
    assert answer(m, 'X27').value == 49
    assert answer(m, 'X27').value == 53
    return answer(m, 'U27')


@pytest.mark.parametrize('name', [None, 'Z27', 'T27', 'W27'])
def test_o0802_unanswered_walk_has_one_b_then_steps(name):
    m, _ = ladder()
    sent = [m.last]
    for _ in range(4):
        sent.append(silence(m) if name is None else answer(m, name))
    assert [(p.value, p.header) for p in sent] == [(13, 'A'), (13, 'B'), (49, 'A'), (49, 'B'), (53, 'A')]
    for _ in range(3):
        p = silence(m)
        assert (p.value, p.header) == (53, 'B')
    for p in sent:
        audio = p4robusttx.robust_short(p.value, p.header)
        r = p4oracle.robust_at(audio, FS, 0, 0)
        assert (r['p4_value'], r['p4_walsh_peak'], r['p4_hdr']) == (p.value, 64, p.header)


@pytest.mark.parametrize('name', ['U27', 'X27', 'V27'])
def test_only_final_value_admits_chirp_in_next_short_slot(name):
    m, _ = ladder()
    assert answer(m, 'X27').value == 49
    assert answer(m, name).value == 53
    at = m.last.stream_sample
    p = answer(m, name)
    assert (p.kind, p.status, p.payload, p.stream_sample) == (Kind.CHIRP, 0x61, MAIL[:22], at+SHORT)
    assert m.state == State.ISS_DATA
    assert m.next_raster(p.stream_sample+round(3.4*FS)) == p.stream_sample+LONG


@pytest.mark.parametrize('name', ['T27', 'W27', 'Z27'])
def test_final_value_rejects_other_words_and_keeps_header_b(name):
    m, _ = ladder()
    answer(m, 'X27')
    answer(m, 'X27')
    for _ in range(3):
        p = answer(m, name)
        assert (p.kind, p.value, p.header) == (Kind.SHORT, 53, 'B')


def test_chirp_counter_payload_repeat_and_slot_are_independent():
    m, a = ladder()
    first = admitted(m)
    original = a._buffer_raw
    p = answer(m, 'T27')
    assert (p.status, p.payload, p.repeat, p.chirp_slot) == (0x61, first.payload, True, 1)
    assert a._buffer_raw == original
    # An unkeyed slot still changes the physical lane parity.
    m.silence(p.stream_sample+LONG-m.lead)
    m.refused('deadline')
    p = m.keyed()
    assert p.chirp_slot == 3 and p.repeat
    p = answer(m, 'X27')
    assert (p.status, p.payload, p.repeat, p.chirp_slot) == (0x62, MAIL[22:], False, 4)
    assert a._buffer_raw == original-22
    assert answer(m, 'X27').repeat
    answer(m, 'V27')
    assert a._buffer_raw == 0
    for p in (first, p):
        r = p4chirprx.decode_packet(p4chirp.entry_packet(p.payload, p.status,
            repeat=p.repeat, rank0_upper=bool(p.chirp_slot % 2)))
        assert r is not None and r.frame.status.byte == p.status

    # An ACK that expires just inside the next prekey window skips that slot.
    m, _ = ladder()
    first = admitted(m)
    onset = first.stream_sample + round(3.4*FS)
    m.observe(word('X27', onset, partial=True, responds_to=first.stream_sample),
              now=onset+round(.210*FS))
    limit = onset + round(.275*FS) + LONG
    m.silence(limit-1)
    m.silence(limit+round(.010*FS))
    p = m.keyed()
    assert p.stream_sample == first.stream_sample+3*LONG
    assert p.chirp_slot == 3 and p.repeat


def test_2213_scripted_continuation_uses_only_unacked_bytes():
    audio = wav.read(require_input(ROOT/'captures/onair-0930-2213/stream.flac'), FS)
    # The measured 1 Hz is X27's carrier; word_at takes the body reference.
    ack = p4oracle.word_at(audio, FS, 0, 32.194,
                           cfo_hz=1.-p4words.CARRIER_HZ['X27'])
    assert ack is not None and ack.word == 'X27'
    m, a = machine(ladder='chirp', data='chirp')
    a.on_host_data(MAIL)
    m._begin_data_turn(round(31.264*FS), State.ISS_DATA_ACK)
    c0 = m.keyed()
    m.observe(word(ack.word, ack.stream_sample, responds_to=c0.stream_sample))
    assert m.keyed().value == 13
    assert answer(m, 'X27').value == 49
    assert answer(m, 'Z27').header == 'B'
    assert silence(m).value == 53
    chirp = answer(m, 'U27')
    assert (chirp.kind, chirp.status, chirp.payload) == (Kind.CHIRP, 0x61, MAIL[31:53])
    assert a._buffer_raw == len(MAIL)-31


@pytest.mark.parametrize('word_yield', [False, True])
def test_chirp_z27_returns_to_ladder_once_for_both_turn_forms(word_yield):
    m, a = ladder()
    m._word_yield = word_yield
    p = admitted(m)
    status = 0x60 if word_yield else 0x61
    assert p.status == status
    assert answer(m, 'Z27').repeat
    assert answer(m, 'Z27').value == 13
    p = admitted(m)
    assert (p.status, p.payload) == (status, MAIL[:22])
    assert p.repeat  # Negotiating again does not make the unacked packet new.
    assert answer(m, 'T27').kind == Kind.CHIRP
    assert answer(m, 'Z27').kind == Kind.CHIRP
    assert answer(m, 'Z27').state == State.ISS_CLOSE
    assert a._buffer_raw == len(MAIL)


@lru_cache(None)
def stock_chirps():
    fs, raw = n9vc_0802()
    return (hilbert(raw.astype(float))*np.exp(-2j*np.pi*400*np.arange(len(raw))/fs)).real


@pytest.mark.parametrize('slot,marker,upper', [(0, 8, False), (1, 9, True), (2, 9, False)])
def test_o0802_render_matches_crc_marker_lanes_and_word_timing(slot, marker, upper):
    audio = stock_chirps()
    # Nominal waveform starts; matched header peaks are about 6 ms later.
    h = p4chirprx.acquire_header(audio, nominal_sample=round((6.660+slot*3.75)*FS))
    assert h is not None
    stock = p4chirprx.decode_packet(audio, start_sample=h.start_sample)
    assert stock is not None and (stock.marker, stock.rank0_upper) == (marker, upper)
    rendered = p4chirp.entry_packet(stock.frame.data, stock.frame.status.byte,
        repeat=slot > 0, rank0_upper=upper)
    ours = p4chirprx.decode_packet(rendered)
    assert ours is not None and ours.frame == stock.frame
    assert (ours.marker, ours.rank0_upper) == (stock.marker, stock.rank0_upper)
    assert len(rendered) == round(3.3075*FS)
    _, symbols, at = p4oracle.placed_despread(audio, FS, 6.330, -8., count=27)
    assert p4words.match(symbols)[0] == 'U27'
    assert abs(6.660-at/FS-.330) < .005


@pytest.mark.parametrize('at,repeat,upper', [(39.750, False, False), (43.502, True, True), (54.750, True, False)])
def test_2206_marker_is_repeat_flag_not_lane_order(at, repeat, upper):
    path = ROOT/'captures/onair-0930-2206/stream.flac'
    require_input(path)
    audio = wav.read(path, FS)
    h = p4chirprx.acquire_header(audio, nominal_sample=round(at*FS), frequency_offset=.44)
    assert h is not None
    stock = p4chirprx.decode_packet(audio, start_sample=h.start_sample, frequency_offset=.44)
    assert stock is not None and (stock.marker, stock.rank0_upper) == (8+repeat, upper)
    ours = p4chirprx.decode_packet(p4chirp.entry_packet(stock.frame.data,
        stock.frame.status.byte, repeat=repeat, rank0_upper=upper))
    assert ours is not None and ours.frame == stock.frame
    assert (ours.marker, ours.rank0_upper) == (stock.marker, stock.rank0_upper)


def witness_words(capture):
    require_input(capture/'witness.wav')
    require_input(ROOT/'tools/witness_align.py')
    from witness_align import _at_48k, witness_rate
    path = capture/'witness.wav'
    x = _at_48k(path, witness_rate(path, 11999.0))
    placed_words = []
    for when, word_cfo, name in ((47.194, 7., 'Z27'), (48.444, 1., 'U27')):
        cfo = word_cfo-p4words.CARRIER_HZ.get(name, 0.)
        o = p4oracle.word_at(x, FS, 0, when+3.7384, cfo_hz=cfo, slack_s=.035)
        assert o is not None and o.word == name and o.score > .90
        early = p4oracle.word_at(x[:round((when+3.7384+.215)*FS)], FS, 0,
            when+3.7384, cfo_hz=cfo, slack_s=.035, provisional=True)
        assert early is not None and early.word == name and not early.weak
        placed_words.append((round((o.stream_sample/FS-3.7384)*FS), o, early))
    return placed_words


def replay(monkeypatch, arm, *, witness=False):
    capture = ROOT/f'captures/onair-0930-{arm}'
    require_input(capture/'stream.flac')
    require_input(capture/'p4-probe.json')
    require_input(ROOT/'working/p4-plan-0930/stall-bisect/drive.py')
    probe = json.loads((capture/'p4-probe.json').read_text())
    opening = probe['emissions'][0]
    request = next(e for e in probe['emissions'] if e.get('status') == 0x40)
    # Repeated bodies are one outstanding host field, not additional mail.
    fields = list(dict.fromkeys(e['payload'] for e in probe['emissions'] if e.get('payload')))
    payload = b''.join(bytes.fromhex(x) for x in fields)
    run, keyed = onair._run_p4_link, P4Turn.keyed
    sent, turns, waves = [], [], []
    def track(m, at=None):
        p = keyed(m, at)
        sent.append(p)
        if p.kind == Kind.CHIRP:
            decoded = p4chirprx.decode_packet(waves[-1])
            assert decoded is not None
            assert decoded.frame.status.byte == p.status
            assert decoded.frame.data[:len(p.payload)] == p.payload
            assert decoded.marker == 8+int(p.repeat)
            assert decoded.rank0_upper == bool(p.chirp_slot % 2)
        return p
    monkeypatch.setattr(P4Turn, 'keyed', track)
    def drive(tx, stream, raster):
        tx.p4_link_options.p4_ladder = 'chirp'
        tx.p4_link_options.p4_data = 'chirp'
        tx.p4_probe.emissions = [opening.copy()]
        take, transmit = stream.take_until, tx._tx
        queued = False
        def feed(end):
            nonlocal queued
            if not queued and end >= request['audio_start']-round(1.3*FS):
                tx.p4_link_host.arq.on_host_data(payload)
                queued = True
            return take(end)
        def tx_audio(audio, what):
            waves.append(audio.copy())
            transmit(audio, what)
        stream.take_until, tx._tx = feed, tx_audio
        run(tx, stream, raster)
        turns.append(tx)
    monkeypatch.setattr(onair, '_run_p4_link', drive)
    if witness:
        # 2153's stream misses 48.444 even on a placed carrier read. Register
        # the independently decoded witness word on the measured stream clock.
        placed_words = witness_words(capture)
        read = P4Receiver._word
        def placed(rx, audio, origin, **kw):
            provisional = kw.get('provisional', False)
            for at, full, early in placed_words:
                o = early if provisional else full
                span = .210 if provisional else .245
                if abs(rx.clock.next_due-at) < round(.035*FS) and origin+len(audio) >= at+round(span*FS):
                    return replace(o, origin=origin, stream_sample=at, first_chip=at-origin,
                                   provisional=provisional, unexpected=False)
            return read(rx, audio, origin, **kw)
        monkeypatch.setattr(P4Receiver, '_word', placed)
    monkeypatch.setattr(sys, 'argv', ['drive.py', str(capture/'stream.flac'),
        str(opening['audio_start'] % SHORT), str(opening['audio_start']), '1',
        str(opening['audio_end']/FS), '65'])
    runpy.run_path(str(ROOT/'working/p4-plan-0930/stall-bisect/drive.py'), run_name='__main__')
    return sent, turns[0]


def test_2153_witness_u27_admits_crc_valid_data_in_driver(monkeypatch):
    key = P4Turn.keyed
    sent, tx = replay(monkeypatch, '2153', witness=True)
    shorts = [p for p in sent if p.kind == Kind.SHORT and p.value in (13, 49, 53)]
    assert [(p.value, p.header) for p in shorts[:5]] == [(13, 'A'), (13, 'B'), (49, 'A'), (49, 'B'), (53, 'A')]
    first = next(p for p in sent if p.kind == Kind.CHIRP)
    # BODY readiness permits c0 before the ladder; fixed tape then admits
    # earlier with X27. Neither that key nor the old witness key is universal.
    c0 = [p for p in sent[:sent.index(shorts[0])]
          if p.kind == Kind.BODY and p.status == 0]
    assert c0 and all(p.state == State.ISS_DATA_ACK and p.payload == MAIL[:31] for p in c0)
    events = tx.p4_probe.link_events
    assert not [e for e in events if e['event'] == 'Refused'
                and e['kind'] == Kind.BODY and e['reason'] == 'waveform not prepared']
    final = shorts[4]
    full = next(h for h in tx.p4_probe.heard
                if h['responds_to'] == final.stream_sample and not h['provisional']
                and h['word'] in ('U27', 'X27', 'V27') and not h['weak'] and not h['unexpected'])
    decision = next(e for e in events if e['event'] == 'Decision'
                    and e['at'] >= full['onset_sample']
                    and e['proposed_at'] == first.stream_sample)
    assert full['received_at_sample'] <= decision['now_sample']
    assert first.stream_sample == decision['scheduled_at'] == final.stream_sample+SHORT
    emission = next(e for e in tx.p4_probe.emissions if e.get('status') == 0x61)
    assert emission['proposed_at'] == emission['keyed_at'] == first.stream_sample
    assert first.status == 0x61 and first.payload == MAIL[:22]
    assert first.state == State.ISS_DATA
    assert tx.p4_probe.reason != 'retry budget exhausted'

    # The fixed tape cannot answer our earlier changed emission. Independently
    # pin its witness U27 with a scripted final-v53 prefix on the same clock.
    at, full, early = witness_words(ROOT/'captures/onair-0930-2153')[-1]
    from hfmodem.shrike.p4observation import Registered, RegisteredObservation
    m, a = machine(ladder='chirp', data='chirp')
    a.on_host_data(MAIL)
    m.start(at-round(.930*FS)-2*SHORT)
    assert key(m).value == 13
    for value in (49, 53):
        onset = m.last.stream_sample+round(.920*FS)
        m.observe(word('X27', onset, responds_to=m.last.stream_sample),
                  now=onset+round(.245*FS))
        final = key(m)
        assert final.value == value
    original_pending_key = final.stream_sample+SHORT
    for obs, outcome, span in ((early, Registered.NEW, .210),
                               (full, Registered.CONFIRMED, .245)):
        obs = replace(obs, stream_sample=at, unexpected=False)
        m.observe(RegisteredObservation(obs, outcome, responds_to=final.stream_sample),
                  now=at+round(span*FS))
        if obs.provisional:
            assert m._next is None  # A partial U27 must not admit data.
    proposal = m._next
    assert proposal is not None and proposal.kind == Kind.CHIRP
    assert proposal.stream_sample == original_pending_key
    first = key(m)
    assert first.stream_sample == original_pending_key
    assert first.status == 0x61 and first.payload == MAIL[:22]


def test_2213_granted_c0_ack_runs_all_three_values_in_driver(monkeypatch):
    sent, _ = replay(monkeypatch, '2213')
    shorts = [p for p in sent if p.kind == Kind.SHORT and p.value in (13, 49, 53)]
    assert abs(shorts[0].stream_sample/FS-32.514) < .010
    assert [(p.value, p.header) for p in shorts[:4]] == [(13, 'A'), (49, 'A'), (49, 'B'), (53, 'A')]
