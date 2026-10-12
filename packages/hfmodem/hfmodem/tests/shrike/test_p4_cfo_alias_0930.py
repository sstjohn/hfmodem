# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Carrier steps and tone aliases from WS8EOC's September 30 recordings."""
from pathlib import Path

import pytest

from hfmodem.shrike import onair, p4breakin, p4words, session
from hfmodem.shrike.p4observation import Kind
from hfmodem.shrike.p4receiver import Clock, Expect, P4Receiver
from hfmodem.tests.shrike.test_p4_link_0927 import FS, Stream, Tx

ROOT = Path(__file__).resolve().parents[5]
GREETING = b'RMS Trimode 1.4.4.0\r\nW9SSJ has '


@pytest.fixture(scope='module')
def recordings():
    paths = [ROOT/f'captures/onair-0930-{arm}/stream.flac' for arm in ('2153', '2221')]
    if not all(p.exists() for p in paths):
        pytest.skip('local September 30 2153/2221 recordings absent')
    return [session.load_wav(str(p), FS) for p in paths]


def test_2153_carrier_step_registers_u27(recordings):
    audio = recordings[0]
    rx = P4Receiver(FS, admitted=True, cfo_hz=10.6, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.WORD, round(45.944*FS), .923, 1.25, 10.6)
    origin = round(45.894*FS)
    regs = rx.read_window(audio[origin:round(46.244*FS)], origin, whole=True)
    assert any(r.observation.word == 'Z27' for r in regs)
    assert 10 < rx.clock.cfo_hz < 12
    # Keep listening through the long-cycle gap; the flight ladder stopped
    # before this word, so the pin isolates the receiver's carrier defect.
    rx.clock.next_due = round(52.194*FS)
    origin = round(52.144*FS)
    regs = rx.read_window(audio[origin:round(52.494*FS)], origin, whole=True)
    words = [r.observation for r in regs if r.observation.word == 'U27']
    assert len(words) == 1
    assert abs(words[0].stream_sample/FS-52.194) < .001
    assert 4 < words[0].cfo_hz < 6
    assert rx.cfo_hz == pytest.approx(words[0].cfo_hz-p4words.CARRIER_HZ['U27'])


def test_2153_step_from_a_calibrated_body_carrier(recordings):
    at = round(52.194*FS)
    origin = at-round(.05*FS)
    # The earlier word was +10.6 Hz; this body reference places U27 there.
    carrier = 10.6-p4words.CARRIER_HZ['U27']
    rx = P4Receiver(FS, admitted=True, cfo_hz=carrier, breakin=True)
    rx.enable_hold()
    rx._body_carrier_known = True
    rx.clock = Clock(Expect.WORD, at, .923, 1.25, carrier)
    regs = rx.read_window(recordings[0][origin:at+round(.30*FS)], origin, whole=True)
    words = [r.observation for r in regs if r.observation.word == 'U27']
    assert len(words) == 1
    assert abs(words[0].stream_sample-at) < round(.001*FS)
    assert 4 < words[0].cfo_hz < 6
    assert rx.cfo_hz == pytest.approx(words[0].cfo_hz-p4words.CARRIER_HZ['U27'])
    assert rx.clock.cfo_hz == rx.cfo_hz


def test_2221_first_lock_and_greeting_in_driver(recordings):
    stream = Stream(recordings[1][:round(16.1*FS)])
    first = 662625
    stream.pos = first+36367
    tx = Tx(stream)
    tx.p4_probe.emissions = [dict(audio_start=first, audio_end=first+36367)]
    tx.p4_link_options.p4_ladder = 'none'
    tx.p4_link_options.p4_data = 'body'
    tx.p4_link_options.p4_normal_ack = 'parity'
    tx.p4_link_options.p4_breakin = 'body'
    onair._run_p4_link(tx, stream, onair._MasterGrid(
        first%60000, 60000, 0, packet_n=36000, cs_n=12000, d_max_n=4800))
    heads = [h for h in tx.p4_probe.heard if h.get('word') == 'Y27'
             and not h.get('provisional')]
    assert heads and 4 < heads[0]['cfo_hz'] < 5
    assert any(h.get('data') == GREETING.hex() for h in tx.p4_probe.heard)
    bodies = [h for h in tx.p4_probe.heard if h.get('data') == GREETING.hex()]
    assert 2.5 < bodies[0]['cfo_hz'] < 3.5
    assert tx.p4_probe.emissions[1]['word'] == 'X27'


def test_2221_existing_alias_recovers_without_extra_trellis(recordings, monkeypatch):
    at, origin = 707883, 705483
    rx = P4Receiver(FS, admitted=True, cfo_hz=115.632576, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.WORD, at, .923, 1.25, rx.cfo_hz)
    decode = p4breakin.decode_scores
    calls = []

    def counted(scores):
        calls.append(1)
        return decode(scores)

    monkeypatch.setattr(p4breakin, 'decode_scores', counted)
    regs = rx.read_window(recordings[1][origin:at+round(.85*FS)], origin, whole=True)
    assert not any(r.observation.kind == Kind.BODY for r in regs)
    assert len(calls) == 1
    at, origin = 887775, 885375
    rx.clock.next_due = at
    regs = rx.read_window(recordings[1][origin:at+round(.85*FS)], origin, whole=True)
    bodies = [r.observation for r in regs if r.observation.kind == Kind.BODY]
    assert len(bodies) == 1 and bodies[0].frame.data == GREETING
    assert 2.5 < rx.clock.cfo_hz < 3.5
    assert len(calls) == 2
    correction, = [e for e in rx.live_events if e['event'] == 'body carrier alias']
    assert correction['previous_cfo_hz'] > 110
    assert correction['cfo_hz'] == pytest.approx(rx.clock.cfo_hz)
    assert len([o for o in rx.observations if o.word == 'Y27'
                and abs(o.stream_sample-at) < FS//10]) == 1
    assert rx.pending_body is None and not rx._body_heads


def test_training_preference_without_crc_does_not_move_carrier(recordings, monkeypatch):
    at, origin = 707883, 705483
    rx = P4Receiver(FS, admitted=True, cfo_hz=115.632576, breakin=True)
    rx.enable_hold()
    rx.clock = Clock(Expect.WORD, at, .923, 1.25, rx.cfo_hz)
    rx.read_window(recordings[1][origin:at+round(.85*FS)], origin, whole=True)
    held = rx.hold.members.copy()
    assert held
    rx._body_carrier_check = True
    monkeypatch.setattr(p4breakin, 'decode_scores', lambda scores: None)
    at, origin = 887775, 885375
    rx.clock.next_due = at
    regs = rx.read_window(recordings[1][origin:at+round(.85*FS)], origin, whole=True)
    assert not any(r.observation.kind == Kind.BODY for r in regs)
    assert rx.clock.cfo_hz > 100
    assert all(any(member is old for member in rx.hold.members) for old in held)
    assert not any(e['event'] == 'body carrier alias' for e in rx.live_events)
