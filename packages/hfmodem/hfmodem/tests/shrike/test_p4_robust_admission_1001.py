# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""D6: a complete Walsh word survives weak spreading and header polarity."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from hfmodem.tests.evidence import require_input
from hfmodem.shrike import p4oracle, p4robusttx, session
from hfmodem.shrike.p4receiver import Clock, Emission, Expect, Kind, P4Receiver, Role

ROOT = Path(__file__).resolve().parents[5]
FS = 48000


@lru_cache(maxsize=3)
def capture(arm):
    arm = arm if len(arm) > 4 else '0930-' + arm
    path = ROOT / 'captures' / f'onair-{arm}' / 'stream.flac'
    require_input(path)
    return session.load_wav(str(path), FS)


def placed(arm, at, cfo, partial=False):
    origin = round((at - .05) * FS)
    audio = capture(arm)[origin:round((at + .85) * FS)]
    rx = P4Receiver(FS, admitted=True, cfo_hz=cfo)
    rx.clock = Clock(Expect.PACKET, round(at * FS), .33, 1.25, cfo)
    obs = rx._packet(audio, origin, partial=partial)
    return rx, obs


@pytest.mark.parametrize('at,coherence', [(63.504, .2451), (64.754, .2438), (67.254, .2835)])
@pytest.mark.parametrize('partial', [False, True])
def test_2031_full_walsh_v53_is_admitted(at, coherence, partial):
    field = p4oracle.robust_at(capture('2031'), FS, at, 2.63)
    assert (field['p4_value'], field['p4_walsh_peak']) == (53, 64)
    assert field['coherence'] == pytest.approx(coherence, abs=.0001)
    rx, obs = placed('2031', at, 2.63, partial)
    assert obs is not None and obs.kind == Kind.SHORT
    assert (obs.value, obs.walsh_peak, obs.provisional) == (53, 64, partial)
    assert abs(obs.stream_sample / FS - at) < .003
    assert rx.packet_read['reason'] == 'decoded'


@pytest.mark.parametrize('at,max_peak', [(66.004, 30), (71.605, 30), (73.2, 30)])
def test_2031_fade_word_and_floor_are_not_shorts(at, max_peak):
    rx, obs = placed('2031', at, 2.63)
    assert rx.packet_read['value'] is None
    assert rx.packet_read['walsh_peak'] <= max_peak
    assert obs is None or obs.kind != Kind.SHORT


def test_2031_short_raster_delivers_both_weak_copies_and_the_later_control():
    from hfmodem.shrike.p4turn import State
    from hfmodem.tests.shrike.test_p4turn import machine

    rx = P4Receiver(FS, admitted=True, cfo_hz=2.63)
    rx.note_emission(Emission(Kind.WORD, Role.IRS, round((61.004 - .33) * FS), .244))
    turn, _ = machine()
    turn.state = State.IRS_ANSWER
    received, answers = [], []
    audio = capture('2031')
    for at in (61.004, 62.254, 63.504, 64.754, 66.004, 67.254):
        origin, early, end = (round(t * FS) for t in (at - .05, at + .81, at + .85))
        records = rx.read_partial(audio[origin:early], origin, deadline=early)
        records += rx.read_window(audio[origin:end], origin, whole=True)
        for record in records:
            turn.observe(record)
            obs = record.observation
            if obs.kind == Kind.SHORT and not obs.provisional:
                received.append((obs.stream_sample / FS, obs.value))
        proposal = turn.decide(end)
        if proposal is not None:
            sent = turn.keyed()
            answers.append(sent.word)
            rx.note_emission(Emission(sent.kind, Role.IRS, sent.stream_sample, .244))
        rx.new_cycle(at=end)
    assert [value for _, value in received] == [49, 49, 53, 53, 53]
    assert [at for at, _ in received] == pytest.approx(
        [61.004, 62.254, 63.504, 64.754, 67.254], abs=.003)
    assert answers == ['X27', 'X27', 'U27', 'U27', 'U27']


@pytest.mark.parametrize('at', [34.744, 93.494, 149.743])
def test_2202_turn_two_header_is_b(at):
    rx, obs = placed('2202', at, 0.)
    assert (obs.value, obs.walsh_peak, obs.header) == (53, 64, 'B')
    assert rx.packet_read['header'] == 'B'


@pytest.mark.parametrize('arm,count', [('2202', 5), ('2229', 8)])
def test_recorded_unknown_headers_keep_the_measured_flag_and_header_knee(arm, count):
    probe = json.loads(require_input(ROOT / 'captures' / f'onair-0930-{arm}' / 'p4-probe.json').read_text())
    rows = {r['onset_sample']: r for r in probe['heard']
            if r.get('kind') == 'short' and r.get('header') == '?'}
    assert len(rows) == count
    for at, row in rows.items():
        rx, obs = placed(arm, at / FS, row['cfo_hz'])
        # longrun2.txt: the first 2229 v53 is A=16/B=5; 2202 at 80.997
        # is A=8/B=13, named B only by that offline script's 12-step knee.
        expected = ('A' if arm == '2229' and at == 1787703 else
                    '?' if arm == '2202' and at == 3887866 else 'B')
        assert obs is not None and obs.header == expected, (at / FS, rx.packet_read)
        assert obs.value == row['value']


@pytest.mark.parametrize('envelope,peak,coherence,accepted', [
    (False, 64, .20, True), (False, 64, .1999, False),
    (False, 62, .2499, False), (False, 62, .25, True),
    (False, 48, .25, True), (False, 46, .5, False),
    (True, 64, .20, True), (True, 62, .25, True),
    (True, 48, .25, True), (True, 46, .5, False),
])
def test_only_a_full_peak_gets_the_lower_coherence_floor(monkeypatch, peak, coherence, accepted, envelope):
    rx = P4Receiver(FS, admitted=True, cfo_hz=0.)
    rx.clock = Clock(Expect.PACKET, 2400, .33, 1.25, 0.)
    field = dict(p4_value=53 if peak >= 48 else None, p4_walsh_peak=peak,
                 coherence=coherence, p4_hdr='B', first_chip=2400)
    monkeypatch.setattr(p4oracle, 'robust_at', lambda *a: field)
    monkeypatch.setattr(p4oracle, 'placed_despread', lambda *a, **kw: (.5, np.empty(0), None))
    monkeypatch.setattr(rx, '_bursts', lambda *a, **kw: [(.05, .76)] if envelope else [])
    obs = rx._packet(np.zeros(FS), 0)
    assert (obs is not None and obs.value is not None) == accepted
    if accepted:
        assert rx.packet_read['reason'] == 'decoded'


@pytest.mark.parametrize('arm,at,cfo,value,header,peak', [
    ('0928-2255', 73.503, 0., 5, 'B', 54),
    ('0928-2301', 43.504, 0., 13, 'B', 58),
    ('0928-2301', 164.778, 0., 52, 'B', 54),
    ('0928-2301', 204.752, 0., 55, 'B', 58),
    ('0929-2352', 62.256, 13.69175975557447, 57, 'A', 64),
    ('0930-0156', 38.505, 3.101209988408553, 5, 'B', 54),
])
def test_envelope_recovers_recorded_shorts_below_placed_coherence(arm, at, cfo, value, header, peak):
    rx, obs = placed(arm, at, cfo)
    assert rx.packet_read['reason'] == 'below-coherence'
    assert obs is not None and obs.kind == Kind.SHORT
    assert (obs.value, obs.header, obs.walsh_peak) == (value, header, peak)
    assert abs(obs.stream_sample / FS - at) < .035


@pytest.mark.parametrize('header', ['A', 'B'])
def test_every_walsh_value_retains_its_header_and_value(header):
    for value in range(64):
        steps = np.asarray(p4robusttx.short_dibits(value, header))
        for turn in range(4):
            symbols = p4robusttx.symbols((steps + turn) % 4)
            assert p4oracle.robust_symbols(symbols, 85) == dict(
                p4_value=value, p4_walsh_peak=64, p4_hdr=header)
