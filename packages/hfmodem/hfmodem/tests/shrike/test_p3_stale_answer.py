# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""KB5LZK answers returning after interference must reach the live reader."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.shrike import arq, onair
from hfmodem.tests.shrike.test_entry_answer import _Session

FIXTURE = Path(__file__).parent / 'fixtures' / 'stale-answer-kb5lzk-0920'
if not (FIXTURE / "provenance.json").is_file():
    pytest.skip("KB5LZK stale-answer recordings are not included in this distribution",
                allow_module_level=True)
META = json.loads((FIXTURE / 'provenance.json').read_text())


def scene(hold, age=None):
    record = next(r for r in META['records'] if r['hold'] == hold)
    fs, raw = wavfile.read(FIXTURE / f'hold_{hold}.wav')
    assert fs == onair.FS
    tx = record['tx']
    grid = SimpleNamespace(
        sending=True, keyed_slot=tx['slot'],
        _p3_keyed_reply=(tx['slot'], tx['audio_start'], tx['audio_end'], 60000),
        rx_due_in=lambda start, end: None, rx_due=lambda slot: 0)
    session = _Session()
    session.rx.new_cycle()
    anchor = META['last_answer']
    if age is not None:
        projected = anchor + ((record['end'] - anchor) // 60000) * 60000
        anchor = projected - age * 60000
    session.rx._p3_answer_at = anchor
    session.rx.p3_receive_offset_hz = META['offset_hz']
    return session, grid, raw.astype(np.float64) / 32768, record['start']


@pytest.mark.parametrize('hold,word', [(243, arq.CS_ACK), (245, arq.CS_ACK),
                                      (280, arq.CS_ACK), (286, arq.CS_REQUEST)])
def test_recorded_returning_answer_is_delivered_once(hold, word):
    s, g, audio, origin = scene(hold)
    heard, _ = s.rx.control_signal_in(audio, origin, g)
    assert heard == word
    assert len(s.events) == 1
    assert s.rx._p3_answer_at >= origin
    assert s.rx.control_signal_in(audio, origin, g) == (None, None)
    assert len(s.events) == 1
    # Even another cycle cannot credit the same recording again.
    s.rx.new_cycle()
    assert s.rx.control_signal_in(audio, origin, g)[0] is None
    assert len(s.events) == 1


@pytest.mark.parametrize('age', [8, 9, 27, 74])
def test_age_does_not_disable_a_current_keyed_answer(age):
    s, g, audio, origin = scene(280, age=age)
    assert s.rx.control_signal_in(audio, origin, g)[0] == arq.CS_ACK


@pytest.mark.parametrize('hold', [220, 230, 240])
def test_recorded_interference_does_not_refresh_answer(hold):
    s, g, audio, origin = scene(hold)
    assert s.rx.control_signal_in(audio, origin, g)[0] is None
    assert s.rx._p3_answer_at == META['last_answer']
    assert not s.events


@pytest.mark.parametrize('reason', ['unkeyed', 'mismatched', 'skipped',
                                   'truncated', 'disconnected', 'noise'])
def test_recovery_preserves_reply_ownership_and_decode_guards(reason):
    s, g, audio, origin = scene(280)
    if reason == 'unkeyed':
        g._p3_keyed_reply = None
    elif reason == 'mismatched':
        g.keyed_slot += 1
    elif reason == 'skipped':
        origin += 60000
    elif reason == 'truncated':
        audio = audio[:onair.P3_CS_N - 1]
    elif reason == 'disconnected':
        s.host.arq.state = onair.State.DISCONNECTED
    elif reason == 'noise':
        audio = np.random.default_rng(20260920).normal(0, .05, audio.size)
    assert s.rx.control_signal_in(audio, origin, g)[0] is None
    assert s.rx._p3_answer_at == META['last_answer']
    assert not s.events


def test_teardown_still_hears_returning_reply():
    s, g, audio, origin = scene(286)
    s.host.arq.state = onair.State.DISCONNECTING
    assert s.rx.control_signal_in(audio, origin, g)[0] == arq.CS_REQUEST


@pytest.mark.parametrize('seq', [0, 1])
def test_recovered_word_preserves_ack_parity_and_inflight_data(seq, capsys):
    s, g, audio, origin = scene(280)
    a = s.host.arq
    a._next_seq = seq
    a.on_host_data(b'hello')
    a.on_cycle()
    packet = a._inflight
    packet.retries = 27
    assert s.rx.control_signal_in(audio, origin, g)[0] == arq.CS_ACK
    if seq == 0:
        assert a._inflight is None
    else:
        # Physical CS1 requests a repeat for odd sequence numbers.
        assert a._inflight is packet
        assert packet.payload == b'hello'
        assert packet.repeats == 1
    assert a.state == onair.State.CONNECTED
    assert capsys.readouterr().out.count('answer recovered after 64 cycles') == 1
    assert s.rx.control_signal_in(audio, origin, g)[0] is None
    assert len(s.events) == 1
    assert 'answer recovered' not in capsys.readouterr().out


@pytest.mark.parametrize('change', ['role', 'protocol', 'disconnect'])
def test_old_answer_does_not_survive_session_context_change(change):
    s, _, _, _ = scene(280)
    if change == 'role':
        s.host.arq.role = arq.IRS
    elif change == 'protocol':
        s.host.protocol = onair.Protocol.PACTOR1
    else:
        s.host.arq.state = onair.State.DISCONNECTED
    s.rx.new_cycle()
    assert s.rx._p3_answer_at is None
