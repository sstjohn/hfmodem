# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Mixed-protocol traffic is physically valid without owning our session."""
import hashlib
from copy import deepcopy

import numpy as np
import pytest

from hfmodem.kestrel.rx import bw500_locator as locator, varahf500 as rx
from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as arq
from hfmodem.tests.kestrel import corpora


_WINDOWS = (
    (4954800, (-18, 237), '2c415360e5f7106209b4e97307bd2ecdd10333cd6c34b17241664cac0ee8ded665c2690504db07878b8a1899b654'),
    (5418032, (4, 226), 'fd990811832af21c6b604c8f914aa4a4f44a84d3976bf3cafbbf171ebecb0f9d69adfefe674c7711ccab5f95d9fd'),
)
_SHA256 = 'fbca871ec1ddb7770bf4edb25edcb30d2f943b7b94bd39baba7f67730597e148'


@pytest.mark.parametrize('failure', ['unknown_end', 'missing_partner', 'bad_partner'])
def test_phy_never_exposes_first_frame_of_incomplete_over(monkeypatch, failure):
    first = _frame(_WINDOWS[0])
    result = rx.DecodeResult([first], [first])
    if failure == 'unknown_end':
        result.observation_complete = False
    elif failure == 'missing_partner':
        first.frames_in_burst = 2
    else:
        second = deepcopy(first)
        second.crc_ok = False
        result.frames.append(second)
        result.data_frames.append(second)
    monkeypatch.setattr(rx, 'decode_stream', lambda samples, **kwargs: result)
    assert phy._decode_base(np.zeros(100), '500') is None


class _IO(arq.VaraIO):
    def __init__(self):
        self.events = []

    def key(self, on): self.events.append(('key', on))
    def tx(self, samples): self.events.append(('tx', len(samples)))
    def pending(self): self.events.append(('pending',))
    def connected(self, *args): self.events.append(('connected', args))
    def log(self, message): self.events.append(('log', message))
    def data(self, payload): self.events.append(('data', payload))


def _station(role, state=arq.VaraState.CONNECTED, turn=arq._TURN_PEER):
    station = arq.VaraStationHandshake(['W9SSJ'], _IO(), bw='500')
    station.role = role
    station.state = state
    station.turn = turn
    station.caller, station.called = ('W9SSJ', 'W1AW') if role == 'initiator' else ('W1AW', 'W9SSJ')
    return station


def _frame(window):
    start, onsets, body = window
    return rx.FrameResult(start=start, onset=onsets, c0=9, self_consistency=.8,
                          crc_ok=True, frame_bytes=bytes.fromhex(body), level=4,
                          burst_columns=rx.NSYM + 9)


def _snapshot(station):
    fields = ('state', 'role', 'turn', 'caller', 'called', '_delivered', '_resent',
              '_resent_held', '_answer_owed', '_owed_block', '_owed_recovery',
              '_peer_over', '_peer_delivery_open', '_held_answer', '_txq',
              '_reacks', '_undecoded', '_keyed_bodies')
    return {name: deepcopy(getattr(station, name)) for name in fields}


@pytest.fixture(scope='module')
def mixed_recording():
    path = corpora.REGRESS_FIXTURES / 'pos_p1_twosided_14110.wav'
    if not path.exists():
        pytest.skip('shared regression corpus not present')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _SHA256
    x = corpora.wav_mono(path)
    assert len(x) == 5760000
    return x


@pytest.mark.corpus
@pytest.mark.parametrize('window', _WINDOWS, ids=['103.225000', '112.875667'])
def test_mixed_recording_preserves_exact_vara_crc_bytes_and_stagger(mixed_recording, window):
    start, expected_onsets, expected_body = window
    cols, offsets, _ = rx._demod_aligned(mixed_recording, start)
    gh = rx._frame_gh(cols, 0)
    _, c0, crc, body, setup = rx._decode_cells(gh, 0, 9)
    assert tuple(offsets.values()) == expected_onsets
    assert c0 == 9
    assert crc
    assert body == bytes.fromhex(expected_body)
    assert not setup


@pytest.mark.parametrize('window', _WINDOWS, ids=['103.225000', '112.875667'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('mode', ['listening', 'our-turn'])
def test_exact_crc_frames_do_not_bypass_session_or_turn_ownership(window, role, mode):
    station = _station(role, state=arq.VaraState.LISTENING if mode == 'listening' else arq.VaraState.CONNECTED,
                       turn=arq._TURN_OURS)
    frame = _frame(window)
    station._resend_decoded = rx.DecodeResult([frame], [frame])
    before = _snapshot(station)
    assert station._peer_data_over_500(np.empty(0), 4.4) == []
    assert _snapshot(station) == before
    assert not station.io.events


@pytest.mark.parametrize('window', _WINDOWS, ids=['103.225000', '112.875667'])
@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_expired_physical_result_cannot_answer_current_peer_window(window, role):
    station = _station(role)
    frame = _frame(window)
    candidate = locator.Candidate(4, frame.start, .7, frame.start + 210000, frame.start)
    over = locator.Over(candidate, generation=station._front_generation, frames=[frame],
                        status='closed', end=frame.start + 210000)
    station._base_ready = [over]
    station._front_heard = over.end + rx.FS + 1
    before = _snapshot(station)
    assert not station._take_located_500()
    assert not station._base_ready
    assert _snapshot(station) == before
    assert not station.io.events


@pytest.mark.parametrize('role', ['initiator', 'responder'])
def test_discontinuity_discards_queued_crc_success_and_changes_generation(role):
    station = _station(role)
    frame = _frame(_WINDOWS[0])
    candidate = locator.Candidate(4, frame.start, .7, frame.start + 210000, frame.start)
    over = locator.Over(candidate, generation=station._front_generation, frames=[frame],
                        status='closed', end=frame.start + 210000)
    station._base_ready = [over]
    generation = station._front_generation
    before = _snapshot(station)
    station.rx_discontinuity()
    assert station._front_generation > generation
    assert not station._base_ready
    station._base_ready = [over]
    assert not station._take_located_500()
    assert not station._base_ready
    assert _snapshot(station) == before
    assert not station.io.events


@pytest.mark.parametrize('role', ['initiator', 'responder'])
@pytest.mark.parametrize('field,value', [('caller', ''), ('called', ''), ('role', None),
                                        ('bw', '2300'), ('state', arq.VaraState.LISTENING),
                                        ('turn', arq._TURN_OURS)])
def test_missing_ownership_context_discards_queued_good_frame(role, field, value):
    station = _station(role)
    frame = _frame(_WINDOWS[0])
    candidate = locator.Candidate(4, frame.start, .7, frame.start + 210000, frame.start)
    station._base_ready = [locator.Over(candidate, generation=station._front_generation,
                                      frames=[frame], status='closed', end=frame.start + 210000)]
    setattr(station, field, value)
    before = _snapshot(station)
    assert not station._base_window_owned()
    assert not station._take_located_500()
    assert not station._base_ready
    assert _snapshot(station) == before
    assert not station.io.events
