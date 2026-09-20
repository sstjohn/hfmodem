# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A muted lead can be replaced only by an independently measured full-link tail."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_over_gate import _connected

FIXTURES = Path(__file__).with_name('fixtures') / 'head-cut-continue'
PAIRS = ((64, 67), (25, 89), (30, 88), (47, 73),
         (42, 88), (71, 103), (56, 76), (73, 79))


def recorded(name):
    path = FIXTURES / (name + '.wav')
    if not path.is_file():
        pytest.skip(f'recorded head-cut continue fixture unavailable: {path}')
    fs, x = wavfile.read(path)
    assert fs == MK.FS and x.ndim == 1
    return x.astype(float) / 32768 if x.dtype == np.int16 else x.astype(float)


def pending():
    hs, io = _connected(bw='2750')
    hs.turn = VA._TURN_OURS
    hs._over = 1  # The short login/proposal over has already been acknowledged.
    hs.send(b'A' * 89 + b'B' * 89 + b'C' * 89 + b'last')
    assert hs._tx_pending[1] == 2 and len(hs._txq) == 3
    return hs, io


def synth(pairs=PAIRS, mute=(0,)):
    x = MK.synth_tone_pairs(pairs)
    for i in mute:
        x[i * MK.HOP:(i + 1) * MK.HOP + MK.WOLA_N] = 0
    return np.pad(x, (4800, 4800))


def strict(x):
    return VA._cont_plateau(x, band=MK.band_for('2750'))


def test_native_night_lead_is_deaf_and_seven_pairs_match_stock():
    hs, _ = pending()
    x = recorded('night-caller-over2')
    assert strict(x) == 0
    assert VF.OVER_CONTINUE_RESPONDER_BY_LINK[('W9SSJ', 'KC9GHZ', '2750')] == PAIRS
    assert hs._peer_over_continue(x)


@pytest.mark.parametrize('name', [
    'clean-caller-over2-stock-continue', 'clean-caller-over3-stock-continue',
    'ack-fault-caller-over2-stock-continue', 'ack-fault-caller-over3-stock-continue',
])
def test_four_host_proven_stock_replies_remain_strict_positives(name):
    hs, _ = pending()
    x = recorded(name)
    track = VA._top3_track(x, band=MK.band_for('2750'))
    at, width = VA._widest(VA._cont_held(*track[:2], VA._live_track(x)))
    assert width >= VA._CONT_PLATEAU
    pairs = MK.demod_tone_pairs(x[(at + width // 2) * VA._ACK_GRID:], 8,
                               band=MK.band_for('2750'))
    assert tuple(pairs) == PAIRS
    assert hs._peer_over_continue(x)
    assert not hs._peer_head_cut_continue(x, track)


def test_loud_wrong_lead_is_not_receiver_mute():
    hs, _ = pending()
    x = synth(((40, 90),) + PAIRS[1:], mute=())
    assert strict(x) == 0
    assert not hs._peer_over_continue(x)


@pytest.mark.parametrize('mute', [(0, 2), (0, 4), (0, 7)])
def test_a_missing_interior_or_final_symbol_cannot_be_waived(mute):
    hs, _ = pending()
    assert not hs._peer_over_continue(synth(mute=mute))


def test_strong_wrong_tail_and_incomplete_tail_do_not_advance():
    hs, io = pending()
    pairs = list(PAIRS)
    pairs[4] = (31, 97)
    bad = synth(tuple(pairs))
    before = hs._tx_pending, list(hs._txq), hs.progress, len(io.sent)
    hs.on_rx_audio(bad)
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == before
    assert not hs._peer_over_continue(synth()[:4800 + 6 * MK.HOP])


@pytest.mark.parametrize('field,value', [
    ('caller', 'OTHER'), ('called', 'KB5LZK'), ('bw', '2300'),
    ('state', VA.VaraState.CONNECTING), ('role', 'responder'),
    ('turn', VA._TURN_PEER), ('turn', VA._TURN_ASKED),
    ('_tx_pending', None), ('_txq', []),
])
def test_head_cut_fallback_requires_full_link_and_sending_state(field, value):
    hs, _ = pending()
    setattr(hs, field, value)
    assert not hs._peer_over_continue(recorded('night-caller-over2'))


def test_short_pending_body_is_not_an_intermediate_full_frame():
    hs, _ = pending()
    hs._tx_pending = (phy.vara_body(b'short', callsign=hs.caller), 2, 3)
    assert not hs._peer_over_continue(recorded('night-caller-over2'))


def test_known_nak_and_ack_vetoes_still_precede_the_fallback(monkeypatch):
    hs, _ = pending()
    x = recorded('night-caller-over2')
    monkeypatch.setattr(hs, '_peer_nak', lambda *a, **kw: True)
    assert not hs._peer_over_continue(x)
    monkeypatch.setattr(hs, '_peer_nak', lambda *a, **kw: False)
    monkeypatch.setattr(VA, '_ack_plateau', lambda *a, **kw: VA._ACK_PLATEAU)
    assert not hs._peer_over_continue(x)


@pytest.mark.parametrize('name', ['request-train-9s', 'request-train-24s'])
def test_native_nonreply_request_train_intervals_do_not_advance(name):
    hs, io = pending()
    x = recorded(name)
    # These intervals fooled an unconditional seven-symbol shape waiver. They
    # are not independently classified as pure noise; keep that provenance.
    before = hs._tx_pending, list(hs._txq), hs.progress, len(io.sent)
    for at in range(0, len(x) - MK.FS + 1, MK.FS // 4):
        y = x[at:at + MK.FS]
        assert not hs._peer_over_continue(y)
        hs.on_rx_audio(y)
    assert (hs._tx_pending, hs._txq, hs.progress, len(io.sent)) == before


@pytest.mark.parametrize('chunk', [512, 4096, 4800])
def test_native_stream_reply_retires_pending_once_and_bracket_cannot_repeat(chunk):
    hs, io = pending()
    x = recorded('night-caller-over2')
    for at in range(0, len(x), chunk):
        hs.on_rx_stream(x[at:at + chunk])
    assert hs._over == 3 and len(io.sent) == 2
    assert hs._tx_pending[1] == 3 and hs._txq == [b'C' * 89, b'last']
    progress = hs.progress
    hs.on_rx_audio(x)
    assert hs._over == 3 and len(io.sent) == 2 and hs.progress == progress


def test_native_bracket_only_reply_still_advances_one_body():
    hs, io = pending()
    hs.on_rx_audio(recorded('night-caller-over2'))
    assert hs._over == 3 and len(io.sent) == 2 and hs._tx_pending[1] == 3


def test_stream_owner_does_not_let_intact_continue_bracket_advance_again():
    hs, io = pending()
    x = recorded('clean-caller-over2-stock-continue')
    for at in range(0, len(x), 4096):
        hs.on_rx_stream(x[at:at + 4096])
    assert hs._over == 3 and len(io.sent) == 2
    hs.on_rx_audio(x)
    assert hs._over == 3 and len(io.sent) == 2
