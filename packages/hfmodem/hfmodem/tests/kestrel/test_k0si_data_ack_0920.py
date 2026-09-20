# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""K0SI's full short ACK, classified by its answer to the same pending DATA query."""
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from .test_data_over_gate import _connected

LINK = ('W9SSJ', 'K0SI', '2300')
PAIRS = ((64, 67), (58, 92), (77, 91), (48, 64),
         (65, 71), (38, 92), (81, 85), (58, 98))


def native(name):
    path = (Path(__file__).with_name('fixtures') /
            'k0si-data-ack-0920' / (name + '.wav'))
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, x = wavfile.read(path)
    assert fs == MK.FS and x.dtype == np.int16
    return x.astype(float) / 32768


def pending():
    hs, io = _connected()
    hs.called = 'K0SI'
    hs._peer_offset = -9.0 / (MK.FS / MK.NFFT)
    hs.turn = VA._TURN_OURS
    hs._over = 1
    hs.send(b'A' * 89 + b'B' * 55)
    return hs, io


def feed(hs, audio, chunk=4096):
    for at in range(0, len(audio), chunk):
        hs.on_rx_stream(audio[at:at + chunk])


def test_native_query_independently_confirms_the_measured_short_reply(monkeypatch):
    hs, io = pending()
    monkeypatch.delitem(VF.OVER_CONTINUE_RESPONDER_BY_LINK, LINK)
    old = hs._tx_pending
    reply = native('intact')
    assert not hs._peer_over_continue(reply)
    assert hs._unclassified_continue[2] == PAIRS
    assert hs._tx_pending == old
    assert hs._probe_intermediate_answer()
    answer = native('query')
    kind = VF.SESSION_INTERMEDIATE_QUERY_ANSWER
    tones = np.asarray(VF.handshake_tones(hs.called, kind))
    track = VA._top3_track(answer, band=hs._band, bin_offset=hs._grid)
    heard, _ = VA._payload_fit(answer, kind, tones, track, hs._band)
    assert np.array_equal(heard[1:], tones[1:])  # All 31, not just a detector vote.
    assert hs._peer_intermediate_query_answer(answer, track)
    hs._took_intermediate_query_answer()
    assert hs._tx_pending[1] == old[1] + 1
    assert (*LINK, PAIRS) in hs._confirmed_continue_pairs
    assert hs._peer_over_continue(reply)


@pytest.mark.parametrize('chunk', [512, 4096, 4800, 8192])
def test_native_known_reply_advances_once_without_a_query(chunk):
    hs, io = pending()
    assert VF.OVER_CONTINUE_RESPONDER_BY_LINK[LINK] == PAIRS
    x = native('intact')
    feed(hs, x, chunk)
    assert hs._tx_pending[1] == 3 and not hs._txq
    assert hs._tx_pending[0].startswith(b'B' * 55)
    assert len(io.sent) == 2 and hs._intermediate_query_attempts == 0
    hs.on_rx_audio(x)
    assert hs._tx_pending[1] == 3 and len(io.sent) == 2


@pytest.mark.parametrize('field,value', [('caller', 'N0XYZ'), ('called', 'KC9GHZ'),
                                        ('bw', '2750'), ('bw', '500')])
def test_native_short_reply_does_not_authorize_other_links(field, value):
    hs, io = pending()
    setattr(hs, field, value)
    old = hs._tx_pending
    assert not hs._peer_over_continue(native('intact'))
    assert hs._tx_pending == old and len(io.sent) == 1


def test_distorted_earlier_reply_still_requires_query_confirmation():
    hs, io = pending()
    hs._peer_offset = -9.3 / (MK.FS / MK.NFFT)
    old = hs._tx_pending
    feed(hs, native('distorted'))
    assert hs._tx_pending == old and len(io.sent) == 1
    assert not hs._peer_over_continue(native('distorted'))
