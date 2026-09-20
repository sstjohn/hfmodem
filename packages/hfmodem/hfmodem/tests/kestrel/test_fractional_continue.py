# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A real gateway's full-body ACK must advance DATA at a half-bin offset.

KB5LZK's carriers lay halfway between our bins on 2026-09-11 and its request for
the next body frame was discarded. What answered it then was a search of the half
bin either side, inside this one recogniser; what answers it now is the link's own
measured frequency, taken off the connect-response that had already fixed
``_peer_shift`` and handed to every reader alike  [vara_arq, _note_peer_offset].
So the offset is set here the way a live session sets it, and the recording is
still what decides whether the frame is read.
"""
from pathlib import Path
import wave

import numpy as np
import pytest

from hfmodem.kestrel.vara import vara_arq as VA, vara_frames as VF, vara_mfsk as MK
from hfmodem.tests.kestrel.test_data_over_gate import _connected
from hfmodem.tests.kestrel.test_turn_law import _off_frequency

FIXTURE = Path(__file__).parent / 'fixtures' / 'kb5-body-continue'


def recorded(name):
    path = FIXTURE / f'{name}.wav'
    if not path.exists():
        pytest.skip(f'KB5LZK continuation recording absent: {path}')
    with wave.open(str(path)) as wav:
        assert wav.getframerate() == MK.FS and wav.getnchannels() == 1
        return np.frombuffer(wav.readframes(wav.getnframes()), '<i2') / 32768.0


def station():
    hs, io = _connected(bw='2750')
    hs.called = 'KB5LZK'
    # Both learned from the actual addressed connect-response: the whole carriers
    # it arrived off by, and the fraction of one under them.
    hs._peer_shift = -1
    hs._peer_offset = 0.5
    return hs, io


def test_recorded_full_body_reply_is_recognized_at_the_connected_frequency():
    hs, _ = station()
    x = recorded('body-response')
    assert VA._cont_plateau(x, -1, band=hs._band) == 0
    assert hs._peer_over_continue(x)


@pytest.mark.parametrize('block', [512, 1500, 4096, 4800])
def test_recorded_reply_retires_only_the_pending_body_and_keys_the_next(block):
    hs, io = station()
    hs.turn = VA._TURN_OURS
    hs._txq = [b'A' * 89, b'B' * 89, b'C' * 89, b'D' * 34]
    hs._tx_data_over()
    original = hs._tx_pending
    x = recorded('body-response')
    for i in range(0, len(x), block):
        hs.on_rx_stream(x[i:i + block])
        if len(io.sent) == 2:
            break  # A real transport now keys and mutes this receive window.
    assert any('rx over-continue' in m for m in io.msgs), io.msgs
    assert len(io.sent) == 2, io.msgs
    assert hs._tx_pending != original
    assert hs._tx_pending[0][:89] == b'B' * 89
    assert hs._tx_pending[1] == original[1] + 1
    assert hs._txq == [b'C' * 89, b'D' * 34]
    assert hs._tx_retries == 0
    # Delivered promptly after the response's last symbol, not the 10 s retry.
    assert (i + len(x[i:i + block])) / MK.FS < 0.8


@pytest.mark.parametrize('name', ['short-turn-ack', 'post-response-noise'])
def test_recorded_final_control_and_noise_do_not_advance_as_continues(name):
    hs, _ = station()
    assert not hs._peer_over_continue(recorded(name))


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
@pytest.mark.parametrize('offset,shift', [(-0.5, -1), (-0.5, 0), (0.5, 0), (0.5, 1)])
def test_fractional_continue_preserves_all_bandwidths(bw, offset, shift):
    hs, _ = _connected(bw=bw)
    hs._peer_shift = shift
    hs._peer_offset = offset
    x = _off_frequency(MK.synth_tone_pairs(VF.over_continue('W9SSJ', bw)),
                       offset + shift)
    x = np.pad(x, (4800, 4800))
    assert hs._peer_over_continue(x)


@pytest.mark.parametrize('offset,shift', [(-0.5, -1), (0.5, 1)])
def test_fractional_nak_keeps_the_pending_frame(offset, shift):
    hs, _ = _connected(bw='2300')
    hs._peer_shift = shift
    hs._peer_offset = offset
    x = _off_frequency(MK.synth_tone_pairs(VF.nak('W9SSJ', '2300')[1]),
                       offset + shift)
    assert not hs._peer_over_continue(np.pad(x, (4800, 4800)))


def test_fractional_reader_does_not_search_an_unrelated_integer_offset():
    hs, _ = _connected(bw='2750')
    x = _off_frequency(MK.synth_tone_pairs(VF.over_continue('W9SSJ', '2750')), 2)
    assert not hs._peer_over_continue(np.pad(x, (4800, 4800)))


@pytest.mark.parametrize('bw', ['500', '2300', '2750'])
def test_a_carrier_no_measurement_produced_is_not_an_ack(bw):
    """NS0A's recording under an ARTIFICIAL whole-carrier shift, which is what the
    fixture was cut for. It used to be refused by a power comparison between two
    grids the reader was trying; there are no two grids now — the link has one,
    taken off the frame that named it — so what refuses this is the same thing
    that refuses every other unmeasured hypothesis: nothing ever proposes it."""
    hs, _ = _connected(bw=bw)
    hs._peer_shift = 1
    assert hs._peer_offset is None
    assert not hs._peer_over_continue(recorded('adjacent-grid-negative'))
