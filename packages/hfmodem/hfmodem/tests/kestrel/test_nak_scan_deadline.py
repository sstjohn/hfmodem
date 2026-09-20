# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""The 22:48 K0SI queries were answered, but two NAKs were scanned too late."""
from pathlib import Path

import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as VA
from hfmodem.kestrel.arq import phy
from .test_data_over_gate import _connected


@pytest.mark.parametrize('name', ['query-8', 'query-10', 'query-11'])
@pytest.mark.parametrize('chunk', [512, 960, 4096, 4800, 6000, None])
def test_native_query_nak_is_answered_once_before_its_listening_gap_expires(name, chunk):
    path = Path(__file__).with_name('fixtures') / 'k0si-2248-0919' / (name + '.wav')
    if not path.is_file():
        pytest.skip(f'optional recording is not included: {path}')
    fs, audio = wavfile.read(path)
    assert fs == 48000
    audio = audio.astype(float) / 32768
    hs, io = _connected()
    hs.called = 'K0SI'
    hs._peer_offset = -8.2 / (fs / 2048)
    hs.turn = VA._TURN_OURS
    hs.send(b'A' * 79)
    assert hs._query_final_answer()
    pending, sent = hs._tx_pending, len(io.sent)
    # Polling can deliver uneven groups of device callbacks. The penultimate
    # scan here falls a few ms before the NAK tail; the old .5s next-scan delay
    # missed the complete audio at 1.46s and rejected the frame at 1.90s.
    stops = ([int(t * fs) for t in (.35, .75, .90, 1.05, 1.2, 1.375, 1.46, 1.90, 2.)]
             if chunk is None else list(range(chunk, len(audio), chunk)) + [len(audio)])
    start = 0
    for stop in stops:
        hs.on_rx_stream(audio[start:stop])
        start = stop
    assert hs._tx_retries == 1
    assert len(io.sent) == sent + 1
    body, over, level = hs._tx_pending
    assert (over, level) == (pending[1], 1)
    assert phy.vara_payload(body, caller=hs.caller, body_len=len(body)) + b''.join(hs._txq) == b'A' * 79
    assert 0 <= hs._nak_held <= VA._GRANT_FRESH_S
    assert not any('listening gap has gone' in msg for msg in io.msgs)
    hs.on_rx_audio(audio)
    assert len(io.sent) == sent + 1
