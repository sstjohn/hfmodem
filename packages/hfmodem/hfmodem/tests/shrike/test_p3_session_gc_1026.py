# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""WS8EOC 2026-10-27 04:11/04:12 UTC: capture runs between slots too."""
import gc

import pytest

from hfmodem.tests.shrike import test_call_cadence as cadence


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_no_collection_until_capture_closes(tmp_path, monkeypatch, held, enabled):
    incoming = gc.isenabled()
    clocks, collections = [], []
    original = cadence._Clock

    def clock():
        result = original()
        clocks.append(result)
        return result

    def collected(phase, info):
        if phase == 'start':
            collections.append(any(c.opened and not c.closed for c in clocks))

    monkeypatch.setattr(cadence, '_Clock', clock)
    gc.enable() if enabled else gc.disable()
    gc.callbacks.append(collected)
    try:
        cadence.session(tmp_path, monkeypatch, '--max-cycles', '4',
                        answer_from=1 if held else None)
        assert gc.isenabled() is enabled
        assert clocks and all(c.opened and c.closed for c in clocks)
        assert not any(collections), 'collection blocked the running capture callback'
    finally:
        gc.callbacks.remove(collected)
        gc.enable() if incoming else gc.disable()


@pytest.mark.parametrize('attempt', ['first', 'second'])
def test_recent_ws8eoc_changeover_gets_ack_without_collection_or_overlap(
        tmp_path, monkeypatch, attempt):
    import hashlib
    import json
    from pathlib import Path

    from scipy.io import wavfile

    from hfmodem.shrike import arq
    from hfmodem.tests.shrike import test_late_entry_loop as loop

    folder = Path(__file__).with_name('fixtures') / 'ws8eoc-1026'
    meta = json.loads((folder / f'{attempt}.json').read_text())
    fs, raw = wavfile.read(folder / f'{attempt}.wav')
    assert fs == loop.FS
    assert hashlib.sha256(raw.tobytes()).hexdigest() == meta['pcm_sha256']
    pcm = raw.astype(float) / 32768
    collections = []

    def collect(generation):
        collections.append(generation)
        raise AssertionError('full GC would stall the live callback between slots')

    def start_p3(peer, entry_at):
        # Both live first heads arrived about 0.96 s after entry. Replay the
        # retained CRC-valid changeover on the peer's own 1.25 s clock.
        peer.p3_phase = entry_at + round(.96 * fs)
        peer.audio[peer.p3_phase - 3840:] = 0
        for k in range(24):
            at = peer.p3_phase + k * loop.SLOT - 3840
            peer.audio[at:at + len(pcm)] = pcm
            peer.responses.append((at + 3840, at + 3840 + round(.84 * fs)))

    monkeypatch.setattr(loop.LatePeer, 'start_p3', start_p3)
    monkeypatch.setattr(gc, 'collect', collect)
    got = loop.run_late(monkeypatch, tmp_path, 1, False,
                        options=('--no-long-cycle', '--no-p3-fallback'))
    peer = got['peer']
    assert not collections
    assert bytes(got['host'].channel(got['host'].ptchn).rx) == b'RMS'
    assert len(peer.entries) == 1, 'entry retransmission covered the peer changeover'
    assert any(ci == arq.CS_ACK and errors == 0 for _, _, ci, errors in peer.controls)
    for first, end, _, _ in peer.controls:
        # Include the local 40 ms PTT lead, not just emitted audio.
        assert all(end <= a or first - round(.04 * fs) >= b for a, b in peer.responses)
