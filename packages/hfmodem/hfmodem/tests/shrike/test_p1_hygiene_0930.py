# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""September 29's missed P1 words, late packet 1 and pending announcement.

The September 29 replay pins two missed control words at 18.605 and 22.355 s,
two final-listen answers at 28.615 and 31.116 s, and packet-one preparation
before the first candidate, without an unrequested local speed fallback.
The audio stays in the station corpus; the replay relocates intact peer bursts.
"""
from pathlib import Path

import numpy as np
import pytest

from hfmodem.shrike import onair, pactor1, p1rx
from hfmodem.tests.shrike.test_p1_jump_to_p4_0929 import (
    BURSTS, FS, PAD, _arm, _gateway,
)

CAPTURES = Path(__file__).resolve().parents[5] / 'captures'


def _recording(arm):
    path = CAPTURES / f'onair-0929-{arm}' / 'stream.flac'
    if not path.exists():
        pytest.skip(f'local September 29 capture absent: {path}')
    return onair.session.load_wav(str(path), FS)


@pytest.mark.parametrize('at,word,error', [(18.605, pactor1.CS_SPEED, 0.0),
                                           (22.355, pactor1.CS_ACK_A, .0196)])
def test_2204_words_at_their_recorded_anchors(at, word, error):
    audio = _recording('2204')
    # Only 22.355 has a recorded +19.6 ms correction; 18.605 is a control.
    event = onair._SessionRx._p1_cs(audio, round((at - error) * FS))
    assert event is not None
    assert (event.cs, event.sense) == (word, False)


def test_1934_two_exact_answers_in_the_final_listen_connect(tmp_path, monkeypatch):
    audio = _recording('1934')
    bursts = [np.zeros(round(.130 * FS), np.float32) for _ in range(7)]
    # The actual two CS4s are two cycles apart, with the same shift. Put the
    # first after the last call and preserve the silent intervening slot.
    for k, at in ((3, 28.615), (5, 31.116)):
        start = round(at * FS)
        burst = audio[start - PAD:start + round(.125 * FS)]
        got = p1rx.cs_anchored(burst, PAD / FS)
        assert (got.index, got.errors, got.sense) == (pactor1.CS_SPEED, 0, 0)
        bursts[k] = burst
    keyed, log, _ = _arm(tmp_path, monkeypatch, bursts, 'silent')
    assert 'FINAL CALL LISTEN' in log
    assert any(what == 'P1 pkt#1 100Bd 7B' for _, what in keyed), log


def test_2208_packet_one_is_rendered_before_the_first_candidate(tmp_path, monkeypatch):
    observed, rendered, packets, calls = [], [], [], []
    observe = onair._SessionRx._on
    render = pactor1.packet_signal
    key = onair.RadioTx._tx

    def candidate(self, event, *args, **kwargs):
        if event.kind == "cs":
            observed.append(event)
        return observe(self, event, *args, **kwargs)

    def waveform(*args, **kwargs):
        rendered.append(len(observed))
        return render(*args, **kwargs)

    def transmit(self, audio, what, *args, **kwargs):
        before = len(self.slots_used)
        result = key(self, audio, what, *args, **kwargs)
        if len(self.slots_used) > before:
            if what.startswith('P1 pkt#1'):
                packets.append((self.slot, np.asarray(audio).copy()))
            elif what.startswith('connect'):
                calls.append(self.slot)
        return result

    monkeypatch.setattr(onair._SessionRx, '_on', candidate)
    monkeypatch.setattr(pactor1, 'packet_signal', waveform)
    # _arm instruments _tx too, so install this as its delegate.
    from hfmodem.tests.shrike import test_p1_jump_to_p4_0929 as replay
    monkeypatch.setattr(replay, '_KEY', transmit)
    keyed, log, _ = _arm(tmp_path, monkeypatch, _gateway(range(len(BURSTS))), 'shorts')
    assert packets and observed, log
    assert rendered and not any(rendered), rendered
    assert packets[0][0] == calls[-1] + 1
    # First packet stays the on-air 0x21 announcement, in either raster sense.
    expected = [render(b'1w9ssj\r', 200, packet_count=1,
                       header=pactor1.DATA_HEADER, bits45=2, invert=sense)
                for sense in (False, True)]
    assert any(np.array_equal(packets[0][1], wave) for wave in expected)
    assert [slot for slot, what in keyed if what.startswith('P1 ')] == [
        packets[0][0] + k for k in range(len(packets))]


def test_2208_pending_announcement_does_not_fall_back_locally(tmp_path, monkeypatch):
    # Silent mode leaves the real phase jump unacted on, exposing the old
    # retry-count speed-down. The peer has never requested CS4 on this tape.
    keyed, log, _ = _arm(tmp_path, monkeypatch, _gateway(range(len(BURSTS))), 'silent')
    p1 = [what for _, what in keyed if what.startswith('P1 ')]
    assert len(p1) > 4
    assert set(p1) == {'P1 pkt#1 200Bd 7B'}, (p1, log)
    assert 'falls back to 100 Bd' not in log


def test_1934_one_word_reobserved_is_not_a_second_answer():
    audio = _recording('1934')
    got = p1rx.cs_anchored(audio, 28.615)
    assert (got.index, got.errors) == (pactor1.CS_SPEED, 0)
    evidence = onair._ConnectEvidence()
    evidence.note_search(.1)
    assert not any(evidence.offer(23, .090) for _ in range(3))


@pytest.mark.parametrize('at', [17.588, 30.087])
@pytest.mark.parametrize('late', [.015, .0175, .020, .0225])
def test_0923_cs1_late_anchor_never_recovers_opposite_shift_cs2(at, late):
    path = CAPTURES / 'onair-0923-1851' / 'stream.flac'
    if not path.exists():
        pytest.skip(f'local September 23 capture absent: {path}')
    audio = onair.session.load_wav(str(path), FS)
    true = onair._SessionRx._p1_cs(audio, round(at * FS))
    assert (true.cs, true.sense) == (pactor1.CS_ACK_A, True)
    got = onair._SessionRx._p1_cs(audio, round((at + late) * FS))
    assert got is None or (got.cs, got.sense) == (true.cs, true.sense)


def test_1318_cs1_late_anchor_never_recovers_four_bit_alias():
    audio = _recording('1318')
    at = 5.087
    true = onair._SessionRx._p1_cs(audio, round(at * FS))
    assert (true.cs, true.sense) == (pactor1.CS_ACK_A, True)
    got = onair._SessionRx._p1_cs(audio, round((at + .0175) * FS))
    assert got is None or (got.cs, got.sense) == (true.cs, true.sense)
