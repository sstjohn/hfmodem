# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""BW2300 limited B→A: full L1 heads and wholly unread newer overs.

Kestrel calls, Xara sends. DATA decode outcomes enter through LevelRead;
query recognition, answers, commit and host IO use the wired adapter. The
release043 fixture extracts e597c9aca unchanged, as in the L2x2 pins.
"""

import numpy as np
import pytest

from hfmodem.kestrel.arq import phy
from hfmodem.kestrel.tx import varahf2300_tx as TX
from hfmodem.kestrel.vara import vara_arq as A, vara_frames as VF, vara_mfsk as MK
from .test_rxwindow_xara_l2x2 import decoded, release043, station


EXP = b'Hello Kestrel from Xara. abcdefghij\r\n' * 100


def full_l1(at):
    return TX.build_frame(EXP[at:at + 9] + b'\x99', 0)


def single(hs, at, now):
    decoded(hs, full_l1(at), now[0] + 6.5, now, level=1)


def query(hs, phase, now, *, form='xara-idle', delay=6.4):
    if form == 'xara-idle':
        kind = VF.SESSION_RESPONDER_OVER_IDLE if phase else VF.SESSION_RESPONDER_IDLE
        audio = MK.synth_burst(hs.called, kind)
    else:
        pairs = VF.over_continue_frames(hs.called, hs.bw, (11 + phase,))[11 + phase][1]
        audio = np.concatenate([np.zeros(MK.HOP), MK.synth_tone_pairs(pairs), np.zeros(MK.HOP)])
    now[0] = hs._unkeyed_at + delay + len(audio) / MK.FS
    hs.on_rx_audio(audio)


def assert_nak_only(hs, sink, before):
    assert len(sink.sent) > before
    assert all('nak' in kind for kind in sink.packets[before:])
    naks = (MK.synth_burst(hs.called, VF.SESSION_OVER_NAK_SHORT),
            MK.synth_tone_pairs(VF.nak(hs.called, hs.bw)[0]))
    for audio in sink.sent[before:]:
        assert any(np.array_equal(audio, nak) for nak in naks)


@pytest.mark.parametrize('companion', [True, False], ids=['clean-double', 'absent-companion'])
def test_certain_slot_stream_callbacks_allow_decoder_to_finish(monkeypatch, companion):
    now = [0.0]
    hs, sink = station(A, monkeypatch, now)
    hs._rx_read, hs._expect_level = (1, None), 1
    for at in range(0, 54, 9):
        single(hs, at, now)
    assert hs._rx_window.R == b''.join(sink.host) == EXP[:54]
    assert hs._holds_epoch == hs._holds_any == 6
    before = len(sink.sent)
    head = TX.synth_frame(full_l1(54), 0, over=7)
    tail = TX.synth_frame(full_l1(63), 0, over=None) if companion else np.zeros(0)
    audio = np.concatenate([np.zeros(MK.FS // 10), head, tail, np.zeros(7 * MK.FS)])
    events = []
    event = hs._rx_window.event

    def record(kind, **kwargs):
        events.append((kind, now[0]))
        return event(kind, **kwargs)

    monkeypatch.setattr(hs._rx_window, 'event', record)
    for start in range(0, len(audio), 1024):
        block = audio[start:start + 1024]
        now[0] += len(block) / MK.FS
        hs.on_rx_stream(block)
        if hs._held_answer is not None:
            hs.idle_keepalive()
    if companion:
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:72]
        assert len(b''.join(sink.host)) == 72
        assert len(sink.sent) == before + 1
        assert all('nak' not in kind for kind in sink.packets[before:])
        np.testing.assert_array_equal(sink.sent[-1],
                                      MK.synth_burst(hs.called, VF.SESSION_OVER_RESPONSE_SHORT))
        assert (hs._rx_window.K.start, hs._rx_window.K.end) == (54, 72)
        assert len(hs._rx_window.K.frames) == 2
        assert [kind for kind, _ in events] == ['F', 'F', 'E']
    else:
        assert_nak_only(hs, sink, before)
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:54]
        assert len(b''.join(sink.host)) == 54
        assert hs._rx_window.K.end == 54
        assert [kind for kind, _ in events] == ['F', 'E']
        assert events[-1][1] >= hs._decoded_frame_end + hs._companion_samples() / MK.FS
    assert hs._rx_window.W is None and hs._held_answer is None


def test_bracket_certain_slot_expires_on_silent_clock(monkeypatch):
    now = [0.0]
    hs, sink = station(A, monkeypatch, now)
    hs._rx_read, hs._expect_level = (1, None), 1
    for at in range(0, 54, 9):
        single(hs, at, now)
    before = len(sink.sent)
    events = []
    event = hs._rx_window.event

    def record(kind, **kwargs):
        events.append((kind, now[0]))
        return event(kind, **kwargs)

    monkeypatch.setattr(hs._rx_window, 'event', record)
    now[0] = 100.0
    hs.on_rx_audio(TX.synth_frame(full_l1(54), 0, over=7))
    assert hs._rx_window.state == 'OPEN' and hs._held_answer is not None
    deadline = hs._decoded_frame_end + hs._companion_samples() / MK.FS
    assert 100 < deadline < 112
    for tick in range(112, 1301, 12):
        now[0] = float(tick)
        hs.idle_keepalive()
        if tick == 112:
            assert ('E', 112.0) in events
            assert_nak_only(hs, sink, before)
            assert hs._rx_window.W is None and hs._held_answer is None
    assert hs._rx_window.state != 'OPEN'
    assert hs.state == A.VaraState.DISCONNECTED
    assert b''.join(sink.host) == EXP[:54]
    assert len(b''.join(sink.host)) == 54


@pytest.mark.parametrize('ending', ['companion', 'slot-end', 'c13', 'c12'])
def test_s12_stream_quiet_inside_certain_companion_slot(monkeypatch, ending):
    now = [0.0]
    hs, sink = station(A, monkeypatch, now)
    hs._rx_read, hs._expect_level = (1, None), 1
    for at in range(0, 540, 9):
        held = at >= 54 and (at - 54) % 18 == 0
        decoded(hs, full_l1(at), 385.19 - (531 - at) * .65, now, level=1, hold=held)
    assert hs._holds_epoch >= 6 and hs._peer_double_verdict() == 'CERTAIN'
    assert hs._rx_window.R == b''.join(sink.host) == EXP[:540]
    assert hs._peer_over == 33
    assert (hs._rx_window.K.start, hs._rx_window.K.end) == (522, 540)
    events = []
    states = []
    event = hs._rx_window.event
    window_state = A._window_state

    def record(kind, **kwargs):
        events.append(kind)
        return event(kind, **kwargs)

    def record_state(mag, rec):
        state = window_state(mag, rec)
        states.append(state)
        return state

    monkeypatch.setattr(hs._rx_window, 'event', record)
    monkeypatch.setattr(A, '_window_state', record_state)
    head, companion = (TX.build_frame(EXP[at:at + 9] + b'\x59', 0)
                       for at in (540, 549))
    assert head[:-2] == EXP[540:549] + b'\x59'
    assert companion[:-2] == EXP[549:558] + b'\x59'
    head_audio = TX.synth_frame(head, 0, over=34)
    quiet = np.zeros(round(.3 * MK.FS))
    before = len(sink.sent)
    now[0] = 391.89
    hs._stream_over(np.concatenate([head_audio, quiet]))
    assert hs._peer_over == 34
    assert states == [A._QUIET]
    assert 'F' in events
    assert 'E' not in events
    assert len(sink.sent) == before
    assert hs._rx_window.state == 'OPEN' and hs._held_answer is not None
    assert hs._rx_window.W.expected == frozenset((2,))
    assert hs._rx_window.R == b''.join(sink.host) == EXP[:540]
    deadline = hs._decoded_frame_end + hs._companion_samples() / MK.FS
    assert 391.89 < 396.68 < deadline + .7
    now[0] = deadline - .001
    hs._release_held_answer(round((now[0] - 391.89) * MK.FS))
    assert 'E' not in events and len(sink.sent) == before
    if ending == 'companion':
        now[0] = 397.39
        tail_audio = TX.synth_frame(companion, 0, over=None)
        hs._stream_over(np.concatenate([tail_audio, quiet]))
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:558]
        assert (hs._rx_window.K.start, hs._rx_window.K.end) == (540, 558)
        assert len(hs._rx_window.K.frames) == 2
        assert len(sink.sent) == before + 1
        np.testing.assert_array_equal(sink.sent[-1],
                                      MK.synth_burst(hs.called, VF.SESSION_OVER_RESPONSE_SHORT))
    else:
        if ending == 'slot-end':
            now[0] = deadline + .001
            hs._release_held_answer(round(.002 * MK.FS))
            assert events.count('E') == 1
        else:
            kind = VF.SESSION_RESPONDER_OVER_IDLE if ending == 'c13' else VF.SESSION_RESPONDER_IDLE
            audio = MK.synth_burst(hs.called, kind)
            now[0] = 399.31 + len(audio) / MK.FS
            hs.on_rx_audio(audio)
            assert 'Q' in events
        assert_nak_only(hs, sink, before)
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:540]
        assert hs._rx_window.K.end == 540 and hs._rx_window.T is None
    assert hs._rx_window.W is None and hs._held_answer is None


@pytest.mark.parametrize('path', ['wired', 'release043'])
def test_l1x2_full_head_never_credits_missing_companion(monkeypatch, release043, path):
    """§3 CERTAIN; OPEN/E → INCOMPLETE → drop W/NAK; I3/§4."""
    now = [100.0]
    hs, sink = station(A if path == 'wired' else release043, monkeypatch, now)
    hs._rx_read, hs._expect_level = (1, None), 1
    for at in range(0, 54, 9):
        single(hs, at, now)
    assert len(sink.sent) == 6 and b''.join(sink.host) == EXP[:54]
    positive = MK.synth_burst(hs.called, VF.SESSION_OVER_RESPONSE_SHORT)
    for audio in sink.sent:
        np.testing.assert_array_equal(audio, positive)
    if path == 'wired':
        assert hs._holds_epoch == hs._holds_any == 6
        assert hs._peer_double_verdict() == 'CERTAIN'

    head, companion = full_l1(54), full_l1(63)
    failed = companion[:-1] + bytes([companion[-1] ^ 1])
    assert VF.crc16_genibus(failed[:-2]) != int.from_bytes(failed[-2:], 'big')
    lead = np.zeros(MK.FS // 10)
    head_audio = TX.synth_frame(head, 0, over=7)
    audio = np.concatenate([lead, head_audio, TX.synth_frame(failed, 0, over=None), lead])
    assert phy.burst_extent(audio, '2300', 1) == 'double'
    before = len(sink.sent)
    now[0] += len(audio) / MK.FS
    hs._level_read = phy.LevelRead(1, (head[:-2],), None, 2,
                                   end=len(lead) + len(head_audio))
    try:
        assert hs._answer_data_over(audio)
    finally:
        hs._level_read = None
    assert b''.join(sink.host) == EXP[:54 if path == 'wired' else 63]
    query(hs, 1, now)

    if path == 'wired':
        assert_nak_only(hs, sink, before)
        assert hs._rx_window.W is None
        assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
        assert hs._rx_window.K.end == 54 and hs._rx_window.T is None
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:54]
        for at in range(54, 81, 9):
            single(hs, at, now)
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:81]
    else:
        assert b''.join(sink.host) == EXP[:63]
        assert len(sink.sent) > before
        np.testing.assert_array_equal(sink.sent[-1], positive)
        single(hs, 72, now)
        assert b''.join(sink.host) == EXP[:63] + EXP[72:81]
        assert len(b''.join(sink.host)) == 72
        assert b''.join(sink.host) != EXP[:81]
    assert hs.state == A.VaraState.CONNECTED


@pytest.mark.parametrize('path,form', [
    ('wired', 'xara-idle'), ('wired', 'phased'), ('release043', 'xara-idle'),
], ids=['wired-Q-minus', 'wired-Q-phase', 'release043'])
def test_unread_newer_l1_over_draws_nak_after_reanswer(monkeypatch, release043, path, form):
    """Q(p) next-attempt row; Q(-) outside-bound timing backstop (§§2,6).

    Over #2 is wholly unread: no decoder event between the two queries.
    The late query allows a full L1 over after the re-answer's unkey.
    """
    now = [100.0]
    hs, sink = station(A if path == 'wired' else release043, monkeypatch, now)
    hs._rx_read, hs._expect_level = (1, None), 1
    single(hs, 0, now)
    assert len(sink.sent) == 1 and b''.join(sink.host) == EXP[:9]
    query(hs, 0, now, form=form, delay=.1)
    assert len(sink.sent) == 2 and b''.join(sink.host) == EXP[:9]
    positive = sink.sent[-1].copy()
    if path == 'wired':
        assert hs._rx_window.A.phase == 0 and hs._rx_window.alt == 1
        assert hs._rx_window.trusted and hs._holds_any == 1

    before = len(sink.sent)
    query(hs, 1, now, form=form)
    if path == 'wired':
        assert_nak_only(hs, sink, before)
        assert not np.array_equal(sink.sent[-1], positive)
        assert hs._rx_window.A.kind == 'nak' and hs._rx_window.A.sent
        assert hs._rx_window.A.phase == 1 and not hs._rx_window.trusted
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:9]
        if form == 'phased':
            assert hs._rx_window.T is None
        else:
            assert (hs._rx_window.T.start, hs._rx_window.T.frontier) == (0, 9)
        single(hs, 9, now)
        assert hs._rx_window.T is None
        assert hs._rx_window.R == b''.join(sink.host) == EXP[:18]
    else:
        assert len(sink.sent) == before + 1
        np.testing.assert_array_equal(sink.sent[-1],
                                      MK.synth_burst(hs.called, VF.SESSION_OVER_RESPONSE))
        assert b''.join(sink.host) == EXP[:9]
        single(hs, 18, now)
        assert b''.join(sink.host) == EXP[:9] + EXP[18:27]
        assert len(b''.join(sink.host)) == 18
        assert b''.join(sink.host) != EXP[:27]
    assert hs.state == A.VaraState.CONNECTED
