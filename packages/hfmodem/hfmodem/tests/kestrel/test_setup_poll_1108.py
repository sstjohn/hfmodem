# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""W0TRM, 2026-11-08: a called station's idle poll after our link-setup is ours.

Its answer to our level-4 setup did not reach us; it then polled, keyed to our
call, every 3.0 s from 3.1 s after the setup, and each poll was refused as a
connected-ack outside the answer window.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from hfmodem.kestrel.vara import vara_arq as A, vara_frames as F, vara_mfsk as M
from .test_setup_turn_idle import IO

FIXTURE = Path(__file__).with_name('fixtures') / 'w0trm-setup-poll-1108'
#: Where the fixture's two polls end, from its first sample.
POLL_ENDS = (2.895 + A._POLL_MIN / M.FS, 5.907 + A._POLL_MIN / M.FS)


def awaiting(called='W0TRM', *, refused=False):
    io = IO()
    hs = A.VaraStationHandshake(['W9SSJ'], io, bw='500', chat=True, high_speed=True)
    hs.originate(called, 'W9SSJ')
    io.accept = not refused
    hs.on_rx_tones(F.handshake_tones(called, F.connect_response('500', 4)),
                   F.connect_response('500'))
    assert hs.step == A._I_LINKSETUP_SENT
    io.sent.clear()
    return hs, io


def recorded():
    meta = json.loads((FIXTURE / 'provenance.json').read_text())
    path = FIXTURE / 'polls.wav'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256']
    rate, x = wavfile.read(path)
    assert rate == M.FS
    return x.astype(float) / 32768


def poll(call='W9SSJ', frame=1):
    return M.synth_tone_pairs(F.control_bursts(call, '500', frame)[1])


def keepalive(called='W0TRM'):
    return M.synth_burst(called, F.for_bw(F.SESSION_KEEPALIVE_B, '500'))


def stream(hs, x, block=960):
    for at in range(0, len(x), block):
        hs.on_rx_stream(x[at:at + block])


def bracket(hs, x, late=3 * M.FS):
    # A bracket's start is reckoned back from the stream's count.
    hs._heard = hs._ack_from + late + len(x)
    hs.on_rx_audio(x)


@pytest.mark.parametrize('block', [960, 4096])
def test_recorded_poll_connects_and_is_answered_in_its_turnaround(block):
    hs, io = awaiting()
    keyed = []
    tx = io.tx
    io.tx = lambda x: keyed.append(hs._heard - hs._ack_from) or tx(x)
    hs.send(b'queued opening')
    pending = hs.payload_ledger.pending
    x = recorded()
    end = int((POLL_ENDS[0] + 0.6) * M.FS)
    stream(hs, x[:end], block)
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.connections) == 1 and hs.turn == A._TURN_PEER
    assert len(io.sent) == 1
    np.testing.assert_allclose(io.sent[0], keepalive())
    assert any('idle poll keyed to our call' in m for m in io.logs)
    assert not any('outside the answer window' in m for m in io.logs)
    # Keyed behind the poll's end, in its turnaround: held at least the
    # answer's hold, at most one stream block plus a scan.
    after = keyed[0] / M.FS - POLL_ENDS[0]
    assert A._ANSWER_HELD_S <= after <= A._GRANT_FRESH_S + block / M.FS, after
    assert hs.payload_ledger.pending == pending and hs.payload_ledger.confirmed == 0
    stream(hs, x[end:], block)
    assert len(io.connections) == 1
    assert len(io.sent) == 2, io.logs


def test_recorded_poll_bracket_connects():
    hs, io = awaiting()
    x = recorded()
    at = int(2.895 * M.FS)
    bracket(hs, x[at - 2400:at + A._POLL_MIN + 4800])
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.sent) == 1
    np.testing.assert_allclose(io.sent[0], keepalive())


@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('late', [3.1, 20.0])
def test_generated_poll_connects_however_late(route, late):
    hs, io = awaiting()
    burst = np.r_[np.zeros(2400), poll(), np.zeros(9600)]
    if route == 'stream':
        stream(hs, np.r_[np.zeros(int(late * M.FS)), burst])
    else:
        bracket(hs, burst, int(late * M.FS))
    assert hs.state == A.VaraState.CONNECTED, io.logs
    assert len(io.sent) == 1 and len(io.connections) == 1
    np.testing.assert_allclose(io.sent[0], keepalive())


@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('case', ['wrong-call', 'frame-0', 'truncated', 'silence', 'unsolicited',
                                  'setup-unsent', 'still-keying', 'still-keying-weaker',
                                  'still-keying-weakest'])
def test_audio_that_is_not_our_poll_leaves_the_setup_waiting(route, case):
    hs, io = awaiting(refused=case == 'setup-unsent')
    assert hs._linksetup_tx == (0 if case == 'setup-unsent' else 1)
    if case == 'unsolicited':
        hs.step = A._I_CR_SENT
    x = poll('W1AW' if case == 'wrong-call' else 'W9SSJ', 0 if case == 'frame-0' else 1)
    if case == 'truncated':
        x = x[:-3 * M.HOP]
    if case == 'silence':
        x = np.zeros(len(x))
    # The peer keys on behind the poll: answering into it keys over it.
    level = {'still-keying': 1, 'still-keying-weaker': .5, 'still-keying-weakest': .25}
    behind = level.get(case, 0) * np.resize(poll('W1AW'), 9600)
    burst = np.r_[np.zeros(2400), x, behind]
    if route == 'stream':
        stream(hs, np.r_[np.zeros(3 * M.FS), burst])
    else:
        bracket(hs, burst)
    assert hs.state == A.VaraState.CONNECTING, io.logs
    assert not io.connections and not io.sent


@pytest.mark.parametrize('route', ['stream', 'bracket'])
@pytest.mark.parametrize('gap,onset,after,beside', [
    (3840, 960, 0, 0), (4560, 240, 0, 0), (4416, 144, 240, 0), (4560, 240, 0, .3)])
def test_a_carrier_inside_the_answer_hold_defers_the_answer(route, gap, onset, after, beside):
    """A quiet gap shorter than the hold, then carrier, down to 3 ms of it
    and beside a strong tone out of the channel: nothing is keyed until the
    hold has run quiet behind the carrier."""
    hs, io = awaiting()
    keyed = []
    tx = io.tx
    io.tx = lambda x: keyed.append(hs._heard - hs._ack_from) or tx(x)
    late = 3 * M.FS
    end = late + 2400 + len(poll()) + gap + onset + after
    carrier = 0.5 * np.resize(poll('W1AW'), onset)
    n = 2400 + len(poll()) + gap + onset + after
    noise = 1e-3 * np.random.default_rng(3).standard_normal(n)
    noise += beside * np.sin(2 * np.pi * 3000 * np.arange(n) / M.FS)
    burst = np.r_[np.zeros(2400), poll(), np.zeros(gap), carrier, np.zeros(after)] + noise
    if route == 'bracket':
        bracket(hs, burst)
        assert hs.state == A.VaraState.CONNECTING and not io.sent, io.logs
        return
    stream(hs, np.r_[np.zeros(late), burst], 48)
    assert hs.state == A.VaraState.CONNECTING and not io.sent, io.logs
    rest = np.arange(n, n + 9600)
    stream(hs, 1e-3 * np.random.default_rng(4).standard_normal(9600)
           + beside * np.sin(2 * np.pi * 3000 * rest / M.FS), 480)
    assert hs.state == A.VaraState.CONNECTED and len(io.sent) == 1, io.logs
    assert keyed[0] - end >= A._ANSWER_HELD_S * M.FS


@pytest.mark.parametrize('onset,behind', [(0, 0), (240, 0), (240, 240), (240, 480)])
def test_a_recorded_poll_is_not_answered_into_a_carrier_at_the_newest_samples(onset, behind):
    """W0TRM's own noise behind its poll, then 5 ms of a two-tone burst at the
    poll's strength ending at or up to 10 ms before the newest sample: the
    channel is not quiet."""
    hs, io = awaiting()
    x = recorded()
    at = int(2.895 * M.FS)
    cut = x[at - 2400:int((POLL_ENDS[0] + 0.15) * M.FS)].copy()
    strength = np.sqrt(np.mean(x[at:at + A._POLL_MIN] ** 2))
    burst = poll('W1AW')[2400:2400 + onset]
    if onset:
        cut[len(cut) - behind - onset:len(cut) - behind] += (
            strength / np.sqrt(np.mean(burst ** 2)) * burst)
    assert hs._peer_setup_poll(cut) == (not onset)
    bracket(hs, cut)
    assert (hs.state == A.VaraState.CONNECTED) == (not onset), io.logs
    assert len(io.sent) == (not onset)


@pytest.mark.parametrize('behind', [0, 240])
def test_a_streamed_poll_answer_waits_a_full_hold_behind_a_frontier_carrier(behind):
    hs, io = awaiting()
    keyed = []
    tx = io.tx
    io.tx = lambda x: keyed.append(hs._heard - hs._ack_from) or tx(x)
    x = recorded()[:int((POLL_ENDS[0] + 0.6) * M.FS)].copy()
    at = int(2.895 * M.FS)
    strength = np.sqrt(np.mean(x[at:at + A._POLL_MIN] ** 2))
    burst = poll('W1AW')[2400:2640]
    ends = int((POLL_ENDS[0] + 0.15) * M.FS) - behind
    x[ends - 240:ends] += strength / np.sqrt(np.mean(burst ** 2)) * burst
    stream(hs, x)
    assert hs.state == A.VaraState.CONNECTED and len(io.sent) == 1, io.logs
    assert keyed[0] - ends >= A._ANSWER_HELD_S * M.FS, keyed


def test_a_poll_the_stream_answered_is_not_answered_again_from_its_bracket():
    hs, io = awaiting()
    x = recorded()
    stream(hs, x[:int((POLL_ENDS[0] + 0.6) * M.FS)])
    assert hs.state == A.VaraState.CONNECTED and len(io.sent) == 1
    at = int(2.895 * M.FS)
    hs.on_rx_audio(x[at - 2400:at + A._POLL_MIN + 4800])
    assert len(io.sent) == 1, io.logs


def test_resent_setup_keeps_the_ask_its_connect_response_earned():
    hs, io = awaiting()
    # W0TRM's answer measured 15.9, which asks the top level.
    hs._heard_db = 15.9
    assert hs.resend_link_setup()
    heard, ask, expect = hs._heard_db, hs._setup_ask, hs._next_500
    assert ask == 6
    # What the stream holds when the driver resends is the peer's poll audio,
    # weaker on the tone measure than its answer was.
    hs._rx_recent = 0.01 * np.random.default_rng(1).standard_normal(2 * M.FS)
    assert hs.resend_link_setup()
    assert (hs._heard_db, hs._setup_ask, hs._next_500) == (heard, ask, expect), io.logs
