# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A connect answer whole before our PTT is read, however late in the gap.

KB2PCN-8 20 m, onair-20261108T232021: from cycle 3 the gateway answered every
call at d = 113.7 ms, CS1 and then CS4 at zero bit errors, and every cycle
logged "RX (nothing decoded) ... CLOSE" until we stopped keying at cycle 23.
VE3KPG 40 m, onair-20261108T232318, answered at 110-115 ms and the same search
read only the copies that came in under 112. The listen window stops a pre-key
lead in front of our audio, 16 ms short of PTT on the FTDX10, and the search
needs a codeword and 2 ms past the top of its band, so 234 ms of window put
that top at 112 ms. The answers ended 17 ms before PTT, inside `_d_max_n`'s
130 ms, in audio `_tx` then dropped with our own burst.

Fixtures: each window from our carrier dropping to the next PTT, on the capture
stream of its arm. `live` is the window the session searched (`captured` in its
log); the rest is what was dropped.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1, spec

from .recorded_pcm import recorded_pcm
from .test_tx_latency_correction import _Stream

FS = onair.FS
D_MAX_N = onair._d_max_n(spec.CYCLE_SHORT_S, 0.04)
CLEAR_N = round((spec.CYCLE_SHORT_S - 0.04 - spec.P1_PACKET_S) * FS)

TAPES = {
    "kb2pcn-cs1": dict(file="kb2pcn-1108-2320-cs1-at-key.wav", sha256=(
        "5b55b26b314cffa4e2df34fe733d2ec80b13185620ab52e7bdc815982af56b8b"),
        start=168854, live=11232, cs=pactor1.CS_ACK_A, d=0.1137),
    "kb2pcn-cs4": dict(file="kb2pcn-1108-2320-cs4-at-key.wav", sha256=(
        "558e4d8138eda428222e59aacd8141f5afbe8da80b62e449428908e1196fba0e"),
        start=528854, live=11184, cs=pactor1.CS_SPEED, d=0.1137),
    "ve3kpg-cs4": dict(file="ve3kpg-1108-2323-cs4-at-key.wav", sha256=(
        "5ea4488059376188aaf6dca0c1e76f61825dfa40820a8a28c59ae892868c48ac"),
        start=888827, live=11280, cs=pactor1.CS_SPEED, d=0.1150),
}


def _search(audio):
    """One connect search over a window that starts as our carrier drops."""
    t0, span = onair._acquisition_window(audio.size, 0, D_MAX_N,
                                         onair.ACQUIRE_READ_TAIL_S)
    got = p1rx.acquire_control_signal(audio, t0, spec.P1_CS_S, span=span) \
        if span > 0 else None
    return None if got is None else (got[0], t0 + got[1],
                                     onair._candidate_excess(audio, t0 + got[1]))


def _connect(tx, audio, start, live):
    """The session's own search behind the key, as `_drive_session` calls it."""
    found, t0, _, heard = onair._connect_search(
        tx, audio[:live], start, (start,), D_MAX_N, onair._ConnectEvidence())
    return found, t0, heard


class _Rx:
    def __init__(self):
        self.skipped, self.bridged = [], 0

    def skip(self, seconds):
        self.skipped.append(round(seconds * FS))

    def bridge(self, audio):
        self.bridged += audio.size

    def __getattr__(self, name):
        return lambda *a, **k: None


SENTINEL = 0.9
"""What the fake capture carries from PTT on: our own key, which nothing reads."""


def _keyed(tmp_path, audio, start, pos, *, drain=True, lose=False):
    """The next call through `RadioTx._tx`, with the read left at `pos`.

    The capture runs a second past PTT, so a read that crosses it shows.
    `drain=False` puts the key's own wait lead at `pos`, so nothing is drained
    in front of the key; `lose` drops capture while the key drains.
    """
    ptt = start + audio.size
    stream_audio = np.concatenate([audio, np.full(FS, SENTINEL, np.float32)])

    class Stream(_Stream):
        def take_until(self, index):
            if lose:
                self.lost += 1
            seg = stream_audio[self.pos - start:max(0, index - start)]
            self.pos += seg.size
            return seg

        def transmit(self, *a, **k):
            got = super().transmit(*a, **k)
            self.samples = ptt + FS     # the codec runs on past our key
            return got

    stream = Stream()
    stream.lost, stream.xruns = 0, 0
    stream.samples, stream.pos = ptt, pos
    rig = SimpleNamespace(ptt=lambda keyed: True, key_failure=lambda: None)
    tx = onair.RadioTx(rig, transmit=True, outdir=tmp_path, settle=.04)
    tx.live, tx.tx_end, tx.sessrx = stream, start, _Rx()
    tx.boundary = ptt + round(tx.settle * FS)
    if not drain:
        lead = onair._prekey_lead(stream, round(tx.settle * FS))
        stream.pos = tx.boundary - lead
    tx._tx(np.ones(46080, np.float32), "connect->KB2PCN-8")
    return tx


@pytest.mark.parametrize("name", TAPES)
def test_the_window_that_flew_cannot_read_it(name):
    tape = TAPES[name]
    audio = recorded_pcm(tape)
    assert _search(audio[:tape["live"]]) is None


@pytest.mark.parametrize("name", TAPES)
def test_the_session_search_reads_it_from_the_audio_to_the_key(tmp_path, name):
    tape = TAPES[name]
    audio = recorded_pcm(tape)
    start, live = tape["start"], tape["live"]
    tx = _keyed(tmp_path, audio, start, start + live)
    origin, kept, _ = tx.heard_to_key
    assert origin == start + live and origin + kept.size == start + audio.size
    found, t0, heard = _connect(tx, audio, start, live)
    assert np.array_equal(heard, audio), "read past PTT, or short of it"
    assert found is not None and found[0] == tape["cs"], found
    d = t0 + found[1]
    assert abs(d - tape["d"]) < 1.5e-3, d
    assert onair._candidate_excess(heard, d) >= onair.ANSWER_CODEWORD_X


def test_with_nothing_drained_the_key_keeps_the_rest_to_ptt(tmp_path):
    tape = TAPES["kb2pcn-cs1"]
    audio = recorded_pcm(tape)
    start = tape["start"]
    tx = _keyed(tmp_path, audio, start, start + tape["live"], drain=False)
    origin, kept, _ = tx.heard_to_key
    assert tx.sessrx.bridged == 0
    assert np.array_equal(kept, audio[origin - start:])


def test_the_decoders_clock_still_skips_from_the_last_read(tmp_path):
    """The kept span is held for the search, and the rolling decoder, which
    never sees it, still owes the clock everything from its last sample on."""
    tape = TAPES["kb2pcn-cs1"]
    audio = recorded_pcm(tape)
    start, live = tape["start"], tape["live"]
    tx = _keyed(tmp_path, audio, start, start + live)
    origin, kept, _ = tx.heard_to_key
    drained = tx.sessrx.bridged
    assert tx.sessrx.skipped == [tx.tx_end - (start + live + drained)]


def test_a_capture_loss_keeps_nothing(tmp_path):
    tape = TAPES["kb2pcn-cs1"]
    audio = recorded_pcm(tape)
    start, live = tape["start"], tape["live"]
    tx = _keyed(tmp_path, audio, start, start + live, lose=True)
    assert tx.heard_to_key is None


def test_a_loss_after_the_key_keeps_it_out_of_the_search(tmp_path):
    tape = TAPES["kb2pcn-cs1"]
    audio = recorded_pcm(tape)
    start, live = tape["start"], tape["live"]
    tx = _keyed(tmp_path, audio, start, start + live)
    tx.live.lost += 1
    assert _connect(tx, audio, start, live)[0] is None


def test_audio_kept_at_another_key_is_not_this_windows(tmp_path):
    tape = TAPES["kb2pcn-cs1"]
    audio = recorded_pcm(tape)
    start, live = tape["start"], tape["live"]
    tx = _keyed(tmp_path, audio, start, start + live)
    assert onair._heard_to_key(tx, audio[:live - 128], start).size == live - 128
    assert onair._heard_to_key(tx, audio[:live], start + 1).size == live


def _planted(d, end_n, seed=20261108):
    """Noise from our carrier drop to `end_n`, with a CS4 starting at `d`."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(end_n) * 0.01).astype(np.float32)
    cs = pactor1.control_signal(pactor1.CS_SPEED).astype(np.float32)
    at = round(d * FS)
    cut = cs[:max(0, min(cs.size, end_n - at))]
    audio[at:at + cut.size] += cut
    return audio


@pytest.mark.parametrize("d", [0.110, 0.115, 0.120, 0.125, 0.128])
def test_a_codeword_whole_before_ptt_is_read(d):
    """A whole codeword by PTT, the top of a 40 ms settle's band included."""
    audio = _planted(d, CLEAR_N)
    cs, at, _ = _search(audio)
    assert cs == pactor1.CS_SPEED and abs(at - d) < 1.5e-3, (cs, at)


@pytest.mark.parametrize("d", [0.113, 0.120, 0.128])
def test_the_pre_key_lead_alone_cannot_read_past_112(d):
    """The window that flew, 234 ms, against the same codewords."""
    assert _search(_planted(d, CLEAR_N)[:11232]) is None


@pytest.mark.parametrize("d", [0.131, 0.135])
def test_a_codeword_still_running_at_ptt_is_not_read(d):
    """Past `_d_max_n` the codeword is cut by our own key, and nothing reads it."""
    assert _search(_planted(d, CLEAR_N)) is None
