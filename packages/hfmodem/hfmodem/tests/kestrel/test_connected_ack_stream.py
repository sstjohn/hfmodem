# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Step 5 is found on the receive stream, not in an energy-gated bracket.

A peer answers 0.12 to 0.17 s after our last transmitted sample (measured off air,
six connect-responses and one real-VARA connected-ack). This station's input used
to be dead for most of that: -83 dBFS against a -13 dBFS band, 502 transmissions
across 29 recordings, median 0.43 s. Shortening the PTT hold brought it inside the
turnaround, and the two things below outlive that fix.

The ack is the burst the gap punishes, because it is the one whose recognisable
part comes FIRST: 0.48 s of which only the four leading preamble symbols can be
checked at all — everything behind them is unreversed session state. A
connect-response is the other way round, fifteen callsign-keyed tones at the back
of twenty-three symbols, which is why every attempt recognised one of those while
none has yet recognised an ack.

So: it is looked for on the receive stream rather than in what an energy gate
brackets, because on this receive chain a gateway's answer sits at or below the
band noise (``VaraStationHandshake.on_rx_stream``); and it is matched on the last
three preamble symbols, because the leading one is what a residual gap or an
ordinary fade takes first — ``qso_vara_ns0a.wav`` holds a real gateway ack whose
symbol 0 lost a carrier and which the four-symbol form threw away.

These tests hold the stream route to the same evidence the bracket route is held
to, bound how long it stays open, and bound the relaxation at one symbol.

The last section is the other half of that: what else is in the turnaround, and
why none of it may stand in for the ack.
"""
from __future__ import annotations

import numpy as np
import pytest

from hfmodem.kestrel.rx import varahf2300 as RX
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import (
    _ACK_PLATEAU,
    _ACK_WINDOW,
    _I_LINKSETUP_SENT,
    _OVER_GUARD_MIN,
    _OVER_NEED,
    _OVER_ONSET_STEP,
    _OVER_REF_COLS,
    _STREAM_BLOCK,
    _UNDECODED,
    VaraState,
    VaraStationHandshake,
    _ack_lock,
    _ack_plateau,
    _pair_track,
    _tone_track,
    _top3_track,
)
from hfmodem.tests.kestrel import corpora

FS = MK.FS
_CALLED = "NS0A"
# Last sample of the link-setup that corpora.OFFAIR/NS0A_2300's ack answers; the
# ack itself begins 0.171 s later.
_OFFAIR_LINKSETUP_END = 17.800
# corpora.ONAIR_REFUSED_ACK: last sample of the first link-setup, read off the
# recording as the start of the receiver mute behind it, and where W8MW's ack lands.
_W8MW_LINKSETUP_END = 23.140
_W8MW_ACK_AT = 23.280
# corpora.ONAIR_CROWDED_ACK, read the same way: our own transmission comes back
# through the rig at -17 to -30 dBFS and stops at 40.79 s, the receiver is dead for
# the 165 ms behind it, and KD0PYG's ack opens 0.30 s out.
_KD0PYG_LINKSETUP_END = 40.790
_KD0PYG_ACK_AT = 41.096
_BLOCK = 4800                       # a device-sized receive callback


class _IO:
    def __init__(self):
        self.log_lines: list[str] = []
        self.tx_bursts = 0
        self.connected_as = None

    def key(self, on): ...
    def tx(self, samples): self.tx_bursts += 1
    def pending(self): ...
    def connected(self, caller, called, bw): self.connected_as = (caller, called, bw)
    def log(self, msg): self.log_lines.append(msg)


def _awaiting_ack(io=None, called: str = _CALLED) -> VaraStationHandshake:
    hs = VaraStationHandshake(["W9SSJ"], io or _IO(), bw="2300", mfsk_only=True)
    hs.originate(called, "W9SSJ")
    hs.step = _I_LINKSETUP_SENT
    hs._linksetup_tx = 1
    hs._reset_stream()
    return hs


def _stream(hs: VaraStationHandshake, x: np.ndarray) -> VaraStationHandshake:
    for i in range(0, len(x), _BLOCK):
        hs.on_rx_stream(x[i:i + _BLOCK])
        if hs.state is VaraState.CONNECTED:
            break
    return hs


def _channel(seconds: float, seed: int, ack_at: float | None = None) -> np.ndarray:
    """Band noise, optionally with a connected-ack laid over it at ``ack_at``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * FS)) * 0.05
    if ack_at is not None:
        ack = MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)
        at = int(ack_at * FS)
        x[at:at + len(ack)] += ack[:len(x) - at]
    return x


# --------------------------------------------------------------------------- #
def test_the_shared_transform_is_what_both_recognisers_read():
    """One batched transform now feeds the single-tone and two-tone searches, and
    a divergence between them would be a silent recognition failure in one."""
    x = _channel(2.0, seed=11, ack_at=0.5)
    top3, clear, _resid = _top3_track(x)
    assert np.array_equal(_tone_track(x), top3[:, 0])
    assert np.array_equal(_pair_track(x), np.sort(top3[:, :2], axis=1))
    assert clear.shape == (len(top3), 2) and (clear >= 0).all()
    assert (_pair_track(x)[:, 0] <= _pair_track(x)[:, 1]).all()


def test_an_ack_in_the_turnaround_window_completes_the_connect():
    io = _IO()
    hs = _stream(_awaiting_ack(io), _channel(3.0, seed=3, ack_at=0.171))
    assert hs.state is VaraState.CONNECTED
    assert io.connected_as == ("W9SSJ", _CALLED, "2300")
    assert io.tx_bursts == 2, "the connect-request and the step-6 session-confirm"


def test_an_ack_arriving_after_the_window_is_not_ours():
    """The ack names nobody, so what makes it ours is that it answers the over we
    just sent. A burst that arrives seconds later is another station's."""
    late = _ACK_WINDOW / FS + 1.0
    hs = _stream(_awaiting_ack(), _channel(late + 2.0, seed=4, ack_at=late))
    assert hs.state is not VaraState.CONNECTED


def test_band_noise_alone_never_completes_the_connect():
    for seed in range(24):
        hs = _stream(_awaiting_ack(), _channel(4.0, seed=100 + seed))
        assert hs.state is not VaraState.CONNECTED, f"noise (seed {seed}) connected"


def test_the_real_off_air_ack_is_found_on_the_stream():
    """The one gateway connected-ack addressed to this station that we hold, fed
    through the route a live attempt uses — off the rig, not synthesised."""
    path = corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip("off-air recording for NS0A_2300 not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    io = _IO()
    hs = _stream(_awaiting_ack(io), x[int(_OFFAIR_LINKSETUP_END * FS):])
    assert hs.state is VaraState.CONNECTED, (
        "a real gateway's connected-ack no longer completes the connect")
    assert any("connected-ack on the receive stream" in m for m in io.log_lines)


def _off_frequency(x: np.ndarray, carriers: int) -> np.ndarray:
    """``x`` as a station ``carriers`` off frequency would have put it on the air."""
    from scipy.signal import hilbert

    t = np.arange(len(x)) / FS
    return np.real(hilbert(x) * np.exp(2j * np.pi * MK.carrier_to_hz(carriers) * t))


def test_an_ack_from_the_station_that_answered_off_frequency_still_completes():
    """A gateway that answers 23 Hz low acks 23 Hz low, and a connect that took the
    one and pinned the other to zero would reach step 5 and stop there.

    The offset is not looked for. ``_peer_shift`` is what the connect-response
    already measured on this attempt — thirteen or more tones keyed to the callsign
    we dialled — so the ack is matched at one known offset rather than searched over
    five, which is what keeps it clear of the PACTOR-1 plateau below.
    """
    path = corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip("off-air recording for NS0A_2300 not present")
    x = corpora.wav_mono(path)
    x = _off_frequency(x / (np.abs(x).max() or 1.0), -1)[int(_OFFAIR_LINKSETUP_END * FS):]
    hs = _awaiting_ack()
    hs._peer_shift = -1
    assert _stream(hs, x).state is VaraState.CONNECTED, (
        "the gateway's ack arrived where its connect-response said it would and the "
        "connect did not complete")
    assert _stream(_awaiting_ack(), x).state is not VaraState.CONNECTED, (
        "an ack one carrier off completed a connect for a peer measured on "
        "frequency — the offset is being searched for rather than carried")


def test_only_a_real_ack_holds_the_preamble_anywhere_in_the_corpus():
    """The false-accept measurement, over every alignment of real HF.

    The stream route scans far more alignments than a bracket does, so its floor is
    measured rather than inherited: across the shared regression corpus at every
    32-sample offset, the only recordings that hold the preamble at all are the two
    carrying a VARA session's own connected-ack. Everything else — the PACTOR-1/2/3,
    ARDOP, FT8 and WSPR fixtures and band noise from four continents — reaches no
    alignment.

    This is the affordable half of the sweep behind ``_ACK_MIN_CARRIERS``, which ran
    the same arithmetic over every recording the project holds — 243 of them,
    31,393,466 alignments, this station's own 209 on-air captures included. Nine
    recordings hold at any alignment and every one of the nine carries a VARA
    session's two-tone control bursts; nothing else reaches four comparable carriers,
    and the best any of them reaches is three.
    """
    fixtures = sorted(corpora.REGRESS_FIXTURES.glob("*.wav"))
    if not fixtures:
        pytest.skip("shared regression corpus not present")
    held = {}
    for path in fixtures:
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        wide = _ack_plateau(x)
        if wide:
            held[path.name] = wide
    assert set(held) == {"qso_vara_kc9ghz.wav", "qso_vara_ns0a.wav"}, (
        f"preamble held in {held}")
    assert min(held.values()) >= 15, (
        "a genuine ack holds over tens of alignments; a shrinking plateau means "
        "the pair reader is losing one of the two carriers")


def test_why_the_ack_carries_an_offset_rather_than_searching_for_one():
    """The audio that would pay for a shift axis here, and what it would cost.

    A gateway tuned off frequency shifts an ack exactly as it shifts a
    connect-response, and the response search sweeps five offsets to find it
    (``vara_arq._RESP_SHIFTS``, after N0LCR-1 answered 23 Hz low on 2026-08-15). The
    ack must not sweep anything: the response is fifteen tones keyed to the callsign
    we dialled, while the ack names nobody and is six bins over three symbols. Sweep
    that and the corpus answers immediately — ``pos_p1_twosided_14110.wav``, a
    two-tone PACTOR-1 exchange with no VARA anywhere in it, holds the ack preamble
    for 51 consecutive alignments one carrier low, a wider plateau than any genuine
    ack in the corpus (49, 46, 19).

    A response accepted off frequency costs a link-setup keyed at a station that
    answered. An ack accepted off frequency costs a link that is not there.
    """
    path = corpora.REGRESS_FIXTURES / "pos_p1_twosided_14110.wav"
    if not path.exists():
        pytest.skip("shared regression corpus not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    assert _ack_plateau(x) == 0, (
        "this fixture is kept because it holds NOTHING at zero offset")
    assert _ack_plateau(x, -1) >= _ACK_PLATEAU, (
        "the PACTOR-1 plateau one carrier low is what keeps the ack's offset a "
        "carried measurement; if it has gone, that wants re-measuring not assuming")


def test_the_same_price_is_paid_for_the_fraction_of_a_carrier():
    """What a measured bin offset costs, on the same recording and at shift zero.

    Swept 2026-09-18 over the whole shared corpus at nine grids from -0.5 to +0.5,
    25,133,481 alignments: at zero the floor is where it always was, and the only
    recordings that hold the preamble at any grid are the four that carry a VARA
    session's own control bursts — each at the grid its own station transmits on,
    which is what the grid is for. The exception is this one. A two-tone PACTOR-1
    exchange with no VARA in it holds NOTHING from -0.125 to +0.5 and holds for
    48, 55 and 59 alignments at -0.25, -0.375 and -0.5.

    So `_ACK_PLATEAU`'s floor is a property of the grid and not of the constant,
    and this is the same recording and the same cost the carrier shift above
    already carries. What makes either of them affordable is the same thing and
    only that thing: the offset is measured off a burst already identified by
    thirteen or more tones keyed to the callsign we dialled, a session holds ONE
    of them, and nothing here ever goes looking  [vara_arq, _note_peer_offset].
    A reader that swept this axis would find the PACTOR-1 hold from a standing
    start, which is why none of them does.
    """
    path = corpora.REGRESS_FIXTURES / "pos_p1_twosided_14110.wav"
    if not path.exists():
        pytest.skip("shared regression corpus not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    assert _ack_plateau(x, 0, bin_offset=-0.125) == 0
    assert _ack_plateau(x, 0, bin_offset=-0.375) >= _ACK_PLATEAU


def test_the_lock_returns_the_sample_the_burst_opens_on():
    """``_ack_plateau`` says an ack is there; reading its seven state symbols
    needs where it started, which is what ``_ack_lock`` adds."""
    pairs = list(VF.CONNECTED_ACK_PREAMBLE) + list(VF.CONTROL_TAIL_TURN)
    burst = MK.synth_tone_pairs(pairs)
    for start in (2500, 5000, 12345):
        x = np.concatenate([np.zeros(start), burst, np.zeros(FS)])
        a = _ack_lock(x)
        assert a is not None, "the preamble holds here; the lock must too"
        assert abs(a - start) < MK.HOP // 4, (start, a)
        assert [tuple(p) for p in MK.demod_tone_pairs(x[a:], VF.CONNECTED_ACK_NSYM)] == [
            tuple(p) for p in pairs]


def test_the_lock_declines_where_the_plateau_does():
    assert _ack_lock(np.zeros(4 * FS)) is None
    rng = np.random.default_rng(7)
    assert _ack_lock(rng.normal(0, 0.1, 4 * FS)) is None
    pairs = list(VF.CONNECTED_ACK_PREAMBLE) + list(VF.CONTROL_TAIL_TURN)
    flush = np.concatenate([MK.synth_tone_pairs(pairs), np.zeros(FS)])
    assert _ack_lock(flush) is None, (
        "the lattice indexes preamble symbol 1, so a burst with no audio in "
        "front of it has nowhere to be located from -- callers pass pre-roll")


def test_a_faded_leading_symbol_does_not_lose_the_ack():
    """``qso_vara_ns0a.wav`` carries a real gateway ack at 19.379 s whose symbols
    1-3 are exact and whose symbol 0 reads (61, 64) for (64, 67) — one carrier of
    the pair gone to a fade. Requiring all four symbols threw it away."""
    path = corpora.REGRESS_FIXTURES / "qso_vara_ns0a.wav"
    if not path.exists():
        pytest.skip("shared regression corpus not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    ack = x[int(19.30 * FS):int(19.95 * FS)]
    assert _ack_plateau(ack) >= _ACK_PLATEAU
    heard = MK.demod_tone_pairs(x[int(19.379 * FS):], VF.CONNECTED_ACK_NSYM)
    assert tuple(heard[1:4]) == VF.CONNECTED_ACK_PREAMBLE[1:]
    assert tuple(heard[0]) != VF.CONNECTED_ACK_PREAMBLE[0], (
        "this fixture is kept for its FADED leading symbol; if that symbol now "
        "reads clean the test no longer covers what it was written for")


def test_two_lost_preamble_symbols_are_still_a_rejection():
    """One symbol of slack, not two. The relaxation is bounded by what was
    measured, and a burst holding only the last two is not accepted."""
    ack = MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)
    clipped = ack[2 * MK.HOP:]                     # symbols 0 and 1 gone
    rng = np.random.default_rng(9)
    x = rng.standard_normal(3 * FS) * 0.05
    x[int(0.171 * FS):int(0.171 * FS) + len(clipped)] += clipped
    assert _stream(_awaiting_ack(), x).state is not VaraState.CONNECTED


@corpora.requires_onair_refused_ack
def test_the_ack_a_busy_channel_buried_completes_the_connect():
    """The 2026-08-19 call to W8MW, and the whole of what this recogniser was
    getting wrong.

    A two-tone symbol's second carrier is a second argmax, so where the band is
    over one of the pair the reader still returns a bin for it — and matched on all
    six bins the ack at 23.280 s holds at no alignment at all, which is what every
    `preamble holds for 0 alignments` in that log was. Four of its six stand clear,
    all four are the preamble's, and it holds for 24.

    Fed from our own last transmitted sample, and again from the 0.10 s of cursor
    the driver skips behind it (`tools/kestrel_connect.RX_ECHO_GUARD_S` plus the
    keyed idle hold): the ack begins 0.14 s out, so the skip is not what loses it
    either.
    """
    x = corpora.wav_mono(corpora.ONAIR_REFUSED_ACK)
    x = x / (np.abs(x).max() or 1.0)
    for at in (_W8MW_LINKSETUP_END, _W8MW_LINKSETUP_END + 0.10):
        io = _IO()
        hs = _stream(_awaiting_ack(io, called="W8MW"), x[int(at * FS):])
        assert hs.state is VaraState.CONNECTED, f"fed from +{at:.2f} s"
        assert any("connected-ack on the receive stream" in m for m in io.log_lines)


@corpora.requires_onair_refused_ack
def test_nothing_else_in_that_recording_holds_the_preamble():
    """The other side of it. One 24-alignment plateau in 70 s of an occupied 80 m
    channel, and it is the ack; the recogniser reads the peer rather than the
    band it is buried in."""
    x = corpora.wav_mono(corpora.ONAIR_REFUSED_ACK)
    x = x / (np.abs(x).max() or 1.0)
    ack = int(_W8MW_ACK_AT * FS)
    assert _ack_plateau(x[ack - MK.NFFT:ack + 8 * MK.HOP]) >= _ACK_PLATEAU
    before, after = x[:ack - MK.NFFT], x[ack + 8 * MK.HOP:]
    assert _ack_plateau(before) == 0 and _ack_plateau(after) == 0, (
        "the preamble holds somewhere in this recording other than at its one ack")


@corpora.requires_onair_crowded_ack
def test_the_ack_an_occupant_stood_on_completes_the_connect():
    """The 2026-08-26 call to KD0PYG, and the other way a shared channel takes a
    carrier.

    W8MW's ack lost carriers to a band that was over them; this one loses a slot to
    a station that is louder than it. Symbol 3's own carriers are 68 and 78, and
    the occupant at 95/96 outranks 78 — so the two strongest bins of that window
    are 68 and a stranger's, which read as a preamble symbol the peer got wrong.
    Symbols 1 and 2 are exact, all six carriers stand clear, and matched on the
    pair the ack holds at NO alignment. It is what five `preamble holds for 0
    alignments` were on that arm, and what six unconnected arms were that night.
    """
    x = corpora.wav_mono(corpora.ONAIR_CROWDED_ACK)
    x = x / (np.abs(x).max() or 1.0)
    io = _IO()
    hs = _stream(_awaiting_ack(io, called="KD0PYG"),
                 x[int(_KD0PYG_LINKSETUP_END * FS):])
    assert hs.state is VaraState.CONNECTED
    assert any("connected-ack on the receive stream" in m for m in io.log_lines)


@corpora.requires_onair_crowded_ack
def test_the_crowded_ack_reads_the_shipped_preamble_where_it_locks():
    """What settles that the constant was never the fault. The lock lands within a
    tenth of a symbol of where the burst opens, and the three compared symbols read
    (56, 74), (64, 69) and (68, 78) — the shipped preamble, off a real gateway."""
    x = corpora.wav_mono(corpora.ONAIR_CROWDED_ACK)
    x = x / (np.abs(x).max() or 1.0)
    at = int(_KD0PYG_ACK_AT * FS)
    seg = x[at - MK.HOP:at + 12 * MK.HOP]
    assert _ack_plateau(seg) >= _ACK_PLATEAU
    lock = _ack_lock(seg)
    assert lock is not None and abs(lock - MK.HOP) < MK.HOP // 4
    heard = MK.demod_tone_pairs(seg[lock:], VF.CONNECTED_ACK_NSYM)
    assert tuple(heard[1:3]) == VF.CONNECTED_ACK_PREAMBLE[1:3]
    assert heard[3][0] == VF.CONNECTED_ACK_PREAMBLE[3][0], (
        "symbol 3's lower carrier is the strongest bin in the band; if it has "
        "moved this fixture is no longer the case it was kept for")


@corpora.requires_onair_crowded_ack
def test_nothing_else_in_the_kd0pyg_recording_holds_the_preamble():
    """The other side of it, over 190 s of an occupied 40 m channel: one plateau,
    and it is the gateway's answer to our first link-setup."""
    x = corpora.wav_mono(corpora.ONAIR_CROWDED_ACK)
    x = x / (np.abs(x).max() or 1.0)
    at = int(_KD0PYG_ACK_AT * FS)
    before, after = x[:at - MK.NFFT], x[at + 12 * MK.HOP:]
    assert _ack_plateau(before) == 0 and _ack_plateau(after) == 0


def test_a_real_ack_outside_the_window_stays_outside_it():
    """The corpus fixture that carries somebody else's connected-ack. The
    recogniser finds it — that is the previous test — and the window is the only
    thing that keeps a stranger's session from completing our connect."""
    path = corpora.REGRESS_FIXTURES / "qso_vara_kc9ghz.wav"
    if not path.exists():
        pytest.skip("shared regression corpus not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    assert _stream(_awaiting_ack(), x).state is not VaraState.CONNECTED


# --------------------------------------------------------------------------- #
# What else is in that window.
#
# A gateway that has taken the link and started sending would be reading the link
# up for us, and a rule that took a DATA over in the turnaround as step 5 would
# reach KC9GHZ where the ack search does not. The three sessions that reached
# CONNECTED are the positives for that; every link-setup that did not is the
# negative; and the two populations are measured below on the same arithmetic the
# data phase already keys the transmitter on.
_MYCALL = "W9SSJ"
_FRAME_S = _OVER_NEED / FS                  # 395 emission columns, 4.213 s
#: The three sessions in the record that reached CONNECTED, as ``(recording, our
#: last link-setup, the peer's first DATA over)``. Seconds, at the 395 emission
#: columns rather than at the preamble in front of them, which is where `corpora`
#: and `test_greeting_by_stream` locate an over.
_TOOK_THE_LINK = (
    (corpora.ONAIR_GATEWAY_GREETING, 51.591, 58.091),
    (corpora.ONAIR_GATEWAY_OVERS, 34.948, 41.403),
    (corpora.ONAIR_FADED_GREETING, 61.330, 67.804),
)
#: The level neither population crosses, against the band each arrived on. The rig
#: attenuates its own receiver while this station keys, so our own link-setup comes
#: back 7.3 to 22.3 dB under the band while all five gateway overs sit 1.5 to 9.5
#: dB over it: one line with four and a half decibels of room on each side.
_ECHO_SPLIT_DB = -3.0
#: Best anything that is not a frame reaches across the 2,434,523 alignments of
#: those 27 windows, against the 16 `_OVER_GUARD_MIN` opens on.
_TURNAROUND_FLOOR = 11


def _norm(path) -> np.ndarray:
    return corpora.wav_mono(path) / 32768.0


def _frame_db(x: np.ndarray, at: float = 0.0, span: float = 0.0) -> float:
    seg = x[int(at * FS):int((at + span) * FS)] if span else x
    f = FS // 100
    rms = np.sqrt((seg[:len(seg) // f * f].reshape(-1, f) ** 2).mean(1))
    return float(np.median(20 * np.log10(rms + 1e-15)))


def _alignments(n: int) -> int:
    """Frame starts a buffer of ``n`` samples offers the onset search."""
    r = RX.RECORDS[RX.BASE_LEVEL]
    return sum(max(0, (n - onset) // r.dw50 - RX._BASE_NCOLS + 1)
               for onset in range(0, r.dw50, _OVER_ONSET_STEP))


def _scan(x: np.ndarray, called: str):
    """Run the data phase's own over search over ``x``, on the live geometry.

    Buffers of ``_OVER_NEED + _STREAM_BLOCK`` advanced a block at a time, each
    scored at every onset and every column start, exactly as ``_stream_over``
    does — and where the guard opens, the recogniser behind it is asked what the
    audio is. Returns the frames it found as ``(hits, body, log line)``, the best
    anything else reached, and how many alignments were scored.
    """
    hs = _awaiting_ack(_IO(), called=called)
    frames, floor, total, last, k = [], 0, 0, -2, -1
    buf = np.zeros(0)
    for i in range(0, len(x), _STREAM_BLOCK):
        buf = np.concatenate([buf, x[i:i + _STREAM_BLOCK]])
        if len(buf) < _OVER_NEED + _STREAM_BLOCK:
            continue
        k += 1
        total += _alignments(len(buf))
        hits, _ = hs._rec3_alignment(buf)
        if hits < _OVER_GUARD_MIN:
            floor = max(floor, hits)
        else:
            before = len(hs.io.log_lines)
            body = hs._peer_data_over(buf)
            body = body[0] if body else None
            if k != last + 1:               # the same frame lands in two scans
                frames.append((hits, body, hs.io.log_lines[before]))
            last = k
        buf = buf[-(_OVER_NEED - 1):]
    return frames, floor, total


@corpora.requires_unconnected_linksetups
def test_a_link_setup_that_brought_no_connect_brought_no_over_either():
    """Every link-setup this station has sent that did not end in a connect.

    27 of them, across three evenings and six callsigns, 861 s of receive audio,
    2,434,523 alignments — and the same one thing in all 27: our own link-setup
    coming back through the receiver, 24 of 24 reference columns, a clean CRC and
    the callsign it names is this station's. Nothing else in any of those windows
    reaches 12 of the 24, against the 16 the guard opens on.

    So there is no peer over in a turnaround anywhere in the record, and a rule
    that read one as the link coming up would have exactly one thing to read: the
    frame we ourselves had just transmitted, in 27 windows out of 27. The 2026-08-19
    call to KC9GHZ is the specimen — its third link-setup keys at 46.87 s and its
    395 emission columns begin at 47.121, which is the 4.21 s frame at "46.94 s"
    that reads as a gateway starting to send.

    `ONAIR_REFUSED_ACK` is in this population and is the case that settles the
    other reading. W8MW DID take the link there — its connected-ack is at 23.280 s
    and the two tests above recover it — and it still sent nothing for the
    remaining 47 s. A peer that has the link waits for the session-confirm.
    """
    seen, floors, total = 0, [], 0
    for path, count in corpora.ONAIR_UNCONNECTED_LINKSETUPS:
        called = path.stem.rsplit("-W9SSJ-", 1)[-1]
        frames, floor, n = _scan(_norm(path), called)
        seen, total = seen + len(frames), total + n
        floors.append((path.name, floor))
        assert len(frames) == count, (
            f"{path.name} holds {count} link-setup(s) and the over search found "
            f"{len(frames)} base frame(s) in it")
        for hits, body, line in frames:
            assert hits == _OVER_REF_COLS, f"{path.name}: {hits}/{_OVER_REF_COLS}"
            assert body is None and f"link-setup naming {_MYCALL}" in line, (
                f"{path.name}: a frame in a turnaround that is not ours — {line}")
    declared = sum(n for _, n in corpora.ONAIR_UNCONNECTED_LINKSETUPS)
    assert seen == declared == 27, (
        f"{seen} link-setups found against {declared} declared — the floor below "
        "was measured over 27, and a window that has gone relaxes it")
    assert total > 2_000_000, f"only {total:,} alignments — the audio has changed"
    worst = max(floors, key=lambda kv: kv[1])
    assert worst[1] <= _TURNAROUND_FLOOR, (
        f"{worst[0]} reaches {worst[1]}/{_OVER_REF_COLS} reference columns on "
        f"something that is not a frame, over the {_TURNAROUND_FLOOR} this "
        f"population was measured at and against the {_OVER_GUARD_MIN} the guard "
        "opens on — the two no longer have room between them")


@corpora.requires_onair_gateway_greeting
@corpora.requires_onair_gateway_overs
@corpora.requires_onair_faded_greeting
def test_a_peer_that_took_the_link_sends_nothing_in_the_ack_turnaround():
    """The positives, and the reason no over in a turnaround can be step 5.

    The three sessions the record holds that reached CONNECTED. In every one the
    peer's first DATA over begins 2.24 to 2.29 s after our link-setup's last
    sample — after its connected-ack, and after our own step-6 session-confirm —
    which is 1.24 s past the far edge of `_ACK_WINDOW`. Not one gateway has ever
    started sending inside the window the ack is looked for in.

    The level says the same thing twice over. Our own link-setup returns 7.3 to
    22.3 dB UNDER the band it was sent into, because the rig attenuates its own
    receiver while this station keys; every one of the five gateway overs reads
    1.5 to 9.5 dB OVER it. Between the two lies the whole of the separation a rule
    here would have to stand on, and it separates ours from the peer's rather than
    a peer that took the link from one that did not.
    """
    for path, ours, theirs in _TOOK_THE_LINK:
        x = _norm(path)
        band = _frame_db(x)
        hs = _awaiting_ack(_IO(), called="KC9GHZ")
        r = RX.RECORDS[RX.BASE_LEVEL]
        for at, mine in ((ours, True), (theirs, False)):
            win = x[int(at * FS) - r.dw50:int(at * FS) + _OVER_NEED + r.dw50]
            hits, _ = hs._rec3_alignment(win)
            level = _frame_db(x, at, _FRAME_S) - band
            assert hits >= _OVER_GUARD_MIN, (
                f"{path.name} at {at} s: {hits}/{_OVER_REF_COLS}")
            before = len(hs.io.log_lines)
            body = hs._peer_data_over(win)
            body = body[0] if body else None
            line = hs.io.log_lines[before]
            if mine:
                assert body is None and f"link-setup naming {_MYCALL}" in line, line
                assert level < _ECHO_SPLIT_DB, (
                    f"{path.name}: our own link-setup returns {level:+.1f} dB "
                    "against the band — the receiver is no longer attenuated "
                    "while this station keys")
            else:
                assert body and body != _UNDECODED, line
                assert level > _ECHO_SPLIT_DB, (
                    f"{path.name}: the gateway's over reads {level:+.1f} dB "
                    "against the band, down among our own transmissions")
                gap = at - (ours + _FRAME_S)
                assert gap > _ACK_WINDOW / FS, (
                    f"{path.name}: the peer's first over begins {gap:.2f} s after "
                    "our link-setup, inside the window the ack is looked for in")
                assert 2.0 < gap < 2.5, (
                    f"{path.name}: {gap:.2f} s, not the 2.24-2.29 s every session "
                    "in the record turned round in")
