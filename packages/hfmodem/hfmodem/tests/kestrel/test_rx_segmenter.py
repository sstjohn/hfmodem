# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The burst segmenter the connect tool puts in front of the handshake.

It had no test of any kind, which is how it went through eight on-air sessions
cutting the gateway's answer in half. It used to threshold a 20 ms RMS envelope at
one level and end the burst at the first falling edge, so a fade inside the
connected-ack produced two pieces, both under the 0.2 s minimum it then applied,
both discarded — and the attempt printed "NOT connected", which is what a gateway
that never answered prints too.

Measured here over a 225-case fade sweep (15 fade positions x 3 durations x 5
depths inside a synthetic connected-ack, 1 s of quiet channel either side),
counting how often the ack reaches the handshake as one piece and how often the
connect completes:

    single-threshold cut, first falling edge    90/225 whole   207/225 connect
    the monitor's gate + its find_bursts crop  134/225 whole   173/225 connect
    the gate alone, whole bracket handed over  225/225 whole   225/225 connect

The middle row is the trap: ``Segmenter._crop`` re-cuts each bracketed region with
``find_bursts``, which thresholds on that region's global maximum and so puts back
the split the hysteresis just rode through. Good boundaries for a traffic log,
worse than useless for the handshake.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hfmodem.tests.kestrel import corpora
from hfmodem.kestrel.vara import vara_frames as VF
from hfmodem.kestrel.vara import vara_mfsk as MK
from hfmodem.kestrel.vara.vara_arq import _I_CR_SENT, _I_LINKSETUP_SENT, VaraState, VaraStationHandshake

kc = corpora.harness("kestrel_connect")
VM = corpora.harness("vara_monitor")

FS = MK.FS
FRAME = VM.SEG_FRAME
_CALLED = "NS0A"
_LEAD = 1.0
_ACK = MK.synth_tone_pairs(VF.CONNECTED_ACK_2300)


class _IO:
    def key(self, on): ...
    def tx(self, samples): ...
    def pending(self): ...
    def connected(self, *a): ...
    def log(self, msg): ...


def _pieces(audio, chunk: int = 4800) -> list[tuple[int, np.ndarray]]:
    """``(start sample, audio)`` for every burst the segmenter hands over.

    Fed in device-sized chunks, exactly as ``AudioVaraIO.next_rx_burst`` feeds it
    from the recording thread's buffer.
    """
    seg = kc._BracketSegmenter()
    out = []
    for i in range(0, len(audio), chunk):
        out += seg.push(audio[i:i + chunk])
    return out + seg.flush()


def _on_air(x, lead: float = _LEAD, tail: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(int(lead * FS)), np.asarray(x, float),
                        np.zeros(int(tail * FS))])
    return y + rng.standard_normal(len(y)) * 1e-4


def _faded(x, at_s: float, dur_s: float, depth_db: float) -> np.ndarray:
    y = np.array(x, float)
    y[int(at_s * FS):int((at_s + dur_s) * FS)] *= 10 ** (depth_db / 20.0)
    return y


def _covers(pieces, lo: int, hi: int) -> bool:
    return any(s <= lo and s + len(p) >= hi for s, p in pieces)


def _connects(pieces, called: str = _CALLED) -> bool:
    hs = VaraStationHandshake(["W9SSJ"], _IO(), bw="2300", mfsk_only=True)
    hs.originate(called, "W9SSJ")
    hs.step = _I_LINKSETUP_SENT
    for _, p in pieces:
        hs.on_rx_audio(p)
    return hs.state is VaraState.CONNECTED


# --------------------------------------------------------------------------- #
# The fade: one burst, one piece.
_FADES = [(at, dur, db)
          for at in (0.06, 0.20, 0.32, 0.44)
          for dur in (0.025, 0.050, 0.090)
          for db in (-60, -16)]


@pytest.mark.parametrize("at,dur,db", _FADES, ids=lambda v: str(v))
def test_a_fade_inside_a_burst_does_not_split_it(at, dur, db):
    """The defect this segmenter exists to fix: a notch in the middle of the
    gateway's answer used to end the burst there. The hysteresis gate closes only
    after SEG_HANG quiet frames (128 ms), so anything shorter is ridden through."""
    lo, hi = int(_LEAD * FS), int(_LEAD * FS) + len(_ACK)
    pieces = _pieces(_on_air(_faded(_ACK, at, dur, db)))
    assert _covers(pieces, lo, hi), (
        f"a {dur*1000:.0f} ms {db} dB fade at {at} s cut the ack into "
        f"{[round(len(p)/FS, 3) for _, p in pieces]}")


def test_the_faded_ack_completes_the_connect():
    """End to end over the same fades: what the segmenter hands over is what the
    handshake has to work with. Over the full 225-case sweep this is 225/225,
    against 207/225 for the single-threshold cut it replaces."""
    ok = sum(_connects(_pieces(_on_air(_faded(_ACK, *c)))) for c in _FADES)
    assert ok == len(_FADES), f"only {ok}/{len(_FADES)} faded acks connected"


def test_a_clipped_burst_is_still_handed_over_whole():
    """A burst clipped by the recording boundary must arrive as one piece, not be
    dropped for being short: the old cut needed a falling edge before it would
    hand anything over at all, and applied a 0.2 s minimum on top."""
    for missing in (1, 4084, len(_ACK) // 2):
        clipped = _ACK[:len(_ACK) - missing]
        pieces = _pieces(_on_air(clipped))
        lo, hi = int(_LEAD * FS), int(_LEAD * FS) + len(clipped)
        assert _covers(pieces, lo, hi), (
            f"an ack {missing} samples short arrived as "
            f"{[round(len(p)/FS, 3) for _, p in pieces]}")


def test_two_bursts_are_still_two_pieces():
    """The property the first-falling-edge cut was protecting, which must survive:
    a following burst must not be swallowed into the one before it. The gateway
    keys its answers a second or so apart, and a merged pair demodulates as
    neither."""
    gap = np.zeros(int(0.8 * FS))
    pieces = _pieces(_on_air(np.concatenate([_ACK, gap, _ACK])))
    assert len(pieces) == 2, (
        f"two bursts 0.8 s apart arrived as {len(pieces)} piece(s): "
        f"{[round(len(p)/FS, 3) for _, p in pieces]}")


def test_a_fade_longer_than_the_hang_yields_two_addable_fragments():
    """Past the hang time the burst genuinely does end, and the two fragments are
    both handed over — neither dropped for being short, which is what left the
    recogniser with one 6-tone piece and nothing to add to it. Between them they
    still hold the whole burst apart from the notch.

    The notch has to reach the channel noise to count. A -60 dB notch does not:
    the ack's RMS is 0.35 against 1e-4 of channel noise, so 60 dB down is still
    11 dB above the floor, and a gate that ends the burst there is ending it on a
    signal it can still hear. The running-minimum floor used to split there
    anyway, because its 1%-per-frame climb toward a frame 500x its own value lifts
    it 22 dB in the single frame that straddles the burst's leading edge — the
    burst set its own exit threshold."""
    pieces = _pieces(_on_air(_faded(_ACK, 0.26, 0.20, -80)))
    assert len(pieces) == 2, f"{len(pieces)} pieces, expected the burst to split in two"
    lo = int(_LEAD * FS)
    head, tail = pieces
    assert head[0] <= lo, "the head of the burst was lost"
    assert tail[0] + len(tail[1]) >= lo + len(_ACK), "the tail of the burst was lost"


def test_a_continuous_carrier_still_ends_a_burst():
    """Without the force-close a carrier — someone tuning up, a stuck transmitter —
    grows one burst until the process runs out of memory. Every piece stays inside
    the maximum, which is also the handshake's worst-case wait for a burst it has
    to answer on cadence."""
    t = np.arange(int(30 * FS)) / FS
    audio = np.concatenate([np.zeros(FS), 0.5 * np.sin(2 * np.pi * 1500 * t)])
    pieces = _pieces(audio + np.random.default_rng(0).standard_normal(len(audio)) * 1e-4)
    assert pieces, "a 30 s carrier produced no burst at all"
    longest = max(len(p) for _, p in pieces) / FS
    assert longest <= kc.SEG_MAX_S + 0.1, (
        f"a burst ran to {longest:.1f} s, past the {kc.SEG_MAX_S} s force-close")


# --------------------------------------------------------------------------- #
# The noise floor the gate thresholds against. It was a running minimum, which
# cannot rise, so the first dropout deep enough set it for the rest of the run —
# and a dropout deep enough is just this rig keying, which mutes its own receive
# audio 35 dB down for a couple of seconds at a time.
class _MonitorGate(VM.MonitorGate):
    """The monitor's own gate, handing over the whole bracket — so a test measures
    what the gate did, not what find_bursts then did to it. It differs from the
    connect tool's in two ways that both belong to it: it opens at 2.5x rather than
    2.0x, and it keeps the frames its receiver was muted through out of the floor
    entirely. See ``vara_monitor.MonitorGate``; the traffic-log consequences of
    both are measured in ``test_monitor_traffic.py``."""

    def _crop(self, chunk, base):
        return [(base, chunk)]


def _brackets(audio, seg=None):
    seg = seg if seg is not None else kc._BracketSegmenter()
    out = []
    for i in range(0, len(audio), 4800):
        out += seg.push(audio[i:i + 4800])
    return out + seg.flush()


def _forced(pieces, max_s) -> int:
    """Brackets that ran to the cap instead of closing on trailing silence."""
    return sum(len(p) / FS >= max_s - 0.05 for _, p in pieces)


def _frame_rms(x) -> np.ndarray:
    """Frame RMS on the gate's own analysis frame — what the floor is read from."""
    fr = x[:len(x) // FRAME * FRAME].reshape(-1, FRAME)
    return np.sqrt((fr * fr).mean(1))


def _noise(secs, rms=1e-3, seed=0):
    return np.random.default_rng(seed).standard_normal(int(secs * FS)) * rms


def test_a_dropout_does_not_pin_the_noise_floor():
    """The defect, at its smallest: two seconds of muted receiver in the middle of
    an otherwise ordinary channel.

    A running minimum takes the dropout as the floor and never comes back, so
    every later frame is 35 dB above it: the twenty seconds of ordinary band noise
    that follow are bracketed as two force-closed bursts, and the floor ends the
    run at -88.9 dBFS against a band noise of -60."""
    audio = np.concatenate([_noise(20), _noise(2, 1e-3 * 10 ** (-35 / 20), seed=1),
                            _noise(20, seed=2)])
    seg = _MonitorGate()
    pieces = _brackets(audio, seg)
    assert not pieces, (
        f"{len(pieces)} burst(s) found in band noise either side of a dropout: "
        f"{[round(len(p)/FS, 2) for _, p in pieces]}")
    floor_db = 20 * np.log10(seg.nf)
    assert floor_db > -65, (
        f"the floor finished at {floor_db:.1f} dBFS, {-60 - floor_db:.0f} dB below "
        f"the band noise it is supposed to describe")


def _busy_channel(n=30, on=0.8, off=0.2, band=1e-3, amp=10.0):
    """A channel occupied ``on/(on+off)`` of the time by bursts ``amp`` x the band."""
    parts = [_noise(2.0, band, seed=99)]
    for k in range(n):
        parts += [_noise(on, band * amp, seed=1000 + k), _noise(off, band, seed=2000 + k)]
    return np.concatenate(parts)


def test_a_burst_does_not_raise_the_floor_the_next_burst_has_to_clear():
    """Thirty bursts on a channel busy 80% of the time, 20 dB over the band.

    The floor is a low quantile of a 30 s window, so a window fed from every frame
    is mostly fed from the traffic: it settles at the level of the bursts, and the
    threshold over it is then twice the amplitude of the very signal it is meant to
    find. Each burst pushes the bar the next one has to clear. Measured on
    rf-corpus ``pos_vara_session`` — 44 s of real VARA session traffic — that is a
    floor walking 18.2 dB up across the run and two bursts bracketed out of
    fourteen; here it is 25 of 30, and the run ends with the gate shut for good.
    """
    audio = _busy_channel()
    band = 1e-3
    every = float(np.quantile(_frame_rms(audio), VM.SEG_FLOOR_Q))
    assert every > band * kc._BracketSegmenter.enter, (
        f"this channel is no longer busy enough to demonstrate anything: a floor "
        f"read from every frame would sit at {every:.5f}, already below the "
        f"{band * kc._BracketSegmenter.enter:.5f} it has to reach to shut the gate")

    seg = kc._BracketSegmenter()
    pieces = _brackets(audio, seg)
    assert len(pieces) == 30, f"{len(pieces)} of 30 bursts bracketed"
    assert abs(20 * np.log10(seg.nf / band)) < 1.5, (
        f"the floor finished {20 * np.log10(seg.nf / band):+.1f} dB off the band "
        f"noise, having read the traffic instead of the channel")


def test_a_carrier_that_never_falls_silent_becomes_the_floor():
    """The other side of it: withholding burst frames must not freeze the floor.

    A floor that only ever learns between bursts has nothing to learn from while a
    signal runs without a gap, and it stays pinned at whatever preceded the signal
    — so every bracket runs to the cap, for as long as the carrier lasts. A burst
    force-closed at ``max_s`` fell silent nowhere in ``max_s`` seconds, which makes
    it channel and not keying, so its frames go into the window: one bracket here
    rather than ten.
    """
    audio = np.concatenate([_noise(2.0, 1e-3, seed=7), _noise(60.0, 1e-2, seed=8)])
    seg = kc._BracketSegmenter()
    pieces = _brackets(audio, seg)
    assert _forced(pieces, kc.SEG_MAX_S) == 1, (
        f"{_forced(pieces, kc.SEG_MAX_S)} force-closed brackets in one carrier — "
        f"the floor never learned it")
    assert abs(20 * np.log10(seg.nf / 1e-2)) < 2.0, (
        f"the floor finished {20 * np.log10(seg.nf / 1e-2):+.1f} dB off the carrier "
        f"it spent a minute inside")


# --------------------------------------------------------------------------- #
# A STEP in what the receiver delivers, which is the other way the floor can be
# left describing a channel that is gone. `_BracketSegmenter._close` carries the
# on-air measurement; these are the two halves of what the rule has to hold.
_STEP_DB = 3.5                      # measured across our own CR3, 2026-09-11


def _stepped_band(band=1e-3, before=20.0, after=40.0):
    """Twenty seconds of band, then the same band 3.5 dB louder for forty more.

    The shape of `logs/onair/20260911T043103Z-W9SSJ-K5FIT.wav`, where the
    receiver's own level stepped -17.5 -> -13.7 dBFS across our third
    connect-request and stayed there: broadband, no carrier, no band edges, and
    the operator heard nothing through any of it.
    """
    return np.concatenate([_noise(before, band, seed=11),
                           _noise(after, band * 10 ** (_STEP_DB / 20), seed=12)])


def test_a_step_in_the_receiver_level_does_not_latch_the_gate_open():
    """Calling K5FIT on 2026-09-11, and the reason requests 4 and 5 never went out.

    The floor is a quantile over SEG_FLOOR_S of between-bracket frames, so it
    lags a step by design — and while a bracket is open it does not move at all.
    A step of a few dB therefore opens a bracket on band noise, the frozen floor
    holds it to the cap, and the frames the force-close contributes are
    outnumbered in a full window by the band that no longer exists. So the next
    frame opens the next bracket: on the air that was 2 x 12.011 s plus twelve
    minimum-length brackets, 26.84 s of empty 40 m handed to the connect loop as
    receive activity, deferring the CR train for 35.72 s against a 3.0 s cadence.
    The replay is in `working/vara-evening-en63bc-0910/analysis/k5fit-burst/`;
    this reproduces the same 14 brackets from noise alone.

    One bracket is the floor honestly lagging a step and is the cost of the
    quantile. A second one is the latch, and there must not be one.
    """
    seg = kc._BracketSegmenter()
    pieces = _brackets(_stepped_band(), seg)
    assert len(pieces) <= 1, (
        f"{len(pieces)} brackets out of band noise either side of a "
        f"{_STEP_DB} dB step: {[round(len(p) / FS, 3) for _, p in pieces]} — the "
        f"gate is latched on a floor that describes a channel that is gone")
    band = 1e-3 * 10 ** (_STEP_DB / 20)
    assert 20 * np.log10(seg.nf / band) > -_STEP_DB, (
        f"the floor finished {20 * np.log10(seg.nf / band):+.1f} dB under the band "
        f"it spent forty seconds in, still describing the level before the step")


def test_the_gate_still_hears_a_peer_through_the_step():
    """The half that must not be bought: superseding a floor is not deafening it.

    The rule fires on a bracket whose own median never reached its entry
    threshold, so what it hands the floor is the band — not a signal. A real
    answer after the step has to open the gate exactly as it would have before
    it. 13 dB over the band is where a VARA over's median frame sits.
    """
    answer = _noise(0.5, 1e-3 * 10 ** ((_STEP_DB + 13) / 20), seed=13)
    audio = np.concatenate([_stepped_band(), answer, _noise(2.0, 1e-3 * 10 **
                                                            (_STEP_DB / 20), seed=14)])
    lo = len(audio) - len(answer) - 2 * FS
    pieces = _brackets(audio, kc._BracketSegmenter())
    assert _covers(pieces, lo, lo + len(answer)), (
        f"a peer 13 dB over the band went unbracketed after the step; the gate "
        f"handed over {[round(len(p) / FS, 3) for _, p in pieces]}")


@corpora.requires_onair_connect_attempt
def test_the_gateway_that_answered_on_air_reaches_the_handshake():
    """The on-air failure of 2026-07-26, off the recording made during it.

    kestrel called KB9MMT eight times and reported no answer, printing a
    ``wideband ~6.0 s over (not decoded)`` about every six seconds — the
    force-close cadence, on a channel where nothing was keyed. KB9MMT had in fact
    answered twice, at 56.02 s and 63.98 s, both decoding 15/15.

    Two things had to be wrong for that. ``AudioVaraIO.tx`` skips everything
    recorded during our own over plus a 0.1 s codec tail, but the rig's receive
    mute outlasts the audio we played by about 0.22 s, so a few frames 35 dB below
    the band noise reach the gate after every transmission — which is all it takes
    to pin a running minimum, and the 133.6 s of receive audio reconstructed that
    way produced 21 brackets, every one force-closed at 6.02 s. Then, with
    the floor tracking properly, the gate has to open on a reply that is only
    6.2 dB above it: this receiver was run with so little dynamic range that its
    band noise sits 4.8 dB under the gateway.

    The force-close assertion is about *what* ran to the cap rather than whether
    anything did. Those 21 brackets were band noise end to end — the cap firing
    because the gate had lost the channel, which is the defect. One bracket does
    reach the cap here, at 122.71 s, and it is 6.8 s of unbroken transmission from
    the other station: its 10th-percentile frame sits at 0.385 against a stream
    median of 0.255, so the cap is cutting a signal, which is what it is for. The
    gate only found it at all once the floor stopped reading the traffic.
    """
    x = corpora.wav_mono(corpora.ONAIR_CONNECT_ATTEMPT)
    x = x / (np.abs(x).max() or 1.0)
    rx = _receive_windows(x)
    pieces = _brackets(rx)
    band = float(np.median(_frame_rms(rx)))
    for s, p in pieces:
        if len(p) / FS < kc.SEG_MAX_S - 0.05:
            continue
        occupied = float(np.quantile(_frame_rms(p), 0.10))
        assert occupied > band, (
            f"the bracket at {s / FS:.2f} s ran to the {kc.SEG_MAX_S} s cap with "
            f"nine tenths of it above only {occupied:.3f}, under the {band:.3f} "
            f"median of the receive audio — the gate lost the channel rather than "
            f"cut a signal")
    kind = VF.CONNECT_RESPONSE
    n_sym = len(kind.preamble) + kind.n_payload
    best = 0
    for _s, p in pieces:
        for off in range(0, max(1, len(p) - MK.STRIDE - (n_sym - 1) * MK.HOP), FRAME):
            _c, m, _tot = VF.best_match(MK.demod_tones(p[off:], n_sym), ["KB9MMT"], kind)
            best = max(best, m)
    assert best >= 12, (
        f"KB9MMT answered twice and the best of the {len(pieces)} bursts handed "
        f"over matched {best}/15 of its connect-response")


def _receive_windows(x, tail_s: float = 0.1):
    """The audio ``AudioVaraIO`` actually scans: everything but our own overs.

    ``tx()`` advances the consumed cursor past everything recorded through the end
    of our own over, so what the gate sees of a transmission is the last of the
    receiver mute — ``tail_s`` of it, kept here because the live path used to leave
    that much in and the gate has to hold up against it. It now waits ``TX_IDLE_HOLD_S``
    before taking the cursor and is handed less. Transmissions are found as runs of
    frames 15 dB below the median lasting over a second — 2.1 s each, eight of them,
    against a band noise that never moves more than 6 dB.
    """
    n = len(x) // FRAME
    r = np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(1)) + 1e-12
    muted = r < np.median(r) * 10 ** (-15 / 20)
    keep, i = np.ones(n, bool), 0
    while i < n:
        j = i
        while j < n and muted[j]:
            j += 1
        if (j - i) * FRAME / FS > 1.0:
            keep[i:j - int(tail_s * FS / FRAME)] = False
        i = j + 1
    return np.concatenate([x[k * FRAME:(k + 1) * FRAME] for k in range(n) if keep[k]])


@corpora.requires_gateway_session
def test_the_off_air_sessions_bracket_without_force_closing():
    """What the traffic monitor gets out of the two recorded gateway sessions.

    A force-close is the gate admitting it has lost track of the channel: it hands
    over a fixed-length slice with a burst boundary somewhere inside it. Against
    the running minimum 7 of 10 brackets were force-closes on each session and the
    median bracket was the full 12.01 s cap. Measured after: 0 of 14 and 0 of 20,
    median 1.61 s and 1.41 s, which is the length of the bursts that are actually
    there.
    """
    for name in ("NS0A_2300", "KC9GHZ_2300"):
        path = corpora.OFFAIR / name / "rig_rx.wav"
        if not path.exists():
            pytest.skip(f"off-air recording for {name} not present")
        x = corpora.wav_mono(path)
        x = x / (np.abs(x).max() or 1.0)
        pieces = _brackets(x, _MonitorGate())
        durs = sorted(len(p) / FS for _, p in pieces)
        assert _forced(pieces, VM.SEG_MAX_S) == 0, (
            f"{name}: {_forced(pieces, VM.SEG_MAX_S)} of {len(pieces)} brackets ran "
            f"to the {VM.SEG_MAX_S} s cap")
        assert durs[len(durs) // 2] < 3.0, (
            f"{name}: median bracket {durs[len(durs)//2]:.2f} s — the gate is "
            f"welding bursts together, not bracketing them")


@corpora.requires_clear_channel
def test_a_quiet_channel_opens_no_bursts():
    """The other side of the sensitivity trade, for both gates. The monitor opens
    at 4.0x the tracked floor and the connect tool at 2.0x, against a measured
    first false open at 2.0x over 84 s of real receiver audio with nothing in it —
    30 s of verified clear channel here, plus the 54 s of receive window between
    our own overs on 2026-07-26. Neither may find a burst on an empty band."""
    x = corpora.wav_mono(corpora.CLEAR_CHANNEL)
    x = x / (np.abs(x).max() or 1.0)
    for seg in (_MonitorGate(), kc._BracketSegmenter()):
        pieces = _brackets(x, seg)
        assert not pieces, (
            f"{seg.enter}x found {len(pieces)} burst(s) on 30 s of verified clear "
            f"channel: {[round(len(p)/FS, 2) for _, p in pieces]}")


@pytest.mark.parametrize("snr_db", (12, 16, 20))
def test_the_monitor_gate_opens_on_a_burst_12_dB_above_the_floor(snr_db):
    """Sensitivity is the enter threshold and nothing else: the gate brackets a
    burst whose frame RMS is SEG_ENTER above the floor, and misses what is below.
    Measured over five noise seeds a synthetic connected-ack is bracketed whole
    5/5 from 12 dB SNR at the monitor's 4.0x, and only from 16 dB at the 6.0x it
    replaces — which is the margin a weak gateway on a noisy band has to live
    in."""
    at = int(8 * FS)
    for seed in range(5):
        y = np.random.default_rng(seed).standard_normal(at + len(_ACK) + 4 * FS)
        y[at:at + len(_ACK)] += _ACK / np.sqrt((_ACK ** 2).mean()) * 10 ** (snr_db / 20)
        pieces = _brackets(y / np.abs(y).max(), _MonitorGate())
        assert _covers(pieces, at, at + len(_ACK)), (
            f"seed {seed}: an ack {snr_db} dB over the noise arrived as "
            f"{[round(len(p)/FS, 2) for _, p in pieces]}")


# --------------------------------------------------------------------------- #
# The plumbing around the gate: the recorder's buffer, and what a transmission
# does to it. Also never tested, and the rig hearing itself is the loudest thing
# on the input.
class _FakeStream:
    def __init__(self, **kw): ...
    def start(self): ...
    def stop(self): ...
    def close(self): ...


class _FakeSD:
    """Enough of sounddevice to build the transport. No device is opened and
    nothing is played, which is what makes this safe to run at the station."""

    def query_devices(self, name): return {"max_output_channels": 2}
    def InputStream(self, **kw): return _FakeStream(**kw)
    def OutputStream(self, **kw): return _FakeOutput()


class _FakeOutput:
    """A transmit stream that consumes and emits nothing, in no time at all."""

    latency = 0.0

    def start(self): ...
    def write(self, block): ...
    def stop(self): ...
    def close(self): ...


def _transport(monkeypatch):
    """One codec both ways, which is the station: the rig monitors its own
    transmit into the receive path, so the echo guard has an echo to guard
    against  [kestrel_connect.AudioVaraIO.tx]."""
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD())
    return kc.AudioVaraIO("codec", "codec")


class _Times:
    def __init__(self, adc: float):
        self.currentTime = adc
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = adc


class _Flags:
    input_overflow = input_underflow = output_underflow = False
    priming_output = False

    def __bool__(self) -> bool:
        return False


def _record(io, x) -> None:
    """One callback's worth, with the converter's timestamps the real one gets.

    `_on_rx` reads them: the capture clock is anchored on `inputBufferAdcTime`,
    so a stand-in that hands over `None` is not a stand-in for this callback.
    """
    x = np.asarray(x, np.float32).reshape(-1, 1)
    io._on_rx(x, len(x), _Times(io.samples / kc.FS), _Flags())


def test_our_own_transmission_is_not_handed_back_as_a_received_burst(monkeypatch):
    """The rig monitors its own transmit into the receive codec, so every over we
    key comes straight back in at full scale. Recognising our own connect-request
    as the gateway's answer is the other way this tool can report a connection
    that did not happen."""
    io = _transport(monkeypatch)
    t = np.arange(int(0.4 * FS)) / FS
    ours = 0.9 * np.sin(2 * np.pi * 500 * t)       # our over, at a tone nothing else uses
    _record(io, np.zeros(int(0.3 * FS)))
    _record(io, ours)                              # the rig hearing us key
    assert io.next_rx_burst(timeout=0.0) is None   # the gate is now mid-bracket on it
    io.tx(np.zeros(int(0.2 * FS)))                 # ...which tx() marks as ours
    _record(io, np.zeros(int(0.4 * FS)))
    _record(io, _ACK)                              # the gateway's actual answer
    _record(io, np.zeros(int(0.4 * FS)))

    burst = io.next_rx_burst(timeout=0.5)
    assert burst is not None, "the gateway's answer never arrived"
    spectrum = np.abs(np.fft.rfft(burst * np.hanning(len(burst))))
    f = np.fft.rfftfreq(len(burst), 1 / FS)
    ours_share = spectrum[(f > 450) & (f < 550)].sum() / spectrum.sum()
    # Measured: 0.003 with the gate restarted at key-down, 0.785 without — the
    # bracket left open across our own over splices it onto the answer, and the
    # spliced piece is not recognised as the ack at all.
    assert ours_share < 0.05, (
        f"{ours_share:.0%} of the burst handed over is our own transmission")
    assert _connects([(0, burst)]), "the answer did not survive the splice"
    assert io.next_rx_burst(timeout=0.2) is None, "our own over came back as a burst"


def test_the_cursor_lands_on_our_last_sample_and_not_on_the_drains_return(monkeypatch):
    """A slow transmit path is not the channel being busy, and it used to be read
    as one.

    `play_drained` returns at a wall-clock instant and the buffer's length there
    stood in for the end of our burst. On the three 2026-08-29 arms it lands
    0.00-0.06 s past our last sample on 48 ordinary turnarounds and 0.62 s past it
    on the three that follow the 11-symbol control burst — where the gateway keys
    the release that hands us the channel, so 0.58 s of a 0.71 s frame went behind
    the cursor three times out of three and the handover was answered with
    keepalives.

    The rig monitors us into the receive codec, so our own last sample is in this
    stream: it is the loud run's last edge, and the cursor is set from that.
    """
    io = _transport(monkeypatch)
    rng = np.random.default_rng(0)
    band, mute = 0.5, 0.06
    ours = rng.normal(0, 0.05, int(0.47 * FS))       # the control burst, monitored back
    peer = rng.normal(0, 0.05, int(0.48 * FS))       # the release, keyed into our tail

    def _slow_drain(block, fs, device):
        _record(io, np.zeros(int(mute * FS)))        # the key-up mute
        _record(io, ours)
        _record(io, np.zeros(int(0.16 * FS)))        # the blackout behind the unkey
        _record(io, peer)                            # ...and the peer, while the output
                                                     #    stream is still being closed

    monkeypatch.setattr(kc, "play_drained", _slow_drain)
    _record(io, rng.normal(0, 0.05, int(band * FS)))
    io.tx(np.zeros(len(ours)))

    end = int((band + mute) * FS) + len(ours)
    assert abs(io._gap_from - end) <= int(kc._MUTE_FRAME_S * FS), (
        "the cursor is anchored somewhere other than our own last sample")
    assert io._consumed >= end, "the guard no longer covers our own transmission"
    assert io._consumed < end + int(0.16 * FS), (
        "the peer's answer is behind the cursor, which is the whole defect")


def test_the_changeover_blackout_is_reported_from_our_last_transmitted_sample(
        monkeypatch, capsys):
    """What the receiver missed of a peer's answer, measured where the peer starts.

    The line used to begin its clock where the echo guard ended and report only
    what was still dead beyond it, so the 2026-08-18 KC9GHZ log carried six gaps
    of 0.02-0.06 s against a real blackout of 0.12-0.16 s — a peer answers from
    0.10 s, so the number that reads as negligible and the number that eats the
    head of an answer were the same measurement. Starting it at the cursor
    instead left the same understatement a codec buffer wide: the 2026-08-23
    session printed +0.10 to +0.18 s over changeovers its recordings hold
    0.16-0.23 s of dead input on.

    Here the receiver is transmitting into its own monitor when `tx` takes the
    cursor, so the window opens on the end of our own burst and the blackout is
    the dead run itself — the guard, and the 0.04 s past it.
    """
    io = _transport(monkeypatch)
    guard = io.tx_tail + kc.RX_ECHO_GUARD_S
    rng = np.random.default_rng(0)
    _record(io, rng.normal(0, 0.05, int(0.3 * FS)))        # our own burst, monitored
    io.tx(np.zeros(int(0.2 * FS)))
    assert io._consumed - io._gap_from == io._gap_lead, (
        "the window and the cursor no longer meet")
    _record(io, np.zeros(int((guard + 0.04) * FS)))        # the mute, either side of it
    _record(io, rng.normal(0, 0.05, kc.GAP_WINDOW))        # the band, back
    io.next_rx_burst(timeout=0.0)

    line = next(ln for ln in capsys.readouterr().out.splitlines()
                if "post-TX level threshold" in ln)
    assert f"+{guard + 0.04:.2f} s after our last transmitted sample" in line, line
    assert f"the cursor stands at +{guard:.2f}" in line, line
    assert f"+{kc.PEER_ANSWERS_FROM_S:.2f} s" in line, line
    assert "not a detected peer reply" in line


# Two-millisecond levels off `logs/onair/20260911T042115Z-W9SSJ-KC9GHZ.wav` at
# 62.025 s, the end of the first DATA over: the burst's own taper, the 25-28 dB
# step at the last sample, and the codec bleed behind it — which holds -45 to
# -50 dBFS for the next 0.06 s and then decays. The bleed is our transmission
# leaving a receiver that is already muted, and it stands over DEAF_DBFS.
_KC9_TAIL_DB = (-19, -16, -18, -21, -23, -25, -28, -30, -32, -34,
                -44, -49, -63, -56, -50, -48, -47, -46, -45, -45,
                -45, -46, -46, -46, -47, -47, -48, -48, -49, -50,
                -50, -51, -52, -52, -53, -54, -54, -55, -56, -57)
_KC9_LAST = 10                     # of those, the ones that are still the burst


def test_our_own_burst_leaving_a_muted_receiver_is_not_the_band_coming_back(
        monkeypatch, capsys):
    """The blackout is over when the BAND is back, not when a frame clears the floor.

    The 2026-09-11 KC9GHZ arms printed +0.04 s over changeovers whose recordings
    hold 0.17: the first frame over DEAF_DBFS is the bleed above, not the channel,
    and a number that says the receiver was away for one symbol where it was away
    for four reads as a path with nothing wrong with it. What separates the two is
    that the bleed is on its way down and the band is not, so liveness is a run
    [kestrel_connect._LIVE_RUN_S] and the line is quoted from the burst's own edge
    [_tx_last_sample] rather than from the frame that edge falls in.
    """
    io = _transport(monkeypatch)
    rng = np.random.default_rng(0)
    sub = int(kc._MUTE_FRAME_S * FS / 10)
    blind_s = 0.17
    levels = np.repeat(10.0 ** (np.asarray(_KC9_TAIL_DB) / 20.0), sub)
    tail = rng.normal(0, 1, len(levels)) * levels
    ours = np.concatenate([rng.normal(0, 0.03, int(0.45 * FS)), tail[:_KC9_LAST * sub]])
    leaving = tail[_KC9_LAST * sub:]
    blackout = np.concatenate([leaving, rng.normal(0, 10 ** (-57 / 20),
                                                   int(blind_s * FS) - len(leaving))])

    def drain(_block, _fs, _device):
        _record(io, ours)                                  # our over, monitored back
        _record(io, blackout)                              # ...and its way out
        _record(io, rng.normal(0, 0.05, kc.GAP_WINDOW))    # the band, in one step

    monkeypatch.setattr(kc, "play_drained", drain)
    _record(io, rng.normal(0, 0.05, int(0.5 * FS)))
    io.tx(np.zeros(len(ours)))
    io.next_rx_burst(timeout=0.0)

    line = next(ln for ln in capsys.readouterr().out.splitlines()
                if "post-TX level threshold" in ln)
    blind = float(line.split("+")[1].split()[0])
    assert blind == pytest.approx(blind_s, abs=kc._MUTE_FRAME_S), (
        f"the receiver was deaf for {blind_s:.2f} s: {line}")


def test_a_shorter_keyed_tail_shortens_the_blackout(monkeypatch):
    """`--tx-tail` is documented against the blackout it moves, and at the flat
    0.1 s guard this replaces it moved the keying and not the blackout."""
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD())
    io = kc.AudioVaraIO("codec", "codec", tx_tail=0.0)
    io.tx(np.zeros(int(0.2 * FS)))
    assert io._consumed == int(kc.RX_ECHO_GUARD_S * FS)


def test_the_cursor_never_skips_past_the_earliest_a_peer_can_answer(monkeypatch):
    """The one bound the echo guard has, and the reason it needs no other.

    What the guard discards is measured dead. Over the 56 changeovers in the two
    KC9GHZ session recordings this station's input sits at the codec floor for
    0.088-0.244 s past our last transmitted sample, and our own burst is 35 dB
    down in the first two-millisecond frame past its last, so 55 times out of 56
    the guard ends while the receiver is still deaf and there is neither an echo
    nor an answer in what it dropped. That makes it safe rather than free: a
    guard reaching past the turnaround would skip a peer's answer whatever the
    receiver was doing, and the turnaround is the only number that says how long
    it may be.
    """
    io = _transport(monkeypatch)
    io.tx(np.zeros(int(0.2 * FS)))
    assert io._consumed <= int(kc.PEER_ANSWERS_FROM_S * FS), (
        f"the cursor skips {io._consumed / FS:.3f} s, past the "
        f"+{kc.PEER_ANSWERS_FROM_S:.3f} s a peer has been heard to answer from")


def test_an_unreadable_burst_is_logged_while_a_turn_request_is_outstanding(
        monkeypatch, capsys):
    """The line the 2026-08-22 KC9GHZ session needed and did not print.

    A bracket the handshake cannot name is noise on a live band, and dropping it
    keeps the log about the attempt. Between a turn-request and its answer it is
    not noise: KC9GHZ answered all three of that session's requests, the gate rode
    the band open and handed each 1.387 s frame over inside a 6.016 s bracket, and
    every one was dropped here — so the log the operator read showed unbroken
    silence against a gateway that was transmitting into every turnaround.
    """
    io = _transport(monkeypatch)
    unread = "rx burst with 139 symbols — not an MFSK handshake burst"
    hs = VaraStationHandshake(["W9SSJ"], io, bw="2300")

    io.log(unread)
    assert unread not in capsys.readouterr().out, "logged with no handshake to ask"
    io.hs = hs
    io.log(unread)
    assert unread not in capsys.readouterr().out, "logged with nothing outstanding"

    hs.turn = kc._TURN_ASKED
    io.log(unread)
    assert unread in capsys.readouterr().out, (
        "a burst arriving on an outstanding turn-request is still silent")


def test_the_recorded_buffer_does_not_grow_with_the_session(monkeypatch):
    """A connect attempt runs for minutes at 48 kHz. The scanned audio is dropped
    as it is consumed; only the tail the gate still holds may stay."""
    io = _transport(monkeypatch)
    for _ in range(40):
        _record(io, np.zeros(int(0.25 * FS)))
        io.next_rx_burst(timeout=0.0)
    assert sum(len(b) for b in io._buf) <= FS, "the RX buffer is accumulating the session"


def test_the_real_gateway_answer_arrives_in_time_to_be_answered():
    """The latency guard, on the one recorded gateway answer we have.

    offair/NS0A_2300 holds a genuine NS0A connect-response at 9.79-10.77 s
    (kestrel/tests/kestrel/test_handshake_offair.py locates it). A burst reaches the
    handshake only when its bracket closes, and with the floor pinned by the
    first dropout nothing ever closed one: this burst was delivered at 17.2 s, six
    and a half seconds after it ended and long past any cadence worth answering
    on. With the floor tracked it arrives at 11.0 s, and that is the floor rather
    than the 6 s force-close doing it — 11.0 s at a 12 s cap too.
    """
    path = corpora.OFFAIR / "NS0A_2300" / "rig_rx.wav"
    if not path.exists():
        pytest.skip("off-air recording for NS0A_2300 not present")
    x = corpora.wav_mono(path)
    x = x / (np.abs(x).max() or 1.0)
    hs = VaraStationHandshake(["W9SSJ"], _IO(), bw="2300", mfsk_only=True)
    hs.originate("NS0A", "W9SSJ")
    hs.step = _I_CR_SENT
    when = None
    for s, p in _pieces(x[:int(20 * FS)]):
        hs.on_rx_audio(p)
        if hs.step == _I_LINKSETUP_SENT and when is None:
            when = (s + len(p)) / FS
    assert when is not None, "the real gateway connect-response was not recognised"
    assert when - 10.77 <= 1.0, (
        f"the connect-response ended at 10.77 s and reached the handshake at "
        f"{when:.1f} s — too late to answer")


# --------------------------------------------------------------------------- #
# The floor the gate opens against describes the band, not the answer.
_RESPONSE = MK.synth_burst(_CALLED, VF.CONNECT_RESPONSE)


def _band(secs: float, seed: int, rms: float = 0.02) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal(int(secs * FS))
    return x * (rms / np.sqrt((x * x).mean()))


def _under(x, seed: int) -> np.ndarray:
    return np.asarray(x, float) + _band(len(x) / FS, seed)


def _answered_first_time(listen_s: float, lead_s: float = 0.1):
    """Every bracket the gate produces around the ack, when the gateway answered
    the *first* connect-request, and where the ack sits inside that window.

    Faithful to the order the live path runs in: ``prime_floor`` pushes the listen
    window before anything is keyed, ``tx`` restarts the gate after each of our own
    overs, and the consumed cursor skips what we transmitted — so between our
    connect-request and our link-setup the gate is handed only the second or so
    that carries the gateway's answer. ``lead_s`` is how much of the band arrives
    ahead of that answer; at 0 the very first frame the gate ever sees is signal.
    """
    seg = kc._BracketSegmenter()
    if listen_s:
        seg.push(_band(listen_s, 1))
    seg.restart()                                          # our connect-request
    mid = [_under(_RESPONSE, 3), _band(0.1, 4)]
    if lead_s:
        mid.insert(0, _band(lead_s, 2))
    seg.push(np.concatenate(mid))
    seg.restart()                                          # our link-setup
    tail = np.concatenate([_band(0.1, 5), _under(_ACK, 6), _band(1.5, 7)])
    out = []
    for i in range(0, len(tail), 4800):
        out += seg.push(tail[i:i + 4800])
    lo = int(0.1 * FS)
    return out + seg.flush(), lo, lo + len(_ACK)


def test_the_gate_takes_its_floor_from_the_band_and_not_from_the_answer():
    """The gateway's own answer must not be the thing that sets the gate shut.

    ``Segmenter`` opens above ``enter`` x a quantile of its recent frames, and on
    the live path those frames only start arriving once the first connect-request
    has gone out. The first thing it ever sees is then the connect-response, and a
    quantile over a second of it sits at the level of the signal: the gate holds at
    twice the response's own amplitude and the connected-ack behind it never opens
    it. The attempt ends as "NOT connected" against a station that answered
    everything — and only when the station answered *promptly*, which is why it
    reads as intermittent. Against a real VARA over the loopback at 20 dB SNR that
    was 2 connects in 6, the four failures being exactly the four first-time
    answers; with the listen window supplying the floor it is 6 in 6.

    Only the frames outside a burst feed the window now, which takes most of the
    weight off that first second: a tenth of a second of band ahead of the answer
    is enough, because the answer itself no longer goes in. What is left is the
    case with no band ahead of it at all, and that is the counterexample below —
    the gate's first frame is the gateway, so the quantile it reads is the gateway,
    and the ack behind it is lost exactly as before. Priming is what guarantees the
    window holds band audio rather than however much of it happened to arrive
    before the answer did.
    """
    assert _covers(*_answered_first_time(listen_s=8.0)), (
        "the connected-ack was not bracketed although the band had been measured "
        "for 8 s before the first request")
    assert not _covers(*_answered_first_time(listen_s=0.0, lead_s=0.0)), (
        "this sequence no longer reproduces the failure, so the assertion above "
        "no longer demonstrates anything")


# --------------------------------------------------------------------------- #
# The held-lift entry, and the two confirmed sessions it exists for.

def _gate_bursts(path):
    x = corpora.wav_mono(path)
    x = x / 32768.0 if np.abs(x).max() > 1.5 else x
    gate = VM.MonitorGate()
    got = [b for i in range(0, len(x), 4800) for _, b in gate.push(x[i:i + 4800])]
    return got + [b for _, b in gate.flush()]


@corpora.requires_monitored_sessions
def test_a_confirmed_session_the_peak_test_cannot_see_still_opens_the_gate():
    """Both sides of ``MONITOR_HELD_ENTER``, on the recordings it was measured on.

    On 2026-08-14 the traffic log's gate handed ``classify`` nothing at all across
    815.6 s of a VARA session the Winlink feed names end to end, and nothing across
    the VARA 500 exchange whose mode and both callsigns a second source attests.
    The loudest 21 ms frame in the first stood +5.98 dB over the gate's own floor
    against the +7.96 dB a 2.5x peak asks for, and the second reached +4.99 dB —
    both of them under, and neither of them retunable, because an empty 40 m
    channel in the same slot peaks at +4.02 dB and a static crash on a summer band
    reaches as far as the over does.

    What separates them is that an over is held. The quiet window has to stay shut
    in the same breath or the threshold has bought nothing.
    """
    for path in corpora.MONITORED_SESSIONS:
        assert _gate_bursts(path), (
            f"{path.parents[1].name} handed nothing to classify, and it carries a "
            "session two independent sources name")
    assert not _gate_bursts(corpora.MONITORED_QUIET), (
        "the ten minutes of 7103.5 kHz that the feed and every classifier agree "
        "were empty opened the gate")


@corpora.requires_onair_gateway_greeting
def test_the_greeting_no_energy_gate_opened_on_now_reaches_its_decoder():
    """The corpus case, and the one that says this is worth having.

    ``ONAIR_GATEWAY_GREETING`` holds one thing the gateway transmitted after the
    connected-ack — its Winlink greeting, an ordinary BW2300 DATA over at
    58.083-62.307 s with 24 of 24 reference columns and a clean CRC — and the mail
    client never left "awaiting greeting", because that over is 4.1 dB over the
    band noise and no energy gate opened on it. It does now, and the frame decodes:
    a CRC is not a statistic, so this is the assertion to keep if any.
    """
    named = [VM.classify(b, [], decode_wideband=True)
             for b in _gate_bursts(corpora.ONAIR_GATEWAY_GREETING)
             if len(b) >= 180000]
    good = [r for r in named if r.kind == "DATA over" and r.quality == "CRC ok"]
    assert good, ("the greeting did not reach a clean CRC; wideband bursts named "
                  + ("; ".join(f"{r.kind} {r.quality}" for r in named) or "none"))


# --------------------------------------------------------------------------- #
# The bracket's length is the burst's length.
#
# ``on_rx_audio`` names a burst by dividing the bracket it was handed, so a
# bracket that carries the gate's own pre-roll and hangover is a burst of the
# wrong length: ``SEG_PAD`` 4 frames plus ``SEG_HANG`` 6 is 10240 samples, which
# is exactly five symbols, and every session frame arrived five symbols long.
# Nothing in ``_NSYM`` answers to 21, 22, 37 or 46, and the +/-3 slop is under
# five and runs only while a kind is expected, which in the connected state is
# never. Measured live on the bench of 2026-08-26: the peer's 32-symbol frames
# reached the handshake as 37 and were logged "not an MFSK handshake burst" —
# every turn grant and every turn-request of the session among them.
_LENGTHS = [
    (len(VF.SESSION_CONFIRM.preamble) + VF.SESSION_CONFIRM.n_payload, "confirm"),
    (len(VF.SESSION_TURN_RELEASE.preamble) + VF.SESSION_TURN_RELEASE.n_payload,
     "turn release"),
    (len(VF.CONNECT_RESPONSE.preamble) + VF.CONNECT_RESPONSE.n_payload,
     "connect response"),
    (len(VF.SESSION_KEEPALIVE_A.preamble) + VF.SESSION_KEEPALIVE_A.n_payload,
     "session frame"),
    (len(VF.CR.preamble) + VF.CR.n_payload, "connect request"),
]


@pytest.mark.parametrize("n_sym,what", _LENGTHS, ids=[w for _, w in _LENGTHS])
def test_a_bracket_is_padded_at_both_ends_and_that_is_deliberate(n_sym, what):
    """The bracket is the burst plus the gate's pre-roll and its hangover.

    Both earn their keep and neither may be trimmed to make a length divide
    cleanly. The pre-roll is the head of a burst the gate opened late on — cut
    it and KB9MMT's real off-air connect-response falls from 15 of 15 to 3. The
    hangover is the trailing window ``demod_tones`` needs for the last symbol of
    a burst that ends at the bracket's edge — cut it and the ack tone track runs
    off the end of a real burst and kestrel's own loopback stops connecting.

    So the padding is a property of the bracket and the reader must not divide
    by it; ``on_rx_audio`` names a burst by recognising it instead.
    """
    alpha = sorted(VF.TONE_ALPHABET)
    burst = MK.synth_tones([alpha[i % len(alpha)] for i in range(n_sym)])
    pieces = _pieces(_on_air(burst))
    assert len(pieces) == 1, f"{what} bracketed as {len(pieces)} pieces"
    start, got = pieces[0]
    assert start <= int(_LEAD * FS), f"{what}: bracket starts inside the burst"
    assert len(got) >= len(burst) + VM.SEG_HANG * FRAME, (
        f"{what}: bracket has no trailing window for the last symbol")


_KINDS = [VF.SESSION_CONFIRM, VF.SESSION_TURN_RELEASE, VF.SESSION_KEEPALIVE_A,
          VF.CR]


@pytest.mark.parametrize("kind", _KINDS, ids=[k.name for k in _KINDS])
def test_the_handshake_reads_a_bracketed_session_frame_as_itself(kind):
    """What ``on_rx_audio`` makes of a real frame that came through the gate.

    Before the pre-roll and the hangover were accounted for, every one of these
    reached the handshake five symbols long and was logged "not an MFSK handshake
    burst" — the peer's turn grants and turn-requests among them, live on the
    bench of 2026-08-26. The confirm and the release are the pair that pins the
    correction's shape: they are 16 and 17 symbols, one pre-roll frame apart, so
    a count alone cannot separate them and the callsign has to.
    """
    seen = []
    hs = VaraStationHandshake(["W9SSJ"], _IO(), bw="2300", mfsk_only=True)
    hs.originate(_CALLED, "W9SSJ")
    hs.state, hs.role = VaraState.CONNECTED, "initiator"
    hs.io.log = seen.append
    for _s, p in _pieces(_on_air(MK.synth_tones(
            VF.handshake_tones(_CALLED, kind)))):
        hs.on_rx_audio(p)
    assert not any("not an MFSK handshake burst" in m for m in seen), seen
