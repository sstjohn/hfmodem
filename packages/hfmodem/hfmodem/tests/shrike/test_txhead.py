# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""What the head of a keyed burst survives, and what the settle actually spends.

Written to settle the hypothesis of 2026-08-14: that gateways may not be
answering because they cannot decode what we transmit,
raised off 19 of 28 bursts logging `PTT LEAD ERODED` at 27-29 ms of the 40 the
FT-891 is set for. It is REFUTED, on recorded audio, and both halves of the
refutation are benched below.

THE OFF-AIR EVIDENCE, which is what actually answers it (measured 2026-08-14 over
`captures/onair-0803-*/kiwi.wav` and `logs/onair/witness-pactor-ws8eoc.wav` --
a KiwiSDR at KF8KK-2, Empire MI, 103 mi away, hearing our own transmitter):

  * nine connect bursts across four recordings decode to `Connect(callsign=
    'WS8EOC')` with all 72 address and 48 redundancy bits exact -- zero errors,
    over the air, at a measured 30.7 dB envelope SNR;
  * the leading edge is SQUARE. 1 ms bins of the 1250-1750 Hz band put the burst
    within 1.6 dB of its body one millisecond after the 50% crossing, with
    nothing above -27 dB in the 5 ms before it; analytic-envelope 10->90% rise is
    3.0 ms against the ideal render's own 2.0 through the same resampling chain.
    No ramp, no ALC droop, no attenuated head;
  * bit 0 is the STRONGEST bit of the head, not the weakest: tone margin
    |M-S|/(M+S) of 0.705 against a 72-bit median of 0.501.

So the rig is fully up before the audio starts, and the erosion costs no head at
all. What the burst can afford to lose anyway is the first half of this file; the
constants it lands on are one bit period, which is the whole margin the 0x55 sync
has.

WHERE THE EROSION LIVES, since it is real and is not what it was read as. It is
the HOLDBACK, exactly, and it is arithmetic rather than jitter:

  * `_tx` takes the pre-key audio with `take_until(boundary - settle_n)`, and
    `read` blocks until the codec has DELIVERED that sample -- one block plus one
    input latency after its ADC instant, which is what `holdback` is defined as;
  * `wait_until(boundary - settle_n)` then finds its deadline already gone and
    returns immediately, so the keyed lead comes out at `settle - holdback`.

Corpus-wide, over 89 session logs under `captures/` and `working/` (measured
2026-08-14): all 89 first connects, which pass `at=None` and never touch that
path, keyed 39-40 ms; all 746 gridded bursts keyed 26-34 ms, median 30. Neither
population reaches the other's range. `holdback` is 13 ms on this machine and
40 - 13 is 27, which is the floor the session report saw.

IT IS DELIBERATELY LEFT ALONE. The other three call sites subtract
`live.holdback` (onair.py:3808, :4375, :4771, and test_duplex.py's own bench), so
restoring the full settle here is one term -- but it costs the last 13 ms before
the key, unbridged and then dropped by `flush_to`, out of a receive budget
`PREKEY_RESERVE_S` documents as having no slack. Against that, the witness
recordings measure the benefit at zero: the FT-891 is already square at 29 ms.
The RF start is untouched either way -- the carrier lands on the boundary in both
orders, `RF started +0.0 ms` -- so what the term would move is the key-down
instant alone.

Run:  pytest hfmodem/tests/shrike/test_txhead.py
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, pactor1

FS = pactor1.FS
BIT_S = 0.010                       # 100 Bd, the speed the connect's address runs at
QUIET_S = 0.5

# What the head survives, from the sweeps below. One bit period is the ceiling
# and the protocol is why: the address field opens on 0x55, which alternates
# every bit, so a lost first bit takes the sync with it. The real over-the-air
# bursts give up 1-2 ms earlier than the render (6-7 ms, and 3-7 across the
# other eight) -- fading, not the transmitter.
CONNECT_HEAD_MS = 8
CS_HEAD_MS = 9


def _pad(rf: np.ndarray) -> np.ndarray:
    quiet = np.zeros(round(QUIET_S * FS), np.float32)
    return np.concatenate([quiet, rf, quiet]).astype(np.float32)


def _muted(rf: np.ndarray, ms: int) -> np.ndarray:
    """The head lost with the timebase kept -- a transmitter not yet radiating.

    This and not truncation is the physical case: the modem's samples leave on
    schedule whatever the PA is doing, so the bits that follow are still where
    the far end's clock expects them.
    """
    out = rf.copy()
    out[:min(out.size, round(ms * FS / 1000))] = 0.0
    return out


def _ramped(rf: np.ndarray, ms: int) -> np.ndarray:
    """The head ATTENUATED rather than lost -- a T/R stage swinging up."""
    n = min(rf.size, round(ms * FS / 1000))
    out = rf.copy()
    if n:
        out[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32)
    return out


def _survives(render, decodes, damage, hi_ms: int) -> int:
    """Largest whole ms of `damage` every rendering still decodes through."""
    worst = hi_ms
    for rf in render:
        ok = -1
        for ms in range(hi_ms + 1):
            if not decodes(_pad(damage(rf, ms)), rf):
                break
            ok = ms
        worst = min(worst, ok)
    return worst


def _connects() -> list[np.ndarray]:
    # `_trim_silence` first, because that is what `_tx` puts on the air: the
    # renderer's 0.5 s pads never reach the rig, so the burst's own first sample
    # is the first sample keyed.
    return [onair._trim_silence(pactor1.connect_signal(call, invert=inv))
            for call in ("W9SSJ", "N5UXT", "KY4RY", "WS8EOC")
            for inv in (False, True)]


def _codewords() -> list[tuple[np.ndarray, int, bool]]:
    return [(onair._trim_silence(pactor1.control_signal(i, invert=inv)), i, inv)
            for i in range(len(pactor1.CONTROL_SIGNALS)) for inv in (False, True)]


def test_the_connect_survives_less_than_one_bit_of_lost_head() -> None:
    """The number the envelope hypothesis needed, and it is eight milliseconds."""
    def reads(audio, rf, want=None):
        got = p1rx.decode_connect(audio)
        return got is not None and got.callsign == want

    for call in ("W9SSJ", "N5UXT", "KY4RY", "WS8EOC"):
        rfs = [onair._trim_silence(pactor1.connect_signal(call, invert=inv))
               for inv in (False, True)]
        n = _survives(rfs, lambda a, rf, c=call: reads(a, rf, c), _muted, 40)
        assert n == CONNECT_HEAD_MS, f"{call}: {n} ms, not {CONNECT_HEAD_MS}"
        assert n < BIT_S * 1000


def test_the_codeword_survives_less_than_one_bit_of_lost_head() -> None:
    """Nine milliseconds guaranteed, and the distance-8 alphabet does not extend it.

    The four words sit at mutual distance 8, which buys certainty about WHICH
    word arrived and nothing about a word whose first bit never did: the reader
    insists on zero errors, and a muted head costs one.

    The eight cases run 9, 9, 9, 9, 13, 16, 19 and 21 ms, and the spread is the
    words' own leading bits rather than any margin the code carries -- a head
    that opens on a repeated bit can lose part of it and still read. Nine is the
    figure to plan with, because which codeword is due is the protocol's
    alternation and not ours to pick.
    """
    per_word = []
    for rf, idx, inv in _codewords():
        def reads(audio, _rf, idx=idx, inv=inv, rf=rf):
            cs = p1rx.decode_control_signal(audio, QUIET_S, rf.size / FS)
            return cs is not None and cs == (idx, 0, int(inv))

        per_word.append(_survives([rf], reads, _muted, 40))

    assert min(per_word) == CS_HEAD_MS, per_word
    assert min(per_word) < BIT_S * 1000
    # No word reaches two bits: a lost head is never recovered, only sometimes
    # indistinguishable from a shorter one.
    assert max(per_word) < 3 * BIT_S * 1000, per_word


def test_an_attenuated_head_is_not_a_lost_one() -> None:
    """A T/R ramp costs nothing, and the distinction is the whole diagnosis.

    Both readers decide a bit by COMPARING the two tones, so scaling the head
    scales both sides of that comparison and leaves it alone. It is why the
    witness recording's 3 ms rise is not damage, and why an envelope suspect has
    to be shown to remove the head rather than merely to shade it.
    """
    # The connect gives out at a 24 ms ramp, three times the 8 ms it allows a
    # muted head; asserted at twice, which is the claim rather than the edge.
    assert _survives(_connects(),
                     lambda a, _rf: p1rx.decode_connect(a) is not None,
                     _ramped, 2 * CONNECT_HEAD_MS) == 2 * CONNECT_HEAD_MS

    for rf, idx, inv in _codewords():
        def reads(audio, _rf, idx=idx, inv=inv, rf=rf):
            cs = p1rx.decode_control_signal(audio, QUIET_S, rf.size / FS)
            return cs is not None and cs == (idx, 0, int(inv))

        # The whole codeword ramped from zero, which is as far as the case goes.
        assert _survives([rf], reads, _ramped, 120) == 120


# -- what the settle is actually spent on ---------------------------------

CYCLE_N = round(0.30 * FS)          # a short raster: the arithmetic is scale-free
BURST_N = round(0.15 * FS)
SETTLE_S = 0.040                    # ota.RIGS["ft891"]
IN_LAT_S = 0.0103                   # what puts `holdback` at the machine's 13 ms
OUT_LAT_S = 0.012
BLK = 128


class _FakeCodec:
    """A duplex `_LiveInput` driven by a thread instead of by a sound card.

    The real `read`, `take_until`, `wait_until` and `transmit` run untouched --
    what is stubbed is the callback, which is the only part that needs a device.
    It publishes the same three things the real one does and at the same cadence:
    the ADC instant of the block's first sample, the DAC instant of the buffer
    being filled, and the block itself on the queue. Separating the two latencies
    is the point -- one term of `holdback` is the input one, and a stand-in that
    collapses them cannot show where the lead goes.
    """

    def __init__(self, in_lat: float = IN_LAT_S, out_lat: float = OUT_LAT_S):
        live = object.__new__(onair._LiveInput)
        live._q = queue.Queue()
        live.fs, live.samples, live.pos = FS, 0, 0
        live.xruns, live.xrun_at, live.underruns = 0, None, 0
        live._rest = np.zeros(0, np.float32)
        live._floor = 0
        live._duplex, live._blk = True, BLK
        live._lat = round((in_lat + out_lat) * FS)
        live.tx_latency_n = 0     # the whole loop is modelled here already
        live.holdback = BLK + round(in_lat * FS)
        live._tx, live._tx_at = None, 0
        live._tx_done = threading.Event()
        live._tx_end_time, live.keyed_s = 0.0, 0.0
        live._last = live._dac = (time.monotonic(), 0)
        self.live, self._in, self._out = live, in_lat, out_lat
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        live, t0, k = self.live, time.monotonic(), 0
        while not self._stop.is_set():
            due = t0 + k * BLK / FS
            while (d := due - time.monotonic()) > 0:
                time.sleep(min(d, 5e-4))
            now, n0 = time.monotonic(), live.samples
            dac, adc = now + self._out, now - self._in
            buf = live._tx
            if buf is not None and n0 + BLK >= live._tx_at + buf.size:
                live._tx = None
                live._tx_end_time = dac + (live._tx_at + buf.size - n0) / FS
                live._tx_done.set()
            live._q.put((adc, n0, np.zeros(BLK, np.float32)))
            live.samples = n0 + BLK
            live._last, live._dac = (adc, n0), (dac, n0)
            k += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join()


def _keyed_leads(hold_back: bool, cycles: int = 4) -> list[float]:
    """Leads in ms from `_tx`'s pre-key sequence, with and without the term."""
    codec = _FakeCodec()
    live = codec.live
    time.sleep(0.25)
    audio = np.zeros(BURST_N, np.float32)
    settle_n = round(SETTLE_S * FS)
    leads: list[float] = []
    try:
        anchor = int(live.samples) + FS // 4
        for c in range(cycles):
            boundary = anchor + c * CYCLE_N
            take_to = boundary - settle_n - (live.holdback if hold_back else 0)
            live.take_until(take_to)                    # onair._tx:725
            live.wait_until(boundary - settle_n)        # onair._tx:728
            at_key: list[float] = []

            def key(on: bool) -> bool:
                if on:
                    at_key.append(time.monotonic())
                return True

            live.transmit(audio, at=boundary, settle=SETTLE_S, key=key)
            dac_at, dac_n = live._dac
            rf_start = dac_at + (live._tx_at - dac_n) / FS
            leads.append((rf_start - at_key[0]) * 1e3)
            live.flush_to(live.pos + audio.size)
    finally:
        codec.close()
    return sorted(leads)


def test_the_gridded_burst_keys_a_holdback_short_of_its_settle() -> None:
    """`take_until` to the wait's own deadline can only ever arrive late.

    `read` blocks until the codec has DELIVERED its last sample, which is one
    block plus one input latency after that sample existed -- `holdback`, by its
    definition. The wait behind it has nothing left to sleep, so the key lands
    that much inside the settle. This is 27-29 ms of the 40 in the session logs,
    and 746 of 746 gridded bursts in the corpus sit between 26 and 34.
    """
    hold_ms = (BLK + round(IN_LAT_S * FS)) / FS * 1e3
    assert 12.0 < hold_ms < 14.0, hold_ms          # the machine's own 13 ms

    leads = _keyed_leads(hold_back=False)
    assert max(leads) < SETTLE_S * 1e3 - 0.5 * hold_ms, leads
    # Late by the holdback, and by the holdback alone: what is left of the settle
    # is bounded below by it, so the shortfall cannot be blamed on the keying
    # command. `Rig.ptt(True)` is two `os.stat`s and one non-blocking write into
    # rigctl's stdin, and it never reads a reply -- 0.04 ms median, measured
    # 2026-08-14 with a Python callback contending at 375 Hz.
    assert min(leads) > SETTLE_S * 1e3 - hold_ms - 8.0, leads


def test_the_holdback_term_is_what_restores_the_whole_settle() -> None:
    """The one term the other three call sites already carry.

    Kept as a measurement and not applied (see this file's header): the witness
    recordings put the benefit at zero, and the 13 ms it would cost comes out of
    a receive budget that has none.
    """
    assert min(_keyed_leads(hold_back=True)) > SETTLE_S * 1e3 - 3.0


def test_the_connect_burst_never_goes_through_that_path() -> None:
    """`at=None` schedules off the clock, so the first connect keeps its settle.

    This is why the hypothesis was answerable at all: the burst a station being
    called has to lock onto is the one burst that takes the whole 40 ms. All 89
    first connects in the corpus keyed 39-40; every short burst behind them is a
    retry or a packet, aimed at a boundary.
    """
    codec = _FakeCodec()
    live = codec.live
    time.sleep(0.25)
    at_key: list[float] = []

    def key(on: bool) -> bool:
        if on:
            at_key.append(time.monotonic())
        return True

    try:
        live.transmit(np.zeros(BURST_N, np.float32), at=None,
                      settle=SETTLE_S, key=key)
        dac_at, dac_n = live._dac
        rf_start = dac_at + (live._tx_at - dac_n) / FS
    finally:
        codec.close()
    assert (rf_start - at_key[0]) * 1e3 > SETTLE_S * 1e3 - 3.0


# -- intended against emitted, on the transmit side alone ------------------
#
# `working/txwitness` measured 21 of 71 keyings of 2026-08-15 putting 0.02 s on
# the air where 0.1 s was intended, with dead carrier to 94% of the keying, and
# it was read as a transmit fault. It is not one. That instrument requires BOTH
# FSK tones inside the same 20 ms step, and PACTOR-1 sends mark or space and
# never the pair, so a keying registers only in the steps whose two symbols
# happen to differ -- which shreds one real burst into a run of short ones and
# prints the fragments as separate keyings. The transmitter's own account of the
# same session says 21 keyings, every one of them 0.960 s of audio under
# 0.988-1.000 s of PTT, and the fragments coalesce back to that: runs separated
# by one 20 ms step span 0.96-1.02 s.
#
# These two are that account, taken where no detector can reach it -- the samples
# handed to the device, and the interval the key was actually held for -- and
# swept across the lengths the question was asked at, 0.02 s at one end and the
# 2.19 s ARDOP frame that measured intact at the other.

SWEEP_S = (0.02, 0.10, 0.12, 0.96, 2.19)


def test_every_burst_reaches_the_device_at_its_full_rendered_length(tmp_path) -> None:
    """Drive-scaling and the silence trim take the padding and nothing else.

    `_tx` normalises to the drive and then trims, so a burst is only ever as long
    as its own modulation -- and that trim is a threshold on AMPLITUDE rather
    than on duration, which is the one way this path could shorten a burst at
    all. Asserted against each renderer's own body length: a PACTOR-1 packet is
    0.96 s at either rate, a codeword 0.12 s.
    """
    tx = onair.RadioTx(None, transmit=False, outdir=tmp_path)
    bursts = {
        "connect": (pactor1.connect_signal("WS8EOC"), 0.96),
        "CS1": (pactor1.control_signal(pactor1.CS_ACK_A), 0.12),
        "CS4": (pactor1.control_signal(pactor1.CS_SPEED), 0.12),
        "packet 100 Bd": (pactor1.packet_signal(b"1W9SSJ\r", 100), 0.96),
        "packet 200 Bd": (pactor1.packet_signal(bytes(20), 200), 0.96),
        "break-in": (pactor1.breakin_signal(b"BK", 100), 0.96),
    }
    wrong = []
    for what, (audio, body) in bursts.items():
        tx._tx(audio, what)
        if abs(tx.last_dur - body) > 1.0 / FS:
            wrong.append((what, tx.last_dur, body))
    assert not wrong, wrong


def test_the_key_is_held_for_the_settle_and_the_whole_burst() -> None:
    """Intended against emitted, as a distribution over burst length.

    Three numbers a burst, and they separate the three faults that were proposed
    for one measurement: the sample count says whether anything was truncated,
    the key-assert to first-sample interval says whether the audio was late, and
    the keyed window says whether the key itself came down early. A fixed loss
    off the front would devastate the 0.02 s row and barely touch the 2.19 s one;
    a short keying would show in the third column at every length.
    """
    codec = _FakeCodec()
    live = codec.live
    time.sleep(0.25)
    rows = []
    try:
        for dur in SWEEP_S:
            n = round(dur * FS)
            at_key: list[float] = []

            def key(on: bool, at_key=at_key) -> bool:
                if on:
                    at_key.append(time.monotonic())
                return True

            first, end = live.transmit(np.zeros(n, np.float32), at=None,
                                       settle=SETTLE_S, key=key)
            rows.append((dur, n, end - first,
                         (live._dac_time(live._tx_at) - at_key[0]) * 1e3,
                         live.keyed_s))
    finally:
        codec.close()

    for dur, want, got, lead_ms, keyed in rows:
        assert got == want, (dur, got, want)
        assert abs(lead_ms - SETTLE_S * 1e3) < 5.0, (dur, lead_ms)
        assert abs(keyed - (SETTLE_S + dur)) < 0.05, (dur, keyed)


# -- what the rig radiated, read back off the session's own recording -------
#
# Everything above this line is either the transmit side reporting on itself or a
# KiwiSDR 103 mi away with no key to measure against. `_LiveInput` writes
# `stream.wav` from the capture callback in front of the discard floor, so a
# session holds our own emission back through the rig, on the same sample clock
# every window sidecar indexes into -- which puts an intended waveform and an
# emitted one in one file for the first time.

#: How far either side of the DAC's own schedule the emission is looked for. The
#: transmit chain's latency is what is being measured, so it cannot be assumed.
SEARCH_S = 0.060
#: A window whose burst matches the render this well was preceded by that burst.
#: The unkeyed cycles of the same session come out at 0.04-0.14.
MATCHED = 0.90
#: What the emission is read inside, because the tap hears more than the station
#: sends. The 40 m sessions carry full-wave-rectified mains at our own signal's
#: level inside the keyed span -- the residual off the render peaks at 120 Hz and
#: is 24-30% of the burst's energy, against 5-6% on 20 m and 80 m -- and a
#: correlation taken across the whole capture band reads 0.83-0.88 for bursts that
#: are 0.97 against the samples that made them. Nothing this station transmits
#: reaches below 380 Hz: PACTOR-3 SL6 is the widest at 380-2620.
TX_BAND = (300.0, 3000.0)


def _in_tx_band(a: np.ndarray) -> np.ndarray:
    """`a` with everything the station cannot have transmitted taken out of it."""
    A = np.fft.rfft(a)
    f = np.fft.rfftfreq(len(a), 1 / FS)
    A[(f < TX_BAND[0]) | (f >= TX_BAND[1])] = 0
    return np.fft.irfft(A, len(a))


def _stream(name: str = "onair-0819-2048"):
    """``(samples, [(window start, window end)])`` of a session, or a skip.

    `flush_to(tx_end)` starts every window on the first sample after our carrier
    dropped, so a window's start IS the DAC instant our burst's last sample left
    on, and `end_stream_sample` in its sidecar puts it on the recording's clock.
    That is the only anchor needed here and it comes off the files themselves.
    """
    import json
    import wave

    from hfmodem.tests import evidence

    d = evidence.CAPTURES / name
    if not (d / "stream.wav").exists():
        pytest.skip(f"no session recording under {d}")
    with wave.open(str(d / "stream.wav")) as w:
        assert w.getframerate() == FS
        a = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768.0
    a = _in_tx_band(a)
    wins = []
    for j in sorted(d.glob("*.json")):
        s = json.loads(j.read_text())
        if "end_stream_sample" in s:
            wins.append((s["end_stream_sample"] - s["samples"], s["end_stream_sample"]))
    return a, sorted(wins)


def _matched(a: np.ndarray, at: int, ref: np.ndarray) -> tuple[int, float]:
    """``(lag, normalised correlation)`` of `ref` against the recording near `at`."""
    seg = a[at - round(SEARCH_S * FS):at + len(ref) + round(SEARCH_S * FS)]
    if len(seg) < len(ref) + 2 * round(SEARCH_S * FS):
        return 0, 0.0
    c = np.correlate(seg, ref, "valid")
    k = int(np.argmax(np.abs(c)))
    energy = (ref @ ref) * (seg[k:k + len(ref)] @ seg[k:k + len(ref)])
    return k - round(SEARCH_S * FS), float(abs(c[k]) / np.sqrt(energy)) if energy else 0.0


def _ours(callsign: str = "K4MSU", name: str = "onair-0819-2048",
          ) -> list[tuple[np.ndarray, np.ndarray, int, float]]:
    """Every burst of ours the session recorded, against the render that made it.

    Both polarities are tried because `_flip` alternates them cycle to cycle, and
    a window with no burst in front of it matches neither.
    """
    a, wins = _stream(name)
    refs = [onair._trim_silence(pactor1.connect_signal(callsign, invert=inv))
            for inv in (False, True)]
    found = []
    for start, _end in wins:
        ref, lag, corr = max(((ref, *_matched(a, start - len(ref), ref)) for ref in refs),
                             key=lambda t: t[2])
        if corr >= MATCHED:
            at = start - len(ref) + lag
            found.append((a[at:at + len(ref)], ref, lag, corr))
    return found


def _per_bit(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Normalised correlation of the emission against the render, bit by bit."""
    n = round(BIT_S * FS)
    m = min(len(got), len(ref)) // n
    g, r = got[:m * n].reshape(-1, n), ref[:m * n].reshape(-1, n)
    return np.abs((g * r).sum(axis=1)) / np.sqrt((g ** 2).sum(axis=1)
                                                 * (r ** 2).sum(axis=1) + 1e-30)


def test_the_session_recorded_its_own_connect_bursts() -> None:
    """The claim the rest of this section rests on: what is in the gap between two
    receive windows is our own transmission, and it is that to a normalised 0.97
    against the very samples that made it.

    The cycles this session did NOT key are the control: they go through the same
    search against the same two renders and come out at 0.04 to 0.14, so the match
    is the burst and not the searching.
    """
    ours = _ours()
    _a, wins = _stream()
    assert 4 <= len(ours) < len(wins), (
        f"{len(ours)} of {len(wins)} cycles matched -- the search is not "
        "discriminating between a keyed cycle and a listening one")
    assert float(np.median([c for *_, c in ours])) > 0.95


def test_bit_zero_leaves_the_transmitter_intact() -> None:
    """The KiwiSDR said the leading edge is square and bit 0 is the strongest bit
    of the head. It could not say WHEN, having no key to measure against, and it
    could not rule out the transmitter swallowing whole bits before the edge it
    saw. This can, and it agrees: over eleven bursts the first twelve bits read
    0.89 to 1.00 against the render that made them, inside the spread of the body,
    where the two bits at the other end read 0.02. Nothing is lost off the front.
    """
    for got, ref, _lag, _corr in _ours():
        head = _per_bit(got, ref)[:12]
        assert head.min() > 0.85, f"the first twelve bits read {head.round(2)}"


def test_the_recording_loses_the_tail_the_air_kept() -> None:
    """WHERE OUR OWN RECORDING IS SHORT, and it is the end and not the head.

    The rig is a fixed latency behind the DAC -- 20.8 to 23.9 ms across the twenty
    sessions of 2026-08-19 and -20, which is what the lag below measures -- and it
    flushes the muted receive pipeline when PTT drops, so that much of our own
    leakage never reaches `stream.wav`. The air kept it: two off-air recordings of
    these same transmissions carry every burst to its final bit. So this is a fact
    about the tap, and the figure it fixes is how much of a burst a session can
    expect to hear of itself.

    NOTHING HERE IS READ OFF A LEVEL. The lag is the peak of a normalised
    correlation against the render that made the burst and the loss is counted in
    bits that stopped matching it, so this is the one reading of the latency that
    no threshold can move -- which is why `tools/txwitness.py` cites it against its
    own detector, whose leading edge lands on that transient on a quarter of
    these sessions and reads the latency 13 ms lower there.

    TWO BITS EXACTLY, on every burst of all twenty. That is the assertion and not
    a tolerance: the loss is quantised to whole bits, so a 22 ms flush reads as 2
    and a 9 ms one would read as 1, and the two candidate latencies are therefore
    told apart by an integer rather than by a bound wide enough to hold both. The
    lag is then required to round to the same two bits, which is an agreement
    between two independent readings rather than a bound chosen to admit them.
    """
    lags, lost = [], []
    for got, ref, lag, _corr in _ours():
        assert lag > 0, "the emission cannot precede the samples that made it"
        bits = _per_bit(got, ref)
        gone = np.flatnonzero(bits < 0.5)
        assert gone.size, "nothing of this burst was lost -- the tail is intact"
        assert gone[0] > len(bits) - 5, (
            f"the burst comes apart at bit {gone[0]} of {len(bits)}, not at its end")
        lags.append(lag / FS * 1e3)
        lost.append(len(bits) - gone[0])
    assert set(lost) == {2}, (
        f"the tail loses {sorted(set(lost))} bits where every burst of the 0819 "
        "and 0820 sessions loses exactly two -- the flush has moved, and the "
        "emission length nothing here reports has moved with it")
    assert round(float(np.median(lags)) / (BIT_S * 1e3)) == 2, (
        "the head's lag and the tail's loss are the same latency and did not "
        f"agree: {np.median(lags):.1f} ms against two bits")


def test_the_unanswered_forty_metre_calls_emitted_the_burst_they_rendered() -> None:
    """Eight calls to three gateways went unanswered on 2026-08-20, and this is the
    half of that this station controls.

    The 40 m tap carries mains hum at our own signal's level inside the keyed span:
    the settle -- PTT up, no audio in it yet -- reads +0.4 to +2.7 dB of the burst
    body at 1250-1750 Hz, where the 20 m and 80 m sessions read 15.7 to 19.8 dB
    UNDER it. Taken across the whole capture band that costs `_matched` 0.12 and
    puts every burst of all three arms under `MATCHED`, which reads as an emission
    that came apart. In the band the station actually transmits in there is nothing
    between the bands and nothing between the days: each arm's twenty bursts sit
    where the 80 m session's do and decode to the callsign they were addressed to.
    """
    for name in ("onair-0820-1037", "onair-0820-1041", "onair-0820-1048"):
        ours = _ours("WS8EOC", name)
        assert len(ours) >= 15, f"{name}: {len(ours)} bursts matched"
        assert float(np.median([c for *_, c in ours])) > 0.95, name
        for got, _ref, _lag, _corr in ours:
            dec = p1rx.decode_connect(_pad(got))
            assert dec is not None and dec.callsign == "WS8EOC", f"{name}: {dec}"
