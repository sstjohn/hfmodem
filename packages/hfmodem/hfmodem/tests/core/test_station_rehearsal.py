# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""The transmit path, end to end, against a card that keeps its own record.

Every other test of the arbiter transmits at the card rate, in card units, into
`ReplayAudio` — a stub that records what it was handed. Two faults lived inside
that blind spot at once: protocol audio was armed without resampling, so ARDOP's
12 kHz would have gone out four times fast and two octaves high, outside the
emission the regulatory gate had just approved; and nothing applied `tx_drive`,
so sabir's peak of 1.415 would have railed a card that clips at 1.0. Neither is
expressible against a stub that only remembers its argument.

So this drives the real `StationAudio`, the real `Rig` and the real `TxArbiter`
against `FakeCard`, which runs the callback on the wall clock at 375 blocks a
second and keeps every block it was handed together with the instant it was
handed one. What reached the transmitter becomes a thing that can be read back:
its rate, its level, its frequency, and where it sits against the keying line —
which is `TimedPtt`, stamping every edge. Nothing here reads the station's own
account of what it did.

Two properties of a real card matter to the timing and are modelled exactly:

  * **The ADC->DAC offset is 1152 samples**, measured on this station's hardware.
    A block the callback writes is not converted for another 24 ms, so *handed to
    the card* and *on the air* are 24 ms apart — the same order as the PTT lead
    being measured. Both are reported, and the one that decides whether a carrier
    is modulated is the second.
  * **The capture hears the transmitter.** What goes into output block `k` comes
    back in capture block `k + 1152/128`, which is what makes the half-duplex
    check a claim about audio rather than about index bookkeeping.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from hfmodem.core import cwid
from hfmodem.core.audio import StationAudio, StreamLane
from hfmodem.core.rates import CARD_RATE_HZ as FS
from hfmodem.core.regulatory import Control, Unregulated, centred
from hfmodem.core.regulatory.part97 import Part97
from hfmodem.core.rig import Cat, Rig
from hfmodem.station.arbiter import LIVE, Identity, Refused, TxArbiter, TxRequest
from hfmodem.tests.core.fakerig import FakePtt, FakeRigctld

MYCALL = "W9SSJ"
BENCH = Unregulated(because="dress rehearsal, no radio")

#: What `examples/station.toml` ships, which is what the operator runs tonight.
SETTLE_S = 0.04
DRIVE = 0.6

BLOCK = 128
#: The ADC->DAC offset measured on this station: 1152 samples, sigma zero.
OFFSET = 1152

#: What the key-up costs between the settle wait returning and the line moving:
#: one `f` round trip to rigctld, the profile check, and arming the watchdog.
#: Measured at 0.4-0.7 ms against a fake daemon over the loopback. It comes
#: straight off the lead, and a real rigctld's `f` is a serial transaction.
_KEY_COST_S = 0.006

#: How far past the last converted sample the key may still be up. Measured here
#: at 0.1-0.6 ms, which is one sample of waiting plus one ioctl; 10 ms is
#: generous and far under the five seconds of 2026-07-28.
_TAIL_S = 0.010


class _Times:
    def __init__(self, adc, dac):
        self.inputBufferAdcTime = adc
        self.outputBufferDacTime = dac


class _Quiet:
    """No xrun. The card's fault paths are `test_audio`'s subject, not this one's."""

    input_overflow = output_underflow = priming_output = False

    def __bool__(self):
        return False


class FakeCard:
    """The PortAudio surface, in real time, on the record.

    It is both the module `StationAudio` imports and the one stream that module
    hands out, because there is one of each and a factory between them would be
    ceremony.

    Deliberately not a way to make a transmission look good. It composes nothing:
    it hands the callback a bed plus whatever the transmitter put on the air 1152
    samples ago, keeps every block in both directions with the instant the
    callback ran, and answers questions about those blocks and nothing else. A
    claim about what the station transmitted is settled against this record.
    """

    def __init__(self, *, bed_hz: float = 1500.0, bed_amp: float = 0.1,
                 offset: int = OFFSET, blocksize: int = BLOCK) -> None:
        if offset % blocksize:
            raise ValueError("the loopback delay must be a whole number of blocks")
        self.offset, self.blocksize = offset, blocksize
        self.bed_hz, self.bed_amp = bed_hz, bed_amp
        self.latency = (offset / 2 / FS, offset / 2 / FS)
        self.active = False
        self.t0 = 0.0
        self.out: list[np.ndarray] = []
        self.inp: list[np.ndarray] = []
        self._cb = None
        self._run = threading.Event()
        self._thread: threading.Thread | None = None

    # -- the surface `StationAudio.open()` uses ----------------------------

    def Stream(self, *, callback, blocksize=BLOCK, **kw):
        self.blocksize = blocksize
        self._cb = callback
        return self

    def start(self) -> None:
        self.active = True
        self._run.set()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.active = False
        self._run.clear()

    def close(self) -> None:
        self._run.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- the clock ---------------------------------------------------------

    def _pump(self) -> None:
        """One block every 2.67 ms, timestamped on the grid rather than on arrival.

        A converter's timestamps step by exactly one block whatever the scheduler
        did with the thread that carries them, and that regularity is the whole
        of what `sample_now()` extrapolates from. Timestamping at wakeup instead
        would put this thread's jitter into the station's clock and then measure
        it back out as the station's.
        """
        b = self.blocksize
        period = b / FS
        lag = self.offset // b
        self.t0 = t0 = time.monotonic()
        n = 0
        while self._run.is_set():
            due = t0 + n * period
            now = time.monotonic()
            if due > now:
                time.sleep(due - now)
            indata = self._capture(n * b, b, lag)
            outdata = np.zeros((b, 1), np.float32)
            self._cb(indata, outdata, b, _Times(due, due + self.offset / FS), _Quiet())
            self.inp.append(indata[:, 0])
            self.out.append(outdata[:, 0].copy())
            n += 1

    def _capture(self, n0: int, n: int, lag: int) -> np.ndarray:
        i = np.arange(n0, n0 + n)
        x = self.bed_amp * np.sin(2 * np.pi * self.bed_hz * i / FS)
        k = len(self.out) - lag
        if k >= 0:
            x = x + self.out[k]
        return x.reshape(-1, 1).astype(np.float32)

    # -- what the transmitter carried --------------------------------------

    @property
    def written(self) -> np.ndarray:
        return np.concatenate(self.out) if self.out else np.zeros(0, np.float32)

    @property
    def heard(self) -> np.ndarray:
        return np.concatenate(self.inp) if self.inp else np.zeros(0, np.float32)

    def carried(self) -> tuple[int, np.ndarray]:
        """The span of non-silence on the output, and the sample it starts at."""
        w = self.written
        nz = np.flatnonzero(w)
        if not len(nz):
            return 0, np.zeros(0, np.float32)
        return int(nz[0]), w[nz[0]:nz[-1] + 1]

    def t_write(self, i: int) -> float:
        """When the card was handed output sample `i`."""
        return self.t0 + (i // self.blocksize) * self.blocksize / FS

    def t_air(self, i: int) -> float:
        """When output sample `i` reaches the converter, 1152 samples later."""
        return self.t0 + (i + self.offset) / FS


class TimedPtt(FakePtt):
    """The keying line, with the instant of every edge that actually took.

    Stamped after the base class has accepted the call, so a refused assert
    leaves no edge — a line that did not move must not read as one that did.
    """

    def __init__(self) -> None:
        super().__init__()
        self.edges: list[tuple[float, bool]] = []

    def assert_(self, on: bool) -> None:
        super().assert_(on)
        self.edges.append((time.monotonic(), on))

    @property
    def up(self) -> float:
        return self.edges[0][0]

    @property
    def down(self) -> float:
        return self.edges[-1][0]


class Bench:
    """One station: the card, the radio, and the arbiter that owns both."""

    def __init__(self, card, audio, rig, ptt, arb, daemon):
        self.card, self.audio, self.rig = card, audio, rig
        self.ptt, self.arb, self.daemon = ptt, arb, daemon

    def request(self, audio, **kw) -> TxRequest:
        kw.setdefault("priority", LIVE)
        kw.setdefault("seq", 1)
        kw.setdefault("protocol", "vara")
        kw.setdefault("rate", FS)
        kw.setdefault("settle_s", SETTLE_S)
        kw.setdefault("emission", centred(7_100_000, 500.0))
        return TxRequest(audio=np.asarray(audio), **kw)


@pytest.fixture
def bench(monkeypatch):
    """The real transmit path, opened the way `station/process.py` opens it."""
    built: list[Bench] = []

    def build(*, profile=None, bed_hz: float = 1500.0) -> Bench:
        card = FakeCard(bed_hz=bed_hz)
        monkeypatch.setitem(sys.modules, "sounddevice", card)
        sa = StationAudio(input_device="fake-in", output_device="fake-out",
                          blocksize=BLOCK)
        sa.open()
        daemon = FakeRigctld(freq=7_100_000)
        ptt = TimedPtt()
        rig = Rig(model="ft891", cat=Cat("127.0.0.1", daemon.port), ptt=ptt,
                  profile=profile or BENCH, control=Control.LOCAL, mycall=MYCALL,
                  transmit=True)
        rig.cat.open()
        rig._armed = True
        rig._dial_hz = daemon.freq
        arb = TxArbiter(sa, rig, identity=Identity(MYCALL, interval_s=600.0),
                        drive=DRIVE, settle_s=SETTLE_S)
        b = Bench(card, sa, rig, ptt, arb, daemon)
        built.append(b)
        return b

    yield build
    for b in built:
        b.rig.close()
        b.audio.close()
        b.daemon.close()


def tone(hz: float, secs: float, rate: int = FS, amp: float = 1.0) -> np.ndarray:
    """A cosine, so the first sample is the peak and the start of the emission is
    unambiguous in a record made of exact zeros and everything else."""
    return amp * np.cos(2 * np.pi * hz * np.arange(int(secs * rate)) / rate)


def peak_hz(x: np.ndarray, rate: int = FS) -> float:
    spec = np.abs(np.fft.rfft(np.asarray(x, float)))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[int(np.argmax(spec))])


def amplitude_at(x: np.ndarray, hz: float, rate: int = FS) -> float:
    """How much of `x` is at `hz`, in the units its samples are in."""
    if not len(x):
        return 0.0
    i = np.arange(len(x))
    return 2 * abs(np.sum(np.asarray(x, float)
                          * np.exp(-2j * np.pi * hz * i / rate))) / len(x)


# -- the keying line against the audio -----------------------------------------

@pytest.mark.parametrize("ahead,lead_lo,lead_hi", [
    pytest.param(FS // 2, SETTLE_S - _KEY_COST_S, SETTLE_S + 0.002, id="scheduled"),
    pytest.param(None, SETTLE_S - _KEY_COST_S, SETTLE_S + 0.002,
                 id="as-soon-as-possible"),
])
def test_the_key_brackets_the_burst_at_both_edges(bench, ahead, lead_lo, lead_hi):
    """Both edges, against the transport, for the two shapes a burst arrives in.

    A burst that names the card index it must land on — a cycle-locked protocol,
    or anything answering inside a turnaround window — gets the settle the
    operator configured, because the arbiter waits out the difference before it
    keys. Measured: the key was up 39.3-39.6 ms before the first sample reached
    the converter, against a configured 40.

    A burst that names no index is the shape all four adapters submit today, and
    it used to get a different answer: `wait_until(at - settle_n)` was already in
    the past, nothing waited, and the lead was whatever `arm_burst`'s minimum
    notice left after `key()` returned — measured 31.7-33.6 ms, from a converter
    offset of 1152 samples plus three blocks of notice. That landed near the
    configured settle by arithmetic coincidence rather than by design, and would
    have moved with the offset or the blocksize. The arbiter now asks for the
    settle explicitly when no instant is named, so both shapes get the number the
    operator configured and the same bounds apply to each.

    Which is why the lower bound that matters is neither of those numbers but the
    sign: **the key must be up before the first sample is handed to the card**,
    and a lead measured at the converter is 24 ms more generous than one measured
    at the write. The ceiling is the other half — a keyed transmitter carrying
    nothing is somebody else's channel — and the tail is the incident of
    2026-07-28, where the operator watched five seconds of dead carrier after a
    burst had ended. Nothing on this path produces one: the unkey lands 0.1-0.6 ms
    past the last converted sample.
    """
    b = bench()
    burst = tone(1000.0, 1.0)
    at = None if ahead is None else int(b.audio.sample_now()) + ahead
    end = b.arb.submit(b.request(burst, at=at))
    time.sleep(0.05)

    i0, carried = b.card.carried()
    assert len(carried) == len(burst), (
        f"{len(burst) / FS:.3f} s composed and {len(carried) / FS:.3f} s reached "
        "the card")
    assert end == i0 + OFFSET + len(carried), (
        "the index the arbiter reported is not where the audio ended on the card")
    assert [on for _, on in b.ptt.edges] == [True, False], (
        f"the keying line moved {len(b.ptt.edges)} time(s) for one burst")

    lead_write = b.card.t_write(i0) - b.ptt.up
    lead_air = b.card.t_air(i0) - b.ptt.up
    assert lead_write > 0, (
        f"the first sample was handed to the card {-lead_write * 1e3:.1f} ms "
        "before the key went up — the transmitter is being asked to carry audio "
        "that is already in the driver's buffer")
    assert lead_lo < lead_air <= lead_hi, (
        f"{lead_air * 1e3:.1f} ms of unmodulated carrier before the burst, "
        f"outside {lead_lo * 1e3:.1f}-{lead_hi * 1e3:.1f} ms")

    tail_air = b.ptt.down - b.card.t_air(i0 + len(carried) - 1)
    assert tail_air >= 0.0, (
        f"the key came down {-tail_air * 1e3:.1f} ms before the last sample was "
        "converted, so the end of the burst never left the antenna")
    assert tail_air <= _TAIL_S, (
        f"the key was held {tail_air * 1e3:.1f} ms past the last converted "
        f"sample, more than {_TAIL_S * 1e3:.0f} ms — an unmodulated transmitter "
        "on a shared band")

    hold = b.ptt.down - b.ptt.up
    assert hold <= SETTLE_S + len(burst) / FS + _TAIL_S, (
        f"the transmitter was up {hold:.3f} s for {len(burst) / FS:.3f} s of "
        f"audio and {SETTLE_S:.3f} s of settle — {hold - len(burst) / FS:.3f} s "
        "of it unexplained")


# -- the identification --------------------------------------------------------

def test_the_identification_reaches_the_card_as_audio(bench):
    """A callsign nobody transmitted looks exactly like the incident it explains.

    §97.119 is discharged by RF, not by a line in the log, and an identification
    whose audio never reached the card would present as several seconds of keyed
    transmitter carrying nothing — which is what the operator reported on
    2026-07-28 and what nothing on the bench has yet accounted for. So the
    callsign is read back off the transport: 3.660 s of it, at the drive the
    operator set, on `cwid.TONE_HZ`, with the key up for its duration and no
    longer.
    """
    b = bench()
    b.arb.identify(centred(7_100_000, cwid.BANDWIDTH_HZ))
    time.sleep(0.05)

    i0, carried = b.card.carried()
    want = cwid.duration(MYCALL)
    assert len(carried) / FS == pytest.approx(want, abs=0.002), (
        f"{want:.3f} s of Morse composed, {len(carried) / FS:.3f} s reached the card")
    assert float(np.abs(carried).max()) == pytest.approx(DRIVE, abs=0.001), (
        f"the callsign went out at {float(np.abs(carried).max()):.3f}, not the "
        f"{DRIVE} drive every other burst leaves at")
    assert peak_hz(carried) == pytest.approx(cwid.TONE_HZ, abs=5.0), (
        f"the identification carried {peak_hz(carried):.0f} Hz, not "
        f"{cwid.TONE_HZ:.0f}")
    # Keyed, not merely non-zero: a continuous tone of the right length and level
    # is a dead carrier that would pass every check above.
    envelope = np.abs(carried) > DRIVE / 4
    assert 0.35 < envelope.mean() < 0.65, (
        f"the transmitter carried an unkeyed tone for {envelope.mean() * 100:.0f}% "
        "of the identification — that is a carrier, not a callsign")

    hold = b.ptt.down - b.ptt.up
    assert hold - want <= _TAIL_S + SETTLE_S, (
        f"the key was up {hold:.3f} s for {want:.3f} s of Morse — "
        f"{hold - want:.3f} s of it carrying nothing")


def test_the_identification_is_keyed_before_its_first_element(bench):
    """A callsign missing its first element is not a shortened identification.

    `identify` used to build its own `TxRequest` and take `settle_s`'s default of
    zero — the only transmission here with no protocol behind it to remember —
    so the arbiter waited for the burst's own start index and keyed *after* it.
    Measured then: the key went up 0.2-0.8 ms after the callsign's first sample
    had been converted, and 24 ms after it was handed to the card.

    On the bench that costs part of a 5 ms raised-cosine rise. At the radio it
    costs the `f` round trip inside `key()` plus the rig's own T/R switching, out
    of a first element 60 ms long at 20 WPM. W9SSJ opens with a dit; lose it and
    the W is an M, and the station has identified as somebody else.

    The settle is a station property now rather than a per-request one, so the
    only caller that could forget it no longer can.
    """
    b = bench()
    b.arb.identify(centred(7_100_000, cwid.BANDWIDTH_HZ))
    time.sleep(0.05)

    i0, _ = b.card.carried()
    lead_air = b.card.t_air(i0) - b.ptt.up
    assert lead_air > 0, (
        f"the callsign began reaching the converter {-lead_air * 1e3:.1f} ms "
        "before the key went up, so its leading edge was never transmitted")


# -- rate and level, at the transport ------------------------------------------

def test_a_protocol_at_its_own_rate_reaches_the_card_at_the_cards(bench):
    """ARDOP works at 12 kHz in int16 and the card plays 48 kHz float in [-1, 1].

    Armed as composed, a second of it goes out in a quarter of one with every
    tone two octaves high — 1500 Hz landing at 6000, outside the passband the
    emission gate approved a moment earlier. The gate cannot see it: it is handed
    the waveform's declared band, never the samples. `test_arbiter` pins the same
    property one layer up, against a stub that is handed the converted audio; this
    reads it off the card, which is the only place the emission is real.

    The level rides with it. sabir composes a peak of 1.415 into a card that rails
    at 1.0, and the four protocols disagree about level by more than a factor of
    two, so an operator who set the rig's input gain against one would be
    overdriving on the next.
    """
    b = bench()
    burst = (tone(1500.0, 1.0, rate=12000) * 27000).astype(np.int16)
    b.arb.submit(b.request(burst, protocol="ardop", rate=12000))
    time.sleep(0.05)

    _, carried = b.card.carried()
    assert len(carried) / FS == pytest.approx(len(burst) / 12000, abs=0.005), (
        f"{len(burst) / 12000:.3f} s composed at 12 kHz, {len(carried) / FS:.3f} s "
        "on a card running at 48 kHz")
    assert peak_hz(carried) == pytest.approx(1500.0, abs=10.0), (
        f"1500 Hz composed, {peak_hz(carried):.0f} Hz reached the transmitter")
    assert float(np.abs(carried).max()) == pytest.approx(DRIVE, abs=0.005), (
        f"the card was handed a peak of {float(np.abs(carried).max()):.3f} against "
        f"a configured drive of {DRIVE}")


# -- half duplex ---------------------------------------------------------------

def test_the_lanes_are_flushed_past_the_stations_own_carrier(bench):
    """A receiver hears its own transmitter, and four decoders must not.

    The card loops the output back into the capture 1152 samples later, which is
    what a radio does to a station listening through its own front end — so
    during the burst the capture really does carry our 1000 Hz tone, and the check
    below is about audio rather than about an index. `finish_burst` advances every
    lane to the sample the emission ended on; if that index is wrong by so much as
    the ADC->DAC offset, our own carrier arrives at the demodulators, and a
    decoder that finds a frame in it has found one of ours.
    """
    b = bench(bed_hz=1500.0)
    lane = b.audio.subscribe(StreamLane(FS))
    burst = tone(1000.0, 1.0)
    end = b.arb.submit(b.request(burst))
    time.sleep(0.25)

    # Where the emission really sat, read off the card rather than off the index
    # the arbiter reported — that index is one of the things under test.
    i0, carried = b.card.carried()
    at, over = i0 + OFFSET, i0 + OFFSET + len(carried)
    assert amplitude_at(b.card.heard[at:over], 1000.0) > DRIVE / 2, (
        f"the capture carried {amplitude_at(b.card.heard[at:over], 1000.0):.3f} of "
        "our own carrier during the burst, so there was nothing here for the flush "
        "to prevent")
    assert end == over, (
        f"the arbiter reported the emission ending at {end} and it ended at {over} — "
        f"the lanes were advanced to the wrong sample by {(end - over) / FS:+.3f} s")

    got = lane.poll()
    assert got, "the lane delivered nothing at all, so it proves nothing about a flush"
    first, audio = got[0]
    assert amplitude_at(audio, 1000.0) < 0.01, (
        f"{amplitude_at(audio, 1000.0):.3f} of our own 1000 Hz carrier reached a "
        "demodulator")
    assert first >= over, (
        f"the lane handed out sample {first}, which was captured "
        f"{(over - first) / FS:.3f} s inside our own transmission")
    assert amplitude_at(audio, b.card.bed_hz) > 0.05, (
        "the flush took the channel with it — the lane is not hearing the band")


# -- a refusal ------------------------------------------------------------------

def test_a_refused_burst_never_reaches_the_keying_line(bench):
    """The gate is only a gate if the transmitter stays cold behind it.

    A 4 kHz emission is refused by §97.307(f)(3) inside `Rig.key`, after the
    arbiter has already armed the audio — so the refusal has to reach the card as
    well as the caller. Nothing may be keyed and nothing may be converted.
    """
    b = bench(profile=Part97(licence="general"))
    with pytest.raises(Refused, match="97.307"):
        b.arb.submit(b.request(tone(1000.0, 1.0),
                               emission=centred(7_100_000, 4000.0)))
    time.sleep(0.05)

    assert b.ptt.edges == [], f"a refused burst keyed the transmitter: {b.ptt.edges}"
    assert b.ptt.line is False
    written = float(np.abs(b.card.written).max())
    assert written == 0.0, (
        f"the card was handed audio peaking at {written:.3f} for a burst the "
        "regulatory gate had refused")
