# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""A station calls us and we answer it, both ends in audio.

The whole `onair --answer` session runs on the bench clock from `test_grid`,
keying through the production emission path. The far end is a caller built
the way the reference master is. Its calls and packets go out on its own
free-running 1.25 s (or 1.4 s) grid, off our tone pair by `hz`, on a clock
`ppm` from ours. It searches for the first control signal across hf-pactor's
0-138.75 ms window past its packet at zero bit errors and latches the middle
of the run of positions that read it; hf-pactor takes the first, which puts
every later read of any responder ~4 ms early, at the edge of the cliff
`test_ackplace` measures. From then on it reads twelve bits there with no
search at all (`test_ackplace._receive_cs`), or our packet from there once the
link turns, and trims that instant by an eighth of what each read measures,
as the reference master does. Its link layer is a second `PtcHost`. It hears
only what our transmitter emitted, in noise, and demodulates it at its own
tones.

Run:  pytest packages/hfmodem/hfmodem/tests/shrike/test_answer_loopback.py
"""
from __future__ import annotations

import contextlib
import io
import sys

import numpy as np
import pytest

from hfmodem.shrike import onair, p1rx, p3acquire, pactor1, ptc, rxfront, spec
from hfmodem.shrike.arq import ISS, State
from hfmodem.tests.shrike import test_grid as bench
from hfmodem.tests.shrike.test_ackplace import _receive_cs

FS = onair.FS
MYCALL, CALLER = "W9SSJ", "K7ABC"
PACKET_N = round(spec.P1_PACKET_S * FS)
CS_N = round(spec.P1_CS_S * FS)
ROT_N = round((spec.P1_PACKET_S - spec.P1_CS_S) * FS)
SEARCH_N = round(0.13875 * FS)
STEP_N = round(0.00125 * FS)
LEAD_N = round(0.020 * FS)
TOL_N = round(0.005 * FS)
PAD_N = round(0.050 * FS)
BIT_N = FS // 100
ASKED = b"1k7abc\rGOOD MORNING FROM THE CALLER\r"
SAID = "GOOD EVENING FROM THE CALLED STATION"


def _words():
    words = {}
    for i in range(len(pactor1.CONTROL_SIGNALS)):
        for inv in (False, True):
            burst = pactor1.control_signal(i, invert=inv)
            words[_receive_cs(np.pad(burst, (0, 600)), 0)] = (i, inv)
    return words


WORDS = _words()


def _read(seg: np.ndarray, at: int, errors: int = 1, only=None):
    """(index, inverted) of the codeword twelve bits from `at`, or None."""
    got = _receive_cs(seg, at)
    for bits, word in WORDS.items():
        if bin(got ^ bits).count("1") <= errors and (only is None or word[0] in only):
            return word
    return None


class _Caller:
    """The far station: a reference master's clock around our own link layer."""

    def __init__(self, air: "_Air", *, variant: str, hz: float, t0_s: float,
                 noise: float, rng, ppm: float = 0.0):
        self.air, self.hz, self.variant = air, hz, variant
        self.slot_n = (spec.CYCLE_SHORT_LONGPATH_S if variant == "longpath"
                       else spec.CYCLE_SHORT_S) * FS * (1 + ppm * 1e-6)
        self.ear = rng.normal(0, noise, air.audio.size).astype(np.float32)
        self.tx0 = round(t0_s * FS)
        self.rx0: float | None = None
        self.k = 0
        self.iss = True
        self.head = False
        self.queue: list[tuple[str, np.ndarray]] = []
        self.sent: list[tuple[int, int, str]] = []
        self.reads: list[tuple[int, int, int, object]] = []
        self.latched: tuple[int, int] | None = None
        self.queued = False
        self.host = ptc.PtcHost(peer=self, mycall=CALLER)
        self.host.stay_in_pactor1 = True
        prefix = {"normal": "", "longpath": "!", "robust": "%"}[variant]
        self.host.arq.on_host_connect(CALLER, prefix + MYCALL)

    # -- the seam PtcHost transmits through ---------------------------------
    def attach(self, host) -> None:
        pass

    def pump(self) -> None:
        pass

    cycle = pump

    def _sense(self) -> bool:
        return bool(self.k & 1)

    def connect_burst(self, mycall, dxcall):
        self.queue.append(("CALL", pactor1.connect_signal(
            dxcall, invert=self._sense(), variant=self.host.arq.connect_variant)))

    def send_p1_packet(self, payload, baud, packet_count, *, header=None,
                       changeover_request=False, qrt=False):
        self.queue.append((f"PKT#{packet_count}", pactor1.packet_signal(
            payload, baud, packet_count=packet_count, header=header,
            changeover_request=changeover_request, qrt=qrt,
            invert=self._sense())))

    def send_p1_breakin(self, payload, baud, packet_count, *, qrt=False):
        self.queue.append((f"BK#{packet_count}", pactor1.breakin_signal(
            payload, baud, packet_count=packet_count, qrt=qrt,
            invert=self._sense())))

    def send_p1_cs(self, index):
        self.queue.append((f"CS{index + 1}", pactor1.control_signal(
            index, invert=self._sense())))

    # -- its clock ------------------------------------------------------------
    def tx(self, k: int) -> int:
        return self.tx0 + round(k * self.slot_n)

    def rx(self, k: int) -> int:
        return round(self.rx0 + k * self.slot_n)

    def _follow(self, err: float) -> None:
        """The reference master's receive anchor follows what it reads, an
        eighth of the error a cycle, never more than a bit away."""
        self.rx0 += max(-BIT_N, min(BIT_N, err)) / 8

    def advance(self, now: int) -> None:
        while self.tx(self.k) - LEAD_N <= now and self.k < 400:
            self._turn()

    def hear_us(self, first: int, audio: np.ndarray) -> None:
        burst = audio * (0.11 / (np.abs(audio).max() or 1.0))
        end = min(first + burst.size, self.ear.size)
        self.ear[first:end] += burst[:end - first]

    def _mine(self, at: int, n: int) -> np.ndarray:
        """`n` samples of what it hears from `at`, brought to its own tones, with
        `PAD_N` either side so the analytic signal's wrap stays off them."""
        seg = self.ear[at - PAD_N:at + n + PAD_N].astype(float)
        return p3acquire.compensate(seg, self.hz)

    def _turn(self) -> None:
        k, arq = self.k, self.host.arq
        live = arq.state in (State.CONNECTING, State.CONNECTED, State.DISCONNECTING)
        if k and live:
            if self.iss:
                if self._hear_cs(k - 1):
                    return
            else:
                self._hear_packet(k - 1)
        if k:
            self.host.tick()
        if arq.state == State.CONNECTED and not self.queued:
            self.queued = True
            arq.on_host_data(ASKED)
            arq.on_host_over()
        self._key(k)
        if arq.role == ISS and not self.iss:
            self.iss, self.rx0 = True, self.rx0 + ROT_N
        self.k += 1

    def _hear_cs(self, j: int) -> bool:
        if self.rx0 is None:
            lo = self.tx(j) + PACKET_N
            seg = self._mine(lo, SEARCH_N + CS_N)
            hits = [(off, word) for off in range(0, SEARCH_N + 1, STEP_N)
                    for word in [_read(seg, PAD_N + off, 0,
                                       (pactor1.CS_ACK_A, pactor1.CS_SPEED))]
                    if word is not None]
            if not hits:
                return False
            run = [off for off, w in hits if w == hits[0][1]
                   and off - hits[0][0] <= 8 * STEP_N]
            off, word = (run[0] + run[-1]) // 2, hits[0][1]
            self.rx0 = float(lo + off - round(j * self.slot_n))
            self.latched = (lo, off)
        else:
            seg = self._mine(self.rx(j), CS_N + BIT_N)
            word = _read(seg, PAD_N)
            if word is not None:
                run = [off for off in range(-BIT_N, BIT_N + 1, STEP_N // 4)
                       if _read(seg, PAD_N + off, 0, (word[0],)) == word]
                if run:
                    self._follow((run[0] + run[-1]) / 2)
        self.reads.append((j, self.rx(j), self.k, word))
        if word is None:
            return False
        index, inv = word
        if index == pactor1.CS_CHANGEOVER:
            self.iss, self.head = False, True
            self.tx0 += ROT_N
            return True
        self.host.on_rx_event(rxfront.Event(
            self.rx(j) / FS, "cs", f"CS{index + 1}", protocol="PACTOR-1",
            cs=index, sense=inv))
        return False

    def _hear_packet(self, j: int) -> None:
        at = self.rx(j)
        seg = self._mine(at, PACKET_N)
        found = p1rx.decode_p1_packets(seg, breakin=self.head)
        self.head = False
        found = [p for p in found if abs(p.start - PAD_N) <= TOL_N]
        self.reads.append((j, at, self.k, found[0] if found else None))
        for p in found[:1]:
            self._follow(p.start - PAD_N)
            self.host.on_rx_event(rxfront._p1_packet_event(p, p.start))

    def _key(self, k: int) -> None:
        queue, self.queue = self.queue, []
        if not queue:
            return
        what, audio = queue[0]
        burst = onair._trim_silence(np.asarray(audio, float))
        burst = p3acquire.compensate(burst, -self.hz).astype(np.float32)
        at = self.tx(k)
        end = min(at + burst.size, self.air.audio.size)
        self.air.audio[at:end] += burst[:end - at]
        self.sent.append((at, end, what))


class _Air(bench._Bench):
    """The bench stream with a caller on the far end of it."""

    caller: _Caller | None = None
    _keying = False

    def _advance(self, to: int) -> None:
        super()._advance(to)
        if self.caller is not None and not self._keying:
            self.caller.advance(self.now)

    def transmit(self, audio, **kw):
        self._keying = True
        try:
            first, end = super().transmit(audio, **kw)
        finally:
            self._keying = False
        self.ours.append((first, np.asarray(audio, np.float32)))
        self.caller.hear_us(first, audio)
        self.caller.advance(self.now)
        return first, end


def _tones(burst: np.ndarray, hz: float) -> tuple[float, float]:
    """The strongest line either side of the pair's centre, to 1/16 Hz."""
    power = np.abs(np.fft.rfft(burst, 16 * FS))
    f = np.fft.rfftfreq(16 * FS, 1 / FS)
    centre = 1500 + hz
    return (f[np.argmax(np.where(f < centre, power, 0))],
            f[np.argmax(np.where(f > centre, power, 0))])


def _fly(monkeypatch, tmp_path, *, variant: str, hz: float, noise: float = 0.05,
         seed: int = 1, ppm: float = 0.0):
    rng = np.random.default_rng(seed)
    made: dict = {}

    def live(*a, **kw):
        air = _Air(seconds=120.0)
        air.audio += rng.normal(0, noise, air.audio.size).astype(np.float32)
        air.ours = []
        air.caller = _Caller(air, variant=variant, hz=hz, t0_s=3.1,
                             noise=noise, rng=rng, ppm=ppm)
        made["air"] = air
        return air

    class Host(ptc.PtcHost):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made["host"] = self

    monkeypatch.setattr(onair, "_LiveInput", live)
    monkeypatch.setattr(onair, "PtcHost", Host)
    monkeypatch.setattr(onair, "find_device", lambda *a, **kw: 0)
    monkeypatch.setattr(onair.ota, "Rig", bench._Rig)
    monkeypatch.setattr(onair, "_save_capture_async", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", [
        "shrike.onair", "--transmit", "--answer", "20", "--mycall", MYCALL,
        "--dial", "7100000", "--serial", "/dev/null", "--outdir", str(tmp_path / "capture"),
        "--hold", "8", "--message", SAID])
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        onair.main()
    return made["air"], made["host"], log.getvalue()


# The quiet case is the one where our own onset reading is sharpest, and it
# was the one a peer-measured break-in placement keyed 4 ms off the read. The
# last caller's sound card runs 150 ppm slow of ours, 6 ms over the session.
CASES = [("normal", 120.0, 0.05, 0.0), ("longpath", -180.0, 0.05, 0.0),
         ("robust", 60.0, 0.05, 0.0), ("normal", -240.0, 0.003, 0.0),
         ("normal", 90.0, 0.05, 150.0)]


@pytest.mark.parametrize("variant,hz,noise,ppm", CASES)
def test_an_answered_call_carries_data_both_ways(monkeypatch, tmp_path,
                                                 variant, hz, noise, ppm):
    air, host, log = _fly(monkeypatch, tmp_path, variant=variant, hz=hz,
                          noise=noise, ppm=ppm)
    caller = air.caller
    assert "CALLED:" in log, log[-3000:]
    assert "NOT KEYING" not in log, [l for l in log.splitlines()
                                     if "NOT KEYING" in l][:3]

    # The answer is keyed ANSWER_D_S past the caller's packet, and the caller's
    # own search finds it there.
    assert caller.latched is not None, log[-3000:]
    end, off = caller.latched
    first, burst = air.ours[0]
    assert abs(first - end - round(onair.ANSWER_D_S * FS)) <= round(0.002 * FS)
    assert abs(first - end - off) <= round(0.002 * FS)

    # ...on the caller's tones: the call search's estimate, and every burst of
    # ours moved there. Read off the first packet-length one, a break-in.
    est = host.called_by.offset_hz
    assert abs(est - hz) < 1, est
    packet = next(a for _, a in air.ours if a.size > 0.9 * FS)
    lo, hi = _tones(packet, est)
    assert abs(lo - pactor1.MARK - est) < 1 and abs(hi - pactor1.SPACE - est) < 1

    # Every later burst of ours starts inside the caller's read anchor, where it
    # reads with no search at all, codewords and packets alike.
    starts = np.array([f for f, _ in air.ours])
    reads = [at for _, at, _, word in caller.reads if word is not None]
    assert len(reads) > 10
    for at in reads:
        gap = starts[np.argmin(np.abs(starts - at))] - at
        assert abs(gap) < round(p1rx.CS_ANCHOR_S * FS), (at, gap)

    # The link came up and carried data in both directions.
    got = bytes(caller.host.channel(caller.host.ptchn).rx)
    assert SAID.encode() in got, (got, log[-4000:])
    assert host.arq.dxcall == CALLER
    assert b"GOOD MORNING FROM THE CALLER" in bytes(host.channel(host.ptchn).rx)
    assert host.arq.goodbye_acked, log[-3000:]
