# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from hfmodem.shrike import onair
from hfmodem.shrike.arq import ISS
from hfmodem.shrike.ptc import PtcHost
from hfmodem.shrike.spec import Protocol


FS = onair.FS
_LIVE = onair._LiveInput
IN_LAT_S, OUT_LAT_S, BLK = .0103, .012, 128


class _Seam:
    """A transmit seam that records instead of keying."""

    def __init__(self) -> None:
        self.sent: list = []

    def attach(self, host) -> None:
        pass

    def connect_burst(self, mycall, dxcall) -> None:
        self.sent.append(("connect", dxcall))

    def p3_slot_keyed(self) -> bool | None:
        slot = getattr(self, "slot", None)
        keyed = getattr(self, "slots_used", None)
        if isinstance(slot, int) and isinstance(keyed, list):
            return bool(keyed and keyed[-1] >= slot)
        return None

    def send_cs(self, i, *, p1_codeword=False, unread=False, maxdown=False,
                packet_phase=None, cycle_long=None) -> None:
        self.sent.append(("cs", i))

    def send_p1_cs(self, i) -> None:
        self.sent.append(("p1cs", i))

    def send_p1_packet(self, payload, baud, packet_count, **kw) -> None:
        self.sent.append(("p1pkt", payload))

    def send_packet(self, sl, payload, status, breakin=False) -> None:
        self.sent.append(("pkt", sl, payload))

    def send_long_packet(self, sl, payload, status, **kw) -> None:
        self.sent.append(("long", sl, payload))

    def deliver(self, data) -> None:
        pass

    def buffer(self, n) -> None:
        pass

    def pump(self) -> None:
        pass

    def cycle(self) -> None:
        pass


class _Session:
    """A `_SessionRx` on a linked PACTOR-3 host that keeps what it delivered."""

    def __init__(self, *, role=ISS, entry_pending: bool = False) -> None:
        host = PtcHost(peer=_Seam(), mycall="W9SSJ")
        host.arq.on_host_listen(True)
        host.arq.role, host.arq.dxcall = role, "DL6MAA"
        host.arq._enter_connected()
        host.protocol = Protocol.PACTOR3
        host.arq.entry_pending = entry_pending
        host.peer.sent.clear()
        self.host = host
        self.rx = onair._SessionRx(host, tag="TEST")
        self.events: list = []
        deliver = host.on_rx_event

        def record(ev):
            self.events.append(ev)
            return deliver(ev)

        host.on_rx_event = record

    def changeover(self, audio: np.ndarray, head_at: int):
        """The cycle a CS3 is due in, read where the grid says it is."""
        self.rx.new_cycle()
        return self.rx.control_signal(audio, 0, head_at)

    @property
    def packets(self) -> list:
        return [ev for ev in self.events if ev.kind == "packet"]


class _Bench:
    """Charge reads and work to one sample clock so overruns are deterministic.

    Admission uses `_LiveInput`'s own notice and clamp arithmetic. A reactive
    peer can answer the actual carrier position and shift rather than an aim.
    """

    # Measured on this machine and hard-coded in `_LiveInput`: 1152 samples from
    # a sample's ADC to its DAC, at a 128-frame block, and three blocks of notice
    # before the callback that fills the buffer after next.
    LAT = 1152

    # The turnaround the schedule is designed for, and it is measured: 105 ms is
    # what `D_NOMINAL_S` reads off a W4DNA -> KE5YTA exchange between two
    # commercial modems. It stood at 55 here, which is `TR_SWITCH_S` exactly --
    # a peer answering at the precise instant this rig stops being deaf, so
    # every tolerance derived from the gap collapsed to nothing and the bench
    # was measuring the edge of audibility rather than a working link.
    D_S = 0.105

    def __init__(self, *, seconds: float = 200.0, blk: int = 128,
                 holdback: int = 608, d: float = D_S, peer=None,
                 lat_in: int = 0):
        self.fs = FS
        self.audio = np.zeros(int(seconds * FS), np.float32)
        # A session that never keys again is the failure this file exists to
        # catch and it does not announce itself: the loop keeps handing slots
        # back, `host.tick()` is never reached, so the ARQ never advances and
        # `--max-cycles` never decrements. Nothing but a clock cap turns that
        # into a result. The scenes here run to ~95 s of bench clock; a
        # non-converging re-grid burns a slot an iteration, so the headroom is
        # about eighty iterations of one.
        self.limit = self.audio.size
        self.now = 0            # the converter's position
        self.pos = 0            # the next sample a reader will get
        self.floor = 0
        # The counters a capture source carries, which the session loop reads
        # straight into every sidecar. `lost` is the blocks a starved
        # interpreter never accepted; a bench is never starved, and the loop
        # still asks.
        self.xruns, self.xrun_at, self.underruns, self.lost = 0, None, 0, 0
        self.holdback, self._blk, self._lat = holdback, blk, self.LAT
        self.tx_latency_n = 0     # a bench station has measured no residue
        self.lat_in = lat_in
        self.keyed_s = 0.0
        self.keyed_at: int | None = None
        self.d_n = round(d * FS)
        self.peer = peer
        self.emissions: list[tuple[int, int]] = []
        self.signal_ends: list[int] = []
        self.leads: list[float] = []      # PTT lead each burst actually got
        self.taken: list[tuple[int, int]] = []      # spans handed to the reader
        self.drained: list[tuple[int, int, int, int]] = []
        # (read start, read end, charge instant, charged samples), retained so
        # the final ID's all-at-once drain is accounted separately from CS work.
        self.charge = 0         # samples of clock the cycle's decode overruns by
        self.flush_n = 0        # ...and what one flush costs
        self.scan_n = 0         # ...and one frame scan
        self.cs_n = 0           # ...and the anchored control-signal read
        self.upgrade_n = 0      # ...and the blind scan for a protocol we are not in
        # ...and the same costs charged PER SAMPLE of the audio handed to them,
        # which is the only way a window's length can pay for itself. Zero unless
        # a scene asks for it, so the fixed-cost scenes above measure what they
        # always measured.
        self.feed_rate = 0.0
        self.scan_rate = 0.0
        self.upgrade_rate = 0.0
        self.drain_rate = 0.0
        # Where the clock stood for every charge since the last carrier, so what
        # a cycle spends INSIDE THE KEYING SETTLE is a measurement rather than an
        # inference. That window is the one this file exists about: work in it is
        # charged to the rig's PTT lead, and past `key_notice` it costs the slot.
        self.spent: list[tuple[int, int]] = []
        self.prekey: list[int] = []
        self.armed = False      # the bulk is one cycle's, not one decode's
        self.answered = False   # ...and so is the peer's transmission
        self.always_placeable = False

    # -- the session's view of the codec ----------------------------------
    @property
    def samples(self) -> int:
        """What the CALLBACK HAS HANDED OVER -- not where the converter stands.

        THE TWO NUMBERS THE ADMISSION GUARD IS THE DIFFERENCE OF. `_LiveInput`
        hands Python one `_blk`-frame block at a time, so the delivered count is
        always a multiple of the block and always lags `sample_now()`; the guard
        (`clamp_late`) is taken on this one while the cycle's last read is
        scheduled on the other. This returned `self.now` -- one continuous
        counter standing in for both -- so the quantisation at the heart of the
        2026-09-11 entry-slot losses did not exist here, and no test in this
        repository could fail the way that arm did. 60000 samples a slot is
        468.75 blocks, so the room a cycle has before the guard trips takes four
        values and returns; a budget sitting on the threshold then loses one slot
        in four, deterministically, which is what
        `test_granted_entry_slots` reproduces.
        """
        return max(0, (self.now - self.lat_in) // self._blk * self._blk)

    @property
    def key_notice(self) -> int:
        return _LIVE.key_notice.fget(self)

    def _advance(self, to: int) -> None:
        self.now = max(self.now, to)
        if self.now > self.limit:
            raise RuntimeError(
                f"the session ran past {self.limit / FS:.0f} s of bench clock "
                f"without finishing: {len(self.emissions)} carriers on the air. "
                f"A loop that hands slots back forever occupies nothing and "
                f"reports nothing wrong.")

    def spend(self, n: int) -> None:
        """Charge `n` samples of clock to work that consumed no audio."""
        self.spent.append((self.now, n))
        self._advance(self.now + n)

    def _grab(self, a: int, b: int) -> np.ndarray:
        seg = np.zeros(max(0, b - a), np.float32)
        lo, hi = min(max(a, 0), self.audio.size), min(max(b, 0), self.audio.size)
        if hi > lo:
            seg[:hi - lo] = self.audio[lo:hi]
        return seg

    def _hand_over(self, a: int, b: int) -> np.ndarray:
        self.pos = b
        if b > a:
            self.taken.append((a, b))
        return self._grab(a, b)

    def read(self, count: int) -> np.ndarray:
        a = max(self.pos, self.floor)
        b = a + max(0, int(count))
        # Live audio cannot be read early: not before it is delivered, an
        # input latency after the converter took it.
        self._advance(b + self.lat_in)
        got = self._hand_over(a, b)
        # A read is not free of the backlog it drains -- see DRAIN_RATE. Charged
        # here rather than at the decode because `RadioTx._tx` runs one of these
        # after the bridge sleep, inside the keying settle, and decodes nothing.
        if self.drain_rate and got.size:
            cost = round(got.size * self.drain_rate)
            self.drained.append((a, b, self.now, cost))
            self.spend(cost)
        return got

    def take(self, seconds: float) -> np.ndarray:
        return self.read(int(seconds * self.fs))

    def take_until(self, index: int) -> np.ndarray:
        return self.read(max(0, index - self.pos))

    def read_ready(self) -> np.ndarray:
        return self.read(max(0, self.samples - max(self.pos, self.floor)))

    def sample_now(self) -> float:
        return float(self.now)

    def wait_until(self, index: int) -> float:
        slept = max(0, index - self.now)
        self._advance(index)
        return slept / self.fs * 1e3

    def clamp_late(self, at: int) -> int:
        # `always_placeable` is the assumption the loop used to run on -- that a
        # boundary aimed at is a boundary reachable -- and it is set only by the
        # negative controls below, to put the session back the way it flew.
        if self.always_placeable:
            return 0
        return _LIVE.clamp_late(self, at)

    def flush_to(self, index: int) -> None:
        self.floor = self.pos = index

    def flush(self, before: float | None = None) -> None:
        self.floor = self.pos = self.now

    # -- an answer armed ahead of its decode (`_LiveInput.arm`) --------------
    # Its audio is in the stream on its instant and its PTT is the keyer's, so
    # a commit that comes late keys the armed audio on time; one in time,
    # while the DAC's notice allows, replaces it.
    _armed = None
    missed = 0

    @property
    def armed_at(self):
        return None if self._armed is None else self._armed[0]

    def arm(self, audio, *, at, settle, key=None, max_key: float = 40.0) -> None:
        earliest = self.samples + 3 * self._blk
        if at - self.LAT < earliest:
            raise onair._MissedTxSlot(earliest - (at - self.LAT))
        self._armed = (at, np.asarray(audio, np.float32), settle, key)

    def disarm(self) -> bool:
        if self._armed is None:
            return True
        at, _, settle, _ = self._armed
        if self.now >= at - round(settle * self.fs):
            return False                 # its PTT is up: it goes out as armed
        self._armed = None
        return True

    def abort(self) -> None:
        self._armed = None

    def transmit(self, audio, *, at=None, settle: float = 0.0, key=None,
                 max_key: float = 40.0, admit=None) -> tuple[int, int]:
        if admit is not None:
            admit()
        earliest = self.samples + 3 * self._blk
        armed, self._armed = self._armed, None
        if armed is not None and (at is None or abs(at - armed[0]) > FS // 4):
            armed = None                 # given way to a burst nothing committed it to
        if armed is not None:
            armed_at, armed_audio, settle, key = armed
            late = earliest - (min(at, armed_at) - self.LAT)
            self.missed = max(late, 0)
            if late > 0:
                audio, at = armed_audio, armed_at
            # On its instant, its PTT the keyer's a whole settle ahead.
            first, lead = at, settle
        else:
            start = earliest if at is None else max(at - self.LAT, earliest)
            first = start + self.LAT
            # PTT cannot be asserted in the past either, so an aim point accepted
            # with less notice than the rig's settle comes out as a SHORT SETTLE
            # rather than as a late burst. That is silent on the air -- the carrier is
            # where it was asked for and the log says so -- and it is what a slot
            # search blind to the settle buys: on the x6100, 32 ms of 400.
            lead = min(settle, max(0.0, (first - self.now) / self.fs))
        self.leads.append(lead)
        # What this burst spent between the bridge and its own key. The bridge
        # runs to `onair._prekey_lead` in front of the boundary -- the PTT instant
        # plus the measured admission reserve -- so a charge recorded at or after
        # that instant is work done inside the window the whole cycle is arranged
        # to keep empty, and the window the admission guard adjudicates.
        # THE OVERLAP, not the charges that START inside it. A decode that begins
        # two milliseconds before the key instant and runs for twenty is twenty
        # milliseconds of work in front of a keyed transmitter, and counting it as
        # none is a gate that cannot see the thing it was written for.
        keyed_at = (None if at is None else
                    at - onair._prekey_lead(self, round(settle * self.fs)))
        self.prekey.append(
            0 if keyed_at is None else
            sum(min(pos + n, first) - max(pos, keyed_at) for pos, n in self.spent
                if pos + n > keyed_at))
        del self.spent[:]
        self._advance(first + audio.size)
        self.keyed_s = lead + audio.size / self.fs
        # Where the carrier came up, which is the audio less the lead the rig
        # actually got rather than the one it was promised.
        self.keyed_at = first - round(lead * self.fs)
        self.emissions.append((first, first + audio.size))
        if key is not None:
            key(True)
            key(False)
        # The peer runs its own clock: a P1 packet's held key is not a later
        # drop for it to answer.
        self.signal_ends.append(first + np.trim_zeros(audio, "b").size)
        if self.peer is not None:
            self.peer(self, self.signal_ends[-1])
        return first, first + audio.size

    def clock_report(self) -> str:
        return f"bench clock: {self.now} samples, no wall time involved"

    def stream_report(self) -> str:
        return "session stream: not taken -- this session had no sound card"

    def close(self) -> None:
        pass


class _Rig:
    """A rig that keys nothing. No serial device is opened by this file."""

    def __init__(self, *a, **kw):
        self.edges: list[bool] = []
        self.dial: int | None = None
        self.mode: str | None = None
        self.passband: int | None = None
        self.closed = False

    def set_freq(self, hz):
        self.dial = int(hz)
        return True

    def get_freq(self):
        # Admission needs a known dial even when no device is attached.
        return self.dial

    def set_mode(self, mode, *, passband=3000):
        self.mode = mode
        self.passband = int(passband)
        return True

    def get_mode(self):
        from hfmodem.core.rxreadiness import RadioMode
        return RadioMode(self.mode, self.passband,
                         f"{self.mode}\n{self.passband}\n")

    def _close(self):
        self.closed = True

    def arm(self):
        pass

    def ptt(self, on):
        self.edges.append(bool(on))
        return True

    def key_failure(self):
        # A rig that keys nothing has nothing to doubt: this file measures
        # where carriers land, not whether a keying channel stood.
        return None

    def stop(self):
        pass


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

    def __init__(self, in_lat: float = IN_LAT_S, out_lat: float = OUT_LAT_S,
                 *, audio=None):
        live = object.__new__(onair._LiveInput)
        live._q = queue.Queue()
        live.fs, live.samples, live.pos = FS, 0, 0
        live.xruns, live.xrun_at, live.underruns = 0, None, 0
        live.lost, live.air_lost = 0, 0
        live.timestamp_limitation = None
        live._fit, live._fits = [], []
        live._rec = None
        live._rest = np.zeros(0, np.float32)
        live._floor = 0
        live._duplex, live._blk = True, BLK
        live._lat = round((in_lat + out_lat) * FS)
        live.tx_latency_n = 0     # the whole loop is modelled here already
        live.geometry_n = 0
        live.holdback = BLK + round(in_lat * FS)
        live._tx, live._tx_at = None, 0
        live._tx_done = threading.Event()
        live._tx_end_time, live.keyed_s = 0.0, 0.0
        live._last = live._dac = (time.monotonic(), 0)
        self.live, self._in, self._out = live, in_lat, out_lat
        self.audio = audio
        self.output = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        live, t0, k = self.live, time.monotonic(), 0
        while not self._stop.is_set():
            due = t0 + k * BLK / FS
            while (d := due - time.monotonic()) > 0:
                time.sleep(min(d, 5e-4))
            n0 = live.samples
            # Callback delivery jitter must not move the converter's timeline.
            dac, adc = due + self._out, due - self._in
            buf = live._tx
            if buf is not None:
                a = max(n0, live._tx_at)
                b = min(n0 + BLK, live._tx_at + buf.size)
                if b > a:
                    self.output.append((a, buf[a - live._tx_at:b - live._tx_at].copy()))
                    if live.output_enqueue_sample is None:
                        live.output_enqueue_sample = a
                        live.driver_offset_samples = live._lat
                    live.output_end_sample = b
                if n0 + BLK >= live._tx_at + buf.size:
                    live._tx = None
                    live._tx_end_time = dac + (live._tx_at + buf.size - n0) / FS
                    live._tx_done.set()
            block = np.zeros(BLK, np.float32)
            if self.audio is not None:
                source = self.audio[n0:n0 + BLK]
                block[:source.size] = source
            live._q.put((adc, n0, block))
            live.samples = n0 + BLK
            live._last, live._dac = (adc, n0), (dac, n0)
            if not live._fit or adc - live._fit[-1][0] >= 1.0:
                live._fit.append((adc, n0))
            k += 1

    def close(self) -> None:
        self.live.abort()
        self._stop.set()
        self._thread.join()


class _CallbackBench(_Bench):
    # PortAudio timestamps the block's first sample but delivers its end.
    # _Bench.lat_in instead delays delivery until after the block's end.
    @property
    def samples(self):
        return max(0, (self.now - self.lat_in) // self._blk + 1) * self._blk

    def read(self, count):
        a = max(self.pos, self.floor)
        b = a + max(0, int(count))
        if b > a:
            self._advance((b - 1) // self._blk * self._blk + self.lat_in)
        return self._hand_over(a, b)
