# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from hfmodem.shrike import arq, onair, p3frame, placement, rxfront, spec
from hfmodem.shrike.p3trial import ReplyClock
from hfmodem.tests.shrike.onair_support import _Bench, _Rig, _Session


FS = onair.FS


class PCMClock:
    key_notice = 1536  # 32 ms, the morning duplex scheduling requirement.
    holdback = 0

    def __init__(self, audio):
        self.audio = audio
        self.pos = self.samples = self.end = 0

    def wait_until(self, until):
        self.end = max(self.pos, min(int(until), len(self.audio)))
        return 0

    def read_ready(self):
        x = self.audio[self.pos:self.end]
        self.pos = self.samples = self.end
        return x

    def take_until(self, until):
        self.wait_until(until)
        return self.read_ready()

    def clamp_late(self, at):
        return max(0, self.samples + self.key_notice - at)

    def sample_now(self):
        return self.samples


def _arm(tmp_path, audio, *, old_long=False, sl=1, expected_seq=1):
    s = _Session(role=arq.IRS)
    host = s.host
    host.arq._cycle_long = old_long
    host.arq._cycle_request = not old_long
    host.arq._cycle_command_emitted = True
    host.arq._expected_seq = expected_seq
    host.arq.speed_level = sl
    # The WS8EOC recording is SL1, where the shipping P3_LADDER carries no long
    # frame, so an IRS would decline the grant. pactor3.md:640 publishes a
    # 36-byte long SL1 payload; give the fixture that rung so the cycle command
    # exists to be tested.
    host.arq._ladder = replace(
        arq.P3_LADDER,
        payloads=((arq.P3_LADDER.payloads[0][0], 36),) + arq.P3_LADDER.payloads[1:])
    tx = onair.RadioTx(None, transmit=False, out_dev=None, outdir=tmp_path,
                      settle=.040)
    host.peer = tx
    tx.attach(host)
    tx.defer_p3_cs = True
    clock = PCMClock(np.pad(audio, (0, max(0, 4*FS-len(audio)))))
    tx.live = clock
    g = onair._MasterGrid(-14400, 60000, 8880, packet_n=46080,
                         cs_n=5760, d_max_n=6240)
    g.protocol = spec.Protocol.PACTOR3
    g.sending = False
    tx.aim(g, 1)
    s.rx._p3_row0 = 7200 - (180000 if old_long else 60000)
    s.rx._p3_cycle_n = 180000 if old_long else 60000
    path = (placement.LONG_PATHS if old_long else placement.SPEED_PATHS)[sl]
    s.rx._p3_span = rxfront._frame_span(path)
    g._p3_command_slot = 0
    g._p3_command_row0 = 7200
    return s, tx, g, clock


class _ClockDone:
    def __init__(self, clock):
        self.clock = clock
        self.settled = False

    def is_set(self):
        return self.settled

    def set(self):
        self.settled = True

    def wait(self, timeout=None):
        if not self.settled:
            self.clock._finish_burst()
        return self.settled


class NativeCycleClock(_Bench):
    armed_at = onair._LiveInput.armed_at
    _propose = onair._LiveInput._propose
    disarm = onair._LiveInput.disarm
    abort = onair._LiveInput.abort
    transmit = onair._LiveInput.transmit
    _close_burst = onair._LiveInput._close_burst
    _emission = onair._LiveInput._emission
    _dac_time = onair._LiveInput._dac_time

    def __init__(self, monkeypatch, *, latency=0, geometry=0, tx_latency=0, **kwargs):
        super().__init__(**kwargs)
        self._lat = latency
        self.geometry_n, self.tx_latency_n = geometry, tx_latency
        self._dac = (latency / FS, 0)
        self._duplex = True
        self._armed = self._tx = None
        self._tx_done = onair.threading.Event()
        self._tx_end_time = 0.0
        self.output_enqueue_sample = self.output_end_sample = None
        self._watchdog_at = None
        self.submitted = []
        self.edges = []
        monkeypatch.setattr(onair.time, 'monotonic', lambda: self.now / FS)
        monkeypatch.setattr(onair.threading, 'Thread',
                            lambda **kwargs: SimpleNamespace(start=lambda: None))

    def _queue(self, audio, **kwargs):
        burst = onair._LiveInput._queue(self, audio, **kwargs)
        burst.done = _ClockDone(self)
        self._advance(self.now)
        return burst

    def arm(self, audio, *, at, settle, key=None, max_key=40):
        self._queue(audio, at=at, settle=settle, key=key, max_key=max_key)

    def _advance(self, to):
        before = self.now
        super()._advance(to)
        burst = self._armed
        if burst is None or burst.done.is_set():
            return
        if burst.stop.is_set():
            self._tx = None
            burst.done.set()
            return
        first = round(self._dac_time(burst.start) * FS)
        ptt = first - round(burst.settle * FS)
        if self.now >= ptt and not burst.keying:
            self.audio[max(0, ptt - self.lat_in):first + burst.size - self.lat_in] = 0
            burst.keying = True
            burst.keyed_at = max(before, ptt)
            burst.up = burst.keyed_at / FS
            self.edges.append((burst.keyed_at, True))
            if burst.key:
                assert burst.key(True) is not False
        if self.now >= first and self.output_enqueue_sample is None:
            self.submitted.append((burst.at, self._tx.copy()))
            self.output_enqueue_sample = burst.start
        if self.now >= first + burst.size:
            self.output_end_sample = burst.start + burst.size
            self._tx_end_time = (first + burst.size) / FS
            self._tx = None
            burst.played = True
            burst.down = self._tx_end_time
            self.edges.append((first + burst.size, False))
            if burst.key:
                burst.key(False)
            self.emissions.append((burst.first - self.geometry_n,
                                   burst.first - self.geometry_n + burst.size))
            burst.done.set()

    def _finish_burst(self):
        burst = self._armed
        end = (self.now if burst.stop.is_set() else
               round(self._dac_time(burst.start + burst.size) * FS))
        self._advance(end)

    def _key(self, burst, *, ahead=False):
        self._finish_burst()


def scene(tmp_path, monkeypatch, *, old_long=False, allow_long=True):
    s = _Session(role=arq.IRS)
    a = s.host.arq
    a.cfg.speed_up = 'hold'
    a.cfg.long_cycle = allow_long
    a._cycle_long = old_long
    a.speed_level = 2
    g = onair._MasterGrid(600000, 60000, 8880, packet_n=38880,
                          cs_n=10080, d_max_n=6240)
    g.protocol, g.sending = spec.Protocol.PACTOR3, False
    g.regear(old_long)
    live = NativeCycleClock(monkeypatch, seconds=120, blk=1)
    tx = onair.RadioTx(_Rig(), transmit=True, settle=.04, outdir=tmp_path,
                       record_audio=False)
    tx.attach(s.host)
    s.host.peer = tx
    tx.live, tx.sessrx = live, s.rx
    tx.defer_p3_cs = True
    tx.reply_clock = g.reply_clock = ReplyClock(entry_phase=1, pulse_epoch=600852)
    tx.aim(g, 0)
    return s, tx, g, live


def packet(s, tx, g, live, *, end_slot, long, status, decision=None, payload=b'X',
           phase=None):
    if phase is None:
        phase = g.boundary(end_slot) - round(
            (onair.P3_COMB_LONG_REPLY_S if long else onair.P3_COMB_REPLY_S) * onair.FS)
    row = phase + p3frame.DATA_OFFSET * rxfront.SPS
    if decision is not None:
        live.spend(max(0, decision - live.now))
    ev = rxfront.Event(t=row / onair.FS, start=row, kind='packet',
                       text='interval fixture', protocol=spec.Protocol.PACTOR3,
                       packet=(2, status, payload, True), cycle_long=long,
                       carrier_swapped=False)
    s.rx._on(ev)


def emit(s, tx, g, live, slot=None):
    if slot is not None:
        tx.aim(g, slot)
        tx.arm_p3_reply()
    if tx._pending_p3_cs is None:
        s.host.arq._rx_this_cycle = False
        s.host.tick(cycle_ticks=g.ticks, elapsed_ticks=g.ticks)
    tx.emit_pending_cs()
    assert not tx.refused
    return tx.slot, tx._p3_emitted_cs


def grant(s, tx, g, live, old_long):
    packet(s, tx, g, live, end_slot=0, long=old_long,
           status=0 if old_long else 0x20, decision=g.boundary(0) - 10000)
    assert tx._pending_p3_cs == arq.CS_CYCLE_TOG
    result = emit(s, tx, g, live)
    assert s.host.arq.cycle_command_emitted
    return result


def put_packet(s, g, live, level, long, status, *, opportunity=0, header_only=False):
    row = g._p3_command_row0 + opportunity * g.slot_n
    wave = placement.link_packet(level, b'X', status, long_cycle=long)
    if header_only:
        wave[10000:] = 0
    offset = row - p3frame.DATA_OFFSET * rxfront.SPS - (
        placement.protocol_config().pulse().size - 1) // 2
    live.audio[offset:offset + wave.size] = wave
    s.rx.sync.packet_level = level
    s.rx._p3_clock_role = arq.IRS
    return row


class ElapsedPCM:
    """A buffered reader cannot rewind the converter clock after decoder work.

    Only this in-memory object implements transmission. Capture position and
    elapsed samples deliberately diverge during a stall, as they do in duplex.
    """
    holdback = 0

    def __init__(self, audio, notice):
        self.audio = audio
        self.key_notice = notice
        self.pos = self.samples = self.end = 0
        self.emissions = []

    def wait_until(self, until):
        self.end = max(self.pos, min(int(until), len(self.audio)))
        self.samples = max(self.samples, self.end)
        return 0

    def read_ready(self):
        self.end = max(self.pos, min(self.samples, len(self.audio)))
        audio = self.audio[self.pos:self.end]
        self.pos = self.end
        return audio

    def take_until(self, until):
        self.wait_until(until)
        # Unlike read_ready, this read stops at the requested sample even if
        # the decoder has allowed later capture to accumulate in the queue.
        audio = self.audio[self.pos:self.end]
        self.pos = self.end
        return audio

    def sample_now(self):
        return self.samples

    def clamp_late(self, at):
        return max(0, self.samples + self.key_notice - at)

    def transmit(self, audio, *, at, settle, key, max_key, admit=None):
        assert not self.clamp_late(at), "production guard allowed late RF"
        assert len(audio) / FS <= max_key
        keyed_at = max(self.samples, at - round(settle * FS))
        self.wait_until(keyed_at)
        if admit is not None:
            admit()
        self.keyed_at = keyed_at
        self.keyed_s = (at + len(audio) - self.keyed_at) / FS
        self.emissions.append((at, at + len(audio), self.keyed_at))
        self.samples = at + len(audio)
        return at, self.samples

    def flush_to(self, end):
        self.pos = self.end = end
        self.samples = max(self.samples, end)


def _duplex(tx, s, audio, notice):
    clock = ElapsedPCM(audio, notice)
    tx.live = clock
    tx.sessrx = s.rx
    tx.rig = SimpleNamespace(ptt=lambda state: None, key_failure=lambda: None)
    tx.transmit = True  # All output is intercepted by ElapsedPCM above.
    return clock


def _buffered_listen(live, n, host, sessrx, limit, **kwargs):
    return live.take_until(live.pos + max(0, n))


LONG_MS, SHORT_MS, ROOM_MS = 9.66, 37.58, 23.0


def _walks(tmp_path, monkeypatch):
    tape = np.zeros(18 * FS)
    s, tx, g, _ = _arm(tmp_path, tape)
    clock = _duplex(tx, s, tape, 1536)
    clock.holdback = 660  # the FT-891's, which the arm's room was measured on
    reads = []
    scan = onair._scan_frame

    def counted(sessrx, audio, origin, **kw):
        reads.append((clock.pos, origin, audio.size, kw.get("p3_row0")))
        ms = LONG_MS if kw.get("p3_row0") is not None else SHORT_MS
        clock.samples += round(ms * FS / 1e3)
        return scan(sessrx, audio, origin, **kw)

    monkeypatch.setattr(onair, "_scan_frame", counted)
    audio, origin = np.zeros(0), 0
    walks = []
    for start in (1, 4, 7, 10):
        tx.aim(g, start)
        until = onair._p3_frame_ready(
            s.rx, onair._p3_decode_deadline(clock, tx.key_instant(g, start), 1920))
        audio, origin, _ = onair._collect(clock, s.rx, audio, origin, until)
        del reads[:]
        end, audio, origin, _ = onair._p3_transition_window(
            clock, g, tx, s.host, s.rx, start, audio, origin, 1920)
        walks.append((start, end, list(reads)))
        tx.aim(g, end)
        tx._send_p3_control(arq.CS_REQUEST)
    return s, tx, g, walks
