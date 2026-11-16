# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Device-free production loop: September 10 fallback, ACK, and continuation.

The first field is original WS8EOC PCM. The subsequent peer is a small scripted
stop-and-wait reference, not another PactorArq: it advances only after an
emitted control signal has the expected counter and occupies its independently
fixed answer slot. This proves simulated continuation, not a new live exchange.
"""
import json
import contextlib
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import onair, p3acquire, placement, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike import test_late_entry_loop as loop
from hfmodem.tests.shrike.p3_reply_support import harness


FS = 48000
CYCLE = 60000
FIXTURES = Path(__file__).with_name("fixtures") / "ws8eoc-0910"
METADATA = FIXTURES / "metadata.json"
ADDITIONAL = b" OK"
# A 600 ms P3 role rotation, on the caller's unchanged 1250 ms comb.
# The measured morning peer onset sits approximately 960 ms past that comb;
# the resulting answer is `P3_COMB_REPLY_S` after the peer phase reference.
PEER_OFFSET = 46080
ANSWER_OFFSET = round(onair.P3_COMB_REPLY_S * FS)


def _recorded_changeover():
    if not METADATA.exists():
        pytest.skip(f"WS8EOC morning recordings absent: {METADATA}")
    metadata = json.loads(METADATA.read_text())
    row = next(r for r in metadata["fixtures"] if r["file"] == "first-rms.wav")
    samples = recorded_pcm({"file": "ws8eoc-0910/" + row["file"],
                            "sha256": row["pcm_sha256"]})
    return samples, row["onset_relative"]


class MorningPeer(loop.LatePeer):
    def __init__(self):
        super().__init__(entry_number=8, lose_first=False)
        self.fell_back = False
        self.pending_entry = None
        self.fallback_packet = None
        self.accepted = []
        self.rejected = []
        self.unsafe = []
        self.counter = 0
        self.ignored_acks = 0
        self.rms, self.rms_head = _recorded_changeover()
        self.data = placement.link_packet(1, ADDITIONAL, 1, swapped=False)
        self.data = np.real(hilbert(self.data) * np.exp(
            2j * np.pi * 50 * np.arange(len(self.data)) / FS))
        # Leave normal SL1 untrimmed and account for its pulse-filter delay.
        # This positions a stimulus; the independent answer slot stays fixed.
        self.data_head = (len(placement.protocol_config().pulse()) - 1) // 2

    def start_p3(self, entry_at):
        # Keep granting until the entry budget really falls back. The test must
        # traverse P1 keying too: a role-only test would miss stale grid geometry.
        self.pending_entry = entry_at

    def _put(self, phase, samples, head):
        at = phase - head
        self.audio[at:at + len(samples)] = samples
        self.responses.append((at, at + len(samples)))

    def _repeat(self, first_phase, samples, head):
        self.audio[first_phase - head:] = 0
        for cycle in range(24):
            self._put(first_phase + cycle * CYCLE, samples, head)

    def transmit(self, samples, **kwargs):
        first, end = super().transmit(samples, **kwargs)
        if (self.fell_back and self.p3_phase is None
                and self.pending_entry is not None and len(samples) >= 45000):
            self.fallback_packet = (first, end)
            # Peer phase is owned by its clock, not by subsequent ACK emissions.
            phase = self.pending_entry + PEER_OFFSET
            while phase - self.rms_head <= end:
                phase += CYCLE
            self.p3_phase = phase
            self._repeat(phase, self.rms, self.rms_head)
            # Explicit bench overrun: the real arm also lost a slot and then
            # collected a whole changeover. Advancing the converter leaves the
            # production slot recovery and contiguous-window collection intact.
            self.spend(round(1.50 * FS))
        elif self.p3_phase is not None and len(samples) < FS // 2:
            _, _, control, errors = self.controls[-1]
            phase = (first - self.p3_phase) % CYCLE
            # Independent SCS CS6 crop: first pulse center is 14.1 ms
            # inside the trimmed staggered waveform (within 0.1 ms for all CS).
            pulse_phase = phase + round(.0141 * FS)
            key_up = self.keyed_at
            previous = first - phase
            # The peer uses nominal 810 ms frame extent and allows filter tails;
            # accept the fixed `P3_COMB_REPLY_S` PULSE window and at least 20 ms
            # clear between the 810 ms data body and our PTT assertion.
            placed = (abs(pulse_phase - ANSWER_OFFSET) <= round(.005 * FS)
                      and key_up >= previous + round(.83 * FS)
                      and end <= previous + CYCLE - round(.02 * FS))
            correct = errors == 0 and control == self.counter % 2 and placed
            record = (first, end, control, errors, phase, key_up)
            if not placed:
                self.unsafe.append(record)
            if not correct:
                self.rejected.append(record)
            elif self.counter == 0 and self.ignored_acks < 2:
                self.ignored_acks += 1
            else:
                self.accepted.append(record)
                if self.counter == 0:
                    self.counter = 1
                    self._repeat(previous + CYCLE, self.data, self.data_head)
        return first, end


def _run_morning(monkeypatch, tmp_path, *, recovery=True, scale=None, measured_costs=None):
    peer = MorningPeer()
    # The one-cycle cap must remain recoverable even when a wider backlog
    # would retain the lost 0x59A anchor and let the coarse list succeed.
    monkeypatch.setattr(onair, "CARRY_MAX_CYCLES", 1)
    if not recovery:
        monkeypatch.setattr(onair._SessionRx, "_p3_recovery_active", lambda rx: False)
    reads = []
    keyed = []
    changeover = onair._SessionRx._p3_changeover_packet
    transmit = peer.transmit

    def on_grid(samples, **kwargs):
        result = transmit(samples, **kwargs)
        keyed.append((result[0], kwargs.get("at")))
        return result

    monkeypatch.setattr(peer, "transmit", on_grid)

    def observe(rx, audio, *args, **kwargs):
        origin = rx._scan_origin
        row0 = rx._p3_row0
        got = changeover(rx, audio, *args, **kwargs)
        reads.append(dict(origin=origin, length=len(audio), row0=row0,
                          recovery=rx._p3_recovery,
                          clock=peer.sample_now(), keyed=len(keyed),
                          start=None if got is None else origin + got[0].start))
        return got

    monkeypatch.setattr(onair._SessionRx, "_p3_changeover_packet", observe)
    fallback = onair.PtcHost._fall_back_to_pactor1

    def note_fallback(host, *args, **kwargs):
        result = fallback(host, *args, **kwargs)
        if peer.pending_entry is not None:
            peer.fell_back = True
        return result

    monkeypatch.setattr(onair.PtcHost, "_fall_back_to_pactor1", note_fallback)
    monkeypatch.setattr(loop, "LatePeer", lambda *a, **kw: peer)
    main = onair.main

    def longer_listen():
        sys.argv[sys.argv.index("--hold") + 1] = "40"
        # This scripted peer accepts the staggered SCS control at pulse-center;
        # ask for that experiment now that the defaults are the other profile.
        sys.argv += ["--p3-control-waveform", "current",
                     "--p3-control-placement", "pulse-center"]
        return main()

    monkeypatch.setattr(onair, "main", longer_listen)
    with _charged_morning(monkeypatch, peer, scale, measured_costs) as work:
        got = loop.run_late(monkeypatch, tmp_path, 8, False)
    (tmp_path / "recovery-reads.json").write_text(json.dumps(reads, indent=2))
    (tmp_path / "peer-reference.json").write_text(json.dumps({
        "expected_answer_phase": ANSWER_OFFSET,
        "fallback_packet": peer.fallback_packet,
        "ignored_acks": peer.ignored_acks,
        "accepted": peer.accepted,
        "rejected_counter": peer.rejected,
        "unsafe": peer.unsafe,
        "final_counter": peer.counter,
    }, indent=2) + "\n")
    return got, peer, reads, keyed, work


def test_morning_fallback_recovers_through_emitted_ack_and_new_payload(
        monkeypatch, tmp_path):
    got, peer, reads, keyed, _ = _run_morning(monkeypatch, tmp_path)
    log = got["log"]
    assert "entry retry budget exhausted" in log
    assert peer.fallback_packet is not None
    assert peer.ignored_acks == 2, log
    assert peer.accepted, log
    assert any(row[2] == 1 for row in peer.accepted), log
    assert not peer.unsafe, peer.unsafe
    # A final ACK for the old counter can already be in flight when the peer
    # advances. It must not advance the scripted peer's counter a second time.
    assert all(row[2:4] == (0, 0) for row in peer.rejected), peer.rejected
    assert peer.counter == 1
    assert bytes(got["host"].channel(got["host"].ptchn).rx) == b"RMS" + ADDITIONAL
    assert got["host"].payload_protocols
    assert all(p is spec.Protocol.PACTOR3 for p in got["host"].payload_protocols)
    after = log.split("peer is transmitting PACTOR-3 -> link follows", 1)[1]
    assert "P1 CS" not in after
    assert "P3 ACK GUARD" not in after
    assert "transmit anchor +600 ms" in log
    assert "clean run -> ask the peer" not in after
    recovered = [r for r in reads if r["recovery"] and r["start"] is not None]
    assert recovered, reads
    first = recovered[0]
    assert first["row0"] is None
    assert first["origin"] == 1320352
    assert first["length"] == 119136
    anchor = peer.call_at + 21 * CYCLE + 46080 + round(peer.d * FS)
    assert anchor + round(.12 * FS) < first["origin"]
    assert first["origin"] <= 1367856
    assert 1367856 + round(.81 * FS) < first["origin"] + first["length"]
    assert abs(first["start"] - 1367856) <= 120
    next_key = next(row for row in keyed if row[0] == peer.controls[0][0])
    assert next_key[0] == next_key[1]


@contextlib.contextmanager
def _charged_morning(monkeypatch, peer, scale, measured_costs=None):
    work, stack = [], []
    if scale is None:
        yield work
        return
    from hfmodem.tests.shrike.p3_recovery_support import (_ScaledClock, _SchedulingClock)

    stages = ((p3acquire, "_search"), (p3acquire, "_inside_burst"),
              (p3acquire.rxfront, "_cs_event"), (onair._SessionRx, "_p2_packet"),
              (onair._SessionRx, "_p3_recovery_changeover"),
              (onair, "_read_codeword_at_bursts"), (onair, "_grid_answer"),
              (onair, "_forecast_next_key"), (onair._CycleEvidence, "record"),
              (onair._PeerSpectrum, "hear"))
    with monkeypatch.context() as scoped:
        scoped.setattr(onair, "time", _SchedulingClock(peer) if measured_costs is not None
                       else _ScaledClock(scale))
        for owner, name in stages:
            original = getattr(owner, name)

            def charged(*args, _original=original, _name=name, **kwargs):
                frame = [time.perf_counter(), 0.0, _name]
                stack.append(frame)
                try:
                    return _original(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - frame[0]
                    stack.pop()
                    exclusive = max(0., elapsed - frame[1])
                    if measured_costs is not None:
                        exclusive = measured_costs.seconds(_name, args, kwargs, exclusive)
                    cost = math.ceil(exclusive * scale * FS)
                    work.append(dict(stage=_name, at=peer.sample_now(), samples=cost,
                                     recovery_work=_name == "_p3_recovery_changeover" or any(
                                         parent[2] == "_p3_recovery_changeover" for parent in stack),
                                     exclusive_ms=exclusive * 1000))
                    peer.spend(cost)
                    if stack:
                        stack[-1][1] += time.perf_counter() - frame[0]

            scoped.setattr(owner, name, charged)
        fits = onair._SessionRx._p3_recovery_fits

        def admitted(receiver, reserve):
            result = fits(receiver, reserve)
            work.append(dict(stage="admission", at=peer.sample_now(), reserve_s=reserve,
                             admitted=result, slot=receiver.host.peer.slot, samples=0))
            return result

        scoped.setattr(onair._SessionRx, "_p3_recovery_fits", admitted)
        yield work


@pytest.mark.parametrize("scale", [
    0., pytest.param(1., marks=pytest.mark.wall_clock),
    pytest.param(2.5, marks=pytest.mark.wall_clock),
    pytest.param(15., marks=pytest.mark.wall_clock),
])
def test_morning_recovery_preserves_keyed_slots_at_scaled_cost(
        monkeypatch, tmp_path, scale):
    from hfmodem.tests.shrike.p3_recovery_support import _MeasuredCosts
    costs = _MeasuredCosts()
    results = []
    for recovery in (False, True):
        directory = tmp_path / ("recovery" if recovery else "baseline")
        directory.mkdir()
        with monkeypatch.context() as scoped:
            got, peer, reads, keyed, work = _run_morning(
                scoped, directory, recovery=recovery, scale=scale, measured_costs=costs)
        results.append(dict(reads=reads, keyed=keyed, work=work,
                            slots=got["host"].peer.slots_used))
    old, new = results
    (tmp_path / "scaled-morning.json").write_text(json.dumps(
        dict(scale=scale, baseline=old, recovery=new), indent=2))
    assert set(old["slots"]) <= set(new["slots"])
    delivered = next((row for row in new["reads"] if row["start"] is not None), None)
    count = len(new["keyed"]) if delivered is None else delivered["keyed"]
    assert new["keyed"][:count] == old["keyed"][:count]
    for actual, requested in new["keyed"][count:]:
        if requested is not None:
            assert actual == requested


def _recovery_seam(tmp_path):
    host, rx, tx, grid, bench = harness(tmp_path, slot=24)
    grid.keyed_slot = tx.slot
    bench.now = bench.pos = tx.key_instant(grid, tx.slot) + 4800
    return host, rx, tx, grid, bench


def test_postkey_recovery_has_the_next_listen_close(tmp_path, monkeypatch):
    host, rx, tx, grid, bench = _recovery_seam(tmp_path)
    assert host.protocol == spec.Protocol.PACTOR1
    assert not rx._p3_acquisition_fits(.035)
    assert rx._p3_recovery_fits(.035)
    key = tx.key_instant(grid, grid.next_slot(tx.slot))
    fixed = key - round(tx.settle * FS) - bench.holdback - round(onair.PREKEY_RESERVE_S * FS)
    close = fixed - 4800
    monkeypatch.setattr(onair, "_p3_window_close", lambda *a: close)
    assert rx._p3_recovery_fits(.035)
    bench.now = bench.pos = close
    assert not rx._p3_recovery_fits(.035)


def test_late_prekey_read_does_not_borrow_recovery_time(tmp_path):
    _, rx, tx, grid, bench = _recovery_seam(tmp_path)
    grid.keyed_slot = tx.slot - 1
    assert bench.clamp_late(tx.key_instant(grid, tx.slot)) > 0
    assert not rx._p3_recovery_fits(.035)
    grid.keyed_slot = tx.slot
    assert rx._p3_recovery_fits(.035)
    tx.aim(grid, grid.next_slot(tx.slot))
    bench.now = bench.pos = tx.key_instant(grid, tx.slot) + 4800
    assert not rx._p3_recovery_fits(.035)


@pytest.mark.parametrize("owner", ["probe", "turn", "p4-link", "ceiling", "p1", "p2"])
def test_recovery_excludes_other_protocol_owners(tmp_path, monkeypatch, owner):
    host, rx, tx, _, _ = _recovery_seam(tmp_path)
    if owner == "probe":
        tx.p4_probe = SimpleNamespace(requested=True)
    elif owner == "turn":
        rx.p4.expected = True
        monkeypatch.setattr(rx, "_p4_read", lambda *a, **k: None)
    elif owner == "p4-link":
        tx.p4_link_options = SimpleNamespace()
    elif owner == "ceiling":
        host.settings["MYLEVEL"] = "2"
    elif owner == "p1":
        host.stay_in_pactor1 = True
    else:
        host.pactor2_only = True
    calls = []
    monkeypatch.setattr(rx, "_p3_recovery_changeover", lambda *a: calls.append(a))
    rx._p3_recovery = True
    try:
        onair._scan_frame(rx, np.zeros(FS), 0, upgrade=True)
    finally:
        rx._p3_recovery = False
    assert not calls


@pytest.mark.parametrize("retry", [False, True])
def test_recovery_admits_each_body_attempt(monkeypatch, retry):
    audio, _ = _recorded_changeover()
    calls = []
    checks = []
    real = p3acquire.rxfront._cs_event

    def body(*args, **kwargs):
        calls.append(args)
        event = real(*args, **kwargs)
        return onair.replace(event, packet=None)

    def fits():
        checks.append(True)
        return retry and len(checks) == 1

    monkeypatch.setattr(p3acquire.rxfront, "_cs_event", body)
    got = p3acquire.changeover(audio, offsets=(-50.,), can_decode=fits)
    assert len(calls) == int(retry)
    assert len(checks) == 1 + int(retry)
    assert got is None or got.event.packet is None


@pytest.mark.parametrize("stop", ["first-chunk", "later-chunk", "refined-body"])
def test_recovery_exhaustion_keeps_the_next_key(tmp_path, monkeypatch, stop):
    _, rx, tx, grid, bench = _recovery_seam(tmp_path)
    next_slot = grid.next_slot(tx.slot)
    key = tx.key_instant(grid, next_slot)
    close = key - round(tx.settle * FS) - bench.holdback - round(onair.PREKEY_RESERVE_S * FS)
    tail_s = rx.P3_RECOVERY_P2_S + rx.P3_RECOVERY_TAIL_S + onair.P3_DECODE_RESERVE_S
    room_s = {"first-chunk": .005, "later-chunk": .045, "refined-body": .075}[stop]
    bench.now = bench.pos = close - bench.key_notice - round((tail_s + room_s) * FS)
    searches, bodies = [], []
    audio, _ = _recorded_changeover()
    offsets = tuple(range(30))
    if stop == "refined-body":
        offsets = (-50.,)
        search, body = p3acquire._search, p3acquire.rxfront._cs_event

        def charged_search(*args):
            got = search(*args)
            bench.spend(round(.004 * FS))
            searches.append(True)
            return got

        def charged_body(*args, **kwargs):
            got = body(*args, **kwargs)
            bench.spend(round(.040 * FS))
            bodies.append(True)
            return onair.replace(got, packet=None)

        monkeypatch.setattr(p3acquire, "_search", charged_search)
        monkeypatch.setattr(p3acquire.rxfront, "_cs_event", charged_body)
    else:
        def charged_chunk(*args, **kwargs):
            searches.append(tuple(kwargs["offsets"]))
            bench.spend(round(.040 * FS))
            return None

        monkeypatch.setattr(p3acquire, "changeover", charged_chunk)
    got = rx._p3_recovery_changeover(audio, offsets, .010, 10)
    assert got is None or got.event.packet is None
    assert len(searches) == (0 if stop == "first-chunk" else 1)
    assert len(bodies) == int(stop == "refined-body")
    bench.spend(round(tail_s * FS))
    tx.aim(grid, next_slot)
    tx.send_p1_cs(0)
    assert not tx.refused
    assert grid.keyed_slot == next_slot
    assert bench.emissions[-1][0] == key
