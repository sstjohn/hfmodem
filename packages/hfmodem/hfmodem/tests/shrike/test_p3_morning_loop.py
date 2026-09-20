# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Device-free production loop: September 10 fallback, ACK, and continuation.

The first field is original WS8EOC PCM. The subsequent peer is a small scripted
stop-and-wait reference, not another PactorArq: it advances only after an
emitted control signal has the expected counter and occupies its independently
fixed answer slot. This proves simulated continuation, not a new live exchange.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import hilbert

from hfmodem.shrike import onair, placement, spec
from hfmodem.tests.shrike.recorded_pcm import recorded_pcm
from hfmodem.tests.shrike import test_late_entry_loop as loop


FS = 48000
CYCLE = 60000
FIXTURES = Path(__file__).with_name("fixtures") / "ws8eoc-0910"
METADATA = FIXTURES / "metadata.json"
ADDITIONAL = b" OK"
# A 600 ms P3 role rotation, on the caller's unchanged 1250 ms comb.
# The measured morning peer onset sits approximately 960 ms past that comb;
# the resulting answer is 890 ms after the peer phase reference.
PEER_OFFSET = 46080
ANSWER_OFFSET = 42720


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
            # accept the fixed 890 ms PULSE window and at least 20 ms
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


def test_morning_fallback_recovers_through_emitted_ack_and_new_payload(
        monkeypatch, tmp_path):
    peer = MorningPeer()
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
    got = loop.run_late(monkeypatch, tmp_path, 8, False)
    log = got["log"]
    (tmp_path / "peer-reference.json").write_text(json.dumps({
        "expected_answer_phase": ANSWER_OFFSET,
        "fallback_packet": peer.fallback_packet,
        "ignored_acks": peer.ignored_acks,
        "accepted": peer.accepted,
        "rejected_counter": peer.rejected,
        "unsafe": peer.unsafe,
        "final_counter": peer.counter,
    }, indent=2) + "\n")
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
